"""The DEPTH and SPEED rows travel to the problem, not only to the sampler.

WHY THIS FILE EXISTS
--------------------
``DEPTH_BOUNDS = (0.15, 1.0)`` m and ``V_BOUNDS = (8.0, 16.0)`` m/s were
declared on ``HydrofoilProblem``, ``HydrofoilWingletProblem`` and
``HydrofoilTailProblem`` as CLASS ATTRIBUTES WITH NO TYPE ANNOTATION. That
one missing annotation is the whole defect: an unannotated name is not a
dataclass field, so no builder keyword could set it, so the two rows that
decide the water families' OPERATING POINT were the only rows in their box a
user could not move.

The failure was silent in the worst available way. ``api._apply_overrides``
writes a widened row into a COPY of the box, and that copy is what the
sampler draws from — so widening ``V_ms`` to (5, 16) gave ``built.bounds``
[5, 16] while ``prob.bounds`` stayed [8, 16], and every draw below 8 m/s came
straight back from ``fg_hydrofoil_tail`` as ``"bounds violation"``. The run
completed, reported a budget spent and quietly searched the published band.
``RunConfig.pinned`` failed the other way and at least failed loudly:
"pinned value 5 for 'V_ms' is outside the box this run searches (8-16)".
This is exactly the bug ``tail.arm_row`` was written for, two rows along —
the seventh and eighth rows learning what six already knew.

WHY IT IS A BLOCKER RATHER THAN A TIDY-UP. The published speed band is
calibrated on a ~600 kg foiling dinghy, and every craft the water families
have since been aimed at flies outside it at both ends
(``LITERATURE_REVIEW_FOILINGBO.md``): take-off is 4.73 m/s for a race
windfoil and 3.75-4.13 m/s for a wingfoil, wingfoil cruise starts at 5 m/s,
and a kitefoil's design speed range is quoted as 5-18 m/s. With the row
frozen the engine could not evaluate a wingfoil AT ITS OWN CRUISE SPEED and
could not reach a kitefoil's top speed. A calibration is a default, not a
ban; this one had become the shape of the craft.

WHAT IS ASSERTED, and why each assertion can fail
-------------------------------------------------
* the defaults are inert — BOTH frozen anchors, bit-for-bit;
* a widened row reaches ``prob.bounds`` AND ``built.bounds``, they agree row
  for row, and ``api.rows_outside_validity`` is empty;
* a design at the widened speed/depth is genuinely FLOWN — checked by the
  closed form ``CL_target = W / (0.5 rho V^2 S)``, written out here from the
  numbers rather than read from the engine, so a vector that was clipped,
  clamped or refused cannot pass;
* the SAME point on the published box is still ``"bounds violation"``. That
  control is what makes the pair a test rather than a restatement: it is the
  before/after of the defect in one file;
* a pin is refused against the row the run ACTUALLY searches, both ways;
* ALL FOUR of ``_operating_row``'s refusals are exercised, each by name, and
  in the order the function asks them: NON-FINITE (inf or NaN at either end),
  ZERO-WIDTH (a pin written as a bound, pointing at ``RunConfig.pinned`` —
  this repo's recorded rule is that a pin is never a collapsed bound),
  NON-POSITIVE, INVERTED. The non-finite one is listed first because it was
  the one with no test at all: an infinite end read as an oversight rather
  than the decision it is, and NaN in particular reaches every comparison
  below it as False, so without that guard the user would be told their band
  "starts at or below zero" when what they handed over was not a number;
* the ORDER of the first two is itself asserted: ``(0.0, 0.0)`` is both
  collapsed and non-positive, and it is diagnosed as the PIN, because
  "raise the low end" is advice that leaves the row still collapsed and
  still wrong;
* a widened depth row reaches ``HydrofoilTailProblem.min_draught_m`` — the
  one consumer of ``DEPTH_BOUNDS[0]`` OUTSIDE the box, and the reason the
  row validators run first of everything in ``__post_init__``. Asserted as
  an outcome: a draught cap that only a widened floor can satisfy.

THE TWO FROZEN ANCHORS
----------------------
Restated from ``tests/test_craft_weight_is_a_flag.py`` (session S3), which
established them by measurement, and re-verified on this tree before any of
this session's edits. Both are on ``hydrofoil + elevator``, whose box carries
the speed and the depth as design variables:

* CENTRE, x = (lo + hi)/2 = (taper 0.6, twist_root 0 deg, twist_tip -2 deg,
  t/c 0.12, depth 0.575 m, V 12 m/s, S_t 0.04 m2, l_t 1.0 m)
  -> L/D = 24.67276683709319 (SM = -0.2276, x_np = 0.1221, mac = 0.1225,
  l_t = 1.0, x_cg = 0.15, i_t = -0.095 deg).

* OFF-CENTRE, x = lo + 0.75*(hi - lo) = (taper 0.8, twist_root 2 deg,
  twist_tip 0 deg, t/c 0.14, depth 0.7875 m, V 14 m/s, S_t 0.05 m2,
  l_t 1.25 m)
  -> L/D = 22.568044635929535 (SM = +0.0858, x_np = 0.1978, x_cg = 0.1875,
  i_t = +0.801 deg).

The off-centre point sits on the far side of the static-margin boundary from
the centre and has the stabiliser LIFTING rather than trimmed down, so a
change that quietly moved the balance shows there even if the centre is
blind to it. This project has twice found a null at a box centre that
reversed at a corner, which is why there are two.
"""

import dataclasses

import numpy as np
import pytest

from aerobo import api, hydrofoil, hydrotail
from aerobo.hydrofoil import RHO_WATER


#: the problem both anchors are measured on: its design vector carries
#: ``depth_m`` at index 4 and ``V_ms`` at index 5, so one build answers every
#: question in this file.
ANCHOR = "hydrofoil + elevator"

ANCHOR_CENTRE_LOD = 24.67276683709319
ANCHOR_OFF_CENTRE_LOD = 22.568044635929535

#: the PUBLISHED operating box, as the three water problem classes shipped
#: it. Restated here as the literal contract this session promised not to
#: move ("keep the defaults EXACTLY as they are"), so that a change to the
#: dataclass default fails here and not silently in a study.
PUBLISHED_DEPTH_BOUNDS = (0.15, 1.0)
PUBLISHED_V_BOUNDS = (8.0, 16.0)

#: A REAL SMALL CRAFT, from LITERATURE_REVIEW_FOILINGBO.md's windfoil entry:
#: 1030 N all-up on a 0.90 m / 0.090 m2 front wing. It matters that the
#: weight and the size are stated together — a 6000 N dinghy on this planform
#: is 5.8x overloaded and will not trim at 6 m/s at all, so the speed row and
#: the weight flag (session S3) only become useful in the same breath.
WINDFOIL = {"weight_n": 1030.0, "b_m": 0.90, "S_m2": 0.090}
WINDFOIL_W_N = 1030.0
WINDFOIL_S_M2 = 0.090

#: the envelope the published band cannot reach, in m/s: a wingfoil cruise
#: point and a kitefoil top-speed point.
WINGFOIL_CRUISE_MS = 5.5
KITEFOIL_TOP_MS = 18.0

#: THE CONTROL ROW for :func:`api.rows_outside_validity`, and the reason
#: this file can assert anything about that function at all.
#:
#: ``rows_outside_validity`` compares ``built.problem.bounds`` (the band the
#: family VALIDATES a candidate against) with ``built.bounds`` (the band the
#: sampler DRAWS from). Every place below that widens ``V_ms`` or
#: ``depth_m`` has just asserted those two arrays EQUAL — which is the whole
#: point of this session — so "the report is empty" there is arithmetically
#: implied and could not go red under any mutation. The honest claim is that
#: the widened row is ABSENT FROM A REPORT THAT IS NOT EMPTY, and that needs
#: a row in the same build whose widening genuinely does NOT travel.
#:
#: ``taper`` is that row. It is in all 156 water families' boxes with the
#: same published band, it travels to none of them, and it is the very
#: example ``rows_outside_validity``'s own docstring measures ("a ``taper``
#: row widened to 0.05-1.0 refuses every draw below 0.20").
STUCK_ROW = "taper"
STUCK_ROW_BAND = (0.05, 1.0)
PUBLISHED_STUCK_ROW_BAND = (0.2, 1.0)


#: THE PHYSICS THE TWO FROZEN ANCHORS WERE MEASURED ON. Session 68 changed
#: two of the elevator family's defaults — the craft became FLAT and its
#: strut became a placed, loaded surface (``hydrotail.strut_model``) — and
#: both move every L/D in the family. Re-pinned rather than re-measured,
#: because what the anchors are FOR is unchanged: this file asks whether the
#: SPEED ROW travels, and the speed row is not what moved. Pinned the same
#: way in ``test_rig_couple_is_a_moment_not_a_cg_move.py`` and
#: ``test_craft_weight_is_a_flag.py``.
PUBLISHED_PHYSICS = {"strut_model": False, "z_t_m": -0.05 * 1.2}


def _built(overrides=None, flags=None):
    return api.PROBLEM_SPECS[ANCHOR].build({}, dict(flags or {}), overrides)


def _row(built, label):
    return list(built.param_labels).index(label)


def _outside_labels(built) -> list:
    """The labels ``api.rows_outside_validity`` names, in its own order."""
    return [r["label"] for r in api.rows_outside_validity(built)]


def _cl_target(V, W=WINDFOIL_W_N, S=WINDFOIL_S_M2):
    """The trim target, written out here rather than read from the engine.

    ``CL_target = L / (q S)`` with ``q = 0.5 rho V^2``. This is the whole
    reason the assertion below is an outcome and not a restatement: it is
    computed from the speed the test PUT IN the design vector, so a run that
    clipped, clamped or refused that speed cannot produce it.
    """
    return W / (0.5 * RHO_WATER * V * V * S)


def _water_families():
    """Every registered water family, off the registry rather than a list.

    The water registry is generated (tip device x arm x stabiliser depth x
    how much of the elevator is designed, then every chosen-section and
    chord-law twin of all of those), so a written list would be stale the day
    another combination is registered — and a stale list is exactly what
    would stop catching the silent-drop defect this file is about.
    """
    return [n for n, s in api.PROBLEM_SPECS.items() if s.medium == "water"]


def _foil(prob):
    """The object that owns the operating box for a built water problem.

    The CST-section composites WRAP a foil problem
    (``HydrofoilSectionProblem.foil``) instead of subclassing it, so the two
    rows live one level down there and at the top level everywhere else.
    """
    return getattr(prob, "foil", prob)


# --------------------------------------------------------------- (a) anchors

def test_the_rows_are_dataclass_fields_on_all_three_families():
    """The defect in one assertion: an unannotated class attribute is not a
    field, and a name that is not a field cannot be a builder keyword."""
    for cls in (hydrofoil.HydrofoilProblem,
                hydrofoil.HydrofoilWingletProblem,
                hydrotail.HydrofoilTailProblem):
        names = {f.name for f in dataclasses.fields(cls)}
        assert "DEPTH_BOUNDS" in names, cls.__name__
        assert "V_BOUNDS" in names, cls.__name__
        # and the defaults did NOT move: this session widens what may be
        # ASKED FOR, it changes no published run
        assert cls.DEPTH_BOUNDS == PUBLISHED_DEPTH_BOUNDS, cls.__name__
        assert cls.V_BOUNDS == PUBLISHED_V_BOUNDS, cls.__name__


def test_defaults_reproduce_the_centre_anchor_bit_for_bit():
    built = _built(flags=PUBLISHED_PHYSICS)
    b = np.asarray(built.bounds, dtype=float)
    x = 0.5 * (b[:, 0] + b[:, 1])
    assert x[_row(built, "depth_m")] == 0.575
    assert x[_row(built, "V_ms")] == 12.0
    lod, _g = built.callable(x)
    assert lod == ANCHOR_CENTRE_LOD


def test_defaults_reproduce_the_off_centre_anchor_bit_for_bit():
    built = _built(flags=PUBLISHED_PHYSICS)
    b = np.asarray(built.bounds, dtype=float)
    x = b[:, 0] + 0.75 * (b[:, 1] - b[:, 0])
    assert x[_row(built, "depth_m")] == 0.7875
    assert x[_row(built, "V_ms")] == 14.0
    lod, g = built.callable(x)
    assert lod == ANCHOR_OFF_CENTRE_LOD
    # the discriminating half: the stabiliser is LIFTING here and the static
    # margin is on the stable side, the opposite of the centre
    assert g[1] > 0.0


def test_every_water_family_opens_on_the_published_operating_box():
    """Derived from the registry, so a newly generated combination joins the
    check automatically."""
    families = _water_families()
    assert families, "no water families registered — the registry moved"
    for name in families:
        prob = _foil(api.PROBLEM_SPECS[name].build({}, {}, None).problem)
        assert tuple(prob.DEPTH_BOUNDS) == PUBLISHED_DEPTH_BOUNDS, name
        assert tuple(prob.V_BOUNDS) == PUBLISHED_V_BOUNDS, name


# ------------------------------------------------------------- (b) the speed

def test_a_widened_speed_row_reaches_both_boxes_and_they_agree():
    band = (5.0, 16.0)
    built = _built({"V_ms": band})
    i = _row(built, "V_ms")

    own = np.asarray(built.problem.bounds, dtype=float)
    box = np.asarray(built.bounds, dtype=float)

    assert tuple(own[i]) == band          # the VALIDATOR's row followed
    assert tuple(box[i]) == band          # the SAMPLER's row followed
    assert own.tolist() == box.tolist()   # ...and nothing else moved

    # THE SAME CLAIM, MADE WHERE IT CAN FAIL. ``rows_outside_validity``
    # compares exactly ``own`` (``built.problem.bounds``) with ``box``
    # (``built.bounds``), so on THIS build — where the two have just been
    # asserted equal — an empty report is arithmetic rather than evidence.
    # So the widening is asked again beside ``taper``, a row that does not
    # travel (:data:`STUCK_ROW`), and the report must name taper ALONE: a
    # V_ms that stopped travelling would join the list.
    report = api.rows_outside_validity(
        _built({"V_ms": band, STUCK_ROW: STUCK_ROW_BAND}))
    assert [r["label"] for r in report] == [STUCK_ROW]
    assert report[0]["searched"] == list(STUCK_ROW_BAND)          # off ``box``
    assert report[0]["validated"] == list(PUBLISHED_STUCK_ROW_BAND)  # ``own``


def test_a_widened_speed_is_actually_flown_and_the_published_box_refuses_it():
    """The before/after of the defect, in one test.

    6.0 m/s is 2 m/s below the published floor. On the widened box the
    design is evaluated at that speed — proved by the trim target, which is
    computed here from 1030 N, 0.090 m2 and 6.0 m/s and must match the
    engine's exactly. On the published box the identical vector comes back
    "bounds violation", which is what every draw below 8 m/s used to do.
    """
    V = 6.0
    wide = _built({"V_ms": (5.0, 16.0)}, WINDFOIL)
    b = np.asarray(wide.bounds, dtype=float)
    x = 0.5 * (b[:, 0] + b[:, 1])
    x[_row(wide, "V_ms")] = V

    res = wide.evaluate(x)
    assert res["reason"] == ""
    assert res["V"] == V
    assert res["CL_target"] == _cl_target(V)
    assert np.isfinite(res["score"])
    assert res["score"] > 0.0             # a real L/D, not the penalty
    assert res["score"] != hydrofoil.PENALTY

    narrow = _built(None, WINDFOIL)
    assert narrow.evaluate(x)["reason"] == "bounds violation"
    assert narrow.evaluate(x)["score"] == hydrofoil.PENALTY


# ------------------------------------------------------------ (c) the pin

def test_a_speed_pin_is_refused_against_the_row_the_run_searches():
    """Both directions. A pin at 5 m/s is nonsense against the published box
    and correct against a box that was widened to contain it — and which one
    is true is a property of the RUN, not of the family."""
    cfg = dict(problem_name=ANCHOR, flags=dict(WINDFOIL), optimiser="random",
               budget=6, seed=0, pinned={"V_ms": 5.0})

    with pytest.raises(ValueError, match=r"outside the box this run searches"):
        api.run(api.RunConfig(**cfg))

    ok = api.run(api.RunConfig(bounds_overrides={"V_ms": (5.0, 16.0)}, **cfg))
    assert ok.pinned == {"V_ms": 5.0}
    assert np.isfinite(ok.best_score)
    i = list(ok.param_labels).index("V_ms")
    assert ok.bounds[i] == [5.0, 5.0]     # a pin collapses the REPORTED row
    assert list(ok.best_x)[i] == 5.0      # ...and the design flew at it


# ------------------------------------------------------------- (d) the depth

def test_a_widened_depth_row_reaches_both_boxes_and_they_agree():
    band = (0.05, 1.0)
    built = _built({"depth_m": band})
    i = _row(built, "depth_m")

    own = np.asarray(built.problem.bounds, dtype=float)
    box = np.asarray(built.bounds, dtype=float)

    assert tuple(own[i]) == band
    assert tuple(box[i]) == band
    assert own.tolist() == box.tolist()

    # independent of the line above, exactly as in (b): ``depth_m`` must be
    # ABSENT from a report that is not empty. ``own`` supplies ``validated``
    # and ``box`` supplies ``searched``, which is the pair of arrays this
    # whole session is about keeping equal for these two rows and only these
    # two.
    assert _outside_labels(
        _built({"depth_m": band, STUCK_ROW: STUCK_ROW_BAND})) == [STUCK_ROW]


def test_a_widened_depth_is_actually_flown_and_the_published_box_refuses_it():
    """0.08 m is half the published floor — the shallow, near-surface ride
    the free-surface image exists to describe."""
    h = 0.08
    wide = _built({"depth_m": (0.05, 1.0)}, WINDFOIL)
    b = np.asarray(wide.bounds, dtype=float)
    x = 0.5 * (b[:, 0] + b[:, 1])
    x[_row(wide, "depth_m")] = h

    deep_x = 0.5 * (b[:, 0] + b[:, 1])
    h_deep = float(deep_x[_row(wide, "depth_m")])

    res = wide.evaluate(x)
    assert res["reason"] == ""
    assert res["depth"] == h
    assert np.isfinite(res["score"])
    assert res["score"] > 0.0

    # ...and it was flown AS a shallower foil, not merely accepted as a
    # number. Two independent mechanisms, in opposite directions, both
    # keyed to the depth the test put in:
    deep = wide.evaluate(deep_x)
    #  (i) the mast is wetted over exactly its submerged length, so its
    #      parasite drag is LINEAR in depth. The ratio is the depth ratio,
    #      written out here from the two depths.
    assert res["cd0_mast"] / deep["cd0_mast"] == pytest.approx(h / h_deep,
                                                              rel=1e-12)
    #  (ii) the free-surface image costs span efficiency, and costs more of
    #       it the closer the foil is to the surface
    assert res["e"] < deep["e"]
    assert res["CDi"] > deep["CDi"]
    #  net: on this craft the wetted mast wins, which is a result the engine
    #  simply could not produce while 0.15 m was a wall
    assert res["score"] > deep["score"]

    narrow = _built(None, WINDFOIL)
    assert narrow.evaluate(x)["reason"] == "bounds violation"


def test_a_depth_pin_is_refused_against_the_row_the_run_searches():
    cfg = dict(problem_name=ANCHOR, flags=dict(WINDFOIL), optimiser="random",
               budget=6, seed=0, pinned={"depth_m": 0.08})

    with pytest.raises(ValueError, match=r"outside the box this run searches"):
        api.run(api.RunConfig(**cfg))

    ok = api.run(api.RunConfig(bounds_overrides={"depth_m": (0.05, 1.0)},
                               **cfg))
    assert ok.pinned == {"depth_m": 0.08}
    assert np.isfinite(ok.best_score)
    i = list(ok.param_labels).index("depth_m")
    assert list(ok.best_x)[i] == 0.08


# ------------------------------------------------------- (e) refused bands

@pytest.mark.parametrize("label,band,unit", [
    ("V_ms", (16.0, 8.0), "m/s"),
    ("depth_m", (1.0, 0.15), "m"),
])
def test_an_inverted_band_is_refused_and_names_the_row(label, band, unit):
    with pytest.raises(ValueError) as exc:
        _built({label: band})
    msg = str(exc.value)
    assert label in msg
    assert unit in msg
    assert "INVERTED" in msg


@pytest.mark.parametrize("label,band", [
    ("V_ms", (12.0, 12.0)),
    ("depth_m", (0.5, 0.5)),
])
def test_a_zero_width_band_is_refused_and_points_at_the_pin(label, band):
    """A zero-width row is a PIN written as a bound. The repo's recorded rule
    is that pinning is ``RunConfig.pinned`` and never a collapsed box, so the
    refusal has to say which mechanism to use, not merely that this one is
    wrong."""
    with pytest.raises(ValueError) as exc:
        _built({label: band})
    msg = str(exc.value)
    assert label in msg
    assert "zero width" in msg
    assert "RunConfig.pinned" in msg


@pytest.mark.parametrize("label,band", [
    ("V_ms", (0.0, 16.0)),
    ("V_ms", (-3.0, 16.0)),
    ("depth_m", (0.0, 1.0)),
    ("depth_m", (-0.1, 1.0)),
])
def test_a_non_positive_band_is_refused_and_says_it_is_not_a_cap(label, band):
    with pytest.raises(ValueError) as exc:
        _built({label: band})
    msg = str(exc.value)
    assert label in msg
    assert "at or below zero" in msg
    assert "No upper bound is imposed" in msg


def test_the_refusals_are_the_row_validators_not_a_bounds_check():
    """Called directly, so the message a script sees is the same one the
    build raises — one refusal in one place, ``tail.arm_row``'s contract."""
    assert hydrofoil.depth_row((0.05, 1.2)) == (0.05, 1.2)
    assert hydrofoil.speed_row([5, 18]) == (5.0, 18.0)     # list in, floats out
    with pytest.raises(ValueError, match="depth_m"):
        hydrofoil.depth_row((0.5, 0.5))
    with pytest.raises(ValueError, match="V_ms"):
        hydrofoil.speed_row((18.0, 5.0))
    with pytest.raises(ValueError, match=r"must be a pair"):
        hydrofoil.speed_row(12.0)


# --------------------------------------------------------- (f) reachability

@pytest.mark.parametrize("V", [WINGFOIL_CRUISE_MS, KITEFOIL_TOP_MS])
def test_the_craft_envelope_is_drawable_and_evaluable(V):
    """A wingfoil cruise point (5.5 m/s) and a kitefoil top-speed point
    (18 m/s), both outside the published (8, 16) band — the two the engine
    literally could not answer about before."""
    band = (5.0, 18.5)
    wide = _built({"V_ms": band}, WINDFOIL)
    i = _row(wide, "V_ms")
    b = np.asarray(wide.bounds, dtype=float)

    # DRAWABLE: inside the box the sampler draws from, inside the box the
    # problem validates against, and acceptable to the pin machinery
    assert b[i, 0] <= V <= b[i, 1]
    assert np.asarray(wide.problem.bounds, dtype=float)[i].tolist() \
        == list(band)
    assert api._pin_of(wide, {"V_ms": V}).fixed == {"V_ms": V}

    # EVALUABLE: a real number out of the solver at that speed, and the trim
    # target the closed form says it must be
    x = 0.5 * (b[:, 0] + b[:, 1])
    x[i] = V
    res = wide.evaluate(x)
    assert res["reason"] == ""
    assert res["CL_target"] == _cl_target(V)
    assert np.isfinite(res["score"]) and res["score"] > 0.0

    # and the published box refuses the same point, which is the "could not"
    assert _built(None, WINDFOIL).evaluate(x)["reason"] == "bounds violation"


def test_a_run_over_the_widened_envelope_searches_the_widened_envelope():
    """End to end: the box the run REPORTS is the box that was asked for, and
    the search inside it produced a real design."""
    band = (5.0, 18.5)
    res = api.run(api.RunConfig(
        problem_name=ANCHOR, flags=dict(WINDFOIL), optimiser="random",
        budget=8, seed=0, bounds_overrides={"V_ms": band}))
    i = list(res.param_labels).index("V_ms")
    assert res.bounds[i] == list(band)
    assert np.isfinite(res.best_score)
    assert res.best_score > 0.0


# --------------------------------------------------------- registry coverage

def test_every_water_family_carries_a_widened_row_into_its_own_box():
    """The rows travel on all of them, measured rather than asserted from a
    list of the four builders that happen to exist today."""
    band_v, band_h = (5.0, 18.5), (0.05, 1.2)
    families = _water_families()
    assert families, "no water families registered — the registry moved"
    for name in families:
        built = api.PROBLEM_SPECS[name].build(
            {}, {}, {"V_ms": band_v, "depth_m": band_h})
        foil = _foil(built.problem)
        assert tuple(foil.V_BOUNDS) == band_v, name
        assert tuple(foil.DEPTH_BOUNDS) == band_h, name

        own = np.asarray(built.problem.bounds, dtype=float)
        box = np.asarray(built.bounds, dtype=float)
        assert own.tolist() == box.tolist(), name

        # ...and, independently of that line (see (b)), the two rows are
        # ABSENT from a report that is not empty. ``taper`` is in all 156
        # water boxes with the same published band and travels to none of
        # them, so it is the control that keeps this from being arithmetic.
        stuck = api.PROBLEM_SPECS[name].build(
            {}, {}, {"V_ms": band_v, "depth_m": band_h,
                     STUCK_ROW: STUCK_ROW_BAND})
        assert _outside_labels(stuck) == [STUCK_ROW], name


# ------------------------------------------- (g) the rest of the refusals

@pytest.mark.parametrize("label,band", [
    ("V_ms", (8.0, float("inf"))),
    ("V_ms", (float("-inf"), 16.0)),
    ("V_ms", (8.0, float("nan"))),
    ("depth_m", (0.15, float("inf"))),
    ("depth_m", (float("nan"), 1.0)),
])
def test_a_non_finite_band_is_refused_as_a_decision_not_an_oversight(label,
                                                                     band):
    """The fourth refusal, which had no test and therefore read as an
    accident.

    ``inf`` is not an interval a Sobol sequence can be scaled into, and NaN
    is worse than useless: every comparison below the guard is False for it,
    so ``not (lo > 0.0)`` fires and the user is told their band "starts at
    or below zero" — a diagnosis of a number they did not write. Both ends
    are covered, and both kinds, because the guard is one ``all``-shaped
    expression and a half of it could rot silently.
    """
    with pytest.raises(ValueError) as exc:
        _built({label: band})
    msg = str(exc.value)
    assert label in msg
    assert "not finite" in msg
    # ...and NOT mis-diagnosed as one of the other three
    assert "at or below zero" not in msg
    assert "zero width" not in msg
    assert "INVERTED" not in msg


@pytest.mark.parametrize("label,band,unit", [
    ("V_ms", (0.0, 0.0), "m/s"),
    ("V_ms", (12.0, 12.0), "m/s"),
    ("depth_m", (0.0, 0.0), "m"),
    ("depth_m", (-0.5, -0.5), "m"),
])
def test_a_collapsed_row_is_a_PIN_wherever_it_collapsed(label, band, unit):
    """ORDER, asserted as an outcome.

    ``(0.0, 0.0)`` satisfies two refusals at once. The positivity message
    tells the user to raise the low end, which leaves the row collapsed and
    the run still broken; the pin message tells them the thing that is
    actually wrong — they wrote ``RunConfig.pinned``'s job as a bound. So
    the collapse is diagnosed first, at zero, below zero and at 12 m/s
    alike, and the two non-collapsed halves of those cases (tested above and
    below) keep their own messages.

    A test that only used ``(12.0, 12.0)`` would pass under either ordering,
    which is exactly why the zero and the negative rows are here.
    """
    with pytest.raises(ValueError) as exc:
        _built({label: band})
    msg = str(exc.value)
    assert label in msg and unit in msg
    assert "zero width" in msg
    assert "RunConfig.pinned" in msg
    # the mis-diagnosis this ordering exists to prevent
    assert "at or below zero" not in msg


def test_the_four_refusals_are_asked_in_one_stated_order():
    """The whole ladder in one place, through the public validators.

    Each band below trips exactly one rung and is worded for it. Written as
    a table rather than four tests so the ORDER is visible: a row that
    qualifies for two rungs appears twice, once here and once in the
    parametrised pin test above.
    """
    cases = [
        ((float("inf"), 16.0), "not finite"),
        ((12.0, 12.0), "zero width"),
        ((0.0, 0.0), "zero width"),          # collapsed beats non-positive
        ((0.0, 16.0), "at or below zero"),
        ((-3.0, 16.0), "at or below zero"),
        ((16.0, 8.0), "INVERTED"),
    ]
    for band, expected in cases:
        with pytest.raises(ValueError) as exc:
            hydrofoil.speed_row(band)
        assert expected in str(exc.value), (band, str(exc.value))
    # ...and a legitimate wide band is still a legitimate wide band
    assert hydrofoil.speed_row((3.5, 25.0)) == (3.5, 25.0)
    assert hydrofoil.depth_row((0.02, 3.0)) == (0.02, 3.0)


# ------------------------------------------ (h) the row OUTSIDE the box

#: The elevator family's unstated stabiliser drop and the published foil
#: span, written out so the draught arithmetic below is derived here and not
#: read from the engine. The stabiliser hangs that fraction of the span below
#: the foil, so the shallowest draught the assembly can reach is the
#: shallowest FOIL DEPTH plus it.
#:
#: THE FRACTION CHANGED (session 68): the craft is FLAT now — front wing,
#: strut and stabiliser bolted to one horizontal fuselage — so the default is
#: ``tail.DZ_GRID_FRAC`` = 0.01 b and not the air families' measurement
#: height of 0.05 b. The assembly therefore draws LESS, and the floor these
#: tests are about moved 0.21 m -> 0.162 m with it.
Z_T_FRAC = 0.01
FOIL_SPAN_M = 1.2
STAB_DROP_M = Z_T_FRAC * FOIL_SPAN_M           # 0.012 m


def test_the_widened_depth_floor_reaches_min_draught_and_the_cap_that_needs_it():
    """THE ROW'S ONE CONSUMER OUTSIDE THE BOX, asserted as an outcome.

    ``HydrofoilTailProblem.min_draught_m`` reads ``DEPTH_BOUNDS[0]`` and adds
    the stabiliser's drop, and ``hydrofoil._check_draught_cap`` refuses any
    cap below it AT CONFIG TIME. That is the reason the two row validators
    run "first of everything" in ``__post_init__``: the cap is checked
    against a floor the widened row has already moved.

    Nothing asserted this. A reviewer caught the sum hard-coded as
    ``0.15 + dz`` mid-flight and no test could see the difference, because
    every existing test uses the published row where 0.15 is the right
    answer for the wrong reason.

    So the assertion is the DECISION, not the attribute: a 0.15 m draught
    cap is arithmetically impossible on the published box (floor 0.162 m) and
    reachable on a box whose depth row starts at 0.05 m (floor 0.11 m). The
    published build must refuse it by name, the widened build must accept
    it, and a design must then FLY inside it — draught 0.14 m against the
    0.15 m cap, with the third margin positive and equal to the difference.
    """
    cap = 0.15
    shallow, published = 0.05, PUBLISHED_DEPTH_BOUNDS[0]
    # the two floors, written out here from the two depth rows
    floor_published = published + STAB_DROP_M        # 0.162 m
    floor_widened = shallow + STAB_DROP_M            # 0.11 m
    assert floor_widened < cap < floor_published     # the cap needs the row

    # the published box cannot reach it, and says so with the number
    with pytest.raises(ValueError) as exc:
        _built(None, {api.DRAUGHT_MAX_KEY: cap})
    msg = str(exc.value)
    assert "draught_max_m" in msg
    assert f"{floor_published:.4g}" in msg

    # the widened one can, and the floor it was checked against is the
    # widened floor — the closed form, not the published constant
    wide = _built({"depth_m": (shallow, 1.0)}, {api.DRAUGHT_MAX_KEY: cap})
    prob = _foil(wide.problem)
    assert tuple(prob.DEPTH_BOUNDS) == (shallow, 1.0)
    assert prob.min_draught_m == floor_widened
    assert prob.min_draught_m != published + STAB_DROP_M
    assert prob.draught_max_m == cap
    assert prob.n_constraints == 3                   # the cap is live

    # ...and it is not merely constructible: a design flies inside it
    b = np.asarray(wide.bounds, dtype=float)
    x = 0.5 * (b[:, 0] + b[:, 1])
    x[_row(wide, "depth_m")] = 0.08                  # below the published row
    res = wide.evaluate(x)
    assert res["reason"] == ""
    assert res["depth"] == 0.08
    assert res["draught_m"] == pytest.approx(0.08 + STAB_DROP_M, abs=1e-12)
    assert res["g"][2] == pytest.approx(cap - res["draught_m"], abs=1e-12)
    assert res["g"][2] > 0.0                         # the cap is satisfied


def test_the_draught_floor_follows_the_depth_row_across_the_whole_band():
    """Three depth rows, three floors, one closed form.

    A single widened row could be matched by a second hard-coded constant.
    This walks the floor and asserts ``min_draught_m`` tracks it exactly —
    and that the CAP refusal moves with it, one measurement either side of
    each floor.
    """
    for shallow in (0.02, 0.05, PUBLISHED_DEPTH_BOUNDS[0], 0.40):
        prob = _foil(_built({"depth_m": (shallow, 1.0)}).problem)
        floor = shallow + STAB_DROP_M
        assert prob.min_draught_m == pytest.approx(floor, abs=1e-15)

        # just above the floor: constructible. Just below: refused by name.
        ok = _built({"depth_m": (shallow, 1.0)},
                    {api.DRAUGHT_MAX_KEY: floor + 1e-3})
        assert _foil(ok.problem).draught_max_m == floor + 1e-3
        with pytest.raises(ValueError, match="draught_max_m"):
            _built({"depth_m": (shallow, 1.0)},
                   {api.DRAUGHT_MAX_KEY: floor - 1e-3})


def test_a_mapping_shaped_row_gets_the_sentence_written_for_it():
    """``{"lo": .., "hi": ..}`` is the shape a caller reaches for when they
    half-remember the interface.

    It indexes as ``band[0]`` and raises ``KeyError`` — not ``IndexError`` —
    so before ``KeyError`` joined the except clause the one sentence written
    for this mistake never reached the person making it, and they got a bare
    ``KeyError: 0`` from inside the validator instead.
    """
    from aerobo import hydrofoil

    for row in ({"lo": 0.1, "hi": 1.0}, {0: 0.1}):
        with pytest.raises(ValueError, match="must be a pair"):
            hydrofoil.depth_row(row)
        with pytest.raises(ValueError, match="must be a pair"):
            api.PROBLEM_SPECS["hydrofoil + elevator"].build(
                {}, {}, {"depth_m": row})
