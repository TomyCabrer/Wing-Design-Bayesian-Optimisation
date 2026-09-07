"""The plate's chord answers to two bands at once.

The design vector states the endplate's chord as a RATIO of the wing's tip
chord, which is the right way to keep a plate proportioned to the wing it
hangs on and useless as a box: the tip chord moves with the span, the taper
and the chord law, so the same ratio draws a different part every iteration.
A rule, a piece of bodywork or a mould is written in METRES.

So both bands are live. The ratio band is the design box's own
``endplate_chord_ratio`` row; the metre band is configuration
(``endplate_chord_min_m`` / ``endplate_chord_max_m``), and what flies is the
ratio's chord CLAMPED into it. What this file holds:

  1. UNSTATED IS UNCHANGED. No band = the published problem, bit-for-bit.
  2. THE CLAMP IS THE FLOWN GEOMETRY, not a read-out: the VLM's device panels
     carry the clamped chord, so the drag build-up, the junction charge and
     the beam are all charged on one part.
  3. THE METRE BAND WINS. Where the two bands disagree the length does, and
     the breakdown reports both the chord asked for and the chord flown — a
     clamped design must never read as the design that was requested.
  4. IT IS A CLAMP, NOT A REFUSAL. Every point of the design box stays
     flyable; the band costs the search no evaluations.
  5. THE SHELLS ASK IT. The card's two fields, their flags, and the number
     keys that stop a typed field rebuilding itself mid-number.
"""

import inspect
from dataclasses import replace

import numpy as np
import pytest

from aerobo import api
from aerobo import endplate as ep

#: a mid-box design asking for the WIDEST plate the family offers (ratio 3)
X = np.array([0.7, 0.0, -2.0, 6.0, 0.45, 0.55, 3.0, 0.12, 0.0, 1.6])

#: ...and one asking for the narrowest (ratio 0.5)
X_NARROW = X.copy()
X_NARROW[6] = 0.5


def _out(x=X, **kw):
    prob = replace(ep.CarWingEndplateProblem(), mount="tips", **kw)
    out = ep.evaluate_car_wing_endplate(np.asarray(x, dtype=float), prob)
    assert out["feasible"], out["reason"]
    return out


# ------------------------------------------------------ 1. unstated = unchanged

def test_no_band_is_the_published_problem_bit_for_bit():
    base = _out()
    both_none = _out(endplate_chord_min_m=None, endplate_chord_max_m=None)
    for k, v in base.items():
        if isinstance(v, float):
            assert both_none[k] == v, k
    assert base["endplate_chord_clamped"] is False
    assert base["endplate_chord_requested_m"] == base["endplate_chord_m"]
    assert base["endplate_chord_min_m"] is None
    assert base["endplate_chord_max_m"] is None


def test_a_band_the_design_already_obeys_changes_nothing():
    base = _out()
    c = base["endplate_chord_m"]
    wide = _out(endplate_chord_min_m=0.5 * c, endplate_chord_max_m=2.0 * c)
    assert wide["endplate_chord_clamped"] is False
    for k, v in base.items():
        if isinstance(v, float):
            assert wide[k] == v, k


# ------------------------------------------------- 2. the clamp is the geometry

def test_a_maximum_clips_the_chord_and_says_so():
    base = _out()
    c = base["endplate_chord_m"]
    lim = 0.5 * c
    out = _out(endplate_chord_max_m=lim)
    assert out["endplate_chord_m"] == pytest.approx(lim)
    assert out["endplate_chord_requested_m"] == pytest.approx(c)
    assert out["endplate_chord_clamped"] is True
    assert out["endplate_chord_max_m"] == pytest.approx(lim)


def test_a_minimum_lifts_a_narrow_plate_up_to_it():
    base = _out(X_NARROW)
    c = base["endplate_chord_m"]
    lim = 2.0 * c
    out = _out(X_NARROW, endplate_chord_min_m=lim)
    assert out["endplate_chord_m"] == pytest.approx(lim)
    assert out["endplate_chord_requested_m"] == pytest.approx(c)
    assert out["endplate_chord_clamped"] is True


def test_the_clamped_chord_is_what_the_solver_flies():
    """Not a relabelled read-out: the device's panels carry it."""
    lim = 0.25
    out = _out(endplate_chord_max_m=lim)
    res = out["vlm"]
    dev = res.is_winglet & (res.y > 0.0)
    assert dev.any()
    # no blend here, so every device panel is the plate's own chord
    assert np.allclose(res.c[dev], lim)


def test_every_reduced_order_model_sees_the_clamped_chord():
    lim = 0.25
    out = _out(endplate_chord_max_m=lim)
    for key in ("endplate_chord_mean_m", "endplate_chord_corner_m",
                "endplate_chord_root_m"):
        assert out[key] == pytest.approx(lim), key
    # the thickness follows the chord it belongs to (t/c is the variable)
    assert out["endplate_t_m"] == pytest.approx(out["endplate_tc"] * lim)
    # ...and the drag falls with the plate, rather than being charged on a
    # chord that is not flown anywhere
    assert out["CD_endplate"] < _out()["CD_endplate"]


def test_the_ratio_reported_is_the_ratio_flown():
    lim = 0.25
    out = _out(endplate_chord_max_m=lim)
    tip = out["endplate_chord_m"] / out["endplate_chord_ratio"]
    assert out["endplate_chord_ratio_requested"] == pytest.approx(X[6])
    assert out["endplate_chord_ratio"] < X[6]
    assert out["endplate_chord_ratio_at_tip"] == pytest.approx(
        out["endplate_chord_ratio"])
    # ...and with a sharp corner that ratio IS the step the surfaces join at
    assert out["endplate_chord_step"] == pytest.approx(
        out["endplate_chord_ratio"])
    assert out["endplate_chord_requested_m"] == pytest.approx(X[6] * tip)


# --------------------------------------------------- 3. the metre band wins

def test_the_metre_band_beats_the_ratio_row():
    """A minimum above what the ratio ceiling can draw still holds."""
    prob = ep.CarWingEndplateProblem(mount="tips")
    ceiling = prob.ENDPLATE_CHORD_RATIO_BOUNDS[1]
    widest = _out()["endplate_chord_m"]           # ratio row at its ceiling
    lim = 1.5 * widest
    out = _out(endplate_chord_min_m=lim)
    assert out["endplate_chord_m"] == pytest.approx(lim)
    assert out["endplate_chord_ratio"] > ceiling      # the box could not ask
    assert out["endplate_chord_clamped"] is True


def test_a_clamped_design_never_reads_as_the_one_requested():
    out = _out(endplate_chord_max_m=0.25)
    assert out["endplate_chord_m"] != out["endplate_chord_requested_m"]
    assert out["endplate_chord_clamped"] is True


# ------------------------------------------------ 4. a clamp, not a refusal

def test_the_band_refuses_nothing_in_the_box():
    prob = replace(ep.CarWingEndplateProblem(), mount="inboard",
                   endplate_chord_min_m=0.10, endplate_chord_max_m=0.20)
    b = prob.bounds
    rng = np.random.default_rng(0)
    xs = rng.uniform(b[:, 0], b[:, 1], size=(24, b.shape[0]))
    lo, hi = prob.chord_band_m
    for x in xs:
        out = ep.evaluate_car_wing_endplate(x, prob)
        if not out["feasible"]:
            # only the family's own gates may fire, never the chord band
            assert "chord" not in out["reason"], out["reason"]
            continue
        assert lo - 1e-12 <= out["endplate_chord_m"] <= hi + 1e-12


def test_the_band_is_flat_where_it_bites():
    """Two different ratios above the ceiling fly the SAME plate."""
    x_hi = X.copy()
    x_mid = X.copy()
    x_mid[6] = 2.0
    a = _out(x_hi, endplate_chord_max_m=0.20)
    b = _out(x_mid, endplate_chord_max_m=0.20)
    assert a["endplate_chord_m"] == pytest.approx(b["endplate_chord_m"])
    assert a["CZ"] == pytest.approx(b["CZ"])


# ------------------------------------------------------------ the contract

def test_the_band_validates_itself():
    with pytest.raises(ValueError, match="exceeds"):
        ep.CarWingEndplateProblem(endplate_chord_min_m=0.4,
                                  endplate_chord_max_m=0.3)
    for kw in ({"endplate_chord_min_m": 0.0}, {"endplate_chord_max_m": -1.0}):
        with pytest.raises(ValueError, match="must be > 0"):
            ep.CarWingEndplateProblem(**kw)


def test_the_band_helpers_read_as_written():
    prob = ep.CarWingEndplateProblem()
    assert prob.chord_band_m == (0.0, float("inf"))
    assert prob.clamp_chord_m(0.31) == 0.31
    prob = ep.CarWingEndplateProblem(endplate_chord_min_m=0.1,
                                     endplate_chord_max_m=0.3)
    assert prob.chord_band_m == (0.1, 0.3)
    assert prob.clamp_chord_m(0.05) == 0.1
    assert prob.clamp_chord_m(0.90) == 0.3
    assert prob.clamp_chord_m(0.20) == 0.2


def test_the_band_travels_as_flags():
    assert "endplate_chord_min_m" in api.CAR_ENDPLATE_KEYS
    assert "endplate_chord_max_m" in api.CAR_ENDPLATE_KEYS
    built = api.PROBLEM_SPECS["car rear wing + endplates"].build(
        {}, {"endplate_chord_min_m": 0.12, "endplate_chord_max_m": 0.34}, None)
    assert built.problem.endplate_chord_min_m == pytest.approx(0.12)
    assert built.problem.endplate_chord_max_m == pytest.approx(0.34)
    # ...and the dimension does not move: this is configuration, not a row
    plain = api.PROBLEM_SPECS["car rear wing + endplates"].build({}, {}, None)
    assert built.dim == plain.dim


# ------------------------------------------------------------- 5. the shells

def test_the_card_asks_for_both_ends_in_metres():
    """Both ends, in metres — and in the shell's own home for a BOUND.

    V1 and V2 have no design box, so they keep the two fields on the card
    (``_car_plate_chord_rows``, which ``_car_controls`` draws for them). V3
    draws them under its DESIGN BOX, beside the ratio row they clip — the
    reader was otherwise holding a band from one screen against a row on
    another — and passes ``plate_chord=False`` to keep the question in one
    place per shell.
    """
    from gui import nice_app as v1
    from gui.v3.stages import wing as v3wing

    src = inspect.getsource(v1._car_plate_chord_rows)
    assert "car_endplate_chord_min_m" in src
    assert "car_endplate_chord_max_m" in src
    assert "Plate chord ≥" in src and "Plate chord ≤" in src
    # ...and V1 still draws it, gated on the parameter V3 turns off
    ctl = inspect.getsource(v1._car_controls)
    assert "_car_plate_chord_rows(ch, set_choice)" in ctl
    assert "plate_chord: bool = True" in ctl

    v3src = inspect.getsource(v3wing)
    assert "car_endplate_chord_min_m" in v3src
    # typed into the BOX view now, so it must be on the list that stops that
    # view being rebuilt from its own field
    assert "car_endplate_chord_min_m" in v3wing.BOX_VALUE_KEYS
    assert "car_endplate_chord_max_m" in v3wing.BOX_VALUE_KEYS


def test_the_card_sends_them_only_where_the_plate_is_designed():
    from gui import nice_app as v1
    ch = dict(v1.BUILDER_DEFAULTS, medium="track", car_endplates=True,
              car_endplate_chord_min_m=0.12, car_endplate_chord_max_m=0.34)
    flags = v1.car_flags(ch)
    assert flags["endplate_chord_min_m"] == pytest.approx(0.12)
    assert flags["endplate_chord_max_m"] == pytest.approx(0.34)
    # the plain fence carries the WING's chord — it has no plate chord to bound
    fence = v1.car_flags(dict(ch, car_endplates=False))
    assert "endplate_chord_min_m" not in fence
    # ...and an unstated band sends nothing at all
    blank = v1.car_flags(dict(ch, car_endplate_chord_min_m=None,
                              car_endplate_chord_max_m=None))
    assert "endplate_chord_min_m" not in blank
    assert "endplate_chord_max_m" not in blank


def test_a_typed_band_does_not_rebuild_its_own_field():
    from gui import nice_app as v1
    from gui.v3.stages import wing as v3_wing
    for key in ("car_endplate_chord_min_m", "car_endplate_chord_max_m"):
        assert key in v1.NUMBER_CHOICE_KEYS, key
        assert key in v3_wing.VALUE_KEYS, key
        assert key in v1.BUILDER_DEFAULTS, key
        assert v1.BUILDER_DEFAULTS[key] is None, key


def test_the_results_page_shows_both_chords():
    from gui import metrics
    specs = list(metrics.PERFORMANCE)
    for group in metrics.SPECIALITY.values():
        specs += list(group)
    keys = {k for spec in specs for k in spec.keys}
    assert "endplate_chord_requested_m" in keys
    assert "endplate_chord_m" in keys
    assert "endplate_chord_ratio" in keys
