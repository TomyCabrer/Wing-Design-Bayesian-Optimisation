"""The winglet families that demand their section from the coupled library.

Two objective.Problem modes — ``winglet_coupled`` and
``winglet_capped_coupled`` — give the nonplanar VLM winglet families the
pre-optimised (t/c x cl) CST section library that the wingless
``wing+airfoil (coupled)`` problem already flies (wing_airfoil.py docstring
A: the summary-variable coupling of Sobieszczanski-Sobieski & Haftka 1997).
The wing level commands structural depth and design lift; a section-level BO
already delivered the shape.

What these tests are FOR
------------------------
The feature itself is thirty lines. What is not thirty lines is the set of
registry-wide tables a new base family has to join, none of which the
feature's own physics can notice is missing. Three of them fail SILENTLY,
and each has a named test below whose job is to go red under exactly that
omission:

  * ``geometry.wing_from_x`` has its OWN mode tuple. Miss it and ``tc_sec``
    is a DEAD ROW: the optimiser moves it and nothing reads it.
    -> ``test_tc_sec_is_a_live_row`` (and, in the sized variant, the weight
       model reads it too: ``test_tc_sec_reaches_the_weight_model``).
  * the span cap is a SECOND tuple. Miss it and the "span-capped" problem
    does not cap — it still builds and still scores, just not as the thing
    its name says.  -> ``test_the_capped_twin_actually_caps``.
  * the new modes must NOT join ``TC_MODES``: membership there routes to the
    ONE-argument ``family.at(tc)``, which would silently fire on a
    two-dimensional library and leave ``cl_sec`` dead.
    -> ``test_the_coupled_modes_are_not_thickness_modes`` and
       ``test_cl_sec_is_a_live_row``.

The controlling identity, and it is an ``==`` not an ``approx``: a coupled
mode differs from the plain winglet mode in EXACTLY ONE WAY — which polar it
flies. Fed the polar its two summary variables select, the plain mode
reproduces it bit-for-bit (``test_coupled_is_the_plain_winglet_on_the
_library_polar``). That single statement gates both mode tables at once,
because the plain capped mode is where the arc-length rescale lives.

NOTE on a control this session had to correct. PLAN_SESSION43.md predicted
that "at zero device height it reproduces ``wing+airfoil (coupled)``, and it
should be ==". It does not, and should not: the winglet families run the
nonplanar VLM and the wingless coupled family runs the lifting line, so the
two disagree at h = 0 by the same solver gap the plain winglet family
already has to ``trim wing``. That gap is measured and pinned below
(``test_the_zero_height_gap_is_the_solver_gap_and_nothing_more``) rather
than asserted away.
"""

import numpy as np
import pytest

from aerobo import api, geometry, objective, polar, wing_airfoil

FREE = "winglet + airfoil (coupled)"
CAPPED = "winglet_capped + airfoil (coupled)"

#: an OFF-CENTRE probe. The box centre of TC_SEC_BOUNDS is 0.12, which is
#: also geometry.TC_DEFAULT — so a dead tc_sec row would be invisible at the
#: centre of the design box. Every thickness assertion here is off it.
X_PROBE = np.array([0.62, 1.0, -2.0, 0.09, 74.0, 0.101, 0.44])


@pytest.fixture(scope="module")
def family():
    return polar.load_cst_polar_family()


def _prob(mode, family, **kw):
    return objective.Problem(mode=mode, polar_family=family, **kw)


# ------------------------------------------------- 1. the mode tables agree

def test_the_coupled_modes_are_not_thickness_modes():
    """Trap 3. ``TC_MODES`` membership routes to the ONE-argument
    ``family.at(tc)``; the coupled library is two-dimensional, so a coupled
    mode in that list would silently fly a NACA 24XX member chosen by
    thickness alone and leave ``cl_sec`` unread."""
    assert set(objective.COUPLED_MODES).isdisjoint(objective.TC_MODES)
    assert set(api.COUPLED_MODES).isdisjoint(api.TC_MODES)
    # ...and the same list is one list: api mirrors objective mirrors
    # geometry, which owns it (three of the places that must agree are in
    # geometry, and a second literal is a second thing to forget)
    assert api.COUPLED_MODES == objective.COUPLED_MODES
    assert objective.COUPLED_MODES == geometry.COUPLED_MODES


def test_a_coupled_mode_carries_a_winglet_and_the_capped_one_caps():
    """Tables 1 and 2. Absent from ``WINGLET_MODES`` the VLM branch never
    runs at all; absent from ``CAPPED_MODES`` the span-capped problem
    builds, scores, and does not cap."""
    assert set(objective.COUPLED_MODES) <= set(objective.WINGLET_MODES)
    assert "winglet_capped_coupled" in objective.CAPPED_MODES
    assert "winglet_coupled" not in objective.CAPPED_MODES
    # the two capped tuples are ONE set: geometry caps the cosine modes in
    # closed form and the blended ones on the developed line, and their
    # union is the list objective rescales the arc length for
    blended_capped = tuple(m for m in objective.CAPPED_MODES
                           if m in objective.BLENDED_MODES)
    assert set(geometry.CAPPED_COSINE_MODES) | set(blended_capped) \
        == set(objective.CAPPED_MODES)


def test_the_wrong_family_arity_is_refused_at_construction(family):
    """One field, two kinds of family, and the mode says which — so a caller
    who hands the wrong one must be told at construction.

    Escaping, it surfaced as a bare ``TypeError: PolarFamily.at() takes 2
    positional arguments but 3 were given``, raised hundreds of lines into
    ``evaluate`` — which breaks this module's failure contract (an
    in-contract failure is a PENALTY dict, never an exception at the
    optimiser) and says nothing about what the caller did wrong.
    """
    from aerobo.polar import default_polar_family

    with pytest.raises(ValueError, match="needs a PolarFamily2D"):
        objective.Problem(mode="winglet_coupled",
                          polar_family=default_polar_family())
    with pytest.raises(ValueError, match="needs a 1-D PolarFamily"):
        objective.Problem(mode="winglet_tc", polar_family=family)
    # ...and the RIGHT pairings still build, both ways round
    assert objective.Problem(mode="winglet_coupled",
                             polar_family=family) is not None
    assert objective.Problem(
        mode="winglet_tc", polar_family=default_polar_family()) is not None
    # a mode that reads NEITHER is untouched by the check
    assert objective.Problem(mode="winglet", polar_family=family) is not None


def test_the_library_hull_has_one_pair_of_numbers():
    """geometry owns the two rows (its ``bounds`` builds the winglet box and
    it may not import wing_airfoil); wing_airfoil re-exports them."""
    assert wing_airfoil.TC_SEC_BOUNDS is geometry.TC_SEC_BOUNDS
    assert wing_airfoil.CL_SEC_BOUNDS is geometry.CL_SEC_BOUNDS
    assert geometry.TC_SEC_BOUNDS == (0.09, 0.15)
    assert geometry.CL_SEC_BOUNDS == (0.3, 0.7)


# ------------------------------------------------- 2. the box and the vector

@pytest.mark.parametrize("mode", objective.COUPLED_MODES)
def test_the_box_is_the_winglet_box_plus_the_two_library_rows(mode):
    box = objective.Problem(mode=mode).bounds
    assert box.shape == (7, 2)
    plain = objective.Problem(mode=mode.replace("_coupled", "")
                              or "winglet").bounds
    assert np.array_equal(box[:5], plain[:5])
    assert tuple(box[5]) == geometry.TC_SEC_BOUNDS
    assert tuple(box[6]) == geometry.CL_SEC_BOUNDS


def test_the_registered_names_declare_the_vector_they_search():
    for name in (FREE, CAPPED):
        spec = api.PROBLEM_SPECS[name]
        assert spec.param_labels[-2:] == ("tc_sec", "cl_sec")
        built = spec.build({}, {}, None)
        assert built.dim == 7 == len(built.param_labels)


# --------------------------------------------- 3. both new rows are LIVE

def test_tc_sec_is_a_live_row(family):
    """Trap 1. ``geometry.wing_from_x`` must read x[5] as the Wing's
    thickness: it is the structural depth of the library member being
    demanded, and it is what the polar lookup is keyed on. Dead, the Wing
    keeps TC_DEFAULT and the row moves nothing."""
    p = _prob("winglet_coupled", family)
    lo, hi = X_PROBE.copy(), X_PROBE.copy()
    lo[5], hi[5] = 0.095, 0.145
    a, b = objective.evaluate(lo, p), objective.evaluate(hi, p)
    assert a["feasible"] and b["feasible"]
    # the row REACHES the geometry...
    assert a["tc"] == pytest.approx(0.095)
    assert b["tc"] == pytest.approx(0.145)
    assert a["tc_sec"] == pytest.approx(0.095)
    # ...and it MOVES the answer
    assert a["score"] != b["score"]
    # ...and it moved it through the SECTION, not through the planform:
    # thickness is not a planform variable, so the wing is the same wing
    assert a["wing"].b == b["wing"].b
    assert a["wing"].S == b["wing"].S
    assert a["wing"].taper == b["wing"].taper


def test_cl_sec_is_a_live_row(family):
    """The second summary variable. If the mode were routed through the
    one-argument thickness lookup this row would be read by nothing."""
    p = _prob("winglet_coupled", family)
    lo, hi = X_PROBE.copy(), X_PROBE.copy()
    lo[6], hi[6] = 0.35, 0.65
    a, b = objective.evaluate(lo, p), objective.evaluate(hi, p)
    assert a["feasible"] and b["feasible"]
    assert a["score"] != b["score"]
    assert a["cl_sec"] == pytest.approx(0.35)
    assert b["cl_sec"] == pytest.approx(0.65)


def test_neither_row_reaches_across_into_the_other(family):
    """The block-boundary check. ``cl_sec`` selects a member; it is not a
    geometric quantity, so it must leave the WING untouched — and ``tc_sec``
    must not move the design lift."""
    p = _prob("winglet_coupled", family)
    base = objective.evaluate(X_PROBE, p)
    moved_cl = X_PROBE.copy()
    moved_cl[6] = 0.65
    out = objective.evaluate(moved_cl, p)
    assert out["tc"] == base["tc"]           # the planform is untouched
    assert out["wing"].b == base["wing"].b
    assert out["tc_sec"] == base["tc_sec"]
    moved_tc = X_PROBE.copy()
    moved_tc[5] = 0.145
    out2 = objective.evaluate(moved_tc, p)
    assert out2["cl_sec"] == base["cl_sec"]


# ------------------------------- 4. the controlling identity (== not approx)

@pytest.mark.parametrize("coupled,plain", [
    ("winglet_coupled", "winglet"),
    ("winglet_capped_coupled", "winglet_capped"),
])
def test_coupled_is_the_plain_winglet_on_the_library_polar(coupled, plain,
                                                           family):
    """The whole feature, stated once: a coupled mode is the plain winglet
    mode flying the polar its two summary variables select. Nothing else
    changes — not the cap, not the arc length, not the trim.

    This is the single strongest gate on both mode tables: the arc-length
    rescale that the cap needs lives in the PLAIN capped branch, so if the
    coupled mode fell out of ``objective.CAPPED_MODES`` (or out of
    ``geometry.CAPPED_COSINE_MODES``) the two sides would part company here.
    """
    pol = family.at(float(X_PROBE[5]), float(X_PROBE[6]))
    got = objective.evaluate(X_PROBE, _prob(coupled, family))
    ref = objective.evaluate(X_PROBE[:5], objective.Problem(mode=plain,
                                                            polar=pol))
    assert got["feasible"] and ref["feasible"]
    assert got["score"] == ref["score"]          # bit-for-bit
    assert got["CD"] == ref["CD"]
    assert got["wing"].b == ref["wing"].b


def test_the_capped_twin_actually_caps(family):
    """Trap 2, positively. The capped mode shrinks the wing panel to pay for
    the device's horizontal projection; the free-span one does not."""
    free = objective.evaluate(X_PROBE, _prob("winglet_coupled", family))
    cap = objective.evaluate(X_PROBE, _prob("winglet_capped_coupled", family))
    b_nom = 10.0
    assert free["wing"].b == b_nom
    h, cant = float(X_PROBE[3]), float(X_PROBE[4])
    assert cap["wing"].b == pytest.approx(
        b_nom * (1.0 - h * np.cos(np.deg2rad(cant))))
    assert cap["wing"].b < free["wing"].b
    # ...and at cant = 90 the projection vanishes and the cap coincides with
    # the free-span mode exactly, as it does for every other capped family
    x90 = X_PROBE.copy()
    x90[4] = 90.0
    a = objective.evaluate(x90, _prob("winglet_coupled", family))
    b = objective.evaluate(x90, _prob("winglet_capped_coupled", family))
    assert a["score"] == b["score"]


def test_the_zero_height_gap_is_the_solver_gap_and_nothing_more(family):
    """At h = 0 there is no device, so the coupled winglet family and the
    wingless ``wing+airfoil (coupled)`` family are the same design — flown by
    two different solvers (nonplanar VLM vs lifting line). The gap between
    them must therefore be EXACTLY the gap the plain winglet family already
    has to ``trim wing`` on the same polar: the feature adds no second
    difference of its own."""
    tc, cl = 0.11, 0.45
    pol = family.at(tc, cl)
    x5 = np.array([0.6, 1.0, -2.0, tc, cl])
    llt = wing_airfoil.evaluate_wing_airfoil(
        x5, wing_airfoil.WingAirfoilProblem(family=family))
    x7 = np.array([0.6, 1.0, -2.0, 0.0, 75.0, tc, cl])
    vlm = objective.evaluate(x7, _prob("winglet_coupled", family))
    # the control pair, same two solvers, no library involved
    llt_ref = objective.evaluate(np.array([0.6, 1.0, -2.0]),
                                 objective.Problem(mode="trim", polar=pol))
    vlm_ref = objective.evaluate(np.array([0.6, 1.0, -2.0, 0.0, 75.0]),
                                 objective.Problem(mode="winglet", polar=pol))
    assert llt["score"] == llt_ref["score"]      # bit-for-bit
    assert vlm["score"] == vlm_ref["score"]      # bit-for-bit
    # frozen 2026-08-07: the VLM reads ~0.24 % higher than the lifting line
    # on this design; it is a solver difference, not a winglet
    assert vlm["score"] - llt["score"] == pytest.approx(0.0840, abs=5e-4)


# ------------------------------------- 5. the tables the registry checks

def test_every_registered_name_builds_and_flies():
    """All 32 (2 families x the 16 modifier subsets they support) build,
    declare a vector of the length they search, and evaluate feasibly at the
    centre of their own box."""
    names = [n for n in api.problem_names()
             if api.base_of(n) in (FREE, CAPPED)]
    assert len(names) == 32
    for name in names:
        built = api.PROBLEM_SPECS[name].build({}, {}, None)
        assert built.dim == len(built.param_labels) == built.bounds.shape[0]
        out = built.evaluate(built.bounds.mean(axis=1))
        assert out["feasible"], (name, out.get("reason"))


def test_the_size_the_flight_state_and_the_chord_law_all_compose():
    for name in (FREE, CAPPED):
        assert api.with_modifiers(name, {"size"}) is not None
        assert api.with_modifiers(name, {"size_ws"}) is not None       # 5
        assert api.with_modifiers(name, {"size_ws_free"}) is not None
        assert api.with_modifiers(name, {"flight", "chord"}) is not None
        # ...and a chosen size is offered, which is table 6: without it the
        # family flies a span nobody can state
        for key in api.PLANFORM_KEYS:
            assert key in api.PROBLEM_SPECS[name].flags, (name, key)


def test_a_fixed_blend_and_a_winglet_type_have_somewhere_to_live():
    """Tables 3 and 4: the cant band (a winglet TYPE) and the fixed blend."""
    for name in (FREE, CAPPED):
        assert api.WINGLET_BLEND_KEY in api.PROBLEM_SPECS[name].flags
    built = api.PROBLEM_SPECS[CAPPED].build(
        {}, {"winglet_type": "raked"}, None)
    lo, hi = built.bounds[4]
    assert (lo, hi) != geometry.WINGLET_CANT_BOUNDS_DEG   # a real sub-band


def test_a_raked_tip_is_refused_on_the_free_span_family():
    """The TENTH table, and a guard the feature walked straight through.

    ``_winglet_cant_bounds`` refused a raked type on a free-span mode by
    matching a hard-coded LITERAL of mode names — so it silently stopped
    applying to every free-span winglet family added after it was written,
    and `winglet + airfoil (coupled)` was accepting the very configuration
    whose error message says it "reports a fabricated L/D". Now asked of the
    mode's capped-ness, so it cannot go stale again.

    Stated over the whole winglet family rather than over the two new names:
    a rule expressed as a list is what created this.
    """
    for name, spec in api.PROBLEM_SPECS.items():
        mode = getattr(spec.build, "objective_mode", None)
        if mode not in api._WINGLET_MODES:
            continue
        capped = mode in objective.CAPPED_MODES
        try:
            spec.build({}, {"winglet_type": "raked"}, None)
            accepted = True
        except ValueError:
            accepted = False
        assert accepted == capped, (name, mode, accepted, capped)
    blended = api.PROBLEM_SPECS[FREE].build(
        {}, {api.WINGLET_BLEND_KEY: 0.4}, None)
    assert blended.problem.blend_frac_fixed == pytest.approx(0.4)


def test_a_chosen_section_is_not_offered_to_a_family_that_designs_one():
    """The same rule the t/c modes obey: these modes read a polar FAMILY per
    candidate, so a fixed table handed in would silently disable the two
    rows they search."""
    for name in (FREE, CAPPED):
        assert api.SECTION_KEY not in api.PROBLEM_SPECS[name].flags


def test_tc_sec_reaches_the_area_the_loading_resolves(family):
    """``geometry.TC_INDEX`` — the OTHER consumer of x[5], and the one whose
    absence is hardest to see.

    In the wing-loading size mode the AREA follows W/S through the weight
    fixed point, so it needs the thickness BEFORE the wing exists — that is
    the only reason ``geometry.tc_from_x`` is called at all
    (objective.evaluate's WS branch). Read off a table that does not list
    these modes it returns TC_DEFAULT, and every candidate's area is closed
    against a 12 % section it is not flying.

    The obvious assertion — that ``W_wing_N`` moves — does NOT gate this:
    that number is reported from ``sized_state``, which is handed
    ``wing.tc`` after the wing is built and is therefore right either way.
    The AREA is the number that goes wrong. (Found by mutating the guarded
    line: the first version of this test survived it.)
    """
    built = api.PROBLEM_SPECS[FREE + " + free span (W/S)"].build({}, {}, None)
    x = np.concatenate([X_PROBE, [12.0]])
    thin, thick = x.copy(), x.copy()
    thin[5], thick[5] = 0.095, 0.145
    a, b = built.evaluate(thin), built.evaluate(thick)
    assert a["feasible"] and b["feasible"]
    assert a["wing"].S != b["wing"].S
    # thicker spar cap -> lighter wing (Raymer) -> less total weight -> less
    # area at the same loading. The SIGN is the physics, not "a number moved"
    assert b["wing"].S < a["wing"].S
    assert b["W_wing_N"] < a["W_wing_N"]


# --------------------------------------------------- 6. the GUI round trip

def test_the_menu_answers_are_about_this_family():
    """Table 8, BOTH halves. A name the inverse table does not know inverts
    to the DEFAULT choices, and every menu entry decided by that round trip
    is then answered about the wrong family."""
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve()
                           .parents[1] / "gui"))
    import nice_app as na

    assert na.derive_problem({"winglets": "free",
                              "airfoil": "coupled"})[0] == FREE
    assert na.derive_problem({"winglets": "capped",
                              "airfoil": "coupled"})[0] == CAPPED
    for name in (FREE, CAPPED):
        assert na.derive_problem(dict(na.PROBLEM_TO_CHOICES[name]))[0] == name
    # and the two new labels have help text (table 9)
    for lbl in ("tc_sec", "cl_sec"):
        assert lbl in na.DESIGN_PARAM_HELP
