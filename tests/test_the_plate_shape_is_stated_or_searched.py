"""The endplate's cant and its root blend: the reader answers, or the search.

Two numbers the shell has always asked for and nothing in this repository
records an answer to — no measurement here says what a rear wing's plates lean
by, or how far its corner is turned. That is the case for handing them to the
optimiser; a plate built to a drawing is the case for keeping the field. So
both, as a toggle.

Each is a DIMENSION change, so each is a registry variant and not a flag —
``api._WING_CANTS``'s rule — and the three consequences that follow are what
is asserted here: the row lands inside the family block where the size rows
are still addressed backwards from the chord coefficients, the STATED flag is
subtracted from the family that searches it, and the shell derives the family
from the toggle and inverts back to it.

Also gated here: the cant the JUNCTION charge is built on. Four consumers of
the angle read the field and a fifth read the literal 90.0, which would have
held every candidate of a searched cant upright in the one term that prices
the corner.
"""

import numpy as np
import pytest

from aerobo import api
from aerobo import endplate as ep

BASE = "car rear wing + endplates"
CANT = f"{BASE} [free cant]"
BLEND = f"{BASE} [free blend]"
BOTH = f"{BASE} [free cant, free blend]"


def _built(name, flags=None):
    return api.PROBLEM_SPECS[name].build({}, dict(flags or {}), None)


def test_each_freedom_is_a_family_and_costs_exactly_one_row():
    """A flag cannot change the length of a design vector, so this is a
    problem and not a switch — and the row count is the check that it is one
    row and not a re-layout."""
    dims = {n: len(api.PROBLEM_SPECS[n].param_labels)
            for n in (BASE, CANT, BLEND, BOTH)}
    assert dims == {BASE: 11, CANT: 12, BLEND: 12, BOTH: 13}
    assert api.PROBLEM_SPECS[CANT].param_labels.count("endplate_cant_deg") == 1
    assert api.PROBLEM_SPECS[BLEND].param_labels.count(
        "endplate_blend_frac") == 1


def test_the_new_rows_sit_inside_the_family_block():
    """carwing.span_from_x and carwing.s_from_x address the size rows
    BACKWARDS from the chord coefficients, so a row appended behind them would
    silently re-address the span and the area with nothing failing. The check
    is that the size block still closes the vector."""
    for name in (BASE, CANT, BLEND, BOTH):
        labels = list(api.PROBLEM_SPECS[name].param_labels)
        assert labels[-2:] == ["S_m2", "b_m"]
        assert labels[:9] == list(ep.CarWingEndplateProblem.BASE_LABELS)
    # ...and with a chord law on top the coefficients are still the tail
    twin = list(api.PROBLEM_SPECS[f"{BOTH} + free chord law"].param_labels)
    assert twin[-3:] == ["chord_k1", "chord_k2", "chord_k3"]
    assert twin[-5:-3] == ["S_m2", "b_m"]
    assert twin[-7:-5] == ["endplate_cant_deg", "endplate_blend_frac"]


def test_the_searched_row_is_the_angle_that_is_actually_flown():
    """A row nothing reads is decoration. Moving it has to move the plate —
    in the width accounting, in the reach margin and in what it costs."""
    built = _built(CANT)
    i = list(built.param_labels).index("endplate_cant_deg")
    lo, hi = built.bounds[i]
    assert (lo, hi) == pytest.approx(tuple(map(
        float, ep.CarWingEndplateProblem.ENDPLATE_CANT_BOUNDS_DEG)))

    x = built.bounds.mean(axis=1)
    x[i] = 90.0
    upright = built.evaluate(x)
    x[i] = 55.0
    leaning = built.evaluate(x)

    assert upright["endplate_cant_deg"] == pytest.approx(90.0)
    assert leaning["endplate_cant_deg"] == pytest.approx(55.0)
    assert upright["endplate_cant_searched"] is True
    # an upright plate projects nothing; a leaning one is paid for out of the
    # OVERALL width, which is what the span row bounds
    assert upright["endplate_projection_m"] == pytest.approx(0.0)
    assert leaning["endplate_projection_m"] >= 0.0
    # ...and it reaches less far DOWN, which is what the deck cares about
    assert leaning["endplate_tip_z_m"] < upright["endplate_tip_z_m"]
    assert leaning["g_reach"] < upright["g_reach"]
    # ...and it is a longer, more loaded member
    assert leaning["endplate_arm_m"] > upright["endplate_arm_m"]


def test_the_searched_blend_is_the_corner_that_is_actually_turned():
    """The blend is priced at three places at once — the corner's
    interference charge, the plate's outboard reach and how far down it still
    gets — so it has an interior answer rather than a bound to ride."""
    built = _built(BLEND, {"junction_drag": True})
    i = list(built.param_labels).index("endplate_blend_frac")
    assert tuple(built.bounds[i]) == pytest.approx((0.0, 1.0))

    x = built.bounds.mean(axis=1)
    x[i] = 0.0
    crease = built.evaluate(x)
    x[i] = 0.6
    blended = built.evaluate(x)

    assert crease["endplate_blend_frac"] == pytest.approx(0.0)
    assert blended["endplate_blend_frac"] == pytest.approx(0.6)
    assert blended["endplate_blend_searched"] is True
    # what a blend BUYS: the corner is cheaper
    assert blended["CD_junction"] < crease["CD_junction"]
    # what it COSTS: the plate reaches outboard, and the wing pays out of the
    # overall width
    assert blended["endplate_projection_m"] > crease["endplate_projection_m"]
    # ...out of the OVERALL width, which is what the span row bounds: the two
    # candidates measure the same across the car, and the blended one has
    # less WING inside that measurement
    assert blended["overall_width_m"] == \
        pytest.approx(crease["overall_width_m"])
    assert blended["b_m"] < crease["b_m"]


def test_stating_a_number_the_search_is_choosing_is_refused():
    """One question, one answer — enforced twice, because the two halves can
    drift: the registry does not DECLARE the flag on the family that searches
    it, and the problem itself raises if both arrive anyway."""
    assert "endplate_cant_deg" in api.PROBLEM_SPECS[BASE].flags
    assert "endplate_cant_deg" not in api.PROBLEM_SPECS[CANT].flags
    assert "blend_frac" in api.PROBLEM_SPECS[BASE].flags
    assert "blend_frac" not in api.PROBLEM_SPECS[BLEND].flags
    # blend_SHAPE survives: it names WHICH law the turn follows, not how much
    # of one there is, and a categorical law is not a length to search
    assert "blend_shape" in api.PROBLEM_SPECS[BLEND].flags

    with pytest.raises(KeyError, match="endplate_cant_deg"):
        api.check_flags(CANT, {"endplate_cant_deg": 60.0})
    with pytest.raises(KeyError, match="blend_frac"):
        api.check_flags(BLEND, {"blend_frac": 0.3})
    # the fixed family still takes both, so nothing was taken away
    api.check_flags(BASE, {"endplate_cant_deg": 60.0, "blend_frac": 0.3})

    with pytest.raises(ValueError, match="second answer"):
        ep.CarWingEndplateProblem(cant_free=True, endplate_cant_deg=60.0)
    with pytest.raises(ValueError, match="second answer"):
        ep.CarWingEndplateProblem(blend_free=True, blend_frac=0.3)


@pytest.mark.parametrize("cant", [90.0, 60.0, 45.0])
def test_the_junction_is_charged_at_the_cant_the_candidate_flew(cant):
    """The fifth consumer of the angle read the literal 90.0 while the other
    four read the field, so a leaning plate was charged a corner built on a
    fillet radius shrunk by cant/90 — and a SEARCHED cant would have held
    every candidate upright in the one term that prices the corner.

    Asserted against the closed form rather than against another run,
    because the geometry moves with the cant too (a leaning plate shortens
    the wing) and a run-to-run comparison cannot separate the two. The fillet
    radius is `(blend_frac*h + wing_arc)/|cant|` in radians, and nothing else
    in the report says which angle the corner was built on.
    """
    blend = 0.5
    prob = ep.CarWingEndplateProblem(blend_frac=blend, junction_drag=True,
                                     endplate_cant_deg=cant)
    x = prob.bounds.mean(axis=1)
    r = ep.evaluate_car_wing_endplate(x, prob)
    assert r["feasible"]
    j = r["junction"]
    want = (blend * r["endplate_h_m"] + j["blend_wing_arc_m"]) \
        / abs(np.deg2rad(cant))
    assert j["blend_radius_m"] == pytest.approx(want, rel=1e-9)
    # ...and the same number when the cant is a SEARCHED row rather than a
    # stated field, which is the case the literal would have broken silently
    built = _built(CANT, {"blend_frac": blend})
    xs = built.bounds.mean(axis=1)
    xs[list(built.param_labels).index("endplate_cant_deg")] = cant
    rs = built.evaluate(xs)
    js = rs["junction"]
    assert js["blend_radius_m"] == pytest.approx(
        (blend * rs["endplate_h_m"] + js["blend_wing_arc_m"])
        / abs(np.deg2rad(cant)), rel=1e-9)


def test_an_untouched_family_is_the_published_run_bit_for_bit():
    """A family gaining two freedoms must not move its own baseline: the
    fixed variant IS the shipped problem, name, vector and numbers."""
    built = _built(BASE)
    assert tuple(built.param_labels) == (
        "taper", "twist_root_deg", "twist_tip_deg", "alpha_deg",
        "endplate_h_m", "ride_height_m", "endplate_chord_ratio",
        "endplate_tc", "endplate_toe_deg", "S_m2", "b_m")
    r = built.evaluate(built.bounds.mean(axis=1))
    assert r["feasible"]
    assert r["endplate_cant_deg"] == pytest.approx(90.0)
    assert r["endplate_blend_frac"] == pytest.approx(0.0)
    assert r["endplate_cant_searched"] is False
    assert r["endplate_blend_searched"] is False
    assert r["endplate_projection_m"] == pytest.approx(0.0)


def test_the_shell_toggle_picks_the_family_and_inverts_back_to_it():
    """The toggle is the whole feature: it must select the family, and a
    stored family must restore the toggle that selected it — otherwise a
    reloaded session shows "I state it" over a row the optimiser is riding."""
    import gui.nice_app as v1

    want = {("fixed", "fixed"): BASE, ("free", "fixed"): CANT,
            ("fixed", "free"): BLEND, ("free", "free"): BOTH}
    for (cant, blend), name in want.items():
        ch = v1.choices_from_problem(BASE)
        ch["car_plate_cant"], ch["car_plate_blend"] = cant, blend
        assert v1.derive_problem(ch)[0] == name
        back = v1.choices_from_problem(name)
        assert back["car_plate_cant"] == cant
        assert back["car_plate_blend"] == blend
        assert v1.derive_problem(back)[0] == name


def test_the_typed_value_survives_a_trip_through_the_search():
    """Answering one more question must never LOSE an answer already given.
    The number lives in ``choices``, which a family change does not filter,
    so the field comes back with what the reader typed in it."""
    import gui.nice_app as v1

    ch = v1.choices_from_problem(BASE)
    ch["car_endplate_cant_deg"] = 62.0
    ch["car_plate_cant"] = "free"
    assert v1.derive_problem(ch)[0] == CANT
    assert v1.car_flags(ch, 55.0).get("endplate_cant_deg") is None
    ch["car_plate_cant"] = "fixed"
    assert v1.derive_problem(ch)[0] == BASE
    assert v1.car_flags(ch, 55.0)["endplate_cant_deg"] == pytest.approx(62.0)


def test_the_two_rows_have_a_name_and_a_reason_in_the_design_box():
    """A searched row a reader cannot name is a row nobody can set a band on;
    api carries no display names, so the shell must."""
    import gui.nice_app as v1

    for row in ("endplate_cant_deg", "endplate_blend_frac"):
        label, why = v1.param_help(row)
        assert label and label != row
        assert len(why) > 80


def test_the_predicates_read_the_registry_and_not_the_name():
    """A shell that branched on a name would be wrong the day a family grew a
    twin — which is what just happened three times over."""
    assert api.plate_cant_is_searched(CANT)
    assert api.plate_cant_is_searched(f"{BOTH} + free chord law")
    assert not api.plate_cant_is_searched(BLEND)
    assert not api.plate_cant_is_searched(BASE)
    assert api.plate_blend_is_searched(BLEND)
    assert not api.plate_blend_is_searched(CANT)
    assert not api.plate_cant_is_searched("car rear wing")
    assert api.PLATE_FREEDOMS == ("cant", "blend")
    assert api.plate_freedom_name(BASE, ()) == BASE
    assert api.plate_freedom_name(BASE, {"cant", "blend"}) == BOTH
