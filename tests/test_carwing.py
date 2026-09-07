"""Car rear wing: mirrored-frame downforce, ground effect, endplates and —
the point of the module — what the MOUNT layout actually changes.

BOTH layouts take the load out through the PLATES; a rear wing is bolted to
the car by the sheets it already carries, and this family offers no pylon.
What the two layouts disagree about is WHERE ALONG THE SPAN the sheets grip,
so the tests check that the grip station changes the things it physically
changes (the beam, the extra sheets' wetted area, the corners they make) and
NOT the things it does not (the solved circulation).
"""

import numpy as np
import pytest

from aerobo import api, carmount
from aerobo.carwing import (
    G_FAIL,
    MOUNTS,
    PENALTY,
    CarWingProblem,
    bending_moment,
    compare_mounts,
    deflection_index,
    evaluate_car_wing,
    fg_car_wing,
)

PROB = CarWingProblem()


def _x(taper=0.8, tw_root=0.0, tw_tip=-2.0, alpha=8.0, ep=0.12, ride=0.30,
       b=1.6):
    """The SOLVER's vector when it is handed no area band: ``ep`` is the
    endplate height IN METRES and ``b`` the SPAN — both are lengths, and the
    span is the last (size-block) entry."""
    return np.array([taper, tw_root, tw_tip, alpha, ep, ride, b])


def _xa(S=0.29, **kw):
    """The REGISTERED family's vector: the area row sits AHEAD of the span
    row, so every span reader keeps its slot (``carwing.s_from_x``)."""
    return np.insert(_x(**kw), 6, S)


# ---------------- beam models (pure) ----------------

def _uniform(n=201, b=1.6, w=100.0):
    y = np.linspace(-b / 2, b / 2, n)
    return y, np.full(n, w)


def test_uniform_load_gives_the_textbook_moment_for_both_layouts():
    """A cantilever off the centreline and a beam simply supported at its
    tips carry the SAME peak moment w b^2/8 under uniform load — the layouts
    differ in sense and in deflection, not in peak demand. Getting this
    identity out of the numerics is the check that both formulas are right."""
    b, w = 1.6, 100.0
    y, load = _uniform(b=b, w=w)
    centre = bending_moment(y, load, b, "centre")
    ends = bending_moment(y, load, b, "ends")
    expect = w * b * b / 8.0
    assert centre.max() == pytest.approx(expect, rel=1e-4)
    assert ends.max() == pytest.approx(expect, rel=1e-4)


def test_moment_vanishes_at_the_free_ends():
    b = 1.6
    y, load = _uniform(b=b)
    centre = bending_moment(y, load, b, "centre")
    ends = bending_moment(y, load, b, "ends")
    assert centre[np.argmax(y)] == pytest.approx(0.0, abs=1e-9)   # free tip
    assert ends[np.argmax(y)] == pytest.approx(0.0, abs=1e-9)     # pinned tip
    assert centre.argmax() == ends.argmax() == int(np.argmin(np.abs(y)))


def test_deflection_matches_the_closed_forms_for_both_layouts():
    """Uniform load w, unit EI. Centre mount = two cantilevers of length
    b/2, free-end droop w(b/2)^4/8 = w b^4/128. End mount = a beam simply
    supported over b, centre sag 5 w b^4/384. So the tip-supported wing
    moves 5/3x as far — same load, same spar, different mount."""
    b, w = 1.6, 100.0
    y, load = _uniform(b=b, w=w)
    d_centre = deflection_index(y, bending_moment(y, load, b, "centre"), b,
                               "centre")
    d_ends = deflection_index(y, bending_moment(y, load, b, "ends"), b, "ends")
    assert d_centre == pytest.approx(w * b**4 / 128.0, rel=1e-3)
    assert d_ends == pytest.approx(5.0 * w * b**4 / 384.0, rel=1e-3)
    assert d_ends / d_centre == pytest.approx(5.0 / 3.0, rel=1e-3)


def test_beam_helpers_validate_their_inputs():
    y, load = _uniform()
    with pytest.raises(ValueError, match="unknown supports"):
        bending_moment(y, load, 1.6, "middle")
    with pytest.raises(ValueError, match="same shape"):
        bending_moment(y, load[:-1], 1.6, "centre")


# ---------------- the aerodynamics ----------------

def test_downforce_is_positive_in_car_terms():
    out = evaluate_car_wing(_x(), PROB)
    assert out["feasible"]
    assert out["CZ"] > 0.0 and out["downforce_N"] > 0.0
    assert out["efficiency"] == pytest.approx(out["CZ"] / out["CD"])


def test_ground_proximity_makes_more_downforce_for_less_induced_drag():
    """The classic effect, and the reason the ride height is a design
    variable rather than a constant."""
    low = evaluate_car_wing(_x(ride=0.15), PROB)
    high = evaluate_car_wing(_x(ride=0.60), PROB)
    assert low["CZ"] > high["CZ"]
    assert low["CDi"] < high["CDi"]
    assert low["efficiency"] > high["efficiency"]


def test_endplates_help_and_are_priced_as_wetted_area():
    bare = evaluate_car_wing(_x(ep=0.0), PROB)
    plated = evaluate_car_wing(_x(ep=0.24), PROB)
    assert plated["CZ"] > bare["CZ"]
    assert plated["CDi"] < bare["CDi"]
    assert plated["CDp"] > bare["CDp"]        # more wetted strips


def test_an_endplate_that_reaches_the_track_is_refused():
    """0.30 * 0.8 m of plate cannot fit under a wing 0.15 m off the ground:
    the image system would overlap the real one."""
    out = evaluate_car_wing(_x(ep=0.24, ride=0.15), PROB)
    assert out["feasible"] is False
    assert "plane" in out["reason"]


def test_in_contract_failures_never_raise():
    assert evaluate_car_wing(np.zeros(3), PROB)["feasible"] is False
    assert fg_car_wing(np.zeros(3), PROB) == (PENALTY, [G_FAIL, G_FAIL])
    bad = _x(alpha=99.0)
    assert evaluate_car_wing(bad, PROB)["feasible"] is False


# ---------------- the mount question ----------------

def test_the_grip_station_changes_the_structure_not_the_circulation():
    """Same wing, same wake — a different beam and a different sheet bill.

    The grip is the only thing that moves between the two layouts, so the
    solved circulation must be bit-identical while the beam and the mount's
    own drag both change, each in a stated direction: an interior support
    cuts the peak moment and the deflection, and pays for it in drag.
    """
    both = compare_mounts(_x(), PROB)
    tips, inboard = both["tips"], both["inboard"]
    assert tips["CZ"] == pytest.approx(inboard["CZ"], rel=1e-12)
    assert tips["CDi"] == pytest.approx(inboard["CDi"], rel=1e-12)
    # ...but the beam and the mount's own drag do differ, and this way round
    assert inboard["M_max_Nm"] < 0.9 * tips["M_max_Nm"]     # peak demand cut
    assert inboard["deflection_m"] < 0.5 * tips["deflection_m"]   # >2x stiffer
    assert inboard["CD"] > tips["CD"]                       # and it costs
    assert inboard["efficiency"] < tips["efficiency"]


def test_the_inboard_grip_pays_sheet_and_junction_drag_and_the_tips_one_does_not():
    """The tips are free: the plates are there for the nonplanar benefit
    anyway, so using them as the load path buys the whole stiffness of a
    simply supported beam for nothing. The inboard grip adds two sheets of
    wetted area and the two corners where they meet the wing, and that sum
    is the WHOLE of the difference in drag."""
    both = compare_mounts(_x(), PROB)
    assert both["inboard"]["cd0_struts"] > 0.0
    assert both["inboard"]["CD_junction"] > 0.0
    assert both["tips"]["cd0_struts"] == 0.0
    assert both["tips"]["CD_junction"] == 0.0
    assert both["inboard"]["CD"] - both["tips"]["CD"] == pytest.approx(
        both["inboard"]["cd0_struts"] + both["inboard"]["CD_junction"])


def test_the_sheet_charge_follows_the_plate_height_and_not_the_ride_height():
    """What replaced "the pylons are as tall as the wing is high".

    There is no pylon left to grow with the ride height: the sheets stand
    beside the endplate, so the mount charge scales with the PLATE's height
    and is untouched by raising the wing. Mounting a rear wing higher now
    costs it nothing but the ground effect it gives up.
    """
    inb = CarWingProblem(mount="inboard")
    short = evaluate_car_wing(_x(ep=0.05), inb)
    tall = evaluate_car_wing(_x(ep=0.24), inb)
    assert short["feasible"] and tall["feasible"]
    # a sheet is as tall as the plate beside it: the charge is proportional
    assert tall["cd0_struts"] / short["cd0_struts"] == pytest.approx(
        0.24 / 0.05, rel=1e-9)

    low = evaluate_car_wing(_x(ride=0.20), inb)
    high = evaluate_car_wing(_x(ride=0.55), inb)
    assert high["cd0_struts"] == pytest.approx(low["cd0_struts"], rel=1e-12)
    assert high["CD"] != pytest.approx(low["CD"], rel=1e-6)   # but the wake did
    # ...and the layout that grips at the tips pays nothing at either height
    for ride in (0.20, 0.55):
        assert evaluate_car_wing(_x(ride=ride), PROB)["cd0_struts"] == 0.0


def test_no_registered_layout_carries_a_pylon():
    """The pylon left the family, and a deck no longer buys a reach margin.

    Both layouts are plate-borne, so there is nothing standing between the
    wing and the car's attachment deck to be too short. Stating a deck
    therefore adds no margin at all — which is the observable half of "there
    is no pylon layout", and would fail the moment one came back.
    """
    assert set(MOUNTS) == {"tips", "inboard"}
    for name, layout in MOUNTS.items():
        p = CarWingProblem(mount=name, deck_height_m=0.25)
        assert p.n_pylons == 0 and layout["n_struts"] == 0
        assert p.reach_required is False
        assert "pylon reach margin" not in p.constraint_labels
        assert layout["supports"] == "ends"
        # the two tables state ONE station, imported rather than pasted: the
        # beam is driven off it, so a disagreement would fly two wings
        assert carmount.PUBLISHED_LAYOUTS[name].station_frac == pytest.approx(
            layout["station_frac"])
        assert carmount.PUBLISHED_LAYOUTS[name].kind == "endplate"
        assert carmount.PUBLISHED_LAYOUTS[name].n_pylons == 0
    assert set(carmount.PUBLISHED_LAYOUTS) == set(MOUNTS)
    # the tips are the default everywhere, because they are the free layout
    assert CarWingProblem().mount == "tips"


def test_the_deflection_constraint_can_decide_between_mounts():
    """With a tight deflection limit the tip-borne wing goes infeasible
    while the inboard grip survives — the same wing, a different answer,
    which is the whole point of asking where the plates grip it."""
    from dataclasses import replace

    x = _x(alpha=11.0, ride=0.20)
    loose = compare_mounts(x, PROB)
    d_t, d_i = loose["tips"]["deflection_m"], loose["inboard"]["deflection_m"]
    assert d_i < d_t
    limit = replace(PROB, deflection_limit_m=0.5 * (d_t + d_i))
    tips = evaluate_car_wing(x, replace(limit, mount="tips"))
    inboard = evaluate_car_wing(x, replace(limit, mount="inboard"))
    assert inboard["g_deflection"] > 0.0 > tips["g_deflection"]


def test_the_tip_borne_beam_is_the_closed_form_and_the_inboard_one_is_not():
    """One SIGNED beam serves both layouts now, at the grip station.

    At the tips that station is +/- b/2, so the answer has to reproduce the
    exported closed form for a beam simply supported at its ends, to the bit
    — that is what says the general integrator did not quietly change the
    published wing. Inboard it must NOT: the moment changes sign at an
    interior support, the span outboard of it HOGS, and integrating the
    magnitude would give a plausible, wrong (larger) deflection there.
    """
    for name in MOUNTS:
        p = CarWingProblem(mount=name)
        out = evaluate_car_wing(_x(), p)
        closed = deflection_index(
            out["y"], bending_moment(out["y"], out["load_Npm"], out["b_m"],
                                     "ends"),
            out["b_m"], "ends") / p.ei_nm2
        if name == "tips":
            assert out["deflection_m"] == pytest.approx(closed, rel=1e-12)
            assert out["bending_sense"] == "sagging"
        else:
            assert out["deflection_m"] < 0.5 * closed
            assert out["bending_sense"] == "hogging"


def test_unknown_mount_is_refused():
    with pytest.raises(ValueError, match="unknown mount"):
        CarWingProblem(mount="glued")
    # the pylon layouts are gone by NAME too, so a stored case that asked for
    # one is refused rather than silently flown as something else
    for gone in ("centre", "ends"):
        with pytest.raises(ValueError, match="unknown mount"):
            CarWingProblem(mount=gone)


# ---------------- API + GUI wiring ----------------

def test_registered_as_an_eight_dimensional_one_constraint_problem():
    """Nothing is budgeted, so the only margin left is the structural one.

    The static registry labels and the live problem's own have to agree —
    ``api.design_report`` reads the live ones, and a family advertising a
    drag budget it does not carry would label the deflection margin as a
    drag one.
    """
    spec = api.PROBLEM_SPECS["car rear wing"]
    assert spec.is_constrained and spec.n_constraints == 1
    assert spec.constraint_labels == ("deflection margin",)
    built = spec.build({}, {}, None)
    assert built.dim == 8 == len(built.param_labels)
    assert built.problem.constraint_labels == spec.constraint_labels
    assert built.problem.CD_budget is None and built.problem.drag_budget_n is None
    f, g = built.callable(_xa())
    assert np.isfinite(f) and len(g) == 1


def test_mount_and_budgets_travel_as_flags():
    spec = api.PROBLEM_SPECS["car rear wing"]
    built = spec.build({}, {"mount": "inboard", "CD_budget": 0.09,
                            "junction_drag": False}, None)
    assert built.problem.mount == "inboard"
    assert built.problem.CD_budget == 0.09
    assert built.problem.junction_drag is False
    # ...and a stated budget is a margin the run actually carries: "nothing
    # is budgeted by default" must not become "nothing may be budgeted"
    assert built.problem.constraint_labels == ("drag budget margin",
                                               "deflection margin")
    assert len(built.callable(_xa())[1]) == 2
    assert built.dim == spec.build({}, {}, None).dim   # flags never add dims


def test_a_mission_card_is_refused_rather_than_ignored():
    """The car problem has no MissionSpec; silently dropping W_N would make
    the Mission card look like it did something."""
    with pytest.raises(ValueError, match="no mission spec"):
        api.PROBLEM_SPECS["car rear wing"].build({"W_N": 700.0}, {}, None)


def test_builder_maps_the_track_medium():
    import gui.nice_app as v1

    # the MOUNT picks the family: car_endplates False = the pylon layout, and
    # the plain fence family with it. The card opens on True (the plates
    # carry the car), which is asserted in test_car_operating_point_and_mount.
    ch = dict(v1.BUILDER_DEFAULTS, medium="track", car_endplates=False)
    assert v1.derive_problem(ch)[0] == "car rear wing"
    notes = v1.derive_problem(dict(ch, winglets="free"))[1]
    assert any("Endplates are already a design variable" in n for n in notes)


# ---------------- the reference area as a design variable ----------------
#
# The area used to be configuration, and one line of the class docstring said
# why: every coefficient is referenced to it, so holding it fixed is what let
# CZ be the score. The REGISTRY now always frees it — how big the wing is is
# a design question exactly as how wide it is — but the SOLVER stays general
# and still fixes its area when handed no band, which is what carsection.py's
# fixed reference planform and compare_split's matched-area comparison need.
# So these tests hold both halves: the solver's published behaviour, and the
# registry's insistence that the score be stated in something a moving
# reference area does not eat.

def test_the_solver_still_fixes_its_area_when_handed_no_band():
    p = CarWingProblem()
    assert p.area_free is False and p.objective == "cz"
    assert p.constraint_labels == ("drag budget margin", "deflection margin")
    assert p.n_constraints == 2 and p.dim == 7
    out = evaluate_car_wing(_x(), p)
    # the score IS the coefficient, and the area is the problem's
    assert out["score"] == pytest.approx(out["CZ"])
    assert out["S_m2"] == pytest.approx(p.S)
    assert len(out["g"]) == 2


def test_a_coefficient_objective_against_a_free_area_is_refused():
    """Not a preference — both coefficients are referenced to the area searched.

    CZ is MAXIMISED by shrinking that area and CD is MINIMISED by growing it,
    so each rides an area bound whatever the aerodynamics do and comes back
    looking like a result. The refusal must also NAME the way out, or it is
    just a wall.
    """
    band = (0.10, 0.48)
    with pytest.raises(ValueError) as exc:
        CarWingProblem(area_bounds_m2=band)                  # objective='cz'
    msg = str(exc.value)
    assert "downforce" in msg and "efficiency" in msg
    with pytest.raises(ValueError) as exc_cd:
        CarWingProblem(area_bounds_m2=band, objective="cd")
    assert "'drag'" in str(exc_cd.value), \
        "the CD refusal does not name the force objective that replaces it"
    # ...and every FORCE objective is accepted at the same band
    for obj in ("downforce", "efficiency", "drag"):
        CarWingProblem(area_bounds_m2=band, objective=obj)


def test_the_area_row_goes_ahead_of_the_span_so_the_span_keeps_its_slot():
    """The structural claim behind ``s_from_x`` / ``span_from_x``.

    Put the area AFTER the span instead and ``span_from_x`` silently returns
    the AREA — a wing solved at a span of 0.29 m — while every shape test
    still passes. So this asserts the two read back the two numbers that went
    in, at both chord orders, which is what pins the order.
    """
    from aerobo.carwing import s_from_x, span_from_x

    for chord_order in (0, 3):
        p = CarWingProblem(area_bounds_m2=(0.10, 0.48),
                           objective="downforce", chord_order=chord_order)
        x = np.concatenate([_xa(S=0.31, b=1.75), np.zeros(chord_order)])
        assert x.size == p.dim
        assert s_from_x(x, chord_order) == pytest.approx(0.31)
        assert span_from_x(x, chord_order) == pytest.approx(1.75)
        out = evaluate_car_wing(x, p)
        assert out["S_m2"] == pytest.approx(0.31)
        assert out["b_m"] == pytest.approx(1.75)


def test_the_force_objective_rewards_a_bigger_wing_and_the_coefficient_did_not():
    """The defect, as an outcome.

    Shrinking the reference area RAISES CZ and CZ/CD and LOWERS the downforce
    the car actually gets. A family scored on CZ therefore answers "make it
    tiny"; scored on the force it answers the question the user asked.
    """
    p = CarWingProblem(area_bounds_m2=(0.10, 0.48), objective="downforce",
                       CD_budget=None)
    got = [evaluate_car_wing(_xa(S=S), p) for S in (0.15, 0.30, 0.45)]
    cz = [o["CZ"] for o in got]
    fz = [o["downforce_N"] for o in got]
    eff = [o["efficiency"] for o in got]
    assert cz[0] > cz[1] > cz[2]            # the coefficient rises as S falls
    assert eff[0] > eff[1] > eff[2]         # ...and so does CZ/CD
    assert fz[0] < fz[1] < fz[2]            # ...while the FORCE does not
    for o in got:                           # and the score follows the force
        assert o["score"] == pytest.approx(o["downforce_N"])


def test_a_budget_switched_off_shortens_the_margin_vector():
    """"No limit" and "the limit did not bind" are different answers.

    A satisfied placeholder would report a margin for a budget that does not
    exist, and ``design_report`` would label a deflection margin as a drag
    one. The failure vector's width has to follow too, or the optimiser gets
    a shape error at the first refused design.
    """
    off = CarWingProblem(CD_budget=None)
    assert off.constraint_labels == ("deflection margin",)
    assert off.n_constraints == 1
    assert len(evaluate_car_wing(_x(), off)["g"]) == 1
    assert len(fg_car_wing(_x() * 0 - 99.0, off)[1]) == 1      # refused design

    both = CarWingProblem(drag_budget_n=90.0, downforce_min_n=300.0)
    assert both.constraint_labels == (
        "drag budget margin", "drag force margin", "deflection margin",
        "downforce floor margin")
    out = evaluate_car_wing(_x(), both)
    assert len(out["g"]) == 4 == both.n_constraints
    assert fg_car_wing(_x(), both)[1] == pytest.approx(out["g"])
    # the force margins are signed the usual way: >= 0 iff satisfied
    assert (out["g_drag_force"] >= 0) == (out["drag_N"] <= 90.0)
    assert (out["g_downforce"] >= 0) == (out["downforce_N"] >= 300.0)


def test_an_aspect_ratio_outside_the_solvers_band_is_refused_in_contract():
    """One chordwise panel is only honest over ``sizing.AR_LIMITS``.

    The default area band cannot reach outside it, so the gate is only
    reachable by widening the SPAN band — which a caller may do, which is
    exactly why the check is per candidate and not an assumption.
    """
    p = CarWingProblem(area_bounds_m2=(0.10, 0.48), objective="downforce",
                       span_bounds_m=(0.6, 4.0))
    lo = evaluate_car_wing(_xa(S=0.48, b=0.8), p)     # AR 1.33
    hi = evaluate_car_wing(_xa(S=0.10, b=4.0), p)     # AR 160
    for out in (lo, hi):
        assert not out["feasible"] and "aspect ratio" in out["reason"]
        assert out["score"] == PENALTY
    assert evaluate_car_wing(_xa(S=0.30, b=1.6), p)["feasible"]


def test_the_registry_budgets_nothing_and_the_allowance_is_still_derivable():
    """An allowance handed to a maximiser is a number the answer RIDES.

    So the registered family opens with no drag margin of either kind, and
    the published allowance survives only as a function a caller may state
    for themselves — still derived from its own definition, never pasted.
    """
    from aerobo.carwing import published_drag_budget_n

    p = CarWingProblem()
    q = 0.5 * p.rho * p.V ** 2
    assert published_drag_budget_n() == pytest.approx(p.CD_budget * q * p.S)

    built = api.PROBLEM_SPECS["car rear wing"].build({}, {}, None)
    assert built.problem.CD_budget is None
    assert built.problem.drag_budget_n is None
    assert built.problem.constraint_labels == ("deflection margin",)
    assert built.problem.objective == api.CAR_DEFAULT_OBJECTIVE == "efficiency"
    # ...and the score IS the ratio, which is what makes a free area and no
    # ceiling a well-posed question at all
    out = built.evaluate(_xa())
    assert out["score"] == pytest.approx(out["CZ"] / out["CD"])
    assert built.callable(_xa())[0] == pytest.approx(out["score"])

    # a ceiling is a LIMIT the user states, in newtons, and it then bites
    capped = api.PROBLEM_SPECS["car rear wing"].build(
        {}, {"drag_budget_n": 10.0, "CD_budget": "off"}, None)
    assert capped.problem.constraint_labels == ("drag force margin",
                                                "deflection margin")
    tight = capped.evaluate(_xa())
    assert tight["g_drag_force"] < 0.0 and tight["drag_N"] > 10.0


def test_the_base_family_is_the_free_area_one_and_composes_with_the_chord_law():
    """There is no fixed-area/free-area PAIR any more: the base name is the
    free-area problem, so nothing has to be renamed and no stored case key is
    orphaned. Six car entries — three bases and their three chord-law twins."""
    assert "car rear wing (free area)" not in api.PROBLEM_SPECS
    # ...and the designed-plate family's own two freedoms — the plate's cant
    # and its root blend — are each a searched twin rather than a flag,
    # because each adds a row to the design vector.
    assert sorted(n for n in api.PROBLEM_SPECS if "car" in n) == sorted([
        "car rear wing",
        "car rear wing (two-element)",
        "car rear wing + endplates",
        "car rear wing + endplates [free cant]",
        "car rear wing + endplates [free blend]",
        "car rear wing + endplates [free cant, free blend]",
        "car rear wing + free chord law",
        "car rear wing (two-element) + free chord law",
        "car rear wing + endplates + free chord law",
        "car rear wing + endplates [free cant] + free chord law",
        "car rear wing + endplates [free blend] + free chord law",
        "car rear wing + endplates [free cant, free blend] + free chord law",
    ])

    spec = api.PROBLEM_SPECS["car rear wing"]
    assert spec.is_constrained and spec.medium == "air"
    built = spec.build({}, {}, None)
    assert built.dim == 8 == len(built.param_labels)
    assert built.param_labels[6:8] == ("S_m2", "b_m")
    assert built.problem.area_free
    # the area row is FLOWN, not decoration: a different S is a different wing
    small = built.evaluate(_xa(S=0.15))
    big = built.evaluate(_xa(S=0.45))
    assert small["S_m2"] == pytest.approx(0.15)
    assert big["S_m2"] == pytest.approx(0.45)
    assert big["downforce_N"] > small["downforce_N"]

    twin = api.PROBLEM_SPECS["car rear wing + free chord law"]
    tb = twin.build({}, {}, None)
    assert tb.dim == 11 and tb.param_labels[6:8] == ("S_m2", "b_m")
    assert tb.param_labels[8:] == ("chord_k1", "chord_k2", "chord_k3")
    assert tb.problem.area_bounds_m2 == built.problem.area_bounds_m2


def test_the_size_bands_are_design_box_rows_and_not_flags():
    """One question, one place. ``b_m`` and ``S_m2`` are ordinary rows of the
    design box, so a band stated there is the band the problem validates and
    reports — and the old flag spellings are refused outright, which is what
    stops a shell quietly re-opening the second channel."""
    from aerobo.carwing import AREA_BOUNDS_M2, SPAN_BOUNDS_M

    plain = api._build_car_wing(None, {}, {})
    assert plain.problem.span_bounds_m is None       # the family's own default
    np.testing.assert_allclose(plain.bounds[7], SPAN_BOUNDS_M)
    np.testing.assert_allclose(plain.bounds[6], AREA_BOUNDS_M2)

    built = api._build_car_wing(None, {}, {"b_m": (1.2, 3.0),
                                           "S_m2": (0.2, 0.3)})
    assert built.problem.span_bounds_m == (1.2, 3.0)
    assert built.problem.area_bounds_m2 == (0.2, 0.3)
    # the box SHOWN and the box the problem validates are one object
    np.testing.assert_allclose(built.bounds[7], [1.2, 3.0])
    np.testing.assert_allclose(built.problem.bounds[7], [1.2, 3.0])
    np.testing.assert_allclose(built.bounds[6], [0.2, 0.3])
    np.testing.assert_allclose(built.problem.bounds[6], [0.2, 0.3])
    assert built.dim == 8                     # a band never adds a dimension

    for gone in ("span_min_m", "span_max_m", "area_min_m2", "area_max_m2"):
        with pytest.raises(KeyError, match=gone):
            api.check_flags("car rear wing", {gone: 1.5})


def test_a_widened_span_row_is_actually_searched():
    """The defect, as an outcome, and the reason the flag had to go.

    The row moved the SAMPLER while ``prob.bounds`` stayed at the family
    default, so every draw above 2.0 m came back ``feasible=False,
    reason='bounds violation'``, score -100: a user who widened the row got a
    run made entirely of refusals.
    """
    built = api._build_car_wing(None, {}, {"b_m": (1.2, 3.0)})
    out = built.evaluate(_xa(S=0.25, b=2.6))
    assert out["feasible"], out["reason"]
    assert out["b_m"] == pytest.approx(2.6)
    assert out["score"] > 0.0 and out["score"] != PENALTY
    # ...and the narrow family default still refuses that same vector, which
    # is what says the row is doing the work and not the width of the vector
    narrow = api._build_car_wing(None, {}, {})
    assert narrow.evaluate(_xa(S=0.25, b=2.6))["reason"] == "bounds violation"


def test_the_size_band_helper_is_idempotent_under_rebuild():
    """A contract gui.v3.relax asserts: it clips a widened row back to the
    validated band, rebuilds, and requires the clip to have converged in ONE
    pass. A helper that re-derived or re-clamped the band it was handed would
    leave the row outside validity on the second pass too, and the whole
    no-solution reach card would abort for the car."""
    first = api._build_car_wing(None, {}, {"b_m": (1.2, 3.0),
                                           "S_m2": (0.2, 0.3)})
    rows = {"b_m": tuple(first.problem.bounds[7]),
            "S_m2": tuple(first.problem.bounds[6])}
    second = api._build_car_wing(None, {}, rows)
    np.testing.assert_allclose(second.problem.bounds, first.problem.bounds)
    np.testing.assert_allclose(second.bounds, first.bounds)


def test_a_malformed_size_row_meets_the_familys_own_message():
    """The row reaches the validator WHOLE, so a band the wrong way round is
    refused by ``carwing``'s own sentence rather than by a TypeError from
    somewhere downstream."""
    with pytest.raises(ValueError, match="span bounds must satisfy"):
        api._build_car_wing(None, {}, {"b_m": (2.0, 1.0)})
    with pytest.raises(ValueError, match="area bounds must satisfy"):
        api._build_car_wing(None, {}, {"S_m2": (0.0, 1.0)})


def test_the_objective_and_the_two_limits_still_travel_as_flags():
    spec = api.PROBLEM_SPECS["car rear wing"]
    built = spec.build({}, {"car_objective": "downforce",
                            "downforce_min_n": 400.0,
                            "CD_budget": "off"}, None)
    assert built.problem.objective == "downforce"
    assert built.problem.CD_budget is None
    assert built.problem.downforce_min_n == 400.0
    assert built.problem.constraint_labels == ("deflection margin",
                                               "downforce floor margin")
    out = built.evaluate(_xa())
    assert out["score"] == pytest.approx(out["downforce_N"])
    assert (out["g_downforce"] >= 0) == (out["downforce_N"] >= 400.0)


def test_switching_the_coefficient_budget_off_is_spelled_not_guessed():
    """``CD_budget`` carries a number or a mode string; anything else raises.

    A flag loop that skips ``None`` cannot carry "switch this off" as a value,
    and a zero budget is a wing that may make no drag at all — so the off
    switch is spelled, and a typo is refused rather than read as 0.
    """
    spec = api.PROBLEM_SPECS["car rear wing"]
    assert spec.build({}, {"CD_budget": "off"}, None).problem.CD_budget is None
    assert spec.build({}, {"CD_budget": "none"}, None).problem.CD_budget is None
    with pytest.raises(ValueError, match="car budget"):
        spec.build({}, {"CD_budget": "no"}, None)
    with pytest.raises(ValueError, match="CD_budget"):
        CarWingProblem(CD_budget=0.0)


def test_the_v1_builder_has_no_free_area_switch_left_to_throw():
    """There is nothing to switch: every car family designs its area, so the
    card asks how big the wing may be in the one place a band belongs (the
    design box's ``S_m2`` row) and the builder emits neither band as a flag."""
    import gui.nice_app as v1

    assert "car_free_area" not in v1.BUILDER_DEFAULTS
    for gone in ("car_span_min_m", "car_span_max_m",
                 "car_area_min_m2", "car_area_max_m2"):
        assert gone not in v1.BUILDER_DEFAULTS
    assert not hasattr(v1, "_car_span_band")
    assert not hasattr(v1, "_car_area_band")

    ch = dict(v1.BUILDER_DEFAULTS, medium="track", car_endplates=False)
    assert v1.derive_problem(ch)[0] == "car rear wing"
    assert v1.derive_problem(dict(ch, car_endplates=True))[0] == (
        "car rear wing + endplates")
    # whatever the card says, no size band leaves it as a flag — and what
    # does leave it is accepted by the family it was derived for
    for extra in ({}, {"car_endplates": True}, {"car_two_element": True}):
        cfg = dict(ch, **extra)
        name, _ = v1.derive_problem(cfg)
        flags = v1.car_flags(cfg)
        assert not ({"span_min_m", "span_max_m", "area_min_m2", "area_max_m2"}
                    & set(flags))
        api.check_flags(name, flags)


def test_the_v1_builder_sends_the_two_limits_and_budgets_nothing_otherwise():
    """D4's other half at the card: the drag menu is gone, so an untouched
    card asks for no drag constraint at all, and a typed ceiling travels."""
    import gui.nice_app as v1

    assert not hasattr(v1, "CAR_DRAG_BUDGET_LABELS")
    assert not hasattr(v1, "_car_drag_budget_default")
    assert "car_drag_budget" not in v1.BUILDER_DEFAULTS

    ch = dict(v1.BUILDER_DEFAULTS, medium="track")
    assert "drag_budget_n" not in v1.car_flags(ch)
    assert api.PROBLEM_SPECS["car rear wing"].build(
        {}, v1.car_flags(ch), None).problem.constraint_labels == (
            "deflection margin",)

    typed = v1.car_flags(dict(ch, car_drag_budget_n=90.0,
                              car_downforce_min_n=400.0))
    assert typed["drag_budget_n"] == 90.0
    assert typed["downforce_min_n"] == 400.0
