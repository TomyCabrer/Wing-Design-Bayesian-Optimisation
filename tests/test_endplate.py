"""Car rear-wing ENDPLATES as a designed part (aerobo.endplate).

Covers, in order:

* the section families — inertia factors derived from the thickness law, not
  quoted, and the flat/shaped stiffness-vs-drag trade that motivates having
  two of them at all;
* the reduced-order component models (parasite drag, yaw load case,
  cantilever deflection and stress) against their closed forms;
* the VLM's new per-panel section: bit-for-bit defaults, and the trap the
  feature exists to close (a CAMBERED zero-lift angle silently toes a
  VERTICAL surface);
* the problem itself — the penalty contract, the drag decomposition, the
  reach-the-car constraint (unconditional now that both mount layouts are
  plate-borne), what the mount DOES still decide — the wing's beam and the
  sheets it is gripped by — and the toe trade having an interior optimum;
* the registry, where the reference AREA is always a design variable, the
  size bands are rows of the design box rather than flags, and nothing is
  budgeted until a ceiling is stated in newtons.
"""

from __future__ import annotations

import numpy as np
import pytest

from aerobo import endplate as ep
from aerobo import geometry, junction
from aerobo.carwing import MOUNTS
from aerobo.geometry import Wing
from aerobo.vlm import VLM

PROB = ep.CarWingEndplateProblem()
XMID = 0.5 * (PROB.bounds[:, 0] + PROB.bounds[:, 1])


def _x(**kw) -> np.ndarray:
    """Midpoint design vector with named entries replaced.

    ``h_m`` is the plate's height IN METRES and ``b`` the wing's SPAN: both
    are lengths in this vector, and the span is its last (size-block) entry.
    """
    idx = {"taper": 0, "twist_root": 1, "twist_tip": 2, "alpha": 3,
           "h_m": 4, "ride": 5, "chord_ratio": 6, "tc": 7, "toe": 8,
           "b": 9}
    x = XMID.copy()
    for k, v in kw.items():
        x[idx[k]] = float(v)
    return x


# --------------------------------------------------------- section families
def test_a_stated_ceiling_does_not_revive_the_coefficient_allowance():
    """One stated limit must produce exactly ONE drag margin.

    The registry switches ``CD_budget`` off so that nothing is budgeted by
    default. Gated on "neither budget stated" — which is how it was first
    written — a user who typed the newton ceiling (the only drag flag any
    shell emits) got the dataclass's 0.11 coefficient allowance back with it,
    and the run carried TWO drag constraints: the one they asked for, and the
    published allowance on a reference area the search is moving. Asserted on
    the MARGIN NAMES, because that is what a reader of the results page sees,
    and the count is what ``diagnose`` indexes margins by.
    """
    from aerobo import api

    for name in ("car rear wing", "car rear wing + endplates"):
        plain = api.constraint_labels_of(name)
        assert not [m for m in plain if "drag" in m], (name, plain)
        capped = api.constraint_labels_of(name, {"drag_budget_n": 90.0})
        drag = [m for m in capped if "drag" in m]
        assert drag == ["drag force margin"], (name, capped)
        built = api.PROBLEM_SPECS[name].build(None, {"drag_budget_n": 90.0},
                                              None)
        assert built.problem.CD_budget is None, name
        assert built.problem.drag_budget_n == 90.0, name
        # ...and a caller who explicitly wants the coefficient one still gets
        # it: "no budget by default" must not become "no budget, ever"
        coef = api.constraint_labels_of(name, {"CD_budget": 0.11})
        assert [m for m in coef if "drag" in m] == ["drag budget margin"], name



def test_section_registry_is_complete_and_validated():
    assert set(ep.SECTIONS) == {"flat", "rounded", "shaped"}
    for name, spec in ep.SECTIONS.items():
        assert spec["label"] and spec["note"]
        assert spec["inertia_factor"] > 0.0
        assert isinstance(spec["streamlined"], bool)
        assert spec["edge_cd"] >= 0.0
        # a streamlined section pays a form factor INSTEAD of an edge charge;
        # a constant-thickness one pays the edge charge instead of a form
        # factor. Neither ever pays both, and neither ever pays nothing.
        assert (spec["edge_cd"] > 0.0) is not spec["streamlined"]
        assert ep.section_spec(name) is spec
    with pytest.raises(ValueError, match="unknown endplate section"):
        ep.section_spec("carbon")


def test_the_rounded_plate_is_the_flat_ones_stiffness_at_a_quarter_the_edge():
    """The family the bracket was missing.

    ``flat`` and ``shaped`` are the two ENDS: stiffest-per-thickness against
    lowest-drag. Measured over this family's own box, ``shaped`` has the lower
    drag at every thickness offered, so ``flat`` can only ever be chosen for
    its 2.1x stiffness — which decides anything only where the plate's
    deflection limit is live. ``rounded`` is the part that actually gets
    bolted to a car: a constant-thickness panel with radiused edges, keeping
    all of the flat plate's inertia and paying a quarter of its edge lump.
    """
    assert (ep.SECTIONS["rounded"]["inertia_factor"]
            == ep.SECTIONS["flat"]["inertia_factor"])
    assert ep.ROUNDED_EDGE_CD == pytest.approx(0.25 * ep.FLAT_EDGE_CD)
    assert not ep.SECTIONS["rounded"]["streamlined"]      # no form factor
    assert ep.form_factor("rounded", 0.12) == 1.0

    kw = dict(s_ref=0.4, rho=1.225, V=55.0, mu=1.8e-5)
    flat = ep.parasite_cd("flat", 0.6, 0.45, 0.01, **kw)
    rnd = ep.parasite_cd("rounded", 0.6, 0.45, 0.01, **kw)
    shaped = ep.parasite_cd("shaped", 0.6, 0.45, 0.01, **kw)
    # same wetted area and skin friction; the edges are the whole difference
    assert rnd["CD_friction"] == pytest.approx(flat["CD_friction"])
    assert rnd["CD_edges"] == pytest.approx(0.25 * flat["CD_edges"])
    assert shaped["CD_edges"] == 0.0
    assert flat["CD"] > rnd["CD"] > shaped["CD"]
    # ...and it is stiffer than the streamlined section it now beats on drag
    # margin, which is the corner the bracket could not express
    assert (ep.bending_inertia("rounded", 0.6, 0.006)
            > 2.0 * ep.bending_inertia("shaped", 0.6, 0.006))


def test_flat_plate_inertia_is_the_rectangle_exactly():
    """A constant-thickness plate's second moment is c t^3/12 — no fudge."""
    assert ep.SECTIONS["flat"]["inertia_factor"] == pytest.approx(1.0 / 12.0,
                                                                 rel=0, abs=0)
    c, t = 0.4, 0.006
    assert ep.bending_inertia("flat", c, t) == pytest.approx(c * t**3 / 12.0)


def test_shaped_inertia_factor_comes_from_the_thickness_law():
    """k_I = (1/12) int (t/t_max)^3 dxi, recomputed here from the NACA
    thickness distribution: a shaped section keeps ~47% of a rectangle's
    bending inertia at the same MAXIMUM thickness, because its material
    tapers away from the edges."""
    xi = np.linspace(0.0, 1.0, 400001)
    t_hat = ep._naca_thickness_shape(xi)
    t_hat = t_hat / t_hat.max()
    expect = float(np.trapezoid(t_hat**3, xi) / 12.0)

    k = ep.SECTIONS["shaped"]["inertia_factor"]
    assert k == pytest.approx(expect, rel=1e-6)
    assert k == pytest.approx(0.0394, abs=5e-4)
    assert 0.44 < k / ep.SECTIONS["flat"]["inertia_factor"] < 0.50


def test_bending_inertia_scales_with_chord_and_cube_of_thickness():
    base = ep.bending_inertia("shaped", 0.3, 0.01)
    assert ep.bending_inertia("shaped", 0.6, 0.01) == pytest.approx(2 * base)
    assert ep.bending_inertia("shaped", 0.3, 0.02) == pytest.approx(8 * base)
    for bad in ((0.0, 0.01), (0.3, 0.0), (-1.0, 0.01)):
        with pytest.raises(ValueError, match="must be > 0"):
            ep.bending_inertia("shaped", *bad)


def test_form_factor_flat_is_one_shaped_is_the_strut_correlation():
    assert ep.form_factor("flat", 0.15) == 1.0        # thickness charged as
    assert ep.form_factor("flat", 0.01) == 1.0        # edge drag instead
    for tc in (0.02, 0.08, 0.15):
        assert ep.form_factor("shaped", tc) == pytest.approx(
            1.0 + 2.0 * tc + 60.0 * tc**4)
    ffs = [ep.form_factor("shaped", tc) for tc in (0.02, 0.06, 0.12, 0.18)]
    assert all(b > a for a, b in zip(ffs, ffs[1:]))
    with pytest.raises(ValueError, match="thickness ratio"):
        ep.form_factor("shaped", 1.5)


# ---------------------------------------------------------- component models

def test_parasite_cd_zero_height_is_zero_and_area_scales_linearly():
    kw = dict(s_ref=0.4, rho=1.225, V=55.0, mu=1.789e-5)
    assert ep.parasite_cd("shaped", 0.3, 0.0, 0.08, **kw)["CD"] == 0.0
    one = ep.parasite_cd("shaped", 0.3, 0.2, 0.08, **kw)
    two = ep.parasite_cd("shaped", 0.3, 0.4, 0.08, **kw)
    assert two["Swet_m2"] == pytest.approx(2 * one["Swet_m2"])
    assert two["CD"] == pytest.approx(2 * one["CD"])
    # both faces of both plates
    assert one["Swet_m2"] == pytest.approx(2 * 2 * 0.3 * 0.2)


def test_flat_plate_pays_edge_drag_and_loses_to_the_shaped_section():
    kw = dict(chord=0.3, height=0.3, tc=0.08, s_ref=0.4, rho=1.225,
              V=55.0, mu=1.789e-5)
    flat = ep.parasite_cd("flat", **kw)
    shaped = ep.parasite_cd("shaped", **kw)
    assert flat["CD_edges"] > 0.0 and shaped["CD_edges"] == 0.0
    assert flat["CD"] > shaped["CD"]
    # same wetted area, same friction: the whole difference is edges vs form
    assert flat["Swet_m2"] == pytest.approx(shaped["Swet_m2"])
    assert flat["CD"] - flat["CD_edges"] < shaped["CD"]


def test_yaw_side_force_is_linear_and_finite_span_corrected():
    q, area = 0.5 * 1.225 * 55.0**2, 0.09
    assert ep.yaw_side_force(q, area, 1.0, 0.0) == 0.0
    y3 = ep.yaw_side_force(q, area, 1.0, 3.0)
    assert ep.yaw_side_force(q, area, 1.0, 6.0) == pytest.approx(2 * y3)
    # the finite-span correction always reduces the 2-D slope, and its
    # effect vanishes as the plate gets tall
    two_d = q * area * ep.TWO_PI * np.deg2rad(3.0)
    assert y3 < two_d
    assert ep.yaw_side_force(q, area, 1e6, 3.0) == pytest.approx(two_d,
                                                                 rel=1e-5)
    assert (ep.yaw_side_force(q, area, 4.0, 3.0)
            > ep.yaw_side_force(q, area, 1.0, 3.0))


def test_cantilever_deflection_and_stress_match_the_closed_forms():
    P, L, E, I = 60.0, 0.25, 7.0e10, 3.0e-8
    assert ep.lateral_deflection(P, L, E, I) == pytest.approx(
        P * L**3 / (3 * E * I))
    assert ep.lateral_deflection(-P, L, E, I) == pytest.approx(
        ep.lateral_deflection(P, L, E, I))          # magnitude
    assert ep.lateral_deflection(P, 2 * L, E, I) == pytest.approx(
        8 * ep.lateral_deflection(P, L, E, I))      # cubic in the arm
    t = 0.01
    assert ep.root_stress(P, L, t, I) == pytest.approx(P * L * (t / 2) / I)
    with pytest.raises(ValueError):
        ep.lateral_deflection(P, L, 0.0, I)
    with pytest.raises(ValueError):
        ep.root_stress(P, L, t, 0.0)


# ------------------------------------------------------ VLM per-panel section

_WING = Wing(b=1.6, S=0.4, taper=0.7, twist_tip_deg=-2.0)
_VLM_KW = dict(N=24, winglet_h_frac=0.3, winglet_cant_deg=90.0, n_winglet=8,
               a=6.1, alpha_L0=np.deg2rad(-2.0), V=55.0)


def test_default_winglet_section_is_bit_for_bit_the_tip_extension_model():
    """The new knobs must not move a single published number: passing their
    own defaults explicitly has to reproduce the untouched model exactly."""
    base = VLM(_WING, **_VLM_KW)
    same = VLM(_WING, **_VLM_KW, winglet_chord_scale=1.0, winglet_toe_deg=0.0,
               winglet_a=None, winglet_alpha_L0=None)
    a, b = base.solve(0.1), same.solve(0.1)
    assert a.CL == b.CL and a.CDi == b.CDi and a.CY == b.CY
    assert np.array_equal(a.Gamma, b.Gamma)
    assert np.array_equal(a.alpha_eff, b.alpha_eff)
    # and the panel section arrays are the wing's everywhere
    assert np.all(base.a_panel == _VLM_KW["a"])
    assert np.all(base.alpha_L0_panel == _VLM_KW["alpha_L0"])


def test_winglet_chord_scale_reaches_the_panel_chords_only():
    scaled = VLM(_WING, **_VLM_KW, winglet_chord_scale=2.0)
    base = VLM(_WING, **_VLM_KW)
    wl = base.is_winglet
    assert np.allclose(scaled.c[wl], 2.0 * base.c[wl])
    assert np.array_equal(scaled.c[~wl], base.c[~wl])
    with pytest.raises(ValueError, match="winglet_chord_scale"):
        VLM(_WING, **_VLM_KW, winglet_chord_scale=0.0)
    with pytest.raises(ValueError, match="winglet_a"):
        VLM(_WING, **_VLM_KW, winglet_a=0.0)


def _starboard_side_force(model: VLM, alpha: float) -> float:
    """Side force on the starboard tip device [N/(rho V^2) units aside]."""
    res = model.solve(alpha)
    star = res.is_winglet & (res.y > 0.0)
    return float(-1.225 * model.V * np.sum(res.Gamma[star] * res.lz[star]))


def test_a_cambered_zero_lift_angle_silently_toes_a_vertical_plate():
    """The trap endplate.py exists to close.

    ``theta = twist - alpha_L0`` builds the zero-lift angle into the panel
    NORMAL, so a nonzero alpha_L0 tilts a vertical fence exactly as it tilts
    a horizontal panel: the plate carries a side force it was never asked
    for. Sweeping the tip device's own alpha_L0 must move that side force
    monotonically — which is the same thing as saying it acts as a toe.
    """
    forces = [_starboard_side_force(
        VLM(_WING, **_VLM_KW, winglet_alpha_L0=np.deg2rad(al)), 0.1)
        for al in (-4.0, -2.0, 0.0, 2.0, 4.0)]
    assert all(b > a for a, b in zip(forces, forces[1:])), forces
    # and it is not a rounding-level effect
    assert abs(forces[-1] - forces[0]) > 5.0

    # a symmetric plate (alpha_L0 = 0) at a given toe is the SAME surface as
    # a plate whose camber supplies that angle: camber IS toe to a linear
    # method (endplate.py, "camber vs toe")
    toed = VLM(_WING, **_VLM_KW, winglet_alpha_L0=0.0, winglet_toe_deg=3.0)
    cambered = VLM(_WING, **_VLM_KW, winglet_alpha_L0=np.deg2rad(-3.0),
                   winglet_toe_deg=0.0)
    assert _starboard_side_force(toed, 0.1) == pytest.approx(
        _starboard_side_force(cambered, 0.1), rel=1e-12)


# ------------------------------------------------------------------- problem

def test_problem_shape_bounds_and_failure_contract():
    assert PROB.dim == 10
    assert PROB.bounds.shape == (10, 2)
    assert np.all(PROB.bounds[:, 1] > PROB.bounds[:, 0])

    for bad in (XMID[:-1], np.concatenate([XMID, [0.0]])):
        out = ep.evaluate_car_wing_endplate(bad, PROB)
        assert not out["feasible"] and len(out["g"]) == 4
    out = ep.evaluate_car_wing_endplate(_x(taper=99.0), PROB)
    assert not out["feasible"] and out["reason"] == "bounds violation"
    f, g = ep.fg_car_wing_endplate(_x(taper=99.0), PROB)
    assert f == ep.PENALTY and g == [ep.G_FAIL] * 4


def test_midpoint_evaluates_and_the_drag_decomposition_adds_up():
    out = ep.evaluate_car_wing_endplate(XMID, PROB)
    assert out["feasible"], out["reason"]
    assert out["CZ"] > 0.0 and out["CD"] > 0.0
    assert out["CD"] == pytest.approx(
        out["CDi"] + out["CDp"] + out["CD_endplate"] + out["cd0_struts"]
        + out["CD_junction"])
    assert out["CD_endplate"] == pytest.approx(
        out["CD_endplate_friction"] + out["CD_endplate_edges"])
    assert out["efficiency"] == pytest.approx(out["CZ"] / out["CD"])
    assert len(out["g"]) == 4
    assert out["frame"].startswith("mirrored")


def test_collapsed_chord_law_is_the_penalty_contract():
    prob = ep.CarWingEndplateProblem(chord_order=3)
    x = 0.5 * (prob.bounds[:, 0] + prob.bounds[:, 1])
    assert prob.dim == 13
    assert ep.evaluate_car_wing_endplate(x, prob)["feasible"]
    x[-3:] = [-0.5, -0.5, -0.5]
    out = ep.evaluate_car_wing_endplate(x, prob)
    assert not out["feasible"] and out["reason"].startswith("planform:")
    assert out["score"] == ep.PENALTY


# ------------------------------------------------ reach: attach to the car

def test_deck_height_must_leave_a_positive_gap_at_every_ride_height():
    with pytest.raises(ValueError, match="deck_height_m"):
        ep.CarWingEndplateProblem(deck_height_m=0.40)   # >= ride lower bound
    with pytest.raises(ValueError, match="deck_height_m"):
        ep.CarWingEndplateProblem(deck_height_m=0.0)


def test_the_wing_must_reach_the_deck_on_every_mount_layout():
    """The plate has to get from the wing down to the bodywork it bolts to.

    That used to be a demand only the plate-mounted layout made — under the
    centre-pylon layout the margin was a dormant +1.0 and ``reach_required``
    said so. There is no pylon layout any more: BOTH layouts take the load
    out through the plates, so the reach is unconditional, and the margin is
    the same signed number whichever station the sheets grip at.
    """
    prob = ep.CarWingEndplateProblem()        # the family's own default mount
    ride = 0.50
    reach = prob.reach_m(ride)
    assert reach == pytest.approx(ride - prob.deck_height_m)

    # h_ep exactly spans the gap -> the margin is exactly zero. The plate's
    # height is a LENGTH, so "exactly spans" is the reach itself — no semi-span
    # anywhere in it, which is the point: the deck does not move when the wing
    # gets wider (see the span sweep below)
    exact = reach
    out = ep.evaluate_car_wing_endplate(
        _x(h_m=exact, ride=ride), prob)
    assert out["reach_required"] is True
    assert out["g_reach"] == pytest.approx(0.0, abs=1e-12)
    assert out["endplate_tip_z_m"] == pytest.approx(reach)

    short = ep.evaluate_car_wing_endplate(_x(h_m=0.6 * exact, ride=ride),
                                          prob)
    tall = ep.evaluate_car_wing_endplate(_x(h_m=1.4 * exact, ride=ride),
                                         prob)
    assert short["g_reach"] < 0.0 < tall["g_reach"]
    # raising the wing lengthens the plate it needs
    higher = ep.evaluate_car_wing_endplate(_x(h_m=exact, ride=ride + 0.1),
                                           prob)
    assert higher["g_reach"] < 0.0
    # ...and WIDENING the wing does not. The plate reaches from the wing to a
    # piece of bodywork; how wide the wing is has nothing to say about it,
    # which is exactly what a height in metres means and a height as a
    # fraction of a (now free) semi-span cannot
    for b in (1.2, 2.0):
        wide = ep.evaluate_car_wing_endplate(_x(h_m=exact, ride=ride, b=b),
                                             prob)
        assert wide["g_reach"] == pytest.approx(out["g_reach"], abs=1e-12)
        assert wide["endplate_h_m"] == pytest.approx(exact)

    # ...and NO layout is excused it. A margin that is a live number on one
    # mount and a constant on another is a margin a reader cannot interpret,
    # so sweep the plate's height on each and demand the same three numbers.
    for mount in MOUNTS:
        other = ep.CarWingEndplateProblem(mount=mount)
        got = [ep.evaluate_car_wing_endplate(_x(h_m=h, ride=ride), other)
               for h in (0.6 * exact, exact, 1.4 * exact)]
        assert [o["reach_required"] for o in got] == [True] * 3
        assert [o["g_reach"] for o in got] == [
            pytest.approx(short["g_reach"]), pytest.approx(0.0, abs=1e-12),
            pytest.approx(tall["g_reach"])], mount


def test_the_mount_moves_the_wing_s_beam_and_the_sheet_drag_not_the_plate_s():
    """Which layout carries no longer changes the PLATE's own beam.

    Both layouts take the load out through the plates, so the plate is
    always a cantilever from the DECK (arm = ride − deck, root chord = the
    plate's far chord) — the hung-off-the-wing branch the pylon layout used
    is gone. What the mount decides now is WHERE ALONG THE SPAN the sheets
    grip: an interior grip is a much stiffer wing, bought with two extra
    sheets of wetted area and the two corners where they meet the wing.
    """
    x = _x(h_m=0.40, ride=0.55, tc=0.02)
    outs = ep.compare_mounts(x, PROB)
    assert set(outs) == set(MOUNTS) == {"tips", "inboard"}
    tips, inboard = outs["tips"], outs["inboard"]

    # the plate: same arm, same load, same deflection, both ways
    for o in (tips, inboard):
        assert o["endplate_arm_m"] == pytest.approx(o["reach_m"])
    assert inboard["endplate_side_load_N"] == pytest.approx(
        tips["endplate_side_load_N"], rel=1e-12)
    assert inboard["endplate_deflection_m"] == pytest.approx(
        tips["endplate_deflection_m"], rel=1e-12)

    # the WING's beam is what moved, and by a factor rather than a rounding
    assert inboard["M_max_Nm"] < tips["M_max_Nm"]
    assert inboard["deflection_m"] < 0.5 * tips["deflection_m"]

    # ...and the stiffness is paid for, in sheets and in corners, on the same
    # lifting system — so it is bought out of efficiency
    assert inboard["cd0_struts"] > 0.0 == tips["cd0_struts"]
    assert inboard["CD_junction"] > tips["CD_junction"]
    assert inboard["CD"] > tips["CD"]
    assert inboard["CZ"] == pytest.approx(tips["CZ"], rel=1e-12)
    assert inboard["efficiency"] < tips["efficiency"]

    # there is no pylon layout left to ask for, and asking says so
    with pytest.raises(ValueError, match="unknown mount"):
        ep.CarWingEndplateProblem(mount="centre")


# --------------------------------------------------------- the design trades

def test_section_family_moves_drag_and_stiffness_but_not_the_aerodynamics():
    x = _x(tc=0.03)
    outs = ep.compare_sections(x, PROB)
    assert set(outs) == set(ep.SECTIONS)
    flat, shaped = outs["flat"], outs["shaped"]
    # identical lifting system: the family is a drag/structure model only
    assert flat["CZ"] == pytest.approx(shaped["CZ"], rel=1e-12)
    assert flat["CDi"] == pytest.approx(shaped["CDi"], rel=1e-12)
    # the trade itself
    assert flat["CD_endplate"] > shaped["CD_endplate"]
    assert flat["endplate_I_m4"] > shaped["endplate_I_m4"]
    assert flat["endplate_deflection_m"] < shaped["endplate_deflection_m"]


def test_thin_plates_bend_and_the_constraint_notices():
    thick = ep.evaluate_car_wing_endplate(_x(tc=0.10), PROB)
    thin = ep.evaluate_car_wing_endplate(_x(tc=0.006), PROB)
    assert thin["endplate_deflection_m"] > thick["endplate_deflection_m"]
    # bending stiffness goes as t^3, so the deflections do too
    assert (thin["endplate_deflection_m"] / thick["endplate_deflection_m"]
            == pytest.approx((0.10 / 0.006) ** 3, rel=1e-6))
    # ...and the constraint is a real gate across the thickness box: 6 mm/m
    # of chord is 21 mm of sideways movement, against a 5 mm limit
    assert thin["g_endplate"] < 0.0 < thick["g_endplate"]


def test_the_plate_is_loaded_even_at_zero_toe():
    """A plate at zero toe still sits in the wing's tip flow, so it carries a
    side force — the structural check is not vacuous for an untoed plate."""
    out = ep.evaluate_car_wing_endplate(_x(toe=0.0), PROB)
    assert abs(out["endplate_side_load_toe_N"]) > 1.0
    assert out["endplate_side_load_yaw_N"] > 0.0
    assert out["endplate_side_load_N"] == pytest.approx(
        abs(out["endplate_side_load_toe_N"])
        + abs(out["endplate_side_load_yaw_N"]))


def test_toe_buys_downforce_and_costs_side_load_with_an_interior_optimum():
    """Positive toe loads the plate INBOARD, i.e. turns the flow outboard.
    That unloads the tip vortex and raises C_Z monotonically — but the
    plate's own induced drag and the side load it has to carry grow too, so
    downforce-per-drag peaks strictly inside the toe box."""
    toes = np.linspace(*PROB.ENDPLATE_TOE_BOUNDS_DEG, 13)
    outs = [ep.evaluate_car_wing_endplate(_x(toe=t), PROB) for t in toes]
    assert all(o["feasible"] for o in outs)
    cz = [o["CZ"] for o in outs]
    assert all(b > a for a, b in zip(cz, cz[1:])), cz
    load = [o["endplate_side_load_toe_N"] for o in outs]
    assert all(b < a for a, b in zip(load, load[1:])), load   # more inboard
    eff = [o["efficiency"] for o in outs]
    assert 0 < int(np.argmax(eff)) < len(eff) - 1, eff


def test_the_plate_has_its_own_linear_range_gate():
    """The offered toe box sits INSIDE the plate's linear range (the worst
    panel reaches ~7 deg at +/-6 deg of toe, once the wing's sidewash is
    added), and widening the box past the gate fails loudly rather than
    buying downforce from a stalled plate."""
    out = ep.evaluate_car_wing_endplate(_x(alpha=12.0, toe=6.0, h_m=0.40,
                                          ride=0.55), PROB)
    assert out["feasible"]
    assert out["endplate_alpha_deg"] < PROB.endplate_alpha_max_deg
    assert out["endplate_alpha_deg"] > abs(6.0)      # sidewash adds to toe

    tight = ep.CarWingEndplateProblem(endplate_alpha_max_deg=3.0)
    out = ep.evaluate_car_wing_endplate(_x(alpha=12.0, toe=6.0, h_m=0.40,
                                          ride=0.55), tight)
    assert not out["feasible"]
    assert "linear-range gate" in out["reason"]
    assert out["score"] == ep.PENALTY


def test_junction_drag_is_charged_on_the_PLATE_s_own_section():
    """The wing/endplate corner is the plate's corner: Hoerner's correlation
    takes the junction member's thickness and chord, which here are the
    endplate's, not the wing's."""
    thin = ep.evaluate_car_wing_endplate(_x(tc=0.06), PROB)
    thick = ep.evaluate_car_wing_endplate(_x(tc=0.16), PROB)
    assert thick["CD_junction"] > thin["CD_junction"]
    expect = junction.junction_cd(0.16, thick["endplate_chord_m"], PROB.S,
                                  radius=0.0, n_junctions=2)
    assert thick["CD_junction"] == pytest.approx(expect)   # ends: no pylons
    off = ep.CarWingEndplateProblem(junction_drag=False)
    off_out = ep.evaluate_car_wing_endplate(_x(tc=0.16), off)
    assert off_out["CD_junction"] == 0.0


def test_a_bigger_plate_costs_drag_and_buys_span_efficiency():
    small = ep.evaluate_car_wing_endplate(_x(h_m=0.16), PROB)
    big = ep.evaluate_car_wing_endplate(_x(h_m=0.44), PROB)
    assert big["CD_endplate"] > small["CD_endplate"]
    assert big["endplate_Swet_m2"] > small["endplate_Swet_m2"]
    assert big["e"] > small["e"]          # nonplanar benefit, referenced to b


def test_ground_proximity_still_raises_downforce():
    low = ep.evaluate_car_wing_endplate(_x(ride=0.32, h_m=0.08), PROB)
    high = ep.evaluate_car_wing_endplate(_x(ride=0.68, h_m=0.08), PROB)
    assert low["CZ"] > high["CZ"]
    assert low["CDi"] < high["CDi"]


def test_track_is_a_hard_boundary_not_a_silent_answer():
    """A plate driven into the track fails loudly (the image system would
    overlap the real one), rather than returning a plausible number."""
    out = ep.evaluate_car_wing_endplate(_x(ride=0.30, h_m=0.32), PROB)
    assert not out["feasible"]
    assert "ground" in out["reason"] and out["score"] == ep.PENALTY


# ------------------------------------------------------------------ registry

def test_registered_problem_and_its_chord_twin_build_and_run():
    """The BASE name is the free-area problem: 11-D, and the three margins
    are the part's own (the beam, the plate, the reach). Nothing is budgeted,
    so a run that comes back near a drag ceiling came back near one the user
    asked for."""
    from aerobo import api

    three = ("wing deflection margin", "endplate deflection margin",
             "endplate reach margin")
    for name, dim in (("car rear wing + endplates", 11),
                      ("car rear wing + endplates + free chord law", 14)):
        spec = api.PROBLEM_SPECS[name]
        assert spec.is_constrained and spec.n_constraints == 3
        assert spec.constraint_labels == three
        built = spec.build({}, {}, None)
        assert built.dim == dim == len(built.param_labels)
        # the area is searched, immediately AHEAD of the span, wherever the
        # chord coefficients push the size block to
        lbl = list(built.param_labels)
        assert lbl.index("S_m2") == lbl.index("b_m") - 1
        assert built.problem.area_free is True
        # the LITERAL as well as the constant: comparing api's constant to
        # itself cannot fail, and this whole file's _free() helper is built
        # FROM that constant, so without a literal here the entire file
        # follows the default wherever it moves. Measured: changing
        # CAR_DEFAULT_OBJECTIVE to "downforce" left every test here green.
        assert built.problem.objective == api.CAR_DEFAULT_OBJECTIVE \
            == "efficiency"
        assert built.problem.CD_budget is None
        assert built.problem.drag_budget_n is None
        assert built.problem.constraint_labels == three
        f, g = built.callable(0.5 * (built.bounds[:, 0] + built.bounds[:, 1]))
        assert np.isfinite(f) and len(g) == 3


def test_the_free_area_twins_are_gone_the_base_family_is_the_free_one():
    """The area is not a question any more, so it has no separate name and no
    switch. Six car entries became twelve when the PLATE's cant and root blend
    each grew a searched twin — a dimension change is a family, not a flag —
    and the point of this test survives that: not one of the twelve answers
    "how big is the wing" by its name."""
    import gui.nice_app as v1
    from aerobo import api

    for dead in ("car rear wing (free area)",
                 "car rear wing (two-element, free area)",
                 "car rear wing + endplates (free area)"):
        assert dead not in api.PROBLEM_SPECS
    cars = [n for n in api.PROBLEM_SPECS if n.startswith("car rear wing")]
    assert len(cars) == 12, cars
    # every one of them SEARCHES its area — which is the property the count
    # was ever standing in for
    for name in cars:
        assert "S_m2" in api.PROBLEM_SPECS[name].default_bounds

    ch = v1.choices_from_problem("car rear wing + endplates")
    assert "car_free_area" not in ch
    assert v1.derive_problem(ch) == ("car rear wing + endplates", [])


def test_flags_reach_the_problem_and_the_gui_round_trips():
    import gui.nice_app as v1
    from aerobo import api

    built = api.PROBLEM_SPECS["car rear wing + endplates"].build(
        {}, {"mount": "inboard", "section": "flat", "deck_height_m": 0.2},
        None)
    assert built.problem.mount == "inboard"
    assert built.problem.section == "flat"
    assert built.problem.deck_height_m == pytest.approx(0.2)

    ch = v1.choices_from_problem("car rear wing + endplates")
    assert ch["medium"] == "track" and ch["car_endplates"] is True
    assert v1.derive_problem(ch) == ("car rear wing + endplates", [])
    # the card opens on the family's own default layout, and the menu offers
    # the two plate-borne ones and nothing else
    assert v1.car_flags(ch) == {"mount": "tips", "section": "shaped"}
    # the literal on one side, so that renaming BOTH the menu and the engine
    # dict together still trips the stale-menu guard rather than passing
    assert set(v1.car_mount_labels()) == set(MOUNTS) == {"tips", "inboard"}
    # a plain car wing sends no section (it has no designed plate)
    assert "section" not in v1.car_flags(
        v1.choices_from_problem("car rear wing"))


def test_a_size_band_is_a_design_box_row_and_it_reaches_the_solver():
    """The b_m / S_m2 bands are rows of the box, not flags.

    Typed as a flag they moved the search while the box on screen showed the
    family default, so the old spellings are refused outright. And a box row
    WIDER than the family's own band used to leave ``prob.bounds`` put, which
    made every draw above it a "bounds violation" scoring the refusal
    sentinel — so this asserts a draw at the top of a widened band actually
    flies.
    """
    from aerobo import api

    name = "car rear wing + endplates"
    for dead in ("span_min_m", "span_max_m", "area_min_m2", "area_max_m2"):
        with pytest.raises(KeyError, match="does not honour"):
            api.check_flags(name, {dead: 1.0})

    spec = api.PROBLEM_SPECS[name]
    default = spec.build({}, {}, None)
    built = spec.build({}, {}, {"b_m": (1.2, 3.0), "S_m2": (0.20, 0.50)})
    lbl = list(built.param_labels)
    i_b, i_s = lbl.index("b_m"), lbl.index("S_m2")
    assert built.bounds[i_b].tolist() == [1.2, 3.0]
    assert built.bounds[i_s].tolist() == [0.20, 0.50]
    assert tuple(built.problem.span_bounds_m) == (1.2, 3.0)
    assert tuple(built.problem.area_bounds_m2) == (0.20, 0.50)
    assert default.bounds[i_b][1] < 3.0        # genuinely wider than default

    x = built.bounds.mean(axis=1)
    x[i_b] = 2.9                               # above the family's own band
    out = built.evaluate(x)
    assert out["feasible"], out["reason"]
    assert out["b_m"] == pytest.approx(2.9)
    assert out["score"] > 0.0
    f, _ = built.callable(x)
    assert np.isfinite(f) and f > 0.0

    # ...and the clip is one pass: feeding the built box back in as the
    # override must not move it (gui/v3/relax.py breaks otherwise)
    again = spec.build({}, {}, {"b_m": tuple(built.bounds[i_b]),
                                "S_m2": tuple(built.bounds[i_s])})
    assert np.array_equal(again.bounds, built.bounds)


def test_mission_kwargs_are_refused_rather_than_dropped():
    from aerobo import api

    with pytest.raises(ValueError, match="no mission spec"):
        api.PROBLEM_SPECS["car rear wing + endplates"].build(
            {"W_N": 5000.0}, {}, None)


def test_geometry_helpers_used_by_the_problem_are_consistent():
    """The plate's reach is the winglet path's own tip height, not a cosine
    — the same rule the blended winglet is span-capped by."""
    h = 0.3
    assert geometry.winglet_tip_height(h, 90.0, 0.0) == pytest.approx(h)
    assert geometry.winglet_tip_height(h, 90.0, 0.5) < h


# --------------------------------------------------------------------------
# the reference AREA as a design variable — which is now every registered run
# (the dataclass still fixes the area when no band is given: the SOLVER stays
# general, only the registry always frees it)
# --------------------------------------------------------------------------

def _free(**kw) -> ep.CarWingEndplateProblem:
    """The area-free problem, opened the way api's builder opens it.

    Kept honest by ``test_the_helper_opens_it_the_way_the_builder_does``
    below, so a change of opening configuration cannot leave this helper
    testing a problem the shell never builds.
    """
    from aerobo.api import CAR_DEFAULT_OBJECTIVE
    from aerobo.carwing import AREA_BOUNDS_M2

    base = dict(area_bounds_m2=AREA_BOUNDS_M2,
                objective=CAR_DEFAULT_OBJECTIVE, CD_budget=None)
    return ep.CarWingEndplateProblem(**{**base, **kw})


def test_the_helper_opens_it_the_way_the_builder_does():
    from aerobo import api

    got = api.PROBLEM_SPECS["car rear wing + endplates"].build({}, {}, None)
    ref = _free()
    for field in ("objective", "CD_budget", "drag_budget_n",
                  "downforce_min_n", "mount", "section", "area_free", "dim"):
        assert getattr(got.problem, field) == getattr(ref, field), field
    assert tuple(got.problem.area_bounds_m2) == tuple(ref.area_bounds_m2)
    assert np.array_equal(got.bounds, ref.bounds)


def test_the_area_row_goes_ahead_of_the_span_so_the_span_keeps_its_slot():
    """The size block is [S, b]. Put the area AFTER the span instead and the
    span silently reads the area while every shape test still passes — so
    this asserts the VALUES the two inverses return, not the widths."""
    from aerobo.carwing import s_from_x, span_from_x

    fixed, free = ep.CarWingEndplateProblem(), _free()
    assert free.dim == fixed.dim + 1
    assert free.area_free and not fixed.area_free
    # the two rows are distinguishable: an area band in m^2 and a span band
    # in m that do not overlap, so a swapped pair could not pass numerically
    assert free.bounds[-2].tolist() == list(map(float, (0.10, 0.48)))
    # the old line here read `free.bounds[-1].tolist() == list(free.bounds[-1])`
    # — row == row, true for any span band whatsoever. It could not see the
    # span band widened until it OVERLAPS the area band, which is the one
    # thing that would make the two rows indistinguishable by value and let a
    # swapped pair pass. So pin the band itself (carwing.SPAN_BOUNDS_M).
    assert free.bounds[-1].tolist() == list(map(float, (1.2, 2.0)))
    assert fixed.bounds[-1].tolist() == free.bounds[-1].tolist()
    assert free.bounds[-2][1] < free.bounds[-1][0]     # disjoint, by value
    x = free.bounds.mean(axis=1)
    assert s_from_x(x, free.chord_order) == pytest.approx(0.29)
    assert span_from_x(x, free.chord_order) == pytest.approx(1.6)
    # ...and the span reads the SAME slot with the area fixed
    assert span_from_x(XMID, fixed.chord_order) == pytest.approx(1.6)


def test_the_area_is_flown_not_merely_carried():
    """Every coefficient, the plate's drag and the reported geometry follow
    the candidate's own area — reading prob.S anywhere would reference them
    to a wing nobody flew."""
    prob = _free(objective="downforce")    # so the score IS the force below
    x = prob.bounds.mean(axis=1)
    small, big = x.copy(), x.copy()
    small[-2], big[-2] = 0.15, 0.45
    a = ep.evaluate_car_wing_endplate(small, prob)
    b = ep.evaluate_car_wing_endplate(big, prob)
    assert a["feasible"] and b["feasible"]
    assert a["S_m2"] == pytest.approx(0.15)
    assert b["S_m2"] == pytest.approx(0.45)
    assert a["area_free"] is True
    # the aspect ratio is b^2/S on the WING's span, so a smaller area is a
    # higher AR — and the plate's parasite drag, a coefficient on the same
    # area, rises when the area it is divided by falls
    assert a["AR"] > b["AR"]
    assert a["CD_endplate"] > b["CD_endplate"]
    # the score is the FORCE, and the force is q S CZ at this candidate
    assert a["score"] == pytest.approx(a["q_Pa"] * a["S_m2"] * a["CZ"])
    assert b["score"] > a["score"]          # more area, more downforce


def test_a_coefficient_objective_is_refused_against_a_free_area():
    with pytest.raises(ValueError, match="referenced to the very area"):
        _free(objective="cz")
    with pytest.raises(ValueError, match="unknown objective"):
        ep.CarWingEndplateProblem(objective="drag")
    # ...and cz is perfectly fine with the area fixed: it is the published one
    assert ep.CarWingEndplateProblem().objective == "cz"


def test_the_aspect_ratio_gate_is_checked_on_the_wing_span_not_the_width():
    """A blended plate reaches outboard and the wing pays for it out of the
    span band, so the AR that must stay inside sizing.AR_LIMITS is what is
    LEFT — refused in contract, never raised."""
    from aerobo.sizing import AR_LIMITS

    prob = _free(area_bounds_m2=(0.10, 1.20))     # a band wide enough to fail
    x = prob.bounds.mean(axis=1)
    x[-2], x[-1] = 1.15, 1.25                     # AR = 1.36, far below 3
    out = ep.evaluate_car_wing_endplate(x, prob)
    assert not out["feasible"] and "aspect ratio" in out["reason"]
    assert len(out["g"]) == prob.n_constraints
    assert AR_LIMITS[0] > 1.25**2 / 1.15
    # the same vector with the DEFAULT band is inside the gate by
    # construction — which is what the derived band is for
    ok = _free()
    y = ok.bounds.mean(axis=1)
    out2 = ep.evaluate_car_wing_endplate(y, ok)
    assert out2["feasible"]


def test_the_margin_vector_follows_the_declaration_not_the_published_four():
    """A budget switched off shortens the vector rather than reporting a
    satisfied placeholder — and the failure vector follows it, because a
    penalty at the wrong width is a shape error inside the optimiser."""
    three = ("wing deflection margin", "endplate deflection margin",
             "endplate reach margin")
    cases = [
        (ep.CarWingEndplateProblem(),
         ("drag budget margin",) + three),
        (ep.CarWingEndplateProblem(CD_budget=None),
         three),
        # the registered configuration: nothing budgeted, so the vector is
        # the part's own three margins and no placeholder
        (_free(), three),
        (_free(downforce_min_n=300.0), three + ("downforce floor margin",)),
        (_free(drag_budget_n=200.0), ("drag force margin",) + three),
        (ep.CarWingEndplateProblem(drag_budget_n=200.0),
         ("drag budget margin", "drag force margin") + three),
    ]
    for prob, labels in cases:
        assert prob.constraint_labels == labels
        assert prob.n_constraints == len(labels)
        x = prob.bounds.mean(axis=1)
        out = ep.evaluate_car_wing_endplate(x, prob)
        assert len(out["g"]) == len(labels)
        assert out["constraint_labels"] == list(labels)
        f, g = ep.fg_car_wing_endplate(x, prob)
        assert len(g) == len(labels)
        # ...and the FAILURE path, which is the one a shape error hides in
        bad = np.full_like(x, 1e6)
        fb, gb = ep.fg_car_wing_endplate(bad, prob)
        assert fb == ep.PENALTY and len(gb) == len(labels)


def test_the_dataclass_default_still_fixes_its_area_and_scores_a_coefficient():
    """The SOLVER stayed general when the registry stopped offering a fixed
    area: constructed with no area band the problem still pins its own S,
    still maximises C_Z, and still returns its four margins in order — which
    is what carsection.py and the split comparison rely on."""
    prob = ep.CarWingEndplateProblem()
    out = ep.evaluate_car_wing_endplate(XMID, prob)
    assert out["score"] == out["CZ"]              # cz is still the default
    assert out["objective"] == "cz"
    assert out["g"] == [out["g_drag"], out["g_deflection"],
                        out["g_endplate"], out["g_reach"]]
    assert out["g_drag_force"] is None and out["g_downforce"] is None
    assert out["S_m2"] == pytest.approx(prob.S) and out["area_free"] is False


def test_the_downforce_floor_binds_and_is_signed_the_right_way():
    prob = _free(objective="efficiency", downforce_min_n=300.0)
    x = prob.bounds.mean(axis=1)
    out = ep.evaluate_car_wing_endplate(x, prob)
    assert out["score"] == pytest.approx(out["CZ"] / out["CD"])
    assert out["g_downforce"] == pytest.approx(
        out["downforce_N"] / 300.0 - 1.0)
    # a wing making less than the floor is INFEASIBLE, not merely worse
    tiny = x.copy()
    tiny[-2] = 0.10
    low = ep.evaluate_car_wing_endplate(tiny, prob)
    assert low["feasible"] and low["downforce_N"] < 300.0
    assert low["g_downforce"] < 0.0


def test_nothing_is_budgeted_until_a_ceiling_is_typed_in_newtons():
    """The published allowance is still a number this family can quote — and
    nothing reaches for it.

    It used to be the DEFAULT, restated per family (96.3 N here against the
    tip-device model's 81.5, because designed plates cost drag it never
    charged) and handed to a maximiser, which spends an allowance rather
    than respecting it. So a registered run now carries no drag margin at
    all, there is no menu that could re-add one, and a ceiling is a number
    the user types — in newtons, on the force the candidate actually flies.
    """
    import gui.nice_app as v1
    from aerobo import api, carwing

    P = ep.CarWingEndplateProblem
    mine = carwing.published_drag_budget_n(P.CD_budget, P.S, P.V, P.rho)
    theirs = carwing.published_drag_budget_n()
    assert mine > theirs
    assert mine / theirs == pytest.approx(P.CD_budget / 0.11)
    assert not hasattr(v1, "_car_drag_budget_default")

    name = "car rear wing + endplates"
    ch = v1.choices_from_problem(name)
    assert ch["car_drag_budget_n"] is None       # a blank field, not a menu
    assert "drag_budget_n" not in v1.car_flags(ch)
    plain = api.PROBLEM_SPECS[name].build({}, v1.car_flags(ch), None)
    assert plain.problem.drag_budget_n is None
    assert plain.problem.CD_budget is None
    assert not any("drag" in s for s in plain.problem.constraint_labels)

    # a CEILING is a limit the user states, and it makes its own margin off
    # the drag FORCE — the only allowance that means anything at a free area
    ch["car_drag_budget_n"] = 120.0
    flags = v1.car_flags(ch)
    assert flags["drag_budget_n"] == pytest.approx(120.0)
    capped = api.PROBLEM_SPECS[name].build({}, flags, None)
    assert "drag force margin" in capped.problem.constraint_labels
    x = capped.bounds.mean(axis=1)
    out = capped.evaluate(x)
    assert out["g_drag_force"] == pytest.approx(1.0 - out["drag_N"] / 120.0)
    assert out["g_drag_force"] > 0.0
    # ...and it BITES: a ceiling under what this candidate costs is refused,
    # not quietly widened
    tight = api.PROBLEM_SPECS[name].build(
        {}, {**flags, "drag_budget_n": 0.5 * float(out["drag_N"])}, None)
    assert tight.evaluate(x)["g_drag_force"] == pytest.approx(-1.0)
