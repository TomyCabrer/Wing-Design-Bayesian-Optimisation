"""The free flight state composes with EVERY air family.

Speed and altitude are not geometry: they set the flow state (ISA density and
Sutherland viscosity) and, through the design WEIGHT, the trim target
CL = W/(q S) that every family in this package already flies at. That makes
them a MODIFIER (api.MODIFIERS) exactly as the chord law is one — every air
family gains a variant carrying two extra rows (api.FLIGHT_TWINS) instead of
"free flight state" being a solver family that resets the winglet, the
section and the tail the moment it is switched on.

What has to hold for that to be honest, and is checked here:

1. OFF IS THE PUBLISHED PROBLEM. Same bounds, same dimension, same number.
2. AT THE FAMILY'S OWN OPERATING POINT the variant returns the base problem's
   score EXACTLY. The design weight is derived from the base problem's own
   CL_target (mission.weight_for), so switching the modifier on and typing
   the published speed back in cannot move the answer.
3. THE STATE IS ACTUALLY FLOWN: a different (V, altitude) must change the
   trim target and the score — not decorate the vector.
4. THE BLOCK SITS BETWEEN the family and the chord coefficients, so the two
   modifiers compose without either one moving the other's rows.
5. THE MISSION CARD STOPS OFFERING V AND ALTITUDE once they are design
   variables (spec.mission_fields), so a typed-in cruise speed can never sit
   beside a speed the optimiser is choosing.
"""

from __future__ import annotations

import numpy as np
import pytest

from aerobo import (aircraft, api, geometry, mission, objective, tail, tandem,
                    wing_airfoil, wingtail)

ORDER = 3


def _mid(prob) -> np.ndarray:
    b = prob.bounds
    return 0.5 * (b[:, 0] + b[:, 1])


def _state_of(prob) -> tuple[float, float]:
    """The (V, altitude) a family flies with the modifier OFF."""
    V = getattr(prob, "V", None)
    if V is None:                     # aircraft carries a MissionSpec instead
        return float(prob.mission.V), float(prob.mission.altitude_m)
    return float(V), 0.0


#: Families whose evaluation is NOT bit-reproducible run to run. Measured
#: (60 identical calls to evaluate_wing_tail return two distinct doubles,
#: 33.22697577315232 / ...33): the nonplanar wing+tail path factorises and
#: back-substitutes through threaded BLAS, whose reduction order depends on
#: runtime scheduling, so the last ulp moves. Pre-existing and unrelated to
#: the modifier — the identities below are therefore asserted to 1e-12
#: RELATIVE for these two, and exactly everywhere else.
_ULP_NOISY = {"wing + tail + winglet (nonplanar)", "wing + tail, free height"}


def _same(name: str, got: float, want: float) -> bool:
    if name in _ULP_NOISY:
        return got == pytest.approx(want, rel=1e-12)
    return got == want


#: (name, factory taking flight_free/chord_order, evaluate, chord laws)
FAMILIES = [
    ("winglet",
     lambda **kw: objective.Problem(mode="winglet", **kw),
     objective.evaluate, 1),
    ("winglet_capped + t/c",
     lambda **kw: objective.Problem(mode="winglet_capped_tc", **kw),
     objective.evaluate, 1),
    ("wing t/c + sweep",
     lambda **kw: objective.Problem(mode="tier_a_plus", **kw),
     objective.evaluate, 1),
    ("tail", tail.TailProblem, tail.evaluate_tail, 1),
    ("tail (fixed arm)",
     lambda **kw: tail.TailProblem(l_t_fixed=5.5, **kw),
     tail.evaluate_tail, 1),
    ("wing + tail + winglet (nonplanar)",
     lambda **kw: wingtail.WingTailProblem(winglet=True, capped=True, **kw),
     wingtail.evaluate_wing_tail, 1),
    ("wing + tail, free height",
     lambda **kw: wingtail.WingTailProblem(free_height=True, tc_free=True,
                                           **kw),
     wingtail.evaluate_wing_tail, 1),
    ("tandem", tandem.TandemProblem, tandem.evaluate_tandem, 2),
    ("wing+airfoil (coupled)", wing_airfoil.WingAirfoilProblem,
     wing_airfoil.evaluate_wing_airfoil, 1),
    ("free planform (aircraft)", aircraft.AircraftProblem,
     aircraft.evaluate_aircraft, 1),
]

IDS = [f[0] for f in FAMILIES]


# --------------------------------------------------------------- 1. identity

@pytest.mark.parametrize("name, factory, evaluate, laws", FAMILIES, ids=IDS)
def test_flight_free_off_is_the_published_problem(name, factory, evaluate,
                                                  laws):
    base, off = factory(), factory(flight_free=False)
    assert off.dim == base.dim
    assert np.array_equal(off.bounds, base.bounds)
    x = _mid(base)
    assert _same(name, evaluate(x, off)["score"], evaluate(x, base)["score"])


@pytest.mark.parametrize("name, factory, evaluate, laws", FAMILIES, ids=IDS)
def test_the_flight_rows_are_two_and_leave_the_others_alone(name, factory,
                                                            evaluate, laws):
    base, free = factory(), factory(flight_free=True)
    assert free.dim == base.dim + 2
    assert np.array_equal(free.bounds[:base.dim], base.bounds)
    v_lo, v_hi = free.bounds[base.dim]
    assert v_lo < _state_of(base)[0] < v_hi      # the design point is INTERIOR
    assert tuple(free.bounds[base.dim + 1]) == geometry.MISSION_ALT_BOUNDS_M


@pytest.mark.parametrize("name, factory, evaluate, laws", FAMILIES, ids=IDS)
def test_the_published_state_reproduces_the_base_problem_exactly(
        name, factory, evaluate, laws):
    """The load-bearing guarantee: a variant flown at its family's own speed
    and sea level IS the fixed-state problem."""
    base, free = factory(), factory(flight_free=True)
    x = _mid(base)
    out_base = evaluate(x, base)
    out_free = evaluate(np.concatenate([x, _state_of(base)]), free)
    assert out_base["feasible"] and out_free["feasible"]
    assert _same(name, out_free["score"], out_base["score"])
    # the trim target itself is pure arithmetic and IS exact everywhere
    if "CL_target" in out_base:
        assert out_free["CL_target"] == out_base["CL_target"]


# ------------------------------------------------------- 2. actually flown

@pytest.mark.parametrize("name, factory, evaluate, laws", FAMILIES, ids=IDS)
def test_a_different_state_re_trims_and_moves_the_score(name, factory,
                                                        evaluate, laws):
    """Not just accepted — used. Flying faster must lower the trim CL
    (CL = W/(q S) with W fixed) and move the score."""
    free = factory(flight_free=True)
    base = factory()
    x = _mid(base)
    V0, h0 = _state_of(base)
    V1 = 0.5 * (V0 + float(free.bounds[base.dim][1]))     # faster, in the box
    slow = evaluate(np.concatenate([x, [V0, h0]]), free)
    fast = evaluate(np.concatenate([x, [V1, h0]]), free)
    assert slow["feasible"] and fast["feasible"]
    assert fast["score"] != slow["score"]
    if "CL_target" in slow and "CL_target" in fast:
        assert fast["CL_target"] < slow["CL_target"]
    # the flown state is reported back, so a result can never be read at the
    # wrong operating point
    assert fast["V"] == pytest.approx(V1)
    assert fast["altitude_m"] == pytest.approx(h0)


@pytest.mark.parametrize("name, factory, evaluate, laws", FAMILIES, ids=IDS)
def test_altitude_thins_the_air_and_raises_the_trim_cl(name, factory,
                                                       evaluate, laws):
    free = factory(flight_free=True)
    base = factory()
    x = _mid(base)
    V0, _ = _state_of(base)
    low = evaluate(np.concatenate([x, [V0, 0.0]]), free)
    high = evaluate(np.concatenate([x, [V0, 2000.0]]), free)
    assert low["feasible"]
    if not high["feasible"]:
        # a thinner-air trim can leave the polar's valid range: that is the
        # documented stall/extrapolation proxy, not a silent number
        assert high["score"] == objective.PENALTY
        return
    assert high["rho"] < low["rho"]
    if "CL_target" in high:      # families that report their trim target
        assert high["CL_target"] > low["CL_target"]


# ------------------------------------------------- 3. composes with chord

@pytest.mark.parametrize("name, factory, evaluate, laws", FAMILIES, ids=IDS)
def test_both_modifiers_stack_family_then_flight_then_chord(name, factory,
                                                            evaluate, laws):
    base = factory()
    both = factory(flight_free=True, chord_order=ORDER)
    assert both.dim == base.dim + 2 + laws * ORDER
    assert np.array_equal(both.bounds[:base.dim], base.bounds)
    assert np.array_equal(both.bounds[base.dim + 2:],
                          np.vstack([geometry.chord_bounds(ORDER)] * laws))
    # Adding the flight block at the family's own state changes NOTHING —
    # compared against the chord-only variant, which isolates this modifier
    # (the chord law's own order-0-vs-zero-coefficients identity is gated in
    # test_chord_everywhere.py, and is exact only for the families listed
    # there; here the two sides carry the identical chord law).
    chord_only = factory(chord_order=ORDER)
    x = _mid(base)
    zeros = np.zeros(laws * ORDER)
    out = evaluate(np.concatenate([x, _state_of(base), zeros]), both)
    ref = evaluate(np.concatenate([x, zeros]), chord_only)
    assert _same(name, out["score"], ref["score"])


def test_flight_from_x_is_the_inverse_of_the_stacking_rule():
    x = np.arange(10, dtype=float)
    assert geometry.flight_from_x(x, False, 0) is None
    assert geometry.flight_from_x(x, True, 0) == (8.0, 9.0)
    assert geometry.flight_from_x(x, True, 3) == (5.0, 6.0)
    with pytest.raises(ValueError):
        geometry.flight_from_x(np.zeros(2), True, 3)


def test_the_design_weight_round_trips_the_published_trim_target():
    """mission.weight_for / flight_state are inverse at the design point."""
    W = mission.weight_for(0.5, 14.6, 10.0)
    spec = mission.MissionSpec(W_N=W, V=14.6, altitude_m=0.0)
    state = mission.flight_state(spec, 14.6, 0.0, 10.0)
    assert state["CL_target"] == 0.5
    assert state["rho"] == mission.isa_density(0.0)


def test_a_water_mission_is_refused_rather_than_flown_in_air():
    spec = mission.MissionSpec(W_N=6000.0, V=10.0, depth_m=0.5, medium="water")
    with pytest.raises(ValueError, match="air-only"):
        mission.flight_state(spec, 10.0, 0.0, 0.144)


# ------------------------------------------------------------ 4. registry

def test_every_air_family_has_a_flight_variant_and_the_others_say_why():
    """The product is complete: every family that can carry the modifier has
    its variant registered, and the four that cannot are the documented
    ones."""
    no_flight = {name for name in api.PROBLEM_SPECS
                 if not api.modifiers_of(name)
                 and api.add_modifier(name, "flight") is None}
    assert no_flight == {
        "hydrofoil", "hydrofoil + winglet",        # speed + depth already free
        *api.HYDRO_TAIL_VARIANTS,                  # ...same family, + elevator
        *api.HYDRO_SECTION_VARIANTS,               # ...and with a CST section
        # ...and every twin of those that flies a CHOSEN section: same
        # families, same reason
        *api.CHOSEN_SECTION_TWINS.values(),
        # the track's speed is a CONDITION and not a design variable —
        # designing the wing's reference area does not change that, and nor
        # does putting a SLOT in the section: the flap's deflection is a
        # design variable, the track's speed is still a condition
        "car rear wing",
        "car rear wing (two-element)",
        "car rear wing + endplates",
        # ...and the endplate's own freedoms, which are three more BASE
        # families of the same vehicle: a cant and a blend are shapes of the
        # plate, and neither turns a wing bolted to a car into something
        # with a trim target.
        "car rear wing + endplates [free cant]",
        "car rear wing + endplates [free blend]",
        "car rear wing + endplates [free cant, free blend]",
        "airfoil (section)",                       # 2-D, no trim target
    }
    # ...and "mission wing" is not a family at all any more: it IS the trim
    # wing carrying this modifier, under the name it shipped with
    assert api.modifiers_of("mission wing") == frozenset({"flight"})
    assert api.base_of("mission wing") == "trim wing"


def test_the_flight_variant_takes_v_and_altitude_off_the_mission_card():
    for base, name in api.FLIGHT_TWINS.items():
        spec = api.PROBLEM_SPECS[name]
        assert "V" not in spec.mission_fields, name
        assert "altitude_m" not in spec.mission_fields, name
        if api.PROBLEM_SPECS[base].uses_mission:
            # the design WEIGHT stays: that is what the modifier trims against
            assert "W_N" in spec.mission_fields, name


@pytest.mark.parametrize("name", sorted(api.FLIGHT_TWINS.values()))
def test_every_flight_variant_builds_and_carries_the_two_rows(name):
    spec = api.PROBLEM_SPECS[name]
    if spec.slow:                     # live XFOIL: built, not evaluated, here
        built = spec.build({}, {}, None)
        assert built.dim == len(built.param_labels)
        assert "V_ms" in built.param_labels
        return
    built = spec.build({}, {}, None)
    assert built.dim == len(built.param_labels) == len(spec.param_labels)
    assert "V_ms" in spec.param_labels and "altitude_m" in spec.param_labels
    out = built.evaluate(0.5 * (built.bounds[:, 0] + built.bounds[:, 1]))
    assert isinstance(out, dict) and "score" in out


def test_the_gui_treats_the_flight_state_as_a_modifier_not_a_family():
    """Switching the flight state on must not reset the winglet, the section
    or the tail — the complaint that started this work."""
    import gui.nice_app as v1

    assert "flight" not in v1.SPECIAL_KEYS
    ch = dict(v1.BUILDER_DEFAULTS, winglets="capped", winglet_type="blended",
              flight="free", chord="free")
    name, notes = v1.derive_problem(ch)
    assert name == ("winglet, blended (span-capped) + free flight state "
                    "+ free chord law")
    assert notes == []
    # ...and the whole registry still inverts through the builder mapping
    for problem in api.FLIGHT_TWINS.values():
        assert v1.derive_problem(v1.choices_from_problem(problem))[0] \
            == problem, problem


def test_the_water_and_track_families_keep_the_control_greyed_out():
    import gui.nice_app as v1

    for choices in (dict(v1.BUILDER_DEFAULTS, medium="water"),
                    dict(v1.BUILDER_DEFAULTS, medium="track")):
        assert not v1.flight_available(choices)
        name, notes = v1.derive_problem(dict(choices, flight="free"))
        assert api.modifiers_of(name) == frozenset()
        assert any("free flight state ignored" in n for n in notes)
