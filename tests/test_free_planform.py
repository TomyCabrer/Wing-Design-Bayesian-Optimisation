"""The SIZE modifier: span and area free, and paid for.

Freeing the span in a pure-aero model is a trap (aircraft.py derives it):
D_i ~ 1/b^2 at fixed lift, so the optimiser slams the box. This modifier is
therefore not "two more rows" — switching it on changes three things at once
and adds a constraint, and this file gates each of them:

1. THE ROWS: b and S appended after the family block and before the flight and
   chord blocks, boxed around the family's OWN size.
2. THE WEIGHT: the wing weighs what its size implies (Raymer, closed as a
   fixed point) and the trim target becomes W_total/(q S).
3. THE OBJECTIVE: payload L/D = W_fixed/D, not L/D — with W varying, "maximise
   L/D" would reward a heavier wing that lifts better.
4. THE CONSTRAINT: a root-bending stress margin, in the log form, appended to
   whatever margins the family already had.
5. THE REGISTRY: is_constrained / n_constraints / constraint_labels move with
   it, the chosen-size FLAGS are dropped (span and area cannot be both), and
   the GUI reaches every combination without resetting anything.
"""

from __future__ import annotations

import numpy as np
import pytest

from aerobo import (api, objective, sizing, tail, tandem, tandemvlm,
                    wing_airfoil, wingtail)

FAMILIES = [
    ("winglet", lambda **kw: objective.Problem(mode="winglet", **kw),
     objective.evaluate, 0),
    ("wing t/c + sweep",
     lambda **kw: objective.Problem(mode="tier_a_plus", **kw),
     objective.evaluate, 0),
    ("tail", tail.TailProblem, tail.evaluate_tail, 1),
    ("wing + tail + winglet",
     lambda **kw: wingtail.WingTailProblem(winglet=True, capped=True, **kw),
     wingtail.evaluate_wing_tail, 1),
    ("tandem", tandem.TandemProblem, tandem.evaluate_tandem, 0),
    ("tandem (nonplanar)",
     lambda **kw: tandemvlm.TandemVLMProblem(winglets=True, **kw),
     tandemvlm.evaluate_tandem_vlm, 0),
    ("wing+airfoil (coupled)", wing_airfoil.WingAirfoilProblem,
     wing_airfoil.evaluate_wing_airfoil, 0),
]
IDS = [f[0] for f in FAMILIES]

#: how many SPAN rows the size block carries for a family: one per surface
#: that has a span of its own. The tandem pair has two wings and they need
#: not be the same width (sizing.span_labels), so its block is
#: [b_m, b_rear_m, S_m2]; everything else is [b_m, S_m2].
N_SPANS = {"tandem": 2, "tandem (nonplanar)": 2}


def _mid(prob) -> np.ndarray:
    b = prob.bounds
    return 0.5 * (b[:, 0] + b[:, 1])


def _size_block(name: str, b0: float, S0: float) -> list:
    """The size rows a candidate carries for this family: a span per surface,
    then the area."""
    return [b0] * N_SPANS.get(name, 1) + [S0]


def _own_size(prob) -> tuple[float, float]:
    return (float(prob.b), float(getattr(prob, "S", None)
                                 or prob.S_total))


# ------------------------------------------------------------- 1. the rows

@pytest.mark.parametrize("name, factory, evaluate, n_own", FAMILIES, ids=IDS)
def test_size_free_off_is_the_published_problem(name, factory, evaluate,
                                                n_own):
    base, off = factory(), factory(size_free=False)
    assert off.dim == base.dim
    assert np.array_equal(off.bounds, base.bounds)


@pytest.mark.parametrize("name, factory, evaluate, n_own", FAMILIES, ids=IDS)
def test_the_size_rows_are_two_and_boxed_around_the_family_s_own(
        name, factory, evaluate, n_own):
    base, free = factory(), factory(size_free=True)
    n_b = N_SPANS.get(name, 1)                  # a span row per surface
    assert free.dim == base.dim + n_b + 1
    assert np.array_equal(free.bounds[:base.dim], base.bounds)
    b0, S0 = _own_size(base)
    box = sizing.size_bounds(b0, S0, n_b)
    for i in range(n_b + 1):
        assert np.allclose(free.bounds[base.dim + i], box[i])
    # the family's own size is strictly INSIDE its box — every span row (the
    # pair's two wings open on the same band around the same nominal span)
    for i in range(n_b):
        assert free.bounds[base.dim + i][0] < b0 < free.bounds[base.dim + i][1]
    assert free.bounds[base.dim + n_b][0] < S0 < free.bounds[base.dim + n_b][1]


def test_the_modifier_blocks_stack_size_then_flight_then_chord():
    base = objective.Problem(mode="winglet")
    both = objective.Problem(mode="winglet", size_free=True,
                             flight_free=True, chord_order=3)
    assert both.dim == base.dim + 2 + 2 + 3
    x = np.arange(both.dim, dtype=float)
    assert sizing.size_from_x(x, True, True, 3) == (5.0, 6.0)
    from aerobo import geometry
    assert geometry.flight_from_x(x, True, 3) == (7.0, 8.0)
    assert geometry.chord_coeffs_from_x(x, 3) == (9.0, 10.0, 11.0)


# ---------------------------------------------------- 2/3/4. the physics

@pytest.mark.parametrize("name, factory, evaluate, n_own", FAMILIES, ids=IDS)
def test_the_wing_weighs_what_its_size_implies(name, factory, evaluate,
                                               n_own):
    free = factory(size_free=True)
    base = factory()
    b0, S0 = _own_size(base)
    x = np.concatenate([_mid(base), _size_block(name, b0, S0)])
    out = evaluate(x, free)
    if not out["feasible"]:
        pytest.skip(f"{name}: box midpoint is infeasible at its own size")
    assert out["W_wing_N"] > 0.0
    # ...and so does the EMPENNAGE, so the identity this modifier closes has
    # THREE terms. It is not a bookkeeping detail: sizing.SizedState keeps
    # W_tail_N out of W_wing_N on purpose ("the wing weighs this" and "the
    # aeroplane's surfaces weigh this" are different statements, and the
    # wing's own composite scores the first), so a two-term assertion here
    # is satisfied only by an empennage that weighs nothing.
    assert out["W_total_N"] == pytest.approx(out["W_fixed_N"]
                                             + out["W_wing_N"]
                                             + out["W_tail_N"])
    # a family that FLIES a tail pays for one; a family that flies none pays
    # exactly zero. Read off the problem (tail_type) rather than the id, so
    # the charge cannot quietly go back to the constant 0.0 it was before
    # weights.empennage_weight existed.
    if getattr(base, "tail_type", None):
        assert out["W_tail_N"] > 0.0
    else:
        assert out["W_tail_N"] == 0.0
    # ...and the trim target is that weight, not the fixed-CL one
    if "CL_target" in out:            # families that report their trim target
        assert out["CL_target"] == pytest.approx(
            out["W_total_N"] / (0.5 * 1.225 * base.V**2 * S0), rel=2e-3)
    # a LONGER span weighs more (the trap-closing mechanism)
    x_long = x.copy()
    x_long[-2] = min(1.6 * b0, free.bounds[-2][1])
    longer = evaluate(x_long, free)
    if longer["feasible"]:
        assert longer["W_wing_N"] > out["W_wing_N"]


@pytest.mark.parametrize("name, factory, evaluate, n_own", FAMILIES, ids=IDS)
def test_the_score_is_payload_lod_and_the_stress_margin_is_appended(
        name, factory, evaluate, n_own):
    free = factory(size_free=True)
    base = factory()
    b0, S0 = _own_size(base)
    out = evaluate(np.concatenate([_mid(base), _size_block(name, b0, S0)]),
                   free)
    if not out["feasible"]:
        pytest.skip(f"{name}: box midpoint is infeasible at its own size")
    assert out["score"] == out["f"]
    assert out["f"] == pytest.approx(out["W_fixed_N"] / out["D_N"])
    assert out["score"] != out["LoD"]           # a DIFFERENT objective
    g = np.atleast_1d(np.asarray(out["g"], dtype=float))
    assert g.size == n_own + 1
    assert g[-1] == pytest.approx(np.log(out["sigma_allow_Pa"]
                                         / out["sigma_root_Pa"]))


def test_a_long_thin_wing_is_overstressed_and_says_so_with_a_number():
    """The constraint has to bite, and to report the violation's magnitude —
    a penalty would hide how far outside the design is."""
    free = objective.Problem(mode="winglet", size_free=True)
    base = objective.Problem(mode="winglet")
    x = np.concatenate([_mid(base), [30.0, 10.0]])       # AR 90 -> refused
    out = objective.evaluate(x, free)
    assert not out["feasible"] and "aspect ratio" in out["reason"]

    x2 = np.concatenate([_mid(base), [24.0, 16.0]])      # AR 36, huge span
    out2 = objective.evaluate(x2, free)
    assert out2["feasible"]
    assert out2["g"] < 0.0                                # overstressed
    assert np.isfinite(out2["score"])                     # ...but scored
    f, g = objective.fg(x2, free)
    assert (f, g) == (out2["score"], out2["g"])


def test_the_aspect_ratio_band_is_the_penalty_contract():
    assert sizing.check_ar(10.0, 10.0) is None
    assert "aspect ratio" in sizing.check_ar(40.0, 8.0)
    assert "aspect ratio" in sizing.check_ar(6.0, 22.0)
    with pytest.raises(ValueError, match="aspect ratio"):
        sizing.sized_state(W_fixed_N=1000.0, b=40.0, S=8.0, taper=0.5,
                           tc=0.12, q_Pa=100.0)


def test_the_weight_loop_closes_on_itself():
    st = sizing.sized_state(W_fixed_N=6000.0, b=12.0, S=14.0, taper=0.5,
                            tc=0.12, q_Pa=1000.0)
    from aerobo import weights
    w_again = weights.wing_weight_raymer(12.0, 14.0, 0.5, 0.0, 0.12,
                                         weights.N_ULT_DEFAULT,
                                         W_dg_N=st.W_total_N, q_Pa=1000.0)
    assert st.W_wing_N == pytest.approx(w_again, rel=1e-6)
    # the tandem pair weighs BOTH wings at the shared gross weight
    pair = sizing.sized_state(W_fixed_N=6000.0, b=12.0, S=14.0, taper=0.5,
                              tc=0.12, q_Pa=1000.0, wing_areas=(7.0, 7.0))
    assert pair.W_wing_N != st.W_wing_N


# ------------------------------------------------------------ 5. registry

def test_the_registry_moves_the_contract_with_the_modifier():
    for base, name in api.CHORD_TWINS.items():
        assert name in api.PROBLEM_SPECS
    sized = {n: v for n, v in api.MODIFIER_VARIANTS.items() if "size" in v[1]}
    assert sized
    for name, (base, mods) in sized.items():
        spec, base_spec = api.PROBLEM_SPECS[name], api.PROBLEM_SPECS[base]
        assert spec.is_constrained, name
        n_base = base_spec.n_constraints if base_spec.is_constrained else 0
        assert spec.n_constraints == n_base + 1, name
        assert "root-bending stress margin" in spec.constraint_labels, name
        # a chosen size cannot travel as a flag when it is a design variable
        assert not set(spec.flags) & set(api.PLANFORM_KEYS), name
        assert spec.param_labels.count("b_m") == 1, name


def test_the_calibrated_aircraft_problem_keeps_its_own_identity():
    """It already has b and S in its vector (and frees t/c), so it is a
    FAMILY, not a modifier variant — offering the modifier there would free
    the same two numbers twice."""
    assert api.add_modifier("free planform (aircraft)", "size") is None
    assert api.add_modifier("free planform (aircraft)", "flight") is not None
    assert api.PROBLEM_SPECS["free planform (aircraft)"].param_labels == (
        "taper", "twist_root_deg", "twist_tip_deg", "b_m", "S_m2", "tc")


def test_the_gui_treats_the_free_planform_as_a_modifier():
    import gui.nice_app as v1

    ch = dict(v1.BUILDER_DEFAULTS, winglets="capped", tail=True,
              planform="free", chord="free", flight="free")
    name, notes = v1.derive_problem(ch)
    assert name == ("tail + winglet (span-capped) + free planform "
                    "+ free flight state + free chord law")
    assert not any("ignored" in n for n in notes)
    assert v1.size_available(ch)
    # the published sizing problem is still reachable, as its own family
    assert v1.derive_problem(
        dict(v1.BUILDER_DEFAULTS, planform="aircraft"))[0] == \
        "free planform (aircraft)"
    # ...and the families that carry calibrated geometry refuse it by name
    water = dict(v1.BUILDER_DEFAULTS, medium="water", planform="free")
    assert not v1.size_available(water)
    assert any("free span + area ignored" in n
               for n in v1.derive_problem(water)[1])
