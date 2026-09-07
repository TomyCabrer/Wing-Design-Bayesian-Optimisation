"""Phase 1d gates: the Tier C cavitation-constrained hydrofoil problem.

Locked design box (pre-checked by hand, see report §11): d = 6,
x = [taper, twist_root, twist_tip, tc, depth 0.15-1.0 m, V 8-16 m/s],
fixed design lift 6 kN on b = 1.2 m / S = 0.144 m^2 seawater foil.

The gates encode the three properties the box was chosen for:
1. the cavitation bucket exists INSIDE the box (slow corner infeasible from
   high-cl suction, fast corner infeasible from dynamic pressure, feasible
   band between);
2. the UNCONSTRAINED optimum region cavitates (thin + shallow + fast has
   the best L/D in the hand-scan but g < 0) — the constraint is active,
   not decorative;
3. contracts: trim hits CL_target(V) exactly, fg returns true-L/D + signed
   g for solvable designs and exactly (PENALTY, G_FAIL) on solver failure.
"""

import numpy as np
import pytest

from aerobo.hydrofoil import (
    G_FAIL,
    PENALTY,
    HydrofoilProblem,
    evaluate_hydrofoil,
    fg_hydrofoil,
    froude_depth,
)


@pytest.fixture(scope="module")
def prob():
    return HydrofoilProblem()


def _x(taper=0.5, twr=0.0, twt=-1.0, tc=0.12, h=0.5, V=12.0):
    return np.array([taper, twr, twt, tc, h, V])


# ------------------------------------------------------------- basic shape

def test_dims_and_bounds(prob):
    assert prob.dim == 6
    assert prob.bounds.shape == (6, 2)
    assert prob.CL_target(10.0) == pytest.approx(
        6000.0 / (0.5 * 1025.0 * 100.0 * 0.144), rel=1e-12)


def test_trim_hits_cl_target(prob):
    o = evaluate_hydrofoil(_x(), prob)
    assert o["feasible"]
    assert o["CL"] == pytest.approx(o["CL_target"], abs=1e-9)
    assert o["LoD"] > 20.0


def test_froude_validity_over_box(prob):
    """High-Fn image model valid over the whole box: worst corner
    (V = 8, h = 1.0) has Fn_h = 2.55 >= 2.5."""
    h_hi, v_lo = prob.DEPTH_BOUNDS[1], prob.V_BOUNDS[0]
    assert froude_depth(v_lo, h_hi) > 2.5


# ------------------------------------------------------- cavitation bucket

def test_bucket_slow_side_infeasible(prob):
    """V = 8 -> CL_target = 1.27: high-cl suction cavitates at any depth."""
    for h in (0.15, 0.5, 1.0):
        y, g = fg_hydrofoil(_x(h=h, V=8.0), prob)
        assert y > PENALTY          # solvable: TRUE L/D reported
        assert g < 0.0              # but cavitating


def test_bucket_fast_side_infeasible_thin_shallow(prob):
    """The best-L/D corner of the hand-scan (thin, shallow, fast) cavitates:
    the constraint is ACTIVE at the unconstrained optimum."""
    y, g = fg_hydrofoil(_x(tc=0.08, h=0.15, V=16.0), prob)
    assert y > 30.0                 # excellent L/D...
    assert g < 0.0                  # ...that a real foil could not fly


def test_bucket_interior_feasible(prob):
    y, g = fg_hydrofoil(_x(tc=0.12, h=0.5, V=12.0), prob)
    assert y > 25.0
    assert g > 0.0


def test_depth_releases_constraint(prob):
    """At fixed V and geometry, going deeper adds static head: g increases."""
    g_shallow = fg_hydrofoil(_x(h=0.15, V=14.0), prob)[1]
    g_deep = fg_hydrofoil(_x(h=1.0, V=14.0), prob)[1]
    assert g_deep > g_shallow


def test_mast_drag_charges_depth(prob):
    """Deep costs parasite drag (the interior-optimum trade)."""
    o_sh = evaluate_hydrofoil(_x(h=0.15, V=12.0), prob)
    o_dp = evaluate_hydrofoil(_x(h=1.0, V=12.0), prob)
    assert o_dp["cd0_mast"] > o_sh["cd0_mast"]
    assert o_dp["cd0_mast"] == pytest.approx(
        prob.cf_mast * 2.0 * 1.0 * prob.c_mast / prob.S, rel=1e-12)
    # and the image charges shallow (both trades present)
    assert o_sh["CDi_surf"] > o_dp["CDi_surf"]


# ------------------------------------------------------------- fg contract

def test_fg_penalty_on_bounds_violation(prob):
    y, g = fg_hydrofoil(np.array([1.5, 0, 0, 0.12, 0.5, 12.0]), prob)
    assert y == PENALTY and g == G_FAIL
    y, g = fg_hydrofoil(np.zeros(3), prob)      # wrong shape
    assert y == PENALTY and g == G_FAIL


def test_fg_matches_evaluate(prob):
    x = _x()
    o = evaluate_hydrofoil(x, prob)
    y, g = fg_hydrofoil(x, prob)
    assert y == pytest.approx(o["score"], rel=1e-14)
    assert g == pytest.approx(o["g"], rel=1e-14)


def test_ops_cache_reused(prob):
    assert prob.ops(0.5) is prob.ops(0.5)
    assert prob.ops(0.5) is not prob.ops(0.6)


def test_deterministic(prob):
    x = _x(taper=0.31, twr=1.2, twt=-2.7, tc=0.113, h=0.42, V=13.1)
    assert fg_hydrofoil(x, prob) == fg_hydrofoil(x, prob)
