"""Phase 2 gates: Tier B weight-coupled aircraft wing sizing (aircraft.py).

The gates encode the properties the problem was built for:
1. BIT-FOR-BIT legacy regression — the aero core is the identical Tier A
   code path (objective.evaluate_geometry): with the mission weight solved
   so the weight loop lands the trim target exactly on the legacy
   CL_target = 0.5 at the b = S = 10 probe, the Tier B breakdown LoD equals
   the Tier A objective to 1e-10 (measured: exactly 0.0 difference);
2. the span trap is CLOSED BY PHYSICS, not by the box: the 25-point span
   sweep has a strict interior maximum and f at the span upper bound is
   worse than the peak;
3. the structural margin g decreases monotonically with span (longer
   moment arm, heavier wing, thinner root — why the constrained optimum
   rides g = 0);
4. contracts: out-of-bounds / solver failure -> exactly (PENALTY, G_FAIL);
   solvable-but-overstressed -> TRUE finite f with signed g < 0;
   constrained BO smoke finds a feasible point within a 12-eval budget.
"""

import numpy as np
import pytest
from scipy.optimize import brentq

from aerobo import geometry, weights
from aerobo.aircraft import (
    G_FAIL,
    PENALTY,
    AircraftProblem,
    evaluate_aircraft,
    fg_aircraft,
)
from aerobo.mission import MissionSpec
from aerobo.objective import Problem, evaluate

PROBE_SHAPE = [0.45, 1.0, -2.0]        # [taper, twist_root, twist_tip]

# objective.evaluate(PROBE_SHAPE, Problem()) LoD — frozen 2026-07-12.
# Tier A defaults: b = S = 10, V = 14.6, CL_target = 0.5, NACA 2412 Re 1e6.
LEGACY_LOD = 34.924363024136234


# ------------------------------------------------------------- shape gates


def test_bounds_shape_and_dim():
    prob = AircraftProblem()
    b = prob.bounds
    assert b.shape == (6, 2)
    assert prob.dim == 6
    # rows: [taper, twist_root, twist_tip, b, S, tc] — first three + tc reuse
    # the geometry constants, span/area are the problem's own class constants
    assert tuple(b[0]) == geometry.TAPER_BOUNDS
    assert tuple(b[1]) == geometry.TWIST_ROOT_BOUNDS_DEG
    assert tuple(b[2]) == geometry.TWIST_TIP_BOUNDS_DEG
    assert tuple(b[3]) == AircraftProblem.B_BOUNDS
    assert tuple(b[4]) == AircraftProblem.S_BOUNDS
    assert tuple(b[5]) == geometry.TC_BOUNDS
    assert np.all(b[:, 1] > b[:, 0])


# ---------------------------------------------------- legacy aero regression


def test_legacy_regression_bit_for_bit():
    """Tier B at the legacy probe reproduces the Tier A objective EXACTLY.

    The wing weight feeds back into the trim target, so CL_target = 0.5
    cannot be dialled in directly through W_N. Instead, solve the mission
    weight with brentq so that the CONVERGED total weight satisfies
    W_total = 0.5 * q * S (i.e. CL_target = W_total/(q S) = 0.5) at the
    b = S = 10, tc = 0.12 probe — then the local trim Problem built inside
    evaluate_aircraft is field-for-field the legacy Tier A Problem and the
    aero core (evaluate_geometry) runs the identical code path.
    Measured 2026-07-12: brentq residual 0.0, CL_target - 0.5 == 0.0,
    LoD difference == 0.0 (bit-for-bit).
    """
    m0 = MissionSpec(W_N=1.0, V=14.6, altitude_m=0.0)   # W_N dummy: q only
    target_W_total = 0.5 * m0.q * 10.0                  # CL_target = 0.5, S = 10

    def resid(w):
        # q_Pa = mission q: mirrors evaluate_aircraft's total_weight call
        W_total, _ = weights.total_weight(
            w, 10.0, 10.0, PROBE_SHAPE[0], 0.0, 0.12, weights.N_ULT_DEFAULT,
            q_Pa=m0.q,
        )
        return W_total - target_W_total

    W_N = brentq(resid, 1.0, target_W_total, xtol=1e-12, rtol=8.9e-16)

    prob = AircraftProblem(mission=MissionSpec(W_N=W_N, V=14.6, altitude_m=0.0))
    out = evaluate_aircraft(np.array(PROBE_SHAPE + [10.0, 10.0, 0.12]), prob)
    legacy = evaluate(np.array(PROBE_SHAPE), Problem())

    assert out["feasible"] and legacy["feasible"]
    assert abs(out["CL_target"] - 0.5) < 1e-12          # measured: exact
    assert abs(out["LoD"] - legacy["LoD"]) < 1e-10      # measured: exact
    assert abs(legacy["LoD"] - LEGACY_LOD) < 1e-9       # frozen number

    # payload-L/D bookkeeping: f = W_fixed/D = LoD * W_fixed/W_total < LoD,
    # and reduces to LoD exactly as W_wing -> 0 (module docstring chain)
    assert out["f"] == pytest.approx(W_N / out["D_N"], rel=1e-12)
    assert out["f"] == pytest.approx(
        out["LoD"] * W_N / out["W_total_N"], rel=1e-12)
    assert out["f"] < out["LoD"]
    assert out["W_total_N"] == pytest.approx(
        W_N + out["W_wing_N"], rel=1e-12)


# ------------------------------------------------------------- span physics


@pytest.fixture(scope="module")
def span_sweep():
    """25-point span sweep at [0.45, 1.0, -2.0, b, S=14, tc=0.12] (the
    calibration sweep from the AircraftProblem docstring)."""
    prob = AircraftProblem()
    bs = np.linspace(prob.B_BOUNDS[0], prob.B_BOUNDS[1], 25)
    f = np.empty(25)
    g = np.empty(25)
    for j, b in enumerate(bs):
        out = evaluate_aircraft(
            np.array(PROBE_SHAPE + [b, 14.0, 0.12]), prob)
        assert out["feasible"], f"sweep point b={b:.2f} failed: {out['reason']}"
        f[j], g[j] = out["f"], out["g_sigma"]
    return bs, f, g


def test_span_optimum_strictly_interior(span_sweep):
    bs, f, _ = span_sweep
    i = int(np.argmax(f))
    # calibrated 2026-07-12: i = 17/24, b* = 30.08 m, f* = 47.57
    assert 0 < i < 24, f"span optimum on the box edge (i={i}, b={bs[i]:.2f})"


def test_weight_coupling_closes_span_trap(span_sweep):
    _, f, _ = span_sweep
    # pure aero would make f monotone in b (D_i ~ 1/b^2) and slam the top
    # bound; the weight loop must make the top bound WORSE than the peak
    assert f[-1] < f.max()


def test_g_monotone_decreasing_in_span(span_sweep):
    _, _, g = span_sweep
    # longer arm + heavier wing + thinner root: sigma up => margin down
    assert np.all(np.diff(g) < 0)
    assert g[0] > 0 > g[-1]        # boundary crossed inside the box


# ------------------------------------------------------------- fg contracts


def test_fg_out_of_bounds_penalty():
    prob = AircraftProblem()
    f, g = fg_aircraft(np.array(PROBE_SHAPE + [99.0, 14.0, 0.12]), prob)
    assert f == PENALTY and g == G_FAIL
    f, g = fg_aircraft(np.array(PROBE_SHAPE), prob)     # wrong shape
    assert f == PENALTY and g == G_FAIL


def test_fg_overstressed_true_f_signed_g():
    """Solvable-but-overstressed returns the TRUE objective with g < 0 —
    never a penalty (constrained BO must see the violation magnitude)."""
    x = np.array(PROBE_SHAPE + [10.0, 18.0, 0.12])
    f0, g0 = fg_aircraft(x, AircraftProblem())
    assert np.isfinite(f0) and f0 != PENALTY and g0 > 0     # baseline feasible

    f1, g1 = fg_aircraft(x, AircraftProblem(n_ult=60.0))    # cranked load case
    assert np.isfinite(f1) and f1 != PENALTY
    assert g1 < 0
    assert f1 < f0     # heavier ultimate sizing -> heavier wing -> more drag


def test_fg_solver_failure_penalty():
    # untrimmable: monstrous fixed weight on the smallest wing pushes
    # CL_target ~ 5, far beyond the polar validity range
    prob = AircraftProblem(mission=MissionSpec(W_N=60000.0, V=50.0))
    f, g = fg_aircraft(np.array(PROBE_SHAPE + [6.0, 8.0, 0.12]), prob)
    assert f == PENALTY and g == G_FAIL


# --------------------------------------------------------------- BO smoke


def test_bo_constrained_smoke():
    """Constrained BO on the real problem finds a feasible point within a
    12-eval budget (measured 2026-07-12: 3 feasible, best f = 20.34, ~2.4 s
    — under the 5 s slow-marker threshold). This is the gate that forced
    the LOG stress margin: with the linear normalised margin the constraint
    GP never located the 8% feasible island (0 feasible in 32 evals)."""
    from aerobo.optimize.constrained import run_bo_constrained

    prob = AircraftProblem()
    hist = run_bo_constrained(
        lambda x: fg_aircraft(x, prob), prob.bounds,
        n_init=6, n_iter=6, seed=0,
    )
    assert len(hist.y) == 12
    assert hist.n_feasible >= 1
    assert np.isfinite(hist.best_y)
