"""Gates for MULTI-constraint support in optimize/constrained.py.

Two parts:

1. ANALYTIC 2-constraint toy (all closed form): maximise
       f(x)  = -(x0 - 0.7)^2 - (x1 - 0.7)^2
   s.t.  g1(x) = 0.5 - x0 >= 0        (pushes x0 off the unconstrained opt)
         g2(x) = x1 - 0.3 >= 0        (slack at the optimum)
   on [0, 1]^2.
   - Unconstrained optimum (0.7, 0.7), f = 0 — INFEASIBLE (g1 = -0.2): an
     optimiser ignoring g1 is caught immediately.
   - Constrained optimum: x0 = 0.5 (g1 ACTIVE), x1 = 0.7 (g2 slack 0.4)
     -> x* = (0.5, 0.7), f* = -0.04.

2. SINGLE-G REGRESSION: the m == 1 path must be bit-for-bit the original
   code. The frozen numbers below were captured from the PRE-multi-g
   constrained.py (git state before this change) by running each scenario
   in a fresh process on this machine (macOS arm64, .venv torch/botorch,
   2026-07-12); BO was run twice in separate processes and reproduced
   best_y and sum(X) exactly, so equality assertions are safe here. Any
   drift means the legacy path's RNG consumption or numerics changed —
   a hard failure for the hydrofoil/rec_rule studies.
"""

import numpy as np
import pytest

from aerobo.optimize.constrained import (
    make_constrained_history,
    run_bo_constrained,
    run_ga_constrained,
    run_penalty_gradient,
    run_random_constrained,
    run_slsqp_constrained,
    run_sobol_constrained,
)

BOUNDS2G = np.array([[0.0, 1.0], [0.0, 1.0]])
F_STAR = -0.04                       # at (0.5, 0.7); g1 active, g2 slack 0.4


def toy2g(x):
    f = -((x[0] - 0.7) ** 2) - (x[1] - 0.7) ** 2
    return f, np.array([0.5 - x[0], x[1] - 0.3])


# ---------------------------------------------------------------- plumbing

def test_history_multig_feasibility_is_all():
    """A point is feasible only if EVERY margin is >= 0."""
    X = np.array([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]])
    y = np.array([5.0, 1.0, 2.0])
    g = np.array([[0.4, -0.1],        # one violated -> INFEASIBLE
                  [0.1, 0.2],         # feasible
                  [0.3, 0.05]])       # feasible
    h = make_constrained_history(X, y, g, "t", 0)
    assert h.g.shape == (3, 2)
    assert h.n_feasible == 2
    # the infeasible y=5 must NOT be the incumbent
    assert h.best_y == pytest.approx(2.0)
    assert h.best_g == pytest.approx(0.05)     # scalar MIN margin at best
    assert h.best_so_far[0] == -np.inf
    assert h.best_so_far[-1] == pytest.approx(2.0)


def test_history_n1_column_squeezes_to_legacy_shape():
    """(n, 1) margins collapse to (n,) — rec_rule.py indexes h.g[i]."""
    h = make_constrained_history([[0.0], [1.0]], [1.0, 2.0],
                                 [[0.5], [-0.5]], "t", 0)
    assert h.g.shape == (2,)
    assert float(h.g[0]) == 0.5                # scalar indexing still works
    assert h.best_y == pytest.approx(1.0)


def test_history_multig_none_feasible():
    h = make_constrained_history([[0.0, 0.0]], [0.0], [[-1.0, 1.0]], "t", 0)
    assert h.best_x is None and h.best_y == -np.inf and h.n_feasible == 0


def test_budget_single_charge_per_point_multig():
    """scipy queries f, g1 and g2 separately at the same x: ONE eval."""
    calls = {"n": 0}

    def fg(x):
        calls["n"] += 1
        return toy2g(x)

    h = run_slsqp_constrained(fg, BOUNDS2G, n_evals=30, seed=0)
    assert calls["n"] == 30
    assert len(h.y) == 30 and h.g.shape == (30, 2)


# ------------------------------------------------------------- optimisers

def test_bo_multig_finds_constrained_optimum():
    h = run_bo_constrained(toy2g, BOUNDS2G, n_init=8, n_iter=24, seed=0)
    assert h.g.shape == (32, 2)
    assert h.best_x is not None
    assert h.best_g >= 0.0                        # min margin — BOTH hold
    assert h.best_y == pytest.approx(F_STAR, abs=0.05)
    assert h.best_x[0] <= 0.5 + 1e-9              # respects g1 ...
    assert h.best_x[1] >= 0.3 - 1e-9              # ... and g2


def test_slsqp_multig_nails_active_constraint():
    h = run_slsqp_constrained(toy2g, BOUNDS2G, n_evals=80, seed=0)
    assert h.best_y == pytest.approx(F_STAR, abs=1e-4)
    assert h.best_x[0] == pytest.approx(0.5, abs=1e-3)   # g1 active
    assert h.best_x[1] == pytest.approx(0.7, abs=1e-3)   # g2 slack
    assert 0.0 <= h.best_g < 1e-3                 # binding margin ~ 0


@pytest.mark.parametrize("runner", [run_ga_constrained, run_penalty_gradient,
                                    run_random_constrained,
                                    run_sobol_constrained])
def test_multig_baselines_coherent(runner):
    h = runner(toy2g, BOUNDS2G, 60, seed=0)
    assert len(h.y) <= 60
    assert h.g.shape == (len(h.y), 2)
    assert h.best_x is not None
    assert h.best_g >= 0.0
    # feasible best can never beat the true constrained optimum
    assert h.best_y <= F_STAR + 1e-12
    assert h.best_y > -0.5                        # found something sensible
    # running best is monotone and ends at best_y
    finite = np.isfinite(h.best_so_far)
    assert np.all(np.diff(h.best_so_far[finite]) >= 0.0)
    assert h.best_so_far[-1] == pytest.approx(h.best_y)


def test_ga_multig_explicit_n_constraints_matches_auto():
    """Declaring m upfront must give the same run as the resize-restart."""
    h_auto = run_ga_constrained(toy2g, BOUNDS2G, 60, seed=3)
    h_decl = run_ga_constrained(toy2g, BOUNDS2G, 60, seed=3, n_constraints=2)
    np.testing.assert_array_equal(h_auto.X, h_decl.X)
    np.testing.assert_array_equal(h_auto.g, h_decl.g)
    assert h_auto.best_y == h_decl.best_y


# --------------------------------------------- single-g frozen regression
#
# Scenarios copied from tests/test_constrained.py; numbers captured from
# the ORIGINAL single-constraint constrained.py (see module docstring).

BOUNDS1G = np.array([[-2.0, 2.0], [-2.0, 2.0]])


def toy1g(x):
    return -(x[0] ** 2 + x[1] ** 2), x[0] + x[1] - 1.0


def _branin_cut(x):
    x1, x2 = x
    a, b, c = 1.0, 5.1 / (4 * np.pi**2), 5.0 / np.pi
    r, s, t = 6.0, 10.0, 1.0 / (8 * np.pi)
    br = a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s
    return -br, x1 - 2.0


BRANIN_BOUNDS = np.array([[-5.0, 10.0], [0.0, 15.0]])

# Frozen outputs of the ORIGINAL code (capture protocol in module docstring).
FROZEN = {
    "bo_toy_best_y": -0.5003651404068994,
    "bo_toy_best_g": 0.000364696572091594,
    "bo_toy_X_sum": 36.02464148293631,
    "bo_branin_best_y": -0.4028162005028779,
    "bo_branin_X_sum": 423.55799544981875,
    "slsqp_toy_best_y": -0.5000000000000001,
    "slsqp_toy_X_sum": 58.93600891767821,
    "ga_toy_best_y": -0.5300186532510787,
    "ga_toy_X_sum": 59.37681454211736,
    "pen_toy_best_y": -0.6714642938844848,
    "pen_toy_X_sum": 76.90836861387338,
    "rand_toy_best_y": -0.8124390913121637,
    "sobol_toy_best_y": -0.8188153585253977,
}

EXACT = dict(rel=1e-12, abs=1e-12)   # bit-for-bit intent, FP-noise headroom


def test_regression_bo_single_g_toy_identical():
    h = run_bo_constrained(toy1g, BOUNDS1G, n_init=8, n_iter=32, seed=0)
    assert h.g.ndim == 1                          # legacy (n,) shape kept
    assert h.best_y == pytest.approx(FROZEN["bo_toy_best_y"], **EXACT)
    assert h.best_g == pytest.approx(FROZEN["bo_toy_best_g"], **EXACT)
    # same EVALUATED POINTS, not just the same winner: RNG stream untouched
    assert float(np.sum(h.X)) == pytest.approx(FROZEN["bo_toy_X_sum"], **EXACT)


def test_regression_bo_single_g_branin_identical():
    h = run_bo_constrained(_branin_cut, BRANIN_BOUNDS, n_init=10, n_iter=40,
                           seed=0)
    assert h.best_y == pytest.approx(FROZEN["bo_branin_best_y"], **EXACT)
    assert float(np.sum(h.X)) == pytest.approx(FROZEN["bo_branin_X_sum"],
                                               **EXACT)


def test_regression_baselines_single_g_identical():
    h = run_slsqp_constrained(toy1g, BOUNDS1G, n_evals=80, seed=0)
    assert h.best_y == pytest.approx(FROZEN["slsqp_toy_best_y"], **EXACT)
    assert float(np.sum(h.X)) == pytest.approx(FROZEN["slsqp_toy_X_sum"],
                                               **EXACT)

    h = run_ga_constrained(toy1g, BOUNDS1G, 60, seed=0)
    assert h.best_y == pytest.approx(FROZEN["ga_toy_best_y"], **EXACT)
    assert float(np.sum(h.X)) == pytest.approx(FROZEN["ga_toy_X_sum"], **EXACT)

    h = run_penalty_gradient(toy1g, BOUNDS1G, 60, seed=0)
    assert h.best_y == pytest.approx(FROZEN["pen_toy_best_y"], **EXACT)
    assert float(np.sum(h.X)) == pytest.approx(FROZEN["pen_toy_X_sum"],
                                               **EXACT)

    h = run_random_constrained(toy1g, BOUNDS1G, 60, seed=0)
    assert h.best_y == pytest.approx(FROZEN["rand_toy_best_y"], **EXACT)

    h = run_sobol_constrained(toy1g, BOUNDS1G, 60, seed=0)
    assert h.best_y == pytest.approx(FROZEN["sobol_toy_best_y"], **EXACT)
