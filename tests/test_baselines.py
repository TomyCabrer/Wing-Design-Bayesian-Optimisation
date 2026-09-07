"""M6 gate: baselines respect the eval budget and make progress; metrics work."""

import numpy as np
import pytest

from aerobo.optimize.baselines import run_ga, run_gradient, run_grid, run_random, run_sobol
from aerobo.optimize.metrics import aggregate_best_so_far, evals_to_target, make_history


def sphere_max(x):
    """Maximise -||x - 0.3||^2 over [0,1]^d; optimum 0 at x = 0.3."""
    return -float(np.sum((x - 0.3) ** 2))


BOUNDS2 = np.array([[0.0, 1.0], [0.0, 1.0]])

#: The bar each runner has to CLEAR at n_evals = 40 on this problem. The old
#: shared ``best_y > -0.5`` gated nothing — only ~11 % of this box scores below
#: -0.5, so forty arbitrary draws met it with probability ~1 and a runner whose
#: search had degenerated to uniform sampling still passed. The three samplers
#: can do no better than their own sampling law (the 6x6 grid's nearest node to
#: the optimum is (0.1, 0.1) away, i.e. exactly -0.02), while the multistart
#: gradient must CONVERGE on a smooth quadratic.
BEST_Y_FLOOR = {run_random: -0.03, run_sobol: -0.03, run_grid: -0.021,
                run_ga: -0.03, run_gradient: -1e-6}


@pytest.mark.parametrize("runner", [run_random, run_sobol, run_grid, run_ga, run_gradient])
def test_baseline_budget_and_progress(runner):
    res = runner(sphere_max, BOUNDS2, n_evals=40, seed=0)
    assert len(res.y) <= 40
    assert len(res.y) >= 30            # actually used most of the budget
    assert res.best_y > BEST_Y_FLOOR[runner], res.best_y
    if runner is run_ga:
        # SELECTION PRESSURE, which no bound on best_y can see here: at this
        # budget (pop 20, two generations) the GA's incumbent IS the best point
        # of its own initial population, so a GA that had lost its fitness sort
        # or its elitism and was drawing uniformly would report the same
        # best_y. What it could not report is a second generation bred from the
        # fitter half of the first — measured population means -0.293 -> -0.177
        # here, against +0.05 +/- 0.05 for uniform draws and -0.14 for a GA
        # handed the objective with the sign flipped.
        y = np.asarray(res.y, dtype=float)
        h = len(y) // 2
        assert y[h:].mean() > y[:h].mean() + 0.09, (y[:h].mean(), y[h:].mean())
    assert np.all(np.diff(res.best_so_far) >= -1e-12)
    assert res.best_y == pytest.approx(np.max(res.y))


def test_gradient_converges_on_smooth_problem():
    res = run_gradient(sphere_max, BOUNDS2, n_evals=60, seed=1)
    assert res.best_y > -1e-6          # FD L-BFGS-B nails a smooth quadratic


def test_evals_to_target():
    y = np.array([-5.0, -3.0, -1.0, -0.5])
    assert evals_to_target(y, -1.0, maximize=True) == 3
    assert evals_to_target(y, -0.1, maximize=True) is None
    assert evals_to_target(-y, 1.0, maximize=False) == 3


def test_aggregate_and_history_truncation():
    h = make_history(np.zeros((10, 2)), np.arange(10.0), "t", 0, n_evals=8)
    assert len(h.y) == 8
    agg = aggregate_best_so_far([np.arange(5.0), np.arange(7.0)])
    assert agg["median"].shape == (5,)
    assert agg["n_seeds"] == 2
