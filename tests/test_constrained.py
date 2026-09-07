"""Gates for optimize/constrained.py on an ANALYTIC toy problem.

Toy (all closed form): maximise f(x) = -(x1^2 + x2^2) subject to
g(x) = x1 + x2 - 1 >= 0 on [-2, 2]^2.

- Unconstrained optimum (0, 0), f = 0 — INFEASIBLE (g = -1): any optimiser
  that ignores the constraint is caught immediately.
- Constrained optimum: nearest point of the half-plane x1 + x2 >= 1 to the
  origin = (0.5, 0.5), f* = -0.5, active constraint g* = 0.
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

BOUNDS = np.array([[-2.0, 2.0], [-2.0, 2.0]])
F_STAR = -0.5


def toy(x):
    return -(x[0] ** 2 + x[1] ** 2), x[0] + x[1] - 1.0


# ---------------------------------------------------------------- plumbing

def test_history_best_is_feasible_only():
    X = np.array([[0.0, 0.0], [1.0, 1.0], [0.6, 0.6]])
    y = np.array([0.0, -2.0, -0.72])
    g = np.array([-1.0, 1.0, 0.2])
    h = make_constrained_history(X, y, g, "t", 0)
    # the infeasible f=0 point must NOT be the incumbent
    assert h.best_y == pytest.approx(-0.72)
    assert h.best_g == pytest.approx(0.2)
    assert h.n_feasible == 2
    assert h.best_so_far[0] == -np.inf          # nothing feasible yet
    assert h.best_so_far[-1] == pytest.approx(-0.72)


def test_history_no_feasible_point():
    h = make_constrained_history([[0.0, 0.0]], [0.0], [-1.0], "t", 0)
    assert h.best_x is None and h.best_y == -np.inf
    assert h.n_feasible == 0


def test_budget_single_charge_per_point():
    """scipy calls f and g separately at the same x: one eval charged."""
    calls = {"n": 0}

    def fg(x):
        calls["n"] += 1
        return toy(x)

    h = run_slsqp_constrained(fg, BOUNDS, n_evals=30, seed=0)
    assert calls["n"] == 30                     # solver calls == budget
    assert len(h.y) == 30 and len(h.g) == 30


# ------------------------------------------------------------- optimisers

def test_constrained_bo_finds_constrained_optimum():
    h = run_bo_constrained(toy, BOUNDS, n_init=8, n_iter=32, seed=0)
    assert h.best_x is not None
    assert h.best_g >= 0.0                      # recommendation is feasible
    assert h.best_y == pytest.approx(F_STAR, abs=0.05)
    # and it did NOT sneak toward the infeasible unconstrained optimum
    assert h.best_x.sum() >= 1.0 - 1e-9


def test_slsqp_nails_active_constraint():
    h = run_slsqp_constrained(toy, BOUNDS, n_evals=80, seed=0)
    assert h.best_y == pytest.approx(F_STAR, abs=1e-3)
    assert abs(h.best_g) < 1e-3                 # constraint active at optimum


@pytest.mark.parametrize("runner", [run_ga_constrained, run_penalty_gradient,
                                    run_random_constrained,
                                    run_sobol_constrained])
def test_baselines_feasible_and_budget_fair(runner):
    h = runner(toy, BOUNDS, 60, seed=0)
    assert len(h.y) <= 60
    assert h.best_x is not None
    assert h.best_g >= 0.0
    assert h.best_y > -2.0                      # found something sensible


# ----------------------------------------------------- constrained Branin

def _branin_cut(x):
    """Constrained Branin [ANALYTIC optimum]: maximise -branin(x) subject to
    g = x1 - 2 >= 0. The cut excludes the global minimum at x1 = -pi but
    keeps the other two ((pi, 2.275), (9.42478, 2.475)), so the constrained
    optimum equals the unconstrained one: branin* = 0.397887."""
    x1, x2 = x
    a, b, c = 1.0, 5.1 / (4 * np.pi**2), 5.0 / np.pi
    r, s, t = 6.0, 10.0, 1.0 / (8 * np.pi)
    br = a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s
    return -br, x1 - 2.0


BRANIN_BOUNDS = np.array([[-5.0, 10.0], [0.0, 15.0]])
BRANIN_STAR = -0.397887


def test_constrained_bo_on_branin_cut():
    """Multimodal gate: BO must find a feasible Branin basin (x1 >= 2),
    not the excluded global minimum at x1 = -pi."""
    h = run_bo_constrained(_branin_cut, BRANIN_BOUNDS, n_init=10, n_iter=40,
                           seed=0)
    assert h.best_x is not None
    assert h.best_g >= 0.0                        # x1 >= 2
    assert h.best_y == pytest.approx(BRANIN_STAR, abs=0.2)
