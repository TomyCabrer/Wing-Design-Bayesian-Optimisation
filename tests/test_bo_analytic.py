"""M2 gate: the BO machinery converges on analytic test functions.

Branin-2D: global minimum f* = 0.397887 at (-pi, 12.275), (pi, 2.275),
(9.42478, 2.475). Forrester-1D: f(x) = (6x-2)^2 sin(12x-4), global minimum
f* ~= -6.02074 at x* ~= 0.757249.
"""

import numpy as np
import pytest

from aerobo.optimize.bo import run_bo

BRANIN_FSTAR = 0.397887
FORRESTER_FSTAR = -6.020740


def branin(x):
    a, b, c = 1.0, 5.1 / (4 * np.pi**2), 5 / np.pi
    r, s, t = 6.0, 10.0, 1 / (8 * np.pi)
    return a * (x[1] - b * x[0] ** 2 + c * x[0] - r) ** 2 + s * (1 - t) * np.cos(x[0]) + s


def forrester(x):
    return float((6 * x[0] - 2) ** 2 * np.sin(12 * x[0] - 4))


def test_branin_converges_under_40_evals():
    bounds = np.array([[-5.0, 10.0], [0.0, 15.0]])
    res = run_bo(branin, bounds, n_init=8, n_iter=30, seed=0, maximize=False)
    assert len(res.y) <= 40
    assert res.best_y < BRANIN_FSTAR + 0.3, f"BO best {res.best_y} too far from {BRANIN_FSTAR}"
    assert not res.failures, f"GP fit failures at iterations {res.failures}"


def test_forrester_converges_under_40_evals():
    bounds = np.array([[0.0, 1.0]])
    res = run_bo(forrester, bounds, n_init=6, n_iter=20, seed=0, maximize=False)
    assert len(res.y) <= 40
    assert res.best_y < FORRESTER_FSTAR + 0.05
    assert abs(res.best_x[0] - 0.757249) < 0.05


@pytest.mark.parametrize("acqf", ["ucb", "qmes"])
def test_alternative_acqfs_run(acqf):
    bounds = np.array([[0.0, 1.0]])
    res = run_bo(forrester, bounds, n_init=6, n_iter=10, seed=1, maximize=False, acqf=acqf)
    # `best_y < 0` could not tell a working acquisition from a dead one: every
    # GP-fit/acqf failure is swallowed into a uniform random draw (recorded
    # only in res.failures), and Forrester is negative over ~52 % of [0, 1], so
    # the Sobol init alone cleared that bar. Assert the fallback never fired,
    # and a bar random search does not reach.
    assert not res.failures, f"acqf {acqf!r} fell back to random at {res.failures}"
    assert res.best_y < -4.0, f"{acqf}: best {res.best_y} vs f* {FORRESTER_FSTAR}"


def test_best_so_far_monotone():
    bounds = np.array([[-5.0, 10.0], [0.0, 15.0]])
    res = run_bo(branin, bounds, n_init=5, n_iter=5, seed=2, maximize=False)
    assert np.all(np.diff(res.best_so_far) <= 1e-12)
