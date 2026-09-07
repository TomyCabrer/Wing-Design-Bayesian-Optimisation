"""One optimiser starting where another stopped.

The question this answers: "starting from 1 optimisation process and then
changing to another?" A handoff is only meaningful if the second run actually
BEGINS at the design the first one ended on. Before this, ``x_seed`` was
accepted by the BO loops and REFUSED by everything else, so the classic
global-then-local pairing (a DOE or a GA, then SLSQP) could not be expressed at
all — which is why these tests assert the start point itself and not merely
that the argument is tolerated.

Three properties, and each is a separate way the feature could be fake:

* the seed is the FIRST point evaluated — otherwise the handoff is decoration;
* later starts are still random — otherwise a handoff silently turns a
  multistart into a single local polish, which is a different method wearing
  the same label;
* a seed outside the box is REFUSED, never clipped — a moved start point makes
  "this run cannot come back worse than the design I gave it" false while the
  caller believes it is armed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aerobo import api                                        # noqa: E402
from aerobo.optimize import baselines, constrained as C       # noqa: E402


BOX = np.array([[-2.0, 2.0], [-2.0, 2.0], [-2.0, 2.0]])


def _quad(x):
    """A smooth unconstrained bowl with its maximum at the origin."""
    x = np.asarray(x, float)
    return -float(np.sum((x - 0.3) ** 2))


def _quad_g(x):
    """The same, with one slack constraint, in the (f, g) shape."""
    return _quad(x), np.array([1.0])


SEED_X = np.array([1.5, -1.25, 0.75])


def test_gradient_starts_at_the_design_it_was_handed():
    h = baselines.run_gradient(_quad, BOX, n_evals=40, seed=0, x_seed=SEED_X)
    assert np.allclose(np.asarray(h.X)[0], SEED_X), (
        f"first evaluated point was {np.asarray(h.X)[0]}, not the handoff "
        f"design {SEED_X} — the seed was accepted and then ignored")


def test_slsqp_starts_at_the_design_it_was_handed():
    h = C.run_slsqp_constrained(_quad_g, BOX, n_evals=40, seed=0,
                                x_seed=SEED_X)
    assert np.allclose(np.asarray(h.X)[0], SEED_X)


def test_penalty_starts_at_the_design_it_was_handed():
    h = C.run_penalty_gradient(_quad_g, BOX, n_evals=40, seed=0,
                               x_seed=SEED_X)
    assert np.allclose(np.asarray(h.X)[0], SEED_X)


def test_without_a_handoff_the_first_point_is_random_and_unchanged():
    """The seedless path must be bit-identical to what it was before."""
    a = baselines.run_gradient(_quad, BOX, n_evals=40, seed=7)
    b = baselines.run_gradient(_quad, BOX, n_evals=40, seed=7, x_seed=None)
    assert np.allclose(np.asarray(a.X), np.asarray(b.X)), (
        "passing x_seed=None changed the run; the handoff is not free")
    assert not np.allclose(np.asarray(a.X)[0], SEED_X)


def test_the_handoff_design_seeds_ONE_start_not_every_restart():
    """A handoff must not collapse a multistart into one repeated polish.

    THIS TEST IS ON ITS THIRD SHAPE, AND THE FIRST TWO WERE WRONG. Version one
    asked whether any evaluation landed far from both the seed and the optimum;
    a mutant that reused the seed for EVERY restart passed it, because the
    repeated trajectory from a fixed start still sweeps that region. Version
    two asked for the seed to be evaluated exactly once; it failed on correct
    code, because scipy legitimately re-evaluates its own start point during
    the finite-difference and line-search steps.

    The signal that actually separates them was measured rather than guessed.
    At n_evals=120 on this bowl:

        correct code   seed evaluated  4x, 112 distinct points of 120
        seed-every-restart mutant      40x,  12 distinct points of 120

    A multistart that keeps drawing fresh starts explores; one that restarts
    from the same design retraces one trajectory. The thresholds below sit far
    from both measurements, so neither ordinary scipy jitter nor a change of
    line search can move the verdict.
    """
    n = 120
    h = baselines.run_gradient(_quad, BOX, n_evals=n, seed=3, x_seed=SEED_X)
    X = np.asarray(h.X)
    distinct = len(np.unique(X.round(9), axis=0))
    assert distinct > n // 2, (
        f"only {distinct} of {n} evaluated points were distinct; the run "
        f"retraced one trajectory instead of drawing fresh starts, so the "
        f"handoff collapsed the multistart into a repeated local polish")
    hits = int(np.sum(np.all(np.isclose(X, SEED_X), axis=1)))
    assert hits < n // 10, (
        f"the handoff design was evaluated {hits} times in {n}; it must seed "
        f"ONE start, not every restart")


@pytest.mark.parametrize("runner", [
    lambda xs: baselines.run_gradient(_quad, BOX, n_evals=20, x_seed=xs),
    lambda xs: C.run_slsqp_constrained(_quad_g, BOX, n_evals=20, x_seed=xs),
    lambda xs: C.run_penalty_gradient(_quad_g, BOX, n_evals=20, x_seed=xs),
])
def test_a_seed_outside_the_box_is_refused_not_clipped(runner):
    with pytest.raises(ValueError, match="outside the box"):
        runner(np.array([9.9, 0.0, 0.0]))


@pytest.mark.parametrize("runner", [
    lambda xs: baselines.run_gradient(_quad, BOX, n_evals=20, x_seed=xs),
    lambda xs: C.run_slsqp_constrained(_quad_g, BOX, n_evals=20, x_seed=xs),
])
def test_a_seed_of_the_wrong_width_is_refused(runner):
    with pytest.raises(ValueError, match="dimension"):
        runner(np.array([0.0, 0.0]))


def test_the_budget_is_unchanged_by_a_handoff():
    """A handoff buys a better start, never extra evaluations."""
    plain = baselines.run_gradient(_quad, BOX, n_evals=40, seed=1)
    handed = baselines.run_gradient(_quad, BOX, n_evals=40, seed=1,
                                    x_seed=SEED_X)
    assert len(np.asarray(plain.X)) == len(np.asarray(handed.X)) == 40


def test_the_api_still_refuses_the_optimisers_that_generate_their_own_points():
    assert api.X_SEED_OPTIMISERS == {"bo", "bo_slsqp", "gradient", "slsqp",
                                     "penalty"}
    for name in ("ga", "sobol", "random"):
        cfg = api.RunConfig(problem_name="trim wing", optimiser=name,
                            budget=8, x_seed=[0.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="cannot take an x_seed"):
            api.run(cfg)


def test_an_optimiser_that_cannot_read_a_seed_says_so_before_the_run():
    """The pre-launch check is the shell's control; it must refuse what the
    run refuses, or the button raises 20 seconds into a search instead."""
    cfg = api.RunConfig(problem_name="tail", optimiser="ga", budget=8,
                        x_seed=[0.5, 0.0, 0.0, 1.5, 5.0])
    with pytest.raises(ValueError, match="cannot take an x_seed"):
        api.check_x_seed(cfg)
