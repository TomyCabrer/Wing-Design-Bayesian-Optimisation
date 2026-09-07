"""The two `acqf` defects, which were STATED and unfixed for a session.

Both are the same failure in different clothing: a run that keeps its label
while doing something else.

**Defect 1 — accepted and ignored.** `optimize/constrained.py` hardwires
LogConstrainedExpectedImprovement and takes no `acqf` argument, but `acqf`
sat in `RUN_POLICY_FLAG_KEYS` unconditionally, so `check_flags` accepted it on
a constrained problem and nothing read it. `logei`, `ucb`, `qmes` and the
literal `"nonsense"` produced a bit-identical evaluation sequence, which means
an arm labelled "bo-ucb (constrained)" was the LogCEI control under a false
name. That is precisely the silently-dropped-flag failure `check_flags` exists
to prevent, living inside the refusal itself.

**Defect 2 — misspelt and swallowed.** The name check lived beside the
dispatch, INSIDE the per-iteration `try`, whose bare `except Exception` treats
any failure as a GP failure and falls back to a uniform random draw. So a typo
did not raise: every iteration degraded to random search while the row kept
its label, and `failures` — a column the wing sweep never stored — was the
only witness.

The tests assert the OUTCOME (does a bad name reach the objective as random
search?), not that the code contains a check.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aerobo import api                                        # noqa: E402
from aerobo.optimize.bo import ACQFS, run_bo                  # noqa: E402


BOX = np.array([[-2.0, 2.0], [-2.0, 2.0]])


def _quad(x):
    x = np.asarray(x, float)
    return -float(np.sum((x - 0.3) ** 2))


# ---------------------------------------------------------------- defect 2

def test_a_misspelt_acqf_raises_instead_of_degrading_to_random():
    with pytest.raises(ValueError, match="unknown acqf"):
        run_bo(_quad, BOX, n_init=4, n_iter=8, seed=0, acqf="nonsense")


def test_the_error_names_the_choices_so_a_typo_is_self_correcting():
    with pytest.raises(ValueError) as e:
        run_bo(_quad, BOX, n_init=4, n_iter=8, seed=0, acqf="logie")
    for name in ACQFS:
        assert name in str(e.value)


def test_it_raises_BEFORE_spending_any_of_the_budget():
    """A bad name is a caller error, and must cost zero evaluations.

    If the check were still inside the loop, the run would complete, return a
    history of `n_evals` uniform random draws, and report itself as BO.
    """
    seen = []

    def counting(x):
        seen.append(x)
        return _quad(x)

    with pytest.raises(ValueError):
        run_bo(counting, BOX, n_init=8, n_iter=32, seed=0, acqf="nope")
    assert seen == [], (
        f"{len(seen)} evaluations were spent before the name was rejected")


@pytest.mark.parametrize("name", sorted(ACQFS))
def test_every_declared_acqf_actually_runs(name):
    """The guard must not be so tight it refuses a name the code supports."""
    h = run_bo(_quad, BOX, n_init=4, n_iter=6, seed=0, acqf=name)
    assert len(np.asarray(h.X)) == 10
    assert not getattr(h, "failures", []), (
        f"{name} produced fallback iterations: {h.failures} — the run would "
        f"be random search wearing a BO label")


# ---------------------------------------------------------------- defect 1

def test_a_constrained_problem_REFUSES_an_acqf_it_cannot_honour():
    constrained = [n for n, s in api.PROBLEM_SPECS.items() if s.is_constrained]
    assert constrained, "no constrained problem to test"
    for name in constrained[:5]:
        assert "acqf" not in api.accepted_flags(name), (
            f"{name} is constrained but still advertises `acqf`, which "
            f"optimize/constrained.py cannot read")
        with pytest.raises(KeyError, match="acqf"):
            api.check_flags(name, {"acqf": "ucb"})


def test_an_unconstrained_problem_STILL_accepts_an_acqf():
    """The fix must not cost the six cases where the choice is real."""
    unc = [n for n, s in api.PROBLEM_SPECS.items() if not s.is_constrained]
    assert unc, "no unconstrained problem to test"
    for name in unc[:5]:
        assert "acqf" in api.accepted_flags(name)
        api.check_flags(name, {"acqf": "ucb"})       # must not raise


def test_the_refusal_is_exactly_the_constrained_set():
    """Not a hand-kept list: derived from each spec's own `is_constrained`."""
    refused = {n for n in api.PROBLEM_SPECS
               if "acqf" not in api.accepted_flags(n)}
    constrained = {n for n, s in api.PROBLEM_SPECS.items() if s.is_constrained}
    assert refused == constrained
