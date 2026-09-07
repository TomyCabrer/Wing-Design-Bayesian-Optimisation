"""Did the constrained acquisition actually RUN? — the assertion that was missing.

`optimize/constrained.py`'s BO loop wraps the whole model-fit-and-optimise step
in a bare ``except Exception``: on any failure it records the iteration in
``failures``, draws ``rng.uniform`` over the box, and carries on. That fallback
is CORRECT — one bad GP fit should not discard a run's paid-for XFOIL sweeps —
but it converts "broken" into "worse", and nothing in the output distinguishes
the two. A run that fell back on every iteration is a **uniform random search**
wearing a Bayesian-optimisation label.

This has already happened twice in this project, both times in the sibling
multi-objective path and both times invisible to every output-shaped test: a
front of 8 non-dominated feasible designs with a monotone hypervolume trace,
produced entirely by Sobol (report §16.3c), and a second, within the hour, from
a purely *diagnostic* line that raised at q > 1 (§16.3d). The tests that now
exist there assert ``gp_failures == []``.

**The scalar constrained path had no such assertion at all.** The only
``.failures`` assertion in the suite was `tests/test_bo_analytic.py:32`, on the
UNCONSTRAINED path — while `run_bo_constrained` is what every composite,
`composite_goal`, `composite_asf` and wing run in this report goes through. So
most published numbers had no in-suite evidence their acquisition was ever
alive. This file is that evidence.

It is parametrised over **every axis the try-block varies on**, because a
fallback that fires on one branch and not another is exactly what a single
happy-path test misses:

* ``m == 1`` against ``m > 1`` — these are two literally different bodies
  inside the try (a hand-written single-constraint path kept bit-for-bit, and a
  general ``ModelListGP`` of 1 + m), with different ``constraints`` dicts;
* every ``refusal`` encoding — the imputed ``y`` feeds the GP fit inside the
  try, and this project has already measured that the sentinel encoding wrecks
  the GP's scale (report §16.2/`refusal-is-scale-aware`), which is precisely
  the kind of thing that makes a fit raise;
* warm start on and off — ``x_init`` changes the training set the fit sees;
* ``iter_cb`` on and off — the diagnostics block is a nested try, and the
  second of the two historical bugs was a *diagnostic* line being counted as a
  GP failure. Here it must NOT be, and that is asserted rather than assumed.

The problem is analytic and tiny: no XFOIL, so this runs in seconds and can
stay in the default suite where a regression will actually be seen.
"""

from __future__ import annotations

import numpy as np
import pytest

from aerobo.optimize import refusal as _refusal
from aerobo.optimize.constrained import run_bo_constrained

BOUNDS = np.array([[-2.0, 2.0], [-2.0, 2.0]])


def _problem(m: int):
    """A smooth constrained problem with ``m`` margins, feasible somewhere.

    Deliberately well-behaved: the point of these tests is that a HEALTHY
    problem never falls back. A pathological one would make an empty
    ``failures`` list uninformative — it could mean "the acquisition worked" or
    "the fallback happened to be fine".
    """
    def f_and_g(x):
        x = np.asarray(x, dtype=float)
        y = -float((x[0] - 0.4) ** 2 + (x[1] + 0.3) ** 2)
        g = [1.0 - float(x[0] ** 2 + x[1] ** 2),          # unit disc
             float(x[0]) + 1.5,                            # half-plane
             1.2 - float(abs(x[1]))][:m]                   # slab
        return y, g
    return f_and_g


def _failures(hist) -> list:
    """The fallback log, wherever this version keeps it."""
    got = getattr(hist, "failures", None)
    if got is None:
        got = (getattr(hist, "meta", None) or {}).get("failures")
    assert got is not None, (
        "the run reported no `failures` field at all — this test cannot tell a "
        "live acquisition from a random search without one")
    return list(got)


@pytest.mark.parametrize("m", [1, 2, 3])
@pytest.mark.parametrize("refusal", _refusal.MODES)
def test_the_acquisition_never_fell_back(m, refusal):
    """THE assertion this file exists for, over the branch and the encoding.

    ``m == 1`` and ``m > 1`` are separate code paths inside the same try, and
    the refusal encoding decides what the objective GP is fitted to.
    """
    hist = run_bo_constrained(_problem(m), BOUNDS, n_init=6, n_iter=6, seed=0,
                              refusal=refusal)
    failures = _failures(hist)
    assert failures == [], (
        f"the constrained acquisition FELL BACK to uniform random draws on "
        f"iterations {failures} (m={m}, refusal={refusal!r}). Those "
        f"evaluations are not Bayesian optimisation, and nothing else in the "
        f"result says so.")
    # …and the run really did the work we think it did
    assert hist.X.shape == (12, 2)
    assert hist.n_feasible >= 1


@pytest.mark.parametrize("m", [1, 3])
def test_a_warm_start_does_not_break_the_fit(m):
    """``x_init`` changes the training set the GP is fitted to, so it is an
    axis of the try-block and gets its own case."""
    x_init = np.array([[0.0, 0.0], [0.5, -0.2]])
    hist = run_bo_constrained(_problem(m), BOUNDS, n_init=6, n_iter=6, seed=1,
                              x_init=x_init)
    assert _failures(hist) == []
    assert hist.X.shape == (12, 2)
    # the seed points were evaluated FIRST and the Sobol draw shrank to match,
    # so warm starting stays budget-fair
    assert hist.X[0] == pytest.approx(x_init[0])
    assert hist.X[1] == pytest.approx(x_init[1])


@pytest.mark.parametrize("m", [1, 2])
def test_a_diagnostic_computation_that_raises_is_not_a_gp_failure(m,
                                                                 monkeypatch):
    """The SECOND of the two historical bugs, in the other direction.

    Computing the per-iteration diagnostics sits in a NESTED try whose comment
    promises they "must NEVER alter the search". The last time a diagnostic was
    allowed to reach the outer handler — a bare ``float(acq_val)`` that raises
    at q > 1 — every batched iteration silently drew at random while the
    acquisition was perfectly healthy (report §16.3d).

    So: break the diagnostics arithmetic specifically (``math.erf`` is used
    only in the p_feasible calculation inside that nested try) and assert the
    SEARCH is untouched. Breaking the callback instead would test something
    else — see the test below.
    """
    import aerobo.optimize.constrained as C

    seen = []
    real_erf = C.math.erf

    def exploding_erf(*a, **kw):
        seen.append(1)
        raise RuntimeError("diagnostic arithmetic blew up")

    monkeypatch.setattr(C.math, "erf", exploding_erf)
    hist = run_bo_constrained(_problem(m), BOUNDS, n_init=6, n_iter=6, seed=2,
                              iter_cb=lambda rec: None)
    monkeypatch.setattr(C.math, "erf", real_erf)

    assert seen, "the diagnostics block never ran, so this proves nothing"
    assert _failures(hist) == [], (
        "a raising DIAGNOSTIC computation was counted as a GP failure, so the "
        "search fell back to random draws for a reason that has nothing to do "
        "with the model")
    assert hist.X.shape == (12, 2)


def test_a_raising_iter_cb_propagates_and_that_is_the_contract():
    """The callback INVOCATION is outside both try blocks, so a caller whose
    ``iter_cb`` raises kills the run rather than silently degrading it.

    Pinned deliberately, because the alternative is worse in both directions:
    swallowing it would hide a broken consumer, and counting it as a GP failure
    would turn a healthy search into a random one. Losing the run is loud, and
    loud is the right failure here. If this is ever changed to keep the
    evaluations already paid for (repo memory: *stop is a stop rule*), this
    test is the one to update — deliberately, not by accident.
    """
    def exploding_cb(rec):
        raise RuntimeError("consumer blew up")

    with pytest.raises(RuntimeError, match="consumer blew up"):
        run_bo_constrained(_problem(2), BOUNDS, n_init=6, n_iter=6, seed=2,
                           iter_cb=exploding_cb)


def test_the_assertion_can_actually_fail(monkeypatch):
    """MUTATION CHECK — the rule this repo learned the hard way: a test that
    restates the code is not a test.

    Break the acquisition on purpose and confirm this file's central assertion
    goes red. Without this, `failures == []` might be passing because the field
    is always empty (never populated, renamed, or read off the wrong object),
    which is indistinguishable from success.
    """
    import aerobo.optimize.constrained as C

    def boom(*a, **kw):
        raise RuntimeError("acquisition is dead")

    monkeypatch.setattr(C, "LogConstrainedExpectedImprovement", boom)
    hist = run_bo_constrained(_problem(2), BOUNDS, n_init=6, n_iter=6, seed=0)
    failures = _failures(hist)
    assert failures, (
        "the acquisition was sabotaged and `failures` STILL came back empty — "
        "so the assertion in this file proves nothing about any run")
    assert len(failures) == 6
    # and the run still returned a usable result, which is why the fallback
    # exists at all and why it is so easy to miss
    assert hist.X.shape == (12, 2)
