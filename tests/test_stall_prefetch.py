"""The stall sweep runs BESIDE the cruise sweep, and changes nothing.

``composite_evaluation`` needs two XFOIL polars per candidate — the cruise
sweep (fired inside ``evaluate_airfoil``) and the wide stall sweep. They are
independent subprocess calls on the same coordinates, so the second is started
speculatively and joined only where its result is read.

What these tests pin is that the speedup is free:

* the same J, from the same calls, with the switch either way;
* a design ``evaluate_airfoil`` refuses BEFORE the solver never starts a sweep
  (the contract ``xfoil_must_not_run`` callers rely on);
* a candidate rejected on its margins returns without waiting for the sweep it
  speculatively started;
* a prefetch that raises costs time, never correctness.
"""
import threading
import time

import numpy as np
import pytest

from aerobo import airfoil_select as A
from aerobo import xfoil_run
from aerobo.airfoil import AirfoilProblem, evaluate_airfoil, xfoil_precheck
from aerobo.airfoil_select import (
    ScoreWeights,
    composite_evaluation,
    load_screen_reference,
)
from aerobo.xfoil_run import XfoilPolarResult


@pytest.fixture()
def prob():
    return AirfoilProblem()


@pytest.fixture()
def reference():
    return load_screen_reference()


@pytest.fixture()
def weights():
    return ScoreWeights()


def _polar(alphas, cm=-0.04):
    a = np.asarray(alphas, dtype=float)
    cl = (1.6 - 0.01 * (a - 14.0) ** 2) if a.max() > 10.5 else 0.25 + 0.11 * a
    return XfoilPolarResult(alpha_deg=a, cl=cl, cd=0.006 + 1e-4 * (a - 1.0) ** 2,
                            cm=np.full_like(a, cm), n_requested=a.size)


def _install(monkeypatch, fake):
    monkeypatch.setattr(xfoil_run, "run_xfoil_polar", fake)


def _x0(prob):
    """A design in the middle of the box — geometry the precheck accepts."""
    b = np.asarray(prob.bounds, dtype=float)
    return 0.5 * (b[:, 0] + b[:, 1])


# ------------------------------------------------ the precheck IS the gate

def test_precheck_agrees_with_the_evaluation_it_was_split_from(prob, monkeypatch):
    """Every reason evaluate_airfoil refuses without XFOIL, xfoil_precheck
    gives — and for a design it accepts, the solver is reached."""
    def boom(*a, **k):  # pragma: no cover - being called IS the failure
        raise AssertionError("run_xfoil_polar called for a refused design")

    _install(monkeypatch, boom)
    b = np.asarray(prob.bounds, dtype=float)

    bad_shape = np.zeros(b.shape[0] + 1)
    assert xfoil_precheck(bad_shape, prob) is not None
    assert evaluate_airfoil(bad_shape, prob)["feasible"] is False

    out_of_box = b[:, 1] + 1.0
    assert xfoil_precheck(out_of_box, prob) == "bounds violation"
    assert evaluate_airfoil(out_of_box, prob)["feasible"] is False

    # Upper surface driven below the lower one: self-intersecting. The old
    # form built this vector from the BOX CORNERS and then guarded it with
    # `if xfoil_precheck(crossed, prob) is not None:` — but the floor/cap
    # (W_UPPER_FLOOR +0.05 / W_LOWER_CAP -0.02) PROVE no in-box crossing, so
    # that branch was dead and deleting the gap check from xfoil_precheck
    # left this test green while the prefetcher started XFOIL on coordinates
    # nobody can panel. Both halves of the proof are now asserted outright.
    thinnest = _x0(prob).copy()
    thinnest[: prob.n_cst] = b[: prob.n_cst, 0]
    thinnest[prob.n_cst:] = b[prob.n_cst:, 1]
    assert xfoil_precheck(thinnest, prob) is None, (
        "the thickness-reducing corner of the box must still be solvable — "
        "the floor/cap proof in the AirfoilProblem docstring")

    # ...and a genuinely crossed pair, reachable only with the box widened
    # (the recipe test_airfoil_problem.test_geometry_precheck_rejects_crossing
    # uses), is refused BY NAME and never reaches the solver (`boom`).
    wide = AirfoilProblem()
    wide._bounds = np.column_stack([np.full(wide.dim, -10.0),
                                    np.full(wide.dim, 10.0)])
    crossed = np.concatenate([np.full(wide.n_cst, 0.05),      # upper
                              np.full(wide.n_cst, 0.30)])     # lower, ABOVE it
    reason = xfoil_precheck(crossed, wide)
    assert reason is not None and "self-intersecting" in reason, reason
    assert evaluate_airfoil(crossed, wide)["feasible"] is False

    assert xfoil_precheck(_x0(prob), prob) is None


def test_a_refused_design_never_starts_a_sweep(prob, reference, weights,
                                               monkeypatch):
    """The speculative sweep must not fire ahead of the geometry gate."""
    def boom(*a, **k):  # pragma: no cover - being called IS the failure
        raise AssertionError("run_xfoil_polar called for a refused design")

    _install(monkeypatch, boom)
    out_of_box = np.asarray(prob.bounds, dtype=float)[:, 1] + 1.0
    res = composite_evaluation(out_of_box, prob, reference, weights)
    assert res["feasible"] is False
    assert res["composite"] is None


# ------------------------------------------------ the answer does not move

def test_the_switch_does_not_change_the_result(prob, reference, weights,
                                               monkeypatch):
    calls = []

    def fake(coords, re, mach, alphas, **kw):
        calls.append(tuple(np.round(np.asarray(alphas, float), 4)))
        return _polar(alphas)

    _install(monkeypatch, fake)
    x = _x0(prob)

    monkeypatch.setattr(A, "PREFETCH_STALL_SWEEP", False)
    calls.clear()
    serial = composite_evaluation(x, prob, reference, weights)
    serial_calls = sorted(calls)

    monkeypatch.setattr(A, "PREFETCH_STALL_SWEEP", True)
    calls.clear()
    parallel = composite_evaluation(x, prob, reference, weights)

    assert sorted(calls) == serial_calls          # same sweeps, same arguments
    assert serial["composite"] == parallel["composite"]
    assert serial["scores"] == parallel["scores"]
    for k in ("clmax", "astall", "ldmax", "ldcr", "f", "score"):
        assert serial[k] == parallel[k], k


def test_a_failed_prefetch_still_scores(prob, reference, weights, monkeypatch):
    """The stall sweep raises only in the side thread; the caller re-runs it."""
    state = {"raised": False}

    def fake(coords, re, mach, alphas, **kw):
        if np.asarray(alphas, float).max() > 10.5 and kw.get("concurrent"):
            state["raised"] = True
            raise RuntimeError("prefetch exploded")
        return _polar(alphas)

    _install(monkeypatch, fake)
    res = composite_evaluation(_x0(prob), prob, reference, weights)
    assert state["raised"] is True
    assert res["composite"] is not None
    assert res["feasible"] is True


# ------------------------------------------------ a rejection does not wait

def test_a_rejected_candidate_does_not_wait_for_the_sweep(prob, reference,
                                                          weights, monkeypatch):
    """A candidate refused on its margins returns while the speculative stall
    sweep is still running — the gate-first saving that is actually worth
    keeping."""
    started = threading.Event()
    release = threading.Event()

    def fake(coords, re, mach, alphas, **kw):
        if np.asarray(alphas, float).max() > 10.5:
            started.set()
            release.wait(timeout=10.0)      # hold the stall sweep open
            return _polar(alphas)
        return _polar(alphas, cm=-0.5)      # |cm| 0.5 > cm_max: margin refused

    _install(monkeypatch, fake)
    t0 = time.perf_counter()
    res = composite_evaluation(_x0(prob), prob, reference, weights)
    elapsed = time.perf_counter() - t0
    release.set()

    assert started.is_set()                 # it WAS started speculatively
    assert res["composite"] is None
    assert "infeasible" in res["reason"]
    assert elapsed < 5.0                    # and it was not waited on


def test_speculation_is_capped_and_falls_back_in_line(prob, reference, weights,
                                                      monkeypatch):
    """With every slot held, the sweep is not speculated — it runs in line, and
    the answer is the same. An unbounded speculator would let the number of live
    XFOIL subprocesses be set by how fast candidates are refused."""
    fired = []

    def fake(coords, re, mach, alphas, **kw):
        if np.asarray(alphas, float).max() > 10.5:
            fired.append(kw.get("concurrent"))
        return _polar(alphas)

    _install(monkeypatch, fake)
    for _ in range(A.MAX_INFLIGHT_STALL_SWEEPS):     # hold every slot
        assert A._STALL_SLOTS.acquire(blocking=False)
    try:
        res = composite_evaluation(_x0(prob), prob, reference, weights)
    finally:
        for _ in range(A.MAX_INFLIGHT_STALL_SWEEPS):
            A._STALL_SLOTS.release()

    assert res["composite"] is not None              # still scored
    assert fired and not any(fired)                  # in line, not speculated


def test_a_finished_sweep_returns_its_slot(prob, reference, weights, monkeypatch):
    """The semaphore must not leak: N evaluations in a row must not exhaust it."""
    _install(monkeypatch, lambda coords, re, mach, alphas, **kw: _polar(alphas))
    for _ in range(A.MAX_INFLIGHT_STALL_SWEEPS + 3):
        assert composite_evaluation(
            _x0(prob), prob, reference, weights)["composite"] is not None
    for _ in range(A.MAX_INFLIGHT_STALL_SWEEPS):     # every slot free again
        assert A._STALL_SLOTS.acquire(blocking=False)
    for _ in range(A.MAX_INFLIGHT_STALL_SWEEPS):
        A._STALL_SLOTS.release()


def test_the_sweep_is_declared_concurrent(prob, reference, weights, monkeypatch):
    """A wall-clock timeout measured beside another sweep must never be cached
    as a property of the section (xfoil_run.run_xfoil_polar)."""
    seen = {}

    def fake(coords, re, mach, alphas, **kw):
        if np.asarray(alphas, float).max() > 10.5:
            seen["concurrent"] = kw.get("concurrent")
        return _polar(alphas)

    _install(monkeypatch, fake)
    composite_evaluation(_x0(prob), prob, reference, weights)
    assert seen.get("concurrent") is True
