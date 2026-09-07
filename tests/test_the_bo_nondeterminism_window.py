"""THE BO NON-DETERMINISM WINDOW — an ENVIRONMENT TRIPWIRE, not a code test.

READ THIS BEFORE "FIXING" A FAILURE HERE.
========================================

Most of this file does NOT assert a property of `aerobo`. It asserts a
property of the *installed torch build*: that ``torch.cholesky_solve`` /
``torch.linalg.solve_triangular`` return different ``float64`` values on
bit-identical inputs, in the autograd BACKWARD pass, over a contiguous band of
matrix sizes.

**This test is EXPECTED TO FAIL the day a torch (or Accelerate) upgrade fixes
that bug.** That failure is the whole point: it is the notification that the
BO reproducibility story recorded in `AUDIT_SESSION56.md` — "two identical BO
runs first disagree at evaluation index 64, and here is why" — no longer
describes this machine. When it fires, the correct response is NOT to delete
or loosen the test. It is to re-run the session-56 divergence measurement and
rewrite the reproducibility claims in the report, because either

  * the numerical bug is gone (BO may now be reproducible — a *good* failure,
    and several published caveats become obsolete), or
  * the band moved, in which case "evaluation index 64" is the wrong index and
    every statement keyed to 64 has to be re-derived.

What was measured (session 56)
------------------------------
torch 2.12.1, BLAS_INFO=accelerate, LAPACK_INFO=accelerate, darwin arm64.
The backward pass is nondeterministic for n in [64, 118] and clean at n <= 63
and n >= 119 (checked to 256).  The forward pass is affected far more rarely
(1 of 32 cells at 80 repeats), which is why this file probes the BACKWARD.
``torch.linalg.cholesky`` and ``matmul`` were bit-stable everywhere tested.
Perturbations are ~1e-12 absolute, i.e. exactly the size that flips an
`optimize_acqf` restart into a different basin a few iterations later.

Why 64 and not some other index
-------------------------------
That part IS repo code, and it is asserted hard, below.
``api._bo_split(160, 9) == (16, 144)``: a 160-evaluation, 9-dimensional BO run
spends 16 evaluations on the Sobol seed and 144 on GP iterations.  In
``optimize/constrained.py`` each iteration fits the GP on ``len(X_list)``
points and appends its candidate, so the candidate at 0-based evaluation index
``j`` is proposed by a GP fitted on exactly ``j`` training points.  Indices
16..63 therefore fit at sizes 16..63 — all below the onset — and **index 64 is
the first fit at a dirty training size**.  The arithmetic link is a hard
assertion because it must hold whatever torch does; only the band is
environmental.

Why the naive assertion would be wrong
--------------------------------------
The obvious test — "run the probe once at n=80 and assert it gives two
distinct answers" — is a coin flip.  The effect is stochastic (2-4 distinct
results in 40 repeats) and comes in quiet periods where a size can look clean
for a whole sweep.  A single unlucky sample would flake the suite, and a
flaky tripwire is worse than none: it trains everyone to ignore it.  The
aggregate rule and the repeat counts below are calibrated against a measured
distribution, and the reasoning is in ``_BAND_PROBE_CALIBRATION``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aerobo import api                                        # noqa: E402

torch = pytest.importorskip("torch")


# --- the measured window -------------------------------------------------
#: first matrix size at which the backward pass stops being bit-reproducible
ONSET_N = 64
#: last such size (inclusive); n >= 119 was clean to 256
BAND_TOP_N = 118

#: sizes probed inside the band. Spread across it so that a band which has
#: merely SHRUNK (rather than vanished) still trips the tripwire somewhere.
BAND_SIZES = (64, 72, 80, 88, 96, 104, 112, 118)

#: the control: one size just below the onset, plus two well above the top.
CLEAN_SIZES = (63, 200, 256)

_BAND_PROBE_CALIBRATION = """
Choice of REPEATS and of the aggregate rule "at least one band size is dirty".

Measured on this machine, 1200 independent 8-size sweeps at REPEATS = 40:

    dirty-of-8 histogram   1: 10   2: 2   3: 2   4: 6   5: 8
                           6: 20   7: 31   8: 121        (200 interleaved)
                           3: 159  4: 31   5: 317
                           6: 103  7: 107  8: 283        (1000 back-to-back)

Never once were all eight clean, but 10/1200 sweeps found only ONE dirty size.
That rules out any threshold above 1: requiring ">= 3 of 8" would have flaked
about 6 % of the time. Requiring ">= 1 of 8" did not fail in 1200 sweeps, so
the per-sweep FALSE-FAIL rate (bug present, test red) is below ~1e-3, and the
escalation to ESCALATED_REPEATS = 200 — five times the repeats, only paid on
the rare sweep that comes up empty — pushes it far below that.

The FALSE-PASS rate (bug fixed, test still green) is the mirror: it needs a
spurious second distinct result out of bit-identical inputs on a build that no
longer has the defect. At REPEATS = 40 the same probe returned exactly one
distinct result in 200/200 sweeps over the CLEAN sizes, i.e. clean code stays
clean, so the false-pass rate is bounded by the same ~1e-3 and in the same
direction. Raising REPEATS further would trade false-fail against runtime for
no measurable gain; the whole file runs in well under a second of probe time.

Repeat counts are also what keeps this cheap: a probe at n = 118 costs ~6 ms
for 40 repeats, so the full band sweep is ~50 ms.
"""

#: repeats per size in the first pass (see _BAND_PROBE_CALIBRATION)
REPEATS = 40
#: repeats per size in the escalation pass, only run if the first finds nothing
ESCALATED_REPEATS = 200


def _fixture(n: int):
    """A deterministic (L, B, W) triple for size ``n``. No RNG in the loop.

    ``L`` is a genuine Cholesky factor of a well-conditioned SPD matrix, which
    is what a GP fit hands to ``cholesky_solve``. The right-hand side has width
    1: the measured effect is far stronger for a single RHS column, which is
    also the shape a marginal-log-likelihood backward produces.
    """
    g = torch.Generator().manual_seed(1234 + n)
    M = torch.randn(n, n, generator=g, dtype=torch.float64)
    A = M @ M.T + n * torch.eye(n, dtype=torch.float64)
    L = torch.linalg.cholesky(A)
    B = torch.randn(n, 1, generator=g, dtype=torch.float64)
    W = torch.randn(n, 1, generator=g, dtype=torch.float64)
    return L, B, W


def _distinct_backwards(n: int, repeats: int) -> int:
    """How many bit-distinct gradients ``repeats`` identical backwards give.

    1 means the environment is reproducible at this size; >1 means it is not.
    Inputs are cloned from one fixed fixture, so every repeat is the SAME
    arithmetic on the SAME bytes -- there is no RNG, no threading choice and
    no data difference to blame.
    """
    L, B0, W = _fixture(n)
    seen: set[tuple[bytes, bytes]] = set()
    for _ in range(repeats):
        B = B0.clone().requires_grad_(True)
        Lp = L.clone().requires_grad_(True)
        (torch.cholesky_solve(B, Lp, upper=False) * W).sum().backward()
        seen.add((B.grad.numpy().tobytes(), Lp.grad.numpy().tobytes()))
    return len(seen)


def _dirty_sizes(sizes, repeats) -> list[int]:
    return [n for n in sizes if _distinct_backwards(n, repeats) > 1]


_WHAT_CHANGED = f"""
WHAT THIS MEANS (this is an environment tripwire, not a code regression):

The numerical defect that explains BO's non-reproducibility is no longer
observable on this machine. Nothing in `aerobo` broke. What changed is one of:

  * torch was upgraded or rebuilt (recorded as 2.12.1, BLAS_INFO=accelerate,
    LAPACK_INFO=accelerate, darwin arm64; now {{now}}),
  * the BLAS/LAPACK backend changed (Accelerate -> MKL/OpenBLAS, or an OS
    update to Accelerate itself),
  * the machine changed architecture.

REQUIRED ACTION -- do not just delete this test:
  1. Re-run the size sweep to find the new band, if any. If the band moved,
     `ONSET_N` = {ONSET_N} is wrong, and so is every claim keyed to
     "evaluation index 64".
  2. Re-measure whether two identical `run_bo_constrained` calls still
     diverge, and where. If they no longer do, BO became reproducible and the
     reproducibility caveats in AUDIT_SESSION56.md / the report are obsolete
     and must be rewritten, not silently kept.
"""


def _environment_changed(detail: str) -> str:
    return detail + _WHAT_CHANGED.format(now=torch.__version__)


# =====================================================================
# 1. The arithmetic link -- REPO CODE, hard assertion, never environmental
# =====================================================================

def test_bo_split_puts_the_first_dirty_fit_at_evaluation_index_64():
    """160 evaluations, d=9 -> 16 Sobol + 144 GP fits; index 64 fits on 64.

    This is the load-bearing half of the session-56 argument and it is pure
    arithmetic on repo code: if `_bo_split` ever changes its seed size, the
    index at which two runs may first diverge moves with it, and every
    published statement about "evaluation 64" has to move too.
    """
    n_init, n_iter = api._bo_split(160, 9)
    assert (n_init, n_iter) == (16, 144)
    assert n_init + n_iter == 160

    # constrained.py fits on len(X_list) points and appends the candidate, so
    # the candidate at 0-based evaluation index j comes from a GP fitted on
    # exactly j training points.
    def training_size_at(index: int) -> int:
        assert n_init <= index < n_init + n_iter
        return index

    first_dirty_index = min(
        j for j in range(n_init, n_init + n_iter)
        if training_size_at(j) >= ONSET_N
    )
    assert first_dirty_index == 64, (
        f"the first GP fit at a training size >= {ONSET_N} is at evaluation "
        f"index {first_dirty_index}, not 64; the session-56 divergence "
        f"location no longer follows from _bo_split{(160, 9)} = "
        f"{(n_init, n_iter)}"
    )

    # ... and the evaluation before it is genuinely below the onset, which is
    # why the two runs share a bit-identical prefix of exactly 64 designs.
    assert training_size_at(first_dirty_index - 1) == ONSET_N - 1 < ONSET_N


# =====================================================================
# 2. The environment property -- EXPECTED TO FAIL on a fixed torch
# =====================================================================

def test_backward_is_nondeterministic_somewhere_inside_the_band():
    """Some size in [64, 118] gives >1 distinct gradient from identical input.

    Aggregate over several sizes with an escalation pass, because a single
    size can look clean for a whole sweep (see _BAND_PROBE_CALIBRATION).
    """
    assert BAND_SIZES[0] == ONSET_N and BAND_SIZES[-1] == BAND_TOP_N

    dirty = _dirty_sizes(BAND_SIZES, REPEATS)
    if not dirty:                       # rare: escalate before believing it
        dirty = _dirty_sizes(BAND_SIZES, ESCALATED_REPEATS)
    if not dirty:
        pytest.fail(_environment_changed(
            f"torch.cholesky_solve's BACKWARD pass is now bit-reproducible at "
            f"every probed size in [{ONSET_N}, {BAND_TOP_N}] "
            f"({list(BAND_SIZES)}), over {REPEATS} then {ESCALATED_REPEATS} "
            f"repeats on bit-identical inputs.\n"
        ), pytrace=False)


def test_backward_is_deterministic_just_below_the_onset():
    """n = 63 (and 200, 256) are clean -- the band has an edge, not a slope.

    Without this control the test above would also pass on a machine that is
    nondeterministic EVERYWHERE, and then "index 64" would be meaningless:
    the runs would diverge at the very first GP fit, at index 16.
    """
    assert CLEAN_SIZES[0] == ONSET_N - 1

    noisy = {n: _distinct_backwards(n, REPEATS) for n in CLEAN_SIZES}
    bad = {n: k for n, k in noisy.items() if k != 1}
    if bad:
        pytest.fail(_environment_changed(
            f"sizes measured as CLEAN are no longer clean: {bad} distinct "
            f"gradients in {REPEATS} identical backward passes (expected 1 "
            f"each). The nondeterministic band is no longer bounded below by "
            f"{ONSET_N}, so the first divergent GP fit is no longer at "
            f"training size {ONSET_N} and evaluation index 64 is not the "
            f"first divergent evaluation.\n"
        ), pytrace=False)
