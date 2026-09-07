"""The screen may skip a draw it can refuse for free — and nothing else.

``SizeGateMargin.excess(x) > 0`` is computable from the design vector with no
physics: it means the vector's own aspect ratio or wing loading is outside a
gate that refuses before any solver runs. So it is a SUFFICIENT condition for
a refusal, and a pool draw that trips it can be skipped without paying for
it. Measured over 1536 draws on each of two live boxes, every gated draw was
also refused (742/742 and 405/405, zero false positives).

This is a filter on a POOL, never a refusal. The pool is 32x the design, and
the top-up loop still evaluates gated draws on demand if the pool cannot fill
the initial design — so the pre-filter is only allowed to change the COST.

**What this file deliberately does NOT claim.** An earlier proposal was to
spend evaluations making the initial design FEASIBLE, on the theory that a
17.3 %-feasible box starves the seed. Three budget-fair A/B runs at budget 72
say the opposite: replacing acquisition evaluations with screen draws loses
0/6 seeds and 1.9 payload-L/D of median on the d=20 box, and 4 losses / 1 win
/ 1 tie on the one box whose refused fraction is low enough for the screen to
bite. At a fixed budget an evaluation spent on a feasible seed is worth less
than the same evaluation spent in the acquisition or the polish. Only the
free half of that idea ships.

**Not covered here.** ``deferred`` is walked in Sobol order so that a starved
pool returns the same design the ungated screen would. On this box that
ordering is never exercised: every non-gated draw SOLVES (kept 7, gated 5,
refused 0 at seed 1), so the deferred list is homogeneous and a mutation that
reorders it is inert. The property is real and the code is written for it; a
box with both gated AND refused draws would be needed to pin it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aerobo import api, sizing                                   # noqa: E402
from aerobo.optimize import feasible as F                        # noqa: E402

#: a live box whose size gate actually fires (48 % of pool draws on the
#: measurement that motivated this), found by prefix so the registry may go
#: on generating twins without this file pinning one spelling.
_NAME = next(n for n in api.PROBLEM_SPECS if n.startswith(
    "tail + winglet (span-capped) [designed tail + tip device] "
    "+ free planform"))


def _fixture():
    spec = api.PROBLEM_SPECS[_NAME]
    prob = spec.build({}, {}, None)
    gate = F.SizeGateMargin(
        list(spec.param_labels), ar_limits=sizing.AR_LIMITS,
        cap_pa=getattr(prob, "wing_loading_max_Pa", None),
        weight_n=getattr(prob, "W_fixed_N", None))
    return prob, np.asarray(prob.bounds, dtype=float), gate


def _counting_fg(prob, box):
    box["n"] = 0

    def fg(x):
        box["n"] += 1
        out = prob.evaluate(np.asarray(x, dtype=float))
        if not out.get("feasible"):
            return F.PENALTY, np.full(2, -1.0)
        return float(out["score"]), np.asarray(out["g"], dtype=float)
    return fg


def test_the_gate_fires_on_this_box_at_all():
    """The control for every assertion below: a box whose gate never fires
    would pass them all for the wrong reason."""
    prob, bounds, gate = _fixture()
    assert gate.live
    pool = F.sobol_pool(bounds, 512, 0)
    gated = sum(1 for x in pool if gate.excess(x) > 0.0)
    assert 0.10 < gated / len(pool) < 0.90, gated / len(pool)


def test_every_gated_draw_really_is_refused():
    """The soundness condition. If a gated draw could SOLVE, the pre-filter
    would be throwing away designs rather than saving calls."""
    prob, bounds, gate = _fixture()
    pool = [x for x in F.sobol_pool(bounds, 192, 7) if gate.excess(x) > 0.0]
    assert pool, "no gated draws to check"
    for x in pool:
        out = prob.evaluate(np.asarray(x, dtype=float))
        assert not out.get("feasible"), (
            f"gate refused a draw the solver accepted: {out.get('reason')}")


def test_the_prefilter_changes_cost_and_not_the_answer():
    """The contract, both halves, from one pair of runs."""
    prob, bounds, gate = _fixture()
    off_box, on_box = {}, {}
    off = F.screened_init(_counting_fg(prob, off_box), bounds, 16, seed=0,
                          constrained=True, gate=None)
    on = F.screened_init(_counting_fg(prob, on_box), bounds, 16, seed=0,
                         constrained=True, gate=gate)

    # the ANSWER: the same initial design, point for point
    assert np.array_equal(np.asarray(off[0]), np.asarray(on[0]))
    # ...and the same values. NOT bitwise: two identical gate=None runs of
    # this evaluator already differ by ~2e-16 (BLAS reduction order), so a
    # bit-equality assertion here would be flaky for a reason that has
    # nothing to do with the pre-filter.
    assert np.allclose(np.asarray(off[1]), np.asarray(on[1]),
                       rtol=1e-12, atol=0.0, equal_nan=True)
    assert off[3] == on[3]              # the screened count is the same

    # the COST: strictly fewer physics calls
    assert on_box["n"] < off_box["n"], (on_box["n"], off_box["n"])


def test_no_gate_is_bit_for_bit_the_published_path():
    """``gate=None`` must leave the shipped screen exactly as it was — the
    pre-filter is opt-in at the call site, not a change of default."""
    prob, bounds, _ = _fixture()
    box = {}
    a = F.screened_init(_counting_fg(prob, box), bounds, 8, seed=3,
                        constrained=True, gate=None)
    n_a = box["n"]
    b = F.screened_init(_counting_fg(prob, box), bounds, 8, seed=3,
                        constrained=True)
    assert np.array_equal(np.asarray(a[0]), np.asarray(b[0]))
    assert box["n"] == n_a


def test_a_gated_pool_can_still_fill_the_design():
    """The property the top-up loop protects: if the gate skipped so much
    that the pool cannot fill the initial design, the gated draws are
    evaluated on demand rather than the design coming back short."""
    prob, bounds, gate = _fixture()
    n_init = 12
    # pool=1 is the starved case ON PURPOSE: 12 draws for 12 wanted points,
    # roughly half of them gated, so the design CANNOT be filled without
    # coming back for them. At the shipped pool of 32 the non-gated draws
    # fill it on their own and this path is never taken — which is exactly
    # how a broken top-up would go unnoticed.
    box = {}
    got = F.screened_init(_counting_fg(prob, box), bounds, n_init, seed=1,
                          constrained=True, gate=gate, pool=1)
    assert len(got[0]) == n_init, "the gated draws were never come back for"
    assert len(got[1]) == n_init

    # ...and it fills it with the SAME points the ungated screen would, so
    # the starved path is still only a cost difference
    ref = {}
    base = F.screened_init(_counting_fg(prob, ref), bounds, n_init, seed=1,
                           constrained=True, gate=None, pool=1)
    assert np.array_equal(np.asarray(base[0]), np.asarray(got[0]))


def test_the_evaluator_is_not_bitwise_reproducible():
    """Recorded because it bounds every equality assertion in this repo: the
    same design evaluated twice differs in the last bits. Any test that
    demands bit-identity of a SCORE is flaky, and the tolerance above is
    chosen four orders of magnitude above this."""
    prob, bounds, _ = _fixture()
    x = (bounds[:, 0] + bounds[:, 1]) / 2.0
    a = prob.evaluate(x)
    b = prob.evaluate(x)
    assert a["score"] == pytest.approx(b["score"], rel=1e-12)
