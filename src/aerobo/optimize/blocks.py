"""Block-coordinate ("portfolio") constrained optimiser.

Alternating Gauss-Seidel over declared variable BLOCKS: optimise one block
with the others frozen at the incumbent, accept the block's best point,
move to the next block, repeat for a number of cycles. Each block names its
own sub-optimiser from the constrained registry — the report §13
prescription made executable: give every variable group the search method
its evaluation cost/structure favours (e.g. BO for an XFOIL-expensive
section block, anything cheap for a millisecond-VLM planform block whose
polar is already cached).

Budget contract
---------------
``n_evals`` counts UNIQUE physics evaluations. Everything the runner does
internally that re-visits a known point — the incumbent warm start, the
acceptance re-check, a cycle-to-cycle warm start — is served by the memo
and costs nothing, so it must not consume budget either. Two consequences,
both of which the first implementation got wrong and are now gated by
tests:

* sub-run budgets are stated in FRESH evaluations, and warm-start rows are
  added ON TOP of them (``run_bo_constrained`` charges ``x_init`` rows
  against ``n_init``, and those rows are memo hits here);
* whatever is left over after the planned cycles is re-spent in a TOP-UP
  pass, so a caller asking for 40 evaluations gets 40, not 28. Without it
  a blocks-vs-BO comparison at equal nominal budget is not budget-fair,
  which is the house rule this project's crossover claims rest on.

Search quality
--------------
* ``n_iter`` is RESERVED before ``n_init`` is sized. The natural-looking
  ``n_init = min(2 d_b, budget - 1)`` silently degrades an 8-D block to a
  Sobol DOE plus a single acquisition step for every per-block budget below
  ~17 — throwing away exactly the BO-over-random margin (results/airfoil.json:
  54.6 vs 59.0 drag counts on this very section block) that motivates using
  BO there at all.
* Each block re-uses everything already evaluated for the SAME sub-problem
  as GP warm-start data. "Same sub-problem" is exact: only rows whose
  frozen (complementary) coordinates equal the current incumbent's qualify.
  Once another block moves, its old rows describe a different function and
  are correctly dropped — Gauss-Seidel amortises where it legitimately can
  and nowhere else.

Feasibility
-----------
* Blocks may declare ``owns_constraints``; when the incumbent is infeasible
  those blocks run FIRST, since no other block can repair the violation.
* If nothing feasible is known, the runner accepts the least-violating
  point found (strictly smaller max-violation). Without this an infeasible
  start can freeze the incumbent for the whole run: every sub-result is
  rejected, every cycle restarts from the same point, and the entire budget
  is spent going nowhere.
"""

from __future__ import annotations

import numpy as np

from . import feasible as _feasible
from .constrained import (
    ConstrainedRunHistory,
    make_constrained_history,
    run_bo_constrained,
    run_ga_constrained,
    REGISTRY_CONSTRAINED,
)

#: a sub-run below this many fresh evaluations cannot say anything useful
MIN_BLOCK_EVALS = 3

#: fraction of a BO sub-run's budget reserved for acquisition steps (the
#: rest seeds the GP) — see "Search quality" above
BO_ACQ_FRACTION = 0.4


def _compose(x_base: np.ndarray, idx: np.ndarray, xb: np.ndarray
             ) -> np.ndarray:
    x = x_base.copy()
    x[idx] = np.asarray(xb, dtype=float)
    return x


def _violation(g: np.ndarray) -> float:
    """Max constraint violation (0.0 when feasible)."""
    return float(np.max(np.maximum(0.0, -np.atleast_1d(g))))


def _validate_blocks(blocks, d: int) -> None:
    if not blocks:
        raise ValueError("run_blocks_constrained needs a non-empty `blocks` "
                         "declaration (indices + optimiser per block)")
    idx_all = sorted(i for b in blocks for i in b["indices"])
    if idx_all != list(range(d)):
        raise ValueError(f"blocks must partition all {d} variables exactly; "
                         f"got {idx_all}")
    for b in blocks:
        name = b.get("optimiser", "bo")
        if name not in REGISTRY_CONSTRAINED:
            raise ValueError(
                f"block {b.get('name', '?')!r} names optimiser {name!r}, "
                f"which is not in the constrained registry "
                f"{sorted(REGISTRY_CONSTRAINED)} — validated up front so a "
                f"typo cannot surface after minutes of physics")
        share = float(b.get("share", 1.0))
        if not np.isfinite(share) or share <= 0.0:
            raise ValueError(f"block {b.get('name', '?')!r} has share "
                             f"{share!r}; shares must be finite and > 0")


def run_blocks_constrained(
    f_and_g,
    bounds: np.ndarray,
    n_evals: int,
    seed: int = 0,
    blocks: tuple | list = (),
    x0: np.ndarray | None = None,
    cycles: int = 2,
    warm_start: bool = True,
    top_up: bool = True,
    feasibility: str = _feasible.DEFAULT_MODE,
) -> ConstrainedRunHistory:
    """Alternating block-coordinate search; maximise f s.t. all g >= 0.

    ``blocks``: iterable of dicts with ``indices`` (variable positions),
    ``optimiser`` (a constrained-registry name), optional ``share`` (budget
    weight, default equal), ``owns_constraints`` (bool) and ``cost``
    (free-form annotation carried into the history meta). ``x0`` is the
    starting incumbent (defaults to the box midpoint). ``n_evals`` counts
    UNIQUE physics evaluations across all block runs.

    ``warm_start`` re-uses same-sub-problem evaluations as GP seed data;
    ``top_up`` re-spends any budget the planned cycles left over. Both
    default ON; setting both False reproduces the plain
    one-pass-per-cycle behaviour.
    """
    bounds = np.asarray(bounds, dtype=float)
    d = bounds.shape[0]
    _validate_blocks(blocks, d)
    n_evals = int(n_evals)
    x_cur = (np.asarray(x0, dtype=float).copy() if x0 is not None
             else 0.5 * (bounds[:, 0] + bounds[:, 1]))
    x_cur = np.clip(x_cur, bounds[:, 0], bounds[:, 1])

    # ---- memoised, logged evaluation ------------------------------------
    X_log: list[np.ndarray] = []
    y_log: list[float] = []
    g_log: list[np.ndarray] = []
    memo: dict[tuple, tuple[float, np.ndarray]] = {}

    def fg(x_full: np.ndarray) -> tuple[float, np.ndarray]:
        key = tuple(np.round(np.asarray(x_full, dtype=float), 12))
        if key in memo:
            return memo[key]
        f, g = f_and_g(np.asarray(x_full, dtype=float))
        g = np.atleast_1d(np.asarray(g, dtype=float))
        memo[key] = (float(f), g)
        X_log.append(np.asarray(x_full, dtype=float).copy())
        y_log.append(float(f))
        g_log.append(g)
        return memo[key]

    def spent() -> int:
        return len(X_log)

    # seed the incumbent (1 eval) so every block starts from a scored point
    f_cur, g_cur = fg(x_cur)
    viol_cur = _violation(g_cur)
    feas_cur = viol_cur <= 0.0

    n_blocks = len(blocks)
    shares = np.array([float(b.get("share", 1.0)) for b in blocks])
    shares = shares / shares.sum()
    per_run = [max(MIN_BLOCK_EVALS,
                   int(round((n_evals - 1) * s / max(1, cycles))))
               for s in shares]

    spend = [0] * n_blocks
    skipped: list[dict] = []

    def block_rows(idx: np.ndarray, x_ref: np.ndarray):
        """(Xb, y, G) already evaluated for THIS block's sub-problem, i.e.
        rows whose frozen coordinates equal the reference incumbent's."""
        other = np.ones(d, dtype=bool)
        other[idx] = False
        sel = [k for k in range(len(X_log))
               if np.array_equal(X_log[k][other], x_ref[other])]
        if not sel:
            return np.empty((0, idx.size)), np.empty(0), []
        return (np.array([X_log[k][idx] for k in sel]),
                np.array([y_log[k] for k in sel]),
                [g_log[k] for k in sel])

    def run_block(bi: int, budget_b: int, tag: str) -> None:
        """One sub-optimiser run on block ``bi``; updates the incumbent."""
        nonlocal x_cur, f_cur, g_cur, feas_cur, viol_cur
        blk = blocks[bi]
        idx = np.asarray(sorted(blk["indices"]), dtype=int)
        sub_bounds = bounds[idx]
        before = spent()
        sub_seed = seed + 1000 * cycle_of[tag] + 17 * bi

        def sub_fg(xb, _idx=idx):
            return fg(_compose(x_cur, _idx, xb))

        # warm start: every point already evaluated for this exact
        # sub-problem (memo hits, so they cost no budget)
        Xb_prev, _, _ = block_rows(idx, x_cur)
        seeds = np.vstack([x_cur[idx][None, :], Xb_prev]) if warm_start \
            else x_cur[idx][None, :]
        seeds = np.unique(np.round(seeds, 12), axis=0) if seeds.size else seeds

        name = blk.get("optimiser", "bo")
        if name == "bo":
            k = seeds.shape[0]
            # reserve acquisition steps BEFORE sizing the Sobol seed, and
            # count only FRESH evaluations against the block budget
            n_iter = max(1, int(np.ceil(BO_ACQ_FRACTION * budget_b)))
            n_init_fresh = max(1, min(2 * idx.size, budget_b - n_iter))
            n_iter = max(1, budget_b - n_init_fresh)
            res = run_bo_constrained(
                sub_fg, sub_bounds, n_init=k + n_init_fresh, n_iter=n_iter,
                seed=sub_seed, x_init=seeds, feasibility=feasibility)
        elif name == "ga":
            res = run_ga_constrained(
                sub_fg, sub_bounds, budget_b, seed=sub_seed,
                pop_size=min(20, max(4, budget_b // 3)))
        else:
            res = REGISTRY_CONSTRAINED[name](
                sub_fg, sub_bounds, budget_b, seed=sub_seed)
        del res            # the incumbent update reads the shared log below
        spend[bi] += spent() - before

        # ---- acceptance, over every row of THIS sub-problem -------------
        Xb, yb, Gb = block_rows(idx, x_cur)
        if Xb.shape[0] == 0:
            return
        viol = np.array([_violation(g) for g in Gb])
        feas = viol <= 0.0
        if feas.any():
            cand = int(np.arange(Xb.shape[0])[feas][np.argmax(yb[feas])])
            better = (not feas_cur) or (yb[cand] > f_cur)
        else:
            # nothing feasible anywhere in this sub-problem: repair instead,
            # otherwise an infeasible incumbent can never move
            cand = int(np.argmin(viol))
            better = (not feas_cur) and (viol[cand] < viol_cur)
        if not better:
            return
        x_new = _compose(x_cur, idx, Xb[cand])
        f_new, g_new = fg(x_new)             # memo hit — no extra eval
        x_cur, f_cur, g_cur = x_new, f_new, g_new
        viol_cur = _violation(g_cur)
        feas_cur = viol_cur <= 0.0

    cycle_of: dict[str, int] = {}

    def order_for_cycle() -> list[int]:
        """Constraint-owning blocks first while the incumbent is infeasible
        — no other block can repair a violation it does not control."""
        ids = list(range(n_blocks))
        if feas_cur:
            return ids
        return sorted(ids, key=lambda i:
                      not bool(blocks[i].get("owns_constraints", False)))

    for cyc in range(cycles):
        for bi in order_for_cycle():
            residual = n_evals - spent()
            if residual < MIN_BLOCK_EVALS:
                break
            budget_b = min(per_run[bi], residual)
            if budget_b < MIN_BLOCK_EVALS:
                skipped.append({"block": blocks[bi].get("name", f"block{bi}"),
                                "cycle": cyc, "budget": int(budget_b)})
                continue
            tag = f"c{cyc}b{bi}"
            cycle_of[tag] = cyc
            run_block(bi, budget_b, tag)

    # ---- top-up: re-spend whatever the planned cycles left over ---------
    round_i = 0
    while top_up and (n_evals - spent()) >= MIN_BLOCK_EVALS:
        progressed = False
        for bi in np.argsort(-shares):
            residual = n_evals - spent()
            if residual < MIN_BLOCK_EVALS:
                break
            tag = f"t{round_i}b{bi}"
            cycle_of[tag] = cycles + round_i
            before = spent()
            run_block(int(bi), int(min(per_run[bi], residual)), tag)
            progressed |= spent() > before
        round_i += 1
        if not progressed:                   # every block is saturated
            break

    # ---- global history over unique evals in call order -----------------
    X = np.vstack(X_log)
    y = np.asarray(y_log, dtype=float)
    G = np.vstack([np.atleast_1d(g) for g in g_log])
    return make_constrained_history(
        X, y, G, name="blocks", seed=seed,
        cycles=cycles,
        top_up_rounds=round_i,
        blocks=[{"name": b.get("name", f"block{i}"),
                 "optimiser": b.get("optimiser", "bo"),
                 "indices": list(b["indices"]),
                 "share": float(b.get("share", 1.0)),
                 "cost": b.get("cost"),
                 "owns_constraints": bool(b.get("owns_constraints", False)),
                 "evals": int(spend[i])}
                for i, b in enumerate(blocks)],
        skipped=skipped,
    )
