"""Run one optimiser, then spend what is left on another seeded at its best.

MEASURED, not assumed. ``RESULTS_HANDOFF.md`` scored 1,558 budget-fair runs
over 13 certified wing cases and found the direction is what governs a
handoff: BO is the best optimiser to hand off FROM and the worst to hand off
TO, because truncating BO throws away the surrogate it spent the run building.
Of the eight pairs measured, ``bo -> slsqp`` is the only one that beats both
its parents pairwise, and on the 53 runs where every constrained arm exists it
takes the top three places by median gap at all three splits.

What is shipped here is exactly the procedure the study ran
(``scripts/handoff_study.py:handoff``), not a re-derivation of it: the same
budget split, the same seed for both phases, and the same rule that what
crosses the handoff is A's best FEASIBLE design. A shipped default that
differed from the measured arm in any of those would be selling a result
nobody collected.

Two things the study did NOT measure, both marked in the meta rather than
hidden:

* ``handed_over=False`` — BO found no feasible design inside its share, so
  there was nothing to seed SLSQP with. The study recorded that outcome as
  CENSORED and stopped. Stopping would strand the rest of the user's budget,
  so SLSQP runs unseeded instead, which is a pure-SLSQP tail and not the
  measured arm. It is recorded, never inferred: see
  ``a-silent-fallback-hides-a-dead-acquisition``.
* any phi other than the three the study ran (0.25, 0.50, 0.75).
"""

from __future__ import annotations

import numpy as np

from .constrained import (ConstrainedRunHistory, make_constrained_history,
                          run_bo_constrained, run_slsqp_constrained)

#: The split shipped as the default: a quarter of the budget on BO, the rest
#: on the SLSQP finisher. Best median gap of the three splits the study ran
#: (2.99e-05 at 0.25 against 3.51e-05 at 0.50 and 4.23e-05 at 0.75), and it
#: beats both parents pairwise there (71.7 % over pure SLSQP, p = 2.2e-03;
#: 75.5 % over pure BO, p = 2.7e-04, n = 53).
#:
#: NOT resolved by the study, and stated here so nobody reads the default as
#: one that was: the win RATE against pure BO moves the other way (75.9 % at
#: 0.25 against 85.2 % at 0.75), and the direct head-to-head between the two
#: is 33/53 = 62.3 % at p = 0.098 — a null. 0.25 wins bigger, 0.75 wins more
#: often, and the data does not separate them.
DEFAULT_PHI = 0.25


def split_budget(n_evals: int, phi: float = DEFAULT_PHI) -> tuple[int, int]:
    """``(n_a, n_b)`` — evaluations for the BO phase and for the finisher.

    ``max(1, round(phi * n_evals))``, exactly as the study split it, so a
    shipped run at a given phi re-flies the arm that was measured. The total
    is the budget the caller asked for and never more: that is the whole basis
    on which a handoff can be compared to a pure arm at all.
    """
    n_evals = int(n_evals)
    phi = float(phi)
    if not 0.0 < phi < 1.0:
        raise ValueError(f"phi must be strictly between 0 and 1 (got {phi!r}); "
                         f"phi=0 is pure SLSQP and phi=1 is pure BO, and both "
                         f"of those are optimisers in their own right")
    if n_evals < 2:
        raise ValueError(f"a handoff needs at least 2 evaluations, one for "
                         f"each phase (got {n_evals})")
    n_a = max(1, int(round(phi * n_evals)))
    n_a = min(n_a, n_evals - 1)          # always leave the finisher one
    return n_a, n_evals - n_a


def best_feasible_x(h: ConstrainedRunHistory) -> np.ndarray | None:
    """The design that crosses the handoff, or ``None``.

    A's best FEASIBLE design, deliberately not its best VALUE: an infeasible
    point scores through the refusal sentinel, and seeding the finisher
    outside the feasible set is a repair experiment, which is a different
    question from the one that was measured.
    """
    x = getattr(h, "best_x", None)
    return None if x is None else np.asarray(x, dtype=float)


def _stack(a, b):
    """Join two phases' margin columns, tolerating the (n,)/(n, m) split."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.ndim != b.ndim:                 # one phase saw m == 1, the other did not
        a = a if a.ndim == 2 else a[:, None]
        b = b if b.ndim == 2 else b[:, None]
    return np.concatenate([a, b], axis=0)


def run_bo_slsqp_constrained(f_and_g, bounds, n_evals: int, seed: int = 0,
                             *, phi: float = DEFAULT_PHI,
                             bo_split, x_init=None, iter_cb=None,
                             refusal=None, feasibility=None, prior=None
                             ) -> ConstrainedRunHistory:
    """BO for ``phi`` of the budget, then SLSQP seeded at BO's best feasible.

    ``bo_split`` is a callable ``budget -> (n_init, n_iter)`` — the caller's
    own splitter, so the BO phase sizes its Sobol block the way every other BO
    run in this repo does rather than growing a second sizing rule here. It is
    called with the PHASE budget, not the total: a quarter-budget BO run is a
    quarter-budget BO run, and sizing its initial design off the full budget
    would spend the finisher's share on Sobol points.

    ``prior`` (optional) is ``(X, y, g)`` a PREVIOUS run of this same arm
    already paid for, handed to the BO phase as its training set exactly as
    :func:`optimize.bo.run_bo`/:func:`run_bo_constrained` take it. Then:

    * the SPLIT is over the NEW evaluations only — ``n_evals - len(prior)``.
      A continuation asked for 8 more spends 8, and splitting the total would
      hand the finisher evaluations that were flown by the run before it;
    * the BO phase draws NO initial design (``n_init = 0``). The prior is a
      better opening than any Sobol block, which is the whole reason a GP can
      be resumed at all;
    * ``x_init`` is refused rather than ignored — an initial design and a
      training set are two answers to the same question, and a caller that
      passed both would have one of them silently dropped;
    * the finisher is seeded from the best feasible point of the prior AND
      the new BO points together, because that is what the BO phase now
      knows.

    The returned history therefore opens with the inherited points, so it is
    still one run of ``n_evals`` in evaluation order. ``meta["n_prior"]`` says
    how many of them were inherited, and ``n_a``/``n_b`` are the NEW split.

    Both phases run at the SAME seed, as the study ran them. The returned
    history is the two phases concatenated in evaluation order, so
    ``best_so_far`` is a single monotone curve over the whole budget and every
    downstream consumer sees one run.
    """
    bounds = np.asarray(bounds, dtype=float)
    n_prior = 0 if prior is None else int(np.atleast_2d(
        np.asarray(prior[0], dtype=float)).shape[0])
    if prior is not None and x_init is not None:
        raise ValueError(
            "a resumed handoff cannot also be given an initial design: the "
            "prior IS the BO phase's opening. Pass one or the other")
    if prior is not None and n_evals - n_prior < 1:
        raise ValueError(
            f"a resumed handoff needs at least one new evaluation: budget "
            f"{n_evals} against {n_prior} already flown")
    n_a, n_b = split_budget(n_evals - n_prior, phi)
    n_init, n_iter = (0, n_a) if prior is not None else bo_split(n_a)

    kw = {}
    if refusal is not None:
        kw["refusal"] = refusal
    if feasibility is not None:
        kw["feasibility"] = feasibility
    if prior is not None:
        kw["prior"] = prior
    ha = run_bo_constrained(f_and_g, bounds, n_init=n_init, n_iter=n_iter,
                            seed=seed, x_init=x_init, iter_cb=iter_cb, **kw)

    xs = best_feasible_x(ha)
    hb = run_slsqp_constrained(f_and_g, bounds, n_b, seed=seed,
                               x_seed=(None if xs is None else xs))

    return make_constrained_history(
        np.concatenate([np.asarray(ha.X, dtype=float),
                        np.asarray(hb.X, dtype=float)], axis=0),
        np.concatenate([np.asarray(ha.y, dtype=float),
                        np.asarray(hb.y, dtype=float)], axis=0),
        _stack(ha.g, hb.g),
        "bo->slsqp", seed, n_evals=n_evals,
        phi=float(phi), n_a=int(n_a), n_b=int(n_b),
        #: how many of the rows above this run INHERITED rather than flew.
        #: 0 on every run that is not a continuation, so the stored shape of
        #: a handoff record never changes.
        n_prior=int(n_prior),
        #: the ONE thing a reader cannot reconstruct from the curve: whether
        #: the finisher was seeded at all. False means the run was a BO head
        #: and a pure-SLSQP tail, which is not the arm RESULTS_HANDOFF.md
        #: measured, and it must never be inferred from a good-looking score.
        handed_over=bool(xs is not None),
        a_best=(None if ha.best_x is None else float(ha.best_y)),
        **{k: v for k, v in (ha.meta or {}).items()
           if k in ("n_screened", "n_rescue", "n_rescue_blind", "failures")})
