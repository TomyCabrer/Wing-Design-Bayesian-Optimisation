"""BoTorch Bayesian-optimisation loop.

Single-objective, box-bounded, noiseless BO:
GP surrogate (SingleTaskGP, Matern-5/2, input Normalize + outcome Standardize)
with a choice of acquisition function: LogEI (default), UCB, or qMES.

Convention: `run_bo` MAXIMISES `f`. Pass `maximize=False` to minimise
(the sign flip is handled internally; the returned history is in the
caller's original convention).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from botorch.acquisition.analytic import (
    LogExpectedImprovement,
    UpperConfidenceBound,
)
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.acquisition.max_value_entropy_search import qMaxValueEntropy
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples
from gpytorch.mlls import ExactMarginalLogLikelihood

from . import feasible as _feasible
from . import refusal as _refusal

_DTYPE = torch.double



#: Every acquisition this module can build. It is a NAMED SET rather than a
#: chain of ``elif``s because the chain lived inside a ``try`` whose bare
#: ``except`` turned a typo into silent random search.
ACQFS: frozenset = frozenset({"logei", "ucb", "qmes", "qlognei"})

@dataclass
class BOResult:
    """History of a BO run, in the caller's convention (maximise or minimise)."""

    X: np.ndarray            # (n, d) evaluated points
    y: np.ndarray            # (n,) objective values as returned by f
    best_x: np.ndarray       # (d,) best point found
    best_y: float            # best objective value found
    best_so_far: np.ndarray  # (n,) running best (caller's convention)
    n_init: int = 0
    acqf: str = "logei"
    seed: int = 0
    failures: list = field(default_factory=list)  # indices where GP fit failed
    refusal: str = _refusal.DEFAULT_MODE  # how -100 was shown to the GP
    feasibility: str = _feasible.DEFAULT_MODE  # what refused draws cost
    n_screened: int = 0      # draws the screened initial design discarded


def _fit_gp(X: torch.Tensor, y: torch.Tensor, bounds: torch.Tensor) -> SingleTaskGP:
    model = SingleTaskGP(
        X,
        y.unsqueeze(-1),
        input_transform=Normalize(d=X.shape[-1], bounds=bounds),
        outcome_transform=Standardize(m=1),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model


def _prior_lists(prior, bounds):
    """``(X_list, y_list)`` from a resumed run's already-evaluated points.

    Shared by the unconstrained and constrained loops (the constrained one
    adds its own margins). Every mismatch is a RAISE and never a silent
    truncation: a resume that quietly dropped half its history would be a
    search claiming a training set it does not have.
    """
    X = np.atleast_2d(np.asarray(prior[0], dtype=float))
    y = np.asarray(prior[1], dtype=float).ravel()
    if X.size == 0:
        raise ValueError("a resumed run needs at least one prior evaluation")
    if X.shape[1] != np.asarray(bounds).shape[0]:
        raise ValueError(f"prior X dim {X.shape[1]} != bounds dim "
                         f"{np.asarray(bounds).shape[0]}")
    if X.shape[0] != y.size:
        raise ValueError(f"prior has {X.shape[0]} points but {y.size} values")
    return [row.copy() for row in X], [float(v) for v in y]


def run_bo(
    f,
    bounds: np.ndarray,
    n_init: int = 8,
    n_iter: int = 32,
    acqf: str = "logei",
    seed: int = 0,
    maximize: bool = True,
    ucb_beta: float = 2.0,
    iter_cb=None,
    refusal: str = _refusal.DEFAULT_MODE,
    feasibility: str = _feasible.DEFAULT_MODE,
    x_init: np.ndarray | None = None,
    prior: tuple | None = None,
    #: the cheap SIZE gate, for the pool pre-filter only —
    #: see optimize.constrained.run_bo_constrained
    gate=None,
) -> BOResult:
    """Run a BO loop on ``f`` over box ``bounds``.

    Parameters
    ----------
    f : callable
        Maps a 1-D numpy array x (shape (d,)) to a scalar. Expensive black box.
    bounds : (d, 2) array
        Lower / upper bounds per dimension.
    n_init : int
        Sobol initial design size.
    n_iter : int
        Number of BO iterations after the initial design
        (total evaluations = n_init + n_iter).
    acqf : {"logei", "ucb", "qmes", "qlognei"}
    maximize : bool
        If False, minimise f.
    iter_cb : callable, optional
        Called once per BO iteration BEFORE the candidate is evaluated with a
        dict of GP diagnostics at the chosen point: ``eval_index`` (1-based
        index the evaluation will get), ``mu``/``sigma`` (posterior mean/std
        in the caller's convention) and ``acq`` (acquisition value), or
        ``fallback=True`` when the GP fit failed and a random point is used.
        Posterior evaluation is deterministic — ``None`` (default) is
        bit-for-bit the legacy path.
    feasibility : {"off", "screen", "rescue"}
        What to do about REFUSED draws in the initial design — see
        :mod:`optimize.feasible`. An unconstrained problem has no feasible
        region to hunt, so only the SCREEN applies here: ``"screen"`` and
        ``"rescue"`` both spend the initial design on draws the objective does
        not refuse outright, which is worth having for the same reason it is on
        the constrained path (a refusal costs no physics, and an all-refused
        initial design gives the GP no scale). ``"off"`` (default) is
        bit-for-bit the legacy path.
    x_init : (k, d) array, optional
        Seed points evaluated FIRST, with the Sobol draw shrunk to
        ``n_init - k`` so the total budget is unchanged. ``None`` (default) is
        bit-for-bit the legacy path. A seed is never screened out (see
        :func:`optimize.feasible.screened_init`): it is a design the caller
        chose, and dropping it would answer a different question.
    prior : ``(X, y)``, optional
        Points ALREADY EVALUATED — by the run this one continues. They become
        the GP's training set without being flown again, no initial design is
        drawn at all (``n_init`` is unread), and ``n_iter`` iterations follow
        them. That is what makes "give this run 8 more evaluations" cost 8
        evaluations instead of ``len(prior) + 8``: the objective is a pure
        function of x, so an observation is worth the same whichever run paid
        for it. ``None`` (default) is bit-for-bit the legacy path.
    refusal : {"sentinel", "worst", "worst_margin"}
        What a refused design (``f`` returned the -100 failure sentinel) is
        shown to the GP as — see :mod:`optimize.refusal`. Only the
        SURROGATE's training targets are affected: the returned history,
        the running best and the incumbent are always the true values.
        ``"sentinel"`` (default) is bit-for-bit the legacy path.
    """
    #: VALIDATED FIRST, BEFORE ANY EVALUATION IS SPENT, beside the other mode
    #: checks. A bad acquisition name is a caller error, not a numerical one:
    #: it cannot become valid later, so a run that has already paid for its
    #: initial design before refusing has charged the caller for a mistake it
    #: could have caught for free.
    #:
    #: It used to live beside the dispatch inside the per-iteration ``try``,
    #: whose bare ``except Exception`` treats any failure as a GP failure and
    #: falls back to a uniform random draw — so a typo did not raise at all:
    #: EVERY iteration silently degraded to random search while the run kept
    #: its label.
    if acqf not in ACQFS:
        raise ValueError(f"unknown acqf {acqf!r}; choices: {sorted(ACQFS)}")
    refusal = _refusal.check_mode(refusal)
    feasibility = _feasible.check_mode(feasibility)
    if refusal != _refusal.DEFAULT_MODE and not maximize:
        raise ValueError("refusal imputation needs the maximise convention")
    if _feasible.screens(feasibility) and not maximize:
        raise ValueError(
            "the screened initial design recognises a refusal by the -100 "
            "sentinel, which is only the worst value under the maximise "
            "convention; flip the sign in the objective, as refusal "
            "imputation already requires")
    bounds = np.asarray(bounds, dtype=float)
    tb = torch.tensor(bounds.T, dtype=_DTYPE)  # (2, d) botorch layout

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    # --- Sobol initial design (screened where asked: a draw the objective
    # refuses outright did no physics, so it is skipped rather than paid for) ---
    n_screened = 0
    if prior is not None:
        # A RESUMED RUN. Every point below was evaluated by the run this one
        # continues, so there is no initial design to draw and nothing to
        # re-fly: the loop opens with a GP fitted to len(X) observations.
        X_list, y_list = _prior_lists(prior, bounds)
    elif _feasible.screens(feasibility):
        X_list, y_list, _g, n_screened = _feasible.screened_init(
            f, bounds, n_init, seed, x_init=x_init,
            constrained=False, gate=gate)
    elif x_init is None:
        X0 = draw_sobol_samples(bounds=tb, n=n_init, q=1, seed=seed).squeeze(1)
        X_list = [x.numpy() for x in X0]
        y_list = [float(f(x)) for x in X_list]
    else:
        seeds_arr = np.atleast_2d(np.asarray(x_init, dtype=float))
        if seeds_arr.shape[1] != bounds.shape[0]:
            raise ValueError(f"x_init dim {seeds_arr.shape[1]} != bounds dim "
                             f"{bounds.shape[0]}")
        if seeds_arr.shape[0] > n_init:
            raise ValueError(f"x_init has {seeds_arr.shape[0]} rows > "
                             f"n_init={n_init}")
        X_list = [row.copy() for row in seeds_arr]
        n_sobol = n_init - seeds_arr.shape[0]
        if n_sobol > 0:
            X0 = draw_sobol_samples(bounds=tb, n=n_sobol, q=1,
                                    seed=seed).squeeze(1)
            X_list += [x.numpy() for x in X0]
        y_list = [float(f(x)) for x in X_list]
    failures = []

    sign = 1.0 if maximize else -1.0

    for it in range(n_iter):
        X_t = torch.tensor(np.array(X_list), dtype=_DTYPE)
        # impute BEFORE the sign flip: the sentinel is the worst value in the
        # objective's own (maximise) convention
        y_t = torch.tensor(_refusal.impute(np.array(y_list), refusal),
                           dtype=_DTYPE) * sign

        try:
            model = _fit_gp(X_t, y_t, tb)
            if acqf == "logei":
                af = LogExpectedImprovement(model, best_f=y_t.max())
            elif acqf == "ucb":
                af = UpperConfidenceBound(model, beta=ucb_beta)
            elif acqf == "qmes":
                cand_set = draw_sobol_samples(
                    bounds=tb, n=512, q=1, seed=seed + it
                ).squeeze(1)
                af = qMaxValueEntropy(model, candidate_set=cand_set)
            elif acqf == "qlognei":
                # noise-aware EI: integrates over the posterior at the
                # observed points instead of trusting the noisy best_f
                af = qLogNoisyExpectedImprovement(
                    model, X_baseline=X_t, prune_baseline=True
                )
            else:                       # pragma: no cover - guarded above
                raise AssertionError(
                    f"acqf {acqf!r} passed validation but has no branch")

            x_next, acq_val = optimize_acqf(
                af, bounds=tb, q=1, num_restarts=10, raw_samples=256
            )
            x_next = x_next.squeeze(0).numpy()
            diag = {"kind": "bo_iter", "iter": it,
                    "eval_index": len(X_list) + 1, "acqf": acqf}
            if iter_cb is not None:
                # diagnostics must NEVER alter the search: any failure here is
                # swallowed (a diag without mu/sigma), not treated as a GP
                # failure — the evaluated sequence stays identical.
                try:
                    with torch.no_grad():
                        post = model.posterior(
                            torch.tensor(x_next[None, :], dtype=_DTYPE))
                        diag["mu"] = float(post.mean.reshape(-1)[0]) * sign
                        diag["sigma"] = float(post.variance.clamp_min(0.0)
                                              .sqrt().reshape(-1)[0])
                    diag["acq"] = float(acq_val)
                except Exception:
                    pass
        except Exception:
            # GP fit / acqf optimisation failure: fall back to a random point
            # so the loop keeps its evaluation budget.
            failures.append(len(X_list))
            x_next = rng.uniform(bounds[:, 0], bounds[:, 1])
            diag = {"kind": "bo_iter", "iter": it,
                    "eval_index": len(X_list) + 1, "fallback": True}

        if iter_cb is not None:
            iter_cb(diag)
        X_list.append(x_next)
        y_list.append(float(f(x_next)))

    X = np.array(X_list)
    y = np.array(y_list)
    running = np.maximum.accumulate(y) if maximize else np.minimum.accumulate(y)
    i_best = int(np.argmax(y)) if maximize else int(np.argmin(y))
    return BOResult(
        X=X,
        y=y,
        best_x=X[i_best],
        best_y=float(y[i_best]),
        best_so_far=running,
        n_init=n_init,
        acqf=acqf,
        seed=seed,
        failures=failures,
        refusal=refusal,
        feasibility=feasibility,
        n_screened=int(n_screened),
    )
