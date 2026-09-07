"""Constrained MULTI-OBJECTIVE BO: a front, not a winner.

Everything else in this package maximises one scalar. That is a choice the
section stage has been making implicitly for the whole project, and
``LITERATURE_REVIEW_S44.md`` is the argument that it is the wrong one: a
weighted sum — with or without a floor under it — returns only points on the
CONVEX HULL of the achievable set, at every weight vector, so whole families
of compromise sections are unreachable however the user sets the weights. A
scalarisation cannot fix that; only optimising the criteria as criteria can.

The field's default for this exact setting (expensive, constrained, few
evaluations, noisy-ish) is qEHVI/qNEHVI (Daulton, Balandat & Bakshy) or ParEGO
(Knowles 2006), and both ship in BoTorch, which is already a dependency here.
Nothing new is installed.

Contract
--------
``fg(x) -> (y, g)`` with ``y`` a length-k vector of objectives, ALL
higher-better (the caller mirrors anything that is not), and ``g`` the same
signed-margin channel every constrained problem here uses: feasible iff all
``g_i >= 0``. A solver failure returns ``y = [PENALTY] * k``, exactly as the
scalar path returns ``PENALTY``.

The hypervolume REFERENCE POINT is the caller's, not a quantile of the
observed data. That is deliberate and it is the same argument as the frozen
band: a reference point derived from the run is non-stationary, so the
hypervolume of run A and the hypervolume of run B would not be the same
number. The section stage passes its SEED, which is also what every other
reference point in this project is.

Refusals
--------
The refusal encoding (``optimize.refusal``) is applied PER OBJECTIVE, for the
reason it exists on the scalar path: the -100 sentinel sets the standardisation
scale of a GP whose real targets are of order 1 to 100, and showing it the
worst design that flew instead is measurably better (report §16.2). A row is a
refusal iff EVERY component is the sentinel, which is what ``fg`` returns.
"""
from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import torch
from botorch.acquisition.multi_objective.logei import (
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.acquisition.multi_objective.objective import (
    IdentityMCMultiOutputObjective,
)
from botorch.acquisition.multi_objective.parego import qLogNParEGO
from botorch.fit import fit_gpytorch_mll
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning,
)
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.sampling import draw_sobol_samples
from gpytorch.mlls import ExactMarginalLogLikelihood

from . import refusal as _refusal

_DTYPE = torch.double

#: the two acquisitions this module offers. ``qnehvi`` is the hypervolume one
#: (one-step Bayes-optimal in the noisy and noiseless settings, and what
#: BoTorch recommends for batch problems); ``qnparego`` is the random-weight
#: augmented-Tchebycheff scalarisation inside ordinary EI — the cheaper
#: batch-1 path, and the direct descendant of Knowles' ParEGO.
ACQFS = ("qnehvi", "qnparego")


@dataclass
class ParetoHistory:
    """What a multi-objective run leaves behind.

    ``Y`` is (n, k) in the MAXIMISING convention the caller supplied.
    ``front_idx`` indexes the FEASIBLE non-dominated rows — the answer. There
    is no ``best_x``: choosing one point off a front is a preference, and this
    module deliberately does not hold one. :func:`rank_front` is how a caller
    applies its own.
    """

    name: str
    seed: int
    X: np.ndarray                    # (n, d)
    Y: np.ndarray                    # (n, k), higher-better
    G: np.ndarray                    # (n, m) signed margins
    feasible: np.ndarray             # (n,) bool
    front_idx: np.ndarray            # indices into X/Y, feasible + nondominated
    ref_point: np.ndarray            # (k,)
    hv_trace: np.ndarray             # (n,) dominated hypervolume after each eval
    #: the SAME trace against the run's own final feasible nadir. The
    #: caller's reference point answers "did anything beat what I have, on
    #: everything?" and is the cross-run number; when nothing does it is 0.0
    #: for a whole run and says nothing about progress. This one is measured
    #: from a reference fixed AT THE END of the run, so it is monotone and
    #: readable as progress — and it is NOT comparable across runs, because
    #: each run sets its own nadir. Never mix the two.
    hv_trace_nadir: np.ndarray = field(
        default_factory=lambda: np.zeros(0))
    nadir: np.ndarray = field(default_factory=lambda: np.zeros(0))
    acqf: str = "qnehvi"
    n_init: int = 0
    gp_failures: list = field(default_factory=list)

    @property
    def hypervolume(self) -> float:
        return float(self.hv_trace[-1]) if self.hv_trace.size else 0.0

    @property
    def hypervolume_nadir(self) -> float:
        """Within-run only. See :attr:`hv_trace_nadir`."""
        return (float(self.hv_trace_nadir[-1])
                if self.hv_trace_nadir.size else 0.0)

    def front(self) -> tuple[np.ndarray, np.ndarray]:
        """``(X_front, Y_front)`` — the designs and their objective vectors."""
        return self.X[self.front_idx], self.Y[self.front_idx]


def _fit(X: torch.Tensor, y: torch.Tensor, bounds: torch.Tensor
         ) -> SingleTaskGP:
    model = SingleTaskGP(
        X, y.unsqueeze(-1),
        input_transform=Normalize(d=X.shape[-1], bounds=bounds),
        outcome_transform=Standardize(m=1),
    )
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    return model


def dominated_hypervolume(Y: np.ndarray, ref_point: np.ndarray,
                          feasible: np.ndarray | None = None) -> float:
    """Hypervolume of the feasible points that dominate ``ref_point``.

    Zero when nothing does — which is the honest answer, not a failure: a run
    that never beat its own reference point on every criterion at once has
    swept no volume.
    """
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    if feasible is not None:
        Y = Y[np.asarray(feasible, dtype=bool)]
    if Y.size == 0:
        return 0.0
    r = torch.tensor(np.asarray(ref_point, dtype=float), dtype=_DTYPE)
    t = torch.tensor(Y, dtype=_DTYPE)
    keep = (t > r).all(dim=-1)
    if not bool(keep.any()):
        return 0.0
    return float(DominatedPartitioning(ref_point=r,
                                       Y=t[keep]).compute_hypervolume())


def pareto_indices(Y: np.ndarray, feasible: np.ndarray) -> np.ndarray:
    """Indices of the FEASIBLE non-dominated rows of ``Y`` (higher-better).

    Infeasible rows are excluded before the comparison rather than after: a
    design that violates a gate is not on the front at any objective value,
    and letting it dominate a feasible one would hand the user a section the
    problem already refused.
    """
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    feas = np.asarray(feasible, dtype=bool)
    idx = np.flatnonzero(feas)
    if idx.size == 0:
        return idx
    nd = is_non_dominated(torch.tensor(Y[idx], dtype=_DTYPE)).numpy()
    return idx[nd]


def rank_front(Y_front: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Order a front by a caller's weighted sum, best first.

    This is the composite doing the job it should always have had: RANKING a
    front the search produced, rather than being the thing the search
    maximised. Ranking a front cannot make an unreachable design reachable,
    which is the whole point — every row here is one the search actually
    found.
    """
    Y_front = np.atleast_2d(np.asarray(Y_front, dtype=float))
    w = np.asarray(weights, dtype=float)
    if w.size != Y_front.shape[1]:
        raise ValueError(f"weights has {w.size} entries for "
                         f"{Y_front.shape[1]} objectives")
    return np.argsort(-(Y_front @ w), kind="stable")


def run_mobo_constrained(
    fg,
    bounds: np.ndarray,
    ref_point,
    n_init: int = 8,
    n_iter: int = 32,
    seed: int = 0,
    q: int = 1,
    acqf: str = "qnehvi",
    x_init: np.ndarray | None = None,
    iter_cb=None,
    parallel: bool = True,
    refusal: str = _refusal.DEFAULT_MODE,
    name: str = "mobo",
) -> ParetoHistory:
    """Constrained multi-objective BO. Maximises every component of ``y``.

    One GP per objective and one per margin, in a ``ModelListGP``; qNEHVI (or
    qNParEGO) with the margins entering through BoTorch's own ``constraints``
    channel as callables on the joint posterior samples, which is the same
    independence assumption the scalar path's constrained EI makes.

    ``q`` is the batch size, and the batch IS evaluated concurrently
    (``parallel=True``, the default). That is not an assumption:
    ``scripts/xfoil_batch_probe.py`` measured it on this machine against a cold
    cache per arm and the same Sobol designs in every arm, over three repeats
    that agreed to 0.2 s —

        q = 1   40.3 s      q = 2   27.3 s (1.48x)      q = 4   16.5 s (2.46x)

    — so a batch acquisition is worth its extra cost here. XFOIL is a
    subprocess, so threads are the right tool and the GIL is not in the way.
    ``parallel=False`` restores the strictly serial path; the VISITED POINTS
    are identical either way, because the acquisition proposes the whole batch
    before any of it is evaluated. Only the wall clock moves.

    ``x_init`` seeds the initial design and shrinks the Sobol draw by the rows
    it adds, so the total budget is unchanged — budget-fair, exactly as
    ``run_bo_constrained``.
    """
    if acqf not in ACQFS:
        raise ValueError(f"unknown acqf {acqf!r}; choose from {list(ACQFS)}")
    if int(q) < 1:
        raise ValueError(f"q must be >= 1, got {q}")
    refusal = _refusal.check_mode(refusal)
    bounds = np.asarray(bounds, dtype=float)
    tb = torch.tensor(bounds.T, dtype=_DTYPE)
    ref = np.asarray(ref_point, dtype=float)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    if x_init is None:
        X_list = [x.numpy() for x in
                  draw_sobol_samples(bounds=tb, n=n_init, q=1,
                                     seed=seed).squeeze(1)]
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
            X_list += [x.numpy() for x in
                       draw_sobol_samples(bounds=tb, n=n_sobol, q=1,
                                          seed=seed).squeeze(1)]

    Y_list: list[np.ndarray] = []
    G_list: list[np.ndarray] = []
    hv: list[float] = []

    def _measure(x):
        y, g = fg(x)
        return (np.atleast_1d(np.asarray(y, dtype=float)),
                np.atleast_1d(np.asarray(g, dtype=float)))

    def _record(y, g) -> None:
        Y_list.append(y)
        G_list.append(g)
        if ref.size != Y_list[0].size:
            # checked on the FIRST evaluation, which is the earliest the
            # objective count is knowable, and before any hypervolume is
            # formed from it — a mismatch here would otherwise surface as a
            # tensor-shape error from inside botorch
            raise ValueError(f"ref_point has {ref.size} entries for "
                             f"{Y_list[0].size} objectives")
        Y = np.array(Y_list)
        feas = (np.array(G_list) >= 0.0).all(axis=1)
        hv.append(dominated_hypervolume(Y, ref, feas))

    def evaluate_batch(xs) -> None:
        """Measure a batch, then record it IN ORDER.

        Concurrency must not reach the history: the hypervolume trace is a
        running quantity, so recording in completion order would make the
        trace depend on which XFOIL finished first. Measured concurrently,
        appended in the order the acquisition proposed them.
        """
        xs = list(xs)
        if parallel and len(xs) > 1:
            with ThreadPoolExecutor(max_workers=len(xs)) as ex:
                got = list(ex.map(_measure, xs))
        else:
            got = [_measure(x) for x in xs]
        for y, g in got:
            _record(y, g)

    evaluate_batch(X_list)
    k, m = Y_list[0].size, G_list[0].size
    failures: list[int] = []

    it = 0
    while it < n_iter:
        batch = min(int(q), n_iter - it)
        X_t = torch.tensor(np.array(X_list), dtype=_DTYPE)
        Y_raw = np.array(Y_list)
        # the refusal encoding, PER OBJECTIVE (see the module docstring)
        Y_imp = np.column_stack([_refusal.impute(Y_raw[:, j], refusal)
                                 for j in range(k)])
        Y_t = torch.tensor(Y_imp, dtype=_DTYPE)
        G_t = torch.tensor(np.array(G_list), dtype=_DTYPE)
        try:
            models = [_fit(X_t, Y_t[:, j], tb) for j in range(k)]
            models += [_fit(X_t, G_t[:, j], tb) for j in range(m)]
            model = ModelListGP(*models)
            # BoTorch's convention: a constraint callable is FEASIBLE where it
            # is <= 0, and our margins are feasible where they are >= 0.
            constraints = [
                (lambda Z, j=k + j_: -Z[..., j]) for j_ in range(m)]
            # The first k outputs of the model list are THE objectives; the
            # margins share the list so one joint posterior sample feeds both
            # the hypervolume and the feasibility weighting. It must be an
            # MCMultiOutputObjective — a plain callable is refused by
            # botorch and, before this was caught, that refusal was swallowed
            # by the fallback below and EVERY iteration silently drew at
            # random (see `gp_failures`, and the test that now pins it empty).
            objective = IdentityMCMultiOutputObjective(
                outcomes=list(range(k)))
            if acqf == "qnehvi":
                af = qLogNoisyExpectedHypervolumeImprovement(
                    model=model, ref_point=list(ref), X_baseline=X_t,
                    prune_baseline=True, constraints=constraints,
                    objective=objective,
                )
            else:
                af = qLogNParEGO(model=model, X_baseline=X_t,
                                 constraints=constraints,
                                 objective=objective)
            x_next, acq_val = optimize_acqf(
                af, bounds=tb, q=batch, num_restarts=10, raw_samples=256,
                sequential=True)
            new = [row.numpy() for row in x_next]
            # `optimize_acqf` returns a SCALAR at q=1 and a (q,) TENSOR under
            # sequential batching, so `float(acq_val)` raises for q>1. It used
            # to sit here inside the try, which meant a purely diagnostic line
            # sent every batched iteration down the random fallback while the
            # acquisition itself was fine — the second time this exact shape of
            # bug has cost a result in this module. Reduced defensively, and
            # the test now parametrises over q for that reason.
            diag = {"kind": "mobo_iter", "iter": it, "acqf": acqf,
                    "eval_index": len(X_list) + 1, "q": batch,
                    "acq": float(np.asarray(
                        acq_val.detach().cpu().numpy()).mean())}
        except Exception as exc:                    # pragma: no cover
            failures.append(len(X_list))
            new = [rng.uniform(bounds[:, 0], bounds[:, 1])
                   for _ in range(batch)]
            diag = {"kind": "mobo_iter", "iter": it, "acqf": "random",
                    "eval_index": len(X_list) + 1, "q": batch,
                    "error": repr(exc)}
        X_list.extend(new)
        evaluate_batch(new)
        if iter_cb is not None:
            try:
                diag["hypervolume"] = float(hv[-1])
                iter_cb(diag)
            except Exception:
                pass
        it += batch

    X = np.array(X_list)
    Y = np.array(Y_list)
    G = np.array(G_list)
    feas = (G >= 0.0).all(axis=1)
    # the WITHIN-RUN progress trace: a reference fixed at the run's own final
    # feasible nadir, nudged strictly outward so the worst point still sweeps a
    # positive volume. Fixed at the end, so the trace is monotone; different in
    # every run, so it is never a cross-run number.
    nadir = np.zeros(Y.shape[1])
    hv_n: list[float] = []
    if feas.any():
        Yf = Y[feas]
        span = Yf.max(axis=0) - Yf.min(axis=0)
        span[span <= 0] = 1.0
        nadir = Yf.min(axis=0) - 0.01 * span
        for i in range(Y.shape[0]):
            hv_n.append(dominated_hypervolume(Y[: i + 1], nadir,
                                              feas[: i + 1]))
    return ParetoHistory(
        name=name, seed=int(seed), X=X, Y=Y, G=G, feasible=feas,
        front_idx=pareto_indices(Y, feas), ref_point=ref,
        hv_trace=np.array(hv, dtype=float),
        hv_trace_nadir=np.array(hv_n, dtype=float),
        nadir=np.asarray(nadir, dtype=float),
        acqf=acqf, n_init=int(n_init), gp_failures=failures)


def hv_needed_for(Y: np.ndarray, ref_point) -> float:
    """The hypervolume ONE point sweeps — a readable per-design number."""
    return dominated_hypervolume(np.atleast_2d(Y), ref_point)


def spread(Y_front: np.ndarray) -> float:
    """A crude, stated spread measure: the mean nearest-neighbour distance on
    the front, normalised by its diagonal. Reported, never optimised — it is
    a description of what the run returned, not a target."""
    Y = np.atleast_2d(np.asarray(Y_front, dtype=float))
    if Y.shape[0] < 2:
        return 0.0
    span = Y.max(axis=0) - Y.min(axis=0)
    span[span <= 0] = 1.0
    Z = Y / span
    d = np.linalg.norm(Z[:, None, :] - Z[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    return float(d.min(axis=1).mean() / math.sqrt(Y.shape[1]))
