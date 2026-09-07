"""Constrained optimisers: maximise f(x) subject to g(x) >= 0.

Built for the Tier C cavitation-constrained hydrofoil (report §11): the
evaluator returns BOTH the objective and a SIGNED constraint margin from
the same physics solve, so every optimiser here takes a single callable

    f_and_g(x) -> (f, g)     # f finite float; g a finite float (legacy,
                             # single constraint) OR a 1-D array-like of
                             # m signed margins; feasible iff ALL g_i >= 0

and charges ONE budget unit per evaluated point (f and all g_i come from
one solver call — charging them separately would double-bill physics that
was computed once). Points where the SOLVER fails should return the usual
f = PENALTY together with finite, clearly-infeasible margins (the caller
decides, e.g. g = -1); a constrained optimiser must always see a finite
violation magnitude, never NaN.

Multiple constraints (Phase 2, aircraft.py: stress + tip-deflection etc.)
are handled natively by every runner; the single-g path is byte-for-byte
the original code (m == 1 branches keep the legacy behaviour, RNG
consumption and history dtypes included), so existing hydrofoil results
and tests are unaffected. ConstrainedRunHistory.g stays (n,) when m == 1
(downstream code, e.g. experiments/rec_rule.py, indexes it as 1-D) and
becomes (n, m) otherwise; best_g is always the SCALAR minimum margin at
the best feasible point (the binding constraint's slack).

Winner metric convention (report §11): best TRUE FEASIBLE value found. An
infeasible recommendation is a failure, not a smaller number — histories
therefore track the running best over FEASIBLE points only (-inf until the
first feasible point).

Methods:
- run_bo_constrained : independent GPs (objective + one per constraint) and
  the analytic log constrained EI = LogEI + sum_i log P(g_i >= 0) (Gardner
  et al. 2014; Letham et al. 2019; botorch
  LogConstrainedExpectedImprovement). While no feasible point has been
  observed, best_f falls back to the worst observed objective, which turns
  the acquisition into a feasibility hunt (the EI factor saturates and
  P(feasible) dominates) — documented standard trick.
- run_ga_constrained : pymoo GA with native constraint handling
  (feasibility-first tournament), G = -g <= 0 elementwise.
- run_slsqp_constrained : scipy SLSQP with one native inequality constraint
  per margin, FD gradients, multistart; every unique x charges one
  evaluation (memoised — scipy calls the objective and each constraint
  separately at the same x).
- run_penalty_gradient : L-BFGS-B on the quadratic-penalty merit
  f - w * sum_i min(0, g_i)^2 (w = 100, documented), multistart, FD.
- run_random_constrained / run_sobol_constrained : rejection-free sampling;
  infeasible points are logged and simply never become the incumbent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
# The GP stack is OPTIONAL here and only here. This module holds the GA,
# SLSQP, penalty and DOE runners for constrained problems as well as
# BO-LogCEI, and those four are pymoo and scipy — so an unguarded import at
# the top made every constrained search depend on a library only one of them
# uses, and a machine with no PyTorch wheel (macOS x86_64; macOS 13 or older
# on arm64) could run none of them. The names stay module-level either way,
# so the import order, the monkeypatch points and every number are what they
# were; what changes is that the failure is caught and named. See
# optimize/torch_optional.py.
from .torch_optional import require as _require_torch
from .torch_optional import torch

try:
    from botorch.acquisition.analytic import LogConstrainedExpectedImprovement
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import ModelListGP, SingleTaskGP
    from botorch.models.transforms.input import Normalize
    from botorch.models.transforms.outcome import Standardize
    from botorch.optim import optimize_acqf
    from botorch.utils.sampling import draw_sobol_samples
    from gpytorch.mlls import ExactMarginalLogLikelihood
except ImportError:                    # no torch here, so no botorch either
    LogConstrainedExpectedImprovement = fit_gpytorch_mll = None
    ModelListGP = SingleTaskGP = Normalize = Standardize = None
    optimize_acqf = draw_sobol_samples = ExactMarginalLogLikelihood = None
from scipy.optimize import minimize as scipy_minimize
from scipy.stats import qmc

from . import feasible as _feasible
from .baselines import _first_start
from . import refusal as _refusal



# ------------------------------------------------------------------ history

@dataclass
class ConstrainedRunHistory:
    """Result container; best_* refer to the best FEASIBLE point (all g >= 0)."""

    X: np.ndarray            # (n, d) evaluated points
    y: np.ndarray            # (n,) objective values
    g: np.ndarray            # (n,) signed margins [m == 1] or (n, m) [m > 1]
    best_x: np.ndarray | None   # None if no feasible point was found
    best_y: float            # -inf if no feasible point was found
    best_g: float | None     # scalar MIN margin at best point (binding slack)
    best_so_far: np.ndarray  # (n,) running best FEASIBLE y (-inf before first)
    n_feasible: int = 0
    name: str = ""
    seed: int = 0
    meta: dict = field(default_factory=dict)


def margin_summary(g) -> tuple[float, bool]:
    """``(binding_margin, feasible)`` for the margin(s) of ONE point.

    Mirrors the rule ``make_constrained_history`` applies to a whole run:
    a point is feasible when ALL its margins are >= 0, and its scalar
    summary is the binding (minimum) margin. Accepts a scalar [legacy
    m == 1, where this returns exactly ``(float(g), g >= 0)``] or an (m,)
    row, so JSON-writing drivers do not need a shape test at every call.
    """
    ga = np.atleast_1d(np.asarray(g, dtype=float))
    return float(ga.min()), bool((ga >= 0.0).all())


def make_constrained_history(X, y, g, name: str, seed: int,
                             n_evals: int | None = None,
                             **meta) -> ConstrainedRunHistory:
    """Build a history, truncating to the eval budget for fair comparison.

    ``g`` may be (n,) scalars [legacy single constraint] or (n, m) rows of
    m margins. An (n, 1) input is squeezed back to (n,) so single-constraint
    histories keep the 1-D shape downstream consumers index into
    (rec_rule.py reads ``h.g[i]`` as a scalar). Feasibility of a point is
    ALL of its margins >= 0; ``best_g`` is the scalar minimum margin at the
    best feasible point — the slack of the binding constraint.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    g = np.asarray(g, dtype=float)
    if g.ndim == 2 and g.shape[1] == 1:
        g = g[:, 0]                      # m == 1: keep legacy (n,) shape
    if n_evals is not None:
        X, y, g = X[:n_evals], y[:n_evals], g[:n_evals]
    G = g if g.ndim == 2 else g[:, None]     # (n, m) view for feasibility
    feas = (G >= 0.0).all(axis=1)
    y_feas = np.where(feas, y, -np.inf)
    running = np.maximum.accumulate(y_feas)
    if feas.any():
        i_best = int(np.argmax(y_feas))
        best_x, best_y = X[i_best], float(y[i_best])
        best_g = float(G[i_best].min())
    else:
        best_x, best_y, best_g = None, float("-inf"), None
    return ConstrainedRunHistory(
        X=X, y=y, g=g, best_x=best_x, best_y=best_y, best_g=best_g,
        best_so_far=running, n_feasible=int(feas.sum()),
        name=name, seed=seed, meta=dict(meta))


class _CountedFG:
    """Budget-counted f_and_g with per-point memoisation.

    scipy evaluates the objective and each constraint through separate
    callbacks, usually at the same x — one physics solve, so one budget
    unit. Memoising by the exact byte pattern of x makes the accounting
    honest without re-solving.

    The wrapped f_and_g may return a scalar margin (legacy) or a 1-D
    array of m margins; internally the memo always stores a 1-D margin
    array, while ``self.g`` records scalars when m == 1 so histories keep
    the legacy (n,) shape.
    """

    class Budget(Exception):
        pass

    def __init__(self, f_and_g, n_evals: int):
        self.fg = f_and_g
        self.n_evals = n_evals
        self.X: list[np.ndarray] = []
        self.y: list[float] = []
        self.g: list = []                     # floats (m == 1) or (m,) arrays
        self.m: int | None = None             # margins per point, set on 1st call
        self._memo: dict[bytes, tuple[float, np.ndarray]] = {}

    def __call__(self, x) -> tuple[float, np.ndarray]:
        x = np.array(x, dtype=float)
        key = x.tobytes()
        if key in self._memo:
            return self._memo[key]
        if len(self.y) >= self.n_evals:
            raise _CountedFG.Budget
        v, gg = self.fg(x)
        v = float(v)
        g_arr = np.atleast_1d(np.asarray(gg, dtype=float))
        self.m = g_arr.size
        self.X.append(x)
        self.y.append(v)
        self.g.append(float(g_arr[0]) if g_arr.size == 1 else g_arr)
        self._memo[key] = (v, g_arr)
        return v, g_arr

    def f(self, x) -> float:
        return self(x)[0]

    def gval(self, x, i: int = 0) -> float:
        """i-th signed margin at x (memoised; for per-constraint callbacks)."""
        return float(self(x)[1][i])


# ---------------------------------------------------------------------- BO

#: the tensor dtype every GP in this module is fit in. None where
#: PyTorch is absent; nothing reads it before _require_torch has run.
_DTYPE = torch.double if torch is not None else None

def _fit_single_gp(X: torch.Tensor, y: torch.Tensor,
                   bounds: torch.Tensor) -> SingleTaskGP:
    _require_torch("A Gaussian-process fit")
    model = SingleTaskGP(
        X, y.unsqueeze(-1),
        input_transform=Normalize(d=X.shape[-1], bounds=bounds),
        outcome_transform=Standardize(m=1),
    )
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    return model


def _feasible_seen(g_rows) -> bool:
    """True iff any observed point has ALL margins >= 0.

    The one test the feasibility phase and the escape hatch are gated on, so
    both can only ever change a run that would otherwise return nothing.
    """
    return any(bool((np.atleast_1d(r) >= 0.0).all()) for r in g_rows)


def run_bo_constrained(
    f_and_g,
    bounds: np.ndarray,
    n_init: int = 8,
    n_iter: int = 32,
    seed: int = 0,
    x_init: np.ndarray | None = None,
    iter_cb=None,
    refusal: str = _refusal.DEFAULT_MODE,
    feasibility: str = _feasible.DEFAULT_MODE,
    prior: tuple | None = None,
    #: the cheap SIZE gate, for the pool pre-filter only
    #: (:data:`optimize.feasible._GATE_PREFILTER`). ``excess(x) > 0``
    #: is a SUFFICIENT condition for a refusal, so a pool draw that
    #: trips it can be skipped without paying the solver. It cannot
    #: change which points the screen keeps — only what they cost.
    gate=None,
) -> ConstrainedRunHistory:
    """Constrained BO: independent objective/constraint GPs + analytic
    log constrained EI (feasibility-weighted LogEI). Maximises f s.t.
    all g_i >= 0.

    One GP per output — ModelListGP of 1 + m models; botorch's
    LogConstrainedExpectedImprovement multiplies LogEI by sum_i
    log P(g_i >= 0) via ``constraints={i: (0.0, None)}`` over the
    constraint output indices 1..m (Gardner et al. 2014 treats the g_i as
    independent, which matches the independent-GP model). The m == 1
    branch below is LITERALLY the original single-constraint code so that
    torch RNG consumption — and therefore every evaluated point — is
    unchanged for existing single-g studies (hydrofoil, rec_rule).

    ``x_init``: optional (k, d) seed points evaluated FIRST, with the Sobol
    draw shrunk to n_init - k so the total budget n_init + n_iter is
    unchanged (warm-start stays budget-fair). ``None`` (default) is
    bit-for-bit the legacy path — same Sobol draw, same RNG streams.

    ``prior``: ``(X, y, g)`` already evaluated — by the run this one
    continues. They are the GPs' training set, no initial design is drawn
    (``n_init`` is unread) and ``n_iter`` iterations follow them, so a
    continuation pays for its ADDED evaluations only. The objective and the
    margins are pure functions of x, so an observation is worth the same
    whichever run bought it. ``None`` (default) is the legacy path.

    ``iter_cb``: optional per-iteration diagnostics callback (see
    ``bo.run_bo``); here the record additionally carries the constraint GPs'
    posterior ``g_mu``/``g_sigma`` lists and ``p_feasible`` (the independent-
    GP product of P(g_i >= 0) at the candidate). Diagnostics are deterministic
    and failure-isolated — ``None`` is bit-for-bit the legacy path.

    ``refusal``: what a SOLVER FAILURE (f == the -100 sentinel) is shown to
    the objective GP as — see :mod:`optimize.refusal`. This is the path
    where it bites: every case in the registry that refuses designs at all
    is constrained, and the sentinel sets the standardisation scale for an
    objective of order 1e-3 (Cd) or 1 (a composite). Only the objective
    GP's targets change — the constraint GPs, the history, the feasibility
    test and the incumbent are untouched, and the ``best_f`` fallback used
    before the first feasible point (the worst observed objective) becomes
    the worst REAL one. ``"sentinel"`` (default) is bit-for-bit legacy.

    ``feasibility``: what to do about a run that has not found a feasible
    design — see :mod:`optimize.feasible`. ``"off"`` (default) is bit-for-bit
    legacy. ``"screen"`` spends the initial design on draws the physics does
    not refuse outright; ``"rescue"`` adds a feasibility phase (while nothing
    feasible has been seen, climb the min-margin surrogate instead of the
    objective) and replaces a repeated candidate — the flat-acquisition corner
    a blind constrained run collapses onto — with a screened draw. Both extra
    mechanisms are gated on there being NO feasible observation, so a run that
    has found one behaves from then on exactly as it always did.
    """
    _require_torch("The constrained BO (BO-LogCEI) optimiser")
    refusal = _refusal.check_mode(refusal)
    feasibility = _feasible.check_mode(feasibility)
    bounds = np.asarray(bounds, dtype=float)
    tb = torch.tensor(bounds.T, dtype=_DTYPE)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n_screened = 0
    n_rescue = 0
    #: feasibility-phase iterations the surrogate could NOT steer
    n_blind = 0

    if prior is not None:
        # A RESUMED RUN — see the docstring. Nothing here is flown.
        from .bo import _prior_lists
        X_list, y_list = _prior_lists(prior, bounds)
        g_rows = [np.atleast_1d(np.asarray(row, dtype=float))
                  for row in np.atleast_2d(np.asarray(prior[2], dtype=float))]
        if len(g_rows) != len(X_list):
            raise ValueError(f"prior has {len(X_list)} points but "
                             f"{len(g_rows)} margin rows")
    elif _feasible.screens(feasibility):
        X_list, y_list, g_rows, n_screened = _feasible.screened_init(
            f_and_g, bounds, n_init, seed, x_init=x_init,
            constrained=True, gate=gate)
    elif x_init is None:
        X0 = draw_sobol_samples(bounds=tb, n=n_init, q=1, seed=seed).squeeze(1)
        X_list = [x.numpy() for x in X0]
    else:
        seeds_arr = np.atleast_2d(np.asarray(x_init, dtype=float))
        if seeds_arr.shape[1] != bounds.shape[0]:
            raise ValueError(
                f"x_init dim {seeds_arr.shape[1]} != bounds dim "
                f"{bounds.shape[0]}")
        if seeds_arr.shape[0] > n_init:
            raise ValueError(
                f"x_init has {seeds_arr.shape[0]} rows > n_init={n_init}")
        n_sobol = n_init - seeds_arr.shape[0]
        X_list = [row.copy() for row in seeds_arr]
        if n_sobol > 0:
            X0 = draw_sobol_samples(
                bounds=tb, n=n_sobol, q=1, seed=seed).squeeze(1)
            X_list += [x.numpy() for x in X0]
    if prior is None and not _feasible.screens(feasibility):
        y_list: list[float] = []
        g_rows: list[np.ndarray] = []            # each (m,)
        for x in X_list:
            v, gg = f_and_g(x)
            y_list.append(float(v))
            g_rows.append(np.atleast_1d(np.asarray(gg, dtype=float)))
    m = g_rows[0].size
    failures = []

    # ---- the feasibility phase (optimize.feasible.rescues). ONLY entered
    # while nothing feasible has been observed, so a run that found a design
    # in its initial design never sees it, and it hands back every iteration
    # it did not need: the total evaluation count is n_init + n_iter either
    # way. What it buys is measured in that module's docstring — a blind
    # constrained run does not recover on its own, it rides a box corner.
    if _feasible.rescues(feasibility):
        while n_rescue < n_iter and not _feasible_seen(g_rows):
            x_next, diag = _feasible.rescue_candidate(
                X_list, g_rows, bounds, rng)
            if x_next is None:
                # THE PHASE DID NOT STEER THIS ONE. Counted, because a
                # feasibility phase that fell back to a uniform draw every
                # iteration is a phase in name only — and that is exactly what
                # a flat refusal margin produces (the surface has no gradient
                # to climb, ``rescue_candidate``'s ``degenerate``). A
                # mechanism whose failure looks like its success from the
                # outside is the bug class this repo keeps finding; this is
                # the number that tells them apart.
                n_blind += 1
                x_next, v, gg, k = _feasible.screened_draw(
                    f_and_g, bounds, rng, constrained=True)
                n_screened += k
            else:
                v, gg = f_and_g(x_next)
            diag.update({"kind": "bo_iter", "iter": n_rescue,
                         "eval_index": len(X_list) + 1})
            if iter_cb is not None:
                iter_cb(diag)
            X_list.append(np.array(x_next, dtype=float))
            y_list.append(float(v))
            g_rows.append(np.atleast_1d(np.asarray(gg, dtype=float)))
            n_rescue += 1

    for it in range(n_iter - n_rescue):
        X_t = torch.tensor(np.array(X_list), dtype=_DTYPE)
        y_t = torch.tensor(_refusal.impute(np.array(y_list), refusal),
                           dtype=_DTYPE)

        try:
            if m == 1:                       # legacy path — bit-for-bit
                g_t = torch.tensor(np.array([r[0] for r in g_rows]),
                                   dtype=_DTYPE)
                model = ModelListGP(_fit_single_gp(X_t, y_t, tb),
                                    _fit_single_gp(X_t, g_t, tb))
                feas = g_t >= 0.0
                constraints = {1: (0.0, None)}
            else:
                G_t = torch.tensor(np.array(g_rows), dtype=_DTYPE)  # (n, m)
                model = ModelListGP(
                    _fit_single_gp(X_t, y_t, tb),
                    *[_fit_single_gp(X_t, G_t[:, j], tb) for j in range(m)])
                feas = (G_t >= 0.0).all(dim=-1)
                constraints = {j: (0.0, None) for j in range(1, m + 1)}
            # no feasible observation yet -> worst observed objective:
            # the EI factor saturates and P(feasible) drives the search
            best_f = y_t[feas].max() if feas.any() else y_t.min()
            af = LogConstrainedExpectedImprovement(
                model, best_f=best_f, objective_index=0,
                constraints=constraints,
            )
            x_next, acq_val = optimize_acqf(
                af, bounds=tb, q=1, num_restarts=10, raw_samples=256
            )
            x_next = x_next.squeeze(0).numpy()
            diag = {"kind": "bo_iter", "iter": it + n_rescue,
                    "eval_index": len(X_list) + 1, "acqf": "logcei"}
            if iter_cb is not None:
                # diagnostics must NEVER alter the search: failures here are
                # swallowed, never treated as a GP failure.
                try:
                    with torch.no_grad():
                        post = model.posterior(torch.tensor(
                            np.asarray(x_next)[None, :], dtype=_DTYPE))
                        means = post.mean.reshape(-1).tolist()
                        sds = (post.variance.clamp_min(0.0).sqrt()
                               .reshape(-1).tolist())
                    diag["mu"] = float(means[0])
                    diag["sigma"] = float(sds[0])
                    diag["g_mu"] = [float(v) for v in means[1:]]
                    diag["g_sigma"] = [float(v) for v in sds[1:]]
                    p_feas = 1.0
                    for mj, sj in zip(means[1:], sds[1:]):
                        z = float(mj) / max(float(sj), 1e-300)
                        p_feas *= 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
                    diag["p_feasible"] = p_feas
                    diag["acq"] = float(acq_val)
                except Exception:
                    pass
        except Exception:
            failures.append(len(X_list))
            x_next = rng.uniform(bounds[:, 0], bounds[:, 1])
            diag = {"kind": "bo_iter", "iter": it + n_rescue,
                    "eval_index": len(X_list) + 1, "fallback": True}

        # NOTE ON THE ESCAPE. The repeated-candidate guard that stops a blind
        # run riding one box corner lives in the feasibility phase above
        # (``feasible.rescue_candidate``), not here, and deliberately: this
        # loop cannot RUN while nothing feasible has been seen, because the
        # phase holds every remaining iteration until it finds something. A
        # copy of the guard here looked prudent and was unreachable — it
        # survived its own mutation test, which is how it was caught.
        if iter_cb is not None:
            iter_cb(diag)
        v, gg = f_and_g(x_next)
        X_list.append(np.array(x_next, dtype=float))
        y_list.append(float(v))
        g_rows.append(np.atleast_1d(np.asarray(gg, dtype=float)))

    # the extra meta only where the extra mechanisms were asked for: a legacy
    # run's history — meta included — is the history it has always been
    extra = ({} if feasibility == _feasible.DEFAULT_MODE
             else {"feasibility": feasibility, "n_screened": int(n_screened),
                   "n_rescue": int(n_rescue), "n_rescue_blind": int(n_blind)})
    return make_constrained_history(
        X_list, y_list, np.array(g_rows), "BO-LogCEI", seed,
        n_init=n_init, failures=failures, refusal=refusal, **extra)


# ---------------------------------------------------------------------- GA

class _GaResize(Exception):
    """Raised inside the pymoo problem when the observed number of margins
    differs from the declared n_ieq_constr; carries the true m."""

    def __init__(self, m: int):
        self.m = m


def run_ga_constrained(f_and_g, bounds, n_evals: int, seed: int = 0,
                       pop_size: int = 20,
                       n_constraints: int | None = None) -> ConstrainedRunHistory:
    """pymoo GA with native constraint handling (G = -g <= 0 feasible),
    n_ieq_constr = m constraints.

    pymoo needs ``n_ieq_constr`` BEFORE the first evaluation, but m is only
    observable from f_and_g's return, and probing f_and_g out-of-band is
    forbidden (a stateful evaluator — e.g. the seeded noise wrapper in
    experiments/hydrofoil_bo.py — would have its stream shifted, breaking
    single-g reproducibility). So: declare m = 1 (legacy), and if the first
    evaluation reveals m > 1, abort and restart pymoo with the correct m.
    Same seed -> pymoo resamples the identical initial population, the
    already-solved first point is served from the memo, and the recorded
    history is exactly as if m had been declared upfront. The single-g path
    never triggers the restart and is byte-for-byte the original. Callers
    who know m can pass ``n_constraints`` to skip the throwaway construction.
    """
    from pymoo.algorithms.soo.nonconvex.ga import GA
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.optimize import minimize as pymoo_minimize

    bounds = np.asarray(bounds, dtype=float)
    counted = _CountedFG(f_and_g, n_evals)

    def _minimize_with(m: int):
        class _P(ElementwiseProblem):
            def __init__(self):
                super().__init__(n_var=bounds.shape[0], n_obj=1,
                                 n_ieq_constr=m,
                                 xl=bounds[:, 0], xu=bounds[:, 1])

            def _evaluate(self, x, out, *args, **kwargs):
                try:
                    v, gg = counted(x)
                    if gg.size != m:
                        raise _GaResize(gg.size)
                    out["F"] = -v               # pymoo minimises
                    out["G"] = -gg              # feasible iff all G <= 0
                except _CountedFG.Budget:
                    out["F"] = np.inf                 # budget spent: poison
                    out["G"] = np.full(m, np.inf)
        pymoo_minimize(_P(), GA(pop_size=pop_size),
                       ("n_evals", n_evals), seed=seed, verbose=False)

    try:
        _minimize_with(1 if n_constraints is None else int(n_constraints))
    except _GaResize as e:
        _minimize_with(e.m)
    return make_constrained_history(counted.X, counted.y, counted.g,
                                    "ga-constrained", seed,
                                    n_evals=n_evals, pop_size=pop_size)


# ------------------------------------------------------------------ gradient

def run_slsqp_constrained(f_and_g, bounds, n_evals: int,
                          seed: int = 0,
                          x_seed=None) -> ConstrainedRunHistory:
    """Multistart SLSQP with the native inequality constraints, FD gradients.

    One scipy constraint dict per margin g_i, each a per-index closure over
    the SAME memoised _CountedFG — scipy queries f and every g_i at the
    same x, but the physics is solved (and the budget charged) once per
    unique point. FD points count — the honest budget for a black-box
    constrained solver. m is sized from the start point's evaluation, which
    scipy would evaluate first anyway (memoised, so nothing is double-
    solved or double-charged and the recorded sequence is unchanged).
    """
    bounds = np.asarray(bounds, dtype=float)
    rng = np.random.default_rng(seed)
    counted = _CountedFG(f_and_g, n_evals)
    #: ``x_seed`` (optional) makes the FIRST start a stated design — the
    #: handoff half of "run A, then continue with B". Later starts stay
    #: random, so a handoff cannot quietly turn a multistart into a single
    #: local polish, and an out-of-box seed is refused, never clipped.
    first = _first_start(x_seed, bounds)

    while len(counted.y) < n_evals:
        if first is not None:
            x0, first = first, None
        else:
            x0 = rng.uniform(bounds[:, 0], bounds[:, 1])
        # cap BEFORE x0 is charged — identical to the original accounting
        maxiter = max(n_evals - len(counted.y), 1)
        try:
            _, g0 = counted(x0)          # sizes m; scipy's x0 eval hits memo
            cons = [{"type": "ineq",
                     "fun": lambda x, j=j: counted.gval(x, j)}
                    for j in range(g0.size)]
            scipy_minimize(
                lambda x: -counted.f(x), x0, method="SLSQP", bounds=bounds,
                constraints=cons,
                options={"maxiter": maxiter},
            )
        except _CountedFG.Budget:
            break
    return make_constrained_history(counted.X, counted.y, counted.g,
                                    "gradient-SLSQP", seed, n_evals=n_evals)


def run_penalty_gradient(f_and_g, bounds, n_evals: int, seed: int = 0,
                         weight: float = 100.0,
                         x_seed=None) -> ConstrainedRunHistory:
    """Multistart L-BFGS-B on the quadratic-penalty merit
    f - weight * sum_i min(0, g_i)^2 — the 'bolt a penalty onto an
    unconstrained optimiser' baseline every practitioner tries first.
    (For m == 1 this is numerically identical to the original scalar
    min(0, g)^2 merit.)
    """
    bounds = np.asarray(bounds, dtype=float)
    rng = np.random.default_rng(seed)
    counted = _CountedFG(f_and_g, n_evals)
    first = _first_start(x_seed, bounds)      # the handoff start; see above

    def neg_merit(x):
        v, gg = counted(x)
        return -(v - weight * float(np.sum(np.minimum(0.0, gg) ** 2)))

    while len(counted.y) < n_evals:
        if first is not None:
            x0, first = first, None
        else:
            x0 = rng.uniform(bounds[:, 0], bounds[:, 1])
        try:
            scipy_minimize(neg_merit, x0, method="L-BFGS-B", bounds=bounds,
                           options={"maxfun": max(n_evals - len(counted.y), 1)})
        except _CountedFG.Budget:
            break
    return make_constrained_history(counted.X, counted.y, counted.g,
                                    "gradient-penalty", seed,
                                    n_evals=n_evals, weight=weight)


# ------------------------------------------------------------ random / DOE

def run_random_constrained(f_and_g, bounds, n_evals: int,
                           seed: int = 0) -> ConstrainedRunHistory:
    bounds = np.asarray(bounds, dtype=float)
    rng = np.random.default_rng(seed)
    X = rng.uniform(bounds[:, 0], bounds[:, 1],
                    size=(n_evals, bounds.shape[0]))
    vals = [f_and_g(x) for x in X]
    return make_constrained_history(X, [v for v, _ in vals],
                                    [gg for _, gg in vals],
                                    "random", seed)


def run_sobol_constrained(f_and_g, bounds, n_evals: int,
                          seed: int = 0) -> ConstrainedRunHistory:
    bounds = np.asarray(bounds, dtype=float)
    sampler = qmc.Sobol(d=bounds.shape[0], scramble=True, seed=seed)
    X = qmc.scale(sampler.random(n_evals), bounds[:, 0], bounds[:, 1])
    vals = [f_and_g(x) for x in X]
    return make_constrained_history(X, [v for v, _ in vals],
                                    [gg for _, gg in vals],
                                    "sobol-doe", seed)


REGISTRY_CONSTRAINED = {
    "bo": run_bo_constrained,
    "ga": run_ga_constrained,
    "slsqp": run_slsqp_constrained,
    "penalty": run_penalty_gradient,
    "random": run_random_constrained,
    "sobol": run_sobol_constrained,
}
