"""Baseline optimisers to benchmark BO against: GA (pymoo), gradient
(scipy L-BFGS-B / SLSQP multistart with finite differences), random search,
Sobol DOE, and full-factorial grid (brute force).

Every baseline MAXIMISES `f` (matching bo.run_bo) and returns a
metrics.RunHistory truncated to exactly `n_evals` evaluations, so
convergence curves are budget-fair. Gradient evaluations spent on finite
differences COUNT toward the budget — that is the honest comparison for
expensive black-box objectives.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize as scipy_minimize
from scipy.stats import qmc

from .metrics import RunHistory, make_history


class _Budget(Exception):
    """Raised internally when the evaluation budget is exhausted."""


class _CountedF:
    def __init__(self, f, n_evals: int):
        self.f = f
        self.n_evals = n_evals
        self.X: list[np.ndarray] = []
        self.y: list[float] = []

    def __call__(self, x: np.ndarray) -> float:
        if len(self.y) >= self.n_evals:
            raise _Budget
        x = np.array(x, dtype=float)
        v = float(self.f(x))
        self.X.append(x)
        self.y.append(v)
        return v


# ---------------------------------------------------------------- random / DOE

def run_random(f, bounds, n_evals: int, seed: int = 0) -> RunHistory:
    bounds = np.asarray(bounds, dtype=float)
    rng = np.random.default_rng(seed)
    X = rng.uniform(bounds[:, 0], bounds[:, 1], size=(n_evals, bounds.shape[0]))
    y = [float(f(x)) for x in X]
    return make_history(X, y, "random", seed)


def run_sobol(f, bounds, n_evals: int, seed: int = 0) -> RunHistory:
    """Sobol space-filling DOE — the 'brute force with a fixed budget' baseline."""
    bounds = np.asarray(bounds, dtype=float)
    sampler = qmc.Sobol(d=bounds.shape[0], scramble=True, seed=seed)
    U = sampler.random(n_evals)
    X = qmc.scale(U, bounds[:, 0], bounds[:, 1])
    y = [float(f(x)) for x in X]
    return make_history(X, y, "sobol-doe", seed)


def run_grid(f, bounds, n_evals: int, seed: int = 0) -> RunHistory:
    """Full-factorial grid with floor(n_evals^(1/d)) points per dimension.

    The classic brute force; the points-per-dim collapse with dimension is
    exactly the curse-of-dimensionality effect the crossover study measures.
    """
    bounds = np.asarray(bounds, dtype=float)
    d = bounds.shape[0]
    n_per = max(int(np.floor(n_evals ** (1.0 / d))), 2)
    axes = [np.linspace(lo, hi, n_per) for lo, hi in bounds]
    X = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, d)
    y = [float(f(x)) for x in X]
    return make_history(X, y, "grid", seed, n_evals=n_evals,
                        n_per_dim=n_per, n_grid_total=X.shape[0])


# ---------------------------------------------------------------------- GA

def run_ga(f, bounds, n_evals: int, seed: int = 0, pop_size: int = 20) -> RunHistory:
    from pymoo.algorithms.soo.nonconvex.ga import GA
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.optimize import minimize as pymoo_minimize

    bounds = np.asarray(bounds, dtype=float)
    counted = _CountedF(f, n_evals)

    class _P(ElementwiseProblem):
        def __init__(self):
            super().__init__(n_var=bounds.shape[0], n_obj=1,
                             xl=bounds[:, 0], xu=bounds[:, 1])

        def _evaluate(self, x, out, *args, **kwargs):
            try:
                out["F"] = -counted(x)          # pymoo minimises
            except _Budget:
                out["F"] = np.inf               # budget spent: poison extras

    pymoo_minimize(_P(), GA(pop_size=pop_size),
                   ("n_evals", n_evals), seed=seed, verbose=False)
    return make_history(counted.X, counted.y, "ga", seed,
                        n_evals=n_evals, pop_size=pop_size)


# ------------------------------------------------------------------ gradient

#: A HANDOFF is one optimiser starting where another stopped. Every multistart
#: runner below draws its own random ``x0``; handing one a design means using
#: it for the FIRST start and leaving the rest random, so the run keeps the
#: restart diversity that makes a multistart a global method at all. The seed
#: is REFUSED rather than clipped when it lies outside the box: a silently
#: moved start point would make "this run cannot come back worse than the
#: design I gave it" false while the caller believed it was armed, which is
#: the dropped-flag failure this package refuses everywhere else.
def _first_start(x_seed, bounds):
    """Validate a handoff design, or None. Raises if it is not in the box."""
    if x_seed is None:
        return None
    x = np.asarray(x_seed, dtype=float).ravel()
    bounds = np.asarray(bounds, dtype=float)
    if x.size != bounds.shape[0]:
        raise ValueError(
            f"x_seed has {x.size} values but the box has {bounds.shape[0]} "
            f"dimensions")
    bad = np.flatnonzero((x < bounds[:, 0]) | (x > bounds[:, 1]))
    if bad.size:
        i = int(bad[0])
        raise ValueError(
            f"x_seed value {x[i]:g} for dimension {i} is outside the box "
            f"[{bounds[i, 0]:g}, {bounds[i, 1]:g}]")
    return x


def run_gradient(f, bounds, n_evals: int, seed: int = 0,
                 method: str = "L-BFGS-B",
                 fd_eps: float | np.ndarray | None = None,
                 x_seed=None) -> RunHistory:
    """Multistart local gradient optimisation (finite-difference gradients).

    Starts are random; each start runs until convergence or budget
    exhaustion, then a fresh start is drawn. FD evals count.

    ``x_seed`` (optional) makes the FIRST start a stated design instead of a
    random one — the handoff half of "run A, then continue with B". Later
    starts stay random, so a handoff cannot quietly turn a multistart into a
    single local polish.

    fd_eps: finite-difference step (scalar or per-dimension array). Leave
    None for scipy's default (~1e-8, correct for noiseless objectives);
    noisy objectives need a much larger, noise-aware step.
    """
    bounds = np.asarray(bounds, dtype=float)
    rng = np.random.default_rng(seed)
    counted = _CountedF(f, n_evals)
    first = _first_start(x_seed, bounds)

    def neg(x):
        return -counted(x)

    options = {"maxfun" if method == "L-BFGS-B" else "maxiter": 1}
    while len(counted.y) < n_evals:
        if first is not None:
            x0, first = first, None
        else:
            x0 = rng.uniform(bounds[:, 0], bounds[:, 1])
        options["maxfun" if method == "L-BFGS-B" else "maxiter"] = \
            max(n_evals - len(counted.y), 1)
        if fd_eps is not None:
            options["eps"] = fd_eps
        try:
            scipy_minimize(neg, x0, method=method, bounds=bounds,
                           options=options)
        except _Budget:
            break
    return make_history(counted.X, counted.y, f"gradient-{method}", seed,
                        n_evals=n_evals, method=method)


# ------------------------------------------------------------------ adjoint

def run_adjoint(tp, bounds, n_evals: int, seed: int = 0) -> RunHistory:
    """Multistart L-BFGS-B with EXACT gradients from reverse-mode autodiff
    (adjoint.TorchLLTProblem). Each (f, grad) pair is charged 2 evaluations
    — one primal + one adjoint solve — independent of dimension, which is
    the whole point of an adjoint method vs finite differences (d+1).
    """
    bounds = np.asarray(bounds, dtype=float)
    rng = np.random.default_rng(seed)
    X: list[np.ndarray] = []
    y: list[float] = []

    def fun(x):
        if len(y) + 2 > n_evals:
            raise _Budget
        v, g = tp.f_and_grad(x)
        for _ in range(2):                    # primal + adjoint both charged
            X.append(np.array(x, dtype=float))
            y.append(v)
        return -v, -g

    while len(y) + 2 <= n_evals:
        x0 = rng.uniform(bounds[:, 0], bounds[:, 1])
        try:
            scipy_minimize(fun, x0, jac=True, method="L-BFGS-B", bounds=bounds)
        except _Budget:
            break
    return make_history(X, y, "adjoint", seed, n_evals=n_evals)


REGISTRY = {
    "random": run_random,
    "sobol": run_sobol,
    "grid": run_grid,
    "ga": run_ga,
    "gradient": run_gradient,
    "adjoint": run_adjoint,
}
