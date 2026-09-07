"""The best design a case HAS, so a search can be scored against it.

WHY THIS EXISTS. Every optimiser number this repo has ever published is a
*best-found-at-budget-B*: the budget study's fitted law (``10 + 2.2 d``
evaluations for "90 % of the way") normalises each case by the best value ANY
arm reached in that sweep, so a method that is uniformly mediocre and a method
that is uniformly excellent produce the same 100 % if they are the only two
arms. That is a ranking, not a distance, and it cannot answer the question a
supervisor actually asks:

    does the search find the GLOBAL optimum, and how much budget does that
    take?

Nothing in the repo could answer it, because no case had a reference optimum.
This module builds one.

WHAT MAKES IT AFFORDABLE. The wing families are analytic (LLT/VLM, no XFOIL):
measured at 0.57-1.89 ms an evaluation over the eight families probed, and
``f(x)`` is bit-for-bit deterministic both within a process and across
processes. So a reference built from ~10^5-10^6 evaluations costs minutes per
case, against the 48-160 evaluations a real run is given. The section is the
opposite case (~4.5 s an XFOIL evaluation) and is deliberately NOT in scope
here: a reference for it can only ever be a best-known-solution, never an
exhausted box, and it must say so.

THE THREE THINGS A REFERENCE HAS TO CARRY, and why each one is not optional:

``f_star`` / ``x_star``
    the best FEASIBLE value found by everything below, pooled. On a
    constrained case the best infeasible point is usually better and is
    reported separately (``f_best_infeasible``) rather than quietly dropped —
    the gap between them is what the constraint costs.

``n_levels`` / ``frac_blind_at_star``
    how many distinct objective LEVELS the multistart converged into, and what
    fraction of BLIND local starts reached the best one. This is the part that
    answers "do we need a global method at all?". A case where blind L-BFGS-B
    lands on the global level 96 % of the time is a case where BO's entire
    apparatus is being paid to handle a difficulty the problem does not have.
    A case where it lands there 4 % of the time is where a global method earns
    its budget. Measured per family, because the answer differs between them.

    A level is a VALUE, not a location, and that distinction is load-bearing.
    Clustering the converged starts by distance instead reported 23 basins for
    the 3-D trim wing whose top five all sat at f = 35.697157 — one ridge
    counted 23 times. ``n_locations_at_star`` and ``invariant_at_star`` report
    that degeneracy separately: on the trim wing the invariant direction comes
    back as root and tip twist moving together, i.e. the absolute twist level,
    which trim absorbs into alpha. The 3-D trim wing searches a 2-D problem.

``on_bound``
    which coordinates of ``x_star`` sit on their box edge. A reference that
    rides a ceiling is reporting the BOX, not the physics, and every
    gap-to-optimum computed against it inherits that. The repo has hit this
    exact failure twice (``C_Z``-only endplate scoring riding ceilings; a
    searched W/S whose answer is always the top of its band), so the reference
    states it up front instead of leaving it for someone to rediscover.

WHAT THIS IS NOT. It is not a proof of global optimality — no black-box method
can supply one over a continuous box. It is a *certificate with a stated
budget*: this many Sobol points, this many polished starts, converged to this
many basins, best value ``f_star``. Quote it as "the best design found by
N evaluations of dense coverage plus M local polishes", and report the number
of independent starts that reached the top basin — the honest measure of how
hard the optimum was to find.

Usage::

    from aerobo import api
    from aerobo.optimize.reference import build_reference

    built = api.PROBLEM_SPECS["trim wing"].build({}, {}, None)
    ref = build_reference(built, n_sobol=65536, n_polish_top=256,
                          n_polish_random=256, seed=0)
    print(ref.f_star, ref.n_levels, ref.frac_blind_at_star, ref.on_bound)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Any, Callable

import numpy as np
from scipy.optimize import minimize as scipy_minimize
from scipy.stats import qmc

#: A converged start counts as reaching the same optimum as another when the
#: two points are within this distance in NORMALISED box coordinates (each
#: dimension scaled to [0, 1], Euclidean). 1e-2 over a unit box is 1 % of each
#: side — loose enough that two runs of the same basin stopping at slightly
#: different tolerances are one basin, tight enough that the synthetic
#: three-basin test in ``tests/test_global_reference.py`` still reads three.
BASIN_TOL = 1e-2

#: A coordinate is ON its bound when it is within this fraction of the box
#: width of an edge. Not exact equality: L-BFGS-B and SLSQP park on a bound to
#: within their own step tolerance, not at the floating-point value.
BOUND_TOL = 1e-6

#: A converged point this far OUTSIDE the feasible set is treated as a
#: boundary landing to be repaired, not as an infeasible point to be dropped.
#:
#: Measured, and it is not a nicety. On the synthetic problem
#: ``max x0 + x1  s.t.  x0^2 + x1^2 <= 1`` SLSQP converges to
#: (0.70710679, 0.70710678) — the exact optimum — with g = -1.5e-9, i.e. one
#: and a half parts in a billion outside. Under a strict ``g >= 0`` read that
#: point is infeasible, the polish falls back to the best interior point it
#: happened to pass, and it returned its own START (f = 0.5 against the true
#: 1.41421356). EVERY constrained family here has its optimum on a constraint
#: — the car wing on its drag cap, the tail on its static margin, the
#: hydrofoil on cavitation — so left unrepaired this would have made the
#: reference systematically report interior, worse designs on exactly the
#: cases the constraint makes interesting.
#:
#: The repair does NOT relax the published rule: :data:`FEASIBLE_TOL` stays at
#: zero and every reported optimum satisfies ``g >= 0`` strictly. The point is
#: moved back inside by bisection towards a known feasible point, so what is
#: published is a real feasible design that sits as close to the boundary as
#: double precision allows.
BOUNDARY_TOL = 1e-6

#: A margin ``g_i`` counts as satisfied at exactly zero. The families return
#: margins that are already scaled (the log stress ratio, the static-margin
#: difference), so no per-constraint tolerance is introduced here — a point
#: that misses by 1e-12 is feasible and one that misses by 0.046 is not, which
#: is the distinction ``optimize.feasible`` was written about.
FEASIBLE_TOL = 0.0

#: THE PLATEAU ESCAPE. A local start that cannot move is walked out of the
#: flat region by a compass probe at these fractions of the box width,
#: coarsest first, before it is allowed to report itself as an optimum.
#:
#: The ladder starts at HALF the box for a measured reason. Two failures have
#: the same shape and neither is exotic:
#:
#: * **a numerically flat tail.** On ``exp(-(x0 - x1 - 0.3)^2 / 2(0.05)^2)``
#:   a start at (0.9, 0.1) sits at f = 1.9e-22 with a finite-difference
#:   gradient of 4e-20, so L-BFGS-B stops after THREE evaluations at its own
#:   start — under its default ``gtol`` of 1e-5 that point is converged.
#:   Sixteen such starts came back as a second "level" at f ~ 1e-7 and a
#:   one-ridge landscape reported two optima;
#: * **a refusing corner.** A family that raises for x0 > 0.5 hands every
#:   probe the same large finite penalty, so the gradient is exactly zero and
#:   the polish returns -inf having never seen a value at all. Stepping out of
#:   a refusal band that wide is what the 0.5 rung is for.
#:
#: Both are the reference's own failure mode rather than the landscape's: a
#: level count and ``frac_blind_at_star`` built from stalled starts measure
#: scipy's stopping tolerance, not the problem. A genuinely converged optimum
#: pays for the ladder once — every probe is worse, nothing is returned, the
#: polish stops — so only stalls pay, and a stall is the case the reference
#: exists to get right.
ESCAPE_SCALES = (0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 3e-3, 1e-3)

#: How many times a polish may be restarted from an escaped point. Bounded
#: because a long shallow tail can be walked one probe at a time; no family
#: here has needed more than three, and the cap stops a pathological case
#: from spending the run.
ESCAPE_RESTARTS = 8


# =====================================================================
# Evaluating a built problem without an optimiser in the way
# =====================================================================

class Evaluator:
    """``built.callable`` as ``(f, g)`` for every problem, with a counter.

    Unconstrained problems get a zero-length margin vector, so one code path
    below serves both registries. A raising evaluation is recorded as a
    refusal (``f = -inf``, infeasible) rather than propagated: the reference
    sweeps far more of the box than any run does and WILL hit corners the
    families refuse, and losing a 65 000-point sweep to one of them would make
    the reference unbuildable on exactly the cases that need it most.

    ``n_evals`` is the honest cost of the certificate and is stored with it.
    """

    def __init__(self, built):
        self._call = built.callable
        self.is_constrained = bool(built.is_constrained)
        self.bounds = np.asarray(built.bounds, dtype=float)
        self.dim = int(built.dim)
        self.n_evals = 0
        self.n_refused = 0

    def __call__(self, x) -> tuple[float, np.ndarray]:
        self.n_evals += 1
        try:
            out = self._call(np.asarray(x, dtype=float))
        except Exception:                     # noqa: BLE001 - see docstring
            self.n_refused += 1
            return -math.inf, np.array([-math.inf])
        if self.is_constrained:
            f, g = out
            g = np.atleast_1d(np.asarray(g, dtype=float)).ravel()
        else:
            f, g = out, np.zeros(0)
        f = float(f)
        if not math.isfinite(f):
            self.n_refused += 1
            return -math.inf, g
        return f, g

    def batch(self, X) -> tuple[np.ndarray, np.ndarray]:
        """``(f, G)`` over a stack of points; ``G`` is ``(n, m)``."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        vals = [self(x) for x in X]
        f = np.array([v for v, _ in vals], dtype=float)
        m = max((gg.size for _, gg in vals), default=0)
        G = np.full((len(vals), m), np.nan, dtype=float)
        for i, (_, gg) in enumerate(vals):
            if gg.size:
                G[i, :gg.size] = gg
        return f, G


#: A LEVEL REACHED BY ONE START IS NOT A CERTIFICATE.
#:
#: Measured, not assumed: the two car wings reach their top level from exactly
#: ONE of 512 polished starts, and their f_star moved by 0.34 % and 0.16 %
#: when the whole build was repeated at a different Sobol seed. The other
#: thirteen cases reach it from 148-256 blind starts and reproduce to 7.6e-9.
#: The difference is not the landscape being unfair, it is a maximum over a
#: sample whose winner is unique — a statistic with an effective size of one,
#: which cannot be reproducible and must not be published to nine figures.
#:
#: Treating the starts as independent draws that hit the top level with
#: probability p = k/N, a rebuild of N starts finds it again with probability
#: 1 - (1 - k/N)^N -> 1 - exp(-k). So k = 1 gives 63 %, k = 3 gives 95 %,
#: k = 5 gives 99 %. Three is the threshold below which this module refuses to
#: call a value certified.
#:
#: The estimate is an UPPER bound, deliberately: starts polished from the top
#: of the sweep are correlated with the sweep and with each other, so the true
#: reproduction probability is lower than this formula says. A case that fails
#: the test on the upper bound fails it for certain.
MIN_STARTS_AT_STAR = 3


def reproducibility(rec) -> dict:
    """Would a rebuild at another seed find this same level again?

    Reads only fields every stored record already has, so it applies to
    certificates built before the check existed.
    """
    k = int(rec.get("n_starts_at_star", 0) or 0)
    p_upper = 1.0 - math.exp(-k) if k > 0 else 0.0
    return {
        "n_starts_at_star": k,
        "p_reproduce_upper_bound": round(p_upper, 6),
        "certifiable": bool(k >= MIN_STARTS_AT_STAR),
        "reason": (
            "" if k >= MIN_STARTS_AT_STAR else
            f"the top level was reached by {k} start"
            f"{'' if k == 1 else 's'} of {rec.get('n_polish', '?')}; a rebuild "
            f"at another seed finds it again with probability at most "
            f"{p_upper:.2f}, so this f_star is a single draw and NOT a "
            f"certificate"),
    }


def is_feasible(g) -> bool:
    """Every margin satisfied (vacuously true for an unconstrained point)."""
    g = np.asarray(g, dtype=float).ravel()
    if g.size == 0:
        return True
    if not np.all(np.isfinite(g)):
        return False
    return bool(np.all(g >= FEASIBLE_TOL))


def feasible_mask(G) -> np.ndarray:
    G = np.atleast_2d(np.asarray(G, dtype=float))
    if G.shape[1] == 0:
        return np.ones(G.shape[0], dtype=bool)
    return np.all(np.isfinite(G), axis=1) & np.all(G >= FEASIBLE_TOL, axis=1)


def rank_order(f, G) -> np.ndarray:
    """Indices best-first: every feasible point above every infeasible one.

    Within each group the ordering is by objective, so "the best 256 points"
    means the best feasible ones and only falls back to infeasible starts when
    the sweep found fewer than 256 that fly. A constrained polish started from
    an infeasible point is the case SLSQP handles worst, and the repo has
    already measured what a blind constrained start does (0 of 72 evaluations
    feasible, and no recovery) — so the ordering is not cosmetic.

    LEXICOGRAPHIC, not a weighted key. The first version returned
    ``feasible * 1e12 + f`` and it silently destroyed the ordering: float64
    resolution at 1e12 is 2.2e-4, so every objective difference finer than
    that vanished into the offset. On the trim wing (f ~ 35.697) that turned
    one optimum into **13 interleaved, non-monotone "levels"**. Any comparison
    that folds a flag and a value into one float has the same defect; the fix
    is to sort on the pair.
    """
    f = np.asarray(f, dtype=float)
    f = np.where(np.isfinite(f), f, -np.inf)
    feas = feasible_mask(G).astype(np.int8)
    # np.lexsort orders by the LAST key first, ascending; negate for best-first
    return np.lexsort((-f, -feas))


def argbest(f, G) -> int:
    """Index of the best point (feasible-first, then objective)."""
    return int(rank_order(f, G)[0])


# =====================================================================
# Local polish
# =====================================================================

def violation(g) -> float:
    """How far outside the feasible set a point is (0.0 when inside)."""
    g = np.atleast_1d(np.asarray(g, dtype=float)).ravel()
    if g.size == 0:
        return 0.0
    if not np.all(np.isfinite(g)):
        return math.inf
    return float(max(0.0, -np.min(g)))


def repair_to_feasible(ev: Evaluator, x_out, x_in, iters: int = 60
                       ) -> tuple[np.ndarray, float, np.ndarray] | None:
    """Bisect from a feasible point towards a boundary landing.

    ``x_in`` must be feasible and ``x_out`` infeasible; the segment between
    them crosses the boundary, so bisection converges on it. Returns the
    feasible side, which is the best strictly-feasible design on that segment
    to within ``2^-iters`` of the crossing — 60 halvings exhausts double
    precision, and at 0.6-3.4 ms an evaluation the whole repair costs less
    than a tenth of a second.

    ``None`` when the premise does not hold (either endpoint mis-typed), so a
    caller cannot quietly publish an unrepaired point believing it was fixed.
    """
    x_out = np.asarray(x_out, dtype=float)
    x_in = np.asarray(x_in, dtype=float)
    f_in, g_in = ev(x_in)
    if not is_feasible(g_in):
        return None
    best = (x_in, f_in, g_in)
    lo, hi = x_in, x_out                      # lo feasible, hi infeasible
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        f_m, g_m = ev(mid)
        if math.isfinite(f_m) and is_feasible(g_m):
            lo = mid
            if f_m > best[1]:
                best = (mid, f_m, g_m)
        else:
            hi = mid
    return best


def polish(ev: Evaluator, x0, maxiter: int = 500) -> tuple[np.ndarray, float,
                                                           np.ndarray]:
    """Run one local optimiser from ``x0`` to ITS convergence, uncapped.

    L-BFGS-B on the objective when unconstrained; SLSQP with the native
    inequality margins when constrained — the same two methods
    ``optimize.baselines`` and ``optimize.constrained`` benchmark, so a basin
    this finds is a basin those arms could in principle reach. Gradients are
    finite-difference, which is what a black-box family gets.

    No evaluation cap: a reference is allowed to spend what a run may not.
    The returned point is re-evaluated so the stored ``(f, g)`` come from the
    same contract everything else here uses rather than from scipy's own
    bookkeeping (which negates, and which reports the merit on the penalty
    path rather than the objective).
    """
    bounds = ev.bounds
    x0 = np.clip(np.asarray(x0, dtype=float), bounds[:, 0], bounds[:, 1])
    memo: dict[bytes, tuple[float, np.ndarray]] = {}

    def cached(x):
        key = np.asarray(x, dtype=float).tobytes()
        hit = memo.get(key)
        if hit is None:
            hit = ev(x)
            memo[key] = hit
        return hit

    def neg_f(x):
        f, _ = cached(x)
        # scipy cannot step away from an infinite value; hand the refusal a
        # large finite number so the line search retreats instead of stalling
        return -f if math.isfinite(f) else 1e12

    width = np.where(bounds[:, 1] > bounds[:, 0],
                     bounds[:, 1] - bounds[:, 0], 1.0)

    def run_scipy(x_start):
        try:
            if ev.is_constrained:
                _, g_start = cached(x_start)
                m = int(np.asarray(g_start).size)
                cons = [{"type": "ineq",
                         "fun": (lambda x, j=j: float(
                             np.atleast_1d(cached(x)[1])[j]
                             if np.all(np.isfinite(cached(x)[1])) else -1e6))}
                        for j in range(m)]
                scipy_minimize(neg_f, x_start, method="SLSQP", bounds=bounds,
                               constraints=cons, options={"maxiter": maxiter})
            else:
                scipy_minimize(neg_f, x_start, method="L-BFGS-B",
                               bounds=bounds,
                               options={"maxfun": maxiter * max(ev.dim, 1)})
        except Exception:                      # noqa: BLE001
            pass

    def rank_key(f, g):
        """Feasible-first, then value — the ordering ``argbest`` publishes."""
        return (1 if is_feasible(g) else 0,
                f if math.isfinite(f) else -math.inf)

    def escape(x_from):
        """The best strictly-better compass probe out of a plateau, or None.

        The coarsest rung that improves anything wins: a flat region is left
        by the widest step that reaches signal, and refining first spends
        2*dim evaluations a rung learning that the plateau is still flat.
        """
        k_from = rank_key(*cached(x_from))
        for s in ESCAPE_SCALES:
            found = None
            for i in range(ev.dim):
                for sgn in (1.0, -1.0):
                    xp = np.array(x_from, dtype=float)
                    xp[i] = min(max(xp[i] + sgn * s * width[i],
                                    bounds[i, 0]), bounds[i, 1])
                    if xp[i] == x_from[i]:
                        continue
                    k = rank_key(*cached(xp))
                    if k > k_from and (found is None or k > found[0]):
                        found = (k, xp)
            if found is not None:
                return found[1]
        return None

    def best_seen():
        """The best point the polish actually SAW, ranked feasible-first.

        scipy's own ``res.x`` is the last iterate, which on a failed line
        search can be worse than a point it already visited.
        """
        xs = np.array([np.frombuffer(k, dtype=float) for k in memo])
        fs = np.array([v[0] for v in memo.values()], dtype=float)
        gs = [v[1] for v in memo.values()]
        m = max((np.asarray(g).size for g in gs), default=0)
        G = np.full((len(gs), m), np.nan)
        for i, g in enumerate(gs):
            g = np.asarray(g, dtype=float).ravel()
            if g.size:
                G[i, :g.size] = g
        i = argbest(fs, G)
        return xs, fs, gs, G, i

    # Polish, then CERTIFY: a compass probe at every rung of ESCAPE_SCALES has
    # to fail to beat the point the polish stopped on. If one beats it, the
    # stop was a tolerance and not an optimum — restart from the better point.
    # Only "no probe is better" ends this loop, so what a polish returns is a
    # point no step this ladder can take improves on, rather than wherever
    # scipy's gradient test happened to give up.
    x_start = np.asarray(x0, dtype=float)
    for _ in range(ESCAPE_RESTARTS + 1):
        run_scipy(x_start)
        xs_now, _, _, _, i_now = best_seen()
        nxt = escape(xs_now[i_now])
        if nxt is None:
            break
        x_start = nxt

    if not memo:
        f0, g0 = ev(x0)
        return np.asarray(x0, dtype=float), f0, g0
    xs, fs, gs, G, i = best_seen()
    best = (xs[i], float(fs[i]), np.asarray(gs[i], dtype=float))
    if not ev.is_constrained:
        return best

    # A converged SLSQP point sits ON the constraint, which in floating point
    # means a hair outside it about half the time. Those points are the
    # answer, not noise: recover them by bisecting back inside rather than
    # falling through to whatever interior point the search happened to pass.
    feas = feasible_mask(G)
    near = [k for k in range(len(fs))
            if math.isfinite(fs[k]) and not feas[k]
            and violation(gs[k]) <= BOUNDARY_TOL * max(1.0, abs(fs[k]))
            and fs[k] > best[1]]
    if not near or not feas.any():
        return best
    for k in sorted(near, key=lambda k: -fs[k]):
        got = repair_to_feasible(ev, xs[k], best[0])
        if got is not None and got[1] > best[1]:
            best = got
        break                                 # the best near-boundary point
    return best


# =====================================================================
# Basins
# =====================================================================

def normalise(X, bounds) -> np.ndarray:
    bounds = np.asarray(bounds, dtype=float)
    width = np.where(bounds[:, 1] > bounds[:, 0],
                     bounds[:, 1] - bounds[:, 0], 1.0)
    return (np.atleast_2d(np.asarray(X, dtype=float)) - bounds[:, 0]) / width


#: Two converged starts are at the SAME objective level when their values
#: differ by less than ``F_RTOL * max(1, |f|)``. Relative, because the
#: families' objectives span L/D ~ 35, downforce coefficients ~ 0.7 and
#: composite scores ~ 75, and one absolute tolerance cannot serve all three.
#:
#: 1e-5 and not tighter, for a MEASURED reason: re-polishing an
#: already-polished point moves the objective by up to 1.5e-6 (car rear wing,
#: four starts), so a tolerance at 1e-6 would split one converged optimum into
#: several on that family. Every reference also stores
#: :attr:`Reference.n_levels_at_rtol` over four decades of this tolerance, so
#: a level count can be checked against the choice rather than trusted.
F_RTOL = 1e-5

#: The tolerances the level count is reported at, so the reader can see
#: whether it is a property of the landscape or of the tolerance.
F_RTOL_SCAN = (1e-7, 1e-6, 1e-5, 1e-4, 1e-3)


@dataclass
class Optimum:
    """One distinct objective LEVEL, and the set of designs that reach it.

    ``n_locations`` is how many geometrically distinct designs in this level
    the polish converged to, and ``spread`` is the largest normalised distance
    between any two of them. The pair is the whole point: an optimum with
    ``n_locations`` > 1 and a large ``spread`` is not several optima that
    happen to tie, it is ONE optimum that the design vector cannot pin down —
    a ridge, a non-identifiable direction, a redundant parameterisation.
    """

    x: list
    f: float
    g: list
    feasible: bool
    n_starts: int
    n_locations: int
    spread: float
    invariant: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


def _distance_clusters(Z, tol: float) -> int:
    """How many groups within ``tol`` (greedy, order as given)."""
    centres: list[np.ndarray] = []
    for z in Z:
        if not any(float(np.linalg.norm(z - c)) <= tol for c in centres):
            centres.append(z)
    return len(centres)


def _invariant_direction(Z, bounds_labels) -> list:
    """The direction the objective does NOT care about, if there is one.

    Principal component of the top level's points in normalised coordinates,
    reported only when that level actually spreads. On the trim wing this
    comes back as (0, +0.71, +0.71) — root and tip twist moving TOGETHER,
    which is the absolute twist level, which trim absorbs into alpha. So the
    3-D trim wing searches a 2-D problem, and an optimiser "converging
    somewhere else" on it is not finding another optimum.
    """
    Z = np.atleast_2d(np.asarray(Z, dtype=float))
    if Z.shape[0] < 3:
        return []
    C = Z - Z.mean(axis=0)
    if not np.isfinite(C).all() or float(np.max(np.abs(C))) < 1e-9:
        return []
    _, s, vt = np.linalg.svd(C, full_matrices=False)
    if s.size == 0 or s[0] <= 0:
        return []
    v = vt[0]
    # sign convention: largest-magnitude entry positive, so two runs of the
    # same study do not report the same direction with opposite signs
    if v[int(np.argmax(np.abs(v)))] < 0:
        v = -v
    frac = float(s[0] ** 2 / np.sum(s ** 2))
    return [{"component": [float(t) for t in np.round(v, 6)],
             "variance_frac": round(frac, 4),
             "labels": list(bounds_labels)}]


def cluster_optima(X, f, G, bounds, f_rtol: float = F_RTOL,
                   tol: float = BASIN_TOL,
                   labels=()) -> list[Optimum]:
    """Group converged starts into distinct objective LEVELS, best first.

    Levels, not locations, and the ordering matters. The first version of this
    clustered on distance alone and reported **23 basins** for the 3-D trim
    wing whose top five all sat at f = 35.697157 — one ridge, counted 23
    times, which would have made every family look pathologically multimodal
    and made "does the search find the global optimum?" unanswerable. A
    distinct optimum is a distinct VALUE; how many designs reach that value is
    a second, separately reported property (``n_locations``/``spread``).

    Refusals (``f = -inf``) are dropped; a start that converged to nothing is
    not an optimum. Infeasible points form their own levels BELOW every
    feasible one, via :func:`rank_order`.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    f = np.asarray(f, dtype=float)
    G = np.atleast_2d(np.asarray(G, dtype=float)) if np.size(G) else \
        np.zeros((X.shape[0], 0))
    Z = normalise(X, bounds)
    feas = feasible_mask(G)
    order = [int(i) for i in rank_order(f, G) if math.isfinite(f[i])]

    out: list[Optimum] = []
    members: list[int] = []

    def flush():
        if not members:
            return
        c = members[0]
        Zi = Z[members]
        pair = 0.0
        if len(members) > 1:
            pair = float(max(np.linalg.norm(a - b)
                             for i, a in enumerate(Zi) for b in Zi[i + 1:]))
        out.append(Optimum(
            x=[float(v) for v in X[c]], f=float(f[c]),
            g=([float(v) for v in np.atleast_1d(G[c])] if G.shape[1] else []),
            feasible=bool(feas[c]), n_starts=len(members),
            n_locations=_distance_clusters(Zi, tol), spread=round(pair, 6),
            invariant=_invariant_direction(Zi, labels) if pair > tol else []))

    for i in order:
        if members:
            f0 = f[members[0]]
            same_level = abs(f0 - f[i]) <= f_rtol * max(1.0, abs(f0))
            if same_level and feas[i] == feas[members[0]]:
                members.append(i)
                continue
            flush()
            members = []
        members.append(i)
    flush()
    return out


def on_bound(x, bounds, tol: float = BOUND_TOL) -> list[int]:
    """Indices of coordinates sitting on their box edge."""
    x = np.asarray(x, dtype=float)
    bounds = np.asarray(bounds, dtype=float)
    width = np.where(bounds[:, 1] > bounds[:, 0],
                     bounds[:, 1] - bounds[:, 0], 1.0)
    lo = (x - bounds[:, 0]) / width <= tol
    hi = (bounds[:, 1] - x) / width <= tol
    return [int(i) for i in np.flatnonzero(lo | hi)]


# =====================================================================
# The reference itself
# =====================================================================

@dataclass
class Reference:
    """The best design a case has, and the budget that certifies it."""

    case: str
    dim: int
    is_constrained: bool
    param_labels: list
    bounds: list
    f_star: float
    x_star: list
    g_star: list
    source: str                     # which stage produced x_star
    n_levels: int                   # distinct objective levels among polishes
    n_starts_at_star: int           # blind + top starts reaching the best level
    frac_starts_at_star: float      # THE reachability number (see docstring)
    frac_blind_at_star: float       # the same, over RANDOM starts only
    level_gap: float | None = None  # f_star minus the second-best level
    n_locations_at_star: int = 1    # distinct designs achieving f_star
    spread_at_star: float = 0.0     # normalised diameter of that set
    invariant_at_star: list = field(default_factory=list)
    optima: list = field(default_factory=list)
    on_bound: list = field(default_factory=list)
    n_levels_at_rtol: dict = field(default_factory=dict)
    star_repolish_gain: float = 0.0
    f_best_infeasible: float | None = None
    f_sobol_best: float | None = None
    n_sobol: int = 0
    n_polish: int = 0
    n_evals: int = 0
    n_refused: int = 0
    frac_feasible_sobol: float | None = None
    seed: int = 0
    basin_tol: float = BASIN_TOL

    def as_dict(self) -> dict:
        d = asdict(self)
        d["optima"] = [b if isinstance(b, dict) else b.as_dict()
                       for b in self.optima]
        return d


def build_reference(built, case: str = "", n_sobol: int = 1 << 16,
                    n_polish_top: int = 256, n_polish_random: int = 256,
                    seed: int = 0, maxiter: int = 500,
                    basin_tol: float = BASIN_TOL,
                    progress: Callable[[str], Any] | None = None
                    ) -> Reference:
    """Dense coverage, then polish from both the best and from fresh starts.

    The two start populations answer two different halves of the question and
    neither replaces the other. The ``top`` starts ask *how good does the best
    design get when you refine it* — they inherit the sweep's coverage, and
    they are what a well-initialised run approximates. The ``random`` starts
    ask *how many basins are there* — they are drawn blind precisely so the
    basin count is not conditioned on the sweep having already found the good
    region, which would collapse it towards one every time.

    ``n_sobol`` is drawn from a scrambled Sobol sequence at the given seed,
    so a second seed is a genuinely different certificate and the two can be
    compared (they should agree on ``f_star``; disagreement is the honest
    signal that the box is not yet exhausted).
    """
    ev = Evaluator(built)
    bounds = ev.bounds
    d = ev.dim
    labels = [str(v) for v in getattr(built, "param_labels", ())]

    def say(msg: str):
        if progress is not None:
            progress(msg)

    # ---- stage A: dense coverage -----------------------------------
    say(f"{case}: sobol {n_sobol}")
    sampler = qmc.Sobol(d=d, scramble=True, seed=seed)
    n_pow = max(0, int(math.ceil(math.log2(max(n_sobol, 1)))))
    Xs = qmc.scale(sampler.random_base2(n_pow), bounds[:, 0], bounds[:, 1])
    Xs = Xs[:n_sobol]
    fs, Gs = ev.batch(Xs)
    feas_s = feasible_mask(Gs)
    frac_feas = float(feas_s.mean()) if feas_s.size else None
    sobol_best = float(np.max(fs[feas_s])) if feas_s.any() else None

    # ---- stage B: polish -------------------------------------------
    order = rank_order(fs, Gs)
    starts = [Xs[i] for i in order[:n_polish_top]]
    rng = np.random.default_rng(seed + 991)
    starts += [rng.uniform(bounds[:, 0], bounds[:, 1])
               for _ in range(n_polish_random)]
    say(f"{case}: polishing {len(starts)} starts")
    Xp, fp, Gp = [], [], []
    for j, x0 in enumerate(starts):
        x, f, g = polish(ev, x0, maxiter=maxiter)
        Xp.append(x)
        fp.append(f)
        Gp.append(np.atleast_1d(np.asarray(g, dtype=float)))
        if progress is not None and (j + 1) % 64 == 0:
            say(f"{case}: polished {j + 1}/{len(starts)}  "
                f"best {max(v for v in fp if math.isfinite(v)):.6f}")
    m = max((g.size for g in Gp), default=0)
    GP = np.full((len(Gp), m), np.nan)
    for i, g in enumerate(Gp):
        if g.size:
            GP[i, :g.size] = g
    Xp = np.array(Xp, dtype=float)
    fp = np.array(fp, dtype=float)

    # ---- stage C: pool, rank, cluster -------------------------------
    Xall = np.vstack([Xs, Xp])
    fall = np.concatenate([fs, fp])
    mm = max(Gs.shape[1], GP.shape[1])
    Gall = np.full((Xall.shape[0], mm), np.nan)
    if Gs.shape[1]:
        Gall[:Xs.shape[0], :Gs.shape[1]] = Gs
    if GP.shape[1]:
        Gall[Xs.shape[0]:, :GP.shape[1]] = GP
    feas_all = feasible_mask(Gall)

    i_star = argbest(fall, Gall)
    x_star, f_star = Xall[i_star], float(fall[i_star])
    g_star = Gall[i_star] if mm else np.zeros(0)
    source = "sobol" if i_star < Xs.shape[0] else (
        "polish-top" if i_star - Xs.shape[0] < n_polish_top
        else "polish-random")

    # ---- stage C1: the certificate's own convergence check ----------
    # A reference whose best point is not itself converged is not a
    # reference; it is one more mediocre run. Re-polish x_star at four times
    # the iteration cap and keep the result if it improves. The gain is
    # STORED, so a reader can see whether the certificate was already at rest
    # (measured at < 2e-6 on the families probed) or was still moving.
    say(f"{case}: re-polishing the best point")
    x_re, f_re, g_re = polish(ev, x_star, maxiter=maxiter * 4)
    repolish_gain = 0.0
    if math.isfinite(f_re) and is_feasible(g_re) and f_re > f_star:
        repolish_gain = float(f_re - f_star)
        x_star, f_star = x_re, float(f_re)
        g_star = np.atleast_1d(np.asarray(g_re, dtype=float)) if mm \
            else np.zeros(0)
        source = source + "+repolish"

    optima = cluster_optima(Xp, fp, GP, bounds, tol=basin_tol, labels=labels)

    # the best LEVEL among the polished starts, matched to f_star by value.
    # Not by position: the sweep occasionally holds the best point outright
    # (source == "sobol"), and then no polish sits exactly on it.
    top = optima[0] if optima else None
    n_at_star = top.n_starts if top else 0
    level_gap = (float(top.f - optima[1].f)
                 if top is not None and len(optima) > 1 else None)
    # over BLIND starts only — the top-of-sweep starts inherit the sweep's
    # coverage, so counting them would answer "does a polish improve a good
    # point" when the question is "does an unaided local method find the
    # global optimum".
    n_blind = 0
    if top is not None and n_polish_random:
        z_top = normalise(np.array(top.x)[None, :], bounds)[0]
        for k in range(n_polish_top, len(starts)):
            if abs(fp[k] - top.f) <= F_RTOL * max(1.0, abs(top.f)):
                n_blind += 1
        del z_top

    infeasible_best = None
    if mm and (~feas_all).any():
        vals = fall[~feas_all]
        vals = vals[np.isfinite(vals)]
        if vals.size:
            infeasible_best = float(np.max(vals))

    return Reference(
        case=case or getattr(built, "display", ""),
        dim=d, is_constrained=ev.is_constrained, param_labels=labels,
        bounds=[[float(a), float(b)] for a, b in bounds],
        f_star=f_star, x_star=[float(v) for v in x_star],
        g_star=[float(v) for v in np.atleast_1d(g_star)] if mm else [],
        source=source, n_levels=len(optima), n_starts_at_star=n_at_star,
        frac_starts_at_star=(round(n_at_star / len(starts), 4)
                             if starts else 0.0),
        frac_blind_at_star=(round(n_blind / n_polish_random, 4)
                            if n_polish_random else 0.0),
        level_gap=level_gap,
        n_locations_at_star=(top.n_locations if top else 0),
        spread_at_star=(top.spread if top else 0.0),
        invariant_at_star=(top.invariant if top else []),
        optima=[b.as_dict() for b in optima],
        on_bound=on_bound(x_star, bounds),
        n_levels_at_rtol={
            f"{r:g}": len(cluster_optima(Xp, fp, GP, bounds, f_rtol=r,
                                         tol=basin_tol, labels=labels))
            for r in F_RTOL_SCAN},
        star_repolish_gain=repolish_gain,
        f_best_infeasible=infeasible_best, f_sobol_best=sobol_best,
        n_sobol=int(Xs.shape[0]), n_polish=len(starts),
        n_evals=ev.n_evals, n_refused=ev.n_refused,
        frac_feasible_sobol=frac_feas, seed=seed, basin_tol=basin_tol,
    )


# =====================================================================
# Scoring a search against a reference
# =====================================================================

def gap_to_reference(y, f_star: float, maximize: bool = True) -> float:
    """How far the best value in ``y`` finished from the reference optimum.

    Absolute, in the objective's own units, and POSITIVE when the run fell
    short. Negative means the run beat the reference, which is not a better
    run — it is a reference that needs rebuilding, and the caller should treat
    it as one (``scripts/global_reference.py`` refuses to publish a table
    containing one).
    """
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return math.inf
    best = float(np.max(y)) if maximize else float(np.min(y))
    return (f_star - best) if maximize else (best - f_star)


def evals_to_within(y, f_star: float, eps: float,
                    maximize: bool = True) -> int | None:
    """Evaluations until within ``eps`` of the reference, or ``None``.

    This is the budget number that means something across cases: "reaches
    within 1 % of the known optimum" is the same statement on a hydrofoil and
    on a car wing, where "90 % of the way to the best this sweep found"
    silently changes meaning with the field of arms.
    """
    y = np.asarray(y, dtype=float)
    target = f_star - eps if maximize else f_star + eps
    hit = y >= target if maximize else y <= target
    idx = np.flatnonzero(hit)
    return int(idx[0]) + 1 if idx.size else None


def relative_gap(y, f_star: float, f_ref0: float | None = None,
                 maximize: bool = True) -> float:
    """``gap_to_reference`` as a fraction of the reference value's magnitude.

    ``f_ref0`` (optional) is a floor to measure the span from — pass the
    median blind draw when one exists, so the fraction is "of the improvement
    that was available" rather than "of the objective's arbitrary scale".
    """
    gap = gap_to_reference(y, f_star, maximize=maximize)
    if not math.isfinite(gap):
        return math.inf
    span = abs(f_star - f_ref0) if f_ref0 is not None else abs(f_star)
    return gap / span if span > 0 else math.inf
