"""Finding a feasible design AT ALL — the layer that makes freeing a variable safe.

WHY THIS EXISTS. A user searched the wing+tail air family with the horizontal
stabiliser's arm SEARCHED over its published band and the run came back with
nothing; the same run with the arm FIXED at 5.5 m — a value inside that band —
returned a design. That is a contradiction on its face: the free box contains
the fixed value, so the free run's reachable set is a SUPERSET of the pinned
run's, and it was verified to be one (the two problems return bit-identical
``(f, g)`` at l_t = 5.5 m). The loss was entirely in the SEARCH, and it was
measured, on the user's own stored run
(``results/gui_runs/20260809T214809_519bbf1a_partial.json``, 20-D, budget 72,
n_init 10):

* 4.7 % of that design box is feasible (12 of 256 Sobol draws);
* 92.6 % of it is REFUSED — before any solve — by two declared, cheap gates:
  ``sizing.check_ar`` (the box's span x area rows span aspect ratios 1.6 to
  200 while the solvers are honest over 3 to 40) and
  ``sizing.check_wing_loading`` (the mission's own W/S ceiling);
* among the draws that pass those gates, 63.2 % FLY;
* so P(a 10-point Sobol init contains no FEASIBLE design) = 0.953^10 = 0.61 —
  the majority of runs start blind. "Feasible", not "solvable": that init did
  contain one design that solved (evaluation 9, f = 11.78, g = (-0.046,
  +3.703)) — it missed the static margin by 0.046. What
  ``best_f = y_t[feas].max() if feas.any() else y_t.min()`` needs is a point
  with ALL margins >= 0, and there was none;
* and a constrained BO run that starts blind does not recover. The collapse is
  PROGRESSIVE, and it is the OBJECTIVE GP that goes flat rather than all of
  them: with ``bo_refusal="worst"`` all 71 refusals are imputed onto that single
  solved value, so the objective GP's targets are one number, ``best_f`` equals
  it, and the posterior mean sits exactly on ``best_f`` at every candidate. The
  constraint GPs stay live (their posteriors move each iteration; p_feasible
  ranges 0 to 0.987). What kills the run is the objective posterior's sd at the
  argmax collapsing from 0.346 to 0.0078 with mu pinned at ``best_f`` — the
  improvement factor goes to zero — and p_feasible at the argmax going to zero
  with it. Left with no interior signal, ``optimize_acqf`` walks to the
  boundary: coordinates-on-a-bound per iteration run 7, 4, 4, 3, 2, 2, 3, 13,
  15, 19, 18, 16, 19, 17 and then all 20 from about iteration 25 onwards; 42 of
  the 62 iterations are full corners, and ``l_t_m`` took exactly three values in
  all 62 (3.0, 4.61, 8.0). 0 of 72 evaluations were feasible and the shell said,
  correctly for what it was given, that there was no solution.

DIMENSION IS NOT THE LEVER, and this bounds what the fix may promise. Pinning
the arm did not remove the failure mode, it re-rolled the same coin: over three
seeds at budget 72 the free run scored 0/0/12 feasible and the run pinned at
l_t_m = 5.5 scored 55/34/0 — the PINNED run returns nothing at seed 2, with 62
of 62 iterations on full corners. The same config re-run gives 27/34/37, and the
BLAS thread count alone flips the free seed-0 run between 0/72 and 2/72. Any
perturbation re-rolls whether the initial design contains a feasible point. So
the honest claim is not "free loses to pinned" but "a blind initial design
loses, and nothing in the acquisition recovers from it" — which is why the fix
belongs in the initial design, and why a GUARANTEE has to live in a SEED (a
design already known to fly, evaluated first) rather than in the acquisition.
Under ``screen`` those three seeds score 44/58/53 and under ``rescue``
50/55/56: not a luckier draw, a different distribution.

THE THREE THINGS THIS MODULE ADDS, and what each one buys:

``screen``
    A SCREENED INITIAL DESIGN. Draw a Sobol POOL rather than exactly n_init
    points, evaluate the pool in Sobol order, and keep the first n_init draws
    that are not refusals. A refusal costs no physics — measured on the user's
    own problem, a gate refusal returns in 0.1 ms against 12 ms for a design
    that is actually solved, 134x cheaper — so the pool is close to free, and
    it is charged accordingly (see :func:`screened_init` on the accounting).
    On the user's box this moves the init from 4.7 % to 63 % feasible per
    draw, i.e. P(blind init) from 0.61 to 5e-5. A probability, not a theorem: no
    published constrained-BO method guarantees feasibility, and the one result
    that does say "always finds a feasible solution" is about the minimiser of
    an exactly-penalised problem, which no finite-budget stochastic search
    inherits.

``rescue``
    Everything ``screen`` does, plus two things that only ever fire when the
    run is ALREADY failing: a FEASIBILITY PHASE (while no feasible design has
    been seen, stop asking for the best design and ask for a feasible one —
    maximise the surrogate's min-margin, the two-phase / feasibility-first
    answer of the constrained-optimisation literature: Deb 2000's feasibility
    rule, Gardner et al. 2014's feasibility-weighted EI, and explicitly
    Eriksson & Poloczek 2021's SCBO, which hunts a feasible point before it
    optimises). Inside that phase two degeneracies are caught rather than
    paid for: a margin surface with no gradient (every draw refused to the same
    sentinel — there is nothing to climb, so draw instead) and a surrogate
    optimum that repeats a point already evaluated, which is the
    flat-acquisition corner above. Both hand the decision back to a screened
    draw, and on the box this module was built for a screened draw flies 63 %
    of the time, so that is not a lesser option.

``off``
    The legacy path, and the default here. Bit-for-bit: the Sobol prefix of a
    pool is the pool's own first rows, so a box in which nothing is refused
    gives the identical initial design either way, and the escape and the
    phase are both gated on ``mode != "off"``.

WHY THE DEFAULT IS ``off`` AND THE SHELL'S DEFAULT IS NOT. Every frozen study
in this repo is a stored comparison between search methods; a mechanism that
changed what a run evaluates would change those numbers underneath their own
result files. So the api default is the legacy path and the V3 shell — which
is where a user meets this — asks for ``rescue``, exactly as it already asks
for ``bo_refusal="worst"`` over the api's ``"sentinel"``.

WHAT IS GUARANTEED AND WHAT IS NOT. Guaranteed by construction: the initial
design never spends a point on a design the physics refuses without solving
(when one that solves can be drawn); a search that has not yet found a
feasible design never repeats a point it has already evaluated; and the
feasibility phase holds every remaining iteration for as long as, and only as
long as, the run would otherwise return nothing — which is also why the escape
lives in the phase and not in the main BO loop, a loop that cannot run while
nothing feasible has been seen. NOT guaranteed: that a feasible design exists in
the box at all. Where none does, the answer is a report — which gate bound,
which row to widen (``gui.diagnose``) — and :func:`min_violation_index` names
the design that came closest, so "no solution" is never the whole answer.
"""

from __future__ import annotations

import numpy as np

# The GP stack is OPTIONAL here. api.run reaches this module on EVERY run,
# through _gate_grader, whatever optimiser was asked for — so an unguarded
# import at the top made torch a hard requirement of the whole program rather
# than of the three functions below that fit or sample from a GP. On a
# machine PyTorch has no wheel for (macOS x86_64; macOS 13 or older on arm64)
# that cost the app its front door. The names stay module-level, so the
# import order, the monkeypatch points and every number are unchanged; the
# failure is merely caught and named. See optimize/torch_optional.py.
from .torch_optional import require as _require_torch
from .torch_optional import torch

try:
    from botorch.acquisition.analytic import UpperConfidenceBound
    from botorch.optim import optimize_acqf
    from botorch.utils.sampling import draw_sobol_samples
except ImportError:                    # no torch here, so no botorch either
    UpperConfidenceBound = optimize_acqf = draw_sobol_samples = None

#: the failure contract's value (``objective.PENALTY``, restated here for the
#: same reason :mod:`optimize.refusal` restates it: the optimisers must not
#: import the physics layer)
PENALTY = -100.0

MODES: tuple[str, ...] = ("off", "screen", "rescue", "guide")
DEFAULT_MODE = "off"

#: how many Sobol draws the screen may spend per initial point it wants. 32 is
#: a documented cap, not a calibration: at the measured 4.7 % feasible fraction
#: of the box that motivated this module it is ~4 expected passers per wanted
#: point, and at a fraction low enough to exhaust it the run has learnt
#: something worth reporting (``n_screened``) rather than something worth
#: searching harder for.
SCREEN_POOL = 32

#: The cheap SIZE gate is a SUFFICIENT condition for a refusal: ``excess() >
#: 0`` means the vector's own aspect ratio or wing loading is outside a gate
#: that refuses before any solver runs. So a pool draw with ``excess() > 0``
#: can be skipped WITHOUT being evaluated and the kept set is unchanged —
#: verified bit-for-bit over 1536 draws on each of two live boxes, with zero
#: draws gated that were not also refused (742/742 and 405/405). It removes
#: 26-48 % of the pool's physics calls and buys nothing else: the ANSWER is
#: untouched, only its cost.
#:
#: It is a FILTER ON A POOL, never a refusal. A false positive would cost one
#: pool draw and can never cost a design, because the pool is 32x the design
#: and the top-up loop below still reaches every gated draw on demand.
_GATE_PREFILTER = True

#: UCB exploration weight in the feasibility phase. The phase maximises the
#: min-margin surrogate, so beta trades "where the model thinks it is
#: feasible" against "where the model does not know" — 2.0 is the repo's own
#: default elsewhere (``optimize.bo.run_bo``'s ``ucb_beta``).
RESCUE_BETA = 2.0

#: the tensor dtype the margin GP is fit in. None where PyTorch is absent;
#: nothing reads it before _require_torch has run.
_DTYPE = torch.double if torch is not None else None



def check_mode(mode: str | None) -> str:
    """Validate a feasibility mode, defaulting to the legacy path."""
    if mode is None:
        return DEFAULT_MODE
    m = str(mode)
    if m not in MODES:
        raise ValueError(f"unknown feasibility mode {mode!r}; "
                         f"choices: {MODES}")
    return m


def screens(mode: str) -> bool:
    """True where the initial design is screened."""
    return check_mode(mode) in ("screen", "rescue", "guide")


def rescues(mode: str) -> bool:
    """True where a run that finds nothing gets a feasibility phase."""
    return check_mode(mode) in ("rescue", "guide")


def grades(mode: str) -> bool:
    """True where a REFUSED design reports how far outside the gate it is.

    The phase :func:`rescue_candidate` runs climbs the min-margin surface, and
    a refusal reports the flat ``-1`` sentinel however far outside it is — so
    on a box that is mostly refusals the surface has no gradient, the phase
    declares itself degenerate and hands back to a uniform screened draw. That
    is a lottery where a direction was available: the two gates that do the
    refusing both compute a real distance and then throw it away
    (:class:`SizeGateMargin`).

    Behind its own mode because it changes what every refused evaluation
    RECORDS, and every frozen study in this repo is a stored comparison.
    """
    return check_mode(mode) == "guide"


def is_refusal(f) -> bool:
    """True iff this objective value is the failure sentinel.

    EXACT equality, which is the contract (:mod:`optimize.refusal`): an
    objective returns ``PENALTY`` itself, never something near it, and no
    real value in the registry — an L/D, a -Cd, a 0-100 composite — can
    reach -100.
    """
    v = float(f)
    return v == PENALTY


class SizeGateMargin:
    """How far outside the two cheap SIZE gates a design vector is, from the
    VECTOR alone — no solver, no physics import.

    Both gates that do 92.6 % of the refusing on the box this module was built
    for are inequalities in numbers the design vector already carries:

    * the aspect ratio ``b*b/S`` against ``sizing.AR_LIMITS``;
    * the wing loading ``W/S`` against the mission's own ceiling — or, in the
      searched-loading mode, the ``ws_pa`` row itself.

    So the distance is computable HERE, where the runner can use it, without
    touching a single family's failure path. That matters twice over: the
    physics contract (a refusal is exactly ``PENALTY``) is untouched, and the
    grading can be switched on per run rather than per package.

    :meth:`excess` returns 0.0 for a design inside both gates — including
    every design that reaches a solver — and otherwise the LARGEST relative
    overshoot, so the binding gate is the one that speaks.

    Not a feasibility measure: a design with ``excess() == 0`` may still be
    refused for a reason this class cannot see (a chord law that collapses the
    chord, a polar asked outside its validity). It is a DIRECTION for the
    feasibility phase, and the phase falls back exactly as before wherever the
    direction is flat.
    """

    def __init__(self, labels, *, ar_limits=None, cap_pa=None,
                 weight_n=None, ws_fixed_pa=None):
        labels = list(labels or ())

        def _at(name):
            return labels.index(name) if name in labels else None

        self.i_b = _at("b_m")
        self.i_s = _at("S_m2")
        self.i_ws = _at("ws_pa")
        self.ar_limits = (None if ar_limits is None
                          else (float(ar_limits[0]), float(ar_limits[1])))
        self.cap_pa = None if cap_pa is None else float(cap_pa)
        self.weight_n = None if weight_n is None else float(weight_n)
        #: the wing loading a DERIVED-area family flies when it is not a
        #: design row (``size_free="wing_loading"``: the mission states it).
        #: With it the area can be bounded, and so can the aspect ratio.
        self.ws_fixed_pa = None if ws_fixed_pa is None else float(ws_fixed_pa)

    @property
    def _ws_source(self):
        """Where this problem's wing loading comes from: the ``ws_pa`` design
        row, the mission's fixed number, or nowhere."""
        if self.i_ws is not None:
            return "row"
        if self.ws_fixed_pa and self.ws_fixed_pa > 0.0:
            return "fixed"
        return None

    @property
    def live(self) -> bool:
        """False when this problem states nothing this class can measure."""
        if self.ar_limits is not None and None not in (self.i_b, self.i_s):
            return True
        # ...and the DERIVED-area families, which have no S_m2 row at all
        # (``size_free`` in {"wing_loading", "wing_loading_free"}): the area
        # comes from S = W_total / (W/S). They were dead here — 444 registered
        # variants of them — while check_ar went on refusing their designs on
        # an area this class could not see.
        if (self.ar_limits is not None and self.i_b is not None
                and self.i_s is None and self.weight_n
                and self._ws_source is not None):
            return True
        if self.cap_pa is None:
            return False
        return self.i_ws is not None or (self.i_s is not None
                                         and self.weight_n is not None)

    def excess(self, x) -> float:
        """The binding gate's relative overshoot, 0.0 inside every gate."""
        x = np.asarray(x, dtype=float).ravel()
        worst = 0.0
        S = (float(x[self.i_s]) if self.i_s is not None
             and self.i_s < x.size else None)
        b = (float(x[self.i_b]) if self.i_b is not None
             and self.i_b < x.size else None)
        if self.ar_limits is not None and S and b and S > 0.0:
            ar_lo, ar_hi = self.ar_limits
            ar = b * b / S
            if ar > ar_hi:
                worst = max(worst, (ar - ar_hi) / ar_hi)
            elif ar < ar_lo:
                worst = max(worst, (ar_lo - ar) / ar_lo)
        elif (self.ar_limits is not None and b and self.i_s is None
                and self.weight_n):
            # DERIVED area: S = W_total / (W/S) and W_total = W_fixed + W_wing
            # with W_wing > 0, so S >= W_fixed / (W/S) and therefore
            #     AR = b^2 / S  <=  b^2 * (W/S) / W_fixed.
            # An UPPER bound only, so only the LOW side of the band is
            # decidable: if even the largest aspect ratio this design could
            # fly is under ar_lo, check_ar will refuse it whatever the weight
            # loop settles at. The high side needs an upper bound on the
            # total weight, which is exactly what the loop has not solved yet
            # — claiming it here would be a false positive, and a false
            # positive in a pre-filter costs a DESIGN.
            ws = None
            if self.i_ws is not None and self.i_ws < x.size:
                ws = float(x[self.i_ws])
            elif self.ws_fixed_pa:
                ws = self.ws_fixed_pa
            if ws and ws > 0.0:
                ar_lo = self.ar_limits[0]
                ar_upper = b * b * ws / self.weight_n
                if ar_upper < ar_lo:
                    worst = max(worst, (ar_lo - ar_upper) / ar_lo)
        if self.cap_pa and self.cap_pa > 0.0:
            ws = None
            if self.i_ws is not None and self.i_ws < x.size:
                ws = float(x[self.i_ws])
            elif S and S > 0.0 and self.weight_n:
                # a LOWER bound on the loading this design will fly: the
                # weight loop only ADDS the wing's own weight to the payload
                ws = self.weight_n / S
            if ws is not None and ws > self.cap_pa:
                worst = max(worst, (ws - self.cap_pa) / self.cap_pa)
        return float(worst)

    def graded(self, x, g):
        """``g`` with every margin replaced by a graded refusal value.

        ``-(1 + log1p(excess))``: at the gate's own boundary it is exactly the
        ``-1`` the failure contract already reports, it decreases monotonically
        with the overshoot, and the log keeps a 200-against-40 aspect ratio
        (excess 4) from sitting 4x further out than the boundary in a surface
        the GP standardises. Always <= -1, so a design that SOLVED and missed a
        limit by 0.046 still ranks above every design that never ran.
        """
        r = self.excess(x)
        if r <= 0.0:
            return g
        v = -(1.0 + float(np.log1p(r)))
        arr = np.atleast_1d(np.asarray(g, dtype=float))
        return np.full(arr.shape, v, dtype=float)


class _Screening:
    """Mark a wrapped callable as screening for as long as this is entered.

    The api wraps the objective in a counter that drives the progress bar and
    the stop rule (``api._ProgressCounter``). A screening draw that is REFUSED
    did no physics and must not be counted as an evaluation — otherwise a
    32x pool would run the progress bar 32x fast and hand the stop rule a
    budget that was never spent. A draw that SOLVES is kept as an initial
    point and is counted exactly once, so the run's accounting is unchanged.

    A plain callable (every direct-call test, every study script) has no such
    attribute and screening is simply invisible to it.
    """

    def __init__(self, fn):
        self.fn = fn
        self.had = None

    def __enter__(self):
        if hasattr(self.fn, "screening"):
            self.had = bool(self.fn.screening)
            self.fn.screening = True
        return self.fn

    def __exit__(self, *exc):
        if self.had is not None:
            self.fn.screening = self.had
        return False


def sobol_pool(bounds: np.ndarray, n: int, seed: int) -> np.ndarray:
    """``(n, d)`` scrambled Sobol draws — the runners' own initial design.

    Split out so the screen draws from the SAME sequence the legacy path does:
    botorch's Sobol engine is seeded per call, so the first k rows of an n-row
    draw are the k rows a k-row draw would have given (asserted in
    tests/test_feasible_init.py). That equality is what makes ``screen``
    bit-for-bit ``off`` on a box where nothing is refused, and it is also why
    the pool cannot shift any later RNG stream: the engine is local to the
    call and the global torch generator is untouched.
    """
    _require_torch("The Sobol pool the feasibility screen draws from")
    tb = torch.tensor(np.asarray(bounds, dtype=float).T, dtype=_DTYPE)
    X = draw_sobol_samples(bounds=tb, n=int(n), q=1, seed=int(seed))
    return X.squeeze(1).numpy()


def screened_init(f_and_g, bounds: np.ndarray, n_init: int, seed: int, *,
                  x_init: np.ndarray | None = None,
                  pool: int = SCREEN_POOL, constrained: bool = True,
                  gate=None):
    """The initial design, with refused draws skipped rather than paid for.

    Returns ``(X_list, y_list, g_rows, n_screened)`` — ``g_rows`` is a list of
    ``(m,)`` margin arrays for a constrained problem and an empty list
    otherwise, and ``n_screened`` counts the draws that were REFUSED and
    therefore discarded.

    ``x_init`` (optional) are seed points evaluated first and never screened —
    a warm start is a design someone chose, and dropping it because it does
    not fly would silently answer a different question than the caller asked.
    The Sobol part shrinks by their count, exactly as the legacy warm start
    does, so the budget is unchanged.

    THE RULE, and it is deterministic: evaluate the pool in Sobol order; keep
    a draw unless it is a refusal (:func:`is_refusal`); stop at ``n_init``
    kept points. If the whole pool cannot fill the init, top up from the
    refused draws IN SOBOL ORDER — so a box where every draw is refused gives
    exactly the legacy prefix and a run that cannot be helped is not also made
    different.

    ACCOUNTING. A refused draw is not charged: it returned before its solver
    ran (that is what the gate refusals in ``sizing``, ``geometry`` and the
    trim solves are), measured at 134x cheaper than a design that solves. A
    refusal that DID cost a solve — a divergence — is the same sentinel and
    cannot be told apart from a gate, so the pool is capped at ``pool`` draws
    per wanted point and the count comes back as ``n_screened``: the price is
    bounded and it is never silent.
    """
    bounds = np.asarray(bounds, dtype=float)
    X_list: list[np.ndarray] = []
    y_list: list[float] = []
    g_rows: list[np.ndarray] = []

    def _record(x, v, gg):
        X_list.append(np.array(x, dtype=float))
        y_list.append(float(v))
        if constrained:
            g_rows.append(np.atleast_1d(np.asarray(gg, dtype=float)))

    def _call(x):
        r = f_and_g(x)
        return (float(r[0]), r[1]) if constrained else (float(r), None)

    n_seed = 0
    if x_init is not None:
        seeds_arr = np.atleast_2d(np.asarray(x_init, dtype=float))
        if seeds_arr.shape[1] != bounds.shape[0]:
            raise ValueError(f"x_init dim {seeds_arr.shape[1]} != bounds dim "
                             f"{bounds.shape[0]}")
        if seeds_arr.shape[0] > n_init:
            raise ValueError(f"x_init has {seeds_arr.shape[0]} rows > "
                             f"n_init={n_init}")
        n_seed = seeds_arr.shape[0]
        for row in seeds_arr:
            v, gg = _call(row)
            _record(row, v, gg)

    n_want = int(n_init) - n_seed
    if n_want <= 0:
        return X_list, y_list, g_rows, 0

    P = sobol_pool(bounds, n_want * max(1, int(pool)), seed)
    # draws the pool did not KEEP, in Sobol order, each either already
    # evaluated (refused) or not yet (skipped by the cheap gate). ONE
    # list because the top-up must walk them in the order the ungated
    # screen would have, or the two designs differ by ordering alone.
    deferred: list[tuple] = []
    n_screened = 0
    with _Screening(f_and_g):
        for x in P:
            if len(X_list) - n_seed >= n_want:
                break
            if (_GATE_PREFILTER and gate is not None
                    and getattr(gate, "live", False)
                    and gate.excess(x) > 0.0):
                # a SUFFICIENT condition for the refusal this loop would have
                # skipped anyway, skipped here without paying the solver.
                # Counted as screened so the price of the pool stays visible,
                # and kept in `gated` so the top-up below can still reach it.
                n_screened += 1
                deferred.append((x, None, None))
                continue
            v, gg = _call(x)
            if is_refusal(v):
                n_screened += 1
                deferred.append((x, v, gg))
                continue
            _record(x, v, gg)
    # the pool could not fill the design: top up in Sobol order, which is the
    # legacy prefix when nothing passed at all. A draw that is KEPT was not
    # discarded, so it comes back out of the screened count — ``n_screened`` is
    # what the screen threw away, not what it looked at.
    with _Screening(f_and_g):
        for x, v, gg in deferred:
            if len(X_list) - n_seed >= n_want:
                break
            if v is None:
                # skipped by the cheap gate and never evaluated. The design
                # needs it after all, so it is paid for now — the pre-filter
                # may only change the COST, never which points come back.
                v, gg = _call(x)
            _record(x, v, gg)
            n_screened -= 1
    return X_list, y_list, g_rows, n_screened


def screened_draw(f_and_g, bounds: np.ndarray, rng, *,
                  tries: int = SCREEN_POOL, constrained: bool = True):
    """One uniform draw that is not a refusal, if ``tries`` can find one.

    The escape hatch of the main loop and the fallback of the feasibility
    phase. Returns ``(x, f, g, n_screened)`` with the LAST draw when every try
    was refused — a point that refuses is still a point, and refusing to
    return one would leave the loop with nothing to evaluate.
    """
    bounds = np.asarray(bounds, dtype=float)
    n_screened = 0
    x = last = None
    with _Screening(f_and_g):
        for _ in range(max(1, int(tries))):
            x = rng.uniform(bounds[:, 0], bounds[:, 1])
            r = f_and_g(x)
            v, gg = (float(r[0]), r[1]) if constrained else (float(r), None)
            last = (x, v, gg)
            if not is_refusal(v):
                return x, v, gg, n_screened
            n_screened += 1
    x, v, gg = last
    # the last refusal is the point that will be recorded, so it was not
    # screened away — only the tries before it were
    return x, v, gg, max(0, n_screened - 1)


def min_violation_index(y, G) -> int | None:
    """Index of the design that came CLOSEST to feasible, or None if empty.

    Ranked on the largest min-margin (the smallest violation of the binding
    constraint), ties broken by the objective. This is the design a run with
    no feasible point should be able to show: "nothing flew" and "nothing came
    close" are different answers and only one of them means widen the box.
    """
    y = np.asarray(y, dtype=float)
    if y.size == 0:
        return None
    G = np.asarray(G, dtype=float)
    if G.ndim == 1:
        G = G[:, None]
    mv = G.min(axis=1)
    best = mv.max()
    cand = np.flatnonzero(mv >= best - 1e-15)
    return int(cand[np.argmax(y[cand])])


def is_duplicate(x, X_list, tol: float = 1e-9) -> bool:
    """True iff ``x`` repeats a point already evaluated (within ``tol``).

    The signature of the flat acquisition this module exists to escape: with
    no feasible observation the GPs are flat, ``optimize_acqf`` returns the
    same box corner every iteration, and the run spends its whole budget
    re-evaluating one design. Measured on the reported run: 62 of 62
    iterations sat on a bound of the arm row.
    """
    if not len(X_list):
        return False
    A = np.asarray(X_list, dtype=float)
    d = np.abs(A - np.asarray(x, dtype=float)[None, :]).max(axis=1)
    return bool((d <= tol).any())


def _fit_margin_gp(X: np.ndarray, mv: np.ndarray, tb: torch.Tensor):
    """A GP on the MIN MARGIN — the surface the feasibility phase climbs."""
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from botorch.models.transforms.input import Normalize
    from botorch.models.transforms.outcome import Standardize
    from gpytorch.mlls import ExactMarginalLogLikelihood

    Xt = torch.tensor(np.asarray(X, dtype=float), dtype=_DTYPE)
    yt = torch.tensor(np.asarray(mv, dtype=float), dtype=_DTYPE)
    model = SingleTaskGP(
        Xt, yt.unsqueeze(-1),
        input_transform=Normalize(d=Xt.shape[-1], bounds=tb),
        outcome_transform=Standardize(m=1))
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    return model


def rescue_candidate(X_list, g_rows, bounds: np.ndarray, rng):
    """The next point to try while NO feasible design has been seen.

    Maximises UCB on the min-margin surrogate: the run stops asking for the
    best design and asks for a feasible one, which is the two-phase answer of
    the constrained literature (Deb 2000's feasibility rule ranks any feasible
    point above every infeasible one; SCBO, Eriksson & Poloczek 2021, spends
    an explicit phase finding one). Returns ``(x, diag)``; ``x`` is None when
    the surrogate could not be fitted or its optimum repeats a point already
    evaluated, and the caller then falls back to a screened draw — which is
    not a lesser option here: on the box this module was built for a screened
    uniform draw flies 63 % of the time.
    """
    diag: dict = {"phase": "feasibility"}
    G = np.asarray(g_rows, dtype=float)
    if G.ndim == 1:
        G = G[:, None]
    mv = G.min(axis=1)
    if not np.isfinite(mv).all() or float(mv.max() - mv.min()) <= 0.0:
        # a constant margin surface (every draw refused to the same sentinel)
        # carries no gradient to climb; the screened draw is the honest move
        diag["degenerate"] = True
        return None, diag
    _require_torch("The feasibility phase's rescue step")
    tb = torch.tensor(np.asarray(bounds, dtype=float).T, dtype=_DTYPE)
    try:
        model = _fit_margin_gp(np.asarray(X_list, dtype=float), mv, tb)
        af = UpperConfidenceBound(model, beta=RESCUE_BETA)
        x_next, acq = optimize_acqf(af, bounds=tb, q=1, num_restarts=10,
                                    raw_samples=256)
        x = x_next.squeeze(0).numpy()
        diag["acq"] = float(acq)
    except Exception:
        diag["fallback"] = True
        return None, diag
    if is_duplicate(x, X_list):
        diag["duplicate"] = True
        return None, diag
    return x, diag


__all__ = [
    "DEFAULT_MODE", "MODES", "PENALTY", "RESCUE_BETA", "SCREEN_POOL",
    "SizeGateMargin", "check_mode", "grades", "is_duplicate", "is_refusal",
    "min_violation_index", "rescue_candidate", "rescues", "screened_draw",
    "screened_init", "screens", "sobol_pool",
]
