"""Why a constrained run came back with NOTHING — read off its own log.

A search that finds no feasible design is the one outcome the shells used to
answer with a count and a shrug: "0 of 64 evaluations were feasible … widen
the design box, or move the mission, and run it again". Which box row, and
which way? The run already knows. Every :class:`api.RunResult` carries
``eval_x`` (where it looked), ``eval_g`` (the signed margin of every
constraint at every one of those points) and ``bounds`` (the box it was
allowed to look in), so the three questions a user actually has —

1. WHICH constraint stopped it, and by how much,
2. is that because no single design could hold both at once, or because one
   of them is simply out of reach in this box,
3. WHICH box row would have to move, and in WHICH direction —

are all answerable from the record without re-running any physics.

The third answer is a measurement, not a rule of thumb. For the binding
constraint, this module correlates each design variable against that
constraint's margin over the run's own evaluations, and recommends moving a
bound only where BOTH hold: the correlation has a usable sign, and the best
point the run found is already sitting on that bound. A variable the search
never pushed against is not the reason it failed, and saying so would send a
user to widen the wrong row.

Everything here is pure — dicts in, dicts out, no NiceGUI — so the wording
and the arithmetic can be tested without a browser, which is the same split
:mod:`gui.metrics` keeps.
"""

from __future__ import annotations

import math

import numpy as np

#: |Pearson r| below this is not a direction. A design variable barely
#: correlated with the binding margin is noise, and recommending it would be
#: worse than saying nothing: the user widens a row, pays for another search
#: and gets the same refusal.
MIN_CORR = 0.15

#: how close to a bound counts as RIDING it, as a fraction of the row's own
#: width. The recommendation is "this box row is what stopped you", and that
#: is only true where the search was pressed up against it.
AT_BOUND_FRAC = 0.02

#: the objective every problem in this project reports for a design its
#: solver could not fly at all (``airfoil.PENALTY`` and its twins). Those
#: evaluations carry sentinel margins (``G_FAIL``), not measured ones, so
#: they are counted and then kept out of every correlation.
PENALTY = -100.0

#: …and the sentinel margin that comes with it.
G_FAIL = -1.0


#: flag keys that name WHAT A RUN MAXIMISED. Read off the record's own
#: ``config.flags``, because a run that returned nothing has no breakdown to
#: read it out of — the objective is the one thing about a failed run that is
#: still knowable, and it is knowable only from the configuration.
#:
#: Three spellings because there are three questions with the word "objective"
#: in them (the wing's, the car's, and the section stage's), and a bare
#: ``objective`` would be all three sharing a key — which is exactly why
#: ``api._CAR_KEY_ALIASES`` namespaces the car's on the way in.
OBJECTIVE_FLAG_KEYS = ("car_objective", "wing_objective", "objective")

#: objective -> {constraint label: what to say when it binds}.
#:
#: A LIMIT THE SCORE ALREADY CHARGES FOR. Most constraints are independent of
#: the objective: a deflection margin says nothing about downforce per unit
#: drag, and a run refused by one has learnt something the score could not
#: have told it. A few are not, and a lap is the case this table was written
#: for: ``lap_time`` prices drag PHYSICALLY — every newton is paid for on the
#: straights at the exchange rate the circuit's corner radii and straight
#: lengths supply — so a drag CEILING beside it is a second, independent
#: refusal on a quantity the score is already spending. It cannot make the
#: answer better; it can only delete designs the lap has already judged worth
#: their drag, and it can delete all of them.
#:
#: That is measured rather than argued. RESULTS_SESSION64_CARSECTION section 7
#: ran the 2-D lap optimum's shapes AND arrangement through 24 independent
#: searches of 4000 evaluations each under the family's published coefficient
#: allowance (CD_budget 0.11) and found NOT ONE feasible candidate — the lap
#: optimum sits at CD 0.181, and the allowance forbids it outright. A user who
#: chooses a lap and leaves a drag ceiling on can be handed that same empty
#: answer, and the honest reading of it is the ceiling's, not the wing's.
#:
#: The rule for adding an entry: the objective must ALREADY spend the quantity
#: the constraint bounds, in the objective's own units. It is not a list of
#: constraints that are merely correlated with the score — every constraint is
#: that — and it is not a licence to drop the limit, which stays exactly as
#: stated. Both car families spell the two drag margins the same way
#: (``CarWingProblem.constraint_labels``), so one entry covers all four
#: registered lap-capable families.
OBJECTIVE_ALREADY_PRICES = {
    "laptime": {
        "drag force margin":
            "Of the limits this run could not satisfy, the one furthest "
            "from being satisfiable is the drag ceiling you stated — and the "
            "LAP already prices drag: every newton of it is paid for on the "
            "straights, at the exchange rate this circuit's corners and "
            "straight lengths supply, and that is inside the score. The "
            "ceiling is a second, independent limit on the same drag, so what "
            "it removed is designs the lap had already judged worth their "
            "drag. Before widening a design-box row, ask whether this run "
            "needs the ceiling at all: with a circuit chosen, the answer to "
            "'how much drag is this downforce worth' is the circuit's, not a "
            "number you have to pick. Measured, on the strongest case there "
            "is: the fastest two-element section this repo has designed has "
            "NO feasible wing under the family's own retired coefficient "
            "allowance — 24 searches of 4000 evaluations found none "
            "(RESULTS_SESSION64_CARSECTION section 7).",
        "drag budget margin":
            "The limit furthest from being satisfiable here is the "
            "COEFFICIENT drag allowance, and it is the worst one to be "
            "stopped by on this family: it is referenced to "
            "the very reference area the search is moving, and the LAP "
            "already prices drag physically on the straights. Neither shell "
            "sends this allowance — the registry switches it off for every "
            "car family (api.CAR_DEFAULT_OBJECTIVE says why) — so a run "
            "carrying it was configured by hand or by an old saved record. "
            "Drop it and let the circuit price the drag.",
    },
    # A drag CEILING beside a drag OBJECTIVE is the same class as the one
    # above and a stronger case of it: the lap prices drag against downforce,
    # where these two price nothing else at all. A ceiling on the quantity
    # being minimised cannot select — it can only refuse.
    "drag": {
        "drag force margin":
            "The limit furthest from being satisfiable is a drag ceiling, on "
            "a run whose SCORE is the drag. The objective already ranks every "
            "design by exactly the quantity this limit bounds, so the ceiling "
            "chooses nothing — it only refuses, and what it refuses is the "
            "high-drag end the score was already sorting to the bottom. Drop "
            "it. What this run is missing is the other half of the question: "
            "a DOWNFORCE FLOOR (downforce_min_n). Without one the "
            "minimum-drag wing is the smallest wing — measured over 3000 "
            "draws of the endplate box it makes 43 N at CZ 0.058.",
        "drag budget margin":
            "The limit furthest from being satisfiable is the COEFFICIENT "
            "drag allowance, on a run whose SCORE is the drag. It bounds the "
            "quantity being minimised, so it can only refuse designs the "
            "objective had already ranked last. Drop it, and state a "
            "downforce FLOOR instead — that is the limit this formulation "
            "needs, and the one that makes it a question rather than a race "
            "to the smallest wing.",
    },
}
#: ``cd`` asks the same question in coefficients and gets the same sentences.
OBJECTIVE_ALREADY_PRICES["cd"] = OBJECTIVE_ALREADY_PRICES["drag"]


def _objective_of(result: dict | None) -> str | None:
    """What this run maximised, off its own stored configuration."""
    flags = ((result or {}).get("config") or {}).get("flags") or {}
    if not isinstance(flags, dict):
        return None
    for key in OBJECTIVE_FLAG_KEYS:
        v = flags.get(key)
        if v:
            return str(v)
    return None


def _matrix(rows) -> np.ndarray | None:
    """``rows`` as a 2-D float array, or None when there is nothing.

    A 1-D input is a COLUMN, not a mistake: a single-constraint run records its
    margins as scalars — ``ConstrainedRunHistory.g`` stays ``(n,)`` when m == 1
    on purpose, so downstream code that indexes ``h.g[i]`` as a number keeps
    working — and ``RunResult.eval_g`` mirrors that shape. Rejecting it made
    every diagnostic in this module unreachable on exactly the families that
    have one margin: measured, ``infeasibility_report`` returned None for a
    single-constraint record and a full report for the same record with its
    margins nested one level deeper. Those families are the majority of the
    registry, and they are the ones a user meets first.
    """
    if rows is None:
        return None
    arr = np.asarray(rows, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2 or arr.size == 0:
        return None
    return arr


def _unflyable(G: np.ndarray, Y: np.ndarray | None) -> np.ndarray:
    """Per-evaluation mask: the solver failed here, so its margins are the
    sentinel and mean nothing about the design box.

    Two independent tells, and BOTH are required, because either one alone
    has a false positive: an objective at the penalty could in principle be a
    real (terrible) score, and an all-``G_FAIL`` margin vector could in
    principle be a real design that misses every limit by exactly one limit's
    width. Together they are the failure contract every problem in this
    project writes (``fg_*``: return ``(PENALTY, [G_FAIL] * m)``).

    A GRADED refusal (``bo_feasibility="guide"``) writes the same value into
    every margin and it is <= ``G_FAIL``, not equal to it — so testing for
    -1 exactly would let every refused design of a guided run through as a
    design that flew, and the nearest-miss and the box moves would then be
    computed off margins no solver produced. The shape of the contract is
    what identifies it: one value, repeated, at or below the sentinel.
    """
    sentinel = (np.all(np.isclose(G, G[:, :1]), axis=1)
                & (G.max(axis=1) <= G_FAIL + 1e-12))
    if Y is None or Y.shape[0] != G.shape[0]:
        return sentinel
    penalised = np.isclose(Y, PENALTY) | ~np.isfinite(Y)
    return sentinel & penalised


def _corr(x: np.ndarray, g: np.ndarray) -> float | None:
    """Pearson r between one design variable and one margin, or None when
    either side never varied (a pinned row, a constant margin)."""
    if x.size < 3:
        return None
    sx, sg = float(np.std(x)), float(np.std(g))
    if sx <= 1e-12 or sg <= 1e-12:
        return None
    r = float(np.corrcoef(x, g)[0, 1])
    return r if math.isfinite(r) else None


def _where_in_box(v: float, lo: float, hi: float) -> float | None:
    """Where ``v`` sits across its row, 0 at the floor and 1 at the cap.
    None for a PINNED row (zero width), which has no inside."""
    width = float(hi) - float(lo)
    if not math.isfinite(width) or width <= 1e-12:
        return None
    return (float(v) - float(lo)) / width


def infeasibility_report(result: dict | None,
                         constraint_labels=()) -> dict | None:
    """Why this run returned no design — or None when it returned one.

    ``result`` is a :meth:`api.RunResult.to_dict` payload.
    ``constraint_labels`` names the margins (``ProblemSpec.constraint_labels``);
    missing names fall back to ``g[j]``.

    Returns None — nothing to explain — for a run that found a feasible
    incumbent, an unconstrained run, and a run whose record carries no
    margins at all (an old file, a front run). Otherwise:

    ``constraints``
        one row per margin: its ``label``, the ``best`` (least negative) value
        any evaluation reached, ``n_violated`` of ``n_measured``, and
        ``ever_met`` — whether ANY single design satisfied it on its own.
    ``binding``
        the constraint with the lowest best-case margin: the one furthest from
        being satisfiable at all.
    ``conflict``
        True when every constraint was met SOMEWHERE but never all at once.
        That is a different failure from an unreachable limit and it takes a
        different fix (a trade, not a wider box), so it is its own field
        rather than a shade of the sentence.
    ``moves``
        the box rows to change, most-correlated first — each with the
        ``variable``, the ``direction`` to move its bound ("raise the upper
        bound" / "lower the lower bound"), the correlation it was chosen on
        and the bound it is currently riding.
    ``n_unflyable``
        evaluations whose solver failed outright. A run that is mostly these
        is not a box problem at all, and the caller says so.
    """
    if not isinstance(result, dict):
        return None
    if result.get("best_x") is not None and result.get("feasible"):
        return None
    G = _matrix(result.get("eval_g"))
    if G is None:
        return None

    Y = None
    if result.get("eval_y") is not None:
        y = np.asarray(result.get("eval_y"), dtype=float)
        Y = y if y.ndim == 1 and y.shape[0] == G.shape[0] else None
    dead = _unflyable(G, Y)
    live = ~dead
    n_evals = int(result.get("n_evals") or G.shape[0])
    labels = list(constraint_labels or ())

    rows: list[dict] = []
    for j in range(G.shape[1]):
        col = G[live, j]
        col = col[np.isfinite(col)]
        rows.append({
            "index": j,
            "label": labels[j] if j < len(labels) else f"g[{j}]",
            "best": float(np.max(col)) if col.size else None,
            "worst": float(np.min(col)) if col.size else None,
            "n_measured": int(col.size),
            "n_violated": int(np.sum(col < 0.0)) if col.size else 0,
            "ever_met": bool(np.any(col >= 0.0)) if col.size else False,
        })
    measured = [r for r in rows if r["best"] is not None]
    binding = min(measured, key=lambda r: r["best"]) if measured else None
    # ...and whether the constraint that stopped it is one this run's own
    # OBJECTIVE was already paying for. Attached to the binding row only:
    # said about a constraint that is not what stopped the run it would be a
    # lecture, and said about one that never bound at all it would be wrong.
    priced = OBJECTIVE_ALREADY_PRICES.get(_objective_of(result) or "", {})
    already_priced = (None if binding is None
                      else priced.get(binding["label"]))
    out = {
        "n_evals": n_evals,
        "n_feasible": int(result.get("n_feasible") or 0),
        # the binding constraint's own sentence, when this run's objective
        # already charged for the quantity it bounds (OBJECTIVE_ALREADY_PRICES)
        "already_priced": already_priced,
        # how many draws the run's own screened initial design threw away as
        # refusals, when it had one (api.RunResult.n_screened). It is the
        # cheapest true statement anyone can make about a box that returned
        # nothing, and it costs no physics to say: the run already paid for it.
        "n_screened": (None if result.get("n_screened") is None
                       else int(result["n_screened"])),
        # ...and what the FEASIBILITY PHASE did with the iterations it took:
        # a phase that could not steer any of them is a uniform search under
        # another name, and the run reports that as its own number rather
        # than leaving it to be inferred (api.RunResult.n_rescue_blind)
        "n_rescue": (None if result.get("n_rescue") is None
                     else int(result["n_rescue"])),
        "n_rescue_blind": (None if result.get("n_rescue_blind") is None
                           else int(result["n_rescue_blind"])),
        # rows the run DREW FROM and the physics then refused, because the
        # override reached the sampler's box and not the problem's
        # (api.rows_outside_validity). A run made mostly of these looks exactly
        # like an impossible box and takes the opposite fix.
        "box_outside_validity": list(result.get("box_outside_validity") or ()),
        "n_unflyable": int(np.sum(dead)),
        "n_measured": int(np.sum(live)),
        "constraints": rows,
        "binding": binding,
        # a CONFLICT needs two constraints to conflict: with one margin,
        # "met somewhere" and "no feasible design" cannot both be true, and
        # calling a one-constraint run a trade-off would be a sentence the
        # record does not support
        "conflict": len(measured) > 1 and all(r["ever_met"]
                                              for r in measured),
        "moves": [],
        "box_is_binding": False,
    }
    # the NEAREST MISS: a run that returns nothing still evaluated designs, and
    # the least-violating one is the difference between "nothing came close"
    # and "one nudge away". Read off the same log, over the points whose
    # physics actually ran — a refusal's sentinel margin is not a distance
    # (aerobo.optimize.feasible.min_violation_index applies the same rule to a
    # live run's own history).
    out["nearest_miss"] = None
    if Y is not None and np.any(live):
        from aerobo.optimize.feasible import min_violation_index

        idx_live = np.flatnonzero(live)
        k = min_violation_index(Y[live], G[live])
        if k is not None:
            i = int(idx_live[k])
            j_bind = int(np.argmin(G[i]))
            X_all = _matrix(result.get("eval_x"))
            has_x = X_all is not None and X_all.shape[0] > i
            out["nearest_miss"] = {
                "index": i,
                "violation": float(G[i].min()),
                "label": (labels[j_bind] if j_bind < len(labels)
                          else f"g[{j_bind}]"),
                "x": ([float(v) for v in X_all[i]] if has_x else None),
            }
    if binding is not None:
        out["moves"] = box_moves(result, binding["index"], live)
        _annotate_trades(out["moves"], result, binding["index"], live, rows)
        out["box_is_binding"] = bool(out["moves"])
    return out


def _annotate_trades(moves, result: dict, j: int, live: np.ndarray,
                     rows: list[dict]) -> None:
    """Mark a recommended box move that would push ANOTHER margin the wrong
    way, in place.

    Widening a bound to clear the binding constraint is only good advice if
    the same move does not spend a constraint that is currently being held.
    That is measurable on the same points: if the margin of some other
    constraint falls as this variable moves the way the recommendation asks,
    the user is being sent to trade, not to fix — and the card has to say so
    or the next run comes back refused for a different reason.
    """
    G, X = _matrix(result.get("eval_g")), _matrix(result.get("eval_x"))
    if G is None or X is None or not len(moves):
        return
    Xl, Gl = X[live], G[live]
    for m in moves:
        k, at = m.get("index"), m.get("at")
        if k is None or at not in ("upper", "lower"):
            m["trades_against"] = []
            continue
        sign = 1.0 if at == "upper" else -1.0
        hit = []
        for other in rows:
            j2 = other["index"]
            if j2 == j or not other["n_measured"]:
                continue
            r2 = _corr(Xl[:, k], Gl[:, j2])
            if r2 is not None and abs(r2) >= MIN_CORR and r2 * sign < 0.0:
                hit.append({"label": other["label"], "corr": r2})
        m["trades_against"] = hit


def box_moves(result: dict, j: int, live: np.ndarray | None = None,
              n_max: int = 3) -> list[dict]:
    """Which design-box rows to move, and which way, to clear constraint ``j``.

    The measurement, stated once: over the evaluations whose physics actually
    ran, correlate each design variable with margin ``j``. A positive
    correlation means the margin rises as that variable rises, so the way out
    is UP — and the box is what is stopping it only if the best point the run
    found is already on the upper bound. Symmetrically for a negative
    correlation and the lower bound.

    Both halves are load-bearing. Correlation alone would recommend widening
    a row the search never reached the end of, which cannot be what stopped
    it. Bound-riding alone would recommend any row the optimum happens to sit
    on, including ones with nothing to do with this constraint.

    A PINNED row (``lo == hi``) has no inside to be at the edge of, so the
    correlation cannot be measured; it is reported anyway, with
    ``pinned: True``, because "you pinned the one variable that could have
    fixed this" is the most actionable sentence on the page.
    """
    G = _matrix(result.get("eval_g"))
    X = _matrix(result.get("eval_x"))
    B = _matrix(result.get("bounds"))
    if G is None or X is None or B is None or j >= G.shape[1]:
        return []
    if X.shape[0] != G.shape[0] or B.shape[0] != X.shape[1]:
        return []
    if live is None:
        # the SAME two tells the report uses, or a direct caller would
        # silently get a stricter filter than the card does
        y = result.get("eval_y")
        y = None if y is None else np.asarray(y, dtype=float)
        live = ~_unflyable(G, y if (y is not None and y.ndim == 1
                                    and y.shape[0] == G.shape[0]) else None)
    Xl, gl = X[live], G[live, j]
    ok = np.isfinite(gl)
    Xl, gl = Xl[ok], gl[ok]
    if gl.size == 0:
        return []
    #: the closest this run ever came on THIS constraint — the point whose
    #: box position says whether the box is the thing in the way
    star = Xl[int(np.argmax(gl))]
    labels = list(result.get("param_labels") or ())
    pinned = dict(result.get("pinned") or {})

    out: list[dict] = []
    for k in range(X.shape[1]):
        name = labels[k] if k < len(labels) else f"x[{k}]"
        lo, hi = float(B[k, 0]), float(B[k, 1])
        pos = _where_in_box(star[k], lo, hi)
        if pos is None:
            if name in pinned:
                out.append({"variable": name, "index": k, "pinned": True,
                            "value": float(star[k]), "lo": lo, "hi": hi,
                            "corr": None, "at": None, "trades_against": [],
                            "direction": "unpin it — it cannot move at all "
                                         "while it is held fixed"})
            continue
        r = _corr(Xl[:, k], gl)
        if r is None or abs(r) < MIN_CORR:
            continue
        if r > 0.0 and pos >= 1.0 - AT_BOUND_FRAC:
            out.append({"variable": name, "index": k, "pinned": False,
                        "corr": r, "trades_against": [],
                        "at": "upper", "bound": hi, "lo": lo, "hi": hi,
                        "value": float(star[k]),
                        "direction": f"RAISE its upper bound (now {hi:.4g})"})
        elif r < 0.0 and pos <= AT_BOUND_FRAC:
            out.append({"variable": name, "index": k, "pinned": False,
                        "corr": r, "trades_against": [],
                        "at": "lower", "bound": lo, "lo": lo, "hi": hi,
                        "value": float(star[k]),
                        "direction": f"LOWER its lower bound (now {lo:.4g})"})
    # the two kinds of entry get SEPARATE budgets, because they are not
    # competing for the same thing. A pinned entry carries no correlation and
    # no bound-riding test — it is emitted for every pinned row, on the
    # strength of the pin alone — so with one shared budget the pins sort
    # ahead of the measured rows and evict them: measured on a record whose
    # free row correlates r = +1.00 with the binding margin and sits exactly
    # on its upper bound, three pins (ordinary — the per-row FIX switch plus
    # the automatic `taper` pin) dropped that row entirely and no "RAISE its
    # upper bound" line was printed anywhere, while the same record with two
    # pins printed it. `gui.v3.relax.plan` then skips the pinned entries,
    # finds nothing left, and calls the run "no-evidence" — the one verdict
    # this record disproves.
    pins = [m for m in out if m.get("pinned")]
    free = sorted((m for m in out if not m.get("pinned")),
                  key=lambda m: -abs(m.get("corr") or 0.0))
    return pins[:n_max] + free[:n_max]


def infeasibility_lines(rep: dict | None) -> list[tuple[str, str]]:
    """``infeasibility_report`` as sentences, each with a severity.

    Severity is the shells' own vocabulary (``"bad"`` / ``"warn"`` / ``""``),
    so a caller renders the list straight into whatever it uses for a hint
    without deciding anything. Returns ``[]`` for None, so "there is nothing
    to explain" needs no branch at the call site.
    """
    if not rep:
        return []
    lines: list[tuple[str, str]] = []
    n, nf = rep["n_evals"], rep["n_feasible"]
    lines.append((f"No design satisfied every constraint: {nf} of {n} "
                  f"evaluations were feasible, so there is nothing to take "
                  f"forward.", "bad"))

    dead, measured = rep["n_unflyable"], rep["n_measured"]
    if dead and dead >= 0.5 * max(1, rep["n_evals"]):
        lines.append((f"{dead} of {n} evaluations could not be FLOWN at all "
                      f"(the objective returned its failure sentinel), so "
                      f"the margins below are read off only {measured} "
                      f"points. Two different things look like this and they "
                      f"take opposite fixes: a solver that could not run at "
                      f"this operating point, and a design the BOX allows "
                      f"that a declared gate refuses outright — an aspect "
                      f"ratio outside the solvers' band, a wing loading above "
                      f"the mission's own ceiling. Measure the design box to "
                      f"see which, and how much of it is refused before it is "
                      f"solved (api.box_refusal_probe).", "warn"))
    elif dead:
        lines.append((f"{dead} of {n} evaluations could not be flown at all; "
                      f"the {measured} that could are what the margins below "
                      f"are measured over.", ""))

    for row in (rep.get("box_outside_validity") or ()):
        lo, hi = row["searched"]
        vlo, vhi = row["validated"]
        lines.append((
            f"{row['label']} is being SEARCHED over {lo:g}-{hi:g} while this "
            f"family only validates {vlo:g}-{vhi:g}: every draw in the excess "
            f"comes back refused, so the row you widened is not the row the "
            f"run used. Narrow it back to the validated band — widening this "
            f"row does not reach the physics yet.", "warn"))

    near = rep.get("nearest_miss")
    if near is not None:
        lines.append((
            f"The design that came CLOSEST missed by {abs(near['violation']):.4g} "
            f"on {near['label']} — evaluation {near['index'] + 1} of {n}. It is "
            f"not an answer (it does not satisfy the constraint), but it is a "
            f"real design and it says how far off this box is: a near miss "
            f"takes a nudge, and nothing coming close takes a different box.",
            ""))

    screened = rep.get("n_screened")
    if screened:
        lines.append((
            f"Before this search began it drew and threw away {screened} "
            f"designs the physics refused outright, to seed itself with "
            f"{'ones' if screened > 1 else 'one'} that fly — so most of this "
            f"design box is not a design at all. Widening a row will not help "
            f"until that is fixed; measure the box to see which gate is "
            f"eating it.", "warn"))

    n_rescue = rep.get("n_rescue") or 0
    if n_rescue:
        blind = rep.get("n_rescue_blind")
        lines.append((
            f"Having found nothing feasible, the search spent {n_rescue} "
            f"evaluation{'s' if n_rescue != 1 else ''} hunting for a design "
            f"that merely FLIES rather than a good one"
            + ("." if blind is None else
               f", and {blind} of those had no gradient to follow — every "
               f"design it had seen was refused at the same flat margin, so "
               f"it could only draw at random."
               if blind else ", steering by the margins every time."), ""))

    for c in rep["constraints"]:
        if c["best"] is None:
            continue
        lines.append((
            f"{c['label']}: the closest any design came was {c['best']:+.3f} "
            f"({'met somewhere, ' if c['ever_met'] else 'never met, '}"
            f"{c['n_violated']} of {c['n_measured']} points violated it).",
            "" if c["ever_met"] else "warn"))

    if rep["conflict"]:
        lines.append((
            "Every constraint was satisfied by SOME design and none by the "
            "same one: this box has no point that holds them all at once. "
            "Widening it may not help — the fix is to relax whichever limit "
            "you can afford to, or to change the operating point the two are "
            "being asked to meet together.", "warn"))

    b = rep["binding"]
    if b is not None and not rep["conflict"]:
        lines.append((f"{b['label']} is the binding one — it is the furthest "
                      f"from ever being satisfied.", "warn"))
    # ...and, where it applies, WHOSE limit that is. A user handed "no design
    # satisfied every constraint" over a limit their own objective was already
    # charging for will widen a design-box row, pay for another search and get
    # the same refusal; the fix is one control away and it is not in the box.
    # Emitted next to the binding line, and only about the binding constraint
    # (infeasibility_report attaches it there).
    if rep.get("already_priced"):
        lines.append((rep["already_priced"], "warn"))

    if rep["moves"]:
        lines.append(("What to change, measured on this run's own points "
                      "(each variable's correlation with that margin, and "
                      "whether the best point is already on its bound):",
                      ""))
        for m in rep["moves"]:
            if m.get("pinned"):
                lines.append((f"    · {m['variable']} — {m['direction']}", ""))
                continue
            trade = m.get("trades_against") or []
            lines.append((
                f"    · {m['variable']} — {m['direction']}; the best point "
                f"sits on it and the margin moves with it "
                f"(r = {m['corr']:+.2f})."
                + ("" if not trade else
                   " NOTE: the same move pushes "
                   + ", ".join(f"{t['label']} (r = {t['corr']:+.2f})"
                               for t in trade)
                   + " the wrong way — this is a trade, not a free fix."),
                "" if not trade else "warn"))
    elif b is not None:
        lines.append((
            "No design-box row is indicated: the best point for that "
            "constraint is not riding any bound, so a wider box is not what "
            "is missing. Relax the limit itself, or move the operating "
            "point.", "warn"))
    return lines


# ============================================================ reaching out
#
# Everything above answers "why did this box return nothing". What follows
# answers the question a user asks straight afterwards — "then what box WOULD
# have held a design, and how far is it from mine" — and it is deliberately
# the same kind of arithmetic: read off the run's own log, no physics re-run.
#
# The split is kept: this module still imports nothing but numpy, so every
# sentence the shell prints about a relaxation is assertable without a
# browser and without a solver (:mod:`gui.v3.relax` is where the api is
# allowed in).

#: below this the fitted slope of a margin against a design variable is not a
#: distance. A correlation with a usable SIGN (``MIN_CORR``) says which way to
#: move; extrapolating HOW FAR along a line that explains a quarter of the
#: variance would put a number on the card that the run's own points do not
#: support, so the move falls back to a fixed step and says which it used.
MIN_R2 = 0.25

#: the fallback move, as a fraction of the row's own width — used when the
#: direction is known and the distance is not.
RELAX_STEP = 0.10

#: a straight line through a curved margin lands short: the fitted fix is the
#: distance at which the LINEAR margin reaches zero, and a margin that flattens
#: as it approaches its limit needs more. A quarter, and it is a headroom, not
#: a calibration — the check that matters is the one after the run.
HEADROOM = 0.25

#: no row moves by more than one of its own widths on a single reach. Past
#: that the answer stops being "closest to your box" and starts being a
#: different question, which is the user's to ask.
MAX_REACH = 1.0


def refused_fraction(record: dict | None) -> float | None:
    """Share of a record's evaluations the solver never ran, or None.

    None — not 0.0 — for a record carrying no margins at all: an old file and
    a run in which nothing was refused are different states, and a card that
    reported the second for the first would tell a user their box is fine on
    the strength of a missing key. Uses :func:`_unflyable`, so a GRADED
    refusal (``bo_feasibility="guide"``, margins repeated below the sentinel)
    counts as the refusal it is.
    """
    if not isinstance(record, dict):
        return None
    G = _matrix(record.get("eval_g"))
    if G is None or G.shape[0] == 0:
        return None
    Y = None
    if record.get("eval_y") is not None:
        y = np.asarray(record["eval_y"], dtype=float)
        Y = y if y.ndim == 1 and y.shape[0] == G.shape[0] else None
    return float(np.mean(_unflyable(G, Y)))


def move_magnitude(result: dict, j: int, move: dict,
                   live: np.ndarray | None = None) -> dict | None:
    """How FAR the row named in ``move`` has to travel to clear constraint j.

    :func:`box_moves` answers which row and which way; it stops there on
    purpose, because a Pearson r is a direction and not a distance. This is
    the distance, and it is the only new arithmetic the reach needs: the
    ordinary-least-squares slope of the binding margin against that one
    variable, over the same live, finite points the correlation was measured
    on, extrapolated to where the margin would reach zero.

    ``{slope, r2, deficit, delta, basis, capped, width}`` — or None when the
    row has no width, the fit has no variance to work with, or the move names
    a variable this record does not carry. ``basis`` is ``"fit"`` when the
    line explains at least :data:`MIN_R2` of the margin and ``"step"`` when it
    does not: BOTH ship, because a direction the run measured is still worth
    acting on, and the card says which one it is looking at rather than
    printing a number of unstated provenance.
    """
    G, X, B = (_matrix(result.get("eval_g")), _matrix(result.get("eval_x")),
               _matrix(result.get("bounds")))
    if G is None or X is None or B is None or j >= G.shape[1]:
        return None
    k = move.get("index")
    if k is None or not (0 <= int(k) < X.shape[1]) or int(k) >= B.shape[0]:
        return None
    k = int(k)
    if live is None:
        y = result.get("eval_y")
        y = None if y is None else np.asarray(y, dtype=float)
        live = ~_unflyable(G, y if (y is not None and y.ndim == 1
                                    and y.shape[0] == G.shape[0]) else None)
    xs, gs = X[live, k], G[live, j]
    ok = np.isfinite(xs) & np.isfinite(gs)
    xs, gs = xs[ok], gs[ok]
    width = float(B[k, 1]) - float(B[k, 0])
    if width <= 1e-12 or xs.size < 3:
        return None
    #: the DEFICIT is the shortfall of the closest point this run reached, not
    #: of the mean: the reach has to hold one design, and the run already
    #: found the one that came nearest to holding it.
    best = float(np.max(gs))
    deficit = max(0.0, -best)
    sx = float(np.std(xs))
    if sx <= 1e-12 or float(np.std(gs)) <= 1e-12:
        return None
    slope = float(np.polyfit(xs, gs, 1)[0])
    r = _corr(xs, gs)
    r2 = 0.0 if r is None else r * r
    # ...and a DEFICIT of zero is not a fit. The binding constraint is the one
    # with the lowest best-case margin, and on a CONFLICT run — every limit met
    # by SOME design and none by the same one — that lowest best is positive.
    # The fit branch then computes a distance of zero, the floor below turns it
    # into a 10 % step, and the card would say a straight-line fit put the fix
    # there while quoting a miss of 0.
    if deficit > 0.0 and r2 >= MIN_R2 and abs(slope) > 1e-12:
        delta, basis = (deficit / abs(slope)) * (1.0 + HEADROOM), "fit"
    else:
        delta, basis = RELAX_STEP * width, "step"
    capped = delta > width
    delta = min(max(delta, RELAX_STEP * width), width)
    return {"slope": slope, "r2": r2, "deficit": deficit, "delta": delta,
            "basis": basis, "capped": bool(capped), "width": width}


def outside_box(bounds, labels, x) -> dict:
    """Where design ``x`` sits relative to a stated box.

    ``{"rows": [...], "dinf": float, "d1": float}``, one row per variable the
    design is actually OUTSIDE — a row it sits inside is absent, so "which
    rows had to move" is ``len(rows)`` and needs no filtering at the caller.

    The distance is the overshoot in units of THAT ROW'S OWN WIDTH, and the
    headline ``dinf`` is the largest of them. Two reasons it is relative and
    not absolute: a box is a statement in several units at once (metres,
    degrees, dimensionless coefficients) and an absolute maximum over those is
    not a number; and a user who narrowed a row to a twentieth of its
    published band means something much tighter by "just outside" than one who
    left it alone. ``d1`` is the sum — how many rows, and how far in total.

    Every sentence that prints one of these MUST say "of that row's own
    width". It is not a physical margin and a card that lets it read as one
    is lying about a number the user cannot check.
    """
    B, xs = _matrix(bounds), np.asarray(x, dtype=float).ravel()
    out: dict = {"rows": [], "dinf": 0.0, "d1": 0.0}
    if B is None or B.shape[1] < 2 or xs.size == 0:
        return out
    names = list(labels or ())
    n = min(B.shape[0], xs.size)
    for k in range(n):
        lo, hi = float(B[k, 0]), float(B[k, 1])
        width = hi - lo
        if not math.isfinite(width) or width <= 1e-12:
            continue
        over = max(lo - xs[k], xs[k] - hi, 0.0)
        if over <= 0.0:
            continue
        out["rows"].append({
            "label": names[k] if k < len(names) else f"x[{k}]",
            "index": k, "value": float(xs[k]), "lo": lo, "hi": hi,
            "over": float(over), "d": float(over / width),
            "end": "top" if xs[k] > hi else "bottom"})
    out["rows"].sort(key=lambda r: -r["d"])
    out["dinf"] = max((r["d"] for r in out["rows"]), default=0.0)
    out["d1"] = float(sum(r["d"] for r in out["rows"]))
    return out


def _feasible_mask(G: np.ndarray, Y: np.ndarray | None) -> np.ndarray:
    """Per-evaluation mask: this design FLEW and met every margin.

    The same rule the incumbent is a max over
    (``optimize.constrained``: every margin >= 0, and >= not > because a
    margin that reaches exactly zero is a design that meets its limit), with
    the refusal mask applied first — a refused design's sentinel margins are
    not measurements and a run of them must never produce an answer.
    """
    live = ~_unflyable(G, Y)
    met = np.all(np.isfinite(G) & (G >= 0.0), axis=1)
    return live & met


def closest_feasible(record: dict, bounds, labels) -> dict | None:
    """The design in ``record``'s own log that met every limit and sits
    CLOSEST to ``bounds`` — a box the record need not have been searched over.

    This is what makes a reach answer the question that was asked. The relaxed
    run maximises the user's own objective, so the wing it finds is a good
    wing; the wing REPORTED is the nearest feasible one it saw to the box the
    user drew. Those are usually different designs, and quoting ``best_x``
    instead would answer "the best design out there" to a user who asked "the
    closest one".

    Ranked by :func:`outside_box`'s ``dinf``, ties broken by the objective, so
    a design that sits inside the box entirely (``dinf == 0``) always wins —
    and when one does, the box was never what stopped the search.

    ``{index, x, y, dinf, d1, rows}`` or None when nothing in the log flew and
    met every limit.
    """
    if not isinstance(record, dict):
        return None
    G, X = _matrix(record.get("eval_g")), _matrix(record.get("eval_x"))
    if G is None or X is None or X.shape[0] != G.shape[0]:
        return None
    Y = None
    if record.get("eval_y") is not None:
        y = np.asarray(record["eval_y"], dtype=float)
        Y = y if y.ndim == 1 and y.shape[0] == G.shape[0] else None
    ok = _feasible_mask(G, Y)
    if not np.any(ok):
        return None
    best = None
    for i in np.flatnonzero(ok):
        i = int(i)
        d = outside_box(bounds, labels, X[i])
        y_i = float(Y[i]) if Y is not None else float("-inf")
        key = (d["dinf"], -y_i)
        if best is None or key < best[0]:
            best = (key, {"index": i, "x": [float(v) for v in X[i]],
                          "y": (None if Y is None else y_i),
                          "dinf": d["dinf"], "d1": d["d1"],
                          "rows": d["rows"]})
    return None if best is None else best[1]
