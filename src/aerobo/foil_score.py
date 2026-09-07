"""The FOILING craft's own weighted composite — four flight points, one J.

``wing_score`` scores ten criteria off ONE evaluation of one design: the
loading, the spar, the stall margin, the mass. That is the right shape for a
wing, and the wrong shape for a board a person rides, because a rider's
priorities are not properties of a design point — they are *other design
points*. "Light wind" is the same craft at 7 m/s. "Top speed" is the same
craft at 16. "Manoeuvrable" is the same craft pulling more than its own
weight through a turn. Nothing about the cruise evaluation answers any of
them, and no weight on a cruise-point criterion can.

So this module scores a design at up to four FLIGHT POINTS and weights those:

  ld_cruise    L/D at the speed the placement states — today's objective, and
               the default (weight it alone and the composite reproduces the
               ordinary run's ranking exactly)
  ld_light     L/D at the light-wind speed. ``None`` — not zero — when the
               craft will not trim there, which is a real answer: the shipped
               windfoil preset is UNTRIMMABLE at 5 m/s
  cav_top      cavitation margin at the top speed, because cavitation is what
               actually ends a foil's top end. The same preset is already
               cavitating at 16 m/s (g_cav = -0.081)
  margin_turn  the worst constraint margin left at ``turn_n`` times the
               craft's weight, at the cruise speed — how much room is left
               before something (cavitation, the stability floor, the
               draught) runs out. A LOAD FACTOR and not a banked turn: the
               rig's stated forces do not rotate with it, because nothing
               downstream is lateral, and the measured direction is the
               opposite of the obvious one (see the criterion's own help)

What is deliberately NOT here
-----------------------------
**Pumping.** It is unsteady: the rider pushes the craft through a heaving,
pitching cycle and lives off the added-mass and wake terms. Every solver in
this package is steady, longitudinal and trimmed, so there is no quantity in
any breakdown that pumping could be read from. A "pumping" weight would have
had to be a proxy wearing a rider's word, and the tool's whole claim is that
what it draws is what it flies. It is a line in FoilingBO's *what this tool
does NOT model* block instead.

How the points are reached
--------------------------
The speed points substitute the ``V_ms`` column of the design vector — the
speed is a ROW in these families, so a second point costs one evaluation
(about 2 ms) and no rebuild. The turn point cannot: the weight is a FLAG
(``weight_n``), so it needs a second built problem, which the registry
wrapper builds once per run (1.8 ms) and only when a turn weight is on.

Everything else — the frozen p5/p95 band measured over the user's own box,
the refusal to score a weighted criterion that was never measured, the
sub-scores, the population rank — follows ``wing_score`` exactly, and this
module reuses its band machinery rather than restating it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .wing_score import (G_FAIL, MIN_REFERENCE_SAMPLES, PENALTY, REF_P_HI,
                         REF_P_LO, REFERENCE_SAMPLES, REFERENCE_VERSION,
                         Criterion, _sample_box, reference_sha)

# ------------------------------------------------------------- the criteria

#: Four criteria, one per flight point. Each says which point it is measured
#: at, because a shell that shows a sub-score has to be able to say WHERE the
#: number came from — "L/D 8.4" means nothing without "at 7 m/s".
FOIL_CRITERIA: tuple[Criterion, ...] = (
    Criterion(
        "ld_cruise", "L/D at your speed", "", "max",
        "Appendage L/D at the speed the placement states. The family's own "
        "objective: weight this alone and the search is the ordinary run."),
    Criterion(
        "ld_light", "L/D in light wind", "", "max",
        "Appendage L/D at the light-wind speed. A craft that will not trim "
        "there has no value for this criterion at all — the design is "
        "refused rather than scored zero, because 'cannot fly' and 'flies "
        "badly' are different answers and only one of them is a design."),
    Criterion(
        "cav_top", "cavitation margin at top speed", "", "max",
        "The worst surface's cavitation margin at the top speed. Cavitation "
        "is what ends a foil's top end — the section stops making lift and "
        "starts making noise — so this is the criterion a strong-wind rider "
        "is actually buying."),
    Criterion(
        "margin_turn", "margin under load", "", "max",
        "The worst constraint margin left when the craft is asked for "
        "`turn_n` times its own weight at the cruise speed: cavitation, the "
        "stability floor and the draught, whichever runs out first. It is a "
        "LOAD FACTOR, not a banked turn — the rig's stated forces do not "
        "rotate with it, because no solver here is lateral. Do not assume it "
        "falls: with a fixed rig couple the stabiliser's download share drops "
        "as the load rises (-0.476 to -0.357 from 1.0 to 1.4 g on the shipped "
        "windfoil), which UNLOADS the front foil and improves its margin."),
)

CRITERIA: tuple[str, ...] = tuple(c.key for c in FOIL_CRITERIA)
CRITERION: dict[str, Criterion] = {c.key: c for c in FOIL_CRITERIA}
LOWER_BETTER = tuple(c.key for c in FOIL_CRITERIA if c.sense == "min")

#: Which flight point each criterion is measured at. Read by the metric
#: extractor to decide what to EVALUATE: a design is only flown at the points
#: some weight actually asks for, so weighting cruise alone costs exactly what
#: the plain run costs.
POINT_OF: dict[str, str] = {"ld_cruise": "cruise", "ld_light": "light",
                            "cav_top": "top", "margin_turn": "turn"}


# ---------------------------------------------------------------- the point

#: Defaults for the three points a placement does not already state. They are
#: DEFAULTS, not limits: FoilingBO asks for the two speeds per craft, because
#: a windfoil's light wind is not a kitefoil's, and a shared fraction of
#: cruise would be wrong for two of the three disciplines.
DEFAULT_LIGHT_MS = 7.0
DEFAULT_TOP_MS = 16.0
DEFAULT_TURN_N = 1.4


@dataclass(frozen=True)
class FlightPoints:
    """Where the four criteria are measured.

    ``light_ms``/``top_ms`` are absolute speeds in m/s rather than multiples
    of the cruise speed, and ``turn_n`` is a load factor (1.0 = straight
    line). A point outside the problem's own speed band is not silently
    clamped — :func:`check` refuses it by name, because a light-wind score
    measured at a speed the run never flies is worse than no score.
    """

    light_ms: float = DEFAULT_LIGHT_MS
    top_ms: float = DEFAULT_TOP_MS
    turn_n: float = DEFAULT_TURN_N

    def check(self, band: tuple[float, float] | None = None,
              wanted: tuple[str, ...] = CRITERIA) -> None:
        if self.turn_n < 1.0 and "margin_turn" in wanted:
            raise ValueError(
                f"turn load factor {self.turn_n:g} is below 1.0 — a turn "
                f"loads the foil MORE than straight-line flight")
        if band is None:
            return
        lo, hi = float(band[0]), float(band[1])
        for key, name in (("light_ms", "ld_light"), ("top_ms", "cav_top")):
            if name not in wanted:
                continue
            v = float(getattr(self, key))
            if not (lo <= v <= hi):
                raise ValueError(
                    f"{key} = {v:g} m/s is outside this problem's speed band "
                    f"({lo:g}-{hi:g} m/s), so the {name!r} criterion would be "
                    f"measured where the run never flies. Widen V_ms, or drop "
                    f"the weight on it.")


# ---------------------------------------------------------------- weights


@dataclass(frozen=True)
class FoilScoreWeights:
    """Criterion weights, normalised to sum 1 in :meth:`normalised`.

    The default is the ordinary run written as weights: cruise L/D alone.
    """

    ld_cruise: float = 1.0
    ld_light: float = 0.0
    cav_top: float = 0.0
    margin_turn: float = 0.0

    def normalised(self) -> dict[str, float]:
        d = asdict(self)
        if any(v < 0.0 for v in d.values()):
            raise ValueError(f"negative weight in {d}")
        tot = sum(d.values())
        if tot <= 0.0:
            raise ValueError("all weights zero")
        return {k: v / tot for k, v in d.items()}

    def active(self) -> tuple[str, ...]:
        d = self.normalised()
        return tuple(k for k in CRITERIA if d[k] > 0.0)

    def points(self) -> tuple[str, ...]:
        """The flight points these weights actually need flown."""
        return tuple(dict.fromkeys(POINT_OF[k] for k in self.active()))


#: Every criterion, equally weighted. Not a preset anybody should SEARCH on —
#: it is the request "measure all four", which the band sweep and every card
#: that shows what a weighting cost need.
ALL_CRITERIA = FoilScoreWeights(ld_cruise=1.0, ld_light=1.0, cav_top=1.0,
                                margin_turn=1.0)


#: Rider-language starting points. A job description, not a rule — every one
#: is editable in the shell, and the shell says which one it started from.
PRESETS: dict[str, FoilScoreWeights] = {
    # the ordinary run, written as weights
    "efficiency": FoilScoreWeights(ld_cruise=1.0),
    # gets going first and keeps going when it drops
    "light wind": FoilScoreWeights(ld_cruise=0.30, ld_light=0.50,
                                   cav_top=0.05, margin_turn=0.15),
    # the top end: cavitation is the wall
    "speed": FoilScoreWeights(ld_cruise=0.35, ld_light=0.05, cav_top=0.45,
                              margin_turn=0.15),
    # turns and transitions before the last knot
    "manoeuvre": FoilScoreWeights(ld_cruise=0.25, ld_light=0.25,
                                  cav_top=0.10, margin_turn=0.40),
    # one board, all day
    "freeride": FoilScoreWeights(ld_cruise=0.40, ld_light=0.25, cav_top=0.15,
                                 margin_turn=0.20),
}


def weights_of(weights) -> FoilScoreWeights:
    """Coerce a preset name / partial dict / weights object to weights.

    A partial dict MERGES over ``efficiency``, the same rule
    ``wing_score.weights_of`` and ``api.screen_weights`` follow, so "put 0.3
    on cav_top" means what it looks like.
    """
    if isinstance(weights, FoilScoreWeights):
        return weights
    if weights is None:
        return FoilScoreWeights()
    if isinstance(weights, str):
        try:
            return PRESETS[weights]
        except KeyError:
            raise ValueError(
                f"unknown foiling weight preset {weights!r} — "
                f"one of {sorted(PRESETS)}") from None
    if isinstance(weights, dict):
        unknown = set(weights) - set(CRITERIA)
        if unknown:
            raise ValueError(
                f"unknown foiling criteria {sorted(unknown)} — "
                f"one of {list(CRITERIA)}")
        base = asdict(FoilScoreWeights())
        base.update({k: float(v) for k, v in weights.items()})
        return FoilScoreWeights(**base)
    raise ValueError(f"cannot read weights from {type(weights).__name__}")


# ------------------------------------------------------- flying the points


def _lod(raw) -> float | None:
    if not isinstance(raw, dict) or not raw.get("feasible"):
        return None
    v = raw.get("LoD")
    return None if v is None else float(v)


def _cav(raw) -> float | None:
    if not isinstance(raw, dict) or not raw.get("feasible"):
        return None
    v = raw.get("g_cav")
    return None if v is None else float(v)


def _worst_margin(raw) -> float | None:
    """The smallest constraint margin the family reports, or None.

    ``g`` is the family's own margin vector and the composite must not invent
    one: a design whose margins are unreported is unscoreable for this
    criterion, not comfortable.
    """
    if not isinstance(raw, dict) or not raw.get("feasible"):
        return None
    g = raw.get("g")
    if g is None:
        return None
    arr = np.atleast_1d(np.asarray(g, dtype=float))
    arr = arr[np.isfinite(arr)]
    return float(arr.min()) if arr.size else None


def flight_metrics(x, *, evaluate, i_v: int, points: FlightPoints,
                   evaluate_turn=None, wanted: tuple[str, ...] = CRITERIA
                   ) -> dict:
    """``{criterion: value|None}`` for one design, flying only what is wanted.

    ``evaluate`` is the family's own full-width evaluate and ``i_v`` the index
    of its ``V_ms`` row — the speed points are that column substituted, which
    is why they cost an evaluation each and not a rebuild. ``evaluate_turn``
    is the same design's evaluate on a problem built at the turn load factor;
    without it ``margin_turn`` is None rather than guessed.

    Every value is None where its point refused. That is the contract the
    composite needs: :func:`composite` refuses to score a weighted criterion
    with no value, so "will not fly light" reaches the optimiser as a
    refusal rather than as a bad number.
    """
    x = np.asarray(x, dtype=float)
    out = {k: None for k in CRITERIA}
    need = set(wanted)

    def at(speed):
        xs = x.copy()
        xs[i_v] = float(speed)
        try:
            return evaluate(xs)
        except Exception:                      # a point the family hates
            return {"feasible": False}

    if "ld_cruise" in need:
        out["ld_cruise"] = _lod(at(x[i_v]))
    if "ld_light" in need:
        out["ld_light"] = _lod(at(points.light_ms))
    if "cav_top" in need:
        out["cav_top"] = _cav(at(points.top_ms))
    if "margin_turn" in need and evaluate_turn is not None:
        try:
            out["margin_turn"] = _worst_margin(evaluate_turn(x))
        except Exception:
            out["margin_turn"] = None
    return out


# ------------------------------------------------------------- the band


@dataclass(frozen=True)
class FoilScoreReference:
    """Frozen per-criterion (lo, hi) band, measured over the user's own box.

    Carries a SUBSET of :data:`CRITERIA`: a box where two thirds of the
    designs will not trim at the light-wind speed has no band for
    ``ld_light``, and :func:`composite` refuses a weight on it rather than
    scoring it zero.
    """

    bounds: dict[str, tuple[float, float]]
    sha: str = ""
    n_samples: int = 0
    n_feasible: int = 0
    problem: str = ""
    seed: int = 0
    points: tuple = ()
    #: per-criterion count of samples that produced a value — the honest
    #: answer to "why is there no light-wind band?"
    n_measured: dict = None
    samples: tuple = ()

    def __post_init__(self) -> None:
        extra = set(self.bounds) - set(CRITERIA)
        if extra:
            raise ValueError(
                f"reference bounds carry unknown criteria {sorted(extra)} "
                f"(known: {list(CRITERIA)})")
        if not self.bounds:
            raise ValueError("reference bounds are empty: nothing measurable")
        for key, (lo, hi) in self.bounds.items():
            if not (np.isfinite(lo) and np.isfinite(hi)):
                raise ValueError(f"non-finite reference band for {key}")
            if hi <= lo:
                raise ValueError(
                    f"degenerate reference band for {key}: hi {hi} <= lo {lo}")

    def band(self, key: str) -> tuple[float, float]:
        lo, hi = self.bounds[key]
        return float(lo), float(hi)

    def payload(self) -> dict:
        return {"version": REFERENCE_VERSION,
                "bounds": {k: [float(v[0]), float(v[1])]
                           for k, v in self.bounds.items()},
                "sha": self.sha, "n_samples": int(self.n_samples),
                "n_feasible": int(self.n_feasible),
                "problem": self.problem, "seed": int(self.seed),
                "points": list(self.points),
                "n_measured": dict(self.n_measured or {}),
                "samples": [{k: (None if v is None else float(v))
                             for k, v in row.items()}
                            for row in self.samples]}


def reference_from_payload(payload: dict) -> FoilScoreReference:
    """Read a stored band, RAISING on any inconsistency.

    Deliberately intolerant, for ``wing_score.reference_from_payload``'s
    reason: a band that loads stale changes every composite number the
    optimiser saw, with no visible symptom.
    """
    if not isinstance(payload, dict):
        raise ValueError("foil score reference: not an object")
    for field in ("version", "bounds", "sha"):
        if field not in payload:
            raise ValueError(f"foil score reference: missing '{field}'")
    if int(payload["version"]) != REFERENCE_VERSION:
        raise ValueError(
            f"foil score reference: format version {payload['version']} "
            f"!= {REFERENCE_VERSION}")
    raw = payload["bounds"]
    if not isinstance(raw, dict) or not raw:
        raise ValueError("foil score reference: 'bounds' is not a non-empty "
                         "object")
    bounds = {}
    for key, pair in raw.items():
        if (not isinstance(pair, (list, tuple)) or len(pair) != 2
                or not all(isinstance(v, (int, float)) for v in pair)):
            raise ValueError(
                f"foil score reference: bounds['{key}'] must be a 2-element "
                f"numeric [lo, hi], got {pair!r}")
        bounds[str(key)] = (float(pair[0]), float(pair[1]))
    rows = payload.get("samples") or ()
    samples = tuple({str(k): (None if v is None else float(v))
                     for k, v in row.items() if str(k) in CRITERIA}
                    for row in rows if isinstance(row, dict))
    return FoilScoreReference(
        bounds=bounds, sha=str(payload["sha"]),
        n_samples=int(payload.get("n_samples", 0)),
        n_feasible=int(payload.get("n_feasible", 0)),
        problem=str(payload.get("problem", "")),
        seed=int(payload.get("seed", 0)),
        points=tuple(payload.get("points") or ()),
        n_measured=dict(payload.get("n_measured") or {}),
        samples=samples)


def sample_reference(metrics_of_x, bounds: np.ndarray, *, problem: str = "",
                     n: int = REFERENCE_SAMPLES, seed: int = 0,
                     labels: tuple = (), p_lo: float = REF_P_LO,
                     p_hi: float = REF_P_HI, progress=None,
                     points: FlightPoints | None = None) -> FoilScoreReference:
    """Measure the frozen band by flying ``n`` box samples at every point.

    ``metrics_of_x(x)`` is :func:`flight_metrics` bound to the run's own
    evaluators and points — the SAME closure the search will score with, so
    the band cannot cover a different set of criteria than the objective
    reads.

    A sample counts as flown when ANY criterion resolved. That is looser than
    ``wing_score``'s feasibility gate on purpose: a design that cruises and
    cavitates at the top speed is a perfectly good member of the population
    for three of the four criteria, and dropping it would measure the light
    band over the subset that also happens to fly fast.
    """
    pts = _sample_box(bounds, n, seed)
    rows: list[dict] = []
    for i, x in enumerate(pts):
        row = metrics_of_x(np.asarray(x, dtype=float))
        if any(v is not None for v in row.values()):
            rows.append(row)
        if progress is not None:
            progress(i + 1, len(pts))

    n_measured = {k: sum(1 for r in rows if r.get(k) is not None)
                  for k in CRITERIA}
    if len(rows) < MIN_REFERENCE_SAMPLES:
        raise ValueError(
            f"only {len(rows)} of {len(pts)} box samples flew at any point — "
            f"too few to measure a normalisation band (need "
            f"{MIN_REFERENCE_SAMPLES}). Narrow the design box to a region "
            f"that flies, then measure it again.")

    bounds_out: dict[str, tuple[float, float]] = {}
    for key in CRITERIA:
        vals = np.array([r[key] for r in rows if r.get(key) is not None],
                        dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size < MIN_REFERENCE_SAMPLES:
            continue                       # this box cannot measure it
        lo = float(np.percentile(vals, p_lo))
        hi = float(np.percentile(vals, p_hi))
        if hi <= lo:
            continue                       # constant across the box
        bounds_out[key] = (lo, hi)
    if not bounds_out:
        raise ValueError(
            "no criterion varied across the box sample, so there is nothing "
            "to normalise against — widen the design box.")

    box = {str(lbl): (float(bounds[i][0]), float(bounds[i][1]))
           for i, lbl in enumerate(labels)} if labels else \
          {str(i): (float(b[0]), float(b[1]))
           for i, b in enumerate(np.asarray(bounds, dtype=float))}
    p = points or FlightPoints()
    # the POINTS are in the fingerprint: the same box measured at a different
    # light-wind speed is a different population, and a band silently reused
    # across two of them would score two runs against each other's physics.
    box["__points__"] = (float(p.light_ms), float(p.top_ms))
    box["__turn__"] = (float(p.turn_n), float(p.turn_n))
    sha = reference_sha(problem, box, len(pts), seed, bounds_out)
    return FoilScoreReference(
        bounds=bounds_out, sha=sha, n_samples=len(pts), n_feasible=len(rows),
        problem=problem, seed=int(seed),
        points=(float(p.light_ms), float(p.top_ms), float(p.turn_n)),
        n_measured=n_measured, samples=tuple(rows))


# ---------------------------------------------------------------- the score


def sub_scores(metrics: dict, reference: FoilScoreReference) -> dict:
    """0-100 sub-score per criterion the band covers (higher = better).

    Unclipped, for ``wing_score.sub_scores``' reason: a design outside the
    band scores past its end, and clipping would flatten the objective
    exactly where the search is winning.
    """
    out: dict[str, float] = {}
    for key, val in metrics.items():
        if val is None or key not in reference.bounds:
            continue
        lo, hi = reference.band(key)
        span = max(np.finfo(float).eps, hi - lo)
        out[key] = (100.0 * (hi - val) / span if key in LOWER_BETTER
                    else 100.0 * (val - lo) / span)
    return out


#: the composite's own units. A 0-100 weighted index (``sub_scores``),
#: not clipped, and comparable with nothing that is not the same
#: weights over the same reference point.
COMPOSITE_UNITS = "composite J (0-100)"


def composite(metrics: dict, reference: FoilScoreReference,
              weights: FoilScoreWeights) -> tuple[float | None, dict, str]:
    """``(J, sub-scores, reason)`` — the weighted composite of one design.

    A weighted criterion with no value (this design would not fly that point)
    or no band (this box could not measure it) makes J ``None`` with a reason
    naming it. Never a silent zero: "did not fly light" and "flies light
    badly" are different answers, and only the second one is a score.
    """
    w = weights.normalised()
    scores = sub_scores(metrics, reference)
    missing = [k for k in CRITERIA if w[k] > 0.0 and k not in scores]
    if missing:
        return None, scores, ("not measured on this design: "
                              + ", ".join(missing))
    total = sum(w[k] * scores[k] for k in CRITERIA if w[k] > 0.0)
    if not np.isfinite(total):
        return None, scores, "non-finite criterion"
    return float(total), scores, ""


def population_rank(value: float | None, reference: FoilScoreReference,
                    weights: FoilScoreWeights) -> float | None:
    """Fraction of the measured box population this J beats, or None."""
    if value is None or not reference.samples:
        return None
    scored = [composite(row, reference, weights)[0]
              for row in reference.samples]
    scored = [s for s in scored if s is not None]
    if not scored:
        return None
    return float(sum(1 for s in scored if float(value) > s)) / len(scored)


def composite_evaluation(x, raw: dict, reference: FoilScoreReference,
                         weights: FoilScoreWeights, metrics_of_x) -> dict:
    """One composite evaluation with its working.

    ``score``/``f`` ARE THE OBJECTIVE THE OPTIMISER SAW. The family's own
    objective stays available under ``f_lod``, so a shell that reads ``score``
    to draw "best objective" cannot print an L/D under a composite run — the
    contract ``wing_score`` and ``airfoil_select`` both keep.
    """
    out = dict(raw)
    # ...and the axis's name changes with it, on EVERY exit. On a
    # SIZED family ``raw`` arrived carrying score_units="payload L/D"
    # (sizing.SizedState.report); leaving it would label a 0-100
    # index — or a PENALTY sentinel — as a W_fixed/D. Set once, here,
    # because two of the three returns below are early ones.
    out["score_units"] = COMPOSITE_UNITS
    # "LoD" first — see wing_score.composite_evaluation. This family has
    # no size modifier today, so the old order was correct by accident;
    # it stops being correct the moment one is registered.
    out["f_lod"] = (raw["LoD"] if raw.get("LoD") is not None
                    else raw.get("score", raw.get("f")))
    out["metrics"], out["scores"], out["composite"] = {}, {}, None
    if not raw.get("feasible"):
        out["score"] = out["f"] = PENALTY
        return out
    measured = metrics_of_x(np.asarray(x, dtype=float))
    out["metrics"] = measured
    j, scores, reason = composite(measured, reference, weights)
    out["scores"] = scores
    if j is None:
        out["score"] = out["f"] = PENALTY
        out["composite_reason"] = reason
        return out
    out["composite"] = j
    out["score"] = out["f"] = j
    return out


def composite_objective(x, evaluate, reference: FoilScoreReference,
                        weights: FoilScoreWeights, metrics_of_x,
                        n_constraints: int = 1, constrained: bool = True):
    """The composite as a harness callable: ``(J, margins)``, or ``J`` alone.

    Same return contract as every ``fg_*`` in the package, and the same
    division of labour as ``wing_score.composite_objective``: the composite
    changes the OBJECTIVE and never the constraint channel, so a composite arm
    and an L/D control share one feasible region. The margins are the CRUISE
    point's — the point the placement states and the family already
    constrains. A turn that runs out of cavitation margin lowers J; it does
    not make the design infeasible, because the user asked to weight the turn,
    not to fly every design through one.
    """
    raw = evaluate(np.asarray(x, dtype=float))
    out = composite_evaluation(x, raw, reference, weights, metrics_of_x)
    j = out["composite"]
    if not constrained:
        return float(j) if j is not None else PENALTY
    if not raw.get("feasible"):
        return PENALTY, np.full(int(n_constraints), G_FAIL)
    g = raw.get("g")
    if g is None:
        return PENALTY, np.full(int(n_constraints), G_FAIL)
    return ((float(j) if j is not None else PENALTY),
            np.asarray(g, dtype=float))


def make_composite_fg(evaluate, reference: FoilScoreReference,
                      weights: FoilScoreWeights, metrics_of_x,
                      n_constraints: int = 1, constrained: bool = True):
    """Bind :func:`composite_objective` to one built problem."""

    def fg(x):
        return composite_objective(x, evaluate, reference, weights,
                                   metrics_of_x, n_constraints=n_constraints,
                                   constrained=constrained)

    return fg


__all__ = [
    "CRITERIA", "CRITERION", "DEFAULT_LIGHT_MS", "DEFAULT_TOP_MS",
    "ALL_CRITERIA", "DEFAULT_TURN_N", "FOIL_CRITERIA", "LOWER_BETTER",
    "POINT_OF", "PRESETS",
    "FlightPoints", "FoilScoreReference", "FoilScoreWeights", "composite",
    "composite_evaluation", "composite_objective", "flight_metrics",
    "make_composite_fg", "population_rank", "reference_from_payload",
    "sample_reference", "sub_scores", "weights_of",
]
