"""The measured answer to "which optimiser, and how many evaluations?".

Until V3.5 both numbers were the user's problem. The shell asked for an
optimiser and a budget, pre-filled with a hand-sized default (BO, 40 for a
wing, 24 for a section) that had never been measured against the problems it
was defaulting for. This module is the other end of
:mod:`aerobo.experiments.budget_study`: the study runs every strategy to a
long budget over problems from 3 to 14 design variables, the analysis fits
what it found, and the fit is frozen into ``data/search_budget.json`` — the
same convention the screen's reference band ships under. Nothing here
computes physics; it reads a measurement and answers a question about it.

The recommendation is a function of exactly the two things the study found
it depends on:

* **how many design variables** — the budget law is linear in the dimension
  (``budget ~ intercept + per_dim * d``), fitted per convergence target;
* **what is being optimised** — a millisecond-per-evaluation VLM planform
  and a seconds-per-candidate XFOIL section are different search problems
  and get different strategies, and a constrained problem's BO is a
  different acquisition (LogCEI) from an unconstrained one's.

and one thing that is the user's to say — ``effort``, i.e. how close to
converged they want to be, since "converged" is a target and not an event:

    quick      90 % of the way to the best design the study ever found
    balanced   95 %
    thorough   99 %

A plan is a recommendation and never a limit. Every field it produces is
what the V3.5 shell PRE-FILLS; the user can always switch to their own
values, and the shell says which of the two is live.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

#: The frozen study payload (written by
#: ``aerobo.experiments.budget_analysis --emit``). Beside ``data/
#: screen_reference.json``, the screen's frozen band, for the same reason:
#: a measurement the package ships and a caller can hash.
PAYLOAD_PATH = (Path(__file__).resolve().parents[3] / "data"
                / "search_budget.json")

#: effort -> the convergence fraction the budget is sized for. The fractions
#: are of the study's own normalised progress scale (0 = one design drawn
#: blind out of the box, 1 = the best design any run of that case found).
#:
#: They stop at 99 % because that is where the EVIDENCE stops: over the
#: study's long budgets, 99.9 % of the best-ever design was reached by too
#: few cases to fit a line through (:data:`budget_analysis.MIN_LAW_CASES`),
#: and an effort setting whose budget is an extrapolation is exactly the
#: hand-sized default this module replaces.
EFFORT_TARGETS = {"quick": 0.90, "balanced": 0.95, "thorough": 0.99}
EFFORTS = tuple(EFFORT_TARGETS)

#: what a shell calls the two classes of problem
KINDS = ("wing", "airfoil")

#: the acquisitions :func:`aerobo.optimize.bo.run_bo` can actually dispatch.
#: The constrained runner has its own ("logcei", hard-wired), so a plan for an
#: unconstrained problem must not forward one from this list's complement.
UNCONSTRAINED_ACQF = ("logei", "ucb", "qmes", "qlognei")


class MissingStudyError(RuntimeError):
    """Raised when the frozen study payload is absent or unreadable."""


_CACHE: dict = {}


def payload(path: str | Path | None = None) -> dict:
    """The frozen study, loaded once.

    Raises rather than falling back to a hard-coded guess: a recommendation
    whose provenance cannot be stated is exactly what this module exists to
    replace.
    """
    p = Path(path) if path is not None else PAYLOAD_PATH
    key = str(p)
    if key not in _CACHE:
        if not p.exists():
            raise MissingStudyError(
                f"no search-budget study at {p}. Run\n"
                f"  .venv/bin/python -m aerobo.experiments.budget_study "
                f"--group wing\n"
                f"  .venv/bin/python -m aerobo.experiments.budget_analysis "
                f"--emit {p}")
        _CACHE[key] = json.loads(p.read_text())
    return _CACHE[key]


def study_stamp(path: str | Path | None = None) -> dict:
    """Provenance a shell can print: when it was measured and over what."""
    pl = payload(path)
    return {k: pl.get(k) for k in ("version", "measured", "n_runs", "machine",
                                   "cases", "note")}


# =====================================================================
# The plan
# =====================================================================

@dataclass(frozen=True)
class SearchPlan:
    """One recommended search: the optimiser, the budget, and what it costs.

    ``flags`` is what a :class:`aerobo.api.RunConfig` needs on top of the
    problem's own physics flags (the acquisition function, and the explicit
    Sobol seed size — the split is a search decision, so a plan states it
    rather than letting it be re-derived from the budget).
    """

    kind: str
    dim: int
    constrained: bool
    optimiser: str
    acqf: str | None
    budget: int
    n_init: int | None
    n_restarts: int
    effort: str
    target: float
    patience: int | None
    tol: float | None
    est_seconds: float | None
    per_eval_s: float | None
    objective: str | None = None
    refusal: str = "sentinel"
    why: str = ""
    source: str = ""
    measured: bool = True

    def flags(self) -> dict:
        out: dict = {}
        if self.optimiser == "bo":
            # "logcei" is the CONSTRAINED runner's acquisition and is hard-wired
            # there; aerobo.optimize.bo.run_bo has no branch for it and raises
            # ValueError inside the per-iteration try, which degrades silently
            # to a uniform random draw. So an unconstrained plan must not ask
            # for it — an acqf the runner cannot dispatch is worse than none.
            if self.acqf and not self.constrained \
                    and self.acqf in UNCONSTRAINED_ACQF:
                out["acqf"] = self.acqf
            if self.n_init is not None:
                out["bo_n_init"] = int(self.n_init)
            # only STATED when it is not the default: an unstated flag is the
            # legacy path, and a plan that sends "sentinel" explicitly would
            # make every recommended config differ from its own control
            if self.refusal and self.refusal != "sentinel":
                out["bo_refusal"] = str(self.refusal)
        return out

    def to_dict(self) -> dict:
        return {"kind": self.kind, "dim": int(self.dim),
                "constrained": bool(self.constrained),
                "objective": self.objective,
                "optimiser": self.optimiser, "acqf": self.acqf,
                "budget": int(self.budget),
                "n_init": None if self.n_init is None else int(self.n_init),
                "n_restarts": int(self.n_restarts), "effort": self.effort,
                "target": float(self.target),
                "patience": self.patience, "tol": self.tol,
                "est_seconds": self.est_seconds,
                "per_eval_s": self.per_eval_s,
                "refusal": self.refusal,
                "why": self.why, "source": self.source,
                "measured": bool(self.measured)}

    # ---- what the shell prints ----
    @property
    def label(self) -> str:
        acq = f" ({self.acqf})" if self.acqf and not self.constrained else ""
        return f"{self.optimiser}{acq}, {self.budget} evaluations"

    @property
    def est_text(self) -> str:
        s = self.est_seconds
        if s is None:
            return "unknown"
        if s < 90:
            return f"~{max(1, int(round(s)))} s"
        if s < 5400:
            return f"~{s / 60:.0f} min"
        return f"~{s / 3600:.1f} h"


# =====================================================================
# The rule
# =====================================================================

def _kind_block(kind: str, pl: dict) -> dict:
    block = (pl.get("kinds") or {}).get(kind)
    if block is None:
        raise MissingStudyError(
            f"the study has no {kind!r} block (has: "
            f"{sorted((pl.get('kinds') or {}))})")
    return block


def _time_boxed(block: dict, kind: str, dim: int, effort: str,
                per_eval_s: float | None, path, restarts: int = 1,
                constrained: bool = True) -> float:
    """Evaluations that fit the effort's wall-clock allowance.

    For a class the study could NOT fit a convergence law to — the live-XFOIL
    section, where no strategy reached 95 % of the best design found inside
    160 evaluations and every curve was still climbing — the honest budget is
    not "enough to converge". It is what fits in the time the user is willing
    to spend, at the cost this machine actually pays per candidate.
    """
    allowance = float((block.get("seconds") or {}).get(effort, 900))
    # the same block `recommend` will read for the method, not the constrained
    # one unconditionally: sizing an unconstrained plan on the constrained
    # winner's overhead prices a run of a method it is not going to fly
    key = "method_constrained" if constrained else "method"
    optimiser = (block.get(key) or block.get("method") or {}) \
        .get("optimiser", "bo")
    lo, hi = block.get("budget_clamp", [8, 400])
    best = int(lo)
    for b in range(int(lo), int(hi) + 1):
        t = est_seconds(kind, optimiser, dim, b, per_eval_s=per_eval_s,
                        path=path)
        if t is None:                       # no cost model: fall back to the
            return float(hi)                # class's longest measured budget
        if t * max(1, int(restarts)) > allowance:
            break
        best = b
    return float(best)


def _law(block: dict, target: float) -> dict:
    laws = block.get("law") or {}
    if not laws:
        # a class with no fitted law is time-boxed, and the target is then
        # only a label on the allowance
        return {"intercept": 0.0, "per_dim": 0.0}
    key = str(target)
    if key in laws:
        return laws[key]
    # nearest measured target, never an extrapolation between two fits
    best = min(laws, key=lambda k: abs(float(k) - target))
    return laws[best]


def budget_for(dim: int, kind: str = "wing", *, effort: str = "balanced",
               path=None) -> int:
    """Just the number of evaluations, for a caller that wants only that."""
    return recommend(dim=dim, kind=kind, effort=effort, path=path).budget


def method_for(kind: str, constrained: bool, path=None) -> tuple[str, str | None]:
    block = _kind_block(kind, payload(path))
    m = block["method_constrained" if constrained else "method"]
    return m["optimiser"], m.get("acqf")


def caveat_for(kind: str, constrained: bool, path=None) -> str:
    """What the payload itself records AGAINST the method it recommends.

    ``method_for`` reads two fields out of a method block and drops everything
    the study wrote down beside them. That is the whole of what the shell and
    the report ever see, so a recommendation arrives looking unanimous when the
    payload says it is not. Two cases on the shipped file, neither visible
    today:

    * ``wing`` unconstrained — ``bo-ucb`` wins 59 % of 400 bootstrap draws and
      is TIED with ``bo-logei`` (41 %). ``decisive`` is False.
    * ``airfoil`` — the point winner is ``ga`` (82.3 % of draws). ``bo-logcei``
      is recommended only because, under the ADOPTED ``worst`` refusal policy,
      BO beats ``ga`` on 18/20 seed pairs. The recommendation is a switch away
      from the arm that won on points, and the reason lives in
      ``under_adopted_refusal`` where nothing reads it.

    Everything here is DERIVED from the stored payload — no number is written
    into this function. A study that re-measures and re-emits changes the
    sentence automatically, which is this repo's rule: a published number needs
    a tripwire, never a pinned constant. Returns ``""`` when the payload
    records nothing against its winner (``wing`` constrained: ``bo-logcei``,
    win rate 1.0, ``decisive`` True).
    """
    m = _kind_block(kind, payload(path))[
        "method_constrained" if constrained else "method"]
    bits: list[str] = []

    if m.get("decisive") is False:
        stab = m.get("stability") or {}
        #: `win_rate` and `tied_with` describe the POINT winner, which is not
        #: always the arm recommended — on `airfoil` the block reads
        #: winner='ga' 82.3 % while `optimiser` is 'bo'. Attributing that rate
        #: to the recommendation would be a plain misstatement, and the first
        #: version of this function made it. Name whose rate it is, and drop
        #: the self-reference the tie list carries for the adopted arm.
        held = str(stab.get("winner") or m.get("label") or m["optimiser"])
        rate, draws = stab.get("win_rate", m.get("win_rate")), stab.get("draws")
        tied = [str(t) for t in (m.get("tied_with") or []) if str(t) != held]
        said = f"`{held}` is NOT decisive"
        if rate is not None and draws:
            said += f" ({float(rate):.1%} of {int(draws)} bootstrap draws)"
        if tied:
            said += "; tied with " + ", ".join(f"`{t}`" for t in tied)
        bits.append(said)

    #: the switch, and WHY — this is the part a reader most needs and the part
    #: `method_for` most completely hides
    sup = m.get("superseded_point_winner")
    ua = m.get("under_adopted_refusal") or {}
    if sup and str(sup) != str(m["optimiser"]):
        why = str(ua.get("why") or "").strip()
        said = (f"`{sup}` won on points; `{m.get('label') or m['optimiser']}` "
                f"is adopted instead")
        bits.append(f"{said} — {why}" if why else said)

    return ". ".join(bits)


def refusal_for(kind: str, path=None) -> str:
    """What a refused design is shown to the GP as, for this class.

    The study's own verdict, or ``"sentinel"`` — the path this repo has
    always taken — for a payload that never measured it. A class where no
    imputation cleared the bar records the sentinel too, and its evidence
    says so: "not adopted" and "not measured" are different answers and the
    payload keeps them apart.
    """
    block = _kind_block(kind, payload(path))
    return str((block.get("refusal") or {}).get("adopt", "sentinel"))


def est_seconds(kind: str, optimiser: str, dim: int, budget: int, *,
                per_eval_s: float | None = None, path=None) -> float | None:
    """Wall clock for ``budget`` evaluations: the physics plus the optimiser.

    The two halves are measured separately and for a reason — on a wing
    problem the physics is ~2 ms and the BO machinery is ~0.3-4 s per
    iteration, so a quote built from the physics alone is wrong by three
    orders of magnitude, and it is the OPTIMISER the user is really buying.
    """
    block = _kind_block(kind, payload(path))
    cost = block.get("cost") or {}
    if per_eval_s is None:
        per_eval_s = cost.get("per_eval_s")
    if per_eval_s is None:
        return None
    total = float(per_eval_s) * int(budget)
    ov = (cost.get("overhead") or {}).get(optimiser)
    if cost.get("includes_overhead"):
        # the class's rate came from a full-length serial run whose stopwatch
        # already contains the surrogate. Adding a fitted overhead on top would
        # charge the model twice.
        ov = None
    if ov:
        b = float(budget)
        extra = (float(ov.get("a0", 0.0)) + float(ov.get("a1", 0.0)) * dim) * b
        extra += (float(ov.get("b0", 0.0))
                  + float(ov.get("b1", 0.0)) * dim) * b * b / 1000.0
        # a least-squares fit through measured wall clocks can put a
        # coefficient below zero; an optimiser that gives time BACK is not a
        # thing, and a quote that shrank as the budget grew would read as a
        # bug rather than as a fit artefact
        total += max(0.0, extra)
    return max(0.0, total)


def recommend(*, dim: int, kind: str = "wing", constrained: bool = False,
              effort: str = "balanced", objective: str | None = None,
              per_eval_s: float | None = None,
              path=None) -> SearchPlan:
    """The measured recommendation for one problem.

    ``dim`` is the design vector's length (``api.RunResult.dim`` / the built
    problem's), ``constrained`` whether the problem carries margins, and
    ``per_eval_s`` an optional MEASURED cost of one evaluation — a shell that
    has just built the problem can time one solve for a few milliseconds and
    get an exact quote instead of the study's median for the class.

    ``objective`` is WHICH SCALAR is being maximised. It is not cosmetic: the
    study measured the same problems against their own physical objective and
    against the weighted composite of six criteria, and a weighted sum of six
    landscapes is not the landscape of any one of them. ``"composite"``
    therefore scales the budget by the measured ratio; anything else (None,
    ``"lod"``, ``"cd"``) is the family's own objective, the case the line was
    fitted on.
    """
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    if effort not in EFFORT_TARGETS:
        raise ValueError(f"effort must be one of {EFFORTS}, got {effort!r}")
    dim = max(1, int(dim))
    pl = payload(path)
    block = _kind_block(kind, pl)
    target = EFFORT_TARGETS[effort]
    law = _law(block, target)

    raw = float(law["intercept"]) + float(law["per_dim"]) * dim
    factor = 1.0
    # ``composite_goal`` is charged the composite's factor: it is the same six
    # landscapes, plus a one-sided penalty that is zero over the whole region a
    # search spends its time in. ``composite_asf`` is charged it too: same six
    # criteria, same two XFOIL sweeps, same design vector — only the reduction
    # over them differs, and a max-min of six landscapes is no cheaper to
    # search than their sum. The factor was MEASURED on "composite" only —
    # said here rather than silently falling back to the family's own line,
    # which is the one thing the measurement says is wrong. It is a borrowed
    # number in both cases and neither has a budget study of its own.
    if str(objective) in ("composite", "composite_goal", "composite_asf"):
        factor = float((block.get("objective_factor") or {})
                       .get("composite", 1.0))
        raw *= factor
    lo, hi = block.get("budget_clamp", [8, 400])
    # RESTARTS SPLIT THE BUDGET, they do not multiply it. The study compared
    # one run of B against k runs of B/k at the SAME total spend, so a
    # recommendation of k restarts is a recommendation about how to spend the
    # same B — a per-run budget of B and k of them would be k times the
    # search the law was fitted for.
    restarts = max(1, int(block.get("restarts", 1)))
    if block.get("budget_mode") == "time":
        # ...and the wall clock is the TOTAL, so the per-run budget is what
        # fits k times over. Solved on the per-run budget directly, because
        # the optimiser's cost is superlinear in it: k runs of B/k are
        # cheaper than one of B, and sizing them from one long run's cost
        # would leave most of the allowance unspent.
        budget = int(_time_boxed(block, kind, dim, effort, per_eval_s, path,
                                 restarts, constrained=bool(constrained)))
        total = budget * restarts
        budget = int(min(int(hi), max(int(lo), budget)))
    else:
        total = int(min(int(hi), max(int(lo), int(math.ceil(raw)))))
        budget = max(int(lo), int(math.ceil(total / restarts))) \
            if restarts > 1 else total

    optimiser, acqf = method_for(kind, constrained, path)
    n_init = None
    if optimiser == "bo":
        spec = block.get("n_init", {})
        # A CONSTRAINED BOX ASKS A DIFFERENT QUESTION OF THE SEED. Unconstrained,
        # the initial design only has to start the surrogate somewhere;
        # constrained, it decides whether the run has seen a FEASIBLE point at
        # all — and a run that has not does not recover, it rides a box corner
        # (optimize.feasible). On the reported 20-D box, 0.5 x dim (10 points)
        # returned nothing on 2 of 3 seeds while dim + 1 (21) returned a design
        # on 3 of 3: 0/0/12 against 28/5/31 feasible evaluations. So the
        # constrained arm may state its own rule, and where it does not, the
        # unconstrained one still applies.
        if constrained and isinstance(spec.get("constrained"), dict):
            spec = {**spec, **spec["constrained"]}
        n_init = int(round(float(spec.get("per_dim", 2.0)) * dim))
        n_init = max(int(spec.get("min", 4)),
                     min(int(spec.get("max", 16)), n_init))
        n_init = max(1, min(n_init, budget - 1))

    stop = block.get("stop") or {}
    est = est_seconds(kind, optimiser, dim, budget, per_eval_s=per_eval_s,
                      path=path)
    if block.get("budget_mode") == "time":
        # the allowance is what was ASKED for; `est` is what the budget the
        # clamp actually allowed will cost. Quoting the allowance while the
        # ceiling binds states a duration the plan itself contradicts, so the
        # sentence names the smaller of the two and says when it is the ceiling
        # rather than the clock doing the limiting.
        allowance = float((block.get("seconds") or {}).get(effort, 900))
        mins = (min(allowance, float(est)) if est else allowance) / 60.0
        capped = bool(est is not None and budget >= int(hi)
                      and float(est) < allowance)
        # whether ANY strategy reached this target is a property of the class's
        # own measurement, not of the mode. The section reaches its lower
        # targets under the adopted refusal policy; it is the top one that
        # nothing reaches, and saying "none of them" flatly was false.
        reached = (block.get("targets_reached") or {}).get(f"{target:g}")
        head = (f"{dim} design variables, {kind}: ")
        if not reached:
            # None means the payload never measured time-to-target, which is
            # the same state that put this class in the time-boxed branch at
            # all — no fitted law, no evidence anything converged. False means
            # it WAS measured and nothing got there. Both say the same thing.
            head += (f"NO strategy in the study reached {target:.0%} of the "
                     f"best design found, at any budget it ran — every curve "
                     f"was still climbing. ")
        else:
            head += (f"{reached} reached {target:.0%} of the best design "
                     f"found in {block.get('targets_evals', {}).get(f'{target:g}', '?')} "
                     f"evaluations on the hardest case measured. ")
        why = (head + f"The budget here is what fits in about {mins:.0f} "
               f"minutes at the measured cost per candidate, not a "
               f"convergence target")
        if capped:
            why += (f" — and it is the ceiling of {int(hi)} evaluations, the "
                    f"longest this class was measured at, that sets it rather "
                    f"than the clock")
    else:
        why = (f"{dim} design variables, {kind}: the study reached "
               f"{target:.1%} of its best design in about {total} "
               f"evaluations ({law['per_dim']:.1f} per variable + "
               f"{law['intercept']:.0f}"
               + (f", x{factor:g} for the composite objective"
                  if factor != 1.0 else "") + ")")
    if restarts > 1:
        why += f", split over {restarts} independent restarts"
    refusal = refusal_for(kind, path) if optimiser == "bo" else "sentinel"
    if refusal != "sentinel":
        why += ("; a refused design is shown to the surrogate as the worst "
                "one that flew, not as the -100 sentinel")
    return SearchPlan(
        kind=kind, dim=dim, constrained=bool(constrained),
        objective=(str(objective) if objective else None),
        optimiser=optimiser, acqf=acqf, budget=budget, n_init=n_init,
        n_restarts=restarts, effort=effort, target=target,
        patience=stop.get("patience"), tol=stop.get("tol"),
        est_seconds=est, per_eval_s=(per_eval_s
                                     if per_eval_s is not None
                                     else (block.get("cost") or {})
                                     .get("per_eval_s")),
        refusal=refusal,
        why=why, source=str(pl.get("measured", "")),
    )


# =====================================================================
# The adaptive stop
# =====================================================================

@dataclass
class ConvergenceStop:
    """"Keep going until it stops improving" — the study's own stopping rule.

    Feed it the running best after every evaluation; ``should_stop`` turns
    True once ``patience`` consecutive evaluations have improved the
    incumbent by less than ``tol`` of the span the search has covered so far.
    The span is measured from the run itself (best minus the first finite
    incumbent), so the rule needs no prior knowledge of the objective's
    scale — which is what lets it run on a problem the study never saw.

    ``max_evals`` is the backstop a shell must always pass: an adaptive stop
    that cannot be exhausted is an unbounded run.
    """

    patience: int
    tol: float
    max_evals: int
    history: list = field(default_factory=list)
    _first: float | None = None

    def update(self, best_so_far: float | None) -> bool:
        v = None if best_so_far is None else float(best_so_far)
        if v is not None and not math.isfinite(v):
            v = None
        self.history.append(v)
        if v is not None and self._first is None:
            self._first = v
        return self.should_stop

    @property
    def n(self) -> int:
        return len(self.history)

    @property
    def should_stop(self) -> bool:
        if self.n >= self.max_evals:
            return True
        finite = [v for v in self.history if v is not None]
        if len(finite) < 2 or self.n <= self.patience:
            return False
        span = abs(finite[-1] - self._first) if self._first is not None else 0.0
        if span <= 0.0:
            return False
        then = self.history[self.n - 1 - self.patience]
        if then is None:
            return False
        return (finite[-1] - float(then)) <= self.tol * span

    @property
    def reason(self) -> str:
        if self.n >= self.max_evals:
            return f"budget spent ({self.max_evals} evaluations)"
        return (f"no improvement worth {self.tol:.1%} of the span in "
                f"{self.patience} evaluations")


def converged_report(history, patience: int | None,
                     tol: float | None) -> dict:
    """Did this finished run flatten out, or was it still climbing?

    The question every "the budget ran out" result raises and nothing
    answered: a user looking at a number has no way to tell whether more
    evaluations would buy anything. This replays the FINISHED run's
    best-so-far log through :class:`ConvergenceStop` itself — the same
    arithmetic, so the verdict on the results page and the rule that would
    have stopped the run can never drift apart.

    ``history`` is ``RunResult.history`` (best-so-far per evaluation, None
    where the incumbent was not finite). ``patience`` / ``tol`` come from the
    run's own plan (:func:`recommend`); either being None means the study
    published no rule for this kind, and the verdict is ``"unmeasured"``.

    Returns ``verdict`` in:

    ``converged``     the last ``patience`` evaluations bought less than
                      ``tol`` of the span the run covered
    ``climbing``      they bought more — the search had not run out of moves
    ``too_short``     the run is not longer than ``patience``, so the rule
                      could not have fired either way. This is the state a
                      RECOMMENDED wing run is in at every budget the study
                      recommends (patience 40 against budgets of 17-53), and
                      saying so is the whole point: a switch that cannot fire
                      must not read as "did not converge".
    ``unmeasured``    no rule exists for this kind
    ``empty``         nothing finite was ever evaluated
    """
    hist = list(history or [])
    if patience is None or tol is None:
        return {"verdict": "unmeasured", "n": len(hist), "patience": patience,
                "tol": tol, "span": None, "gain": None, "gain_frac": None,
                "need": None,
                "text": "this kind of search has no measured stopping rule"}
    patience, tol = int(patience), float(tol)
    finite = [v for v in hist if v is not None]
    need = patience + 1
    if len(finite) < 2:
        return {"verdict": "empty", "n": len(hist), "patience": patience,
                "tol": tol, "span": None, "gain": None, "gain_frac": None,
                "need": need,
                "text": "fewer than two evaluations produced a finite score"}
    first = finite[0]
    span = abs(finite[-1] - first)
    if len(hist) <= patience:
        return {"verdict": "too_short", "n": len(hist), "patience": patience,
                "tol": tol, "span": span, "gain": None, "gain_frac": None,
                "need": need,
                "text": (f"the rule needs more than {patience} evaluations "
                         f"before it can fire and this run made {len(hist)}, "
                         f"so it says nothing either way")}
    then = hist[len(hist) - 1 - patience]
    if then is None:
        return {"verdict": "climbing", "n": len(hist), "patience": patience,
                "tol": tol, "span": span, "gain": None, "gain_frac": None,
                "need": need,
                "text": (f"there was no finite incumbent {patience} "
                         f"evaluations ago, so the run was still finding its "
                         f"first feasible designs")}
    gain = float(finite[-1]) - float(then)
    frac = None if span <= 0.0 else gain / span
    stopped = span > 0.0 and gain <= tol * span
    return {"verdict": "converged" if stopped else "climbing",
            "n": len(hist), "patience": patience, "tol": tol, "span": span,
            "gain": gain, "gain_frac": frac, "need": need,
            "text": (f"the last {patience} evaluations bought "
                     f"{'nothing' if frac is None else f'{frac:.1%}'} of the "
                     f"span this run covered"
                     + (f", under the {tol:.1%} the rule calls flat"
                        if stopped else
                        f", over the {tol:.1%} the rule calls flat"))}
