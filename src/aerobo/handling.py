"""MIL-F-8785C handling qualities as CONSTRAINT ROWS the optimiser can search.

One objective (L/D) and three named rungs. Two of them are MIL-F-8785C's own
— Class I (light aeroplanes), Category B (cruise) — and the third is this
package's, which is why the ladder is NOT nested and why that has to be said
in the first paragraph:

    Level 1   satisfactory: MIL-F-8785C's thresholds, verbatim
    Level 2   acceptable: MIL-F-8785C's thresholds, verbatim
    Level 3   STABLE: every mode of the aeroplane decays. NOT the standard's
              Level 3, which permits a spiral doubling in 4 s and a phugoid
              doubling in 55 s

WHY LEVEL 3 IS NOT THE STANDARD'S. The standard grades PILOT WORKLOAD, and it
never asks the spiral or the phugoid to converge — only to diverge slowly
enough that the pilot corrects without noticing. That is the right question
for a certification document and the wrong one for a search whose designer
asked for an aeroplane that does not wander off on its own. So rung 3 was
redefined, deliberately, to the thing the number cannot be read off the
standard for: ``lambda < 0`` on the spiral and the roll, ``zeta > 0`` on all
three oscillations.

TWO CONSEQUENCES, BOTH LOAD-BEARING. The rungs are no longer nested — rung 3
REFUSES designs Level 2 admits (a 12 s spiral divergence) and ADMITS designs
Level 3 of the standard would have taken at ``zeta_sp = 0.15``. And the
ordering by strictness is :data:`LADDER` = ``(2, 1, 3)``, not the numbers:
a menu that sorts by the level number sorts by nothing.

WHY A STANDARD AND NOT A NUMBER FITTED TO THIS BOX. Measured over 256 Sobol
draws of the shell's own box on `tail [free height]`, the SPIRAL clause the
ladder is named for removes ONE draw across all three levels: every divergent
spiral on that box doubles in 15.9 s or more, so 4 s and 12 s admit all of
them, and at Level 1 its 20 s takes exactly one the phugoid had not already
taken. All but the whole cost of every level is bought by phugoid damping
(``scripts/gate_ladder.py`` prints the clause-by-clause table).

The handover called that out and refused to ship "a gate whose name is not
what it does". The fix is NOT to re-fit the spiral threshold to 15.9 s: that
number is a property of one family's box and transfers nowhere — the same
measurement on `tail + winglet` finds 5.9 % divergent spirals against 21.9 %,
and its Level 1 is dearer (-4.46 % of L/D against -1.66 %) while its Levels 3
and 2 are free. A constant fitted to one box, shipped as a standard's name, is
the defect wearing a different hat.

So the thresholds stay the standard's, and every evaluation reports
:attr:`Margins.binding_clause` — WHICH ROW was the minimum on THIS design.
The gate says what it does per design and per box, at run time, instead of a
document arguing about it. A user who gates at Level 1 and reads "binding
clause: phugoid" on every candidate has learnt the thing the study had to be
run to find out.

SIGNED AND CONTINUOUS, NOT BOOLEAN. ``scripts/gate_ladder.py`` screens a box
with predicates, which is all a screen needs. A constrained optimiser needs a
margin: :mod:`optimize.constrained` tests ``(g >= 0).all()`` but MODELS ``g``,
so a boolean cliff gives the surrogate nothing to follow and the feasible
region has to be found by luck. Every clause here is therefore a real number
that crosses zero exactly where the standard does, and each is written in the
form that stays smooth:

``spiral``
    ``ln2/T2_min - lambda``, on the ROOT and not on T2. The requirement is a
    time to double, but T2 = ln2/lambda has a POLE at lambda -> 0 — the exact
    place a design under this gate is being pushed towards — so a margin
    written in T2 goes to infinity next to the boundary it is meant to
    resolve. On the root it is affine, and a convergent spiral (lambda < 0)
    scores positive without a special case.
``phugoid``, ``dutch roll``, ``short period``
    ``zeta - zeta_min``, the damping ratio against the level's floor.
    A level whose ``phugoid_z`` is ``None`` asks for a time to double
    instead (:data:`PHUGOID_T2_LEVEL_3`), and that row is then a root margin,
    in 1/s rather than dimensionless. NO SHIPPED RUNG USES THAT FORM any
    more — it is the standard's Level 3, which rung 3 replaced — but the code
    path stays, because reinstating MIL's own Level 3 is then one dict entry
    and not a rewrite.

    VALUES ARE NOT COMPARABLE ACROSS RUNGS in either arrangement, and
    FEASIBILITY is no longer monotone either: the ladder is 2 -> 1 -> 3
    (:data:`LADDER`) and rung 3 is a different requirement, not a tighter
    one. What a test can assert is the direction of each ROW.
``roll subsidence``
    ``-lambda - 1/tau_max``. ONE expression carrying BOTH halves of the
    requirement: the mode must be convergent AND its time constant no slower
    than the level allows. A design with a divergent roll (lambda > 0) scores
    negative from the first term, so there is no separate stability test to
    keep in step with the time-constant one.

A MODE THE DESIGN DOES NOT HAVE CANNOT FAIL ITS OWN REQUIREMENT. A design
that does not weathercock has no spiral and no Dutch roll — ``modes.classify``
returns those roots under :data:`modes.SLOW_LATERAL` and
:data:`modes.LATERAL_OSC` instead, and they are NOT looked up here through
``modes.find``: gating a sideslip subsidence on a spiral requirement would be
the two-instruments defect that ``classify`` grew a ``Cn_beta`` argument to
close. The row is reported SATISFIED at :data:`NOT_APPLICABLE`, with the
reason recorded beside it, and the vector keeps its width. That width is a
contract the optimiser depends on, so it is pinned by value and not only by
length (a margin that silently changed from a vector to a scalar has cost
this package a study before).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import modes as _modes

__all__ = ["LEVELS", "CLAUSES", "Margins", "margins", "level_of",
           "NOT_APPLICABLE", "PHUGOID_T2_LEVEL_3", "NON_OSCILLATORY_ZETA",
           "constraint_labels", "N_ROWS", "gate", "G_FAIL_ROWS",
           "LEVEL_NAMES", "LEVEL_MEANING", "CLAUSE_MEANING",
           "level_name", "clause_rows", "MEASURED_COST_PCT", "LADDER",
           "LEVEL_SOURCE", "level_source"]


#: MIL-F-8785C, Class I, Category B. THE GATE IS A STANDARD, NOT A TASTE, and
#: every threshold in this package is in this one dict — ``scripts/
#: gate_ladder.py`` imports it rather than restating it, because the screen
#: and the shipped gate disagreeing about a threshold is the one failure
#: neither of them could detect.
#:
#: Levels 1 and 2 are the standard's, verbatim. RUNG 3 IS NOT: it is
#: "every mode decays", written in the same five keys so that one gate, one
#: label table and one margin function serve all three.
#:
#: An INFINITE time is how a convergence requirement is written here without
#: a second code path: the spiral margin is ``ln2/T2_min - lambda``, so
#: ``T2_min = inf`` is exactly ``-lambda >= 0``, and the roll margin
#: ``-lambda - 1/tau_max`` is exactly ``-lambda >= 0`` at ``tau_max = inf``.
#: Both stay finite and affine in the root, which is the property the
#: surrogate needs.
#:
#: ...AND A NEGATIVE TIME IS A TIME TO **HALF**, in the same one code path:
#: ``ln2/(-T) - lambda >= 0`` is ``lambda <= -ln2/T``, a mode required to
#: DECAY by half within ``T`` seconds. Same function, same affinity in the
#: root, opposite sign.
#:
#: WHY RUNG 3'S SPIRAL NEEDS ONE. ``lambda < 0`` is a bound with no margin,
#: and an optimiser rides a bound: measured over 5 seeds of
#: `tandem (nonplanar) [free cant]` at SLSQP, the winners came back at
#: lambda = -0.000015, -0.000067, -0.000101, -0.002197 and -0.007237 — a
#: spiral that halves in 46,000 s is not a stable aeroplane, it is a
#: satisfied inequality. So rung 3 asks the spiral to HALVE within
#: :data:`SPIRAL_T_HALF_LEVEL_3` seconds instead.
#:
#: THE NUMBER IS THE STANDARD'S OWN, MIRRORED: MIL Level 1 lets a DIVERGENT
#: spiral take 20 s to double, so "stable" asks the convergent one to halve
#: in the same 20 s. Measured reachable, with room, wherever the clause can
#: be met at all — the cant needed is 7.5 deg on `tail [free cant]` (L/D
#: 27.29 against a 27.30 peak over that row), about 6 deg on
#: `tail + winglet [free cant]` (at its peak), and 4.8 deg on the nonplanar
#: pair (25.03 against 25.13). It refuses nothing that ``lambda < 0`` did
#: not already refuse: at zero dihedral those three sit at lambda = +0.053,
#: +0.019 and +0.071, divergent under either reading.
#:
#: A ``phugoid_z`` of ``None`` asks for :data:`PHUGOID_T2_LEVEL_3` instead of
#: a damping ratio. Nothing uses it now — it was MIL's own Level 3, which
#: rung 3 replaced — and it is kept so that reinstating that rung is one
#: line.
#: how fast rung 3 asks a CONVERGENT spiral to decay [s to half].
#:
#: MIL-F-8785C Level 1 permits a DIVERGENT spiral to double in 20 s; this is
#: that number mirrored, and it is the whole difference between "stable" and
#: "not quite unstable". Carried as its own constant, not inlined, because it
#: is the one threshold on this ladder that the standard does not state — a
#: reader has to be able to find it, and ``scripts/gate_ladder.py`` has to be
#: able to import it rather than restate it.
SPIRAL_T_HALF_LEVEL_3 = 20.0

LEVELS: dict[int, dict[str, float | None]] = {
    1: {"spiral_T2": 20.0, "phugoid_z": 0.04, "dr_z": 0.08,
        "sp_z": 0.35, "roll_tau": 1.0},
    2: {"spiral_T2": 12.0, "phugoid_z": 0.00, "dr_z": 0.02,
        "sp_z": 0.25, "roll_tau": 1.4},
    3: {"spiral_T2": -SPIRAL_T_HALF_LEVEL_3, "phugoid_z": 0.00, "dr_z": 0.00,
        "sp_z": 0.00, "roll_tau": math.inf},
}

#: THE RUNGS IN ORDER, LOOSEST FIRST — which is not the order of the numbers
#: and cannot be derived from them. Rung 3 asks for convergence, so it is
#: the strictest of the three on the spiral and the phugoid while being the
#: loosest on the short period. A menu offers them in THIS order.
LADDER: tuple[int, ...] = (2, 1, 3)

#: what Level 3 asks of the phugoid instead of a damping ratio [s].
PHUGOID_T2_LEVEL_3 = 55.0

#: THE ROW ORDER, and therefore the constraint order. Fixed: a caller reading
#: ``g[2]`` must get the same clause on every design and under every flag.
CLAUSES: tuple[str, ...] = ("spiral", "phugoid", "dutch roll",
                            "short period", "roll subsidence")

#: how many constraint rows the gate adds. Read this rather than counting.
N_ROWS = len(CLAUSES)

#: the margin a clause scores when the design HAS NO SUCH MODE.
#:
#: Positive, because a requirement about a mode that does not exist is met
#: rather than failed. SMALL, and deliberately not a large sentinel: the
#: surrogate models ``g``, and a -100/+100 sentinel sets the SCALE of
#: everything else it sees — this package has already paid for that lesson
#: once, in a refusal value that wrecked a GP's fit. 1.0 reads as
#: "comfortably satisfied" beside real margins of order 0.01-2 and distorts
#: nothing.
NOT_APPLICABLE = 1.0

#: the damping ratio a NON-OSCILLATORY mode is scored at where the level asks
#: for one. It is not a damping ratio at all — the mode is two real roots, not
#: an oscillation — and the honest reading is that the requirement cannot be
#: met by something that is not the mode the requirement describes. Negative,
#: so it fails, and it is the value ``scripts/gate_ladder.py`` has always
#: used, so the screen and the gate cannot disagree about such a design.
NON_OSCILLATORY_ZETA = -1.0

#: WHAT EACH LEVEL IS CALLED, in one word.
#:
#: The standard numbers its levels and names them nowhere a menu can reach,
#: and the number alone misleads in both directions: it counts DOWNWARDS to
#: the strictest, so "Level 3" reads like the best of three, and it says
#: nothing about what is being bought. These are MIL-F-8785C's own
#: definitions (1.5) compressed to their subject. A shell shows the word and
#: the number together — never the word alone, because the number is what
#: the flag carries and what a report prints.
LEVEL_NAMES: dict[int, str] = {
    1: "satisfactory",
    2: "acceptable",
    3: "stable",
}

#: the standard's definition of the level, in a sentence, for a reader who
#: has to choose one and has not read MIL-F-8785C. Paraphrased tightly: what
#: is being graded is the PILOT'S workload and whether the mission gets done,
#: not whether the aeroplane is stable — a Level 3 aeroplane is a flyable
#: one, and Level 1 is what a light aeroplane is normally expected to be.
LEVEL_MEANING: dict[int, str] = {
    1: ("clearly adequate for the job — the pilot flies the mission and "
        "does not have to fight the aeroplane to do it"),
    2: ("adequate: the mission still gets done, but the pilot works harder "
        "for it or does it less well"),
    3: ("every mode of the aeroplane decays on its own: it does not wander "
        "off in a spiral, wallow, wag or ring. Stricter than the standard "
        "asks — MIL-F-8785C never requires the spiral or the phugoid to "
        "converge at all"),
}

#: WHAT THE CLAUSE IS, for a reader who knows the aeroplane and not the
#: mode's name. One sentence each, in the same order as :data:`CLAUSES`, so
#: a shell can put the physics beside the threshold instead of printing
#: "dutch roll zeta >= 0.08" at somebody and hoping.
CLAUSE_MEANING: dict[str, str] = {
    "spiral": ("hands off the stick, a small bank angle either washes out "
               "or winds up into a spiral dive — this row says how fast it "
               "may wind up, or how fast it must wash OUT"),
    "phugoid": ("the slow trade of speed against height after a gust, tens "
                "of seconds a cycle — this asks that it dies away instead "
                "of growing"),
    "dutch roll": ("the yaw-and-roll wag that follows a rudder kick or a "
                   "side gust — this asks how quickly the wag damps out"),
    "short period": ("the quick pitch bob in the second after a stick "
                     "input — this asks that it settles rather than rings"),
    "roll subsidence": ("how the roll RATE settles to a steady value when "
                        "the ailerons move — that it settles, and on the "
                        "MIL rungs how quickly"),
}


#: WHAT A LEVEL COSTS A SEARCH, as a percentage of L/D against the shipped
#: gate (the static margin alone). Measured by
#: ``scripts/gate_costs_a_search.py`` — 24 seeds x 48 evaluations per arm,
#: PAIRED per seed, on ``tail [free height]`` and ``tail + winglet``,
#: 2026-09-02. Each entry is (cheapest family, dearest family) so a shell can
#: quote a range rather than one box's number.
#:
#: These are SEARCH costs, not the box screen's. ``scripts/gate_ladder.py``
#: screens draws and gets a different answer in both directions — it called
#: Level 3 free on a family where a search pays 0.6 % — so never quote the
#: screen here.
#: RUNG 3 IS ``None``: its -0.212/-0.598 % was measured against the
#: standard's Level 3, which is not what rung 3 asks any more. A number
#: measured against a requirement that has since changed is worse than no
#: number, because nothing on the screen would say so.
MEASURED_COST_PCT: dict[int, tuple[float, float] | None] = {
    1: (-2.747, -3.063),
    2: (-0.886, -1.253),
    3: None,
}


#: WHERE THE RUNG COMES FROM. Two of the three are a citation and one is not,
#: and a card that printed "MIL-F-8785C" over all three would be claiming a
#: document says something it does not. Every screen that names a rung names
#: its source with it.
LEVEL_SOURCE: dict[int, str] = {
    1: "MIL-F-8785C Class I (light), Category B (cruise)",
    2: "MIL-F-8785C Class I (light), Category B (cruise)",
    3: ("this package's own rung, not MIL-F-8785C — the standard never asks "
        "the spiral or the phugoid to converge"),
}


def level_source(level) -> str:
    """The document a rung's thresholds come from, or the fact that there
    isn't one."""
    return LEVEL_SOURCE[level_of(level)]


def level_name(level) -> str:
    """``'Level 2 (acceptable)'`` — the number the flag carries AND the word.

    Both, always. The number is what ``handling_level`` is and what every
    report prints; the word is the only half a reader who has not met the
    standard can act on.
    """
    lvl = level_of(level)
    return f"Level {lvl} ({LEVEL_NAMES[lvl]})"


def clause_rows(level) -> tuple[tuple[str, str, str], ...]:
    """``(clause, label-with-threshold, what it means)`` per row, in
    :data:`CLAUSES` order.

    One call, so a shell listing the level's requirements cannot drift out
    of step with :func:`constraint_labels` or drop a row the gate scores.
    """
    lvl = level_of(level)
    return tuple(zip(CLAUSES, constraint_labels(lvl),
                     tuple(CLAUSE_MEANING[c] for c in CLAUSES)))


def level_of(level) -> int:
    """Coerce and validate a level, raising with the choices named.

    Accepts the int or a string of it, because a flag arrives from a shell
    as whatever the widget produced and one coercion beats three. It does NOT
    accept anything that has to be truncated to get there: ``int(2.5)`` is 2,
    so a bare ``int()`` silently ran a Level 2 gate for somebody who asked for
    something else and reported Level 2 in the breakdown. A level is one of
    three names, not a quantity to round.
    """
    try:
        lvl = int(level)
        if lvl != float(level):
            raise ValueError
    except (TypeError, ValueError):
        raise ValueError(
            f"handling level must be one of {sorted(LEVELS)}, "
            f"got {level!r}") from None
    if lvl not in LEVELS:
        raise ValueError(
            f"handling level must be one of {sorted(LEVELS)} "
            f"(MIL-F-8785C Class I, Category B), got {lvl}")
    return lvl


def constraint_labels(level) -> tuple[str, ...]:
    """The label of each row, with the threshold that row is against.

    Labels carry the NUMBER, not just the clause: a report that says
    "spiral" tells a reader nothing about which level it was run at, and the
    label is what ``api.design_report`` prints beside the margin.
    """
    L = LEVELS[level_of(level)]
    # an infinite limit IS the convergence requirement (see LEVELS), and it
    # has to be said in words: "spiral T2 >= inf s" is not a sentence.
    # ...and a NEGATIVE one is a time to HALF, which has to be said in words
    # for a stronger reason: "spiral T2 >= -20 s" reads as a LOOSER
    # requirement than Level 2's 12 s when it is the strictest rung on the
    # ladder. The label is what a report prints beside the margin, so the
    # sign convention must not reach a reader.
    sp = ("spiral converges (lambda < 0)" if math.isinf(L["spiral_T2"])
          else "spiral halves in <= %.0f s" % abs(L["spiral_T2"])
          if L["spiral_T2"] < 0.0
          else "spiral T2 >= %.0f s" % L["spiral_T2"])
    ph = ("phugoid T2 >= %.0f s" % PHUGOID_T2_LEVEL_3
          if L["phugoid_z"] is None else "phugoid zeta >= %.2f" % L["phugoid_z"])
    roll = ("roll converges (lambda < 0)" if math.isinf(L["roll_tau"])
            else "roll tau <= %.1f s" % L["roll_tau"])
    return (
        sp,
        ph,
        "dutch roll zeta >= %.2f" % L["dr_z"],
        "short period zeta >= %.2f" % L["sp_z"],
        roll,
    )


@dataclass(frozen=True)
class Margins:
    """The five signed margins of one design, and which one bound it."""

    level: int
    #: one per :data:`CLAUSES`, in that order. ``g >= 0`` is met.
    g: np.ndarray
    #: clause -> why this row is :data:`NOT_APPLICABLE`. Empty on a design
    #: that has all five modes.
    reasons: dict = field(default_factory=dict)
    #: the named modes these margins were read off, when :func:`gate` produced
    #: them. Carried so a caller that wants the eigenvalues does not linearise
    #: a second time, and so a report can print the mode beside its margin.
    named: dict = field(default_factory=dict, repr=False)

    @property
    def binding_clause(self) -> str:
        """The clause with the SMALLEST margin — the one this design is
        actually being held by, whether or not it passes.

        THE ANSWER TO "a gate whose name is not what it does". On the box
        this ladder was measured over it reads ``phugoid`` on almost every
        design, which is the finding the study had to be run to produce, now
        available per design without running one.
        """
        return CLAUSES[int(np.argmin(self.g))]

    @property
    def feasible(self) -> bool:
        return bool((self.g >= 0.0).all())

    @property
    def worst(self) -> float:
        return float(self.g.min())

    def report(self) -> dict:
        """JSON-safe, for the breakdown a shell reads."""
        return {"handling_level": int(self.level),
                "handling_g": [float(v) for v in self.g],
                "handling_labels": list(constraint_labels(self.level)),
                "handling_binding": self.binding_clause,
                "handling_feasible": self.feasible,
                "handling_not_applicable": dict(self.reasons)}


def _root_margin(mode, limit_s: float) -> float:
    """``ln2/limit - lambda``: a time requirement, on the root.

    See the module docstring for why the root and not T2. THE SIGN OF
    ``limit_s`` IS THE REQUIREMENT (:data:`LEVELS` says so where the table
    is): positive is the slowest DIVERGENCE the level tolerates, so the
    margin is positive for anything convergent and for anything diverging
    more slowly than that; ``inf`` is bare convergence; and NEGATIVE is a
    time to HALF, ``lambda <= -ln2/|limit|``, which is a convergence
    requirement with a margin in it rather than a bound to be ridden.

    One expression for all three, and affine in the root in all three, which
    is the property the surrogate needs.
    """
    return math.log(2.0) / float(limit_s) - float(mode.real)


def _zeta_margin(mode, floor: float) -> float:
    """``zeta - floor``, with a non-oscillatory mode scored at
    :data:`NON_OSCILLATORY_ZETA` rather than skipped."""
    z = mode.damping
    return (NON_OSCILLATORY_ZETA if z is None else float(z)) - float(floor)


def _roll_margin(mode, tau_max: float) -> float:
    """``-lambda - 1/tau_max``: convergent AND fast enough, in one number."""
    return -float(mode.real) - 1.0 / float(tau_max)


def margins(named: dict, level, *, Cn_beta=None) -> Margins:
    """The five signed margins of a classified aeroplane.

    ``named`` is what :func:`modes.classify` returned — pass it the deck's
    ``Cn_beta`` there, so a design with no yaw stiffness arrives here with no
    ``"spiral"`` and no ``"dutch roll"`` key rather than with two roots
    wearing names they have not earned. ``Cn_beta`` is optional HERE and only
    to record the reason: the gate's behaviour is set by which keys exist.
    """
    lvl = level_of(level)
    L = LEVELS[lvl]
    g = np.empty(N_ROWS, dtype=float)
    reasons: dict[str, str] = {}

    def _absent(i: int, clause: str, why: str):
        g[i] = NOT_APPLICABLE
        reasons[clause] = why

    sp = named.get("spiral")
    if sp is None:
        _absent(0, "spiral",
                _modes.why_not_a_spiral(Cn_beta)
                or "this design has no spiral mode: no slow lateral real "
                   "root was found in its linearisation")
    else:
        g[0] = _root_margin(sp, L["spiral_T2"])

    ph = named.get("phugoid")
    if ph is None:
        _absent(1, "phugoid", "no phugoid: this design has no slow "
                              "longitudinal oscillation")
    elif L["phugoid_z"] is None:
        g[1] = _root_margin(ph, PHUGOID_T2_LEVEL_3)
    else:
        g[1] = _zeta_margin(ph, L["phugoid_z"])

    dr = named.get("dutch roll")
    if dr is None:
        _absent(2, "dutch roll",
                _modes.why_not_a_spiral(Cn_beta)
                or "no Dutch roll: this design has no lateral oscillation")
    else:
        g[2] = _zeta_margin(dr, L["dr_z"])

    spd = named.get("short period")
    if spd is None:
        _absent(3, "short period", "no short period: this design has no fast "
                                   "longitudinal oscillation")
    else:
        g[3] = _zeta_margin(spd, L["sp_z"])

    rl = named.get("roll subsidence")
    if rl is None:
        _absent(4, "roll subsidence", "no roll subsidence: this design has no "
                                      "fast lateral real root")
    else:
        g[4] = _roll_margin(rl, L["roll_tau"])

    return Margins(level=lvl, g=g, reasons=reasons)


#: the finite infeasible margin a failed evaluation reports. THE PACKAGE-WIDE
#: CONVENTION, restated here as every family restates it (``tail.G_FAIL``,
#: ``wingtail``'s, ``carwing``'s, and eight more): the constrained harness
#: needs a FINITE number on a failure, so the surrogate has a point to fit
#: rather than a hole. Consolidating the eleven copies is worth doing and is
#: not this change.
G_FAIL = -1.0


def G_FAIL_ROWS() -> np.ndarray:
    """All five rows failed — a design the gate could not measure at all.

    Full width, so a failure and a success are the same SHAPE to the
    optimiser: a penalty returned at the wrong width is a shape error inside
    the harness, not a bad score.
    """
    return np.full(N_ROWS, float(G_FAIL))


def gate(deck, level, *, mass_kg: float, b_m: float, body_length_m: float,
         rho: float, CD0: float, oswald_e: float,
         AR_vertical: float | None = None, breakdown: dict | None = None,
         altitude_m: float = 0.0) -> Margins:
    """Fly the deck that was SCORED, and gate it.

    ``deck`` is a :class:`dynamics.Deck` linearised about the design's own
    trimmed attitude — in the wing+tail families, the one the evaluation
    already solved for and already paid for. Everything else here is a number
    the evaluation has in hand.

    WHY NOT ``flightmodel.build_flight_model``. That function rebuilds a
    flyable model from a REPORT, and stages 5 and 6 are right to use it: it is
    the only route from a stored run back to an aeroplane. Inside an
    evaluation it would be the wrong instrument for two reasons. It re-panels
    the whole lattice from the report's planform — 4.8 ms against the 2.2 ms
    the rest of this costs, on an evaluation that is 1.7 ms — and, more to the
    point, it makes the gated aeroplane the one a rebuild RECONSTRUCTS rather
    than the one the objective scored. Those two have agreed since the tip
    device was fixed, and the whole history of this package says that is a
    property to keep testing rather than one to build on: measuring here makes
    it true by construction instead of by agreement.

    WHAT IT SHARES WITH THE REBUILD, deliberately, so the two instruments can
    be compared at all: :func:`sixdof.Inertia.from_layout`, the body length
    from ``drag.body_length_for_arm``, the fin's aspect ratio as ``h^2/(h c)``,
    and ``flightmodel._stall_from_report`` for the section's own stall. What it
    does NOT share is the guessing: the rebuild assumes ``oswald_e = 0.85``
    and reads ``CD0`` back out of a report, while the caller here passes the
    solver's own ``e`` and ``CDp``. That difference is real and it lands on the
    phugoid, which is the clause that binds — so a test pins the size of it
    rather than a docstring claiming it is small.

    Returns :class:`Margins`, or raises nothing: a design that cannot be
    assembled, trimmed or linearised comes back with :func:`G_FAIL_ROWS` and
    the reason on every clause, because an evaluation that raised here would
    turn a handling failure into a crashed run.
    """
    from . import sixdof as sd

    lvl = level_of(level)
    try:
        from .flightmodel import _stall_from_report

        inert = sd.Inertia.from_layout(float(mass_kg), float(b_m),
                                       float(body_length_m))
        stall, _why = _stall_from_report(breakdown or {}, deck)
        ac = sd.Aircraft(deck=deck, inertia=inert, rho=float(rho),
                         CD0=float(CD0), oswald_e=float(oswald_e),
                         AR_vertical=(None if AR_vertical is None
                                      else float(AR_vertical)),
                         stall=stall)
        # THE STATE IS THE ONE THE DESIGN WAS TRIMMED AT — ``deck.alpha_ref``
        # — and ``level_at`` hangs the thrust that holds it there. Not
        # ``trim_level``: this deck carries no control column to solve for,
        # and the pitch trim was already done by the evaluation that built it.
        st, trimmed = sd.level_at(ac, float(deck.V), float(deck.alpha_ref),
                                  float(altitude_m))
        named = _modes.classify(sd.linearise(trimmed, st), float(st.V),
                                Cn_beta=deck.Cn_beta)
    except Exception as exc:                                    # noqa: BLE001
        why = f"the handling gate could not be measured on this design: {exc}"
        return Margins(level=lvl, g=G_FAIL_ROWS(),
                       reasons={c: why for c in CLAUSES})
    m = margins(named, lvl, Cn_beta=deck.Cn_beta)
    return Margins(level=m.level, g=m.g, reasons=m.reasons, named=named)
