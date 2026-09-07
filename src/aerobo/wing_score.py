"""The WING's own weighted composite — ten criteria, one number, one search.

The 2-D layer already had this: the library screen ranks sections on six
weighted criteria and ``airfoil_select.composite_objective`` makes that same
weighted J the thing the optimiser maximises. Everything above the section —
a wing, a wing + tail, a tandem pair, a hydrofoil — had exactly one number
instead: L/D (or payload L/D under the size modifier). So a user who cares
about the stall margin, the spar, or a tip chord anyone can actually build
could read those numbers afterwards and change nothing about the search.

This module is the wing-level twin of ``airfoil_select``'s scoring half:

* :data:`WING_CRITERIA` — twelve criteria measured off the breakdown a family
  ALREADY returns, plus the problem that produced it (no extra solve, no
  extra XFOIL call);
* :class:`WingScoreWeights` — the user's own weights, normalised to sum 1;
* :class:`WingScoreReference` — the frozen (lo, hi) normalisation band, and
* :func:`sample_reference` — how that band is measured: a quasi-random sweep
  of the SAME design box the search will run in, before the search starts.

Why the band is measured and not shipped
----------------------------------------
The 2-D composite normalises against 631 screened library sections — a fixed
population that exists whatever problem is being solved. A wing has no
library: an L/D of 30 is excellent for a car wing and poor for a sailplane,
and the same is true criterion by criterion. The only population that means
anything is THE BOX THE USER DREW, so the band is p5/p95 over a sample of it,
measured once, hashed, and then frozen for the whole run.

Frozen is the load-bearing word, and it is the same argument as the 2-D case:
a live min-max over the evaluations seen so far is not a fixed function of x,
so best-so-far would not be monotone, two seeds would not be comparable, and
a two-point population would score 100 and 0 whatever it contained.

What the ten criteria are, and what they are not
------------------------------------------------
Every one of them is read off the breakdown dict the family's own
``evaluate`` returns — from the loading the solver actually solved — plus
the section polars and thicknesses that live on the problem rather than in
its answer:

  lod      the family's OWN objective (L/D, or payload L/D when sized)
  cl_peak  the largest |c_l| any strip flies — the stall proxy, since the
           first strip to reach its section ceiling is what stalls the wing
  espan    span efficiency (reported ``e``/``e_total``, else CL^2/(pi AR CDi))
  bend     root bending index = max over surfaces of |INT c_l c y dy| [m^3],
           the spar load per unit dynamic pressure
  build    the smallest chord actually flown [m], tip devices excluded — a
           mould line and a Reynolds number both have a floor
  alpha    trim attitude [deg]
  clmax    usable CL before the first strip leaves its polar (the loading
           SHAPE held fixed — the critical-section estimate, not a solve)
  astall   degrees of incidence left on that same strip
  vol      volume enclosed by the surfaces [m^3], 0.685 t c per section
  mass     Raymer's statistical GA wing weight of the surfaces [kg]

What they are NOT: ``clmax`` is not ``stall.wing_clmax``'s exact affine
solve (that one needs a per-strip 2-D ceiling and a plain free-air lifting
line, which a tandem, a tail and every VLM family do not have), ``mass`` is
a CORRELATION and not a structure (``sizing.py`` owns the real root-bending
constraint, and it still gates every sized run), and ``vol`` is a section-
area proxy, not a CAD volume. Each is a number already implied by the
answer, given a sense and a band so a user can say which of them the search
should buy.

A criterion this family cannot measure comes back None — never zero — and
:func:`composite` refuses to score a weight on it rather than pretending.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass

import numpy as np

# Failure contract, restated here rather than imported: this module is the
# scoring layer a shell reads criterion LABELS from, and importing
# objective.py for two floats would drag the whole physics stack into
# ``import aerobo.wing_score``. tests/test_wing_composite.py pins both
# constants against objective.PENALTY / objective.G_FAIL so the restatement
# cannot drift.
PENALTY = -100.0
G_FAIL = -1.0

# Band percentiles. Wider than airfoil_select's p2/p98 because the population
# is a few dozen box samples rather than 631 library sections: at n = 32, p2
# and p98 are pure extrapolation between the two extreme samples.
REF_P_LO, REF_P_HI = 5.0, 95.0

# Reference payload format; bumped whenever a stored field changes meaning so
# an old payload fails loudly instead of being read under new semantics.
REFERENCE_VERSION = 1

# Default size of the band sweep. Each sample costs ONE evaluation of the
# family — microseconds for a lifting line, seconds for anything holding an
# XFOIL sweep — so the shell states the cost before it spends it.
REFERENCE_SAMPLES = 32

# A band measured over fewer feasible samples than this is refused: p5/p95 of
# five points is two points and a straight line through the middle.
MIN_REFERENCE_SAMPLES = 8


# ------------------------------------------------------------- the criteria


@dataclass(frozen=True)
class Criterion:
    """One scored quantity: where it comes from, and which way is better."""

    key: str
    label: str
    unit: str
    sense: str          # "max" | "min"
    help: str


#: Static margin past which MORE IS NOT BETTER, as a fraction of the mean
#: chord. The ``stab`` criterion saturates here.
#:
#: Two separate reasons, and the second is the one that forced it.
#:
#: PHYSICS: a large margin is trim drag and a sluggish pitch response. A
#: conventional light aircraft is flown somewhere around 0.10-0.25 mac; a
#: criterion that scores 1.14 as better than 0.25 is not describing an
#: aeroplane anybody wants. ``SM_min`` (0.08) is the floor the CONSTRAINT
#: enforces; this is the ceiling past which the OBJECTIVE stops paying.
#:
#: SCALARISATION: the composite is a weighted SUM, and a weighted sum can
#: only find vertices of its frontier's convex hull — so without a ceiling
#: the margin dial is bang-bang. MEASURED on the `tail` box with a fuselage,
#: over 2894 gate-passing draws: the argmax sat at SM 0.119 for every weight
#: up to 0.42 and then jumped straight to SM 1.143, buying 9.6x the margin
#: for 18.8 % of L/D in one step. There is no weight that asks for "a bit
#: more margin". Saturated at 0.25 the same sweep gives 2.35x the margin for
#: 3.67 % of L/D and stops there. (The package's own general fix for this is
#: ``airfoil_select``'s achievement scalarising function, which exists
#: because "a weighted sum sells" — the wing composite has no ASF yet, and
#: this ceiling is the cheap stand-in.)
#:
#: A CALIBRATION, NOT A BAN: nothing refuses a design above it, and the
#: run's own constraint is untouched. Raise it and the criterion will buy
#: more margin; it is one number and this is the only place it is written.
STAB_TARGET_MAC = 0.25

WING_CRITERIA: tuple[Criterion, ...] = (
    Criterion(
        "lod", "L/D at the design point", "", "max",
        "The family's own objective — L/D at the trimmed design lift, or "
        "payload L/D where the size modifier makes the weight a variable. "
        "Weight this alone and the composite reproduces the ordinary run."),
    Criterion(
        "stab", "static margin", "mac", "max",
        "How far ahead of the neutral point the CG sits, as a fraction of "
        "the mean chord — the pitch stiffness the design actually flies "
        "with. Every wing+tail family already computes it (it is the "
        "constraint the run is gated on), so weighting it costs no extra "
        "physics. It is a PREFERENCE ABOVE A FLOOR, not a second copy of "
        "that gate: the constraint refuses anything under SM_min, and this "
        "buys margin over it with L/D. SATURATES at "
        "wing_score.STAB_TARGET_MAC, because past a normal flying margin "
        "more is not better and an unbounded term makes the weight "
        "bang-bang.\n\n"
        "MEASURED on the `tail` box with a fuselage charged, over 2894 "
        "draws that pass the margin gate: the `stable` preset buys 2.35x "
        "the static margin (0.107 -> 0.252) for 3.67 % of L/D. Read those "
        "two numbers together — a margin gain quoted without its L/D price "
        "is not an answer.\n\n"
        "TWO THINGS THAT WILL MISLEAD YOU IF YOU MEASURE THIS YOURSELF. "
        "(1) Sample only designs that pass the run's own constraint: the "
        "L/D optimum sits ON the SM_min floor, so a sweep that includes "
        "gate-failing draws reports a trade the optimiser can never make. "
        "(2) Charge the fuselage. Without it the tail arm is free, the arm "
        "raises the margin at almost no cost, and this criterion buys a "
        "longer aeroplane rather than a better-balanced one — measured, "
        "the L/D-alone winner's arm falls from 6.79 m to 3.03 m the moment "
        "a 0.35 m body is charged."),
    Criterion(
        "spiral", "spiral divergence", "", "max",
        "The lateral half of stability: Cl_beta*Cn_r - Cn_beta*Cl_r, the "
        "textbook spiral criterion, scored as min(margin, 0) so it measures "
        "DIVERGENCE ONLY. Negative is a wing that rolls off; zero is "
        "convergent and every convergent design ties there. That "
        "saturation is the point and not a simplification: the raw margin "
        "rewarded without limit is a RATCHET — dJ/dGamma stays positive "
        "out to 15 deg of dihedral, so any weight past about a sixth of "
        "L/D's drives the design straight to the dihedral bound. Clipped "
        "at the physical boundary it buys a convergent spiral and then "
        "stops paying, which is what a designer actually wants. It costs a "
        "lateral deck on the SCORED lattice (+44 % of an evaluation), so "
        "it is measured only when it is weighted, and it is measured on "
        "the aeroplane that was scored — the flight rebuild drops the tip "
        "device, which carries most of Cl_beta."),
    Criterion(
        "cl_peak", "peak section c_l", "", "min",
        "The largest |c_l| any strip flies at the design point. The wing "
        "stalls when the first strip reaches its own 2-D ceiling, so a lower "
        "peak at the same CL is stall margin — and a flatter, more elliptic "
        "load."),
    Criterion(
        "espan", "span efficiency", "", "max",
        "e from the solver where it reports one, else CL^2/(pi AR CDi) over "
        "the whole system — how close the induced drag is to the elliptic "
        "minimum for the span being flown."),
    Criterion(
        "bend", "root bending index", "m^3", "min",
        "INT c_l c y dy over the semispan, worst surface — the root bending "
        "moment per unit dynamic pressure. Falls when the lift moves inboard "
        "AND when the span shrinks, which is exactly the trade a long thin "
        "wing is making."),
    Criterion(
        "build", "smallest chord flown", "m", "max",
        "The narrowest chord on any lifting surface, tip devices excluded. "
        "A mould line, a spar cap and a section Reynolds number all have a "
        "floor, and a taper the optimiser likes can walk under all three."),
    Criterion(
        "alpha", "trim attitude", "deg", "min",
        "The root incidence the trim solve landed on. A design that reaches "
        "its lift at a lower attitude carries the fuselage flatter and keeps "
        "more of the polar's linear range in hand."),
    Criterion(
        "clmax", "usable CL before stall", "", "max",
        "CL at the point where the FIRST strip reaches the top of its own "
        "section polar, holding the flown loading shape: CL x min_y(c_l "
        "ceiling / |c_l(y)|). The ceiling is the highest c_l the section's "
        "table is valid at — the same limit the objective already treats as "
        "its stall proxy. A landing speed, in the units the wing works in."),
    Criterion(
        "astall", "stall-angle margin", "deg", "max",
        "Degrees of incidence left on the most-loaded strip before it leaves "
        "the section polar's valid range. Where c_l max asks how much LIFT "
        "is left, this asks how much ANGLE — the number a gust, a turn or a "
        "flare eats into."),
    Criterion(
        "vol", "internal volume", "m^3", "max",
        "Volume enclosed by the lifting surfaces: INT A_section dy with "
        "A = 0.685 t c (the NACA 4-digit area coefficient). Spar depth, "
        "fuel, batteries, payload and every servo run live in it, and a "
        "planform optimised on drag alone spends it first."),
    Criterion(
        "mass", "structural mass", "kg", "min",
        "Raymer's statistical GA wing weight (weights.wing_weight_raymer) "
        "for each surface at the design weight the trim implies, summed. A "
        "CORRELATION, not a structure: it is calibrated on light aircraft, "
        "so read it as a relative ranking inside one design box and never "
        "as a mass estimate for a hydrofoil or a car wing."),
)

CRITERIA: tuple[str, ...] = tuple(c.key for c in WING_CRITERIA)
CRITERION: dict[str, Criterion] = {c.key: c for c in WING_CRITERIA}
HIGHER_BETTER = tuple(c.key for c in WING_CRITERIA if c.sense == "max")
LOWER_BETTER = tuple(c.key for c in WING_CRITERIA if c.sense == "min")


# ---------------------------------------------------------------- weights


@dataclass(frozen=True)
class WingScoreWeights:
    """Criterion weights (normalised to sum 1 in :meth:`normalised`).

    The default is the ordinary run written as weights: L/D alone. Anything
    else is a deliberate statement that the search should pay L/D for
    something, which is the whole point of the composite.
    """

    lod: float = 1.0
    stab: float = 0.0
    spiral: float = 0.0
    cl_peak: float = 0.0
    espan: float = 0.0
    bend: float = 0.0
    build: float = 0.0
    alpha: float = 0.0
    clmax: float = 0.0
    astall: float = 0.0
    vol: float = 0.0
    mass: float = 0.0

    def normalised(self) -> dict[str, float]:
        d = asdict(self)
        if any(v < 0.0 for v in d.values()):
            raise ValueError(f"negative weight in {d}")
        tot = sum(d.values())
        if tot <= 0.0:
            raise ValueError("all weights zero")
        return {k: v / tot for k, v in d.items()}

    def active(self) -> tuple[str, ...]:
        """Criteria with a non-zero weight — the ones a band must cover."""
        d = self.normalised()
        return tuple(k for k in CRITERIA if d[k] > 0.0)


#: Named starting points, in the same spirit as airfoil_select.PRESETS: a job
#: description, not a rule. Every one of them is editable in the shell, and
#: the shell says which preset it started from.
PRESETS: dict[str, WingScoreWeights] = {
    # the ordinary run, written as weights
    "efficiency": WingScoreWeights(lod=1.0),
    # cruise first, but pay a little for the stall margin and the load path
    "cruise": WingScoreWeights(lod=0.55, cl_peak=0.10, espan=0.10,
                               bend=0.10, build=0.05, clmax=0.10),
    # the ask in two terms: cruise, bought against pitch stiffness. Only
    # the wing+tail families can measure it — elsewhere the composite says
    # so rather than quietly scoring the pair as one.
    #
    # MEASURED on the `tail` box, fuselage charged, 2894 gate-passing draws:
    # L/D alone lands on L/D 29.8456 at SM 0.1071; this preset on L/D 28.7493
    # at SM 0.2521 — 2.35x the margin for 3.67 % of L/D, and it stops there
    # because the criterion saturates at STAB_TARGET_MAC.
    #
    # An earlier version of this comment claimed 3.85x for 1.68 %. That was
    # measured over a sample that included designs FAILING the run's own
    # static-margin constraint, and with no fuselage drag, so most of what
    # it "bought" was a longer aeroplane the optimiser could not have
    # chosen. Both mistakes are named in the criterion's help text because
    # they are the two that anyone re-measuring this will make.
    "stable": WingScoreWeights(lod=0.70, stab=0.30),
    # handling: what the wing does at the slow end, and how flat it cruises
    "docile": WingScoreWeights(lod=0.30, cl_peak=0.15, espan=0.05,
                               build=0.05, alpha=0.10, clmax=0.20,
                               astall=0.15),
    # structure and shop floor: the spar load, the mass and the chord
    "buildable": WingScoreWeights(lod=0.35, espan=0.05, bend=0.20,
                                  build=0.15, vol=0.10, mass=0.15),
}


def weights_of(weights) -> WingScoreWeights:
    """Coerce a preset name / partial dict / WingScoreWeights to weights.

    A partial dict MERGES over the ``efficiency`` preset — the same rule
    ``api.screen_weights`` follows for the 2-D screen, so "set bend to 0.3"
    means what it looks like and the untouched criteria keep their default.
    """
    if isinstance(weights, WingScoreWeights):
        return weights
    if weights is None:
        return WingScoreWeights()
    if isinstance(weights, str):
        try:
            return PRESETS[weights]
        except KeyError:
            raise ValueError(
                f"unknown wing weight preset {weights!r} — "
                f"one of {sorted(PRESETS)}") from None
    if isinstance(weights, dict):
        unknown = set(weights) - set(CRITERIA)
        if unknown:
            raise ValueError(
                f"unknown wing criteria {sorted(unknown)} — "
                f"one of {list(CRITERIA)}")
        base = asdict(WingScoreWeights())
        base.update({k: float(v) for k, v in weights.items()})
        return WingScoreWeights(**base)
    raise ValueError(f"cannot read weights from {type(weights).__name__}")


# ------------------------------------------------------- the flown surfaces


def _arr(obj, *names):
    """First 1-D numeric array attribute of ``obj`` among ``names``.

    Callables are skipped rather than read: ``geometry.Wing`` answers
    ``chord`` as a METHOD (the chord law sampled wherever you ask), and
    coercing a bound method to an array is a TypeError, not a surface.
    """
    for name in names:
        val = getattr(obj, name, None)
        if val is None or callable(val):
            continue
        try:
            arr = np.asarray(val, dtype=float)
        except (TypeError, ValueError):
            continue
        if arr.ndim == 1 and arr.size:
            return arr
    return None


def _mask(obj, name, n: int):
    val = getattr(obj, name, None)
    if val is None:
        return None
    arr = np.asarray(val, dtype=bool)
    return arr if arr.shape == (n,) else None


def _surface(obj, name: str) -> dict | None:
    """One lifting surface as plain arrays, or None if ``obj`` is not one.

    Reads BOTH result vocabularies without knowing which solver it is looking
    at: a lifting line names the loading ``Cl_y`` and the effective angle
    ``alpha_eff_y`` (radians, per station), a panel solver names them ``cl``
    and ``alpha_eff`` (radians, per panel). Anything carrying a matching
    ``y``/``c`` pair is a surface; anything else is not.
    """
    y = _arr(obj, "y")
    c = _arr(obj, "c", "chord")
    if y is None or c is None or y.shape != c.shape:
        return None
    cl = _arr(obj, "Cl_y", "cl")
    if cl is not None and cl.shape != y.shape:
        cl = None
    # the effective angle is RADIANS in both vocabularies (llt.LLTResult and
    # the VLM's panel array); degrees is what a polar's own validity range
    # is quoted in, so the conversion happens once, here
    ae = _arr(obj, "alpha_eff_y", "alpha_eff")
    if ae is not None and ae.shape != y.shape:
        ae = None
    return {"name": name, "y": y, "c": c, "cl": cl,
            "alpha_eff_deg": None if ae is None else np.rad2deg(ae),
            "device": _mask(obj, "is_winglet", y.size)}


def _drop_vertical(surf: dict, obj) -> dict:
    """Take the FIN's panels out before anything is scored on them.

    The third surface in the same array, and the one neither splitter knew
    about (api._drop_vertical_panels is the twin of this, on the geometry the
    report draws). A fin is not a lifting surface of the aeroplane — both
    solvers already say so, taking their aero deck off the same mask
    ("the deck is free of charge to the objective") — so scoring it as wing
    charges the wing a spar, a volume and a minimum chord that belong to a
    vertical plank standing at y = 0.

    NOT cosmetic, because these blocks feed the COMPOSITE OBJECTIVE. Measured
    on `tail [free cant]` at its box centre with 12 deg of dihedral, arming
    the lateral deck (``handling_level=3``) moved the wing block from 40 to 52
    stations, its narrowest chord from 0.7504 to 0.6963 m — the FIN's chord —
    and with them ``vol`` 0.914746 -> 0.898506 (-1.78 %) and ``bend`` 5.85118
    -> 5.86476. That is the SEARCH being ranked on it, not a picture.

    And it does not cancel against the frozen normalisation band: the fin is
    sized from b, S and the arm (``fin.size_fin``), and the arm is a design
    variable of this family, so the bias moves with the candidate.
    """
    n = surf["y"].size
    m = _mask(obj, "is_vertical", n)
    if m is None or not m.any():
        return surf
    keep = ~m
    if not keep.any():
        return surf
    return {k: (v[keep] if isinstance(v, np.ndarray) and v.shape == m.shape
                else v)
            for k, v in surf.items()}


def _split(surf: dict, obj) -> list[dict]:
    """Split a multi-surface panel array into one block per surface.

    The nonplanar wing+tail solver and the nonplanar tandem return ONE panel
    array covering both surfaces with an ``is_tail`` / ``is_second`` mask on
    it (api._split_surface_panels draws them the same way). Scoring the
    concatenation would charge one spar the moment of two wings and report
    the pair's narrowest chord as "the wing's", so the mask is honoured here
    exactly as it is when the same arrays are drawn.

    The VERTICAL surface leaves first (:func:`_drop_vertical`) and does not
    come back as a block: it is not one of the aeroplane's lifting surfaces
    and nothing here scores it.

    Every mask is read at the FULL panel length and trimmed by the same keep,
    because ``_mask`` returns None for an array whose shape does not match the
    length asked for — trimming the surface first and then asking for
    ``is_tail`` silently answered "this result has no tail", which put the
    stabiliser's 24 panels back into the wing block. Measured while getting
    this wrong: 40 wing panels became 64 and ``vol`` read 0.648209.
    """
    n = surf["y"].size
    vert = _mask(obj, "is_vertical", n)
    keep = None if vert is None or not vert.any() else ~vert
    masks = [(_mask(obj, attr, n), name)
             for attr, name in (("is_tail", "tail"), ("is_second", "rear"))]
    if keep is not None:
        if not keep.any():
            return [surf]
        surf = _drop_vertical(surf, obj)
        masks = [(None if m is None else m[keep], name) for m, name in masks]
    n = surf["y"].size
    for m, name in masks:
        if m is None or not m.any() or m.all():
            continue
        out = []
        for sel, nm in ((~m, surf["name"]), (m, name)):
            blk = {"name": nm, "y": surf["y"][sel], "c": surf["c"][sel],
                   "cl": None if surf["cl"] is None else surf["cl"][sel],
                   "alpha_eff_deg": (None if surf["alpha_eff_deg"] is None
                                     else surf["alpha_eff_deg"][sel]),
                   "device": (None if surf["device"] is None
                              else surf["device"][sel])}
            out.append(blk)
        return out
    return [surf]


def surfaces_of(raw: dict) -> list[dict]:
    """Every lifting surface in a breakdown, as ``{name, y, c, cl, device}``.

    Mirrors ``api._fill_llt_geometry``'s search order, and for the same
    reason: a solver may return its lifting line directly (LLTResult), as a
    PAIR of surfaces (``front``/``rear`` on a TandemResult — the tail problem
    shares the container), inside one masked panel array (the nonplanar
    wing+tail and tandem), or one level down as an attribute of its own
    result (``HydrofoilResult.foil``). All four are the same physics with
    four containers, so all four are read.
    """
    out: list[dict] = []
    for val in raw.values():
        if val is None or isinstance(val, (str, bytes, bool, int, float,
                                           dict, list, tuple, np.ndarray)):
            continue
        pair = [(getattr(val, a, None), a) for a in ("front", "rear")]
        if all(o is not None for o, _ in pair):
            for obj, name in pair:
                surf = _surface(obj, name)
                if surf is not None:
                    out.extend(_split(surf, obj))
            if out:
                return out
        surf = _surface(val, "wing")
        if surf is not None:
            out.extend(_split(surf, val))
    if out:
        return out
    # nothing spanwise at the top level: unwrap the result objects one level
    # (the hydrofoil keeps its lifting line under .foil)
    for val in raw.values():
        nested = getattr(val, "__dict__", None)
        if not isinstance(nested, dict):
            continue
        for inner in nested.values():
            surf = _surface(inner, "wing")
            if surf is not None:
                out.extend(_split(surf, inner))
        if out:
            return out
    return out


# ---------------------------------------------------------------- metrics


def _first(raw: dict, *keys):
    """First finite float among ``keys`` (else None)."""
    for key in keys:
        val = raw.get(key)
        if isinstance(val, (int, float, np.integer, np.floating)) \
                and not isinstance(val, bool) and np.isfinite(float(val)):
            return float(val)
    return None


def _span_efficiency(raw: dict, surfaces: list[dict]) -> float | None:
    """Span efficiency: the solver's own where it reports one, else derived.

    A lifting line reports ``e`` and the imaged hydrofoil ``e_total``; the
    tandem reports one per wing and the wing+tail solver reports none at all,
    because "the tail's own e" is not the quantity anybody means. So where
    there is no single number, the SYSTEM Oswald factor is derived from the
    numbers every family does report — CL^2 / (pi AR CDi) at the flown
    point, with AR from the span and area the surfaces were solved on. Same
    definition, one surface or two.
    """
    e = _first(raw, "e", "e_total")
    if e is not None:
        return e
    cl = _first(raw, "CL_total", "CL")
    cdi = _first(raw, "CDi_total", "CDi", "CDi_self")
    if cl is None or cdi is None or cdi <= 0.0:
        return None
    ar = _first(raw, "AR")
    if ar is None:
        b = _first(raw, "b", "b_front")
        s = _first(raw, "Sref", "S", "S_lift")
        if b is None or not s:
            # last resort: the MAIN surface's own flown geometry. Its area,
            # not the system's: a wing+tail solver quotes CDi on the wing
            # reference and reports no ``S`` of its own, and adding the tail's
            # area to the denominator would make a bigger stabiliser look
            # like worse span efficiency.
            main = next((s_ for s_ in surfaces if s_["y"].size > 1), None)
            if main is None:
                return None
            b = 2.0 * float(np.max(np.abs(main["y"])))
            s = float(np.trapezoid(main["c"], main["y"]))
            if not b or s <= 0.0:
                return None
        ar = b * b / s
    if not np.isfinite(ar) or ar <= 0.0:
        return None
    return float(cl * cl / (np.pi * ar * cdi))


def _bending_index(surfaces: list[dict]) -> float | None:
    """Worst surface's |INT c_l c y dy| over its semispan [m^3].

    Per unit dynamic pressure, so it is the shape of the load path rather
    than a stress: q cancels out of every comparison inside one box. The
    WORST surface, not the sum — a tandem's two spars are two structures, and
    the one that has to carry the most is what sizes the aircraft.

    Tip devices are excluded: a winglet's panels sit at the tip with a large
    moment arm and a chord that is not the wing's, and charging its lift to
    the wing spar would make every winglet look structurally ruinous when
    ``geometry.py`` already prices its own span.
    """
    worst = None
    for surf in surfaces:
        cl, y, c = surf["cl"], surf["y"], surf["c"]
        if cl is None or y.size < 2:
            continue
        keep = np.ones(y.size, dtype=bool)
        if surf["device"] is not None:
            keep &= ~surf["device"]
        keep &= y >= 0.0                      # ONE semispan (the load is
        #                                       symmetric; both would cancel)
        if keep.sum() < 2:
            continue
        order = np.argsort(y[keep])
        yy = y[keep][order]
        integ = float(np.trapezoid((cl[keep] * c[keep] * y[keep])[order], yy))
        val = abs(integ)
        if np.isfinite(val) and (worst is None or val > worst):
            worst = val
    return worst


def _peak_cl(surfaces: list[dict]) -> float | None:
    """Largest |c_l| flown on any surface, tip devices excluded.

    |c_l| rather than c_l because a stabiliser earns its keep pushing DOWN
    (tail.py flies its section inverted) and a car wing does nothing else:
    the distance from zero is what approaches a section ceiling, whichever
    side of zero the surface works on.
    """
    peak = None
    for surf in surfaces:
        cl = surf["cl"]
        if cl is None or not cl.size:
            continue
        keep = (np.ones(cl.size, dtype=bool) if surf["device"] is None
                else ~surf["device"])
        if not keep.any():
            continue
        val = float(np.max(np.abs(cl[keep])))
        if np.isfinite(val) and (peak is None or val > peak):
            peak = val
    return peak


def _min_chord(surfaces: list[dict]) -> float | None:
    """Narrowest chord flown, tip devices excluded [m]."""
    smallest = None
    for surf in surfaces:
        c = surf["c"]
        keep = (np.ones(c.size, dtype=bool) if surf["device"] is None
                else ~surf["device"])
        if not keep.any():
            continue
        val = float(np.min(c[keep]))
        if np.isfinite(val) and val > 0.0 \
                and (smallest is None or val < smallest):
            smallest = val
    return smallest


#: WHICH polar each surface flies, by the name :func:`surfaces_of` gives it.
#: The attribute names are the solvers' own (``tail.TailProblem.polar_tail``,
#: ``tandem.TandemProblem.polar_rear``, ``hydrofoil.section_polar``), and a
#: surface with no entry of its own falls back to the wing's — which is what
#: those solvers do by default anyway.
_POLAR_ATTRS = {
    "wing": ("polar", "section_polar"),
    "front": ("polar", "section_polar"),
    "rear": ("polar_rear", "polar", "section_polar"),
    "tail": ("polar_tail", "polar", "section_polar"),
}

#: Section area as a fraction of t*c. 0.685 is the NACA 4-digit thickness
#: form's own integral (INT y_t dx / (t c) over the closed section), so a
#: 12% 1 m chord encloses 0.0822 m^2. Cambered or laminar sections differ by
#: a few percent; this is a volume PROXY, and it is the same proxy for every
#: candidate in a box.
SECTION_AREA_COEFF = 0.685

#: Ultimate load factor the structural-mass proxy is sized at, when the
#: problem does not carry its own (``aircraft.AircraftProblem.n_ult``).
#: 1.5 x a 2.5 g limit case — the GA certification pair Raymer's correlation
#: was fitted on.
N_ULT_DEFAULT = 3.75

#: A NACA 4-digit designation states its own thickness: the last two digits
#: ARE t/c in per cent. Matched on the POLAR's name (which comes from the
#: data file it was loaded from), never on a problem name.
_NACA4_NAME = re.compile(r"naca\s*(\d{4})", re.IGNORECASE)


def _tc_from_name(name) -> float | None:
    """t/c from a NACA 4-digit section designation, or None."""
    if not isinstance(name, str):
        return None
    m = _NACA4_NAME.search(name)
    if m is None:
        return None
    tc = int(m.group(1)[2:]) / 100.0
    return tc if tc > 0.0 else None


def _flow_state(raw: dict, prob) -> tuple[float, float] | None:
    """``(rho, V)`` the candidate flew at, or None.

    Three places state it and different families use different ones: the
    breakdown (a free flight state reports what it resolved), the problem
    (the fixed operating point), and the problem's MISSION (the sized
    families keep theirs there).
    """
    rho = _first(raw, "rho")
    v = _first(raw, "V")
    for src in (prob, getattr(prob, "mission", None)):
        if src is None:
            continue
        if rho is None:
            val = getattr(src, "rho", None)
            rho = float(val) if isinstance(val, (int, float)) else None
        if v is None:
            val = getattr(src, "V", None)
            v = float(val) if isinstance(val, (int, float)) else None
    return (rho, v) if rho and v else None


def _polar_of(prob, name: str, tc: float | None = None):
    """The polar the named surface flies, or None.

    A family that SELECTS its section by thickness carries a
    ``polar.PolarFamily`` instead of a table (the t/c design variable is the
    whole point), so the family is asked for the member the design actually
    flew rather than skipped.
    """
    for attr in (*_POLAR_ATTRS.get(name, ("polar",)), "polar_family"):
        pol = getattr(prob, attr, None)
        if pol is None:
            continue
        if hasattr(pol, "at") and not hasattr(pol, "alpha_valid"):
            if tc is None:
                continue
            try:
                return pol.at(float(tc))
            except (ValueError, TypeError):
                continue
        return pol
    return None


def _polar_ceiling(pol) -> tuple[float, float] | None:
    """``(c_l ceiling, alpha ceiling [deg])`` of a section polar, or None.

    The ceiling is the top of the polar's OWN validity range — the highest
    c_l it is willing to answer for. It is a table limit, not a measured
    c_l max, and that is exactly why it is used: ``objective.evaluate_geometry``
    already treats leaving this range as the stall/extrapolation proxy and
    PENALISES the design for it, so scoring against the same line keeps the
    criterion and the feasibility rule telling one story.
    """
    try:
        lo, hi = pol.alpha_valid
        alphas = np.linspace(float(lo), float(hi), 41)
        cl = np.asarray(pol.cl(alphas), dtype=float)
    except Exception:                    # a polar that cannot answer: skip
        return None
    if cl.size == 0 or not np.all(np.isfinite(cl)):
        return None
    i = int(np.argmax(cl))
    return float(cl[i]), float(alphas[i])


def _stall_numbers(raw: dict, surfaces: list[dict], prob
                   ) -> tuple[float | None, float | None]:
    """``(usable CL, incidence margin [deg])`` — the two stall criteria.

    Both come from the SAME critical strip: the one closest to the top of
    its own polar at the flown point. The CL number holds the loading SHAPE
    fixed while scaling it up, which is the standard critical-section
    estimate and an approximation — the twist changes the shape as the
    incidence rises (``stall.wing_clmax`` is the exact affine solve, and it
    needs a per-strip 2-D ceiling and a plain free-air lifting line, which a
    tandem, a tail and every VLM family do not have). Read it as "how much
    lift is left", not as a certified CL max.
    """
    if prob is None:
        return None, None
    cl_ratio = None
    margin = None
    for surf in surfaces:
        pol = _polar_of(prob, surf["name"],
                        _thickness_of(surf["name"], raw, prob, None))
        ceiling = None if pol is None else _polar_ceiling(pol)
        if ceiling is None:
            continue
        cl_ceiling, a_ceiling = ceiling
        keep = (np.ones(surf["y"].size, dtype=bool) if surf["device"] is None
                else ~surf["device"])
        if not keep.any() or cl_ceiling <= 0.0:
            continue
        cl = surf["cl"]
        if cl is not None:
            peak = float(np.max(np.abs(cl[keep])))
            if peak > 0.0:
                ratio = cl_ceiling / peak
                cl_ratio = ratio if cl_ratio is None else min(cl_ratio, ratio)
        ae = surf["alpha_eff_deg"]
        if ae is not None and ae.size:
            left = a_ceiling - float(np.max(ae[keep]))
            margin = left if margin is None else min(margin, left)
    cl_flown = _first(raw, "CL_total", "CL", "CL_target")
    clmax = (None if cl_ratio is None or cl_flown is None
             else abs(cl_flown) * cl_ratio)
    return clmax, margin


def _surface_planform(surf: dict) -> tuple[float, float, float] | None:
    """``(span, area, taper)`` of one surface from the geometry it flew."""
    keep = (np.ones(surf["y"].size, dtype=bool) if surf["device"] is None
            else ~surf["device"])
    y, c = surf["y"][keep], surf["c"][keep]
    if y.size < 2:
        return None
    order = np.argsort(y)
    y, c = y[order], c[order]
    b = float(y[-1] - y[0])
    s = float(np.trapezoid(c, y))
    if b <= 0.0 or s <= 0.0:
        return None
    c_root, c_tip = float(np.max(c)), float(np.min(c))
    taper = c_tip / c_root if c_root > 0.0 else 0.0
    return b, s, max(1e-6, min(1.0, taper))


def _thickness_of(name: str, raw: dict, prob, hints: dict | None) -> float | None:
    """The t/c the named surface flies, or None.

    Ladder, most specific first: an explicit hint (the section stage 2 or 2.5
    pinned, resolved by ``api.section_thickness``), the thickness the
    breakdown itself reports (the families that fly a t/c row report it), the
    problem's own. A second surface with no hint of its own follows the
    wing's, because that is what the solvers do when nothing is chosen
    (``tail.polar_tail`` inverts the WING's section).
    """
    hints = hints or {}
    for key in ((name,) if name in ("wing", "front") else (name, "wing")):
        val = hints.get(key)
        if isinstance(val, (int, float)) and float(val) > 0.0:
            return float(val)
    tc = _first(raw, "tc")
    if tc is None:
        wing = raw.get("wing")
        tc = getattr(wing, "tc", None) if wing is not None else None
        tc = float(tc) if isinstance(tc, (int, float)) else None
    if tc is None and prob is not None:
        val = getattr(prob, "tc", None)
        tc = float(val) if isinstance(val, (int, float)) else None
    if tc is None and prob is not None:
        # last rung: the section's own DESIGNATION. A polar named NACA 2412
        # is 12 % thick by definition of the 4-digit series — this reads the
        # data's own name, not a problem name, and it is the only thickness
        # a fixed-table family ever states.
        tc = _tc_from_name(getattr(_polar_of(prob, name), "name", None))
    return tc if tc and tc > 0.0 else None


def _volume(raw: dict, surfaces: list[dict], prob, hints) -> float | None:
    """Enclosed volume of every lifting surface [m^3]."""
    total = 0.0
    seen = False
    for surf in surfaces:
        tc = _thickness_of(surf["name"], raw, prob, hints)
        plan = _surface_planform(surf)
        if tc is None or plan is None:
            continue
        keep = (np.ones(surf["y"].size, dtype=bool) if surf["device"] is None
                else ~surf["device"])
        y, c = surf["y"][keep], surf["c"][keep]
        order = np.argsort(y)
        total += SECTION_AREA_COEFF * tc * float(
            np.trapezoid(c[order] ** 2, y[order]))
        seen = True
    return float(total) if seen and total > 0.0 else None


def _mass(raw: dict, surfaces: list[dict], prob, hints) -> float | None:
    """Structural mass of the lifting surfaces [kg], Raymer GA correlation.

    The design weight is the one the trim implies (W = CL q S, or whatever
    the sized families already report), and q comes from the flow state the
    candidate flew at — so a wing that grows, thins or unloads sees the
    correlation move for the reason it should.
    """
    from .materials import resolve as _resolve_material
    from .weights import wing_weight_raymer

    state = _flow_state(raw, prob)
    if state is None:
        return None
    rho, v = state
    q = 0.5 * float(rho) * float(v) ** 2
    w_dg = _first(raw, "W_total_N", "W_fixed_N")
    if w_dg is None:
        cl = _first(raw, "CL_total", "CL", "CL_target")
        s_ref = _first(raw, "Sref", "S") or getattr(prob, "S", None)
        if cl is None or not s_ref:
            return None
        w_dg = abs(float(cl)) * q * float(s_ref)
    n_ult = float(getattr(prob, "n_ult", None) or N_ULT_DEFAULT)
    # ...and of the MATERIAL, or the criterion the composite calls "mass"
    # would price a carbon wing as an aluminium one — the one weighting most
    # likely to be turned up by a user who has just chosen a material.
    k_w = _resolve_material(getattr(prob, "material", None)).k_w
    sweep = _first(raw, "sweep_deg") or 0.0
    total, seen = 0.0, False
    for surf in surfaces:
        tc = _thickness_of(surf["name"], raw, prob, hints)
        plan = _surface_planform(surf)
        if tc is None or plan is None:
            continue
        b, s, taper = plan
        try:
            total += wing_weight_raymer(b, s, taper, sweep, tc, n_ult,
                                        float(w_dg), q_Pa=q, k_w=k_w)
        except ValueError:               # outside the correlation's domain
            return None
        seen = True
    return float(total / 9.80665) if seen and total > 0.0 else None


def spiral_refusal(raw: dict) -> str:
    """Why the ``spiral`` criterion refuses this design, or ``""``.

    A MARGIN BOUGHT BY TAKING THE YAW STIFFNESS NEGATIVE IS ARITHMETIC, NOT
    STABILITY. The criterion is ``Cl_beta(Cn_r - t.Cn_p) - Cn_beta(Cl_r -
    t.Cl_p)``, so driving ``Cn_beta`` through zero flips the sign of the
    second product and "converges" the spiral on paper for an aeroplane that
    yaws AWAY from the airflow. MEASURED on `tail [free cant]` at 9 deg of
    dihedral, shrinking the fin (RESULTS_SESSION67_WING_CANT.md):

        fin_volume_coeff   Cn_beta      margin      criterion said
        0.040 (default)   +0.079349   +0.012228    converges
        0.010             -0.000373   +0.004462    converges
        0.002             -0.022511   +0.002473    converges

    ``gui/v4/stages/controls.py``'s fin scan already filters its candidates
    this way and REPORTS the rejected ones; the scored criterion had no such
    guard, so a search could buy the weight by shrinking the fin. Refusing
    is what the composite already does for a criterion it cannot honestly
    score (:func:`composite` returns ``None`` with a reason, and the
    evaluation becomes a refusal) — the design is not scored badly, it is
    not scored at all, and the reason is this sentence.

    Note the guard is on ``Cn_beta`` ALONE and not on the margin's sign: a
    directionally stable design with a divergent spiral is exactly what the
    criterion exists to price, and refusing it would delete the term.
    """
    if _first(raw, "spiral_margin") is None:
        return ""
    cnb = _first(raw, "Cn_beta")
    # ONE definition of "does this aeroplane weathercock", shared with the
    # mode namer — the two used to disagree while both printed "stable".
    from .dynamics import weathercocks

    if cnb is None or weathercocks(cnb):
        return ""
    return (f"the spiral criterion is refused: Cn_beta = {float(cnb):+.6f} "
            f"<= 0, so this aeroplane is DIRECTIONALLY unstable and a "
            f"positive spiral margin on it is arithmetic, not stability")


def _spiral(raw: dict) -> float | None:
    """``min(margin, 0)``, or None where it must not be scored."""
    m = _first(raw, "spiral_margin")
    if m is None or spiral_refusal(raw):
        return None
    return min(float(m), 0.0)


def design_metrics(raw: dict, prob=None, tc: dict | None = None
                   ) -> dict[str, float | None]:
    """Every criterion of ONE evaluated design, from its breakdown.

    Returns every criterion key, with ``None`` where this family does not
    report what the criterion needs (a solver that exports no spanwise
    loading has no peak c_l and no bending index). ``None`` is never a bad
    score — a criterion the family cannot measure is REFUSED as a weighted
    objective rather than scored as zero (:func:`composite`).

    ``prob`` is the built problem, and it is what the stall criteria need: a
    surface's section POLAR is not in the breakdown, it is on the problem
    (:data:`_POLAR_ATTRS`). ``tc`` is ``{surface: thickness}`` for the
    volume and mass proxies — the section the pipeline pinned, which the
    breakdown of a fixed-table family does not carry either. Both optional:
    without them those four criteria are simply not measured.

    Costs nothing: every number here is already in the dict the family's own
    ``evaluate`` returned, or on the problem that produced it.
    """
    surfaces = surfaces_of(raw)
    clmax, astall = _stall_numbers(raw, surfaces, prob)
    return {
        "lod": _first(raw, "score", "LoD", "f"),
        # ...and NOT a fallback to another key. Only the families with a
        # second surface trim, so only they have a neutral point to measure
        # a margin against; everyone else reports None, which the composite
        # REFUSES to weight rather than scoring as zero.
        # SATURATED at STAB_TARGET_MAC: past a normal flying margin more is
        # not better, and an unbounded term makes the weight bang-bang.
        "stab": (None if _first(raw, "SM") is None
                 else min(float(_first(raw, "SM")), STAB_TARGET_MAC)),
        # DIVERGENCE ONLY: clipped at the physical boundary so a weight
        # cannot ratchet the design into the dihedral bound buying margin
        # nobody asked for. None (not zero) where no lateral deck was run
        # — a family that did not measure it must not be scored on it —
        # and None where the aeroplane does not weathercock (_spiral).
        "spiral": _spiral(raw),
        "cl_peak": _peak_cl(surfaces),
        "espan": _span_efficiency(raw, surfaces),
        "bend": _bending_index(surfaces),
        "build": _min_chord(surfaces),
        "alpha": _first(raw, "alpha_deg"),
        "clmax": clmax,
        "astall": astall,
        "vol": _volume(raw, surfaces, prob, tc),
        "mass": _mass(raw, surfaces, prob, tc),
    }


# ------------------------------------------------------ frozen normalisation


def reference_sha(problem: str, box: dict, samples: int, seed: int,
                  bounds: dict) -> str:
    """Fingerprint of a band: the POPULATION it was measured over (problem,
    design box, sample count, sweep seed) AND the bands themselves.

    The bands are in the blob for the reason ``airfoil_select.reference_sha``
    puts them there: hashing only the provenance would let an edited band
    pass the integrity check silently, and every composite number in the run
    is computed from those two floats.
    """
    blob = json.dumps(
        {"problem": str(problem),
         "box": {str(k): [float(v[0]), float(v[1])]
                 for k, v in sorted(box.items())},
         "samples": int(samples), "seed": int(seed),
         "bounds": {k: [float(bounds[k][0]), float(bounds[k][1])]
                    for k in sorted(bounds)}},
        sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class WingScoreReference:
    """Frozen per-criterion (lo, hi) band, measured over a design box.

    Carries a SUBSET of :data:`CRITERIA` — a family that reports no spanwise
    loading has no band for ``bend`` — and :func:`composite` refuses a weight
    on a criterion the band does not cover, rather than scoring it zero.
    """

    bounds: dict[str, tuple[float, float]]
    sha: str = ""
    n_samples: int = 0
    n_feasible: int = 0
    problem: str = ""
    seed: int = 0
    #: the measured POPULATION itself — one metrics dict per feasible box
    #: sample. Display-only (nothing in J reads it), and deliberately outside
    #: the SHA for that reason: it is what lets a shell answer "how good is
    #: this design compared with the box it came from?" under whatever
    #: weights the user has now, which a pair of percentiles cannot.
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
        """JSON-safe form — what travels on a RunConfig's flags."""
        return {"version": REFERENCE_VERSION,
                "bounds": {k: [float(v[0]), float(v[1])]
                           for k, v in self.bounds.items()},
                "sha": self.sha, "n_samples": int(self.n_samples),
                "n_feasible": int(self.n_feasible),
                "problem": self.problem, "seed": int(self.seed),
                "samples": [{k: (None if v is None else float(v))
                             for k, v in row.items()}
                            for row in self.samples]}


def reference_from_payload(payload: dict) -> WingScoreReference:
    """Read a stored band, RAISING on any inconsistency.

    Deliberately intolerant, exactly as ``load_screen_reference`` is: a band
    that silently loads stale changes every composite number the optimiser
    saw, with no visible symptom.
    """
    if not isinstance(payload, dict):
        raise ValueError("wing score reference: not an object")
    for field in ("version", "bounds", "sha"):
        if field not in payload:
            raise ValueError(f"wing score reference: missing '{field}'")
    if int(payload["version"]) != REFERENCE_VERSION:
        raise ValueError(
            f"wing score reference: format version {payload['version']} "
            f"!= {REFERENCE_VERSION}")
    raw = payload["bounds"]
    if not isinstance(raw, dict) or not raw:
        raise ValueError("wing score reference: 'bounds' is not a non-empty "
                         "object")
    bounds = {}
    for key, pair in raw.items():
        if (not isinstance(pair, (list, tuple)) or len(pair) != 2
                or not all(isinstance(v, (int, float)) for v in pair)):
            raise ValueError(
                f"wing score reference: bounds['{key}'] must be a 2-element "
                f"numeric [lo, hi], got {pair!r}")
        bounds[str(key)] = (float(pair[0]), float(pair[1]))
    rows = payload.get("samples") or ()
    samples = tuple({str(k): (None if v is None else float(v))
                     for k, v in row.items() if str(k) in CRITERIA}
                    for row in rows if isinstance(row, dict))
    return WingScoreReference(
        bounds=bounds, sha=str(payload["sha"]),
        n_samples=int(payload.get("n_samples", 0)),
        n_feasible=int(payload.get("n_feasible", 0)),
        problem=str(payload.get("problem", "")),
        seed=int(payload.get("seed", 0)), samples=samples)


def _sample_box(bounds: np.ndarray, n: int, seed: int) -> np.ndarray:
    """``n`` quasi-random points in the box, plus its CENTRE as sample 0.

    Sobol where scipy has it (a low-discrepancy sweep covers a 10-D box far
    more evenly than uniform draws at n = 32), uniform otherwise. The centre
    is always sample 0 because it is the design the seed-vs-optimised block
    reports against: measuring the band without it would leave the baseline
    outside its own population.
    """
    bounds = np.asarray(bounds, dtype=float)
    d = bounds.shape[0]
    lo, hi = bounds[:, 0], bounds[:, 1]
    centre = 0.5 * (lo + hi)
    n_rest = max(0, int(n) - 1)
    unit = None
    if n_rest:
        try:
            import warnings

            from scipy.stats import qmc
            with warnings.catch_warnings():
                # the centre takes one of the n samples, so the sequence
                # length is n-1 and Sobol says so; the balance property it
                # warns about is not what a p5/p95 band needs
                warnings.simplefilter("ignore", UserWarning)
                unit = qmc.Sobol(d, scramble=True,
                                 seed=int(seed)).random(n_rest)
        except (ImportError, ValueError):
            unit = np.random.default_rng(int(seed)).random((n_rest, d))
    pts = [centre[None, :]]
    if unit is not None:
        pts.append(lo + unit * (hi - lo))
    return np.vstack(pts)


def sample_reference(evaluate, bounds: np.ndarray, *, problem: str = "",
                     n: int = REFERENCE_SAMPLES, seed: int = 0,
                     labels: tuple = (), p_lo: float = REF_P_LO,
                     p_hi: float = REF_P_HI, metrics=None,
                     progress=None, screen: bool = False,
                     pool: int = 32) -> WingScoreReference:
    """Measure the frozen band by sweeping the design box ``n`` times.

    ``evaluate`` is the family's own ``_BuiltProblem.evaluate``; the sweep
    keeps the FEASIBLE samples only (an infeasible design has no criteria —
    its breakdown carries a reason, not a loading) and takes the (p_lo, p_hi)
    percentile pair per criterion over them.

    Raises ValueError when fewer than :data:`MIN_REFERENCE_SAMPLES` samples
    come back feasible, or when no criterion survives with a non-degenerate
    band: a box where nothing flies cannot be normalised against, and saying
    so beats returning a band that scores every design 50.

    ``metrics`` is the extractor — pass the SAME closure the run will score
    with (``design_metrics`` bound to the problem and its thickness hints),
    or the band would cover a different set of criteria than the objective
    reads. ``progress(i, n)`` is called after each sample so a shell can show
    the sweep — this is the one part of the composite that costs evaluations.

    ``screen`` keeps drawing until ``n`` samples have FLOWN, rather than taking
    whatever flies out of ``n`` draws. It exists because the gate above turned
    into a refusal of boxes the SEARCH solves perfectly well: on the family a
    user reported (``tail + winglet (span-capped) … + free planform + free
    chord law``, whose mission states a wing-loading ceiling), 92.6 % of the
    box is refused before any solver runs, so about 2 of 32 draws fly and the
    band cannot be measured — while a screened search on the same box finds a
    design in 39 of 72 evaluations. The population being estimated is the same
    one either way, "the designs this box produces that fly"; the screen just
    stops estimating it from two samples. It is off by default because a band
    is hashed into a run's provenance and every stored study measured its own
    the unscreened way; the V3 shell asks for it. ``pool`` caps the draws per
    wanted sample, so the cost is bounded even on a box that flies rarely.
    """
    metrics = metrics or design_metrics
    pts = _sample_box(bounds, n if not screen else n * max(1, int(pool)), seed)
    rows: list[dict] = []
    n_drawn = 0
    for i, x in enumerate(pts):
        if screen and len(rows) >= n:
            break
        try:
            raw = evaluate(np.asarray(x, dtype=float))
        except Exception:                     # a box corner the family hates
            raw = {"feasible": False}
        n_drawn += 1
        if isinstance(raw, dict) and raw.get("feasible"):
            rows.append(metrics(raw))
        if progress is not None:
            progress(min(len(rows) + 1, n) if screen else i + 1, n)
    if len(rows) < MIN_REFERENCE_SAMPLES:
        raise ValueError(
            f"only {len(rows)} of {n_drawn} box samples flew — too few to "
            f"measure a normalisation band (need {MIN_REFERENCE_SAMPLES}). "
            "Narrow the design box to a region that flies, then measure it "
            "again.")

    bounds_out: dict[str, tuple[float, float]] = {}
    for key in CRITERIA:
        vals = np.array([r[key] for r in rows if r[key] is not None],
                        dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size < MIN_REFERENCE_SAMPLES:
            continue                          # this family cannot measure it
        lo = float(np.percentile(vals, p_lo))
        hi = float(np.percentile(vals, p_hi))
        if hi <= lo:
            continue                          # constant in this box
        bounds_out[key] = (lo, hi)
    if not bounds_out:
        raise ValueError(
            "no criterion varied across the box sample, so there is nothing "
            "to normalise against — widen the design box.")

    box = {str(lbl): (float(bounds[i][0]), float(bounds[i][1]))
           for i, lbl in enumerate(labels)} if labels else \
          {str(i): (float(b[0]), float(b[1]))
           for i, b in enumerate(np.asarray(bounds, dtype=float))}
    # the DRAWS, not the pool: a screened sweep stops as soon as it has its n
    # flyable samples, so len(pts) is a ceiling nobody paid and quoting it
    # would put a number in the provenance hash that no run performed
    sha = reference_sha(problem, box, n_drawn, seed, bounds_out)
    return WingScoreReference(bounds=bounds_out, sha=sha, n_samples=n_drawn,
                              n_feasible=len(rows), problem=problem,
                              seed=int(seed), samples=tuple(rows))


# ---------------------------------------------------------------- the score


def sub_scores(metrics: dict, reference: WingScoreReference) -> dict:
    """0-100 sub-score per criterion the band covers (higher = better).

    NOT clipped, for ``score_candidates``' reason: a design outside the band
    scores past its end, and clipping would flatten the objective exactly
    where the search is winning — at the point where a candidate beats
    everything the box sample contained.
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


def composite(metrics: dict, reference: WingScoreReference,
              weights: WingScoreWeights) -> tuple[float | None, dict, str]:
    """``(J, sub-scores, reason)`` — the weighted composite of one design.

    ``J = sum_k w_k s_k`` over the weighted criteria, each normalised against
    the FROZEN band. A criterion with a weight but no value (this family does
    not report it) or no band (it did not vary across the box sample) makes J
    ``None`` with a reason naming it — never a silent zero, which would read
    as "scored badly" for something that was never measured.
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


def population_rank(value: float | None, reference: WingScoreReference,
                    weights: WingScoreWeights) -> float | None:
    """Fraction of the BOX SAMPLE this J beats, or None if unanswerable.

    The honest answer to "did the search do anything?". Comparing the
    optimised design with the box centre alone is one draw against one draw;
    against the whole measured population it is a rank, and the population is
    already on disk because the band came from it. Scored under the weights
    in force NOW, which is why the sample metrics are stored rather than a
    pair of percentiles.
    """
    if value is None or not reference.samples:
        return None
    scored = [composite(row, reference, weights)[0]
              for row in reference.samples]
    scored = [s for s in scored if s is not None]
    if not scored:
        return None
    beaten = sum(1 for s in scored if float(value) > s)
    return float(beaten) / float(len(scored))


def composite_evaluation(raw: dict, reference: WingScoreReference,
                         weights: WingScoreWeights, metrics=None) -> dict:
    """One composite evaluation with its working — the family's breakdown
    extended with ``metrics``, ``scores``, ``composite``.

    ``score`` / ``f`` ARE THE OBJECTIVE THE OPTIMISER SAW: J where it
    resolved, ``PENALTY`` where it did not. The family's own objective stays
    available under ``f_lod``, because a shell that reads ``score`` to draw
    "best objective" must not print an L/D under a run labelled composite —
    the same contract ``airfoil_select.composite_evaluation`` keeps for
    ``f_cd``.
    """
    out = dict(raw)
    # ...and the axis's name changes with it, on EVERY exit. On a
    # SIZED family ``raw`` arrived carrying score_units="payload L/D"
    # (sizing.SizedState.report); leaving it would label a 0-100
    # index — or a PENALTY sentinel — as a W_fixed/D. Set once, here,
    # because two of the three returns below are early ones.
    out["score_units"] = COMPOSITE_UNITS
    # "LoD" FIRST, not "score". On a sized family raw["score"] is already
    # W_fixed/D (sizing.SizedState.payload_lod), so reading score first
    # made the one key whose name promises an L/D hand back a payload
    # L/D: measured 14.883534 where raw["LoD"] was 47.506273.
    out["f_lod"] = _first(raw, "LoD", "score", "f")
    out["metrics"], out["scores"], out["composite"] = {}, {}, None
    if not raw.get("feasible"):
        out["score"] = out["f"] = PENALTY
        return out
    measured = (metrics or design_metrics)(raw)
    out["metrics"] = measured
    j, scores, reason = composite(measured, reference, weights)
    out["scores"] = scores
    if j is None:
        out["score"] = out["f"] = PENALTY
        # ...and the reason says WHICH of the two ways a criterion goes
        # missing. ``composite`` sees only the metrics, where a refused
        # criterion and an unmeasured one are both None; ``raw`` is what
        # tells them apart, and this is the one place that holds both.
        why = spiral_refusal(raw)
        out["composite_reason"] = (f"{reason} — {why}"
                                   if why and "spiral" in reason else reason)
        return out
    out["composite"] = j
    out["score"] = out["f"] = j
    return out


def composite_objective(x, evaluate, reference: WingScoreReference,
                        weights: WingScoreWeights, n_constraints: int = 1,
                        constrained: bool = True, metrics=None):
    """The composite as a harness callable: ``(J, margins)``, or ``J`` alone.

    Return contract, identical to every ``fg_*`` in the package
    (objective.fg's docstring is the reference):

    * solver failure -> ``(PENALTY, [G_FAIL] * n_constraints)``
    * feasible -> ``(J, g)`` with the family's OWN margins, untouched, so the
      composite arms and the L/D controls share one feasible region;
    * feasible but unscoreable (a weighted criterion this family does not
      report) -> ``(PENALTY, g)`` with the true margins.

    An UNCONSTRAINED family returns the bare float, because that is what its
    harness expects — the composite changes the objective, never the channel.
    """
    raw = evaluate(np.asarray(x, dtype=float))
    out = composite_evaluation(raw, reference, weights, metrics=metrics)
    j = out["composite"]
    if not constrained:
        return float(j) if j is not None else PENALTY
    if not raw.get("feasible"):
        return PENALTY, np.full(int(n_constraints), G_FAIL)
    g = raw.get("g")
    if g is None:
        # a family that declares constraints must report them; scoring one
        # that does not would invent a margin the optimiser then trusts
        return PENALTY, np.full(int(n_constraints), G_FAIL)
    g = np.asarray(g, dtype=float)
    return (float(j) if j is not None else PENALTY), g


def make_composite_fg(evaluate, reference: WingScoreReference,
                      weights: WingScoreWeights, n_constraints: int = 1,
                      constrained: bool = True, metrics=None):
    """Bind :func:`composite_objective` to one built problem's ``evaluate``."""

    def fg(x):
        return composite_objective(x, evaluate, reference, weights,
                                   n_constraints=n_constraints,
                                   constrained=constrained, metrics=metrics)

    return fg


__all__ = [
    "CRITERIA", "CRITERION", "G_FAIL", "HIGHER_BETTER", "LOWER_BETTER",
    "MIN_REFERENCE_SAMPLES", "PENALTY", "PRESETS", "REFERENCE_SAMPLES",
    "REFERENCE_VERSION", "REF_P_HI", "REF_P_LO", "WING_CRITERIA",
    "Criterion", "WingScoreReference", "WingScoreWeights", "composite",
    "composite_evaluation", "composite_objective", "design_metrics",
    "make_composite_fg", "population_rank", "reference_from_payload",
    "reference_sha",
    "sample_reference", "spiral_refusal", "sub_scores", "surfaces_of",
    "weights_of",
]
