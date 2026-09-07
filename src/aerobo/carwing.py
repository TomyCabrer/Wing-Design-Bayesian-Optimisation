"""Race-car rear wing: inverted lifting surface near the track, with
endplates and a MOUNT layout (centre pylons vs endplate-mounted).

Frame (the trick that keeps every sign honest)
----------------------------------------------
The whole problem is solved in a MIRRORED frame: the model is the car
flipped upside down. In that frame

    model lift (+z)         IS the car's DOWNFORCE
    the track               is a rigid wall ABOVE the wing, at the ride
                            height, i.e. vlm.ImagePlane(z=+h, "ground")
    an endplate hanging     points +z in the model, i.e. cant = +90 deg
    DOWN towards the track

so the section polar, the twist convention and the image sign are all used
exactly as everywhere else in the package — nothing is inverted by hand,
which is where sign errors in downforce codes come from. The rigid-wall
image is side-agnostic (w = 0 on the plane either way), so the same
ImagePlane serves. Reported quantities are named for the CAR: ``CZ`` is the
downforce coefficient, positive downwards.

What is modelled
----------------
* Nonplanar imaged VLM (vlm.py): wing + endplates + the track's image.
  Ground proximity raises downforce and cuts induced drag exactly as it
  raises lift for an aircraft wing in ground effect.
* Endplates as tip devices of free height IN METRES, pointing at the track
  (a plate reaches the car or it does not; that is a length, not a fraction
  of a span which is itself a design variable here).
* Section profile drag from the polar family, integrated over wing AND
  endplate strips (same treatment as objective.evaluate_winglet).
* MOUNT layout — the point of the module (see below).

The mount question
------------------
Where the wing is attached changes three things this code can compute:

1. STRUCTURE. The spanwise load is the same, but the beam is not:
   ``centre`` mounts (twin pylon / swan neck) make each half a CANTILEVER
   off the centreline, while ``ends`` mounts (load taken out through the
   endplates) make the wing a beam SIMPLY SUPPORTED at its tips. Same load,
   different bending-moment distribution and — the part that actually bites
   on a car — different deflection under load, which moves the wing's
   incidence and ride height at speed. Both are computed from the solved
   circulation (:func:`bending_moment`, :func:`deflection_index`).
2. PARASITE DRAG. Centre pylons are extra wetted area in the flow; an
   endplate-mounted wing carries its load through plates that are there
   anyway. Charged with the same reduced-order strut model as the
   hydrofoil's mast.
3. JUNCTION (interference) DRAG. A centre mount ADDS two wing/pylon
   corners; an endplate mount adds none beyond the plates already present.
   Priced with junction.py, which is a correlation plus a labelled fillet
   credit — see that module.

The mount as a continuum (carmount.py), opt-in
----------------------------------------------
The two-valued flag above answers two independent questions with one word —
WHERE the load leaves the wing (a spanwise station) and WHAT carries it from
there (pylons to a deck, or the plates). carmount.py takes them apart, and
:attr:`CarWingProblem.mount_spec` adopts it. Setting it changes five things,
each of which the flag could not express:

* the SUPPORT STATION drives the beam (``carmount.beam_moment`` /
  ``beam_deflection_index``, which contain both published ends to the bit);
* the PYLON LENGTH comes off the car's attachment DECK, not off the track
  (:func:`carmount.pylon_length_m`) — see the deck-height note below;
* the PYLON DRAG can be a wetted-area build-up at the pylon's own Reynolds
  number instead of the crude fixed-Cf plate (``MountSpec.pylon_drag_model``,
  whose default is the crude one so adoption moves nothing it need not);
* the JUNCTION is priced at the chord where the pylon actually meets the
  wing, not at the root chord;
* the TORSIONAL WINDUP is applied. A mount that is not at the aerodynamic
  centre winds the wing up (or down) under its own load, which changes the
  incidence, which changes the load — so it needs a RE-SOLVE, and this
  module runs the fixed point (solve, wind up, re-solve) to a stated
  tolerance and iteration cap. Non-convergence is DIVERGENCE and is refused
  in contract with the divergence margin attached, never raised.

  The fixed point also MEASURES what carmount's single-mode reduction only
  predicts. The map's feedback gain (``windup_contraction``, the ratio of
  successive corrections) is the aeroelastic amplification's own eigenvalue,
  and carmount's ``q/q_div`` is the single-DOF estimate of it. They are not
  the same number: swept over GJ on the published wing the measured gain is
  0.528 to 0.575 times ``q/q_div``, so the wing is still converging — still
  statically stable — at up to about 1.8x the single-mode divergence
  pressure. That is the direction carmount.divergence_q's own docstring
  declares ("the 2-D lift slope ... puts q_div LOW: CONSERVATIVE"), now with
  a size on it: a finite wing in ground effect does not deliver the
  strip-theory response. The margin is reported as carmount computes it —
  conservative — and the measured gain is reported beside it.

  One more measured fact about this family, and it is not a detail: on the
  published wing the windup runs the section out of its POLAR before the
  map stops contracting. At 8 deg incidence the refusal at GJ 24 N m^2 is
  "effective AoA outside polar validity", not divergence; the divergence
  refusal is reachable at 1 deg, where GJ 13 gives a gain of 1.047. A wing
  that winds up towards stall is refused for stalling, which is the honest
  reason.

  The elastic twist phi(y) is distributed and this family's twist law is
  LINEAR, so the windup enters as the chord-weighted least-squares
  projection of phi onto that linear law (:func:`_twist_projection`). The
  projection preserves the added lift exactly — the residual of a weighted
  least-squares fit is orthogonal to the constant basis function, so the
  chord-weighted mean twist is reproduced — and is exact whenever phi is
  itself linear. What it does not reproduce is the CURVATURE of phi, and
  therefore the exact spanwise redistribution of the added load; the
  residual is reported as ``windup_fit_residual_deg``.

The ride height, the deck, and the mission (all opt-in)
-------------------------------------------------------
* DECK HEIGHT. A pylon runs from the wing down to the car's attachment deck,
  a distance ``ride - deck``; endplate.py already measures the SAME piece of
  car that way (its ``reach_m``). This module charged its pylons over the
  whole ride height, i.e. down to the track, through a gap where there is no
  pylon. :attr:`CarWingProblem.deck_height_m` is the correction, off by
  default so no published number moves, and it brings a PYLON REACH margin
  with it — the centre mount's twin of endplate.py's reach constraint for
  the ends.
* THE MISSION. ``CD_budget`` is an allowance, not a task. With
  :attr:`CarWingProblem.track_spec` (and a :class:`cartrack.CarSpec`) the
  task becomes a LAP: the wing is evaluated at the speeds the lap actually
  spends its time at, each at its own Reynolds number and — under a heave
  law — its own ride height, and the lap time follows from the resulting
  ``CZ*S`` and ``CD*S``. ``objective="laptime"`` maximises minus that time.
* RIDE-HEIGHT SENSITIVITY. dCZ/dh by central difference, because a rear
  wing's ride height is not a number the car holds still: a large |dCZ/dh|
  is a knife-edge aero platform whose downforce (and therefore whose
  balance) moves with every bump, kerb and braking event. It is an OPT-IN
  (:attr:`CarWingProblem.dczdh_report`, and implied by a
  :attr:`CarWingProblem.dczdh_max_per_m` ceiling) and it is off by default,
  because it is a DIAGNOSTIC and it costs two more builds and solves —
  measured at the published design point, 1.55 ms per evaluation off
  against 4.58 ms on, a factor of 2.97, paid inside every optimiser's inner
  loop. When it is off the four ``dCZ_dh_*`` keys are ABSENT from the
  breakdown rather than present and None: a reader that asks for a number
  nobody measured gets a ``KeyError``, not a silence it can mistake for a
  measurement.

What is NOT modelled (and matters in reality)
---------------------------------------------
* The real reason for a swan-neck mount: attaching on the PRESSURE side
  leaves the suction surface undisturbed, while an underslung pylon sits in
  the suction peak and costs local downforce. That is a chordwise,
  viscous, geometry-specific effect — no lifting-surface method or
  reduced-order correlation gives it honestly, so it is left out rather
  than invented. It enters as a per-strip loss through
  ``carmount.MountSpec.suction_loss``, whose default is exactly zero and
  whose magnitude is the user's to supply from CFD or tunnel data. What
  this module does with it is knock the strip LOADS down and take the
  span-integrated loss off CZ; it does NOT re-solve the circulation with
  the loss in place, so the wake is that of the unknocked wing and the
  induced drag is not re-derived. Bias direction: the lift loss is
  carmount.lift_knockdown's own upper bound and the drag is unchanged, so
  the reported CZ and CZ/CD are both CONSERVATIVE.
* The elastic twist is flown as a linear twist increment (above), so the
  curvature of phi(y) is not flown; and the wing's bending deflection is
  computed but NOT fed back into the ride height, which a real car's mount
  does at speed.
* Diffuser/floor interaction, wheel wakes, flap systems, DRS, yaw,
  compressibility (all irrelevant at the modelled speeds).
* The track is a FIXED wall: no moving-belt boundary layer, and no
  rotation of the flow it would carry.
* The lap is cartrack.py's quasi-steady point mass — read that module's own
  omissions, all of which are OPTIMISTIC. One more is added here: the
  representative speeds are sampled from a lap flown on the wing's
  coefficients AT :attr:`CarWingProblem.V`, and the lap is not re-sampled
  after the wing has been re-evaluated at them. Measured on cartrack's
  reference car and lap: a second pass moves the lap time by EXACTLY 0.0 s
  with the shipped polars (this family's coefficients are speed-independent
  to round-off — one ulp, because the circulation is solved at the speed and
  the coefficient divides it back out — unless :attr:`flown_reynolds` is on
  or the pylons are charged as a build-up) and by 6.7e-7 s with
  ``flown_reynolds`` on, against a 61.3 s lap. One pass, measured, rather
  than one pass assumed.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from . import carmount, geometry, junction
from .vlm import VLM, ImagePlane

# cartrack.py imports THIS module (for RHO_AIR), so the mission is imported
# where it is used and never at module scope. carmount.py deliberately does
# not import carwing, so it can come in at the top.

RHO_AIR = 1.225           # kg/m^3
MU_AIR = 1.789e-5         # Pa s

PENALTY = -100.0          # objective.py contract (solver failure only)
G_FAIL = -1.0             # finite infeasible margin on solver failure

#: default span band of a car rear wing [m]. A LENGTH band, not a fraction of
#: the family's own span the way sizing.span_bounds defaults: a rear wing is
#: as wide as the regulations or the bodywork let it be, and 0.6-4x of a
#: touring-car 1.6 m would offer a 6.4 m wing on a car. The published 1.6 m
#: sits inside it, so the box never picks the answer, and at the published
#: 0.4 m^2 reference area the band spans aspect ratios 3.6-10 — inside the
#: range a one-chordwise-panel VLM describes honestly (sizing.AR_LIMITS).
SPAN_BOUNDS_M = (1.2, 2.0)

#: index of the RIDE HEIGHT in the family block of the design vector. Named
#: once because :func:`_bounds_reason` reads it and the unpacking below states
#: it positionally; two literals for one slot is how they come to disagree.
_RIDE_ROW = 5


def span_row(span_bounds_m=None) -> np.ndarray:
    """(1, 2) design-box row for the SPAN, in metres.

    Shared by both car families (carwing.py and endplate.py) so "how wide may
    this wing be" is asked once, in one unit, with one validation.
    """
    lo, hi = (SPAN_BOUNDS_M if span_bounds_m is None
              else (float(span_bounds_m[0]), float(span_bounds_m[1])))
    if not (float(lo) > 0.0 and float(hi) > float(lo)):
        raise ValueError(
            f"span bounds must satisfy 0 < min < max, got ({lo}, {hi})")
    return np.array([[float(lo), float(hi)]], dtype=float)


#: default reference-AREA band of a car rear wing [m^2], used when the area is
#: a design variable. DERIVED, not chosen: the honest aspect-ratio band for
#: these one-chordwise-panel solvers is ``sizing.AR_LIMITS`` = (3, 40), and
#: AR = b^2/S, so over the default span band :data:`SPAN_BOUNDS_M` = (1.2, 2.0)
#: every corner stays inside it iff
#:
#:     S <= b_min^2 / AR_min = 1.44 / 3  = 0.48 m^2
#:     S >= b_max^2 / AR_max = 4.00 / 40 = 0.10 m^2
#:
#: The published 0.4 m^2 sits inside, so the box never picks the answer. A
#: caller who widens the span band past its default can still reach a corner
#: outside AR_LIMITS — that is refused per candidate in ``evaluate_car_wing``
#: (``sizing.check_ar``) rather than papered over here, because the limit
#: belongs to the SOLVER and the band is only a convenient default.
AREA_BOUNDS_M2 = (0.10, 0.48)

#: what a car wing run MAXIMISES. Three genuinely different questions, and the
#: distinction only becomes visible once the reference area can move:
#:
#:   ``cz``          the downforce COEFFICIENT (the published objective).
#:   ``downforce``   the downforce FORCE F_z = q S CZ, in newtons.
#:   ``efficiency``  CZ/CD, which is also F_z/D — the area cancels.
#:
#: At a FIXED area all three rank the candidates the same way up to the drag
#: budget, which is why one objective sufficed. At a free area they do not,
#: because a coefficient is referenced to an area that is now moving: measured
#: at a fixed design vector, CZ rises 0.528 -> 0.739 and CZ/CD rises 18.1 ->
#: 35.5 as S falls 0.90 -> 0.15 m^2, while the downforce actually produced
#: FALLS 881 -> 205 N. So ``cz`` against a free area is not a hard problem, it
#: is a mis-stated one — it is maximised by shrinking the reference area, and
#: is refused rather than run (:meth:`CarWingProblem.__post_init__`).
#:
#: ``efficiency`` is NOT degenerate in the same way, and it was worth checking
#: rather than assuming: over the default area band at the box centre it peaks
#: at S ~ 0.14 m^2 (35.5) and falls off both sides, because the induced-drag
#: gain from a higher aspect ratio eventually loses to the profile and strut
#: terms. It rides no bound — it just answers a question most users do not
#: mean, which is why :attr:`CarWingProblem.downforce_min_n` exists.
#:
#: ``laptime`` is the fourth, and it is the only one that is a TASK rather
#: than a figure of merit: it needs a car and a circuit
#: (:attr:`CarWingProblem.track_spec`) and is refused by name without one.
#: The score is ``-lap_time_s``. The transform has to be strictly DECREASING
#: (the harness maximises) and the negation is the only decreasing transform
#: that leaves the units and the increments alone: a score difference of 0.1
#: IS a tenth of a second, at every lap time. ``1/t`` would also rank
#: correctly and would compress the slow end — a tenth of a second would be
#: worth twice as much on a 40 s lap as on a 60 s one, which is a weighting
#: nobody asked for.
#:
#: ONE TRAP, stated because the sentinel is older than this objective:
#: :data:`PENALTY` is -100.0, and a lap longer than 100 s therefore scores
#: WORSE than a refused design. Nothing in this module can fix that — the
#: sentinel is objective.py's contract and it is not scale-aware — so a
#: circuit whose lap exceeds 100 s must be run under the shipped scale-aware
#: refusal policy (``bo_refusal``, which imputes a refusal from the observed
#: scores) rather than the raw sentinel. cartrack's own reference lap is
#: 61.3 s, so the shipped default is not near it.
#: ``cd`` and ``drag`` are the fifth and sixth, and they exist for ONE
#: formulation: maximise nothing and minimise the drag, subject to a
#: :attr:`CarWingProblem.downforce_min_n` floor. That is the epsilon-constraint
#: statement of the same trade the first three scalarise, and it is measurably
#: the better way to ask it. Over a 14 803-design grid of the endplate family
#: whose (CZ, CD) Pareto front has 618 points, sweeping each formulation's own
#: knob reaches:
#:
#: =========================================  ====================
#: formulation                                front points reached
#: =========================================  ====================
#: ``w*CZ + (1-w)*efficiency``                2.6 %
#: ``CZ - lambda*CD``                         6.3 %
#: augmented Tchebycheff                      54.5 %
#: ``efficiency`` s.t. ``downforce_min_n``    38.5 %
#: ``drag`` s.t. ``downforce_min_n``          **98.5 %**
#: =========================================  ====================
#:
#: The front is strongly concave, and a linear weight cannot see a concave
#: region however it is normalised — the biggest gap a lambda sweep jumps is
#: 13.9 % of the front in one step. An epsilon-constraint does not care.
#:
#: WHY MINIMISING DRAG BEATS MAXIMISING EFFICIENCY UNDER THE SAME FLOOR, which
#: is not obvious and is the reason both are offered: maximising a RATIO under
#: a floor lets the answer climb above the floor chasing a better quotient. At
#: a 200 N floor the efficiency winner makes 297.6 N for 9.0 N of drag where
#: 201.5 N for 6.5 N was available — 49 % more downforce than was asked for,
#: bought with 38 % more drag, and reported as the better wing. Once the floor
#: states the downforce, every remaining freedom belongs to the drag.
#:
#: DEGENERATE WITHOUT A FLOOR, and NOT refused for it. Over 3000 draws of the
#: endplate box the minimum-drag design is CZ 0.058 / 43 N — a wing that is
#: not a wing. That is exactly the status ``efficiency`` already has (see
#: above): a question most users do not mean, answered honestly, with
#: ``downforce_min_n`` as the fix. Refusing a computable objective because one
#: way of asking it is silly would be a ban, and this package does not ship
#: bans it can state as defaults.
#:
#: ``cd`` is refused at a FREE AREA for the reason ``cz`` is, and it is the
#: same reason read backwards: a drag COEFFICIENT referenced to the area being
#: searched is minimised by growing the wing. ``drag`` is the force, so it is
#: the free-area-safe half of the pair — the same split ``cz``/``downforce``
#: already carries.
#:
#: WHICH LEAVES ``cd`` DOING NOTHING ``drag`` DOES NOT, and that is stated
#: rather than hidden: at a fixed area the two differ by the constant ``q S``
#: and rank every design identically, and at a free area ``cd`` is refused. It
#: is kept because this family already carries the coefficient/force split in
#: three other places (``cz``/``downforce``, ``CD_budget``/``drag_budget_n``)
#: and a user who states a requirement as a coefficient should be able to
#: score one — not because it can reach an answer ``drag`` cannot. The SHELL
#: offers only ``drag`` for exactly this reason: one question, one place.
#:
#: THE SAME SENTINEL TRAP ``laptime`` HAS, and it bites sooner. ``drag`` scores
#: ``-D`` in newtons while :data:`PENALTY` is -100.0, so any design dragging
#: more than 100 N scores WORSE than a refusal. At the published S = 0.4 m^2
#: that is CD 0.135 at 55 m/s (out of reach) but CD 0.056 at 85 m/s (well
#: inside the box). A car run at high speed on this objective must use the
#: scale-aware refusal policy (``bo_refusal``), exactly as a >100 s lap must.
#: ``cd`` scores ``-CD`` ~ -0.03 and is nowhere near the sentinel.
#: ``downforce_plus_drag`` is the seventh, and it is the only entry here that
#: ADDS the two forces instead of trading them: ``F_z + D``, in newtons. It is
#: the total load the wing puts into the car — what the mount carries, and
#: what a braking-limited case wants, where the drag retards the car directly
#: and the downforce retards it through the tyres. (The general braking law is
#: ``D + mu*F_z``; this is that law at ``mu = 1``, and a slick's mu is nearer
#: 1.5, so it UNDER-weights the downforce. ``laptime`` is where that exchange
#: rate is measured rather than assumed.)
#:
#: IT IS A RATCHET, measured rather than feared — Nelder-Mead from the four
#: best of 512 Sobol draws, on each family's registered box:
#:
#: =============================  ===========  =============  =============
#: family                         ``F_z + D``  ``S_m2``       ``alpha_deg``
#: =============================  ===========  =============  =============
#: ``car rear wing``              1648.5 N     100 % of band  100 % of band
#: ``car rear wing + endplates``  1676.2 N     100 % of band  100 % of band
#: =============================  ===========  =============  =============
#:
#: — with every margin slack (0.85, 1.00, 2.49), so nothing physical stops it:
#: more area at the same CZ makes more of both terms, and more incidence makes
#: more of both until the section stalls. The answer is the top of your area
#: row and the top of your alpha sweep, and that is the sentence a user of
#: this objective has to read first.
#:
#: NOT REFUSED FOR IT, and the distinction from ``cz`` is the whole reason.
#: ``cz`` against a free area is an ARTEFACT: it is maximised by shrinking the
#: very area it is referenced to, so the answer moves OPPOSITE to the physics.
#: Here the preference IS the physics — a bigger wing really does put more
#: load into the car — and the area band is a regulation or a piece of
#: bodywork (:func:`area_row`), so spending all of it is the right answer to
#: the question as asked. What it needs is what ``downforce`` needs beside it:
#: a :attr:`CarWingProblem.drag_budget_n` ceiling. Measured on ``car rear
#: wing``, the ceiling is what turns the SPAN from a bound into an answer —
#:
#: ====================  ===========  =========  =============
#: ``drag_budget_n``     ``F_z + D``  ``D``      ``b_m``
#: ====================  ===========  =========  =============
#: none                  1648.5 N     114.8 N    0 % of band
#: 120 N (not binding)   1648.5 N     114.8 N    0 % of band
#: 81.5 N                1623.0 N     81.5 N     58 % of band
#: 50 N                  1580.1 N     50.0 N     100 % of band
#: ====================  ===========  =========  =============
#:
#: — the drag sits exactly ON the ceiling and buys span to stay there, while
#: the area and the incidence ride their ceilings under every one of them.
#:
#: THE SENTINEL TRAP ``drag`` HAS, read the other way round: this score is a
#: POSITIVE force of some hundreds of newtons against :data:`PENALTY` = -100,
#: so a refusal can never be mistaken for a good design here. ``drag`` owns
#: the dangerous end of that scale; this owns the safe one.
CAR_OBJECTIVES: dict[str, str] = {
    "cz": "downforce coefficient CZ",
    "downforce": "downforce force F_z [N]",
    "efficiency": "CZ/CD (= F_z/D)",
    "cd": "minus the drag coefficient CD (maximise => minimise CD)",
    "drag": "minus the drag force D [N] (maximise => minimise D)",
    "laptime": "minus the lap time [-s] (maximise => minimise the lap)",
    "downforce_plus_drag": "downforce + drag F_z + D [N] (the total load)",
}


def published_drag_budget_n(CD_budget: float = 0.11, S: float = 0.4,
                            V: float = 55.0, rho: float = RHO_AIR) -> float:
    """The coefficient drag allowance restated as a FORCE [N], at a given point.

    The free-area family opens on ``CD_budget`` q S evaluated at the published
    operating point — 81.5 N — so it inherits the same physical allowance the
    fixed-area family always had, rather than a fresh number chosen to make
    the new mode look good. Derived, never pasted: a budget that stopped
    tracking its own definition is how a calibration becomes folklore.
    """
    return float(CD_budget) * 0.5 * float(rho) * float(V) ** 2 * float(S)


def area_row(area_bounds_m2=None) -> np.ndarray:
    """(1, 2) design-box row for the reference AREA, in m^2.

    The area's twin of :func:`span_row`, and stated in the same unit and for
    the same reason: how big a rear wing may be is decided by a regulation or
    by the bodywork, never by a fraction of some default.
    """
    lo, hi = (AREA_BOUNDS_M2 if area_bounds_m2 is None
              else (float(area_bounds_m2[0]), float(area_bounds_m2[1])))
    if not (float(lo) > 0.0 and float(hi) > float(lo)):
        raise ValueError(
            f"area bounds must satisfy 0 < min < max, got ({lo}, {hi})")
    return np.array([[float(lo), float(hi)]], dtype=float)


def s_from_x(x, chord_order: int = 0) -> float:
    """The reference AREA read back out of a design vector [m^2].

    The area row sits immediately AHEAD of the span row — the package stacks
    ``[family][size][chord]`` and this family's size block is ``[S, b]`` — so
    it is two entries ahead of the chord coefficients. Ordered that way on
    purpose: it leaves :func:`span_from_x` reading the same slot whether or
    not the area is free, so every existing caller of the span is unchanged.
    """
    xs = np.asarray(x, dtype=float)
    n = int(chord_order)
    if xs.size < n + 2:
        raise ValueError(
            f"design vector of length {xs.size} carries no room for an area "
            f"and a span ahead of {n} chord coefficients")
    return float(xs[xs.size - n - 2])


def span_from_x(x, chord_order: int = 0) -> float:
    """The SPAN read back out of a design vector [m].

    The span row is the family's size block, and the package's stacking rule
    puts the chord coefficients last — so the span is the entry immediately
    ahead of them, needing no knowledge of the family block's length (the
    same inverse geometry.flight_from_x uses).
    """
    xs = np.asarray(x, dtype=float)
    n = int(chord_order)
    if xs.size < n + 1:
        raise ValueError(
            f"design vector of length {xs.size} carries no room for a span "
            f"ahead of {n} chord coefficients")
    return float(xs[xs.size - n - 1])

#: MOUNT LAYOUTS. Both of them take the load out through the PLATES — a rear
#: wing is bolted to the car by the sheets it already carries, and a pylon
#: reaching up from the deck through the wake is a thing this family no longer
#: offers. What the two layouts disagree about is WHERE ALONG THE SPAN the
#: sheets grip, which is the question that actually changes the answer:
#:
#:   ``tips``     the endplates themselves carry, at the tip. Two sheets, no
#:                new corners — the plates are there for the nonplanar
#:                benefit anyway, so the load path is free. The wing is a
#:                beam simply supported at +/- b/2 and SAGS at the centre.
#:   ``inboard``  a second pair of sheets grips at
#:                :data:`INBOARD_STATION_FRAC` of the semi-span. The span
#:                outboard of the grip HOGS while the middle sags, and the
#:                largest movement is much smaller than the tip-borne wing's
#:                — a support inboard of the tip is the classical way to cut
#:                a beam's deflection. It is NOT free: the two extra sheets
#:                are wetted area nobody was charging, and each one meets the
#:                wing in a corner, so this layout carries a sheet charge and
#:                two junctions that ``tips`` does not.
#:
#: ``supports`` is kept at ``ends`` on both — every consumer keyed on it is
#: asking "are the plates the load path", and now the answer is always yes.
#: ``n_struts`` is 0 on both and stays only so that :func:`_strut_cd0`'s
#: contract (and endplate.py's reading of this dict) is unchanged; the sheets
#: are charged through ``n_sheets`` with the same reduced-order plate model.
#:
#: THE PRICE OF DROPPING THE PYLON, measured at the box centre of the family
#: as it was: the centre-pylon layout cost CD 0.02374 against the
#: plate-mounted 0.02028 (+17.1 %: 0.00225 of pylon wetted area and 0.00121
#: of junction) and bought a deflection of 0.272 mm against 0.600 mm (2.2x
#: stiffer). That trade is gone; the ``inboard`` layout is how this family
#: buys stiffness now, and it buys it with sheets instead of struts.
#:
#: ``station_frac`` is a fraction of the SEMI-SPAN, not a length, for the
#: reason carmount.MountSpec gives: "at the tips" has to track a span that is
#: itself a design variable. The number is carmount's, imported rather than
#: pasted, so this dict and PUBLISHED_LAYOUTS cannot state different stations.
from .carmount import INBOARD_STATION_FRAC          # noqa: E402

MOUNTS: dict[str, dict] = {
    "tips": {
        "supports": "ends", "n_struts": 0, "n_junctions": 0,
        "station_frac": 1.0, "n_sheets": 0,
        "label": "the endplates carry, at the tips",
        "note": "the wing is simply supported at its tips and sags at the "
                "centre; the plates are load path and cost nothing extra, "
                "because they are there for the nonplanar benefit anyway",
    },
    "inboard": {
        "supports": "ends", "n_struts": 0, "n_junctions": 2,
        "station_frac": INBOARD_STATION_FRAC, "n_sheets": 2,
        "label": f"a second pair of plates carries, "
                 f"{INBOARD_STATION_FRAC:.0%} out",
        "note": "an inboard grip cuts the deflection sharply — the span "
                "outboard of it hogs while the middle sags — and pays for "
                "two extra sheets of wetted area and the two corners where "
                "they meet the wing",
    },
}


# ---------------------------------------------------------------- structure

def bending_moment(y: np.ndarray, load: np.ndarray, b: float,
                   supports: str = "centre") -> np.ndarray:
    """Bending-moment magnitude [N m] along the span for a support layout.

    ``load`` is the sectional force per unit span [N/m] at stations ``y``
    (symmetric, both sides). The half-span is treated by symmetry.

        centre : cantilever off the centreline —
                 M(y) = int_y^{b/2} l(e) (e - y) de
        ends   : simply supported at +/- b/2, reaction R = half the total —
                 M(y) = R (b/2 - y) - int_y^{b/2} l(e) (e - y) de

    Returned as MAGNITUDE: the two layouts bend the spar in opposite senses
    (hogging vs sagging), which changes which surface is in tension but not
    the size of the demand on the section.
    """
    y = np.asarray(y, dtype=float)
    load = np.asarray(load, dtype=float)
    if y.shape != load.shape:
        raise ValueError("y and load must have the same shape")
    if supports not in ("centre", "ends"):
        raise ValueError(f"unknown supports {supports!r}")
    order = np.argsort(y)
    ys, ls = y[order], load[order]
    half = ys >= 0.0
    yh, lh = ys[half], ls[half]
    if yh.size < 2:
        raise ValueError("need at least two stations on the half span")

    out = np.empty_like(yh)
    for i, yi in enumerate(yh):
        arm = yh[i:] - yi
        out[i] = np.trapezoid(lh[i:] * arm, yh[i:])
    if supports == "ends":
        total_half = float(np.trapezoid(lh, yh))
        out = total_half * (b / 2.0 - yh) - out
    # mirror back onto the caller's station order
    full = np.interp(np.abs(ys), yh, np.abs(out))
    inverse = np.empty_like(order)
    inverse[order] = np.arange(order.size)
    return full[inverse]


def deflection_index(y: np.ndarray, moment: np.ndarray, b: float,
                     supports: str = "centre") -> float:
    """Deflection of the wing RELATIVE TO ITS MOUNT, per unit EI [m^3/N... ].

    Curvature is M/EI; with EI factored out this is a stiffness-independent
    comparison between layouts, which is what the mount question needs. The
    symmetry condition gives zero slope at the centreline in both cases;
    the layouts differ in where the deflection is pinned:

        centre : pinned at the centreline  -> index = |w(b/2)| (tip droop)
        ends   : pinned at the tips        -> index = |w(0)|   (centre sag)

    Both are "how far the wing moves away from where it is bolted", i.e.
    the aeroelastic ride-height / incidence change at speed.
    """
    y = np.asarray(y, dtype=float)
    moment = np.asarray(moment, dtype=float)
    order = np.argsort(y)
    ys, ms = y[order], moment[order]
    half = ys >= 0.0
    yh, mh = ys[half], ms[half]
    slope = np.concatenate([[0.0], np.cumsum(
        0.5 * (mh[1:] + mh[:-1]) * np.diff(yh))])          # int kappa dy
    w = np.concatenate([[0.0], np.cumsum(
        0.5 * (slope[1:] + slope[:-1]) * np.diff(yh))])    # int slope dy
    if supports == "centre":
        return float(abs(w[-1]))          # tip, relative to the centreline
    return float(abs(w[-1] - w[0]))       # centre, relative to the tips


# ------------------------------------------------- the elastic twist, flown

def _twist_projection(y: np.ndarray, phi_rad: np.ndarray,
                      weight: np.ndarray, b: float) -> tuple:
    """``(dtw_root_deg, dtw_tip_deg)``: phi(y) as THIS family's twist law.

    The wing's twist law is linear in ``eta = |y| / (b/2)`` (geometry.Wing),
    and the elastic twist from a torsion shaft is not — so flying the windup
    means choosing the linear law that best represents it. The choice made
    here is the ``weight``-weighted least-squares projection of phi onto
    ``p0 + p1 eta``, with ``weight`` the strip AREA c dy.

    Two properties, and both are reasons rather than conveniences:

    * the residual of a weighted least-squares fit is orthogonal to every
      basis function, and the constant is one of them, so the AREA-weighted
      mean twist is reproduced EXACTLY. Under strip theory the added lift of
      a twist distribution is ``q a int phi c dy``, so preserving that mean
      preserves the added lift to first order — the fit does not lose the
      downforce the windup makes;
    * a phi that is already linear is reproduced exactly, so the projection
      is invisible wherever it has nothing to do.

    What it does not reproduce is the CURVATURE of phi, and therefore the
    exact spanwise redistribution of the added load (a torsion cantilever's
    phi is concave, so the fit sits below it inboard and above it outboard).
    The caller reports the residual.

    Degenerate weights (a zero-area strip set) fall back to the weighted
    mean, i.e. a uniform increment, rather than dividing by zero.
    """
    eta = np.abs(np.asarray(y, dtype=float)) / (0.5 * float(b))
    phi = np.rad2deg(np.asarray(phi_rad, dtype=float))
    w = np.asarray(weight, dtype=float)
    s0 = float(np.sum(w))
    if not s0 > 0.0:
        return (0.0, 0.0)
    s1 = float(np.sum(w * eta))
    s2 = float(np.sum(w * eta * eta))
    t0 = float(np.sum(w * phi))
    t1 = float(np.sum(w * eta * phi))
    det = s0 * s2 - s1 * s1
    if not det > 0.0:                      # every station at one eta
        mean = t0 / s0
        return (float(mean), float(mean))
    p0 = (t0 * s2 - t1 * s1) / det
    p1 = (s0 * t1 - s1 * t0) / det
    return (float(p0), float(p0 + p1))


def _twist_fit_residual_deg(y: np.ndarray, phi_rad: np.ndarray,
                            fit: tuple, b: float) -> float:
    """Largest |phi - fit| over the span [deg] — what the linear law lost."""
    eta = np.abs(np.asarray(y, dtype=float)) / (0.5 * float(b))
    phi = np.rad2deg(np.asarray(phi_rad, dtype=float))
    line = fit[0] + (fit[1] - fit[0]) * eta
    return float(np.max(np.abs(phi - line))) if phi.size else 0.0


# ----------------------------------------------- the polar at the flown Re

@lru_cache(maxsize=1)
def _re_bank() -> dict:
    """The NACA 24XX Reynolds bank, read from disk ONCE.

    ``polar.polar_re_bank`` re-reads every ``.pol`` file on every call (7.6 ms
    measured), which is five times a whole car-wing evaluation. The bank is
    static data on disk, so it is cached here rather than in polar.py, which
    is another session's file.
    """
    from .polar import polar_re_bank
    return polar_re_bank()


def _polar_at_flown_re(tc: float, re: float):
    """The section polar AT the flown Reynolds number (polar.polar_at_re).

    Raises ``ValueError`` — the in-contract caller catches it — when the
    thickness is not a bank member or the Reynolds number is outside the
    bank. Neither is papered over: an extrapolated viscous polar is a
    fabrication, and a t/c between two members needs the t/c blend first.
    """
    from .polar import polar_at_re
    return polar_at_re(float(tc), float(re), bank=_re_bank())


# ---------------------------------------------------------------- problem

def _bounds_reason(x: np.ndarray, bnds: np.ndarray, prob) -> str:
    """The refusal for a design outside the box — with the DECK named when it
    is the deck that moved the wall.

    Every family in this package answers an out-of-box design with the bare
    words "bounds violation", and this returns them too. The one case it says
    more about is the one this family made: the ride row's floor is DERIVED
    (:attr:`CarWingProblem.ride_row_m`) and rises to the attachment deck when
    a pylon is carrying, so a wing asked for below its own deck now meets the
    bounds check where it used to meet ``MountSpec.violation``'s sentence
    about a pylon of negative length. Losing that sentence would trade a
    reason a reader can act on for a generic one, which is a regression
    dressed as a refusal.
    """
    lo = float(bnds[_RIDE_ROW, 0])
    v = float(x[_RIDE_ROW])
    if (prob.reach_required and v < lo - 1e-12
            and abs(lo - float(prob.deck_m)) <= 1e-12):
        return (f"bounds violation: the wing is asked for {v:.4g} m above the "
                f"track and its attachment deck is at {float(prob.deck_m):g} "
                f"m, so the pylon would have to be "
                f"{v - float(prob.deck_m):.3g} m long. The ride row starts at "
                f"the deck for that reason (CarWingProblem.ride_row_m); lower "
                f"the deck or raise the wing")
    return "bounds violation"


def _strut_cd0(n_struts: int, height: float, chord: float, s_ref: float,
               cf: float = 0.005) -> float:
    """Reduced-order pylon parasite drag, referenced to the wing area.

    Wetted area = 2 sides x height x chord per strut, flat-plate turbulent
    Cf ~ 0.005 and no form factor — the same deliberately crude model as
    hydrofoil._mast_cd0, and stated as such.
    """
    return float(n_struts * cf * (2.0 * height * chord) / s_ref)


@dataclass
class CarWingProblem:
    """Downforce maximisation for a rear wing, at a drag and deflection budget.

    Design vector::

        x = [taper, twist_root_deg, twist_tip_deg, alpha_deg,
             endplate_h_m, ride_height_m, (S_m2,) b_m]

    with the ``S_m2`` row present only when :attr:`area_bounds_m2` is set.

    ``alpha_deg`` is the wing's incidence in the MIRRORED frame (bigger =
    more downforce); ``endplate_h_m`` is the endplate's arc length IN METRES,
    pointing at the track; ``ride_height_m`` is the wing's height above it;
    ``b_m`` is the OVERALL WIDTH — the wing plus whatever its plates project
    outboard, which is what a regulation or a piece of bodywork measures. A
    vertical unblended plate projects nothing, so there b_m is the wing's
    span exactly; lean or blend the plate and the wing is shortened to keep
    the total inside the row (``overall_width_m`` and
    ``endplate_projection_m`` are both reported).

    Two things about that vector are deliberate, and both say the same thing —
    a car wing's dimensions are set by the car, not by a fraction of itself:

    * THE ENDPLATE HEIGHT IS A LENGTH. It used to be an arc length over the
      semi-span, which made it move whenever the span moved. But what a plate
      has to do is reach from the wing to the bodywork — a distance in metres
      that does not care how wide the wing is (endplate.py's reach constraint
      is exactly that distance). A fraction of a span that is itself a design
      variable would have been one number meaning two things.
    * THE SPAN IS A DESIGN VARIABLE. It is the rear wing's first-order
      variable: at the fixed reference AREA the span IS the aspect ratio, so
      it trades induced drag against the bending the mount has to carry, and
      both constraints below see it. Its box is a length band in metres
      (:attr:`span_bounds_m`) because a rear wing's width is decided by a
      regulation or by the bodywork — never by a fraction of some default.

    The reference area ``S`` is CONFIGURATION BY DEFAULT, for the same reason
    the score can be a coefficient: CZ, the drag budget and the reported
    efficiency are all referenced to it, so every candidate is compared on
    one area and only the shape of the planform moves. Set
    :attr:`area_bounds_m2` and it becomes a design variable instead — one row
    ahead of the span — but then the score has to be stated in FORCES, which
    is what :attr:`objective` is for. See :data:`CAR_OBJECTIVES`.

    Objective (MAXIMISE): :attr:`objective`, default the downforce coefficient
    CZ = -CL_model (the published one).

    Constraints (all >= 0, and only the ones this problem DECLARES — see
    :attr:`constraint_labels`, which is what ``api.design_report`` reads):

        CD_budget - CD                    [iff ``CD_budget``]
             a drag COEFFICIENT allowance, because unconstrained downforce is
             a meaningless target at a fixed area (more angle always makes
             more, until it stalls). Switchable off: "maximise it and do not
             budget it" is a real question for a wing whose drag is paid for
             elsewhere, and the family used to have no way to ask it.
        1 - D / drag_budget_n             [iff ``drag_budget_n``]
             the same allowance in NEWTONS. The one that means something when
             the area is free: a coefficient budget referenced to a moving
             area is not a fixed allowance on anything the car can feel.
        1 - deflection / deflection_limit  [always]
             the mount's structural say in the answer, computed from the
             solved load. Always present, so there is always at least one
             margin and the family is never a constrained problem with
             nothing to constrain; a caller who wants it out of the way
             raises the limit rather than removing the beam.
        F_z / downforce_min_n - 1         [iff ``downforce_min_n``]
             a downforce FLOOR — "the most efficient wing that still makes
             the downforce I need". Not what makes ``efficiency`` well-posed
             (it has its own shallow interior peak in S, measured at
             S ~ 0.14 m^2), but what makes it USEFUL: without a floor the
             most efficient rear wing is a small one making little
             downforce, and the floor is the statement of how much is worth
             having. The answer then sits ON the floor rather than at a box
             end, which is the correct shape of answer. With
             ``downforce_min_v_ms`` the floor is required AT a stated speed
             rather than at :attr:`V`.
        ride / deck - 1                   [iff a deck AND a pylon]
             the PYLON REACH margin: a wing at or below its own attachment
             deck has no pylon to build. endplate.py's reach constraint is
             the same statement for the plate that has to span the same gap.
        1 - |dCZ/dh| / dczdh_max_per_m    [iff ``dczdh_max_per_m``]
             a ceiling on the ride-height sensitivity — see the field.
        min(balance - lo, hi - balance)   [iff the car states a window]
             the aero balance, from ``cartrack.balance_margin``.

    Scale: a touring-car-ish rear wing, b = 1.6 m, S = 0.4 m^2 (chord
    0.25 m, AR 6.4), 55 m/s (~200 km/h), Re_mac ~ 9.5e5 at the published
    design point against the family's Re = 1e6 polars — the same flagged
    mismatch the hydrofoil problem carries, and now switchable
    (:attr:`flown_reynolds`). Measured at the published design point
    (Re_mac 9.4539e5 against the 1e6 tables): reading the polar at the flown
    Reynolds number raises CZ by 0.660 %, CD by 1.152 % (CDp 1.169 %, CDi
    1.313 % — the blended table's lift slope is higher too, so the same
    incidence makes more lift and more induced drag) and therefore COSTS
    0.486 % of CZ/CD.
    """

    b: float = 1.6
    S: float = 0.4
    N_vlm: int = 40
    n_endplate: int = 8
    V: float = 55.0
    rho: float = RHO_AIR
    mu: float = MU_AIR
    tc: float = 0.12
    #: which of the two plate-borne layouts (:data:`MOUNTS`) — where along
    #: the span the sheets grip. Defaults to the tips, which is the layout
    #: the plates already there provide for nothing.
    mount: str = "tips"
    strut_chord: float = 0.12       # pylon chord [m]
    strut_cf: float = 0.005
    ei_nm2: float = 4.0e4           # spar bending stiffness EI [N m^2]
    #: drag COEFFICIENT allowance on S, or None for no coefficient drag
    #: margin at all. None is the "maximise it, do not budget it" mode: it is a
    #: legitimate question for a wing whose drag is paid for elsewhere in the
    #: car's budget, and refusing to ask it was the reason the family could
    #: only ever be run one way. Note what a COEFFICIENT budget means once the
    #: area is free — an allowance referenced to a moving area, which is not a
    #: fixed allowance on anything physical; :attr:`drag_budget_n` is the same
    #: statement in newtons and is the one to use there.
    CD_budget: float | None = 0.11
    #: drag FORCE allowance [N], or None. The area-independent budget: D =
    #: q S CD <= drag_budget_n. Off by default, so every published run keeps
    #: exactly the two margins it always had.
    drag_budget_n: float | None = None
    #: downforce FLOOR [N], or None. What turns ``objective="efficiency"``
    #: into the question a race engineer actually has — "the most efficient
    #: wing that still makes the downforce I need". Measured over the default
    #: area band at the box centre, efficiency alone peaks at S ~ 0.14 m^2
    #: (35.5) and falls away either side, so it is not degenerate without a
    #: floor; but that peak is a wing making 190 N, and the floor is what
    #: says how much downforce is worth having. With a 400 N floor the answer
    #: moves to S = 0.33 m^2 and sits ON the floor, which is the correct shape
    #: of answer: spend exactly the area the downforce demands, no more.
    downforce_min_n: float | None = None
    deflection_limit_m: float = 0.010   # movement relative to the mount [m]
    blend_frac: float = 0.0         # endplate root transition (geometry.py)
    blend_shape: str = "arc"        # arc | smooth (geometry.BLEND_SHAPES)
    #: WHICH WAY the plate leaves the wing, in degrees from the wing plane.
    #: 90 is straight towards the track and is what every published run flew;
    #: below 90 the plate leans outboard, which is what a real rear wing's
    #: plates do and what ``geometry.developed_semispan`` has always priced —
    #: the number was simply hard-wired at the one call site. The height
    #: ``endplate_h_m`` is an ARC LENGTH along the plate, so canting trades
    #: the plate's vertical reach for outboard projection at constant plate.
    endplate_cant_deg: float = 90.0
    #:
    #: PRICED, and only since the span row became the OVERALL width on this
    #: family. While ``b_m`` was read as the wing's span a lean was a free
    #: lunch — cant 60 scored 39.79 against 90's 38.28 by flying 1.725 m of
    #: wing inside a 1.6 m band — so the flag was withheld; that never worked,
    #: because ``blend_frac`` was declared and walked through the same gap.
    #: ``evaluate_car_wing`` now takes the projection out of the wing's span
    #: before anything is flown, so leaning COSTS here exactly as it does on
    #: ``endplate.CarWingEndplateProblem``.
    #: does the plate carry the wing's CHORD LAW past the tip, or the wing's
    #: TIP CHORD the whole way? False — a rectangle — is the published plate.
    #: On this family the plate has no chord of its own (that is what the
    #: designed-endplate family adds), so this is the only way to make a
    #: tapered wing's plate taper with it.
    endplate_chord_follows: bool = False
    junction_drag: bool = True      # a mount corner is exactly what junction
    #                                 drag is for; ON here, unlike the
    #                                 aircraft problems
    chord_order: int = 0            # free chord law (geometry.py); 0 = taper
    chord_max_frac: float = geometry.CHORD_COEFF_BOUND
    chord_law: str = geometry.DEFAULT_CHORD_LAW   # WHICH SHAPE the
    #   coefficients above describe (geometry.CHORD_LAWS). Changes what
    #   the design vector DRAWS, never how long it is.
    chord_limits: "geometry.ChordLimits | None" = None   # what the chord
    #   DISTRIBUTION has to obey, in metres and degrees (geometry.ChordLimits:
    #   a minimum chord, a maximum, a maximum local taper ANGLE, and which end
    #   is the wide one). None = unconstrained, which is every published run.
    #   Checked where the planform is built, so a violation is an in-contract
    #   failure (the penalty contract), never an exception at the optimiser.
    #: the aspect-ratio band the USER will accept b^2/S in, as
    #: ``(ar_min, ar_max)`` with either end None (api.AR_LIMIT_KEYS).
    #: A SECOND band beside ``sizing.AR_LIMITS``, never a replacement
    #: for it, and None — no limit at all — is every published run.
    ar_limits: "tuple | None" = None
    polar_family: Any = None
    #: fly a CHOSEN section instead of selecting one by thickness off the
    #: NACA 24XX family. None keeps the published behaviour bit-for-bit; a
    #: polar here also supplies the thickness (``.tc``), since the shape
    #: already has one. The design vector is UNCHANGED either way — this
    #: family's t/c was never a design variable.
    section_polar: Any = None
    alpha_bracket_deg: tuple = (-2.0, 16.0)
    #: the span band this wing may be searched over [m], or None for
    #: :data:`SPAN_BOUNDS_M`. A regulation, a bodywork width or a transporter
    #: — stated as a length, because that is what it is.
    span_bounds_m: tuple | None = None
    #: WHICH SCALAR is maximised (:data:`CAR_OBJECTIVES`). ``cz`` is the
    #: published objective and the default.
    objective: str = "cz"
    #: make the reference AREA a design variable, over this band in m^2
    #: (None = the published FIXED area :attr:`S`; ``True`` or an empty tuple
    #: is not accepted — a band is a band). Adds ONE row, immediately ahead of
    #: the span row, so :func:`span_from_x` is unchanged.
    area_bounds_m2: tuple | None = None

    # ------------------------------------------------------------------
    # Everything below is OFF by default and each default is the published
    # behaviour, so ``CarWingProblem()`` evaluates bit-for-bit as it always
    # has. None of them touches the design vector.
    # ------------------------------------------------------------------

    #: the mount as a continuum (:class:`carmount.MountSpec`) instead of the
    #: two-valued :data:`MOUNTS` lookup. ``None`` keeps the lookup. See the
    #: module docstring for the five things it changes; note that adopting it
    #: also adopts carmount's PYLON LENGTH (ride - deck), which is a
    #: correction and not a switch, and therefore moves the strut drag.
    mount_spec: "carmount.MountSpec | None" = None
    #: the car's attachment deck above the track [m], for the published
    #: :data:`MOUNTS` path. ``None`` = the published behaviour, which charges
    #: the pylon over the WHOLE ride height — down to the track, through a gap
    #: where there is no pylon. Setting it charges the pylon over ``ride -
    #: deck`` (endplate.py measures the same piece of car that way) and
    #: declares a PYLON REACH margin. MEASURED at the published point (ride
    #: 0.30 m, deck 0.25 m, the centre mount): the pylon charge falls exactly
    #: 6x (0.0018 -> 0.0003), CD falls 4.715 % (1.112 N of drag at 55 m/s)
    #: and CZ/CD rises 4.948 %. Stated on a ``mount_spec`` instead when one
    #: is given: two homes for one number is how they drift apart.
    deck_height_m: float | None = None
    #: the car the wing is bolted to (:class:`cartrack.CarSpec`). Needed for a
    #: lap and for the aero balance; ``None`` with a track means cartrack's
    #: own stated reference car.
    car_spec: Any = None
    #: the circuit (:class:`cartrack.TrackSpec`). ``None`` = no mission, which
    #: is every published run. With one set the wing is evaluated at the lap's
    #: representative speeds and the breakdown carries ``lap_time_s``.
    track_spec: Any = None
    #: how many representative speeds the lap is sampled at (cartrack's own
    #: default is 4). Each one is a full re-solve of the wing.
    track_points: int = 4
    #: the speed [m/s] the DOWNFORCE FLOOR is required at, or None for
    #: :attr:`V`. E5's "a force requirement stated AT a speed": a rear wing
    #: that makes its minimum downforce at 55 m/s may not make it in the slow
    #: corner where the driver wants it. Needs :attr:`downforce_min_n`.
    downforce_min_v_ms: float | None = None
    #: the speed [m/s] the DRAG BUDGET is charged at, or None for :attr:`V`.
    #: The twin of the above, and the more useful of the two: a drag
    #: allowance bites at the end of the longest straight, not at the mean
    #: speed. Needs :attr:`drag_budget_n`.
    drag_budget_v_ms: float | None = None
    #: MEASURE dCZ/dh and put it in the breakdown. Off by default, and that
    #: default is a COST decision rather than a taste one: the quotient needs
    #: the wing built and solved at two more ride heights, which at the
    #: published design point takes the evaluation from 1.55 ms to 4.58 ms
    #: (x2.97, measured — see :func:`evaluate_car_wing`). The car families are
    #: the most expensive objects an optimiser in this repo is asked to
    #: search, and carwing_multi pays this inside its own inner loop, so a
    #: diagnostic nobody asked for is not charged to everybody. Implied by
    #: :attr:`dczdh_max_per_m` (a ceiling cannot be checked without the
    #: number) — ask :attr:`dczdh_required` rather than either field.
    dczdh_report: bool = False
    #: central-difference step [m] for dCZ/dh. 5 mm is about 1.7 % of the
    #: published 0.30 m ride height; the quotient is second-order accurate in
    #: it and the VLM is deterministic, so there is no noise floor to stay
    #: above. MEASURED at the published point: halving the step from 0.04 m
    #: divides the change by 4.10, 4.03, 4.01, 4.00 — the ratio 4 that h^2
    #: promises — and the default step lands 1.8e-4 (0.04 %) from the
    #: Richardson limit of -0.45052 1/m.
    dczdh_step_m: float = 0.005
    #: ceiling on |dCZ/dh| [1/m], or None. Declares a RIDE-HEIGHT SENSITIVITY
    #: margin. Physically: how much downforce coefficient the wing loses per
    #: metre it rises. It is NEGATIVE (ground proximity makes downforce), and
    #: a large magnitude is a knife-edge aero platform — the downforce, and
    #: therefore the balance, moves with every bump, kerb and braking event.
    dczdh_max_per_m: float | None = None
    #: must the tip plate stop short of the car's attachment deck? Under a
    #: PYLON mount the plate is a fence of free height and its height is a
    #: design row: nothing stopped it running past the deck and into the
    #: bodywork, which is where a plate longer than the pylon that holds the
    #: wing up would actually be. Declares a MOUNT CLEARANCE margin wherever
    #: a deck and pylons are both stated (:attr:`plate_clears_deck`) — which
    #: is no published run, since both :data:`MOUNTS` layouts are plate-borne.
    #: True by default because it is the geometry of a car and not a
    #: preference; False is the way to ask the unbounded question back, and
    #: the margin's absence then says so in ``constraint_labels`` rather than
    #: the box quietly hiding it.
    plate_deck_clearance: bool = True
    #: read the section polar at the FLOWN Reynolds number
    #: (``polar.polar_at_re``) instead of the family's Re = 1e6 tables. Off by
    #: default: every published number in this family was flown on the 1e6
    #: tables at Re_mac ~ 9.5e5, a mismatch the class docstring has always
    #: flagged. Needs the NACA 24XX bank, so it is refused with a chosen
    #: ``section_polar`` (one table at one Reynolds number).
    flown_reynolds: bool = False
    #: fixed-point tolerance for the torsional windup, in DEGREES of applied
    #: incidence. 1e-4 deg is two orders below anything this family resolves.
    windup_tol_deg: float = 1e-4
    #: iteration cap for that fixed point, and the BACKSTOP rather than the
    #: divergence detector — divergence is detected on the map itself (two
    #: consecutive rounds whose feedback gain is >= 1, i.e. a re-solve that
    #: has stopped contracting), which costs three solves rather than this
    #: many. What the cap catches is a windup that IS contracting but too
    #: slowly to settle. MEASURED (published wing, 1 deg incidence, centre
    #: mount at 0.35 chord, GJ swept): 40 rounds resolves a feedback gain of
    #: 0.774 (39 rounds) and refuses 0.786 (41 rounds). Both outcomes are
    #: refused IN CONTRACT, with the margin and the measured gain attached,
    #: and the two reasons say which of them happened.
    windup_max_iter: int = 40

    TAPER_BOUNDS = (0.4, 1.0)
    TWIST_ROOT_BOUNDS_DEG = (-4.0, 4.0)
    TWIST_TIP_BOUNDS_DEG = (-6.0, 2.0)
    ALPHA_BOUNDS_DEG = (0.0, 12.0)
    #: endplate arc length, IN METRES (see the class docstring). The band is
    #: the published fraction (0-0.30 of the semi-span) evaluated at the
    #: published 1.6 m span, so the family opens on the same plates it always
    #: offered — they are simply no longer tied to a span that now moves.
    ENDPLATE_H_M_BOUNDS = (0.0, 0.25)
    RIDE_HEIGHT_BOUNDS_M = (0.15, 0.60)

    def __post_init__(self):
        if self.mount not in MOUNTS:
            raise ValueError(
                f"unknown mount {self.mount!r}; choose from {sorted(MOUNTS)}")
        if self.blend_shape not in geometry.BLEND_SHAPES:
            raise ValueError(
                f"unknown blend_shape {self.blend_shape!r}; "
                f"choose from {list(geometry.BLEND_SHAPES)}")
        if self.polar_family is None and self.section_polar is None:
            from .polar import default_polar_family
            self.polar_family = default_polar_family()
        if self.section_polar is not None:
            # the chosen shape's own thickness — it feeds the endplate
            # junction charge and the reported geometry
            self.tc = float(getattr(self.section_polar, "tc", self.tc))
        if self.objective not in CAR_OBJECTIVES:
            raise ValueError(
                f"unknown objective {self.objective!r}; "
                f"choose from {sorted(CAR_OBJECTIVES)}")
        if self.area_free and self.objective == "cd":
            # the same refusal as 'cz' below, read backwards: a drag
            # COEFFICIENT referenced to the area being searched is minimised
            # by GROWING the wing, so the answer is the top of the area band
            # whatever the aerodynamics do. Use the force.
            raise ValueError(
                "objective 'cd' is meaningless with a free reference area: "
                "the drag COEFFICIENT is referenced to the very area being "
                "searched, so it is minimised by growing the wing. Use "
                "objective='drag' for the force (with a downforce_min_n "
                "floor) — or fix the area (area_bounds_m2=None)")
        if self.area_free and self.objective == "cz":
            # NOT a preference. CZ is referenced to S, so maximising it with S
            # free is maximised by shrinking S — the answer is the bottom of
            # the area band whatever the aerodynamics do, and it comes back
            # looking like a result. State the question in forces instead.
            raise ValueError(
                "objective 'cz' is meaningless with a free reference area: "
                "the downforce COEFFICIENT is referenced to the very area "
                "being searched, so it is maximised by shrinking the wing "
                "(CZ 0.528 -> 0.739 as S falls 0.90 -> 0.15 m^2, while the "
                "downforce produced falls 881 -> 205 N). Use "
                "objective='downforce' for the force, or 'efficiency' with a "
                "downforce_min_n floor — or fix the area (area_bounds_m2=None)")
        if self.area_free:
            area_row(self.area_bounds_m2)          # validate the band now
        if self.CD_budget is not None and not float(self.CD_budget) > 0.0:
            raise ValueError(
                f"CD_budget must be > 0 or None (no coefficient drag margin), "
                f"got {self.CD_budget!r}")
        for name in ("drag_budget_n", "downforce_min_n"):
            v = getattr(self, name)
            if v is not None and not float(v) > 0.0:
                raise ValueError(f"{name} must be > 0 or None, got {v!r}")
        self._check_mount_spec()
        self._check_mission()
        self._check_sensitivity_and_reynolds()

    # -- the opt-in validations, one method per question they answer --------

    def _check_mount_spec(self):
        if self.mount_spec is None:
            if self.deck_height_m is not None and not float(
                    self.deck_height_m) > 0.0:
                raise ValueError(
                    f"deck_height_m must be > 0 or None (the published "
                    f"behaviour, which charges the pylon over the whole ride "
                    f"height), got {self.deck_height_m!r}")
            return
        if not isinstance(self.mount_spec, carmount.MountSpec):
            raise ValueError(
                f"mount_spec must be a carmount.MountSpec or None, got "
                f"{type(self.mount_spec).__name__}")
        if self.deck_height_m is not None:
            raise ValueError(
                "the attachment deck is stated once: mount_spec carries its "
                "own deck_height_m, so setting the problem's as well gives "
                "one number two homes and lets them drift apart. Set "
                "carmount.MountSpec(deck_height_m=...) instead")
        if self.mount != type(self).mount:
            # compared against the FIELD's own default rather than a pasted
            # literal: the default layout's name is a thing that moves (it
            # was "centre" until the pylon left this family), and a sentinel
            # written down twice is a sentinel that goes stale in one of them
            raise ValueError(
                f"the mount is stated once: mount_spec is a full mount "
                f"description, so mount={self.mount!r} is a second, "
                f"contradictory answer. Drop one — carmount.published_layout"
                f"({self.mount!r}) is the MOUNTS entry as a MountSpec")
        if not float(self.windup_tol_deg) > 0.0:
            raise ValueError(
                f"windup_tol_deg must be > 0, got {self.windup_tol_deg!r}")
        if int(self.windup_max_iter) < 1:
            raise ValueError(
                f"windup_max_iter must be >= 1, got {self.windup_max_iter!r}")

    def _check_mission(self):
        if self.objective == "laptime" and self.track_spec is None:
            raise ValueError(
                "objective 'laptime' needs a circuit: set track_spec (a "
                "cartrack.TrackSpec, e.g. cartrack.synthetic_lap()) and, if "
                "the car is not cartrack's reference car, car_spec. A lap "
                "time is not a property of a wing")
        if self.track_spec is not None and int(self.track_points) < 1:
            raise ValueError(
                f"track_points must be >= 1, got {self.track_points!r}")
        for speed, force in (("downforce_min_v_ms", "downforce_min_n"),
                             ("drag_budget_v_ms", "drag_budget_n")):
            v = getattr(self, speed)
            if v is None:
                continue
            if not float(v) > 0.0:
                raise ValueError(f"{speed} must be > 0 or None, got {v!r}")
            if getattr(self, force) is None:
                raise ValueError(
                    f"{speed} states the speed a requirement is made AT, and "
                    f"{force} is the requirement. A speed without a force is "
                    f"not a requirement — set {force} too, or drop {speed}")

    def _check_sensitivity_and_reynolds(self):
        if not float(self.dczdh_step_m) > 0.0:
            raise ValueError(
                f"dczdh_step_m must be > 0, got {self.dczdh_step_m!r}")
        if (self.dczdh_max_per_m is not None
                and not float(self.dczdh_max_per_m) > 0.0):
            raise ValueError(
                f"dczdh_max_per_m is a ceiling on |dCZ/dh| and must be > 0 or "
                f"None, got {self.dczdh_max_per_m!r}")
        if self.flown_reynolds and self.section_polar is not None:
            raise ValueError(
                "flown_reynolds reads the NACA 24XX Reynolds bank "
                "(polar.polar_at_re), and a chosen section_polar is ONE table "
                "at ONE Reynolds number — there is nothing to read it at. "
                "Drop one of the two")

    @property
    def area_free(self) -> bool:
        """Is the reference area a design variable?"""
        return self.area_bounds_m2 is not None

    @property
    def ride_row_m(self) -> tuple:
        """The ride band this problem actually SEARCHES [m].

        :data:`RIDE_HEIGHT_BOUNDS_M` is the band of a wing over a track. Bolt
        a PYLON mount to it and the bottom of that band stops being a design
        at all: the pylon's length is ``ride - deck``
        (:func:`carmount.pylon_length_m`), so every ride height at or below
        the deck asks for a member of zero or negative length, which
        :meth:`carmount.MountSpec.violation` refuses by name. On the shipped
        deck of 0.25 m that is 0.10 m of a 0.45 m row — 22.2 % of the row
        REFUSED rather than scored, and the box centre sitting at a 0.125 m
        stub because half the band it averages does not exist.

        So the floor rises to the deck when a pylon is carrying, and the box
        shown is the box searched. Two things this deliberately is NOT:

        * not a NARROWING of anything reachable — the region removed is
          exactly the region that returns a refusal, and the endplate family
          states the same rule the other way round, refusing a deck at or
          above its band floor (``endplate.CarWingEndplateProblem``);
        * not a floor on how SHORT a pylon may be. There is no measurement in
          this package for a minimum buildable strut, and inventing one would
          be a fabricated calibration. The edge itself is a zero-length pylon
          and is still refused, by the same violation as before.

        A plate-borne mount (every published layout) keeps the published band
        to the bit, because it has no pylon and states no deck.
        """
        lo, hi = (float(self.RIDE_HEIGHT_BOUNDS_M[0]),
                  float(self.RIDE_HEIGHT_BOUNDS_M[1]))
        if not self.reach_required:
            return (lo, hi)
        deck = float(self.deck_m)
        if deck >= hi:
            raise ValueError(
                f"the attachment deck is at {deck:g} m and the ride band tops "
                f"out at {hi:g} m, so no wing in this box sits above its own "
                f"deck and every pylon would have a negative length. Lower "
                f"the deck (deck_height_m) or raise the ride band "
                f"(bounds_overrides['ride_height_m'])")
        return (max(lo, deck), hi)

    @property
    def bounds(self) -> np.ndarray:
        # [family][size][chord] — the package's block order. The size block is
        # [S, b] when the area is free and [b] when it is not, and the area
        # goes AHEAD so the span keeps its slot either way (s_from_x).
        family = np.array([
            self.TAPER_BOUNDS,
            self.TWIST_ROOT_BOUNDS_DEG,
            self.TWIST_TIP_BOUNDS_DEG,
            self.ALPHA_BOUNDS_DEG,
            self.ENDPLATE_H_M_BOUNDS,
            self.ride_row_m,
        ], dtype=float)
        size = ([area_row(self.area_bounds_m2)] if self.area_free else [])
        box = np.vstack([family, *size, span_row(self.span_bounds_m)])
        return geometry.with_chord_bounds(box, self.chord_order,
                                          self.chord_max_frac,
                                          self.chord_law)

    @property
    def dim(self) -> int:
        return int(self.bounds.shape[0])

    @property
    def constraint_labels(self) -> tuple:
        """Display names for the margins this problem actually returns.

        ``api.design_report`` prefers these over the static ProblemSpec
        labels, which describe the family with its published budgets (the
        precedent is ``hydrofoil.HydrofoilWingletProblem`` under its draught
        cap). Without this, a run with the coefficient budget off would report
        its deflection margin under the drag budget's name.
        """
        out = []
        if self.CD_budget is not None:
            out.append("drag budget margin")
        if self.drag_budget_n is not None:
            out.append("drag force margin")
        out.append("deflection margin")
        if self.downforce_min_n is not None:
            out.append("downforce floor margin")
        # the opt-in margins go AFTER the published four, so a run that turns
        # one on keeps every existing label at the index it always had
        if self.reach_required:
            out.append("pylon reach margin")
        if self.dczdh_max_per_m is not None:
            out.append("ride-height sensitivity margin")
        if self.balance_required:
            out.append("aero balance margin")
        # ...and the newest one LAST, for the reason stated above: every
        # margin that existed before it keeps the index it always had, so a
        # reader (and gui/diagnose.py, which names margins BY INDEX) cannot
        # be handed a different question under an old number.
        if self.plate_clears_deck:
            out.append("mount clearance margin")
        return tuple(out)

    @property
    def n_constraints(self) -> int:
        return len(self.constraint_labels)

    @property
    def mount_layout(self) -> dict:
        """The published two-valued :data:`MOUNTS` entry.

        Was called ``mount_spec`` until that name became the field holding a
        :class:`carmount.MountSpec`. It stays the answer whenever no
        MountSpec is given, and it is what endplate.py and carwing_multi.py
        still ask their own copies for.
        """
        return MOUNTS[self.mount]

    @property
    def deck_m(self) -> float | None:
        """The attachment deck above the track [m], from wherever it is stated.

        One place to ask, so the pylon length, the reach margin and the
        breakdown cannot disagree about which deck they meant.
        """
        if self.mount_spec is not None:
            return float(self.mount_spec.deck_height_m)
        return (None if self.deck_height_m is None
                else float(self.deck_height_m))

    @property
    def n_pylons(self) -> int:
        """How many pylons this mount charges, from either statement."""
        if self.mount_spec is not None:
            return int(self.mount_spec.n_pylons)
        return int(MOUNTS[self.mount]["n_struts"])

    @property
    def reach_required(self) -> bool:
        """Is there a pylon that has to REACH a stated deck?

        Both halves are needed, and each is a different sentence: a deck the
        caller never stated is the published behaviour (the pylon is charged
        to the track and has nothing to reach), and a mount with no pylons —
        the endplate-mounted wing — has nothing to do the reaching. So the
        margin exists iff a pylon and a deck are both stated, which is the
        family's rule that a margin exists iff its budget does.
        """
        return self.deck_m is not None and self.n_pylons > 0

    @property
    def plate_clears_deck(self) -> bool:
        """Must the tip plate stop short of the car's attachment deck?

        A plate on a PYLON-borne wing is a fence of free height, and its
        height is a design row the optimiser is free to run up. Nothing said
        where it had to stop. But the deck is a piece of bodywork at a stated
        height, and a fence hanging past it does not hang in air — it hangs
        into the car. So under a pylon mount the plate is bounded by the very
        gap the pylon spans, which is the same statement as "the pylons are
        the tallest thing under the wing".

        Exists on exactly the runs that HAVE both parts, which is the family's
        rule that a margin exists iff its budget does: a deck must be stated
        (:attr:`deck_m`) and pylons must be doing the carrying
        (:attr:`n_pylons`) — the same two halves :attr:`reach_required` asks
        for. It is therefore absent from every published run, because both
        :data:`MOUNTS` layouts are plate-borne and neither states a deck.

        Switchable, because a rule about the shape of a car's bodywork is a
        statement about a car and not a fact about aerodynamics: set
        ``plate_deck_clearance=False`` and the plate may run past the deck
        again, with the run saying so rather than the box hiding it.
        """
        return bool(self.plate_deck_clearance) and self.reach_required

    @property
    def dczdh_required(self) -> bool:
        """Is dCZ/dh MEASURED on this run (and therefore in the breakdown)?

        Two ways to ask for it and one place to answer, so the evaluation,
        the margin and the breakdown cannot disagree about whether the number
        exists: the caller asked for the diagnostic
        (:attr:`dczdh_report`), or the caller declared a ceiling on it
        (:attr:`dczdh_max_per_m`), which cannot be checked without it. The
        ceiling therefore switches the measurement on rather than failing
        against a number that was never taken.
        """
        return bool(self.dczdh_report) or self.dczdh_max_per_m is not None

    @property
    def balance_required(self) -> bool:
        """Is the aero balance CONSTRAINED (as opposed to merely reported)?

        Iff the car states a balance WINDOW. cartrack.CarSpec makes the window
        optional with a derived default (equation 7: the static front weight
        fraction, where the front/rear grip split does not move with speed),
        and that default is a good thing to report and a bad thing to enforce
        by accident: measured on cartrack's reference car with this family's
        published wing, the aero balance is 0.4336 against a derived window of
        (0.48, 0.58), so declaring the margin on the FRAME alone would make
        every lap run start 4.6 points of balance infeasible. That number is
        cartrack's own finding — "a rear wing alone drives the balance
        rearward" — and a finding belongs in the breakdown, not in a margin
        nobody asked for. The balance is therefore always reported when a car
        is stated, and constrained only when a window is.
        """
        return (self.car_spec is not None
                and getattr(self.car_spec, "balance_window", None) is not None)

    @property
    def car_used(self):
        """The :class:`cartrack.CarSpec` in force; cartrack's reference car
        when a track was stated without one."""
        if self.car_spec is not None:
            return self.car_spec
        from . import cartrack
        return cartrack.CarSpec()


def evaluate_car_wing(x: np.ndarray,
                      prob: CarWingProblem | None = None) -> dict:
    """Full evaluation with breakdown; never raises for in-contract failures.

    The published path is ONE solve, exactly the one it always was. Each
    opt-in below buys its own solves and nothing else pays for them:

    * dCZ/dh: two, and only when it was asked for
      (:attr:`CarWingProblem.dczdh_required` — the report switch or a
      ceiling). MEASURED at the published design point (``_x()`` of
      tests/test_carwing_mount_mission.py, ``N_vlm=40``; median of five
      interleaved blocks of 20 calls after a warm-up, spread across blocks
      under 1 %): 1.55 ms with it off against 4.58 ms with it on, a factor
      of 2.97 — two more builds and solves on a path that was one. It was
      unconditional until session 62;
      that made the published path three solves, and the car families are
      the most expensive objects an optimiser here searches;
    * a MountSpec whose attachment is off the aerodynamic centre: as many as
      the windup fixed point needs (one when it is at the quarter chord,
      because the twist is then an exact array of zeros);
    * a track: one per representative speed.

    Every solve at a repeated (speed, ride height) is cached inside the call,
    so a force requirement stated at :attr:`CarWingProblem.V` costs nothing.

    WHAT IS ABSENT WHEN AN OPT-IN IS OFF. The four ``dCZ_dh_*`` keys are not
    in the returned dict at all unless the sensitivity was measured, so a
    reader that wants the number and does not get it raises ``KeyError``
    where it asked. The MARGIN key ``g_dczdh`` is the exception, and follows
    the module's own margin convention instead: every ``g_*`` is present and
    None when its budget is not declared, which is how a caller tells "the
    limit did not bind" from "there is no limit". The two rules cannot
    disagree, because a declared ceiling implies the measurement.
    """
    from .objective import _fail
    from .sizing import check_ar as _sizing_check_ar

    prob = prob or CarWingProblem()
    x = np.asarray(x, dtype=float)
    bnds = prob.bounds
    if x.shape != (bnds.shape[0],):
        return _fail(f"design vector shape {x.shape} != ({bnds.shape[0]},)")
    if np.any(x < bnds[:, 0] - 1e-12) or np.any(x > bnds[:, 1] + 1e-12):
        return _fail(_bounds_reason(x, bnds, prob))

    taper, tw_root, tw_tip, alpha_deg, ep_h, ride = (
        float(v) for v in x[:6])
    # the SPAN this candidate flies, and — when it is free — its AREA. Every
    # b and S below is this candidate's: reading prob.b or prob.S anywhere
    # past here would size the beam, or reference the coefficients, of a wing
    # nobody flew.
    # THE SPAN ROW IS THE OVERALL WIDTH, PLATES INCLUDED. What bounds a rear
    # wing is a regulation, a piece of bodywork or a transporter, and none of
    # those measures the wing and then ignores what is bolted to its tips. A
    # plate at cant 90 with no blend projects nothing and the wing spans the
    # whole row; a plate that LEANS or is BLENDED leaves the wing plane and
    # reaches outboard, and the wing pays for that reach out of its own span
    # — on the same developed line endplate.py and the span-capped aircraft
    # families already use.
    #
    # THIS IS NOT A REFINEMENT, IT CLOSES A HOLE. Measured on the family
    # before this block existed, blend_frac=1.0 — a declared, settable flag —
    # bought +12.3 % CZ (0.7116 -> 0.7993) by flying 1.92 m of hardware
    # inside a stated 1.6 m band. The cant was withheld from the api for
    # exactly that reason (api._CAR_CORE_KEYS) while the blend walked
    # straight through it.
    #
    # The GUARD is what makes the published plate bit-for-bit rather than
    # nearly so. developed_semispan's reach is h*cos(90 deg), i.e. h*6.1e-17,
    # which absorbs in the subtraction at every one of the 8262 points of
    # this family's own default box — but drifts one ULP once h > 0.9*semi,
    # and a caller may widen the height band. So the vertical unblended plate
    # never enters the arithmetic at all; geometry.wing_from_x guards its own
    # developed span the same way.
    b_overall = span_from_x(x, prob.chord_order)
    if prob.blend_frac > 0.0 or prob.endplate_cant_deg != 90.0:
        semi = geometry.developed_semispan(0.5 * b_overall, ep_h,
                                           prob.endplate_cant_deg,
                                           prob.blend_frac, prob.blend_shape)
        if semi <= 0.0:
            return _fail(
                f"the endplates' outboard projection "
                f"({0.5 * b_overall - semi:.3g} m a side) uses up the whole "
                f"{b_overall:.3g} m width: there is no wing left to carry "
                f"them. Blend less, lean less, or shorten the plates.")
    else:
        semi = 0.5 * b_overall
    b = 2.0 * semi                      # the WING's span — beam, VLM, AR
    s_ref = (s_from_x(x, prob.chord_order) if prob.area_free
             else float(prob.S))
    # the solvers are one chordwise panel, and that is only honest over
    # sizing.AR_LIMITS. AREA_BOUNDS_M2 keeps every corner of the DEFAULT box
    # inside it, but a caller may widen the span band, so the limit is checked
    # per candidate and refused in contract rather than assumed.
    #
    # Asked of the WING's span and not of the width: a blended wing that
    # passed an AR gate on b_overall would be passing a gate it does not fly.
    if prob.area_free:
        why = _sizing_check_ar(b, s_ref)
        if why is not None:
            return _fail(why)
    # ...and the ASPECT-RATIO LIMIT THE USER SET. A different question
    # from the validity band, so it is asked whether or not the size is
    # a design variable: a limit that only applied to searched sizes
    # would go quiet exactly when the user typed the number it is about.
    if prob.ar_limits is not None:
        from .sizing import check_ar_limit as _check_ar_limit
        why_ar = _check_ar_limit(b, s_ref, prob.ar_limits)
        if why_ar is not None:
            return _fail(f"size: {why_ar}")
    # the endplate is a LENGTH; the panel builder wants it as a fraction of
    # THIS candidate's semi-span, which is the one conversion in the module.
    # The plate keeps its metres as the wing shortens under it, so the
    # FRACTION rises — endplate.py:752 makes the same conversion.
    ep_frac = ep_h / semi
    chord_coeffs = geometry.chord_coeffs_from_x(x, prob.chord_order,
                                                prob.chord_law)

    def _build_wing(dtw_root: float = 0.0, dtw_tip: float = 0.0):
        """The planform, optionally carrying an elastic twist increment.

        At (0.0, 0.0) the two sums are exact identities in IEEE-754, so the
        rigid wing is the wing this family always built.
        """
        return geometry.Wing(
            b=b, S=s_ref, taper=taper, twist_root_deg=tw_root + dtw_root,
            twist_tip_deg=tw_tip + dtw_tip, tc=prob.tc,
            chord_limits=prob.chord_limits, chord_coeffs=chord_coeffs)

    try:
        wing = _build_wing()
    except ValueError as exc:
        return _fail(f"planform: {exc}")

    mac = float(wing.mac)               # twist-free, so one MAC for every point
    mount = prob.mount_spec
    spec = prob.mount_layout
    # THE SUPPORT STATION, for either way of describing the mount. Both
    # published layouts are plate-borne and differ only in where the sheets
    # grip (MOUNTS), so the plain path has a station too — it is no longer a
    # two-valued beam, and carmount's signed-moment integrator is what
    # resolves an interior support honestly
    y_station = (float(spec["station_frac"]) * 0.5 * b if mount is None
                 else mount.station_m(b))

    polars: dict[float, Any] = {}

    def _polar_for(V: float):
        """The section polar this point flies, and the Re it was read at.

        Cached on the Reynolds number, which depends on the speed and the MAC
        and NOT on the ride height or the twist — so the three solves of a
        central difference share one polar.
        """
        if prob.section_polar is not None:
            return prob.section_polar, None
        if not prob.flown_reynolds:
            return prob.polar_family.at(prob.tc), None
        re = float(prob.rho) * float(V) * mac / float(prob.mu)
        if re not in polars:
            polars[re] = _polar_at_flown_re(prob.tc, re)
        return polars[re], re

    solved: dict[tuple, dict] = {}

    def _point(V: float, ride_h: float) -> dict:
        """One flight point: build, solve, wind up, re-solve, and price it.

        Returns the same in-contract failure dict the module contract uses,
        or a feasible dict of the pieces the breakdown is assembled from.
        """
        key = (float(V), float(ride_h))
        if key in solved:
            return solved[key]
        if mount is not None:
            # a station outboard of the tip, or a wing at or below its own
            # attachment deck: both are designs an optimiser can propose from
            # inside its own box, so both are reasons and not exceptions. Asked
            # PER POINT, because the ride height moves with the heave law and
            # with the sensitivity difference.
            why = mount.violation(b, ride_h)
            if why is not None:
                out = _fail(f"mount: {why}")
                solved[key] = out
                return out
        try:
            pol, re_flown = _polar_for(V)
        except ValueError as exc:
            out = _fail(f"polar family: {exc}")
            solved[key] = out
            return out

        dtw = (0.0, 0.0)
        iters = 0
        last_delta, contraction, stuck = 0.0, None, 0
        while True:
            try:
                w = wing if dtw == (0.0, 0.0) else _build_wing(*dtw)
                model = VLM(
                    w, N=prob.N_vlm, winglet_h_frac=ep_frac,
                    winglet_cant_deg=prob.endplate_cant_deg,   # 90 = the
                    #                           published plate, towards the
                    #                           track; less leans it outboard
                    n_winglet=prob.n_endplate,
                    winglet_chord_follows=prob.endplate_chord_follows,
                    winglet_blend_frac=prob.blend_frac,
                    winglet_blend_shape=prob.blend_shape,
                    a=pol.a_lin, alpha_L0=pol.alpha_L0, V=V,
                    image=ImagePlane(z=ride_h, kind="ground"),  # the track
                )
                res = model.solve(np.deg2rad(alpha_deg))
            except (ValueError, np.linalg.LinAlgError) as exc:
                out = _fail(f"solver failure: {exc}")
                solved[key] = out
                return out

            alpha_eff_deg = np.rad2deg(res.alpha_eff)
            lo, hi = pol.alpha_valid
            if alpha_eff_deg.min() < lo or alpha_eff_deg.max() > hi:
                out = _fail(
                    "effective AoA outside polar validity "
                    "(stall/extrapolation proxy)",
                    alpha_eff_range=(float(alpha_eff_deg.min()),
                                     float(alpha_eff_deg.max())),
                )
                solved[key] = out
                return out

            main = ~res.is_winglet
            y_m, gam_m = res.y[main], res.Gamma[main]
            c_m, wid_m = res.c[main], res.width[main]
            # the RIGID strip load [N/m]: what the solved circulation carries
            load = prob.rho * V * np.abs(gam_m)
            if mount is None:
                break
            # the mount's own footprint on the wing. Exactly 1.0 everywhere
            # unless the user set a suction-side loss, so `load * mods` and
            # `knock` are bit-for-bit identities at the default.
            mods = mount.modifiers(y_m, c_m, b, width=wid_m)
            knock = carmount.lift_knockdown(load, wid_m, mods)
            load = load * mods
            # the windup: a fixed point on the incidence the wing actually
            # flies. The map is (solve -> load -> twist -> apply -> solve),
            # and it is a CONTRACTION exactly as long as the aeroelastic
            # feedback is less than unity — so failing to settle IS the
            # divergence case, and is refused in contract below. The measured
            # contraction ratio is reported beside carmount's single-mode
            # q/q_div, because the two are not the same number and the
            # difference is the reduction's own conservatism (carmount's
            # divergence_q uses the 2-D lift slope, which a finite wing in
            # ground effect does not deliver).
            m_span = carmount.torsion_moment(y_m, load, c_m,
                                             mount.x_attach_frac)
            phi = carmount.torsion_twist(y_m, m_span, y_station, mount.gj_nm2)
            nxt = _twist_projection(y_m, phi, c_m * wid_m, b)
            delta = max(abs(nxt[0] - dtw[0]), abs(nxt[1] - dtw[1]))
            contraction = (delta / last_delta if last_delta > 0.0 else None)
            last_delta = delta
            if delta <= float(prob.windup_tol_deg):
                break
            stuck = (stuck + 1 if (contraction is not None
                                   and contraction >= 1.0) else 0)
            iters += 1
            if stuck >= 2 or iters > int(prob.windup_max_iter):
                q_here = 0.5 * prob.rho * V**2
                q_div = carmount.divergence_q(
                    mount.gj_nm2, b, s_ref, mac, pol.a_lin, mount.e_frac,
                    y_station)
                why = (
                    f"the re-solve has stopped contracting (feedback gain "
                    f"{contraction:.4g} >= 1 on {stuck} consecutive rounds): "
                    f"the wing winds itself up without limit, which is "
                    f"divergence"
                    if stuck >= 2 else
                    f"the windup had not settled after "
                    f"{int(prob.windup_max_iter)} re-solves — still moving "
                    f"{delta:.3g} deg against a "
                    f"{float(prob.windup_tol_deg):.3g} deg tolerance, at a "
                    f"measured feedback gain of {contraction:.4g} per round. "
                    f"It is contracting, so a wing meant to be flown this "
                    f"close to divergence needs a larger windup_max_iter")
                out = _fail(
                    f"torsional windup: {why}. The mount sits "
                    f"{mount.e_frac:+.3g} chords aft of the aerodynamic "
                    f"centre, which puts the single-mode divergence pressure "
                    f"at {q_div:.5g} Pa against {q_here:.5g} Pa flown",
                    g_divergence=carmount.divergence_margin(q_here, q_div),
                    q_div_Pa=float(q_div), windup_residual_deg=float(delta),
                    windup_contraction=(None if contraction is None
                                        else float(contraction)),
                    diverged=bool(stuck >= 2),
                )
                solved[key] = out
                return out
            dtw = nxt

        # --- drag. Same three terms in the same order as the published run
        CDp = float(np.sum(pol.cd(alpha_eff_deg) * res.c * res.width) / res.S)
        if mount is None:
            # NO PYLON on either published layout — the plates are the load
            # path in both (MOUNTS). What the inboard layout pays for instead
            # is its EXTRA SHEETS: two more plates of the wing's local chord,
            # standing the endplate's own height, charged with the same
            # reduced-order flat-plate model the pylon used (_strut_cd0 is
            # wetted area x cf either way, which is why it is reused rather
            # than reimplemented under a new name).
            #
            # Charged at the LOCAL chord at the grip station, not the root
            # chord: a sheet is as long as the wing is where it grips it, and
            # on a tapered planform those differ.
            c_station = float(w.chord(np.array([min(y_station, 0.5 * b)]))[0])
            cd0_struts = _strut_cd0(spec["n_sheets"], ep_h, c_station,
                                    s_ref, prob.strut_cf)
            pylon = None
            cd_junction = 0.0
            if prob.junction_drag and spec["n_junctions"]:
                # ...and the corners those sheets make with the wing, priced
                # at the same local chord for the same reason
                cd_junction = junction.junction_cd(
                    prob.tc, c_station, s_ref,
                    radius=0.0, n_junctions=spec["n_junctions"])
        else:
            # the candidate's OWN plate height and grip chord, because a
            # sheet is as tall as the plate and as long as the wing is where
            # it grips it — both design variables. Handing the spec neither
            # would price the sheets off its own defaults, which is how the
            # continuum path came to charge nothing for the inboard grip
            pylon = mount.pylon_cd(
                ride_h, s_ref, prob.rho, V, prob.mu,
                sheet_height_m=ep_h,
                sheet_chord_m=float(w.chord(np.array(
                    [min(y_station, 0.5 * b)]))[0]))
            cd0_struts = float(pylon["CD"])
            cd_junction = (mount.junction_cd(w, s_ref, tc=prob.tc)
                           if prob.junction_drag else 0.0)
        CD = res.CDi + CDp + cd0_struts + cd_junction
        CZ = res.CL                # model lift IS the downforce (mirrored frame)
        if mount is not None:
            # the footprint's span-integrated lift loss (carmount's own upper
            # bound: the wake is not re-solved). Exactly 0.0 when off.
            CZ = CZ * (1.0 - knock)
        if not np.isfinite(CD) or CD <= 0.0:
            out = _fail("non-finite drag")
            solved[key] = out
            return out

        # --- structure: same load, different beam. On the SPAN this candidate
        # flies — a wider wing carries the same downforce on a longer arm,
        # which is the half of the span trade the aerodynamics never shows
        # ONE beam for both ways of describing the mount, and always the
        # SIGNED moment: at an interior station (which "inboard" is) the
        # moment changes sign at the support, and integrating a MAGNITUDE
        # gives a plausible, wrong deflection there — carwing.deflection_index
        # cannot express it at all, since it only knows the two ends.
        m_signed = carmount.beam_moment(y_m, load, y_station)
        moment = np.abs(m_signed)
        defl = (carmount.beam_deflection_index(y_m, m_signed, y_station)
                / prob.ei_nm2)
        sense = carmount.bending_sense(m_signed)

        out = {
            "feasible": True, "reason": "",
            "V": float(V), "ride_height_m": float(ride_h),
            "CZ": float(CZ), "CD": float(CD), "CDi": float(res.CDi),
            "CDp": CDp, "cd0_struts": cd0_struts, "CD_junction": cd_junction,
            "CL_model": float(res.CL), "e": float(res.e), "AR": float(res.AR),
            "res": res, "wing": w, "polar": pol, "Re_flown": re_flown,
            "y": y_m, "load": load, "chord": c_m, "width": wid_m,
            "moment": moment, "sense": sense, "deflection_m": float(defl),
            "dtw_deg": dtw, "windup_iters": int(iters),
            "windup_contraction": contraction,
            "knockdown": (0.0 if mount is None else float(knock)),
            "pylon": pylon, "phi_rad": (None if mount is None else phi),
        }
        solved[key] = out
        return out

    base = _point(prob.V, ride)
    if not base["feasible"]:
        return base                      # the published failure dict, verbatim

    res, w = base["res"], base["wing"]
    CZ, CD = base["CZ"], base["CD"]
    y_m, load, moment = base["y"], base["load"], base["moment"]
    defl = base["deflection_m"]
    # --- the forces this candidate actually makes. Both are q S x a
    # coefficient on THIS candidate's area, which is the whole reason a free
    # area needs them: a coefficient budget on a moving reference area is not
    # an allowance on anything the car can feel.
    q = 0.5 * prob.rho * prob.V**2
    downforce_n = float(q * s_ref * CZ)
    drag_n = float(q * s_ref * CD)

    def _force_at(V_req, which: str):
        """``(force [N], speed)`` for a requirement stated AT a speed.

        A force is q(V) S C(V), so BOTH factors move with the speed and the
        coefficient is re-solved there rather than referred by a q ratio. At
        the family's own speed the cache returns the base point and the force
        is the base one, to the bit.
        """
        if V_req is None:
            return (downforce_n if which == "cz" else drag_n), float(prob.V)
        V_req = float(V_req)
        h = _ride_at(V_req)
        pt = _point(V_req, h)
        if not pt["feasible"]:
            return None, V_req
        qq = 0.5 * prob.rho * V_req**2
        return float(qq * s_ref * pt["CZ" if which == "cz" else "CD"]), V_req

    def _ride_at(V_req: float) -> float:
        """The ride height at a stated speed: the heave law's, or the stated
        one. With no car (or a rigid one) this is ``ride`` bit-for-bit."""
        if prob.car_spec is None and prob.track_spec is None:
            return ride
        return float(prob.car_used.ride_height_at(ride, float(V_req)))

    # --- ride-height sensitivity, IFF it was asked for. dCZ/dh is the wing's
    # exposure to the one thing about its operating point the car does not
    # hold still, and it is a diagnostic: it costs two more builds and solves
    # (x2.97 on the whole evaluation, measured in this function's docstring),
    # so it is charged to the run that wants it and to no other.
    # The low side is refused OUTRIGHT below the track rather than left to the
    # solver: the image plane is side-agnostic, so a negative ride height
    # solves happily as a wing above a plane BELOW it and would return a
    # difference quotient of two unrelated flows.
    dczdh = dczdh_scheme = dczdh_why = None
    step = float(prob.dczdh_step_m)
    if prob.dczdh_required:
        up = _point(prob.V, ride + step)
        dn = (_point(prob.V, ride - step) if ride - step > 0.0 else
              _fail(f"a ride height of {ride - step:.4g} m is below the track"))
        if up["feasible"] and dn["feasible"]:
            dczdh = (up["CZ"] - dn["CZ"]) / (2.0 * step)
            dczdh_scheme, dczdh_why = "central", ""
        elif up["feasible"]:
            dczdh = (up["CZ"] - CZ) / step
            dczdh_scheme = "forward"
            dczdh_why = f"h - {step:g} m: {dn['reason']}"
        elif dn["feasible"]:
            dczdh = (CZ - dn["CZ"]) / step
            dczdh_scheme = "backward"
            dczdh_why = f"h + {step:g} m: {up['reason']}"
        else:
            dczdh, dczdh_scheme = None, "unavailable"
            dczdh_why = (f"h + {step:g} m: {up['reason']}; "
                         f"h - {step:g} m: {dn['reason']}")

    # --- the lap, when a circuit was stated. The wing is evaluated AT the
    # speeds the lap spends its time at (cartrack's E5 interface) and the
    # coefficients travel back as laws of speed, rather than one point being
    # referred to another by a q ratio.
    lap = track_points = cz_law = None
    if prob.track_spec is not None:
        from . import cartrack
        car = prob.car_used
        seed = cartrack.representative_points(
            CZ * s_ref, CD * s_ref, car, prob.track_spec,
            n_points=int(prob.track_points), ride_height_m=ride)
        if not seed["feasible"]:
            return _fail(f"track: {seed['reason']}")
        rows = []
        for pt in sorted(seed["points"], key=lambda p: p["V"]):
            got = _point(pt["V"], pt.get("ride_height_m", ride))
            if not got["feasible"]:
                return _fail(
                    f"the lap flies this wing at {pt['V']:.4g} m/s and "
                    f"{pt.get('ride_height_m', ride):.4g} m: {got['reason']}")
            rows.append({"V": float(pt["V"]), "weight": float(pt["weight"]),
                         "ride_height_m": float(pt.get("ride_height_m", ride)),
                         "CZ": got["CZ"], "CD": got["CD"],
                         "cz_a_m2": got["CZ"] * s_ref,
                         "cd_a_m2": got["CD"] * s_ref,
                         "Re_flown": got["Re_flown"]})
        Vs = np.array([r["V"] for r in rows], dtype=float)
        cza = np.array([r["cz_a_m2"] for r in rows], dtype=float)
        cda = np.array([r["cd_a_m2"] for r in rows], dtype=float)
        # linear in V between the evaluated points and CLAMPED outside them:
        # np.interp holds the end values, so the lap never extrapolates a
        # coefficient into a speed the wing was not flown at
        cz_law = lambda V: float(np.interp(V, Vs, cza))   # noqa: E731
        lap = cartrack.lap_time(cz_law,
                                lambda V: float(np.interp(V, Vs, cda)),
                                car, prob.track_spec)
        if not lap["feasible"]:
            return _fail(f"lap: {lap['reason']}")
        track_points = rows

    # --- the aero balance, reported whenever a car is stated and constrained
    # only when that car states a window (CarWingProblem.balance_required)
    balance = None
    if prob.car_spec is not None or prob.track_spec is not None:
        from . import cartrack
        # asked at the lap's mean speed when there is a lap, off the SAME
        # coefficient law the lap itself flew — a second solve there would be
        # a third opinion about one wing
        v_bal = (float(lap["V_mean"]) if lap is not None else float(prob.V))
        cz_a_bal = (cz_law(v_bal) if cz_law is not None else CZ * s_ref)
        balance = cartrack.balance_margin(cz_a_bal, prob.car_used, v_bal)
        if not balance["feasible"]:
            return _fail(f"balance: {balance['reason']}")

    # --- the margins this problem declares, in constraint_labels order. Each
    # optional one is present iff its budget is set, so a run that switched a
    # budget off returns a SHORTER vector rather than a satisfied placeholder
    # — the difference between "the limit did not bind" and "there is no
    # limit", which a placeholder would erase.
    margins = []
    g_drag = g_drag_n = g_down = g_reach = g_dczdh = g_balance = None
    g_clear = None
    if prob.CD_budget is not None:
        g_drag = float(prob.CD_budget - CD)
        margins.append(g_drag)
    drag_req_n, drag_req_v = drag_n, float(prob.V)
    if prob.drag_budget_n is not None:
        drag_req_n, drag_req_v = _force_at(prob.drag_budget_v_ms, "cd")
        if drag_req_n is None:
            return _fail(
                f"the drag budget is stated at {drag_req_v:.4g} m/s and the "
                f"wing cannot be flown there: "
                f"{_point(drag_req_v, _ride_at(drag_req_v))['reason']}")
        g_drag_n = float(1.0 - drag_req_n / prob.drag_budget_n)
        margins.append(g_drag_n)
    g_defl = float(1.0 - defl / prob.deflection_limit_m)
    margins.append(g_defl)
    down_req_n, down_req_v = downforce_n, float(prob.V)
    if prob.downforce_min_n is not None:
        down_req_n, down_req_v = _force_at(prob.downforce_min_v_ms, "cz")
        if down_req_n is None:
            return _fail(
                f"the downforce floor is stated at {down_req_v:.4g} m/s and "
                f"the wing cannot be flown there: "
                f"{_point(down_req_v, _ride_at(down_req_v))['reason']}")
        g_down = float(down_req_n / prob.downforce_min_n - 1.0)
        margins.append(g_down)
    if prob.reach_required:
        g_reach = float(ride / prob.deck_m - 1.0)
        margins.append(g_reach)
    if prob.dczdh_max_per_m is not None:
        if dczdh is None:
            return _fail(
                f"a ride-height sensitivity ceiling is declared and dCZ/dh "
                f"could not be measured at either side of h = {ride:.4g} m "
                f"({dczdh_why})")
        g_dczdh = float(1.0 - abs(dczdh) / prob.dczdh_max_per_m)
        margins.append(g_dczdh)
    if prob.balance_required:
        g_balance = float(balance["g"])
        margins.append(g_balance)
    if prob.plate_clears_deck:
        # THE PYLON IS THE TALLEST THING UNDER THE WING. The pylon spans the
        # gap from the wing to the deck (ride - deck); the plate hangs the
        # same way off the same wing, and past that gap there is bodywork
        # rather than air. Written as a fraction of the gap so the margin is
        # scale-free like every other one here, and so a wing that raises
        # itself buys plate rather than being clipped.
        #
        # It bites, and it bit at the box centre: the gap there was 0.125 m
        # and the plate row's own centre is 0.125 m, so the two were exactly
        # equal and the whole top half of the plate row was reaching into the
        # car — buying score the whole way (measured: 34.918 at a plate of
        # 0.125 m against 36.320 at 0.200 m, which is 75 mm inside the
        # bodywork).
        #
        # LAST in the vector, so every margin that existed before it keeps
        # the index it always had.
        # ...compared on the plate's VERTICAL REACH, not on its arc length.
        # The gap is a vertical distance, so the plate's arc is the wrong
        # number to hold against it: a canted or blended plate spends part of
        # its length going sideways and gets less far down (h sin(cant) for a
        # straight lean), so charging the arc would refuse a leaning plate
        # that clears the deck comfortably.
        gap = float(ride) - float(prob.deck_m)
        reach_down = geometry.winglet_tip_height(
            ep_h, prob.endplate_cant_deg, prob.blend_frac, prob.blend_shape)
        g_clear = (float((gap - reach_down) / gap) if gap > 0.0
                   else -1.0 - float(reach_down))
        margins.append(g_clear)

    score = {"cz": float(CZ), "downforce": downforce_n,
             "efficiency": float(CZ / CD),
             # MINUS the drag: the harness maximises and drag is won by being
             # small. Negation rather than 1/D or -log D for the reason the
             # lap gives — it is the only decreasing transform that leaves the
             # units and the increments alone, so a score difference of 1.0 IS
             # a newton at every drag level.
             "cd": -float(CD), "drag": -float(drag_n),
             "laptime": (None if lap is None
                         else -float(lap["lap_time_s"])),
             # ADDED, not traded — the one entry that wants both terms big.
             # Both are FORCES, so it is free-area-safe in the sense 'cz' and
             # 'cd' are not; it rides the area row's ceiling for a physical
             # reason rather than a referencing artefact (see CAR_OBJECTIVES).
             "downforce_plus_drag": float(downforce_n) + float(drag_n),
             }[prob.objective]

    out = {
        "feasible": True, "reason": "",
        "score": float(score), "CZ": float(CZ),
        "objective": prob.objective,
        "objective_label": CAR_OBJECTIVES[prob.objective],
        "g": margins,
        "g_drag": g_drag, "g_deflection": g_defl,
        "g_drag_force": g_drag_n, "g_downforce": g_down,
        "constraint_labels": list(prob.constraint_labels),
        "CL_model": float(base["CL_model"]), "CD": float(CD),
        "CDi": float(base["CDi"]), "CDp": base["CDp"],
        "cd0_struts": base["cd0_struts"], "CD_junction": base["CD_junction"],
        "efficiency": float(CZ / CD),
        "e": float(base["e"]), "AR": float(base["AR"]),
        "alpha_deg": alpha_deg,
        "ride_height_m": ride, "ride_height_over_b": float(ride / b),
        "b_m": float(b), "S_m2": float(s_ref),
        # ...and the two numbers that say what "b_m" just cost. The design
        # box row is the OVERALL width; b_m is the wing left after the plates
        # have taken their outboard reach out of it. Both are reported so a
        # blended answer can never be read as a wing that grew for free.
        "overall_width_m": float(b_overall),
        "endplate_projection_m": float(0.5 * b_overall - semi),
        "area_free": bool(prob.area_free),
        "endplate_h_m": float(ep_h), "endplate_h_frac": float(ep_frac),
        "mount": prob.mount if mount is None else mount.kind,
        "mount_label": (spec["label"] if mount is None
                        else mount.kind_spec["label"]),
        "supports": (spec["supports"] if mount is None
                     else _station_name(y_station, b)),
        # WHICH WAY UP THE NUMBERS ARE. Everything in this dict is in the
        # MODEL frame — the car flipped over — so an endplate that reaches
        # for the deck has a POSITIVE height here and the track is the plane
        # above the wing. The geometry views read this key and turn the
        # PICTURE the right way up (the wing on top of its endplates,
        # geometry.is_mirrored); the numbers beside them stay in the frame
        # they were solved in, which is the frame every margin above is in.
        "frame": geometry.MIRRORED_FRAME,
        "M_max_Nm": float(np.max(moment)),
        "M_root_Nm": float(moment[np.argmin(np.abs(y_m))]),
        "deflection_m": float(defl),
        "downforce_N": downforce_n, "drag_N": drag_n,
        "q_Pa": float(q),
        "Re_mac": float(prob.rho * prob.V * w.mac / prob.mu),
        "polar": getattr(base["polar"], "name", "unknown"),
        "y": y_m, "load_Npm": load, "moment_Nm": moment,
        "wing": w, "vlm": res,
        # --- the opt-ins, always keyed so a reader never has to ask whether
        # the family was built with them
        "bending_sense": base["sense"],
        "Re_flown": base["Re_flown"], "flown_reynolds": bool(prob.flown_reynolds),
        # ...except the sensitivity, whose four keys are ABSENT when it was
        # not measured (see this function's docstring): None here would read
        # as "measured, and there is no answer", which is what
        # ``dCZ_dh_scheme == "unavailable"`` means and is a different fact.
        # ``g_dczdh`` stays, under the margin convention.
        "g_dczdh": g_dczdh,
        "deck_height_m": prob.deck_m,
        "pylon_length_m": (None if prob.deck_m is None
                           else float(max(ride - prob.deck_m, 0.0))),
        "g_pylon_reach": g_reach, "g_mount_clearance": g_clear,
        "mount_clearance_m": (
            None if prob.deck_m is None else float(
                ride - prob.deck_m - geometry.winglet_tip_height(
                    ep_h, prob.endplate_cant_deg, prob.blend_frac,
                    prob.blend_shape))),
        "drag_req_N": drag_req_n, "drag_req_V_ms": drag_req_v,
        "downforce_req_N": down_req_n, "downforce_req_V_ms": down_req_v,
        "lap_time_s": (None if lap is None else float(lap["lap_time_s"])),
        "lap": lap, "track_points": track_points,
        "aero_balance": (None if balance is None
                         else float(balance["aero_balance"])),
        "balance": balance, "g_balance": g_balance,
    }
    if prob.dczdh_required:
        out.update({
            "dCZ_dh_per_m": dczdh, "dCZ_dh_scheme": dczdh_scheme,
            "dCZ_dh_step_m": step, "dCZ_dh_reason": dczdh_why,
        })
    if mount is not None:
        out.update(_mount_report(prob, mount, base, y_station, b, s_ref, mac))
    return out


def _station_name(y_station: float, b: float) -> str:
    """The published ``supports`` vocabulary, extended to the continuum.

    The two named layouts are the two ends of the station continuum, so they
    keep their names — a reader (and endplate.py's own copy of this key) sees
    ``centre`` and ``ends`` exactly where it always did — and everything in
    between is honestly called what it is.
    """
    semi = 0.5 * float(b)
    if float(y_station) <= 0.0:
        return "centre"
    if float(y_station) >= semi - 1e-12:
        return "ends"
    return "station"


def _mount_report(prob, mount, base, y_station: float, b: float,
                  s_ref: float, mac: float) -> dict:
    """What a :class:`carmount.MountSpec` adds to the breakdown.

    Kept out of the assembly above because none of it exists on the published
    path: a reader of a published run should not have to skip a block of
    keys that are all None.
    """
    pol = base["polar"]
    q = 0.5 * prob.rho * base["V"]**2
    q_div = carmount.divergence_q(mount.gj_nm2, b, s_ref, mac, pol.a_lin,
                                  mount.e_frac, y_station)
    phi = base["phi_rad"]
    fit = base["dtw_deg"]
    return {
        "mount_kind": mount.kind, "mount_side": mount.side,
        "y_station_m": float(y_station),
        "y_station_frac": float(y_station / (0.5 * b)),
        "n_pylons": int(mount.n_pylons), "n_junctions": int(mount.n_junctions),
        "pylon": base["pylon"],
        "pylon_cf": (None if base["pylon"] is None
                     else float(base["pylon"]["Cf"])),
        "pylon_re": (None if base["pylon"] is None
                     else float(base["pylon"]["Re"])),
        "x_attach_frac": float(mount.x_attach_frac),
        "e_frac": float(mount.e_frac),
        "q_div_Pa": float(q_div),
        "g_divergence": float(carmount.divergence_margin(q, q_div)),
        "amplification": float(carmount.twist_amplification(q, q_div)),
        "windup_root_deg": float(fit[0]), "windup_tip_deg": float(fit[1]),
        "windup_max_deg": float(np.rad2deg(np.max(np.abs(phi)))),
        "windup_iters": int(base["windup_iters"]),
        "windup_contraction": base["windup_contraction"],
        "windup_fit_residual_deg": _twist_fit_residual_deg(
            base["y"], phi, fit, b),
        "mount_lift_knockdown": float(base["knockdown"]),
        "suction_loss": float(mount.suction_loss),
    }


def fg_car_wing(x: np.ndarray, prob: CarWingProblem | None = None
                ) -> tuple[float, list]:
    """Constrained-harness callable: ``(score, margins)``.

    The failure vector's WIDTH follows the problem's own declaration, not the
    published pair: a run with the coefficient budget off returns three
    margins where the default returns two, and a penalty returned at the wrong
    width is a shape error inside the optimiser rather than a bad score
    (``hydrofoil.fg_hydrofoil_winglet`` under its draught cap, same rule).
    """
    prob = prob or CarWingProblem()
    out = evaluate_car_wing(x, prob)
    if not out["feasible"]:
        return PENALTY, [G_FAIL] * prob.n_constraints
    return float(out["score"]), [float(g) for g in out["g"]]


#: the two objectives a car-wing FRONT is searched over, in the order the
#: front's columns carry them. Both are FORCES and both are stated
#: higher-better, which is the convention ``optimize.mobo`` maximises in:
#: downforce as it stands, drag NEGATED.
#:
#: WHY FORCES AND NOT COEFFICIENTS, which is the same argument ``drag`` makes
#: against ``cd`` one dict up: the reference area is a design variable on
#: every car family, so a coefficient front would be a front over a quantity
#: referenced to the very thing being searched — CZ maximised by shrinking the
#: wing and CD minimised by growing it, on the same axis pair. In newtons the
#: two axes are things the car can feel and the front is over the trade the
#: user actually has.
#:
#: WHY A FRONT AT ALL. Every scalar objective in :data:`CAR_OBJECTIVES` picks
#: ONE point of this trade and cannot be pointed at another without changing
#: the question. Measured over a 14 803-design grid of the endplate family
#: whose front has 618 points, sweeping each scalar form's own knob reaches
#: 2.6-6.3 % of it (weights), 38.5-98.5 % (an epsilon-constraint under a
#: floor) — and the front is strongly CONCAVE, so a linear weight is blind to
#: a region no normalisation recovers. A front needs no weight, no floor and
#: no circuit: it returns the trade and the user picks off it.
CAR_PARETO_OBJECTIVES: tuple = ("downforce_N", "drag_N")

#: how each column is turned into the maximising convention.
CAR_PARETO_SIGNS: tuple = (+1.0, -1.0)


def car_pareto_point(out: dict) -> list:
    """One evaluation's objective vector, higher-better, or None if refused."""
    if not out.get("feasible") or CAR_PARETO_OBJECTIVES[0] not in out:
        return None
    return [sign * float(out[key])
            for key, sign in zip(CAR_PARETO_OBJECTIVES, CAR_PARETO_SIGNS)]


def make_car_pareto_fg(prob, evaluate):
    """``fg(x) -> (y_vector, margins)`` for :func:`optimize.mobo.run_mobo_constrained`.

    ``evaluate`` is the family's own evaluator, passed in rather than looked
    up, because the three car families each own theirs and a table here would
    be a fourth place for them to disagree.

    A refusal returns the SCALAR penalty in both columns and the failure
    margin vector at the problem's own declared width — the same contract
    ``fg_car_wing`` honours, extended componentwise. The penalty's scale is
    wrong for forces (it is -100 against a downforce of some hundreds of
    newtons), which is exactly why a front run belongs under the scale-aware
    refusal policy; ``mobo`` takes that as its ``refusal=`` argument and the
    sentinel is then imputed rather than believed.
    """
    n_g = int(prob.n_constraints)

    def fg(x):
        out = evaluate(x, prob)
        y = car_pareto_point(out)
        if y is None:
            return [PENALTY, PENALTY], [G_FAIL] * n_g
        return y, [float(g) for g in out["g"]]

    return fg


def car_pareto_reference(prob, evaluate) -> list:
    """The hypervolume reference point: the BOX CENTRE's own (F_z, -D).

    Fixed before the run and read off the incumbent, so a design contributes
    volume only if it beats the wing the user already has on BOTH axes, and
    two runs of the same problem produce comparable hypervolumes. A reference
    read off the run itself would not — ``api.pareto_airfoil`` makes the same
    choice for the same reason, against the seed section there.

    Returns None when the box centre does not fly, and the caller then
    refuses to start rather than inventing a reference out of a failed
    evaluation.
    """
    b = np.asarray(prob.bounds, dtype=float)
    return car_pareto_point(evaluate(0.5 * (b[:, 0] + b[:, 1]), prob))


def fg_car_wing_pareto(x: np.ndarray, prob: "CarWingProblem | None" = None):
    """:func:`fg_car_wing`'s two-objective twin."""
    prob = prob or CarWingProblem()
    return make_car_pareto_fg(prob, evaluate_car_wing)(x)


def compare_mounts(x: np.ndarray, prob: CarWingProblem | None = None) -> dict:
    """Same design, both mount layouts — the direct answer to "ends or centre".

    Returns ``{mount: breakdown}``; the aerodynamics differ only through the
    mount's own drag, so the interesting columns are the structural ones.

    Refused on a problem carrying a :class:`carmount.MountSpec`, and not out
    of tidiness: this compares the two ENDS of the station continuum, so on a
    problem whose mount is somewhere else it would have to either ignore the
    mount it was handed or invent two new ones. Both would answer a question
    nobody asked. ``carmount.published_layout('tips' | 'inboard')`` builds the
    two ends as MountSpecs if the two ends are what you want.
    """
    from dataclasses import replace

    prob = prob or CarWingProblem()
    if prob.mount_spec is not None:
        raise ValueError(
            "compare_mounts asks the two-valued mount question, and this "
            "problem states its mount as a carmount.MountSpec — a station, a "
            "kind and a chordwise attachment. Compare two MountSpecs by "
            "evaluating each (carmount.published_layout builds the two "
            "published ends), or drop mount_spec")
    return {m: evaluate_car_wing(x, replace(prob, mount=m)) for m in MOUNTS}
