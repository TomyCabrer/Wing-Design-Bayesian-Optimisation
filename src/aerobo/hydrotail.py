"""Hydrofoil + ELEVATOR under a free surface — the foiling craft, trimmed.

Why this module exists
----------------------
``hydrofoil.py`` designs one lifting surface under the free surface, trimmed
in LIFT only: it can say how much drag a foil makes carrying a weight without
cavitating, and nothing at all about whether the craft sits at that
incidence. Every real foiling craft (Moth, foiling cat, hydrofoil ferry)
carries a second surface aft — the elevator / rear stabiliser — and its size
and arm are exactly what the designer trades against the main foil.

That gap is closed the way ``wingtail.py`` closed it in air: the stabiliser
goes INTO the same solve rather than beside it. The pieces already existed
and are simply composed here for the first time —

* ``vlm.ImagePlane(kind="free_surface")``: the high-depth-Froude free-surface
  image (hydrofoil.py's derivation, same-sign mirror);
* ``vlm.TailSurface``: a second surface in the SAME influence matrix, whose
  incidence is the second trim unknown, so trim in (alpha, i_t) is the
  closed-form 2x2 solve and the neutral point comes off the same basis
  solutions;
* an optional tip device on the main foil, with the SIGNED cant band the
  water problems use (+ up towards the surface, − down away from it);
* ``hydrofoil.cavitation_margin_panels``: the cavitation margin evaluated at
  every panel's OWN submergence — which now includes the stabiliser's, and
  that is the physically interesting part (see below).

so this is one influence matrix over main foil + tip device + stabiliser +
their images, one Trefftz plane, one trim, one cavitation sweep.

What is genuinely new here as a DESIGN QUESTION
-----------------------------------------------
The stabiliser sits BELOW the main foil (negative z here, since +z is up
towards the surface) — but only just, and that is a change. THE CRAFT IS
FLAT: a windfoil, a wingfoil and a Moth are a front wing, a strut and a
stabiliser bolted to ONE horizontal fuselage, so the two lifting surfaces
are coplanar. The unstated separation is therefore Z_T_FRAC_DEFAULT*b =
tail.DZ_GRID_FRAC*b = 0.01 b (12 mm under the shipped foil), which is a
SOLVER line and not a design one: at dz = 0 the stabiliser's collocation
points lie in the foil's trailing sheet and the L/D moves 6.6 % with the
panel count, against 0.27 % at 0.01 b. Depth below the foil still BUYS
cavitation margin — the static head at the stabiliser is rho g (depth - z)
with z < 0 — so the vertical placement remains a real trade, offered as a
design variable (``free_height``) and typeable exactly (``z_t_fixed``,
including 0.0). What it is no longer is 0.05 b by default, which was the
air families' MEASUREMENT height quietly acting as this craft's layout.

THE STRUT IS A SURFACE, AND IT STANDS BETWEEN THEM
--------------------------------------------------
It used to be a drag term with no position: ``hydrofoil._mast_cd0`` charged
two sides of depth x chord at a constant flat-plate Cf, and ``fin.mast``
reported its quarter chord AT THE FOIL (x = 0) because there was nothing
else to say. Three things were wrong with that on a craft that is mostly
mast. It stood in the wrong place — a real mast is bolted to the fuselage
BETWEEN the two wings (``X_MAST_FRAC``, and ``free_mast_station`` searches
it). Its section did nothing — ``fin_tc``, asked on stage 2.7, reached no
number, because a wetted-area charge has no form factor. And it carried no
SIDE FORCE, so ``rig.RigLoads.side_n`` — 274 N measured on a windfoil, 615 N
on a kitefoil — was a load nothing in the craft resisted: no leeway, no
induced drag for it, no upwind case. ``hydrofoil.strut_report`` is those
three, and ``strut_model`` (on here, off in the single-foil family) is the
switch back to the published charge.

The cavitation constraint is what makes the pair interesting: the main foil
runs at the lift the craft's weight demands, the stabiliser runs at whatever
incidence TRIMS it, and either can be the surface that cavitates first. A
planar single-foil model cannot see that at all.

The stabiliser is a SURFACE, not a fitting
------------------------------------------
Its area and its arm were the only things about it a designer could move
here, which said — without meaning to — that the elevator is a fixed
rectangle bolted on aft. It is not: it is a small wing, and the same three
questions the main foil answers are open on it. So it carries, exactly as
``wingtail.py``'s tail does in air:

* its own SECTION (``polar_tail``). None keeps the documented default — the
  stabiliser flies the main foil's section — so every published run is
  bit-for-bit what it was;
* its own PLANFORM (``tail_free``): taper, aspect ratio and washout. Its
  incidence stays the trim unknown, so the twist freedom is the washout — a
  root twist would be that same angle twice;
* its own TIP DEVICE (``tail_winglet``), with the water problems' SIGNED
  cant band, panelised by the same VLM constructor as the foil's;
* its own CHORD LAW, one block per surface, wherever the foil carries one.

The RIG is a LOAD, not a shape (``rig.py``)
-------------------------------------------
Everything above designs an APPENDAGE. What drives a windfoil, a wingfoil or
a kitefoil is a rig, and it pushes at a centre of effort one to three metres
ABOVE the water while the resistance it balances acts a few tenths of a metre
below it. That is a COUPLE — ~399 N.m for a windfoil, re-derived three
independent ways in ``rig.py`` — and it is the largest longitudinal moment on
the craft. A kite additionally carries 21-46 % of the rider's weight
vertically, so the foil is otherwise being sized to a load it does not carry.

``rig`` (``None`` = no rig, every published run bit-for-bit) closes both. The
couple enters ``cm_ac_model`` and NOTHING else, because ``cm_ac`` is
independent of ``(alpha, i_t)`` and therefore rides in the constant of the
affine trim map: the Jacobian, ``x_np`` and the static margin are untouched
AT FIXED GEOMETRY, by construction.

That qualifier is load-bearing and it is measured, not hedging. ``cm_ac``
never enters the influence matrix, so for one fixed aeroplane the panel
normals, ``_Gam1`` and ``neutral_point()`` are bit-identical with the rig
and without it — pin ``tail_inverted`` and the two agree to the last bit
(``test_the_couple_leaves_the_neutral_point_and_the_margin_alone``). But the
rig flips the SIGN of the stabiliser's load, and two things in this family
follow that sign and are geometry:

* the section MOUNTING (``tail.orient_section``). A surface that pushes down
  mounts its camber the other way, and the mirrored ``alpha_L0`` is baked
  into the panel normals — so it is a different aeroplane with a legitimately
  different neutral point. Measured on ``hydrofoil + elevator`` with the
  driven rig: x_np 0.1221206762291545 -> 0.12225796485175514 at the box
  centre (+0.11 % MAC) and 0.1978401601187913 -> 0.19796415303466938 at
  lo + 0.75(hi - lo) (+0.10 % MAC). With the mounting pinned, EXACTLY zero.
* the stabiliser's TIP DEVICE, where it follows its load
  (``tail_winglet_follow``): the device swaps sides outright and the solve is
  re-flown before ``neutral_point()`` is read. Measured on
  ``hydrofoil + elevator [designed elevator + tip device]``, same rig:
  x_np 0.14219956586471855 -> 0.1432741908779214, +0.88 % MAC, an order of
  magnitude more than the mounting alone.

So the honest statement is the narrow one: the COUPLE moves neither, and
what moves them is the aeroplane changing shape underneath it.

The relief enters ``L_required``, the single author of how
much lift the foil owes. What the rig must NOT do is move ``x_cg`` — that
substitution reproduces the trim to machine precision and turns the stability
constraint into a margin about a CG the craft does not have (SM -0.2276 ->
+3.2837 at the published centre, i.e. satisfied everywhere). ``rig.py``
derives it; ``tests/test_rig_couple_is_a_moment_not_a_cg_move.py`` uses it as
an oracle and nothing else does.

Design vector (blocks; the chord law LAST, package rule)
--------------------------------------------------------
    [taper, twist_root_deg, twist_tip_deg, tc, depth_m, V_ms, S_t_m2]
    [+ l_t_m            unless the arm is fixed by a flag]
    [+ z_t_m            if the stabiliser's depth below the foil is free]
    [+ taper_t, AR_t, washout_t_deg       if the stabiliser is designed]
    [+ winglet_h_frac_t, winglet_cant_t_deg   ...and carries a tip device]
    [+ winglet_h_frac, winglet_cant_deg   if the FOIL carries a tip device]
    [+ chord_k1..k3     if the chord law is free]
    [+ chord_k1..k3_t   ...and the stabiliser is designed (one law each)]

Speed and depth are already design variables (the Tier C convention), which
is why this family takes no flight-state modifier: it has carried its own
flight state from the start.

Objective (MAXIMISE): L/D at the 2-D trim point.
Constraints (feasible >= 0, both signed and reported):
    g0 = cavitation margin  = min over panels of (sigma_cav(h_local) + Cp_min)
    g1 = static margin      = (SM - SM_min)

Calibration (MEASURED on this problem, not assumed): x_cg is a FRACTION of
the arm rather than a fixed station, because the arm itself is a design
variable here and a fixed CG would turn the static margin into a near-
function of l_t alone. A 4-point-per-axis sweep of the whole 8-D box (65536
designs, 61031 solvable) at X_CG_FRAC = 0.15 gives a static-margin term
spanning -1.167 to +0.452 — the SM = SM_min boundary is strictly interior —
with 8.4 % of the box feasible on BOTH constraints, the same active-
constraint story as the hydrofoil and the free-planform aircraft. Moving the
CG forward switches the constraint off: at 0.02 the margin never goes
negative (span +0.026 .. +2.077), i.e. every solvable design is stable and
the stability question stops being asked. Gated in tests/test_hydrotail.py.

Failure contract: the package's. evaluate_hydrofoil_tail never raises for
in-contract failures; fg_hydrofoil_tail returns exactly
(PENALTY, [G_FAIL, G_FAIL]) on solver failure or an untrimmable design, and a
solvable-but-infeasible design returns its TRUE L/D with signed margins.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from . import fin as _fin
from . import geometry
from . import hydrofoil as _hf
from . import tail as _tail
from .hydrofoil import (MU_WATER, P_VAP, RHO_WATER, _mast_cd0,
                        cavitation_margin_panels,
                        froude_depth, heel_lift_factor, heeled_z,
                        immersion_margin)
from .objective import PENALTY, _fail
from .polar import InvertedPolar as _InvertedPolar
from .polar import section_cm_ac as _section_cm_ac
from .rig import RigLoads
from .tail import (DZ_FRAC, DZ_GRID_FRAC, TAIL_AR, arm_row,  # noqa: F401
                   height_row, tail_span)
# ``DZ_FRAC`` is re-exported rather than used: it is the separation this
# family used to default to, and the way back to every published number is
# ``z_t_fixed=-DZ_FRAC*b`` — spelt off this module, beside the default that
# replaced it, so a study reproducing a recorded run cannot pick up the air
# families' constant by a different name.
from .tail import L_T_MIN_M as tailmod_L_T_MIN_M
from .tail import orient_section as _orient_section
from .tail import trim_lift_coefficient as _trim_lift_coefficient
from .vlm import VLM, ImagePlane, TailSurface
# the designed stabiliser's boxes are the air tail's, imported rather than
# restated: a rear stabiliser is a rear stabiliser, and two copies of the
# same aspect-ratio band would be two chances for them to drift apart. (The
# CANT band is NOT shared — water's is signed, see below.)
from .wingtail import TAIL_AR_BOUNDS, TAIL_WASHOUT_BOUNDS_DEG

G_FAIL = -1.0        # finite infeasible margin reported on solver failure

#: stabiliser area [m^2] — 14-42 % of the main foil's 0.144 m^2, the range
#: foiling craft actually use for a rear stabiliser.
S_T_BOUNDS = (0.02, 0.06)

#: stabiliser arm [m] aft of the main foil's quarter chord. The craft is
#: ~1-2 m long at this foil's scale, so this box is the hull.
L_T_BOUNDS = (0.5, 1.5)

#: stabiliser depth BELOW the main foil, as a fraction of the foil span —
#: the box this row OPENS on, not a limit on it (tail.height_row carries the
#: user's own band in). The deep end is 0.30 b, past which the pair is two
#: independent foils on a long strut rather than a craft.
#:
#: THE SHALLOW END IS ``tail.DZ_GRID_FRAC``, NOT ``tail.DZ_FRAC``. The two
#: numbers were confused for a long time and tail.py now separates them:
#: 0.05 b is where the air families' second surface was MEASURED, and 0.01 b
#: is where its downwash stops being grid-converged. A foiling craft is a
#: front wing, a strut and a stabiliser bolted to ONE horizontal fuselage —
#: they are coplanar, and a box whose shallowest layout hung the stabiliser
#: 0.05 b (60 mm on the shipped foil) below the wing plane could not express
#: the ordinary craft at all.
Z_T_FRAC_BOUNDS = (DZ_GRID_FRAC, 0.30)

#: ...AND WHAT AN UNSTATED SEPARATION IS. The craft is FLAT: wing, strut and
#: stabiliser in one plane, which is how a windfoil, a wingfoil and a Moth
#: are actually built — one fuselage, two horizontal surfaces bolted to it.
#:
#: It is ``DZ_GRID_FRAC`` rather than exactly zero, and that is a solver
#: statement rather than a design one. At dz = 0 the stabiliser's
#: collocation points lie IN the foil's trailing-vortex sheet, where the leg
#: kernel dy/(dy^2 + dz^2) is grid-singular: the measured L/D spread over
#: three grids is 6.6 % at zero against 0.27 % at 0.01 b (tail.DZ_GRID_FRAC,
#: session 36's re-measurement). 0.01 b is 12 mm under the shipped 1.2 m
#: foil — a fuselage's own thickness, and coplanar to anything this solver
#: can resolve. An exact zero stays typeable through ``z_t_fixed``; what it
#: buys is a number that moves 6.6 % with the panel count.
#:
#: This REPLACED ``DZ_FRAC`` as the default (session 68) and every published
#: hydrotail number moved with it. ``tests/test_the_craft_is_flat.py`` pins
#: the deltas.
Z_T_FRAC_DEFAULT = DZ_GRID_FRAC

#: WHERE THE STRUT STANDS, as a fraction of the stabiliser arm: 0 is the
#: main foil's quarter chord and 1 is the stabiliser's, so the strut is
#: BETWEEN THE TWO SURFACES the way a real craft's mast is — bolted to the
#: fuselage that joins them, not growing out of the front wing.
#:
#: Asked as a FRACTION for the reason :data:`Z_T_FRAC_LABEL` is: the arm is
#: itself a design variable, so a station in metres means a different layout
#: at each end of its band.
#:
#: The number is the ONE MEASURED LAYOUT this repo carries. FoilingBO's
#: windfoil (``gui/foiling/layout.py``) places the front wing 0.06 m FORWARD
#: of the mast foot on a 0.95 m fuselage, so the mast's station is
#: 0.06 / 0.95 of the arm — written as that ratio rather than as a rounded
#: decimal, so the derivation is the code.
#:
#: Note what it does NOT do: with the CG at ``X_CG_FRAC`` = 0.15 of the arm,
#: a strut at 0.063 sits AHEAD of the CG and its yaw moment is
#: DESTABILISING. That is what a windfoil is — the rider stands over the
#: mast and steers it — and it is reported rather than tuned away. Moving
#: the strut aft of the CG is a design choice the station row now allows.
X_MAST_FRAC = 0.06 / 0.95

#: ...and the band it is SEARCHED over where ``free_mast_station`` opens it.
#: Bounded off both ends: at 0 the strut is inside the foil's own chord and
#: at 1 it is inside the stabiliser's, and a strut that shares a station
#: with a lifting surface is a junction this package does not model.
X_MAST_FRAC_BOUNDS = (0.05, 0.85)

#: what that row is CALLED.
X_MAST_FRAC_LABEL = "x_mast_frac"

#: ...and what that row is CALLED when it is searched as a fraction rather
#: than in metres — which is exactly when the SPAN is searched too. A metre
#: band is a statement about one span, and with the span itself a design
#: variable it would mean two different layouts at the two ends of its band:
#: -0.36 m under a 0.6 m foil is 0.60 of the span, and under a 2.4 m foil it
#: is 0.15 of it. Asked as a fraction the row means the same thing for every
#: candidate, and it is the form the band was written in to begin with. This
#: is what replaced a flat refusal of the pair.
Z_T_FRAC_LABEL = "z_t_frac"

#: CG station as a FRACTION of the stabiliser arm (module docstring: the arm
#: is a design variable, so a fixed CG would collapse the stability question
#: into "how long is the boat").
X_CG_FRAC = 0.15

#: signed cant band for a tip device in water (hydrofoil.py's rule: + points
#: the device up towards the surface, − down away from it, and the trade is
#: the sign, not the projection).
WINGLET_CANT_BOUNDS_DEG = (-90.0, 90.0)
WINGLET_H_FRAC_BOUNDS = (0.0, 0.15)


@dataclass
class HydrofoilTailProblem:
    """Main foil (+ tip device) + stabiliser under a free surface.

    Which design variables exist is fixed at construction (``winglet``,
    ``free_height``, ``l_t_fixed``, ``chord_order``) because the dimension is
    part of the problem's identity — the package registers one ProblemSpec
    per combination rather than letting a flag change the vector's length
    underneath a static label list.
    """

    # ---- main foil size (the calibrated Tier C geometry)
    b: float = 1.2
    S: float = 0.144
    #: THE MAIN FOIL'S OWN CANT AND SWEEP [deg], stated.
    #: ``hydrofoil.HydrofoilWingletProblem``'s pair, same contract and the
    #: same measured trades (sweep buys cavitation margin, dihedral trades
    #: submergence against draught) — and the same warning: NOT a spiral
    #: lever, because this craft's only vertical is the mast and it stands
    #: ahead of the CG. 0.0 is every published run, bit-for-bit.
    wing_dihedral_deg: float = 0.0
    wing_sweep_deg: float = 0.0
    # ---- resolution
    N_vlm: int = 40            # main-foil panels (full span)
    n_winglet: int = 8         # panels per tip device
    N_t: int = 20              # stabiliser panels
    # ---- operating point / references
    L_design: float = 6000.0   # [N] craft weight the pair must carry
    rig: "RigLoads | None" = None   # the RIG's loads (rig.py). ``None`` is
    #   NO RIG, which is every published run of this family, bit-for-bit —
    #   and it is the whole of the "off" switch: the couple, the vertical
    #   relief and the report block all hang off this one ``is None``.
    #   Set, it does three things and only three. (1) It subtracts the rig's
    #   vertical share from the lift the foil must make (``L_required``
    #   below, the one author). (2) It adds its pitching moment to
    #   ``cm_ac_model``, which reaches BOTH the pre-trim closed form and the
    #   3-way solve, because they read the same variable — a rig only one of
    #   them could see would quote a stabiliser load sign the run does not
    #   fly, which is a bug this repo has already had once. (3) It reports
    #   itself, including this engine's own appendage L/D beside the craft
    #   L/D that divided the couple, because those two differ by 3-4x and the
    #   gap is a calibration statement rather than an error (rig.py).
    #   What it deliberately does NOT do is move ``x_cg``. That substitution
    #   reproduces the trim exactly and makes the static-margin constraint
    #   vacuous; the derivation and the measured numbers are in rig.py.
    rho: float = RHO_WATER
    mu: float = MU_WATER
    p_vap: float = P_VAP       # water vapour pressure (sea vs fresh)
    c_mast: float = 0.08
    cf_mast: float = 0.004
    #: THE STRUT, ASKED. ``fin`` is the presence key every family answers
    #: through ``fin.has_fin``: True (the default) charges the mast exactly
    #: as every published water run was charged, and False is a craft with
    #: no vertical surface at all — no mast drag, nothing to choose a
    #: section for, and the honest zero yaw stiffness that goes with it.
    #: ``fin_tc`` is its SECTION'S thickness, chosen on stage 2.7 and
    #: reported, so the loft and the flight rebuild carry the shape that was
    #: picked. It does NOT enter ``_mast_cd0``, which charges wetted area
    #: with no form factor — so a thicker strut section moves no published
    #: number, and that is said here rather than implied.
    fin: bool = True
    fin_tc: float | None = None
    polar_family: Any = None
    section_polar: Any = None   # a DESIGNED section (hydrofoil_section.py):
    #   when set, the section is fixed by its CST weights, so the t/c row
    #   leaves the vector (the weights ARE the thickness) and this polar —
    #   which must carry a Cp_min table, or the cavitation constraint has
    #   nothing to read — drives the solve.
    #: which way up the stabiliser's section is mounted — tail.tail_polar's
    #: rule, shared so the three cores cannot fly one surface three ways.
    #: None (default) follows the load; this craft's elevator LIFTS, so it
    #: stays upright unless a stated CG puts a download on it.
    tail_inverted: bool | None = None
    polar_tail: Any = None      # the STABILISER's own section. None keeps
    #   the documented default (it flies the main foil's), so every
    #   published run is bit-for-bit unchanged. Like the foil's, it must
    #   carry a Cp_min table: the stabiliser sits deeper, and it is
    #   perfectly capable of being the surface that cavitates first.
    alpha_bracket_deg: tuple[float, float] = (-10.0, 15.0)
    i_t_max_deg: float = 15.0
    SM_min: float = 0.08
    draught_max_m: float | None = None   # the DRAUGHT CAP [m] — how deep the
    #   whole assembly (foil, stabiliser and both tip devices) may reach below
    #   the free surface. ``None`` = uncapped, every published run bit-for-bit
    #   and the margin vector stays width 2. Set, it appends a THIRD margin.
    #   Rationale, and why this rather than a projected-span cap, is in
    #   ``hydrofoil.HydrofoilWingletProblem.draught_max_m`` and report SS17.10.
    x_cg: float | None = None     # [m]; None -> X_CG_FRAC * arm
    AR_t: float = TAIL_AR
    #: THE MAIN FOIL'S SIZE, WHERE THE USER SEARCHES IT — the planar twin's
    #: fields of the same name, validated by the same ``hydrofoil._size_check``
    #: and appended to the design vector in the same place (last but for the
    #: chord blocks), so every published index keeps its meaning. ``None`` is
    #: the fixed published planform, bit-for-bit.
    #:
    #: The SPAN row carries no refusal at all now. It used to be withheld
    #: beside a vertical surface AT ZERO HEEL, because nothing there reads
    #: how wide the foil is — which is true, and is said rather than enforced
    #: (``hydrofoil.SPAN_IS_UNPRICED_AT_ZERO_HEEL``,
    #: ``hydrofoil.span_pricing_note``): the answer is then the top of the
    #: band rather than an optimum, and that is a reading, not a reason to
    #: withhold the row from the default configuration of every water family.
    #: It used to carry a second one — not beside a SEARCHED stabiliser depth —
    #: and that one is gone: the coupling it named was in the depth row's
    #: UNITS, so the row changes units instead (:data:`Z_T_FRAC_LABEL`). The
    #: AREA row has neither: nothing in this family's box is stated as a
    #: fraction of the area, which is why the two rows are separate fields
    #: and not one "free size" switch.
    span_bounds_m: tuple | None = None
    area_bounds_m2: tuple | None = None
    #: THE CRAFT'S ROLL and the tip immersion it must keep — see
    #: ``hydrofoil.HydrofoilProblem.heel_deg``. This is the family a wind
    #: foiler actually flies (it is the only one that can carry a
    #: ``rig``), so it is the one where the heel and the searched span meet:
    #: the rig sets the side force, the rider sets the heel, and the heel is
    #: what finally makes the span a question with an interior answer.
    heel_deg: float = 0.0
    tip_clearance_m: float | None = None
    #: READ EACH SURFACE'S SECTION AT ITS OWN FLOWN REYNOLDS NUMBER — see
    #: ``hydrofoil.HydrofoilProblem.flown_reynolds``. This is the family it
    #: matters most on, because it has TWO surfaces at two chords in the same
    #: water at the same speed: at the published box centre the stabiliser's
    #: Re is 0.816x the foil's (1.139e6 against 1.395e6), and both were being
    #: read off the same Re-1e6 table. Off by default; every published number
    #: is unchanged.
    flown_reynolds: bool = False
    # ---- design freedoms (each changes the design vector)
    winglet: bool = False
    free_height: bool = False
    l_t_fixed: float | None = None
    l_t_bounds_m: tuple | None = None  # (min, max) the SEARCHED arm is boxed
    #                             by — the user's band, checked against the
    #                             trim solve's own floor and nothing else
    #                             (tail.arm_row). None -> L_T_BOUNDS above.
    s_t_bounds_m2: tuple | None = None  # ...and the SEARCHED stabiliser
    #                             AREA's own band [m^2] (tail.area_row).
    #                             None -> S_T_BOUNDS above, this craft's own
    #                             calibrated band. It travels for the reason
    #                             the arm's does: a row widened in a shell
    #                             alone moves the sampler's box while the
    #                             problem keeps its own, and every draw
    #                             outside it comes back a bounds violation.
    tail_limits: "_tail.TailLimits | None" = None   # what the stabiliser's
    #                             own SPAN and CHORD may measure, in metres
    #                             (tail.TailLimits). None = unconstrained.
    z_t_bounds_m: tuple | None = None  # ...and the same for the VERTICAL
    #                             separation when free_height searches it:
    #                             the user's band in metres, NEGATIVE down
    #                             (tail.height_row). None -> Z_T_FRAC_BOUNDS
    #                             on this span.
    z_t_fixed: float | None = None   # the stabiliser's vertical separation
    #                             below the foil [m, NEGATIVE down]. Only
    #                             meaningful while free_height is off. None
    #                             keeps the layout's own
    #                             -Z_T_FRAC_DEFAULT*b — the COPLANAR craft,
    #                             a DEFAULT and not a floor
    #                             (tail.height_row). An exact 0.0 is
    #                             typeable and the constant says what it
    #                             costs in grid convergence.
    # ---- THE STRUT, as a placed and flown surface (hydrofoil.strut_report)
    #: ON, and this family is where it is on. A strut only HAS a station
    #: once there is a second surface to stand between, so the single-foil
    #: ``HydrofoilProblem`` keeps the published wetted-area charge and this
    #: one does not. Off reproduces that charge bit-for-bit
    #: (``hydrofoil._mast_cd0``), which is what every recorded hydrotail
    #: number was scored on.
    strut_model: bool = True
    #: WHERE IT STANDS, as a fraction of the arm (:data:`X_MAST_FRAC`).
    #: None takes the measured windfoil layout; a stated value places it.
    #: Ignored — and refused — beside ``free_mast_station``.
    x_mast_frac: float | None = None
    #: ...or SEARCHED, which is the same question asked of the optimiser.
    free_mast_station: bool = False
    #: ...over the user's own band, else :data:`X_MAST_FRAC_BOUNDS`.
    x_mast_frac_bounds: tuple | None = None
    chord_order: int = 0
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
    winglet_chord_follows: bool = False   # the tip device's chord CONTINUES
    #   the surface's own chord law past the tip instead of holding the tip
    #   chord (vlm.py; objective.Problem carries the air twin of this field).
    #   False = the rectangular device every published run flew, bit-for-bit.
    #   It reaches the STABILISER's tip device too: it is a statement about
    #   the design, not about one surface.
    # ---- the STABILISER as a designed surface (each changes the vector).
    # Off, it is the published rectangle at AR 4 whose only freedoms are its
    # area, its arm and (free_height) its depth. On, it is a wing in its own
    # right — taper, aspect ratio, washout, its own chord law wherever the
    # foil carries one, and (tail_winglet) its own tip device — panelised by
    # the same VLM constructor as the main foil. Its INCIDENCE stays the
    # trim unknown either way.
    tail_free: bool = False
    tail_winglet: bool = False
    tail_cant_bounds: tuple | None = None
    #: tip-device HEIGHT bands, as a fraction of each surface's
    #: semi-span. None = the published (0, 0.15) — where the model
    #: was MEASURED, not a cap. geometry.winglet_h_row validates.
    winglet_h_bounds: tuple | None = None
    winglet_h_bounds_t: tuple | None = None
    #: ...and the MAIN FOIL's own cant band, which is what a device TYPE
    #: narrows (a vertical fence is |cant| 84-90). ``tail_cant_bounds``
    #: above has been the stabiliser's since the tip-device shape became a
    #: question there; the foil's device had no such field at all, so the
    #: fence was the one shape no water craft's main foil could be given.
    #: Signed, like its default: under water the side is part of the answer.
    winglet_cant_bounds: tuple | None = None
    tail_winglet_follow: bool = False   # the stabiliser's tip device takes
    #                             its SIGN from the load the trim solve puts
    #                             on it (wingtail.WingTailProblem's field of
    #                             the same name): the cant's magnitude stays
    #                             the design variable, the side is derived.
    # ---- the tip devices' root TRANSITION, as VALUES (never dimensions).
    # Same three fields, same meanings and same defaults as
    # hydrofoil.HydrofoilWingletProblem, which is the single-surface twin of
    # this solver — so "build it blended" says the same thing in both, and
    # 0.0 is the sharp corner every published run flies, bit-for-bit.
    blend_frac_fixed: float = 0.0   # how much of the device's arc length is
    #                             spent turning out of its surface's plane.
    #                             Reaches BOTH devices (the foil's and the
    #                             stabiliser's) for the reason
    #                             winglet_chord_follows does.
    wing_blend_frac: float = 0.0    # ...and how much of that turn the SURFACE
    #                             does, as a fraction of its own semi-span
    tail_blend_frac_fixed: float | None = None   # ...and the STABILISER's own
    #                             device where it differs (None = the design's
    #                             blend above; wingtail.WingTailProblem's field
    #                             of the same name and the same rule)
    blend_shape: str = "arc"        # which turn law draws it
    junction_drag: bool = False     # charge the corner's interference drag
    #                             (junction.py). A blend defaults it ON: that
    #                             add-on IS the reason to blend.

    # ---- the OPERATING POINT's two rows, as FIELDS with the published
    # bands as their defaults: hydrofoil.HydrofoilProblem's fields of the
    # same name, validated by the same two functions, and its ``depth_row``
    # docstring is the one that explains why they stopped being bare class
    # attributes. Untouched, this problem is bit-for-bit the published one.
    DEPTH_BOUNDS: tuple[float, float] = (0.15, 1.0)   # [m] foil below the
    #                                     free surface
    V_BOUNDS: tuple[float, float] = (8.0, 16.0)       # [m/s]

    def __post_init__(self):
        # first of everything: this family's ``min_draught_m`` is
        # DEPTH_BOUNDS[0] PLUS the stabiliser's separation, and the draught
        # cap below is refused against that sum
        from .hydrofoil import depth_row, speed_row
        self.DEPTH_BOUNDS = depth_row(self.DEPTH_BOUNDS)
        self.V_BOUNDS = speed_row(self.V_BOUNDS)
        if self.polar_family is None:
            from .polar import default_polar_family
            self.polar_family = default_polar_family()
        # this family evaluates cavitation too (cavitation_margin_panels,
        # below), so it carries exactly the same Cp_min requirement as the
        # two in hydrofoil.py and gets the same config-time refusal.
        from .hydrofoil import (_check_cp_min_available, _check_draught_cap,
                                _clearance, _size_check, heel_angle)
        _size_check(self)
        self.heel_deg = heel_angle(self.heel_deg)
        self.tip_clearance_m = _clearance(self.tip_clearance_m,
                                          "HydrofoilTailProblem")
        _hf._check_flown_reynolds(self, "HydrofoilTailProblem")
        _check_cp_min_available(self, "HydrofoilTailProblem")
        _check_draught_cap(self, "HydrofoilTailProblem")
        # the RIG, checked against the craft it is bolted to. Two refusals,
        # both at CONFIG time so a search cannot pay for a full budget of
        # designs that were never going to mean anything: a rig that is not a
        # RigLoads (a dict of the right keys would be accepted and then do
        # nothing, since every read below goes through the value object's own
        # methods), and a vertical share that leaves the foil no lift to make
        # (rig.RigLoads.validate_for_weight, which is where the reasoning is).
        if self.rig is not None:
            if not isinstance(self.rig, RigLoads):
                raise TypeError(
                    f"rig must be a rig.RigLoads or None, not "
                    f"{type(self.rig).__name__}: the couple, the relief and "
                    f"the drive closure are all methods on that value "
                    f"object, so anything else would be accepted and "
                    f"silently ignored")
            self.rig.validate_for_weight(self.L_design)
        if self.blend_shape not in geometry.BLEND_SHAPES:
            raise ValueError(
                f"unknown blend_shape {self.blend_shape!r}; "
                f"choose from {list(geometry.BLEND_SHAPES)}")
        blo, bhi = geometry.WINGLET_BLEND_BOUNDS
        if not (blo <= float(self.blend_frac_fixed) <= bhi):
            raise ValueError(
                f"blend_frac_fixed {self.blend_frac_fixed} outside the "
                f"design band {geometry.WINGLET_BLEND_BOUNDS}")
        wlo, whi = geometry.WING_BLEND_FRAC_BOUNDS
        if not (wlo <= float(self.wing_blend_frac) <= whi):
            raise ValueError(
                f"wing_blend_frac {self.wing_blend_frac} outside the design "
                f"band {geometry.WING_BLEND_FRAC_BOUNDS}")
        if float(self.blend_frac_fixed) > 0.0 and not (
                self.winglet or self.tail_winglet):
            raise ValueError(
                "blend_frac_fixed needs a tip device to blend into the "
                "surface it sits on; this problem carries none")
        if self.tail_blend_frac_fixed is not None:
            if not (blo <= float(self.tail_blend_frac_fixed) <= bhi):
                raise ValueError(
                    f"tail_blend_frac_fixed {self.tail_blend_frac_fixed} "
                    f"outside the design band "
                    f"{geometry.WINGLET_BLEND_BOUNDS}")
            if float(self.tail_blend_frac_fixed) > 0.0 \
                    and not self.tail_winglet:
                raise ValueError(
                    "tail_blend_frac_fixed needs a tip device ON THE "
                    "STABILISER to blend into it (tail_winglet=True)")
        if self.l_t_bounds_m is not None:
            if self.l_t_fixed is not None:
                raise ValueError(
                    "the arm cannot be both a stated value (l_t_fixed) and a "
                    "searched band (l_t_bounds_m) — state it, or box it")
            arm_row(self.l_t_bounds_m, L_T_BOUNDS)   # validated once, here
        if self.z_t_bounds_m is not None:
            if self.z_t_fixed is not None:
                raise ValueError(
                    "the stabiliser's separation cannot be both a stated "
                    "value (z_t_fixed) and a searched band (z_t_bounds_m) — "
                    "state it, or box it")
            height_row(self.z_t_bounds_m, (-1.0, 0.0))   # validated once
        # ---- THE STRUT'S STATION: stated or searched, never both, and
        # never outside the fuselage it is bolted to.
        if self.free_mast_station and self.x_mast_frac is not None:
            raise ValueError(
                "the strut's station cannot be both a stated fraction "
                "(x_mast_frac) and a searched row (free_mast_station) — "
                "state it, or search it")
        if self.x_mast_frac is not None:
            f = float(self.x_mast_frac)
            if not (0.0 <= f <= 1.0):
                raise ValueError(
                    f"x_mast_frac {f} is outside the fuselage: 0 is the "
                    f"main foil's quarter chord and 1 is the stabiliser's, "
                    f"and the strut stands between them")
        if self.x_mast_frac_bounds is not None:
            if not self.free_mast_station:
                raise ValueError(
                    "x_mast_frac_bounds boxes a row this problem does not "
                    "search; set free_mast_station=True or drop the band")
            lo, hi = (float(v) for v in self.x_mast_frac_bounds)
            if not (0.0 <= lo < hi <= 1.0):
                raise ValueError(
                    f"x_mast_frac_bounds {(lo, hi)} must satisfy "
                    f"0 <= lo < hi <= 1 — the strut's station is a fraction "
                    f"of the arm, measured from the main foil")
        if not self.strut_model and self.free_mast_station:
            raise ValueError(
                "free_mast_station searches where the strut stands, and "
                "with strut_model off nothing reads the station: the "
                "published charge (hydrofoil._mast_cd0) is wetted area "
                "alone. Turn the model on, or state the station")
        if self.l_t_fixed is not None:
            if float(self.l_t_fixed) < tailmod_L_T_MIN_M:
                raise ValueError(
                    f"stated arm {self.l_t_fixed} m is below the floor "
                    f"{tailmod_L_T_MIN_M} m: with no moment arm the "
                    f"stabiliser has no pitch authority and the trim solve "
                    f"is singular. (The CALIBRATED band is {L_T_BOUNDS} m — "
                    f"the box the search opens on — but a longer craft is a "
                    f"design decision, not an out-of-contract input.)")
        if (self.z_t_bounds_m is not None and self.free_height
                and self.span_bounds_m is not None):
            raise ValueError(
                "the stabiliser's separation is searched as a FRACTION of "
                "the span while the span is searched too, so a band in "
                f"METRES cannot be stated for it: the {Z_T_FRAC_LABEL!r} "
                f"design-box row takes {Z_T_FRAC_BOUNDS} (of the span). "
                "State the span instead if the band has to be in metres.")
        if self.z_t_fixed is not None:
            if self.free_height:
                raise ValueError(
                    "the stabiliser's separation cannot be both a design "
                    "variable and a stated value (free_height with "
                    "z_t_fixed) — free it, or state it, not both")
            # ...and NO ceiling on the value — see tail.height_row. It is
            # still measured DOWN from the foil, so a separation below it is
            # negative, but -0.06 m on a 1.2 m foil was never a limit: it is
            # where this family's stabiliser heights were measured.
        if self.tail_winglet and not self.tail_free:
            # the rectangular stabiliser is drawn from its area alone, so
            # there is no tip to hang a device on — refuse rather than
            # silently drop two design variables
            raise ValueError(
                "a tip device on the stabiliser needs the stabiliser to be "
                "a designed surface (tail_free=True)")

    # ---------------------------------------------------------- the vector

    @property
    def tail_blend_frac(self) -> float:
        """The blend the STABILISER's own tip device is drawn with — the
        design's unless this surface answered for itself."""
        return float(self.blend_frac_fixed
                     if self.tail_blend_frac_fixed is None
                     else self.tail_blend_frac_fixed)

    @property
    def height_is_a_fraction(self) -> bool:
        """Is the stabiliser's depth searched as a FRACTION of the span?

        Only where BOTH are design variables — see :data:`Z_T_FRAC_LABEL`
        for why the units have to follow the span, and
        ``hydrofoil.SPAN_IS_UNPRICED_AT_ZERO_HEEL`` for what a searched span
        is worth on a craft flown flat (a bound, not an optimum — said, not
        refused).
        """
        return bool(self.free_height) and self.span_bounds_m is not None

    @property
    def z_t_frac_bounds(self) -> tuple[float, float]:
        """...and its box, which is the published fraction band itself.

        No user override: a band in metres is refused beside a searched span
        (``__post_init__``), and there is nothing else to narrow it with yet.
        """
        return (float(Z_T_FRAC_BOUNDS[0]), float(Z_T_FRAC_BOUNDS[1]))

    @property
    def x_mast_frac_box(self) -> tuple[float, float]:
        """The searched strut-station row, as a fraction of the arm.

        The user's band where there is one (validated in ``__post_init__``),
        else :data:`X_MAST_FRAC_BOUNDS`.
        """
        if self.x_mast_frac_bounds is None:
            return (float(X_MAST_FRAC_BOUNDS[0]),
                    float(X_MAST_FRAC_BOUNDS[1]))
        lo, hi = (float(v) for v in self.x_mast_frac_bounds)
        return (lo, hi)

    @property
    def z_t_bounds(self) -> tuple[float, float]:
        """Stabiliser depth box [m], SIGNED: below the foil is negative.

        The USER's band where there is one, else the span fractions on THIS
        foil's span (``tail.height_row``).
        """
        lo, hi = Z_T_FRAC_BOUNDS
        return height_row(self.z_t_bounds_m, (-hi * self.b, -lo * self.b))

    @property
    def param_labels(self) -> tuple:
        labels = ["taper", "twist_root_deg", "twist_tip_deg"]
        if self.section_polar is None:
            labels.append("tc")
        labels += ["depth_m", "V_ms", "S_t_m2"]
        if self.l_t_fixed is None:
            labels.append("l_t_m")
        if self.free_height:
            labels.append(Z_T_FRAC_LABEL if self.height_is_a_fraction
                          else "z_t_m")
        if self.free_mast_station:
            labels.append(X_MAST_FRAC_LABEL)
        if self.tail_free:
            labels += ["taper_t", "AR_t", "washout_t_deg"]
            if self.tail_winglet:
                labels += ["winglet_h_frac_t", "winglet_cant_t_deg"]
        if self.winglet:
            labels += ["winglet_h_frac", "winglet_cant_deg"]
        labels += list(_hf.size_labels(self))   # see HydrofoilProblem
        chord = geometry.chord_labels(self.chord_order,
                                      self.chord_law)
        if self.tail_free and chord:
            # one chord law per SURFACE, foil first then stabiliser — the
            # tandem pair's convention (tandem.py), which wingtail.py's
            # designed tail follows too, so a trailing block is read the same
            # way wherever two surfaces are designed together
            chord = chord + tuple(f"{lbl}_t" for lbl in chord)
        return tuple(labels) + chord

    @property
    def bounds(self) -> np.ndarray:
        rows = [geometry.TAPER_BOUNDS,
                geometry.TWIST_ROOT_BOUNDS_DEG,
                geometry.TWIST_TIP_BOUNDS_DEG]
        if self.section_polar is None:
            rows.append(geometry.TC_BOUNDS)
        # the stabiliser's SIZE rows, narrowed first to whatever the user
        # stated in metres — a span or chord limit IS an area and an
        # aspect-ratio band (tail.TailLimits.narrow)
        s_row = _tail.area_row(self.s_t_bounds_m2, S_T_BOUNDS)
        ar_row = TAIL_AR_BOUNDS if self.tail_free else None
        if self.tail_limits is not None:
            s_row, ar_row = self.tail_limits.narrow(s_row, ar_row,
                                                    _tail.TAIL_AR)
        rows += [self.DEPTH_BOUNDS, self.V_BOUNDS, s_row]
        if self.l_t_fixed is None:
            rows.append(arm_row(self.l_t_bounds_m, L_T_BOUNDS))
        if self.free_height:
            rows.append(self.z_t_frac_bounds if self.height_is_a_fraction
                        else self.z_t_bounds)
        if self.free_mast_station:
            rows.append(self.x_mast_frac_box)
        if self.tail_free:
            rows += [geometry.TAPER_BOUNDS, ar_row,
                     TAIL_WASHOUT_BOUNDS_DEG]
            if self.tail_winglet:
                rows += [geometry.winglet_h_row(self.winglet_h_bounds_t),
                         self._tail_cant_box()]
        if self.winglet:
            rows += [geometry.winglet_h_row(self.winglet_h_bounds),
                     self._foil_cant_box()]
        rows += _hf.size_rows(self)
        box = geometry.with_chord_bounds(np.array(rows, dtype=float),
                                         self.chord_order,
                                         self.chord_max_frac, self.chord_law)
        if self.tail_free and self.chord_order:
            # the stabiliser's own chord law, appended after the foil's
            # (the labels above name them ``*_t``)
            box = geometry.with_chord_bounds(box, self.chord_order,
                                             self.chord_max_frac,
                                             self.chord_law)
        return box

    def _foil_cant_box(self) -> tuple:
        """Cant band for the tip device ON THE MAIN FOIL.

        The twin of :meth:`_tail_cant_box`, and signed for the same reason:
        the water band's sign IS the trade (up towards the surface at less
        static head, down away from it). A shape that narrows the magnitude
        — a vertical fence — therefore has to name a side, and the family
        that builds this problem states DOWN, the side with the cavitation
        margin. None = the published band, unchanged.
        """
        if self.winglet_cant_bounds is None:
            return WINGLET_CANT_BOUNDS_DEG
        lo, hi = (float(v) for v in self.winglet_cant_bounds)
        cmin, cmax = geometry.WINGLET_CANT_LIMITS_SIGNED_DEG
        if not (cmin <= lo < hi <= cmax):
            raise ValueError(
                f"main foil cant_bounds {(lo, hi)} outside the physical "
                f"VLM range {geometry.WINGLET_CANT_LIMITS_SIGNED_DEG}")
        return (lo, hi)

    def _tail_cant_box(self) -> tuple:
        """Cant band for a tip device ON THE STABILISER.

        The water band by default, and it is signed on purpose: under water
        the trade is which WAY the device points — up towards the surface
        buys nonplanar span at less static head, down away from it the
        reverse — not how much span it projects.
        """
        if self.tail_cant_bounds is None:
            return WINGLET_CANT_BOUNDS_DEG
        lo, hi = float(self.tail_cant_bounds[0]), float(self.tail_cant_bounds[1])
        # SIGNED limits: this surface's default band is already signed, so
        # validating an override against the positive-only aircraft range
        # would have refused half of its own default (wingtail._cant_box)
        cmin, cmax = geometry.WINGLET_CANT_LIMITS_SIGNED_DEG
        if not (cmin <= lo < hi <= cmax):
            raise ValueError(
                f"stabiliser cant_bounds {(lo, hi)} outside the physical "
                f"VLM range {geometry.WINGLET_CANT_LIMITS_SIGNED_DEG}")
        return (lo, hi)

    @property
    def dim(self) -> int:
        return int(self.bounds.shape[0])

    @property
    def min_draught_m(self) -> float:
        """The shallowest draught any design in this box can reach [m].

        NOT ``DEPTH_BOUNDS[0]``: this family carries a STABILISER below the
        foil, so even the shallowest, flattest design draws the foil depth
        plus that separation. With the height free the search may go deeper
        still, but never shallower than the ``Z_T_FRAC_DEFAULT*b``
        separation,
        and a fixed height pins it exactly.

        Getting this wrong made every cap in [0.15, 0.21) constructible and
        then infeasible everywhere — the "search finds nothing after paying
        for a full budget" outcome the config-time check exists to prevent.
        """
        # ...and with the SPAN searched the floor is at the band's SMALLEST
        # span, because the separation grows with it. Reading ``self.b`` here
        # while ``b`` was a design variable would refuse a cap the run can
        # actually meet (or, at the other end, accept one it cannot) — the
        # hazard class ``ops()`` and ``CL_target`` were both fixed for.
        b_min = (float(self.b) if self.span_bounds_m is None
                 else float(min(self.span_bounds_m)))
        dz = (Z_T_FRAC_DEFAULT * b_min if self.z_t_fixed is None
              else abs(float(self.z_t_fixed)))
        return float(self.DEPTH_BOUNDS[0]) + float(dz)

    @property
    def immersion_capped(self) -> bool:
        """See ``hydrofoil.HydrofoilProblem.immersion_capped``."""
        return self.tip_clearance_m is not None

    @property
    def n_constraints(self) -> int:
        return (2 + (0 if self.draught_max_m is None else 1)
                + (1 if self.immersion_capped else 0))

    @property
    def constraint_labels(self) -> tuple:
        """The margins this problem actually returns, in order.

        ``api.design_report`` prefers these over the static ProblemSpec
        labels, which describe the UNCAPPED family.
        """
        from .hydrofoil import (DRAUGHT_CONSTRAINT_LABEL,
                                IMMERSION_CONSTRAINT_LABEL)
        return (("cavitation margin", "static margin - SM_min")
                + (() if self.draught_max_m is None
                   else (DRAUGHT_CONSTRAINT_LABEL,))
                + ((IMMERSION_CONSTRAINT_LABEL,) if self.immersion_capped
                   else ()))

    @property
    def n_chord_rows(self) -> int:
        """Trailing chord-coefficient rows: one block per DESIGNED surface."""
        return int(self.chord_order) * (2 if self.tail_free else 1)

    @property
    def L_required(self) -> float:
        """The lift the FOIL must make [N] — THE ONE AUTHOR of that number.

        ``L_design`` is what the CRAFT weighs; this is what is left for the
        foil once the rig has taken its vertical share (a kite's tether pulls
        21-46 % of the rider's weight straight up — rig.py carries the five
        measured cases). Without a rig the two are the same number, which is
        why every published run is unchanged bit-for-bit.

        It is a property and not an expression repeated at each use because
        the repo has a recorded bug of exactly that shape: a rule applied in
        two places drifts, and the two authors then disagree about which
        craft is being flown. ``CL_target`` reads this; so does anything that
        wants to know how hard the foil is working. Nothing subtracts
        ``vertical_n`` for itself.
        """
        base = (float(self.L_design) if self.rig is None
                else self.rig.lift_required_n(self.L_design))
        # ...and a HEELED foil points its lift off the vertical, so it must
        # make ``W / cos(phi)`` to hold the craft up. Here rather than at
        # ``CL_target`` for the reason this property exists at all: one
        # author for "how heavy is this craft", so nothing downstream can
        # disagree about it. At zero heel the factor is exactly 1.0.
        return float(base * heel_lift_factor(self.heel_deg))

    def CL_target(self, V: float, S: float | None = None) -> float:
        """Trim CL on the MAIN foil's area for the lift the foil must make.

        ``S`` defaults to the problem's own so every published caller is
        unchanged; a run that SEARCHES the area passes the candidate's, and
        must — the area is the loading (``hydrofoil.HydrofoilProblem
        .CL_target`` states the same rule and had the same bug to close).
        """
        area = float(self.S if S is None else S)
        return float(self.L_required / (0.5 * self.rho * V**2 * area))

    def x_cg_for(self, l_t: float) -> float:
        """CG station [m] for a design whose stabiliser arm is ``l_t``.

        The rule, not a number: the CG is a FRACTION of the arm here
        (module docstring — the arm is itself a design variable, so a fixed
        station would slide the static margin around with it), unless the
        caller pinned ``x_cg``. Lives on the problem so the evaluation and
        anything that asks what this craft trims at read the same rule.
        """
        return (X_CG_FRAC * float(l_t) if self.x_cg is None
                else float(self.x_cg))

    def unpack(self, x: np.ndarray) -> dict:
        """Design vector -> named values (the ONE place the layout is read)."""
        x = np.asarray(x, dtype=float)
        i = 3
        out = {"taper": float(x[0]), "twist_root_deg": float(x[1]),
               "twist_tip_deg": float(x[2])}
        # THE SIZE THIS CANDIDATE FLIES, read BY LABEL (hydrofoil.size_from_x)
        # and read FIRST, because the stabiliser's clearance floor below is
        # ``Z_T_FRAC_DEFAULT * b`` — with a searched span that has to be the
        # candidate's span. Reading the nominal one there would hang the
        # stabiliser at a separation no design in the run actually has.
        out["b"], out["S"] = _hf.size_from_x(self, x)
        if self.section_polar is None:
            out["tc"] = float(x[i]); i += 1
        else:
            # the designed section's own thickness, reported not designed
            out["tc"] = float(getattr(self.section_polar, "tc", 0.12))
        out["depth"], out["V"], out["S_t"] = (float(x[i]), float(x[i + 1]),
                                              float(x[i + 2]))
        i += 3
        if self.l_t_fixed is None:
            out["l_t"] = float(x[i]); i += 1
        else:
            out["l_t"] = float(self.l_t_fixed)
        if self.free_height:
            # a FRACTION where the span is searched too, so the separation
            # is the same shape for every candidate (Z_T_FRAC_LABEL); the
            # sign convention is unchanged, below the foil is negative
            out["z_t"] = (-float(x[i]) * out["b"] if self.height_is_a_fraction
                          else float(x[i]))
            i += 1
        else:
            # the coplanar default, unless the caller stated a separation
            out["z_t"] = (-Z_T_FRAC_DEFAULT * out["b"]
                          if self.z_t_fixed is None
                          else float(self.z_t_fixed))
        # WHERE THE STRUT STANDS. A fraction of THIS candidate's arm, so a
        # searched arm moves the mast with the fuselage it is bolted to
        # rather than leaving it at a station the craft no longer has.
        if self.free_mast_station:
            out["x_mast_frac"] = float(x[i]); i += 1
        else:
            out["x_mast_frac"] = (X_MAST_FRAC if self.x_mast_frac is None
                                  else float(self.x_mast_frac))
        out["x_mast"] = float(out["x_mast_frac"]) * float(out["l_t"])
        out["taper_t"], out["AR_t"], out["washout_t"] = None, None, 0.0
        out["h_frac_t"], out["cant_t_deg"] = 0.0, -90.0
        if self.tail_free:
            out["taper_t"] = float(x[i])
            out["AR_t"] = float(x[i + 1])
            out["washout_t"] = float(x[i + 2])
            i += 3
            if self.tail_winglet:
                out["h_frac_t"] = float(x[i])
                out["cant_t_deg"] = float(x[i + 1])
                i += 2
        if self.winglet:
            out["h_frac"], out["cant_deg"] = float(x[i]), float(x[i + 1])
            i += 2
        else:
            out["h_frac"], out["cant_deg"] = 0.0, 90.0
        m = int(self.chord_order)
        if self.tail_free and m:
            # two trailing chord blocks, foil then stabiliser
            out["chord_coeffs"] = geometry.ChordCoeffs(x[-2 * m:-m],
                                                       self.chord_law)
            out["chord_coeffs_t"] = geometry.ChordCoeffs(x[-m:],
                                                         self.chord_law)
        else:
            out["chord_coeffs"] = geometry.chord_coeffs_from_x(
                x, m, self.chord_law)
            out["chord_coeffs_t"] = ()
        return out


def _worse_cavitation(foil: dict, stab: dict) -> dict:
    """The binding one of two per-surface cavitation sweeps.

    ``min`` over the union of the panels is the ``min`` of the two minima, so
    this reports exactly what one sweep over both surfaces would have — it
    exists only because Cp_min comes from a SECTION, and with a section per
    surface there are two tables to read it from.
    """
    worst = foil if foil["g"] <= stab["g"] else stab
    out = dict(worst)
    out["g_y"] = np.concatenate([foil["g_y"], stab["g_y"]])
    out["surface_worst"] = "foil" if worst is foil else "stabiliser"
    out["g_foil"], out["g_stab"] = float(foil["g"]), float(stab["g"])
    return out


# ---------------------------------------------------------------- evaluate


def evaluate_hydrofoil_tail(x: np.ndarray,
                            prob: HydrofoilTailProblem | None = None) -> dict:
    """Full evaluation with breakdown; penalty contract on failure."""
    prob = prob or HydrofoilTailProblem()
    x = np.asarray(x, dtype=float)
    bnds = prob.bounds
    if x.shape != (bnds.shape[0],):
        return _fail(f"design vector shape {x.shape} != ({bnds.shape[0]},)")
    if np.any(x < bnds[:, 0] - 1e-12) or np.any(x > bnds[:, 1] + 1e-12):
        return _fail("bounds violation")

    v = prob.unpack(x)
    # THE SIZE THIS CANDIDATE FLIES. Everything below references it and not
    # the problem's nominal planform: with the span or the area searched,
    # ``prob.S`` in a trim, a moment coefficient or the mast's reference area
    # would fly one craft and score another — the four hazards session 71
    # closed on the two planar families, closed here too.
    b_x, S_x = float(v["b"]), float(v["S"])
    why = _hf.size_refusal(prob, b_x, S_x)
    if why is not None:
        return _fail(f"size: {why}")
    try:
        foil = geometry.Wing(
            b=b_x, S=S_x, taper=v["taper"],
            twist_root_deg=v["twist_root_deg"],
            twist_tip_deg=v["twist_tip_deg"], tc=v["tc"],
            # the MAIN FOIL's own cant and sweep. The stabiliser keeps its
            # own plane: a dihedral is a statement about the surface that
            # carries the craft, and canting the elevator with it would move
            # the arm the pitch solve is closed on.
            sweep_deg=prob.wing_sweep_deg,
            dihedral_deg=prob.wing_dihedral_deg,
            chord_limits=prob.chord_limits,
            chord_coeffs=v["chord_coeffs"])
    except ValueError as exc:
        return _fail(f"planform: {exc}")

    # THE SECTION THIS CANDIDATE FLIES, picked AFTER the planform because the
    # Reynolds number is ``rho V mac / mu`` and the mac is the candidate's —
    # with the size rows open it moves with the span and the area as well as
    # with the speed. Off (every published run) this is the family's Re-1e6
    # table and the reordering changes nothing at all.
    try:
        pol = _hf.polar_for(prob, v["tc"], foil.mac, v["V"],
                            prob.section_polar)
    except ValueError as exc:
        return _fail(f"polar family: {exc}")

    # the stabiliser: rectangular equivalent of aspect ratio AR_t, BELOW the
    # main foil (negative z) and aft of it (positive x, x aft convention) —
    # or, when it is DESIGNED, a wing in its own right at its own aspect
    # ratio, panelised by the same constructor as the foil
    S_t = float(v["S_t"])
    AR_t = prob.AR_t if v["AR_t"] is None else float(v["AR_t"])
    b_t = tail_span(S_t) if AR_t == TAIL_AR else float(np.sqrt(AR_t * S_t))
    # THE STABILISER'S OWN REYNOLDS NUMBER. It is the surface this whole
    # question bites hardest on: it flies the same water at the same speed on
    # a chord a fraction of the foil's, so its Re is a fraction of the foil's
    # too — 0.816x at the published box centre (chord 0.1000 m against the
    # foil's mac 0.1225 m, Re 1.139e6 against 1.395e6), and further apart
    # wherever the stabiliser area row is searched down — and until now both
    # surfaces were looked up in the same Re-1e6 table.
    #
    # ``pol_t is pol`` STAYS TRUE when the flag is off, and that identity is
    # load-bearing further down: it is what chooses one cavitation sweep over
    # two, and what ``tail_section_inverted`` is read against. So the branch
    # is on the flag, not on the numbers — off, this is the line it always
    # was; on, the two surfaces genuinely fly two tables and the two-sweep
    # path is the correct one.
    #
    # The chord is the RECTANGULAR EQUIVALENT ``S_t / b_t``, the same one
    # ``b_t`` above is derived from, and not a designed stabiliser's own mac
    # (which is built below, from a planform this choice does not change).
    if prob.polar_tail is not None:
        pol_t = prob.polar_tail
    elif prob.flown_reynolds:
        try:
            pol_t = _hf.polar_for(prob, v["tc"], S_t / b_t, v["V"])
        except ValueError as exc:
            return _fail(f"polar family (stabiliser): {exc}")
    else:
        pol_t = pol
    # ...and the separation is NOT re-checked against the default here: the
    # box is the statement about where to search (tail.height_row), and this
    # solve is smooth through 0.05 b to zero. Re-checking made the floor a
    # third, independent ban that survived widening the box.
    z_t = float(v["z_t"])
    stab_wing = None
    if prob.tail_free:
        # its washout is measured from a root at zero: the uniform part of
        # the stabiliser's incidence IS the trim unknown i_t
        try:
            stab_wing = geometry.Wing(
                b=b_t, S=S_t, taper=float(v["taper_t"]),
                twist_root_deg=0.0, twist_tip_deg=float(v["washout_t"]),
                tc=float(getattr(pol_t, "tc", v["tc"])),
                chord_limits=prob.chord_limits,
                chord_coeffs=v["chord_coeffs_t"])
        except ValueError as exc:
            return _fail(f"stabiliser planform: {exc}")
    x_cg = prob.x_cg_for(float(v["l_t"]))
    mac = foil.mac
    V, depth = float(v["V"]), float(v["depth"])
    CLt = prob.CL_target(V, S_x)

    # the two sections' own couple on (S, mac) — a constant the lattice
    # cannot produce (vlm.VLM.solve: one chordwise panel per strip), handed
    # in the same way wingtail.py does it. It is what makes the ELEVATOR
    # react the main foil's nose-down moment instead of ignoring it; a
    # symmetric section returns 0.0 and the published run is unchanged.
    _mac_t = (stab_wing.mac if stab_wing is not None
              else float(S_t / b_t) if b_t > 0.0 else 0.0)
    # ...and the RIG's couple, on the same terms and through the same
    # channel. It is the dominant longitudinal moment on a real foiling craft
    # (rig.py: ~399 N.m for a windfoil, three independent derivations) and it
    # belongs HERE, beside the sections' couple, for one reason: both are
    # constant in (alpha, i_t), so both ride in the CONSTANT of the affine
    # trim map and neither touches the Jacobian, the neutral point or the
    # static margin AT FIXED GEOMETRY. (Not end to end: below, the load's
    # SIGN chooses the section's mounting and — where the family asks for
    # it — the stabiliser's tip device side, and both are shape. The module
    # docstring names and prices the two.) Applying it as a CG shift
    # instead reproduces the trim
    # bit-for-bit and makes the stability constraint vacuous — the trap is
    # derived, measured and refused in rig.py's docstring.
    #
    # EXACTLY 0.0 with no rig, added rather than branched around, so the
    # published number is bit-for-bit what it was (adding a zero to a finite
    # float is exact).
    q = 0.5 * prob.rho * V * V
    M_rig = (0.0 if prob.rig is None
             else prob.rig.pitching_moment_nm(weight_n=prob.L_design,
                                              depth_m=depth, x_cg_m=x_cg))
    cm_rig = 0.0 if prob.rig is None else float(M_rig / (q * S_x * mac))
    cm_ac_model = float(
        (_section_cm_ac(pol) * foil.S * foil.mac
         + _section_cm_ac(pol_t) * S_t * _mac_t) / (S_x * mac)) + cm_rig

    # ...and which way up the stabiliser flies it (tail.tail_polar): a
    # surface that pushes DOWN mounts its camber the other way. This craft's
    # CG sits BETWEEN its foils, so the elevator lifts and the rule leaves it
    # upright — but the rule is asked rather than assumed, because a stated
    # CG can put it the other way round.
    cl_stab = _trim_lift_coefficient(
        CLt, S_x, x_cg, S_t, float(v["l_t"]),
        M_ac=cm_ac_model * S_x * mac)
    pol_t = _orient_section(pol_t, cl_stab, prob.tail_inverted)
    if isinstance(pol_t, _InvertedPolar):
        cm_ac_model = float(
            (_section_cm_ac(pol) * foil.S * foil.mac
             + _section_cm_ac(pol_t) * S_t * _mac_t) / (S_x * mac)) + cm_rig
        # ...and the READOUT is re-read on the couple actually flown, exactly
        # as api.trim_surface_cl re-reads it after mirroring. The ORIENTATION
        # stays decided on the reading above (mirroring a section mirrors its
        # couple, so letting the choice vote on itself is self-reference —
        # tail.stabiliser_load); what changes here is only the number this
        # evaluation quotes as "what the stabiliser is being asked for",
        # which must be the one belonging to the aeroplane that flew.
        cl_stab = _trim_lift_coefficient(
            CLt, S_x, x_cg, S_t, float(v["l_t"]),
            M_ac=cm_ac_model * S_x * mac)

    def _fly(cant_t: float):
        """One imaged build + trim solve at a stated stabiliser cant."""
        surf = TailSurface(S=S_t, x=float(v["l_t"]), z=z_t, AR=AR_t,
                           N=prob.N_t, a=pol_t.a_lin,
                           alpha_L0=pol_t.alpha_L0, wing=stab_wing,
                           winglet_h_frac=float(v["h_frac_t"]),
                           winglet_cant_deg=float(cant_t),
                           n_winglet=prob.n_winglet,
                           winglet_blend_frac=prob.tail_blend_frac,
                           winglet_blend_shape=prob.blend_shape,
                           winglet_wing_blend_frac=prob.wing_blend_frac)
        m = VLM(foil, N=prob.N_vlm, winglet_h_frac=v["h_frac"],
                winglet_cant_deg=v["cant_deg"], n_winglet=prob.n_winglet,
                winglet_blend_frac=prob.blend_frac_fixed,
                winglet_blend_shape=prob.blend_shape,
                winglet_wing_blend_frac=prob.wing_blend_frac,
                winglet_chord_follows=prob.winglet_chord_follows,
                a=pol.a_lin, alpha_L0=pol.alpha_L0, V=V,
                image=ImagePlane(z=depth, kind="free_surface"), tail=surf)
        al, it, rr = m.solve_trim_moment(
            CLt, x_cg, mac,
            alpha_bracket=(np.deg2rad(prob.alpha_bracket_deg[0]),
                           np.deg2rad(prob.alpha_bracket_deg[1])),
            cm_ac=cm_ac_model)
        return m, surf, al, it, rr

    def _stab_lift(rr) -> float:
        """The stabiliser's own lift coefficient, on ITS own area."""
        m = rr.is_tail
        return (2.0 * float(rr.Gamma[m] @ rr.ly[m]) / (V * S_t)
                if (m.any() and S_t > 0.0) else 0.0)

    cant_t = float(v["cant_t_deg"])
    followed = None
    try:
        model, stab, alpha, i_t, res = _fly(cant_t)
        # the device follows the LOAD (wingtail.evaluate_wing_tail): under
        # water the same mirror argument carries an extra term — a device
        # canted UP sits in less static head, so the direction the lift
        # wants and the direction the cavitation margin wants agree on a
        # stabiliser carrying a download and disagree on one carrying lift.
        # The margin is a CONSTRAINT and the lift sets the drag, so the
        # sign follows the lift and the margin is reported as it lands.
        if prob.tail_winglet_follow and float(v["h_frac_t"]) > 0.0:
            want = -1.0 if _stab_lift(res) < 0.0 else 1.0
            cant_t = float(np.copysign(abs(cant_t), want))
            followed = "up" if cant_t >= 0.0 else "down"
            if cant_t != float(v["cant_t_deg"]):
                model, stab, alpha, i_t, res = _fly(cant_t)
        x_np = model.neutral_point()
    except (ValueError, np.linalg.LinAlgError) as exc:
        return _fail(f"solver failure: {exc}")

    if abs(i_t) > np.deg2rad(prob.i_t_max_deg):
        return _fail("untrimmable: |i_t| exceeds limit",
                     i_t_deg=float(np.rad2deg(i_t)))

    alpha_eff_deg = np.rad2deg(res.alpha_eff)
    tail_mask = res.is_tail
    # each surface is judged against ITS OWN polar's validity range; with one
    # section they are the same range, so this is the previous test verbatim
    lo_p, hi_p = pol.alpha_valid
    lo_t, hi_t = pol_t.alpha_valid
    ae_f, ae_t = alpha_eff_deg[~tail_mask], alpha_eff_deg[tail_mask]
    if (ae_f.min() < lo_p or ae_f.max() > hi_p
            or (ae_t.size and (ae_t.min() < lo_t or ae_t.max() > hi_t))):
        return _fail(
            "effective AoA outside polar validity (stall/extrapolation proxy)",
            alpha_eff_range=(float(alpha_eff_deg.min()),
                             float(alpha_eff_deg.max())))

    # ---- drag: one Trefftz plane over both surfaces + strip profile drag
    strip = np.where(tail_mask, pol_t.cd(alpha_eff_deg),
                     pol.cd(alpha_eff_deg)) * res.c * res.width
    CDp_foil = float(np.sum(strip[~tail_mask]) / res.S)
    CDp_stab = float(np.sum(strip[tail_mask]) / res.S)
    # THE STRUT IS A SURFACE THE CRAFT CAN BE ASKED ABOUT. Charged exactly
    # as it always was where there is one (``fin.has_fin``, True unless the
    # design says otherwise), and zero where the design states none — the
    # same three-state contract every other family's vertical surface has,
    # so "no strut" cannot mean "a strut nobody pays for".
    # ...and where ``strut_model`` is on it is a PLACED, LOADED surface
    # rather than a rectangle of wetted area: its own Reynolds number, its
    # own section (the ``fin_tc`` stage 2.7 asks for, which reached nothing
    # before), the two corners where it meets the fuselage, and the LEEWAY
    # it flies to carry the rig's side load. ``hydrofoil.strut_report`` is
    # the one author; this reads it.
    strut = None
    if not _fin.has_fin(prob):
        cd0_mast = CD_strut = 0.0
    elif prob.strut_model:
        strut = _hf.strut_report(
            depth=depth, chord=float(prob.c_mast),
            tc=float(prob.fin_tc if prob.fin_tc else _fin.FIN_TC_DEFAULT),
            S_ref=S_x, V=V, rho=prob.rho, mu=prob.mu,
            side_n=(float(prob.rig.side_n) if prob.rig is not None else 0.0),
            x_qc=float(v["x_mast"]), x_cg=x_cg)
        if not strut["linear"]:
            # A CRAFT THAT CANNOT HOLD ITS SIDE LOAD is not a craft with
            # extra drag. Refused through the penalty contract, like every
            # other out-of-range read in this evaluation, rather than
            # returning the induced drag of a lift the section cannot make.
            return _fail(
                f"the strut cannot carry the rig's side force at this "
                f"speed: it would fly {strut['leeway_deg']:.1f} deg of "
                f"leeway, past the {_hf.STRUT_LINEAR_DEG:g} deg its "
                f"side-force slope is read off",
                strut_leeway_deg=float(strut["leeway_deg"]))
        cd0_mast, CD_strut = strut["cd0"], strut["CD"]
    else:
        cd0_mast = CD_strut = _mast_cd0(depth, S_x, prob.c_mast,
                                        prob.cf_mast)
    # ...and the corner where a blended device leaves its surface. Charged on
    # the FOIL's device alone, exactly as hydrofoil.py charges it: the
    # correlation is a wing-root/tip-device junction one and the stabiliser's
    # own corner sits on a surface whose chord is a fraction of the foil's,
    # so folding it in would quote a correlation outside its own band.
    jr = None
    if prob.junction_drag and float(v["h_frac"]) > 0.0:
        from . import junction
        c_tip = float(foil.chord(np.array([foil.b / 2.0]))[0])
        jr = junction.report(tc=foil.tc, chord=c_tip, s_ref=foil.S,
                             h=float(v["h_frac"]) * foil.b / 2.0,
                             cant_deg=abs(float(v["cant_deg"])),
                             blend_frac=prob.blend_frac_fixed, n_junctions=2,
                             blend_shape=prob.blend_shape,
                             wing_arc=prob.wing_blend_frac * foil.b / 2.0)
    # ...and the STABILISER's own corner, on its own tip chord and area:
    # charging only the foil's would make a blend on the second surface a
    # pure wake-geometry benefit with no cost.
    jr_t = None
    if prob.junction_drag and prob.tail_winglet \
            and float(v["h_frac_t"]) > 0.0 and stab_wing is not None:
        from . import junction
        c_tip_t = float(stab_wing.chord(np.array([stab_wing.b / 2.0]))[0])
        jr_t = junction.report(
            tc=stab_wing.tc, chord=c_tip_t, s_ref=S_x,
            h=float(v["h_frac_t"]) * stab_wing.b / 2.0,
            # the cant FLOWN is signed here (the water band is); the fillet
            # radius is a magnitude
            cant_deg=abs(float(cant_t)),
            blend_frac=prob.tail_blend_frac, n_junctions=2,
            blend_shape=prob.blend_shape,
            wing_arc=prob.wing_blend_frac * stab_wing.b / 2.0)
    CD_junction = float((jr or {}).get("CD_junction", 0.0)) \
        + float((jr_t or {}).get("CD_junction", 0.0))
    CD = res.CDi + CDp_foil + CDp_stab + CD_strut + CD_junction
    LoD = res.CL / CD
    if not (np.isfinite(LoD) and CD > 0.0):
        return _fail("non-finite L/D")

    # ---- cavitation at EVERY panel's own submergence (the stabiliser sits
    # deeper, so it carries more static head — that is the trade). With a
    # section per surface the two are swept separately and the WORSE one is
    # the constraint: Cp_min is a property of the shape, so the stabiliser's
    # margin has to be read off the stabiliser's own polar.
    # And "its own polar" is the one FLOWN, not the one STATED: the branch
    # asks ``pol_t is pol`` rather than ``prob.polar_tail is None`` because
    # ``_orient_section`` above can have wrapped the shared section in an
    # ``_InvertedPolar`` — the common case once a rig couple puts the
    # stabiliser in download. Mirroring a cambered section moves Cp_min, so
    # the user's statement "I gave one section" is the wrong question; the
    # right one is "are these two surfaces flying the same table".
    # BOTH SURFACES HAVE TO BE IN THE WATER, checked before the cavitation
    # book because an emerged panel makes that book meaningless rather than
    # pessimistic. The stabiliser hangs BELOW the foil, so at heel the panel
    # nearest the surface is on the main foil's rising tip — and how far it
    # rises is ``(b/2) sin(phi)``, the one term in this family that reads the
    # span.
    imm = immersion_margin(depth, res.y, res.z, prob.heel_deg,
                           prob.tip_clearance_m or 0.0)
    if imm["h_min_m"] <= 0.0:
        return _fail(
            f"panel emerged: at {prob.heel_deg:g} deg of heel the assembly "
            f"reaches {-imm['h_min_m']:.3g} m above a free surface "
            f"{depth:.3g} m over the foil (span {b_x:.3g} m, at "
            f"y = {imm['y_at_h_min_m']:.3g} m). A partly emerged foil is a "
            f"different problem, not a worse design.")

    try:
        if pol_t is pol:
            cav = cavitation_margin_panels(alpha_eff_deg, res.z, depth, pol,
                                           V=V, rho=prob.rho,
                                           p_vap=prob.p_vap, y=res.y,
                                           heel_deg=prob.heel_deg)
        else:
            cav = _worse_cavitation(
                cavitation_margin_panels(alpha_eff_deg[~tail_mask],
                                         res.z[~tail_mask], depth, pol,
                                         V=V, rho=prob.rho,
                                         p_vap=prob.p_vap,
                                         y=res.y[~tail_mask],
                                         heel_deg=prob.heel_deg),
                cavitation_margin_panels(alpha_eff_deg[tail_mask],
                                         res.z[tail_mask], depth, pol_t,
                                         V=V, rho=prob.rho,
                                         p_vap=prob.p_vap,
                                         y=res.y[tail_mask],
                                         heel_deg=prob.heel_deg))
    except ValueError as exc:
        return _fail(f"cavitation: {exc}")

    SM = (x_np - x_cg) / mac
    # DRAUGHT, over EVERY panel — foil, stabiliser and both tip devices. z is
    # up positive with the free surface at z = depth, so the deepest point of
    # the whole assembly is the smallest z. Read off the solved geometry for
    # the same reason as in hydrofoil.py: a blended or canted device's tip is
    # not where a closed form would put it. See HydrofoilWingletProblem
    # .draught_max_m for why this, and not a projected-span cap.
    # ...measured on the HEELED geometry, because rolling the craft changes
    # which panel is the deepest one (hydrofoil's winglet evaluator says the
    # same thing about the same quantity).
    draught_m = float(depth - float(heeled_z(res.y, res.z,
                                             prob.heel_deg).min()))
    g = np.array([float(cav["g"]), float(SM - prob.SM_min)])
    if prob.draught_max_m is not None:
        g = np.append(g, float(prob.draught_max_m) - draught_m)
    if prob.immersion_capped:
        g = np.append(g, float(imm["g"]))
    if not np.all(np.isfinite(g)):
        return _fail("non-finite margins")

    CL_foil = 2.0 * float(res.Gamma[~tail_mask] @ res.ly[~tail_mask]) \
        / (V * S_x)
    CL_stab = (2.0 * float(res.Gamma[tail_mask] @ res.ly[tail_mask])
               / (V * S_t)) if S_t > 0.0 else 0.0
    # each surface's tip device is its own: the stabiliser's panels are
    # winglet panels too, and counting them in the FOIL's device area would
    # report a foil winglet that grew when the stabiliser's did
    wl_foil = res.is_winglet & ~tail_mask
    wl_stab = res.is_winglet & tail_mask
    S_wl = float(np.sum((res.c * res.width)[wl_foil]))

    out = {
        "feasible": True, "reason": "", "score": float(LoD),
        "LoD": float(LoD), "g": g,
        "g_cav": float(g[0]), "g_sm": float(g[1]),
        "draught_m": draught_m,
        "draught_max_m": (None if prob.draught_max_m is None
                          else float(prob.draught_max_m)),
        "g_draught": (None if prob.draught_max_m is None else float(g[2])),
        "heel_deg": float(prob.heel_deg),
        "immersion_m": float(imm["h_min_m"]),
        "y_at_immersion_m": float(imm["y_at_h_min_m"]),
        "tip_clearance_m": prob.tip_clearance_m,
        "g_immersion": (float(g[-1]) if prob.immersion_capped else None),
        "b": float(b_x), "S": float(S_x),
        "AR": float(b_x * b_x / S_x),
        "sigma_cav": cav["sigma_cav"], "sigma_cav_root": cav["sigma_cav_root"],
        "cp_min_worst": cav["cp_min_worst"], "depth_worst": cav["depth_worst"],
        "SM": float(SM), "SM_min": prob.SM_min,
        "x_np": float(x_np), "x_cg": float(x_cg),
        "CL": float(res.CL), "CL_target": float(CLt),
        "CL_foil": CL_foil, "CL_stab": CL_stab,
        # the sections' own couple, both ways round (M/q [m^3] and the Cm it
        # contributes on (Sref, mac)), and which way up the stabiliser flies
        # its section — the geometry views and the CAD export read the flag,
        # so a core that mirrors the section in SILENCE would be drawn and
        # exported the right way up while flying the other one
        "M_ac_m3": float(cm_ac_model * S_x * mac),
        "Cm_ac": float(cm_ac_model),
        # the PRE-TRIM readout of the stabiliser's load, on the couple
        # actually flown (tail.trim_lift_coefficient's closed form: the two
        # trim equations already fix what each surface carries, so it needs
        # no solver). Reported because it is the number a card can quote
        # BEFORE a run, and the whole point of routing the rig couple through
        # cm_ac_model — which both authors read — is that this and the solved
        # CL_stab below cannot disagree about the SIGN.
        "CL_stab_pretrim": float(cl_stab),
        "tail_section_inverted": bool(isinstance(pol_t, _InvertedPolar)),
        "tail_section_follows_load": prob.tail_inverted is None,
        "lift_share_stab": (float(CL_stab * S_t / (res.CL * S_x))
                            if res.CL else float("nan")),
        "alpha_rad": float(alpha), "alpha_deg": float(np.rad2deg(alpha)),
        "i_t_rad": float(i_t), "i_t_deg": float(np.rad2deg(i_t)),
        "Cm_cg": float(res.Cm), "cm_residual": float(res.Cm),
        "cl_residual": float(res.CL - CLt),
        "CDi": float(res.CDi), "CDi_total": float(res.CDi),
        "CDp": float(CDp_foil + CDp_stab), "CDp_foil": CDp_foil,
        "CDp_stab": CDp_stab, "cd0_mast": float(cd0_mast),
        # the strut's TOTAL — parasite + its two corners + the induced drag
        # of the side force it carries. Equal to ``cd0_mast`` exactly where
        # the published charge is flown (``strut_model`` off), so a
        # breakdown that adds this instead cannot double-count.
        "CD_strut": float(CD_strut),
        "CD_junction": CD_junction, "CD": float(CD),
        "e": float(res.e), "e_total": float(res.e), "AR": float(res.AR),
        "alpha_eff_range_deg": (float(alpha_eff_deg.min()),
                                float(alpha_eff_deg.max())),
        "depth": depth, "V": V, "Fn_h": froude_depth(V, depth),
        "Re_mac": float(prob.rho * V * mac / prob.mu),
        "polar": getattr(pol, "name", "unknown"),
        "polar_tail": getattr(pol_t, "name", "unknown"),
        # the Reynolds number each surface's table was measured at, beside
        # the one the FOIL is flying (``Re_mac``). Two surfaces, two chords,
        # two tables once ``flown_reynolds`` is on.
        "section_re": (float(pol.Re) if getattr(pol, "Re", None) is not None
                       else None),
        "section_re_tail": (float(pol_t.Re)
                            if getattr(pol_t, "Re", None) is not None
                            else None),
        "S_t": S_t, "b_t": float(b_t), "AR_t": float(AR_t),
        "l_t": float(v["l_t"]),
        "z_t": z_t, "stab_depth_m": float(depth - z_t),
        "mac": float(mac), "tc": float(v["tc"]), "taper": float(foil.taper),
        "moment_trim": True, "wing": foil, "vlm": res,
    }
    if strut is not None:
        # WHERE IT STANDS, in the same frame as ``l_t`` (0 is the main
        # foil's quarter chord). ``api.design_report`` puts this on the fin
        # block as the quarter-chord station, which is the yaw arm the
        # flight rebuild flies — it was 0 for every craft before.
        #
        # ABSENT, not zero, where the published charge is flown: with
        # ``strut_model`` off the run is reproducing a recorded number, and
        # that number was reported by a fin block standing at x = 0. A
        # station emitted anyway would move the rebuilt craft's yaw
        # stiffness while claiming to reproduce the run.
        out["x_mast"] = float(v["x_mast"])
        out["x_mast_frac"] = float(v["x_mast_frac"])
        # THE STRUT AS A SURFACE. Reported whole, because three of these
        # numbers are new physics and a breakdown that showed only their sum
        # would hide which one moved: ``leeway_deg`` and ``CDi_side`` are
        # the rig's side load finally being carried by something, and
        # ``FF``/``Re`` are what make ``fin_tc`` a number instead of a label.
        out["strut"] = dict(strut)
        # ...and the two numbers a CARD has to be able to show, flat. The
        # presentation layer resolves breakdown keys by name and cannot walk
        # into a block (``gui/metrics.py``), and a shell that could only show
        # ``cd0_mast`` would print a mast charge that is no longer the mast's
        # total. Copied BY NAME from the block above — one author, two
        # spellings — rather than recomputed.
        out["CDi_strut_side"] = float(strut["CDi_side"])
        out["strut_leeway_deg"] = float(strut["leeway_deg"])
    if prob.rig is not None:
        # THE RIG, as flown. Everything here is a report — the couple has
        # already done its work through ``cm_ac_model`` — but two of these
        # entries are the point of the block rather than decoration:
        #
        # ``craft_LoD_stated`` beside ``appendage_LoD``. The first divided
        # the lift to give the drive force; the second is what this engine
        # thinks the appendage achieves. They differ by 3-4x and that is not
        # a bug in either (rig.py derives it): the engine models a wing and a
        # mast, the measurements are of a whole appendage. Printing them side
        # by side is the declared-omission idiom this repo already uses, and
        # it is the only defence against someone closing the loop.
        #
        # ``couple_as_cg_shift_m`` is the CG move that would produce the same
        # trim, reported as a DIAGNOSTIC and never as a station: it is how
        # large the couple is in this craft's own units (a third of a metre
        # against a 0.1225 m mean chord at the published centre, i.e. ~3 MAC).
        # rig.py explains why implementing it that way would be wrong.
        r = prob.rig
        out["rig"] = {
            "thrust_n": float(r.drive_n(prob.L_design)),
            "thrust_stated": r.thrust_n is not None,
            "side_n": float(r.side_n),
            "vertical_n": float(r.vertical_n),
            "z_ce_m": float(r.z_ce_m), "x_ce_m": float(r.x_ce_m),
            "lever_m": float(r.lever_m(depth)),
            "craft_LoD_stated": float(r.craft_lod),
            "appendage_LoD": float(LoD),
            "M_rig_nm": float(M_rig), "Cm_rig": float(cm_rig),
            "weight_n": float(prob.L_design),
            "lift_required_n": float(prob.L_required),
            "lift_relief_frac": (float(r.vertical_n) / float(prob.L_design)
                                 if prob.L_design else float("nan")),
            "couple_as_cg_shift_m": float(r.equivalent_cg_shift_m(
                weight_n=prob.L_design, depth_m=depth, x_cg_m=x_cg)),
        }
    if pol_t is not pol:
        # which surface the binding margin came off — with one TABLE the
        # question does not arise, with two it is the answer. Keyed on the
        # same test as the sweep above and for the same reason: a mirrored
        # shared section is a second table, so the two-sweep answer exists
        # and a report that stayed silent about it would be the readout
        # disagreeing with the run.
        out["cav_surface_worst"] = cav.get("surface_worst", "foil")
        out["g_cav_foil"] = cav.get("g_foil")
        out["g_cav_stab"] = cav.get("g_stab")
    if prob.winglet:
        # the FOIL's device: its own panels, so a tip device on the
        # stabiliser cannot inflate the height reported for this one
        z_top = float(res.z[wl_foil].max()) if wl_foil.any() else float(
            res.z[~tail_mask].max())
        out["winglet"] = {
            "h_frac": float(v["h_frac"]), "cant_deg": float(v["cant_deg"]),
            # the height that reached the lattice; h_frac stays the
            # REQUESTED number, so the pair reads asked / flown
            "h_m": float(model.winglet_h_m),
            "S_planform": S_wl,
            "tip_z_m": z_top,
            "tip_depth_m": float(depth - z_top),
            "blend_frac": float(prob.blend_frac_fixed),
            "blend_shape": str(prob.blend_shape),
            "wing_blend_frac": float(prob.wing_blend_frac),
            "wing_blend_arc_m": float(prob.wing_blend_frac * foil.b / 2.0),
            **({"junction": jr} if jr else {}),
        }
    if prob.tail_free:
        out["tail_planform"] = {
            "taper": float(v["taper_t"]), "AR": float(AR_t),
            "washout_deg": float(v["washout_t"]), "b_m": float(b_t),
            "S_m2": S_t,
            **({"chord_coeffs": list(stab_wing.chord_coeffs),
                "chord_dev": stab_wing.chord_dev}
               if stab_wing is not None and stab_wing.chord_coeffs else {}),
        }
    if prob.tail_winglet:
        z_top_t = (float(res.z[wl_stab].max()) if wl_stab.any()
                   else float(res.z[tail_mask].max()))
        out["tail_winglet"] = {
            "h_frac": float(v["h_frac_t"]),
            # the cant FLOWN, which in follow mode is the design variable's
            # magnitude on the side the stabiliser's own load chose
            "cant_deg": float(cant_t),
            "side": "up" if cant_t >= 0.0 else "down",
            "follows_load": followed is not None,
            # the height that reached the lattice; h_frac stays the
            # REQUESTED number, so the pair reads asked / flown
            "h_m": float(model.winglet_h_m_tail),
            "S_planform": float(np.sum((res.c * res.width)[wl_stab])),
            "tip_z_m": z_top_t,
            "tip_depth_m": float(depth - z_top_t),
            "blend_frac": prob.tail_blend_frac,
            "blend_shape": str(prob.blend_shape),
            **({"junction": jr_t} if jr_t else {}),
        }
    if foil.chord_coeffs:
        out["chord_coeffs"] = list(foil.chord_coeffs)
        out["chord_dev"] = foil.chord_dev
    return out


def fg_hydrofoil_tail(x: np.ndarray,
                      prob: HydrofoilTailProblem | None = None
                      ) -> tuple[float, np.ndarray]:
    """Constrained-harness callable: (L/D, [cavitation, SM[, draught]]).

    Width follows the problem — 2 uncapped (every published run), 3 when
    ``draught_max_m`` is set. The failure path must return the SAME width,
    because the optimiser's constraint block is sized up front off one call.
    """
    prob = prob or HydrofoilTailProblem()
    m = prob.n_constraints
    out = evaluate_hydrofoil_tail(x, prob)
    if not out["feasible"]:
        return PENALTY, np.full(m, G_FAIL)
    return float(out["LoD"]), np.asarray(out["g"], dtype=float)
