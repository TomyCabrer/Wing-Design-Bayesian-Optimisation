"""Car rear-wing ENDPLATES as a designed component — and as the mount.

Why this module exists
----------------------
carwing.py models the endplate the way a lifting-surface method naturally
does: a tip device of free height, at cant 90, carrying the wing's tip chord
and the wing's own section. That prices the nonplanar benefit and nothing
else. It cannot answer the question a rear wing actually poses, which is that
**the wing has to be attached to the car**. On a car whose wing is carried by
its endplates, the plate is not decoration around the tip vortex — it is the
load path, and its height is not free: it has to span from the wing down to
the bodywork it bolts to.

So here the endplate is a designed part with

    height          h_ep = endplate_h_m            (a LENGTH, in metres)
    chord           c_ep = endplate_chord_ratio * (wing tip chord)
    section         a family (:data:`SECTIONS`) + a thickness ratio t/c
    toe             endplate_toe_deg — the plate's own incidence

...and a corner, because the plate is bolted to a wing. ``blend_frac`` (a
FRACTION, 0 to 1, of the plate's height) softens the wing/plate junction the
way every other family softens its tip device's root, and here it has to
carry more than a path: the plate is a genuinely different surface — up to
three times the wing's tip chord, its own symmetric section, its own toe —
so its chord, section and toe RAMP over the turn (vlm.transition_ramp) rather
than stepping at the junction. Without that the model drew a smooth corner
between a 0.21 m chord and a 0.62 m one and then claimed a fillet credit for
it. A blend is not free either: the plate ends shorter (the reach constraint
tightens by what the turn consumed) and it reaches OUTBOARD, which the wing
pays for out of the span band — see the class docstring.

and three consequences that the tip-device model has no way to see:

1. REACH. The plate must get from the wing down to the car's attachment
   deck. With the wing at ``ride`` above the track and the deck at
   ``deck_height_m``, that is a hard geometric demand h_ep >= ride - deck
   whenever the mount is ``ends`` — reported as a signed constraint, not as
   a solver failure, so the optimiser can see which way to move. The other
   end of the window is the track itself (vlm.MIN_PLANE_CLEARANCE_FRAC).
   Raise the wing and the plate that carries it must grow.
2. DRAG. Wetted area (both faces, both plates), a thickness form factor and
   — for a constant-thickness plate — its square edges. Charged from a
   reduced-order build-up rather than from the wing's polar, because the
   wing's polar is a CAMBERED aerofoil's and a vertical plate is not.
3. STRUCTURE. The plate is deep in its own plane (it takes the wing's
   downforce and drag as in-plane shear) and weak out of it, so what sizes
   it is the LATERAL load: its own toe side force plus a yaw load case. The
   mount decides the beam — bolted to the car, the plate is a cantilever
   from the deck of length ``ride - deck``; hung off the wing (centre-pylon
   mount) it is a cantilever from the wing of length h_ep.

The section trap this module closes
-----------------------------------
A vertical surface given a CAMBERED aerofoil's zero-lift angle carries a
side force at zero toe: the panel normal is built with theta = twist -
alpha_L0, and alpha_L0 != 0 tilts a vertical panel just as surely as it
tilts a horizontal one. carwing.py's endplates inherit the wing's NACA 24XX
polar and therefore do exactly that (the two plates' side forces cancel in
CY, so it is invisible in the totals, and each plate still pays the induced
drag). Here the endplate carries its OWN section: symmetric, alpha_L0 = 0,
thin-aerofoil slope — so a plate at zero toe is a clean end fence, and any
side force is one the designer asked for.

Camber vs toe (a modelling decision, stated)
--------------------------------------------
To a linear method a cambered plate at zero toe and a symmetric plate at the
equivalent toe are the SAME surface: both enter only through the effective
angle. So camber is not a separate variable here — ``endplate_toe_deg``
carries it, and the section family sets the DRAG and the STIFFNESS, which is
where a cambered and a symmetric section genuinely differ. What is lost is
the drag bucket (a cambered section is cheaper at its design lift and dearer
away from it); that needs a section polar for the plate, which would mean
running the polar family at the plate's own Reynolds number, and is left as
an extension point rather than invented.

What is NOT modelled
--------------------
* The plate above the wing. Real endplates straddle the wing, and the part
  ABOVE it (in car terms) is aerodynamically live. The panel builder draws a
  single connected polyline from tip to tip, so a two-sided plate would be a
  branched bound-vortex system — legitimate but a different builder. Here
  the whole plate hangs on the track side of the wing, which is where most
  of a real one is and, crucially, is the part that reaches the car.
* Outwash and the plate's effect on the diffuser / wheel wake. The Trefftz
  plane sees the wake trace, not the flow field around the bodywork.
* Louvres, cut-outs, footplates, and any of the local devices that make real
  endplates the shape they are.
* Mass: the plate's own weight, and the cornering-inertia load it puts on
  the mount, need a mass model this package does not carry for cars.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from . import carmount, geometry, junction
from .carwing import (
    CAR_OBJECTIVES,
    G_FAIL,
    MOUNTS,
    PENALTY,
    RHO_AIR,
    MU_AIR,
    _strut_cd0,
    area_row,
    bending_moment,
    deflection_index,
    s_from_x,
    span_from_x,
    span_row,
)
from .drag import skin_friction_cf
from .vlm import VLM, ImagePlane

TWO_PI = 2.0 * np.pi

#: thickness ratio band offered to every section family. The family changes
#: the drag and stiffness LAWS, not which thicknesses are buildable, so the
#: box is shared — a flag that moved the bounds would leave the GUI's
#: (flag-free) default box describing a different problem than the one run.
TC_BOUNDS = (0.005, 0.20)

#: lumped drag coefficient of a constant-thickness plate's SQUARE EDGES, on
#: the edge area t*h. MODEL CHOICE: Hoerner's blunt-base coefficient is ~0.14
#: on the base area; the value here also stands in for the leading edge's
#: separation, which a square-cut plate has and a rounded one does not. It is
#: a calibrated lump, not a measurement, and it is the only thing that makes
#: the flat family lose to the shaped one at equal thickness.
FLAT_EDGE_CD = 0.20

#: the same lump for a constant-thickness plate whose EDGES ARE ROUNDED. A
#: quarter of the square-cut value: the rounded leading edge stops separating
#: (that is most of what the lump above stands in for) while the trailing edge
#: keeps a base, so this is a reduction, not a removal. MODEL CHOICE on the
#: same footing as FLAT_EDGE_CD — a calibrated lump, not a measurement — and
#: the ratio is the claim, not the absolute number.
ROUNDED_EDGE_CD = 0.05


def _naca_thickness_shape(xi: np.ndarray) -> np.ndarray:
    """NACA 4-digit symmetric half-thickness law / (t/c), unit chord."""
    xi = np.asarray(xi, dtype=float)
    return (0.2969 * np.sqrt(xi) - 0.1260 * xi - 0.3516 * xi**2
            + 0.2843 * xi**3 - 0.1015 * xi**4)


def inertia_factor_of(xi, t_hat) -> float:
    """k_I of a solid section from its own THICKNESS DISTRIBUTION.

    I = int_0^c t(x)^3/12 dx = (c t_max^3/12) * int_0^1 (t/t_max)^3 dxi, so
    k_I is that dimensionless integral: bending about the chord line, which
    is the plate's weak axis and the one the lateral load works on.

    Split out of :func:`_naca_inertia_factor` so that a section given as
    COORDINATES — the aerofoil the plate is designed with, rather than one of
    the three construction families — gets its stiffness from its own shape
    instead of borrowing a family's constant. Same integral either way, and
    the NACA number below is unchanged to the last bit.

    ``xi`` need not be sorted or normalised; ``t_hat`` is any positive
    thickness in any units (it is divided by its own maximum).
    """
    xi = np.asarray(xi, dtype=float)
    t_hat = np.asarray(t_hat, dtype=float)
    if xi.shape != t_hat.shape or xi.size < 3:
        raise ValueError(
            f"need matching xi/thickness arrays of at least 3 points, got "
            f"{xi.shape} and {t_hat.shape}")
    order = np.argsort(xi)
    xi, t_hat = xi[order], t_hat[order]
    span = float(xi[-1] - xi[0])
    t_max = float(np.max(t_hat))
    if not (span > 0.0 and t_max > 0.0):
        raise ValueError(
            f"need a positive chord and a positive maximum thickness "
            f"(got span {span}, t_max {t_max})")
    xi = (xi - xi[0]) / span
    return float(np.trapezoid((t_hat / t_max) ** 3, xi) / 12.0)


def section_inertia_factor(coords) -> float:
    """k_I of a section given as its (x, y) outline — a DESIGNED aerofoil.

    The generalisation :func:`_naca_inertia_factor` is the closed-form case
    of. The outline is split at its leading edge (the minimum x) into an
    upper and a lower branch, both are interpolated onto one chordwise grid,
    and the local thickness is their difference. That is the only way to get
    a plate's stiffness from a shape the package DESIGNED rather than from a
    family label: without it a searched aerofoil would have to borrow
    ``SECTIONS["shaped"]``'s NACA constant, and a section is chosen partly
    for the stiffness it buys.

    Raises rather than guessing on an outline that is not a single-valued
    aerofoil — a k_I that silently describes a different shape is exactly
    what :func:`_naca_inertia_factor` was written to avoid.
    """
    pts = np.asarray(coords, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 5:
        raise ValueError(
            f"need an (N, 2) outline of at least 5 points, got {pts.shape}")
    x, y = pts[:, 0], pts[:, 1]
    i_le = int(np.argmin(x))
    if i_le in (0, len(x) - 1):
        raise ValueError(
            "the outline's leading edge is at one of its ends: this is not a "
            "closed aerofoil loop running trailing edge -> nose -> trailing "
            "edge")
    grid = np.linspace(float(x[i_le]), float(np.max(x)), 400)

    def _branch(sl) -> np.ndarray:
        xs, ys = x[sl], y[sl]
        order = np.argsort(xs)
        return np.interp(grid, xs[order], ys[order])

    t = np.abs(_branch(slice(0, i_le + 1)) - _branch(slice(i_le, None)))
    return inertia_factor_of(grid, t)


def _naca_inertia_factor(n: int = 200001) -> float:
    """k_I of a solid NACA symmetric section: I = k_I * c * t_max^3.

    Bending about the chord line of a plate-like section whose thickness
    varies along the chord: I = int_0^c t(x)^3/12 dx = (c t_max^3/12) *
    int_0^1 (t/t_max)^3 dxi. Computed from the thickness law itself rather
    than quoted, so the number cannot drift from the shape it describes
    (gated in tests: ~0.0394, i.e. 47% of a rectangle's 1/12 — a shaped
    section spends its material where the bending stress is lowest).
    """
    xi = np.linspace(0.0, 1.0, int(n))
    return inertia_factor_of(xi, _naca_thickness_shape(xi))


#: section families. ``inertia_factor`` k_I gives I = k_I * c * t^3 for
#: bending about the chord line (the plate's WEAK axis, which is the one the
#: lateral load works on). ``streamlined`` selects the drag law; ``edge_cd``
#: is the lumped edge coefficient a section that is NOT streamlined pays on
#: its edge area t*h (a streamlined one has no edges to charge and pays a
#: thickness form factor instead).
#:
#: Read as a set they are a BRACKET, and the middle entry is the one a real
#: rear wing usually bolts on. Measured over this family's own box, ``shaped``
#: has strictly the lower drag at every thickness offered, so ``flat`` can
#: only ever win through stiffness — 2.1x, and decisive only below t/c ~ 0.01
#: where the plate's deflection limit is live. That corner is exactly where
#: ``rounded`` lives: it keeps the constant-thickness inertia and pays a
#: quarter of the edge penalty, which is the part the choice was really
#: trading. Offering only the two ends made a thin plate choose between full
#: edge drag and 47% of its stiffness; the part in the paddock does neither.
#: WHICH SCALARS THIS FAMILY CAN SCORE — ``carwing.CAR_OBJECTIVES`` minus the
#: one that is a TASK rather than a figure of merit.
#:
#: ``laptime`` is scored from ``lap_time_s``, which is a property of a wing ON
#: A CIRCUIT, and this family has no circuit: :class:`CarWingEndplateProblem`
#: carries no ``track_spec``, no ``car_spec`` and no ``track_points``, and
#: ``api.CAR_ENDPLATE_KEYS`` declares neither lap key for exactly that reason.
#: Validating against the full ``CAR_OBJECTIVES`` therefore ACCEPTED the name
#: at construction and raised ``KeyError('laptime')`` out of the scorer's own
#: lookup at the first evaluation — an in-contract question answered with a
#: programming error, and invisible until something ran.
#:
#: Same shape and the same words as ``carwing_multi.CAR_MULTI_OBJECTIVES``,
#: which was written for the same hole one family across. The two are kept
#: apart rather than merged because they are different SETS: the slotted
#: family gained a circuit and can time a lap, this one cannot, and a single
#: shared set would have to be the intersection or the union — one of which
#: hides a capability and the other of which is the bug above.
ENDPLATE_OBJECTIVES: dict[str, str] = {
    k: v for k, v in CAR_OBJECTIVES.items() if k != "laptime"}


SECTIONS: dict[str, dict] = {
    "flat": {
        "label": "flat plate (constant thickness, square edges)",
        "inertia_factor": 1.0 / 12.0,          # rectangle, exactly
        "streamlined": False,
        "edge_cd": FLAT_EDGE_CD,
        "note": "constant thickness: the stiffest section per unit thickness "
                "(all the material at full depth) and the draggiest, because "
                "the square edges carry a base/separation penalty no form "
                "factor can remove",
    },
    "rounded": {
        "label": "flat plate, rounded edges (constant thickness)",
        "inertia_factor": 1.0 / 12.0,          # same rectangle as flat
        "streamlined": False,
        "edge_cd": ROUNDED_EDGE_CD,
        "note": "the plate most rear wings actually carry: a constant-"
                "thickness panel with radiused edges. Full flat-plate bending "
                "inertia, and a quarter of the square-cut edge penalty — the "
                "rounded nose stops separating, the trailing edge still has a "
                "base. Still no form factor: there is no streamwise pressure "
                "recovery to charge for",
    },
    "shaped": {
        "label": "streamlined symmetric section (NACA 00xx-like)",
        "inertia_factor": _naca_inertia_factor(),
        "streamlined": True,
        "edge_cd": 0.0,
        "note": "thickness tapers to the edges: ~47% of a flat plate's "
                "bending inertia at the same maximum thickness, in exchange "
                "for attached flow and a thickness form factor instead of an "
                "edge penalty",
    },
}


def section_spec(name: str) -> dict:
    if name not in SECTIONS:
        raise ValueError(
            f"unknown endplate section {name!r}; "
            f"choose from {sorted(SECTIONS)}")
    return SECTIONS[name]


def bending_inertia(section: str, chord: float, thickness: float) -> float:
    """Second moment of area [m^4] about the chord line (the weak axis)."""
    if not (float(chord) > 0.0 and float(thickness) > 0.0):
        raise ValueError(
            f"chord and thickness must be > 0 (got {chord}, {thickness})")
    return float(section_spec(section)["inertia_factor"]
                 * float(chord) * float(thickness) ** 3)


def form_factor(section: str, tc: float) -> float:
    """Profile-drag form factor of the plate's section.

    Streamlined sections use the classical strut correlation
    FF = 1 + 2(t/c) + 60(t/c)^4 (Hoerner, *Fluid-Dynamic Drag*, ch. 6; the
    same expression Raymer 2018 eq. 12.32 gives for struts and pylons). A
    constant-thickness plate gets FF = 1: it has no streamwise pressure
    recovery to charge for — its thickness is charged as edge drag instead.
    """
    tc = float(tc)
    if not (0.0 < tc < 1.0):
        raise ValueError(f"thickness ratio must be in (0, 1), got {tc}")
    if not section_spec(section)["streamlined"]:
        return 1.0
    return float(1.0 + 2.0 * tc + 60.0 * tc**4)


def parasite_cd(section: str, chord: float, height: float, tc: float,
                s_ref: float, rho: float, V: float, mu: float,
                n_plates: int = 2, cd_section: float | None = None) -> dict:
    """Endplate parasite drag as a coefficient on ``s_ref``, with its parts.

    TWO WAYS TO CHARGE THE SAME DRAG, and which one is used is whether the
    plate has an AEROFOIL.

    Without one (``polar=None``, every published run) the plate is one of the
    three construction families and the charge is a build-up: wetted area is
    BOTH faces of each plate, skin friction is the turbulent flat-plate value
    at the plate's own Reynolds number (drag.py, the same correlation the
    aircraft build-up uses), a form factor stands in for the thickness of a
    streamlined one and a lumped edge coefficient for the square edges of one
    that is not. The alpha-dependence of the plate's own profile drag is not
    modelled — a symmetric section at the few degrees of toe this problem
    allows sits in its drag bucket.

    WITH one, the drag bucket is no longer assumed: ``cd_section`` is the
    plate's own section drag coefficient, integrated panel by panel over the
    plate at the angles the lattice says each panel flies, from a polar
    measured at the PLATE's own Reynolds number. It replaces the whole
    friction-times-form-factor term. That is the extension point this
    module's docstring named, and it is the only way a DESIGNED plate
    aerofoil can be worth designing: a build-up is blind to the shape, so
    every section of a given thickness would score identically. Referenced
    the way a two-dimensional cd is defined — on the PLANFORM area
    chord*height, not on the wetted area — and the edge term is dropped,
    because a section with coordinates has no square edges to lump.

    Either way the plate's INDUCED drag is fully in the VLM's Trefftz plane
    and must not be added again here.
    """
    chord, height, tc = float(chord), float(height), float(tc)
    if height <= 0.0 or n_plates <= 0:
        return {"CD": 0.0, "CD_friction": 0.0, "CD_edges": 0.0,
                "Swet_m2": 0.0, "Cf": 0.0, "FF": form_factor(section, tc),
                "Re": 0.0, "cd_section": 0.0, "designed": False}
    Re = rho * V * chord / mu
    swet = n_plates * 2.0 * chord * height
    if cd_section is not None:
        cd2d = float(cd_section)
        # a section cd is per unit PLANFORM area, and both faces of the plate
        # are already inside it — so the reference is chord*height per plate,
        # never the wetted area, which would double the charge
        cd_f = cd2d * n_plates * chord * height / s_ref
        return {"CD": float(cd_f), "CD_friction": float(cd_f),
                "CD_edges": 0.0, "Swet_m2": float(swet),
                "Cf": float(skin_friction_cf(Re, lref=chord)), "FF": 1.0,
                "Re": float(Re), "cd_section": cd2d, "designed": True}
    cf = skin_friction_cf(Re, lref=chord)
    ff = form_factor(section, tc)
    cd_f = cf * ff * swet / s_ref
    cd_e = 0.0
    if not section_spec(section)["streamlined"]:
        # exposed edges: a lumped base/leading-edge coefficient on t*h, the
        # family's own (square-cut pays four times what a radiused edge does)
        cd_e = (section_spec(section)["edge_cd"]
                * n_plates * (tc * chord) * height / s_ref)
    return {"CD": float(cd_f + cd_e), "CD_friction": float(cd_f),
            "CD_edges": float(cd_e), "Swet_m2": float(swet),
            "Cf": float(cf), "FF": float(ff), "Re": float(Re),
            "cd_section": 0.0, "designed": False}


def yaw_side_force(q: float, area: float, aspect_ratio: float,
                   yaw_deg: float, a_2d: float = TWO_PI) -> float:
    """Side force [N] on ONE plate in a steady-sideslip load case.

    The plate is a low-aspect-ratio lifting surface seeing the sideslip
    angle as its incidence, with the classical finite-span correction

        a_eff = a_2d / (1 + a_2d / (pi AR)),   AR = height / chord.

    The GEOMETRIC aspect ratio is used. The plate is joined to the wing at
    one end, which acts as a partial reflection plane and would raise the
    effective AR (and so the load); using the geometric value is therefore
    the LESS conservative choice on the structure and the more conservative
    one on the aerodynamic benefit. Stated rather than silently assumed.
    """
    ar = max(float(aspect_ratio), 1e-6)
    a_eff = a_2d / (1.0 + a_2d / (np.pi * ar))
    return float(q * float(area) * a_eff * np.deg2rad(float(yaw_deg)))


def lateral_deflection(load: float, arm: float, E: float, I: float) -> float:
    """Tip deflection [m] of a cantilever of length ``arm`` under a tip load.

    delta = P L^3 / (3 E I). The lateral load is applied at the wing station
    — i.e. at the full arm — rather than distributed over the plate: the
    worst case for the quantity that matters, which is how far the WING
    moves sideways relative to where it is bolted.
    """
    if not (float(E) > 0.0 and float(I) > 0.0):
        raise ValueError(f"need E, I > 0 (got {E}, {I})")
    return float(abs(load) * float(arm) ** 3 / (3.0 * float(E) * float(I)))


def root_stress(load: float, arm: float, thickness: float, I: float) -> float:
    """Bending stress [Pa] at the built-in end: sigma = M (t/2) / I."""
    if not (float(I) > 0.0):
        raise ValueError(f"need I > 0 (got {I})")
    return float(abs(load) * float(arm) * (0.5 * float(thickness))
                 / float(I))


# ------------------------------------------------------------------ problem

@dataclass
class CarWingEndplateProblem:
    """Rear wing whose ENDPLATES are designed, and may be the mount.

    Design vector (the first six entries are carwing.CarWingProblem's, so the
    two problems can be read side by side; the SPAN closes it, as the size
    block the package's stacking rule puts ahead of the chord law)::

        x = [taper, twist_root_deg, twist_tip_deg, alpha_deg,
             endplate_h_m, ride_height_m,
             endplate_chord_ratio, endplate_tc, endplate_toe_deg,
             (S_m2,) b_m]

    ``S_m2`` is there only when :attr:`area_bounds_m2` is set — the size block
    is ``[S, b]`` free and ``[b]`` fixed, with the area AHEAD so the span
    keeps its slot either way (carwing.s_from_x / carwing.span_from_x, and
    the reason is written there).

    The plate's height is a LENGTH and the span is a design variable — see
    carwing.CarWingProblem, which says why, and note how much harder the
    reach constraint below leans on it: ``g4`` compares the plate's tip
    height with ``ride - deck``, a distance in metres between two pieces of
    car. Expressed as a fraction of a semi-span that the optimiser is also
    moving, the same plate would reach the deck or not depending on how wide
    the wing happened to be that iteration.

    ``b_m`` is the OVERALL WIDTH — the wing and whatever its plates project.
    What bounds a rear wing is a regulation or the bodywork, and neither
    measures the wing and then ignores what is bolted to its tips. A plate at
    cant 90 projects nothing, so with no blend the wing spans the whole box
    and this is the published problem exactly; a BLENDED plate leaves the wing
    plane tangentially and reaches outboard (0.27 m a side at full blend, on
    the published box), and the wing shortens to pay for it — on the same
    developed line the span-capped aircraft families use. Without that, blend
    bought free width: a 1.6 m wing measuring 2.15 m across inside a 2.0 m
    band. ``overall_width_m`` and ``b_m`` in the breakdown are those two
    numbers, and ``endplate_projection_m`` is the difference.

    The reference area ``S`` is CONFIGURATION BY DEFAULT, exactly as in
    carwing.CarWingProblem and for the same reason: CZ, the drag budget and
    the reported efficiency are all referenced to it. Set
    :attr:`area_bounds_m2` and it becomes a design variable, and then the
    score has to be stated in FORCES — :attr:`objective`, carwing.CAR_OBJECTIVES.

    Objective (MAXIMISE): :attr:`objective`, default the downforce
    coefficient CZ (the published one).
    Constraints (all >= 0, and only the ones this problem DECLARES — see
    :attr:`constraint_labels`, which is what ``api.design_report`` reads):
        CD_budget - CD                          [iff ``CD_budget``]
             a drag COEFFICIENT allowance. Switchable off, because an
             allowance referenced to an area that is now moving is not an
             allowance on anything the car can feel.
        1 - D / drag_budget_n                   [iff ``drag_budget_n``]
             the same allowance in NEWTONS — the one to use at a free area.
        1 - wing deflection / limit             [always] the wing's own beam
        1 - endplate deflection / limit         [always] the plate as a
             structure
        endplate reach margin                   [always] can it get to the car?
             h_ep(vertical) / (ride - deck) - 1, on EVERY layout. Both
             layouts take the load out through the sheets, so there is no
             arrangement in which the plate is free to be as short as the
             aerodynamics likes. It used to return a dormant +1 under the
             pylon layout, reported as ``reach_required`` so the run could
             not be misread; that flag is now a constant True and is kept
             only so a reader of an older stored run still finds the key.
        F_z / downforce_min_n - 1               [iff ``downforce_min_n``]
             a downforce FLOOR, which is what makes ``objective="efficiency"``
             the question a race engineer actually has.

    Frame: carwing.py's mirrored frame throughout (model +z is the car's
    downward direction; the track is a rigid-wall ImagePlane above the wing;
    model lift IS downforce). The endplate at cant +90 therefore points at
    the track, which is the direction the car is in.
    """

    b: float = 1.6
    S: float = 0.4
    N_vlm: int = 40
    n_endplate: int = 10
    V: float = 55.0
    rho: float = RHO_AIR
    mu: float = MU_AIR
    tc: float = 0.12                # WING section thickness
    #: which plate-borne layout (carwing.MOUNTS) — where along the span the
    #: sheets grip. This family's whole subject is the plate AS the mount, and
    #: both layouts now are, so the default is the tips: the plates it already
    #: designs, carrying at their own station and costing nothing extra.
    mount: str = "tips"
    section: str = "shaped"         # endplate section family (SECTIONS)
    deck_height_m: float = 0.25     # car attachment hardpoint above the track
    strut_chord: float = 0.12
    strut_cf: float = 0.005
    ei_nm2: float = 4.0e4           # WING spar bending stiffness EI
    E_endplate_pa: float = 7.0e10   # plate material modulus (Al / stiff
    #                                 laminate). SOLID section assumed: a
    #                                 sandwich panel of the same depth is
    #                                 lighter for the same stiffness, so this
    #                                 is an upper bound on stiffness per unit
    #                                 thickness, and it is a constraint SHAPE.
    yaw_deg: float = 3.0            # sideslip load case for the plate
    endplate_alpha_max_deg: float = 9.0
    #: linear-range gate on the PLATE's own effective angle. A DESIGN GATE in
    #: the same status as tail.i_t_max_deg, not solver output: past the knee
    #: of a thin symmetric section's lift curve a linear VLM has nothing to
    #: say, and toeing the plate further would keep buying downforce for
    #: free. Calibrated on the model itself — at the +/-6 deg toe box the
    #: worst panel sits at 7.0 deg (cl 0.76) once the wing's sidewash is
    #: added on top, so this gate is just outside the offered box and only
    #: bites if the box is widened.
    #: drag COEFFICIENT allowance, or None for "maximise it and do not budget
    #: it" (carwing.CarWingProblem.CD_budget, same switch and same reason).
    CD_budget: float | None = 0.13
    #: drag FORCE allowance [N], or None. The area-independent budget:
    #: D = q S CD <= drag_budget_n, and the one that means something once the
    #: reference area is a design variable.
    drag_budget_n: float | None = None
    #: downforce FLOOR [N], or None. What makes ``objective="efficiency"``
    #: useful rather than merely well-posed: without a floor the most
    #: efficient rear wing is a small one making little downforce.
    downforce_min_n: float | None = None
    #: WHICH SCALAR is maximised (carwing.CAR_OBJECTIVES). ``cz`` is the
    #: published objective and the default.
    objective: str = "cz"
    #: make the reference AREA a design variable, over this band in m^2
    #: (None = the published FIXED area :attr:`S`). Adds ONE row, immediately
    #: ahead of the span row, so :func:`carwing.span_from_x` is unchanged —
    #: and note that the span row here is the OVERALL WIDTH, so the aspect
    #: ratio this area is checked against is the WING's, after the plates
    #: have taken their projection out of it.
    area_bounds_m2: tuple | None = None
    deflection_limit_m: float = 0.010        # wing, relative to its mount
    endplate_deflection_limit_m: float = 0.005   # wing, sideways
    blend_frac: float = 0.0
    blend_shape: str = "arc"        # arc | smooth (geometry.BLEND_SHAPES)
    #: the plate's cant, as carwing.CarWingProblem documents it. 90 is the
    #: published plate. There is deliberately no ``endplate_chord_follows``
    #: here: on THIS family the plate's chord is a design row of its own
    #: (``endplate_chord_ratio``, clipped into a metre band), so continuing
    #: the wing's law onto it would be a second answer to a question the
    #: search is already answering.
    endplate_cant_deg: float = 90.0
    #: ...or SEARCH it instead of stating it. A DIMENSION change, which is why
    #: it is a registry variant axis (``api.PLATE_FREEDOMS``) and not a flag:
    #: the cant becomes a row of the family block and ``endplate_cant_deg``
    #: above stops being read. Every consumer of the angle — the width
    #: accounting, the lattice, the structural arm, the reach margin and the
    #: junction radius — then takes the candidate's own value.
    #:
    #: WHY IT IS WORTH A DESIGN VARIABLE. On this family the cant is priced
    #: at BOTH ends and in opposite directions, so there is an interior
    #: answer rather than a bound to ride: leaning the plate out projects it
    #: OUTBOARD, which comes straight out of the span row because ``b_m``
    #: here is the OVERALL width, and it shortens the plate's vertical reach
    #: (h·sin), which the reach margin charges against the deck. Against that
    #: it buys the nonplanar benefit the plate exists for. Nothing else in
    #: the package has to be told what a rear wing's plates lean by — and
    #: nothing in the repository records a number for it, which is exactly
    #: the case for asking the optimiser rather than the user.
    cant_free: bool = False
    #: the same for the wing/plate CORNER. ``blend_frac`` above becomes a row
    #: over [0, 1] and the search trades the corner's interference charge
    #: against a plate that ends SHORTER (the reach margin tightens) and
    #: reaches OUTBOARD (the span row pays). ``blend_shape`` stays a stated
    #: VALUE either way: it names WHICH LAW the turn follows
    #: (geometry.BLEND_SHAPES), not how much of one there is, and a search
    #: over a categorical law is a different kind of question.
    blend_free: bool = False
    #: the plate's chord in METRES, where a rule states one. The design
    #: variable is a RATIO of the wing's tip chord, which is the right way to
    #: ask for a plate proportioned to its wing and the wrong way to state a
    #: box the part has to fit inside: the tip chord moves with the span, the
    #: taper and the chord law, so the same ratio draws a different part every
    #: iteration. Both bands are therefore offered and both are live — the
    #: ratio band is the design box's own row (ENDPLATE_CHORD_RATIO_BOUNDS),
    #: this one is the metre box — and the flown chord is the ratio's chord
    #: CLAMPED into these limits. The metre band wins where the two disagree,
    #: because it is the physical one; ``endplate_chord_requested_m`` and
    #: ``endplate_chord_clamped`` in the breakdown say when it bit, so a
    #: clamped design can never be read as the one that was asked for.
    #: None = unconstrained, which is every published run.
    endplate_chord_min_m: float | None = None
    endplate_chord_max_m: float | None = None
    junction_drag: bool = True
    chord_order: int = 0            # free chord law (geometry.py)
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
    #: fly a CHOSEN section (carwing.CarWingProblem.section_polar's twin).
    #: None keeps the published t/c lookup; the vector is unchanged either
    #: way, since this family's thickness was never a design variable.
    section_polar: Any = None
    #: THE PLATE'S OWN AEROFOIL — a section polar measured at the PLATE's
    #: Reynolds number, not the wing's. None keeps the build-up every
    #: published run flies (:func:`parasite_cd`, the three :data:`SECTIONS`
    #: families); given one, the plate's profile drag is that section's own
    #: cd at the angle the lattice says it flies, and its bending inertia is
    #: integrated from the section's coordinates
    #: (:func:`section_inertia_factor`) instead of borrowing a family's
    #: constant. This is the extension point the module docstring names.
    #:
    #: IT MUST BE SYMMETRIC. A cambered section has alpha_L0 != 0, and a
    #: vertical panel built at theta = twist - alpha_L0 carries a side force
    #: at zero toe — the trap this whole module exists to close. Checked, not
    #: assumed: a polar whose zero-lift angle is not zero is refused at
    #: construction rather than flown.
    endplate_polar: Any = None
    #: the coordinates ``endplate_polar`` was measured on, as (N, 2), where
    #: the caller has them. Optional and only ever used for STIFFNESS: with
    #: them the plate's k_I is the section's own integral, without them it
    #: falls back to the streamlined family's NACA constant and says so in
    #: the breakdown (``endplate_inertia_source``). A polar is a table of
    #: forces and carries no shape, so this cannot be derived from it.
    endplate_coords: Any = None
    #: span band [m] (carwing.SPAN_BOUNDS_M when None)
    span_bounds_m: tuple | None = None

    TAPER_BOUNDS = (0.4, 1.0)
    TWIST_ROOT_BOUNDS_DEG = (-4.0, 4.0)
    TWIST_TIP_BOUNDS_DEG = (-6.0, 2.0)
    ALPHA_BOUNDS_DEG = (0.0, 12.0)
    #: endplate arc length [m] — tall enough to reach the deck from the top
    #: of the ride-height band (0.70 - 0.25 = 0.45 m) with room over it, and
    #: the published fraction band (0-0.75 of the 1.6 m span's semi-span)
    #: evaluated at that span, so the family opens on the same plates
    ENDPLATE_H_M_BOUNDS = (0.0, 0.60)
    RIDE_HEIGHT_BOUNDS_M = (0.30, 0.70)      # above the deck by construction
    ENDPLATE_CHORD_RATIO_BOUNDS = (0.5, 3.0)
    ENDPLATE_TC_BOUNDS = TC_BOUNDS
    ENDPLATE_TOE_BOUNDS_DEG = (-6.0, 6.0)
    #: the SEARCHED cant's band. The nonplanar lattice's own validity and
    #: nothing narrower (``geometry.WINGLET_CANT_LIMITS_DEG``, 5-90), which is
    #: the same source the shell's typed field takes its limits from — one
    #: number, one owner. The shallow end of it is not a flyable corner at
    #: every ride height, and that is deliberately left to the REACH MARGIN
    #: to say: below about 48 deg the top of the ride band stops being
    #: reachable at any plate height the box offers, and the run reports that
    #: as a refusal it can see the sign of rather than a bound it cannot ask
    #: past. A calibration is a default, not a ban.
    ENDPLATE_CANT_BOUNDS_DEG = geometry.WINGLET_CANT_LIMITS_DEG
    #: ...and the searched BLEND's, the full fraction of the plate's height
    #: the wing/plate corner may be turned over. Both ends are real: 0 is the
    #: crease every published run flies, 1 is a plate that is all turn.
    ENDPLATE_BLEND_BOUNDS = (0.0, 1.0)

    #: the rows the two freedoms add, in the order they are appended to the
    #: FAMILY block — read by :attr:`param_labels` and by the evaluator's
    #: unpack, so the label and the number can never disagree about which is
    #: which.
    FREE_ROWS = (("cant_free", "endplate_cant_deg", "ENDPLATE_CANT_BOUNDS_DEG"),
                 ("blend_free", "endplate_blend_frac", "ENDPLATE_BLEND_BOUNDS"))

    #: the 9 rows every variant carries, ahead of whatever the freedoms add
    BASE_LABELS = ("taper", "twist_root_deg", "twist_tip_deg", "alpha_deg",
                   "endplate_h_m", "ride_height_m", "endplate_chord_ratio",
                   "endplate_tc", "endplate_toe_deg")

    def __post_init__(self):
        if self.mount not in MOUNTS:
            raise ValueError(
                f"unknown mount {self.mount!r}; choose from {sorted(MOUNTS)}")
        if self.blend_shape not in geometry.BLEND_SHAPES:
            raise ValueError(
                f"unknown blend_shape {self.blend_shape!r}; "
                f"choose from {list(geometry.BLEND_SHAPES)}")
        section_spec(self.section)               # validates the family
        # ONE ANSWER PER QUESTION. api drops the stated flag on the variant
        # that searches it (the rule api._WING_CANTS obeys), and this is the
        # belt to that braces: a caller reaching the dataclass directly still
        # cannot state a number the design vector is already choosing.
        if self.cant_free and float(self.endplate_cant_deg) != 90.0:
            raise ValueError(
                f"cant_free=True searches the plate's cant, so stating "
                f"endplate_cant_deg={self.endplate_cant_deg} is a second "
                f"answer to the same question. Drop one: state the angle on "
                f"the fixed-cant family, or let this one choose it")
        if self.blend_free and float(self.blend_frac) != 0.0:
            raise ValueError(
                f"blend_free=True searches the wing/plate corner, so stating "
                f"blend_frac={self.blend_frac} is a second answer to the "
                f"same question. blend_SHAPE is still yours to state — it "
                f"names the law, not how much of it")
        if self.endplate_polar is not None:
            self._check_symmetric()
            # NOTE the section's own thickness does NOT overwrite anything
            # here, unlike `section_polar` above: the plate's t/c is a DESIGN
            # ROW (`endplate_tc`), so a chosen section's thickness reaches it
            # as a PIN from the shell (api.RunConfig.pinned) — a row cannot be
            # both searched and stated, and pinning is how this package states
            # one that exists.
        for name in ("endplate_chord_min_m", "endplate_chord_max_m"):
            v = getattr(self, name)
            if v is not None:
                v = float(v)
                setattr(self, name, v)
                if not v > 0.0:
                    raise ValueError(f"{name} must be > 0, got {v}")
        if (self.endplate_chord_min_m is not None
                and self.endplate_chord_max_m is not None
                and self.endplate_chord_min_m > self.endplate_chord_max_m):
            raise ValueError(
                f"endplate_chord_min_m {self.endplate_chord_min_m} exceeds "
                f"endplate_chord_max_m {self.endplate_chord_max_m}")
        lo, _hi = self.RIDE_HEIGHT_BOUNDS_M
        if not (0.0 < self.deck_height_m < lo):
            raise ValueError(
                f"deck_height_m must be in (0, {lo}) so every ride height in "
                f"the box leaves a positive gap to the car (got "
                f"{self.deck_height_m})")
        if self.polar_family is None and self.section_polar is None:
            from .polar import default_polar_family
            self.polar_family = default_polar_family()
        if self.section_polar is not None:
            self.tc = float(getattr(self.section_polar, "tc", self.tc))
        if self.objective not in ENDPLATE_OBJECTIVES:
            extra = ""
            if self.objective in CAR_OBJECTIVES:
                extra = (
                    f" — {self.objective!r} is declared by "
                    f"carwing.CAR_OBJECTIVES but this family has no scorer "
                    f"for it (ENDPLATE_OBJECTIVES is the set it can score). "
                    f"A lap time is a property of a wing ON A CIRCUIT and "
                    f"this family has no track_spec to time one over; the "
                    f"two that do are carwing.CarWingProblem and "
                    f"carwing_multi.CarWingMultiProblem")
            raise ValueError(
                f"unknown objective {self.objective!r}; "
                f"choose from {sorted(ENDPLATE_OBJECTIVES)}{extra}")
        if self.area_free and self.objective == "cd":
            # carwing.CarWingProblem's refusal, verbatim in reason: a drag
            # COEFFICIENT referenced to the area being searched is minimised
            # by growing the wing.
            raise ValueError(
                "objective 'cd' is meaningless with a free reference area: "
                "the drag COEFFICIENT is referenced to the very area being "
                "searched, so it is minimised by growing the wing. Use "
                "objective='drag' for the force (with a downforce_min_n "
                "floor) — or fix the area (area_bounds_m2=None). Same "
                "refusal, same reason, as carwing.CarWingProblem")
        if self.area_free and self.objective == "cz":
            # the same refusal carwing.CarWingProblem makes, for the same
            # reason: CZ is referenced to S, so maximising it with S free is
            # maximised by shrinking S, and the answer comes back looking
            # like a result. State the question in forces instead.
            raise ValueError(
                "objective 'cz' is meaningless with a free reference area: "
                "the downforce COEFFICIENT is referenced to the very area "
                "being searched, so it is maximised by shrinking the wing. "
                "Use objective='downforce' for the force, or 'efficiency' "
                "with a downforce_min_n floor — or fix the area "
                "(area_bounds_m2=None)")
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

    #: how far a plate's zero-lift angle may sit from zero [deg], where the
    #: section arrives as a POLAR and there are no coordinates to measure.
    #:
    #: DERIVED, not chosen: the plate's incidence is a design variable over
    #: :attr:`ENDPLATE_TOE_BOUNDS_DEG`, so the question "is this section
    #: symmetric enough" is really "could its zero-lift angle command a side
    #: force the toe row cannot trivially cancel". One per cent of that box's
    #: half-width is the answer — at a thin-aerofoil slope 0.06 deg is cl
    #: 0.0066 against the toe box's own +/-0.66 — and a measured polar's
    #: alpha_L0 is a fit through XFOIL points, so it carries noise of about
    #: this size on a section that is symmetric by construction (a library
    #: NACA 00xx fits at -0.009 deg, which a stricter gate refused).
    ALPHA_L0_TOL_DEG = 0.01 * max(abs(b) for b in ENDPLATE_TOE_BOUNDS_DEG)

    def _check_symmetric(self) -> None:
        """Refuse a CAMBERED section on the plate — the trap this module is
        built around.

        A cambered section's alpha_L0 tilts a vertical panel exactly as it
        tilts a horizontal one, so a plate at zero toe would carry a side
        force nobody asked for; and the two plates' side forces cancel in CY,
        so the design would look clean while each plate paid the induced drag.

        ONE DEFINITION OF SYMMETRIC, and it is the library screen's. Where the
        coordinates are given, symmetry is the same measurement
        ``api.symmetric_section_names`` makes to decide what may be OFFERED
        (max mean-line camber against ``api.SYMMETRIC_CAMBER_TOL``) — a
        second, stricter test here would refuse sections the shell had just
        put in the menu, which is one question with two answers. Only where a
        polar arrives with no shape behind it does the zero-lift angle stand
        in, against :attr:`ALPHA_L0_TOL_DEG`.
        """
        if self.endplate_coords is not None:
            from .api import SYMMETRIC_CAMBER_TOL, section_max_camber

            camber = section_max_camber(self.endplate_coords)
            if camber > SYMMETRIC_CAMBER_TOL:
                raise ValueError(
                    f"the endplate's section must be SYMMETRIC: this one has "
                    f"{camber:.4g} of chord of camber against the library "
                    f"screen's {SYMMETRIC_CAMBER_TOL:g}, so the plate would "
                    f"carry a side force at zero toe. Choose a symmetric "
                    f"section (api.symmetric_section_names) or design one "
                    f"with symmetric=True")
            return
        a_l0 = float(getattr(self.endplate_polar, "alpha_L0", 0.0) or 0.0)
        if abs(a_l0) > self.ALPHA_L0_TOL_DEG:
            raise ValueError(
                f"the endplate's section must be SYMMETRIC: this one has a "
                f"zero-lift angle of {a_l0:.4g} deg, past the "
                f"{self.ALPHA_L0_TOL_DEG:g} deg this family allows (one per "
                f"cent of its own toe box), so the plate would carry a side "
                f"force at zero toe. Choose a symmetric section "
                f"(api.symmetric_section_names) or design one with "
                f"symmetric=True")

    @property
    def area_free(self) -> bool:
        """Is the reference area a design variable?"""
        return self.area_bounds_m2 is not None

    @property
    def constraint_labels(self) -> tuple:
        """Display names for the margins this problem actually returns.

        ``api.design_report`` prefers these over the static ProblemSpec
        labels (precedent: ``hydrofoil.HydrofoilWingletProblem`` under its
        draught cap). The three structural/geometric margins are always
        present — the beam, the plate and the reach are properties of the
        part, not budgets a caller may switch off — so a run with both drag
        budgets off still has something to constrain it.
        """
        out = []
        if self.CD_budget is not None:
            out.append("drag budget margin")
        if self.drag_budget_n is not None:
            out.append("drag force margin")
        # spelled exactly as the static ProblemSpec spells them, so a run
        # whose budgets are the published ones reports the labels it always
        # did — the property exists to make the OPTIONAL margins honest, not
        # to rename the mandatory three under everyone
        out += ["wing deflection margin", "endplate deflection margin",
                "endplate reach margin"]
        if self.downforce_min_n is not None:
            out.append("downforce floor margin")
        return tuple(out)

    @property
    def n_constraints(self) -> int:
        return len(self.constraint_labels)

    @property
    def free_names(self) -> tuple:
        """Labels of the rows this variant's own freedoms add, in row order."""
        return tuple(label for attr, label, _band in self.FREE_ROWS
                     if getattr(self, attr))

    @property
    def free_rows(self) -> list:
        """...and their bands, in the same order. One table, read twice."""
        return [tuple(getattr(self, band))
                for attr, _label, band in self.FREE_ROWS
                if getattr(self, attr)]

    @property
    def param_labels(self) -> tuple:
        """The design vector's labels, from the problem that lays it out.

        The registry used to state this family's labels as a static tuple
        beside a box the problem built — two sources for one order, which is
        the transposition defect ``api._mission_mode_labels`` documents
        (``b_m`` carrying a speed band, and pinning ``V_ms`` pinning a span).
        With the plate's freedoms adding rows in the middle of the vector
        there is no version of that split that stays right, so the problem
        owns both and ``api`` asks it.
        """
        size = (("S_m2",) if self.area_free else ()) + ("b_m",)
        return (self.BASE_LABELS + self.free_names + size
                + tuple(geometry.chord_labels(self.chord_order,
                                              self.chord_law)))

    @property
    def bounds(self) -> np.ndarray:
        family = np.array([
            self.TAPER_BOUNDS,
            self.TWIST_ROOT_BOUNDS_DEG,
            self.TWIST_TIP_BOUNDS_DEG,
            self.ALPHA_BOUNDS_DEG,
            self.ENDPLATE_H_M_BOUNDS,
            self.RIDE_HEIGHT_BOUNDS_M,
            self.ENDPLATE_CHORD_RATIO_BOUNDS,
            self.ENDPLATE_TC_BOUNDS,
            self.ENDPLATE_TOE_BOUNDS_DEG,
            # ...then whatever the plate's own freedoms add, INSIDE the family
            # block. It has to be here and nowhere else: carwing.span_from_x
            # and carwing.s_from_x address the size rows BACKWARDS from the
            # chord coefficients, so a row appended anywhere behind them would
            # silently re-address the span and the area with no test failing.
            *self.free_rows,
        ], dtype=float)
        # [family][size][chord] — the package's block order. The size block is
        # [S, b] when the area is free and [b] when it is not, and the area
        # goes AHEAD so the span keeps its slot either way (carwing.s_from_x).
        size = ([area_row(self.area_bounds_m2)] if self.area_free else [])
        box = np.vstack([family, *size, span_row(self.span_bounds_m)])
        return geometry.with_chord_bounds(box, self.chord_order,
                                          self.chord_max_frac,
                                          self.chord_law)

    @property
    def dim(self) -> int:
        return int(self.bounds.shape[0])

    @property
    def mount_spec(self) -> dict:
        return MOUNTS[self.mount]

    @property
    def section_spec(self) -> dict:
        return SECTIONS[self.section]

    def reach_m(self, ride: float) -> float:
        """Vertical gap the plate (or the pylons) must span [m]."""
        return float(ride) - float(self.deck_height_m)

    @property
    def chord_band_m(self) -> tuple[float, float]:
        """The plate's metre band as (lo, hi); unstated ends are 0 and inf."""
        lo = (0.0 if self.endplate_chord_min_m is None
              else float(self.endplate_chord_min_m))
        hi = (float("inf") if self.endplate_chord_max_m is None
              else float(self.endplate_chord_max_m))
        return lo, hi

    def clamp_chord_m(self, chord: float) -> float:
        """The chord this plate is allowed to be, given one it asked for.

        The ratio row's answer projected onto the metre band. A clamp rather
        than a refusal: every point of the design box stays flyable, so the
        band costs the search no evaluations and leaves no cliff for a GP to
        model — at the price of a plateau, which the breakdown makes visible
        (``endplate_chord_requested_m`` beside ``endplate_chord_m``).
        """
        lo, hi = self.chord_band_m
        return float(min(max(float(chord), lo), hi))


def evaluate_car_wing_endplate(
        x: np.ndarray,
        prob: CarWingEndplateProblem | None = None) -> dict:
    """Full evaluation with breakdown; in-contract failures never raise."""
    from .objective import _fail
    from .sizing import check_ar as _sizing_check_ar

    prob = prob or CarWingEndplateProblem()
    x = np.asarray(x, dtype=float)
    bnds = prob.bounds
    if x.shape != (bnds.shape[0],):
        return _fail(f"design vector shape {x.shape} != ({bnds.shape[0]},)",
                     g=[G_FAIL] * prob.n_constraints)
    if np.any(x < bnds[:, 0] - 1e-12) or np.any(x > bnds[:, 1] + 1e-12):
        return _fail("bounds violation", g=[G_FAIL] * prob.n_constraints)

    (taper, tw_root, tw_tip, alpha_deg, h_ep, ride,
     ep_chord_ratio, ep_tc, ep_toe_deg) = (float(v) for v in x[:9])
    # ...and the plate's OWN freedoms, where this variant carries them. Read
    # off `free_names` rather than by a hard-coded index, so the row order
    # here and the label order the registry publishes come from one table
    # (CarWingEndplateProblem.FREE_ROWS) and cannot drift apart. Absent, both
    # fall back to the stated fields, which is every published run.
    _free = dict(zip(prob.free_names, (float(v) for v in x[9:9 + len(
        prob.free_names)])))
    ep_cant = _free.get("endplate_cant_deg", prob.endplate_cant_deg)
    ep_blend = _free.get("endplate_blend_frac", prob.blend_frac)
    # The span row is the OVERALL WIDTH, plates included. What bounds a rear
    # wing is a regulation or the bodywork, and neither measures the wing and
    # then ignores what is bolted to its tips: a plate at cant 90 projects
    # nothing, but a BLENDED one leaves the wing plane tangentially and
    # reaches outboard, so blending would otherwise buy free width (0.27 m a
    # side at full blend — a 1.6 m wing 2.15 m wide inside a 2.0 m box). The
    # wing therefore pays for its plates' projection out of its own span, on
    # the same developed line the span-capped aircraft families use. At zero
    # blend the projection is exactly zero and the wing spans the whole box,
    # bit-for-bit the published problem.
    b_overall = span_from_x(x, prob.chord_order)
    semi = geometry.developed_semispan(0.5 * b_overall, h_ep,
                                       ep_cant,
                                       ep_blend, prob.blend_shape)
    if semi <= 0.0:
        return _fail(
            f"the endplates' outboard projection ({0.5 * b_overall - semi:.3g} "
            f"m a side) uses up the whole {b_overall:.3g} m width: there is no "
            f"wing left to carry them. Blend less or narrow the plates.",
            g=[G_FAIL] * prob.n_constraints)
    b = 2.0 * semi                     # the WING's span — beam, VLM, AR
    ep_frac = h_ep / semi              # the plate keeps its length in metres
    # the AREA this candidate flies, when it is one. Every S past here is
    # this candidate's: reading prob.S would reference the coefficients, and
    # charge the plate's drag against, a wing nobody flew.
    s_ref = (s_from_x(x, prob.chord_order) if prob.area_free
             else float(prob.S))
    # one chordwise panel is only honest over sizing.AR_LIMITS, and the aspect
    # ratio that matters is the WING's — b, not the overall width, so a
    # blended plate that has eaten into the span is checked on what is left
    if prob.area_free:
        why = _sizing_check_ar(b, s_ref)
        if why is not None:
            return _fail(why, g=[G_FAIL] * prob.n_constraints)
    # ...and the ASPECT-RATIO LIMIT THE USER SET. A different question
    # from the validity band, so it is asked whether or not the size is
    # a design variable: a limit that only applied to searched sizes
    # would go quiet exactly when the user typed the number it is about.
    if prob.ar_limits is not None:
        from .sizing import check_ar_limit as _check_ar_limit
        why_ar = _check_ar_limit(b, s_ref, prob.ar_limits)
        if why_ar is not None:
            return _fail(f"size: {why_ar}", g=[G_FAIL] * prob.n_constraints)

    try:
        wing = geometry.Wing(
            b=b, S=s_ref, taper=taper, twist_root_deg=tw_root,
            twist_tip_deg=tw_tip, tc=prob.tc,
            chord_limits=prob.chord_limits,
            chord_coeffs=geometry.chord_coeffs_from_x(x, prob.chord_order,
                                                      prob.chord_law))
    except ValueError as exc:
        return _fail(f"planform: {exc}", g=[G_FAIL] * prob.n_constraints)
    try:
        pol = (prob.section_polar if prob.section_polar is not None
               else prob.polar_family.at(prob.tc))
    except ValueError as exc:
        return _fail(f"polar family: {exc}", g=[G_FAIL] * prob.n_constraints)

    c_tip = float(wing.chord(np.array([b / 2.0]))[0])
    # the plate's chord is asked for as a RATIO of the tip chord and bounded
    # in METRES; what flies is the first projected onto the second. The
    # EFFECTIVE ratio is what goes to the solver, so the plate the VLM builds,
    # the drag build-up charges and the beam sizes is one part, not three.
    c_ep_req = ep_chord_ratio * c_tip
    c_ep = prob.clamp_chord_m(c_ep_req)
    if not c_ep > 0.0:
        return _fail(f"endplate chord of {c_ep:.4g} m is not a surface",
                     g=[G_FAIL] * prob.n_constraints)
    ratio_eff = c_ep / c_tip
    chord_clamped = abs(c_ep - c_ep_req) > 1e-12 * max(1.0, abs(c_ep_req))
    t_ep = ep_tc * c_ep

    try:
        model = VLM(
            wing, N=prob.N_vlm, winglet_h_frac=ep_frac,
            winglet_cant_deg=ep_cant,                  # 90 = towards the
            #                       track (published); less leans it outboard
            n_winglet=prob.n_endplate, winglet_blend_frac=ep_blend,
            winglet_blend_shape=prob.blend_shape,
            a=pol.a_lin, alpha_L0=pol.alpha_L0, V=prob.V,
            image=ImagePlane(z=ride, kind="ground"),     # the track
            # the plate carries its OWN symmetric section: alpha_L0 = 0, so a
            # plate at zero toe is a clean fence (module docstring)
            winglet_chord_scale=ratio_eff,
            winglet_toe_deg=ep_toe_deg,
            winglet_a=TWO_PI, winglet_alpha_L0=0.0,
        )
        res = model.solve(np.deg2rad(alpha_deg))
    except (ValueError, np.linalg.LinAlgError) as exc:
        return _fail(f"solver failure: {exc}", g=[G_FAIL] * prob.n_constraints)

    main = ~res.is_winglet
    alpha_eff_deg = np.rad2deg(res.alpha_eff[main])
    lo, hi = pol.alpha_valid
    if alpha_eff_deg.min() < lo or alpha_eff_deg.max() > hi:
        return _fail(
            "effective AoA outside polar validity (stall/extrapolation proxy)",
            g=[G_FAIL] * prob.n_constraints,
            alpha_eff_range=(float(alpha_eff_deg.min()),
                             float(alpha_eff_deg.max())),
        )
    # the plate has its own linear range, and it is checked against ITS
    # section, not the wing's cambered polar (see the class docstring)
    plate_alpha_deg = float(np.max(np.abs(np.rad2deg(
        res.alpha_eff[res.is_winglet])))) if res.is_winglet.any() else 0.0
    if plate_alpha_deg > prob.endplate_alpha_max_deg:
        return _fail(
            "endplate effective angle past its linear-range gate "
            f"({plate_alpha_deg:.2f} > {prob.endplate_alpha_max_deg:g} deg)",
            g=[G_FAIL] * prob.n_constraints, endplate_alpha_deg=plate_alpha_deg)

    spec = prob.mount_spec
    reach = prob.reach_m(ride)
    q = 0.5 * prob.rho * prob.V**2

    # --- the chords the plate is actually FLOWN with. A blended corner ramps
    # the chord from the wing's tip chord up to the plate's over the turn
    # (vlm.transition_ramp), so ``c_ep`` is the plate's chord only where the
    # turn has finished. Every reduced-order model below that asks for "the
    # plate's chord" therefore asks the flown geometry for the chord AT ITS
    # OWN STATION: the drag build-up gets the mean over the plate, the
    # junction charge the chord at the corner, the beam the chord at its
    # built-in end. With no blend all four are ``c_ep`` and nothing moves.
    star_dev = res.is_winglet & (res.y > 0.0)
    if star_dev.any():
        c_bar = float(np.sum(res.c[star_dev] * res.width[star_dev])
                      / np.sum(res.width[star_dev]))
        c_corner = float(res.c[star_dev][0])     # panels run junction -> tip
        c_far = float(res.c[star_dev][-1])
    else:
        c_bar = c_corner = c_far = c_ep
    # THICKNESS RATIO is what is held through the ramp, not thickness: a
    # blend lofts the section with the chord, exactly as the wing holds one
    # t/c across its own taper. Holding the thickness in metres instead would
    # leave the transition a wedge — at full blend a 0.31 m chord carrying the
    # 0.94 m plate's 0.11 m thickness, t/c = 0.36, outside the junction
    # correlation's own validity band and charged as if it were a strut.
    t_corner = ep_tc * c_corner

    # --- drag: wing sections from the polar, plate from its own build-up
    CDp = float(np.sum(pol.cd(alpha_eff_deg) * res.c[main] * res.width[main])
                / res.S)
    # THE PLATE'S OWN PROFILE DRAG. With an aerofoil on it, integrated panel
    # by panel over the plate at the angle each panel actually flies —
    # exactly the integral the wing's CDp above is, and for the same reason:
    # a section's cd is convex in alpha, so charging the mean angle is not
    # the mean charge. Without one, `parasite_cd` falls back to the
    # build-up and this whole block is skipped, bit-for-bit.
    ep_cd2d = None
    if prob.endplate_polar is not None and res.is_winglet.any():
        dev = res.is_winglet
        a_dev = np.rad2deg(res.alpha_eff[dev])
        lo_p, hi_p = prob.endplate_polar.alpha_valid
        if a_dev.min() < lo_p or a_dev.max() > hi_p:
            return _fail(
                f"the plate's own section is not measured at the angles it "
                f"flies ({a_dev.min():.2f} to {a_dev.max():.2f} deg, table "
                f"{lo_p:g} to {hi_p:g})",
                g=[G_FAIL] * prob.n_constraints,
                endplate_alpha_range=(float(a_dev.min()), float(a_dev.max())))
        area_dev = res.c[dev] * res.width[dev]
        tot = float(np.sum(area_dev))
        ep_cd2d = (float(np.sum(prob.endplate_polar.cd(a_dev) * area_dev)
                         / tot) if tot > 0.0 else 0.0)
    elif prob.endplate_polar is not None:
        # the lattice dropped the plate as too short to panel, and a section
        # cd with no panels to charge it on is zero — same convention the
        # build-up's h_ep <= 0 branch already uses
        ep_cd2d = 0.0
    ep_drag = parasite_cd(prob.section, c_bar, h_ep, ep_tc, s_ref,
                          prob.rho, prob.V, prob.mu, n_plates=2,
                          cd_section=ep_cd2d)
    # the mount's own parasite charge. No pylon in either layout — what the
    # inboard layout pays for is its two EXTRA sheets, standing the plate's
    # height at the wing's chord where they grip it (carwing.MOUNTS). The
    # designed plates themselves are already charged by ``parasite_cd``
    # above, so these are the additional ones and nothing is billed twice.
    c_grip = float(wing.chord(np.array([min(
        float(spec["station_frac"]) * 0.5 * b, 0.5 * b)]))[0])
    cd0_struts = _strut_cd0(spec["n_sheets"], h_ep, c_grip,
                            s_ref, prob.strut_cf)
    cd_junction = 0.0
    junc = {}
    if prob.junction_drag:
        # the wing/endplate corner exists in BOTH layouts; a centre mount
        # adds its pylon corners on top
        # AT THE CANT THAT WAS FLOWN. This read the literal 90.0 while every
        # other consumer of the angle read the field, so a LEANING plate was
        # charged a junction built on a fillet radius shrunk by cant/90:
        # measured at the box centre, cant 45 and blend 0.05, CD_junction
        # 4.70e-4 coded against 3.02e-4 honest — an overcharge of 1.69e-4 on
        # a CD of 0.0308 — and `blend_radius_m` was wrong at every blend
        # (0.0955 reported against 0.1432 at cant 60, blend 0.5). The same
        # class of defect the reach margin's own note above records, still
        # live here, and it would have silently held the cant at 90 for every
        # candidate of a SEARCHED cant.
        junc = junction.report(ep_tc, c_corner, s_ref, h_ep, ep_cant,
                               ep_blend, n_junctions=2,
                               blend_shape=prob.blend_shape)
        cd_junction = junc["CD_junction"]
        if spec["n_junctions"]:
            cd_junction += junction.junction_cd(
                prob.tc, float(wing.chord(np.array([0.0]))[0]), s_ref,
                radius=0.0, n_junctions=spec["n_junctions"])
    CD = res.CDi + CDp + ep_drag["CD"] + cd0_struts + cd_junction
    CZ = res.CL                  # model lift IS downforce (mirrored frame)
    if not np.isfinite(CD) or CD <= 0.0:
        return _fail("non-finite drag", g=[G_FAIL] * prob.n_constraints)

    # --- wing beam: same load, different support layout, on the SPAN this
    # candidate flies
    y_m, gam_m = res.y[main], res.Gamma[main]
    load = prob.rho * prob.V * np.abs(gam_m)          # [N/m]
    # ONE beam for both layouts, at the station where the sheets grip
    # (carwing.MOUNTS). SIGNED, because "inboard" is an interior support: the
    # moment changes sign there, and integrating a magnitude gives a
    # plausible, wrong deflection — which is also why carwing.bending_moment
    # (a two-valued end-or-centre model) cannot serve here any more.
    y_station = float(spec["station_frac"]) * 0.5 * b
    m_signed = carmount.beam_moment(y_m, load, y_station)
    moment = np.abs(m_signed)
    defl = (carmount.beam_deflection_index(y_m, m_signed, y_station)
            / prob.ei_nm2)

    # --- endplate: lateral load (its own toe side force + a yaw case)
    star = star_dev                                   # starboard plate
    y_toe = float(-prob.rho * prob.V * np.sum(res.Gamma[star] * res.lz[star]))
    # ...and the SAME force resolved onto the plate's own normal, which is
    # what bends it. Kutta-Joukowski on a bound segment l = (0, ly, lz) gives
    # dF = rho V Gamma (0, -lz, ly): a VERTICAL plate has ly = 0, so its whole
    # load is the side force above and the two numbers are identical — every
    # published run, bit-for-bit. A LEANING plate carries a vertical component
    # too, and reading only the y part understated its own bending by 1/sin
    # of the cant. Taken from the lattice's own ly/lz rather than from
    # sin(cant), because a BLENDED plate's cant varies along it and there is
    # no single angle to divide by.
    z_toe = float(prob.rho * prob.V * np.sum(res.Gamma[star] * res.ly[star]))
    n_toe = float(np.hypot(y_toe, z_toe))
    ar_ep = (h_ep / c_bar) if (h_ep > 0.0 and c_bar > 0.0) else 0.0
    # THE YAW CASE IS A VERTICAL-PLATE CORRELATION and stays one: it applies
    # the whole sideslip angle as incidence over the whole plate area, all of
    # which a leaning plate sees reduced, and this package has no measurement
    # that says by how much. Left uncanted rather than scaled by an invented
    # factor; the bias is CONSERVATIVE (the load is charged in full) and is
    # named in yaw_side_force's own docstring.
    y_yaw = yaw_side_force(q, c_bar * h_ep, ar_ep, prob.yaw_deg) \
        if h_ep > 0.0 else 0.0
    # worst case: the yaw case and the toe load in the same sense on one
    # plate (they cancel on the other) — stated, not hidden
    y_lat = n_toe + abs(y_yaw)
    # THE PLATE IS ALWAYS BUILT IN AT THE CAR. Both layouts take the load out
    # through the sheets (carwing.MOUNTS), so the plate is always a cantilever
    # from the DECK — its far end, at the full plate chord — and never the
    # hung-off-the-wing cantilever the pylon layout used to make it. That
    # branch is deleted rather than left unreachable: an arm of h_ep on a
    # plate that is carrying the wing would understate the bending it is
    # actually doing.
    # ...over the length the plate ACTUALLY spans to get there. The arm is
    # the developed arc from the deck up to the wing, which equals the
    # vertical gap only for a straight plate at cant 90 — every published run,
    # so this moves nothing published. A plate at cant 60 spans reach/sin(60)
    # = 1.155x the gap, and deflection goes as the cube of it: reading the gap
    # understated a leaning plate's bending by 54%. Optimistic, and in the
    # one direction a structural margin must never be.
    arm = geometry.winglet_arc_to_height(reach, h_ep, ep_cant,
                                         ep_blend, prob.blend_shape) \
        if h_ep > 0.0 else reach
    c_root = c_far
    t_root = ep_tc * c_root
    # WHICH SHAPE'S STIFFNESS. A designed plate is not one of the three
    # construction families, so it must not borrow one of their constants: a
    # section is chosen partly for the stiffness it buys, and handing every
    # designed aerofoil `SECTIONS["shaped"]`'s NACA number would make that
    # half of the choice invisible. With coordinates the k_I is the section's
    # own integral; with a polar but no coordinates it falls back to the
    # streamlined family's and SAYS SO in the breakdown, because a fallback
    # nobody can see is the same as a wrong number.
    k_source = prob.section
    if prob.endplate_coords is not None:
        k_source = "designed section (its own coordinates)"
    elif prob.endplate_polar is not None:
        k_source = f"{prob.section} (no coordinates given for the section)"
    if h_ep > 0.0:
        I_ep = (float(section_inertia_factor(prob.endplate_coords))
                * c_root * t_root ** 3
                if prob.endplate_coords is not None
                else bending_inertia(prob.section, c_root, t_root))
        ep_defl = lateral_deflection(y_lat, arm, prob.E_endplate_pa, I_ep)
        ep_stress = root_stress(y_lat, arm, t_root, I_ep)
    else:
        # no plate: nothing to bend. A wing with no plate has nothing to
        # carry it either, and that is caught by the reach constraint below
        # rather than silently blessed here.
        I_ep, ep_defl, ep_stress = 0.0, 0.0, 0.0

    # AT THE CANT THAT WAS FLOWN. This read 90.0 — a literal, not the field
    # — so a LEANING plate was credited with the reach of a vertical one: at
    # h_ep 0.30 m the tip sits 0.2598 m below the wing at cant 60, and the
    # margin was computed on 0.3000 m regardless, reporting +0.200 where the
    # honest number is +0.039. Every other consumer of the cant on this path
    # already reads the field (the developed semi-span above, the VLM below),
    # which is what made the omission invisible: the plate leaned in the
    # lattice and in the width accounting, and stood up straight in the one
    # constraint that says it can still reach the car.
    tip_z = geometry.winglet_tip_height(h_ep, ep_cant,
                                        ep_blend, prob.blend_shape)
    # THE PLATE MUST REACH THE DECK, always. It used to be required only
    # under the plate-mounted layout and returned a dormant +1.0 sentinel
    # otherwise; with both layouts plate-borne there is no "otherwise", and a
    # margin that is a constant on some runs is a margin a reader cannot
    # interpret. A wing whose plates do not get down to the car is not
    # attached to it.
    g_reach = float(tip_z / reach - 1.0)

    # the forces, which is what a budget on a MOVING reference area has to be
    # stated in — a coefficient allowance referenced to the area being
    # searched is not an allowance on anything the car can feel
    downforce_n = float(q * s_ref * CZ)
    drag_n = float(q * s_ref * CD)

    # --- the margins this problem declares, in constraint_labels order. Each
    # optional one is present iff its budget is set, so a run with a budget
    # off returns a SHORTER vector rather than a satisfied placeholder — the
    # difference between "the limit did not bind" and "there is no limit".
    g_defl = float(1.0 - defl / prob.deflection_limit_m)
    g_ep = float(1.0 - ep_defl / prob.endplate_deflection_limit_m)
    margins, g_drag, g_drag_n, g_down = [], None, None, None
    if prob.CD_budget is not None:
        g_drag = float(prob.CD_budget - CD)
        margins.append(g_drag)
    if prob.drag_budget_n is not None:
        g_drag_n = float(1.0 - drag_n / prob.drag_budget_n)
        margins.append(g_drag_n)
    margins += [g_defl, g_ep, g_reach]
    if prob.downforce_min_n is not None:
        g_down = float(downforce_n / prob.downforce_min_n - 1.0)
        margins.append(g_down)

    score = {"cz": float(CZ), "downforce": downforce_n,
             # MINUS the drag (carwing.CAR_OBJECTIVES says why the transform
             # is a negation, and states the PENALTY sentinel trap 'drag'
             # shares with 'laptime'). The epsilon-constraint half of the
             # trade: pair either with downforce_min_n.
             "cd": -float(CD), "drag": -float(drag_n),
             "efficiency": float(CZ / CD),
             # ADDED, not traded (carwing.CAR_OBJECTIVES carries the
             # measurement): the total load the wing puts into the car. On
             # THIS family that load is the one the plates carry, since both
             # MOUNTS layouts are plate-borne.
             "downforce_plus_drag": float(downforce_n) + float(drag_n),
             }[prob.objective]

    return {
        "feasible": True, "reason": "",
        "score": float(score), "CZ": float(CZ),
        "objective": prob.objective,
        "objective_label": ENDPLATE_OBJECTIVES[prob.objective],
        "g": margins,
        "g_drag": g_drag, "g_deflection": g_defl,
        "g_endplate": g_ep, "g_reach": g_reach,
        "g_drag_force": g_drag_n, "g_downforce": g_down,
        "constraint_labels": list(prob.constraint_labels),
        "CL_model": float(res.CL), "CD": float(CD),
        "CDi": float(res.CDi), "CDp": CDp,
        "CD_endplate": ep_drag["CD"],
        "CD_endplate_friction": ep_drag["CD_friction"],
        "CD_endplate_edges": ep_drag["CD_edges"],
        "cd0_struts": cd0_struts, "CD_junction": cd_junction,
        "efficiency": float(CZ / CD),
        "e": float(res.e), "AR": float(res.AR),
        "alpha_deg": alpha_deg,
        "ride_height_m": ride, "deck_height_m": prob.deck_height_m,
        # b_m is the WING's span (what the beam and the AR are), overall_width
        # what the span band bounds. They differ only by what the plates
        # project, and only a blended plate projects anything.
        "b_m": float(b), "S_m2": float(s_ref),
        "area_free": bool(prob.area_free),
        "overall_width_m": float(b_overall),
        "endplate_projection_m": float(0.5 * b_overall - semi),
        # the wing/plate chord STEP. 1.0 means the two surfaces meet; a
        # blended corner ramps the chord over the turn (vlm.transition_ramp)
        # and so meets the wing at 1.0 whatever this says, while a SHARP
        # corner joins them at this ratio and this is the size of the joint
        "endplate_chord_step": float(ratio_eff if ep_blend <= 0.0
                                     else 1.0),
        "endplate_chord_ratio_at_tip": float(ratio_eff),
        "reach_m": reach, "endplate_tip_z_m": float(tip_z),
        # a constant now (see the constraint list above); kept so a stored
        # run written before the pylon left this family still reads back
        "reach_required": True,
        "endplate_h_m": float(h_ep), "endplate_h_frac": float(ep_frac),
        "endplate_chord_m": float(c_ep),
        # what the plate is FLOWN with, station by station, once a blend has
        # ramped its chord: the mean the drag build-up used, the chord at the
        # corner the junction charge used, the chord at the beam's built-in
        # end. All three are endplate_chord_m when there is no blend.
        "endplate_chord_mean_m": float(c_bar),
        "endplate_chord_corner_m": float(c_corner),
        "endplate_chord_root_m": float(c_root),
        "endplate_t_corner_m": float(t_corner),
        "endplate_t_root_m": float(t_root),
        # the plate's chord under BOTH bands: what the ratio row asked for,
        # what the metre band allowed, and whether the two differ. A design
        # whose chord was clamped is a different part from the one the vector
        # names, so it says so rather than reporting the flown number alone.
        "endplate_chord_ratio": float(ratio_eff),
        "endplate_chord_ratio_requested": float(ep_chord_ratio),
        "endplate_chord_requested_m": float(c_ep_req),
        "endplate_chord_clamped": bool(chord_clamped),
        "endplate_chord_min_m": prob.endplate_chord_min_m,
        "endplate_chord_max_m": prob.endplate_chord_max_m,
        "endplate_tc": ep_tc, "endplate_t_m": float(t_ep),
        "endplate_toe_deg": ep_toe_deg,
        "endplate_alpha_deg": plate_alpha_deg,
        "endplate_AR": float(ar_ep),
        "endplate_section": prob.section,
        "endplate_section_label": prob.section_spec["label"],
        # the wing/plate CORNER, flat beside the plate's other numbers: this
        # family's tip device is the plate, so this is its blend, and the
        # rule every other family obeys is that a blend which changed the
        # solve appears in the read-out (a shape reported nowhere is a shape
        # nobody can check). ``junction`` below carries the charge it bought.
        "endplate_blend_frac": float(ep_blend),
        # THE CANT THE CANDIDATE ACTUALLY FLEW. It was never reported: the
        # angle was a constant, so the run's own record of it was the flag it
        # was launched with — and a searched cant has no flag to read.
        "endplate_cant_deg": float(ep_cant),
        "endplate_cant_searched": bool(prob.cant_free),
        "endplate_blend_searched": bool(prob.blend_free),
        # ...and WHAT THE PLATE IS MADE OF, when it is an aerofoil rather
        # than a construction family. `endplate_Re` was computed on every run
        # and reported on none, which is what made it impossible to check
        # what polar a plate would even have needed.
        "endplate_Re": float(ep_drag["Re"]),
        "endplate_cd_section": float(ep_drag["cd_section"]),
        "endplate_section_designed": bool(ep_drag["designed"]),
        "endplate_polar": (getattr(prob.endplate_polar, "name", None)
                           if prob.endplate_polar is not None else None),
        "endplate_inertia_source": k_source,
        "endplate_blend_shape": str(prob.blend_shape),
        "endplate_Swet_m2": ep_drag["Swet_m2"],
        "endplate_I_m4": float(I_ep),
        "endplate_side_load_N": float(y_lat),
        "endplate_side_load_toe_N": float(y_toe),
        # the same aerodynamic load resolved onto the plate's OWN normal —
        # equal to the side force above on a vertical plate, larger on a
        # leaning one, and the number its deflection and stress are computed
        # from. Reported so the two can never be read as one.
        "endplate_normal_load_toe_N": float(n_toe),
        "endplate_arm_m": float(arm),
        "endplate_side_load_yaw_N": float(y_yaw),
        "endplate_arm_m": float(arm),
        "endplate_deflection_m": float(ep_defl),
        "endplate_stress_Pa": float(ep_stress),
        "mount": prob.mount, "mount_label": spec["label"],
        "supports": spec["supports"],
        "frame": geometry.MIRRORED_FRAME,
        "M_max_Nm": float(np.max(moment)),
        "deflection_m": float(defl),
        "downforce_N": downforce_n, "drag_N": drag_n, "q_Pa": float(q),
        "Re_mac": float(prob.rho * prob.V * wing.mac / prob.mu),
        "junction": junc,
        "polar": getattr(pol, "name", "unknown"),
        "y": y_m, "load_Npm": load, "moment_Nm": moment,
        "wing": wing, "vlm": res,
    }


def fg_car_wing_endplate(x: np.ndarray,
                         prob: CarWingEndplateProblem | None = None
                         ) -> tuple[float, list]:
    """Constrained-harness callable: ``(score, margins)``.

    The failure vector's WIDTH follows the problem's own declaration, not the
    published four: a run with the coefficient budget off and a downforce
    floor on returns four margins where the default returns four different
    ones, and a penalty at the wrong width is a shape error inside the
    optimiser rather than a bad score (``carwing.fg_car_wing``, same rule).
    """
    prob = prob or CarWingEndplateProblem()
    out = evaluate_car_wing_endplate(x, prob)
    if not out["feasible"]:
        return PENALTY, [G_FAIL] * prob.n_constraints
    return float(out["score"]), [float(g) for g in out["g"]]


def compare_sections(x: np.ndarray,
                     prob: CarWingEndplateProblem | None = None) -> dict:
    """Same design, every endplate section family — the direct comparison.

    The aerodynamics of the lifting system are identical (the section family
    changes neither the panel geometry nor the circulation); what moves is
    the plate's parasite drag and its bending stiffness.
    """
    prob = prob or CarWingEndplateProblem()
    return {s: evaluate_car_wing_endplate(x, replace(prob, section=s))
            for s in SECTIONS}


def compare_mounts(x: np.ndarray,
                   prob: CarWingEndplateProblem | None = None) -> dict:
    """Same design, both mount layouts — with the endplate's own structure.

    Unlike carwing.compare_mounts this also moves the ENDPLATE's beam: bolted
    to the car the plate is a cantilever from the deck; hung off the wing it
    is a cantilever from the wing. Same plate, different arm.
    """
    prob = prob or CarWingEndplateProblem()
    return {m: evaluate_car_wing_endplate(x, replace(prob, mount=m))
            for m in MOUNTS}
