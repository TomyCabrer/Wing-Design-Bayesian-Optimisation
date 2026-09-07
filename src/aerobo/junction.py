"""Junction (interference) drag of an intersecting-surface corner.

Why this module exists
----------------------
The Trefftz plane sees the wake trace and nothing else, so a lifting-surface
method values a wing/winglet junction ONLY through the (y, z) line it draws.
By that measure a sharp corner and a smooth blend of the same arc length are
just two different lines, and the reason real aircraft blend the corner —
the interference drag of two surfaces meeting at an angle, and the
separation risk in the corner — is invisible.

That drag is a viscous/pressure effect at the corner. It is modelled here as
a reduced-order ADD-ON, kept out of the solvers and out of every default
path, so no existing result moves.

The literature part
-------------------
Hoerner (*Fluid-Dynamic Drag*, 1965, ch. 8) correlates the interference drag
of an unfilleted junction between two streamlined surfaces meeting at a
right angle as a DRAG AREA scaled on the junction member's thickness:

    D_int / q = t^2 * (17 (t/c)^2 - 0.05)                            (1)

with ``t`` the maximum thickness and ``c`` the chord at the junction. The
correlation is quoted for roughly 0.05 <= t/c <= 0.20; outside that band it
is extrapolation (and below t/c ~ 0.054 expression (1) turns negative, which
is where the correlation stops meaning anything — the floor is applied
explicitly rather than letting a negative drag through).

The modelling-choice part (LABELLED, not literature)
----------------------------------------------------
Hoerner reports that a fillet reduces junction drag substantially, and the
standard design rule of thumb is that a fillet of radius ~10% of the local
chord removes most of it. There is no closed-form radius law in the source,
so the credit here is a straight-line decay from 1 (sharp) to
``FILLET_CREDIT_FLOOR`` at ``FILLET_FULL_R_OVER_C``, both named constants:

    credit(R) = max(floor, 1 - (1 - floor) * (R/c) / (R/c)_full)     (2)

This is a CALIBRATED SHAPE, not a measurement. It is monotone, bounded, and
reduces to the unfilleted correlation at R = 0, which is what a design study
needs; it should not be read as a prediction of a particular fillet's drag.

Consequently the junction term is OFF by default everywhere. Where it is
switched on, the breakdown reports it as its own line item so it can never
be mistaken for solver output.
"""

from __future__ import annotations

import numpy as np

#: validity band of the Hoerner correlation (thickness ratio at the corner)
TC_VALID = (0.05, 0.20)

#: R/c at which the fillet credit reaches its floor (design rule of thumb:
#: a fillet of ~10% chord removes most of the junction drag). MODEL CHOICE.
FILLET_FULL_R_OVER_C = 0.10

#: residual fraction of the unfilleted drag left by a large fillet. Not zero:
#: a blended corner still carries some interference. MODEL CHOICE.
FILLET_CREDIT_FLOOR = 0.15


def hoerner_drag_area(tc: float, chord: float) -> float:
    """Unfilleted right-angle junction drag AREA D/q [m^2] — Hoerner eq. (1).

    Returns 0 (never a negative drag) where the correlation goes negative at
    very small thickness ratios.
    """
    tc = float(tc)
    if not (0.0 < tc < 1.0):
        raise ValueError(f"thickness ratio must be in (0, 1), got {tc}")
    if not (float(chord) > 0.0):
        raise ValueError(f"chord must be > 0, got {chord}")
    t = tc * float(chord)
    return float(max(0.0, t * t * (17.0 * tc * tc - 0.05)))


def fillet_credit(radius: float, chord: float) -> float:
    """Multiplier in [FILLET_CREDIT_FLOOR, 1] for a fillet of ``radius``.

    MODEL CHOICE (see module docstring) — a bounded linear decay, not a
    measured law.
    """
    if float(chord) <= 0.0:
        raise ValueError(f"chord must be > 0, got {chord}")
    r_c = max(0.0, float(radius)) / float(chord)
    frac = min(1.0, r_c / FILLET_FULL_R_OVER_C)
    return float(1.0 - (1.0 - FILLET_CREDIT_FLOOR) * frac)


def junction_cd(tc: float, chord: float, s_ref: float,
                radius: float = 0.0, n_junctions: int = 2) -> float:
    """Interference-drag coefficient on ``s_ref`` for ``n_junctions`` corners.

    ``radius`` is the blend/fillet radius at the corner [m]; 0 is the sharp
    junction. A wing with a winglet on each side has two junctions.
    """
    if int(n_junctions) < 0:
        raise ValueError("n_junctions must be >= 0")
    if not (float(s_ref) > 0.0):
        raise ValueError(f"reference area must be > 0, got {s_ref}")
    area = hoerner_drag_area(tc, chord) * fillet_credit(radius, chord)
    return float(int(n_junctions) * area / float(s_ref))


def blend_radius(h: float, cant_deg: float, blend_frac: float,
                 wing_arc: float = 0.0) -> float:
    """MEAN radius of curvature [m] of the corner geometry.span_path draws.

    ``blend_frac * h`` of arc on the winglet side plus ``wing_arc`` metres on
    the wing side, turning through the cant angle, i.e.
    R = (beta h + s_in) / |cant|. Zero blend (or zero cant) is a sharp
    corner, reported as radius 0.

    ``wing_arc`` is what lifts this off the winglet's own scale: confined to
    the device, R <= h/|cant| is a fraction of a tip chord and the fillet
    credit below saturates almost immediately, which is exactly why a blend
    that only turns on the winglet reads as a corner.

    Shape-INDEPENDENT on purpose. Total turn / total turning length is the
    same for every turn law that reaches the same cant over the same arc
    (geometry.BLEND_SHAPES), and it is the only radius this reduced-order
    correlation can honestly be a function of: the fillet credit below is a
    calibrated shape, not a measurement, so letting it separate a circular
    fillet from a curvature-continuous one would be inventing a distinction
    the model cannot see. The two shapes differ in the WAKE they draw (which
    the Trefftz plane does see) and in their minimum radius, reported
    alongside by :func:`blend_radius_min`.
    """
    phi = abs(np.deg2rad(float(cant_deg)))
    s_tot = float(blend_frac) * float(h) + float(wing_arc)
    if s_tot <= 0.0 or phi == 0.0:
        return 0.0
    return float(s_tot / phi)


def blend_radius_min(h: float, cant_deg: float, blend_frac: float,
                     blend_shape: str = "arc", wing_arc: float = 0.0) -> float:
    """Tightest radius of curvature [m] anywhere in the transition.

    Equal to :func:`blend_radius` for the constant-curvature ``arc``. A
    crease-free law spends its ends turning less, so it must turn harder in
    the middle: the ``smooth`` smoothstep peaks at 1.5x the mean curvature
    (minimum radius 2/3 of the fillet's) and the ``spiral`` clothoid at 4/3
    (minimum radius 3/4). Reported (never charged) so a design study can see
    what a curvature-continuous junction costs in local tightness — and why
    the gentler of the two G2 laws is the clothoid.
    """
    from . import geometry
    r_mean = blend_radius(h, cant_deg, blend_frac, wing_arc)
    if r_mean <= 0.0:
        return 0.0
    k = np.abs(geometry.winglet_curvature(
        np.linspace(-float(wing_arc), blend_frac * float(h), 129), h,
        cant_deg, blend_frac, blend_shape, wing_arc))
    k_max = float(np.max(k))
    return float(1.0 / k_max) if k_max > 0.0 else r_mean


def report(tc: float, chord: float, s_ref: float, h: float, cant_deg: float,
           blend_frac: float, n_junctions: int = 2,
           blend_shape: str = "arc", wing_arc: float = 0.0) -> dict:
    """Full junction line item for a breakdown: radius, credit and CD."""
    r = blend_radius(h, cant_deg, blend_frac, wing_arc)
    sharp = junction_cd(tc, chord, s_ref, radius=0.0,
                        n_junctions=n_junctions)
    cd = junction_cd(tc, chord, s_ref, radius=r, n_junctions=n_junctions)
    return {
        "CD_junction": cd,
        "CD_junction_sharp": sharp,
        "blend_radius_m": r,
        "blend_radius_min_m": blend_radius_min(h, cant_deg, blend_frac,
                                               blend_shape, wing_arc),
        "blend_shape": str(blend_shape),
        "blend_wing_arc_m": float(wing_arc),
        "blend_r_over_c": float(r / chord) if chord > 0 else 0.0,
        "fillet_credit": fillet_credit(r, chord),
        "n_junctions": int(n_junctions),
        "tc_in_correlation_band": bool(TC_VALID[0] <= tc <= TC_VALID[1]),
    }
