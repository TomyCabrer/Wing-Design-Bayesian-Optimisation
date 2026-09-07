"""Raymer wetted-area parasite-drag build-up (single wing, Tier A).

numpy port of ~/Desktop/GDP/tandem_wing/core/drag_buildup.m, reduced to the
single-wing case with an optional fuselage extension point.

    CD0 = (1/Sref) * sum_c [ Cf_c * FF_c * Q_c * Swet_c ] * (1 + misc_frac)

Cf: turbulent flat-plate skin friction with a roughness Reynolds cut-off
(Raymer eq. 12.27 / 12.28), optional laminar (Blasius) blend.

DOUBLE-COUNT GUARD: when the objective integrates section profile drag from
an airfoil polar (polar.py), the wing's own friction+form drag is already in
CDp — do NOT add wing_cd0 on top. wing_cd0 exists for (a) polar-free
estimates and (b) cross-checks; fuselage_cd0 is the non-wing extension point
(mirrors CD0_nonwing in the MATLAB reference).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

K_ROUGH_SMOOTH_PAINT = 6.34e-7  # m, Raymer Table 12.4

#: OVERALL BODY LENGTH as a multiple of the wing-to-tail arm.
#:
#: A fuselage runs forward of the wing as well as aft of it, so the length
#: its drag and its fineness ratio are built on is NOT the arm — and
#: ``fuselage_cd0``'s form factor ``1 + 60/fr^3 + fr/400`` is meaningless
#: for half a body. This is the package's own long-standing proxy for the
#: whole: it was a bare ``1.6`` inside ``flightmodel.build_flight_model``,
#: used to give a rebuilt design an inertia length. Naming it here — beside
#: the body model that consumes it — is what stops a second caller writing
#: ``1.6`` again and the two drifting, which is exactly how the FIN came to
#: have two sizes (see aerobo.fin).
#:
#: It is a proxy and not a measurement: a design that states its own body
#: length should be believed instead.
BODY_LENGTH_FRAC = 1.6


def body_length_for_arm(l_t: float, frac: float = BODY_LENGTH_FRAC) -> float:
    """Overall body length [m] implied by a wing-to-tail arm.

    ``abs`` because a CANARD's arm is negative — the surface is ahead of the
    wing — and a body length is a length either way.
    """
    return float(frac) * abs(float(l_t))


#: the fineness ratio ``fuselage_cd0``'s own form factor is minimised at,
#: ``(180 * 400) ** 0.25`` exactly.
#:
#: ``ff = 1 + 60/fr^3 + fr/400`` falls as the body slims (the base-pressure
#: term) and rises as it lengthens (the skin term); the two cross at 16.38,
#: where ``ff = 1.0546`` against 1.2928 at a stubby 6.
#:
#: IT IS NOT A DRAG OPTIMUM AND MUST NOT BE SOLD AS ONE. Wetted area falls
#: with diameter at fixed length, so the drag of a body alone is minimised
#: by making it infinitely thin — nothing in this package sizes a fuselage
#: from what it has to carry, so no diameter can be derived from the
#: physics. What this constant is for is a DEFAULT: the shape at which the
#: form penalty of the law the design is actually charged with has stopped
#: mattering, so a user who has not thought about their fuselage is not
#: paying a shape penalty they did not choose.
SLENDER_FINENESS = (180.0 * 400.0) ** 0.25


def diameter_for_fineness(l_t: float, fr: float = SLENDER_FINENESS,
                          frac: float = BODY_LENGTH_FRAC) -> float:
    """Body diameter [m] giving fineness ratio ``fr`` at a wing-to-tail arm.

    The inverse of :func:`body_length_for_arm` over ``fr``, so a shell can
    offer a diameter that is a statement about the SHAPE rather than a
    number somebody guessed. ``fuselage_cd0`` refuses ``fr <= 1``; this
    cannot produce one, because ``fr`` is the input.
    """
    if not float(fr) > 0.0:
        raise ValueError(f"fineness ratio must be > 0 (got {fr!r})")
    return body_length_for_arm(l_t, frac) / float(fr)


def skin_friction_cf(
    Re: float,
    M: float = 0.0,
    lref: float = 1.0,
    k_rough: float = K_ROUGH_SMOOTH_PAINT,
    laminar_frac: float = 0.0,
) -> float:
    """Flat-plate Cf: turbulent (Raymer 12.27) with roughness cut-off (12.28),
    blended with Blasius laminar by ``laminar_frac``."""
    Re = max(Re, 1e3)
    Re_cut = 38.21 * (lref / k_rough) ** 1.053
    Re_eff = min(Re, Re_cut)
    cf_turb = 0.455 / ((np.log10(Re_eff)) ** 2.58 * (1 + 0.144 * M**2) ** 0.65)
    cf_lam = 1.328 / np.sqrt(Re_eff)
    return float((1 - laminar_frac) * cf_turb + laminar_frac * cf_lam)


def wing_form_factor(tc: float, xc_m: float = 0.30, sweep_deg: float = 0.0,
                     M: float = 0.0) -> float:
    """Raymer wing/tail form factor (Raymer 2018, eq. 12.30):

        FF = [1 + (0.6/(x/c)_m)(t/c) + 100(t/c)^4] * [1.34 M^0.18 (cos Lam_m)^0.28]

    Two documented departures from the raw correlation:
    - the 1.34*M^0.18 compressibility factor is a cruise-Mach correlation and
      drops below 1 for M < ~0.2, unphysical for a form factor; it is floored
      at 1 so the incompressible FF reduces to the thickness term;
    - the sweep term (cos Lam_m)^0.28 sits OUTSIDE that floor so sweep acts at
      all Mach numbers (the design's quarter-chord sweep is used as a proxy
      for the max-thickness-line sweep Lam_m — reduced-order approximation).
      At sweep = 0 the term is exactly 1: pre-existing behaviour unchanged.
    """
    thickness = 1 + 0.6 / xc_m * tc + 100 * tc**4
    mach_fac = max(1.34 * max(M, 1e-6) ** 0.18, 1.0)
    sweep_fac = np.cos(np.deg2rad(sweep_deg)) ** 0.28
    return float(thickness * mach_fac * sweep_fac)


@dataclass
class ComponentCD0:
    name: str
    Swet: float
    Cf: float
    FF: float
    Q: float
    CD0: float  # referenced to Sref, incl. misc factor


def wing_cd0(
    Sref: float,
    Sexp: float,
    tc: float,
    mac: float,
    rho: float,
    V: float,
    mu: float,
    xc_m: float = 0.30,
    sweep_deg: float = 0.0,
    Q: float = 1.0,
    M: float = 0.0,
    laminar_frac: float = 0.0,
    misc_frac: float = 0.05,
) -> ComponentCD0:
    """Wing parasite CD0 (see module docstring for when NOT to add this)."""
    Swet = Sexp * (1.977 + 0.52 * tc)
    Re = rho * V * mac / mu
    cf = skin_friction_cf(Re, M=M, lref=mac, laminar_frac=laminar_frac)
    ff = wing_form_factor(tc, xc_m=xc_m, sweep_deg=sweep_deg, M=M)
    cd0 = cf * ff * Q * Swet / Sref * (1 + misc_frac)
    return ComponentCD0("wing", Swet, cf, ff, Q, float(cd0))


def fuselage_cd0(
    Sref: float,
    L: float,
    D: float,
    rho: float,
    V: float,
    mu: float,
    Q: float = 1.0,
    M: float = 0.0,
    misc_frac: float = 0.05,
) -> ComponentCD0:
    """Fuselage parasite CD0 — Tier A extension point (not used by default)."""
    fr = L / D
    Swet = np.pi * D * L * (1 - 2 / fr) ** (2 / 3) * (1 + 1 / fr**2)
    Re = rho * V * L / mu
    cf = skin_friction_cf(Re, M=M, lref=L)
    ff = 1 + 60 / fr**3 + fr / 400
    cd0 = cf * ff * Q * Swet / Sref * (1 + misc_frac)
    return ComponentCD0("fuselage", float(Swet), cf, float(ff), Q, float(cd0))
