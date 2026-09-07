"""Phase 4: H-tail as a small rear wing — coupled trim + static margin.

Model
-----
Main wing (Tier A planform: b = 10 m, S = 10 m^2, taper + linear twist from
the design vector) with its quarter-chord line ("wing AC") at x = 0; a
RECTANGULAR horizontal tail of area S_t at moment arm l_t AFT of the wing
AC (x is downstream-positive, so the tail sits at x = l_t > 0). The tail
aspect ratio is FIXED at AR_t = 4 (b_t = sqrt(AR_t * S_t)): tail span is
not a design freedom at this phase — AR_t ~ 3-5 is the conventional H-tail
range (Raymer, *Aircraft Design: A Conceptual Approach*, tail-geometry
guidance) and freezing it keeps the design vector at d = 5.

The pair is solved with tandem.py's ONE coupled (N_w + N_t)^2 linear
system (Surface / build_coupling / solve_tandem, REUSED — not
reimplemented). The wing-on-tail downwash therefore comes OUT of the
coupled solve: there is NO separate epsilon model. This is deliberately
better than the classic handbook chain

    eps = 2 CL_w / (pi AR_w),   d eps/d alpha = 2 CL_alpha_w / (pi AR_w)

(e.g. Etkin & Reid, *Dynamics of Flight*, stability build-up): the
panel-horseshoe kernel resolves the spanwise-VARYING downwash at the
tail's actual (x, z), the finite-arm growth toward the doubled Trefftz
value (w_influence's (1 + dx/r) factor — at l_t = 3-8 m the tail sees
1.5-1.9x the at-the-line downwash, not the fixed 2x the handbook assumes),
and the tail's upwash back-reaction on the wing.

Vertical placement (regularisation, MEASURED — a DEFAULT, not a floor; a
stated or searched height is free, see height_row): the tail is raised
dz = DZ_FRAC * b = 0.05 b = 0.5 m above the wing plane. This is the same
kernel-regularisation device as tandem.py (TandemProblem uses dz = 0.1 b):
at dz = 0 the tail's collocation points lie IN the wing's trailing-vortex
sheet where the leg kernel dy/(dy^2 + dz^2) is grid-singular. Measured on
the mid-box config [taper 0.6, tw 1/-2, S_t 1.5, l_t 5]: at dz = 0 the
mean tail downwash is non-convergent in the wing grid (-2.46 / -3.17 /
-2.36 deg at N_w = 40/60/120, per-station excursions past -13 deg); at
dz = 0.05 b it is grid-converged (-1.9826 / -1.9844 / -1.9855 deg, a
0.15 % drift 40 -> 120). 0.05 b suffices here (vs tandem's 0.1 b) because
only the TAIL's collocation points must clear the sheet and dz = 0.5 m
exceeds the wing's mid-span panel pitch (~0.26 m at N_w = 60).

Pitching moment about the CG — sign convention DERIVED, not assumed
-------------------------------------------------------------------
Axes: x downstream (+), z up (+), y starboard; (x_hat, y_hat, z_hat)
right-handed (x_hat x y_hat = z_hat). The nose points UPSTREAM (-x).
Pitch is the moment about y_hat; check its sense: a small rotation
d_theta about +y_hat moves the nose at -l x_hat by
d_theta (y_hat x (-l x_hat)) = +l d_theta z_hat — the nose RISES, so
+y_hat moment = NOSE-UP positive (the standard body-axis convention).

Each surface's lift L_i z_hat acts at its quarter-chord line x_i
(lifting-line: the bound vortex IS the quarter-chord = section AC;
x_w = 0, x_t = l_t). Moment about the CG at x_cg:

    M_y = [(x_i - x_cg) x_hat] x [L_i z_hat] . y_hat
        = (x_i - x_cg) L_i (x_hat x z_hat) . y_hat = -(x_i - x_cg) L_i

(x_hat x z_hat = -y_hat). Lift AFT of the CG pitches nose-DOWN — the
expected sign. Non-dimensionalised on the WING's nominal area and mac
(standard aircraft references):

    Cm_cg = [ (x_cg - x_w) CL_w S_w + (x_cg - l_t) CL_t S_t + M_ac ]
            / (Sref mac)

with CL_i S_i = L_i / q from the coupled solve (numerical trapezoid areas,
tandem.py convention).

The last term is the SECTIONS' OWN COUPLE, M_ac = sum_i cm_ac,i S_i c_i
(:func:`section_moment`), and it is not a lift moment: a cambered section
pitches nose-down at EVERY lift, including zero, so no arm makes it go
away. It was omitted here until 2026-08-01 as a "reduced-order choice",
on the argument that a constant offset in i_t does not touch the
L/D-vs-SM trade. That argument is right about the trade and wrong about
the aircraft: the constant it drops is the term that decides the SIGN of
what the stabiliser carries. For the published wing (NACA 2412,
cm_ac = -0.054) it is worth Cm = -0.060 on (Sref, mac) — HALF the trim
load — and the tail's own section adds a tenth of that again.

With it carried, the load the tail flies is

    CL_t = [ CL_target Sref (x_cg - x_w) + M_ac ] / (S_t (l_t - x_w)),

zero at x_cg = 0.123 m aft of the wing AC and NEGATIVE — the classic
DOWNLOAD — at every CG forward of that, which is every conventional
loading. At the calibrated x_cg = 0.05 m (30 % MAC) the published tail
carries CL_t = -0.038 and trims at i_t = -4.93 deg. A canard, sitting
upstream with l_t < 0, correctly comes out LIFTING (+0.27).

M_ac is constant in (alpha, i_t), so it shifts r0 of the affine map below
and nothing else: the Jacobian, the neutral point and the static margin
are alpha-derivatives and do not see it. A symmetric section returns
exactly 0.0 and reproduces the pre-2026-08-01 numbers.

2-D trim by elimination (the phase core)
----------------------------------------
Find (alpha, i_t) with CL_total = CL_target AND Cm_cg = 0. The coupled
LLT is one linear system in which the angles enter the RHS only, so BOTH
CL_total and Cm_cg are exactly AFFINE in (alpha, i_t):

    [CL; Cm](alpha, i_t) = J [alpha; i_t] + r0.

Three coupled solves — (0,0), (d,0), (0,d) — give r0 and the exact 2x2
Jacobian J by finite difference (EXACT for a linear map, not an
approximation), and trim is the closed-form 2x2 solve. This generalises
the 1-D tandem trim trick ("CL affine in alpha -> 2 solves",
tandem.evaluate_tandem / hydrofoil.solve_hydrofoil_trim) to two trim
unknowns. The affinity claim is test-gated: a 4th solve at the returned
(alpha, i_t) reproduces CL/Cm to 1e-10 (tests/test_tail.py).

Static margin (coupled derivatives)
-----------------------------------
With a_i = d(CL_i S_i)/d alpha the COUPLED lift-curve slopes (finite
difference of two coupled solves at fixed i_t — again exact for a linear
system; the tail's a_t automatically contains the d eps/d alpha downwash
lag and the wing's a_w the tail's upwash back-reaction),

    dCm/d alpha = [x_cg (a_w + a_t) - (x_w a_w + x_t a_t)] / (Sref mac)
                = -(a_w + a_t) (x_np - x_cg) / (Sref mac),

    x_np = (x_w a_w + x_t a_t) / (a_w + a_t),    SM = (x_np - x_cg) / mac,

i.e. Cm_alpha = -CL_alpha * SM: stable (Cm_alpha < 0) iff x_cg < x_np
(Etkin & Reid; Raymer Ch. 16). Constraint g = (SM - SM_min) / 1.0 with
SM_min = 0.08 (the /1.0 is the explicit O(1) normalisation scale — SM is
already in mac units, hydrofoil/aircraft g-scale precedent).

Calibration duty: x_cg (RE-MEASURED 2026-08-01)
----------------------------------------------
x_cg = 0.05 m aft of the wing AC — a CG at 30 % MAC, i.e. a real cruise
loading (X_CG_BY_TYPE), which ALSO discharges the calibration duty: the
SM = SM_min boundary has to cross the design-box INTERIOR or the
stability constraint is decoration. Measured at the 8 (taper, S_t, l_t)
box corners (twists 0 — they do not enter the derivatives): SM spans
+0.0074 (S_t 0.5, l_t 3.0) to +1.1937 (taper 1.0, S_t 3.0, l_t 8.0) —
both sides of SM_min = 0.08, gated in tests/test_tail.py
(test_sm_boundary_crosses_box). Small-tail short-arm corners are
UNSTABLE-side (g < 0), big-tail long-arm corners deeply stable: the
constraint is genuinely active inside the box. It is the most forward CG
for which that holds — at x_cg = 0.00 every corner is stable (min SM
+0.153) and two hit the penalty contract.

The pre-2026-08-01 value was +0.25 m (49.5 % MAC): calibrated for the
same duty, but aft of any real aft-CG limit, and the reason the
stabiliser used to come out lifting. The SM spread at that CG was
-0.1671 .. +0.9937 — the same interval shifted by the CG move, since
x_np does not depend on the CG at all. All 27 points of the
3x3x3 (S_t, l_t, taper) trim grid trim with machine-zero residuals and
|i_t| <= 9.3 deg (limit 15); the S_t -> 0 probe reproduces the Tier A
wing-only trim LoD to 2.0e-4 relative (gate tolerance 5e-3).

Drag bookkeeping (no double count)
----------------------------------
CDi: tandem.py's convention verbatim — per-surface Fourier self terms
area-weighted onto Sref ONCE plus the two mutual (Munk pair) terms from
mutual_cdi, all inside TandemResult.CDi_total with Sref = S_wing.
Profile: the section polar integrated over wing strips AND tail strips at
their own alpha_eff(y). The tail REUSES the wing polar by default —
documented approximation with the winglet precedent
(objective.evaluate_winglet) — unless a section was chosen for it
(``polar_tail``), in which case its slope, zero-lift angle, stall proxy and
profile drag are all its own.
cd0_extra is the non-modelled-component hook (e.g. vtail_cd0 below).
LoD = CL_target / CD_total (CL_total = CL_target at trim by construction).

Failure contract
----------------
objective.py / hydrofoil.py contract: evaluate_tail never raises for
in-contract failures and fg_tail returns exactly (PENALTY, G_FAIL) =
(-100.0, -1.0) — finite, never NaN — on solver failure or untrimmable
designs (|i_t| > 15 deg, alpha outside bracket, singular trim Jacobian,
polar-validity stall proxy). Solvable-but-SM-violating designs return
their TRUE LoD with signed g < 0 (constrained BO must see the magnitude).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from . import drag as _drag
from . import fin as _fin
from . import geometry
from .drag import skin_friction_cf, wing_form_factor
from .llt import cosine_stations
from . import fin as _finmod
from .objective import MU_SL, PENALTY, RHO_SL, _fail
from . import polar as _polar
from .polar import default_polar
from .tandem import Surface, TandemResult, VortexSystem, build_coupling, solve_tandem

G_FAIL = -1.0        # finite infeasible margin reported on solver failure

TAIL_AR = 4.0        # fixed H-tail aspect ratio (module docstring)
DZ_FRAC = 0.05       # tail height above the wing plane / wing span (MEASURED
#                      regularisation choice — module docstring)

#: ...and the height below which the second surface's downwash stops being
#: GRID-CONVERGED, which is a different number and was for a long time
#: confused with the one above. :data:`DZ_FRAC` is where the tail heights
#: were measured; the grid study behind it (module docstring) was run at
#: dz = 0 and at 0.05 b and its conclusion — "not converged at zero" — was
#: generalised into a floor 25-50x higher than the evidence supports.
#:
#: Re-measured (session 36) on the same mid-box configuration, sweeping the
#: grid at each height, as the L/D spread over three grids:
#:
#:   coupled LLT (tail.py, N_w = 40/60/120)
#:       0.05 b 0.033 %   0.01 b 0.030 %   0.002 b 0.018 %   0 b 1.106 %
#:   nonplanar VLM (wingtail.py, (N_vlm, N_t) = (30,18)/(40,24)/(60,36))
#:       0.05 b 0.098 %   0.01 b 0.273 %   0.002 b 1.915 %   0 b 6.622 %
#:
#: The LLT holds to 0.002 b and only loses it AT zero; the VLM, which
#: resolves the sheet with real panels, starts drifting an order of
#: magnitude earlier. 0.01 b is where both are still inside 0.3 %, so it is
#: the honest line — five times below the floor that used to be enforced.
#:
#: The STATIC MARGIN is the more sensitive quantity, and the one that
#: matters, because it is what the gate is on: same sweep, the spread in g
#: is 0.0002 at 0.05 b, 0.0004 at 0.01 b, 0.0087 at 0.002 b and 0.0353 at
#: zero. That last is the same size as the margins themselves near the
#: SM_min boundary, so at z_t = 0 the FEASIBILITY VERDICT can flip with the
#: panel count — which is the concrete thing the shell's caution is warning
#: about, and it puts 0.01 b (0.0004, converged) exactly where it belongs.
#:
#: It is a CAUTION, not a limit, and nothing in this package checks it: a
#: surface in the wing plane is the ordinary layout and it is built and
#: flown as asked (see :func:`height_row`). What it is for is the shell,
#: which says that below it the induced downwash — and so the neutral point
#: and the trimmed L/D — depend on the panel count.
DZ_GRID_FRAC = 0.01

#: the aeroplane every band below was CALIBRATED on — a 10 m span carrying
#: 10 m^2, which is what ``TailProblem`` opens on when nobody states a size.
#: Named because the bands are stated as FRACTIONS of the wing and this is
#: the wing that turns them back into the published metres.
B_REF_M = 10.0
S_REF_M2 = 10.0

#: the tail box, as a fraction of the WING it has to trim.
#:
#: A stabiliser is not a size, it is a RATIO: a surface a tenth of the wing's
#: area on an arm half its span is the same aeroplane at 10 m as at 1 m, and
#: nothing in this model knows how big an aeroplane is except the wing it is
#: handed. Stated in absolute m^2 and m — which is how they were written,
#: because the family that needed them had one size — the same box was
#: searched whatever the user typed into the size card. Measured on a 0.5 kg
#: model (1.2 m span, 0.18 m^2): the FLOOR of the area box was 0.5 m^2, 2.8x
#: the whole wing and a tail span of 1.41 m against the wing's 1.2 m, and the
#: arm box put a 1.2 m aeroplane's tail 3 to 8 m behind it. Every point of it
#: is "feasible" — the static-margin gate has a floor and no ceiling, so
#: SM 12-48 mac passes — so the optimiser drove to the floor of a box that
#: was still absurd, and the answer came back as an aeroplane nobody could
#: build.
#:
#: The fractions are the published box divided by the reference aeroplane
#: above, so at (b, S) = (10, 10) they are (0.5, 3.0) m^2 and (3.0, 8.0) m
#: EXACTLY and no published result moves. Read as ratios they are also the
#: conventional ranges: a horizontal tail 5-30 % of wing area, on an arm
#: 0.3-0.8 of the span (Raymer, tail sizing) — wide on both sides, so the
#: constraint stays the physics and not the box.
S_T_FRAC_BOUNDS = (0.05, 0.30)    # of the WING's reference area
L_T_FRAC_BOUNDS = (0.30, 0.80)    # of the WING's span

S_T_BOUNDS = (S_T_FRAC_BOUNDS[0] * S_REF_M2,
              S_T_FRAC_BOUNDS[1] * S_REF_M2)   # [m^2] on the reference wing
L_T_BOUNDS = (L_T_FRAC_BOUNDS[0] * B_REF_M,
              L_T_FRAC_BOUNDS[1] * B_REF_M)    # [m]   on the reference wing


def area_band(S: float = S_REF_M2) -> tuple:
    """The ``S_t_m2`` design-box row [m^2] for a wing of area ``S``."""
    return (S_T_FRAC_BOUNDS[0] * float(S), S_T_FRAC_BOUNDS[1] * float(S))


def arm_band(b: float = B_REF_M) -> tuple:
    """The ``l_t_m`` design-box row [m] for a wing of span ``b``."""
    return (L_T_FRAC_BOUNDS[0] * float(b), L_T_FRAC_BOUNDS[1] * float(b))


def default_arm(b: float = B_REF_M) -> float:
    """The arm a FIXED-arm layout flies when the user states none — the
    middle of :func:`arm_band`, which on the reference aeroplane is the
    5.5 m every published fixed-arm run was measured at."""
    lo, hi = arm_band(b)
    return 0.5 * (lo + hi)
#: ...and the FLOOR a stated arm has to clear. The band above is the box the
#: SEARCH opens on — a calibration, and the interval every published result
#: was measured over — but it is not a limit on the aeroplane: a longer
#: fuselage is a design decision, not an out-of-contract input, and refusing
#: one made the horizontal separation the single dimension of this model a
#: user could not state freely. What is genuinely required is that the
#: surface be BEHIND (or, for a canard, ahead of) the wing's own aerodynamic
#: centre by a real distance: at zero arm there is no moment arm to trim
#: with, the tail volume is zero and the pitch solve is singular. So a stated
#: arm is checked against this floor only, and a run outside the calibrated
#: band is honest extrapolation the shell says so about rather than a
#: refusal.
L_T_MIN_M = 0.10


def arm_row(l_t_bounds_m, default=L_T_BOUNDS) -> tuple:
    """The ``l_t_m`` design-box row [m] — the USER's band where there is one.

    The band above is a calibration, and the same sentence that makes a
    STATED arm free above :data:`L_T_MIN_M` makes a SEARCHED one free too:
    how long the aeroplane is, is a design decision. It was a ban in one
    direction only, and silently — widening the row in a shell's design box
    left the problem's own bounds at 3-8 m, so ``fg_tail`` returned
    "bounds violation" for every draw outside it and the search reported a
    penalty landscape instead of a longer aeroplane.

    So the band travels INTO the problem (``l_t_bounds_m``), exactly as the
    span's does (``sizing.span_bounds``), and the only thing checked is what
    the pitch solve genuinely needs: a real moment arm, low end first.

    ``None`` -> the family's own published band, so an untouched run is the
    published run bit-for-bit.
    """
    if l_t_bounds_m is None:
        return (float(default[0]), float(default[1]))
    lo, hi = float(l_t_bounds_m[0]), float(l_t_bounds_m[1])
    if lo < L_T_MIN_M:
        raise ValueError(
            f"searched arm band ({lo}, {hi}) m starts below the floor "
            f"{L_T_MIN_M} m: with no moment arm the tail volume is zero and "
            f"the pitch trim is singular. Any band above that floor is "
            f"yours — the CALIBRATED one is {tuple(default)} m, which is "
            f"where this family's results were measured, not a limit on the "
            f"aeroplane.")
    if not hi > lo:
        raise ValueError(
            f"searched arm band ({lo}, {hi}) m has hi <= lo; a zero-width "
            f"dimension breaks the samplers and silently degrades BO to "
            f"random search — state the arm instead (l_t_m as a flag), "
            f"which drops it from the design vector honestly")
    return (lo, hi)


@dataclass(frozen=True)
class TailLimits:
    """What the SECOND surface's own size has to obey, in METRES.

    The design box states the stabiliser as an AREA and (where the surface
    is designed) an ASPECT RATIO, which are the right variables for a solver
    and the wrong ones for a person: what a builder has is a fuselage that
    is only so wide, a mould that is only so long, and a rib pitch. Span and
    chord are those questions; area and aspect ratio are what they imply.

        b_t = sqrt(AR_t S_t)        c_t = sqrt(S_t / AR_t)

    Both directions are exact, which is what makes the band worth stating in
    metres rather than approximating: a span band and a chord band ARE an
    area band and an aspect-ratio band, and :meth:`narrow` is that change of
    variables. With the aspect ratio fixed (the plain tail, at
    :data:`TAIL_AR`) each band alone fixes the area exactly. With it free
    the image of a (span, chord) rectangle is not a rectangle in
    (area, AR) — so :meth:`narrow` returns the tightest box that CONTAINS
    it, and :meth:`violation` refuses the corners it added, the same
    division of labour ``geometry.ChordLimits`` uses on the wing.

    Every field optional; ``None`` is unconstrained, and an inactive
    TailLimits is the published behaviour exactly.
    """

    b_min_m: float | None = None
    b_max_m: float | None = None
    c_min_m: float | None = None
    c_max_m: float | None = None

    def __post_init__(self):
        for name in ("b_min_m", "b_max_m", "c_min_m", "c_max_m"):
            v = getattr(self, name)
            if v is not None and not float(v) > 0.0:
                raise ValueError(f"tail {name} must be > 0, got {v}")
        for lo, hi, what in ((self.b_min_m, self.b_max_m, "span"),
                             (self.c_min_m, self.c_max_m, "chord")):
            if lo is not None and hi is not None and not float(hi) > float(lo):
                raise ValueError(
                    f"tail {what} limits ({lo}, {hi}) m have max <= min")

    @property
    def active(self) -> bool:
        return any(v is not None for v in
                   (self.b_min_m, self.b_max_m, self.c_min_m, self.c_max_m))

    def violation(self, b_t: float, c_t) -> str | None:
        """Why this surface is refused, or None. ``c_t`` is a scalar mean
        chord or the drawn distribution — a designed tail is tested at every
        station, so a taper that meets the mean and misses at the tip is
        caught where it actually happens."""
        if not self.active:
            return None
        b = float(b_t)
        if self.b_min_m is not None and b < float(self.b_min_m) - 1e-12:
            return (f"tail span {b:.4g} m is below the {self.b_min_m:.4g} m "
                    f"minimum")
        if self.b_max_m is not None and b > float(self.b_max_m) + 1e-12:
            return (f"tail span {b:.4g} m exceeds the {self.b_max_m:.4g} m "
                    f"maximum")
        c = np.atleast_1d(np.asarray(c_t, dtype=float))
        if self.c_min_m is not None and float(c.min()) < float(self.c_min_m) \
                - 1e-12:
            return (f"tail chord {float(c.min()):.4g} m is below the "
                    f"{self.c_min_m:.4g} m minimum")
        if self.c_max_m is not None and float(c.max()) > float(self.c_max_m) \
                + 1e-12:
            return (f"tail chord {float(c.max()):.4g} m exceeds the "
                    f"{self.c_max_m:.4g} m maximum")
        return None

    def narrow(self, s_row, ar_row=None, ar_fixed: float = None) -> tuple:
        """(area row, AR row) narrowed to what these limits allow.

        Returns the rows unchanged when nothing is stated. ``ar_row`` None
        means the aspect ratio is not a design variable and ``ar_fixed`` is
        the value the family flies.
        """
        if not self.active:
            return (tuple(s_row), None if ar_row is None else tuple(ar_row))
        s_lo, s_hi = float(s_row[0]), float(s_row[1])
        b_lo = None if self.b_min_m is None else float(self.b_min_m)
        b_hi = None if self.b_max_m is None else float(self.b_max_m)
        c_lo = None if self.c_min_m is None else float(self.c_min_m)
        c_hi = None if self.c_max_m is None else float(self.c_max_m)
        if ar_row is None:
            ar = float(TAIL_AR if ar_fixed is None else ar_fixed)
            # S = b^2/AR = AR c^2, both exact at a fixed aspect ratio
            if b_lo is not None:
                s_lo = max(s_lo, b_lo * b_lo / ar)
            if b_hi is not None:
                s_hi = min(s_hi, b_hi * b_hi / ar)
            if c_lo is not None:
                s_lo = max(s_lo, ar * c_lo * c_lo)
            if c_hi is not None:
                s_hi = min(s_hi, ar * c_hi * c_hi)
            out_ar = None
        else:
            a_lo, a_hi = float(ar_row[0]), float(ar_row[1])
            # every implication of b = sqrt(AR S) and c = sqrt(S/AR), taken
            # against the OTHER row as well as against the other band — a
            # span cap alone says nothing about S or AR separately, but a
            # span cap AND an aspect-ratio row do (S <= b_max^2/AR_min).
            # Twice, because narrowing S narrows AR and vice versa; a third
            # pass adds nothing (each bound is monotone in the other).
            for _ in range(2):
                if b_lo is not None and c_lo is not None:
                    s_lo = max(s_lo, b_lo * c_lo)
                if b_hi is not None and c_hi is not None:
                    s_hi = min(s_hi, b_hi * c_hi)
                if b_lo is not None:
                    s_lo = max(s_lo, b_lo * b_lo / a_hi)
                if b_hi is not None:
                    s_hi = min(s_hi, b_hi * b_hi / a_lo)
                if c_lo is not None:
                    s_lo = max(s_lo, a_lo * c_lo * c_lo)
                if c_hi is not None:
                    s_hi = min(s_hi, a_hi * c_hi * c_hi)
                if not s_hi > s_lo:
                    break
                if b_lo is not None and c_hi is not None:
                    a_lo = max(a_lo, b_lo / c_hi)
                if b_hi is not None and c_lo is not None:
                    a_hi = min(a_hi, b_hi / c_lo)
                if b_lo is not None:
                    a_lo = max(a_lo, b_lo * b_lo / s_hi)
                if b_hi is not None:
                    a_hi = min(a_hi, b_hi * b_hi / s_lo)
                if c_lo is not None:
                    a_hi = min(a_hi, s_hi / (c_lo * c_lo))
                if c_hi is not None:
                    a_lo = max(a_lo, s_lo / (c_hi * c_hi))
                if not a_hi > a_lo:
                    break
            if not a_hi > a_lo:
                raise ValueError(
                    f"tail span limits ({b_lo}, {b_hi}) m and chord limits "
                    f"({c_lo}, {c_hi}) m leave no aspect ratio inside "
                    f"{tuple(ar_row)}: widen one of them, or state the "
                    f"surface instead of searching it")
            out_ar = (a_lo, a_hi)
        if not s_hi > s_lo:
            raise ValueError(
                f"tail span limits ({b_lo}, {b_hi}) m and chord limits "
                f"({c_lo}, {c_hi}) m leave no area inside "
                f"{(float(s_row[0]), float(s_row[1]))} m^2 — the band the "
                f"area is searched over. Widen a limit, or widen the S_t_m2 "
                f"row they are being intersected with.")
        return ((s_lo, s_hi), out_ar)


def area_row(s_t_bounds_m2, default) -> tuple:
    """The ``S_t_m2`` design-box row [m^2] — the USER's band where there
    is one.

    :func:`arm_row`'s twin, and it exists for the same reason twice over.
    The band the family opens on is now a FRACTION of the wing
    (:func:`area_band`), which is the right default and is emphatically not
    a limit: how big a stabiliser is, is a design decision, and a
    tail-volume argument, a rule-of-thumb from a similar aeroplane, or a
    surface already built are all reasons to search a different band.

    It also has to travel because a row that moves only in the SHELL is the
    bug ``_arm_band_kwargs`` documents: widening ``S_t_m2`` in the design
    box moved the sampler's box while the problem's own stayed the family's,
    and every draw outside it came back "bounds violation" — a search made
    entirely of penalties, reported as a search. Invisible while the box was
    a constant nobody could reach past; not invisible now that the default
    is sized to the wing.

    ``None`` -> the family's own band, so an untouched run is bit-for-bit.
    """
    if s_t_bounds_m2 is None:
        return (float(default[0]), float(default[1]))
    lo, hi = float(s_t_bounds_m2[0]), float(s_t_bounds_m2[1])
    # A band that STARTS at zero is a real question — "how small can this
    # surface get before the aeroplane stops trimming?" — and the solve
    # answers it honestly, returning `untrimmable: |i_t| exceeds limit` on
    # the designs that are too small. Refusing the number the user typed put
    # the form in the way of an answer the physics already gives. A NEGATIVE
    # area is still refused: that is not a smaller surface.
    from .sizing import positive_floor
    lo = positive_floor(
        lo, hi, f"the searched tail-area band ({lo}, {hi}) m^2")
    if not hi > lo:
        raise ValueError(
            f"searched tail-area band ({lo}, {hi}) m^2 has hi <= lo; a "
            f"zero-width dimension breaks the samplers and silently degrades "
            f"BO to random search — state the area instead")
    return (lo, hi)


def height_row(z_t_bounds_m, default) -> tuple:
    """The ``z_t_m`` design-box row [m] — the USER's band where there is one.

    The VERTICAL separation's twin of :func:`arm_row`, and it exists for the
    same reason. :data:`DZ_FRAC` * b is where this model's tail heights were
    MEASURED; it was also being enforced as a floor, in three places at once
    (a stated height was refused at construction, a searched one was bounded
    below by it, and every draw was re-checked inside ``fg_``), which made
    0.05 b the smallest vertical separation the whole model could express.

    That is not a small restriction. A conventional aeroplane's tailplane
    sits at or near the wing plane — 0.05 b is 0.5 m on a 10 m span and
    0.06 m on a 1.2 m hydrofoil — so the layout the model refused is the
    ordinary one, and the refusal bit hardest on exactly the SMALL aircraft
    the shell is used to size.

    It was also not true. The three solvers were measured through the floor
    to z_t = 0 and past it (scratch sweeps, session 36): every one is
    smooth, finite and monotone the whole way down, because the kernel that
    was said to be grid-singular is only singular ON its own line, which the
    principal-value guards in ``tandem.w_influence`` (rho2 / cross2 <
    1e-30) already handle, and a surface an arm's length downstream is never
    on it. Trailing-sheet effects a rigid inviscid wake genuinely cannot
    describe (roll-up, the dynamic-pressure deficit, deep-stall pitch-up)
    are unchanged by any floor — they were absent above 0.05 b too, so the
    floor bought no fidelity, only a ban.

    So :data:`DZ_FRAC` stays as the DEFAULT height a layout derives when
    nobody states one (:func:`tail_height`), and the searched band opens on
    it, and neither is a limit. What is still refused here is only what the
    samplers need: an interval with a low end first.

    ``None`` -> the family's own published band, so an untouched run is the
    published run bit-for-bit.
    """
    if z_t_bounds_m is None:
        return (float(default[0]), float(default[1]))
    lo, hi = float(z_t_bounds_m[0]), float(z_t_bounds_m[1])
    if not hi > lo:
        raise ValueError(
            f"searched height band ({lo}, {hi}) m has hi <= lo; a zero-width "
            f"dimension breaks the samplers and silently degrades BO to "
            f"random search — state the height instead (z_t_m as a flag), "
            f"which drops it from the design vector honestly. (In water the "
            f"separation is measured DOWN and is negative, so the low end is "
            f"the DEEPER one.)")
    return (lo, hi)


#: DESIGNED-tail boxes (tail_free). Aspect ratio: the published surface is
#: fixed at TAIL_AR = 4 and the conventional H-tail range is 3-5 (Raymer);
#: the box is opened either side of that so the answer comes from the
#: physics rather than the box. Washout is the TIP twist relative to a root
#: at zero — the uniform part of a tail's incidence IS the trim unknown, so
#: carrying a root twist too would leave the search a flat direction.
TAIL_AR_BOUNDS = (2.0, 8.0)
TAIL_WASHOUT_BOUNDS_DEG = (-6.0, 2.0)

#: supported horizontal-tail configurations (see the module docstring's
#: "Tail configuration" section for what each one does and does NOT model)
TAIL_TYPES = ("conventional", "t_tail", "v_tail", "canard")

#: Raymer vertical-tail sizing, reused from :func:`vtail_cd0` so the T-tail
#: height and the fin drag can never describe different fins.
#: re-exported from :mod:`fin`, which owns the fin's whole sizing law. They
#: keep their names here because ``tail_height`` and ``vtail_cd0`` have always
#: read them, and a second definition is exactly the two-authors problem
#: fin.py exists to end.
V_V_DEFAULT = _fin.V_V_DEFAULT      # vertical-tail volume coeff (Raymer 6.4)
AR_VT_DEFAULT = _fin.AR_VT_DEFAULT  # fin aspect ratio (conventional 1.3-2)

#: V-tail dihedral is a CONFIGURATION choice, never a design variable: with
#: no directional-stability constraint in this model a free dihedral would
#: be driven straight to 0 (all the pitch authority, none of the cost).
DIHEDRAL_BOUNDS_DEG = (20.0, 50.0)

#: CG position per configuration [m aft of the wing AC]. The aft-tail value
#: is a REAL LOADING first and a calibration second (MEASURED 2026-08-01,
#: docstring above); the canard's is a calibration, for the reason given.
#:   conventional/t_tail/v_tail: +0.05 m = 0.049 mac aft of the wing AC.
#:   The wing AC sits at 0.25c, so this is a CG at 30 % MAC — the middle of
#:   the 25-35 % MAC cruise range every reference quotes for this class
#:   (Raymer Ch. 16; Torenbeek Ch. 8). It was +0.25 m = 49.5 % MAC until
#:   2026-08-01: aft of any real aft limit, and the single reason the
#:   stabiliser came out LIFTING. At 30 % MAC the published tail carries
#:   CL_t = -0.038 (download) and every corner of the box does too
#:   (-0.018 .. -0.254), which is the textbook answer.
#:   The calibration duty survives the move and was re-measured, not
#:   assumed: SM spans +0.007 .. +1.194 over the 8 (taper, S_t, l_t)
#:   corners, so the SM = SM_min = 0.08 boundary still crosses the box
#:   interior, all 8 corners trim (|i_t| <= 6.96 deg against the 15 deg
#:   limit) and none hits the penalty contract. Moving the CG further
#:   forward loses that: at x_cg = 0.00 the minimum corner SM is +0.153,
#:   the stability constraint is vacuous everywhere and 2 corners fail.
#:   0.05 m is the most forward CG for which the constraint stays active.
#:   canard: -0.25 (RE-MEASURED 2026-08-01; was -0.40, MEASURED 2026-07-23,
#:   experiments/canard_xcg_calibration.py). A canard's neutral point lies
#:   FORWARD of the wing AC (x_np spans -1.633 .. -0.123 m over the box), so
#:   the aft-tail CG would report SM ~ -0.7 everywhere: every canard would
#:   look plausible and be unconditionally unstable. The CG stays forward of
#:   the wing AC and the canard stays a LIFTING surface (+0.194 at the
#:   published point) — which is what a canard is: it sits ahead of the CG,
#:   so it holds the nose up rather than down, and carrying the section
#:   couple only asks it for more of the same. That is why the value had to
#:   move: at -0.40 with the couple included the small-canard/short-arm
#:   corners demand CL_t ~ 1.7 and go untrimmable (|i_t| > 15 deg), which is
#:   the penalty contract, not a design. At -0.25 the margin spans
#:   -1.383 .. +0.125 (crossing SM_min = 0.08 inside the box), all 8 corners
#:   trim with |i_t| <= 12.21 deg, and it is the most AFT such CG: by -0.20
#:   the SM boundary has left the box.
#:
#: Quoted as a FRACTION OF THE REFERENCE CHORD (S/b) for the reason the tail
#: box above is quoted as a fraction of the wing: where a CG sits is a
#: statement about the aeroplane's own chord, and 0.05 m is 0.05 mac on the
#: 10 m aeroplane these were calibrated on but a third of the chord on a
#: 1.2 m one — which loads a model aeroplane a third of a chord aft of its
#: wing AC without anybody asking for it. :data:`X_CG_BY_TYPE` keeps the
#: published metres for the reference aeroplane, and every problem derives
#: its own through :func:`default_x_cg`.
X_CG_MAC_BY_TYPE = {"conventional": 0.05, "t_tail": 0.05, "v_tail": 0.05,
                    "canard": -0.25}

#: ...the same numbers in metres on the reference aeroplane (S/b = 1.0 m),
#: which is what every published run flew.
X_CG_BY_TYPE = {k: v * (S_REF_M2 / B_REF_M)
                for k, v in X_CG_MAC_BY_TYPE.items()}


def default_x_cg(tail_type: str, b: float = B_REF_M,
                 S: float = S_REF_M2) -> float:
    """The layout's calibrated CG [m aft of the wing AC] on THIS wing.

    The reference chord S/b and not the candidate's own MAC: the CG is a
    property of the problem, fixed before any design vector exists, and a
    number that moved with the taper being searched would make the balance
    a function of the thing it is balancing.
    """
    return X_CG_MAC_BY_TYPE[tail_type] * (float(S) / float(b))


def cg_reference_size(b: float, S: float, *, size_free=False,
                      span_bounds_m=None, area_bounds_m2=None,
                      ws_bounds_pa=None, wing_loading_Pa=None,
                      W_design_N: float | None = None) -> tuple:
    """The (span, area) :func:`default_x_cg` calibrates the CG on.

    The family's own reference wing where the size is FIXED — that pair IS
    the aeroplane, and nothing here changes it. Where the size is a design
    variable it is that same pair CLAMPED INTO THE BOX THE RUN SEARCHES,
    which is the whole of this function and the defect it closes.

    THE REPORTED FAILURE. On a sized family ``b`` and ``S`` are the seeds of
    two design rows, not the aircraft: a shell that states a mission moves
    the ROWS (``api._size_band_kwargs`` -> ``span_bounds_m`` /
    ``area_bounds_m2``) and leaves the seeds at the family's published 10 m
    and 10 m². The CG is fixed before the design vector exists, so it was
    derived from those seeds — 0.05 m, which is 0.05 mac on the 10 m
    aeroplane and 0.62 mac on the 0.8 m one the mission actually stated. A
    CG two thirds of a chord aft of the wing AC is behind the neutral point
    of anything this family can build, so EVERY design in the box came back
    statically unstable (SM -0.25 against SM_min 0.08), the measured box was
    empty, and the shell said "no solution" about an aeroplane that flies
    perfectly well: reported as *"5N plane (wing + tail) 0.8m span no
    solution found"*, and reproduced at 0 of 32 admissible draws against
    9 of 32 with the CG this function returns.

    WHY CLAMP RATHER THAN TAKE THE BOX CENTRE. The published runs pass the
    same bands the family derives for itself, and those bracket the
    reference wing: clamping returns it unchanged, so every published number
    reproduces bit-for-bit, while a centre would move all of them by the
    band's own asymmetry. Where the box does NOT contain the reference wing
    the nearest end is the closest aeroplane the run can actually build, and
    it is a property of the BOX — fixed before the vector, which is the rule
    :func:`default_x_cg` exists to keep.

    The two WING-LOADING modes carry no area row: the area follows the
    loading through the weight loop (``sizing.area_for_wing_loading``), so
    the band it may take is read back off the loading — W/S_hi to W/S_lo, at
    the design weight. That weight is the payload, not the closed gross
    weight the loop lands on; the CG is a reference, and a number that
    waited for the structure loop would depend on the span it is balancing.
    """
    b_ref, S_ref = float(b), float(S)
    if not size_free:
        return b_ref, S_ref

    def _clamp(v, band):
        # a band may be HALF-OPEN: a shell that states only a minimum span
        # sends (8.0, None), and an open end constrains nothing
        lo = None if band[0] is None else float(band[0])
        hi = None if band[1] is None else float(band[1])
        if lo is not None and hi is not None and hi < lo:
            lo, hi = hi, lo
        if lo is not None:
            v = max(v, lo)
        if hi is not None:
            v = min(v, hi)
        return v

    if span_bounds_m is not None:
        b_ref = _clamp(b_ref, span_bounds_m)
    if area_bounds_m2 is not None:
        S_ref = _clamp(S_ref, area_bounds_m2)
    elif wing_loading_Pa or ws_bounds_pa:
        w = float(W_design_N) if W_design_N else 0.0
        if w > 0.0:
            if wing_loading_Pa:
                S_ref = w / float(wing_loading_Pa)
            else:
                lo, hi = ws_bounds_pa[0], ws_bounds_pa[1]
                lo = None if lo is None or float(lo) <= 0.0 else float(lo)
                hi = None if hi is None or float(hi) <= 0.0 else float(hi)
                if lo is not None or hi is not None:
                    S_ref = _clamp(S_ref, (None if hi is None else w / hi,
                                           None if lo is None else w / lo))
    return b_ref, S_ref


def _cg_design_weight(prob) -> float:
    """The design weight the CG reference reads an area back off [N].

    The mission's own where there is one — a shell states it — and the
    weight this family's own trim point implies otherwise
    (``mission.weight_for``), which is the same number every sized family
    already uses for its payload (``sizing.py``: ``W_fixed`` defaults to it).
    Only the two wing-loading modes ask, and only to turn a loading back into
    an area.
    """
    from .mission import weight_for
    w = getattr(prob.mission, "W_N", None)
    if w is not None and float(w) > 0.0:
        return float(w)
    return weight_for(prob.CL_target, prob.V, prob.S)


def flap_effectiveness(chord_frac: float) -> float:
    """Thin-aerofoil flap-effectiveness factor tau(c_e/c_t).

    tau = 1 - (theta_f - sin theta_f)/pi with theta_f = arccos(2 c_e/c_t - 1)
    (the standard hinged-flap result, e.g. Etkin & Reid; Raymer Ch. 16):
    the fraction of a whole-surface incidence change that a trailing-edge
    deflection of the same angle achieves. At c_e/c_t = 0.30, tau = 0.661.

    This is a RE-PARAMETERISATION of the trim solution, not new physics: a
    linear flap enters the surface's boundary condition as the same uniform
    RHS shift as incidence, so L/D and the trim state are unchanged and only
    the required deflection (and hence the deflection LIMIT) differs.
    """
    r = float(np.clip(chord_frac, 1e-6, 1.0))
    theta_f = float(np.arccos(2.0 * r - 1.0))
    return float(1.0 - (theta_f - np.sin(theta_f)) / np.pi)


def tail_span(S_t: float) -> float:
    """Tail span from the fixed aspect ratio: b_t = sqrt(AR_t * S_t)."""
    return float(np.sqrt(TAIL_AR * S_t))


# ---- layout geometry, as PURE functions ------------------------------------
# TailProblem's methods below delegate to these unchanged; they live at module
# level so the nonplanar sibling (wingtail.py — wing + winglet + tail in ONE
# VLM) draws the same layouts from the same source instead of restating them.


def surface_x(tail_type: str, dist: float) -> float:
    """Signed streamwise station of the surface [m] (canard: upstream)."""
    return -float(dist) if tail_type == "canard" else float(dist)


def lifting_area(tail_type: str, dihedral_deg: float, S_t: float) -> float:
    """Area of the EQUIVALENT horizontal surface [m^2] (V-tail: S cos^2 G)."""
    if tail_type != "v_tail":
        return float(S_t)
    return float(S_t) * float(np.cos(np.deg2rad(dihedral_deg))) ** 2


def tail_height(tail_type: str, dist: float, b: float, S: float) -> float:
    """Height of the surface above the wing plane [m].

    DZ_FRAC*b (the MEASURED kernel-regularisation floor) for every layout
    except the T-tail, which sits on top of the fin implied by the same
    Raymer sizing the drag build-up uses. See TailProblem.dz_for.
    """
    floor = DZ_FRAC * b
    if tail_type != "t_tail":
        return floor
    S_vt = V_V_DEFAULT * b * S / max(float(abs(dist)), 1e-6)
    return float(max(floor, np.sqrt(AR_VT_DEFAULT * S_vt)))


# ------------------------------------------------------------------ problem


@dataclass
class TailProblem:
    """Phase 4 wing + H-tail problem: trimmed L/D with a static-margin gate.

    Design vector (d = 5):

        x = [taper, twist_root_deg, twist_tip_deg, S_t_m2, l_t_m]

    Wing taper/twist conventions are geometry.py's (twist + = nose-up,
    washout = negative tip). S_t in [0.5, 3.0] m^2, l_t in [3.0, 8.0] m.
    x_cg = 0.25 m aft of the wing AC is a CALIBRATED constant (module
    docstring — the SM boundary crosses the box interior). CL_target is
    the SYSTEM CL on Sref = S (wing area, aircraft convention).

    Objective (MAXIMISE): L/D at the 2-D trim point.
    Constraint: g = (SM - SM_min)/1.0 >= 0.
    Contract: fg_tail(x) -> (LoD, g) for every solvable design (true value
    + signed margin); failures return exactly (PENALTY, G_FAIL).
    """

    b: float = 10.0            # wing span [m]
    S: float = 10.0            # wing area [m^2] — also Sref for ALL coefficients
    N: int = 60                # wing LLT stations (matches Tier A Problem.N so
    #                            the S_t -> 0 limit reproduces objective.evaluate)
    N_t: int = 30              # tail LLT stations
    V: float = 14.6            # m/s -> Re_mac ~ 1e6 (Tier A flow state)
    rho: float = RHO_SL
    mu: float = MU_SL
    CL_target: float = 0.5
    x_cg: float | None = None  # [m] CG aft of wing AC; None -> X_CG_BY_TYPE
    #: [m] the surface's height above the wing plane, STATED. None (the
    #: default, and every published run) keeps the layout's own —
    #: :func:`tail_height`, i.e. the measured DZ_FRAC*b clearance or a
    #: T-tail's fin span. Stating one is a real physical choice here and not
    #: a relabelling: dz enters the coupled solve through the horseshoe
    #: kernel, so a surface further out of the wing's trailing sheet sees
    #: less downwash and the neutral point moves aft with it. The magnitude
    #: is floored at the measured regularisation clearance (below it the
    #: kernel is grid-singular — module docstring) and a T-TAIL refuses one
    #: outright, because its height IS the fin span its own Raymer sizing
    #: implies. Sign: + above the wing plane, − below it — but the planar
    #: kernel dy/(dy^2 + dz^2) is EVEN in dz, so this model cannot tell a
    #: surface above the wing from the same one below it (measured: ±1.0 m
    #: give identical SM and L/D). Documented limit, not a bug: distinguishing
    #: them needs the wake's roll-up, which a rigid planar sheet has not got.
    z_t_fixed: float | None = None
    #: which way up the STABILISER's section flies. None (the default) =
    #: FOLLOW THE LOAD — inverted where the surface pushes down, upright
    #: where it lifts (:func:`tail_polar`), the same rule the tail's tip
    #: device already obeys. True/False state it and the load gets no vote,
    #: which is how the two orientations are compared. A symmetric section
    #: is its own mirror, so this cannot move one.
    tail_inverted: bool | None = None
    SM_min: float = 0.08
    i_t_max_deg: float = 15.0  # untrimmable proxy: |i_t| beyond this fails
    cd0_extra: float = 0.0     # non-modelled parasite hook (e.g. vtail_cd0)
    polar: Any = field(default_factory=default_polar)
    #: the TAIL's own section, when it is not the wing's. None (default)
    #: keeps the documented "tail reuses the wing polar" approximation, so
    #: every published tail run is bit-for-bit unchanged; setting it makes
    #: the stabiliser's lift slope, zero-lift angle, stall proxy and profile
    #: drag its own — which is what choosing a section per surface means.
    polar_tail: Any = None
    alpha_bracket_deg: tuple[float, float] = (-10.0, 15.0)
    # ---- configuration (all defaults reproduce the 5-D aft-tail problem)
    tail_type: str = "conventional"
    dihedral_deg: float = 0.0        # V-tail panel dihedral [deg]
    #: THE FIN'S OWN SHAPE, where a user states one. ``None`` on all three
    #: means the published law (``fin.V_V_DEFAULT`` 0.04, ``AR_VT_DEFAULT``
    #: 1.5, ``FIN_TC_DEFAULT`` 0.10), so every stored run is bit-for-bit.
    #:
    #: They travel TOGETHER to every consumer — the drag book that charges
    #: the fin, the lateral deck that flies it, the report that states it,
    #: the exporter that lofts it — because the defect this whole surface
    #: was built out of was a fin sized one way in one place and another way
    #: somewhere else. A stated volume coefficient that reached the drawing
    #: but not the drag would be the same bug wearing a control.
    fin_volume_coeff: float | None = None
    fin_ar: float | None = None
    fin_tc: float | None = None
    #: IS THERE A FIN AT ALL. ``True`` reproduces every published run
    #: bit-for-bit, because until this field existed there was no way to say
    #: otherwise: the layout answered the question on its own
    #: (``fin.size_fin`` returns None for a V-tail) and the shell's switch
    #: reached nothing. It reached nothing loudly — a design with the fin
    #: switched off was still charged ``cd0_fin`` 0.000885 (5.79 % of L/D
    #: here), still weighed a surface that is 82 % of the empennage book,
    #: still had one drawn in its report and still flew one in stage 6,
    #: while the mission card said "Cn_beta is exactly zero — not small".
    #:
    #: Read through ``fin.has_fin``, never directly, because a V-TAIL is the
    #: other way of answering no and the two must not be checked in
    #: different orders by different consumers.
    fin: bool = True
    #: FUSELAGE DIAMETER [m] — the one number that makes a SEARCHED arm a
    #: well-posed question, and ``None`` (off) reproduces every published run
    #: bit-for-bit.
    #:
    #: Without it the wing-to-tail separation is a RATCHET. The fin's area
    #: goes as ``V_v b S / l_t``, so a longer arm needs less fin and less fin
    #: drag, the tail shrinks for the same static margin, and nothing at all
    #: pushes back: measured on this family at b = 10 m, L/D climbs
    #: monotonically 25.41 -> 32.57 from a 1 m arm to a 100 m one, with the
    #: static margin running to +8.7 and the fin down to 0.04 m^2. The
    #: optimiser is right and the model is wrong — the FUSELAGE is free.
    #:
    #: ``aircraft.py`` records the omission ("W_fixed carries the fuselage/
    #: tail/systems WEIGHT but their parasite drag is not modelled at this
    #: tier ... hook non-wing parasite drag through cd0_extra when a fuselage
    #: model arrives"). The model arrived long ago — ``drag.fuselage_cd0``,
    #: a Raymer body, sitting unused. Stating a diameter wires it in, with
    #: the length from ``drag.body_length_for_arm``, and the sum
    #: ``cd0_fus + cd0_fin`` then has an INTERIOR minimum: at D = 0.35 m it
    #: is at 3.0 m of arm, the bottom of this family's own calibrated band.
    #:
    #: It is a fact about the aeroplane, so it is asked and never defaulted.
    fuselage_diameter_m: float | None = None
    control: str = "stabilator"      # "stabilator" (all-moving) | "elevator"
    elevator_chord_frac: float = 0.30
    delta_e_max_deg: float = 25.0
    l_t_fixed: float | None = None   # fix the arm -> 4-D problem
    l_t_bounds_m: tuple | None = None  # (min, max) the SEARCHED arm is boxed
    #                                  by; None -> arm_band(b), the
    #                                  calibrated band (:func:`arm_row`)
    s_t_bounds_m2: tuple | None = None  # ...and the SEARCHED tail area's own
    #                                  band [m^2]; None -> area_band(S)
    #                                  (:func:`area_row`)
    tail_limits: "TailLimits | None" = None   # what the SECOND surface's
    #                                  own SPAN and CHORD have to obey, in
    #                                  metres (:class:`TailLimits`). None,
    #                                  or an inactive one, is the published
    #                                  behaviour exactly.
    chord_order: int = 0             # free chord law (geometry.py) — the
    #                                  WING's, and the tail's too when the
    #                                  tail is a designed surface
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
    # ---- the TAIL as a designed surface. Off (default), it is the published
    # RECTANGLE at aspect ratio TAIL_AR whose only freedoms are its area and
    # its arm — every frozen tail study, bit-for-bit. On, it carries its own
    # taper, aspect ratio and washout, plus its own chord law when the wing
    # has one, exactly as the wing does. Its incidence stays the trim
    # unknown, so the washout (tip relative to root) is the twist freedom:
    # a root twist would be the same variable as i_t twice over. There is no
    # tip device here — these lifting lines are planar by construction; that
    # freedom lives in the nonplanar wingtail.py family.
    #: the aspect-ratio band the USER will accept b^2/S in, as
    #: ``(ar_min, ar_max)`` with either end None (api.AR_LIMIT_KEYS).
    #: A SECOND band beside ``sizing.AR_LIMITS``, never a replacement
    #: for it, and None — no limit at all — is every published run.
    ar_limits: "tuple | None" = None
    tail_free: bool = False
    wing_loading_Pa: float | None = None   # W/S for the wing-loading size
    #                                  mode (sizing.SIZE_MODE_WS): the WING's
    #                                  area follows it, so the vector carries
    #                                  the span alone. None -> the loading
    #                                  this problem already flies,
    #                                  CL_target x q. The TAIL's area stays a
    #                                  design variable of its own — sizing the
    #                                  aircraft is not sizing its stabiliser.
    span_bounds_m: tuple | None = None     # (min, max) span for that row
    area_bounds_m2: tuple | None = None    # (min, max) AREA row of the free
    #                                  planform mode [m^2]; None -> the
    #                                  fractional band around this problem's
    #                                  own area (sizing.size_bounds)
    ws_bounds_pa: tuple | None = None      # (min, max) of the SEARCHED
    #                                  wing-loading row [Pa]
    #                                  (sizing.SIZE_MODE_WS_FREE); None ->
    #                                  sizing.WS_FRAC_BOUNDS around the
    #                                  loading this problem already flies
    wing_loading_max_Pa: float | None = None   # the MISSION's ceiling on W/S
    #                                  (constraint_diagram): clips that band
    #                                  and refuses any candidate above it, in
    #                                  every sized mode
    material: object = None   # WHAT THE WING IS BUILT OF
    #   (materials.Material, or a key from materials.MATERIALS).
    #   None = 2024-T3 aluminium, the material Raymer's correlation
    #   was regressed over, so every published run is bit-identical.
    #   It sets BOTH the allowable the spar is sized to and the
    #   factor the statistical wing weight is scaled by — the second
    #   is the one the span answer actually rides (materials.py).
    size_free: bool | str = False    # SIZE modifier (sizing.py): span and
    #                                  area become design variables, the wing
    #                                  weighs what its size implies, the trim
    #                                  target follows, the score becomes
    #                                  payload L/D and a root-bending stress
    #                                  margin joins the static margin as a
    #                                  SECOND constraint. The tail's own mass
    #                                  sits inside W_fixed (the Raymer
    #                                  correlation is a WING weight).
    W_fixed_N: float | None = None   # non-wing weight for the size modifier
    flight_free: bool = False        # speed + altitude as design variables
    #                                  (geometry.py's flight modifier). The
    #                                  static-margin constraint is unmoved —
    #                                  this inviscid neutral point does not
    #                                  depend on speed — so the modifier buys
    #                                  the tail sizing at a CHOSEN cruise
    #                                  point rather than the fixed one.
    mission: Any = None              # mission.MissionSpec (design weight);
    #                                  defaults, when the flight state is
    #                                  free, to the weight this problem's own
    #                                  CL_target already implies
    _coupling_cache: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        if self.flight_free and self.mission is None:
            from .mission import MissionSpec, isa_density
            q0 = 0.5 * isa_density(0.0) * self.V**2
            self.mission = MissionSpec(W_N=self.CL_target * (q0 * self.S),
                                       V=self.V, altitude_m=0.0)
        if self.tail_type not in TAIL_TYPES:
            raise ValueError(f"unknown tail_type {self.tail_type!r}; "
                             f"choices: {list(TAIL_TYPES)}")
        if self.control not in ("stabilator", "elevator"):
            raise ValueError(f"unknown control {self.control!r}; "
                             f"choices: ['stabilator', 'elevator']")
        if self.x_cg is None:
            # ...on the wing THIS PROBLEM can build, which on a sized family
            # is not (b, S): those are the seeds of two design rows, and the
            # mission's own aeroplane is in the bands (cg_reference_size)
            self.x_cg = default_x_cg(self.tail_type, *cg_reference_size(
                self.b, self.S, size_free=self.size_free,
                span_bounds_m=self.span_bounds_m,
                area_bounds_m2=self.area_bounds_m2,
                ws_bounds_pa=self.ws_bounds_pa,
                wing_loading_Pa=self.wing_loading_Pa,
                W_design_N=_cg_design_weight(self)))
        if self.tail_type == "v_tail":
            lo, hi = DIHEDRAL_BOUNDS_DEG
            if not (lo <= float(self.dihedral_deg) <= hi):
                raise ValueError(
                    f"V-tail dihedral {self.dihedral_deg} deg outside the "
                    f"supported range {DIHEDRAL_BOUNDS_DEG} — outside it the "
                    f"area-equivalence is not a V-tail any more")
        if self.l_t_bounds_m is not None:
            if self.l_t_fixed is not None:
                raise ValueError(
                    "the arm cannot be both a stated value (l_t_fixed) and a "
                    "searched band (l_t_bounds_m) — state it, or box it")
            arm_row(self.l_t_bounds_m)          # validated once, here
        if self.l_t_fixed is not None and float(self.l_t_fixed) < L_T_MIN_M:
            raise ValueError(
                f"stated arm {self.l_t_fixed} m is below the floor "
                f"{L_T_MIN_M} m: with no moment arm the tail volume is zero "
                f"and the pitch trim is singular. (The CALIBRATED band is "
                f"{L_T_BOUNDS} m — the box the search opens on — but a "
                f"longer aeroplane is a design decision, not an "
                f"out-of-contract input.)")
        if self.z_t_fixed is not None:
            if self.tail_type == "t_tail":
                raise ValueError(
                    "a T-tail's height IS the fin span its own Raymer sizing "
                    "implies (tail_height), so it cannot also be stated — "
                    "use the conventional layout to place the surface")
            # ...and NO floor on the value: DZ_FRAC*b is where the heights
            # were measured, not the smallest one that exists. A tailplane
            # in the wing plane is the ordinary layout, and the solve is
            # smooth through 0.05 b to zero (tail.height_row).

    # ---- configuration-dependent geometry ------------------------------

    def surface_x(self, dist: float) -> float:
        """Signed streamwise station of the surface [m].

        ``dist`` is always the DISTANCE from the wing AC; a canard sits
        upstream, at negative x. The horseshoe kernel handles that natively
        (its (1 + dx/r) factor tends to 0 far upstream, i.e. no wake ahead
        of the wing), so the same coupled solve serves both layouts.
        """
        return surface_x(self.tail_type, dist)

    def lifting_area(self, S_t: float) -> float:
        """Area of the EQUIVALENT horizontal surface [m^2].

        A V-tail panel at dihedral Gamma sees alpha*cos(Gamma) normal to
        itself and returns cos(Gamma) of that normal force vertically, so
        its pitch effectiveness is that of a flat tail of area
        S_t cos^2(Gamma) — and, since sqrt(AR_t * S_t cos^2) equals the
        horizontal projected span, at the SAME aspect ratio. The wetted
        area does NOT shrink with dihedral; that is charged separately in
        the profile-drag term.
        """
        return lifting_area(self.tail_type, self.dihedral_deg, S_t)

    def dz_for(self, dist: float) -> float:
        """Height of the surface above the wing plane [m].

        For every layout except the T-tail this is the MEASURED kernel
        regularisation DZ_FRAC*b = 0.05 b (module docstring). A T-tail sits
        on top of the fin, so its height is the fin span implied by the same
        Raymer sizing the drag build-up uses, h = sqrt(AR_vt * S_vt) with
        S_vt = V_v b S / dist — 1.41 m at dist = 3 m falling to 0.87 m at
        8 m. Every value clears the 0.5 m regularisation floor, so the
        T-tail is always inside the grid-converged regime, and the floor is
        enforced rather than assumed.

        A STATED height (``z_t_fixed``) overrides both — it is the same
        quantity, answered by the user instead of by the layout, and it is
        validated once at construction rather than clamped here.

        This raises the tail out of the wing's trailing sheet and genuinely
        reduces the downwash it sees — an INVISCID effect only. The wake
        here is rigid, planar and inviscid: there is no dynamic-pressure
        deficit, no roll-up, no wake sink and no deep-stall pitch-up, all of
        which are real T-tail considerations this model cannot speak to.
        """
        if self.z_t_fixed is not None:
            return float(self.z_t_fixed)
        return tail_height(self.tail_type, dist, self.b, self.S)

    @property
    def dz(self) -> float:
        """Tail height for the DEFAULT arm (kernel regularisation).

        Retained for callers that want a single number; the solve itself
        uses :meth:`dz_for`, because a T-tail's height depends on the arm.
        """
        return self.dz_for(default_arm(self.b))

    @property
    def tau_elevator(self) -> float:
        """Control effectiveness: 1.0 for an all-moving stabilator."""
        if self.control != "elevator":
            return 1.0
        return flap_effectiveness(self.elevator_chord_frac)

    @property
    def bounds(self) -> np.ndarray:
        # the tail's SIZE rows, in the units the solver searches — and
        # narrowed first to whatever the user stated in metres, because a
        # span or chord limit IS an area/aspect-ratio band (TailLimits)
        s_row = area_row(self.s_t_bounds_m2, area_band(self.S))
        ar_row = TAIL_AR_BOUNDS if self.tail_free else None
        if self.tail_limits is not None:
            s_row, ar_row = self.tail_limits.narrow(s_row, ar_row, TAIL_AR)
        rows = [
            geometry.TAPER_BOUNDS,
            geometry.TWIST_ROOT_BOUNDS_DEG,
            geometry.TWIST_TIP_BOUNDS_DEG,
            s_row,
        ]
        if self.l_t_fixed is None:
            rows.append(arm_row(self.l_t_bounds_m, arm_band(self.b)))
        if self.tail_free:
            rows += [geometry.TAPER_BOUNDS, ar_row,
                     TAIL_WASHOUT_BOUNDS_DEG]
        from .sizing import with_size_bounds, ws_of
        box = with_size_bounds(np.array(rows, dtype=float), self.size_free,
                               self.b, self.S, self.span_bounds_m,
                               area_bounds_m2=self.area_bounds_m2,
                               ws0=ws_of(self.CL_target, self.rho, self.V),
                               ws_bounds_pa=self.ws_bounds_pa,
                               wing_loading_max_Pa=self.wing_loading_max_Pa)
        box = geometry.with_flight_bounds(box, self.flight_free)
        box = geometry.with_chord_bounds(box, self.chord_order,
                                         self.chord_max_frac, self.chord_law)
        if self.tail_free and self.chord_order:
            box = geometry.with_chord_bounds(box, self.chord_order,
                                             self.chord_max_frac,
                                             self.chord_law)
        return box

    @property
    def param_labels(self) -> tuple:
        """Design-vector names — the ONE place this family's layout is
        written down (api mirrors it for the static ProblemSpec)."""
        labels = ["taper", "twist_root_deg", "twist_tip_deg", "S_t_m2"]
        if self.l_t_fixed is None:
            labels.append("l_t_m")
        if self.tail_free:
            labels += ["taper_t", "AR_t", "washout_t_deg"]
        from .sizing import size_labels
        chord = geometry.chord_labels(self.chord_order,
                                      self.chord_law)
        if self.tail_free and chord:
            chord = chord + tuple(f"{lbl}_t" for lbl in chord)
        return (tuple(labels) + size_labels(self.size_free)
                + geometry.flight_labels(self.flight_free) + chord)

    @property
    def n_chord_rows(self) -> int:
        """Trailing chord rows: one block per DESIGNED surface."""
        return int(self.chord_order) * (2 if self.tail_free else 1)

    @property
    def dim(self) -> int:
        return int(self.bounds.shape[0])

    def coupling(self, b_t: float, x_t: float, dz: float):
        """Coupling operators for (tail SPAN, signed station, height).

        Keyed on all three because the T-tail's height and the canard's
        sign both vary with the design — the previous (S_t, l_t) key would
        silently serve a conventional tail's operators to a T-tail. The span
        is the key rather than the area because a DESIGNED tail chooses its
        own aspect ratio: two tails of equal area no longer share a wake.
        """
        key = (round(float(b_t), 12), round(float(x_t), 12),
               round(float(dz), 12))
        if key not in self._coupling_cache:
            fs = VortexSystem(b=self.b, N=self.N, x=0.0, z=0.0)
            rs = VortexSystem(b=float(b_t), N=self.N_t,
                              x=float(x_t), z=float(dz))
            self._coupling_cache[key] = build_coupling(fs, rs)
        return self._coupling_cache[key]


# ------------------------------------------------------------------ moments


def section_moment(cfg: "_TailConfig", pol, pol_t) -> float:
    """The two surfaces' zero-lift section couple, as M/q [m^3].

    A cambered section carries a nose-down couple that does NOT vanish with
    its lift, so it is not a lift-times-arm term and no arm makes it
    disappear: the surface can be at the CG and still pitch the aircraft.
    Integrated on the same grids the coupled solve integrates lift on,

        M/q = integral cm_ac(y) c(y)^2 dy = cm_ac * S * mac

    (the second form because MAC is DEFINED as (1/S) integral c^2 dy — the
    identity is exact here, not a small-taper approximation, which is why
    the section moment can be charged without a separate mean chord).

    Both surfaces contribute. ``cm_ac`` is read once per section at its own
    zero-lift angle (:func:`polar.section_cm_ac`), where cm(c/4) IS the
    moment about the aerodynamic centre; it is a CONSTANT in (alpha, i_t),
    so every affine structure downstream — the 2x2 trim elimination, the
    neutral point, the static margin — is untouched by adding it.

    A symmetric section returns exactly 0.0 and reproduces every result
    taken before this term existed.

    V-tail limit: ``cfg.c_t``/``cfg.y_t`` already describe the EQUIVALENT
    HORIZONTAL surface (S cos^2 G), so the couple is charged on cos^2 G
    where the strict pitch component of a canted section moment is cos G.
    The difference is ~0.006 in Cm at G = 35 deg — below the term's own
    modelling error, and it keeps ONE area convention for the whole
    surface rather than two.
    """
    from .polar import section_cm_ac
    m_w = section_cm_ac(pol) * float(np.trapezoid(cfg.c_w ** 2, cfg.y_w))
    m_t = section_cm_ac(pol_t) * float(np.trapezoid(cfg.c_t ** 2, cfg.y_t))
    return float(m_w + m_t)


def cm_about_cg(res: TandemResult, x_cg: float, l_t: float,
                Sref: float, mac: float, x_w: float = 0.0,
                M_ac: float = 0.0) -> float:
    """Pitching-moment coefficient about the CG, NOSE-UP POSITIVE.

    Derived in the module docstring (M_y = -(x_i - x_cg) L_i per surface),
    plus the sections' own couple (:func:`section_moment`, ``M_ac`` = M/q
    in m^3, nose-up positive like everything else here):

        Cm_cg = [(x_cg - x_w) CL_w S_w + (x_cg - l_t) CL_t S_t + M_ac]
                / (Sref mac)

    with CL_i S_i = L_i / q taken from the coupled solve's per-surface
    results (numerical areas, tandem.py convention). Lift aft of the CG
    (x_i > x_cg) contributes NEGATIVE (nose-down) — the sign the SM gate
    and the aft-CG consistency test lean on.

    ``M_ac`` defaults to 0.0: the symmetric-section value, i.e. the honest
    "this caller has no section moment to report", which reproduces every
    result taken before the term existed.
    """
    Lw = res.front.CL * res.front.S          # L/q of the wing
    Lt = res.rear.CL * res.rear.S            # L/q of the tail
    return float(((x_cg - x_w) * Lw + (x_cg - l_t) * Lt + M_ac)
                 / (Sref * mac))


def trim_lift_coefficient(CL_target: float, Sref: float, x_cg: float,
                          S_lift: float, l_t: float,
                          x_w: float = 0.0, M_ac: float = 0.0) -> float:
    """Lift coefficient the TRIMMING surface must carry, in closed form.

    The trim is two equations (the solve at the bottom of this module finds
    the (alpha, i_t) that satisfy them, but the LOADS they imply do not need
    the solve):

        Cm_cg = 0:  (x_cg - x_w) CL_w S_w + (x_cg - l_t) CL_t S_t + M_ac = 0
        CL_total:   (CL_w S_w + CL_t S_t) / Sref = CL_target

    Eliminating CL_w S_w between them leaves the surface's share as pure
    geometry plus the sections' own couple:

        CL_t = [CL_target Sref (x_cg - x_w) + M_ac] / (S_t (l_t - x_w))

    — no lift-curve slope, downwash or solver fidelity appears, because the
    two constraints already fix what each surface carries. ``S_t`` is the
    EQUIVALENT HORIZONTAL area (:func:`lifting_area` — a V-tail carries its
    lift on S cos^2 G) and ``l_t`` is the SIGNED station
    (:func:`surface_x` — a canard is upstream, and so is its CG).

    ``M_ac`` (:func:`section_moment`, M/q in m^3) is the term that decides
    the SIGN of a stabiliser's load on a real aircraft. A cambered wing
    pitches nose-down at every lift, and something has to hold the nose up:

        CL_t = 0  at  x_cg - x_w = -M_ac / (CL_target Sref)

    which for the published wing (NACA 2412, cm_ac = -0.054, Sref 10 m^2,
    mac 1.02 m, CL 0.5) sits 0.12 m AFT of the wing AC. A CG forward of
    that — i.e. any conventional loading — makes CL_t NEGATIVE: the
    textbook DOWNLOAD. With M_ac = 0 (a symmetric wing, or a caller with no
    section moment to report) the load is pure CG offset and the sign is
    just sign(x_cg - x_w), which is what this package computed before the
    couple was carried.

    Returns NaN where the balance says nothing (no area, no arm): a
    tailless-limit design has no trim surface to speak for.
    """
    if not (S_lift > 0.0) or l_t == x_w or Sref <= 0.0:
        return float("nan")
    return float((CL_target * Sref * (x_cg - x_w) + M_ac)
                 / (S_lift * (l_t - x_w)))


# ------------------------------------------------------------------ solves


@dataclass(frozen=True)
class _TailConfig:
    """Sampled geometry for one design point (internal)."""

    wing: geometry.Wing
    y_w: np.ndarray
    c_w: np.ndarray
    tw_w: np.ndarray          # wing twist [rad] at the wing stations
    S_t: float                # PANEL area (wetted-area bookkeeping)
    l_t: float                # SIGNED streamwise station (canard: negative)
    b_t: float
    y_t: np.ndarray
    c_t: np.ndarray
    coupling: Any
    tw_t: Any = None          # tail twist per station [rad] (0 = rectangle)
    wing_t: Any = None        # geometry.Wing of a DESIGNED tail, else None
    dist: float = 0.0         # distance from the wing AC (always >= 0)
    S_lift: float = 0.0       # equivalent horizontal area (V-tail: cos^2)
    dz: float = 0.0           # height above the wing plane


def empennage_surfaces(*, tail_type: str, S_t: float, b: float, S: float,
                       l_t: float, dz: float, AR_t=None, taper_t=None,
                       tc=None, fin_kwargs=None, fin: bool = True) -> tuple:
    """WHAT SURFACES THIS EMPENNAGE HAS, as weight-correlation inputs.

    ONE reader, for the same reason ``fin.fin_for_layout`` is one: the layout
    decides how many surfaces there are, and a second copy of that rule is
    how an aeroplane comes to be weighed as a conventional tail while it is
    drawn, flown and exported as a T-tail.

    * CONVENTIONAL / CANARD — a tailplane and a fin, weighed separately;
    * T-TAIL — the same two, with the fin flagged as CARRYING the tailplane
      (``tailplane_on_fin``): Raymer's vertical-tail equation charges 20 %
      for that load path, and it is the only place the connection costs
      anything in mass;
    * NO FIN — ``fin=False``, the answer stage 1 asks for. Weighs the
      tailplane alone, exactly as a V-tail weighs its panels alone. The fin
      is 82 % of this book on a sized family, so an aeroplane told to drop
      it and weighed with it anyway is not a rounding;
    * V-TAIL — the two canted panels and NO fin. The panels are given at
      their TRUE area (``S_t``), not the ``cos^2 gamma`` equivalent flat tail
      the aerodynamics uses: cant costs pitch effectiveness, it does not
      remove structure. The fin law returns None for this layout and nothing
      is charged for one, which is the mass half of what a V-tail is for.

    The fin is sized by the SAME law the drag book, the lattice, the report
    and the CAD export use (``fin.fin_for_layout``), so the surface weighed
    here is the surface flown. Its taper is 1.0 — the law draws a rectangle
    of area ``S_vt`` at aspect ratio ``AR_vt`` and has no taper to report,
    and Raymer's ``lam^0.039`` moves 2.7 % over the whole taper range, so
    naming the rectangle it actually draws beats inventing a number.

    Returns an EMPTY tuple for a non-positive tail area — a family with no
    second surface weighs no empennage, and the caller's fixed point then
    reproduces its published weight exactly.
    """
    from . import fin as _fin
    from .geometry import TC_DEFAULT
    from .weights import EmpennageSurface

    if not (float(S_t) > 0.0):
        return ()
    t = str(tail_type)
    tc_t = float(TC_DEFAULT if tc in (None, "") else tc)
    out = [EmpennageSurface(S=float(S_t),
                       AR=float(TAIL_AR if AR_t in (None, "") else AR_t),
                       taper=float(1.0 if taper_t in (None, "") else taper_t),
                       tc=tc_t, sweep_deg=0.0, vertical=False)]
    g = _fin.fin_for_layout(b=float(b), S=float(S), l_t=float(l_t),
                            tail_type=t, dz=float(dz), fin=bool(fin),
                            **(fin_kwargs or {}))
    if g is not None:
        out.append(EmpennageSurface(S=float(g.S), AR=float(g.AR), taper=1.0,
                               tc=float(g.tc), sweep_deg=0.0, vertical=True,
                               tailplane_on_fin=(t == "t_tail")))
    return tuple(out)


def tail_polar(prob: TailProblem, cl_trim: float | None = None):
    """The section the STABILISER flies, AND WHICH WAY UP.

    The section itself is its own when one was given, the wing's otherwise
    (the documented default — see ``TailProblem.polar_tail``).

    The ORIENTATION follows the load, the same way the tail's tip device
    does (``wingtail.tail_winglet_follow``): a surface that pushes DOWN
    flies its section INVERTED, because that is what mounting a cambered
    aerofoil upside down is for — its camber then works the side the
    surface is actually loaded on, instead of the trim incidence being
    cranked nose-down to fight it. Measured on the published tail: the
    stabiliser's profile drag falls 3.9 % and i_t goes -4.93 -> -0.31 deg,
    i.e. the surface stops flying sideways to its own shape.

    ``cl_trim`` is that load, signed. Pass it and the rule applies; leave it
    out (or pin ``TailProblem.tail_inverted``) and it does not:

    * ``tail_inverted=None`` (default) — FOLLOW the load, when one is given;
    * ``True`` / ``False`` — state it, and the load does not get a vote.

    A SYMMETRIC section is its own mirror, so the rule is a no-op there,
    which is why a symmetric stabiliser reproduces its numbers either way.
    """
    pol = prob.polar if prob.polar_tail is None else prob.polar_tail
    return orient_section(pol, cl_trim, prob.tail_inverted)


def orient_section(pol, cl_trim: float | None = None,
                   stated: bool | None = None):
    """A section as MOUNTED — the rule :func:`tail_polar` applies, alone.

    Split out because the three cores reach their section by three
    different routes (the LLT pair reads ``prob.polar``, the water family
    picks its own off a thickness family), and one rule shared beats three
    copies that can drift. ``stated`` wins when it is not None; otherwise
    the sign of ``cl_trim`` decides, and no load means no change.
    """
    want = stated
    if want is None:
        want = cl_trim is not None and float(cl_trim) < 0.0
    return _polar.inverted(pol) if want else pol


def stabiliser_load(cfg: "_TailConfig", prob: TailProblem) -> float:
    """The signed load the stabiliser carries — the ORIENTATION reference.

    :func:`trim_lift_coefficient` on this candidate's own geometry, with the
    WING's couple and not the stabiliser's. That exclusion is the whole
    point: mirroring a section mirrors its couple, so a rule that read the
    surface's own contribution would be letting the choice vote on itself,
    and near the crossing the two orientations disagree about which side of
    it they are on. The wing's couple is nine tenths of the total and is
    invariant under the choice, so deciding on it is both deterministic and
    the physically meaningful question: what does the REST of the aircraft
    ask this surface for?

    There is still a narrow band — a few thousandths in CL either side of
    the crossing — where the mirrored surface ends up marginally the other
    sign from the reading that mirrored it. That is real (the crossing is
    genuinely in a different place for the two orientations), it is small,
    and it is deterministic, which is what a design tool needs. The solvers
    and :func:`api.trim_surface_cl` apply this same rule, so the estimate
    and the run never disagree about which aeroplane is being flown.
    """
    M_ac_wing = (_polar.section_cm_ac(prob.polar)
                 * float(np.trapezoid(cfg.c_w ** 2, cfg.y_w)))
    return trim_lift_coefficient(prob.CL_target, prob.S, prob.x_cg,
                                 cfg.S_lift, cfg.l_t, M_ac=M_ac_wing)


def _config(x: np.ndarray, prob: TailProblem) -> _TailConfig:
    xs = np.asarray(x, dtype=float)
    n_base = 4 if prob.l_t_fixed is not None else 5
    vals = [float(v) for v in xs[:n_base]]
    if prob.l_t_fixed is not None:
        taper, twr, twt, S_t = vals
        dist = float(prob.l_t_fixed)
    else:
        taper, twr, twt, S_t, dist = vals
    m = int(prob.chord_order)
    if prob.tail_free and m:
        # two trailing chord blocks, wing then tail (tandem.py's convention)
        coeffs_w = geometry.ChordCoeffs(xs[-2 * m:-m], prob.chord_law)
        coeffs_t = geometry.ChordCoeffs(xs[-m:], prob.chord_law)
    else:
        coeffs_w = geometry.chord_coeffs_from_x(xs, m, prob.chord_law)
        coeffs_t = ()
    wing = geometry.Wing(
        b=prob.b, S=prob.S, taper=taper, twist_root_deg=twr,
        twist_tip_deg=twt, chord_limits=prob.chord_limits,
        chord_coeffs=coeffs_w)
    y_w, c_w, tw_w = wing.sample(prob.N)
    S_lift = prob.lifting_area(S_t)
    wing_t = None
    if prob.tail_free:
        taper_t, ar_t, washout_t = (float(v) for v in
                                    xs[n_base:n_base + 3])
        wing_t = geometry.Wing(
            b=float(np.sqrt(ar_t * S_lift)), S=S_lift, taper=taper_t,
            twist_root_deg=0.0, twist_tip_deg=washout_t,
            # the TAIL's chord limits are the tail's — see TailLimits.
            # This used to be handed ``prob.chord_limits``, i.e. the WING's
            # metres: a 0.5 m minimum chord meant for a 10 m wing refused
            # every stabiliser that was correctly smaller than it, and the
            # search came back all penalties for a limit the user had set on
            # a different surface.
            #
            # ...but that fix was written as ``chord_limits=None``, and it
            # threw out the two limits that are NOT in metres with the two
            # that are. ``trend`` ("the root is the largest chord") and
            # ``rate_max_deg`` (a local taper ANGLE) are both invariant under
            # a uniform rescale, so they mean exactly the same thing on a
            # 0.6 m stabiliser as on a 10 m wing — and the shell says so on
            # screen ("it holds on every designed surface"). The water twin
            # (hydrotail) and both tandems have always enforced them here;
            # only the air stabiliser did not, so a chord that grew outboard
            # was refused on the wing and flew on the tail.
            chord_limits=(None if prob.chord_limits is None else
                          replace(prob.chord_limits,
                                  c_min_m=None, c_max_m=None)),
            chord_coeffs=coeffs_t)
        b_t = wing_t.b
        y_t, c_t, tw_t = wing_t.sample(prob.N_t)
    else:
        b_t = tail_span(S_lift)
        y_t = cosine_stations(prob.N_t, b_t)[1]
        c_t = np.full(prob.N_t, S_lift / b_t)   # rectangular equivalent tail
        tw_t = np.zeros(prob.N_t)
    # ...and what the user said the surface may MEASURE, in metres. Checked
    # on the tail actually drawn — every station of it where the planform is
    # designed — because the box narrowing (TailLimits.narrow) is exact only
    # where the aspect ratio is fixed; with it free the box is the smallest
    # one CONTAINING the allowed set and these are the corners it added.
    # A V-tail's b_t/c_t describe the equivalent horizontal surface, which
    # is the same convention its area is stated in (lifting_area).
    if prob.tail_limits is not None:
        why = prob.tail_limits.violation(b_t, c_t)
        if why is not None:
            raise ValueError(why)
    x_t = prob.surface_x(dist)
    dz = prob.dz_for(dist)
    return _TailConfig(wing=wing, y_w=y_w, c_w=c_w, tw_w=tw_w, S_t=S_t,
                       l_t=x_t, b_t=b_t, y_t=y_t, c_t=c_t, tw_t=tw_t,
                       wing_t=wing_t,
                       coupling=prob.coupling(b_t, x_t, dz),
                       dist=dist, S_lift=S_lift, dz=dz)


def _solve(cfg: _TailConfig, prob: TailProblem,
           alpha: float, i_t: float) -> TandemResult:
    """One coupled solve at explicit (alpha, i_t) [rad]; Sref = wing area."""
    pol = prob.polar
    pol_t = tail_polar(prob, stabiliser_load(cfg, prob))
    front = Surface(b=prob.b, c=cfg.c_w, alpha_geo=alpha + cfg.tw_w,
                    a=pol.a_lin, alpha_L0=pol.alpha_L0, x=0.0, z=0.0)
    tw_t = 0.0 if cfg.tw_t is None else cfg.tw_t
    rear = Surface(b=cfg.b_t, c=cfg.c_t,
                   alpha_geo=(alpha + i_t) + tw_t,
                   a=pol_t.a_lin, alpha_L0=pol_t.alpha_L0,
                   x=cfg.l_t, z=cfg.dz)
    return solve_tandem(front, rear, V=prob.V, Sref=prob.S,
                        coupling=cfg.coupling)


def solve_tail_point(x: np.ndarray, alpha_rad: float, i_t_rad: float,
                     prob: TailProblem | None = None
                     ) -> tuple[float, float, TandemResult]:
    """(CL_total, Cm_cg, TandemResult) at explicit angles — the raw affine
    map that the trim elimination inverts.

    Reference/validation path: tests/test_tail.py drives this through a
    nested brentq (outer i_t on Cm = 0, inner alpha on CL = CL_target) and
    cross-checks the closed-form elimination against it to 1e-8. Raises on
    linear-algebra failure (evaluate_tail owns the failure contract)."""
    prob = prob or TailProblem()
    cfg = _config(np.asarray(x, dtype=float), prob)
    res = _solve(cfg, prob, float(alpha_rad), float(i_t_rad))
    cm = cm_about_cg(
        res, prob.x_cg, cfg.l_t, prob.S, cfg.wing.mac,
        M_ac=section_moment(cfg, prob.polar,
                            tail_polar(prob, stabiliser_load(cfg, prob))))
    return float(res.CL_total), cm, res


# ------------------------------------------------------------------ evaluate


def evaluate_tail(x: np.ndarray, prob: TailProblem | None = None,
                  moment_trim: bool = True) -> dict:
    """Full evaluation with breakdown; -100.0 contract on failure.

    ``moment_trim=True`` (default, the BO path): 2-D trim by elimination
    (module docstring) — 3 coupled solves for the exact affine map, a 2x2
    solve for (alpha, i_t), a 4th coupled solve at the trim point.

    ``moment_trim=False`` (documented physics-limit probe, NOT the BO
    path): i_t is held at 0 and only CL is trimmed (1-D affine, the tandem
    trick verbatim). This is the S_t -> 0 limit path: the moment trim
    demands CL_t ~ 1/S_t and diverges as the tail vanishes, so the
    zero-tail gate (LoD -> wing-only Tier A LoD) can only be checked with
    the moment equation released. In this mode ONLY, the S_t LOWER bound
    is relaxed to S_t > 0 so the probe can step below the BO box; the
    Cm_cg field reports the (nonzero) residual moment.
    """
    prob = prob or TailProblem()
    x = np.asarray(x, dtype=float)
    bnds = prob.bounds.copy()
    if not moment_trim:
        # S_t is index 3 in BOTH the 5-D and the fixed-arm 4-D vectors;
        # asserted rather than assumed, since the probe silently relaxing
        # the wrong variable would corrupt the S_t -> 0 validation gate.
        n_base = 4 if prob.l_t_fixed is not None else 5
        from .sizing import n_size_rows
        assert bnds.shape[0] == (n_base + prob.n_chord_rows
                                 + (2 if prob.flight_free else 0)
                                 + n_size_rows(prob.size_free)), bnds.shape
        bnds[3, 0] = 1e-9          # S_t -> 0 physics-limit probe (docstring)
    if x.shape != (bnds.shape[0],):
        return _fail(f"design vector shape {x.shape} != ({bnds.shape[0]},)")
    if np.any(x < bnds[:, 0] - 1e-12) or np.any(x > bnds[:, 1] + 1e-12):
        return _fail("bounds violation")

    # ---- size (sizing.py's modifier): the candidate span and area replace
    # the problem's own before anything is built. The coupling cache is
    # keyed on (S_lift, x_t, dz) and NOT on the wing span, so a resized copy
    # must start with a fresh one or it would reuse another wing's operators.
    size_req = bool(prob.size_free)
    W_fixed_ref = None
    if size_req:
        from .mission import weight_for as _weight_for0
        from .sizing import (WS_MODES, check_ar, resolve_size,
                             size_mode)
        # the payload weight is the SAME CONSTANT for every candidate — that
        # is what makes payload L/D = W_fixed/D a minimum-drag objective
        # (sizing.py). Read it from the problem's OWN published area, before
        # the candidate size replaces it: taken afterwards it scaled with the
        # area, so a bigger wing carried a bigger "fixed" payload and the
        # score rewarded growing the wing.
        W_fixed_ref = (prob.W_fixed_N if prob.W_fixed_N is not None
                       else _weight_for0(prob.CL_target, prob.V, prob.S))
        kw = {"wing_loading_max_Pa": prob.wing_loading_max_Pa}
        if size_mode(prob.size_free) in WS_MODES:
            # the WING's area follows the mission's wing loading (the tail's
            # own area stays a design variable): read the flow state first —
            # it needs no area — and close the area's fixed point against it
            from .sizing import flow_state_for
            rho_ws, V_ws = flow_state_for(
                x, mission=prob.mission, flight_free=prob.flight_free,
                n_trailing=prob.n_chord_rows, rho=prob.rho, V=prob.V)
            kw.update(
                # ignored by the SEARCHED-loading mode, which reads its own
                # row out of the vector (sizing.resolve_spans)
                wing_loading_Pa=(prob.wing_loading_Pa
                                 if prob.wing_loading_Pa is not None
                                 else prob.CL_target * 0.5 * prob.rho
                                 * prob.V ** 2),
                W_fixed_N=W_fixed_ref,
                taper=float(x[0]), tc=geometry.TC_DEFAULT,
                q_Pa=0.5 * rho_ws * V_ws ** 2)
        try:
            b_use, S_use = resolve_size(x, prob.size_free, prob.flight_free,
                                        prob.n_chord_rows, **kw)
        except (ValueError, OverflowError) as exc:
            return _fail(f"size: {exc}")
        reason = check_ar(b_use, S_use)
        if reason is not None:
            return _fail(f"size: {reason}")
    # ...and the ASPECT-RATIO LIMIT THE USER SET. A different question
    # from the validity band, so it is asked whether or not the size is
    # a design variable: a limit that only applied to searched sizes
    # would go quiet exactly when the user typed the number it is about.
    if prob.ar_limits is not None:
        from .sizing import check_ar_limit as _check_ar_limit
        why_ar = _check_ar_limit(b_use if size_req else float(prob.b),
                                 S_use if size_req else float(prob.S),
                                 prob.ar_limits)
        if why_ar is not None:
            return _fail(f"size: {why_ar}")
    if prob.size_free:
        prob = replace(prob, b=b_use, S=S_use, size_free=False,
                       _coupling_cache={})

    # ---- flight state (geometry.py's flight modifier): the pair ahead of
    # the chord block rebuilds the flow state and the trim target
    flight = geometry.flight_from_x(x, prob.flight_free, prob.n_chord_rows)
    if flight is not None:
        from .mission import flight_state
        try:
            prob = replace(prob, flight_free=False, mission=None,
                           **flight_state(prob.mission, flight[0], flight[1],
                                          prob.S))
        except ValueError as exc:      # water medium, or ISA validity
            return _fail(f"flight state: {exc}")

    pol = prob.polar
    try:
        cfg = _config(x, prob)
    except ValueError as exc:          # collapsed chord law (geometry.Wing)
        return _fail(f"planform: {exc}")
    # which way up the stabiliser's section flies — decided here, once, from
    # the load the balance already implies (tail_polar / stabiliser_load),
    # so every solve below and the drag integration share one orientation
    cl_stab = stabiliser_load(cfg, prob)
    pol_t = tail_polar(prob, cl_stab)
    mac = cfg.wing.mac

    sized = None
    CL_trim = prob.CL_target
    if size_req:
        from .sizing import sized_state
        try:
            sized = sized_state(
                W_fixed_N=W_fixed_ref,
                b=prob.b, S=prob.S, taper=cfg.wing.taper, tc=cfg.wing.tc,
                q_Pa=0.5 * prob.rho * prob.V**2,
                c_root=float(cfg.wing.chord(np.array([0.0]))[0]),
                # THE EMPENNAGE WEIGHS SOMETHING, and what it weighs depends
                # on the layout: a V-tail carries no fin and a T-tail's fin
                # carries the tailplane. Before this the tail's mass sat
                # inside W_fixed as a constant, so sizing it moved its drag
                # and its trim but never its weight.
                empennage=empennage_surfaces(
                    tail_type=prob.tail_type, S_t=cfg.S_t, b=prob.b,
                    S=prob.S, l_t=cfg.dist, dz=cfg.dz,
                    AR_t=(cfg.wing_t.AR if cfg.wing_t is not None else None),
                    taper_t=(cfg.wing_t.taper if cfg.wing_t is not None
                             else None),
                    tc=cfg.wing.tc, fin=_fin.has_fin(prob),
                    fin_kwargs=_fin.fin_law_kwargs(prob)),
                wing_loading_max_Pa=prob.wing_loading_max_Pa,
                material=prob.material)
        except (ValueError, RuntimeError, OverflowError) as exc:
            return _fail(f"size: {exc}")
        CL_trim = sized.CL_target
        prob = replace(prob, CL_target=CL_trim)

    # the sections' own couple: a CONSTANT in (alpha, i_t), so it shifts
    # r0 of the affine map and nothing else — the Jacobian, the neutral
    # point and the static margin are all alpha-derivatives and do not see
    # it (gated in tests/test_tail_download.py)
    M_ac = section_moment(cfg, pol, pol_t)

    def _clcm(res: TandemResult) -> tuple[float, float]:
        return res.CL_total, cm_about_cg(res, prob.x_cg, cfg.l_t, prob.S,
                                         mac, M_ac=M_ac)

    try:
        d = np.deg2rad(1.0)
        r00 = _solve(cfg, prob, 0.0, 0.0)
        ra = _solve(cfg, prob, d, 0.0)
        CL0, Cm0 = _clcm(r00)
        CLa, Cma = _clcm(ra)

        # coupled alpha-derivatives (exact FD on a linear system) -> SM
        a_w = (ra.front.CL * ra.front.S - r00.front.CL * r00.front.S) / d
        a_t = (ra.rear.CL * ra.rear.S - r00.rear.CL * r00.rear.S) / d
        if not (np.isfinite(a_w) and np.isfinite(a_t)) or (a_w + a_t) <= 0.0:
            return _fail("degenerate system lift-curve slope")
        x_np = (0.0 * a_w + cfg.l_t * a_t) / (a_w + a_t)
        SM = (x_np - prob.x_cg) / mac
        g = (SM - prob.SM_min) / 1.0

        dCL_da = (CLa - CL0) / d
        dCm_da = (Cma - Cm0) / d
        if moment_trim:
            ri = _solve(cfg, prob, 0.0, d)
            CLi, Cmi = _clcm(ri)
            J = np.array([[dCL_da, (CLi - CL0) / d],
                          [dCm_da, (Cmi - Cm0) / d]])
            if abs(np.linalg.det(J)) < 1e-12:
                return _fail("degenerate trim Jacobian")
            sol = np.linalg.solve(J, np.array([prob.CL_target - CL0, -Cm0]))
            alpha, i_t = float(sol[0]), float(sol[1])
        else:
            if abs(dCL_da) < 1e-9:
                return _fail("degenerate CL_alpha")
            alpha, i_t = float((prob.CL_target - CL0) / dCL_da), 0.0

        lo, hi = (np.deg2rad(prob.alpha_bracket_deg[0]),
                  np.deg2rad(prob.alpha_bracket_deg[1]))
        if not (lo <= alpha <= hi):
            return _fail("untrimmable: alpha outside bracket",
                         alpha_deg=float(np.rad2deg(alpha)))
        if abs(i_t) > np.deg2rad(prob.i_t_max_deg):
            return _fail("untrimmable: |i_t| exceeds limit",
                         i_t_deg=float(np.rad2deg(i_t)))
        # A hinged elevator reaches the same trim as an all-moving surface
        # with a LARGER deflection, delta_e = i_t / tau. The physics and the
        # L/D are identical (a linear flap enters the RHS as the same
        # uniform shift) — what changes is which designs are reachable, so
        # this is a gate and nothing else.
        delta_e = float(i_t) / prob.tau_elevator
        if abs(delta_e) > np.deg2rad(prob.delta_e_max_deg):
            return _fail("untrimmable: |elevator deflection| exceeds limit",
                         delta_e_deg=float(np.rad2deg(delta_e)),
                         i_t_deg=float(np.rad2deg(i_t)))

        res = _solve(cfg, prob, alpha, i_t)      # 4th solve (affinity gate)
    except np.linalg.LinAlgError as exc:
        return _fail(f"solver failure: {exc}")

    CL_fin, Cm_fin = _clcm(res)
    cl_residual = CL_fin - prob.CL_target

    # stall / extrapolation proxy on BOTH surfaces (tandem.py convention),
    # each against ITS OWN polar's validity range — with one section they are
    # the same range, so this is bit-for-bit the previous test
    ae_w = np.rad2deg(res.front.alpha_eff_y)
    ae_t = np.rad2deg(res.rear.alpha_eff_y)
    lo_p, hi_p = pol.alpha_valid
    lo_t, hi_t = pol_t.alpha_valid
    if (ae_w.min() < lo_p or ae_w.max() > hi_p
            or ae_t.min() < lo_t or ae_t.max() > hi_t):
        return _fail(
            "effective AoA outside polar validity (stall/extrapolation proxy)",
            alpha_eff_range=(float(min(ae_w.min(), ae_t.min())),
                             float(max(ae_w.max(), ae_t.max()))),
        )

    # profile drag: wing strips AND tail strips, each on its own section
    # polar (the tail reuses the wing's unless one was chosen for it).
    # The tail term is charged on the PANEL area, not the equivalent
    # horizontal area: a V-tail's dihedral costs pitch effectiveness but
    # wets exactly as much surface as before. Charging cos^2 twice would
    # hand the optimiser a free lunch of the same order (~2.6 % of L/D at
    # 45 deg) as the effect being studied. wet_ratio == 1.0 exactly for
    # every non-V layout, so this is bit-for-bit for them.
    wet_ratio = cfg.S_t / cfg.S_lift if cfg.S_lift > 0.0 else 1.0
    CDp_w = float(np.trapezoid(pol.cd(ae_w) * cfg.c_w, cfg.y_w) / res.front.S)
    CDp_t = float(np.trapezoid(pol_t.cd(ae_t) * cfg.c_t, cfg.y_t)
                  / res.rear.S)
    CDp = (CDp_w * res.front.S + CDp_t * res.rear.S * wet_ratio) / prob.S
    # FIN PARASITE DRAG, charged the way the TAILPLANE's is: always. It was
    # opt-in and off, so the shipped design flew a surface it did not pay
    # for — worth +5.5 % L/D at the shipped tail family's box centre
    # (cd0_fin 0 -> 0.000885, L/D 32.696 -> 30.907). An allowance that big,
    # reachable only through a switch, is not a choice a user should have to
    # find: the horizontal tail is not optional in the drag book and neither
    # is this one. Design-dependent, and it scales as 1/arm.
    #
    # A V-TAIL still charges none, because it HAS none — that is the whole
    # raison d'être of the layout, and the comparison it exists to win is
    # only meaningful now that the others pay. AN AEROPLANE TOLD TO CARRY NO
    # FIN charges none for the same reason arrived at the other way
    # (``fin.has_fin``): the answer stage 1 asks for used to reach nothing at
    # all, so a design with the fin switched off still paid this 0.000885 —
    # 5.79 % of its L/D on this family's box centre.
    cd0_fin = 0.0
    if _finmod.has_fin(prob):
        cd0_fin = vtail_cd0(prob.S, cfg.dist, b=prob.b, S=prob.S,
                            V=prob.V, rho=prob.rho, mu=prob.mu,
                            **_finmod.fin_drag_kwargs(prob))
    # ...and the FUSELAGE the arm implies, where a diameter was stated. This
    # is the term that stops a longer arm being free; see
    # TailProblem.fuselage_diameter_m for the measurement.
    cd0_fus, fus_L, fus_fr = _fuselage_charge(prob, cfg.dist)
    CD = res.CDi_total + CDp + prob.cd0_extra + cd0_fin + cd0_fus
    LoD = prob.CL_target / CD
    if not (np.isfinite(LoD) and np.isfinite(g)):
        return _fail("non-finite L/D or SM margin")

    sized_extra: dict = {}
    if sized is not None:
        # payload L/D and the SECOND constraint (sizing.py)
        q = 0.5 * prob.rho * prob.V**2
        f = sized.payload_lod(q, CD)
        if not np.isfinite(f):
            return _fail("non-finite payload L/D")
        sized_extra = {**sized.report(), "f": float(f), "g_sm": float(g),
                       "D_N": float(q * sized.S * CD)}

    return {
        **({"V": float(prob.V), "altitude_m": float(flight[1]),
            "rho": float(prob.rho), "mu": float(prob.mu)}
           if flight is not None else {}),
        **sized_extra,
        "CL_target": float(prob.CL_target),
        "feasible": True, "reason": "",
        "score": float(LoD if sized is None else sized_extra["f"]),
        "LoD": float(LoD),
        "g": (float(g) if sized is None
              else np.array([float(g), float(sized.g_sigma)])),
        "SM": float(SM), "SM_min": prob.SM_min,
        "x_np": float(x_np), "x_cg": prob.x_cg,
        "CL_total": float(CL_fin), "CL_w": res.front.CL, "CL_t": res.rear.CL,
        "lift_share_tail": float(res.rear.CL * res.rear.S
                                 / (CL_fin * prob.S)),
        "alpha_rad": float(alpha), "i_t_rad": float(i_t),
        "alpha_deg": float(np.rad2deg(alpha)),
        "i_t_deg": float(np.rad2deg(i_t)),
        "Cm_cg": float(Cm_fin), "cm_residual": float(Cm_fin),
        # the sections' own couple, both ways round: as M/q [m^3] and as the
        # Cm it contributes on (Sref, mac). Reported because it is what sets
        # the SIGN of CL_t, and a reader comparing a download against a CG
        # alone will not otherwise find where it came from.
        "M_ac_m3": float(M_ac), "Cm_ac": float(M_ac / (prob.S * mac)),
        # which way up the stabiliser's section is mounted, and why — a
        # surface that pushes down flies its camber the other way, and the
        # geometry views and the CAD export have to be told
        "tail_section_inverted": bool(
            isinstance(pol_t, _polar.InvertedPolar)),
        "tail_section_follows_load": prob.tail_inverted is None,
        "cl_residual": float(cl_residual),
        "CL_alpha": float(dCL_da),
        "CLa_wing": float(a_w / prob.S), "CLa_tail": float(a_t / prob.S),
        "CDi_total": res.CDi_total,
        "CDi_self_wing": res.front.CDi, "CDi_self_tail": res.rear.CDi,
        "CDi_mut": res.CDi_mut, "CDi_mut_wing": res.CDi_mut_front,
        "CDi_mut_tail": res.CDi_mut_rear,
        "CDp": float(CDp), "CDp_wing": CDp_w, "CDp_tail": CDp_t,
        "cd0_extra": prob.cd0_extra, "cd0_fin": float(cd0_fin),
        # the fuselage: its drag, the length it was built on and the FINENESS
        # RATIO, which is the number that says "not an aeroplane" long before
        # a drag count does (a 40 m arm asks for a 183:1 body)
        "cd0_fus": float(cd0_fus),
        "fuselage_length_m": float(fus_L),
        "fuselage_fineness": float(fus_fr),
        "fuselage_diameter_m": (None if prob.fuselage_diameter_m is None
                                else float(prob.fuselage_diameter_m)),
        "CD": float(CD),
        "eps_tail_mean_deg": float(np.rad2deg(np.mean(res.eps_rear))),
        "tail_type": prob.tail_type,
        "dihedral_deg": float(prob.dihedral_deg),
        "S_lift": float(cfg.S_lift), "dz_tail": float(cfg.dz),
        "dist_from_wing_m": float(cfg.dist),
        "control": prob.control,
        "tau_elevator": float(prob.tau_elevator),
        "delta_e_deg": float(np.rad2deg(i_t / prob.tau_elevator)),
        "S_t": cfg.S_t, "l_t": cfg.l_t, "b_t": cfg.b_t, "mac": float(mac),
        "moment_trim": bool(moment_trim),
        "polar": getattr(pol, "name", "unknown"),
        "polar_tail": getattr(pol_t, "name", "unknown"),
        "tandem": res,
    }


def fg_tail(x: np.ndarray, prob: TailProblem | None = None):
    """Constrained-harness callable: (L/D at trim, signed SM margin g).

    Solvable-but-marginally-stable designs return their TRUE LoD with
    g < 0 (constrained BO needs the violation magnitude); solver failures
    and untrimmable designs return exactly (PENALTY, G_FAIL) — the
    hydrofoil.fg_hydrofoil / aircraft.fg_aircraft contract.

    With the SIZE modifier on, the objective is payload L/D and the margin
    is the PAIR [static margin, root-bending stress] — both changes are part
    of that variant's identity (sizing.py)."""
    sized = prob is not None and prob.size_free
    out = evaluate_tail(x, prob)
    if not out["feasible"]:
        return (PENALTY, np.array([G_FAIL, G_FAIL])) if sized else (PENALTY,
                                                                    G_FAIL)
    return float(out["score"]), (out["g"] if sized else float(out["g"]))


# ------------------------------------------------------------------ V-tail


def _fuselage_charge(prob, l_t: float) -> tuple:
    """``(cd0_fus, L_body, fineness)`` for a stated fuselage, else all zero.

    The body's length comes from ``drag.body_length_for_arm`` — one constant,
    shared with the inertia length ``flightmodel`` builds — and its drag from
    ``drag.fuselage_cd0``, the Raymer body this package has carried unused
    since the beginning. Nothing is charged until a DIAMETER is stated, so an
    unset problem is the published one bit-for-bit.

    A body no longer than it is wide is not a body; that is refused rather
    than returned, because ``fuselage_cd0``'s form factor ``1 + 60/fr^3``
    diverges as the fineness ratio goes to zero and would otherwise report a
    colossal drag instead of a bad input.
    """
    D = getattr(prob, "fuselage_diameter_m", None)
    if D is None:
        return 0.0, 0.0, 0.0
    D = float(D)
    if not D > 0.0:
        raise ValueError(
            f"fuselage diameter must be > 0 m (got {D!r}); leave it unset to "
            f"charge no fuselage at all")
    L = _drag.body_length_for_arm(l_t)
    fr = L / D
    if fr <= 1.0:
        raise ValueError(
            f"a {L:.3g} m body of {D:.3g} m diameter has a fineness ratio of "
            f"{fr:.3g}: that is not a fuselage. Shorten the diameter or "
            f"lengthen the arm")
    comp = _drag.fuselage_cd0(prob.S, L, D, prob.rho, prob.V, prob.mu)
    return float(comp.CD0), float(L), float(fr)


def vtail_cd0(
    S_ref: float,
    l_t: float,
    b: float = 10.0,
    S: float | None = None,
    V_v: float = 0.04,
    tc: float = 0.10,
    AR_vt: float = 1.5,
    V: float = 14.6,
    rho: float = RHO_SL,
    mu: float = MU_SL,
    Q: float = 1.05,
) -> float:
    """Vertical-tail parasite-drag increment referenced to ``S_ref``.

    Volume-coefficient sizing (Raymer, *Aircraft Design: A Conceptual
    Approach*, Table 6.4: c_VT ~ 0.04 for single-engine GA; the VERTICAL
    tail volume uses wing SPAN, not mac):

        S_vt = V_v * b * S / l_t

    (S defaults to S_ref — the usual case where the drag reference IS the
    wing area). Drag build-up reuses drag.py's Raymer pieces: turbulent
    flat-plate Cf (skin_friction_cf, eq. 12.27/12.28) at the fin's mean
    chord c_vt = sqrt(S_vt / AR_vt) (AR_vt = 1.5, conventional fin range
    1.3-2), Raymer wing/tail form factor at t/c = 0.10
    (wing_form_factor, eq. 12.30), tail interference Q = 1.05
    (Raymer Sec. 12.5), and a flat-plate 2-SIDED wetted area
    Swet = 2 S_vt (1.3 % below Raymer's 1.977 + 0.52 t/c panel formula
    at t/c = 0.10 — documented simplification per the phase spec):

        cd0_vt = Cf * FF * Q * (2 S_vt) / S_ref.

    Scales ~ 1/l_t through S_vt (a longer arm needs less fin area), with
    a weak opposing Cf(Re(c_vt)) drift — the bookkeeping that would make
    long tail arms cheaper in a directional-stability-aware objective.
    Default magnitude ~ 9e-4 at l_t = 5.5 m (test-gated 0.0005-0.005).
    Pure bookkeeping helper: NOT added to evaluate_tail by default — hook
    it in via TailProblem.cd0_extra (drag.py double-count guard style).
    """
    if S is None:
        S = S_ref
    # ONE AUTHOR. The sizing law lives in fin.py, which is also what the
    # flight rebuild and the CAD export ask — before that this function and
    # ``flightmodel`` sized two different fins for the same aeroplane, and
    # only one of them was ever charged.
    geo = _fin.size_fin(b=b, S=S, l_t=l_t, V_v=V_v, AR=AR_vt, tc=tc)
    S_vt = geo.S
    c_vt = geo.chord
    Re = rho * V * c_vt / mu
    cf = skin_friction_cf(Re, lref=c_vt)
    ff = wing_form_factor(tc)
    Swet = 2.0 * S_vt
    return float(cf * ff * Q * Swet / S_ref)
