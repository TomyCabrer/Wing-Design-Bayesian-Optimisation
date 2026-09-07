"""Structural closure for the aircraft problem: wing weight + root spar stress.

Tier A optimised a wing of FIXED weight — nothing stopped span from being
whatever the box bounds allowed. This module supplies the two reduced-order
structural models that close the aero-structural loop for aircraft.py:

1. ``wing_weight_raymer`` — a statistical general-aviation wing-weight
   correlation (Raymer). It couples the planform design variables
   (b, S, taper, sweep, t/c) and the ultimate load factor to the weight
   the wing must LIFT, creating the classic span/weight trade: more span
   cuts induced drag but grows the structure the aircraft has to carry.
   ``total_weight`` closes the W_wing(W_total) circularity by fixed point.
2. ``root_bending_stress`` — a rectangular-spar-cap estimate of the wing
   root bending stress, feeding the strength constraint
   sigma <= SIGMA_ALLOW_PA that stops the optimiser cashing in unbounded
   span (stress grows ~ b^4 at fixed S, see the function docstring).

Unit convention (the module's #1 trap)
--------------------------------------
ALL public interfaces are SI: metres, m^2, newtons, pascals; return values
are newtons (weights are FORCES throughout the framework — mission.py
carries W_N, L = W in trim) and pascals. Raymer's correlation is IMPERIAL
inside (ft^2, lb, psf); the conversion happens exactly once, through the
named constants below (international foot 0.3048 m and pound-force
4.4482216152605 N, both exact by definition).

Sign/geometry conventions (match geometry.py)
---------------------------------------------
``sweep_deg`` is the quarter-chord sweep angle in degrees, aft positive;
``taper`` = c_tip/c_root; ``tc`` = section thickness/chord;
c_root = 2 S / (b (1 + taper)) exactly as geometry.Wing.c_root.

Tier-A honesty labels
---------------------
* The Raymer correlation is CALIBRATED FOR GA-CLASS AIRCRAFT (unpressurised
  singles/light twins, W_dg ~ 1-8 klb). Use outside that range is smooth
  extrapolation of a power law, not physics — flagged, and acceptable at
  Tier level because the optimiser only needs a monotone, correctly-signed
  weight gradient.
* The spar model is a REDUCED-ORDER two-cap estimate (no shear web, no
  inertial relief from the wing's own mass, no taper of the cap area along
  the span). It is a constraint SHAPE, calibrated at one reference design
  (see K_CAP), not a stress prediction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mission import G0  # 9.80665 m/s^2 — single source of truth (mission.py)

__all__ = [
    "G0",
    "N_ULT_DEFAULT",
    "SIGMA_ALLOW_PA",
    "K_CAP",
    "H_BOX_FRAC",
    "Q_PA_DEFAULT",
    "wing_weight_raymer",
    "EmpennageSurface",
    "tail_weight_raymer",
    "empennage_weight",
    "total_weight",
    "root_bending_stress",
]

# ---------------------------------------------------------------- load factors

N_ULT_DEFAULT = 5.7      # ultimate load factor: 3.8 limit x 1.5 (GA / FAR-23
#                          normal category; Raymer's N_z is the ULTIMATE factor)

# ---------------------------------------------------------------- unit constants
# Exact by definition: international foot and avoirdupois pound-force.

M_PER_FT = 0.3048                      # m/ft (exact)
FT_PER_M = 1.0 / M_PER_FT              # ft/m
FT2_PER_M2 = FT_PER_M**2               # ft^2/m^2
N_PER_LBF = 4.4482216152605            # N/lbf (exact)
LBF_PER_N = 1.0 / N_PER_LBF            # lbf/N
PA_PER_PSF = N_PER_LBF / M_PER_FT**2   # = 47.88025898... Pa per lbf/ft^2
PSF_PER_PA = 1.0 / PA_PER_PSF          # psf/Pa

# Default cruise dynamic pressure for the q^0.006 term: GA cruise ~50 m/s
# (~97 kt) at sea level, q = 1/2 * 1.225 * 50^2 = 1531.25 Pa (~32 psf).
# The exponent 0.006 makes W_wing insensitive to this choice: over the whole
# plausible band q in [100, 5000] Pa the factor moves by (50)^0.006 ~ 2.4%.
# aircraft.py passes the actual mission q explicitly; this is the fallback.
Q_PA_DEFAULT = 0.5 * 1.225 * 50.0**2   # Pa

# ---------------------------------------------------------------- material

# Allowable stress: aluminium 2024-T3 sheet (the classic GA spar-cap alloy),
# tensile yield ~345 MPa (typical handbook/MMPDS value), static safety
# factor 1.5 (FAR 23.303) applied to yield:
SIGMA_ALLOW_PA = 345e6 / 1.5           # = 230 MPa allowable

# ---------------------------------------------------------------- spar sizing

H_BOX_FRAC = 0.85   # spar-box depth / max section thickness (typical wing box)
K_CAP = 0.0025      # spar-cap area A_cap = K_CAP * tc * c_root^2 — calibrated
#                     so the reference design (b = 11 m, S = 16 m^2,
#                     taper 0.45, tc 0.12, n_ult 5.7, W_total = 9.5 kN) sits
#                     NEAR the allowable: sigma_ref = 255.8 MPa
#                     = 1.11 x SIGMA_ALLOW_PA (see root_bending_stress).


# ---------------------------------------------------------------- wing weight


def wing_weight_raymer(
    b: float,
    S: float,
    taper: float,
    sweep_deg: float,
    tc: float,
    n_ult: float,
    W_dg_N: float,
    *,
    W_fw_N: float = 0.0,
    q_Pa: float = Q_PA_DEFAULT,
    k_w: float = 1.0,
) -> float:
    """Statistical GA wing weight [N] (Raymer general-aviation correlation).

    ``k_w`` is the MATERIAL weight factor (:mod:`aerobo.materials`). The
    correlation was regressed over aluminium GA aeroplanes and has no
    material argument of its own, so a material enters here and only here —
    as a multiplier, with 1.0 meaning "built of what Raymer's aeroplanes were
    built of" (2024-T3). Read ``materials`` before trusting a value other
    than 1.0: the factor carries buckling, minimum gauge and joints as a
    single fitted number, and the correlation's exponents stay aluminium's.

    Raymer, *Aircraft Design: A Conceptual Approach*, 5th ed., AIAA
    Education Series, 2012, Ch. 15 (statistical group weights method),
    general-aviation wing weight — eq. 15.46 in the 4th/5th-edition
    numbering:

        W_wing [lb] = 0.036 * S_w^0.758 * W_fw^0.0035
                      * (A / cos^2 L)^0.6 * q^0.006 * lam^0.04
                      * (100 t/c / cos L)^-0.3 * (N_z W_dg)^0.49

    with S_w exposed planform area [ft^2], W_fw fuel weight in the wing
    [lb], A aspect ratio, L quarter-chord sweep, q cruise dynamic pressure
    [psf], lam taper ratio, t/c thickness ratio, N_z ULTIMATE load factor,
    W_dg design gross weight [lb].

    Interface is SI (b [m], S [m^2], weights [N], q_Pa [Pa]); the
    SI -> imperial -> SI conversion is explicit through the module's named
    constants (FT2_PER_M2, LBF_PER_N, PSF_PER_PA, N_PER_LBF) and happens
    nowhere else.

    W_fw = 0 guard: the correlation's W_fw^0.0035 factor is a statistical
    nudge that equals 1.016 for W_fw = 100 lb and 1.024 for 1000 lb — but
    evaluates to 0^0.0035 = 0 for a dry wing, killing the whole product.
    For W_fw_N <= 0 the factor is therefore taken as EXACTLY 1.0 (the
    correct no-fuel-in-wing limit of a ~1.0 factor), keeping W_wing finite
    and equal to the physical no-fuel limit for the default dry-wing
    problem (the factor jumps from W_fw^0.0035 to 1.0 AT zero — fine while
    W_fw is never a design variable; clamp with a floor if it ever is).

    Honesty label: calibrated for GA-class aircraft (unpressurised
    singles/light twins); outside that class this is smooth power-law
    extrapolation, not physics. Signs (asserted in tests/test_weights.py):
    W_wing UP in b (A^0.6), UP in n_ult ((N_z W_dg)^0.49), DOWN in tc
    (exponent -0.3 — in the statistical record, thicker sections carry
    bending with less cap material), UP in sweep (net cos^-0.9).
    """
    if b <= 0.0 or S <= 0.0:
        raise ValueError(f"b and S must be > 0 (got b={b}, S={S})")
    if taper <= 0.0 or tc <= 0.0:
        raise ValueError(
            f"taper and tc must be > 0 (got taper={taper}, tc={tc}) — "
            "taper^0.04 and (100 tc)^-0.3 degenerate at 0"
        )
    if not -90.0 < sweep_deg < 90.0:
        raise ValueError(f"quarter-chord sweep {sweep_deg} deg outside (-90, 90)")
    if n_ult <= 0.0 or W_dg_N < 0.0 or q_Pa <= 0.0:
        raise ValueError(
            f"need n_ult > 0, W_dg_N >= 0, q_Pa > 0 "
            f"(got {n_ult}, {W_dg_N}, {q_Pa})"
        )

    # SI -> imperial (the ONLY place units change)
    S_ft2 = S * FT2_PER_M2
    W_dg_lb = W_dg_N * LBF_PER_N
    W_fw_lb = W_fw_N * LBF_PER_N
    q_psf = q_Pa * PSF_PER_PA

    A = b**2 / S                                   # aspect ratio (unitless)
    cosL = float(np.cos(np.deg2rad(sweep_deg)))
    fw_factor = W_fw_lb**0.0035 if W_fw_lb > 0.0 else 1.0   # guard: see docstring

    W_wing_lb = (
        0.036
        * S_ft2**0.758
        * fw_factor
        * (A / cosL**2) ** 0.6
        * q_psf**0.006
        * taper**0.04
        * (100.0 * tc / cosL) ** -0.3
        * (n_ult * W_dg_lb) ** 0.49
    )
    if not (k_w > 0.0):
        raise ValueError(f"material weight factor must be > 0 (got {k_w})")
    return float(k_w) * W_wing_lb * N_PER_LBF      # imperial -> SI [N]


# ---------------------------------------------------------------- empennage


@dataclass(frozen=True)
class EmpennageSurface:
    """ONE empennage surface, as the weight correlation asks for it.

    Named apart from ``vlm.TailSurface``, which is the same surface's PANELS
    in the lattice: this one is a structural description and carries no
    geometry the solver could fly.

    The whole point of naming a type here is that the LAYOUT decides which
    surfaces exist, and the layout is a mission answer:

    * a CONVENTIONAL tail is one horizontal surface and one vertical one;
    * a T-TAIL is the same two, connected — the tailplane stands on the fin's
      tip, so the fin carries it and ``tailplane_on_fin`` turns Raymer's
      ``1 + 0.2 H_t/H_v`` term on (H_t/H_v = 1). It is the ONLY place that
      connection costs anything in mass, and it is why a T-tail is heavier
      than a conventional tail of the same areas;
    * a V-TAIL is two panels and NO vertical surface at all: both panels are
      given as horizontal surfaces at their true (uncanted) area, and no
      vertical term is charged, because there is no fin to charge.

    ``S`` is the surface's own exposed area [m2] — the TRUE panel area for a
    V-tail, not the ``cos^2 gamma`` equivalent flat tail the aerodynamics
    uses: dihedral costs pitch effectiveness, it does not remove structure.
    """

    S: float                    # [m2] the surface's own area
    AR: float                   # its aspect ratio (h^2/S for a fin)
    taper: float = 1.0          # c_tip / c_root
    tc: float = 0.10            # thickness ratio of ITS section
    sweep_deg: float = 0.0      # quarter-chord sweep, aft positive
    vertical: bool = False      # a fin (else a tailplane / V panel)
    tailplane_on_fin: bool = False    # T-tail: H_t/H_v = 1 (vertical only)


def tail_weight_raymer(surf: EmpennageSurface, n_ult: float, W_dg_N: float, *,
                       q_Pa: float = Q_PA_DEFAULT, k_w: float = 1.0) -> float:
    """Statistical GA empennage weight [N] for ONE surface (Raymer).

    Raymer, *Aircraft Design: A Conceptual Approach*, Ch. 15, the
    general-aviation group weights — the horizontal and vertical tail
    equations that sit beside the wing one this module already carries:

        W_ht [lb] = 0.016 (N_z W_dg)^0.414 q^0.168 S_ht^0.896
                    (100 t/c / cos L)^-0.12 (A / cos^2 L)^0.043 lam^-0.02

        W_vt [lb] = 0.073 (1 + 0.2 H_t/H_v) (N_z W_dg)^0.376 q^0.122
                    S_vt^0.873 (100 t/c / cos L)^-0.49
                    (A / cos^2 L)^0.357 lam^0.039

    with the same units and the same conversion constants as
    :func:`wing_weight_raymer` (ft2, lb, psf), and the same honesty label:
    calibrated for GA-class aeroplanes, a smooth power-law extrapolation
    outside that class.

    ``H_t / H_v`` is the tailplane's height as a fraction of the fin's, so it
    is 1 for a T-TAIL (the tailplane sits on the fin's tip) and 0 for every
    layout that hangs the tailplane off the body. That 20 % is the mass
    penalty for the connection — the fin becomes a load path for the surface
    above it — and it is the only term in this module that knows about the
    empennage layout at all.

    ``k_w`` is the material factor (:mod:`aerobo.materials`), applied for the
    same reason it is applied to the wing: the correlation was regressed over
    aluminium aeroplanes and a composite empennage is not one. It is the
    WING's material — this package asks the question once — so a wing and a
    tail built of different things is not expressible here, and saying so is
    better than a second factor nobody sets.

    Both weights grow with the gross weight they stabilise (exponents 0.414
    and 0.376, both < 1), so adding them to the weight fixed point leaves it
    a contraction — see :func:`total_weight`.
    """
    if surf.S <= 0.0 or surf.AR <= 0.0:
        raise ValueError(
            f"tail area and aspect ratio must be > 0 (got S={surf.S}, "
            f"AR={surf.AR})")
    if surf.taper <= 0.0 or surf.tc <= 0.0:
        raise ValueError(
            f"tail taper and t/c must be > 0 (got taper={surf.taper}, "
            f"tc={surf.tc}) — lam^p and (100 t/c)^-p degenerate at 0")
    if not -90.0 < surf.sweep_deg < 90.0:
        raise ValueError(
            f"tail quarter-chord sweep {surf.sweep_deg} deg outside (-90, 90)")
    if n_ult <= 0.0 or W_dg_N < 0.0 or q_Pa <= 0.0:
        raise ValueError(
            f"need n_ult > 0, W_dg_N >= 0, q_Pa > 0 (got {n_ult}, {W_dg_N}, "
            f"{q_Pa})")
    if not (k_w > 0.0):
        raise ValueError(f"material weight factor must be > 0 (got {k_w})")

    S_ft2 = float(surf.S) * FT2_PER_M2
    W_dg_lb = float(W_dg_N) * LBF_PER_N
    q_psf = float(q_Pa) * PSF_PER_PA
    cosL = float(np.cos(np.deg2rad(surf.sweep_deg)))
    A = float(surf.AR)
    lam = float(surf.taper)
    tc100 = 100.0 * float(surf.tc) / cosL

    if surf.vertical:
        h_ratio = 1.0 if surf.tailplane_on_fin else 0.0
        W_lb = (
            0.073
            * (1.0 + 0.2 * h_ratio)
            * (n_ult * W_dg_lb) ** 0.376
            * q_psf**0.122
            * S_ft2**0.873
            * tc100**-0.49
            * (A / cosL**2) ** 0.357
            * lam**0.039
        )
    else:
        W_lb = (
            0.016
            * (n_ult * W_dg_lb) ** 0.414
            * q_psf**0.168
            * S_ft2**0.896
            * tc100**-0.12
            * (A / cosL**2) ** 0.043
            * lam**-0.02
        )
    return float(k_w) * W_lb * N_PER_LBF          # imperial -> SI [N]


def empennage_weight(surfaces, n_ult: float, W_dg_N: float, *,
                     q_Pa: float = Q_PA_DEFAULT, k_w: float = 1.0) -> float:
    """Every empennage surface's weight [N] at one gross weight.

    Empty in, ZERO out — and that is the interface the whole change hangs
    on: a family with no empennage description weighs exactly what it always
    weighed, so nothing that never had a tail moves.
    """
    return float(sum(
        tail_weight_raymer(s, n_ult, W_dg_N, q_Pa=q_Pa, k_w=k_w)
        for s in (surfaces or ())))


def total_weight(
    W_fixed_N: float,
    b: float,
    S: float,
    taper: float,
    sweep_deg: float,
    tc: float,
    n_ult: float = N_ULT_DEFAULT,
    *,
    W_fw_N: float = 0.0,
    q_Pa: float = Q_PA_DEFAULT,
    k_w: float = 1.0,
    tol_N: float = 1e-8,
    max_iter: int = 100,
) -> tuple[float, float]:
    """Close the weight loop: W_total = W_fixed + W_wing(W_dg = W_total).

    The wing weight depends on the gross weight it must carry, which
    includes the wing itself — a fixed-point problem. Iterate

        W(k+1) = W_fixed + W_wing(W_dg = W(k)),   W(0) = W_fixed.

    Convergence is guaranteed and geometric: W_wing = C * W_dg^0.49, so the
    iteration map g(W) = W_fixed + C W^0.49 has slope

        g'(W) = 0.49 * W_wing / W  <  0.49  <  1

    on W >= W_fixed > W_wing's contribution — a Banach contraction; in
    practice |dW| < 1e-8 N in ~10 iterations.

    ``k_w`` (materials.py) scales C and therefore W_wing, and does NOT break
    the contraction — the slope is 0.49 * W_wing / W whatever C is. What a
    large factor CAN do is push the fixed point past a weight the caller's
    other models accept, so callers keep catching this: a heavy material
    (foam at k_w 2.3) failing to close is a real answer about that wing, not
    a bug, and every caller already maps the exception onto the penalty
    contract.

    Returns ``(W_total_N, W_wing_N)`` in newtons.
    """
    if W_fixed_N <= 0.0:
        raise ValueError(f"W_fixed_N must be > 0 (got {W_fixed_N})")

    W_total = W_fixed_N
    resid = float("inf")
    for _ in range(max_iter):
        W_wing = wing_weight_raymer(
            b, S, taper, sweep_deg, tc, n_ult, W_dg_N=W_total,
            W_fw_N=W_fw_N, q_Pa=q_Pa, k_w=k_w,
        )
        W_next = W_fixed_N + W_wing
        resid = abs(W_next - W_total)
        if resid < tol_N:
            return W_next, W_wing
        W_total = W_next
    raise RuntimeError(
        f"weight fixed point did not converge in {max_iter} iterations "
        f"(last |dW| = {resid:.3e} N) — should be impossible for the "
        "0.49-exponent contraction; check inputs"
    )


# ---------------------------------------------------------------- spar stress


def root_bending_stress(
    b: float,
    S: float,
    taper: float,
    tc: float,
    W_total_N: float,
    n_ult: float = N_ULT_DEFAULT,
    c_root: float | None = None,
) -> float:
    """Root bending stress [Pa] of a two-cap spar — REDUCED-ORDER estimate.

    Model, step by step (all SI):

    1. Load: at the ultimate load factor each semi-span carries
       n_ult * W_total / 2 of lift. For an elliptic-ish distribution the
       semi-span lift centroid sits at

           y_bar = (4 / (3 pi)) * (b / 2)

       (first moment of an elliptic loading over the half-span), so the
       root bending moment is

           M_root = n_ult * (W_total / 2) * y_bar.

       Inertial relief (the wing's own weight bending the other way) is
       NEGLECTED — conservative.
    2. Structure: a two-cap spar at the root section. The caps sit at the
       top/bottom of a structural box of depth

           h_box = H_BOX_FRAC * tc * c_root = 0.85 * tc * c_root

       (85% of the max section thickness — typical front/rear-spar box
       depth), with c_root = 2 S / (b (1 + taper)) exactly as
       geometry.Wing.c_root. Each cap has area

           A_cap = K_CAP * tc * c_root^2

       — cap area scales with the local section cross-section (thicker or
       bigger sections carry proportionally bigger caps). K_CAP = 0.0025
       is CALIBRATED, not derived: chosen so the reference design
       b = 11 m, S = 16 m^2, taper = 0.45, tc = 0.12, n_ult = 5.7,
       W_total = 9.5 kN (c_root = 2.006 m, h_box = 0.2046 m,
       A_cap = 12.08 cm^2, M_root = 63.20 kN m) lands NEAR the allowable:
       sigma_ref = 255.8 MPa = 1.11 x SIGMA_ALLOW_PA — i.e. the strength
       constraint is mildly ACTIVE at the reference planform, which is
       what makes the constrained problem interesting.
    3. Two-cap bending: the couple M = sigma * A_cap * h_box gives

           sigma = M_root / (A_cap * h_box)   [Pa].

    Sign sanity (asserted in tests/test_weights.py): at fixed S and taper,
    c_root ~ 1/b, so sigma ~ n_ult * W * b / (tc^2 * c_root^3) ~ b^4 —
    steeply UP in span (longer arm AND thinner sections), UP in n_ult and
    W_total (linear), DOWN in tc (~tc^-2) and DOWN in c_root (bigger box:
    both cap area and depth grow).

    ``c_root`` overrides the trapezoidal root chord of step 2. It exists for
    planforms that are NOT trapezoids: with a free chord law
    (geometry.Wing.chord_coeffs) the root section — i.e. the spar box the
    caps live in — is no longer 2S/(b(1+taper)), and sizing the box off the
    taper alone would let a design widen its root for free. Pass
    ``wing.chord(0.0)``; ``None`` reproduces the trapezoidal formula exactly.

    Honesty label: this is a constraint SHAPE for the Tier optimiser, not
    a stress prediction — no shear web, no cap taper along span, no
    inertial relief, single load case. The LOAD term is likewise
    planform-blind (an elliptic centroid at 4/(3 pi) * b/2), so a chord law
    moves the section this model sizes, not the moment it carries.
    """
    if b <= 0.0 or S <= 0.0 or taper <= 0.0 or tc <= 0.0:
        raise ValueError(
            f"need b, S, taper, tc > 0 (got {b}, {S}, {taper}, {tc})"
        )
    if W_total_N < 0.0 or n_ult <= 0.0:
        raise ValueError(f"need W_total_N >= 0, n_ult > 0 (got {W_total_N}, {n_ult})")

    y_bar = (4.0 / (3.0 * np.pi)) * (b / 2.0)      # elliptic semi-span centroid
    M_root = n_ult * (W_total_N / 2.0) * y_bar     # [N m]

    if c_root is None:
        c_root = 2.0 * S / (b * (1.0 + taper))     # geometry.Wing.c_root
    else:
        c_root = float(c_root)
        if not (c_root > 0.0):
            raise ValueError(f"c_root must be > 0 (got {c_root})")
    h_box = H_BOX_FRAC * tc * c_root               # [m]
    A_cap = K_CAP * tc * c_root**2                 # [m^2]

    return float(M_root / (A_cap * h_box))         # [Pa]
