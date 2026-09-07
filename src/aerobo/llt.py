"""Single-wing numerical lifting-line (Prandtl/Glauert Fourier-series method).

numpy port of the single-wing kernel `solve_llt` in
~/Desktop/GDP/tandem_wing/core/lift_drag_strip.m (MATLAB reference):

    theta_i = (i - 1/2) * pi / N            midpoint cosine grid (avoids tips)
    y_i     = -(b/2) * cos(theta_i)
    Gamma(theta) = 2 b V sum_n A_n sin(n theta)
    M_ij  = (4 b / (a_i c_i)) sin(j theta_i) + j sin(j theta_i) / sin(theta_i)
    RHS_i = alpha_i - alpha_L0_i
    CL    = pi * AR * A_1
    CDi   = pi * AR * sum_n n A_n^2
    e     = A_1^2 / sum_n n A_n^2
    alpha_induced(y) = sum_n n A_n sin(n theta) / sin(theta)

Incompressible (M = 0, no Prandtl-Glauert correction) — Tier A scope.
Extension points: compressibility via a beta_PG factor on `a`, per-station
polar slopes via the `a` and `alpha_L0` arrays (see polar.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

TWO_PI = 2.0 * np.pi

# Prandtl-Glauert validity ceiling: the linearised subsonic correction (and its
# 1/beta singularity) loses meaning as M -> 1; the drag-divergence / transonic
# regime is out of this reduced-order model's scope. The caller maps the raised
# ValueError to the objective's PENALTY (never a NaN to the optimiser).
PG_MACH_MAX = 0.7


def pg_beta(mach: float) -> float:
    """Prandtl-Glauert compressibility factor beta = sqrt(1 - M^2).

    Applied to the SECTION lift-curve slope: a -> a / beta (Anderson,
    *Fundamentals of Aerodynamics*, subsonic-compressible thin-aerofoil
    result; Glauert 1928). Physically, the linearised subsonic equation
    (1 - M^2) phi_xx + phi_yy = 0 maps to the incompressible Laplace problem
    under the Prandtl-Glauert stretch x -> x/beta, and the pressure/lift
    scaling that undoes the stretch multiplies the incompressible section
    slope by 1/beta. Since beta = sqrt(1 - M^2) < 1 for 0 < M < 1, the slope
    RISES with Mach (compressibility steepens the lift curve) — the sign is
    fixed by 1/beta being monotone increasing in M, not assumed.

    SCOPE / honest label: this is a SECTION-slope-only correction. The 3-D
    lifting line then propagates the corrected slope through the SAME
    incompressible monoplane equation (`fourier_system` builds M from the
    section a) — the full Goethert 3-D affine scaling of the planform
    (span/aspect-ratio stretch by beta) is DELIBERATELY not applied. This is
    a reduced-order, section-level compressibility term, labelled as such;
    wave drag remains out of scope (see drag.py, whose Raymer FF Mach factor
    is floored at 1 and models no wave rise).

    Parameters
    ----------
    mach : free-stream Mach number, 0 <= M < 0.7.

    Raises
    ------
    ValueError
        If M < 0 (unphysical) or M >= 0.7 (transonic — outside the
        linearised subsonic regime; the caller maps this to PENALTY).
    """
    if mach < 0.0:
        raise ValueError(f"Mach must be non-negative, got {mach}")
    if mach >= PG_MACH_MAX:
        raise ValueError(
            f"Mach {mach} >= {PG_MACH_MAX}: transonic, Prandtl-Glauert invalid"
        )
    return float(np.sqrt(1.0 - mach * mach))


def swept_section_slope(a0, sweep_deg: float, mach: float = 0.0):
    """Effective section lift-curve slope under simple sweep (+ compressibility).

    Incompressible simple-sweep theory: only the freestream component normal
    to the quarter-chord line does aerodynamic work on the section (sweep
    independence principle, R. T. Jones 1946), so a wing swept by Λ has
    effective section slope a0 cos Λ. REDUCED-ORDER correction, mirroring the
    Tier A style: the lifting line itself stays geometrically UNSWEPT
    (straight, planar) — this is not a swept-wake solver, and sweep-induced
    root/tip loading distortion is not modelled. Sweep otherwise enters only
    the parasite form factor (drag.py).

    Compressible simple sweep (``mach`` > 0): the Prandtl-Glauert factor is
    formed from the Mach component NORMAL to the quarter-chord, M_n = M cos Λ,
    since only that normal flow is compressed on the section (Kuchemann,
    *The Aerodynamic Design of Aircraft*; Raymer, *Aircraft Design*, simple-
    sweep compressibility). Combining the cos Λ slope reduction with the
    1/beta_n compressibility rise gives the standard form

        a_eff = a0 cos Λ / sqrt(1 - M^2 cos^2 Λ).

    Note the NORMAL Mach: at Λ = 60 deg, M = 0.6 the correction uses
    M cos Λ = 0.3, NOT 0.6 — sweep relieves compressibility. beta_n rises
    toward 1 as sweep increases, so at fixed M the slope INCREASES with Mach
    (1/beta_n > 1) — sign fixed by construction, not assumed. Only the
    section slope is corrected; the 3-D LLT geometry is unchanged (see
    `pg_beta` for the deliberate non-application of full Goethert scaling).

    ``mach`` == 0.0 (the default) short-circuits to the IDENTICAL float
    expression a0 * cos Λ used before this term existed, preserving the Tier A
    / tier_a_plus objective BIT-FOR-BIT (regression-gated). The free-stream
    transonic validity guard lives upstream in `pg_beta` / the objective; this
    routine is the pure slope formula.
    """
    cosL = np.cos(np.deg2rad(sweep_deg))
    a0 = np.asarray(a0, dtype=float)
    if mach == 0.0:
        # Legacy path: execute the EXACT pre-compressibility expression so the
        # M = 0 result is bit-for-bit identical (== float equality).
        return a0 * cosL
    Mn = mach * cosL
    beta_n = np.sqrt(1.0 - Mn * Mn)
    return a0 * cosL / beta_n


def cosine_stations(N: int, b: float) -> tuple[np.ndarray, np.ndarray]:
    """Midpoint cosine grid: theta in (0, pi), y = -(b/2) cos(theta)."""
    theta = (np.arange(1, N + 1) - 0.5) * np.pi / N
    y = -(b / 2.0) * np.cos(theta)
    return theta, y


def elliptic_chord(y: np.ndarray, b: float, S: float) -> np.ndarray:
    """Elliptic planform chord distribution with area S: c0 = 4S/(pi*b)."""
    c0 = 4.0 * S / (np.pi * b)
    return c0 * np.sqrt(np.maximum(1.0 - (2.0 * y / b) ** 2, 0.0))


@dataclass
class LLTResult:
    CL: float
    CDi: float
    e: float                 # span efficiency = A1^2 / sum(n An^2)
    AR: float
    S: float
    A: np.ndarray            # (N,) Fourier coefficients A_n
    theta: np.ndarray        # (N,) grid
    y: np.ndarray            # (N,)
    c: np.ndarray            # (N,)
    Gamma: np.ndarray        # (N,) circulation (for V passed to solve_llt)
    Cl_y: np.ndarray         # (N,) local section lift coefficient
    alpha_i_y: np.ndarray    # (N,) induced AoA [rad]
    alpha_eff_y: np.ndarray  # (N,) effective AoA = alpha_geo - alpha_i [rad]


def fourier_system(
    b: float, c: np.ndarray, a: np.ndarray | float = TWO_PI
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Monoplane-equation building blocks on the midpoint cosine grid.

    Returns (theta, y, sin_jt, n_sin_jt, M) with
        M_ij = (4b/(a_i c_i)) sin(j theta_i) + j sin(j theta_i)/sin(theta_i)
    so that ``M @ A = alpha_geo - alpha_L0`` is the single-wing LLT and
    ``n_sin_jt @ A`` is the self-induced angle. Shared by `solve_llt` and
    the Tier B tandem coupled solver (tandem.py), which stacks two of
    these blocks with cross-induction coupling terms.
    """
    c = np.asarray(c, dtype=float)
    N = c.size
    theta, y = cosine_stations(N, b)
    a_y = np.broadcast_to(np.asarray(a, dtype=float), (N,))

    n = np.arange(1, N + 1)
    sin_jt = np.sin(np.outer(theta, n))                    # (N, N)
    n_sin_jt = sin_jt * n / np.sin(theta)[:, None]

    M = (4.0 * b / (a_y * c))[:, None] * sin_jt + n_sin_jt
    return theta, y, sin_jt, n_sin_jt, M


def solve_llt(
    b: float,
    c: np.ndarray,
    alpha_geo: np.ndarray,
    a: np.ndarray | float = TWO_PI,
    alpha_L0: np.ndarray | float = 0.0,
    V: float = 1.0,
) -> LLTResult:
    """Solve the monoplane equation on the midpoint cosine grid.

    Parameters
    ----------
    b : span [m].
    c : (N,) local chord sampled at ``cosine_stations(N, b)`` [m].
    alpha_geo : (N,) local geometric AoA incl. twist [rad].
    a : section lift-curve slope [1/rad] (scalar or per-station).
    alpha_L0 : section zero-lift angle [rad] (scalar or per-station).
    V : freestream speed (only scales Gamma; coefficients are V-independent).
    """
    c = np.asarray(c, dtype=float)
    N = c.size
    alpha_geo = np.broadcast_to(np.asarray(alpha_geo, dtype=float), (N,))
    a0_y = np.broadcast_to(np.asarray(alpha_L0, dtype=float), (N,))

    theta, y, sin_jt, n_sin_jt, M = fourier_system(b, c, a)
    n = np.arange(1, N + 1)
    A = np.linalg.solve(M, alpha_geo - a0_y)

    S = np.trapezoid(c, y)                                 # planform area
    AR = b**2 / S

    CL = np.pi * AR * A[0]
    sum_nA2 = float(np.sum(n * A**2))
    CDi = np.pi * AR * sum_nA2
    e = A[0] ** 2 / sum_nA2 if sum_nA2 > 0 else np.nan

    Gamma = 2.0 * b * V * (sin_jt @ A)
    Cl_y = 2.0 * Gamma / (V * c)
    alpha_i_y = n_sin_jt @ A

    return LLTResult(
        CL=float(CL), CDi=float(CDi), e=float(e), AR=float(AR), S=float(S),
        A=A, theta=theta, y=y, c=c, Gamma=Gamma, Cl_y=Cl_y,
        alpha_i_y=alpha_i_y, alpha_eff_y=alpha_geo - alpha_i_y,
    )


def solve_llt_trim(
    b: float,
    c: np.ndarray,
    twist_y: np.ndarray,
    CL_target: float,
    a: np.ndarray | float = TWO_PI,
    alpha_L0: np.ndarray | float = 0.0,
    V: float = 1.0,
    alpha_bracket: tuple[float, float] = (np.deg2rad(-10.0), np.deg2rad(15.0)),
) -> tuple[float, LLTResult]:
    """Solve root AoA ``alpha`` so that CL = CL_target (brentq).

    ``twist_y`` [rad] is the local incidence RELATIVE to the root
    (+ = nose-up here; the geometry layer owns the washout sign convention).
    Local geometric AoA = alpha + twist_y.

    Raises ValueError if the bracket does not contain the trim point — and
    says what that MEANS. scipy's own wording ("f(a) and f(b) must have
    different signs") is true and useless: it names the root finder, while the
    thing that happened is that this wing cannot reach the lift it needs at
    any incidence in the bracket. That is a real design outcome, not a
    numerical accident — it is what a wing too small for its weight does, and
    it is where a searched wing loading riding the top of its band lands. It
    reaches the caller through the penalty contract either way; the difference
    is whether the run can be read afterwards.
    """
    twist_y = np.asarray(twist_y, dtype=float)

    def resid(alpha: float) -> float:
        return solve_llt(b, c, alpha + twist_y, a, alpha_L0, V).CL - CL_target

    lo, hi = float(alpha_bracket[0]), float(alpha_bracket[1])
    r_lo, r_hi = resid(lo), resid(hi)
    if not np.isfinite(r_lo) or not np.isfinite(r_hi):
        raise ValueError(
            f"trim residual is not finite over the incidence bracket "
            f"[{np.rad2deg(lo):g}, {np.rad2deg(hi):g}] deg (CL_target = "
            f"{float(CL_target):g})")
    if r_lo * r_hi > 0.0:
        short = r_hi < 0.0        # even at the top of the bracket, not enough
        raise ValueError(
            f"wing {'cannot reach' if short else 'never falls to'} CL_target "
            f"= {float(CL_target):.4g} anywhere in the incidence bracket "
            f"[{np.rad2deg(lo):g}, {np.rad2deg(hi):g}] deg, over which CL "
            f"spans [{r_lo + float(CL_target):.4g}, "
            f"{r_hi + float(CL_target):.4g}]: the wing is "
            f"{'too small for' if short else 'too large for'} the lift asked "
            f"of it. A design outcome, not a solver fault")
    alpha = brentq(resid, lo, hi, xtol=1e-10)
    res = solve_llt(b, c, alpha + twist_y, a, alpha_L0, V)
    return float(alpha), res


def elliptic_twist_alpha(
    y: np.ndarray,
    b: float,
    c: np.ndarray,
    CL_target: float,
    a: np.ndarray | float = TWO_PI,
    alpha_L0: np.ndarray | float = 0.0,
) -> np.ndarray:
    """Closed-form local geometric AoA producing elliptic loading at CL_target.

    Single-wing degenerate case of inverse_twist_elliptic.m (validation
    oracle): with Gamma_t = Gamma0 sqrt(1 - eta^2), Gamma0 = 2 CL V S/(pi b),

        Cl_t(y)       = 2 Gamma_t / (V c)
        alpha_induced = CL / (pi AR)     (constant for elliptic loading)
        alpha_req(y)  = Cl_t / a + alpha_L0 + alpha_induced   [rad]
    """
    y = np.asarray(y, dtype=float)
    c = np.asarray(c, dtype=float)
    S = np.trapezoid(c, y)
    AR = b**2 / S
    eta = 2.0 * y / b
    Gamma_t_over_V = (2.0 * CL_target * S / (np.pi * b)) * np.sqrt(
        np.maximum(1.0 - eta**2, 0.0)
    )
    Cl_t = 2.0 * Gamma_t_over_V / c
    alpha_i = CL_target / (np.pi * AR)
    return Cl_t / np.asarray(a, dtype=float) + np.asarray(alpha_L0, dtype=float) + alpha_i
