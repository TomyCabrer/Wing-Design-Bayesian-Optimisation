"""Tier C companion: wing in ground effect — rigid-wall (positive-image) LLT.

Model: a lifting wing at height ``h_agl`` (above ground level) over a flat,
rigid ground plane. The ground is captured by an image vortex system
mirrored across the plane, exactly as the free-surface hydrofoil
(hydrofoil.py) mirrors across the water surface — but with the OPPOSITE
image circulation sign. As in the tandem coupling blocks (tandem.py) the
image's circulation is LINEAR in the real wing's circulation, so it folds
into the influence matrix and the problem stays ONE linear system with no
new unknowns.

Image sign (the classic trap — DERIVED from the wall condition, not assumed)
--------------------------------------------------------------------------
The ground is a solid wall: the flow cannot cross it, so the total vertical
(normal) velocity must vanish on the plane,

    w_total(x, y, z_ground) = 0        for all x, y.

This is the rigid-wall / low-Froude limit of hydrofoil.py's linearised
free-surface condition ``U^2 phi_xx + g phi_z = 0`` with Fn -> 0, giving
``phi_z = 0`` — a SYMMETRIC extension of the potential across the plane.
Symmetric phi means w (= phi_z) is ODD in the wall-normal coordinate, so w
changes sign across the plane and therefore vanishes ON it. In vortex
terms this is realised by the mirror-image system carrying the OPPOSITE
fixed-frame circulation.

Verify it on a single filament (done numerically in
tests/test_ground_effect.py::test_boundary_condition). Take a streamwise
(trailing) vortex of strength ``s`` at height ``h`` above the wall (z = 0),
i.e. at (y0, +h). The image sits at the mirror point (y0, -h). Using the
trailing-leg kernel of tandem.w_influence (upwash positive),

    w(y, 0) from real  = (s / 4pi) * (y - y0) / ((y - y0)^2 + h^2) * (1 + dx/r)
    w(y, 0) from image = (s_img / 4pi) * (y - y0) / ((y - y0)^2 + h^2) * (1 + dx/r)

because the kernel depends on the height only through dz^2 (it is EVEN in
dz, and both filaments sit at |dz| = h from the wall point). The two
contributions cancel on the wall iff ``s_img = -s``: the OPPOSITE-sign
image. (This evenness is also why build_ground_operators may place the
image geometrically below the wing at ``z - 2 h`` and re-use the SAME
influence machinery as the free surface: the influence matrix is identical
to the hydrofoil's image-above matrix; only the coupling SIGN differs, and
that sign is applied explicitly at the coupling term below — never by
hacking geometry.)

Physical consequence — the OPPOSITE of the free surface
-------------------------------------------------------
The real wing's trailing vortices induce DOWNWASH on themselves (that is
the isolated induced drag). The opposite-sign image trailing vortices,
mirrored below, therefore induce UPWASH at the real wing:

    eps_ground(y) > 0  (upwash)  ==>  alpha_i drops at fixed alpha
    CL_alpha RISES,  CDi_total DROPS,  e_total > e_isolated

as ``h_agl -> 0``. This is the classic Wieselsberger / Prandtl ground
effect. Quantitatively it is Prandtl's biplane at gap ``2 h`` with the
OPPOSITE-circulation partner: at fixed elliptic loading the image-induced
drag on the real wing is

    CDi_ground = -sigma(2h/b) * CDi_self,
    sigma(g/b) = (1 - 0.66 g/b) / (1.05 + 3.7 g/b)     (Prandtl 1924)

so the induced-drag reduction fraction is +sigma(2h/b) — the exact SIGN
flip of hydrofoil.py's ``+sigma`` drag INCREASE. (hydrofoil.py's module
docstring calls the free-surface image the "negative image": that name
refers to the sign flip of the POTENTIAL, whose Fn -> inf limit is the
anti-symmetric / same-fixed-frame-circulation case. The ground is its
Fn -> 0 mirror: symmetric potential, opposite-circulation image.)

Only the REAL wing is physical, so the ground-induced ("mutual") drag is
counted once — on the real wing — as in hydrofoil.py.

Validity and limitations
------------------------
* Lifting-line scope: the image captures the 3-D induced part of ground
  effect (trailing-vortex upwash), which dominates for span-order heights
  h/b ~ O(1). The 2-D "ram"/chordwise cushion (image BOUND vortex raising
  the section lift for h of order a chord) is outside lifting-line
  resolution — flagged, not modelled.
* Flat, level, rigid ground; small bank/pitch; steady. No wall boundary
  layer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .llt import LLTResult, TWO_PI, fourier_system
from .tandem import Surface, VortexSystem, mutual_cdi, w_influence, _panel_quadrature

# ---------------------------------------------------------------- image sign
#
# Rigid-wall (no-penetration) boundary condition => the mirror-image vortex
# system carries the OPPOSITE fixed-frame circulation sense to the real wing
# (derived in the module docstring; verified numerically by the boundary
# condition test). This is the ONE sign that distinguishes ground effect
# (this module) from the free-surface hydrofoil (hydrofoil.py, image sign
# +1). It is applied EXPLICITLY at the coupling term in solve_wing_ige, not
# by moving the image geometry.
GROUND_IMAGE_SIGN = -1.0


# ---------------------------------------------------------------- image operators


@dataclass
class GroundOperators:
    """Cached influence of the (opposite-sign) ground image on the real wing.

    Mirrors hydrofoil.ImageOperators. The stored matrices are the influence
    of a UNIT-circulation image system (geometry only); the physical
    OPPOSITE sign is applied at the coupling term, so ``W``/``Wq`` here are
    numerically identical to the free-surface image-above matrices at the
    same gap (the kernel is even in the wall-normal offset).

    W  : (N, N)          image panel circulations -> w at collocation points
    Wq : (N*n_gauss, N)  same, at the wing's drag-quadrature points
    """

    img_sys: VortexSystem
    W: np.ndarray
    Wq: np.ndarray
    h_agl: float
    n_gauss: int = 8


def build_ground_operators(sys: VortexSystem, h_agl: float,
                           n_gauss: int = 8) -> GroundOperators:
    """Image system for a wing ``h_agl`` above a rigid ground plane.

    The ground lies at z = sys.z - h_agl; the geometric image is the mirror
    of the (planar-horizontal) lifting line across that plane, i.e. a pure
    z-translation to z = sys.z - 2*h_agl, with UNIT circulation mapping.
    The physical opposite-sign circulation (module docstring) is NOT baked
    into the geometry — it is applied at the coupling term in
    ``solve_wing_ige`` via ``GROUND_IMAGE_SIGN``.
    """
    if h_agl <= 0.0:
        raise ValueError(f"h_agl must be > 0 (got {h_agl})")
    img = VortexSystem(b=sys.b, N=sys.N, x=sys.x, z=sys.z - 2.0 * h_agl)
    pts = np.column_stack([np.full(sys.N, sys.x), sys.y_col,
                           np.full(sys.N, sys.z)])
    qpts, _ = _panel_quadrature(sys, n_gauss)
    return GroundOperators(
        img_sys=img,
        W=w_influence(pts, img),
        Wq=w_influence(qpts, img),
        h_agl=float(h_agl),
        n_gauss=n_gauss,
    )


# ---------------------------------------------------------------- coupled solve


@dataclass
class GroundEffectResult:
    foil: LLTResult           # per-wing result; alpha_eff_y includes eps_ground
    eps_ground: np.ndarray    # (N,) image-induced angle at the wing [rad]
    #                           (UPWASH: eps_ground > 0 for a lifting wing)
    CL: float
    CDi_self: float           # Fourier self-induced drag (isolated-wing term)
    CDi_ground: float         # image-induced drag, counted ONCE (NEGATIVE: a
    #                           reduction — ground-effect induced "thrust")
    CDi_total: float          # CDi_self + CDi_ground  (< CDi_self)
    e_total: float            # CL^2 / (pi AR CDi_total) — ground-boosted e > iso
    h_agl: float
    Sref: float


def solve_wing_ige(
    surf: Surface,
    h_agl: float,
    V: float = 1.0,
    Sref: float | None = None,
    ops: GroundOperators | None = None,
) -> GroundEffectResult:
    """Solve the monoplane equation with the ground image folded in.

    With G = 2 b V sin(j theta) mapping Fourier coefficients to panel
    circulations and s = GROUND_IMAGE_SIGN (= -1, the rigid-wall image),
    the image adds eps_ground = s * (1/V) W G A to the effective angle, so
    the single linear system is

        (M - s (1/V) W G) A = alpha_geo - alpha_L0.

    This is the same block as one off-diagonal tandem coupling and as the
    hydrofoil image system — the ONLY difference from hydrofoil.py is the
    factor s: +1 there (same-sign, drag up), -1 here (opposite-sign wall,
    drag DOWN). The sign lives here, on the coupling term, exactly once.
    """
    n = surf.N
    a0 = np.broadcast_to(np.asarray(surf.alpha_L0, float), (n,))
    ag = np.broadcast_to(np.asarray(surf.alpha_geo, float), (n,))

    theta, y, sin_jt, nsin_jt, M = fourier_system(surf.b, surf.c, surf.a)
    G = 2.0 * surf.b * V * sin_jt                  # A -> panel Gamma

    if ops is None:
        ops = build_ground_operators(surf.system, h_agl)

    s = GROUND_IMAGE_SIGN
    K = M - s * (1.0 / V) * ops.W @ G              # = M + (1/V) W G
    A = np.linalg.solve(K, ag - a0)
    Gam = G @ A
    eps = s * (ops.W @ Gam) / V                    # image-induced angle (upwash>0)

    c = np.asarray(surf.c, float)
    S = np.trapezoid(c, y)
    AR = surf.b**2 / S
    nvec = np.arange(1, n + 1)
    sum_nA2 = float(np.sum(nvec * A**2))
    alpha_i = nsin_jt @ A
    CL = float(np.pi * AR * A[0])
    CDi_self = float(np.pi * AR * sum_nA2)

    foil = LLTResult(
        CL=CL, CDi=CDi_self,
        e=float(A[0] ** 2 / sum_nA2) if sum_nA2 > 0 else np.nan,
        AR=float(AR), S=float(S), A=A, theta=theta, y=y, c=c,
        Gamma=Gam, Cl_y=2.0 * Gam / (V * c), alpha_i_y=alpha_i,
        alpha_eff_y=ag + eps - alpha_i,
    )

    Sref = Sref if Sref is not None else float(S)
    # ground drag on the REAL wing only; the image's circulation is s*Gam
    # (opposite sign), so passing Gamma_b = s*Gam yields CDi_ground < 0:
    cdi_ground, _ = mutual_cdi(
        surf.system, ops.img_sys, Gam, s * Gam, V, Sref,
        n_gauss=ops.n_gauss, W_a=ops.Wq,
    )
    CL_ref = CL * S / Sref
    CDi_self_ref = CDi_self * S / Sref
    CDi_total = CDi_self_ref + cdi_ground
    e_total = float(CL_ref**2 / (np.pi * (surf.b**2 / Sref) * CDi_total)) \
        if CDi_total > 0 else np.nan

    return GroundEffectResult(
        foil=foil, eps_ground=eps, CL=CL_ref,
        CDi_self=CDi_self_ref, CDi_ground=float(cdi_ground),
        CDi_total=float(CDi_total), e_total=e_total,
        h_agl=float(h_agl), Sref=float(Sref),
    )


def solve_wing_ige_trim(
    b: float,
    c: np.ndarray,
    twist_y: np.ndarray,
    CL_target: float,
    h_agl: float,
    a: np.ndarray | float = TWO_PI,
    alpha_L0: np.ndarray | float = 0.0,
    V: float = 1.0,
    alpha_bracket_deg: tuple[float, float] = (-10.0, 15.0),
    ops: GroundOperators | None = None,
) -> tuple[float, GroundEffectResult]:
    """Trim root AoA so CL = CL_target (closed form — CL is affine in alpha).

    Mirrors solve_hydrofoil_trim: the coupled system is linear and alpha
    enters only the RHS, so two solves give the exact trim point. In ground
    effect the trimmed wing needs LESS alpha than in free air (the image
    upwash raises CL_alpha).
    """
    twist_y = np.asarray(twist_y, dtype=float)
    N = np.asarray(c).size
    sys0 = VortexSystem(b=b, N=N)
    if ops is None:
        ops = build_ground_operators(sys0, h_agl)

    def _solve(alpha: float) -> GroundEffectResult:
        return solve_wing_ige(
            Surface(b=b, c=c, alpha_geo=alpha + twist_y, a=a, alpha_L0=alpha_L0),
            h_agl, V=V, ops=ops,
        )

    r0 = _solve(0.0)
    r1 = _solve(np.deg2rad(1.0))
    slope = (r1.CL - r0.CL) / np.deg2rad(1.0)
    if abs(slope) < 1e-9:
        raise ValueError("degenerate CL_alpha in ground-effect trim")
    alpha = (CL_target - r0.CL) / slope
    lo, hi = np.deg2rad(alpha_bracket_deg[0]), np.deg2rad(alpha_bracket_deg[1])
    if not (lo <= alpha <= hi):
        raise ValueError(
            f"untrimmable: alpha = {np.rad2deg(alpha):.2f} deg outside bracket"
        )
    return float(alpha), _solve(alpha)
