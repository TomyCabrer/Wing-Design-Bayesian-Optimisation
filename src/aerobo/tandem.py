"""Tier B: tandem two-surface lifting-line with mutual induced-velocity coupling.

Physics ported from the MATLAB reference
~/Desktop/GDP/tandem_wing/core/lift_drag_strip.m (with downwash_strip.m /
downwash_strip_back.m), which implements Kryvokhatko (2023) Ch. 2's "direct
method": each wing is a Fourier-series LLT, the OTHER wing's induced angle
enters its monoplane equation as extra spanwise twist (the reference's
eps21 = front->rear downwash, eps12 = rear->front up/downwash), and the
mutual induced drag is the lift-vector tilt

    D_mut = -rho V \\int Gamma(y) * (w_cross(y)/V) dy        (per wing)

so downwash on the rear wing (w < 0) is drag and upwash on the front wing
(w > 0, from the rear's bound vortex) is induced THRUST — the classic
tandem trade (reference lift_drag_strip.m lines 229-238, "book Fig 2.16").

Two deliberate upgrades over the reference (documented, test-gated):

1. Coupling kernel. The reference models the inducing wing as a SINGLE
   rolled-up horseshoe (tip separation l0, viscous core r0 = 0.014 b —
   book eqs. 2.4/2.5). Here the inducing wing is the full distributed
   trailing-vortex system implied by its panelised Fourier circulation:
   one horseshoe per spanwise panel — bound segment on the lifting line,
   semi-infinite trailing legs at the panel edges with strengths equal to
   the circulation jumps. Munk's stagger theorem holds EXACTLY for this
   discrete vortex system, which tests/test_tandem.py exploits as the
   classic tandem validation gate.

2. Direct coupled solve. The reference Picard-iterates two scalar CLs
   (omega = 0.5, tol 1e-4, maxit 30). LLT is linear, so the two Fourier
   systems + coupling are assembled into ONE (N1+N2)x(N1+N2) linear
   system and solved directly — no iteration, no relaxation, and the
   fixed point is exact to machine precision.

Conventions
-----------
x downstream (+), z up (+); each surface's quarter-chord lifting line is
straight along y at its own (x, z); a tandem places the rear surface at
(x, z) = (dx, dz) relative to the front. eps = w/V with w the
cross-induced VERTICAL velocity, upwash POSITIVE:

    alpha_eff = alpha_geo + eps_cross - alpha_i_self

Downwash on the rear wing is eps < 0 (numerically matching the reference,
where eps21 < 0 lowers the rear wing's effective AoA). Note the physics
consequence tested in test_tandem.py: pure streamwise separation NEVER
decouples the pair — the rear wing sits in the front wing's trailing
wake at any dx (far downwash -> 2x the at-the-line value); decoupling
requires a large VERTICAL gap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import fin as _fin
from . import geometry
from .llt import TWO_PI, LLTResult, cosine_stations, fourier_system
from .objective import MU_SL, PENALTY, RHO_SL, _fail
from .polar import default_polar

# ------------------------------------------------------------------ vortex system


@dataclass(frozen=True)
class VortexSystem:
    """Panelised horseshoe representation of one lifting line.

    Panels are the N midpoint-cosine LLT stations; panel EDGES are at
    theta = k*pi/N (k = 0..N). Panel circulations live at the stations;
    trailing-leg strengths are the circulation jumps at the edges.
    """

    b: float
    N: int
    x: float = 0.0
    z: float = 0.0

    @property
    def y_edges(self) -> np.ndarray:
        theta_e = np.arange(self.N + 1) * np.pi / self.N
        return -(self.b / 2.0) * np.cos(theta_e)

    @property
    def y_col(self) -> np.ndarray:
        return cosine_stations(self.N, self.b)[1]


def _shed_matrix(N: int) -> np.ndarray:
    """(N+1, N) map: panel circulations -> trailing-leg strengths at edges.

    Edge e sheds s_e = Gamma_{e-1} - Gamma_e (Gamma_{-1} = Gamma_N = 0),
    tangent +x (downstream); right tip sheds +Gamma_last, left tip
    -Gamma_first — Helmholtz conservation per panel horseshoe.
    """
    E = np.zeros((N + 1, N))
    idx = np.arange(N)
    E[idx + 1, idx] += 1.0
    E[idx, idx] -= 1.0
    return E


def w_influence(points: np.ndarray, src: VortexSystem,
                rcore: float = 0.0) -> np.ndarray:
    """(M, N_src) matrix: unit panel circulations -> vertical velocity w at points.

    Biot-Savart of the full panel-horseshoe system (upwash positive):
      * semi-infinite trailing legs from each panel edge, tangent +x:
            w = s/(4 pi) * dy/(dy^2+dz^2) * (1 + dx/r)
        — the (1 + dx/r) streamwise factor is the same structure as the
        reference downwash_strip.m kernel: 1 at the line, 2 far downstream,
        0 far upstream (signed dx handles the rear->front direction, cf.
        downwash_strip_back.m).
      * finite bound segments along the lifting line (Katz & Plotkin
        straight-segment formula, z-component) — this is the term that
        gives the front wing its upwash from the rear wing.

    ``rcore`` is an optional Lamb-Oseen-like core radius (the reference
    uses 0.014 b on its rolled-up tip vortices); default 0 keeps the
    kernel exact, which the Munk stagger gate requires.
    """
    P = np.asarray(points, dtype=float)
    ye = src.y_edges
    N = src.N

    dx = P[:, 0:1] - src.x                       # (M, 1)
    dz = P[:, 2:3] - src.z

    # --- trailing legs (M, N+1) ---
    dy_t = P[:, 1:2] - ye[None, :]               # (M, N+1)
    rho2 = dy_t**2 + dz**2 + rcore**2
    r = np.sqrt(dx**2 + dy_t**2 + dz**2 + rcore**2)
    with np.errstate(divide="ignore", invalid="ignore"):
        Wt = (1.0 / (4.0 * np.pi)) * dy_t / rho2 * (1.0 + dx / r)
    Wt = np.where(rho2 < 1e-30, 0.0, Wt)         # principal value on the line

    # --- bound segments (M, N): edge e -> e+1 along +y at (src.x, src.z) ---
    ya, yb = ye[:-1], ye[1:]
    dy1 = P[:, 1:2] - ya[None, :]                # r1 = P - A
    dy2 = P[:, 1:2] - yb[None, :]                # r2 = P - B
    r1 = np.sqrt(dx**2 + dy1**2 + dz**2)
    r2 = np.sqrt(dx**2 + dy2**2 + dz**2)
    # r1 x r2 for r1=(dx,dy1,dz), r2=(dx,dy2,dz):
    #   x: dy1*dz - dz*dy2 ; y: dz*dx - dx*dz = 0 ; z: dx*dy2 - dy1*dx
    cross_x = (dy1 - dy2) * dz
    cross_z = dx * (dy2 - dy1)
    cross2 = cross_x**2 + cross_z**2 + (rcore * (yb - ya))[None, :] ** 2
    # r0 = B - A = (0, yb-ya, 0):  r0.r1 = (yb-ya)*dy1, r0.r2 = (yb-ya)*dy2
    with np.errstate(divide="ignore", invalid="ignore"):
        dot_term = (yb - ya)[None, :] * (dy1 / r1 - dy2 / r2)
        Wb = (1.0 / (4.0 * np.pi)) * cross_z / cross2 * dot_term
    Wb = np.where(cross2 < 1e-30, 0.0, Wb)       # on the bound line itself

    return Wt @ _shed_matrix(N) + Wb


# ------------------------------------------------------------------ mutual drag

_GAUSS_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _gauss(n: int) -> tuple[np.ndarray, np.ndarray]:
    if n not in _GAUSS_CACHE:
        _GAUSS_CACHE[n] = np.polynomial.legendre.leggauss(n)
    return _GAUSS_CACHE[n]


def _panel_quadrature(sys: VortexSystem, n_gauss: int
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes (as (M,3) points on the lifting line) and weights
    for integrating along each panel of ``sys``; M = N * n_gauss."""
    xg, wg = _gauss(n_gauss)
    ye = sys.y_edges
    ya, yb = ye[:-1], ye[1:]
    half = 0.5 * (yb - ya)
    mid = 0.5 * (yb + ya)
    y_nodes = (mid[:, None] + half[:, None] * xg[None, :]).ravel()
    weights = (half[:, None] * wg[None, :]).ravel()
    pts = np.column_stack([np.full_like(y_nodes, sys.x), y_nodes,
                           np.full_like(y_nodes, sys.z)])
    return pts, weights


def mutual_cdi(
    sys_a: VortexSystem,
    sys_b: VortexSystem,
    Gamma_a: np.ndarray,
    Gamma_b: np.ndarray,
    V: float,
    Sref: float,
    n_gauss: int = 8,
    rcore: float = 0.0,
    W_a: np.ndarray | None = None,
    W_b: np.ndarray | None = None,
) -> tuple[float, float]:
    """Mutual induced-drag coefficients (CDi_a<-b, CDi_b<-a) on ``Sref``.

    D_a<-b = -rho * sum_panels Gamma_a,p * \\int_panel w_b(y) dy, evaluated
    with per-panel Gauss-Legendre quadrature along a's bound line (the
    trailing legs carry no drag: F = rho V x Gamma has no x-component on
    x-aligned filaments). For the exact panel-horseshoe velocity fields
    Munk's stagger theorem makes CDi_a<-b + CDi_b<-a independent of the
    streamwise separation — the quadrature is the only approximation.

    ``W_a``/``W_b`` allow passing precomputed influence matrices at the
    quadrature points (built by ``build_coupling``).
    """
    pa, wa = _panel_quadrature(sys_a, n_gauss)
    pb, wb = _panel_quadrature(sys_b, n_gauss)
    if W_a is None:
        W_a = w_influence(pa, sys_b, rcore)      # w at a's line from b
    if W_b is None:
        W_b = w_influence(pb, sys_a, rcore)      # w at b's line from a
    Ga = np.repeat(np.asarray(Gamma_a, float), n_gauss)
    Gb = np.repeat(np.asarray(Gamma_b, float), n_gauss)
    # CD = D/(q Sref) with q = rho V^2 / 2:  -rho*int(Gam*w)/(q Sref)
    #    = -2/(V^2 Sref) * int(Gam * w)          [w already includes Gamma_src]
    cdi_a = -2.0 / (V**2 * Sref) * float(np.sum(wa * Ga * (W_a @ Gamma_b)))
    cdi_b = -2.0 / (V**2 * Sref) * float(np.sum(wb * Gb * (W_b @ Gamma_a)))
    return cdi_a, cdi_b


# ------------------------------------------------------------------ coupled solve


@dataclass
class Surface:
    """One lifting surface sampled at its own N midpoint-cosine stations."""

    b: float
    c: np.ndarray                     # (N,) chord at cosine_stations(N, b)
    alpha_geo: np.ndarray             # (N,) local geometric AoA [rad]
    a: np.ndarray | float = TWO_PI    # section lift-curve slope [1/rad]
    alpha_L0: np.ndarray | float = 0.0
    x: float = 0.0                    # quarter-chord streamwise position
    z: float = 0.0                    # quarter-chord height

    @property
    def N(self) -> int:
        return np.asarray(self.c).size

    @property
    def system(self) -> VortexSystem:
        return VortexSystem(b=self.b, N=self.N, x=self.x, z=self.z)


@dataclass
class Coupling:
    """Geometry-only influence operators (chord/twist independent -> cacheable)."""

    W_fr: np.ndarray      # w at front collocation per rear panel Gamma
    W_rf: np.ndarray      # w at rear collocation per front panel Gamma
    Wq_fr: np.ndarray     # same, at front drag-quadrature points
    Wq_rf: np.ndarray     # same, at rear drag-quadrature points
    n_gauss: int = 8


def build_coupling(front_sys: VortexSystem, rear_sys: VortexSystem,
                   n_gauss: int = 8, rcore: float = 0.0) -> Coupling:
    pf = np.column_stack([np.full(front_sys.N, front_sys.x), front_sys.y_col,
                          np.full(front_sys.N, front_sys.z)])
    pr = np.column_stack([np.full(rear_sys.N, rear_sys.x), rear_sys.y_col,
                          np.full(rear_sys.N, rear_sys.z)])
    qf, _ = _panel_quadrature(front_sys, n_gauss)
    qr, _ = _panel_quadrature(rear_sys, n_gauss)
    return Coupling(
        W_fr=w_influence(pf, rear_sys, rcore),
        W_rf=w_influence(pr, front_sys, rcore),
        Wq_fr=w_influence(qf, rear_sys, rcore),
        Wq_rf=w_influence(qr, front_sys, rcore),
        n_gauss=n_gauss,
    )


@dataclass
class TandemResult:
    front: LLTResult          # per-surface results; alpha_eff_y includes eps
    rear: LLTResult
    eps_front: np.ndarray     # (N1,) cross-induced angle at front [rad], upwash +
    eps_rear: np.ndarray      # (N2,) front->rear downwash is eps_rear < 0
    CL_total: float           # on Sref
    CDi_total: float          # self (area-weighted) + mutual, on Sref
    CDi_mut: float
    CDi_mut_front: float      # < 0 = induced thrust from rear-wing upwash
    CDi_mut_rear: float       # > 0 = downwash drag behind the front wing
    Sref: float


def solve_tandem(
    front: Surface,
    rear: Surface,
    V: float = 1.0,
    rcore: float = 0.0,
    Sref: float | None = None,
    coupling: Coupling | None = None,
) -> TandemResult:
    """Directly solve the coupled two-surface monoplane equations.

    Block system (LLT is linear — this replaces the reference's Picard loop):

        [ M_f            -(1/V) W_fr G_r ] [A_f]   [alpha_f - a0_f]
        [ -(1/V) W_rf G_f     M_r        ] [A_r] = [alpha_r - a0_r]

    where G = 2 b V sin(j theta) maps Fourier coefficients to panel
    circulations and W maps the other surface's panel circulations to the
    cross-induced vertical velocity at this surface's collocation points.
    """
    n1, n2 = front.N, rear.N
    a0_f = np.broadcast_to(np.asarray(front.alpha_L0, float), (n1,))
    a0_r = np.broadcast_to(np.asarray(rear.alpha_L0, float), (n2,))
    ag_f = np.broadcast_to(np.asarray(front.alpha_geo, float), (n1,))
    ag_r = np.broadcast_to(np.asarray(rear.alpha_geo, float), (n2,))

    th_f, y_f, sin_f, nsin_f, M_f = fourier_system(front.b, front.c, front.a)
    th_r, y_r, sin_r, nsin_r, M_r = fourier_system(rear.b, rear.c, rear.a)
    G_f = 2.0 * front.b * V * sin_f              # A -> panel Gamma
    G_r = 2.0 * rear.b * V * sin_r

    if coupling is None:
        coupling = build_coupling(front.system, rear.system, rcore=rcore)

    K = np.zeros((n1 + n2, n1 + n2))
    K[:n1, :n1] = M_f
    K[n1:, n1:] = M_r
    K[:n1, n1:] = -(1.0 / V) * coupling.W_fr @ G_r
    K[n1:, :n1] = -(1.0 / V) * coupling.W_rf @ G_f
    rhs = np.concatenate([ag_f - a0_f, ag_r - a0_r])
    A = np.linalg.solve(K, rhs)
    A_f, A_r = A[:n1], A[n1:]

    Gam_f = G_f @ A_f
    Gam_r = G_r @ A_r
    eps_f = (coupling.W_fr @ Gam_r) / V
    eps_r = (coupling.W_rf @ Gam_f) / V

    def _pack(b, c, y, th, sin_jt, nsin, A_i, Gam, ag, eps) -> LLTResult:
        S = np.trapezoid(c, y)
        AR = b**2 / S
        n = np.arange(1, c.size + 1)
        sum_nA2 = float(np.sum(n * A_i**2))
        alpha_i = nsin @ A_i
        return LLTResult(
            CL=float(np.pi * AR * A_i[0]),
            CDi=float(np.pi * AR * sum_nA2),
            e=float(A_i[0] ** 2 / sum_nA2) if sum_nA2 > 0 else np.nan,
            AR=float(AR), S=float(S), A=A_i, theta=th, y=y, c=c,
            Gamma=Gam, Cl_y=2.0 * Gam / (V * c), alpha_i_y=alpha_i,
            alpha_eff_y=ag + eps - alpha_i,
        )

    res_f = _pack(front.b, np.asarray(front.c, float), y_f, th_f, sin_f,
                  nsin_f, A_f, Gam_f, ag_f, eps_f)
    res_r = _pack(rear.b, np.asarray(rear.c, float), y_r, th_r, sin_r,
                  nsin_r, A_r, Gam_r, ag_r, eps_r)

    Sref = Sref if Sref is not None else res_f.S + res_r.S
    cdi_mf, cdi_mr = mutual_cdi(
        front.system, rear.system, Gam_f, Gam_r, V, Sref,
        n_gauss=coupling.n_gauss, rcore=rcore,
        W_a=coupling.Wq_fr, W_b=coupling.Wq_rf,
    )
    CL_total = (res_f.CL * res_f.S + res_r.CL * res_r.S) / Sref
    CDi_total = (res_f.CDi * res_f.S + res_r.CDi * res_r.S) / Sref \
        + cdi_mf + cdi_mr

    return TandemResult(
        front=res_f, rear=res_r, eps_front=eps_f, eps_rear=eps_r,
        CL_total=float(CL_total), CDi_total=float(CDi_total),
        CDi_mut=float(cdi_mf + cdi_mr), CDi_mut_front=float(cdi_mf),
        CDi_mut_rear=float(cdi_mr), Sref=float(Sref),
    )


# ------------------------------------------------------------------ Tier B objective

KNOT_BOUNDS_DEG = (-6.0, 2.0)      # matches the crossover twist-knot family
SPLIT_BOUNDS = (0.3, 0.7)          # front area fraction S_f / S_total
DECALAGE_BOUNDS_DEG = (-4.0, 4.0)  # rear incidence - front incidence


@dataclass
class TandemProblem:
    """Tier B tandem L/D problem: two coupled wings, system trim to CL_target.

    Design vector (d = 2 + 2*n_knots + 2 = 10 for n_knots = 3):

        x = [taper_f, taper_r,
             front twist knots 1..k (deg, relative, root fixed at 0),
             rear  twist knots 1..k (deg, relative, root fixed at 0),
             area split s = S_f/S_total,
             decalage (deg, rear incidence - front incidence)]

    The system AoA alpha is eliminated by trimming TOTAL CL (on
    Sref = S_f + S_r) to CL_target — the coupled LLT is linear in alpha,
    so trim is closed-form from two solves (same trick as adjoint.py).
    Front-wing constant twist is absorbed by alpha (root knots fixed at 0)
    and the rear's constant offset IS the decalage: no flat ridge.
    Failure contract: exactly PENALTY (-100.0), as in objective.py.
    """

    b: float = 10.0            # the FRONT wing's span
    #: THE REAR WING'S SPAN, when it is not the front's. None (the default)
    #: flies both wings at ``b`` — bit-for-bit the published pair — and any
    #: length in metres makes the two wings the two widths they really are.
    #: A tandem's wings share a fuselage, not a span: they carry different
    #: shares of the lift at different local Reynolds numbers, and the mutual
    #: induction this module exists to model is a function of BOTH spans (the
    #: rear wing sitting inside a narrower wake, or outboard of it entirely,
    #: is the trade). Stated here it is configuration and travels verbatim —
    #: like the stagger below; under the SIZE modifier both spans are design
    #: variables instead, one row each (:func:`sizing.span_labels`).
    b_rear: float | None = None
    S_total: float = 20.0      # even split -> two Tier A wings (S=10, AR=10)
    #: THE STAGGER, IN METRES, AND IT IS THE USER'S. ``dx`` is how far aft of
    #: the front wing's quarter-chord the rear wing's sits; ``dz`` is how far
    #: above it. The defaults are the published pair's (b/2 and 0.1 b at the
    #: published 10 m span), and api.py derives the same two fractions from a
    #: span the USER types — but nothing downstream of that ever moves them
    #: again. In particular the SIZE modifier, which puts the span in the
    #: design vector, leaves them exactly where they were put: where the two
    #: surfaces sit relative to each other is a layout decision (a fuselage
    #: length, a boom, a wing box), and letting it ride on an optimiser-chosen
    #: span made the one geometric quantity this problem exists to study a
    #: quantity the optimiser chose.
    dx: float = 5.0            # rear QC downstream of front QC [m]
    dz: float = 1.0            # rear QC above front QC [m]
    N: int = 40                # LLT stations per wing
    V: float = 14.6            # Re_mac ~ 1e6 at c_mean = 1 m (even split)
    rho: float = RHO_SL
    mu: float = MU_SL
    CL_target: float = 0.5     # SYSTEM CL on Sref = S_total
    n_knots: int = 3
    cd0_extra: float = 0.0
    chord_order: int = 0       # free chord law (geometry.py), PER WING: the
    #   two wings carry independent laws, so the vector grows by 2 *
    #   chord_order — front coefficients first, then rear. 0 = straight
    #   taper on both, bit-for-bit the published problem.
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
    material: object = None   # WHAT THE WING IS BUILT OF
    #   (materials.Material, or a key from materials.MATERIALS).
    #   None = 2024-T3 aluminium, the material Raymer's correlation
    #   was regressed over, so every published run is bit-identical.
    #   It sets BOTH the allowable the spar is sized to and the
    #   factor the statistical wing weight is scaled by — the second
    #   is the one the span answer actually rides (materials.py).
    size_free: bool = False    # SIZE modifier (sizing.py): EACH WING'S
    #   SPAN becomes a design variable — two span rows, because the two wings
    #   are two wings — and, in the two-variable mode, the pair's TOTAL area
    #   with them. Each wing is
    #   weighed as its own wing at the shared gross weight (the pair really
    #   is two structures), the trim target follows the total weight, the
    #   score becomes payload L/D and a root-bending stress margin — sized on
    #   the reference area — joins as the problem's first constraint. The
    #   STAGGER does NOT travel with it: dx and dz stay the metres they were
    #   given (see the fields above).
    #   ``sizing.SIZE_MODE_WS`` is the other mode: the pair's total area
    #   follows the chosen WING LOADING through the weight loop and only the
    #   two spans are searched, which is the pairing the constraint needs —
    #   a span is a hangar, a trailer, a spar, and W/S is the mission's.
    wing_loading_Pa: float | None = None   # W/S the TOTAL area follows in
    #   the wing-loading mode. None = the loading this problem already flies,
    #   CL_target * q on S_total, so the mode opens on the published pair.
    span_bounds_m: tuple | None = None     # (min, max) BOTH span rows are
    #   searched in [m]. None = the same fractional band around b the
    #   two-variable mode uses (sizing.span_bounds). One band, not two: the
    #   two wings are alternatives for the same job, and opening the rear
    #   one narrower would answer the question the search is being asked.
    area_bounds_m2: tuple | None = None    # (min, max) AREA row of the free
    #   planform mode [m^2]; None -> the fractional band around this
    #   problem's own reference area (sizing.size_bounds)
    ws_bounds_pa: tuple | None = None      # (min, max) of the SEARCHED
    #   wing-loading row [Pa] (sizing.SIZE_MODE_WS_FREE); None ->
    #   sizing.WS_FRAC_BOUNDS around the loading this problem already flies
    wing_loading_max_Pa: float | None = None   # the MISSION's ceiling on W/S
    #   (constraint_diagram): clips that band and refuses any candidate above
    #   it, in every sized mode
    W_fixed_N: float | None = None    # non-wing weight for the size modifier
    flight_free: bool = False  # speed + altitude as design variables
    #   (geometry.py's flight modifier). The pair is trimmed as a SYSTEM to
    #   CL_target on S_total, so the trim target this candidate flies is
    #   W/(q S_total) — one weight for the aeroplane, not one per wing.
    mission: Any = None        # mission.MissionSpec carrying the design
    #   weight; defaults (flight_free only) to the weight this problem's own
    #   CL_target implies at its own speed.
    polar: Any = field(default_factory=default_polar)
    #: the REAR wing's own section, when it is not the front's. None
    #: (default) flies one section on both surfaces exactly as before —
    #: bit-for-bit the published tandem study; setting it gives the rear
    #: wing its own lift slope, zero-lift angle, stall proxy and profile
    #: drag, which is what a per-surface section choice means.
    polar_rear: Any = None
    #: SECTION THICKNESS AS A DESIGN VARIABLE, off the NACA 24XX polar family
    #: — the same modifier ``wingtail.py`` carries, and the freedom the pair
    #: was refused for no reason but a missing variable. TWO rows, one per
    #: wing, because this family already gives each surface its own span, its
    #: own chord law and its own section: a pair whose two wings must be the
    #: same thickness is a constraint nobody asked for. False (the default)
    #: leaves the vector and the answer bit-for-bit the published study.
    tc_free: bool = False
    polar_family: Any = None    # resolved from default_polar_family when free
    #: THE PAIR'S VERTICAL SURFACE, asked exactly as a wing+tail asks it.
    #:
    #: A tandem has always HAD one here in every sense but the one that
    #: costs: ``api.design_report`` sizes it against the STAGGER and writes
    #: the block, ``cad.fin_surface`` lofts and exports it, and
    #: ``flightmodel`` flies it — the whole of the pair's ``Cn_beta``
    #: (+0.1174 at the box centre) comes off it. Nothing charged its drag
    #: and no stage could choose its section, so the one surface the pair
    #: was never asked about was the one carrying its directional stability.
    #:
    #: ``fin`` is the presence key (``fin.has_fin``) and the other three are
    #: the shape the sizing law, the drag book, the report and the loft all
    #: read through ``fin.fin_law_kwargs`` / ``fin.fin_drag_kwargs`` — one
    #: reader each, so the surface charged is the surface flown.
    #:
    #: THIS MOVES EVERY PUBLISHED TANDEM NUMBER. The fin was free before and
    #: is not now: at the box centre ``cd0_fin`` is 0.000903 and L/D goes
    #: 25.687 -> 24.548 (-4.44 %). That is the correction, not a side
    #: effect — a surface that is drawn, exported and flown has to be paid
    #: for. ``fin=False`` is the pair with no vertical surface at all, and
    #: it reproduces the old numbers exactly.
    fin: bool = True
    fin_volume_coeff: float | None = None
    fin_ar: float | None = None
    fin_tc: float | None = None
    #: the BOOM aft of the rear wing [m] — ``tandemvlm.TandemVLMProblem``'s
    #: twin, and it is here for the same reason the three keys above are: the
    #: lifting-line pair never flies its fin, but it CHARGES it, and the
    #: charge reads the arm. A pair whose two engines stood the fin in
    #: different places would report one drag and fly another.
    fin_boom_m: float | None = None
    alpha_bracket_deg: tuple[float, float] = (-10.0, 15.0)
    _coupling: Coupling | None = field(default=None, repr=False)

    def __post_init__(self):
        if self.b_rear is not None and not float(self.b_rear) > 0.0:
            raise ValueError(
                f"the rear wing's span must be a positive length in metres "
                f"(got {self.b_rear!r}); leave it None for a pair whose two "
                f"wings are the same width")
        if self.tc_free:
            if self.polar_family is None:
                from .polar import default_polar_family
                self.polar_family = default_polar_family()
            if self.polar_rear is not None:
                # a CHOSEN rear section is a fixed table; a searched t/c picks
                # the table. Both at once means one of the two silently loses,
                # and which one would depend on the order of two lines here.
                raise ValueError(
                    "a tandem cannot both search its section thickness "
                    "(tc_free) and be handed a fixed rear-wing section "
                    "(polar_rear): the thickness row CHOOSES the table the "
                    "wing flies, so a stated table would be discarded without "
                    "saying so. Pick one")
        if self.flight_free and self.mission is None:
            from .mission import MissionSpec, isa_density
            q0 = 0.5 * isa_density(0.0) * self.V**2
            self.mission = MissionSpec(W_N=self.CL_target * (q0 * self.S_total),
                                       V=self.V, altitude_m=0.0)

    @property
    def b_r(self) -> float:
        """The rear wing's span [m] — its own where it has one, else the
        front's. Read everywhere the rear surface is built, so a pair with
        one span and a pair with two go through the same code."""
        return float(self.b if self.b_rear is None else self.b_rear)

    @property
    def bounds(self) -> np.ndarray:
        k = self.n_knots
        rows = np.vstack([
            np.array([geometry.TAPER_BOUNDS, geometry.TAPER_BOUNDS]),
            np.tile(KNOT_BOUNDS_DEG, (2 * k, 1)),
            np.array([SPLIT_BOUNDS, DECALAGE_BOUNDS_DEG]),
        ]).astype(float)
        if self.tc_free:
            # AHEAD of the size / flight / chord blocks, which is where
            # wingtail.py puts its own t/c row: those three are all read back
            # by counting from the END of the vector, so a family block that
            # grew at the tail would silently re-index every one of them
            rows = np.vstack([rows, np.array([geometry.TC_BOUNDS,
                                              geometry.TC_BOUNDS])])
        from .sizing import with_size_bounds, ws_of
        rows = with_size_bounds(rows, self.size_free, self.b, self.S_total,
                                span_bounds_m=self.span_bounds_m, n_spans=2,
                                area_bounds_m2=self.area_bounds_m2,
                                ws0=ws_of(self.CL_target, self.rho, self.V),
                                ws_bounds_pa=self.ws_bounds_pa,
                                wing_loading_max_Pa=self.wing_loading_max_Pa)
        rows = geometry.with_flight_bounds(rows, self.flight_free)
        chord = geometry.chord_bounds(self.chord_order, self.chord_max_frac,
                                      self.chord_law)
        if chord.size:
            rows = np.vstack([rows, chord, chord])      # front, then rear
        return rows

    @property
    def dim(self) -> int:
        return int(self.bounds.shape[0])

    @property
    def coupling(self) -> Coupling:
        if self._coupling is None:
            fs = VortexSystem(b=self.b, N=self.N, x=0.0, z=0.0)
            rs = VortexSystem(b=self.b_r, N=self.N, x=self.dx, z=self.dz)
            self._coupling = build_coupling(fs, rs)
        return self._coupling


def _twist_from_knots(knots_deg: np.ndarray, eta: np.ndarray) -> np.ndarray:
    """Relative twist knots at eta = linspace(0,1,k+1)[1:], root fixed 0 [rad]."""
    k = knots_deg.size
    knot_eta = np.linspace(0.0, 1.0, k + 1)
    return np.deg2rad(np.interp(eta, knot_eta, np.concatenate([[0.0], knots_deg])))


def evaluate_tandem(x: np.ndarray, prob: TandemProblem | None = None) -> dict:
    """Full tandem evaluation with breakdown; -100.0 contract on failure."""
    prob = prob or TandemProblem()
    x = np.asarray(x, dtype=float)
    bnds = prob.bounds
    if x.shape != (bnds.shape[0],):
        return _fail(f"design vector shape {x.shape} != ({bnds.shape[0]},)")
    if np.any(x < bnds[:, 0] - 1e-12) or np.any(x > bnds[:, 1] + 1e-12):
        return _fail("bounds violation")

    k = prob.n_knots
    taper_f, taper_r = float(x[0]), float(x[1])
    knots_f, knots_r = x[2:2 + k], x[2 + k:2 + 2 * k]
    # POSITIVE indices: the chord-law coefficients (if any) follow, so
    # x[-2:] is no longer the split/decalage pair
    split, dec_deg = float(x[2 + 2 * k]), float(x[3 + 2 * k])
    # the two thickness rows, one per wing, sit immediately after the family
    # block and ahead of size/flight/chord — read by POSITIVE index for the
    # same reason the pair above is
    tc_f = tc_r = None
    if prob.tc_free:
        tc_f, tc_r = float(x[4 + 2 * k]), float(x[5 + 2 * k])
    # the two chord laws are the TRAILING block (front then rear), which is
    # what lets the flight modifier sit between them and the family block
    m = int(prob.chord_order)
    law = prob.chord_law
    coeffs_f = geometry.ChordCoeffs(x[-2 * m:-m], law) if m else ()
    coeffs_r = geometry.ChordCoeffs(x[-m:], law) if m else ()

    # ---- size (sizing.py's modifier): the per-wing span and the pair's
    # total area. The STAGGER is left alone — it is the layout the user
    # stated, not something a candidate span may move (see the dx/dz fields)
    size_req = bool(prob.size_free)
    W_fixed_ref = None
    if size_req:
        from dataclasses import replace as _replace

        from .mission import weight_for as _weight_for0
        from .sizing import (WS_MODES, check_ar, flow_state_for,
                             resolve_spans, size_mode)
        # the payload weight is the SAME CONSTANT for every candidate — that
        # is what makes payload L/D = W_fixed/D a minimum-drag objective
        # (sizing.py). Read from the problem's OWN published area, BEFORE the
        # candidate size replaces it: taken afterwards it scaled with the
        # area, so a bigger wing carried a bigger "fixed" payload and the
        # score rewarded growing the wing.
        W_fixed_ref = (prob.W_fixed_N if prob.W_fixed_N is not None
                       else _weight_for0(prob.CL_target, prob.V,
                                         prob.S_total))
        size_kw: dict = {"wing_loading_max_Pa": prob.wing_loading_max_Pa}
        if size_mode(prob.size_free) in WS_MODES:
            # the pair's TOTAL area follows the chosen wing loading, and a
            # loading is quoted at the flow state the candidate flies in — so
            # the flight block is read first (it needs no area) and the
            # area's own fixed point is closed against that q, exactly as
            # objective.py does it. The area is then split by the SAME split
            # this candidate flies, so each wing is weighed as the wing it is:
            # its own share of the area, on its own span.
            rho_ws, V_ws = flow_state_for(
                x, mission=prob.mission, flight_free=prob.flight_free,
                n_trailing=2 * m, rho=prob.rho, V=prob.V)
            size_kw.update(
                # ignored by the SEARCHED-loading mode, which reads its own
                # row out of the vector (sizing.resolve_spans)
                wing_loading_Pa=(prob.wing_loading_Pa
                                 if prob.wing_loading_Pa is not None
                                 else prob.CL_target * 0.5 * prob.rho
                                 * prob.V ** 2),
                W_fixed_N=W_fixed_ref,
                taper=0.5 * (taper_f + taper_r), tc=0.12,
                q_Pa=0.5 * rho_ws * V_ws ** 2,
                area_fracs=(split, 1.0 - split))
        try:
            (b_front, b_rear), S_use = resolve_spans(
                x, prob.size_free, prob.flight_free, 2 * m, n_spans=2,
                **size_kw)
        except (ValueError, OverflowError) as exc:
            return _fail(f"size: {exc}")
        # validated on ONE wing's aspect ratio: the pair's total area would
        # report half the AR each surface actually flies at. Both wings are
        # checked — with two span rows either one can leave the band, and an
        # equal-span pair makes the same call twice for the same answer
        for b_use in (b_front, b_rear):
            reason = check_ar(b_use, 0.5 * S_use)
            if reason is not None:
                return _fail(f"size: {reason}")
    # ...and the ASPECT-RATIO LIMIT THE USER SET. A different question
    # from the validity band, so it is asked whether or not the size is
    # a design variable: a limit that only applied to searched sizes
    # would go quiet exactly when the user typed the number it is about.
    if prob.ar_limits is not None:
        from .sizing import check_ar_limit as _check_ar_limit
        # per WING, as the validity gate above is: the pair's total area
        # reports half the aspect ratio either surface actually flies at
        S_pair = float(S_use if prob.size_free else prob.S_total)
        for b_wing in ((b_front, b_rear) if prob.size_free
                       else (float(prob.b), prob.b_r)):
            why_ar = _check_ar_limit(b_wing, 0.5 * S_pair, prob.ar_limits)
            if why_ar is not None:
                return _fail(f"size: {why_ar}")
    if prob.size_free:
        prob = _replace(prob, b=b_front, b_rear=b_rear, S_total=S_use,
                        size_free=False, _coupling=None)

    # ---- flight state (geometry.py's flight modifier): the pair ahead of
    # the two chord blocks rebuilds the flow state and the SYSTEM trim target
    flight = geometry.flight_from_x(x, prob.flight_free, 2 * m)
    if flight is not None:
        from dataclasses import replace as _replace

        from .mission import flight_state
        try:
            prob = _replace(prob, flight_free=False, mission=None,
                            _coupling=prob.coupling,
                            **flight_state(prob.mission, flight[0], flight[1],
                                           prob.S_total))
        except ValueError as exc:      # water medium, or ISA validity
            return _fail(f"flight state: {exc}")

    S_f, S_r = split * prob.S_total, (1.0 - split) * prob.S_total
    sized = None
    if size_req:
        from dataclasses import replace as _replace2

        from .sizing import sized_state
        try:
            sized = sized_state(
                W_fixed_N=W_fixed_ref,
                b=prob.b, S=prob.S_total, taper=0.5 * (taper_f + taper_r),
                tc=0.12, q_Pa=0.5 * prob.rho * prob.V**2,
                wing_areas=(S_f, S_r),
                # each wing weighs — and is stressed — as the wing it is: two
                # span rows mean the rear one can be the shorter structure
                wing_spans=(prob.b, prob.b_r),
                wing_loading_max_Pa=prob.wing_loading_max_Pa,
                material=prob.material)
        except (ValueError, RuntimeError, OverflowError) as exc:
            return _fail(f"size: {exc}")
        prob = _replace2(prob, CL_target=sized.CL_target,
                         _coupling=prob.coupling)
    try:
        wing_f = geometry.Wing(b=prob.b, S=S_f, taper=taper_f,
                               chord_limits=prob.chord_limits,
                               chord_coeffs=coeffs_f)
        wing_r = geometry.Wing(b=prob.b_r, S=S_r, taper=taper_r,
                               chord_limits=prob.chord_limits,
                               chord_coeffs=coeffs_r)
    except ValueError as exc:          # collapsed chord law on either wing
        return _fail(f"planform: {exc}")
    # each wing is sampled on ITS OWN cosine stations: they are the stations
    # of that lifting line, and with two spans they are two different sets of
    # y. (Equal spans give the same array twice — the published pair.) The
    # twist knots are placed on each wing's own eta for the same reason: a
    # knot at eta = 0.5 is half way out THAT wing.
    yv = cosine_stations(prob.N, prob.b)[1]
    yv_r = cosine_stations(prob.N, prob.b_r)[1]
    eta = np.abs(2.0 * yv / prob.b)
    eta_r = np.abs(2.0 * yv_r / prob.b_r)
    c_f, c_r = wing_f.chord(yv), wing_r.chord(yv_r)
    tw_f = _twist_from_knots(knots_f, eta)
    tw_r = _twist_from_knots(knots_r, eta_r) + np.deg2rad(dec_deg)

    pol = prob.polar
    pol_r = pol if prob.polar_rear is None else prob.polar_rear
    if tc_f is not None:
        # each wing flies the family member ITS OWN thickness row picks (exact
        # member or a blend between two) — the section is per surface here for
        # the same reason the span and the chord law are
        try:
            pol = prob.polar_family.at(tc_f)
            pol_r = prob.polar_family.at(tc_r)
        except ValueError as exc:
            return _fail(f"polar family: {exc}")
    coupling = prob.coupling

    def _solve(alpha_rad: float) -> TandemResult:
        return solve_tandem(
            Surface(b=prob.b, c=c_f, alpha_geo=alpha_rad + tw_f,
                    a=pol.a_lin, alpha_L0=pol.alpha_L0, x=0.0, z=0.0),
            Surface(b=prob.b_r, c=c_r, alpha_geo=alpha_rad + tw_r,
                    a=pol_r.a_lin, alpha_L0=pol_r.alpha_L0,
                    x=prob.dx, z=prob.dz),
            V=prob.V, Sref=prob.S_total, coupling=coupling,
        )

    try:
        # CL_total is exactly affine in alpha (linear system, alpha only in RHS)
        r0, r1 = _solve(0.0), _solve(np.deg2rad(1.0))
        slope = (r1.CL_total - r0.CL_total) / np.deg2rad(1.0)
        if abs(slope) < 1e-9:
            return _fail("degenerate CL_alpha")
        alpha = (prob.CL_target - r0.CL_total) / slope
        lo, hi = np.deg2rad(prob.alpha_bracket_deg[0]), np.deg2rad(prob.alpha_bracket_deg[1])
        if not (lo <= alpha <= hi):
            return _fail("untrimmable: alpha outside bracket",
                         alpha_deg=float(np.rad2deg(alpha)))
        res = _solve(alpha)
    except np.linalg.LinAlgError as exc:
        return _fail(f"solver failure: {exc}")

    # stall / extrapolation proxy on BOTH wings
    ae_f = np.rad2deg(res.front.alpha_eff_y)
    ae_r = np.rad2deg(res.rear.alpha_eff_y)
    lo_p, hi_p = pol.alpha_valid
    lo_r, hi_r = pol_r.alpha_valid
    if (ae_f.min() < lo_p or ae_f.max() > hi_p
            or ae_r.min() < lo_r or ae_r.max() > hi_r):
        return _fail(
            "effective AoA outside polar validity (stall/extrapolation proxy)",
            alpha_eff_range=(float(min(ae_f.min(), ae_r.min())),
                             float(max(ae_f.max(), ae_r.max()))),
        )

    CDp_f = float(np.trapezoid(pol.cd(ae_f) * c_f, yv) / res.front.S)
    CDp_r = float(np.trapezoid(pol_r.cd(ae_r) * c_r, yv_r) / res.rear.S)
    CDp = (CDp_f * res.front.S + CDp_r * res.rear.S) / res.Sref
    # ...AND THE FIN'S PARASITE DRAG, the way ``tail.evaluate_tail`` charges
    # it. The pair's arm is the station ``fin.tandem_fin_station`` puts the
    # surface at — ON the rear wing, the one structure a pair already has —
    # and that one function is what ``api.design_report`` sizes, reports and
    # lofts the fin at too, so the surface paid for here is the surface
    # drawn, exported and flown. That identity was broken once, while this
    # line did not exist at all (cd0_fin charged nowhere, 4.44 % of L/D at
    # the box centre).
    #
    # A stagger of zero has no arm and therefore no volume-coefficient fin;
    # ``fin.size_fin`` says so by raising, and the report's own block is
    # absent for the same candidate, so the two agree about a pair that
    # carries none.
    cd0_fin = 0.0
    if _fin.has_fin(prob):
        # IMPORTED HERE, not at the top: ``tail`` imports THIS module (it
        # reuses the pair's coupled solve for its own two surfaces), so a
        # module-level import back the other way is a cycle — and it is the
        # kind that only fails on some import ORDERS, which is worse than
        # one that always does.
        from . import tail as _tail
        try:
            arm, _z = _fin.tandem_fin_station(prob.dx, prob.dz,
                                              prob.fin_boom_m)
            cd0_fin = float(_tail.vtail_cd0(
                res.Sref, arm, b=prob.b, S=res.Sref,
                V=prob.V, rho=prob.rho, mu=prob.mu,
                **_fin.fin_drag_kwargs(prob)))
        except ValueError:
            cd0_fin = 0.0
    CD = res.CDi_total + CDp + prob.cd0_extra + cd0_fin
    LoD = res.CL_total / CD
    if not np.isfinite(LoD):
        return _fail("non-finite L/D")

    flown = ({"V": float(prob.V), "altitude_m": float(flight[1]),
              "rho": float(prob.rho), "mu": float(prob.mu)}
             if flight is not None else {})
    sized_extra: dict = {}
    if sized is not None:
        q = 0.5 * prob.rho * prob.V**2
        f = sized.payload_lod(q, CD)
        if not np.isfinite(f):
            return _fail("non-finite payload L/D")
        sized_extra = {**sized.report(), "f": float(f),
                       "g": float(sized.g_sigma),
                       "D_N": float(q * sized.S * CD)}
    return {
        **flown, **sized_extra,
        "feasible": True, "reason": "",
        "score": float(LoD if sized is None else sized_extra["f"]),
        "LoD": float(LoD),
        "CL_total": res.CL_total, "CL_front": res.front.CL, "CL_rear": res.rear.CL,
        "lift_share_front": float(res.front.CL * res.front.S
                                  / (res.CL_total * res.Sref)),
        "CDi_total": res.CDi_total,
        "CDi_self_front": res.front.CDi, "CDi_self_rear": res.rear.CDi,
        "CDi_mut": res.CDi_mut, "CDi_mut_front": res.CDi_mut_front,
        "CDi_mut_rear": res.CDi_mut_rear,
        "CDp": float(CDp), "CDp_front": CDp_f, "CDp_rear": CDp_r,
        "cd0_extra": prob.cd0_extra, "cd0_fin": float(cd0_fin),
        "CD": float(CD),
        "e_front": res.front.e, "e_rear": res.rear.e,
        "alpha_deg": float(np.rad2deg(alpha)),
        "eps_rear_mean_deg": float(np.rad2deg(np.mean(res.eps_rear))),
        "eps_front_mean_deg": float(np.rad2deg(np.mean(res.eps_front))),
        "S_front": res.front.S, "S_rear": res.rear.S, "Sref": res.Sref,
        # the LAYOUT this candidate flew: reported because it is the pair's
        # defining geometry and it is the user's to state — a result that did
        # not say where the second wing was left the reader to assume. ``b``
        # stays the FRONT wing's span (every reader of this key means that);
        # the two named keys beside it are what a pair with two spans needs
        "b": float(prob.b), "b_front": float(prob.b), "b_rear": float(prob.b_r),
        "dx": float(prob.dx), "dz": float(prob.dz),
        "polar": getattr(pol, "name", "unknown"),
        "polar_rear": getattr(pol_r, "name", "unknown"),
        # None on a fixed-section pair, so a stored breakdown says whether the
        # thickness was searched rather than leaving it to be inferred
        "tc_front": tc_f, "tc_rear": tc_r,
        "tandem": res,
    }


def objective_tandem(x: np.ndarray, prob: TandemProblem | None = None) -> float:
    """Scalar objective (MAXIMISE): system L/D, or PENALTY on any failure."""
    out = evaluate_tandem(x, prob)
    return float(out["score"]) if out["feasible"] else PENALTY


#: finite infeasible margin reported on solver failure by the SIZED variant
G_FAIL = -1.0


def fg_tandem(x: np.ndarray, prob: TandemProblem | None = None
              ) -> tuple[float, float]:
    """Constrained-harness callable for the SIZE modifier: (payload L/D,
    signed root-bending stress margin). The unsized pair is unconstrained and
    uses :func:`objective_tandem`."""
    out = evaluate_tandem(x, prob)
    if not out["feasible"]:
        return PENALTY, G_FAIL
    if "g" not in out:
        raise ValueError("fg_tandem() is for the size-modifier variant — the "
                         "plain tandem problem has no constraint")
    return float(out["score"]), float(out["g"])
