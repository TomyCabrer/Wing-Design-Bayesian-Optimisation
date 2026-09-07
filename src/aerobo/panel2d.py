"""2-D multi-body panel method (Hess & Smith) — the chordwise physics the
lattice cannot see.

Why this module exists
----------------------
Everything else in the package resolves the SPANWISE problem: llt.py,
tandem.py and vlm.py all carry one chordwise panel per strip, so a section
enters as a polar (a table) and a surface's chordwise pressure distribution
is never computed. That is the right economy for a wing. It is exactly
wrong for a multi-element section: a race-car rear wing's flap sits in the
main element's own downwash and its own upwash feeds back, at a gap of a
few per cent of chord — a CHORDWISE interaction between two bodies at the
SAME spanwise station. One chordwise panel per strip cannot represent two
bodies at one station at all, and no polar table can, because the flap's
polar depends on the slot it is sitting in. Either the slot is resolved
here, in 2-D, with two bodies, or it is not resolved.

The method is Hess & Smith's: constant-strength SOURCE panels (one unknown
density per panel) plus ONE constant-strength VORTEX density shared by all
panels of a body (one unknown per body, closed by that body's Kutta
condition). The sources carry thickness/displacement, the single vortex
carries circulation, and the split is what makes the system well posed for
several bodies at once: N_i + 1 unknowns and N_i + 1 equations per body,
with every body's panels seeing every other body's panels through the same
dense matrix. There is no iteration and no relaxation between bodies — the
coupling is linear and it is solved in one shot, the same discipline as
tandem.py's single (N_w + N_t)^2 system.

Panel kernels — DERIVED (Katz & Plotkin's straight-segment integrals; the
package already cites Katz & Plotkin for the VLM horseshoe kernel)
--------------------------------------------------------------------------
Work in a panel-local right-handed frame (xi, zeta): the panel lies on
zeta = 0 from xi = 0 to xi = L, xi_hat along the panel's own traversal
direction and zeta_hat = rot90(xi_hat) (i.e. xi_hat rotated +90 deg).
Write s1 = xi, s2 = xi - L for the distances of the field point ahead of
the two ends, r1^2 = s1^2 + zeta^2, r2^2 = s2^2 + zeta^2, and
theta_k = atan2(zeta, s_k).

SOURCE, constant density sigma (volume flux per unit length). A point
source of strength sigma dxi0 at (xi0, 0) has potential
(sigma dxi0 / 2pi) ln r, so

    u = d/dxi  int = (sigma/2pi) int_0^L s/(s^2+zeta^2) dxi0
      = (sigma/4pi) ln(r1^2 / r2^2)                              == sigma * U
    w = d/dzeta int = (sigma/2pi) int_0^L zeta/(s^2+zeta^2) dxi0
      = (sigma/2pi) (theta2 - theta1)                            == sigma * W

(substitute s = xi - xi0, ds = -dxi0; the first integral is
(1/2) ln(s^2+zeta^2) evaluated between the ends, the second is
atan(s/zeta), and atan(s/zeta) = pi/2 - atan2(zeta, s) for zeta > 0 turns
it into the subtended-angle form, which is the one that stays correct on
both sides of the sheet.)

VORTEX, constant density gamma. Sign convention fixed HERE and used
everywhere below: gamma is circulation per unit length, POSITIVE
COUNTER-CLOCKWISE (mathematically positive) in the global plane. A point
vortex dGamma = gamma dxi0 at (xi0, 0) induces
u = -dGamma zeta / (2pi r^2), w = +dGamma s / (2pi r^2), so the same two
integrals return, swapped:

    u = -gamma (theta2 - theta1) / 2pi = -gamma * W
    w = +gamma ln(r1^2/r2^2) / 4pi     = +gamma * U

i.e. the vortex field is the source field rotated by 90 deg, which is the
Cauchy-Riemann statement that ln z and i ln z are conjugate. (Katz &
Plotkin print the vortex panel with the opposite sign because their
positive Gamma is clockwise; the formulae here are self-derived and
self-consistent, and the sign is gated by the tests, not by a citation.)

Both fields are INVARIANT under reversing the panel's node order: under
a <-> b one has xi -> L - xi, zeta -> -zeta, hence U -> -U, W -> -W and
xi_hat -> -xi_hat, zeta_hat -> -zeta_hat, so sigma*(U xi_hat + W zeta_hat)
and gamma*(-W xi_hat + U zeta_hat) are unchanged. That invariance is what
makes the wall image below a two-line change instead of a sign minefield:
sigma and gamma are densities on a segment, not on an ordered segment.

Self-influence — DERIVED, not quoted
------------------------------------
Control points sit at panel midpoints, ON the panel, so the kernels must be
evaluated as a one-sided limit. At the midpoint of its own panel
s1 = +L/2, s2 = -L/2, so r1 = r2 and U = 0 exactly. Approaching from the
zeta > 0 side, theta1 -> 0 and theta2 -> +pi, so W -> +1/2; from the
zeta < 0 side, W -> -1/2. Hence

    a source panel induces sigma/2 NORMAL to itself, away from the side it
      is approached from, and nothing tangentially;
    a vortex panel induces gamma/2 TANGENTIALLY, and nothing normally,
    with the tangential jump u(below) - u(above) = gamma — the vortex-sheet
      jump, which is just Stokes' theorem on a flat rectangle straddling
      the sheet.

The 0.5s are therefore not a convention to be looked up but the value of
atan2 across a cut, and the SIDE matters: the limit must be taken from the
fluid, i.e. from OUTSIDE the body. Which side that is follows from the
loop orientation, next. In floating point the on-panel entries evaluate to
atan2(+-1e-17, -L/2), whose sign is noise, so the diagonal is written in
by hand rather than computed.

Loop orientation — a CHECKED convention, because a reversed loop flips
every normal and silently returns the negative of the answer
--------------------------------------------------------------------------
Bodies are given as the package's own aerofoil loop (airfoil.naca4_coords,
airfoil.cst_coords, "XFOIL LOAD order": trailing edge -> upper surface ->
leading edge -> lower surface -> trailing edge). MEASURED on
naca4_coords("2412"): the shoelace signed area is +0.0814 (and 0.0814 is
the right area for a 12 % section of unit chord), so that order is
COUNTER-CLOCKWISE. This module requires it and refuses the reverse
(:func:`check_body`), rather than silently re-orienting, because a caller
who hands a clockwise loop has a sign error somewhere upstream that a
silent fix would hide.

For a counter-clockwise loop with tangent t = (b - a)/L the OUTWARD normal
is n = (t_y, -t_x): check on the upper surface, where the traversal runs
forward (t = (-1, 0)) and the body is below, giving n = (0, +1), which
points away from the body. The right-handed local frame used by the
kernels has zeta_hat = rot90(t) = (-t_y, t_x) = -n, so local zeta > 0 is
INSIDE the body and the fluid-side limit is zeta -> 0^-, i.e. W_ii = -1/2.
Projected onto the outward normal that returns the textbook +sigma/2
outward self-induction (v.n = U (t_j.n_i) - W (n_j.n_i) = +1/2 on the
diagonal), which is the arithmetic check that the frame is consistent.

Trailing edge, and what closes the loop
---------------------------------------
The CLOSING segment (last node -> first node) is the trailing-edge BASE.
The package's sharp-TE arrays repeat the TE node exactly (measured:
naca4_coords and cst_coords with dz_te = 0 both return
coords[0] == coords[-1]), so the base is degenerate and the duplicate node
is dropped, leaving the two panels that touch the TE node as the Kutta
pair (first, last). With dz_te > 0 the arrays do NOT repeat (measured), the
closing segment is a real base panel carrying its own source strength, and
the Kutta pair is (first, last-but-one) — the two panels that touch the TE
along the SURFACE in both cases.

Kutta condition
---------------
One per body. Equal static pressure on the two sides of the trailing edge
means equal SPEED there, and the two TE panels are traversed in opposite
senses relative to the flow (the first leaves the TE forward along the
upper surface, the last arrives at it along the lower), so the condition is

    v_t(first panel) + v_t(last panel) = 0,

which is invariant under reversing the whole loop (both tangents flip).
This is the discrete statement that the flow leaves the trailing edge
smoothly; it is a constraint on the SOLUTION, so it enters as an extra row,
and the extra unknown it pays for is that body's gamma.

Circulation and force — the multi-body trap
-------------------------------------------
A source distribution has a single-valued potential, so it contributes
nothing to a contour integral around the body; the body's circulation is
carried entirely by its vortex density and is EXACT (not a quadrature):

    Gamma_i = gamma_i * perimeter_i          (positive counter-clockwise).

Kutta-Joukowski then gives L = -rho V Gamma: a positive counter-clockwise
circulation with the stream along +x induces -x velocity ABOVE the body and
+x below, so the fast side is underneath and the force is DOWN. Positive
lift means negative counter-clockwise circulation. That is reported as
``cl_gamma``.

It is NOT the reported ``cl`` when there is more than one body. For a
single body in an unbounded stream, force = rho V Gamma is exact; for two
bodies it is not, because each body also feels the other's non-uniform
field (the Lagally/Blasius interaction terms). The reported ``cl``, ``cd``
and ``cm`` are therefore SURFACE PRESSURE INTEGRALS of the panel solution,

    dF_i / q = -Cp_i n_i L_i,  Cp_i = 1 - (v_t,i / V)^2,

which is exactly what the discrete solution says the body feels, and which
also gives a free error estimate: d'Alembert requires the drag summed over
all bodies to vanish, so ``PanelSolution.cd`` is a pure discretisation
residual and is reported rather than hidden. For an isolated body ``cl``
and ``cl_gamma`` agree to the discretisation error (MEASURED: 0.02 % at
N = 200 on the Joukowski gate).

Moment sign — DERIVED
---------------------
In this 2-D frame x is downstream and y is up, so the out-of-plane axis
z = x_hat cross y_hat points to the reader. A positive (counter-clockwise)
z-moment carries a point at (-c, 0) — the nose — to (-c cos p, -c sin p),
i.e. DOWNWARD. Nose-up, the conventional positive sense for cm, is
therefore the NEGATIVE z-moment, and ``cm`` returned here is
-(M_z)/(q c_ref^2). Gated, independently of any tolerance, by the classical
identity cm_LE = -cl/4 for a symmetric section (its aerodynamic centre is
at the quarter chord) and by cm_c/4 < 0 for positive camber.

Wall image (diagnostic only — cascade.py must NOT use it)
---------------------------------------------------------
``image=Wall(y)`` adds an explicit mirrored panel system. The signs are
derived from the wall condition exactly as ground_effect.py derives the
3-D one, and they come out the same way:

  * A solid wall carries no through-flow: v.n = 0 on y = y_wall.
  * A point SOURCE q at height h has v_y(x, 0) = -(q/2pi) h/(x^2+h^2);
    its mirror at -h contributes +(q_img/2pi) h/(x^2+h^2). These cancel
    iff q_img = +q. The image source keeps its sign (the potential is
    symmetric across the wall).
  * A point VORTEX Gamma at +h has v_y(x, 0) = Gamma x / (2pi (x^2+h^2));
    the mirror at -h gives Gamma_img x / (2pi (x^2+h^2)). These cancel iff
    Gamma_img = -Gamma. The image vortex flips sign — the same
    OPPOSITE-circulation rigid-wall image as ground_effect.GROUND_IMAGE_SIGN
    = -1, and the opposite of the free-surface (hydrofoil.py) image.

Because the kernels are node-order invariant (above), the image system is
literally the mirrored node coordinates carrying (+sigma_j, -gamma_i); no
re-ordering, no per-panel sign bookkeeping. The image panels carry no
control points and no equations: their strengths are slaved to the real
ones, so the system size is unchanged, exactly as the ground image folds
into tandem.py's matrix without new unknowns.

The freestream must be PARALLEL to the wall, and this is enforced
(alpha = 0 with an image, refused otherwise) rather than assumed: mirroring
cancels the bodies' own normal velocity on the plane but does nothing to
the freestream's V sin(alpha), so an inclined stream through a "wall" is
not a wall at all. Incidence is expressed by rotating the GEOMETRY. This
is why the option is a diagnostic: cascade.py resolves a car wing whose
3-D lattice already carries the ground image, and switching this on as well
would count ground effect twice.

What is NOT modelled
--------------------
* VISCOSITY, entirely. No boundary layer, no displacement thickness, no
  separation, no viscous decambering. For a slotted flap this is the
  central limitation and its direction is known: the whole engineering
  purpose of a slot is to feed high-energy air into the flap's boundary
  layer, and an inviscid method sees only the CIRCULATION half of that
  (the dumping/circulation effects of A. M. O. Smith's classification).
  The bias is OPTIMISTIC: cl and the leading-edge suction peaks are upper
  bounds, and the gap at which a real slot stalls cannot be found here.
  Cp_min is reported precisely so a caller can bound the peak externally.
* WAKES. The Kutta condition implies a wake but no wake panels are shed,
  so an upstream element's wake does not pass over a downstream element.
  Also optimistic for a two-element system.
* Compressibility (incompressible Laplace; apply Prandtl-Glauert outside if
  the caller needs it), unsteadiness, and 3-D — this is a SECTION method,
  and the finite-span downwash belongs to the lattice that calls it.
* A wall boundary layer under the image (rigid, slip wall), as in
  ground_effect.py.
* Trailing-edge base pressure: a dz_te > 0 base panel is solved inviscidly,
  which puts near-stagnation pressure on the base and returns a spurious
  thrust of order the base area. It shows up in ``cd`` and is not corrected.

MEASURED — panel-count convergence (the grid study, in the style of
tail.py's)
-------------------------------------------------------------------
Cambered Joukowski (eps/a = 0.10, mu_y/a = 0.05, alpha = 5 deg) against the
exact conformal solution; N is the panel count, err(cl) is |cl - cl_exact|
/ cl_exact, err(Cp) is the max |Cp - Cp_exact| over all control points
excluding the two panels touching the trailing-edge cusp (where the exact
solution is 0/0 and the panel surface is chordally offset from the true
one), and "order" is the local log-log slope of err(cl) between successive
rows:

      N     cl        err(cl)      order      max err(Cp)
     ---   -------   ---------   --------   -------------
      25   1.10633   1.028e-02      -          1.056e-01
      50   1.11553   1.812e-03     2.50        2.878e-02
     100   1.11731   2.331e-04     2.96        7.343e-03
     200   1.11757   1.669e-05     3.80        1.844e-03
     400   1.11759   1.184e-06     3.82        4.616e-04
     800   1.11759   8.222e-07     0.53        1.155e-04

cl converges at roughly 3rd order until it hits the 1e-6 floor set by the
trailing-edge cusp panels; surface Cp converges at 2nd order (the ratios
are 3.67 / 3.92 / 3.98 / 3.99 = 4, clean second order) with no floor in
this range. 200 panels buys cl to 2e-5 relative and Cp to 2e-3 absolute,
which is the recommended working count and the one the cost below is
quoted at.

MEASURED — cost (the reason the kernel build is vectorised)
-----------------------------------------------------------
Two 100-panel bodies (200 panels, 202 unknowns), on the development
machine (Darwin 24.5.0, .venv numpy):

    Body construction (validation + self-intersection scan)  0.29 ms
    solve()                                                  2.21 ms
    solve() with image=Wall(...)                             3.24 ms

so a 2-element section costs ~2.5 ms, i.e. an optimiser can afford
O(10^4) of them. The influence build is one (P, N) broadcast per kernel:
a Python double loop over the 40 000 panel pairs was measured at 0.55 s,
250x slower, and is what the vectorisation buys.

References
----------
Hess, J. L., and Smith, A. M. O., "Calculation of Potential Flow About
    Arbitrary Bodies," Progress in Aerospace Sciences, Vol. 8, 1967,
    pp. 1-138.
Katz, J., and Plotkin, A., "Low-Speed Aerodynamics," 2nd ed., Cambridge
    University Press, 2001 (constant-strength source and vortex panels,
    Ch. 10-11; the same text the VLM kernel cites).
Smith, A. M. O., "High-Lift Aerodynamics," Journal of Aircraft, Vol. 12,
    No. 6, 1975, pp. 501-530 (the five slot effects; only the inviscid
    ones are in scope here).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

_TWO_PI = 2.0 * np.pi
_FOUR_PI = 4.0 * np.pi

#: Fewest panels a body may be given with. Not a taste: MEASURED on the
#: symmetric Joukowski gate, an 8-panel loop already reproduces the exact
#: cl to 6.4 % and a 6-panel one to 12 %, while below that the "aerofoil"
#: is a quadrilateral whose leading edge carries no curvature at all and
#: whose Cp array cannot resolve a suction peak. It is a refusal threshold,
#: not an accuracy claim — see the convergence table for the working count.
MIN_PANELS = 8

#: Largest trailing-edge base, as a fraction of the loop's x-extent, that this
#: method will solve — and it is EFFECTIVELY ZERO, because a base panel is not
#: something this formulation can carry.
#:
#: MEASURED, and the reason the number is what it is. A blunt base was first
#: admitted here up to 0.05 chords, panelled as one more solid panel with the
#: Kutta pair taken either side of it. Against XFOIL's inviscid solve on
#: BIT-IDENTICAL coordinates (NACA 2412, XFOIL's own 160-node panelling, base
#: 0.00252 c — a quarter of one per cent) that costs
#:
#:     alpha      0      2      4      6      8   deg
#:     XFOIL   0.2554 0.4968 0.7376 0.9775 1.2163
#:     here    0.2322 0.4562 0.6796 0.9022 1.1238     -7.6 % at alpha 4
#:
#: and — the part that says it is a formulation error and not a discretisation
#: one — REFINING MAKES IT WORSE: on a NACA 2412 at alpha 4 the deficit runs
#: 4.9 % at 100 panels, 6.5 % at 400, 8.9 % at 1600, because the base length is
#: fixed by the section while every other panel shrinks around it. The same
#: section with the base CLOSED agrees with XFOIL on identical coordinates to
#: 0.15 % in the pressure lift at alpha 4 (0.74248 against 0.74140), which is
#: the accuracy this method is supposed to have and does have.
#:
#: So a blunt section is CLOSED before it is flown (:func:`close_trailing_edge`)
#: and a base thicker than this is refused with the reason. Closing is not free
#: and its price is measured too: XFOIL puts the sharp 2412 at 0.7414 against
#: the blunt 0.7376 at alpha 4, i.e. +0.5 % — one thirteenth of the error it
#: removes, and a change to the SECTION, which is honest, rather than a change
#: to the answer for a section nobody flew.
#:
#: This module's own choice and its own measurement, not a literature number.
TE_GAP_MAX_FRAC = 1e-3

#: Largest included wedge angle at the trailing edge. Its real job is to
#: catch a loop that does not START at the trailing edge: at a round leading
#: edge the two surface panels leaving the node point nearly opposite ways
#: (wedge ~ 180 deg), at any trailing edge they converge (10-20 deg on real
#: sections). 90 deg is deliberately generous.
TE_WEDGE_MAX_RAD = 0.5 * np.pi

#: Relative tolerance used to decide "the same point" and "zero length",
#: scaled by the loop's own extent.
_GEOM_TOL = 1e-9


# ----------------------------------------------------------------- geometry


@dataclass(frozen=True, eq=False)
class PanelGeom:
    """Panel geometry of one closed loop (or of a mirrored image of one).

    ``a``/``b`` are the (N, 2) start and end nodes, ``cp`` the control
    points (midpoints), ``t`` the unit tangents along the traversal, ``n``
    the unit normals (t_y, -t_x) — OUTWARD for the counter-clockwise loop
    this module requires — and ``L`` the panel lengths.
    """

    a: np.ndarray
    b: np.ndarray
    cp: np.ndarray
    t: np.ndarray
    n: np.ndarray
    L: np.ndarray

    @property
    def n_panels(self) -> int:
        return int(self.L.size)

    @property
    def perimeter(self) -> float:
        return float(self.L.sum())


def _geom_from_ab(a: np.ndarray, b: np.ndarray) -> PanelGeom:
    d = b - a
    L = np.hypot(d[:, 0], d[:, 1])
    t = d / L[:, None]
    n = np.column_stack([t[:, 1], -t[:, 0]])
    return PanelGeom(a=a, b=b, cp=0.5 * (a + b), t=t, n=n, L=L)


def _geom_from_nodes(nodes: np.ndarray) -> PanelGeom:
    return _geom_from_ab(nodes, np.roll(nodes, -1, axis=0))


def _mirror_geom(geom: PanelGeom, y_wall: float) -> PanelGeom:
    """Mirror a panel system in the line y = ``y_wall``.

    Node ORDER is left alone: the kernels are node-order invariant (module
    docstring), so the mirrored segments carry the mirrored field with no
    further bookkeeping. The mirrored loop is clockwise and its ``n`` points
    inward; that is harmless because no control point ever lies on an image
    panel (bodies crossing the wall are refused) and the projections only
    use the right-handed (t, -n) pairing, never the outwardness.
    """
    flip = np.array([1.0, -1.0])
    off = np.array([0.0, 2.0 * y_wall])
    return _geom_from_ab(geom.a * flip + off, geom.b * flip + off)


def _signed_area(nodes: np.ndarray) -> float:
    """Shoelace signed area; positive counter-clockwise."""
    x, y = nodes[:, 0], nodes[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _self_intersects(nodes: np.ndarray) -> bool:
    """True if any two non-adjacent panels PROPERLY cross.

    Strict-sign orientation test, so segments that merely touch (a cusped
    trailing edge, where the last upper and last lower panels come
    arbitrarily close) are not flagged. Vectorised over the (N, N) pair
    grid: N = 200 is 4e4 pairs, microseconds.
    """
    n = nodes.shape[0]
    p = nodes
    q = np.roll(nodes, -1, axis=0)
    r = q - p                                     # (N, 2) segment vectors

    # cross(r_i, p_j - p_i) for every ordered pair (i, j)
    dpx = p[None, :, 0] - p[:, None, 0]
    dpy = p[None, :, 1] - p[:, None, 1]
    dqx = q[None, :, 0] - p[:, None, 0]
    dqy = q[None, :, 1] - p[:, None, 1]
    d1 = r[:, None, 0] * dpy - r[:, None, 1] * dpx      # i x (p_j - p_i)
    d2 = r[:, None, 0] * dqy - r[:, None, 1] * dqx      # i x (q_j - p_i)

    cross = (d1 * d2 < 0.0) & (d1.T * d2.T < 0.0)

    idx = np.arange(n)
    sep = np.abs(idx[:, None] - idx[None, :])
    adjacent = (sep <= 1) | (sep >= n - 1)              # incl. the wrap pair
    return bool(np.any(cross & ~adjacent))


def _prepare_loop(x, y, min_panels: int = MIN_PANELS):
    """Validate a closed loop and return ``(nodes, kutta, reason)``.

    On success ``reason`` is None, ``nodes`` is the (N, 2) node array with
    any duplicated closing node removed, and ``kutta`` is the index pair of
    the two panels that touch the trailing edge along the surface. On
    failure the first two are None and ``reason`` is the refusal.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size != y.size:
        return None, None, f"x and y must have equal length, got {x.size} and {y.size}"
    if x.size < 4:
        return None, None, f"a closed loop needs at least 4 points, got {x.size}"
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        return None, None, "loop coordinates contain a non-finite value"

    nodes = np.column_stack([x, y])
    scale = float(max(np.ptp(x), np.ptp(y)))
    if not scale > 0.0:
        return None, None, "loop has zero extent"
    tol = _GEOM_TOL * scale

    # Which segment closes the loop decides where the Kutta pair is, so the
    # two spellings of "closed" are separated HERE and nowhere else. If the
    # trailing-edge node is repeated (the package's sharp-TE arrays), the
    # base is degenerate: drop the repeat and the closing segment is the
    # last SURFACE panel. If it is not repeated (dz_te > 0), the closing
    # segment IS the base and the last surface panel is the one before it.
    explicitly_closed = bool(np.hypot(*(nodes[0] - nodes[-1])) <= tol)
    if explicitly_closed:
        nodes = nodes[:-1]
    n = nodes.shape[0]
    if n < min_panels:
        return None, None, (
            f"loop has {n} panels, fewer than the {min_panels} this method "
            "will solve")

    seg = np.roll(nodes, -1, axis=0) - nodes
    L = np.hypot(seg[:, 0], seg[:, 1])
    short = np.flatnonzero(L <= tol)
    if short.size:
        return None, None, (
            f"panel {int(short[0])} has zero length (repeated node at index "
            f"{int(short[0])})")

    chord = float(np.ptp(nodes[:, 0]))
    gap = 0.0 if explicitly_closed else float(L[-1])
    if gap > TE_GAP_MAX_FRAC * chord:
        return None, None, (
            f"trailing edge is blunt: the last node is {gap / chord:.5f} "
            f"chords from the first, over the {TE_GAP_MAX_FRAC:g}-chord base "
            "this method solves. A base panel is not a trailing edge — the "
            "single-gamma Kutta closure has nothing to close on it, and the "
            "measured cost is 7.6 % of cl at a 0.0025-chord base, WORSENING "
            "with panel count (see TE_GAP_MAX_FRAC). Close the section first: "
            "panel2d.close_trailing_edge(x, y)")

    area = _signed_area(nodes)
    if area <= 0.0:
        return None, None, (
            "loop is clockwise (signed area "
            f"{area:.6g}); this method requires the package's counter-"
            "clockwise order, trailing edge -> upper surface -> leading "
            "edge -> lower surface")

    if _self_intersects(nodes):
        return None, None, "loop is self-intersecting"

    # Trailing edge: the last SURFACE panel is the closing segment when the
    # base is degenerate, and the one before it when the base is a panel.
    last = n - 1 if explicitly_closed else n - 2
    if last <= 0:
        return None, None, "loop has no surface panels either side of the base"
    t_first = seg[0] / L[0]
    t_last = seg[last] / L[last]
    # wedge between the two surface directions LEAVING the trailing edge
    cos_wedge = float(np.dot(t_first, -t_last))
    wedge = float(np.arccos(np.clip(cos_wedge, -1.0, 1.0)))
    if wedge > TE_WEDGE_MAX_RAD:
        return None, None, (
            f"trailing-edge wedge angle is {np.rad2deg(wedge):.1f} deg, over "
            f"the {np.rad2deg(TE_WEDGE_MAX_RAD):.0f} deg this method accepts; "
            "the loop must START at the trailing edge")

    return nodes, (0, last), None


def close_trailing_edge(x, y):
    """Pinch a blunt trailing edge shut, and return the closed loop.

    A base panel is not something this formulation can carry (see
    :data:`TE_GAP_MAX_FRAC` for the measurement that says so), so a section
    with one is CLOSED before it is flown. The rule, stated because a closure
    is a change to the section and the reader is entitled to know which one:

        each surface is moved towards the chord line by HALF the trailing-edge
        gap, weighted linearly in x from 0 at the leading edge to 1 at the
        trailing edge:      y <- y - 0.5 * dy_te * (x - x_le) / chord

    with ``dy_te`` the signed y-gap between the first and last node. Linear in
    x is the same taper XFOIL's own ``TGAP`` uses: it removes the base without
    moving the leading edge, without introducing a kink, and without touching
    the camber line — both surfaces move by the same amount in opposite senses,
    so the thickness closes and the camber is unchanged to first order.

    The x-gap, if any, is closed the same way, so a section whose two trailing
    nodes differ in x as well (a cut TE) lands on their midpoint.

    Returns ``(x, y)`` as new arrays; the input is not modified. A loop that
    already closes comes back unchanged to round-off (assert it with ==).
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size != y.size:
        raise ValueError(
            f"x and y must have equal length, got {x.size} and {y.size}")
    if x.size < 4:
        raise ValueError(f"a loop needs at least 4 points, got {x.size}")
    dx_te = float(x[0] - x[-1])
    dy_te = float(y[0] - y[-1])
    x_le = float(x.min())
    chord = float(np.ptp(x))
    if not chord > 0.0:
        raise ValueError("loop has zero chord")
    if np.hypot(dx_te, dy_te) <= _GEOM_TOL * chord:
        return x.copy(), y.copy()          # already closed: an exact no-op
    # +1 on the first-node side, -1 on the last-node side. The loop runs
    # TE -> upper -> LE -> lower -> TE, so "which side am I on" is decided by
    # position along the loop, not by the sign of y: a heavily cambered
    # section can have both trailing nodes above the chord line.
    n = x.size
    i_le = int(np.argmin(x))
    first = np.arange(n) <= i_le
    side = np.where(first, 1.0, -1.0)
    # Each side's weight is normalised on ITS OWN trailing node, so the taper
    # reaches exactly 1 there and the two nodes land on each other exactly.
    # Normalising both on the loop's x-extent instead leaves the shorter side
    # short, and the "closed" loop still carries a base of ~1e-7 chords —
    # under the refusal, and therefore invisible, which is worse than a gap.
    span_first = float(x[0]) - x_le
    span_last = float(x[-1]) - x_le
    if not (span_first > 0.0 and span_last > 0.0):
        raise ValueError(
            "loop does not run trailing edge -> leading edge -> trailing "
            "edge: one of its ends IS the leading edge")
    w = np.where(first, (x - x_le) / span_first, (x - x_le) / span_last)
    return x - 0.5 * dx_te * side * w, y - 0.5 * dy_te * side * w


def check_body(x, y, min_panels: int = MIN_PANELS) -> str | None:
    """Refusal reason for a candidate loop, or None if it is solvable.

    The in-contract path: an optimiser can hand this a self-intersecting
    aerofoil out of a CST vector, which is an infeasible DESIGN, not a
    programming error, so it gets a reason (objective._fail's contract)
    rather than an exception. :class:`Body` raises the same string for
    call-site errors.
    """
    return _prepare_loop(x, y, min_panels)[2]


@dataclass(frozen=True, eq=False)
class Body:
    """One closed body: a counter-clockwise loop in the package's order.

    ``x``/``y`` are the loop nodes in airfoil.py's "XFOIL LOAD order"
    (trailing edge -> upper surface -> leading edge -> lower surface ->
    trailing edge), i.e. COUNTER-CLOCKWISE with x downstream and y up, the
    order naca4_coords and cst_coords already return. The closing segment
    (last node -> first) is the trailing-edge base; a sharp section repeats
    the trailing-edge node and the degenerate base is dropped.

    Construction validates and RAISES ValueError with the refusal reason;
    :func:`check_body` returns the same string without raising, for callers
    inside an optimiser loop.
    """

    x: np.ndarray
    y: np.ndarray
    name: str = ""
    min_panels: int = MIN_PANELS
    nodes: np.ndarray = field(default=None, repr=False, compare=False)
    geom: PanelGeom = field(default=None, repr=False, compare=False)
    kutta: tuple = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        nodes, kutta, reason = _prepare_loop(self.x, self.y, self.min_panels)
        if reason is not None:
            raise ValueError(reason)
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "kutta", kutta)
        object.__setattr__(self, "geom", _geom_from_nodes(nodes))

    @property
    def n_panels(self) -> int:
        return self.geom.n_panels

    def _moved(self, nodes: np.ndarray) -> "Body":
        """A copy carrying already-validated ``nodes`` under a RIGID MOTION.

        Why this exists, and why the obvious ``Body(x, y)`` is wrong here.
        ``_prepare_loop`` distinguishes two spellings of "closed": a loop whose
        trailing-edge node is REPEATED (the package's sharp-TE arrays) has the
        repeat dropped, and its closing segment is then the last SURFACE panel;
        a loop with no repeat closes on a BASE. ``self.nodes`` is the array
        with the repeat already dropped, so handing it back to the constructor
        re-reads it as the second kind — and the last surface panel is
        reclassified as a trailing-edge base. The Kutta pair then moves inboard
        by one panel, silently, and the flap of a two-element section is solved
        with its closure in the wrong place.

        A rotation or a translation is a rigid motion: it preserves the
        orientation, every panel length, the wedge angle and the closure. So
        nothing needs re-validating, and re-deriving the closure from scratch
        is exactly the step that loses it. The Kutta pair travels unchanged.
        """
        out = Body.__new__(Body)
        object.__setattr__(out, "x", nodes[:, 0])
        object.__setattr__(out, "y", nodes[:, 1])
        object.__setattr__(out, "name", self.name)
        object.__setattr__(out, "min_panels", self.min_panels)
        object.__setattr__(out, "nodes", nodes)
        object.__setattr__(out, "kutta", self.kutta)
        object.__setattr__(out, "geom", _geom_from_nodes(nodes))
        return out

    def translated(self, dx: float, dy: float) -> "Body":
        """A copy shifted by (dx, dy) — the multi-element placement move."""
        return self._moved(self.nodes + np.array([float(dx), float(dy)]))

    def rotated(self, angle_rad: float, x0: float = 0.0, y0: float = 0.0
                ) -> "Body":
        """A copy rotated by ``angle_rad`` (counter-clockwise) about (x0, y0).

        Counter-clockwise is NOSE-DOWN in this frame (module docstring), so a
        flap deflected trailing-edge-down takes a NEGATIVE angle. Rotation
        preserves the loop's orientation, so the result still validates.
        """
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        dx = self.nodes[:, 0] - x0
        dy = self.nodes[:, 1] - y0
        return self._moved(np.column_stack(
            [x0 + c * dx - s * dy, y0 + s * dx + c * dy]))

    def _rebuilt(self) -> "Body":
        """Re-validate this loop through the constructor.

        Only a test wants this: it is the round trip :meth:`_moved` exists to
        avoid, kept reachable so the defect that motivated ``_moved`` stays
        demonstrable rather than merely asserted in a comment.
        """
        return Body(self.nodes[:, 0], self.nodes[:, 1],
                    name=self.name, min_panels=self.min_panels)


@dataclass(frozen=True)
class Wall:
    """A rigid wall at y = ``y``: an explicit mirrored panel system.

    DIAGNOSTIC. The image signs are derived in the module docstring
    (+sigma, -gamma — the rigid-wall image of ground_effect.py, opposite to
    the free surface of hydrofoil.py). The freestream must be parallel to
    the wall, so ``solve`` refuses a non-zero alpha with an image: express
    incidence by rotating the bodies (:meth:`Body.rotated`).

    cascade.py must not use this: a car wing's 3-D lattice already carries
    the ground image, and both together would count ground effect twice.
    """

    y: float = 0.0


# ------------------------------------------------------------- influence


def _kernels(targets: np.ndarray, geom: PanelGeom, self_panels: bool
             ) -> tuple[np.ndarray, np.ndarray]:
    """(P, N) arrays U and W of the module docstring, fully vectorised.

    ``self_panels`` says the P targets ARE this geometry's control points in
    order, so the diagonal is the on-panel limit taken from the FLUID side
    (local zeta -> 0^-, outside a counter-clockwise body): U = 0, W = -1/2.
    Computing it instead would return atan2 of a rounding error.
    """
    d = targets[:, None, :] - geom.a[None, :, :]
    xl = d[..., 0] * geom.t[None, :, 0] + d[..., 1] * geom.t[None, :, 1]
    # local zeta_hat = rot90(t) = -n
    zl = -(d[..., 0] * geom.n[None, :, 0] + d[..., 1] * geom.n[None, :, 1])
    xl2 = xl - geom.L[None, :]
    r1sq = xl * xl + zl * zl
    r2sq = xl2 * xl2 + zl * zl

    with np.errstate(divide="ignore", invalid="ignore"):
        U = np.log(r1sq / r2sq) / _FOUR_PI
    W = (np.arctan2(zl, xl2) - np.arctan2(zl, xl)) / _TWO_PI

    if self_panels:
        k = np.arange(geom.n_panels)
        U[k, k] = 0.0
        W[k, k] = -0.5
    return U, W


def _project(U: np.ndarray, W: np.ndarray, geom: PanelGeom,
             t_tar: np.ndarray, n_tar: np.ndarray):
    """Kernels -> the four (P, N) influence blocks.

    v_source = U t_j - W n_j and v_vortex = -W t_j - U n_j (the local frame
    has zeta_hat = -n, module docstring), projected on the target panel's
    own tangent and normal.
    """
    ctt = t_tar @ geom.t.T
    ctn = n_tar @ geom.t.T
    cnt = t_tar @ geom.n.T
    cnn = n_tar @ geom.n.T
    vn_s = U * ctn - W * cnn
    vt_s = U * ctt - W * cnt
    vn_v = -W * ctn - U * cnn
    vt_v = -W * ctt - U * cnt
    return vn_s, vt_s, vn_v, vt_v


# --------------------------------------------------------------- results


@dataclass(frozen=True, eq=False)
class BodySolution:
    """One body's share of the solution.

    ``cl``/``cd``/``cm`` are surface-pressure integrals on the ASSEMBLY's
    reference chord and moment point, so the per-body values sum exactly to
    the assembly values. ``cl_gamma`` is the Kutta-Joukowski value from this
    body's own circulation: equal to ``cl`` for an isolated body (to the
    discretisation error), and deliberately NOT equal when bodies interact.
    """

    name: str
    Gamma: float
    gamma: float
    cl: float
    cd: float
    cm: float
    cl_gamma: float
    cp: np.ndarray
    cp_min: float
    vt: np.ndarray
    sigma: np.ndarray
    kutta_residual: float
    x_cp: np.ndarray
    y_cp: np.ndarray
    nx: np.ndarray
    ny: np.ndarray
    panel_len: np.ndarray
    nodes: np.ndarray


@dataclass(frozen=True, eq=False)
class PanelSolution:
    """Assembly solution. ``cd`` is a d'Alembert residual, not a drag."""

    bodies: tuple
    cl: float
    cd: float
    cm: float
    cp_min: float
    alpha_rad: float
    V: float
    c_ref: float
    ref_point: tuple
    image: Wall | None
    tangency_residual: float
    kutta_residual: float

    def body(self, key) -> BodySolution:
        """Look a body up by index or by name."""
        if isinstance(key, int):
            return self.bodies[key]
        for b in self.bodies:
            if b.name == key:
                return b
        raise KeyError(f"no body named {key!r}")


# ----------------------------------------------------------------- solve


def solve(bodies, alpha_rad: float, V: float = 1.0,
          image: Wall | None = None, c_ref: float | None = None,
          ref_point: tuple | None = None) -> PanelSolution:
    """Solve the assembly and return per-body and total loads.

    ``bodies``     one :class:`Body` or a sequence of them.
    ``alpha_rad``  freestream direction; V_inf = V (cos a, sin a). Must be
                   0 when ``image`` is given (the wall condition needs the
                   stream parallel to the wall — module docstring).
    ``c_ref``      reference chord for every coefficient. Default: the
                   assembly's total x-extent, which for a two-element
                   section is the STOWED-to-flap-TE length, i.e. the
                   convention that makes the elements' cl add up to the
                   section's.
    ``ref_point``  moment reference. Default: the quarter chord of the
                   assembly's x-extent, on y = 0.

    Raises ValueError for call-site errors (no bodies, alpha with an image,
    a body crossing the wall). Body geometry was already refused at
    :class:`Body` construction; :func:`check_body` is the non-raising path.
    """
    if isinstance(bodies, Body):
        bodies = (bodies,)
    bodies = tuple(bodies)
    if not bodies:
        raise ValueError("no bodies to solve")
    if not all(isinstance(b, Body) for b in bodies):
        raise ValueError("every body must be a panel2d.Body")
    if not V > 0.0:
        raise ValueError(f"freestream speed must be positive, got {V}")

    geoms = [b.geom for b in bodies]
    counts = [g.n_panels for g in geoms]
    offs = np.concatenate([[0], np.cumsum(counts)])
    M = int(offs[-1])
    nb = len(bodies)

    if image is not None:
        if abs(alpha_rad) > 0.0:
            raise ValueError(
                "a wall image requires the freestream parallel to the wall "
                f"(alpha = 0); got alpha = {np.rad2deg(alpha_rad):.4g} deg. "
                "Rotate the geometry instead (Body.rotated).")
        for b in bodies:
            dy = b.nodes[:, 1] - image.y
            if np.min(dy) * np.max(dy) <= 0.0:
                raise ValueError(
                    f"body {b.name!r} touches or crosses the wall at "
                    f"y = {image.y}")

    cp_all = np.vstack([g.cp for g in geoms])
    t_all = np.vstack([g.t for g in geoms])
    n_all = np.vstack([g.n for g in geoms])

    # ---- influence blocks (real system, then the slaved image system)
    Vn_s = np.empty((M, M))
    Vt_s = np.empty((M, M))
    Vn_v = np.empty((M, M))
    Vt_v = np.empty((M, M))
    for j, g in enumerate(geoms):
        sl = slice(offs[j], offs[j + 1])
        blocks = []
        for i, gi in enumerate(geoms):
            U, W = _kernels(gi.cp, g, self_panels=(i == j))
            blocks.append(_project(U, W, g, gi.t, gi.n))
        Vn_s[:, sl] = np.vstack([b[0] for b in blocks])
        Vt_s[:, sl] = np.vstack([b[1] for b in blocks])
        Vn_v[:, sl] = np.vstack([b[2] for b in blocks])
        Vt_v[:, sl] = np.vstack([b[3] for b in blocks])

    if image is not None:
        for j, g in enumerate(geoms):
            sl = slice(offs[j], offs[j + 1])
            gm = _mirror_geom(g, image.y)
            U, W = _kernels(cp_all, gm, self_panels=False)
            vn_s, vt_s, vn_v, vt_v = _project(U, W, gm, t_all, n_all)
            Vn_s[:, sl] += vn_s          # image source: SAME strength
            Vt_s[:, sl] += vt_s
            Vn_v[:, sl] -= vn_v          # image vortex: OPPOSITE strength
            Vt_v[:, sl] -= vt_v

    # per-body vortex columns: one gamma shared by the body's panels
    Cn = np.column_stack([Vn_v[:, offs[j]:offs[j + 1]].sum(axis=1)
                          for j in range(nb)])
    Ct = np.column_stack([Vt_v[:, offs[j]:offs[j + 1]].sum(axis=1)
                          for j in range(nb)])

    # ---- assemble
    Vinf = V * np.array([np.cos(alpha_rad), np.sin(alpha_rad)])
    A = np.zeros((M + nb, M + nb))
    rhs = np.zeros(M + nb)

    A[:M, :M] = Vn_s
    A[:M, M:] = Cn
    rhs[:M] = -(n_all @ Vinf)

    kutta_rows = []
    for j, b in enumerate(bodies):
        f, l = b.kutta
        f += offs[j]
        l += offs[j]
        kutta_rows.append((f, l))
        A[M + j, :M] = Vt_s[f] + Vt_s[l]
        A[M + j, M:] = Ct[f] + Ct[l]
        rhs[M + j] = -((t_all[f] + t_all[l]) @ Vinf)

    sol = np.linalg.solve(A, rhs)
    sigma = sol[:M]
    gamma = sol[M:]

    # ---- recover the surface flow
    gam_panel = np.concatenate([np.full(counts[j], gamma[j])
                                for j in range(nb)])
    vt = t_all @ Vinf + Vt_s @ sigma + Ct @ gamma
    vn = n_all @ Vinf + Vn_s @ sigma + Cn @ gamma
    cp = 1.0 - (vt / V) ** 2

    if c_ref is None:
        xs = np.concatenate([b.nodes[:, 0] for b in bodies])
        c_ref = float(np.ptp(xs))
    c_ref = float(c_ref)
    if not c_ref > 0.0:
        raise ValueError(f"reference chord must be positive, got {c_ref}")
    if ref_point is None:
        xs = np.concatenate([b.nodes[:, 0] for b in bodies])
        ref_point = (float(xs.min()) + 0.25 * c_ref, 0.0)
    x0, y0 = float(ref_point[0]), float(ref_point[1])

    lift_dir = np.array([-np.sin(alpha_rad), np.cos(alpha_rad)])
    drag_dir = np.array([np.cos(alpha_rad), np.sin(alpha_rad)])

    out = []
    cl_tot = cd_tot = cm_tot = 0.0
    for j, b in enumerate(bodies):
        sl = slice(offs[j], offs[j + 1])
        g = b.geom
        cp_j = cp[sl]
        # dF / q = -Cp n dl  (pressure acts against the outward normal)
        dF = -(cp_j * g.L)[:, None] * g.n
        F = dF.sum(axis=0)
        Mz = float(np.sum((g.cp[:, 0] - x0) * dF[:, 1]
                          - (g.cp[:, 1] - y0) * dF[:, 0]))
        cl_j = float(F @ lift_dir) / c_ref
        cd_j = float(F @ drag_dir) / c_ref
        cm_j = -Mz / (c_ref * c_ref)          # nose-up positive (derived)
        Gam = float(gamma[j] * g.perimeter)
        f, l = kutta_rows[j]
        out.append(BodySolution(
            name=b.name or f"body{j}",
            Gamma=Gam, gamma=float(gamma[j]),
            cl=cl_j, cd=cd_j, cm=cm_j,
            cl_gamma=float(-2.0 * Gam / (V * c_ref)),
            cp=cp_j, cp_min=float(cp_j.min()), vt=vt[sl], sigma=sigma[sl],
            kutta_residual=float(abs(vt[f] + vt[l]) / V),
            x_cp=g.cp[:, 0], y_cp=g.cp[:, 1], nx=g.n[:, 0], ny=g.n[:, 1],
            panel_len=g.L, nodes=b.nodes))
        cl_tot += cl_j
        cd_tot += cd_j
        cm_tot += cm_j

    return PanelSolution(
        bodies=tuple(out), cl=cl_tot, cd=cd_tot, cm=cm_tot,
        cp_min=float(cp.min()), alpha_rad=float(alpha_rad), V=float(V),
        c_ref=c_ref, ref_point=(x0, y0), image=image,
        tangency_residual=float(np.max(np.abs(vn)) / V),
        kutta_residual=float(max(b.kutta_residual for b in out)),
        )


def velocity(sol: PanelSolution, points, bodies) -> np.ndarray:
    """Total velocity at arbitrary (P, 2) ``points`` for a solved assembly.

    ``bodies`` is the same sequence handed to :func:`solve` (the solution
    stores strengths and panel geometry, not the Body objects, so the caller
    passes them back). Off-body only: a point ON a panel returns the
    one-sided limit of whichever side round-off lands it.
    """
    if isinstance(bodies, Body):
        bodies = (bodies,)
    bodies = tuple(bodies)
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    Vinf = sol.V * np.array([np.cos(sol.alpha_rad), np.sin(sol.alpha_rad)])
    v = np.broadcast_to(Vinf, pts.shape).copy()

    for b, bs in zip(bodies, sol.bodies):
        for geom, s_sign, g_sign in _systems(b.geom, sol.image):
            U, W = _kernels(pts, geom, self_panels=False)
            # v = sigma (U t - W n) + gamma (-W t - U n)
            us = s_sign * (U * bs.sigma[None, :])
            ws = s_sign * (W * bs.sigma[None, :])
            uv = g_sign * (-W * bs.gamma)
            wv = g_sign * (-U * bs.gamma)
            coef_t = us + uv
            coef_n = -ws + wv
            v[:, 0] += coef_t @ geom.t[:, 0] + coef_n @ geom.n[:, 0]
            v[:, 1] += coef_t @ geom.t[:, 1] + coef_n @ geom.n[:, 1]
    return v


def _systems(geom: PanelGeom, image: Wall | None):
    """The real panel system, plus the slaved image system if there is one."""
    yield geom, 1.0, 1.0
    if image is not None:
        yield _mirror_geom(geom, image.y), 1.0, -1.0


# ------------------------------------------------- exact validation machine


def joukowski_coords(n_panels: int, eps_over_a: float = 0.10,
                     camber_over_a: float = 0.0, a: float = 1.0
                     ) -> np.ndarray:
    """Joukowski aerofoil nodes, in this module's loop convention.

    The map z = zeta + a^2/zeta carries the circle
    zeta = mu + R exp(i theta), with mu = -eps + i (camber) and
    R = |a - mu| so the circle passes through zeta = a, to an aerofoil with
    a cusped trailing edge at z = 2a. The map is analytic with dz/dzeta != 0
    on the circle except at zeta = a, so it preserves orientation: theta
    increasing (counter-clockwise) gives a counter-clockwise loop, and it
    starts at the trailing edge and runs over the UPPER surface first
    (Im z ~ +R^3 theta^3 / a^2 just off the cusp) — the package's own order,
    no reordering needed.

    Uniform spacing in theta is deliberately NOT uniform in z: dz/dzeta
    vanishes at zeta = a and is small near the leading edge (|dz/dzeta| ~
    4 eps/a there), so the nodes cluster at both ends, which is the
    clustering a panel method wants and the reason no cosine law is applied
    on top.

    Returns (n_panels + 1, 2) with the trailing-edge node repeated, the
    package's sharp-TE spelling.
    """
    mu = complex(-eps_over_a * a, camber_over_a * a)
    R = abs(a - mu)
    th_T = np.angle(a - mu)
    th = th_T + np.linspace(0.0, 2.0 * np.pi, n_panels + 1)
    zeta = mu + R * np.exp(1j * th)
    z = zeta + a * a / zeta
    z[-1] = z[0]                     # close it exactly, not to round-off
    return np.column_stack([z.real, z.imag])


def joukowski_exact(alpha_rad: float, n_theta: int = 20001,
                    eps_over_a: float = 0.10, camber_over_a: float = 0.0,
                    a: float = 1.0, V: float = 1.0) -> dict:
    """Exact conformal solution for the same aerofoil — the real gate.

    Complex potential in the circle plane, with the circulation fixed by
    putting the rear stagnation point at the cusp pre-image zeta = a:

        W(zeta) = V (zeta - mu) e^{-i alpha} + V R^2 e^{i alpha}/(zeta - mu)
                  - i Gamma / (2 pi) ln(zeta - mu),
        dW/dzeta = 0 at zeta = a   =>   Gamma = 4 pi V R sin(theta_T - alpha)

    with Gamma POSITIVE COUNTER-CLOCKWISE (the -i/2pi ln form: at a point to
    the right of a positive vortex the velocity is upward). Then
    w(z) = (dW/dzeta) / (dz/dzeta), Cp = 1 - |w/V|^2, and
    cl = -2 Gamma / (V c) with c the aerofoil's own chord (its x-extent),
    which is 8 pi R sin(alpha - theta_T) / c.

    Returns the surface points, the exact Cp on them, cl, and cm about the
    quarter chord obtained by integrating that exact Cp over the exact
    contour (the definition of cm, evaluated on the analytic solution — an
    independent reference for the panel answer, and self-checked here by
    reproducing cl from the same integral).
    """
    mu = complex(-eps_over_a * a, camber_over_a * a)
    R = abs(a - mu)
    th_T = np.angle(a - mu)
    Gamma = 4.0 * np.pi * V * R * np.sin(th_T - alpha_rad)

    th = th_T + np.linspace(0.0, 2.0 * np.pi, n_theta)
    zeta = mu + R * np.exp(1j * th)
    z = zeta + a * a / zeta
    dWdz_num = (V * np.exp(-1j * alpha_rad)
                - V * R * R * np.exp(1j * alpha_rad) / (zeta - mu) ** 2
                - 1j * Gamma / (_TWO_PI * (zeta - mu)))
    dzdzeta = 1.0 - a * a / zeta ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        w = dWdz_num / dzdzeta
    q = np.abs(w)
    cp = 1.0 - (q / V) ** 2

    x, y = z.real, z.imag
    chord = float(np.ptp(x))
    cl = float(-2.0 * Gamma / (V * chord))

    # pressure integral over the exact contour, for cm (and to self-check cl)
    good = np.isfinite(cp)
    xs, ys, cps = x[good], y[good], cp[good]
    ax, ay = xs[:-1], ys[:-1]
    bx, by = xs[1:], ys[1:]
    dx, dy = bx - ax, by - ay
    L = np.hypot(dx, dy)
    nx, ny = dy / L, -dx / L
    cpm = 0.5 * (cps[:-1] + cps[1:])
    dFx = -cpm * L * nx
    dFy = -cpm * L * ny
    x0 = float(xs.min()) + 0.25 * chord
    Mz = np.sum((0.5 * (ax + bx) - x0) * dFy - 0.5 * (ay + by) * dFx)
    lift = -np.sin(alpha_rad) * dFx.sum() + np.cos(alpha_rad) * dFy.sum()

    return {"x": x, "y": y, "cp": cp, "theta": th, "chord": chord,
            "cl": cl, "cl_pressure": float(lift / chord),
            "cm_c4": float(-Mz / chord ** 2), "Gamma": float(Gamma),
            "R": float(R), "theta_te": float(th_T), "x_ref": x0}
