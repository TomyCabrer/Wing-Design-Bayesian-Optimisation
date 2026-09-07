"""Nonplanar horseshoe-vortex lifting-surface solver (Weissinger VLM) — Tier A+ winglets.

numpy port of the panel / influence machinery in the MATLAB reference
~/Desktop/GDP/tandem_wing/control_aero/control_aero.m (`hshoe`/`vline`/`seminf`
Biot-Savart, bound vortex at c/4, flow tangency at an aft control point,
twist/dihedral-rotated panel normals, influence matrix built + LU-factored once
per geometry), reduced to a single wing with optional canted tip extensions
(winglets). Pure numpy — MATLAB is never called.

Axes (MATLAB convention): x aft (downstream) +, y starboard +, z up +.
Freestream is LINEARISED in alpha: Vinf = V (x_hat + alpha z_hat), so the
whole solve is exactly linear in alpha (same linearisation as the LLT, whose
monoplane RHS is alpha - alpha_L0); trim to a CL target is closed-form from
two back-substitutions (cf. adjoint.py).

Model: ONE chordwise panel per strip (Weissinger / extended lifting line).
The control point sits d = a c / (4 pi) aft of the bound vortex — the 2-D
flow-tangency balance Gamma/(2 pi d) = V alpha then reproduces the section
lift-curve slope `a` exactly (d = c/2, i.e. the classical 3/4-chord point,
for a = 2 pi). This is how the section polar's a_lin enters, mirroring the
per-station `a` of llt.py. The zero-lift angle alpha_L0 is built into the
panel normals (theta = twist - alpha_L0), as in the MATLAB reference.

Outputs:
- CL from Kutta-Joukowski with the freestream: L = rho V sum Gamma_i l_y,i
  (the winglets' horizontal projection contributes; side forces cancel by
  symmetry and are returned as a CY diagnostic).
- CDi in the TREFFTZ PLANE (far field): near-field VLM induced drag is
  unreliable at one chordwise panel. Far downstream each trailing leg is a
  2-D point vortex at the panel-edge (y, z) trace and

      Di = -(rho/2) sum_i Gamma_i (w_i . n_i) s_i

  over the wake traces of the bound segments (Katz & Plotkin 2001, ch. 8;
  the streamwise collapse is Munk's stagger theorem, Munk 1921).
  PAIRING MATTERS: panel edges sit at cosine angles theta_k = k pi/N and all
  spanwise stations (control points, chord/twist sampling, Trefftz wash
  evaluation) at the INTERLACED cosine midpoints theta = (k+1/2) pi/N — the
  same interlacing as llt.py's midpoint grid. Under this quadrature the
  discrete Trefftz sum is exact for Fourier loading modes (a prescribed
  elliptic circulation returns e = 1.000000 at any N), so planar wings
  respect the Munk bound. Using geometric panel midpoints instead biases
  CDi low by O(1/N) and lets planar wings appear to beat the bound
  (verified numerically before this fix).
- Span efficiency referenced to the PLANAR span b and wing area S:
  e = CL^2 / (pi AR CDi), AR = b^2/S. Winglets add wetted area but no
  reference area/span, so a nonplanar system can exceed e = 1 (Munk 1921).

Winglet parameterisation (Tier A+ design variables):
- height fraction h_frac = h / (b/2) in [0, 0.15]: arc length of the winglet
  quarter-chord line beyond the tip. Winglets below MIN_WINGLET_FRAC of the
  semi-span are dropped (aerodynamically nil, avoids degenerate panels).
- cant angle in [60, 90] deg, measured from the wing plane: 90 deg = vertical
  winglet (pure nonplanar effect, projected span unchanged); lower cant tilts
  the winglet outboard, trading towards a raked span extension (projected
  span grows by h cos(cant) per side — deliberately allowed; e is referenced
  to the nominal b so the benefit is visible in e).
- blend fraction / shape (``winglet_blend_frac`` / ``winglet_blend_shape``):
  the device may leave the wing plane through a TURN instead of a corner —
  a circular fillet, the curvature-continuous smoothstep, or the clothoid
  pair. The panels simply follow geometry.span_path, so the Trefftz plane
  sees the real wake trace of whichever transition was drawn.
- wing-side blend (``winglet_wing_blend_frac``): the same turn may START
  INBOARD of the tip, so the OUTER WING bends up into the device instead of
  meeting it at a fixed point. The whole spanwise line is then parameterised
  by DEVELOPED ARC LENGTH from the root: panel edges and stations are cosine
  distributed in arc, and the wing's chord and twist laws are sampled at arc
  (which is identical to sampling at y for the flat wing, so a run without a
  wing-side blend is bit-for-bit its published self). Holding arc rather than
  y fixed is what keeps b, S, AR and CL_target meaning the same thing across
  the family: a blend moves projected span into height, it does not delete
  wing.
- winglet chord = wing tip chord, incidence = wing tip twist, zero toe by
  DEFAULT (the winglet is an untwisted tip extension). Each of those is
  overridable — ``winglet_chord_scale``, ``winglet_chord_follows``,
  ``winglet_toe_deg``, and
  ``winglet_a`` / ``winglet_alpha_L0`` for a tip device with its OWN section
  — because a car endplate is not a tip extension of the wing: it is a
  separate surface with its own chord, its own aerofoil and (if it is
  cambered or toed) its own side load. Leaving them at their defaults
  reproduces the tip-extension model exactly, so no existing result moves.
  The section slope ``a`` and zero-lift angle ``alpha_L0`` are therefore
  PER PANEL: they set the control-point offset a c/(4 pi) and the panel
  normal's built-in incidence, both of which are local quantities.
- ``winglet_chord_follows`` (default False = the rectangular device above,
  bit-for-bit): the device's chord CONTINUES the wing's own chord law
  instead of holding the tip chord. The whole spanwise line is already
  parameterised by developed arc, so this is one clip removed — the law is
  sampled at the panel's own arc, i.e. at eta > 1, and wing and device carry
  ONE chord distribution. A device whose continued chord runs to a knife
  edge (below :data:`DEVICE_CHORD_FLOOR` of the wing's tip chord) is
  REFUSED here, not quietly rectangularised: it is the same in-contract
  failure a collapsed chord law already is.
- a blend BLENDS. Everything in the bullet above that makes the device a
  different surface from the wing — its chord scale, its section
  (``a`` / ``alpha_L0``) and its toe — used to change in ONE STEP at the
  junction panel while the quarter-chord line turned smoothly through the
  blend. Drawn, that is a surface of one chord curving into a surface of
  another: at the car endplate's chord ratio of 3 the leading edge jumps
  0.10 m forward and the trailing edge 0.31 m aft across a corner the model
  was simultaneously calling filleted. So those three quantities now RAMP
  along the turn, on the turn's own law (:func:`geometry.winglet_turn_angle`
  normalised by the cant): the surface changes what it IS exactly where it
  changes where it POINTS, and the junction the fillet credit is claimed for
  is a junction both surfaces actually reach. With ``blend_frac = 0`` the
  turn angle steps at the junction, so the ramp is the step and every
  published run is bit-for-bit unmoved; with a WING-side blend the ramp
  starts inboard of the tip with the turn, which is where that transition
  starts. The ramp is a straight linear interpolation of each quantity — a
  MODEL CHOICE, and the mildest one: it makes the surface continuous
  without inventing a shape for how a chord grows.

Image plane (:class:`ImagePlane`) — a flat reflecting boundary at z = z_p:
- ``ground`` (rigid wall, w = 0 on the plane): image circulation OPPOSITE.
- ``free_surface`` (high depth-Froude limit, phi = 0 on the plane): image
  circulation SAME sign. Both signs are derived, not assumed, in
  ground_effect.py and hydrofoil.py respectively, and are re-verified here
  DIRECTLY on the boundary condition (tests/test_vlm_image.py).
The image is folded into the SAME linear system (its circulation is the
real system's, mirrored) so nothing is added to the unknowns — exactly the
structure of the LLT image solvers. Both the near-field tangency matrix and
the Trefftz-plane wash carry it; forces are summed over the REAL panels
only, since the image is a boundary condition and not a physical surface.

Second surface (:class:`TailSurface`) — an optional horizontal tail in the
SAME solve, so a winglet and a tail can finally be designed together:
- panels are appended to the same system (the horseshoe kernel does not care
  where a surface sits, and the Trefftz plane collapses x by Munk's stagger
  theorem), so the wing-on-tail downwash and the tail's upwash back-reaction
  come OUT of the solve — there is no separate d(eps)/d(alpha) model.
- its INCIDENCE is not geometry but the second trim unknown. It enters the
  LINEARISED boundary condition as a uniform RHS shift on the tail panels,
  which keeps CL and Cm exactly affine in (alpha, i_t): trim is then a
  closed-form 2x2 solve and the static margin is read straight off the same
  basis solutions (:meth:`VLM.solve_trim_moment`, :meth:`VLM.neutral_point`).
  The dropped term is O(i_t) x O(alpha) — second order, the same
  small-disturbance argument the alpha-linearisation already rests on.
- the tail must clear the wing's trailing sheet: at z = 0 the collocation
  points sit IN it, where the leg kernel is grid-singular (tail.py measured
  the 0.05 b floor; the caller enforces it).

The nonplanar case is precisely why this exists: for a PLANAR wing the
image of a horizontal vortex system is a pure translation, which is what
ground_effect.py / hydrofoil.py exploit. Add a winglet and the mirror is no
longer a translation, so the imaged lifting-surface solve has to be done
here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import lu_factor, lu_solve

from .geometry import Wing, dihedral_rotate, span_path, winglet_path

TWO_PI = 2.0 * np.pi
FOUR_PI = 4.0 * np.pi
_XHAT = np.array([1.0, 0.0, 0.0])  # downstream (trailing-leg) direction

#: winglets shorter than this fraction of the semi-span are dropped
MIN_WINGLET_FRAC = 0.01


def device_is_flown(h_frac: float, n_winglet: int = 8) -> bool:
    """Does a tip device at this height fraction reach the lattice?

    ONE definition of the drop rule, for readers that hold a fraction and no
    VLM — a breakdown, a CAD loft, the bound-riding census. ``VLM.__init__``
    asks the same question in metres (bit-for-bit identical for b > 0, since
    h = h_frac * b / 2), and the drop-rule test holds the two together.
    """
    return bool(float(h_frac) >= MIN_WINGLET_FRAC and int(n_winglet) > 0)

#: floor on a tip device's CONTINUED chord (``winglet_chord_follows``), as a
#: fraction of the wing's tip chord. Same number and same job as
#: geometry.CHORD_MULT_FLOOR — past it the device stops being a surface — but
#: measured against the tip chord, because that is the chord the device
#: continues. Only reachable with the continuation on: a rectangular device
#: is the tip chord by construction.
DEVICE_CHORD_FLOOR = 0.05

#: image circulation sign per boundary condition. Rigid wall: w = 0 on the
#: plane needs the OPPOSITE-circulation mirror. Free surface in the high
#: depth-Froude limit: phi = 0 on the plane needs the SAME-circulation
#: mirror. (Both derived in ground_effect.py / hydrofoil.py; asserted
#: against the boundary conditions themselves in tests/test_vlm_image.py.)
IMAGE_SIGNS = {"ground": -1.0, "free_surface": +1.0}

#: a surface closer to the plane than this (times the semi-span) is refused:
#: the image panels start to overlap the real ones and the linearised
#: boundary condition stops meaning anything.
MIN_PLANE_CLEARANCE_FRAC = 0.02


@dataclass(frozen=True)
class ImagePlane:
    """Flat reflecting boundary at ``z`` of a given ``kind``.

    ``kind`` selects the image sign rather than the caller passing one, so
    the classic sign trap cannot be re-introduced at a call site.
    """

    z: float
    kind: str = "free_surface"

    def __post_init__(self):
        if self.kind not in IMAGE_SIGNS:
            raise ValueError(
                f"unknown image plane kind {self.kind!r}; "
                f"choose from {sorted(IMAGE_SIGNS)}")

    @property
    def sign(self) -> float:
        return IMAGE_SIGNS[self.kind]

    def mirror(self, P: np.ndarray, axis: int = 2) -> np.ndarray:
        """Reflect points ``P`` in the plane (z is column ``axis``)."""
        out = np.array(P, dtype=float, copy=True)
        out[..., axis] = 2.0 * self.z - out[..., axis]
        return out


# ----------------------------------------------------------------------
# vortex math (port of hshoe/vline/seminf in control_aero.m, vectorised)
# ----------------------------------------------------------------------

def _vline(P: np.ndarray, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Finite vortex segment A->B, unit circulation: velocity at points P.

    P: (M, 3) eval points; A, B: (N, 3) segment ends -> (M, N, 3).
    """
    r1 = P[:, None, :] - A[None, :, :]
    r2 = P[:, None, :] - B[None, :, :]
    r0 = (B - A)[None, :, :]
    cr = np.cross(r1, r2)
    n2 = np.sum(cr * cr, axis=-1)
    n1 = np.linalg.norm(r1, axis=-1)[..., None]
    nb = np.linalg.norm(r2, axis=-1)[..., None]
    K = np.sum(r0 * (r1 / n1 - r2 / nb), axis=-1) / (FOUR_PI * np.maximum(n2, 1e-12))
    v = K[..., None] * cr
    v[n2 < 1e-10] = 0.0
    return v


def _seminf(P: np.ndarray, X: np.ndarray, uhat: np.ndarray = _XHAT) -> np.ndarray:
    """Semi-infinite vortex from X along uhat, unit circulation: velocity at P."""
    a = P[:, None, :] - X[None, :, :]
    na = np.linalg.norm(a, axis=-1)
    cr = np.cross(np.broadcast_to(uhat, a.shape), a)
    nc2 = np.sum(cr * cr, axis=-1)
    K = (1.0 + (a @ uhat) / np.maximum(na, 1e-12)) / (FOUR_PI * np.maximum(nc2, 1e-12))
    v = K[..., None] * cr
    v[nc2 < 1e-10] = 0.0
    return v


def _hshoe(P: np.ndarray, A: np.ndarray, B: np.ndarray,
           uhat: np.ndarray = _XHAT) -> np.ndarray:
    """Horseshoe (bound A->B + trailing legs downstream to infinity) at P."""
    return _vline(P, A, B) + _seminf(P, B, uhat) - _seminf(P, A, uhat)


def _trefftz_kernel(P2: np.ndarray, Q2: np.ndarray) -> np.ndarray:
    """2-D point-vortex velocity in the Trefftz (y, z) plane.

    Unit-circulation vortices (right-handed about +x) at Q2 (N, 2), evaluated
    at P2 (M, 2): v = (1/(2 pi r^2)) (-(z-zq), (y-yq)) -> (M, N, 2).
    """
    r = P2[:, None, :] - Q2[None, :, :]
    r2 = np.sum(r * r, axis=-1)
    v = np.stack([-r[..., 1], r[..., 0]], axis=-1) / (
        TWO_PI * np.maximum(r2, 1e-12)
    )[..., None]
    v[r2 < 1e-10] = 0.0
    return v


# ----------------------------------------------------------------------
# model
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class TailSurface:
    """A second lifting surface (the horizontal tail) in the SAME solve.

    At a signed streamwise station ``x`` (a canard is upstream, at negative
    x) and a height ``z`` above the wing plane. ``S`` is the LIFTING area
    used to draw it — a V-tail's equivalent flat area belongs here and its
    panel area is a drag-bookkeeping matter for the caller (tail.py's
    convention, kept).

    Two shapes, one contract:

    * ``wing is None`` — the published RECTANGULAR tail at aspect ratio
      ``AR``, drawn from ``S`` alone. Bit-for-bit the surface every frozen
      tail study flew;
    * ``wing`` given — a DESIGNED tail: a :class:`geometry.Wing` in its own
      right (its own taper, chord law and twist) which may itself carry a
      tip device (``winglet_h_frac`` / ``winglet_cant_deg`` …). It is
      panelised by the same constructor as the tandem pair's second wing,
      so the tail is drawn by exactly the code that draws a wing.

    Its INCIDENCE is not a field either way: it is the second trim unknown
    and enters through the boundary condition as a uniform shift on the
    tail's panels (see VLM). A designed tail's own twist is therefore a
    WASHOUT about that incidence — the constant part would be degenerate
    with the trim variable, and the solver would have a flat direction.
    """

    S: float
    x: float
    z: float
    AR: float = 4.0
    N: int = 20
    a: float | None = None          # section slope; None -> the wing's
    alpha_L0: float | None = None   # zero-lift angle; None -> the wing's
    dihedral_deg: float = 0.0       # panel cant [deg] — a V-TAIL is this and
    #                               nothing else. It is the ONE place the cant
    #                               is stated: a designed tail must not also
    #                               put it on its own ``wing``, and
    #                               __post_init__ refuses that. ``S`` stays
    #                               the PANEL area; the equivalent flat area
    #                               S cos^2(Gamma) is a CONSEQUENCE the
    #                               lattice produces rather than an input
    #                               (tail.lifting_area's closed form, which
    #                               the V5 tests hold it to).
    # ---- designed tail (all inert while ``wing`` is None)
    wing: object = None             # geometry.Wing — its own planform
    winglet_h_frac: float = 0.0     # tip device ON THE TAIL, as a fraction
    winglet_cant_deg: float = 90.0  # of the tail semispan
    n_winglet: int = 8
    winglet_blend_frac: float = 0.0
    winglet_blend_shape: str = "arc"
    winglet_wing_blend_frac: float = 0.0   # ...and how much of the turn this
    #                                surface itself does (geometry.span_path)

    def __post_init__(self):
        # ONE QUESTION, ONE PLACE. A designed tail is a Wing, and a Wing now
        # has a dihedral of its own — so the cant could be stated twice and
        # the lattice would apply it twice. Refuse at construction rather
        # than fly a surface at 2 Gamma.
        w_dih = float(getattr(self.wing, "dihedral_deg", 0.0) or 0.0)
        if w_dih and float(self.dihedral_deg):
            raise ValueError(
                f"the tail's cant is stated twice: TailSurface.dihedral_deg="
                f"{self.dihedral_deg} and its wing's dihedral_deg={w_dih}. "
                f"State it once, on the TailSurface")

    @property
    def designed(self) -> bool:
        return self.wing is not None

    @property
    def span(self) -> float:
        if self.wing is not None:
            return float(self.wing.b)
        return float(np.sqrt(self.AR * self.S))

    @property
    def chord(self) -> float:
        """MEAN chord — the constant chord of the rectangular tail, and the
        area/span of a designed one (whose chord varies along the span)."""
        return float(self.S / self.span)


@dataclass(frozen=True)
class VerticalSurface:
    """A fin: a rectangular surface whose span runs along z rather than y.

    This is the surface the package did not have, and its absence was not a
    modelling simplification — it was a measured hole. With a wing and a tail
    and nothing else, ``Cn_beta`` is EXACTLY zero: no panel in the lattice
    carries side force, so a sideslip has nothing to push on. Tip winglets do
    not close it either; they sit on the wing's quarter-chord line, ahead of
    the CG, and measure DEstabilising (``dynamics.py``, and the test that
    pins it).

    Geometry, all in the lattice frame (x AFT, y starboard, z up):

    ``height``
        the fin's span, from ``z_root`` upward. Negative height is how a
        VENTRAL fin is asked for — and how a hydrofoil's strut is asked for,
        since that surface hangs DOWN from the craft.
    ``x``
        the station of the fin's quarter-chord line. This is the arm that
        makes ``Cn_beta``, so it is the number that decides whether the
        surface stabilises or not: aft of the CG stabilises, ahead of it does
        the opposite, and the lattice will report either without complaint.
    ``y``
        0 for a centreline fin; non-zero for one of a twin pair (build two).
    ``z_root``
        where the fin meets the body. For a HYDROFOIL the vertical sits
        BETWEEN the wing and the tail rather than on the tail, because that
        is where the strut already is — the strut stops being dead structure
        and becomes a steerable surface.
    ``toe_deg``
        a fixed side incidence, and the channel a RUDDER deflection is
        applied through: it rotates the panel normal in the x-y plane, which
        is what a rudder does. Positive toe turns the leading edge to
        starboard.

    The fin contributes nothing to CL (its bound segments have no y extent)
    and everything to CY, which is exactly right and falls out of the
    existing Kutta-Joukowski sum rather than being special-cased.
    """

    height: float                 # [m] span along z; NEGATIVE = ventral
    chord: float                  # [m]
    x: float                      # [m] quarter-chord station (the yaw arm)
    z_root: float = 0.0           # [m] where it meets the body
    y: float = 0.0                # [m] 0 = centreline
    toe_deg: float = 0.0          # [deg] fixed incidence / rudder channel
    N: int = 12                   # panels along the height
    a: float | None = None        # section lift slope; None = the wing's
    alpha_L0: float | None = None  # None = 0, i.e. a SYMMETRIC section, which
    #                               is what a fin almost always is: a cambered
    #                               fin flies with a permanent side load.

    def __post_init__(self):
        if not abs(float(self.height)) > 0.0:
            raise ValueError(
                f"vertical surface height must be non-zero (negative is a "
                f"ventral fin), got {self.height!r}")
        if not float(self.chord) > 0.0:
            raise ValueError(
                f"vertical surface chord must be > 0, got {self.chord!r}")
        if int(self.N) < 2:
            raise ValueError(
                f"a vertical surface needs at least 2 panels, got {self.N!r}")

    @property
    def S(self) -> float:
        """Fin area [m2] — always positive, ventral or dorsal."""
        return abs(float(self.height)) * float(self.chord)

    @property
    def AR(self) -> float:
        """Geometric aspect ratio of the fin ALONE (not the effective one).

        A fin against a body behaves as though it were part of a larger
        reflected surface; that end-plating is a real effect this rectangular
        model does not carry, so quote this as the geometric number it is.
        """
        return abs(float(self.height)) / float(self.chord)


@dataclass(frozen=True)
class SecondWing:
    """A full second WING in the same solve — the tandem pair's rear surface.

    Unlike :class:`TailSurface` (a rectangular surface whose incidence is a
    trim unknown) this is a :class:`geometry.Wing` in its own right: its own
    planform, chord law, twist and — through ``winglet_h_frac`` / cant — its
    own tip device. Its incidence is GEOMETRY (the pair's decalage lives in
    its twist), because a tandem is trimmed in lift alone: there is no second
    trim unknown to spend on it.

    ``x`` / ``z`` place its quarter-chord line relative to the first wing's
    (x aft, z up), which is exactly the stagger tandem.py defines.
    """

    wing: object                     # geometry.Wing
    x: float = 0.0
    z: float = 0.0
    N: int = 40
    winglet_h_frac: float = 0.0
    winglet_cant_deg: float = 90.0
    n_winglet: int = 8
    winglet_blend_frac: float = 0.0
    winglet_blend_shape: str = "arc"
    winglet_wing_blend_frac: float = 0.0   # ...and how much of the turn this
    #                                surface itself does (geometry.span_path),
    #                                as a fraction of ITS own semi-span
    a: float | None = None           # section slope; None -> the first wing's
    alpha_L0: float | None = None


@dataclass
class VLMResult:
    CL: float
    CDi: float               # TREFFTZ-plane (far-field) induced drag
    e: float                 # span efficiency vs planar span: CL^2/(pi AR CDi)
    AR: float                # planar b^2/S (reference values, winglet excluded)
    S: float                 # reference area (wing planform)
    alpha: float             # root AoA [rad]
    CY: float                # side-force coefficient (symmetry diagnostic)
    Gamma: np.ndarray        # (Np,) circulation per panel
    cl: np.ndarray           # (Np,) section lift coefficient 2 Gamma/(V c)
    alpha_eff: np.ndarray    # (Np,) effective section AoA [rad] (alpha-method)
    y: np.ndarray            # (Np,) panel-midpoint y
    z: np.ndarray            # (Np,) panel-midpoint z
    c: np.ndarray            # (Np,) panel chord
    width: np.ndarray        # (Np,) panel width (spanwise arc length)
    is_winglet: np.ndarray   # (Np,) bool mask
    ly: np.ndarray | None = None   # (Np,) bound-segment y extent
    lz: np.ndarray | None = None   # (Np,) bound-segment z extent — with
    #                          Gamma this gives the PER-PANEL side force
    #                          rho V Gamma lz, which the total CY cancels by
    #                          symmetry but a structural model needs per side
    is_tail: np.ndarray | None = None   # (Np,) bool mask (tail panels)
    is_second: np.ndarray | None = None  # (Np,) bool mask (SecondWing panels)
    #   — the tandem pair's rear wing and ITS tip device. Carried on the
    #   result, not just on the model, because everything downstream of a
    #   solve (the breakdown, the geometry export, a view, a CAD loft) has to
    #   be able to tell the two wings apart: read as one array they draw as
    #   one surface, with the front wing's starboard device and the rear
    #   wing's port device joined into a single winglet across the aircraft.
    is_vertical: np.ndarray | None = None  # (Np,) bool mask (fin / mast)
    #   — and the SAME argument for the same reason, one surface later. This
    #   mask existed on the model (both solvers take the aero deck off it:
    #   ``wingtail``'s ``solid`` and ``tandemvlm``'s) and stopped there, so
    #   the geometry export could not tell a fin from the wing and kept its
    #   panels: y = 0, z climbing from the fin root to its tip, on the end of
    #   the wing's own spanwise polyline. Measured with the lateral deck armed
    #   (``handling_level=3``) on `tail [free cant]`, 40 wing stations became
    #   52 and the drawn wing turned ninety degrees at the root and ran up a
    #   fin that was ALSO being drawn from ``geometry["fin"]``.
    x: np.ndarray | None = None         # (Np,) bound-vortex station [m]
    i_t: float = 0.0                    # tail incidence flown [rad]
    Cm: float | None = None             # about the x_cg the caller asked for
    x_cg: float | None = None


def _mirror_span(s_signed: np.ndarray, semispan: float, h: float,
                 cant_deg: float, path_kw: dict
                 ) -> tuple[np.ndarray, np.ndarray]:
    """(y, z) of a SIGNED developed arc coordinate — port side mirrored.

    Only |s| carries geometry (the aircraft is symmetric); the sign picks the
    side. With no wing-side blend geometry.span_path is the identity on the
    wing, so this returns ``(s, 0)`` exactly and the published panel arrays
    are unchanged bit-for-bit.
    """
    y, z = span_path(np.abs(s_signed), semispan, h, cant_deg, **path_kw)
    return np.where(s_signed < 0.0, -y, y), z


def transition_ramp(s_from_junction: np.ndarray, is_device: np.ndarray,
                    h: float, cant_deg: float, blend_frac: float,
                    blend_shape: str, wing_arc: float) -> np.ndarray:
    """Fraction of the wing->device TRANSITION each panel has completed.

    ``s_from_junction`` is the panel station's developed arc measured from the
    junction (− inboard on the wing, + outboard on the device — the mirrored
    sides fold onto |arc| exactly as the geometry does). The value returned is
    the turn angle normalised by the cant,

        w(s) = psi(s) / phi   in [0, 1],

    which is 0 before the transition starts, 1 once the device is at its cant,
    and follows whichever turn law (:data:`geometry.BLEND_SHAPES`) is drawing
    the corner in between. It is what the device's chord scale, section and
    toe are interpolated on, so the surface finishes becoming the device
    exactly where it finishes turning into it.

    A SHARP corner has no turning arc, and geometry.winglet_turn_angle then
    reports the full cant everywhere; the second line below is what makes that
    the step it is — nothing turns before the transition starts, so every wing
    panel stays the wing and every device panel is fully the device. That is
    the published geometry, bit-for-bit, and it is the reason a run with
    ``blend_frac = 0`` cannot move.

    A zero cant is a coplanar tip extension: there is no turn to ramp on, and
    the device is the device from the junction out.
    """
    from .geometry import winglet_turn_angle

    phi = np.deg2rad(float(cant_deg))
    if phi == 0.0:
        return np.asarray(is_device, dtype=float)
    psi = winglet_turn_angle(s_from_junction, h, cant_deg, blend_frac,
                             blend_shape, wing_arc)
    w = np.clip(psi / phi, 0.0, 1.0)
    return np.where(np.asarray(s_from_junction) > -float(wing_arc), w, 0.0)


class VLM:
    """Build once per geometry (panels + LU-factored influence matrices),
    then every alpha / trim solve is a pair of back-substitutions —
    millisecond-scale, mirroring the build-once/eval-many MATLAB design."""

    def __init__(
        self,
        wing: Wing,
        N: int = 40,
        winglet_h_frac: float = 0.0,
        winglet_cant_deg: float = 90.0,
        n_winglet: int = 8,
        winglet_blend_frac: float = 0.0,
        winglet_blend_shape: str = "arc",
        winglet_wing_blend_frac: float = 0.0,
        a: float = TWO_PI,
        alpha_L0: float = 0.0,
        V: float = 1.0,
        image: "ImagePlane | None" = None,
        winglet_chord_scale: float = 1.0,
        winglet_chord_follows: bool = False,
        winglet_toe_deg: float = 0.0,
        winglet_a: float | None = None,
        winglet_alpha_L0: float | None = None,
        tail: "TailSurface | None" = None,
        second: "SecondWing | None" = None,
        vertical: "VerticalSurface | None" = None,
        S_ref: float | None = None,
    ):
        self.wing = wing
        self.image = image
        self.blend_frac = float(winglet_blend_frac)
        self.blend_shape = str(winglet_blend_shape)
        self.wing_blend_frac = float(winglet_wing_blend_frac)
        self.V = float(V)
        self.a = float(a)
        self.alpha_L0 = float(alpha_L0)
        if not (float(winglet_chord_scale) > 0.0):
            raise ValueError(
                f"winglet_chord_scale must be > 0, got {winglet_chord_scale}")
        self.winglet_chord_scale = float(winglet_chord_scale)
        self.winglet_chord_follows = bool(winglet_chord_follows)
        self.winglet_toe_deg = float(winglet_toe_deg)
        # a tip device with no section of its own inherits the wing's
        # (the published behaviour, kept bit-for-bit)
        self.winglet_a = float(a if winglet_a is None else winglet_a)
        self.winglet_alpha_L0 = float(
            alpha_L0 if winglet_alpha_L0 is None else winglet_alpha_L0)
        if not (self.winglet_a > 0.0):
            raise ValueError(
                f"winglet_a must be > 0, got {self.winglet_a}")
        b, S = wing.b, wing.S
        # a TANDEM pair is referenced to its TOTAL area (tandem.py's Sref),
        # not to the front wing's — the caller passes it, and everything
        # coefficient-shaped below reads self.S
        self.S = float(S if S_ref is None else S_ref)
        S = self.S
        self.AR = b**2 / S

        # ---- panel-edge polyline in (y, z): port winglet tip -> starboard ----
        # Edges at cosine angles k pi/N; STATIONS (collocation, chord/twist
        # sampling, Trefftz wash evaluation) at the interlaced cosine
        # midpoints (k+1/2) pi/N — see module docstring (pairing matters).
        #
        # The distribution is laid out in DEVELOPED ARC LENGTH (se, se_st,
        # signed: − is port) and then mapped through geometry.span_path to
        # (y, z). Without a wing-side blend the wing is flat and that map is
        # the identity on the wing, so the arrays are bit-for-bit the
        # published ones; with one, the outer wing bends up and only the map
        # changes — the cosine clustering, the panel count and the chord/twist
        # laws are untouched.
        h = float(winglet_h_frac) * b / 2.0
        # a dropped winglet drops its transition with it: a turn that starts
        # on the wing and has no device to finish on would bend the tip up
        # for nothing (and the same rule keeps the h = 0 limit planar)
        has_winglet = h >= MIN_WINGLET_FRAC * b / 2.0 and n_winglet > 0
        wbf = float(winglet_wing_blend_frac) if has_winglet else 0.0
        # WHAT WAS FLOWN, not what was asked for. Below MIN_WINGLET_FRAC the
        # device never enters the lattice, and a reader that recomputes the
        # height from the design variable reports a winglet nothing flew:
        # an optimum parked on this plateau then reads as a small device
        # (bound census u = 0.05) instead of as no device at all.
        self.has_winglet = bool(has_winglet)
        self.winglet_h_m = float(h) if has_winglet else 0.0
        self.winglet_h_frac_flown = (float(winglet_h_frac)
                                     if has_winglet else 0.0)
        self.winglet_h_m_second = 0.0
        self.winglet_h_m_tail = 0.0
        path_kw = dict(blend_frac=winglet_blend_frac,
                       blend_shape=winglet_blend_shape, wing_blend_frac=wbf)
        th = np.pi * np.arange(N + 1) / N
        se = -(b / 2.0) * np.cos(th)
        th_st = np.pi * (np.arange(N) + 0.5) / N
        se_st = -(b / 2.0) * np.cos(th_st)
        ye, ze = _mirror_span(se, b / 2.0, h, winglet_cant_deg, path_kw)
        yst, zst = _mirror_span(se_st, b / 2.0, h, winglet_cant_deg, path_kw)
        wl_panel = np.zeros(N, dtype=bool)
        if has_winglet:
            j = np.arange(1, n_winglet + 1)
            t = (h / 2.0) * (1.0 - np.cos(np.pi * j / n_winglet))  # cosine arc
            t_st = (h / 2.0) * (1.0 - np.cos(np.pi * (j - 0.5) / n_winglet))
            # the junction is where the wing's developed arc runs out; a
            # wing-side blend has already lifted it off the wing plane
            yj, zj = span_path(np.array([b / 2.0]), b / 2.0, h,
                               winglet_cant_deg, **path_kw)
            wing_arc = wbf * b / 2.0
            dy, zs = winglet_path(t, h, winglet_cant_deg, winglet_blend_frac,
                                  winglet_blend_shape, wing_arc)
            dy_st, zs_st = winglet_path(t_st, h, winglet_cant_deg,
                                        winglet_blend_frac,
                                        winglet_blend_shape, wing_arc)
            ys = yj[0] + dy
            ys_st = yj[0] + dy_st
            zs = zj[0] + zs
            zs_st = zj[0] + zs_st
            se = np.concatenate([-(b / 2.0 + t[::-1]), se, b / 2.0 + t])
            se_st = np.concatenate([-(b / 2.0 + t_st[::-1]), se_st,
                                    b / 2.0 + t_st])
            ye = np.concatenate([-ys[::-1], ye, ys])
            ze = np.concatenate([zs[::-1], ze, zs])
            yst = np.concatenate([-ys_st[::-1], yst, ys_st])
            zst = np.concatenate([zs_st[::-1], zst, zs_st])
            wl_panel = np.concatenate([
                np.ones(n_winglet, bool), wl_panel, np.ones(n_winglet, bool),
            ])

        # ---- DIHEDRAL: a rigid rotation of the assembled curve ----
        # Applied here, to the wing AND the device together, because the
        # device's cant is measured FROM THE WING PLANE — it turns with the
        # surface it is bolted to. Because a rotation is an isometry, every
        # array built above (developed arc, chord, twist, the cosine
        # clustering) is untouched, and the panel WIDTHS below are unchanged:
        # what moves is the direction each bound segment points, and with it
        # the normal, the y-extent that makes lift and the z-extent that makes
        # side force. That last one is the whole point: a dihedralled panel is
        # the only thing on a wing that answers a sideslip.
        self.dihedral_deg = float(getattr(wing, "dihedral_deg", 0.0) or 0.0)
        ye, ze = dihedral_rotate(ye, ze, self.dihedral_deg)
        yst, zst = dihedral_rotate(yst, zst, self.dihedral_deg)
        Np = ye.size - 1

        # ---- SWEEP: the quarter-chord line is no longer a straight line
        # across x = 0 ----
        # Every bound segment used to sit at x = 0 whatever ``sweep_deg`` had
        # been scored, so the lattice was unswept and the neutral point never
        # moved. Lambda is the PLANFORM-VIEW angle (the one a drawing is
        # dimensioned in), hence |y| after the dihedral rotation rather than
        # the developed arc: a vertical tip device then inherits its junction's
        # station, which is what a device standing at the tip of a swept wing
        # actually does.
        #
        # The section slope is NOT touched here. ``objective.py`` hands this
        # constructor the UNSWEPT ``a_lin/beta`` and applies simple sweep
        # theory only on the reduced-order lifting-line path, so the geometry
        # is owned here and the section slope there — never both.
        self.sweep_deg = float(getattr(wing, "sweep_deg", 0.0) or 0.0)
        tanL = np.tan(np.deg2rad(self.sweep_deg))
        xe = np.abs(ye) * tanL
        xst = np.abs(yst) * tanL

        A3 = np.column_stack([xe[:-1], ye[:-1], ze[:-1]])        # bound ends
        B3 = np.column_stack([xe[1:], ye[1:], ze[1:]])
        st3 = np.column_stack([xst, yst, zst])          # stations

        # chord & twist: main panels sample the wing at the station ARC
        # (identical to its y for a flat wing); winglet panels clip to the tip
        # and so carry the tip chord and tip twist.
        y_smp = np.clip(se_st, -b / 2.0, b / 2.0)
        c = wing.chord(y_smp)
        twist = np.deg2rad(wing.twist_deg(y_smp))
        # section properties are PER PANEL: the tip device may carry its own
        # aerofoil (see the module docstring). Defaults make both arrays
        # constant and equal to the wing's, i.e. the published model.
        a_panel = np.full(Np, self.a)
        aL0_panel = np.full(Np, self.alpha_L0)
        self.blend_ramp = np.zeros(Np)   # no device: nothing transitions
        if wl_panel.any():
            c_tip = float(wing.chord(np.array([b / 2.0]))[0])
            tw_tip = float(np.deg2rad(wing.twist_deg(np.array([b / 2.0]))[0]))
            toe = np.deg2rad(self.winglet_toe_deg)
            # how much of the TURN each panel has completed, in [0, 1] — the
            # ramp that makes a blend a blend (module docstring). Zero on the
            # wing before the transition starts, one once the device has
            # reached its cant. A sharp corner (no turning arc) has turn angle
            # = cant everywhere, so this is 0 on the wing and 1 on the device:
            # the STEP, bit-for-bit.
            self.blend_ramp = transition_ramp(
                np.abs(se_st) - b / 2.0, wl_panel, h, winglet_cant_deg,
                winglet_blend_frac, winglet_blend_shape, wbf * b / 2.0)
            w = self.blend_ramp
            # each quantity that makes the device a DIFFERENT surface from the
            # wing is interpolated on that ramp
            scale_eff = 1.0 + (self.winglet_chord_scale - 1.0) * w
            if self.winglet_chord_follows:
                # the chord law does not stop at the tip: sample it at the
                # device panel's OWN developed arc (eta > 1), which is the
                # same sampling the main panels get with the clip removed —
                # so wing and device are one chord distribution rather than a
                # distribution and a rectangle
                base = wing.chord(se_st)
                # the knife-edge guard is on the chord the device would CARRY,
                # i.e. before the ramp: the ramp only ever moves a device
                # chord between the tip chord and that value, so it can add no
                # knife edge of its own, and testing it here keeps the refusal
                # exactly the one the published problem makes
                c_dev = base * self.winglet_chord_scale
                thin = float(np.min(c_dev[wl_panel]))
                if thin <= DEVICE_CHORD_FLOOR * c_tip * self.winglet_chord_scale:
                    raise ValueError(
                        f"the chord law continued onto the tip device runs it "
                        f"to {thin:.4g} m, at or below "
                        f"{DEVICE_CHORD_FLOOR:g} of the tip chord "
                        f"({c_tip * self.winglet_chord_scale:.4g} m): that is "
                        f"a knife edge, not a surface. Bend the chord less, "
                        f"shorten the device, or switch "
                        f"winglet_chord_follows off")
            else:
                base = np.where(wl_panel, c_tip, c)
            c = base * scale_eff
            twist = np.where(wl_panel, tw_tip, twist) + toe * w
            a_panel = self.a + (self.winglet_a - self.a) * w
            aL0_panel = self.alpha_L0 + (self.winglet_alpha_L0
                                         - self.alpha_L0) * w

        # ---- optional SECOND WING panels ----
        # A tandem pair is two wings, not a wing and a rectangle: the rear
        # surface carries its own planform, chord law, twist and tip device.
        # Rather than restate the panelisation above (which would drift the
        # moment either copy changed), the second wing is panelised by THIS
        # SAME constructor on its own and its arrays are lifted out and
        # offset to the pair's stagger. The throwaway solve that costs is a
        # 40-panel LU — microseconds beside the coupled system it feeds.
        sd_panel = np.zeros(Np, dtype=bool)
        self.second = second
        if second is not None:
            sub_vlm = VLM(
                second.wing, N=second.N,
                winglet_h_frac=second.winglet_h_frac,
                winglet_cant_deg=second.winglet_cant_deg,
                n_winglet=second.n_winglet,
                winglet_blend_frac=second.winglet_blend_frac,
                winglet_blend_shape=second.winglet_blend_shape,
                winglet_wing_blend_frac=second.winglet_wing_blend_frac,
                # how a tip device takes its chord is a statement about the
                # DESIGN, not about one surface — the same rule that puts one
                # ChordLimits on every wing of a problem
                winglet_chord_follows=self.winglet_chord_follows,
                a=self.a if second.a is None else float(second.a),
                alpha_L0=(self.alpha_L0 if second.alpha_L0 is None
                          else float(second.alpha_L0)),
                V=self.V)
            off = np.array([float(second.x), 0.0, float(second.z)])
            n2 = sub_vlm.c.size
            A3 = np.vstack([A3, sub_vlm.A3 + off])
            B3 = np.vstack([B3, sub_vlm.B3 + off])
            # ...AT ITS OWN SWEPT STATION. A3/B3 above carry the
            # second wing's whole geometry (sub_vlm.A3 + off), sweep
            # included, and this line used to rebuild the STATIONS at the
            # constant ``second.x`` — so the control points (``cp = st3 + d
            # xhat``) sat square while the bound vortex they enforce the
            # tangency condition on was swept back by |y| tan(Lambda). At 20
            # deg on a 10 m pair the control point ended up 1.82 m AHEAD of
            # its own vortex, and the pair trimmed into nonsense: a stated
            # ``wing_sweep_deg`` of 6 deg or more came back "effective AoA
            # outside polar validity", blaming the SECTION for a lattice
            # that was drawn wrong. tan(0) is exactly 0.0, so an unswept
            # pair — every published run — is bit-for-bit unchanged.
            st3 = np.vstack([st3, np.column_stack([
                sub_vlm.st3[:, 0] + float(second.x), sub_vlm.y,
                sub_vlm.z + float(second.z)])])
            c = np.concatenate([c, sub_vlm.c])
            twist = np.concatenate([twist, sub_vlm.twist])
            a_panel = np.concatenate([a_panel, sub_vlm.a_panel])
            aL0_panel = np.concatenate([aL0_panel, sub_vlm.alpha_L0_panel])
            wl_panel = np.concatenate([wl_panel, sub_vlm.is_winglet])
            sd_panel = np.concatenate([sd_panel, np.ones(n2, bool)])
            Np = Np + n2
            # the sub-lattice is discarded below; its FLOWN device height is
            # the only thing a report cannot recompute (the drop rule)
            self.winglet_h_m_second = sub_vlm.winglet_h_m

        # ---- optional TAIL panels, appended to the same system ----
        # A second surface changes nothing structural: the horseshoe kernel
        # is position-agnostic, and the Trefftz plane collapses x (Munk's
        # stagger theorem), so the tail's wake trace joins the wing's in the
        # same far-field sum. Its INCIDENCE is the one thing held out — it is
        # a trim unknown, carried below as a third basis solution.
        tl_panel = np.zeros(Np, dtype=bool)
        if tail is not None and tail.designed:
            # a DESIGNED tail is a wing, so it is panelised by this same
            # constructor (the second-wing path above, verbatim) and only
            # its MASK differs: these panels take the trim incidence.
            sub_t = VLM(
                tail.wing, N=tail.N,
                winglet_h_frac=tail.winglet_h_frac,
                winglet_cant_deg=tail.winglet_cant_deg,
                n_winglet=tail.n_winglet,
                winglet_blend_frac=tail.winglet_blend_frac,
                winglet_blend_shape=tail.winglet_blend_shape,
                winglet_wing_blend_frac=tail.winglet_wing_blend_frac,
                winglet_chord_follows=self.winglet_chord_follows,
                a=self.a if tail.a is None else float(tail.a),
                alpha_L0=(self.alpha_L0 if tail.alpha_L0 is None
                          else float(tail.alpha_L0)),
                V=self.V)
            off_t = np.array([float(tail.x), 0.0, float(tail.z)])
            n_t = sub_t.c.size
            # the cant, if it was stated on the TailSurface rather than on the
            # tail's own Wing. Only one of the two can be set (__post_init__),
            # so a wing-stated cant is already baked into ``sub_t`` and this
            # rotation is the identity — never applied twice.
            gam_t = float(tail.dihedral_deg)
            a_y, a_z = dihedral_rotate(sub_t.A3[:, 1], sub_t.A3[:, 2], gam_t)
            b_y, b_z = dihedral_rotate(sub_t.B3[:, 1], sub_t.B3[:, 2], gam_t)
            s_y, s_z = dihedral_rotate(sub_t.y, sub_t.z, gam_t)
            A3 = np.vstack([A3, np.column_stack([sub_t.A3[:, 0], a_y, a_z])
                            + off_t])
            B3 = np.vstack([B3, np.column_stack([sub_t.B3[:, 0], b_y, b_z])
                            + off_t])
            # the same station-versus-vortex agreement the second wing
            # needs above: ``A3``/``B3`` take ``sub_t.A3[:, 0]``, so the
            # stations must take ``sub_t.st3[:, 0]``. No registered family
            # hands this path a swept tail today, which is exactly why it
            # would have drifted — one of a pair left fixed is how the two
            # stop agreeing.
            st3 = np.vstack([st3, np.column_stack([
                sub_t.st3[:, 0] + float(tail.x), s_y, s_z + float(tail.z)])])
            c = np.concatenate([c, sub_t.c])
            # the tail's own twist is a WASHOUT about the trim incidence
            # (TailSurface docstring): the uniform part is i_t itself
            twist = np.concatenate([twist, sub_t.twist])
            a_panel = np.concatenate([a_panel, sub_t.a_panel])
            aL0_panel = np.concatenate([aL0_panel, sub_t.alpha_L0_panel])
            wl_panel = np.concatenate([wl_panel, sub_t.is_winglet])
            tl_panel = np.concatenate([tl_panel, np.ones(n_t, bool)])
            sd_panel = np.concatenate([sd_panel, np.zeros(n_t, bool)])
            Np = Np + n_t
            self.winglet_h_m_tail = sub_t.winglet_h_m
        elif tail is not None:
            b_t = tail.span
            th_t = np.pi * np.arange(tail.N + 1) / tail.N
            ye_t = -(b_t / 2.0) * np.cos(th_t)
            yst_t = -(b_t / 2.0) * np.cos(
                np.pi * (np.arange(tail.N) + 0.5) / tail.N)
            xs = np.full(tail.N, float(tail.x))
            # A V-TAIL IS THIS LINE. The surface is canted about its own root
            # — the same rigid rotation the wing gets, applied relative to
            # ``tail.z`` so the junction stays where the layout put it. Its
            # span here is the PANEL span, so the horizontal projection comes
            # out as b_t cos(Gamma) and the lift as S cos^2(Gamma) of a flat
            # tail, with the second cosine coming from the normal the rotation
            # produced. Neither is coded: both are consequences.
            ye_t, ze_t = dihedral_rotate(ye_t, np.zeros(tail.N + 1),
                                         float(tail.dihedral_deg))
            yst_t, zst_t = dihedral_rotate(yst_t, np.zeros(tail.N),
                                           float(tail.dihedral_deg))
            ze_t = ze_t + float(tail.z)
            zst_t = zst_t + float(tail.z)
            A3 = np.vstack([A3, np.column_stack([xs, ye_t[:-1], ze_t[:-1]])])
            B3 = np.vstack([B3, np.column_stack([xs, ye_t[1:], ze_t[1:]])])
            st3 = np.vstack([st3, np.column_stack([xs, yst_t, zst_t])])
            c = np.concatenate([c, np.full(tail.N, tail.chord)])
            # zero geometric twist: the tail's incidence is the trim variable
            twist = np.concatenate([twist, np.zeros(tail.N)])
            a_panel = np.concatenate([
                a_panel, np.full(tail.N,
                                 self.a if tail.a is None else float(tail.a))])
            aL0_panel = np.concatenate([
                aL0_panel,
                np.full(tail.N, self.alpha_L0 if tail.alpha_L0 is None
                        else float(tail.alpha_L0))])
            wl_panel = np.concatenate([wl_panel, np.zeros(tail.N, bool)])
            tl_panel = np.concatenate([tl_panel, np.ones(tail.N, bool)])
            sd_panel = np.concatenate([sd_panel, np.zeros(tail.N, bool)])
            Np = Np + tail.N

        # ---- the VERTICAL surface (fin / rudder, or a hydrofoil strut) ----
        # A fin needs no new panel machinery: it is this same lattice with its
        # bound segments run along z instead of y. The normal construction
        # downstream is n0 = x_hat x t_hat, so t_hat = z_hat gives n0 = -y_hat
        # — a panel that carries SIDE force — and the twist rotation about
        # t_hat then sweeps that normal through x, which is exactly a rudder
        # deflection. Nothing below this point knows the difference, including
        # the Trefftz plane: a fin's wake trace is a vertical line in (y, z)
        # and the existing kernel integrates it.
        vt_panel = np.zeros(Np, bool)
        if vertical is not None:
            nv = int(vertical.N)
            # SAME quadrature as every other surface: edges at the cosine
            # angles, stations at the INTERLACED midpoints. Getting this
            # wrong biases the Trefftz sum exactly as the module docstring
            # warns for the wing.
            th_v = np.pi * np.arange(nv + 1) / nv
            frac_e = 0.5 * (1.0 - np.cos(th_v))
            frac_s = 0.5 * (1.0 - np.cos(np.pi * (np.arange(nv) + 0.5) / nv))
            ze_v = vertical.z_root + vertical.height * frac_e
            zst_v = vertical.z_root + vertical.height * frac_s
            xs_v = np.full(nv, float(vertical.x))
            ys_v = np.full(nv, float(vertical.y))
            A3 = np.vstack([A3, np.column_stack([xs_v, ys_v, ze_v[:-1]])])
            B3 = np.vstack([B3, np.column_stack([xs_v, ys_v, ze_v[1:]])])
            st3 = np.vstack([st3, np.column_stack([xs_v, ys_v, zst_v])])
            c = np.concatenate([c, np.full(nv, float(vertical.chord))])
            # the fin's "twist" is its RUDDER DEFLECTION plus any fixed toe:
            # one uniform angle about the vertical bound line
            twist = np.concatenate([
                twist, np.full(nv, np.deg2rad(float(vertical.toe_deg)))])
            a_panel = np.concatenate([
                a_panel, np.full(nv, self.a if vertical.a is None
                                 else float(vertical.a))])
            aL0_panel = np.concatenate([
                aL0_panel,
                np.full(nv, 0.0 if vertical.alpha_L0 is None
                        else float(vertical.alpha_L0))])
            wl_panel = np.concatenate([wl_panel, np.zeros(nv, bool)])
            tl_panel = np.concatenate([tl_panel, np.zeros(nv, bool)])
            sd_panel = np.concatenate([sd_panel, np.zeros(nv, bool)])
            vt_panel = np.concatenate([vt_panel, np.ones(nv, bool)])
            Np = Np + nv

        lvec = B3 - A3
        width = np.linalg.norm(lvec, axis=1)
        that = lvec / width[:, None]
        # ...and the length of the same segment's WAKE TRACE, which is its
        # (y, z) extent alone. Munk's stagger theorem collapses x, so a swept
        # panel's trace is SHORTER than the panel: charging the Trefftz sum
        # the full 3-D length would inflate CDi by 1/cos(Lambda) for no
        # physical reason. An unswept lattice takes ``width`` ITSELF rather
        # than a hypot that happens to agree: np.hypot and np.linalg.norm are
        # free to differ in the last ulp, and every published induced drag in
        # this package is quoted from an unswept lattice.
        width_wake = (width if not np.any(lvec[:, 0])
                      else np.hypot(lvec[:, 1], lvec[:, 2]))

        # panel normals: zero-incidence normal n0 = x_hat x t_hat, rotated
        # nose-up about the local spanwise axis by theta = twist - alpha_L0
        # (Rodrigues; reduces to the MATLAB Rx*Ry*[0;0;1] construction).
        n0 = np.cross(np.broadcast_to(_XHAT, that.shape), that)
        n0 /= np.linalg.norm(n0, axis=1)[:, None]
        theta = twist - aL0_panel
        nrm = n0 * np.cos(theta)[:, None] + np.cross(that, n0) * np.sin(theta)[:, None]

        # control points: d = a c/(4 pi) aft of the bound vortex (3/4-chord
        # point for a = 2 pi) -> section slope `a` is reproduced exactly.
        d = a_panel * c / FOUR_PI
        cp = st3 + d[:, None] * _XHAT

        # ---- influence matrix, LU-factored once ----
        vel = _hshoe(cp, A3, B3)                          # (Np, Np, 3)
        if image is not None:
            z_all = np.concatenate([A3[:, 2], B3[:, 2], st3[:, 2]])
            # crossing is reported first: it names the real geometry error
            # (a winglet through the free surface), while the clearance test
            # below catches merely-too-close surfaces
            if not (np.all(z_all <= image.z) or np.all(z_all >= image.z)):
                raise ValueError(
                    f"surface crosses the {image.kind} plane at z = "
                    f"{image.z:g}")
            gap = float(np.abs(z_all - image.z).min())
            if gap < MIN_PLANE_CLEARANCE_FRAC * b / 2.0:
                raise ValueError(
                    f"surface comes within {gap:.4g} m of the {image.kind} "
                    f"plane (limit {MIN_PLANE_CLEARANCE_FRAC * b / 2.0:.4g} "
                    f"m): the image system overlaps the real one")
            # the image carries the SAME unknowns (its circulation is the
            # real one, signed by the boundary condition), so this stays a
            # (Np, Np) system — no new degrees of freedom
            vel = vel + image.sign * _hshoe(cp, image.mirror(A3),
                                            image.mirror(B3))
        AIC = np.einsum("ijk,ik->ij", vel, nrm)
        self._lu = lu_factor(AIC)

        # linear-in-alpha circulation: rhs(alpha) = -V (n_x + alpha n_z)
        self._Gam0 = lu_solve(self._lu, -self.V * nrm[:, 0])
        self._Gam1 = lu_solve(self._lu, -self.V * nrm[:, 2])
        # ...and linear-in-i_t: a tail incidence enters the LINEARISED
        # boundary condition as one more uniform RHS shift on the tail
        # panels, tau = (t_hat x n).x_hat (exactly 1 for a level tail). The
        # O(i_t) change to the influence matrix itself multiplies an induced
        # velocity that is already O(alpha), so it is second order — the same
        # small-disturbance argument that makes the alpha-linearisation
        # exact-to-first-order, and what keeps CL and Cm AFFINE in (alpha,
        # i_t) so trim is a closed-form 2x2 solve rather than an iteration.
        self._Gam2 = None
        if tail is not None:
            tau = np.einsum("ij,j->i", np.cross(that, nrm), _XHAT)
            tau = np.where(tl_panel, tau, 0.0)
            self._Gam2 = lu_solve(self._lu, -self.V * tau)

        # ---- Trefftz-plane influence: wake trace = panel edges in (y, z),
        # wash evaluated at the interlaced stations ----
        a2 = A3[:, 1:].copy()
        b2 = B3[:, 1:].copy()
        p2 = st3[:, 1:].copy()
        n2d = np.column_stack([-that[:, 2], that[:, 1]])  # x_hat x t_hat in 2-D
        n2d /= np.linalg.norm(n2d, axis=1)[:, None]
        W = _trefftz_kernel(p2, b2) - _trefftz_kernel(p2, a2)  # (Np, Np, 2)
        if image is not None:
            # the image's wake traces are the real ones mirrored in z; the
            # trailing legs still run downstream, so the 2-D strength map is
            # unchanged and only the boundary-condition sign enters. The
            # wash is evaluated at the REAL stations only — the image is not
            # a surface and carries no force (cf. hydrofoil.CDi_surf, which
            # is likewise counted once, on the real foil).
            a2i = image.mirror(a2, axis=1)
            b2i = image.mirror(b2, axis=1)
            W = W + image.sign * (_trefftz_kernel(p2, b2i)
                                  - _trefftz_kernel(p2, a2i))
        self._WN = np.einsum("ijk,ik->ij", W, n2d)

        self.A3 = A3          # bound-vortex ends (kept for image/BC checks)
        self.B3 = B3
        self.lvec = lvec
        self.width = width
        self.width_wake = width_wake
        self.c = c
        # the bound segment's MIDPOINT station. On an unswept lattice A3 and
        # B3 share an x, so this is 0.5*(v + v) = v exactly and every
        # published moment is bit-for-bit; on a SWEPT one the segment spans a
        # range of x and taking its inboard end would bias every panel's
        # moment arm by half a panel's sweep offset.
        self.x = 0.5 * (A3[:, 0] + B3[:, 0])   # bound-vortex (c/4) station
        self.y = st3[:, 1].copy()
        self.z = st3[:, 2].copy()
        # The panel normals and control points are what the boundary condition
        # is written AT, so anything that adds a new right-hand side needs
        # them: a body rate contributes -(Omega x r) at the control point, a
        # sideslip a uniform -V beta y_hat, a control-surface deflection a
        # rotation of the normal. They were local to the build until
        # dynamics.py needed them; recomputing them outside (as the session-65
        # prototype did) duplicates the Rodrigues construction above and would
        # drift from it silently. self.st3 is the bound-vortex STATION (the
        # force acts there); self.cp is the flow-tangency point, d aft of it.
        self.nrm = nrm
        self.cp = cp
        self.st3 = st3
        self.is_winglet = wl_panel
        self.is_tail = tl_panel
        self.is_second = sd_panel
        self.is_vertical = vt_panel
        self.vertical = vertical
        self.twist = twist
        self.tail = tail
        self.a_panel = a_panel
        self.alpha_L0_panel = aL0_panel
        # linear CL(alpha) = CL0 + alpha * CL_alpha (Kutta-Joukowski, freestream)
        self._CL0 = 2.0 * float(self._Gam0 @ lvec[:, 1]) / (self.V * S)
        self._CLa = 2.0 * float(self._Gam1 @ lvec[:, 1]) / (self.V * S)
        self._CLi = (2.0 * float(self._Gam2 @ lvec[:, 1]) / (self.V * S)
                     if self._Gam2 is not None else 0.0)

    # ------------------------------------------------------------------
    @property
    def n_panels(self) -> int:
        return self.c.size

    @property
    def has_tail(self) -> bool:
        return self._Gam2 is not None

    def _gamma(self, alpha: float, i_t: float = 0.0) -> np.ndarray:
        Gam = self._Gam0 + float(alpha) * self._Gam1
        if self._Gam2 is not None:
            Gam = Gam + float(i_t) * self._Gam2
        return Gam

    def _cm(self, Gam: np.ndarray, x_cg: float, mac: float) -> float:
        """Pitching moment about ``x_cg``, NOSE-UP POSITIVE, on (S, mac).

        Per PANEL rather than per surface: each bound segment carries
        dL/q = 2 Gamma l_y and acts at its own quarter-chord station x_i, so

            Cm = sum_i (x_cg - x_i) * 2 Gamma_i l_y,i / (V S mac)

        which is tail.py's two-surface formula at panel resolution (lift aft
        of the CG pitches nose-down — the sign the static-margin gate leans
        on). Winglet panels are included through their own l_y, exactly as
        they are in CL.
        """
        dcl = 2.0 * Gam * self.lvec[:, 1] / (self.V * self.S)
        return float(np.sum((float(x_cg) - self.x) * dcl) / float(mac))

    def solve(self, alpha: float, i_t: float = 0.0,
              x_cg: float | None = None, mac: float | None = None,
              cm_ac: float = 0.0) -> VLMResult:
        """Solve at root AoA ``alpha`` [rad] (cached back-substitutions).

        ``i_t`` is the tail incidence [rad] (ignored without a tail). Pass
        ``x_cg`` and ``mac`` to get the pitching moment about that CG in the
        result; leaving them out keeps the published single-surface result
        exactly as it was.

        ``cm_ac`` is the SECTIONS' OWN COUPLE, already non-dimensionalised
        on this model's (S, mac) — a constant this lattice cannot produce
        for itself: with ONE chordwise panel per strip every bound vortex
        sits on the quarter-chord line, so the lattice yields lift-times-arm
        and nothing else. A cambered section's nose-down couple is real and
        does not vanish with lift (tail.section_moment derives it), so the
        caller, which knows its surfaces' polars, hands it in. 0.0 is the
        symmetric-section value and reproduces every earlier result.
        """
        Gam = self._gamma(alpha, i_t)
        V, S = self.V, self.S

        CL = 2.0 * float(Gam @ self.lvec[:, 1]) / (V * S)
        CY = -2.0 * float(Gam @ self.lvec[:, 2]) / (V * S)

        wn = self._WN @ Gam                               # normal wash at traces
        CDi = -float((self.width_wake * Gam) @ wn) / (V**2 * S)
        e = (CL**2 / (np.pi * self.AR * CDi)) if CDi > 1e-14 else float("nan")

        cl = 2.0 * Gam / (V * self.c)
        # alpha-method: the effective angle is the one that WOULD produce
        # this section cl in 2-D, so it already carries everything the panel
        # is flying — twist, tail incidence and the induced field alike.
        # (Adding i_t on top of it would count the incidence twice.)
        alpha_eff = self.alpha_L0_panel + cl / self.a_panel

        return VLMResult(
            CL=CL, CDi=CDi, e=float(e), AR=float(self.AR), S=S,
            alpha=float(alpha), CY=CY, Gamma=Gam, cl=cl, alpha_eff=alpha_eff,
            y=self.y, z=self.z, c=self.c, width=self.width,
            is_winglet=self.is_winglet,
            ly=self.lvec[:, 1].copy(), lz=self.lvec[:, 2].copy(),
            is_tail=self.is_tail.copy(), is_second=self.is_second.copy(),
            is_vertical=self.is_vertical.copy(),
            x=self.x.copy(), i_t=float(i_t),
            Cm=(None if x_cg is None or mac is None
                else self._cm(Gam, x_cg, mac) + float(cm_ac)),
            x_cg=(None if x_cg is None else float(x_cg)),
        )

    # ------------------------------------------------------------------
    # two-surface trim + stability (exactly affine — see the tail block above)

    def lift_derivatives(self) -> tuple[np.ndarray, np.ndarray]:
        """Per-panel d(dCL_i)/d(alpha) and d(dCL_i)/d(i_t).

        These ARE the coupled lift-curve slopes: the wing's carries the
        tail's upwash back-reaction and the tail's carries the downwash lag,
        because both come out of the one coupled solve rather than a
        handbook d(eps)/d(alpha).
        """
        k = 2.0 * self.lvec[:, 1] / (self.V * self.S)
        d_alpha = self._Gam1 * k
        d_it = (self._Gam2 * k if self._Gam2 is not None
                else np.zeros_like(d_alpha))
        return d_alpha, d_it

    def neutral_point(self) -> float:
        """x_np [m]: the lift-weighted station of the alpha-derivative.

        x_np = sum_i x_i (dCL_i/dalpha) / sum_i (dCL_i/dalpha) — tail.py's
        two-surface expression at panel resolution. With SM = (x_np - x_cg)/mac
        this satisfies Cm_alpha = -CL_alpha * SM identically (gated in tests).
        """
        d_alpha, _ = self.lift_derivatives()
        tot = float(np.sum(d_alpha))
        if abs(tot) < 1e-14:
            raise ValueError("degenerate system lift-curve slope")
        return float(np.sum(self.x * d_alpha) / tot)

    def solve_trim_moment(
        self,
        CL_target: float,
        x_cg: float,
        mac: float,
        alpha_bracket: tuple[float, float] = (np.deg2rad(-10.0),
                                              np.deg2rad(15.0)),
        cm_ac: float = 0.0,
    ) -> tuple[float, float, VLMResult]:
        """(alpha, i_t, result) for CL = CL_target AND Cm(x_cg) = 0.

        Both CL and Cm are EXACTLY affine in (alpha, i_t) here — the three
        basis circulations are the whole map — so this is a closed-form 2x2
        solve, not an iteration. Same structure as tail.py's elimination,
        with the Jacobian read off the basis solutions instead of finite
        differences.

        ``cm_ac`` (:meth:`solve`) enters the CONSTANT of that map only: it
        is independent of (alpha, i_t), so it shifts ``Cm0`` and leaves the
        Jacobian — and therefore :meth:`neutral_point` and the static margin
        — untouched. Trimming it out is what makes the stabiliser react the
        wing's couple rather than ignore it.
        """
        if self._Gam2 is None:
            raise ValueError("no tail to trim in pitch")
        CL0 = self._CL0
        Cm0 = self._cm(self._Gam0, x_cg, mac) + float(cm_ac)
        # columns of the Jacobian: the alpha- and i_t-basis solutions
        J = np.array([[self._CLa, self._CLi],
                      [self._cm(self._Gam1, x_cg, mac),
                       self._cm(self._Gam2, x_cg, mac)]])
        if abs(np.linalg.det(J)) < 1e-14:
            raise ValueError("degenerate trim Jacobian")
        sol = np.linalg.solve(J, np.array([CL_target - CL0, -Cm0]))
        alpha, i_t = float(sol[0]), float(sol[1])
        if not (alpha_bracket[0] <= alpha <= alpha_bracket[1]):
            raise ValueError(
                f"trim alpha {np.rad2deg(alpha):.2f} deg outside bracket")
        return alpha, i_t, self.solve(alpha, i_t, x_cg=x_cg, mac=mac,
                                      cm_ac=cm_ac)

    def solve_trim(
        self,
        CL_target: float,
        alpha_bracket: tuple[float, float] = (np.deg2rad(-10.0), np.deg2rad(15.0)),
    ) -> tuple[float, VLMResult]:
        """Root AoA for CL = CL_target — closed form (CL is exactly linear in
        alpha here, cf. adjoint.py). Raises ValueError outside the bracket,
        mirroring solve_llt_trim's brentq failure contract."""
        if abs(self._CLa) < 1e-12:
            raise ValueError("degenerate lift-curve slope")
        alpha = (CL_target - self._CL0) / self._CLa
        if not (alpha_bracket[0] <= alpha <= alpha_bracket[1]):
            raise ValueError(
                f"trim alpha {np.rad2deg(alpha):.2f} deg outside bracket"
            )
        return float(alpha), self.solve(alpha)


def solve_vlm(
    wing: Wing,
    alpha: float,
    N: int = 40,
    winglet_h_frac: float = 0.0,
    winglet_cant_deg: float = 90.0,
    n_winglet: int = 8,
    winglet_blend_frac: float = 0.0,
    winglet_blend_shape: str = "arc",
    a: float = TWO_PI,
    alpha_L0: float = 0.0,
    V: float = 1.0,
    image: "ImagePlane | None" = None,
    **winglet_section,
) -> VLMResult:
    """One-shot convenience wrapper (build + solve).

    ``winglet_section`` passes the tip device's own section through
    (``winglet_chord_scale`` / ``winglet_chord_follows`` /
    ``winglet_toe_deg`` / ``winglet_a`` / ``winglet_alpha_L0``); empty means
    the tip-extension model.
    """
    return VLM(
        wing, N=N, winglet_h_frac=winglet_h_frac,
        winglet_cant_deg=winglet_cant_deg, n_winglet=n_winglet,
        winglet_blend_frac=winglet_blend_frac,
        winglet_blend_shape=winglet_blend_shape,
        a=a, alpha_L0=alpha_L0, V=V, image=image, **winglet_section,
    ).solve(alpha)
