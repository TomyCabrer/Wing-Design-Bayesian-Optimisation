"""Design vector -> wing geometry (Tier A: trapezoidal planform, linear twist).

Fixed span b and area S (AR = b^2/S ~ 10 for the baseline); the design
vector controls taper and the linear twist law:

    x_trim    = [taper, twist_root_deg, twist_tip_deg]           (trim mode)
    x_soft    = [taper, twist_root_deg, twist_tip_deg, alpha_deg] (soft mode)
    x_a+      = [taper, twist_root_deg, twist_tip_deg,
                 sweep_deg, tc]                             (tier_a_plus mode)
    x_winglet = [taper, twist_root_deg, twist_tip_deg,
                 winglet_h_frac, winglet_cant_deg]  (Tier A+ winglet mode:
                 trim vector + winglet height fraction of the semi-span and
                 cant angle from the wing plane — see vlm.py)
    x_mission = [taper, twist_root_deg, twist_tip_deg,
                 V_ms, altitude_m]                       (mission mode:
                 speed and altitude are bounded design variables; the trim
                 CL_target is recomputed per candidate from the mission
                 design weight — see mission.py and objective.evaluate)

Any mode may additionally carry a CHORD LAW: ``chord_order`` coefficients
APPENDED to the end of the vector (so every existing index keeps its
meaning), which reshape the straight-taper baseline into a general chord
distribution at UNCHANGED area — see :func:`chord_multiplier` and
:attr:`Wing.chord_coeffs`. ``chord_order = 0`` (the default) adds no
variables and reproduces the trapezoidal planform bit-for-bit.

Sign convention: twist is LOCAL INCIDENCE in degrees, + = nose-up
(added to the root alpha). Washout (tip nose-down) is therefore a
NEGATIVE tip twist, matching the kickoff bounds tip in [-6, 2] deg.

Tier A+ fields: ``sweep_deg`` is the QUARTER-CHORD sweep angle (deg, aft
positive); it enters the physics only as reduced-order corrections (simple
sweep theory on the section lift-curve slope, Raymer sweep term in the form
factor) — the planform used by the LLT stays geometrically unswept.
``tc`` is the section thickness ratio, which selects/interpolates the
section polar from the NACA 24XX PolarFamily (polar.py).

Remaining extension points: winglets, per-station twist knots — add fields
here and rows to `bounds()`; llt.py already takes per-station arrays.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .llt import cosine_stations

#: WHAT FRAME A REPORT WAS SOLVED IN, where it is not the package's own.
#: A car family models the vehicle MIRRORED about the horizontal plane, so
#: the model's +z is the CAR'S DOWNWARD direction: model lift IS downforce
#: (carwing._), the ground plane sits ABOVE the wing, and an endplate that
#: reaches down for the deck reaches +z in the model. Three solvers write
#: this string (``carwing``, ``carwing_multi``, ``endplate``) and every
#: drawer has to be able to ask, so it is declared once, here, beside the
#: geometry it describes rather than restated in each of them.
MIRRORED_FRAME = "mirrored: model +z is the car's downward direction"


def is_mirrored(block) -> bool:
    """True where this report block was solved in the mirrored (car) frame.

    Accepts either half of a report — the breakdown, which is where the
    solvers write it, or the ``geometry`` block ``api.design_report`` carries
    it out on — because the drawers only ever hold the second one.

    A picture of a mirrored design is UPSIDE DOWN unless the drawer flips it:
    the endplates stand above the wing and the track is the plane above them.
    """
    if not isinstance(block, dict):
        return False
    return str(block.get("frame") or "").startswith("mirrored")


TAPER_BOUNDS = (0.2, 1.0)
TWIST_ROOT_BOUNDS_DEG = (-4.0, 4.0)
TWIST_TIP_BOUNDS_DEG = (-6.0, 2.0)
ALPHA_BOUNDS_DEG = (-2.0, 8.0)
SWEEP_BOUNDS_DEG = (0.0, 30.0)   # quarter-chord sweep, aft positive
#: wing dihedral, tips-up positive. The upper end is anhedral-to-gull
#: territory rather than a handling limit: it is wide enough that the SPIRAL
#: gate, not the box, is what stops the search — a calibration is a default,
#: not a ban. The lower end is negative because anhedral is a real answer
#: (a high wing with too much effective dihedral asks for it).
DIHEDRAL_BOUNDS_DEG = (-10.0, 15.0)
TC_BOUNDS = (0.08, 0.16)         # inside the NACA 24XX family (0.06-0.18)
#: the SUMMARY-VARIABLE section box: the hull of the pre-optimised CST
#: section library (the 3 x 3 (t/c, cl_design) grid of
#: scripts/gen_cst_library.py, read through polar.PolarFamily2D). These are
#: the two rows the COUPLED families search instead of a shape — a demand on
#: a section team, in the hierarchical-decomposition sense
#: (Sobieszczanski-Sobieski & Haftka 1997), not a geometry.
#:
#: They live here, beside every other design-box row, because
#: ``geometry.bounds`` builds the winglet+coupled box and geometry may not
#: import wing_airfoil (which imports geometry). ``wing_airfoil`` re-exports
#: them under its published names, so there is exactly one pair of numbers.
TC_SEC_BOUNDS = (0.09, 0.15)
CL_SEC_BOUNDS = (0.3, 0.7)
#: the objective.Problem modes whose vector carries those two rows — the
#: winglet families that fly the coupled section library. Defined HERE, not
#: in objective.py, because three of the places that must agree about them
#: are in this module (the design box, the thickness index and the span cap)
#: and a second literal is a second thing to forget; objective.py re-exports
#: it and a registry-wide test gates the membership rules
#: (subset of WINGLET_MODES, disjoint from TC_MODES).
COUPLED_MODES = ("winglet_coupled", "winglet_capped_coupled")
#: span-capped winglet modes whose cap is the plain COSINE projection
#: b_wing = b (1 - h_frac cos(cant)). The capped BLENDED modes are not here:
#: a blended device leaves the wing plane tangentially and so reaches further
#: outboard than h cos(cant), and they are capped on the developed line
#: instead (see wing_from_x). Together the two sets are
#: objective.CAPPED_MODES, and a test gates that.
CAPPED_COSINE_MODES = ("winglet_capped", "winglet_capped_tc",
                       "winglet_capped_coupled")
WINGLET_H_FRAC_BOUNDS = (0.0, 0.15)     # winglet height / semi-span


def winglet_h_row(h_bounds, default=WINGLET_H_FRAC_BOUNDS) -> tuple:
    """The device-HEIGHT design-box row (fraction of semi-span), validated.

    :func:`tail.arm_row`'s twin for the one tip-device row that never
    travelled. :data:`WINGLET_H_FRAC_BOUNDS` is where the nonplanar VLM was
    MEASURED, and this repo's rule is that a calibration is a default and
    never a ban: at the published box design the solver is smooth, finite
    and feasible from 0 to 1.5 semi-spans, with an INTERIOR optimum near
    h_frac 0.45 (L/D 39.233 against 37.768 at the 0.15 ceiling) — so a
    device taller than the calibrated band is a design decision, and the
    reason the row rides its ceiling in four of five families is the
    ceiling.

    What is refused is exactly what the samplers and the panel builder
    need: a PAIR, two FINITE numbers, non-collapsed, low end not negative,
    low end first. A collapsed row is diagnosed as a PIN wherever it
    collapsed. There is NO ceiling.

    ``None`` -> the published band, so an untouched run is the published
    run bit-for-bit.
    """
    if h_bounds is None:
        return (float(default[0]), float(default[1]))
    try:
        lo, hi = float(h_bounds[0]), float(h_bounds[1])
    except (TypeError, IndexError, ValueError, KeyError):
        raise ValueError(
            f"the tip-device height design-box row must be a pair (lo, hi) "
            f"as a fraction of the semi-span; got {h_bounds!r}") from None
    if not (np.isfinite(lo) and np.isfinite(hi)):
        raise ValueError(
            f"the tip-device height row ({lo}, {hi}) is not finite")
    if hi == lo:
        raise ValueError(
            f"the tip-device height row ({lo}, {hi}) has zero width. That "
            f"is a PIN written as a bound, and this repo does pins the one "
            f"honest way: RunConfig.pinned takes the height OUT of the "
            f"design vector, where a collapsed box instead breaks the Sobol "
            f"engine and silently degrades BO to random draws on every "
            f"iteration while still calling itself BO.")
    if lo < 0.0:
        raise ValueError(
            f"the tip-device height row ({lo}, {hi}) starts below zero. The "
            f"height is a MAGNITUDE — which way the device points is the "
            f"CANT's sign (geometry.WINGLET_CANT_LIMITS_SIGNED_DEG), and a "
            f"negative height would mirror it twice. No upper bound is "
            f"imposed: {tuple(default)} is where this model was measured, "
            f"not a cap on the device you are designing.")
    if hi < lo:
        raise ValueError(
            f"the tip-device height row ({lo}, {hi}) has hi < lo")
    return (lo, hi)

#: fraction of the winglet's arc length spent turning out of the wing plane
#: (0 = sharp corner = legacy geometry, 1 = pure arc). See winglet_path.
WINGLET_BLEND_BOUNDS = (0.0, 1.0)
#: fraction of the SEMI-SPAN spent turning INBOARD of the tip — the wing-side
#: half of a blended transition (see span_path). 0 = the whole turn sits on
#: the winglet, which is the legacy geometry, bit-for-bit.
#:
#: Why this exists: with the turn confined to the winglet the transition arc
#: is at most ``blend_frac * h <= 0.15 * b/2``, so the blend radius can never
#: exceed ~0.9 tip chords and, at the blend fraction the span-capped
#: optimiser actually settles on (~0.15), it is ~0.13 tip chords — a corner,
#: whatever turn law draws it. A real blended winglet starts turning INBOARD
#: of the tip, so its radius is set by the wing, not by the device.
WING_BLEND_FRAC_BOUNDS = (0.0, 0.15)
#: how the turn out of the wing plane is DISTRIBUTED along that arc length.
#:   "arc"    — constant curvature (a circular fillet): the tangent is
#:              continuous at both ends but the CURVATURE jumps 0 -> 1/R at
#:              the junction and 1/R -> 0 at the outer end. Legacy shape,
#:              reproduced bit-for-bit.
#:   "smooth" — smoothstep turn law: curvature RISES FROM ZERO at the
#:              junction and returns to zero at the outer end, so the wing
#:              and the winglet meet with continuous curvature (G2). This is
#:              the "smooth transition from horizontal to vertical" a blended
#:              winglet is drawn for; the circular fillet still has a visible
#:              crease in curvature at both ends.
#:   "spiral" — clothoid (Euler-spiral) pair: curvature ramps LINEARLY from
#:              zero over SPIRAL_RAMP_FRAC of the turn, holds constant
#:              through the middle, and ramps back to zero. Also G2, but its
#:              PEAK curvature is only 1/(1 - r) = 4/3 of the mean, against
#:              the smoothstep's 3/2 — i.e. it buys the same crease-free
#:              junction with a 11% gentler elbow, because it does not dump
#:              the whole turn into the middle. This is the standard
#:              road/rail transition spiral and the shape a real blended
#:              winglet is drawn with.
#: All three shapes turn through the SAME cant over the SAME arc length, so
#: they are the same size and differ only in shape (see winglet_path).
BLEND_SHAPES = ("arc", "smooth", "spiral")
#: fraction of the clothoid turn spent on EACH curvature ramp. r -> 0
#: recovers the circular fillet (peak curvature -> mean, creases back), r =
#: 1/3 matches the smoothstep's 3/2 peak, r = 1/2 is the ramp-only triangle
#: (peak 2x mean). 1/4 is the MODEL CHOICE: crease-free, and the gentlest
#: elbow of the G2 laws offered here. Not a measurement.
SPIRAL_RAMP_FRAC = 0.25
#: Gauss-Legendre nodes used to integrate the smooth turn law. The integrand
#: is analytic, so this is machine-accurate; it is FIXED (not derived from the
#: caller's sampling) so winglet_path(t) depends on t alone — the same
#: grid-independence rule Wing.chord obeys.
_BLEND_QUAD_N = 32
WINGLET_CANT_BOUNDS_DEG = (60.0, 90.0)  # from the wing plane; 90 = vertical
#: full physical cant span the nonplanar VLM stays valid over (0 = planar
#: raked extension, 90 = vertical fence). The winglet-TYPE presets in
#: objective.WINGLET_TYPES carve non-overlapping sub-bands out of this; the
#: default (60, 90) above is kept as the legacy bound so a plain "winglet"
#: run is bit-for-bit unchanged.
WINGLET_CANT_LIMITS_DEG = (5.0, 90.0)
#: ...and the SIGNED span of the same thing. The geometry itself has always
#: been signed (:func:`winglet_path`: + is up, − is down) and the water
#: families fly the signed band, because near a free surface WHICH WAY the
#: device points is the trade. The positive-only limit above is the AIRCRAFT
#: convention — ground clearance, not physics — so a surface whose own load
#: changes sign (a TRIMMING surface) validates against this one instead.
WINGLET_CANT_LIMITS_SIGNED_DEG = (-90.0, 90.0)
MISSION_V_BOUNDS = (10.0, 25.0)         # m/s, mission-mode speed variable
MISSION_ALT_BOUNDS_M = (0.0, 3000.0)    # m, mission-mode altitude variable
#                                         (well inside ISA troposphere validity)

# ---------------------------------------------------------------- chord law
#
# The trapezoid is a ONE-parameter planform: c_trap(eta) is linear in eta, so
# a single taper ratio cannot reach a general (e.g. elliptic) loading. The
# chord law adds a polynomial multiplier on top of that baseline,
#
#     c(eta) = c_trap(eta) * (1 + sum_j k_j eta^j) * F,   eta = |2y/b|
#
# anchored at 1 at the root exactly as the twist law is anchored at the root
# incidence: the root chord is already set by S and the taper, so the
# coefficients reshape the planform OUTBOARD rather than re-deriving a number
# the caller gave. ``F`` is the AREA-PRESERVING rescale (chord_area_factor):
# without it the multiplier would change S, and with it CL_target = W/(qS) —
# the design point the run is scored at. F is computed in CLOSED FORM here
# (section_wing.chord_law does the same job numerically, on its own station
# grid, for the section-design problem; the closed form is used here because
# Wing.chord(y) must be callable on ANY y array — an integral over whatever
# grid the caller passed would make the planform depend on the sampling).
#
#: polynomial chord orders offered. 0 = straight taper (no design variables).
CHORD_ORDERS = (0, 1, 2, 3)
CHORD_ORDER_NAMES = {0: "fixed (straight taper)", 1: "linear",
                     2: "quadratic", 3: "cubic"}
#: default half-width of the per-coefficient box (the coefficients can add
#: up, so this alone does NOT bound the deviation — see chord_dev). Measured
#: on a rectangular baseline with a cubic law: this box reaches span
#: efficiency 0.998 (vs 0.921 straight, 0.984 for the best straight taper)
#: while ~96% of the box is a flyable planform, so almost no evaluation is
#: spent on the collapse penalty. Widening to +/-2 buys e = 0.99914 and
#: costs a third of the box — a bad trade for a fixed budget.
CHORD_COEFF_BOUND = 0.5
#: largest coefficient box allowed. Past this the reachable e is flat and
#: the feasible fraction keeps falling, so it is a guard, not a knob.
CHORD_COEFF_LIMIT = 2.0
#: multiplier at or below which the planform stops being a wing. A chord law
#: that collapses a station to (near) zero chord cannot be flown, so Wing
#: refuses to construct — objective.evaluate maps that to the penalty
#: contract rather than reporting a plausible-looking L/D.
CHORD_MULT_FLOOR = 0.05

# ------------------------------------------------------- which LAW, not just
#
# The polynomial above is one chord law, not THE chord law, and which shape
# the coefficients describe is a design decision of its own. Every law offered
# here is AFFINE in its own parameters,
#
#     m(eta) = 1 + sum_j p_j phi_j(eta)
#
# and that is not a coincidence — it is the property everything downstream
# rests on. The deviation from straight taper then SCALES with the parameters
# (which is what lets chord_reach pull a collapsed law back along its own ray
# in closed form), the area integral is linear in them (so the rescale F has a
# closed form for every law), and a box on the parameters maps to a band on
# the planform the same way for all of them. A law that is not affine — a
# crank whose BREAK STATION is searched, say — would need its own version of
# all three, so it is not offered rather than offered and drawn wrongly.
#
# So a law IS its basis {phi_j} plus the area weights of that basis, and the
# four below differ only in those two things:
#
#   poly      phi_j = eta^j
#             The published law. The ends move: the multiplier at the tip is
#             1 + sum k_j, so a law that fills the tip in also lifts the root
#             chord once the area is rescaled back.
#   ends      phi_j = interior bumps with ZERO weighted area
#             Vanish at both ends and integrate to nothing against the
#             trapezoid, so the root chord, the tip chord AND the area are all
#             held exactly whatever the optimiser does between them. This is
#             the law to search when the root and tip chords are the user's
#             own numbers rather than the search's.
#   kinked    phi_j = hat functions at equally spaced interior stations
#             A cranked planform: straight panels meeting at a break. Also
#             vanishes at both ends, but its area is NOT zero, so the rescale
#             moves both end chords together (the taper is held; the size of
#             the trapezoid it is measured on is not).
#   elliptic  phi_1 = the ellipse, as a multiplier on the trapezoid
#             One number: how far from the straight taper towards a true
#             elliptic chord distribution. p = 1 would be the ellipse itself,
#             whose tip chord is zero, so the box stops short of it.
#
#: the chord laws offered. ``poly`` is the published one and every default.
CHORD_LAWS = ("poly", "ends", "kinked", "elliptic")
DEFAULT_CHORD_LAW = "poly"

CHORD_LAW_NAMES = {
    "poly": "polynomial",
    "ends": "interior only (root and tip chord held)",
    "kinked": "cranked (straight panels)",
    "elliptic": "elliptic blend",
}

CHORD_LAW_NOTES = {
    "poly": "The published law: the trapezoid's chord times "
            "1 + k₁η + k₂η² + k₃η³, rescaled to hold the area. The root and "
            "tip chords move with it — the law reshapes the whole planform, "
            "including its ends.",
    "ends": "The same freedom BETWEEN the ends, and none at them: every "
            "basis shape vanishes at the root and at the tip and integrates "
            "to zero area against the trapezoid, so the root chord, the tip "
            "chord and the area are exactly what the planform states no "
            "matter what the optimiser does with the middle.",
    "kinked": "A cranked planform — straight panels meeting at equally "
              "spaced breaks, which is what a wing built from constant-taper "
              "sections actually is. The break chords are the design "
              "variables; the ends are held in shape and rescaled with the "
              "area.",
    "elliptic": "One number: how far the chord distribution moves from the "
                "straight taper towards a true ellipse. The ellipse itself "
                "has zero tip chord, so the box stops short of it.",
}

#: how many design variables each law carries at a given ``chord_order``.
#: ``elliptic`` is ONE number whatever the order asks for — "how elliptic" is
#: not a question with a cubic's worth of answers — and the rest carry the
#: order itself.
CHORD_LAW_ORDERS = {
    "poly": CHORD_ORDERS,
    "ends": CHORD_ORDERS,
    "kinked": CHORD_ORDERS,
    "elliptic": (0, 1),
}

#: the ELLIPTIC blend's own ceiling. At p = 1 the law IS the ellipse and its
#: tip chord is zero — refused by CHORD_MULT_FLOOR, and singular in the
#: lifting line before that — so the box stops here. 0.9 leaves the tip at
#: 10% of its trapezoidal chord, which is a real (if extreme) planform.
CHORD_ELLIPTIC_MAX = 0.9


def chord_law_of(coeffs, default: str = DEFAULT_CHORD_LAW) -> str:
    """Which law a coefficient tuple is the parameters OF.

    The law rides on the tuple (:class:`ChordCoeffs`) rather than travelling
    as a second argument through eleven families and every ``Wing(...)`` they
    build. A plain tuple is the polynomial law, which is what every published
    problem carries and what keeps the legacy path bit-for-bit.
    """
    return str(getattr(coeffs, "law", default) or default)


class ChordCoeffs(tuple):
    """The chord law's parameters, and WHICH law they parameterise.

    A tuple of floats in every way that matters — it compares, unpacks and
    iterates exactly as the plain tuple it replaces — carrying one extra
    attribute. That is what lets :func:`chord_coeffs_from_x` answer the whole
    question ("these numbers, that shape") in one value, so a family that
    reads its design vector and builds a :class:`Wing` needs no new argument
    to pass along and cannot forget to pass it.
    """

    # (no __slots__: a tuple subclass cannot carry one, so the attribute
    # lives in the instance dict — one dict per Wing built, which is
    # nothing beside the solve it is built for)

    def __new__(cls, values=(), law: str = DEFAULT_CHORD_LAW):
        return super().__new__(cls, (float(v) for v in values))

    def __init__(self, values=(), law: str = DEFAULT_CHORD_LAW):
        super().__init__()
        self.law = check_chord_law(law)

    def __repr__(self) -> str:
        return f"ChordCoeffs({tuple(self)!r}, law={self.law!r})"

    def __reduce__(self):
        return (ChordCoeffs, (tuple(self), self.law))


def check_chord_law(law: str) -> str:
    if str(law) not in CHORD_LAWS:
        raise ValueError(f"unknown chord law {law!r}; "
                         f"choose from {list(CHORD_LAWS)}")
    return str(law)


def chord_param_count(order: int, law: str = DEFAULT_CHORD_LAW) -> int:
    """How many design variables ``law`` carries at ``chord_order``.

    The order ITSELF, always — and an order the law cannot carry raises here
    rather than being quietly clamped. That equality is load-bearing far from
    this function: ``chord_order`` is what every family passes as the number
    of TRAILING design-vector entries the chord block occupies (the flight
    state, the size block and the span all read the entries in front of it,
    ``sizing.resolve_size``), so a law whose parameter count disagreed with
    its order would mis-slice every one of those without an error anywhere.

    Callers that choose the law for a user (``api._chord_kwargs``) clamp the
    order to what the law offers before building the problem, which is where
    a "cubic elliptic blend" turns into the one number an elliptic blend has.
    """
    law = check_chord_law(law)
    n = int(order)
    if n <= 0:
        return 0
    if n not in CHORD_ORDERS:
        raise ValueError(
            f"chord_order must be one of {CHORD_ORDERS}, got {order}")
    if n not in CHORD_LAW_ORDERS[law]:
        raise ValueError(
            f"the {law!r} chord law carries "
            f"{max(CHORD_LAW_ORDERS[law])} parameter(s) at most, so "
            f"chord_order={order} has no meaning for it")
    return n


def _ends_basis_poly(order: int, lam: float) -> list:
    """Polynomials that vanish at BOTH ends and carry zero weighted area.

    Built from the bumps b_j = eta^j (1 - eta), each of which already vanishes
    at eta = 0 and eta = 1. Subtracting the right multiple of b_1 from each of
    b_2..b_{n+1} kills its integral against the trapezoid as well, which is
    what makes the area rescale exactly 1 and therefore leaves the two end
    chords at exactly the values the planform states.

    Each is then scaled to PEAK AT ONE, so that a coefficient means the same
    order of thing it means under the polynomial law — how far, as a fraction
    of the local chord, this shape may push the planform. Without it the raw
    bumps peak near 0.15 and the whole calibrated box (CHORD_COEFF_BOUND, and
    the CHORD_COEFF_LIMIT guard above it) would mean something different for
    this law than for the one it was measured on.
    """
    def bump(j):
        # eta^j - eta^(j+1)
        c = [0.0] * (j + 2)
        c[j], c[j + 1] = 1.0, -1.0
        return np.polynomial.Polynomial(c)

    trap = np.polynomial.Polynomial([1.0, -(1.0 - lam)])

    def area(p):
        q = (p * trap).integ()
        return float(q(1.0) - q(0.0))

    b1 = bump(1)
    a1 = area(b1)
    out = []
    for j in range(2, int(order) + 2):
        bj = bump(j)
        psi = bj - (area(bj) / a1) * b1
        lo, hi = _poly_extrema(psi)
        peak = max(abs(lo), abs(hi))
        out.append(psi / peak if peak > 0.0 else psi)
    return out


def _kink_stations(order: int) -> np.ndarray:
    """The break stations of the cranked law: equally spaced, ends excluded."""
    n = int(order)
    return np.arange(1, n + 1, dtype=float) / (n + 1.0)


def chord_basis(eta, order: int, law: str = DEFAULT_CHORD_LAW,
                taper: float = 1.0) -> np.ndarray:
    """``(n, len(eta))`` — the law's basis shapes at ``eta``.

    Row j is phi_j, so the multiplier of a law with parameters p is
    ``1 + p @ chord_basis(...)``. The polynomial law's rows are eta**j,
    computed exactly as they always were, so every published number is
    unchanged.
    """
    law = check_chord_law(law)
    e = np.asarray(eta, dtype=float)
    n = int(order)
    if n <= 0:
        return np.zeros((0, e.size), dtype=float)
    if law == "poly":
        powers = np.arange(1, n + 1, dtype=float)
        return e[None, :] ** powers[:, None]
    if law == "ends":
        return np.stack([p(e) for p in _ends_basis_poly(n, float(taper))])
    if law == "kinked":
        # the hats live on the CHORD, not on the multiplier: c_trap is itself
        # linear in eta, so a hat applied to the multiplier would draw a
        # piecewise QUADRATIC chord — curved panels, which is not what a
        # cranked wing is. Dividing by the trapezoid puts the straight panels
        # where they belong (c = c_trap + sum p_j hat_j, then rescaled).
        h = 1.0 / (n + 1.0)
        centres = _kink_stations(n)
        hats = np.maximum(0.0,
                          1.0 - np.abs(e[None, :] - centres[:, None]) / h)
        return hats / _trap(e, taper)[None, :]
    # elliptic: the ellipse as a MULTIPLIER on the trapezoid, minus the
    # trapezoid itself, so that p = 0 is straight taper and p = 1 is the
    # ellipse.
    ell = np.sqrt(np.maximum(0.0, 1.0 - e ** 2))
    return (ell / _trap(e, taper) - 1.0)[None, :]


def _trap(eta, taper: float) -> np.ndarray:
    """``c_trap / c_root`` at ``eta``, floored off zero.

    Two of the laws are defined as a shape ON the trapezoid and so divide by
    it. It is positive on [0, 1] for every taper the families publish, but a
    taper is a design-box row like any other and this session may have FIXED
    one outside that band (api.RunConfig.pinned) — at taper 0 the trapezoid's
    tip chord IS zero, and the division would hand the solver a NaN planform
    instead of the collapsed one it knows how to refuse.
    """
    e = np.asarray(eta, dtype=float)
    return np.maximum(1.0 - (1.0 - float(taper)) * e, CHORD_MULT_FLOOR * 1e-3)


def chord_area_weights(order: int, law: str = DEFAULT_CHORD_LAW,
                       taper: float = 1.0) -> np.ndarray:
    """``(n,)`` — ``int_0^1 phi_j(eta) c_trap(eta) deta`` in units of c_root.

    The area rescale is ``F = base / (base + p @ w)`` with
    ``base = (1 + lam) / 2`` for EVERY law, because every law is affine: this
    is the only law-dependent piece of it.
    """
    law = check_chord_law(law)
    n = int(order)
    lam = float(taper)
    if n <= 0:
        return np.zeros(0, dtype=float)
    if law == "poly":
        powers = np.arange(1, n + 1, dtype=float)
        return np.array([1.0 / (j + 1) - (1.0 - lam) / (j + 2)
                         for j in powers])
    if law == "ends":
        # zero BY CONSTRUCTION (_ends_basis_poly) — stated, not integrated,
        # so the exactness the whole law rests on cannot be lost to a
        # quadrature rule
        return np.zeros(n, dtype=float)
    if law == "kinked":
        # phi_j c_trap IS the hat (the division above cancels), so the weight
        # is the hat's own area: a triangle of base 2h and height 1
        h = 1.0 / (n + 1.0)
        return np.full(n, h, dtype=float)
    # elliptic: int (ell/trap - 1) * trap = int ell - int trap = pi/4 - (1+lam)/2
    return np.array([np.pi / 4.0 - 0.5 * (1.0 + lam)], dtype=float)


def chord_multiplier(eta, coeffs, law: str | None = None,
                     taper: float = 1.0) -> np.ndarray:
    """The chord law BEFORE area normalisation.

    ``1 + sum_j coeffs[j-1] eta^j`` for the polynomial law (unchanged), and
    ``1 + p @ phi`` for every other. ``law`` defaults to the law the
    coefficients themselves carry (:func:`chord_law_of`).
    """
    eta = np.asarray(eta, dtype=float)
    law = chord_law_of(coeffs) if law is None else check_chord_law(law)
    if law == "poly":
        out = np.ones_like(eta, dtype=float)
        for j, kj in enumerate(coeffs, start=1):
            out = out + float(kj) * eta ** j
        return out
    if not len(coeffs):
        return np.ones_like(eta, dtype=float)
    p = np.asarray([float(v) for v in coeffs], dtype=float)
    flat = np.atleast_1d(eta).ravel()
    out = 1.0 + p @ chord_basis(flat, len(p), law, taper)
    return out.reshape(np.shape(eta)) if np.shape(eta) else float(out[0])


def _poly_extrema(p) -> tuple[float, float]:
    """(min, max) of a polynomial over [0, 1] — endpoints plus its own
    stationary points, so a narrow interior dip cannot be missed."""
    cand = [0.0, 1.0]
    dp = p.deriv()
    if dp.degree() >= 1:
        for r in np.atleast_1d(dp.roots()):
            if abs(float(np.imag(r))) < 1e-12 and 0.0 <= float(np.real(r)) <= 1.0:
                cand.append(float(np.real(r)))
    vals = p(np.array(cand, dtype=float))
    return float(np.min(vals)), float(np.max(vals))


def chord_multiplier_extrema(coeffs, law: str | None = None,
                             taper: float = 1.0) -> tuple[float, float]:
    """(min, max) of the chord multiplier over eta in [0, 1] — EXACT.

    Exact for every law, by its own argument rather than by sampling: a grid
    can miss a narrow interior dip and let a collapsed planform through, and
    this is the test that refuses one.

    * polynomial laws (``poly``, ``ends``) — endpoints plus the real roots of
      the derivative;
    * ``kinked`` — piecewise linear, so its extrema are its knots;
    * ``elliptic`` — one closed form: the multiplier's only interior
      stationary point is at ``eta = 1 - lam`` (differentiate
      sqrt(1-eta^2)/(1-(1-lam)eta) and the algebra collapses to that), and it
      falls monotonically to ``1 - p`` at the tip.
    """
    if not len(coeffs):
        return 1.0, 1.0
    law = chord_law_of(coeffs) if law is None else check_chord_law(law)
    ks = [float(k) for k in coeffs]
    if law == "poly":
        return _poly_extrema(np.polynomial.Polynomial([1.0, *ks]))
    if law == "ends":
        p = np.polynomial.Polynomial([1.0])
        for k, b in zip(ks, _ends_basis_poly(len(ks), float(taper))):
            p = p + k * b
        return _poly_extrema(p)
    if law == "kinked":
        knots = np.concatenate([[0.0], _kink_stations(len(ks)), [1.0]])
        vals = 1.0 + np.asarray(ks) @ chord_basis(knots, len(ks), law,
                                                  float(taper))
        return float(np.min(vals)), float(np.max(vals))
    lam = float(taper)
    p = ks[0]
    k = 1.0 - lam
    if not 0.0 <= k < 1.0:
        # a taper at or below zero: the trapezoid's own tip chord is zero, so
        # sqrt(1-eta^2)/c_trap is unbounded and there is no closed form to
        # quote. Sampled on the SAME floored trapezoid the multiplier itself
        # uses, so the gate and the planform cannot disagree — a wing this
        # degenerate is refused by the collapse floor either way.
        m = chord_multiplier(np.linspace(0.0, 1.0, 4001),
                             ChordCoeffs(ks, "elliptic"), taper=lam)
        return float(np.min(m)), float(np.max(m))
    cand = [1.0, 1.0 - p]                       # eta = 0 and eta = 1
    if 0.0 < k < 1.0:
        cand.append(1.0 - p + p / np.sqrt(1.0 - k * k))
    return float(min(cand)), float(max(cand))


def chord_bounds(order: int, chord_max_frac: float = CHORD_COEFF_BOUND,
                 law: str = DEFAULT_CHORD_LAW) -> np.ndarray:
    """``(n, 2)`` box for the chord parameters.

    ``+/- chord_max_frac`` per parameter for every law whose parameters are
    coefficients of a shape (``poly``, ``ends``, ``kinked``). The ELLIPTIC
    law's one parameter is not a coefficient but a fraction — "how far
    towards the ellipse" — so its box is ``[0, min(frac, 0.9)]``: negative
    would bend the chord AWAY from elliptic into a shape nothing asked for,
    and 1 is the ellipse itself, whose zero tip chord no wing has.
    """
    law = check_chord_law(law)
    n = chord_param_count(order, law)
    f = float(chord_max_frac)
    if not (0.0 < f <= CHORD_COEFF_LIMIT):
        raise ValueError(
            f"chord_max_frac must be in (0, {CHORD_COEFF_LIMIT}], got {f}")
    if n == 0:
        return np.zeros((0, 2), dtype=float)
    if law == "elliptic":
        return np.array([[0.0, min(f, CHORD_ELLIPTIC_MAX)]], dtype=float)
    return np.tile([-f, f], (n, 1)).astype(float)


def chord_labels(order: int, law: str = DEFAULT_CHORD_LAW) -> tuple:
    """Design-vector labels for the appended chord parameters.

    One stem for every law: the row IS "the chord law's j-th parameter", and
    what the numbers mean is the law's business, said once where the law is
    chosen rather than encoded in a name that would then differ between a
    saved run and the run that reproduces it.
    """
    return tuple(f"chord_k{j}"
                 for j in range(1, chord_param_count(order, law) + 1))


def with_chord_bounds(box, order: int,
                      chord_max_frac: float = CHORD_COEFF_BOUND,
                      law: str = DEFAULT_CHORD_LAW) -> np.ndarray:
    """Append the chord-law coefficient rows to a problem's own box.

    The chord law is orthogonal to every other design variable — it only
    changes ``Wing.chord(y)``, which every solver in the package samples the
    same way — so any problem that builds a :class:`Wing` can carry it by
    appending these rows LAST and reading the same number of TRAILING
    entries back (:func:`chord_coeffs_from_x`). Keeping the rows at the end
    is what lets the existing indices (alpha, the winglet pair, depth/V,
    ride height...) keep their positions and their meaning.

    ``order = 0`` returns the box unchanged, so every problem's legacy
    design vector is bit-for-bit its published self.
    """
    base = np.asarray(box, dtype=float)
    rows = chord_bounds(order, chord_max_frac, law)
    return np.vstack([base, rows]) if rows.size else base


def chord_coeffs_from_x(x, order: int,
                        law: str = DEFAULT_CHORD_LAW) -> tuple:
    """The trailing entries of ``x`` as chord-law parameters.

    Returns a :class:`ChordCoeffs`, i.e. the numbers AND the law they
    parameterise, so the :class:`Wing` a family builds from them knows which
    shape it is being asked for without the family having to say so twice.
    A law with no parameters is the plain empty tuple, which is the straight
    taper and the bit-for-bit legacy path.
    """
    n = chord_param_count(order, law)
    if n <= 0:
        return ()
    xs = np.asarray(x, dtype=float)
    if xs.size < n:
        raise ValueError(
            f"design vector of length {xs.size} carries no room for "
            f"{n} chord coefficients")
    return ChordCoeffs(xs[-n:], law)


# ------------------------------------------------- what a chord BOX draws
#
# The rows above are the right variables for a solver and the wrong ones for
# a person: nobody knows what ``chord_k2 = -0.31`` draws, and widening a
# coefficient's bounds says nothing about the planform it buys. What a box on
# the coefficients IS, in the drawing, is a BAND: at every station, the
# narrowest and the widest chord any candidate inside the box can put there.
# That band is what :func:`chord_reach` computes, in units of the
# straight-taper chord the law starts from, so a design box can be READ.
#
# It is sampled rather than solved, and WHERE it is sampled is the whole
# accuracy question. The chord ratio c/c_trap = m(eta) F is linear-fractional
# in the coefficients (F, the area rescale, is a ratio of linear functions of
# them), so over a box its extremes sit at the box's CORNERS — which a
# linspace grid contains. But a corner is usually a planform the solver
# refuses (the chord collapses, CHORD_MULT_FLOOR), and the reachable set is
# then the box MINUS that region: the extremes move onto the collapse
# boundary, which no grid over the box lands on. The narrowest chords are
# exactly the ones near collapse, so a plain grid understated them by ~2% of
# the chord on the published box and by 30% on the widest one allowed.
#
# So every grid point that collapses is pulled back along its own ray to the
# boundary, which is a closed form: scaling the coefficients by t scales the
# multiplier's deviation by t, so the largest flyable t is
# (1 - floor) / -min_eta(m - 1). That samples the boundary at grid
# resolution, and the residual under-coverage over 200k random flyable laws
# is ~5e-4 of the chord on the published box.
#
# The pull is towards the ORIGIN, so it assumes the box contains straight
# taper — every box this tool builds is symmetric (chord_bounds), and the grid
# is odd, so K = 0 is in the sample. A hand-typed asymmetric box is drawn a
# little generously (measured on [0.2, 0.5] x [-0.5, 0.5]^2: the band sits
# ~2e-2 of the chord outside what that box reaches), because part of a pulled
# ray lies outside it. Left as it is deliberately: correcting it would move a
# picture, the boxes the tool itself writes are symmetric, and the alternative
# — clipping the pulled point back into the box — is no longer on the
# boundary the pull exists to sample.
#
# The collapse floor is not the only thing that refuses a law: a live
# :class:`ChordLimits` does too, and for the same reason the band has to know
# about it. A band drawn over candidates the run scores the penalty for is a
# picture of a design that does not exist. The TREND goes in as a mask
# (``_trend_ok``, no closed-form projection); the limits measured in METRES
# and DEGREES go in as both — a mask on the grid, so ``flyable_frac`` counts
# them, and a pull-back, because they too are a boundary no grid lands on and
# the projection onto them IS a scaling of the coefficients: c(eta) is
# linear-fractional in t along the ray, so "min chord >= c_min", "max chord <=
# c_max" and "|dc/dy| <= tan(rate)" are each LINEAR in t at every station.
# The admissible t of a ray is therefore an interval, its endpoints are closed
# form, and c is monotone in t between them, so the band's extremes are the
# two endpoints (:func:`_chord_limit_segment`).

#: spanwise stations the reach band is drawn on (root to tip inclusive).
CHORD_REACH_STATIONS = 65
#: values per coefficient axis. 17^3 ~ 4.9k planforms judged on 65 stations
#: each; the grid is a linspace, so the box's corners are IN the sample.
CHORD_REACH_GRID = 17
#: how far INSIDE the collapse boundary a pulled-back point is placed. The
#: boundary itself is refused (Wing tests the floor with <=), so a point
#: landing exactly on it would be dropped again.
_CHORD_REACH_PULL = 1.0 - 1e-9


@dataclass(frozen=True)
class ChordReach:
    """The band of chord distributions a box on the coefficients reaches.

    ``lo``/``hi`` are the extreme values of ``c(eta) / c_trap(eta)`` — the
    flown chord over the straight-taper chord of the SAME wing — over every
    flyable point of the box, station by station. 1.0 everywhere is straight
    taper, so ``dev_max`` is exactly :attr:`Wing.chord_dev` maximised over
    the box: "how far from a trapezoid this box lets the planform go".

    ``flyable_frac`` is the fraction of the sampled box that builds a wing at
    all; the rest collapses the chord somewhere and scores the penalty. A box
    with none is reported as ``lo = hi = nan``, because there is no planform
    to draw rather than a baseline one.

    ``trend`` is the chord TREND the band was drawn under (CHORD_TRENDS). It
    is not a second kind of bound: a law the trend forbids is refused where
    the planform is built (ChordLimits), exactly as a collapsed one is, so
    under a trend the band is narrower and ``flyable_frac`` smaller because
    fewer of the box's laws reach a wing — which is the honest picture of
    what that box now buys.

    ``limits`` is the whole statement the band was drawn under — the trend
    above PLUS the limits measured in metres and degrees — or None where the
    band is the box's own reach. It is the ChordLimits that was applied, not
    the one that was asked for: they are the same object, because a limit
    that cannot be measured (a metre without a length to measure it in) is
    refused at the call rather than dropped, so a caller can read this field
    as "these are the candidates in the picture".
    """

    eta: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    dev_max: float
    flyable_frac: float
    taper: float
    trend: str = "free"
    limits: "ChordLimits | None" = None

    @property
    def empty(self) -> bool:
        """Does NOTHING in this box build a wing?"""
        return not self.flyable_frac > 0.0

    @property
    def root(self) -> tuple[float, float]:
        """(narrowest, widest) root chord, in straight-taper root chords."""
        return float(self.lo[0]), float(self.hi[0])

    @property
    def tip(self) -> tuple[float, float]:
        """(narrowest, widest) tip chord, in straight-taper tip chords."""
        return float(self.lo[-1]), float(self.hi[-1])


def _trend_ok(coeffs, lam: float, trend: str,
              law: str = DEFAULT_CHORD_LAW) -> np.ndarray:
    """Which of the ``(N, order)`` chord laws draw a chord obeying ``trend``.

    Judged on the CHORD (the law times the straight-taper baseline), not on
    the multiplier, because the trend is a statement about the planform: a
    rising multiplier on a sharply tapered wing still draws a chord that
    falls. The area rescale F is a positive constant along the span, so it
    cannot change the sign of a station-to-station difference and is left
    out — which is what lets this be tested before F is computed.

    Sampled on CHORD_TREND_STATIONS, the grid :meth:`ChordLimits.violation`
    uses, NOT on the band's own drawing grid. The two must agree law by law
    or the picture would admit planforms the run refuses, and a quartic can
    turn over inside one drawing station: a law rising by 2e-5 of a chord
    over the first 0.7% of the span and falling thereafter reads as
    root-largest on 65 stations and is refused on 129.
    """
    K = np.atleast_2d(np.asarray(coeffs, dtype=float))
    if trend == "free":
        return np.ones(len(K), dtype=bool)
    eta = np.linspace(0.0, 1.0, CHORD_TREND_STATIONS)
    order = K.shape[1]
    mult = np.ones((len(K), eta.size)) if order == 0 else \
        1.0 + K @ chord_basis(eta, order, law, lam)
    c = mult * (1.0 - (1.0 - lam) * eta)[None, :]
    d = np.diff(c, axis=1)
    tol = CHORD_TREND_TOL * np.mean(c, axis=1)
    if trend == "root_largest":
        return np.max(d, axis=1) <= tol
    return np.min(d, axis=1) >= -tol


def _reach_limits(limits, trend: str, c_root, semi):
    """``(ChordLimits | None, trend)`` the band is to be drawn under.

    One statement, from two arguments that predate each other: ``trend`` is
    the older one and stays what every published call passes, ``limits`` is
    the whole ChordLimits the run will carry. They may not disagree — a
    picture drawn under one trend and a run flown under another is the exact
    failure this argument exists to end — and the effective trend is whichever
    of them is not "free".

    A limit in METRES or DEGREES needs the length that measures it, so
    ``c_root`` (the straight-taper root chord of the surface) and, for the
    rate, ``semi`` (its semi-span) are REQUIRED when one is live. Missing
    them raises rather than quietly dropping the limit: a band that ignored a
    live limit would be the drawing and the run disagreeing again, and a
    caller that cannot supply the lengths wants to know it is drawing the
    box's own reach.
    """
    if trend not in CHORD_TRENDS:
        raise ValueError(f"unknown chord trend {trend!r}; "
                         f"choose from {list(CHORD_TRENDS)}")
    if limits is None:
        return None, trend
    if limits.trend not in CHORD_TRENDS:
        raise ValueError(f"unknown chord trend {limits.trend!r}; "
                         f"choose from {list(CHORD_TRENDS)}")
    if "free" not in (limits.trend, trend) and limits.trend != trend:
        raise ValueError(
            f"chord trend asked twice and differently: trend={trend!r} "
            f"against limits.trend={limits.trend!r}")
    eff = limits.trend if limits.trend != "free" else trend
    if any(v is not None for v in (limits.c_min_m, limits.c_max_m,
                                   limits.rate_max_deg)):
        if c_root is None or not float(c_root) > 0.0:
            raise ValueError(
                "a chord limit in metres needs the root chord that measures "
                f"it: pass c_root > 0 (got {c_root!r})")
    if limits.rate_max_deg is not None and (semi is None
                                            or not float(semi) > 0.0):
        raise ValueError(
            "rate_max_deg is a limit per metre of SPAN: pass semi > 0 "
            f"(got {semi!r})")
    return replace(limits, trend=eff), eff


def _chord_metre(coeffs, lam: float, c_root: float, n_check: int,
                 law: str = DEFAULT_CHORD_LAW):
    """``(y-less chord [m], area factor)`` of ``(N, order)`` laws.

    The chord the wing is BUILT with — after the area rescale — at the
    stations :meth:`ChordLimits.violation` tests, which is the grid a limit
    in metres has to be judged on for the band and the run to agree.
    """
    K = np.atleast_2d(np.asarray(coeffs, dtype=float))
    eta = np.linspace(0.0, 1.0, int(n_check))
    order = K.shape[1]
    if order == 0:
        mult = np.ones((len(K), eta.size))
        F = np.ones(len(K))
    else:
        mult = 1.0 + K @ chord_basis(eta, order, law, lam)
        base = 0.5 * (1.0 + lam)
        w = chord_area_weights(order, law, lam)
        with np.errstate(divide="ignore", invalid="ignore"):
            F = base / (base + K @ w)
    return float(c_root) * mult * F[:, None] \
        * (1.0 - (1.0 - lam) * eta)[None, :], F


def _metre_ok(coeffs, lam: float, limits, c_root, semi,
              law: str = DEFAULT_CHORD_LAW) -> np.ndarray:
    """Which of the ``(N, order)`` laws obey the limits measured in METRES.

    The trend is NOT tested here (``_trend_ok`` owns it, and it is the one
    limit that needs no length): what is tested is exactly what
    :meth:`ChordLimits.violation` tests before it — the minimum chord, the
    maximum chord and the local taper angle — on the same stations, with the
    same finite-difference rate, on the chord after the area rescale.
    """
    K = np.atleast_2d(np.asarray(coeffs, dtype=float))
    ok = np.ones(len(K), dtype=bool)
    if limits is None:
        return ok
    c_min, c_max, rate = (limits.c_min_m, limits.c_max_m, limits.rate_max_deg)
    if c_min is None and c_max is None and rate is None:
        return ok
    c, _ = _chord_metre(K, lam, c_root, limits.n_check, law)
    with np.errstate(invalid="ignore"):
        if c_min is not None:
            ok &= np.nanmin(c, axis=1) >= float(c_min)
        if c_max is not None:
            ok &= np.nanmax(c, axis=1) <= float(c_max)
        if rate is not None:
            y = np.linspace(0.0, float(semi), int(limits.n_check))
            deg = np.rad2deg(np.arctan(np.abs(np.gradient(c, y, axis=1))))
            ok &= np.nanmax(deg, axis=1) <= float(rate)
    return ok & np.all(np.isfinite(c), axis=1)


def _fold(t_lo, t_hi, dead, alpha, beta):
    """Fold ``alpha * t <= beta`` (per law, per station) into ``[t_lo, t_hi]``.

    Every limit measured in metres is linear in the ray parameter t (see the
    block comment above), so each arrives here as one such family: a positive
    alpha is a ceiling on t, a negative one a floor, and a zero alpha with a
    negative beta is a law no t on this ray can save.
    """
    alpha, beta = np.broadcast_arrays(np.asarray(alpha, dtype=float),
                                      np.asarray(beta, dtype=float))
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = beta / alpha
    up, dn = alpha > 0.0, alpha < 0.0
    t_hi = np.minimum(t_hi, np.where(up, ratio, np.inf).min(axis=1))
    t_lo = np.maximum(t_lo, np.where(dn, ratio, -np.inf).max(axis=1))
    dead = dead | np.any(~up & ~dn & (beta < 0.0), axis=1) \
        | ~np.all(np.isfinite(alpha) & np.isfinite(beta), axis=1)
    return t_lo, t_hi, dead


def _chord_limit_segment(K, lam: float, limits, c_root, semi,
                         law: str = DEFAULT_CHORD_LAW):
    """``(t_lo, t_hi, ok)``: the admissible stretch of each law's own ray.

    ``t * K`` is the law ``K`` pulled towards straight taper (t = 0) or left
    where it is (t = 1). Along that ray the flown chord is

        c(eta, t) = B b(eta) (1 + t a(eta)) / (B + t d)

    — linear over linear in t — so each limit is a linear inequality in t at
    every station, the admissible set is an INTERVAL, and c is monotone in t
    inside it. That is what makes the boundary reachable in closed form and
    the band's extremes its two endpoints.

    ``ok`` is False for a ray with no admissible t at all. t is not clipped
    to [0, 1] here by the lower end: an interval that starts above 0 is a
    real answer (the straight taper itself may break a minimum chord the law
    can lift it over), and the caller samples both ends.
    """
    K = np.atleast_2d(np.asarray(K, dtype=float))
    n, order = K.shape
    t_lo = np.zeros(n)
    t_hi = np.ones(n)
    dead = np.zeros(n, dtype=bool)
    n_check = int(limits.n_check)
    eta = np.linspace(0.0, 1.0, n_check)
    B = 0.5 * (1.0 + lam)
    a = K @ chord_basis(eta, order, law, lam) if order \
        else np.zeros((n, n_check))
    w = chord_area_weights(order, law, lam)
    d = (K @ w if order else np.zeros(n))[:, None]
    # the area has to stay positive (B + t d > 0), and the multiplier has to
    # stay off the collapse floor — the two the free path already enforces
    t_lo, t_hi, dead = _fold(t_lo, t_hi, dead, -d, np.full((1, 1), B))
    t_lo, t_hi, dead = _fold(t_lo, t_hi, dead, -a,
                             np.full((1, 1), 1.0 - CHORD_MULT_FLOOR))
    b = float(c_root) * (1.0 - (1.0 - lam) * eta)[None, :]
    if limits.c_min_m is not None:
        c_min = float(limits.c_min_m)
        t_lo, t_hi, dead = _fold(t_lo, t_hi, dead,
                                 c_min * d - B * b * a, B * (b - c_min))
    if limits.c_max_m is not None:
        c_max = float(limits.c_max_m)
        t_lo, t_hi, dead = _fold(t_lo, t_hi, dead,
                                 B * b * a - c_max * d, B * (c_max - b))
    if limits.rate_max_deg is not None:
        y = np.linspace(0.0, float(semi), n_check)
        R = float(np.tan(np.deg2rad(float(limits.rate_max_deg))))
        # np.gradient is linear in the chord, and the chord's numerator is
        # linear in t, so the rate limit is two linear inequalities in t
        gP = np.gradient(np.broadcast_to(B * b, (1, n_check)), y, axis=1)
        gQ = np.gradient(B * b * a, y, axis=1)
        t_lo, t_hi, dead = _fold(t_lo, t_hi, dead, gQ - R * d, R * B - gP)
        t_lo, t_hi, dead = _fold(t_lo, t_hi, dead, -gQ - R * d, R * B + gP)
    t_lo = np.maximum(t_lo, 0.0)
    t_hi = np.minimum(t_hi, 1.0)
    ok = ~dead & (t_lo <= t_hi)
    # inside the boundary, never on it: every one of these limits is tested
    # with a strict comparison somewhere, so a point landing exactly on the
    # boundary would be dropped again (_CHORD_REACH_PULL, as for collapse)
    shrunk = np.where(t_hi < 1.0, _CHORD_REACH_PULL * t_hi, t_hi)
    grown = np.where(t_lo > 0.0, t_lo + (1.0 - _CHORD_REACH_PULL) * t_lo, t_lo)
    return grown, np.maximum(shrunk, grown), ok


def chord_reach(bounds, taper: float = 1.0,
                n_eta: int = CHORD_REACH_STATIONS,
                n_grid: int = CHORD_REACH_GRID,
                trend: str = "free",
                limits: "ChordLimits | None" = None,
                c_root: float | None = None,
                semi: float | None = None,
                law: str = DEFAULT_CHORD_LAW) -> ChordReach:
    """What ``bounds`` on the chord coefficients draws (see :class:`ChordReach`).

    ``bounds`` is the (order, 2) box of the ``chord_k*`` rows — the same rows
    :func:`chord_bounds` builds and a design box may have narrowed. ``taper``
    is the baseline the law multiplies: the area rescale integrates against
    the trapezoid, so the same coefficients bend a rectangular wing and a
    sharply tapered one by different amounts.

    An empty box (``order = 0``, i.e. no chord law) is straight taper: the
    band is 1.0 everywhere and nothing is deviated.

    ``flyable_frac`` is measured on the GRID — the fraction of the box that
    builds a wing — while the band is drawn on the grid's flyable points PLUS
    the collapse boundary the rest are pulled back to, because that boundary
    is where the narrowest flyable chords live.

    ``trend`` (CHORD_TRENDS) draws the band the same box reaches UNDER that
    trend: a law whose chord grows outboard is refused by ChordLimits before
    it is ever flown, so under ``root_largest`` it belongs in neither the
    band nor the flyable fraction. ``"free"`` is every published call, and
    returns the same numbers it always did.

    Accuracy under a trend is a little looser than without one, and for the
    same reason the collapse boundary is pulled back to above: the extreme
    admitted laws now sit on the TREND boundary, which is not a corner of the
    box and which no grid lands on. Measured over 4k random admitted laws per
    (taper, trend), the residual under-coverage is ~4e-3 of the chord —
    against ~5e-4 free, and still a line's width on the picture the band is
    for. It is not pulled back: unlike the collapse boundary, the projection
    onto "the nearest law that keeps the trend" is not a scaling of the
    coefficients, so there is no closed form to do it with.

    ``limits`` (a :class:`ChordLimits`) draws the band under the WHOLE
    statement the run will carry: the trend above and, with ``c_root`` (and
    ``semi`` for the rate), the minimum chord, the maximum chord and the
    steepest local taper angle, in the metres and degrees they are asked in.
    Those three ARE pulled back to — along the ray they are linear in t (see
    the block comment above), so each law's admissible stretch is an interval
    in closed form and the band's extremes are its endpoints — which is why
    the band reaches a live minimum chord instead of stopping a grid step
    short of it. Measured over 4k random admitted laws per (taper, limit set),
    the residual under-coverage is ~1e-3 of the chord — better than the ~4e-3
    a trend leaves, because a trend has no such projection. Passing none of
    them is every published call, bit-for-bit.

    A limit in metres without the length that measures it RAISES: silently
    drawing the box's own reach under a live limit is the disagreement
    between the picture and the run this argument exists to end.

    ``law`` (CHORD_LAWS) is which SHAPE those rows parameterise. Everything
    here works for any of them for one reason: every law offered is affine in
    its parameters, so the multiplier is ``1 + p @ basis`` and the area is
    ``base + p @ weights``, and the two matrices are the only law-dependent
    thing in the whole calculation. The polynomial law's are computed exactly
    as they always were, so every published band is unchanged.
    """
    box = np.asarray(bounds, dtype=float).reshape(-1, 2)
    lam = float(taper)
    law = check_chord_law(law)
    eta = np.linspace(0.0, 1.0, int(n_eta))
    order = box.shape[0]
    lim, trend = _reach_limits(limits, trend, c_root, semi)
    metred = lim is not None and any(
        v is not None for v in (lim.c_min_m, lim.c_max_m, lim.rate_max_deg))
    if order == 0:
        # straight taper: the ONE law in the box, judged by the same rule as
        # every other (a trapezoid that widens outboard is not root-largest,
        # and one whose tip falls below a minimum chord is refused too)
        ones = np.ones_like(eta)
        ok = bool(_trend_ok(np.zeros((1, 0)), lam, trend, law)[0]) and bool(
            _metre_ok(np.zeros((1, 0)), lam, lim, c_root, semi, law)[0])
        nan = np.full_like(eta, np.nan)
        return ChordReach(eta=eta, lo=ones if ok else nan,
                          hi=ones.copy() if ok else nan.copy(), dev_max=0.0,
                          flyable_frac=1.0 if ok else 0.0, taper=lam,
                          trend=trend, limits=lim)
    if order > max(CHORD_LAW_ORDERS[law]):
        raise ValueError(
            f"chord law {law!r} of order {order} exceeds the largest order "
            f"it offers ({max(CHORD_LAW_ORDERS[law])})")
    if np.any(box[:, 1] < box[:, 0]):
        raise ValueError("chord bounds must be given as [low, high]")

    # every combination of the per-coefficient grids: (N, order)
    axes = [np.linspace(lo, hi, int(n_grid)) for lo, hi in box]
    K = np.stack([g.ravel() for g in np.meshgrid(*axes, indexing="ij")],
                 axis=-1) if order > 1 else axes[0].reshape(-1, 1)
    stations = chord_basis(eta, order, law, lam)
    dev = K @ stations                  # multiplier - 1, at every station
    worst = dev.min(axis=1)
    collapsed = worst <= CHORD_MULT_FLOOR - 1.0
    # a law is FLYABLE when it builds a wing AND obeys the trend: both are
    # refused in the same place (Wing -> ChordLimits) and score the same
    # penalty, so both belong in this fraction. The two masks stay separate
    # below, because only the COLLAPSED ones are pulled back to a boundary.
    flyable = ~collapsed & _trend_ok(K, lam, trend, law) \
        & _metre_ok(K, lam, lim, c_root, semi, law)
    frac = float(np.mean(flyable))
    # nothing on the grid builds a wing = nothing to draw. The pull-back below
    # can reach laws no grid point is, but not from a grid with none: the box
    # is symmetric and n_grid odd, so straight taper (K = 0) is IN the sample,
    # and a box whose own baseline is refused has no band worth claiming.
    if not flyable.any():
        nan = np.full_like(eta, np.nan)
        return ChordReach(eta=eta, lo=nan, hi=nan.copy(), dev_max=0.0,
                          flyable_frac=0.0, taper=lam, trend=trend,
                          limits=lim)
    if metred:
        # every limit measured in metres is a boundary the grid does not land
        # on and CAN be projected onto, so each ray is walked to both ends of
        # its admissible interval — the band's extremes are there, and c is
        # monotone in t in between
        t_lo, t_hi, ray = _chord_limit_segment(K, lam, lim, c_root, semi, law)
        sample = np.concatenate([K * t_lo[:, None], K * t_hi[:, None]])
        keep = np.concatenate([ray, ray])
    else:
        # a collapsed point pulled back along its own ray to the boundary: the
        # deviation scales with the coefficients, so the largest flyable
        # multiple of it is a closed form. The band's extremes sit ON that
        # boundary, so this is what makes the grid's answer a usable one.
        pull = np.ones(len(K))
        pull[collapsed] = _CHORD_REACH_PULL * (1.0 - CHORD_MULT_FLOOR) \
            / (-worst[collapsed])
        sample = K * pull[:, None]
        keep = np.ones(len(sample), dtype=bool)
    mult = 1.0 + sample @ stations
    # the trend is re-tested on the PULLED law: shrinking the coefficients
    # towards straight taper can turn a law the trend refuses into one it
    # allows, and vice versa. So are the metre limits, and for the same
    # reason plus one more — the projection is exact in arithmetic and not in
    # floating point, so what the band claims is flyable is TESTED, never
    # assumed.
    live = keep & (mult.min(axis=1) > CHORD_MULT_FLOOR) \
        & _trend_ok(sample, lam, trend, law) \
        & _metre_ok(sample, lam, lim, c_root, semi, law)
    if not live.any():
        nan = np.full_like(eta, np.nan)
        return ChordReach(eta=eta, lo=nan, hi=nan.copy(), dev_max=0.0,
                          flyable_frac=0.0, taper=lam, trend=trend,
                          limits=lim)
    # the area rescale, exactly as Wing.chord_area_factor computes it
    base = 0.5 * (1.0 + lam)
    w = chord_area_weights(order, law, lam)
    F = base / (base + sample[live] @ w)
    ratio = mult[live] * F[:, None]
    lo = ratio.min(axis=0)
    hi = ratio.max(axis=0)
    return ChordReach(eta=eta, lo=lo, hi=hi,
                      dev_max=float(max(np.max(np.abs(lo - 1.0)),
                                        np.max(np.abs(hi - 1.0)))),
                      flyable_frac=frac, taper=lam, trend=trend, limits=lim)


def chord_bound_for_dev(dev: float, order: int = max(CHORD_ORDERS),
                        taper: float = 1.0, law: str = DEFAULT_CHORD_LAW,
                        **kw) -> float:
    """The symmetric coefficient bound whose box bends the chord by ``dev``.

    The inverse of ``chord_reach(chord_bounds(order, f), taper).dev_max``,
    which is what lets a design box ask its question in the planform ("the
    chord may deviate up to 30% from straight taper") and store the answer in
    the coefficients the solver searches.

    Monotone, so bisected: a wider coefficient box CONTAINS a narrower one,
    so it reaches every planform the narrower one does and more. The result
    is clamped to (0, CHORD_COEFF_LIMIT] — past that the reachable span
    efficiency is flat and the flyable fraction keeps falling, so a request
    for more comes back as the limit rather than as a box nobody should
    search — and rounded to four decimals, because a design box that reads
    ``|k| <= 0.1289998046875`` is the bisection's arithmetic showing through,
    not an answer anyone gave.
    """
    want = float(dev)
    if not want > 0.0:
        raise ValueError(f"deviation must be > 0, got {dev}")
    lo, hi = 1e-4, float(CHORD_COEFF_LIMIT)
    if chord_reach(chord_bounds(order, hi, law), taper,
                   law=law, **kw).dev_max <= want:
        return hi
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if chord_reach(chord_bounds(order, mid, law), taper,
                       law=law, **kw).dev_max < want:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-5:
            break
    return float(max(round(hi, 4), 1e-4))


# ---------------------------------------------------------------- flight state
#
# The second MODIFIER (the chord law above is the first). Speed and altitude
# are not geometry at all: they set the flow state (rho, mu from ISA and
# Sutherland) and, through it, the trim target CL = W/(q S) that every family
# in this package already flies at. So a family can carry them exactly as it
# carries the chord law — two extra rows, the same solver — provided it knows
# the design WEIGHT the trim target comes from (mission.MissionSpec).
#
# BLOCK ORDER, and why it is fixed here rather than per family: the modifiers
# stack as
#
#     [family block][flight block][chord block]
#
# and both modifier blocks are read back from the END of the vector — the
# chord coefficients are the last ``order`` entries, the flight pair the two
# before those. A family therefore never has to know where its own block ends
# to find a modifier, which is what lets ANY problem class gain either
# modifier without an index audit (the trap that a middle-inserted block sets:
# geometry.wing_from_x's x[3]/x[4]/x[5] would silently move).
#
#: design-vector labels of the flight block, in order.
FLIGHT_LABELS = ("V_ms", "altitude_m")


def flight_bounds(flight_free: bool, v_bounds=None, alt_bounds=None
                  ) -> np.ndarray:
    """(2, 2) box for the flight block, or (0, 2) when it is switched off.

    ``v_bounds`` defaults to the Tier A speed band (MISSION_V_BOUNDS, built
    around V = 14.6 m/s) — every family that flies THAT operating point
    inherits it. A family with its own cruise speed must pass its own band:
    a 50 m/s aircraft handed the 10-25 m/s box would have its published
    design point sitting outside the box, which is a bounds violation on
    every candidate rather than a design freedom (that is exactly how this
    was caught). The altitude band stays ISA-valid by construction.
    """
    if not flight_free:
        return np.zeros((0, 2), dtype=float)
    v = MISSION_V_BOUNDS if v_bounds is None else (float(v_bounds[0]),
                                                   float(v_bounds[1]))
    h = MISSION_ALT_BOUNDS_M if alt_bounds is None else (float(alt_bounds[0]),
                                                         float(alt_bounds[1]))
    for name, (lo, hi) in (("speed", v), ("altitude", h)):
        if not (hi > lo):
            raise ValueError(f"flight {name} bounds {(lo, hi)} are not "
                             f"increasing")
    return np.array([v, h], dtype=float)


def flight_labels(flight_free: bool) -> tuple:
    """Design-vector labels for the flight block (empty when off)."""
    return FLIGHT_LABELS if flight_free else ()


def with_flight_bounds(box, flight_free: bool, v_bounds=None,
                       alt_bounds=None) -> np.ndarray:
    """Append the flight rows to a family's own box.

    Call this BEFORE :func:`with_chord_bounds` — the chord coefficients are
    the trailing block by package rule, so the flight pair sits between the
    family block and them (see the block-order note above). ``False`` returns
    the box unchanged, so a family that does not carry the modifier is
    bit-for-bit its published self.
    """
    base = np.asarray(box, dtype=float)
    rows = flight_bounds(flight_free, v_bounds, alt_bounds)
    return np.vstack([base, rows]) if rows.size else base


def flight_from_x(x, flight_free: bool, chord_order: int = 0):
    """``(V_ms, altitude_m)`` read back out of a design vector, or None.

    The pair sits immediately BEFORE the ``chord_order`` trailing chord
    coefficients, so this is the exact inverse of the stacking rule above and
    needs no knowledge of the family block's length.
    """
    if not flight_free:
        return None
    xs = np.asarray(x, dtype=float)
    n = int(chord_order)
    if xs.size < n + 2:
        raise ValueError(
            f"design vector of length {xs.size} carries no room for a flight "
            f"block ahead of {n} chord coefficients")
    j = xs.size - n
    return float(xs[j - 2]), float(xs[j - 1])


# -------------------------------------------------------- winglet transition


def _check_blend_shape(shape: str) -> str:
    if shape not in BLEND_SHAPES:
        raise ValueError(
            f"unknown blend_shape {shape!r}; choose from {list(BLEND_SHAPES)}")
    return shape


def _turn_law(u: np.ndarray, shape: str) -> np.ndarray:
    """Normalised turn law L(u) on u in [0, 1]: L(0) = 0, L(1) = 1.

    The turn angle is ``phi * L(u)`` and the curvature ``(phi/s_tot) L'(u)``,
    so L alone decides the SHAPE of a transition — its length, its total
    turn and therefore its arc length and wetted area are fixed outside.
    """
    if shape == "arc":
        return u
    if shape == "smooth":
        return u * u * (3.0 - 2.0 * u)
    # clothoid: quadratic ramp in, linear middle, quadratic ramp out. K is
    # the plateau slope, fixed by L(1) = 1 = K (1 - r).
    r = SPIRAL_RAMP_FRAC
    K = 1.0 / (1.0 - r)
    return np.where(
        u < r, 0.5 * K * u * u / r,
        np.where(u <= 1.0 - r, K * (u - 0.5 * r),
                 1.0 - 0.5 * K * (1.0 - u) ** 2 / r))


def _turn_law_deriv(u: np.ndarray, shape: str) -> np.ndarray:
    """L'(u) — the normalised CURVATURE profile of a turn law.

    ``arc`` is flat at 1 (so it steps at both ends), ``smooth`` is the
    parabola 6u(1-u) peaking at 3/2, ``spiral`` is the trapezoid peaking at
    1/(1 - r) = 4/3. Peak/mean IS the tightness penalty a crease-free
    junction pays; see junction.blend_radius_min.
    """
    if shape == "arc":
        return np.ones_like(u)
    if shape == "smooth":
        return 6.0 * u * (1.0 - u)
    r = SPIRAL_RAMP_FRAC
    K = 1.0 / (1.0 - r)
    return np.where(u < r, K * u / r,
                    np.where(u <= 1.0 - r, np.full_like(u, K),
                             K * (1.0 - u) / r))


#: internal breakpoints (in u) of each turn law — the stations where its
#: curvature profile stops being one analytic piece. Gauss-Legendre is
#: spectrally accurate only WITHIN a smooth piece, and the path must be
#: unit-speed to machine precision or arc length (the invariant of the whole
#: family) leaks: integrating the clothoid in one span loses ~4e-7 of it.
_TURN_BREAKS = {"spiral": (SPIRAL_RAMP_FRAC, 1.0 - SPIRAL_RAMP_FRAC)}


def _blend_arcs(h: float, blend_frac: float, wing_arc: float
                ) -> tuple[float, float, float]:
    """(s_in, s_out, s_tot) — the transition's arc INBOARD of the junction on
    the wing, OUTBOARD on the winglet, and their sum."""
    s_in = float(wing_arc)
    if s_in < 0.0:
        raise ValueError(f"wing_arc must be >= 0, got {s_in}")
    s_out = float(blend_frac) * float(h)
    return s_in, s_out, s_in + s_out


def winglet_turn_angle(t, h: float, cant_deg: float, blend_frac: float = 0.0,
                       blend_shape: str = "arc",
                       wing_arc: float = 0.0) -> np.ndarray:
    """Angle out of the wing plane [rad] at arc length ``t`` — the TURN LAW.

    ``t`` is measured from the wing/winglet JUNCTION, + outboard along the
    winglet and − inboard along the wing; only a transition with a wing-side
    arc (``wing_arc`` > 0) turns at negative ``t``.

    The whole transition geometry is this one function integrated: the path
    is unit-speed by construction, so ``(y, z)`` is the integral of
    ``(cos psi, sin psi)`` and arc length is conserved exactly whatever the
    turn law does.

        psi(t) = phi * L(u),   u = (t + s_in) / (s_in + s_out)

    with ``s_out = blend_frac * h`` the winglet-side arc, ``s_in = wing_arc``
    the wing-side arc, and L the normalised law picked by ``blend_shape``
    (:func:`_turn_law`). Curvature is dpsi/dt = (phi/s_tot) L'(u): the
    circular fillet's is constant inside the turn and 0 outside, i.e. it
    JUMPS at both ends, while ``smooth`` and ``spiral`` VANISH at both ends —
    the wing and the winglet meet with continuous curvature. Every law
    reaches exactly ``phi`` at the outer end and holds it after, so every
    device has the same cant, the same arc length and the same wetted area.
    """
    t = np.asarray(t, dtype=float)
    phi = np.deg2rad(float(cant_deg))
    _, _, s_tot = _blend_arcs(h, blend_frac, wing_arc)
    if s_tot <= 0.0:
        return np.full_like(t, phi)       # sharp corner: no turning region
    u = np.clip((t + float(wing_arc)) / s_tot, 0.0, 1.0)
    return phi * _turn_law(u, _check_blend_shape(blend_shape))


def winglet_curvature(t, h: float, cant_deg: float, blend_frac: float = 0.0,
                      blend_shape: str = "arc",
                      wing_arc: float = 0.0) -> np.ndarray:
    """Curvature [1/m] of the transition line at arc length ``t`` (signed).

    d(psi)/dt of :func:`winglet_turn_angle`, on the same junction-centred
    ``t``. Reported so the SHAPE of a transition can be checked directly (the
    circular fillet is a step, the other two are bumps that start and end at
    zero) and so the junction model can price the tightest radius it draws.
    """
    t = np.asarray(t, dtype=float)
    phi = np.deg2rad(float(cant_deg))
    s_in, s_out, s_tot = _blend_arcs(h, blend_frac, wing_arc)
    if s_tot <= 0.0:
        return np.zeros_like(t)
    inside = (t >= -s_in) & (t <= s_out)
    u = np.clip((t + s_in) / s_tot, 0.0, 1.0)
    k = (phi / s_tot) * _turn_law_deriv(u, _check_blend_shape(blend_shape))
    return np.where(inside, k, 0.0)


def _turn_xy(s: np.ndarray, s_tot: float, phi: float, shape: str
             ) -> tuple[np.ndarray, np.ndarray]:
    """(y, z) accumulated from the START of the turn, for ``s`` in [0, s_tot].

    The circular fillet integrates in closed form; the others use fixed
    Gauss-Legendre quadrature. The node count is fixed (_BLEND_QUAD_N) rather
    than derived from ``s``, so the returned point depends on ITS OWN arc
    length only — sampling the path at 8 or at 800 stations gives the same
    geometry, which is the same grid-independence rule the closed-form chord
    area factor exists for.
    """
    if shape == "arc":
        R = s_tot / phi                                    # signed radius
        psi = np.clip(s / R, min(0.0, phi), max(0.0, phi))
        return R * np.sin(psi), R * (1.0 - np.cos(psi))
    xi, w = np.polynomial.legendre.leggauss(_BLEND_QUAD_N)
    breaks = _TURN_BREAKS.get(shape, ())
    if not breaks:
        tau = s[:, None] * 0.5 * (xi + 1.0)[None, :]    # (M, n) nodes in [0,s]
        psi = phi * _turn_law(np.clip(tau / s_tot, 0.0, 1.0), shape)
        scale = 0.5 * s
        return scale * (np.cos(psi) @ w), scale * (np.sin(psi) @ w)
    # piecewise-analytic law: one Gauss rule PER smooth piece, each clipped
    # to [0, s], so the panel stations inside a ramp are as exact as the ones
    # inside the plateau (and the sum still depends on s alone)
    edge = np.array([0.0, *breaks, 1.0], dtype=float) * s_tot
    lo = np.clip(edge[:-1][None, :], 0.0, s[:, None])       # (M, K)
    hi = np.clip(edge[1:][None, :], 0.0, s[:, None])
    half = 0.5 * (hi - lo)
    tau = (0.5 * (lo + hi))[:, :, None] + half[:, :, None] * xi   # (M, K, n)
    psi = phi * _turn_law(np.clip(tau / s_tot, 0.0, 1.0), shape)
    return (np.einsum("mkn,mk,n->m", np.cos(psi), half, w),
            np.einsum("mkn,mk,n->m", np.sin(psi), half, w))


def dihedral_rotate(y, z, dihedral_deg: float):
    """Rotate a MIRROR-SYMMETRIC (y, z) quarter-chord curve about the x axis.

    Dihedral is a rigid rotation of each semi-span, tips up for positive
    ``dihedral_deg``. It is applied to the assembled curve — wing AND any tip
    device — rather than folded into :func:`span_path`, because that is what
    "cant measured from the wing plane" means: the device turns with the
    surface it is bolted to, and its cant is unchanged by dihedral.

    Rigid is the whole point. A rotation is an isometry, so the DEVELOPED arc
    length of the curve is untouched — and arc length is the invariant this
    module's whole nonplanar family is built around (:func:`span_path`). Span,
    area, the chord law and the twist law are therefore all unchanged; what
    changes is where the surface points, and so what the Trefftz plane and the
    panel normals see. Projected span goes as ``cos Gamma``; developed span
    does not move.

    Mirror symmetry is why this is not simply ``R_x @ [y, z]``: a plain
    rotation takes the PORT tip (y < 0) DOWN, which is a rolled aeroplane, not
    a dihedralled one. Each side rotates away from the centreline, so the
    magnitude |y| carries the geometry and the sign picks the side — exactly
    the convention :func:`_mirror_span` uses in ``vlm``.

    ``dihedral_deg = 0`` short-circuits. That is for COST and for intent, not
    for correctness: ``cos(0)`` and ``sin(0)`` are exact, so the general path
    already returns the inputs bit-for-bit on any finite array, and a mutant
    that removes the branch is semantically equivalent (measured —
    ``scripts/v5_mutate.py``). It stays because a planar wing is the common
    case and because the branch says out loud that planar is meant to be
    untouched.
    """
    gam = float(dihedral_deg)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    if gam == 0.0:
        return y, z
    g = np.deg2rad(gam)
    cg, sg = np.cos(g), np.sin(g)
    ay = np.abs(y)
    return np.sign(y) * (ay * cg - z * sg), ay * sg + z * cg


def span_path(s, semispan: float, h: float, cant_deg: float,
              blend_frac: float = 0.0, blend_shape: str = "arc",
              wing_blend_frac: float = 0.0
              ) -> tuple[np.ndarray, np.ndarray]:
    """(y, z) of the quarter-chord line at DEVELOPED arc length ``s`` from the
    root — wing and winglet as ONE curve.

    ``s`` runs from 0 at the centreline to ``semispan + h`` at the winglet
    tip; the wing occupies ``s <= semispan``, the winglet the rest. Arc
    length is the invariant of the whole family: ``semispan`` is the
    DEVELOPED semi-span, so a blend never changes how much wing there is, how
    much chord it carries or how much area it has — it changes only where
    that wing points. What it trades is PROJECTED span for height, which is
    exactly what the Trefftz plane is entitled to see.

    The transition straddles the junction:

        s_in  = wing_blend_frac * semispan     arc spent turning ON THE WING
        s_out = blend_frac * h                 arc spent turning on the winglet

    ``wing_blend_frac = 0`` puts the whole turn on the winglet and reproduces
    the published geometry bit-for-bit. It is the knob that makes a blend
    look like a blend: confined to the winglet the turning arc is at most
    ``h``, so at the blend fractions a span-capped optimiser settles on the
    radius is a fraction of a tip chord and every turn law draws the same
    corner. Letting the turn start inboard of the tip decouples the blend
    radius from the winglet's height — which is how a real blended winglet
    is built.
    """
    s = np.asarray(s, dtype=float)
    phi = np.deg2rad(float(cant_deg))
    beta = float(blend_frac)
    w = float(wing_blend_frac)
    if not (0.0 <= beta <= 1.0):
        raise ValueError(f"blend_frac must be in [0, 1], got {beta}")
    if not (0.0 <= w <= 1.0):
        raise ValueError(f"wing_blend_frac must be in [0, 1], got {w}")
    _check_blend_shape(blend_shape)
    s_in, s_out, s_tot = _blend_arcs(h, beta, w * float(semispan))
    s0 = float(semispan) - s_in                  # arc where the turn starts
    if s_tot <= 0.0 or phi == 0.0:
        # sharp corner (or a planar raked tip): flat wing, then a straight ray
        out = np.maximum(s - float(semispan), 0.0)
        return (np.minimum(s, float(semispan)) + out * np.cos(phi),
                out * np.sin(phi))
    ty, tz = _turn_xy(np.clip(s - s0, 0.0, s_tot), s_tot, phi, blend_shape)
    straight = np.maximum(s - (s0 + s_tot), 0.0)   # 0 until the turn is done
    return (np.minimum(s, s0) + ty + straight * np.cos(phi),
            tz + straight * np.sin(phi))


def winglet_path(t, h: float, cant_deg: float, blend_frac: float = 0.0,
                 blend_shape: str = "arc",
                 wing_arc: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """(y, z) of the winglet quarter-chord line at arc length ``t`` from the
    tip, RELATIVE TO THE JUNCTION — the wing/winglet TRANSITION geometry.

    ``blend_frac`` in [0, 1] is the fraction of the winglet's arc length
    spent turning: 0 is the sharp corner (a straight canted ray leaving the
    wing plane at a kink — the legacy geometry, reproduced bit-for-bit),
    1 spends the whole device turning with no straight run at all. In
    between, the line leaves the wing plane TANGENTIALLY, turns through the
    cant angle over ``blend_frac * h`` of arc, then runs straight.

    ``wing_arc`` [m] extends that same turn INBOARD of the tip, onto the
    wing: the line then leaves the wing plane before the tip and only
    ``s_out = blend_frac * h`` of the turn is left for the winglet. 0 (the
    default) is the published geometry. Use :func:`span_path` to draw the
    wing and the winglet as the one curve they are — this function returns
    the winglet leg alone, measured from the junction, and so it cannot show
    the wing-side half of the turn.

    ``blend_shape`` picks HOW that turn is distributed (BLEND_SHAPES):
    ``"arc"`` is the constant-radius fillet (legacy, bit-for-bit),
    ``"smooth"`` the curvature-continuous smoothstep and ``"spiral"`` the
    clothoid pair — the two G2 laws differ in how tight their middle gets
    (see :func:`winglet_turn_angle`).

    Arc length is the invariant: every (blend_frac, blend_shape) winglet of
    the same ``h`` has the same quarter-chord length (and, at equal chord,
    the same wetted area), so comparing them compares SHAPE, not size. What
    the blend actually trades is TIP HEIGHT for PROJECTED SPAN plus a smooth
    junction — the height loss and the span gain are both visible to the
    Trefftz plane (vlm.py), while the junction's own drag is priced
    separately (junction.py) because a lifting-surface method cannot see it.

    Signed cant: + is up, − is down; the turn goes the same way.
    """
    t = np.asarray(t, dtype=float)
    a = float(wing_arc)
    if a < 0.0:
        raise ValueError(f"wing_arc must be >= 0, got {a}")
    # the wing-side arc is expressed as a semispan FRACTION by span_path, so
    # a semi-span of exactly `a` with fraction 1 puts the turn's start at the
    # origin; a = 0 collapses to the published expressions unchanged.
    kw = dict(blend_frac=blend_frac, blend_shape=blend_shape,
              wing_blend_frac=1.0 if a > 0.0 else 0.0)
    y, z = span_path(a + t, a, h, cant_deg, **kw)
    if a > 0.0:
        y0, z0 = span_path(np.array([a]), a, h, cant_deg, **kw)
        y, z = y - y0[0], z - z0[0]
    return y, z


def winglet_projection(h: float, cant_deg: float, blend_frac: float = 0.0,
                       blend_shape: str = "arc",
                       wing_arc: float = 0.0) -> float:
    """Horizontal (spanwise) reach of the winglet tip beyond the JUNCTION.

    ``h cos(cant)`` for the sharp ray, but a blended device leaves the wing
    plane tangentially and therefore reaches FURTHER outboard for the same
    arc length. Span-capped accounting must use this number, not the cosine,
    or a blend would smuggle in free projected span. With a wing-side arc the
    wing itself also loses projection — :func:`span_projection` is the
    quantity a cap should use then.
    """
    return float(winglet_path(np.array([float(h)]), h, cant_deg,
                              blend_frac, blend_shape, wing_arc)[0][0])


def winglet_tip_height(h: float, cant_deg: float, blend_frac: float = 0.0,
                       blend_shape: str = "arc",
                       wing_arc: float = 0.0) -> float:
    """Vertical reach of the winglet tip above (+) / below (−) the junction."""
    return float(winglet_path(np.array([float(h)]), h, cant_deg,
                              blend_frac, blend_shape, wing_arc)[1][0])


def winglet_arc_to_height(z_target: float, h: float, cant_deg: float,
                          blend_frac: float = 0.0,
                          blend_shape: str = "arc",
                          wing_arc: float = 0.0) -> float:
    """Developed arc [m] along the device at which it has reached ``z_target``.

    The inverse of :func:`winglet_tip_height` in its first argument, and the
    number a STRUCTURAL model of the device needs: a plate that has to get
    from the wing down to the car's deck is a cantilever whose length is the
    arc it actually spans, not the vertical gap it closes. Those are the same
    number only for a straight device at cant 90 — the case every published
    run flies, which is why this returns ``z_target`` there exactly and moves
    no published answer.

    A canted device covers ``z_target`` over ``z_target / sin(cant)`` of arc,
    and a BLENDED one over more again, because it leaves the wing plane
    tangentially and spends the start of its run going sideways. Both make
    the member LONGER than the gap, so using the gap understates its bending
    — the optimistic direction, which is why this exists.

    z(t) is monotone in t over ``cant in (0, 90]``, so a bisection is exact
    rather than sampled: the answer does not depend on any grid, which is the
    same rule :meth:`Wing.chord` obeys. When the device never reaches
    ``z_target`` at all its whole arc ``h`` is returned — the caller's reach
    constraint is the thing that refuses such a device, and a structural
    length is not the place to raise about it.
    """
    h = float(h)
    z_target = float(z_target)
    if h <= 0.0 or z_target <= 0.0:
        return 0.0
    def _z(t):
        # z at arc ``t`` ALONG A DEVICE OF HEIGHT h — never the tip height of
        # a shorter device, which would redistribute the blend (the turning
        # arc is blend_frac * h) and answer a different question
        return float(winglet_path(np.array([float(t)]), h, cant_deg,
                                  blend_frac, blend_shape, wing_arc)[1][0])

    if _z(h) <= z_target:
        return h
    lo, hi = 0.0, h
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _z(mid) < z_target:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


def span_projection(semispan: float, h: float, cant_deg: float,
                    blend_frac: float = 0.0, blend_shape: str = "arc",
                    wing_blend_frac: float = 0.0) -> float:
    """PROJECTED semi-span of the whole developed line — wing tip plus device.

    This is what a span cap must hold: with the turn confined to the winglet
    it is ``semispan + winglet_projection(...)``, but a wing-side blend bends
    the OUTER WING up too, so the wing's own projection is short of its
    developed semi-span and the sum is the only honest number.
    """
    return float(span_path(np.array([float(semispan) + float(h)]), semispan,
                           h, cant_deg, blend_frac, blend_shape,
                           wing_blend_frac)[0][0])


def developed_semispan(target_semispan: float, h: float, cant_deg: float,
                       blend_frac: float = 0.0, blend_shape: str = "arc",
                       wing_arc: float = 0.0) -> float:
    """DEVELOPED semi-span whose projection is exactly ``target_semispan``.

    Closed form, because everything inboard of the turn projects 1:1: the
    projected semi-span is ``(semispan - s_in) + P``, where P — the reach of
    the transition plus the straight winglet run, measured from where the
    turn starts — depends only on the transition, not on how much flat wing
    precedes it. So

        semispan = target - P + s_in.

    ``wing_arc`` is passed in METRES (not as a semi-span fraction) precisely
    because the semi-span is what is being solved for: fixing the arc against
    the problem's nominal span keeps this an equation rather than a fixed
    point, and matches how the winglet's own arc ``h`` is already fixed
    against the nominal span in :func:`wing_from_x`.
    """
    a = float(wing_arc)
    # P: project the transition + straight run, starting the line AT the turn
    reach = span_projection(a, h, cant_deg, blend_frac, blend_shape,
                            1.0 if a > 0.0 else 0.0)
    return float(target_semispan) - reach + a


#: how a chord distribution may TREND outboard, when the user has an opinion.
#: "free" is every published run: the law may bulge wherever the loading
#: wants it. The other two are buildability/handling statements — a planform
#: whose root is the biggest chord is the conventional one, and its inverse
#: is asked for often enough (inverse taper, and any surface whose tip has to
#: carry chord) that refusing to say it would be the missing control.
CHORD_TRENDS = ("free", "root_largest", "root_smallest")

#: relative slack on the trend test: the chord is sampled on a finite grid
#: and the area rescale is a division, so an exactly-rectangular law comes
#: back with ~1e-16 wobble. A trend is violated when it is violated by more
#: than this fraction of the mean chord.
CHORD_TREND_TOL = 1e-9

#: stations the chord distribution is tested on. ONE number, because two
#: things test it — ChordLimits, which refuses a candidate, and chord_reach,
#: which draws the band of candidates a design box may propose — and a
#: quartic can turn over between stations, so a coarser grid on either side
#: would let the picture admit a planform the run refuses.
CHORD_TREND_STATIONS = 129


@dataclass(frozen=True)
class ChordLimits:
    """What the chord distribution has to obey, in METRES and DEGREES.

    The design box speaks in chord-law COEFFICIENTS (``chord_k1..k3``, the
    polynomial multiplying the taper baseline at constant area). They are the
    right variables for a solver and the wrong ones for a person: nobody
    knows what ``k2 = -0.31`` draws, and the quantities an engineer actually
    has to hold — a minimum chord for the structure or the section's
    Reynolds number, a maximum for the mould, a limit on how fast the chord
    may change, and which end of the wing is the wide one — are all
    NONLINEAR functions of those coefficients once the area rescale is
    applied. So they are stated here, in their own units, and checked on the
    chord distribution the candidate actually draws.

    Every field is optional and ``None`` means UNCONSTRAINED: an empty
    ChordLimits is the published behaviour exactly, and switching one limit
    on costs the search nothing anywhere else.

    ``rate_max_deg`` is the local taper ANGLE: ``atan(|dc/dy|)``, the angle
    between the leading and trailing edges' mean line and the span. A
    straight-taper wing has one value of it; a chord law has a distribution,
    and its peak is what a mould-line or a rib-pitch limit really constrains.
    """

    c_min_m: float | None = None
    c_max_m: float | None = None
    rate_max_deg: float | None = None
    trend: str = "free"
    n_check: int = CHORD_TREND_STATIONS   # stations the limits are tested on
    #                            (shared with chord_reach's trend test — see
    #                            CHORD_TREND_STATIONS)

    def __post_init__(self):
        if self.trend not in CHORD_TRENDS:
            raise ValueError(
                f"unknown chord trend {self.trend!r}; "
                f"choose from {list(CHORD_TRENDS)}")
        for name in ("c_min_m", "c_max_m"):
            v = getattr(self, name)
            if v is not None and not float(v) > 0.0:
                raise ValueError(f"{name} must be > 0, got {v}")
        if (self.c_min_m is not None and self.c_max_m is not None
                and float(self.c_min_m) > float(self.c_max_m)):
            raise ValueError(
                f"c_min_m {self.c_min_m} exceeds c_max_m {self.c_max_m}")
        if self.rate_max_deg is not None and not (
                0.0 < float(self.rate_max_deg) < 90.0):
            raise ValueError(
                f"rate_max_deg must be in (0, 90), got {self.rate_max_deg}")
        if int(self.n_check) < 5:
            raise ValueError("n_check must be >= 5")

    @property
    def active(self) -> bool:
        return (self.c_min_m is not None or self.c_max_m is not None
                or self.rate_max_deg is not None or self.trend != "free")

    def violation(self, y, chord) -> str | None:
        """Why ``chord(y)`` breaks these limits, or None if it does not.

        ``y`` runs root -> tip on ONE side; the wing is symmetric, so that is
        the whole distribution.
        """
        if not self.active:
            return None
        y = np.asarray(y, dtype=float)
        c = np.asarray(chord, dtype=float)
        if self.c_min_m is not None and float(np.min(c)) < float(self.c_min_m):
            return (f"chord falls to {float(np.min(c)):.4g} m, below the "
                    f"{float(self.c_min_m):.4g} m minimum")
        if self.c_max_m is not None and float(np.max(c)) > float(self.c_max_m):
            return (f"chord reaches {float(np.max(c)):.4g} m, above the "
                    f"{float(self.c_max_m):.4g} m maximum")
        if self.rate_max_deg is not None:
            rate = np.rad2deg(np.arctan(np.abs(np.gradient(c, y))))
            worst = float(np.max(rate))
            if worst > float(self.rate_max_deg):
                return (f"chord changes at {worst:.4g} deg, above the "
                        f"{float(self.rate_max_deg):.4g} deg limit")
        if self.trend != "free":
            tol = CHORD_TREND_TOL * float(np.mean(c))
            d = np.diff(c)
            if self.trend == "root_largest" and float(np.max(d)) > tol:
                return ("chord grows outboard, and this design asks for the "
                        "root to be the largest chord")
            if self.trend == "root_smallest" and float(np.min(d)) < -tol:
                return ("chord falls outboard, and this design asks for the "
                        "root to be the smallest chord")
        return None


@dataclass
class Wing:
    """Trapezoidal wing, fixed b and S, linear twist root->tip in |2y/b|.

    ``chord_coeffs`` (default empty = straight taper, bit-for-bit) turns the
    trapezoid into a general planform via the area-preserving polynomial
    chord law documented above. The area stays exactly S, so CL_target,
    AR and every reference quantity are untouched; what changes is the
    SHAPE of the chord distribution, and with it the spanwise loading the
    lifting line can reach. A law that collapses the chord anywhere in
    eta in [0, 1] is refused at construction (ValueError).
    """

    b: float = 10.0
    S: float = 10.0
    taper: float = 1.0
    twist_root_deg: float = 0.0
    twist_tip_deg: float = 0.0
    sweep_deg: float = 0.0      # quarter-chord sweep [deg]; aft positive.
    #                           Reduced-order (section slope, form factor,
    #                           Raymer weight) AND lattice geometry — see
    #                           vlm.VLM, which offsets the quarter-chord line
    #                           by |y| tan(sweep) and so moves x_np with it.
    dihedral_deg: float = 0.0   # tips-up cant of the whole semi-span [deg].
    #                           A RIGID rotation (geometry.dihedral_rotate),
    #                           so span, area, chord and twist are untouched
    #                           and only the direction the surface points
    #                           changes. It is the only source of Cl_beta a
    #                           lifting surface carries AT ANY LIFT: the cant
    #                           puts a y-component in the panel normal, so it
    #                           acts through the BOUNDARY CONDITION and its
    #                           value does not move with CL.
    #                           ``sweep_deg`` gives one too and it is NOT the
    #                           same lever. Measured on this lattice it is
    #                           Cl_beta = -0.22 CL tan(Lambda) — exactly
    #                           proportional to lift — and it arrives ONLY
    #                           through the local-velocity Kutta-Joukowski
    #                           term (``dynamics.deck(local_velocity=False)``
    #                           reports it as exactly 0), because a flat
    #                           swept wing's normals have no y-component for
    #                           a sideslip to push on. Sweep raises Cl_r
    #                           about as fast, so it does not buy the spiral:
    #                           tests/test_sweep_is_a_cl_beta_that_goes_as_lift.py
    #                           measures both. Without a cant the fin carries
    #                           the dihedral effect and the yaw stiffness at
    #                           once, which is why no fin size makes the
    #                           reference design spirally stable.
    tc: float = 0.12            # thickness/chord; selects the section polar
                                # in tier_a_plus mode (and feeds the FF)
    chord_coeffs: tuple = ()    # polynomial chord law; () = straight taper
    chord_limits: "ChordLimits | None" = None   # what the DISTRIBUTION has
    #                           to obey, in metres and degrees (ChordLimits).
    #                           None = unconstrained, which is every
    #                           published run. Checked here, at construction,
    #                           so that every family gets it from the one
    #                           place they all build their planform — and a
    #                           violation arrives as the ValueError each of
    #                           them already turns into the penalty contract.

    def __post_init__(self):
        # Empty coefficients short-circuit: no validation cost and no
        # behaviour change on the legacy path.
        if not len(self.chord_coeffs):
            self.chord_coeffs = ()
            self._check_chord_limits()
            return
        law = chord_law_of(self.chord_coeffs)
        self.chord_coeffs = ChordCoeffs(self.chord_coeffs, law)
        if len(self.chord_coeffs) > max(CHORD_LAW_ORDERS[law]):
            raise ValueError(
                f"chord law {law!r} of order {len(self.chord_coeffs)} exceeds "
                f"the largest order it offers "
                f"({max(CHORD_LAW_ORDERS[law])})")
        m_min, _ = chord_multiplier_extrema(self.chord_coeffs,
                                            taper=self.taper)
        if m_min <= CHORD_MULT_FLOOR:
            raise ValueError(
                f"chord law collapses the chord (min multiplier {m_min:.4f} "
                f"<= {CHORD_MULT_FLOOR})")
        self._check_chord_limits()

    @property
    def chord_law(self) -> str:
        """Which SHAPE this wing's chord parameters describe (CHORD_LAWS).

        Carried by the coefficients themselves (:class:`ChordCoeffs`), so a
        family that reads its design vector and builds a wing passes the law
        along without knowing it exists — and a plain tuple, which is what
        every published problem hands over, is the polynomial law.
        """
        return chord_law_of(self.chord_coeffs)

    def _check_chord_limits(self):
        """Refuse a planform that breaks the chord limits, if any are set.

        Sampled on ONE side (the wing is symmetric) at the limits' own grid,
        including both ends, so the root and tip chords are tested exactly
        rather than at the nearest interior station.
        """
        lim = self.chord_limits
        if lim is None or not lim.active:
            return
        y = np.linspace(0.0, self.b / 2.0, int(lim.n_check))
        why = lim.violation(y, self.chord(y))
        if why is not None:
            raise ValueError(f"chord limits: {why}")

    @property
    def AR(self) -> float:
        return self.b**2 / self.S

    @property
    def c_root(self) -> float:
        return 2.0 * self.S / (self.b * (1.0 + self.taper))

    @property
    def chord_area_factor(self) -> float:
        """Closed-form rescale that holds the planform area at exactly S.

        S = b * int_0^1 c(eta) deta, and int_0^1 c_trap eta^j deta =
        c_root [1/(j+1) - (1-lam)/(j+2)], so the factor is the ratio of the
        baseline integral to the reshaped one. Returns 1.0 exactly when
        there is no chord law (the legacy path stays bit-for-bit).
        """
        if not self.chord_coeffs:
            return 1.0
        lam = self.taper
        base = 0.5 * (1.0 + lam)
        if self.chord_law == "poly":
            extra = sum(float(k) * (1.0 / (j + 1) - (1.0 - lam) / (j + 2))
                        for j, k in enumerate(self.chord_coeffs, start=1))
        else:
            # every law is affine in its parameters, so the reshaped area is
            # base + p @ w with w the basis' own weights — the same closed
            # form, one matrix further out
            extra = float(np.asarray(self.chord_coeffs, dtype=float)
                          @ chord_area_weights(len(self.chord_coeffs),
                                               self.chord_law, lam))
        den = base + extra
        if not (den > 0.0):   # unreachable while the multiplier floor holds
            raise ValueError("chord law integrates to a non-positive area")
        return base / den

    @property
    def chord_dev(self) -> float:
        """max |c / c_trap - 1| over the span — "how far from straight taper".

        Measured on the FLOWN chord (after area normalisation), so it is the
        deviation visible in the planform, not the raw coefficient sum.
        """
        if not self.chord_coeffs:
            return 0.0
        f = self.chord_area_factor
        lo, hi = chord_multiplier_extrema(self.chord_coeffs, taper=self.taper)
        return float(max(abs(lo * f - 1.0), abs(hi * f - 1.0)))

    @property
    def mac(self) -> float:
        # An ALL-ZERO law is the straight taper, so it takes the straight
        # taper's branch — not merely a branch that agrees with it. The two
        # forms are equal in exact arithmetic and 1 ulp apart in floating
        # point (1.020833333333333 vs 1.0208333333333333 at the published
        # wing), which is enough to break the guarantee the whole chord-law
        # family rests on: a twin with a flat law IS its base, bit-for-bit
        # (tests/test_chord_everywhere.py). Every other quantity already
        # satisfied it — the sampled chords are byte-identical and
        # chord_area_factor is exactly 1.0 — so mac was the one place the
        # promise held by luck rather than by construction.
        if not self.chord_coeffs or not any(self.chord_coeffs):
            lam = self.taper
            return (2.0 / 3.0) * self.c_root * (1 + lam + lam**2) / (1 + lam)
        # General definition int c^2 dy / int c dy (the closed form above is
        # its straight-taper special case). The y -> eta jacobian is constant
        # and cancels, and c/(c_root F) is a polynomial in eta, so both
        # integrals are exact — no quadrature error to leak into Re_mac.
        trap = np.polynomial.Polynomial([1.0, -(1.0 - self.taper)])
        law = self.chord_law
        if law in ("poly", "ends"):
            p = self._multiplier_poly() * trap
            num = (p * p).integ()
            den = p.integ()
            i2 = float(num(1.0) - num(0.0))
            i1 = float(den(1.0) - den(0.0))
        elif law == "kinked":
            # the CHORD is piecewise linear here (the multiplier is not), so
            # the segments already carry c/(c_root F) and integrate exactly —
            # the kink never lands inside a quadrature interval
            i1 = i2 = 0.0
            for a, b, seg in self._kinked_segments():
                num, den = (seg * seg).integ(), seg.integ()
                i2 += float(num(b) - num(a))
                i1 += float(den(b) - den(a))
        else:
            # elliptic: c/(c_root F) = (1 - t) c_trap + t sqrt(1 - eta^2), and
            # all three integrals of that pair are closed forms (int sqrt =
            # pi/4, int eta sqrt = 1/3), so this is exact too
            t = float(self.chord_coeffs[0])
            k = 1.0 - self.taper
            i1 = (1.0 - t) * (1.0 - 0.5 * k) + t * np.pi / 4.0
            i2 = ((1.0 - t) ** 2 * (1.0 - k + k * k / 3.0)
                  + 2.0 * t * (1.0 - t) * (np.pi / 4.0 - k / 3.0)
                  + t * t * (2.0 / 3.0))
        return float(self.c_root * self.chord_area_factor * i2 / i1)

    def _multiplier_poly(self):
        """The chord multiplier as a polynomial in eta (polynomial laws)."""
        if self.chord_law == "poly":
            return np.polynomial.Polynomial([1.0, *self.chord_coeffs])
        p = np.polynomial.Polynomial([1.0])
        for k, b in zip(self.chord_coeffs,
                        _ends_basis_poly(len(self.chord_coeffs), self.taper)):
            p = p + float(k) * b
        return p

    def _kinked_segments(self):
        """``(a, b, c/(c_root F) on [a, b])`` of the cranked law's panels.

        The CHORD, not the multiplier: this law's panels are straight in the
        chord, which is what makes it a cranked planform rather than a curved
        one, so the piecewise-linear object is ``m * c_trap``.
        """
        n = len(self.chord_coeffs)
        knots = np.concatenate([[0.0], _kink_stations(n), [1.0]])
        vals = (1.0 + np.asarray(self.chord_coeffs, dtype=float) @ chord_basis(
            knots, n, "kinked", self.taper)) \
            * (1.0 - (1.0 - self.taper) * knots)
        out = []
        for i in range(len(knots) - 1):
            a, b = float(knots[i]), float(knots[i + 1])
            slope = (vals[i + 1] - vals[i]) / (b - a)
            out.append((a, b, np.polynomial.Polynomial(
                [float(vals[i]) - slope * a, float(slope)])))
        return out

    def chord(self, y: np.ndarray) -> np.ndarray:
        eta = np.abs(2.0 * np.asarray(y) / self.b)
        c = self.c_root * (1.0 - (1.0 - self.taper) * eta)
        if not self.chord_coeffs:
            return c
        return c * chord_multiplier(eta, self.chord_coeffs,
                                    taper=self.taper) \
            * self.chord_area_factor

    def twist_deg(self, y: np.ndarray) -> np.ndarray:
        eta = np.abs(2.0 * np.asarray(y) / self.b)
        return self.twist_root_deg + (self.twist_tip_deg - self.twist_root_deg) * eta

    def sample(self, N: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(y, c, twist_rad) at the N midpoint-cosine LLT stations."""
        _, y = cosine_stations(N, self.b)
        return y, self.chord(y), np.deg2rad(self.twist_deg(y))


def bounds(mode: str = "trim",
           cant_bounds: tuple[float, float] | None = None,
           h_bounds: tuple[float, float] | None = None,
           chord_order: int = 0,
           chord_max_frac: float = CHORD_COEFF_BOUND,
           flight_free: bool = False,
           size_rows=None,
           chord_law: str = DEFAULT_CHORD_LAW) -> np.ndarray:
    """(d, 2) box bounds for the Tier A / Tier A+ design vector.

    ``chord_order`` (0 = off, bit-for-bit) APPENDS that many chord-law
    coefficient rows AFTER every mode row, so existing indices — alpha in
    soft mode, the winglet pair, V/altitude in mission mode — keep their
    positions and their meaning.

    ``flight_free`` (False = off, bit-for-bit) inserts the flight block
    (V, altitude) between the mode rows and the chord rows — the modifier
    stacking order documented at :func:`with_flight_bounds`. It is refused in
    ``mode="mission"``, whose vector already carries that pair: the mission
    mode IS the trim wing with this modifier on, and offering it twice would
    build a 7-D problem with two speeds.

    ``size_rows`` (None = off) is the SIZE block (span + area), passed as
    plain rows rather than as a flag because its box depends on the problem's
    own published size — which lives on the Problem, not here (sizing.py owns
    that physics; passing data keeps this module free of the weight model and
    of the import cycle it would drag in). It goes FIRST among the modifier
    blocks: [family][size][flight][chord].

    ``cant_bounds`` (winglet modes only) overrides the winglet cant row —
    this is how the winglet-TYPE presets (canted / vertical fence / raked
    wingtip) select their own cant sub-band. ``None`` keeps the legacy
    (60, 90) bound, so a default winglet run is bit-for-bit unchanged. The
    override is validated against the physical VLM limits
    (WINGLET_CANT_LIMITS_DEG) and must have hi > lo.
    """
    cant_row = WINGLET_CANT_BOUNDS_DEG
    if cant_bounds is not None:
        lo, hi = float(cant_bounds[0]), float(cant_bounds[1])
        cmin, cmax = WINGLET_CANT_LIMITS_DEG
        if not (cmin <= lo < hi <= cmax):
            raise ValueError(
                f"winglet cant_bounds {(lo, hi)} outside the physical VLM "
                f"range {WINGLET_CANT_LIMITS_DEG} or non-increasing")
        cant_row = (lo, hi)
    rows = [TAPER_BOUNDS, TWIST_ROOT_BOUNDS_DEG, TWIST_TIP_BOUNDS_DEG]
    # the device-height row, validated in ONE place. None -> the
    # published band, so an untouched run is bit-for-bit the
    # published run; a stated band is FLOWN rather than refused,
    # because 0.15 is where this model was measured and not a cap
    # on the device the user is designing.
    h_row = winglet_h_row(h_bounds)
    if mode == "soft":
        rows.append(ALPHA_BOUNDS_DEG)
    elif mode == "tier_a_plus":
        rows.append(SWEEP_BOUNDS_DEG)
        rows.append(TC_BOUNDS)
    elif mode in ("winglet", "winglet_capped"):
        rows += [h_row, cant_row]
    elif mode in ("winglet_tc", "winglet_capped_tc"):
        rows += [h_row, cant_row, TC_BOUNDS]
    elif mode in COUPLED_MODES:
        # the winglet vector + the two SUMMARY VARIABLES that index the
        # pre-optimised CST library: structural depth t/c and design lift cl.
        # Two rows, not one, and neither is the NACA thickness — that is why
        # these modes are NOT in the t/c list (a one-argument family lookup
        # on a two-dimensional library).
        rows += [h_row, cant_row, TC_SEC_BOUNDS,
                 CL_SEC_BOUNDS]
    elif mode in ("winglet_blended", "winglet_capped_blended"):
        rows += [h_row, cant_row, WINGLET_BLEND_BOUNDS]
    elif mode == "winglet_capped_blended_wing":
        # the blended vector plus the WING-side arc of the transition — the
        # freedom that lets the blend radius exceed the winglet's own height
        rows += [h_row, cant_row, WINGLET_BLEND_BOUNDS,
                 WING_BLEND_FRAC_BOUNDS]
    elif mode == "mission":
        rows += [MISSION_V_BOUNDS, MISSION_ALT_BOUNDS_M]
    elif mode != "trim":
        raise ValueError(f"unknown mode {mode!r}")
    if flight_free and mode == "mission":
        raise ValueError(
            "mode='mission' already carries the flight block (V, altitude) — "
            "it IS the trim wing with flight_free on; pass one or the other")
    box = np.array(rows, dtype=float)
    if size_rows is not None:
        box = np.vstack([box, np.asarray(size_rows, dtype=float)])
    box = with_flight_bounds(box, flight_free)
    return with_chord_bounds(box, chord_order, chord_max_frac, chord_law)


def wing_blend_from_x(x, mode: str, wing_blend_frac: float = 0.0) -> float:
    """Wing-side blend fraction in force for a candidate.

    ``mode="winglet_capped_blended_wing"`` reads x[6] — it is a DESIGN
    VARIABLE there. Every other mode takes the problem's own value (a flag,
    default 0.0 = the published geometry), so the wing-side blend composes
    with the existing winglet families without changing their vectors.
    """
    if mode == "winglet_capped_blended_wing":
        return float(np.asarray(x, dtype=float)[6])
    return float(wing_blend_frac)


#: modes whose design vector carries the section THICKNESS, and where.
#: One table, so a caller that needs t/c BEFORE the wing is built (the
#: weight model in the wing-loading size mode) reads it from the same place
#: :func:`wing_from_x` does instead of repeating the indices.
#: The COUPLED modes are here too, at the same row: their ``tc_sec`` is a
#: real structural depth (it is what the library member was optimised at and
#: what the section physically is), so the weight model must read it rather
#: than fall back to TC_DEFAULT — otherwise a wing-loading candidate is
#: weighed on a 12 % section it is not flying.
TC_INDEX = {"tier_a_plus": 4, "winglet_tc": 5, "winglet_capped_tc": 5,
            "winglet_coupled": 5, "winglet_capped_coupled": 5}

#: the thickness a Wing carries when its mode does not design one. Equal to
#: geometry.Wing's own default, i.e. the default polar's NACA 2412.
TC_DEFAULT = 0.12


def tc_from_x(x, mode: str = "trim") -> float:
    """Section thickness a candidate flies: from the vector where the mode
    designs it, the family default otherwise."""
    i = TC_INDEX.get(mode)
    return TC_DEFAULT if i is None else float(np.asarray(x, dtype=float)[i])


def wing_from_x(x: np.ndarray, b: float = 10.0, S: float = 10.0,
                mode: str = "trim", chord_order: int = 0,
                blend_shape: str = "arc",
                wing_blend_frac: float = 0.0,
                blend_frac: float = 0.0,
                chord_limits: "ChordLimits | None" = None,
                chord_law: str = DEFAULT_CHORD_LAW) -> Wing:
    """Design vector -> Wing.

    First three entries are always [taper, twist_root_deg, twist_tip_deg].
    ``mode="tier_a_plus"`` additionally reads x[3] = sweep_deg, x[4] = tc;
    in soft mode x[3] is alpha and is handled by the objective, not here.
    ``mode="mission"`` likewise uses only the first three entries — x[3] = V
    and x[4] = altitude are flow-state variables handled by the objective.

    ``mode="winglet_capped"``: the TOTAL PROJECTED SPAN is capped at ``b``
    — the wing panel shrinks to pay for the winglet's horizontal projection
    (arc length h = x[3] * b/2 at cant x[4] from the wing plane projects
    h cos(cant) per side), at constant area S:

        b_wing = b * (1 - x[3] * cos(cant))

    At cant = 90 deg the projection vanishes and the mode coincides with
    the free-span "winglet" mode exactly (gated in tests).

    ``chord_order`` > 0 reads that many TRAILING entries as the chord-law
    coefficients (bounds() appends them there). Raises ValueError — via the
    Wing constructor — if the law collapses the chord.
    """
    x = np.asarray(x, dtype=float)
    kw = {}
    if int(chord_order) > 0:
        kw["chord_coeffs"] = chord_coeffs_from_x(x, int(chord_order),
                                                 chord_law)
    if mode == "tier_a_plus":
        kw.update(sweep_deg=float(x[3]), tc=float(x[4]))
    if mode in ("winglet_tc", "winglet_capped_tc"):
        # winglet vector + section thickness: x[5] = tc selects the polar
        # family member (objective) and is recorded on the Wing (unswept VLM).
        kw.update(tc=float(x[5]))
    if mode in COUPLED_MODES:
        # winglet vector + the two summary variables. x[5] = tc_sec is the
        # STRUCTURAL DEPTH of the library member the search is demanding, so
        # it is the Wing's thickness for every consumer that asks (the weight
        # model, the breakdown, the CAD loft); x[6] = cl_sec selects WHICH
        # pre-optimised member, and is read by the objective, not here — a
        # design lift is not a property of a planform.
        kw.update(tc=float(x[5]))
    if mode in CAPPED_COSINE_MODES:
        if float(blend_frac) > 0.0 or float(wing_blend_frac) > 0.0:
            # a FIXED blend (objective.Problem.blend_frac_fixed) on a mode
            # whose vector has no blend row. It reaches further outboard than
            # h cos(cant) exactly as a designed blend does, so it is capped
            # on the same developed line — capping it on the cosine would
            # hand the blend free projected span.
            b = 2.0 * developed_semispan(
                b / 2.0, float(x[3]) * b / 2.0, float(x[4]),
                float(blend_frac), blend_shape,
                wing_arc=float(wing_blend_frac) * b / 2.0)
        else:
            b = b * (1.0 - float(x[3]) * np.cos(np.deg2rad(float(x[4]))))
    elif mode in ("winglet_capped_blended", "winglet_capped_blended_wing"):
        # a blended device leaves the wing plane tangentially and so reaches
        # FURTHER outboard than h cos(cant) — capping on the cosine would
        # hand the blend free projected span (the raked-tip trap again). A
        # WING-side blend bends the outer wing up as well, so the cap is
        # solved on the whole developed line, not on the device alone.
        w = wing_blend_from_x(x, mode, wing_blend_frac)
        b = 2.0 * developed_semispan(
            b / 2.0, float(x[3]) * b / 2.0, float(x[4]), float(x[5]),
            blend_shape, wing_arc=w * b / 2.0)
    return Wing(b=b, S=S, taper=float(x[0]),
                twist_root_deg=float(x[1]), twist_tip_deg=float(x[2]),
                chord_limits=chord_limits, **kw)
