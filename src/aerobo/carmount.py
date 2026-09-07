"""How a rear wing is ATTACHED to the car — the mount as a continuum.

Why this module exists
----------------------
carwing.py asks the mount question as a two-valued flag
(``carwing.MOUNTS`` = ``centre`` | ``ends``) and answers it with two hard-coded
beam formulas. That flag conflates two independent statements:

    WHERE the load leaves the wing   — a spanwise STATION, y_s in metres
    WHAT carries it from there       — pylons to a deck, or the endplates

``centre`` is (station 0, pylons); ``ends`` is (station b/2, endplates). Every
other combination is a real rear wing and none of them can be asked for. A
mount 35% out along the semi-span — the arrangement that halves the bending
demand — is not an exotic layout, it is what a wide wing with a narrow chassis
does, and the published family cannot express it. Worse, the two named ends
carry a hidden third statement each: ``centre`` charges two pylons whose length
is the whole ride height, and neither end says anything about the chordwise
attachment point, which is where the wing's incidence at speed is decided.

So this module takes the mount apart into the five things it actually is, each
a pure function or a small dataclass, each independently testable:

1. A SUPPORT STATION (:func:`beam_moment`, :func:`beam_deflection`).
   Symmetric supports at +/- y_s with free overhangs outboard. The continuum
   contains both published ends: at y_s = 0 it reproduces the cantilever, at
   y_s = b/2 the simply-supported beam, on the same load array and to the
   bit (tests/test_carmount.py).
2. A PYLON LENGTH (:func:`pylon_length_m`). A pylon runs from the wing to the
   car's DECK, not to the track.
3. PYLON PARASITE DRAG (:func:`pylon_parasite_cd`) as a wetted-area build-up
   at the pylon's own Reynolds number, with the crude flat-plate model kept
   reachable beside it.
4. A JUNCTION AT ITS OWN STATION (:func:`mount_junction_cd`). The corner is
   priced on the chord where the pylon actually meets the wing.
5. TORSION (:func:`torsion_moment`, :func:`torsion_twist`,
   :func:`divergence_q`), and therefore a CHORDWISE attachment point, because
   a mount that is not at the aerodynamic centre winds the wing up or down at
   speed and changes its incidence.
6. A SUCTION-SIDE LOSS (:meth:`MountSpec.modifiers`) — carwing.py's own
   docstring prescribes the shape ("it enters as a per-strip loss and belongs
   next to slipstream.py's modifiers") and this is that hook, with a magnitude
   the user owns and a default of exactly zero.

:class:`MountSpec` ties them together, and :data:`PUBLISHED_LAYOUTS` expresses
carwing.MOUNTS in the new vocabulary so the older family can adopt this module
without moving any published answer.

This module does NOT import carwing.py. The dependency runs the other way (a
family imports its mount, not the reverse), which is also why every function
here takes arrays and floats rather than a problem object.

Frame and sign conventions (DERIVED, not assumed)
-------------------------------------------------
carwing.py solves the car flipped upside down: model +z IS the car's downward
direction, model lift IS downforce, the track is a rigid wall ABOVE the wing.
This module inherits that frame and states everything in it, so nothing is
inverted by hand.

* ``load`` l(y) [N/m] is the sectional aerodynamic force per unit span, taken
  POSITIVE in the direction it actually acts (model +z = towards the track =
  the car's downforce direction). In car terms the LOAD SIDE of the wing is
  the surface facing the track, which on an inverted wing is its SUCTION
  side. That identity is why item 6 lives in this module and not elsewhere: a
  pylon that reaches the wing from the car reaches it on the load side.

* BENDING. M(y) is the moment about station y of everything OUTBOARD of y:

      M(y) = int_y^{b/2} l(e) (e - y) de  -  R * max(y_s - y, 0)          (1)

  with R = int_0^{b/2} l de the reaction at one support (half the total load,
  by symmetry). Euler-Bernoulli in this convention reads EI w'' = M with w the
  deflection measured POSITIVE IN THE LOAD DIRECTION, which is derived rather
  than declared: a cantilever off the support has M > 0 and droops in the load
  direction (w'' > 0), a beam between two supports has M < 0 at midspan and
  sags in the load direction (w'' < 0 with w' = 0 at the centre). So

      M > 0  outboard of the supports  ->  HOGGING: tension on the face AWAY
             from the load, i.e. on the car's PRESSURE side
      M < 0  between the supports      ->  SAGGING: tension on the LOAD side,
             i.e. on the car's SUCTION side

  and a mount anywhere in between puts the wing in both senses at once, with
  the sign change at the station where (1) crosses zero. Which face is in
  tension is not decoration: it decides where a laminate's plies and a
  fastener's pull-out go, and the two published mounts answer it oppositely.
  :func:`bending_sense` names it for a given moment array.

* TORSION. phi(y) is the elastic twist, positive in the sense that INCREASES
  the model incidence (= increases downforce). The sectional torque about the
  mount line is

      m(y) = l(y) * ( x_mount(y) - x_ac(y) )                              (2)

  with x measured AFT from the local leading edge. A mount AFT of the
  aerodynamic centre gives m > 0 for l > 0: the wing winds up towards more
  downforce, which feeds back on itself and is the divergence-prone sense. A
  mount AHEAD of it sheds incidence with speed. Both signs are offered; the
  default attachment fraction is 0.25, the quarter chord, which is where the
  VLM actually puts its bound vortex, so the DEFAULT WINDUP IS EXACTLY ZERO
  and this module changes no published answer by existing.

What is modelled, and how crudely
---------------------------------
* The beam is a single spanwise Euler-Bernoulli member of uniform EI, and the
  torsion member a single spanwise St-Venant shaft of uniform GJ. Neither
  stiffness varies with the chord the design vector is moving. That is the
  same reduced-order statement carwing.py already makes with ``ei_nm2``, kept
  rather than quietly upgraded.
* The supports are PINNED IN BENDING and FIXED IN TORSION. The first is not a
  choice: the spar runs continuously through the mount, so the mount takes
  shear and the bending moment passes through it. The second is not a choice
  either, in the opposite direction: a mount that could not react torque would
  leave the wing's incidence undefined, so a real mount always has chordwise
  extent (a pylon foot, or an endplate whose depth is its chord). What IS a
  choice is treating that reaction as RIGID — the pylon's own bending and the
  deck's local compliance are not in series with EI or GJ. Bias direction:
  OPTIMISTIC. The real wing moves further than this says and diverges at a
  lower speed.
* The sectional lift slope used for divergence is the 2-D one. The finite wing
  responds less than a strip-theory sum of sections, so using the 2-D slope
  puts q_div LOW: CONSERVATIVE, deliberately.
* Divergence is reduced to ONE degree of freedom and ONE mode, calibrated on
  the distributed solution's own peak twist under a uniform torque
  (:func:`torsion_effective_length`, derived below). There is no camber term,
  no aerodynamic-moment (Cm0) term, no aeroelastic coupling between bending and
  torsion, and no unsteady term — so no flutter, only static divergence.

What is NOT modelled (and matters in reality)
---------------------------------------------
* THE SIZE OF THE SUCTION-SIDE LOSS. Item 6 supplies the SHAPE of the
  knockdown and its footprint; the magnitude is a number this package cannot
  produce. It defaults to exactly zero and must be set by whoever has CFD or
  tunnel data. A default that guessed would be a fabricated measurement.
* The pylon as a lifting/interfering body in its own right: it is charged
  wetted area and a junction, not a wake that the wing then flies through, and
  not a blockage that changes the local ride height.
* Pylons at a spanwise station different from the beam's supports, pylons that
  are not in a symmetric pair, and swept or raked pylons.
* The mount's own mass and the inertia loads (cornering, kerbs, braking) that
  usually size a real rear-wing mount. This package carries no mass model for
  cars.
* Fatigue, bolt-up preload, and every failure mode that is not a peak
  deflection.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import geometry, junction, slipstream
from .drag import skin_friction_cf, wing_form_factor

#: what carries the wing's load down to the car. Split from the STATION on
#: purpose: carwing.MOUNTS answers both questions with one word, which is why
#: it can offer only two of the four corners and none of the interior.
#:
#: ``n_junctions_per_pylon`` is 1 — a pylon meeting the wing makes one corner —
#: so a twin-pylon mount prices two. carwing.MOUNTS has no pylon layout any
#: more (both of its layouts are plate-borne), so this is reachable only
#: through a hand-built MountSpec — see PUBLISHED_LAYOUTS.
MOUNT_KINDS: dict[str, dict] = {
    "pylon": {
        "label": "pylon(s) to the car's deck",
        "carries_pylons": True,
        "note": "struts from the wing down to the attachment deck: extra "
                "wetted area, a wing/pylon corner apiece, and — if they reach "
                "the wing on its suction side — a per-strip lift loss over "
                "their footprint",
    },
    "endplate": {
        "label": "through the endplates",
        "carries_pylons": False,
        "note": "the load leaves through plates that are there anyway: no "
                "extra wetted area, no new corner, no footprint on the wing. "
                "The plate's own structure is endplate.py's question, not "
                "this module's",
    },
}

#: WHICH SIDE of the wing the mount reaches it on, and therefore whether the
#: suction-side loss applies at all.
#:
#: The pressure-side entry is the whole reason the knob exists. A swan-neck
#: pylon comes over the top of the wing and attaches on the PRESSURE side
#: precisely so that the suction surface — which on an inverted wing is the
#: one facing the track, and is where the downforce is made — is left
#: undisturbed. An underslung pylon sits in the suction peak instead. That
#: trade is the reason swan necks are built at all, and this package has no
#: way to price it from first principles: a lifting-surface method sees only
#: the (y, z) line, and no reduced-order correlation gives a chordwise,
#: viscous, geometry-specific separation honestly. So the SIDE decides whether
#: a loss applies and the MAGNITUDE is the user's to supply.
MOUNT_SIDES: dict[str, dict] = {
    "suction": {
        "label": "underslung (attaches on the suction side)",
        "loses_lift": True,
        "note": "the pylon sits in the wing's suction peak — the side facing "
                "the track on an inverted wing, and the side the downforce "
                "comes from",
    },
    "pressure": {
        "label": "swan neck (attaches on the pressure side)",
        "loses_lift": False,
        "note": "the pylon reaches over the top and lands on the pressure "
                "side, leaving the suction surface clean. This is what a swan "
                "neck buys, and why the loss knob has an OFF setting that is "
                "a geometry statement rather than a magnitude of zero",
    },
}

#: how wide the pylon's footprint on the wing is, as a MULTIPLE of the pylon's
#: own maximum thickness. MODEL CHOICE, not a measurement: the disturbed band
#: is flat-topped over the pylon's own thickness and tapers to nothing at
#: +/- half of ``spread`` thicknesses either side (see
#: :func:`suction_loss_modifiers`, where the flat core is DERIVED from this
#: number rather than stated separately). A real footprint is set by the
#: pylon's wake and by the wing's own chordwise pressure gradient, neither of
#: which this package resolves. The number matters only in proportion to a
#: magnitude whose default is zero.
DEFAULT_LOSS_SPREAD = 3.0

#: how much of the footprint's own integral a POINT-SAMPLED knockdown may lose
#: (or gain) before :func:`footprint_is_resolved` refuses the grid.
#:
#: A MODEL CHOICE — but the measurement says it is not a load-bearing one. The
#: point sample's error against the footprint's integral is governed by the
#: panel width in units of the footprint HALF-width, r = max(dy)/w, and its
#: envelope over the PHASE (where the pylon happens to land between two
#: stations) has a plateau and then a cliff. Swept over 120 phases on the
#: published pylon's footprint (w = 0.0216 m, core 1/3) the worst error is
#: 0.1% at r = 0.2, 1.9% at r = 0.5, 5.1% at r = 0.6, 17% at r = 0.8 and 28%
#: at r = 1.0. So every tolerance in 2%-10% puts the cut in nearly the same
#: place — r* = 0.50, 0.57 and 0.73, i.e. 2.7 to 4 stations across the
#: footprint's full width — and 5% sits in the middle of that plateau, which
#: is why the exact number chosen does not decide many grids. Below the cut
#: the verdict is a property of the GRID; above it, it is a property of the
#: phase, which is what a resolution test must not depend on.
#:
#: Re-derived, not pinned, in tests/test_carmount.py.
FOOTPRINT_INTEGRAL_TOL = 0.05

#: sub-intervals per footprint half-width in the reference quadrature of
#: :func:`footprint_integral_error`. The reference has to be converged far
#: past the tolerance it is compared against, and it is: at this count the
#: midpoint rule reproduces the isolated footprint's closed-form integral
#: (1 + core_frac) * w to better than 2e-8 relative — more than six orders
#: below :data:`FOOTPRINT_INTEGRAL_TOL`, so the guard's verdict is a statement
#: about the caller's grid and never about this one. Measured in
#: tests/test_carmount.py against the closed form, not against itself.
_QUAD_PER_HALF_WIDTH = 200

#: chordwise position of the section aerodynamic centre, as a fraction of the
#: local chord. Not an assumption bolted on here: it is where vlm.py puts the
#: bound vortex, so a mount at this fraction is a mount on the very line the
#: solver applies the lift to, and the torsional moment (2) is then exactly
#: zero rather than nearly zero.
X_AC_FRAC = 0.25


# ------------------------------------------------------------ half-span views

def _halves(y: np.ndarray, arr: np.ndarray):
    """Sort onto the half span, the way carwing.bending_moment does.

    Returns ``(order, ys, yh, ah)``: the sort permutation, the sorted
    stations, and the y >= 0 half of the stations and of ``arr``. The wing is
    symmetric, so every beam quantity here is a function of |y| and is
    computed once on the half span.
    """
    y = np.asarray(y, dtype=float)
    arr = np.asarray(arr, dtype=float)
    if y.shape != arr.shape:
        raise ValueError(
            f"y and the per-station array must have the same shape "
            f"(got {y.shape} and {arr.shape})")
    if y.ndim != 1:
        raise ValueError(f"y must be one-dimensional, got shape {y.shape}")
    order = np.argsort(y)
    ys, a_s = y[order], arr[order]
    half = ys >= 0.0
    yh, ah = ys[half], a_s[half]
    if yh.size < 2:
        raise ValueError(
            f"need at least two stations on the half span, got {yh.size}")
    return order, ys, yh, ah


def _mirror(order: np.ndarray, ys: np.ndarray, yh: np.ndarray,
            vals: np.ndarray) -> np.ndarray:
    """Put a half-span quantity back onto the caller's own station order."""
    full = np.interp(np.abs(ys), yh, vals)
    inverse = np.empty_like(order)
    inverse[order] = np.arange(order.size)
    return full[inverse]


# ------------------------------------------------------------------- 1. beam

def support_reaction(y: np.ndarray, load: np.ndarray) -> float:
    """Vertical reaction at ONE support [N].

    The layout is symmetric and the load is symmetric, so each of the two
    supports carries exactly half the total: R = int_0^{b/2} l(y) dy, whatever
    the station. This is why moving the mount changes the bending DISTRIBUTION
    and never the reaction, and why the pylons of a wide-set mount are no
    lighter than those of a narrow-set one.
    """
    _order, _ys, yh, lh = _halves(y, load)
    return float(np.trapezoid(lh, yh))


def beam_moment(y: np.ndarray, load: np.ndarray,
                y_station: float) -> np.ndarray:
    """SIGNED bending moment [N m] for symmetric supports at +/- ``y_station``.

    Equation (1) of the module docstring, evaluated at the caller's stations::

        M(y) = int_|y|^{y_max} l(e) (e - |y|) de  -  R * max(y_s - |y|, 0)

    Sign: positive is HOGGING (tension away from the load), negative is
    SAGGING (tension on the load side) — derived in the module docstring.
    Take ``np.abs`` for the demand on the section, which is what
    carwing.bending_moment returns.

    The two published layouts are the two ends of this one expression:

        y_station = 0     the reaction arm is zero everywhere on the half
                          span, so the second term vanishes IDENTICALLY (not
                          approximately) and M is the cantilever integral —
                          carwing.bending_moment(..., "centre"), to the bit.
        y_station = b/2   the second term acts everywhere, and the result is
                          the exact negation of
                          carwing.bending_moment(..., "ends"), whose magnitude
                          is therefore the same float.

    ``y_station`` is a LENGTH in metres, not a fraction: where a wing bolts to
    a chassis is a distance between two pieces of car, and the span it is
    measured against is itself a design variable in this family. The span
    enters this function only through the station.
    """
    order, ys, yh, lh = _halves(y, load)
    y_s = float(y_station)
    if y_s < 0.0:
        raise ValueError(f"y_station must be >= 0 (got {y_s})")
    out = np.empty_like(yh)
    for i, yi in enumerate(yh):
        arm = yh[i:] - yi
        out[i] = np.trapezoid(lh[i:] * arm, yh[i:])
    reaction = float(np.trapezoid(lh, yh))
    # max(y_s - y, 0): outboard of the support there is no reaction in the
    # free body, and at y_s = 0 this is an exact 0.0 for every station, so the
    # cantilever branch is bit-for-bit carwing's.
    out = out - reaction * np.maximum(y_s - yh, 0.0)
    return _mirror(order, ys, yh, out)


def bending_sense(moment: np.ndarray) -> str:
    """``"hogging"`` | ``"sagging"`` | ``"mixed"`` for a signed moment array.

    Names what :func:`beam_moment`'s sign means, so a breakdown can say which
    face of the wing is in tension instead of leaving it to be re-derived.
    A mount with free overhangs is generically ``mixed``: hogging outboard of
    the supports, sagging between them.
    """
    m = np.asarray(moment, dtype=float)
    pos = bool(np.any(m > 0.0))
    neg = bool(np.any(m < 0.0))
    if pos and neg:
        return "mixed"
    if neg:
        return "sagging"
    return "hogging"


def _deflection_half(y: np.ndarray, moment: np.ndarray, y_station: float):
    """``(yh, w_rel)``: half-span deflection per unit EI, relative to the mount.

    Curvature is M/EI; with EI factored out this is the stiffness-independent
    comparison between layouts that the mount question needs, and it is the
    same contract carwing.deflection_index carries. Integrated twice from the
    innermost station with zero slope there (symmetry), then referred to the
    support by subtracting w(y_s) — which is what "movement RELATIVE TO ITS
    MOUNT" means, and what makes the two published indices two readings of one
    array.

    The SIGNED moment must be passed. carwing.deflection_index is handed a
    magnitude, which is harmless for its two layouts because each of them has
    one sign of moment throughout, and wrong for every station in between,
    where the moment changes sign at the support.
    """
    order, ys, yh, mh = _halves(y, moment)
    y_s = float(y_station)
    slope = np.concatenate([[0.0], np.cumsum(
        0.5 * (mh[1:] + mh[:-1]) * np.diff(yh))])          # int kappa dy
    w = np.concatenate([[0.0], np.cumsum(
        0.5 * (slope[1:] + slope[:-1]) * np.diff(yh))])    # int slope dy
    # np.interp clamps outside the sampled range, which is exactly right at
    # both published ends: a station at 0 lands on w[0] = 0 (an exact
    # subtraction of 0.0) and a station at b/2 lands on the outermost panel's
    # w, which is the datum carwing uses for its "ends" index.
    w_mount = float(np.interp(abs(y_s), yh, w))
    return order, ys, yh, w - w_mount


def beam_deflection(y: np.ndarray, moment: np.ndarray,
                    y_station: float) -> np.ndarray:
    """Deflection per unit EI [m^3/N] relative to the mount, at each station.

    Positive is movement in the LOAD direction (towards the track, on a car).
    Zero at the support by construction.
    """
    order, ys, yh, w = _deflection_half(y, moment, y_station)
    return _mirror(order, ys, yh, w)


def beam_deflection_index(y: np.ndarray, moment: np.ndarray,
                          y_station: float) -> float:
    """Largest movement of the wing relative to its mount, per unit EI.

    Same contract as carwing.deflection_index — divide by EI for metres — and
    the same two numbers at the two published stations:

        y_station = 0     the largest movement is the tip's, so this is the
                          tip droop carwing reports for "centre"
        y_station = b/2   the largest movement is the centre's, so this is the
                          centre sag carwing reports for "ends"

    Taken as a MAXIMUM rather than read off a named station, because at an
    interior mount neither end is automatically the worst: the wing has both a
    tip and a centre free to move, and which of them wins is the whole point
    of choosing a station.
    """
    _order, _ys, _yh, w = _deflection_half(y, moment, y_station)
    return float(np.max(np.abs(w)))


def beam_report(y: np.ndarray, load: np.ndarray, y_station: float,
                ei_nm2: float | None = None) -> dict:
    """Everything the beam says about one support station, in one dict."""
    m = beam_moment(y, load, y_station)
    order, ys, yh, w = _deflection_half(y, m, y_station)
    idx = float(np.max(np.abs(w)))
    out = {
        "y_station_m": float(y_station),
        "reaction_N": support_reaction(y, load),
        "moment_Nm": m,
        "moment_abs_Nm": np.abs(m),
        "M_max_Nm": float(np.max(np.abs(m))),
        "M_hogging_max_Nm": float(max(0.0, np.max(m))),
        "M_sagging_max_Nm": float(max(0.0, -np.min(m))),
        "sense": bending_sense(m),
        "deflection_index": idx,
        "w_tip_index": float(w[-1]),
        "w_centre_index": float(w[0]),
    }
    if ei_nm2 is not None:
        ei = float(ei_nm2)
        if not ei > 0.0:
            raise ValueError(f"ei_nm2 must be > 0 (got {ei_nm2})")
        out["deflection_m"] = idx / ei
        out["w_tip_m"] = float(w[-1]) / ei
        out["w_centre_m"] = float(w[0]) / ei
        out["EI_Nm2"] = ei
    return out


# ------------------------------- the overhanging beam, in closed form -------

def uniform_moment(y, w0: float, b: float, y_station: float) -> np.ndarray:
    """Closed-form M(y) [N m] for a UNIFORM load on the overhanging beam.

    Derivation (the module's own, from equation (1) with l = w0 and
    L = b/2, a = y_s, so that R = w0 L):

      overhang, a <= |y| <= L
          M = int_|y|^L w0 (e - |y|) de = (w0/2) (L - |y|)^2               (3)
      interior, |y| < a
          M = (w0/2)(L - |y|)^2 - w0 L (a - |y|)
            = (w0/2) ( y^2 + L^2 - 2 L a )                                 (4)

    Two readings that are worth having in closed form. (4) is a parabola in y
    with its extremum at the centreline, so the peak SAGGING moment is always
    at midspan; and (3) puts the peak HOGGING moment at the support, w0 e^2/2
    with e = L - a the overhang. Setting the two magnitudes equal,

          (L - a)^2 = 2 L a - L^2   ->   a^2 - 4 L a + 2 L^2 = 0
          a = L (2 - sqrt(2)) = 0.5858 L                                   (5)

    is the support station at which the wing's worst hogging and worst sagging
    demands are the same size — the balanced-moment mount, and the station a
    structure sized by peak moment alone would pick.
    """
    yy = np.abs(np.asarray(y, dtype=float))
    L, a = 0.5 * float(b), float(y_station)
    w0 = float(w0)
    return (0.5 * w0 * (L - yy) ** 2
            - (w0 * L) * np.maximum(a - yy, 0.0))


def uniform_deflection(w0: float, b: float, y_station: float) -> dict:
    """Closed-form deflections per unit EI of the uniformly loaded overhang.

    Integrating (4) twice from the centreline with w(0) = 0 and w'(0) = 0
    (symmetry), then (3) outwards from the support:

        EI w(y)  = (w0/2) ( y^4/12 + (L^2 - 2 L a) y^2 / 2 )     y <= a
        EI w'(a) = (w0/2) ( a^3/3  + (L^2 - 2 L a) a )
        EI w(L)  = EI w(a) + EI w'(a) e + (w0/8) e^4,   e = L - a

    so, relative to the mount,

        EI * (w_tip    - w(a)) = (w0/2)(a^3/3 + (L^2 - 2La) a) e + w0 e^4/8
        EI * (w_centre - w(a)) = -(w0/2)( a^4/12 + (L^2 - 2La) a^2/2 )

    Both ends check against the published closed forms, which is what makes
    this an independent gate rather than a restatement of the numerics:

        a = 0 : tip = w0 L^4/8 = w0 b^4/128, centre = 0 (the two cantilevers)
        a = L : centre = 5 w0 L^4/24 = 5 w0 b^4/384, tip = 0 (simply supported)

    Returns ``tip``, ``centre`` (both relative to the mount, positive in the
    load direction) and ``index`` = the larger magnitude of the two. ``index``
    is the closed form of :func:`beam_deflection_index` wherever the largest
    movement is at an end, which is everywhere except a narrow window around
    a/L in (0.5, 0.551) — there the interior slope turns over before reaching
    the support and the extremum is inside the span. Stated rather than
    papered over: outside that window the two agree, inside it the numerical
    index is the honest one.
    """
    L, a = 0.5 * float(b), float(y_station)
    w0, e = float(w0), 0.5 * float(b) - float(y_station)
    k = L * L - 2.0 * L * a
    w_a = 0.5 * w0 * (a ** 4 / 12.0 + k * a * a / 2.0)
    slope_a = 0.5 * w0 * (a ** 3 / 3.0 + k * a)
    tip = slope_a * e + w0 * e ** 4 / 8.0
    centre = -w_a
    return {"tip": float(tip), "centre": float(centre),
            "index": float(max(abs(tip), abs(centre)))}


def balanced_moment_station(b: float) -> float:
    """Support station [m] where peak hogging and peak sagging are equal.

    Root (5) of the module's own derivation, a = (b/2)(2 - sqrt(2)). Derived
    here rather than pasted so it cannot drift from the formula it came from.
    """
    return float(0.5 * float(b) * (2.0 - np.sqrt(2.0)))


# --------------------------------------------------------------- 2. the pylon

def pylon_length_m(ride_height_m: float, deck_height_m: float) -> float:
    """Length of ONE pylon [m]: from the wing down to the car's DECK.

        L_pylon = ride - deck                                             (6)

    A pylon does not reach the track. It reaches the bodywork hardpoint the
    wing bolts to, which sits ``deck_height_m`` above the track — the same
    distance endplate.py calls ``reach_m`` and applies to the plate that has
    to span it. carwing._strut_cd0 is handed the whole RIDE HEIGHT instead
    (``_strut_cd0(spec["n_struts"], ride, ...)``), so it charges the pylon
    over a length that includes the gap between the deck and the track, where
    there is no pylon.

    Bias direction of that error: the wetted area is too large, so the drag is
    too high and the drag-budget margin too tight — CONSERVATIVE, but by a
    factor, not a trim. At the published operating point (ride 0.30 m,
    deck 0.25 m) the correct length is 0.05 m against 0.30 m charged: a factor
    of six, measured in tests/test_carmount.py.

    Returns a length that may be zero or negative; the caller decides what
    that means. :meth:`MountSpec.violation` turns it into an in-contract
    reason, because a wing sitting at or below its own attachment deck is a
    design the optimiser can propose and not a programming error.
    """
    return float(ride_height_m) - float(deck_height_m)


def pylon_parasite_cd(n_pylons: int, length_m: float, chord_m: float,
                      tc: float, s_ref: float,
                      rho: float, V: float, mu: float,
                      model: str = "buildup", cf: float = 0.005) -> dict:
    """Pylon parasite drag as a coefficient on ``s_ref``, with its parts.

    Two models, both reachable, so the comparison can be made rather than
    asserted:

    ``buildup``     wetted-area build-up at the PYLON'S OWN Reynolds number,
                    exactly the way endplate.parasite_cd treats the plate:

                        Swet = n * 2 * L * c        (both faces of each pylon)
                        Re   = rho V c / mu         (on the PYLON's chord)
                        CD   = Cf(Re) * FF(t/c) * Swet / Sref              (7)

                    with ``drag.skin_friction_cf`` for Cf and
                    ``drag.wing_form_factor`` for FF. Nothing here is charged
                    at the wing's Reynolds number: a 0.12 m pylon behind a
                    0.25 m wing chord sits at about half the wing's Re and has
                    a measurably higher Cf for it.

    ``flat_plate``  the crude model carwing._strut_cd0 uses: a fixed turbulent
                    Cf of 0.005 and NO form factor. Written with the same
                    expression tree as _strut_cd0 so that a family adopting
                    this module reports the identical float rather than one
                    that merely rounds to it.

    Which is conservative? Neither, uniformly — the crude model is missing two
    things that pull in opposite directions and one that does not:

      * no form factor at all (FF = 1 against 1.26 at t/c = 0.12), which
        UNDER-charges by 26%;
      * a Cf frozen at 0.005 against 0.0052 at the published pylon's own Re,
        which under-charges by a further 4%;
      * so at the same length the flat-plate model is OPTIMISTIC by about
        31%, measured in tests/test_carmount.py.

    It is only conservative overall because carwing feeds it the wrong LENGTH
    (see :func:`pylon_length_m`), and a 6x length error swamps a 1.3x model
    error. Two wrongs of different sizes are not a calibration.
    """
    n = int(n_pylons)
    if n < 0:
        raise ValueError(f"n_pylons must be >= 0 (got {n_pylons})")
    if model not in ("buildup", "flat_plate"):
        raise ValueError(
            f"unknown pylon drag model {model!r}; "
            f"choose from ('buildup', 'flat_plate')")
    if not float(s_ref) > 0.0:
        raise ValueError(f"reference area must be > 0 (got {s_ref})")
    length, chord = float(length_m), float(chord_m)
    ff = wing_form_factor(float(tc))
    if n == 0 or length <= 0.0 or chord <= 0.0:
        return {"CD": 0.0, "CD_per_pylon": 0.0, "Swet_m2": 0.0,
                "Cf": 0.0, "FF": ff, "Re": 0.0, "model": model,
                "n_pylons": n, "length_m": length, "chord_m": chord}
    swet = n * 2.0 * length * chord
    if model == "flat_plate":
        cf_used, ff_used = float(cf), 1.0
        # carwing._strut_cd0's own grouping, kept character for character so
        # the adoption is bit-for-bit and not merely close
        cd = float(n * float(cf) * (2.0 * length * chord) / float(s_ref))
    else:
        cf_used = skin_friction_cf(float(rho) * float(V) * chord / float(mu),
                                   lref=chord)
        ff_used = ff
        cd = float(cf_used * ff_used * swet / float(s_ref))
    return {
        "CD": cd, "CD_per_pylon": cd / n,
        "Swet_m2": float(swet), "Cf": float(cf_used), "FF": float(ff_used),
        "Re": float(float(rho) * float(V) * chord / float(mu)),
        "model": model, "n_pylons": n,
        "length_m": length, "chord_m": chord,
    }


# ------------------------------------------------- 4. the junction, on station

def mount_junction_cd(wing: "geometry.Wing", y_station: float, s_ref: float,
                      n_junctions: int, tc: float | None = None,
                      radius: float = 0.0) -> float:
    """Junction drag of the mount's corners, at the LOCAL chord [-].

    carwing.py prices the wing/pylon corner at the ROOT chord whatever the
    mount is::

        junction.junction_cd(prob.tc, wing.chord([0.0])[0], s_ref, ...)

    which is right only for a mount at the centreline. The corner is a local
    object: Hoerner's correlation is a drag AREA built on the thickness at the
    junction, t = (t/c) c(y_s), and the area therefore scales as c(y_s)^2 at
    fixed thickness ratio. On a tapered wing that is not a detail — moving the
    pylons from the root to 70% of the semi-span on the published planform
    (taper 0.8) drops the charge by the square of the chord ratio.

    ``tc`` defaults to the WING's own thickness ratio, which is what carwing
    passes; the correlation's ``t`` is then the wing's thickness at the corner.
    A caller who would rather price the corner on the pylon's thickness passes
    it. Either reading is defensible — Hoerner's ``t`` is "the maximum
    thickness" of the junction — and the default is the one that keeps the
    published answer.

    At ``y_station`` = 0, ``tc`` = None and ``radius`` = 0 this returns exactly
    what carwing computes today.
    """
    y_s = float(y_station)
    semi = 0.5 * float(wing.b)
    if not (0.0 <= y_s <= semi + 1e-12):
        raise ValueError(
            f"y_station {y_s} is outside the semi-span [0, {semi}]")
    if int(n_junctions) == 0:
        return 0.0
    chord = float(wing.chord(np.array([min(y_s, semi)]))[0])
    thickness_ratio = float(wing.tc if tc is None else tc)
    return junction.junction_cd(thickness_ratio, chord, float(s_ref),
                                radius=float(radius),
                                n_junctions=int(n_junctions))


# ---------------------------------------------------------------- 5. torsion

def torsion_moment(y: np.ndarray, load: np.ndarray, chord: np.ndarray,
                   x_attach_frac: float,
                   x_ac_frac: float = X_AC_FRAC) -> np.ndarray:
    """Sectional torque about the mount line [N m / m].

    Equation (2) of the module docstring, per unit span::

        m(y) = l(y) * c(y) * (x_attach_frac - x_ac_frac)                  (8)

    with both fractions measured AFT from the local leading edge. Positive m
    winds the wing towards MORE incidence (more downforce, in the mirrored
    frame).

    The offset enters only as a difference of two fractions, so at
    ``x_attach_frac == x_ac_frac`` this is an exact array of zeros — not a
    small number. That is the OFF invariant of the whole torsion model, and it
    is why the default attachment fraction is the quarter chord.
    """
    y = np.asarray(y, dtype=float)
    load = np.asarray(load, dtype=float)
    chord = np.asarray(chord, dtype=float)
    if not (y.shape == load.shape == chord.shape):
        raise ValueError(
            f"y, load and chord must have the same shape (got {y.shape}, "
            f"{load.shape}, {chord.shape})")
    return load * chord * (float(x_attach_frac) - float(x_ac_frac))


def torsion_twist(y: np.ndarray, m_per_span: np.ndarray, y_station: float,
                  gj_nm2: float) -> np.ndarray:
    """Elastic twist phi(y) [rad] about the mount line, zero at the support.

    St-Venant torsion of a uniform shaft, GJ phi' = T, with the internal
    torque built exactly the way the shear is built for bending:

        T(y) = int_|y|^{y_max} m(e) de  -  T_R * [ |y| < y_s ]            (9)
        T_R  = int_0^{y_max} m(e) de

    T_R is not a free constant. The half wing's symmetry gives phi'(0) = 0,
    hence T(0) = 0, hence T_R is the whole half wing's torque — which is also
    what a moment balance on the half wing gives, so the two derivations
    agree. The support carries it: a mount that could not react torque would
    leave the wing's incidence undefined at every speed.

    BOUNDARY CONDITIONS, per support layout, all three of which fall out of
    (9) rather than being imposed:

        y_s = 0     phi(0) = 0 and T(b/2) = 0 — each half is a torsion
                    cantilever built in at the centreline with a free tip.
        y_s = b/2   phi(b/2) = 0 at both tips and phi'(0) = 0 at the
                    centreline — the fixed-fixed shaft.
        0 < y_s     both of the above at once: built in at the support, free
                    at the tip, symmetric at the centreline.

    Under a UNIFORM torque the closed forms are phi(tip) = m a^2 ... see
    :func:`uniform_twist`; the two published layouts happen to give the SAME
    peak twist, m (b/2)^2 / (2 GJ), and an interior mount gives less.

    The reaction is a STEP in T at the support, so the integration is done
    piecewise about y_s rather than by smearing the step over one panel —
    otherwise an interior mount would carry a discretisation error of order
    (panel width / semi-span) in a quantity whose whole point is that it is
    small.
    """
    order, ys, yh, mh = _halves(y, m_per_span)
    gj = float(gj_nm2)
    if not gj > 0.0:
        raise ValueError(f"gj_nm2 must be > 0 (got {gj_nm2})")
    y_s = float(y_station)
    # T_out(y) = int_y^{ymax} m de, the free-overhang torque
    t_out = np.empty_like(yh)
    for i in range(yh.size):
        t_out[i] = np.trapezoid(mh[i:], yh[i:])
    t_reaction = float(t_out[0])
    # Phi_out(y) = int_{yh[0]}^{y} T_out de
    phi_out = np.concatenate([[0.0], np.cumsum(
        0.5 * (t_out[1:] + t_out[:-1]) * np.diff(yh))])
    phi_at_station = float(np.interp(y_s, yh, phi_out))
    phi = phi_out - phi_at_station
    # inboard of the support the internal torque is lower by the reaction,
    # which integrates to an exact linear term rather than a smeared step
    inboard = yh < y_s
    phi = np.where(inboard, phi - t_reaction * (yh - y_s), phi) / gj
    return _mirror(order, ys, yh, phi)


def uniform_twist(m0: float, b: float, y_station: float,
                  gj_nm2: float) -> dict:
    """Closed-form twist of the uniformly loaded shaft, per support station.

    With m constant, L = b/2 and a = y_s, equation (9) integrates to

        phi(0) = m a^2 / (2 GJ)        the centre, relative to the support
        phi(L) = m (L - a)^2 / (2 GJ)  the tip, relative to the support

    — a pair of squares, which is the whole story of where to put a mount in
    torsion. Both published layouts sit at one end of it and give the same
    peak, m L^2 / (2 GJ): a torsion cantilever of length L and a fixed-fixed
    shaft of length 2L under the same distributed torque twist by exactly the
    same amount, m L^2/(2 GJ) = m b^2/(8 GJ), which is a genuine identity and
    not a coincidence of this discretisation.

    The peak, max(a, L - a)^2 m/(2 GJ), is minimised at a = L/2 and is then
    FOUR TIMES smaller than at either published end. That factor is the reason
    :func:`torsion_effective_length` is a function of the station at all.
    """
    L, a = 0.5 * float(b), float(y_station)
    gj = float(gj_nm2)
    if not gj > 0.0:
        raise ValueError(f"gj_nm2 must be > 0 (got {gj_nm2})")
    centre = float(m0) * a * a / (2.0 * gj)
    tip = float(m0) * (L - a) ** 2 / (2.0 * gj)
    return {"centre": centre, "tip": tip,
            "peak": float(max(abs(centre), abs(tip)))}


def torsion_effective_length(b: float, y_station: float) -> float:
    """Effective torsion-bar length [m] of the equivalent single-DOF section.

    DERIVED, by demanding that a single torsion spring K_theta = GJ / l_eff
    reproduce the DISTRIBUTED model's own peak twist under a uniform torque.
    The half wing carries a total torque m L (L = b/2) and twists by
    max(a, L - a)^2 m / (2 GJ) (:func:`uniform_twist`), so

        K_theta = (total torque) / (peak twist)
                = m L / [ max(a, L-a)^2 m / (2 GJ) ]
                = 2 GJ L / max(a, L-a)^2
        l_eff   = GJ / K_theta = max(a, L-a)^2 / (2 L)                   (10)

    Checks: at a = 0 or a = L this is L/2 = b/4, i.e. HALF the physical
    torsion length — the classical statement that a uniformly distributed
    torque is equivalent to a tip torque acting at the half length. At
    a = L/2 it is L/8, a four-fold stiffening, which is the same factor of
    four :func:`uniform_twist` shows directly.

    Everything downstream of this — q_div, the margin, the amplification — is
    the single-mode reduction of the distributed answer, and this is where the
    reduction happens. It is calibrated on a UNIFORM torque; a real load is
    not uniform, and a load concentrated further outboard than uniform makes
    l_eff larger (a softer wing) than this says. Bias direction: OPTIMISTIC.
    """
    L, a = 0.5 * float(b), float(y_station)
    if not L > 0.0:
        raise ValueError(f"b must be > 0 (got {b})")
    if not (0.0 <= a <= L + 1e-12):
        raise ValueError(f"y_station {a} is outside the semi-span [0, {L}]")
    return float(max(a, L - a) ** 2 / (2.0 * L))


def divergence_q(gj_nm2: float, b: float, s_ref: float, mac: float,
                 a_lin: float, e_frac: float, y_station: float) -> float:
    """Divergence dynamic pressure [Pa] of the equivalent single-DOF section.

    DERIVATION, in full, because the constant in front is where these formulas
    usually go wrong.

    Take the half wing as one torsion spring K_theta [N m/rad] about the mount
    line, twisting in one mode of amplitude theta. A twist theta adds a lift
    q c a theta per unit span (a = dCL/dalpha, per radian) acting at the
    aerodynamic centre, whose moment arm about the mount line is e_m = ehat c
    with ehat = (x_attach - x_ac)/c the DIMENSIONLESS offset. Integrating the
    added torque over the half wing with the twist held at theta,

        Delta T = q a ehat theta int_0^{b/2} c(y)^2 dy
                = q a ehat theta * (S/2) * MAC                           (11)

    because int c^2 dy / int c dy over the half span is the mean aerodynamic
    chord BY DEFINITION, and int c dy = S/2. So the reference chord in the
    reduction is not a choice: it is the MAC, exactly.

    Static equilibrium K_theta theta = Delta T_rigid + Delta T(theta) gives

        theta = theta_rigid / (1 - q / q_div),
        q_div = K_theta / ( a ehat (S/2) MAC )
              = GJ_eff / ( a S MAC ehat ),   GJ_eff = 2 K_theta = 2 GJ/l_eff
                                                                        (12)

    which is the form the specification asks for, with GJ_eff carrying the
    torsion length and the factor 2 that converts the half wing's area to the
    reference area. At the published stations l_eff = b/4 and (12) reads
    q_div = 8 GJ / (a S MAC ehat b).

    ``e_frac`` <= 0 returns ``inf``: a mount at or ahead of the aerodynamic
    centre has no divergence at any speed — the feedback is negative and the
    wing washes OUT with speed instead of winding up. That is a physical
    statement, not a guard against division by zero, and it is the reason the
    default attachment fraction is safe by construction.
    """
    e = float(e_frac)
    if e <= 0.0:
        return float("inf")
    l_eff = torsion_effective_length(b, y_station)
    gj_eff = 2.0 * float(gj_nm2) / l_eff
    denom = float(a_lin) * float(s_ref) * float(mac) * e
    if not denom > 0.0:
        raise ValueError(
            f"a_lin, s_ref and mac must all be > 0 (got {a_lin}, {s_ref}, "
            f"{mac})")
    return float(gj_eff / denom)


def divergence_margin(q: float, q_div: float) -> float:
    """``q_div / q - 1``: positive is safe, zero is divergence, in q.

    Stated as a RATIO minus one rather than a difference so it reads as a
    fraction of the operating point and carries the sign convention every
    other margin in this package uses (>= 0 feasible). Infinite where there is
    no divergence at all (mount at or ahead of the aerodynamic centre).
    """
    if not float(q) > 0.0:
        raise ValueError(f"q must be > 0 (got {q})")
    return float(float(q_div) / float(q) - 1.0)


def twist_amplification(q: float, q_div: float) -> float:
    """Aeroelastic amplification 1 / (1 - q/q_div) of the rigid-load twist.

    The single-mode result (12): the flexible twist is the twist the RIGID
    load would have produced, divided by (1 - q/q_div). Exactly 1 at q = 0 and
    exactly 1 as GJ -> infinity (q_div -> infinity), so a stiff wing recovers
    the rigid answer rather than approaching it; it grows without bound as
    q -> q_div from below.

    At or above q_div there is no static equilibrium. This returns ``inf``
    rather than a negative number, because a negative amplification is not a
    smaller twist — it is the formula being read past the point where it means
    anything, and a design that reached it has already failed.
    """
    qd = float(q_div)
    if not np.isfinite(qd):
        return 1.0
    ratio = float(q) / qd
    if ratio >= 1.0:
        return float("inf")
    return float(1.0 / (1.0 - ratio))


# ------------------------------------------------------ 6. the suction-side loss

def _loss_shape(y: np.ndarray, stations: np.ndarray, half_width_m: float,
                core_frac: float) -> np.ndarray:
    """f(y) in [0, 1]: how much of the peak loss lands at each point."""
    shape = np.zeros_like(np.asarray(y, dtype=float))
    for y_p in stations:
        s = np.abs(y - float(y_p)) / float(half_width_m)
        shape = np.maximum(shape, slipstream.axial_shape(s, core_frac))
    return shape


def station_widths(y: np.ndarray) -> np.ndarray:
    """Panel widths [m] that a bare station grid implies.

    Edges at the midpoints between neighbouring stations, and the outermost
    edge half a neighbour-spacing beyond the outermost station. That is not a
    convention picked here: it is the quadrature a POINT SAMPLE performs. A
    per-strip multiplier k(y_i) is applied to a strip load the solver already
    integrates as l_i * width_i, so the lift a point-sampled modifier removes
    is exactly sum (1 - k_i) l_i width_i — and when the caller has supplied no
    widths, these are the widths that sum runs on.

    At least two stations are required: one station implies no width at all,
    which is a caller error and not a candidate that failed.
    """
    y = np.sort(np.asarray(y, dtype=float).ravel())
    if y.size < 2:
        raise ValueError(
            f"a station grid needs at least two stations for its panel widths "
            f"to be defined (got {y.size})")
    edges = np.empty(y.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (y[1:] + y[:-1])
    edges[0] = y[0] - 0.5 * (y[1] - y[0])
    edges[-1] = y[-1] + 0.5 * (y[-1] - y[-2])
    return np.diff(edges)


def footprint_integral_error(y: np.ndarray, stations, half_width_m: float,
                             core_frac: float = 0.0,
                             width: np.ndarray | None = None) -> float:
    """Signed relative error of the grid's POINT-SAMPLED footprint integral.

    The quantity a per-strip knockdown actually removes is an INTEGRAL — the
    footprint's shape f (13) against the spanwise measure — so the only honest
    question to ask of a station grid is whether it recovers that integral::

        recovered = sum_i f(y_i) * width_i                                (15)
        exact     = int f(e) de   over the same span the panels cover
        error     = recovered / exact - 1

    with ``width`` the caller's own panel widths, or :func:`station_widths`'
    midpoint widths when none are given. ``exact`` is a midpoint quadrature of
    the SAME f at :data:`_QUAD_PER_HALF_WIDTH` sub-intervals per half-width,
    so this compares one quadrature of the shape with a far finer one and not
    the shape with a different model. It is integrated only over the
    footprints themselves — f is exactly zero outside them — and over their
    UNION, because (13) takes a max over pylons and an overlap must not be
    counted twice.

    COVERAGE IS NOT RESOLUTION, and the reference is clipped to the span the
    panels cover for exactly that reason: a footprint centred at the tip has
    half of itself hanging off the wing, where there is no lift to remove, and
    that is a fact about the layout rather than a failure of the grid. What is
    measured here is only whether the stations that DO cover the footprint
    integrate it.

    Returns 0.0 when there is nothing to recover: no stations, no width, or a
    footprint that lies entirely outside the panels.
    """
    y = np.asarray(y, dtype=float).ravel()
    stations = np.atleast_1d(np.asarray(stations, dtype=float)).ravel()
    hw = float(half_width_m)
    if stations.size == 0 or hw <= 0.0:
        return 0.0
    if width is None:
        y = np.sort(y)
        w = station_widths(y)
    else:
        w = np.asarray(width, dtype=float).ravel()
        if w.shape != y.shape:
            raise ValueError(
                f"width and y must have the same shape (got {w.shape} vs "
                f"{y.shape})")
    recovered = float(np.sum(_loss_shape(y, stations, hw, core_frac) * w))
    lo = float(np.min(y - 0.5 * w))
    hi = float(np.max(y + 0.5 * w))
    spans: list[list[float]] = []
    for y_p in np.sort(stations):
        a, c = max(float(y_p) - hw, lo), min(float(y_p) + hw, hi)
        if not c > a:
            continue
        if spans and a <= spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], c)
        else:
            spans.append([a, c])
    exact = 0.0
    for a, c in spans:
        # the max is a DEGENERACY guard and not an accuracy one: c > a is
        # already strict, so the ceiling is at least 1 unless the span is thin
        # enough to underflow the ratio, and a zero-interval quadrature would
        # be a division by zero rather than a small number. The accuracy is
        # set entirely by _QUAD_PER_HALF_WIDTH.
        n = max(int(np.ceil((c - a) * _QUAD_PER_HALF_WIDTH / hw)), 1)
        edges = np.linspace(a, c, n + 1)
        mid = 0.5 * (edges[1:] + edges[:-1])
        exact += float(np.sum(_loss_shape(mid, stations, hw, core_frac))
                       ) * (c - a) / n
    if not exact > 0.0:
        return 0.0
    return float(recovered / exact - 1.0)


def footprint_is_resolved(y: np.ndarray, stations, half_width_m: float,
                          core_frac: float = 0.0,
                          width: np.ndarray | None = None,
                          tol: float = FOOTPRINT_INTEGRAL_TOL) -> bool:
    """Does the grid recover the footprint's INTEGRAL, to ``tol``?

    A pylon's footprint is CENTIMETRES wide (14) and a car-wing VLM's centre
    panels are wider than that — cosine spacing clusters at the tips, so the
    coarsest panels sit exactly where a centre-mounted pylon does. Sampling
    the knockdown at panel midpoints therefore returns exactly 1.0 at every
    station for a footprint that falls between two of them, which is a live
    calibration reading as the OFF default. Measured on the published
    40-panel grid: the centre panels are 0.063 m wide against a 0.043 m
    footprint, and the point-sampled knockdown of a centre mount is 0.0000%
    at ANY magnitude.

    So this is not a nicety. Either the caller supplies panel widths and the
    loss is averaged over each panel (which preserves the integral whatever
    the grid), or the footprint has to be resolved by the grid, and this says
    which case it is — by MEASURING the recovery
    (:func:`footprint_integral_error`) and comparing it against ``tol``, whose
    default and its derivation are :data:`FOOTPRINT_INTEGRAL_TOL`.

    WHAT THIS USED TO TEST, and why it was a different question. The criterion
    was ``count_nonzero(|y - y_p| < half_width) >= 2``: a proximity COUNT with
    an untested threshold. It is not a resolution test, because the raised
    cosine of (13) tapers to zero at the edge of the footprint, so two
    stations can sit inside the band and still sample almost none of it.
    Measured over 2400 uniform grids (panel width 0.05-3 half-widths, 40
    phases each) on the published pylon's footprint: the count passed 1080 of
    them, and among those the recovered integral ran from -98% to +28%, with
    28% of the passing grids wrong by more than 10%. The single worst was a
    grid of panel width 1.96 half-widths phased so its only two stations
    inside the footprint land at +/- 0.98 half-widths — both on the taper's
    outer skirt, where f is 0.002 — which the count passed and which recovers
    0.7% of the loss. :meth:`MountSpec.modifiers` handed those back with no
    raise, i.e. the guard's whole purpose was defeated in exactly the regime
    it exists for. Those numbers are re-derived by tests/test_carmount.py,
    which keeps the old criterion runnable rather than describing it.

    ``core_frac`` is the shape's own flat-core fraction and belongs here for
    the same reason it belongs in (13): the integral being recovered is the
    integral of the shape that will actually be applied, not of a nearby one.
    """
    return bool(abs(footprint_integral_error(
        y, stations, half_width_m, core_frac=core_frac,
        width=width)) <= float(tol))


def suction_loss_modifiers(y: np.ndarray, stations, half_width_m: float,
                           magnitude: float, core_frac: float = 0.0,
                           width: np.ndarray | None = None) -> np.ndarray:
    """Per-strip LIFT multipliers over the pylons' footprint.

    Returns one multiplier per station in ``y``, in (0, 1]::

        k(y) = 1 - magnitude * max_p f( |y - y_p| / half_width )         (13)

    with ``f`` slipstream.axial_shape: flat at 1 over the core, then a
    cos^2 taper to exactly 0 at the edge of the footprint. Reused rather than
    re-derived, and for the same reason slipstream uses it — it is smooth,
    bounded, and reaches its ends exactly.

    The MAX over pylons, never the sum. Two footprints that overlap do not
    remove twice the lift, any more than two overlapping propeller wakes
    superpose their dynamic pressure (slipstream.slipstream_profiles takes the
    same max, for the same reason).

    ``width`` — the panel widths at the same stations — switches (13) from a
    POINT SAMPLE at the station to the panel AVERAGE of k over each panel.
    Pass it. A pylon's footprint is centimetres wide and a car wing's centre
    panels are wider (:func:`footprint_is_resolved`), so the point sample of a
    live loss can be identically 1.0; the panel average removes the right
    amount of lift on any grid, because it integrates the footprint rather
    than probing it. The average is a midpoint-rule quadrature whose
    sub-step is derived from the footprint width, not chosen.

    OFF INVARIANT: at ``magnitude`` = 0.0 every returned multiplier is
    EXACTLY 1.0, bit for bit — 0.0 * f is an exact zero for the bounded f
    above, and 1.0 - 0.0 is exactly 1.0. It holds on BOTH paths, because a
    panel entirely outside the footprint averages an exact array of zeros.
    Asserted with ``==`` in tests/test_carmount.py, not with a tolerance,
    because the whole point of a default-zero calibration is that a run which
    does not set it is the run that was published.
    """
    y = np.asarray(y, dtype=float)
    mag = float(magnitude)
    if not (0.0 <= mag < 1.0):
        raise ValueError(
            f"suction loss magnitude must be in [0, 1) (got {magnitude})")
    stations = np.atleast_1d(np.asarray(stations, dtype=float)).ravel()
    w = float(half_width_m)
    if stations.size == 0 or w <= 0.0 or mag == 0.0:
        return np.ones_like(y)
    if width is None:
        return 1.0 - mag * _loss_shape(y, stations, w, core_frac)
    width = np.asarray(width, dtype=float)
    if width.shape != y.shape:
        raise ValueError(
            f"width and y must have the same shape (got {width.shape} vs "
            f"{y.shape})")
    # sub-step derived from the footprint, so the quadrature resolves the
    # narrowest feature there is rather than a count somebody picked: at least
    # four sub-points across a half-width, an odd count so the panel's own
    # midpoint is one of them, and capped because a degenerate panel must not
    # turn a modifier into an unbounded loop.
    n_sub = int(np.clip(
        2 * int(np.ceil(2.0 * float(np.max(width)) / w)) + 1, 9, 2001))
    frac = (np.arange(n_sub) + 0.5) / n_sub - 0.5           # midpoint rule
    sub = y[:, None] + width[:, None] * frac[None, :]
    shape = _loss_shape(sub, stations, w, core_frac).mean(axis=1)
    return 1.0 - mag * shape


def footprint_half_width_m(pylon_chord_m: float, pylon_tc: float,
                           spread: float = DEFAULT_LOSS_SPREAD) -> float:
    """Half-width [m] of one pylon's footprint on the wing.

    DERIVED from the pylon's own thickness and one stated spread::

        t_pylon = pylon_tc * pylon_chord_m
        half_width = 0.5 * spread * t_pylon                              (14)

    so the disturbed band is ``spread`` pylon-thicknesses wide in total, and
    the flat-topped core of :func:`suction_loss_modifiers` is exactly the
    pylon's own thickness when ``core_frac`` is set to 1/spread — which is
    what :meth:`MountSpec.modifiers` does, rather than stating a second
    number. The pylon fully blankets what it covers and tapers over the rest.

    ``spread`` is the module's only free shape parameter here and it is a
    MODEL CHOICE (:data:`DEFAULT_LOSS_SPREAD`), not a measurement. At the
    published pylon (chord 0.12 m, t/c 0.12) it makes a band 0.043 m wide on a
    1.6 m span — 2.7% of the span for a twin-pylon centre mount.
    """
    spread = float(spread)
    if spread < 1.0:
        raise ValueError(
            f"spread must be >= 1 — the disturbed band cannot be narrower "
            f"than the pylon that causes it (got {spread})")
    return float(0.5 * spread * float(pylon_tc) * float(pylon_chord_m))


def lift_knockdown(load: np.ndarray, width: np.ndarray,
                   modifiers: np.ndarray) -> float:
    """Fraction of the span-integrated lift the footprint removes [-].

    ``1 - sum(k l dy) / sum(l dy)`` — the first-order reading of a per-strip
    multiplier, and the only one available without re-solving: applying k to
    the strip loads changes the trailing wake, and the circulation would
    redistribute. That redistribution recovers some of the loss (a wing does
    not simply lose the strip), so this is an UPPER bound on the knockdown.
    Bias direction: CONSERVATIVE.
    """
    load = np.asarray(load, dtype=float)
    width = np.asarray(width, dtype=float)
    mods = np.asarray(modifiers, dtype=float)
    if not (load.shape == width.shape == mods.shape):
        raise ValueError("load, width and modifiers must have the same shape")
    total = float(np.sum(load * width))
    if total == 0.0:
        return 0.0
    return float(1.0 - np.sum(mods * load * width) / total)


# ------------------------------------------------------------------ the spec

@dataclass
class MountSpec:
    """One rear-wing mount, stated once.

    The station is asked for in EXACTLY ONE of two ways, and both exist
    because they are two different physical statements:

        ``y_station_m``   a LENGTH from the centreline. What a chassis
                          hardpoint is: a distance between two pieces of car,
                          which does not move when the wing's span does.
        ``station_frac``  a fraction of the semi-span. What "at the tips"
                          means: a statement that TRACKS the span, which is
                          the only way to express the endplate-mounted layout
                          while the span is a design variable.

    Setting both, or neither, is refused at construction rather than resolved
    by a precedence rule, because a precedence rule would let one of the two
    be silently ignored.

    The defaults are the OFF configuration in every sense that matters — they
    are chosen so that adopting this module moves nothing:

        ``x_attach_frac`` = 0.25 = :data:`X_AC_FRAC`, so the torsional moment
        is an exact zero and the twist an exact zero array;
        ``suction_loss`` = 0.0, so the lift modifiers are exactly 1.0;
        ``pylon_drag_model`` = ``"flat_plate"`` with ``pylon_cf`` = 0.005, so
        the pylon charge is bit-for-bit carwing._strut_cd0's.

    The one thing that does NOT default to the published behaviour is the
    pylon LENGTH, which is ride - deck here and the whole ride height in
    carwing.py. That is a correction, not a switch, and it is measured rather
    than argued (:func:`pylon_length_m`).
    """

    kind: str = "pylon"
    #: the station, as a length. Exactly one of this and ``station_frac``.
    y_station_m: float | None = None
    #: the station, as a fraction of the semi-span. Exactly one of this and
    #: ``y_station_m``.
    station_frac: float | None = None
    #: chordwise attachment, as a fraction of the LOCAL chord, aft from the
    #: leading edge. 0.25 is the aerodynamic centre and gives exactly no
    #: windup; aft of it the wing winds up towards more downforce and can
    #: diverge; ahead of it the wing sheds incidence with speed.
    x_attach_frac: float = X_AC_FRAC
    n_pylons: int = 2
    #: EXTRA SHEETS this mount adds, beyond the endplates the wing carries
    #: anyway. An ``endplate`` mount at the TIP adds none — the plates are
    #: there for the nonplanar benefit and the load path is free — but a grip
    #: INBOARD of the tip is a second pair of plates, which is wetted area
    #: and two more wing corners. Without this field a MountSpec priced only
    #: pylons, and since neither published layout has any, the continuum path
    #: handed the inboard grip its stiffness for FREE: measured at the box
    #: centre, ``car_mount_model='continuum'`` with ``mount='inboard'``
    #: reported CD 0.018046 and score 38.283 — bit-identical to the tip-borne
    #: wing — while keeping 0.1012 mm of deflection against its 0.4574 mm.
    #: The optimiser takes a free lunch every time it is offered one.
    n_sheets: int = 0
    #: ...and the height each of those sheets stands, [m]. ``None`` means "the
    #: endplate's own height", which is what the caller passes at evaluation
    #: time; it is a field only so a spec can state a different one.
    sheet_height_m: float | None = None
    pylon_chord_m: float = 0.12
    #: pylon thickness ratio. 0.12 is the wing section's own default in this
    #: family, repeated here rather than invented, and it sits inside
    #: junction.TC_VALID (0.05-0.20) so the corner correlation is being read
    #: inside its band.
    pylon_tc: float = 0.12
    #: how the pylon's parasite drag is charged: ``"buildup"`` (7) or
    #: ``"flat_plate"`` (carwing's crude model, the default so that adoption
    #: is bit-for-bit).
    pylon_drag_model: str = "flat_plate"
    pylon_cf: float = 0.005
    #: the car's attachment deck above the track [m]. endplate.py's own
    #: default, so the two car families measure the same car.
    deck_height_m: float = 0.25
    #: torsional stiffness GJ [N m^2]. Defaults to carwing's own EI rather
    #: than to a second, unstated number: for a thin-walled closed section GJ
    #: and EI are the same order of magnitude, and their ratio is a section
    #: property this package does not model. A caller who knows the section
    #: should state it.
    gj_nm2: float = 4.0e4
    #: which side of the wing the mount reaches it on (:data:`MOUNT_SIDES`).
    side: str = "suction"
    #: peak per-strip LIFT loss under a pylon, as a fraction. A CALIBRATION
    #: THE USER OWNS, defaulting to exactly zero. This package cannot produce
    #: it: it is a chordwise, viscous, geometry-specific effect and no
    #: lifting-surface method or reduced-order correlation here gives it
    #: honestly. Set it from CFD or tunnel data, or leave it off.
    suction_loss: float = 0.0
    #: footprint width, in pylon thicknesses (:data:`DEFAULT_LOSS_SPREAD`).
    loss_spread: float = DEFAULT_LOSS_SPREAD

    def __post_init__(self):
        if self.kind not in MOUNT_KINDS:
            raise ValueError(
                f"unknown mount kind {self.kind!r}; "
                f"choose from {sorted(MOUNT_KINDS)}")
        if self.side not in MOUNT_SIDES:
            raise ValueError(
                f"unknown mount side {self.side!r}; "
                f"choose from {sorted(MOUNT_SIDES)}")
        stated = [n for n in ("y_station_m", "station_frac")
                  if getattr(self, n) is not None]
        if len(stated) != 1:
            raise ValueError(
                "state the mount station exactly once: y_station_m (a length "
                "from the centreline, what a chassis hardpoint is) OR "
                "station_frac (a fraction of the semi-span, which is how "
                "'at the tips' tracks a span that is itself a design "
                f"variable). Got {stated or 'neither'}")
        if self.y_station_m is not None:
            self.y_station_m = float(self.y_station_m)
            if self.y_station_m < 0.0:
                raise ValueError(
                    f"y_station_m must be >= 0 (got {self.y_station_m})")
        else:
            self.station_frac = float(self.station_frac)
            if not (0.0 <= self.station_frac <= 1.0):
                raise ValueError(
                    f"station_frac must be in [0, 1] — a support outboard of "
                    f"the tip is not a mount (got {self.station_frac})")
        if not (0.0 <= float(self.x_attach_frac) <= 1.0):
            raise ValueError(
                f"x_attach_frac must be in [0, 1] (aft from the leading edge, "
                f"as a fraction of the local chord); got {self.x_attach_frac}")
        self.n_pylons = int(self.n_pylons)
        if self.n_pylons < 0:
            raise ValueError(f"n_pylons must be >= 0 (got {self.n_pylons})")
        if self.carries_pylons and self.n_pylons == 0:
            raise ValueError(
                "a 'pylon' mount with no pylons carries nothing; use "
                "kind='endplate' if the plates are the load path")
        self.n_sheets = int(self.n_sheets)
        if self.n_sheets < 0:
            raise ValueError(f"n_sheets must be >= 0 (got {self.n_sheets})")
        if self.carries_pylons and self.n_sheets:
            raise ValueError(
                "a 'pylon' mount carries its load on pylons; n_sheets "
                "describes the EXTRA PLATES an endplate mount adds, and "
                "charging both would price one load path twice")
        if not self.carries_pylons and self.n_pylons != 0:
            raise ValueError(
                f"an 'endplate' mount has no pylons to charge, got "
                f"n_pylons={self.n_pylons}")
        if self.n_pylons and not float(self.pylon_chord_m) > 0.0:
            raise ValueError(
                f"pylon_chord_m must be > 0 (got {self.pylon_chord_m})")
        if not (0.0 < float(self.pylon_tc) < 1.0):
            raise ValueError(
                f"pylon_tc must be in (0, 1) (got {self.pylon_tc})")
        if self.pylon_drag_model not in ("buildup", "flat_plate"):
            raise ValueError(
                f"unknown pylon_drag_model {self.pylon_drag_model!r}; "
                f"choose from ('buildup', 'flat_plate')")
        if not float(self.deck_height_m) > 0.0:
            raise ValueError(
                f"deck_height_m must be > 0 (got {self.deck_height_m})")
        if not float(self.gj_nm2) > 0.0:
            raise ValueError(f"gj_nm2 must be > 0 (got {self.gj_nm2})")
        if not (0.0 <= float(self.suction_loss) < 1.0):
            raise ValueError(
                f"suction_loss is a fraction of the local lift and must be in "
                f"[0, 1) (got {self.suction_loss})")
        if float(self.loss_spread) < 1.0:
            raise ValueError(
                f"loss_spread must be >= 1 — the disturbed band cannot be "
                f"narrower than the pylon (got {self.loss_spread})")

    # ---------------------------------------------------------- derived views

    @property
    def kind_spec(self) -> dict:
        return MOUNT_KINDS[self.kind]

    @property
    def side_spec(self) -> dict:
        return MOUNT_SIDES[self.side]

    @property
    def carries_pylons(self) -> bool:
        return bool(MOUNT_KINDS[self.kind]["carries_pylons"])

    @property
    def n_junctions(self) -> int:
        """New wing corners the mount introduces.

        DERIVED, one per piece of structure that meets the wing: one per
        PYLON on a pylon mount, and one per EXTRA SHEET on an endplate mount.
        A tip-borne endplate mount adds none, because the plates it uses are
        there anyway; an inboard grip adds one corner per sheet, exactly as a
        pylon does. That second half was missing, and its absence was not
        cosmetic — see :attr:`n_sheets`.
        """
        return int(self.n_pylons) if self.carries_pylons else int(self.n_sheets)

    @property
    def e_frac(self) -> float:
        """Elastic-axis offset ehat = (x_attach - x_ac)/c [-], signed.

        Positive is a mount AFT of the aerodynamic centre: the divergence-
        prone sense.
        """
        return float(self.x_attach_frac) - X_AC_FRAC

    def station_m(self, b: float) -> float:
        """The support station [m] for a wing of span ``b``."""
        if self.y_station_m is not None:
            return float(self.y_station_m)
        return float(self.station_frac) * 0.5 * float(b)

    def pylon_length_m(self, ride_height_m: float) -> float:
        """Length of one pylon [m] at this ride height (6)."""
        return pylon_length_m(ride_height_m, self.deck_height_m)

    def footprint_half_width_m(self) -> float:
        """Half-width of one pylon's footprint on the wing [m] (14)."""
        if not self.n_pylons:
            return 0.0
        return footprint_half_width_m(self.pylon_chord_m, self.pylon_tc,
                                      self.loss_spread)

    def support_stations_m(self, b: float) -> np.ndarray:
        """Spanwise stations the mount actually HOLDS the wing at [m].

        The pair +/- y_s, or the single centreline station when y_s is zero
        — the same symmetry :func:`beam_moment` is written on, and therefore
        the only places a picture may put a strut.

        NOT :meth:`footprint_stations_m`, which is the SUCTION-side loss
        footprint: that one is deliberately empty on a swan neck (nothing is
        lost, so nothing is marked) and on an endplate mount (no footprint at
        all). A drawing keyed on it would show a swan-neck wing floating with
        no visible support, which is exactly the layout most worth seeing.

        ``n_pylons`` is not a station count — it counts WETTED MEMBERS, and a
        twin-pylon mount is two struts a few centimetres apart that the beam
        sees as one place. So this returns places, and the member count is a
        caption.
        """
        y_s = self.station_m(b)
        if y_s == 0.0:
            return np.zeros(1, dtype=float)
        return np.array([-y_s, y_s], dtype=float)

    def footprint_stations_m(self, b: float) -> np.ndarray:
        """Spanwise centres of the footprints [m], symmetric about y = 0.

        The mount is symmetric — that is what makes the beam a half-span
        problem — so the footprints are the pair +/- y_s, or the single
        centreline band when y_s is zero and the pylons stack there.
        ``n_pylons`` counts WETTED MEMBERS, not stations: a twin-pylon centre
        mount is two struts a few centimetres apart, which both the beam and
        this footprint see as one place. Pylons at a station other than the
        beam's supports are in the module docstring's not-modelled list.
        """
        if not (self.n_pylons and self.side_spec["loses_lift"]):
            return np.zeros(0, dtype=float)
        y_s = self.station_m(b)
        if y_s == 0.0:
            return np.zeros(1, dtype=float)
        return np.array([-y_s, y_s], dtype=float)

    # ------------------------------------------------------------- contracts

    def violation(self, b: float, ride_height_m: float) -> str | None:
        """Why this mount cannot be built on this candidate, or ``None``.

        The package's in-contract failure shape: a reason string, never an
        exception. Both of these are designs an optimiser can propose from
        inside its own box, not programming errors — the station is checked
        against a span that moves, and the pylon length against a ride height
        that moves.
        """
        semi = 0.5 * float(b)
        y_s = self.station_m(b)
        if y_s > semi + 1e-12:
            return (f"the mount station ({y_s:.4g} m from the centreline) is "
                    f"outboard of the tip ({semi:.4g} m): there is no wing "
                    f"there to bolt to")
        if self.carries_pylons:
            length = self.pylon_length_m(ride_height_m)
            if not length > 0.0:
                return (f"the wing sits {float(ride_height_m):.4g} m above the "
                        f"track and its attachment deck is at "
                        f"{self.deck_height_m:.4g} m, so the pylon would have "
                        f"to be {length:.4g} m long: raise the wing or lower "
                        f"the deck")
        return None

    # ------------------------------------------------------------- the pieces

    def beam(self, y, load, b, ei_nm2: float | None = None) -> dict:
        """The mount's beam answer (:func:`beam_report`) at its own station."""
        return beam_report(y, load, self.station_m(b), ei_nm2=ei_nm2)

    def pylon_cd(self, ride_height_m: float, s_ref: float,
                 rho: float, V: float, mu: float,
                 sheet_height_m: float | None = None,
                 sheet_chord_m: float | None = None) -> dict:
        """The mount's own parasite drag — its PYLONS or its EXTRA SHEETS.

        One method because it answers one question: what does this mount cost
        in wetted area that the wing was not already paying for. A pylon mount
        charges its struts over their length off the deck; an endplate mount
        charges the sheets it ADDS (never the plates the wing carries anyway),
        over the height they stand and the chord where they grip.

        ``sheet_height_m`` / ``sheet_chord_m`` come from the candidate being
        flown, because both do: the plate's height is a design variable and
        the chord at the grip station moves with the taper. Falls back to
        :attr:`sheet_height_m` and the pylon chord when the caller has
        neither, so a spec can still be priced on its own.
        """
        if self.carries_pylons:
            return pylon_parasite_cd(
                self.n_pylons, self.pylon_length_m(ride_height_m),
                self.pylon_chord_m, self.pylon_tc, s_ref, rho, V, mu,
                model=self.pylon_drag_model, cf=self.pylon_cf)
        if not self.n_sheets:
            return pylon_parasite_cd(0, 0.0, self.pylon_chord_m,
                                     self.pylon_tc, s_ref, rho, V, mu,
                                     model=self.pylon_drag_model,
                                     cf=self.pylon_cf)
        height = (float(sheet_height_m) if sheet_height_m is not None
                  else float(self.sheet_height_m or 0.0))
        chord = (float(sheet_chord_m) if sheet_chord_m is not None
                 else float(self.pylon_chord_m))
        return pylon_parasite_cd(
            self.n_sheets, height, chord, self.pylon_tc, s_ref, rho, V, mu,
            model=self.pylon_drag_model, cf=self.pylon_cf)

    def junction_cd(self, wing: "geometry.Wing", s_ref: float,
                    tc: float | None = None, radius: float = 0.0) -> float:
        """The mount's junction drag, at the chord where it meets the wing."""
        return mount_junction_cd(wing, self.station_m(wing.b), s_ref,
                                 self.n_junctions, tc=tc, radius=radius)

    def torsion(self, y, load, chord, b, s_ref, mac, q,
                a_lin: float = 2.0 * np.pi) -> dict:
        """The mount's torsion answer: distribution, divergence and margin.

        ``phi_rigid`` is the twist the SOLVED (rigid) load produces;
        ``phi`` is that distribution scaled by the single-mode amplification
        1/(1 - q/q_div), which is where the feedback enters. One mode, one
        degree of freedom, no camber term and no bending coupling — see the
        module docstring.
        """
        y_s = self.station_m(b)
        m = torsion_moment(y, load, chord, self.x_attach_frac)
        phi_rigid = torsion_twist(y, m, y_s, self.gj_nm2)
        q_div = divergence_q(self.gj_nm2, b, s_ref, mac, a_lin,
                             self.e_frac, y_s)
        amp = twist_amplification(q, q_div)
        phi = phi_rigid * amp if np.isfinite(amp) else np.full_like(
            np.asarray(phi_rigid, dtype=float), np.inf)
        return {
            "m_per_span_Nm_per_m": m,
            "phi_rigid_rad": phi_rigid,
            "phi_rad": phi,
            "phi_rigid_max_deg": float(np.rad2deg(np.max(np.abs(phi_rigid)))),
            "phi_max_deg": float(np.rad2deg(np.max(np.abs(phi)))),
            "e_frac": self.e_frac,
            "l_eff_m": torsion_effective_length(b, y_s),
            "q_div_Pa": q_div,
            "q_Pa": float(q),
            "g_divergence": divergence_margin(q, q_div),
            "amplification": amp,
            "x_attach_frac": float(self.x_attach_frac),
            "x_ac_frac": X_AC_FRAC,
        }

    def modifiers(self, y, c, b: float | None = None,
                  width: np.ndarray | None = None) -> np.ndarray:
        """Per-station LIFT multipliers over the mount's footprint.

        slipstream.SlipstreamSpec.modifiers' shape and its OFF invariant:
        with ``suction_loss`` at 0.0 — or with the mount on the pressure side,
        or with no pylons at all — every returned multiplier is EXACTLY 1.0.

        ``c`` is the per-station chord at the same stations as ``y``. It is
        validated for alignment and is not otherwise used: the footprint's
        width comes from the PYLON's geometry, not the wing's, which is the
        one place this differs from the slipstream hook. It is required
        anyway, so that a caller who later needs a chordwise term does not
        have to change every call site. ``b`` is needed only to resolve a
        station stated as a fraction of the semi-span.

        ``width`` is the panel width at each station, and it is REQUIRED
        whenever the loss is live on a grid that does not resolve the
        footprint. What "resolve" means is MEASURED, not counted: the grid has
        to recover the footprint's own integral to
        :data:`FOOTPRINT_INTEGRAL_TOL` (:func:`footprint_integral_error`), and
        the reason it fires is quoted back in the message as the recovery it
        measured. That refusal is deliberate and it is not a preference: the
        worst case is a modifier that returns exactly 1.0 for a magnitude the
        user set — a calibration silently reading as its own OFF default — and
        the general case is a knockdown scaled by an arbitrary factor that
        depends on nothing but where the pylon fell between two stations. The
        caller already has the widths — every solve in this package carries
        them beside its stations — so this costs an argument, not a
        measurement.
        """
        y = np.asarray(y, dtype=float).ravel()
        c = np.asarray(c, dtype=float).ravel()
        if c.shape != y.shape:
            raise ValueError(
                f"c and y must have the same length (got {c.shape} vs "
                f"{y.shape})")
        if self.y_station_m is None and b is None:
            raise ValueError(
                "this mount states its station as a fraction of the "
                "semi-span, so the span b is needed to place its footprint")
        span = float(b) if b is not None else 0.0
        stations = self.footprint_stations_m(span)
        half_w = self.footprint_half_width_m()
        core = 1.0 / float(self.loss_spread)
        if self.suction_loss > 0.0 and stations.size and width is None:
            err = footprint_integral_error(y, stations, half_w,
                                           core_frac=core)
            if abs(err) > FOOTPRINT_INTEGRAL_TOL:
                raise ValueError(
                    f"this mount's footprint is {2.0 * half_w:.4g} m wide and "
                    f"the station grid does not resolve it: point-sampled, it "
                    f"recovers {1.0 + err:.1%} of the footprint's own integral "
                    f"({err:+.1%} error, against a tolerance of "
                    f"{FOOTPRINT_INTEGRAL_TOL:.0%}), so a knockdown of "
                    f"{self.suction_loss:g} would come back "
                    + ("as exactly no loss at all" if err <= -1.0 + 1e-12
                       else f"scaled by {1.0 + err:.2f}")
                    + ". Pass the panel widths (width=...) so the loss is "
                      "averaged over each panel")
        if width is not None:
            width = np.asarray(width, dtype=float).ravel()
        return suction_loss_modifiers(
            y, stations, half_w, self.suction_loss,
            core_frac=core, width=width)


#: where the inboard layout's sheets grip, as a fraction of the SEMI-SPAN.
#: Stated here rather than in carwing.py because this module owns the station
#: vocabulary; carwing imports it, so the two dicts cannot drift apart.
INBOARD_STATION_FRAC = 0.35

#: carwing.MOUNTS restated in this module's vocabulary, so the older family
#: can adopt :class:`MountSpec` without either layout moving.
#:
#: Every field here is the published behaviour, including the two that are
#: really defaults rather than statements: ``x_attach_frac`` at the quarter
#: chord (carwing has no chordwise attachment point, and a mount at the
#: aerodynamic centre is the only assumption under which that silence is
#: correct) and ``suction_loss`` at zero (carwing's docstring says explicitly
#: that the effect is "left out rather than invented"). The deck height is the
#: one genuinely new number: carwing has no deck at all, which is exactly why
#: it charges its pylons over the whole ride height.
#: BOTH published layouts are ``kind="endplate"`` now: a rear wing is bolted
#: to the car by the sheets it already carries, and carwing.MOUNTS no longer
#: offers a pylon at all. So what these two differ in is only the STATION,
#: which is exactly the axis this module was written to resolve — the flag
#: became a fraction of the semi-span, and the beam that reads it is the
#: signed one below rather than a two-valued end-or-centre branch.
#:
#: ``kind="pylon"`` is NOT deleted. carwing's published families do not offer
#: it, but a caller who wants to price a swan neck can still build the spec
#: by hand, and every pylon term in this module (length off the deck, the
#: wetted build-up at its own Reynolds number, the footprint knockdown, the
#: suction-side loss) is still reachable and still tested. What changed is
#: which layouts a registered problem hands you, not what this module can
#: describe.
PUBLISHED_LAYOUTS: dict[str, MountSpec] = {
    "tips": MountSpec(kind="endplate", station_frac=1.0, n_pylons=0),
    "inboard": MountSpec(kind="endplate", station_frac=INBOARD_STATION_FRAC,
                         n_pylons=0, n_sheets=2),
}


def published_layout(name: str) -> MountSpec:
    """A fresh :class:`MountSpec` for one of carwing.MOUNTS' two layouts.

    Returned by value rather than shared, so a caller that edits the spec it
    was handed cannot move the published layout under the next caller.
    """
    from dataclasses import replace
    if name not in PUBLISHED_LAYOUTS:
        raise ValueError(
            f"unknown published layout {name!r}; "
            f"choose from {sorted(PUBLISHED_LAYOUTS)}")
    return replace(PUBLISHED_LAYOUTS[name])
