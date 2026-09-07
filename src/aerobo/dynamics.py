"""Stability and control derivatives from the lattice that was already flown.

The point of this module is that it does NOT build a second aerodynamic model.
:class:`vlm.VLM` factorises its influence matrix once (``self._lu``) and every
answer it gives is a back-substitution against those factors. A body rate, a
sideslip and a control deflection are each nothing more than ONE MORE
RIGHT-HAND SIDE on that same system, so the whole derivative deck costs a
handful of back-substitutions — measured at 0.41 ms for five columns on an
80-panel wing-and-tail, against 5.0 ms to build the lattice in the first place.

What each column is
-------------------
The flow-tangency condition is written at the control points as
``(V_local . n) = 0``. The freestream part is what :class:`vlm.VLM` already
solves for; a disturbance adds a known extra wind ``u`` at each control point
and therefore a known extra right-hand side ``-u . n``:

===============  ==========================================================
disturbance      extra wind seen at the control point
===============  ==========================================================
alpha            ``+V alpha z_hat``   (freestream tilts up; x is AFT here)
beta             ``-V beta  y_hat``   (the aircraft slides to starboard, so
                                       the flow slides to port)
body rate Omega  ``-(Omega x r)``     (r from the CG to the control point)
===============  ==========================================================

The alpha column produced this way is bit-for-bit ``VLM._Gam1`` — the lattice's
own lift-curve basis — which is the cheapest possible check that this module is
riding the same boundary condition as the solver, and it is gated in the tests.

Axes: THREE frames, and only one conversion
-------------------------------------------
``vlm.py`` works in **x AFT, y starboard, z UP**. Flight dynamics wants
**x FORWARD, y starboard, z DOWN**. The map between them is

    T = diag(-1, +1, -1)

which is a rotation by pi about y: ``det T = +1``, so it is a PROPER rotation
and angular velocities and moments transform the same way positions do (no
pseudovector sign to get wrong). Everything in this module is computed in the
lattice's frame and converted exactly once, in :func:`to_body`. Doing it
anywhere else is how a fin ends up on the nose.

Note which derivatives are INVARIANT under that map, because it is a useful
sanity rail rather than a coincidence: ``Cl_p``, ``Cm_q``, ``Cn_r``, ``Cl_r``
and ``Cn_p`` all divide a sign-flipped moment by a sign-flipped rate. What DOES
flip is anything pairing a flipped output with an unflipped input, e.g.
``Cn_beta`` and ``Cl_beta``.

The force, and why it needs a base state
----------------------------------------
The Kutta-Joukowski force crosses the circulation with the LOCAL velocity,

    V_local = V_inf + u_disturbance + w_induced

not with the freestream alone. The freestream-only force is exact for lift and
useless for two derivatives: ``Cl_r`` lives in the differential dynamic
pressure a yaw rate puts across the span, and ``Cn_p`` lives in the tilt of
the force vector by the induced field — adverse yaw is induced drag being
asymmetric. Neither survives a cross product with a uniform freestream, and
with it both came out at exactly zero.

TWO TERMS, AND THEY HAVE VERY DIFFERENT CREDIBILITY.

``u_disturbance`` is exact. It is the wind the body's own motion puts at each
panel, known in closed form, and it alone supplies both new derivatives:
``Cl_r`` from the differential dynamic pressure (verified against a closed
form to a ratio of 1.00000) and ``Cn_p`` from the tilt of the force vector by
the roll-rate flow — which is the mechanism the classical ``-CL/8`` estimate
describes. It is ON by default.

``w_induced`` (:func:`induced_velocity`) is the lifting-line closure: half the
Trefftz wash, along the panel normal. Its TOTAL is exact — near-field drag
built from it reproduces the published Trefftz ``CDi`` to a ratio of
1.000000 — but its SPANWISE DISTRIBUTION is not, and that is the only thing
it would be used for. Measured on a rectangular wing, whose loading is close
to elliptic and whose induced angle should therefore be nearly uniform, it
varies by 85 % across the inner span, and on a taper-0.4 wing it is not even
monotonic. This is the near-field unreliability :class:`vlm.VLM` warns about
in its own docstring: one chordwise panel cannot resolve where the drag acts,
only how much there is in total.

So ``induced_tilt`` is OFF by default. Turning it on moves ``Cn_p`` by 22 %
and does not fix what it was wanted for — see "adverse yaw" below.

ADVERSE YAW IS NOT MODELLED, and this is a measured negative result rather
than an omission. A deflected aileron in this model yaws PROVERSELY
(``Cn_da < 0`` for ``Cl_da < 0``), because the only yaw it can produce is the
body-axis forward lean of the extra lift vector. Real adverse yaw is the
induced-drag asymmetry between the two wings, and that needs the spanwise
drag distribution measured unreliable above. Enabling ``induced_tilt``
reduces the proverse answer by 43 % and does not change its sign. A model
that reported adverse yaw here would be reporting a number it cannot
compute.

Both new terms are BILINEAR: perturbation circulation against base induced
field, and base circulation against perturbation field. So the deck is no
longer scale-free — it must be linearised about a state, and ``Cl_r`` and
``Cn_p`` are proportional to the base ``CL``. **At ``alpha = 0`` they are
still zero, and that is now a statement about an unloaded wing rather than
about the force model.** Pass ``deck(..., alpha=...)`` at the flight
condition, or those two derivatives will silently be absent.

``local_velocity=False`` restores the freestream-only force exactly, and is
what :meth:`vlm.VLM.neutral_point`'s lift-weighted station corresponds to.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from scipy.linalg import lu_solve

if TYPE_CHECKING:                                    # pragma: no cover
    from .vlm import VLM

__all__ = [
    "AXIS_FLIP", "Deck", "to_body", "rhs_circulation", "panel_loads",
    "deck", "DISTURBANCES", "ControlSurface", "aileron", "flap", "rudder",
    "elevator", "control_column", "spiral_margin", "spiral_margin_of",
]

_XHAT = np.array([1.0, 0.0, 0.0])
_YHAT = np.array([0.0, 1.0, 0.0])
_ZHAT = np.array([0.0, 0.0, 1.0])

#: lattice frame (x aft, z up) -> body frame (x forward, z down). A proper
#: rotation (det = +1), applied ONCE, in :func:`to_body`.
AXIS_FLIP = np.array([-1.0, 1.0, -1.0])

#: the columns :func:`deck` solves, in the order it reports them.
DISTURBANCES = ("alpha", "beta", "p", "q", "r")


# --------------------------------------------------------------- the columns

def rhs_circulation(model: "VLM", u: np.ndarray) -> np.ndarray:
    """Circulation induced by an extra wind ``u`` at the control points.

    ``u`` is ``(n_panels, 3)`` — the disturbance velocity each panel's
    flow-tangency point sees, in the LATTICE frame. The boundary condition is
    linear, so this is one back-substitution against factors that already
    exist; nothing is refactorised and the geometry is not rebuilt.
    """
    u = np.asarray(u, dtype=float)
    if u.shape != model.nrm.shape:
        raise ValueError(
            f"extra wind must be one 3-vector per panel {model.nrm.shape}, "
            f"got {u.shape}")
    return lu_solve(model._lu, -np.einsum("ij,ij->i", u, model.nrm))


def _disturbance_wind(model: "VLM", kind: str, r: np.ndarray) -> np.ndarray:
    """The extra wind field for one named disturbance, at UNIT amplitude.

    ``r`` is the arm from the CG to wherever the field is wanted. The BOUNDARY
    CONDITION needs it at the control points; the FORCE needs it at the bound
    segments, because that is where the load acts. Passing the wrong one is a
    quiet error of order the panel chord, so the caller names the point.
    """
    n = model.n_panels
    if kind == "alpha":
        return model.V * np.tile(_ZHAT, (n, 1))
    if kind == "beta":
        return -model.V * np.tile(_YHAT, (n, 1))
    if kind in ("p", "q", "r"):
        omega = np.zeros(3)
        omega["pqr".index(kind)] = 1.0
        # A UNIT BODY RATE, WRITTEN IN LATTICE COMPONENTS. The disturbance is
        # asked for in the frame the answer is reported in, so the rate has to
        # cross the frames too: x_body = -x_lattice and z_body = -z_lattice,
        # hence Omega_lattice = AXIS_FLIP * Omega_body. Flipping only the
        # OUTPUT moment (and leaving the input rate in lattice axes) is what
        # made Cl_p come out at +0.5346 — roll ANTI-damping, an aeroplane that
        # rolls faster the more it rolls. The invariance noted in the module
        # docstring is exactly this double flip; it is not automatic.
        omega = omega * AXIS_FLIP
        # the panel MOVES with the body, so the air it meets moves the other
        # way: the relative wind is -(Omega x r).
        return -np.cross(omega, r)
    raise ValueError(f"unknown disturbance {kind!r}; "
                     f"expected one of {DISTURBANCES}")


# ------------------------------------------------------- control surfaces

@dataclass(frozen=True)
class ControlSurface:
    """A hinged surface, as the ONE extra right-hand side it really is.

    A trailing-edge deflection enters the linearised boundary condition as a
    uniform incidence shift on the panels it covers, scaled by the
    thin-aerofoil effectiveness ``tau(c_e/c)``
    (:func:`tail.flap_effectiveness`). That is precisely the channel
    :class:`vlm.VLM` already uses for the tail incidence ``i_t``, so a
    control surface costs one back-substitution and no new physics.

    ``gain`` is per-panel and SIGNED: the sign is what makes an aileron
    antisymmetric and a flap symmetric, and it is the only difference between
    the two.
    """

    name: str
    kind: str                     # aileron | flap | elevator | rudder
    gain: np.ndarray              # (n_panels,) equivalent incidence per rad
    chord_frac: float

    @property
    def n_panels(self) -> int:
        return int(np.count_nonzero(self.gain))


def hinge_couple(chord_frac: float) -> float:
    """The section's own nose-down couple per radian, about the quarter chord.

        d(cm_c/4)/d(delta) = -(1/2) sin(theta_f) (1 - cos(theta_f))
        theta_f = arccos(2 c_e/c - 1)

    (thin-aerofoil hinged-flap theory, the companion result to the
    ``tau`` in :func:`tail.flap_effectiveness`.)

    THIS IS NOT DECORATION. With one chordwise panel per strip every bound
    vortex sits on the quarter-chord line, so the lattice can only ever
    produce lift-times-arm — exactly the limitation :meth:`vlm.VLM.solve`
    documents for a cambered section's couple, which it makes the CALLER hand
    in as ``cm_ac``. Without this term a flap measures as pitching the
    aircraft NOSE UP (+1.08 on the reference case), because the lattice puts
    the extra lift on the quarter-chord line ahead of the CG and has no way
    to know the load actually sits near the flap. A pilot trusting that sign
    would be flown into the ground by their own flaps.

    Both ends are zero and that is the check worth remembering: a
    whole-chord "flap" (``c_e/c = 1``) is just an incidence change and makes
    no couple about the quarter chord, and no flap at all makes none either.
    """
    r = float(np.clip(chord_frac, 1e-6, 1.0))
    th = float(np.arccos(2.0 * r - 1.0))
    return float(-0.5 * np.sin(th) * (1.0 - np.cos(th)))


def _tau_vector(model: "VLM") -> np.ndarray:
    """``(t_hat x n) . x_hat`` per panel — how much a unit incidence change on
    this panel shifts its own boundary condition.

    This is :class:`vlm.VLM`'s own tail-incidence construction, evaluated for
    every panel instead of only the tail's. It is 1 for a level wing panel
    AND 1 for a vertical fin panel (``t_hat = z_hat``, ``n = -y_hat``, so
    ``t_hat x n = x_hat``), which is why a rudder needs no separate code path
    from an aileron.
    """
    that = model.lvec / model.width[:, None]
    return np.einsum("ij,j->i", np.cross(that, model.nrm), _XHAT)


def _band_mask(coord: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (np.abs(coord) >= lo) & (np.abs(coord) <= hi)


def _mask(model: "VLM", name: str) -> np.ndarray:
    """One of the lattice's surface masks, as a real array on every model.

    ``VLM`` leaves a mask ``None`` where the surface is absent, and three
    control builders here were reading them with ``|`` — which raises on a
    None rather than reporting an aeroplane without that surface.
    """
    m = getattr(model, name, None)
    return (np.zeros(model.n_panels, dtype=bool) if m is None
            else np.asarray(m, dtype=bool))


def _main_wing(model: "VLM") -> np.ndarray:
    """The panels a wing-mounted control is cut out of.

    THE REAR WING OF A TANDEM IS NOT PART OF IT. It was, until this line
    existed: the mask was ``~(is_tail | is_vertical)``, and a tandem's rear
    surface is neither a tail nor a vertical, so an aileron band selected the
    outboard panels of BOTH wings — two hinges on the same trailing edge the
    elevon below is cut out of. The pair now splits the way a pilot would
    split it: roll on the front wing, pitch on the rear one.
    """
    return ~(_mask(model, "is_tail") | _mask(model, "is_vertical")
             | _mask(model, "is_second"))


def aileron(model: "VLM", span_frac: tuple[float, float] = (0.60, 0.98),
            chord_frac: float = 0.25) -> ControlSurface:
    """An ANTISYMMETRIC pair on the outboard wing.

    Sign convention, stated once and asserted in the tests: **positive
    ``delta_a`` puts the STARBOARD aileron trailing edge DOWN**. Starboard
    then makes more lift, so the aircraft rolls to PORT and ``Cl_da < 0`` in
    body axes (where positive roll is starboard-wing-down).
    """
    from .tail import flap_effectiveness
    wing = _main_wing(model)
    semi = float(np.abs(model.y[wing]).max())
    band = _band_mask(model.y, span_frac[0] * semi, span_frac[1] * semi)
    gain = np.zeros(model.n_panels)
    sel = wing & band
    # sign(y): +1 to starboard, -1 to port — the antisymmetry IS the aileron
    gain[sel] = flap_effectiveness(chord_frac) * np.sign(model.y[sel])
    return ControlSurface("aileron", "aileron", gain, float(chord_frac))


def flap(model: "VLM", span_frac: tuple[float, float] = (0.05, 0.55),
         chord_frac: float = 0.30) -> ControlSurface:
    """The SYMMETRIC version of the same surface, inboard.

    Positive ``delta_f`` is trailing edge down on both sides: more lift, and
    (because the extra lift acts aft of the quarter-chord line the lattice
    puts it on, and behind the CG of a conventional layout) a nose-down
    pitching moment. Only the SIGN of the gain differs from an aileron.
    """
    from .tail import flap_effectiveness
    wing = _main_wing(model)
    semi = float(np.abs(model.y[wing]).max())
    band = _band_mask(model.y, span_frac[0] * semi, span_frac[1] * semi)
    gain = np.zeros(model.n_panels)
    gain[wing & band] = flap_effectiveness(chord_frac)
    return ControlSurface("flap", "flap", gain, float(chord_frac))


def elevator(model: "VLM", chord_frac: float = 0.40) -> ControlSurface:
    """A hinged elevator on the surface that trims the aircraft in pitch.

    ``VLM._Gam2`` already gives the all-moving case (``chord_frac = 1``, and
    ``flap_effectiveness(1.0) == 1``); this is the same channel at partial
    chord. Positive ``delta_e`` is trailing edge down, which pitches nose
    DOWN.

    WHICH SURFACE CARRIES IT IS THE LAYOUT'S ANSWER, not a caller's. A
    wing+tail hinges the tailplane. A TANDEM has no tailplane and is not
    thereby an aeroplane with no pitch control: its rear wing is the aft
    surface, sitting at the stagger the pair was scored with, so the hinge
    goes there and the column is still called ``elevator`` — an ELEVON on
    the rear wing, which is what a tandem's pitch control is.

    Before this, the gain was ``model.is_tail`` alone, so a tandem's deck
    came back with columns ``[alpha, beta, p, q, r, aileron, rudder]``,
    ``sixdof.trim_level`` had nothing to solve for, and V4's stage 6 refused
    to arm a two-surface aircraft with the words "it carries one lifting
    surface". Every registered tandem family was unflyable.

    A configuration with NEITHER surface (a single wing) still gets an
    all-zero gain, which is the honest answer and the one stage 6's refusal
    is really about.
    """
    from .tail import flap_effectiveness
    gain = np.zeros(model.n_panels)
    sel = _mask(model, "is_tail")
    if not bool(sel.any()):
        sel = _mask(model, "is_second")
    gain[sel] = flap_effectiveness(chord_frac)
    return ControlSurface("elevator", "elevator", gain, float(chord_frac))


def rudder(model: "VLM", chord_frac: float = 0.40) -> ControlSurface:
    """A hinged rudder on the vertical surface.

    Needs a fin: with no vertical panels the gain is all zeros and every
    rudder derivative is zero — which is the honest answer, not a failure.
    """
    from .tail import flap_effectiveness
    gain = np.zeros(model.n_panels)
    gain[_mask(model, "is_vertical")] = flap_effectiveness(chord_frac)
    return ControlSurface("rudder", "rudder", gain, float(chord_frac))


def control_column(model: "VLM", surf: ControlSurface) -> np.ndarray:
    """Circulation per radian of deflection of ``surf``."""
    return lu_solve(model._lu, -model.V * _tau_vector(model) * surf.gain)


# ----------------------------------------------------------------- the loads

def panel_loads(model: "VLM", Gam: np.ndarray, x_cg: float,
                z_cg: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """``(F, M)`` per unit density about ``(x_cg, 0, z_cg)``, LATTICE frame.

    Kutta-Joukowski on each bound segment, ``F = rho V_inf x Gamma l``, with
    the force applied at the segment's own quarter-chord station. Summing the
    moment of those forces is :meth:`vlm.VLM._cm` generalised from one axis to
    three: with ``x_cg`` on the x-axis and only the y-component of ``l``
    contributing, the pitching component here reduces to that method exactly
    (gated in the tests), so the two cannot drift apart.

    ``rho`` is carried as 1: every consumer divides it straight back out
    through the dynamic pressure.
    """
    Gam = np.asarray(Gam, dtype=float)
    F = np.cross(np.broadcast_to(model.V * _XHAT, model.lvec.shape),
                 Gam[:, None] * model.lvec)
    arm = model.st3 - np.array([float(x_cg), 0.0, float(z_cg)])
    return F, np.cross(arm, F)


def induced_velocity(model: "VLM", Gam: np.ndarray) -> np.ndarray:
    """Induced velocity AT the bound segments, ``(n_panels, 3)``.

    Lifting-line closure: the wash at the wing is HALF the value the wake
    reaches far downstream, and it acts along the panel normal. The lattice
    already carries the Trefftz-plane wash operator ``_WN`` (it is what the
    published induced drag is computed from), so this needs no new influence
    matrix and — the point — no evaluation of a bound vortex at its own
    singular midpoint.

    The factor of one half is not asserted, it is MEASURED: near-field drag
    built from this velocity, ``D = -rho * sum(w Gamma width)``, reproduces
    the Trefftz ``CDi`` to a ratio of 1.000000 across taper and incidence
    (gated in the tests). That agreement is the classical statement that the
    two drag routes are the same integral, and it is what licenses using this
    velocity in the force.
    """
    return 0.5 * (model._WN @ Gam)[:, None] * model.nrm


def panel_loads_local(model: "VLM", Gam: np.ndarray, x_cg: float,
                      z_cg: float = 0.0, u_dist: np.ndarray | None = None,
                      induced_tilt: bool = False
                      ) -> tuple[np.ndarray, np.ndarray]:
    """``(F, M)`` with the LOCAL velocity in the Kutta-Joukowski cross product.

    ``F = rho V_local x Gamma l`` with

        V_local = V_inf + u_dist + w_induced

    Two things appear that the freestream-only force cannot produce, and they
    are the whole reason this function exists:

    * ``u_dist`` carries the DIFFERENTIAL DYNAMIC PRESSURE a yaw rate puts
      across the span — one wing advancing, one retreating — which is
      ``Cl_r``;
    * ``w_induced`` TILTS the force vector out of the vertical, turning lift
      into a streamwise component. That tilt is induced drag, and its
      spanwise asymmetry under a roll rate is adverse yaw, ``Cn_p``.

    Both are bilinear — a perturbation times the base loading — so unlike the
    freestream force they depend on the state the deck is linearised about.
    That is why :func:`deck` takes an ``alpha``.
    """
    Gam = np.asarray(Gam, dtype=float)
    V_local = np.broadcast_to(model.V * _XHAT, model.lvec.shape).copy()
    if u_dist is not None:
        V_local = V_local + u_dist
    if induced_tilt:
        V_local = V_local + induced_velocity(model, Gam)
    F = np.cross(V_local, Gam[:, None] * model.lvec)
    arm = model.st3 - np.array([float(x_cg), 0.0, float(z_cg)])
    return F, np.cross(arm, F)


def _coefficients(model: "VLM", Gam: np.ndarray, x_cg: float, z_cg: float,
                  b: float, mac: float, u_dist=None,
                  local: bool = False, induced_tilt: bool = False
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Force and moment COEFFICIENTS in the lattice frame."""
    if local:
        F, M = panel_loads_local(model, Gam, x_cg, z_cg, u_dist, induced_tilt)
    else:
        F, M = panel_loads(model, Gam, x_cg, z_cg)
    q = 0.5 * model.V**2
    CF = F.sum(axis=0) / (q * model.S)
    CM = M.sum(axis=0) / (q * model.S * np.array([b, mac, b]))
    return CF, CM


def to_body(CF: np.ndarray, CM: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Lattice frame (x aft, z up) -> body frame (x forward, z down).

    One multiplication by :data:`AXIS_FLIP`, and it is the ONLY place in the
    package where that conversion happens.
    """
    return np.asarray(CF) * AXIS_FLIP, np.asarray(CM) * AXIS_FLIP


# ------------------------------------------------------------------ the deck

@dataclass(frozen=True)
class Deck:
    """A first-order aerodynamic model, in BODY axes, about one CG.

    The columns are derivatives with respect to the non-dimensional
    disturbance: ``alpha`` and ``beta`` in radians, and the rates in the usual
    half-times ``p b / 2V``, ``q mac / 2V``, ``r b / 2V``.

    IT IS A TAYLOR EXPANSION ABOUT A STATE, AND THE STATE TRAVELS WITH IT.
    ``const`` is the coefficient set AT the base state — the one named by
    :attr:`alpha_ref` and :attr:`i_t_ref` — and the columns are the slopes
    THERE, so the model a caller evaluates is

        C(x) = const + sum_k column[k] * (x_k - x_ref[k])

    with ``x_ref`` zero on every channel except ``alpha``. Dropping the
    ``- alpha_ref`` is not a rounding matter: it adds the base lift twice.
    Measured on a wing+tail+fin at 30 m/s, a deck built at ``alpha = 3 deg``
    carries ``const["CZ"] = -0.286449``, which is ``CL_alpha x 3 deg`` to
    six figures — so a consumer that added ``CL_alpha x alpha`` on top of it
    flew an aeroplane trimming at a body incidence exactly ``alpha_ref`` too
    low (measured: base 0 deg -> trim +2.313, base 3 deg -> -0.669, base
    6 deg -> -3.654 — a 1:1 slide).

    Why the base cannot simply be dropped instead: with the local-velocity
    force (:func:`panel_loads_local`) the coefficients are BILINEAR, so
    ``Cl_r`` and ``Cn_p`` are proportional to the base ``CL`` and exist only
    away from ``alpha = 0``. The expansion point is part of the physics,
    which is exactly why it has to be part of the object.
    """

    #: disturbance -> ``{"CX", "CY", "CZ", "Cl", "Cm", "Cn"}``
    columns: dict[str, dict[str, float]]
    const: dict[str, float]
    #: reference quantities the non-dimensionalisation used
    S: float
    b: float
    mac: float
    V: float
    x_cg: float
    z_cg: float = 0.0
    #: THE STATE ``const`` AND THE COLUMNS BELONG TO [rad]. A consumer
    #: evaluates the alpha column at ``alpha - alpha_ref``, never at
    #: ``alpha``. ``i_t_ref`` is recorded rather than subtracted: the tail
    #: incidence is folded into the base and has no column of its own (the
    #: elevator column is what moves it).
    alpha_ref: float = 0.0
    i_t_ref: float = 0.0
    #: derivatives this deck reports as exactly zero together with WHY, so a
    #: consumer can say it on screen instead of implying a stable aeroplane
    #: (``PLAN_SESSION65_V4.md`` item 10).
    zeros: dict[str, str] = field(default_factory=dict)

    def disturbance(self, alpha: float) -> float:
        """The alpha DISTURBANCE for a total incidence — ``alpha -
        alpha_ref``.

        A named method rather than a subtraction at each call site, because
        there are three of them (:meth:`sixdof.Aircraft.coefficients`,
        :func:`sixdof.trim_level` and the stage-5 check) and the whole class
        of bug this fixes is one of them forgetting.
        """
        return float(alpha) - float(self.alpha_ref)

    # -- the named derivatives, so callers read physics and not dict keys ---
    @property
    def CL_alpha(self) -> float:
        """Lift-curve slope [1/rad]. ``CZ`` is DOWN in body axes, so lift is
        its negative."""
        return -self.columns["alpha"]["CZ"]

    @property
    def Cm_alpha(self) -> float:
        return self.columns["alpha"]["Cm"]

    @property
    def Cl_p(self) -> float:
        return self.columns["p"]["Cl"]

    @property
    def Cm_q(self) -> float:
        return self.columns["q"]["Cm"]

    @property
    def Cn_r(self) -> float:
        return self.columns["r"]["Cn"]

    @property
    def Cl_r(self) -> float:
        return self.columns["r"]["Cl"]

    @property
    def Cn_p(self) -> float:
        return self.columns["p"]["Cn"]

    @property
    def CY_beta(self) -> float:
        return self.columns["beta"]["CY"]

    @property
    def Cl_beta(self) -> float:
        return self.columns["beta"]["Cl"]

    @property
    def Cn_beta(self) -> float:
        return self.columns["beta"]["Cn"]

    @property
    def static_margin(self) -> float:
        """``-Cm_alpha / CL_alpha`` [mac]. Positive is statically stable.

        This is the SAME number :meth:`vlm.VLM.neutral_point` gives as
        ``(x_np - x_cg)/mac``; the two are gated equal in the tests, which is
        what stops this module inventing a second answer to a question the
        solver already answers.
        """
        if abs(self.CL_alpha) < 1e-12:
            raise ValueError("degenerate lift-curve slope")
        return -self.Cm_alpha / self.CL_alpha


def deck(model: "VLM", x_cg: float, mac: float | None = None,
         b: float | None = None, z_cg: float = 0.0,
         controls: "list[ControlSurface] | None" = None,
         alpha: float = 0.0, i_t: float = 0.0,
         local_velocity: bool = True, induced_tilt: bool = False,
         eps: float = 1e-5) -> Deck:
    """Every disturbance column, against the lattice's existing factors.

    ``b`` and ``mac`` default to the lattice's own reference quantities —
    ``b = sqrt(AR S)`` is the NOMINAL span the aspect ratio and the span
    efficiency are already referenced to, so a tip device changes the loading
    without changing what the coefficients mean.

    ``alpha`` (and ``i_t``) name the state the deck is LINEARISED ABOUT.
    They are stored on the returned :class:`Deck` as ``alpha_ref`` /
    ``i_t_ref`` and ``const`` is the coefficient set THERE, not at zero — so
    every consumer evaluates the alpha column at ``deck.disturbance(alpha)``
    and never at ``alpha``. With ``local_velocity`` on they matter twice
    over. The freestream-only force is
    exactly linear, so a base state is irrelevant to it; the local-velocity
    force is bilinear — perturbation circulation times base induced field,
    plus base circulation times perturbation induced field — so ``Cl_r`` and
    ``Cn_p`` are proportional to the base ``CL`` and simply do not exist at
    ``alpha = 0``. Asking for them on an unloaded wing returns zero because
    that is the answer, not because the term is missing.

    ``local_velocity=False`` restores the freestream-only force exactly, which
    is what the published lift and moment derivatives were computed with.
    """
    b = float(b) if b is not None else float(np.sqrt(model.AR * model.S))
    mac = float(mac) if mac is not None else float(model.S / b)
    cg = np.array([float(x_cg), 0.0, float(z_cg)])
    r_cp = model.cp - cg
    r_st = model.st3 - cg              # where the FORCE acts, not the BC

    # the rate columns are asked for at the amplitude that makes the answer a
    # derivative with respect to the NON-DIMENSIONAL rate
    hat = {"alpha": 1.0, "beta": 1.0,
           "p": 2.0 * model.V / b, "q": 2.0 * model.V / mac,
           "r": 2.0 * model.V / b}

    keys = ("CX", "CY", "CZ", "Cl", "Cm", "Cn")
    columns: dict[str, dict[str, float]] = {}

    # ---- the basis: one circulation column and one wind field per channel,
    # both already scaled so a unit amplitude IS a unit of the reported
    # derivative.
    gam_of: dict[str, np.ndarray] = {}
    wind_of: dict[str, np.ndarray] = {}
    for kind in DISTURBANCES:
        gam_of[kind] = hat[kind] * rhs_circulation(
            model, _disturbance_wind(model, kind, r_cp))
        wind_of[kind] = hat[kind] * _disturbance_wind(model, kind, r_st)
    for surf in (controls or ()):
        gam_of[surf.name] = control_column(model, surf)
        # a deflection moves the SURFACE, not the air: no free-stream
        # disturbance travels with it
        wind_of[surf.name] = np.zeros_like(model.lvec)

    gam_base = model._gamma(float(alpha), float(i_t))
    wind_base = float(alpha) * _disturbance_wind(model, "alpha", r_st)

    def _load(amps: dict) -> tuple[np.ndarray, np.ndarray]:
        Gam = gam_base.copy()
        u = wind_base.copy()
        for k, a in amps.items():
            if a:
                Gam = Gam + a * gam_of[k]
                u = u + a * wind_of[k]
        return _coefficients(model, Gam, x_cg, z_cg, b, mac, u_dist=u,
                             local=local_velocity, induced_tilt=induced_tilt)

    if not local_velocity:
        # the exactly-linear path: one evaluation per column, no base state
        for kind in DISTURBANCES:
            CF, CM = to_body(*_coefficients(model, gam_of[kind], x_cg, z_cg,
                                            b, mac))
            columns[kind] = dict(zip(keys,
                                     [*map(float, CF), *map(float, CM)]))
    else:
        # CENTRAL DIFFERENCES, because the force is no longer linear in the
        # disturbance. Two extra back-substitution-free evaluations per
        # column; the circulations were already solved above.
        for kind in DISTURBANCES:
            Fp, Mp = _load({kind: +eps})
            Fm, Mm = _load({kind: -eps})
            CF, CM = to_body((Fp - Fm) / (2 * eps), (Mp - Mm) / (2 * eps))
            columns[kind] = dict(zip(keys,
                                     [*map(float, CF), *map(float, CM)]))

    # ...and one column per control surface, through the same factors. The
    # deflection is in RADIANS, so these columns are already in the same
    # units the integrator wants.
    for surf in (controls or ()):
        if local_velocity:
            Fp, Mp = _load({surf.name: +eps})
            Fm, Mm = _load({surf.name: -eps})
            CF, CM = to_body((Fp - Fm) / (2 * eps), (Mp - Mm) / (2 * eps))
        else:
            CF, CM = to_body(*_coefficients(model, gam_of[surf.name],
                                            x_cg, z_cg, b, mac))
        col = dict(zip(keys, [*map(float, CF), *map(float, CM)]))
        # ...plus the section couple the one-panel lattice cannot make
        # (:func:`hinge_couple`). It acts about each deflected panel's own
        # quarter chord, so it is an area-weighted sum and it does NOT move
        # with the CG. A vertical surface's couple is about its own axis and
        # contributes to yaw, not pitch, so it is routed by the panel's own
        # orientation rather than assumed to be pitching.
        on = surf.gain != 0.0
        if np.any(on):
            # The couple is a moment about the PANEL'S OWN SPAN AXIS t_hat, so
            # it is routed by geometry rather than special-cased per surface:
            # t_hat = y_hat on a wing panel makes it a pitching moment, and
            # t_hat = z_hat on a fin panel makes the identical formula a
            # YAWING one. Dimensionally it is cm * q * (c * width) * c — the
            # section moment coefficient carries a chord of its OWN on top of
            # the panel area, and that missing factor of c is what left the
            # flap still reading nose-up after the couple was first added.
            that = (model.lvec / model.width[:, None])[on]
            scale = (hinge_couple(surf.chord_frac) * np.sign(surf.gain[on])
                     * model.c[on] ** 2 * model.width[on])
            M_couple = (scale[:, None] * that).sum(axis=0)
            _, CM_c = to_body(np.zeros(3),
                              M_couple / (model.S * np.array([b, mac, b])))
            for _k, _v in zip(("Cl", "Cm", "Cn"), CM_c):
                col[_k] += float(_v)
        columns[surf.name] = col

    if local_velocity:
        CF0, CM0 = to_body(*_load({}))
    else:
        CF0, CM0 = to_body(*_coefficients(model, gam_base, x_cg, z_cg, b, mac))
    const = dict(zip(keys, [*map(float, CF0), *map(float, CM0)]))

    # WHY a derivative is zero, in the deck itself. A consumer that prints a
    # bare 0.0 for Cn_beta is describing an aeroplane with no fin as though it
    # were neutrally stable in yaw, which is not the same statement at all.
    zeros: dict[str, str] = {}
    if abs(columns["beta"]["Cn"]) < 1e-12:
        zeros["Cn_beta"] = (
            "no surface in this lattice carries side force: with no fin, no "
            "canted tip device and no dihedral there is nothing for a "
            "sideslip to push on. This is a GEOMETRY gap, not a stable "
            "aeroplane — add a vertical surface before reading yaw dynamics.")
    unloaded = abs(columns["alpha"]["CZ"]) > 0 and abs(const["CZ"]) < 1e-9
    why_base = (
        "this deck is linearised about an UNLOADED state (the base CL is "
        "zero), and this derivative is proportional to the base CL: a yaw "
        "rate can only redistribute lift that already exists, and adverse "
        "yaw is an asymmetry of induced drag that a wing making no lift does "
        "not have. Rebuild the deck at the trim angle of attack — pass "
        "deck(..., alpha=alpha_trim) — and it becomes a real number.")
    why_model = (
        "the freestream-only force is in use (local_velocity=False), which "
        "cannot see either the differential dynamic pressure across the span "
        "or the tilt of the force vector by the induced field.")
    if abs(columns["r"]["Cl"]) < 1e-12:
        zeros["Cl_r"] = why_base if (local_velocity and unloaded) else why_model
    if abs(columns["p"]["Cn"]) < 1e-12:
        zeros["Cn_p"] = why_base if (local_velocity and unloaded) else why_model

    return Deck(columns=columns, const=const, S=float(model.S), b=b, mac=mac,
                V=float(model.V), x_cg=float(x_cg), z_cg=float(z_cg),
                alpha_ref=float(alpha), i_t_ref=float(i_t), zeros=zeros)


def weathercocks(Cn_beta) -> bool:
    """Does this aeroplane yaw INTO the airflow?

    ONE definition, because two instruments were disagreeing on it and both
    were printing the word "stable". ``Cn_beta > 0`` is directional
    stability; at or below zero the aeroplane turns AWAY from a sideslip
    and two separate things stop being true at once:

    * the spiral CRITERION stops meaning anything — it is
      ``Cl_beta(Cn_r - t.Cn_p) - Cn_beta(Cl_r - t.Cl_p)``, so the second
      product changes sign and a design "converges" on paper
      (:func:`wing_score.spiral_refusal`, which asks this question);
    * and the slow lateral EIGENVALUE stops being a spiral at all — with no
      yaw stiffness the yaw equation decouples, so what is left is a
      sideslip subsidence wearing the spiral's name.

    MEASURED, on the two designs that made this necessary. A wing+tail with
    the fin switched OFF: ``Cn_beta`` exactly -0.0, the criterion exactly
    0.0, and ``modes.classify`` naming a "spiral" at -0.02221 that HALVES in
    31.2 s — the Flight stage telling a user their finless aeroplane's
    spiral converges. And `hydrofoil + elevator` at its box centre, the
    other way round: ``Cn_beta`` -0.19599, criterion **+0.02016** (which
    reads as convergent), eigenvalue **+0.44979**, which DOUBLES in 1.54 s.

    NOT a tolerance band. Exactly zero is the finless case, where the
    surface producing the stiffness does not exist, so the boundary is the
    sign and nothing is rounded into stability.
    """
    if isinstance(Cn_beta, bool) or not isinstance(Cn_beta, (int, float)):
        return False
    v = float(Cn_beta)
    return v == v and v > 0.0


def spiral_margin(*, Cl_beta: float, Cn_r: float,
                  Cn_beta: float, Cl_r: float,
                  Cn_p: float = 0.0, Cl_p: float = 0.0,
                  theta0_rad: float = 0.0) -> float:
    """``Cl_beta*(Cn_r - t.Cn_p) - Cn_beta*(Cl_r - t.Cl_p)``, ``t = tan(th0)``:
    positive is a convergent spiral.

    The textbook criterion, written as the quantity itself rather than as a
    yes/no, because the useful question is not "does it diverge" but "by how
    much, and which term is winning". Both products are negative-times-
    positive in the stable case, so the sign carries the whole answer: the
    dihedral effect working through yaw damping must beat the fin's yaw
    stiffness working through roll-due-to-yaw.

    It is a criterion on the DECK, so it can be evaluated wherever there is
    one — stage 5, where there is no state to linearise about, and now also
    inside an OBJECTIVE, which is why it lives here. It was written in
    ``gui/v4/modes.py``, and a number the engine scores designs on cannot
    have its definition in a shell: ``modes`` re-exports this one so the two
    can never drift.

    **AND IT CARRIES THE TRIM ATTITUDE, because the textbook form is exact
    only at ``theta0 = 0``.** The four-state lateral model closes on
    ``phi_dot = p + tan(theta0) r``; the classical criterion drops the second
    term, which is free of charge on an aeroplane flying level at zero
    attitude and is not free on one that trims nose-up. Restoring it moves
    ``Cn_r -> Cn_r - t.Cn_p`` and ``Cl_r -> Cl_r - t.Cl_p``.

    MEASURED (2026-08-30, ``scripts/cant_arms.py --crossings``; the sweep and
    both families are in ``RESULTS_SESSION67_WING_CANT.md``). Sweeping the
    dihedral row and asking where each instrument crosses zero:

        family                        classical   THIS      6-DOF eigenvalue
        tail + winglet [free cant]      0.876     3.007     3.006 deg
        tail [free cant]                3.560     5.687     5.687 deg

    — so the classical form is OPTIMISTIC by 2.13 deg of dihedral on both,
    and this one lands on the mode it is a proxy for to 0.001 deg. These
    designs trim about 4.2 deg nose-up and ``Cl_p`` is -0.55 to -0.62, so
    ``tan(theta0) Cl_p`` is not a small correction; the error scales with
    ``tan(theta0)`` and reaches ~4.0 deg at 12 deg of attitude.

    It cost the shipped recommendation six designs in 42: every one of the
    spiral-weighted winners satisfied the classical criterion and 6 were
    divergent by the eigenvalue, because ``min(margin, 0)`` STOPS PAYING
    2.1 deg before the mode turns. All six carried anhedral.

    ``theta0_rad = 0.0`` is the default so that a caller with no attitude
    gets the textbook number bit-for-bit — ``t`` is exactly 0.0 and both
    corrections vanish — but a caller that HAS a deck should use
    :func:`spiral_margin_of`, which reads the attitude the deck was
    linearised at.
    """
    t = float(np.tan(float(theta0_rad)))
    return (float(Cl_beta) * (float(Cn_r) - t * float(Cn_p))
            - float(Cn_beta) * (float(Cl_r) - t * float(Cl_p)))


def spiral_margin_of(deck: "Deck") -> float:
    """:func:`spiral_margin` for a deck, at the attitude the deck was built at.

    ``deck.alpha_ref`` is the angle of attack the columns were linearised
    about, and in LEVEL flight the pitch attitude IS the angle of attack
    (``theta0 = alpha + gamma``, and a level flight path is ``gamma = 0``).
    Every caller in this package flies level, so this is the one call that
    gets the attitude right without anybody having to remember it — which is
    the whole reason the classical form went 2.13 deg optimistic for as long
    as it did.

    A caller flying a climb or a descent must pass ``theta0_rad`` to
    :func:`spiral_margin` itself.
    """
    return spiral_margin(Cl_beta=deck.Cl_beta, Cn_r=deck.Cn_r,
                         Cn_beta=deck.Cn_beta, Cl_r=deck.Cl_r,
                         Cn_p=deck.Cn_p, Cl_p=deck.Cl_p,
                         theta0_rad=float(deck.alpha_ref))
