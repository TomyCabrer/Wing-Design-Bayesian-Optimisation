"""Six-degree-of-freedom flight simulation on the deck the lattice produced.

This is the piece that lets a user FEEL a design instead of reading a table of
it: thrust, angle of attack, weight, CG, roll rate and the accelerations that
follow. It integrates rigid-body motion in the standard aircraft body frame
(x forward, y starboard, z down) driven by :mod:`dynamics`, so every number it
flies came out of the same lattice the optimiser scored.

There is deliberately NO PROPELLER MODEL. Thrust is a scalar along body x with
a stated vertical offset, because a slipstream that washed the wing would be a
second aerodynamic model and this module's whole point is that there is only
one. ``slipstream.py`` exists for the day that changes.

What is modelled, and what is assumed
-------------------------------------
* **Aerodynamics** — affine in ``(alpha, beta, p, q, r, controls)`` from
  :class:`dynamics.Deck`, which is exact for the lattice within its linear
  range and is blended out past the stall (:class:`Stall`).
* **Drag** — the lattice gives induced drag in the Trefftz plane and nothing
  else, so profile drag is supplied by the caller as ``CD0`` and the induced
  part is rebuilt as ``CL^2 / (pi AR e)``. Both numbers come off the design's
  own report; neither is invented here.
* **Inertia** — :class:`Inertia`, and it is the one quantity in this module
  the package has never had. It is an ESTIMATE with a stated construction,
  exposed so the user can overrule it, never a hidden constant.
* **Gravity** — flat earth, constant g. At the speeds and durations a design
  review flies, curvature and rotation are noise.
* **Atmosphere** — constant density, taken at the state the design was flown
  at. A climb does not thin the air here.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import ClassVar

import numpy as np

__all__ = ["G", "Inertia", "Stall", "Propulsion", "Aircraft", "State",
           "body_axis_gravity", "quat_to_dcm", "dcm_to_euler", "trim_level",
           "step", "integrate", "linearise"]

G = 9.80665                     # [m/s2]


# ------------------------------------------------------------------ inertia

@dataclass(frozen=True)
class Inertia:
    """Mass and the principal moments, with the construction stated.

    The package models weight (:mod:`weights`) and has never modelled
    INERTIA, so this is new and it is an estimate. Two honest routes, and the
    caller picks:

    :meth:`from_layout`
        integrate the masses the design already implies — the wing as a
        uniform spanwise bar, the fuselage and tail as a bar along x, with
        the tail's own mass at its own arm. Closed forms all the way, so
        every number can be checked by hand.
    :meth:`from_radii`
        non-dimensional radii of gyration, the handbook route (Raymer Ch. 16
        tabulates them by aircraft class). Use when a class is known and a
        layout is not.

    Products of inertia are taken as zero: the vehicle is assumed
    symmetric about its own x-z plane, which every configuration this package
    builds is. ``Ixz`` is the term that couples roll and yaw in a swept
    aircraft, and leaving it out is a real simplification — it is stated here
    rather than buried.
    """

    mass_kg: float
    Ixx: float
    Iyy: float
    Izz: float
    #: free-text record of WHERE these numbers came from, carried so a view
    #: can print it beside them
    basis: str = "given"

    def __post_init__(self):
        for name in ("mass_kg", "Ixx", "Iyy", "Izz"):
            if not float(getattr(self, name)) > 0.0:
                raise ValueError(
                    f"{name} must be > 0, got {getattr(self, name)!r}")

    @property
    def tensor(self) -> np.ndarray:
        return np.diag([self.Ixx, self.Iyy, self.Izz])

    @classmethod
    def from_layout(cls, mass_kg: float, b: float, length_m: float,
                    wing_mass_frac: float = 0.30,
                    tail_mass_frac: float = 0.06,
                    tail_arm_m: float | None = None) -> "Inertia":
        """Integrate the layout, in closed form.

        * the wing, a uniform bar of span ``b`` about the roll axis:
          ``I = m_w b^2 / 12``
        * the body, a uniform bar of length ``length_m`` about the pitch
          axis: ``I = m_b L^2 / 12``
        * the tail, a point mass at its arm: ``I = m_t l_t^2`` in BOTH pitch
          and yaw
        * yaw takes the wing's span term and the body's length term together,
          which is the perpendicular-axis result for a vehicle that is much
          flatter than it is wide or long.
        """
        m = float(mass_kg)
        if not m > 0.0:
            raise ValueError(f"mass_kg must be > 0, got {mass_kg!r}")
        m_w = m * float(wing_mass_frac)
        m_t = m * float(tail_mass_frac)
        m_b = m - m_w - m_t
        if m_b <= 0.0:
            raise ValueError(
                f"wing_mass_frac + tail_mass_frac must be < 1, got "
                f"{wing_mass_frac!r} + {tail_mass_frac!r}")
        l_t = float(tail_arm_m if tail_arm_m is not None
                    else 0.45 * float(length_m))
        Ixx = m_w * float(b) ** 2 / 12.0
        Iyy = m_b * float(length_m) ** 2 / 12.0 + m_t * l_t ** 2
        Izz = Ixx + Iyy
        return cls(m, Ixx, Iyy, Izz,
                   basis=(f"layout: wing bar b={b:.3g} m at "
                          f"{wing_mass_frac:.0%} of mass, body bar "
                          f"L={length_m:.3g} m, tail point mass at "
                          f"l_t={l_t:.3g} m"))

    @classmethod
    def from_radii(cls, mass_kg: float, b: float, length_m: float,
                   Rx: float = 0.25, Ry: float = 0.38,
                   Rz: float = 0.39) -> "Inertia":
        """Handbook radii of gyration: ``I = m (R L)^2`` on the matching
        reference length (span for roll, body length for pitch, their mean
        for yaw)."""
        m = float(mass_kg)
        mean = 0.5 * (float(b) + float(length_m))
        return cls(m, m * (Rx * float(b)) ** 2,
                   m * (Ry * float(length_m)) ** 2, m * (Rz * mean) ** 2,
                   basis=f"radii of gyration Rx={Rx}, Ry={Ry}, Rz={Rz}")


# -------------------------------------------------------------------- stall

@dataclass(frozen=True)
class Stall:
    """Where the affine deck stops being allowed to make lift.

    The deck is linear in alpha and will happily produce CL = 3 at 30 degrees.
    That is exactly the region a user pokes first, so it must not be left
    linear.

    Lift is saturated with a p-norm soft clip

        CL_eff = CL_lin / (1 + |CL_lin/CL_max|^n)^(1/n)

    which asymptotes to ``CL_max`` and — the property that matters — leaves
    normal flight ALONE. The obvious ``CL_max * tanh(CL_lin/CL_max)`` does
    not: at a perfectly ordinary cruise ``CL = 0.95`` against ``CL_max =
    1.4`` it removes 13 % of the lift, which showed up here as an aircraft
    that could not be trimmed. At ``n = 8`` the same point loses 0.5 %.

    Past ``alpha_stall`` two more things happen, because a stall a user can
    feel has to be more than a ceiling: lift DROPS by ``lift_drop``, and a
    separated-drag term is added. Both ramp in smoothly over ``ramp``.

    Set ``enabled = False`` to fly the bare linear deck, which is what the
    stability-mode analysis wants.
    """

    CL_max: float = 1.4
    alpha_stall_rad: float = np.deg2rad(14.0)
    dCD_separated: float = 0.9
    lift_drop: float = 0.35
    sharpness: float = 8.0
    ramp_rad: float = np.deg2rad(8.0)
    enabled: bool = True

    def _over(self, alpha: float) -> float:
        """How far past the stall, as a smooth 0..1 ramp."""
        over = abs(float(alpha)) - self.alpha_stall_rad
        if over <= 0.0:
            return 0.0
        return float(np.tanh(over / self.ramp_rad) ** 2)

    def lift(self, CL_lin: float, alpha: float = 0.0) -> float:
        if not self.enabled or self.CL_max <= 0.0:
            return float(CL_lin)
        n = float(self.sharpness)
        clipped = float(CL_lin) / (1.0 + abs(float(CL_lin) / self.CL_max)
                                   ** n) ** (1.0 / n)
        return clipped * (1.0 - self.lift_drop * self._over(alpha))

    def slope_factor(self, CL_lin: float, alpha: float = 0.0) -> float:
        """``d(CL_eff)/d(CL_lin)`` — how much lift slope the wing has LEFT.

        THE CLIP HAS TO REACH THE CONTROLS, or a fully stalled aeroplane
        keeps every one of them. Measured before this existed, on the tail
        design across ``alpha = 2, 14, 25, 40 deg``: the rolling moment from
        20 deg of aileron was ``-0.10041`` at EVERY one of them, and the roll
        damping at ``p = 0.5 rad/s`` was ``-0.04463`` at every one. A wing
        that has stopped making lift was still making full rolling moment,
        so the model could not drop a wing, could not depart and could not
        spin — which is the first region a pilot pokes.

        Differentiating the p-norm clip gives a closed form and not a fudge::

            f(x)  = x (1 + u)^(-1/n),      u = |x/CL_max|^n
            f'(x) = (1 + u)^(-(n+1)/n)

        (the algebra: ``x du/dx = n u``, so the two terms collapse to
        ``(1+u)^(-1/n) [1 - u/(1+u)]``.) It has the same property the clip
        was chosen for — it leaves normal flight ALONE: at ``CL = 0.5``
        against ``CL_max = 1.4`` it is 0.99977, and at twice ``CL_max`` it is
        0.0020. The post-stall drop multiplies it for the same reason it
        multiplies the lift.

        This is the WING's remaining slope. It is applied to the wing-carried
        channels only (:data:`Aircraft.STALL_SCALED`), never to the fin or
        the tail, which are at their own local incidence and unstalled.
        """
        if not self.enabled or self.CL_max <= 0.0:
            return 1.0
        n = float(self.sharpness)
        u = abs(float(CL_lin) / self.CL_max) ** n
        return float((1.0 + u) ** (-(n + 1.0) / n)
                     * (1.0 - self.lift_drop * self._over(alpha)))

    def extra_drag(self, alpha: float) -> float:
        if not self.enabled:
            return 0.0
        return float(self.dCD_separated * self._over(alpha))


# --------------------------------------------------------------- propulsion

@dataclass(frozen=True)
class Propulsion:
    """A scalar thrust along body x, applied AT A STATED POINT.

    WHERE THE THRUST VECTOR STARTS, because it is a question and not a
    constant. The force is ``(T, 0, 0)`` in body axes applied at

        r = (0, 0, +z_offset_m)

    i.e. on the CG's x and y, ``z_offset_m`` metres BELOW the CG (body z is
    DOWN, so below is positive). Three consequences worth stating, because
    each of them is a thing a user asked about:

    * an **x** offset would make no moment at all — ``r x F`` is zero when
      both vectors lie along x — so there is nothing to ask and nothing is
      asked. A thrust line ahead of or behind the CG is the same aeroplane;
    * a **y** offset is asymmetric thrust, which needs a second engine to be
      a question. It is fixed at zero and this is the only place that says so;
    * a **z** offset is the whole handling property: it is what makes the
      throttle a PITCH input, and ``z_offset_m = 0`` is thrust THROUGH the
      CG, which makes no pitching moment at all. That is the answerable
      "no moment" case and it is the default.

    THE SIGN, WRITTEN OUT ONCE::

        r x F = (0, 0, z) x (T, 0, 0) = (0*0 - z*0, z*T - 0*0, 0*0 - 0*T)
              = (0, +T z, 0)

    and a positive moment about body y takes +x toward -z, which is z UP,
    which is NOSE UP. So a thrust line BELOW the CG pitches nose up as power
    comes in, which is the sign a pilot expects — and it needs a PLUS here.
    It was a minus, which flew the opposite aeroplane: measured on the tail
    design, a 1 m offset at +50 % throttle took the pitch from +2.31 deg to
    -8.98 deg in two seconds.

    No propeller, no slipstream, no torque — see the module docstring.
    """

    thrust_n: float = 0.0
    z_offset_m: float = 0.0

    def force_moment(self) -> tuple[np.ndarray, np.ndarray]:
        T = float(self.thrust_n)
        F = np.array([T, 0.0, 0.0])
        M = np.array([0.0, T * float(self.z_offset_m), 0.0])
        return F, M


# ------------------------------------------------------------------- state

@dataclass(frozen=True)
class State:
    """Rigid-body state. Position in earth axes (z DOWN), the rest in body."""

    pos: np.ndarray = field(
        default_factory=lambda: np.zeros(3))          # [m] north, east, down
    quat: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))   # w x y z
    vel: np.ndarray = field(
        default_factory=lambda: np.array([30.0, 0.0, 0.0]))       # u v w
    rates: np.ndarray = field(
        default_factory=lambda: np.zeros(3))          # p q r [rad/s]

    def to_vector(self) -> np.ndarray:
        return np.concatenate([self.pos, self.quat, self.vel, self.rates])

    @staticmethod
    def from_vector(v: np.ndarray) -> "State":
        v = np.asarray(v, dtype=float)
        q = v[3:7]
        n = float(np.linalg.norm(q))
        return State(pos=v[0:3].copy(),
                     quat=(q / n if n > 0 else np.array([1.0, 0, 0, 0])),
                     vel=v[7:10].copy(), rates=v[10:13].copy())

    # -- what a pilot reads -------------------------------------------------
    @property
    def V(self) -> float:
        return float(np.linalg.norm(self.vel))

    @property
    def alpha(self) -> float:
        """[rad] — atan2(w, u), so it stays right through inverted flight."""
        return float(np.arctan2(self.vel[2], self.vel[0]))

    @property
    def beta(self) -> float:
        V = self.V
        return 0.0 if V < 1e-9 else float(np.arcsin(
            np.clip(self.vel[1] / V, -1.0, 1.0)))

    @property
    def altitude_m(self) -> float:
        return -float(self.pos[2])                     # z is DOWN

    @property
    def euler(self) -> tuple[float, float, float]:
        return dcm_to_euler(quat_to_dcm(self.quat))


def quat_to_dcm(q: np.ndarray) -> np.ndarray:
    """Body-from-earth direction cosine matrix for ``q = (w, x, y, z)``.

    Returned as a MATRIX rather than Euler angles because that is what both
    the equations of motion and the 3-D view want, and because a matrix has
    no gimbal lock to design around.
    """
    q = np.asarray(q, dtype=float)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        raise ValueError("degenerate quaternion")
    w, x, y, z = q / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)],
        [2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)],
        [2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)],
    ])


def dcm_to_euler(R: np.ndarray) -> tuple[float, float, float]:
    """``(roll, pitch, yaw)`` [rad] from a body-from-earth DCM."""
    pitch = float(-np.arcsin(np.clip(R[0, 2], -1.0, 1.0)))
    roll = float(np.arctan2(R[1, 2], R[2, 2]))
    yaw = float(np.arctan2(R[0, 1], R[0, 0]))
    return roll, pitch, yaw


def body_axis_gravity(quat: np.ndarray) -> np.ndarray:
    """Gravity resolved into body axes [m/s2]."""
    return quat_to_dcm(quat) @ np.array([0.0, 0.0, G])


# ---------------------------------------------------------------- the model

@dataclass
class Aircraft:
    """Everything the integrator needs, assembled once."""

    #: the deck channels the stall's remaining lift slope is applied to —
    #: the ones a WING carries. ``p`` is roll damping and ``aileron`` and
    #: ``flap`` are cut out of the wing, so all three die with its lift
    #: slope. ``beta``, ``q``, ``r``, ``elevator`` and ``rudder`` are the
    #: fin's and the stabiliser's, which fly at their own local incidence and
    #: are not stalled when the wing is; scaling them would model a tailplane
    #: that stops working because the wing did. ``alpha`` is NOT here either:
    #: its lift is clipped by :meth:`Stall.lift` directly, and scaling the
    #: column as well would apply the saturation twice.
    #:
    #: A ClassVar, not a field: a dataclass attribute with a default that
    #: preceded ``deck`` would make ``deck`` a non-default argument after a
    #: default one, which is an import-time error.
    STALL_SCALED: ClassVar[tuple] = ("p", "aileron", "flap")

    deck: "object"                       # dynamics.Deck
    inertia: Inertia
    rho: float = 1.225
    CD0: float = 0.025
    oswald_e: float = 0.85
    #: aspect ratio of the VERTICAL surface, for the induced drag its side
    #: force costs. None -> the aeroplane has no measured fin and the term
    #: is absent rather than assumed (:meth:`coefficients`).
    AR_vertical: float | None = None
    stall: Stall = field(default_factory=Stall)
    prop: Propulsion = field(default_factory=Propulsion)
    #: control deflections [rad], by the deck's own column names
    controls: dict = field(default_factory=dict)

    @property
    def AR(self) -> float:
        return float(self.deck.b ** 2 / self.deck.S)

    def coefficients(self, st: State) -> dict:
        """The affine deck evaluated at this state, with the stall blended in.

        Rates are non-dimensionalised on the deck's OWN reference lengths and
        on the CURRENT speed — using the reference speed instead is the
        classic way a simulation quietly stops damping as it slows down.
        """
        D = self.deck
        V = max(st.V, 1e-6)
        # THE ALPHA COLUMN IS EVALUATED AT THE DISTURBANCE, NOT AT ALPHA.
        # ``D.const`` is the coefficient set at the deck's own base state, so
        # adding ``CL_alpha * alpha`` on top of it counts the base incidence
        # twice — see dynamics.Deck. Every other channel has a base of zero.
        a_hat = D.disturbance(st.alpha) if hasattr(D, "disturbance") \
            else st.alpha
        hats = {"alpha": a_hat, "beta": st.beta,
                "p": st.rates[0] * D.b / (2 * V),
                "q": st.rates[1] * D.mac / (2 * V),
                "r": st.rates[2] * D.b / (2 * V)}
        out = dict(D.const)

        # the wing's REMAINING lift slope, from a first pass with the stall
        # off. It scales the wing-carried channels only; the fin and the
        # stabiliser fly at their own incidence and are not stalled.
        lin = dict(out)
        for name, amp in hats.items():
            col = D.columns.get(name)
            if col:
                for k, v in col.items():
                    lin[k] += v * amp
        eta = self.stall.slope_factor(-lin["CZ"], st.alpha)

        for name, amp in hats.items():
            col = D.columns.get(name)
            if col:
                s = eta if name in self.STALL_SCALED else 1.0
                for k, v in col.items():
                    out[k] += v * amp * s
        for name, amp in self.controls.items():
            col = D.columns.get(name)
            if col:
                s = eta if name in self.STALL_SCALED else 1.0
                for k, v in col.items():
                    out[k] += v * float(amp) * s

        # lift through the stall saturation, then drag rebuilt from it
        CL = self.stall.lift(-out["CZ"], st.alpha)
        CDi = CL ** 2 / (np.pi * self.AR * self.oswald_e)
        # ...AND THE SIDE FORCE COSTS INDUCED DRAG TOO. A lifting surface
        # pays for every force it makes, and the fin is a lifting surface.
        # Without this a sideslip was exactly free: measured CD = 0.03169 at
        # beta = 0, 15 AND 30 deg, so full rudder into a 30 deg skid neither
        # slowed the aeroplane nor cost it anything. Same closed form as the
        # wing's, on the FIN's own aspect ratio, which the bridge measures
        # off the surface it built. None = no vertical surface was measured,
        # and then the term is honestly absent rather than guessed.
        CDv = 0.0
        if self.AR_vertical:
            CDv = out["CY"] ** 2 / (np.pi * float(self.AR_vertical)
                                    * self.oswald_e)
        CD = self.CD0 + CDi + CDv + self.stall.extra_drag(st.alpha)
        out["CL"], out["CD"], out["CDv"] = CL, CD, CDv
        # ...and the moment of the lift the stall REMOVED. It acted at the
        # wing's own aerodynamic centre — the quarter-chord line, which the
        # lattice puts at x = 0 — so its arm about the CG is x_cg exactly,
        # and the pitch break is geometry rather than a tuned constant. It
        # is the WING's break only: the downwash change at the tail is not
        # in it, so this understates a real pitch-down rather than inventing
        # one.
        dCL = CL - (-lin["CZ"])
        out["Cm"] += dCL * float(D.x_cg) / float(D.mac)
        # back into body axes through the wind-axis rotation. THE SIDESLIP IS
        # IN IT: the drag acts along -V, and V leans out of the plane of
        # symmetry by beta, so cos(beta) belongs on the two symmetric
        # components and the drag acquires a body-y component of its own.
        # At beta = 0 every line below is what it always was.
        ca, sa = np.cos(st.alpha), np.sin(st.alpha)
        cb, sb = np.cos(st.beta), np.sin(st.beta)
        out["CX"] = -CD * ca * cb + CL * sa
        out["CY"] = out["CY"] - CD * sb
        out["CZ"] = -CD * sa * cb - CL * ca
        return out

    def forces_moments(self, st: State) -> tuple[np.ndarray, np.ndarray]:
        C = self.coefficients(st)
        D = self.deck
        qbar = 0.5 * self.rho * st.V ** 2
        F = qbar * D.S * np.array([C["CX"], C["CY"], C["CZ"]])
        M = qbar * D.S * np.array([C["Cl"] * D.b, C["Cm"] * D.mac,
                                   C["Cn"] * D.b])
        Fp, Mp = self.prop.force_moment()
        return F + Fp, M + Mp


def derivative(ac: Aircraft, st: State) -> np.ndarray:
    """State derivative — the equations of motion, written once."""
    F, M = ac.forces_moments(st)
    m = ac.inertia.mass_kg
    I = ac.inertia.tensor
    u, v, w = st.vel
    p, q, r = st.rates
    g_b = body_axis_gravity(st.quat)

    acc = F / m + g_b - np.cross(st.rates, st.vel)
    ang = np.linalg.solve(I, M - np.cross(st.rates, I @ st.rates))

    R = quat_to_dcm(st.quat)
    pos_dot = R.T @ st.vel                       # earth-frame velocity
    qw, qx, qy, qz = st.quat
    quat_dot = 0.5 * np.array([
        -qx * p - qy * q - qz * r,
        qw * p + qy * r - qz * q,
        qw * q + qz * p - qx * r,
        qw * r + qx * q - qy * p,
    ])
    del u, v, w
    return np.concatenate([pos_dot, quat_dot, acc, ang])


def step(ac: Aircraft, st: State, dt: float) -> State:
    """One classical RK4 step, with the quaternion renormalised after."""
    y = st.to_vector()
    k1 = derivative(ac, State.from_vector(y))
    k2 = derivative(ac, State.from_vector(y + 0.5 * dt * k1))
    k3 = derivative(ac, State.from_vector(y + 0.5 * dt * k2))
    k4 = derivative(ac, State.from_vector(y + dt * k3))
    return State.from_vector(y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4))


def integrate(ac: Aircraft, st: State, dt: float, n: int) -> list[State]:
    out = [st]
    for _ in range(int(n)):
        st = step(ac, st, dt)
        out.append(st)
    return out


# -------------------------------------------------------------------- trim

def trim_level(ac: Aircraft, V: float, altitude_m: float = 0.0,
               control: str = "elevator",
               alpha_bracket=(np.deg2rad(-12.0), np.deg2rad(16.0))
               ) -> tuple[State, Aircraft]:
    """Straight-and-level trim: solve for ``(alpha, control)``, then thrust.

    Lift and pitching moment are both AFFINE in ``(alpha, delta)``, exactly as
    they are in :meth:`vlm.VLM.solve_trim_moment`, so this is a 2x2 linear
    solve and not a search. Thrust then follows in closed form from the drag
    at the trimmed state.

    Returns the trimmed state and a COPY of the aircraft carrying the trim
    control setting and thrust — the input is left alone, because a trim that
    silently mutated the model would make every later comparison suspect.
    """
    D = ac.deck
    col = D.columns.get(control)
    if not col:
        raise ValueError(
            f"no control column named {control!r} in this deck; "
            f"available: {sorted(set(D.columns) - set(('alpha', 'beta', 'p', 'q', 'r')))}")
    W = ac.inertia.mass_kg * G
    qbar = 0.5 * ac.rho * float(V) ** 2

    # THE JACOBIAN, and it is the only linear algebra left here:
    #
    #     [ CZ_a  CZ_d ] [d alpha]   [ -residual_z  ]
    #     [ Cm_a  Cm_d ] [d delta] = [ -residual_m  ]
    A = np.array([[D.columns["alpha"]["CZ"], col["CZ"]],
                  [D.columns["alpha"]["Cm"], col["Cm"]]])
    if abs(np.linalg.det(A)) < 1e-14:
        raise ValueError("degenerate trim Jacobian: this control cannot trim "
                         "pitch independently of alpha")

    # NEWTON ON THE EQUATIONS OF MOTION THEMSELVES, not on a hand-resolved
    # restatement of them. The two residuals are exactly what ``derivative``
    # has to make zero for straight and level flight with no rates:
    #
    #   body z:  qbar S CZ + m g cos(alpha) = 0
    #   pitch:   Cm = 0
    #
    # (theta = alpha when the flight path is level, so the body-axis gravity
    # is ``g (-sin a, 0, cos a)``.) Driving THOSE to zero rather than a
    # ``CL + CD tan(alpha) = W/(qbar S)`` rewrite is what keeps the trim
    # exact as :meth:`Aircraft.coefficients` grows terms: the stall clip, the
    # stall's pitch break, the fin's induced drag and the deck's own
    # ``alpha_ref`` are all in it for free, and none of them has to be
    # restated here. The frozen Jacobian is the affine deck, which is a good
    # enough approximation that this converges in three or four passes.
    #
    # Starting AT the deck's own base state is a first guess and not a
    # correctness requirement — Newton reaches the same root from zero — but
    # it is the incidence the columns were measured at, so the first step is
    # the smallest one available and the frozen Jacobian is most nearly
    # right there. What IS load-bearing is that ``coefficients`` evaluates
    # the alpha column at the disturbance; this loop never sees that.
    a_ref = float(getattr(D, "alpha_ref", 0.0))
    alpha, delta = a_ref, 0.0
    Wq = W / (qbar * D.S)

    def _level(a: float) -> State:
        return State(pos=np.array([0.0, 0.0, -float(altitude_m)]),
                     quat=_quat_from_euler(0.0, float(a), 0.0),
                     vel=float(V) * np.array([np.cos(a), 0.0, np.sin(a)]),
                     rates=np.zeros(3))

    # ALPHA IS HELD IN THE BRACKET WHILE IT ITERATES, so a speed the wing
    # cannot make the lift for ends with alpha pinned at a stop and a
    # residual left over — which is the SAME answer as before ("cannot fly
    # level") rather than a Newton diagnostic. Without the clamp the
    # saturated lift curve simply has no root and the iterate walks off.
    lo, hi = float(alpha_bracket[0]), float(alpha_bracket[1])
    res = np.array([np.inf, np.inf])
    for _ in range(40):
        C = replace(ac, controls={**ac.controls, control: delta}
                    ).coefficients(_level(alpha))
        res = np.array([C["CZ"] + Wq * np.cos(alpha), C["Cm"]])
        if float(np.max(np.abs(res))) < 1e-13:
            break
        step = np.linalg.solve(A, -res)
        alpha = float(np.clip(alpha + float(step[0]), lo, hi))
        delta += float(step[1])
    if not float(np.max(np.abs(res))) < 1e-10:
        if alpha <= lo + 1e-12 or alpha >= hi - 1e-12:
            raise ValueError(
                f"trim alpha ran to {np.rad2deg(alpha):.2f} deg, the edge of "
                f"the bracket, with lift still short — this aircraft cannot "
                f"fly level at {V:.1f} m/s")
        raise ValueError(
            f"trim did not converge: residual {res} in (CZ, Cm) after 40 "
            f"Newton passes at V = {V:.1f} m/s")

    return level_at(replace(ac, controls={**ac.controls, control: delta}),
                    V, alpha, altitude_m)


def level_at(ac: Aircraft, V: float, alpha: float,
             altitude_m: float = 0.0) -> tuple[State, Aircraft]:
    """The level state at a GIVEN alpha, and the thrust that holds it there.

    The second half of :func:`trim_level`, split out because not every caller
    has a pitch control to solve for. A design that arrives here has ALREADY
    been trimmed in lift and pitching moment by the solver that produced it
    (the wing+tail families do it in one 2x2 elimination, on ``alpha`` and the
    stabiliser incidence), and its derivative deck is linearised about exactly
    that point — so re-solving for an elevator would be answering a question
    that has already been answered, with a control the deck need not even
    carry.

    WHAT IT IS FOR, AND WHY IT IS NOT OPTIONAL. The returned aircraft carries
    the THRUST. Linearising the one that went in instead is linearising about
    a state that is not an equilibrium — the aeroplane is decelerating along
    its own flight path — and the phugoid, being the exchange between speed
    and height, is precisely the mode that feels it. Measured over 24 draws of
    the `tail [free height]` box, five cross from a divergent phugoid to a
    convergent one on that difference alone.

    Thrust comes from the body-x equilibrium, NOT from drag alone. In level
    flight the body axis is pitched up by alpha relative to the flight path,
    so TWO things sit on the x-axis besides thrust: the aerodynamic x-force
    (in which the lift vector leans forward by sin alpha) and a component of
    WEIGHT (which leans aft by the same angle). Balancing drag on its own gave
    -1136 N on this family — an aeroplane that needs to be pulled backwards to
    stay level — because the lift-lean term alone already exceeds the drag at
    L/D 15.
    """
    a = float(alpha)
    st = State(pos=np.array([0.0, 0.0, -float(altitude_m)]),
               quat=_quat_from_euler(0.0, a, 0.0),
               vel=float(V) * np.array([np.cos(a), 0.0, np.sin(a)]),
               rates=np.zeros(3))
    F_aero, _M = replace(ac, prop=Propulsion()).forces_moments(st)
    g_b = body_axis_gravity(st.quat)
    thrust = -float(F_aero[0]) - ac.inertia.mass_kg * float(g_b[0])
    return st, replace(ac, prop=replace(ac.prop, thrust_n=float(thrust)))


def _quat_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array([cr * cp * cy + sr * sp * sy,
                     sr * cp * cy - cr * sp * sy,
                     cr * sp * cy + sr * cp * sy,
                     cr * cp * sy - sr * sp * cy])


# ------------------------------------------------------------------- modes

def linearise(ac: Aircraft, st: State, eps: float = 1e-6) -> np.ndarray:
    """Numerical Jacobian of the equations of motion about ``st``.

    Its eigenvalues are the phugoid, short period, roll subsidence, Dutch roll
    and spiral. Central differences on the 13-state vector; the four
    quaternion columns are kept rather than reduced to three, so the matrix
    carries one structurally zero eigenvalue (the norm direction) which the
    consumer should expect and ignore.
    """
    y0 = st.to_vector()
    n = y0.size
    J = np.zeros((n, n))
    for i in range(n):
        dy = np.zeros(n)
        dy[i] = eps
        f_p = derivative(ac, State.from_vector(y0 + dy))
        f_m = derivative(ac, State.from_vector(y0 - dy))
        J[:, i] = (f_p - f_m) / (2 * eps)
    return J
