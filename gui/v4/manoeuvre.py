"""Canned excitations: the five stability manoeuvres, as data.

A mode is a prediction. The Modes view says the phugoid of this design has
a period of 14.6 s and a damping ratio of 0.07, and the only way to believe
that is to watch the aeroplane do it — which by hand means holding a
2-degree elevator pulse for exactly half a second and then not touching
anything for a minute. Nobody flies that with a keyboard, so the pulse is a
script.

WHAT EACH ONE EXCITES, AND WHY IT IS THAT SHAPE:

phugoid
    a SPEED error at constant incidence. The phugoid is an exchange of
    height for speed at nearly constant alpha, so the clean excitation is
    to add speed along the flight path and let go. A pitch pulse would
    excite the short period as well and the first four seconds of the trace
    would be somebody else's mode.
short period
    an elevator DOUBLET — one way, then the other, then centre. The doublet
    ends with the aeroplane back at its trim attitude and with the speed
    barely changed, which is exactly the initial condition the short period
    wants and the phugoid does not.
dutch roll
    a rudder doublet, for the same reason on the other axis: it leaves
    sideslip and yaw rate behind and almost no bank, so the roll subsidence
    and the spiral stay quiet.
roll subsidence
    an aileron STEP, held. The roll rate rises to its steady value with a
    time constant that is the mode, and the steady rate itself is the roll
    control power — two numbers from one input.
spiral
    a bank angle, and then nothing at all. This is the one manoeuvre with
    no control input in it: the whole question is what a released stick
    does with 10 degrees of bank, and any input at all answers a different
    one.

Pure data and pure arithmetic — no nicegui, no session, no solver. The
initial perturbation is returned as a DESCRIPTION (four numbers) rather than
as a State, so a test can assert what a manoeuvre asks for without building
an aeroplane.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["Manoeuvre", "MANOEUVRES", "BY_KEY", "inputs_at",
           "duration_s", "perturbation"]


@dataclass(frozen=True)
class Manoeuvre:
    """One named excitation.

    ``script`` is a tuple of ``(t_from, t_to, axis, degrees)`` windows. They
    are applied in order and the LAST window covering a time wins, so an
    overlapping pair is a well-defined thing to write rather than a bug.
    """

    key: str
    label: str
    mode: str                  # the mode in gui.v4.modes this shows
    hint: str
    dV_frac: float = 0.0       # + = faster along the flight path
    dalpha_deg: float = 0.0
    dbeta_deg: float = 0.0
    dbank_deg: float = 0.0
    script: tuple = ()
    watch_s: float = 60.0      # how long the interesting part lasts

    @property
    def has_input(self) -> bool:
        return bool(self.script)

    @property
    def axes(self) -> tuple:
        """The control axes this manoeuvre's script actually moves."""
        return tuple(dict.fromkeys(axis for _f, _t, axis, _d in self.script))

    def flyable_on(self, columns) -> bool:
        """Can this deck fly it?

        A script written into an axis the deck has not got is not a degraded
        manoeuvre — it is a flat trace printed under a mode's numbers, which
        reads as "the aeroplane did nothing" rather than "the aeroplane has
        no rudder". A craft with no control surface (a foiler) can still fly
        the two that need no input at all.
        """
        cols = set(columns or ())
        return all(axis in cols for axis in self.axes)


#: half-amplitude and half-width of the doublets. Small enough that the
#: linear deck is still the aeroplane being flown — the point of the
#: manoeuvre is to see the MODE, and a 15-degree pull shows the stall model
#: instead.
_PULSE_DEG = 2.0
_PULSE_S = 0.5

MANOEUVRES: tuple[Manoeuvre, ...] = (
    Manoeuvre(
        key="phugoid", label="Phugoid", mode="phugoid",
        hint="+10 % speed at trim incidence, stick free. Watch the height "
             "and the speed trade, slowly and almost undamped.",
        dV_frac=0.10, watch_s=90.0),
    Manoeuvre(
        key="short_period", label="Short period", mode="short period",
        hint="An elevator doublet. It ends back at the trim attitude, so "
             "what is left is alpha and pitch rate — the short period, and "
             "very little phugoid.",
        script=((0.0, _PULSE_S, "elevator", -_PULSE_DEG),
                (_PULSE_S, 2 * _PULSE_S, "elevator", +_PULSE_DEG)),
        watch_s=15.0),
    Manoeuvre(
        key="dutch_roll", label="Dutch roll", mode="dutch roll",
        hint="A rudder doublet. Leaves sideslip and yaw rate and almost no "
             "bank — the nose scribes an ellipse and the wings follow it.",
        script=((0.0, _PULSE_S, "rudder", +5.0),
                (_PULSE_S, 2 * _PULSE_S, "rudder", -5.0)),
        watch_s=25.0),
    Manoeuvre(
        key="roll", label="Roll subsidence", mode="roll subsidence",
        hint="An aileron step, HELD for two seconds. The roll rate rises to "
             "a steady value: the rise is the mode, the value is the roll "
             "control power.",
        script=((0.0, 2.0, "aileron", 10.0),),
        watch_s=10.0),
    Manoeuvre(
        key="spiral", label="Spiral", mode="spiral",
        hint="10 degrees of bank and NOTHING else. A convergent spiral rolls "
             "the wings back level; a divergent one tightens, and the time "
             "to double is how long the pilot has to notice.",
        dbank_deg=10.0, watch_s=90.0),
)

#: keyed, for the stage
BY_KEY: dict[str, Manoeuvre] = {m.key: m for m in MANOEUVRES}


def duration_s(m: Manoeuvre) -> float:
    """When the script stops driving. Zero for a released-stick manoeuvre."""
    return max((float(w[1]) for w in m.script), default=0.0)


def inputs_at(m: Manoeuvre, t: float) -> dict:
    """The commanded surface deflections [deg] at ``t`` seconds in.

    Empty once the script has run out, which is the signal to hand the axis
    back to the spring rather than to hold the last value for ever.
    """
    out: dict[str, float] = {}
    for t0, t1, axis, deg in m.script:
        if float(t0) <= float(t) < float(t1):
            out[str(axis)] = float(deg)
    return out


def perturbation(vel, quat, m: Manoeuvre):
    """``(vel, quat)`` for the aeroplane at the instant a manoeuvre starts.

    ``vel`` is body-axis ``[u, v, w]`` and ``quat`` is ``[w, x, y, z]``, the
    two things a mode excitation has to touch: the aerodynamic angles live
    in the first and the bank angle in the second. Position and body rates
    are left exactly as they were, so the aeroplane starts where it was and
    is not also spinning.

    The bank is applied as a BODY-axis roll — ``q * q_x`` rather than
    ``q_x * q`` — so "10 degrees of bank" means 10 degrees about the
    aeroplane's own nose axis. Pre-multiplying would roll it about the
    earth's north axis, which is the same thing only in level flight and is
    a different manoeuvre the moment there is any pitch attitude.
    """
    import numpy as np

    u, v, w = (float(x) for x in vel)
    V = math.sqrt(u * u + v * v + w * w)
    if V > 1e-9:
        alpha = math.atan2(w, u)
        beta = math.asin(max(-1.0, min(1.0, v / V)))
        V *= 1.0 + float(m.dV_frac)
        alpha += math.radians(m.dalpha_deg)
        beta += math.radians(m.dbeta_deg)
        # the standard decomposition, so a pure alpha change leaves beta
        # alone and a pure beta change leaves alpha alone
        u = V * math.cos(alpha) * math.cos(beta)
        v = V * math.sin(beta)
        w = V * math.sin(alpha) * math.cos(beta)
    q = np.asarray(quat, dtype=float)
    if abs(m.dbank_deg) > 1e-12:
        h = math.radians(m.dbank_deg) / 2.0
        r = np.array([math.cos(h), math.sin(h), 0.0, 0.0])
        q = np.array([
            q[0] * r[0] - q[1] * r[1] - q[2] * r[2] - q[3] * r[3],
            q[0] * r[1] + q[1] * r[0] + q[2] * r[3] - q[3] * r[2],
            q[0] * r[2] - q[1] * r[3] + q[2] * r[0] + q[3] * r[1],
            q[0] * r[3] + q[1] * r[2] - q[2] * r[1] + q[3] * r[0]])
        q = q / max(float(np.linalg.norm(q)), 1e-12)
    return np.array([u, v, w]), q
