"""The pilot's inputs: what a key press means, and how the stick moves.

Pure arithmetic — no nicegui, no aerobo, nothing to mock. The Fly view is a
picture over this; the tests are against this.

Two different things happen when a key goes down, and conflating them is why
a keyboard-flown simulation feels wrong:

* the **stick** is spring-centred. Holding an arrow drives the surface toward
  its stop at a finite rate, and RELEASING it lets the surface come back to
  centre. A key that toggled a deflection would leave the aeroplane rolling
  after the pilot let go.
* the **throttle** is not. W and S move it while held and it STAYS where it
  was left, which is what a throttle is.

Rates rather than steps, so the feel does not depend on the frame rate: at
30 Hz and at 120 Hz the same hold produces the same deflection at the same
wall-clock instant.
"""

from __future__ import annotations

#: key (lower-cased ``KeyboardKey.name``) -> (axis, direction).
#:
#: The directions are the SIGN CONVENTIONS the derivative deck was measured
#: with (``tests/test_control_surfaces.py``), not a guess:
#:
#: * positive elevator pitches NOSE DOWN, so pitching up is negative;
#: * positive rudder puts the nose to STARBOARD, so yawing right is positive;
#: * positive aileron rolls to PORT, so rolling left is positive.
#:
#: Getting one of these backwards is invisible in a screenshot and obvious
#: in the air, which is why the mapping is a table with a test on it rather
#: than three signs scattered through a view function.
KEYMAP: dict[str, tuple[str, int]] = {
    "arrowup": ("elevator", -1),        # pitch up
    "arrowdown": ("elevator", +1),      # pitch down
    "arrowleft": ("rudder", -1),        # yaw left
    "arrowright": ("rudder", +1),       # yaw right
    "a": ("aileron", +1),               # roll left
    "d": ("aileron", -1),               # roll right
    "w": ("thrust", +1),                # more thrust
    "s": ("thrust", -1),                # less thrust
}

#: axis -> full deflection [deg]. This table IS the set of sprung axes:
#: :func:`step` runs exactly these back to centre and invents no other.
LIMITS: dict[str, float] = {"elevator": 20.0, "aileron": 25.0,
                            "rudder": 25.0}

#: seconds from centre to the stop while a key is held, and back to centre
#: after it is released. The spring is quicker than the pilot, as it is on
#: anything with a spring in it.
TRAVEL_S = 0.35
RETURN_S = 0.25

#: seconds to run the throttle across its whole range
THROTTLE_S = 4.0

#: what the keys do, as one line per axis, for the legend under the picture
LEGEND = (("↑ / ↓", "pitch up / down"),
          ("← / →", "yaw left / right"),
          ("A / D", "roll left / right"),
          ("W / S", "thrust up / down"))


def key_axis(name: str) -> tuple[str, int] | None:
    """``(axis, direction)`` for a key name, or ``None`` if it is not ours.

    Case-insensitive: the browser reports ``"a"`` and ``"A"`` for the same
    physical key depending on whether shift is down, and a pilot holding
    shift is still rolling left.
    """
    return KEYMAP.get((name or "").lower())


def demand(v) -> float:
    """A held demand as the spring sees it: a number in [-1, 1].

    THE ONE LINE WHERE A SWITCH AND A STICK STOP BEING DIFFERENT THINGS.
    This used to be ``int(...)``, which was invisible while a keyboard was
    the only pilot — every demand a key can make is already an integer — and
    which floored an analogue pad to "centre or full deflection" the moment
    one was plugged in. A gamepad asking for 31 % of the aileron would have
    got 0 %, and the bug would have looked like a dead controller rather
    than like a cast.

    The clip is not a limit on the pilot: it is what makes two devices
    safely ADDABLE (:mod:`gui.v4.pad` and the keyboard hold the same axis),
    since a sum of two full demands is still one full demand.
    """
    return min(1.0, max(-1.0, float(v or 0.0)))


def advance_axis(deflection: float, hold: float, dt: float, limit: float,
                 *, travel_s: float = TRAVEL_S,
                 return_s: float = RETURN_S) -> float:
    """One frame of a spring-centred axis, in degrees.

    ``hold`` is the demand in [-1, 1] — the FRACTION of the stop the pilot
    is asking for, not the deflection. A key can only ever say -1, 0 or +1;
    an analogue stick (:mod:`gui.v4.pad`) says 0.31, and the law was already
    written for it. With ``hold == 0`` the target is centre, which is the
    spring. Never overshoots its target, so a large ``dt`` snaps to the stop
    rather than flying past it.

    The RATE does not scale with the demand: a half-deflection is reached in
    half the time, not at half the speed. That is what a spring against a
    stick does, and it is why a small correction is quick.
    """
    target = float(hold) * limit
    seconds = travel_s if hold else return_s
    step = abs(limit) / max(seconds, 1e-6) * max(dt, 0.0)
    if deflection < target:
        return min(target, deflection + step)
    if deflection > target:
        return max(target, deflection - step)
    return float(deflection)


def advance_throttle(thrust: float, hold: float, dt: float,
                     lo: float, hi: float,
                     *, seconds: float = THROTTLE_S) -> float:
    """One frame of the throttle, in newtons. Held, not sprung.

    ``hold`` is fractional for the same reason the stick's is: a trigger
    squeezed a third of the way opens the throttle at a third of the rate,
    which is the difference between trimming the last newton on and
    hunting it with a key.
    """
    if hi <= lo:
        return float(min(max(thrust, lo), hi if hi > lo else lo))
    step = float(hold) * (hi - lo) / max(seconds, 1e-6) * max(dt, 0.0)
    return float(min(hi, max(lo, thrust + step)))


def step(stick: dict, hold: dict, dt: float,
         limits: dict | None = None) -> dict:
    """Advance every spring axis in ``stick`` from the held directions.

    Mutates ``stick`` (it is the session's own sub-dict) and returns it, so
    a caller can read the new deflections without a second lookup. Only the
    axes in ``limits`` are touched: anything else in the dict is somebody
    else's, and a loop that is not connected to it must not move it.
    """
    lim = LIMITS if limits is None else limits
    for axis, limit in lim.items():
        stick[axis] = advance_axis(float(stick.get(axis, 0.0)),
                                   demand(hold.get(axis, 0)), dt, limit)
    return stick
