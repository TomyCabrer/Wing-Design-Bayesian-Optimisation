"""The pilot's other hand: a DualSense (PS5) pad, as demands.

Pure arithmetic — no nicegui, no aerobo, nothing to mock. The browser polls
the pad and the server receives a demand; both ends compute that demand with
the numbers in THIS module, so there is one law and not two that drift.

WHY A PAD IS NOT A KEYBOARD, and why the difference is the whole point.

A key is a switch: :mod:`gui.v4.stick` reads it as -1, 0 or +1 and the spring
runs the surface toward that stop at a fixed rate. Nothing about a keyboard
can ask for HALF an aileron, so a keyboard pilot flies by tapping — and a
tapped roll is why the aeroplane felt like a flip-book to fly even after the
picture stopped being one.

A stick is a position. ``axes[2] = 0.31`` is a demand for 31 % of the stop,
and the same spring law serves it unchanged: ``advance_axis`` was already
written as ``target = hold * limit`` and only ever LOOKED like a three-state
control because an ``int()`` upstream made it one. So the pad is not a second
control path bolted beside the keyboard; it is the same demand with the
resolution the keyboard could not express, and the analogue value goes
straight through to the target the spring is chasing.

THE THREE NUMBERS A PAD NEEDS, and what each one is for:

* a **dead zone**, because a DualSense at rest does not read zero. Worn
  sticks idle around 0.02-0.08, and without the cut the aeroplane rolls
  slowly forever with nobody touching it — which reads as a trim fault and
  is not one.
* an **exponent**, because a stop is 25 deg and a pilot spends the flight
  within about 3 of centre. A linear stick gives that band a tenth of its
  travel; ``EXPO`` spends more of the stick on the deflections that are
  actually flown, and it is a shape, not a limit — full travel still reaches
  the stop exactly.
* a **quantum**, because this demand crosses a websocket. The browser sends
  only when the rounded value CHANGES, so a stick held anywhere — including
  held at centre — costs nothing at all. It is the same economy the pose
  push is built on (:mod:`gui.v4.live`), applied to the traffic going the
  other way.

THE SIGNS ARE THE DERIVATIVE DECK'S, not a guess, and they are the same
conventions ``stick.KEYMAP`` is written against: positive elevator pitches
nose DOWN, positive rudder puts the nose to STARBOARD, positive aileron
rolls to PORT. A gamepad's own Y axis is negative UPWARDS, so pushing the
stick forward — nose down — arrives as ``-1`` and leaves here as ``+1``.
"""

from __future__ import annotations

import math

__all__ = ["AXES", "TRIGGER_UP", "TRIGGER_DOWN", "BUTTON_ACTIONS",
           "DEADZONE", "EXPO", "TRIGGER_DEADZONE", "QUANTUM", "PRESS",
           "MAX_HZ", "LEGEND", "curve", "trigger", "quantise", "holds",
           "actions", "js_config"]

#: gamepad axis index -> (control axis, sign), in the browser's STANDARD
#: mapping — the one Chrome reports for a DualSense over USB or Bluetooth.
#:
#: Right stick flies the aeroplane and the left stick's X steers it on the
#: ground and in yaw, which is the layout every flight game uses; putting
#: pitch on the left stick would be a defensible choice and an unfamiliar
#: one, and a control nobody expects is a control nobody finds.
AXES: dict[int, tuple[str, int]] = {
    2: ("aileron", -1),      # right stick X: right = roll right = aileron -1
    3: ("elevator", -1),     # right stick Y: forward (-) = nose down = +1
    0: ("rudder", +1),       # left stick X: right = nose to starboard
}

#: the two analogue triggers, as a THROTTLE — held, not sprung, exactly as W
#: and S are. R2 opens it and L2 closes it, and because both are analogue the
#: rate is proportional: a feathered trigger trims the last newton on, where
#: a key can only run the whole range at one speed.
TRIGGER_UP = 7          # R2
TRIGGER_DOWN = 6        # L2

#: button index -> what it does, on the PRESS edge only. Three, because a
#: pad that can start, stop and re-trim the flight is a pad you never have
#: to put down; anything more belongs on the panel where it can be labelled.
BUTTON_ACTIONS: dict[int, str] = {
    0: "fly",        # cross  — fly / pause
    1: "reset",      # circle — re-trim and put it back at the mission point
    9: "view",       # options — chase camera / engineering camera
}

#: below this the stick is centred. Measured against the resting noise of a
#: worn DualSense (0.02-0.08), with room over it.
DEADZONE = 0.10

#: shape of the stick's response. 1.0 is linear; larger spends more travel
#: near centre. Not a limit: ``curve(1.0) == 1.0`` for every exponent.
EXPO = 1.7

#: the triggers rest at exactly 0 and only need noise rejection, so their cut
#: is small and their response is LINEAR — a throttle rate that curved would
#: make the same trigger position mean two different things at two states of
#: the engine, and a throttle is the one control that must not surprise.
TRIGGER_DEADZONE = 0.06

#: what the wire can distinguish. 0.02 of full stick is 0.5 deg of aileron —
#: below anything a pilot can hold, and it takes a stick swept corner to
#: corner from 200 messages to 100.
QUANTUM = 0.02

#: a button counts as down above this. The face buttons are digital, but the
#: API reports them as floats and the triggers share the same array.
PRESS = 0.5

#: ceiling on how often the browser may report, in Hz. The pad is polled
#: every animation frame — 120 on this display — and the demand is smoothed
#: by the spring anyway, so reporting faster than the physics push (FPS in
#: the stage, 30) buys nothing and costs a message.
MAX_HZ = 30.0

#: what the pad does, as one line per control, for the panel under the
#: picture. Same shape as ``stick.LEGEND`` because it is shown beside it.
LEGEND = (("Right stick", "pitch / roll"),
          ("Left stick ← →", "yaw"),
          ("R2 / L2", "thrust up / down"),
          ("✕", "fly / pause"),
          ("○", "reset"),
          ("Options", "chase / engineering camera"))


def curve(raw: float, *, dead: float = DEADZONE,
          expo: float = EXPO) -> float:
    """One stick axis, raw [-1, 1] -> demand [-1, 1].

    Dead zone first and RESCALED after it, so the first millimetre outside
    the cut asks for nothing rather than for ``dead`` worth of aileron: a
    step at the edge of the dead zone is felt as a notch in the control and
    is the usual reason a dead zone is described as making a pad feel worse.
    """
    v = float(raw)
    m = min(1.0, abs(v))
    if m <= dead:
        return 0.0
    m = (m - dead) / (1.0 - dead)
    return math.copysign(m ** expo, v)


def trigger(raw: float, *, dead: float = TRIGGER_DEADZONE) -> float:
    """One analogue trigger, raw [0, 1] -> demand [0, 1]. Linear."""
    v = min(1.0, max(0.0, float(raw)))
    return 0.0 if v <= dead else (v - dead) / (1.0 - dead)


def quantise(v: float, *, q: float = QUANTUM) -> float:
    """The value as the wire carries it. Rounding is what makes "unchanged"
    a decidable question, and therefore what makes a held stick free."""
    if q <= 0.0:
        return float(v)
    return round(float(v) / q) * q


def holds(axes, buttons) -> dict:
    """A pad reading -> the demand dict the stage adds to ``F['hold']``.

    ``axes`` is the gamepad's axis array and ``buttons`` its button VALUES
    (floats, not the objects — the browser sends ``b.value``). Short arrays
    are tolerated and read as zero: a pad that reports six axes is not a
    reason to raise, it is a reason to fly straight.
    """
    def ax(i):
        return float(axes[i]) if axes is not None and i < len(axes) else 0.0

    def bt(i):
        return (float(buttons[i])
                if buttons is not None and i < len(buttons) else 0.0)

    out = {name: quantise(sign * curve(ax(i)))
           for i, (name, sign) in AXES.items()}
    out["thrust"] = quantise(trigger(bt(TRIGGER_UP))
                             - trigger(bt(TRIGGER_DOWN)))
    return out


def actions(now, before=None) -> tuple:
    """Which buttons were just PRESSED, in index order.

    Edges, not states: a held cross must start the flight once and not
    thirty times a second, which is the same reason the keyboard is built
    with ``repeating=False``.
    """
    def down(vals, i):
        return (vals is not None and i < len(vals)
                and float(vals[i]) >= PRESS)

    return tuple(name for i, name in sorted(BUTTON_ACTIONS.items())
                 if down(now, i) and not down(before, i))


def js_config() -> dict:
    """The same numbers, for the browser half.

    Sent rather than duplicated in the script: a constant written twice is a
    constant that will be changed once. ``tests/test_v4_pad.py`` runs the
    shipped script under node against :func:`holds` to pin that the two
    agree at real stick positions.
    """
    return {"axes": {str(i): [name, sign] for i, (name, sign) in AXES.items()},
            "up": TRIGGER_UP, "down": TRIGGER_DOWN,
            "buttons": {str(i): n for i, n in BUTTON_ACTIONS.items()},
            "dead": DEADZONE, "expo": EXPO, "tdead": TRIGGER_DEADZONE,
            "q": QUANTUM, "press": PRESS, "hz": MAX_HZ}
