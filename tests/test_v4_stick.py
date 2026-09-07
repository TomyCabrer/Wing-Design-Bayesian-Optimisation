"""The pilot's inputs — gui.v4.stick.

Pure arithmetic, so these are outcome tests with no shell anywhere near
them. What they pin is the two things that are invisible in a screenshot and
obvious in the air: which key does what, and the fact that letting go of a
key brings the surface back.
"""

import pytest

from gui.v4 import stick as stk


# --------------------------------------------------------------- the keys

@pytest.mark.parametrize("key,axis,direction", [
    ("ArrowUp", "elevator", -1),        # pitch up
    ("ArrowDown", "elevator", +1),      # pitch down
    ("ArrowLeft", "rudder", -1),        # yaw left
    ("ArrowRight", "rudder", +1),       # yaw right
    ("a", "aileron", +1),               # roll left
    ("d", "aileron", -1),               # roll right
    ("w", "thrust", +1),
    ("s", "thrust", -1),
])
def test_each_key_is_the_axis_and_the_direction_it_was_asked_for(
        key, axis, direction):
    assert stk.key_axis(key) == (axis, direction)


def test_the_directions_agree_with_the_MEASURED_sign_conventions():
    """Not a convention chosen here — the deck's, measured in
    tests/test_control_surfaces.py.

    Positive elevator pitches nose DOWN, positive rudder puts the nose to
    STARBOARD, positive aileron rolls to PORT. So pitching up is negative
    elevator, yawing right is positive rudder, rolling left is positive
    aileron. One of these backwards is a simulation that flies mirrored.
    """
    assert stk.key_axis("arrowup")[1] == -stk.key_axis("arrowdown")[1]
    assert stk.key_axis("arrowright")[1] > 0        # +rudder -> nose right
    assert stk.key_axis("a")[1] > 0                 # +aileron -> roll port
    assert stk.key_axis("d")[1] == -stk.key_axis("a")[1]


def test_a_shifted_key_is_the_same_key():
    """The browser reports "A" while shift is down, and a pilot holding
    shift is still rolling left."""
    assert stk.key_axis("A") == stk.key_axis("a")
    assert stk.key_axis("ArrowUp") == stk.key_axis("arrowup")


def test_a_key_that_is_not_ours_is_none():
    for key in ("Enter", "Escape", "q", "", None, "Tab"):
        assert stk.key_axis(key) is None


def test_every_stick_axis_on_the_keyboard_has_a_limit():
    """A key mapped to an axis with no travel limit would deflect for ever."""
    for axis, _d in stk.KEYMAP.values():
        assert axis == "thrust" or axis in stk.LIMITS, axis


def test_the_sprung_axes_are_exactly_the_three_the_deck_carries():
    """``LIMITS`` IS the set of sprung axes. A fourth in here would be
    driven back to centre by ``step`` every frame whether the deck had a
    column for it or not."""
    assert set(stk.LIMITS) == {"elevator", "aileron", "rudder"}
    assert {a for a, _d in stk.KEYMAP.values()} == \
        {"elevator", "aileron", "rudder", "thrust"}


# -------------------------------------------------------------- the spring

def test_holding_deflects_toward_the_stop_and_stops_there():
    d, limit = 0.0, 25.0
    for _ in range(1000):
        d = stk.advance_axis(d, +1, 0.01, limit)
    assert d == pytest.approx(limit)


def test_it_never_overshoots_even_on_one_enormous_step():
    assert stk.advance_axis(0.0, +1, 99.0, 25.0) == pytest.approx(25.0)
    assert stk.advance_axis(0.0, -1, 99.0, 25.0) == pytest.approx(-25.0)


def test_releasing_brings_it_back_to_EXACTLY_centre():
    d = stk.advance_axis(0.0, +1, stk.TRAVEL_S, 25.0)
    assert d == pytest.approx(25.0)
    for _ in range(200):
        d = stk.advance_axis(d, 0, 0.01, 25.0)
    assert d == 0.0, "a spring that stops near centre leaves a standing roll"


def test_the_deflection_is_a_RATE_so_the_frame_rate_does_not_change_the_feel():
    """The same hold for the same wall-clock time gives the same angle at
    30 Hz and at 240 Hz. A per-frame step would give eight times as much."""
    slow = fast = 0.0
    for _ in range(9):                      # 9 x 1/30 s = 0.30 s
        slow = stk.advance_axis(slow, +1, 1 / 30, 20.0)
    for _ in range(72):                     # 72 x 1/240 s = 0.30 s
        fast = stk.advance_axis(fast, +1, 1 / 240, 20.0)
    assert slow == pytest.approx(fast, rel=1e-12)


def test_full_travel_takes_the_stated_time():
    d, dt, n = 0.0, 1e-3, round(stk.TRAVEL_S / 1e-3)
    for _ in range(n):
        d = stk.advance_axis(d, +1, dt, 20.0)
    assert d == pytest.approx(20.0, rel=1e-9)


def test_step_springs_every_limited_axis_and_touches_nothing_else():
    """Only the axes in ``limits`` are flown back to centre. Anything
    else in the dict belongs to somebody the loop is not connected to."""
    stick = {"elevator": 5.0, "aileron": -8.0, "rudder": 2.0,
             "trim": 30.0}
    hold = {"elevator": 0, "aileron": 0, "rudder": 0}
    stk.step(stick, hold, 10.0)
    assert stick["elevator"] == 0.0
    assert stick["aileron"] == 0.0
    assert stick["rudder"] == 0.0
    assert stick["trim"] == 30.0, "step moved an axis it has no limit for"


def test_step_drives_only_the_axis_that_is_held():
    stick = dict.fromkeys(("elevator", "aileron", "rudder"), 0.0)
    stk.step(stick, {"aileron": +1}, 0.05)
    assert stick["aileron"] > 0.0
    assert stick["elevator"] == 0.0 and stick["rudder"] == 0.0


# ------------------------------------------------------------ the throttle

def test_the_throttle_is_HELD_not_sprung():
    """W and S move it; letting go leaves it where it was. A sprung throttle
    would idle the moment the pilot stopped pressing."""
    t = stk.advance_throttle(0.0, +1, 1.0, 0.0, 400.0)
    assert t > 0.0
    assert stk.advance_throttle(t, 0, 5.0, 0.0, 400.0) == pytest.approx(t)


def test_the_throttle_clamps_at_both_stops():
    assert stk.advance_throttle(0.0, -1, 99.0, 0.0, 400.0) == 0.0
    assert stk.advance_throttle(400.0, +1, 99.0, 0.0, 400.0) == 400.0


def test_the_throttle_crosses_its_range_in_the_stated_time():
    t, dt = 0.0, 1e-3
    for _ in range(round(stk.THROTTLE_S / dt)):
        t = stk.advance_throttle(t, +1, dt, 0.0, 400.0)
    assert t == pytest.approx(400.0, rel=1e-9)


def test_a_degenerate_band_does_not_divide_by_zero():
    assert stk.advance_throttle(5.0, +1, 1.0, 3.0, 3.0) == pytest.approx(3.0)


# ---------------------------------------------------------------- the legend

def test_the_legend_names_every_key_group_the_user_asked_for():
    text = " ".join(k + " " + v for k, v in stk.LEGEND).lower()
    for word in ("pitch", "yaw", "roll", "thrust"):
        assert word in text
    assert "↑" in text and "←" in text and "a" in text and "w" in text
