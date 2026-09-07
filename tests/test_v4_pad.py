"""The gamepad — gui.v4.pad, and the browser half that reads it.

Three things are worth a test here and the rest is arithmetic anybody can
read:

* the SIGNS, because a reversed axis is invisible in a screenshot and
  obvious in the air — the same reason ``stick.KEYMAP`` is a table with a
  test on it rather than three signs scattered through a view;
* that the demand is genuinely ANALOGUE all the way to the spring. The law
  was always ``target = hold * limit``; an ``int()`` in ``stick.step`` made
  it a three-state control, and nothing could see that until a pad arrived;
* that the browser computes the SAME demand this module does. The script is
  run under node against these functions rather than grepped, because a
  script that returns early keeps every word a grep would look for.
"""

import json

import pytest

import tests.v4_js as jsx
from gui.v4 import live, pad
from gui.v4 import hud, stick as stk


# --------------------------------------------------------------- the curve

def test_a_resting_stick_asks_for_nothing():
    """A DualSense at rest does not read zero. Without the cut the aeroplane
    rolls slowly for ever with nobody touching it, which reads as a trim
    fault and is not one."""
    for raw in (0.0, 0.02, 0.05, 0.09, -0.09):
        assert pad.curve(raw) == 0.0, raw


def test_the_dead_zone_has_no_step_at_its_edge():
    """Rescaled after the cut, not merely cut: a stick that jumped to
    ``DEADZONE`` worth of aileron the moment it left the zone would be felt
    as a notch, and that is the usual reason a dead zone is blamed for
    making a pad worse."""
    just_out = pad.curve(pad.DEADZONE + 1e-6)
    assert 0.0 < just_out < 1e-3


def test_full_travel_still_reaches_the_stop():
    """The exponent is a SHAPE, not a limit."""
    assert pad.curve(1.0) == pytest.approx(1.0)
    assert pad.curve(-1.0) == pytest.approx(-1.0)
    assert pad.curve(1.4) == pytest.approx(1.0), "over-range must clamp"


def test_the_curve_spends_more_stick_near_centre():
    """The point of the exponent: half travel asks for LESS than half the
    stop, so the deflections actually flown get more of the stick."""
    assert pad.curve(0.5) < 0.5
    # ...and it is monotone, or the stick would fight the pilot somewhere
    vals = [pad.curve(x / 20.0) for x in range(21)]
    assert all(b >= a for a, b in zip(vals, vals[1:]))


def test_a_trigger_is_linear_and_starts_at_zero():
    """A throttle rate that curved would make one trigger position mean two
    different things at two states of the engine."""
    assert pad.trigger(0.0) == 0.0
    assert pad.trigger(pad.TRIGGER_DEADZONE) == 0.0
    assert pad.trigger(1.0) == pytest.approx(1.0)
    mid = pad.trigger(0.5)
    assert pad.trigger(0.75) - mid == pytest.approx(mid - pad.trigger(0.25))


# --------------------------------------------------------------- the signs

def _axes(lx=0.0, ly=0.0, rx=0.0, ry=0.0):
    return [lx, ly, rx, ry]


def _buttons(pressed=None):
    b = [0.0] * 17
    for i, v in (pressed or {}).items():
        b[int(i)] = v
    return b


def test_pushing_the_stick_forward_pitches_the_nose_down():
    """A gamepad's Y axis is negative UPWARDS, and positive elevator is nose
    DOWN (the convention the derivative deck was measured with). So forward
    stick — which is nose down on anything with a stick — must arrive as a
    POSITIVE elevator demand."""
    h = pad.holds(_axes(ry=-1.0), _buttons())
    assert h["elevator"] == pytest.approx(1.0)
    assert pad.holds(_axes(ry=+1.0), _buttons())["elevator"] == pytest.approx(-1.0)


def test_stick_right_rolls_right_and_that_is_a_negative_aileron():
    """``KEYMAP['d']`` is ``('aileron', -1)`` — positive aileron rolls to
    PORT. The pad has to agree with the keyboard or the same aeroplane
    answers two pilots differently."""
    h = pad.holds(_axes(rx=+1.0), _buttons())
    assert h["aileron"] == pytest.approx(-1.0)
    assert stk.KEYMAP["d"] == ("aileron", -1)


def test_the_left_stick_yaws_the_way_the_arrow_keys_do():
    h = pad.holds(_axes(lx=+1.0), _buttons())
    assert h["rudder"] == pytest.approx(1.0)
    assert stk.KEYMAP["arrowright"] == ("rudder", +1)


def test_the_triggers_are_a_throttle_and_they_oppose():
    up = pad.holds(_axes(), _buttons({pad.TRIGGER_UP: 1.0}))
    dn = pad.holds(_axes(), _buttons({pad.TRIGGER_DOWN: 1.0}))
    both = pad.holds(_axes(), _buttons({pad.TRIGGER_UP: 1.0,
                                        pad.TRIGGER_DOWN: 1.0}))
    assert up["thrust"] == pytest.approx(1.0)
    assert dn["thrust"] == pytest.approx(-1.0)
    assert both["thrust"] == pytest.approx(0.0)


def test_a_short_reading_is_flown_straight_and_not_raised_over():
    """A pad that reports six axes is not a reason to raise."""
    h = pad.holds([0.0, 0.0], [])
    assert set(h) == {"elevator", "aileron", "rudder", "thrust"}
    assert all(v == 0.0 for v in h.values())
    assert pad.holds(None, None)["elevator"] == 0.0


# ------------------------------------------------------------- the buttons

def test_a_button_fires_on_the_press_and_not_while_it_is_held():
    """A held cross must start the flight once, not thirty times a second —
    the same reason the keyboard is built with ``repeating=False``."""
    down = _buttons({0: 1.0})
    assert pad.actions(down, None) == ("fly",)
    assert pad.actions(down, down) == ()
    assert pad.actions(_buttons(), down) == (), "a release is not an action"


def test_two_presses_in_one_frame_both_arrive():
    now = _buttons({0: 1.0, 9: 1.0})
    assert set(pad.actions(now, None)) == {"fly", "view"}


# ------------------------------------------------- analogue, to the spring

def test_the_spring_reads_a_fraction_and_not_a_switch():
    """THE REGRESSION THIS MODULE EXISTS AGAINST. ``stick.step`` cast the
    demand to ``int``, which is a no-op for every demand a key can make and
    silently floored a 31 % stick to nothing."""
    assert stk.demand(0.31) == pytest.approx(0.31)
    stick, hold = {"aileron": 0.0}, {"aileron": 0.4}
    # long enough to reach whatever it is chasing
    stk.step(stick, hold, 10.0, limits={"aileron": 25.0})
    assert stick["aileron"] == pytest.approx(0.4 * 25.0)


def test_a_fraction_reaches_its_target_sooner_not_slower():
    """A spring against a stick: a half deflection is reached in half the
    time, at the same rate. That is why a small correction is quick."""
    half = stk.advance_axis(0.0, 0.5, stk.TRAVEL_S / 2.0, 25.0)
    full = stk.advance_axis(0.0, 1.0, stk.TRAVEL_S / 2.0, 25.0)
    assert half == pytest.approx(0.5 * 25.0)      # already there
    assert full == pytest.approx(0.5 * 25.0)      # half way to the stop


def test_two_devices_on_one_axis_are_added_and_clipped():
    assert stk.demand(1.0 + 0.6) == pytest.approx(1.0)
    assert stk.demand(-1.0 - 0.6) == pytest.approx(-1.0)
    assert stk.demand(None) == 0.0


# ------------------------------------------------------- the browser half

PAD_CFG = pad.js_config()
CFG = dict(jsx.CFG, pad=PAD_CFG)

#: a navigator whose pad is scripted frame by frame, plus a recorder for
#: what the driver emitted. Prefixed to the shipped script, so what runs
#: below is gui/v4/live.py's own source and not a copy of it.
_PAD_STUB = """
globalThis.__emitted = [];
globalThis.emitEvent = (name, payload) => {
  globalThis.__emitted.push([name, payload]);
};
globalThis.__pads = [null];
// node >= 21 ships a getter-only `navigator`, so it is REDEFINED rather
// than assigned — the shipped script reads navigator.getGamepads and has
// to find the scripted one.
Object.defineProperty(globalThis, 'navigator', {
  value: {getGamepads: () => globalThis.__pads},
  configurable: true, writable: true,
});
globalThis.setPad = (axes, buttons) => {
  globalThis.__pads = (axes === null) ? [null] : [{
    connected: true, axes: axes,
    buttons: (buttons || []).map((v) => ({value: v, pressed: v >= 0.5})),
  }];
};
"""


def _js(script, *, cfg=None):
    return jsx.run(cfg or CFG, script, live_js=_PAD_STUB + live._JS,
                   apply_js=hud.APPLY_JS)


@pytest.mark.skipif(jsx.NODE is None, reason="node is not installed")
@pytest.mark.parametrize("axes,buttons", [
    ([0.0, 0.0, 0.0, 0.0], [0.0] * 17),
    ([0.4, 0.0, -0.7, 0.31], [0.0] * 17),
    ([-1.0, 0.0, 1.0, -1.0], [0.0] * 17),
    ([0.05, 0.0, 0.09, -0.05], [0.0] * 17),          # inside the dead zone
    ([0.0, 0.0, 0.62, 0.0], [0.0] * 6 + [0.3, 0.8] + [0.0] * 9),
])
def test_the_browser_computes_the_same_demand_python_does(axes, buttons):
    """ONE LAW, NOT TWO. The constants are sent to the script (``cfg.pad``)
    precisely so this comparison can be exact rather than approximate: a
    disagreement of one quantum is a curve that has drifted."""
    out = _js("""
      setPad(%s, %s);
      tick(1000);
      OUT = {emitted: globalThis.__emitted};
    """ % (json.dumps(axes), json.dumps(buttons)))
    sent = [p for name, p in out["emitted"] if name == "aerobo_pad"]
    assert len(sent) == 1, "one poll, one report"
    assert sent[0]["h"] == pad.holds(axes, buttons)


@pytest.mark.skipif(jsx.NODE is None, reason="node is not installed")
def test_a_held_stick_is_one_message_and_not_thirty_a_second():
    """The whole reason the report is on CHANGE. A stick held at 40 % costs
    one message; a stick at rest costs none after the first."""
    out = _js("""
      setPad([0, 0, 0.4, 0], []);
      let t = 1000;
      for (let i = 0; i < 40; i++) { t += 1000 / 120; tick(t); }
      OUT = {n: globalThis.__emitted.length};
    """)
    assert out["n"] == 1


@pytest.mark.skipif(jsx.NODE is None, reason="node is not installed")
def test_a_moving_stick_reports_but_no_faster_than_the_ceiling():
    """Polled every animation frame (120 Hz here), rate-limited to MAX_HZ:
    the demand is smoothed by the spring anyway, so reporting faster than
    the physics push buys nothing and costs a message."""
    out = _js("""
      let t = 1000, n = 0;
      for (let i = 0; i < 120; i++) {          // 1 s at 120 Hz
        setPad([0, 0, 0.2 + i * 0.006, 0], []);
        t += 1000 / 120; tick(t);
      }
      OUT = {n: globalThis.__emitted.length};
    """)
    assert 1 <= out["n"] <= pad.MAX_HZ + 2, out["n"]


@pytest.mark.skipif(jsx.NODE is None, reason="node is not installed")
def test_a_pad_that_is_unplugged_lets_go_of_the_stick():
    """Unplugged mid-roll, the last demand would otherwise stand for ever —
    the same shape of defect as a keyup landing on a stopped loop."""
    out = _js("""
      setPad([0, 0, 1.0, 0], []);
      tick(1000);
      setPad(null);
      tick(1100);
      tick(1200);
      OUT = {emitted: globalThis.__emitted};
    """)
    sent = [p for _n, p in out["emitted"]]
    assert sent[0]["on"] is True
    assert sent[-1]["on"] is False, "the disconnect was never reported"
    assert len(sent) == 2, "and it is reported ONCE, not every frame"


@pytest.mark.skipif(jsx.NODE is None, reason="node is not installed")
def test_a_button_press_is_never_eaten_by_the_rate_limit():
    """A press is one instant and the limiter is a window. A cross that
    starts the flight must not be dropped because a stick moved 8 ms ago."""
    out = _js("""
      let t = 1000;
      setPad([0, 0, 0.5, 0], []);  tick(t);          // a report, just now
      t += 1000 / 120;
      setPad([0, 0, 0.5, 0], [1.0]);  tick(t);       // cross, same window
      OUT = {emitted: globalThis.__emitted};
    """)
    acts = [a for _n, p in out["emitted"] for a in p["a"]]
    assert acts == ["fly"]


@pytest.mark.skipif(jsx.NODE is None, reason="node is not installed")
def test_the_pad_is_polled_before_there_is_anything_to_draw():
    """The cross that STARTS the flight is pressed when nothing is flying,
    and nothing has been pushed. A poll that lived after the tick's
    "nothing pending" return could never send it."""
    out = _js("""
      setPad([0, 0, 0, 0], [1.0]);
      tick(1000);                      // no push() at all, ever
      OUT = {emitted: globalThis.__emitted};
    """)
    assert [a for _n, p in out["emitted"] for a in p["a"]] == ["fly"]


@pytest.mark.skipif(jsx.NODE is None, reason="node is not installed")
def test_a_driver_built_without_a_pad_config_polls_nothing():
    """``pad_cfg={}`` is how the pose tests build a driver with no pad. It
    must not throw and must not emit."""
    out = jsx.run(jsx.CFG, """
      globalThis.__emitted = [];
      tick(1000);
      push({q: null, z: 1, w: [0, 0, -300], game: true});
      tick(1016);
      OUT = {n: globalThis.__emitted.length, z: A.now.z};
    """, live_js=_PAD_STUB + live._JS, apply_js=hud.APPLY_JS)
    assert out["n"] == 0
    assert out["z"] > 0.0, "the rest of the frame loop still ran"


def test_the_stage_installs_the_shipped_pad_config():
    """The script carries the module's numbers, not a second copy of them."""
    js = live.INSTALL_JS(
        scene_id=1, group_id="g", world_id="w", hud_id=2, readout_id=3,
        banner_id=4, panel_id=5, thrust_id=6, span_m=10.0, snap_m=480.0,
        back=2.6, up=0.42, lead=0.35, apply_js="null", gauge_js="null")
    cfg = json.loads(js[js.index("init(") + 5:js.rindex(", null, null);")])
    assert cfg["pad"] == pad.js_config()
    assert cfg["pad"]["dead"] == pad.DEADZONE


# ------------------------------------------------- through the stage itself

def _pad(ctx, h=None, a=None, on=True):
    """One report, the shape the browser emits it in (varargs list)."""
    from types import SimpleNamespace as NS

    ctx.act("flight_pad", NS(args=[{"h": h or {}, "a": a or [], "on": on}]))


def test_the_pad_and_the_keyboard_are_added_and_neither_overrules(capsys):
    """TWO HANDS ON ONE CONTROL. A pad let go to centre must not centre an
    axis whose key is still down — the keys-down defect, one device out."""
    from types import SimpleNamespace as NS

    from tests.test_v4_stages import _armed

    ctx = _armed()
    ctx.render("controls", "derivatives")
    ctx.act("flight_arm")
    F = ctx.S["flight"]
    F["running"] = True

    ctx.act("flight_key", NS(key=NS(name="a"),
                             action=NS(keydown=True, keyup=False)))
    assert F["hold"]["aileron"] == pytest.approx(1.0)     # roll left, held

    _pad(ctx, {"aileron": -1.0})                          # pad rolls right
    assert F["hold"]["aileron"] == pytest.approx(0.0), \
        "two opposite demands on one control did not cancel"

    _pad(ctx, {"aileron": 0.0})                           # pad let go
    assert F["hold"]["aileron"] == pytest.approx(1.0), \
        "the pad centring the axis threw away the key still down"
    F["running"] = False
    capsys.readouterr()


def test_an_analogue_stick_deflects_part_way(capsys):
    """What the keyboard could not ask for, end to end through the stage."""
    from tests.test_v4_stages import _armed

    ctx = _armed()
    ctx.render("controls", "derivatives")
    ctx.act("flight_arm")
    F = ctx.S["flight"]
    F["running"] = True
    _pad(ctx, {"aileron": 0.4})
    for _ in range(120):
        ctx.act("flight_advance", 1 / 60)
    assert F["stick"]["aileron"] == pytest.approx(0.4 * stk.LIMITS["aileron"])
    F["running"] = False
    capsys.readouterr()


def test_the_stick_is_ignored_while_stopped_and_the_buttons_are_not(capsys):
    """The rule the keyboard already flies under, with the exception a pad
    needs: nothing may deflect a surface that is not in the air, but the
    pilot has to be able to press a button on the pad they are holding."""
    from tests.test_v4_stages import _armed

    ctx = _armed()
    ctx.render("controls", "derivatives")
    ctx.act("flight_arm")
    F = ctx.S["flight"]
    F["running"] = False

    _pad(ctx, {"aileron": 1.0, "elevator": 1.0, "thrust": 1.0})
    assert F["hold"]["aileron"] == 0.0
    assert F["hold"]["elevator"] == 0.0
    assert F["hold"]["thrust"] == 0.0

    was = F.get("mode") or "game"
    _pad(ctx, {}, ["view"])
    assert (F.get("mode") or "game") != was, \
        "a button was swallowed because nothing was flying"
    capsys.readouterr()


def test_the_pad_lets_go_of_the_stick_when_the_flight_is_paused(capsys):
    """``_teardown`` clears the keyboard's set for this reason; the pad's
    half of the demand has to go with it, or it is added back into the next
    axis the keyboard touches."""
    from tests.test_v4_stages import _armed

    ctx = _armed()
    ctx.render("controls", "derivatives")
    ctx.act("flight_arm")
    F = ctx.S["flight"]
    F["running"] = True
    _pad(ctx, {"aileron": 1.0})
    assert F["pad"]["aileron"] == pytest.approx(1.0)
    ctx.act("flight_run", False)
    assert F["pad"]["aileron"] == 0.0
    assert F["hold"]["aileron"] == 0.0
    capsys.readouterr()


def test_a_disconnect_is_recorded_so_the_panel_can_say_so(capsys):
    from tests.test_v4_stages import _armed

    ctx = _armed()
    ctx.render("controls", "derivatives")
    ctx.act("flight_arm")
    F = ctx.S["flight"]
    _pad(ctx, {}, [], True)
    assert F["pad_on"] is True
    _pad(ctx, {}, [], False)
    assert F["pad_on"] is False
    capsys.readouterr()


def test_the_fly_view_says_whether_a_pad_is_being_seen(capsys):
    """The failure a browser gives you here is SILENT — no pad, no error, a
    controller that simply does nothing. So the panel has to say which case
    the pilot is in, and it has to say it before the flight starts."""
    from tests.test_v4_stages import _armed

    ctx = _armed()
    ctx.render("controls", "derivatives")
    ctx.act("flight_arm")
    ctx.render("flight", "fly")     # the legend lives beside the keys
    err = capsys.readouterr().err
    assert "failed to render" not in err, err
    assert "no gamepad" in ctx.S["flight"]["pad_status"]
    _pad(ctx, {}, [], True)
    assert ctx.S["flight"]["pad_status"] == "gamepad connected"
    _pad(ctx, {}, [], False)
    assert "no gamepad" in ctx.S["flight"]["pad_status"]
    capsys.readouterr()


def test_the_cross_starts_the_flight_with_the_stick_it_was_pressed_with(capsys):
    """THE ORDER INSIDE ONE REPORT. Because the browser only speaks when the
    demand changes, a pilot who holds the stick over while pressing ✕ sends
    exactly one message. If the sticks were read before the buttons they
    would be zeroed (nothing is flying yet) and the aeroplane would take off
    with a centred stick until the pilot moved it."""
    from tests.test_v4_stages import _armed

    ctx = _armed()
    ctx.render("controls", "derivatives")
    ctx.act("flight_arm")
    F = ctx.S["flight"]
    assert F.get("running") is not True

    _pad(ctx, {"aileron": 0.5}, ["fly"])
    assert F["running"] is True, "the pad could not start the flight"
    assert F["hold"]["aileron"] == pytest.approx(0.5), \
        "the stick held through the press was thrown away"

    ctx.act("flight_run", False)
    capsys.readouterr()
