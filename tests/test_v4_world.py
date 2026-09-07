"""Aircraft-centred rendering — gui.v4.world.

Pure geometry, so these are closed forms and invariants rather than
screenshots. The two properties that matter are the ones a picture cannot
show you: that the chase frame drops HEADING but keeps bank, and that the
floating origin is genuinely continuous across a tile boundary.
"""

import json

import numpy as np
import pytest

import tests.v4_js as jsx
from gui.v4 import world as W

SPACING = 200.0


def _q(roll=0.0, pitch=0.0, yaw=0.0):
    """Body quaternion from 3-2-1 Euler angles, via the DCM the sim uses."""
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array([cr * cp * cy + sr * sp * sy,
                     sr * cp * cy - cr * sp * sy,
                     cr * sp * cy + sr * cp * sy,
                     cr * cp * sy - sr * sp * cy])


DATUM = 300.0


def _draw(pos, v_local, datum=DATUM):
    """Where a ground marker at scene-local ``v_local`` lands on screen."""
    t = np.asarray(W.world_offset(pos, SPACING, datum), dtype=float)
    return np.asarray(v_local, dtype=float) + t


# ------------------------------------------------------------- the heading

@pytest.mark.parametrize("psi_deg", [0.0, 30.0, 90.0, 179.0, -120.0])
def test_the_heading_comes_back_out_of_the_quaternion(psi_deg):
    psi = np.deg2rad(psi_deg)
    assert W.heading_of(_q(yaw=psi)) == pytest.approx(psi, abs=1e-9)


def test_the_heading_survives_a_steep_pitch_where_euler_would_not():
    """Taken off the DCM's first column, not from an Euler conversion, so it
    does not blow up on the way to vertical."""
    for pitch_deg in (60.0, 85.0, 89.9):
        psi = W.heading_of(_q(pitch=np.deg2rad(pitch_deg), yaw=0.7))
        assert psi == pytest.approx(0.7, abs=1e-6), pitch_deg


def test_straight_up_is_not_an_exception():
    """The heading is genuinely undefined there. It must not raise, and it
    must not be a nan that poisons a rotation matrix."""
    psi = W.heading_of(_q(pitch=np.pi / 2))
    assert np.isfinite(psi)


def test_yaw_dcm_is_a_rotation():
    for a in (0.0, 0.4, -2.0):
        R = W.yaw_dcm(a)
        assert np.linalg.det(R) == pytest.approx(1.0)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)


# ------------------------------------------------------------ the camera

def _R(**kw):
    """flight.scene_rotation for these Euler angles — the real one."""
    from gui.v4.stages import flight as fl

    return fl.scene_rotation(_q(**kw))


def test_the_camera_sits_BEHIND_the_aeroplane_and_looks_ahead():
    eye, target = W.chase_eye(_R(), span_m=10.0)
    # nose is +x in the scene for a level aeroplane heading north
    assert eye[0] < -10.0, "the camera is not behind the tail"
    assert target[0] > 0.0, "it is not looking where the aeroplane is going"
    assert target[0] > eye[0]


def test_the_camera_sits_ABOVE_the_aeroplane_not_under_it():
    """`up` in the scene is +z, but BODY up is (0,0,-1) because body z is
    DOWN. Using (0,0,1) puts the camera under the belly — which looks almost
    plausible in a still and is wrong the moment anything pitches.
    """
    eye, _t = W.chase_eye(_R(), span_m=10.0)
    assert eye[2] > 0.5, "the camera is below the aeroplane"


def test_the_camera_FOLLOWS_the_heading_round():
    """Turn east and the camera swings to the west side, or it is not a
    chase camera at all."""
    e_n, _ = W.chase_eye(_R(), span_m=10.0)
    e_e, _ = W.chase_eye(_R(yaw=np.deg2rad(90.0)), span_m=10.0)
    assert e_n[0] < -10.0 and abs(e_n[1]) < 1e-6      # heading north: due south
    # heading east: the camera is due WEST, which is +y in a port-handed scene
    assert abs(e_e[0]) < 1e-6 and e_e[1] > 10.0


def test_the_camera_distance_is_in_SPANS_so_it_frames_any_aeroplane():
    small, _ = W.chase_eye(_R(), span_m=1.0)
    big, _ = W.chase_eye(_R(), span_m=20.0)
    assert big[0] == pytest.approx(20.0 * small[0])
    assert big[2] == pytest.approx(20.0 * small[2])


def test_pitching_up_tilts_the_camera_with_it():
    level, _ = W.chase_eye(_R(), span_m=10.0)
    up, _ = W.chase_eye(_R(pitch=np.deg2rad(25.0)), span_m=10.0)
    assert up[2] < level[2], "climbing did not drop the camera behind"


def test_the_camera_rides_the_drawn_altitude():
    lo_e, lo_t = W.chase_eye(_R(), span_m=10.0, drawn_z=0.0)
    hi_e, hi_t = W.chase_eye(_R(), span_m=10.0, drawn_z=50.0)
    assert hi_e[2] == pytest.approx(lo_e[2] + 50.0)
    assert hi_t[2] == pytest.approx(lo_t[2] + 50.0)


# ------------------------------------------------------ the floating origin

def test_the_world_translation_stays_BOUNDED_however_far_it_flies():
    """160 km downrange must cost no more scene precision than 160 m."""
    for north in (0.0, 137.0, 4_000.0, 160_000.0, -98_765.4):
        t = W.world_offset(np.array([north, 0.0, -300.0]), SPACING, DATUM)
        assert abs(t[0]) <= SPACING / 2 + 1e-9, north


def test_flying_one_whole_TILE_leaves_the_picture_identical():
    """The exact statement of the floating origin, with no epsilon in it.

    The lattice is periodic with period ``spacing``, and the snap shifts by
    exactly one period when the aeroplane advances by one, so the drawn field
    is BIT-identical a tile later. That is what makes the ground infinite
    without any object ever being moved or created.
    """
    pts = W.lattice_offsets(SPACING, 3)
    for start_n in (0.0, 37.3, -1234.5):
        a = np.array([start_n, 0.0, -300.0])
        b = np.array([start_n + SPACING, 0.0, -300.0])
        for v in pts:
            assert np.allclose(_draw(a, v), _draw(b, v), atol=1e-9), \
                (start_n, v)


def test_the_ground_never_LURCHES_as_the_origin_re_snaps():
    """Continuity of the PICTURE, sampled densely across two whole tiles.

    Note what is NOT asserted: that each marker moves by one step. It does
    not. When the origin re-snaps, the group's translation jumps by a whole
    tile and every marker jumps with it — and the picture is unchanged
    anyway, because the lattice maps onto itself and the markers are
    identical. Continuity is a property of the SET, not of any object in it.

    So the measurement is nearest-neighbour: how far is each drawn marker
    from the closest marker in the previous frame. That has to be the
    aircraft's own step, at every sample, including the ones that straddle a
    snap. An earlier version of this test asserted per-object movement and
    reported a 195 m lurch for a 5 m step, which was the test being wrong
    rather than the field.
    """
    pts = W.lattice_offsets(SPACING, 3)
    step = SPACING / 40.0
    prev = None
    worst = 0.0
    for k in range(81):
        pos = np.array([k * step, 0.0, -300.0])
        drawn = np.array([_draw(pos, v) for v in pts])
        if prev is not None:
            d = np.linalg.norm(drawn[:, None, :] - prev[None, :, :], axis=2)
            worst = max(worst, float(np.median(d.min(axis=1))))
        prev = drawn
    assert worst == pytest.approx(step, abs=1e-9), \
        f"the field moved {worst:.4g} m for a {step:.4g} m step"


def test_flying_forward_moves_the_ground_BACKWARDS():
    """The whole point: the world streams past, or there is no sense of
    speed at all."""
    a = _draw(np.array([0.0, 0.0, -300.0]), (50.0, 0.0, 0.0))
    b = _draw(np.array([10.0, 0.0, -300.0]), (50.0, 0.0, 0.0))
    assert b[0] < a[0] - 9.9, "the ground did not move under the aeroplane"


def test_the_ground_sits_at_the_DATUM_and_not_at_zero():
    """Scene z = 0 is where the built-in grid puts its floor, and that is a
    false floor at the DATUM height — the aircraft is drawn at
    ``altitude - datum``, so it would pass straight through it on the way
    down. The ground belongs at ``-datum``, where altitude zero meets it.
    """
    for alt in (0.0, 300.0, 5_000.0):
        t = W.world_offset(np.array([0.0, 0.0, -alt]), SPACING, DATUM)
        assert t[2] == pytest.approx(-DATUM), \
            "the ground moved when the aeroplane climbed"
    # ...and at the datum altitude the aircraft's own drawn z is zero, so it
    # is exactly DATUM above the ground: descending to zero closes the gap
    assert W.world_offset(np.zeros(3), SPACING, 0.0)[2] == pytest.approx(0.0)


# ---------------------------------------------------------------- the tiles

def test_the_lattice_keeps_EVERY_node_including_the_middle_one():
    """Dropping "the one under the aeroplane" punches a hole that DRIFTS.

    The markers sit on lattice nodes, not under the aircraft, so the missing
    one slides backwards through an otherwise regular field and reappears a
    tile later. Caught by the tile-crossing test, which is the only thing
    that could have caught it.
    """
    pts = W.lattice_offsets(SPACING, 2)
    assert (0.0, 0.0, 0.0) in pts
    assert len(pts) == (2 * 2 + 1) ** 2


def test_the_markers_are_STATIC_which_is_why_this_costs_two_messages():
    """Called twice, the lattice is identical — nothing about it depends on
    where the aeroplane is. Everything that moves is in the group transform.
    """
    assert W.lattice_offsets(SPACING, 3) == W.lattice_offsets(SPACING, 3)


def test_a_degenerate_lattice_is_empty_not_an_error():
    assert W.lattice_offsets(0.0, 3) == []
    assert W.lattice_offsets(SPACING, -1) == []
    assert W.snap(12.0, 0.0) == 0.0


# ------------------------------------------------- the HUD, as pure data

def _kw(**over):
    kw = dict(speed=40.0, altitude=300.0, heading_deg=0.0, pitch_deg=0.0,
              roll_deg=0.0, alpha_deg=0.0, g=1.0, throttle=0.0,
              throttle_max=100.0)
    kw.update(over)
    return kw


def test_the_skeleton_carries_a_node_for_every_number_that_moves():
    """The whole point of the split: if an id is missing the browser half
    writes into nothing and the glass silently freezes at its first frame."""
    from gui.v4 import hud

    svg = hud.skeleton()
    assert svg.startswith("<svg") and svg.endswith("</svg>")
    for node in ("hud-ladder", "hud-bank", "hud-hdg", "hud-alpha", "hud-g",
                 "hud-thr", "hud-ias", "hud-alt", "hud-ias-minor",
                 "hud-ias-major", "hud-alt-minor", "hud-alt-major",
                 "hud-warn0", "hud-warn1", "hud-warn2"):
        assert f'id="{node}"' in svg, node
    for j in range(9):
        assert f'id="hud-ias-l{j}"' in svg and f'id="hud-alt-l{j}"' in svg
    assert "IAS m/s" in svg and "ALT m" in svg and "THR" in svg


def test_the_skeleton_does_not_depend_on_any_flight_number():
    """It is sent once and never again, so it must be a CONSTANT."""
    from gui.v4 import hud

    assert hud.skeleton() == hud.skeleton()


def test_the_state_is_small_enough_to_send_every_frame():
    """The measured reason this module was split in two. The old glass was
    14.8 kB of markup a frame; a report has to be a rounding error next to
    it or nothing has been fixed."""
    import json

    from gui.v4 import hud

    n = len(json.dumps(hud.state(**_kw(speed=47.5, altitude=312.0)),
                       separators=(",", ":")))
    assert n < 700, n
    assert len(hud.skeleton()) > 20 * n


def test_the_state_carries_every_number_a_pilot_flies_on():
    from gui.v4 import hud

    s = hud.state(**_kw(speed=47.5, altitude=312.0, heading_deg=95.0,
                        pitch_deg=6.0, roll_deg=-12.0, alpha_deg=3.4,
                        g=1.05, throttle=120.0, throttle_max=400.0))
    assert s["ias"] == pytest.approx(47.5)
    assert s["alt"] == pytest.approx(312.0)
    assert s["hdg"] == pytest.approx(95.0)
    assert s["roll"] == pytest.approx(-12.0)
    assert s["alpha"] == pytest.approx(3.4)
    assert s["g"] == pytest.approx(1.05)
    assert s["thr_h"] == pytest.approx(hud._THR_H * 120.0 / 400.0)


def test_the_ladder_moves_DOWN_when_the_nose_comes_UP():
    """A pitch ladder is fixed to the WORLD. Climb and the horizon leaves
    the top of the screen — the boresight has gone above it.

    The first version had the sign the other way: at pitch zero it drew the
    +10 line BELOW the horizon and walked the horizon UP as the nose came up.
    Level flight looks identical either way, which is why this is a test and
    not a screenshot.
    """
    from gui.v4 import hud

    level = hud.state(**_kw(pitch_deg=0.0))["pitch_px"]
    up = hud.state(**_kw(pitch_deg=10.0))["pitch_px"]
    down = hud.state(**_kw(pitch_deg=-10.0))["pitch_px"]
    assert level == pytest.approx(0.0)
    # SVG y grows downward, so a POSITIVE translate moves the ladder down
    assert up > 0.0 and down < 0.0
    assert up == pytest.approx(-down)

    # ...and the SKELETON has to be drawn the matching way round, or the
    # translation is applied to a ladder that is already upside down and the
    # two errors cancel at zero and nowhere else.
    svg = hud.skeleton()
    # the horizon is the only row spanning +-300 from the centre
    horizon = float(svg.split('<line x1="200" y1="')[1].split('"')[0])
    assert horizon == pytest.approx(hud.VIEW_H / 2.0)

    # every ladder row, with the flag that says which side it belongs on.
    # Parsed as a SET rather than by label, because the labels are |deg| and
    # so +10 and -10 are both "10".
    body = svg.split('<g id="hud-ladder">')[1].split("</g></g>")[0]
    climb, dive = [], []
    for chunk in body.split("<line ")[1:]:
        y = float(chunk.split('y1="')[1].split('"')[0])
        if y == pytest.approx(horizon):
            continue                                    # the horizon itself
        (dive if "stroke-dasharray" in chunk.split("/>")[0]
         else climb).append(y)
    assert climb and dive
    # SVG y grows downward, so CLIMB lines are above the horizon and the
    # dashed DIVE lines below it. Get this backwards and the ladder still
    # looks right in level flight.
    assert max(climb) < horizon, "a climb line was drawn below the horizon"
    assert min(dive) > horizon, "a dive line was drawn above the horizon"


def test_the_ladder_ROLLS_with_the_aeroplane_the_right_way():
    """+roll is the right wing down, so the world turns anticlockwise on
    screen — and SVG's rotate is clockwise-positive, hence the sign."""
    from gui.v4 import hud

    assert hud.state(**_kw(roll_deg=25.0))["roll"] == pytest.approx(25.0)
    assert "rotate(' + (-s.roll)" in hud.APPLY_JS


def test_a_tape_translates_by_the_FRACTION_of_a_step():
    """The tick marks are periodic, which is the whole reason they never
    have to be redrawn. Move by exactly one step and the tape is where it
    started; the LABELS are what changed."""
    from gui.v4 import hud

    a = hud.state(**_kw(speed=40.0))
    b = hud.state(**_kw(speed=40.0 + hud.IAS_STEP))
    # one MINOR step: the minor ticks are back where they started...
    assert a["ias_minor"] == pytest.approx(b["ias_minor"])
    # ...but the labelled ticks are only half way, so they must have moved
    assert a["ias_major"] != pytest.approx(b["ias_major"])
    assert a["ias_base"] == b["ias_base"]

    two = hud.state(**_kw(speed=40.0 + 2 * hud.IAS_STEP))
    # one MAJOR step: everything is back, and every label has stepped on
    assert a["ias_major"] == pytest.approx(two["ias_major"])
    assert a["ias_minor"] == pytest.approx(two["ias_minor"])
    assert two["ias_base"] == a["ias_base"] + 1


def test_a_tape_label_reads_the_value_that_is_beside_it():
    """The closed form the browser half runs, checked here: label ``j`` sits
    ``(j - 4)`` major steps below centre, so it must read that much LESS."""
    from gui.v4 import hud

    for value, step, base_key, step_key in (
            (47.0, hud.IAS_STEP, "ias_base", "ias_step"),
            (312.0, hud.ALT_STEP, "alt_base", "alt_step")):
        kw = _kw(speed=value) if base_key == "ias_base" \
            else _kw(altitude=value)
        s = hud.state(**kw)
        major = 2.0 * step
        for j in range(9):
            shown = (s[base_key] - (j - 4)) * s[step_key]
            # every label is within five major steps: the base tick is
            # inside one step of the value and the labels reach four either
            # side of it
            assert abs(shown - value) <= 5.0 * major + 1e-9
        centre = (s[base_key] + 4 - 4) * s[step_key]
        assert value - major < centre <= value


def test_the_throttle_bar_is_a_FRACTION_of_its_own_band():
    from gui.v4 import hud

    assert hud.state(**_kw(throttle=0.0, throttle_max=400.0))["thr_h"] == 0.0
    full = hud.state(**_kw(throttle=400.0, throttle_max=400.0))
    over = hud.state(**_kw(throttle=9e9, throttle_max=400.0))
    assert full["thr_h"] == pytest.approx(hud._THR_H)
    assert over["thr_h"] == pytest.approx(hud._THR_H), \
        "the bar left its gauge"
    # ...and it grows UPWARD from a fixed foot
    assert over["thr_y"] < full["thr_y"] + 1e-9
    assert full["thr_y"] + full["thr_h"] == pytest.approx(
        hud.VIEW_H - 50.0)


def test_a_zero_width_throttle_band_does_not_divide_by_zero():
    from gui.v4 import hud

    assert hud.state(**_kw(throttle=5.0, throttle_max=0.0))["thr_h"] == 0.0


def test_the_heading_wraps_instead_of_reading_361():
    from gui.v4 import hud

    assert hud.state(**_kw(heading_deg=365.0))["hdg"] == pytest.approx(5.0)
    assert hud.state(**_kw(heading_deg=-5.0))["hdg"] == pytest.approx(355.0)


def test_warnings_reach_the_glass_and_are_capped():
    from gui.v4 import hud

    s = hud.state(warnings=["GROUND CONTACT", "ALPHA", "LOW", "EXTRA"],
                  **_kw())
    assert s["warn"] == ["GROUND CONTACT", "ALPHA", "LOW"]
    assert len(s["warn"]) <= 3, "an unbounded list would cover the sky"


def test_alpha_and_g_go_AMBER_when_they_should():
    from gui.v4 import hud

    amber = "#ffcf5a"
    assert hud.state(**_kw(alpha_deg=2.0, g=1.0))["alpha_col"] != amber
    assert hud.state(**_kw(alpha_deg=16.0, g=1.0))["alpha_col"] == amber
    assert hud.state(**_kw(alpha_deg=2.0, g=4.5))["g_col"] == amber
    assert hud.state(**_kw(alpha_deg=2.0, g=-0.5))["g_col"] == amber
    assert hud.state(**_kw(alpha_deg=2.0, g=1.0))["g_col"] != amber


# ---------------------------------------------- the browser half, as source

def test_the_apply_script_writes_every_id_the_skeleton_offers():
    """A typo in an id is a field that silently never updates, and no Python
    test would see it. So the two halves are checked against each other."""
    from gui.v4 import hud

    js = hud.APPLY_JS
    for stem in ("hud-ladder", "hud-bank", "hud-hdg", "hud-alpha", "hud-g",
                 "hud-thr", "hud-warn"):
        assert stem in js.replace("' + ", "").replace(" + '", ""), stem
    # the tape ids are built from a loop, so check the pieces
    assert "'-minor'" in js or "-minor" in js
    assert "'-major'" in js or "-major" in js


def test_the_apply_script_only_writes_a_node_when_it_CHANGED():
    """Setting textContent to what it already says still dirties the node for
    layout, sixty times a second, for a number that did not move."""
    from gui.v4 import hud

    assert "c.last[key] === text" in hud.APPLY_JS
    assert "c.last[key] === value" in hud.APPLY_JS


def test_the_apply_script_caches_its_lookups():
    from gui.v4 import hud

    assert "__hudCache" in hud.APPLY_JS


# ------------------------------------------------- the one-message-a-frame

def test_a_report_is_hundreds_of_bytes_not_tens_of_thousands():
    """The measurement that started all this: 155 messages/s and 515 kB/s,
    of which the glass alone was 497.6."""
    import json

    from gui.v4 import hud, live

    p = {"q": [0.1, 0.2, 0.3, 0.9], "z": 12.5, "w": [1.0, 2.0, -300.0],
         "game": True, "hud": hud.state(**_kw())}
    call = live.push_call(p)
    assert len(call) < 1000, len(call)
    assert call.startswith("window.__aerobo")
    # ...and it is still valid JSON inside
    body = call[call.index("(") + 1:call.rindex(")")]
    assert json.loads(body)["z"] == 12.5


def test_the_scene_fps_is_high_enough_that_the_DISPLAY_governs():
    """``ui.scene(fps=n)`` is not a target rate. scene.js:217 reads

        requestAnimationFrame(() => setTimeout(() => render(), 1000 / fps))

    so the delay is added AFTER the animation frame and the two compose in
    series: fps=60 costs 16.7 ms of rAF plus a 16.7 ms timeout, i.e. about
    30 Hz. The number has to be large enough for that second term to vanish.
    """
    from gui.v4 import live

    added_ms = 1000.0 / live.SCENE_FPS
    assert added_ms <= 2.0, added_ms

    # ...and the smoothing time constant is bounded on BOTH sides against
    # the report rate. Too short and the flip-book shows through between
    # reports; too long and the aeroplane visibly lags the keyboard, which
    # is the very complaint this work started from.
    from gui.v4.stages.flight import FPS

    interval = 1.0 / FPS
    assert live.SMOOTH_TAU_S >= 0.5 * interval, live.SMOOTH_TAU_S
    assert live.SMOOTH_TAU_S <= 2.0 * interval, live.SMOOTH_TAU_S


def test_the_installer_carries_the_ids_the_browser_needs():
    from gui.v4 import gauge, hud, live

    # the two kinds of id are NOT interchangeable: a scene element is an
    # integer nicegui id, a scene OBJECT is a uuid string keying the scene's
    # own map. Swap them and the picture silently never moves.
    js = live.INSTALL_JS(scene_id=5, group_id="a-uuid", world_id="b-uuid",
                         hud_id=8, readout_id=11, banner_id=12,
                         panel_id=13, thrust_id=14,
                         span_m=10.0, snap_m=480.0, back=2.6,
                         up=0.42, lead=0.35, apply_js=hud.APPLY_JS,
                         gauge_js=gauge.APPLY_JS)
    assert '"scene":5' in js and '"hud":8' in js
    assert '"readout":11' in js and '"banner":12' in js
    assert '"panel":13' in js and '"thrust":14' in js
    assert '"group":"a-uuid"' in js and '"world":"b-uuid"' in js
    assert '"snap":480.0' in js
    assert "__hudCache" in js, "the glass half never reached the page"
    assert "__gaugeCache" in js, "the attitude panel never reached the page"
    assert "requestAnimationFrame" in js


# --------------------------------- the browser half, RUN rather than grepped

def _js(script: str) -> dict:
    from gui.v4 import hud, live

    return jsx.run(jsx.CFG, script, live_js=live._JS, apply_js=hud.APPLY_JS)


@pytest.mark.skipif(jsx.NODE is None, reason="node is not installed")
def test_the_browser_half_smooths_toward_the_reported_pose():
    """Thirty reports a second become as many frames as the display draws.
    The law is first order with the configured time constant, so the first
    frame of a step must cover exactly ``1 - exp(-dt/tau)`` of it."""
    out = _js("""
      let t = 1000; tick(t);            // the first tick SNAPS; start after it
      push({q: null, z: 100, w: [0, 0, -300], game: true});
      const trace = [];
      for (let i = 0; i < 20; i++) { t += 1000 / 60; tick(t);
        trace.push(A.now.z); }
      OUT = {trace, first: trace[0], last: trace[trace.length - 1]};
    """)
    tau, dt = jsx.CFG["tau"], 1.0 / 60.0
    assert out["first"] == pytest.approx(100.0 * (1 - np.exp(-dt / tau)),
                                         rel=1e-6)
    assert out["last"] == pytest.approx(100.0, abs=0.5)
    tr = out["trace"]
    assert all(b >= a for a, b in zip(tr, tr[1:])), "the filter is not monotone"
    assert max(tr) <= 100.0 + 1e-9, "the filter overshot its target"


@pytest.mark.skipif(jsx.NODE is None, reason="node is not installed")
def test_the_FIRST_frame_snaps_and_a_repeated_instant_does_not():
    """Two different things that both look like ``dt == 0``.

    The first frame after arming must SNAP — the aeroplane appears where it
    was reported, rather than sliding in from the origin over a time
    constant. But a second tick in the same instant must move nothing: the
    filter gain for zero elapsed time is zero, not one. Writing the
    condition as ``dt <= 0`` makes the second case behave like the first,
    which is a smoother that stops smoothing whenever the clock repeats.
    """
    out = _js("""
      push({q: null, z: 500, w: [0, 0, -300], game: true});
      tick(999999);                       // a wildly late FIRST frame
      const snapped = A.now.z;
      push({q: null, z: 900, w: [0, 0, -300], game: true});
      tick(999999);                       // ...and the very same instant
      OUT = {snapped, sameInstant: A.now.z};
    """)
    assert out["snapped"] == 500.0, "the first frame did not show the report"
    assert out["sameInstant"] == 500.0, "zero elapsed time moved the aeroplane"


@pytest.mark.skipif(jsx.NODE is None, reason="node is not installed")
def test_the_browser_half_REANCHORS_the_ground_before_it_smooths():
    """The floating origin is a SAWTOOTH: the offset jumps a whole tile when
    the aeroplane crosses one, and the picture is unchanged because the
    lattice has exactly that period. Smoothing across that discontinuity
    instead slides the whole world 480 m over one time constant.

    So the thing to measure is drawn motion MODULO the tile. This is the
    test that a source-grep for the word "reanchor" cannot be: a function
    that returns early still contains the word.
    """
    snap = jsx.CFG["snap"]
    out = _js(f"""
      const per = {snap};
      A.now.w = [per / 2 - 1, 0, -300];        // just short of the boundary
      A.last = 1000;
      push({{q: null, z: 0, w: [-per / 2 + 1, 0, -300], game: true}});
      let t = 1000; const steps = []; let prev = A.now.w[0];
      for (let i = 0; i < 10; i++) {{ t += 1000 / 60; tick(t);
        let d = A.now.w[0] - prev; prev = A.now.w[0];
        d = ((d % per) + per * 1.5) % per - per / 2;   // modulo the tile
        steps.push(Math.abs(d)); }}
      OUT = {{steps, total: steps.reduce((a, b) => a + b, 0)}};
    """)
    # the gap the short way round is 2 m; the long way is 478
    assert out["total"] < 2.5, out["total"]
    assert max(out["steps"]) < 1.0, out["steps"]


@pytest.mark.skipif(jsx.NODE is None, reason="node is not installed")
def test_the_game_camera_sits_behind_the_tail_and_looks_ahead():
    """Computed in the browser from the pose ON SCREEN. Checked against the
    same closed form :func:`world.chase_eye` implements, because two copies
    of a formula is exactly how a chase view ends up under the belly."""
    out = _js("""
      push({q: [0, 0, 0, 1], z: 0, w: [0, 0, -300], game: true});
      A.last = 1000; let t = 1000;
      for (let i = 0; i < 400; i++) { t += 1000 / 60; tick(t); }
      OUT = {eye: [__scene.camera.position.x, __scene.camera.position.y,
                   __scene.camera.position.z],
             look: __scene.camera.lookedAt,
             controls: __scene.controls.enabled,
             tween: __scene.camera_tween};
    """)
    eye, look = W.chase_eye(np.eye(3), jsx.CFG["span"],
                            back=jsx.CFG["back"], up=jsx.CFG["up"],
                            lead=jsx.CFG["lead"], drawn_z=0.0)
    assert np.allclose(out["eye"], eye, atol=1e-6)
    assert np.allclose(out["look"], look, atol=1e-6)
    # the orbit controls and the tween must both let go, or they fight the
    # direct writes and the camera judders between two owners
    assert out["controls"] is False
    assert out["tween"] is None


@pytest.mark.skipif(jsx.NODE is None, reason="node is not installed")
def test_the_engineering_view_hands_the_camera_BACK():
    out = _js("""
      push({q: [0, 0, 0, 1], z: 0, w: [0, 0, -300], game: false});
      A.last = 1000; let t = 1000;
      for (let i = 0; i < 5; i++) { t += 1000 / 60; tick(t); }
      OUT = {controls: __scene.controls.enabled,
             moved: __scene.camera.lookedAt || null};
    """)
    assert out["controls"] is True
    assert out["moved"] is None, "the game camera ran in the engineering view"


@pytest.mark.skipif(jsx.NODE is None, reason="node is not installed")
def test_the_glass_writes_the_numbers_it_was_HANDED():
    """End to end: a state dict from Python, through the real apply script,
    onto the real ids the real skeleton carries."""
    from gui.v4 import hud

    st = hud.state(**_kw(speed=47.4, altitude=312.0, heading_deg=95.0,
                         roll_deg=-12.0, alpha_deg=3.4, g=1.05,
                         throttle=120.0, throttle_max=400.0))
    out = _js(f"""
      push({{q: null, z: 0, w: [0, 0, -300], game: true,
             hud: {json.dumps(st)}}});
      A.last = 1000; tick(1016);
      const n = {{}};
      // the stub keys a child as 'root>#id' so two builds cannot share
      // one glass; this test only cares WHICH node, not whose
      for (const [k, v] of __nodes)
        n[k.slice(k.indexOf('>') + 2)] = {{t: v.textContent, a: v.attrs}};
      OUT = {{nodes: n}};
    """)
    n = out["nodes"]
    assert n["hud-ias"]["t"] == "47"
    assert n["hud-alt"]["t"] == "312"
    assert n["hud-hdg"]["t"] == "095"
    assert n["hud-alpha"]["t"] == "a +3.4"
    assert n["hud-g"]["t"] == "g +1.05"
    assert n["hud-thr"]["a"]["height"] == f"{hud._THR_H * 0.3:.1f}"
    assert "rotate(12.00" in n["hud-ladder"]["a"]["transform"]
    # every id the skeleton offers was actually addressed
    skel = hud.skeleton()
    for key in n:
        assert f'id="{key}"' in skel, f"the script wrote into {key}, which " \
                                      f"the skeleton does not have"


@pytest.mark.skipif(jsx.NODE is None, reason="node is not installed")
def test_the_glass_only_writes_a_node_that_CHANGED():
    """Setting textContent to what it already says still dirties the node for
    layout — sixty times a second, for a number that did not move."""
    from gui.v4 import hud

    st = json.dumps(hud.state(**_kw(speed=47.4, altitude=312.0)))
    out = _js(f"""
      A.last = 1000; let t = 1000;
      push({{q: null, z: 0, w: [0, 0, -300], game: true, hud: {st}}});
      t += 16; tick(t);
      const first = __writes;
      for (let i = 0; i < 30; i++) {{ t += 16; tick(t); }}
      OUT = {{first, extra: __writes - first}};
    """)
    assert out["first"] > 20, "the glass was never written at all"
    assert out["extra"] == 0, (f"{out['extra']} DOM writes for 30 frames in "
                               f"which nothing changed")


def test_the_browser_half_CLAMPS_its_frame_time_at_both_ends():
    """The upper clamp is a tab that was in the background for a minute. The
    lower one matters more and is easier to miss: a negative dt makes
    ``exp(-dt/tau)`` overflow, the filter gain go to -Infinity, and the
    aeroplane land somewhere no coordinate can name."""
    from gui.v4 import live

    assert "Math.max(0, Math.min(" in live._JS
    assert "0.25" in live._JS


def test_the_browser_half_SLERPS_the_attitude():
    """A rotation matrix cannot be interpolated; a quaternion can. That is
    the whole reason the report carries four numbers and not nine."""
    from gui.v4 import live

    assert "slerp" in live._JS


def test_the_game_camera_costs_no_messages():
    """It is computed from the pose ON SCREEN, so it is both smoother than a
    pose that arrived 30 ms ago and free."""
    from gui.v4 import live

    assert "camera.position.set" in live._JS
    assert "camera.lookAt" in live._JS
    # ...and the tween is disowned, or it fights the direct writes
    assert "camera_tween = null" in live._JS


# ------------------------------------------- a matrix, as a quaternion

def test_a_rotation_matrix_survives_the_round_trip_as_a_quaternion():
    """Every attitude, not just the easy branch: the naive single-branch
    conversion loses all its precision near a half turn, and a half turn is
    an ordinary attitude for an aeroplane."""
    rng = np.random.default_rng(0)
    for _ in range(200):
        A = rng.normal(size=(3, 3))
        Q, R_ = np.linalg.qr(A)
        Q = Q * np.sign(np.diag(R_))
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1.0
        x, y, z, w = W.quat_from_dcm(Q)
        assert abs(x * x + y * y + z * z + w * w - 1.0) < 1e-9
        # rebuild the matrix from the quaternion
        back = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])
        assert np.allclose(back, Q, atol=1e-9)


def test_the_half_turns_are_exactly_the_branches_that_used_to_break():
    """trace = -1 is where ``sqrt(1 + trace)`` dies."""
    for R_ in (np.diag([1.0, -1.0, -1.0]), np.diag([-1.0, 1.0, -1.0]),
               np.diag([-1.0, -1.0, 1.0])):
        assert np.trace(R_) == pytest.approx(-1.0)
        x, y, z, w = W.quat_from_dcm(R_)
        assert abs(x * x + y * y + z * z + w * w - 1.0) < 1e-12


@pytest.mark.skipif(jsx.NODE is None, reason="node is not installed")
def test_the_numbers_under_the_picture_are_written_into_the_DOM():
    """The read-out and the banner travel with the frame, not as a nicegui
    element update.

    This is the whole lag fix and it has to be asserted where it happens.
    Measured in a visible Chrome, flying, by dropping one socket message kind
    at a time: the element updates alone took the display from 120 Hz to
    97.7 Hz and its 99th-percentile frame from 9.4 ms to 33.8 ms, while the
    pose pushes — four times as many messages — cost nothing measurable.
    """
    out = _js("""
      tick(1000);
      push({q: null, z: 0, w: [0, 0, 0], game: true,
            txt: {readout: 'V 14.60', banner: 'ALPHA', color: '#b26a00'}});
      tick(1016);
      const R = document.getElementById('c' + CFG.readout);
      const B = document.getElementById('c' + CFG.banner);
      OUT = {readout: R.textContent, banner: B.textContent,
             colour: B.style.color};
    """)
    assert out["readout"] == "V 14.60"
    assert out["banner"] == "ALPHA"
    assert out["colour"] == "#b26a00"


@pytest.mark.skipif(jsx.NODE is None, reason="node is not installed")
def test_the_LIVE_THRUST_reaches_the_panel_that_owns_its_limit():
    """RUN, not grepped — because this one shipped broken.

    The server was building the string and putting it in the frame payload,
    a Python test asserted it was there, and the browser half had no branch
    for it at all: the label under "max thrust" sat at the value it was
    created with while W ran the throttle up and a typed ceiling of 240 N
    never appeared. The payload test could not see it and neither could
    reading the source, because the missing thing is an absence.

    So the assertion is on the DOM the shipped script writes into.
    """
    out = _js("""
      tick(1000);
      push({q: null, z: 0, w: [0, 0, 0], game: true,
            txt: {readout: 'x', banner: '', color: 'inherit',
                  thrust: '22 N   (0 \u2013 54 N)'}});
      tick(1016);
      const first = document.getElementById('c' + CFG.thrust).textContent;
      push({q: null, z: 0, w: [0, 0, 0], game: true,
            txt: {readout: 'x', banner: '', color: 'inherit',
                  thrust: '137 N   (0 \u2013 240 N)'}});
      tick(1032);
      OUT = {first, second: document.getElementById('c' + CFG.thrust)
                              .textContent};
    """)
    assert out["first"] == "22 N   (0 \u2013 54 N)"
    assert out["second"] == "137 N   (0 \u2013 240 N)", \
        "the throttle value and its band never reached the panel"


@pytest.mark.skipif(jsx.NODE is None, reason="node is not installed")
def test_an_unchanged_number_is_not_written_again():
    """Guarded on change, like the glass. A frame that says the same thing
    must touch no node at all — otherwise this trades a Vue patch for a
    layout-invalidating DOM write thirty times a second and buys nothing."""
    out = _js("""
      tick(1000);
      const txt = {readout: 'same', banner: '', color: 'inherit'};
      push({q: null, z: 0, w: [0, 0, 0], game: true, txt});
      tick(1016);
      const after_first = globalThis.__writes;
      for (let i = 0; i < 10; i++) { push({q: null, z: 0, w: [0, 0, 0],
        game: true, txt}); tick(1016 + 16 * (i + 1)); }
      const after_ten = globalThis.__writes;
      push({q: null, z: 0, w: [0, 0, 0], game: true,
            txt: {readout: 'changed', banner: '', color: 'inherit'}});
      tick(1200);
      OUT = {first: after_first, ten: after_ten, changed: globalThis.__writes};
    """)
    assert out["first"] > 0, "the first frame wrote nothing"
    assert out["ten"] == out["first"], \
        f"{out['ten'] - out['first']} writes for numbers that did not change"
    assert out["changed"] > out["ten"], "a changed number was not written"


@pytest.mark.skipif(jsx.NODE is None, reason="node is not installed")
def test_a_warning_that_clears_also_clears_its_colour():
    """The banner used to set a colour on the warning branches and only the
    TEXT on the quiet one, so a stall warning that cleared left the next
    empty banner red for ever. The colour travels with the text."""
    out = _js("""
      tick(1000);
      push({q: null, z: 0, w: [0, 0, 0], game: true,
            txt: {readout: 'x', banner: 'ALPHA', color: '#b26a00'}});
      tick(1016);
      const hot = document.getElementById('c' + CFG.banner).style.color;
      push({q: null, z: 0, w: [0, 0, 0], game: true,
            txt: {readout: 'x', banner: '', color: 'inherit'}});
      tick(1032);
      const cool = document.getElementById('c' + CFG.banner).style.color;
      OUT = {hot, cool};
    """)
    assert out["hot"] == "#b26a00"
    assert out["cool"] == "inherit", "the banner stayed warning-coloured"


@pytest.mark.skipif(jsx.NODE is None, reason="node is not installed")
def test_a_REBUILT_view_is_re_resolved_and_not_left_pointing_at_the_old_one():
    """THE PICTURE.

    ``window.__aerobo`` is created once per page but installed once per BUILT
    view, and the shell builds the fly view at least twice on the way to
    stage 6 — once when the page lays out all six stages, and again when the
    pilot clicks into it. Everything the first install resolved (the scene
    component, the group, the world, the glass, the read-out) belongs to a
    view that has since been cleared.

    Keeping those handles produced no error anywhere: the stale objects still
    moved, and every number read back correctly from the driver. On screen
    the camera never left its default — the aeroplane filled the frame from a
    few centimetres away — no ground appeared, the glass stayed blank and the
    read-out under it stayed empty. Which is what was actually being reported
    as "doesn't work very well".
    """
    from gui.v4 import hud, live

    cfg2 = dict(jsx.CFG, scene=91, group="g2", world="w2", hud=92,
                readout=93, banner=94)
    out = jsx.run(jsx.CFG, """
      tick(1000);
      push({q: null, z: 10, w: [1, 2, -300], game: true,
            txt: {readout: 'FIRST', banner: '', color: 'inherit'}});
      tick(1016);
      const oldGroup = A.group, oldScene = A.scene;
      // where the OLD view was left. It smoothed one frame toward z = 10,
      // so this is not 10 — what matters is that it never moves again.
      const frozen = {z: oldGroup.position.z, cam: oldScene.camera.position.x,
                      txt: document.getElementById('c' + CFG.readout).textContent};

      // ---- the view is rebuilt: new scene, new objects, new elements
      const SCENE2 = {objects: new Map(), camera: mkObj(),
                      look_at: new V3(), controls: {enabled: true,
                      target: new V3()}, camera_tween: {}};
      SCENE2.camera.lookAt = function (x, y, z) { this.lookedAt = [x, y, z]; };
      SCENE2.objects.set(CFG2.group, mkObj());
      SCENE2.objects.set(CFG2.world, mkObj());
      globalThis.getElement = (id) => (id === CFG2.scene ? SCENE2 : null);
      A.init(CFG2, APPLY, GAUGE);

      push({q: null, z: 40, w: [5, 6, -300], game: true,
            txt: {readout: 'SECOND', banner: '', color: 'inherit'}});
      tick(1032);
      OUT = {
        scene_swapped: A.scene === SCENE2,
        new_group_z: SCENE2.objects.get(CFG2.group).position.z,
        old_group_moved: oldGroup.position.z !== frozen.z,
        old_cam_moved: oldScene.camera.position.x !== frozen.cam,
        old_txt_frozen: frozen.txt,
        new_scene_cam: SCENE2.camera.position.x,
        new_readout: document.getElementById('c' + CFG2.readout).textContent,
        old_readout: document.getElementById('c' + CFG.readout).textContent,
      };
    """, live_js=(
        f"const CFG2 = {json.dumps(cfg2)};\n" + live._JS),
        apply_js=hud.APPLY_JS)

    assert out["scene_swapped"] is True, "the driver kept the destroyed scene"
    # the rebuilt view's group is the one that moves, and it SNAPS (the first
    # tick after an install is the first frame, so no smoothing)
    assert out["new_group_z"] == pytest.approx(40.0)
    assert out["new_scene_cam"] == pytest.approx(-26.0), \
        "the rebuilt view got no chase camera"
    assert out["new_readout"] == "SECOND"
    # ...and NOTHING in the view that was cleared is touched again
    assert out["old_group_moved"] is False, \
        "the destroyed view's group was still being driven"
    assert out["old_cam_moved"] is False, \
        "the destroyed view's camera was still being driven"
    assert out["old_readout"] == out["old_txt_frozen"] == "FIRST", \
        "the destroyed view's read-out was written again"


# ============================================================================
# THE ATTITUDE PANEL, EXECUTED
# ============================================================================

def _gauge_run(script: str) -> dict:
    from gui.v4 import gauge as gg, hud, live

    return jsx.run(jsx.CFG, script, live_js=live._JS, apply_js=hud.APPLY_JS,
                   gauge_js=gg.APPLY_JS)


def _panel(**kw):
    from gui.v4 import gauge as gg

    base = dict(roll_deg=0.0, pitch_deg=0.0, yaw_deg=0.0,
                rates_dps=(0.0, 0.0, 0.0))
    base.update(kw)
    return gg.attitude_state(**base)


def test_the_panel_is_written_into_the_DOM():
    """Not asserted about the string — RUN. The apply function is generic
    (it takes ``{text, attr}`` maps and knows nothing about a bank angle),
    so the only thing worth checking is that the maps land on the right
    nodes of the right root."""
    st = _panel(roll_deg=-12.4, pitch_deg=1.5, yaw_deg=20.1,
                rates_dps=(-4.1, 0.3, -0.2))
    out = _gauge_run("""
      push({q: null, z: 0, w: [0,0,0], game: true, g: %s});
      tick(0); tick(16);
      const P = document.getElementById('c' + CFG.panel);
      OUT = {
        roll: P.querySelector('#g-ang-roll').textContent,
        pitch: P.querySelector('#g-ang-pitch').textContent,
        yaw: P.querySelector('#g-ang-yaw').textContent,
        p: P.querySelector('#g-rate-roll').textContent,
        q: P.querySelector('#g-rate-pitch').textContent,
        r: P.querySelector('#g-rate-yaw').textContent,
        wings: P.querySelector('#g-wings').attrs.transform,
        pbar: P.querySelector('#g-ratebar-roll').attrs.width,
      };
    """ % json.dumps(st))
    assert (out["roll"], out["pitch"], out["yaw"]) == ("-12.4", "+1.5",
                                                       " 20.1")
    assert (out["p"], out["q"], out["r"]) == ("-4.1", "+0.3", "-0.2")
    assert out["wings"] == st["attr"]["g-wings"]["transform"]
    assert out["pbar"] == st["attr"]["g-ratebar-roll"]["width"]


def test_two_panels_do_not_share_a_node_cache():
    """The cache lives on the ROOT element, so one apply function can serve
    any number of panels and a lookup made for one is never handed to
    another. Driven directly, with two roots and two states."""
    from gui.v4 import gauge as gg

    a = _panel(roll_deg=5.0)
    b = _panel(roll_deg=-45.0)
    out = jsx.run(jsx.CFG, """
      const A1 = document.getElementById('cA');
      const B1 = document.getElementById('cB');
      GAUGE(A1, %s); GAUGE(B1, %s);
      // ...and again the other way round, which a shared cache would skip
      GAUGE(B1, %s); GAUGE(A1, %s);
      OUT = {a: A1.querySelector('#g-ang-roll').textContent,
             b: B1.querySelector('#g-ang-roll').textContent};
    """ % (json.dumps(a), json.dumps(b), json.dumps(b), json.dumps(a)),
        live_js=__import__("gui.v4.live", fromlist=["live"])._JS,
        apply_js=__import__("gui.v4.hud", fromlist=["hud"]).APPLY_JS,
        gauge_js=gg.APPLY_JS)
    assert out["a"] == "+5.0"
    assert out["b"] == "-45.0", \
        "the second panel got the first one's cached node"


def test_an_unchanged_panel_number_is_not_written_again():
    """Setting textContent to what it already says still dirties the node
    for layout, thirty times a second, for ever."""
    out = _gauge_run("""
      push({q: null, z: 0, w: [0,0,0], game: true, g: %s});
      tick(0); tick(16);
      const first = globalThis.__writes;
      for (let i = 0; i < 40; i++) tick(16 + 16 * (i + 1));
      OUT = {first: first, after: globalThis.__writes};
    """ % json.dumps(_panel(roll_deg=1.0, rates_dps=(2.0, 0.0, 0.0))))
    assert out["first"] > 0, "nothing was ever written"
    assert out["after"] == out["first"], \
        f"{out['after'] - out['first']} redundant writes over 40 frames"


def test_a_REBUILT_view_re_resolves_its_panel_too():
    """The bug that cost the whole picture, on the new element: the driver
    is created once per PAGE and installed once per BUILT VIEW, so every
    handle it holds belongs to a view that may since have been cleared."""
    out = _gauge_run("""
      push({q: null, z: 0, w: [0,0,0], game: true, g: %s});
      tick(0); tick(16);
      const OLD = document.getElementById('c' + CFG.panel);
      const old_first = OLD.querySelector('#g-ang-roll').textContent;
      const CFG2 = Object.assign({}, CFG, {scene: 91, group: 'g2',
                                           world: 'w2', panel: 95});
      const SCENE2 = {objects: new Map(), camera: mkObj(),
                      look_at: new V3(),
                      controls: {enabled: true, target: new V3()},
                      camera_tween: {}};
      SCENE2.camera.lookAt = function (x, y, z) { this.lookedAt = [x,y,z]; };
      SCENE2.objects.set(CFG2.group, mkObj());
      SCENE2.objects.set(CFG2.world, mkObj());
      globalThis.getElement = (id) => (id === CFG2.scene ? SCENE2 : null);
      window.__aerobo.init(CFG2, APPLY, GAUGE);
      push({q: null, z: 0, w: [0,0,0], game: true, g: %s});
      tick(200); tick(216);
      OUT = {
        old_first: old_first,
        old_now: OLD.querySelector('#g-ang-roll').textContent,
        new_now: document.getElementById('c' + CFG2.panel)
                   .querySelector('#g-ang-roll').textContent,
      };
    """ % (json.dumps(_panel(roll_deg=7.0)),
           json.dumps(_panel(roll_deg=-33.0))))
    assert out["old_first"] == "+7.0"
    assert out["new_now"] == "-33.0", "the new panel never received a frame"
    assert out["old_now"] == "+7.0", \
        "the driver went on writing a panel from the destroyed view"


def test_nothing_the_panel_draws_is_OUTSIDE_its_viewBox():
    """THE BUG A SCREENSHOT FOUND AND NO TEST COULD.

    The panel's height was a typed 250 while its last caption sat at
    y = 255, so that row was outside the viewBox and the browser silently
    did not draw it. Every assertion passed: the skeleton was a correct
    string, the state dict correct numbers, and the apply function wrote
    them into the right nodes. The only thing wrong was that a coordinate
    was bigger than a constant.

    So the height is now derived from the layout, and this reads every
    coordinate back out of the shipped markup — and fails on a box that is
    far too big, so "make it enormous" is not the fix either.
    """
    import re

    from gui.v4 import gauge as gg

    markup, w, h = gg.attitude_skeleton(), gg.PANEL_W, gg.PANEL_H
    assert f'viewBox="0 0 {w:.0f} {h:.0f}"' in markup
    ys = [float(v) for v in re.findall(r'\by2?="(-?[\d.]+)"', markup)]
    xs = [float(v) for v in re.findall(r'\bx2?="(-?[\d.]+)"', markup)]
    assert ys, "no coordinates were found — this test cannot fire"
    assert max(ys) <= h, f"a row is drawn at y={max(ys)} in a box {h} tall"
    assert min(ys) >= 0.0 and min(xs) >= 0.0
    assert max(xs) <= w, f"something is drawn at x={max(xs)} in a box {w} wide"
    assert h - max(ys) < 40.0, "the panel has a large empty margin"


def test_the_stick_angles_are_written_into_the_panel():
    """Executed, not grepped: the three surface angles and their bars go
    through the same generic apply as everything else, so what is checked is
    that they land on their own nodes."""
    from gui.v4 import gauge as gg

    st = _panel(stick={"elevator": -8.0, "aileron": 25.0, "rudder": 0.0},
                limits={"elevator": 20.0, "aileron": 25.0, "rudder": 25.0})
    out = _gauge_run("""
      push({q: null, z: 0, w: [0,0,0], game: true, g: %s});
      tick(0); tick(16);
      const P = document.getElementById('c' + CFG.panel);
      OUT = {
        elev: P.querySelector('#g-st-elevator').textContent,
        ail: P.querySelector('#g-st-aileron').textContent,
        rud: P.querySelector('#g-st-rudder').textContent,
        elev_x: P.querySelector('#g-stbar-elevator').attrs.x,
        ail_x: P.querySelector('#g-stbar-aileron').attrs.x,
        ail_w: P.querySelector('#g-stbar-aileron').attrs.width,
      };
    """ % json.dumps(st))
    assert (out["elev"], out["ail"], out["rud"]) == ("-8.0", "+25.0", "+0.0")
    # a negative deflection grows to the LEFT of centre, a positive one to
    # the right — the bar starts at the mid-point in one case and short of
    # it in the other
    assert float(out["elev_x"]) < float(out["ail_x"])
    # full aileron fills its half exactly, because each bar is scaled by its
    # own stop rather than by a shared one
    assert float(out["ail_w"]) == pytest.approx(
        gg._STK_X1 - gg._STK_MID, abs=1e-6)


def test_the_thrust_NUMBER_is_on_the_glass():
    """It moved off a Quasar slider that could not follow the keyboard and
    onto the HUD, beside the bar it fills. Executed through the glass's own
    apply function, so a number that never reaches a text node fails here."""
    from gui.v4 import hud

    st = hud.state(speed=20.0, altitude=300.0, heading_deg=20.0,
                   pitch_deg=1.5, roll_deg=0.0, alpha_deg=1.5, g=1.0,
                   throttle=35.0, throttle_min=0.0, throttle_max=54.0)
    out = _js("""
      push({q: null, z: 0, w: [0,0,0], game: true, hud: %s});
      tick(0); tick(16);
      const H = document.getElementById('c' + CFG.hud);
      OUT = {n: H.querySelector('#hud-thrn').textContent,
             h: H.querySelector('#hud-thr').attrs.height};
    """ % json.dumps(st))
    assert out["n"] == "35"
    assert float(out["h"]) == pytest.approx(
        hud._THR_H * 35.0 / 54.0, abs=0.2)


def test_no_two_things_on_the_glass_share_the_same_GROUND():
    """THE OTHER HALF OF THE viewBox BUG: inside the box, on top of each
    other.

    The throttle column started at ``VIEW_H - 190`` = 430 and the altitude
    tape's box runs to ``_CY + 150`` = 460 — and they share an x range, so
    the bar's top thirty units and the number written above it were drawn
    underneath a tape full of numbers. Legible in neither, and again nothing
    in the state dict or the DOM could see it.

    Only the two that actually collided are checked, as rectangles, because
    a general no-overlap rule over a HUD would be false (the ladder crosses
    everything on purpose).
    """
    from gui.v4 import hud

    thr = (hud._THR_X, hud._THR_TOP, hud._THR_X + 26.0, hud._THR_BOT)
    # the altitude tape's box, from _tape_skeleton
    alt = (hud._ALT_X - 46.0, hud._CY - hud._TAPE_H / 2,
           hud._ALT_X + 46.0, hud._CY + hud._TAPE_H / 2)
    # the NUMBER is part of the column: it is written above the bar, and it
    # was the half of this that a shorter bar did not fix
    thr = (thr[0], hud._THR_TOP - 30.0, thr[2], thr[3])
    overlap_x = min(thr[2], alt[2]) - max(thr[0], alt[0])
    overlap_y = min(thr[3], alt[3]) - max(thr[1], alt[1])
    assert not (overlap_x > 0 and overlap_y > 0), (
        f"the throttle column and the altitude tape overlap by "
        f"{overlap_x:.0f} x {overlap_y:.0f} units")
    assert hud._THR_BOT <= hud.VIEW_H and hud._THR_TOP - 30.0 >= 0.0
