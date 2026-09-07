"""Stage 6 — FLIGHT: fly the design that was scored.

Six degrees of freedom on the derivative deck stage 5 built, rendered live in
a three.js scene whose aircraft is the SAME lofted surface the Results stage
exports to STL. There is no second geometry and no second aerodynamic model
anywhere in this stage; if the picture and the numbers ever disagree, one of
them is a bug rather than a different approximation.

WHAT IS BESIDE THE PICTURE, AND WHY. Weight, thrust, speed, altitude and
the CG are not set-up fields asked before the run — they are levers down the
right of the moving scene, and they move while it flies. That is the whole
stage: nobody learns what a static margin is by typing a number into a form
and pressing go. They learn it by dragging the CG aft until the readout's SM
goes red and the aeroplane starts to diverge under them.

Two of those levers mean a RE-TRIM (speed and altitude) and three do not
(thrust, weight, CG). The difference is stated on the panel rather than
hidden, because a re-trim restarts the clock and the trace.

THE TWO PANELS are :mod:`gui.v4.gauge` — bank and roll rate on the left,
the throttle and the two stops it works between on the right. Both are
written by the frame path and not by nicegui, for the reason measured in
:mod:`gui.v4.live`: an element update once a frame costs a quarter of the
frame. The right-hand one exists because W and S moved a number with
nowhere to be seen; the left-hand one because roll is the axis that goes
wrong first and a tilted bar reads faster than a signed number in a row of
thirty.

THE MODES ARE NAMED, NOT SORTED (:mod:`gui.v4.modes`), and five canned
excitations fly them (:mod:`gui.v4.manoeuvre`) — a phugoid is a prediction
until something has actually flown one.

Flown from the keyboard (:mod:`gui.v4.stick`):

    ↑ / ↓   pitch up / down          W / S   thrust up / down
    ← / →   yaw left / right         A / D   roll left / right

...or from a gamepad (:mod:`gui.v4.pad`), which is the same demand with the
resolution a keyboard cannot express: right stick pitch and roll, left stick
yaw, the triggers as a throttle, ✕ to fly or pause. Both hands are ADDED
onto one demand rather than one overruling the other.

The stick is spring-centred — hold to deflect, release and it comes back —
and the throttle is not, because a throttle stays where it is left.

No propeller. Thrust is a scalar along body x, with an optional offset from
the CG so throttle is a pitch input as well as a speed one, and the honesty
panel says exactly that.

The render loop is a NiceGUI timer that advances the integrator and then
moves ONE scene group per frame. Moving a group is a single websocket
message carrying a position and a rotation MATRIX (``Object3D.rotate_R``,
straight off the quaternion), so there is no Euler singularity in the render
path and no per-frame geometry rebuild.
"""

from __future__ import annotations

import hashlib
import time
import traceback

import numpy as np

from aerobo import fin as _fin

from gui.v3 import figstyle, theme, widgets

from .. import (fields as fl, gauge as gg, hud as hud_svg, live as lv,
                manoeuvre as mv, modes as md, pad as pd, stick as stk,
                world as wld)

#: how often the SERVER REPORTS a pose. This is not the frame rate and has
#: not been since :mod:`gui.v4.live` existed: the browser smooths between
#: reports and draws at the display's own rate, so this number buys latency
#: (half a push interval of it) rather than smoothness.
#:
#: It stopped mattering for the PHYSICS the moment the integrator started
#: reading a wall clock instead of counting ticks — which is the whole reason
#: this number can be chosen for the network rather than for the flying.
#: (asyncio.sleep overshoots by about 1.06 ms, so 1/32 lands near 30 Hz.)
FPS = 32.0

#: how many frames between two redraws of the Traces tab, when it is the tab
#: on screen. Six plotly payloads is a real message and the eye reads a
#: trace at two updates a second as live, so this is FPS/2 rounded to a
#: frame count rather than a rate of its own.
TRACE_EVERY = 16

#: the integrator's OWN step. Fixed, and much smaller than a frame: the
#: physics must not change when the frame rate does.
SUBSTEP_S = 1.0 / 240.0

#: the most wall-clock a single frame is allowed to make up. Without this, a
#: tab left in the background for a minute comes back and integrates a minute
#: of flight in one frame — which is both a freeze and, at 240 Hz, a
#: 14 000-step loop.
MAX_CATCHUP_S = 0.25

DEG = 57.29577951308232
RAD = 0.017453292519943295

#: both side columns, at whatever width is substituted in. They scroll on
#: their own rather than with the page, and they are the same height as the
#: viewport between them, so a short window squeezes the picture instead of
#: pushing a lever off the bottom of the screen.
#: the two lever columns beside the picture. THEIR HEIGHT IS THE ROW'S, and
#: the row's is what is left of the PANE — not of the window.
#:
#: It used to be ``calc(100vh - 300px)``, which is a window measurement made
#: inside a pane the window does not own: the work area is one of four panes
#: and about 520 px tall on a 763 px viewport, so a 463 px picture with 145 px
#: of chrome above it overflowed by 90 px and the view scrolled. Scrolling it
#: far enough to read the numbers under the picture took the TOP of the left
#: panel — the bank dial and the whole ROLL row — off the top of the pane.
#: That is what "I can't see the rates" was: they were computed, pinned by
#: two tests, pushed thirty times a second, and off the screen.
_SIDE = "width:%dpx;flex:0 0 %dpx;overflow-y:auto;min-height:0;height:100%%"

#: the Fly view is a flex COLUMN that fills its pane and never scrolls: the
#: picture row takes what is left, and everything under it gets its own
#: scroller so it can never push the row out of view.
_FLY_COL = ("display:flex;flex-direction:column;height:100%;min-height:0;"
            "overflow:hidden")
_FLY_ROW = "flex:1 1 auto;min-height:340px;min-width:0"
_FLY_REST = "flex:0 1 auto;min-height:0;overflow-y:auto"

# ---------------------------------------------------------------- the world
#: the period the floating origin snaps to. EVERY ground lattice must have a
#: period that divides this, or the re-anchor will not map it onto itself and
#: the scenery will jump once a tile. Grid lines and posts are both chosen
#: against it, and :mod:`gui.v4.world` explains why.
WORLD_SNAP_M = 480.0
GRID_PITCH_M = 120.0            # 480 = 4 x 120
POST_PITCH_M = 480.0            # ...and 480 = 1 x 480
GROUND_HALF_M = 2400.0          # how far the drawn ground reaches
POST_HALF_M = 1920.0

#: camera stand-off, in SPANS, so the same numbers frame any aeroplane.
#: ``LEAD`` is how far ahead of the aeroplane the camera looks: some is what
#: makes it feel like flying rather than watching, too much drops the
#: aeroplane out of the bottom of the frame and away from the boresight.
CHASE_BACK = 2.6
CHASE_UP = 0.42
CHASE_LEAD = 0.35

#: how long the ENGINEERING view takes to swing its camera back. The game
#: view no longer uses ``move_camera`` at all — the browser computes the
#: chase eye from the pose it is drawing, which is both smoother and free.
CHASE_TWEEN_S = 0.4

#: THE THREE FRAMES, and the two fixed rotations between them. Getting these
#: wrong is invisible in a screenshot of level flight and wrong in every
#: manoeuvre — a right bank drew the right wing UP before this was written
#: down, because the earth frame was handed to the scene unreversed.
#:
#: * ``G`` — the lattice/loft frame the STL is written in: x AFT, y starboard,
#:   z UP.
#: * ``B`` — flight-dynamics body axes: x FORWARD, y starboard, z DOWN.
#: * ``W`` — what the scene draws: x forward, y PORT, z up. Port rather than
#:   starboard because "x forward, y starboard, z up" is left-handed and a
#:   three.js scene is not.
#:
#: A point fixed in the body at ``v_G`` is drawn at ``M @ C.T @ A @ v_G``,
#: where ``C = quat_to_dcm(q)`` maps earth to body. ``A`` is a half turn about
#: y and ``M`` a half turn about x; both are proper rotations (det +1), so
#: neither mirrors the aeroplane.
A_G_TO_BODY = ((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0))
#: ...and the earth->scene half turn, which :mod:`gui.v4.world` also needs
#: and which must not be written down twice. world.py may not import THIS
#: module (it would drag nicegui into a pure-numpy one), but this module
#: already imports world.py, so the copy belongs here and the definition
#: there.
M_EARTH_TO_SCENE = tuple(tuple(float(v) for v in row)
                         for row in wld.M_EARTH_TO_SCENE)


def scene_rotation(quat):
    """``M @ C.T`` — what the scene has to be handed for an attitude.

    Pure, and at module level, so the frame chain can be checked against a
    closed form without a browser: the whole class of bug this guards is a
    picture that is right in level flight and mirrored in every manoeuvre.
    """
    import numpy as np

    from aerobo import sixdof as sd

    return np.asarray(M_EARTH_TO_SCENE) @ sd.quat_to_dcm(quat).T


def scene_quat(quat):
    """The scene attitude as ``(x, y, z, w)``, ready to SLERP in a browser.

    Four floats instead of nine, and — the reason it exists — a rotation
    matrix cannot be interpolated but a quaternion can, which is what lets
    thirty reports a second be drawn as sixty frames.
    """
    return wld.quat_from_dcm(scene_rotation(quat))


def scene_point(quat, v_G):
    """Where a point of the LOFT ends up on screen: ``M @ C.T @ A @ v_G``."""
    import numpy as np

    return scene_rotation(quat) @ (np.asarray(A_G_TO_BODY)
                                   @ np.asarray(v_G, dtype=float))


def body_offset(x_cg: float, z_cg: float = 0.0) -> tuple:
    """Where the airframe sits so that its CG lands on the group's origin.

    ``t + A @ (x_cg, 0, z_cg) == 0``, and ``A`` is its own inverse, so
    ``t = -A @ (x_cg, 0, z_cg) = (x_cg, 0, z_cg)``. Written out because the
    coincidence that the offset equals the CG station is a property of THIS
    ``A`` and would silently stop being true if the frames moved.
    """
    import numpy as np

    t = -(np.asarray(A_G_TO_BODY) @ np.array([x_cg, 0.0, z_cg]))
    return float(t[0]), float(t[1]), float(t[2])


#: WHICH BROWSER CLIENTS ARE ALREADY SUBSCRIBED to the page-level events
#: (the gamepad, and the release-the-stick report).
#:
#: A module global rather than a session key, because that is the LIFETIME
#: it has. ``ui.on`` registers on the client's layout, which outlives both
#: the view and the session; keeping the registry in ``S["flight"]`` meant
#: File > New session — which refills every sub-dict in place — deleted the
#: record while the live handler was still attached, so the next build
#: subscribed a SECOND listener and every pad report was answered twice
#: (press Fly on the pad: it started and immediately stopped).
_PAGE_WIRED: set = set()


def build(ctx):     # noqa: PLR0915  (one stage, built whole)
    from nicegui import ui

    S = ctx.S
    F = S["flight"]
    C = S["controls"]
    #: everything that belongs to the CURRENT Fly view and dies with it
    view: dict = {"group": None, "body": None, "scene": None, "timer": None,
                  "keyboard": None, "readout": None, "banner": None,
                  "fields": {}, "url": None, "url_key": None,
                  "deck_cg": None, "deck": None, "deck_fm": None,
                  "world": None, "ref_grid": None, "cg_marker": None,
                  "np_marker": None, "eng_cam": None, "span": None,
                  "hud": None, "wall": None, "acc": 0.0,
                  "panel": None, "gauge_state": None, "stab": None,
                  "traces": None, "trace_n": 0,
                  "client": None, "installed": False, "hud_state": None,
                  "padlbl": None}

    # ------------------------------------------------------------- the model
    def _fm():
        """The flight model for the design ON SCREEN NOW.

        Stage 5 owns the rebuild, so ask it rather than reading the cache it
        keeps: a run that replaced ``S["run"]["report"]`` leaves ``C["fm"]``
        pointing at the previous design's lattice, and nothing in this stage
        could tell — the aeroplane simply was not the one stage 4 was
        showing. ``controls_ensure_deck`` is a single identity comparison
        when the deck is already current, and ``ctx.act`` on an unregistered
        name is a no-op, so a headless shell with no stage 5 built behaves
        exactly as before.
        """
        ctx.act("controls_ensure_deck")
        return C.get("fm")

    def _mission_point() -> dict:
        """THE CONDITIONS THE DESIGN WAS SIZED AT — stage 1's, not this
        stage's. Stage 6 flies the aeroplane at the point it was designed
        for unless the pilot says otherwise, so this is where the speed, the
        density and the weight come from, and it is shown beside the picture
        so nobody has to trust that it happened.
        """
        from gui.v3 import session as v3s

        try:
            dp = v3s.design_point(S)
        except Exception:                              # noqa: BLE001
            dp = {}
        if not isinstance(dp, dict) or dp.get("error"):
            dp = {}
        m = S.get("mission") or {}
        # NOT the design weight and NOT the design CL. Both were in here
        # and neither had a reader: the weight the simulation flies comes
        # off ``trimmed.inertia.mass_kg`` through the mass lever, so a
        # second copy here could only ever disagree with it.
        return {"V": float(dp.get("v_ms") or m.get("V") or 45.0),
                "rho": float(dp.get("rho") or 1.225),
                "altitude_m": float(dp.get("altitude_m")
                                    or m.get("altitude_m") or 0.0),
                "medium": dp.get("medium") or "air"}

    def _V():
        return float(F.get("V_trim") or _mission_point()["V"])

    def _altitude() -> float:
        """The datum the altitude trace is drawn against.

        The density the simulation flies is the MISSION's and is constant
        (:class:`sixdof.Aircraft` carries one rho), so this number is a
        datum, not a second atmosphere. It opens on the mission's own
        altitude — except at sea level, where an aeroplane that starts at
        zero has nowhere to descend to and every trace clips at the floor.
        """
        if F.get("altitude_m") is not None:
            return float(F["altitude_m"])
        alt = _mission_point()["altitude_m"]
        return float(alt) if alt >= 50.0 else 300.0

    def _conditions(ac):
        """The flight CONDITION answers — the ones that mean a re-trim."""
        from dataclasses import replace

        from aerobo import sixdof as sd

        if F.get("CD0"):
            ac = replace(ac, CD0=float(F["CD0"]))
        if F.get("oswald_e"):
            ac = replace(ac, oswald_e=float(F["oswald_e"]))
        if not F.get("stall_on", True):
            ac = replace(ac, stall=sd.Stall(enabled=False))
        return ac

    def _live(ac, *, with_thrust: bool = True):
        """Apply the three LIVE levers on top of an aircraft.

        The CG is not a cosmetic field: moving it REBUILDS the derivative
        deck, because the static margin is a moment arm. The rebuilt deck is
        cached so that dragging the slider back and forth over a value
        already seen costs nothing.

        THE KEY IS THE MODEL AS WELL AS THE CG, and it has to be. Keyed on
        the CG number alone, this cache was never invalidated by anything —
        not by a new report, not by ``_teardown``, not by stage 5 — so once
        the CG lever had been touched, changing the FIN re-armed onto the
        PREVIOUS fin's deck while ``AR_vertical`` and the drag term updated
        to the new one. Measured before this line, on a 2.5 m fin: the
        aeroplane flew ``Cn_beta`` 0.106967 while the stage beside it
        printed 0.367735, and four seconds of held rudder ended at 5.859 deg
        of roll instead of 10.450. ``_fm`` returns a NEW ``FlightModel``
        whenever stage 5 rebuilds (controls.py ``rebuild``), so identity is
        the whole test — and holding the model in ``view`` is what keeps
        that identity meaningful rather than a recycled ``id()``.
        """
        from dataclasses import replace

        from aerobo import dynamics as dyn, sixdof as sd

        fm, L = _fm(), F["live"]
        cg = L.get("x_cg_m")
        if cg is not None and fm is not None \
                and abs(float(cg) - float(ac.deck.x_cg)) > 1e-12:
            if view["deck_cg"] != float(cg) or view["deck_fm"] is not fm:
                view["deck"] = dyn.deck(
                    fm.model, x_cg=float(cg), mac=ac.deck.mac, b=ac.deck.b,
                    controls=fm.controls, alpha=fm.alpha_trim,
                    i_t=fm.i_t_trim)
                view["deck_cg"] = float(cg)
                view["deck_fm"] = fm
            ac = replace(ac, deck=view["deck"])
        if L.get("mass_kg"):
            i = ac.inertia
            scale = float(L["mass_kg"]) / i.mass_kg
            ac = replace(ac, inertia=sd.Inertia(
                float(L["mass_kg"]), i.Ixx * scale, i.Iyy * scale,
                i.Izz * scale, basis=i.basis + " (mass moved in flight)"))
        if with_thrust and L.get("thrust_n") is not None:
            ac = replace(ac, prop=replace(ac.prop,
                                          thrust_n=float(L["thrust_n"])))
        return ac

    def arm():
        """Trim the aircraft and put it in the air. Idempotent.

        Trims at whatever the live levers currently say, so re-arming after
        dragging the CG aft gives the trim for THAT aeroplane rather than
        for the one the search produced.
        """
        from aerobo import sixdof as sd

        fm = _fm()
        if fm is None:
            F["error"] = "stage 5 has not built a deck yet"
            F["armed_fm"] = None
            return False
        # WHY THERE IS NO PITCH CONTROL, IN THE WORDS OF THE ANSWER THAT
        # REMOVED IT. ``trim_level`` solves for alpha AND one control, so a
        # deck with no elevator column cannot be trimmed at all — and what
        # it raises is "no control column named 'elevator' in this deck;
        # available: ['aileron', 'rudder']", which names a data structure
        # rather than a question the pilot can go and answer. The only way
        # to get here now is a configuration with no second surface: stage
        # 5's elevator switch means "all-moving" and sends a whole-chord
        # hinge (see controls.py::_spec).
        if "elevator" not in fm.deck.columns:
            F["error"] = (
                "this aeroplane has nothing to trim it in pitch: it carries "
                "one lifting surface, so there is no elevator and no "
                "all-moving stabiliser either. Turn “add a tail (trims and "
                "stabilises)” on in stage 1 and run the design again — a "
                "wing alone can be scored, but nothing balances its "
                "pitching moment, so there is no level flight to start "
                "from.")
            F["ac"] = F["state"] = None
            F["armed_fm"] = None
            return False
        try:
            L = F["live"]
            ac = _live(_conditions(fm.aircraft), with_thrust=False)
            V = _V()
            st, trimmed = sd.trim_level(ac, V=V, altitude_m=_altitude())
            # the levers start AT the design: whatever was not answered is
            # filled from the aeroplane that was just trimmed, so the first
            # frame is the one the search produced and every later frame is
            # a departure the pilot made on purpose
            L.setdefault("thrust_n", None)
            if L.get("thrust_n") is None:
                L["thrust_n"] = float(trimmed.prop.thrust_n)
            else:
                from dataclasses import replace
                trimmed = replace(trimmed, prop=replace(
                    trimmed.prop, thrust_n=float(L["thrust_n"])))
            if L.get("mass_kg") is None:
                L["mass_kg"] = float(trimmed.inertia.mass_kg)
            if L.get("x_cg_m") is None:
                L["x_cg_m"] = float(trimmed.deck.x_cg)
            F["ac"] = trimmed
            F["state"] = st
            F["trim"] = {"alpha_deg": st.alpha * DEG,
                         "elevator_deg": trimmed.controls.get("elevator", 0.0)
                         * DEG,
                         "thrust_n": float(trimmed.prop.thrust_n), "V": V,
                         "mass_kg": float(trimmed.inertia.mass_kg),
                         "x_cg_m": float(trimmed.deck.x_cg),
                         "mac": float(trimmed.deck.mac)}
            F["history"] = [st]
            F["history_t"] = [0.0]
            F["t"] = 0.0
            view["wall"] = None
            view["acc"] = 0.0
            F["error"] = None
            F["crashed"] = False
            F["crash_banner"] = None
            # WHICH DECK THIS AEROPLANE IS. Stage 5 rebuilds the flight model
            # on every edit but refreshes only the four V3 stages and its own
            # (controls.py::edit), so nothing here is told that the aeroplane
            # changed: ``_render_fly`` re-armed only when ``F["ac"]`` was
            # None, and an ``ac`` left over from the previous deck is not
            # None. Selecting stage 6 after moving an aileron band therefore
            # went on flying the PREVIOUS aircraft while stage 5 described a
            # different one. The model object itself, not its id() — a dead
            # model's id can be handed to its replacement.
            F["armed_fm"] = fm
            F["stick"].update(dict.fromkeys(F["stick"], 0.0))
            F["hold"].update(dict.fromkeys(F["hold"], 0))
            F.setdefault("keys_down", set()).clear()
            return True
        except Exception as exc:                       # noqa: BLE001
            F["error"] = f"{type(exc).__name__}: {exc}"
            F["ac"] = F["state"] = None
            F["armed_fm"] = None
            print("[v4] flight arm failed:\n" + traceback.format_exc())
            return False

    # ------------------------------------------------------- the live levers
    #: the throttle's floor [N]. FIXED, and fixed at zero: there is no
    #: propeller in this stage — thrust is a force along body x — and an
    #: aeroplane with a stopped engine makes none, not a negative one. It
    #: was once a field ("least available") that could be typed negative to
    #: get reverse thrust; the field is gone because the ask was for one
    #: number and a floor of zero, and a second field that is answered once
    #: in a hundred flights is a question in the way of the one that is
    #: answered every time.
    THRUST_MIN_N = 0.0

    def _thrust_ceiling(stored) -> float:
        """The ceiling implied by ``stored`` — the WHOLE default rule, once.

        Split out of :func:`_thrust_band` so the panel can ask what the field
        would be if it were blank without restating ``2.5 x the trim
        thrust``. A restatement is how the box and the stop drift apart.
        """
        t = (F.get("trim") or {}).get("thrust_n") or 100.0
        return (max(2.5 * abs(float(t)), 10.0) if stored in (None, "")
                else float(stored))

    def _thrust_band() -> tuple[float, float]:
        """``(0, max)`` — the stops W and S work between.

        The CEILING is answerable and the floor is not. A maximum thrust is
        a property of a propulsion system this stage does not model, so
        deriving it from the trim thrust is an assumption and has to be one
        the pilot can overrule — a calibration is a default, not a ban, and
        a throttle whose limit is invisible gives the keyboard no feedback
        at all: holding W past the stop feels identical to holding W with
        nothing connected. So the default stays, it is drawn on the bar
        beside the value (:func:`gui.v4.gauge.thrust_state`), and it is a
        field on the panel beside the picture.
        """
        lo = THRUST_MIN_N
        hi = _thrust_ceiling(F.get("thrust_max_n"))
        if hi <= lo:                    # a band with no width has no throttle
            hi = lo + max(abs(lo), 10.0)
        return lo, hi

    def _thrust_text() -> str:
        """The thrust the aeroplane is making, AND the band it runs in.

        Both in the one live string, because both move and neither may go
        stale: the value moves thirty times a second under W and S, and the
        ceiling moves whenever it is typed — and the panel deliberately does
        not rebuild when it is (see :func:`_set_thrust_max`). A caption that
        still read "between 0 and 54 N" after 180 was typed would be a
        number that is wrong on the screen while being right in the model,
        which is the failure mode this whole stage is built against.
        """
        lo, hi = _thrust_band()
        # ...and WHOSE ceiling it is. The panel deliberately does not rebuild
        # when the field is typed (see _set_thrust_max), so a caption printed
        # at build time could not carry this — it rides the frame with the
        # value, where it cannot go stale.
        whose = "set" if fl.is_pinned(F.get("thrust_max_n")) else "auto"
        return (f"{float(F['live'].get('thrust_n') or 0.0):,.0f} N"
                f"   ({lo:,.0f} – {hi:,.0f} N {whose})")

    def _set_thrust_max(e):
        """The ceiling moved. RE-CLAMP, DO NOT RE-RENDER.

        Rebuilding the fly view here would tear down the scene, re-fetch the
        aircraft mesh and stop the simulation — which is what ``edit`` does,
        and what every other field on this panel deliberately avoids. Nothing
        about the AEROPLANE changes when a stop moves; only the throttle's
        band does, and that rides the frame.

        The current thrust is pulled back inside the new band, because a
        throttle sitting above its own ceiling is a stop that is not one.
        """
        v = _number(e, blank=None)
        if v is _KEEP:
            return
        F["thrust_max_n"] = None if v is None else float(v)
        lo, hi = _thrust_band()
        cur = F["live"].get("thrust_n")
        if cur is not None and not (lo <= float(cur) <= hi):
            _set_live("thrust_n", min(max(float(cur), lo), hi))
        else:
            # the glass draws the bar against the band, so the scale moved
            # even when the value did not
            _readout()
            _pose()

    def _speed_band() -> tuple[float, float]:
        v = float(_mission_point()["V"]) or 45.0
        return 0.35 * v, 2.0 * v

    def _altitude_band() -> tuple[float, float]:
        return 0.0, max(3.0 * _altitude(), 1000.0)

    def _mass_band() -> tuple[float, float]:
        m = (F.get("trim") or {}).get("mass_kg") or 100.0
        return 0.4 * float(m), 2.0 * float(m)

    def _cg_band() -> tuple[float, float]:
        """How far the CG lever travels, and it must REACH the answer.

        A fixed +-0.35 mac about the design CG does not: on any design whose
        static margin is bigger than that, the aft stop is still statically
        stable, so the red "CG IS AFT OF THE NEUTRAL POINT" banner and its
        HUD warning are unreachable — dead code on the stage whose whole
        lesson they are. Measured on the shipped tail design: band
        -0.307..+0.407 m against a neutral point at 0.443 m, so the aft stop
        left +0.037 mac of margin and the readout never went red.

        So the aft stop is derived from the MEASUREMENT it exists to
        demonstrate, and only ever widened by it.
        """
        t = F.get("trim") or {}
        cg, mac = float(t.get("x_cg_m") or 0.0), float(t.get("mac") or 1.0)
        lo, hi = cg - 0.35 * mac, cg + 0.35 * mac
        x_np = _x_np()
        if x_np is not None and np.isfinite(x_np):
            hi = max(hi, float(x_np) + 0.10 * mac)
        return lo, hi

    def _set_live(key, value):
        """A live lever moved. Re-apply, DO NOT re-render.

        Rebuilding the view here would destroy the scene group mid-flight and
        take the typed field's focus with it — a typed number must not rebuild
        its own field.
        """
        F["live"][key] = None if value is None else float(value)
        ac = F.get("ac")
        if ac is not None:
            F["ac"] = _live(ac)
        if key == "x_cg_m":
            _place_cg()            # the marker stays; the aeroplane moves
            _stability()           # ...and the modes move with it
        elif key == "mass_kg":
            _stability()           # inertia is in the Jacobian
        _readout()
        # ...and report it, so a lever moved while PAUSED still reaches the
        # glass rather than waiting for a frame that is not coming
        _pose()

    def _set_altitude(key, value):
        """The height datum, WITHOUT a re-trim.

        Everything that makes the speed lever a re-trim — a new equilibrium,
        a new elevator angle, a new thrust — is absent here: the density is
        the mission's and constant, so ``trim_level`` uses the altitude for
        the state's position and for nothing else. Sending it through
        ``_set_condition`` therefore re-armed, nulled the pilot's throttle,
        cleared the manoeuvre and replaced the trace with one sample, to
        arrive at exactly the aeroplane that was already flying.
        """
        del key
        import numpy as _np

        st = F.get("state")
        prev = _altitude()
        F["altitude_m"] = None if value in (None, "") else float(value)
        if st is not None:
            # the aeroplane keeps its HEIGHT ABOVE THE DATUM, so the picture
            # does not jump when the ground moves under it.
            #
            # A NEW STATE, not a mutated one: ``_advance`` appends the very
            # object ``F["state"]`` holds to the history, so writing into
            # ``st.pos`` would edit a sample the trace has already drawn.
            pos = _np.array(st.pos, dtype=float)
            pos[2] -= (_altitude() - prev)
            F["state"] = st.__class__(pos=pos, quat=st.quat, vel=st.vel,
                                      rates=st.rates)
        _place_cg()
        _readout()
        _pose()

    def _set_condition(key, value):
        """A lever that means a NEW TRIM — the speed or the altitude.

        Weight, thrust and the CG are applied to the aeroplane that is
        already flying. These two are not: an aeroplane shoved to 25 m/s at
        the elevator setting that trimmed it at 14.6 will simply fly back,
        and a slider that does that teaches nothing. So the condition is
        answered and the aircraft is TRIMMED there, which restarts the
        clock and the trace — said on the panel, because a lever that
        silently discards a trace is worse than one that does it loudly.

        The trim thrust goes with it. Keeping the pilot's own throttle over
        a re-trim would leave the aeroplane out of equilibrium at the speed
        it was just trimmed for, which is the same lie one level down.
        """
        # ...EXCEPT the altitude, which is not a condition at all. It is
        # delegated rather than merely routed around, so every caller —
        # this stage's own lever, a test, anything registered — gets the
        # same answer to "what does moving the altitude do".
        if key == "altitude_m":
            return _set_altitude(key, value)
        prev, prev_thrust = F.get(key), F["live"].get("thrust_n")
        F[key] = None if value in (None, "") else float(value)
        F["live"]["thrust_n"] = None
        if not arm():
            # a speed below the stall (or an altitude the trim cannot
            # reach) is a real answer to the question and must not leave the
            # stage with no aeroplane in it. Put the last one back, keep
            # the reason on screen.
            why = F.get("error")
            F[key], F["live"]["thrust_n"] = prev, prev_thrust
            arm()
            F["error"] = why
        F["manoeuvre"] = None
        _place_cg()
        _stability()
        _readout()
        _pose()

    # ------------------------------------------------------- what it will do
    def _stability():
        """Name the modes of the aeroplane as it is CURRENTLY levered.

        Not once a frame: this is a 13x13 numerical Jacobian and an
        eigensolve, measured at 1.58 ms, which is a fifth of a frame at
        120 Hz and would put back the cost the frame path was rebuilt to
        remove. It is recomputed when something that changes the aeroplane
        changes — a lever, a re-trim, a reset — which is exactly when the
        answer can differ.
        """
        ac, st = F.get("ac"), F.get("state")
        if ac is None or st is None:
            F["modes"] = None
        else:
            from aerobo import sixdof as sd
            try:
                F["modes"] = md.classify(
                    sd.linearise(ac, st), float(st.V),
                    Cn_beta=getattr(getattr(ac, "deck", None),
                                    "Cn_beta", None))
            except Exception:                          # noqa: BLE001
                F["modes"] = None
        _render_stability()

    def _manoeuvre(key):
        """Fly one of the canned excitations (:mod:`gui.v4.manoeuvre`).

        RE-TRIMMED FIRST, and with the trim thrust rather than the pilot's.
        A mode is a property of the linearisation about EQUILIBRIUM, so a
        phugoid flown from an aeroplane that is already accelerating is a
        phugoid plus whatever the pilot left behind, and the trace cannot
        be compared with the eigenvalue that predicted it.
        """
        m = mv.BY_KEY.get(key)
        if m is None:
            return
        F["live"]["thrust_n"] = None
        if not arm():
            _render_stability()
            return
        st = F["state"]
        vel, quat = mv.perturbation(st.vel, st.quat, m)
        F["state"] = st.__class__(pos=st.pos, quat=quat, vel=vel,
                                  rates=st.rates)
        F["history"], F["history_t"] = [F["state"]], [0.0]
        F["manoeuvre"] = key
        _place_cg()
        _stability()
        _readout()
        _pose()
        _run(True)

    # ------------------------------------------------------------ the loop
    def _elapsed() -> float:
        """Wall-clock seconds since the last frame, clamped.

        THE SIMULATION USED TO COUNT TICKS. ``F["dt"]`` was 1/60 while the
        timer fired at 1/30, so it bought a sixtieth of a second of flight
        per thirtieth of a second of real time: the aeroplane flew at half
        speed and the rate selector marked "1x" was 0.5x. Counting ticks is
        wrong however carefully the two numbers are matched, because the
        timer does not fire at the rate it is asked for — asyncio overshoots
        every sleep by about a millisecond, and a busy frame overshoots by
        more. So the clock is the clock.
        """
        now = time.perf_counter()
        last = view.get("wall")
        view["wall"] = now
        if last is None:                 # first frame after arming or resume
            return 0.0
        return min(max(now - last, 0.0), MAX_CATCHUP_S)

    def _advance(elapsed: float | None = None):
        """Fly ``elapsed`` seconds of wall clock, in fixed integrator steps.

        ``elapsed=None`` reads the clock, which is what the frame loop does.
        A caller that passes a number is asking for a known amount of flight
        — that is how the tests drive it, and it is the honest way to test
        the integrator and the stick without also testing ``perf_counter``.
        """
        from dataclasses import replace

        from aerobo import sixdof as sd

        ac, st = F.get("ac"), F.get("state")
        if ac is None or st is None:
            return
        # the catch-up clamp lives HERE, not in _elapsed, so it holds however
        # the frame is driven — a caller passing a number gets the same
        # ceiling the clock path gets
        wall = _elapsed() if elapsed is None else float(elapsed)
        wall = min(max(wall, 0.0), MAX_CATCHUP_S)
        dt = wall * float(F.get("speed") or 1.0)

        # the pilot first: springs, then the throttle, then the surfaces.
        # ONCE per frame with the frame's own elapsed, not once per substep:
        # the stick is an input, not a state being integrated.
        #
        # ...AND IN WALL TIME, NOT SIM TIME. ``dt`` is ``wall * speed``, and
        # driving the spring with it made the stick four times quicker at
        # the 4x multiplier: full deflection in 0.09 s of wall clock instead
        # of the 0.35 s ``stick.TRAVEL_S`` names, so the faster the
        # simulation ran the less flyable it was. A pilot's hand does not
        # speed up because the clock did; the stick is an INPUT and its rate
        # belongs to the person, while ``dt`` belongs to the aeroplane.
        stk.step(F["stick"], F["hold"], wall)
        if F["hold"].get("thrust"):
            lo, hi = _thrust_band()
            F["live"]["thrust_n"] = stk.advance_throttle(
                float(F["live"].get("thrust_n") or 0.0),
                stk.demand(F["hold"]["thrust"]), wall, lo, hi)
            ac = replace(ac, prop=replace(
                ac.prop, thrust_n=float(F["live"]["thrust_n"])))

        # ---- a canned excitation OVERRIDES the spring while it runs.
        # After stk.step, not before: the spring would otherwise pull the
        # commanded deflection back toward centre within the same frame. The
        # window closes by itself (inputs_at returns nothing past the end)
        # and the spring takes the axis back, which is what a released stick
        # does and what the mode wants to be measured from.
        if F.get("manoeuvre"):
            for axis, deg in mv.inputs_at(
                    mv.BY_KEY[F["manoeuvre"]], float(F.get("t") or 0.0)
                    ).items():
                F["stick"][axis] = float(deg)

        stick = F["stick"]
        ac = replace(ac, controls={
            **ac.controls,
            **{k: float(v) * RAD for k, v in stick.items()
               if k in ac.deck.columns and k != "elevator"},
            # the elevator stick is a PERTURBATION from trim, not an absolute
            # deflection: centre stick has to mean the trimmed aeroplane
            "elevator": (F["trim"]["elevator_deg"] * RAD
                         + float(stick.get("elevator", 0.0)) * RAD),
        })
        F["ac"] = ac

        # ---- fixed-step integration, however long the frame was
        acc = float(view.get("acc") or 0.0) + dt
        n = 0
        # the substep cap is the same ceiling expressed the other way round,
        # so a speed multiplier cannot smuggle a long frame past the clamp
        cap = max(1, int(round(MAX_CATCHUP_S * float(F.get("speed") or 1.0)
                               / SUBSTEP_S)))
        while acc >= SUBSTEP_S and n < cap:
            st = sd.step(ac, st, SUBSTEP_S)
            F["t"] = float(F.get("t") or 0.0) + SUBSTEP_S
            acc -= SUBSTEP_S
            n += 1
            if st.altitude_m <= 0.0:
                _crash(st)
                break
        view["acc"] = acc
        if n == 0:
            return                       # not enough time passed to fly yet
        F["state"] = st

        # the trace is sampled once a FRAME, not once a substep — 240 Hz of
        # history is four times the data for none of the picture — and it
        # carries its own time base, because the frames are not evenly spaced
        hist = F.setdefault("history", [])
        hist.append(st)
        times = F.setdefault("history_t", [])
        times.append(float(F.get("t") or 0.0))
        if len(hist) > 6000:                    # ~100 s at 60 Hz
            del hist[:len(hist) - 6000]
            del times[:len(times) - 6000]

    def _frame():
        """One tick: integrate, then move the scene group. Nothing is
        rebuilt — a repaint is not a rebuild."""
        if not F.get("running"):
            return
        try:
            _advance()
            _readout()
            _pose()
            # ...and the traces, if that is the tab on screen. At a fraction
            # of the frame rate: six plotly payloads is a real message, and
            # the eye reads a trace at two updates a second as "live".
            view["trace_n"] = int(view.get("trace_n") or 0) + 1
            if view["trace_n"] % TRACE_EVERY == 0:
                _traces_tick()
        except Exception:                              # noqa: BLE001
            # THE WHOLE TEARDOWN, not just the flag. Setting ``running``
            # False on its own left the 30 Hz timer alive and — the one that
            # bites — left every held key held, so the next Fly began with
            # the stick already at a stop nobody was touching.
            _teardown()
            F["error"] = traceback.format_exc().strip().splitlines()[-1]
            print("[v4] flight frame failed:\n" + traceback.format_exc())

    def _payload() -> dict | None:
        """Everything that changed this frame, as one small dict.

        Pure enough to assert on: given a state it returns the numbers, and a
        test can check the frame chain and the floating origin without a
        browser anywhere near it.
        """
        st = F.get("state")
        if st is None:
            return None
        # scene-from-body = M . C^T  (see the frame note at the top of this
        # module). Handing the scene C^T alone draws every roll mirrored.
        # As a quaternion, because the browser SLERPs between reports.
        q = scene_quat(st.quat)
        # the aircraft is held at the origin laterally and only its ALTITUDE
        # is drawn: a design review wants to see the attitude, not watch the
        # model leave the scene at 45 m/s. The ground slides underneath.
        w = wld.world_offset(st.pos, WORLD_SNAP_M, _altitude())
        # rounded because the browser cannot see the difference and the
        # digits are paid for thirty times a second
        out = {"q": [round(v, 6) for v in q],
               "z": round(float(st.altitude_m - _altitude()), 4),
               "w": [round(float(v), 3) for v in w],
               "game": bool(_game())}
        hud = view.get("hud_state")
        if hud is not None:
            out["hud"] = hud
        txt = view.get("text_state")
        if txt is not None:
            out["txt"] = txt
        g = view.get("gauge_state")
        if g is not None:
            out["g"] = g
        return out

    def _pose():
        """Report the frame. ONE websocket message, and no rebuild.

        It used to be four ``run_method`` calls and a 14.8 kB repaint of the
        glass — 155 messages/s and 515 kB/s for one aeroplane, measured. The
        browser now owns the camera and the smoothing (:mod:`gui.v4.live`),
        so what leaves this process is about 300 bytes.
        """
        p = _payload()
        if p is None or not view.get("installed"):
            return
        _emit(lv.push_call(p))

    def _place_cg():
        """Slide the airframe so the CG sits on the pivot.

        The group's origin is what the aeroplane rotates about, and an
        aeroplane rotates about its CG — so the CG marker stays put at the
        origin and the AIRFRAME moves when the CG lever does. That is what
        makes the lever visible: drag it aft and the whole aircraft slides
        forward under the marker, toward the neutral point drawn with it.
        """
        body = view.get("body")
        if body is None:
            return
        ac = F.get("ac")
        cg = (float(F["live"]["x_cg_m"]) if F["live"].get("x_cg_m") is not None
              else float(ac.deck.x_cg) if ac is not None else 0.0)
        z = float(getattr(ac.deck, "z_cg", 0.0)) if ac is not None else 0.0
        body.move(*body_offset(cg, z))

    # --------------------------------------------------------- the keyboard
    def _on_key(e):
        """Arrow keys, WASD. Sets a DEMAND, never a deflection.

        Ignored while the simulation is not running, so the arrow keys still
        scroll the page and A/D still type into a field when nothing is in
        the air. Quasar inputs are excluded at the source (``ignore=``), so
        typing a mass of 25 kg does not roll the aeroplane left twice.
        """
        name = (getattr(e.key, "name", "") or str(e.key)).lower()
        hit = stk.key_axis(name)
        if hit is None:
            return
        axis, direction = hit
        # WHAT IS HELD IS A SET OF KEYS, NOT A LAST-WRITER-WINS SIGN.
        #
        # ``hold[axis] = 0`` on any keyup meant that pressing A, then D
        # without releasing A, then releasing A, centred an axis whose other
        # key was still physically down: the aeroplane stopped rolling with
        # the pilot's finger on the stick. Both directions of an axis are
        # one control, so what the axis is doing has to be derived from
        # every key currently down on it.
        down = F.setdefault("keys_down", set())
        if e.action.keydown:
            down.add(name)
        else:
            down.discard(name)
        # ...and a keyUP is honoured even when the simulation has stopped.
        # The guard used to cover both edges, so a key still down when a
        # frame threw, or when Pause was pressed, was held for ever.
        if not F.get("running") and e.action.keydown:
            return
        _apply_hold(axis)

    def _release_keys():
        """Let go of every key, because the keyboard stopped being ours.

        The browser half (:func:`gui.v4.live.INSTALL_JS`) reports this when
        the page loses focus, when the tab is hidden, and when focus lands
        on a widget nicegui's keyboard refuses to read — which is the case
        that actually bites: clicking into the max-thrust box while holding
        A swallows the keyup, and the aeroplane keeps rolling with nothing
        held.
        """
        down = F.get("keys_down")
        if not down:
            return
        down.clear()
        for axis in stk.LIMITS:
            _apply_hold(axis)

    def _hold_of(axis: str, down) -> int:
        """-1, 0 or +1 for one axis, from the keys actually down on it.

        With both keys down the LAST one still counts for nothing: they are
        opposite demands on one control and they cancel, which is what a
        centred stick is.
        """
        return int(sum(d for k in down
                       for a, d in [stk.key_axis(k) or ("", 0)] if a == axis))

    def _apply_hold(axis: str):
        """One axis' demand, from EVERY device holding it.

        TWO HANDS ON ONE CONTROL, AND NEITHER IS A LAST WRITER. This is the
        keys-down lesson one device further out: the keyboard's ±1 and the
        pad's fraction are ADDED and clipped, so a pad let go to centre
        cannot centre an axis whose key is still down, and a key cannot
        overrule a stick that is deflected. The clip is in
        ``stick.demand`` — a sum of two full demands is still one full
        demand, not two.
        """
        keys = _hold_of(axis, F.get("keys_down") or set())
        held = float((F.get("pad") or {}).get(axis, 0.0))
        F["hold"][axis] = stk.demand(keys + held)

    # ------------------------------------------------------------- the pad
    def _on_pad(e):
        """One report from the gamepad: a demand, and any button just
        pressed.

        The browser polls (a Gamepad has no input event) and reports only
        when the rounded demand CHANGES, so a stick held anywhere — centre
        included — costs nothing. See :mod:`gui.v4.pad`.
        """
        args = getattr(e, "args", None)
        # nicegui hands a listener the EMITTED VARARGS as a list, and the
        # payload is the first of them. Both shapes are accepted because a
        # test that calls this directly has no reason to build the wrapper.
        if isinstance(args, (list, tuple)):
            args = args[0] if args else {}
        d = args if isinstance(args, dict) else {}

        was_on = bool(F.get("pad_on"))
        F["pad_on"] = bool(d.get("on"))
        if was_on != F["pad_on"]:
            _pad_status()

        # THE BUTTONS ARE READ FIRST, AND THE ORDER IS THE POINT. A report
        # is only sent when the demand CHANGES, so if the sticks were read
        # first they would be zeroed (nothing is flying yet), the cross
        # would then start the flight — and a pilot holding the stick
        # steady through the press would send nothing more and fly with a
        # centred stick until they moved it.
        for act in (d.get("a") or []):
            _pad_action(str(act))

        h = d.get("h") or {}
        held = F.setdefault("pad", {})
        for axis in ("elevator", "aileron", "rudder", "thrust"):
            # THE STICK IS IGNORED WHILE STOPPED, THE BUTTONS ARE NOT — the
            # same rule the keyboard flies under. Nothing may deflect a
            # surface that is not in the air, but the pilot has to be able
            # to start the flight with the pad they are holding.
            v = float(h.get(axis, 0.0) or 0.0) if F.get("running") else 0.0
            held[axis] = stk.demand(v)
            _apply_hold(axis)

    def _pad_action(name: str):
        """A button press, as the thing the panel beside it would do.

        Three, and each one routed to the SAME function the button on the
        panel calls: a pad that re-implemented "reset" would be a second
        answer to a question that already has one.
        """
        if name == "fly":
            _run(not F.get("running"))
        elif name == "reset":
            _reset()
        elif name == "view":
            _set_mode("engineering" if _game() else "game")

    def _wire_pad():
        """Subscribe this PAGE to the gamepad, once.

        ``ui.on`` registers on the client's LAYOUT, which outlives the Fly
        view: registering it where the view is built would add a handler
        every time the pilot leaves stage 6 and comes back, and thirty
        reports a second would then be answered five times each.
        """
        # ...and what is registered is a TRAMPOLINE, not this build's own
        # handler. A view can be built more than once for one page, and a
        # subscription that closed over the first build would go on writing
        # into a view that no longer exists — the stale-handle defect the
        # browser half has already been bitten by. The indirection means
        # the LATEST build always owns the pad.
        F["pad_handler"] = _on_pad
        F["release_handler"] = _release_keys
        client = ui.context.client
        wired = _PAGE_WIRED
        cid = getattr(client, "id", None)
        if cid in wired:
            return
        wired.add(cid)

        def _dispatch(e):
            fn = F.get("pad_handler")
            if fn is not None:
                fn(e)

        def _dispatch_release(e):
            del e
            fn = F.get("release_handler")
            if fn is not None:
                fn()

        ui.on("aerobo_pad", _dispatch)
        ui.on("aerobo_release", _dispatch_release)

    def _pad_status():
        """Say whether a pad is actually being seen.

        ONE ELEMENT UPDATE PER PLUG EVENT, and none per frame — the rule
        this stage is built on (:mod:`gui.v4.live`). A pilot with a pad that
        the page cannot see needs to be told that, because the failure is
        silent and looks exactly like a broken controller.
        """
        on = bool(F.get("pad_on"))
        # RECORDED as well as shown. The label belongs to a view that dies
        # with the render; the sentence is the stage's answer to "is a pad
        # being seen", and a test cannot read a Quasar widget.
        F["pad_status"] = ("gamepad connected" if on else
                           "no gamepad — press a button on it")
        lbl = view.get("padlbl")
        if lbl is None:
            return
        lbl.text = F["pad_status"]
        lbl.style(f"color:{theme.GOOD if on else theme.INK_MUTED}")

    # ---------------------------------------------------------- view: setup
    def edit(key, value):
        F[key] = value
        arm()
        ctx.render_when_shown("flight")

    #: the browser waits this long after the last keystroke before sending
    #: the number. ``ui.number`` fires its handler on every keystroke and
    #: ``edit`` re-trims and repaints, so an undebounced field committed the
    #: first digit and rebuilt itself under the cursor.
    FIELD_DEBOUNCE_MS = 600

    def num(*a, **kw):
        """:func:`gui.v3.widgets.number_field`, committing when you stop
        typing."""
        el = widgets.number_field(*a, **kw)
        el.props(f"debounce={FIELD_DEBOUNCE_MS}")
        return el

    #: see :func:`_number` — "nothing usable arrived", kept apart from a
    #: blank that MEANS something (CD0, e and the two thrust stops all read
    #: "blank = the design's own")
    _KEEP = object()

    def _number(e, blank=_KEEP):
        """THE NUMBER, off a nicegui CHANGE EVENT.

        ``ui.number(on_change=...)`` hands its handler a
        ``ValueChangeEventArguments`` and not a value. Treating it as one
        means ``float(event)``, which raises inside the handler where nicegui
        logs it and the user sees the field not take — no number typed on
        this stage ever committed. Stored raw it is worse: the next render
        hands the event to ``ui.number`` as a value and the view dies in
        nicegui's own formatter.
        """
        v = getattr(e, "value", e)
        if v is None or v == "":
            return blank
        try:
            return float(v)
        except (TypeError, ValueError):
            return _KEEP

    def edit_num(key, e, *, blank=_KEEP):
        v = _number(e, blank)
        if v is not _KEEP:
            edit(key, v)

    def _render_setup():
        box = ctx.views[("flight", "setup")]
        box.clear()
        with box:
            fm = _fm()
            if fm is None:
                widgets.hint("Stage 5 has not built a derivative deck yet — "
                             "there is nothing to fly.", "warn")
                return
            ac = F.get("ac") or fm.aircraft
            widgets.hint_help(
                "The answers that mean a RE-TRIM. Asked here, not in flight.",
                "Weight, thrust and the CG are deliberately not on this "
                "tab: they are levers beside the picture on the Fly tab, "
                "because the only way to learn what one of them does is to "
                "move it while the aeroplane is in the air.\n\n"
                "What is here instead is everything that changes the "
                "equilibrium the flight STARTS from.",
                title="What belongs on this tab")
            widgets.kv("flying now",
                       f"{_V():.1f} m/s at {_altitude():.0f} m")
            with widgets.group_box("Aerodynamic model"):
                with ui.row().classes("w-full items-center gap-1 no-wrap"):
                    ui.switch("model the stall",
                              value=F.get("stall_on", True),
                              on_change=lambda e: edit("stall_on",
                                                       bool(e.value)))
                widgets.hint_help(
                    "Off = linear at every angle.",
                    "The stall is a p-norm soft clip on CL with the exact "
                    "derivative of that clip applied to the rate and "
                    "control columns too, so a stalled wing loses roll "
                    "damping and aileron power together rather than lift "
                    "alone.\n\n"
                    "Switching it off is what the MODE analysis wants — "
                    "eigenvalues of a linear system — and a lie anywhere "
                    "near the buffet.",
                    title="What the stall model does")
            # ONE QUESTION, ONE PLACE — and one LINE. The thrust line is a
            # property of the airframe, so it is answered in stage 5 and
            # only reported here. This was a whole group box carrying two
            # paragraphs that restated stage 5's Propulsion tab almost word
            # for word, with no control of any kind on it; a box a reader
            # has to scroll past to reach the three fields that ARE here.
            _zt = float(getattr(ac.prop, "z_offset_m", 0.0) or 0.0)
            with ui.row().classes("w-full items-center gap-1 no-wrap"):
                widgets.kv("thrust line",
                           ("through the CG — no pitching moment" if not _zt
                            else f"{_zt:.3g} m below the CG — throttle "
                                 f"pitches the nose UP"))
                widgets.help_dot(
                    "Answered in stage 5 (Controls > Propulsion), where the "
                    "deck that carries it is built. There is one number to "
                    "answer and it is the vertical one: a thrust line ahead "
                    "of or behind the CG makes no moment at all.\n\n"
                    "Thrust itself is a scalar along the body x-axis. No "
                    "propeller, no slipstream over the wing, no torque "
                    "reaction — modelling those would mean a second "
                    "aerodynamic model beside the lattice. How much is "
                    "AVAILABLE is the max-thrust stop beside the picture.",
                    title="The propulsion model, in full")
            with widgets.group_box("Inertia"):
                widgets.kv("Ixx / Iyy / Izz",
                           f"{ac.inertia.Ixx:.0f} / {ac.inertia.Iyy:.0f} / "
                           f"{ac.inertia.Izz:.0f} kg m2")
                widgets.hint_help(
                    f"Estimate, not a measurement. Basis: "
                    f"{ac.inertia.basis}.",
                    "This package has never modelled inertia — no report "
                    "carries one — so the three numbers above are "
                    "constructed, and the basis names how.\n\n"
                    "Moving the mass lever beside the picture scales all "
                    "three together, which is what makes the roll response "
                    "change when the aeroplane is made heavier.",
                    title="Where the inertia comes from")
            with widgets.group_box("Drag polar the simulation flies"):
                # both open on what the AIRCRAFT is flying, so a blank
                # never reads as "no answer" on a design that has one
                num("CD0", fl.shown_value(
                        F["CD0"], getattr(F.get("ac"), "CD0", None), 5),
                    lambda e: edit_num("CD0", e, blank=None),
                    note="zero-lift drag",
                    tip=f"blank = {fm.aircraft.CD0:.4g} from the report")
                num("span efficiency e", fl.shown_value(
                        F["oswald_e"], getattr(F.get("ac"), "oswald_e", None),
                        3),
                    lambda e: edit_num("oswald_e", e, blank=None),
                    note="Oswald e, 1.0 = elliptic",
                    tip=f"blank = {fm.aircraft.oswald_e:.2f}")
                widgets.hint_help(
                    "Profile drag from the report; induced from CL²/(π AR e).",
                    "The lattice gives induced drag in the Trefftz plane "
                    "and nothing else, so the profile part cannot come from "
                    "it and is taken from the design's own stored "
                    "breakdown.\n\n"
                    "Both fields are blank-means-the-model's: typing one "
                    "flies YOUR number, clearing it gives the design's back.",
                    title="How the drag polar is assembled")
            if F.get("trim"):
                t = F["trim"]
                with widgets.group_box("Trimmed"):
                    widgets.kv("alpha", f"{t['alpha_deg']:+.2f} deg")
                    widgets.kv("elevator", f"{t['elevator_deg']:+.2f} deg")
                    widgets.kv("thrust", f"{t['thrust_n']:.0f} N")
            if F.get("error"):
                widgets.hint(F["error"], "warn")

    # ------------------------------------------------------------ view: fly
    def _render_fly():
        box = ctx.views[("flight", "fly")]
        # .work-pad is 10px/12px of padding on every stage view (theme.py),
        # which is right for a form and wrong for a viewport
        box.classes(remove="work-pad")
        # a rebuilt view has a new scene and a new keyboard listener; the old
        # ones are gone from the page, so a frame loop still pointing at them
        # would move nothing and cost everything. Torn down DIRECTLY rather
        # than through _run: this runs inside a render, and _run repaints the
        # chrome, which would rebuild views from inside one being built.
        _teardown()
        view["group"] = view["scene"] = view["readout"] = None
        view["panel"] = view["stab"] = view["thrust"] = None
        view["installed"] = False
        view["hud_state"] = view["gauge_state"] = None
        # ...AND THE FLAG THAT REMEMBERS WHAT THE GLASS IS SHOWING. The
        # ``ui.html`` below is created fresh with no display property at
        # all, so a cached "already off" skipped the write and the dead
        # skeleton — ladder, tapes, bank arc — came back over the picture
        # with the switch beside it still reading off.
        view["hud_on"] = None
        view["fields"] = {}
        box.clear()
        # the view fills its pane and does NOT scroll (see _SIDE)
        box.style(_FLY_COL)
        with box:
            if _fm() is None:
                widgets.hint("Nothing to fly yet — build the controls in "
                             "stage 5.", "warn")
                return
            # RE-TRIM WHENEVER THE DECK IS A DIFFERENT ONE. "ac is None" was
            # the only trigger, and stage 5 does not leave it None — it
            # builds a new flight model and refreshes only the four V3
            # stages and itself, so an aeroplane armed before the edit was
            # still here and still flown. ``arm`` is a 2x2 Newton on an
            # affine deck; paying it once per entry to this view is nothing
            # beside flying the previous aircraft.
            if F.get("ac") is None or F.get("armed_fm") is not _fm():
                arm()
            if F.get("error"):
                widgets.hint(F["error"], "warn")
            # ...AND THE COCKPIT STAYS, unless there is genuinely no
            # aeroplane. A refusal used to return here, which took Fly,
            # Pause and Re-trim with it — and `F["error"]` is STICKY: it is
            # cleared only by a successful `arm`, and `_set_condition`
            # deliberately keeps the reason after putting a rejected speed
            # back. So one speed below the stall left a perfectly good
            # trimmed aeroplane in `F["ac"]` behind a warning card with no
            # button on it, for the rest of the session. The reason belongs
            # on screen; the controls that answer it belong there too.
            if F.get("ac") is None:
                return

            with ui.row().classes("items-center gap-2"):
                ui.button("Fly", icon="play_arrow",
                          on_click=lambda: _run(True)).props("dense")
                ui.button("Pause", icon="pause",
                          on_click=lambda: _run(False)).props("dense flat")
                ui.button("Re-trim & reset", icon="restart_alt",
                          on_click=_reset).props("dense flat")
                widgets.select_field(
                    "rate", {0.25: "0.25x", 0.5: "0.5x", 1.0: "1x",
                             2.0: "2x"},
                    F.get("speed", 1.0),
                    lambda e: F.__setitem__("speed", float(e.value)),
                    help="Simulated seconds per wall-clock second. The "
                         "physics step does not change with it — only how "
                         "many of them are taken per frame — so a mode "
                         "measured at 0.25x and at 2x is the same mode.\n\n"
                         "The stick is driven by the WALL clock either way, "
                         "so a key held for a quarter of a second gives the "
                         "same deflection at every rate.",
                    help_title="What the rate multiplier changes")
                widgets.select_field(
                    "view", {"game": "Fly it", "engineering": "Inspect it"},
                    F.get("mode", "game"),
                    lambda e: _set_mode(e.value), width="w-40",
                    help="Fly it: the camera sits behind the aeroplane and "
                         "the picture is what a pilot sees.\n\n"
                         "Inspect it: a fixed three-quarter view with the "
                         "CG and the neutral point drawn, so the static "
                         "margin is a distance on screen.",
                    help_title="The two cameras")
                with ui.row().classes("items-center gap-1 no-wrap"):
                    ui.switch("HUD", value=bool(F.get("hud", True)),
                              on_change=lambda e: (F.__setitem__(
                                  "hud", bool(e.value)), _readout(),
                                  _pose()))
                    widgets.help_dot(
                        "The overlay on the picture: attitude ladder, "
                        "speed, altitude, angle of attack and the three "
                        "body rates.\n\n"
                        "It is drawn in the BROWSER from the frame payload "
                        "the server sends, not rebuilt as nicegui elements "
                        "— which is why it costs nothing to leave on.",
                        title="What the HUD shows")

            # ---- the picture, and the three levers DOWN ITS SIDE
            # no-wrap: a ui.row wraps by default, which put the levers
            # BELOW a viewport that is most of the window tall — i.e. off the
            # bottom of the screen. Squeezing the picture on a narrow window
            # is the better failure.
            with ui.row().classes("w-full items-stretch gap-3 no-wrap") \
                    .style(_FLY_ROW):
                with ui.column().classes("gap-2").style(_SIDE % (232, 232)):
                    _roll_panel()
                # a position:relative wrapper so the HUD can be a later
                # absolutely-positioned SIBLING of the canvas. This is the
                # pattern ui.scene itself uses for its CSS2D/CSS3D layers,
                # so it needs no z-index anywhere.
                shell = ui.element("div").style(
                    "position:relative;flex:1 1 auto;min-width:0;"
                    "height:100%;overflow:hidden;border-radius:3px")
                with shell:
                    _build_scene()
                    # the glass is drawn ONCE and then only written into.
                    # See gui/v4/hud.py for what rebuilding it cost.
                    view["hud"] = ui.html(hud_svg.skeleton()).style(
                        "position:absolute;inset:0;pointer-events:none")
                with ui.column().classes("gap-2").style(_SIDE % (290, 290)):
                    _live_panel()

            # EVERYTHING UNDER THE PICTURE SCROLLS ON ITS OWN. If it shared
            # the view's scroller, reaching the stick panel would take the
            # picture — and the attitude panel beside it — off the top of
            # the pane, which is exactly how three live rates came to be
            # invisible.
            with ui.column().classes("w-full gap-3").style(_FLY_REST):
                # pre-wrap because the read-out is THREE lines and a label is
                # a div: without it the newlines collapse and the whole frame
                # prints as one run-on line.
                view["readout"] = ui.label().classes("mono") \
                    .style("white-space:pre-wrap")
                view["banner"] = ui.label().classes("mono")
                # ...and only NOW is the browser half installed: it carries
                # the element ids of the glass AND of these two labels, so
                # every one of them has to exist first.
                _install()
                _stick_panel()
                # ONE keyboard for this view. Built here rather than in _run
                # so it has a slot to live in; it refuses every key while the
                # simulation is stopped, so it costs nothing when nothing
                # flies. repeating=False: the stick is a DEMAND held in
                # Python, so the operating system's key-repeat adds nothing
                # but about 30 round trips a second per held key.
                view["keyboard"] = ui.keyboard(on_key=_on_key,
                                               repeating=False)
                # ...and the OTHER pilot. Not an element: the pad reports
                # through a page-level event, so this only has to happen
                # once per client and is guarded to.
                _wire_pad()
                widgets.hint_help(
                    "Held at the origin: attitude and altitude are drawn.",
                    "The aeroplane does not translate across the scene. A "
                    "design review wants to see HOW it flies, not watch it "
                    "leave the picture at 45 m/s.\n\n"
                    "Its position over the ground is still integrated and "
                    "still on the Traces tab; only the camera pretends "
                    "otherwise.",
                    title="Why the aeroplane does not move")
            _stability()
            _readout()

    def _reset():
        """Re-trim, and put the aeroplane back at the mission point.

        IT MUST NOT REBUILD THE VIEW. ``ctx.render`` here cleared the box and
        built a second ``ui.scene``, which means a fresh WebGL context and a
        re-fetch and re-parse of the aircraft mesh (1.1 MB on the tail
        problem) — and, because ``_render_fly`` tears the loop down before it
        repaints, it also stopped the simulation. From the pilot's seat the
        button froze the page for about a second and then nothing was flying,
        which is exactly what "reset and trim doesn't work" looks like.

        Nothing a re-trim changes needs a rebuild: the levers already hold
        the pilot's own numbers (``arm`` only fills the ones nobody has
        answered), so the sliders are already right, and the picture and the
        numbers both travel by the frame path. Only a FAILED trim needs the
        view back, because that is the one case with an error card to show.
        """
        crashed = bool(F.get("crashed"))
        F["manoeuvre"] = None
        F["live"]["thrust_n"] = None      # RE-trim means the trim's thrust
        if not arm():
            ctx.render("flight", "fly")
            return
        _place_cg()
        _stability()
        _readout()
        _pose()
        # the ground stopped the loop, the pilot did not: a reset after a
        # crash is a request to fly again, not to sit still
        if crashed and view.get("timer") is None:
            _run(True)

    def _roll_panel():
        """LEFT OF THE PICTURE: three attitudes, three rates, three sticks.

        The glass shows what the attitude IS and says nothing about which
        way it is going. Ten degrees of bank rolling back to level and ten
        degrees rolling away from it look identical on a bank pointer, and
        on a design with a divergent spiral that is the entire question — so
        every angle here is paired with its own body rate on the same row.

        Under them, the three stability answers in the pilot's words rather
        than the deck's. Those are nicegui elements and not part of the
        frame payload, because they only change when the AEROPLANE changes
        — a lever, a re-trim — and not once a frame.
        """
        view["panel"] = ui.html(gg.attitude_skeleton()).classes("w-full")
        view["stab"] = ui.column().classes("w-full gap-1")
        _render_stability()

    #: the three rows of the stability strip: (label, key in F["modes"] or
    #: None for a static sign, how to read it). Kept short on purpose —
    #: every eigenvalue is one tab away on Modes, and a panel that lists
    #: them all is a panel nobody reads while flying.
    def _render_stability():
        box = view.get("stab")
        if box is None:
            return
        box.clear()
        D = getattr(F.get("ac"), "deck", None)
        M = F.get("modes") or {}

        def line(name, verdict, detail, ok):
            # NOT widgets.kv: its key column is 150 px wide, which in a
            # 232 px panel leaves 80 for the value and wrapped every row
            # onto three lines. A narrow strip needs a narrow label.
            with ui.row().classes("w-full items-baseline gap-1 no-wrap"):
                ui.label(name).classes("field-label").style(
                    f"color:{theme.INK_MUTED};width:44px;flex:0 0 44px")
                ui.label(verdict).classes("readout").style(
                    f"color:{theme.GOOD if ok else theme.BAD};"
                    "font-size:11px")
                ui.label(detail).classes("readout").style(
                    f"color:{theme.INK_MUTED};font-size:11px")

        with box:
            if D is None:
                widgets.hint("not armed", "warn")
                return
            for name, val, ok, unit in (
                    ("pitch", float(D.Cm_alpha), D.Cm_alpha < 0, "Cm_a"),
                    ("roll", float(D.Cl_beta), D.Cl_beta < 0, "Cl_b"),
                    ("yaw", float(D.Cn_beta), D.Cn_beta > 0, "Cn_b")):
                line(name, "stable" if ok else "UNSTABLE",
                     f"{unit} {val:+.3f}", ok)
            # ...UNDER WHATEVER NAME IT CAME BACK. On a design that does
            # not weathercock the slow lateral root is not a spiral, and it
            # is still the number that says whether this aeroplane rolls
            # off: `hydrofoil + elevator` doubles it in 1.54 s. Losing the
            # row there would be the same silence, one step further on.
            sp, sp_name = md.find(M, "spiral")
            if sp is not None:
                # THREE ANSWERS, NOT TWO. A real part that the Jacobian
                # cannot resolve from zero is NEUTRAL, and calling it
                # "DIVERGES x2 115795 s" is a red word for thirty-two hours
                # (gui.v4.modes.NEUTRAL_1_PER_S).
                t2 = sp.double_or_half_s
                word = ("neutral" if sp.neutral
                        else ("DIVERGES" if sp.diverges else "converges"))
                line("spiral" if sp_name == "spiral" else "lat root", word,
                     ("" if t2 is None
                      else f"{'x2' if sp.diverges else '/2'} {t2:.0f} s"),
                     sp.stable)
            widgets.hint_help(
                "Three signs, then the spiral — which is not a sign.",
                "The first three rows each test one derivative against the "
                "sign stability asks of it. The spiral is a RACE between "
                "the dihedral effect and the fin (Cl_beta·Cn_r − "
                "Cn_beta·Cl_r), so an aeroplane can pass all three and "
                "still roll off with the stick free."
                + (" And on this design no fin SIZE turns it: growing the "
                   "fin raises the dihedral effect and the yaw stiffness "
                   "together, so the race does not change. What turns it is "
                   "wing dihedral, which stage 5 measures."
                   if sp is not None and sp.diverges else
                   " Stage 5 sizes the fin against it."),
                title="What these four rows are")

    def _mode_line(mode) -> str:
        """One mode, in the shortest form that still says what it does.

        An oscillation is a period and a damping ratio; a subsidence or a
        divergence is a time. The time is the TIME TO DOUBLE OR HALVE rather
        than the time constant, because that is what a handling requirement
        is written in and what a pilot can compare with their own reaction:
        a spiral that doubles in 60 s is a trim change, one that doubles in
        8 s is workload.
        """
        if mode is None:
            return "—"
        if mode.oscillatory:
            return (f"T {mode.period_s:5.1f} s   z {mode.damping:+.2f}"
                    + ("  GROWS" if mode.diverges else ""))
        t2 = mode.double_or_half_s
        if t2 is None:
            return "neutral"
        return (f"x2 in {t2:5.1f} s" if mode.diverges
                else f"half in {t2:5.1f} s")

    def _live_panel():
        """RIGHT OF THE PICTURE: everything the pilot moves, and nothing else.

        Five levers, and the two of them that mean a re-trim say so. The
        throttle is the odd one: its VALUE is on the gauge above the slider
        rather than on the slider, because W and S move it and a Quasar
        widget cannot follow a keyboard without one element update a frame.
        """
        L = F["live"]

        def row(key, label, lo, hi, step, unit, tip, *, setter=None,
                title=True, value=None):
            # A LEVER OPENS ON WHAT IS BEING FLOWN, not on its own floor.
            # The two condition levers are None until somebody answers them
            # — None meaning "the mission's own" — so falling back to `lo`
            # put the speed slider at the bottom of its band while the
            # aeroplane flew the mission point, i.e. the panel disagreed
            # with the aircraft in the picture beside it.
            cur = value if value is not None else (
                L.get(key) if setter is None else F.get(key))
            cur = float(lo if cur is None else cur)
            # ...and the band always CONTAINS it: a speed typed outside the
            # default band must not be silently clamped by the slider that
            # is meant to be showing it
            lo, hi = min(lo, cur), max(hi, cur)
            with widgets.group_box(label if title else ""):
                sl = ui.slider(min=lo, max=hi, step=step, value=cur) \
                    .props("dense label")
                n = ui.number(value=cur, step=step, format="%.4g") \
                    .props("dense outlined").classes("w-28")

                def push(v, src):
                    if v is None:
                        return
                    (setter or _set_live)(key, v)
                    # the two widgets are one control: whichever moved, the
                    # other follows WITHOUT a rebuild, so the number keeps
                    # its focus while the slider tracks it
                    (n if src is sl else sl).set_value(float(v))

                sl.on_value_change(lambda e: push(e.value, sl))
                n.on_value_change(lambda e: push(e.value, n))
                view["fields"][key] = (sl, n)
                widgets.hint(f"{unit} · {tip}")

        t = F.get("trim") or {}
        mp = _mission_point()

        # ---- thrust: ONE STOP THE PILOT OWNS, and the value it is holding.
        #
        # MAX is answerable and MIN is not. A ceiling is a property of a
        # propulsion system this stage does not model, so its default (2.5x
        # the trim thrust) is an assumption the pilot must be able to
        # overrule — a calibration is a default, not a ban. The floor is
        # zero because a stopped engine makes no thrust, and there is no
        # propeller here to reverse.
        #
        # CURRENT is beside it, and it is LIVE. It cannot be a Quasar widget:
        # W and S move it thirty times a second and an element update once a
        # frame costs a quarter of the frame (gui/v4/live.py). So it is a
        # plain label written straight into the DOM off the frame payload,
        # the same route as the read-out and the banner.
        with widgets.group_box("Thrust"):
            # OPENS ON THE CEILING IN FORCE, never empty. It used to show a
            # blank box on an aeroplane whose throttle stopped at 52 N —
            # "default was 52 but the box is empty" — because blank means
            # "2.5x the trim thrust" and nothing put that number anywhere a
            # user looks. Nothing is STORED until it is typed, so a blank
            # ceiling still follows a re-trim; which of the two it is rides
            # the live caption below (``auto`` / ``set``), because this panel
            # does not rebuild when the field is answered.
            num(
                "max thrust",
                fl.shown_value(F["thrust_max_n"], _thrust_band()[1], 1),
                _set_thrust_max,
                unit="N", width="w-24",
                tip="blank = 2.5x the trim thrust, and clearing the box goes "
                    "back to it. This is the stop W runs into; S runs down "
                    "to a floor of zero, which is fixed — no propeller, and "
                    "a stopped engine makes no thrust.")
            with ui.row().classes("items-baseline gap-2 no-wrap"):
                ui.label("now").classes("field-label").style(
                    f"color:{theme.INK_MUTED}")
                view["thrust"] = ui.label(_thrust_text()).classes("readout")
            widgets.hint_help(
                f"W and S run between the stops. Trim was "
                f"{t.get('thrust_n', 0.0):.0f} N.",
                "The band beside the live value is what the throttle keys "
                "run between, and it follows the max-thrust field above "
                "without rebuilding this panel — so the number keeps "
                "updating while you type.\n\n"
                "No propeller: thrust is a force along body x, and its line "
                "of action is stage 5's answer.",
                title="What the throttle does")

        lo, hi = _mass_band()
        row("mass_kg", "Weight", lo, hi, max((hi - lo) / 200.0, 1e-3), "kg",
            f"design {t.get('mass_kg', 0.0):.0f} kg. Inertia scales with it.")

        lo, hi = _cg_band()
        x_np = _x_np()
        row("x_cg_m", "CG station", lo, hi, max((hi - lo) / 200.0, 1e-4), "m",
            (f"neutral point at {x_np:.3f} m — the ORANGE marker in the "
             f"picture is the CG and the BLUE one is the neutral point, so "
             f"the gap between them is the static margin to scale. Drag this "
             f"and the airframe slides under a marker that stays put."
             if x_np is not None else
             "aft is a smaller static margin. Past the neutral point the "
             "aeroplane diverges."))

        # ---- the two that mean a NEW TRIM. Same widget as the three above
        # so they are reached the same way, and labelled so the difference
        # is not a surprise: these restart the clock.
        lo, hi = _speed_band()
        row("V_trim", "Speed  (re-trims)", lo, hi, max((hi - lo) / 200.0,
                                                       1e-3), "m/s",
            f"the design point is {mp['V']:.1f} m/s in "
            f"{mp['rho']:.4g} kg/m3 ({mp['medium']}). The aeroplane is "
            f"TRIMMED at whatever is set here — alpha, elevator and thrust "
            f"all move — so the clock and the trace start again.",
            setter=_set_condition, value=_V())
        lo, hi = _altitude_band()
        # NOT a re-trim. ``sixdof.trim_level`` takes an altitude and uses it
        # for the state's POSITION and nothing else — the density is the
        # mission's and lives on the Aircraft — so routing this lever
        # through the condition setter re-armed, nulled the pilot's
        # throttle, cleared the manoeuvre and threw the whole trace away to
        # arrive at the identical equilibrium. It moves the datum.
        row("altitude_m", "Altitude", lo, hi,
            max((hi - lo) / 200.0, 1e-3), "m",
            "where the aeroplane is put back. The density is the mission's "
            "and does not change with it — this stage flies ONE atmosphere, "
            "so treat it as the height the trace is drawn against. It does "
            "NOT re-trim: the equilibrium is the same at every height here.",
            setter=_set_altitude, value=_altitude())

        # ---- the canned excitations
        with widgets.group_box("Stability manoeuvres"):
            M = F.get("modes") or {}
            cols = F.get("ac").deck.columns if F.get("ac") else {}
            # ...ONLY THE ONES THIS DECK CAN FLY. A doublet written into an
            # axis the deck has not got produces a flat trace under a mode's
            # numbers, which reads as "the aeroplane did nothing" rather
            # than "the craft has no rudder".
            offered = [m for m in mv.MANOEUVRES if m.flyable_on(cols)]
            dropped = [m for m in mv.MANOEUVRES if m not in offered]
            for m in offered:
                # ...UNDER THE NAME THE ROOT CAME BACK WITH. The button is
                # a MANOEUVRE ("10 deg of bank and nothing else") and stays
                # flyable on any design; the number beside it is a MODE,
                # and on a design that does not weathercock the root the
                # bank excites is not a spiral. Printing its numbers under
                # the manoeuvre's name without saying so is the same
                # invented stability the namer was taught to refuse.
                mode, mode_name = md.find(M, m.mode)
                with ui.row().classes("w-full items-center gap-2 no-wrap"):
                    # the ?, not a tooltip: each of these is two or three
                    # sentences about what the manoeuvre EXCITES, which is
                    # the whole reason to press one rather than another
                    widgets.explain(
                        ui.button(m.label,
                                  on_click=lambda _=None, k=m.key:
                                  _manoeuvre(k))
                        .props("dense flat no-caps").classes("w-36"),
                        m.hint, title=m.label)
                    ui.label(_mode_line(mode)
                             + ("" if mode_name == m.mode
                                else f"   [{mode_name}]")) \
                        .classes("readout").style(
                        f"color:{theme.BAD if (mode is not None and not mode.stable) else theme.INK_MUTED}")
            if dropped:
                widgets.hint_help(
                    "Not offered here: "
                    + ", ".join(m.label for m in dropped) + ".",
                    "Each one is a doublet or a step into a control this "
                    "craft has not got.\n\n"
                    "A script written into a missing axis flies nothing "
                    "while still printing a mode's numbers underneath it, "
                    "which reads as 'the aeroplane did not respond' rather "
                    "than 'there is no such control'.",
                    title="Why these are missing", kind="warn")
            widgets.hint_help(
                "Each re-trims first, then applies its own excitation.",
                "The number beside a manoeuvre is what the LINEARISED model "
                "predicts; the trace on the next tab is what the aeroplane "
                "actually did.\n\n"
                "They are two different calculations off one deck — "
                "eigenvalues of a numerical Jacobian against a full "
                "nonlinear integration — so a disagreement between them is "
                "worth looking at.",
                title="What a manoeuvre button does")

        with widgets.group_box("Keys"):
            for keys, what in stk.LEGEND:
                widgets.kv(keys, what)

        with widgets.group_box("Gamepad (DualSense)"):
            view["padlbl"] = ui.label().classes("readout")
            _pad_status()
            for keys, what in pd.LEGEND:
                widgets.kv(keys, what)
            widgets.hint_help(
                "Analogue stick: half travel is half deflection.",
                "That is the one thing a keyboard cannot ask for — the keys "
                "are rate-based and spring-centred.\n\n"
                "A browser only sees a pad after a button has been pressed "
                "on it, so press one if this says no gamepad. The Gamepad "
                "API is a BROWSER feature: run the shell with --browser if "
                "the native window never sees it.\n\n"
                "The pad is polled in the animation frame, which the "
                "browser stops for a window that is not in front, so the "
                "flight window has to be the one you are looking at.",
                title="Using a gamepad")

    def _stick_panel():
        holder = ui.element("div").style("width:560px;max-width:100%")
        with holder, \
                widgets.group_box("Stick (degrees) — drag, or read the "
                                  "keyboard's demand off the panel left of "
                                  "the picture"):
            for key, label, lo, hi in (
                    ("elevator", "elevator (+ = nose down)", -20, 20),
                    ("aileron", "aileron (+ = roll to port)", -25, 25),
                    ("rudder", "rudder (+ = nose to starboard)", -25, 25)):
                cols = F.get("ac").deck.columns if F.get("ac") else {}
                if key not in cols:
                    continue
                ui.label(label).classes("field-label")
                sl = ui.slider(min=lo, max=hi, step=0.5,
                               value=F["stick"].get(key, 0.0)) \
                    .props("dense label")
                sl.on_value_change(
                    lambda e, k=key: F["stick"].__setitem__(k, float(e.value)))
                view["fields"][f"stick_{key}"] = sl

    def _teardown():
        """Stop the loop and let go of the stick, without repainting."""
        F["running"] = False
        timer = view.pop("timer", None)
        view["timer"] = None
        if timer is not None:
            timer.cancel()
        view["wall"] = None
        F["hold"].update(dict.fromkeys(F["hold"], 0.0))
        # the KEYS too, or the derived hold is rebuilt from a stale set the
        # moment one of them is released
        F.setdefault("keys_down", set()).clear()
        # ...and the PAD's half of the demand, for exactly the same reason:
        # a stick still deflected when Pause was pressed would otherwise be
        # added back into the next axis the keyboard touches.
        F.setdefault("pad", {}).update(
            dict.fromkeys(F.get("pad") or {}, 0.0))

    def _run(on: bool):
        """Start or stop flying — and CREATE OR DESTROY the timer with it.

        A ui.timer left running at 30 Hz for the whole session costs a client
        round trip every tick whether or not anything is moving: the shell
        idled at 69 % CPU and the browser stopped answering, with the
        simulation stopped. A frame loop must not exist while there is
        nothing to animate.
        """
        if on:
            F["running"] = True
            if F.get("ac") is None:
                arm()
            # the clock restarts HERE, not where it stopped: a simulation
            # paused for a minute must not resume by integrating a minute
            view["wall"] = None
            if view.get("timer") is None:
                view["timer"] = ui.timer(1.0 / FPS, _frame)
        else:
            # let go of the stick when the pilot lets go of the aeroplane:
            # a key still held when Pause was pressed would otherwise be
            # held for ever, because the keyup lands on a stopped loop
            _teardown()
        ctx.refresh("flight")

    def _build_scene():
        """The aircraft, from the SAME loft the Results stage exports.

        Three levels, and each one is a frame:

        * the OUTER group is the aeroplane's attitude — its origin is the CG,
          because that is what an aeroplane rotates about;
        * the BODY group inside it carries the fixed G->body half turn and the
          CG offset, so moving the CG lever slides the airframe under a
          marker that does not move;
        * the markers: the CG on the pivot, and the NEUTRAL POINT drawn with
          the airframe. The gap between them IS the static margin, to scale.
        """
        from nicegui import app

        fm = _fm()
        b = float(fm.deck.b) if fm else 10.0
        view["span"] = b
        # the markers are sized off the SPAN, not in metres: a 0.12 m sphere
        # is invisible on a 10 m wing and fills the picture on a 0.8 m one
        r = max(0.030 * b, 0.05)
        try:
            url = _stl_url(app)
        except Exception:                              # noqa: BLE001
            url = None
            print("[v4] flight mesh failed:\n" + traceback.format_exc())
        # grid=False, and it is NOT a style choice. The built-in GridHelper
        # goes straight onto THREE.Scene and is never entered in the
        # addressable object map (scene.js:184-188), so no Python call can
        # move, hide or resize it — and it sits at scene z = 0, which is a
        # false floor at the DATUM altitude rather than the ground. Ours is
        # below, at -datum, where altitude zero actually is.
        with ui.scene(width=1280, height=720, grid=False, fps=lv.SCENE_FPS,
                      camera=ui.scene.perspective_camera(
                          fov=70, near=max(0.05, 0.02 * b),
                          far=4.0 * GROUND_HALF_M),
                      background_color="#8fb7dd") as scene:
            view["scene"] = scene
            _build_world(b)
            with ui.scene.group() as g:
                with ui.scene.group() as body:
                    body.rotate_R([list(row) for row in A_G_TO_BODY])
                    if url:
                        ui.scene.stl(url).material("#2f4a63")
                    else:
                        # the loft could not be built; draw the PLANFORM as a
                        # box rather than an empty scene
                        ui.scene.box(fm.deck.mac if fm else 1.0, b, 0.08) \
                            .material("#2f4a63")
                    # the neutral point, in the SAME frame as the loft, so it
                    # lands where the lattice put it rather than where a
                    # separate drawing routine thinks it should be
                    x_np = _x_np()
                    if x_np is not None:
                        view["np_marker"] = [
                            ui.scene.sphere(0.7 * r).material("#1565c0")
                            .move(float(x_np), 0.0, 0.0),
                            # BELOW the airframe, while the CG's label is
                            # above: the two markers are a static margin
                            # apart, which on a short chord is close enough
                            # for two labels at the same height to sit on
                            # top of each other
                            ui.scene.text("NP",
                                          "color:#1565c0;font-size:11px")
                            .move(float(x_np), 0.0, -2.4 * r)]
                view["body"] = body
                # ...and the CG, ON THE PIVOT. Drawn as a sphere between two
                # stubs so it reads as a point on an axis rather than a
                # floating ball.
                view["cg_marker"] = [
                    ui.scene.sphere(r).material("#d98324"),
                    ui.scene.line([-2.2 * r, 0.0, 0.0],
                                  [2.2 * r, 0.0, 0.0]).material("#d98324"),
                    ui.scene.line([0.0, 0.0, -2.2 * r],
                                  [0.0, 0.0, 2.2 * r]).material("#d98324"),
                    ui.scene.text("CG", "color:#a35c10;font-size:11px")
                    .move(0.0, 0.0, 2.6 * r)]
            view["group"] = g
        scene.style("width:100%;height:100%")
        # the ENGINEERING camera, stashed so the toggle can put it back
        view["eng_cam"] = dict(x=-0.80 * b, y=-0.80 * b, z=0.34 * b,
                               look_at_x=0.0, look_at_y=0.0, look_at_z=0.0)
        _place_cg()
        _apply_mode()
        _pose()

    def _emit(code: str) -> bool:
        """Send a script to this view's page, or report that there is nowhere
        to send it.

        A headless render — the test harness, and any render that happens
        before the server's event loop exists — has no loop, and nicegui's
        ``AwaitableResponse`` asserts on exactly that. The precondition is
        CHECKED rather than the assertion caught: swallowing it here would
        hide a genuine transport failure behind a picture that simply never
        moves.
        """
        from nicegui import core

        client = view.get("client")
        if client is None or core.loop is None:
            return False
        client.run_javascript(code)
        return True

    def _install():
        """Hand the browser its half of the render loop. Once per view.

        It has to happen after BOTH the scene and the glass exist, because it
        carries their element ids; and it is fire-and-forget — an awaited
        ``run_javascript`` would block this render on a round trip.
        """
        scene, g, world = (view.get("scene"), view.get("group"),
                           view.get("world"))
        hud, read = view.get("hud"), view.get("readout")
        banner = view.get("banner")
        panel, thrust = view.get("panel"), view.get("thrust")
        if None in (scene, g, world, hud, read, banner, panel, thrust):
            return
        view["client"] = ui.context.client
        # ONE dict, sent and recorded. Building the probe's record separately
        # from the arguments lets the two disagree, and then a test that reads
        # the record cannot see an id being handed over the wrong way round —
        # which is the mistake this record exists to catch.
        ids = {"scene_id": scene.id, "group_id": g.id, "world_id": world.id,
               "hud_id": hud.id, "readout_id": read.id,
               "banner_id": banner.id, "panel_id": panel.id,
               "thrust_id": thrust.id}
        view["install"] = {k[:-3]: v for k, v in ids.items()}
        view["installed"] = _emit(lv.INSTALL_JS(
            **ids, span_m=float(view.get("span") or 10.0),
            snap_m=WORLD_SNAP_M, back=CHASE_BACK, up=CHASE_UP,
            lead=CHASE_LEAD, apply_js=hud_svg.APPLY_JS,
            gauge_js=gg.APPLY_JS, pad_cfg=pd.js_config()))
        _readout()
        _pose()

    def _crash(st):
        """The ground is not a suggestion.

        Kept HERE and out of :mod:`aerobo.sixdof` on purpose: the engine is
        shared with the trim solver and with ``linearise``, whose numerical
        Jacobian perturbs the state in BOTH directions and would be corrupted
        by a hard floor. A stop rule belongs to the thing being flown, not to
        the equations of motion.
        """
        F["running"] = False
        F["crashed"] = True
        _teardown()
        F["error"] = (
            f"GROUND CONTACT at {st.V:.1f} m/s, "
            f"{np.rad2deg(st.euler[1]):+.0f} deg pitch, "
            f"{np.rad2deg(st.euler[0]):+.0f} deg bank. Re-trim & reset to fly "
            f"again.")
        # ...AND IT HAS TO REACH THE SCREEN. ``F["error"]`` is drawn by
        # ``_render_fly``, which nothing calls here, and the HUD warning
        # that carries the short form is suppressed outside the game view —
        # so in Inspect the picture simply stopped, with no word anywhere
        # about why. The banner is the frame path's own channel and it is
        # repainted below, so the sentence lands in both views.
        F["crash_banner"] = F["error"]
        _readout()

    def _game() -> bool:
        return (F.get("mode") or "game") == "game"

    def _chase_eye(st):
        """Where the chase camera would sit — the SAME closed form the
        browser half runs, kept here so it can be checked in Python.

        The game view does not send this: :mod:`gui.v4.live` computes it from
        the pose it is actually drawing, which is both smoother than a pose
        that arrived 30 ms ago and costs no message at all.
        """
        return wld.chase_eye(
            scene_rotation(st.quat), view.get("span") or 10.0,
            back=CHASE_BACK, up=CHASE_UP, lead=CHASE_LEAD,
            drawn_z=float(st.altitude_m - _altitude()))

    def _apply_mode():
        """Show the world or the workshop. NOT a rebuild — four properties.

        A rebuild here would re-download the loft and re-create two hundred
        ground objects to answer a question about visibility.
        """
        game = _game()
        for key, on in (("world", game), ("ref_grid", not game),
                        ("cg_marker", not game), ("np_marker", not game)):
            # some slots hold a GROUP and some hold a list of loose objects
            # (a sphere, two stubs and a label are not worth a group of their
            # own), so both shapes are accepted here rather than forcing one
            held = view.get(key)
            if held is None:
                continue
            for obj in (held if isinstance(held, (list, tuple)) else [held]):
                obj.visible(on)
        scene = view.get("scene")
        if scene is None:
            return
        if game:
            # nothing to do: the next report carries ``game`` and the browser
            # takes the camera back over. Pushed now so a PAUSED simulation
            # still swings round when the mode is switched.
            _pose()
        else:
            scene.move_camera(duration=CHASE_TWEEN_S,
                              **(view.get("eng_cam") or {}))
            _pose()

    def _set_mode(mode: str):
        F["mode"] = "game" if mode == "game" else "engineering"
        _apply_mode()
        _readout()
        _pose()

    def _build_world(b: float):
        """The ground, drawn once and then never touched again.

        Every object here is STATIC in its own local coordinates: what makes
        the world stream past is the single translation on the group
        (:func:`gui.v4.world.world_offset`). That is the whole reason the
        picture costs one message a frame instead of two hundred.

        Two lattices, and both periods divide :data:`WORLD_SNAP_M`, or the
        re-anchor would not map them onto themselves and the scenery would
        jump once a tile.
        """
        with ui.scene.group() as world:
            n = int(GROUND_HALF_M / GRID_PITCH_M)
            for i in range(-n, n + 1):
                c = i * GRID_PITCH_M
                ui.scene.line([-GROUND_HALF_M, c, 0.0],
                              [GROUND_HALF_M, c, 0.0]).material("#4a7a43")
                ui.scene.line([c, -GROUND_HALF_M, 0.0],
                              [c, GROUND_HALF_M, 0.0]).material("#4a7a43")
            # posts, because a flat lattice gives distance but not HEIGHT,
            # and height is what tells a pilot how low they are.
            #
            # PLACED BY ``world.lattice_offsets``, which is where the
            # tile-crossing guarantee is written down and tested. It was
            # tested against a copy of these two loops, so the property the
            # tests pin was not the property the picture had.
            for x, y, _z in wld.lattice_offsets(
                    POST_PITCH_M, int(POST_HALF_M / POST_PITCH_M)):
                ui.scene.box(9.0, 9.0, 34.0).material("#3f5d3a", 0.9) \
                    .move(x, y, 17.0)
        view["world"] = world
        # ...and a span-sized reference lattice for the ENGINEERING view,
        # which is what the built-in grid used to be and is better sized off
        # the aeroplane than fixed at 100 m
        with ui.scene.group() as ref:
            k, pitch = 6, max(b / 4.0, 0.25)
            for i in range(-k, k + 1):
                c = i * pitch
                ui.scene.line([-k * pitch, c, 0.0],
                              [k * pitch, c, 0.0]).material("#c3ccd6")
                ui.scene.line([c, -k * pitch, 0.0],
                              [c, k * pitch, 0.0]).material("#c3ccd6")
        view["ref_grid"] = ref

    def _x_np() -> float | None:
        """The neutral point in the loft's own frame, or None if the rebuilt
        lattice will not give one. Never invented: a marker with no number
        behind it would be a drawing, not a measurement."""
        fm = _fm()
        if fm is None:
            return None
        try:
            return float(fm.model.neutral_point())
        except Exception:                              # noqa: BLE001
            return None

    def _flown_geometry():
        """The report's geometry with its FIN REPLACED BY THE FLOWN ONE.

        The loft and the lattice had two different authors for one surface.
        ``cad.fin_surface`` reads ``geometry["fin"]`` — the DESIGN REPORT's
        block — while the aeroplane the integrator flies carries
        ``fm.model.vertical``, built from stage 5's :class:`ControlsSpec`
        (flightmodel.py:554-587). They coincide only until somebody answers
        stage 5, and then every fin lever moved the physics and left the
        picture bit-identical: a fin switched OFF still drew a fin, and a fin
        resized to 2.5 m drew the report's 1.04 m one.

        So the drawing is taken from the surface that is FLOWN, and the
        report's block is the fallback for a headless shell with no stage 5.
        ``height`` is signed here exactly as it is in the lattice, so a
        ventral fin draws hanging down without a second convention.

        The lattice's ``vertical.y`` is not carried: ``fin_surface`` puts the
        fin on the plane of symmetry and uses y for the section's THICKNESS.
        Nothing in this package builds an offset fin, and drawing one would
        need a loft that does not exist rather than a line here.
        """
        rep = S["run"].get("report") or {}
        geom = rep.get("geometry") or {}
        fm = _fm()
        if fm is None:
            return rep, geom, None
        v = getattr(fm.model, "vertical", None)
        geom = dict(geom)
        if v is None:
            # THE FLOWN AEROPLANE HAS NO VERTICAL SURFACE — a V-tail, a
            # report that states none, or stage 5's switch off. The deck says
            # so (no ``rudder`` column, ``Cn_beta`` an honest zero) and now
            # the picture does too.
            geom["fin"] = None
            return rep, geom, ("none",)
        blk = dict((geom.get("fin") or {}))
        blk.update(chord_m=float(v.chord), height_m=float(v.height),
                   z_root_m=float(v.z_root), x_qc_m=float(v.x))
        blk.setdefault("tc", _fin.FIN_TC_DEFAULT)
        geom["fin"] = blk
        return rep, geom, (blk["chord_m"], blk["height_m"],
                           blk["z_root_m"], blk["x_qc_m"], blk["tc"])

    def _stl_url(app):
        """Write the loft and serve it, once per AEROPLANE.

        Cached on the report AND the flown fin, not on "have we ever written
        one": ``view`` outlives every rebuild of the Fly view, so a URL keyed
        on nothing served the first design's mesh for the rest of the
        session.
        """
        import tempfile
        from pathlib import Path

        from aerobo import cad

        rep, geom, fin_key = _flown_geometry()
        key = view.get("url_key")
        if view.get("url") and key is not None \
                and key[0] is rep and key[1] == fin_key:
            return view["url"]
        surfs = cad.surfaces(geom)
        if not surfs:
            return None
        d = Path(tempfile.mkdtemp(prefix="aerobo_flight_"))
        p = d / "aircraft.stl"
        blob = cad.stl_bytes(surfs, binary=True)
        p.write_bytes(blob)
        # KEYED ON THE MESH, NOT THE CLOCK. A whole-second wall-clock
        # stamp gave two different lofts the same path whenever they were
        # built inside one second — switch the fin off in stage 5 and click straight through
        # to stage 6 — and Starlette resolves to the FIRST route registered,
        # so the browser was served the aeroplane that is no longer flying.
        # A content hash cannot collide with a different mesh, and it makes
        # the route reusable: the same geometry twice registers once.
        url = f"/_flight/{hashlib.sha1(blob).hexdigest()[:16]}.stl"
        app.add_static_file(local_file=str(p), url_path=url)
        view["url"] = url
        view["url_key"] = (rep, fin_key)
        return url

    def _readout():
        """The numbers, once. Label, banner and the HUD's state all come off
        ONE evaluation of the equations of motion — it used to be two, which
        is a whole extra aerodynamic solve per frame to print the same g."""
        st = F.get("state")
        if st is None:
            return
        from aerobo import sixdof as sd

        ac = F.get("ac")
        roll, pitch, yaw = st.euler
        gz, sm = 1.0, float("nan")
        if ac is not None:
            # THE LOAD FACTOR IS A SPECIFIC FORCE, so it is taken off the
            # forces and not reconstructed from the state derivative.
            # ``derivative`` returns ``F/m + g_b - w x v``; the old
            # expression added the gravity term back and silently dropped
            # the rate one, so the number was right only while p = q = r = 0
            # — which is exactly when nobody is looking at it. Measured on
            # the shipped tail design, 0.7 s into a pull at q = 31.8 deg/s:
            # it printed +0.917 against a true +1.714, and in a 60 deg
            # banked turn it read -0.463 for a load factor of +1.014, which
            # took the glass amber while the aeroplane was pulling positive
            # g. It is also cheaper: ``derivative`` calls this itself and
            # the other twelve components were thrown away here.
            Fb, _Mb = ac.forces_moments(st)
            gz = float(-Fb[2] / (ac.inertia.mass_kg * sd.G))
            sm = float(ac.deck.static_margin)
        view["hud_state"] = _hud_state(st, gz, sm)

        L = F["live"]
        # WHAT THE GLASS AND THE PANELS ALREADY SAY IS NOT REPEATED HERE.
        #
        # The line used to be four rows of thirty numbers, and most of them
        # were on screen twice: speed and altitude are the two HUD tapes,
        # alpha and the load factor are on the glass, the heading is its
        # compass, pitch is the ladder, and bank, roll rate and the three
        # stick angles are the panel on the left. A number in two places is
        # a number nobody reads in either.
        #
        # What is left is exactly what nothing else shows: the clock, the
        # SIDESLIP (the glass has no beta, and the panel's three rows are
        # attitude and not the aerodynamic angles), and the three levers
        # whose whole purpose is that moving one changes the static margin
        # printed beside it. All three body rates went to the panel with
        # their angles when pitch and yaw joined roll there.
        text = (
            f"t {F.get('t', 0.0):6.1f} s   "
            f"sideslip {np.rad2deg(st.beta):+5.2f} deg\n"
            f"mass {float(L.get('mass_kg') or 0.0):6.0f} kg   "
            f"x_cg {float(L.get('x_cg_m') or 0.0):+6.3f} m   "
            f"SM {sm:+.3f}")
        del gz
        # the one number that has to SHOUT: an aeroplane whose CG is aft of
        # its neutral point is not badly trimmed, it is unstable, and the
        # divergence the pilot is about to watch is the model working
        if F.get("crash_banner"):
            # STICKY, and first: the aeroplane is on the ground, which is
            # the whole of what the reader needs to know about the picture
            # that just stopped moving. Cleared by a re-trim, not by a
            # frame.
            banner, colour = str(F["crash_banner"]), "#c62828"
        elif sm == sm and sm < 0.0:
            banner = (f"CG IS AFT OF THE NEUTRAL POINT — static margin "
                      f"{sm:+.3f}. This aeroplane is statically unstable "
                      f"in pitch.")
            colour = "#c62828"
        elif abs(np.rad2deg(st.alpha)) > 12.0:
            banner = (f"alpha {np.rad2deg(st.alpha):+.1f} deg — at or past "
                      f"the stall the deck is extrapolated, not measured.")
            colour = "#b26a00"
        else:
            banner, colour = "", "inherit"
        # NOT written onto the labels. See gui/v4/live.py's writeText for the
        # measurement: an element update once a frame cost a quarter of every
        # frame, because nicegui patches a page holding all six stages. The
        # colour goes with the text so a warning that clears also clears its
        # red, which the old branch forgot to do.
        view["text_state"] = {"readout": text, "banner": banner,
                              "color": colour, "thrust": _thrust_text()}
        # the panel rides the same frame. Every decision about what it
        # shows is made HERE, in Python, so the browser half of
        # gui/v4/gauge.py knows nothing about a bank angle or a rate scale.
        view["gauge_state"] = gg.attitude_state(
            roll_deg=float(np.rad2deg(roll)),
            pitch_deg=float(np.rad2deg(pitch)),
            yaw_deg=float(np.rad2deg(yaw)),
            rates_dps=tuple(float(np.rad2deg(v)) for v in st.rates),
            # the keyboard's DEMAND, on the side of the picture, with each
            # bar scaled by its own stop. It cannot live on the Quasar
            # sliders: those are Vue-owned, so tracking a held key on one is
            # an element update a frame
            stick=F["stick"], limits=stk.LIMITS)

    def _hud_state(st, gz: float, sm: float) -> dict | None:
        """The twenty numbers on the glass, or None when there is no glass.

        Switching it off HIDES the layer rather than blanking its numbers:
        the skeleton is a ladder, two tapes and a bank scale, and clearing
        the text would leave all of that on the glass. The style is only
        written when the answer CHANGES, so this costs nothing per frame.
        """
        box, on = view.get("hud"), bool(F.get("hud", True) and _game())
        if box is not None and view.get("hud_on") != on:
            box.style("display:block" if on else "display:none")
            view["hud_on"] = on
        if not on:
            return None
        roll, pitch, yaw = st.euler
        warn = []
        if F.get("crashed"):
            warn.append("GROUND CONTACT")
        if abs(np.rad2deg(st.alpha)) > 12.0:
            warn.append("ALPHA")
        if sm == sm and sm < 0.0:
            warn.append("CG AFT OF NEUTRAL POINT")
        if st.altitude_m < 40.0:
            warn.append("LOW")
        return hud_svg.state(
            speed=float(st.V), altitude=float(st.altitude_m),
            heading_deg=float(np.rad2deg(yaw)),
            pitch_deg=float(np.rad2deg(pitch)),
            roll_deg=float(np.rad2deg(roll)),
            alpha_deg=float(np.rad2deg(st.alpha)), g=float(gz),
            throttle=float(F["live"].get("thrust_n") or 0.0),
            throttle_min=_thrust_band()[0],
            throttle_max=_thrust_band()[1], warnings=warn)

    # --------------------------------------------------------- view: traces
    def _trace_series():
        """The six traces, as ``(name, t, y)``, off the recorded history.

        The recorded TIMES, not ``len * dt``: the frames are not evenly
        spaced and pretending they are stretches every trace.
        """
        import numpy as np

        hist = F.get("history") or []
        if len(hist) < 2:
            return []
        times = F.get("history_t") or []
        t = (np.asarray(times, dtype=float) if len(times) == len(hist)
             else np.arange(len(hist)) / FPS)
        return [
            (name, t, y) for name, y in (
                ("Speed [m/s]", [s.V for s in hist]),
                ("Altitude [m]", [s.altitude_m for s in hist]),
                ("alpha [deg]", [np.rad2deg(s.alpha) for s in hist]),
                ("Roll rate p [deg/s]",
                 [np.rad2deg(s.rates[0]) for s in hist]),
                ("Bank [deg]", [np.rad2deg(s.euler[0]) for s in hist]),
                ("Pitch [deg]", [np.rad2deg(s.euler[1]) for s in hist]),
            )]

    def _trace_fig(name, t, y):
        import plotly.graph_objects as go

        fig = go.Figure(go.Scatter(x=t, y=y, mode="lines", name=name))
        fig.update_layout(margin=dict(l=48, r=8, t=8, b=32),
                          xaxis_title="t [s]", yaxis_title=name,
                          showlegend=False)
        return fig

    def _traces_tick():
        """Redraw the six panes IN PLACE while the aeroplane is still flying.

        Leaving this tab was never meant to pause the simulation (a test
        pins that it does not), so the plots froze at the instant the tab
        was opened while the history went on growing behind them — and the
        only way to see the rest of a manoeuvre was to click away and back,
        which rebuilds the Fly view and tears the flight down. So "go and
        look at the trace" quietly ended the flight it was about.

        ``figstyle.update`` rather than a rebuild: a redrawn container is a
        new DOM node, and a view that changes height twice a second throws
        the page back to the top (the repaint-is-not-a-rebuild rule this
        shell already paid for once).
        """
        panes = view.get("traces")
        if not panes or not F.get("running"):
            return
        if S["ui"]["tab"].get("flight") != "traces":
            return          # not on screen: a redraw nobody can see
        for name, t, y in _trace_series():
            figstyle.update(panes.get(name), _trace_fig(name, t, y),
                            f"flight_{name}", 200)

    def _render_traces():
        box = ctx.views[("flight", "traces")]
        box.clear()
        view["traces"] = {}
        with box:
            series = _trace_series()
            if not series:
                widgets.hint("Fly first — there is no history yet.")
                return
            widgets.hint_help(
                "Live while the aeroplane is flying.",
                "The six panes are redrawn in place about twice a second "
                "for as long as this tab is the one on screen.\n\n"
                "Leaving stage 6 puts the aeroplane down; moving between "
                "its tabs does not, so a manoeuvre can be watched here "
                "while it happens.",
                title="What these traces are")
            for name, t, y in series:
                with widgets.group_box(name, pad=False):
                    view["traces"][name] = figstyle.show(
                        _trace_fig(name, t, y), f"flight_{name}", 200)

    # ---------------------------------------------------------- view: modes
    def _render_modes():
        box = ctx.views[("flight", "modes")]
        box.clear()
        with box:
            ac, st = F.get("ac"), F.get("state")
            if ac is None or st is None:
                widgets.hint("Arm the simulation first (open the Fly tab).")
                return
            import numpy as np

            from aerobo import sixdof as sd

            J = sd.linearise(ac, st)
            # NAMED first, because "which eigenvalue is the spiral" is the
            # question, and sorting by real part does not answer it. The
            # participation is printed with the name so a label that came
            # out of an ambiguous split can be disbelieved.
            # ...AND THE NAMER IS TOLD WHETHER THIS AEROPLANE WEATHERCOCKS.
            # Without it the slow lateral root is called "spiral" on every
            # design, so a finless one read "spiral -0.02221, halves in
            # 31.2 s" — a converging spiral on an aircraft whose Cn_beta is
            # exactly -0.0, while stage 3's card refused the same design.
            #
            # TAKEN FROM ``_stability``, NOT RECOMPUTED. That function
            # exists to classify once per change of aeroplane and store the
            # answer, and the Fly tab's strip and the manoeuvre rows both
            # read it — this view, whose whole job is the modes, was the one
            # place that ran its own classify. Two instruments, one
            # aeroplane, and nothing forcing them to agree.
            _cnb = getattr(getattr(ac, "deck", None), "Cn_beta", None)
            if F.get("modes") is None:
                _stability()
            named = F.get("modes") or md.classify(J, float(st.V),
                                                  Cn_beta=_cnb)
            why_not = md.why_not_a_spiral(_cnb)
            with widgets.group_box("The modes"):
                if why_not:
                    widgets.hint(why_not, "warn")
                for key in ("phugoid", "short period", "dutch roll",
                            md.LATERAL_OSC, "roll subsidence", "spiral",
                            md.SLOW_LATERAL):
                    m = named.get(key)
                    if m is None:
                        # a name that is absent BECAUSE this design does not
                        # weathercock is not a gap in the measurement, and
                        # saying "not present" of both is how the two got
                        # confused in the first place
                        if why_not and key in ("spiral", "dutch roll"):
                            continue
                        if key in (md.SLOW_LATERAL, md.LATERAL_OSC):
                            continue   # an alias with no root is not a gap
                        widgets.kv(key, "not present in this design",
                                   color=theme.INK_FAINT)
                        continue
                    widgets.kv(
                        key,
                        f"{_mode_line(m)}      "
                        f"{m.real:+.5f}"
                        + (f" +- {m.imag:.5f} j" if m.oscillatory else "")
                        + f"   ({'lateral' if m.lateral > 0.5 else 'longitudinal'}"
                        f" {max(m.lateral, 1 - m.lateral):.2f})",
                        color=("" if m.stable else theme.BAD))
                widgets.hint_help(
                    "Each name comes from the EIGENVECTOR, not the order.",
                    "A mode is called longitudinal or lateral according to "
                    "how much of its eigenvector is pitch-and-speed versus "
                    "roll-yaw-and-sideslip, and that fraction is the number "
                    "in brackets.\n\n"
                    "On a symmetric aeroplane in level flight the split is "
                    "1.00 either way. A fraction near 0.5 means the name "
                    "beside it is a guess.",
                    title="How a mode gets its name")
            ev = np.linalg.eigvals(J)
            ev = ev[np.argsort(-ev.real)]
            with widgets.group_box("Every eigenvalue"):
                shown = 0
                for e in ev:
                    if abs(e.real) < 1e-9 and abs(e.imag) < 1e-9:
                        continue          # the quaternion-norm direction
                    if abs(e.imag) > 1e-6:
                        T = 2 * np.pi / abs(e.imag)
                        zeta = -e.real / abs(e)
                        widgets.kv(
                            f"{e.real:+.5f} ± {abs(e.imag):.5f} j",
                            f"period {T:7.2f} s, damping {zeta:+.4f}",
                            color=("#c62828" if e.real > 0 else ""))
                    else:
                        tau = (1.0 / abs(e.real)) if abs(e.real) > 1e-9 \
                            else float("inf")
                        widgets.kv(f"{e.real:+.5f}",
                                   f"time constant {tau:8.2f} s",
                                   color=("#c62828" if e.real > 0 else ""))
                    shown += 1
                    if shown >= 10:
                        break
            widgets.hint_help(
                "Red GROWS. Jacobian of the same equations the Fly tab flies.",
                "A slightly divergent phugoid is ordinary and flyable; a "
                "divergent short period or spiral is not. A root within "
                "1e-4 per second of zero is reported as NEUTRAL rather than "
                "divergent, because 'doubles in 32 hours' is not a "
                "stability finding.\n\n"
                "Four of the thirteen eigenvalues are the position states "
                "and are structurally zero — they are not modes.\n\n"
                "These come from a numerical Jacobian of the same equations "
                "the Fly tab integrates, so a mode seen here is the one "
                "flown there, not a separate linear model.",
                title="Reading the eigenvalues")

    # ---- LEAVING THE STAGE PUTS THE AEROPLANE DOWN.
    #
    # The frame loop is a ``ui.timer``, and a timer does not care which
    # stage is on screen: selecting stage 5 left the simulation integrating
    # at 30 Hz behind a page nobody was watching. Measured on the tail
    # design: three seconds spent on the Controls stage took it from 300 m
    # to 62 m and from level to 36 degrees nose down, and it would have hit
    # the ground unattended a second later — so "I went to Controls and came
    # back and everything had changed" is not a redraw, it is the aeroplane
    # having actually flown there. Coming BACK then stops the loop (the Fly
    # view is rebuilt, and a render tears the timer down), which is what
    # made it look like a reset rather than a runaway.
    #
    # Hooked on the shell's ``select`` rather than inside ``_render_fly``
    # because leaving is the event: by the time another stage renders, this
    # stage's own renderer is not called at all.
    _shell_select = ctx._select

    def _select(stage: str, view_key: str | None = None):
        _shell_select(stage, view_key)
        # ...on where the shell ENDED UP, not on where it was asked to go: a
        # locked stage is refused and leaves the selection alone, and that
        # must not put the aeroplane down.
        if S["ui"]["selected"] != "flight" and F.get("running"):
            _run(False)
            ctx.log("flight paused — the aeroplane is where you left it, "
                    "at t = %.1f s. Fly picks it up from there." %
                    float(F.get("t") or 0.0), "info")

    ctx._select = _select

    ctx.on_render("flight", "fly", _render_fly)
    ctx.on_render("flight", "setup", _render_setup)
    ctx.on_render("flight", "traces", _render_traces)
    ctx.on_render("flight", "modes", _render_modes)
    ctx.register("flight_arm", arm)
    ctx.register("flight_key", _on_key)
    ctx.register("flight_pad", _on_pad)
    ctx.register("flight_run", _run)
    ctx.register("flight_set_live", _set_live)
    ctx.register("flight_advance", _advance)
    # a read-only probe of what is actually in the scene. The alternative is
    # a test that asserts the code it is testing, or none at all.
    ctx.register("flight_set_mode", _set_mode)
    ctx.register("flight_pose", _pose)
    ctx.register("flight_readout", _readout)
    ctx.register("flight_install", _install)
    ctx.register("flight_reset", _reset)
    ctx.register("flight_payload", _payload)
    ctx.register("flight_chase_eye", _chase_eye)
    ctx.register("flight_set_condition", _set_condition)
    ctx.register("flight_set_thrust_max", _set_thrust_max)
    ctx.register("flight_thrust_text", _thrust_text)
    ctx.register("flight_manoeuvre", _manoeuvre)
    ctx.register("flight_stability", _stability)
    # THE GEOMETRY THE SCENE IS LOFTED FROM, reachable without a browser.
    # ``_stl_url`` writes a file and serves it, so the only way to ask "does
    # the picture carry the fin the aeroplane is flying" used to be to look
    # at it. Returns ``(report, geometry, fin_key)``.
    ctx.register("flight_drawn_geometry", _flown_geometry)
    ctx.register("flight_frame_traces", _traces_tick)
    ctx.register("flight_probe", lambda: {
        "body_xyz": ((view["body"].x, view["body"].y, view["body"].z)
                     if view.get("body") is not None else None),
        # the group is no longer moved from Python in the game view — the
        # browser is — so what a test has to check is the REPORT
        "payload": _payload(),
        "labels": sorted(o.args[0] for o in
                         (view["scene"].objects.values()
                          if view.get("scene") is not None else [])
                         if o.type == "text"),
        "x_np": _x_np(),
        # the six trace panes, by name, so a test can tell a redraw from a
        # rebuild by object identity
        "traces": view.get("traces"),
        # the CG lever's TRAVEL, because the stage's headline lesson is
        # dragging the CG past the neutral point and a band that cannot
        # reach it makes the red banner dead code
        "cg_band": _cg_band(),
        "V": _V(), "rho": _mission_point()["rho"],
        "altitude": _altitude(),
        "world_xyz": ((_payload() or {}).get("w")),
        "world_visible": (bool(view["world"].visible_)
                          if view.get("world") is not None else None),
        "cg_visible": (bool(view["cg_marker"][0].visible_)
                       if view.get("cg_marker") else None),
        # object IDENTITY, so a test can tell a repaint from a rebuild
        "object_ids": (sorted(view["scene"].objects)
                       if view.get("scene") is not None else None),
        # what the browser half was actually told to address. A test that
        # only checks INSTALL_JS formats its arguments cannot see the ids
        # being handed over the wrong way round.
        "install": view.get("install"),
        # the numbers, and the two places they could be. `text_state` is what
        # travels with the frame; `label_text` is what nicegui's element
        # holds — and it has to stay EMPTY, because a frame that fills it is
        # a Vue patch of a page carrying all six stages.
        "text_state": view.get("text_state"),
        "label_text": ((view["readout"].text if view.get("readout")
                        is not None else None),
                       (view["banner"].text if view.get("banner")
                        is not None else None)),
        "running": bool(F.get("running")),
        "has_timer": view.get("timer") is not None,
        "crashed": bool(F.get("crashed")),
        # the two side panels: their state travels with the frame, and the
        # widgets whose values a keyboard is allowed to fall out of step with
        "gauge_state": view.get("gauge_state"),
        # the two panels' MARKUP. It is written once and must never be
        # written again: a ui.html re-assignment is innerHTML, which is what
        # the HUD cost 497.6 kB/s before it was split into skeleton + state.
        "panel_html": (view["panel"].content
                       if view.get("panel") is not None else None),
        "thrust_band": _thrust_band(),
        # the LIVE thrust label: its element must exist (the browser half
        # addresses it by id) and must stay EMPTY of frame writes, for the
        # same reason the read-out does
        "thrust_label": (view["thrust"].text if view.get("thrust")
                         is not None else None),
        "lever_values": {k: (v[0].value if isinstance(v, tuple) else None)
                         for k, v in view["fields"].items()},
        "modes": F.get("modes"),
        "manoeuvre": F.get("manoeuvre"),
        "stick": dict(F["stick"])})
