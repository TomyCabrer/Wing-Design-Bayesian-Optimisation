"""One message a frame, and the browser does the rest.

WHY THIS EXISTS. Stage 6 used to drive the picture the obvious way: move the
aeroplane, move the ground, move the camera, repaint the glass — four
``run_method`` calls and one 14.8 kB ``ui.html`` update, thirty times a
second. Measured on a page doing nothing else:

    155.0 messages/s and 515.3 kB/s, for ONE aeroplane

and the render loop it was feeding cannot go faster than about 30 Hz anyway
(see :data:`SCENE_FPS`). A simulation pushed frame-by-frame down a websocket
is a flip-book, and it looks like one.

So the server stops animating and starts REPORTING. It sends one small
payload per physics push — a quaternion, a height, a ground offset and the
HUD's twenty numbers, about 300 bytes — and a script installed once in the
page does three things the server cannot:

* **smooths**, so 30 pushes a second become as many frames as the display
  will draw. A first-order filter with a time constant near the push
  interval, on the aircraft's position and (by slerp) its attitude;
* **flies the camera**, computed from the SMOOTHED pose rather than from a
  pose that arrived 30 ms ago, so the chase view is locked to what is
  actually on screen and costs no messages at all;
* **updates the glass** by writing text nodes and transforms into a skeleton
  that is already in the DOM (:mod:`gui.v4.hud`);
* **polls the gamepad** (:mod:`gui.v4.pad`), because the Gamepad API has no
  input event to subscribe to — its readings are only fresh inside an
  animation frame, and this is the only animation frame there is. It reports
  the stick to the server on CHANGE, so a held stick is one message and a
  released one is none: the same economy as the pose push, pointed the other
  way down the socket.

The floating origin is the one thing a smoother must not smooth. The ground
offset is a sawtooth — it jumps a whole tile whenever the aeroplane crosses
one — so the filter is RE-ANCHORED by whole tiles before it interpolates,
which is the same trick the offset itself is built on and is why the world
never slides 480 m over 50 ms.
"""

from __future__ import annotations

import json

from . import pad as _pad

__all__ = ["INSTALL_JS", "push_call", "SCENE_FPS", "SMOOTH_TAU_S"]

#: What ``ui.scene(fps=...)`` really does, from ``scene.js``::
#:
#:     requestAnimationFrame(() => setTimeout(() => render(), 1000 / this.fps));
#:
#: It is a delay imposed AFTER the animation frame, not a target rate, and
#: the two compose in series. At ``fps=60`` a frame costs one animation frame
#: to arrive (16.7 ms on a 60 Hz display) plus a 16.7 ms timeout before the
#: next one is even requested — about 33 ms, i.e. 30 Hz, for a number that
#: says 60. NiceGUI's default of 20 gives 16.7 + 50 = 15 Hz.
#:
#: So the number is set high enough that the added delay is noise and the
#: display's own refresh governs, which is the only rate worth drawing at.
SCENE_FPS = 1000

#: time constant of the pose filter [s]. Near the push interval: shorter and
#: the flip-book shows through, longer and the aeroplane lags the stick.
SMOOTH_TAU_S = 0.045

_JS = r"""
window.__aerobo = window.__aerobo || (function () {
  const A = {
    cfg: null, pending: null, scene: null, group: null, world: null,
    hud: null, applyHud: null, applyGauge: null, raf: null, last: null,
    // the two lines of numbers under the picture, and what they last said
    txt: {}, els: {},
    // the smoothed state the screen is actually showing
    now: {z: 0, w: [0, 0, 0], q: null}, ready: false,
    // the gamepad's last REPORTED demand, the buttons it was holding, and
    // when it last spoke. `h: null` means "has never reported", which is a
    // different thing from "reported zeros" and is why the first report of
    // a centred stick still goes out.
    pad: {on: false, h: null, b: null, t: null},
  };

  A.init = function (cfg, applyHud, applyGauge) {
    A.cfg = cfg;
    A.applyHud = applyHud;
    A.applyGauge = applyGauge;
    // A VIEW IS BUILT MORE THAN ONCE, AND EVERY HANDLE HAS TO BE DROPPED.
    //
    // This object is created once per PAGE (`window.__aerobo || ...`) but
    // installed once per BUILT VIEW, and the shell builds the fly view at
    // least twice on the way to it: once when the page lays out all six
    // stages, and again when the pilot clicks into stage 6. Everything
    // resolve() found the first time — the scene component, the two scene
    // objects, the glass, the read-out — belongs to a view that has since
    // been cleared.
    //
    // Keeping them cost the whole picture and gave no error to find it by:
    // the group and the world still moved, so the numbers read correctly
    // from here, but they were objects in a destroyed scene. On screen the
    // camera never left its default (the aeroplane filled the frame from a
    // few centimetres away), no ground appeared, the glass stayed blank and
    // the read-out under it stayed empty. So a re-install drops EVERY
    // resolved handle and starts again.
    A.ready = false;
    A.scene = A.group = A.world = A.hud = null;
    A.els = {}; A.txt = {};
    // `last = null` is the load-bearing one: it makes the next tick a FIRST
    // frame, which snaps. Clearing `now` with it is belt and braces — the
    // snap already overwrites z and w, and resolve() re-reads the attitude
    // off the new group — so no test can tell the two apart.
    A.now = {z: 0, w: [0, 0, 0], q: null};
    A.last = null;
    // ...and the pad forgets what it was holding, so the rebuilt view is
    // told the demand again rather than inheriting a report the server it
    // is talking to never received.
    A.pad = {on: false, h: null, b: null, t: null};
    A.wireRelease();
    if (!A.raf) A.raf = requestAnimationFrame(A.tick);
  };

  // A KEY THE BROWSER NEVER TELLS US WAS RELEASED.
  //
  // nicegui's keyboard binds ONE listener on `document` and drops the event
  // outright when `document.activeElement` is an input, a select, a button
  // or a textarea. So the keyup that ends a roll input is swallowed the
  // moment the pilot clicks into the max-thrust box beside the picture, and
  // the axis stays at full demand for ever — the aeroplane goes on rolling
  // with nothing held. cmd-tabbing away is the same hole from the other
  // side: macOS delivers no keyup at all to a window that lost focus.
  //
  // A watchdog cannot fix it, because a held key sends ONE keydown and no
  // repeats (`repeating=false`), so "no news for two seconds" is exactly
  // what a correctly held stick looks like. What CAN be observed is the
  // moment the keyboard stops being ours: focus leaving the page, the tab
  // being hidden, or focus landing on a widget that eats keys. Each of
  // those releases everything, which is what a pilot letting go of the
  // keyboard means.
  A.wireRelease = function () {
    if (A.released) return;
    const go = function () {
      if (typeof emitEvent === 'function') emitEvent('aerobo_release', {});
    };
    A.emitRelease = go;
    // FEATURE-TESTED, not environment-guessed. The page's own test harness
    // runs this whole script under node against a stub `window`, so
    // `typeof window !== 'undefined'` is true there and
    // `window.addEventListener` is not a function — the browser half is run
    // rather than grepped, and this is what that buys.
    const on = function (target, name, fn) {
      if (target && typeof target.addEventListener === 'function') {
        target.addEventListener(name, fn);
        return true;
      }
      return false;
    };
    const w = (typeof window === 'undefined') ? null : window;
    const d = (typeof document === 'undefined') ? null : document;
    let wired = on(w, 'blur', go);
    wired = on(d, 'visibilitychange', function () {
      if (d.hidden) go();
    }) || wired;
    wired = on(d, 'focusin', function (e) {
      const t = ((e && e.target && e.target.tagName) || '').toUpperCase();
      if (t === 'INPUT' || t === 'TEXTAREA' || t === 'SELECT'
          || t === 'BUTTON') go();
    }) || wired;
    A.released = wired;
  };

  A.resolve = function () {
    if (A.ready) return true;
    const sc = (typeof getElement === 'function')
      ? getElement(A.cfg.scene) : null;
    // the scene mounts asynchronously (it dynamically imports three.js), so
    // everything here is retried until it is actually there
    if (!sc || !sc.objects || !sc.camera) return false;
    const g = sc.objects.get(A.cfg.group);
    const w = sc.objects.get(A.cfg.world);
    if (!g || !w) return false;
    A.scene = sc; A.group = g; A.world = w;
    A.hud = document.getElementById('c' + A.cfg.hud);
    A.now.q = g.quaternion.clone();
    A.ready = true;
    return true;
  };

  A.push = function (p) { A.pending = p; };

  // An element looked up on FIRST USE, not at install: the read-out labels
  // are created after the scene, and a null is retried rather than cached.
  A.el = function (name) {
    if (!A.els[name]) A.els[name] = document.getElementById('c' + A.cfg[name]);
    return A.els[name];
  };

  // THE READ-OUT IS TEXT, AND TEXT MUST NOT GO THROUGH VUE.
  //
  // These two lines used to be written with `label.text = ...` once a frame.
  // That is a nicegui element update, and an element update re-patches a
  // page carrying all six stages at once. Measured in a real visible
  // Chrome, flying, by dropping one message kind at socket.onevent:
  //
  //   nothing dropped        97.7 Hz   p90  9.4   p99 33.8   max 41.5 ms
  //   drop the updates      119.2 Hz   p90  9.2   p99  9.4
  //   drop run_javascript    98.3 Hz   p90 16.5   p99 33.6
  //   drop both             120.0 Hz   p90  9.0   p99  9.3   max  9.4 ms
  //
  // ~10 updates a second (they coalesce) were costing a quarter of every
  // frame, while the 40 pose pushes a second cost nothing measurable. So
  // the numbers are written straight into the DOM, guarded on change, and
  // the frame loop sends no element update at all.
  A.writeText = function (t) {
    const c = A.txt;
    if (t.readout !== undefined && c.readout !== t.readout) {
      c.readout = t.readout;
      const el = A.el('readout');
      if (el) el.textContent = t.readout;
    }
    if (t.banner !== undefined && c.banner !== t.banner) {
      c.banner = t.banner;
      const el = A.el('banner');
      if (el) el.textContent = t.banner;
    }
    if (t.color !== undefined && c.color !== t.color) {
      c.color = t.color;
      const el = A.el('banner');
      if (el) el.style.color = t.color;
    }
    // the throttle, on the panel that owns its LIMIT — the value W and S are
    // moving, and the band they run between. Same route and the same reason:
    // both change without the panel being rebuilt, and a Quasar widget
    // cannot follow a held key without one element update a frame.
    if (t.thrust !== undefined && c.thrust !== t.thrust) {
      c.thrust = t.thrust;
      const el = A.el('thrust');
      if (el) el.textContent = t.thrust;
    }
  };

  // the ground offset is a SAWTOOTH: it jumps a whole tile when the
  // aeroplane crosses one. Re-anchor the filter's current value by whole
  // tiles before interpolating, or the world slides a tile over one time
  // constant every time the origin re-snaps.
  const reanchor = function (cur, tgt, period) {
    if (!period) return cur;
    let d = tgt - cur;
    while (d > period / 2) { cur += period; d -= period; }
    while (d < -period / 2) { cur -= period; d += period; }
    return cur;
  };

  // ------------------------------------------------------------- the pad
  // A GAMEPAD IS POLLED, NOT LISTENED TO. The Gamepad API has no input
  // event: `navigator.getGamepads()` hands back a SNAPSHOT that is only
  // refreshed inside an animation frame. So the poll lives in the render
  // tick that is already running, and costs no timer of its own.
  //
  // It reports on CHANGE and not on a clock. A stick held at 40 % is one
  // message; a stick at rest is none. That is the same economy the pose
  // push is built on, pointed the other way down the socket — and it is
  // why the quantum exists: it makes "changed" a decidable question.
  //
  // The arithmetic is NOT written twice. Every constant here arrives in
  // `cfg.pad` from gui/v4/pad.py, and tests/test_v4_pad.py runs this very
  // script under node against that module's own functions.
  A.padCurve = function (raw) {
    const p = A.cfg.pad, v = Math.min(1, Math.abs(raw));
    if (v <= p.dead) return 0;
    return Math.sign(raw) * Math.pow((v - p.dead) / (1 - p.dead), p.expo);
  };

  A.padTrigger = function (raw) {
    const p = A.cfg.pad, v = Math.min(1, Math.max(0, raw));
    return v <= p.tdead ? 0 : (v - p.tdead) / (1 - p.tdead);
  };

  // ROUNDING HAS TO MATCH PYTHON'S, not JavaScript's. Math.round sends a
  // half UP (Math.round(-0.5) === -0) while Python's round() sends it to
  // the EVEN quantum, and a disagreement of one quantum is a disagreement
  // the cross-check test would report as a broken curve.
  A.padQ = function (v) {
    const q = A.cfg.pad.q;
    if (!(q > 0)) return v;
    const n = Math.abs(v) / q, f = Math.floor(n), r = n - f;
    const k = r > 0.5 ? f + 1 : (r < 0.5 ? f : (f % 2 === 0 ? f : f + 1));
    return Math.sign(v) * k * q;
  };

  A.emitPad = function (p) {
    if (typeof emitEvent === 'function') emitEvent('aerobo_pad', p);
  };

  A.pollPad = function (t) {
    const cfg = A.cfg.pad;
    if (!cfg || typeof navigator === 'undefined' ||
        typeof navigator.getGamepads !== 'function') return;
    const pads = navigator.getGamepads() || [];
    let gp = null;
    for (let i = 0; i < pads.length; i++) {
      if (pads[i] && pads[i].connected) { gp = pads[i]; break; }
    }
    // A PAD THAT WENT AWAY MUST LET GO OF THE STICK. Unplugged mid-roll,
    // the last demand would otherwise stand for ever — the same shape of
    // defect as a keyup landing on a stopped loop, which stage 6 has been
    // bitten by once already.
    if (!gp) {
      if (A.pad.on) {
        A.pad = {on: false, h: null, b: null, t: null};
        A.emitPad({h: {}, a: [], on: false});
      }
      return;
    }
    const ax = gp.axes || [];
    const bs = (gp.buttons || []).map(function (b) {
      return (b && typeof b.value === 'number') ? b.value : (b ? 1 : 0);
    });
    const h = {};
    for (const i in cfg.axes) {
      const spec = cfg.axes[i];
      h[spec[0]] = A.padQ(spec[1] * A.padCurve(ax[i] || 0));
    }
    h.thrust = A.padQ(A.padTrigger(bs[cfg.up] || 0)
                      - A.padTrigger(bs[cfg.down] || 0));
    // EDGES ARE NOT SUBJECT TO THE RATE LIMIT. A press is one instant and
    // the limiter is a window; a cross that starts the flight must not be
    // eaten because a stick moved 8 ms ago.
    const acts = [], nb = {};
    for (const i in cfg.buttons) {
      const now = (bs[i] || 0) >= cfg.press;
      if (now && !(A.pad.b && A.pad.b[i])) acts.push(cfg.buttons[i]);
      nb[i] = now;
    }
    A.pad.b = nb;
    let moved = !A.pad.on || !A.pad.h;
    if (!moved) {
      for (const k in h) if (h[k] !== A.pad.h[k]) { moved = true; break; }
    }
    const due = A.pad.t === null || (t - A.pad.t) / 1000 >= 1 / cfg.hz;
    if (!acts.length && !(moved && due)) return;
    A.pad.on = true;
    if (moved && due) { A.pad.h = h; A.pad.t = t; }
    A.emitPad({h: h, a: acts, on: true});
  };

  A.tick = function (t) {
    A.raf = requestAnimationFrame(A.tick);
    // FIRST FRAME is not the same thing as NO TIME PASSED, and conflating
    // them is a real bug: the first frame after arming (or after a resume)
    // must SNAP, so the aeroplane appears at the pose that was reported
    // rather than sliding in from the origin — but two ticks landing in the
    // same instant must move nothing at all.
    const first = A.last === null;
    // clamped at BOTH ends. The upper clamp is the tab that was in the
    // background for a minute; the lower one is that a negative dt makes
    // exp(-dt/tau) overflow and the filter gain go to -Infinity, which puts
    // the aeroplane somewhere no coordinate can name.
    const dt = first ? 0
      : Math.max(0, Math.min((t - A.last) / 1000, 0.25));
    A.last = t;
    // before resolve(), and before the "nothing pending" return below: the
    // cross that STARTS the flight is pressed when nothing is flying, and a
    // pad polled only once there is a pose to draw could never send it.
    A.pollPad(t);
    if (!A.resolve()) return;
    const p = A.pending;
    if (!p) return;

    // ---- smooth toward the last reported pose
    const k = first ? 1 : 1 - Math.exp(-dt / A.cfg.tau);
    const n = A.now;
    n.z += (p.z - n.z) * k;
    const per = A.cfg.snap;
    for (let i = 0; i < 3; i++) {
      // z of the ground is the altitude datum and does not wrap
      n.w[i] = i < 2 ? reanchor(n.w[i], p.w[i], per) : n.w[i];
      n.w[i] += (p.w[i] - n.w[i]) * k;
    }
    if (p.q) {
      const tq = A.group.quaternion.clone();
      tq.set(p.q[0], p.q[1], p.q[2], p.q[3]);
      n.q.slerp(tq, k);
      A.group.quaternion.copy(n.q);
    }
    A.group.position.set(0, 0, n.z);
    A.world.position.set(n.w[0], n.w[1], n.w[2]);

    // ---- the camera, from the pose ON SCREEN rather than the one that
    // arrived. In the engineering view the mouse owns the camera instead.
    if (p.game) {
      if (A.scene.controls) A.scene.controls.enabled = false;
      A.scene.camera_tween = null;
      const c = A.cfg, b = c.span;
      const v = A.group.position.clone();
      const nose = v.clone().set(1, 0, 0).applyQuaternion(n.q);
      const up = v.clone().set(0, 0, -1).applyQuaternion(n.q);
      const ex = -c.back * b * nose.x + c.up * b * up.x;
      const ey = -c.back * b * nose.y + c.up * b * up.y;
      const ez = n.z - c.back * b * nose.z + c.up * b * up.z;
      const tx = c.lead * b * nose.x;
      const ty = c.lead * b * nose.y;
      const tz = n.z + c.lead * b * nose.z;
      A.scene.camera.position.set(ex, ey, ez);
      A.scene.camera.lookAt(tx, ty, tz);
      if (A.scene.look_at) A.scene.look_at.set(tx, ty, tz);
      if (A.scene.controls) A.scene.controls.target.set(tx, ty, tz);
    } else if (A.scene.controls && !A.scene.controls.enabled) {
      A.scene.controls.enabled = true;
    }

    // ---- the glass
    if (p.hud && A.applyHud) A.applyHud(A.hud, p.hud);
    // ---- ...the attitude panel, through a GENERIC apply. Its state is a
    // {text, attr} map built in gui/v4/gauge.py, so nothing about a bank
    // angle or a rate scale is decided here — and the same function serves
    // any second panel, with its node cache kept on the root element.
    if (p.g && A.applyGauge) A.applyGauge(A.el('panel'), p.g);
    // ---- ...and the numbers under it
    if (p.txt) A.writeText(p.txt);
  };

  return A;
})();
"""


def INSTALL_JS(*, scene_id: int, group_id: str, world_id: str, hud_id: int,
               readout_id: int, banner_id: int, panel_id: int,
               thrust_id: int,
               span_m: float, snap_m: float, back: float, up: float,
               lead: float, apply_js: str, gauge_js: str,
               tau_s: float = SMOOTH_TAU_S,
               pad_cfg: dict | None = None) -> str:
    """The whole browser half, as one script. Sent once per built view.

    THE TWO KINDS OF ID. ``scene_id``, ``hud_id``, ``readout_id``,
    ``banner_id``, ``panel_id`` and ``thrust_id`` are nicegui ELEMENT ids —
    integers, and
    what ``getElement`` and ``'c' + id`` take. ``group_id``
    and ``world_id`` are SCENE-OBJECT ids, which are uuid strings and key
    the scene's own ``objects`` map. Coercing either to the other's type
    silently addresses nothing, and the picture simply never moves.

    ``pad_cfg`` is :func:`gui.v4.pad.js_config` — the gamepad's dead zone,
    curve, signs and wire quantum. It is SENT rather than written into the
    script because a constant written twice is a constant that gets changed
    once; omit it (``None`` takes the shipped one) and pass ``{}`` only to
    build a driver with no pad at all, which is what the pose tests do.
    """
    cfg = {"scene": int(scene_id), "group": str(group_id),
           "world": str(world_id), "hud": int(hud_id),
           "readout": int(readout_id), "banner": int(banner_id),
           "panel": int(panel_id), "thrust": int(thrust_id),
           "span": float(span_m), "snap": float(snap_m),
           "back": float(back), "up": float(up), "lead": float(lead),
           "tau": float(tau_s),
           "pad": _pad.js_config() if pad_cfg is None else dict(pad_cfg)}
    return (f"{_JS}\nwindow.__aerobo.init("
            + json.dumps(cfg, separators=(",", ":"))
            + f", {apply_js}, {gauge_js});")


def push_call(payload: dict) -> str:
    """One frame, as the smallest thing that can carry it.

    ``separators`` matters here rather than being a style choice: it is about
    12 % of the payload, every frame, for ever.
    """
    return ("window.__aerobo&&window.__aerobo.push("
            + json.dumps(payload, separators=(",", ":")) + ")")
