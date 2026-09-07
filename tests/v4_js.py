"""Run gui/v4's browser half under node, so it can be TESTED and not grepped.

The client-side driver (:mod:`gui.v4.live`) and the HUD's apply function
(:mod:`gui.v4.hud`) are the only parts of the render path Python never
executes, and they are where the interesting mistakes live: a smoothing law,
a floating-origin re-anchor and a hundred DOM writes a second. Asserting that
the SOURCE contains the word "reanchor" is not a test of any of that — a
mutation that returns early from the function keeps the word.

So this module stands up just enough of a browser — a Vector3, a Quaternion,
a scene with an object map, and a DOM that records every write — runs the
real scripts against it under node, and hands the result back as JSON.

The quaternion's ``slerp`` here is a normalised linear blend rather than
three.js's. That is deliberate and it is the limit of what this harness
claims: it pins that the attitude is interpolated toward the report and
converges, not that the interpolation follows a great circle.
"""

from __future__ import annotations

import json
import shutil
import subprocess

NODE = shutil.which("node")

_STUB = r"""
globalThis.window = globalThis;
globalThis.__raf = [];
globalThis.requestAnimationFrame = (fn) => globalThis.__raf.push(fn);

class V3 {
  constructor(x = 0, y = 0, z = 0) { this.x = x; this.y = y; this.z = z; }
  clone() { return new V3(this.x, this.y, this.z); }
  set(x, y, z) { this.x = x; this.y = y; this.z = z; return this; }
  applyQuaternion(q) {
    const { x, y, z } = this, qx = q.x, qy = q.y, qz = q.z, qw = q.w;
    const ix = qw * x + qy * z - qz * y, iy = qw * y + qz * x - qx * z;
    const iz = qw * z + qx * y - qy * x, iw = -qx * x - qy * y - qz * z;
    this.x = ix * qw + iw * -qx + iy * -qz - iz * -qy;
    this.y = iy * qw + iw * -qy + iz * -qx - ix * -qz;
    this.z = iz * qw + iw * -qz + ix * -qy - iy * -qx;
    return this;
  }
}
class Q {
  constructor(x = 0, y = 0, z = 0, w = 1) {
    this.x = x; this.y = y; this.z = z; this.w = w;
  }
  clone() { return new Q(this.x, this.y, this.z, this.w); }
  set(x, y, z, w) { this.x = x; this.y = y; this.z = z; this.w = w; return this; }
  copy(q) { return this.set(q.x, q.y, q.z, q.w); }
  slerp(q, t) {
    const d = this.x * q.x + this.y * q.y + this.z * q.z + this.w * q.w;
    const s = d < 0 ? -1 : 1;
    this.x += (s * q.x - this.x) * t; this.y += (s * q.y - this.y) * t;
    this.z += (s * q.z - this.z) * t; this.w += (s * q.w - this.w) * t;
    const n = Math.hypot(this.x, this.y, this.z, this.w) || 1;
    this.x /= n; this.y /= n; this.z /= n; this.w /= n;
    return this;
  }
}
const mkObj = () => ({ position: new V3(), quaternion: new Q() });

globalThis.__writes = 0;
const mkNode = (id) => ({
  id, attrs: {}, _text: null,
  // a real element's style is an object that RECORDS what was set on it,
  // because the banner's colour is written the same way its text is
  style: new Proxy({}, {
    set(t, k, v) { globalThis.__writes++; t[k] = v; return true; },
  }),
  setAttribute(k, v) { globalThis.__writes++; this.attrs[k] = v; },
  get textContent() { return this._text; },
  set textContent(v) { globalThis.__writes++; this._text = v; },
});
// every node can be queried into, and each root owns ITS OWN children —
// two builds of the same view must not share a single glass
globalThis.__nodes = new Map();
const kids = (root) => (sel) => {
  const key = root + '>' + sel;
  if (!globalThis.__nodes.has(key)) globalThis.__nodes.set(key, mkNode(key));
  return globalThis.__nodes.get(key);
};

const SCENE = {
  objects: new Map(), camera: mkObj(), look_at: new V3(),
  controls: { enabled: true, target: new V3() }, camera_tween: {},
};
SCENE.camera.lookAt = function (x, y, z) { this.lookedAt = [x, y, z]; };
globalThis.__scene = SCENE;
globalThis.getElement = (id) => (id === CFG.scene ? SCENE : null);
// EVERY id resolves, because a rebuilt view has ids the first build never
// had. The read-out and the banner are plain DOM elements looked up by id,
// and they are what the frame loop writes INSTEAD of a nicegui update.
globalThis.__byId = new Map();
globalThis.byId = (id) => {
  if (!globalThis.__byId.has(id)) {
    const n = mkNode(id);
    n.querySelector = kids(id);
    globalThis.__byId.set(id, n);
  }
  return globalThis.__byId.get(id);
};
globalThis.document = { getElementById: (id) => globalThis.byId(id) };
"""


def run(cfg: dict, script: str, *, live_js: str, apply_js: str,
        gauge_js: str | None = None) -> dict:
    """Execute ``script`` with the real browser half loaded.

    ``script`` may call ``tick(ms)`` and ``push(payload)``, and must end by
    assigning the object it wants back to ``OUT``.

    ``gauge_js`` is the SHIPPED side-panel apply function. It defaults to
    the one gui.v4.gauge exports rather than to a stub, because a harness
    that installs its own would test a function nobody runs.
    """
    if gauge_js is None:
        from gui.v4 import gauge as _gauge
        gauge_js = _gauge.APPLY_JS
    if NODE is None:                                    # pragma: no cover
        raise RuntimeError("node is not installed")
    src = "\n".join([
        f"const CFG = {json.dumps(cfg)};",
        _STUB,
        "SCENE.objects.set(CFG.group, mkObj());",
        "SCENE.objects.set(CFG.world, mkObj());",
        live_js,
        f"const APPLY = {apply_js};",
        f"const GAUGE = {gauge_js};",
        "window.__aerobo.init(CFG, APPLY, GAUGE);",
        "const A = window.__aerobo;",
        "const tick = (ms) => A.tick(ms);",
        "const push = (p) => A.push(p);",
        "let OUT = {};",
        script,
        "console.log(JSON.stringify(OUT));",
    ])
    r = subprocess.run([NODE, "--input-type=module", "-e", src],
                       capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        raise AssertionError(f"node failed:\n{r.stderr}")
    return json.loads(r.stdout.strip().splitlines()[-1])


def _shipped_tau() -> float:
    """The tuning constant the STAGE would install, not a copy of it.

    A harness that hard-codes the number tests a law nobody ships.
    """
    from gui.v4 import live

    return float(live.SMOOTH_TAU_S)


CFG = {"scene": 1, "group": "g-uuid", "world": "w-uuid", "hud": 2,
       "readout": 3, "banner": 4, "panel": 5, "thrust": 6,
       "span": 10.0, "snap": 480.0, "back": 2.6, "up": 0.42, "lead": 0.35,
       "tau": _shipped_tau()}
