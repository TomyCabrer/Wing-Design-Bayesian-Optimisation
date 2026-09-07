"""The panel left of the picture: attitude, rates and the stick.

:mod:`gui.v4.hud`'s trick in a smaller box — a skeleton drawn ONCE and numbers
pushed at it — for the same reason: it changes every frame, and a nicegui
element update once a frame re-patches a page carrying all six stages and
costs a quarter of the frame (the measurement is in :mod:`gui.v4.live`).

WHAT IS ON IT, AND WHY IT IS NOT ON THE GLASS. The head-up display shows the
FLYING numbers — speed, height, heading, the ladder, alpha, g, thrust. This
panel shows the three attitude ANGLES with their three RATES side by side,
which the glass cannot: a pitch ladder says what the attitude is and says
nothing about which way it is going, and on a design with a divergent spiral
(:mod:`gui.v4.modes`) "ten degrees of bank" and "ten degrees of bank opening
at two degrees a second" are the whole difference. Under them are the three
surface angles the keyboard is demanding, which cannot live on their Quasar
sliders for the reason above.

The bank DIAL stays a picture rather than becoming a fourth number, because
roll is the axis a design review loses first and a tilted bar reads at a
glance where a signed number does not.

Every decision about format, colour and rotation is made in Python; the
browser half takes ``{"text": {id: string}, "attr": {id: {name: value}}}``
and knows nothing about a bank angle. One generic function therefore serves
this panel and any future one, with the node cache kept per ROOT so two
panels in one page never see each other's ids.
"""

from __future__ import annotations

import math

from gui.v3 import theme

__all__ = ["attitude_skeleton", "attitude_state", "APPLY_JS",
           "PANEL_W", "PANEL_H", "RATE_FULL_DPS", "AXES", "STICK_AXES",
           "RATE_FONT", "RATE_EM", "RATE_COL_W"]

PANEL_W = 240.0

#: the font stack, RE-QUOTED for an SVG attribute. ``theme.MONO`` carries
#: ``"SF Mono"`` in double quotes and the attribute is written in double
#: quotes, so interpolating it raw closed the attribute early and the rest of
#: the stack was parsed as bare attributes on the ``<svg>`` tag.
_MONO = theme.MONO.replace('"', "'")
_INK = theme.INK
_MUTED = theme.INK_MUTED
_FAINT = theme.INK_FAINT
_RULE = theme.RULE_SOFT
_ACCENT = theme.ACCENT
_BAD = theme.BAD

# ---- the dial
_CX, _CY, _R = PANEL_W / 2.0, 100.0, 68.0

# ---- the three attitude rows
#: (key, label). Roll first because the dial above it is roll.
AXES = (("roll", "ROLL"), ("pitch", "PITCH"), ("yaw", "YAW"))
_ATT_CAP_Y = 182.0
_ATT_TOP = 198.0
_ATT_PITCH = 26.0
_ANG_X = 100.0                      # right edge of the angle column
_RB_X0, _RB_X1 = 108.0, 172.0       # the rate bar
_RB_MID = 0.5 * (_RB_X0 + _RB_X1)
_RATE_X = PANEL_W - 12.0            # right edge of the rate column

#: rate that fills a bar to one end [deg/s], PER AXIS. Roll rates are several
#: times the other two on anything with ailerons, so one shared scale would
#: leave the pitch and yaw bars permanently dead. Past the end the bar pins
#: and reddens and the NUMBER carries the excess — a bar that runs off its
#: track has stopped meaning anything.
RATE_FULL_DPS = {"roll": 60.0, "pitch": 30.0, "yaw": 30.0}

#: THE RATE IS NOT A FOOTNOTE. It was set at 12 against the angle's 17, in
#: the narrowest column on the panel, and a six-character rate (``-103.0``)
#: is 43 px wide in a 32 px gap — so it ran back under the bar it belongs
#: to. The bar gave up 24 px and the number went up two points; the widest
#: string this can hold is checked against the column in a test rather than
#: eyeballed.
RATE_FONT = 14.0
#: how wide a monospace glyph is at ``RATE_FONT``, for that check. SVG's own
#: advance for the ui-monospace stack is 0.6 em and every fallback in
#: :data:`gui.v3.theme.MONO` is a 0.6-em monospace.
RATE_EM = 0.6

# ---- the three surfaces
_STICK_CAP_Y = _ATT_TOP + _ATT_PITCH * len(AXES) + 6.0
_STICK_TOP = _STICK_CAP_Y + 18.0
_STICK_PITCH = 20.0
STICK_AXES = ("elevator", "aileron", "rudder")
STICK_LABEL = {"elevator": "ELEV", "aileron": "AIL", "rudder": "RUD"}
_STK_X0, _STK_X1 = 92.0, 188.0
_STK_MID = 0.5 * (_STK_X0 + _STK_X1)

#: the space the rate number has to itself: from the end of the bar to its
#: own right edge. Derived, so moving either end cannot silently squeeze it.
RATE_COL_W = _RATE_X - _RB_X1

#: THE HEIGHT IS DERIVED, NOT TYPED. It was once a typed 250 while the last
#: caption sat at y = 255, so that row was outside the viewBox and simply not
#: drawn — invisible to every test, because the skeleton string and the state
#: dict were both perfectly correct. A screenshot caught it. The box is now
#: the end of the layout chain plus a margin, and a test reads every
#: coordinate back out of the shipped markup.
PANEL_H = _STICK_TOP + _STICK_PITCH * len(STICK_AXES) + 4.0


def _tick(deg: float, *, major: bool) -> str:
    a = math.radians(deg)
    length = 13.0 if major else 8.0
    x0, y0 = _CX + _R * math.sin(a), _CY - _R * math.cos(a)
    x1 = _CX + (_R - length) * math.sin(a)
    y1 = _CY - (_R - length) * math.cos(a)
    return (f'<line x1="{x0:.2f}" y1="{y0:.2f}" x2="{x1:.2f}" y2="{y1:.2f}" '
            f'stroke="{_MUTED if major else _RULE}" '
            f'stroke-width="{2 if major else 1.4}"/>')


def _rate_row(y: float, tag: str) -> list:
    """A centred rate bar with its zero mark. Shared by every axis."""
    return [f'<line x1="{_RB_X0:.0f}" y1="{y:.0f}" x2="{_RB_X1:.0f}" '
            f'y2="{y:.0f}" stroke="{_RULE}" stroke-width="7"/>',
            f'<line x1="{_RB_MID:.0f}" y1="{y - 6:.0f}" x2="{_RB_MID:.0f}" '
            f'y2="{y + 6:.0f}" stroke="{_MUTED}" stroke-width="1"/>',
            f'<rect id="g-ratebar-{tag}" x="{_RB_MID:.0f}" '
            f'y="{y - 3.5:.0f}" width="0" height="7" fill="{_ACCENT}"/>']


def attitude_skeleton() -> str:
    """Bank dial, three attitude rows, three surface rows. Drawn once."""
    parts = [f'<svg viewBox="0 0 {PANEL_W:.0f} {PANEL_H:.0f}" '
             f'width="100%" style="display:block" '
             f'font-family="{_MONO}">']
    # the fixed scale and its index
    for d in (-60, -45, -30, -15, 0, 15, 30, 45, 60):
        parts.append(_tick(float(d), major=d % 30 == 0))
    parts.append(f'<polygon points="{_CX:.1f},{_CY - _R + 1:.1f} '
                 f'{_CX - 6:.1f},{_CY - _R - 9:.1f} '
                 f'{_CX + 6:.1f},{_CY - _R - 9:.1f}" fill="{_MUTED}"/>')
    for d, lbl in ((-60, "60"), (-30, "30"), (30, "30"), (60, "60")):
        a = math.radians(d)
        x, y = _CX + (_R + 14) * math.sin(a), _CY - (_R + 14) * math.cos(a)
        parts.append(f'<text x="{x:.1f}" y="{y + 4:.1f}" font-size="11" '
                     f'text-anchor="middle" fill="{_FAINT}">{lbl}</text>')

    # the aircraft: one bar that tilts
    parts.append(f'<g id="g-wings">'
                 f'<line x1="{_CX - 56:.0f}" y1="{_CY:.0f}" '
                 f'x2="{_CX - 11:.0f}" y2="{_CY:.0f}" stroke="{_INK}" '
                 f'stroke-width="3"/>'
                 f'<line x1="{_CX + 11:.0f}" y1="{_CY:.0f}" '
                 f'x2="{_CX + 56:.0f}" y2="{_CY:.0f}" stroke="{_INK}" '
                 f'stroke-width="3"/>'
                 f'<circle cx="{_CX:.0f}" cy="{_CY:.0f}" r="3.5" '
                 f'fill="{_INK}"/>'
                 f'<line x1="{_CX:.0f}" y1="{_CY:.0f}" x2="{_CX:.0f}" '
                 f'y2="{_CY + 13:.0f}" stroke="{_INK}" stroke-width="3"/>'
                 f'</g>')

    # ---- the three attitude rows, IDENTICAL in shape: an angle, a centred
    # rate bar, and the rate. Roll used to be a big number and the other two
    # were not on the panel at all.
    parts.append(f'<text x="{_ANG_X:.0f}" y="{_ATT_CAP_Y:.0f}" '
                 f'font-size="9" text-anchor="end" fill="{_FAINT}">deg</text>')
    parts.append(f'<text x="{_RATE_X:.0f}" y="{_ATT_CAP_Y:.0f}" '
                 f'font-size="9" text-anchor="end" fill="{_FAINT}">deg/s'
                 f'</text>')
    for i, (key, lbl) in enumerate(AXES):
        y = _ATT_TOP + i * _ATT_PITCH
        parts.append(f'<text x="12" y="{y + 5:.0f}" font-size="10" '
                     f'fill="{_FAINT}">{lbl}</text>')
        parts.append(f'<text id="g-ang-{key}" x="{_ANG_X:.0f}" '
                     f'y="{y + 6:.0f}" font-size="17" text-anchor="end" '
                     f'fill="{_INK}">+0.0</text>')
        parts.extend(_rate_row(y, key))
        parts.append(f'<text id="g-rate-{key}" x="{_RATE_X:.0f}" '
                     f'y="{y + 5:.0f}" font-size="{RATE_FONT:.0f}" '
                     f'text-anchor="end" fill="{_INK}">+0.0</text>')

    # ---- the three surfaces the keyboard is demanding
    parts.append(f'<line x1="12" y1="{_STICK_CAP_Y - 10:.0f}" '
                 f'x2="{PANEL_W - 12:.0f}" y2="{_STICK_CAP_Y - 10:.0f}" '
                 f'stroke="{_RULE}"/>')
    parts.append(f'<text x="12" y="{_STICK_CAP_Y:.0f}" font-size="9" '
                 f'fill="{_FAINT}">STICK</text>')
    parts.append(f'<text x="{_RATE_X:.0f}" y="{_STICK_CAP_Y:.0f}" '
                 f'font-size="9" text-anchor="end" fill="{_FAINT}">deg'
                 f'</text>')
    for i, axis in enumerate(STICK_AXES):
        y = _STICK_TOP + i * _STICK_PITCH
        parts.append(f'<text x="12" y="{y + 4:.0f}" font-size="10" '
                     f'fill="{_FAINT}">{STICK_LABEL[axis]}</text>')
        parts.append(f'<line x1="{_STK_X0:.0f}" y1="{y:.0f}" '
                     f'x2="{_STK_X1:.0f}" y2="{y:.0f}" stroke="{_RULE}" '
                     f'stroke-width="6"/>')
        parts.append(f'<line x1="{_STK_MID:.0f}" y1="{y - 6:.0f}" '
                     f'x2="{_STK_MID:.0f}" y2="{y + 6:.0f}" '
                     f'stroke="{_MUTED}" stroke-width="1"/>')
        parts.append(f'<rect id="g-stbar-{axis}" x="{_STK_MID:.0f}" '
                     f'y="{y - 3:.0f}" width="0" height="6" '
                     f'fill="{_ACCENT}"/>')
        parts.append(f'<text id="g-st-{axis}" x="{_RATE_X:.0f}" '
                     f'y="{y + 4:.0f}" font-size="12" text-anchor="end" '
                     f'fill="{_INK}">+0.0</text>')
    parts.append("</svg>")
    return "".join(parts)


def _bar(frac: float, x0: float, x1: float) -> tuple:
    """``(x, width)`` for a bar growing either side of the mid-point."""
    mid = 0.5 * (x0 + x1)
    w = abs(frac) * (x1 - mid)
    return (mid if frac >= 0 else mid - w), w


def attitude_state(*, roll_deg: float, pitch_deg: float, yaw_deg: float,
                   rates_dps, stick=None, limits=None) -> dict:
    """Everything that moves on the panel.

    ``rates_dps`` is ``(p, q, r)`` in degrees a second — body rates, which is
    what the equations integrate, and NOT the derivatives of the three Euler
    angles beside them. They agree at level flight and diverge with bank, and
    the row labels say ``deg/s`` against the body axis on purpose: p is what
    the roll damping acts on.
    """
    ang = {"roll": float(roll_deg), "pitch": float(pitch_deg),
           "yaw": float(yaw_deg) % 360.0}
    rate = dict(zip(("roll", "pitch", "yaw"),
                    (float(v) for v in rates_dps)))
    text, attr = {}, {}
    for key, _lbl in AXES:
        text[f"g-ang-{key}"] = f"{ang[key]:+.1f}" if key != "yaw" \
            else f"{ang[key]:5.1f}"
        r = rate[key]
        text[f"g-rate-{key}"] = f"{r:+.1f}"
        full = RATE_FULL_DPS[key]
        f = max(-1.0, min(1.0, r / full)) if full > 0 else 0.0
        x, w = _bar(f, _RB_X0, _RB_X1)
        attr[f"g-ratebar-{key}"] = {
            "x": f"{x:.2f}", "width": f"{w:.2f}",
            "fill": _BAD if abs(f) >= 1.0 else _ACCENT}

    st, lim = dict(stick or {}), dict(limits or {})
    for axis in STICK_AXES:
        d = float(st.get(axis, 0.0))
        text[f"g-st-{axis}"] = f"{d:+.1f}"
        # each bar is scaled by its OWN stop, so full aileron and full
        # elevator look the same length even though they are 25 and 20
        # degrees. A shared scale would say the elevator is doing less.
        lo = abs(float(lim.get(axis) or 0.0))
        f = 0.0 if lo <= 0 else max(-1.0, min(1.0, d / lo))
        x, w = _bar(f, _STK_X0, _STK_X1)
        attr[f"g-stbar-{axis}"] = {"x": f"{x:.2f}", "width": f"{w:.2f}"}

    # positive roll is RIGHT WING DOWN, and an SVG rotate by a positive angle
    # turns clockwise on a y-down canvas, so the sign goes straight through.
    # The HUD's ladder rotates by MINUS the same number because there it is
    # the world that banks.
    attr["g-wings"] = {"transform":
                       f"rotate({ang['roll']:.2f} {_CX:.0f} {_CY:.0f})"}
    return {"text": text, "attr": attr}


#: The browser half — generic, because every decision was made in Python.
#: Node lookups and last-written values are cached on the ROOT element, so
#: two panels never share a cache and a frame costs a handful of string
#: compares.
APPLY_JS = r"""
function(root, s) {
  if (!root || !s) return;
  let c = root.__gaugeCache;
  if (!c) c = root.__gaugeCache = {node: {}, last: {}};
  const at = (id) => (id in c.node) ? c.node[id]
    : (c.node[id] = root.querySelector('#' + id));
  if (s.text) for (const id in s.text) {
    const v = s.text[id], k = 't:' + id;
    if (c.last[k] === v) continue;
    c.last[k] = v;
    const el = at(id); if (el) el.textContent = v;
  }
  if (s.attr) for (const id in s.attr) {
    const a = s.attr[id];
    for (const name in a) {
      const v = a[name], k = 'a:' + id + ':' + name;
      if (c.last[k] === v) continue;
      c.last[k] = v;
      const el = at(id); if (el) el.setAttribute(name, v);
    }
  }
}
"""
