"""The head-up display: a skeleton drawn ONCE, and numbers pushed at it.

WHY THIS IS NOT A STRING ANY MORE. The first version rebuilt the whole glass
every frame and handed it to ``ui.html``, which assigns ``innerHTML``. That
was measured, on a page doing nothing but flying:

    with the HUD      155.0 messages/s      515.3 kB/s
    HUD suppressed    124.6 messages/s       17.7 kB/s
    ------------------------------------------------
    the HUD alone      30.4 messages/s      497.6 kB/s   <- 97 % of the wire

14.8 kB of markup and 203 SVG nodes, serialised, sent, parsed and laid out
thirty times a second, to move about a dozen numbers. So the glass is now
split in two:

:func:`skeleton`
    every node the HUD will ever have, with an ``id`` on the ones that move.
    Sent once, when the view is built.
:func:`state`
    the ~20 floats that change, with every clamp, wrap and threshold already
    applied ON THIS SIDE. Roughly 300 bytes, and it is the only thing that
    travels per frame.
:data:`APPLY_JS`
    the browser-side half: it maps a :func:`state` dict onto the skeleton by
    setting transforms and text nodes. No parsing, no layout thrash, and it
    can run at the display's own rate rather than at the server's.

Both halves are pure — one takes numbers and returns a string, the other
takes numbers and returns a dict — so every symbol on the glass is still
checkable by asserting on data rather than by looking at a screenshot.

THE TAPES ARE PERIODIC, WHICH IS WHY THIS WORKS. A rolling tape's tick marks
are identical every ``step``, so the marks never need redrawing: the group is
translated by the fractional part of ``value / step`` and only the LABELS'
text changes. The same trick carries the pitch ladder, which is one group
with one rotate and one translate.
"""

from __future__ import annotations

import math

__all__ = ["skeleton", "state", "APPLY_JS", "VIEW_W", "VIEW_H",
           "IAS_STEP", "ALT_STEP"]

#: the SVG's own coordinate system. Fixed, so every number below is in
#: "HUD units" and the display scales with its container rather than with
#: the window.
VIEW_W = 1000.0
VIEW_H = 620.0

#: degrees of pitch per 10 HUD units on the ladder
_PITCH_SCALE = 7.0
#: HUD units of ladder travel per degree of pitch
_PITCH_PX = 10.0 / _PITCH_SCALE
_CX, _CY = VIEW_W / 2.0, VIEW_H / 2.0

#: tape geometry. ``_SPAN`` intervals of ``step`` fill ``_TAPE_H`` units, so
#: a minor tick is _TAPE_H/_SPAN apart and a labelled one twice that.
_TAPE_H = 300.0
_SPAN = 12
_MINOR_PX = _TAPE_H / _SPAN            # 25
_MAJOR_PX = 2.0 * _MINOR_PX            # 50

IAS_STEP = 5.0
ALT_STEP = 25.0

_IAS_X = 150.0
_ALT_X = VIEW_W - 150.0

_GREEN = "#7ef5a0"
_AMBER = "#ffcf5a"
_RED = "#ff6b6b"

#: alpha and load factor past these light up amber
ALPHA_WARN_DEG = 12.0
G_WARN_HI, G_WARN_LO = 3.0, 0.0

#: the throttle column. It USED to sit at VIEW_W - 130, which is INSIDE the
#: altitude tape's box (that tape is 92 units wide about _ALT_X, i.e. 804 to
#: 896, and the bar ran 870 to 896) — so the bar's top and the number above
#: it were drawn under a column full of altitude labels. Shortening the bar
#: only half fixed it: the NUMBER was still in the tape's rows. So the whole
#: column moved LEFT of the tape instead, which is x-disjoint from it and
#: from the pitch ladder, and the bar keeps its full height.
_THR_X = _ALT_X - 46.0 - 60.0
_THR_BOT = VIEW_H - 50.0
_THR_H = 140.0
_THR_TOP = _THR_BOT - _THR_H


def _frac(x: float) -> float:
    return float(x) - math.floor(float(x))


# ------------------------------------------------------------ the skeleton

def _tape_skeleton(x: float, tag: str, side: str, label: str) -> str:
    """One rolling tape: a clipped box, a minor group, a major group with
    its labels, a readout window and a title. Nothing here depends on a
    value — that is the whole point."""
    sgn = 1 if side == "right" else -1
    out = [f'<clipPath id="hud-clip-{tag}"><rect x="{x - 46:.0f}" '
           f'y="{_CY - _TAPE_H / 2:.0f}" width="92" height="{_TAPE_H:.0f}"/>'
           f'</clipPath>',
           f'<rect x="{x - 46:.0f}" y="{_CY - _TAPE_H / 2:.0f}" width="92" '
           f'height="{_TAPE_H:.0f}" fill="#00120a" fill-opacity="0.30" '
           f'stroke="{_GREEN}" stroke-opacity="0.45"/>',
           f'<g clip-path="url(#hud-clip-{tag})">']

    # minor ticks, one _MINOR_PX apart, drawn past the box on both sides so
    # the translation never exposes an end
    rows = []
    for k in range(-8, 9):
        y = _CY + k * _MINOR_PX
        x0 = x + sgn * 46
        rows.append(f'<line x1="{x0:.0f}" y1="{y:.0f}" '
                    f'x2="{x0 - sgn * 16:.0f}" y2="{y:.0f}" '
                    f'stroke="{_GREEN}" stroke-width="2" '
                    f'stroke-opacity="0.8"/>')
    out.append(f'<g id="hud-{tag}-minor">' + "".join(rows) + "</g>")

    # labelled ticks, twice as far apart, each with a text node the browser
    # side rewrites. The label INDEX is fixed; only its number moves.
    rows = []
    for j in range(-4, 5):
        y = _CY + j * _MAJOR_PX
        x0 = x + sgn * 46
        anchor = "end" if side == "right" else "start"
        rows.append(f'<line x1="{x0:.0f}" y1="{y:.0f}" '
                    f'x2="{x0 - sgn * 32:.0f}" y2="{y:.0f}" '
                    f'stroke="{_GREEN}" stroke-width="2"/>'
                    f'<text id="hud-{tag}-l{j + 4}" x="{x0 - sgn * 40:.0f}" '
                    f'y="{y + 5:.0f}" fill="{_GREEN}" font-size="20" '
                    f'text-anchor="{anchor}" fill-opacity="0.75"></text>')
    out.append(f'<g id="hud-{tag}-major">' + "".join(rows) + "</g></g>")

    out.append(f'<rect x="{x - 52:.0f}" y="{_CY - 22:.0f}" width="104" '
               f'height="44" fill="#001b10" stroke="{_GREEN}" '
               f'stroke-width="2"/>'
               f'<text id="hud-{tag}" x="{x:.0f}" y="{_CY + 9:.0f}" '
               f'fill="{_GREEN}" font-size="28" text-anchor="middle"></text>'
               f'<text x="{x:.0f}" y="{_CY - _TAPE_H / 2 - 15:.0f}" '
               f'fill="{_GREEN}" font-size="18" text-anchor="middle" '
               f'fill-opacity="0.8">{label}</text>')
    return "".join(out)


def _ladder_skeleton() -> str:
    """The pitch ladder, drawn at pitch zero and roll zero.

    THE SIGN. A ladder is fixed to the WORLD: pitch the nose up and the
    horizon goes DOWN the screen, because the boresight — which is fixed to
    the aeroplane — has climbed above it. So the line for ``theta`` sits at
    ``_CY + (pitch - theta) * _PITCH_PX``: drawn here at pitch zero, and
    translated by ``+pitch * _PITCH_PX`` per frame.

    The first version had ``(theta - pitch)``, which drew the +10 line BELOW
    the horizon and walked the horizon UP as the nose came up. It is
    invisible in a screenshot of level flight and wrong in every climb.
    """
    rows = []
    for deg in range(-90, 91, 10):
        y = _CY - deg * _PITCH_PX
        if deg == 0:
            rows.append(f'<line x1="{_CX - 300:.0f}" y1="{y:.1f}" '
                        f'x2="{_CX + 300:.0f}" y2="{y:.1f}" '
                        f'stroke="{_GREEN}" stroke-width="3"/>')
            continue
        # a dive line is dashed, a climb line solid — the standard cue for
        # which side of the horizon you are looking at
        w, dash = 110, ("" if deg > 0 else ' stroke-dasharray="12 10"')
        for sgn in (-1, 1):
            x0 = _CX + sgn * 60
            rows.append(f'<line x1="{x0:.0f}" y1="{y:.1f}" '
                        f'x2="{x0 + sgn * w:.0f}" y2="{y:.1f}" '
                        f'stroke="{_GREEN}" stroke-width="2" '
                        f'stroke-opacity="0.85"{dash}/>'
                        f'<text x="{x0 + sgn * (w + 12):.0f}" '
                        f'y="{y + 6:.1f}" fill="{_GREEN}" font-size="18" '
                        f'text-anchor="{"start" if sgn > 0 else "end"}" '
                        f'fill-opacity="0.8">{abs(deg)}</text>')
    # the clip is on an OUTER group with no transform, so it stays fixed to
    # the screen while the ladder inside it rotates
    return (f'<clipPath id="hud-clip-ladder"><rect x="{_CX - 330:.0f}" '
            f'y="{_CY - 230:.0f}" width="660" height="460"/></clipPath>'
            f'<g clip-path="url(#hud-clip-ladder)">'
            f'<g id="hud-ladder">' + "".join(rows) + "</g></g>")


def _bank_skeleton() -> str:
    ticks = []
    for deg in (-60, -45, -30, -20, -10, 0, 10, 20, 30, 45, 60):
        a = math.radians(deg - 90.0)
        r0, r1 = 250.0, 250.0 - (22 if deg % 30 == 0 else 12)
        ticks.append(f'<line x1="{_CX + r0 * math.cos(a):.1f}" '
                     f'y1="{_CY + r0 * math.sin(a):.1f}" '
                     f'x2="{_CX + r1 * math.cos(a):.1f}" '
                     f'y2="{_CY + r1 * math.sin(a):.1f}" stroke="{_GREEN}" '
                     f'stroke-width="2" stroke-opacity="0.7"/>')
    # the pointer is a dot at the top of the arc inside a group that ROTATES,
    # so one transform puts it at any bank angle
    ticks.append(f'<g id="hud-bank"><circle cx="{_CX:.0f}" '
                 f'cy="{_CY - 228:.0f}" r="8" fill="{_GREEN}"/></g>')
    return "".join(ticks)


def skeleton() -> str:
    """Every node the HUD will ever have. Sent once, never again."""
    parts = [
        f'<svg viewBox="0 0 {VIEW_W:.0f} {VIEW_H:.0f}" '
        f'preserveAspectRatio="xMidYMid meet" '
        f'style="width:100%;height:100%;font-family:ui-monospace,monospace">',
        _ladder_skeleton(),
        _bank_skeleton(),
        # the boresight — fixed, because it is where the AEROPLANE points
        f'<path d="M {_CX - 70:.0f} {_CY} h 44 l 14 14 l 14 -14 h 44" '
        f'fill="none" stroke="#ffffff" stroke-width="3"/>',
        _tape_skeleton(_IAS_X, "ias", "right", "IAS m/s"),
        _tape_skeleton(_ALT_X, "alt", "left", "ALT m"),
        f'<rect x="{_CX - 70:.0f}" y="28" width="140" height="40" '
        f'fill="#001b10" stroke="{_GREEN}" stroke-width="2"/>'
        f'<text id="hud-hdg" x="{_CX:.0f}" y="57" fill="{_GREEN}" '
        f'font-size="26" text-anchor="middle"></text>',
        f'<text id="hud-alpha" x="60" y="{VIEW_H - 90:.0f}" fill="{_GREEN}" '
        f'font-size="24"></text>'
        f'<text id="hud-g" x="60" y="{VIEW_H - 56:.0f}" fill="{_GREEN}" '
        f'font-size="24"></text>',
        f'<rect x="{_THR_X:.0f}" y="{_THR_TOP:.0f}" width="26" '
        f'height="{_THR_H:.0f}" fill="#00120a" fill-opacity="0.4" '
        f'stroke="{_GREEN}" stroke-opacity="0.6"/>'
        f'<rect id="hud-thr" x="{_THR_X:.0f}" y="{_THR_BOT:.0f}" '
        f'width="26" height="0" fill="{_GREEN}" fill-opacity="0.75"/>'
        # THE NUMBER, ON THE GLASS. The bar alone says "about two thirds",
        # which is not a thrust — and the value used to live on a Quasar
        # slider that could not follow the keyboard moving it. It belongs
        # here, over the bar it fills, where the frame path writes it.
        f'<text id="hud-thrn" x="{_THR_X + 13:.0f}" '
        f'y="{_THR_TOP - 12:.0f}" fill="{_GREEN}" font-size="24" '
        f'text-anchor="middle">0</text>'
        f'<text x="{_THR_X + 13:.0f}" y="{VIEW_H - 26:.0f}" fill="{_GREEN}" '
        f'font-size="18" text-anchor="middle">THR N</text>',
    ]
    for i in range(3):
        parts.append(f'<text id="hud-warn{i}" x="{_CX:.0f}" '
                     f'y="{_CY + 200 + i * 30:.0f}" fill="{_RED}" '
                     f'font-size="24" text-anchor="middle"></text>')
    parts.append("</svg>")
    return "".join(parts)


# --------------------------------------------------------------- the state

def state(*, speed: float, altitude: float, heading_deg: float,
          pitch_deg: float, roll_deg: float, alpha_deg: float, g: float,
          throttle: float, throttle_max: float, throttle_min: float = 0.0,
          warnings=()) -> dict:
    """The ~20 numbers that change, with every clamp and wrap ALREADY DONE.

    Kept on this side so the thresholds stay testable in Python and the
    browser half is pure geometry with no policy in it.
    """
    # the bar fills between the two ANSWERED stops, not from zero: with a
    # reverse-capable floor, "no thrust" and "the bottom of the bar" stop
    # being the same place, and a bar measured from zero would show a
    # third of a bar for an idle engine
    lo, hi = float(throttle_min), float(throttle_max)
    span = hi - lo
    thr = 0.0 if span <= 0 else max(
        0.0, min(1.0, (float(throttle) - lo) / span))
    ias, alt = float(speed), float(altitude)
    return {
        # tapes: the translation, the first label's value, and the readout
        "ias": ias,
        "ias_minor": _frac(ias / IAS_STEP) * _MINOR_PX,
        "ias_major": _frac(ias / (2.0 * IAS_STEP)) * _MAJOR_PX,
        "ias_base": math.floor(ias / (2.0 * IAS_STEP)),
        "ias_step": 2.0 * IAS_STEP,
        "alt": alt,
        "alt_minor": _frac(alt / ALT_STEP) * _MINOR_PX,
        "alt_major": _frac(alt / (2.0 * ALT_STEP)) * _MAJOR_PX,
        "alt_base": math.floor(alt / (2.0 * ALT_STEP)),
        "alt_step": 2.0 * ALT_STEP,
        # attitude
        "pitch_px": float(pitch_deg) * _PITCH_PX,
        "roll": float(roll_deg),
        "cx": _CX, "cy": _CY,
        # numbers, with their colours decided here
        "hdg": float(heading_deg) % 360.0,
        "alpha": float(alpha_deg),
        "alpha_col": _AMBER if abs(float(alpha_deg)) > ALPHA_WARN_DEG
        else _GREEN,
        "g": float(g),
        "g_col": _AMBER if (g > G_WARN_HI or g < G_WARN_LO) else _GREEN,
        "thr_h": _THR_H * thr,
        "thr_y": _THR_BOT - _THR_H * thr,
        "thr_n": f"{float(throttle):,.0f}",
        "warn": [str(w) for w in list(warnings)[:3]],
    }


#: The browser half. A function body, installed once, that maps a
#: :func:`state` dict onto the skeleton. Node lookups are cached on the root
#: element the first time through, so a frame costs a handful of attribute
#: writes and no query at all.
APPLY_JS = r"""
function(root, s) {
  if (!root) return;
  let c = root.__hudCache;
  if (!c) {
    const q = (id) => root.querySelector('#' + id);
    c = root.__hudCache = {
      ladder: q('hud-ladder'), bank: q('hud-bank'),
      hdg: q('hud-hdg'), alpha: q('hud-alpha'), g: q('hud-g'),
      thr: q('hud-thr'), thrn: q('hud-thrn'),
      warn: [0,1,2].map(i => q('hud-warn' + i)),
      tapes: {}, last: {},
    };
    for (const t of ['ias', 'alt']) {
      c.tapes[t] = {
        minor: q('hud-' + t + '-minor'), major: q('hud-' + t + '-major'),
        read: q('hud-' + t),
        labels: [0,1,2,3,4,5,6,7,8].map(j => q('hud-' + t + '-l' + j)),
      };
    }
  }
  // a text node is only touched when its STRING changes: setting
  // textContent to what it already is still dirties the node for layout
  const put = (node, key, text) => {
    if (!node || c.last[key] === text) return;
    c.last[key] = text; node.textContent = text;
  };
  const attr = (node, key, name, value) => {
    if (!node || c.last[key] === value) return;
    c.last[key] = value; node.setAttribute(name, value);
  };

  attr(c.ladder, 'ladder', 'transform',
       'rotate(' + (-s.roll).toFixed(2) + ' ' + s.cx + ' ' + s.cy + ') ' +
       'translate(0 ' + s.pitch_px.toFixed(2) + ')');
  attr(c.bank, 'bank', 'transform',
       'rotate(' + (-s.roll).toFixed(2) + ' ' + s.cx + ' ' + s.cy + ')');

  for (const t of ['ias', 'alt']) {
    const tp = c.tapes[t];
    attr(tp.minor, t + 'mi', 'transform',
         'translate(0 ' + s[t + '_minor'].toFixed(2) + ')');
    attr(tp.major, t + 'ma', 'transform',
         'translate(0 ' + s[t + '_major'].toFixed(2) + ')');
    for (let j = 0; j < 9; j++) {
      // label j sits (j - 4) major steps BELOW centre, so it reads lower
      const v = (s[t + '_base'] - (j - 4)) * s[t + '_step'];
      put(tp.labels[j], t + 'l' + j, v.toFixed(0));
    }
    put(tp.read, t + 'r', s[t].toFixed(0));
  }

  put(c.hdg, 'hdg', ('00' + Math.round(s.hdg)).slice(-3));
  put(c.alpha, 'alpha', 'a ' + (s.alpha >= 0 ? '+' : '') + s.alpha.toFixed(1));
  attr(c.alpha, 'alphac', 'fill', s.alpha_col);
  put(c.g, 'g', 'g ' + (s.g >= 0 ? '+' : '') + s.g.toFixed(2));
  attr(c.g, 'gc', 'fill', s.g_col);
  attr(c.thr, 'thrh', 'height', s.thr_h.toFixed(1));
  attr(c.thr, 'thry', 'y', s.thr_y.toFixed(1));
  put(c.thrn, 'thrn', s.thr_n);
  for (let i = 0; i < 3; i++) put(c.warn[i], 'w' + i, s.warn[i] || '');
}
"""
