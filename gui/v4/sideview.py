"""The design in SIDE ELEVATION, drawn off the lattice that is flown.

Stage 5 asks for one number that nothing before it has ever asked for: where
along the aircraft the vertical surface goes. That question is unanswerable
in the abstract — "4.5 m" means nothing until you know that the wing is at 0
and the stabiliser is at 4.6 — so the stage draws the aeroplane from the
side, with the wing, the stabiliser, the fin, the CG and the neutral point on
one x axis, and puts the same stations in a table beside it.

STATED IS NOT FLOWN, so every box here is measured off the PANELS of the
built :class:`aerobo.vlm.VLM` — ``x`` (each panel's quarter-chord station),
``c`` (its chord) and ``z`` — and never off the fields that asked for them.
A blank fin chord is a default that lives inside
:func:`aerobo.flightmodel.build_flight_model`; reading the field would draw a
fin that is not there.

Two coordinate conventions meet here and both are stated on the picture:

* the lattice frame is **x AFT**, **z UP**, with the wing's quarter-chord
  line at ``x = 0``. That is the origin every station is quoted from;
* SVG is **y DOWN**, so the transform is ``sy = (z_hi - z) / k`` — written
  once, in :func:`_project`.

The viewBox is DERIVED from the drawn extents rather than typed. A typed box
is how a whole row of this package's own attitude panel came to sit outside
its viewBox and simply not be drawn, with the markup and the state both
perfectly correct (:mod:`gui.v4.gauge`). :func:`svg` therefore returns the
box it drew in, and a test reads every coordinate in the shipped markup back
out and asserts it is inside.
"""

from __future__ import annotations

import numpy as np

from aerobo.flightmodel import QUARTER
from gui.v3 import theme

__all__ = ["GROUPS", "stations", "svg", "Box"]

#: (key, label, colour role, is the surface a VERTICAL one).
#: Order is draw order: the wing last so it sits over the fin's root.
GROUPS = (("second", "rear wing", "wing", False),
          ("tail", "stabiliser", "tail", False),
          ("fin", "fin", "fin", True),
          ("winglet", "tip device", "tail", False),
          ("wing", "wing", "wing", False))

#: a horizontal surface has no thickness in this view, so it would draw as a
#: line of zero height. Floored at this fraction of the x extent — a drawing
#: convention, applied to nothing but the RECTANGLE, and never to a number.
_FLAT_FRAC = 0.012

#: the z half-extent is padded to at least this fraction of the x extent, so
#: that a design with no fin does not produce a viewBox 400 units wide and 2
#: high. The SCALE stays uniform: this adds empty air, it does not stretch.
_MIN_Z_FRAC = 0.16

_PAD_FRAC = 0.06                      # margin around the drawn extents

#: :data:`gui.v3.theme.MONO` contains DOUBLE QUOTES (``"SF Mono"``), and an
#: SVG attribute is written between double quotes. Interpolated raw, the
#: attribute closes at the first inner quote and the rest of the font stack
#: is parsed as a run of bare attributes. CSS accepts single-quoted family
#: names identically, so the stack is re-quoted once, here.
_MONO = theme.MONO.replace('"', "'")


class Box(dict):
    """One surface's extent in the side elevation, in metres."""


def _mask(model, key: str):
    """The panels of one group, as a boolean mask over the lattice."""
    vert = np.asarray(model.is_vertical, dtype=bool)
    tail = np.asarray(model.is_tail, dtype=bool)
    wl = np.asarray(model.is_winglet, dtype=bool)
    sec = np.asarray(getattr(model, "is_second", np.zeros_like(vert)),
                     dtype=bool)
    if key == "fin":
        return vert
    if key == "tail":
        return tail & ~vert
    if key == "winglet":
        return wl & ~vert & ~tail
    if key == "second":
        return sec & ~vert & ~tail & ~wl
    return ~(vert | tail | wl | sec)


def stations(model, *, x_cg: float, x_np: float | None = None) -> dict:
    """Every station the side elevation draws, in metres, off the PANELS.

    Returns ``{"boxes": {key: Box}, "x_cg": …, "x_np": …}``. A group with no
    panels is absent rather than present and empty, so a caller cannot draw
    a stabiliser onto a design that has none.

    Each box carries ``x_le``/``x_te`` (the extremes of the group's leading
    and trailing edges), ``x_qc`` (its MEAN quarter-chord station — the arm
    the derivatives are built on) and the z extent.
    """
    x = np.asarray(model.x, dtype=float)
    c = np.asarray(model.c, dtype=float)
    z = np.asarray(model.z, dtype=float)
    boxes: dict = {}
    for key, _label, _role, _vertical in GROUPS:
        m = _mask(model, key)
        if not m.any():
            continue
        le = x[m] - QUARTER * c[m]
        te = x[m] + (1.0 - QUARTER) * c[m]
        boxes[key] = Box(x_le=float(le.min()), x_te=float(te.max()),
                         x_qc=float(x[m].mean()),
                         chord=float(c[m].mean()),
                         z_lo=float(z[m].min()), z_hi=float(z[m].max()))
    return {"boxes": boxes, "x_cg": float(x_cg),
            "x_np": (None if x_np is None else float(x_np))}


def _extent(st: dict) -> tuple:
    """``(x0, x1, z0, z1)`` — the box the picture is drawn in [m]."""
    xs, zs = [], []
    for b in st["boxes"].values():
        xs += [b["x_le"], b["x_te"]]
        zs += [b["z_lo"], b["z_hi"]]
    for k in ("x_cg", "x_np"):
        if st.get(k) is not None:
            xs.append(float(st[k]))
    if not xs:
        return (0.0, 1.0, -0.5, 0.5)
    zs = zs or [0.0]
    x0, x1 = min(xs), max(xs)
    z0, z1 = min(zs), max(zs)
    span = max(x1 - x0, 1e-6)
    # pad z symmetrically about what is there, to keep the scale uniform
    half = max(0.5 * (z1 - z0), _MIN_Z_FRAC * span * 0.5)
    mid = 0.5 * (z0 + z1)
    z0, z1 = mid - half, mid + half
    pad = _PAD_FRAC * span
    return (x0 - pad, x1 + pad, z0 - pad, z1 + pad)


def _project(x0: float, z1: float):
    """Lattice ``(x, z)`` -> SVG ``(sx, sy)``, at unit scale (1 m = 1 unit).

    One place, because the z flip is the kind of sign that is right in the
    data and wrong on the screen.
    """
    def f(x: float, z: float) -> tuple:
        return (float(x) - x0, z1 - float(z))
    return f


def svg(st: dict, *, height_px: int = 190) -> str:
    """The side elevation as one self-contained SVG string.

    Drawn at 1 unit per metre in a viewBox derived from the extents, so the
    scale is UNIFORM in x and z — a side elevation that stretched one axis
    would answer the question this picture exists for ("is the fin near the
    tail?") with a lie.
    """
    x0, x1, z0, z1 = _extent(st)
    W, H = x1 - x0, z1 - z0
    P = _project(x0, z1)
    span = max(W, 1e-6)
    flat = _FLAT_FRAC * span
    stroke = 0.004 * span

    colour = {"wing": theme.INK, "tail": theme.INK_MUTED,
              "fin": theme.ACCENT}
    out = [f'<svg viewBox="0 0 {W:.4f} {H:.4f}" width="100%" '
           f'height="{height_px}" preserveAspectRatio="xMidYMid meet" '
           f'style="display:block" font-family="{_MONO}">']

    # the waterline: z = 0 is the wing plane, and every height is from it
    y0 = P(0.0, 0.0)[1]
    out.append(f'<line x1="0" y1="{y0:.4f}" x2="{W:.4f}" y2="{y0:.4f}" '
               f'stroke="{theme.RULE_SOFT}" stroke-width="{stroke:.4f}" '
               f'stroke-dasharray="{6 * stroke:.4f} {4 * stroke:.4f}"/>')

    for key, _label, role, vertical in GROUPS:
        b = st["boxes"].get(key)
        if b is None:
            continue
        h = b["z_hi"] - b["z_lo"]
        if not vertical and h < flat:
            lo = 0.5 * (b["z_lo"] + b["z_hi"]) - 0.5 * flat
            h = flat
        else:
            lo = b["z_lo"]
        sx, sy = P(b["x_le"], lo + h)
        out.append(
            f'<rect x="{sx:.4f}" y="{sy:.4f}" '
            f'width="{b["x_te"] - b["x_le"]:.4f}" height="{h:.4f}" '
            f'fill="{colour[role]}" fill-opacity="{0.75 if vertical else 0.9}"'
            f'/>')

    # the CG and the neutral point, on the wing plane, as the two markers
    # stage 6 draws in the 3-D picture — same colours, same meaning.
    for key, col, r in (("x_np", theme.ACCENT, 0.9), ("x_cg", "#e07b2a", 1.2)):
        v = st.get(key)
        if v is None:
            continue
        cx, cy = P(v, 0.0)
        out.append(f'<circle cx="{cx:.4f}" cy="{cy:.4f}" '
                   f'r="{r * flat:.4f}" fill="{col}"/>')
    out.append("</svg>")
    return "".join(out)
