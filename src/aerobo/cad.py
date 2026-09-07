"""The flown geometry as CAD — STL solids plus an OpenVSP build script.

The solvers report their geometry as spanwise ARRAYS (panel stations: y, z,
chord, the winglet mask, the tail block) because that is what a lifting-surface
method needs. A CAD package needs a SURFACE. This module is the one place that
turns the former into the latter, and it does it exactly once: the loft here is
the same loft the 3-D view draws (``gui/nice_app.fig_wing3d`` delegates to it),
so what lands in OpenVSP is what the GUI shows is what the solver flew. If the
two ever disagreed, the export would be decoration.

What is exported

* ``surfaces()``   — named (X, Y, Z) structured grids, one per lifting surface
  (wing + tip device as ONE grid, because the blend has no crease; the tail;
  the second wing of a tandem; the elevator plate).
* ``stl_bytes()`` / ``write_stl()`` — binary or ASCII STL. Closed lofts are
  capped at both span ends and stitched along the trailing edge, so each solid
  is watertight (``tests/test_cad_export.py`` checks every edge is shared by
  exactly two facets) and OpenVSP's mesh import does not have to guess.
* ``vsp_script()`` — a Python script for the OpenVSP API that rebuilds wing and
  tail as NATIVE WingGeoms (per-section span/chord/twist/dihedral, the section
  read from a written .dat, the elevator as a control sub-surface). Native geoms
  are what VSPAERO's vortex lattice wants; an imported STL is a MeshGeom and
  only the panel solver will touch it. The script falls back to importing the
  STL if the parametric rebuild fails on the installed VSP version.
* ``airfoil_dat()`` — the section in Selig/XFOIL order, for the script above and
  for any other tool.

Axes and units. The package's frame throughout: **x downstream +, y starboard,
z up**, metres, quarter chord at x = 0 for the wing. That is also OpenVSP's
frame, so the export needs no transform — an aft tail imports aft, a canard
ahead.

What is NOT claimed. Four things are drawn as the solver's own idealisation
rather than as manufacturing geometry, and each is called out where it happens:
the quarter-chord line is straight (sweep enters the reduced-order drag
build-up, never the lifting-surface geometry — see ``geometry.Wing``); the
elevator is a flat plate at the hinge station, because the physics models it as
a flap EFFECTIVENESS τ, never as a geometric hinge (``tail.flap_effectiveness``);
a V-tail is drawn as the two dihedralled panels its equivalent flat surface
stands for; and the car's MOUNT is a straight member from the wing's skin to
the deck, whatever side it is declared on. A swan neck's whole point is that it
wraps OVER the wing and lands on the pressure side, and that route is a path
``carmount.pylon_length_m`` does not model — it returns a difference of two
heights, not a member length — so drawing the wrap would be drawing a length
nobody computed. Exporting more than was solved would put numbers in a CAD file
that no part of the pipeline ever computed.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "Surface", "naca4_section", "section_path", "planform_arrays",
    "wing_twist", "tc_for_view", "path_normals", "loft_surface", "loft_path",
    "v_tail_path", "fold_to_vee", "tail_path",
    "elevator_grid", "nonplanar_arrays", "canted_arrays", "sweep_offset",
    "tail_surfaces",
    "second_wing_twist", "second_wing_surfaces", "fin_surface", "surfaces",
    "triangles", "stl_bytes", "write_stl", "airfoil_dat", "wing_sections",
    "vsp_script", "export", "READ_ME",
    "FILE_SUFFIXES", "export_name",
]

#: WHAT EACH EXPORTED FILE IS CALLED, given the name the user chose. One
#: table, because the stem is a CONTRACT and not a label: the generated
#: OpenVSP script re-derives ``STEM + "_wing.dat"``, ``STEM + "_tail.dat"``,
#: ``STEM + "_fin.dat"``
#: and ``STEM + ".stl"`` to find its inputs and writes ``STEM + ".vsp3"``
#: beside itself (:func:`vsp_script`), and :func:`aerobo.vsp.build_model`
#: hands that same ``.vsp3`` name to another interpreter. Spelled out in
#: each of those places, one of them drifts and the bundle stops rebuilding
#: — which is a broken export that still writes every file it promised.
#:
#: The per-surface STLs are not here: their name carries the SURFACE
#: (``{stem}_{part}.stl``), so there is one per part rather than one per
#: kind, and :func:`export` builds them from the surface list it has.
FILE_SUFFIXES = {
    "stl": ".stl",
    "stl_ascii": "_ascii.stl",
    "dat": "_wing.dat",
    "dat_aft": "_tail.dat",
    "dat_fin": "_fin.dat",
    "vsp": "_vsp.py",
    "vsp3": ".vsp3",
    "readme": "_README.txt",
    # ...and the SECTION on its own, which stage 2 exports long before there
    # is an aircraft to put it on (and in an aerofoil-only session, instead
    # of one). Separate names from "dat" above on purpose: that one is the
    # section the RUN flew, these are the section the user CHOSE, and a
    # session where the two differ must not have one silently overwrite the
    # other. Nothing re-derives these three the way the OpenVSP script
    # re-derives the wing .dat, but they live in this table all the same —
    # a second naming table is exactly how the first one came to drift.
    "section_dat": "_section.dat",
    "section_polar": "_section_polar.csv",
    "section_json": "_section.json",
}


def export_name(stem: str, kind: str) -> str:
    """The file name one export ``kind`` gets for this ``stem``.

    Refuses an unknown kind by name. A shell that asks for a file it cannot
    spell should hear about it rather than be handed a default — the failure
    this replaces was a ``kind`` the writer had no branch for falling through
    an ``else`` and silently writing the aerofoil .dat instead.
    """
    try:
        return f"{stem}{FILE_SUFFIXES[kind]}"
    except KeyError:
        raise ValueError(
            f"unknown export kind {kind!r}; "
            f"choose from {sorted(FILE_SUFFIXES)}") from None

# Two points closer than this (in metres, on a metre-scale wing) are the same
# point: it is what decides whether the chordwise loop is duplicated at the
# trailing edge and therefore whether the TE stitch would be degenerate.
SEAM_TOL = 1e-9

# A facet smaller than this in area is dropped rather than written. Zero-area
# facets are legal STL but every downstream mesher complains about them, and
# they are exactly what a sharp trailing edge or a repeated station produces.
MIN_FACET_AREA = 1e-14


# --------------------------------------------------------------- the section

def naca4_section(tc: float, m: float = 0.02, p: float = 0.4, n: int = 21):
    """NACA 4-digit section outline at unit chord.

    Returns (xc, zc): one closed chordwise path TE -> upper -> LE -> lower ->
    TE (length 2n-1). Thickness uses the closed-trailing-edge -0.1036
    coefficient; camber is the standard two-parabola mean line (m, p) — the
    NACA 24XX family (2 % camber at 40 %) the polar stack uses.

    This is the STAND-IN section: it is lofted only when the design does not
    carry a CST section of its own (:func:`section_path`). ``n`` defaults to
    the count the 3-D view has always drawn; a CAD export asks for more
    (:func:`surfaces`), because a facet is a chord of the arc it replaces and
    a mesher will not add points back.
    """
    tc = float(max(tc, 1e-3))
    xb = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, int(n))))   # cosine
    yt = 5.0 * tc * (0.2969 * np.sqrt(xb) - 0.1260 * xb - 0.3516 * xb ** 2
                     + 0.2843 * xb ** 3 - 0.1036 * xb ** 4)
    yc = np.where(xb < p,
                  m / p ** 2 * (2 * p * xb - xb ** 2),
                  m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * xb - xb ** 2)) \
        if m > 0 else np.zeros_like(xb)
    xc = np.concatenate([xb[::-1], xb[1:]])                 # TE->LE->TE
    zc = np.concatenate([(yc + yt)[::-1], (yc - yt)[1:]])   # upper then lower
    return xc, zc


def tc_for_view(geom: dict, x_best=None, labels=None) -> float:
    """Section t/c to loft: the solved value if the design carries one, else
    the NACA 2412 default the Tier A polar represents."""
    for key in ("tc", "tc_sec"):
        v = geom.get(key)
        if isinstance(v, (int, float)) and 0.0 < float(v) < 0.5:
            return float(v)
        if x_best is not None and labels and key in list(labels):
            return float(x_best[list(labels).index(key)])
    return 0.12


def _coords_from_section(section):
    """(n, 2) coordinates out of whatever the caller had to hand, or None.

    Accepts the raw closed loop (``api.section_coords``), the whole section
    report (``api.section_report``, whose ``design.coords`` is that loop) or
    None. Taking the report too is not politeness: the result page carries the
    report, and a shell that passed it where coordinates were expected used to
    fall through to the NACA stand-in without saying so — exporting a section
    the design never flew.
    """
    if section is None:
        return None
    if isinstance(section, dict):
        for key in ("design", "section"):
            blk = section.get(key)
            if isinstance(blk, dict) and blk.get("coords") is not None:
                section = blk["coords"]
                break
        else:
            section = section.get("coords")
        if section is None:
            return None
    pts = np.asarray(section, dtype=float)
    if pts.ndim == 2 and pts.shape[1] == 2 and pts.shape[0] >= 10:
        return pts
    return None


def section_path(geom: dict, x_best=None, labels=None, section=None,
                 n: int = 21):
    """(xc, zc) unit-chord section path: the OPTIMISED CST section when the
    design carries one, else the NACA 4-digit stand-in at the solved t/c."""
    pts = _coords_from_section(section)
    if pts is not None:
        return pts[:, 0], pts[:, 1]
    return naca4_section(tc_for_view(geom, x_best, labels), n=n)


# ------------------------------------------------------- spanwise geometry

def planform_arrays(geom: dict, x_best=None, labels=None):
    """(y, chord) full-span arrays from a design-report geometry dict.

    Prefers the solver's own spanwise arrays; falls back to a trapezoid from
    b/S/taper. A half-span export is mirrored, and the result is sorted in y.
    """
    y = geom.get("y")
    c = geom.get("chord")
    if y is not None and c is not None and len(y) and len(y) == len(c):
        y = np.asarray(y, float)
        c = np.asarray(c, float)
    else:
        b, S = geom.get("b"), geom.get("S")
        taper = geom.get("taper")
        if taper is None and x_best is not None and labels:
            if "taper" in list(labels):
                taper = float(x_best[list(labels).index("taper")])
        if b is None or S is None or taper is None:
            return None, None
        c_root = 2.0 * float(S) / (float(b) * (1.0 + float(taper)))
        y = np.linspace(-b / 2, b / 2, 81)
        c = c_root * (1.0 - (1.0 - float(taper)) * np.abs(2.0 * y / b))
    if y.size == 0:
        return None, None
    if y.min() >= -1e-9:                       # mirror a half-span
        y = np.concatenate([-y[::-1], y])
        c = np.concatenate([c[::-1], c])
    order = np.argsort(y)
    return y[order], c[order]


def wing_twist(geom: dict, y, x_best=None, labels=None):
    """Linear twist law sampled at |y|, clipped at the tip — the same sampling
    the VLM uses, so winglet panels inherit the tip twist."""
    tw_r, tw_t = geom.get("twist_root_deg"), geom.get("twist_tip_deg")
    lab = list(labels or [])
    if x_best is not None and lab:
        if tw_r is None and "twist_root_deg" in lab:
            tw_r = float(x_best[lab.index("twist_root_deg")])
        if tw_t is None and "twist_tip_deg" in lab:
            tw_t = float(x_best[lab.index("twist_tip_deg")])
    tw_r, tw_t = float(tw_r or 0.0), float(tw_t or 0.0)
    y = np.asarray(y, float)
    b = float(geom.get("b") or 0.0)
    if b <= 0.0:
        b = float(np.max(np.abs(y)) * 2.0) or 1.0
    eta = np.clip(np.abs(2.0 * y / b), 0.0, 1.0)
    return np.deg2rad(tw_r + (tw_t - tw_r) * eta)


def nonplanar_arrays(geom: dict):
    """(y, z, chord, is_winglet) for a solver that flew OUT of the wing plane.

    The VLM winglet modes export their panel stations in path order — port
    winglet tip -> port wing tip -> starboard wing tip -> starboard winglet tip
    — so the arrays trace one continuous spanwise polyline in (y, z). Returns
    None unless winglet panels were actually flown (the VLM drops winglets
    below ``vlm.MIN_WINGLET_FRAC``, and an export must carry what was flown,
    not what was asked for).
    """
    y, z, c = geom.get("y"), geom.get("z"), geom.get("is_winglet")
    ch = geom.get("chord")
    if y is None or z is None or c is None or ch is None:
        return None
    if not (len(y) == len(z) == len(c) == len(ch)):
        return None
    mask = np.asarray(c, dtype=bool)
    if not mask.any():
        return None
    return (np.asarray(y, float), np.asarray(z, float),
            np.asarray(ch, float), mask)


def extend_to_edges(y, z, c, mask, developed_span: float):
    """Push the two end stations out to the surface's true EDGE.

    Solvers report panel CENTRES. Lofting between them stops half a panel
    short at each tip, which a picture forgives and a CAD file does not: the
    first model built this way measured 10.347 m of developed span where the
    design has 10.750, and a tail 6.7 % short of its own area — i.e. the file
    disagreed with every number on the result page (measured in OpenVSP
    3.51.2, which is exactly what the round trip is for).

    So the ends are extended along the local tangent to make the polyline as
    long as the design says the surface is, with the chord extrapolated at the
    rate the last two stations set. The deficit is split equally because these
    geometries are symmetric. Nothing is invented when the numbers disagree:
    if the deficit is negative (the stations already reach the edge) or bigger
    than one panel (the ``developed_span`` handed in does not belong to these
    arrays), the arrays come back untouched.
    """
    y = np.asarray(y, float)
    z = np.asarray(z, float)
    c = np.asarray(c, float)
    mask = np.asarray(mask, dtype=bool)
    if y.size < 2 or not np.isfinite(developed_span):
        return y, z, c, mask
    seg = np.hypot(np.diff(y), np.diff(z))
    deficit = float(developed_span) - float(seg.sum())
    if deficit <= 1e-12 or deficit > 2.0 * float(seg.max()):
        return y, z, c, mask
    half = 0.5 * deficit
    ends = []
    for end, nb, step in ((0, 1, seg[0]), (-1, -2, seg[-1])):
        ty = (y[end] - y[nb]) / step                   # unit tangent, outward
        tz = (z[end] - z[nb]) / step
        dc = (c[end] - c[nb]) / step                   # chord rate per arc
        ends.append((y[end] + ty * half, z[end] + tz * half,
                     max(c[end] + dc * half, 1e-6), bool(mask[end])))
    head, tail = ends
    out_y = np.concatenate([[head[0]], y, [tail[0]]])
    out_z = np.concatenate([[head[1]], z, [tail[1]]])
    out_c = np.concatenate([[head[2]], c, [tail[2]]])
    out_m = np.concatenate([[head[3]], mask, [tail[3]]])
    return out_y, out_z, out_c, out_m


def _rooted(y, z, c, mask, tw):
    """Prepend the CENTRELINE station to a half-span chain (y[0] -> 0).

    Everything is extrapolated linearly in y from the two innermost stations,
    which is exact for a trapezoid and inboard-of-a-panel accurate for a chord
    law. A chain that already starts at y = 0 is returned untouched.
    """
    y = np.asarray(y, float)
    if y.size < 2 or abs(float(y[0])) <= 1e-9:
        return y, np.asarray(z, float), np.asarray(c, float), \
            np.asarray(mask, dtype=bool), np.asarray(tw, float)
    dy = float(y[1] - y[0])
    if abs(dy) < 1e-12:
        return y, np.asarray(z, float), np.asarray(c, float), \
            np.asarray(mask, dtype=bool), np.asarray(tw, float)
    f = float(y[0]) / dy                       # steps to walk back to y = 0

    def back(a):
        a = np.asarray(a, float)
        return np.concatenate([[a[0] - f * (a[1] - a[0])], a])

    return (np.concatenate([[0.0], y]), back(z), back(c),
            np.concatenate([[bool(mask[0])], np.asarray(mask, dtype=bool)]),
            back(tw))


def developed_span_of(geom: dict, winglet: dict | None = None) -> float:
    """How long the wing's quarter-chord line IS, tip device included.

    The span in the report is the wing's own (developed) span; a tip device
    adds its arc length to the line, which is what :func:`extend_to_edges`
    has to match. Returns NaN when the report does not say — and then nothing
    is extended.
    """
    b = geom.get("b")
    if not isinstance(b, (int, float)) or not b:
        return float("nan")
    wl = winglet if winglet is not None else (geom.get("winglet") or {})
    h = float(wl.get("h_m", 0.0) or 0.0)
    return float(b) + 2.0 * h


def canted_arrays(geom: dict):
    """(y, z, chord, mask) for a wing that leaves the plane with NO tip device.

    :func:`nonplanar_arrays` asks "did a tip device fly", which was the only
    way out of the wing plane when it was written. A WING DIHEDRAL is the
    other one (``geometry.dihedral_rotate``, searched since the cant became a
    design variable), and a design that carries one without a device answered
    that question False: the report stated ``z`` rising to 1.293 m at 15 deg
    and the loft drew ``z`` in [-0.052, +0.098] — i.e. flat — so the STL, the
    OpenVSP script, the 3-D view and the V4 flight scene all showed an
    aeroplane the solver did not fly.

    Deliberately NOT folded into :func:`nonplanar_arrays`: that gate is also
    asked of the TAIL and the rear-wing blocks, whose ``z`` array is ABSOLUTE
    (0.5 everywhere on a tail at ``z_offset`` 0.5) and whose height is applied
    a second time as ``z0``. Widening the shared gate would draw those
    surfaces at twice their height. This one is the WING's, and only
    :func:`wing_path` asks it.

    ``mask`` comes back all-False: there is no device, so there is no crease
    for :func:`path_normals` to keep. Returns None for a genuinely planar wing
    (every published export is bit-for-bit unchanged).
    """
    y, z, ch = geom.get("y"), geom.get("z"), geom.get("chord")
    if y is None or z is None or ch is None:
        return None
    y = np.asarray(y, float)
    z = np.asarray(z, float)
    ch = np.asarray(ch, float)
    if not (y.size == z.size == ch.size) or y.size < 2:
        return None
    if not np.any(z):
        return None
    return y, z, ch, np.zeros(y.size, dtype=bool)


def sweep_offset(geom: dict, y):
    """Where a swept wing's quarter-chord line SITS: ``|y| tan Lambda``.

    The lattice's own rule (``vlm.VLM``: ``xe = np.abs(ye) * tanL``, measured
    on the PLANFORM-VIEW y — i.e. after the dihedral rotation — so a vertical
    tip device inherits its junction's station). It was in the solver and in
    no drawer: measured on `tail + winglet [free cant]`, the lofted wing spans
    x [-0.3076, +0.9228] at 0, 10 AND 20 deg of sweep, bit-identical, while
    the report states the angle the design flew.

    Returns a plain ``0.0`` for an unswept wing, which keeps every published
    export bit-for-bit.

    AND FOR A REDUCED-ORDER ONE, whatever angle it states. Sweep reaches a
    LIFTING LINE as simple-sweep theory on the section slope and a drag
    build-up, never as geometry — its quarter-chord line really is straight,
    and `wing t/c + sweep` reports 25 deg with no panel path at all. Drawing
    that one swept would export a wing that was never flown, which is the
    warning ``vsp_script``'s docstring has been carrying. The discriminator is
    the panel path itself: ``geometry["z"]`` exists exactly where a lattice
    bent the line (``vlm.VLM``) and nowhere else.
    """
    geom = geom or {}
    lam = float(geom.get("sweep_deg") or 0.0)
    if not lam or geom.get("z") is None:
        return 0.0
    return np.abs(np.asarray(y, float)) * np.tan(np.deg2rad(lam))


def wing_path(geom: dict, x_best=None, labels=None, extend: bool = True):
    """(y, z, chord, is_winglet, twist_rad) for the wing, tip to tip.

    One place decides what the wing IS — the solver's own nonplanar path where
    there is one, else the planar stations (else a trapezoid) — so the STL and
    the OpenVSP section chain cannot be built on different wings. Returns None
    when the report carries no planform at all.

    "Nonplanar" is a device OR a dihedral (:func:`canted_arrays`): both leave
    the wing plane, and the loft has to follow whichever did.
    """
    np_geom = nonplanar_arrays(geom) or canted_arrays(geom)
    if np_geom is not None:
        y, z, c, mask = np_geom
    else:
        y, c = planform_arrays(geom, x_best, labels)
        if y is None:
            return None
        z = np.zeros_like(y)
        mask = np.zeros(y.size, dtype=bool)
    if extend:
        y, z, c, mask = extend_to_edges(y, z, c, mask,
                                        developed_span_of(geom))
    return y, z, c, mask, wing_twist(geom, y, x_best, labels)


def path_normals(y, z, mask, smooth: bool = False):
    """Unit (tangent, normal) along a spanwise (y, z) path.

    Tangents are differenced SEGMENT-WISE (wing panels vs winglet panels) so a
    SHARP junction keeps its crease instead of being smeared by a central
    difference straddling the kink. The normal is the in-plane left turn of the
    tangent: +z on the main wing, tilting inboard-up along a canted winglet —
    i.e. the local lift direction, mirrored on both sides.

    ``smooth=True`` differences the WHOLE path instead, which is what a BLENDED
    transition needs: the line leaves the wing plane tangentially, so there is
    no crease, and segmenting at the junction would build a kink the geometry
    does not have — precisely what a blended winglet exists to avoid.

    ``mask`` is cut wherever its VALUE changes, so a boolean is-winglet flag
    (one crease, at the tip junction) and an integer segment id (a V-tail with
    a tip device has three: the root vertex and a device junction per side)
    both work. A boolean is the two-segment case of the integer one.
    """
    y = np.asarray(y, float)
    z = np.asarray(z, float)
    mask = np.asarray(mask)
    mask = mask.astype(int) if mask.dtype == bool else mask
    T = np.zeros((y.size, 2))
    cuts = (np.array([], dtype=int) if smooth
            else np.flatnonzero(np.diff(mask) != 0) + 1)
    for seg in np.split(np.arange(y.size), cuts):
        if seg.size < 2:
            T[seg] = (1.0, 0.0)
            continue
        T[seg] = np.gradient(np.column_stack([y[seg], z[seg]]), axis=0)
    nrm = np.linalg.norm(T, axis=1, keepdims=True)
    nrm[nrm == 0.0] = 1.0
    T = T / nrm
    return T, np.column_stack([-T[:, 1], T[:, 0]])


def _spanwise(v):
    """A scalar offset, or a per-station one turned into a span COLUMN.

    ``X`` is (n_span, n_chord); an ``(n_span,)`` offset added bare would
    broadcast across the CHORD and silently loft nonsense whenever the span
    and chord counts happened to match.
    """
    v = np.asarray(v, float)
    return v[:, None] if v.ndim == 1 else v


def loft_surface(y, c, twist_rad, xc, zc, x0=0.0, z0=0.0):
    """Span-loft a unit-chord section into (X, Y, Z) grids, shape (ns, nc).

    In the PACKAGE's axes: x DOWNSTREAM positive, z up. The chordwise
    coordinate ``s = xc - 0.25`` puts the quarter chord at 0, the leading edge
    at -0.25 c (upstream, where the nose belongs) and the trailing edge at
    +0.75 c, so the section faces the way every number in the breakdown says it
    does — an aft tail at l_t > 0 sits AFT, a canard at l_t < 0 ahead, and the
    CG / neutral-point stations are at their own values. Nose-up twist raises
    the leading edge: rotating by +theta about +y takes (s, z) to
    (s cos + z sin, -s sin + z cos), which lifts the LE at s = -0.25 c.

    ``x0``/``z0`` are a scalar offset for the whole surface (a tandem's
    stagger) OR one value PER STATION — which is what a swept quarter-chord
    line is (:func:`sweep_offset`). Broadcast down the span rather than across
    the chord, which is what a bare ``+ x0`` would have done to an array.
    """
    y = np.asarray(y, float)
    c = np.asarray(c, float)
    xc = np.asarray(xc, float)
    zc = np.asarray(zc, float)
    twist_rad = np.asarray(twist_rad, float)
    s = xc - 0.25                    # chordwise coord: c/4 at 0, LE at -0.25
    cs, sn = np.cos(twist_rad), np.sin(twist_rad)
    Sg = np.outer(np.ones_like(y), s) * c[:, None]
    Zg = np.outer(np.ones_like(y), zc) * c[:, None]
    X = Sg * cs[:, None] + Zg * sn[:, None] + _spanwise(x0)
    Z = -Sg * sn[:, None] + Zg * cs[:, None] + _spanwise(z0)
    Y = np.repeat(y[:, None], xc.size, axis=1)
    return X, Y, Z


def loft_path(y, z, c, twist_rad, xc, zc, mask, smooth: bool = False,
              x0: float = 0.0):
    """Loft a unit-chord section along an arbitrary (y, z) spanwise path.

    Reduces EXACTLY to :func:`loft_surface` for a planar path (normal = +z
    everywhere), and bends the section with the winglet where the path leaves
    the wing plane. ``smooth`` passes through to :func:`path_normals` — a
    blended junction has no crease to preserve.

    ``x0`` is a scalar or one value per station — see :func:`loft_surface`.
    """
    y = np.asarray(y, float)
    z = np.asarray(z, float)
    c = np.asarray(c, float)
    xc = np.asarray(xc, float)
    zc = np.asarray(zc, float)
    twist_rad = np.asarray(twist_rad, float)
    _, Nv = path_normals(y, z, mask, smooth=smooth)
    s = xc - 0.25                                   # x aft +, LE at -0.25 c
    cs, sn = np.cos(twist_rad), np.sin(twist_rad)
    Sg = np.outer(np.ones_like(y), s) * c[:, None]
    Zg = np.outer(np.ones_like(y), zc) * c[:, None]
    X = Sg * cs[:, None] + Zg * sn[:, None] + _spanwise(x0)
    off = -Sg * sn[:, None] + Zg * cs[:, None]      # along the local normal
    Y = y[:, None] + off * Nv[:, 0][:, None]
    Z = z[:, None] + off * Nv[:, 1][:, None]
    return X, Y, Z


def v_tail_path(tail: dict, z0: float):
    """(y, z, chord, mask) of a V-tail's two PANELS, or None for a flat tail.

    The solver flies the equivalent flat surface of area S_t cos^2 Gamma (the
    panel both sees and returns cos Gamma of the vertical — ``tail.lifting_area``)
    at the same aspect ratio, so the flat surface's span IS the panels'
    horizontal projection. Drawing the panels from that:

        half-projection b_t/2  ->  panel length (b_t/2)/cos Gamma,
        panel chord = S_t cos Gamma / b_t   (both panels total the PANEL area),
        z rises by |y| tan Gamma.

    Which is why exporting this is honest rather than decoration: the wetted
    area written to the file is the one the profile drag was charged on, and
    its projection is the one the pitch effectiveness came from.
    """
    gam = np.deg2rad(float(tail.get("dihedral_deg") or 0.0))
    b_t = float(tail.get("b_t") or 0.0)
    S_t = float(tail.get("S_t") or 0.0)
    if tail.get("type") != "v_tail" or gam <= 0.0 or b_t <= 0.0 or S_t <= 0.0:
        return None
    n = 12
    y = np.concatenate([np.linspace(-b_t / 2.0, 0.0, n),
                        np.linspace(0.0, b_t / 2.0, n)[1:]])
    z = z0 + np.abs(y) * np.tan(gam)
    c = np.full(y.size, S_t * np.cos(gam) / b_t)
    return y, z, c, (y >= 0.0)


def fold_to_vee(y, z, c, mask, z0: float, gamma_deg: float):
    """Fold a flat tail path — panel AND the tip device it carries — into a
    V-tail's two dihedralled panels. Returns (y, z, chord, is_winglet, seg).

    :func:`v_tail_path` builds the panels from the tail block alone, which is
    all a published lifting-line tail reports. A tail solved in the nonplanar
    wing+tail system reports its OWN spanwise path, and a stabiliser that
    carries a tip device is exactly that case — so the two shapes met here,
    and the own path won: the V was drawn FLAT, with only the device leaving
    the plane. The one thing the layout choice is about was invisible on the
    one configuration that states it twice.

    The fold keeps every number :func:`v_tail_path` keeps, and the device on
    top of them:

    * the PANEL stations keep their y — the flat equivalent's span is the
      panels' horizontal projection — and rise to ``z0 + |y| tan Gamma``;
    * the DEVICE is rigid on its panel: its offsets from the panel tip are
      rotated by Gamma (mirrored per side), so its arc length and its cant
      RELATIVE to the panel it grows from are the ones that were flown;
    * chords scale by 1 / cos Gamma. Folding stretches the panel's arc by
      that factor, and the flown flat surface has area S_t cos^2 Gamma, so
      this is what makes the drawn panels total S_t — the area the profile
      drag was charged on. The device's chord scales with them, which is the
      junction staying continuous rather than a claim about the device.

    ``seg`` is the segment id for :func:`path_normals`: a V-tail with a device
    creases at the root vertex and at each device junction, and a boolean
    is-winglet flag can only mark the junctions. Returns None if the path
    carries fewer than two panel stations to fold about.
    """
    y = np.asarray(y, float)
    z = np.asarray(z, float)
    c = np.asarray(c, float)
    mask = np.asarray(mask, dtype=bool)
    gam = np.deg2rad(float(gamma_deg))
    if gam <= 0.0 or y.size < 2 or (~mask).sum() < 2:
        return None
    out_y, out_z = y.copy(), np.full(y.size, float(z0))
    panel = ~mask
    out_y[panel] = y[panel]
    out_z[panel] = float(z0) + np.abs(y[panel]) * np.tan(gam)
    cuts = np.flatnonzero(np.diff(mask.astype(int)) != 0) + 1
    runs = np.split(np.arange(y.size), cuts)
    seg = np.zeros(y.size, dtype=int)
    k = 0
    for run in runs:
        if not mask[run[0]]:
            # the panel run spans BOTH sides: it creases at the root, which
            # is a segment boundary the is-winglet flag has no way to say
            side = np.where(y[run] < 0.0, -1, 1)
            for s in (-1, 1):
                part = run[side == s]
                if part.size:
                    seg[part] = k
                    k += 1
            continue
        # the device hangs off the panel station next to it in path order
        nb = run[0] - 1 if run[0] > 0 and panel[run[0] - 1] else run[-1] + 1
        if not (0 <= nb < y.size and panel[nb]):
            seg[run] = k
            k += 1
            continue
        s = -1.0 if y[nb] < 0.0 else 1.0
        ca, sa = np.cos(s * gam), np.sin(s * gam)
        dy, dz = y[run] - y[nb], z[run] - z[nb]
        out_y[run] = y[nb] + dy * ca - dz * sa
        out_z[run] = out_z[nb] + dy * sa + dz * ca
        seg[run] = k
        k += 1
    return out_y, out_z, c / np.cos(gam), mask, seg


def tail_path(tail: dict, sf: dict):
    """The spanwise path the tail would be BUILT on, or None for a flat tail
    with nothing out of plane. Returns (y, z, chord, is_winglet, seg).

    One place decides what the second surface IS, so the 3-D view, the front
    view and the STL cannot disagree about it — which they did, in opposite
    directions, on the one shape that is both: a V-tail carrying a tip device
    was drawn flat in 3-D (the device won) and without its device in the front
    view (the V won).
    """
    z0 = float(sf.get("z_offset", 0.0))
    gam = (float(tail.get("dihedral_deg") or 0.0)
           if tail.get("type") == "v_tail" else 0.0)
    own = nonplanar_arrays(sf)
    if own is not None:
        y, z, c, mask = own
        folded = fold_to_vee(y, z, c, mask, z0, gam) if gam > 0.0 else None
        return folded or (y, z, c, mask, mask.astype(int))
    vee = v_tail_path(tail, z0)
    if vee is None:
        return None
    y, z, c, side = vee
    # ``side`` is the root crease, NOT a tip device: handing it on as an
    # is-winglet flag drew the starboard panel in the device's colour and
    # labelled it "tail tip device" in the front view.
    return y, z, c, np.zeros(y.size, dtype=bool), side.astype(int)


def elevator_grid(y, c, x0, z0, frac, delta_deg, nrm=None, n: int = 9):
    """(X, Y, Z) of the deflected-elevator flat plate, hinged at (1 - frac) c.

    A schematic, deliberately: the physics models the elevator as a flap
    EFFECTIVENESS tau that converts delta_e into an equivalent tail incidence
    (``tail.flap_effectiveness``), never as a geometric hinge, so a contoured
    deflected section would imply more than was solved. What the plate does
    carry honestly is the hinge station and the sign and size of the deflection
    the trim landed on — delta_e > 0 is trailing edge DOWN.

    ``z0`` may be a per-station array and ``nrm`` the local (y, z) surface
    normal, which is how the plate follows a V-tail's dihedralled panels
    instead of hanging flat off them.
    """
    y = np.asarray(y, dtype=float)
    c = np.asarray(c, dtype=float)
    d = np.deg2rad(float(delta_deg))
    s_h = 0.75 - float(frac)              # hinge, in the c/4 frame (x aft +)
    t = np.linspace(0.0, float(frac), int(n))     # distance aft of the hinge
    X = x0 + s_h * c[:, None] + np.outer(c, t) * np.cos(d)
    drop = -np.outer(c, t) * np.sin(d)            # + delta_e = TE down
    z_base = np.asarray(z0, dtype=float)
    if z_base.ndim == 0:
        z_base = np.full(y.shape, float(z_base))
    if nrm is None:
        Z = z_base[:, None] + drop
        Y = np.repeat(y[:, None], t.size, axis=1)
    else:
        # offset along the panel's own normal: the plate stays ON the surface
        nrm = np.asarray(nrm, dtype=float)
        Y = y[:, None] + drop * nrm[:, 0][:, None]
        Z = z_base[:, None] + drop * nrm[:, 1][:, None]
    return X, Y, Z


# ------------------------------------------------------------- the surfaces

@dataclass(frozen=True)
class Surface:
    """One lofted surface as a structured grid, span-major.

    ``closed`` says whether the chordwise direction is a closed loop that can
    be turned into a watertight solid (a lofted wing) or an open sheet that can
    only ever be a two-sided membrane (the elevator plate).
    """

    name: str
    X: np.ndarray
    Y: np.ndarray
    Z: np.ndarray
    closed: bool = True
    y: np.ndarray | None = None     # the spanwise stations it was lofted on,
    #                                 kept so a caller can colour by a
    #                                 per-station quantity (local Cl) without
    #                                 re-deriving which station each row is
    #: WHAT this surface is, for consumers that must treat structure
    #: differently from a lifting surface. ``"lifting"`` is a wing, a tail, a
    #: fin or a tip device — anything with a section the solver flew and a
    #: local Cl to colour by. ``"structure"`` is a member that carries load
    #: and makes no lift the solver counted: today the MOUNT. A field rather
    #: than a name prefix, because a name is a second naming contract of
    #: exactly the kind FILE_SUFFIXES' own docstring warns about — and
    #: ``fig_sections_as_flown`` already filters by the literal string
    #: "elevator", which is what that warning looks like when it comes true.
    kind: str = "lifting"

    @property
    def shape(self) -> tuple:
        return tuple(self.X.shape)

    def points(self) -> np.ndarray:
        """(ns, nc, 3) grid of points in the package's axes."""
        return np.stack([self.X, self.Y, self.Z], axis=-1)


def tail_surfaces(tail: dict, sf: dict, xc, zc, name: str = "tail"
                  ) -> list[Surface]:
    """The tail that was flown: the panel(s) plus, for an elevator, the plate.

    Two shapes:

    * a path out of the tail's own plane — its own panel chain where it flies
      a tip device (``wingtail.py``), a V-tail's dihedralled panels, or BOTH
      folded together (:func:`tail_path`), so the device appears at its own
      height and cant on the panel it grows from;
    * a flat panel at the height and station the solver put it at.
    """
    x0 = float(sf.get("x_offset", 0.0))          # x aft +, as the solver says
    z0 = float(sf.get("z_offset", 0.0))
    stab = tail.get("control", "stabilator") == "stabilator"
    i_t = np.deg2rad(float(tail.get("i_t_deg", 0.0))) if stab else 0.0
    # a surface that pushes DOWN flies its section mounted upside down
    # (tail.tail_polar), so it is lofted upside down: the drawn shape is the
    # one the polar was read from, camber and all. Symmetric sections are
    # their own mirror, so this is invisible on one.
    if tail.get("section_inverted"):
        from .polar import invert_coords
        xz = invert_coords(np.column_stack([np.asarray(xc, dtype=float),
                                            np.asarray(zc, dtype=float)]))
        xc, zc = xz[:, 0], xz[:, 1]
    nrm = None
    path = tail_path(tail, sf)
    if path is not None:
        # a tail out of its own plane does not say how tall its device is, so
        # there is nothing to extend the stations to — left as flown
        y, z, c, _wl, seg = path
        tw = np.full(y.size, i_t)
        X, Y, Z = loft_path(y, z, c, tw, xc, zc, seg, x0=x0)
        _, nrm = path_normals(y, z, seg)
        z_ref = z
    else:
        y, c = planform_arrays({"y": sf.get("y"), "chord": sf.get("chord")})
        if y is None:
            return []
        # panel centres again: a flat tail's own span says where its edge
        # is, and without this the exported surface is short of the area
        # the drag was charged on (see extend_to_edges)
        b_t = float(tail.get("b_t") or 0.0)
        if b_t > 0.0:
            y, _z, c, _m = extend_to_edges(
                y, np.zeros_like(y), c, np.zeros(y.size, dtype=bool), b_t)
        tw = np.full_like(y, i_t)
        X, Y, Z = loft_surface(y, c, tw, xc, zc, x0=x0, z0=z0)
        z_ref = np.full(y.shape, z0)
    out = [Surface(name, X, Y, Z, closed=True, y=y)]
    frac = float(tail.get("elevator_chord_frac") or 0.0)
    if frac > 0.0 and tail.get("control") == "elevator":
        ex, ey, ez = elevator_grid(y, c, x0, z_ref, frac,
                                   float(tail.get("delta_e_deg", 0.0)),
                                   nrm=nrm)
        out.append(Surface("elevator", ex, ey, ez, closed=False, y=y))
    return out


def second_wing_twist(geom: dict, y, x_best=None, labels=None, b=None):
    """The REAR wing's own twist law sampled at |y| [rad].

    A tandem's two wings do not share a twist law — the rear one carries the
    pair's DECALAGE on top of its own root/tip law, which is the whole point
    of the variable. Read off the design vector by the rear wing's own label
    names; a vector that does not carry them (the front-only families) falls
    back to the wing's law, so nothing else changes.

    ``b`` is the span the twist is stretched over, and for this surface it is
    the REAR wing's: the two wings need not be the same width (the pair can
    fly a span per wing), and normalising the rear one's tip twist on the
    front one's span put the stated tip twist somewhere out past its tip.
    Left None it falls back to the report's ``b`` — the front wing's, which
    IS the rear wing's whenever the pair has one span.
    """
    lab = list(labels or [])
    if x_best is None or "twist_root_rear_deg" not in lab:
        return wing_twist(geom, y, x_best, labels)
    dec = (float(x_best[lab.index("decalage_deg")])
           if "decalage_deg" in lab else 0.0)
    tw_r = float(x_best[lab.index("twist_root_rear_deg")]) + dec
    tw_t = (float(x_best[lab.index("twist_tip_rear_deg")]) + dec
            if "twist_tip_rear_deg" in lab else tw_r)
    y = np.asarray(y, float)
    b_use = float(b or 0.0) or float(geom.get("b") or 0.0) \
        or (float(np.max(np.abs(y))) * 2.0) or 1.0
    eta = np.clip(np.abs(2.0 * y / b_use), 0.0, 1.0)
    return np.deg2rad(tw_r + (tw_t - tw_r) * eta)


def second_wing_surfaces(geom: dict, sf: dict, xc, zc, x_best=None,
                         labels=None) -> list[Surface]:
    """The SECOND WING of a tandem pair, lofted as its own surface.

    Same job :func:`tail_surfaces` does for a tail, and a separate function
    for the same reason its geometry block is a separate key: this surface is
    a WING. It has no trim incidence and no elevator plate, it carries its own
    twist law and decalage, and — the part that shows — it carries its own tip
    device, which has to be lofted along its own (y, z) path rather than
    inherited from the front wing's.
    """
    x0 = float(sf.get("x_offset", 0.0))
    z0 = float(sf.get("z_offset", 0.0))
    # the rear wing's OWN span where the report carries one: a pair may fly a
    # span per wing, and this surface's twist law belongs to this surface
    b_r = sf.get("b")
    own = nonplanar_arrays(sf)
    if own is not None:
        y, z, c, mask = own
        tw = second_wing_twist(geom, y, x_best, labels, b=b_r)
        X, Y, Z = loft_path(y, z, c, tw, xc, zc, mask,
                            x0=x0 + sweep_offset(geom, y))
        return [Surface("rear", X, Y, Z, closed=True, y=y)]
    y, c = planform_arrays({"y": sf.get("y"), "chord": sf.get("chord")})
    if y is None:
        return []
    tw = second_wing_twist(geom, y, x_best, labels, b=b_r)
    # THE PAIR'S SWEEP IS THE PAIR'S: ``tandemvlm`` builds both wings from one
    # ``wing_sweep_deg``, so the rear surface is drawn swept exactly as the
    # front one is, on top of its own stagger.
    X, Y, Z = loft_surface(y, c, tw, xc, zc, x0=x0 + sweep_offset(geom, y),
                           z0=z0)
    return [Surface("rear", X, Y, Z, closed=True, y=y)]


def mount_surfaces(geom: dict, x_best=None, labels=None,
                   n_chord: int = 31, n_z: int = 2) -> list[Surface]:
    """The MOUNT that was flown, as drawable prisms — one per support station.

    ``carmount.py`` has priced a strut-borne car wing since it was written and
    no figure has ever drawn one: the picture showed a wing floating over the
    track with the single body that distinguishes a swan-neck layout from a
    plate-borne one invisible. This closes that, and it draws only what the
    solver actually solved:

    * WHERE — ``geom["mount"]["stations_m"]``, which is
      ``MountSpec.support_stations_m``: the stations the beam is supported at,
      not the suction-side loss footprint (empty on a swan neck, which is the
      layout most worth seeing).
    * HOW LONG — ``length_m`` = ride minus deck height. A pylon reaches the
      car's bodywork, not the track, so it stops short of the ground plane and
      the gap below it is real, not a drawing error.
    * WHERE ON THE CHORD — ``x_attach_frac``, the number that decides whether
      the wing winds up towards more downforce with speed or sheds incidence.
      A mount drawn at the quarter chord when the run flew it aft would hide
      exactly the geometry that divergence argument is about.

    An ENDPLATE mount returns nothing: its load path is the plate, which is
    already in the picture as the wing's own outboard panels. Returning a
    body for it would draw a second plate that no solver flew.

    The strut is the SECTION its drag was charged at, not a box. It used to
    be a five-point rectangle — a square-edged slab with sharp corners — while
    the very model that priced it multiplies by ``drag.wing_form_factor(t/c)``
    (``carmount.pylon_parasite_cd``'s build-up branch, FF = 1.26 at t/c 0.12).
    A form factor is a statement that the body is streamlined; the picture
    said it was not. It is now lofted the way :func:`fin_surface` lofts a fin,
    from the same symmetric NACA stand-in every other surface falls back to,
    at the strut's own ``tc``.

    SYMMETRIC, and that is load-bearing twice: a strut at zero yaw must make
    no side force (the argument :func:`fin_surface` writes out in full), and a
    section symmetric in y puts the body's y-midpoint exactly on its station,
    which is how a picture can be checked against the beam it was drawn from.

    It starts at the wing's SKIN, not at its chord line. The chord line is
    inside the solid, so a prism starting there buried part of its own length
    in the wing: measured at the box centre the wing's skin reaches
    z = 0.01516 m over the strut's x band, which is 12.1 % of the 0.125 m
    strut the ride band used to average to and 8.7 % of the 0.175 m one it
    averages to now. STL has no boolean union, so that overlap shipped as two
    interpenetrating closed shells — legal to every reader and wrong to every
    mesher. The body is TRANSLATED, never stretched: its z extent is still
    exactly ``length_m``, which is the number the beam is solved on.
    """
    mount = geom.get("mount") or {}
    if str(mount.get("kind")) != "pylon":
        return []
    stations = [float(v) for v in (mount.get("stations_m") or ())]
    length = float(mount.get("length_m", 0.0) or 0.0)
    if not stations or length <= 0.0:
        return []

    chord_m = float(mount.get("chord_m", 0.0) or 0.0)
    tc = float(mount.get("tc", 0.0) or 0.0)
    frac = float(mount.get("x_attach_frac", 0.25))
    y_w, c_w = planform_arrays(geom, x_best, labels)
    z_w = np.asarray(geom.get("z") if geom.get("z") is not None else [],
                     dtype=float)
    y_all = np.asarray(geom.get("y") if geom.get("y") is not None else [],
                       dtype=float)
    if not (chord_m > 0.0 and tc > 0.0):
        return []

    # the strut's own section, in the (x, y) plane: x is the chord, y carries
    # the thickness. MID-chord at 0, because the mount LINE is what the
    # torsion model takes its moment about (carmount eq. 2) and what
    # ``x_attach_frac`` names.
    xs, zs = naca4_section(tc, m=0.0, p=0.4, n=n_chord)
    sx = xs - 0.5
    # cosine stations along the member. Two is the prism, and two is what
    # this draws today; the array is the door to a real routed path later
    # without a second rewrite of every consumer.
    n_z = max(2, int(n_z))
    fz = 0.5 * (1.0 - np.cos(np.pi * np.arange(n_z) / (n_z - 1)))
    xc_w, zc_w = section_path(geom, x_best, labels, None, n=61)
    xc_w = np.asarray(xc_w, dtype=float)
    zc_w = np.asarray(zc_w, dtype=float)

    out: list[Surface] = []
    for i, y_s in enumerate(stations):
        # the wing AT THIS STATION: its chord sets where the attachment sits
        # along x, and its own z is where the strut has to start — a station
        # inside a blend is already off the wing plane
        c_s = (float(np.interp(y_s, y_w, c_w))
               if y_w is not None and len(y_w) else 0.0)
        z_ch = (float(np.interp(y_s, y_all, z_w))
                if y_all.size and y_all.size == z_w.size else 0.0)
        # x aft +, quarter chord at 0 (loft_surface's own convention), so the
        # attachment is (frac - 0.25) of the local chord
        x_mid = (frac - 0.25) * c_s
        # ...and the SKIN the strut leaves from, taken over the strut's own
        # chordwise footprint rather than at a point: the deepest part of the
        # wing over that band is what a member has to clear. +z is the model
        # frame's track direction and the car's DOWN (carwing.py's frame
        # statement), and the deck sits at +length with the track image plane
        # beyond it — so both the strut and the endplates hang the same way.
        z0 = z_ch
        if c_s > 0.0:
            # over the strut's own FOOTPRINT in both directions, not at its
            # centreline: the wing's own z moves across the strut's thickness
            # wherever the surface is not flat, and a centreline-only start
            # left 4.6e-05 m of the outboard edge still inside the solid
            # (measured). Three stations span a monotone blend exactly.
            half_t = 0.5 * chord_m * tc
            for y_f in (y_s - half_t, y_s, y_s + half_t):
                c_f = (float(np.interp(y_f, y_w, c_w))
                       if y_w is not None and len(y_w) else c_s)
                z_f = (float(np.interp(y_f, y_all, z_w))
                       if y_all.size and y_all.size == z_w.size else z_ch)
                xw = (xc_w - 0.25) * c_f
                zw = zc_w * c_f + z_f
                band = ((xw >= x_mid - 0.5 * chord_m)
                        & (xw <= x_mid + 0.5 * chord_m))
                if band.any():
                    z0 = max(z0, float(zw[band].max()))
        z = z0 + length * fz
        X = np.outer(np.ones_like(z), sx * chord_m) + x_mid
        Y = np.outer(np.ones_like(z), zs * chord_m) + y_s
        Z = np.repeat(z[:, None], xs.size, axis=1)
        out.append(Surface(f"mount{i}", X, Y, Z, closed=True, y=z,
                           kind="structure"))
    return out


def fin_surface(geom: dict, xc, zc, n_span: int = 21) -> list["Surface"]:
    """The VERTICAL surface, lofted from ``geometry["fin"]``.

    A fin is this package's own wing loft turned ninety degrees about x: its
    span runs along z, its chord along x, and its thickness — the section's
    own ``zc`` — along y. So the same ``(xc, zc)`` section every other
    surface is lofted from is reused rather than a plate being drawn.

    ``height`` is SIGNED, and a ventral fin (negative) is drawn hanging DOWN
    from ``z_root`` by writing its stations in that direction. Nothing else
    changes: the grid is span-major like every other :class:`Surface`, so the
    STL triangulation and the VSP writer take it without a special case.

    Empty list when the report states no fin, which is how a V-tail — and any
    report written before the fin block existed — exports without one rather
    than with an invented one.
    """
    blk = (geom or {}).get("fin")
    if not blk:
        return []
    c = float(blk["chord_m"])
    h = float(blk["height_m"])
    z0 = float(blk.get("z_root_m", 0.0))
    x_qc = float(blk["x_qc_m"])
    # A FIN IS SYMMETRIC, and the wing's section is not. Lofting the fin from
    # the wing's ``(xc, zc)`` draws a CAMBERED fin — a surface that flies a
    # permanent side load, which is exactly what ``vlm.VerticalSurface``'s
    # ``alpha_L0 = 0`` says it does not. So it is drawn at the thickness its
    # own drag was charged at (``fin.FIN_TC_DEFAULT``, carried in the block),
    # with no camber, from the same NACA stand-in every other surface falls
    # back to. ``xc``/``zc`` are ignored on purpose and kept in the signature
    # because a fin WILL get its own exported section.
    xc, zc = naca4_section(float(blk.get("tc", 0.10)), m=0.0, p=0.4,
                           n=(np.asarray(xc).size + 1) // 2)
    # cosine spanwise stations, clustered at the root and tip exactly as the
    # lifting surfaces are, so a mixed export has one refinement rule
    frac = 0.5 * (1.0 - np.cos(np.pi * np.arange(n_span) / (n_span - 1)))
    z = z0 + h * frac
    s = xc - 0.25                      # c/4 at 0, LE at -0.25 c (loft_surface)
    X = np.outer(np.ones_like(z), s * c) + x_qc
    Y = np.outer(np.ones_like(z), zc * c)      # thickness runs in y
    Z = np.repeat(z[:, None], xc.size, axis=1)
    return [Surface("fin", X, Y, Z, closed=True, y=z)]


def surfaces(geom: dict, x_best=None, labels=None, section=None,
             section_aft=None, section_n: int = 61) -> list[Surface]:
    """Every lofted surface in a design-report geometry dict.

    Mirrors what the 3-D view draws, in the same three cases:

    1. a NONPLANAR solve (the VLM winglet modes) — wing and tip device are ONE
       grid lofted along the solver's own (y, z) path, so a blended transition
       exports without an artificial crease at the junction;
    2. a two-surface result (tandem, or wing + tail) — each surface at its own
       streamwise and vertical offset;
    3. a plain planar wing.

    A SECOND SURFACE solved in the same panel array (``wingtail.py``, and the
    hydrofoil's elevator in ``hydrotail.py``) leaves the report as its own
    ``tail_surface`` block, so it is appended to cases 1 and 3 alike — the
    hydrofoil is planar and still flies an elevator.

    ``section_aft`` is the second surface's own section when it flies one (the
    tail can be given a different section from the wing); it falls back to the
    wing's.
    """
    xc, zc = section_path(geom, x_best, labels, section, n=section_n)
    xa, za = ((xc, zc) if section_aft is None
              else section_path(geom, x_best, labels, section_aft,
                                n=section_n))
    out: list[Surface] = []
    tail = geom.get("tail") or {}
    tail_sf = geom.get("tail_surface")
    second_sf = geom.get("second_surface")

    np_geom = nonplanar_arrays(geom) or canted_arrays(geom)
    if np_geom is not None:
        y, z, c, mask, twist = wing_path(geom, x_best, labels)
        # a blended junction has no crease — see path_normals. Either side of
        # the transition counts: a wing-side blend leaves the wing plane before
        # the tip, so segmenting the normals at the junction would build a kink
        # geometry.span_path did not put there.
        wl = geom.get("winglet") or {}
        smooth = (float(wl.get("blend_frac", 0.0) or 0.0) > 0.0
                  or float(wl.get("wing_blend_frac", 0.0) or 0.0) > 0.0)
        X, Y, Z = loft_path(y, z, c, twist, xc, zc, mask, smooth=smooth,
                            x0=sweep_offset(geom, y))
        out.append(Surface("wing", X, Y, Z, closed=True, y=y))
        if second_sf:
            out += second_wing_surfaces(geom, second_sf, xa, za, x_best,
                                        labels)
        if tail_sf:
            out += tail_surfaces(tail, tail_sf, xa, za)
        out += fin_surface(geom, xc, zc)
        out += mount_surfaces(geom, x_best, labels)
        return out

    surfs = geom.get("surfaces")
    if surfs:
        for sf in surfs:
            nm = str(sf.get("name") or "surface")
            if nm == "tail":
                out += tail_surfaces(tail, sf, xa, za)
                continue
            y, c = planform_arrays({"y": sf.get("y"), "chord": sf.get("chord")})
            if y is None:
                continue
            sec_x, sec_z = (xa, za) if nm in ("rear", "tail") else (xc, zc)
            tw = wing_twist(geom, y, x_best, labels)
            # ...at the sweep the surface FLIES. Both wings of a pair carry
            # the pair's own quarter-chord sweep (``tandemvlm`` builds them
            # from one ``wing_sweep_deg``), so it is added to each surface's
            # own stagger rather than to the front wing alone.
            X, Y, Z = loft_surface(y, c, tw, sec_x, sec_z,
                                   x0=(float(sf.get("x_offset", 0.0))
                                       + sweep_offset(geom, y)),
                                   z0=float(sf.get("z_offset", 0.0)))
            out.append(Surface(nm, X, Y, Z, closed=True, y=y))
        out += fin_surface(geom, xc, zc)
        out += mount_surfaces(geom, x_best, labels)
        return out

    got = wing_path(geom, x_best, labels)
    if got is None:
        return out
    y, _z, c, _m, twist = got
    X, Y, Z = loft_surface(y, c, twist, xc, zc, x0=sweep_offset(geom, y))
    out.append(Surface("wing", X, Y, Z, closed=True, y=y))
    if second_sf:
        out += second_wing_surfaces(geom, second_sf, xa, za, x_best, labels)
    if tail_sf:
        out += tail_surfaces(tail, tail_sf, xa, za)
    out += fin_surface(geom, xc, zc)
    # THE MOUNT IS A PART OF THE CAR, so it leaves with the rest of it. It was
    # drawn in exactly one plotly view and exported nowhere, which meant the
    # STL and the OpenVSP script both shipped a rear wing floating over a
    # track with nothing holding it up — while ``mount_surfaces``' own
    # docstring claimed the body was "watertight and could be exported
    # as-is". Appended in ALL THREE branches: a car with plates takes the
    # nonplanar one, and the SAME car with the plate height pinned to zero
    # falls through to this planar one with its mount block intact, so
    # appending to one would lose the strut on exactly the layout that has
    # nothing else holding the wing. Every other geometry gets [] back.
    out += mount_surfaces(geom, x_best, labels)
    return out


# ------------------------------------------------------------ triangulation

def _drop_seam(P: np.ndarray) -> np.ndarray:
    """Drop a chordwise column that repeats the first one.

    A closed section arrives either with the trailing-edge point written twice
    (a sharp TE: NACA, or CST at dz_te = 0) or with two distinct TE points (an
    open TE). Removing the repeat lets the wrap-around quad below be the ONE
    rule that closes the loop in both cases — stitching a finite TE gap in the
    second, and drawing nothing degenerate in the first.
    """
    if P.shape[1] >= 3 and np.allclose(P[:, 0], P[:, -1], atol=SEAM_TOL):
        return P[:, :-1]
    return P


def _fan(ring: np.ndarray, flip: bool) -> np.ndarray:
    """Centroid fan over one closed ring of points -> (n, 3, 3) triangles."""
    cen = ring.mean(axis=0)
    j = np.arange(ring.shape[0])
    j1 = (j + 1) % ring.shape[0]
    a = np.repeat(cen[None, :], ring.shape[0], axis=0)
    b, c = ring[j], ring[j1]
    return np.stack([a, c, b] if flip else [a, b, c], axis=-2)


def triangles(surf: Surface, cap: bool = True) -> np.ndarray:
    """(n, 3, 3) triangles for one surface, wound so the normals point OUT.

    Closed lofts are wrapped in the chordwise direction (the trailing-edge
    stitch) and capped at both span ends with a centroid fan, which is what
    makes each solid watertight. Winding is built consistently from the grid
    and then flipped as a whole if the enclosed signed volume comes out
    negative — a global test, so it cannot leave one facet inconsistent with
    its neighbour. Open sheets (the elevator plate) get neither wrap, cap nor
    flip: they enclose nothing, and a membrane's "outward" is not defined.
    """
    P = surf.points()
    if P.shape[0] < 2 or P.shape[1] < 2:
        return np.zeros((0, 3, 3))
    wrap = bool(surf.closed)
    if wrap:
        P = _drop_seam(P)
    ns, nc = P.shape[0], P.shape[1]
    nj = nc if wrap else nc - 1
    if nj < 1:
        return np.zeros((0, 3, 3))
    ii, jj = np.meshgrid(np.arange(ns - 1), np.arange(nj), indexing="ij")
    j1 = (jj + 1) % nc
    a, b = P[ii, jj], P[ii + 1, jj]
    c, d = P[ii + 1, j1], P[ii, j1]
    tris = [np.stack([a, b, c], axis=-2).reshape(-1, 3, 3),
            np.stack([a, c, d], axis=-2).reshape(-1, 3, 3)]
    if wrap and cap and nc >= 3:
        # the two caps are wound OPPOSITE to each other and consistently with
        # the side quads above; the global flip below then orients everything
        tris.append(_fan(P[0], flip=False))
        tris.append(_fan(P[-1], flip=True))
    T = np.concatenate(tris, axis=0)
    T = T[_facet_area(T) > MIN_FACET_AREA]
    if wrap and cap and T.shape[0]:
        # signed volume of a closed, consistently wound mesh: sum a.(b x c)/6
        vol = float(np.einsum("ij,ij->i",
                              T[:, 0], np.cross(T[:, 1], T[:, 2])).sum() / 6.0)
        if vol < 0.0:
            T = T[:, ::-1, :]
    return T


def _facet_area(T: np.ndarray) -> np.ndarray:
    n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    return 0.5 * np.linalg.norm(n, axis=1)


def _facet_normals(T: np.ndarray) -> np.ndarray:
    n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    mag = np.linalg.norm(n, axis=1, keepdims=True)
    mag[mag == 0.0] = 1.0
    return n / mag


# --------------------------------------------------------------------- STL

def stl_bytes(surfs, binary: bool = True, name: str = "aerobo", cap: bool = True):
    """The surfaces as one STL — ``bytes`` if binary, ``str`` if ASCII.

    Binary STL has no notion of parts, so everything lands in one blob (write
    per-part files through :func:`export` when the parts must stay apart in the
    importer). ASCII STL does: each surface becomes its own named ``solid``,
    which is how "wing", "tail" and "elevator" stay legible after import.
    """
    parts = [(s.name, triangles(s, cap=cap)) for s in surfs]
    parts = [(nm, T) for nm, T in parts if T.shape[0]]
    if not binary:
        lines = []
        for nm, T in parts:
            N = _facet_normals(T)
            lines.append(f"solid {nm}")
            for tri, nvec in zip(T, N):
                lines.append("  facet normal "
                             f"{nvec[0]:.6e} {nvec[1]:.6e} {nvec[2]:.6e}")
                lines.append("    outer loop")
                for p in tri:
                    lines.append(f"      vertex {p[0]:.6e} {p[1]:.6e} "
                                 f"{p[2]:.6e}")
                lines.append("    endloop")
                lines.append("  endfacet")
            lines.append(f"endsolid {nm}")
        return "\n".join(lines) + "\n"
    T = (np.concatenate([t for _, t in parts], axis=0) if parts
         else np.zeros((0, 3, 3)))
    N = _facet_normals(T) if T.shape[0] else np.zeros((0, 3))
    head = f"aerobo cad export — {name} — x aft, y starboard, z up, metres"
    buf = bytearray(head.encode("ascii", "replace")[:80].ljust(80, b" "))
    buf += struct.pack("<I", T.shape[0])
    rows = np.concatenate([N, T.reshape(-1, 9)], axis=1).astype("<f4")
    for row in rows:
        buf += row.tobytes()
        buf += b"\x00\x00"
    return bytes(buf)


def write_stl(path, surfs, binary: bool = True, name: str = "aerobo") -> Path:
    """Write one STL file; returns the path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = stl_bytes(surfs, binary=binary, name=name)
    if isinstance(data, bytes):
        p.write_bytes(data)
    else:
        p.write_text(data)
    return p


# ----------------------------------------------------------- airfoil / .dat

def airfoil_dat(xc, zc, name: str = "aerobo section") -> str:
    """Selig/XFOIL-order .dat text for a unit-chord section (TE -> LE -> TE)."""
    xc = np.asarray(xc, float)
    zc = np.asarray(zc, float)
    rows = "\n".join(f"{a:12.7f} {b:12.7f}" for a, b in zip(xc, zc))
    return f"{name}\n{rows}\n"


# ------------------------------------------------- native OpenVSP rebuild

def wing_sections(geom: dict, x_best=None, labels=None, max_sections: int = 12
                  ) -> list[dict]:
    """The starboard half of the wing reduced to OpenVSP wing SECTIONS.

    OpenVSP builds a wing as a chain of sections, each with its own span, root
    and tip chord, twist and dihedral, lofted LINEARLY between them. The
    solver's geometry is a per-panel polyline, so this walks the starboard half
    in developed arc length and picks break stations — always including the
    wing/winglet junction, so the dihedral change lands on a section boundary
    rather than being averaged across one.

    Each entry carries ``span`` (arc length of the section, which is what VSP's
    Span parm means once the dihedral is applied), ``root_chord``/``tip_chord``,
    ``twist_deg`` at its outboard station, ``dihedral_deg`` from the path, and
    ``is_winglet``. ``chord_error`` on the first entry reports the largest
    chord the linear rebuild misses by, as a fraction — the honest measure of
    how much of a polynomial chord law survives the reduction.
    """
    got = wing_path(geom, x_best, labels)
    if got is None:
        return []
    y, z, c, mask, twist = got
    # starboard half, in path order (the path runs port tip -> starboard tip)
    i0 = int(np.argmin(np.abs(y)))
    y, z, c, mask = y[i0:], z[i0:], c[i0:], mask[i0:]
    if y.size < 2:
        return []
    tw = np.rad2deg(twist[i0:])
    # ...starting ON the plane of symmetry. The innermost station is half a
    # panel outboard of it (the centreline is a panel EDGE), and VSP mirrors
    # whatever it is given: leaving the chain there built a wing 0.39 m
    # narrower than the design, and a tail 6.7 % short of its area. Measured,
    # not reasoned — that is what the round trip through OpenVSP 3.51.2 is for.
    y, z, c, mask, tw = _rooted(y, z, c, mask, tw)
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(y), np.diff(z)))])
    if s[-1] <= 0.0:
        return []

    # break stations: the ends, BOTH stations either side of every mask change,
    # then arc-length fill. Both, because the junction is a panel EDGE and the
    # stations are panel centres: breaking on one side only would let a section
    # run from mid-wing across the corner, and a linear loft would then cut the
    # corner off. Isolating it keeps the error inside half a panel.
    breaks = {0, y.size - 1}
    for i in np.flatnonzero(np.diff(mask.astype(int))):
        breaks |= {int(i), int(i) + 1}
    n_fill = max(int(max_sections) - len(breaks), 0)
    if n_fill:
        for target in np.linspace(0.0, s[-1], n_fill + 2)[1:-1]:
            breaks.add(int(np.argmin(np.abs(s - target))))
    idx = sorted(breaks)

    out: list[dict] = []
    for a, b in zip(idx[:-1], idx[1:]):
        span = float(s[b] - s[a])
        if span <= 1e-9:
            continue
        dy, dz = float(y[b] - y[a]), float(z[b] - z[a])
        out.append({
            "span": span,
            "root_chord": float(c[a]),
            "tip_chord": float(c[b]),
            "twist_deg": float(tw[b]),
            "dihedral_deg": float(np.rad2deg(np.arctan2(dz, dy))),
            "is_winglet": bool(mask[b]),
        })
    if out:
        # what the linear rebuild costs: compare the piecewise-linear chord
        # through the break stations against every panel chord it skipped
        c_lin = np.interp(s, s[idx], c[idx])
        cmax = float(np.max(c)) or 1.0
        out[0]["chord_error"] = float(np.max(np.abs(c_lin - c)) / cmax)
        out[0]["root_twist_deg"] = float(tw[0])
    return out


def _tail_sections(geom: dict) -> list[dict]:
    """The tail's own starboard sections, or [] when no tail was flown."""
    tail = geom.get("tail") or {}
    # the SECOND surface, under whichever key it left the report by: a tail
    # solved with the wing, a tandem's rear WING, or the published pair's
    # two-entry list. One surface, three exporters — see api.py's splitters.
    sf = geom.get("tail_surface") or geom.get("second_surface")
    if not sf:
        for s in (geom.get("surfaces") or []):
            if s.get("name") in ("tail", "rear"):
                sf = s
                break
    if not sf:
        return []
    y, c = planform_arrays({"y": sf.get("y"), "chord": sf.get("chord")})
    if y is None:
        return []
    z0 = float(sf.get("z_offset", 0.0))
    vee = v_tail_path(tail, z0)
    if vee is not None:
        y, _z, c, _m = vee
    b_t = float(tail.get("b_t") or 0.0)
    if vee is None and b_t > 0.0:      # stations are centres — reach the tip
        y, _z, c, _m = extend_to_edges(y, np.zeros_like(y), c,
                                       np.zeros(y.size, dtype=bool), b_t)
    half = y >= 0.0
    y, c = y[half], c[half]
    if y.size < 2:
        return []
    y, _z, c, _m, _t = _rooted(y, np.zeros_like(y), c,
                               np.zeros(y.size, dtype=bool),
                               np.zeros(y.size))
    gam = float(tail.get("dihedral_deg") or 0.0) \
        if tail.get("type") == "v_tail" else 0.0
    i_t = (float(tail.get("i_t_deg", 0.0))
           if tail.get("control", "stabilator") == "stabilator" else 0.0)
    span = float(y[-1] - y[0]) / float(max(np.cos(np.deg2rad(gam)), 1e-9))
    return [{"span": span,
             "root_chord": float(c[0]), "tip_chord": float(c[-1]),
             "twist_deg": i_t, "dihedral_deg": gam, "is_winglet": False,
             "root_twist_deg": i_t}]


def _fin_sections(geom: dict) -> list[dict]:
    """The VERTICAL surface as one OpenVSP wing section, stood on edge.

    A fin IS a wing in OpenVSP; what makes it vertical is a dihedral of 90
    degrees (-90 for a ventral one, which is how a hanging surface is asked
    for everywhere else in this package). Reusing ``build_wing`` rather than
    adding a second geometry path is the same decision ``vlm`` made when it
    ran a lifting surface's bound segments along z.

    ``[]`` when the report states no fin — a V-tail, or a report older than
    the fin block — so the script builds no surface rather than an invented
    one.
    """
    blk = (geom or {}).get("fin")
    if not blk:
        return []
    c = float(blk["chord_m"])
    h = float(blk["height_m"])
    return [{"span": abs(h),
             "root_chord": c, "tip_chord": c,
             "twist_deg": 0.0,
             "dihedral_deg": 90.0 if h >= 0.0 else -90.0,
             "is_winglet": False, "root_twist_deg": 0.0}]


def _mount_sections(geom: dict, x_best=None, labels=None) -> list[dict]:
    """The MOUNT as one OpenVSP wing section, stood on edge.

    Same trick as :func:`_fin_sections`: a strut IS a wing in OpenVSP and what
    makes it vertical is a dihedral of 90. It is emitted for the STARBOARD
    station only, because :func:`wing_sections` already relies on VSP mirroring
    a wing geom about the XZ plane (its own docstring: "VSP mirrors whatever it
    is given") and a pair of struts is exactly that mirror. Emitting both would
    build four.

    ``[]`` for a plate-borne mount, for a report with no mount block, and for
    a mount on the centreline — where the mirror would put a second strut
    inside the first rather than beside it, and the pair is one member anyway.

    The geometry is read off :func:`mount_surfaces`, not re-derived, so the
    native rebuild and the exported STL cannot disagree about where the strut
    starts or how long it is. That disagreement is the reason this exists:
    ``vsp_script`` built wing, tail and fin natively while its own
    ``import_mesh`` fallback read an STL that (now) carries the struts, so the
    same script produced a car with or without pylons depending on which path
    it took.
    """
    bodies = mount_surfaces(geom, x_best, labels)
    if not bodies:
        return []
    star = [b for b in bodies if float(b.Y.mean()) > 1e-9]
    if not star:
        return []
    b = max(star, key=lambda s: float(s.Y.mean()))
    chord = float(b.X.max() - b.X.min())
    span = float(b.Z.max() - b.Z.min())
    if not (chord > 0.0 and span > 0.0):
        return []
    return [{"span": span, "root_chord": chord, "tip_chord": chord,
             "twist_deg": 0.0, "dihedral_deg": 90.0,
             "is_winglet": False, "root_twist_deg": 0.0,
             # where VSP has to put it: the LEADING EDGE of the root section
             "x_le_m": float(b.X.min()),
             "y_m": float(0.5 * (b.Y.min() + b.Y.max())),
             "z_root_m": float(b.Z.min())}]


_VSP_HEADER = '''"""OpenVSP model of an aerobo design — GENERATED, safe to edit.

    python {stem}_vsp.py          # writes {stem}.vsp3 next to this file

Needs the OpenVSP Python API (``pip install openvsp``, or the API shipped in
the OpenVSP install's python/ directory). Axes match: OpenVSP and aerobo both
use x aft, y starboard, z up, and the file is in metres.

What this rebuild is, and is not:

* the wing is a NATIVE WingGeom — sections carry the solver's own chord,
  twist and dihedral, so VSPAERO's vortex lattice will run on it. The chord
  law is lofted LINEARLY between {n_sec} sections; the largest chord that
  misses by is {chord_err:.2%} of the root chord.
* the quarter-chord line carries {sweep:+.2f} deg of sweep{sweep_note}. On a
  LATTICE-backed design that is geometry the solver actually flew (``vlm.VLM``
  bends the line at ``xe = |ye| tan Lambda``); on a REDUCED-ORDER one sweep
  enters the section slope and the drag build-up only, its line really is
  straight, and this writes 0 rather than exporting a wing that was never
  flown. ``SWEEP_DEG`` below overrides either way.
* an elevator is a CONTROL SUB-SURFACE at the hinge station. aerobo trims it
  through a flap effectiveness tau, not a geometric hinge, so the deflection
  ({delta_e:+.2f} deg) is written as a comment and a VSPAERO control group,
  not baked into the surface.
* dihedral and twist conventions differ between OpenVSP versions (absolute vs
  relative per section). Every parm below is set through ``setp``, which warns
  instead of raising if a name is not present, so check the wing in the GUI
  once before trusting a sweep built on it.

If the parametric rebuild fails, ``main`` falls back to importing {stem}.stl,
which is the same geometry as a mesh.
"""
'''


_SEC_KEYS = ("span", "root_chord", "tip_chord", "twist_deg", "dihedral_deg")


def _sections_literal(name: str, secs: list) -> str:
    """A section list as a readable, dependency-free Python literal.

    Every value goes through ``float``/``bool`` first: a numpy scalar's repr is
    ``np.float64(0.4)``, which would make the generated script import numpy to
    read its own data — and fail with a NameError if it did not.
    """
    if not secs:
        return f"{name} = []"
    rows = []
    for s in secs:
        fields = ", ".join(f"{k}={float(s[k])!r}" for k in _SEC_KEYS)
        rows.append(f"    dict({fields}, "
                    f"is_winglet={bool(s.get('is_winglet', False))!r}),")
    return f"{name} = [\n" + "\n".join(rows) + "\n]"


def vsp_script(geom: dict, x_best=None, labels=None, stem: str = "aerobo",
               max_sections: int = 12) -> str:
    """A standalone OpenVSP API script that rebuilds this design natively."""
    wing = wing_sections(geom, x_best, labels, max_sections=max_sections)
    tail_sec = _tail_sections(geom)
    tail = geom.get("tail") or {}
    sf = geom.get("tail_surface") or geom.get("second_surface") or next(
        (s for s in (geom.get("surfaces") or [])
         if s.get("name") in ("tail", "rear")), {})
    fin_sec = _fin_sections(geom)
    _finblk = geom.get("fin") or {}
    fin_x = (float(_finblk.get("x_le_m", 0.0)) if fin_sec else 0.0)
    fin_z = (float(_finblk.get("z_root_m", 0.0)) if fin_sec else 0.0)
    mount_sec = _mount_sections(geom, x_best, labels)
    _mnt = mount_sec[0] if mount_sec else {}
    mount_x = float(_mnt.get("x_le_m", 0.0))
    mount_y = float(_mnt.get("y_m", 0.0))
    mount_z = float(_mnt.get("z_root_m", 0.0))
    # THE SWEEP THE SOLVER FLEW, by the same discriminator the loft uses
    # (``sweep_offset``): a panel path in the report means a lattice bent the
    # quarter-chord line, and this file draws what was flown. Without one the
    # angle belongs to a reduced-order build-up and the line stays straight.
    sweep_stated = float(geom.get("sweep_deg") or 0.0)
    sweep = sweep_stated if geom.get("z") is not None else 0.0
    head = _VSP_HEADER.format(
        stem=stem, n_sec=len(wing) or 1,
        chord_err=float(wing[0].get("chord_error", 0.0)) if wing else 0.0,
        sweep=sweep,
        sweep_note=(f", which is the {sweep_stated:+.2f} deg this design "
                    f"states" if abs(sweep) > 1e-9 else
                    (f" (the design states {sweep_stated:+.2f} deg, and it is "
                     f"not in its geometry — see below)"
                     if abs(sweep_stated) > 1e-9 else "")),
        delta_e=float(tail.get("delta_e_deg", 0.0)))

    body = f'''
import os
import sys

try:
    import openvsp as vsp
except ImportError:                       # older installs ship it as ``vsp``
    try:
        import vsp                        # noqa: F401
    except ImportError:
        sys.exit("OpenVSP Python API not found — pip install openvsp")

HERE = os.path.dirname(os.path.abspath(__file__))
STEM = {stem!r}
SWEEP_DEG = {sweep!r}        # the design's own, where it is in the geometry
SPAN_PANELS = 30                # spanwise panels shared out over the sections
#
# 30, not more: VSPAERO's induced drag on a THICK lofted surface drifts with
# spanwise refinement. Measured on a clean single-section trapezoid built
# inside OpenVSP 3.51.2 itself (nothing to do with this export) —
#     panels   10      20      40      80
#     e        0.985   0.990   0.927   0.762
# so a finer lattice is not a better answer here. Raise it only alongside
# VSPAERO's wake/core settings, and check the trend before trusting it.
DEVICE_PANELS = 8               # floor for a tip device, however short it is
AIRFOIL = os.path.join(HERE, STEM + "_wing.dat")
AIRFOIL_AFT = os.path.join(HERE, STEM + "_tail.dat")
AIRFOIL_FIN = os.path.join(HERE, STEM + "_fin.dat")

{_sections_literal("WING_SECTIONS", wing)}
WING_ROOT_TWIST_DEG = {float(wing[0].get("root_twist_deg", 0.0)) if wing else 0.0!r}
{_sections_literal("TAIL_SECTIONS", tail_sec)}
TAIL_ROOT_TWIST_DEG = {float(tail_sec[0].get("root_twist_deg", 0.0)) if tail_sec else 0.0!r}
TAIL_X = {float(sf.get("x_offset", 0.0))!r}          # aft of the wing c/4 [m]
TAIL_Z = {float(sf.get("z_offset", 0.0))!r}          # above the wing plane [m]
ELEVATOR_CHORD_FRAC = {float(tail.get("elevator_chord_frac") or 0.0)!r}
ELEVATOR_DEFLECTION_DEG = {float(tail.get("delta_e_deg", 0.0))!r}
{_sections_literal("FIN_SECTIONS", fin_sec)}
# the fin's LEADING EDGE is what a drawing states; VSP places a wing by its
# root leading edge, so this is x_qc - c/4 and no conversion is left to do
FIN_X = {fin_x!r}                 # leading edge, aft of the wing c/4 [m]
FIN_Z = {fin_z!r}                 # root, above the wing plane [m]
{_sections_literal("MOUNT_SECTIONS", mount_sec)}
# THE MOUNT, for the STARBOARD station only: VSP mirrors a wing geom about
# XZ, which is the same fact WING_SECTIONS relies on, so this builds the pair.
MOUNT_X = {mount_x!r}               # root leading edge, aft of the wing c/4 [m]
MOUNT_Y = {mount_y!r}               # the station the beam is supported at [m]
MOUNT_Z = {mount_z!r}               # where it leaves the wing SKIN [m]


def setp(gid, name, group, value):
    """Set a parm by name, warning (never raising) if this VSP build lacks it."""
    pid = vsp.GetParm(gid, name, group)
    if not pid:
        print("[warn] parm not found: %s/%s" % (group, name))
        return
    vsp.SetParmValUpdate(pid, float(value))


def build_wing(sections, root_twist, name, airfoil, x=0.0, z=0.0, y=0.0):
    gid = vsp.AddGeom("WING")
    vsp.SetGeomName(gid, name)
    setp(gid, "X_Rel_Location", "XForm", x)
    setp(gid, "Z_Rel_Location", "XForm", z)
    # ...and the SPANWISE station, which only an off-centreline member needs
    # (the mount). Left at zero for every surface that is already symmetric
    # about the centreline, so nothing else moves.
    setp(gid, "Y_Rel_Location", "XForm", y)
    # one VSP section per solver section: the default wing already has one
    for i in range(1, len(sections)):
        vsp.InsertXSec(gid, i, vsp.XS_FILE_AIRFOIL)
    vsp.Update()
    total = sum(s["span"] for s in sections) or 1.0
    for i, sec in enumerate(sections, start=1):
        grp = "XSec_%d" % i
        setp(gid, "Span", grp, sec["span"])
        setp(gid, "Root_Chord", grp, sec["root_chord"])
        setp(gid, "Tip_Chord", grp, sec["tip_chord"])
        setp(gid, "Sweep", grp, SWEEP_DEG)
        setp(gid, "Sweep_Location", grp, 0.25)
        setp(gid, "Twist", grp, sec["twist_deg"])
        setp(gid, "Twist_Location", grp, 0.25)
        setp(gid, "Dihedral", grp, sec["dihedral_deg"])
        # panels in PROPORTION to the section's span, spread evenly inside it.
        # VSP's defaults give every section the same 6 panels and cluster them
        # against both its edges, so a chain of sections — which is what a
        # sampled chord law is — builds a lattice that is dense at every joint
        # and coarse in between. Measured on a plain tapered wing, the default
        # chain put span efficiency at 0.84 where the same planform as ONE
        # section gives 0.94; evening it out recovers most of that.
        # ...with a FLOOR on the tip device. It is short — a few per cent of
        # the span — so a purely proportional share gave the winglet two
        # panels, and VSPAERO then credited it about a quarter of the span
        # efficiency aerobo's own nonplanar VLM does. The device is the whole
        # point of the geometry; it gets resolved.
        floor = DEVICE_PANELS if sec.get("is_winglet") else 2
        setp(gid, "SectTess_U", grp,
             max(floor, int(round(SPAN_PANELS * sec["span"] / total))))
        setp(gid, "InCluster", grp, 1.0)
        setp(gid, "OutCluster", grp, 1.0)
    setp(gid, "Twist", "XSec_0", root_twist)
    vsp.Update()
    if os.path.exists(airfoil):
        try:
            surf = vsp.GetXSecSurf(gid, 0)
            for i in range(vsp.GetNumXSec(surf)):
                vsp.ChangeXSecShape(surf, i, vsp.XS_FILE_AIRFOIL)
                vsp.ReadFileAirfoil(vsp.GetXSec(surf, i), airfoil)
            vsp.Update()
        except Exception as exc:       # keep the planform, lose the section
            print("[warn] could not apply %s (%s); the VSP default section "
                  "stays" % (airfoil, exc))
    else:
        print("[warn] section file missing, keeping the VSP default: %s"
              % airfoil)
    return gid


def add_elevator(gid, frac, deflection_deg):
    """The hinged elevator as a trailing-edge control sub-surface."""
    ss = vsp.AddSubSurf(gid, vsp.SS_CONTROL)
    setp(ss, "Length_C_Start", "SS_Control", frac)
    setp(ss, "Length_C_End", "SS_Control", frac)
    setp(ss, "UStart", "SS_Control", 0.0)
    setp(ss, "UEnd", "SS_Control", 1.0)
    try:                               # the signature moved between versions
        vsp.SetSubSurfName(gid, ss, "elevator")
    except Exception:
        try:
            vsp.SetSubSurfName(ss, "elevator")
        except Exception as exc:
            print("[warn] could not name the sub-surface (%s)" % exc)
    vsp.Update()
    print("[note] elevator hinge at %.1f%% chord; aerobo trimmed it to "
          "%+.2f deg — set that on the VSPAERO control group." %
          (100.0 * (1.0 - frac), deflection_deg))
    return ss


def build_native():
    vsp.VSPRenew()
    build_wing(WING_SECTIONS, WING_ROOT_TWIST_DEG, "wing", AIRFOIL)
    if TAIL_SECTIONS:
        aft = AIRFOIL_AFT if os.path.exists(AIRFOIL_AFT) else AIRFOIL
        tid = build_wing(TAIL_SECTIONS, TAIL_ROOT_TWIST_DEG, "tail", aft,
                         x=TAIL_X, z=TAIL_Z)
        if ELEVATOR_CHORD_FRAC > 0.0:
            add_elevator(tid, ELEVATOR_CHORD_FRAC, ELEVATOR_DEFLECTION_DEG)
    if FIN_SECTIONS:
        # the fin flies its OWN symmetric section, at the thickness its drag
        # was charged at; the wing's cambered file would build a fin holding
        # a permanent side load
        fin_foil = AIRFOIL_FIN if os.path.exists(AIRFOIL_FIN) else AIRFOIL
        build_wing(FIN_SECTIONS, 0.0, "fin", fin_foil, x=FIN_X, z=FIN_Z)
    if MOUNT_SECTIONS:
        # the strut is STRUCTURE: symmetric, at zero twist, and it makes no
        # side force at zero yaw. It reuses the fin's file for the same
        # reason the fin does not use the wing's — a cambered strut flies a
        # permanent side load nothing asked it for.
        mnt_foil = AIRFOIL_FIN if os.path.exists(AIRFOIL_FIN) else AIRFOIL
        build_wing(MOUNT_SECTIONS, 0.0, "mount", mnt_foil,
                   x=MOUNT_X, y=MOUNT_Y, z=MOUNT_Z)
    return True


def import_mesh():
    """Fallback: the same geometry as the exported STL mesh."""
    stl = os.path.join(HERE, STEM + ".stl")
    if not os.path.exists(stl):
        print("[error] no STL to fall back on: %s" % stl)
        return False
    vsp.VSPRenew()
    vsp.ImportFile(stl, vsp.IMPORT_STL, "")
    return True


def main():
    try:
        ok = build_native()
    except Exception as exc:               # noqa: BLE001 — fall back, say why
        print("[warn] native rebuild failed (%s: %s); importing the STL"
              % (type(exc).__name__, exc))
        ok = import_mesh()
    if not ok:
        sys.exit(1)
    vsp.Update()
    out = os.path.join(HERE, STEM + ".vsp3")
    vsp.WriteVSPFile(out, vsp.SET_ALL)
    print("wrote %s" % out)


if __name__ == "__main__":
    main()
'''
    return head + body


READ_ME = """aerobo — CAD export
===================

Files
-----
  {stem}.stl            every surface, binary STL, metres
  {stem}_ascii.stl      the same, ASCII, one named solid per part
                        (parts are the lifting surfaces AND the mount, where
                        the design has one — see "Parts exported" below)
  {stem}_<part>.stl     one file per part (wing / tail / elevator / fin /
                        mount0, mount1 — the car's struts)
  {stem}_wing.dat       the section that was flown, Selig/XFOIL order
  {stem}_vsp.py         OpenVSP API script — native rebuild, VSPAERO-ready

Axes: x downstream +, y starboard, z up. Metres. Wing quarter chord at x = 0,
so the tail sits at its solved arm and a canard at a negative one.

Into OpenVSP
------------
  Native (preferred — VSPAERO's vortex lattice needs real geoms):
      python {stem}_vsp.py         # writes {stem}.vsp3, then open it
  Mesh (any version, no Python API):
      File > Import > Stereolith (.stl)   ->  a MeshGeom you can measure and
      view; the panel solver will run on it, the vortex lattice will not.

What is idealised
-----------------
  * quarter-chord line straight — sweep never enters the flown geometry
  * elevator drawn as a flat plate at the hinge (the physics uses a flap
    effectiveness tau, not a geometric hinge)
  * a V-tail is the two panels its equivalent flat surface stands for
  * spanwise resolution is the solver's own panel count

{parts}
"""


def export(geom: dict, outdir, stem: str = "aerobo", x_best=None, labels=None,
           section=None, section_aft=None, split: bool = True,
           ascii_stl: bool = True, max_sections: int = 12,
           section_n: int = 61) -> dict:
    """Write the whole CAD bundle for one design; returns {kind: path}.

    ``geom`` is ``api.design_report(...)["geometry"]``; ``section`` is either
    ``api.section_coords(...)`` or the whole ``api.section_report(...)``, and
    ``section_aft`` the second surface's own section when it flies one.
    """
    d = Path(outdir)
    d.mkdir(parents=True, exist_ok=True)
    surfs = surfaces(geom, x_best, labels, section, section_aft,
                     section_n=section_n)
    if not surfs:
        raise ValueError("this design report carries no geometry to export")
    written: dict = {}
    written["stl"] = str(write_stl(d / export_name(stem, "stl"), surfs,
                                   binary=True, name=stem))
    if ascii_stl:
        written["stl_ascii"] = str(write_stl(d / export_name(stem, "stl_ascii"),
                                             surfs, binary=False, name=stem))
    if split:
        seen: dict = {}
        for s in surfs:
            # two surfaces can share a name (an unnamed pair arrives as
            # "surface" twice); numbering the repeat keeps the second from
            # silently overwriting the first
            seen[s.name] = seen.get(s.name, 0) + 1
            nm = s.name if seen[s.name] == 1 else f"{s.name}{seen[s.name]}"
            written[f"stl_{nm}"] = str(
                write_stl(d / f"{stem}_{nm}.stl", [s], binary=True,
                          name=f"{stem} {nm}"))
    xc, zc = section_path(geom, x_best, labels, section, n=section_n)
    written["dat"] = str(_write(d / export_name(stem, "dat"),
                                airfoil_dat(xc, zc, f"{stem} wing section")))
    # The aft section file. Written when the second surface flies a section
    # of its own — and ALSO when it merely flies the wing's section the other
    # way up, because "the same aerofoil mounted inverted" is a different
    # aerofoil to anything reading a .dat. The OpenVSP script falls back to
    # the wing's file when this one is absent, so without that second case a
    # downloading tail is built in VSP with its camber the WRONG WAY and the
    # VSPAERO case then reports the stabiliser LIFTING while this package
    # trims it as a download. The mirror lives here rather than in
    # section_path: that helper answers "what section is this", not "which
    # way up is it mounted" (the STL path mirrors inside tail_surfaces).
    inverted_aft = bool((geom.get("tail") or {}).get("section_inverted"))
    if section_aft is not None or inverted_aft:
        xa, za = ((xc, zc) if section_aft is None
                  else section_path(geom, x_best, labels, section_aft,
                                    n=section_n))
        if inverted_aft:
            from .polar import invert_coords
            _xz = invert_coords(np.column_stack(
                [np.asarray(xa, dtype=float), np.asarray(za, dtype=float)]))
            xa, za = _xz[:, 0], _xz[:, 1]
        written["dat_aft"] = str(_write(
            d / export_name(stem, "dat_aft"),
            airfoil_dat(xa, za, f"{stem} aft section"
                                + (" (MOUNTED INVERTED)" if inverted_aft
                                   else ""))))
    # THE FIN'S OWN SECTION. It is symmetric — a fin at zero sideslip must
    # make no side force — and at the fin's own thickness, which is what its
    # parasite drag was charged at. Written only where there IS a fin, so a
    # V-tail's bundle has no file promising a surface it does not have.
    _finblk = geom.get("fin") or {}
    if _finblk:
        _fx, _fz = naca4_section(float(_finblk.get("tc", 0.10)), m=0.0,
                                 p=0.4, n=(section_n + 1) // 2)
        written["dat_fin"] = str(_write(
            d / export_name(stem, "dat_fin"),
            airfoil_dat(_fx, _fz, f"{stem} fin section (symmetric, "
                                  f"t/c {float(_finblk.get('tc', 0.10)):.3f})")))
    written["vsp"] = str(_write(
        d / export_name(stem, "vsp"),
        vsp_script(geom, x_best, labels, stem=stem,
                   max_sections=max_sections)))
    parts = "Parts exported: " + ", ".join(
        f"{s.name} ({s.shape[0]}x{s.shape[1]} stations, "
        f"{triangles(s).shape[0]} facets)" for s in surfs)
    written["readme"] = str(_write(d / export_name(stem, "readme"),
                                   READ_ME.format(stem=stem, parts=parts)))
    return written


def _write(path: Path, text: str) -> Path:
    path.write_text(text)
    return path
