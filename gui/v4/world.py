"""Aircraft-centred rendering: the aeroplane stays put and the WORLD moves.

This is the geometry behind the game view, and it is here — pure, with no
nicegui and no aerobo — because it is the part that is easy to get subtly
wrong and impossible to check by looking at a screenshot.

WHY THE GROUND MOVES AND THE AEROPLANE DOES NOT. The aircraft is drawn at
the scene origin with its full attitude, and the ground slides underneath.
That keeps the engineering view and the game view sharing ONE aircraft
transform — the CG stays on the pivot, the markers stay put, and the mode
toggle is three property changes rather than a second scene.

The camera then has to orbit to stay behind the aeroplane, and it turns out
that is where the smoothing wants to be anyway: ``move_camera`` runs a
**TWEEN updated on every browser render frame** (scene.js:219, :494), and it
is the ONLY thing in ``ui.scene`` that interpolates client-side —
``Object3D.move`` and ``rotate_R`` take no duration at all. So the camera is
given a tween slightly longer than a frame and the browser fills in the
motion at its own rate, while everything else snaps.

One thing the camera must NOT be given is an ``up`` vector. If any of
``up_x/up_y/up_z`` is non-null, the tween's ``onComplete`` **disposes and
rebuilds the OrbitControls object** (scene.js:526-533) — sixty times a
second, for a camera that rolls.

AND THE BUILT-IN GRID CANNOT BE USED. ``grid=True`` adds its GridHelper and
ground plate straight to ``THREE.Scene`` and never registers them in the
addressable object map (scene.js:184-188), so every mutator early-returns and
no Python call can move, hide or resize them. Worse, that grid sits at scene
z = 0, which is not the ground: the aircraft is drawn at ``altitude - datum``,
so z = 0 is a false floor at the DATUM height. The ground has to be ours, and
it has to sit at ``-datum``.

THE FLOATING ORIGIN. An aeroplane at 45 m/s is 160 km from its start after an
hour, and single-precision scene coordinates lose centimetres long before
that. So the ground is a PERIODIC lattice of identical markers whose local
coordinates never change, and the world group is translated by the aircraft's
offset from the nearest lattice node — a number bounded by half a tile. When
the aeroplane crosses a tile boundary the offset jumps by one full tile and
the picture is unchanged, because the lattice has exactly that period. The
markers are static: they are never moved, at any frame rate, for ever.

FRAMES. The same three as :mod:`gui.v4.stages.flight` — G (loft: x aft,
y starboard, z up), B (body: x forward, z down) and W (scene: x forward,
y port, z up). Nothing new is introduced here; the ground is placed in W
directly, and the derivation is written out in :func:`world_offset`.
"""

from __future__ import annotations

import numpy as np

#: NED -> scene: north stays x, east becomes -y, down becomes -z. A half turn
#: about x, det +1. Identical to ``flight.M_EARTH_TO_SCENE`` and deliberately
#: not imported from there: this module must not drag a nicegui import in.
M_EARTH_TO_SCENE = np.array([[1.0, 0.0, 0.0],
                             [0.0, -1.0, 0.0],
                             [0.0, 0.0, -1.0]])

__all__ = ["M_EARTH_TO_SCENE", "heading_of", "yaw_dcm", "world_offset",
           "chase_eye", "lattice_offsets", "snap", "quat_from_dcm"]


def quat_from_dcm(R) -> tuple:
    """A rotation matrix as ``(x, y, z, w)`` — three.js's component order.

    The browser half of the render loop interpolates the attitude by SLERP,
    which needs a quaternion; sending one also costs four floats instead of
    nine. Shepperd's method rather than the textbook single branch: the naive
    ``w = sqrt(1 + trace)/2`` loses all its precision when the trace
    approaches -1, which is a half turn — and a half turn is not an exotic
    attitude in an aeroplane that can roll.

    The sign is not observable: ``q`` and ``-q`` are the same rotation, and
    SLERP takes the short way round either way.
    """
    R = np.asarray(R, dtype=float)
    m00, m11, m22 = R[0, 0], R[1, 1], R[2, 2]
    tr = m00 + m11 + m22
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif m00 > m11 and m00 > m22:
        s = np.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif m11 > m22:
        s = np.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return float(x), float(y), float(z), float(w)


def heading_of(quat) -> float:
    """Heading [rad] from a body quaternion — the yaw Euler angle.

    Taken from the DCM's first column rather than from a full Euler
    conversion, so it stays finite at 90 degrees of pitch where the pitch
    Euler angle is singular. At exactly vertical the heading is genuinely
    undefined and this returns the previous-ish value 0; the caller carries
    the last good one.
    """
    q = np.asarray(quat, dtype=float)
    w, x, y, z = q / max(float(np.linalg.norm(q)), 1e-12)
    # C[0] is the body x-axis expressed in earth axes
    north = 1.0 - 2.0 * (y * y + z * z)
    east = 2.0 * (x * y + w * z)
    if abs(north) < 1e-12 and abs(east) < 1e-12:
        return 0.0
    return float(np.arctan2(east, north))


def yaw_dcm(psi: float) -> np.ndarray:
    """The DCM that takes EARTH axes to a frame yawed by ``psi`` about down."""
    c, s = float(np.cos(psi)), float(np.sin(psi))
    return np.array([[c, s, 0.0],
                     [-s, c, 0.0],
                     [0.0, 0.0, 1.0]])


def snap(value: float, spacing: float) -> float:
    """Nearest lattice node. ``spacing <= 0`` disables snapping."""
    if not spacing > 0.0:
        return 0.0
    return float(np.round(float(value) / spacing) * spacing)


def world_offset(pos_ned, spacing: float, datum_m: float) -> tuple:
    """Where to put the GROUND group so the aeroplane appears to fly over it.

    THE DERIVATION, because the signs are not guessable. A ground feature at
    earth ``(Nf, Ef, 0)`` and an aircraft at ``(N, E, -alt)`` are separated
    in earth axes by ``(Nf-N, Ef-E, alt)``. Through ``M = diag(1,-1,-1)``
    that is ``(Nf-N, E-Ef, -alt)`` in scene axes. The aircraft itself is
    drawn at ``(0, 0, alt-datum)`` — that is the one line the engineering
    view already had — so the feature lands at::

        (Nf - N,  E - Ef,  -datum)

    A group whose child sits at local ``(Nf, -Ef, 0)`` therefore needs the
    translation returned here. Note the z: the ground is at **-datum**, not
    at zero. Zero is where the built-in grid puts it, which is a false floor
    at the datum altitude — fly down to sea level and you pass straight
    through it.

    THE FLOATING ORIGIN. ``N`` and ``E`` grow without bound — 160 km in an
    hour at 45 m/s — and scene coordinates are single-precision. Because the
    lattice is periodic with period ``spacing``, replacing the position by
    its offset from the nearest node leaves the picture identical while
    keeping every number below half a tile, for ever.
    """
    p = np.asarray(pos_ned, dtype=float)
    return (-(float(p[0]) - snap(p[0], spacing)),
            +(float(p[1]) - snap(p[1], spacing)),
            -float(datum_m))


def chase_eye(R, span_m: float, *, back: float = 2.6, up: float = 0.55,
              lead: float = 0.9, drawn_z: float = 0.0) -> tuple:
    """``(eye, target)`` in scene axes for a camera behind the tail.

    ``R`` is ``flight.scene_rotation(quat)``, which maps BODY vectors into
    the scene. So the nose direction is ``R @ (1,0,0)`` and "up" is
    ``R @ (0,0,-1)`` — body z is DOWN, and using ``(0,0,1)`` puts the camera
    under the aeroplane looking at its belly.

    Distances are in SPANS, not metres: the same numbers frame a 0.8 m model
    and a 30 m aeroplane.

    No ``up`` vector is returned, and none must be passed to ``move_camera``
    — see the module docstring. The consequence is that the horizon does not
    roll with the aircraft; the aeroplane rolls against a level horizon,
    which is what an external chase view looks like in most games anyway.
    """
    R = np.asarray(R, dtype=float)
    nose = R @ np.array([1.0, 0.0, 0.0])
    upv = R @ np.array([0.0, 0.0, -1.0])
    origin = np.array([0.0, 0.0, float(drawn_z)])
    b = max(float(span_m), 1e-3)
    target = origin + lead * b * nose
    eye = origin - back * b * nose + up * b * upv
    return tuple(float(v) for v in eye), tuple(float(v) for v in target)


def lattice_offsets(spacing: float, n_side: int) -> list:
    """The STATIC local positions of the ground markers, in earth axes.

    ``n_side`` nodes each side of the origin, so ``(2n+1)^2`` of them. They
    are created once and never moved again; the world group's transform is
    what makes them stream past.

    EVERY node is kept, including the one at the origin. Dropping it as
    "the one under the aeroplane" is wrong and the tile-crossing test says
    so: the markers sit on LATTICE nodes, not under the aircraft, so a
    dropped node is a permanent hole that drifts backwards through an
    otherwise regular field and then reappears a tile later. The aeroplane
    is hundreds of metres above the ground plane anyway — nothing is ever
    inside it.
    """
    if not spacing > 0.0 or n_side < 0:
        return []
    return [(float(i * spacing), float(j * spacing), 0.0)
            for i in range(-n_side, n_side + 1)
            for j in range(-n_side, n_side + 1)]
