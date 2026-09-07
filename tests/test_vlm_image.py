"""Image plane in the nonplanar VLM: boundary conditions, sign convention,
and agreement with the two lifting-line image solvers.

This is what makes a WINGLET near a boundary solvable: for a planar wing the
mirror of a horizontal vortex system is a pure translation (ground_effect.py
and hydrofoil.py exploit exactly that), but a canted tip device breaks the
translation, so the imaged solve has to happen in the lifting-surface code.

The two signs are the classic trap, so they are re-derived here against the
boundary conditions themselves rather than trusted:
    rigid ground   w = 0 on the plane  -> OPPOSITE-circulation image
    free surface   phi = 0 on the plane -> SAME-circulation image
(phi = 0 means the TANGENTIAL velocities vanish on the plane, not w.)
"""

import numpy as np
import pytest

from aerobo.geometry import Wing
from aerobo.ground_effect import solve_wing_ige
from aerobo.hydrofoil import solve_hydrofoil
from aerobo.llt import solve_llt
from aerobo.tandem import Surface
from aerobo.vlm import IMAGE_SIGNS, VLM, ImagePlane, _hshoe

WING = Wing(taper=0.6, twist_tip_deg=-2.0)
ALPHA = np.deg2rad(4.0)
N = 60


def _plane_points(z: float, n: int = 63) -> np.ndarray:
    """A scatter of points ON the plane, spread over and behind the wing."""
    x = np.linspace(-3.0, 8.0, 7)
    y = np.linspace(-7.0, 7.0, 9)
    X, Y = np.meshgrid(x, y)
    return np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, z)])


def _induced_on_plane(model: VLM, res) -> np.ndarray:
    """Perturbation velocity of the real + image systems on the plane."""
    img = model.image
    P = _plane_points(img.z)
    vel = (_hshoe(P, model.A3, model.B3)
           + img.sign * _hshoe(P, img.mirror(model.A3), img.mirror(model.B3)))
    return np.einsum("ijk,j->ik", vel, res.Gamma)


# ---------------- the boundary conditions themselves ----------------

def test_ground_image_kills_the_normal_velocity_on_the_wall():
    img = ImagePlane(z=-1.0, kind="ground")
    m = VLM(WING, N=N, winglet_h_frac=0.10, winglet_cant_deg=80.0, image=img)
    v = _induced_on_plane(m, m.solve(ALPHA))
    assert np.abs(v[:, 2]).max() < 1e-14        # w = 0: the wall is solid
    assert np.abs(v[:, :2]).max() > 1e-3        # and the flow slips along it


def test_free_surface_image_kills_the_tangential_velocity_on_the_plane():
    img = ImagePlane(z=1.0, kind="free_surface")
    m = VLM(WING, N=N, winglet_h_frac=0.10, winglet_cant_deg=-80.0, image=img)
    v = _induced_on_plane(m, m.solve(ALPHA))
    assert np.abs(v[:, :2]).max() < 1e-14       # phi = 0 => u = v = 0
    assert np.abs(v[:, 2]).max() > 1e-3         # the surface still moves


def test_image_kinds_and_signs_are_explicit():
    assert IMAGE_SIGNS == {"ground": -1.0, "free_surface": +1.0}
    assert ImagePlane(z=0.0, kind="ground").sign == -1.0
    assert ImagePlane(z=0.0, kind="free_surface").sign == +1.0
    with pytest.raises(ValueError, match="unknown image plane kind"):
        ImagePlane(z=0.0, kind="wall")


def test_mirror_reflects_only_the_named_axis():
    img = ImagePlane(z=2.0, kind="ground")
    P = np.array([[1.0, 3.0, 0.5]])
    assert img.mirror(P).tolist() == [[1.0, 3.0, 3.5]]
    assert img.mirror(np.array([[3.0, 0.5]]), axis=1).tolist() == [[3.0, 3.5]]


# ---------------- limits and directions ----------------

def test_a_distant_plane_recovers_the_free_air_solution():
    free = VLM(WING, N=N).solve(ALPHA)
    for kind, z in (("ground", -500.0), ("free_surface", 500.0)):
        r = VLM(WING, N=N, image=ImagePlane(z=z, kind=kind)).solve(ALPHA)
        assert r.CL == pytest.approx(free.CL, rel=1e-4)
        assert r.CDi == pytest.approx(free.CDi, rel=1e-3)


def test_ground_helps_and_the_free_surface_hurts():
    """Opposite signs, opposite physics: the wall's upwash unloads the
    trailing system, the free surface's same-sign image loads it."""
    free = VLM(WING, N=N).solve(ALPHA)
    gnd = VLM(WING, N=N, image=ImagePlane(z=-1.0, kind="ground")).solve(ALPHA)
    fs = VLM(WING, N=N,
             image=ImagePlane(z=1.0, kind="free_surface")).solve(ALPHA)
    assert gnd.CL > free.CL > fs.CL
    assert gnd.CDi < free.CDi < fs.CDi
    assert gnd.e > 1.0 > fs.e


# ---------------- cross-check against the LLT image solvers ----------------

@pytest.mark.parametrize("h", [5.0, 10.0])
def test_matches_the_lifting_line_image_solvers_away_from_the_plane(h):
    """Both codes model the same trailing-vortex image, so the RATIO to
    their own free-air answer must agree once the bound-vortex image (which
    the lifting line cannot see) is negligible — i.e. at h/b >~ 0.5."""
    y, c, tw = WING.sample(80)
    surf = Surface(b=WING.b, c=c, alpha_geo=ALPHA + tw)
    free_llt = solve_llt(WING.b, c, ALPHA + tw, V=1.0)
    free_vlm = VLM(WING, N=80).solve(ALPHA)

    gnd_llt = solve_wing_ige(surf, h, V=1.0)
    gnd_vlm = VLM(WING, N=80,
                  image=ImagePlane(z=-h, kind="ground")).solve(ALPHA)
    assert gnd_vlm.CL / free_vlm.CL == pytest.approx(
        gnd_llt.CL / free_llt.CL, rel=5e-3)
    assert gnd_vlm.CDi / free_vlm.CDi == pytest.approx(
        gnd_llt.CDi_total / free_llt.CDi, rel=1e-2)

    fs_llt = solve_hydrofoil(surf, h, V=1.0)
    fs_vlm = VLM(WING, N=80,
                 image=ImagePlane(z=h, kind="free_surface")).solve(ALPHA)
    assert fs_vlm.CL / free_vlm.CL == pytest.approx(
        fs_llt.CL / free_llt.CL, rel=5e-3)
    assert fs_vlm.CDi / free_vlm.CDi == pytest.approx(
        fs_llt.CDi_total / free_llt.CDi, rel=1e-2)


def test_close_to_the_plane_the_surface_solver_sees_more_than_the_line():
    """At h/b = 0.05 the two DISAGREE, and the direction is the physics:
    the VLM's image includes the BOUND vortex, whose section-level effect
    hydrofoil.py explicitly flags as outside lifting-line resolution. So
    the surface solver must report a STRONGER ground effect, not a
    matching one — this test pins that expectation instead of hiding it."""
    h = 0.5
    y, c, tw = WING.sample(80)
    surf = Surface(b=WING.b, c=c, alpha_geo=ALPHA + tw)
    llt_gain = (solve_wing_ige(surf, h, V=1.0).CL
                / solve_llt(WING.b, c, ALPHA + tw, V=1.0).CL)
    vlm_gain = (VLM(WING, N=80,
                    image=ImagePlane(z=-h, kind="ground")).solve(ALPHA).CL
                / VLM(WING, N=80).solve(ALPHA).CL)
    assert vlm_gain > llt_gain * 1.1


# ---------------- geometry guards ----------------

def test_a_winglet_that_pierces_the_plane_is_refused():
    """0.12 * 5 m of vertical winglet cannot fit under a surface 0.3 m up."""
    with pytest.raises(ValueError, match="crosses the free_surface plane"):
        VLM(WING, N=N, winglet_h_frac=0.12, winglet_cant_deg=90.0,
            image=ImagePlane(z=0.3, kind="free_surface"))


def test_a_surface_touching_the_plane_is_refused():
    with pytest.raises(ValueError, match="comes within"):
        VLM(WING, N=N, image=ImagePlane(z=0.02, kind="free_surface"))


# ---------------- the payoff: winglet direction near a boundary ----------

def test_a_winglet_pointing_away_from_the_free_surface_wins():
    """The trade the air problem does not have: at equal trimmed lift and
    equal winglet ARC LENGTH, canting the tip device DOWN (away from the
    surface, where its own image is weaker) beats canting it up, and both
    beat no device at all."""
    kw = dict(N=N, winglet_h_frac=0.12,
              image=ImagePlane(z=1.0, kind="free_surface"))
    up = VLM(WING, winglet_cant_deg=90.0, **kw).solve_trim(0.5)[1]
    down = VLM(WING, winglet_cant_deg=-90.0, **kw).solve_trim(0.5)[1]
    bare = VLM(WING, N=N,
               image=ImagePlane(z=1.0, kind="free_surface")).solve_trim(0.5)[1]
    assert down.CDi < up.CDi < bare.CDi
