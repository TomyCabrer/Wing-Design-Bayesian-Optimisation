"""V5 item 1 — dihedral is a RIGID rotation, and it is the only wing-side
source of ``Cl_beta``.

Every test here asserts an outcome (a force, a length, a sign, a crossing),
not a restatement of the arithmetic in ``geometry.dihedral_rotate``. Where a
closed form exists it is written out independently — strip theory for the
dihedral effect, ``cos Gamma`` for the projected span, the isometry property
for arc length.
"""
from __future__ import annotations

import numpy as np
import pytest

from aerobo import dynamics as dyn
from aerobo.geometry import DIHEDRAL_BOUNDS_DEG, Wing, dihedral_rotate
from aerobo.vlm import VLM

WING = dict(b=10.0, S=10.0, taper=0.6, twist_tip_deg=-2.0)
ALPHA = np.deg2rad(4.0)


def _model(gamma_deg: float, **kw) -> VLM:
    return VLM(Wing(**WING, dihedral_deg=gamma_deg), N=40, V=30.0, **kw)


def _arc(y: np.ndarray, z: np.ndarray) -> float:
    """Developed length of a (y, z) polyline."""
    return float(np.sum(np.hypot(np.diff(y), np.diff(z))))


# ------------------------------------------------------------ the primitive

def test_zero_dihedral_is_the_identity_not_a_rounding_of_it():
    """A published planar geometry must come back bit-for-bit, not to 1e-16.

    ``cos(0) == 1.0`` exactly, so a naive implementation would pass a
    tolerance test — this one asserts EXACT equality, which is what "every
    frozen study still reproduces" requires.
    """
    y = np.array([-4.0, -1.5, 0.0, 1.5, 4.0])
    z = np.array([0.0, 0.2, 0.0, 0.2, 0.0])
    ry, rz = dihedral_rotate(y, z, 0.0)
    assert np.array_equal(ry, y)
    assert np.array_equal(rz, z)


def test_both_tips_go_up_which_a_plain_rotation_does_not_do():
    """Dihedral is mirror-symmetric; a rigid ``R_x`` is a ROLLED aeroplane.

    This is the one property that separates the two, and getting it wrong
    produces a lattice that is asymmetric in a way every symmetric solve
    would quietly average away.
    """
    y = np.array([-5.0, 5.0])
    z = np.zeros(2)
    ry, rz = dihedral_rotate(y, z, 8.0)
    assert rz[0] > 0.0 and rz[1] > 0.0          # BOTH tips up
    assert rz[0] == pytest.approx(rz[1])        # and by the same amount
    assert ry[0] < 0.0 < ry[1]                  # port stays port


def test_anhedral_puts_both_tips_down():
    ry, rz = dihedral_rotate(np.array([-5.0, 5.0]), np.zeros(2), -8.0)
    assert rz[0] < 0.0 and rz[1] < 0.0
    assert ry[0] < 0.0 < ry[1]


def test_the_rotation_is_an_isometry_so_developed_span_never_moves():
    """Arc length is the invariant the whole nonplanar family is built on.

    If dihedral changed it, the wing would silently gain or lose span — and
    with it area, chord and every reference quantity.
    """
    s = np.linspace(-5.0, 5.0, 201)
    y, z = s, 0.15 * np.abs(s) ** 2 / 25.0        # a curved (gull) starting line
    flat = _arc(y, z)
    for g in (-10.0, -3.0, 3.0, 7.5, 15.0):
        ry, rz = dihedral_rotate(y, z, g)
        assert _arc(ry, rz) == pytest.approx(flat, rel=1e-12)


# ------------------------------------------------------------- in the lattice

def test_a_dihedralled_lattice_keeps_every_array_the_planar_one_had():
    """Rigid means the CHORD, TWIST and panel WIDTH arrays cannot move.

    Only the direction each bound segment points may change. A test that
    only checked forces would pass an implementation that rescaled the
    span while rotating it.
    """
    flat, cant = _model(0.0), _model(6.0)
    assert np.array_equal(flat.c, cant.c)
    assert np.array_equal(flat.twist, cant.twist)
    w_flat = np.linalg.norm(flat.B3 - flat.A3, axis=1)
    w_cant = np.linalg.norm(cant.B3 - cant.A3, axis=1)
    assert w_flat == pytest.approx(w_cant, rel=1e-12)
    assert not np.allclose(flat.z, cant.z)        # something DID move


def test_zero_dihedral_reproduces_a_winglet_lattice_bit_for_bit():
    kw = dict(winglet_h_frac=0.12, winglet_cant_deg=75.0,
              winglet_blend_frac=0.4)
    a = VLM(Wing(**WING), N=40, V=30.0, **kw)
    b = _model(0.0, **kw)
    ra, rb = a.solve(ALPHA), b.solve(ALPHA)
    # arrays exact, forces to the BLAS floor (test_v5_sweep.py measures it)
    for got, want in ((ra.CL, rb.CL), (ra.CDi, rb.CDi), (ra.e, rb.e)):
        assert got == pytest.approx(want, rel=1e-14)
    assert np.array_equal(a.A3, b.A3) and np.array_equal(a.st3, b.st3)


def test_projected_span_goes_as_cos_gamma():
    """What dihedral TRADES is projected span for height — measure the trade."""
    flat = _model(0.0)
    span0 = flat.y.max() - flat.y.min()
    for g in (3.0, 6.0, 12.0):
        m = _model(g)
        span = m.y.max() - m.y.min()
        assert span / span0 == pytest.approx(np.cos(np.deg2rad(g)), rel=1e-9)
        assert m.z.max() / span0 == pytest.approx(
            0.5 * np.sin(np.deg2rad(g)), rel=1e-9)


# ------------------------------------------------------------- the derivative

def _cl_beta(gamma_deg: float) -> float:
    m = _model(gamma_deg)
    return dyn.deck(m, x_cg=0.3, mac=1.0, b=WING["b"], alpha=ALPHA).Cl_beta


def test_a_planar_wing_alone_has_exactly_no_dihedral_effect():
    """The hole this item fills: with no cant and no fin, ``Cl_beta`` is 0.

    Not "small" — zero. Nothing in a flat lattice carries side force, so a
    sideslip has nothing to push on, which is why the fin was carrying the
    dihedral effect and the yaw stiffness at the same time.
    """
    assert abs(_cl_beta(0.0)) < 1e-12


def test_dihedral_effect_is_negative_linear_and_under_the_strip_bound():
    """Sign, linearity and MAGNITUDE, against an independently written bound.

    Strip theory integrates the local incidence change a sideslip makes on a
    canted panel:  ``dCl_beta/dGamma = -(2 a / (S b)) * int_0^{b/2} c(y) y dy``.
    It ignores the induced reaction, so it is an OVER-estimate: the lattice
    must land below it in magnitude, and not by an order of magnitude.
    """
    w = Wing(**WING)
    y = np.linspace(0.0, w.b / 2.0, 2001)
    strip = -(2.0 * (2.0 * np.pi) / (w.S * w.b)) * np.trapezoid(w.chord(y) * y, y)

    small = [_cl_beta(g) for g in (1.0, 2.0, 3.0)]
    assert all(v < 0.0 for v in small)
    # linear in Gamma over a small range: the 2 deg value is the mean of its
    # neighbours to a tight tolerance
    assert small[1] == pytest.approx(0.5 * (small[0] + small[2]), rel=2e-3)

    slope = small[2] / np.deg2rad(3.0)
    assert strip < slope < 0.4 * strip, (
        f"lattice {slope:.4f} /rad vs strip bound {strip:.4f} /rad")


def test_anhedral_flips_the_sign_of_the_dihedral_effect():
    assert _cl_beta(-4.0) == pytest.approx(-_cl_beta(4.0), rel=2e-2)


def test_the_bounds_admit_anhedral_and_reach_past_the_measured_fix():
    """A calibration is a default, not a ban.

    The reference design needs 2.64 deg; a box that stopped near there would
    be a gate wearing a bound's clothes.
    """
    lo, hi = DIHEDRAL_BOUNDS_DEG
    assert lo < 0.0, "anhedral is a real answer"
    assert hi >= 3.0 * 2.64
