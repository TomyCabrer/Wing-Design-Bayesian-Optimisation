"""The vertical surface — vlm.VerticalSurface, and the yaw stiffness it buys.

The fin is the geometry this package did not have. Before it, `Cn_beta` was
EXACTLY zero for every configuration that was not carrying canted winglets,
and the winglet case measured the wrong SIGN. So these tests are about a
surface existing at all, and about the arm being the thing that decides what
it does.
"""

import numpy as np
import pytest

from aerobo import dynamics as dyn
from aerobo.geometry import Wing
from aerobo.vlm import VLM, TailSurface, VerticalSurface

WING = dict(b=10.0, S=10.0, taper=0.5)
X_CG = 0.3


def _model(vertical=None, N=40):
    return VLM(Wing(**WING), N=N, V=30.0,
               tail=TailSurface(S=2.0, x=4.0, z=0.5, AR=4.0, N=20),
               vertical=vertical)


def _deck(vertical=None, N=40):
    return dyn.deck(_model(vertical, N), x_cg=X_CG, mac=1.0, b=10.0)


def _fin(**kw):
    kw = dict(height=1.2, chord=0.7, x=4.0, z_root=0.5, N=10) | kw
    return VerticalSurface(**kw)


# ------------------------------------------------------- it closes the hole

def test_a_fin_makes_yaw_stiffness_where_there_was_none():
    """Cn_beta: exactly 0 without a fin, positive (stable) with one."""
    assert _deck().Cn_beta == pytest.approx(0.0, abs=1e-12)
    assert _deck(_fin()).Cn_beta > 0.01


def test_a_fin_makes_side_force_opposing_the_sideslip():
    assert _deck(_fin()).CY_beta < -0.05


def test_the_deck_stops_flagging_cn_beta_as_a_geometry_gap():
    """The honesty channel has to CLEAR when the geometry is fixed.

    A warning that survives its own cause is worse than no warning: it trains
    the reader to ignore it.
    """
    assert "Cn_beta" in _deck().zeros
    assert "Cn_beta" not in _deck(_fin()).zeros


def test_a_fin_damps_yaw():
    """Cn_r < 0 — and it is the fin that provides it."""
    assert _deck().Cn_r == pytest.approx(0.0, abs=1e-12)
    assert _deck(_fin()).Cn_r < -0.01


# --------------------------------------------------- the arm decides the sign

def test_yaw_stiffness_is_zero_when_the_fin_sits_ON_the_cg():
    """The closed form: Cn_beta is side force times ARM, so a zero arm is a
    zero moment however big the surface is. Nothing else in the model has to
    cooperate for this to hold, which is what makes it a good rail."""
    D = _deck(_fin(x=X_CG))
    assert D.CY_beta < -0.05                      # the surface is still there
    assert D.Cn_beta == pytest.approx(0.0, abs=1e-9)


def test_a_fin_AHEAD_of_the_cg_is_destabilising():
    assert _deck(_fin(x=-2.0)).Cn_beta < 0.0


def test_yaw_stiffness_grows_with_the_arm():
    vals = [_deck(_fin(x=x)).Cn_beta for x in (1.0, 2.5, 4.0, 7.0)]
    assert all(np.diff(vals) > 0.0)


def test_yaw_stiffness_grows_with_fin_area():
    small = _deck(_fin(height=0.6)).Cn_beta
    big = _deck(_fin(height=1.8)).Cn_beta
    assert 0.0 < small < big


# ------------------------------------------------ dorsal versus ventral

def test_a_dorsal_fin_is_stabilising_in_ROLL_and_a_ventral_one_is_not():
    """Both make yaw stiffness; they disagree about roll.

    A fin above the CG turns the sideslip's side force into a roll AWAY from
    the sideslip (stable, Cl_beta < 0). Hang the same fin below and the arm
    reverses. This is the sign that decides whether a hydrofoil's strut helps
    or hurts, so it is asserted rather than assumed.
    """
    up = _deck(_fin(height=+1.2, z_root=0.0))
    down = _deck(_fin(height=-1.2, z_root=0.0))
    assert up.Cn_beta > 0.0 and down.Cn_beta > 0.0
    assert up.Cl_beta < 0.0
    assert down.Cl_beta > 0.0


# --------------------------------------- it must not disturb the longitudinal

def test_a_fin_contributes_exactly_no_lift_and_exactly_no_pitching_moment():
    """The closed form, asserted on the FIN'S OWN PANELS.

    A vertical bound segment has ``l = (0, 0, dz)``, so Kutta-Joukowski with
    the freestream gives ``F = V Gamma (x_hat x z_hat) dz`` — a pure side
    force along -y. Crossing an arm ``(a_x, a_y, a_z)`` with ``(0, -f, 0)``
    leaves ``(a_z f, 0, -a_x f)``: roll and yaw, and the pitching component is
    IDENTICALLY zero, not merely small. Assert that, because it is the
    statement that a fin cannot contaminate a wing result.
    """
    m = _model(_fin())
    F, M = dyn.panel_loads(m, m._Gam1, X_CG)
    v = m.is_vertical
    assert np.all(F[v, 2] == 0.0)                 # no lift from a fin panel
    assert np.all(M[v, 1] == 0.0)                 # no pitching moment either
    assert np.any(np.abs(F[v, 1]) > 0.0)          # ...but it IS loaded


def test_adding_a_fin_leaves_the_longitudinal_derivatives_where_they_were():
    """Unchanged to machine precision — but NOT bit-for-bit, and that is the
    honest claim: a fin adds panels, so the linear system it is solved with
    has a different dimension and a different summation order. The residual
    measured here is ~2 ulp, which is rounding, not physics.
    """
    a, b = _deck(), _deck(_fin())
    assert b.CL_alpha == pytest.approx(a.CL_alpha, rel=1e-12)
    assert b.Cm_alpha == pytest.approx(a.Cm_alpha, rel=1e-12)
    assert b.Cm_q == pytest.approx(a.Cm_q, rel=1e-12)


def test_a_dorsal_fin_ADDS_roll_damping():
    """Cl_p is lateral, not longitudinal, and the fin is entitled to move it.

    Rolling at p carries the fin sideways through the air at ``p * z_arm``,
    so it makes a side force, and a surface above the CG turns that into a
    moment opposing the roll. Measured at ~0.14 % here — small, real, and the
    reason Cl_p does NOT belong in the "unchanged" test above.
    """
    a, b = _deck().Cl_p, _deck(_fin()).Cl_p
    assert b < a < 0.0


# --------------------------------------------------------------- the contract

def test_area_and_aspect_ratio_are_positive_for_a_ventral_fin():
    f = _fin(height=-1.5, chord=0.5)
    assert f.S == pytest.approx(0.75)
    assert f.AR == pytest.approx(3.0)


@pytest.mark.parametrize("kw,match", [
    (dict(height=0.0), "height must be non-zero"),
    (dict(chord=0.0), "chord must be > 0"),
    (dict(chord=-1.0), "chord must be > 0"),
    (dict(N=1), "at least 2 panels"),
])
def test_degenerate_fins_are_refused(kw, match):
    with pytest.raises(ValueError, match=match):
        _fin(**kw)


def test_panels_are_tagged_and_counted():
    m = _model(_fin(N=10))
    assert int(m.is_vertical.sum()) == 10
    assert m.is_vertical.shape == (m.n_panels,)
    # a fin panel is not also a tail panel or a winglet panel
    assert not np.any(m.is_vertical & m.is_tail)
    assert not np.any(m.is_vertical & m.is_winglet)


def test_no_fin_means_an_all_false_mask_of_the_right_length():
    m = _model()
    assert m.is_vertical.shape == (m.n_panels,)
    assert not m.is_vertical.any()
    assert m.vertical is None
