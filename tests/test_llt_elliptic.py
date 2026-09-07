"""M3 gate: the lifting-line kernel reproduces classical analytic identities.

1. Elliptic planform, no twist: e = 1.000 (+-0.005), CDi = CL^2/(pi*AR).
2. CL_alpha (elliptic) = 2*pi*AR/(AR+2).
3. Elliptic loading is the minimum-induced-drag optimum at fixed CL.
4. The analytic elliptic-twist law (port of inverse_twist_elliptic.m, single
   wing) drives a RECTANGULAR planform to e ~= 1 at the target CL.
"""

import os
import shutil

import numpy as np
import pytest

from aerobo.llt import (
    cosine_stations,
    elliptic_chord,
    elliptic_twist_alpha,
    solve_llt,
    solve_llt_trim,
)

AR = 10.0
B = 10.0            # span [m]
S = B**2 / AR       # area [m^2]
N = 60


def _elliptic_wing():
    theta, y = cosine_stations(N, B)
    c = elliptic_chord(y, B, S)
    return theta, y, c


def test_elliptic_planform_e_unity_and_glauert_cdi():
    _, y, c = _elliptic_wing()
    alpha = np.full(N, np.deg2rad(5.0))
    res = solve_llt(B, c, alpha)
    assert res.e == pytest.approx(1.0, abs=0.005)
    assert res.CDi == pytest.approx(res.CL**2 / (np.pi * res.AR), rel=1e-3)
    assert res.AR == pytest.approx(AR, rel=1e-3)


def test_elliptic_cl_alpha():
    _, y, c = _elliptic_wing()
    cl = []
    for a_deg in (4.0, 6.0):
        res = solve_llt(B, c, np.full(N, np.deg2rad(a_deg)))
        cl.append(res.CL)
    cl_alpha = (cl[1] - cl[0]) / np.deg2rad(2.0)
    assert cl_alpha == pytest.approx(2 * np.pi * AR / (AR + 2), rel=0.01)


def test_elliptic_induced_aoa_constant():
    _, y, c = _elliptic_wing()
    res = solve_llt(B, c, np.full(N, np.deg2rad(5.0)))
    expected = res.CL / (np.pi * res.AR)
    assert np.allclose(res.alpha_i_y, expected, rtol=1e-3)


def test_elliptic_loading_is_min_cdi_at_fixed_cl():
    """Any twisted rectangular wing at the same CL has CDi >= CL^2/(pi*AR)."""
    theta, y = cosine_stations(N, B)
    c_rect = np.full(N, S / B)
    rng = np.random.default_rng(0)
    CL_target = 0.5
    for _ in range(5):
        twist = np.deg2rad(rng.uniform(-4, 4, size=3))
        # linear-in-|y| twist law from 3 random (root, mid, tip) knots
        eta = np.abs(2 * y / B)
        tw = np.interp(eta, [0.0, 0.5, 1.0], twist)
        alpha, res = solve_llt_trim(B, c_rect, tw, CL_target)
        assert res.CL == pytest.approx(CL_target, abs=1e-6)
        cdi_min = CL_target**2 / (np.pi * res.AR)
        assert res.CDi >= cdi_min * (1 - 1e-9)


def test_rectangular_wing_e_below_unity():
    theta, y = cosine_stations(N, B)
    c = np.full(N, S / B)
    res = solve_llt(B, c, np.full(N, np.deg2rad(5.0)))
    assert 0.85 < res.e < 1.0


def test_analytic_elliptic_twist_recovers_e_unity_on_rectangle():
    """Validation oracle: the closed-form twist law (inverse_twist_elliptic.m,
    single-wing degenerate case) produces elliptic loading on a rectangle."""
    theta, y = cosine_stations(N, B)
    c = np.full(N, S / B)
    CL_target = 0.5
    alpha_geo = elliptic_twist_alpha(y, B, c, CL_target)
    res = solve_llt(B, c, alpha_geo)
    assert res.CL == pytest.approx(CL_target, rel=1e-3)
    assert res.e == pytest.approx(1.0, abs=0.005)
    assert res.CDi == pytest.approx(CL_target**2 / (np.pi * res.AR), rel=0.005)


@pytest.mark.skipif(
    os.environ.get("AEROBO_MATLAB") != "1" or shutil.which("matlab") is None,
    reason="opt-in (AEROBO_MATLAB=1, matlab on PATH): ~20 s matlab -batch launch",
)
def test_cross_check_vs_matlab_solve_llt():
    """Run the verbatim reference solve_llt (extracted from lift_drag_strip.m)
    on a tapered + twisted wing and compare against the numpy port."""
    from matlab_crosscheck import run_matlab_solve_llt

    ref = run_matlab_solve_llt(N=N, b=B, S=S, taper=0.5, alpha_deg=5.0,
                               twist_root=0.0, twist_tip=-3.0, a0_deg=-2.0)

    theta, y = cosine_stations(N, B)
    c_root = 2 * S / (B * (1 + 0.5))
    eta = np.abs(2 * y / B)
    c = c_root * (1 - (1 - 0.5) * eta)
    alpha = np.deg2rad(5.0) + np.deg2rad(0.0 + (-3.0 - 0.0) * eta)
    res = solve_llt(B, c, alpha, a=2 * np.pi, alpha_L0=np.deg2rad(-2.0))

    # Same linear system -> Fourier coefficients match to machine precision,
    # as do the Glauert CDi and e formulas (identical in both codes).
    assert res.A[0] == pytest.approx(ref["A1"], rel=1e-10)
    assert res.CDi == pytest.approx(ref["CDi"], rel=1e-10)
    assert res.e == pytest.approx(ref["e"], rel=1e-10)
    # CL formula differs by construction: the reference integrates Gamma with
    # trapz on the midpoint cosine grid (no tip stations), the port uses the
    # exact Fourier identity CL = pi*AR*A1 — agreement is discretisation-level.
    assert res.CL == pytest.approx(ref["CL"], rel=1e-3)
    assert np.pi * ref["AR"] * ref["A1"] == pytest.approx(res.CL, rel=1e-10)
