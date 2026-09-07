"""Phase 1a gates: free-surface (negative-image) hydrofoil lifting line.

1. Deep limit: h -> 1e4 spans reproduces the isolated single-wing LLT
   (mirror of the tandem decoupling gate).
2. Biplane analogue [ANALYTIC, Prandtl 1924]: the same-sign image at gap 2h
   is Prandtl's biplane; at FIXED elliptic circulation the surface-induced
   drag on the real foil is +sigma(2h/b) * CDi_self, with sigma matching
   Prandtl's multiplane formula to 2 % in its validity range. The SIGN is
   the anti-ground-effect: CDi INCREASES near the surface.
3. Monotonicity: CL_alpha(h) and e_total(h) both monotone increasing in
   depth; e_total < 1 always (the image destroys, never helps).
4. Symmetry / translation invariance (mirror of test_tandem.py):
   z-translation of foil+surface together changes nothing; symmetric
   geometry -> symmetric loading; downwash sign.
5. Trim: closed-form trim hits CL_target exactly; deep trim == isolated trim.
"""

import numpy as np
import pytest

from aerobo.hydrofoil import (
    HydrofoilResult,
    build_image_operators,
    froude_depth,
    solve_hydrofoil,
    solve_hydrofoil_trim,
)
from aerobo.llt import cosine_stations, elliptic_chord, solve_llt, solve_llt_trim
from aerobo.tandem import Surface, VortexSystem, mutual_cdi

B = 10.0
AR = 10.0
S = B**2 / AR
N = 60


def _rect_foil(alpha_deg: float = 5.0) -> Surface:
    c = np.full(N, S / B)
    return Surface(b=B, c=c, alpha_geo=np.full(N, np.deg2rad(alpha_deg)))


def _elliptic_gamma(alpha_deg: float = 5.0):
    _, y = cosine_stations(N, B)
    c = elliptic_chord(y, B, S)
    res = solve_llt(B, c, np.full(N, np.deg2rad(alpha_deg)))
    return y, c, res


# ------------------------------------------------------------- 1. deep limit

def test_deep_limit_reproduces_isolated_wing():
    """h = 1e4 spans -> isolated LLT to ~1e-10 (image influence vanishes)."""
    foil = _rect_foil()
    hr = solve_hydrofoil(foil, depth=1e4 * B)
    iso = solve_llt(B, foil.c, foil.alpha_geo)
    assert hr.CL == pytest.approx(iso.CL, rel=1e-9)
    assert hr.CDi_self == pytest.approx(iso.CDi, rel=1e-9)
    assert hr.e_total == pytest.approx(iso.e, rel=1e-8)
    assert abs(hr.CDi_surf) < 1e-11
    assert np.max(np.abs(hr.eps_surf)) < 1e-10


# ------------------------------------------------- 2. biplane analogue gate

def test_image_interference_matches_prandtl_sigma():
    """[ANALYTIC] fixed elliptic Gamma: CDi_surf / CDi_self == +sigma(2h/b),
    Prandtl (1924) sigma = (1 - 0.66 g/b)/(1.05 + 3.7 g/b) to 2 % for
    0.05 <= g/b <= 0.5 (g = 2h). Same formula as the tandem biplane gate;
    here the interference is the drag-INCREASING, same-sign-image case."""
    y, c, res = _elliptic_gamma()
    real = VortexSystem(b=B, N=N)
    depths = np.array([0.025, 0.05, 0.10, 0.20, 0.25]) * B    # g/b in [.05, .5]
    for h in depths:
        img = VortexSystem(b=B, N=N, z=2.0 * h)
        cdi_surf, _ = mutual_cdi(real, img, res.Gamma, res.Gamma,
                                 V=1.0, Sref=S)
        sigma_num = cdi_surf / res.CDi
        gb = 2.0 * h / B
        sigma_prandtl = (1 - 0.66 * gb) / (1.05 + 3.7 * gb)
        assert sigma_num > 0.0                    # drag INCREASE (anti-ground)
        assert sigma_num == pytest.approx(sigma_prandtl, rel=0.02)


def test_image_interference_limits_and_monotonicity():
    """sigma(h): monotone decreasing, -> 1 as h -> 0 (doubled monoplane),
    -> 0 as h -> inf (isolated foil)."""
    y, c, res = _elliptic_gamma()
    real = VortexSystem(b=B, N=N)
    depths = np.array([0.01, 0.025, 0.05, 0.10, 0.25, 0.5, 2.5]) * B
    sigma = []
    for h in depths:
        img = VortexSystem(b=B, N=N, z=2.0 * h)
        cdi_surf, _ = mutual_cdi(real, img, res.Gamma, res.Gamma,
                                 V=1.0, Sref=S)
        sigma.append(cdi_surf / res.CDi)
    sigma = np.array(sigma)
    assert np.all(np.diff(sigma) < 0)
    assert sigma[0] > 0.85
    assert sigma[-1] < 0.01


# ----------------------------------------------------------- 3. monotonicity

def test_cl_alpha_and_e_monotone_in_depth():
    """CL_alpha(h) and e_total(h) increase with depth; e_total < 1 always;
    both approach the isolated values from below."""
    c = np.full(N, S / B)
    al1, al2 = np.full(N, np.deg2rad(4.0)), np.full(N, np.deg2rad(6.0))
    iso1 = solve_llt(B, c, al1)
    iso2 = solve_llt(B, c, al2)
    cl_alpha_iso = (iso2.CL - iso1.CL) / np.deg2rad(2.0)

    depths = np.array([0.05, 0.1, 0.2, 0.5, 1.0, 2.0]) * B
    cl_alpha, e_tot = [], []
    for h in depths:
        r1 = solve_hydrofoil(Surface(b=B, c=c, alpha_geo=al1), depth=h)
        r2 = solve_hydrofoil(Surface(b=B, c=c, alpha_geo=al2), depth=h)
        cl_alpha.append((r2.CL - r1.CL) / np.deg2rad(2.0))
        e_tot.append(r2.e_total)
    cl_alpha, e_tot = np.array(cl_alpha), np.array(e_tot)

    assert np.all(np.diff(cl_alpha) > 0)          # slope recovers with depth
    assert np.all(np.diff(e_tot) > 0)             # e recovers with depth
    assert np.all(cl_alpha < cl_alpha_iso)        # image only DEGRADES
    assert np.all(e_tot < iso2.e)
    assert np.all(e_tot < 1.0)                    # never helps
    # shallow degradation is substantial (this is the Tier C physics story)
    # measured at h = 0.05 b: CL_alpha 4.600 vs isolated 5.049 (-8.9 %)
    assert cl_alpha[0] < 0.95 * cl_alpha_iso
    assert e_tot[0] < 0.7 * iso2.e


def test_downwash_sign():
    """The image of a lifting foil induces DOWNWASH at the real foil."""
    foil = _rect_foil()
    hr = solve_hydrofoil(foil, depth=0.1 * B)
    _, y = cosine_stations(N, B)
    mid = np.abs(2 * y / B) < 0.8
    assert np.all(hr.eps_surf[mid] < 0.0)
    assert hr.CDi_surf > 0.0


# ----------------------------------------- 4. symmetry / translation gates

def test_z_translation_invariance():
    """Foil at z0 with surface at z0+h == foil at 0 with surface at h."""
    c = np.full(N, S / B)
    al = np.full(N, np.deg2rad(5.0))
    r0 = solve_hydrofoil(Surface(b=B, c=c, alpha_geo=al, z=0.0), depth=0.8)
    r5 = solve_hydrofoil(Surface(b=B, c=c, alpha_geo=al, z=5.0), depth=0.8)
    assert r0.CL == pytest.approx(r5.CL, rel=1e-12)
    assert r0.CDi_total == pytest.approx(r5.CDi_total, rel=1e-12)


def test_symmetric_loading():
    foil = _rect_foil()
    hr = solve_hydrofoil(foil, depth=0.1 * B)
    assert np.max(np.abs(hr.foil.Gamma - hr.foil.Gamma[::-1])) < 1e-10


def test_invalid_depth_raises():
    with pytest.raises(ValueError):
        build_image_operators(VortexSystem(b=B, N=N), depth=0.0)


def test_froude_depth():
    """[ANALYTIC] Fn_h = V / sqrt(g h): 10 m/s at 1 m -> 3.193."""
    assert froude_depth(10.0, 1.0) == pytest.approx(10.0 / np.sqrt(9.81), rel=1e-12)


# ------------------------------------------------------------------ 5. trim

def test_trim_hits_target_and_deep_matches_isolated():
    c = np.full(N, S / B)
    twist = np.zeros(N)
    alpha, hr = solve_hydrofoil_trim(B, c, twist, CL_target=0.5, depth=0.1 * B)
    assert hr.CL == pytest.approx(0.5, abs=1e-10)
    assert isinstance(hr, HydrofoilResult)

    a_deep, hr_deep = solve_hydrofoil_trim(B, c, twist, CL_target=0.5,
                                           depth=1e4 * B)
    a_iso, iso = solve_llt_trim(B, c, twist, 0.5)
    assert a_deep == pytest.approx(a_iso, rel=1e-6)
    assert hr_deep.CDi_total == pytest.approx(iso.CDi, rel=1e-8)
    # shallow foil needs MORE alpha for the same CL (image downwash)
    assert alpha > a_deep


def test_shallow_costs_drag_at_fixed_lift():
    """Full-solve version of the sign gate: trimmed to the same CL, the
    shallow foil pays more induced drag than the deep one."""
    c = np.full(N, S / B)
    twist = np.zeros(N)
    _, sh = solve_hydrofoil_trim(B, c, twist, CL_target=0.5, depth=0.05 * B)
    _, dp = solve_hydrofoil_trim(B, c, twist, CL_target=0.5, depth=5.0 * B)
    assert sh.CDi_total > 1.2 * dp.CDi_total
