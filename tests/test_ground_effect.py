"""Phase 1b gates: rigid-ground (positive-image) wing-in-ground-effect LLT.

The whole point is the SIGN. Ground effect is the free-surface hydrofoil's
mirror image with the OPPOSITE image circulation (rigid-wall w = 0 vs the
zero-pressure surface), so every gate here is the sign-flip of a
test_hydrofoil.py gate:

1. Boundary condition [NUMERICAL]: the wall condition that DEFINES the image
   sign — total normal velocity w (real + image) vanishes ON the ground
   plane. This is what fixes GROUND_IMAGE_SIGN = -1.
2. Deep limit: h -> inf reproduces the isolated single-wing LLT.
3. Sign / monotonicity: CDi_total DECREASES and CL(fixed alpha) INCREASES as
   the wing nears the ground; e_total > isolated e for small h (the exact
   opposite of the hydrofoil, which DEGRADES all three).
4. Magnitude anchor: induced-drag reduction at h/b = 0.25 lands in the
   classic 20-45 % band (Wieselsberger 1922 / Prandtl biplane sigma).
5. Opposite-of-hydrofoil [DEFINITIVE]: same geometry, same gap — the
   hydrofoil image RAISES CDi, the ground image LOWERS it, by the exact
   same magnitude at fixed circulation.
6. Trim: closed-form trim hits CL_target; trimmed L/D IMPROVES in ground
   effect vs free air at the same CL_target.
"""

import numpy as np
import pytest

from aerobo.ground_effect import (
    GROUND_IMAGE_SIGN,
    GroundEffectResult,
    build_ground_operators,
    solve_wing_ige,
    solve_wing_ige_trim,
)
from aerobo.hydrofoil import solve_hydrofoil
from aerobo.llt import cosine_stations, elliptic_chord, solve_llt, solve_llt_trim
from aerobo.tandem import Surface, VortexSystem, mutual_cdi, w_influence

B = 10.0
AR = 10.0
S = B**2 / AR
N = 60


def _rect_wing(alpha_deg: float = 5.0) -> Surface:
    c = np.full(N, S / B)
    return Surface(b=B, c=c, alpha_geo=np.full(N, np.deg2rad(alpha_deg)))


# ------------------------------------------------- 1. boundary condition gate

def test_boundary_condition_w_vanishes_on_ground():
    """[NUMERICAL] The rigid-wall condition that DEFINES the image sign:
    total normal velocity (real wing + opposite-sign image) vanishes on the
    ground plane, at 20 random points. This is the numerical proof of
    GROUND_IMAGE_SIGN = -1 — flip the sign and this blows up."""
    h = 0.2 * B
    surf = _rect_wing()
    ops = build_ground_operators(surf.system, h)
    hr = solve_wing_ige(surf, h, ops=ops)
    Gam = hr.foil.Gamma

    rng = np.random.default_rng(0)
    z_ground = surf.z - h
    pts = np.column_stack([
        rng.uniform(-2.0 * B, 5.0 * B, 20),      # x: on the line, up- & down-stream
        rng.uniform(-0.9 * B / 2, 0.9 * B / 2, 20),
        np.full(20, z_ground),                   # ON the ground plane
    ])
    w_real = w_influence(pts, surf.system) @ Gam
    w_img = w_influence(pts, ops.img_sys) @ (GROUND_IMAGE_SIGN * Gam)
    w_tot = w_real + w_img
    assert np.max(np.abs(w_tot)) < 1e-10 * np.max(np.abs(w_real))

    # sanity: with the WRONG (same-sign) image the wall leaks badly
    w_wrong = w_real + w_influence(pts, ops.img_sys) @ Gam
    assert np.max(np.abs(w_wrong)) > 0.1 * np.max(np.abs(w_real))


# --------------------------------------------------------------- 2. deep limit

def test_deep_limit_reproduces_isolated_wing():
    """h -> inf -> isolated LLT. The ground image decays as 1/h^2, so the
    strict 1e-9 gate is taken at h = 1e4 b (as test_hydrofoil's deep gate
    does); at the nominal-deep h = 100 b the residual is ~1e-6 (1/h^2 law),
    checked to a realistic tolerance. [Deviation from the brief's literal
    "100 b -> 1e-9": 1e-9 at 100 b is not physically attainable for a
    1/h^2-decaying image; both depths are asserted instead.]"""
    surf = _rect_wing()
    iso = solve_llt(B, surf.c, surf.alpha_geo)

    hr_deep = solve_wing_ige(surf, h_agl=1e4 * B)
    assert hr_deep.CL == pytest.approx(iso.CL, rel=1e-9)
    assert hr_deep.CDi_total == pytest.approx(iso.CDi, rel=1e-9)
    assert hr_deep.e_total == pytest.approx(iso.e, rel=1e-8)
    assert abs(hr_deep.CDi_ground) < 1e-11
    assert np.max(np.abs(hr_deep.eps_ground)) < 1e-10

    hr_100 = solve_wing_ige(surf, h_agl=100.0 * B)
    assert hr_100.CL == pytest.approx(iso.CL, rel=1e-5)
    assert hr_100.CDi_total == pytest.approx(iso.CDi, rel=1e-5)


# --------------------------------------------------- 3. sign / monotonicity

def test_cdi_decreases_and_cl_increases_toward_ground():
    """Fixed alpha, h/b in [0.1, 2]: as the wing nears the ground,
    CDi_total DECREASES monotonically, CL INCREASES monotonically, CL_alpha
    rises, and e_total exceeds the isolated e for small h — the opposite of
    the hydrofoil's degradation."""
    c = np.full(N, S / B)
    al = np.full(N, np.deg2rad(5.0))
    al1, al2 = np.full(N, np.deg2rad(4.0)), np.full(N, np.deg2rad(6.0))
    iso = solve_llt(B, c, al)
    iso1, iso2 = solve_llt(B, c, al1), solve_llt(B, c, al2)
    cl_alpha_iso = (iso2.CL - iso1.CL) / np.deg2rad(2.0)

    h_over_b = np.array([2.0, 1.0, 0.5, 0.25, 0.1])   # DECREASING height
    cdi_tot, cl, cl_alpha, e_tot = [], [], [], []
    for hb in h_over_b:
        r = solve_wing_ige(Surface(b=B, c=c, alpha_geo=al), h_agl=hb * B)
        r1 = solve_wing_ige(Surface(b=B, c=c, alpha_geo=al1), h_agl=hb * B)
        r2 = solve_wing_ige(Surface(b=B, c=c, alpha_geo=al2), h_agl=hb * B)
        cdi_tot.append(r.CDi_total)
        cl.append(r.CL)
        cl_alpha.append((r2.CL - r1.CL) / np.deg2rad(2.0))
        e_tot.append(r.e_total)
    cdi_tot, cl = np.array(cdi_tot), np.array(cl)
    cl_alpha, e_tot = np.array(cl_alpha), np.array(e_tot)

    assert np.all(np.diff(cdi_tot) < 0)          # CDi drops as h drops
    assert np.all(np.diff(cl) > 0)               # CL rises at fixed alpha
    assert np.all(np.diff(cl_alpha) > 0)         # slope rises as h drops
    assert np.all(cl_alpha > cl_alpha_iso)       # ground only HELPS (vs iso)
    assert e_tot[-1] > iso.e                      # small h: e beats isolated
    assert np.all(np.diff(e_tot) > 0)            # e rises monotonically
    # ground drag term is a genuine reduction (negative) at every height
    for hb in h_over_b:
        r = solve_wing_ige(Surface(b=B, c=c, alpha_geo=al), h_agl=hb * B)
        assert r.CDi_ground < 0.0


# ----------------------------------------------------- 4. magnitude anchor

def test_induced_drag_reduction_in_classic_band():
    """[ANCHOR] At h/b = 0.25 the induced-drag reduction (elliptic wing,
    trimmed to a fixed CL) falls in the classic 20-45 % band. This is
    Prandtl's biplane sigma(g/b = 0.5) = 0.231 with an opposite-sign
    partner (Wieselsberger 1922 ground-effect correlation)."""
    _, y = cosine_stations(N, B)
    c = elliptic_chord(y, B, S)
    twist = np.zeros(N)
    CL_target = 0.5

    _, r_iso = solve_llt_trim(B, c, twist, CL_target)
    _, r_ge = solve_wing_ige_trim(B, c, twist, CL_target=CL_target, h_agl=0.25 * B)
    reduction = 1.0 - r_ge.CDi_total / r_iso.CDi

    print(f"\n[ground effect] induced-drag reduction at h/b = 0.25: "
          f"{reduction * 100:.2f} %  (CDi {r_iso.CDi:.5f} -> {r_ge.CDi_total:.5f})")
    assert 0.20 <= reduction <= 0.45


# ----------------------------------------------- 5. opposite-of-hydrofoil

def test_opposite_of_hydrofoil_fixed_circulation():
    """[DEFINITIVE SIGN TEST] Same wing, same gap, same fixed circulation:
    the free-surface (same-sign) image RAISES CDi by +sigma*CDi_self, the
    ground (opposite-sign) image LOWERS it by exactly the same amount."""
    _, y = cosine_stations(N, B)
    c = elliptic_chord(y, B, S)
    res = solve_llt(B, c, np.full(N, np.deg2rad(5.0)))
    real = VortexSystem(b=B, N=N)
    h = 0.15 * B

    img_surface = VortexSystem(b=B, N=N, z=+2.0 * h)   # free surface: above
    img_ground = VortexSystem(b=B, N=N, z=-2.0 * h)    # ground: below

    cdi_hydro, _ = mutual_cdi(real, img_surface, res.Gamma, res.Gamma,
                              V=1.0, Sref=S)
    cdi_ground, _ = mutual_cdi(real, img_ground, res.Gamma,
                               GROUND_IMAGE_SIGN * res.Gamma, V=1.0, Sref=S)

    assert cdi_hydro > 0.0                       # anti-ground: drag UP
    assert cdi_ground < 0.0                       # ground: drag DOWN
    assert cdi_ground == pytest.approx(-cdi_hydro, rel=1e-12)


def test_opposite_of_hydrofoil_full_solve():
    """Same A/B on the coupled solves: bracketing the isolated wing, the
    hydrofoil sits ABOVE its CDi and the ground wing BELOW."""
    surf = _rect_wing()
    h = 0.15 * B
    iso = solve_llt(B, surf.c, surf.alpha_geo)
    hr_hydro = solve_hydrofoil(surf, depth=h)
    hr_ground = solve_wing_ige(surf, h_agl=h)

    assert hr_hydro.CDi_surf > 0.0 > hr_ground.CDi_ground
    assert hr_hydro.CDi_total > iso.CDi > hr_ground.CDi_total
    assert hr_ground.e_total > iso.e > hr_hydro.e_total


# ------------------------------------------------------------------- 6. trim

def test_trim_hits_target_and_improves_LoD():
    """Closed-form trim hits CL_target to 1e-10; at the same CL_target the
    ground wing needs LESS alpha and its trimmed L/D IMPROVES vs free air."""
    c = np.full(N, S / B)
    twist = np.zeros(N)
    CL_target = 0.5
    CD0 = 0.02                                    # representative parasite floor

    a_iso, r_iso = solve_llt_trim(B, c, twist, CL_target)
    a_ge, r_ge = solve_wing_ige_trim(B, c, twist, CL_target=CL_target, h_agl=0.2 * B)

    assert r_ge.CL == pytest.approx(CL_target, abs=1e-10)
    assert isinstance(r_ge, GroundEffectResult)
    assert a_ge < a_iso                           # ground raises CL_alpha
    assert r_ge.CDi_total < r_iso.CDi             # induced drag drops

    LoD_iso = CL_target / (CD0 + r_iso.CDi)
    LoD_ge = CL_target / (CD0 + r_ge.CDi_total)
    print(f"\n[ground effect] trimmed L/D {LoD_iso:.2f} (free) -> "
          f"{LoD_ge:.2f} (h/b=0.2)")
    assert LoD_ge > LoD_iso


def test_deep_trim_matches_isolated():
    """Very high above ground, trim collapses to the isolated-wing trim."""
    c = np.full(N, S / B)
    twist = np.zeros(N)
    a_deep, r_deep = solve_wing_ige_trim(B, c, twist, CL_target=0.5, h_agl=1e4 * B)
    a_iso, iso = solve_llt_trim(B, c, twist, 0.5)
    assert a_deep == pytest.approx(a_iso, rel=1e-6)
    assert r_deep.CDi_total == pytest.approx(iso.CDi, rel=1e-8)


# ------------------------------------------------------------- guard / hygiene

def test_invalid_h_agl_raises():
    with pytest.raises(ValueError):
        build_ground_operators(VortexSystem(b=B, N=N), h_agl=0.0)
    with pytest.raises(ValueError):
        build_ground_operators(VortexSystem(b=B, N=N), h_agl=-1.0)


def test_z_translation_invariance():
    """Wing + ground translated together in z changes nothing (the image is
    always 2 h below the wing)."""
    c = np.full(N, S / B)
    al = np.full(N, np.deg2rad(5.0))
    r0 = solve_wing_ige(Surface(b=B, c=c, alpha_geo=al, z=0.0), h_agl=0.8)
    r5 = solve_wing_ige(Surface(b=B, c=c, alpha_geo=al, z=5.0), h_agl=0.8)
    assert r0.CL == pytest.approx(r5.CL, rel=1e-12)
    assert r0.CDi_total == pytest.approx(r5.CDi_total, rel=1e-12)
