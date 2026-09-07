"""Item 3 gate: nonplanar VLM (winglets) agrees with the LLT on planar wings
and reproduces the classical nonplanar physics.

1. Planar rectangular AR=10: VLM CL within 5% of the LLT kernel (measured
   -4.6%, the converged Weissinger/lifting-surface vs lifting-line offset);
   CDi and e at fixed lift within a few %.
2. Tapered + twisted planar wing at trim: CDi/e within 2% of the LLT.
3. Munk bound: every planar wing has CDi >= CL^2/(pi*AR) and e < 1 — the
   discrete Trefftz quadrature (edges at cosine angles, wash at interlaced
   cosine midpoints) is exact for Fourier loading modes, so the bound holds
   without slack (Munk 1921).
4. Zero lift -> zero CDi (exact for symmetric sections; ~0 with camber).
5. Symmetric geometry -> symmetric loading and zero side force.
6. Winglet physics: at fixed total lift, Trefftz span efficiency e rises
   monotonically with winglet height and exceeds e = 1 (referenced to the
   projected planar span — possible only for nonplanar systems, Munk 1921).
7. Objective wiring: winglet mode (d = 5) with the -100.0 penalty contract.
"""

import time

import numpy as np
import pytest

from aerobo.geometry import Wing, bounds
from aerobo.llt import solve_llt, solve_llt_trim
from aerobo.objective import PENALTY, Problem, evaluate, objective
from aerobo.vlm import VLM, solve_vlm

AR = 10.0
B = 10.0
S = B**2 / AR
N_LLT = 60
N_VLM = 40
TWO_PI = 2.0 * np.pi


def _llt(wing: Wing, alpha_deg: float | None = None, CL_t: float | None = None):
    _, c, tw = wing.sample(N_LLT)
    if CL_t is not None:
        return solve_llt_trim(B, c, tw, CL_t)[1]
    return solve_llt(B, c, np.deg2rad(alpha_deg) + tw)


# ---------------- planar agreement with the LLT kernel ----------------

def test_vlm_matches_llt_rectangular():
    w = Wing()  # rectangular, untwisted
    llt = _llt(w, alpha_deg=5.0)
    v = solve_vlm(w, np.deg2rad(5.0), N=N_VLM)
    # lifting-surface (Weissinger) vs lifting-line CL offset: measured -4.6%,
    # N-independent — the physical difference between the two theories.
    assert v.CL == pytest.approx(llt.CL, rel=0.05)
    # at FIXED LIFT the induced-drag agreement is the meaningful one:
    m = VLM(w, N=N_VLM)
    _, vt = m.solve_trim(0.5)
    _, lt = solve_llt_trim(B, w.sample(N_LLT)[1], np.zeros(N_LLT), 0.5)
    assert vt.CDi == pytest.approx(lt.CDi, rel=0.05)
    assert vt.e == pytest.approx(lt.e, rel=0.05)
    assert v.AR == pytest.approx(AR, rel=1e-9)


def test_vlm_matches_llt_tapered_twisted():
    w = Wing(taper=0.5, twist_root_deg=0.0, twist_tip_deg=-3.0)
    m = VLM(w, N=N_VLM)
    _, v = m.solve_trim(0.5)
    _, c, tw = w.sample(N_LLT)
    _, llt = solve_llt_trim(B, c, tw, 0.5)
    assert v.CL == pytest.approx(0.5, abs=1e-12)   # closed-form trim is exact
    assert v.CDi == pytest.approx(llt.CDi, rel=0.02)   # measured -0.8%
    assert v.e == pytest.approx(llt.e, rel=0.02)


# ---------------- Munk bound (planar) ----------------

def test_planar_munk_bound_and_e_below_unity():
    """Any planar wing: CDi >= CL^2/(pi*AR), i.e. e < 1 (Munk 1921). The
    interlaced-cosine Trefftz quadrature makes this hold with NO slack."""
    rng = np.random.default_rng(0)
    for _ in range(10):
        x = rng.uniform([0.2, -4.0, -6.0], [1.0, 4.0, 2.0])
        w = Wing(taper=x[0], twist_root_deg=x[1], twist_tip_deg=x[2])
        _, v = VLM(w, N=N_VLM).solve_trim(0.5)
        munk = v.CL**2 / (np.pi * v.AR)
        assert v.CDi >= munk * (1.0 - 1e-9), (x, v.CDi / munk)
        assert v.e < 1.0


def test_prescribed_elliptic_loading_gives_e_unity():
    """Discrete Trefftz consistency: elliptic circulation prescribed directly
    on the lattice returns e = 1 to near machine precision at any N."""
    for N in (20, 40):
        m = VLM(Wing(), N=N)
        G = np.sqrt(np.maximum(1.0 - (2.0 * m.y / B) ** 2, 0.0))
        CL = 2.0 * float(G @ m.lvec[:, 1]) / m.S
        CDi = -float((m.width * G) @ (m._WN @ G)) / m.S
        e = CL**2 / (np.pi * m.AR * CDi)
        assert e == pytest.approx(1.0, abs=1e-9)


# ---------------- degenerate cases ----------------

def test_zero_lift_zero_cdi():
    # symmetric section (alpha_L0 = 0), no twist, alpha = 0: Gamma = 0 exactly
    for h in (0.0, 0.1):
        r = solve_vlm(Wing(), 0.0, N=N_VLM, winglet_h_frac=h)
        assert abs(r.CL) < 1e-14
        assert abs(r.CDi) < 1e-14
    # cambered section at its zero-lift alpha: CL ~ 0, CDi ~ 0 (planar wing;
    # the alpha-linearised freestream leaves an O(theta^3) residual)
    r = solve_vlm(Wing(), np.deg2rad(-2.0), N=N_VLM, alpha_L0=np.deg2rad(-2.0))
    assert abs(r.CL) < 1e-3
    assert r.CDi < 1e-8


def test_symmetric_loading_and_zero_side_force():
    m = VLM(Wing(taper=0.6, twist_tip_deg=-2.0), N=N_VLM,
            winglet_h_frac=0.1, winglet_cant_deg=75.0)
    r = m.solve(np.deg2rad(5.0))
    scale = np.abs(r.Gamma).max()
    assert np.abs(r.Gamma - r.Gamma[::-1]).max() < 1e-12 * scale
    assert abs(r.CY) < 1e-12


# ---------------- winglet physics ----------------

def test_winglet_raises_trefftz_e_monotonically():
    """Vertical winglet at fixed total lift: e rises with height and exceeds
    the planar-span limit e = 1 (nonplanar systems only — Munk 1921).
    Measured (rectangle, cant 90): e = 0.962, 1.019, 1.076, 1.130."""
    es = []
    for h in (0.0, 0.05, 0.10, 0.15):
        m = VLM(Wing(), N=N_VLM, winglet_h_frac=h, winglet_cant_deg=90.0)
        _, r = m.solve_trim(0.5)
        es.append(r.e)
    assert all(e2 > e1 for e1, e2 in zip(es, es[1:])), es
    assert es[2] > es[0] + 0.05          # h = 0.1 semi-span: > +5 pts of e
    assert es[3] > 1.0                   # nonplanar: beats the planar bound


def test_winglet_cant_below_90_raises_e_further():
    """Lower cant tilts the winglet outboard (raked-tip limit): projected
    span grows, so e referenced to the nominal b keeps rising."""
    es = {}
    for cant in (60.0, 90.0):
        m = VLM(Wing(), N=N_VLM, winglet_h_frac=0.15, winglet_cant_deg=cant)
        _, r = m.solve_trim(0.5)
        es[cant] = r.e
    assert es[60.0] > es[90.0] > 1.0


# ---------------- objective wiring (winglet mode) ----------------

def test_winglet_bounds_shape():
    assert bounds("winglet").shape == (5, 2)
    assert Problem(mode="winglet").dim == 5


def test_winglet_mode_breakdown_consistency():
    prob = Problem(mode="winglet")
    out = evaluate(np.array([0.5, 0.0, 0.0, 0.10, 90.0]), prob)
    assert out["feasible"], out["reason"]
    assert out["CL"] == pytest.approx(0.5, abs=1e-9)
    assert out["CD"] == pytest.approx(out["CDi"] + out["CDp"] + out["cd0_extra"])
    assert out["LoD"] == pytest.approx(out["CL"] / out["CD"], rel=1e-9)
    assert 10.0 < out["LoD"] < 60.0
    assert out["winglet"]["h_m"] == pytest.approx(0.5)
    assert out["winglet"]["S_planform"] > 0.0
    # winglet mode may exceed e = 1 (nonplanar), but not the h=0.1 optimum
    assert 0.9 < out["e"] < 1.15


def test_winglet_h_zero_matches_planar_and_llt():
    prob = Problem(mode="winglet")
    out0 = evaluate(np.array([0.5, 0.0, 0.0, 0.0, 90.0]), prob)
    # below MIN_WINGLET_FRAC the winglet is dropped: bit-identical to h = 0
    out_eps = evaluate(np.array([0.5, 0.0, 0.0, 0.004, 90.0]), prob)
    assert out0["LoD"] == out_eps["LoD"]
    assert out0["winglet"]["S_planform"] == 0.0
    # cross-solver: VLM planar objective within 5% of the LLT trim objective
    out_llt = evaluate(np.array([0.5, 0.0, 0.0]), Problem(mode="trim"))
    assert out0["LoD"] == pytest.approx(out_llt["LoD"], rel=0.05)


def test_winglet_penalty_contract():
    prob = Problem(mode="winglet")
    assert objective(np.array([0.5, 0.0, 0.0, 0.20, 90.0]), prob) == PENALTY
    assert objective(np.array([0.5, 0.0, 0.0, 0.10, 45.0]), prob) == PENALTY
    assert objective(np.array([0.5, 0.0, 0.0, 0.10, 90.0]),
                     Problem(mode="winglet", CL_target=5.0)) == PENALTY


def test_winglet_objective_finite_over_random_sample():
    rng = np.random.default_rng(3)
    prob = Problem(mode="winglet")
    bnds = prob.bounds
    for _ in range(10):
        x = rng.uniform(bnds[:, 0], bnds[:, 1])
        v = objective(x, prob)
        assert np.isfinite(v)
        assert v == PENALTY or v > 0.0


def test_vlm_is_fast_enough():
    """Build + trim at default panel counts: measured ~1 ms (loose guard)."""
    t0 = time.perf_counter()
    for _ in range(10):
        VLM(Wing(taper=0.5), N=N_VLM, winglet_h_frac=0.1).solve_trim(0.5)
    assert (time.perf_counter() - t0) / 10 < 0.25
