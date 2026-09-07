"""Phase 1b gates: cavitation number, Cp_min data plumbing, constraint trend.

1. [ANALYTIC] sigma_cav(h, V) against hand numbers; monotone in h (deeper =
   more static head = harder to cavitate) and V (faster = tighter).
2. Cp_min interpolation is EXACT at tabulated alphas (real XFOIL CPMN data,
   companion .cpmin files) for every NACA 24XX family member; blended
   polars mix members linearly (exact at members).
3. AnalyticPolar.cp_min PLACEHOLDER stays close to its fit source (real
   NACA 2412 CPMN at alpha 0-6 deg).
4. Constraint activation on a FIXED foil: g > 0 slow/deep, g < 0
   fast/shallow, monotone released by slowing down or going deeper.
"""

import numpy as np
import pytest
from pathlib import Path

from aerobo.hydrofoil import (
    P_ATM,
    P_VAP,
    RHO_WATER,
    cavitation_margin,
    sigma_cav,
    solve_hydrofoil_trim,
)
from aerobo.polar import AnalyticPolar, default_polar, default_polar_family

DATA = Path(__file__).resolve().parents[1] / "data" / "airfoils"

# realistic Tier C scale: b = 1.2 m, AR = 10 -> S = 0.144 m^2, c = 0.12 m
B, S, N = 1.2, 0.144, 40


def _trimmed(depth: float, CL=0.5):
    pol = default_polar()
    c = np.full(N, S / B)
    return pol, solve_hydrofoil_trim(
        B, c, np.zeros(N), CL_target=CL, depth=depth,
        a=pol.a_lin, alpha_L0=pol.alpha_L0,
    )[1]


# ------------------------------------------------------------ 1. sigma_cav

def test_sigma_cav_hand_number():
    """[ANALYTIC] V = 10 m/s, h = 1 m, seawater:
    (101325 + 1025*9.81*1 - 2340) / (0.5*1025*100) = 2.12761..."""
    expect = (101325.0 + 1025.0 * 9.81 * 1.0 - 2340.0) / (0.5 * 1025.0 * 100.0)
    assert sigma_cav(1.0, 10.0) == pytest.approx(expect, rel=1e-12)
    assert sigma_cav(1.0, 10.0) == pytest.approx(2.12761, abs=1e-4)
    # constants sanity
    assert P_ATM == 101325.0 and P_VAP == 2340.0 and RHO_WATER == 1025.0


def test_sigma_cav_monotone():
    assert sigma_cav(2.0, 10.0) > sigma_cav(0.5, 10.0)     # deeper = safer
    assert sigma_cav(1.0, 8.0) > sigma_cav(1.0, 14.0)      # slower = safer


# ------------------------------------------------------- 2. cp_min plumbing

def test_cp_min_exact_at_tabulated_alphas():
    fam = default_polar_family()
    for tc, member in fam.members.items():
        assert member.has_cp_min, f"missing .cpmin for t/c={tc}"
        raw = np.loadtxt(DATA / f"naca24{int(tc*100):02d}_re1e6.cpmin")
        a_tab, cp_tab = raw[:, 0], raw[:, 1]
        assert np.allclose(member.cp_min(a_tab), cp_tab, atol=1e-12)


def test_cp_min_blend_linear_between_members():
    fam = default_polar_family()
    tc = 0.135                                  # midway 2412 <-> 2415
    blend = fam.at(tc)
    lo, hi = fam.at(0.12), fam.at(0.15)
    a = np.array([0.0, 3.0, 6.0])
    assert np.allclose(blend.cp_min(a), 0.5 * (lo.cp_min(a) + hi.cp_min(a)),
                       rtol=1e-12)


def test_cp_min_thicker_section_milder_suction():
    """Physics sanity on the REAL data: at moderate lift, thicker NACA 24XX
    sections have milder suction peaks (less negative Cp_min)."""
    fam = default_polar_family()
    a = 4.0
    assert fam.at(0.18).cp_min(a) > fam.at(0.12).cp_min(a) > fam.at(0.09).cp_min(a)


def test_analytic_placeholder_close_to_fit_source():
    ana = AnalyticPolar()
    real = default_polar()      # NACA 2412 XFOIL (fit source)
    for a in (0.0, 2.0, 4.0, 6.0):
        assert ana.cp_min(a) < 0.0
        # PLACEHOLDER tracks the real data loosely in its fit range
        assert ana.cp_min(a) == pytest.approx(real.cp_min(a), abs=0.35)
    assert ana.cp_min(6.0) < ana.cp_min(2.0)    # more lift = deeper suction


# --------------------------------------------------- 4. activation trend

def test_constraint_activates_fast_shallow_releases_slow_deep():
    pol, hr_shallow = _trimmed(depth=0.3)
    _, hr_deep = _trimmed(depth=1.0)

    g_fast_shallow = cavitation_margin(hr_shallow, pol, V=16.0)["g"]
    g_slow_shallow = cavitation_margin(hr_shallow, pol, V=8.0)["g"]
    g_fast_deep = cavitation_margin(hr_deep, pol, V=16.0)["g"]
    g_slow_deep = cavitation_margin(hr_deep, pol, V=8.0)["g"]

    assert g_fast_shallow < 0.0                 # cavitates: fast + shallow
    assert g_slow_deep > 0.0                    # safe: slow + deep
    assert g_slow_shallow > g_fast_shallow      # slowing down releases
    assert g_fast_deep > g_fast_shallow         # going deeper releases


def test_margin_breakdown_fields():
    pol, hr = _trimmed(depth=0.5)
    m = cavitation_margin(hr, pol, V=12.0)
    assert m["g"] == pytest.approx(m["sigma_cav"] + m["cp_min_worst"], rel=1e-12)
    assert m["cp_min_worst"] < 0.0
    assert m["cp_min_y"].shape == (N,)
    assert abs(m["y_worst"]) <= B / 2
