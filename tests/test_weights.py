"""Structural closure gates: Raymer wing weight + weight fixed point +
root-spar bending stress (weights.py).

Gates:
  - Contract constants frozen: G0 (identity with mission.py), N_ULT_DEFAULT.
  - wing_weight_raymer monotone: UP in b and n_ult, DOWN in tc (Raymer's
    -0.3 thickness exponent — statistically, thicker sections carry bending
    with less cap material), UP in sweep.
  - Unit spot-check: the GA reference design lands in a plausible GA band
    (400-1600 N ~ 40-160 kg) — catches any SI/imperial conversion slip.
  - W_fw = 0 guard: dry wing gives finite NONZERO weight (0^0.0035 trap),
    and the factor for 100 lb of wing fuel is the documented ~1.016.
  - total_weight fixed point: converged residual < 1e-6 N, W_total >
    W_fixed, returned W_wing consistent.
  - root_bending_stress sign sanities (up in b and n_ult, down in tc) and
    the reference design within a factor ~2 of SIGMA_ALLOW_PA (calibration
    freeze: 255.8 MPa = 1.11 x allowable).
"""

import numpy as np
import pytest

from aerobo import mission
from aerobo.weights import (
    G0,
    K_CAP,
    N_ULT_DEFAULT,
    SIGMA_ALLOW_PA,
    root_bending_stress,
    total_weight,
    wing_weight_raymer,
)

# Reference design used throughout (spec + K_CAP calibration point):
REF = dict(b=11.0, S=16.0, taper=0.45, sweep_deg=0.0, tc=0.12, n_ult=5.7)
REF_W_DG_N = 9500.0    # ~970 kg design gross weight


def w_ref(**over):
    kw = {**REF, "W_dg_N": REF_W_DG_N, **over}
    return wing_weight_raymer(
        kw["b"], kw["S"], kw["taper"], kw["sweep_deg"], kw["tc"],
        kw["n_ult"], kw["W_dg_N"],
    )


# ---------------- contract constants ----------------


def test_contract_constants_frozen():
    assert G0 == 9.80665
    assert G0 is mission.G0            # single source of truth, by identity
    assert N_ULT_DEFAULT == pytest.approx(5.7)   # 3.8 limit x 1.5


# ---------------- Raymer wing weight ----------------


def test_wing_weight_monotone_increasing_in_span():
    weights = [w_ref(b=b) for b in (8.0, 10.0, 12.0, 14.0)]
    assert all(w1 > w0 for w0, w1 in zip(weights, weights[1:])), weights


def test_wing_weight_monotone_increasing_in_n_ult():
    weights = [w_ref(n_ult=n) for n in (3.0, 4.5, 5.7, 7.0)]
    assert all(w1 > w0 for w0, w1 in zip(weights, weights[1:])), weights


def test_wing_weight_monotone_decreasing_in_tc():
    # Raymer's (100 t/c / cos L)^-0.3: thicker section = LIGHTER wing for
    # the same strength in the statistical record (deeper structural box).
    weights = [w_ref(tc=tc) for tc in (0.08, 0.10, 0.12, 0.16)]
    assert all(w1 < w0 for w0, w1 in zip(weights, weights[1:])), weights


def test_wing_weight_monotone_increasing_in_sweep():
    # (A/cos^2 L)^0.6 * (100 tc / cos L)^-0.3 -> net cos^-0.9: sweep costs.
    weights = [w_ref(sweep_deg=s) for s in (0.0, 10.0, 20.0, 30.0)]
    assert all(w1 > w0 for w0, w1 in zip(weights, weights[1:])), weights


def test_wing_weight_ga_reference_band():
    # Unit spot-check: any SI/imperial slip moves this by orders of magnitude
    # (e.g. feeding m^2 as ft^2 or N as lb). Frozen value: 1258.6 N.
    W = w_ref()
    assert 400.0 < W < 1600.0, f"W_wing = {W:.1f} N outside GA band 400-1600 N"
    assert W == pytest.approx(1258.6349, rel=1e-6)   # calibration freeze


def test_wing_weight_dry_wing_guard():
    # W_fw = 0 must NOT zero the product (0^0.0035 = 0 trap): finite, nonzero.
    W_dry = w_ref()          # default W_fw_N = 0
    assert np.isfinite(W_dry) and W_dry > 0.0
    # 100 lb of wing fuel nudges the factor by the documented ~1.6%.
    W_fuel = wing_weight_raymer(
        REF["b"], REF["S"], REF["taper"], REF["sweep_deg"], REF["tc"],
        REF["n_ult"], REF_W_DG_N, W_fw_N=100.0 * 4.4482216152605,
    )
    assert W_fuel / W_dry == pytest.approx(1.0162, abs=2e-3)


def test_wing_weight_rejects_degenerate_inputs():
    with pytest.raises(ValueError):
        w_ref(taper=0.0)      # taper^0.04 degenerates
    with pytest.raises(ValueError):
        w_ref(tc=0.0)
    with pytest.raises(ValueError):
        w_ref(sweep_deg=90.0)


# ---------------- weight fixed point ----------------


def test_total_weight_fixed_point_converged():
    W_fixed = 8000.0
    W_total, W_wing = total_weight(
        W_fixed, REF["b"], REF["S"], REF["taper"], REF["sweep_deg"],
        REF["tc"], REF["n_ult"],
    )
    # self-consistency: W_total = W_fixed + W_wing(W_dg = W_total)
    resid = W_total - (W_fixed + wing_weight_raymer(
        REF["b"], REF["S"], REF["taper"], REF["sweep_deg"], REF["tc"],
        REF["n_ult"], W_dg_N=W_total,
    ))
    assert abs(resid) < 1e-6, f"fixed-point residual {resid:.3e} N"
    assert W_total > W_fixed
    assert W_total == pytest.approx(W_fixed + W_wing, abs=1e-9)


def test_total_weight_grows_with_span():
    # the aero-structural trade the aircraft problem is built on
    totals = [
        total_weight(8000.0, b, REF["S"], REF["taper"], 0.0, REF["tc"])[0]
        for b in (8.0, 11.0, 14.0)
    ]
    assert totals[0] < totals[1] < totals[2], totals


# ---------------- root bending stress ----------------


def sig_ref(**over):
    kw = {**REF, "W_total_N": REF_W_DG_N, **over}
    return root_bending_stress(
        kw["b"], kw["S"], kw["taper"], kw["tc"], kw["W_total_N"], kw["n_ult"],
    )


def test_stress_up_in_span():
    # sigma ~ b^4 at fixed S, taper (arm up, c_root and box down)
    s = [sig_ref(b=b) for b in (9.0, 11.0, 13.0)]
    assert s[0] < s[1] < s[2], s


def test_stress_up_in_n_ult():
    assert sig_ref(n_ult=7.0) > sig_ref(n_ult=5.7) > sig_ref(n_ult=3.0)


def test_stress_down_in_tc():
    # ~ tc^-2 (cap area AND box depth both scale with tc)
    s = [sig_ref(tc=tc) for tc in (0.08, 0.12, 0.16)]
    assert s[0] > s[1] > s[2], s


def test_reference_design_near_allowable():
    # K_CAP calibration freeze: sigma_ref = 255.8 MPa = 1.11 x allowable,
    # i.e. the strength constraint is mildly ACTIVE at the reference design.
    sigma = sig_ref()
    assert SIGMA_ALLOW_PA / 2.0 < sigma < SIGMA_ALLOW_PA * 2.0, (
        f"sigma_ref = {sigma/1e6:.1f} MPa vs allowable "
        f"{SIGMA_ALLOW_PA/1e6:.0f} MPa"
    )
    assert sigma == pytest.approx(255.76e6, rel=1e-3)
    assert K_CAP == pytest.approx(0.0025)
    assert SIGMA_ALLOW_PA == pytest.approx(230e6)   # 2024-T3 345 MPa / 1.5
