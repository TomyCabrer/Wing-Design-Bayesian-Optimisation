"""Phase 1 gate: mission layer (ISA + Sutherland + water branch) and the
mission design-vector mode.

Gates:
  - ISA spot values (sea level, tropopause) + temperature continuity at 11 km
    + ValueError above the 20 km validity ceiling.
  - Water branch returns the hydrofoil.py constants BY IDENTITY (single
    source of truth, not re-typed numbers).
  - default_mission reproduces the legacy Problem defaults: cl_target = 0.5
    exactly, and Problem(mission=...) is bit-for-bit the legacy objective.
  - mission mode: 5-D bounds, sea-level point reproduces the trim probe,
    altitude raises CL_target (same weight, thinner air), -100.0 contract.
"""

import numpy as np
import pytest

from aerobo import hydrofoil
from aerobo.geometry import MISSION_ALT_BOUNDS_M, MISSION_V_BOUNDS, bounds
from aerobo.mission import (
    MissionSpec,
    default_mission,
    isa_density,
    isa_pressure,
    isa_temperature,
    sutherland_mu,
)
from aerobo.objective import PENALTY, Problem, evaluate, objective

# ---------------- ISA + Sutherland ----------------


def test_isa_sea_level_density():
    assert isa_density(0.0) == pytest.approx(1.225, rel=1e-3)


def test_isa_tropopause_density():
    # US Standard Atmosphere 1976 tabulated value at 11 km
    assert isa_density(11000.0) == pytest.approx(0.3639, rel=2e-3)


def test_isa_temperature_continuous_at_tropopause():
    assert isa_temperature(11000.0 - 1e-6) == pytest.approx(
        isa_temperature(11000.0 + 1e-6), rel=1e-9
    )
    assert isa_temperature(11000.0) == pytest.approx(216.65, abs=1e-9)


def test_isa_pressure_continuous_at_tropopause():
    assert isa_pressure(11000.0) == pytest.approx(
        isa_pressure(11000.0 + 1e-6), rel=1e-9
    )


def test_sutherland_sea_level_viscosity():
    assert sutherland_mu(isa_temperature(0.0)) == pytest.approx(1.789e-5, rel=1e-2)


def test_isa_raises_above_20km():
    with pytest.raises(ValueError):
        isa_temperature(20000.1)
    m = MissionSpec(W_N=1000.0, V=15.0, altitude_m=25000.0)
    with pytest.raises(ValueError):
        m.rho
    with pytest.raises(ValueError):
        m.mu


# ---------------- water branch ----------------


def test_water_branch_uses_hydrofoil_constants_by_identity():
    m = MissionSpec(W_N=6000.0, V=10.0, depth_m=0.5, medium="water")
    # single source of truth: the SAME objects, not re-typed numbers
    assert m.rho is hydrofoil.RHO_WATER
    assert m.mu is hydrofoil.MU_WATER


def test_unknown_medium_rejected():
    with pytest.raises(ValueError):
        MissionSpec(W_N=1.0, V=1.0, medium="vacuum")


# ---------------- default mission ----------------


def test_default_mission_hits_legacy_cl_target_exactly():
    m = default_mission()
    assert m.cl_target(10.0) == pytest.approx(0.5, abs=1e-14)
    assert (m.V, m.altitude_m, m.medium) == (14.6, 0.0, "air")


# ---------------- bit-for-bit regression ----------------

PROBE_X = np.array([0.4, 1.0, -2.0])
PROBE_LOD = 34.761763013754106   # frozen 2026-07-12 phase 1


def test_mission_default_is_bit_for_bit_with_legacy():
    y_legacy = objective(PROBE_X, Problem())
    y_mission = objective(PROBE_X, Problem(mission=default_mission()))
    assert y_legacy == y_mission                      # EXACT, no tolerance
    assert y_legacy == pytest.approx(PROBE_LOD, abs=1e-12)


# ---------------- mission mode ----------------


def test_mission_mode_bounds_shape_and_rows():
    b = bounds("mission")
    assert b.shape == (5, 2)
    assert b[3].tolist() == list(MISSION_V_BOUNDS)
    assert b[4].tolist() == list(MISSION_ALT_BOUNDS_M)
    assert Problem(mode="mission", mission=default_mission()).dim == 5


def test_mission_mode_sea_level_reproduces_trim_probe():
    prob = Problem(mode="mission", mission=default_mission())
    out = evaluate(np.array([0.4, 1.0, -2.0, 14.6, 0.0]), prob)
    assert out["feasible"], out["reason"]
    assert out["LoD"] == pytest.approx(PROBE_LOD, abs=1e-12)
    assert out["CL_target"] == pytest.approx(0.5, abs=1e-14)


def test_mission_mode_altitude_raises_cl_target():
    # same weight, thinner air -> higher trim CL (and it must still trim)
    prob = Problem(mode="mission", mission=default_mission())
    sl = evaluate(np.array([0.4, 1.0, -2.0, 14.6, 0.0]), prob)
    hi = evaluate(np.array([0.4, 1.0, -2.0, 14.6, 3000.0]), prob)
    assert sl["feasible"] and hi["feasible"], (sl["reason"], hi["reason"])
    assert hi["CL_target"] > sl["CL_target"]
    assert hi["CL"] == pytest.approx(hi["CL_target"], abs=1e-6)  # trim hit


def test_mission_mode_out_of_bounds_v_is_penalty():
    prob = Problem(mode="mission", mission=default_mission())
    assert objective(np.array([0.4, 1.0, -2.0, 30.0, 0.0]), prob) == PENALTY
    assert objective(np.array([0.4, 1.0, -2.0, 5.0, 0.0]), prob) == PENALTY
    assert objective(np.array([0.4, 1.0, -2.0, 14.6, 5000.0]), prob) == PENALTY


def test_mission_mode_without_mission_is_penalty():
    prob = Problem(mode="mission")   # no MissionSpec -> in-contract failure
    x = np.array([0.4, 1.0, -2.0, 14.6, 0.0])
    assert objective(x, prob) == PENALTY
    out = evaluate(x, prob)
    assert not out["feasible"]
    assert "mission" in out["reason"]


def test_mission_mode_water_medium_is_penalty():
    # mode="mission" is air-only (altitude design variable, air polar);
    # water missions belong to hydrofoil.HydrofoilProblem.
    water = MissionSpec(W_N=6000.0, V=12.0, depth_m=0.5, medium="water")
    prob = Problem(mode="mission", mission=water)
    x = np.array([0.4, 1.0, -2.0, 14.6, 0.0])
    assert objective(x, prob) == PENALTY
    out = evaluate(x, prob)
    assert not out["feasible"]
    assert "air-only" in out["reason"]
