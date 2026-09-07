"""M4 gate: polar ingestion, drag build-up, geometry and objective contract."""

import numpy as np
import pytest

from aerobo.drag import fuselage_cd0, skin_friction_cf, wing_cd0
from aerobo.geometry import Wing, bounds, wing_from_x
from aerobo.objective import PENALTY, Problem, evaluate, objective
from aerobo.polar import AnalyticPolar, default_polar

# ---------------- polar ----------------

def test_xfoil_polar_loads_and_extracts_linear_region():
    pol = default_polar()
    if isinstance(pol, AnalyticPolar):
        pytest.skip("no XFOIL polar file present; analytic fallback in use")
    # NACA 2412 at Re 1e6: alpha_L0 ~ -2 deg, a_lin ~ 2*pi (within ~15%)
    assert -4.0 < np.rad2deg(pol.alpha_L0) < -1.0
    assert 0.85 * 2 * np.pi < pol.a_lin < 1.15 * 2 * np.pi
    assert pol.cd(2.0) < 0.02
    lo, hi = pol.alpha_valid
    assert lo <= -5.0 and hi >= 10.0


def test_analytic_polar_is_thin_airfoil():
    pol = AnalyticPolar(alpha_L0_deg=-2.0)
    assert pol.cl(-2.0) == pytest.approx(0.0, abs=1e-12)
    assert pol.cl(3.73) == pytest.approx(2 * np.pi * np.deg2rad(5.73), rel=1e-3)
    assert "PLACEHOLDER" in pol.name


# ---------------- drag ----------------

def test_skin_friction_turbulent_magnitude():
    # Re = 1e6 turbulent flat plate: Cf ~ 0.0045 (Raymer 12.27)
    cf = skin_friction_cf(1e6, lref=1.0)
    assert 0.003 < cf < 0.006


def test_wing_cd0_sensible():
    comp = wing_cd0(Sref=10.0, Sexp=10.0, tc=0.12, mac=1.0,
                    rho=1.225, V=14.6, mu=1.789e-5)
    assert 0.004 < comp.CD0 < 0.020
    assert comp.FF > 1.0
    assert comp.Swet == pytest.approx(10.0 * (1.977 + 0.52 * 0.12))


def test_fuselage_cd0_extension_point():
    comp = fuselage_cd0(Sref=10.0, L=8.0, D=1.0, rho=1.225, V=14.6, mu=1.789e-5)
    assert comp.CD0 > 0.0


# ---------------- geometry ----------------

def test_wing_geometry_area_and_taper():
    w = Wing(b=10.0, S=10.0, taper=0.5)
    y = np.linspace(-5, 5, 20001)
    S_num = np.trapezoid(w.chord(y), y)
    assert S_num == pytest.approx(10.0, rel=1e-4)
    assert w.chord(np.array([0.0]))[0] == pytest.approx(w.c_root)
    assert w.chord(np.array([5.0]))[0] == pytest.approx(w.c_root * 0.5)
    assert w.AR == pytest.approx(10.0)


def test_twist_linear_and_sign_convention():
    w = Wing(twist_root_deg=2.0, twist_tip_deg=-4.0)
    tw = w.twist_deg(np.array([0.0, 2.5, 5.0]))
    assert tw == pytest.approx([2.0, -1.0, -4.0])


# ---------------- objective contract ----------------

def test_penalty_on_bounds_violation():
    assert objective(np.array([0.05, 0.0, 0.0])) == PENALTY   # taper below lb
    assert objective(np.array([0.5, 10.0, 0.0])) == PENALTY   # twist above ub


def test_penalty_on_untrimmable_cl():
    prob = Problem(CL_target=5.0)  # unreachable within the alpha bracket
    assert objective(np.array([0.5, 0.0, 0.0]), prob) == PENALTY
    out = evaluate(np.array([0.5, 0.0, 0.0]), prob)
    assert not out["feasible"]


def test_known_geometry_breakdown_consistency():
    out = evaluate(np.array([0.5, 0.0, 0.0]))
    assert out["feasible"], out["reason"]
    assert out["CL"] == pytest.approx(0.5, abs=1e-6)      # trim hits target
    assert out["CD"] == pytest.approx(out["CDi"] + out["CDp"] + out["cd0_extra"])
    assert out["LoD"] == pytest.approx(out["CL"] / out["CD"], rel=1e-9)
    assert 10.0 < out["LoD"] < 60.0                       # sane wing, AR 10
    assert 0.85 < out["e"] <= 1.0
    assert out["Re_mac"] == pytest.approx(1e6, rel=0.05)
    # CDi bounded below by the elliptic optimum
    assert out["CDi"] >= out["CL"] ** 2 / (np.pi * out["AR"]) * (1 - 1e-9)


def test_soft_mode_penalises_cl_miss():
    prob = Problem(mode="soft")
    x = np.array([0.5, 0.0, 0.0, 4.0])
    out = evaluate(x, prob)
    assert out["feasible"], out["reason"]
    assert out["score"] == pytest.approx(
        out["LoD"] - prob.soft_weight * (out["CL"] - prob.CL_target) ** 2
    )


def test_objective_finite_over_random_sample():
    rng = np.random.default_rng(3)
    prob = Problem()
    bnds = prob.bounds
    for _ in range(10):
        x = rng.uniform(bnds[:, 0], bnds[:, 1])
        v = objective(x, prob)
        assert np.isfinite(v)
        assert v == PENALTY or v > 0.0


def test_wing_from_x_roundtrip():
    x = np.array([0.7, 1.0, -3.0])
    w = wing_from_x(x)
    assert (w.taper, w.twist_root_deg, w.twist_tip_deg) == (0.7, 1.0, -3.0)
    assert bounds("trim").shape == (3, 2)
    assert bounds("soft").shape == (4, 2)
