"""Item 2 gate: t/c + sweep dimensions (XFOIL polar family, tier_a_plus mode).

Gates:
  - PolarFamily reduces EXACTLY to the member polar at member t/c and blends
    linearly in t/c between members.
  - Raymer FF rises with t/c and falls with sweep; sweep = 0 keeps the old
    incompressible value.
  - tier_a_plus at sweep = 0, tc = 0.12 reproduces the Tier A 3-D trim
    objective EXACTLY (bit-for-bit regression gate).
  - Physics direction checks + the -100.0 penalty contract on the 5-D box.
  - BO smoke run (budget 30) beats the untwisted-rectangle baseline.
"""

import numpy as np
import pytest

from aerobo.drag import wing_form_factor
from aerobo.geometry import Wing, bounds, wing_from_x
from aerobo.llt import swept_section_slope
from aerobo.objective import PENALTY, Problem, evaluate, objective
from aerobo.optimize.bo import run_bo
from aerobo.polar import BlendedPolar, default_polar_family

try:
    FAMILY = default_polar_family()
except FileNotFoundError:  # pragma: no cover - data files are committed
    FAMILY = None

needs_family = pytest.mark.skipif(
    FAMILY is None, reason="naca24XX polar family files missing"
)

# ---------------- polar family ----------------


@needs_family
def test_family_covers_naca24xx_range():
    assert FAMILY.tc_range == (0.06, 0.18)
    assert sorted(FAMILY.members) == [0.06, 0.09, 0.12, 0.15, 0.18]


@needs_family
def test_family_exact_at_members():
    # at a member t/c the family must reduce EXACTLY to that member polar
    for tc, member in FAMILY.members.items():
        got = FAMILY.at(tc)
        assert got is member


@needs_family
def test_family_linear_blend_between_members():
    a = np.linspace(-4.0, 8.0, 25)
    p09, p12 = FAMILY.members[0.09], FAMILY.members[0.12]
    mid = FAMILY.at(0.105)  # exactly halfway
    assert isinstance(mid, BlendedPolar)
    assert np.array_equal(mid.cd(a), 0.5 * (p09.cd(a) + p12.cd(a)))
    assert np.array_equal(mid.cl(a), 0.5 * (p09.cl(a) + p12.cl(a)))
    assert mid.a_lin == pytest.approx(0.5 * (p09.a_lin + p12.a_lin))
    # blend weight moves monotonically: cd(2 deg) between the members
    lo, hi = sorted([p09.cd(2.0), p12.cd(2.0)])
    assert lo < FAMILY.at(0.10).cd(2.0) < hi


@needs_family
def test_family_cd_rises_with_thickness():
    # NACA 24XX at Re 1e6: profile drag near cruise alpha grows with t/c
    cds = [FAMILY.at(tc).cd(2.0) for tc in (0.08, 0.10, 0.12, 0.14, 0.16)]
    assert np.all(np.diff(cds) > 0)


@needs_family
def test_family_alpha_valid_is_intersection():
    lo, hi = FAMILY.at(0.08).alpha_valid  # blends 2406 (narrow) and 2409
    lo06, hi06 = FAMILY.members[0.06].alpha_valid
    assert lo == pytest.approx(max(lo06, FAMILY.members[0.09].alpha_valid[0]))
    assert hi == pytest.approx(min(hi06, FAMILY.members[0.09].alpha_valid[1]))


@needs_family
def test_family_no_extrapolation():
    with pytest.raises(ValueError):
        FAMILY.at(0.05)
    with pytest.raises(ValueError):
        FAMILY.at(0.19)


# ---------------- sweep + t/c physics terms ----------------


def test_swept_section_slope_cosine_rule():
    a0 = 6.3
    assert float(swept_section_slope(a0, 0.0)) == a0          # exact at 0
    assert float(swept_section_slope(a0, 30.0)) == pytest.approx(
        a0 * np.cos(np.deg2rad(30.0))
    )


def test_form_factor_rises_with_tc_and_falls_with_sweep():
    ffs = [wing_form_factor(tc) for tc in (0.08, 0.10, 0.12, 0.14, 0.16)]
    assert np.all(np.diff(ffs) > 0)
    assert wing_form_factor(0.12, sweep_deg=30.0) < wing_form_factor(0.12)
    # sweep = 0, M = 0: exactly the thickness term (old behaviour preserved)
    tc = 0.12
    assert wing_form_factor(tc) == pytest.approx(1 + 0.6 / 0.30 * tc + 100 * tc**4)


def test_tier_a_plus_bounds_and_wing_roundtrip():
    b = bounds("tier_a_plus")
    assert b.shape == (5, 2)
    assert b[3].tolist() == [0.0, 30.0] and b[4].tolist() == [0.08, 0.16]
    w = wing_from_x(np.array([0.7, 1.0, -3.0, 12.0, 0.10]), mode="tier_a_plus")
    assert (w.sweep_deg, w.tc) == (12.0, 0.10)
    # default Wing keeps the Tier A values
    assert (Wing().sweep_deg, Wing().tc) == (0.0, 0.12)


# ---------------- objective: regression + contract ----------------


@needs_family
def test_tier_a_plus_reduces_exactly_to_tier_a():
    """sweep = 0, tc = 0.12 must reproduce the old 3-D trim objective EXACTLY."""
    p3, p5 = Problem(), Problem(mode="tier_a_plus")
    for x3 in ([0.5, 0.0, 0.0], [0.35, 2.0, -3.0], [1.0, 0.0, 0.0],
               [0.2, -4.0, 2.0]):
        o3 = evaluate(np.array(x3), p3)
        o5 = evaluate(np.array(x3 + [0.0, 0.12]), p5)
        assert o3["feasible"] and o5["feasible"]
        # bit-for-bit: cos(0) = 1.0 and the family returns the member polar
        assert o5["score"] == o3["score"]
        assert o5["CDi"] == o3["CDi"]
        assert o5["CDp"] == o3["CDp"]
        assert o5["alpha_deg"] == o3["alpha_deg"]


@needs_family
def test_tier_a_plus_physics_directions():
    prob = Problem(mode="tier_a_plus")
    base = evaluate(np.array([0.5, 0, 0, 0.0, 0.12]), prob)["LoD"]
    # sweep at M = 0 is pure loss: lower slope -> higher trim alpha -> more cd
    assert evaluate(np.array([0.5, 0, 0, 20.0, 0.12]), prob)["LoD"] < base
    # thinner NACA 24XX has less profile drag at this Re/CL
    assert evaluate(np.array([0.5, 0, 0, 0.0, 0.09]), prob)["LoD"] > base
    assert evaluate(np.array([0.5, 0, 0, 0.0, 0.16]), prob)["LoD"] < base


@needs_family
def test_tier_a_plus_penalty_contract():
    prob = Problem(mode="tier_a_plus")
    assert objective(np.array([0.5, 0, 0, -5.0, 0.12]), prob) == PENALTY
    assert objective(np.array([0.5, 0, 0, 0.0, 0.30]), prob) == PENALTY
    assert objective(np.array([0.5, 0, 0, 0.0, 0.12, 1.0]), prob) == PENALTY


@needs_family
def test_tier_a_plus_double_count_guard_breakdown():
    # wing CD0 must NOT appear on top of the integrated polar CDp
    out = evaluate(np.array([0.5, 0.0, 0.0, 10.0, 0.10]), Problem(mode="tier_a_plus"))
    assert out["feasible"], out["reason"]
    assert out["CD"] == pytest.approx(out["CDi"] + out["CDp"] + out["cd0_extra"])
    assert out["cd0_extra"] == 0.0


# ---------------- BO smoke run on the 5-D mode ----------------


@needs_family
def test_bo_smoke_5d_beats_untwisted_rectangle():
    prob = Problem(mode="tier_a_plus")
    baseline = objective(np.array([1.0, 0.0, 0.0, 0.0, 0.12]), prob)
    assert baseline > 0  # sanity: rectangle is feasible

    res = run_bo(lambda x: objective(x, prob), prob.bounds,
                 n_init=10, n_iter=20, seed=0)   # budget 30
    assert res.best_y > baseline
    out = evaluate(res.best_x, prob)
    assert out["feasible"], out["reason"]
