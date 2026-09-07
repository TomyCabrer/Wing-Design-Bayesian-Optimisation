"""Gates for the Tier B section-level airfoil BO problem (AirfoilProblem).

MOCKED-XFOIL tests (monkeypatched run_xfoil_polar, no subprocess): a
synthetic smooth polar

    cl = 0.11 (alpha + 2),   cd = 0.006 + 0.0004 (cl - 0.4)^2,   cm = -0.05

makes every interpolated quantity closed-form: at cl_design = 0.5,
alpha = 0.5/0.11 - 2, cd = 0.006 + 0.0004 * 0.1^2 = 0.006004, so
f = -0.006004 exactly (np.interp on an exactly-quadratic-in-cl cd is NOT
exact between nodes, but alpha = 2.5455 is bracketed by nodes 0.5 apart and
the quadratic's curvature contributes < 1e-9 — asserted with abs tol 1e-6).

Gates:
  1. coherent breakdown on the NACA 2412 anchor (f = -cd interp, g signs:
     anchor tc 0.1199 vs tc_min 0.10 -> g0 > 0; |cm| 0.05 vs cap 0.08 ->
     g1 = 0.375);
  2. constraint signs flip where they should (thin box corner -> g0 < 0;
     tighter cm cap -> g1 < 0) while the solver stays feasible — the
     evaluate-and-report contract constrained BO relies on;
  3. pre-stall monotone-branch interpolation ignores post-stall rows;
  4. PENALTY dict on skinny garbage / bounds violation / wrong shape,
     fg = (PENALTY, [G_FAIL, G_FAIL]) without ever calling XFOIL;
  5. geometry precheck rejects self-intersecting loops before XFOIL;
  6. unconverged / cl-starved polars -> PENALTY (unconvergeable section =
     failed design);
  7. the recorded design box: floors/caps hold, w0 interior, dim 8, and a
     64-point Sobol re-check of the >= 80% validity criterion the box was
     derived under (200-point derivation recorded in the class docstring).

REAL-XFOIL smoke (skipped without /opt/homebrew/bin/xfoil): the 2412
anchor end-to-end, cd(cl=0.5) in (0.004, 0.02), cache round-trip.
Measured 0.8 s fresh -> NOT marked slow (< 5 s rule).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aerobo import xfoil_run
from aerobo.airfoil import (
    ALPHA_SWEEP_DEG,
    G_FAIL,
    PENALTY,
    R_LE_MIN,
    W_LOWER_CAP,
    W_UPPER_FLOOR,
    AirfoilProblem,
    _longest_increasing_run,
    _surface_y,
    alpha_sweep,
    cst_anchor,
    cst_le_radius,
    cst_thickness,
    evaluate_airfoil,
    fg_airfoil,
)
from aerobo.xfoil_run import XfoilPolarResult

XFOIL_BIN = "/opt/homebrew/bin/xfoil"
needs_xfoil = pytest.mark.skipif(
    not Path(XFOIL_BIN).exists(), reason="local XFOIL binary not installed"
)


@pytest.fixture(scope="module")
def prob():
    return AirfoilProblem()


def _synthetic_polar(alphas, cl_fn=lambda a: 0.11 * (a + 2.0),
                     cd_fn=lambda cl: 0.006 + 0.0004 * (cl - 0.4) ** 2,
                     cm_fn=lambda cl: np.full_like(cl, -0.05)):
    a = np.sort(np.asarray(list(alphas), dtype=float))
    cl = cl_fn(a)
    return XfoilPolarResult(alpha_deg=a, cl=cl, cd=cd_fn(cl), cm=cm_fn(cl),
                            n_requested=len(a), from_cache=False)


@pytest.fixture()
def mock_xfoil(monkeypatch):
    """Default smooth mock; returns the call log for cache/arg assertions."""
    calls = []

    def fake(coords, re, mach, alphas, **kw):
        calls.append({"coords": np.asarray(coords), "re": re, "mach": mach,
                      "alphas": list(alphas), **kw})
        return _synthetic_polar(alphas)

    monkeypatch.setattr(xfoil_run, "run_xfoil_polar", fake)
    return calls


@pytest.fixture()
def xfoil_must_not_run(monkeypatch):
    def boom(*a, **k):  # pragma: no cover - failing is the assertion
        raise AssertionError("run_xfoil_polar called for an invalid design")

    monkeypatch.setattr(xfoil_run, "run_xfoil_polar", boom)


# --------------------------------------------------- 1. coherent breakdown

def test_anchor_breakdown_mocked(prob, mock_xfoil):
    out = evaluate_airfoil(prob.w0, prob)
    assert out["feasible"] and out["reason"] == ""
    # closed-form synthetic values at cl_design = 0.5
    assert out["cd"] == pytest.approx(0.006004, abs=1e-6)
    assert out["score"] == pytest.approx(-out["cd"], rel=1e-14)  # f = -cd
    assert out["f"] == out["score"]
    assert out["cm"] == pytest.approx(-0.05, abs=1e-12)
    assert out["alpha_deg"] == pytest.approx(0.5 / 0.11 - 2.0, abs=1e-6)
    # anchor geometry: t/c = 0.1199 (recorded), computed WITHOUT XFOIL
    assert out["tc"] == pytest.approx(0.1199, abs=1e-3)
    # g signs: 2412 satisfies both constraints
    assert out["g"].shape == (2,)
    assert out["g_tc"] == pytest.approx((out["tc"] - 0.10) / 0.10, rel=1e-12)
    assert out["g_cm"] == pytest.approx((0.08 - 0.05) / 0.08, abs=1e-9)
    assert out["g_tc"] > 0 and out["g_cm"] > 0
    # whole synthetic sweep is monotone: branch = all requested alphas
    assert out["n_branch"] == out["n_converged"] == 29
    # one polar per evaluation, run at the problem's conditions
    assert len(mock_xfoil) == 1
    assert mock_xfoil[0]["re"] == prob.re
    np.testing.assert_allclose(mock_xfoil[0]["alphas"], prob.alphas)


def test_fg_matches_evaluate(prob, mock_xfoil):
    f, g = fg_airfoil(prob.w0, prob)
    out = evaluate_airfoil(prob.w0, prob)
    assert f == pytest.approx(out["score"], rel=1e-14)
    assert isinstance(g, np.ndarray) and g.shape == (2,)
    np.testing.assert_allclose(g, out["g"], rtol=1e-14)


# --------------------------------------------------- 2. constraint signs

def test_thin_corner_violates_tc_but_solves(prob, mock_xfoil):
    """Thinnest box corner: solver fine (true f reported), g0 < 0."""
    b = prob.bounds
    x = np.concatenate([b[:4, 0], b[4:, 1]])     # upper at floor, lower at cap
    tc = cst_thickness(x[:4], x[4:])
    assert tc < prob.tc_min                       # genuinely thin corner
    out = evaluate_airfoil(x, prob)
    assert out["feasible"]                        # solver ok...
    assert out["g_tc"] < 0.0                      # ...but structurally infeasible
    f, g = fg_airfoil(x, prob)
    assert f > PENALTY and g[0] < 0.0             # true value + signed violation


def test_cm_cap_flips_g1(prob, mock_xfoil):
    tight = AirfoilProblem(cm_max=0.04)           # below the mocked |cm| = 0.05
    out = evaluate_airfoil(tight.w0, tight)
    assert out["feasible"]
    assert out["g_cm"] == pytest.approx((0.04 - 0.05) / 0.04, abs=1e-9)
    assert out["g_cm"] < 0.0


# --------------------------------------- 3. pre-stall monotone branch

def test_stall_rows_do_not_corrupt_interp(prob, monkeypatch):
    """cl folds over past alpha = 6: interpolation must use the pre-stall
    increasing branch only, reproducing the clean-polar cd exactly."""

    def fake(coords, re, mach, alphas, **kw):
        a = np.sort(np.asarray(list(alphas), dtype=float))
        cl = np.where(a <= 6.0, 0.11 * (a + 2.0), 0.88 - 0.05 * (a - 6.0))
        cd = np.where(a <= 6.0, 0.006 + 0.0004 * (cl - 0.4) ** 2, 0.05)
        return XfoilPolarResult(alpha_deg=a, cl=cl, cd=cd,
                                cm=np.full_like(a, -0.05),
                                n_requested=len(a))

    monkeypatch.setattr(xfoil_run, "run_xfoil_polar", fake)
    out = evaluate_airfoil(prob.w0, prob)
    assert out["feasible"]
    assert out["n_branch"] < out["n_converged"]   # stall rows dropped
    assert out["cd"] == pytest.approx(0.006004, abs=1e-6)


def test_longest_increasing_run():
    assert _longest_increasing_run(np.array([])) == slice(0, 0)
    assert _longest_increasing_run(np.array([1.0])) == slice(0, 1)
    v = np.array([0.0, 1.0, 2.0, 1.5, 1.6, 1.7, 1.8])   # 4-long tail run
    assert _longest_increasing_run(v) == slice(3, 7)
    v = np.array([0.0, 1.0, 2.0, 3.0, 2.5, 2.6])        # 4-long head run
    assert _longest_increasing_run(v) == slice(0, 4)
    assert _longest_increasing_run(np.array([2.0, 2.0, 2.0])) == slice(0, 1)


# ------------------------------------------------ 4./5. failure contract

def test_penalty_skinny_garbage_no_xfoil(prob, xfoil_must_not_run):
    """All-near-zero weights: outside the box (floor/cap) -> PENALTY dict,
    and XFOIL is never launched for it."""
    out = evaluate_airfoil(np.full(8, 1e-3), prob)
    assert not out["feasible"] and out["score"] == PENALTY
    f, g = fg_airfoil(np.zeros(8), prob)
    assert f == PENALTY
    np.testing.assert_array_equal(g, [G_FAIL, G_FAIL])


def test_fg_contract_wrong_shape(prob, xfoil_must_not_run):
    f, g = fg_airfoil(np.zeros(3), prob)
    assert f == PENALTY
    assert g.shape == (2,)
    np.testing.assert_array_equal(g, [G_FAIL, G_FAIL])


def test_geometry_precheck_rejects_crossing(prob, xfoil_must_not_run):
    """Self-intersecting geometry is impossible IN-box (floor/cap proof in
    the class docstring), so widen the cached bounds and feed a crossing
    pair: the precheck must catch it before any XFOIL launch."""
    wide = AirfoilProblem()
    wide._bounds = np.column_stack([np.full(8, -10.0), np.full(8, 10.0)])
    x = np.concatenate([np.full(4, 0.05), np.full(4, +0.30)])  # lower ABOVE upper
    out = evaluate_airfoil(x, wide)
    assert not out["feasible"] and out["score"] == PENALTY
    assert "geometry" in out["reason"]
    f, g = fg_airfoil(x, wide)
    assert f == PENALTY
    np.testing.assert_array_equal(g, [G_FAIL, G_FAIL])


# ------------------------------------------------ 6. unconvergeable section

def test_penalty_on_empty_polar(prob, monkeypatch):
    monkeypatch.setattr(
        xfoil_run, "run_xfoil_polar",
        lambda coords, re, mach, alphas, **kw:
            XfoilPolarResult(*(np.empty(0),) * 4, n_requested=len(list(alphas))))
    out = evaluate_airfoil(prob.w0, prob)
    assert not out["feasible"] and out["score"] == PENALTY
    f, g = fg_airfoil(prob.w0, prob)
    assert f == PENALTY
    np.testing.assert_array_equal(g, [G_FAIL, G_FAIL])


def test_penalty_when_cl_design_unreachable(prob, monkeypatch):
    """Section converges but never reaches cl = 0.5 -> failed design."""

    def fake(coords, re, mach, alphas, **kw):
        return _synthetic_polar(alphas, cl_fn=lambda a: 0.02 * (a + 2.0))

    monkeypatch.setattr(xfoil_run, "run_xfoil_polar", fake)
    out = evaluate_airfoil(prob.w0, prob)
    assert not out["feasible"] and out["score"] == PENALTY
    assert out["cl_max"] < prob.cl_design


# --------------------------------- 6b. the alpha sweep is a stated ceiling
#
# The sweep's top end caps the design lift the family can be ASKED for, and
# until session 51 it was unreachable from anywhere and the refusal blamed the
# section for it. These four hold the two halves of the fix: the default must
# not move (every cached polar's key hashes it) and a refusal must say which
# of the two ceilings it hit.

def test_default_sweep_is_the_published_one_bit_for_bit():
    """The invariant that protects every frozen study: same array, same key.

    Not a restatement of ``alpha_sweep`` — the literal here is the historical
    ``AirfoilProblem.alphas`` factory, and the second half asserts the thing
    that actually breaks if it drifts, which is the XFOIL CACHE KEY. A sweep
    that differs in the last bit re-runs the whole repo.
    """
    published = np.arange(-4.0, 10.5, 0.5)
    np.testing.assert_array_equal(alpha_sweep(), published)
    np.testing.assert_array_equal(AirfoilProblem().alphas, published)
    coords = AirfoilProblem().w0
    from aerobo.airfoil import cst_coords
    c = cst_coords(*np.split(coords, 2))
    assert (xfoil_run.cache_key(c, 1e6, 0.0, alpha_sweep(), n_panel=200)
            == xfoil_run.cache_key(c, 1e6, 0.0, published, n_panel=200))


@pytest.mark.parametrize("hi, last", [(10.0, 10.0), (10.3, 10.0),
                                      (12.25, 12.0), (18.0, 18.0)])
def test_a_stated_sweep_ceiling_is_never_exceeded(hi, last):
    """An off-grid ceiling must round DOWN, not up.

    The obvious ``np.arange(lo, hi + step, step)`` sweeps 10.3 deg to 10.5 —
    past a number the caller stated — and silently mints a different cache
    key from the one they asked for. The off-grid rows are the mutation
    guard; the on-grid ones alone would pass either implementation.
    """
    a = alpha_sweep(alpha_max_deg=hi)
    assert a[-1] == pytest.approx(last)
    assert a.max() <= hi + 1e-12
    assert a[0] == ALPHA_SWEEP_DEG[0]        # the unstated end stays published


def test_a_sweep_limited_refusal_says_so_and_widening_fixes_it(monkeypatch):
    """Lift still rising at the top of the sweep -> the SWEEP is the ceiling.

    The outcome, not the wording: the same design at the same Reynolds number
    is refused on the published sweep and FEASIBLE on a wider one. That is the
    whole defect — 0/14 feasible at cl_design 1.3 was the 10 deg ceiling, not
    the aerodynamics.
    """
    monkeypatch.setattr(
        xfoil_run, "run_xfoil_polar",
        lambda coords, re, mach, alphas, **kw:
            _synthetic_polar(alphas, cl_fn=lambda a: 0.11 * (a + 2.0)))

    narrow = AirfoilProblem(cl_design=1.5)               # branch tops at 1.32
    out = evaluate_airfoil(narrow.w0, narrow)
    assert not out["feasible"] and out["sweep_limited"] is True
    assert out["cl_branch_max"] < 1.5
    assert "airfoil_sweep_alpha_max_deg" in out["reason"]

    wide = AirfoilProblem(cl_design=1.5,
                          alphas=alpha_sweep(alpha_max_deg=20.0))
    assert evaluate_airfoil(wide.w0, wide)["feasible"] is True


def test_a_stalling_section_is_not_blamed_on_the_sweep(monkeypatch):
    """The discriminator: a branch that ENDS inside the sweep really stalled.

    Without this, ``sweep_limited = True`` hard-coded passes the test above.
    Here the lift curve peaks at 4 deg with six degrees of sweep to spare, so
    widening cannot help and the refusal must not advise it.
    """
    monkeypatch.setattr(
        xfoil_run, "run_xfoil_polar",
        lambda coords, re, mach, alphas, **kw:
            _synthetic_polar(alphas, cl_fn=lambda a: 1.0 - 0.01 * (a - 4.0)**2))

    prob = AirfoilProblem(cl_design=1.5)
    out = evaluate_airfoil(prob.w0, prob)
    assert not out["feasible"] and out["sweep_limited"] is False
    assert "stalls below it" in out["reason"]
    assert "airfoil_sweep_alpha_max_deg" not in out["reason"]
    # ...and widening genuinely does NOT rescue it, which is the claim
    wide = AirfoilProblem(cl_design=1.5,
                          alphas=alpha_sweep(alpha_max_deg=20.0))
    assert evaluate_airfoil(wide.w0, wide)["feasible"] is False


def test_sweep_flags_reach_the_problem_and_silence_leaves_it_published():
    from aerobo import api

    published = AirfoilProblem().alphas
    for flags in (None, {}, {"airfoil_re": 8.5e5}):
        p = AirfoilProblem(**api._airfoil_kwargs(flags))
        np.testing.assert_array_equal(p.alphas, published)

    p = AirfoilProblem(**api._airfoil_kwargs(
        {"airfoil_sweep_alpha_max_deg": 18.0}))
    assert p.alphas[-1] == pytest.approx(18.0)
    assert p.alphas[0] == published[0]

    api.check_flags("airfoil (section)", {"airfoil_sweep_alpha_max_deg": 18.0})
    with pytest.raises(KeyError):        # the near-miss is still refused
        api.check_flags("airfoil (section)", {"airfoil_sweep_alpha_maxx": 18.0})


# ------------------------------------------------ 7. the recorded design box

def test_box_shape_and_anchor(prob):
    assert prob.dim == 8
    b = prob.bounds
    assert b.shape == (8, 2)
    assert np.all(b[:, 0] < b[:, 1])
    # anchor strictly interior (it is the baseline the sweep must beat)
    assert np.all(prob.w0 > b[:, 0]) and np.all(prob.w0 < b[:, 1])
    # floors/caps: each surface stays on its own side of the chord line
    assert np.all(b[:4, 0] >= W_UPPER_FLOOR - 1e-15)
    assert np.all(b[4:, 1] <= W_LOWER_CAP + 1e-15)
    # anchor reproduces the 2412 thickness
    w_u, w_l = cst_anchor("2412", 4)
    assert cst_thickness(w_u, w_l) == pytest.approx(0.1199, abs=1e-3)


def test_box_sobol_validity(prob):
    """Re-check of the derivation criterion at 64 points (power of 2):
    >= 80% of Sobol samples non-self-intersecting with t/c in (0.06, 0.20).
    The recorded 200-point derivation measured 100%."""
    from scipy.stats import qmc

    b = prob.bounds
    X = qmc.scale(qmc.Sobol(d=8, scramble=True, seed=0).random(64),
                  b[:, 0], b[:, 1])
    psi = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, 201)))[1:-1]
    n_ok = 0
    for x in X:
        gap = _surface_y(psi, x[:4], 0.0) - _surface_y(psi, x[4:], 0.0)
        tc = cst_thickness(x[:4], x[4:])
        n_ok += bool(np.all(gap > 0.0) and 0.06 < tc < 0.20)
    assert n_ok / 64 >= 0.8


# ------------------------------------------ 8. LE-radius third constraint

def test_fg_m2_bit_for_bit_when_r_le_min_none(prob, mock_xfoil):
    """r_le_min=None (default) => the m = 2 problem, bit-for-bit: fg returns
    a length-2 margin array exactly equal to evaluate's g."""
    assert prob.r_le_min is None
    f, g = fg_airfoil(prob.w0, prob)
    out = evaluate_airfoil(prob.w0, prob)
    assert isinstance(g, np.ndarray) and g.shape == (2,)
    np.testing.assert_array_equal(g, out["g"])
    assert out["g"].shape == (2,)
    assert out["g_rle"] is None                 # no LE margin reported
    # solver/geometry failure still returns exactly the legacy (2,) sentinel
    f2, g2 = fg_airfoil(np.zeros(8), prob)
    assert f2 == PENALTY
    np.testing.assert_array_equal(g2, [G_FAIL, G_FAIL])


def test_third_constraint_shape_and_positive_at_anchor(mock_xfoil):
    """r_le_min=R_LE_MIN => m = 3; the 2412 anchor's nose (min r_LE ~ 2x the
    floor, le_radius_audit) clears the floor, so g2 > 0."""
    prob = AirfoilProblem(r_le_min=R_LE_MIN)
    out = evaluate_airfoil(prob.w0, prob)
    assert out["feasible"]
    assert out["g"].shape == (3,)
    w_u, w_l = prob.w0[:4], prob.w0[4:]
    r_le = min(cst_le_radius(w_u), cst_le_radius(w_l))
    assert out["r_le"] == pytest.approx(r_le, rel=1e-12)
    assert out["g_rle"] == pytest.approx((r_le - R_LE_MIN) / R_LE_MIN, rel=1e-12)
    assert out["g_rle"] > 0.0                   # 2412 nose is comfortably blunt
    # fg mirrors evaluate: length-3 margin vector, third entry the LE margin
    f, g = fg_airfoil(prob.w0, prob)
    assert g.shape == (3,)
    np.testing.assert_allclose(g, out["g"], rtol=1e-14)
    assert g[0] > 0 and g[1] > 0 and g[2] > 0   # 2412 feasible on all three


def test_third_constraint_negative_for_sharp_nose(mock_xfoil):
    """A blunt-upper / knife-edge-lower design (lower nose at the box cap
    W_LOWER_CAP -> r_LE ~ 2e-4, ~27x below R_LE_MIN) still SOLVES but its LE
    margin g2 < 0, while the other two margins stay whatever they are."""
    prob = AirfoilProblem(r_le_min=R_LE_MIN)
    w_u, w_l = cst_anchor("2412", 4)
    x = np.concatenate([w_u, [W_LOWER_CAP, w_l[1], w_l[2], w_l[3]]])
    r_lower = cst_le_radius(np.array([W_LOWER_CAP]))
    assert r_lower < R_LE_MIN                    # genuinely sub-floor nose
    out = evaluate_airfoil(x, prob)
    assert out["feasible"]                       # solver fine...
    assert out["g"].shape == (3,)
    assert out["r_le"] == pytest.approx(r_lower, rel=1e-12)
    assert out["g_rle"] < 0.0                    # ...but structurally too sharp
    f, g = fg_airfoil(x, prob)
    assert f > PENALTY and g[2] < 0.0            # true value + signed violation


def test_fg_failure_sizes_margin_to_m3(mock_xfoil):
    """With the LE floor active, a failed design returns [G_FAIL]*3 so the
    pymoo GA's 3-wide constraint block is fed the right width."""
    prob = AirfoilProblem(r_le_min=R_LE_MIN)
    f, g = fg_airfoil(np.zeros(8), prob)         # all-zero -> out of box
    assert f == PENALTY
    assert g.shape == (3,)
    np.testing.assert_array_equal(g, [G_FAIL, G_FAIL, G_FAIL])


# ------------------------------------------------ real-XFOIL smoke

@needs_xfoil
def test_real_anchor_end_to_end(tmp_path):
    """2412 anchor through the real binary: sane cd, feasible constraints,
    second call served from the polar cache. Measured 0.8 s -> not slow."""
    prob = AirfoilProblem(cache_dir=tmp_path)
    out = evaluate_airfoil(prob.w0, prob)
    assert out["feasible"], out["reason"]
    assert 0.004 < out["cd"] < 0.02
    assert out["n_converged"] >= 3
    assert out["g_tc"] > 0 and out["g_cm"] > 0    # the 2412 is feasible
    assert not out["from_cache"]

    out2 = evaluate_airfoil(prob.w0, prob)        # identical -> cache hit
    assert out2["from_cache"]
    assert out2["cd"] == out["cd"] and out2["score"] == out["score"]

    f, g = fg_airfoil(prob.w0, prob)
    assert f == pytest.approx(-out["cd"], rel=1e-14)
    assert np.all(g >= 0.0)
