"""Gates for the CST (Kulfan) parametrisation + NACA 4-digit generator.

1. Round-trip: fit_cst on naca4_coords -> cst_coords reproduces the section
   to 0.1 % chord, and cst_thickness recovers t/c = 0.12 to 1e-3.

   Metric note (physics, not a loophole): the exact Abbott & von Doenhoff
   construction applies thickness PERPENDICULAR to the camber line, so a
   cambered nose overhangs upstream of x = 0 (NACA 2412: to x ~ -8e-5) and
   the upper surface crosses x = 0 at y ~ +3e-3. A vertical-ordinate
   comparison |Dy| at fixed x therefore DIVERGES with surface slope at the
   nose (|dy/dx| -> inf as x -> 0) even though the two shapes are ~5e-4
   apart geometrically. CST pins y(0) = 0 by construction, so no weight
   vector can zero that artifact. Hence the cambered 2412 gate is split:
   * ordinate gate |y_fit - y_naca| < 1e-3 on common x in [0.005, 1]
     (99.5 % of the chord, where y(x) is a well-posed comparison), plus
   * a STRICTER full-loop geometric gate: every fitted point within 1e-3
     of the dense exact-NACA polyline, nose included (measured ~5e-4).
   The symmetric 0012 (no camber -> no abscissa offset, y(x) well-posed
   everywhere) gets the verbatim full-domain [0, 1] ordinate gate.

2. Bernstein partition of unity: sum_i B_i(psi) = 1 for random psi — the
   property that keeps CST weights well-scaled as a BO design vector.

3. Loop order (frozen inter-agent contract, XFOIL LOAD format):
   TE-upper (1, ~0) -> LE (0, 0) mid-array -> TE-lower (1, ~-0 or -dz/2),
   n_pts total, LE exactly once, no duplicate points.

4. TE gap: cst_coords TE opening equals dz_te exactly (linear zeta_T term).

5. Degenerate all-zero weights: zero thickness, finite coords, no crash
   (the -100 failure contract upstream needs geometry generation itself to
   never raise on valid inputs).

6. Leading-edge radius, cst_le_radius(w) = w_0^2 / 2 (N1 = 0.5 class):
   * the nose limit itself — y^2 / (2 psi) -> r as psi -> 0 is the
     osculating circle of a curve with a vertical tangent at the origin,
     and for a CST surface it equals ((1-psi) S(psi))^2 / 2 exactly, so
     the identity is checked to O(psi) at psi = 1e-8;
   * the analytic cross-check against NACA 4-digit, r_LE/c = 1.1019 t^2
     [Abbott & von Doenhoff 1959, §6]. A least-squares CST fit does NOT
     reproduce it exactly: the fit spends its 4 weights on the whole
     surface, not on the nose, and lands 9 % low (ratio 0.90920 at
     n_cst = 4, rising monotonically to 0.96869 at n_cst = 16 as the
     basis resolves the nose). The ratio is INDEPENDENT of t to 1e-9 (the
     NACA thickness form is exactly linear in t, so the weights are too),
     which is the sharp statement that the 9 % is fit truncation and not
     a wrong formula. Cambered sections are covered by the geometric mean
     sqrt(r_u r_l), which recovers the same ratio to 0.5 %;
   * monotonicity/scaling: r strictly increases with w_0, quadratically,
     sign-blind, and is untouched by w_1.. (only B_0 survives at psi = 0);
   * R_LE_MIN derivation, and the fact that the recorded design box
     admits noses far below it (W_LOWER_CAP = -0.02 -> r = 2e-4).
"""

import numpy as np
import pytest

from aerobo.airfoil import (
    N1,
    N2,
    R_LE_MIN,
    W_LOWER_CAP,
    W_UPPER_FLOOR,
    _surface_y,
    bernstein_matrix,
    cst_anchor,
    cst_coords,
    cst_le_radius,
    cst_thickness,
    fit_cst,
    naca4_coords,
    norm_margin,
)

# ------------------------------------------------------------------ helpers


def split_surfaces(coords):
    """Closed loop (TE-up -> LE -> TE-lo) -> (xu, yu, xl, yl), x ascending.

    Split at the min-x point; sort each branch by x (the exact cambered
    NACA nose is a hair non-monotonic in x, see module docstring)."""
    x, y = coords[:, 0], coords[:, 1]
    i_le = int(np.argmin(x))
    xu, yu = x[: i_le + 1][::-1], y[: i_le + 1][::-1]
    xl, yl = x[i_le:], y[i_le:]
    su, sl = np.argsort(xu), np.argsort(xl)
    return xu[su], yu[su], xl[sl], yl[sl]


def surfaces_on(coords, x_common):
    """Upper/lower ordinates of a loop interpolated onto a common x grid."""
    xu, yu, xl, yl = split_surfaces(coords)
    return np.interp(x_common, xu, yu), np.interp(x_common, xl, yl)


def max_dist_to_polyline(pts, poly):
    """Max over pts of the min point-to-segment distance to polyline poly."""
    a, b = poly[:-1], poly[1:]
    ab = b - a
    len2 = np.maximum((ab**2).sum(axis=1), 1e-300)
    worst = 0.0
    for p in pts:
        t = np.clip(((p - a) * ab).sum(axis=1) / len2, 0.0, 1.0)
        proj = a + t[:, None] * ab
        worst = max(worst, np.sqrt(((p - proj) ** 2).sum(axis=1).min()))
    return worst


def cosine_grid(n, lo=0.0, hi=1.0):
    return lo + (hi - lo) * 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, n)))


def roundtrip(code, n_cst, x_fit):
    """fit_cst on a dense naca4 loop -> (w_u, w_l, dz, fitted loop, ref loop)."""
    ref = naca4_coords(code, n_pts=4000)
    yu, yl = surfaces_on(ref, x_fit)
    w_u, w_l, dz = fit_cst(x_fit, yu, yl, n_cst=n_cst)
    fit = cst_coords(w_u, w_l, n_pts=800, dz_te=dz)
    return w_u, w_l, dz, fit, ref


# ------------------------------------------------- 1. round-trip fidelity


@pytest.mark.parametrize("n_cst", [4, 6])
def test_roundtrip_naca2412(n_cst):
    """Cambered flagship: <1e-3 ordinates on [0.005, 1] + <1e-3 geometric
    everywhere (nose included) + thickness to 1e-3."""
    x_fit = cosine_grid(201)
    w_u, w_l, dz, fit, ref = roundtrip("2412", n_cst, x_fit)

    x_cmp = cosine_grid(201, lo=0.005)
    yu_ref, yl_ref = surfaces_on(ref, x_cmp)
    yu_fit, yl_fit = surfaces_on(fit, x_cmp)
    assert np.max(np.abs(yu_fit - yu_ref)) < 1e-3
    assert np.max(np.abs(yl_fit - yl_ref)) < 1e-3

    # nose covered by the geometric bound — stricter than the ordinate one
    assert max_dist_to_polyline(fit, ref) < 1e-3

    assert abs(cst_thickness(w_u, w_l, dz_te=dz) - 0.12) < 1e-3
    # conventional section in the unnegated Kulfan convention
    assert np.all(w_u > 0) and np.all(w_l < 0)
    assert dz == pytest.approx(0.0, abs=1e-12)   # -0.1036 closes the TE


@pytest.mark.parametrize("n_cst", [4, 6])
def test_roundtrip_naca0012_full_domain(n_cst):
    """Symmetric section: verbatim spec gate, common x over ALL of [0, 1]."""
    x_fit = cosine_grid(201)
    w_u, w_l, dz, fit, ref = roundtrip("0012", n_cst, x_fit)

    yu_ref, yl_ref = surfaces_on(ref, x_fit)
    yu_fit, yl_fit = surfaces_on(fit, x_fit)
    assert np.max(np.abs(yu_fit - yu_ref)) < 1e-3
    assert np.max(np.abs(yl_fit - yl_ref)) < 1e-3
    assert abs(cst_thickness(w_u, w_l, dz_te=dz) - 0.12) < 1e-3
    # symmetry survives the two independent lstsq solves (atol reflects the
    # upper/lower loop branches sitting on slightly different interp grids)
    np.testing.assert_allclose(w_u, -w_l, atol=1e-5)


# ------------------------------------------------- 2. Bernstein basis


def test_bernstein_partition_of_unity():
    rng = np.random.default_rng(7)
    psi = rng.uniform(0.0, 1.0, size=200)
    for n_w in (1, 2, 5, 9):
        np.testing.assert_allclose(
            bernstein_matrix(psi, n_w).sum(axis=1), 1.0, atol=1e-12
        )


def test_class_function_exponents():
    """Frozen contract: N1 = 0.5 (round nose), N2 = 1.0 (sharp TE)."""
    assert N1 == 0.5 and N2 == 1.0


# ------------------------------------------------- 3. loop order / XFOIL format


@pytest.mark.parametrize(
    "coords",
    [
        cst_coords(np.array([0.2, 0.21, 0.2, 0.22]),
                   np.array([-0.15, -0.08, -0.09, -0.07]), n_pts=160),
        naca4_coords("2412", n_pts=160),
        naca4_coords("0012", n_pts=161),   # odd n_pts too
    ],
    ids=["cst", "naca2412", "naca0012_odd"],
)
def test_loop_order(coords):
    n_pts = coords.shape[0]
    assert coords.shape[1] == 2
    # starts at upper TE (1, ~0)
    assert coords[0, 0] == pytest.approx(1.0, abs=1e-12)
    assert abs(coords[0, 1]) < 5e-3
    # closes at lower TE (1, ~-0)
    assert coords[-1, 0] == pytest.approx(1.0, abs=1e-12)
    assert abs(coords[-1, 1]) < 5e-3
    assert coords[-1, 1] <= coords[0, 1] + 1e-12     # lower below upper at TE
    # LE (0, 0) appears once, mid-array
    i_le = int(np.argmin(coords[:, 0]))
    assert abs(i_le - n_pts / 2) <= 1.0
    assert np.hypot(*coords[i_le]) < 5e-4
    # upper half above lower half on average (right way round for XFOIL)
    assert coords[: i_le, 1].mean() > coords[i_le + 1:, 1].mean()
    # no duplicate consecutive points (XFOIL panelling chokes on those)
    assert np.min(np.hypot(*np.diff(coords, axis=0).T)) > 1e-9
    # cosine clustering: LE panels much finer than mid-chord panels
    dx = np.abs(np.diff(coords[:, 0]))
    assert dx[i_le - 1] < 0.05 * dx.max()


def test_le_point_exact_for_cst():
    coords = cst_coords(np.full(4, 0.2), np.full(4, -0.2), n_pts=160,
                        dz_te=0.002)
    i_le = int(np.argmin(coords[:, 0]))
    assert coords[i_le, 0] == 0.0 and coords[i_le, 1] == 0.0


# ------------------------------------------------- 4. TE gap


def test_te_gap_equals_dz_te():
    dz = 0.004
    coords = cst_coords(np.full(5, 0.25), np.full(5, -0.2), n_pts=200,
                        dz_te=dz)
    gap = coords[0, 1] - coords[-1, 1]
    assert gap == pytest.approx(dz, abs=1e-14)
    # split symmetrically about the chord line: +dz/2 up, -dz/2 down
    assert coords[0, 1] == pytest.approx(+dz / 2, abs=1e-14)
    assert coords[-1, 1] == pytest.approx(-dz / 2, abs=1e-14)


def test_naca_closed_te():
    """-0.1036 coefficient: TE gap identically zero (loop truly closes)."""
    for code in ("2412", "0012", "4415"):
        coords = naca4_coords(code)
        assert abs(coords[0, 1] - coords[-1, 1]) < 1e-12


# ------------------------------------------------- 5. degenerate inputs


def test_degenerate_zero_weights():
    w0 = np.zeros(4)
    coords = cst_coords(w0, w0)
    assert np.all(np.isfinite(coords))
    assert np.max(np.abs(coords[:, 1])) == 0.0
    assert cst_thickness(w0, w0) == 0.0


def test_fit_cst_input_validation():
    x = cosine_grid(50)
    y = np.zeros(50)
    with pytest.raises(ValueError):
        fit_cst(x, y, np.zeros(49))          # shape mismatch
    with pytest.raises(ValueError):
        fit_cst(x + 0.5, y, y)               # x outside [0, 1]


def test_naca4_bad_code():
    for bad in ("241", "24a2", "24122"):
        with pytest.raises(ValueError):
            naca4_coords(bad)


# ------------------------------------------------- 6. leading-edge radius

NACA_R_LE = 1.1019          # r_LE/c = 1.1019 t^2 [Abbott & von Doenhoff §6]


def test_le_radius_is_the_nose_limit_of_the_surface():
    """r = lim_{psi->0} y^2 / (2 psi) — the osculating circle at a
    vertical-tangent nose — equals w_0^2 / 2 for any weight vector."""
    rng = np.random.default_rng(3)
    for _ in range(5):
        w = rng.uniform(0.05, 0.4, size=5)
        for sign in (+1.0, -1.0):          # lower surface: same radius
            psi = np.array([1e-8])
            y = _surface_y(psi, sign * w, 0.0)[0]
            r_limit = y**2 / (2.0 * psi[0])
            assert r_limit == pytest.approx(cst_le_radius(sign * w), rel=1e-6)


def test_le_radius_vs_analytic_naca_symmetric():
    """Symmetric NACA 00XX: the fitted radius tracks 1.1019 t^2 with a
    THICKNESS-INDEPENDENT truncation ratio (the fit is linear in t)."""
    ratios = []
    for code in ("0009", "0012", "0015"):
        t = int(code[2:]) / 100.0
        w_u, w_l = cst_anchor(code, n_cst=4)
        r_u, r_l = cst_le_radius(w_u), cst_le_radius(w_l)
        assert r_u == pytest.approx(r_l, rel=1e-5)      # symmetric section
        ratios.append(r_u / (NACA_R_LE * t**2))
    assert ratios[0] == pytest.approx(ratios[1], rel=1e-9)
    assert ratios[1] == pytest.approx(ratios[2], rel=1e-9)
    assert ratios[0] == pytest.approx(0.9092, abs=1e-3)


def test_le_radius_converges_to_analytic_with_basis_order():
    """The 9 % gap is fit truncation: it shrinks monotonically as the
    Bernstein basis gets enough weights to resolve the nose."""
    exact = NACA_R_LE * 0.12**2
    ratios = [cst_le_radius(cst_anchor("0012", n_cst=n)[0]) / exact
              for n in (4, 8, 16)]
    assert ratios[0] < ratios[1] < ratios[2] < 1.0
    assert ratios[2] > 0.96


def test_le_radius_cambered_geometric_mean():
    """NACA 2412: camber splits the nose into two radii (upper blunter,
    lower sharper); their geometric mean is the thickness-form radius."""
    w_u, w_l = cst_anchor("2412", n_cst=4)
    r_u, r_l = cst_le_radius(w_u), cst_le_radius(w_l)
    assert r_u > r_l > 0.0
    exact = NACA_R_LE * 0.12**2
    assert np.sqrt(r_u * r_l) / exact == pytest.approx(0.9092, abs=5e-3)


def test_le_radius_monotone_and_quadratic_in_w0():
    """Bigger w_0 -> blunter nose, r ~ w_0^2, sign-blind, and w_1.. do not
    touch it (only B_0 survives at psi = 0)."""
    w0 = np.linspace(0.02, 0.6, 40)
    r = np.array([cst_le_radius(np.r_[v, 0.2, 0.2, 0.2]) for v in w0])
    assert np.all(np.diff(r) > 0.0)
    base = np.array([0.15, 0.2, 0.18, 0.21])
    assert cst_le_radius(2.0 * base[0:1]) == pytest.approx(
        4.0 * cst_le_radius(base), rel=1e-12)
    assert cst_le_radius(-base) == pytest.approx(cst_le_radius(base))
    tweaked = base.copy()
    tweaked[1:] = [-0.5, 0.9, 0.0]
    assert cst_le_radius(tweaked) == pytest.approx(cst_le_radius(base))
    with pytest.raises(ValueError):
        cst_le_radius(np.array([]))


def test_r_le_min_derivation_and_box_reach():
    """R_LE_MIN = 0.5 * 1.1019 * tc_min^2 at tc_min = 0.10, and the
    recorded design box reaches noses an order of magnitude below it."""
    assert R_LE_MIN == pytest.approx(0.5 * NACA_R_LE * 0.10**2, rel=1e-12)
    assert R_LE_MIN == pytest.approx(0.00551, abs=5e-6)
    # sharpest nose the floor/cap admit, per surface
    assert cst_le_radius(np.array([W_UPPER_FLOOR])) == pytest.approx(1.25e-3)
    assert cst_le_radius(np.array([W_LOWER_CAP])) == pytest.approx(2.0e-4)
    assert cst_le_radius(np.array([W_LOWER_CAP])) < 0.05 * R_LE_MIN
    # the anchor itself is comfortably blunt
    w_u, w_l = cst_anchor("2412", n_cst=4)
    assert min(cst_le_radius(w_u), cst_le_radius(w_l)) > 2.0 * R_LE_MIN


def test_norm_margin_is_the_old_expression_and_survives_a_zero_limit():
    """A gate of ZERO is a setting ("no floor" / "no moment at all"), not a
    ZeroDivisionError — the V3 shell sends tc_min = 0 for a gate switched
    off. For every positive finite limit the value is the old division,
    bit-for-bit."""
    for slack, lim in [(0.02, 0.10), (-0.031, 0.08), (1.0, 3.0)]:
        assert norm_margin(slack, lim) == slack / lim
    # zero limit -> the RAW slack: same sign, same feasible set
    assert norm_margin(0.12, 0.0) == 0.12          # t/c 0.12 vs no floor
    assert norm_margin(-0.046, 0.0) == -0.046      # |cm| 0.046 vs a 0 cap
    assert norm_margin(0.12, -1.0) == 0.12         # a negative limit likewise
    # an "off" upper cap divides normally and reads as a margin of ~1
    assert norm_margin(1e9 - 0.05, 1e9) == pytest.approx(1.0, abs=1e-9)
    assert norm_margin(0.5, float("inf")) == 0.5


def test_zero_gates_evaluate_instead_of_raising(monkeypatch):
    """evaluate_airfoil at tc_min = 0 / cm_max = 0 returns finite margins of
    the right sign (it used to raise ZeroDivisionError), and the standard
    0.10 / 0.08 gates are unchanged."""
    from aerobo import airfoil

    w_u, w_l = cst_anchor("2412", n_cst=4)
    x = np.concatenate([w_u, w_l])

    def run(**kw):
        return airfoil.evaluate_airfoil(x, airfoil.AirfoilProblem(**kw))

    base = run(tc_min=0.10, cm_max=0.08)
    assert base["feasible"] and base["g_tc"] > 0 and base["g_cm"] > 0

    off = run(tc_min=0.0, cm_max=1.0e9)              # both gates switched off
    assert off["g_tc"] == pytest.approx(off["tc"])   # raw thickness slack
    assert off["g_cm"] == pytest.approx(1.0, abs=1e-6)
    assert off["score"] == pytest.approx(base["score"])   # same aerodynamics

    hard = run(tc_min=0.10, cm_max=0.0)              # "no pitching moment"
    assert hard["g_cm"] == pytest.approx(-abs(hard["cm"]))
    assert hard["g_cm"] < 0                          # correctly infeasible
