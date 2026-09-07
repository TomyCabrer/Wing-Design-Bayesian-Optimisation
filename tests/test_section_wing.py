"""Section + twist co-design for a guessed wing (src/aerobo/section_wing.py).

The physics contract of the section-ONLY optimiser's wing mode: the area
GUESS derives the design lift coefficient and the section Reynolds number,
the twist law is a root-anchored polynomial under two angle caps, and the
score is the wing L/D of the trimmed lifting line with profile drag taken
from the section's own viscous polar.

The XFOIL-backed tests are skipped without a local binary; everything that
can be checked without one (derivations, twist algebra, the box, the failure
contract) runs unconditionally.
"""

from pathlib import Path

import numpy as np
import pytest

from aerobo import airfoil
from aerobo import section_wing as sw

_XFOIL_BIN = "/opt/homebrew/bin/xfoil"
needs_xfoil = pytest.mark.skipif(
    not Path(_XFOIL_BIN).exists(), reason="local XFOIL binary not installed")


# ------------------------------------------------------- 1. the area guess


def test_wing_guess_derives_the_design_point_from_the_area():
    """CL = W/(qS) and Re = rho V mac / mu, both from the guess alone."""
    g = sw.WingGuess(mass_kg=80.0, v_ms=25.0, altitude_m=0.0,
                     s_ref_m2=8.0, aspect_ratio=9.0, taper=0.5)
    q = 0.5 * g.rho * 25.0 ** 2
    assert g.weight_n == pytest.approx(80.0 * sw.G0)
    assert g.q == pytest.approx(q)
    assert g.cl_design == pytest.approx(80.0 * sw.G0 / (q * 8.0))
    # planform closed forms
    assert g.b == pytest.approx(np.sqrt(9.0 * 8.0))
    assert g.c_root == pytest.approx(2.0 * 8.0 / (g.b * 1.5))
    assert g.mac == pytest.approx((2.0 / 3.0) * g.c_root * 1.75 / 1.5)
    assert g.re_mac == pytest.approx(g.rho * 25.0 * g.mac / g.mu)
    # the whole point of the mode: a SMALLER area guess demands MORE lift
    small = sw.WingGuess(mass_kg=80.0, v_ms=25.0, s_ref_m2=4.0)
    assert small.cl_design > g.cl_design


def test_wing_guess_stations_and_area_are_consistent():
    g = sw.WingGuess(s_ref_m2=6.0, aspect_ratio=10.0, taper=0.6)
    y, c, eta = g.stations()
    assert y.shape == c.shape == eta.shape == (g.n_stations,)
    assert eta.min() >= 0.0 and eta.max() <= 1.0
    # trapezoid over the cosine grid under-reads the exact area by O(1/N^2);
    # the default station count keeps that below 0.2%
    assert np.trapezoid(c, y) == pytest.approx(6.0, rel=2e-3)
    assert g.to_dict()["cl_design"] == pytest.approx(g.cl_design)


@pytest.mark.parametrize("kw", [{"mass_kg": 0.0}, {"s_ref_m2": -1.0},
                                {"taper": 0.0}, {"taper": 1.5},
                                {"aspect_ratio": 0.0}, {"n_stations": 4}])
def test_wing_guess_rejects_unphysical_input(kw):
    with pytest.raises(ValueError):
        sw.WingGuess(**kw)


# ------------------------------------------------------- 2. the twist law


def test_twist_law_is_root_anchored_and_linear_tip_is_the_coefficient():
    assert sw.twist_deg(np.array([0.0]), [-4.0])[0] == pytest.approx(0.0)
    assert sw.twist_deg(np.array([1.0]), [-4.0])[0] == pytest.approx(-4.0)
    # order 1 envelope IS |t1|, so the box alone already respects the cap
    assert sw.twist_envelope_deg([-4.0]) == pytest.approx(4.0)
    assert sw.twist_envelope_deg([]) == 0.0


def test_twist_envelope_catches_an_inboard_extremum():
    """A high-order law can peak INBOARD; bounding the tip would miss it."""
    coeffs = [12.0, -12.0]                 # theta = 12 eta (1 - eta)
    assert sw.twist_deg(np.array([1.0]), coeffs)[0] == pytest.approx(0.0)
    assert sw.twist_envelope_deg(coeffs) == pytest.approx(3.0, abs=1e-3)


def test_twist_bounds_box_and_validation():
    b = sw.twist_bounds(3, 6.0)
    assert b.shape == (3, 2)
    assert np.allclose(b[:, 0], -6.0) and np.allclose(b[:, 1], 6.0)
    with pytest.raises(ValueError):
        sw.twist_bounds(9, 6.0)
    with pytest.raises(ValueError):
        sw.twist_bounds(1, 0.0)


def test_section_line_fit_recovers_a_known_line():
    a, a_l0 = 6.1, np.deg2rad(-2.5)
    alpha = np.arange(-4.0, 10.5, 0.5)
    cl = a * (np.deg2rad(alpha) - a_l0)
    got_a, got_l0 = sw.section_line_fit(cl, alpha, cl_design=0.5)
    assert got_a == pytest.approx(a, rel=1e-9)
    assert got_l0 == pytest.approx(a_l0, abs=1e-9)


def test_section_line_fit_rejects_a_useless_polar():
    alpha = np.arange(0.0, 5.0, 1.0)
    with pytest.raises(ValueError):
        sw.section_line_fit(-0.1 * alpha, alpha, cl_design=0.0)


# --------------------------------------------------------- 3. the problem


def test_problem_box_grows_by_the_twist_order_and_derives_its_point():
    g = sw.WingGuess(mass_kg=70.0, v_ms=21.0, s_ref_m2=6.5)
    base = airfoil.AirfoilProblem(re=1.0, cl_design=99.0, tc_min=0.13)
    p = sw.SectionWingProblem(airfoil=base, wing=g, twist_order=3,
                              twist_max_deg=5.0)
    assert p.dim == 8 + 3
    assert p.param_labels[-3:] == ("twist_1_deg", "twist_2_deg",
                                   "twist_3_deg")
    assert p.bounds.shape == (11, 2)
    # the WING owns the operating point; the typed re/cl_design are discarded
    assert p.section.re == pytest.approx(g.re_mac)
    assert p.section.cl_design == pytest.approx(g.cl_design)
    assert p.section.tc_min == 0.13          # gates survive
    assert np.array_equal(p.section.alphas, sw.WING_ALPHAS)
    # section box is untouched; twist rows are the cap
    assert np.array_equal(p.bounds[:8], p.section.bounds)
    assert np.allclose(p.bounds[8:], [[-5.0, 5.0]] * 3)
    # baseline design = the anchor section, untwisted
    assert np.array_equal(p.x0[:8], p.section.w0)
    assert np.allclose(p.x0[8:], 0.0)
    assert p.n_constraints == 4
    assert p.constraint_labels == sw.CONSTRAINT_LABELS


def test_problem_rejects_bad_twist_settings():
    with pytest.raises(ValueError):
        sw.SectionWingProblem(twist_order=7)
    with pytest.raises(ValueError):
        sw.SectionWingProblem(alpha_max_deg=0.0)


def test_failure_contract_needs_no_xfoil():
    """Shape / bounds failures return the finite penalty pair, never raise."""
    p = sw.SectionWingProblem(twist_order=2)
    for bad in (np.zeros(3), p.bounds[:, 1] + 10.0):
        f, g = sw.fg_section_wing(np.asarray(bad, dtype=float), p)
        assert f == airfoil.PENALTY
        assert g.shape == (4,)
        assert np.all(g == airfoil.G_FAIL)


# ------------------------------------------------- 4. the real evaluation


@needs_xfoil
def test_untwisted_evaluation_matches_the_lifting_line_closed_forms():
    p = sw.SectionWingProblem(twist_order=1, twist_max_deg=6.0)
    out = sw.evaluate_section_wing(p.x0, p)
    assert out["feasible"], out.get("reason")
    # trimmed to the DERIVED design CL
    assert out["CL"] == pytest.approx(p.wing.cl_design, rel=1e-6)
    # CDi is the textbook CL^2/(pi AR e) of the loading the solver returned
    assert out["CDi"] == pytest.approx(
        out["CL"] ** 2 / (np.pi * out["AR"] * out["e"]), rel=1e-9)
    # profile drag is the strip integral of the section's own polar, so it
    # sits within a few percent of the section cd at the design lift
    assert out["CDp"] == pytest.approx(out["cd"], rel=0.25)
    assert out["LD"] == pytest.approx(out["CL"] / (out["CDi"] + out["CDp"]))
    assert out["LD"] > 10.0
    # untwisted => zero envelope => that margin is fully slack
    assert out["twist_env_deg"] == 0.0
    assert out["g_twist"] == pytest.approx(1.0)
    assert out["alpha_geo_max_deg"] == pytest.approx(
        abs(out["alpha_root_deg"]))
    # NACA 2412 anchor: alpha_L0 near -2 deg, slope near 2 pi
    assert -3.5 < out["alpha_L0_deg"] < -1.0
    assert 5.5 < out["a_per_rad"] < 7.5


@needs_xfoil
def test_twist_only_change_reuses_the_cached_polar():
    """Twist moves must not cost XFOIL — that is what makes the law cheap."""
    p = sw.SectionWingProblem(twist_order=1, twist_max_deg=6.0)
    sw.evaluate_section_wing(p.x0, p)                 # warm the polar
    x = p.x0.copy()
    x[-1] = -3.0
    out = sw.evaluate_section_wing(x, p)
    assert out["feasible"], out.get("reason")
    assert out["from_cache"] is True
    assert out["twist_tip_deg"] == pytest.approx(-3.0)
    assert out["twist_env_deg"] == pytest.approx(3.0)
    assert out["g_twist"] == pytest.approx(0.5)


@needs_xfoil
def test_angle_caps_are_reported_as_signed_margins_not_hidden():
    """An over-twisted design is EVALUATED and returns a negative margin."""
    p = sw.SectionWingProblem(twist_order=2, twist_max_deg=4.0,
                              alpha_max_deg=10.0)
    x = np.concatenate([p.section.w0, [-4.0, -4.0]])   # envelope 8 deg > cap
    out = sw.evaluate_section_wing(x, p)
    assert out["feasible"]                    # solver-feasible …
    assert out["twist_env_deg"] == pytest.approx(8.0)
    assert out["g_twist"] < 0.0               # … but design-infeasible
    assert out["LD"] > 0.0                    # true objective still reported
    f, g = sw.fg_section_wing(x, p)
    assert f == pytest.approx(out["LD"])
    assert g[2] < 0.0


# ------------------------------------------------- 5. the chord law


def test_chord_law_is_area_preserving_and_root_anchored():
    """Area must not move: it is what set CL_design in the first place."""
    wg = sw.WingGuess()
    y0, c0, eta0 = wg.stations()
    a0 = float(np.trapezoid(c0, y0))
    for coeffs in ([0.3], [-0.4], [0.5, -0.5], [-0.2, 0.4, -0.3]):
        y, c, eta = wg.stations(coeffs)
        assert np.array_equal(y, y0) and np.array_equal(eta, eta0)
        assert float(np.trapezoid(c, y)) == pytest.approx(a0, rel=1e-12)
        # anchored at the root: the multiplier is 1 there, so the only change
        # to the root chord is the (small) area-preserving rescale
        assert sw.chord_multiplier(np.array([0.0]), coeffs)[0] == 1.0
        assert c[np.argmin(np.abs(y))] == pytest.approx(
            c0[np.argmin(np.abs(y0))], rel=0.5)


def test_zero_chord_coefficients_reproduce_the_trapezoid_bit_for_bit():
    """The whole legacy-planform guarantee, asserted on exact doubles."""
    wg = sw.WingGuess()
    _, c0, _ = wg.stations()
    for order in sw.CHORD_ORDERS[1:]:
        _, c, _ = wg.stations(np.zeros(order))
        assert np.array_equal(c, c0), f"order {order} drifted"


def test_chord_law_refuses_to_collapse_the_chord():
    wg = sw.WingGuess()
    with pytest.raises(ValueError, match="collapses the chord"):
        wg.stations([-0.99])
    # and the deviation envelope is measured on the FLOWN chord
    _, c, _ = wg.stations([0.3])
    _, c_trap, _ = wg.stations()
    assert sw.chord_dev_envelope(c, c_trap) == pytest.approx(
        float(np.max(np.abs(c / c_trap - 1.0))))
    assert sw.chord_dev_envelope(c_trap, c_trap) == 0.0


def test_chord_bounds_and_order_validation():
    assert sw.chord_bounds(0, 0.5).shape == (0, 2)
    b = sw.chord_bounds(3, 0.4)
    assert b.shape == (3, 2)
    assert np.all(b[:, 0] == -0.4) and np.all(b[:, 1] == 0.4)
    for bad in (-1, 4, 9):
        with pytest.raises(ValueError, match="chord_order"):
            sw.chord_bounds(bad, 0.5)
    # a non-integral order truncates rather than raising, exactly as
    # twist_bounds does — the two laws must not disagree about their input
    assert sw.chord_bounds(1.5, 0.5).shape == sw.twist_bounds(1, 0.5).shape
    for bad in (0.0, -0.1, 1.0, 2.0):
        with pytest.raises(ValueError, match="chord_max_frac"):
            sw.chord_bounds(2, bad)


def test_mac_of_matches_the_trapezoidal_closed_form():
    wg = sw.WingGuess(taper=0.6)
    y, c, _ = wg.stations()
    # the general definition int c^2 / int c must agree with the closed form
    # the straight-taper WingGuess reports (to the grid's own accuracy)
    assert sw.mac_of(c, y) == pytest.approx(wg.mac, rel=2e-3)


def test_chord_order_grows_the_vector_bounds_labels_and_constraints():
    for m in sw.CHORD_ORDERS:
        p = sw.SectionWingProblem(twist_order=2, chord_order=m,
                                  chord_max_frac=0.4)
        assert p.dim == p.section.dim + 2 + m
        assert p.bounds.shape == (p.dim, 2)
        assert len(p.param_labels) == p.dim
        assert p.param_labels[p.section.dim + 2:] == tuple(
            f"chord_{j}" for j in range(1, m + 1))
        assert p.n_constraints == 4 + (1 if m else 0)
        assert len(p.constraint_labels) == p.n_constraints
        # x0 is the anchor section, untwisted AND unreshaped
        assert np.array_equal(p.x0[p.section.dim:], np.zeros(2 + m))
        if m:
            assert np.all(p.bounds[-m:, 1] == 0.4)


@needs_xfoil
def test_chord_mode_at_zero_coefficients_is_the_legacy_evaluation():
    """Opening the chord law must not move a single published number."""
    p0 = sw.SectionWingProblem(twist_order=1)
    p1 = sw.SectionWingProblem(twist_order=1, chord_order=3)
    r0 = sw.evaluate_section_wing(p0.x0, p0)
    r1 = sw.evaluate_section_wing(p1.x0, p1)
    assert r0["feasible"] and r1["feasible"]
    for key in ("LoD", "LD", "CL", "CD", "CDi", "CDp", "e", "S_llt",
                "alpha_root_deg", "mac_true", "cd", "cm", "tc"):
        assert r0[key] == r1[key], f"{key} moved: {r0[key]!r} vs {r1[key]!r}"
    assert r0["g_chord"] is None and r1["g_chord"] == pytest.approx(1.0)
    assert len(r0["g"]) == 4 and len(r1["g"]) == 5


@needs_xfoil
def test_chord_only_change_reuses_the_cached_polar_and_holds_area():
    """Chord moves are as cheap as twist moves: same section, same polar."""
    p = sw.SectionWingProblem(twist_order=1, chord_order=1,
                              chord_max_frac=0.5)
    base = sw.evaluate_section_wing(p.x0, p)          # warm the polar
    x = p.x0.copy()
    x[-1] = -0.3                                      # sharper taper outboard
    out = sw.evaluate_section_wing(x, p)
    assert out["feasible"], out.get("reason")
    assert out["from_cache"] is True
    # area (and so the design lift) is untouched; the SHAPE is not
    assert out["S_llt"] == pytest.approx(base["S_llt"], rel=1e-12)
    assert out["CL"] == pytest.approx(base["CL"], rel=1e-9)
    assert out["taper_flown"] < base["taper_flown"]
    assert out["chord_dev"] > 0.0
    assert out["g_chord"] == pytest.approx(
        (0.5 - out["chord_dev"]) / 0.5)


@needs_xfoil
def test_chord_deviation_cap_is_a_signed_margin_not_a_hidden_clip():
    """The envelope constraint exists because COEFFICIENTS ADD UP.

    At order 1 the box IS the cap (one coefficient cannot exceed it), so the
    margin can only bind from order 2 up — which is exactly the case the
    constraint was written for.
    """
    p = sw.SectionWingProblem(twist_order=1, chord_order=2,
                              chord_max_frac=0.3)
    x = np.concatenate([p.section.w0, [0.0], [-0.3, -0.3]])   # inside the box
    out = sw.evaluate_section_wing(x, p)
    assert out["feasible"]                 # solver-feasible …
    assert out["chord_dev"] > 0.3
    assert out["g_chord"] < 0.0            # … but design-infeasible
    assert out["LD"] > 0.0                 # true objective still reported
    f, g = sw.fg_section_wing(x, p)
    assert f == pytest.approx(out["LD"])
    assert len(g) == 5 and g[4] < 0.0
    # order 1 cannot violate it: the box already bounds the envelope
    p1 = sw.SectionWingProblem(twist_order=1, chord_order=1,
                               chord_max_frac=0.3)
    edge = np.concatenate([p1.section.w0, [0.0], [-0.3]])
    assert sw.evaluate_section_wing(edge, p1)["g_chord"] >= -1e-9


def test_chord_failure_returns_the_five_entry_penalty_contract():
    """A collapsed planform fails with the RIGHT NUMBER of margins."""
    p = sw.SectionWingProblem(twist_order=1, chord_order=2,
                              chord_max_frac=0.9)
    x = np.concatenate([p.section.w0, [0.0], [-0.9, -0.9]])
    f, g = sw.fg_section_wing(x, p)
    assert f == airfoil.PENALTY
    assert g.shape == (5,)
    assert np.all(g == airfoil.G_FAIL)
