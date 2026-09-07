"""Free chord law on the Tier A wing: area preservation, legacy identity,
the collapse contract, and the physics payoff (near-elliptic loading).

The chord law is the planform's second freedom: the trapezoid is linear in
eta, so ONE taper ratio cannot reach a general (elliptic) loading. These
tests pin the three properties that make the law safe to optimise over —
the reference area never moves, the OFF path is bit-for-bit legacy, and a
non-flyable planform is refused rather than scored.
"""

import numpy as np
import pytest

from aerobo import api, geometry as g
from aerobo.llt import solve_llt
from aerobo.objective import PENALTY, Problem, evaluate, objective

ALPHA = np.deg2rad(3.0)


def _area(w: g.Wing, n: int = 200_001) -> float:
    y = np.linspace(-w.b / 2.0, w.b / 2.0, n)
    return float(np.trapezoid(w.chord(y), y))


def _e(w: g.Wing) -> float:
    _, c, tw = w.sample(60)
    return float(solve_llt(w.b, c, ALPHA + tw).e)


# ---------------- geometry: area, identity, guards ----------------

@pytest.mark.parametrize("coeffs", [(0.3,), (-0.2, 0.4), (0.25, -0.3, 0.2),
                                    (-0.5, 0.5, -0.5)])
@pytest.mark.parametrize("taper", [0.2, 0.6, 1.0])
def test_chord_law_holds_the_reference_area(coeffs, taper):
    """The rescale is closed-form, so S is held to integration accuracy —
    which matters because CL_target = W/(qS) is what the run is scored at."""
    w = g.Wing(b=10.0, S=10.0, taper=taper, chord_coeffs=coeffs)
    assert _area(w) == pytest.approx(10.0, rel=1e-9)


def test_area_factor_is_grid_independent():
    """chord(y) must not depend on the y array it is called with (the reason
    the normalisation is analytic and not an integral over the caller's grid)."""
    w = g.Wing(taper=0.5, chord_coeffs=(0.3, -0.4))
    y = np.array([0.0, 1.0, 2.5, 4.9])
    coarse = w.chord(y)
    fine = w.chord(np.linspace(-5.0, 5.0, 10_001))
    dense_at = np.interp(y, np.linspace(-5.0, 5.0, 10_001), fine)
    assert coarse == pytest.approx(dense_at, rel=1e-6)


def test_empty_chord_law_is_bit_for_bit_legacy():
    y = np.linspace(-5.0, 5.0, 257)
    legacy = g.Wing(taper=0.45, twist_tip_deg=-2.0)
    with_law = g.Wing(taper=0.45, twist_tip_deg=-2.0, chord_coeffs=())
    assert np.array_equal(legacy.chord(y), with_law.chord(y))
    assert legacy.mac == with_law.mac
    assert with_law.chord_area_factor == 1.0
    assert with_law.chord_dev == 0.0


def test_mac_matches_the_closed_form_when_the_law_is_flat():
    """A law with zero coefficients goes down the general (numeric) branch
    only if it is non-empty; a coefficient of exactly 0 keeps the trapezoid,
    so the general MAC must agree with the closed form there."""
    lam = 0.55
    closed = g.Wing(taper=lam).mac
    general = g.Wing(taper=lam, chord_coeffs=(0.0, 0.0)).mac
    assert general == pytest.approx(closed, rel=1e-9)


def test_collapsing_law_is_refused_at_construction():
    with pytest.raises(ValueError, match="collapses the chord"):
        g.Wing(taper=1.0, chord_coeffs=(-1.5,))


def test_multiplier_extrema_are_exact_not_sampled():
    """The extrema come from the derivative's roots, so they are the TRUE
    bounds: a coarse grid can only ever overestimate the minimum, and a very
    fine one converges onto the same number."""
    coeffs = (-3.0, 3.4, -1.2)          # interior minimum at eta ~ 0.703
    lo, hi = g.chord_multiplier_extrema(coeffs)
    coarse = g.chord_multiplier(np.linspace(0.0, 1.0, 6), coeffs)
    fine = g.chord_multiplier(np.linspace(0.0, 1.0, 2_000_001), coeffs)
    assert lo < coarse.min()
    assert lo == pytest.approx(float(fine.min()), abs=1e-12)
    assert hi == pytest.approx(float(fine.max()), abs=1e-12)


def test_interior_collapse_is_caught_even_though_the_ends_are_healthy():
    """Both ends are healthy (m(0) = 1, m(1) = 2.02) but the law dives
    NEGATIVE mid-span — an endpoint check alone would have flown it."""
    coeffs = (-8.0, 14.7, -5.68)
    assert g.chord_multiplier(np.array([0.0, 1.0]), coeffs).min() > 0.9
    with pytest.raises(ValueError, match="collapses the chord"):
        g.Wing(chord_coeffs=coeffs)


def test_chord_dev_is_measured_on_the_flown_chord():
    w = g.Wing(taper=1.0, chord_coeffs=(0.4,))
    y = np.linspace(-5.0, 5.0, 1001)
    trap = g.Wing(taper=1.0).chord(y)
    assert w.chord_dev == pytest.approx(
        float(np.max(np.abs(w.chord(y) / trap - 1.0))), rel=1e-6)


# ---------------- design vector plumbing ----------------

def test_chord_rows_are_appended_after_every_mode_row():
    base = g.bounds("winglet")
    box = g.bounds("winglet", chord_order=2)
    assert box.shape == (base.shape[0] + 2, 2)
    assert np.array_equal(box[: base.shape[0]], base)
    assert np.array_equal(box[-2:], np.tile([-g.CHORD_COEFF_BOUND,
                                             g.CHORD_COEFF_BOUND], (2, 1)))


def test_wing_from_x_reads_the_trailing_entries():
    x = [0.5, 1.0, -2.0, 0.1, 80.0, 0.2, -0.3]
    w = g.wing_from_x(np.array(x), mode="winglet", chord_order=2)
    assert w.taper == 0.5 and w.chord_coeffs == (0.2, -0.3)
    # the winglet pair still reads from ITS indices, not the chord rows
    assert g.bounds("winglet", chord_order=2).shape[0] == len(x)


def test_bad_chord_order_and_box_are_refused():
    with pytest.raises(ValueError, match="chord_order"):
        g.chord_bounds(7, 0.3)
    with pytest.raises(ValueError, match="chord_max_frac"):
        g.chord_bounds(2, 10.0)


# ---------------- objective contract ----------------

def test_chord_order_zero_reproduces_the_legacy_objective_exactly():
    x = np.array([0.6, 1.0, -3.0])
    assert objective(x, Problem()) == objective(x, Problem(chord_order=0))


def test_flat_law_reproduces_the_trapezoid_score():
    """Zero coefficients must give the SAME L/D as the 3-D vector — the law
    is a strict generalisation, not a re-parameterisation."""
    x3 = np.array([0.6, 1.0, -3.0])
    x6 = np.array([0.6, 1.0, -3.0, 0.0, 0.0, 0.0])
    a = evaluate(x3, Problem())
    b = evaluate(x6, Problem(chord_order=3))
    assert b["LoD"] == pytest.approx(a["LoD"], rel=1e-12)
    assert b["chord_dev"] == 0.0


def test_collapsing_candidate_returns_the_penalty_contract():
    prob = Problem(chord_order=1, chord_max_frac=1.5)
    out = evaluate(np.array([1.0, 0.0, 0.0, -1.4]), prob)
    assert out["feasible"] is False and "collapses the chord" in out["reason"]
    assert objective(np.array([1.0, 0.0, 0.0, -1.4]), prob) == PENALTY


# ---------------- physics payoff ----------------

def test_cubic_law_reaches_near_elliptic_loading():
    """The point of the whole feature: a rectangular baseline reshaped by a
    cubic law reaches e ~ 0.999, past the best straight taper's 0.984."""
    rect = _e(g.Wing(taper=1.0))
    best_taper = max(_e(g.Wing(taper=t)) for t in np.linspace(0.2, 1.0, 81))
    # coefficients fitted offline against the elliptic chord
    elliptic_ish = _e(g.Wing(taper=1.0,
                             chord_coeffs=(-0.5273, 1.1818, -1.4895)))
    assert rect < 0.93
    assert 0.98 < best_taper < 0.99
    assert elliptic_ish > 0.999


def test_law_can_only_help_the_search():
    """Whatever the best trapezoid scores, the 6-D box contains it (the law
    is anchored at the same taper with zero coefficients)."""
    prob3, prob6 = Problem(), Problem(chord_order=3)
    x = np.array([0.45, 2.0, -3.0])
    assert evaluate(np.concatenate([x, np.zeros(3)]), prob6)["score"] == \
        pytest.approx(evaluate(x, prob3)["score"], rel=1e-12)


# ---------------- API wiring ----------------

@pytest.mark.parametrize("name,dim", [("wing (free chord law)", 6),
                                      ("winglet + free chord law", 8)])
def test_chord_law_specs_build_and_evaluate(name, dim):
    built = api.PROBLEM_SPECS[name].build({}, {}, None)
    assert built.dim == dim == len(built.param_labels)
    assert built.param_labels[-3:] == ("chord_k1", "chord_k2", "chord_k3")
    out = built.evaluate(built.bounds.mean(axis=1))
    assert out["feasible"] and out["LoD"] > 0.0


def test_chord_max_frac_flag_widens_the_box_without_changing_the_dimension():
    spec = api.PROBLEM_SPECS["wing (free chord law)"]
    wide = spec.build({}, {"chord_max_frac": 1.0}, None)
    assert wide.dim == spec.build({}, {}, None).dim
    assert wide.bounds[-1].tolist() == [-1.0, 1.0]


def test_adjoint_is_not_offered_for_the_chord_law_variant():
    """The adjoint problem is the 3-D torch trim vector; offering it on a
    6-D vector would silently optimise the wrong thing."""
    assert "adjoint" not in api.compatible_optimisers("wing (free chord law)")
