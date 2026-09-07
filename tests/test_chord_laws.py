"""The chord law is a LAW, not just a cubic.

``c(eta) = c_trap(eta) * m(eta) * F`` was one shape with three coefficients.
It is now four shapes, and which one the coefficients describe is a design
decision of its own (``geometry.CHORD_LAWS``):

* ``poly``      — the published cubic. Its ends move: the multiplier at the
                  tip is 1 + sum k_j, so filling the tip in lifts the root
                  chord too once the area is rescaled back.
* ``ends``      — the same freedom BETWEEN the ends and none at them. Every
                  basis shape vanishes at the root and the tip AND integrates
                  to zero area against the trapezoid, so the root chord, the
                  tip chord and the area are exactly what the planform states
                  whatever the optimiser does with the middle. This is the law
                  to search when the two end chords are the user's numbers.
* ``kinked``    — a cranked planform: straight panels meeting at breaks.
* ``elliptic``  — one number: how far towards a true elliptic distribution.

The property that makes all four work is that each is AFFINE in its own
parameters, ``m = 1 + p @ phi``. That is what gives every law a closed-form
area rescale, an exact MAC, and a band (``chord_reach``) drawn the same way.

The contract held here:

* the AREA is exactly S under every law — that is what keeps CL_target, AR
  and every reference quantity meaning what they meant;
* the ``ends`` law holds both end chords EXACTLY, and still moves the middle;
* the closed-form MAC agrees with quadrature for every law, because Re_mac is
  computed from it;
* the collapse test is exact for every law (a grid can miss a narrow dip);
* ``poly`` is untouched: same numbers, same design vector, same everything.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aerobo import geometry as g       # noqa: E402

TAPERS = (1.0, 0.6, 0.25)
LAWS = g.CHORD_LAWS


def _params(law, order, rng):
    """A random point of the law's own box."""
    box = g.chord_bounds(order, 0.5, law)
    return rng.uniform(box[:, 0], box[:, 1])


def _wing(law, params, taper=0.6, b=10.0, S=10.0):
    return g.Wing(b=b, S=S, taper=taper,
                  chord_coeffs=g.ChordCoeffs(params, law))


def _area(w, n=200_001):
    y = np.linspace(0.0, w.b / 2.0, n)
    return 2.0 * np.trapezoid(w.chord(y), y)


# ------------------------------------------------------ 1. the area is the area

@pytest.mark.parametrize("law", LAWS)
@pytest.mark.parametrize("taper", TAPERS)
def test_every_law_holds_the_area(law, taper):
    """S is what CL_target, AR and every reference quantity are quoted on, so
    a law that moved it would move the design point rather than the shape."""
    rng = np.random.default_rng(7)
    order = max(g.CHORD_LAW_ORDERS[law])
    for _ in range(12):
        p = _params(law, order, rng)
        try:
            w = _wing(law, p, taper=taper)
        except ValueError:
            continue                       # a collapsed law is refused, fine
        assert _area(w) == pytest.approx(w.S, rel=1e-7)


# --------------------------------------------------- 2. the ends, held or not

@pytest.mark.parametrize("taper", TAPERS)
def test_the_ends_law_holds_both_end_chords_exactly(taper):
    """The whole point of the law: state the root and tip chords and they are
    what flies, whatever the search does between them."""
    rng = np.random.default_rng(3)
    base = g.Wing(b=10.0, S=10.0, taper=taper)
    c_root, c_tip = float(base.chord(0.0)), float(base.chord(5.0))
    moved = 0
    for _ in range(15):
        p = _params("ends", 3, rng)
        try:
            w = _wing("ends", p, taper=taper)
        except ValueError:
            continue
        assert float(w.chord(0.0)) == pytest.approx(c_root, rel=1e-12)
        assert float(w.chord(5.0)) == pytest.approx(c_tip, rel=1e-12)
        assert w.chord_area_factor == pytest.approx(1.0, abs=1e-12)
        # ...and the interior is NOT held, or the law would be a straight
        # taper with extra steps
        if abs(float(w.chord(2.5)) - float(base.chord(2.5))) > 1e-6:
            moved += 1
    assert moved >= 10


@pytest.mark.parametrize("taper", TAPERS)
def test_the_polynomial_law_does_move_its_ends(taper):
    """The contrast that makes the ``ends`` law worth having: the published
    law reshapes the whole planform, ends included."""
    base = g.Wing(b=10.0, S=10.0, taper=taper)
    w = _wing("poly", (0.4, -0.2, 0.1), taper=taper)
    assert abs(float(w.chord(0.0)) - float(base.chord(0.0))) > 1e-3


def test_the_kinked_law_is_made_of_straight_panels():
    """A cranked planform is straight between its breaks — that is what makes
    it buildable, and it is a statement a smooth cubic cannot make."""
    w = _wing("kinked", (0.3, -0.25, 0.2), taper=0.6)
    n = len(w.chord_coeffs)
    knots = np.concatenate([[0.0], np.arange(1, n + 1) / (n + 1.0), [1.0]])
    for a, b in zip(knots[:-1], knots[1:]):
        eta = np.linspace(a + 1e-6, b - 1e-6, 9)
        c = w.chord(eta * w.b / 2.0)
        # a straight panel in eta: second differences vanish
        assert np.allclose(np.diff(c, 2), 0.0, atol=1e-12)
    # ...and it really kinks: the slope changes across a break
    h = 1e-4
    for k in knots[1:-1]:
        lo = (w.chord((k - h) * w.b / 2) - w.chord((k - 2 * h) * w.b / 2)) / h
        hi = (w.chord((k + 2 * h) * w.b / 2) - w.chord((k + h) * w.b / 2)) / h
        assert abs(hi - lo) > 1e-3


def test_the_elliptic_law_walks_towards_an_ellipse():
    """Its one number is 'how elliptic', so the distance to the ellipse has
    to fall monotonically in it."""
    eta = np.linspace(0.0, 1.0, 401)
    ell = np.sqrt(np.maximum(0.0, 1.0 - eta ** 2))
    ell = ell / np.trapezoid(ell, eta)
    prev = None
    for p in (0.0, 0.2, 0.5, 0.9):
        w = _wing("elliptic", (p,), taper=0.6)
        c = w.chord(eta * w.b / 2.0)
        c = c / np.trapezoid(c, eta)
        err = float(np.max(np.abs(c - ell)))
        if prev is not None:
            assert err < prev
        prev = err


# ------------------------------------------------------------- 3. exactness

@pytest.mark.parametrize("law", LAWS)
@pytest.mark.parametrize("taper", TAPERS)
def test_the_closed_form_mac_agrees_with_quadrature(law, taper):
    """Re_mac is computed from it, so an approximate MAC is a wrong Reynolds
    number in every polar lookup downstream."""
    rng = np.random.default_rng(11)
    order = max(g.CHORD_LAW_ORDERS[law])
    for _ in range(8):
        p = _params(law, order, rng)
        try:
            w = _wing(law, p, taper=taper)
        except ValueError:
            continue
        y = np.linspace(0.0, w.b / 2.0, 400_001)
        c = w.chord(y)
        num = float(np.trapezoid(c * c, y) / np.trapezoid(c, y))
        assert w.mac == pytest.approx(num, rel=2e-8)


@pytest.mark.parametrize("law", LAWS)
def test_the_collapse_test_is_exact(law):
    """The multiplier's extrema decide whether a planform is refused; a
    sampled answer can miss a narrow interior dip and fly a collapsed wing."""
    rng = np.random.default_rng(19)
    order = max(g.CHORD_LAW_ORDERS[law])
    eta = np.linspace(0.0, 1.0, 200_001)
    for _ in range(20):
        p = _params(law, order, rng)
        lo, hi = g.chord_multiplier_extrema(g.ChordCoeffs(p, law), taper=0.45)
        m = g.chord_multiplier(eta, g.ChordCoeffs(p, law), taper=0.45)
        assert lo <= float(m.min()) + 1e-9
        assert hi >= float(m.max()) - 1e-9
        assert lo == pytest.approx(float(m.min()), abs=1e-6)
        assert hi == pytest.approx(float(m.max()), abs=1e-6)


# ---------------------------------------------- 4. the published law is intact

def test_a_plain_tuple_is_the_published_law():
    w = _wing("poly", (0.2, -0.1, 0.05))
    plain = g.Wing(b=10.0, S=10.0, taper=0.6, chord_coeffs=(0.2, -0.1, 0.05))
    assert plain.chord_law == "poly"
    assert plain.mac == w.mac
    assert plain.chord_area_factor == w.chord_area_factor
    y = np.linspace(0.0, 5.0, 33)
    assert np.array_equal(plain.chord(y), w.chord(y))


@pytest.mark.parametrize("law", LAWS)
def test_a_flat_law_is_the_straight_taper_under_every_law(law):
    """Zero parameters must be the trapezoid EXACTLY under every law, or a
    twin cannot contain the problem it is a twin of."""
    n = max(g.CHORD_LAW_ORDERS[law])
    base = g.Wing(b=10.0, S=10.0, taper=0.55)
    w = _wing(law, np.zeros(n), taper=0.55)
    y = np.linspace(0.0, 5.0, 51)
    assert np.allclose(w.chord(y), base.chord(y), rtol=0, atol=1e-15)
    assert w.mac == pytest.approx(base.mac, rel=1e-15)


def test_the_law_rides_on_the_coefficients():
    """The design vector's trailing entries answer 'which numbers' AND 'which
    shape' in one value, so no family can read one without the other."""
    x = np.array([0.5, 0.0, -2.0, 0.3, -0.2, 0.1])
    for law in LAWS:
        n = max(g.CHORD_LAW_ORDERS[law])
        cc = g.chord_coeffs_from_x(x, n, law)
        assert g.chord_law_of(cc) == law
        assert tuple(cc) == tuple(float(v) for v in x[-n:])
        assert g.Wing(b=10, S=10, taper=0.6, chord_coeffs=cc).chord_law == law
    assert g.chord_law_of(()) == "poly"


# ---------------------------------------------------------- 5. box and labels

def test_the_elliptic_law_carries_one_number_and_says_so():
    assert g.chord_param_count(1, "elliptic") == 1
    with pytest.raises(ValueError, match="at most"):
        g.chord_param_count(3, "elliptic")
    box = g.chord_bounds(1, 0.5, "elliptic")
    assert box.tolist() == [[0.0, 0.5]]          # a fraction, never negative
    assert g.chord_bounds(1, 2.0, "elliptic")[0][1] == g.CHORD_ELLIPTIC_MAX


@pytest.mark.parametrize("law", LAWS)
def test_the_labels_are_the_same_stems_for_every_law(law):
    """The row is 'the chord law's j-th parameter'. Naming the rows per law
    would make a saved run's labels disagree with the run reproducing it the
    moment the law moved."""
    n = max(g.CHORD_LAW_ORDERS[law])
    assert g.chord_labels(n, law) == tuple(f"chord_k{j}"
                                           for j in range(1, n + 1))


# ------------------------------------------------------------- 6. the band

@pytest.mark.parametrize("law", LAWS)
@pytest.mark.parametrize("taper", (1.0, 0.45))
def test_the_band_contains_every_law_in_the_box(law, taper):
    """chord_reach is what the design box SHOWS. A band that does not contain
    the planforms the box searches is the picture and the run disagreeing."""
    rng = np.random.default_rng(5)
    order = max(g.CHORD_LAW_ORDERS[law])
    box = g.chord_bounds(order, 0.35, law)
    reach = g.chord_reach(box, taper, law=law)
    assert not reach.empty
    for _ in range(40):
        p = rng.uniform(box[:, 0], box[:, 1])
        try:
            w = _wing(law, p, taper=taper)
        except ValueError:
            continue
        eta = reach.eta
        ratio = w.chord(eta * w.b / 2.0) / (w.c_root
                                            * (1.0 - (1.0 - taper) * eta))
        assert np.all(ratio >= reach.lo - 2e-3)
        assert np.all(ratio <= reach.hi + 2e-3)


@pytest.mark.parametrize("law", LAWS)
def test_the_deviation_inverse_answers_in_the_law_it_was_asked_in(law):
    """The design box asks 'how far may the law bend the chord' and stores the
    answer in the parameters; the two have to be inverses under each law."""
    order = max(g.CHORD_LAW_ORDERS[law])
    for want in (0.15, 0.35):
        f = g.chord_bound_for_dev(want, order, 0.6, law=law)
        got = g.chord_reach(g.chord_bounds(order, f, law), 0.6,
                            law=law).dev_max
        if f >= g.CHORD_COEFF_LIMIT:
            # the documented saturation: past the guard the answer comes back
            # as the limit rather than as a box nobody should search. It must
            # then be an UNDER-shoot, never a claim it reached the deviation.
            assert got <= want
            continue
        assert got == pytest.approx(want, rel=0.05)
