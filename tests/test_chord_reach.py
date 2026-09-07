"""What a BOX on the chord coefficients draws — the planform, not the numbers.

``chord_k1..k3`` are the right variables for a solver and the wrong ones for a
person: nobody knows what ``chord_k2 = -0.31`` draws, and widening a
coefficient's bounds says nothing about the wing it buys.
:func:`aerobo.geometry.chord_reach` answers that in the drawing — at every
station, the narrowest and the widest chord any candidate inside the box can
put there, in units of the straight-taper chord the law multiplies — and
:func:`aerobo.geometry.chord_bound_for_dev` inverts it, so a design box can
ask "how far may the law bend the chord" and store the answer in the
coefficients the solver actually searches.

The contract:

* it is measured on the chord a candidate really flies (after the
  area-preserving rescale), so the band IS :attr:`Wing.chord_dev` maximised
  over the box, and every flyable law in the box draws inside it;
* a corner the solver REFUSES (the chord collapses, ``CHORD_MULT_FLOOR``) is
  not in the band — it is counted in ``flyable_frac`` instead;
* nothing here is a design variable: it is presentation arithmetic over a box
  that already exists, so no published run moves.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aerobo import geometry     # noqa: E402


def _ratio(coeffs, taper, eta):
    """c/c_trap of a wing that really carries ``coeffs`` — the definition the
    band has to bracket, taken from the flown geometry rather than restated."""
    w = geometry.Wing(b=10.0, S=10.0, taper=taper, chord_coeffs=tuple(coeffs))
    y = 0.5 * w.b * np.asarray(eta, dtype=float)
    trap = w.c_root * (1.0 - (1.0 - taper) * np.asarray(eta, dtype=float))
    return w.chord(y) / trap


# --------------------------------------------------------------- the band
def test_no_chord_law_is_straight_taper_exactly():
    reach = geometry.chord_reach(np.zeros((0, 2)), taper=0.4)
    assert reach.dev_max == 0.0
    assert reach.flyable_frac == 1.0
    assert np.allclose(reach.lo, 1.0) and np.allclose(reach.hi, 1.0)
    assert reach.root == (1.0, 1.0) and reach.tip == (1.0, 1.0)
    assert not reach.empty


#: how far outside the drawn band a flyable law may fall. The band is
#: sampled, so it is not a bound in the interval-arithmetic sense; this is
#: the residual measured over 200k random flyable laws, and it is a
#: thousandth of a chord — a line's width on the picture the band is for.
BAND_TOL = 2e-3


@pytest.mark.parametrize("taper", [1.0, 0.45, 0.2])
def test_every_flyable_law_in_the_box_draws_inside_the_band(taper):
    """The band is a promise about the search, so it has to hold for laws the
    grid never sampled."""
    box = geometry.chord_bounds(3, 0.5)
    reach = geometry.chord_reach(box, taper)
    rng = np.random.default_rng(0)
    checked = 0
    for k in rng.uniform(-0.5, 0.5, size=(400, 3)):
        if geometry.chord_multiplier_extrema(k)[0] <= geometry.CHORD_MULT_FLOOR:
            continue                      # the solver refuses this planform
        r = _ratio(k, taper, reach.eta)
        assert np.all(r >= reach.lo - BAND_TOL), k
        assert np.all(r <= reach.hi + BAND_TOL), k
        checked += 1
    assert checked > 200                  # the sample really was flyable


def test_the_law_that_almost_collapses_is_inside_the_band_too():
    """The narrowest flyable chords sit ON the collapse boundary, which no
    grid over the box lands on — the reason collapsed grid points are pulled
    back to it. Without that the band understated the narrow side by ~2% of
    the chord on the published box, and by 30% on the widest one allowed.
    """
    taper = 0.45
    k = (0.0, 0.0, -0.94)                 # multiplier 0.06 at the tip: legal
    assert geometry.chord_multiplier_extrema(k)[0] > geometry.CHORD_MULT_FLOOR
    reach = geometry.chord_reach(geometry.chord_bounds(3, 1.0), taper)
    r = _ratio(k, taper, reach.eta)
    assert np.all(r >= reach.lo - BAND_TOL)
    assert np.all(r <= reach.hi + BAND_TOL)
    assert reach.lo[-1] < 0.12            # and it really is a narrow tip


def test_the_band_is_reached_and_the_deviation_is_the_flown_one():
    """Not a loose envelope: the extremes belong to real planforms, and
    ``dev_max`` is Wing.chord_dev at its worst over the box."""
    taper = 0.45
    box = geometry.chord_bounds(3, 0.5)
    reach = geometry.chord_reach(box, taper)

    # the same grid, taken the long way round: build the WING and ask it.
    # The band also carries the collapse boundary the grid cannot land on, so
    # it may only exceed this — and not by much, or it would be describing
    # planforms the box does not hold.
    axis = np.linspace(-0.5, 0.5, geometry.CHORD_REACH_GRID)
    best = 0.0
    for k in np.ndindex(*(len(axis),) * 3):
        c = axis[list(k)]
        if geometry.chord_multiplier_extrema(c)[0] <= geometry.CHORD_MULT_FLOOR:
            continue
        best = max(best, geometry.Wing(b=10.0, S=10.0, taper=taper,
                                       chord_coeffs=tuple(c)).chord_dev)
    assert best <= reach.dev_max <= 1.05 * best
    # the published box is FREE: the tip chord can nearly vanish or exceed the
    # trapezoid's — the fact the coefficient row (+/-0.5) cannot state
    assert reach.tip[0] < 0.15 and reach.tip[1] > 1.5
    assert reach.dev_max > 0.5


def test_the_area_is_held_so_the_band_is_a_reshape_not_a_resize():
    w = geometry.Wing(b=10.0, S=10.0, taper=0.45,
                      chord_coeffs=(0.5, -0.25, 0.1))
    y = np.linspace(0.0, w.b / 2.0, 20001)
    assert 2.0 * np.trapezoid(w.chord(y), y) == pytest.approx(w.S, rel=1e-6)


def test_a_refused_corner_is_counted_not_drawn():
    """k = (-0.5, -0.5, -0.5) collapses the chord, so Wing refuses it. It must
    not appear in the band it would otherwise dominate."""
    with pytest.raises(ValueError, match="collapses the chord"):
        geometry.Wing(chord_coeffs=(-0.5, -0.5, -0.5))

    reach = geometry.chord_reach(geometry.chord_bounds(3, 0.5), 1.0)
    assert np.all(reach.lo > 0.0)
    # the documented flyable share of the published box (~96%)
    assert 0.9 < reach.flyable_frac < 1.0

    tight = geometry.chord_reach(geometry.chord_bounds(3, 0.1), 1.0)
    assert tight.flyable_frac == 1.0
    assert tight.dev_max < reach.dev_max


def test_a_box_with_no_wing_in_it_says_so():
    box = [[-2.0, -1.9], [-2.0, -1.9], [-2.0, -1.9]]
    reach = geometry.chord_reach(box, 1.0)
    assert reach.empty and reach.flyable_frac == 0.0
    assert np.all(np.isnan(reach.lo)) and np.all(np.isnan(reach.hi))


def test_the_baseline_taper_changes_what_the_same_coefficients_draw():
    """The area rescale integrates against the trapezoid, so the band is a
    statement about a SURFACE, not about the coefficients alone."""
    box = geometry.chord_bounds(3, 0.5)
    rect = geometry.chord_reach(box, 1.0)
    pointy = geometry.chord_reach(box, 0.2)
    assert rect.taper == 1.0 and pointy.taper == 0.2
    assert not np.allclose(rect.hi, pointy.hi)
    assert pointy.tip[1] > rect.tip[1]      # a small baseline is easier to grow


def test_a_bad_box_is_refused():
    with pytest.raises(ValueError, match=r"\[low, high\]"):
        geometry.chord_reach([[0.5, -0.5]], 1.0)
    # ...and a box with more rows than the LAW offers parameters for. The
    # message names the law, because since the laws arrived the largest order
    # is a property of one (elliptic offers exactly 1).
    with pytest.raises(ValueError, match="exceeds the largest order"):
        geometry.chord_reach(np.zeros((max(geometry.CHORD_ORDERS) + 1, 2)))
    with pytest.raises(ValueError, match="exceeds the largest order"):
        geometry.chord_reach(np.zeros((2, 2)), law="elliptic")


# ------------------------------------------------------------- the inverse
@pytest.mark.parametrize("want", [0.1, 0.3, 0.6])
@pytest.mark.parametrize("taper", [1.0, 0.45])
def test_a_deviation_inverts_to_the_box_that_draws_it(want, taper):
    f = geometry.chord_bound_for_dev(want, 3, taper)
    assert 0.0 < f <= geometry.CHORD_COEFF_LIMIT
    got = geometry.chord_reach(geometry.chord_bounds(3, f), taper).dev_max
    assert got == pytest.approx(want, abs=0.005)


def test_the_inverse_is_monotone_and_clamped():
    devs = [0.05, 0.2, 0.4, 0.8]
    fs = [geometry.chord_bound_for_dev(d, 3, 0.45) for d in devs]
    assert fs == sorted(fs)
    # more than the guard can buy comes back AS the guard, not as a box
    # nobody should search
    assert geometry.chord_bound_for_dev(50.0, 3, 0.45) == \
        geometry.CHORD_COEFF_LIMIT
    with pytest.raises(ValueError, match="must be > 0"):
        geometry.chord_bound_for_dev(0.0)


def test_the_stored_bound_is_a_number_a_person_can_read():
    """A design box that reads ``|k| <= 0.1289998046875`` is the bisection's
    arithmetic showing through, not an answer anyone gave."""
    f = geometry.chord_bound_for_dev(0.3, 3, 0.6)
    assert f == round(f, 4)
