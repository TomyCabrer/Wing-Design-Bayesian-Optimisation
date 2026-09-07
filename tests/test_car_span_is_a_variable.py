"""The car rear wing's SIZE is designed, and its endplate is a LENGTH.

Three changes to the same family, and they are one change: a rear wing's
dimensions are set by the car, not by fractions of the wing itself and not by
a constructor.

1. THE ENDPLATE HEIGHT IS IN METRES. It used to be an arc length over the
   semi-span. What a plate has to do is reach from the wing to the bodywork —
   a distance between two pieces of car — so expressing it as a fraction of a
   span that is now itself a design variable made the reach constraint mean
   something different at every candidate.
2. THE SPAN IS SEARCHED, and so is the reference AREA. Together they are the
   aspect ratio, so they trade induced drag against the bending the mount has
   to carry, and both are quoted in the unit the car states them in — a
   regulation or the bodywork bounds a LENGTH and an AREA, never a fraction
   of some default.
3. THE BAND IS A ROW OF THE DESIGN BOX, and only that. ``b_m`` and ``S_m2``
   are ordinary rows, so "how wide may this wing be" is asked exactly where
   every other bound is asked. There is no ``span_min_m`` / ``area_min_m2``
   flag any more, and the second channel is not merely tidied away: it broke
   in both directions at once, and both directions are pinned below
   (``test_a_widened_row_moves_the_search_too`` and
   ``test_the_band_has_one_channel_and_the_design_box_is_it``).

Every registered car family carries all three, so all three families are
checked here.
"""

import numpy as np
import pytest

from aerobo import api
from aerobo import carwing as cw
from aerobo import endplate as ep
from aerobo.objective import PENALTY

FAMILIES = ("car rear wing", "car rear wing (two-element)",
            "car rear wing + endplates")

#: the size block, in the order the vector carries it (api._CAR_SIZE_ROWS)
SIZE_ROWS = ("S_m2", "b_m")


def _built(name, flags=None, box=None):
    """The family as the registry builds it: ``flags`` are VALUE changes,
    ``box`` is the design box's own rows (bounds_overrides)."""
    return api.PROBLEM_SPECS[name].build({}, dict(flags or {}),
                                         dict(box) if box else None)


def _row(built, label):
    i = list(built.param_labels).index(label)
    return (float(built.bounds[i][0]), float(built.bounds[i][1]))


def _validated_row(built, label):
    """The band the PROBLEM itself validated, which is the one a widened box
    row is clipped back to (api.rows_outside_validity)."""
    i = list(built.param_labels).index(label)
    row = built.problem.bounds[i]
    return (float(row[0]), float(row[1]))


def _mid(built, **named):
    x = 0.5 * (built.bounds[:, 0] + built.bounds[:, 1])
    lab = list(built.param_labels)
    for k, v in named.items():
        x[lab.index(k)] = float(v)
    return x


# ------------------------------------------------- the vector says what it is

@pytest.mark.parametrize("name", FAMILIES)
def test_the_size_block_is_two_lengths_and_the_plate_is_a_length(name):
    built = _built(name)
    lab = list(built.param_labels)
    assert "endplate_h_m" in lab
    assert "endplate_h_frac" not in lab
    # the size block sits AHEAD of the chord coefficients (the package's
    # [family][size][chord] stacking rule), i.e. last in a chord-free vector,
    # and the AREA goes ahead of the SPAN so the span keeps its slot whether
    # or not the area is free (carwing.s_from_x)
    assert tuple(lab[-2:]) == SIZE_ROWS
    assert _row(built, "b_m") == cw.SPAN_BOUNDS_M
    assert _row(built, "S_m2") == cw.AREA_BOUNDS_M2
    # ...and the published wing is INTERIOR to both rows, so the box never
    # picks the answer by having the design point on its own edge
    lo, hi = _row(built, "b_m")
    assert lo < built.problem.b < hi
    lo, hi = _row(built, "S_m2")
    assert lo < built.problem.S < hi


def test_the_area_is_designed_by_the_family_itself_not_by_a_twin():
    """There is no fixed-area/free-area PAIR any more. How big the wing is is
    a design question exactly as how wide it is, and it is asked in the same
    place — so the base name IS the free-area problem and the old twins are
    not registered at all (a stale name would offer a user a family whose
    S_m2 row is missing from every card that draws one)."""
    for gone in ("car rear wing (free area)",
                 "car rear wing (two-element, free area)",
                 "car rear wing + endplates (free area)"):
        assert gone not in api.PROBLEM_SPECS
    cars = [n for n in api.PROBLEM_SPECS if n.startswith("car ")]
    # the designed-plate family also carries a twin per PLATE freedom (its
    # cant and its root blend, each a design row on the free arm), and the
    # chord law composes on all of them
    plates = ["car rear wing + endplates [free cant]",
              "car rear wing + endplates [free blend]",
              "car rear wing + endplates [free cant, free blend]"]
    bases = list(FAMILIES) + plates
    assert sorted(cars) == sorted(
        bases + [api.CHORD_TWINS[n] for n in bases])
    for name in bases:
        assert "S_m2" in api.PROBLEM_SPECS[name].default_bounds


@pytest.mark.parametrize("name", FAMILIES)
def test_the_chord_twin_keeps_the_size_block_ahead_of_the_chord_block(name):
    """Adding the chord law must not move the size rows: everything reads the
    modifier blocks back from the END of the vector."""
    plain = _built(name)
    twin = _built(api.CHORD_TWINS[name])
    lab = list(twin.param_labels)
    n = len(plain.param_labels)
    assert lab[:n] == list(plain.param_labels)
    assert tuple(lab[n - 2:n]) == SIZE_ROWS
    assert all(s.startswith("chord_k") for s in lab[n:])
    # the twin searches the same two size bands, so composing the chord law
    # with a designed size is not a different question about the size
    for label in SIZE_ROWS:
        assert _row(twin, label) == _row(plain, label)


# ------------------------------------------------------------- the size trade

@pytest.mark.parametrize("name", FAMILIES)
def test_more_span_at_a_held_area_is_more_downforce_and_more_bending(name):
    """The trade the span variable exists for: at a HELD reference area a
    wider wing is a higher aspect ratio (less induced drag, more downforce)
    carried on a longer arm (more deflection for the mount to hold)."""
    built = _built(name)
    lo, hi = cw.SPAN_BOUNDS_M
    held = 0.5 * (cw.AREA_BOUNDS_M2[0] + cw.AREA_BOUNDS_M2[1])
    narrow = built.evaluate(_mid(built, b_m=lo, S_m2=held, alpha_deg=8.0))
    wide = built.evaluate(_mid(built, b_m=hi, S_m2=held, alpha_deg=8.0))
    assert narrow["feasible"] and wide["feasible"]
    assert wide["AR"] > narrow["AR"]
    assert wide["CZ"] > narrow["CZ"]
    assert wide["CDi"] < narrow["CDi"]
    assert wide["deflection_m"] > narrow["deflection_m"]
    # the area the coefficients are referenced to is the one in the VECTOR —
    # comparing two spans by a COEFFICIENT is honest only because the area
    # was held HERE, and holding it is now something the caller does rather
    # than something the family does for them
    assert wide["S_m2"] == narrow["S_m2"] == pytest.approx(held)
    assert built.problem.S != pytest.approx(held)


@pytest.mark.parametrize("name", FAMILIES)
def test_the_beam_and_the_coefficients_are_the_wing_that_was_flown(name):
    """Not the problem's nominal wing. A candidate whose span and area differ
    from the constructor values must report the aspect ratio and the bending
    of the wing it actually flew."""
    built = _built(name)
    out = built.evaluate(_mid(built, b_m=1.9, S_m2=0.25))
    assert out["b_m"] == pytest.approx(1.9)
    assert out["S_m2"] == pytest.approx(0.25)
    assert out["AR"] == pytest.approx(1.9 ** 2 / 0.25, rel=1e-9)
    # the nominals are untouched, which is what makes the assertion above a
    # statement about the vector and not a coincidence
    assert built.problem.b != pytest.approx(1.9)
    assert built.problem.S != pytest.approx(0.25)


# ---------------------------------------------------- the plate is a length

@pytest.mark.parametrize("name", FAMILIES)
def test_a_plate_of_a_given_height_is_the_same_plate_at_any_size(name):
    """The one property a fraction of the semi-span could not have — and it
    now has to survive a moving AREA as well as a moving span."""
    built = _built(name)
    heights = []
    for b, s in ((1.3, 0.15), (1.6, 0.29), (1.9, 0.45)):
        out = built.evaluate(_mid(built, b_m=b, S_m2=s, endplate_h_m=0.20))
        assert out["feasible"], out["reason"]
        heights.append(out["endplate_h_m"])
    assert heights == pytest.approx([0.20] * 3)


@pytest.mark.parametrize("mount", sorted(cw.MOUNTS))
def test_the_reach_is_a_distance_between_two_pieces_of_car(mount):
    """The plate spans from the wing down to the attachment deck. Both ends
    are heights above the track, so the margin may not depend on the wing's
    size at all — and it is the plate's tip height, in metres, that is
    compared with them.

    ON EVERY LAYOUT. Both registered mounts take the load out through the
    plates, so there is no arrangement in which the plate may stop short of
    the car: the margin is live on both, and this replaces the older test
    that pinned it INACTIVE (a dormant +1.0 sentinel) under the pylon layout
    that no longer exists.
    """
    prob = ep.CarWingEndplateProblem(mount=mount)
    built = _built("car rear wing + endplates", flags={"mount": mount})
    ride = 0.50
    reach = prob.reach_m(ride)
    margins, short = [], []
    for b, s in ((1.2, 0.15), (1.6, 0.29), (2.0, 0.45)):
        out = built.evaluate(_mid(built, b_m=b, S_m2=s, endplate_h_m=reach,
                                  ride_height_m=ride))
        assert out["reach_required"] is True
        assert "endplate reach margin" in out["constraint_labels"]
        assert out["endplate_tip_z_m"] == pytest.approx(reach)
        margins.append(out["g_reach"])
        # ...and a plate that stops half way to the deck is refused BY THE
        # MARGIN, on this layout too: a constant would pass here
        half = built.evaluate(_mid(built, b_m=b, S_m2=s,
                                   endplate_h_m=0.5 * reach,
                                   ride_height_m=ride))
        short.append(half["g_reach"])
    assert margins == pytest.approx([0.0] * 3, abs=1e-12)
    assert all(g < 0.0 for g in short)


# --------------------------------------------------------- the size box ROWS

@pytest.mark.parametrize("name", FAMILIES)
def test_the_design_box_rows_are_the_bands_the_problem_searches(name):
    """A regulation or a bodywork width, stated once, in metres and m^2 — as
    ROWS, which is the only place either is asked."""
    built = _built(name, box={"b_m": (1.0, 1.4), "S_m2": (0.20, 0.30)})
    assert _row(built, "b_m") == (1.0, 1.4)
    assert _row(built, "S_m2") == (0.20, 0.30)
    # ...and the PROBLEM got them, which is what stops the sampler and the
    # problem's own bounds check from describing two different boxes
    assert built.problem.span_bounds_m == pytest.approx((1.0, 1.4))
    assert built.problem.area_bounds_m2 == pytest.approx((0.20, 0.30))
    # each row is a complete sentence on its own: "no wider than 1.4 m" says
    # nothing about how big the wing may be
    span_only = _built(name, box={"b_m": (1.0, 1.4)})
    assert _row(span_only, "b_m") == (1.0, 1.4)
    assert _row(span_only, "S_m2") == cw.AREA_BOUNDS_M2
    area_only = _built(name, box={"S_m2": (0.20, 0.30)})
    assert _row(area_only, "S_m2") == (0.20, 0.30)
    assert _row(area_only, "b_m") == cw.SPAN_BOUNDS_M


@pytest.mark.parametrize("name", FAMILIES)
def test_a_widened_row_moves_the_search_too(name):
    """THE DEFECT THIS FIXED. Widening the ``b_m`` row past the family's own
    band used to move the box the sampler drew from and leave the problem's
    bounds where they were, so every draw above the family band came back
    refused — a run made entirely of ``bounds violation`` at the sentinel
    score, on the row the user had just widened."""
    wide_span = 1.5 * cw.SPAN_BOUNDS_M[1]
    # what it did before, and still does when the row was NOT widened: the
    # problem refuses a wing outside its own band rather than flying it
    put = _built(name)
    refused = put.evaluate(_mid(put, b_m=wide_span))
    assert refused["feasible"] is False
    assert refused["reason"] == "bounds violation"
    assert refused["score"] == PENALTY
    # ...and what it does now that the row IS the band
    built = _built(name, box={"b_m": (cw.SPAN_BOUNDS_M[0], wide_span)})
    assert _row(built, "b_m") == (cw.SPAN_BOUNDS_M[0], wide_span)
    out = built.evaluate(_mid(built, b_m=wide_span))
    assert out["feasible"] is True, out["reason"]
    assert out["b_m"] == pytest.approx(wide_span)
    assert out["score"] > PENALTY
    # the shell's own reader agrees: no row of this box is drawn from and
    # then refused, which is the test gui.v3.relax runs before offering a
    # widening to a user with no solution
    assert api.rows_outside_validity(built) == []


@pytest.mark.parametrize("name", FAMILIES)
def test_the_band_that_goes_in_comes_back_out_unchanged(name):
    """IDEMPOTENT UNDER REBUILD, which ``gui.v3.relax.clip_to_validity``
    depends on: it clips a widened row back to the problem's validated band,
    rebuilds, and aborts the whole no-solution card if anything is still
    outside. A family that re-derived or re-clamped the band it was handed
    would fail on the second pass too."""
    for box in ({}, {"b_m": (1.2, 3.0)}, {"S_m2": (0.05, 0.9)}):
        first = _built(name, box=box or None)
        # what the PROBLEM validated — the band relax hands back, not the one
        # the box asked for
        fed_back = {label: _validated_row(first, label) for label in SIZE_ROWS}
        second = _built(name, box=fed_back)
        assert {label: _validated_row(second, label)
                for label in SIZE_ROWS} == fed_back
        # ...and one pass was enough: nothing is left drawn-from-but-refused,
        # which is the condition relax aborts the no-solution card on
        assert api.rows_outside_validity(second) == []


@pytest.mark.parametrize("name", FAMILIES)
@pytest.mark.parametrize("label, bad, message", [
    ("b_m", (2.0, 1.0), "span bounds"),
    ("S_m2", (0.30, 0.20), "area bounds"),
])
def test_an_inverted_size_row_is_refused_at_build_time(name, label, bad,
                                                       message):
    """And in the physics module's own words: the row reaches the validator
    whole, so a malformed band meets a sentence about spans or areas rather
    than a TypeError from somewhere downstream."""
    with pytest.raises(ValueError, match=message):
        _built(name, box={label: bad})


def test_the_band_has_one_channel_and_the_design_box_is_it():
    """THE OTHER DIRECTION OF THE SAME DEFECT. A band typed as a FLAG moved
    the search while the design box on screen showed the family default — the
    box shown was not the box searched, on the row the card had just been
    used to state. With one channel there is nothing left to disagree: the
    old flags are refused outright, and the card sends none of them."""
    from gui import nice_app as v1

    for name in FAMILIES:
        for gone in ("span_min_m", "span_max_m", "area_min_m2", "area_max_m2"):
            with pytest.raises(KeyError, match=gone):
                api.check_flags(name, {gone: 1.4})
        # the reader every card quotes the band from agrees with the box the
        # run is built with, which is what "shown == searched" means here
        assert api.span_box(name) == _row(_built(name), "b_m")
        assert api.span_box(name, label="S_m2") == _row(_built(name), "S_m2")
    # ...and the V1 card has no second channel to send: the helpers that read
    # its own span/area fields are gone with the fields
    assert not hasattr(v1, "_car_span_band")
    assert not hasattr(v1, "_car_area_band")
    ch = {"medium": "track", "car_span_min_m": 1.0, "car_span_max_m": 1.4,
          "car_area_min_m2": 0.2, "car_area_max_m2": 0.3}
    flags = v1.car_flags(ch)
    assert not any(k in flags for k in ("span_min_m", "span_max_m",
                                        "area_min_m2", "area_max_m2"))
    api.check_flags("car rear wing", flags)     # every key it does send lands


# ----------------------------------------------------------- still in contract

@pytest.mark.parametrize("name", FAMILIES)
def test_in_contract_failures_still_never_raise(name):
    built = _built(name)
    out = built.evaluate(np.zeros(built.dim))    # a zero span and area, among
    assert out["feasible"] is False              # others
    assert isinstance(out["reason"], str) and out["reason"]
