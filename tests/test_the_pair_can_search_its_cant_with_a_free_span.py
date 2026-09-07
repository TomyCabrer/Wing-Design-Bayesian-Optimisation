"""The user's sentence, as three tests: "Doesn't let tandem and
dihedral/swept with free span, not spiral stable."

Three separate failures wearing one complaint:

1. NO TANDEM COULD SEARCH ITS CANT. 80 registered tandem problems and not
   one ``[free cant]`` among them — the axis was generated for the wing+tail
   families and never for the pair, so ``wing_cant="free"`` on a tandem was
   answered by silently keeping the stated family and printing a note.
2. WITH A FREE SPAN THE BOX CENTRE DID NOT FLY. The stagger's unstated
   fractions were read off the family's NOMINAL 10 m span while the size
   modifier searched 6-40 m, so a 23 m pair's tip devices — whose height is
   a fraction of their own semi-span — met inside a gap sized for a pair
   2.3x narrower. 5 of 18 spans across the row flew.
3. AND NOTHING PRICED THE CANT ANYWAY, which is the "not spiral stable"
   half: ``TandemVLMProblem`` computed no lateral deck, so its breakdown
   carried no ``spiral_margin``, the measured band had no ``spiral`` row,
   and asking the composite for one raised. The shell said, in
   ``api.lateral_verdict``'s own words, "Weight `spiral` on stage 3's
   objective card" — on a family that had no such row.
"""
from __future__ import annotations

import sys

import numpy as np
import pytest

from aerobo import api

sys.path.insert(0, "gui")


def _derive(**choices):
    from gui.nice_app import BUILDER_DEFAULTS, derive_problem
    ch = dict(BUILDER_DEFAULTS)
    ch.update(choices)
    return derive_problem(ch)[0]


# ----------------------------------------------------------------- 1. the row

@pytest.mark.parametrize("extra", [
    {},
    {"winglets": "free"},
    {"planform": "free"},
    {"winglets": "free", "planform": "free"},
])
def test_asking_a_pair_for_a_searched_cant_lands_on_a_family_that_has_one(extra):
    """Every route the user can take, including the one they described:
    a tandem, a dihedral and a sweep, and a free span."""
    name = _derive(system="tandem", wing_cant="free", **extra)
    assert api.cant_is_searched(name), name
    labels = api.PROBLEM_SPECS[name].param_labels
    assert "wing_dihedral_deg" in labels and "wing_sweep_deg" in labels
    # ...and the STATED flag is gone, or it is the same question twice
    assert not set(api.WING_CANT_KEYS) & set(api.PROBLEM_SPECS[name].flags)


def test_a_pair_that_was_not_asked_is_the_family_it_always_was():
    assert _derive(system="tandem") == "tandem"
    assert _derive(system="tandem", winglets="free") == \
        "tandem (nonplanar) + winglets"
    assert _derive(system="tandem", airfoil="tc_sweep") == "tandem + t/c"


# ---------------------------------------------------------------- 2. the span

def _span_sweep(name):
    built = api.PROBLEM_SPECS[name].build({}, {}, None)
    x = np.asarray(built.bounds, dtype=float).mean(axis=1)
    lab = list(built.param_labels)
    i, j = lab.index("b_m"), lab.index("b_rear_m")
    flew = []
    for span in range(6, 42, 2):
        xx = x.copy()
        xx[i] = xx[j] = float(span)
        if built.evaluate(xx)["feasible"]:
            flew.append(span)
    return built, x, flew


def test_a_searched_span_gets_a_stagger_its_own_box_can_fly():
    name = "tandem (nonplanar) + winglets + free span (W/S)"
    built, x, flew = _span_sweep(name)
    out = built.evaluate(x)
    assert out["feasible"], out["reason"]          # the box CENTRE, first
    # most of the row, not a quarter of it (5 of 18 before)
    assert len(flew) >= 12, flew


def test_a_stated_stagger_is_still_the_users_outright():
    """The contract that had to survive the fix: the layout is a statement,
    never something read off a span the optimiser chose."""
    name = "tandem (nonplanar) + winglets + free span (W/S)"
    prob = api.PROBLEM_SPECS[name].build(
        {}, {"dx_m": 6.0, "dz_m": 1.5}, None).problem
    assert (prob.dx, prob.dz) == (6.0, 1.5)
    # ...and a FIXED-size pair is untouched, bit-for-bit
    fixed = api.PROBLEM_SPECS["tandem (nonplanar) + winglets"].build(
        {}, {}, None).problem
    assert (fixed.dx, fixed.dz) == (5.0, 1.0)


# -------------------------------------------------------------- 3. the price

def test_weighting_the_spiral_on_a_pair_builds_instead_of_raising():
    """The shell told the user to do this and the family had no row for it."""
    name = "tandem (nonplanar) + winglets [free cant]"
    flags = {"wing_objective": "composite",
             "wing_score_weights": {"lod": 0.7, "spiral": 0.3}}
    ref = api.wing_score_reference(name, flags=flags, n=8)
    assert "spiral" in ref["bounds"], sorted(ref["bounds"])
    built = api.PROBLEM_SPECS[name].build(
        {}, dict(flags, wing_score_reference=ref), None)
    out = built.evaluate(np.asarray(built.bounds, dtype=float).mean(axis=1))
    assert np.isfinite(out["score"]), out


def test_the_cant_row_turns_a_divergent_pair_convergent_and_charges_for_it():
    """BOTH directions. A test that only checked the margin improves could
    not see an objective that still prefers a flat pair."""
    name = "tandem (nonplanar) + winglets [free cant]"
    built = api.PROBLEM_SPECS[name].build({}, {}, None)
    built.problem.lateral = True                  # what a spiral weight arms
    lab = list(built.param_labels)
    i = lab.index("wing_dihedral_deg")
    x = np.asarray(built.bounds, dtype=float).mean(axis=1)

    def at(gamma):
        xx = x.copy()
        xx[i] = gamma
        return built.evaluate(xx)

    flat, canted = at(0.0), at(12.0)
    assert flat["spiral_margin"] < 0.0 < canted["spiral_margin"], (
        flat["spiral_margin"], canted["spiral_margin"])
    assert canted["LoD"] < flat["LoD"], (flat["LoD"], canted["LoD"])


def test_the_fin_in_the_lattice_is_the_fin_the_pair_is_charged_for():
    """One surface, one author: arming the deck must not charge it twice."""
    name = "tandem (nonplanar) + winglets"
    built = api.PROBLEM_SPECS[name].build({}, {}, None)
    x = np.asarray(built.bounds, dtype=float).mean(axis=1)
    off = built.evaluate(x)
    built.problem.lateral = True
    on = built.evaluate(x)
    # THE DOUBLE CHARGE, exactly: the fin's parasite drag has one author
    # (``cd0_fin``), so its panels must stay out of the strip sum. Unmasked
    # this moves CDp by ~4 % of L/D — a real charge, not a rounding.
    for key in ("CDp_front", "CDp_rear", "cd0_fin"):
        assert on[key] == off[key], (key, off[key], on[key])
    # ...and the STALL GATE, which must be taken over the same panels it is
    # enforced on. Equal to a rounding here and not bit-for-bit: a vertical
    # surface in the influence matrix perturbs the symmetric solve in the
    # last ulp, which is exactly why ``lateral`` defaults off and is armed
    # by the spiral weight alone rather than always being on.
    assert on["alpha_eff_range_deg"] == pytest.approx(
        off["alpha_eff_range_deg"], rel=1e-12)
    assert on["Cn_beta"] > 0.0, on["Cn_beta"]     # the pair weathercocks
