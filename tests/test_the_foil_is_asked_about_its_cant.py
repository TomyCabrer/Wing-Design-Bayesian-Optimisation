"""The foil had no dihedral and no sweep at all — 0 of 156 water families.

They are LATTICE geometry, so they are declared on the families whose solver
is an imaged lattice and refused on the planar ones: ``hydrofoil`` solves a
monoplane Fourier equation whose quarter-chord line is straight along y, and
a dihedral there would have nothing to act on.

STATED, NOT SEARCHED, and the reason is the one already tested in
tests/test_the_refusal_names_the_solver_it_is_running.py: ``fin.mast`` puts
the strut's quarter chord at the foil while the elevator puts the CG aft of
it, so the one vertical surface a foiling craft has is destabilising
(``Cn_beta < 0``) and ``wing_score.spiral_refusal`` refuses the design rather
than scoring it. The criterion that prices a searched cant in air reads
nothing at all under water — so there is no row, and this file does not ask
for one.

What the STATED pair is worth is measured here instead, as outcomes.
"""
from __future__ import annotations

import numpy as np
import pytest

from aerobo import api

LATTICE = ("hydrofoil + winglet", "hydrofoil + elevator")
PLANAR = "hydrofoil"


def _centre(name, **flags):
    built = api.PROBLEM_SPECS[name].build({}, flags, None)
    return built, np.asarray(built.bounds, dtype=float).mean(axis=1)


@pytest.mark.parametrize("name", LATTICE)
def test_a_stated_cant_reaches_the_solver_and_changes_the_answer(name):
    """Declared, offered AND honoured — a flag swallowed on the way to the
    run is the defect class the flag census exists to catch."""
    api.check_flags(name, {"wing_dihedral_deg": 10.0, "wing_sweep_deg": 5.0})
    built, x = _centre(name)
    flat = built.evaluate(x)
    canted = api.PROBLEM_SPECS[name].build(
        {}, {"wing_dihedral_deg": 10.0}, None).evaluate(x)
    assert canted["LoD"] != flat["LoD"], "the flag was swallowed"


def test_sweep_buys_cavitation_margin_which_is_why_it_is_worth_stating():
    """A swept foil spreads its load, so its peak suction falls and the
    cavitation margin rises — at a small cost in L/D. Both directions, or
    the test cannot see a trade that has gone one-sided."""
    built, x = _centre("hydrofoil + winglet")
    flat = built.evaluate(x)
    swept = api.PROBLEM_SPECS["hydrofoil + winglet"].build(
        {}, {"wing_sweep_deg": 20.0}, None).evaluate(x)
    assert swept["g"] > flat["g"], (flat["g"], swept["g"])
    assert swept["LoD"] < flat["LoD"], (flat["LoD"], swept["LoD"])


def test_sweep_moves_the_neutral_point_of_a_craft_that_has_an_elevator():
    """The larger consequence, on the family with a pitch balance: at its
    own box centre the static margin is VIOLATED, and 20 deg of sweep moves
    x_np aft far enough to satisfy it — while gaining L/D."""
    built, x = _centre("hydrofoil + elevator")
    flat = built.evaluate(x)
    swept = api.PROBLEM_SPECS["hydrofoil + elevator"].build(
        {}, {"wing_sweep_deg": 20.0}, None).evaluate(x)
    sm_flat = float(np.atleast_1d(flat["g"])[1])
    sm_swept = float(np.atleast_1d(swept["g"])[1])
    assert sm_flat < 0.0 < sm_swept, (sm_flat, sm_swept)
    assert swept["LoD"] > flat["LoD"], (flat["LoD"], swept["LoD"])


def test_anhedral_buys_submergence_and_pays_in_draught():
    """The other trade, and the cost that closes it: tips down sit deeper,
    which is what the draught cap is about."""
    built, x = _centre("hydrofoil + winglet")
    flat = built.evaluate(x)
    down = api.PROBLEM_SPECS["hydrofoil + winglet"].build(
        {}, {"wing_dihedral_deg": -20.0}, None).evaluate(x)
    assert down["draught_m"] > flat["draught_m"], (
        flat["draught_m"], down["draught_m"])


def test_the_lifting_line_family_refuses_a_cant_it_could_not_act_on():
    with pytest.raises(KeyError, match="wing_dihedral_deg"):
        api.check_flags(PLANAR, {"wing_dihedral_deg": 5.0})


def test_the_census_is_the_lattice_families_and_only_those():
    water = [n for n, s in api.PROBLEM_SPECS.items() if s.medium == "water"]
    stated = [n for n in water
              if set(api.WING_CANT_KEYS) <= set(api.PROBLEM_SPECS[n].flags)]
    assert len(water) >= 156 and len(stated) == 150, (len(water), len(stated))
    # ...and NO water family searches one
    assert not any(api.cant_is_searched(n) for n in water)
