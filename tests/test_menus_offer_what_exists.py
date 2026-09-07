"""The builder menus offer exactly what the registry can solve.

The bug this file exists to prevent: a solver was ADDED (a tip device on the
tandem pair, a designed section beside it) and the problem was registered and
reachable through ``derive_problem`` — but the menu that selects it was still
greyed out by a blanket ``locked = water or track or tandem`` left over from
when those combinations had no solver. The capability existed and could not be
clicked.

The fix is to stop deciding availability in the presentation layer:
``nice_app.option_available`` applies the choice, derives the problem, and
reads the problem's own inverse choices back. If the value survives that round
trip there is a solver for it. These tests gate the two directions —

* every menu entry offered IS honoured (nothing decorative);
* every combination the registry can solve IS offered (nothing hidden).
"""

from __future__ import annotations

import itertools

import pytest

from aerobo import api
import gui.nice_app as v1

#: (car_two_element, fly_section) — the last two switches of the sweep below,
#: flattened into one loop so the nesting stays readable.
_BOTH_WAYS = list(itertools.product((False, True), (False, True)))

CONFIGS = {
    "air wing": dict(v1.BUILDER_DEFAULTS),
    # ...and the same wing WITH a tip device. Every configuration in this
    # table used to carry winglets="none", which is why the winglet+XFOIL
    # families' inverse could name the wrong airfoil entry for as long as it
    # did: the round trip that decides a menu was never asked about a
    # configuration holding a tip device.
    "air + tip device": dict(v1.BUILDER_DEFAULTS, winglets="capped"),
    "air + tail": dict(v1.BUILDER_DEFAULTS, tail=True),
    "tandem": dict(v1.BUILDER_DEFAULTS, system="tandem"),
    "water": dict(v1.BUILDER_DEFAULTS, medium="water"),
    "water + elevator": dict(v1.BUILDER_DEFAULTS, medium="water", tail=True),
    "track": dict(v1.BUILDER_DEFAULTS, medium="track"),
}


# ------------------------------------------- 1. nothing offered is decorative

@pytest.mark.parametrize("name", sorted(CONFIGS), ids=sorted(CONFIGS))
def test_every_offered_menu_entry_is_actually_honoured(name):
    ch = CONFIGS[name]
    for key, offered in (("airfoil", v1.airfoil_options(ch)),
                         ("winglets", v1.winglet_options(ch))):
        for value in offered:
            trial = dict(ch)
            if key == "winglets":
                wl, wtype = v1.WINGLET_OPTIONS[value]
                trial["winglets"], trial["winglet_type"] = wl, wtype
            else:
                trial[key] = value
            problem, notes = v1.derive_problem(trial)
            assert problem in api.PROBLEM_SPECS, (name, key, value)
            # the choice must not come back as "ignored" — that is exactly
            # the silent drop the menu is meant to make impossible
            label = v1._SPECIAL_LABEL[key]
            assert not any(f"{label} ignored" in n for n in notes), \
                (name, key, value)


# ------------------------------------------- 2. nothing solvable is hidden

def test_the_tandem_pair_offers_its_tip_device_and_its_designed_section():
    """The reported bug, gated directly."""
    ch = CONFIGS["tandem"]
    assert "section_wing" in v1.airfoil_options(ch)
    assert "canted_free" in v1.winglet_options(ch)
    got, _ = v1.derive_problem(dict(ch, airfoil="section_wing"))
    assert got == "tandem (nonplanar) + CST section (XFOIL)"
    got, _ = v1.derive_problem(dict(ch, winglets="free",
                                    airfoil="section_wing"))
    assert got == "tandem (nonplanar) + winglets + CST section (XFOIL)"


def test_a_tip_device_does_not_hide_the_designed_section():
    """A wing with a tip device CAN have its section shaped with it.

    ``winglet + airfoil (XFOIL)`` is the wing planform, the tip device and
    eight CST weights in one vector — the same builder as the wingless
    ``wing + airfoil (XFOIL)``, two rows longer. It inverted to
    ``section_only`` (the pure 2-D problem, which has no wing at all), so
    the round trip in :func:`option_available` refused ``section_wing`` for
    every tip-device configuration and V3 disabled its "let the wing search
    reshape it" switch with a reason that was not true.
    """
    for shape in ("vertical", "canted", "blended"):
        ch = v1.start_choices()
        v1.set_winglet_shape(ch, shape)
        v1.normalise_choices(ch)
        opts = v1.airfoil_options(ch)
        assert "section_wing" in opts, shape
        # ...and the 2-D entry is NOT offered beside a wing, which is what
        # its own reason has always said
        assert "section_only" not in opts, shape
        name, notes = v1.derive_problem(dict(ch, airfoil="section_wing"))
        assert "airfoil (XFOIL)" in name, (shape, name)
        assert not any("ignored" in n for n in notes), shape


def test_water_offers_its_tip_device_and_its_designed_section():
    for key in ("water", "water + elevator"):
        ch = CONFIGS[key]
        assert "section_wing" in v1.airfoil_options(ch), key
        assert "canted_free" in v1.winglet_options(ch), key


@pytest.mark.parametrize("name", sorted(CONFIGS), ids=sorted(CONFIGS))
def test_every_registered_problem_of_a_family_is_reachable_from_its_menus(
        name):
    """Sweep the menus of a configuration and collect what they reach; every
    problem whose builder choices match that configuration must be in it."""
    ch = CONFIGS[name]
    reachable = set()
    # every control the builder actually offers for this configuration: the
    # two menus under test, the three modifiers, and the sub-cards that carry
    # their own dimension-changing choices (the tail's arm, height and how
    # much of it is designed; the car's designed endplates)
    #
    # The tip device is chosen FIRST and the airfoil menu is then asked of
    # that state, because the two constrain each other and a session walks
    # them in that order: a wing with a device offers "shape section + wing"
    # and not the pure 2-D problem, a wing without one offers both. Asking
    # both menus of the STARTING state instead would sweep pairs the shell
    # would never show together — and miss the pairs it does.
    for wl in v1.winglet_options(ch):
        trial0 = dict(ch)
        trial0["winglets"], trial0["winglet_type"] = v1.WINGLET_OPTIONS[wl]
        for af in v1.airfoil_options(trial0):
            trial = dict(trial0, airfoil=af)
            for planform in v1.PLANFORM_CHOICE_LABELS:
                for chord in ("fixed", "free"):
                    for flight in ("fixed", "free"):
                        for arm in ("free", "fixed"):
                            for height in ("fixed", "free"):
                                for design in v1.TAIL_DESIGN_LABELS:
                                    for plates in (False, True):
                                        # ...and whether the car wing's
                                        # reference AREA is designed too: a
                                        # separate family (it adds a row and
                                        # changes the objective to a force),
                                        # so a menu that never set it would
                                        # leave that family unreachable
                                        for free_area in (False, True):
                                            # ...and whether the rear wing's
                                            # SECTION is slotted: another
                                            # family (four design rows), and
                                            # another control that would
                                            # leave it unreachable if this
                                            # sweep never set it
                                            for slot, fly in _BOTH_WAYS:
                                                # ...and whether the WING's
                                                # dihedral and sweep are
                                                # searched. A variant axis
                                                # rather than a flag (it
                                                # changes the design
                                                # DIMENSION), so it is a
                                                # different registered
                                                # family — 1536 of them —
                                                # and a sweep that never set
                                                # it left every one
                                                # unreachable.
                                                for cant in api.WING_CANTS:
                                                    reachable.add(
                                                        v1.derive_problem(dict(
                                                            trial,
                                                            planform=planform,
                                                            chord=chord,
                                                            flight=flight,
                                                            tail_arm=arm,
                                                            tail_height=height,
                                                            tail_design=design,
                                                            fly_section=fly,
                                                            car_free_area=free_area,
                                                            car_two_element=slot,
                                                            wing_cant=cant,
                                                            car_endplates=plates))[0])
    # every problem whose OWN inverse choices are this configuration's family
    # has to be one of them
    for problem in api.PROBLEM_SPECS:
        back = v1.choices_from_problem(problem)
        same_family = (back["medium"] == ch["medium"]
                       and back["system"] == ch["system"]
                       and bool(back["tail"]) == bool(ch["tail"]))
        if same_family:
            assert problem in reachable, (name, problem)


# --------------------------------------------- 3. the refusals stay honest

def test_a_family_with_no_solver_says_so_instead_of_offering_nothing():
    """A missing entry has to be explainable, or it is just a mystery."""
    for name, ch in CONFIGS.items():
        note = v1.missing_options_note(v1.airfoil_options(ch),
                                       v1.AIRFOIL_OPTION_LABELS,
                                       v1.AIRFOIL_OPTION_WHY)
        if len(v1.airfoil_options(ch)) < len(v1.AIRFOIL_OPTION_LABELS):
            assert note.startswith("Not offered here:"), name
        wnote = v1.missing_options_note(v1.winglet_options(ch),
                                        v1.WINGLET_OPTION_LABELS,
                                        v1.WINGLET_OPTION_WHY)
        if len(v1.winglet_options(ch)) < len(v1.WINGLET_OPTION_LABELS):
            assert wnote.startswith("Not offered here:"), name


def test_the_car_wing_offers_neither_and_that_is_the_point():
    """Endplates ARE the car wing's tip device and its section is fixed, so
    both menus collapse to one entry — which the UI shows as disabled with a
    reason rather than as an empty control."""
    ch = CONFIGS["track"]
    assert list(v1.airfoil_options(ch)) == ["fixed"]
    assert list(v1.winglet_options(ch)) == ["none"]


# ------------------------------------- 4. the reasons are about THIS family

@pytest.mark.parametrize("name", sorted(CONFIGS), ids=sorted(CONFIGS))
def test_a_refusal_never_explains_itself_by_naming_another_family(name):
    """"The water tip device is scored span-free" under a TANDEM pair is a
    non-answer. Every reason shown has to be true of the configuration on
    screen, so a reason that is specific to one medium lives in
    ``OPTION_WHY_BY_MEDIUM`` and only that medium sees it."""
    ch = CONFIGS[name]
    medium = ch["medium"]
    elsewhere = {"air": ("water", "track", "car "),
                 "water": ("track", "car "),
                 "track": ("water",)}[medium]
    for table in (v1.WINGLET_OPTION_WHY, v1.AIRFOIL_OPTION_WHY):
        for key, reason in v1.option_why(ch, table).items():
            low = reason.lower()
            assert not [w for w in elsewhere if w in low], (name, key, reason)


def test_a_shared_reason_is_given_once_for_all_the_entries_it_covers():
    """The car wing refuses three winglet entries for one reason; saying it
    three times reads as noise rather than as an answer."""
    ch = CONFIGS["track"]
    note = v1.missing_options_note(
        v1.winglet_options(ch), v1.WINGLET_OPTION_LABELS,
        v1.option_why(ch, v1.WINGLET_OPTION_WHY))
    shared = "the endplates ARE this wing's tip device"
    assert note.count(shared) == 1
    # ...and it still names every entry it covers
    for label in ("canted — free span extension", "canted — span-capped",
                  "vertical fence (near-90° cant)"):
        assert label in note


def test_every_missing_entry_is_explained_in_every_configuration():
    """A menu entry that vanishes with no reason is the mystery the note
    exists to prevent — so ``why`` has to cover every key that can go
    missing, in every configuration."""
    for name, ch in CONFIGS.items():
        for offered, labels, table in (
                (v1.airfoil_options(ch), v1.AIRFOIL_OPTION_LABELS,
                 v1.AIRFOIL_OPTION_WHY),
                (v1.winglet_options(ch), v1.WINGLET_OPTION_LABELS,
                 v1.WINGLET_OPTION_WHY)):
            why = v1.option_why(ch, table)
            for key in labels:
                if key not in offered:
                    assert key in why and why[key], (name, key)
