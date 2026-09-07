"""Choosing how the wing is SIZED never takes a surface away.

The reported bug: pick a canted winglet (or switch the second surface on),
then set the planform control to "free span + area" or "area from the
mission's W/S" — and the tip device reverted to "none" and the tail switched
itself off, with the shell announcing "no combined solver".

The registry says otherwise. Sizing is a MODIFIER (``api.MODIFIERS`` carries
``size`` and ``size_ws``), so ``winglet + free planform`` and ``winglet +
free span (W/S)`` are registered problems, along with their capped, blended,
t/c and tail variants. What was stale was the hand-kept exclusion matrix
``nice_app.specials_compatible``, which treated the whole ``planform`` key as
a solver FAMILY — true of exactly one of its values, the published
aircraft-sizing problem.

So the rule is now read off ``MODIFIER_ON`` (``PLANFORM_MODIFIER_VALUES``):
a planform value that names a modifier composes, and only ``"aircraft"``
resets anything. Where a family genuinely has no twin for the modifier —
``winglet + airfoil (XFOIL)`` has no W/S variant — the menu does not offer it
and ``normalise_choices`` is the net; both read the registry rather than a
list kept by hand.

And ``"aircraft"`` — a family of its own, with no tip-device or tail solver —
stops being a trap: ``planform_options`` drops it from the menu while there
is anything to lose, with ``PLANFORM_OPTION_WHY`` naming the composable
freedom to use instead. A sizing control can then never delete a surface,
whichever entry is picked.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aerobo import api                                          # noqa: E402
from gui import nice_app as v1                                  # noqa: E402

SIZINGS = ("free", "wing_loading")


def _set(ch: dict, key: str, value) -> tuple[list[str], list[str]]:
    """Apply one builder choice exactly as the shells' ``set_choice`` does.

    The pre-filter (the speciality reset) and the net (``normalise_choices``)
    in one place, so a test cannot pass by driving only one of them.
    """
    v1._set_option(ch, key, value)
    losers = [k for k in v1.SPECIAL_KEYS
              if k != key and v1._held_option(ch, k) != v1.BUILDER_DEFAULTS[k]
              and not v1.specials_compatible(key, k, ch)]
    for k in losers:
        v1._set_option(ch, k, v1.BUILDER_DEFAULTS[k])
    return losers, v1.normalise_choices(ch, keep=key)


# ------------------------------------------------- 1. the reported bug, gated

@pytest.mark.parametrize("sizing", SIZINGS)
@pytest.mark.parametrize("shape", ("vertical", "canted", "blended"))
def test_choosing_a_size_keeps_the_tip_device(shape, sizing):
    ch = v1.start_choices(medium="air")
    v1.set_winglet_shape(ch, shape)
    v1.normalise_choices(ch, keep="winglets")
    before = v1.winglet_shape_key(ch)
    assert before == shape

    assert v1.option_available(ch, "planform", sizing), (shape, sizing)
    losers, dropped = _set(ch, "planform", sizing)

    assert losers == [] and dropped == [], (shape, sizing, losers, dropped)
    assert v1.winglet_shape_key(ch) == shape
    name, notes = v1.derive_problem(ch)
    assert name in api.PROBLEM_SPECS
    assert "winglet" in name, name
    assert not any("ignored" in n for n in notes), notes


@pytest.mark.parametrize("sizing", SIZINGS)
@pytest.mark.parametrize("shape", ("vertical", "canted", "blended"))
def test_the_other_order_survives_too(shape, sizing):
    """Asking for the size first and the device second is the same design."""
    ch = v1.start_choices(medium="air")
    _set(ch, "planform", sizing)
    v1.set_winglet_shape(ch, shape)
    losers, dropped = _set(ch, "winglets", v1.winglet_option_key(ch))

    assert losers == [] and dropped == [], (shape, sizing, losers, dropped)
    assert ch["planform"] == sizing
    assert v1.winglet_shape_key(ch) == shape


@pytest.mark.parametrize("sizing", SIZINGS)
def test_a_device_flown_beside_a_size_is_the_registry_s_own_problem(sizing):
    """Not just "a problem exists" — the one the registry names for it."""
    ch = v1.start_choices(medium="air")
    v1.set_winglet_shape(ch, "canted")
    _set(ch, "planform", sizing)
    base, _ = v1.derive_problem(dict(ch, planform="fixed"))
    name, _ = v1.derive_problem(ch)
    mod = "size" if sizing == "free" else "size_ws"
    assert name == api.add_modifier(base, mod), (sizing, base, name)


# ------------------------------------ 2. the tail and the section come too

@pytest.mark.parametrize("sizing", SIZINGS)
def test_the_tail_survives_the_same_choice(sizing):
    ch = v1.start_choices(medium="air")
    ch["tail"] = True
    v1.normalise_choices(ch, keep="tail")
    losers, dropped = _set(ch, "planform", sizing)
    assert losers == [] and dropped == []
    assert ch["tail"] is True
    assert "tail" in v1.derive_problem(ch)[0]


def test_a_family_with_no_twin_for_the_modifier_is_not_offered_it():
    """The honest gap: the live-XFOIL section has no wing-loading variant, so
    the menu never offers one — the registry refusing, not the matrix."""
    ch = v1.start_choices(medium="air")
    _set(ch, "airfoil", "section_wing")
    assert api.add_modifier(v1.derive_problem(ch)[0], "size_ws") is None
    assert not v1.option_available(ch, "planform", "wing_loading")
    assert v1.option_available(ch, "planform", "free")


# --------------------------- 3. the one exclusion leaves the menu instead

def _with(**kw):
    ch = v1.start_choices(medium="air")
    for key, value in kw.items():
        if key == "winglets":
            v1.set_winglet_shape(ch, value)
            key = "winglets"
        else:
            ch[key] = value
        v1.normalise_choices(ch, keep=key)
    return ch


@pytest.mark.parametrize("state", [
    dict(winglets="vertical"), dict(winglets="canted"),
    dict(winglets="blended"), dict(tail=True),
    dict(airfoil="tc_sweep"), dict(airfoil="section_wing"),
    dict(tail=True, winglets="canted"),
], ids=lambda s: "+".join(f"{k}={v}" for k, v in s.items()))
def test_the_aircraft_family_leaves_the_menu_when_it_would_cost_something(
        state):
    """The fix for the second half of the report. The published problem has
    no combined solver with any of these, so the ENTRY goes rather than the
    surface: a sizing control that deletes the tail the user just switched on
    is a trap, however loudly it announces itself afterwards."""
    ch = _with(**state)
    offered = v1.planform_options(ch)
    assert "aircraft" not in offered, (state, list(offered))
    assert "fixed" in offered and "free" in offered
    note = v1.missing_options_note(offered, v1.PLANFORM_CHOICE_LABELS,
                                   v1.PLANFORM_OPTION_WHY)
    assert "free span + area" in note, note


def test_the_aircraft_family_is_offered_when_nothing_would_be_lost():
    """Hiding it always would be deleting the published problem instead."""
    ch = v1.start_choices(medium="air")
    assert list(v1.planform_options(ch)) == list(v1.PLANFORM_CHOICE_LABELS)
    _set(ch, "planform", "aircraft")
    assert v1.derive_problem(ch)[0].startswith("free planform (aircraft)")
    # ...and it stays offered while it is the one held, so the select can
    # never read "fixed" over an aircraft-sizing run
    assert "aircraft" in v1.planform_options(ch)


def test_every_planform_entry_that_drops_out_says_why():
    """The rule the airfoil and winglet menus already obey."""
    for medium in ("air", "water", "track"):
        for state in ({}, {"tail": True}, {"winglets": "canted"}):
            ch = v1.start_choices(medium=medium)
            for key, value in state.items():
                if key == "winglets":
                    v1.set_winglet_shape(ch, value)
                else:
                    ch[key] = value
                v1.normalise_choices(ch, keep=key)
            offered = v1.planform_options(ch)
            for key in v1.PLANFORM_CHOICE_LABELS:
                if key not in offered:
                    assert v1.PLANFORM_OPTION_WHY.get(key), (medium, key)


@pytest.mark.parametrize("shape", ("vertical", "canted", "blended"))
def test_forced_past_the_menu_the_reset_still_explains_itself(shape):
    """A preset, a saved record or the API can still ask for the pair the
    menu no longer offers. The refusal then has to be the loud one — never a
    surface that quietly stops being flown."""
    ch = v1.start_choices(medium="air")
    v1.set_winglet_shape(ch, shape)
    v1.normalise_choices(ch, keep="winglets")
    losers, _ = _set(ch, "planform", "aircraft")
    assert losers == ["winglets"]
    assert v1.winglet_shape_key(ch) == "none"
    assert v1.derive_problem(ch)[0].startswith("free planform (aircraft)")
    assert "does not compose with the tip device" \
        in v1.PLANFORM_CHOICE_NOTES["aircraft"]


def test_the_newest_choice_wins_against_the_aircraft_family():
    """Reset in the other direction: asking for a device while the aircraft
    problem is selected drops the SIZING, not the device just asked for."""
    ch = v1.start_choices(medium="air")
    _set(ch, "planform", "aircraft")
    v1.set_winglet_shape(ch, "canted")
    losers, _ = _set(ch, "winglets", v1.winglet_option_key(ch))
    assert losers == ["planform"]
    assert ch["planform"] == "fixed"
    assert v1.winglet_shape_key(ch) == "canted"


# --------------------------------------- 4. the rule is read, not hand-kept

def test_the_compatible_sizes_are_read_off_the_modifier_table():
    assert v1.PLANFORM_MODIFIER_VALUES == {
        val for key, val in v1.MODIFIER_ON.values() if key == "planform"}
    assert v1.PLANFORM_MODIFIER_VALUES == {"free", "wing_loading",
                                           "wing_loading_free"}
    assert "aircraft" not in v1.PLANFORM_MODIFIER_VALUES

    ch = v1.start_choices(medium="air")
    for sizing in SIZINGS:
        ch["planform"] = sizing
        for other in ("winglets", "tail", "airfoil"):
            assert v1.specials_compatible("planform", other, ch)
            assert v1.specials_compatible(other, "planform", ch)
    ch["planform"] = "aircraft"
    for other in ("winglets", "tail", "airfoil"):
        assert not v1.specials_compatible("planform", other, ch)
        assert not v1.specials_compatible(other, "planform", ch)


def test_a_size_is_not_a_speciality_so_it_never_claims_a_family():
    """``active_specials`` and ``specials_compatible`` have to agree about
    what a planform value IS — one deciding the refusal notes, the other the
    reset. They read the same table now."""
    ch = v1.start_choices(medium="air")
    for sizing in SIZINGS:
        ch["planform"] = sizing
        assert "planform" not in v1.active_specials(ch)
    ch["planform"] = "aircraft"
    assert "planform" in v1.active_specials(ch)


# ------------------------------------------------- 5. through the V3 shell

@pytest.mark.parametrize("sizing", SIZINGS)
def test_the_v3_stage_keeps_the_device_when_the_size_is_freed(sizing, capsys):
    from gui.v3.app import assemble

    ctx = assemble("air")
    ctx.act("set_winglet", "canted")
    S = ctx.S
    assert "winglet" in S["wing"]["problem"]

    ctx.act("set_choice", "planform", sizing)
    assert S["wing"]["choices"]["planform"] == sizing
    assert v1.winglet_shape_key(S["wing"]["choices"]) == "canted"
    assert "winglet" in S["wing"]["problem"], S["wing"]["problem"]

    # ...and the menu still shows it, so state and control agree
    ctx.render("wing", "type")
    shown = [e.value for e in ctx.views[("wing", "type")].descendants()
             if isinstance(getattr(e, "options", None), dict)
             and set(e.options) == {"none", "vertical", "canted", "blended"}]
    assert shown == ["canted"], shown

    err = capsys.readouterr().err
    assert "Traceback" not in err, err


def test_the_v3_stage_keeps_the_second_surface_and_its_device(capsys):
    """The report, end to end: a tail and a tip device, then a size."""
    from gui.v3.app import assemble

    ctx = assemble("air")
    ctx.act("set_second_surface", True)
    ctx.act("set_winglet", "canted")
    S = ctx.S
    ch = S["wing"]["choices"]
    assert ch["tail"] is True and v1.winglet_shape_key(ch) == "canted"

    ctx.act("set_choice", "planform", "free")
    assert ch["tail"] is True, "the second surface switched itself off"
    assert v1.winglet_shape_key(ch) == "canted"
    assert ch["planform"] == "free"
    assert S["wing"]["problem"] == ("tail + winglet (span-capped) "
                                   "[designed tail] + free planform "
                                   "+ free chord law")

    err = capsys.readouterr().err
    assert "Traceback" not in err, err


def test_the_v3_planform_menu_never_offers_the_entry_that_would_delete_them(
        capsys):
    from gui.v3.app import assemble

    ctx = assemble("air")
    ctx.act("set_second_surface", True)
    ctx.act("set_winglet", "canted")
    ctx.render("wing", "type")
    menus = [e.options for e in ctx.views[("wing", "type")].descendants()
             if isinstance(getattr(e, "options", None), dict)
             and "fixed" in e.options and "free" in e.options]
    assert menus, "the planform select is not on screen"
    for opts in menus:
        assert "aircraft" not in opts, opts
    texts = [str(getattr(e, "text", ""))
             for e in ctx.views[("wing", "type")].descendants()]
    assert any("free span + area" in t and "Not offered here" in t
               for t in texts), "the entry vanished without a reason"

    err = capsys.readouterr().err
    assert "Traceback" not in err, err
