""""Still no dihedral or/and sweep" — and the card was there all along.

V5 shipped `wing_dihedral_deg` and `wing_sweep_deg`, a lattice that flies
them, a design-box row, and a card. The user opened the shell and reported
that none of it existed. Both halves of that were true at once, and neither
was catchable by the suite V5 wrote, because every one of its assertions was
about `api.PROBLEM_SPECS[...].flags` and none of them rendered a view.

**Where it was.** The card lived in the wing stage's SOLVER tab, in the
right-hand column beside the optimiser and the budget. A user asking "can I
give this wing dihedral?" opens Wing type — the card that asks for the tail
layout, the tip device and the planform — sees no such question, and stops.
It is geometry, so it is now on the geometry card.

**What it said when they got there.** The gate is real: a cant is scored by
the panel solver, and the family V3's own "add a tail" action derives
(`tail [designed tail]`) solves its wing on a lifting line, which has no
out-of-plane geometry to give a dihedral effect to. So the card correctly
refused — and then named `tail + t/c` as the family that would work. That
address is unreachable from this shell (nothing in V3 writes the thickness
sweep since the stage-3 section panel was deleted) and it silently drops the
designed tail the user had just asked for.

A refusal a user cannot act on is the same as no feature — and the answer
to that, for a while, was a refusal card that offered ROUTES: buttons that
changed the family to one which could score a cant. The user has since
asked for those to go (they read the card on a hydrofoil): a heading that
appears only to say "not here", above a button whose only action is to
become a different aeroplane, is not the question the heading asks.

So this file now pins the other half of the same rule. Where the family
CAN score a cant the card is on the geometry tab, one press from the
fields. Where it cannot, stage 3 says nothing about a cant at all — and
the ordinary controls (the tail's height, a tip device, the tail's own
planform) still reach a family that can, which is what makes the silence
honest rather than a dead end.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aerobo import api                                          # noqa: E402
from gui.v3 import session                                      # noqa: E402


def _shell():
    from gui.v3.app import assemble
    return assemble("air")


def _water():
    """A craft with no cant answer at ALL — no flags, no rows, no twin."""
    from gui.v3.app import assemble
    return assemble("water")


def _text(view) -> str:
    return " ".join((getattr(e, "text", "") or "")
                    for e in view.descendants())


def _buttons(view) -> list[str]:
    return [(getattr(e, "text", "") or "").strip()
            for e in view.descendants()
            if type(e).__name__ == "Button"]


def _toggle(element, value: bool) -> None:
    """Move a switch the way the browser does — through its own handler."""
    class _E:                                    # the ValueChangeEventArguments
        def __init__(self, v):
            self.value = v
    element.value = value
    for listener in (getattr(element, "_event_listeners", None) or {}).values():
        if listener.type in ("update:model-value", "change"):
            listener.handler(_E(value))
            return
    handler = getattr(element, "_change_handler", None) or getattr(
        element, "_handle_value_change", None)
    if handler is None:
        raise AssertionError(f"{element} has no change handler")
    handler(value)


def _labels(view) -> list[str]:
    return [(getattr(e, "text", "") or "").strip() for e in view.descendants()
            if "field-label" in (getattr(e, "_classes", None) or [])]


def _with_tail():
    """A wing + tail whose family CANNOT score a cant — this file's subject.

    The shell used to open on exactly this. It does not any more: since
    "the V3 shell opens on a whole aeroplane" the tail's HEIGHT opens FREE,
    which is one of the routes below, so the default configuration now
    answers the cant question and no refusal is drawn on it at all. That is
    the good news the premise test exists to make somebody notice
    (:func:`test_the_shell_now_opens_on_a_family_that_answers_the_cant`
    states it); the refusal is still one click away, because a FIXED tail
    height is an ordinary answer a user gives, and that is the
    configuration the rest of this file pins.
    """
    ctx = _shell()
    ctx.act("set_choice", "tail", True)
    ctx.act("set_choice", "tail_height", "fixed")
    return ctx


# ------------------------------------------------------------- where it is

def test_the_cant_question_is_on_the_geometry_card_not_the_solver_tab():
    """Asked on a family that HAS the rows — the shell's own default, whose
    tail height opens free. The fixed-height twin below is asked nothing at
    all now, so it can no longer carry this assertion."""
    ctx = _shell()
    ctx.act("set_choice", "tail", True)
    ctx.render("wing", "type")
    assert "Wing cant and sweep" in _text(ctx.views[("wing", "type")]), \
        "the question is not on the card that asks about the wing's shape"


def test_it_is_asked_once():
    """One question, one place: it must not ALSO still be on the solver."""
    ctx = _shell()
    ctx.act("set_choice", "tail", True)
    ctx.render("wing", "type")
    ctx.render("wing", "solver")
    assert "Wing cant and sweep" not in _text(ctx.views[("wing", "solver")])


# ------------------------------------------------- the refusal is actionable

def test_the_shell_now_opens_on_a_family_that_answers_the_cant():
    """THE GOOD NEWS, said out loud rather than passing silently.

    The premise of this file used to be that V3's own "add a tail" landed
    on a family with no cant question. It no longer does: the shell opens
    the tail's height FREE, which makes the family lattice-backed, so the
    cant is asked rather than refused on the configuration a user meets
    first. Pinned so that a future change quietly putting the refusal back
    on the default has to argue with a test.
    """
    ctx = _shell()
    ctx.act("set_choice", "tail", True)
    spec = api.PROBLEM_SPECS[ctx.S["wing"]["problem"]]
    assert (set(api.WING_CANT_KEYS) <= set(spec.flags)
            or api.cant_is_searched(ctx.S["wing"]["problem"])), \
        f"{ctx.S['wing']['problem']!r} cannot be asked about its cant"


def test_the_fixed_height_tail_configuration_still_cannot_score_a_cant():
    """The premise, on the configuration this file is now about. If this
    ever stops being true the rest of the file is about a refusal that no
    longer happens — which would be good news, and would need saying rather
    than silently passing."""
    ctx = _with_tail()
    spec = api.PROBLEM_SPECS[ctx.S["wing"]["problem"]]
    assert not set(api.WING_CANT_KEYS) <= set(spec.flags)
    assert not api.cant_is_searched(ctx.S["wing"]["problem"])


def test_a_family_that_cannot_score_a_cant_is_asked_nothing_about_one():
    """No heading, no sentence, no route button — the user's ask.

    Every string the old card could put on screen is asserted absent, not
    just the heading, because the complaint was about the BUTTON and a
    sentence with no heading would read the same way.

    ASKED OF A CONFIGURATION THAT REALLY CANNOT. This used to be
    :func:`_with_tail`, and that was too wide a reading of the ask, which
    was scoped to "everywhere it cannot be SCORED": the fixed-height tail
    scores a cant perfectly well one toggle away
    (``tail [designed tail, free cant]``), so what the deletion took from it
    was the ordinary question and not a refusal. The water craft is the
    configuration the complaint was actually made on, and it is the one
    with no answer at all — its only vertical surface is the mast, which
    stands ahead of the CG.
    """
    ctx = _water()
    ctx.render("wing", "type")
    view = ctx.views[("wing", "type")]
    text, buttons = _text(view), _buttons(view)
    assert not session.cant_is_answerable(ctx.S["wing"]["problem"],
                                          ctx.S["wing"]["choices"]), \
        "this configuration can be asked, so it is the wrong premise"
    assert "Wing cant and sweep" not in text, text[:600]
    assert "LIFTING LINE" not in text
    for gone in ("let the tail's height be optimised", "add a tip device",
                 "design the tail's own planform",
                 "optimise the dihedral and sweep"):
        assert gone not in buttons, f"{gone!r} is still offered"


def test_the_route_buttons_are_gone_where_the_question_is_asked_too():
    """The user's ask, on the configuration that DOES get the card back.

    The heading is drawn on the fixed-height tail again — the cant is
    scoreable there, one ordinary toggle away — so the thing the ask was
    actually about has to be checked where the card exists: no button whose
    action is to become a different aeroplane.
    """
    ctx = _with_tail()
    ctx.render("wing", "type")
    view = ctx.views[("wing", "type")]
    assert "Wing cant and sweep" in _text(view)
    assert "LIFTING LINE" not in _text(view)
    for gone in ("let the tail's height be optimised", "add a tip device",
                 "design the tail's own planform",
                 "optimise the dihedral and sweep"):
        assert gone not in _buttons(view), f"{gone!r} is still offered"


@pytest.mark.parametrize(
    "choice,value", [("tail_height", "free"), ("winglets", "canted"),
                     ("tail_design", "planform+tip")])
def test_an_ordinary_answer_produces_the_fields(choice, value):
    """The whole point: the question becomes answerable WITHOUT a route.

    These three were the refusal card's buttons. They are ordinary controls
    on the same card — the tail's height, the tip device, the tail's own
    planform — so deleting the buttons took away a shortcut, not an
    address. Driven through the shell's own action for each, because a
    control that cannot actually reach the family is the failure being
    guarded against.
    """
    ctx = _with_tail()
    ctx.render("wing", "type")
    # BEFORE: the family scores no cant of its own. The card is drawn all
    # the same (the question is reachable from here), so what is pinned is
    # the SOLVER's state, which is what these three controls change.
    assert not set(api.WING_CANT_KEYS) <= set(
        api.PROBLEM_SPECS[ctx.S["wing"]["problem"]].flags)
    if choice == "winglets":
        ctx.act("set_winglet", value)
    else:
        ctx.act("set_choice", choice, value)

    spec = api.PROBLEM_SPECS[ctx.S["wing"]["problem"]]
    assert set(api.WING_CANT_KEYS) <= set(spec.flags), \
        f"{choice}={value!r} did not reach a family that scores a cant"
    ctx.render("wing", "type")
    # ...AND THE QUESTION IS ASKED. The card puts the pair behind a switch
    # ("does this wing have a dihedral" is a fact about the aeroplane, and
    # two empty boxes are not a way to say no), so the fields come out of
    # the switch. Both halves are asserted: the question being askable is
    # what the control reached, and the fields being one press away is what
    # makes that worth anything.
    view = ctx.views[("wing", "type")]
    sw = next((e for e in view.descendants()
               if type(e).__name__ == "Switch"
               and "dihedral" in (getattr(e, "text", "") or "")), None)
    assert sw is not None, \
        f"{choice}={value!r} reached the family and the card does not ask"
    _toggle(sw, True)
    ctx.render("wing", "type")
    labels = _labels(ctx.views[("wing", "type")])
    assert "dihedral" in labels and "quarter-chord sweep" in labels, labels


def test_the_answer_that_reaches_it_keeps_every_other_answer():
    """One choice, not a reset. The designed tail the user asked for must
    survive being given a dihedral."""
    ctx = _with_tail()
    ctx.act("set_choice", "tail_design", "planform")
    before = dict(ctx.S["wing"]["choices"])
    assert before.get("tail_height") == "fixed"
    ctx.act("set_choice", "tail_height", "free")
    after = ctx.S["wing"]["choices"]
    assert after.get("tail_design") == "planform", \
        "the route dropped the designed tail"
    changed = {k for k in before
               if before.get(k) != after.get(k)}
    assert "tail_height" in changed
    assert set(api.WING_CANT_KEYS) <= set(
        api.PROBLEM_SPECS[ctx.S["wing"]["problem"]].flags)
