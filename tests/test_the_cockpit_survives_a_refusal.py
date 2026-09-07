"""Stage 6 refuses, and then there is no way back — three shapes of one bug.

The report was "when doing flight, sometimes after doing something different
(control etc.) it doesn't let me fly it", and the three paths that produce it
are not the same defect:

* **The switch meant the opposite of what it did.** Stage 5's elevator switch
  is captioned "hinged elevator (off = all-moving stabiliser)" and its own
  hint says a whole-chord hinge IS that stabiliser. Off sent
  ``ControlsSpec(elevator=False)``, which builds a deck with NO elevator
  column, and ``sixdof.trim_level`` solves for alpha AND one pitch control —
  so it raised before stage 6's first frame.

* **The refusal took the buttons with it.** ``_render_fly`` returned on
  ``F["error"]``, before Fly, Pause and Re-trim were created. ``F["error"]``
  is sticky (only a successful ``arm`` clears it) and ``_crash`` sets it to a
  message that says "Re-trim & reset to fly again" — so crashing, then
  looking at any other view, left the pilot facing that instruction with no
  button to obey it and no way out for the rest of the session.

* **The aeroplane on screen was not the one being flown.** Stage 5 refreshes
  the four V3 stages and itself, never stage 6; ``_render_fly`` re-armed only
  when ``F["ac"]`` was None, and an aircraft from the previous deck is not
  None. And ``_ensure_deck`` was gated on "no deck and no error", so a new run
  never replaced the deck and one failed rebuild latched for the session.

Every assertion below is an OUTCOME — a trim that succeeds, a button that
exists, an object identity that moved — not a restatement of the code.
"""

from __future__ import annotations

import pytest

from gui.v4 import app as v4app


def _report(seed: int = 0, at: float = 0.5):
    """A design report for the ``tail`` problem at a point in its own box."""
    from aerobo import api

    cfg = api.RunConfig(problem_name="tail", budget=4, seed=seed)
    built = api.PROBLEM_SPECS["tail"].build({}, {}, None)
    lo, hi = built.bounds[:, 0], built.bounds[:, 1]
    return api.design_report(cfg, lo + at * (hi - lo))


def _armed():
    """A V4 shell with a real design behind it. Stage 5 has not built yet."""
    ctx = v4app.assemble()
    ctx.S["run"]["record"] = {"pretend": "a completed run"}
    ctx.S["run"]["report"] = _report()
    return ctx


def _buttons(ctx, stage: str, view: str) -> list[str]:
    """Every button caption in one view, by walking the real element tree."""
    return [t for t in (getattr(e, "text", None)
                        for e in ctx.views[(stage, view)].descendants())
            if t]


# --------------------------------------------- off means all-moving, not gone

def test_the_all_moving_stabiliser_still_trims_the_aeroplane(capsys):
    """The switch's own caption, asserted where it broke.

    Turning the hinge off must leave a pitch control — the whole surface —
    and stage 6 must arm on it. Before this, ``arm`` came back False with
    "no control column named 'elevator' in this deck".
    """
    ctx = _armed()
    ctx.render("controls", "derivatives")             # builds the first deck
    C, F = ctx.S["controls"], ctx.S["flight"]
    assert ctx.act("flight_arm") is True, F.get("error")

    C["elevator"]["on"] = False                       # ...all-moving
    ctx.act("controls_rebuild")
    assert C.get("error") is None, C.get("error")
    assert "elevator" in C["deck"].columns, \
        "the all-moving stabiliser left the deck with no pitch control"
    assert ctx.act("flight_arm") is True, F.get("error")
    assert F.get("error") is None
    assert F.get("ac") is not None
    capsys.readouterr()


def test_the_all_moving_surface_is_the_STRONGER_of_the_two(capsys):
    """Not merely present — the whole chord, which is what "all-moving"
    means. A whole-surface incidence change out-pitches a partial hinge, so
    ``Cm_delta`` has to grow in magnitude when the switch goes off. This is
    what would fail if the fix had quietly sent the hinged chord anyway.
    """
    ctx = _armed()
    ctx.render("controls", "derivatives")
    C = ctx.S["controls"]
    C["elevator"]["chord_frac"] = 0.30
    ctx.act("controls_rebuild")
    hinged = float(C["deck"].columns["elevator"]["Cm"])

    C["elevator"]["on"] = False
    ctx.act("controls_rebuild")
    whole = float(C["deck"].columns["elevator"]["Cm"])

    assert hinged < 0.0 and whole < 0.0               # both pitch nose down
    assert abs(whole) > abs(hinged), (hinged, whole)
    capsys.readouterr()


# ------------------------------------------- a refusal keeps its own way out

def test_a_crash_does_not_take_the_retrim_button_with_it(capsys):
    """``_crash`` writes "Re-trim & reset to fly again" — so that button has
    to be on the view that shows the message."""
    ctx = _armed()
    ctx.render("controls", "derivatives")
    ctx.render("flight", "fly")
    F = ctx.S["flight"]
    assert F.get("ac") is not None, F.get("error")
    assert "Re-trim & reset" in _buttons(ctx, "flight", "fly")

    F["crashed"] = True
    F["error"] = ("GROUND CONTACT at 40.0 m/s, -20 deg pitch, +0 deg bank. "
                  "Re-trim & reset to fly again.")
    ctx.render("flight", "fly")

    captions = _buttons(ctx, "flight", "fly")
    assert "Re-trim & reset" in captions, \
        "the crash message names a button the view no longer draws"
    assert "Fly" in captions and "Pause" in captions
    texts = [getattr(e, "text", "") or ""
             for e in ctx.views[("flight", "fly")].descendants()]
    assert any("GROUND CONTACT" in t for t in texts), \
        "the reason went missing along with the dead end"
    capsys.readouterr()


def test_a_rejected_flight_condition_does_not_empty_the_cockpit(capsys):
    """``_set_condition`` puts a rejected speed back and KEEPS the reason, so
    a good aeroplane sits behind a stale warning. It must not be hidden by
    it: the aircraft is armed, so the picture and the levers belong on
    screen with the message.
    """
    ctx = _armed()
    ctx.render("controls", "derivatives")
    ctx.render("flight", "fly")
    F = ctx.S["flight"]
    good = F["ac"]

    ctx.act("flight_set_condition", "V_trim", 2.0)    # far below the stall
    assert F.get("error"), "a 2 m/s trim was accepted"
    assert F.get("ac") is not None, "the rejected speed left no aeroplane"

    ctx.render("flight", "fly")
    assert "Fly" in _buttons(ctx, "flight", "fly")
    assert F["ac"] is not None and good is not None
    capsys.readouterr()


def test_with_no_pitch_control_at_all_the_refusal_names_the_answer(capsys):
    """A single-surface design has no elevator to make all-moving. That is a
    real refusal — but it has to point at the stage-1 switch that fixes it,
    not print a Python exception about a missing dict key.
    """
    from aerobo import api

    ctx = v4app.assemble()
    cfg = api.RunConfig(problem_name="trim wing", budget=4, seed=0)
    built = api.PROBLEM_SPECS["trim wing"].build({}, {}, None)
    ctx.S["run"]["record"] = {"pretend": "a completed run"}
    ctx.S["run"]["report"] = api.design_report(cfg,
                                               built.bounds.mean(axis=1))
    ctx.render("controls", "derivatives")
    assert "elevator" not in ctx.S["controls"]["deck"].columns

    assert ctx.act("flight_arm") is False
    why = ctx.S["flight"]["error"]
    assert "ValueError" not in why and "column" not in why, why
    assert "tail" in why and "stage 1" in why, why
    capsys.readouterr()


# ------------------------------- the aeroplane flown is the one on screen

def test_moving_a_control_surface_re_arms_the_flight_stage(capsys):
    """Stage 5 never refreshes stage 6, so stage 6 has to notice for itself.
    The deck it armed on is remembered; a different one means a re-trim.
    """
    ctx = _armed()
    ctx.render("controls", "derivatives")
    ctx.render("flight", "fly")
    C, F = ctx.S["controls"], ctx.S["flight"]
    first = F["ac"]
    assert F["armed_fm"] is C["fm"]

    C["aileron"]["chord_frac"] = 0.35
    ctx.act("controls_rebuild")
    assert F["armed_fm"] is not C["fm"], "stage 5 did not build a new deck"

    ctx.render("flight", "fly")
    assert F["armed_fm"] is C["fm"], \
        "stage 6 went on flying the aeroplane from before the edit"
    assert F["ac"] is not first
    capsys.readouterr()


def test_a_new_run_replaces_the_deck_stage_6_flies(capsys):
    """The deck belongs to a REPORT. A second run is a second aeroplane, and
    nothing on stage 5 or 6 was told — ``_ensure_deck`` saw a deck and
    stopped.
    """
    ctx = _armed()
    ctx.render("controls", "derivatives")
    C = ctx.S["controls"]
    old_fm, old_deck = C["fm"], C["deck"]
    assert old_fm is not None

    ctx.S["run"]["report"] = _report(seed=1, at=0.35)   # a different design
    ctx.render("flight", "fly")

    assert C["fm"] is not old_fm, "the new run kept the old flight model"
    assert C["deck"] is not old_deck
    assert ctx.S["flight"]["armed_fm"] is C["fm"]
    capsys.readouterr()


def test_a_failed_rebuild_does_not_latch_for_the_session(capsys):
    """The no-report branch fires whenever stage 4 has nothing yet — a page
    reload, a re-run in progress. It used to set an error that
    ``_ensure_deck`` then refused to clear, so the stage stayed broken until
    an unrelated field was nudged.
    """
    ctx = v4app.assemble()                             # no design at all
    ctx.act("controls_rebuild")
    C = ctx.S["controls"]
    assert C.get("error")
    assert C.get("fm") is None, \
        "a report-less rebuild left the previous design's model behind"
    assert C.get("deck") is None

    ctx.S["run"]["record"] = {"pretend": "a completed run"}
    ctx.S["run"]["report"] = _report()
    ctx.render("controls", "derivatives")

    assert C.get("error") is None, C.get("error")
    assert C.get("deck") is not None
    capsys.readouterr()


def test_a_design_that_cannot_build_is_not_rebuilt_on_every_paint(capsys):
    """The other half of keying on the report: a rebuild that RAISED still
    stamped the report it tried, so repainting a view does not re-pay it.
    """
    ctx = _armed()
    ctx.S["controls"]["vertical"]["rudder_chord_frac"] = "not a number"
    ctx.render("controls", "derivatives")
    C = ctx.S["controls"]
    assert C.get("error"), "a junk answer built a deck"
    tried = C["deck_report"]
    assert tried is ctx.S["run"]["report"]

    ctx.render("controls", "derivatives")
    assert C["deck_report"] is tried
    capsys.readouterr()


# -------------------------------------------------- stage 1: the empennage

def test_the_empennage_field_does_not_print_its_label_over_its_value(capsys):
    """theme.py pins every field at 26 px and kills the container's top
    padding with ``!important``, so Quasar's floating label has nowhere to
    float to: it lifted 4 px and landed ON the value, printing "empennage"
    through "conventional (aft, on the fuselage)". ``widgets.select_field``
    exists for exactly this and says so in its own docstring — this was the
    only ``label=`` field left in the shell.
    """
    from gui.v3.app import assemble

    ctx = assemble()
    ctx.act("set_choice", "tail", True)
    ctx.render("mission", "operating")
    view = ctx.views[("mission", "operating")]

    # FIELDS only: ``label`` is an ordinary prop on a button and on an
    # expansion header, where it is the caption and floats nowhere.
    from nicegui.elements.input import Input
    from nicegui.elements.number import Number
    from nicegui.elements.select import Select

    labelled = [e for e in view.descendants()
                if isinstance(e, (Select, Input, Number))
                and (getattr(e, "_props", None) or {}).get("label")]
    assert not labelled, \
        f"a field carries a floating label: {[e._props for e in labelled]}"

    texts = [getattr(e, "text", "") or "" for e in view.descendants()]
    assert "empennage" in texts, "the question lost its name entirely"
    capsys.readouterr()


@pytest.mark.parametrize("tail_type", ["conventional", "t_tail", "v_tail",
                                       "canard"])
def test_every_empennage_can_still_be_chosen_through_the_handler(tail_type,
                                                                 capsys):
    """The select was rebuilt; the answers it sends must still land."""
    from gui.v3.app import assemble

    ctx = assemble()
    ctx.act("set_choice", "tail", True)
    ctx.act("set_choice", "tail_type", tail_type)
    ctx.render("mission", "operating")
    assert ctx.S["wing"]["choices"]["tail_type"] == tail_type
    capsys.readouterr()
