"""The floor covers the SEARCHED row. Two other ways in were silent.

A searched ``wing_dihedral_deg`` is floored at 0 while nothing prices its
sign (``gui.v3.session.CANT_FLOOR_DEG``) — a default, not a ban. The same
number can reach the run two other ways and neither said anything:

* STATED. Typing -60 into the card's own field was reported back as
  "-60 deg — your own number, not one of the steps above", which is true
  and says nothing about what it does.
* PINNED. A pin is not a band, so it goes round the floor by construction:
  the row leaves the design vector at whatever value it holds, and the run
  flies it verbatim.

NEITHER IS REFUSED — a stated number is the user's answer, and this package
does not turn a measurement into a gate. What it must not do is take that
answer in silence. One sentence, one author (``api.anhedral_note``), in all
three places the cant can be met: the field, the pin, and the result.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aerobo import api                                          # noqa: E402

ROW = api.WING_CANT_KEYS[0]


def _text(view) -> str:
    return " ".join((getattr(e, "text", "") or "")
                    for e in view.descendants())


# --------------------------------------------------------- the sentence

def test_the_note_is_only_about_a_negative_cant():
    assert api.anhedral_note(-6.0)
    assert api.anhedral_note(-0.5)
    assert api.anhedral_note(0.0) == ""
    assert api.anhedral_note(7.0) == ""
    assert api.anhedral_note(None) == ""
    assert api.anhedral_note(float("nan")) == ""
    assert api.anhedral_note(True) == ""            # a bool is not an angle


def test_it_says_which_way_the_tips_point_and_what_that_does():
    note = api.anhedral_note(-10.0)
    assert "10.00 deg BELOW" in note
    assert "rolls INTO a sideslip" in note


def test_the_result_read_out_uses_the_same_sentence():
    """ONE author: the answer's own stage must not paraphrase it."""
    says = api.lateral_verdict({"wing_dihedral_deg": -10.0})["says"]
    assert api.anhedral_note(-10.0).rstrip(".") in says


# ------------------------------------------------------------- the field

def test_a_stated_anhedral_is_named_on_the_card():
    from gui.v3.app import assemble

    ctx = assemble("air")
    ctx.act("accept_mission")
    ctx.act("set_wing_cant", True)
    ctx.S["wing"]["flags"][ROW] = -60.0
    ctx.render("wing", "type")
    text = _text(ctx.views[("wing", "type")])
    assert "ANHEDRAL" in text, text[-800:]
    assert "60.00 deg BELOW" in text


def test_a_stated_DIHEDRAL_is_not_warned_about():
    """The control. A card that warns either way says nothing."""
    from gui.v3.app import assemble

    ctx = assemble("air")
    ctx.act("accept_mission")
    ctx.act("set_wing_cant", True)
    ctx.S["wing"]["flags"][ROW] = +6.0
    ctx.render("wing", "type")
    assert "ANHEDRAL" not in _text(ctx.views[("wing", "type")])


def test_it_is_named_and_NOT_refused():
    """The value reaches the run exactly as typed — this is a note, not a
    gate (``a-calibration-is-a-default-not-a-ban``)."""
    from gui.v3 import config
    from gui.v3.app import assemble

    ctx = assemble("air")
    ctx.act("accept_mission")
    ctx.act("set_wing_cant", True)
    ctx.S["wing"]["flags"][ROW] = -60.0
    assert config.flags(ctx.S)[ROW] == -60.0


# --------------------------------------------------------------- the pin

def test_a_pinned_anhedral_is_named_and_says_the_floor_does_not_apply():
    from gui.v3.app import assemble

    ctx = assemble("air")
    ctx.act("accept_mission")
    ctx.act("set_choice", "wing_cant", "free")
    assert api.cant_is_searched(ctx.S["wing"]["problem"])
    ctx.act("set_row_fixed", ROW, True)
    ctx.act("set_fixed_value", ROW, -8.0)
    ctx.render("wing", "box")
    text = _text(ctx.views[("wing", "box")])
    assert "ANHEDRAL" in text, text[-800:]
    assert "8.00 deg BELOW" in text
    assert "the floor that holds the SEARCHED row at 0 does not apply" in text


def test_the_pin_really_does_go_round_the_floor():
    """The premise. If a pin ever started being floored, the sentence above
    would be a lie and this file would be about nothing."""
    from gui.v3 import config, session
    from gui.v3.app import assemble

    ctx = assemble("air")
    ctx.act("accept_mission")
    ctx.act("set_choice", "wing_cant", "free")
    band, source = config.effective_bounds(ctx.S)[ROW]
    assert band[0] == session.CANT_FLOOR_DEG and source == "unpriced"
    ctx.act("set_row_fixed", ROW, True)
    ctx.act("set_fixed_value", ROW, -8.0)
    assert config.fixed_rows(ctx.S)[ROW] == -8.0


def test_a_pinned_positive_cant_draws_no_warning():
    from gui.v3.app import assemble

    ctx = assemble("air")
    ctx.act("accept_mission")
    ctx.act("set_choice", "wing_cant", "free")
    ctx.act("set_row_fixed", ROW, True)
    ctx.act("set_fixed_value", ROW, +8.0)
    ctx.render("wing", "box")
    assert "ANHEDRAL" not in _text(ctx.views[("wing", "box")])
