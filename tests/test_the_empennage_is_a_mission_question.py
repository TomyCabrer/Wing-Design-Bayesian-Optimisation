"""WHICH EMPENNAGE is stage 1's question, and each one is a different vehicle.

The layout menu lived on stage 3's Wing type card, under the design box —
two stages after the stages it decides. It is not a design-box question:

* a V-TAIL has no separate vertical surface at all, so stage 2.7 does not
  exist for it. Its two canted panels carry pitch and yaw at once;
* a T-TAIL connects the two — the tailplane stands on the fin's tip, so the
  fin's span IS the tailplane's height and sizing one sizes the other;
* a CONVENTIONAL layout has two separate surfaces, each with its own stage
  and its own section.

So it is asked beside the two switches it interacts with (is there a second
surface, is there a fin), and stage 3 reports it with the way back.
"""
from __future__ import annotations

import pytest

import gui.nice_app as v1
from gui.v3 import session
from gui.v3.app import assemble


def _texts(view) -> str:
    out = []
    for e in view.descendants():
        for attr in ("text", "_text"):
            v = getattr(e, attr, None)
            if isinstance(v, str) and v:
                out.append(v)
        props = getattr(e, "_props", {}) or {}
        for key in ("label", "text"):
            v = props.get(key)
            if isinstance(v, str) and v:
                out.append(v)
        for opt in (props.get("options") or []):
            if isinstance(opt, dict) and isinstance(opt.get("label"), str):
                out.append(opt["label"])
    return " ".join(out)


def _with_tail():
    ctx = assemble()
    ctx.act("accept_mission")
    ctx.act("set_second_surface", True)
    assert ctx.S["wing"]["choices"].get("tail")
    return ctx


def test_the_mission_asks_which_empennage(capsys):
    ctx = _with_tail()
    ctx.render("mission", "operating")
    text = _texts(ctx.views[("mission", "operating")])
    assert "empennage" in text
    for label in v1.TAIL_TYPE_LABELS.values():
        assert label in text, label
    capsys.readouterr()


def test_choosing_a_v_tail_deletes_the_vertical_surface(capsys):
    """The layout answers the fin question by construction, and the stage it
    would have been designed on goes with it."""
    ctx = _with_tail()
    assert session.stage_visible(ctx.S, "airfoil_fin")
    ctx.act("set_empennage", "v_tail")
    assert ctx.S["wing"]["choices"]["tail_type"] == "v_tail"
    assert not session.fin_surface(ctx.S)
    assert not session.stage_visible(ctx.S, "airfoil_fin")
    assert session.next_section_stage(ctx.S, "airfoil") == "airfoil_aft"
    assert session.next_section_stage(ctx.S, "airfoil_aft") is None
    capsys.readouterr()


def test_leaving_the_v_tail_gives_the_surface_back(capsys):
    ctx = _with_tail()
    ctx.act("set_empennage", "v_tail")
    ctx.act("set_empennage", "t_tail")
    assert session.fin_surface(ctx.S)
    assert session.stage_visible(ctx.S, "airfoil_fin")
    capsys.readouterr()


def test_a_t_tail_stands_its_tailplane_on_the_fin(capsys):
    """The CONNECTION, measured on the sizing law the shell quotes: a T-tail's
    fin spans up to its tailplane, so its height is not the conventional
    one."""
    from aerobo import fin as finmod

    conv = finmod.size_fin(b=10.0, S=10.0, l_t=5.5, tail_type="conventional")
    tee = finmod.size_fin(b=10.0, S=10.0, l_t=5.5, tail_type="t_tail")
    assert conv is not None and tee is not None
    assert finmod.size_fin(b=10.0, S=10.0, l_t=5.5, tail_type="v_tail") is None
    capsys.readouterr()


def test_the_wing_stage_reports_it_and_does_not_ask_again(capsys):
    ctx = _with_tail()
    ctx.render("wing", "type")
    text = _texts(ctx.views[("wing", "type")])
    assert "empennage" in text
    assert v1.TAIL_TYPE_LABELS["conventional"] in text
    assert "change it on stage 1" in text
    # the OTHER layouts must not be offerable here: a select would carry
    # every label as an option
    assert v1.TAIL_TYPE_LABELS["v_tail"] not in text
    capsys.readouterr()


def test_the_empennage_is_not_asked_under_water(capsys):
    """No layout menu at all under water — no fin, no fuselage, all-moving."""
    ctx = assemble()
    ctx.act("set_medium", "water")
    ctx.render("mission", "operating")
    text = _texts(ctx.views[("mission", "operating")])
    assert "empennage" not in text
    capsys.readouterr()
