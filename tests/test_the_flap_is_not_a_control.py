"""The flap is not one of the controls this shell offers.

"Eliminate flaps from control and make sure it works without it."

It was a switch that was off in every session anybody opened, and switching
it on bought a lift column the stall clip ate and a drag it never paid — so
the only thing a flap could actually do to the flown model was move the
stall to a LOWER incidence. Cutting it takes a group box off stage 5's first
tab, two rows off the Derivatives table, a slider off stage 6, and one key
off the session.

Both halves of the ask are pinned here: it is gone from every place it was
asked or shown, and the aeroplane still builds a deck, arms, trims and
flies without it.
"""
from __future__ import annotations

import pytest

from gui.v4 import app as v4app, session, stick as stk


def _armed():
    """A V4 shell with a real design behind it, as stage 5 receives it."""
    from aerobo import api

    ctx = v4app.assemble()
    cfg = api.RunConfig(problem_name="tail", budget=4, seed=0)
    built = api.PROBLEM_SPECS["tail"].build({}, {}, None)
    ctx.S["run"]["record"] = {"pretend": "a completed run"}
    ctx.S["run"]["report"] = api.design_report(cfg, built.bounds.mean(axis=1))
    return ctx


def _said(view) -> str:
    """Every word the view puts on screen: element text AND the string
    props Quasar draws labels and captions out of."""
    out = []
    for el in view.descendants():
        t = getattr(el, "text", "")
        if isinstance(t, str):
            out.append(t)
        for v in getattr(el, "_props", {}).values():
            if isinstance(v, str):
                out.append(v)
    return " ".join(out).lower()


# --------------------------------------------------------- it is not asked

def test_no_stage_asks_for_a_flap():
    """A question with no answer anywhere is a question nobody can be left
    half way through."""
    assert "flap" not in session.CONTROLS_DEFAULTS
    assert "flap" not in session.make_session()["controls"]
    assert "flap" not in session.FLIGHT_DEFAULTS["stick"]
    assert "flap" not in session.make_session()["flight"]["stick"]


def test_it_is_not_a_stick_axis_or_a_key():
    assert "flap" not in stk.LIMITS
    assert "flap" not in {a for a, _d in stk.KEYMAP.values()}


def test_the_tab_is_not_named_after_it():
    """The first tab of stage 5 was called "Ailerons & flaps"; half of that
    name is now a control the stage does not have."""
    labels = [lbl for _k, lbl, _i in session.views_of(session.make_session(),
                                                      "controls")]
    assert not any("flap" in lbl.lower() for lbl in labels), labels
    assert labels[0] == "Ailerons & elevator"


# -------------------------------------------------------- it is not on show

def test_stage_5_shows_no_flap_control_and_still_shows_the_aileron(capsys):
    ctx = _armed()
    ctx.render("controls", "surfaces")
    assert "failed to render" not in capsys.readouterr().err
    said = _said(ctx.views[("controls", "surfaces")])
    assert "flap" not in said
    assert "fit ailerons" in said, "the surfaces view lost its own subject"
    assert "elevator" in said


def test_the_derivatives_table_drops_the_flap_rows_and_keeps_the_others(
        capsys):
    ctx = _armed()
    ctx.render("controls", "derivatives")
    assert "failed to render" not in capsys.readouterr().err
    said = _said(ctx.views[("controls", "derivatives")])
    assert "cz_df" not in said and "cm_df" not in said
    for kept in ("cl_da", "cn_da", "cm_de", "cn_dr", "cy_dr"):
        assert kept in said, kept


def test_stage_6_offers_three_stick_axes(capsys):
    """The fourth slider was drawn only when the deck carried a flap column,
    so the way to see it is gone is that the three that remain are all
    there."""
    ctx = _armed()
    ctx.render("controls", "derivatives")
    ctx.act("flight_arm")
    ctx.render("flight", "fly")
    assert "failed to render" not in capsys.readouterr().err
    said = _said(ctx.views[("flight", "fly")])
    assert "flap" not in said
    for axis in ("elevator (+ = nose down)", "aileron (+ = roll to port)",
                 "rudder (+ = nose to starboard)"):
        assert axis in said, axis


# ------------------------------------------------------ it still works

def test_the_deck_it_builds_has_no_flap_column_and_the_rest_of_it(capsys):
    ctx = _armed()
    ctx.render("controls", "derivatives")
    capsys.readouterr()
    cols = ctx.S["controls"]["deck"].columns
    assert "flap" not in cols
    for kept in ("alpha", "beta", "aileron", "elevator", "rudder"):
        assert kept in cols, kept


def test_it_arms_trims_and_flies(capsys):
    """The whole point of the ask's second half: a design with no flap is
    still a design that flies. One second of wall clock, one second of
    flight, and it is still in the air at the end of it."""
    ctx = _armed()
    ctx.render("controls", "derivatives")
    assert ctx.act("flight_arm") is True, ctx.S["flight"].get("error")
    F = ctx.S["flight"]
    assert F["trim"]["alpha_deg"] == pytest.approx(
        F["trim"]["alpha_deg"])                     # it solved at all
    assert "flap_deg" not in F["trim"]
    ctx.act("flight_run", True)
    for _ in range(60):
        ctx.act("flight_advance", 1.0 / 60.0)
    capsys.readouterr()
    assert F["t"] == pytest.approx(1.0, abs=2 / 240)
    assert F.get("crashed") in (None, False), F.get("crashed")


def test_every_manoeuvre_is_still_offered(capsys):
    """Nothing was scripted into the flap axis, so cutting it must not cost
    a manoeuvre."""
    from gui.v4 import manoeuvre as mv

    ctx = _armed()
    ctx.render("controls", "derivatives")
    capsys.readouterr()
    cols = set(ctx.S["controls"]["deck"].columns)
    assert {m.key for m in mv.MANOEUVRES if m.flyable_on(cols)} == \
        {m.key for m in mv.MANOEUVRES}
