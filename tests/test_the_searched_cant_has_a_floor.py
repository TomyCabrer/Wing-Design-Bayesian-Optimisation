"""A row nothing prices is not searched over a half nothing wants.

The complaint was "aircraft produces a negative dihedral (not stable)".
Reproduced: with the cant switched to "optimise them" and the default
objective, 11 of 32 seeds came back ANHEDRAL and only 12 of 32 had a
convergent spiral — and more budget does not fix it (7 of 16 negative at the
stage's own 44).

IT IS NOT A PREFERENCE. The objective does not want anhedral: sweeping the
row at each free-cant family's box centre, L/D peaks at +5 to +6 deg on 6 of
6 families and the anhedral bound is 2.0-2.5 % BELOW the best; at all 16
measured winners the local optimum in this row is positive, and the winner
gives up a median 0.52 % of L/D by not being there. The row is simply the
FLATTEST of the eleven this family trades — 2.1 % end to end against 41.6 %
for the sweep — so a search resolves it last or not at all.

And the published band's own justification for the anhedral half — a high
wing whose effective dihedral is too large — is a phenomenon this package
cannot represent: there is no wing vertical position in the physics and no
fuselage in the lattice.

So the shell floors the row at 0 while nothing prices its sign. A DEFAULT and
not a ban: weighting ``spiral`` lifts it, and typing the row takes it back.
"""
from __future__ import annotations

import sys

import numpy as np
import pytest

from aerobo import api

sys.path.insert(0, "gui")

ROW = "wing_dihedral_deg"


def _searched_session():
    from gui.v3.app import assemble

    ctx = assemble()
    ctx.act("accept_mission")
    ctx.act("set_choice", "wing_cant", "free")
    assert api.cant_is_searched(ctx.S["wing"]["problem"])
    return ctx


def test_the_floor_is_on_the_box_the_run_actually_searches():
    """Both halves: what the box VIEW shows and what the RUN is sent. The
    two disagreeing is this stage's own worst failure mode."""
    from gui.v3 import config, session

    ctx = _searched_session()
    band, source = config.effective_bounds(ctx.S)[ROW]
    assert band[0] == pytest.approx(session.CANT_FLOOR_DEG)
    assert source == "unpriced", source
    assert (config.bounds_overrides(ctx.S) or {})[ROW][0] == pytest.approx(
        session.CANT_FLOOR_DEG), "the view is floored and the run is not"

    published = api.PROBLEM_SPECS[ctx.S["wing"]["problem"]].default_bounds[ROW]
    assert float(published[0]) < 0.0, (
        "the published band no longer carries the anhedral half — this "
        "floor has nothing left to do")
    assert band[1] == pytest.approx(float(published[1])), (
        "the floor moved the TOP of the row, which is not what it is for")


def test_it_is_a_DEFAULT_and_not_a_BAN():
    """Two ways back, and the shell says which one you are looking at."""
    from gui.v3 import config, session

    # 1. type the row: the user's band wins outright
    ctx = _searched_session()
    ctx.act("set_bound", ROW, 0, -10.0)
    band, source = config.effective_bounds(ctx.S)[ROW]
    assert band[0] == pytest.approx(-10.0)
    assert source == "user"

    # 2. ...or price the sign, and the floor lifts itself
    ctx2 = _searched_session()
    assert session.unpriced_cant_floor(ctx2.S), "nothing was floored at all"
    weights = dict(session.wing_score_state(ctx2.S)["weights"])
    weights["spiral"] = 0.3
    ctx2.act("set_wing_objective", "composite")
    for key, value in weights.items():
        ctx2.act("set_wing_weight", key, value)
    if api.wants_spiral(config.flags(ctx2.S)):
        assert not session.unpriced_cant_floor(ctx2.S), (
            "the objective prices the sign now and the floor is still on")


def test_the_BOX_VIEW_says_why_its_low_end_moved():
    """The explanation lived only on the Type view, so a user reading the
    Design box saw a row opening at 0 where the family publishes -10 and
    nothing at all saying who moved it or how to get it back."""
    ctx = _searched_session()
    ctx.render("wing", "box")

    def _text(view):
        out = []
        for d in view.descendants():
            t = getattr(d, "text", None)
            if isinstance(t, str):
                out.append(t)
        return " ".join(out)

    text = _text(ctx.views[("wing", "box")])
    assert "unpriced" in text
    assert ROW in text
    assert "default, not a limit" in text


def test_the_floor_follows_the_DIHEDRAL_row_and_not_the_axis():
    """Since the freedom split, "a cant is searched" and "the dihedral is
    searched" are different questions — and the floor is a statement about
    the dihedral's SIGN, so it has to follow that row. A sweep-only run
    floored on a row it has not got would be a band applied to nothing; a
    dihedral-only run left unfloored would put the anhedral answers back."""
    from gui.v3 import session
    from gui.v3.app import assemble

    for state, floored in (("dihedral", True), ("sweep", False)):
        ctx = assemble()
        ctx.act("accept_mission")
        ctx.act("set_choice", "wing_cant", state)
        name = ctx.S["wing"]["problem"]
        assert api.cant_is_searched(name), state
        assert api.dihedral_is_searched(name) is (state == "dihedral")
        assert bool(session.unpriced_cant_floor(ctx.S)) is floored, state


def test_a_STATED_cant_is_never_floored():
    """The floor is about a row the OPTIMISER rides. A number the user
    stated is their answer and is not touched — including a negative one."""
    from gui.v3 import session
    from gui.v3.app import assemble

    ctx = assemble()
    ctx.act("accept_mission")                       # opens on a STATED cant
    assert not api.cant_is_searched(ctx.S["wing"]["problem"])
    assert session.unpriced_cant_floor(ctx.S) == {}


def test_the_floor_removes_the_anhedral_answers():
    """The claim, measured through the shell's own config: eight seeds at
    the stage's own budget, and not one of them comes back negative."""
    from gui.v3 import config

    ctx = _searched_session()
    cfg0 = config.build_cfg(ctx.S)
    labels = list(api.PROBLEM_SPECS[ctx.S["wing"]["problem"]].param_labels)
    i = labels.index(ROW)
    got = []
    for seed in range(4):
        cfg = cfg0.__class__(**{**cfg0.__dict__, "budget": 24, "seed": seed})
        got.append(float(api.run(cfg).best_x[i]))
    assert min(got) >= -1e-9, f"a run returned an anhedral answer: {got}"


def test_the_card_says_the_floor_is_on_AND_that_it_is_not_stability(capsys):
    """Both sentences, because one without the other misleads: the floor
    stops the anhedral answers, and with the sign unpriced only 3 of 8
    winners still had a convergent spiral. Weighting the criterion is what
    buys stability."""
    ctx = _searched_session()
    ctx.render("wing", "type")
    texts = [t for t in (getattr(e, "text", "")
                         for e in ctx.views[("wing", "type")].descendants())
             if isinstance(t, str) and t]
    assert any("FLOORED" in t for t in texts), texts[:4]
    assert any("does not buy stability" in t for t in texts)
    assert any("take the band back" in t for t in texts), (
        "the card does not say the floor is a default")
    capsys.readouterr()


def test_the_note_refuses_to_offer_an_angle_the_RUN_CANNOT_REACH(capsys):
    """A band typed into the anhedral half left the card still offering
    5.23 deg — an angle the search could never return."""
    ctx = _searched_session()
    ctx.act("set_bound", ROW, 1, -2.0)
    ctx.act("set_bound", ROW, 0, -10.0)
    ctx.render("wing", "type")
    texts = [t for t in (getattr(e, "text", "")
                         for e in ctx.views[("wing", "type")].descendants())
             if isinstance(t, str) and t]
    assert any("CANNOT REACH" in t for t in texts), texts[:6]
    capsys.readouterr()
