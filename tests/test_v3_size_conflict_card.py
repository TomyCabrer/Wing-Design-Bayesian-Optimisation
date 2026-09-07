"""The wing stage must say a box is empty BEFORE the run, not after it.

The reported sequence: the mission's weight was raised, the wing rows were
not, and three searches came back "no solution was found" in 12-21 s each —
0 feasible of 72, 69 and 56 evaluations. Nothing was wrong with the search:
at 7000 N against the 75.2 Pa ceiling off the mission's own constraint
diagram the wing has to be at least 93 m^2, and the area row stopped at 22.

``api.size_box_conflicts`` decides that in closed form (tests/
test_size_box_conflicts.py holds the arithmetic against the physics). These
tests hold the SHELL's half of it: the sentence has to reach the two views a
user is looking at — the one with the Launch button and the one with the row
to move — and it has to go away when the row is moved.
"""
import pytest


def _texts(view):
    return [getattr(e, "text", "") or "" for e in view.descendants()]


def _heavy_shell():
    """A session whose mission the size rows cannot meet — the reported one."""
    from gui.v3 import session
    from gui.v3.app import assemble

    ctx = assemble()
    ctx.act("accept_mission")
    ctx.S["mission"]["W_N"] = 7000.0      # the weight, raised; the stall
    session.sync_wing_from_mission(ctx.S)  # requirement behind the ceiling, not
    assert session.set_planform(ctx.S, "free") == []
    return ctx


@pytest.mark.parametrize("view", ["solver", "box"])
def test_v3_an_empty_box_is_named_on_the_view_that_can_act_on_it(view, capsys):
    """Both views, because they are two different moments: the solver view is
    where the user is about to spend 72 evaluations, and the box view is where
    the row that empties it can be widened."""
    from gui.v3 import session

    ctx = _heavy_shell()
    ctx.render("wing", view)
    texts = _texts(ctx.views[("wing", view)])
    assert any("refused before its solver" in t for t in texts), texts[-12:]
    # DERIVED, never pinned: a literal here would go stale the first time the
    # ceiling or the mission moved, and would still pass while saying the
    # wrong number on screen
    needed = ctx.S["mission"]["W_N"] / session.mission_ws_ceiling(ctx.S)
    assert any(f"{needed:.4g}" in t for t in texts), (
        "the sentence must carry the area the mission needs", needed)
    # ...and the way out, as a control rather than as advice
    assert any(t.startswith("set S_m2 to") for t in texts), texts[-12:]
    capsys.readouterr()


def test_v3_taking_the_offer_clears_the_warning(capsys):
    """The band the card offers, written where the card's own button writes
    it. If the offer did not resolve the conflict the warning would still be
    there — which is the failure mode a label-only test cannot see."""
    from aerobo import api
    from gui.v3 import config

    ctx = _heavy_shell()
    band = api.size_box_conflicts(config.build_cfg(ctx.S))[0]["suggest"]
    ctx.S["wing"]["bounds"]["S_m2"] = [float(band[0]), float(band[1])]
    ctx.render("wing", "solver")
    texts = _texts(ctx.views[("wing", "solver")])
    assert not any("refused before its solver" in t for t in texts)
    assert api.size_box_conflicts(config.build_cfg(ctx.S)) == [] or all(
        not f["empty"]
        for f in api.size_box_conflicts(config.build_cfg(ctx.S)))
    capsys.readouterr()


def test_v3_the_published_mission_gets_no_such_card(capsys):
    """An untouched session must not be warned about anything: this card is a
    refusal to run a box that cannot answer, and a card that appeared on every
    run would be read as decoration."""
    from gui.v3 import session
    from gui.v3.app import assemble

    ctx = assemble()
    ctx.act("accept_mission")
    assert session.set_planform(ctx.S, "free") == []
    ctx.render("wing", "solver")
    texts = _texts(ctx.views[("wing", "solver")])
    assert not any("refused before its solver" in t for t in texts)
    capsys.readouterr()
