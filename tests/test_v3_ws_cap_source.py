"""A user who has DETERMINED their wing loading can say so.

The shell held two answers to one question — the W/S the user typed and the
ceiling the constraint diagram derives from the stall/landing fields — and
only the ceiling reached the solver, as a refusal. The comparison card said
so; it could not be answered. This is the answer: one switch, on stage 1.

What these tests are really about is that the switch moves the SOLVER and not
only the card. A switch that silenced the warning and left the run refusing
every design would be strictly worse than the warning it removed, so every
test here pairs "the sentence is gone" with "the number the run carries
changed with it".
"""
import pytest


def _texts(view):
    return [getattr(e, "text", "") or "" for e in view.descendants()]


def _toggle_labels(view):
    """Every label of every ``ui.toggle`` on the view. A toggle carries its
    options as a prop, not as child text, so ``_texts`` cannot see it — and a
    switch nobody can see is the thing these tests are checking for."""
    out = []
    for e in view.descendants():
        for opt in getattr(e, "_props", {}).get("options") or []:
            if isinstance(opt, dict) and "label" in opt:
                out.append(str(opt["label"]))
    return out


def _shell(medium: str = "air"):
    from gui.v3.app import assemble

    ctx = assemble(medium)
    ctx.act("accept_mission")
    return ctx


def _conflicted(ctx, ws: float = 750.0, weight: float = 7000.0):
    """The reported session: a weight raised, a stall requirement left at its
    default, and a loading the search would refuse every design of."""
    from gui.v3 import session

    ctx.S["mission"]["W_N"] = float(weight)
    assert session.set_wing_loading(ctx.S, ws)
    session.sync_wing_from_mission(ctx.S)
    assert session.ws_over_ceiling(ctx.S) is not None, (
        "this fixture is only a fixture if the two numbers do disagree")
    return ctx


# --------------------------------------------------------------- the switch
def test_the_default_is_the_mission(capsys):
    """Untouched sessions must behave exactly as they did: the diagram's
    ceiling is a REQUIREMENT and it binds."""
    from gui.v3 import session

    ctx = _conflicted(_shell())
    assert session.ws_cap_source(ctx.S) == session.WS_CAP_MISSION
    assert session.mission_ws_ceiling(ctx.S) == \
        session.diagram_ws_ceiling(ctx.S)
    capsys.readouterr()


def test_stating_it_makes_the_users_number_the_ceiling(capsys):
    from gui.v3 import session

    ctx = _conflicted(_shell(), ws=750.0)
    diagram = session.diagram_ws_ceiling(ctx.S)
    assert diagram and diagram < 750.0

    assert session.set_ws_cap_source(ctx.S, session.WS_CAP_STATED)
    assert session.mission_ws_ceiling(ctx.S) == pytest.approx(750.0)
    # ...and the derivation is untouched — the card still draws, it just
    # stops deciding
    assert session.diagram_ws_ceiling(ctx.S) == pytest.approx(diagram)
    capsys.readouterr()


def test_a_source_that_is_neither_is_refused(capsys):
    from gui.v3 import session

    ctx = _shell()
    assert not session.set_ws_cap_source(ctx.S, "whatever")
    assert session.ws_cap_source(ctx.S) == session.WS_CAP_MISSION
    # ...and a session written before the switch existed reads as the default
    ctx.S["mission"].pop("ws_cap_source", None)
    assert session.ws_cap_source(ctx.S) == session.WS_CAP_MISSION
    capsys.readouterr()


# ------------------------------------ what the RUN carries, not just the card
def test_the_flag_that_reaches_the_solver_moves_with_the_switch(capsys):
    """The load-bearing one. ``wing_loading_limit_pa`` is what refuses a
    design before its solver runs; if it kept the diagram's number the user
    would have silenced the explanation and kept the empty run."""
    from gui.v3 import config, session

    ctx = _conflicted(_shell(), ws=750.0)
    assert session.set_planform(ctx.S, "free") == []   # a family that sizes
    key = "wing_loading_limit_pa"
    before = config.cfg_dict(ctx.S)["flags"].get(key)
    assert before and before < 750.0, "the diagram's ceiling, as today"

    assert session.set_ws_cap_source(ctx.S, session.WS_CAP_STATED)
    after = config.cfg_dict(ctx.S)["flags"].get(key)
    assert after == pytest.approx(750.0)

    # the limit MOVED; it was not deleted. Every sized family can fly past
    # whatever loading it is given, and with no ceiling at all the search
    # buys payload L/D with a wing that lands at any speed it likes.
    assert key in config.cfg_dict(ctx.S)["flags"]
    capsys.readouterr()


def test_the_box_stops_being_provably_empty(capsys):
    """The whole point, end to end. On the reported mission the area row
    cannot reach the area the diagram's ceiling demands, so the box is empty
    before any evaluation — ``api.size_box_conflicts`` proves it. Owning the
    loading is one of the two honest ways out of that."""
    from aerobo import api

    from gui.v3 import config, session

    ctx = _conflicted(_shell(), ws=750.0, weight=7000.0)
    assert session.set_planform(ctx.S, "free") == []
    hits = api.size_box_conflicts(config.build_cfg(ctx.S, seed=0))
    assert any(h["kind"] == "wing loading" and h["empty"] for h in hits), (
        f"the fixture must start from an empty box: {hits}")

    assert session.set_ws_cap_source(ctx.S, session.WS_CAP_STATED)
    hits = api.size_box_conflicts(config.build_cfg(ctx.S, seed=0))
    assert not any(h["kind"] == "wing loading" and h["empty"]
                   for h in hits), hits
    capsys.readouterr()


def test_a_searched_loading_reopens_its_band(capsys):
    """The band a searched W/S opens on is centred on the ceiling AND capped
    by it — there is no interior optimum, so the band is the answer, not the
    frame. It has to follow the switch, or the run searches up to a limit the
    user has just said is not theirs."""
    from gui.v3 import session

    ctx = _conflicted(_shell(), ws=750.0)
    assert session.set_planform(ctx.S, "wing_loading_free") == []
    assert session.loading_is_searched(ctx.S)
    diagram = session.diagram_ws_ceiling(ctx.S)
    assert session.ws_band(ctx.S)[1] == pytest.approx(diagram)

    assert session.set_ws_cap_source(ctx.S, session.WS_CAP_STATED)
    assert session.ws_band(ctx.S)[1] == pytest.approx(750.0)
    capsys.readouterr()


def test_a_band_the_user_typed_is_not_overwritten(capsys):
    """...but only the band this shell wrote. A typed band is an answer of
    its own, and re-opening it would be the shell deleting a number the user
    gave — the failure this repo calls "the box shown is the box searched",
    from the other side."""
    from gui.v3 import session

    ctx = _conflicted(_shell(), ws=750.0)
    assert session.set_planform(ctx.S, "wing_loading_free") == []
    ctx.S["wing"]["bounds"][session.WS_ROW] = [40.0, 60.0]
    assert session.set_ws_cap_source(ctx.S, session.WS_CAP_STATED)
    assert session.ws_band(ctx.S) == pytest.approx((40.0, 60.0))
    capsys.readouterr()


# ------------------------------------------------------------- what is drawn
def test_the_card_stops_saying_it_and_the_switch_stays(capsys):
    """What the user asked for: the sentence goes. And the control that made
    it go must stay on screen — a switch reachable only from the warning it
    removes cannot be switched back."""
    from gui.v3 import session

    ctx = _conflicted(_shell(), ws=750.0)
    ctx.render("mission", "operating")
    view = ctx.views[("mission", "operating")]
    assert any("You have stated W/S" in t for t in _texts(view))
    assert "The W/S I typed" in _toggle_labels(view), "the switch is missing"

    assert session.set_ws_cap_source(ctx.S, session.WS_CAP_STATED)
    ctx.render("mission", "operating")
    view = ctx.views[("mission", "operating")]
    texts = _texts(view)
    assert not any("You have stated W/S" in t for t in texts), texts[-8:]
    assert "The W/S I typed" in _toggle_labels(view), "no way back"
    assert any("Your number is the ceiling" in t for t in texts), texts[-8:]
    capsys.readouterr()


def test_the_diagram_says_it_is_not_deciding(capsys):
    """The constraint diagram still draws — it is how a user prices their own
    number — so it has to say it is binding nothing, or it reads as the limit
    the solver applies."""
    from gui.v3 import session

    ctx = _conflicted(_shell(), ws=750.0)
    assert session.set_ws_cap_source(ctx.S, session.WS_CAP_STATED)
    ctx.render("mission", "operating")
    texts = _texts(ctx.views[("mission", "operating")])
    assert any("nothing on this card reaches the solver" in t.lower()
               for t in texts), texts[-10:]
    capsys.readouterr()


def test_clicking_it_changes_what_the_run_carries(capsys):
    """Through the CONTROL, not the helper behind it. Every other test here
    calls ``set_ws_cap_source``; this one finds the toggle the user actually
    clicks and sets it, so the stage's own handler is on the path."""
    from gui.v3 import config, session

    ctx = _conflicted(_shell(), ws=750.0)
    assert session.set_planform(ctx.S, "free") == []
    ctx.render("mission", "operating")
    view = ctx.views[("mission", "operating")]
    key = "wing_loading_limit_pa"
    assert config.cfg_dict(ctx.S)["flags"][key] < 750.0

    toggles = [e for e in view.descendants()
               if "The W/S I typed" in [
                   o.get("label")
                   for o in (getattr(e, "_props", {}).get("options") or [])
                   if isinstance(o, dict)]]
    assert len(toggles) == 1, "one question, one control"
    toggles[0].set_value(session.WS_CAP_STATED)

    assert session.ws_cap_source(ctx.S) == session.WS_CAP_STATED
    assert config.cfg_dict(ctx.S)["flags"][key] == pytest.approx(750.0)
    capsys.readouterr()


def test_the_empty_box_card_points_at_the_switch(capsys):
    """The user meets this failure at stage 3, and the way out is at stage 1.
    A pointer, not a second control: a question answerable in two places is
    answered twice."""
    from gui.v3 import session

    ctx = _conflicted(_shell(), ws=750.0, weight=7000.0)
    assert session.set_planform(ctx.S, "free") == []
    ctx.render("wing", "box")
    texts = _texts(ctx.views[("wing", "box")])
    assert any("The W/S I typed" in t for t in texts), texts[-6:]

    # ...and it stops pointing once the answer has been given
    assert session.set_ws_cap_source(ctx.S, session.WS_CAP_STATED)
    ctx.render("wing", "box")
    assert not any("The W/S I typed" in t
                   for t in _texts(ctx.views[("wing", "box")]))
    capsys.readouterr()


def test_switching_back_restores_the_requirement(capsys):
    from gui.v3 import config, session

    ctx = _conflicted(_shell(), ws=750.0)
    assert session.set_planform(ctx.S, "free") == []
    diagram = session.diagram_ws_ceiling(ctx.S)
    assert session.set_ws_cap_source(ctx.S, session.WS_CAP_STATED)
    assert session.set_ws_cap_source(ctx.S, session.WS_CAP_MISSION)
    assert session.mission_ws_ceiling(ctx.S) == pytest.approx(diagram)
    assert config.cfg_dict(ctx.S)["flags"]["wing_loading_limit_pa"] == \
        pytest.approx(diagram)
    assert session.ws_over_ceiling(ctx.S) is not None
    capsys.readouterr()
