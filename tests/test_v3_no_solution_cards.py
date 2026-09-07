"""What the shell says when a run comes back with no design.

Three sentences the user asked for, each on the view that can act on it:

* stage 1 — the wing loading you STATED against the one this mission ALLOWS,
  because both are already in that form and only one of them travelled to the
  solver, as a refusal;
* stage 3 — the design that came CLOSEST, offered as the next run's starting
  point rather than quoted at them;
* stage 3 — which design-box ROW is the limiting factor.

Rendered, not unit-tested through the helpers: the failure these exist for is
a sentence that was never on screen.
"""
import numpy as np


def _texts(view):
    return [getattr(e, "text", "") or "" for e in view.descendants()]


def _shell(medium: str = "air"):
    from gui.v3.app import assemble

    ctx = assemble(medium)
    ctx.act("accept_mission")
    return ctx


# ------------------------------------------- stage 1: stated vs allowed
def test_v3_a_stated_wing_loading_over_the_ceiling_is_named_at_stage_1(capsys):
    """The reported session: a weight raised, a stall requirement left alone,
    and a loading the search would refuse every design of."""
    from gui.v3 import session

    ctx = _shell()
    assert session.ws_over_ceiling(ctx.S) is None, (
        "the published mission must not be warned about anything")
    ctx.S["mission"]["W_N"] = 7000.0
    assert session.set_wing_loading(ctx.S, 318.0)
    session.sync_wing_from_mission(ctx.S)

    c = session.ws_over_ceiling(ctx.S)
    assert c and c["stated"] > c["cap"]
    ctx.render("mission", "operating")
    texts = _texts(ctx.views[("mission", "operating")])
    assert any("You have stated W/S" in t for t in texts), texts[-8:]
    assert any(f"{c['cap']:.0f} N/m²" in t for t in texts)
    assert any(t.startswith("Use ") for t in texts)
    capsys.readouterr()


def test_v3_taking_the_ceiling_resolves_it(capsys):
    """The button has to fix the thing it appears next to. It adopts the
    CEILING, not the diagram's design point — that is None on a diagram which
    does not close, which is the state this card fires in."""
    from gui.v3 import session

    ctx = _shell()
    ctx.S["mission"]["W_N"] = 7000.0
    session.set_wing_loading(ctx.S, 318.0)
    session.sync_wing_from_mission(ctx.S)
    cap = session.ws_over_ceiling(ctx.S)["cap"]

    assert session.set_wing_loading(ctx.S, cap)
    session.sync_wing_from_mission(ctx.S)
    assert session.ws_over_ceiling(ctx.S) is None
    # ...and the area that implies is the one the mission needs
    assert float(ctx.S["mission"]["s_ref_m2"]) == 7000.0 / cap
    ctx.render("mission", "operating")
    assert not any("You have stated W/S" in t
                   for t in _texts(ctx.views[("mission", "operating")]))
    capsys.readouterr()


# --------------------------------- stage 3: the design that came closest
def _record_with_a_near_miss(ctx):
    """A record from a run that solved designs and missed a limit on all of
    them — built through ``api.partial_result``, the same call the runner
    makes, so the card is read off the record the shell really stores."""
    from aerobo import api

    from gui.v3 import config

    cfg = config.build_cfg(ctx.S, seed=0)
    dim = len(api.PROBLEM_SPECS[cfg.problem_name].param_labels)
    rng = np.random.default_rng(0)
    rows = []
    for i in range(6):
        # every margin strictly NEGATIVE — a run with an incumbent is not the
        # state this card is for, and a margin that reaches 0.0 is an
        # incumbent (the feasibility rule is >= 0, not > 0)
        rows.append({"x": [float(v) for v in rng.uniform(size=dim)],
                     "f": 10.0 + i, "g": [-0.5 + 0.05 * i],
                     "feasible": False})
    out = api.partial_result(cfg, rows).to_dict()
    assert out["best_x"] is None and out["best_score"] is None
    return out


def _finished_run(ctx, record):
    """Put the shell in the state a FINISHED run leaves it in: the record on
    the session and a job on the manager, exactly as the smoke test's own
    cancelled-run fixture does. The card lives on the Convergence view, which
    draws nothing at all without a job — so a test that only set the record
    would pass on a page the user never sees."""
    from gui import nice_app as v1
    from gui.v3 import config

    ctx.S["run"]["record"] = record
    job = v1.RunJob(cfg=config.build_cfg(ctx.S, seed=0),
                    label="probe · seed 0", budget=40)
    job.status = "done"
    ctx.manager.jobs = [job]
    ctx.manager.version += 1
    return ctx


def test_v3_the_closest_design_is_offered_as_a_starting_point(capsys):
    from gui.v3 import session

    ctx = _shell()
    assert session.set_planform(ctx.S, "free") == []
    _finished_run(ctx, _record_with_a_near_miss(ctx))
    ctx.render("wing", "run")
    texts = _texts(ctx.views[("wing", "run")])
    assert any("Start again from the closest design" in t for t in texts), \
        texts[-10:]
    # the number it quotes is the miss the record itself carries: the last row
    # is the least-violating one (-0.5 + 0.5 = 0.0 is not reached; -0.0…)
    assert any("short of" in t for t in texts)
    capsys.readouterr()


def test_v3_a_run_whose_every_draw_was_refused_offers_no_design(capsys):
    """A refusal is not a wing. A box where nothing solved has a nearest
    REFUSAL and no nearest design, and offering one would be offering a design
    nobody flew."""
    from aerobo import api

    from gui.v3 import config, session

    ctx = _shell()
    assert session.set_planform(ctx.S, "free") == []
    cfg = config.build_cfg(ctx.S, seed=0)
    dim = len(api.PROBLEM_SPECS[cfg.problem_name].param_labels)
    rows = [{"x": [0.5] * dim, "f": -100.0, "g": [-1.0], "feasible": False}
            for _ in range(4)]
    _finished_run(ctx, api.partial_result(cfg, rows).to_dict())
    ctx.render("wing", "run")
    texts = _texts(ctx.views[("wing", "run")])
    assert not any("Start again from the closest design" in t for t in texts)
    capsys.readouterr()


# ------------------------- the recommendation, pressed where it is offered
#
# Reported: "it gives a recommendation but it doesn't work. Gives multiple
# recommendations but only for area. And doesn't solve the problem." Two of
# those three were the shell's, and both were invisible to a test that
# simulated the press by writing the band itself.
#
# ``_widen_row`` repainted the DESIGN BOX and the solver view. This card is
# also drawn on the Convergence view, by ``_render_no_feasible``, and
# ``ctx.refresh`` paints the shell chrome and no view at all — so a press made
# where the run failed moved the row and left the identical warning and the
# identical button on screen. Nothing the user could see changed.
#
# And two gates naming the same row drew the same button twice, which is the
# "multiple recommendations" half.

def _empty_box_shell():
    """The reported session: a weight raised, its landing requirement not, and
    a free planform whose area row cannot meet the ceiling that implies."""
    from gui.v3 import session

    ctx = _shell()
    ctx.S["mission"]["W_N"] = 7000.0
    session.sync_wing_from_mission(ctx.S)
    assert session.set_planform(ctx.S, "free") == []
    return ctx


def _offers(view):
    return [e for e in view.descendants()
            if type(e).__name__ == "Button"
            and (getattr(e, "text", "") or "").startswith("set ")]


def _press(button):
    """Fire the button's own click handler — not what we think it does."""
    handler = next(iter(button._event_listeners.values())).handler
    handler(None)


def test_v3_pressing_the_offer_repaints_the_card_that_made_it(capsys):
    """The press has to change the view it was made on.

    The assertion is deliberately about the view's CONTENT after the handler
    ran and before anything else re-renders: a test that renders again itself
    passes on the broken shell, which is exactly how this shipped.
    """
    from aerobo import api

    from gui.v3 import config

    ctx = _empty_box_shell()
    cfg = config.build_cfg(ctx.S, seed=0)
    dim = len(api.PROBLEM_SPECS[cfg.problem_name].param_labels)
    rows = [{"x": [0.5] * dim, "f": -100.0, "g": [-1.0], "feasible": False}
            for _ in range(4)]
    _finished_run(ctx, api.partial_result(cfg, rows).to_dict())

    ctx.render("wing", "run")
    view = ctx.views[("wing", "run")]
    assert any("refused before its solver" in t for t in _texts(view))
    offers = _offers(view)
    assert offers, _texts(view)[-12:]

    _press(offers[0])
    assert not any("refused before its solver" in t for t in _texts(view)), (
        "the card that offered the press did not repaint, so the "
        "recommendation reads as one that did nothing")
    assert not _offers(view)
    capsys.readouterr()


def test_v3_one_button_per_band(capsys):
    """Two gates can want the same row moved to the same place. The second
    button does exactly what the first did, and a user who pressed it and saw
    another one reads that as a recommendation that failed."""
    ctx = _empty_box_shell()
    ctx.render("wing", "box")
    labels = [b.text for b in _offers(ctx.views[("wing", "box")])]
    assert labels, "the empty box offered no way out at all"
    assert len(labels) == len(set(labels)), labels
    capsys.readouterr()


def test_v3_a_released_row_comes_back_on_when_its_band_is_taken(capsys):
    """A band written for a RELEASED row never reaches the run
    (``config.bounds_overrides`` drops it), so the press would move nothing
    and the card would keep asking for the press it had just been given."""
    from gui.v3 import config

    ctx = _empty_box_shell()
    ctx.S["wing"]["bounds_off"] = ["S_m2"]
    ctx.render("wing", "box")
    offers = _offers(ctx.views[("wing", "box")])
    assert offers
    _press(offers[0])
    assert "S_m2" not in config.released_rows(ctx.S)
    assert "S_m2" in (config.bounds_overrides(ctx.S) or {})
    capsys.readouterr()
