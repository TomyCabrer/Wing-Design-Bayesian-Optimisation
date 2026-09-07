"""The shell chrome must not be a second implementation of the stages.

Three defects of one shape — the menu bar and the two read-only panels doing
their own version of what a stage already owns:

* **Edit ▸ Reset design box** cleared three keys while stage 3's own button
  clears seven. Because it cleared ``W["fixed"]`` without clearing the
  automatic recommendation's stamp, ``_auto_wanted()`` went True and the very
  next 0.5 s heartbeat re-measured and wrote the bands back: the log said
  "design box reset" and half a second later the measured rows were live
  again. And it never called ``write_size_bands``, so the span row went back
  to the family's PUBLISHED band for every mission.
* **File ▸ New session** guarded the run manager and the reach but not the
  two section workers per surface, although the exact expression is what the
  toolbar's Stop and the enabled-check already use. The sub-dicts are refilled
  IN PLACE, so an orphaned XFOIL search kept writing its score into the fresh
  session's optimise tab.
* **The status bar and the Properties grid** quoted the raw session fields. A
  fresh air session opens in "recommended", where the run is built from
  ``session.effective_wing_search`` — so the two panels a user reads before
  pressing Run said "budget 40" against a flown 29, and after an
  own → ga → recommended round trip named a solver the run would not use.

Every assertion below is on the OUTCOME — the bounds
``config.effective_bounds`` reports, the rows ``_properties`` returns, the
labels the status bar holds — never on which function was called.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _open(medium: str = "air", planform: str = "free"):
    from gui.v3.app import assemble

    ctx = assemble(medium)
    ctx.act("accept_mission")
    if planform:
        # the free planform is what puts the two SIZE rows (``b_m``, ``S_m2``)
        # in the box, and the span is the row a reset that skipped
        # ``write_size_bands`` handed back the family's published band for
        ctx.act("set_planform", planform)
    return ctx


def _status_cells(ctx) -> list[str]:
    return [getattr(e, "text", "") or ""
            for e in ctx.statusbar.cell_row.descendants()]


# ------------------------------------------------- D6: one button, one reset

def test_the_edit_menu_reset_is_the_stage_reset(capsys):
    """Both entry points must leave the same box behind, not merely a box."""
    from gui.v3 import app, config

    by_menu = _open()
    by_menu.act("auto_recommend")
    app._reset_box(by_menu)

    by_stage = _open()
    by_stage.act("auto_recommend")
    by_stage.act("reset_box")

    assert config.effective_bounds(by_menu.S) == \
        config.effective_bounds(by_stage.S)
    # ...and the keys the menu copy used to leave behind, which are what made
    # the two boxes diverge half a second after the button was pressed
    assert by_menu.S["wing"]["recommend"] == by_stage.S["wing"]["recommend"]
    assert by_menu.S["wing"]["bounds_source"] == \
        by_stage.S["wing"]["bounds_source"]
    assert by_menu.S["wing"]["auto_rec"]["off"] is True
    assert "Traceback" not in capsys.readouterr().err


def test_the_menu_reset_survives_the_heartbeat_and_keeps_the_mission(capsys):
    """The two consequences the missing keys had, asserted separately.

    A repaint is what the heartbeat does, so rendering the box twice is the
    measured reproduction of "half a second later the rows were back".
    """
    import pytest

    from gui.v3 import app, config, session

    ctx = _open()
    ctx.act("auto_recommend")
    assert config.effective_bounds(ctx.S)["b_m"][1] == "recommended"

    app._reset_box(ctx)
    after = dict(config.effective_bounds(ctx.S))
    ctx.render("wing", "box")
    ctx.render("wing", "box")
    assert config.effective_bounds(ctx.S) == after
    assert all(source != "recommended"
               for _row, source in config.effective_bounds(ctx.S).values())

    # and the span is this MISSION's band, not the family's published row —
    # "the sized box never heard the mission" is the defect a reset that
    # skipped ``write_size_bands`` re-opened
    assert ctx.S["wing"]["bounds"].get("b_m") == pytest.approx(
        list(session.size_band_defaults(ctx.S)["b_m"]))
    assert "Traceback" not in capsys.readouterr().err


# ------------------------------- D7: a new session cannot land on a live run

def test_new_session_refuses_while_a_section_search_runs(capsys):
    """The fresh session must not be built under the two airfoil workers."""
    from gui.v3 import app, session

    for surface, lane in (("main", "screen"), ("main", "opt"),
                          ("aft", "screen"), ("aft", "opt")):
        ctx = _open()
        ctx.S["wing"]["seed"] = 4242                 # a mark the reset erases
        state = session.airfoil_state(ctx.S, surface)
        held = state[lane]                           # what the worker captured
        held["running"] = True

        app._new_session(ctx)

        assert ctx.S["wing"]["seed"] == 4242, (surface, lane)
        assert state[lane] is held and held["running"] is True
        held["running"] = False
    assert "Traceback" not in capsys.readouterr().err


def test_new_session_still_resets_when_nothing_is_running(capsys):
    """The guard must be a guard, not a ban."""
    from gui.v3 import app

    ctx = _open()
    ctx.S["wing"]["seed"] = 4242
    app._new_session(ctx)
    assert ctx.S["wing"]["seed"] != 4242
    assert ctx.S["ui"]["selected"] == "mission"
    assert "Traceback" not in capsys.readouterr().err


# ------------------- D15: the panels quote the search that runs, not the dict

def test_the_status_bar_quotes_the_budget_that_will_be_spent(capsys):
    from gui.v3 import config, session

    ctx = _open()
    eff = session.effective_wing_search(ctx.S)
    assert eff["source"] == "recommended"
    # the defect is only visible because the two disagree on a fresh session
    assert int(eff["budget"]) != int(ctx.S["wing"]["budget"])
    assert int(eff["budget"]) == int(config.cfg_dict(ctx.S)["budget"])

    ctx.refresh()
    assert f"budget {int(eff['budget'])}" in _status_cells(ctx)
    assert f"budget {int(ctx.S['wing']['budget'])}" not in _status_cells(ctx)
    assert "Traceback" not in capsys.readouterr().err


def test_the_properties_grid_names_the_optimiser_that_will_fly(capsys):
    """"own → pick ga → back to recommended" leaves ``ga`` in the field."""
    from gui.v3 import app, config, session

    from aerobo import api

    ctx = _open()
    ctx.S["ui"]["selected"] = "wing"
    ctx.S["wing"]["optimiser"] = "ga"
    ctx.S["wing"]["budget"] = 120
    # the flown numbers are re-derived, never pinned: the recommendation is a
    # function of the installed study and of this stage's design vector
    eff = session.effective_wing_search(ctx.S)
    cfg = config.cfg_dict(ctx.S)
    assert (cfg["optimiser"], int(cfg["budget"])) == \
        (eff["optimiser"], int(eff["budget"]))
    assert eff["optimiser"] != "ga" and int(eff["budget"]) != 120

    rows = {r[0]: r[1] for r in app._properties(ctx) if r[0] != "group"}
    assert rows["optimiser"] == \
        api.OPTIMISER_SPECS[eff["optimiser"]].display.split(" — ")[0]
    assert rows["budget"] == f"{int(eff['budget'])} evaluations"
    assert "Traceback" not in capsys.readouterr().err
