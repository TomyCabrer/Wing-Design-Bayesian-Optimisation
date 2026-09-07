"""V3 stage 2: the THIRD hard gate has a control, and says when it fired.

``nose_down`` refuses every REFLEXED section — the gated quantity is -cm_at,
so the floor 0 admits only cm <= 0 (``airfoil_select.NOSE_DOWN_FLOOR``). It is
on by default for a wing that has a surface to trim it, and the reason is
measured: |Cm| is a lower-better criterion, so ranking on it rewards reflex,
and reflex is what a section does INSTEAD of having a tail — flown under a
stabiliser it inverts its job.

It was a DEFAULT in the docstring only. ``nose_down_required`` read
``airfoil_state(S, surface)["nose_down"]``, no such key existed in the
workspace, and nothing in gui/ ever wrote one: the gate could not be turned
off from anywhere in the shell, and nothing on the form, in the ranking header
or in the "eligible N of M" read-out said a third gate was live. Measured on
the shipped tailed-wing screen at the library point, on this stage's own
``wing-trimmed`` weights: 165 sections clear the two gates on the form, 125
clear all three — 40 sections leave the ranking silently.

What is gated here is the outcome, not the wiring: the floors dict the screen
is actually RUN with, the sentence in the ranking header, and the fact that
the shipped default answer is unchanged (mrc-20 wins either way).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _texts(view):
    return [getattr(e, "text", "") or "" for e in view.descendants()]


def _tailed_shell():
    """The shell, on the one configuration the gate is a question for."""
    from gui.v3.app import assemble

    ctx = assemble()
    ctx.act("accept_mission")
    ctx.act("set_second_surface", True)
    return ctx


def _wait(ctx, stage="airfoil", timeout=60.0):
    t0 = time.time()
    while ctx.S[stage]["screen"]["running"] and time.time() - t0 < timeout:
        time.sleep(0.02)
    assert not ctx.S[stage]["screen"]["running"], "screen never finished"
    assert ctx.S[stage]["screen"]["error"] is None, \
        ctx.S[stage]["screen"]["error"]


# ------------------------------------------------- the key that holds it
def test_the_gate_is_a_key_of_every_surfaces_workspace():
    """Both surfaces are asked the same question from the same factory, so
    both carry the key — and it opens on "follow the configuration", which is
    what ``nose_down_required`` falls back to."""
    from gui.v3 import session as ses

    S = ses.make_session("air")
    assert "nose_down" in S["airfoil"]
    assert "nose_down" in S["airfoil_aft"]
    assert S["airfoil"]["nose_down"] is None
    assert ses.airfoil_state(S, "main")["nose_down"] is None
    assert ses.airfoil_state(S, "aft")["nose_down"] is None


# --------------------------------------------------------- the control
def test_the_form_carries_the_gate_only_where_it_decides_something(capsys):
    """The row is the "wing-trimmed" job's alone: a tailless wing, and the
    trimming surface itself, would be shown a control whose answer never
    changes what the screen runs with."""
    from gui.v3.app import assemble

    ctx = _tailed_shell()
    ctx.render("airfoil", "screen")
    ctx.render("airfoil_aft", "screen")
    main = _texts(ctx.views[("airfoil", "screen")])
    aft = _texts(ctx.views[("airfoil_aft", "screen")])
    assert any("nose-down only" == t for t in main)
    assert not any("nose-down only" == t for t in aft)
    # and the two-gate sentence stops promising that switching ONE off
    # leaves the database unfiltered
    assert any("Switch them all off" in t for t in main)

    plain = assemble()
    plain.act("accept_mission")
    plain.render("airfoil", "screen")
    solo = _texts(plain.views[("airfoil", "screen")])
    assert not any("nose-down only" == t for t in solo)
    assert any("Switch one off" in t for t in solo)
    capsys.readouterr()


def test_the_gate_the_screen_runs_with_follows_the_control(capsys, monkeypatch):
    """THE OUTCOME IS THE FLOORS DICT THE SCREEN IS HANDED.

    Driven through the registered handler and the toolbar's own Run, with the
    screen itself stubbed: what is asserted is the argument
    ``api.screen_airfoils`` receives, which is the only thing that decides
    whether 40 sections are in the ranking or not.
    """
    from aerobo import api

    seen: dict = {}

    def fake_screen(**kw):
        seen.clear()
        seen.update(kw)
        return {"conditions": {"re": kw["re"], "mach": kw["mach"],
                               "cl_design": kw["cl_design"],
                               "tc_min": kw["tc_min"], "cm_max": kw["cm_max"]},
                "weights": kw["weights"],
                "floors": dict(kw.get("floors") or {}),
                "n_screened": 2174, "n_eligible": 125, "status_counts": {},
                "point": {}, "ranked": [], "wall_time_s": 0.1, "winner": None}

    monkeypatch.setattr(api, "screen_airfoils", fake_screen)
    monkeypatch.setattr(api, "screen_seed_candidates", lambda rep, n=0: [])

    ctx = _tailed_shell()
    # the cached library point: one call to screen_airfoils, no live sweep
    ctx.S["airfoil"]["re_source"] = "library"

    ctx.act("run_airfoil")
    _wait(ctx)
    assert seen["floors"].get(api.SCREEN_NOSE_DOWN_KEY) == 0.0

    # ...and the user can turn it off, which is what the docstring always
    # claimed and no control ever did
    assert "set_nose_down_gate" in ctx.actions
    ctx.act("set_nose_down_gate", "off")
    assert ctx.S["airfoil"]["nose_down"] is False
    from gui.v3 import session as ses
    assert ses.nose_down_required(ctx.S, "main") is False

    ctx.act("run_airfoil")
    _wait(ctx)
    assert api.SCREEN_NOSE_DOWN_KEY not in seen["floors"]

    # "automatic" is a third answer, not a synonym for on: it goes back to
    # following the configuration
    ctx.act("set_nose_down_gate", "auto")
    assert ctx.S["airfoil"]["nose_down"] is None
    assert ses.nose_down_required(ctx.S, "main") is True
    ctx.act("run_airfoil")
    _wait(ctx)
    assert seen["floors"].get(api.SCREEN_NOSE_DOWN_KEY) == 0.0
    capsys.readouterr()


def test_the_ranking_names_the_gate_when_it_fired(capsys, monkeypatch):
    """The header reads the REPORT's own floors, so a ranking produced under
    the gate goes on saying so after the control is moved — and one produced
    without it never claims a gate that did not run."""
    from aerobo import api

    def fake_screen(**kw):
        return {"conditions": {"re": kw["re"], "mach": kw["mach"],
                               "cl_design": kw["cl_design"],
                               "tc_min": kw["tc_min"], "cm_max": kw["cm_max"]},
                "weights": kw["weights"],
                "floors": dict(kw.get("floors") or {}),
                "n_screened": 2174, "n_eligible": 125, "status_counts": {},
                "point": {}, "ranked": [], "wall_time_s": 0.1, "winner": None}

    monkeypatch.setattr(api, "screen_airfoils", fake_screen)
    monkeypatch.setattr(api, "screen_seed_candidates", lambda rep, n=0: [])

    ctx = _tailed_shell()
    ctx.S["airfoil"]["re_source"] = "library"
    ctx.act("run_airfoil")
    _wait(ctx)
    ctx.render("airfoil", "ranking")
    on = _texts(ctx.views[("airfoil", "ranking")])
    assert any("nose-down gate was live" in t for t in on), on
    # it says what the eligible count is an eligibility for
    assert any("REFLEXED" in t and "125 of 2174" in t for t in on)

    ctx.act("set_nose_down_gate", "off")
    ctx.act("run_airfoil")
    _wait(ctx)
    ctx.render("airfoil", "ranking")
    off = _texts(ctx.views[("airfoil", "ranking")])
    assert not any("nose-down gate was live" in t for t in off), off
    capsys.readouterr()


# ------------------------------------------------- what it costs, measured
def test_the_gate_costs_forty_sections_and_moves_no_answer():
    """The real screen, at the library point, on this stage's own weights.

    Two claims, and they are separate: the gate is not free (40 sections
    leave the ranking, which is why the shell has to say it is on), and it is
    not a wrong default (the section the shipped configuration actually
    carries forward is the same either way).
    """
    import gui.nice_app as v1
    from aerobo import api
    from gui.v3 import session as ses

    if ses.library_point() is None:
        pytest.skip("no screening cache on this machine")

    S = ses.make_session("air")
    S["wing"]["choices"]["tail"] = True
    v1.normalise_choices(S["wing"]["choices"], keep="tail")
    ses.apply_choices(S)
    A = ses.airfoil_state(S, "main")
    A["re_source"] = "library"
    assert ses.surface_job(S, "main") == "wing-trimmed"
    assert ses.nose_down_required(S, "main") is True
    cond = ses.section_conditions(S, "main")

    def screen(floors):
        return api.screen_airfoils(
            weights=dict(A["weights"]), re=cond["re"], mach=cond["mach"],
            cl_design=cond["cl_design"], tc_min=cond["tc_min"],
            cm_max=cond["cm_max"], floors=floors, top_n=5, with_shape=False)

    gated = screen({api.SCREEN_NOSE_DOWN_KEY: 0.0})
    free = screen({})
    assert free["n_eligible"] > gated["n_eligible"]
    assert free["n_eligible"] - gated["n_eligible"] >= 20, \
        (free["n_eligible"], gated["n_eligible"])
    # the reflexed leaders the gate exists to remove are exactly what comes
    # back when it is off (hg* = Horten flying-wing sections)
    assert any(r["name"].startswith("hg") for r in free["ranked"])
    assert not any(r["name"].startswith("hg") for r in gated["ranked"])
    # ...and the section the default user flies is the same one
    assert gated["ranked"][0]["name"] == free["ranked"][0]["name"]
