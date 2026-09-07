"""Import-safety + assembly smoke tests for the V3 CAE shell.

Mirrors the V1/V2 smoke tests: importing ``gui.v3.*`` must be side-effect
free — no server, no window, no ``aerobo.api`` and no ``nicegui`` import at
module scope — and ``assemble()`` must build the WHOLE shell (every stage
and every view) without a server, because a stage that fails to build would
otherwise only show up at launch. The airfoil stage is built TWICE — the
wing's, and the second surface's stage 2.5 — from the one module.

The configuration sweep at the bottom is the V3 copy of the rule that
stopped the stale-menu bug class: every builder configuration the registry
can reach has to survive being selected THROUGH the stage's own handler,
with every view re-rendering afterwards.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

V3_MODULES = ("gui.v3", "gui.v3.theme", "gui.v3.figstyle", "gui.v3.widgets",
              "gui.v3.context", "gui.v3.session", "gui.v3.config",
              "gui.v3.app", "gui.v3.stages.mission", "gui.v3.stages.airfoil",
              "gui.v3.stages.wing", "gui.v3.stages.results")


def test_v3_imports_without_side_effects():
    """Run in a SUBPROCESS: inside the shared pytest interpreter earlier
    tests have already imported nicegui and aerobo.api, so a same-process
    check would depend on test ordering."""
    import subprocess

    code = (
        "import sys\n"
        f"sys.path.insert(0, {str(_REPO_ROOT)!r})\n"
        f"for m in {V3_MODULES!r}:\n"
        "    __import__(m)\n"
        "assert 'nicegui' not in sys.modules, 'nicegui leaked'\n"
        "assert 'aerobo.api' not in sys.modules, 'aerobo.api leaked'\n"
    )
    res = subprocess.run([sys.executable, "-c", code],
                         capture_output=True, text=True, timeout=180)
    assert res.returncode == 0, res.stderr


def test_v3_figstyle_is_pure_and_light():
    import plotly.graph_objects as go

    from gui.v3 import figstyle, theme

    fig = figstyle.empty("placeholder")
    assert isinstance(figstyle.plot(fig), go.Figure)
    # the shell is light: figures share the work surface, not a dark canvas
    assert figstyle.plot(go.Figure()).layout.paper_bgcolor == theme.WELL
    assert figstyle.export_config("x")["toImageButtonOptions"]["scale"] == 3
    # V1's palette is remapped, not kept
    v1_line = go.Figure()
    v1_line.add_scatter(x=[0, 1], y=[0, 1], line=dict(color="#0ea5e9"))
    assert figstyle.plot(v1_line).data[0].line.color == theme.ACCENT


def test_v3_assemble_builds_every_stage_and_view(capsys):
    from gui.v3 import session
    from gui.v3.app import assemble

    ctx = assemble()
    err = capsys.readouterr().err
    assert "failed to build" not in err, err

    expected = {(stage, key)
                for stage in session.STAGES
                for key, _label, _icon in session.VIEWS[stage]}
    assert set(ctx.views) == expected
    assert set(ctx.renderers) == expected          # every view can redraw

    for action in ("accept_mission", "run_airfoil", "stop_airfoil",
                   "run_airfoil_aft", "stop_airfoil_aft",
                   "launch", "cancel", "set_result", "set_choice",
                   "snippet", "clear_section",
                   # V3.5: the search policy is stage 1's, and every stage
                   # that searches reads it (gui/v3/stages/mission.py)
                   "set_search_mode", "set_search_effort", "set_search_stop",
                   "adopt_search_values",
                   # ...and the reach, offered on the Convergence view when a
                   # run comes back with no design (gui/v3/relax.py)
                   "relax_reach", "relax_stop", "relax_adopt"):
        assert action in ctx.actions, action
    assert ctx.act("definitely_not_registered") is None

    # assembly starts no worker and no run
    assert not ctx.manager.running
    assert ctx.manager.jobs == []
    assert ctx.polls                               # the heartbeat has work


@pytest.mark.parametrize("medium", ["air", "water", "track"])
def test_v3_assembles_and_renders_on_every_medium(medium, capsys):
    from gui.v3 import session
    from gui.v3.app import assemble

    ctx = assemble(medium)
    for stage in session.STAGES:
        for key, _label, _icon in session.VIEWS[stage]:
            ctx.render(stage, key)
    err = capsys.readouterr().err
    assert "failed to build" not in err, (medium, err)
    assert "Traceback" not in err, (medium, err)
    assert ctx.S["wing"]["choices"]["medium"] == medium


def test_v3_navigation_refuses_a_locked_stage_and_says_why(capsys):
    from gui.v3.app import assemble

    ctx = assemble()
    ctx.select("wing")                       # locked: the mission is the gate
    assert ctx.S["ui"]["selected"] == "mission"

    ctx.act("accept_mission")                # unlocks stage 2 and moves there
    assert ctx.S["mission"]["accepted"]
    assert ctx.S["ui"]["selected"] == "airfoil"

    # ...and stage 2 is NOT a second gate: a wing with no chosen section flies
    # the family's own published one, which is a complete design
    assert ctx.S["airfoil"]["decision"] is None
    ctx.select("wing")
    assert ctx.S["ui"]["selected"] == "wing"
    capsys.readouterr()


def test_v3_selecting_a_view_switches_exactly_one_container(capsys):
    from gui.v3.app import assemble

    ctx = assemble()
    ctx.select("mission", "point")
    visible = [k for k, col in ctx.views.items() if col.visible]
    assert visible == [("mission", "point")]
    capsys.readouterr()


def _builder_configurations():
    """One configuration per distinct branch of the wing builder — the twin
    of the helper in the V1/V2 smoke tests."""
    from aerobo import api

    import gui.nice_app as v1

    seen, out = set(), []
    for name in api.PROBLEM_SPECS:
        ch = v1.choices_from_problem(name)
        # tail_design is part of the KEY: the second surface's own
        # planform and tip device select different solvers, so a sweep
        # blind to them would leave the designed-tail and designed-
        # elevator families untested by the rule that keeps a menu
        # from offering what has no solver behind it
        key = (ch["medium"], ch["system"], bool(ch["tail"]), ch["airfoil"],
               v1.winglet_option_key(ch), ch["planform"],
               ch.get("tail_design", "fixed"))
        if key in seen:
            continue
        seen.add(key)
        out.append((name, ch))
    return out


def test_v3_wing_stage_survives_every_builder_configuration(capsys):
    """Selecting any reachable configuration through the stage's own
    handler must leave the state consistent with its menus and every view
    re-rendered — the rule that keeps a menu from offering what has no
    solver, or hiding what does."""
    from aerobo import api

    import gui.nice_app as v1
    from gui.v3 import config, session
    from gui.v3.app import assemble

    ctx = assemble()
    ctx.act("accept_mission")
    capsys.readouterr()

    for name, ch in _builder_configurations():
        ctx.act("set_choice", "medium", ch["medium"])
        ctx.act("set_choice", "system", ch["system"])
        ctx.act("set_choice", "tail", bool(ch["tail"]))
        # V3 asks for the tip device's SHAPE (v1.WINGLET_SHAPES), not for the
        # study menu's span-accounting key: the accounting is chosen by the
        # honest rule, and the blend is a value the shape carries.
        ctx.act("set_winglet", v1.winglet_shape_key(ch))
        ctx.act("set_choice", "airfoil", ch["airfoil"])
        ctx.act("set_choice", "planform", ch["planform"])
        ctx.act("set_choice", "tail_design",
                ch.get("tail_design", "fixed"))
        err = capsys.readouterr().err
        assert "Traceback" not in err, (name, err)

        S = ctx.S
        assert v1.choices_consistent(S["wing"]["choices"]), name
        assert S["wing"]["problem"] in api.PROBLEM_SPECS, name
        assert S["wing"]["optimiser"] in api.compatible_optimisers(
            S["wing"]["problem"]), name
        # whatever the configuration, it still assembles a runnable config
        cfg = config.build_cfg(S)
        assert cfg.problem_name == S["wing"]["problem"]
        for stage in session.STAGES:
            ctx.render(stage)
        assert "Traceback" not in capsys.readouterr().err, name


def test_v3_medium_change_drops_what_the_new_family_cannot_solve(capsys):
    """The same wiring test V2 has: a choice made through the real handler
    leaves the state consistent with the menus it is drawn from."""
    import gui.nice_app as v1
    from gui.v3.app import assemble

    ctx = assemble()
    ctx.act("accept_mission")
    S = ctx.S

    # every name here carries the chord law the shell opens on (BUILDER_START);
    # the medium switch drops the SECTION treatment, not the modifier, because
    # every one of these families has a chord-law twin.
    #
    # ...AND THE SHELL OPENS ON A WHOLE AEROPLANE. It used to open on a bare
    # wing, so this path read `wing + airfoil (XFOIL)`; since "the V3 shell
    # opens on a whole aeroplane" the default carries a tail with its height
    # free and its planform designed, and that family spells its section
    # twin `CST section (XFOIL)`. The WIRING under test — a choice made
    # through the real handler leaves the state consistent with the menus —
    # is unchanged, and is what the rest of this test checks.
    assert S["wing"]["problem"] == \
        "tail [free height, designed tail] + free chord law", \
        "the shell no longer opens where this test starts"
    ctx.act("set_choice", "airfoil", "section_wing")
    assert S["wing"]["problem"] == \
        "tail [free height, designed tail] + CST section (XFOIL) + free chord law"

    ctx.act("set_choice", "medium", "track")
    assert S["wing"]["choices"]["airfoil"] == "fixed"     # dropped, not left
    assert v1.choices_consistent(S["wing"]["choices"])
    assert S["wing"]["problem"] == _untouched("track")

    ctx.act("set_choice", "medium", "water")
    assert S["wing"]["problem"] == "hydrofoil + free chord law"
    _, notes = v1.derive_problem(S["wing"]["choices"])
    assert not [n for n in notes if "ignored" in n], notes
    capsys.readouterr()


def _selects(view):
    """Every ui.select in a built view (options dict + where it sits)."""
    return [e for e in view.descendants()
            if isinstance(getattr(e, "options", None), dict)]


def test_v3_wing_stage_does_not_ask_for_the_section_twice(capsys):
    """Stage 2 chooses the section. The wing stage's configuration list must
    therefore NOT carry a section menu — that redundancy is what made the
    pipeline read as though the earlier answer had not counted.

    What is genuinely still open (may the wing search RESHAPE that section?)
    stays reachable, and the two families that discard it stay reachable
    behind the advanced disclosure — but the 2-D section problem is not
    offered here at all, because it has no wing and is what stage 2 does.
    """
    import gui.nice_app as v1
    from gui.v3 import session
    from gui.v3.app import assemble

    ctx = assemble()
    ctx.act("accept_mission")
    ctx.render("wing", "type")
    capsys.readouterr()

    view = ctx.views[("wing", "type")]
    # "fixed" alone does not identify a section menu — the planform select
    # and the chord-law toggle carry that key too
    airfoil_keys = set(v1.AIRFOIL_OPTION_LABELS) - {"fixed"}
    section_menus = [s for s in _selects(view)
                     if set(s.options) & airfoil_keys]
    # the escape hatch is allowed, but only inside the advanced disclosure
    expansions = [e for e in view.descendants()
                  if type(e).__name__ == "Expansion"]
    inside = {id(d) for e in expansions for d in e.descendants()}
    for menu in section_menus:
        assert id(menu) in inside, "a section menu is back in the open list"
        assert "section_only" not in menu.options

    # ...and the one control that IS open is the reshape switch
    texts = [getattr(e, "text", "") or "" for e in view.descendants()]
    assert any("reshape" in t for t in texts)
    # the rest of the configuration list is untouched
    assert "planform" in texts and "tip device" in texts

    # the airfoil appears the way the MEDIUM does: the value it holds, and
    # the way back to the stage that owns it
    assert "airfoil" in texts
    assert "change in stage 2" in texts
    assert session.section_summary(ctx.S) in texts


def test_v3_the_reshape_switch_is_the_only_section_decision_left(capsys):
    """Flipping it must move the SOLVER, i.e. it is the same decision the
    old menu made — one control, named for the section rather than for the
    family behind it."""
    from gui.v3.app import assemble
    from gui.v3.stages import wing as wing_stage

    ctx = assemble()
    ctx.act("accept_mission")
    assert ctx.S["wing"]["choices"]["airfoil"] == "fixed"

    # ...ON WHATEVER THE SHELL OPENS ON, which is a whole aeroplane now and
    # was a bare wing. What this test is about is that the switch MOVES the
    # solver and comes back, so the two names are read off the round trip
    # rather than written down: hard-coding them made this a test about the
    # default family instead of about the switch.
    before = ctx.S["wing"]["problem"]
    ctx.act("set_choice", "airfoil", "section_wing")
    after = ctx.S["wing"]["problem"]
    assert after != before, "the reshape switch moved no solver"
    assert "XFOIL" in after, after
    ctx.act("set_choice", "airfoil", "fixed")
    assert ctx.S["wing"]["problem"] == before, "it did not come back"
    # the 2-D section problem is not among the treatments this stage names
    assert "section_only" not in wing_stage.TREATMENTS
    capsys.readouterr()


def test_v3_the_chosen_section_twin_cannot_outlive_the_section_it_flies(
        capsys):
    """Clearing the section must switch the twin off too.

    RULE 6 — state can never disagree with its own menus. On the water
    families the section is SELECTED BY THICKNESS, so "fly the one I chose"
    is not a flag on a run but a different problem
    (``api.CHOSEN_SECTION_TWINS``): t/c leaves the design vector because the
    polar and the Cp_min table come from one cached sweep of that shape.
    Leaving ``fly_section`` on after the section has gone left the shell
    selecting exactly that problem with no section to give it —
    ``api.needs_chosen_section`` true and no ``api.SECTION_KEY`` in the
    flags — while stage 3 still reported "ready" and stage 3's own switch
    was no longer drawn, so nothing on screen could show or undo it. That
    switch has since been deleted with the rest of stage 3's section panel,
    which makes ``release_chosen_section`` the only guard there is. The
    launch then died inside ``api.run`` with ``MissingSectionError``, which
    is deliberately not a ``ValueError`` and so is not absorbed by the
    penalty contract: it kills the job.

    Driven through the Edit menu's own handler, which is now the path that
    removes a section: the stage-2 button that used to do it is gone.
    """
    from aerobo import api
    from gui.v3 import config, session
    from gui.v3.app import assemble

    ctx = assemble("water")           # the only medium with the twins
    ctx.act("accept_mission")
    S = ctx.S
    session.set_section(S, {"name": "a_lin", "source": "library", "tc": 0.12},
                        "library")
    ctx.act("set_choice", "fly_section", True)
    assert S["wing"]["problem"] == "hydrofoil [chosen section] + free chord law"
    assert config.build_cfg(S).flags[api.SECTION_KEY] == "a_lin"

    ctx.act("clear_section")
    assert S["airfoil"]["decision"] is None
    assert S["wing"]["choices"]["fly_section"] is False
    assert S["wing"]["problem"] == "hydrofoil + free chord law"
    assert "tc" in api.PROBLEM_SPECS[S["wing"]["problem"]].default_bounds

    # the invariant, however it is satisfied: what the shell would launch
    # from here never asks for a section nobody can supply
    cfg = config.build_cfg(S)
    assert (not api.needs_chosen_section(cfg.problem_name)
            or api.SECTION_KEY in cfg.flags), "MissingSectionError on launch"
    # ...and it is genuinely launchable, so the guard is not merely unreached
    assert session.stage_states(S)["wing"][0] == "ready"
    err = capsys.readouterr().err
    assert "Traceback" not in err, err


def test_v3_the_span_is_asked_on_stage_3_and_nowhere_else(capsys):
    """The mission states no planform, stage 2 carries the aspect-ratio
    ESTIMATE, and stage 3 asks for a SPAN — as a length on its size card
    while the planform is fixed, and as the b_m BAND of the design box once
    the wing-loading mode makes it a design variable. Nobody types an aspect
    ratio anywhere: it is b²/S."""
    from gui.v3 import session
    from gui.v3.app import assemble

    ctx = assemble()
    ctx.render("mission", "operating")
    ctx.render("airfoil", "screen")
    ctx.render("wing", "type")
    capsys.readouterr()

    def _numbers(view):
        return [e for e in view.descendants()
                if type(e).__name__ == "Number"]

    ar = session.section_aspect_ratio(ctx.S)
    mission_values = [n.value for n in
                      _numbers(ctx.views[("mission", "operating")])]
    assert ar not in mission_values, "the mission still asks for an AR"
    assert ar in [n.value for n in _numbers(ctx.views[("airfoil", "screen")])]
    # ...and stage 3 asks for no aspect ratio at all
    assert ar not in [n.value for n in _numbers(ctx.views[("wing", "type")])]

    # the span IS asked on stage 3, as a LENGTH, while the planform is fixed
    ctx.act("set_span", 12.0)
    assert session.chosen_span(ctx.S) == 12.0
    ctx.render("wing", "type")
    assert 12.0 in [n.value for n in _numbers(ctx.views[("wing", "type")])]
    # ...and the design box carries no band for it: nothing is searching one
    ctx.render("wing", "box")
    assert "b_m" not in [e.text for e in ctx.views[("wing", "box")].descendants()
                         if type(e).__name__ == "Label"]

    # switch the mode: now it is a design variable, and the two cells are its
    # band
    ctx.act("set_planform", "wing_loading")
    ctx.act("set_bound", "b_m", 0, 8.0)
    ctx.act("set_bound", "b_m", 1, 12.0)
    assert session.span_box(ctx.S) == (8.0, 12.0)
    # (a typed bound does NOT rebuild the view it was typed into — that would
    # swallow the rest of the number — so the band is read back off a fresh
    # render, which is what any other control's redraw produces)
    ctx.render("wing", "box")
    box_values = [n.value for n in _numbers(ctx.views[("wing", "box")])]
    assert 8.0 in box_values and 12.0 in box_values
    capsys.readouterr()


def test_v3_a_divergent_aspect_ratio_offers_the_way_out_on_both_stages(
        capsys):
    """Saying "the section was designed for a different wing" is only honest
    if the shell also offers to resolve it. Both stages must carry the
    action, not just the warning."""
    from gui.v3 import session
    from gui.v3.app import assemble

    ctx = assemble()
    ctx.act("accept_mission")
    # choose a span, which is what makes the flown aspect ratio (b²/S)
    # something other than the guess the section was designed for
    ctx.act("set_span", 16.0)
    ctx.render("wing", "type")
    ctx.render("airfoil", "screen")
    capsys.readouterr()

    def _texts(view):
        return [getattr(e, "text", "") or "" for e in view.descendants()]

    wing_texts = _texts(ctx.views[("wing", "type")])
    assert any("re-design the section" in t for t in wing_texts)
    air_texts = _texts(ctx.views[("airfoil", "screen")])
    assert any("design for the flown wing" in t for t in air_texts)

    # ...and once resolved, neither offers it any more
    assert session.set_section_aspect_ratio(
        ctx.S, session.flown_aspect_ratio(ctx.S))
    ctx.render("wing", "type")
    ctx.render("airfoil", "screen")
    assert not any("re-design the section" in t
                   for t in _texts(ctx.views[("wing", "type")]))
    assert not any("design for the flown wing" in t
                   for t in _texts(ctx.views[("airfoil", "screen")]))
    capsys.readouterr()


def test_v3_a_ranking_from_another_design_point_says_so_on_screen(capsys):
    """The staleness the session reports has to reach the table itself —
    a ranking is an ORDER at one Reynolds number, and the tab that shows it
    is where that matters."""
    from gui.v3 import session
    from gui.v3.app import assemble

    ctx = assemble()
    ctx.act("accept_mission")
    S = ctx.S
    S["airfoil"]["re_source"] = "mission"        # so the estimate moves it
    S["airfoil"]["screen"]["report"] = {
        "conditions": {"re": session.section_conditions(S)["re"],
                       "cl_design": 0.5},
        "n_eligible": 1, "n_screened": 1, "wall_time_s": 0.0,
        "ranked": [{"name": "test", "tc": 0.12}]}
    ctx.render("airfoil", "ranking")
    texts = [getattr(e, "text", "") or ""
             for e in ctx.views[("airfoil", "ranking")].descendants()]
    assert not any("Screened at Re" in t for t in texts)

    assert session.set_section_aspect_ratio(S, 22.0)
    ctx.render("airfoil", "ranking")
    err = capsys.readouterr().err
    assert "Traceback" not in err, err
    texts = [getattr(e, "text", "") or ""
             for e in ctx.views[("airfoil", "ranking")].descendants()]
    assert any("Screened at Re" in t for t in texts)
    assert any(t == "re-screen" for t in texts)


def test_v3_a_ranking_ordered_at_another_design_lift_says_so_on_screen(capsys):
    """...and the banner has to fire on the CACHED point too. There the cache
    pins the Reynolds number, so the Re half can never move: what moves is the
    design lift, and the warning must name THAT one rather than quote a
    Reynolds number that did not budge."""
    from gui.v3 import session
    from gui.v3.app import assemble

    ctx = assemble()
    ctx.act("accept_mission")
    S = ctx.S
    S["airfoil"]["re_source"] = "library"   # the branch where only Cl can move
    cond = session.section_conditions(S)
    S["airfoil"]["screen"]["report"] = {
        "conditions": dict(cond),
        "n_eligible": 1, "n_screened": 1, "wall_time_s": 0.0,
        "ranked": [{"name": "test", "tc": 0.12}]}
    ctx.render("airfoil", "ranking")
    texts = [getattr(e, "text", "") or ""
             for e in ctx.views[("airfoil", "ranking")].descendants()]
    assert not any("Screened at" in t for t in texts)

    S["mission"]["V"] = 25.0                     # CL = W/(qS) moves with it
    ctx.render("airfoil", "ranking")
    err = capsys.readouterr().err
    assert "Traceback" not in err, err
    texts = [getattr(e, "text", "") or ""
             for e in ctx.views[("airfoil", "ranking")].descendants()]
    assert any("Screened at Cl" in t for t in texts)
    assert not any("Screened at Re" in t for t in texts)   # it did not move
    assert any(t == "re-screen" for t in texts)


def test_v3_results_place_the_best_design_inside_its_own_box(capsys):
    """A winning value means nothing on its own: the results page has to show
    it against the LOW and HIGH of the box that run searched, and say when
    the box — not the physics — is what set it."""
    from gui.v3 import theme
    from gui.v3.app import _properties, assemble

    ctx = assemble()
    ctx.S["run"]["record"] = {
        "param_labels": ["AR", "taper", "twist_tip_deg"],
        "best_x": [11.98, 0.55, -2.0],           # AR is on its high bound
        "bounds": [[6.0, 12.0], [0.3, 0.9], [-5.0, 0.0]],
        "best_score": 30.0, "n_evals": 20, "wall_time_s": 1.0,
        "breakdown": {"LoD": 30.0, "CL": 0.5},
        "config": {"problem_name": "trim wing", "optimiser": "bo",
                   "budget": 20, "seed": 0,
                   "bounds_overrides": {"taper": [0.3, 0.9]}}}
    ctx.render("results", "summary")
    err = capsys.readouterr().err
    assert "Traceback" not in err, err

    texts = [getattr(e, "text", "") or ""
             for e in ctx.views[("results", "summary")].descendants()]
    assert "low" in texts and "high" in texts        # the box IS on screen
    assert any(t == "6" for t in texts)              # AR's own low bound
    assert any(t == "12" for t in texts)             # ... and its high one
    assert any("at the high bound" in t for t in texts)
    assert any("narrowed" == t for t in texts)       # taper was overridden
    assert any("AR" in t and "ran into a bound" in t for t in texts)

    # and the properties panel counts it, so the fact survives leaving the tab
    ctx.S["ui"]["selected"] = "results"
    assert ("at a box bound", "1 of 3 variables", theme.WARN) \
        in _properties(ctx)
    capsys.readouterr()


def test_v3_results_without_a_recorded_box_say_so_instead_of_inventing_one(
        capsys):
    from gui.v3.app import assemble

    ctx = assemble()
    ctx.S["run"]["record"] = {
        "param_labels": ["AR"], "best_x": [9.0], "best_score": 30.0,
        "n_evals": 3, "wall_time_s": 0.1, "breakdown": {"LoD": 30.0},
        "config": {"problem_name": "trim wing"}}
    ctx.render("results", "summary")
    err = capsys.readouterr().err
    assert "Traceback" not in err, err
    texts = [getattr(e, "text", "") or ""
             for e in ctx.views[("results", "summary")].descendants()]
    assert any("carries no design box" in t for t in texts)
    assert any(t == "9" for t in texts)              # the value still shows
    capsys.readouterr()


def test_v3_output_log_is_capped(capsys):
    from gui.v3.app import assemble

    ctx = assemble()
    ctx.output.max_lines = 5
    for i in range(20):
        ctx.log(f"line {i}")
    assert len(ctx.output.lines) == 5
    capsys.readouterr()


# ------------------------------------------- typing into a numeric field
def _numbers(view):
    return [e for e in view.descendants() if type(e).__name__ == "Number"]


def _alive(view, element) -> bool:
    return any(e is element for e in view.descendants())


def _type_into_every_number(ctx, stage: str, view_key: str) -> list:
    """One keystroke into every number field of a view, one field at a time.

    The view is re-rendered BEFORE each field, so a field destroyed by the
    previous one's handler is not mistaken for a field that destroys itself:
    what is being tested is whether typing into a field takes THAT field
    away. Returns the value each broken field was showing.
    """
    view = ctx.views[(stage, view_key)]
    ctx.render(stage, view_key)
    count = len(_numbers(view))
    broken = []
    for i in range(count):
        ctx.render(stage, view_key)
        fields = _numbers(view)
        assert len(fields) == count, "the view changed shape while typing"
        num = fields[i]
        was = num.value
        # what a keystroke does: post one new value through on_change
        num.set_value((float(was) if was is not None else 0.0) + 0.001)
        if not _alive(view, num):
            broken.append(was)
    return broken


def test_typing_a_number_never_rebuilds_the_view_the_field_lives_in(capsys):
    """A number field that redraws its own container swallows the number.

    The input element is DESTROYED and recreated on the first posted digit,
    so it loses the focus and the rest of what is being typed goes nowhere:
    ``0.06`` → ``0.12`` in the CST half-width left the session on 0.005 (the
    floor, from the half-typed ``0.0``), ``5.5`` → ``7.5`` in the tail arm
    left 7.0, and a half-typed ``2.`` in the disk diameter is posted as
    empty, so ``float(e.value or 0.0)`` wrote a 0 m propeller and then took
    the field away.

    The file already engineers around this everywhere it matters — the
    aspect ratio draws its note into ``boxes["size"]``, the design box only
    stores and repaints the shell — and these seven fields were the ones
    that did not. The rule is structural, so the test is: type into every
    number field of every wing view and require the element to still be
    there afterwards.
    """
    from gui.v3 import session
    from gui.v3.app import assemble

    ctx = assemble()
    ctx.act("accept_mission")
    # a section carrying CST weights. It used to draw the design-box link's
    # half-width field, the worst offender of the seven; that field went with
    # the "Section carried from stage 2" card, and the link now always runs
    # at its default half-width. The section stays here because it changes
    # which design-box rows exist, and the box is typed into below.
    session.set_section(ctx.S, {"name": "probe", "source": "library",
                                "tc": 0.12,
                                "w_upper": [0.15, 0.16, 0.17, 0.18],
                                "w_lower": [-0.1, -0.11, -0.09, -0.08]},
                        "library")
    for key, value in (("tail", True), ("tail_arm", "fixed"),
                       ("tail_type", "v_tail"),
                       ("airfoil", "section_wing")):
        ctx.act("set_choice", key, value)
    ctx.render("wing")
    capsys.readouterr()

    shown = [n.value for n in _numbers(ctx.views[("wing", "type")])]
    # the two fields of the wing type view, both really on screen. The
    # elevator chord was a third until the control toggle went (V3 flies the
    # all-moving surface always — session.V3_PINNED_CHOICES), and the CST
    # half-width a fourth until the section card was deleted
    for value in (5.5, 35.0):                   # arm, dihedral
        assert value in shown, (value, shown)
    assert _type_into_every_number(ctx, "wing", "type") == []
    assert _type_into_every_number(ctx, "wing", "box") == []

    # the propeller card lives on a family that declares the slipstream flag
    # — and the shell's default is no longer one. It opened on a bare wing
    # when this was written; since "the V3 shell opens on a whole aeroplane"
    # the default carries a tail, and only the objective's own wing modes
    # read ``slipstream``. So the card is asked for rather than assumed, or
    # this half of the test renders nothing and asserts about it.
    prop = assemble()
    prop.act("accept_mission")
    prop.act("set_choice", "tail", False)
    from aerobo import api as _api
    assert "slipstream" in _api.PROBLEM_SPECS[
        prop.S["wing"]["problem"]].flags, prop.S["wing"]["problem"]
    prop.S["wing"]["prop"].update({"enabled": True, "layout": "pair"})
    prop.render("wing")
    shown = [n.value for n in _numbers(prop.views[("wing", "solver")])]
    for value in (2.0, 2.5, 0.6):               # diameter, pair centre, CT
        assert value in shown, (value, shown)
    assert _type_into_every_number(prop, "wing", "solver") == []
    # ...and an empty field is not a zero-diameter propeller
    prop.S["wing"]["prop"]["D_p"] = 2.0
    prop.render("wing", "solver")
    fields = _numbers(prop.views[("wing", "solver")])
    [d_p] = [n for n in fields if n.value == 2.0]
    d_p.set_value(None)                          # the half-typed "2."
    assert prop.S["wing"]["prop"]["D_p"] == 2.0
    err = capsys.readouterr().err
    assert "Traceback" not in err, err


def test_a_run_with_no_feasible_incumbent_still_says_the_run_is_over(capsys):
    """A search that finds nothing admissible is an OUTCOME, not a crash.

    ``RunResult.to_dict`` always emits ``best_score``, so the ``float('nan')``
    default in the poll's log line never fired: on a constrained family that
    was stopped (or finished) with no feasible design the key is present and
    None, ``f"{None:.6g}"`` raised TypeError inside the completion branch,
    and the branch had already marked the job as seen — so it never re-ran.
    ``_tick`` swallowed the traceback and the status bar kept the busy dot,
    the ``— 0/N`` text and a progress bar for the rest of the session while
    the tree marked Results done. The completion line never reached the log.
    """
    from aerobo import api

    from gui import nice_app as v1
    from gui.v3 import config
    from gui.v3.app import assemble

    ctx = assemble("water")                  # a constrained family
    ctx.act("accept_mission")
    assert api.PROBLEM_SPECS[ctx.S["wing"]["problem"]].is_constrained

    cfg = config.build_cfg(ctx.S, seed=0)
    result = api.partial_result(cfg, [])     # stopped before anything feasible
    rd = result.to_dict()
    assert "best_score" in rd and rd["best_score"] is None

    job = v1.RunJob(cfg=cfg, label="probe · seed 0", budget=40)
    job.status, job.result = "cancelled", result
    ctx.manager.jobs = [job]
    ctx.manager.version += 1
    ctx.status("running…", "busy", 0.0)
    for fn in list(ctx.polls):               # the shell's own 0.5 s heartbeat
        fn()

    assert ctx.S["run"]["record"] == rd      # the record still reaches stage 4
    assert ctx.statusbar.msg.text == "run complete"
    assert not ctx.statusbar.bar.visible     # no progress bar left behind
    logged = [getattr(e, "text", "") or ""
              for row in ctx.output.lines for e in row.descendants()]
    assert any("no feasible incumbent" in t for t in logged), logged
    err = capsys.readouterr().err
    assert "Traceback" not in err, err


def test_the_arm_note_quotes_the_box_the_run_will_search(capsys):
    """One question, one place: the arm is ASKED in the design box and READ
    OUT on the tail card, so the two cannot disagree.

    The note is the only sentence in stage 3 that says where the second
    surface will end up. It read the family's PUBLISHED box, so after
    narrowing ``l_t_m`` to 3–4 m in the design box — which ``bounds_overrides``
    really does send to the run — the card went on promising a solver free to
    place the tail anywhere from 3 to 8 m aft.
    """
    from gui.v3 import config
    from gui.v3.app import assemble

    ctx = assemble()
    ctx.act("accept_mission")
    ctx.act("set_choice", "tail", True)
    S = ctx.S
    assert S["wing"]["choices"]["tail_arm"] == "free"     # a searched arm
    ctx.render("wing", "type")
    capsys.readouterr()

    def _texts(view):
        return [getattr(e, "text", "") or "" for e in view.descendants()]

    assert any("3 – 8 m aft of the wing" in t
               for t in _texts(ctx.views[("wing", "type")]))

    # narrow the row through the design box's OWN input, as a user would
    ctx.render("wing", "box")
    box = ctx.views[("wing", "box")]
    seen, fields = False, []
    for e in box.descendants():
        if getattr(e, "text", None) == "l_t_m":
            seen = True
        elif seen and type(e).__name__ == "Number":
            fields.append(e)
    assert [f.value for f in fields[:2]] == [3.0, 8.0]
    fields[1].set_value(4.0)
    assert config.bounds_overrides(S)["l_t_m"] == [3.0, 4.0]

    ctx.render("wing", "type")
    texts = _texts(ctx.views[("wing", "type")])
    assert any("3 – 4 m aft of the wing" in t for t in texts), texts
    assert not any("3 – 8 m" in t for t in texts)
    # ...and it says the box is one that was typed, not the published one
    assert any("typed" in t for t in texts)
    err = capsys.readouterr().err
    assert "Traceback" not in err, err


# ------------------------------------ stage 1's own switches obey the rules

def _untouched(medium: str) -> str:
    """The family an UNTOUCHED card of this medium derives.

    Read off the shell's own start state rather than pasted, because WHICH
    family a medium opens on is a decision that moves: the track card opened
    on the pylon-borne plain wing until ``car_endplates`` became True
    (5ea524f — a rear wing is bolted to the car by the plate, so the plate is
    a designed structure and a different solver family). Three assertions
    below pinned the old string and went red on that flip while the shell was
    right; the invariant they were reaching for is that a medium switch lands
    where an untouched card of the new medium lands, which is what this
    states.
    """
    import gui.nice_app as v1

    ch = v1.start_choices(medium=medium)
    v1.normalise_choices(ch)
    return v1.derive_problem(ch)[0]


def _mission_toggle(ctx, option: str):
    """The ``ui.toggle`` in the mission stage's operating view that offers
    ``option`` — i.e. the widget a user actually clicks, with its real
    ``on_change`` attached. Driving the stage's handler by hand would pass
    while the control the user has stayed wired to the old one."""
    from nicegui import ui

    for e in ctx.views[("mission", "operating")].descendants():
        if isinstance(e, ui.toggle) and option in (e.options or {}):
            return e
    raise AssertionError(f"no mission toggle offers {option!r}")


def test_a_stage_one_switch_leaves_no_choice_the_new_family_cannot_solve(
        capsys):
    """The medium and the lifting system are builder changes like any other.

    Session 24c stopped the stale-menu bug class by putting every builder
    change through ``nice_app.normalise_choices``: it drops a speciality the
    newly derived family has no solver for, so state can never hold a value
    its own menu no longer offers (rule 6). V3 wired that into stage 3's
    ``set_choice`` and into the second-surface switch — but stage 1 owns two
    switches of its own, and they wrote the choice and called
    ``apply_choices`` directly. A raked tip device asked for in air therefore
    survived into the hydrofoil: the tip-device select fell back to "none"
    while ``winglet_type='raked'`` was still in the state and still went out
    in the run's flags, against a spec that declares no such flag.

    Driven through the toggles themselves, because the defect was in the path
    the widget takes, not in the derivation it ends at.
    """
    import gui.nice_app as v1
    from gui.v3 import config
    from gui.v3.app import assemble

    # (a) a tip device the water family cannot fly. V3 asks for the SHAPE
    # (v1.WINGLET_SHAPES): "canted" is span-capped in air, and the water
    # family is scored span-free — its trade is the cant's SIGN, not its
    # projection — so the capped device has nowhere to go.
    ctx = assemble()
    ctx.act("accept_mission")
    # ...on whatever the shell opens on. It opened on a bare wing when this
    # was written and opens on a whole aeroplane now, so the family NAME is
    # incidental — what this test is about is that the span-capped device
    # is asked for here and is gone after the medium switch below.
    ctx.act("set_winglet", "canted")
    S = ctx.S
    assert "winglet" in S["wing"]["problem"] and \
        "span-capped" in S["wing"]["problem"] or \
        S["wing"]["problem"].startswith("winglet_capped"), \
        S["wing"]["problem"]

    _mission_toggle(ctx, "water").set_value("water")
    ch = S["wing"]["choices"]
    assert v1.choices_consistent(ch), ch
    assert v1.winglet_shape_key(ch) in v1.winglet_shapes(ch)
    assert "winglet_type" not in config.cfg_dict(S)["flags"]
    # a WATER family, and the span-capped device is gone from it. The tail
    # travels across the medium now (the shell opens with one), so what is
    # asserted is the DEVICE's absence rather than a bare family name.
    assert S["wing"]["problem"].startswith("hydrofoil"), S["wing"]["problem"]
    assert "winglet" not in S["wing"]["problem"], S["wing"]["problem"]

    # (a2) the same for a VALUE the new family cannot draw: a vertical fence
    # carries a cant-band flag, and the car's endplates ARE its tip device,
    # so both the device and its flag have to go rather than travel in the
    # run's flags against a spec that declares neither.
    ctx = assemble()
    ctx.act("accept_mission")
    ctx.act("set_winglet", "vertical")
    S = ctx.S
    assert "winglet" in S["wing"]["problem"], S["wing"]["problem"]
    assert config.cfg_dict(S)["flags"]["winglet_type"] == "vertical"
    _mission_toggle(ctx, "track").set_value("track")
    assert v1.choices_consistent(S["wing"]["choices"])
    assert "winglet_type" not in config.cfg_dict(S)["flags"]
    assert S["wing"]["problem"] == _untouched("track")

    # (b) a designed section the car wing cannot fly
    ctx = assemble()
    ctx.act("accept_mission")
    ctx.act("set_choice", "airfoil", "section_wing")
    S = ctx.S
    assert "XFOIL" in S["wing"]["problem"], S["wing"]["problem"]
    _mission_toggle(ctx, "track").set_value("track")
    assert S["wing"]["choices"]["airfoil"] == "fixed"     # dropped, not left
    assert v1.choices_consistent(S["wing"]["choices"])
    assert S["wing"]["problem"] == _untouched("track")

    # (c) a thickness sweep the tandem pair CAN now fly — the OTHER switch,
    # and the other half of the rule.
    #
    # This case used to assert the choice was dropped, because the pair had no
    # solver for it: 57 configurations were refused for a missing design
    # VARIABLE, not for missing physics. Now that `tandem + t/c` exists the
    # same switch must KEEP the freedom rather than silently discarding one
    # the new family can honour — dropping a solvable choice is the same class
    # of bug as keeping an unsolvable one, seen from the other side. Case (b)
    # above still covers the drop path.
    ctx = assemble()
    ctx.act("accept_mission")
    ctx.act("set_choice", "airfoil", "tc_sweep")
    S = ctx.S
    # the t/c sweep is asked for HERE — on whatever the shell opens on —
    # and the point below is that the TANDEM keeps it
    assert "t/c" in S["wing"]["problem"], S["wing"]["problem"]
    _mission_toggle(ctx, "tandem").set_value("tandem")
    assert S["wing"]["choices"]["airfoil"] == "tc_sweep"      # kept, not dropped
    assert v1.choices_consistent(S["wing"]["choices"])
    assert S["wing"]["problem"] == "tandem + t/c + free chord law"

    err = capsys.readouterr().err
    assert "Traceback" not in err, err


def test_the_lifting_system_never_survives_a_medium_that_cannot_fly_it(
        capsys):
    """How many surfaces carry the load is state, and state is re-derived.

    ``system`` is not one of ``nice_app.NORMALISED_KEYS`` — it is not a
    speciality — so ``normalise_choices`` never touched it and "tandem"
    survived a switch to water or to the track, where ``derive_problem``
    quietly ignores it. Nothing downstream then agreed with anything else:
    the disabled second-surface switch gave the car's reason in its tooltip
    and the tandem's reason in the hint underneath, and
    ``session.second_surface_name`` reads ``system`` before ``tail``, so the
    hydrofoil's ELEVATOR was called the "rear wing" in stage 2's surface
    toggle, its design-point heading and its adopt log.
    """
    import gui.nice_app as v1
    from gui.v3 import session
    from gui.v3.app import assemble

    # (a) water: the second surface is an ELEVATOR, and is named one
    ctx = assemble()
    S = ctx.S
    _mission_toggle(ctx, "tandem").set_value("tandem")
    assert S["wing"]["problem"] == "tandem + free chord law"
    _mission_toggle(ctx, "water").set_value("water")
    assert S["wing"]["choices"]["system"] == "single"
    assert v1.option_available(S["wing"]["choices"], "system",
                               S["wing"]["choices"]["system"])
    ctx.act("set_second_surface", True)
    # a surface that has just appeared opens DESIGNED (v1.tail_design_start):
    # taper, aspect ratio and washout are what a stabiliser IS. The SEPARATION
    # opens free too now — the shell opens on a whole aeroplane and the tail's
    # height carries across the medium — so the claim is asserted rather than
    # the whole name spelled out.
    assert S["wing"]["problem"].startswith("hydrofoil + elevator ["), \
        S["wing"]["problem"]
    assert "designed elevator" in S["wing"]["problem"], S["wing"]["problem"]
    assert session.second_surface_name(S) == "elevator"
    assert session.aft_surface(S) == "elevator"

    # (b) track: the refusal gives ONE reason, and it is the car's own
    ctx = assemble()
    S = ctx.S
    _mission_toggle(ctx, "tandem").set_value("tandem")
    _mission_toggle(ctx, "track").set_value("track")
    assert S["wing"]["choices"]["system"] == "single"
    texts = [getattr(e, "text", "") or ""
             for e in ctx.views[("mission", "operating")].descendants()]
    assert any("nothing behind it to trim with" in t for t in texts), texts
    assert not [t for t in texts if "tandem pair already carries" in t]

    err = capsys.readouterr().err
    assert "Traceback" not in err, err


def test_the_speed_sensitivity_chart_is_flown_in_the_water_the_mission_chose(
        capsys):
    """One question, one place: the chart and the table are the same point.

    The "Sensitivity to speed" figure rebuilds the ``api.design_point`` call
    by hand — the only place a stage does — and it left out ``water=``, so it
    took the argument's "sea" default while the read-outs two boxes above it
    came from ``session.design_point`` in the water the mission chose. In
    fresh water the dashed mission-speed line did not cross the numbers the
    same page states: CL 2.6 % and Re 4.7 % out, under a caption promising
    "every point is one api.design_point call at the same weight and area".
    """
    from gui.v3 import session
    from gui.v3.app import assemble

    ctx = assemble("water")
    S = ctx.S
    _mission_toggle(ctx, "fresh").set_value("fresh")
    assert S["water"] == "fresh"

    ctx.render("mission", "point")
    from nicegui import ui
    figs = [e.figure for e in ctx.views[("mission", "point")].descendants()
            if isinstance(e, ui.plotly)]
    assert len(figs) == 1
    traces = {t["name"]: t for t in figs[0]["data"] if t.get("name")}

    v0 = float(S["mission"]["V"])
    dp = session.design_point(S)
    for name, key in (("CL design", "cl_design"), ("Re at MAC", "re_mac")):
        tr = traces[name]
        i = min(range(len(tr["x"])), key=lambda j: abs(tr["x"][j] - v0))
        assert tr["x"][i] == pytest.approx(v0)
        assert tr["y"][i] == pytest.approx(dp[key], rel=1e-9), name

    err = capsys.readouterr().err
    assert "Traceback" not in err, err


# ------------------------------- stage 4 reads a run that found nothing
def _no_incumbent_record(medium: str = "water", n_rows: int = 2) -> dict:
    """A REAL record from a constrained family that found nothing feasible.

    Built the way the shell builds it — ``config.build_cfg`` on the assembled
    session, then ``api.partial_result`` over evaluations whose margins are
    all negative — so the test is pinned to the record the runner really
    hands to ``set_result``, not to a hand-written dict that could drift away
    from it.
    """
    from aerobo import api

    from gui.v3 import config
    from gui.v3.app import assemble

    ctx = assemble(medium)
    ctx.act("accept_mission")
    cfg = config.build_cfg(ctx.S, seed=0)
    dim = len(api.PROBLEM_SPECS[cfg.problem_name].param_labels)
    rows = [{"x": [0.0] * dim, "f": -1.0, "g": [-0.5], "feasible": False}
            for _ in range(n_rows)]
    return ctx, api.partial_result(cfg, rows).to_dict()


def _settled(pred, timeout: float = 20.0) -> bool:
    """Stage 4 re-evaluates the winner in a BACKGROUND thread by design (on an
    XFOIL family that re-evaluation is a real solve and must never block the
    shell), so the test waits on the value that thread publishes rather than
    on a sleep."""
    import time

    t0 = time.time()
    while time.time() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.01)
    return False


def test_a_constrained_run_that_found_nothing_is_not_called_unconstrained(
        capsys):
    """Whether a problem is CONSTRAINED is a property of the problem, never of
    the record.

    ``_constraints`` built its rows out of the incumbent's margins, and a
    constrained search that finds no feasible design has no incumbent: no
    ``best_g``, no ``breakdown``, therefore no rows. The fall-through then
    printed "Unconstrained problem — every evaluated design is admissible" for
    a family whose spec declares ``is_constrained`` and whose CONSTRAINED chip
    the tool bar was drawing on the same screen — the exact inverse of what
    happened, on the one result a user most needs read correctly.

    The wording is now taken from the run's own ``is_constrained`` (and the
    registry spec of the problem it names — a LOOKUP, never a match on a
    problem name: the ~1445 registered problems are generated twins and a
    name branch would miss every one of them), so silence is reported as
    silence.
    """
    from aerobo import api

    from gui.v3.app import assemble

    ctx, rd = _no_incumbent_record()
    spec = api.PROBLEM_SPECS[rd["config"]["problem_name"]]
    assert spec.is_constrained and spec.constraint_labels
    assert rd["best_g"] is None and rd["breakdown"] is None
    assert rd["n_feasible"] == 0 and rd["n_evals"] == 2

    ctx.S["run"]["record"] = rd
    ctx.render("results", "summary")
    texts = [getattr(e, "text", "") or ""
             for e in ctx.views[("results", "summary")].descendants()]
    assert not any("Unconstrained problem" in t for t in texts), texts
    # the sentence now comes from ``gui.diagnose`` — it says the same thing
    # and then says WHICH constraint, so the block can no longer end on
    # "widen the design box" with no object
    assert any("no design satisfied every constraint" in t.lower()
               for t in texts), texts
    assert not any("Widen the design box, or move the mission" in t
                   for t in texts), texts
    assert any("cavitation margin" in t for t in texts), texts
    assert any("0 of 2" in t for t in texts), texts

    # ...and an unconstrained family still says what it always said
    plain = assemble()
    plain.S["run"]["record"] = {
        "param_labels": ["AR"], "best_x": [9.0], "best_score": 30.0,
        "n_evals": 3, "wall_time_s": 0.1, "breakdown": {"LoD": 30.0},
        "config": {"problem_name": "trim wing"}}
    plain.render("results", "summary")
    plain_texts = [getattr(e, "text", "") or ""
                   for e in plain.views[("results", "summary")].descendants()]
    assert any("Unconstrained problem — every evaluated design is admissible."
               == t for t in plain_texts), plain_texts

    err = capsys.readouterr().err
    assert "Traceback" not in err, err


def test_the_section_report_settles_when_the_run_recorded_no_design_vector(
        capsys):
    """``None`` means STILL RUNNING, so every way out of the re-evaluation has
    to publish a settled value.

    ``set_result`` clears ``section_report`` to None and every reader of it —
    the geometry view's flown section outline, the section ``.dat`` export —
    treats None as "the polar is still being swept". ``_compute`` returned
    early when there was no design vector to sweep (or when the run's config
    could not be rebuilt at all) without ever publishing, so those readers
    waited for the rest of the session on a run where nothing is running and
    nothing ever will be, while every other view of the same record said
    honestly that it recorded no design vector.
    """
    from gui.v3.app import assemble

    ctx, rd = _no_incumbent_record()
    assert rd["best_x"] is None

    ctx.act("set_result", rd)
    R = ctx.S["run"]
    assert _settled(lambda: R.get("report") is not None)
    assert _settled(lambda: R.get("section_report") is not None), \
        "the section report was left on its 'still running' value for good"
    assert "no design vector" in str(R["section_report"].get("error", "")), \
        R["section_report"]

    # (b) the other way out: the run's configuration cannot be rebuilt at all
    ctx2 = assemble("water")
    broken = dict(rd)
    broken.pop("config")                       # RunConfig(**…) never happens
    ctx2.act("set_result", broken)
    R2 = ctx2.S["run"]
    assert _settled(lambda: R2.get("section_report") is not None), \
        "a record the config cannot be rebuilt from waits the same way"

    err = capsys.readouterr().err
    assert "Traceback" not in err, err


def test_results_has_no_section_view(capsys):
    """The aerofoil is stage 2's answer (and stage 2.5's for the second
    surface): its shape, its polar and the ranking it came from all live
    there. A Section tab in the RESULTS answered the same question a second
    time, in a second place — the pipeline's one rule.

    What must NOT go with it: the designed shape itself. ``section_report`` is
    still computed, the geometry view still draws the flown outline, and the
    section ``.dat`` is still exportable.
    """
    from gui.v3 import session
    from gui.v3.app import assemble

    keys = [k for k, _lbl, _icon in session.VIEWS["results"]]
    assert keys == ["summary", "geometry", "loading", "log"], keys

    ctx = assemble("air")
    assert ("results", "section") not in ctx.views

    # ...and the section is still REPORTED, just not as a tab of its own
    rd = {"problem_name": "trim wing", "optimiser": "bo", "seed": 0,
          "best_f": 20.0, "best_x": [0.5, 0.0, -2.0], "history": [],
          "n_evals": 3, "wall_time_s": 0.1, "breakdown": {"LoD": 20.0},
          "config": {"problem_name": "trim wing"}}
    ctx.act("set_result", rd)
    assert _settled(lambda: ctx.S["run"].get("section_report") is not None)
    ctx.render("results", "geometry")

    err = capsys.readouterr().err
    assert "Traceback" not in err, err


# --------------------------------------------------- File ▸ New session
def test_a_new_session_leaves_every_stage_writing_where_the_shell_reads(
        capsys):
    """A stage is built ONCE and closes over its own sub-dict — ``W =
    S["wing"]`` (wing.py), the airfoil workspace (airfoil.py), ``R =
    S["run"]`` (results.py) — so "New session" has to refill those dicts
    rather than rebind them.

    Rebinding them left every handler writing into an object nothing else
    read any more, which is the worst failure this shell can have: after a
    reset, stage 3 redrew ITSELF as the hydrofoil while ``S["wing"]`` — the
    tree badge, the status bar, the properties grid, Tools ▸ Copy run
    snippet and ``config.build_cfg``, the ONE place V3 builds a run — still
    held the air problem. The screen showed one design and Run launched
    another. The same orphaning swallowed a finished run (Results stayed
    "locked — no completed run yet" with the result on screen) and a stage-2
    screening (Stop could not cancel what it could not see).

    The three assertions below are the three symptoms, and they are
    deliberately BEHAVIOURAL: they say the stages and the shell agree after
    a reset, not how the reset is implemented — refilling the dicts in place
    and rebuilding the stages both satisfy them.
    """
    from aerobo import api

    from gui.v3 import app, config, session
    from gui.v3.app import assemble

    ctx = assemble("air")
    ctx.act("accept_mission")

    app._new_session(ctx)
    assert not ctx.S["mission"]["accepted"]          # it really did reset

    # (a) stage 3: what the stage DRAWS is what the shell would LAUNCH
    ctx.act("accept_mission")
    ctx.act("set_choice", "medium", "water")
    S = ctx.S
    assert S["wing"]["choices"]["medium"] == "water"
    launched = config.cfg_dict(S)["problem_name"]
    assert launched == S["wing"]["problem"]
    drawn = [getattr(e, "text", "") or ""
             for e in ctx.views[("wing", "type")].descendants()]
    assert api.PROBLEM_SPECS[launched].display in drawn, drawn

    # (b) stage 4: a finished run reaches the session the tree reads
    cfg = config.build_cfg(S, seed=0)
    spec = api.PROBLEM_SPECS[cfg.problem_name]
    rd = api.partial_result(cfg, [{
        "x": [0.0] * len(spec.param_labels), "f": -1.0,
        "g": [-0.5] * len(spec.constraint_labels or ()),
        "feasible": False}]).to_dict()
    ctx.act("set_result", rd)
    assert S["run"]["record"] is rd
    assert session.stage_states(S)["results"][0] != "locked"

    # (c) stage 2: Stop still cancels a screening the session says is on
    session.airfoil_state(S, "main")["screen"]["running"] = True
    app._stop(ctx)
    log = [" ".join(getattr(e, "text", "") or "" for e in row.descendants())
           for row in ctx.output.lines]
    assert any("screening will stop" in line for line in log), log[-4:]

    err = capsys.readouterr().err
    assert "Traceback" not in err, err


def test_v3_the_search_policy_moves_through_its_own_handlers(capsys):
    """V3.5: switching between the measured recommendation and the user's own
    values goes through stage 1's handlers, and leaves every view able to
    re-render — the same rule that keeps a menu from disagreeing with the
    state behind it."""
    from gui.v3 import config, session
    from gui.v3.app import assemble

    ctx = assemble()
    capsys.readouterr()
    assert session.search_is_recommended(ctx.S)

    for mode in ("own", "recommended", "own"):
        ctx.act("set_search_mode", mode)
        assert session.search_state(ctx.S)["mode"] == mode
        for stage in session.STAGES:
            for key, _label, _icon in session.VIEWS[stage]:
                ctx.render(stage, key)
        err = capsys.readouterr().err
        assert "Traceback" not in err, (mode, err)
        # ...and it still assembles a runnable config either way
        d = config.cfg_dict(ctx.S)
        assert d["budget"] >= 2 and d["optimiser"]

    for effort in session.SEARCH_EFFORTS:
        ctx.act("set_search_effort", effort)
        assert session.search_state(ctx.S)["effort"] == effort
    ctx.act("set_search_stop", False)
    assert session.search_state(ctx.S)["stop_when_converged"] is False
    ctx.act("adopt_search_values")
    assert not session.search_is_recommended(ctx.S)
    capsys.readouterr()


def test_v3_every_view_renders_with_a_study_installed(tmp_path, monkeypatch,
                                                      capsys):
    """The RECOMMENDED branch of stages 1, 2 and 3 only exists when a frozen
    study is present, so the shell has to be assembled against one — without
    it every card falls back to the user's own fields and the branch under
    test is never entered."""
    import json

    from aerobo.optimize import budget as B
    from gui.v3 import session
    from gui.v3.app import assemble

    block = {
        "method": {"optimiser": "bo", "acqf": "logei", "label": "bo-logei"},
        "method_constrained": {"optimiser": "bo", "acqf": None,
                               "label": "bo-logcei"},
        "law": {"0.9": {"intercept": 4.0, "per_dim": 2.0},
                "0.95": {"intercept": 10.0, "per_dim": 6.0},
                "0.99": {"intercept": 20.0, "per_dim": 12.0}},
        "n_init": {"per_dim": 2.0, "min": 4, "max": 16},
        "restarts": 1, "stop": {"patience": 25, "tol": 0.005},
        "cost": {"per_eval_s": 0.002, "overhead": {}},
        "budget_clamp": [12, 400],
    }
    path = tmp_path / "search_budget.json"
    path.write_text(json.dumps(
        {"version": 1, "measured": "2026-01-01", "n_runs": 42, "cases": 9,
         "machine": "test",
         "kinds": {"wing": block,
                   "airfoil": {**block,
                               "cost": {"per_eval_s": 1.5, "overhead": {}},
                               "budget_clamp": [12, 200]}}}))
    monkeypatch.setattr(B, "PAYLOAD_PATH", path)
    B._CACHE.clear()

    ctx = assemble()
    assert session.search_study() is not None
    for stage in session.STAGES:
        for key, _label, _icon in session.VIEWS[stage]:
            ctx.render(stage, key)
    err = capsys.readouterr().err
    assert "Traceback" not in err, err
    assert "failed to render" not in err, err

    # ...and the same again with the second surface open, which is what puts
    # stage 2.5's own recommended budget on screen
    ctx.act("set_second_surface", True)
    for stage in session.STAGES:
        for key, _label, _icon in session.VIEWS[stage]:
            ctx.render(stage, key)
    err = capsys.readouterr().err
    assert "Traceback" not in err, err
    B._CACHE.clear()
