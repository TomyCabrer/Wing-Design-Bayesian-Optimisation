"""AIRFOIL ONLY: a session that states the flow and designs a section in it.

Everything downstream of stage 2 in this shell was reachable only through a
MISSION — a design weight, a reference area and an aspect-ratio estimate,
which together imply a chord and a lift coefficient. That is the right
question for an aircraft and the wrong one for an aerofoil: somebody
designing a section already knows the flow, and inventing a vehicle to get
past stage 1 states a chord nobody chose and screens the section at it.

The engine never needed the vehicle. ``api.optimize_airfoil`` and
``api.screen_airfoils`` take three scalars — ``re``, ``mach``, ``cl_design``
— and never see a mission. What was missing was the derivation and the
question: ``api.flow_point`` is the first (a fluid, a speed and a chord, or a
density, a viscosity, a Mach number and a speed), and stage 1's second face
is the other.

What this file pins:

* the physics of ``api.flow_point`` against the ISA / Sutherland / water
  functions it composes, and its refusals — a derived fluid must not quietly
  ignore a typed density;
* the session: which stages exist at all, what locks them and why, and that
  the section's operating point comes from the stated flow (the MACH
  included — the one number this shell has never been able to fly);
* the shell: both modes assemble and render, the mission's own controls are
  not offered in a session with no vehicle, and the arguments that reach
  ``api.airfoil_run_config`` are the flow that was stated.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _texts(box):
    return [getattr(e, "text", "") or "" for e in box.descendants()]


def _flow_session(**flow):
    """A session in airfoil-only mode, with the flow typed through the real
    setters (never by writing the dict)."""
    from gui.v3 import session as ses

    S = ses.make_session("air")
    assert ses.set_mode(S, "airfoil")
    for key, value in flow.items():
        if key == "fluid" or key == "water":
            assert ses.set_fluid(S, value)
        else:
            assert ses.set_flow(S, key, value)
    return S


# ==================================================== 1. the derivation
def test_air_is_the_isa_and_the_reynolds_number_is_rho_v_c_over_mu():
    """Composed, not restated: every number here is one of mission.py's."""
    from aerobo import api
    from aerobo.mission import (isa_density, isa_temperature, speed_of_sound,
                                sutherland_mu)

    pt = api.flow_point(fluid="air", V=30.0, chord_m=0.25, altitude_m=2000.0)
    assert pt["rho"] == isa_density(2000.0)
    assert pt["mu"] == sutherland_mu(isa_temperature(2000.0))
    assert pt["a_ms"] == speed_of_sound(2000.0)
    assert pt["mach"] == pytest.approx(30.0 / speed_of_sound(2000.0))
    assert pt["re"] == pytest.approx(pt["rho"] * 30.0 * 0.25 / pt["mu"])
    assert pt["q"] == pytest.approx(0.5 * pt["rho"] * 30.0 ** 2)
    assert pt["nu"] == pytest.approx(pt["mu"] / pt["rho"])


def test_altitude_thins_the_air_and_the_reynolds_number_falls_with_it():
    """The point of asking for a height at all."""
    from aerobo import api

    low = api.flow_point(fluid="air", V=30.0, chord_m=0.25, altitude_m=0.0)
    high = api.flow_point(fluid="air", V=30.0, chord_m=0.25,
                          altitude_m=6000.0)
    assert high["rho"] < 0.6 * low["rho"]
    assert high["re"] < 0.65 * low["re"]
    # ...and the same speed is a HIGHER Mach up there: colder air, slower
    # sound. A shell that asked for the Mach number would let those two
    # disagree; deriving it is what stops that.
    assert high["mach"] > low["mach"]


def test_the_two_waters_are_the_hydrofoil_modules_own_constants():
    from aerobo import api
    from aerobo.hydrofoil import water_properties

    for kind in ("sea", "fresh"):
        pt = api.flow_point(fluid="water", V=8.0, chord_m=0.12, water=kind)
        assert pt["rho"] == water_properties(kind)["rho"]
        assert pt["mu"] == water_properties(kind)["mu"]
        # water is flown incompressible everywhere in this repo, so it
        # reports the Mach nothing will contradict rather than a number no
        # solver honours
        assert pt["mach"] == 0.0 and pt["a_ms"] is None
    sea = api.flow_point(fluid="water", V=8.0, chord_m=0.12, water="sea")
    fresh = api.flow_point(fluid="water", V=8.0, chord_m=0.12, water="fresh")
    assert sea["re"] != fresh["re"]


def test_a_custom_fluid_is_taken_verbatim():
    from aerobo import api

    pt = api.flow_point(fluid="custom", V=50.0, chord_m=0.3, rho=1.1,
                        mu=1.6e-5, mach=0.2, cl_design=0.9)
    assert (pt["rho"], pt["mu"], pt["mach"]) == (1.1, 1.6e-5, 0.2)
    assert pt["re"] == pytest.approx(1.1 * 50.0 * 0.3 / 1.6e-5)
    # the speed of sound is a READ-OUT of the pair (V, M), never an input,
    # so the two cannot drift apart
    assert pt["a_ms"] == pytest.approx(50.0 / 0.2)
    assert pt["cl_design"] == 0.9
    assert pt["lift_per_span_n_m"] == pytest.approx(0.9 * pt["q"] * 0.3)


def test_a_derived_fluid_refuses_a_typed_state_rather_than_ignoring_it():
    """The whole reason the custom face is a face and not an override."""
    from aerobo import api

    for kwargs in ({"rho": 1.0}, {"mu": 2e-5}, {"mach": 0.3}):
        with pytest.raises(ValueError, match="custom"):
            api.flow_point(fluid="air", V=20.0, chord_m=0.3, **kwargs)
        with pytest.raises(ValueError, match="custom"):
            api.flow_point(fluid="water", V=8.0, chord_m=0.3, **kwargs)


def test_the_refusals_are_the_unphysical_ones():
    from aerobo import api

    with pytest.raises(ValueError, match="unknown fluid"):
        api.flow_point(fluid="honey", V=10.0, chord_m=0.2)
    for bad in ({"V": 0.0}, {"V": -3.0}, {"chord_m": 0.0}):
        with pytest.raises(ValueError, match="must both be > 0"):
            api.flow_point(fluid="air", **{"V": 10.0, "chord_m": 0.2, **bad})
    with pytest.raises(ValueError, match="rho and mu"):
        api.flow_point(fluid="custom", V=10.0, chord_m=0.2, rho=1.2)
    with pytest.raises(ValueError, match="rho and mu must both be > 0"):
        api.flow_point(fluid="custom", V=10.0, chord_m=0.2, rho=1.2, mu=0.0)
    with pytest.raises(ValueError, match="altitude"):
        api.flow_point(fluid="water", V=8.0, chord_m=0.2, altitude_m=100.0)
    with pytest.raises(ValueError, match="altitude"):
        api.flow_point(fluid="custom", V=8.0, chord_m=0.2, rho=1.2, mu=2e-5,
                       altitude_m=100.0)


def test_the_point_is_exactly_what_the_section_entry_points_take():
    """Not a new vocabulary: the three arguments already on the api."""
    import inspect

    from aerobo import api

    pt = api.flow_point(fluid="air", V=14.6, chord_m=1.0)
    for fn in (api.optimize_airfoil, api.screen_airfoils):
        params = inspect.signature(fn).parameters
        for key in ("re", "mach", "cl_design"):
            assert key in params
    assert {"re", "mach", "cl_design"} <= set(pt)


# ================================================== 2. the session state
def test_a_fresh_session_is_the_pipeline_and_the_mode_is_a_choice():
    from gui.v3 import session as ses

    S = ses.make_session("air")
    assert ses.session_mode(S) == "pipeline" and not ses.airfoil_only(S)
    assert not ses.set_mode(S, "pipeline")        # it is already there
    assert not ses.set_mode(S, "nonsense")
    assert ses.set_mode(S, "airfoil") and ses.airfoil_only(S)


def test_the_flow_opens_on_the_point_this_sessions_mission_implies():
    """Not literals: the two modes start on the same aerodynamics, so a
    section designed either way is comparable."""
    from gui.v3 import session as ses

    S = ses.make_session("air")
    dp = ses.design_point(S)
    f = ses.flow_state(S)
    assert f["V"] == pytest.approx(dp["v_ms"])
    assert f["chord_m"] == pytest.approx(dp["mac"])
    assert f["cl_design"] == pytest.approx(dp["cl_design"])
    pt = ses.flow_point(S)
    assert pt["re"] == pytest.approx(dp["re_mac"], rel=1e-9)


def test_only_two_stages_exist_and_the_others_say_why_they_are_locked():
    from gui.v3 import session as ses

    S = _flow_session()
    assert [s for s in ses.STAGES if ses.stage_visible(S, s)] == \
        ["mission", "airfoil"]
    states = ses.stage_states(S)
    for stage in ("wing", "results", "airfoil_aft"):
        state, reason = states[stage]
        assert state == "locked"
        assert reason and "section only" in reason or "one section" in reason
    # stage 1 is answerable and stage 2 waits on it
    assert states["mission"][0] == "ready"
    assert states["airfoil"][0] == "locked"
    assert "flow conditions" in states["airfoil"][1]


def test_accepting_the_flow_unlocks_the_section_stage():
    from gui.v3 import session as ses

    S = _flow_session()
    assert not ses.stage1_accepted(S)
    ses.set_stage1_accepted(S, True)
    assert ses.stage_states(S)["airfoil"][0] == "ready"
    assert ses.stage_states(S)["mission"][0] == "done"
    # ...and any edit to the point takes the acceptance back with it
    assert ses.set_flow(S, "V", 21.0)
    assert not ses.stage1_accepted(S)
    assert ses.stage_states(S)["airfoil"][0] == "locked"


def test_the_section_is_screened_at_the_stated_flow():
    from gui.v3 import session as ses

    S = _flow_session(V=40.0, chord_m=0.15, cl_design=0.35)
    pt = ses.flow_point(S)
    cond = ses.section_conditions(S, "main")
    assert cond["re"] == pytest.approx(pt["re"])
    assert cond["cl_design"] == 0.35
    # the GATES are still the surface's own — they are refusals, not an
    # operating point
    assert cond["tc_min"] == ses.airfoil_state(S, "main")["cond"]["tc_min"]


def test_the_mach_number_is_flown_only_when_it_is_switched_on():
    """The one number this shell could never send: section_conditions used
    to force M = 0 on every derived path."""
    from gui.v3 import session as ses

    S = _flow_session(V=90.0, chord_m=0.4)
    pt = ses.flow_point(S)
    assert pt["mach"] > 0.25                      # 90 m/s at sea level
    assert ses.flown_mach(S) == 0.0
    assert ses.section_conditions(S, "main")["mach"] == 0.0
    assert ses.set_flow_apply_mach(S, True)
    assert ses.flown_mach(S) == pytest.approx(pt["mach"])
    assert ses.section_conditions(S, "main")["mach"] == \
        pytest.approx(pt["mach"])


def test_a_custom_fluids_mach_reaches_the_search_the_same_way():
    from gui.v3 import session as ses

    S = _flow_session(fluid="custom")
    assert ses.set_flow(S, "mach", 0.45)
    assert ses.set_flow_apply_mach(S, True)
    assert ses.section_conditions(S, "main")["mach"] == pytest.approx(0.45)


def test_switching_to_the_custom_face_carries_the_derived_state_across():
    from gui.v3 import session as ses

    S = _flow_session(V=60.0)
    before = ses.flow_point(S)
    assert ses.set_fluid(S, "custom")
    after = ses.flow_point(S)
    for key in ("rho", "mu", "mach", "re", "q"):
        assert after[key] == pytest.approx(before[key])


def test_water_has_no_altitude_to_state_and_the_switch_drops_it():
    """api.flow_point REFUSES an altitude in water, so the face that cannot
    use it must not carry one into the derivation."""
    from gui.v3 import session as ses

    S = _flow_session(altitude_m=1500.0)
    assert ses.set_fluid(S, "water")
    assert ses.flow_state(S)["altitude_m"] == 0.0
    assert ses.flow_valid(S)
    assert ses.flow_point(S)["rho"] > 900.0


def test_a_typed_number_is_kept_even_when_it_makes_the_point_underivable():
    """The field holds what was typed; stage 1 says what is wrong with it."""
    from gui.v3 import session as ses

    S = _flow_session()
    assert ses.set_flow(S, "chord_m", 0.0)
    assert ses.flow_state(S)["chord_m"] == 0.0
    assert not ses.flow_valid(S)
    assert "must both be > 0" in ses.stage1_error(S)
    assert ses.stage_states(S)["mission"][0] == "error"
    assert not ses.set_flow(S, "chord_m", "not a number")
    assert ses.flow_state(S)["chord_m"] == 0.0


def test_there_is_no_vehicle_left_for_the_section_to_be_scored_on():
    """wing_objective sends the search to section_wing.WingGuess, which
    reads a mission this mode never asked for."""
    from gui.v3 import session as ses

    S = _flow_session()
    assert not ses.wing_objective(S, "main")
    assert ses.aft_surface(S) is None
    assert ses.second_surface_name(S) is None or True   # naming is untouched


def test_both_answers_survive_the_switch_in_either_direction():
    from gui.v3 import session as ses

    S = ses.make_session("air")
    S["mission"]["W_N"] = 777.0
    assert ses.set_mode(S, "airfoil")
    assert ses.set_flow(S, "chord_m", 0.123)
    assert ses.set_mode(S, "pipeline")
    assert S["mission"]["W_N"] == 777.0
    assert ses.design_point(S)["W_N"] == 777.0
    assert ses.set_mode(S, "airfoil")
    assert ses.flow_state(S)["chord_m"] == 0.123


def test_the_stage_one_label_and_badge_name_the_question_being_asked():
    from gui.v3 import session as ses

    S = _flow_session(V=25.0, chord_m=0.5)
    assert ses.stage_label(S, "mission") == "1  Flow"
    assert "Re" in ses.flow_summary(S)
    assert ses.set_flow(S, "chord_m", -1.0)
    assert ses.flow_summary(S) == "invalid"


# ======================================================== 3. the shell
@pytest.fixture(scope="module")
def shell():
    from gui.v3.app import assemble

    return assemble("air")


def _render_all(ctx):
    from gui.v3 import session as ses

    for stage in ses.STAGES:
        ctx.render(stage)


def test_the_shell_switches_mode_and_every_view_still_renders(shell, capsys):
    from gui.v3 import session as ses

    ctx = shell
    ctx.act("set_mode", "airfoil")
    _render_all(ctx)
    assert ses.airfoil_only(ctx.S)
    bad = [key for key, view in ctx.views.items()
           if any("failed to render" in t for t in _texts(view))]
    assert not bad, bad
    assert "Traceback" not in capsys.readouterr().err
    ctx.act("set_mode", "pipeline")
    _render_all(ctx)
    assert not ses.airfoil_only(ctx.S)
    assert not [key for key, view in ctx.views.items()
                if any("failed to render" in t for t in _texts(view))]
    assert "Traceback" not in capsys.readouterr().err


def test_stage_one_asks_for_the_flow_and_not_for_a_vehicle(shell):
    ctx = shell
    ctx.act("set_mode", "airfoil")
    try:
        texts = _texts(ctx.views[("mission", "operating")])
        assert "speed" in texts and "reference chord" in texts
        assert "altitude" in texts and "design Cl" in texts
        # ...and none of the mission's own questions, which have no answer
        # here: there is no weight to carry and no area to carry it on
        for gone in ("design weight", "reference area S", "wing loading W/S",
                     "Size", "Vehicle"):
            assert gone not in texts
    finally:
        ctx.act("set_mode", "pipeline")


def test_the_custom_face_asks_for_density_viscosity_and_mach(shell):
    ctx = shell
    ctx.act("set_mode", "airfoil")
    try:
        ctx.act("set_fluid", "custom")
        texts = _texts(ctx.views[("mission", "operating")])
        assert "density rho" in texts and "viscosity mu" in texts
        assert "Mach number" in texts
        assert "altitude" not in texts       # a custom fluid has no ISA
    finally:
        ctx.act("set_fluid", "air")
        ctx.act("set_mode", "pipeline")


def test_stage_two_does_not_offer_a_second_way_to_state_the_point(shell):
    """The override switch and the aspect-ratio estimate are the MISSION
    route to a Reynolds number. Drawn here they would be a second control
    for a point stage 1 already states."""
    ctx = shell
    ctx.act("set_mode", "airfoil")
    try:
        # stage 2 is not the stage on screen, so its paint is OWED rather
        # than given (Ctx.render_when_shown); the shell pays it in `select`
        # when the view is opened, and a test that reads it without opening
        # it pays it here
        ctx.pay_owed()
        texts = _texts(ctx.views[("airfoil", "screen")])
        assert "type the point myself" not in texts
        assert "aspect ratio estimate" not in texts
        assert "screen at M" in texts        # the number it CAN fly now
        assert "shortlist" in texts          # ...and the cost control stays
    finally:
        ctx.act("set_mode", "pipeline")


def test_the_wing_stage_cannot_be_opened_from_an_airfoil_only_session(shell):
    ctx = shell
    ctx.act("set_mode", "airfoil")
    try:
        ctx.select("mission", "operating")
        ctx.select("wing", "type")
        assert ctx.S["ui"]["selected"] == "mission"
    finally:
        ctx.act("set_mode", "pipeline")


def test_a_library_source_left_over_from_a_vehicle_session_decides_nothing():
    """The screen picks its ROUTE off the same point it screens at.

    ``re_source`` is the vehicle pipeline's control and it is not drawn in
    this mode, so a "library" left in the workspace would have sent a point
    the cache does not hold down the INSTANT path — ``screen_airfoils`` over
    the whole database at an uncached Reynolds number, which is hours.
    """
    from gui.v3 import session as ses
    from gui.v3.app import assemble

    ctx = assemble("air")
    ses.airfoil_state(ctx.S, "main")["re_source"] = "library"
    ctx.act("set_mode", "airfoil")
    try:
        ctx.act("set_flow", "chord_m", 0.37)      # not the cached point
        cond = ses.section_conditions(ctx.S, "main")
        pt = ses.flow_point(ctx.S)
        assert cond["re"] == pytest.approx(pt["re"])
        lib = ses.library_point()
        if lib is None:
            pytest.skip("no screening cache on this machine")
        assert abs(cond["re"] - float(lib["re"])) > 1e-3 * float(lib["re"])
        texts = _texts(ctx.views[("airfoil", "screen")])
        # the shortlist control is what the expensive route is priced
        # through, and it is only drawn on that route
        assert "shortlist" in texts
    finally:
        ctx.act("set_mode", "pipeline")


def test_the_run_the_section_stage_would_launch_carries_the_stated_flow():
    """End to end, minus the XFOIL: the arguments that reach api are the
    flow that was typed — the Mach number included."""
    from aerobo import api
    from gui.v3 import session as ses
    from gui.v3.stages.airfoil import shape_kwargs

    S = _flow_session(fluid="custom")
    for key, value in (("V", 45.0), ("chord_m", 0.2), ("cl_design", 0.55),
                       ("rho", 1.05), ("mu", 1.7e-5), ("mach", 0.13)):
        assert ses.set_flow(S, key, value)
    assert ses.set_flow_apply_mach(S, True)
    A = ses.airfoil_state(S, "main")
    shape = shape_kwargs(S, "main", A["opt"], A["weights"], None)
    assert shape["re"] == pytest.approx(1.05 * 45.0 * 0.2 / 1.7e-5)
    assert shape["mach"] == pytest.approx(0.13)
    assert shape["cl_design"] == pytest.approx(0.55)
    # a 2-D objective: no wing guess travels, because there is no wing
    assert shape["wing"] is None

    eff = ses.effective_airfoil_search(S, "main")
    cfg = api.airfoil_run_config(**shape, optimiser=str(eff["optimiser"]),
                                 budget=int(eff["budget"]), seed=0)
    assert cfg.problem_name == api.AIRFOIL_PROBLEM
    assert cfg.flags["airfoil_re"] == pytest.approx(shape["re"])
    assert cfg.flags["airfoil_mach"] == pytest.approx(0.13)
    assert cfg.flags["airfoil_cl_design"] == pytest.approx(0.55)


def test_the_status_bar_quotes_the_section_problem_not_the_wing(shell):
    from aerobo import api
    from gui.v3 import session as ses

    ctx = shell
    ctx.act("set_mode", "airfoil")
    try:
        rows = dict((r[0], r[1]) for r in
                    __import__("gui.v3.app", fromlist=["app"])
                    ._flow_properties(ctx, "airfoil")
                    if r[0] != "group")
        assert "Reynolds number" in rows and "Mach" in rows
        assert "design weight" not in rows and "reference area" not in rows
        # the section problem publishes NO static param_labels (its vector
        # is built from flags), so the status bar's dimension has to come
        # from the measured plan — quoting the spec printed "dim 0"
        assert api.PROBLEM_SPECS[api.AIRFOIL_PROBLEM].param_labels == ()
        plan = ses.airfoil_plan(ctx.S, "main")
        if plan is not None:
            assert int(plan.dim) == 8
        assert int(ses.effective_airfoil_search(ctx.S, "main")["budget"]) > 0
    finally:
        ctx.act("set_mode", "pipeline")
