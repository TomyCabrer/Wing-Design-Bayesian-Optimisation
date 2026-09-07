"""V4 — the two stages V3 does NOT have: 5 Controls and 6 Flight.

Two contracts are asserted here and they pull against each other, which is
why they share a file.

**V3 IS UNCHANGED.** The V3 shell has four stages and no fifth. Not "has
them hidden" — cannot see them: nothing in ``gui.v3`` names a controls stage
or a flight stage, its session dict carries neither workspace, and its
stage-module table would refuse to mount either. Every assertion in the
first block below fails the moment somebody adds a V4 line to a V3 file.

**V4 IS V3.** Not a fork of it. :mod:`gui.v4.app` executes ``gui/v3/app.py``'s
own source, so the tree, the tab strip and the breadcrumb are V3's code
reading V4's stage list — and a V3 chrome fix cannot fail to reach V4.
"""

import numpy as np
import pytest

from gui.v3 import app as v3app, session as v3session
from gui.v4 import app as v4app, session, stick as stk


def _fresh(**kw):
    return session.make_session(**kw)


@pytest.fixture()
def S():
    return _fresh()


# ------------------------------------------------------- V3 IS UNCHANGED

def test_v3_has_no_controls_stage_and_no_flight_stage():
    """The headline. V3 is the four-stage pipeline it has always been."""
    assert "controls" not in v3session.STAGES
    assert "flight" not in v3session.STAGES
    assert v3session.STAGES[-1] == "results"
    for table in (v3session.VIEWS, v3session.STAGE_LABELS,
                  v3app.STAGE_MODULES):
        assert "controls" not in table
        assert "flight" not in table


def test_a_v3_session_carries_no_v4_workspace():
    """Not merely absent from the tree — absent from the state, so nothing
    in V3 can accumulate a control surface it will never show."""
    S3 = v3session.make_session()
    assert "controls" not in S3
    assert "flight" not in S3
    assert set(S3["ui"]["tab"]) == set(v3session.STAGES)
    assert set(S3["ui"]["expanded"]) == set(v3session.STAGES)


def test_v3_never_reports_a_state_for_a_stage_it_does_not_have():
    S3 = v3session.make_session()
    S3["run"]["record"] = {"pretend": "a completed run"}
    states = v3session.stage_states(S3)
    assert set(states) == set(v3session.STAGES)


#: the identifiers that only exist because V4 does. ``"flight"`` on its own
#: is NOT one of them: V3 has had a ``flight`` builder choice (fixed vs free
#: flight state) since long before any of this, and a test that grepped for
#: the bare word would fail on V3's own vocabulary.
V4_ONLY = ("CONTROLS_DEFAULTS", "FLIGHT_DEFAULTS", "gui.v4", "gui/v4",
           "stages.controls", "stages.flight", "5  Controls", "6  Flight",
           "flightmodel", "sixdof", "aerobo.dynamics", "controls_rebuild",
           "flight_arm")


def test_no_v3_source_file_MENTIONS_the_v4_machinery():
    """The file-level version of the same claim, so a stray import or a
    leftover menu entry is caught even where no table lists it."""
    from pathlib import Path

    root = Path(v3session.__file__).resolve().parent
    offenders = []
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        text = path.read_text()
        offenders += [f"{path.name}: {n}" for n in V4_ONLY if n in text]
    assert not offenders, "V4 leaked into V3: " + "; ".join(offenders)


def test_the_leak_detector_can_actually_fire():
    """A grep test that matches nothing is not a test. Prove the needles
    find V4's own files before trusting their silence about V3's."""
    from pathlib import Path

    v4 = (Path(v3session.__file__).resolve().parent.parent / "v4")
    text = "\n".join(p.read_text() for p in v4.rglob("*.py")
                     if "__pycache__" not in p.parts)
    hits = [n for n in V4_ONLY if n in text]
    assert len(hits) >= 6, hits


# ------------------------------------------------------------ V4 IS V3

def test_v4_runs_v3s_own_shell_source_rather_than_a_copy_of_it():
    """The no-drift claim, asserted at the only place it can break."""
    shell = v4app._load_shell()
    assert shell.__file__ == v3app.__file__
    assert shell is not v3app
    assert shell.session is session
    assert shell.session is not v3session


def test_the_two_shells_do_not_share_a_stage_list():
    shell = v4app._load_shell()
    assert shell.session.STAGES == session.STAGES
    assert v3app.session.STAGES == v3session.STAGES
    assert len(session.STAGES) == len(v3session.STAGES) + 2


def test_v4_re_exports_every_name_v3s_session_defines():
    """A stage module written against ``session.X`` must not care which
    shell mounted it."""
    missing = [n for n in vars(v3session)
               if not n.startswith("__") and not hasattr(session, n)]
    assert not missing, missing


def test_the_first_five_stages_are_v3s_in_v3s_order():
    assert session.STAGES[:len(v3session.STAGES)] == v3session.STAGES
    for stage in v3session.STAGES:
        assert session.VIEWS[stage] is v3session.VIEWS[stage]


def test_adding_a_v4_view_cannot_add_a_tab_to_v3():
    """The tables are copies, not the same object."""
    assert session.VIEWS is not v3session.VIEWS
    assert session.STAGE_LABELS is not v3session.STAGE_LABELS


# ------------------------------------------------------------ the contract

def test_the_two_new_stages_are_in_the_pipeline_after_results():
    assert session.STAGES[-2:] == ("controls", "flight")
    assert session.STAGES.index("controls") > session.STAGES.index("results")


def test_every_stage_has_views_a_label_and_a_module():
    for stage in session.STAGES:
        assert stage in session.VIEWS, f"{stage} has no views"
        assert session.VIEWS[stage], f"{stage} has an empty view tuple"
        assert stage in session.STAGE_LABELS
        assert stage in v4app.STAGE_MODULES, f"{stage} would never mount"


def test_the_new_stages_map_to_their_own_modules():
    assert v4app.STAGE_MODULES["controls"] == "gui.v4.stages.controls"
    assert v4app.STAGE_MODULES["flight"] == "gui.v4.stages.flight"
    for stage in v3session.STAGES:
        assert v4app.STAGE_MODULES[stage].startswith("gui.v3.stages.")


def test_view_keys_are_unique_within_a_stage():
    for stage in ("controls", "flight"):
        keys = [k for k, _lbl, _icon in session.VIEWS[stage]]
        assert len(keys) == len(set(keys))


def test_the_default_ui_tab_exists_for_the_new_stages(S):
    for stage in ("controls", "flight"):
        assert S["ui"]["tab"][stage] == session.VIEWS[stage][0][0]


def test_both_new_stages_have_a_label(S):
    assert session.stage_label(S, "controls") == "5  Controls"
    assert session.stage_label(S, "flight") == "6  Flight"
    # ...and V3's own labelling rules still apply to V3's own stages
    assert session.stage_label(S, "wing") == v3session.stage_label(S, "wing")


# ------------------------------------------------------------- the gating

def test_both_new_stages_are_locked_before_a_run_AND_say_why(S):
    st = session.stage_states(S)
    for stage in ("controls", "flight"):
        state, reason = st[stage]
        assert state == "locked"
        assert reason, f"{stage} is locked with no reason given"
        assert len(reason) > 20, "a reason must be a sentence, not a word"


def test_flight_stays_locked_until_controls_has_built_a_deck(S):
    S["run"]["record"] = {"pretend": "a completed run"}
    S["run"]["report"] = {"geometry": {}, "breakdown": {}}
    st = session.stage_states(S)
    assert st["controls"][0] == "ready"
    assert st["flight"][0] == "locked"
    assert "stage 5" in st["flight"][1] or "control" in st["flight"][1]


def test_controls_reports_done_once_it_has_a_deck(S):
    S["run"]["record"] = {"pretend": "a completed run"}
    S["controls"]["deck"] = object()
    st = session.stage_states(S)
    assert st["controls"][0] == "done"
    assert st["flight"][0] == "ready"


def test_the_controls_lock_reason_names_the_zero_yaw_stiffness(S):
    """The measured fact the stage exists for has to reach the user."""
    S["run"]["record"] = {"pretend": "a completed run"}
    reason = session.stage_states(S)["controls"][1]
    assert "yaw" in reason.lower()
    assert "zero" in reason.lower()


def test_v3s_own_gates_are_not_disturbed(S):
    """V4 adds two entries; it must not change any of V3's five."""
    S["run"]["record"] = {"pretend": "a completed run"}
    mine = session.stage_states(S)
    theirs = v3session.stage_states(S)
    for stage in v3session.STAGES:
        assert mine[stage] == theirs[stage], stage


# ------------------------------------------------- airfoil-only has no vehicle

def test_an_airfoil_only_session_hides_AND_locks_both_new_stages(S):
    """Hiding is only cosmetic — Ctx.select gates on stage_states — so both
    halves have to be said, exactly as the wing and results stages do."""
    S["mode"] = "airfoil"
    if not session.airfoil_only(S):
        pytest.skip("this session has no airfoil-only switch at S['mode']")
    for stage in ("controls", "flight"):
        assert not session.stage_visible(S, stage)
        state, reason = session.stage_states(S)[stage]
        assert state == "locked"
        assert "section" in reason.lower()


# ------------------------------------------- a car wing is not flown at all

def test_a_car_wing_session_hides_AND_locks_both_new_stages():
    """The second way to have nothing to fly. A rear wing is bolted to a
    car: it makes a load rather than carrying one, nothing weighs it, and it
    has no free-flight degrees of freedom — so both halves are said here for
    the same reason the airfoil-only pair above says them."""
    S = _fresh(medium="track")
    assert v3session.car_wing(S)
    for stage in ("controls", "flight"):
        assert not session.stage_visible(S, stage)
        state, reason = session.stage_states(S)[stage]
        assert state == "locked"
        assert "car" in reason.lower()


def test_the_car_lock_survives_a_completed_run():
    """THE ORDERING TEST. The run-record branch's own reason invites the
    user in ("size and fly a wing first"), so a car branch written after it
    would let a finished car run open stage 5 and rebuild a downforce
    surface right way up as an aeroplane at a speed nobody chose."""
    S = _fresh(medium="track")
    S["run"]["record"] = {"pretend": "a completed run"}
    S["run"]["report"] = {"geometry": {}, "breakdown": {}}
    for stage in ("controls", "flight"):
        state, reason = session.stage_states(S)[stage]
        assert state == "locked"
        assert "car" in reason.lower()


def test_the_air_session_still_gets_both_stages():
    """THE POSITIVE CONTROL. Without it a predicate that is accidentally
    true everywhere deletes stage 5 and stage 6 from every session and every
    other claim in this file still passes."""
    S = _fresh(medium="air")
    assert v3session.airfoil_only(S) is False and not v3session.car_wing(S)
    assert session.free_flight(S) is True
    for stage in ("controls", "flight"):
        assert session.stage_visible(S, stage) is True
        assert "car" not in session.stage_states(S)[stage][1].lower()


@pytest.mark.parametrize("medium", ["water", "track"])
def test_a_craft_that_does_not_fly_free_gets_NEITHER_stage(medium):
    """The rule the two of them share, and the reason it is one predicate.

    A car rear wing is bolted to a car; a foiling craft is held by the water
    and driven by a rig. Neither has a free-flight degree of freedom, so
    neither has a control surface to cut or an aeroplane to fly, and the
    pipeline ends at stage 4 for both."""
    S = _fresh(medium=medium)
    assert session.free_flight(S) is False
    for stage in ("controls", "flight"):
        assert session.stage_visible(S, stage) is False
        state, reason = session.stage_states(S)[stage]
        assert state == "locked"
        # ...and the node SAYS why. A greyed stage with no explanation is
        # the thing this shell exists to avoid.
        assert ("car" if medium == "track" else "foiling") in reason.lower()
        assert "stage 4 is where" in reason.lower()


def test_the_foiler_lock_names_the_measurement_that_justifies_it():
    """Not "a foiler is different" — the number. Its only vertical surface
    is the mast, whose quarter chord stands AHEAD of the CG, so the yaw
    stiffness has the wrong SIGN and no criterion in this package buys it
    back (RESULTS_SESSION74_LATERAL_EVERYWHERE.md)."""
    S = _fresh(medium="water")
    reason = session.stage_states(S)["flight"][1]
    assert "mast" in reason.lower() and "-0.196" in reason


def test_a_foiling_craft_refuses_to_OPEN_either_stage():
    """Hiding is cosmetic; ``Ctx.select`` is the half that refuses. The car
    already had this test and the foiler must not be one lock short of it."""
    ctx = v4app.assemble("water")
    for stage in ("controls", "flight"):
        ctx.select(stage)
        assert ctx.S["ui"]["selected"] != stage


def test_the_tree_drops_both_stages_on_a_foiling_craft():
    ctx = v4app.assemble("water")
    shell = v4app._load_shell()
    keys = [n["key"] for n in shell._nodes(ctx)]
    assert "controls" not in keys and "flight" not in keys
    assert "results" in keys, "the water pipeline still ends at stage 4"


def test_switching_the_medium_to_water_takes_both_stages_away():
    """The LIVE path, and it must not be one-way."""
    ctx = v4app.assemble()
    assert all(session.stage_visible(ctx.S, s) for s in ("controls",
                                                         "flight"))
    ctx.act("set_medium", "water")
    for stage in ("controls", "flight"):
        assert not session.stage_visible(ctx.S, stage)
        assert session.stage_states(ctx.S)[stage][0] == "locked"
    assert session.stage_visible(ctx.S, ctx.S["ui"]["selected"])
    ctx.act("set_medium", "air")
    assert all(session.stage_visible(ctx.S, s) for s in ("controls",
                                                         "flight"))


def test_the_car_rule_does_not_disturb_v3s_own_visibility():
    """V4's override answers for its own two stages and DELEGATES for the
    rest: on the track V3 hides the second surface's and the fin's section
    stages and SHOWS the endplate's, and a rule written one branch too wide
    would take the endplate's stage with it."""
    S = _fresh(medium="track")
    for stage in v3session.STAGES:
        assert session.stage_visible(S, stage) == \
            v3session.stage_visible(S, stage), stage


def test_v3s_own_gates_are_not_disturbed_on_the_track():
    """The new predicate lives in a V3 file; it must not move stages 1-4."""
    S = _fresh(medium="track")
    S["run"]["record"] = {"pretend": "a completed run"}
    mine, theirs = session.stage_states(S), v3session.stage_states(S)
    for stage in v3session.STAGES:
        assert mine[stage] == theirs[stage], stage


@pytest.mark.parametrize("medium", ["air", "water", "track"])
@pytest.mark.parametrize("mode", ["pipeline", "airfoil"])
def test_the_predicate_is_the_car_and_not_the_mode(medium, mode):
    """A section-only session in the track medium is a SECTION session: the
    airfoil-only lock is the one that fires, and its reason is about the
    section and not about a car."""
    S = _fresh(medium=medium)
    S["mode"] = mode
    assert v3session.car_wing(S) == (medium == "track"
                                     and not v3session.airfoil_only(S))


def test_the_tree_drops_both_stages_on_a_car_wing():
    """The shell-level half: the two stages are not nodes of the tree a car
    session paints, and stage 4 still is."""
    ctx = v4app.assemble("track")
    shell = v4app._load_shell()
    keys = [n["key"] for n in shell._nodes(ctx)]
    assert "controls" not in keys and "flight" not in keys
    assert "results" in keys, "the track pipeline still ends at stage 4"


def test_a_car_wing_refuses_to_open_stage_6():
    """Hiding is cosmetic; ``Ctx.select`` is the half that refuses."""
    ctx = v4app.assemble("track")
    ctx.select("flight")
    assert ctx.S["ui"]["selected"] != "flight"


def test_switching_the_medium_to_track_takes_both_stages_away():
    """The LIVE path, and it must not be one-way: the workspaces stay in the
    session dict, so switching back has to bring the stages back with them."""
    ctx = v4app.assemble()
    assert all(session.stage_visible(ctx.S, s) for s in ("controls",
                                                         "flight"))
    ctx.act("set_medium", "track")
    for stage in ("controls", "flight"):
        assert not session.stage_visible(ctx.S, stage)
        assert session.stage_states(ctx.S)[stage][0] == "locked"
    assert session.stage_visible(ctx.S, ctx.S["ui"]["selected"])
    ctx.act("set_medium", "air")
    assert all(session.stage_visible(ctx.S, s) for s in ("controls",
                                                         "flight"))


# ------------------------------------------------------------ session reset

def test_a_new_session_does_not_inherit_the_previous_controls():
    a = _fresh()
    a["controls"]["aileron"]["span_from"] = 0.11
    a["flight"]["live"]["mass_kg"] = 4321.0
    b = _fresh()
    assert b["controls"]["aileron"]["span_from"] == \
        session.CONTROLS_DEFAULTS["aileron"]["span_from"]
    assert b["flight"]["live"]["mass_kg"] is None


def test_the_defaults_are_deep_copied_not_shared():
    """A shallow copy would let one session edit another's nested dict."""
    a, b = _fresh(), _fresh()
    a["controls"]["vertical"]["ventral"] = True
    assert b["controls"]["vertical"]["ventral"] is False
    assert session.CONTROLS_DEFAULTS["vertical"]["ventral"] is False


def test_defaults_cover_every_key_the_stages_read():
    for key in ("aileron", "elevator", "vertical", "thrust"):
        assert key in session.CONTROLS_DEFAULTS
    for key in ("live", "altitude_m", "stall_on",
                "stick", "hold", "keys_down"):
        assert key in session.FLIGHT_DEFAULTS
    # ONE QUESTION, ONE PLACE: the thrust LINE is a property of the airframe
    # and belongs with the control surfaces, not among a flight's levers.
    assert "thrust_z_m" not in session.FLIGHT_DEFAULTS
    assert set(session.CONTROLS_DEFAULTS["thrust"]) == {
        "through_cg", "z_below_cg_m"}
    for k in ("elevator", "aileron", "rudder"):
        assert k in session.FLIGHT_DEFAULTS["stick"]


def test_the_fin_is_on_by_default():
    """A fin is the difference between a flyable model and one with no
    yaw stiffness at all, so it is the one surface that is not a choice.
    (The flap is not a choice either any more — it is gone, and
    tests/test_the_flap_is_not_a_control.py says so.)"""
    assert session.CONTROLS_DEFAULTS["vertical"]["on"] is True


# ------------------------------------- mass, thrust and CG are LIVE, not setup

def test_the_fly_tab_is_the_first_thing_stage_6_opens_on():
    """You open a flight simulator on the aeroplane, not on a form."""
    assert session.VIEWS["flight"][0][0] == "fly"


def test_mass_thrust_and_cg_are_live_levers_and_not_setup_fields():
    """The user's ask, asserted where it can break.

    They live in ``F["live"]``, which the Fly view owns; the setup tab is
    the flight CONDITION only. A top-level ``mass_kg`` would be a field on a
    form again.
    """
    assert set(session.FLIGHT_DEFAULTS["live"]) == {"thrust_n", "mass_kg",
                                                    "x_cg_m"}
    for key in ("mass_kg", "x_cg_m", "thrust_n"):
        assert key not in session.FLIGHT_DEFAULTS, \
            f"{key} is a live lever, not a set-up field"


def test_the_live_levers_start_unanswered_so_the_design_flies_itself():
    """None means "whatever the design says" — the first frame is always the
    aeroplane the search produced, not a number typed into a form."""
    assert all(v is None for v in _fresh()["flight"]["live"].values())


def test_no_setup_field_is_a_live_lever_under_another_name():
    """The set-up tab must not re-ask a question the Fly tab owns —
    one question, one place."""
    import inspect

    from gui.v4.stages import flight

    src = inspect.getsource(flight)
    setup = src[src.index("def _render_setup"):src.index("def _render_fly")]
    for word in ("mass [kg]", "x_cg", "thrust [N]"):
        assert word not in setup, f"{word!r} is asked on the set-up tab"


# ---------------------------------------------------------- the keyboard

def test_the_stage_maps_the_keys_the_user_asked_for():
    """Arrow keys pitch and yaw, A/D roll, W/S throttle."""
    assert stk.key_axis("ArrowUp")[0] == "elevator"
    assert stk.key_axis("ArrowRight")[0] == "rudder"
    assert stk.key_axis("a")[0] == "aileron"
    assert stk.key_axis("w") == ("thrust", +1)
    assert stk.key_axis("s") == ("thrust", -1)


# --------------------------------------------- the deck must not need a nudge

def _armed(capsys=None):
    """A V4 shell with a real design behind it and stage 5's deck built."""
    from aerobo import api

    ctx = v4app.assemble()
    cfg = api.RunConfig(problem_name="tail", budget=4, seed=0)
    built = api.PROBLEM_SPECS["tail"].build({}, {}, None)
    ctx.S["run"]["record"] = {"pretend": "a completed run"}
    ctx.S["run"]["report"] = api.design_report(cfg, built.bounds.mean(axis=1))
    return ctx


# ------------------------------------ a foiler has no stage 5 and no stage 6

def test_stage_5_offers_ONE_tab_list_and_it_does_not_depend_on_the_medium():
    """Stage 5's tab list used to be cut down on water — three of the five,
    because nothing on a foiler is hinged. The stage is gone there now, so
    there is one list again and no session can be looking at a tab its
    medium does not own."""
    air = _armed()
    keys = [k for k, _l, _i in session.views_of(air.S, "controls")]
    assert keys == ["surfaces", "propulsion", "vertical", "derivatives"]
    assert air.S["ui"]["tab"]["controls"] == "surfaces"
    for medium in ("air", "water", "track"):
        S = _fresh(medium=medium)
        assert [k for k, _l, _i in session.views_of(S, "controls")] == keys


def test_no_v4_workspace_key_survives_that_only_a_foiler_answered():
    """The foiler's two levers (its speed, and where the rider stands) were
    stage-5 state. A stored key nothing can reach is exactly the class of
    dead control the "where" menu was — it stored a word no code read."""
    assert "water" not in session.CONTROLS_DEFAULTS
    S = _fresh(medium="water")
    assert "water" not in S["controls"]


def test_no_v4_stage_module_still_branches_on_the_medium():
    """The stage modules cannot decide anything per-medium any more,
    because the only medium they are ever mounted for is air. A leftover
    branch is dead code that reads as a supported configuration."""
    from pathlib import Path

    from gui.v4 import stages as v4stages

    root = Path(v4stages.__file__).resolve().parent
    for path in sorted(root.glob("*.py")):
        src = path.read_text()
        assert 'medium") == "water"' not in src, path.name
        assert "_water()" not in src, path.name


def test_a_manoeuvre_is_offered_only_where_its_AXIS_exists():
    """A doublet written into a missing axis flies nothing while still
    printing a mode's numbers under its name."""
    from gui.v4 import manoeuvre as mv

    stabilator_only = {"alpha", "beta", "p", "q", "r", "elevator"}
    flyable = {m.key for m in mv.MANOEUVRES if m.flyable_on(stabilator_only)}
    assert flyable == {"phugoid", "spiral", "short_period"}
    bare = {"alpha", "beta", "p", "q", "r"}
    assert {m.key for m in mv.MANOEUVRES if m.flyable_on(bare)} == \
        {"phugoid", "spiral"}
    full = stabilator_only | {"aileron", "rudder"}
    assert {m.key for m in mv.MANOEUVRES if m.flyable_on(full)} == \
        {m.key for m in mv.MANOEUVRES}


def test_opening_the_derivatives_view_BUILDS_the_deck(capsys):
    """A view must not depend on having been edited first.

    The deck used to appear only as a side effect of the stage's edit()
    handler, so a user who opened stage 5 and went straight to Derivatives
    saw "no deck yet" for ever and had to nudge an unrelated field to make
    the stage work. Drive the render path directly and assert the deck
    exists afterwards.
    """
    ctx = _armed()
    assert ctx.S["controls"].get("deck") is None      # nothing built yet
    ctx.render("controls", "derivatives")
    assert "failed to render" not in capsys.readouterr().err
    assert ctx.S["controls"].get("deck") is not None, \
        "the derivatives view rendered without building a deck"
    assert ctx.S["controls"]["deck"].Cn_beta > 0.0    # a fin is fitted


def test_the_stability_check_view_also_builds_its_own_deck(capsys):
    ctx = _armed()
    ctx.render("controls", "derivatives")
    assert "failed to render" not in capsys.readouterr().err
    assert ctx.S["controls"].get("deck") is not None


# ------------------------------------------------- the live levers, driven

def test_arming_fills_the_levers_from_the_design_itself(capsys):
    ctx = _armed()
    ctx.render("controls", "derivatives")
    assert ctx.act("flight_arm") is True, ctx.S["flight"].get("error")
    L, ac = ctx.S["flight"]["live"], ctx.S["flight"]["ac"]
    assert L["mass_kg"] == pytest.approx(ac.inertia.mass_kg)
    assert L["x_cg_m"] == pytest.approx(ac.deck.x_cg)
    assert L["thrust_n"] == pytest.approx(ac.prop.thrust_n)
    capsys.readouterr()


def test_moving_the_mass_lever_moves_the_aeroplane_not_just_the_label(capsys):
    ctx = _armed()
    ctx.render("controls", "derivatives")
    ctx.act("flight_arm")
    was = ctx.S["flight"]["ac"].inertia
    ctx.act("flight_set_live", "mass_kg", was.mass_kg * 2.0)
    now = ctx.S["flight"]["ac"].inertia
    assert now.mass_kg == pytest.approx(2.0 * was.mass_kg)
    # inertia scales with it, or a heavy aeroplane rolls like a light one
    assert now.Ixx == pytest.approx(2.0 * was.Ixx)
    capsys.readouterr()


def test_moving_the_cg_lever_REBUILDS_the_deck_and_moves_the_margin(capsys):
    """The CG is a moment arm, not a label. Aft is less margin."""
    ctx = _armed()
    ctx.render("controls", "derivatives")
    ctx.act("flight_arm")
    ac = ctx.S["flight"]["ac"]
    cg0, mac = float(ac.deck.x_cg), float(ac.deck.mac)
    sm0 = float(ac.deck.static_margin)
    ctx.act("flight_set_live", "x_cg_m", cg0 + 0.3 * mac)
    sm1 = float(ctx.S["flight"]["ac"].deck.static_margin)
    assert ctx.S["flight"]["ac"].deck.x_cg == pytest.approx(cg0 + 0.3 * mac)
    assert sm1 < sm0 - 1e-6, "moving the CG aft did not cost static margin"
    capsys.readouterr()


def test_a_live_lever_does_NOT_rebuild_the_view(capsys):
    """Rebuilding the Fly view mid-flight destroys the scene group and takes
    the typed field's focus with it. A lever repaints the readout only."""
    ctx = _armed()
    ctx.render("controls", "derivatives")
    ctx.act("flight_arm")
    calls = []
    original = ctx.renderers[("flight", "fly")]
    ctx.renderers[("flight", "fly")] = lambda: (calls.append(1), original())
    ctx.act("flight_set_live", "thrust_n", 123.0)
    assert calls == [], "a live lever rebuilt the whole Fly view"
    assert ctx.S["flight"]["ac"].prop.thrust_n == pytest.approx(123.0)
    capsys.readouterr()


def test_the_keyboard_deflects_the_stick_and_releasing_it_centres(capsys):
    """End to end through the stage's own handler: key down, frames, key up.

    Asserted as a SIGN and a return to centre, not a magnitude — the
    magnitude is gui.v4.stick's, and it is tested there.
    """
    from types import SimpleNamespace as NS

    ctx = _armed()
    ctx.render("controls", "derivatives")
    ctx.act("flight_arm")
    F = ctx.S["flight"]
    F["running"] = True

    def press(name, down):
        ctx.act("flight_key", NS(key=NS(name=name),
                                 action=NS(keydown=down, keyup=not down)))

    press("ArrowUp", True)                      # pitch up
    assert F["hold"]["elevator"] == -1
    for _ in range(10):
        ctx.act("flight_advance", 1 / 60)
    assert F["stick"]["elevator"] < -1.0, "holding up did not deflect"
    press("ArrowUp", False)
    assert F["hold"]["elevator"] == 0
    for _ in range(60):
        ctx.act("flight_advance", 1 / 60)
    assert F["stick"]["elevator"] == 0.0, "the stick did not spring back"
    F["running"] = False
    capsys.readouterr()


def test_the_browser_half_is_pointed_at_the_REAL_objects(capsys):
    """Two kinds of id, and they are not interchangeable.

    ``scene`` and ``hud`` are nicegui ELEMENT ids — integers, and what
    ``getElement`` takes. ``group`` and ``world`` are SCENE-OBJECT ids, which
    are uuid strings keying the scene's own map. Hand either one the other's
    value and every lookup returns undefined: no error, no warning, and a
    picture that simply never moves.
    """
    ctx = _armed()
    ctx.render("controls", "derivatives")
    ctx.render("flight", "fly")
    p = ctx.act("flight_probe")
    inst = p["install"]
    assert inst is not None, "nothing was installed at all"
    assert isinstance(inst["scene"], int) and isinstance(inst["hud"], int)
    assert isinstance(inst["group"], str) and isinstance(inst["world"], str)
    assert inst["scene"] != inst["hud"]
    # ...and the two object ids are ids the scene actually holds
    assert inst["group"] in p["object_ids"]
    assert inst["world"] in p["object_ids"]
    assert inst["group"] != inst["world"]
    assert "failed to render" not in capsys.readouterr().err


def test_a_held_key_does_not_become_a_message_storm(capsys):
    """The stick is a DEMAND held in Python: keydown sets it, keyup clears
    it, and nothing in between is read. So the operating system's auto-repeat
    — about 30 events a second per held key, each a websocket round trip and
    a handler call — buys exactly nothing.

    ``repeating=False`` drops ONLY ``evt.repeat`` keydowns (keyboard.js), so
    the first press and the release still arrive. Asserted here because
    getting it wrong the other way is silent: the stick would latch on and
    never centre.
    """
    from types import SimpleNamespace as NS

    from nicegui import ui

    built = []
    original = ui.keyboard
    try:
        ui.keyboard = lambda *a, **kw: (built.append(kw), original(*a, **kw))[1]
        ctx = _armed()
        ctx.render("controls", "derivatives")
        ctx.render("flight", "fly")
    finally:
        ui.keyboard = original
    assert built, "the fly view built no keyboard at all"
    assert all(kw.get("repeating") is False for kw in built), built

    # ...and ONE keydown, held, still reaches full deflection and still
    # centres on the keyup — which is what would break if the flag also
    # suppressed the first press or the release
    F = ctx.S["flight"]
    F["running"] = True

    def press(name, down):
        ctx.act("flight_key", NS(key=NS(name=name),
                                 action=NS(keydown=down, keyup=not down)))

    press("ArrowUp", True)
    for _ in range(60):
        ctx.act("flight_advance", 1 / 60)
    assert F["stick"]["elevator"] < -1.0, F["stick"]["elevator"]
    press("ArrowUp", False)
    for _ in range(60):
        ctx.act("flight_advance", 1 / 60)
    assert F["stick"]["elevator"] == 0.0
    F["running"] = False
    capsys.readouterr()


def test_a_key_does_nothing_while_the_simulation_is_stopped(capsys):
    """Or the arrow keys would hijack the page whenever stage 6 is open."""
    from types import SimpleNamespace as NS

    ctx = _armed()
    ctx.render("controls", "derivatives")
    ctx.act("flight_arm")
    ctx.S["flight"]["running"] = False
    ctx.act("flight_key", NS(key=NS(name="ArrowUp"),
                             action=NS(keydown=True, keyup=False)))
    assert ctx.S["flight"]["hold"]["elevator"] == 0
    capsys.readouterr()


def test_the_throttle_key_moves_thrust_and_STAYS(capsys):
    ctx = _armed()
    ctx.render("controls", "derivatives")
    ctx.act("flight_arm")
    F = ctx.S["flight"]
    F["running"] = True
    t0 = float(F["live"]["thrust_n"])
    F["hold"]["thrust"] = +1
    for _ in range(20):
        ctx.act("flight_advance", 1 / 60)
    t1 = float(F["live"]["thrust_n"])
    assert t1 > t0
    F["hold"]["thrust"] = 0
    for _ in range(60):
        ctx.act("flight_advance", 1 / 60)
    assert F["live"]["thrust_n"] == pytest.approx(t1), \
        "the throttle sprang back — it is a throttle, not a stick"
    F["running"] = False
    capsys.readouterr()


def test_stopping_lets_go_of_every_held_key(capsys):
    """A key still down when Pause is pressed would be held for ever,
    because its keyup lands on a stopped loop."""
    ctx = _armed()
    ctx.render("controls", "derivatives")
    ctx.act("flight_arm")
    ctx.S["flight"]["hold"].update({"aileron": +1, "thrust": -1})
    ctx.act("flight_run", False)
    assert set(ctx.S["flight"]["hold"].values()) == {0}
    capsys.readouterr()


# ------------------------------- the picture: mission point, and the CG

def test_the_frame_chain_puts_a_level_aeroplane_the_right_way_up():
    """Closed form, no browser. The loft is drawn in the lattice's frame
    (x AFT, y starboard, z up) and the attitude arrives in body axes (x
    forward, z DOWN); handing the scene one without the other is a picture
    that is right in level flight and mirrored in every manoeuvre.
    """
    import numpy as np

    from aerobo import sixdof as sd
    from gui.v4.stages import flight as fl

    level = sd.State().quat
    nose = fl.scene_point(level, (-1.0, 0.0, 0.0))     # -x in the loft frame
    stbd = fl.scene_point(level, (0.0, 1.0, 0.0))
    up = fl.scene_point(level, (0.0, 0.0, 1.0))
    assert nose[0] > 0.9, "the nose is not pointing forwards"
    assert up[2] > 0.9, "the aeroplane is upside down"
    assert abs(stbd[2]) < 1e-9, "a level wing is not level"
    # ...and the two fixed rotations are ROTATIONS, not mirrors
    for R in (fl.A_G_TO_BODY, fl.M_EARTH_TO_SCENE):
        assert np.linalg.det(np.asarray(R)) == pytest.approx(1.0)


def test_a_right_bank_draws_the_right_wing_DOWN():
    """The bug this frame note exists for. With C^T handed straight to the
    scene, a right bank drew the right wing UP and nobody could see it in a
    still picture."""
    import numpy as np

    from gui.v4.stages import flight as fl

    phi = np.deg2rad(30.0)
    q = np.array([np.cos(phi / 2), np.sin(phi / 2), 0.0, 0.0])   # roll right
    stbd = fl.scene_point(q, (0.0, 1.0, 0.0))
    assert stbd[2] < -0.4, "a right bank did not put the right wing down"


def test_a_nose_up_pitch_raises_the_nose_on_screen():
    import numpy as np

    from gui.v4.stages import flight as fl

    th = np.deg2rad(20.0)
    q = np.array([np.cos(th / 2), 0.0, np.sin(th / 2), 0.0])
    nose = fl.scene_point(q, (-1.0, 0.0, 0.0))
    assert nose[2] > 0.2, "pitching up drove the nose down the screen"


def test_the_airframe_offset_puts_the_cg_on_the_pivot():
    """An aeroplane rotates about its CG, so the group's origin has to BE
    the CG — which means the airframe moves and the marker does not."""
    import numpy as np

    from gui.v4.stages import flight as fl

    for cg, z in ((0.0, 0.0), (0.35, 0.0), (-0.2, 0.1)):
        t = np.asarray(fl.body_offset(cg, z))
        landed = t + np.asarray(fl.A_G_TO_BODY) @ np.array([cg, 0.0, z])
        assert np.allclose(landed, 0.0, atol=1e-12), \
            "the CG did not land on the group origin"


def test_the_fly_tab_opens_already_trimmed_at_the_mission_point(capsys):
    """The user's ask: open stage 6 and the aeroplane is THERE, at the
    conditions stage 1 stated — not a form waiting to be filled in."""
    ctx = _armed()
    ctx.render("controls", "derivatives")
    ctx.render("flight", "fly")
    assert "failed to render" not in capsys.readouterr().err
    F = ctx.S["flight"]
    assert F.get("error") is None, F.get("error")
    assert F.get("ac") is not None, "nothing was armed on opening the tab"
    assert F.get("state") is not None
    probe = ctx.act("flight_probe")
    dp = v3session.design_point(ctx.S)
    assert probe["V"] == pytest.approx(float(dp["v_ms"]))
    assert probe["rho"] == pytest.approx(float(dp["rho"]))
    # and it is TRIMMED there, not merely placed there
    assert F["trim"]["V"] == pytest.approx(float(dp["v_ms"]))
    assert abs(F["state"].alpha) < 0.35


def test_the_simulation_flies_the_missions_air_not_sea_level(capsys):
    """A deck built at 1.225 for a design sized at altitude — or in water —
    is a different aeroplane, and stage 6 would fly that one.

    Asserted at 3000 m, where the mission's own density is 0.909 and the
    sea-level default is 1.225. At the default altitude the two agree to
    seven figures, so a test written there would pass over the bug.
    """
    ctx = _armed()
    ctx.S["mission"]["altitude_m"] = 3000.0
    rho = float(v3session.design_point(ctx.S)["rho"])
    assert rho < 1.0, "this test needs a density unlike sea level"
    ctx.act("controls_rebuild")
    assert ctx.S["controls"]["fm"].aircraft.rho == pytest.approx(rho)
    capsys.readouterr()


def test_the_cg_and_the_neutral_point_are_BOTH_drawn(capsys):
    """"CG should be visible" — as a marker in the picture, with the neutral
    point beside it so the gap is the static margin to scale."""
    ctx = _armed()
    ctx.render("controls", "derivatives")
    ctx.render("flight", "fly")
    probe = ctx.act("flight_probe")
    assert "CG" in probe["labels"]
    assert "NP" in probe["labels"]
    assert probe["x_np"] is not None
    capsys.readouterr()


def test_moving_the_cg_lever_MOVES_THE_AIRFRAME_in_the_picture(capsys):
    """Not a number that changes somewhere off screen. The marker is the
    pivot, so the aeroplane slides under it."""
    ctx = _armed()
    ctx.render("controls", "derivatives")
    ctx.render("flight", "fly")
    before = ctx.act("flight_probe")["body_xyz"]
    cg0 = float(ctx.S["flight"]["live"]["x_cg_m"])
    ctx.act("flight_set_live", "x_cg_m", cg0 + 0.25)
    after = ctx.act("flight_probe")["body_xyz"]
    assert after != before, "the CG lever did not move anything on screen"
    assert after[0] == pytest.approx(before[0] + 0.25)
    capsys.readouterr()


def test_the_scene_carries_the_flown_attitude_not_a_default_one(capsys):
    """The group's rotation on open is the trimmed attitude, so the wing is
    shown at the incidence it actually flies at."""
    import numpy as np

    from gui.v4.stages import flight as fl

    ctx = _armed()
    ctx.render("controls", "derivatives")
    ctx.render("flight", "fly")
    q = np.asarray(ctx.act("flight_probe")["payload"]["q"], dtype=float)
    x, y, z, w = q
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    want = fl.scene_rotation(ctx.S["flight"]["state"].quat)
    # the report rounds to six decimals on purpose — the browser cannot see
    # the difference and the digits are paid for thirty times a second — so
    # the round trip through a quaternion is exact to about 1e-6, not to 1e-12
    assert np.allclose(R, want, atol=1e-5)


# ------------------------------------------------- the clock, which was wrong

def _fly(ctx, seconds, hz):
    """Fly ``seconds`` of wall clock in ``hz`` frames per second."""
    n = int(round(seconds * hz))
    for _ in range(n):
        ctx.act("flight_advance", 1.0 / hz)
    return ctx.S["flight"]


def test_one_second_of_WALL_CLOCK_is_one_second_of_flight(capsys):
    """The bug this test exists for: the stage stored dt = 1/60 and ticked
    at 1/30, so it bought a sixtieth of a second of flight per thirtieth of
    a second of real time. The aeroplane flew at HALF SPEED and the rate
    selector marked "1x" was really 0.5x — silently, for as long as it
    shipped, because nothing anywhere compared the two numbers.
    """
    ctx = _armed()
    ctx.render("controls", "derivatives")
    ctx.act("flight_arm")
    F = _fly(ctx, 1.0, 60)
    assert F["t"] == pytest.approx(1.0, abs=2 * (1 / 240)), \
        f"one second of wall clock bought {F['t']:.4f} s of flight"
    capsys.readouterr()


@pytest.mark.parametrize("hz", [20, 30, 60, 144])
def test_the_frame_RATE_does_not_change_how_far_it_flies(hz, capsys):
    """A fixed integrator substep behind a wall clock, so the physics is the
    same on a fast machine and a slow one. Counting ticks made the aeroplane
    fly at a speed set by the frame rate."""
    ctx = _armed()
    ctx.render("controls", "derivatives")
    ctx.act("flight_arm")
    F = _fly(ctx, 2.0, hz)
    assert F["t"] == pytest.approx(2.0, abs=2 * (1 / 240))
    capsys.readouterr()


def test_two_frame_rates_reach_the_SAME_STATE(capsys):
    """Not just the same clock — the same aeroplane, in the same place."""
    import numpy as np

    out = {}
    for hz in (30, 120):
        ctx = _armed()
        ctx.render("controls", "derivatives")
        ctx.act("flight_arm")
        out[hz] = _fly(ctx, 3.0, hz)["state"]
    a, b = out[30], out[120]
    assert a.V == pytest.approx(b.V, rel=1e-6)
    assert a.altitude_m == pytest.approx(b.altitude_m, abs=1e-3)
    assert np.allclose(a.pos, b.pos, atol=1e-3)
    capsys.readouterr()


def test_the_rate_selector_is_a_REAL_multiplier(capsys):
    ctx = _armed()
    ctx.render("controls", "derivatives")
    ctx.act("flight_arm")
    ctx.S["flight"]["speed"] = 2.0
    F = _fly(ctx, 1.0, 60)
    assert F["t"] == pytest.approx(2.0, abs=2 * (1 / 240)), \
        "the 2x selector did not fly twice as far"
    capsys.readouterr()


def test_a_frame_that_took_for_ever_does_not_integrate_the_whole_gap(capsys):
    """A tab left in the background for a minute must not come back and fly
    a minute in one frame — that is a freeze AND a 14 000-step loop."""
    from gui.v4.stages import flight as fl

    ctx = _armed()
    ctx.render("controls", "derivatives")
    ctx.act("flight_arm")
    ctx.act("flight_advance", 60.0)
    assert ctx.S["flight"]["t"] <= fl.MAX_CATCHUP_S + 1e-9, \
        "a 60 s stall was integrated in full"
    assert ctx.S["flight"]["t"] > 0.0, "and it must still fly SOMETHING"
    capsys.readouterr()


def test_the_trace_carries_its_own_TIMES_not_a_nominal_dt(capsys):
    """The frames are not evenly spaced, so t = index * dt stretches every
    trace. Sampled once a frame, timestamped from the simulation clock."""
    ctx = _armed()
    ctx.render("controls", "derivatives")
    ctx.act("flight_arm")
    F = _fly(ctx, 0.5, 60)
    assert len(F["history_t"]) == len(F["history"])
    assert F["history_t"] == sorted(F["history_t"])
    assert F["history_t"][-1] == pytest.approx(F["t"], abs=1e-9)
    capsys.readouterr()


def test_history_is_sampled_per_FRAME_not_per_SUBSTEP(capsys):
    """240 Hz of history is four times the data for none of the picture."""
    ctx = _armed()
    ctx.render("controls", "derivatives")
    ctx.act("flight_arm")
    F = _fly(ctx, 1.0, 60)
    assert 55 <= len(F["history"]) <= 66, len(F["history"])
    capsys.readouterr()


def test_pausing_does_not_BANK_the_time_it_was_paused_for(capsys):
    """_run(False) drops the clock, so resuming starts from now."""
    import time

    ctx = _armed()
    ctx.render("controls", "derivatives")
    ctx.act("flight_arm")
    ctx.act("flight_run", True)
    ctx.act("flight_advance")          # the real clock path, first frame
    ctx.act("flight_run", False)
    time.sleep(0.05)
    ctx.act("flight_run", True)
    before = float(ctx.S["flight"]["t"])
    ctx.act("flight_advance")          # first frame after resuming
    assert ctx.S["flight"]["t"] == pytest.approx(before, abs=1e-9), \
        "resuming flew the gap it was paused for"
    ctx.act("flight_run", False)
    capsys.readouterr()


# --------------------------------------------- the game view, and the ground

def test_the_stage_opens_in_GAME_mode_on_the_aeroplane():
    assert session.FLIGHT_DEFAULTS["mode"] == "game"
    assert session.FLIGHT_DEFAULTS["hud"] is True


def test_the_engineering_view_SURVIVES_as_a_toggle(capsys):
    """A game view that deleted the CG and neutral-point markers would have
    thrown away the thing stage 6 was built for. Both live in one scene."""
    ctx = _armed()
    ctx.render("controls", "derivatives")
    ctx.render("flight", "fly")
    probe = ctx.act("flight_probe")
    assert "CG" in probe["labels"] and "NP" in probe["labels"]
    ctx.act("flight_set_mode", "engineering")
    assert ctx.S["flight"]["mode"] == "engineering"
    ctx.act("flight_set_mode", "game")
    assert ctx.S["flight"]["mode"] == "game"
    capsys.readouterr()


def test_the_mode_toggle_does_NOT_rebuild_the_scene(capsys):
    """Rebuilding would re-download the loft and re-create two hundred ground
    objects to answer a question about visibility."""
    ctx = _armed()
    ctx.render("controls", "derivatives")
    ctx.render("flight", "fly")
    before = ctx.act("flight_probe")["object_ids"]
    calls = []
    original = ctx.renderers[("flight", "fly")]
    ctx.renderers[("flight", "fly")] = lambda: (calls.append(1), original())
    ctx.act("flight_set_mode", "engineering")
    assert calls == [], "the mode toggle rebuilt the whole view"
    assert ctx.act("flight_probe")["object_ids"] == before
    capsys.readouterr()


def test_the_world_is_hidden_in_the_engineering_view_and_shown_in_game(capsys):
    ctx = _armed()
    ctx.render("controls", "derivatives")
    ctx.render("flight", "fly")
    ctx.act("flight_set_mode", "game")
    p = ctx.act("flight_probe")
    assert p["world_visible"] is True and p["cg_visible"] is False
    ctx.act("flight_set_mode", "engineering")
    p = ctx.act("flight_probe")
    assert p["world_visible"] is False and p["cg_visible"] is True
    capsys.readouterr()


def test_every_ground_lattice_PERIOD_divides_the_snap(capsys):
    """Or the floating origin will not map the scenery onto itself and the
    world will jump once a tile. This is the one thing that ties
    gui.v4.world's periodicity guarantee to what is actually drawn."""
    from gui.v4.stages import flight as fl

    for pitch in (fl.GRID_PITCH_M, fl.POST_PITCH_M):
        q = fl.WORLD_SNAP_M / pitch
        assert q == pytest.approx(round(q)), \
            f"{pitch} m does not divide the {fl.WORLD_SNAP_M} m snap"
    del capsys


def test_flying_forward_SCROLLS_the_world_group(capsys):
    """The whole sense of speed. One message a frame, and it has to move."""
    ctx = _armed()
    ctx.render("controls", "derivatives")
    ctx.render("flight", "fly")
    before = ctx.act("flight_probe")["world_xyz"]
    _fly(ctx, 2.0, 60)
    ctx.act("flight_pose")
    after = ctx.act("flight_probe")["world_xyz"]
    assert after != before, "two seconds of flight moved no scenery"
    assert ctx.S["flight"]["state"].pos[0] > 10.0
    capsys.readouterr()


def test_the_ground_STOPS_the_aeroplane(capsys):
    """It is not a suggestion. Kept out of sixdof on purpose — the engine is
    shared with the trim solver and with linearise's two-sided Jacobian."""
    ctx = _armed()
    ctx.render("controls", "derivatives")
    ctx.act("flight_arm")
    F = ctx.S["flight"]
    # start it low and push the nose down. The stick is SPRUNG, so writing a
    # deflection into F["stick"] would be undone on the next frame — the
    # pilot input is the HOLD, exactly as the keyboard sets it.
    F["state"].pos[2] = -25.0
    F["live"]["thrust_n"] = 0.0
    F["hold"]["elevator"] = +1               # nose down
    F["running"] = True
    for _ in range(4000):
        ctx.act("flight_advance", 1 / 60)
        if F.get("crashed"):
            break
    assert F["crashed"] is True, \
        f"flew through the ground to {F['state'].altitude_m:.1f} m"
    assert F["running"] is False
    assert "GROUND CONTACT" in (F.get("error") or "")
    assert F["state"].altitude_m <= 0.0
    capsys.readouterr()


def test_re_arming_clears_the_crash(capsys):
    ctx = _armed()
    ctx.render("controls", "derivatives")
    ctx.act("flight_arm")
    ctx.S["flight"]["crashed"] = True
    ctx.act("flight_arm")
    assert ctx.S["flight"]["crashed"] is False
    assert ctx.S["flight"]["error"] is None
    capsys.readouterr()


# --------------------------------------------- the frame is not a Vue patch

def test_a_frame_writes_no_nicegui_element(capsys):
    """THE LAG.

    The read-out used to be written with ``label.text = ...`` once a frame.
    That is a nicegui element update, and an element update re-patches a page
    holding all six stages at once. Measured in a real visible Chrome while
    flying, by dropping one socket message kind at a time at
    ``socket.onevent``:

        nothing dropped        97.7 Hz   p90  9.4   p99 33.8   max 41.5 ms
        drop the updates      119.2 Hz   p90  9.2   p99  9.4
        drop run_javascript    98.3 Hz   p90 16.5   p99 33.6
        drop both             120.0 Hz   p90  9.0   p99  9.3   max  9.4 ms

    Ten element updates a second (they coalesce) cost a quarter of every
    frame; forty pose pushes a second cost nothing measurable. So the numbers
    must leave this process on the frame payload and nowhere else.
    """
    ctx = _armed()
    ctx.render("controls", "derivatives")
    ctx.render("flight", "fly")
    seen = set()
    for _ in range(30):
        # the frame's own order: fly, then read the numbers off ONE
        # evaluation of the equations of motion, then report
        ctx.act("flight_advance", 1 / 30)
        ctx.act("flight_readout")
        p = ctx.act("flight_probe")
        seen.add(p["text_state"]["readout"])
        assert p["label_text"] == ("", ""), \
            f"a frame handed nicegui {p['label_text']!r}"
    assert len(seen) > 1, "the read-out never changed — nothing was flown"
    assert "SM " in next(iter(seen))
    assert "failed to render" not in capsys.readouterr().err


def test_the_frame_payload_carries_the_numbers_it_no_longer_writes(capsys):
    """...and they reach the browser THROUGH the payload, not beside it."""
    ctx = _armed()
    ctx.render("controls", "derivatives")
    ctx.render("flight", "fly")
    ctx.act("flight_advance", 1 / 30)
    ctx.act("flight_readout")
    pay = ctx.act("flight_payload")
    assert "txt" in pay, "the frame does not carry the read-out at all"
    assert set(pay["txt"]) == {"readout", "banner", "color", "thrust"}
    assert "SM " in pay["txt"]["readout"]
    assert pay["txt"]["thrust"].endswith((" N auto)", " N set)"))
    ins = ctx.act("flight_probe")["install"]
    for k in ("readout", "banner", "thrust"):
        assert isinstance(ins[k], int)
    assert len({ins["scene"], ins["hud"], ins["readout"], ins["banner"],
                ins["panel"], ins["thrust"]}) == 6
    assert "failed to render" not in capsys.readouterr().err


# ------------------------------------------------------ re-trim and reset

def test_re_trim_and_reset_does_not_rebuild_the_view(capsys):
    """"Reset and trim doesn't work".

    It called ``ctx.render("flight", "fly")``, which clears the box and
    builds a SECOND ``ui.scene`` — a fresh WebGL context and a re-fetch and
    re-parse of the aircraft mesh — and, because ``_render_fly`` tears the
    loop down before it repaints, it also stopped the simulation. Object
    identity is what tells a repaint from a rebuild.
    """
    ctx = _armed()
    ctx.render("controls", "derivatives")
    ctx.render("flight", "fly")
    before = ctx.act("flight_probe")
    ctx.act("flight_run", True)
    for _ in range(10):
        ctx.act("flight_advance", 1 / 30)
    flown = ctx.S["flight"]["t"]
    assert flown > 0.0

    ctx.act("flight_reset")
    after = ctx.act("flight_probe")
    assert after["install"] == before["install"], "the view was rebuilt"
    assert after["object_ids"] == before["object_ids"], "the scene was rebuilt"
    # ...and it actually reset
    assert ctx.S["flight"]["t"] == 0.0, "the clock did not go back"
    assert len(ctx.S["flight"]["history"]) == 1
    # ...and it is still flying, because the pilot did not press Pause
    assert after["running"] is True and after["has_timer"] is True
    assert "failed to render" not in capsys.readouterr().err


def test_reset_after_a_crash_flies_again(capsys):
    """The ground stopped the loop, the pilot did not — so a reset after a
    crash is a request to fly again, not to sit still looking at a stopped
    aeroplane with no way back except Fly."""
    ctx = _armed()
    ctx.render("controls", "derivatives")
    ctx.render("flight", "fly")
    ctx.act("flight_run", True)
    F = ctx.S["flight"]
    import numpy as np
    from dataclasses import replace
    # altitude is -pos[2], so a positive z IS below the ground
    F["state"] = replace(F["state"], pos=np.array([0.0, 0.0, 5.0]))
    for _ in range(400):
        ctx.act("flight_advance", 1 / 30)
        if F.get("crashed"):
            break
    if not F.get("crashed"):
        pytest.skip("this design did not reach the ground in 13 s")
    assert ctx.act("flight_probe")["has_timer"] is False
    ctx.act("flight_reset")
    p = ctx.act("flight_probe")
    assert p["crashed"] is False and p["running"] is True
    assert p["has_timer"] is True, "reset left the aeroplane on the ground"
    assert "failed to render" not in capsys.readouterr().err


def test_a_failed_re_trim_still_puts_the_error_on_screen(capsys):
    """The one case that DOES need the view back: there is an error card to
    show and nothing left to fly."""
    ctx = _armed()
    ctx.render("controls", "derivatives")
    ctx.render("flight", "fly")
    from aerobo import sixdof as sd

    def boom(*a, **kw):
        raise RuntimeError("no trim here")

    orig, sd.trim_level = sd.trim_level, boom
    try:
        ctx.act("flight_reset")
    finally:
        sd.trim_level = orig
    assert "no trim here" in (ctx.S["flight"].get("error") or "")
    capsys.readouterr()


def test_installing_into_a_view_that_was_never_BUILT_is_a_no_op(capsys):
    """A reachable half-built view, not a defensive nicety.

    Opening stage 6 with no deck behind it shows a hint and builds nothing —
    no scene, no glass, no labels — and ``_render_fly`` clears every handle
    on its way in. Reaching for ``readout.id`` there is an AttributeError
    inside a render, which the shell turns into "failed to render" and an
    empty stage.
    """
    ctx = v4app.assemble()                    # no design at all
    ctx.render("flight", "fly")
    assert ctx.S["flight"].get("ac") is None
    ctx.act("flight_install")                 # must not raise
    assert ctx.act("flight_probe")["install"] is None
    assert "failed to render" not in capsys.readouterr().err


def test_the_keyboards_demand_is_visible_somewhere(capsys):
    """The stick panel used to say "the keyboard moves these" over sliders
    that nothing updates — a Quasar slider is Vue-owned, so tracking the
    keyboard on it is one element update a frame, which is the cost the
    frame path was rebuilt to stop paying. So the three surface angles are
    on the LEFT panel, which travels with the frame and costs nothing.
    """
    import inspect
    from types import SimpleNamespace as NS

    from gui.v4.stages import flight

    src = inspect.getsource(flight)
    assert "the keyboard moves these" not in src

    ctx = _armed()
    ctx.render("controls", "derivatives")
    ctx.render("flight", "fly")
    ctx.act("flight_readout")
    quiet = ctx.act("flight_probe")["gauge_state"]["text"]
    assert quiet["g-st-elevator"] == "+0.0"

    # the keyboard is ignored while nothing is in the air, so that the arrow
    # keys still scroll the page — put it in the air first
    ctx.act("flight_run", True)
    ctx.act("flight_key", NS(key=NS(name="ArrowUp"), action=NS(keydown=True,
                                                              keyup=False)))
    for _ in range(20):
        ctx.act("flight_advance", 1 / 30)
    ctx.act("flight_readout")
    held = ctx.act("flight_probe")["gauge_state"]
    assert held["text"] != quiet, "a held key changed nothing the pilot sees"
    assert abs(float(held["text"]["g-st-elevator"])) > 1.0, held["text"]
    # ...and the BAR moved with the number, off centre in the pulled-up
    # direction: a number that moves over a bar that does not is a bar that
    # is decoration
    assert float(held["attr"]["g-stbar-elevator"]["width"]) > 1.0
    assert "failed to render" not in capsys.readouterr().err


def test_the_read_out_does_not_REPEAT_the_glass_or_the_panels(capsys):
    """"There are some things redundant from HUD."

    Speed and altitude are the two HUD tapes, alpha and the load factor are
    on the glass, the heading is its compass and pitch is its ladder; bank,
    roll rate and the three stick angles are the left-hand panel; thrust is
    the right-hand one. All of that used to be on the read-out as well —
    four rows of about thirty numbers, most of them on screen twice.

    What is left is what nothing else shows: the clock, the SIDESLIP (the
    glass has no beta, and the panel's rows are ATTITUDE, not the
    aerodynamic angles), and the three levers whose point is that moving one
    changes the static margin printed beside it.
    """
    ctx = _armed()
    _flying(ctx)
    ctx.act("flight_advance", 1 / 30)
    ctx.act("flight_readout")
    p = ctx.act("flight_probe")
    line = p["text_state"]["readout"]
    hud, panel = p["payload"]["hud"], p["gauge_state"]["text"]

    for word in ("V ", "alt ", " a ", "roll ", "pitch ", "yaw ",
                 "thrust", " n ", "stick", " p ", " q ", " r "):
        assert word not in line, \
            f"{word!r} is on the read-out AND on the glass or a panel"
    # the things it must still carry, each because NOTHING else has them
    for word in ("t ", "sideslip", "mass", "x_cg", "SM"):
        assert word in line, f"{word!r} vanished with the duplicates"
    assert len(line.splitlines()) == 2

    # ...and the duplicates really are elsewhere, so this test is about
    # where a number lives and not about deleting numbers
    assert hud is not None and "ias" in hud and "alt" in hud
    assert "thr_n" in hud, "the thrust left the panel and never reached the glass"
    for key in ("g-ang-roll", "g-ang-pitch", "g-ang-yaw", "g-rate-roll",
                "g-rate-pitch", "g-rate-yaw", "g-st-rudder"):
        assert key in panel
    capsys.readouterr()


# ============================================================================
# THE TWO PANELS BESIDE THE PICTURE, AND THE THROTTLE THAT HAD NOWHERE TO SHOW
# ============================================================================

def _flying(ctx):
    """Armed, rendered and running, which is what the keyboard needs."""
    ctx.render("controls", "derivatives")
    ctx.render("flight", "fly")
    ctx.act("flight_arm")
    ctx.S["flight"]["running"] = True
    return ctx.S["flight"]


def test_W_moves_the_thrust_AND_THE_GLASS_says_so(capsys):
    """THE REPORTED BUG, at the place it was actually wrong.

    Holding W moved ``F["live"]["thrust_n"]`` all along — the aeroplane
    accelerated. What did not move was anything the pilot could see: the
    slider and the number box beside the picture are Vue-owned and cannot be
    written once a frame (that is the element update measured in
    gui/v4/live.py, a quarter of every frame), so they went on reporting the
    trim value while the aeroplane flew at 62 % more thrust.

    Measured before the fix: 21.60 N -> 35.10 N of physics, 21.60 N on the
    panel. The value now lives on the GLASS, beside the throttle bar it
    fills, which the frame path writes for nothing.
    """
    ctx = _armed()
    F = _flying(ctx)
    t0 = float(F["live"]["thrust_n"])
    F["hold"]["thrust"] = +1
    for _ in range(60):
        ctx.act("flight_advance", 1 / 60)
        ctx.act("flight_readout")
    t1 = float(F["live"]["thrust_n"])
    assert t1 > t0 * 1.2, "the throttle key did not move the thrust at all"
    hud = ctx.act("flight_probe")["payload"]["hud"]
    assert hud["thr_n"] == f"{t1:,.0f}", \
        f"the aeroplane is at {t1:.1f} N and the glass says {hud['thr_n']}"
    capsys.readouterr()


def test_the_throttle_has_no_widget_left_to_go_stale(capsys):
    """"In the right only select Min thrust and Max."

    There is no slider and no number box for the thrust VALUE any more, and
    that is the point: the one that existed could not follow the keyboard
    without an element update a frame, so it lied for as long as a key was
    held. W and S are the throttle, the glass is the read-out, and the two
    fields on the right are the stops they run between.
    """
    ctx = _armed()
    _flying(ctx)
    fields = ctx.act("flight_probe")["lever_values"]
    assert "thrust_n" not in fields, \
        "a throttle widget is back, and nothing can keep it in step"
    assert {"mass_kg", "x_cg_m", "V_trim", "altitude_m"} <= set(fields)
    capsys.readouterr()


def test_the_two_STOPS_are_the_ones_the_keyboard_uses(capsys):
    """"on the right it should be displayed max and min thrust available".

    They are asked on the right, and they are the same two numbers
    ``advance_throttle`` is handed — not a slider's range and not a constant
    computed somewhere else. Pinned by holding W into the stop.
    """
    ctx = _armed()
    F = _flying(ctx)
    lo, hi = ctx.act("flight_probe")["thrust_band"]
    assert hi > lo
    F["hold"]["thrust"] = +1
    for _ in range(600):                       # ten seconds on a 4 s throttle
        ctx.act("flight_advance", 1 / 60)
    assert float(F["live"]["thrust_n"]) == pytest.approx(hi), \
        "the throttle did not stop where the panel says it stops"
    capsys.readouterr()


def test_the_stops_are_ANSWERABLE_and_the_answer_reaches_the_keyboard(capsys):
    """A calibration is a default, not a ban. 2.5x the trim thrust is a
    guess about a propulsion system this stage does not model, so it has to
    be a number the pilot can type over — and typing it has to move the stop
    the KEY runs into, not just the caption."""
    ctx = _armed()
    F = _flying(ctx)
    _lo, hi0 = ctx.act("flight_probe")["thrust_band"]
    F["thrust_max_n"] = 500.0
    ctx.act("flight_readout")
    lo, hi = ctx.act("flight_probe")["thrust_band"]
    assert hi == pytest.approx(500.0) and hi != pytest.approx(hi0)
    F["hold"]["thrust"] = +1
    for _ in range(600):
        ctx.act("flight_advance", 1 / 60)
    assert float(F["live"]["thrust_n"]) > hi0, \
        "the raised ceiling did not reach stick.advance_throttle"
    capsys.readouterr()


def test_the_throttle_FLOOR_is_zero_and_is_not_a_field(capsys):
    """"Max thrust which the user can change (min thrust = 0)".

    The floor is FIXED. There is no propeller in this stage — thrust is a
    force along body x — so a stopped engine makes none and not a negative
    one, and the second field that used to ask for a reverse-thrust floor is
    gone from the panel AND from the session. Writing the old key does
    nothing, which is the assertion that catches a half-removal.
    """
    ctx = _armed()
    F = _flying(ctx)
    assert "thrust_min_n" not in session.FLIGHT_DEFAULTS
    assert ctx.act("flight_probe")["thrust_band"][0] == 0.0
    F["thrust_min_n"] = -40.0                  # the key that no longer exists
    ctx.act("flight_readout")
    assert ctx.act("flight_probe")["thrust_band"][0] == 0.0, \
        "a dead key still moves the floor — the field was only half removed"
    # ...and holding S runs the throttle down to it and stops there
    F["hold"]["thrust"] = -1
    for _ in range(600):
        ctx.act("flight_advance", 1 / 60)
    assert float(F["live"]["thrust_n"]) == pytest.approx(0.0)
    capsys.readouterr()


def test_the_glass_bar_fills_between_ZERO_and_the_ANSWERED_ceiling(capsys):
    """The bar is a fraction of the band, so raising the ceiling with the
    thrust unchanged has to SHORTEN it. A bar that ignored the ceiling would
    draw the same height for a 40 N engine and a 4000 N one."""
    ctx = _armed()
    F = _flying(ctx)
    ctx.act("flight_readout")
    tall = ctx.act("flight_probe")["payload"]["hud"]["thr_h"]
    F["thrust_max_n"] = 10.0 * ctx.act("flight_probe")["thrust_band"][1]
    ctx.act("flight_readout")
    p = ctx.act("flight_probe")
    assert p["thrust_band"][0] == 0.0
    assert p["payload"]["hud"]["thr_h"] < tall, \
        "the same thrust drew the same bar over a band ten times as wide"
    # ...and the NUMBER on the glass is the thrust, not a fraction of it
    assert p["payload"]["hud"]["thr_n"] == \
        f"{float(F['live']['thrust_n']):,.0f}"
    capsys.readouterr()


def test_the_panel_carries_ALL_THREE_angles_AND_all_three_rates(capsys):
    """"On the left we should also see the Pitch and Yaw the same as roll."

    Three rows of identical shape — angle, centred rate bar, rate — because
    the glass says what the attitude IS and nothing about which way it is
    going. On a design with a divergent spiral, ten degrees of bank rolling
    back to level and ten degrees rolling away from it are the whole
    question and they look identical on a bank pointer.
    """
    ctx = _armed()
    F = _flying(ctx)
    # HELD, not written into the stick: the spring pulls a written
    # deflection back toward centre on the very next frame, so a test that
    # set it directly banked 1.6 degrees in a second and a half
    F["hold"]["aileron"] = +1
    F["hold"]["rudder"] = +1
    for _ in range(120):
        ctx.act("flight_advance", 1 / 60)
    ctx.act("flight_readout")
    st = F["state"]
    g = ctx.act("flight_probe")["gauge_state"]
    roll, pitch, yaw = (float(np.rad2deg(v)) for v in st.euler)
    assert abs(roll) > 5.0, "it never banked"
    assert g["text"]["g-ang-roll"] == f"{roll:+.1f}"
    assert g["text"]["g-ang-pitch"] == f"{pitch:+.1f}"
    assert g["text"]["g-ang-yaw"] == f"{yaw % 360.0:5.1f}"
    for i, key in enumerate(("roll", "pitch", "yaw")):
        assert g["text"][f"g-rate-{key}"] == \
            f"{np.rad2deg(st.rates[i]):+.1f}"
        assert "width" in g["attr"][f"g-ratebar-{key}"]
    assert g["attr"]["g-wings"]["transform"].startswith("rotate(")
    capsys.readouterr()


def test_the_three_rates_are_the_BODY_rates_and_not_euler_derivatives(capsys):
    """p, q, r — what the equations integrate and what the damping
    derivatives act on. They agree with the Euler-angle rates only at wings
    level, so the check is made in a bank."""
    ctx = _armed()
    F = _flying(ctx)
    F["hold"]["aileron"] = +1
    for _ in range(150):
        ctx.act("flight_advance", 1 / 60)
    ctx.act("flight_readout")
    g = ctx.act("flight_probe")["gauge_state"]
    assert abs(float(np.rad2deg(F["state"].euler[0]))) > 20.0
    for i, key in enumerate(("roll", "pitch", "yaw")):
        assert float(g["text"][f"g-rate-{key}"]) == pytest.approx(
            float(np.rad2deg(F["state"].rates[i])), abs=0.05)
    capsys.readouterr()


def test_a_right_bank_tilts_the_bank_panels_right_wing_DOWN(capsys):
    """The same claim the scene makes, on the panel. A positive roll is
    right-wing-down, and an SVG rotate by a positive angle turns clockwise
    on a y-down canvas — so the sign goes straight through, and getting it
    backwards is invisible in every number and obvious on screen.
    """
    from gui.v4 import gauge as gg

    s = gg.attitude_state(roll_deg=+30.0, pitch_deg=0.0, yaw_deg=0.0,
                          rates_dps=(0.0, 0.0, 0.0))
    ang = float(s["attr"]["g-wings"]["transform"].split("(")[1].split()[0])
    assert ang == pytest.approx(30.0), \
        "the wing bar rotates against the bank — the picture is mirrored"
    del capsys


def test_a_rate_bar_pins_at_its_end_and_the_number_does_not(capsys):
    """A bar that runs off its track has stopped meaning anything, so it
    pins and reddens; the NUMBER carries the excess, because the excess is
    the interesting part."""
    from gui.v4 import gauge as gg

    def at(p, q, r):
        return gg.attitude_state(roll_deg=0.0, pitch_deg=0.0, yaw_deg=0.0,
                                 rates_dps=(p, q, r))

    fast = at(400.0, 0.0, 0.0)
    edge = at(gg.RATE_FULL_DPS["roll"], 0.0, 0.0)
    assert fast["attr"]["g-ratebar-roll"]["width"] == \
        edge["attr"]["g-ratebar-roll"]["width"]
    assert fast["attr"]["g-ratebar-roll"]["fill"] != \
        at(1.0, 0.0, 0.0)["attr"]["g-ratebar-roll"]["fill"]
    assert fast["text"]["g-rate-roll"] == "+400.0"
    del capsys


def test_each_rate_bar_is_scaled_by_ITS_OWN_axis(capsys):
    """Roll rates are several times the other two on anything with
    ailerons. One shared scale would leave the pitch and yaw bars
    permanently dead, which is a bar that says nothing."""
    from gui.v4 import gauge as gg

    assert gg.RATE_FULL_DPS["roll"] > gg.RATE_FULL_DPS["pitch"]
    same = gg.attitude_state(roll_deg=0.0, pitch_deg=0.0, yaw_deg=0.0,
                             rates_dps=(20.0, 20.0, 20.0))
    w = {k: float(same["attr"][f"g-ratebar-{k}"]["width"])
         for k in ("roll", "pitch", "yaw")}
    assert w["pitch"] > w["roll"], \
        "the same rate drew the same bar on two different axes"
    assert w["pitch"] == pytest.approx(w["yaw"])
    del capsys


def test_a_frame_writes_no_PANEL_markup_either(capsys):
    """The same rule the read-out is under, for the two new panels.

    A ``ui.html`` assignment is innerHTML — 203 nodes and 14.8 kB when the
    HUD did it, 497.6 kB/s of the wire. Both panels are a skeleton drawn
    ONCE and written into by id, so the markup must be byte-identical after
    a hundred frames of flying.
    """
    from gui.v4 import gauge as gg

    ctx = _armed()
    F = _flying(ctx)
    before = ctx.act("flight_probe")["panel_html"]
    assert before == gg.attitude_skeleton()
    F["stick"]["aileron"] = 10.0
    F["hold"]["thrust"] = +1
    for _ in range(100):
        ctx.act("flight_advance", 1 / 60)
        ctx.act("flight_readout")
        assert ctx.act("flight_probe")["panel_html"] == before, \
            "a frame re-wrote a panel's markup"
    capsys.readouterr()


def test_the_frame_payload_carries_the_panel(capsys):
    ctx = _armed()
    _flying(ctx)
    ctx.act("flight_advance", 1 / 30)
    ctx.act("flight_readout")
    pay = ctx.act("flight_payload")
    assert set(pay["g"]) == {"text", "attr"}
    capsys.readouterr()


def test_the_installer_carries_the_panel_id(capsys):
    """A nicegui ELEMENT id, like the glass and the read-out — and it has to
    be the id of the panel that was just built, not of one a previous render
    left behind."""
    ctx = _armed()
    ctx.render("controls", "derivatives")
    ctx.render("flight", "fly")
    ids = ctx.act("flight_probe")["install"]
    assert "panel" in ids
    assert len({ids["panel"], ids["hud"], ids["readout"],
                ids["banner"]}) == 4
    assert all(isinstance(ids[k], int)
               for k in ("panel", "hud", "readout", "banner"))
    capsys.readouterr()


# ------------------------------------------- the two levers that RE-TRIM

def test_speed_is_a_lever_and_it_TRIMS_there(capsys):
    """"Add a button on the right to change altitude and speed to a certain
    value (same as weight)."

    Same widget as the weight lever, different semantics, and the difference
    is the point: an aeroplane shoved to 20 m/s at the elevator that trimmed
    it at 14.6 simply flies back. So the condition is answered and the
    aircraft is TRIMMED at it — alpha, elevator and thrust all move.
    """
    ctx = _armed()
    F = _flying(ctx)
    v0 = float(F["trim"]["V"])
    thrust0 = float(F["live"]["thrust_n"])
    ctx.act("flight_set_condition", "V_trim", v0 + 5.0)
    assert float(F["trim"]["V"]) == pytest.approx(v0 + 5.0)
    assert float(F["state"].V) == pytest.approx(v0 + 5.0, rel=1e-6)
    assert float(F["live"]["thrust_n"]) != pytest.approx(thrust0), \
        "the re-trim kept the old thrust, so it is not trimmed at all"
    # ...and the trimmed aeroplane HOLDS that speed with nothing touched
    for _ in range(120):
        ctx.act("flight_advance", 1 / 60)
    assert float(F["state"].V) == pytest.approx(v0 + 5.0, rel=0.02)
    capsys.readouterr()


def test_the_speed_lever_puts_its_own_thrust_ON_THE_GLASS(capsys):
    ctx = _armed()
    F = _flying(ctx)
    ctx.act("flight_set_condition", "V_trim", float(F["trim"]["V"]) + 4.0)
    hud = ctx.act("flight_probe")["payload"]["hud"]
    assert hud["thr_n"] == f"{float(F['live']['thrust_n']):,.0f}"
    capsys.readouterr()


def test_a_speed_that_cannot_be_TRIMMED_keeps_the_last_one_and_says_why(
        capsys):
    """A refused answer must not leave the stage with no aeroplane in it.
    The previous condition goes back and the reason stays on screen — the
    alternative is a blank picture and a stage that has to be reloaded."""
    ctx = _armed()
    F = _flying(ctx)
    good = float(F["trim"]["V"])
    ctx.act("flight_set_condition", "V_trim", 1e-6)
    assert F.get("ac") is not None and F.get("state") is not None
    assert float(F["trim"]["V"]) == pytest.approx(good) or F.get("error")
    capsys.readouterr()


def test_altitude_is_a_lever_and_the_aeroplane_goes_there(capsys):
    ctx = _armed()
    F = _flying(ctx)
    ctx.act("flight_set_condition", "altitude_m", 1200.0)
    assert float(F["state"].altitude_m) == pytest.approx(1200.0)
    assert ctx.act("flight_probe")["altitude"] == pytest.approx(1200.0)
    capsys.readouterr()


def test_the_two_condition_levers_OPEN_on_what_is_being_flown(capsys):
    """They are None until somebody answers them, and None means "the
    mission's own". Falling back to the bottom of the slider's band put the
    speed lever at 5.1 m/s while the aeroplane flew the mission point at
    14.6 — a panel disagreeing with the aircraft beside it."""
    ctx = _armed()
    ctx.render("controls", "derivatives")
    ctx.render("flight", "fly")
    p = ctx.act("flight_probe")
    assert p["lever_values"]["V_trim"] == pytest.approx(p["V"])
    assert p["lever_values"]["altitude_m"] == pytest.approx(p["altitude"])
    capsys.readouterr()


def test_speed_and_altitude_are_no_longer_ASKED_on_the_set_up_tab():
    """One question, one place. They moved to the picture; the set-up tab
    must not go on offering a second copy of either."""
    import inspect

    from gui.v4.stages import flight

    src = inspect.getsource(flight)
    setup = src[src.index("def _render_setup"):src.index("def _render_fly")]
    for word in ("mass [kg]", "x_cg", "thrust [N]",
                 "trim speed [m/s]", "altitude datum [m]",
                 "least thrust available", "most thrust available"):
        assert word not in setup, f"{word!r} is asked on the set-up tab"


# -------------------------------------------------- the canned excitations

@pytest.mark.parametrize("key", ["phugoid", "short_period", "dutch_roll",
                                 "roll", "spiral"])
def test_every_manoeuvre_flies(key, capsys):
    """Each one re-trims, perturbs and starts running. Driven all the way
    through its own watch time, so a manoeuvre that crashes the integrator
    or leaves the stick jammed fails here rather than on screen."""
    from gui.v4 import manoeuvre as mv

    ctx = _armed()
    _flying(ctx)
    ctx.act("flight_manoeuvre", key)
    F = ctx.S["flight"]
    assert F["manoeuvre"] == key
    assert F.get("error") is None
    assert F["running"] is True
    assert len(F["history"]) == 1, "the trace was not started clean"
    for _ in range(int(60 * min(mv.BY_KEY[key].watch_s, 12.0))):
        ctx.act("flight_advance", 1 / 60)
    assert F.get("error") is None
    assert np.isfinite(F["state"].V)
    capsys.readouterr()


def test_a_manoeuvre_starts_from_EQUILIBRIUM_and_not_from_the_pilots_throttle(
        capsys):
    """A mode is a property of the linearisation about equilibrium. Flown
    from an aeroplane that is already accelerating, the trace is the mode
    plus whatever the pilot left behind, and it cannot be compared with the
    eigenvalue that predicted it."""
    ctx = _armed()
    F = _flying(ctx)
    trim = float(F["trim"]["thrust_n"])
    ctx.act("flight_set_live", "thrust_n", trim * 1.8)
    ctx.act("flight_manoeuvre", "phugoid")
    assert float(F["live"]["thrust_n"]) == pytest.approx(
        float(F["trim"]["thrust_n"]))
    capsys.readouterr()


def test_the_scripted_input_RUNS_and_then_lets_go(capsys):
    """The doublet has to survive the spring: ``stk.step`` pulls every axis
    toward centre on the same frame, so a script applied before it would be
    swallowed. And it has to STOP — an input still held when the window
    closes is a trim change the mode is then measured about."""
    ctx = _armed()
    _flying(ctx)
    ctx.act("flight_manoeuvre", "roll")
    for _ in range(30):                       # half a second into a 2 s step
        ctx.act("flight_advance", 1 / 60)
    assert ctx.act("flight_probe")["stick"]["aileron"] == pytest.approx(10.0)
    for _ in range(180):                      # past the end, then some
        ctx.act("flight_advance", 1 / 60)
    assert abs(ctx.act("flight_probe")["stick"]["aileron"]) < 0.1, \
        "the aileron was still held after the script ran out"
    capsys.readouterr()


def test_the_roll_step_produces_a_STEADY_roll_rate(capsys):
    """Roll subsidence: the rate rises to a steady value with a time
    constant that is the mode. The steady value is the roll control power,
    so a step that produced no rate at all would mean the ailerons are not
    connected."""
    ctx = _armed()
    F = _flying(ctx)
    ctx.act("flight_manoeuvre", "roll")
    rates = []
    for i in range(120):
        ctx.act("flight_advance", 1 / 60)
        if i >= 60:                            # after one second of step
            rates.append(float(F["state"].rates[0]))
    assert abs(np.mean(rates)) > 0.1, "the aileron step rolled nothing"
    assert np.std(rates) < 0.15 * abs(np.mean(rates)), \
        "the roll rate never settled — this is not roll subsidence"
    capsys.readouterr()


def test_the_spiral_manoeuvre_touches_NOTHING(capsys):
    """The whole question is what a released stick does with ten degrees of
    bank. Any input at all answers a different one."""
    ctx = _armed()
    F = _flying(ctx)
    ctx.act("flight_manoeuvre", "spiral")
    assert np.rad2deg(F["state"].euler[0]) == pytest.approx(10.0, abs=0.01)
    for _ in range(300):
        ctx.act("flight_advance", 1 / 60)
        assert all(abs(v) < 1e-9 for k, v in F["stick"].items()
                   if k != "elevator"), "something moved the stick"
    capsys.readouterr()


def test_a_manoeuvre_is_forgotten_by_a_reset(capsys):
    """Otherwise the next reset re-fires a script whose clock has been put
    back to zero, and the aeroplane doublets on its own."""
    ctx = _armed()
    F = _flying(ctx)
    ctx.act("flight_manoeuvre", "short_period")
    ctx.act("flight_reset")
    assert F["manoeuvre"] is None
    capsys.readouterr()


# ------------------------------------------ the stability strip, and its cost

def test_a_FRAME_does_not_linearise(capsys):
    """The Jacobian is 13x13 by central differences and the eigensolve on
    top: 1.58 ms measured, a fifth of a frame at 120 Hz. Putting it in the
    frame would hand back the cost the whole render path was rebuilt to
    remove, so it is recomputed when the AEROPLANE changes and not when the
    clock does."""
    from aerobo import sixdof as sd

    ctx = _armed()
    _flying(ctx)
    calls = []
    real = sd.linearise
    sd.linearise = lambda *a, **k: (calls.append(1), real(*a, **k))[1]
    try:
        for _ in range(60):
            ctx.act("flight_advance", 1 / 60)
            ctx.act("flight_readout")
        assert calls == [], f"a frame linearised {len(calls)} times"
        ctx.act("flight_set_live", "x_cg_m",
                float(ctx.S["flight"]["live"]["x_cg_m"]) + 0.05)
        assert calls, "moving the CG did not recompute the modes"
    finally:
        sd.linearise = real
    capsys.readouterr()


def test_the_modes_move_when_the_CG_does(capsys):
    """The spiral and the short period are both functions of the CG, so a
    strip that did not follow the lever would be reporting the aeroplane the
    stage opened with."""
    ctx = _armed()
    F = _flying(ctx)
    ctx.act("flight_stability")
    before = F["modes"]["short period"].real
    ctx.act("flight_set_live", "x_cg_m", float(F["live"]["x_cg_m"]) + 0.15)
    assert F["modes"]["short period"].real != pytest.approx(before)
    capsys.readouterr()


# ============================================================================
# STAGE 5 — SIZING THE FIN AND ITS RUDDER
# ============================================================================

def test_the_fin_panel_reads_the_BUILT_surface_not_the_three_fields(capsys):
    """STATED IS NOT FLOWN, in one panel.

    Height, chord and station may all be blank, and blank means a default
    computed inside ``build_flight_model``. A panel that read the fields
    would print "None" for the fin the lattice actually flew.

    WHAT that default is has moved once already: V5 made it the DESIGN'S own
    fin (``geometry["fin"]``, sized by volume coefficient — the same surface
    the drag book charges) instead of 12 % of the span. So the expected
    numbers are read off the report rather than recomputed from a rule, which
    is the same discipline the panel itself is being tested for.
    """
    ctx = _armed()
    ctx.render("controls", "vertical")
    assert ctx.S["controls"]["vertical"]["height_m"] is None
    g = ctx.act("controls_fin")
    assert g is not None
    assert g["h"] > 0 and g["c"] > 0
    blk = (ctx.S["run"]["report"]["geometry"].get("fin") or {})
    want_h = (abs(float(blk["height_m"])) if blk else 0.12 * g["b_w"])
    want_c = (float(blk["chord_m"]) if blk else None)
    assert g["h"] == pytest.approx(want_h)
    if want_c is not None:
        assert g["c"] == pytest.approx(want_c)
    assert g["S_v"] == pytest.approx(g["h"] * g["c"])
    assert g["AR"] == pytest.approx(g["h"] / g["c"])
    assert "failed to render" not in capsys.readouterr().err


def test_the_fin_volume_coefficient_is_its_definition(capsys):
    g = ctx_fin(capsys)
    assert g["V_v"] == pytest.approx(g["S_v"] * g["arm"] / (g["S_w"]
                                                            * g["b_w"]))


def ctx_fin(capsys):
    ctx = _armed()
    ctx.render("controls", "vertical")
    capsys.readouterr()
    return ctx.act("controls_fin")


def test_the_vertical_view_BUILDS_its_own_deck(capsys):
    """The same rule Derivatives and Check are already under: a view must
    not depend on having been edited into existence first. The sizing block
    needs a deck, so this view needs one too."""
    ctx = _armed()
    assert ctx.S["controls"].get("deck") is None
    ctx.render("controls", "vertical")
    assert ctx.S["controls"].get("deck") is not None
    assert "failed to render" not in capsys.readouterr().err


def test_sizing_to_a_VOLUME_COEFFICIENT_hits_it_exactly(capsys):
    """Closed form: V_v = h.c.l_v/(S.b) with the chord and the arm fixed, so
    the height is one division and there is nothing to converge."""
    ctx = _armed()
    ctx.render("controls", "vertical")
    ctx.S["controls"]["size_Vv"] = 0.030
    ctx.act("controls_size_vv")
    assert ctx.act("controls_fin")["V_v"] == pytest.approx(0.030, rel=1e-9)
    capsys.readouterr()


def test_sizing_to_a_YAW_STIFFNESS_hits_it_by_rebuilding(capsys):
    """Not a scaling law. Cn_beta is nearly linear in fin area but Cl_r and
    Cn_r are not, and the criterion the fin is really chosen against is a
    product of all four — so the search rebuilds the lattice."""
    ctx = _armed()
    ctx.render("controls", "vertical")
    before = float(ctx.S["controls"]["deck"].Cn_beta)
    target = 0.6 * before
    ctx.S["controls"]["size_Cnb"] = target
    ctx.act("controls_size_cnb")
    assert float(ctx.S["controls"]["deck"].Cn_beta) == pytest.approx(
        target, rel=1e-4)
    assert ctx.S["controls"]["vertical"]["height_m"] is not None
    capsys.readouterr()


def test_a_target_OUT_OF_REACH_leaves_the_fin_alone(capsys):
    """A recommendation is not a limit, and a silent clamp is a
    recommendation that lies: asking for a yaw stiffness no fin between 1 %
    and 35 % of the span can reach must not quietly fit the biggest one."""
    ctx = _armed()
    ctx.render("controls", "vertical")
    h0 = ctx.act("controls_fin")["h"]
    ctx.S["controls"]["size_Cnb"] = 9.0
    ctx.act("controls_size_cnb")
    assert ctx.act("controls_fin")["h"] == pytest.approx(h0)
    assert ctx.S["controls"]["vertical"]["height_m"] is None
    capsys.readouterr()


def test_the_spiral_scan_is_a_MEASUREMENT_and_it_says_no(capsys):
    """The finding the panel exists to report.

    Shrinking the fin lowers the yaw stiffness, which looks like the fix for
    a divergent spiral. It is not, on this design: the lattice wing carries
    no dihedral, so the fin is where ALL of Cl_beta comes from and shrinking
    it takes the dihedral effect down with the yaw stiffness. Ten builds
    over 1 %-30 % of the span, measured:

        h/b    Cn_beta     margin
        0.01   +0.00107   -0.000052
        0.12   +0.13696   -0.006624
        0.30   +0.47510   -0.022977

    Every one negative, and monotone the wrong way. So the answer the panel
    gives is "no, and here is why", not a fin size.
    """
    ctx = _armed()
    ctx.render("controls", "vertical")
    ctx.act("controls_scan_spiral")
    scan = ctx.S["controls"]["spiral_scan"]
    assert scan and len(scan["rows"]) >= 8
    assert all(m < 0 for _f, _h, m, _c in scan["rows"]), \
        "a fin size now makes this spiral converge — the panel's text is " \
        "out of date, not the code"
    # ...and bigger is strictly worse, which is the shape of the finding
    margins = [m for _f, _h, m, _c in scan["rows"]]
    assert margins == sorted(margins, reverse=True)
    assert scan["best"][2] == max(margins)
    capsys.readouterr()


def test_the_scan_is_thrown_away_when_the_fin_changes(capsys):
    """A stale recommendation is worse than none — it is a verdict about a
    fin that is no longer fitted."""
    ctx = _armed()
    ctx.render("controls", "vertical")
    ctx.act("controls_scan_spiral")
    assert ctx.S["controls"]["spiral_scan"] is not None
    ctx.render("controls", "vertical")           # a render must NOT clear it
    assert ctx.S["controls"]["spiral_scan"] is not None
    ctx.S["controls"]["size_Vv"] = 0.02
    ctx.act("controls_size_vv")                  # ...an edit must
    assert ctx.S["controls"]["spiral_scan"] is None
    capsys.readouterr()


def test_the_rudder_hinge_fraction_reaches_the_DECK(capsys):
    """The rudder's area is arithmetic on the panel; its yaw power is not —
    it comes back out of the lattice, so a hinge fraction that only moved
    the caption would be caught here."""
    ctx = _armed()
    ctx.render("controls", "vertical")
    cn0 = float(ctx.S["controls"]["deck"].columns["rudder"]["Cn"])
    ctx.S["controls"]["vertical"]["rudder_chord_frac"] = 0.20
    ctx.act("controls_rebuild")
    cn1 = float(ctx.S["controls"]["deck"].columns["rudder"]["Cn"])
    assert abs(cn1) < abs(cn0), "a smaller rudder was not less powerful"
    capsys.readouterr()


# ============================================================================
# THE WINDOW IS NOT THE SESSION
#
# nicegui 3 builds the index page from a ROOT FUNCTION, once per request. A
# shell that declares its UI at module scope has no such function, so nicegui
# falls back to SCRIPT MODE and re-executes the launcher with ``runpy`` to
# get one — which called ``assemble()``, which calls ``make_session``. Every
# page load therefore handed back a brand-new session: a reload, a native
# window that lost its socket and reloaded itself, a second tab, or any URL
# that 404s (nicegui's 404 handler builds the root page too).
#
# Measured in a browser before the fix: an aileron station typed as 0.42 read
# 0.6 again after one reload, and the shell was back on stage 1 with the run
# record, the section and the controls gone.
# ============================================================================

@pytest.fixture()
def blank_page():
    """No page has been built in this process yet."""
    was = v4app.PAGE_STATE["S"]
    v4app.PAGE_STATE["S"] = None
    yield
    v4app.PAGE_STATE["S"] = was


def test_the_shell_gives_nicegui_a_ROOT_FUNCTION(monkeypatch):
    """The whole defect in one assertion: with no root function nicegui
    re-runs the launcher script per request, and the script builds a
    session."""
    from nicegui import ui

    seen = {}
    monkeypatch.setattr(ui, "run", lambda *a, **kw: seen.update(
        {"root": a[0] if a else kw.get("root"), "kw": kw}))
    v4app.run(native=False, port=0)
    assert seen["root"] is v4app.page, \
        "ui.run got no root page function — nicegui will re-execute the " \
        "launcher for every page load"
    assert seen["kw"]["reload"] is False


def test_a_reload_does_not_hand_the_user_a_NEW_DESIGN(blank_page):
    """Build a page, answer something, build the page again."""
    first = v4app.page()
    first.S["controls"]["aileron"]["span_from"] = 0.42
    first.S["run"]["record"] = {"pretend": "a completed run"}

    second = v4app.page()
    assert second is not first                     # a new window, genuinely
    assert second.S is first.S                     # the same design
    assert second.S["controls"]["aileron"]["span_from"] == 0.42
    assert second.S["run"]["record"] is not None


def test_the_REBUILT_stages_write_into_the_design_that_was_carried(blank_page):
    """The trap this could have fallen into: every stage closes over its own
    sub-dict at BUILD time, so handing the session over after the shell is
    assembled leaves seven stages writing into a dict nothing reads. Drive a
    stage action on the SECOND page and read the answer off the FIRST."""
    from aerobo import api

    first = v4app.page()
    cfg = api.RunConfig(problem_name="tail", budget=4, seed=0)
    built = api.PROBLEM_SPECS["tail"].build({}, {}, None)
    first.S["run"]["record"] = {"pretend": "a completed run"}
    first.S["run"]["report"] = api.design_report(cfg,
                                                 built.bounds.mean(axis=1))
    second = v4app.page()
    second.act("controls_rebuild")
    assert first.S["controls"]["deck"] is not None, \
        "the rebuilt page's stage 5 wrote into a dict the session dropped"


def test_a_rebuilt_window_is_not_still_FLYING(blank_page):
    """The frame loop is a ui.timer and it died with the page. A session
    that still said it was running would paint a stage the tree calls
    'running' with nothing moving in it.

    Two things enforce this — :func:`gui.v4.app.page` clears the flag, and
    the Fly view tears the loop down as the first act of every render — so
    the mutation gate has to break BOTH to see this test fail. It is pinned
    as an outcome and not as either mechanism on purpose.
    """
    first = v4app.page()
    first.S["flight"]["running"] = True
    first.S["flight"]["t"] = 12.5
    second = v4app.page()
    assert second.S["flight"]["running"] is False
    assert second.S["flight"]["t"] == 12.5     # ...but not put back to zero


def test_assemble_ON_ITS_OWN_is_still_a_fresh_session(blank_page):
    """The persistence belongs to the PAGE, not to assembly: every other
    caller in this suite asks for a shell and must get a clean one."""
    a = v4app.assemble()
    b = v4app.assemble()
    assert a.S is not b.S
    assert v4app.PAGE_STATE["S"] is None       # and nothing was stashed


def test_FILE_NEW_SESSION_still_resets_everything(blank_page):
    """The pin must last exactly one assembly. Left installed, the shell's
    own reset would be handed the session it is resetting — and its in-place
    refill (``ctx.S.clear()`` then ``update(fresh)``) would empty it."""
    v4app.page()
    ctx = v4app.page()               # ...a RELOAD, which is what installs it
    ctx.S["controls"]["aileron"]["span_from"] = 0.42
    ctx.S["run"]["record"] = {"pretend": "a completed run"}
    v4app._load_shell()._new_session(ctx)
    assert ctx.S["controls"]["aileron"]["span_from"] == \
        session.CONTROLS_DEFAULTS["aileron"]["span_from"]
    assert ctx.S["run"]["record"] is None
    assert ctx.S["mission"], "the reset emptied the session instead"


# ============================================================================
# LEAVING STAGE 6 PUTS THE AEROPLANE DOWN
#
# The frame loop is a ui.timer and a timer does not care which stage is on
# screen. Selecting stage 5 left the simulation integrating at 30 Hz behind a
# page nobody was watching: measured on the tail design, three seconds spent
# on the Controls stage took it from 300 m to 62 m and from level to 36
# degrees nose down. Coming back then stopped the loop — because the Fly view
# is rebuilt and a render tears the timer down — so what the pilot saw was an
# aeroplane somewhere else, with nothing moving.
# ============================================================================

def _in_the_air(capsys=None):
    """A shell with a deck, armed, flying."""
    ctx = _armed()
    ctx.render("controls", "derivatives")
    ctx.select("flight", "fly")          # the shell's own path, not a render
    assert ctx.S["ui"]["selected"] == "flight"
    ctx.act("flight_run", True)
    for _ in range(10):
        ctx.act("flight_advance", 1 / 30)
    assert ctx.S["flight"]["running"] is True
    assert ctx.S["flight"]["t"] > 0.0
    return ctx


def test_LEAVING_the_flight_stage_stops_the_loop(capsys):
    ctx = _in_the_air()
    ctx.select("controls")
    assert ctx.S["flight"]["running"] is False, \
        "the simulation went on flying behind the Controls stage"
    assert ctx.act("flight_probe")["has_timer"] is False, \
        "the 30 Hz timer outlived the stage that owns it"
    capsys.readouterr()


def test_leaving_does_not_MOVE_the_aeroplane(capsys):
    """Paused, not reset: the design review carries on from where it was."""
    ctx = _in_the_air()
    t, st = float(ctx.S["flight"]["t"]), ctx.S["flight"]["state"]
    n = len(ctx.S["flight"]["history"])
    ctx.select("controls")
    ctx.select("flight", "fly")
    assert ctx.S["flight"]["t"] == pytest.approx(t, abs=1e-12)
    assert ctx.S["flight"]["state"] is st
    assert len(ctx.S["flight"]["history"]) == n
    assert ctx.S["flight"]["running"] is False   # ...and it waits for Fly
    capsys.readouterr()


def test_a_stage_that_is_NOT_flight_can_still_be_left(capsys):
    """The hook wraps the shell's own select and must not eat anything
    else: every other move still lands, and a stopped flight is not
    re-paused."""
    ctx = _armed()
    ctx.render("controls", "derivatives")
    ctx.select("controls", "vertical")
    assert ctx.S["ui"]["selected"] == "controls"
    assert ctx.S["ui"]["tab"]["controls"] == "vertical"
    ctx.select("flight", "modes")
    assert ctx.S["ui"]["selected"] == "flight"
    assert ctx.S["ui"]["tab"]["flight"] == "modes"
    ctx.select("results")
    assert ctx.S["ui"]["selected"] == "results"
    capsys.readouterr()


def test_moving_BETWEEN_the_flight_tabs_does_not_pause_it(capsys):
    """Traces and Modes are the same stage. A pilot looking at the trace of
    the manoeuvre still running must not have it stopped underneath them."""
    ctx = _in_the_air()
    ctx.select("flight", "traces")
    assert ctx.S["flight"]["running"] is True, \
        "changing tab inside stage 6 stopped the simulation"
    capsys.readouterr()


# ============================================================================
# A FIELD IS HANDED AN EVENT, NOT A VALUE
#
# ``ui.number(on_change=...)`` and ``ui.select(on_change=...)`` call their
# handler with a ``ValueChangeEventArguments``. Every numeric field in stage 5
# and stage 6 treated it as the value, which had a quiet half and a loud half:
#
#   quiet — ``float(event)`` raises inside the handler, where nicegui logs it
#           and the user sees the field simply not take. No number typed into
#           either stage had ever committed.
#   loud  — the three fields that stored the argument straight put the event
#           object into the session, so the NEXT render handed it to
#           ``ui.number`` as a value and the view died inside nicegui's own
#           formatter: "int() argument must be ... not
#           'ValueChangeEventArguments'".
#
# These drive the REAL widgets through ``set_value``, which is the same path a
# keystroke takes, and check the number comes back.
# ============================================================================

def _numbers(ctx, stage, view):
    """Every ui.number in a built view, in page order."""
    from nicegui import ui

    out = []

    def walk(el):
        for ch in el.default_slot.children:
            if isinstance(ch, ui.number):
                out.append(ch)
            walk(ch)

    walk(ctx.views[(stage, view)])
    return out


def _round_trips(ctx, stage, view, capsys=None):
    """Type a distinct number into every field of a view and read it back."""
    n = len(_numbers(ctx, stage, view))
    assert n > 0, f"{stage}/{view} has no numeric field to test"
    for i in range(n):
        target = 0.30 + 0.005 * i          # valid as a fraction AND as metres
        _numbers(ctx, stage, view)[i].set_value(target)
        ctx.render(stage, view)            # the render that used to explode
        back = _numbers(ctx, stage, view)
        assert len(back) == n, "the view lost a field on the way back"
        assert back[i].value == pytest.approx(target), \
            f"{stage}/{view} field {i} did not take the number typed into it"
    return n


def test_every_number_typed_into_stage_5_ACTUALLY_LANDS(capsys):
    ctx = _armed()
    ctx.render("controls", "surfaces")
    ctx.render("controls", "vertical")
    # 4 = the aileron band's three numbers plus the elevator's hinge
    # chord. It was 7 while a flap band sat between them; the floor
    # moves with the census so a LOST field still fails here.
    assert _round_trips(ctx, "controls", "surfaces") >= 4
    assert _round_trips(ctx, "controls", "vertical") >= 6
    assert "failed to render" not in capsys.readouterr().err


def test_a_typed_fin_height_does_not_POISON_the_next_render(capsys):
    """The reported crash, exactly: type a height, render again."""
    ctx = _armed()
    ctx.render("controls", "vertical")
    # BY POSITION, not by "the first empty box". It used to be located that
    # way, which stopped meaning "the fin height" the moment a defaulted
    # field learned to open on the value it is flying — the first empty box
    # became a sizing TARGET further down, and this test silently moved to
    # a field it was not about.
    height = _numbers_of(ctx, "controls", "vertical")[0]
    height.set_value(0.9)
    stored = ctx.S["controls"]["vertical"]["height_m"]
    assert isinstance(stored, float), \
        f"the session stored a {type(stored).__name__}, not a number"
    ctx.render("controls", "vertical")
    assert "failed to render" not in capsys.readouterr().err


def test_a_CLEARED_field_means_what_that_field_says_blank_means(capsys):
    """Blank is an answer on the fin's three fields ('let the model choose')
    and is NOT one on a hinge fraction, where it would build a wing with an
    aileron of no chord."""
    ctx = _armed()
    ctx.render("controls", "vertical")
    C = ctx.S["controls"]
    C["vertical"]["height_m"] = 1.4
    ctx.render("controls", "vertical")
    height = next(n for n in _numbers(ctx, "controls", "vertical")
                  if n.value == 1.4)
    height.set_value(None)
    assert C["vertical"]["height_m"] is None, "blank did not clear the fin"

    ctx.render("controls", "surfaces")
    before = C["aileron"]["chord_frac"]
    frac = next(n for n in _numbers(ctx, "controls", "surfaces")
                if n.value == before)
    frac.set_value(None)
    assert C["aileron"]["chord_frac"] == before, \
        "a cleared hinge fraction was stored as a blank"
    capsys.readouterr()


def test_every_number_typed_into_stage_6_actually_lands(capsys):
    ctx = _armed()
    ctx.render("controls", "derivatives")
    ctx.render("flight", "setup")
    assert _round_trips(ctx, "flight", "setup") >= 2
    assert "failed to render" not in capsys.readouterr().err


def test_two_keys_on_one_axis_do_not_leave_the_stick_stuck(capsys):
    """A and D are opposite demands on ONE control, not two switches.

    ``hold[axis] = 0`` on any keyup meant that pressing A, then D without
    letting go of A, then releasing A, centred an axis whose other key was
    still physically down — the aeroplane stopped rolling with the pilot's
    finger still on the stick.
    """
    from types import SimpleNamespace as NS

    ctx = _armed()
    ctx.render("controls", "derivatives")
    assert ctx.act("flight_arm") is True
    F = ctx.S["flight"]
    F["running"] = True

    def press(name, down):
        ctx.act("flight_key", NS(key=NS(name=name),
                                 action=NS(keydown=down, keyup=not down)))

    press("a", True)
    assert F["hold"]["aileron"] == +1
    press("d", True)                     # both down: they cancel, as a
    assert F["hold"]["aileron"] == 0     # centred stick does
    press("a", False)                    # ...and D is STILL down
    assert F["hold"]["aileron"] == -1, \
        "releasing the key that was let go of centred the stick"
    press("d", False)
    assert F["hold"]["aileron"] == 0

    # ...and a keyUP is honoured even when the simulation has stopped, or a
    # key held through a pause is held for ever
    press("a", True)
    F["running"] = False
    press("a", False)
    assert F["hold"]["aileron"] == 0
    assert not F["keys_down"]
    capsys.readouterr()


def test_the_stick_moves_in_WALL_time_not_in_simulated_time(capsys):
    """A pilot's hand does not speed up because the clock did.

    ``dt`` is ``wall * speed``; driving the spring with it made the stick
    four times quicker at the 4x multiplier — full deflection in 0.09 s of
    wall clock instead of the 0.35 s ``stick.TRAVEL_S`` names — so the faster
    the simulation ran the less flyable the aeroplane was.
    """
    from types import SimpleNamespace as NS

    from gui.v4 import stick as stk

    got = {}
    for speed in (1.0, 4.0):
        ctx = _armed()
        ctx.render("controls", "derivatives")
        assert ctx.act("flight_arm") is True
        F = ctx.S["flight"]
        F["speed"] = speed
        F["running"] = True
        ctx.act("flight_key", NS(key=NS(name="a"),
                                 action=NS(keydown=True, keyup=False)))
        for _ in range(6):                       # 0.1 s of WALL clock
            ctx.act("flight_advance", 1 / 60)
        got[speed] = F["stick"]["aileron"]
        F["running"] = False

    assert got[1.0] == pytest.approx(got[4.0], rel=1e-9), (
        f"the stick moved {got[4.0] / max(got[1.0], 1e-9):.2f}x as far at 4x "
        f"speed: {got}")
    # ...and it is the rate stick.py names: 0.1 s of a TRAVEL_S sweep
    assert got[1.0] == pytest.approx(
        stk.LIMITS["aileron"] * 0.1 / stk.TRAVEL_S, rel=1e-6)
    capsys.readouterr()


def test_the_thrust_line_is_asked_in_stage_5_and_only_reported_in_stage_6(
        capsys):
    """ONE QUESTION, ONE PLACE. The thrust line is a property of the
    AIRFRAME, so it belongs beside the fin and the ailerons — not among the
    levers of one particular flight, where stage 5's deck could not see it
    and the same design had two answers."""
    ctx = _armed()
    C = ctx.S["controls"]

    # ...it is a switch, and turning it off exposes exactly one number
    ctx.render("controls", "propulsion")
    assert C["thrust"]["through_cg"] is True
    assert not _numbers(ctx, "controls", "propulsion"),         "'through the CG' must not show an offset field to answer"
    C["thrust"]["through_cg"] = False
    ctx.render("controls", "propulsion")
    nums = _numbers(ctx, "controls", "propulsion")
    assert len(nums) == 1
    nums[0].set_value(0.4)
    assert C["thrust"]["z_below_cg_m"] == pytest.approx(0.4)

    # ...and it reaches the AEROPLANE, not just the session
    ctx.act("controls_rebuild")
    fm = ctx.S["controls"]["fm"]
    assert fm.aircraft.prop.z_offset_m == pytest.approx(0.4)

    # ...while stage 6 has no field for it at all
    ctx.render("flight", "setup")
    assert "thrust_z_m" not in ctx.S["flight"]
    capsys.readouterr()


def test_the_VIEW_selector_reads_its_own_value(capsys):
    """``_set_mode`` was wired as the select's handler directly, so it
    compared a ValueChangeEventArguments to the string "game" — which is
    never equal, so every choice landed on 'engineering' and the game view
    could not be got back."""
    from nicegui import ui

    ctx = _armed()
    ctx.render("controls", "derivatives")
    ctx.render("flight", "fly")

    def selects():
        out = []

        def walk(el):
            for ch in el.default_slot.children:
                if isinstance(ch, ui.select):
                    out.append(ch)
                walk(ch)

        walk(ctx.views[("flight", "fly")])
        return out

    rate, view = None, None
    for sel in selects():
        opts = sel.options if isinstance(sel.options, dict) else {}
        if "game" in opts:
            view = sel
        elif 1.0 in opts:
            rate = sel
    assert view is not None and rate is not None
    view.set_value("engineering")
    assert ctx.S["flight"]["mode"] == "engineering"
    view.set_value("game")
    assert ctx.S["flight"]["mode"] == "game", \
        "the view selector could not get back to the game view"
    rate.set_value(2.0)
    assert ctx.S["flight"]["speed"] == pytest.approx(2.0)
    capsys.readouterr()


def test_the_fin_has_NO_where_MENU_anywhere(capsys):
    """A control that cannot move the model is worse than no control.

    "where" stored ``"tail"`` or ``"between"`` and NOTHING read it: the fin
    went where the station said, or to the tail station when the station was
    blank, whichever the menu was set to. So the menu is gone — from the
    view, from the session, and from the spec the lattice is built with —
    and the station it never affected is now the leading edge, which is a
    number the picture above it is drawn in.
    """
    from nicegui import ui

    ctx = _armed()
    ctx.render("controls", "vertical")
    assert "where" not in ctx.S["controls"]["vertical"]
    assert "where" not in session.CONTROLS_DEFAULTS["vertical"]

    out = []

    def walk(el):
        for ch in el.default_slot.children:
            if isinstance(ch, ui.select):
                out.append(ch)
            walk(ch)

    walk(ctx.views[("controls", "vertical")])
    assert not [s for s in out if "tail" in (s.options or {})], \
        "the dead 'where' menu is still on the Fin & rudder view"
    capsys.readouterr()


# ============================================================================
# THE FIN HAS A REYNOLDS NUMBER AND THE LATTICE DOES NOT KNOW IT
#
# The deck gives the fin a lift-curve slope with no Reynolds number anywhere
# in it. The chord it runs at is small — 0.65 of the wing MAC by default, and
# the RUDDER is a fraction of that again — so stage 5 now reports both, beside
# the Reynolds number this design's own section data is at, and recommends the
# chord that would match them.
#
# The recommendation prices itself with a real build, because fin VOLUME is
# not a sufficient statistic for yaw stiffness: holding V_v while the chord
# grows lowers the aspect ratio, and measured on the tail design that is
# Cn_beta +0.13696 -> +0.08016 (-41.5 %) for the same V_v of 0.0434.
# ============================================================================

def _fin_ctx():
    ctx = _armed()
    ctx.render("controls", "vertical")
    return ctx


def test_the_fin_reynolds_is_the_MISSION_fluid_and_the_FIN_chord(capsys):
    from gui.v3 import session as v3s

    ctx = _fin_ctx()
    dp = v3s.design_point(ctx.S)
    r = ctx.act("controls_reynolds")
    g = ctx.act("controls_fin")
    assert r["re_now"] == pytest.approx(
        dp["rho"] * dp["v_ms"] * g["c"] / dp["mu"], rel=1e-9)
    # ...and it is the CHORD's, so doubling the chord doubles it
    ctx.S["controls"]["vertical"]["chord_m"] = 2.0 * g["c"]
    ctx.act("controls_rebuild")
    assert ctx.act("controls_reynolds")["re_now"] == \
        pytest.approx(2.0 * r["re_now"], rel=1e-6)
    capsys.readouterr()


def test_the_rudder_reynolds_is_a_FRACTION_of_the_fins(capsys):
    ctx = _fin_ctx()
    r = ctx.act("controls_reynolds")
    frac = ctx.S["controls"]["vertical"]["rudder_chord_frac"]
    assert r["re_rudder"] == pytest.approx(r["re_now"] * frac, rel=1e-9)
    ctx.S["controls"]["vertical"]["rudder_chord_frac"] = 0.5 * frac
    ctx.act("controls_rebuild")
    assert ctx.act("controls_reynolds")["re_rudder"] == \
        pytest.approx(0.5 * r["re_rudder"], rel=1e-6)
    capsys.readouterr()


def test_the_recommendation_REACHES_the_reference_and_holds_V_v(capsys):
    ctx = _fin_ctx()
    before = ctx.act("controls_reynolds")
    assert before["needed"] is True          # the default fin is below it
    v_v = ctx.act("controls_fin")["V_v"]
    ctx.act("controls_apply_re")
    after = ctx.act("controls_reynolds")
    assert after["re_now"] == pytest.approx(before["re_ref"], rel=1e-6)
    assert ctx.act("controls_fin")["V_v"] == pytest.approx(v_v, rel=1e-9), \
        "the Reynolds fix silently moved the fin volume the user sized for"
    assert after["needed"] is False
    capsys.readouterr()


def test_the_quoted_COST_is_the_one_actually_paid(capsys):
    """The cost line is a real rebuild, not a scaling law. Take the
    recommendation and check the deck agrees with what was promised."""
    ctx = _fin_ctx()
    rec = ctx.act("controls_reynolds")
    promised = rec["cnb_rec"]
    assert promised is not None
    ctx.act("controls_apply_re")
    assert float(ctx.S["controls"]["deck"].Cn_beta) == \
        pytest.approx(promised, rel=1e-9)


def test_fin_VOLUME_is_not_a_sufficient_statistic_for_yaw_stiffness(capsys):
    """The finding the cost line exists for. Same V_v, different aspect
    ratio, materially different Cn_beta — so 'size it by V_v' cannot be the
    whole story and the panel must not pretend otherwise."""
    ctx = _fin_ctx()
    rec = ctx.act("controls_reynolds")
    assert rec["AR_rec"] < rec["AR_now"]
    lost = 1.0 - rec["cnb_rec"] / rec["cnb_now"]
    assert lost > 0.2, \
        f"only {lost:.1%} lost — if a fatter fin were nearly free at equal " \
        f"V_v there would be nothing to warn about"
    capsys.readouterr()


def test_nothing_ever_recommends_making_the_fin_SMALLER(capsys):
    ctx = _fin_ctx()
    ref = ctx.act("controls_reynolds")["re_ref"]
    g = ctx.act("controls_fin")
    big = g["c"] * 2.0 * ref / ctx.act("controls_reynolds")["re_now"]
    ctx.S["controls"]["vertical"]["chord_m"] = big
    ctx.act("controls_rebuild")
    r = ctx.act("controls_reynolds")
    assert r["re_now"] > ref
    assert r["needed"] is False
    assert "c_rec" not in r, "it offered to shrink a fin that is already big"
    ctx.act("controls_apply_re")
    assert ctx.S["controls"]["vertical"]["chord_m"] == pytest.approx(big), \
        "applying a recommendation there was none of moved the fin"
    capsys.readouterr()


def test_the_reynolds_recommendation_is_not_a_LIMIT(capsys):
    """A calibration is a default, not a ban: a chord far below the
    reference is accepted, flown, and reported — not clamped."""
    ctx = _fin_ctx()
    ctx.S["controls"]["vertical"]["chord_m"] = 0.05
    ctx.act("controls_rebuild")
    assert ctx.S["controls"]["error"] is None
    assert ctx.act("controls_fin")["c"] == pytest.approx(0.05)
    r = ctx.act("controls_reynolds")
    assert r["needed"] is True and r["re_now"] < r["re_ref"]
    capsys.readouterr()


def test_every_numeric_field_in_V4_waits_for_you_to_STOP_typing():
    """``ui.number`` fires its handler on every keystroke and both V4 stages
    rebuild the panel the field lives in — the panels are derived from the
    fin and from the trim, so there is nothing to show if they do not.

    Measured in a browser with no debounce: typing "1.85" into the fin height
    committed "1", rebuilt the view under the cursor and threw ".85" away.
    With it, the same keystrokes leave 1.85 in the field and 1.85 in the
    session. This checks the shipped widgets carry it; the browser is what
    checked that it works.
    """
    ctx = _armed()
    ctx.render("controls", "derivatives")
    seen, levers = 0, 0
    for stage, view in (("controls", "surfaces"), ("controls", "vertical"),
                        ("flight", "setup"), ("flight", "fly")):
        ctx.render(stage, view)
        for el in _numbers(ctx, stage, view):
            # the Fly tab's LEVERS are a slider and a number that are one
            # control: whichever moves, the other is set WITHOUT a rebuild,
            # so the number keeps its focus and needs no debounce. They are
            # the raw ui.number; every field row is widgets.number_field.
            if "hide-bottom-space" not in el._props:
                levers += 1
                continue
            debounce = el._props.get("debounce")
            assert debounce, \
                f"{stage}/{view}: a numeric field commits on every keystroke"
            assert float(debounce) >= 200
            seen += 1
    # 14 is the census with the flap band gone: 4 on surfaces, 7 on
    # vertical, 2 on the flight setup, 1 on Fly. It was 17. The floor
    # moves with the census so a field that LOSES its debounce, or
    # goes missing, still fails here.
    assert seen >= 14, f"only {seen} fields checked"
    assert levers >= 3, "the levers moved somewhere this test cannot see"


# ============================================================================
# SESSION 70 — THE FIN IS PLACED WHERE YOU CAN SEE IT, THE RUDDER IS
# RECOMMENDED, AND STAGE 5 STOPS REPAINTING THE FOUR STAGES BEHIND IT
#
# Four asks, in the user's words:
#
#   "rudder should be given a recomended value"
#   "'where' should be eliminated, instead its leading edge should be placed
#    from x station [m]"
#   "the x position of the wing and satbiliser should be presented ... along
#    side a 2D lateral view"
#   "when accessing control for first time in a run it goes extremely laggy"
# ============================================================================

def _view_text(ctx, stage, view) -> str:
    """Every label, hint, caption and field label in a built view, joined.

    ``descendants()`` walks the real element tree, so a control that was
    never created cannot be found here — which is the point.
    """
    parts = []
    for e in ctx.views[(stage, view)].descendants():
        t = getattr(e, "text", None)
        if isinstance(t, str):
            parts.append(t)
        for k in ("label", "hint"):
            v = (getattr(e, "props", {}) or {}).get(k)
            if isinstance(v, str):
                parts.append(v)
    return "\n".join(parts)


def _view_html(ctx, stage, view) -> str:
    """Every ui.html body in a built view — the pictures."""
    return "\n".join(
        (getattr(e, "content", "") or "")
        for e in ctx.views[(stage, view)].descendants())


def _fin_of(ctx):
    """The VerticalSurface the LATTICE built, not the fields that asked."""
    return ctx.S["controls"]["fm"].model.vertical


def test_the_fin_is_placed_by_its_LEADING_EDGE(capsys):
    """The number typed is the leading edge; the lattice is placed by the
    quarter chord. One conversion, and it is the RIGHT way round.

    Asserted against the built surface rather than the field, because the
    whole point of the change is that the two are no longer the same number.
    """
    ctx = _armed()
    ctx.act("controls_rebuild")
    chord = float(_fin_of(ctx).chord)
    ctx.S["controls"]["vertical"]["x_le_m"] = 3.0
    ctx.act("controls_rebuild")
    v = _fin_of(ctx)
    assert float(v.chord) == pytest.approx(chord)      # nothing else moved
    assert float(v.x) == pytest.approx(3.0 + 0.25 * chord), \
        "the station was used as the quarter chord, not the leading edge"
    assert float(v.x) != pytest.approx(3.0)
    # ...and the stage reports the pair, so the arm is still readable
    g = ctx.act("controls_fin")
    assert g["x_le"] == pytest.approx(3.0)
    assert g["x"] == pytest.approx(3.0 + 0.25 * chord)
    capsys.readouterr()


def test_a_BLANK_station_still_lands_the_quarter_chord_at_the_tail(capsys):
    """The default is unchanged by the change of units. Blank meant "quarter
    chord at the tail station" before and it means the same now — otherwise
    every design silently grew or lost a quarter of a fin chord of arm."""
    ctx = _armed()
    ctx.S["controls"]["vertical"]["x_le_m"] = None
    ctx.act("controls_rebuild")
    v = _fin_of(ctx)
    x_t = float(ctx.S["run"]["report"]["geometry"]["tail"]["dist_m"])
    assert float(v.x) == pytest.approx(x_t), \
        "a blank station moved the fin's quarter chord off the tail station"
    assert ctx.act("controls_fin")["x_le"] == pytest.approx(
        x_t - 0.25 * float(v.chord))
    capsys.readouterr()


def test_the_side_elevation_is_measured_off_the_LATTICE(capsys):
    """STATED IS NOT FLOWN. Every station drawn comes off the panels, so a
    design whose fin chord is BLANK — a default that lives inside
    build_flight_model and is not in the session at all — still draws a fin
    with a real chord in a real place.
    """
    ctx = _armed()
    ctx.S["controls"]["vertical"]["chord_m"] = None
    ctx.S["controls"]["vertical"]["x_le_m"] = None
    ctx.act("controls_rebuild")
    st = ctx.act("controls_layout")
    v = _fin_of(ctx)
    b = st["boxes"]
    assert {"wing", "tail", "fin"} <= set(b)
    assert b["fin"]["x_le"] == pytest.approx(float(v.x) - 0.25 * v.chord)
    assert b["fin"]["x_te"] == pytest.approx(float(v.x) + 0.75 * v.chord)
    # the wing's quarter-chord line IS the origin every station is quoted from
    assert b["wing"]["x_qc"] == pytest.approx(0.0, abs=1e-9)
    assert b["wing"]["x_le"] < 0.0 < b["wing"]["x_te"]
    # the stabiliser is aft of the wing and the fin is with it
    assert b["tail"]["x_qc"] > b["wing"]["x_qc"]
    assert st["x_np"] is not None and st["x_cg"] is not None
    capsys.readouterr()


def test_every_coordinate_drawn_is_INSIDE_the_side_elevations_viewBox():
    """The gauge panel's own lesson, applied before it can happen twice: a
    row can sit outside the viewBox with the markup and the state both
    perfectly correct, and no assertion about either can see it.

    Read the shipped SVG back and check every coordinate against the box the
    same string declares.
    """
    import re

    from gui.v4 import sideview as sv

    st = {"boxes": {
        "wing": {"x_le": -0.3, "x_te": 0.9, "x_qc": 0.0, "chord": 1.2,
                 "z_lo": 0.0, "z_hi": 0.0},
        "tail": {"x_le": 5.3, "x_te": 6.0, "x_qc": 5.5, "chord": 0.66,
                 "z_lo": 0.5, "z_hi": 0.5},
        "fin": {"x_le": 5.3, "x_te": 6.0, "x_qc": 5.5, "chord": 0.66,
                "z_lo": 0.5, "z_hi": 1.7}},
        "x_cg": 0.05, "x_np": 0.45}
    s = sv.svg(st)
    vb = [float(v) for v in
          re.search(r'viewBox="([^"]+)"', s).group(1).split()]
    assert vb[0] == 0.0 and vb[1] == 0.0 and vb[2] > 0 and vb[3] > 0
    W, H = vb[2], vb[3]
    seen = 0
    for m in re.finditer(r'<rect ([^/>]+)/>', s):
        a = dict(re.findall(r'(\w+)="([^"]*)"', m.group(1)))
        x, y = float(a["x"]), float(a["y"])
        w, h = float(a["width"]), float(a["height"])
        assert -1e-9 <= x and x + w <= W + 1e-9, f"rect off the box in x: {a}"
        assert -1e-9 <= y and y + h <= H + 1e-9, f"rect off the box in y: {a}"
        seen += 1
    for m in re.finditer(r'<circle ([^/>]+)/>', s):
        a = dict(re.findall(r'(\w+)="([^"]*)"', m.group(1)))
        cx, cy, r = float(a["cx"]), float(a["cy"]), float(a["r"])
        assert -r <= cx <= W + r and -r <= cy <= H + r, f"marker off: {a}"
        seen += 1
    assert seen >= 5, "the picture drew almost nothing"
    # the font stack is single-quoted, or the attribute closes on "SF Mono"
    # and the rest of the stack is parsed as bare attributes on the <svg>
    assert '"SF Mono"' not in s


def test_the_side_elevation_keeps_ONE_SCALE_on_both_axes():
    """A side elevation that stretched an axis would answer the question it
    exists for — "is the fin near the tail?" — with a lie. Two designs whose
    x extents differ by a factor of ten must differ by the same factor in
    the viewBox, with the drawn fin the same height in metres."""
    from gui.v4 import sideview as sv

    def box(x_tail):
        st = {"boxes": {
            "wing": {"x_le": -0.5, "x_te": 0.5, "x_qc": 0.0, "chord": 1.0,
                     "z_lo": 0.0, "z_hi": 0.0},
            "fin": {"x_le": x_tail, "x_te": x_tail + 1.0, "x_qc": x_tail,
                    "chord": 1.0, "z_lo": 0.0, "z_hi": 1.0}},
            "x_cg": 0.0, "x_np": 0.1}
        s = sv.svg(st)
        import re
        vb = [float(v) for v in
              re.search(r'viewBox="([^"]+)"', s).group(1).split()]
        rect = re.findall(r'<rect [^/>]*height="([\d.]+)"', s)
        return vb[2], vb[3], [float(h) for h in rect]

    w1, h1, r1 = box(4.0)
    w2, h2, r2 = box(40.0)
    # the drawn x extents are 5.5 m and 41.5 m, each padded by the same
    # FRACTION, so the ratio of the viewBox widths is the ratio of the
    # extents exactly
    assert w2 / w1 == pytest.approx(41.5 / 5.5, rel=1e-9)
    # the FIN is one metre tall in both, so at one unit per metre its
    # rectangle is the same height in both pictures
    assert max(r1) == pytest.approx(1.0)
    assert max(r2) == pytest.approx(1.0)
    assert h2 > h1                       # only the empty air grew
    # ...and the picture cannot go arbitrarily thin as the design gets
    # longer: the z pad is a FRACTION of the x extent, so the box keeps a
    # shape. Without that floor a 40 m layout draws 46 units wide and 6
    # high — a letterbox nothing is legible in — while the numbers stay
    # perfectly correct.
    for w, h in ((w1, h1), (w2, h2)):
        assert h / w >= sv._MIN_Z_FRAC, \
            f"the picture went to {w:.1f} x {h:.1f}, aspect {h / w:.3f}"


# ---------------------------------------------------------------- the rudder

def _beta_held(deck, d_max=25.0):
    col = deck.columns["rudder"]
    return abs(col["Cn"]) * d_max / abs(deck.Cn_beta)


def test_the_recommended_rudder_HOLDS_the_sideslip_it_promises(capsys):
    """The outcome, not the formula. Take the recommendation, rebuild the
    deck from it, and measure the sideslip full rudder holds — it has to be
    the target that was asked for.

    Cn_dr is only NEARLY proportional to the thin-aerofoil effectiveness
    (0.7 % out at a hinge fraction of 0.3, 2.8 % at 0.9), so a closed-form
    inverse alone would miss; the recommendation re-solves from a measured
    beta and this is what pins that it converged.
    """
    ctx = _armed()
    ctx.act("controls_rebuild")
    ctx.S["controls"]["vertical"]["rudder_beta_deg"] = 14.0
    rec = ctx.act("controls_rudder_rec")
    assert rec["reachable"] and rec["f_rec"] is not None
    ctx.act("controls_apply_rudder")
    assert ctx.S["controls"]["vertical"]["rudder_chord_frac"] == \
        pytest.approx(rec["f_rec"])
    got = _beta_held(ctx.S["controls"]["deck"], rec["d_max"])
    assert got == pytest.approx(14.0, rel=2e-3), \
        f"the rudder it recommended holds {got:.3f} deg, not 14"
    capsys.readouterr()


def test_the_recommendation_moves_with_the_TARGET(capsys):
    """It is a solve against a requirement, not a constant dressed up as
    one. A bigger sideslip demand needs more rudder, monotonically."""
    ctx = _armed()
    ctx.act("controls_rebuild")
    out = []
    for target in (6.0, 12.0, 18.0):
        ctx.S["controls"]["vertical"]["rudder_beta_deg"] = target
        out.append(ctx.act("controls_rudder_rec")["f_rec"])
    assert out[0] < out[1] < out[2], out
    capsys.readouterr()


def test_a_sideslip_NO_rudder_can_hold_is_reported_not_clamped(capsys):
    """A recommendation is not a limit, and a silent clamp is a
    recommendation that lies. At a hinge fraction of 1.0 the rudder IS the
    fin — all-moving — and nothing holds more than that, so a target past it
    comes back as out of reach with the ceiling measured, NOT as 1.0.
    """
    ctx = _armed()
    ctx.act("controls_rebuild")
    ceiling = ctx.act("controls_rudder_rec")["beta_max"]
    ctx.S["controls"]["vertical"]["rudder_beta_deg"] = 5.0 * ceiling
    rec = ctx.act("controls_rudder_rec")
    assert rec["reachable"] is False
    assert rec.get("f_rec") is None, "it clamped to the edge of the search"
    assert rec["beta_max"] == pytest.approx(ceiling)
    # ...and nothing was applied
    before = ctx.S["controls"]["vertical"]["rudder_chord_frac"]
    ctx.act("controls_apply_rudder")
    assert ctx.S["controls"]["vertical"]["rudder_chord_frac"] == before
    capsys.readouterr()


def test_the_rudder_target_is_a_DEFAULT_and_never_a_ban(capsys):
    """A calibration is a default, not a ban. Blank falls back to the stated
    10 deg; any number is accepted, including one no rudder can reach and
    one that is absurdly small."""
    ctx = _armed()
    ctx.act("controls_rebuild")
    ctx.S["controls"]["vertical"]["rudder_beta_deg"] = None
    assert ctx.act("controls_rudder_rec")["target"] == pytest.approx(10.0)
    for target in (0.5, 3.0, 999.0):
        ctx.S["controls"]["vertical"]["rudder_beta_deg"] = target
        assert ctx.act("controls_rudder_rec")["target"] == \
            pytest.approx(target)
    # a fraction the recommendation would never pick is still accepted
    ctx.S["controls"]["vertical"]["rudder_chord_frac"] = 0.02
    ctx.act("controls_rebuild")
    assert ctx.S["controls"]["deck"] is not None
    capsys.readouterr()


def test_the_recommended_rudder_is_a_NUMBER_on_the_panel(capsys):
    """"rudder should be given a recomended value" — a value, on the view,
    without pressing anything. A recommendation you have to ask for is not
    one."""
    ctx = _armed()
    ctx.render("controls", "vertical")
    text = _view_text(ctx, "controls", "vertical")
    assert "recommended hinge chord" in text
    assert "hold this much sideslip" in text
    capsys.readouterr()


def test_the_stations_are_ON_the_panel_with_the_picture(capsys):
    """"the x position of the wing and satbiliser should be presented ...
    along side a 2D lateral view". Both: the picture cannot be typed into
    and the table cannot be read at a glance."""
    ctx = _armed()
    ctx.render("controls", "vertical")
    text = _view_text(ctx, "controls", "vertical")
    assert "side elevation" in text
    assert "wing x" in text and "stabiliser x" in text
    assert "neutral point" in text
    assert "leading edge at x [m]" in text
    assert "<svg" in _view_html(ctx, "controls", "vertical")
    capsys.readouterr()


# ------------------------------------------------------------------ the lag

def test_editing_a_control_surface_does_not_REPAINT_the_first_four_stages(
        capsys):
    """"when accessing control for first time in a run it goes extremely
    laggy."

    ``ctx.refresh()`` repaints every DERIVED view — mission/operating,
    mission/point, mission/search, airfoil/screen, airfoil/optimise and
    wing/solver. Not one of them can quote a control surface: an aileron
    band is not in the design vector, not in the mission and not in the
    design box, and stage 5 runs entirely after the search. Measured in a
    browser on a seeded shell, ONE switch on this stage:

        before   1360 ms   6829 DOM mutations   1264 nodes added
        after     253 ms    168 DOM mutations     50 nodes added

    The second half of the same fix: only the view being LOOKED AT is
    rebuilt. The other three are re-rendered by ``Ctx.select`` the moment
    their tab is clicked, so building them here builds them twice.
    """
    ctx = _armed()
    ctx.S["ui"]["tab"]["controls"] = "vertical"
    ctx.render("controls", "vertical")
    assert [v for (s_, v) in ctx.derived if s_ != "controls"], \
        "no V3 stage declared a derived view — this test proves nothing"

    seen = []
    for key, fn in list(ctx.renderers.items()):
        def wrapped(k=key, f=fn):
            seen.append(k)
            return f()
        ctx.renderers[key] = wrapped

    ctx.act("controls_apply_rudder")           # a real setter, through edit()

    assert seen, "the edit rendered nothing at all"
    assert {st for st, _v in seen} == {"controls"}, \
        f"stage 5 repainted stages it cannot move: {sorted(set(seen))}"
    assert seen == [("controls", "vertical")], \
        f"the edit rebuilt views nobody is looking at: {seen}"
    capsys.readouterr()


# ---------------------------------------------------------------- the thrust

def test_the_MAX_thrust_field_does_not_rebuild_the_view(capsys):
    """A lever beside the picture must not tear the picture down.

    ``edit`` on this stage re-trims and calls ``ctx.render("flight")``, which
    rebuilds the Fly view — a fresh WebGL context, a re-fetch of the aircraft
    mesh, and a STOPPED simulation, because the render tears the frame loop
    down as its first act. The thrust ceiling is only a stop: nothing about
    the aeroplane changes when it moves, so it gets its own setter.
    """
    ctx = _armed()
    F = _flying(ctx)
    before = ctx.act("flight_probe")
    ids, state = before["object_ids"], F["state"]
    ctx.act("flight_set_thrust_max", type("E", (), {"value": 700.0})())
    after = ctx.act("flight_probe")
    assert F["thrust_max_n"] == pytest.approx(700.0)
    assert after["thrust_band"][1] == pytest.approx(700.0)
    assert after["object_ids"] == ids, "the scene was rebuilt"
    assert F["state"] is state, "the simulation was re-armed"
    assert F["running"] is True, "moving a stop stopped the simulation"
    capsys.readouterr()


def test_lowering_the_ceiling_PULLS_THE_THROTTLE_BACK(capsys):
    """A throttle sitting above its own ceiling is a stop that is not one.
    Run the thrust up, then type a ceiling below it."""
    ctx = _armed()
    F = _flying(ctx)
    F["hold"]["thrust"] = +1
    for _ in range(240):
        ctx.act("flight_advance", 1 / 60)
    F["hold"]["thrust"] = 0
    high = float(F["live"]["thrust_n"])
    assert high > 1.0
    ctx.act("flight_set_thrust_max",
            type("E", (), {"value": 0.5 * high})())
    assert float(F["live"]["thrust_n"]) == pytest.approx(0.5 * high), \
        "the throttle was left above the ceiling that was just typed"
    # ...and the AEROPLANE is flying the clamped thrust, not the old one:
    # clamping the number and leaving the aircraft at the old force is the
    # same lie one level down
    assert float(F["ac"].prop.thrust_n) == pytest.approx(0.5 * high)
    # ...and the caption beside it says the NEW band, without the panel
    # having been rebuilt: a stale "0 to 54 N" under a typed 180 is a number
    # that is wrong on the screen while being right in the model
    txt = ctx.act("flight_thrust_text")
    assert txt.startswith(f"{float(F['live']['thrust_n']):,.0f} N")
    assert f"({0:,.0f} – {0.5 * high:,.0f} N set)" in txt, txt
    capsys.readouterr()


def test_the_CURRENT_thrust_rides_the_frame_and_not_a_widget(capsys):
    """"Thrust available should present the Max thrust value which the user
    can change and the current thrust."

    The current value cannot be a Quasar widget: W and S move it thirty
    times a second and an element update once a frame costs a quarter of the
    frame (gui/v4/live.py). So it travels in the frame payload, beside the
    read-out and the banner, and the nicegui label it is written into stays
    empty of frame writes.
    """
    ctx = _armed()
    F = _flying(ctx)
    ctx.act("flight_readout")
    p0 = ctx.act("flight_probe")
    t0 = float(F["live"]["thrust_n"])
    lo, hi = p0["thrust_band"]
    assert p0["payload"]["txt"]["thrust"] == \
        f"{t0:,.0f} N   ({lo:,.0f} – {hi:,.0f} N auto)"
    assert p0["install"]["thrust"] is not None, \
        "the browser half was never told which element the thrust is in"

    F["hold"]["thrust"] = +1
    for _ in range(120):
        ctx.act("flight_advance", 1 / 60)
    ctx.act("flight_readout")
    p1 = ctx.act("flight_probe")
    t1 = float(F["live"]["thrust_n"])
    assert t1 > t0
    assert p1["payload"]["txt"]["thrust"].startswith(f"{t1:,.0f} N")
    assert p1["payload"]["txt"]["thrust"] != p0["payload"]["txt"]["thrust"]
    # ...and the element itself was never written to
    assert p1["thrust_label"] == p0["thrust_label"], \
        "the frame path wrote a nicegui element — that is a page patch"
    capsys.readouterr()


# ---------------------------------------------------- the panel, ON THE PANE

def test_the_rate_number_FITS_the_column_it_is_drawn_in(capsys):
    """"On the left I also want to see the roll, yaw, and pitch rates."

    They were there — computed, pushed thirty times a second, and pinned by
    two tests. What was wrong is that you could not READ them: the rate was
    12 px against the angle's 17, in the narrowest column on the panel, and
    a six-character rate is wider than the gap it was given, so it ran back
    under its own bar.

    The widest string the format can produce is checked against the column
    rather than eyeballed, so moving either end of the bar cannot squeeze it
    again.
    """
    from gui.v4 import gauge as gg

    widest = max(
        len(gg.attitude_state(roll_deg=0.0, pitch_deg=0.0, yaw_deg=0.0,
                              rates_dps=(p, p, p))["text"]["g-rate-roll"])
        for p in (-999.9, -103.0, +0.0, +48.4))
    assert widest == 6
    assert widest * gg.RATE_FONT * gg.RATE_EM <= gg.RATE_COL_W, \
        (f"a {widest}-character rate at {gg.RATE_FONT:.0f} px needs "
         f"{widest * gg.RATE_FONT * gg.RATE_EM:.1f} px and the column is "
         f"{gg.RATE_COL_W:.1f}")
    # ...and it is not a footnote beside the angle any more
    assert gg.RATE_FONT >= 14.0
    del capsys


def test_every_rate_row_is_inside_the_panels_viewBox():
    """The gauge's own lesson: a typed height once left a whole row outside
    the box, invisible to every test because the markup and the state were
    both correct. Read the shipped skeleton back."""
    import re

    from gui.v4 import gauge as gg

    s = gg.attitude_skeleton()
    vb = [float(v) for v in
          re.search(r'viewBox="([^"]+)"', s).group(1).split()]
    W, H = vb[2], vb[3]
    seen = 0
    for m in re.finditer(r'<(text|rect) ([^/>]+)[/>]', s):
        a = dict(re.findall(r'(\w+)="([^"]*)"', m.group(2)))
        if "id" not in a or not a["id"].startswith(("g-rate", "g-ratebar",
                                                    "g-ang")):
            continue
        x = float(a["x"]) + float(a.get("width", 0.0))
        y = float(a["y"]) + float(a.get("height", 0.0))
        assert 0.0 <= x <= W and 0.0 <= y <= H, f"outside the box: {a}"
        seen += 1
    assert seen == 9, f"expected three angles, three bars, three rates: {seen}"
    # the font stack must be single-quoted, or the attribute closes on
    # "SF Mono" and the rest is parsed as bare attributes on the <svg>
    assert '"SF Mono"' not in s


def test_the_fly_view_is_sized_against_its_PANE_and_not_the_window(capsys):
    """The view lives in one of four panes and was measured against the
    WINDOW. On a 763 px viewport the work pane is ~520 px, so a 463 px
    picture under 145 px of chrome overflowed by 90 — and scrolling far
    enough to read the numbers under the picture took the top of the left
    panel, the bank dial and the whole ROLL row, off the top of the pane.

    Asserted as the absence of a window unit anywhere in the Fly view's own
    geometry, plus the flex chain that replaced it.
    """
    ctx = _armed()
    ctx.render("controls", "derivatives")
    ctx.render("flight", "fly")
    box = ctx.views[("flight", "fly")]
    styles = [(getattr(e, "_style", None) or {}) for e in box.descendants()]
    styles.append(getattr(box, "_style", None) or {})
    flat = " ".join(f"{k}:{v}" for st in styles for k, v in st.items())
    assert "vh" not in flat, \
        f"the Fly view still measures itself against the window: {flat}"
    # the chain that makes it fill the pane instead
    assert (box._style.get("height") == "100%"
            and box._style.get("min-height") == "0"), box._style
    rows = [st for st in styles if st.get("flex") == "1 1 auto"
            and st.get("min-height") == "340px"]
    assert len(rows) == 1, "the picture row does not take what is left"
    rest = [st for st in styles if st.get("overflow-y") == "auto"
            and st.get("flex") == "0 1 auto"]
    assert len(rest) == 1, \
        "what is under the picture has no scroller of its own, so reaching " \
        "it scrolls the picture and the attitude panel away"
    capsys.readouterr()


def test_the_thrust_CAPTION_cannot_go_stale(capsys):
    """A card that draws once goes stale. The panel deliberately does NOT
    rebuild when the ceiling is typed — rebuilding it would tear the picture
    down — so the ceiling has to live in the string the FRAME writes, beside
    the value, and not in a caption printed at build time."""
    ctx = _armed()
    F = _flying(ctx)
    ctx.act("flight_readout")
    before = ctx.act("flight_thrust_text")
    assert f"{ctx.act('flight_probe')['thrust_band'][1]:,.0f} N auto)" \
        in before
    ctx.act("flight_set_thrust_max", type("E", (), {"value": 180.0})())
    assert F["thrust_max_n"] == 180.0
    ctx.act("flight_readout")
    txt = ctx.act("flight_probe")["payload"]["txt"]["thrust"]
    assert "180" in txt, f"the band beside the value is stale: {txt}"
    assert "(0 – 180 N set)" in txt, txt
    # ...and nothing on the panel PRINTS a band at build time any more
    text = _view_text(ctx, "flight", "fly")
    assert "run between 0 and" not in text
    capsys.readouterr()


# ============================================================================
# A FIELD WHOSE BLANK MEANS A DEFAULT OPENS ON THAT DEFAULT
#
#   "The height, chord and x position is empty (there is a design but the
#    box it empty)"
#   "Max thrust section is empty (default was 52 but the box is empty)"
#
# Five fields, one defect. Blank means "the model chooses", which is the
# right RULE and was the wrong PICTURE: the number existed, the lattice was
# flying it, and the one place a user looks for it said nothing.
# ============================================================================

def _numbers_of(ctx, stage, view):
    """Every ui.number in a built view, in page order."""
    from nicegui import ui

    out = []

    def walk(el):
        for ch in el.default_slot.children:
            if isinstance(ch, ui.number):
                out.append(ch)
            walk(ch)

    walk(ctx.views[(stage, view)])
    return out


def test_the_three_fin_fields_OPEN_ON_THE_FIN_THAT_IS_FLOWN(capsys):
    """Not empty, and not a restatement of the rule either: the number in
    the box is read off the surface the lattice BUILT.

    Pinned to the built fin rather than to 0.12*b so that a shell which
    recomputed the default from its own copy of the rule would fail here the
    moment the two disagreed.
    """
    ctx = _armed()
    ctx.render("controls", "vertical")
    v = ctx.S["controls"]["vertical"]
    assert (v["height_m"], v["chord_m"], v["x_le_m"]) == (None, None, None), \
        "the fields must still STORE nothing — a blank follows the design"
    g = ctx.act("controls_fin")
    shown = [n.value for n in _numbers_of(ctx, "controls", "vertical")][:3]
    # to 0.1 mm: the raw default is 0.6635416666666665, which overflows the
    # box and reads as noise. Only the DEFAULT is rounded, never an answer.
    assert shown == pytest.approx([g["h"], g["c"], g["x_le"]], abs=5e-5), \
        f"the boxes show {shown}, the fin that is flown is {g}"
    assert all(x is not None for x in shown)
    capsys.readouterr()


def test_a_blank_fin_field_still_FOLLOWS_the_design(capsys):
    """The half that makes storing nothing worth it. Show the default and
    store it, and the fin stops tracking the design it is bolted to.

    WHAT it tracks moved in V5: with a ``geometry["fin"]`` block the default
    is the DESIGN's fin, and without one it is still 12 % of the span. Both
    are exercised, because a change that made only one of them follow would
    otherwise pass.
    """
    # (a) with a fin block: the fin the DESIGN states
    ctx = _armed()
    ctx.render("controls", "vertical")
    rep = ctx.S["run"]["report"]
    assert rep["geometry"].get("fin"), "this design should carry a fin block"
    h0 = ctx.act("controls_fin")["h"]
    rep["geometry"]["fin"]["height_m"] = 2.0 * float(
        rep["geometry"]["fin"]["height_m"])
    ctx.act("controls_rebuild")
    ctx.render("controls", "vertical")
    h1 = ctx.act("controls_fin")["h"]
    assert h1 == pytest.approx(2.0 * h0), \
        "a blank height stopped following the design's own fin"
    assert _numbers_of(ctx, "controls", "vertical")[0].value == \
        pytest.approx(h1, abs=5e-5)
    assert ctx.S["controls"]["vertical"]["height_m"] is None

    # (b) WITHOUT one — a report older than the block — the span rule stands
    ctx = _armed()
    ctx.render("controls", "vertical")
    rep = ctx.S["run"]["report"]
    rep["geometry"].pop("fin")
    ctx.act("controls_rebuild")
    ctx.render("controls", "vertical")
    h0 = ctx.act("controls_fin")["h"]
    rep["geometry"]["b"] = 2.0 * float(rep["geometry"]["b"])
    ctx.act("controls_rebuild")
    ctx.render("controls", "vertical")
    assert ctx.act("controls_fin")["h"] == pytest.approx(2.0 * h0), \
        "with no fin block a blank height stopped following the span"
    capsys.readouterr()


def test_a_PINNED_fin_field_says_so_and_says_what_clearing_gives_back(capsys):
    """"1.200" typed by the user and "1.200" chosen by the model are the same
    pixels, so the row has to say which. And the fallback it quotes is ASKED
    of build_flight_model, not restated — a copy of the rule here would drift
    the first time it moved, WHICH IT DID: V5 made the default the design's
    own fin (``geometry["fin"]``, sized by volume coefficient) instead of
    12 % of the span, and the version of this test that wrote ``0.12 * b``
    went red for the right reason. So the expected number comes from the
    report, which is where the design states it."""
    ctx = _armed()
    ctx.S["controls"]["vertical"]["height_m"] = 2.5
    ctx.act("controls_rebuild")
    ctx.render("controls", "vertical")
    assert _numbers_of(ctx, "controls", "vertical")[0].value == \
        pytest.approx(2.5)
    text = _view_text(ctx, "controls", "vertical")
    assert "YOURS, not the model's" in text
    rep = ctx.S["run"]["report"]
    blk = (rep["geometry"].get("fin") or {})
    want = (abs(float(blk["height_m"])) if blk
            else 0.12 * float(rep["geometry"]["b"]))
    assert f"{want:.4g}" in text, \
        "the row does not say what clearing the box would give back"
    capsys.readouterr()


def test_clearing_a_pinned_fin_field_puts_the_model_back(capsys):
    """A box here can never stay empty: clearing it stores None, and None
    re-renders as the model's own number."""
    ctx = _armed()
    ctx.S["controls"]["vertical"]["chord_m"] = 1.9
    ctx.act("controls_rebuild")
    ctx.render("controls", "vertical")
    assert _numbers_of(ctx, "controls", "vertical")[1].value == \
        pytest.approx(1.9)
    # the real handler, with the blank a cleared box actually sends
    _numbers_of(ctx, "controls", "vertical")[1].set_value(None)
    assert ctx.S["controls"]["vertical"]["chord_m"] is None
    ctx.render("controls", "vertical")
    shown = _numbers_of(ctx, "controls", "vertical")[1].value
    assert shown == pytest.approx(ctx.act("controls_fin")["c"], abs=5e-5)
    assert shown is not None and shown != pytest.approx(1.9)
    capsys.readouterr()


def test_the_MAX_THRUST_box_opens_on_the_ceiling_in_force(capsys):
    """"default was 52 but the box is empty"."""
    ctx = _armed()
    F = _flying(ctx)
    assert F["thrust_max_n"] is None
    hi = ctx.act("flight_probe")["thrust_band"][1]
    assert hi > 0
    shown = [n.value for n in _numbers_of(ctx, "flight", "fly")]
    assert shown and shown[0] == pytest.approx(hi, abs=0.05), \
        f"the max-thrust box shows {shown[:1]}, the stop is {hi}"
    capsys.readouterr()


def test_the_thrust_caption_says_WHOSE_ceiling_it_is(capsys):
    """The field opens on the ceiling either way, so the difference between
    "the model chose 54" and "you typed 54" has to be said somewhere that
    cannot go stale — and this panel deliberately does not rebuild when the
    field is answered, so it rides the frame."""
    ctx = _armed()
    F = _flying(ctx)
    ctx.act("flight_readout")
    assert ctx.act("flight_thrust_text").endswith("auto)")
    ctx.act("flight_set_thrust_max", type("E", (), {"value": 240.0})())
    assert ctx.act("flight_thrust_text").endswith("240 N set)")
    # ...and clearing it goes back to the model's own ceiling AND says so
    ctx.act("flight_set_thrust_max", type("E", (), {"value": None})())
    assert F["thrust_max_n"] is None
    assert ctx.act("flight_thrust_text").endswith("auto)")
    capsys.readouterr()


def test_the_DRAG_POLAR_boxes_open_on_what_the_aircraft_flies(capsys):
    """CD0 and the span efficiency are the same defect: blank means the
    report's own number, and blank read as "no answer"."""
    ctx = _armed()
    F = _flying(ctx)
    ctx.render("flight", "setup")
    assert F["CD0"] is None and F["oswald_e"] is None
    shown = [n.value for n in _numbers_of(ctx, "flight", "setup")]
    assert float(F["ac"].CD0) in [pytest.approx(v, abs=5e-6) for v in shown], \
        shown
    assert float(F["ac"].oswald_e) in [pytest.approx(v, abs=5e-4)
                                       for v in shown], shown
    capsys.readouterr()


# ---------------------------------------------------------------- the cache

def test_the_recommendations_are_bought_ONCE_PER_REBUILD(capsys):
    """The Fin & rudder view repaints on every edit, on every tab click and
    once at assembly, and it was paying six lattices each time — one for the
    Reynolds recommendation, four for the rudder's solve, one for the
    all-blank fin. They depend on nothing but the answers the rebuild read.

    Counted at ``build_flight_model``, which is the thing that costs.
    """
    import aerobo.flightmodel as fmod

    ctx = _armed()
    ctx.act("controls_rebuild")
    ctx.render("controls", "vertical")          # first paint fills the cache

    calls = []
    real = fmod.build_flight_model
    try:
        fmod.build_flight_model = lambda *a, **k: (calls.append(1),
                                                   real(*a, **k))[1]
        for _ in range(4):
            ctx.render("controls", "vertical")
        assert calls == [], \
            f"{len(calls)} lattices for four repaints of the same deck"
        # ...and a REBUILD throws the answers away with the deck
        ctx.act("controls_rebuild")
        ctx.render("controls", "vertical")
        assert calls, "the cache outlived the deck it describes"
    finally:
        fmod.build_flight_model = real
    capsys.readouterr()


def _same_answer(a, b, *, rel=1e-9):
    """Two recommendations agree — to floating point, not bit for bit.

    This package's own lesson, three times over: the lattice solve is not
    bit-reproducible on this machine, and two identical calls differ in the
    twelfth figure. An identity assertion here would be a flake, and
    loosening it later to make the flake go away is how a real regression
    gets waved through. Same KEYS exactly; same numbers to 1e-9.
    """
    assert set(a) == set(b), (sorted(set(a) ^ set(b)))
    for k in a:
        if isinstance(a[k], float) and isinstance(b[k], float):
            assert a[k] == pytest.approx(b[k], rel=rel), k
        else:
            assert a[k] == b[k], k


def test_the_cache_does_not_change_the_ANSWER(capsys):
    """A cache that changes what is reported is not a cache. Compare the
    memoised recommendation with one computed against an empty cache."""
    ctx = _armed()
    ctx.act("controls_rebuild")
    warm = ctx.act("controls_rudder_rec")
    cold_re = ctx.act("controls_reynolds")
    ctx.S["controls"]["rec_cache"] = {}
    _same_answer(ctx.act("controls_rudder_rec"), warm)
    _same_answer(ctx.act("controls_reynolds"), cold_re)
    # and with NO cache dict at all it still answers
    ctx.S["controls"].pop("rec_cache", None)
    _same_answer(ctx.act("controls_rudder_rec"), warm)
    capsys.readouterr()


def test_ENTERING_stage_5_is_what_builds_the_deck(capsys):
    """"When control section is clicked the screen goes grey and then flight
    and control is available."

    Stage 5 opens on Ailerons & elevator, and that view read no derivative,
    so
    it built nothing. Stage 5 therefore stayed "ready" instead of "done" and
    stage 6 — locked until stage 5 has produced a deck — stayed LOCKED,
    until the user happened to click one of the other three tabs. The second
    half arrived for no reason the user could see.

    Driven through ``ctx.select``, which is what clicking the tree does, and
    asserted on the STAGE STATES, which is what the tree draws.
    """
    ctx = _armed()
    assert ctx.S["ui"]["tab"]["controls"] == "surfaces", \
        "this test is about the tab the stage OPENS on"
    before = session.stage_states(ctx.S)
    assert before["controls"][0] == "ready"
    assert before["flight"][0] == "locked"

    ctx.select("controls")

    assert ctx.S["controls"]["deck"] is not None, \
        "entering the stage built no deck"
    after = session.stage_states(ctx.S)
    assert after["controls"][0] == "done"
    assert after["flight"][0] != "locked", \
        "stage 6 is still locked after stage 5 was opened"
    capsys.readouterr()


def test_an_ANSWER_shows_even_before_anything_is_built():
    """Why ``shown_value`` branches on ``stored`` at all.

    Once a model exists the two agree — a pinned height IS the height the
    lattice built from it — so the branch looks redundant and a mutation
    that drops it passes every view test. It is not redundant in the one
    state where they differ: an answer typed before (or without) a build.
    Blank there is genuinely empty; an answer there is the answer.
    """
    from gui.v4 import fields as fl

    assert fl.shown_value(2.5, None) == pytest.approx(2.5)
    assert fl.shown_value(None, None) is None
    # ...and an ANSWER is never rounded, only a default is
    assert fl.shown_value(0.123456789, None, 4) == pytest.approx(0.123456789)
    assert fl.shown_value(None, 0.123456789, 4) == pytest.approx(0.1235)


def test_a_defaulted_box_is_ROUNDED_enough_to_read():
    """The model's own fin chord is 0.6635416666666665 and its throttle
    ceiling 53.99481778190805. Eighteen digits overflow the box and read as
    noise, and four of them matter."""
    from gui.v4 import fields as fl

    for flown, dp, want in ((0.6635416666666665, 4, "0.6635"),
                            (53.99481778190805, 1, "54.0"),
                            (0.007191076495494447, 5, "0.00719")):
        got = fl.shown_value(None, flown, dp)
        assert repr(float(got)).rstrip("0").rstrip(".") == \
            want.rstrip("0").rstrip("."), (flown, dp, got)
        assert len(repr(float(got))) <= 8, (flown, dp, got)
