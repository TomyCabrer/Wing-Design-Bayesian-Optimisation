"""Stage 1 answers the wing-loading question instead of just asking it.

``CL = W/(qS)`` — so W/S decides the operating point before any aerodynamics
happens, and a bare number field asked for it with no way to answer it. The
card now carries the constraint diagram (``aerobo.constraint_diagram``): the
mission's requirements go in, the band they leave comes out with the binding
constraint named, and one click writes the answer into the same field. Typing
W/S by hand is untouched — this is an answer to that question, not a second
question.

Two rules it has to keep:

* the OPERATING POINT is not asked for here. The mission already states the
  speed, the altitude (and so the density) and the depth, and the session
  already owns an aspect ratio (stage 3's flown one, or stage 2's estimate
  while stage 3 follows it) — the diagram takes all four. What it asks for is
  only what the mission does NOT state: the stall / fly-up speed, CL_max, the
  drag pair, and whichever field-length or manoeuvre requirements apply;
* an untouched session is unchanged: the card computes, it does not write.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def test_the_diagram_never_asks_for_the_operating_point_twice():
    """Speed, density, depth and the aspect ratio are already answered — by
    the mission fields beside the card and by the wing — so the diagram TAKES
    them. Asking again would be the same number in two places."""
    from gui.v3 import session

    S = session.make_session("air")
    asked = session.ws_inputs(S)
    for key in ("v_cruise_ms", "rho_cruise", "aspect_ratio", "altitude_m"):
        assert key not in asked, key

    # move the mission's speed and the diagram follows it
    before = session.ws_diagram(S).ws_min_drag_pa
    S["mission"]["V"] = 22.0
    assert session.ws_diagram(S).ws_min_drag_pa != before

    # ...and the aspect ratio the session owns likewise (stage 2's estimate:
    # nobody types a wing aspect ratio, the span is what stage 3 constrains)
    before = session.ws_diagram(S).ws_min_drag_pa
    session.set_section_aspect_ratio(S, 20.0)
    assert session.ws_diagram(S).ws_min_drag_pa != before


def test_the_water_diagram_takes_the_speed_the_depth_and_the_water():
    from gui.v3 import session

    S = session.make_session("water")
    asked = session.ws_inputs(S)
    for key in ("v_max_ms", "depth_m", "rho", "p_vap"):
        assert key not in asked, key

    cav = next(c for c in session.ws_diagram(S).constraints
               if c.name == "cavitation")
    S["mission"]["depth_m"] = float(S["mission"]["depth_m"]) + 0.5
    deeper = next(c for c in session.ws_diagram(S).constraints
                  if c.name == "cavitation")
    assert deeper.ws_pa > cav.ws_pa       # more static head, more margin


def test_the_one_speed_it_does_ask_for_opens_beside_the_missions():
    """The stall speed (fly-up, in water) is the one speed the mission does
    not state — so the card opens it AT a fraction of the mission's, not at
    some number of its own. A card that opened at an 18 m/s stall beside a
    14.6 m/s cruise would describe no mission at all."""
    from gui.v3 import session

    S = session.make_session("air")
    v = float(S["mission"]["V"])
    assert session.ws_inputs(S)["v_stall_ms"] == pytest.approx(
        round(session.STALL_SPEED_FRAC * v, 2))
    assert session.ws_inputs(S)["v_stall_ms"] < v
    assert session.ws_diagram(S) is not None

    # ...AND NEVER BELOW THE SPEED THE CRAFT ACTUALLY FLIES AT. In water the
    # fraction alone asked a 0.144 m² foil to carry 6 kN at 6 m/s — CL 2.26
    # against the 0.9 the card is drawn with — so every fresh session opened
    # 2.51x over its own ceiling and a sized run refused every design.
    W = session.make_session("water")
    v_w = float(W["mission"]["V"])
    opened = session.ws_inputs(W)["v_takeoff_ms"]
    assert opened > session.FLYUP_SPEED_FRAC * v_w      # the fraction lost
    assert opened == pytest.approx(round(
        session.OPENING_SPEED_MARGIN
        * session.stall_speed_at(W, session.published_wing_loading(W),
                                 cl_max=session.ws_inputs(W)["cl_max"]), 2))
    assert opened < v_w


def test_cl_max_comes_from_the_section_not_from_the_user():
    """"How would I know CL_max?" — you would not: the SECTION does, and
    stage 2 already screened it (``clmax`` off airfoil_select's stall sweep).
    The wing's is that times a stated allowance, and the card shows it as a
    read-out instead of a field."""
    from gui.v3 import session

    S = session.make_session("air")
    assert session.section_cl_max(S) is None      # nothing chosen yet

    S["airfoil"]["decision"] = "library"
    S["airfoil"]["section"] = {"name": "goe398", "clmax": 1.55, "tc": 0.12}
    cl, why = session.section_cl_max(S)
    assert cl == pytest.approx(session.WING_CLMAX_FRAC * 1.55)
    assert "goe398" in why

    # ...and it is what the stall line is drawn from, whatever was typed
    session.set_ws_input(S, "cl_max", 0.4)
    stall = next(c for c in session.ws_diagram(S).constraints
                 if c.name == "stall / approach")
    v_s = session.ws_inputs(S)["v_stall_ms"]
    rho = session.design_point(S)["rho"]
    assert stall.ws_pa == pytest.approx(0.5 * 1.225 * v_s ** 2 * cl)
    assert rho == pytest.approx(1.225)


def test_the_stall_speed_is_a_requirement_not_a_guess():
    """The relation read the other way round: at the wing loading the
    mission already states, the chosen section stalls at a speed the card
    shows — so the field is 'the speed I require it not to exceed'."""
    from gui.v3 import session

    S = session.make_session("air")
    assert session.stall_speed_at(S) is None      # no section, no answer
    S["airfoil"]["decision"] = "library"
    S["airfoil"]["section"] = {"name": "goe398", "clmax": 1.55, "tc": 0.12}

    cl = session.section_cl_max(S)[0]
    ws = session.wing_loading(S)
    v_s = session.stall_speed_at(S)
    assert v_s == pytest.approx((2.0 * ws / (1.225 * cl)) ** 0.5)
    # a heavier wing loading stalls faster — the coupling the card explains
    session.set_wing_loading(S, 2.0 * ws)
    assert session.stall_speed_at(S) > v_s


def test_the_card_shows_cl_max_as_a_read_out_once_a_section_exists(capsys):
    from gui.v3.app import assemble

    ctx = assemble("air")
    S = ctx.S
    S["airfoil"]["decision"] = "library"
    S["airfoil"]["section"] = {"name": "goe398", "clmax": 1.55, "tc": 0.12}
    ctx.render("mission", "operating")
    texts = [getattr(e, "text", "") or ""
             for e in ctx.views[("mission", "operating")].descendants()]
    assert any("CL_max (wing)" in t for t in texts), texts[:6]
    assert any("do not have to guess the stall speed" in t for t in texts)
    err = capsys.readouterr().err
    assert "Traceback" not in err, err


def test_the_air_diagram_recommends_and_names_its_binding_constraint():
    from gui.v3 import session

    S = session.make_session("air")
    r = session.ws_diagram(S).recommend()
    assert r["binding_constraint"] == "stall / approach"
    assert r["wing_loading_pa"] > 0.0
    assert r["twr_required"] > 0.0


def test_the_water_diagram_is_the_foils_own_two_limits():
    from gui.v3 import session

    S = session.make_session("water")
    d = session.ws_diagram(S)
    assert {c.name for c in d.constraints} == {"fly-up", "cavitation"}
    assert d.ws_max_pa == min(c.ws_pa for c in d.constraints)


def test_the_track_has_no_wing_loading_to_derive():
    from gui.v3 import session

    S = session.make_session("track")
    assert session.ws_diagram(S) is None


def test_clearing_a_requirement_drops_its_line():
    from gui.v3 import session

    S = session.make_session("air")
    assert session.set_ws_input(S, "climb_rate_ms", None)
    assert all(c.name != "climb" for c in session.ws_diagram(S).constraints)
    assert session.set_ws_input(S, "takeoff_distance_m", 400.0)
    assert any(c.name == "take-off distance"
               for c in session.ws_diagram(S).constraints)
    assert not session.set_ws_input(S, "not_a_requirement", 1.0)


def test_adopting_it_writes_the_area_the_solver_will_fly(capsys):
    """...and "the solver will fly" is asserted against the built config.

    This test used to stop at the mission dict, which is the one place the
    adopted size was never in doubt. The handler wrote ``s_ref_m2`` and did
    NOT run ``session.sync_wing_from_mission``, so the area never reached
    ``S["wing"]["flags"]``: the card, the log line and stage 3's design box
    all quoted 8.6806 m² while ``config.cfg_dict`` carried no ``S_m2``/
    ``b_m`` at all and the run flew the family's published 10 m² / 10 m.
    """
    from gui.v3 import config, session
    from gui.v3.app import assemble

    ctx = assemble("air")
    S = ctx.S
    before = session.wing_loading(S)
    fresh = config.cfg_dict(S).get("flags") or {}
    assert fresh.get("S_m2") is None, "a fresh session states no size"
    ws = ctx.act("adopt_ws")
    after = session.wing_loading(S)
    # the action returns what it wrote (it used to return None, which made
    # the assertion that stood here true for every possible outcome)
    assert ws == pytest.approx(after)
    assert after != before
    assert after == pytest.approx(session.ws_diagram(S).ws_max_pa)
    # the AREA is the stored quantity, and it followed
    area = float(S["mission"]["s_ref_m2"])
    assert area == pytest.approx(float(S["mission"]["W_N"]) / after)
    # ...and the RUN is built from it: the flags the config carries are the
    # adopted size, not the family's published one
    flags = config.cfg_dict(S).get("flags") or {}
    assert flags.get("S_m2") == pytest.approx(area)
    assert flags.get("b_m") is not None
    assert flags["b_m"] ** 2 / flags["S_m2"] == pytest.approx(
        session.nominal_aspect_ratio(S))
    err = capsys.readouterr().err
    assert "Traceback" not in err, err


def test_an_untouched_session_is_untouched(capsys):
    """The card computes; it does not write. A fresh session still assembles
    the published run bit-for-bit."""
    from gui.v3 import config, session
    from gui.v3.app import assemble

    ctx = assemble("air")
    ctx.render("mission", "operating")
    d = config.cfg_dict(ctx.S)
    assert d["problem_name"] == "wing (free chord law)"
    assert d["mission_kwargs"] == {}
    assert config.physics_flags(d) == config.stated_physics(d["problem_name"])
    assert session.wing_loading(ctx.S) == pytest.approx(
        session.wing_loading(session.make_session("air")))
    err = capsys.readouterr().err
    assert "Traceback" not in err, err


def test_the_card_renders_its_figure_and_its_recommendation(capsys):
    """Each medium's operating card draws its OWN explanation of the sizing.

    The track's phrase moved with the card. "carries no weight" was the
    wing-loading block's placeholder for a medium that has no diagram; the
    track card is purpose-built now — it asks the two numbers a rear wing
    actually reads and says where the size comes from instead — so the
    assertion is on the sentence that carries the same fact: a rear wing is
    bounded by a regulation or the bodywork, and there is no weight to
    divide by a loading in the first place.
    """
    from gui.v3.app import assemble

    for medium, phrase in (("air", "stall / approach"),
                           ("water", "fly-up"),
                           ("track", "no weight to divide by")):
        ctx = assemble(medium)
        ctx.render("mission", "operating")
        texts = [getattr(e, "text", "") or ""
                 for e in ctx.views[("mission", "operating")].descendants()]
        assert any(phrase in t for t in texts), (medium, texts[:5])
    err = capsys.readouterr().err
    assert "Traceback" not in err, err


def test_the_figure_draws_every_line_it_was_given():
    from aerobo import constraint_diagram as cd
    from gui import nice_app as v1

    d = cd.air_diagram(cd.AirMission(
        v_stall_ms=18.0, cl_max=1.6, v_cruise_ms=45.0, climb_rate_ms=4.0,
        takeoff_distance_m=400.0, landing_distance_m=350.0,
        turn_load_factor=2.0))
    fig = v1.fig_constraint_diagram(d)
    names = {t.name for t in fig.data}
    assert {"cruise", "climb", "required T/W"} <= names
    assert any("limit" in n for n in names)
    # ...and the water one degrades to the band and its edges
    w = cd.water_diagram(cd.WaterMission(v_takeoff_ms=6.0, v_max_ms=14.0,
                                         cl_max=0.9))
    assert len(v1.fig_constraint_diagram(w).data) == 2
