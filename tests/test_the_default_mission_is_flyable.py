"""The mission a fresh session OPENS on must not be refused by its own card.

Every V3 session opens on a family's published design point, and stage 1 then
derives a wing-loading ceiling from requirements the mission does not state:
the stall (fly-up) speed, and the section's suction peak. Both were defaults
with nothing under them, and in water both were wrong in the same direction —
a fresh hydrofoil session stated W/S = 41 667 Pa against a ceiling of
16 605 Pa, 2.51x over, before the user typed anything at all. That ceiling is
not decoration: ``config.flags`` sends it to every sized run as
``wing_loading_limit_pa``, so the opening state of the shell was one whose
runs return no design.

What is pinned here:

* the opening REQUIREMENT is one the published craft can meet, in every
  medium that draws a diagram;
* the cavitation line is read off the section the SOLVER flies, not off the
  placeholder pair whose own field comment says "meant to be replaced";
* and neither of those makes the ceiling a function of what the user typed —
  the band a searched W/S gets is still a statement about the mission.
"""

from __future__ import annotations

import pytest

from aerobo import api
from aerobo import constraint_diagram as cd
from gui.v3 import session


# --------------------------------------------------- the headline
@pytest.mark.parametrize("medium", ["air", "water"])
def test_a_fresh_session_states_a_loading_its_own_card_allows(medium):
    """The whole point. ``ws_over_ceiling`` is the card that says "you have
    stated X, this mission allows Y" — on an untouched session it must have
    nothing to say."""
    S = session.make_session(medium)
    stated = session.wing_loading(S)
    cap = session.mission_ws_ceiling(S)
    assert stated and cap
    assert stated <= cap, (f"{medium}: opens {stated / cap:.2f}x over its own "
                           f"ceiling ({stated:.0f} > {cap:.0f} Pa)")
    assert session.ws_over_ceiling(S) is None


def test_the_water_default_was_the_one_that_was_broken():
    """The numbers, so a regression names itself. The margin is
    OPENING_SPEED_MARGIN squared, because W/S goes as V²."""
    S = session.make_session("water")
    assert session.wing_loading(S) == pytest.approx(41666.667, rel=1e-5)
    assert session.mission_ws_ceiling(S) == pytest.approx(45940.68, rel=1e-5)
    assert (session.mission_ws_ceiling(S) / session.wing_loading(S)
            == pytest.approx(session.OPENING_SPEED_MARGIN ** 2, rel=1e-3))


# --------------------------------------------------- the opening speed
def test_the_opening_speed_is_a_speed_the_published_craft_reaches():
    """A requirement the mission cannot meet is not a requirement."""
    W = session.make_session("water")
    v_open = session.ws_inputs(W)["v_takeoff_ms"]
    v_min = session.stall_speed_at(W, session.published_wing_loading(W),
                                   cl_max=session.ws_inputs(W)["cl_max"])
    assert v_min == pytest.approx(9.5044, rel=1e-3)
    assert v_open >= v_min, "opens on a fly-up speed the foil cannot make"
    assert v_open < float(W["mission"]["V"])   # still a TAKE-OFF speed


def test_the_air_default_still_comes_from_the_fraction():
    """The fraction is not deleted — it is a floor, and in air it already
    cleared the speed the published wing flies at (8.16 m/s), so nothing
    about an air session moves."""
    S = session.make_session("air")
    v = float(S["mission"]["V"])
    assert session.ws_inputs(S)["v_stall_ms"] == pytest.approx(
        round(session.STALL_SPEED_FRAC * v, 2))
    assert session.mission_ws_ceiling(S) == pytest.approx(75.202848)


def test_the_ceiling_does_not_follow_the_area_the_user_typed():
    """THE INVARIANT THE FIX MUST NOT BREAK. A searched W/S has no interior
    optimum, so the answer IS the top of its band; a ceiling derived from the
    loading in the FORM would make that answer a function of the area
    somebody happened to type. It is derived from the family's published
    point instead, which does not move."""
    caps, stated = set(), []
    for scale in (0.4, 1.0, 2.5):
        S = session.make_session("water")
        session.set_wing_loading(S, scale * session.wing_loading(S))
        stated.append(session.wing_loading(S))
        caps.add(round(float(session.mission_ws_ceiling(S)), 9))
    assert len(set(round(v, 6) for v in stated)) == 3     # the seeds differed
    assert len(caps) == 1                                 # the ceiling did not


def test_the_opening_speed_stays_below_the_cruise_speed():
    """water_diagram needs v_takeoff < v_max, and a card that cannot be drawn
    at all is a worse answer than a conflict the conflict card can state."""
    S = session.make_session("water")
    v = float(S["mission"]["V"])
    # a section that can only hold cl 0.3 needs 16.5 m/s to lift the
    # published craft — above its own cruise speed, so the requirement is
    # capped and the card still draws
    assert session.opening_speed(S, session.FLYUP_SPEED_FRAC, 0.3) == \
        pytest.approx(session.OPENING_SPEED_CEILING_FRAC * v)
    session.set_ws_input(S, "v_takeoff_ms",
                         session.OPENING_SPEED_CEILING_FRAC * v)
    assert session.ws_diagram(S) is not None


# --------------------------------------------------- the cavitation line
def _water_mission(**kw):
    from aerobo.hydrofoil import water_properties

    w = water_properties("sea")
    base = dict(v_takeoff_ms=6.0, v_max_ms=12.0, cl_max=0.9, depth_m=0.575,
                rho=float(w["rho"]), p_vap=float(w["p_vap"]))
    base.update(kw)
    return cd.WaterMission(**base)


def test_the_line_crosses_where_the_real_section_crosses():
    """cp_min_line is a TANGENT at the crossing, so the linearised law and
    the polar it came from agree on the one number the diagram reads: the cl
    at which -Cp_min reaches the cavitation number."""
    m = _water_mission()
    pol = api.water_family_polar(0.12)
    a, b = cd.cp_min_line(pol, m)
    cl_line = cd.cl_cavitation_limited(cd.WaterMission(
        **{**m.__dict__, "cp_min_a": a, "cp_min_b": b}), m.v_max_ms)
    sigma = cd.cavitation_number(m, m.v_max_ms)
    assert -float(pol.cp_min(float(_alpha_at_cl(pol, cl_line)))) == \
        pytest.approx(sigma, abs=5e-3)
    assert cl_line == pytest.approx(0.7361, rel=2e-3)


def _alpha_at_cl(pol, cl_target):
    import numpy as np

    al = np.arange(-4.0, 12.0, 0.05)
    cl = np.array([float(pol.cl(a)) for a in al])
    return float(np.interp(cl_target, cl, al))


def test_the_line_is_a_tangent_not_a_chord():
    """The slope is the section's OWN at the crossing, so the line stays
    right as the mission's speed moves off the one it was taken at. A chord
    through the same point with the placeholder's slope (2.2) is 0.024 in cl
    out at +5 % speed; the tangent is 0.002."""
    m = _water_mission()
    pol = api.water_family_polar(0.12)
    a, b = cd.cp_min_line(pol, m)
    faster = 12.6
    tangent = cd.cl_cavitation_limited(cd.WaterMission(
        **{**m.__dict__, "cp_min_a": a, "cp_min_b": b}), faster)
    sigma = cd.cavitation_number(m, faster)
    truth = _cl_at_cp_min(pol, sigma)
    assert truth == pytest.approx(0.6996, rel=2e-3)
    assert tangent == pytest.approx(truth, abs=5e-3)
    chord = cd.cl_cavitation_limited(cd.WaterMission(
        **{**m.__dict__, "cp_min_a": sigma_at_12(m) - 2.2 * 0.7361,
           "cp_min_b": 2.2}), faster)
    assert abs(chord - truth) > 0.02       # the slope IS load-bearing


def sigma_at_12(m):
    return cd.cavitation_number(m, 12.0)


def _cl_at_cp_min(pol, sigma):
    """The cl where the section's suction peak reaches ``sigma``, off the
    polar itself — the number the linearisation has to reproduce."""
    import numpy as np

    al = np.arange(-4.0, 12.0, 0.01)
    cl = np.array([float(pol.cl(x)) for x in al])
    mcp = np.array([float(-pol.cp_min(x)) for x in al])
    k = int(np.argmin(mcp))
    j = k + 1 + int(np.nonzero(mcp[k + 1:] > sigma)[0][0])
    return float(np.interp(sigma, [mcp[j - 1], mcp[j]], [cl[j - 1], cl[j]]))


def test_a_section_that_cavitates_at_every_lift_says_zero():
    """Fast and shallow: the bucket's own FLOOR is at vapour pressure, so no
    wing loading is admissible. The line has to say zero, not the lift at
    the floor — and its slope has to come off the rising wall, because the
    slope at the floor is zero and a zero slope has no crossing."""
    m = _water_mission(v_max_ms=40.0, depth_m=0.1)
    pol = api.water_family_polar(0.12)
    a, b = cd.cp_min_line(pol, m)
    assert b > 0.0
    assert a == pytest.approx(cd.cavitation_number(m, 40.0))
    assert cd.cl_cavitation_limited(cd.WaterMission(
        **{**m.__dict__, "cp_min_a": a, "cp_min_b": b}), 40.0) == 0.0


def test_the_placeholder_refused_the_point_the_solver_flies():
    """Why this was a bug and not a preference: at the published design point
    the placeholder pair says CAVITATING and the section says it is not."""
    m = _water_mission()
    sigma = cd.cavitation_number(m, m.v_max_ms)
    cl_design = 6000.0 / (0.5 * m.rho * m.v_max_ms ** 2 * 0.144)
    placeholder = m.cp_min_a + m.cp_min_b * cl_design
    measured = -float(api.water_family_polar(0.12).cp_min(
        _alpha_at_cl(api.water_family_polar(0.12), cl_design)))
    assert cl_design == pytest.approx(0.5646, rel=1e-3)
    assert placeholder > sigma            # the placeholder refuses it
    assert measured < sigma               # the real section does not
    assert measured == pytest.approx(0.961, rel=5e-3)


def test_a_section_that_cannot_answer_leaves_the_pair_alone():
    """No Cp_min table is not a licence to fit one. The caller keeps the
    placeholder, which is at least declared."""
    class _NoCpMin:
        def cl(self, a):
            return 0.1 * a

        def cp_min(self, a):
            raise ValueError("polar 'x' has no Cp_min table")

    assert cd.cp_min_line(_NoCpMin(), _water_mission()) is None
    # ...and the session route falls through to the declared pair rather
    # than dropping the cavitation line off the card
    S = session.make_session("water")
    real = session.cavitation_polar
    try:
        session.cavitation_polar = lambda _S: _NoCpMin()
        assert session.cavitation_line(S, _water_mission()) is None
        d = session.ws_diagram(S)
    finally:
        session.cavitation_polar = real
    cav = [c for c in d.constraints if c.name == "cavitation"][0]
    assert cav.ws_pa == pytest.approx(27493.09, rel=1e-4)   # the placeholder


def test_the_shell_draws_the_line_it_hands_the_solver():
    """The session's own route: an unanswered stage 2 draws the section
    ``HydrofoilProblem`` evaluates its cavitation constraint on."""
    S = session.make_session("water")
    m = _water_mission()
    assert session.cavitation_polar(S).name == \
        api.water_family_polar(0.12).name
    assert session.cavitation_line(S, m) == cd.cp_min_line(
        api.water_family_polar(0.12), m)
    d = session.ws_diagram(S)
    cav = [c for c in d.constraints if c.name == "cavitation"][0]
    assert cav.ws_pa == pytest.approx(54326.4, rel=1e-4)


def test_the_thickness_is_the_one_the_run_will_search():
    """Not a constant: the line follows the design box's t/c."""
    S = session.make_session("water")
    S["wing"]["bounds"]["tc"] = [0.16, 0.16]
    assert session.cavitation_polar(S).name == \
        api.water_family_polar(0.16).name
    assert api.water_family_polar(0.16).name != api.water_family_polar(0.12).name


def test_a_thickness_outside_the_bank_is_clamped_not_refused():
    """A calibration is a default, not a ban."""
    assert api.water_family_polar(0.02).name == api.water_family_polar(0.06).name
    assert api.water_family_polar(0.99).name == api.water_family_polar(0.18).name
