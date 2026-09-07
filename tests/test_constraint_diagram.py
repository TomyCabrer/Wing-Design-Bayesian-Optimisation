"""W/S is a MISSION answer, and this is how it is answered.

``CL = W/(qS)`` is the trim target every problem in this package flies at, so
the wing loading decides the operating point before any aerodynamics happens.
The shells used to ask for it as a bare number. ``constraint_diagram`` is the
standard way it is actually derived — stall/field length/cruise/climb/turn in
air, fly-up and cavitation in water — stated as pure functions so the numbers
can be tested and quoted rather than trusted.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aerobo import constraint_diagram as cd     # noqa: E402


def _air(**kw) -> cd.AirMission:
    base = dict(v_stall_ms=18.0, cl_max=1.6, v_cruise_ms=45.0,
                aspect_ratio=10.0, cd0=0.025, oswald_e=0.85)
    base.update(kw)
    return cd.AirMission(**base)


# ------------------------------------------------------------------- air
def test_the_stall_limit_is_the_definition():
    m = _air()
    d = cd.air_diagram(m)
    expect = 0.5 * cd.RHO_SL * m.v_stall_ms ** 2 * m.cl_max
    stall = next(c for c in d.constraints if c.name == "stall / approach")
    assert stall.ws_pa == pytest.approx(expect)
    assert d.ws_max_pa == pytest.approx(expect)
    assert d.binding == "stall / approach"


def test_the_cruise_curve_bottoms_at_the_minimum_drag_loading():
    """(W/S)* = q√(πAe·CD0) — the classic result, and the number a pure
    aerodynamicist would choose if the mission let them."""
    m = _air()
    d = cd.air_diagram(m, n_grid=4001)
    q = m.q_cruise()
    k = 1.0 / (np.pi * m.aspect_ratio * m.oswald_e)
    assert d.ws_min_drag_pa == pytest.approx(q * np.sqrt(m.cd0 / k))

    cruise = next(c for c in d.constraints if c.name == "cruise")
    i = int(np.argmin(cruise.twr))
    assert d.ws_grid_pa[i] == pytest.approx(d.ws_min_drag_pa,
                                            rel=2.0 / 4001 * 1.6)


def test_a_manoeuvre_requirement_pushes_the_loading_down():
    """The induced term carries n², so asking for a sustained turn asks for
    a bigger wing at the same thrust."""
    plain = cd.air_diagram(_air())
    turning = cd.air_diagram(_air(turn_load_factor=2.5))
    ws = 300.0
    assert turning.twr_at(ws) > plain.twr_at(ws)


def test_climb_is_a_floor_on_thrust_not_on_area():
    d = cd.air_diagram(_air(climb_rate_ms=5.0))
    climb = next(c for c in d.constraints if c.name == "climb")
    assert np.allclose(climb.twr, climb.twr[0])
    assert climb.twr[0] > 0.0


def test_the_landing_field_can_bind_before_the_stall_speed():
    short = cd.air_diagram(_air(landing_distance_m=150.0))
    assert short.binding == "landing distance"
    assert short.ws_max_pa < 0.5 * cd.RHO_SL * 18.0 ** 2 * 1.6


def test_the_diagram_says_when_the_aerodynamic_optimum_is_out_of_reach():
    d = cd.air_diagram(_air())
    assert d.ws_min_drag_pa > d.ws_max_pa
    assert any("OUTSIDE the allowed band" in n for n in d.notes), d.notes


def test_a_mission_that_makes_no_sense_is_refused():
    with pytest.raises(ValueError, match="cruise speed must exceed"):
        cd.air_diagram(_air(v_cruise_ms=10.0))
    with pytest.raises(ValueError, match="must both be > 0"):
        cd.air_diagram(_air(cl_max=0.0))


def test_the_recommendation_carries_its_reason():
    r = cd.air_diagram(_air(landing_distance_m=150.0)).recommend()
    assert r["binding_constraint"] == "landing distance"
    assert r["twr_required"] > 0.0
    assert r["wing_loading_min_drag_pa"] > r["wing_loading_pa"]


# ----------------------------------------------------------------- water
def test_the_foil_is_bounded_by_fly_up_and_by_cavitation():
    m = cd.WaterMission(v_takeoff_ms=6.0, v_max_ms=14.0, cl_max=0.9,
                        depth_m=0.4)
    d = cd.water_diagram(m)
    names = {c.name for c in d.constraints}
    assert names == {"fly-up", "cavitation"}
    assert d.ws_max_pa == min(c.ws_pa for c in d.constraints)


def test_cavitation_tightens_with_speed_and_relaxes_with_depth():
    shallow = cd.WaterMission(v_takeoff_ms=6.0, v_max_ms=14.0, cl_max=0.9,
                              depth_m=0.2)
    deep = cd.WaterMission(v_takeoff_ms=6.0, v_max_ms=14.0, cl_max=0.9,
                           depth_m=1.0)
    assert (cd.cl_cavitation_limited(deep, 14.0)
            > cd.cl_cavitation_limited(shallow, 14.0))
    assert (cd.cl_cavitation_limited(shallow, 20.0)
            < cd.cl_cavitation_limited(shallow, 14.0))


def test_a_speed_where_even_zero_lift_cavitates_admits_no_loading():
    m = cd.WaterMission(v_takeoff_ms=6.0, v_max_ms=60.0, cl_max=0.9,
                        depth_m=0.2)
    assert cd.cl_cavitation_limited(m, 60.0) == 0.0
    d = cd.water_diagram(m)
    assert d.ws_max_pa == 0.0
    assert any("NO positive wing loading" in n for n in d.notes), d.notes


# ---------------------------------------------------------------- the use
def test_the_loading_fixes_the_lift_coefficient_and_the_area():
    """The identity the whole module exists for: choose W/S and the trim CL
    is chosen with it, whatever the area turns out to be."""
    ws, w = 320.0, 8000.0
    S = cd.wing_loading_to_area(ws, w)
    assert S == pytest.approx(w / ws)
    cl = cd.cl_at(ws, cd.RHO_SL, 45.0)
    assert cl == pytest.approx(w / (0.5 * cd.RHO_SL * 45.0 ** 2 * S))


def test_a_tandem_pair_is_quoted_on_its_total_area():
    ar = cd.tandem_effective_aspect_ratio(b=10.0, s_total=20.0)
    assert ar == pytest.approx(5.0)
    with pytest.raises(ValueError):
        cd.tandem_effective_aspect_ratio(b=0.0, s_total=20.0)


# ---------------- closing the diagram: the matching point ----------------
#
# The diagram computed the T/W its constraints REQUIRE and never asked what
# the aircraft HAS, so the only thing it could recommend was the ws_max
# LIMIT. A limit is not a design point, and adopting one as if it were is why
# a searched wing loading had nothing to stop it below the top of its band
# (results/ws_band_study.json: no interior optimum in W/S, in any objective).

def _thrust_limited_mission(**kw):
    """A mission whose take-off curve rises INSIDE the allowed band.

    Needed because in the usual case the aerodynamic optimum sits outside the
    allowed band, the envelope falls all the way to the cap, and thrust can
    never bind — so a test built on the usual mission would exercise only the
    branch where the cap wins and would pass with the whole feature stubbed.
    """
    return cd.AirMission(v_stall_ms=48.0, cl_max=1.6, v_cruise_ms=95.0,
                         climb_rate_ms=4.0, cd0=0.028, aspect_ratio=8.0,
                         takeoff_distance_m=300.0, **kw)


def test_without_a_stated_thrust_the_diagram_reports_a_limit_and_says_so():
    d = cd.air_diagram(_thrust_limited_mission())
    assert d.ws_matched_pa is None and d.matched_binding == ""
    # it still recommends something — the limit — but the recommendation is
    # labelled as the limit it is, and the notes say what is missing
    assert d.ws_design_pa == pytest.approx(d.ws_max_pa)
    r = d.recommend()
    assert r["wing_loading_matched_pa"] is None
    assert r["wing_loading_max_pa"] == pytest.approx(d.ws_max_pa)
    assert any("twr_available" in n for n in r["notes"])


def test_more_thrust_buys_a_higher_flyable_wing_loading():
    """The matching point is a real function of the thrust, not a relabelling.

    This is the branch the whole feature exists for: below the crossover the
    engine decides the wing loading, and the answer MOVES with it.
    """
    import dataclasses as dc

    base = _thrust_limited_mission()
    got = []
    for twr in (0.20, 0.30, 0.40):
        d = cd.air_diagram(dc.replace(base, twr_available=twr))
        assert d.matched_binding == "thrust"
        assert d.ws_matched_pa < d.ws_max_pa
        # the point is ON the envelope it was solved against
        assert d.twr_at(d.ws_matched_pa) <= twr + 1e-9
        got.append(d.ws_matched_pa)
    assert got[0] < got[1] < got[2]


def test_past_the_crossover_the_cap_takes_over_and_is_reported_exactly():
    """Enough thrust and the LIMIT binds again — named as the limit, not as
    thrust, and at the cap's exact value rather than the grid point below it
    (a quantised answer reads as thrust-limited one step early)."""
    import dataclasses as dc

    base = _thrust_limited_mission()
    cap = cd.air_diagram(base).ws_max_pa
    for twr in (0.6, 1.0, 5.0):
        d = cd.air_diagram(dc.replace(base, twr_available=twr))
        assert d.matched_binding != "thrust"
        assert d.ws_matched_pa == pytest.approx(cap)     # exact, not quantised


def test_an_unflyable_mission_has_no_design_point_at_all():
    """The silent-fallback guard.

    When no loading is both allowed and flyable, falling back to the ws_max
    line would hand back a design point for an aircraft that cannot be built
    — and every downstream caller would read it as a normal answer.
    """
    d = cd.air_diagram(_thrust_limited_mission(twr_available=0.02))
    assert d.ws_matched_pa is None and d.matched_binding == "infeasible"
    assert d.ws_design_pa is None                    # NOT d.ws_max_pa
    assert d.recommend()["wing_loading_pa"] is None
    assert d.ws_max_pa is not None                   # the limit still exists
    assert any("infeasible as stated" in n for n in d.notes)


def test_stating_thrust_never_raises_the_recommendation():
    """A capability can only ever tighten the answer, never loosen it."""
    import dataclasses as dc

    for m in (_thrust_limited_mission(),
              cd.AirMission(v_stall_ms=25.0, cl_max=1.6, v_cruise_ms=55.0,
                            climb_rate_ms=5.0, takeoff_distance_m=400.0)):
        limit = cd.air_diagram(m).ws_max_pa
        for twr in (0.25, 0.5, 2.0):
            ws = cd.air_diagram(dc.replace(m, twr_available=twr)).ws_design_pa
            if ws is not None:
                assert ws <= limit + 1e-9


def test_the_water_diagram_is_untouched_by_the_air_matching_point():
    """A foiling craft states no thrust here, so its recommendation is the
    same limit it always was — bit-for-bit, not merely 'still works'."""
    m = cd.WaterMission(v_takeoff_ms=6.0, cl_max=0.9, v_max_ms=14.0)
    d = cd.water_diagram(m)
    assert d.ws_matched_pa is None and d.matched_binding == ""
    assert d.recommend()["wing_loading_pa"] == pytest.approx(d.ws_max_pa)
