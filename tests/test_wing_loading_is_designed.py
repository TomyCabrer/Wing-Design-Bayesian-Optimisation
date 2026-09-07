"""The wing loading as a QUESTION the design answers, and as a LIMIT.

Three things were true before this module existed, and each is tested here
as the thing it was:

1. W/S was reachable only through the AREA row of the free-planform mode,
   whose band is a fraction of the family's own reference area — so the
   loadings a search could reach were an accident of that reference size. A
   1500 N aircraft at 40 N/m² wants 37 m^2 and the box stopped at 22.
   :data:`sizing.SIZE_MODE_WS_FREE` puts the loading itself in the vector,
   in N/m², which is the unit its own question (the constraint diagram) is
   answered in.

2. Those size bands could not be widened. ``api._apply_overrides`` moves the
   SAMPLER's box; the problem re-checks its OWN box and returns "bounds
   violation" (the penalty contract) for anything outside it, so a user who
   widened the area row to reach their mission bought a run made of
   penalties. The design-box rows now reach the problem's own bands
   (``api._size_band_kwargs``).

3. The constraint diagram was a card at stage 1 and nothing else. A sized
   search was free to buy payload L/D with a loading the aircraft could not
   land at — and did: the published free planform optimises to ~128 N/m²
   against a mission stating 65, with a stall speed inside 20 % of cruise.
   ``wing_loading_limit_pa`` makes the mission's own ceiling refuse those
   candidates, in-contract, like the aspect-ratio band.
"""

from __future__ import annotations

import numpy as np
import pytest

from aerobo import api, objective, sizing

WS_FREE = "trim wing + free span + free W/S"


def _built(name=WS_FREE, flags=None, over=None):
    return api.PROBLEM_SPECS[name].build({}, flags or {}, over)


def _q():
    p = objective.Problem(mode="trim")
    return 0.5 * p.rho * p.V ** 2


# ------------------------------------------------- 1. the loading is a row
def test_the_vector_carries_the_span_and_the_loading():
    b = _built()
    assert b.param_labels == ("taper", "twist_root_deg", "twist_tip_deg",
                              "b_m", "ws_pa")
    # the AREA is not in it: it follows the loading through the weight loop
    assert "S_m2" not in b.param_labels
    assert b.is_constrained         # the same spar margin every sized mode adds


def test_the_band_opens_on_the_loading_the_family_already_flies():
    """A default that is not the current design is a default that has chosen
    something. The band is WS_FRAC_BOUNDS around CL_target x q."""
    ws0 = objective.Problem(mode="trim").CL_target * _q()
    row = _built().bounds[-1]
    assert row[0] == pytest.approx(sizing.WS_FRAC_BOUNDS[0] * ws0)
    assert row[1] == pytest.approx(sizing.WS_FRAC_BOUNDS[1] * ws0)


def test_the_candidate_flies_the_loading_it_was_handed():
    """The whole point: W/S is read off the vector, the area is derived from
    it, and the trim CL is (W/S)/q for THAT loading."""
    b = _built()
    q = _q()
    for ws in (45.0, 70.0, 110.0):
        out = b.evaluate(np.array([0.6, 0.0, -2.0, 12.0, ws]))
        assert out["feasible"], out["reason"]
        assert out["W_total_N"] / out["S_m2"] == pytest.approx(ws, rel=1e-6)
        assert out["CL"] == pytest.approx(ws / q, rel=1e-6)
    # ...and the area MOVES with it, which is what "the area is not a design
    # variable" means here
    lo = b.evaluate(np.array([0.6, 0.0, -2.0, 12.0, 45.0]))["S_m2"]
    hi = b.evaluate(np.array([0.6, 0.0, -2.0, 12.0, 110.0]))["S_m2"]
    assert lo > hi


def test_the_band_is_the_users_in_the_units_the_mission_thinks_in():
    b = _built(flags={"ws_min_pa": 40.0, "ws_max_pa": 90.0})
    assert list(b.bounds[-1]) == [40.0, 90.0]
    # one end alone is a complete sentence
    top = _built(flags={"ws_max_pa": 90.0})
    assert top.bounds[-1][1] == 90.0
    ws0 = objective.Problem(mode="trim").CL_target * _q()
    assert top.bounds[-1][0] == pytest.approx(sizing.WS_FRAC_BOUNDS[0] * ws0)


def test_it_composes_with_the_other_modifiers():
    for name in (f"{WS_FREE} + free chord law",
                 f"{WS_FREE} + free flight state",
                 "winglet_capped + free span + free W/S",
                 "tail + free span + free W/S"):
        b = _built(name)
        assert "ws_pa" in b.param_labels and "S_m2" not in b.param_labels
        out = b.evaluate(b.bounds.mean(axis=1))
        assert np.isfinite(out["score"]), (name, out["reason"])


def test_the_three_size_modes_are_alternatives():
    assert set(api.SIZE_MODIFIERS) == {"size", "size_ws", "size_ws_free"}
    assert api.with_modifiers("trim wing", {"size", "size_ws_free"}) is None
    assert api.add_modifier(WS_FREE, "size") is None
    assert api.add_modifier("trim wing + free planform", "size_ws_free") is None


def test_the_tandem_searches_two_spans_and_one_loading():
    """The pair's TOTAL area follows the loading; the two spans stay the
    thing the family exists to trade."""
    b = _built("tandem + free span + free W/S")
    assert b.param_labels[-3:] == ("b_m", "b_rear_m", "ws_pa")


# ---------------------------------------- 2. the box shown is the box searched
def test_the_free_planform_area_band_can_be_widened():
    """The bug this fixes: the row moved the sampler's box, the problem kept
    its own, and every draw outside came back a penalty."""
    name = "trim wing + free planform"
    wide = _built(name, over={"S_m2": (3.0, 40.0)})
    assert list(wide.bounds[-1]) == [3.0, 40.0]
    out = wide.evaluate(np.array([0.6, 0.0, -2.0, 10.0, 30.0]))
    assert out["feasible"], out["reason"]        # was "bounds violation"
    assert out["S_m2"] == pytest.approx(30.0)
    # ...and the published box is unchanged when nobody typed a row
    assert list(_built(name).bounds[-1]) == [8.0, 22.0]


def test_a_heavier_mission_can_reach_its_own_wing_loading():
    """1500 N at 40 N/m² is 37.5 m² of wing, and the published area box
    stops at 22 — the reference size was choosing the answer."""
    name = "trim wing + free planform"
    b = api.PROBLEM_SPECS[name].build({"W_N": 1500.0}, {},
                                      {"S_m2": (20.0, 60.0)})
    out = b.evaluate(np.array([0.6, 0.0, -2.0, 20.0, 40.0]))
    assert out["feasible"], out["reason"]
    assert out["S_m2"] == pytest.approx(40.0)


def test_the_span_band_reaches_every_sized_mode():
    for name in ("trim wing + free planform", WS_FREE,
                 "trim wing + free span (W/S)"):
        b = _built(name, flags={"span_min_m": 9.0, "span_max_m": 15.0})
        row = list(b.bounds[api.PROBLEM_SPECS[name].param_labels.index("b_m")])
        assert row == [9.0, 15.0], name


def test_the_wing_tail_families_stop_dropping_the_loading_flags():
    """They DECLARED wing_loading_pa and never read it: a stated W/S changed
    nothing on any nonplanar wing+tail variant."""
    name = "tail [designed tail + tip device] + free span (W/S)"
    prob = _built(name, flags={"wing_loading_pa": 120.0,
                               "span_min_m": 8.0}).problem
    assert prob.wing_loading_Pa == pytest.approx(120.0)
    assert prob.span_bounds_m[0] == pytest.approx(8.0)


# ------------------------------------ 3. the mission's ceiling is a real limit
def test_the_ceiling_refuses_a_candidate_in_every_sized_mode():
    q = _q()
    free = _built("trim wing + free planform",
                  flags={"wing_loading_limit_pa": 80.0})
    heavy = free.evaluate(np.array([0.6, 0.0, -2.0, 10.0, 8.0]))
    assert not heavy["feasible"]
    assert "above the mission" in heavy["reason"]
    assert heavy["score"] == objective.PENALTY      # in-contract, not an error
    light = free.evaluate(np.array([0.6, 0.0, -2.0, 10.0, 14.0]))
    assert light["feasible"] and light["W_total_N"] / light["S_m2"] <= 80.0
    # ...and the searched-loading mode never even offers the forbidden half
    clipped = _built(flags={"wing_loading_limit_pa": 80.0})
    assert clipped.bounds[-1][1] == pytest.approx(80.0)
    assert clipped.bounds[-1][0] == pytest.approx(
        sizing.WS_FRAC_BOUNDS[0] * objective.Problem(mode="trim").CL_target * q)


def test_a_band_the_mission_forbids_is_refused_loudly():
    """Two things the user stated disagree; quietly moving one of them is how
    a limit stops being a limit."""
    with pytest.raises(ValueError, match="above the mission"):
        _built(flags={"wing_loading_limit_pa": 60.0},
               over={"ws_pa": (40.0, 120.0)}).bounds
    with pytest.raises(ValueError, match="allows no loading"):
        _built(flags={"wing_loading_limit_pa": 20.0}).bounds


def test_no_ceiling_no_change():
    """Every published run states no limit, and must be bit-for-bit itself.

    The old form compared ``_built("trim wing + free planform")`` against a
    second IDENTICAL build of the same spec, so it could only fail on
    non-determinism: a regression that read an ABSENT ceiling as some default
    cap inside the evaluate path sent both sides to PENALTY together and
    stayed green. What is pinned now is the outcome an absent ceiling has to
    have — this candidate flies a loading a stated ceiling DOES refuse (the
    80 Pa of the test above), and with none stated it is flown, scored, and
    scored the same as under a ceiling far above it.
    """
    x = np.array([0.6, 0.0, -2.0, 10.0, 8.0])
    plain = _built("trim wing + free planform")
    assert plain.problem.wing_loading_max_Pa is None
    out = plain.evaluate(x)
    assert out["feasible"], out["reason"]
    assert out["score"] != objective.PENALTY and np.isfinite(out["score"])
    # the precondition that makes the rest of this mean anything: it IS a
    # loading a mission ceiling would refuse, so a default cap cannot hide
    assert out["W_total_N"] / out["S_m2"] > 80.0
    # ...and a ceiling far above the design changes nothing either — the same
    # statement from the other side, and the one an in-evaluate default breaks
    high = _built("trim wing + free planform",
                  flags={"wing_loading_limit_pa": 1.0e4})
    assert high.evaluate(x)["score"] == pytest.approx(out["score"],
                                                      rel=0, abs=0.0)


def test_every_sized_result_reports_the_loading_it_flew():
    """An output in the free mode, an input in the other two, the same
    quantity either way — so a result can always be read as a wing loading."""
    for name in ("trim wing + free planform", WS_FREE,
                 "trim wing + free span (W/S)"):
        b = _built(name)
        out = b.evaluate(b.bounds.mean(axis=1))
        assert out["wing_loading_Pa"] == pytest.approx(
            out["W_total_N"] / out["S_m2"], rel=1e-9), name
