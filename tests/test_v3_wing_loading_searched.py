"""V3: the wing loading as a third planform answer, and as the mission's cap.

The shell's half of ``test_wing_loading_is_designed``: the menu offers the
mode, choosing it writes the band into the design box (where every other band
is asked), the band the card quotes is the band the run gets, and the
mission's own constraint diagram reaches the solver instead of only being
drawn at stage 1.
"""

from __future__ import annotations

import pytest

from aerobo import api
from gui import nice_app as v1
from gui.v3 import config, session


def _air():
    return session.make_session("air")


# ------------------------------------------------------------- the menu
def test_the_menu_offers_the_third_answer():
    S = _air()
    offered = v1.planform_options(S["wing"]["choices"])
    assert set(offered) >= {"fixed", "wing_loading", "wing_loading_free",
                            "free"}
    # ...and it names what it searches, in the unit it searches it in
    assert "N/m²" in v1.PLANFORM_CHOICE_LABELS["wing_loading_free"]


def test_choosing_it_selects_the_searched_loading_twin():
    S = _air()
    session.set_planform(S, "wing_loading_free")
    assert "size_ws_free" in api.modifiers_of(S["wing"]["problem"])
    assert session.loading_is_searched(S)
    # the span is searched here too — the mode puts BOTH in the vector
    assert session.span_is_searched(S)
    labels = api.PROBLEM_SPECS[S["wing"]["problem"]].param_labels
    assert "ws_pa" in labels and "S_m2" not in labels


def test_leaving_the_mode_takes_the_row_back_out():
    S = _air()
    session.set_planform(S, "wing_loading_free")
    assert session.WS_ROW in S["wing"]["bounds"]
    session.set_planform(S, "fixed")
    assert session.WS_ROW not in S["wing"]["bounds"]
    assert not session.loading_is_searched(S)


# -------------------------------------------------------- the band asked
def test_the_band_is_written_into_the_design_box():
    """Written rather than implicit, for the reason the span band is: an
    implicit band moves whenever the mission's W/S moves."""
    S = _air()
    ws = session.wing_loading(S)
    session.set_planform(S, "wing_loading_free")
    row = S["wing"]["bounds"][session.WS_ROW]
    # it stops where the MISSION does, and reaches down from there — the
    # centre is the mission's design point, not the area the user typed
    cap = session.mission_ws_ceiling(S)
    assert cap and row[1] == pytest.approx(cap)
    assert row[0] == pytest.approx(min(0.5 * cap, ws))


def test_the_band_is_stamped_as_the_shell_s_and_survives_a_problem_change():
    """PROVENANCE, which is what makes a band the shell's to maintain.

    Written straight into ``W["bounds"]`` with no ``bounds_source`` entry the
    row was "user" from birth, so ``apply_choices`` wiped it on any problem
    change and its compensating ``write_size_bands`` could not bring it back
    — measured on the default air mission, selecting a tip device dropped the
    searched row from the mission's 37.601-75.203 Pa to the family's
    published 32.640-75.203 Pa while the box painted it "default"."""
    S = _air()
    session.set_planform(S, "wing_loading_free")
    asked = tuple(S["wing"]["bounds"][session.WS_ROW])
    assert session.bounds_source(S)[session.WS_ROW] == "shell"
    assert config.effective_bounds(S)[session.WS_ROW][1] == "mission"

    S["wing"]["choices"]["winglets"] = "capped"      # a different family
    session.apply_choices(S)
    assert "winglet" in S["wing"]["problem"]
    assert session.loading_is_searched(S)
    row, source = config.effective_bounds(S)[session.WS_ROW]
    assert tuple(row) == pytest.approx(asked)
    assert source == "mission"
    # ...and leaving the mode still takes the stamp with the row
    session.set_planform(S, "fixed")
    assert session.WS_ROW not in session.bounds_source(S)


def test_the_band_follows_the_mission_that_derived_it():
    """The other half of the same provenance: ``ws_band_default`` centres the
    band on the mission's own ceiling, so tightening a requirement moves it.
    A "user" row is never refreshed, so the band stayed above the new ceiling
    and the run died in the worker on a row nobody had typed — "the
    wing-loading band … reaches above the mission's own ceiling of …"."""
    S = _air()
    session.set_planform(S, "wing_loading_free")
    assert session.set_ws_input(S, "v_stall_ms", 7.0)     # a tighter stall
    cap = session.mission_ws_ceiling(S)
    session.sync_wing_from_mission(S)                     # the mission moved
    assert session.ws_band(S)[1] == pytest.approx(cap)
    assert session.ws_band(S) == pytest.approx(session.ws_band_default(S))

    # ...but a band the USER typed is still theirs, and is left alone
    session.bounds_source(S).pop(session.WS_ROW, None)
    S["wing"]["bounds"][session.WS_ROW] = [20.0, 30.0]
    assert session.set_ws_input(S, "v_stall_ms", 6.0)
    session.sync_wing_from_mission(S)
    assert session.ws_band(S) == pytest.approx((20.0, 30.0))


def test_the_band_the_card_quotes_is_the_band_the_run_gets():
    S = _air()
    session.set_planform(S, "wing_loading_free")
    S["wing"]["bounds"][session.WS_ROW] = [45.0, 70.0]
    quoted = session.ws_band(S)
    built = api.PROBLEM_SPECS[S["wing"]["problem"]].build(
        {}, config.flags(S), config.bounds_overrides(S))
    i = built.param_labels.index("ws_pa")
    assert quoted == pytest.approx((45.0, 70.0))
    assert list(built.bounds[i]) == [45.0, 70.0]
    # ...and the problem's OWN box agrees, so no draw in it is a penalty
    assert built.problem.ws_bounds_pa == (45.0, 70.0)


def test_a_run_in_the_mode_flies_the_loading_it_searched():
    S = _air()
    session.set_planform(S, "wing_loading_free")
    built = api.PROBLEM_SPECS[S["wing"]["problem"]].build(
        {}, config.flags(S), config.bounds_overrides(S))
    out = built.evaluate(built.bounds.mean(axis=1))
    assert out["feasible"], out["reason"]
    i = built.param_labels.index("ws_pa")
    assert out["W_total_N"] / out["S_m2"] == pytest.approx(
        built.bounds.mean(axis=1)[i], rel=1e-6)


# ------------------------------------------------ the mission's own ceiling
def test_the_mission_ceiling_reaches_the_solver():
    """It was a card at stage 1 and nothing else."""
    S = _air()
    cap = session.mission_ws_ceiling(S)
    assert cap and cap > 0.0
    for mode in ("free", "wing_loading", "wing_loading_free"):
        session.set_planform(S, mode)
        flags = config.flags(S)
        assert flags.get("wing_loading_limit_pa") == pytest.approx(cap), mode


def test_a_mission_that_bounds_nothing_sends_nothing():
    """The diagram is drawn from numbers the user gave; an unanswered stall
    speed leaves the search exactly as free as it was."""
    S = _air()
    session.set_planform(S, "free")
    for key in ("v_stall_ms", "cl_max", "landing_distance_m"):
        S["mission"].setdefault("ws_inputs", {})
    inputs = session.ws_inputs(S)
    for key in inputs:
        session.set_ws_input(S, key, None) if hasattr(
            session, "set_ws_input") else None
    S["mission"]["ws_inputs"] = {k: None for k in inputs}
    assert session.mission_ws_ceiling(S) is None
    assert "wing_loading_limit_pa" not in config.flags(S)


def test_the_cap_refuses_the_loading_the_free_planform_used_to_buy():
    """The measured failure: free, the planform optimises to about twice the
    mission's stated W/S — a stall speed inside 20 % of cruise."""
    S = _air()
    session.set_planform(S, "free")
    built = api.PROBLEM_SPECS[S["wing"]["problem"]].build(
        {}, config.flags(S), config.bounds_overrides(S))
    cap = session.mission_ws_ceiling(S)
    i = built.param_labels.index("S_m2")
    j = built.param_labels.index("b_m")
    x = built.bounds.mean(axis=1)
    # the smallest wing in the box, on a span that keeps the aspect ratio
    # inside the band — so the only thing that can refuse it is the loading
    x[i] = built.bounds[i][0]
    x[j] = 10.0
    out = built.evaluate(x)
    if out["feasible"]:
        assert out["W_total_N"] / out["S_m2"] <= cap + 1e-9
    else:
        assert "above the mission" in out["reason"]


# --------------------------------------- the band's CENTRE is the mission's
def test_the_band_does_not_depend_on_the_area_the_user_typed():
    """The headline. A searched W/S has no interior optimum, so the answer IS
    the top of the band — and a band centred on the seed made that answer a
    function of the area somebody happened to type. Centred on the mission's
    own ceiling it is a function of the mission."""
    tops, seeds = set(), []
    for scale in (0.4, 1.0, 2.0):
        S = _air()
        session.set_wing_loading(S, scale * session.wing_loading(S))
        seeds.append(session.wing_loading(S))
        lo, hi = session.ws_band_default(S)
        tops.add(round(hi, 9))
    assert len(set(round(s, 6) for s in seeds)) == 3   # the seeds DID differ
    assert len(tops) == 1                              # ...the answer did not


def test_the_band_is_widened_to_hold_todays_wing_but_never_past_the_ceiling():
    """The mode exists to re-open the question, so it has to be able to
    return to the current answer — unless the current answer is one the
    mission forbids, where agreeing with it would be the shell's mistake."""
    S = _air()
    cap = session.mission_ws_ceiling(S)
    # a wing BIGGER than the design point: the band reaches down to it
    session.set_wing_loading(S, 0.3 * cap)
    lo, hi = session.ws_band_default(S)
    assert lo == pytest.approx(0.3 * cap) and hi == pytest.approx(cap)
    # ...and one SMALLER than the mission allows is left outside, deliberately
    session.set_wing_loading(S, 1.5 * cap)
    lo, hi = session.ws_band_default(S)
    assert hi == pytest.approx(cap) and lo < hi
    assert session.wing_loading(S) > hi


def test_an_unaffordable_thrust_does_not_delete_the_stall_limit():
    """Stating a capability must not WIDEN the search. The recommendation is
    None for an infeasible mission — there is nothing to recommend — but the
    ws_max line is a requirement, and an engine that cannot meet it does not
    repeal it. Measured: T/W 0.30 against the 0.44 the climb demands used to
    take the band from 32.6-75.2 Pa to 32.6-130.6 Pa."""
    S = _air()
    before = session.ws_band_default(S)
    assert session.set_ws_input(S, "twr_available", 0.30)
    d = session.ws_diagram(S)
    assert d.matched_binding == "infeasible"
    assert d.ws_design_pa is None                  # no design point...
    assert session.mission_ws_ceiling(S) == pytest.approx(d.ws_max_pa)
    assert session.ws_band_default(S) == pytest.approx(before)
    # ...and the shell still refuses to adopt one
    assert session.adopt_ws_recommendation(S) is None


def test_a_mission_with_no_ceiling_still_opens_around_its_own_loading():
    """No diagram, no design point — the band falls back to the fractional
    band around the loading the aircraft already flies, which is the only
    centre there is."""
    from aerobo.sizing import WS_FRAC_BOUNDS

    S = _air()
    for key in ("v_stall_ms", "cl_max", "landing_distance_m"):
        session.set_ws_input(S, key, None)
    assert session.mission_ws_ceiling(S) is None
    ws = session.wing_loading(S)
    assert session.ws_band_default(S) == pytest.approx(
        (WS_FRAC_BOUNDS[0] * ws, WS_FRAC_BOUNDS[1] * ws))


# ------------------------------------- the card says what the row will do
def test_a_searched_loading_says_it_will_ride_to_its_ceiling():
    """A warn, not a refusal: the row is legal and the physics is sound. What
    the card has to add is the consequence — the loading is not traded, it is
    driven to its ceiling, so the number deciding the answer is the mission's
    and not the optimiser's."""
    from gui.v3.stages.wing import ws_ratchet_note

    S = _air()
    assert ws_ratchet_note(S) is None          # nothing to say when it is not
    session.set_planform(S, "wing_loading_free")
    warn, follow_up = ws_ratchet_note(S)
    assert "no interior optimum" in warn
    cap = session.mission_ws_ceiling(S)
    assert f"{cap:.4g}" in warn                # it names the number, not "high"
    assert "limit" in warn                     # ...and calls a limit a limit
    # the diagram is open (no thrust stated), so it says how to close it
    assert follow_up and "thrust-to-weight" in follow_up


def test_the_warning_does_not_recommend_the_composite():
    """Its own data refutes that advice: the composite parks HARDER than L/D
    (16/16 against 12/16). A card that sent the user there would be selling
    the arm the measurement rejected."""
    from gui.v3.stages.wing import ws_ratchet_note

    S = _air()
    session.set_planform(S, "wing_loading_free")
    for objective in ("lod", "composite"):
        session.wing_score_state(S)["objective"] = objective
        warn, _ = ws_ratchet_note(S)
        # the same warning either way — the ratchet is a property of the row
        assert "composite parks harder" in warn
        assert "pick the composite" not in warn.lower()


def test_a_closed_diagram_changes_what_the_warning_promises():
    """With the thrust stated the ceiling is a MATCHING POINT, and the card
    stops asking for a number it already has."""
    from gui.v3.stages.wing import ws_ratchet_note

    S = _air()
    session.set_planform(S, "wing_loading_free")
    assert session.set_ws_input(S, "twr_available", 0.8)
    assert session.ws_diagram(S).ws_matched_pa is not None
    warn, follow_up = ws_ratchet_note(S)
    assert "matching point" in warn and follow_up is None


# ------------------------------- the track card's own numbers keep the focus
def test_the_car_limit_numbers_can_actually_be_typed_into():
    """The car's two LIMITS live in V3's VALUE_KEYS, so writing one must not
    rebuild the card its own field lives in — a rebuilt input loses the focus
    and swallows the rest of the number (typing "95.5" stored 9, then
    nothing), which is the bug the stagger fields had.

    Two keys, not four: the area and span BANDS were the other two, and they
    are rows of the design box now rather than fields on any card, so there
    is nothing here to type them into. What remains is the drag ceiling and
    the downforce floor.

    THE VIEW MOVED, and the guarantee did not. Both fields used to be drawn
    under the DESIGN BOX and are drawn under the MAXIMISE SELECT now, beside
    the objective whose other half they are — so the card that must survive
    a keystroke is the TYPE card, and the two keys left ``BOX_VALUE_KEYS``
    with them. Rendering the box here instead would assert nothing: the
    fields are not in it any more, so every id would trivially match.

    Asserted on the WIDGETS and on the stored values, not on the two tuples:
    a subset check between them only restates the code it guards.
    """
    from gui.v3.app import assemble

    ctx = assemble()
    ctx.act("accept_mission")
    ctx.act("set_choice", "medium", "track")
    ctx.act("set_choice", "car_objective", "efficiency")
    ctx.render("wing", "type")

    view = ctx.views[("wing", "type")]
    fields = [id(e) for e in view.descendants()
              if type(e).__name__ == "Number"]
    assert fields, "no number field beside the Maximise select"

    typed = {"car_drag_budget_n": (9.0, 95.5),
             "car_downforce_min_n": (4.0, 425.0)}
    for key, (first, whole) in typed.items():
        ctx.act("set_choice", key, first)      # ...the way a person types it
        ctx.act("set_choice", key, whole)
    assert [id(e) for e in view.descendants()
            if type(e).__name__ == "Number"] == fields
    ch = ctx.S["wing"]["choices"]
    for key, (_, whole) in typed.items():
        assert ch[key] == whole, key


def test_the_car_size_is_typed_into_the_design_box_and_nowhere_else():
    """The other half: the four card fields are GONE, and the question they
    asked is answered by the box rows instead.

    Both spellings are checked — the choice keys must not linger in the
    shell's own value list, and the labels must not be drawn on the card —
    because either one coming back re-opens the second channel that made the
    box on screen and the box searched disagree.
    """
    from gui.v3.app import assemble
    from gui.v3.stages import wing as wing_stage
    from gui.v3 import config

    for key in ("car_span_min_m", "car_span_max_m",
                "car_area_min_m2", "car_area_max_m2"):
        assert key not in wing_stage.VALUE_KEYS, key

    ctx = assemble()
    ctx.act("accept_mission")
    ctx.act("set_choice", "medium", "track")
    ctx.render("wing", "type")
    texts = " ".join(t for t in (getattr(e, "text", "") or ""
                                 for e in ctx.views[("wing", "type")]
                                 .descendants()) if t)
    for gone in ("Span, no less than", "Span, no more than",
                 "Area, no less than", "Area, no more than",
                 "design the reference area too"):
        assert gone not in texts, gone

    # ...and the question IS answered, as two ordinary rows of the box
    rows = config.effective_bounds(ctx.S)
    assert "b_m" in rows and "S_m2" in rows, sorted(rows)
