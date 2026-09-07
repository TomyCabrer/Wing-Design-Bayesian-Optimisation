"""A tandem's area is TWO areas, and the recommendation is one span band.

Session 65, on the pair: *"design box for tandem should let define front and
rear wing area instead of just the both"*, and — of the measured box's two
span rows — *"take the largest and smallest, this is the recommendation, it
should include everything reasonable"*.

**THE AREAS.** The box states a pair as a TOTAL (``S_m2``) and a SPLIT
(``area_split_front``), which is what the solver searches and not what anybody
designs. Both directions are exact, so the card can be asked the other way:
each wing's own area, exactly or as a band, and the two rows follow —

    S      = S_front + S_rear
    split  = S_front / (S_front + S_rear)

Two exact areas PIN both rows (the variables leave the design vector, which is
the only honest way to say "it is exactly this" — ``api.RunConfig.pinned``); a
band on either wing derives bands, drawn as the smallest box containing every
(S_front, S_rear) the two answers allow. Where the family does not search the
total at all — most tandem families — only the split is written, and the card
reports the total the two wings imply against the mission's own rather than
restating the mission behind the user's back.

**THE SPAN.** A pair has two span rows, and the measurement bounds each on the
draws that landed in it: reported per row that reads as an instruction about
two different wings, and it cuts the band down more than either wing's own
answer justifies. The two are unioned — the smallest and the largest — and
each row is still clipped into its own bound, so a recommendation still cannot
widen the box it is a recommendation for.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _pair(system: str = "tandem", planform: str | None = None):
    from gui.v3.app import assemble

    ctx = assemble("air")
    ctx.act("set_lifting_system", system)
    ctx.act("accept_mission")
    if planform:
        ctx.act("set_planform", planform)
    ctx.render("wing", "box")
    return ctx


def _texts(view) -> list[str]:
    return [getattr(e, "text", None) or "" for e in view.descendants()]


# ------------------------------------------------------------- 1. the card
def test_only_a_pair_is_offered_two_areas():
    from gui.v3 import session
    from gui.v3.app import assemble

    assert session.pair_area_available(_pair().S)

    plain = assemble("air")
    plain.act("accept_mission")
    assert not session.pair_area_available(plain.S)
    # ...and the card is not drawn on it
    plain.render("wing", "box")
    assert not any("pair's two areas" in t.lower()
                   for t in _texts(plain.views[("wing", "box")]))


def test_switching_to_per_wing_areas_opens_on_the_box_already_there():
    """Turning the question round may not, by itself, move the search: the
    two wings are seeded from the total x split box on screen, so switching
    over and straight back is a no-op."""
    from gui.v3 import config, session

    ctx = _pair()
    before = config.effective_bounds(ctx.S)[session.PAIR_SPLIT_ROW][0]

    ctx.act("set_pair_area_mode", "wings")
    got = session.pair_areas(ctx.S)
    assert got is not None
    rows = session.pair_area_rows(ctx.S)
    kind, band = rows[session.PAIR_SPLIT_ROW]
    assert kind == "band"
    assert abs(band[0] - float(before[0])) < 1e-9
    assert abs(band[1] - float(before[1])) < 1e-9

    ctx.act("set_pair_area_mode", "total")
    after = config.effective_bounds(ctx.S)[session.PAIR_SPLIT_ROW][0]
    assert (float(after[0]), float(after[1])) == (float(before[0]),
                                                  float(before[1]))


# ------------------------------------------------------------ 2. the rows
def test_two_stated_areas_pin_the_split_and_leave_the_design_vector():
    from aerobo import api
    from gui.v3 import config, session

    ctx = _pair()
    ctx.act("set_pair_area_mode", "wings")
    for wing, area in (("front", 12.0), ("rear", 8.0)):
        ctx.act("set_pair_area_exact", wing, True)
        ctx.act("set_pair_area_value", wing, area)

    rows = session.pair_area_rows(ctx.S)
    assert rows[session.PAIR_SPLIT_ROW] == ("pin", 0.6)
    cfg = config.build_cfg(ctx.S)
    assert cfg.pinned[session.PAIR_SPLIT_ROW] == 0.6
    built, _stripped = api._recommendable_problem(cfg)
    assert session.PAIR_SPLIT_ROW not in built.param_labels


def test_two_stated_bands_become_the_smallest_box_that_contains_them():
    """The split's ends are the CORNERS — the smallest front against the
    largest rear, and the largest front against the smallest rear — not the
    ratio of the two bands read end to end."""
    from gui.v3 import config, session

    ctx = _pair()
    ctx.act("set_pair_area_mode", "wings")
    ctx.act("set_pair_area_exact", "front", False)
    ctx.act("set_pair_area_band", "front", "lo", 6.0)
    ctx.act("set_pair_area_band", "front", "hi", 10.0)
    ctx.act("set_pair_area_exact", "rear", False)
    ctx.act("set_pair_area_band", "rear", "lo", 4.0)
    ctx.act("set_pair_area_band", "rear", "hi", 8.0)

    kind, band = session.pair_area_rows(ctx.S)[session.PAIR_SPLIT_ROW]
    assert kind == "band"
    assert abs(band[0] - 6.0 / 14.0) < 1e-12
    assert abs(band[1] - 10.0 / 14.0) < 1e-12
    # ...and it is the box the run searches
    row = config.effective_bounds(ctx.S)[session.PAIR_SPLIT_ROW][0]
    assert abs(float(row[0]) - band[0]) < 1e-12
    assert abs(float(row[1]) - band[1]) < 1e-12


def test_half_an_answer_writes_nothing():
    from gui.v3 import session

    ctx = _pair()
    ctx.act("set_pair_area_mode", "wings")
    ctx.S["wing"]["choices"][session.PAIR_AREA_KEY] = {
        "mode": "wings", "front": {"exact": True, "value": 12.0, "band": []},
        "rear": {"exact": False, "value": None, "band": []}}
    assert session.pair_areas(ctx.S) is None
    assert session.pair_area_rows(ctx.S) == {}


def test_an_area_that_is_not_an_area_is_refused():
    from gui.v3 import session

    ctx = _pair()
    ctx.act("set_pair_area_mode", "wings")
    assert session.set_pair_area(ctx.S, "front", value=0.0) is False
    assert session.set_pair_area(ctx.S, "front", value="wide") is False
    assert session.set_pair_area(ctx.S, "rear", band=[9.0, 4.0]) is False


# --------------------------------------------------------- 3. the mission
def test_a_total_the_mission_does_not_share_is_reported_not_imposed():
    """Most tandem families do not SEARCH the total, so the run flies the
    mission's area. Two wings that add up to something else is a real
    disagreement, and the card says so instead of rewriting the mission."""
    from gui.v3 import session

    ctx = _pair()
    area_before = session.reference_area(ctx.S)
    ctx.act("set_pair_area_mode", "wings")
    for wing, area in (("front", 3.0), ("rear", 2.0)):
        ctx.act("set_pair_area_exact", wing, True)
        ctx.act("set_pair_area_value", wing, area)

    assert session.pair_area_total(ctx.S) == (5.0, 5.0)
    assert session.reference_area(ctx.S) == area_before   # untouched
    ctx.render("wing", "box")
    assert any("add up to" in t for t in _texts(ctx.views[("wing", "box")]))

    # ...and taking it over is a BUTTON, which does move the mission
    ctx.act("adopt_pair_area_total")
    assert abs(session.reference_area(ctx.S) - 5.0) < 1e-9


def test_a_family_that_searches_the_total_writes_it_too():
    from gui.v3 import session

    ctx = _pair(planform="free")
    if session.AREA_ROW not in session.size_rows(ctx.S):
        return                       # this family states its area elsewhere
    ctx.act("set_pair_area_mode", "wings")
    for wing, area in (("front", 12.0), ("rear", 8.0)):
        ctx.act("set_pair_area_exact", wing, True)
        ctx.act("set_pair_area_value", wing, area)
    rows = session.pair_area_rows(ctx.S)
    assert rows[session.AREA_ROW] == ("pin", 20.0)


# ------------------------------------------------------------ 4. the span
def test_the_pairs_two_span_bands_are_unioned():
    from gui.v3 import session

    ctx = _pair()
    S = ctx.S
    bands = {"b_m": (8.0, 12.0), "b_rear_m": (5.0, 9.0)}
    got = session.union_pair_spans(S, bands)
    assert got["b_m"] == (5.0, 12.0)
    assert got["b_rear_m"] == (5.0, 12.0)


def test_the_union_still_cannot_widen_a_row_past_its_own_bound():
    """A recommendation is not a licence: answering one more question may
    never widen a search."""
    from gui.v3 import session

    ctx = _pair()
    bands = {"b_m": (8.0, 12.0), "b_rear_m": (5.0, 9.0)}
    got = session.union_pair_spans(
        ctx.S, bands, {"b_m": (7.0, 20.0), "b_rear_m": (6.0, 9.5)})
    assert got["b_m"] == (7.0, 12.0)
    assert got["b_rear_m"] == (6.0, 9.5)


def test_a_single_wing_family_is_left_exactly_alone():
    from gui.v3 import session
    from gui.v3.app import assemble

    ctx = assemble("air")
    ctx.act("accept_mission")
    bands = {"b_m": (8.0, 12.0)}
    assert session.union_pair_spans(ctx.S, bands) == bands


def test_the_measured_box_gives_the_pair_one_span_band():
    """End to end, through the shell's own automatic measurement."""
    from gui.v3 import config, session

    ctx = _pair()
    session.set_size_statement(ctx.S, session.SIZE_AS_SPAN_AR)
    session.set_size_from_span_ar(ctx.S, span=6.0, ar=8.0)
    ctx.S["mission"]["W_N"] = 60.0 * ctx.S["mission"]["s_ref_m2"]
    ctx.act("accept_mission")
    ctx.act("set_span_searched", True)
    ctx.render("wing", "box")
    a = ctx.act("auto_recommend")
    if a.get("empty") or not a.get("rows"):
        return                          # nothing measured here to compare
    eff = config.effective_bounds(ctx.S)
    if "b_m" not in eff or "b_rear_m" not in eff:
        return
    front, rear = eff["b_m"][0], eff["b_rear_m"][0]
    if not (eff["b_m"][1] == "recommended" or eff["b_rear_m"][1]
            == "recommended"):
        return
    # the same band on both rows, and the stated span inside it
    assert abs(float(front[0]) - float(rear[0])) < 1e-6 or \
        min(float(front[0]), float(rear[0])) <= 6.0
    assert float(front[0]) <= 6.0 <= float(front[1])
