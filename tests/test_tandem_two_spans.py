"""A pair has two spans, and it is asked for two — both ways round.

``tests/test_tandem_span.py`` made the rear wing's span a real span in the
PHYSICS: two lifting lines, two structures, two tip devices. This file is the
other half — the two states the shell asks it in, and the size mode that was
missing underneath the second one:

* PLANFORM FIXED — TWO NUMBERS, both typed, both on the same card. The front
  wing's span was on stage 3's size card and the rear wing's three cards away
  beside the stagger, which is the same question asked in two places; V3 now
  asks both together (``session.span_rows`` / ``chosen_span`` / ``set_span_m``
  per row, ``nice_app._tandem_controls(with_span=False)``). V1 and V2 have no
  such card and keep the field where it was.
* PLANFORM SEARCHED — TWO BANDS, one design-box row per wing. That state did
  not exist for a pair at all: no tandem family carried the ``size_ws``
  modifier, so "area from the mission's W/S, span optimised" was an entry the
  planform menu could not offer. Now the pair's TOTAL area follows the
  loading through the weight loop — split by the split the candidate flies,
  each wing weighed on its own span (``sizing.resolve_spans`` /
  ``area_for_wing_loading(wing_spans=...)``) — and the two spans are what is
  left to search.

The rule both states obey: a span is asked ONCE. Typed, it travels as a flag;
searched, it is a row and the typed value does not travel at all.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aerobo import api                      # noqa: E402
from aerobo import sizing                   # noqa: E402
from aerobo import tandem as td             # noqa: E402
from aerobo import tandemvlm as tvm         # noqa: E402
from aerobo.objective import PENALTY        # noqa: E402

WS = "tandem + free span (W/S)"
WS_VLM = "tandem (nonplanar) + winglets + free span (W/S)"


def _mid(box):
    return 0.5 * (box[:, 0] + box[:, 1])


def _v3(system: str = "tandem"):
    """A V3 shell on stage 3, with the pair selected."""
    from gui.v3.app import assemble

    ctx = assemble("air")
    ctx.act("accept_mission")
    ctx.act("set_choice", "system", system)
    return ctx


def _texts(ctx, stage, view):
    ctx.render(stage, view)
    return [getattr(e, "text", "") or ""
            for e in ctx.views[(stage, view)].descendants()]


# ------------------------------------------------- 1. the size mode exists

def test_the_pair_can_search_its_spans_at_the_missions_wing_loading():
    """The mode the planform menu could not offer. Two span rows and NO area
    row: the area is not a design variable in this mode at all."""
    for name in (WS, WS_VLM):
        spec = api.PROBLEM_SPECS[name]
        assert spec.param_labels[-2:] == ("b_m", "b_rear_m")
        assert "S_m2" not in spec.param_labels
        # ...and the "choose a size" flags are gone, both of them: a family
        # whose spans are searched may not also be handed typed ones
        assert not set(api.PLANFORM_KEYS) & set(spec.flags)
        assert not set(api.TANDEM_SPAN_KEYS) & set(spec.flags)
        assert set(api.WING_LOADING_KEYS) <= set(spec.flags)


def test_both_spans_are_flown_as_the_two_spans_they_are():
    built = api.PROBLEM_SPECS[WS].build({}, {}, None)
    x = _mid(built.bounds)
    i, j = built.param_labels.index("b_m"), built.param_labels.index("b_rear_m")
    x[i], x[j] = 14.0, 8.0
    out = built.evaluate(x)
    assert out["feasible"], out["reason"]
    assert out["b_front"] == pytest.approx(14.0)
    assert out["b_rear"] == pytest.approx(8.0)
    # the AREA is not in the vector: it came out of the wing loading
    assert out["S_m2"] > 0.0


def test_the_area_is_the_one_the_loading_implies_and_the_trim_cl_follows():
    """W/S fixed means the trim lift coefficient is fixed at (W/S)/q whatever
    span comes out — the whole reason this pairing is the honest one."""
    loading = 70.0
    built = api.PROBLEM_SPECS[WS].build({}, {"wing_loading_pa": loading}, None)
    q = 0.5 * built.problem.rho * built.problem.V ** 2
    x = _mid(built.bounds)
    i, j = built.param_labels.index("b_m"), built.param_labels.index("b_rear_m")
    seen = []
    for b in (10.0, 18.0):
        x[i], x[j] = b, 0.8 * b
        out = built.evaluate(x)
        assert out["feasible"], out["reason"]
        seen.append((out["S_m2"], out["W_total_N"], out["CL_total"]))
        assert out["W_total_N"] / out["S_m2"] == pytest.approx(loading)
        assert out["CL_total"] == pytest.approx(loading / q, rel=1e-6)
    # a bigger span is a heavier wing, so the closed loop lands on more area
    assert seen[1][0] > seen[0][0]


def test_the_rear_wings_span_pays_for_itself_in_the_weight_loop():
    """Each wing is weighed on ITS OWN span inside the area fixed point. With
    both wings weighed on the front one's, a shorter rear wing was free."""
    kw = dict(W_fixed_N=1000.0, wing_loading_Pa=70.0, taper=0.6, tc=0.12,
              q_Pa=130.0, area_fracs=(0.5, 0.5))
    same = sizing.area_for_wing_loading(b=12.0, wing_spans=(12.0, 12.0), **kw)
    short = sizing.area_for_wing_loading(b=12.0, wing_spans=(12.0, 6.0), **kw)
    assert short < same                    # the shorter structure weighs less
    # ...and omitting the spans is the equal-span pair, bit-for-bit
    assert sizing.area_for_wing_loading(b=12.0, **kw) == pytest.approx(same)
    with pytest.raises(ValueError):
        sizing.area_for_wing_loading(b=12.0, wing_spans=(12.0,), **kw)


def test_resolve_spans_is_resolve_sizes_general_case():
    """One entry point, any number of surfaces: the single-span answer is the
    two-span one read at n_spans=1, so no family had to change to get it."""
    prob = api.PROBLEM_SPECS["trim wing + free span (W/S)"].build({}, {}, None)
    x = _mid(prob.bounds)
    kw = dict(wing_loading_Pa=65.0, W_fixed_N=900.0, taper=0.6, tc=0.12,
              q_Pa=130.0)
    spans, area = sizing.resolve_spans(x, sizing.SIZE_MODE_WS, **kw)
    b, area2 = sizing.resolve_size(x, sizing.SIZE_MODE_WS, **kw)
    assert len(spans) == 1 and spans[0] == pytest.approx(b)
    assert area == pytest.approx(area2)
    assert sizing.resolve_spans(x, False) is None


def test_the_published_pair_did_not_move():
    """Nothing sent, nothing changed: the free-planform pair and the plain one
    fly exactly what they flew."""
    plain = api.PROBLEM_SPECS["tandem"].build({}, {}, None)
    assert plain.problem.size_free is False
    assert plain.problem.wing_loading_Pa is None
    assert plain.problem.span_bounds_m is None
    assert plain.problem.b_rear is None
    free = api.PROBLEM_SPECS["tandem + free planform"].build({}, {}, None)
    assert free.param_labels[-3:] == ("b_m", "b_rear_m", "S_m2")
    # the two span rows open on the SAME band (sizing.size_bounds): the wings
    # are alternatives for the same job
    i, j = free.param_labels.index("b_m"), free.param_labels.index("b_rear_m")
    assert tuple(free.bounds[i]) == tuple(free.bounds[j])
    x = _mid(free.bounds)
    x[i], x[j] = 10.0, 10.0
    out = free.evaluate(x)
    assert out["feasible"], out["reason"]
    assert out["b_front"] == pytest.approx(out["b_rear"])


def test_a_stated_band_bounds_both_wings():
    """``span_min_m`` / ``span_max_m`` are the mode's own value flags, and a
    pair has two rows to apply them to — the same band, because the two wings
    are alternatives for the same job."""
    built = api.PROBLEM_SPECS[WS].build(
        {}, {"span_min_m": 8.0, "span_max_m": 15.0}, None)
    i, j = built.param_labels.index("b_m"), built.param_labels.index("b_rear_m")
    assert tuple(built.bounds[i]) == (8.0, 15.0)
    assert tuple(built.bounds[j]) == (8.0, 15.0)


def test_every_freedom_still_composes_with_it():
    """The mode is a MODIFIER, so it stacks with the chord law, the flight
    state, the tip devices and the designed section — and the vector it
    produces is the one the box describes."""
    names = [n for n in api.problem_names()
             if "free span (W/S)" in n and "tandem" in n]
    # every tandem base that carries the mode x 4 modifier combinations of
    # chord law and flight state. DERIVED, not pinned: the nonplanar family
    # is a product over tip devices, designed section and the pair's own
    # cant now, so a literal here would have to be edited every time an axis
    # lands — and the thing worth asserting was never the number, it is that
    # the mode composes with all four modifier combinations on every base
    # that has it.
    bases = {n for n in api.problem_names()
             if "tandem" in n and not api.modifiers_of(n)}
    with_mode = {b for b in bases
                 if api.add_modifier(b, "size_ws") is not None}
    assert len(names) == 4 * len(with_mode), (len(names), sorted(with_mode))
    assert with_mode, "no tandem base carries the wing-loading size mode"
    for name in names:
        spec = api.PROBLEM_SPECS[name]
        assert spec.param_labels.count("b_m") == 1
        assert spec.param_labels.count("b_rear_m") == 1
        if "CST" in name:
            continue                        # live XFOIL: built, not evaluated
        built = api.PROBLEM_SPECS[name].build({}, {}, None)
        assert built.bounds.shape[0] == len(built.param_labels)
        # "composes" means the composed problem FLIES SOMEWHERE IN ITS OWN BOX,
        # not that its callable returned a float: the refusal sentinel is
        # PENALTY = -100.0, a perfectly finite number, so the old
        # `np.isfinite(...)` was green for a composition that refused every
        # candidate — a chord law double-counted under the W/S span mode would
        # leave all 20 names returning (PENALTY, G_FAIL) with nothing here able
        # to see it.
        #
        # It is deliberately NOT asserted at the midpoint. Measured over these
        # 12 evaluable names, the four `tandem (nonplanar) + winglets` ones
        # legitimately refuse theirs — the two tip devices are not separated
        # there — while 5-12 of 64 sampled points fly. A refused midpoint is a
        # real answer about a real geometry; a box with NO feasible point is
        # the defect.
        lo = np.asarray(built.bounds)[:, 0]
        hi = np.asarray(built.bounds)[:, 1]
        rng = np.random.default_rng(0)          # fixed: the same points always
        pts = np.vstack([0.5 * (lo + hi),
                         lo + rng.random((64, lo.size)) * (hi - lo)])
        flew = [float(built.callable(p)[0]) for p in pts]
        best = max(flew)
        assert best > PENALTY, (
            f"{name}: refused EVERY one of {len(pts)} points in its own box — "
            f"the composition does not fly. Midpoint reason: "
            f"{built.evaluate(pts[0]).get('reason')}")
        assert np.isfinite(best) and best > 0.0   # a system L/D, not a sentinel
        # ...and a refusal is explained rather than handed back as a bare number
        if flew[0] <= PENALTY:
            assert str(built.evaluate(pts[0]).get("reason") or "").strip(), \
                f"{name}: refused its midpoint without saying why"


# ------------------------------------------ 2. the shell asks for two spans

def test_the_size_card_asks_for_a_span_per_wing():
    from gui.v3 import config, session as ses

    ctx = _v3()
    S = ctx.S
    assert ses.span_rows(S) == ("b_m", "b_rear_m")
    labels = [t for t in _texts(ctx, "wing", "type") if t.startswith("span b")]
    assert labels == ["span b — front wing", "span b — rear wing"]

    # ...and each one is the span its own wing flies
    assert ses.set_span_m(S, 12.0, "b_m")
    assert ses.set_span_m(S, 7.5, "b_rear_m")
    flags = config.flags(S)
    assert flags["b_m"] == pytest.approx(12.0)
    assert flags["b_rear_m"] == pytest.approx(7.5)
    built = api.PROBLEM_SPECS[S["wing"]["problem"]].build(
        config.mission_kwargs(S), flags, None)
    assert built.problem.b == pytest.approx(12.0)
    assert built.problem.b_rear == pytest.approx(7.5)
    out = built.evaluate(_mid(built.bounds))
    assert out["feasible"], out["reason"]
    assert (out["b_front"], out["b_rear"]) == pytest.approx((12.0, 7.5))


def test_a_cleared_rear_span_is_as_wide_as_the_front_one():
    """Blank is not zero and not an error: it is the pair every published run
    flew, and the card says which number it will fly."""
    from gui.v3 import config, session as ses

    ctx = _v3()
    S = ctx.S
    assert ses.set_span_m(S, 12.0, "b_m")
    assert ses.set_span_m(S, None, "b_rear_m")
    assert ses.chosen_span(S, "b_rear_m") is None
    assert ses.nominal_span(S, "b_rear_m") == pytest.approx(12.0)
    assert "b_rear_m" not in config.flags(S)
    assert not ses.set_span_m(S, -1.0, "b_rear_m")      # refused, not stored
    assert ses.chosen_span(S, "b_rear_m") is None


def test_the_rear_span_is_asked_once_and_the_size_card_is_where():
    """It used to be asked beside the stagger as well. V1/V2 keep it there —
    they have no size card — and V3 takes it out rather than showing the same
    question twice."""
    import gui.nice_app as v1

    ctx = _v3()
    types = _texts(ctx, "wing", "type")
    assert types.count("span b — rear wing") == 1
    assert "Rear wing span" not in types
    assert "Rear wing, aft by" in types             # the STAGGER stays
    # ...and the shared control still offers it to the shells that ask there
    assert "with_span" in v1._tandem_controls.__code__.co_varnames


def test_a_single_wing_asks_one_span_and_says_nothing_about_wings():
    """The generalisation may not leak: a family with one lifting surface
    keeps the card it had."""
    from gui.v3 import session as ses

    ctx = _v3(system="single")
    assert ses.span_rows(ctx.S) == ("b_m",)
    labels = [t for t in _texts(ctx, "wing", "type") if t.startswith("span b")]
    assert labels == ["span b"]
    assert ses.chosen_span(ctx.S) is None
    assert ses.wing_area_share(ctx.S) == 1.0


# --------------------------------------- 3. searched: one BAND per wing

def test_choosing_the_wing_loading_mode_opens_a_band_per_wing():
    from gui.v3 import config, session as ses

    ctx = _v3()
    S = ctx.S
    assert ses.set_span_m(S, 12.0, "b_m")
    assert ses.set_span_m(S, 7.5, "b_rear_m")
    assert not ses.span_is_searched(S)
    ses.set_planform(S, "wing_loading")
    assert ses.span_is_searched(S)
    assert "free span (W/S)" in S["wing"]["problem"]

    lo, hi = ses.span_band_default(S)
    # ...and what the box HOLDS is that band CLIPPED, which is the band that
    # will actually be searched. `clip_size_box` removes the corners the
    # aspect-ratio gate would refuse, and on a pair it does it with the
    # gate's own arithmetic (each span row against HALF the area row, since
    # the two wings split it). Measured on this session: the default
    # (7.2, 48.0) is stored as (7.2, 20.0).
    #
    # Asserting the UNCLIPPED band here was asserting a box nobody searches —
    # the box shown has to be the box searched, and once the clip landed
    # those were two different numbers.
    box = ses.clip_size_box(S, {r: [lo, hi] for r in ses.span_rows(S)})
    assert S["wing"]["bounds"]["b_m"] == list(box["b_m"])
    assert S["wing"]["bounds"]["b_rear_m"] == list(box["b_rear_m"])
    assert box["b_m"] == box["b_rear_m"]    # ONE band, whichever row asks
    assert ses.span_box(S, "b_m") == box["b_m"]
    assert ses.span_box(S, "b_rear_m") == box["b_rear_m"]
    # ...and switching the search on has still not excluded the pair that was
    # on screen. Checked on the CLIPPED band, which is where the rule can now
    # actually be broken: the unclipped one is 4x the nominal span and holds
    # any typed pair by construction.
    b_lo, b_hi = box["b_m"]
    assert b_lo <= 7.5 and b_hi >= 12.0

    # the typed spans do NOT travel: two answers to one question
    flags = config.flags(S)
    assert "b_m" not in flags and "b_rear_m" not in flags
    assert flags["wing_loading_pa"] > 0.0

    # ...and the box the view shows is the box the run searches
    built = api.PROBLEM_SPECS[S["wing"]["problem"]].build(
        config.mission_kwargs(S), flags, config.effective_overrides(S)
        if hasattr(config, "effective_overrides") else
        {k: list(v[0]) for k, v in config.effective_bounds(S).items()})
    for row in ("b_m", "b_rear_m"):
        # ...the CLIPPED one, the same band the view above is showing. This
        # is the assertion that makes the clip safe: the shell and the engine
        # are reading one number, so there is no state in which the card
        # offers 48 m and the run searches 20.
        assert tuple(built.bounds[built.param_labels.index(row)]) \
            == tuple(box[row])
        assert tuple(box[row]) == tuple(S["wing"]["bounds"][row])


def test_the_design_box_shows_two_span_rows_and_locks_both():
    """A row per wing, leading the table, and neither can be released from
    inside it: giving a span up is the planform menu's question."""
    from gui.v3 import config, session as ses

    ctx = _v3()
    S = ctx.S
    ses.set_planform(S, "wing_loading")
    assert {"b_m", "b_rear_m"} <= set(config.effective_bounds(S))
    box = _texts(ctx, "wing", "box")
    # SPAN FIRST, and both of them: reading them apart — front span at the
    # top, rear span among the twists — hid the one comparison a pair is for
    rows = [t for t in box if t in config.effective_bounds(S)]
    assert rows[:2] == ["b_m", "b_rear_m"]
    note = [t for t in box if "are the SPANS" in t]
    assert note and "one per wing" in note[0]


def test_leaving_the_mode_takes_both_rows_back_out():
    from gui.v3 import session as ses

    ctx = _v3()
    S = ctx.S
    ses.set_planform(S, "wing_loading")
    assert set(S["wing"]["bounds"]) >= {"b_m", "b_rear_m"}
    ses.set_planform(S, "fixed")
    assert "b_m" not in S["wing"]["bounds"]
    assert "b_rear_m" not in S["wing"]["bounds"]
    # ...and the typed spans come back, exactly like a design-box bound
    assert ses.span_rows(S) == ("b_m", "b_rear_m")


def test_each_wings_aspect_ratio_is_read_on_its_own_area():
    """The number the solver checks. ``sizing.check_ar`` is called per wing on
    that wing's share of the pair's total, so a card quoting b²/S_total would
    report half the aspect ratio a refusal is about."""
    from gui.v3 import session as ses

    ctx = _v3()
    S = ctx.S
    assert ses.set_span_m(S, 12.0, "b_m")
    assert ses.set_span_m(S, 6.0, "b_rear_m")
    share = ses.wing_area_share(S, "b_m")
    assert 0.0 < share < 1.0
    assert ses.wing_area_share(S, "b_rear_m") == pytest.approx(1.0 - share)
    area = float(S["mission"]["s_ref_m2"])
    assert ses.wing_aspect_ratio(S, "b_m") == pytest.approx(
        144.0 / (area * share))
    assert ses.wing_aspect_ratio(S, "b_rear_m") == pytest.approx(
        36.0 / (area * (1.0 - share)))
    # and it is the one the physics enforces. `... is None or True` was a
    # tautology — X or True is True for every X — so the card could quote an
    # aspect ratio the solver refuses (a share read off the whole pair, or off
    # the wrong wing, puts b^2/(S*share) outside sizing.AR_LIMITS) and this
    # line stayed green. Asserted on the share the card itself used.
    assert sizing.check_ar(12.0, area * share) is None
    assert sizing.check_ar(6.0, area * (1.0 - share)) is None


def test_the_two_solvers_agree_that_the_mode_exists():
    """Planar and nonplanar: the same modifier, the same two rows, and the
    nonplanar one still sizes each tip device on its own semi-span."""
    planar = td.TandemProblem(size_free=sizing.SIZE_MODE_WS)
    vlm = tvm.TandemVLMProblem(winglets=True, size_free=sizing.SIZE_MODE_WS)
    assert planar.bounds.shape[0] == planar.dim
    assert vlm.param_labels[-2:] == ("b_m", "b_rear_m")
    assert planar.wing_loading_Pa is None and vlm.wing_loading_Pa is None
    banded = tvm.TandemVLMProblem(winglets=True,
                                  size_free=sizing.SIZE_MODE_WS,
                                  span_bounds_m=(12.0, 16.0), dz=2.0)
    i = banded.param_labels.index("b_m")
    assert tuple(banded.bounds[i]) == (12.0, 16.0)
    out = tvm.evaluate_tandem_vlm(_mid(banded.bounds), banded)
    assert out["feasible"], out["reason"]
    assert out["b_front"] == pytest.approx(14.0)
    assert out["b_rear"] == pytest.approx(14.0)


def test_the_mode_has_an_interior_optimum_rather_than_a_boundary_one():
    """What a sizing mode is FOR. At the pair's own loading the area closes
    against the two-structure weight, so a wider pair is a heavier, bigger
    pair: L/D climbs to an interior best and falls away again, and the ends of
    the published band are refused on aspect ratio rather than scored.

    (Both refusals are the honest kind — ``sizing.check_ar`` per wing on its
    own share of the area — and stage 3 says so under the band.)
    """
    built = api.PROBLEM_SPECS[WS].build({}, {}, None)
    i = built.param_labels.index("b_m")
    j = built.param_labels.index("b_rear_m")
    x0 = _mid(built.bounds)
    best, scores = None, {}
    for span in (8.0, 12.0, 16.0, 20.0, 24.0, 30.0, 40.0):
        x = x0.copy()
        x[i] = x[j] = span
        out = built.evaluate(x)
        scores[span] = out["LoD"] if out["feasible"] else None
        if out["feasible"] and (best is None or out["LoD"] > scores[best]):
            best = span
    assert scores[8.0] is None and scores[40.0] is None      # AR, both ends
    assert best is not None and 12.0 < best < 30.0
    assert scores[12.0] < scores[best] > scores[30.0]
