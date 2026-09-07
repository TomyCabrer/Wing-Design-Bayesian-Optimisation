"""The design box is about the wing on screen, and 0 is an answer.

Two defects with one shape: a number the user stated could not reach the box.

**The size box ignored the mission entirely.** ``api._make_variant_spec``
strips ``b_m``/``S_m2`` from a SIZED spec's flags — deliberately, because a
typed size beside a searched one is two answers to one question — and
``objective.Problem`` then falls back on its own 10 m / 10 m^2 reference. Every
sized family therefore opened on ``sizing.size_bounds(10.0, 10.0)``: span
6-40 m and area 8-22 m^2, for EVERY mission. A 25 kg aeroplane whose mission
states 4 m and 2.0 m^2 was searched over a box whose smallest wing is four
times its own, and a 64-draw probe put the best payload L/D at 14.6 there
against 36.2 over the mission-sized box.

**And 0 was refused where it is a real question.** "As small as this can get"
is what a designer means by typing zero into a lower bound — a vanishing tip
chord, a stabiliser shrunk until the trim solve gives up, a foil at the
surface. The solve answers all three honestly (INFEASIBLE, with a reason); it
was the form that said no.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aerobo import api                                   # noqa: E402
from aerobo.sizing import MIN_POSITIVE                   # noqa: E402


def _shell():
    from gui.v3.app import assemble

    ctx = assemble("air")
    ctx.act("accept_mission")
    return ctx


def _mission(ctx, span=4.0, ar=8.0):
    from gui.v3 import session

    assert session.set_size_from_span_ar(ctx.S, span, ar)
    session.sync_wing_from_mission(ctx.S)
    return ctx


def _built(S):
    from gui.v3 import config

    cfg = config.build_cfg(S)
    return cfg, api.PROBLEM_SPECS[cfg.problem_name].build(
        cfg.mission_kwargs, cfg.flags, cfg.bounds_overrides)


# ----------------------------------------------- 1. the box follows the mission

def test_the_free_planform_searches_the_wing_the_mission_asked_for():
    from gui.v3 import config, session

    ctx = _mission(_shell())
    ctx.act("set_planform", "free")
    S = ctx.S
    assert session.nominal_span(S) == 4.0
    assert S["mission"]["s_ref_m2"] == 2.0

    eff = config.effective_bounds(S)
    span, area = eff["b_m"][0], eff["S_m2"][0]
    # the fractional band the engine uses, about the MISSION's wing
    assert span[0] < 4.0 < span[1]
    assert area[0] < 2.0 < area[1]
    assert span[1] < 20.0 and area[1] < 6.0      # not the 10 m reference wing

    # ...and the box SHOWN is the box SEARCHED
    _cfg, built = _built(S)
    labs = list(built.param_labels)
    box = np.asarray(built.bounds, dtype=float)
    assert np.allclose(box[labs.index("b_m")], span)
    assert np.allclose(box[labs.index("S_m2")], area)


def test_a_mission_stated_after_the_mode_still_moves_the_band():
    """The natural order — the session opens on a complete default mission, so
    a user picks the planform first and states the real one afterwards. The
    band used to be written once, tagged as if the user had typed it, and
    never revisited."""
    from gui.v3 import config

    ctx = _shell()
    ctx.act("set_planform", "wing_loading")
    was = list(config.effective_bounds(ctx.S)["b_m"][0])

    _mission(ctx)
    now = config.effective_bounds(ctx.S)["b_m"]
    assert list(now[0]) != was
    assert now[0][0] < 4.0 < now[0][1]
    assert now[1] == "mission"          # the shell wrote it, and says so


def test_a_band_the_user_typed_is_never_moved_by_the_mission():
    from gui.v3 import config

    ctx = _mission(_shell())
    ctx.act("set_planform", "free")
    ctx.act("set_bound", "b_m", 0, 3.0)
    ctx.act("set_bound", "b_m", 1, 9.0)
    assert config.effective_bounds(ctx.S)["b_m"] == ([3.0, 9.0], "user")

    _mission(ctx, span=6.0, ar=8.0)
    assert config.effective_bounds(ctx.S)["b_m"] == ([3.0, 9.0], "user")
    # ...while the row NEXT to it, which the shell owns, did follow
    assert config.effective_bounds(ctx.S)["S_m2"][1] == "mission"


def test_choosing_the_free_planform_no_longer_deletes_the_span_band():
    """``span_is_searched`` excludes the ``size`` modifier on purpose (it asks
    "is the span searched against a LOADING"), and the mode switch read that
    as "the span is not a question here" — so choosing the one mode whose span
    really is a design variable popped the row."""
    from gui.v3 import config

    ctx = _mission(_shell())
    ctx.act("set_planform", "free")
    assert "b_m" in ctx.S["wing"]["bounds"]
    assert config.effective_bounds(ctx.S)["b_m"][0][1] < 20.0


def test_the_mission_sized_box_lets_the_conflict_card_speak():
    """A box four times too large hid the mission's own wing-loading ceiling:
    ``api.size_box_conflicts`` proves emptiness DOWNWARDS, so an oversized
    area row always satisfied it. Sized to the mission, a mission whose weight
    and area disagree is caught before the run."""
    ctx = _mission(_shell())          # 652.8 N over 2.0 m^2 = 326 Pa
    ctx.act("set_planform", "free")
    cfg, _built_ = _built(ctx.S)
    found = api.size_box_conflicts(cfg)
    assert found
    assert any(f["row"] == "S_m2" for f in found)


# ------------------------------------------------------------ 2. zero, honestly

def test_zero_is_read_as_as_close_to_zero_as_the_row_allows():
    """Only for the SCALE rows, and every one of them is named.

    These are the quantities the solve divides by or derives an aspect ratio
    or a loading from, so zero is degenerate for the arithmetic while being a
    perfectly good question about the design.
    """
    for name, row, hi in [("tail", "S_t_m2", 3.0),
                          ("free planform (aircraft)", "b_m", 40.0),
                          ("trim wing + free planform", "S_m2", 22.0),
                          ("trim wing", "taper", 1.0)]:
        assert row in api.ZERO_MEANS_SMALLEST
        built = api.PROBLEM_SPECS[name].build({}, {}, {row: [0.0, hi]})
        i = list(built.param_labels).index(row)
        lo = float(np.asarray(built.bounds, dtype=float)[i][0])
        assert lo == MIN_POSITIVE, (name, row, lo)


def test_a_row_where_zero_is_an_ordinary_value_is_passed_through_exactly():
    """``z_t_m`` = 0 is a COPLANAR tail — a layout this repo deliberately
    allows, not a degenerate one. An epsilon there would answer a question
    nobody asked, and it is what the first cut of this rule got wrong."""
    name = "tail + winglet [free height]"
    built = api.PROBLEM_SPECS[name].build(
        {}, {}, {api.TAIL_HEIGHT_KEY: [0.0, 3.0]})
    i = list(built.param_labels).index(api.TAIL_HEIGHT_KEY)
    assert tuple(np.asarray(built.bounds, dtype=float)[i]) == (0.0, 3.0)
    assert api.TAIL_HEIGHT_KEY not in api.ZERO_MEANS_SMALLEST


def test_a_row_the_solve_would_LIE_about_at_zero_keeps_its_refusal():
    """The other half of the rule, and the reason it is a whitelist.

    Measured at 1e-9: ``V_ms`` returns ``solver failure: untrimmable:
    alpha = 9122722`` — the trim solve blowing up, reported as if it were a
    finding — and ``depth_m`` returns FEASIBLE, a foil flying happily at the
    free surface it would ventilate through. A refusal that says why beats an
    answer that is wrong.
    """
    for row, band in [("V_ms", [0.0, 16.0]), ("depth_m", [0.0, 1.2])]:
        assert row not in api.ZERO_MEANS_SMALLEST
        try:
            api.PROBLEM_SPECS["hydrofoil"].build({}, {}, {row: band})
        except ValueError as exc:
            assert "at or below zero" in str(exc)
        else:                                     # pragma: no cover
            raise AssertionError(f"{row} accepted a zero low end")


def test_a_row_that_may_be_negative_keeps_its_negative_end():
    """The test is the row's PUBLISHED bound, never the sign of the number
    typed: a rule keyed on "the value is <= 0" would silently rewrite a typed
    washout of -6 degrees into +1e-9."""
    for name, row, band in [("trim wing", "twist_root_deg", [-4.0, 4.0]),
                            ("trim wing", "twist_tip_deg", [-6.0, 2.0]),
                            ("wing (free chord law)", "chord_k1", [-0.5, 0.5])]:
        built = api.PROBLEM_SPECS[name].build({}, {}, {row: band})
        i = list(built.param_labels).index(row)
        assert np.allclose(np.asarray(built.bounds, dtype=float)[i], band)


def test_a_zero_band_is_answered_by_the_physics_not_refused_by_the_form():
    """The point of allowing it: the search reports what happens in the limit
    rather than the form deciding nobody may ask."""
    built = api.PROBLEM_SPECS["tail"].build({}, {}, {"S_t_m2": [0.0, 3.0]})
    labs = list(built.param_labels)
    x = np.asarray(built.bounds, dtype=float).mean(axis=1)
    x[labs.index("S_t_m2")] = MIN_POSITIVE
    got = built.evaluate(x.tolist())
    assert got.get("feasible") is False
    assert got.get("reason")                    # it SAYS why, in its own words


def test_a_negative_low_end_is_still_refused_where_it_cannot_mean_anything():
    for name, ov in [("tail", {"S_t_m2": [-1.0, 3.0]}),
                     ("hydrofoil", {"depth_m": [-0.5, 1.2]})]:
        try:
            api.PROBLEM_SPECS[name].build({}, {}, ov)
        except ValueError as exc:
            assert "below" in str(exc).lower()
        else:                                     # pragma: no cover
            raise AssertionError(f"{name} accepted a negative low end")


# ------------------------------------------- 3. a zero the shell used to eat

def test_a_flag_worth_zero_reaches_the_run():
    """``if v not in (None, False)`` dropped every 0.0-valued flag, because
    ``0.0 in (None, False)`` is True. A card could show a zero the run never
    received, and the family's own default was used with nothing saying so."""
    from gui.v3 import config

    ctx = _shell()
    S = ctx.S
    sent = config.flags(S)
    for key in ("chord_min_m",):
        if key not in api.PROBLEM_SPECS[S["wing"]["problem"]].flags:
            continue
        S["wing"]["flags"][key] = 0.0
        assert config.flags(S).get(key) == 0.0
        del S["wing"]["flags"][key]
    assert sent is not None


def _box_number_ids(ctx):
    """The identity of every number input in the design box, in order.

    Identity, not value: an input the user is typing into that comes back
    with the same value but a NEW id was destroyed and rebuilt under their
    cursor, which swallows the rest of the number. Nothing else can see that.
    """
    return [e.id for e in ctx.views[("wing", "box")].descendants()
            if type(e).__name__ == "Number"]


def test_typing_zero_into_a_chord_limit_does_not_blank_the_design_box():
    """``ChordLimits`` refuses a chord minimum of 0 — rightly, a minimum of
    zero is not a constraint — and the refusal used to be raised from the
    RENDERER, which ``Ctx.render`` contains by replacing the whole design box
    with a traceback panel. The flag stayed 0.0, so every later render did it
    again and the box never came back.

    AND THE FIELD ITSELF HAS TO SURVIVE THE REFUSAL. The rolled-back flag was
    only half of it: the failure path went on to call ``_render_box`` — which
    begins ``box.clear()`` — from inside the handler of a ``ui.number``
    ``_render_box`` had created, so a keystroke this card refuses replaced
    all 16 number inputs in the box while one of them was being typed in.
    Select-all-and-retype and any transient min > max reach here, so it was
    the ordinary way to edit these two fields.
    """
    ctx = _shell()
    ctx.render("wing", "box")
    ctx.act("set_chord_limit_on", "chord_min_m", True)
    before = float(ctx.S["wing"]["flags"]["chord_min_m"])
    ids = _box_number_ids(ctx)
    assert ids, "the design box must have number inputs to lose"

    ctx.act("set_chord_limit", "chord_min_m", 0.0)
    assert ctx.S["wing"]["flags"]["chord_min_m"] == before      # rolled back
    assert _box_number_ids(ctx) == ids, (
        "a REFUSED keystroke rebuilt the fields under the cursor")
    # ...and the accepted keystroke is the control: it never rebuilt them
    ctx.act("set_chord_limit", "chord_min_m", before * 1.05)
    assert _box_number_ids(ctx) == ids

    ctx.render("wing", "box")
    texts = [getattr(e, "text", None) or ""
             for e in ctx.views[("wing", "box")].descendants()]
    assert not any("failed to render" in t for t in texts)


def test_a_refused_tail_size_limit_keeps_the_field_it_was_typed_in():
    """The SECOND surface's half of the same card, and the same failure.

    Untestable until now for a duller reason: ``set_tail_limit`` and
    ``set_tail_limit_on`` were the only two size handlers in the stage never
    registered as ``ctx.act`` names, so no test could drive the real path at
    all — the wing half was covered and this one was not.
    """
    from gui.v3.app import assemble

    ctx = assemble("air")
    ctx.act("accept_mission")
    ctx.act("set_choice", "tail", True)
    ctx.render("wing", "box")
    ctx.act("set_tail_limit_on", "tail_chord_min_m", True)
    ctx.act("set_tail_limit_on", "tail_chord_max_m", True)
    lo = float(ctx.S["wing"]["flags"]["tail_chord_min_m"])
    hi = float(ctx.S["wing"]["flags"]["tail_chord_max_m"])
    ids = _box_number_ids(ctx)
    assert ids

    # a minimum above the maximum — what a half-finished retype looks like
    ctx.act("set_tail_limit", "tail_chord_min_m", hi + 1.0)
    assert ctx.S["wing"]["flags"]["tail_chord_min_m"] == lo     # rolled back
    assert _box_number_ids(ctx) == ids, (
        "a REFUSED keystroke rebuilt the fields under the cursor")
    ctx.act("set_tail_limit", "tail_chord_min_m", 0.5 * lo)
    assert ctx.S["wing"]["flags"]["tail_chord_min_m"] == 0.5 * lo
    assert _box_number_ids(ctx) == ids
