"""The design box opens on the box this mission FLIES, not on a wider one.

Two defects, one report: *"the recommended for this mission button doesn't
work, it doesn't change anything"* and *"in mission it is given that the span
is 1 m and the recommendation 7 – 17 m"*.

**The measurement was behind a button.** Measured on the free planform at the
shipped air mission, **11 of 128** Sobol draws over the family's published box
fly it (8.6 %); over the box the measurement recommends, **94 of 128** (73.4 %).
The span row alone goes 6 – 40 m to 8.95 – 14.79 m — the same row about six
times narrower. All of that was an OFFER, so the 8.6 % box is what anybody who
did not press the button searched. It is now the default.

**And a taken band outlived its mission.** ``_take_recommendation`` POPPED the
row's ``bounds_source``, which made a measured band indistinguishable from a
typed one — and a typed row is never refreshed. So taking the recommendation
for the 10 m reference wing and then stating a 1 m one in stage 1 left the box
searching **8.95 – 14.79 m** against a mission that said 1 m, with nothing on
screen saying why, and pressing the button again re-measured a box the mission
had already emptied and could only answer "no region". A recommended row is now
its own source, and ``session.drop_recommended_bounds`` takes it back when the
mission moves.

The rules the default has to keep are asserted here, not just the narrowing:
it never overrules an answer the user gave, it is not a ratchet, and "Reset to
the solver's box" means it.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _open(planform: str = "free"):
    from gui.v3.app import assemble

    ctx = assemble("air")
    ctx.act("accept_mission")
    if planform:
        ctx.act("set_planform", planform)
    ctx.render("wing", "box")
    return ctx


def _texts(ctx):
    return [getattr(e, "text", None) or ""
            for e in ctx.views[("wing", "box")].descendants()]


def _in_popup(el) -> bool:
    """Is this label the COPY behind a ``?`` rather than the line on
    screen? ``widgets.help_dot`` puts the rest of a long sentence in a
    menu classed ``help-pop``, so a card that says something once now
    renders two labels saying it."""
    node = el
    while node is not None:
        if "help-pop" in " ".join(getattr(node, "_classes", []) or []):
            return True
        slot = getattr(node, "parent_slot", None)
        node = getattr(slot, "parent", None) if slot else None
    return False


def _on_screen(ctx):
    """The card's own lines, without the popup copies of them."""
    return [getattr(e, "text", None) or ""
            for e in ctx.views[("wing", "box")].descendants()
            if not _in_popup(e)]


# ------------------------------------------- 1. it is the default, not an offer

def test_the_box_opens_on_the_band_this_mission_flies(capsys):
    from gui.v3 import config

    from aerobo import api

    ctx = _open()
    S = ctx.S
    start = {k: list(v) for k, v in S["wing"]["bounds"].items()}
    # the box the stage OPENS on is already this mission's, not the family's
    # published row — that is a separate fix and its own file
    # (``test_v3_the_box_is_sized_to_the_mission``); what is asserted here is
    # that the measurement then narrows it FURTHER
    published = api.PROBLEM_SPECS[S["wing"]["problem"]].default_bounds["b_m"]
    assert start["b_m"] != [float(published[0]), float(published[1])]
    assert config.effective_bounds(S)["b_m"][1] == "mission"

    got = ctx.act("auto_recommend")
    assert got["error"] is None and not got["empty"]
    assert "b_m" in got["rows"] and "S_m2" in got["rows"]

    eff = config.effective_bounds(S)
    lo, hi = eff["b_m"][0]
    assert eff["b_m"][1] == "recommended"
    # the row NARROWED, and it is the narrowing the user could see
    assert (hi - lo) < 0.75 * (start["b_m"][1] - start["b_m"][0])
    # ...INTO the box it was handed, never wider — the rule that keeps a
    # recommendation from being a limit
    assert start["b_m"][0] <= lo and hi <= start["b_m"][1]

    # ...and it bought what the card claims it bought
    assert got["frac_after"] > got["frac_before"] > 0.0
    assert "Traceback" not in capsys.readouterr().err


def test_the_card_says_what_it_did(capsys):
    ctx = _open()
    ctx.act("auto_recommend")
    text = " ".join(_texts(ctx))
    assert "Measured for this mission and APPLIED" in text
    assert "Reset to the solver" in text
    assert "Traceback" not in capsys.readouterr().err


# ------------------------------------------------ 2. it never overrules an answer

def test_a_row_the_user_typed_is_never_overwritten(capsys):
    from gui.v3 import config, session

    ctx = _open()
    ctx.act("set_bound", "b_m", 0, 7.5)
    ctx.act("set_bound", "b_m", 1, 12.0)
    assert session.bounds_source(ctx.S).get("b_m") is None   # theirs

    ctx.act("auto_recommend")
    band, source = config.effective_bounds(ctx.S)["b_m"]
    assert [round(v, 9) for v in band] == [7.5, 12.0]
    assert source == "user"
    # ...and the rows they did NOT answer were still measured
    assert config.effective_bounds(ctx.S)["S_m2"][1] == "recommended"
    assert "Traceback" not in capsys.readouterr().err


def test_a_fixed_row_is_not_given_a_band(capsys):
    from gui.v3 import config

    ctx = _open()
    ctx.act("set_row_fixed", "twist_tip_deg", True)
    ctx.act("set_fixed_value", "twist_tip_deg", -1.0)
    ctx.act("auto_recommend")
    assert "twist_tip_deg" not in (ctx.S["wing"].get("bounds_source") or {})
    assert "twist_tip_deg" in config.fixed_rows(ctx.S)
    assert "Traceback" not in capsys.readouterr().err


def test_typing_a_bound_does_not_re_arm_the_measurement(capsys):
    """A band is part of the configuration, so a stamp that included the box
    would re-measure on every number typed into it — a second of solver per
    keystroke, and a repaint of the table the cursor is in."""
    ctx = _open()
    ctx.act("auto_recommend")
    S = ctx.S
    S["ui"]["selected"] = "wing"
    for fn in ctx.polls:
        fn()
    assert S["wing"]["auto_rec"].get("busy") is not True

    ctx.act("set_bound", "twist_root_deg", 0, -2.0)
    for fn in ctx.polls:
        fn()
    assert S["wing"]["auto_rec"].get("busy") is not True, \
        "a typed bound re-armed the measurement"

    # ...but the PROBLEM changing does re-arm it
    ctx.act("set_choice", "tail", True)
    kicked = any(bool(fn()) for fn in ctx.polls) or \
        S["wing"]["auto_rec"].get("busy")
    assert kicked
    assert "Traceback" not in capsys.readouterr().err


# --------------------------------------------------------- 3. it is not a ratchet

def test_measuring_again_measures_the_same_box(capsys):
    """Every pass measures over a box with no measured rows in it.

    Without that each pass narrows what the last pass narrowed — every step
    defensible on its own, and three menu changes later the session is
    searching a box nobody chose.
    """
    ctx = _open()
    seen = []
    for _ in range(3):
        ctx.act("auto_recommend")
        seen.append([round(v, 9) for v in ctx.S["wing"]["bounds"]["b_m"]])
    assert seen[0] == seen[1] == seen[2], seen
    assert "Traceback" not in capsys.readouterr().err


# ------------------------------------------------------ 4. reset means reset

def test_reset_puts_the_published_box_back_and_stops_it(capsys):
    import pytest

    from gui.v3 import config, session

    ctx = _open()
    ctx.act("auto_recommend")
    assert config.effective_bounds(ctx.S)["b_m"][1] == "recommended"

    ctx.act("reset_box")
    assert ctx.S["wing"]["auto_rec"]["off"] is True
    ctx.render("wing", "box")
    assert any("is not being measured" in t for t in _texts(ctx))

    # a repaint must not re-narrow it under the cursor. "The solver's box" is
    # the box the stage OPENS on, which is this mission's own band and not the
    # family's published row (``session.size_band_defaults``) — reset means
    # un-measured, not un-missioned.
    ctx.render("wing", "box")
    assert ctx.S["wing"]["bounds"].get("b_m") == pytest.approx(
        list(session.size_band_defaults(ctx.S)["b_m"]))

    # ...and asking for it explicitly turns it back on
    ctx.act("auto_recommend")
    assert ctx.S["wing"]["auto_rec"]["off"] is False
    assert config.effective_bounds(ctx.S)["b_m"][1] == "recommended"
    assert "Traceback" not in capsys.readouterr().err


# -------------------------------------------------- 5. it follows the mission

def test_the_applied_box_does_not_outlive_its_mission(capsys):
    """The user's report: mission 1 m, design box 8.95 – 14.79 m."""
    from gui.v3 import config, session

    ctx = _open()
    ctx.act("auto_recommend")
    was = list(ctx.S["wing"]["bounds"]["b_m"])
    assert was[1] > 5.0

    assert session.set_size_from_span_ar(ctx.S, 1.0, 8.0)
    session.sync_wing_from_mission(ctx.S)
    assert session.nominal_span(ctx.S) == 1.0

    band, source = config.effective_bounds(ctx.S)["b_m"]
    assert source == "mission", source
    assert band != was
    assert band[0] <= 1.0 <= band[1], band     # a band about the wing on screen
    assert band[1] <= 4.0, band

    # ...and so do the rows that have no mission band of their own. A taper
    # measured over the 10 m wing is just as stale as its span was, and it has
    # nothing to be rewritten TO — so it goes back to the family's published
    # row rather than sitting there tagged "recommended" for a mission that
    # never saw it.
    src = ctx.S["wing"].get("bounds_source") or {}
    assert not [r for r, v in src.items() if v == "recommended"], src
    for row in ("taper", "twist_root_deg", "twist_tip_deg"):
        assert row not in ctx.S["wing"]["bounds"], row
        assert config.effective_bounds(ctx.S)[row][1] == "default", row
    assert "Traceback" not in capsys.readouterr().err


# ------------------------------------------- 6. an empty box names the gate

def test_an_empty_box_names_the_limit_that_emptied_it(capsys):
    """A 1 m wing carrying the reference load flies nothing, and saying so is
    half the job — the other half is WHICH limit refused."""
    from gui.v3 import session

    ctx = _open()
    assert session.set_size_from_span_ar(ctx.S, 1.0, 8.0)
    session.sync_wing_from_mission(ctx.S)
    got = ctx.act("auto_recommend")
    assert got["empty"] is True
    assert got["reasons"], "an empty box with no reason is the old card"
    ctx.render("wing", "box")
    text = " ".join(_texts(ctx))
    assert "None of" in text and "fly this mission" in text
    # the gate is NAMED, not merely counted
    assert any(word in text for word in ("aspect ratio", "wing loading",
                                         "chord"))
    # ...and the box is left exactly as the family publishes it
    assert ctx.S["wing"]["bounds"].get("b_m") == [0.6, 4.0]
    assert "Traceback" not in capsys.readouterr().err


# ---------------------------------- 6b. both tables, from ONE set of draws

def test_the_second_surface_s_rows_are_measured_too(capsys):
    """The draws that flew are jointly admissible, so one measurement writes
    both tables — and says so once, not once per table."""
    from gui.v3 import config
    from gui.v3.stages.wing import AFT_ROW

    ctx = _open(planform="")
    ctx.act("set_choice", "tail", True)
    ctx.act("set_planform", "free")
    got = ctx.act("auto_recommend")
    rows = got["rows"]
    assert any(AFT_ROW.search(r) for r in rows), rows
    assert any(not AFT_ROW.search(r) for r in rows), rows
    eff = config.effective_bounds(ctx.S)
    assert all(eff[r][1] == "recommended" for r in rows)
    assert got["frac_after"] > got["frac_before"]

    ctx.render("wing", "box")
    said = [t for t in _on_screen(ctx)
            if "Measured for this mission and APPLIED" in t]
    assert len(said) == 1, said
    assert "Traceback" not in capsys.readouterr().err


# ----------------------------------------------------- 7. there is no button

def test_the_box_carries_no_recommendation_button(capsys):
    """A measurement offered behind a button is a measurement most sessions
    never see. The only control left is the one that opts OUT."""
    ctx = _open()
    ctx.act("auto_recommend")
    ctx.render("wing", "box")
    buttons = [getattr(e, "text", None) or ""
               for e in ctx.views[("wing", "box")].descendants()
               if type(e).__name__ == "Button"]
    joined = " ".join(buttons)
    assert "Recommend for this mission" not in joined, buttons
    assert "Take these bands" not in joined, buttons
    assert "Dismiss" not in joined, buttons
    assert any("Reset to the solver" in b for b in buttons), buttons
    assert "Traceback" not in capsys.readouterr().err


def test_a_slow_family_is_measured_too_and_says_it_takes_a_while(capsys):
    """There is no button to fall back on, so an XFOIL family is measured in
    the background like every other — it just says so while it runs."""
    from aerobo import api

    ctx = _open(planform="")
    ctx.act("set_choice", "airfoil", "section_wing")
    assert api.PROBLEM_SPECS[ctx.S["wing"]["problem"]].slow
    ctx.S["wing"].setdefault("auto_rec", {})["busy"] = True
    ctx.render("wing", "box")
    text = " ".join(_texts(ctx))
    assert "XFOIL sweep" in text and "takes a while" in text
    assert "Traceback" not in capsys.readouterr().err


# ------------------------------------ 8. the second surface's span is QUOTED

def test_the_tail_span_is_measured_and_not_imposed(capsys):
    """b_t = sqrt(AR_t x S_t) is a DIAGONAL across two rows the box already
    bounds, so writing it as a third row can only cut their corners. Measured
    on the reference tail it took the admissible fraction 0.727 -> 0.531 and
    left the best design identical to fifteen digits."""
    ctx = _open(planform="")
    ctx.act("set_choice", "tail", True)
    ctx.act("set_choice", "tail_height", "free")
    ctx.act("set_planform", "free")
    got = ctx.act("auto_recommend")

    band = got.get("tail_span")
    assert band and band[1] > band[0], band            # it WAS measured
    # ...and not written into the flag pair the row uses
    flags = ctx.S["wing"]["flags"]
    assert flags.get("tail_span_min_m") is None
    assert flags.get("tail_span_max_m") is None
    assert "tail span" not in (ctx.S["wing"].get("bounds_source") or {})

    ctx.render("wing", "box")
    text = " ".join(_texts(ctx))
    assert "m across" in text and "NOT written into the span row" in text
    assert "Traceback" not in capsys.readouterr().err


def test_the_wing_tail_size_and_separation_rows_are_all_measured(capsys):
    """The five numbers this is for: both spans' worth of size, and how far
    apart the two surfaces are, horizontally and vertically."""
    from gui.v3 import config

    ctx = _open(planform="")
    ctx.act("set_choice", "tail", True)
    ctx.act("set_choice", "tail_height", "free")
    ctx.act("set_planform", "free")
    eff_before = config.effective_bounds(ctx.S)
    for row in ("b_m", "S_m2", "S_t_m2", "l_t_m", "z_t_m"):
        assert row in eff_before, row

    got = ctx.act("auto_recommend")
    eff = config.effective_bounds(ctx.S)
    measured = []
    for row in ("b_m", "S_m2", "S_t_m2", "l_t_m", "z_t_m"):
        was, now = eff_before[row][0], eff[row][0]
        # NONE of the five is left on the family's published row: it is either
        # this mission's own band or the measurement taken over it
        assert eff[row][1] in ("mission", "recommended"), (row, eff[row][1])
        assert was[0] <= now[0] and now[1] <= was[1], row   # never wider
        if eff[row][1] == "recommended":
            assert (now[1] - now[0]) < (was[1] - was[0]), row
            measured.append(row)
    # ...and the four SIZE numbers are measured. The vertical separation is
    # the one that need not be: the box it starts from is already the
    # mission's 0.05-0.30 b, and on the reference aeroplane the kept designs
    # use all of it, so there is nothing to narrow and the card counts it
    # held rather than proposing a band that says nothing.
    assert set(measured) >= {"b_m", "S_m2", "S_t_m2", "l_t_m"}, measured
    assert got["frac_after"] > got["frac_before"]
    assert "Traceback" not in capsys.readouterr().err
