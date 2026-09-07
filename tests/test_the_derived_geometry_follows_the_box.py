"""The card derived FROM the box may not contradict the box.

Stage 3's "Derived geometry" panel sits one card below the design-box table.
With the planform searched, the table read ``b_m 9.91 – 11.92 m`` and
``S_m2 18.0 – 21.6 m²`` while the panel under it said

    span b        10 m · fixed
    area S        10 m² · fixed
    aspect ratio  10 · fixed

— three rows that were wrong in the same way: the wing they quoted is the
family's PUBLISHED planform, which no searched-size run flies, and the word
"fixed" denied the two rows the table had just shown as bands. That is this
repo's own "the box shown is the box searched" failure, drawn under the box
itself.

The cause was one line: ``nice_app.geometry_summary`` chose its branch from
the FAMILY (only ``free planform (aircraft)`` and ``tandem`` got band rows),
when what decides whether a span is a design variable is whether it is a ROW
— which the size modifier's twins, both wing-loading modes and the car all
are. So the branch is taken from the box now, and these tests hold it there.

Three modes, three different truths about the area, and none of them may be
invented:

* free span + area   — both are rows, so both are bands;
* free span (W/S)    — the area is S = W/(W/S), the MISSION's number, which
                       is not the family's published one and is passed in;
* free span + W/S    — the loading is a row too, so the area sweeps with it
                       and no single number is honest at all.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _card(ctx) -> dict:
    """The Derived geometry rows as ``{label: value}``, as RENDERED."""
    ctx.render("wing")
    view = ctx.views[("wing", "box")]
    texts = [getattr(e, "text", "") or "" for e in view.descendants()]
    start = [i for i, t in enumerate(texts) if "Derived geometry" in t]
    assert start, "the wing's box view drew no Derived geometry card"
    # the card is the LAST thing this view builds (``boxes["geo"]``), and its
    # rows are label/value pairs with empty container elements between them
    rest = [t for t in texts[start[0] + 1:] if t]
    return dict(zip(rest[::2], rest[1::2]))


def _shell(planform: str):
    from gui.v3.app import assemble

    ctx = assemble("air")
    ctx.act("accept_mission")
    ctx.act("set_winglet", "none")
    ctx.act("set_planform", planform)
    return ctx


# ------------------------------------------- the searched size is searched

def test_a_searched_span_and_area_are_reported_as_the_bands_they_are():
    """The reported defect, as arithmetic: every number in the card is a
    corner of the box, and the word "fixed" appears nowhere."""
    from gui.v3 import config, session

    ctx = _shell("free")
    eff = config.effective_bounds(ctx.S)
    b_lo, b_hi = eff["b_m"][0]
    s_lo, s_hi = eff[session.AREA_ROW][0]
    rows = _card(ctx)

    assert "searched" in rows["span b"] and "fixed" not in rows["span b"]
    assert "searched" in rows["area S"] and "fixed" not in rows["area S"]
    assert f"{b_lo:.4g}" in rows["span b"] and f"{b_hi:.4g}" in rows["span b"]
    assert f"{s_lo:.4g}" in rows["area S"] and f"{s_hi:.4g}" in rows["area S"]
    # ...and the aspect ratio is the box's own corners, not b²/S of a wing
    # nobody is flying
    ar = rows["aspect ratio"]
    assert f"{b_lo ** 2 / s_hi:.3g}" in ar and f"{b_hi ** 2 / s_lo:.3g}" in ar


def test_a_searched_span_against_a_stated_area_quotes_the_mission():
    """The wing-loading mode. The area is not a design row — so it is not a
    band — but it is the MISSION's S = W/(W/S), and the family's published
    10 m² stood there for every mission that states another."""
    ctx = _shell("wing_loading")
    ctx.S["mission"]["s_ref_m2"] = 17.0
    rows = _card(ctx)

    assert "searched" in rows["span b"]
    assert "17" in rows["area S"], rows["area S"]
    assert "fixed" not in rows["area S"]


def test_a_searched_loading_has_no_area_to_quote():
    """Both rows searched: the area is W_total/(W/S) and W_total grows with
    the wing, so no number here would be true. The row says what sets it."""
    ctx = _shell("wing_loading_free")
    rows = _card(ctx)

    assert "searched" in rows["span b"]
    assert "W/S" in rows["area S"]
    # nothing invented: no bare "10 m²" pretending to be the area
    assert "m²" not in rows["area S"].split("—")[0]


def test_a_fixed_planform_still_reports_the_wing_it_flies():
    """The control. With the size chosen there IS one planform, and this card
    is where it is read — the fix may not turn that into a band."""
    ctx = _shell("fixed")
    rows = _card(ctx)
    assert "fixed" in rows["span b"] and "fixed" in rows["area S"]


# ------------------------------------------------- the pair, on its own share

def test_a_pair_reports_both_spans_on_half_the_area():
    """A tandem searches a span per wing against the pair's TOTAL area, and
    the solvers check each wing at ``0.5 * S_total`` (the convention
    ``session.clip_size_box`` is written on). One AR row per wing, on that
    share — not b²/S_total, which is neither wing's aspect ratio."""
    from gui.v3 import config, session

    from gui.v3.app import assemble

    ctx = assemble("air")
    ctx.act("accept_mission")
    ctx.act("set_lifting_system", "tandem")
    ctx.act("set_planform", "free")
    eff = config.effective_bounds(ctx.S)
    rows = _card(ctx)

    assert "front span" in rows and "rear span" in rows
    b_lo, b_hi = eff["b_m"][0]
    s_lo, s_hi = eff[session.AREA_ROW][0]
    ar = rows["front span AR"]
    assert f"{b_lo ** 2 / (0.5 * s_hi):.3g}" in ar
    assert f"{b_hi ** 2 / (0.5 * s_lo):.3g}" in ar


# --------------------------------------------- the rule, at its own level

def test_the_box_decides_the_branch_not_the_family():
    """``geometry_summary`` on its own terms: the same family reports a wing
    or a box depending on what the box carries."""
    from gui import nice_app as v1

    published = dict(v1.geometry_summary("trim wing", {"taper": [0.2, 1.0]}))
    assert published["span b"] == "10 m · fixed"

    searched = dict(v1.geometry_summary(
        "trim wing", {"taper": [0.2, 1.0], "b_m": [9.0, 12.0],
                      "S_m2": [18.0, 22.0]}))
    assert "searched" in searched["span b"]
    assert "9" in searched["span b"] and "12" in searched["span b"]
    assert "aspect ratio" in searched


def test_a_caller_that_cannot_state_the_area_gets_no_area_row():
    """A number this panel made up is worse than a row it does not draw."""
    from gui import nice_app as v1

    rows = dict(v1.geometry_summary("trim wing",
                                    {"taper": [0.2, 1.0], "b_m": [9.0, 12.0]}))
    assert "searched" in rows["span b"]
    assert "area S" not in rows
    assert "aspect ratio" not in rows
