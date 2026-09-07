"""The design box is one table PER SURFACE, and a span is a row of one.

Two changes, and the second one settled the first.

The second surface's span was asked in a card of its own, two panels below
the box, as a "minimum span"/"maximum span" pair. It is a BAND on a length,
which is what every row of the design box already is, so it became a row.

And the box itself was ONE table: "how big may the tail be" is a question
about one object, and it was answered by four rows scattered through eleven,
between the wing's twists and its chord law. So the table is split by the
surface each row belongs to (``AFT_ROW``) — a WING design box and a second
surface's — and the span band leads its own surface's table the way ``b_m``
leads the wing's.

A TANDEM pair is deliberately not split: its two spans belong side by side.

The size-limit cards below are one per surface too, each holding the chords
only, each saying where its span went.
"""

from __future__ import annotations

import pytest

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _open(medium: str = "air", planform: str | None = None, **choices):
    from gui.v3.app import assemble

    ctx = assemble(medium)
    ctx.act("accept_mission")
    for key, value in choices.items():
        ctx.act("set_choice", key, value)
    if planform:
        ctx.act("set_planform", planform)
    ctx.render("wing", "box")
    return ctx


def _texts(view):
    return [getattr(e, "text", None) or "" for e in view.descendants()]


def _switch_after(view, label: str):
    from nicegui import ui

    seen = False
    for e in view.descendants():
        if getattr(e, "text", None) == label:
            seen = True
        elif seen and isinstance(e, ui.switch):
            return e
    raise AssertionError(f"no switch after {label!r}")


def _fields_after(view, label: str):
    seen, out = False, []
    for e in view.descendants():
        if getattr(e, "text", None) == label:
            seen = True
        elif seen and type(e).__name__ == "Number":
            out.append(e)
    return out


def test_the_span_is_in_the_box_and_the_card_says_where(capsys):
    ctx = _open(tail=True)
    texts = _texts(ctx.views[("wing", "box")])

    # asked ONCE, and the place is the TAIL's own table
    assert texts.count("tail span") == 1
    assert texts.index("Tail design box") < texts.index("tail span")
    assert texts.index("tail span") < texts.index("Tail size limits")
    assert "tail_span_min_m / tail_span_max_m" in texts

    # ...and the card it left no longer asks it, in either direction
    assert "minimum span" not in texts and "maximum span" not in texts
    assert any(t.startswith("The tail SPAN is not asked here") for t in texts)
    assert "Traceback" not in capsys.readouterr().err


def test_each_table_is_led_by_its_own_surface_s_span(capsys):
    """The wing-loading families SEARCH b_m, so that row leads the WING's
    table; the band leads the tail's. Each surface's span is the first thing
    said about it."""
    ctx = _open(tail=True, planform="wing_loading")
    texts = _texts(ctx.views[("wing", "box")])
    assert "b_m" in texts, "this family searches the span"
    i_wing = texts.index("Wing design box")
    i_aft = texts.index("Tail design box")
    assert i_wing < texts.index("b_m") < i_aft < texts.index("tail span")
    # nothing of the wing's between the heading and its span row
    lead = texts[i_wing:texts.index("b_m")]
    assert not any(t == "taper" or t.startswith("twist") for t in lead)
    # ...and nothing of the tail's between ITS heading and its span row
    lead = texts[i_aft:texts.index("tail span")]
    assert not any(t in ("S_t_m2", "l_t_m", "AR_t") for t in lead)
    assert "Traceback" not in capsys.readouterr().err


def test_every_row_is_filed_by_the_surface_it_constrains(capsys):
    """The split is the point: a tail row in the wing's table is the bug this
    guards. Checked against the rows the family actually declares, and in
    BOTH directions — a wing row swept into the tail's table is the same
    defect mirrored.
    """
    from gui.v3 import config
    from gui.v3.stages.wing import AFT_ROW

    from gui.v3.stages.wing import CHORD_ROW

    ctx = _open(tail=True, tail_height="free", tail_design="planform")
    texts = _texts(ctx.views[("wing", "box")])
    i_aft = texts.index("Tail design box")
    # the chord-law COEFFICIENTS are not in either table: each law has a
    # panel of its own below both, which is where its own rows are asked
    rows = [r for r in config.effective_bounds(ctx.S)
            if r in texts and not CHORD_ROW.fullmatch(r)]
    assert len(rows) > 6, "expected a family with rows on both surfaces"
    for row in rows:
        aft = bool(AFT_ROW.search(row))
        assert (texts.index(row) > i_aft) is aft, row
    # and the marker really did split them, rather than matching everything
    assert any(AFT_ROW.search(r) for r in rows)
    assert any(not AFT_ROW.search(r) for r in rows)
    assert "Traceback" not in capsys.readouterr().err


def test_a_pair_is_not_split_and_keeps_its_two_spans_together(capsys):
    """A tandem's rear wing is a second surface, and splitting it out would
    put the pair's two span bands in two tables — which is the reading the
    single table exists to prevent."""
    ctx = _open(system="tandem")
    texts = _texts(ctx.views[("wing", "box")])
    assert "Design box" in texts
    assert "Wing design box" not in texts
    assert not any(t == "Rear wing design box" for t in texts)
    assert "Traceback" not in capsys.readouterr().err


def test_switching_it_on_writes_both_ends_and_reaches_the_run(capsys):
    from gui.v3 import config

    ctx = _open(tail=True)
    box = ctx.views[("wing", "box")]
    assert ctx.S["wing"]["flags"].get("tail_span_min_m") is None

    _switch_after(box, "tail_span_min_m / tail_span_max_m").set_value(True)
    ctx.render("wing", "box")
    flags = ctx.S["wing"]["flags"]
    lo, hi = flags["tail_span_min_m"], flags["tail_span_max_m"]
    assert lo is not None and hi is not None and lo < hi

    # a band is only a band if it is TYPED into, and the number the user
    # types is the number the problem is built with
    box = ctx.views[("wing", "box")]
    fields = _fields_after(box, "tail_span_min_m / tail_span_max_m")
    fields[1].set_value(2.0)
    built = config.build_cfg(ctx.S)
    assert built.flags["tail_span_max_m"] == 2.0

    from aerobo import api, tail as tailmod
    problem = api.PROBLEM_SPECS[built.problem_name].build(
        built.mission_kwargs, built.flags, None)
    labels = list(problem.param_labels)
    s_hi = float(problem.bounds[labels.index("S_t_m2")][1])
    if "AR_t" not in labels:
        # A FIXED aspect ratio: b = sqrt(AR·S) is exact in one variable, so
        # the cap IS the area row and the corner the sampler may draw obeys it
        ar = float(tailmod.TAIL_AR)
        assert (ar * s_hi) ** 0.5 <= 2.0 + 1e-9
    else:
        # A DESIGNED tail searches its own aspect ratio, and b <= 2 is then a
        # hyperbola in (AR, S) that no rectangle can be: the box is narrowed
        # to the tightest one that CONTAINS the feasible set (S <=
        # b_max²/AR_min), and the corner outside it is refused rather than
        # quietly reshaped — which is what TAIL_LIMIT_KEYS says these are.
        ar_lo = float(problem.bounds[labels.index("AR_t")][0])
        ar_hi = float(problem.bounds[labels.index("AR_t")][1])
        assert s_hi == pytest.approx(2.0 * 2.0 / ar_lo)
        lim = api.tail_limits_of(built.flags)
        corner_b = (ar_hi * s_hi) ** 0.5
        assert corner_b > 2.0                       # the corner IS outside
        assert lim.violation(corner_b, (s_hi / corner_b)), \
            "a design past the stated span has to be refused by the limit"
        assert lim.violation((ar_lo * s_hi) ** 0.5,
                             (s_hi / ((ar_lo * s_hi) ** 0.5))) is None
    assert "Traceback" not in capsys.readouterr().err


def test_switching_it_off_is_the_published_problem(capsys):
    from gui.v3 import config

    ctx = _open(tail=True)
    published = config.build_cfg(ctx.S).flags
    box = ctx.views[("wing", "box")]
    sw = _switch_after(box, "tail_span_min_m / tail_span_max_m")
    sw.set_value(True)
    ctx.render("wing", "box")
    box = ctx.views[("wing", "box")]
    _switch_after(box, "tail_span_min_m / tail_span_max_m").set_value(False)
    after = config.build_cfg(ctx.S).flags
    assert after.get("tail_span_min_m") is None
    assert after.get("tail_span_max_m") is None
    assert {k: v for k, v in after.items() if k.startswith("tail_span")} == \
        {k: v for k, v in published.items() if k.startswith("tail_span")}
    assert "Traceback" not in capsys.readouterr().err


def test_a_pair_calls_it_by_the_surface_s_own_name(capsys):
    """The row and the card are named off the CONFIGURATION
    (``session.second_surface_name``), so a pair's second surface is a "rear
    wing" wherever it is mentioned — a hard-coded "tail" is right on three
    families and wrong on this one.

    WHICH BRANCH APPLIES IS THE REGISTRY'S ANSWER, not the page's. The old
    form read it off the page — no size-limit card, therefore no row — and
    that is the only path a tandem can take today, so it returned before the
    naming assertion and the traceback guard ever ran, and a box that
    rendered NOTHING (or a shell that stopped building the pair at all) went
    green on the fallback negative.
    """
    from aerobo import api
    from gui.v3 import session
    from gui.v3.stages.wing import TAIL_SPAN_ROW

    ctx = _open(system="tandem")
    texts = _texts(ctx.views[("wing", "box")])
    # the preconditions the case rests on, asserted instead of assumed
    assert session.second_surface_name(ctx.S) == "rear wing", \
        "this test is about a pair whose second surface has a name of its own"
    assert "Design box" in texts, "the pair's design box did not render"

    spec = api.PROBLEM_SPECS[ctx.S["wing"]["problem"]]
    if all(k in spec.flags for k in TAIL_SPAN_ROW):
        # the family accepts TailLimits: the band is offered, under the
        # PAIR's word for the surface and never the tail's
        assert "rear wing span" in texts
        assert f"{TAIL_SPAN_ROW[0]} / {TAIL_SPAN_ROW[1]}" in texts
        assert "tail span" not in texts
        # ...and the chord card beside it, wherever the family asks for one
        assert "Tail size limits" not in texts
    else:
        # it does not: then the band must not be offered AT ALL — a row here
        # would send the run a flag the builder never reads (api.check_flags
        # refuses it) — and no card may name this surface "tail" either
        assert "rear wing span" not in texts
        assert "tail span" not in texts
        assert not any(t.endswith("size limits") and t != "Wing size limits"
                       for t in texts)
    assert "Traceback" not in capsys.readouterr().err


def test_no_second_surface_no_row(capsys):
    ctx = _open()
    texts = _texts(ctx.views[("wing", "box")])
    # nothing to split, so the table keeps its published name
    assert "Design box" in texts and "Wing design box" not in texts
    assert not any(t.endswith(" span") and t != "span" for t in texts)
    assert "Wing size limits" in texts
    assert not any(t.endswith("size limits") and t != "Wing size limits"
                   for t in texts)
    assert "Traceback" not in capsys.readouterr().err


def test_a_maximum_under_the_minimum_is_refused_here_and_not_at_launch(capsys):
    """``TailLimits`` raises on an inverted band, and it raises inside the
    problem BUILDER — so a card that accepted one turned Launch into a
    traceback. The guard was there and was dead: ``effective_bounds``
    narrows S_t_m2 through the limits and never constructs them.
    """
    from gui.v3 import config

    ctx = _open(tail=True)
    box = ctx.views[("wing", "box")]
    _switch_after(box, "tail_span_min_m / tail_span_max_m").set_value(True)
    ctx.render("wing", "box")
    box = ctx.views[("wing", "box")]
    lo = ctx.S["wing"]["flags"]["tail_span_min_m"]

    _fields_after(box, "tail_span_min_m / tail_span_max_m")[1] \
        .set_value(0.5 * lo)
    # the typed number did not stick...
    assert ctx.S["wing"]["flags"]["tail_span_max_m"] > lo
    # ...and the run it would have broken still builds
    built = config.build_cfg(ctx.S)
    from aerobo import api
    api.PROBLEM_SPECS[built.problem_name].build(
        built.mission_kwargs, built.flags, None)
    assert "Traceback" not in capsys.readouterr().err
