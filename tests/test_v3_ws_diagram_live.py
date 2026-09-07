"""The constraint diagram moves when the mission does.

The (W/S, T/W) card drew itself ONCE, at build time. Every number that
decides what it shows — the mission's speed and density on the left, the
stall speed and the drag pair and the available T/W inside the expansion —
could then be typed without the picture moving, so a user who raised the
weight or tightened the stall requirement went on reading the diagram of a
mission they no longer had, with an "Allowed: W/S <= ..." line and an adopt
button that both quoted the old one.

Two things have to be true at once, and they pull against each other:

* the derived half redraws on every edit (the figure, the band, the notes,
  the adopt button, and the ceiling read out above the expansion);
* the FIELDS do not. Rebuilding a number field from inside its own
  ``on_change`` swallows the rest of the number being typed, so these tests
  assert the input elements survive the redraw as the same objects.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _shell(medium: str = "air"):
    from gui.v3.app import assemble

    return assemble(medium)


def _view(ctx):
    return ctx.views[("mission", "operating")]


def _numbers(view) -> dict:
    """Every numeric field on the view, keyed by the label beside it.

    ``widgets.number_field`` writes the label and then the input, so a
    depth-first walk carries the caption forward onto the field it belongs
    to — the fields themselves carry no label prop to match on.
    """
    from nicegui.elements.label import Label
    from nicegui.elements.number import Number

    out, last = {}, ""
    for e in view.descendants():
        if isinstance(e, Label):
            last = (e.text or "").strip() or last
        elif isinstance(e, Number):
            out[last] = e
    return out


def _figures(view) -> list:
    from nicegui.elements.plotly import Plotly

    return [e for e in view.descendants() if isinstance(e, Plotly)]


def _ws_max_line(view) -> float:
    """Where the binding W/S limit is DRAWN — one number off the figure the
    user is looking at, rather than off the model behind it.

    The limit is a two-point vertical trace. The long curves next to it come
    through as binary blobs (plotly base64-encodes numpy arrays into
    ``{"dtype", "bdata"}``), which is why only plain coordinate LISTS are
    read here.
    """
    figs = _figures(view)
    assert figs, "the diagram card draws no figure at all"
    fig = figs[0].figure
    xs = []
    for tr in fig.get("data", ()):
        x = tr.get("x")
        if isinstance(x, (list, tuple)) and len(x) == 2 and x[0] == x[1]:
            xs.append(float(x[0]))
    assert xs, "no vertical W/S limit is drawn on this diagram"
    return min(xs)


def _cruise_curve(view):
    """The cruise T/W requirement as the client receives it — the blob is
    compared, not decoded: what matters is whether the picture changed."""
    fig = _figures(view)[0].figure
    tr = next(t for t in fig["data"] if t.get("name") == "cruise")
    return str(tr.get("y"))


def test_a_typed_requirement_redraws_the_diagram():
    """The stall speed IS the binding limit in air: type a slower one and
    the vertical line the whole card is about has to move with it."""
    ctx = _shell("air")
    view = _view(ctx)
    field = _numbers(view)["stall / approach speed"]
    before = _ws_max_line(view)

    field.value = float(field.value) * 0.8         # fires the real handler

    after = _ws_max_line(view)
    assert after < before, (
        f"a slower stall speed must lower the drawn W/S limit "
        f"({before:.1f} -> {after:.1f} Pa)")


def test_the_typed_field_survives_its_own_redraw():
    """The focus trap: the picture redraws, the input the cursor is in does
    not. Same element object, same value, before and after."""
    ctx = _shell("air")
    view = _view(ctx)
    field = _numbers(view)["stall / approach speed"]

    field.value = 12.34

    again = _numbers(_view(ctx))["stall / approach speed"]
    assert again is field, "the field was rebuilt under the cursor"
    assert float(again.value) == 12.34


def test_a_mission_edit_redraws_the_diagram():
    """The diagram is drawn AT the operating point, and the operating point
    is asked for on the left of the same card — not inside it."""
    ctx = _shell("air")
    view = _view(ctx)
    speed = _numbers(view)["speed"]
    before = _cruise_curve(view)

    speed.value = float(speed.value) * 1.5

    # the cruise requirement carries q, so a faster mission is a different
    # T/W curve — the picture is not the same picture
    assert _cruise_curve(_view(ctx)) != before


def test_the_ceiling_above_the_expansion_moves_with_it():
    """The number that reaches the SOLVER is read out above the expansion.
    A requirement typed inside it that left that sentence alone would be the
    card disagreeing with itself about what refuses a design."""
    from gui.v3 import session

    ctx = _shell("air")
    view = _view(ctx)
    cap_before = session.mission_ws_ceiling(ctx.S)
    assert cap_before is not None, (
        "this mission must already state a ceiling, or there is nothing "
        "above the expansion for the typed requirement to move")

    field = _numbers(view)["stall / approach speed"]
    field.value = float(field.value) * 1.2   # still below the cruise speed

    cap_after = session.mission_ws_ceiling(ctx.S)
    assert cap_after > cap_before
    texts_after = [(getattr(e, "text", "") or "") for e in _view(ctx).descendants()]

    # THE SENTENCE ABOVE THE EXPANSION, by the number it quotes. The old form
    # asked only whether the new value was SOMEWHERE on the card — which the
    # "Allowed: W/S <= ..." line inside the expansion answers, and a
    # different renderer draws it — and then scanned for the old value only
    # among strings that were not on the card before, so a ceiling sentence
    # that never redrew was filtered out of its own staleness check. Deleting
    # _render_ws_cap() from _set_ws left both halves green.
    quoted = [t for t in texts_after if "caps W/S at" in t]
    assert len(quoted) == 1, (
        f"the ws-cap box must state the binding ceiling exactly once, "
        f"found {quoted!r}")
    said = float(re.search(r"caps W/S at ([0-9.]+) N/m²", quoted[0]).group(1))
    assert abs(said - cap_after) <= 0.5, (
        f"the sentence above the expansion still reads {said:.0f} N/m² "
        f"while the ceiling that reaches the solver is {cap_after:.0f} "
        f"(it was {cap_before:.0f})")
    assert abs(said - cap_before) > 0.5, "the old ceiling is still quoted"


def test_the_track_card_still_has_no_diagram_to_redraw():
    """A car's rear wing has no wing loading, so neither derived container is
    built — and the redraw hooks must be no-ops rather than clearing a pane
    that belongs to another medium."""
    ctx = _shell("track")
    view = _view(ctx)
    assert not _figures(view)
    # the mission edit path runs the same two renderers
    _numbers(view)["speed"].value = 33.0
    assert not _figures(_view(ctx))
