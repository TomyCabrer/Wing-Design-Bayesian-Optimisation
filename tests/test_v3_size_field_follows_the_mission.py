"""The size field shows the size the mission currently states.

Stage 1 asks how big the surface is in ONE control — W/S, or the area, or a
span with an aspect ratio — and every one of those is a number DERIVED from
the mission around it. The weight is the numerator of W/S, so a weight
typed six inches to the left moves the loading the field is showing.

``touch`` redrew seven derived containers and not that field, and stale is
not merely wrong on screen. Measured on a fresh air session: type W = 1000 N
and the box still read 65.2802 under a hint saying "S = W/(W/S) = 10 m²"
that its own two visible numbers do not satisfy; then ONE step-arrow press
submitted 70.2802, ``widgets.is_echo`` compared it against the current 100
N/m², found no echo, and the loading was dropped to 70.2802 — growing the
reference area 10 -> 14.2288 m² from a single arrow key. Adopting the
mission's own W/S ceiling did the same thing more loudly: the area went
10 -> 39.8921 m² while the field went on reading 65.2802.

The pull in the other direction is the focus trap the size fields were split
into their own container for: a field rebuilt from inside its OWN on_change
swallows the rest of the number being typed. So the rule is not "always
repaint" but "repaint unless the edit came from this field" — ``touch``'s
``size_field`` flag, False only in the size setters themselves.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _shell(medium: str = "air"):
    from gui.v3.app import assemble

    ctx = assemble(medium)
    ctx.render("mission", "operating")
    return ctx


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


def _press(button):
    """Fire the button's own click handler — not what we think it does."""
    handler = next(iter(button._event_listeners.values())).handler
    return handler() if not inspect.signature(handler).parameters \
        else handler(None)


def _ceiling_button(view):
    """The conflict card's "Use ... N/m²", NOT the diagram's.

    Both cards can offer the same number (the ceiling usually IS the
    diagram's binding limit), so they are told apart by position: the
    conflict card sits above the "Where does this come from?" expansion, and
    the diagram's button is emitted after its figure.
    """
    from nicegui.elements.plotly import Plotly

    found = []
    for e in view.descendants():
        if isinstance(e, Plotly):
            break
        if type(e).__name__ == "Button" \
                and (getattr(e, "text", "") or "").startswith("Use "):
            found.append(e)
    assert found, "the conflict card offers no ceiling to adopt"
    return found[0]


def test_typing_the_weight_repaints_the_wing_loading_field(capsys):
    from gui.v3 import session, widgets

    ctx = _shell("air")
    before = _numbers(_view(ctx))["wing loading W/S"]
    assert before.value == pytest.approx(
        widgets.shown(session.wing_loading(ctx.S)))

    _numbers(_view(ctx))["design weight"].value = 1000.0

    field = _numbers(_view(ctx))["wing loading W/S"]
    assert session.wing_loading(ctx.S) == pytest.approx(100.0)
    assert field.value == pytest.approx(100.0), \
        "the field is still showing the loading of the old weight"
    # it is a repaint, so the element the user was NOT typing into is a new
    # one; the field the edit came from is untouched
    assert field is not before
    err = capsys.readouterr().err
    assert "Traceback" not in err, err


def test_one_step_arrow_after_a_weight_edit_does_not_resize_the_wing(capsys):
    """The failure the repaint exists to stop, driven end to end."""
    from gui.v3 import session

    ctx = _shell("air")
    _numbers(_view(ctx))["design weight"].value = 1000.0
    field = _numbers(_view(ctx))["wing loading W/S"]

    # one press of the field's own step arrow: it submits what it SHOWS plus
    # one step (5 N/m²), whatever that is
    field.value = float(field.value) + 5.0

    assert session.wing_loading(ctx.S) == pytest.approx(105.0)
    assert float(ctx.S["mission"]["s_ref_m2"]) == pytest.approx(
        1000.0 / 105.0)
    # the stale field submitted 70.2802 and grew the area to 14.2288 m²
    assert float(ctx.S["mission"]["s_ref_m2"]) < 11.0
    err = capsys.readouterr().err
    assert "Traceback" not in err, err


def test_adopting_the_ceiling_repaints_the_field(capsys):
    from gui.v3 import session

    ctx = _shell("air")
    _numbers(_view(ctx))["design weight"].value = 3000.0
    over = session.ws_over_ceiling(ctx.S)
    assert over is not None and over["over"] > 1.0

    _press(_ceiling_button(_view(ctx)))

    cap = float(over["cap"])
    assert session.wing_loading(ctx.S) == pytest.approx(cap)
    field = _numbers(_view(ctx))["wing loading W/S"]
    assert field.value == pytest.approx(cap, abs=5e-5), \
        "the adopted ceiling is not the number the field is showing"
    # ...and the area it resized to (10 -> 39.8921 m² measured) is the one
    # the run is built from, not just the one the card printed
    from gui.v3 import config
    flags = config.cfg_dict(ctx.S).get("flags") or {}
    assert flags.get("S_m2") == pytest.approx(3000.0 / cap)
    err = capsys.readouterr().err
    assert "Traceback" not in err, err


def test_a_size_field_is_never_rebuilt_from_its_own_keystroke(capsys):
    """The other half of the rule: the focus trap must stay closed.

    Both faces that hold two fields at once are checked, because a rebuild
    here does not fail loudly — it silently swallows the rest of a number
    the user is still typing.
    """
    from gui.v3 import session

    ctx = _shell("air")
    field = _numbers(_view(ctx))["wing loading W/S"]
    field.value = 80.0
    assert session.wing_loading(ctx.S) == pytest.approx(80.0)
    assert _numbers(_view(ctx))["wing loading W/S"] is field

    assert session.set_size_statement(ctx.S, session.SIZE_AS_SPAN_AR)
    ctx.render("mission", "operating")
    span = _numbers(_view(ctx))["span (sketch)"]
    ar = _numbers(_view(ctx))["aspect ratio (estimate)"]
    span.value = float(span.value) * 1.5
    now = _numbers(_view(ctx))
    assert now["span (sketch)"] is span
    assert now["aspect ratio (estimate)"] is ar
    ar.value = 14.0
    assert session.section_aspect_ratio(ctx.S) == pytest.approx(14.0)
    assert _numbers(_view(ctx))["aspect ratio (estimate)"] is ar
    err = capsys.readouterr().err
    assert "Traceback" not in err, err
