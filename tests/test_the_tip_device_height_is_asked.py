"""The device's height is asked ONCE — in the design box, and nowhere else.

Stage 3 briefly asked it twice: as a switch under the tip-device menu ("state
it myself", with a number beside it) and as ``winglet_h_frac`` in the design
box, where every other searched-or-stated row is asked. Both wrote the same
pin, so nothing on screen could disagree — and it was still two homes for one
question, which is the rule this shell is built on. The switch is gone; the
box is the place.

What survives is the part that is not a control:

* the height writes through the design box's OWN pin, so the table, the
  reproduce snippet and the run cannot disagree about it;
* a stated height under ``vlm.MIN_WINGLET_FRAC`` flies no device, and the
  CANT beside it is then not a weak row but identically a dead one (flat to
  3.6e-15 across its band). It is pinned with the height — a pin derived
  from a pin, like the taper — rather than left for the GP to fit noise on.

The height is a free lunch as this model prices it (``sizing.sized_state``
takes no winglet argument, so weight and root stress are BIT-IDENTICAL across
the band while the score climbs, and the row rides its ceiling in four of five
families that carry it). That is a note on the row, not a second control.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aerobo.vlm import MIN_WINGLET_FRAC                          # noqa: E402


def _session(problem: str) -> dict:
    from gui.v3 import session as ses
    S = ses.make_session()
    S["wing"]["problem"] = problem
    return S


_DEVICE = "tail + winglet (span-capped) [designed tail + tip device]"


def _rendered_labels(problem: str, device: bool = True) -> list:
    """The text the wing stage actually PUT ON SCREEN for this family.

    Recorded by instrumenting the widget factories for the duration of one
    render, rather than grepping the module: a gutted function keeps its
    name, and a control that is defined but never called would pass a grep
    and fail a user. Scoped to one render because V3 tests share a client.
    """
    from nicegui import ui
    from gui.v3.app import assemble

    seen: list = []

    def _spy(name):
        real = getattr(ui, name)

        def wrapped(*a, **kw):
            for v in list(a) + list(kw.values()):
                if isinstance(v, str) and v:
                    seen.append(v.lower())
            return real(*a, **kw)
        return real, wrapped

    originals = {}
    for name in ("switch", "number", "label"):
        real, wrapped = _spy(name)
        originals[name] = real
        setattr(ui, name, wrapped)
    try:
        ctx = assemble()
        ctx.act("accept_mission")
        # the CHOICE is what gates the control, not the problem name: the
        # question only exists where the family flies a device
        if device:
            ctx.S["wing"]["choices"]["winglets"] = "canted"
        ctx.S["wing"]["problem"] = problem
        ctx.render("wing", "type")
    finally:
        for name, real in originals.items():
            setattr(ui, name, real)
    return seen


# --------------------------------------- the question has exactly one home

def test_the_type_view_does_not_ask_how_tall():
    """RENDERED, not grepped: the tip-device card offers the SHAPE and the
    chord rule, and no height question of its own."""
    seen = _rendered_labels(_DEVICE)
    assert any("tip device" in t for t in seen), sorted(set(seen))[:40]
    assert not any("state it myself" in t for t in seen)
    assert not any("how tall" in t for t in seen)


def test_a_family_with_no_device_is_not_asked_either():
    seen = _rendered_labels("trim wing", device=False)
    assert not any("state it myself" in t for t in seen)
    assert not any("how tall" in t for t in seen)


def test_the_design_box_is_where_the_height_is_asked():
    """The one home: the row is in the box the stage draws, in both states."""
    from gui.v3 import config

    S = _session(_DEVICE)
    S["wing"]["choices"]["winglets"] = "canted"
    assert "winglet_h_frac" in config.default_bounds(S)
    S["wing"]["fixed"] = {"winglet_h_frac": 0.08}
    assert config.fixed_rows(S)["winglet_h_frac"] == pytest.approx(0.08)


# ------------------------------------------- a dead cant is pinned with it

def test_a_height_below_the_cliff_pins_the_cant_with_it():
    """Below ``MIN_WINGLET_FRAC`` no device is built, so the cant is a dead
    dimension. It is derived, not asked."""
    from gui.v3 import config

    S = _session(_DEVICE)
    S["wing"]["fixed"] = {"winglet_h_frac": 0.004}
    rows = config.fixed_rows(S)
    assert rows["winglet_h_frac"] == 0.0          # what will FLY, recorded
    assert "winglet_cant_deg" in rows             # ...and the dead row with it
    band = config.default_bounds(S)["winglet_cant_deg"]
    assert min(band) <= rows["winglet_cant_deg"] <= max(band)


def test_a_height_that_flies_leaves_the_cant_alone():
    """The control. A live device's cant is a real question and must stay in
    the search — pinning it would answer something nobody asked."""
    from gui.v3 import config

    S = _session(_DEVICE)
    S["wing"]["fixed"] = {"winglet_h_frac": 0.08}
    rows = config.fixed_rows(S)
    assert rows["winglet_h_frac"] == pytest.approx(0.08)
    assert "winglet_cant_deg" not in rows


def test_a_cant_the_user_stated_is_not_overwritten():
    """A derived pin may not overrule a typed one."""
    from gui.v3 import config

    S = _session(_DEVICE)
    S["wing"]["fixed"] = {"winglet_h_frac": 0.004,
                          "winglet_cant_deg": 71.5}
    rows = config.fixed_rows(S)
    assert rows["winglet_cant_deg"] == pytest.approx(71.5)


def test_the_cliff_is_the_published_constant():
    """A tripwire: if the drop rule is ever retuned, the derived pin has to
    be re-read rather than the number quietly re-pinned here."""
    from gui.v3 import config

    S = _session(_DEVICE)
    just_under = MIN_WINGLET_FRAC * (1.0 - 1e-9)
    S["wing"]["fixed"] = {"winglet_h_frac": just_under}
    assert "winglet_cant_deg" in config.fixed_rows(S)

    S["wing"]["fixed"] = {"winglet_h_frac": MIN_WINGLET_FRAC}
    assert "winglet_cant_deg" not in config.fixed_rows(S)


def test_the_derived_pin_reaches_the_run_config():
    """The whole point of writing through the design box: what the shell
    derived is what the run pins, with no second vocabulary in between."""
    from gui.v3 import config

    S = _session(_DEVICE)
    S["wing"]["fixed"] = {"winglet_h_frac": 0.004}
    pinned = config.fixed_rows(S)
    from aerobo import api
    labels = list(api.PROBLEM_SPECS[_DEVICE].param_labels)
    for row in ("winglet_h_frac", "winglet_cant_deg"):
        assert row in labels, row
        assert row in pinned, row


def test_a_family_with_no_device_pins_nothing():
    """Empty in, empty out — a family without the rows must be untouched."""
    from gui.v3 import config

    S = _session("trim wing")
    S["wing"]["fixed"] = {"taper": 0.5}
    rows = config.fixed_rows(S)
    assert "winglet_cant_deg" not in rows
    assert rows["taper"] == pytest.approx(0.5)
