"""Switching a limit ON may not write a pair the run refuses.

    minimum aspect ratio: on, typed 40  ·  maximum aspect ratio: on

    ValueError: the aspect-ratio limit must have ar_max > ar_min (got 40 and 40)
      gui/v3/stages/wing.py  _set_ar_limit_on -> _render_box -> _ar_limit_rows
      src/aerobo/api.py      ar_limits_of

``_ar_limit_default`` picked a starting value from the wing on screen (±20 %
of its aspect ratio) and clipped it into the solvers' band — and read neither
end of the limit already live. With the minimum at the solvers' ceiling both
±20 % clip to 40, so the maximum was switched on at 40 as well; the pair is
strict (``ar_max > ar_min``), so the very next read raised, out of the card
that draws the limit, and stage 3's whole Design box became a traceback.

The typed field never had this bug — ``_set_ar_limit`` validates through the
same reader and puts the old pair back — so this is the toggle catching up
with the field beside it. Where the live end leaves no room the switch does
not move at all, and says which end to move first: a limit that refuses every
design is not a limit.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _shell():
    from gui.v3.app import assemble

    ctx = assemble("air")
    ctx.act("accept_mission")
    return ctx


def _ar(ctx) -> dict:
    return {k: v for k, v in ctx.S["wing"]["flags"].items()
            if k in ("ar_min", "ar_max")}


def _box_is_drawn(ctx) -> bool:
    ctx.render("wing")
    view = ctx.views[("wing", "box")]
    return not any("failed to render" in (getattr(e, "text", "") or "")
                   for e in view.descendants())


# ------------------------------------------------------- the reported crash

def test_a_minimum_at_the_ceiling_does_not_take_the_box_down():
    """The repro: the two ends would have to be equal, so the switch refuses
    rather than storing a pair the reader rejects."""
    from aerobo import api

    ctx = _shell()
    ctx.act("set_ar_limit_on", "ar_min", True)
    ctx.act("set_ar_limit", "ar_min", 40.0)
    assert _ar(ctx) == {"ar_min": 40.0}

    ctx.act("set_ar_limit_on", "ar_max", True)     # this raised
    assert "ar_max" not in _ar(ctx)
    assert api.ar_limits_of(ctx.S["wing"]["flags"]) == (40.0, None)
    assert _box_is_drawn(ctx)


def test_a_maximum_at_the_floor_refuses_the_other_end_too():
    """The mirror image, which the same clip produced on a tiny wing."""
    from aerobo import api
    from aerobo.sizing import AR_LIMITS

    ctx = _shell()
    ctx.act("set_ar_limit_on", "ar_max", True)
    ctx.act("set_ar_limit", "ar_max", float(AR_LIMITS[0]))
    ctx.act("set_ar_limit_on", "ar_min", True)
    assert "ar_min" not in _ar(ctx)
    assert api.ar_limits_of(ctx.S["wing"]["flags"]) == (None, float(AR_LIMITS[0]))
    assert _box_is_drawn(ctx)


# ------------------------------------------------- and the ordinary pair works

def test_both_ends_switched_on_give_a_band_the_reader_accepts():
    """The control. This is the path everybody takes and it may not move."""
    from aerobo import api

    ctx = _shell()
    ctx.act("set_ar_limit_on", "ar_min", True)
    ctx.act("set_ar_limit_on", "ar_max", True)
    lo, hi = api.ar_limits_of(ctx.S["wing"]["flags"])
    assert lo is not None and hi is not None and hi > lo
    assert _box_is_drawn(ctx)


def test_the_second_end_is_placed_inside_the_room_the_first_leaves():
    """Not merely valid — placed with the published gap, so the band is one
    somebody could search."""
    from gui.v3.stages.wing import AR_LIMIT_GAP

    ctx = _shell()
    ctx.act("set_ar_limit_on", "ar_max", True)
    ctx.act("set_ar_limit", "ar_max", 5.0)
    ctx.act("set_ar_limit_on", "ar_min", True)
    got = _ar(ctx)
    assert got["ar_min"] <= 5.0 - AR_LIMIT_GAP + 1e-9, got


def test_switching_a_limit_off_and_on_again_is_still_free():
    """Turning one end off releases the room it was holding."""
    ctx = _shell()
    ctx.act("set_ar_limit_on", "ar_min", True)
    ctx.act("set_ar_limit", "ar_min", 40.0)
    ctx.act("set_ar_limit_on", "ar_min", False)
    ctx.act("set_ar_limit_on", "ar_max", True)
    assert "ar_max" in _ar(ctx) and "ar_min" not in _ar(ctx)
    assert _box_is_drawn(ctx)
