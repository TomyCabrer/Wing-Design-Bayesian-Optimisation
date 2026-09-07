"""A fallback may leave a row BEHIND. It may not leave a row that cannot exist.

Three clicks took stage 3's whole Design box down:

    planform: free span + area  ·  objective: composite  ·  chord law: elliptic

    ValueError: chord law 'elliptic' of order 3 exceeds the largest order it
                offers (1)
      gui/v3/stages/wing.py  _chord_law_panel -> _chord_reach
      src/aerobo/geometry.py chord_reach

Two correct behaviours meeting badly:

* the COMPOSITE wing objective refuses to build until its normalisation band
  has been measured over the box (``api.wing_score_reference``) — a live
  min-max would make J non-stationary, which is a real rule and not a bug;
* ``config._flag_moved_rows`` builds the problem to read the rows a FLAG
  moves, and a build that fails leaves the static box, because a read-out one
  step behind tells the user more than a design box that vanishes.

The second is right about bands and wrong about EXISTENCE. Every law but the
elliptic blend carries its order; the blend is one number whatever the
registered twin declares (``geometry.CHORD_LAW_ORDERS``, clamped for the run
in ``api.chord_order_for``). So the fallback kept ``chord_k1..k3`` under a
one-parameter law, and the panel that draws the law fed three rows to
``geometry.chord_reach``, which refuses them — a design box that did vanish,
and took its click handler with it (``_set_chord_law`` calls ``_render_box``
directly, outside ``Ctx.render``'s containment).

The fallback now answers the existence question off the law itself.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _chord_rows(S) -> list:
    from gui.v3 import config

    return sorted(k for k in config.default_bounds(S) if k.startswith("chord_"))


# ------------------------------------------------------- the reported crash

def test_the_three_clicks_that_took_the_design_box_down():
    """The repro, driven through the shell's own actions."""
    from gui.v3.app import assemble

    ctx = assemble("air")
    ctx.act("accept_mission")
    ctx.act("set_planform", "free")
    assert _chord_rows(ctx.S) == ["chord_k1", "chord_k2", "chord_k3"]

    ctx.act("set_wing_objective", "composite")     # the build now refuses
    ctx.act("set_chord_law", "elliptic")           # ...and this raised

    assert _chord_rows(ctx.S) == ["chord_k1"]
    ctx.render("wing")
    view = ctx.views[("wing", "box")]
    texts = [getattr(e, "text", "") or "" for e in view.descendants()]
    assert not any("failed to render" in t for t in texts)


def test_the_same_law_switch_with_a_buildable_objective():
    """The control: with the build working the rows come off the BUILT
    problem, and they always did. The fix may not change this path."""
    from gui.v3.app import assemble

    ctx = assemble("air")
    ctx.act("accept_mission")
    ctx.act("set_planform", "free")
    ctx.act("set_chord_law", "elliptic")
    assert _chord_rows(ctx.S) == ["chord_k1"]


def test_a_failed_build_still_falls_back_for_the_bands():
    """What the guard is FOR is untouched: a row whose band a flag moves
    still quotes the published band rather than disappearing."""
    from gui.v3 import config
    from gui.v3.app import assemble

    ctx = assemble("air")
    ctx.act("accept_mission")
    ctx.act("set_planform", "free")
    ctx.act("set_wing_objective", "composite")
    rows = config.default_bounds(ctx.S)
    for row in ("b_m", "S_m2", "taper"):
        assert row in rows, row
        assert rows[row][1] > rows[row][0], row


# ------------------------------------------------- the rule, on its own terms

def test_only_the_rows_the_law_cannot_carry_are_dropped():
    """Per BLOCK: a pair carries one law per wing and a designed tail its
    own, so the count is per name family and not over the whole box."""
    from gui.v3 import config, session

    S = session.make_session()
    S["wing"]["flags"] = {}
    chord = ["chord_k1", "chord_k2", "chord_k3",
             "chord_front_k1", "chord_front_k2",
             "chord_rear_k1", "chord_rear_k2",
             "chord_k1_t", "chord_k2_t"]

    # the published law carries its order, so nothing is dropped
    assert config._chord_rows_the_law_carries(S, chord) == {}

    from aerobo import api
    S["wing"]["flags"] = {api.CHORD_LAW_KEY: "elliptic"}
    dropped = config._chord_rows_the_law_carries(S, chord)
    assert set(dropped) == {"chord_k2", "chord_k3", "chord_front_k2",
                            "chord_rear_k2", "chord_k2_t"}
    assert set(dropped.values()) == {None}       # None = "this row does not exist"


def test_the_order_comes_from_the_published_table():
    """A tripwire: if a law's parameter count is ever retuned, this fallback
    has to follow it rather than keep a number pinned here."""
    from aerobo import geometry

    assert max(geometry.CHORD_LAW_ORDERS["elliptic"]) == 1
    for law in ("poly", "ends", "kinked"):
        assert max(geometry.CHORD_LAW_ORDERS[law]) > 1, law
