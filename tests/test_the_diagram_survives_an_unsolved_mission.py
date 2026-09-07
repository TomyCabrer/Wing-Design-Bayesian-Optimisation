"""A mission that does not solve must not take stage 1's view down with it.

Recorded twice in ``results/v3_render_errors.log`` on 2026-09-01, in a water
session::

    File "gui/v3/session.py", line 4299, in ws_diagram
        v_max_ms=float(point["v_ms"]),
    KeyError: 'v_ms'

``design_point`` answers ``{"error": ...}`` — that key and nothing else —
for a mission it cannot solve, and ``ws_diagram`` read the speed off it
anyway. Its ``except`` named ``ValueError`` and ``TypeError``, so the
KeyError went straight through ``_render_ws_cap`` and ``_render_ws_diagram``
into ``Ctx.render``, which replaced the WHOLE Operating point view with a
traceback panel: one blank field on stage 1 and the stage stops working.

The assertion is the OUTCOME the shell needs — an answer of "no diagram"
rather than an exception — for both readers, at the one state that produced
it: a mission with no speed in it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


@pytest.fixture(params=["water", "air"])
def unsolved(request):
    """A session whose design point is an error and nothing else."""
    from gui.v3 import session

    S = session.make_session(request.param)
    S["mission"]["V"] = None                    # the field the user blanked
    point = session.design_point(S)
    assert set(point) == {"error"}, point       # the premise, not the claim
    return S


def test_the_constraint_diagram_answers_none_instead_of_raising(unsolved):
    from gui.v3 import session

    assert session.ws_diagram(unsolved) is None


def test_the_ws_ceiling_answers_none_instead_of_raising(unsolved):
    """The ceiling reads the diagram, so it is the second way in."""
    from gui.v3 import session

    assert session.mission_ws_ceiling(unsolved) is None
