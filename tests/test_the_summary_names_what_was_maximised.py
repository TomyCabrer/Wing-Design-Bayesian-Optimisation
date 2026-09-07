"""The headline number says what it is, and the L/D is on the page too.

Free the span or the area and the optimiser stops maximising L/D: the score
becomes payload L/D, ``W_fixed/D``. The summary printed it as an unnamed
"objective", or worse put the aero ``LoD`` in the biggest type on the page —
a number no optimiser in this repo maximised, 17.0 to 34.7 L/D points away
from the one that was.

Both numbers are real and the user wants both. What they may never do is
stand in for each other, so the headline carries the family's OWN name for
what it maximised and the aero L/D gets its own readout beside it.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _summary_readouts(problem: str) -> list:
    """(label, value) for every readout the summary view actually renders."""
    import gui.v3.widgets as widgets
    from gui.v3.app import assemble
    from aerobo import api

    seen: list = []
    real = widgets.readout

    def spy(label, value, unit=None, **kw):
        seen.append((str(label), str(value)))
        return real(label, value, unit, **kw)

    widgets.readout = spy
    try:
        ctx = assemble()
        ctx.act("accept_mission")
        r = api.run(api.RunConfig(problem_name=problem, optimiser="sobol",
                                  budget=16, seed=0))
        ctx.S["wing"]["problem"] = problem
        ctx.S["run"]["record"] = r.to_dict()
        ctx.render("results", "summary")
    finally:
        widgets.readout = real
    return seen


def test_a_sized_run_names_its_objective_and_shows_the_lod():
    rows = _summary_readouts("trim wing + free planform")
    labels = [l for l, _ in rows]
    assert "payload L/D" in labels, labels[-12:]
    assert "L/D" in labels, labels[-12:]
    assert "objective" not in labels          # never the unnamed number
    payload = float(dict(rows)["payload L/D"])
    lod = float(dict(rows)["L/D"])
    # they are genuinely different numbers, so this is not one value twice
    assert payload < lod


def test_an_unsized_run_shows_one_lod_and_no_payload_row():
    """The control: without the size block the two ARE one number, and the
    page must not grow a second copy of it."""
    rows = _summary_readouts("trim wing")
    labels = [l for l, _ in rows]
    assert "L/D" in labels
    assert "payload L/D" not in labels
    assert labels.count("L/D") == 1
