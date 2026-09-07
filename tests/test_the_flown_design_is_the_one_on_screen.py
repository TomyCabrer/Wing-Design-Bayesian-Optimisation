"""Stage 6 flies the design stage 4 is showing — including the SECOND one.

The complaint was "Flight only uses the first design; if it is re-done and
refreshed it still uses the first."

It was not the deck. Stage 5 stamps ``C["deck_report"]`` and rebuilds on a
new report, and stage 6's Fly view re-arms whenever the model object moves;
both of those worked. What did not was ``S["flight"]["live"]`` — the mass,
the CG and the thrust the aeroplane is ACTUALLY flown at. Those three
distinguish "a default derived from the design" from "a number the pilot
typed" by one sentinel, ``None``, and ``arm()`` spends that sentinel on the
first design of the session. Nothing ever put it back, and the shell carries
``S`` across every page build on purpose, so the first design's weight and
thrust became properties of the SESSION and survived both the re-run and the
refresh.

Measured on ``tail`` with the mission weight raised between the two runs:
66.57 kg and 21.60 N flown, against 142.76 kg and 72.15 N honest. The deck on
screen was right and the aeroplane behind it was not, which is the worst
version of this — nothing on screen disagreed with anything else.

The stamp is the REPORT and it is by identity. Not the flight model: stage 5
builds a new one on every edit under the same report, and moving an aileron
band must not throw away a mass the pilot typed.
"""
from __future__ import annotations

import numpy as np
import pytest

from aerobo import api
from gui.v4 import app as v4app


def _report(x=None, flags=None):
    cfg = api.RunConfig(problem_name="tail", budget=4, seed=0,
                        flags=flags or {})
    built = api.PROBLEM_SPECS["tail"].build({}, flags or {}, None)
    b = np.asarray(built.bounds, dtype=float)
    return api.design_report(cfg, b.mean(axis=1) if x is None else x)


def _other_design():
    built = api.PROBLEM_SPECS["tail"].build({}, {}, None)
    b = np.asarray(built.bounds, dtype=float)
    return _report(b[:, 0] + 0.15 * (b[:, 1] - b[:, 0]))


def _flying(report):
    ctx = v4app.assemble()
    ctx.S["run"]["record"] = {"pretend": "a completed run"}
    ctx.S["run"]["report"] = report
    ctx.render("flight", "fly")
    return ctx


def test_a_new_run_refills_the_levers_from_the_new_design():
    """The headline, and it is asserted against a FRESH session on the same
    report rather than against a remembered number: "what this design would
    be flown at" is exactly what a session that had never seen the first one
    produces."""
    r1, r2 = _report(), _other_design()
    ctx = _flying(r1)
    first = dict(ctx.S["flight"]["live"])

    ctx.S["run"]["report"] = r2
    ctx.render("flight", "fly")
    after = dict(ctx.S["flight"]["live"])

    honest = dict(_flying(r2).S["flight"]["live"])
    assert after != first, "the levers stayed on the first design"
    for k, v in honest.items():
        assert after[k] == pytest.approx(v), \
            f"{k}: flying {after[k]} where this design is {v}"


def test_a_lever_the_pilot_typed_survives_a_stage_5_edit():
    """The other half, and the reason the stamp is the report and not the
    flight model: a stage-5 edit makes a NEW model under the SAME report."""
    ctx = _flying(_report())
    ctx.act("flight_set_live", "mass_kg", 80.0)
    ctx.S["controls"]["aileron"]["chord_frac"] = 0.35
    ctx.act("controls_rebuild")
    ctx.render("flight", "fly")
    assert ctx.S["flight"]["live"]["mass_kg"] == pytest.approx(80.0)


def test_the_transient_between_two_runs_does_not_clear_the_levers():
    """``set_result`` clears the report and the new one lands from a worker,
    so ``None`` is a real state. Clearing on it would drop the pilot's own
    numbers and then refill them from the stale deck a moment later."""
    ctx = _flying(_report())
    ctx.act("flight_set_live", "mass_kg", 80.0)
    ctx.S["run"]["report"] = None
    ctx.act("flight_arm")
    assert ctx.S["flight"]["live"]["mass_kg"] == pytest.approx(80.0)


def test_the_other_three_tabs_re_arm_too():
    """The stage remembers which tab it was left on, and the re-arm test
    lived in the Fly view alone. Modes never even rebuilt the deck."""
    r1, r2 = _report(), _other_design()
    for tab in ("setup", "modes", "traces"):
        ctx = _flying(r1)
        ctx.S["run"]["report"] = r2
        ctx.render("flight", tab)
        fm = ctx.S["controls"]["fm"]
        assert ctx.S["flight"]["armed_fm"] is fm, \
            f"the {tab} tab describes the previous design"


def test_the_fin_scans_are_dropped_with_the_design():
    """``CONTROLS_DEFAULTS`` promises this in as many words — "every one of
    them is cleared whenever the design changes" — and nothing implemented
    it: ``rebuild`` threw away ``rec_cache`` and left the two scans."""
    ctx = _flying(_report())
    C = ctx.S["controls"]
    C["spiral_scan"] = {"pretend": "a scan of design #1"}
    C["dihedral_scan"] = {"pretend": "and its wing answer"}
    ctx.S["run"]["report"] = _other_design()
    ctx.act("controls_ensure_deck")
    assert C["spiral_scan"] is None
    assert C["dihedral_scan"] is None


def test_a_render_alone_does_NOT_drop_them():
    """...and only when the design moved. A scan thrown away on every repaint
    is the same defect pointing the other way."""
    ctx = _flying(_report())
    C = ctx.S["controls"]
    C["spiral_scan"] = {"pretend": "a scan"}
    ctx.act("controls_ensure_deck")
    ctx.render("controls", "derivatives")
    assert C["spiral_scan"] == {"pretend": "a scan"}


def test_two_meshes_in_one_second_get_two_urls():
    """``/_flight/{int(time.time())}.stl`` is one URL per WALL SECOND, and a
    duplicate static route resolves to the FIRST one registered — a literal
    "it draws the first design", on the one surface that is hardest to
    disbelieve."""
    import inspect

    from gui.v4.stages import flight

    src = inspect.getsource(flight)
    assert "int(time.time())}.stl" not in src, \
        "the mesh URL is minted from a one-second clock again"
    assert "uuid.uuid4().hex" in src
