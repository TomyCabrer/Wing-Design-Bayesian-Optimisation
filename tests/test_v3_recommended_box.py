"""The design box can be RECOMMENDED for the mission, per surface.

THE OFFER IS NO LONGER ON SCREEN. The box measures itself for the mission and
applies the result (``test_v3_the_box_opens_on_the_measurement.py``), because a
measurement behind a button is a measurement most sessions never see. What the
two actions here still are is the per-TABLE path: the machinery that filters one
measurement into the rows one surface’s table owns, and that a test (or a
future control) drives one surface at a time. A recommendation is not a limit.

What the shell adds on top of ``aerobo.recommend`` is the part that makes the
measurement readable — each table proposes only its OWN rows, only where the
measurement actually narrows one, and it quotes what taking it is worth as a
number measured on the box THAT TABLE would produce.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _open(**choices):
    from gui.v3.app import assemble

    ctx = assemble("air")
    ctx.act("accept_mission")
    for key, value in choices.items():
        ctx.act("set_choice", key, value)
    ctx.render("wing", "box")
    return ctx


def _texts(view):
    return [getattr(e, "text", None) or "" for e in view.descendants()]


def _buttons(view):
    return [getattr(e, "text", None) or "" for e in view.descendants()
            if type(e).__name__ == "Button"]


def test_no_table_carries_a_recommendation_button(capsys):
    """The offer used to be one button per table. Both are gone: the only
    control left on this card is the one that opts OUT of the measurement."""
    for ctx in (_open(tail=True), _open()):
        buttons = _buttons(ctx.views[("wing", "box")])
        assert not [b for b in buttons if "Recommend for this mission" in b]
        assert not [b for b in buttons if "Take these bands" in b]
        assert [b for b in buttons if "Reset to the solver" in b]
    assert "Traceback" not in capsys.readouterr().err


def test_measuring_one_does_not_change_the_box(capsys):
    """Offered, never applied."""
    from gui.v3 import config

    ctx = _open(tail=True)
    before = {k: list(v[0]) for k, v in config.effective_bounds(ctx.S).items()}
    ctx.act("recommend_box", "aft")
    after = {k: list(v[0]) for k, v in config.effective_bounds(ctx.S).items()}
    assert after == before
    # ...and nothing it wrote is a MEASURED band: the rows that are here are
    # the shell's own mission bands, which were here before it ran
    assert "recommended" not in ctx.S["wing"]["bounds_source"].values()
    # ...and it produced something a caller could take
    assert ctx.S["wing"]["recommend"]["aft"]["rows"]
    assert "Traceback" not in capsys.readouterr().err


def test_taking_it_writes_the_bands_and_says_they_were_measured(capsys):
    """Through ``W['bounds']`` — the same place a typed bound lands — and
    tagged ``"recommended"``, not left untagged as a typed one.

    It used to POP the row's source, which made a measured band
    indistinguishable from a typed one, and a typed row is never refreshed by
    the mission. So taking the recommendation for the reference wing and then
    stating a 1 m one left the box searching the OLD wing for the rest of the
    session (see ``test_a_taken_band_does_not_outlive_its_mission``)."""
    from gui.v3 import config

    ctx = _open(tail=True)
    ctx.act("recommend_box", "aft")
    proposed = dict(ctx.S["wing"]["recommend"]["aft"]["rows"])
    assert proposed, "expected the reference tail to narrow at least one row"
    ctx.act("take_recommendation", "aft")
    eff = config.effective_bounds(ctx.S)
    for label, band in proposed.items():
        assert [round(v, 9) for v in eff[label][0]] == \
            [round(v, 9) for v in band], label
        # the band the RUN searches, and it says where it came from
        assert eff[label][1] == "recommended", label
    # taken, so the offer is gone
    assert "aft" not in (ctx.S["wing"].get("recommend") or {})
    assert "Traceback" not in capsys.readouterr().err


def test_a_taken_band_does_not_outlive_its_mission(capsys):
    """The reported defect, in one function.

    Take the recommendation for the shipped wing, then state a 1 m one in the
    mission. Before the fix the box still read 8.95 – 14.79 m — the mission
    said 1 m and the design box said the wing from five minutes ago, with
    nothing on screen saying why, and pressing the button again could only
    answer "no region" because it re-measured a box the mission had emptied.
    """
    from gui.v3 import config, session

    ctx = _open()
    ctx.act("set_planform", "free")
    ctx.act("recommend_box", "wing")
    assert (ctx.S["wing"]["recommend"]["wing"].get("rows") or {}).get("b_m")
    ctx.act("take_recommendation", "wing")
    took = list(ctx.S["wing"]["bounds"]["b_m"])
    assert took[1] > 5.0                      # the reference wing's own band

    # ...and now the user says the wing is 1 m across
    assert session.set_size_from_span_ar(ctx.S, 1.0, 8.0)
    session.sync_wing_from_mission(ctx.S)
    assert session.nominal_span(ctx.S) == 1.0

    band = config.effective_bounds(ctx.S)["b_m"]
    assert band[1] == "mission", band          # the shell's, not frozen as a user row
    assert band[0] != took
    assert band[0][1] <= 4.0, band             # a band about a 1 m wing
    assert band[0][0] <= 1.0 <= band[0][1], band
    assert "Traceback" not in capsys.readouterr().err


def test_a_table_proposes_only_its_own_surface_s_rows(capsys):
    from gui.v3.stages.wing import AFT_ROW

    ctx = _open(tail=True)
    ctx.act("recommend_box", "aft")
    ctx.act("recommend_box", "wing")
    rec = ctx.S["wing"]["recommend"]
    assert rec["aft"]["rows"] and rec["wing"]["rows"]
    assert all(AFT_ROW.search(r) for r in rec["aft"]["rows"])
    assert not any(AFT_ROW.search(r) for r in rec["wing"]["rows"])
    assert "Traceback" not in capsys.readouterr().err


def test_a_row_the_measurement_barely_moves_is_not_proposed(capsys):
    """A finite sample bounds nearly every row a hair inside its own edges —
    the reference tail's arm came back 3–7.98 against a published 3–8. Listing
    those beside a row that halved buries the one finding the card had."""
    from gui.v3 import config
    from gui.v3.stages.wing import RECOMMEND_MIN_SHRINK

    ctx = _open(tail=True)
    ctx.act("recommend_box", "aft")
    got = ctx.S["wing"]["recommend"]["aft"]
    eff = config.effective_bounds(ctx.S)
    for label, band in got["rows"].items():
        now = eff[label][0]
        was = float(now[1]) - float(now[0])
        assert (band[1] - band[0]) / was <= 1.0 - RECOMMEND_MIN_SHRINK, label
    # the ones held back are counted, not hidden
    assert got["n_held"] >= 1
    assert "Traceback" not in capsys.readouterr().err


def test_what_it_quotes_is_the_box_that_table_would_produce(capsys):
    """Not the whole-vector figure. Under a table offering two of its rows
    that number quotes a box nobody is being offered, and it moves when the
    OTHER surface's rows move."""
    import numpy as np

    from aerobo import api, recommend
    from gui.v3 import config

    ctx = _open(tail=True)
    ctx.act("recommend_box", "aft")
    got = ctx.S["wing"]["recommend"]["aft"]
    cfg = config.build_cfg(ctx.S)
    built = api.PROBLEM_SPECS[cfg.problem_name].build(
        cfg.mission_kwargs, cfg.flags, cfg.bounds_overrides)
    cand = np.asarray(built.bounds, dtype=float).copy()
    labels = list(built.param_labels)
    for label, band in got["rows"].items():
        cand[labels.index(label)] = band
    frac, _best = recommend.probe_box(built, cand, got["n"], got["seed"])
    assert got["frac_after"] == frac
    assert "Traceback" not in capsys.readouterr().err


def test_a_recommendation_does_not_survive_the_problem_it_was_measured_over(
        capsys):
    """It quoted a mission and a box, and both move under it. A read-out that
    goes stale under the control that invalidates it is the trap this shell
    keeps falling into, and this one is expensive enough to be kept between
    renders — so it is stamped and the stamp is checked at draw time.
    """
    from gui.v3 import config

    ctx = _open(tail=True)
    ctx.act("recommend_box", "aft")
    assert ctx.S["wing"]["recommend"]["aft"]["rows"]

    # a typed bound changes the box the measurement was taken over
    ctx.act("set_bound", "S_t_m2", 1, 2.5)
    before = {k: list(v[0])
              for k, v in config.effective_bounds(ctx.S).items()}
    ctx.act("take_recommendation", "aft")
    after = {k: list(v[0])
             for k, v in config.effective_bounds(ctx.S).items()}
    # ...so TAKING it is refused: it was measured over a box that is gone
    assert after == before
    assert "Traceback" not in capsys.readouterr().err


def test_the_search_policy_does_not_throw_a_measurement_away(capsys):
    """The optimiser and the budget decide how the box is searched, not which
    designs fly this mission — so they must not invalidate a measurement that
    cost 128 evaluations."""
    from gui.v3 import config

    ctx = _open(tail=True)
    ctx.act("recommend_box", "aft")
    proposed = dict(ctx.S["wing"]["recommend"]["aft"]["rows"])
    assert proposed
    ctx.S["wing"]["budget"] = int(ctx.S["wing"]["budget"]) + 8
    ctx.act("take_recommendation", "aft")
    eff = config.effective_bounds(ctx.S)
    for label, band in proposed.items():
        assert [round(v, 9) for v in eff[label][0]] == \
            [round(v, 9) for v in band], label
    assert "Traceback" not in capsys.readouterr().err


def test_resetting_the_box_drops_it(capsys):
    ctx = _open(tail=True)
    ctx.act("recommend_box", "aft")
    assert ctx.S["wing"]["recommend"]["aft"]["rows"]
    ctx.act("reset_box")
    assert not (ctx.S["wing"].get("recommend") or {})
    assert "Traceback" not in capsys.readouterr().err


def test_it_says_a_narrower_box_is_not_a_better_answer(capsys):
    """Over 42 seeds this project measured no objective gain from narrowing a
    band, and the tightest arm was the worst of them. A card selling a
    narrowing as a gain would contradict its own study."""
    ctx = _open(tail=True)
    ctx.act("auto_recommend")
    texts = _texts(ctx.views[("wing", "box")])
    assert any("SMALLER SEARCH, not a better answer" in t for t in texts)
    assert any("the tightest arm of that study was the worst" in t
               for t in texts)
    assert "Traceback" not in capsys.readouterr().err
