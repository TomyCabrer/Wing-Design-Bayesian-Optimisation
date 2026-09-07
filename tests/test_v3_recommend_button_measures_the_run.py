"""The recommendation measures the search that will actually run.

The button was building the problem by hand — ``PROBLEM_SPECS[name].build(...)``
— and so skipped the three things ``api.box_refusal_probe`` had always done:

* ``sanitise_flags``;
* the WING OBJECTIVE. A composite run with no frozen normalisation band cannot
  be built at all, so the bare build RAISED, the caller's ``except`` logged one
  grey line into a 150-pixel output strip and returned BEFORE ``_render_box()``.
  Not one pixel moved. That is the whole of "the recommended button doesn't
  work, it doesn't change anything";
* the PINS. ``_pinned_built`` rewrites ``param_labels`` to the free rows, so a
  FIXED row was being measured as free and proposed as a band — and pressing
  "Take these bands" then wrote a band for a variable the run never searches,
  which the config layer widens around the pin anyway. Two clicks, nothing
  changes.

All three now live behind ``api.recommend_box``, next to the sibling probe that
had them right.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aerobo import api                                   # noqa: E402


def _shell():
    from gui.v3.app import assemble

    ctx = assemble("air")
    ctx.act("accept_mission")
    ctx.render("wing", "box")
    return ctx


def _texts(ctx):
    return [getattr(e, "text", None) or ""
            for e in ctx.views[("wing", "box")].descendants()]


def test_the_composite_no_longer_makes_the_measurement_inert():
    from gui.v3 import config

    ctx = _shell()
    ctx.act("set_wing_objective", "composite")
    ctx.render("wing", "box")
    before = _texts(ctx)

    # the build the shell used to do by hand really does refuse this config
    cfg = config.build_cfg(ctx.S)
    try:
        api.PROBLEM_SPECS[cfg.problem_name].build(
            cfg.mission_kwargs, cfg.flags, cfg.bounds_overrides)
    except ValueError as exc:
        assert "band" in str(exc)
    else:                                          # pragma: no cover
        raise AssertionError("the composite built without a band")

    ctx.act("auto_recommend")
    ctx.render("wing", "box")
    after = _texts(ctx)
    assert after != before                         # the card MOVED
    a = ctx.S["wing"]["auto_rec"]
    assert a.get("stripped") is True
    assert a.get("rows"), "the measurement produced nothing"
    assert any("own L/D" in t for t in after)      # and says what it ranked on


def test_a_fixed_row_is_not_measured_and_not_proposed():
    ctx = _shell()
    ctx.act("set_row_fixed", "twist_tip_deg", True)
    ctx.act("set_fixed_value", "twist_tip_deg", -1.0)
    ctx.act("recommend_box", "wing")

    got = ctx.S["wing"]["recommend"]["wing"]
    assert "twist_tip_deg" not in got["labels"]
    assert "twist_tip_deg" not in got["rows"]


def test_a_measurement_that_cannot_be_taken_says_so_on_the_card(
        monkeypatch):
    """A measurement that fails is an answer about this configuration and
    belongs on the card. It used to be a `ctx.log` line and a `return` placed
    BEFORE the repaint, so the card was byte-identical afterwards — and the
    box is left exactly as the family publishes it, not half-narrowed."""
    from gui.v3 import config

    ctx = _shell()

    def _boom(*_a, **_kw):
        raise ValueError("the solver said no")

    monkeypatch.setattr(api, "recommend_box", _boom)
    before = _texts(ctx)
    box_before = {k: list(v[0])
                  for k, v in config.effective_bounds(ctx.S).items()}
    ctx.act("auto_recommend")
    ctx.render("wing", "box")

    a = ctx.S["wing"]["auto_rec"]
    assert "the solver said no" in (a.get("error") or "")
    after = _texts(ctx)
    assert after != before
    assert any("could not be measured" in t for t in after)
    assert {k: list(v[0])
            for k, v in config.effective_bounds(ctx.S).items()} == box_before


def test_the_api_owns_the_measurement_for_scripts_too():
    """``api.recommend_box`` is the entry point, so a study and the shell
    measure the same box."""
    from gui.v3 import config

    ctx = _shell()
    cfg = config.build_cfg(ctx.S)
    got = api.recommend_box(cfg, n=32)
    assert set(got["rows"]) <= set(got["labels"])
    assert got["n"] == 32
    assert got["objective_stripped"] is False


# ------------------------------- the second surface's span is recommended too

def _tail_shell():
    from gui.v3.app import assemble

    ctx = assemble("air")
    ctx.act("accept_mission")
    ctx.act("set_choice", "tail", True)
    ctx.act("set_choice", "tail_design", "planform")
    ctx.render("wing", "box")
    return ctx


def test_the_second_surface_s_span_is_recommended_from_the_designs():
    """It could never appear in ``rows``: the solver searches that surface as
    an area and an aspect ratio, and the span is a FLAG PAIR narrowed through
    ``tail.TailLimits``. So it is measured on the POINTS the draws produced —
    ``recommend_box`` returns them now — as sqrt(AR_t x S_t), which is the
    span those designs are actually drawn at."""
    import numpy as np

    ctx = _tail_shell()
    ctx.act("recommend_box", "aft")
    got = ctx.S["wing"]["recommend"]["aft"]

    assert got["best_x"], got
    assert len(got["best_x"][0]) == len(got["labels"])
    band = got.get("tail_span")
    assert band and band[0] < band[1]

    # every design the band was measured on lies inside it
    labs = list(got["labels"])
    spans = [float((x[labs.index("AR_t")] * x[labs.index("S_t_m2")]) ** 0.5)
             for x in got["best_x"]]
    assert band[0] <= min(spans) and max(spans) <= band[1]

    # ...and it is NARROWER than the corner band the two rows alone imply,
    # which is the whole reason it is measured on the designs
    from gui.v3 import config

    eff = config.effective_bounds(ctx.S)
    corner = ((eff["AR_t"][0][0] * eff["S_t_m2"][0][0]) ** 0.5,
              (eff["AR_t"][0][1] * eff["S_t_m2"][0][1]) ** 0.5)
    assert (band[1] - band[0]) < (corner[1] - corner[0])
    assert np.isfinite(band).all()


def test_taking_it_writes_the_flag_pair_the_row_reads():
    ctx = _tail_shell()
    ctx.act("recommend_box", "aft")
    band = ctx.S["wing"]["recommend"]["aft"].get("tail_span")
    assert band

    assert ctx.S["wing"]["flags"].get("tail_span_min_m") is None
    ctx.act("take_recommendation", "aft")
    assert ctx.S["wing"]["flags"]["tail_span_min_m"] == band[0]
    assert ctx.S["wing"]["flags"]["tail_span_max_m"] == band[1]

    # ...and the run really carries it
    from gui.v3 import config

    flags = config.cfg_dict(ctx.S)["flags"]
    assert flags["tail_span_min_m"] == band[0]
    assert flags["tail_span_max_m"] == band[1]


def test_the_span_is_offered_under_the_aft_table_only():
    ctx = _tail_shell()
    ctx.act("recommend_box", "wing")
    assert "tail_span" not in ctx.S["wing"]["recommend"]["wing"]


def test_the_card_names_the_span_beside_the_rows():
    """It is QUOTED, not written into the row.

    b_t = sqrt(AR_t x S_t) is a diagonal across two rows the box already
    bounds, so imposing it can only cut their corners: measured on the
    reference tail it took the admissible fraction 0.727 to 0.531 and left the
    best design identical to fifteen digits. So the number is on the card
    beside the row it describes, and the row stays the user's to set.
    """
    ctx = _tail_shell()
    ctx.act("auto_recommend")
    ctx.render("wing", "box")
    band = ctx.S["wing"]["auto_rec"]["tail_span"]
    assert band and band[1] > band[0]
    texts = " ".join(getattr(e, "text", None) or ""
                     for e in ctx.views[("wing", "box")].descendants())
    assert f"{band[0]:.4g} \u2013 {band[1]:.4g} m across" in texts
    # ...and the row it describes was NOT set
    assert ctx.S["wing"]["flags"].get("tail_span_min_m") is None
