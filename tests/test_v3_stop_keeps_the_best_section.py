"""Stop keeps the best aerofoil, and a shape run always compares with its seed.

Two faults of the same shape on stage 2 — the run had the answer and threw it
away:

* STOP DISCARDED THE SEARCH. The button raised ``_Cancelled`` out of the
  progress callback, which unwound ``api.run`` before it could assemble
  anything. Every XFOIL sweep already paid for (minutes to an hour of solver
  time) went with it, including the best section found so far: the stage kept
  a convergence trace and no aerofoil, and there was nothing to adopt. It is
  a STOP RULE now (:func:`stages.airfoil.stop_or_converged`), so ``api.run``
  ends the search between evaluations and rebuilds the result from its own
  per-evaluation log — ``partial=True``, the incumbent as ``best_x``, and
  therefore a ``section`` block the stage scores and adopts like any other.
* A 2-D RUN HAD NOTHING TO COMPARE AGAINST. ``optimize_airfoil`` evaluated
  the seed only in wing mode, so a section (or composite) run came back with
  no ``baseline``: the result table compared the optimised section against
  nothing and the polar plot drew one curve, because the seed's sweep is not
  the run's own and the overlay is deliberately cache-only. The seed is
  evaluated in every mode now — one XFOIL sweep, which is what makes the rest
  of the card readable.

Both are checked at the level they live at: the rule as a pure function, the
report through the real ``api.optimize_airfoil``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aerobo import api                              # noqa: E402
from gui import nice_app as v1                      # noqa: E402
from gui.v3.stages import airfoil as stage          # noqa: E402

#: a cheap seeded section run — the anchor is a named NACA so the seed is a
#: real, evaluable section and the polars are the shared cache's
_RUN = dict(anchor="2412", optimiser="sobol", seed=0)


# --------------------------------------------------------------- the rule
def test_the_button_alone_is_a_stop_rule():
    """Without the adaptive stop there was no rule at all, so the button had
    nothing to fire. It is passed on EVERY run now."""
    pressed = {"yes": False}
    rule = stage.stop_or_converged(None, lambda: pressed["yes"])
    assert rule(1, -1.0) is False
    assert rule(2, -0.9) is False
    pressed["yes"] = True
    assert rule(3, -0.8) is True


def test_it_does_not_swallow_the_convergence_rule():
    """The two stops are ORed, not replaced: a plateau still ends a run
    nobody pressed anything on."""
    seen = []

    def converged(i, best):
        seen.append((i, best))
        return i >= 3

    rule = stage.stop_or_converged(converged, lambda: False)
    assert rule(1, -1.0) is False
    assert rule(3, -0.5) is True
    assert seen == [(1, -1.0), (3, -0.5)]


def test_the_button_wins_before_the_rule_is_even_asked():
    """A pressed button must not depend on what a plateau detector thinks."""
    def converged(i, best):
        raise AssertionError("the convergence rule was consulted")

    rule = stage.stop_or_converged(converged, lambda: True)
    assert rule(1, -1.0) is True


def test_the_worker_no_longer_cancels_out_of_the_progress_callback():
    """The regression this file exists for, as a rule about the source: the
    section worker's progress callback RECORDS, and stopping is a rule."""
    src = (_REPO_ROOT / "gui/v3/stages/airfoil.py").read_text()
    worker = src.split("def _opt_worker(", 1)[1].split("\n    # =====", 1)[0]
    assert "_Cancelled" not in worker
    assert "stop_or_converged(" in worker


# ------------------------------------------------- what a stopped run keeps
def test_a_stopped_section_run_keeps_the_best_aerofoil():
    """The point of the change: an interrupted run still hands stage 3 a
    section it can fly."""
    fired = {"n": 0}

    def stop_rule(i, best):
        fired["n"] = int(i)
        return int(i) >= 3          # the "button", pressed at evaluation 3

    rep = api.optimize_airfoil(budget=12, stop_rule=stop_rule, **_RUN)
    res = rep["result"]
    assert res["partial"] is True
    assert res["stop_reason"]
    assert 0 < res["n_evals"] < 12
    assert res["best_x"] and res["best_score"] is not None

    # ...and the aerofoil itself, which is what "keep the best one" means
    sec = rep["section"]
    assert sec["design"]["coords"]
    # ADOPTABLE: a designed section travels as its CST weights and nothing
    # else, so a report the stage cannot read weights out of is a report it
    # has to refuse (stages.airfoil.optimised_weights)
    weights = stage.optimised_weights(rep)
    assert weights is not None
    w_u, w_l = weights
    assert len(w_u) == len(w_l) == len(res["best_x"]) // 2


def test_even_the_earliest_stop_keeps_what_it_paid_for():
    """Pressed on the first evaluation, the run keeps that one section — the
    rule is applied AFTER an evaluation, so the sweep is already bought."""
    rep = api.optimize_airfoil(budget=8, stop_rule=lambda i, best: True,
                               **_RUN)
    res = rep["result"]
    assert res["partial"] is True and res["n_evals"] == 1
    assert res["best_x"] and rep["section"]["design"]["coords"]


def test_a_stopped_run_with_no_feasible_design_carries_no_section():
    """...and when nothing flew there is nothing to keep, which the stage
    reports rather than adopting an empty report (the poll's own branch)."""
    cfg = api.airfoil_run_config(budget=8, seed=0)
    res = api.partial_result(cfg, [])          # stopped before any evaluation
    assert res.partial is True and res.best_x is None


# --------------------------------------------------- the seed to compare to
def test_a_two_d_run_comes_back_with_its_seed():
    rep = api.optimize_airfoil(budget=4, **_RUN)
    base = rep.get("baseline") or {}
    assert base.get("breakdown"), "a 2-D run returned no seed to compare with"
    # the SAME point, not the screening table's: both sides are the run's own
    assert base["breakdown"]["cl_design"] == rep["conditions"]["cl_design"]
    # ...and the seed's polar is now on disk, so the section view's
    # cache-only overlay has curves to draw
    assert (rep["section"].get("baseline") or {}).get("polar")


def test_the_card_can_draw_a_real_comparison_from_a_two_d_run():
    """Rows AND a two-curve polar — the compare the user asked for, in the
    mode that had neither."""
    rep = api.optimize_airfoil(budget=4, **_RUN)
    rows = {r["metric"] for r in v1.airfoil_compare_rows(rep)}
    assert rows, "no comparison rows at all"
    assert any("c_d at design c_l" in m for m in rows)
    assert any("c_m" in m for m in rows)
    assert any("t/c" in m for m in rows)

    fig = v1.fig_section_polars(rep["section"])
    names = {t.name for t in fig.data}
    assert "optimised" in names and "anchor" in names


# ------------------------------------------------------------- the stage
def test_the_stage_keeps_scores_and_adopts_the_stopped_run(monkeypatch):
    """End to end on the shell: Stop mid-search leaves stage 3 holding the
    best section the run reached, not the library pick and not nothing."""
    import time

    from gui.v3 import session
    from gui.v3.app import assemble

    # a real partial report, produced once (its sweeps are the shared cache's)
    stopped = api.optimize_airfoil(budget=6, stop_rule=lambda i, b: i >= 2,
                                   **_RUN)
    assert stopped["result"]["partial"] is True

    ctx = assemble("air")
    S = ctx.S
    A = session.airfoil_state(S, "main")
    S["ui"]["tab"]["airfoil"] = "optimise"
    seen = {}

    def fake_optimize(**kw):
        rule, prog = kw["stop_rule"], kw["progress_cb"]
        prog(1, -0.01, f=-0.01, x=[0.1] * 8)
        seen["before"] = rule(1, -0.01)      # nobody has pressed anything
        ctx.act("stop_airfoil")              # ...and now the user does
        seen["after"] = rule(2, -0.009)
        return stopped

    monkeypatch.setattr(api, "optimize_airfoil", fake_optimize)
    ctx.act("run_airfoil")
    deadline = time.time() + 60.0
    while A["opt"]["running"] and time.time() < deadline:
        time.sleep(0.05)
    assert not A["opt"]["running"], "the worker never finished"
    for _ in range(400):                     # drain the stage's own polling
        for fn in list(ctx.polls):
            fn()
        if A["opt"].get("score") is not None or A["opt"].get("score_error"):
            break
        time.sleep(0.05)

    # the button IS the stop rule…
    assert seen == {"before": False, "after": True}
    # …the run is kept, not discarded as an error…
    assert A["opt"]["stopped"] is True and not A["opt"]["error"]
    assert A["opt"]["report"] is stopped
    # …it is scored against its seed like any other run…
    assert (A["opt"].get("score") or {}).get("optimised")
    # …and the section it found is the one the pipeline now carries
    sec = session.section_of(S, "main") or {}
    assert sec.get("source") == "optimised"
    assert sec.get("w_upper") and sec.get("w_lower")


def test_the_seed_can_be_switched_off_for_a_caller_that_is_timing_the_search():
    """One extra XFOIL sweep is nothing beside a search, but a study MEASURING
    the search must be able to decline it."""
    rep = api.optimize_airfoil(budget=4, with_baseline=False, **_RUN)
    assert "baseline" not in rep
    assert rep.get("design"), "the incumbent's own report is not the opt-out"
