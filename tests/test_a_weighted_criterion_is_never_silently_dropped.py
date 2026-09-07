"""A weight the user set either binds, or the shell says why it did not.

Both reference objectives build their per-criterion terms by iterating the
SEED's measured criteria (``goals.goals``), so a criterion the seed never
produced is not merely unweighted — it is absent from the objective's iteration
domain entirely. Weight ``ldcr`` at 0.35 (the largest in the shipped preset)
against a seed whose polar could not bracket the design lift and NOTHING in the
search rewards improving it, while the weights panel goes on showing 0.35.

The arithmetic is unchanged here: every published study's numbers are exactly
what they were. What is added is the RECORD of what the arithmetic already did
silently, and the two places on screen that read it.

The commonest cause is censoring — a stall march that stops converging before
it stalls — which is measured at 29.1 % of all refusals on this population. So
the cure is offered beside the alert: scoring a censored design at its bound is
a true lower bound on J for any non-negative weights, and it is now a control
rather than an argument only a script could pass.

WHERE THE ALERT IS NOT ENOUGH. Naming a dropped criterion is the whole fix for
the goal composite, whose ``J`` still carries it — but the ASF's scalar IS the
achievement, so there the weight buys literally nothing. That case has its own
mode and its own file: ``airfoil_asf_missing = "band_floor"`` puts the
criterion back at the bottom of the frozen band, measured in
``tests/test_asf_missing_criterion.py``. This file still pins the DEFAULT, so a
failure here is a change to what every published run did.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aerobo import airfoil_select as sel                 # noqa: E402


def _weights(**kw):
    return sel.ScoreWeights(**{k: float(v) for k, v in kw.items()})


def _reference():
    return sel.load_screen_reference()


def _goals(values: dict, **kw):
    return sel.ScoreGoals(goals=dict(values), **kw)


# ------------------------------------------------- 1. the engine writes it down

def test_a_criterion_the_seed_never_produced_is_recorded_as_dropped():
    ref = _reference()
    w = _weights(ldcr=0.35, clmax=0.2, thick=0.1, ldmax=0.15, cm=0.2)
    # a seed that produced everything EXCEPT ldcr
    goals = _goals({k: 1.0 for k in sel.CRITERIA if k != "ldcr"})
    scores = {k: 50.0 for k in sel.CRITERIA}

    got = sel.asf_terms(scores, goals, ref, w)
    assert "ldcr" not in got["rows"]                  # it never entered J...
    assert got["dropped"]["ldcr"]                     # ...and that is recorded
    assert "reference" in got["dropped"]["ldcr"]

    short = sel.goal_shortfalls(scores, goals, ref, w)
    assert "ldcr" not in short["rows"]
    assert "ldcr" in short["dropped"]


def test_a_criterion_nobody_weighted_is_not_reported_as_dropped():
    """"Dropped" means "your weight bought nothing". A criterion weighted 0
    was never in J and was never meant to be — saying it was dropped would be
    a charge the objective never made."""
    ref = _reference()
    w = _weights(ldcr=0.0, clmax=1.0)
    goals = _goals({k: 1.0 for k in sel.CRITERIA if k != "ldcr"})
    got = sel.asf_terms({k: 50.0 for k in sel.CRITERIA}, goals, ref, w)
    assert "ldcr" not in got["dropped"]


def test_a_candidate_that_produced_no_value_is_recorded_too():
    ref = _reference()
    w = _weights(ldcr=0.35, clmax=0.2)
    goals = _goals({k: 1.0 for k in sel.CRITERIA})
    scores = {k: 50.0 for k in sel.CRITERIA}
    scores["ldcr"] = float("nan")

    got = sel.asf_terms(scores, goals, ref, w)
    assert "ldcr" not in got["rows"]
    assert "this design" in got["dropped"]["ldcr"]


def test_an_empty_reference_point_still_reports_every_weighted_criterion():
    """The wholly-unscoreable seed — the case the fallback warning exists for
    and which, until now, could not be drawn."""
    ref = _reference()
    w = _weights(ldcr=0.35, clmax=0.2)
    got = sel.asf_terms({k: 50.0 for k in sel.CRITERIA}, _goals({}), ref, w)
    assert got["rows"] == {}
    # every criterion the WEIGHTS name, which is the shipped preset's five
    # (``astall`` is weighted 0 there and is therefore not a loss)
    want = {k for k in sel.CRITERIA if w.normalised()[k] > 0.0}
    assert set(got["dropped"]) == want
    assert "astall" not in got["dropped"]


def test_the_aggregates_are_untouched():
    """Everything above is metadata. A study re-run must be bit-for-bit."""
    ref = _reference()
    w = _weights(ldcr=0.35, clmax=0.2, thick=0.1, ldmax=0.15, cm=0.2)
    goals = _goals({k: 1.0 for k in sel.CRITERIA})
    scores = {k: 40.0 + 3.0 * i for i, k in enumerate(sel.CRITERIA)}

    got = sel.asf_terms(scores, goals, ref, w)
    rows = {k: v["term"] for k, v in got["rows"].items()}
    assert got["min"] == min(rows.values())
    assert got["sum"] == sum(rows.values())
    assert got["worst"] == min(rows, key=rows.get)
    assert got["dropped"] == {}


# --------------------------------------------------------- 2. the shell says it

def _card(objective: str, working: dict, weights: dict):
    """The optimise view, with a finished run of ``objective`` on it."""
    from gui.v3.app import assemble

    ctx = assemble("air")
    ctx.act("accept_mission")
    A = ctx.S["airfoil"]
    A["weights"] = dict(weights)
    A["opt"]["objective"] = objective
    key = "asf" if objective == "composite_asf" else "goal"
    A["opt"]["report"] = {
        "conditions": {"objective": objective, "cl_design": 0.5},
        "result": {"best_x": [0.1] * 8, "best_score": 60.0, "n_evals": 24,
                   "budget": 24},
        "wall_time_s": 1.0,
        "design": {"breakdown": {key: working}},
    }
    A["opt"]["score"] = {
        "weights": dict(weights),
        "seed": {"composite": 50.0, "scores": {"clmax": 50.0}},
        "optimised": {"composite": 60.0, "scores": {"clmax": 60.0}},
        "delta": {"composite": 10.0, "scores": {"clmax": 10.0}},
    }
    ctx.render("airfoil", "optimise")
    return [getattr(e, "text", None) or ""
            for e in ctx.views[("airfoil", "optimise")].descendants()]


WEIGHTS = {"ldcr": 0.35, "clmax": 0.2, "thick": 0.1,
           "ldmax": 0.15, "cm": 0.2, "astall": 0.0}


def test_the_fallback_warning_can_actually_be_drawn():
    """It sat below an ``if not rows: return`` guard, and ``fallback`` is set
    precisely WHEN ``rows`` is empty — so the one warning that says "your
    weights did not reach this run" had never appeared on screen."""
    texts = _card("composite_asf",
                  {"rows": {}, "min": None, "sum": 0.0, "worst": None,
                   "fallback": "the seed could not be scored",
                   "dropped": {"ldcr": "the reference design has no value "
                                       "for it"}},
                  WEIGHTS)
    assert any("the reference point was empty" in t for t in texts), texts


def test_the_censoring_policy_is_a_control_and_travels_only_on_a_composite():
    """``api._airfoil_censored`` raises if a non-default mode reaches a -cd
    run, so the state may hold an answer the current objective cannot use —
    but nothing may SEND one."""
    from gui.v3.stages import airfoil as stage

    kw = stage.objective_kwargs("composite_asf", wing_guess=None,
                                weights={"clmax": 1.0}, reference=None,
                                censored="lower_bound")
    assert kw["censored"] == "lower_bound"

    kw = stage.objective_kwargs("cd", wing_guess={"mass_kg": 60.0},
                                weights={"clmax": 1.0}, reference=None,
                                censored="lower_bound")
    assert "censored" not in kw

    kw = stage.objective_kwargs("composite_asf", wing_guess=None,
                                weights={"clmax": 1.0}, reference=None)
    assert "censored" not in kw          # the published path, unchanged


def test_the_alert_names_the_criterion_and_its_weight():
    texts = _card("composite_asf",
                  {"rows": {"clmax": {"term": 1.0, "weight": 0.2}},
                   "min": 1.0, "sum": 1.0, "worst": "clmax",
                   "dropped": {"ldcr": "the reference design has no value "
                                       "for it"}},
                  WEIGHTS)
    named = [t for t in texts if "carry a weight and were NOT" in t]
    assert named, texts
    assert "0.35" in named[0]
    assert "reference design has no value" in named[0]


def test_the_goal_objective_says_when_nothing_was_floored():
    """An empty floor is not "no penalty was paid": it is "there was no
    floor", which is a different run from the one the card promised."""
    texts = _card("composite_goal",
                  {"rows": {}, "penalty": 0.0, "worst": None,
                   "dropped": {"ldcr": "the reference design has no value "
                                       "for it"}},
                  WEIGHTS)
    assert any("nothing was floored" in t for t in texts), texts
    assert any("carry a weight and were NOT" in t for t in texts), texts
