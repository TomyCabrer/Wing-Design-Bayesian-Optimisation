"""A finished section search can be given more evaluations — the same search.

Stage 2 could already STOP a search and keep the best section it had reached
(session 58): the stop is a stop rule, so ``api.run`` rebuilds the result from
its own per-evaluation log instead of unwinding, and every XFOIL sweep already
paid for survives. What it could not do was the other half of that sentence —
**give a search that had not converged more evaluations.** The only way to
spend a bigger budget was to press Optimise again, which starts from zero and
throws away every polar the first run bought.

Three statements are asserted here.

**One argument list, not two.** A run's arguments are now built once
(``stages.airfoil.shape_kwargs``) and handed to both api calls it makes, and
kept as the run's LAUNCH SNAPSHOT. That is what makes a continuation possible
at all, and the test binds the dict against both signatures — a misspelled or
stale keyword is then a failure here rather than a TypeError minutes into a
live XFOIL run.

**A continuation re-flies the run, not the form.** ``session.continue_section``
arms off the snapshot, so a user who edited the Reynolds number after the run
gets the OLD search lengthened and is told, in words, what has moved. The
alternative — re-reading the form — would silently relabel a different search
as a continuation.

**The honesty note travels.** Same config + same seed + bigger budget is
prefix-identical for the optimisers in ``api.CONTINUABLE_OPTIMISERS`` and NOT
for ``ga`` or for a BO run whose ``n_init`` was clamped by its own budget
(``api.continue_run_config`` measures both). Stage 2 shows that verdict rather
than promising containment it cannot deliver.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aerobo import api                                        # noqa: E402


def _session():
    from gui.v3 import session

    return session.make_session("air")


def _launch(budget: int = 24, seed: int = 3, optimiser: str = "bo",
            n_restarts: int = 1, **shape) -> dict:
    """A launch snapshot shaped like the one the worker stores."""
    base = {"re": 1.0e6, "mach": 0.0, "cl_design": 0.5, "tc_min": 0.10,
            "cm_max": 0.08, "anchor": None, "twist_order": 1,
            "twist_max_deg": 6.0, "alpha_max_deg": 10.0, "chord_order": 0,
            "chord_max_frac": 0.5, "objective": "cd", "wing": None}
    base.update(shape)
    return {"shape": base, "seed": int(seed),
            "search": {"optimiser": optimiser, "budget": int(budget),
                       "refusal": None, "n_init": None},
            "n_restarts": int(n_restarts)}


def _record(budget: int = 24, spent: int = 24, partial: bool = False,
            optimiser: str = "bo", dim: int = 8) -> dict:
    """A finished section report, shaped like ``api.optimize_airfoil``'s."""
    return {"config": {"problem_name": "airfoil (section)",
                       "optimiser": optimiser, "budget": int(budget),
                       "seed": 3, "flags": {}},
            "result": {"n_evals": int(spent), "budget": int(budget),
                       "partial": bool(partial), "dim": int(dim)}}


# ------------------------------------------- 1. one argument list, both calls

def test_the_stage_builds_one_argument_list_that_both_api_calls_accept():
    """The snapshot IS the call. Asserted against the real signatures.

    ``shape_kwargs`` exists so the live sampler's config and the search it
    samples cannot drift apart, and so a continuation has something exact to
    re-fly. Both properties die the moment one of its keys stops being an
    argument of the functions it is spread into, and that failure would
    otherwise surface as a TypeError after the user has waited for a run.
    """
    from gui.v3 import session
    from gui.v3.stages import airfoil

    S = _session()
    A = session.airfoil_state(S, "main")
    shape = airfoil.shape_kwargs(S, "main", A["opt"], A["weights"], None)

    assert shape["objective"], "the snapshot must name the scalar it maximises"
    for fn in (api.airfoil_run_config, api.optimize_airfoil):
        names = set(inspect.signature(fn).parameters)
        unknown = sorted(set(shape) - names)
        assert not unknown, f"{fn.__name__} does not take {unknown}"
        # and the whole call the worker makes, not only the shape half
        inspect.signature(fn).bind_partial(
            **shape, optimiser="bo", budget=8, seed=1, refusal=None,
            n_init=None, feasibility="screen")


def test_the_snapshot_carries_the_form_not_a_default():
    """A changed field reaches the snapshot — otherwise "the same search" is
    a statement about the shipped defaults rather than about the run."""
    from gui.v3 import session
    from gui.v3.stages import airfoil

    S = _session()
    A = session.airfoil_state(S, "main")
    A["opt"]["twist_max_deg"] = 4.25
    A["opt"]["chord_order"] = 2
    shape = airfoil.shape_kwargs(S, "main", A["opt"], A["weights"], None)
    assert shape["twist_max_deg"] == pytest.approx(4.25)
    assert shape["chord_order"] == 2


# ----------------------------------------------------- 2. what is on offer

def test_a_stopped_run_is_offered_the_rest_of_the_budget_it_was_launched_with():
    """The user already chose that number and then interrupted it: "continue"
    means finish what was started, not start a second policy."""
    from gui.v3 import session

    out = session.continue_section(_record(budget=24, spent=9, partial=True),
                                   _launch(budget=24))
    assert out["error"] is None
    assert out["extra"] == 15
    assert out["note"]["budget"] == 24
    assert out["cont"]["search"]["budget"] == 24


def test_a_run_that_spent_its_whole_budget_is_offered_half_of_it_again():
    from gui.v3 import session

    out = session.continue_section(_record(budget=24, spent=24), _launch(24))
    assert out["extra"] == 12
    assert out["cont"]["search"]["budget"] == 36


def test_the_offer_can_be_overridden():
    from gui.v3 import session

    out = session.continue_section(_record(), _launch(), extra=5)
    assert out["cont"]["search"]["budget"] == 29


# --------------------------------- 3. it re-flies the run, and says so out loud

def test_a_continuation_re_flies_the_snapshot_and_names_what_has_moved():
    """THE WHOLE POINT. The form moved; the continuation did not follow it.

    If this ever reads the form instead, the button becomes "run a different
    search at a bigger budget" while still calling itself a continuation —
    and the evaluations it claims to have re-used were never flown.
    """
    from gui.v3 import session

    launch = _launch(budget=20, seed=7)
    now = dict(launch["shape"])
    now["re"] = 3.0e5                       # the user edited the design point
    now["cl_design"] = 0.9
    out = session.continue_section(_record(budget=20), launch, shape_now=now)

    assert out["error"] is None
    assert out["cont"]["shape"]["re"] == pytest.approx(1.0e6)
    assert out["cont"]["shape"]["cl_design"] == pytest.approx(0.5)
    assert out["cont"]["seed"] == 7, "a different seed is a different search"
    assert sorted(out["drift"]) == ["the Reynolds number",
                                    "the design lift coefficient"]


def test_an_unmoved_form_reports_no_drift():
    from gui.v3 import session

    launch = _launch()
    out = session.continue_section(_record(), launch,
                                   shape_now=dict(launch["shape"]))
    assert out["drift"] == []


def test_a_run_this_shell_did_not_launch_cannot_be_continued():
    """No snapshot, no continuation — and the reason is the one the user can
    act on. A report reloaded after a restart is still readable; what it
    cannot do is claim to re-fly a search nobody recorded."""
    from gui.v3 import session

    out = session.continue_section(_record(), {})
    assert out["cont"] is None
    assert "optimise again" in out["error"]


def test_several_independent_searches_are_not_one_search():
    from gui.v3 import session

    out = session.continue_section(_record(), _launch(n_restarts=3))
    assert out["cont"] is None
    assert "independent searches" in out["error"]


def test_there_is_nothing_to_continue_before_the_first_run():
    from gui.v3 import session

    out = session.continue_section({}, _launch())
    assert out["cont"] is None and out["error"]


# ------------------------------------------------- 4. the honesty note travels

def test_an_optimiser_that_is_not_contained_says_so():
    """``ga`` sizes its population from the budget, so a longer run is a
    different search from its first evaluation. The stage must not paint that
    as "the evaluations you paid for come back"."""
    from gui.v3 import session

    out = session.continue_section(_record(optimiser="ga"),
                                   _launch(optimiser="ga"))
    assert out["note"]["exact"] is False
    assert "not re-flown" in out["note"]["why"]

    ok = session.continue_section(_record(), _launch())
    assert ok["note"]["exact"] is True


def test_the_continuation_keeps_the_optimiser_the_run_flew():
    """Stage 1's policy may have moved since the run; the continuation is the
    run's own search, so its optimiser comes off the snapshot."""
    from gui.v3 import session

    out = session.continue_section(_record(optimiser="slsqp"),
                                   _launch(optimiser="slsqp"))
    assert out["cont"]["search"]["optimiser"] == "slsqp"


# ------------------------------- 5. the claim itself, on the real section problem

@pytest.mark.slow
def test_the_evaluations_already_paid_for_come_back(tmp_path):
    """The load-bearing measurement, on THIS stage's problem.

    Everything above is arithmetic about a promise; this is the promise. Same
    section configuration, same seed, bigger budget — the longer run must fly
    the shorter one's designs first, or "continue" is a lie and the honest
    button would be "search again".
    """
    import numpy as np

    common = dict(re=5.0e5, mach=0.0, cl_design=0.5, tc_min=0.10, cm_max=0.08,
                  optimiser="bo", seed=1, n_init=4, with_section=False,
                  with_baseline=False, results_dir=str(tmp_path))
    short = api.optimize_airfoil(budget=6, **common)["result"]
    long_ = api.optimize_airfoil(budget=9, **common)["result"]

    a = np.asarray(short["eval_x"], dtype=float)
    b = np.asarray(long_["eval_x"], dtype=float)
    assert a.shape[0] == 6 and b.shape[0] == 9
    assert np.max(np.abs(b[:6] - a)) == 0.0, "the continuation is not a prefix"


def test_finishing_a_stopped_run_re_flies_the_same_configuration():
    """9 of 24 flown means 15 to go and 24 in total.

    The arithmetic this guards against is `old + (old - spent)` = 39: a button
    that says "finish what you started" while launching a search 62 % bigger
    than the one the user interrupted. And because the configuration itself
    does not change, the containment question does not arise for ANY
    optimiser — including the ones a longer budget would re-plan.
    """
    from gui.v3 import session

    out = session.continue_section(_record(budget=24, spent=9, partial=True),
                                   _launch(24))
    assert (out["note"]["was"], out["note"]["added"],
            out["note"]["budget"]) == (9, 15, 24)
    assert out["note"]["exact"] is True

    ga = session.continue_section(
        _record(budget=24, spent=9, partial=True, optimiser="ga"),
        _launch(24, optimiser="ga"))
    assert ga["note"]["exact"] is True and ga["note"]["budget"] == 24


def test_asking_for_more_than_the_remainder_grows_the_budget():
    """Past the interrupted budget it IS a longer search again, and the
    api's containment verdict comes back with it."""
    from gui.v3 import session

    out = session.continue_section(_record(budget=24, spent=9, partial=True),
                                   _launch(24), extra=20)
    assert out["cont"]["search"]["budget"] == 29
    assert (out["note"]["was"], out["note"]["added"]) == (9, 20)
    assert out["note"]["exact"] is True


def test_a_front_run_is_not_promised_its_evaluations_back():
    """The containment measurement was made on SCALAR runs.

    A front flies a batch acquisition over the whole Pareto set, and nothing
    here has re-flown one at two budgets. It can still be continued — what it
    cannot do is claim the paid evaluations come back.
    """
    from gui.v3 import session

    out = session.continue_section(_record(), _launch(objective="pareto"))
    assert out["error"] is None and out["cont"] is not None
    assert out["note"]["exact"] is False
    assert "FRONT" in out["note"]["why"]


def test_the_runs_own_adopted_winner_is_not_drift():
    """A finished search adopts its own section — that is not the form moving.

    Without this the shell warned, after EVERY successful run, that the seed
    had changed, and pointed the user at the one alternative this repo has
    measured and lost with: starting a fresh search from the winner.
    """
    from gui.v3 import session

    launch = _launch()
    won = {"w_upper": [0.2, 0.3, 0.25, 0.2], "w_lower": [-0.1, -0.2, -0.1, 0.0]}
    record = {**_record(), "section": {"design": won}}
    now = dict(launch["shape"])
    now["anchor"] = [won["w_upper"], won["w_lower"]]

    assert session.continue_section(record, launch, shape_now=now)["drift"] == []

    # a section the user picked by hand IS drift — the suppression is about
    # this run's own output, not about the anchor field in general
    other = dict(now)
    other["anchor"] = [[0.9, 0.9, 0.9, 0.9], [-0.9, -0.9, -0.9, -0.9]]
    assert session.continue_section(record, launch, shape_now=other)["drift"] \
        == ["the seed section"]
