"""The measured search recommendation: the rule, the flags, the stop.

The STUDY is not re-run here (it is hours of solver time); what is gated is
everything downstream of it — that a frozen payload is read rather than
guessed, that the budget really is a function of the dimension and the
class, that a plan's flags are the flags a run would carry, and that the
adaptive stop stops for the reason it says it does.

The payload every test builds is SYNTHETIC and written to ``tmp_path``, so
these tests neither depend on the shipped study's numbers nor break when it
is re-measured. One test does read the shipped file — to gate its SHAPE, not
its values.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aerobo import api                                          # noqa: E402
from aerobo.optimize import budget as B                         # noqa: E402


from aerobo.optimize.refusal import MODES as B_REFUSAL_MODES  # noqa: E402


def _payload(**over) -> dict:
    wing = {
        "method": {"optimiser": "bo", "acqf": "ucb", "label": "bo-ucb"},
        "method_constrained": {"optimiser": "slsqp", "acqf": None,
                               "label": "slsqp"},
        "law": {"0.9": {"intercept": 5.0, "per_dim": 2.0},
                "0.95": {"intercept": 10.0, "per_dim": 6.0},
                "0.99": {"intercept": 20.0, "per_dim": 12.0}},
        "n_init": {"per_dim": 1.5, "min": 4, "max": 16},
        "restarts": 1,
        "stop": {"patience": 25, "tol": 0.005},
        "cost": {"per_eval_s": 0.002,
                 "overhead": {"bo": {"a0": 0.1, "a1": 0.01,
                                     "b0": 0.0, "b1": 0.0}}},
        "budget_clamp": [12, 400],
    }
    airfoil = dict(wing)
    airfoil = {**wing,
               "method": {"optimiser": "bo", "acqf": "logei",
                          "label": "bo-logei"},
               "method_constrained": {"optimiser": "bo", "acqf": None,
                                      "label": "bo-logcei"},
               "law": {"0.95": {"intercept": 8.0, "per_dim": 5.0}},
               "cost": {"per_eval_s": 1.5, "overhead": {}},
               "budget_clamp": [12, 200]}
    pl = {"version": 1, "measured": "2026-01-01", "n_runs": 100, "cases": 9,
          "machine": "test", "kinds": {"wing": wing, "airfoil": airfoil}}
    pl.update(over)
    return pl


@pytest.fixture()
def study(tmp_path) -> Path:
    p = tmp_path / "search_budget.json"
    p.write_text(json.dumps(_payload()))
    B._CACHE.clear()
    yield p
    B._CACHE.clear()


# --------------------------------------------------------------- the payload
def test_a_missing_study_raises_rather_than_inventing_a_default(tmp_path):
    """The whole point of the module is provenance: a recommendation nobody
    measured must not appear out of a hard-coded constant."""
    B._CACHE.clear()
    with pytest.raises(B.MissingStudyError):
        B.payload(tmp_path / "nope.json")


def test_the_shipped_study_has_the_shape_the_rule_reads():
    """Gates the SHAPE of data/search_budget.json, never its numbers."""
    if not B.PAYLOAD_PATH.exists():
        pytest.skip("the study has not been frozen in this checkout")
    B._CACHE.clear()
    pl = B.payload()
    assert pl["version"] == 1
    assert set(pl["kinds"]) >= {"wing", "airfoil"}
    for kind, block in pl["kinds"].items():
        # a class is EITHER fitted to a convergence law or time-boxed; the
        # section is the second kind, because nothing converged there inside
        # a budget anyone waits for
        assert block["budget_mode"] in ("law", "time")
        if block["budget_mode"] == "law":
            assert block["law"], f"{kind} claims a law and has none"
            for target, law in block["law"].items():
                assert 0.0 < float(target) <= 1.0
                assert law["per_dim"] > 0.0, "a budget must grow with d"
        else:
            assert set(block["seconds"]) == set(B.EFFORTS)
            assert (block["cost"] or {}).get("per_eval_s")
        assert block["method"]["optimiser"] in api.optimiser_names()
        assert block["method_constrained"]["optimiser"] in api.optimiser_names()
        lo, hi = block["budget_clamp"]
        assert 1 <= lo < hi
        # ...and every effort setting produces a runnable plan either way
        for effort in B.EFFORTS:
            plan = B.recommend(dim=8, kind=kind, constrained=True,
                               effort=effort)
            assert plan.budget >= lo
            assert plan.n_restarts >= 1
            assert plan.optimiser in api.optimiser_names()
    B._CACHE.clear()


# ------------------------------------------------------------------ the rule
def test_the_budget_grows_with_the_design_vector(study):
    small = B.recommend(dim=3, kind="wing", path=study)
    big = B.recommend(dim=14, kind="wing", path=study)
    assert big.budget > small.budget
    # ...and it is the fitted line, not a guess: 10 + 6 d at the 0.99 target
    assert small.budget == 10 + 6 * 3
    assert big.budget == 10 + 6 * 14


def test_effort_moves_the_target_and_the_budget_with_it(study):
    got = {e: B.recommend(dim=8, kind="wing", effort=e, path=study).budget
           for e in B.EFFORTS}
    assert got["quick"] < got["balanced"] < got["thorough"]
    assert B.recommend(dim=8, kind="wing", effort="quick",
                       path=study).target == 0.90


def test_the_class_decides_the_strategy_not_the_dimension(study):
    """A wing planform and an XFOIL section are different search problems and
    the study measured them apart; the rule must keep them apart."""
    w = B.recommend(dim=8, kind="wing", path=study)
    a = B.recommend(dim=8, kind="airfoil", path=study)
    assert w.optimiser == "bo" and w.acqf == "ucb"
    assert a.optimiser == "bo" and a.acqf == "logei"
    assert w.budget != a.budget


def test_a_constrained_problem_gets_the_constrained_winner(study):
    free = B.recommend(dim=6, kind="wing", constrained=False, path=study)
    con = B.recommend(dim=6, kind="wing", constrained=True, path=study)
    assert free.optimiser == "bo"
    assert con.optimiser == "slsqp"
    # a non-BO plan carries no BO flags at all
    assert con.flags() == {}
    assert con.n_init is None


def test_the_plan_states_its_split_so_a_truncated_run_means_something(study):
    plan = B.recommend(dim=6, kind="wing", path=study)
    assert plan.n_init == max(4, round(1.5 * 6))
    assert plan.flags() == {"acqf": "ucb", "bo_n_init": plan.n_init}
    # and the split is clamped by its own measured band
    assert B.recommend(dim=40, kind="wing", path=study).n_init == 16
    assert B.recommend(dim=1, kind="wing", path=study).n_init == 4


def test_the_budget_is_clamped_to_the_evidence(study):
    """Recommending past the longest budget the study ran is extrapolation."""
    assert B.recommend(dim=200, kind="wing", path=study).budget == 400
    assert B.recommend(dim=200, kind="airfoil", path=study).budget == 200


def test_a_measured_cost_beats_the_class_median(study):
    slow = B.recommend(dim=5, kind="wing", per_eval_s=10.0, path=study)
    fast = B.recommend(dim=5, kind="wing", path=study)
    assert slow.est_seconds > fast.est_seconds
    # the quote is in the unit a user reads: seconds, minutes or hours
    assert fast.est_text.endswith(" s")
    assert "min" in slow.est_text or "h" in slow.est_text


def test_an_unknown_kind_or_effort_is_refused(study):
    with pytest.raises(ValueError):
        B.recommend(dim=5, kind="fuselage", path=study)
    with pytest.raises(ValueError):
        B.recommend(dim=5, kind="wing", effort="whenever", path=study)


# --------------------------------------------------------------- the api hook
def test_search_kind_splits_the_section_from_everything_else():
    assert api.search_kind("airfoil (section)") == "airfoil"
    assert api.search_kind("trim wing") == "wing"
    # a wing that DESIGNS a section is still a planform search
    assert api.search_kind("wing + airfoil (XFOIL)") == "wing"


def test_recommended_search_reads_the_built_problem(study, monkeypatch):
    monkeypatch.setattr(B, "PAYLOAD_PATH", study)
    B._CACHE.clear()
    plan = api.recommended_search("tail")
    built = api.PROBLEM_SPECS["tail"].build({}, {}, None)
    assert plan.dim == built.dim
    assert plan.constrained is True
    # the dimension comes off the BUILT problem, so a freedom moves the budget
    wide = api.recommended_search(
        "tail + winglet + t/c [free height, designed tail + tip device]")
    assert wide.dim > plan.dim
    assert wide.budget > plan.budget
    B._CACHE.clear()


def test_a_slow_family_is_not_timed_to_find_out_it_is_slow(study, monkeypatch):
    """Timing one evaluation of an XFOIL family costs seconds; the study
    already measured that class, so the plan quotes it instead."""
    monkeypatch.setattr(B, "PAYLOAD_PATH", study)
    B._CACHE.clear()

    def _boom(*a, **kw):
        raise AssertionError("a slow problem must not be timed live")

    monkeypatch.setattr(api, "measure_eval_cost", _boom)
    plan = api.recommended_search("airfoil (section)",
                                  flags=api.airfoil_run_config().flags)
    assert plan.per_eval_s == 1.5
    B._CACHE.clear()


# ------------------------------------------------------------- the BO split
def test_bo_n_init_flag_states_the_split_and_the_default_is_untouched():
    assert api._bo_split(40, 5) == (10, 30)          # 2d, capped at 16
    assert api._bo_split(40, 12) == (16, 24)
    assert api._bo_split(40, 5, 20) == (20, 20)      # stated
    assert api._bo_split(40, 5, 999) == (39, 1)      # clamped, still runs
    with pytest.raises(ValueError):
        api._bo_n_init(api.RunConfig(problem_name="trim wing",
                                     flags={api.BO_N_INIT_FLAG: 0}))


def test_the_split_flag_reaches_the_run():
    cfg = api.RunConfig(problem_name="trim wing", optimiser="bo", budget=14,
                        seed=0, flags={api.BO_N_INIT_FLAG: 12})
    res = api.run(cfg)
    assert res.n_evals == 14
    # 12 Sobol + 2 acquisition steps: the BO diagnostics fire once per
    # acquisition step, so the split is observable in the record
    assert len(res.bo_iters or []) == 2


# --------------------------------------------------------- the adaptive stop
def test_the_stop_needs_a_span_before_it_can_call_anything_flat():
    stop = B.ConvergenceStop(patience=3, tol=0.01, max_evals=100)
    for _ in range(10):
        assert stop.update(1.0) is False       # no span yet: never "converged"
    assert stop.n == 10


def test_the_stop_fires_on_a_plateau_and_says_why():
    stop = B.ConvergenceStop(patience=5, tol=0.01, max_evals=100)
    fired = False
    for v in [1.0, 2.0, 3.0, 4.0, 5.0] + [5.0] * 20:
        fired = stop.update(v)
        if fired:
            break
    assert fired
    assert stop.n < 26
    assert "no improvement" in stop.reason


def test_the_budget_is_always_the_backstop():
    stop = B.ConvergenceStop(patience=1000, tol=0.5, max_evals=7)
    fired = [stop.update(float(i)) for i in range(7)]
    assert fired[-1] is True
    assert "budget spent" in stop.reason


def test_a_pre_feasible_constrained_run_is_not_a_plateau():
    """A constrained search reports None until the first feasible point; a
    rule that read that as "no improvement" would kill every hard problem
    before it found anything at all."""
    stop = B.ConvergenceStop(patience=3, tol=0.01, max_evals=100)
    for _ in range(20):
        assert stop.update(None) is False


def test_run_with_a_stop_rule_keeps_every_evaluation_it_paid_for():
    stop = B.ConvergenceStop(patience=6, tol=0.02, max_evals=80)
    cfg = api.RunConfig(problem_name="trim wing", optimiser="bo", budget=80,
                        seed=0, flags={api.BO_N_INIT_FLAG: 6})
    res = api.run(cfg, stop_rule=lambda i, best: stop.update(best))
    assert res.partial is True
    assert res.stop_reason
    assert 0 < res.n_evals < 80
    assert res.n_evals == len(res.eval_y) == len(res.history)
    assert res.best_score is not None and res.best_x is not None


def test_no_stop_rule_is_the_legacy_path_bit_for_bit():
    cfg = api.RunConfig(problem_name="trim wing", optimiser="bo", budget=16,
                        seed=3)
    a = api.run(cfg)
    b = api.run(cfg, stop_rule=lambda i, best: False)
    assert a.n_evals == b.n_evals == 16
    assert a.best_score == b.best_score
    assert a.eval_y == b.eval_y
    assert a.partial is False and b.partial is False


def test_the_composite_objective_is_paid_for(tmp_path):
    """A weighted sum of six criteria is not the landscape of any one of
    them, and the study measured what that costs. The factor is applied to
    the SAME line, so a composite run's budget is still a function of its
    design vector."""
    pl = _payload()
    pl["kinds"]["wing"]["objective_factor"] = {"composite": 1.6,
                                               "measured": True}
    p = tmp_path / "search_budget.json"
    p.write_text(json.dumps(pl))
    B._CACHE.clear()
    own = B.recommend(dim=10, kind="wing", path=p)
    comp = B.recommend(dim=10, kind="wing", objective="composite", path=p)
    assert comp.budget > own.budget
    assert comp.budget == int(round(1.6 * (10.0 + 6.0 * 10)))
    assert "composite" in comp.why
    # an unmeasured factor changes nothing rather than guessing one
    pl["kinds"]["wing"].pop("objective_factor")
    p.write_text(json.dumps(pl))
    B._CACHE.clear()
    assert B.recommend(dim=10, kind="wing", objective="composite",
                       path=p).budget == own.budget
    B._CACHE.clear()


def test_restarts_split_the_budget_they_do_not_multiply_it(tmp_path):
    """The study compared 1 x B against k x B/k at the SAME total spend, so a
    plan of k restarts has to hand the shell a per-run budget of B/k — k runs
    of B each would be k times the search the law was fitted for."""
    pl = _payload()
    pl["kinds"]["wing"]["restarts"] = 2
    p = tmp_path / "search_budget.json"
    p.write_text(json.dumps(pl))
    B._CACHE.clear()
    plan = B.recommend(dim=10, kind="wing", path=p)
    total = 10 + 6 * 10
    assert plan.n_restarts == 2
    assert plan.budget == total // 2
    assert plan.budget * plan.n_restarts == total
    assert "restarts" in plan.why
    B._CACHE.clear()


def test_a_class_with_no_convergence_law_is_time_boxed(tmp_path):
    """The section search never reached the study's target at any budget it
    ran. That is an answer, not a gap: the budget then follows the WALL CLOCK
    the user is willing to spend, at the measured cost per candidate."""
    pl = _payload()
    pl["kinds"]["airfoil"] = {
        **pl["kinds"]["airfoil"],
        "law": {}, "budget_mode": "time",
        "seconds": {"quick": 60, "balanced": 300, "thorough": 900},
        "cost": {"per_eval_s": 2.0, "overhead": {}},
        "stop": {"patience": None, "tol": None, "measured": True,
                 "reason": "this class does not flatten"},
    }
    p = tmp_path / "search_budget.json"
    p.write_text(json.dumps(pl))
    B._CACHE.clear()
    quick = B.recommend(dim=8, kind="airfoil", effort="quick", path=p)
    balanced = B.recommend(dim=8, kind="airfoil", effort="balanced", path=p)
    # 60 s and 300 s of 2 s candidates, minus nothing (no overhead declared)
    assert quick.budget == 30
    assert balanced.budget == 150
    assert quick.est_seconds <= 60.0 and balanced.est_seconds <= 300.0
    assert "still climbing" in quick.why
    # ...and nothing arms a plateau detector on a class with no plateau
    assert quick.patience is None and quick.tol is None
    B._CACHE.clear()


def test_a_measured_cost_moves_a_time_boxed_budget(tmp_path):
    pl = _payload()
    pl["kinds"]["airfoil"] = {
        **pl["kinds"]["airfoil"], "law": {}, "budget_mode": "time",
        "seconds": {"quick": 60, "balanced": 300, "thorough": 900},
        "cost": {"per_eval_s": 2.0, "overhead": {}}, "budget_clamp": [12, 200],
    }
    p = tmp_path / "search_budget.json"
    p.write_text(json.dumps(pl))
    B._CACHE.clear()
    slow = B.recommend(dim=8, kind="airfoil", per_eval_s=10.0, path=p)
    fast = B.recommend(dim=8, kind="airfoil", per_eval_s=1.0, path=p)
    assert slow.budget < fast.budget
    assert slow.budget == 30                  # 300 s / 10 s a candidate
    B._CACHE.clear()


# ------------------------------------------------- what a refusal looks like

def _payload_with_refusal(tmp_path, adopt):
    """The shipped payload with one class's refusal verdict overridden."""
    pl = json.loads(json.dumps(B.payload()))
    pl["kinds"]["wing"]["refusal"] = {"adopt": adopt, "modes": [],
                                      "why": "a test"}
    p = tmp_path / "search_budget.json"
    p.write_text(json.dumps(pl))
    return p


def test_a_payload_that_never_measured_refusals_ships_the_sentinel(tmp_path):
    """Backward compatibility with the frozen study, stated as a test: no
    ``refusal`` block anywhere means the legacy path, not a crash.

    Built from a STRIPPED payload, not the shipped one — the shipped payload
    has measured refusals since 2026-08-04, and a test that reads it to assert
    the legacy default is testing today's measurement, not the fallback.
    """
    pl = json.loads(json.dumps(B.payload()))
    for block in pl["kinds"].values():
        block.pop("refusal", None)
    p = tmp_path / "search_budget.json"
    p.write_text(json.dumps(pl))
    B._CACHE.clear()
    assert B.refusal_for("wing", path=p) == "sentinel"
    plan = B.recommend(dim=8, kind="wing", constrained=True, path=p)
    assert plan.refusal == "sentinel"
    assert "bo_refusal" not in plan.flags()
    B._CACHE.clear()


def test_the_shipped_payload_adopts_the_measured_mode():
    """The other side of the same contract: what the study actually concluded
    is what a shell gets by default."""
    assert B.refusal_for("wing") in B_REFUSAL_MODES
    plan = B.recommend(dim=8, kind="wing", constrained=True)
    assert plan.refusal == "worst"
    assert plan.flags()["bo_refusal"] == "worst"


def test_an_adopted_mode_reaches_the_run_config(tmp_path):
    p = _payload_with_refusal(tmp_path, "worst")
    plan = B.recommend(dim=8, kind="wing", constrained=True, path=p)
    assert plan.refusal == "worst"
    assert plan.flags()["bo_refusal"] == "worst"
    assert "worst one that flew" in plan.why
    assert plan.to_dict()["refusal"] == "worst"


def test_the_sentinel_is_never_sent_as_a_flag(tmp_path):
    """An explicit ``bo_refusal="sentinel"`` would make every recommended
    config differ from the control it is identical to."""
    p = _payload_with_refusal(tmp_path, "sentinel")
    plan = B.recommend(dim=8, kind="wing", constrained=True, path=p)
    assert plan.flags().get("bo_refusal") is None


def test_a_non_bo_recommendation_carries_no_refusal_mode(tmp_path):
    """The imputation is a property of the SURROGATE; a GA has none."""
    pl = json.loads(json.dumps(B.payload()))
    pl["kinds"]["wing"]["refusal"] = {"adopt": "worst", "modes": []}
    pl["kinds"]["wing"]["method_constrained"] = {"optimiser": "ga",
                                                 "acqf": None}
    p = tmp_path / "ga.json"
    p.write_text(json.dumps(pl))
    plan = B.recommend(dim=8, kind="wing", constrained=True, path=p)
    assert plan.optimiser == "ga" and plan.refusal == "sentinel"
    assert plan.flags() == {}
