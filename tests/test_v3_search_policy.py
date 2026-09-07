"""V3.5: WHICH optimiser and HOW MANY evaluations are asked once, at stage 1.

The contracts this file gates are the ones that make the policy a policy
rather than a fourth place to type a budget:

1. a fresh session opens on the RECOMMENDATION, and the recommendation is a
   function of the stage's own design vector — open a freedom and it moves;
2. the thing SHOWN and the thing LAUNCHED are the same object: the run config
   is built from ``effective_wing_search``, never from the stage's fields;
3. "use my own values" hands the numbers over rather than reverting them, so
   nothing jumps when the user takes control;
4. with no frozen study installed the shell falls back to the user's own
   fields and says so — it never invents a budget;
5. the section stage asks the same question about ITS vector, per surface.
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
from gui.v3 import config, session                              # noqa: E402
from gui.v3.stages import airfoil as stage                      # noqa: E402


def _payload() -> dict:
    block = {
        "method": {"optimiser": "bo", "acqf": "logei", "label": "bo-logei"},
        "method_constrained": {"optimiser": "bo", "acqf": None,
                               "label": "bo-logcei"},
        "law": {"0.9": {"intercept": 4.0, "per_dim": 2.0},
                "0.95": {"intercept": 10.0, "per_dim": 6.0},
                "0.99": {"intercept": 20.0, "per_dim": 12.0}},
        "n_init": {"per_dim": 2.0, "min": 4, "max": 16},
        "restarts": 1,
        "stop": {"patience": 25, "tol": 0.005},
        "cost": {"per_eval_s": 0.002, "overhead": {}},
        "budget_clamp": [12, 400],
    }
    return {"version": 1, "measured": "2026-01-01", "n_runs": 42, "cases": 9,
            "machine": "test",
            "kinds": {"wing": block,
                      "airfoil": {**block,
                                  "cost": {"per_eval_s": 1.5, "overhead": {}},
                                  "budget_clamp": [12, 200]}}}


@pytest.fixture()
def study(tmp_path, monkeypatch):
    p = tmp_path / "search_budget.json"
    p.write_text(json.dumps(_payload()))
    monkeypatch.setattr(B, "PAYLOAD_PATH", p)
    B._CACHE.clear()
    yield p
    B._CACHE.clear()


@pytest.fixture()
def no_study(tmp_path, monkeypatch):
    monkeypatch.setattr(B, "PAYLOAD_PATH", tmp_path / "absent.json")
    B._CACHE.clear()
    yield
    B._CACHE.clear()


# ------------------------------------------------------------- 1. the policy
def test_a_fresh_session_opens_on_the_recommendation():
    S = session.make_session("air")
    assert session.search_is_recommended(S)
    assert session.search_state(S)["effort"] == "balanced"
    assert session.search_state(S)["stop_when_converged"] is True


def test_the_policy_is_asked_once_and_not_per_stage():
    """It lives at the top of the session, not inside ``wing`` — the section
    stage searches too."""
    S = session.make_session("air")
    assert "search" in S
    assert "search" not in S["wing"]
    assert "search" not in S["airfoil"]


def test_a_session_written_before_v3_5_still_answers(study):
    S = session.make_session("air")
    del S["search"]
    assert session.search_is_recommended(S)          # created on demand
    assert session.effective_wing_search(S)["source"] == "recommended"


def test_the_budget_follows_the_design_vector(study):
    S = session.make_session("air")
    before = session.effective_wing_search(S)
    dim_before = before["plan"].dim
    # open a freedom: the tail joins the problem, the vector grows
    S["wing"]["choices"]["tail"] = True
    session.apply_choices(S)
    after = session.effective_wing_search(S)
    assert after["plan"].dim > dim_before
    assert after["budget"] > before["budget"]


def test_effort_moves_every_stage_at_once(study):
    S = session.make_session("air")
    balanced = session.effective_wing_search(S)["budget"]
    sec_balanced = session.effective_airfoil_search(S)["budget"]
    assert session.set_search_effort(S, "quick")
    assert session.effective_wing_search(S)["budget"] < balanced
    assert session.effective_airfoil_search(S)["budget"] < sec_balanced


# ------------------------------------------------- 2. shown == launched
def test_the_run_config_is_built_from_the_policy_not_from_the_fields(study):
    S = session.make_session("air")
    S["wing"]["budget"] = 7                 # a field nobody is looking at
    S["wing"]["optimiser"] = "random"
    eff = session.effective_wing_search(S)
    d = config.cfg_dict(S)
    assert d["budget"] == eff["budget"] != 7
    assert d["optimiser"] == eff["optimiser"] == "bo"
    # ...and the split the plan states travels with it, so a truncated run
    # means what the study says it means
    assert d["flags"][api.BO_N_INIT_FLAG] == eff["n_init"]


def test_the_run_config_is_the_users_own_when_they_say_so(study):
    S = session.make_session("air")
    session.set_search_mode(S, "own")
    S["wing"]["budget"] = 7
    S["wing"]["optimiser"] = "random"
    d = config.cfg_dict(S)
    assert d["budget"] == 7
    assert d["optimiser"] == "random"
    assert api.BO_N_INIT_FLAG not in d["flags"]


def test_a_recommended_config_still_builds_and_runs(study):
    S = session.make_session("air")
    d = config.cfg_dict(S)
    d["budget"] = 12                        # the study's budget, but quick
    res = api.run(api.RunConfig(**d))
    assert res.n_evals == 12
    assert res.dim == session.effective_wing_search(S)["plan"].dim


# ------------------------------------------------------- 3. taking over
def test_taking_over_hands_the_numbers_across_rather_than_reverting(study):
    S = session.make_session("air")
    eff = session.effective_wing_search(S)
    sec = session.effective_airfoil_search(S)
    session.adopt_recommendation(S)
    assert not session.search_is_recommended(S)
    assert S["wing"]["budget"] == eff["budget"]
    assert S["wing"]["optimiser"] == eff["optimiser"]
    assert S["airfoil"]["opt"]["budget"] == sec["budget"]
    # and nothing moves them again
    S["wing"]["budget"] = 5
    assert config.cfg_dict(S)["budget"] == 5


def test_taking_over_keeps_the_two_decisions_it_has_no_field_for(
        study_impute):
    """The two the study MEASURED and the stage cannot show: what a refused
    design is imputed as, and the Sobol seed size. Copying only the fields
    handed the GP the -100 sentinel again (worst beat it 20/24 seed pairs)
    and grew the seed 4 -> 12 of a 29-evaluation budget, on a button whose
    label promises to hand the recommendation over."""
    S = session.make_session("air")
    eff = session.effective_wing_search(S)
    sec = session.effective_airfoil_search(S)
    assert eff["refusal"] == "worst" and eff["n_init"]      # the plan's own
    assert sec["refusal"] == "worst" and sec["n_init"]

    session.adopt_recommendation(S)
    after = session.effective_wing_search(S)
    assert after["source"] == "own"
    assert after["refusal"] == eff["refusal"]
    assert after["n_init"] == eff["n_init"]
    # ...through to the flags the run is actually built with
    d = config.cfg_dict(S)
    assert d["flags"][api.BO_REFUSAL_FLAG] == eff["refusal"]
    assert d["flags"][api.BO_N_INIT_FLAG] == eff["n_init"]
    # ...and the section search, which loses the same two on the same button
    sec_after = session.effective_airfoil_search(S)
    assert sec_after["source"] == "own"
    assert (sec_after["refusal"], sec_after["n_init"]) == (sec["refusal"],
                                                           sec["n_init"])


def test_choosing_my_own_values_by_hand_still_drops_them(study_impute):
    """The other half of the contract, and the reason this is a stash rather
    than a second recommendation: the RADIO is a fresh statement, so it
    leaves the shell's published defaults — and it takes back an adoption
    made earlier in the session."""
    S = session.make_session("air")
    session.set_search_mode(S, "own")
    eff = session.effective_wing_search(S)
    assert (eff["refusal"], eff["n_init"]) == ("sentinel", None)
    assert api.BO_N_INIT_FLAG not in config.cfg_dict(S)["flags"]

    session.set_search_mode(S, "recommended")
    session.adopt_recommendation(S)
    assert session.effective_wing_search(S)["refusal"] == "worst"
    session.set_search_mode(S, "recommended")
    session.set_search_mode(S, "own")
    eff = session.effective_wing_search(S)
    assert (eff["refusal"], eff["n_init"]) == ("sentinel", None)


# --------------------------------------------------- 4. no study installed
def test_without_a_study_the_users_own_numbers_run(no_study):
    S = session.make_session("air")
    assert session.search_is_recommended(S)         # the mode is still theirs
    eff = session.effective_wing_search(S)
    assert eff["source"] == "own"
    assert eff["budget"] == S["wing"]["budget"]
    assert session.search_study() is None
    assert session.search_state(S)["error"]         # ...and it says why


def test_without_a_study_there_is_no_stop_rule(no_study):
    S = session.make_session("air")
    assert session.stop_rule_factory(S) is None


# ------------------------------------------------------- 5. the section
def test_the_section_plan_counts_the_section_vector(study):
    S = session.make_session("air")
    oc = S["airfoil"]["opt"]
    # THE STAGE OPENS ON A COMPOSITE, and J scores a 2-D section: 8 CST
    # weights and nothing else, because neither a twist nor a chord law can
    # move a number that cannot see them (api.airfoil_run_config refuses the
    # pair outright)
    #
    # WHICH composite is a measured decision and it has moved once: stage 2
    # opened on plain `composite` until session 44 priced the sale (42 paired
    # seeds: 34/42 of its winners came back draggier than their own seed,
    # against 1/42 under the floor) and the default became `composite_goal`.
    # This assertion is therefore on the SET — the thing the dimension count
    # below actually depends on — plus the current default named once, so a
    # future measured change moves one line and does not silently pass.
    assert oc["objective"] in stage.COMPOSITE_OBJECTIVES
    assert oc["objective"] == "composite_goal"
    plan = session.airfoil_plan(S)
    assert plan is not None
    assert plan.kind == "airfoil"
    assert plan.dim == 8

    # ...and the PHYSICAL objective counts what it opens: the twist and chord
    # coefficients too, wherever that objective is a wing L/D
    oc["objective"] = "cd"
    plan = session.airfoil_plan(S)
    extra = (int(oc["twist_order"]) + int(oc["chord_order"])
             if session.wing_objective(S, "main") else 0)
    assert plan.dim == 8 + extra
    # ...and a wider law is paid for
    if session.wing_objective(S, "main"):
        oc["twist_order"] = int(oc["twist_order"]) + 1
        oc["chord_order"] = int(oc["chord_order"]) + 1
        wide = session.airfoil_plan(S)
        assert wide.dim == plan.dim + 2
        assert wide.budget > plan.budget


def test_each_surface_gets_its_own_section_plan(study):
    S = session.make_session("air")
    S["wing"]["choices"]["tail"] = True
    session.apply_choices(S)
    if not session.stage_visible(S, "airfoil_aft"):
        pytest.skip("this configuration has no second surface stage")
    main = session.effective_airfoil_search(S, "main")
    aft = session.effective_airfoil_search(S, "aft")
    assert main["source"] == aft["source"] == "recommended"
    assert main["plan"] is not None and aft["plan"] is not None


# ------------------------------------------------------------- the stop rule
def test_the_stop_rule_is_a_factory_so_seeds_do_not_share_a_plateau(study):
    S = session.make_session("air")
    factory = session.stop_rule_factory(S)
    assert factory is not None
    a, b = factory(), factory()
    for v in (1.0, 2.0, 3.0):
        a(1, v)
    assert b(1, 1.0) is False        # b has its own history


def test_switching_the_stop_off_removes_it(study):
    S = session.make_session("air")
    session.set_search_stop(S, False)
    assert session.stop_rule_factory(S) is None


def test_own_values_mean_the_run_goes_the_whole_way(study):
    """An adaptive stop is part of the recommendation; a user who took the
    numbers over did not also ask for their run to end early."""
    S = session.make_session("air")
    session.adopt_recommendation(S)
    assert session.stop_rule_factory(S) is None


# ------------------------------------------- 6. the refusal is a search flag

@pytest.fixture()
def study_impute(tmp_path, monkeypatch):
    """A study whose verdict is to impute refusals, on both classes."""
    pl = _payload()
    for k in pl["kinds"]:
        pl["kinds"][k]["refusal"] = {"adopt": "worst", "modes": [],
                                     "why": "a test"}
    p = tmp_path / "search_budget.json"
    p.write_text(json.dumps(pl))
    monkeypatch.setattr(B, "PAYLOAD_PATH", p)
    B._CACHE.clear()
    yield p
    B._CACHE.clear()


def test_an_adopted_imputation_reaches_the_wing_run(study_impute):
    S = session.make_session("air")
    eff = session.effective_wing_search(S)
    assert eff["refusal"] == "worst"
    assert config.cfg_dict(S)["flags"][api.BO_REFUSAL_FLAG] == "worst"


def test_it_is_a_SEARCH_flag_and_the_problem_is_unchanged(study_impute):
    """The V3 assembly contract is about the PHYSICS: a session that touched
    nothing must send the same problem whatever the search policy decided."""
    S = session.make_session("air")
    with_policy = config.physics_flags(config.cfg_dict(S))
    session.set_search_mode(S, "own")
    own = config.physics_flags(config.cfg_dict(S))
    assert with_policy == own
    assert api.BO_REFUSAL_FLAG not in own


def test_the_users_own_values_never_impute(study_impute):
    S = session.make_session("air")
    session.set_search_mode(S, "own")
    assert session.effective_wing_search(S)["refusal"] == "sentinel"
    assert api.BO_REFUSAL_FLAG not in config.cfg_dict(S)["flags"]


def test_a_study_without_the_verdict_sends_nothing(study):
    S = session.make_session("air")
    assert session.effective_wing_search(S)["refusal"] == "sentinel"
    assert api.BO_REFUSAL_FLAG not in config.cfg_dict(S)["flags"]
    assert session.effective_airfoil_search(S)["refusal"] == "sentinel"


def test_the_section_stage_carries_it_too(study_impute):
    S = session.make_session("air")
    eff = session.effective_airfoil_search(S)
    assert eff["refusal"] == "worst"
    cfg = api.airfoil_run_config(re=1e6, refusal=eff["refusal"])
    assert cfg.flags[api.BO_REFUSAL_FLAG] == "worst"
