"""The composite with a FLOOR under it: ``objective="composite_goal"``.

A weighted sum buys and sells. Measured on the frozen ``sec8_composite`` case
(NACA 2412 anchor, budget 48), the plain composite's winner gains ~11 points of
J while cruise L/D falls 12 % and section drag at the design lift RISES 14 % —
because over the shipped band 0.01 of |Cm| is worth four counts of L/D and the
stall angle is weighted zero. Across ten runs (two anchors x five seeds) every
winner regressed on one to three of the six criteria the user asked for.

``composite_goal`` keeps that same J and subtracts what falling BELOW a
reference point costs:

    J_goal = J - penalty * sum_k w_k * max(0, s_k(goal) - tol - s_k(x))

with the goals measured off the SEED by default. What these tests pin:

* the penalty is ONE-SIDED — a criterion above its goal contributes nothing,
  so a design that dominates the seed scores exactly the composite;
* the price is the closed form above, weight for weight, and ``penalty=0``
  recovers ``composite_objective`` bit-for-bit;
* the seed itself pays EXACTLY zero (the reference point carries the seed's own
  sub-scores, so the CST-vs-geometric t/c difference cannot fake a shortfall);
* a criterion weighted at zero is not defended, because it is not in J either;
* the plain composite, its flags and its frozen cells are untouched;
* the goal flags are refused on an objective that has no goals, rather than
  silently dropped.

The XFOIL sweeps are faked (tests/test_composite_objective.py's convention):
the branch under test is the scoring policy, not the boundary-layer march.
"""

from __future__ import annotations

import numpy as np
import pytest

from aerobo import api, xfoil_run
from aerobo.airfoil import AirfoilProblem
from aerobo.airfoil_select import (
    CRITERIA,
    GOAL_PENALTY,
    PRESETS,
    ScoreGoals,
    ScoreWeights,
    composite_evaluation,
    composite_objective,
    exchange_rates,
    goal_evaluation,
    goal_objective,
    load_screen_reference,
    score_candidates,
    seed_goals,
    sub_score,
)
from gui.v3.stages import airfoil as stage

WING = {"mass_kg": 12.0, "v_ms": 20.0, "altitude_m": 0.0,
        "s_ref_m2": 1.0, "aspect_ratio": 8.0, "taper": 0.6}


def _install_fake_xfoil(monkeypatch, *, cm=-0.04, censored=False):
    """Cruise sweep vs the wide stall sweep, dispatched on the alpha range."""

    def fake(coords, re, mach, alphas, **kw):
        a = np.asarray(alphas, dtype=float)
        if a.max() > 10.5:
            cl = (0.3 + 0.09 * a if censored
                  else 1.6 - 0.01 * (a - 14.0) ** 2)
        else:
            cl = 0.25 + 0.11 * a
        cd = 0.006 + 1e-4 * (a - 1.0) ** 2
        return xfoil_run.XfoilPolarResult(alpha_deg=a, cl=cl, cd=cd,
                                          cm=np.full_like(a, cm),
                                          n_requested=a.size)

    monkeypatch.setattr(xfoil_run, "run_xfoil_polar", fake)


@pytest.fixture()
def setup(monkeypatch):
    """(problem, frozen band, gdp-sweep weights, the seed's own evaluation)."""
    _install_fake_xfoil(monkeypatch)
    prob = AirfoilProblem()
    ref, w = load_screen_reference(), PRESETS["gdp-sweep"]
    out = composite_evaluation(prob.w0, prob, ref, w)
    assert out["composite"] is not None, "the fixture's seed must be scoreable"
    return prob, ref, w, out


def _goal_at(key, score_target, reference):
    """The RAW value whose sub-score is ``score_target`` — the band inverted.

    Lets a test state "the goal sits ten points above where this design
    landed" and then assert the price of exactly those ten points, without
    restating the penalty formula to compute the expectation.
    """
    lo, hi = reference.band(key)
    span = hi - lo
    if key == "cm":                      # lower-better: mirrored
        return hi - score_target * span / 100.0
    return lo + score_target * span / 100.0


# ------------------------------------------------------------ the arithmetic


def test_a_design_at_its_goals_pays_nothing(setup):
    """The reference point is the seed, so the SEED is not a shortfall. Exact,
    not approximate: the goals carry the sub-scores the composite itself
    produced, so the CST-vs-geometric t/c gap cannot fake a hundredth of a
    point of regression."""
    prob, ref, w, seed = setup
    goals = seed_goals(prob, ref, w)

    out = goal_evaluation(prob.w0, prob, ref, w, goals)

    assert out["goal_penalty"] == 0.0
    assert out["composite_goal"] == out["composite"] == seed["composite"]
    assert out["score"] == out["f"] == out["composite_goal"]


def test_a_shortfall_costs_penalty_times_weight_times_points(setup):
    """Ten sub-score points of ldcr, priced by hand: the goal is placed ten
    points above where the design lands, so the whole penalty must be
    ``penalty * w_ldcr * 10`` and nothing else — no other criterion can
    contribute, because every other goal is placed below the design."""
    prob, ref, w, seed = setup
    s = seed["scores"]
    goals = ScoreGoals(
        goals={k: _goal_at(k, s[k] + (10.0 if k == "ldcr" else -25.0), ref)
               for k in CRITERIA},
        penalty=4.0, tol=0.0, source="hand-placed")

    out = goal_evaluation(prob.w0, prob, ref, w, goals)

    expected = 4.0 * w.normalised()["ldcr"] * 10.0
    assert out["goal_penalty"] == pytest.approx(expected, abs=1e-9)
    assert out["composite_goal"] == pytest.approx(
        seed["composite"] - expected, abs=1e-9)
    rows = out["goal"]["rows"]
    assert rows["ldcr"]["shortfall"] == pytest.approx(10.0, abs=1e-9)
    assert out["goal"]["worst"] == "ldcr"
    assert all(rows[k]["shortfall"] == 0.0 for k in CRITERIA if k != "ldcr")


def test_being_better_than_the_goal_is_never_priced(setup):
    """One-sided. A symmetric distance-to-goal would punish a design for
    beating the seed, which is the opposite of the point."""
    prob, ref, w, seed = setup
    s = seed["scores"]
    goals = ScoreGoals(goals={k: _goal_at(k, s[k] - 40.0, ref)
                              for k in CRITERIA},
                       penalty=1e3, source="far below")

    out = goal_evaluation(prob.w0, prob, ref, w, goals)

    assert out["goal_penalty"] == 0.0
    assert out["composite_goal"] == seed["composite"]


def test_the_goal_score_is_never_above_the_composite(setup):
    """J_goal <= J everywhere: the term only ever subtracts."""
    prob, ref, w, seed = setup
    s = seed["scores"]
    for offset in (-30.0, -5.0, 0.0, 5.0, 30.0):
        goals = ScoreGoals(goals={k: _goal_at(k, s[k] + offset, ref)
                                  for k in CRITERIA}, penalty=2.0)
        out = goal_evaluation(prob.w0, prob, ref, w, goals)
        assert out["composite_goal"] <= out["composite"] + 1e-12


def test_tol_moves_the_floor_down_by_exactly_that_many_points(setup):
    """``tol`` is slack in sub-score points, not a rescaling: 6 points of
    shortfall with 2.5 points of slack is 3.5 points of shortfall."""
    prob, ref, w, seed = setup
    s = seed["scores"]
    raw = {k: _goal_at(k, s[k] + (6.0 if k == "clmax" else -25.0), ref)
           for k in CRITERIA}

    tight = goal_evaluation(prob.w0, prob, ref, w,
                            ScoreGoals(goals=raw, penalty=1.0, tol=0.0))
    slack = goal_evaluation(prob.w0, prob, ref, w,
                            ScoreGoals(goals=raw, penalty=1.0, tol=2.5))

    w_clmax = w.normalised()["clmax"]
    assert tight["goal_penalty"] == pytest.approx(6.0 * w_clmax, abs=1e-9)
    assert slack["goal_penalty"] == pytest.approx(3.5 * w_clmax, abs=1e-9)


def test_a_criterion_weighted_at_zero_is_not_defended(setup):
    """astall is 0.0 in the gdp-sweep preset — it is not in J, so protecting it
    would be a preference nobody stated. Give it a weight and it is defended;
    that is the same lever, said once."""
    prob, ref, w, seed = setup
    s = seed["scores"]
    raw = {k: _goal_at(k, s[k] + (12.0 if k == "astall" else -25.0), ref)
           for k in CRITERIA}

    unweighted = goal_evaluation(prob.w0, prob, ref, w,
                                 ScoreGoals(goals=raw, penalty=1.0))
    w2 = ScoreWeights(thick=0.10, clmax=0.20, ldmax=0.15, ldcr=0.35, cm=0.20,
                      astall=0.20)
    weighted = goal_evaluation(prob.w0, prob, ref, w2,
                               ScoreGoals(goals=raw, penalty=1.0))

    assert unweighted["goal_penalty"] == 0.0
    assert weighted["goal_penalty"] == pytest.approx(
        12.0 * w2.normalised()["astall"], abs=1e-9)


def test_penalty_zero_recovers_the_plain_composite(setup):
    """The exact-penalty knob at its bottom stop: same number, same margins."""
    prob, ref, w, _seed = setup
    goals = ScoreGoals(goals=seed_goals(prob, ref, w).goals, penalty=0.0)

    j_goal, g_goal = goal_objective(prob.w0, prob, ref, w, goals)
    j_comp, g_comp = composite_objective(prob.w0, prob, ref, w)

    assert j_goal == j_comp
    assert np.array_equal(g_goal, g_comp)


def test_the_price_scales_linearly_with_the_multiplier(setup):
    prob, ref, w, seed = setup
    s = seed["scores"]
    raw = {k: _goal_at(k, s[k] + (8.0 if k == "ldmax" else -25.0), ref)
           for k in CRITERIA}
    one = goal_evaluation(prob.w0, prob, ref, w,
                          ScoreGoals(goals=raw, penalty=1.0))["goal_penalty"]
    ten = goal_evaluation(prob.w0, prob, ref, w,
                          ScoreGoals(goals=raw, penalty=10.0))["goal_penalty"]
    assert ten == pytest.approx(10.0 * one, abs=1e-9)
    assert one > 0.0


# ------------------------------------------------------------ the one map


def test_a_goal_and_a_candidate_are_scored_by_one_map(setup):
    """``sub_score`` is the map ``score_candidates`` applies. If the two ever
    drift, a goal stated in raw units means something the objective does not."""
    _prob, ref, w, seed = setup
    rec = {"name": "candidate", "eligible": True, "path": "",
           "tc": 0.14, "clmax": 1.5, "ldmax": 110.0, "ldcr": 80.0,
           "astall": 15.0, "cm_at": -0.02, "status": "ok"}
    scored = score_candidates([dict(rec)], w, reference=ref)[0]

    for key, value in (("thick", rec["tc"]), ("clmax", rec["clmax"]),
                       ("ldmax", rec["ldmax"]), ("ldcr", rec["ldcr"]),
                       ("astall", rec["astall"]),
                       ("cm", abs(rec["cm_at"]))):
        assert sub_score(key, value, ref) == scored[f"score_{key}"]


def test_an_exchange_rate_is_what_the_composite_really_moves_by(setup):
    """``exchange_rates`` must equal the J the composite actually pays for one
    unit — measured by scoring two records that differ by one unit of a single
    criterion, not by restating the formula."""
    _prob, ref, w, _seed = setup
    base = {"name": "a", "eligible": True, "path": "", "status": "ok",
            "tc": 0.14, "clmax": 1.5, "ldmax": 110.0, "ldcr": 80.0,
            "astall": 15.0, "cm_at": -0.02}
    bumped = dict(base, name="b", ldcr=81.0)
    scored = {r["name"]: r["composite"]
              for r in score_candidates([base, bumped], w, reference=ref)}

    rates = exchange_rates(w, ref)
    assert (scored["b"] - scored["a"]) == pytest.approx(
        rates["ldcr"]["per_unit"], abs=1e-9)


def test_the_shell_can_ask_for_the_rates_without_a_band_of_its_own():
    """The public read-out: pure, and empty rather than invented when no
    frozen band can be loaded."""
    rates = api.score_exchange_rates("gdp-sweep")
    assert set(rates) == set(CRITERIA)
    assert rates["cm"]["per_unit"] > rates["ldcr"]["per_unit"]
    assert api.score_exchange_rates("gdp-sweep", {"bounds": {}}) == {}


def test_the_weights_do_not_say_what_the_search_will_trade(setup):
    """The finding behind the goal objective, pinned as a number: under the
    shipped band and the GDP preset, one hundredth of |Cm| is worth about four
    counts of cruise L/D — while cruise L/D carries the LARGEST weight (0.35)
    and |Cm| carries 0.20. And the stall angle, at weight 0, is free to sell."""
    _prob, ref, w, _seed = setup
    rates = exchange_rates(w, ref)

    counts_of_ldcr = (rates["cm"]["per_unit"] * 0.01) / rates["ldcr"]["per_unit"]
    assert counts_of_ldcr == pytest.approx(4.0, abs=0.2)
    assert w.normalised()["ldcr"] > w.normalised()["cm"]     # the inversion
    assert rates["astall"]["per_step"] == 0.0


def test_the_reference_point_states_both_spellings(setup):
    """The seed's goals carry raw metrics AND the sub-scores the composite
    produced for them; the sub-scores bind (they are what J is built from)."""
    prob, ref, w, seed = setup
    goals = seed_goals(prob, ref, w)

    assert set(goals.goals) == set(CRITERIA)
    assert goals.scores == {k: seed["scores"][k] for k in CRITERIA}
    for key in CRITERIA:
        assert goals.score_of(key, ref) == seed["scores"][key]
    assert ScoreGoals.from_dict(goals.to_dict()) == goals


def test_an_unscoreable_seed_leaves_the_goals_empty_and_says_so(monkeypatch):
    """A seed whose cl_max is censored cannot be a reference point. The goals
    are then empty — the objective falls back to the plain composite — and the
    source says why, instead of a floor invented out of a failed evaluation."""
    _install_fake_xfoil(monkeypatch, censored=True)
    prob = AirfoilProblem()
    ref, w = load_screen_reference(), PRESETS["gdp-sweep"]

    goals = seed_goals(prob, ref, w)

    assert goals.goals == {}
    assert "not scoreable" in goals.source and "censored" in goals.source


# ------------------------------------------------------------ the plumbing


def test_the_config_carries_the_objective_and_only_what_moved():
    cfg = api.airfoil_run_config(objective="composite_goal",
                                 score_weights="gdp-sweep")
    assert cfg.flags["airfoil_objective"] == "composite_goal"
    # goals unstated = measured off the seed, so nothing extra is declared
    assert "airfoil_score_goals" not in cfg.flags
    assert "airfoil_goal_penalty" not in cfg.flags

    stated = api.airfoil_run_config(objective="composite_goal",
                                    score_goals={"ldcr": 80.0},
                                    goal_penalty=3.0, goal_tol=1.0)
    assert stated.flags["airfoil_score_goals"] == {"ldcr": 80.0}
    assert stated.flags["airfoil_goal_penalty"] == 3.0
    assert stated.flags["airfoil_goal_tol"] == 1.0


def test_the_seed_is_in_the_training_set(monkeypatch):
    """The objective defends the seed, so the search must have SEEN it: the
    warm start puts w0 in the initial design (it used to read only ``x0`` and
    no-op on every 2-D section problem), and the run really evaluates it."""
    _install_fake_xfoil(monkeypatch)
    cfg = api.airfoil_run_config(objective="composite_goal", budget=6)
    assert cfg.flags[api.BO_WARM_START_FLAG] is True
    built = api.PROBLEM_SPECS[cfg.problem_name].build(
        cfg.mission_kwargs or {}, cfg.flags or {}, cfg.bounds_overrides)

    x_init = api._bo_x_init(cfg, built, built.bounds)
    w0 = np.clip(np.asarray(AirfoilProblem().w0, dtype=float),
                 built.bounds[:, 0], built.bounds[:, 1])
    assert x_init is not None and np.allclose(x_init[0], w0)

    res = api.run(cfg)
    visited = np.asarray(res.eval_x, dtype=float)
    assert np.isclose(np.abs(visited - w0).sum(axis=1).min(), 0.0, atol=1e-9)
    # …and the plain composite is NOT given the warm start it never had
    assert api.BO_WARM_START_FLAG not in api.airfoil_run_config(
        objective="composite").flags


def test_the_plain_composite_run_is_untouched_flag_for_flag():
    """Adding an objective must not change the one already frozen: a composite
    config still sends exactly the keys it sent, so its cached cells and every
    published number keep meaning what they meant."""
    cfg = api.airfoil_run_config(objective="composite",
                                 score_weights="gdp-sweep")
    assert set(cfg.flags) == {"airfoil_re", "airfoil_mach",
                              "airfoil_cl_design", "airfoil_tc_min",
                              "airfoil_cm_max", "airfoil_objective",
                              "airfoil_score_weights"}
    assert api.airfoil_run_config().flags == {
        "airfoil_re": 1e6, "airfoil_mach": 0.0, "airfoil_cl_design": 0.5,
        "airfoil_tc_min": 0.10, "airfoil_cm_max": 0.08}


@pytest.mark.parametrize("objective", ["cd", "composite"])
def test_a_goal_stated_on_another_objective_is_refused(objective):
    """A flag that is silently dropped is worse than one that is refused: only
    composite_goal reads a reference point."""
    with pytest.raises(ValueError, match="composite_goal"):
        api.airfoil_run_config(objective=objective, goal_penalty=2.0)
    with pytest.raises(ValueError, match="composite_goal"):
        api.airfoil_run_config(objective=objective,
                               score_goals={"ldcr": 80.0})


def test_the_penalty_default_is_the_calibrated_one():
    """`GOAL_PENALTY` is 3, and it is MEASURED — pinned so it cannot drift back.

    It shipped at 10 on an argument. Two studies then priced it: the mu sweep
    (12 paired seeds) found criteria-sold saturating by mu = 3 while J kept
    falling to mu = 100, and the powered head-to-head (`RESULTS_MU_ARM_STUDY.md`,
    pre-registered in `PREREG_SESSION46.md` §C, 42 paired seeds) found mu = 3
    better on BOTH endpoints — criteria sold -0.357 (sign 0.0227 / Wilcoxon
    0.0106 / t 0.0095) and plain J +1.14 median (sign 0.0436 / Wilcoxon 0.0074
    / t 0.0272).

    A default with a measurement behind it is worth a test; a default with an
    argument behind it is what this replaced.
    """
    assert GOAL_PENALTY == 3.0
    # …and it is still a DEFAULT, not a ban: both ends stay reachable and 0
    # still recovers the frozen composite exactly
    for mu in (0.0, 1.0, 3.0, 10.0, 100.0):
        cfg = api.airfoil_run_config(objective="composite_goal",
                                     goal_penalty=mu)
        assert cfg.flags["airfoil_goal_penalty"] == mu


def test_a_hand_built_config_is_refused_too():
    """Every study script assembles a RunConfig directly. A goal stated on an
    objective that has none must be refused at BUILD time as well, or it is
    dropped exactly where nobody is looking."""
    cfg = api.RunConfig(problem_name="airfoil (section)",
                        flags={"airfoil_objective": "composite",
                               "airfoil_goal_penalty": 3.0},
                        optimiser="random", budget=2, seed=0)
    with pytest.raises(ValueError, match="composite_goal"):
        api.PROBLEM_SPECS[cfg.problem_name].build(
            cfg.mission_kwargs or {}, cfg.flags or {}, cfg.bounds_overrides)


def test_an_unknown_goal_criterion_is_named_not_ignored():
    with pytest.raises(ValueError, match="unknown goal criteria"):
        api.airfoil_run_config(objective="composite_goal",
                               score_goals={"drag": 1.0})


def test_the_composites_two_refusals_are_inherited():
    """Same 2-D score, so the same two refusals: wing mode and a missing
    band."""
    with pytest.raises(ValueError, match="2-D SECTION"):
        api.airfoil_run_config(objective="composite_goal", wing=WING)


def test_the_built_problem_evaluates_the_goal_score(monkeypatch):
    _install_fake_xfoil(monkeypatch)
    cfg = api.airfoil_run_config(objective="composite_goal",
                                 score_goals={"ldcr": 200.0},
                                 goal_penalty=5.0)
    built = api.PROBLEM_SPECS[cfg.problem_name].build(
        cfg.mission_kwargs or {}, cfg.flags or {}, cfg.bounds_overrides)
    prob = AirfoilProblem()
    ref, w = load_screen_reference(), PRESETS["gdp-sweep"]

    j, g = built.callable(prob.w0)
    j_ref, g_ref = goal_objective(
        prob.w0, prob, ref, w,
        ScoreGoals(goals={"ldcr": 200.0}, penalty=5.0, source="stated"))
    j_plain, _ = composite_objective(prob.w0, prob, ref, w)

    assert j == pytest.approx(j_ref) and np.allclose(g, g_ref)
    # an out-of-reach goal is a REAL price, so the two objectives disagree
    assert j < j_plain - 1.0
    # …and the breakdown the shell draws carries the working
    out = built.evaluate(prob.w0)
    assert out["goal"]["rows"]["ldcr"]["shortfall"] > 0.0
    assert out["composite_goal"] == pytest.approx(j)


def test_the_reference_point_is_resolved_once_per_built_problem(monkeypatch):
    """A moving reference point is a moving objective — the same reason the
    band is frozen. The seed is evaluated once, however many candidates are."""
    _install_fake_xfoil(monkeypatch)
    calls = {"n": 0}
    real = api._resolve_airfoil_goals

    def counted(*a, **kw):
        calls["n"] += 1
        return real(*a, **kw)

    monkeypatch.setattr(api, "_resolve_airfoil_goals", counted)
    cfg = api.airfoil_run_config(objective="composite_goal")
    built = api.PROBLEM_SPECS[cfg.problem_name].build(
        cfg.mission_kwargs or {}, cfg.flags or {}, cfg.bounds_overrides)

    assert calls["n"] == 0                    # building must stay cheap
    prob = AirfoilProblem()
    for _ in range(3):
        built.callable(prob.w0)
    assert calls["n"] == 1


def test_the_report_states_the_floor_it_was_run_against(monkeypatch):
    """The reference point is as much a part of the scalar as the band, so a
    stored run says what it was told not to fall below."""
    _install_fake_xfoil(monkeypatch)
    rep = api.optimize_airfoil(objective="composite_goal",
                               score_weights="gdp-sweep",
                               optimiser="random", budget=3, seed=0)

    assert rep["conditions"]["objective"] == "composite_goal"
    goals = rep["conditions"]["score"]["goals"]
    assert goals["source"] == "seed"
    assert goals["penalty"] == GOAL_PENALTY
    assert set(goals["goals"]) == set(CRITERIA)
    assert rep["design"]["breakdown"]["composite_goal"] == pytest.approx(
        rep["result"]["best_score"])


# ------------------------------------------------------------ the card


def test_the_menu_offers_the_goal_objective():
    for wing_mode in (True, False):
        choices = stage.OBJECTIVE_CHOICES(wing_mode)
        assert "composite_goal" in choices
        assert "seed" in choices["composite_goal"]
    assert stage.is_composite("composite_goal")
    assert stage.is_composite("composite")
    assert not stage.is_composite("cd")


def test_the_goal_objective_never_travels_with_a_wing_guess():
    kw = stage.objective_kwargs("composite_goal", wing_guess=WING,
                                weights={"clmax": 1.0}, reference=None)
    assert kw["objective"] == "composite_goal" and kw["wing"] is None
    assert kw["score_weights"] == {"clmax": 1.0}


def test_the_shell_plans_a_goal_run_as_the_section_run_it_is():
    """The budget plan reads the objective by NAME. Reading only "composite"
    planned a goal run as a -cd run — wing guess, twist law, the wrong design
    vector and the wrong budget — the stale-branch bug class this repo keeps
    meeting. The plan must match the plain composite's, not the -cd one's."""
    from gui.v3 import session

    S = session.make_session("air")
    A = session.airfoil_state(S, "main")

    A["opt"]["objective"] = "composite"
    plain = session.airfoil_plan(S, "main")
    A["opt"]["objective"] = "composite_goal"
    goal = session.airfoil_plan(S, "main")
    A["opt"]["objective"] = "cd"
    own = session.airfoil_plan(S, "main")

    assert plain is not None and goal is not None and own is not None
    assert goal.budget == plain.budget
    assert goal.dim == plain.dim == 8               # a SECTION, not a wing
    assert goal.objective == "composite_goal"       # planned under its own name
    assert (own.dim, own.budget) != (goal.dim, goal.budget)


def test_the_comparison_covers_every_criterion_on_any_objective():
    """The metric table reports the SAME six criteria whatever was maximised.

    Only a composite run computes the wide stall sweep, so a -cd or wing run's
    own breakdown carries no cl_max, stall angle or either L/D — and the table
    used to answer "what did the optimisation change?" with four rows while
    the card beside it scored six criteria. The scoring block measures all six
    for any objective, so it is handed to the table."""
    from gui import nice_app as v1

    metrics = {"tc": 0.1199, "clmax": 1.4443, "astall": 15.5, "ldmax": 102.88,
               "ldcr": 83.94, "cd_at": 0.005957, "cm_at": -0.0464,
               "alpha_at": 2.46}
    moved = dict(metrics, tc=0.1496, clmax=1.5279, astall=14.5, ldmax=121.91,
                 ldcr=73.63, cd_at=0.006791, cm_at=-0.0041, alpha_at=3.62)
    rep = {"baseline": {"breakdown": {}}, "design": {"breakdown": {}}}
    score = {"seed": {"composite": 59.32, "metrics": metrics},
             "optimised": {"composite": 70.59, "metrics": moved}}

    bare = {r["metric"] for r in v1.airfoil_compare_rows(rep)}
    full = {r["metric"]: r for r in v1.airfoil_compare_rows(rep, score)}

    assert bare == set()                       # nothing measured, nothing said
    for label in ("c_l max", "stall angle [deg]", "(L/D) max",
                  "L/D at design c_l", "section t/c",
                  "c_d at design c_l [counts]", "|c_m| at design c_l",
                  "composite score J"):
        assert label in full, label
    assert full["L/D at design c_l"]["dir"] == "worse"      # the sale, judged
    assert full["c_d at design c_l [counts]"]["dir"] == "worse"
    assert full["c_l max"]["dir"] == "better"
    # …and a wing quantity NEITHER side measured is still absent, not a dash
    assert "wing L/D" not in full


def test_the_score_rows_show_every_criterion_and_mark_a_regression():
    """Each criterion appears once, carrying the RAW values behind its
    sub-scores, and a criterion that ended below the seed is marked — the
    reading this table exists to make possible."""
    score = {
        "weights": {"ldcr": 0.35, "ldmax": 0.15, "clmax": 0.20,
                    "astall": 0.0, "thick": 0.10, "cm": 0.20},
        "seed": {"composite": 59.32,
                 "scores": {"ldcr": 84.26, "ldmax": 57.17, "clmax": 58.94,
                            "astall": 60.53, "thick": 11.69, "cm": 41.50},
                 "metrics": {"ldcr": 83.94, "ldmax": 102.88, "clmax": 1.4443,
                             "astall": 15.5, "tc": 0.1199, "cm_at": -0.0464}},
        "optimised": {"composite": 70.59,
                      "scores": {"ldcr": 65.34, "ldmax": 79.12,
                                 "clmax": 68.72, "astall": 51.75,
                                 "thick": 29.67, "cm": 95.72},
                      "metrics": {"ldcr": 73.63, "ldmax": 121.91,
                                  "clmax": 1.5279, "astall": 14.5,
                                  "tc": 0.1496, "cm_at": -0.0041}},
        "delta": {"composite": 11.27,
                  "scores": {"ldcr": -18.92, "ldmax": 21.94, "clmax": 9.78,
                             "astall": -8.77, "thick": 17.99, "cm": 54.22}},
    }
    rows = stage.score_rows(score)

    assert len(rows) == 1 + len(CRITERIA)            # J, then one each
    by_key = {r["metric"]: r for r in rows}
    ldcr = next(v for k, v in by_key.items() if k.startswith("L/D at design"))
    assert "83.9" in ldcr["metric"] and "73.6" in ldcr["metric"]
    assert "below the seed" in ldcr["metric"]        # the sale, marked
    assert ldcr["change"] == "-18.9"
    stall = next(v for k, v in by_key.items() if k.startswith("stall angle"))
    assert "weight 0.00" in stall["metric"]          # counted for nothing
    assert "below the seed" in stall["metric"]
    ldmax = next(v for k, v in by_key.items() if k.startswith("(L/D) max"))
    assert "below the seed" not in ldmax["metric"]
