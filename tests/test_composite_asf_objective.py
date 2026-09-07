"""The augmented Tchebycheff of the six criteria: ``objective="composite_asf"``.

``composite_goal`` fixed the SALE — it prices a criterion sold below the seed.
It did not fix REACHABILITY, and that is a theorem rather than an oversight: a
weighted sum plus a hinge penalty is still a weighted sum, and a weighted sum
returns only points on the CONVEX HULL of the achievable set, at every weight
vector (LITERATURE_REVIEW_S44.md §1). ``composite_asf`` changes the FUNCTION:

    J_asf(x) = min_k [ w_k (s_k(x) - r_k) ] + rho * sum_k w_k (s_k(x) - r_k)

on the same frozen band, the same weights and the same ``ScoreGoals``
reference point. What these tests pin, as OUTCOMES rather than as a
restatement of the arithmetic:

* the seed scores EXACTLY zero, because the reference point carries the seed's
  own sub-scores;
* the ``min`` binds — a design that sells cruise L/D to buy |Cm| scores WORSE
  than one that is merely level, while the weighted sum of the very same terms
  ranks the two the other way round. That inversion is computed in the test,
  so "drop the min" is a mutation the test can see;
* the augmentation breaks weakly-Pareto ties that ``rho = 0`` cannot see, and
  ``rho = 0`` is reachable and asserted rather than merely documented;
* a criterion weighted at zero is DROPPED, not carried at ``w_k = 0`` — the
  latter would pin the worst-of at zero for every candidate and switch the
  Tchebycheff term off. The mutant's answer is computed alongside the real one;
* the plain composite's flag set and its number are untouched;
* ``airfoil_goal_penalty`` (no multiplier in an achievement function) and
  ``airfoil_asf_rho`` (no augmentation in a hinge) are each refused on the
  other objective, at config time AND at build time.

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
    PRESETS,
    RHO_ASF,
    ScoreGoals,
    ScoreWeights,
    asf_evaluation,
    asf_objective,
    composite_evaluation,
    composite_objective,
    load_screen_reference,
    seed_goals,
)
from gui.v3.stages import airfoil as stage

WING = {"mass_kg": 12.0, "v_ms": 20.0, "altitude_m": 0.0,
        "s_ref_m2": 1.0, "aspect_ratio": 8.0, "taper": 0.6}

#: the gdp-sweep preset, spelled out here so a test states the weight it
#: multiplies by instead of asking the code under test what it is
W_GDP = {"thick": 0.10, "clmax": 0.20, "ldmax": 0.15, "ldcr": 0.35,
         "astall": 0.00, "cm": 0.20,
         # appended criterion, ZERO in every GDP preset: the point of
         # spelling the vector out is that a preset cannot move under a
         # test, and a criterion arriving at weight 0 must not move it
         # either — this line is the assertion that it did not.
         "cdcr": 0.00}


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
    assert w.normalised() == pytest.approx(W_GDP), \
        "the preset moved: every literal below is stated against these weights"
    return prob, ref, w, out


def _goal_at(key, score_target, reference):
    """The RAW value whose sub-score is ``score_target`` — the band inverted.

    Lets a test say "the reference point sits ten points below where this
    design landed" and then assert the achievement of exactly those ten
    points, without restating the scoring map to build the expectation.
    """
    lo, hi = reference.band(key)
    span = hi - lo
    if key == "cm":                      # lower-better: mirrored
        return hi - score_target * span / 100.0
    return lo + score_target * span / 100.0


def _reference_below(seed_scores, reference, delta=None, tol=0.0):
    """A ScoreGoals sitting ``delta`` sub-score points BELOW the seed.

    The candidate under test is always the seed itself (``prob.w0``), because
    that is the one design these tests can evaluate for real without inventing
    an XFOIL polar. Moving the REFERENCE POINT instead of the candidate gives
    the same achievement vector — ``w_k (s_k - r_k) = w_k * delta_k`` — while
    keeping every assertion on the value the objective actually returns.

    That distinction is load-bearing: an earlier version of this file asserted
    on ``asf_terms``' aggregates and reduced them itself, and a mutant that
    dropped the ``min`` from ``asf_evaluation`` altogether SURVIVED the whole
    file. A test that restates the reduction cannot see the reduction change.
    """
    delta = delta or {}
    target = {k: seed_scores[k] - float(delta.get(k, 0.0)) for k in CRITERIA}
    return ScoreGoals(
        goals={k: _goal_at(k, target[k], reference) for k in CRITERIA},
        penalty=0.0, tol=tol, source="test", scores=target)


# ----------------------------------------------------------- the arithmetic


def test_the_seed_scores_exactly_zero(setup):
    """The reference point IS the seed, so the seed's achievement is nil —
    exactly, not to a tolerance. The goals carry the sub-scores the composite
    itself produced, so the CST-vs-geometric t/c gap cannot fake a
    ten-thousandth of a point of achievement at the point it is measured
    from."""
    prob, ref, w, _seed = setup
    goals = seed_goals(prob, ref, w, penalty=0.0)
    out = asf_evaluation(prob.w0, prob, ref, w, goals)

    assert out["composite_asf"] == 0.0
    assert out["asf"]["min"] == 0.0 and out["asf"]["sum"] == 0.0
    # …and the plain composite underneath is still reported, untouched
    assert out["composite"] == pytest.approx(
        sum(W_GDP[k] * out["scores"][k] for k in CRITERIA), abs=1e-9)


def test_the_scalar_is_the_augmented_tchebycheff(setup):
    """One design, six stated offsets, and the answer written out in full.

    thick +6 -> 0.60   clmax  -4 -> -0.80   ldmax +11 -> +1.65
    ldcr -1.5 -> -0.525   cm   +2 -> +0.40   astall +30 -> DROPPED (w = 0)

    min = -0.80 (clmax), sum = +1.325, J_asf = -0.80 + 0.05 x 1.325

    Asserted on what ``asf_evaluation`` RETURNS, not on a reduction this test
    performs itself.
    """
    prob, ref, w, seed = setup
    goals = _reference_below(seed["scores"], ref,
                             {"thick": 6.0, "clmax": -4.0, "ldmax": 11.0,
                              "ldcr": -1.5, "astall": 30.0, "cm": 2.0})
    out = asf_evaluation(prob.w0, prob, ref, w, goals)
    work = out["asf"]

    assert set(work["rows"]) == {"thick", "clmax", "ldmax", "ldcr", "cm"}
    assert work["worst"] == "clmax"
    assert work["min"] == pytest.approx(-0.80, abs=1e-9)
    assert work["sum"] == pytest.approx(+1.325, abs=1e-9)
    assert out["composite_asf"] == pytest.approx(-0.73375, abs=1e-9)


def test_the_worst_criterion_binds_and_a_weighted_sum_would_not(setup):
    """THE reason this objective exists, as an inversion.

    The "seller" buys 40 points of |Cm| with 10 points of cruise L/D — the
    exact trade the frozen study measured the composite making. The "level"
    design merely reproduces the reference point. A weighted sum of the SAME
    terms scores the seller +4.50 against the level design's 0.00 and prefers
    it; the augmented Tchebycheff scores it -3.275 and refuses it.

    Both columns are read off the objective's own return, so "drop the min and
    keep the augmentation" is a mutation this test can SEE: that mutant scores
    the seller +0.225 and puts it back on top.
    """
    prob, ref, w, seed = setup

    def evaluated(delta):
        return asf_evaluation(prob.w0, prob, ref, w,
                              _reference_below(seed["scores"], ref, delta))

    seller = evaluated({"cm": 40.0, "ldcr": -10.0})
    level = evaluated({})

    # what a WEIGHTED SUM of the same terms says — the mutant's answer
    assert seller["asf"]["sum"] == pytest.approx(+4.50, abs=1e-9)
    assert level["asf"]["sum"] == pytest.approx(0.0, abs=1e-9)
    assert seller["asf"]["sum"] > level["asf"]["sum"]

    # …and what the objective returns, the other way round
    assert seller["asf"]["worst"] == "ldcr"
    assert seller["composite_asf"] == pytest.approx(-3.275, abs=1e-9)
    assert level["composite_asf"] == pytest.approx(0.0, abs=1e-9)
    assert seller["composite_asf"] < level["composite_asf"]


def test_the_augmentation_breaks_a_weakly_pareto_tie(setup):
    """Two designs that bind at the same criterion, one dominating the other.

    Both sit 2 points of cruise L/D below the reference point (term -0.70);
    only one also gains 5 points of |Cm| (+1.00). The PURE Tchebycheff cannot
    tell them apart — that is what "weakly Pareto" means and why the
    augmentation is not optional. ``rho = 0`` is a reachable setting, so the
    degenerate case is MEASURED here rather than asserted in a comment.
    """
    prob, ref, w, seed = setup

    def evaluated(delta, rho):
        return asf_evaluation(prob.w0, prob, ref, w,
                              _reference_below(seed["scores"], ref, delta),
                              rho)["composite_asf"]

    weak, dom = {"ldcr": -2.0}, {"ldcr": -2.0, "cm": 5.0}

    assert evaluated(weak, 0.0) == evaluated(dom, 0.0)      # rho=0 is BLIND
    assert evaluated(weak, 0.0) == pytest.approx(-0.70, abs=1e-9)

    assert evaluated(weak, RHO_ASF) == pytest.approx(-0.735, abs=1e-9)
    assert evaluated(dom, RHO_ASF) == pytest.approx(-0.685, abs=1e-9)
    assert evaluated(dom, RHO_ASF) > evaluated(weak, RHO_ASF)


def test_a_zero_weight_criterion_is_dropped_not_held_at_zero(setup):
    """astall is 0.0 in the gdp-sweep preset.

    Carrying it into the min at ``w_k = 0`` would contribute a term that is
    identically zero, so the worst-of would be pinned at 0.00 for EVERY
    candidate and the Tchebycheff term would be silently off. Here a design 10
    points ahead on all six returns 1.50; the mutant that keeps the zero-weight
    term returns 0.50 and would rank it level with a design that gained nothing
    at all.
    """
    prob, ref, w, seed = setup
    goals = _reference_below(seed["scores"], ref, {k: 10.0 for k in CRITERIA})
    out = asf_evaluation(prob.w0, prob, ref, w, goals)

    assert "astall" not in out["asf"]["rows"]
    assert out["asf"]["worst"] == "thick"            # the smallest WEIGHT
    assert out["asf"]["min"] == pytest.approx(1.00, abs=1e-9)
    assert out["asf"]["sum"] == pytest.approx(10.00, abs=1e-9)
    assert out["composite_asf"] == pytest.approx(1.50, abs=1e-9)

    # the mutant's answer, stated so the difference is a number
    assert RHO_ASF * out["asf"]["sum"] == pytest.approx(0.50, abs=1e-9)


def test_a_large_weight_gets_LESS_attention_above_the_reference(setup):
    """The counter-intuitive property of this form, pinned because it is real.

    Above the reference point every term is positive, so the min sits at the
    SMALLEST product — and a heavily weighted criterion needs a smaller
    achievement to clear the same bar. A design +10 points on all five weighted
    criteria therefore binds at `thick` (w 0.10, product 1.00), not at `ldcr`
    (w 0.35, product 3.50); `ldcr` has to fall below +2.857 points before it
    can bind against `thick` at +10.

    The weighted sum overspends the criterion it weights HIGHEST (that is
    §16.3b's whole finding); this function attends to the one it weights
    LOWEST. Neither is what the weights say, and the second is a property of
    `lambda` being a SCALING coefficient in Wierzbicki's ASF rather than a
    preference weight. This test exists so that the behaviour is a recorded
    choice and not a surprise.
    """
    prob, ref, w, seed = setup

    flat = asf_evaluation(prob.w0, prob, ref, w,
                          _reference_below(seed["scores"],
                                           ref, {k: 10.0 for k in CRITERIA}))
    assert flat["asf"]["worst"] == "thick"

    # the exact crossover: at ldcr +2.857 the two products are equal
    tie = asf_evaluation(prob.w0, prob, ref, w, _reference_below(
        seed["scores"], ref,
        {k: 10.0 for k in CRITERIA} | {"ldcr": 1.00 / 0.35}))
    assert tie["asf"]["rows"]["ldcr"]["term"] == pytest.approx(1.00, abs=1e-9)

    # a shade below it and ldcr finally binds
    binds = asf_evaluation(prob.w0, prob, ref, w, _reference_below(
        seed["scores"], ref, {k: 10.0 for k in CRITERIA} | {"ldcr": 2.5}))
    assert binds["asf"]["worst"] == "ldcr"


def test_giving_the_zero_weight_criterion_a_weight_defends_it(setup):
    """The same lever, said once: a criterion nobody weighted is not in J and
    is not held here either — weight it and the max-min binds on it."""
    prob, ref, w, seed = setup
    goals = _reference_below(seed["scores"], ref,
                             {k: 10.0 for k in CRITERIA} | {"astall": -20.0})

    assert asf_evaluation(prob.w0, prob, ref, w,
                          goals)["asf"]["worst"] == "thick"

    w2 = ScoreWeights(thick=0.10, clmax=0.20, ldmax=0.15, ldcr=0.35, cm=0.20,
                      astall=0.20)
    held = asf_evaluation(prob.w0, prob, ref, w2, goals)
    assert held["asf"]["worst"] == "astall"
    assert held["composite_asf"] < 0.0
    assert held["asf"]["min"] == pytest.approx(
        w2.normalised()["astall"] * -20.0, abs=1e-9)


def test_tol_lowers_the_reference_point_by_that_many_points(setup):
    """``tol`` is the same slack it is on the goal composite: it shifts every
    reference sub-score DOWN by that many points, so a design AT the reference
    point gains ``w_k * tol`` on criterion k. min = 0.10 x tol (the smallest
    weight) and sum = 1.00 x tol (the weights that are not zero sum to one),
    so J_asf = 0.15 x tol exactly."""
    prob, ref, w, seed = setup
    out = asf_evaluation(prob.w0, prob, ref, w,
                         _reference_below(seed["scores"], ref, tol=2.0))

    assert out["asf"]["worst"] == "thick"
    assert out["asf"]["min"] == pytest.approx(0.20, abs=1e-9)
    assert out["asf"]["sum"] == pytest.approx(2.00, abs=1e-9)
    assert out["composite_asf"] == pytest.approx(0.30, abs=1e-9)


def test_rho_scales_only_the_augmentation(setup):
    """The two terms are separable and only one of them moves with rho: at the
    same design, J_asf(rho) is affine in rho with slope ``sum`` and intercept
    ``min``. Measured at three values rather than asserted."""
    prob, ref, w, seed = setup
    goals = _reference_below(seed["scores"], ref,
                             {"thick": 6.0, "clmax": -4.0, "ldmax": 11.0,
                              "ldcr": -1.5, "cm": 2.0})

    def at(rho):
        return asf_evaluation(prob.w0, prob, ref, w, goals,
                              rho)["composite_asf"]

    assert at(0.0) == pytest.approx(-0.80, abs=1e-9)          # the min alone
    assert at(0.05) == pytest.approx(-0.73375, abs=1e-9)
    assert at(0.20) == pytest.approx(-0.80 + 0.20 * 1.325, abs=1e-9)


def test_the_scalar_is_never_reported_as_a_composite(setup):
    """J_asf is a weighted sub-score DIFFERENCE, not a J. A run that mixed the
    two would compare an ~0 against a ~68 and call one of them better."""
    prob, ref, w, seed = setup
    goals = _reference_below(seed["scores"], ref)
    out = asf_evaluation(prob.w0, prob, ref, w, goals)

    assert out["composite"] > 50.0                  # the plain J, untouched
    assert out["composite_asf"] == 0.0
    assert out["score"] == out["f"] == out["composite_asf"]
    j, _g = composite_objective(prob.w0, prob, ref, w)
    assert j == out["composite"] != out["composite_asf"]


def test_the_constraint_channel_is_the_composites_own(setup):
    """Same margins, so an ASF run and a composite run share one feasible
    region and their winners compare candidate for candidate."""
    prob, ref, w, seed = setup
    goals = _reference_below(seed["scores"], ref)
    j_asf, g_asf = asf_objective(prob.w0, prob, ref, w, goals)
    j_c, g_c = composite_objective(prob.w0, prob, ref, w)

    assert np.allclose(g_asf, g_c)
    assert j_asf == 0.0 and j_c == pytest.approx(seed["composite"])


def test_an_empty_reference_point_falls_back_and_says_so(setup):
    """An unscoreable seed leaves the achievement undefined. Falling back to
    the plain composite is a decision about the BUILT PROBLEM (the goals are
    resolved once), so a run never changes scale halfway through — but it must
    be readable, not silent."""
    prob, ref, w, seed = setup
    out = asf_evaluation(prob.w0, prob, ref, w,
                         ScoreGoals(goals={}, penalty=0.0, source="empty"))

    assert out["composite_asf"] == pytest.approx(seed["composite"])
    assert out["asf"]["fallback"]
    assert out["asf"]["min"] is None


def test_a_negative_augmentation_is_refused_not_clipped(setup):
    """rho < 0 would reward being WORSE on the criteria that are not binding."""
    prob, ref, w, seed = setup
    goals = _reference_below(seed["scores"], ref)
    with pytest.raises(ValueError, match="rho"):
        asf_evaluation(prob.w0, prob, ref, w, goals, -0.01)
    with pytest.raises(ValueError, match="rho"):
        asf_evaluation(prob.w0, prob, ref, w, goals, float("nan"))


# ------------------------------------------------- the frozen composite


def test_the_plain_composite_run_is_untouched_flag_for_flag():
    """Adding a fourth objective must not move the one every published number
    depends on: the composite config still sends exactly the keys it sent, and
    a default call still sends the legacy five."""
    cfg = api.airfoil_run_config(objective="composite",
                                 score_weights="gdp-sweep")
    assert set(cfg.flags) == {"airfoil_re", "airfoil_mach",
                              "airfoil_cl_design", "airfoil_tc_min",
                              "airfoil_cm_max", "airfoil_objective",
                              "airfoil_score_weights"}
    assert api.airfoil_run_config().flags == {
        "airfoil_re": 1e6, "airfoil_mach": 0.0, "airfoil_cl_design": 0.5,
        "airfoil_tc_min": 0.10, "airfoil_cm_max": 0.08}
    # …and in particular the composite does NOT pick up the warm start the two
    # reference-point objectives switch on for themselves
    assert api.BO_WARM_START_FLAG not in cfg.flags


def test_the_plain_composite_is_still_the_weighted_sum(setup):
    """``composite_objective`` is FROZEN. Its number is the weighted sum of
    the six sub-scores and nothing in this module's new half touches it."""
    prob, ref, w, seed = setup
    j, _g = composite_objective(prob.w0, prob, ref, w)

    assert j == seed["composite"]
    assert j == pytest.approx(sum(W_GDP[k] * seed["scores"][k]
                                  for k in CRITERIA), abs=1e-9)


# ------------------------------------------------------- the flag ownership


@pytest.mark.parametrize("objective", ["cd", "composite", "composite_goal"])
def test_the_augmentation_is_refused_where_nothing_reads_it(objective):
    """``airfoil_asf_rho`` weights a term only the ASF has. A flag that is
    silently dropped is worse than one that is refused."""
    with pytest.raises(ValueError, match="airfoil_asf_rho"):
        api.airfoil_run_config(objective=objective, asf_rho=0.1)


def test_the_penalty_is_refused_on_the_asf():
    """There is no multiplier in an achievement function — the min does the
    work a penalty was standing in for — so stating one raises rather than
    being accepted and ignored."""
    with pytest.raises(ValueError, match="airfoil_goal_penalty"):
        api.airfoil_run_config(objective="composite_asf", goal_penalty=3.0)


def test_the_reference_point_flags_ARE_shared():
    """The two objectives differ in what they do with the reference point, not
    in whether they have one: goals and tol belong to both."""
    cfg = api.airfoil_run_config(objective="composite_asf",
                                 score_goals={"ldcr": 80.0}, goal_tol=1.0,
                                 asf_rho=0.02)
    assert cfg.flags["airfoil_score_goals"] == {"ldcr": 80.0}
    assert cfg.flags["airfoil_goal_tol"] == 1.0
    assert cfg.flags["airfoil_asf_rho"] == 0.02


def test_rho_is_declared_only_when_it_is_stated():
    """An untouched ASF call must not send a flag the default already means —
    the same rule the objective, the censoring and the refusal all follow."""
    assert "airfoil_asf_rho" not in api.airfoil_run_config(
        objective="composite_asf").flags
    assert api.airfoil_run_config(objective="composite_asf",
                                  asf_rho=0.0).flags["airfoil_asf_rho"] == 0.0


@pytest.mark.parametrize("flags,match", [
    ({"airfoil_objective": "composite_asf", "airfoil_goal_penalty": 3.0},
     "airfoil_goal_penalty"),
    ({"airfoil_objective": "composite_goal", "airfoil_asf_rho": 0.1},
     "airfoil_asf_rho"),
    ({"airfoil_objective": "composite", "airfoil_asf_rho": 0.1},
     "airfoil_asf_rho"),
])
def test_a_hand_built_config_is_refused_at_build_time_too(flags, match):
    """Every study script assembles a RunConfig directly, so a term stated on
    an objective that does not read it must be refused where the problem is
    BUILT as well — that is where nobody is looking."""
    with pytest.raises(ValueError, match=match):
        api.PROBLEM_SPECS["airfoil (section)"].build({}, flags, None)


def test_a_negative_rho_is_refused_at_the_surface_that_takes_it():
    with pytest.raises(ValueError, match="rho"):
        api.airfoil_run_config(objective="composite_asf", asf_rho=-1.0)


# ------------------------------------------------------------- the plumbing


def test_the_built_problem_evaluates_the_asf(monkeypatch):
    _install_fake_xfoil(monkeypatch)
    cfg = api.airfoil_run_config(objective="composite_asf",
                                 score_weights="gdp-sweep")
    built = api.PROBLEM_SPECS[cfg.problem_name].build(
        cfg.mission_kwargs or {}, cfg.flags or {}, cfg.bounds_overrides)
    prob = AirfoilProblem()

    out = built.evaluate(prob.w0)
    assert out["composite_asf"] == 0.0            # the seed IS the reference
    assert out["score"] == out["f"] == out["composite_asf"]
    assert out["asf_rho"] == RHO_ASF
    j, g = built.callable(prob.w0)
    assert j == 0.0 and np.all(np.asarray(g) >= 0.0)


def test_the_reference_point_is_resolved_once_per_built_problem(monkeypatch):
    """A moving reference point is a moving objective — the same reason the
    band is frozen, and the same guarantee the goal composite already holds."""
    _install_fake_xfoil(monkeypatch)
    calls = {"n": 0}
    real = api._resolve_airfoil_goals

    def counted(*a, **kw):
        calls["n"] += 1
        return real(*a, **kw)

    monkeypatch.setattr(api, "_resolve_airfoil_goals", counted)
    cfg = api.airfoil_run_config(objective="composite_asf")
    built = api.PROBLEM_SPECS[cfg.problem_name].build(
        cfg.mission_kwargs or {}, cfg.flags or {}, cfg.bounds_overrides)

    assert calls["n"] == 0                    # building must stay cheap
    prob = AirfoilProblem()
    for _ in range(3):
        built.callable(prob.w0)
    assert calls["n"] == 1


def test_an_asf_reference_point_carries_no_multiplier(monkeypatch):
    """A stored ASF run reporting penalty 10 could be read as having applied
    one. It applies none, so it says none."""
    _install_fake_xfoil(monkeypatch)
    prob, ref = AirfoilProblem(), load_screen_reference()
    w = PRESETS["gdp-sweep"]
    cfg = api.airfoil_run_config(objective="composite_asf")
    goals = api._resolve_airfoil_goals(prob, cfg.flags, ref, w, "refuse",
                                       "composite_asf")
    assert goals.penalty == 0.0 and goals.source == "seed"


def test_the_seed_is_in_the_training_set():
    """Both reference-point objectives are defined RELATIVE to the seed, so a
    search that never evaluates it can still return something worse than it."""
    for name in ("composite_goal", "composite_asf"):
        cfg = api.airfoil_run_config(objective=name)
        assert cfg.flags[api.BO_WARM_START_FLAG] is True


def test_the_report_states_what_it_was_measured_from(monkeypatch):
    """The reference point and the augmentation are as much a part of the
    scalar as the band: a stored run says which function it maximised."""
    _install_fake_xfoil(monkeypatch)
    rep = api.optimize_airfoil(objective="composite_asf",
                               score_weights="gdp-sweep", asf_rho=0.02,
                               optimiser="random", budget=3, seed=0)

    assert rep["conditions"]["objective"] == "composite_asf"
    score = rep["conditions"]["score"]
    assert score["goals"]["source"] == "seed"
    assert score["goals"]["penalty"] == 0.0
    assert score["asf_rho"] == 0.02
    assert rep["design"]["breakdown"]["composite_asf"] == pytest.approx(
        rep["result"]["best_score"])


def test_the_asf_inherits_the_composites_two_refusals():
    """Same 2-D section problem, so wing mode is refused for the same reason:
    a twist law searched against a number that cannot see it."""
    with pytest.raises(ValueError, match="2-D SECTION"):
        api.airfoil_run_config(objective="composite_asf", wing=WING)


def test_the_budget_plan_charges_the_composites_factor(monkeypatch):
    """Same six criteria, same two XFOIL sweeps, same design vector — a
    max-min of six landscapes is no cheaper to search than their sum, so it is
    charged the composite's factor rather than the family's own line. The
    factor is BORROWED and neither reference-point objective has a study of its
    own.

    Two facts stated rather than hidden. First, the SECTION kind is time-boxed
    (``budget_mode`` "time"), so the factor never reaches its budget at all
    and every objective plans the same number there — that is measured below,
    not assumed. Second, the shipped factor is 1.0 and UNMEASURED
    (``n_pairs`` 0), so an equality would pass whether or not the name reached
    the branch. The branch is therefore exercised where it applies — the
    law-mode ``wing`` kind, with the factor perturbed to 2.0 — which is the
    assertion that can actually fail.
    """
    import copy

    from aerobo.optimize import budget as B

    def plan(objective, kind, path=None):
        return B.recommend(dim=8, kind=kind, constrained=True,
                           objective=objective)

    # the section: time-boxed, so the objective does not move the budget
    assert (plan("composite_asf", "airfoil").budget
            == plan("composite", "airfoil").budget
            == plan("cd", "airfoil").budget)

    doubled = copy.deepcopy(B.payload())
    B._kind_block("wing", doubled)["objective_factor"] = {
        "composite": 2.0, "measured": True, "n_pairs": 1, "ratio": 2.0}
    monkeypatch.setattr(B, "payload", lambda path=None: doubled)

    own = plan("cd", "wing").budget
    charged = {name: plan(name, "wing").budget
               for name in ("composite", "composite_goal", "composite_asf")}
    # all three take the SAME branch, and it is not the family's own line
    assert len(set(charged.values())) == 1, charged
    assert charged["composite_asf"] == pytest.approx(2 * own, abs=1)
    assert charged["composite_asf"] > own


# --------------------------------------------------------------- the shell


def test_the_menu_offers_the_asf_and_calls_it_composite():
    for wing_mode in (True, False):
        choices = stage.OBJECTIVE_CHOICES(wing_mode)
        assert "composite_asf" in choices
        assert "Tchebycheff" in choices["composite_asf"]
    assert stage.is_composite("composite_asf")
    assert set(stage.COMPOSITE_OBJECTIVES) == set(
        api.AIRFOIL_COMPOSITE_OBJECTIVES)


def test_the_asf_never_travels_with_a_wing_guess():
    kw = stage.objective_kwargs("composite_asf", wing_guess=WING,
                                weights={"clmax": 1.0}, reference=None)
    assert kw["objective"] == "composite_asf" and kw["wing"] is None


def test_the_shell_plans_an_asf_run_as_the_section_run_it_is():
    """The stale-branch bug class, one objective on: reading a literal tuple
    of names planned a goal run as a -cd run once already. The plan must match
    the plain composite's, not the -cd one's."""
    from gui.v3 import session

    S = session.make_session("air")
    A = session.airfoil_state(S, "main")

    A["opt"]["objective"] = "composite"
    plain = session.airfoil_plan(S, "main")
    A["opt"]["objective"] = "composite_asf"
    asf = session.airfoil_plan(S, "main")
    A["opt"]["objective"] = "cd"
    own = session.airfoil_plan(S, "main")

    assert plain is not None and asf is not None and own is not None
    assert asf.budget == plain.budget
    assert asf.dim == plain.dim == 8                # a SECTION, not a wing
    assert asf.objective == "composite_asf"
    assert (own.dim, own.budget) != (asf.dim, asf.budget)
