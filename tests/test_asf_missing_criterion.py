"""A weighted criterion the REFERENCE POINT does not carry: ``asf_missing``.

The hole this closes. The ASF's iteration domain is the reference point, not
the weights: every term is ``lambda_k (s_k - r_k)``, so a criterion with no
``r_k`` has no term at all. The plain composite and the goal composite both
still carry such a criterion inside ``J`` — the weight buys something there —
but ``J_asf`` IS NOT ``J``, so under the ASF the number the user typed bought
exactly nothing, and the weights panel went on showing it.

``airfoil_asf_missing`` decides what happens instead:

* ``"drop"`` — the default, the legacy arithmetic, and what every published
  ASF run was measured under. The criterion leaves the objective and is
  RECORDED in ``dropped`` so a card can say so.
* ``"band_floor"`` — the criterion is held to the bottom of the frozen band
  (the raw value that scores 0 sub-score points) and re-enters the aggregates.

What these tests pin, as OUTCOMES rather than as a restatement of the
arithmetic — every assertion is on a number ``asf_evaluation`` returned, and
the comparison is between two DESIGNS, so a mutant that drops the floored rows
from the aggregates is visible:

* the default's gradient on a missing criterion is EXACTLY zero — two designs
  that differ only in cruise L/D score bit-for-bit the same ``composite_asf``;
* ``band_floor``'s is ``rho * w_k`` and strictly positive, so the better design
  wins — that is "the weight binds", measured;
* the floor cannot HIJACK the max-min: the binding criterion and the ``min``
  are unchanged, because a floored term sits ~100 ``w_k`` above the
  seed-relative ones;
* the floor never pre-empts the empty-reference FALLBACK, and the test computes
  the plain composite's own gradient to show why flooring there would defend
  the same weights ~20x LESS;
* a criterion weighted 0 is still dropped and never floored — membership is
  the weights' decision under either mode.

The XFOIL sweeps are faked (tests/test_composite_asf_objective.py's
convention): the branch under test is the scoring policy, not the
boundary-layer march.
"""

from __future__ import annotations

import numpy as np
import pytest

from aerobo import api, xfoil_run
from aerobo.airfoil import AirfoilProblem
from aerobo.airfoil_select import (
    ASF_MISSING_MODES,
    CRITERIA,
    DEFAULT_ASF_MISSING,
    LOWER_BETTER,
    PRESETS,
    RHO_ASF,
    ScoreGoals,
    asf_evaluation,
    asf_terms,
    band_floor_value,
    check_asf_missing,
    composite_evaluation,
    load_screen_reference,
    sub_score,
)

#: the gdp-sweep preset, spelled out so a test states the weight it multiplies
#: by instead of asking the code under test what it is
W_GDP = {"thick": 0.10, "clmax": 0.20, "ldmax": 0.15, "ldcr": 0.35,
         "astall": 0.00, "cm": 0.20,
         # appended criterion, ZERO in every GDP preset: the point of
         # spelling the vector out is that a preset cannot move under a
         # test, and a criterion arriving at weight 0 must not move it
         # either — this line is the assertion that it did not.
         "cdcr": 0.00}

#: the criteria the reference point below DOES carry. cd enters none of them,
#: which is what makes the two designs differ in the missing criteria ALONE.
REF_KEYS = ("thick", "clmax", "cm")
#: weighted, and absent from that reference point — the user's own example
#: ("no cd calculated for the seed but given a weight")
MISSING_KEYS = ("ldmax", "ldcr")


def _install_fake_xfoil(monkeypatch, *, cd0=0.006, cm=-0.04):
    """Cruise sweep vs the wide stall sweep, dispatched on the alpha range.

    ``cd0`` is the only knob the tests turn: it moves cruise L/D and best L/D
    and NOTHING else — not t/c, not cl_max, not Cm — so two evaluations at two
    cd0 differ exactly in the criteria the reference point is missing.
    """

    def fake(coords, re, mach, alphas, **kw):
        a = np.asarray(alphas, dtype=float)
        cl = (1.6 - 0.01 * (a - 14.0) ** 2 if a.max() > 10.5
              else 0.25 + 0.11 * a)
        cd = cd0 + 1e-4 * (a - 1.0) ** 2
        return xfoil_run.XfoilPolarResult(alpha_deg=a, cl=cl, cd=cd,
                                          cm=np.full_like(a, cm),
                                          n_requested=a.size)

    monkeypatch.setattr(xfoil_run, "run_xfoil_polar", fake)


def _goal_at(key, score_target, reference):
    """The RAW value whose sub-score is ``score_target`` — the band inverted."""
    lo, hi = reference.band(key)
    span = hi - lo
    if key in LOWER_BETTER:
        return hi - score_target * span / 100.0
    return lo + score_target * span / 100.0


def _partial_reference(seed_scores, reference, keys=REF_KEYS, tol=0.0):
    """A ScoreGoals over ``keys`` ONLY, sitting exactly at the seed's scores.

    Everything outside ``keys`` has no ``r_k`` at all — which is the state this
    whole module is about, reachable in production two ways: a seed whose polar
    could not produce the metric, and an ``airfoil_score_goals`` stated over a
    subset.
    """
    return ScoreGoals(
        goals={k: _goal_at(k, seed_scores[k], reference) for k in keys},
        penalty=0.0, tol=tol, source="test",
        scores={k: seed_scores[k] for k in keys})


def _evaluate(monkeypatch, cd0, missing, goals_from):
    """One ``asf_evaluation`` at ``cd0``, against a reference point built from
    the seed's scores at the BASELINE cd0 — so the reference is one fixed point
    and only the candidate moves."""
    _install_fake_xfoil(monkeypatch, cd0=cd0)
    prob = AirfoilProblem()
    ref, w = load_screen_reference(), PRESETS["gdp-sweep"]
    return asf_evaluation(prob.w0, prob, ref, w, goals_from, RHO_ASF,
                          missing=missing), ref, w


@pytest.fixture()
def baseline(monkeypatch):
    """(the seed's own scores at cd0 = 0.006, the frozen band, the weights)."""
    _install_fake_xfoil(monkeypatch, cd0=0.006)
    prob = AirfoilProblem()
    ref, w = load_screen_reference(), PRESETS["gdp-sweep"]
    out = composite_evaluation(prob.w0, prob, ref, w)
    assert out["composite"] is not None, "the fixture's seed must be scoreable"
    assert w.normalised() == pytest.approx(W_GDP), \
        "the preset moved: every literal below is stated against these weights"
    return out["scores"], ref, w


def _pair(monkeypatch, goals, missing):
    """The same design at two drag levels, scored under one reference point.

    Returns ``(worse, better)`` evaluations — ``better`` has the lower cd, so
    it is strictly ahead on BOTH missing criteria and identical on all three
    the reference point carries.
    """
    worse, _r, _w = _evaluate(monkeypatch, 0.006, missing, goals)
    better, _r, _w = _evaluate(monkeypatch, 0.005, missing, goals)
    assert worse["composite_asf"] is not None
    assert better["composite_asf"] is not None
    for k in MISSING_KEYS:
        assert better["scores"][k] > worse["scores"][k], \
            f"the fixture must move {k}: it is what 'the weight binds' is about"
    for k in REF_KEYS:
        assert better["scores"][k] == pytest.approx(worse["scores"][k]), \
            f"the fixture must NOT move {k}: it is in the reference point"
    return worse, better


# ------------------------------------------------------- the mode is checked


def test_the_default_is_the_legacy_drop():
    assert DEFAULT_ASF_MISSING == "drop"
    assert check_asf_missing(None) == "drop"
    assert set(ASF_MISSING_MODES) == {"drop", "band_floor"}


def test_an_unknown_mode_is_refused_rather_than_silently_defaulted():
    with pytest.raises(ValueError, match="asf missing-criterion mode"):
        check_asf_missing("impute")


def test_the_band_floor_is_the_value_that_scores_zero():
    """``band_floor_value`` inverts :func:`sub_score` at 0 — for a
    higher-better criterion (``lo``) and for the mirrored lower-better one
    (``hi``), which is the pair a sign error would get exactly backwards."""
    ref = load_screen_reference()
    for key in CRITERIA:
        assert sub_score(key, band_floor_value(key, ref), ref) == \
            pytest.approx(0.0, abs=1e-9)
    assert band_floor_value("cm", ref) == pytest.approx(ref.band("cm")[1])
    assert band_floor_value("ldcr", ref) == pytest.approx(ref.band("ldcr")[0])


# ------------------------------------------------------------ the hole itself


def test_by_default_a_missing_criterion_buys_exactly_nothing(
        monkeypatch, baseline):
    """THE DEFECT, measured. Two designs, one 17 % better on cruise L/D and on
    best L/D, both carrying a 0.35 and a 0.15 weight — and the objective
    cannot tell them apart. Bit-for-bit, not to a tolerance."""
    seed_scores, ref, _w = baseline
    goals = _partial_reference(seed_scores, ref)
    worse, better = _pair(monkeypatch, goals, "drop")

    assert better["composite_asf"] == worse["composite_asf"]
    assert set(better["asf"]["rows"]) == set(REF_KEYS)
    assert set(better["asf"]["dropped"]) == set(MISSING_KEYS)
    assert not better["asf"]["floored"]


def test_band_floor_makes_the_missing_weight_bind(monkeypatch, baseline):
    """…and the fix, measured the same way: the better design now WINS, by
    exactly the augmentation's own exchange rate ``rho * sum_k w_k ds_k`` over
    the two floored criteria.

    The expectation is built from the sub-scores the evaluation returned, not
    from a second copy of the scoring map, so a band that moves does not move
    this test's answer."""
    seed_scores, ref, _w = baseline
    goals = _partial_reference(seed_scores, ref)
    worse, better = _pair(monkeypatch, goals, "band_floor")

    gain = better["composite_asf"] - worse["composite_asf"]
    expected = RHO_ASF * sum(
        W_GDP[k] * (better["scores"][k] - worse["scores"][k])
        for k in MISSING_KEYS)
    assert gain > 0.0, "the whole point: the weight has to buy something"
    assert gain == pytest.approx(expected, abs=1e-9)
    assert set(better["asf"]["rows"]) == set(REF_KEYS) | set(MISSING_KEYS)
    assert set(better["asf"]["floored"]) == set(MISSING_KEYS)
    assert not better["asf"]["dropped"]
    assert all(better["asf"]["rows"][k]["floored"] for k in MISSING_KEYS)
    assert not any(better["asf"]["rows"][k]["floored"] for k in REF_KEYS)


def test_the_floor_cannot_hijack_the_max_min(monkeypatch, baseline):
    """A floored term sits ~100 ``w_k`` above the seed-relative ones, so it
    must not become the criterion the Tchebycheff is held to — the binding
    criterion and the ``min`` are the ones the reference point chose, and the
    floor is visible ONLY in the augmentation.

    Stated because the docstring promises it and a reader would otherwise have
    to take the scale argument on trust."""
    seed_scores, ref, _w = baseline
    goals = _partial_reference(seed_scores, ref)
    dropped, _b = _pair(monkeypatch, goals, "drop")
    floored, _b2 = _pair(monkeypatch, goals, "band_floor")

    assert floored["asf"]["worst"] == dropped["asf"]["worst"]
    assert floored["asf"]["worst"] in REF_KEYS
    assert floored["asf"]["min"] == pytest.approx(dropped["asf"]["min"])
    assert floored["asf"]["sum"] > dropped["asf"]["sum"]


def test_a_zero_weight_criterion_is_never_floored(monkeypatch, baseline):
    """Membership is the WEIGHTS' decision under either mode. ``astall`` is
    weighted 0 in the gdp-sweep preset and is absent from the reference point,
    so it is the one criterion that qualifies for the floor on every count
    except the one that matters."""
    seed_scores, ref, _w = baseline
    goals = _partial_reference(seed_scores, ref)
    out, _r, _w2 = _evaluate(monkeypatch, 0.006, "band_floor", goals)

    assert W_GDP["astall"] == 0.0
    assert "astall" not in out["asf"]["rows"]
    assert "astall" not in out["asf"]["floored"]
    assert "astall" not in out["asf"]["dropped"]


# --------------------------------------------------- the fallback is not lost


def test_the_floor_never_pre_empts_the_empty_reference_fallback(
        monkeypatch, baseline):
    """An unscoreable seed already falls back to the PLAIN COMPOSITE, and this
    test computes why flooring there would be a downgrade: the plain
    composite's gradient on those same weights is ``w_k``, the floor's is
    ``rho * w_k`` — a factor of 1/rho = 20 at the shipped value.

    So ``band_floor`` leaves the fallback alone, and says so in ``dropped``.
    """
    _seed_scores, _ref, _w = baseline
    empty = ScoreGoals(goals={}, penalty=0.0, tol=0.0,
                       source="seed (not scoreable: no_cl_bracket)")
    worse, better = _pair(monkeypatch, empty, "band_floor")

    assert better["asf"]["min"] is None
    assert better["asf"]["fallback"]
    assert better["composite_asf"] == better["composite"]
    assert not better["asf"]["floored"]
    assert set(better["asf"]["dropped"]) == {
        k for k in CRITERIA if W_GDP[k] > 0.0}
    assert "falls back to the plain composite" in \
        better["asf"]["dropped"]["ldcr"]

    # …and the measurement behind the rule
    plain_gain = better["composite_asf"] - worse["composite_asf"]
    floor_gain = RHO_ASF * sum(
        W_GDP[k] * (better["scores"][k] - worse["scores"][k])
        for k in MISSING_KEYS)
    assert plain_gain > floor_gain / RHO_ASF * 0.9, \
        "the fallback must defend the weights at least as hard as w_k does"
    assert plain_gain > floor_gain


def test_the_default_arithmetic_is_untouched(monkeypatch, baseline):
    """Every published ASF run was measured at ``drop``. Calling ``asf_terms``
    without the new argument and with it set to the default must be one
    function, on a reference point that is COMPLETE as well as on a partial
    one."""
    seed_scores, ref, w = baseline
    for keys in (CRITERIA, REF_KEYS):
        goals = _partial_reference(seed_scores, ref, keys=keys)
        legacy = asf_terms(seed_scores, goals, ref, w, "weights")
        stated = asf_terms(seed_scores, goals, ref, w, "weights",
                           missing="drop")
        assert legacy["min"] == stated["min"]
        assert legacy["sum"] == stated["sum"]
        assert legacy["worst"] == stated["worst"]
        assert set(legacy["rows"]) == set(stated["rows"])
        assert legacy["dropped"] == stated["dropped"]


# ------------------------------------------------------------ the flag itself


def test_the_flag_is_declared_only_when_it_is_not_the_default():
    """Same rule as the objective, the censoring and ``asf_lambda``: a run at
    the legacy setting sends no flag, so every stored ASF config stays
    byte-for-byte what it was measured as."""
    assert "airfoil_asf_missing" not in api.airfoil_run_config(
        objective="composite_asf", score_weights="gdp-sweep").flags
    assert "airfoil_asf_missing" not in api.airfoil_run_config(
        objective="composite_asf", score_weights="gdp-sweep",
        asf_missing="drop").flags
    cfg = api.airfoil_run_config(objective="composite_asf",
                                 score_weights="gdp-sweep",
                                 asf_missing="band_floor")
    assert cfg.flags["airfoil_asf_missing"] == "band_floor"


@pytest.mark.parametrize("objective", ["composite", "composite_goal",
                                       "pareto"])
def test_the_flag_is_refused_on_an_objective_that_cannot_read_it(objective):
    """There is no reference point to be missing from in a weighted sum, and a
    hinge already carries the criterion inside J. Stated there, it RAISES —
    the same table and the same rule as ``asf_rho`` on a goal run."""
    with pytest.raises(ValueError, match="airfoil_asf_missing"):
        api.airfoil_run_config(objective=objective, score_weights="gdp-sweep",
                               asf_missing="band_floor")


def test_an_unknown_mode_is_refused_before_any_xfoil_runs():
    with pytest.raises(ValueError, match="asf missing-criterion mode"):
        api.airfoil_run_config(objective="composite_asf",
                               score_weights="gdp-sweep",
                               asf_missing="impute")


def test_the_built_problem_honours_the_flag(monkeypatch, baseline):
    """The flag has to reach the EVALUATION, not merely the config: an
    accepted flag that no evaluation reads is the exact bug class
    ``check_flags`` exists for."""
    seed_scores, _ref, _w = baseline
    cfg = api.airfoil_run_config(objective="composite_asf",
                                 score_weights="gdp-sweep",
                                 score_goals={k: _goal_at(
                                     k, seed_scores[k],
                                     load_screen_reference())
                                     for k in REF_KEYS},
                                 asf_missing="band_floor")
    api.check_flags(cfg.problem_name, cfg.flags)
    built = api.PROBLEM_SPECS[cfg.problem_name].build(
        cfg.mission_kwargs or {}, cfg.flags or {}, cfg.bounds_overrides)
    out = built.evaluate(AirfoilProblem().w0)
    assert out["asf_missing"] == "band_floor"
    assert set(out["asf"]["floored"]) == set(MISSING_KEYS)


# ------------------------------------------------------------- the shell says


def _card(working: dict, weights: dict):
    """The optimise view, with a finished ``composite_asf`` run on it.

    Same harness as
    ``tests/test_a_weighted_criterion_is_never_silently_dropped.py`` — kept
    here rather than imported so that file stays a statement about the
    DEFAULT and this one about the mode.
    """
    from gui.v3.app import assemble

    ctx = assemble("air")
    ctx.act("accept_mission")
    A = ctx.S["airfoil"]
    A["weights"] = dict(weights)
    A["opt"]["objective"] = "composite_asf"
    A["opt"]["report"] = {
        "conditions": {"objective": "composite_asf", "cl_design": 0.5},
        "result": {"best_x": [0.1] * 8, "best_score": 60.0, "n_evals": 24,
                   "budget": 24},
        "wall_time_s": 1.0,
        "design": {"breakdown": {"asf": working}},
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


def test_a_floored_criterion_gets_its_own_sentence_not_the_accusation():
    """"your weight bought nothing" is FALSE of a floored criterion — it
    bound, weakly. Reporting it beside the ones that really did buy nothing
    would be the same false charge the zero-weight rule exists to avoid, so
    the card says the other thing and names the weight."""
    texts = _card({"rows": {"clmax": {"term": 1.0, "weight": 0.2},
                            "ldcr": {"term": 19.25, "weight": 0.35,
                                     "floored": True}},
                   "min": 1.0, "sum": 20.25, "worst": "clmax",
                   "dropped": {},
                   "floored": {"ldcr": "the reference design has no value "
                                       "for it, so it is held to the bottom "
                                       "of the frozen band"}},
                  W_GDP)

    said = [t for t in texts if "BOTTOM OF THE FROZEN BAND" in t]
    assert said, texts
    assert "0.35" in said[0]
    assert "Their weights bind" in said[0]
    assert not any("carry a weight and were NOT" in t for t in texts), \
        "a floored criterion must not be reported as one that bought nothing"


def test_both_sentences_can_appear_on_one_run():
    """A criterion the CANDIDATE could not produce is still dropped even under
    ``band_floor`` — the two states are independent and the card carries
    both."""
    texts = _card({"rows": {"clmax": {"term": 1.0, "weight": 0.2}},
                   "min": 1.0, "sum": 1.0, "worst": "clmax",
                   "dropped": {"thick": "this design produced no value for it"},
                   "floored": {"ldcr": "the reference design has no value for "
                                       "it, so it is held to the bottom of "
                                       "the frozen band"}},
                  W_GDP)

    assert any("carry a weight and were NOT" in t for t in texts), texts
    assert any("BOTTOM OF THE FROZEN BAND" in t for t in texts), texts
