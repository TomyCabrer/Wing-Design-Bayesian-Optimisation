"""A right-censored ``cl_max`` is a LOWER BOUND, not a solver failure.

The stall sweep stops at the last alpha XFOIL converged. When ``cl`` is still
rising there, ``cl_max`` has not been measured — it has been bounded below.
Until now the composite treated that as a refusal and threw the observation
away, which is exactly the arm Hutter, Hoos & Leyton-Brown (*BO With Censored
Response Data*) measure as the worst of the three available handlings — worse
than treating a censored value as uncensored, and far worse than imputing it
from the model's predictive distribution truncated at the bound.

What these tests pin:

* the default (``"refuse"``) is bit-for-bit the behaviour every frozen study
  was run under — same call, same number, and the mode cannot be changed by
  accident;
* ``"lower_bound"`` keeps the candidate and scores it at the bound, flagging
  that the number is a bound (``clmax_lower_bound``);
* the soundness claim itself: J is monotone non-decreasing in ``clmax`` and in
  ``astall``, the only two censored criteria, so scoring at the bound gives a
  true lower bound on J for ANY non-negative weight vector — not a guess;
* an unknown mode is refused by name rather than silently defaulted.

The censoring is forced by monkeypatching ``polar_metrics`` rather than by
hunting for a section XFOIL happens to censor: the branch under test is the
scoring policy, not the boundary-layer march. The XFOIL calls underneath run
against the shared cache.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from aerobo import airfoil_select as A
from aerobo.airfoil import PENALTY, AirfoilProblem
from aerobo.airfoil_select import (
    CENSORED_MODES,
    DEFAULT_CENSORED,
    PRESETS,
    check_censored,
    composite_evaluation,
    composite_objective,
    load_screen_reference,
    make_composite_fg,
)

WEIGHTS = PRESETS["gdp-sweep"]
#: the frozen stage-3 study's base (hg40) — the anchor every composite number
#: in the report was measured at. A bare ``AirfoilProblem()`` anchors on NACA
#: 2412 instead and scores 59.32, so a test that pins 68.124 must say which.
FROZEN = Path(__file__).resolve().parents[1] / "results" / "airfoil_pipeline.json"
SEED_J = 68.12429540293857


@pytest.fixture(scope="module")
def setup():
    """(problem anchored on hg40, frozen reference, GDP bulk-sweep weights)."""
    if not FROZEN.exists():
        pytest.skip(f"{FROZEN.name} not present: no frozen anchor to pin")
    base = json.loads(FROZEN.read_text())["base"]
    prob = AirfoilProblem(anchor=(np.array(base["w_upper"]),
                                  np.array(base["w_lower"])))
    return prob, load_screen_reference(), WEIGHTS


def _force_censored(monkeypatch, censored=True):
    """Make every scored record report a censored cl_max."""
    real = A.polar_metrics

    def patched(*a, **kw):
        rec = dict(real(*a, **kw))
        rec["clmax_censored"] = censored
        return rec

    monkeypatch.setattr(A, "polar_metrics", patched)


# ------------------------------------------------------------------ the modes

def test_the_default_is_the_legacy_refusal():
    assert DEFAULT_CENSORED == "refuse"
    assert set(CENSORED_MODES) == {"refuse", "lower_bound"}
    assert check_censored(None) == "refuse"


def test_an_unknown_mode_is_refused_by_name():
    with pytest.raises(ValueError, match="unknown censored mode"):
        check_censored("impute_it_somehow")


def test_refuse_throws_the_observation_away(setup, monkeypatch):
    prob, ref, w = setup
    _force_censored(monkeypatch)
    out = composite_evaluation(prob.w0, prob, ref, w)          # default
    assert out["composite"] is None
    assert out["feasible"] is False
    assert "censored" in out["reason"]
    assert out["score"] == out["f"] == PENALTY


def test_lower_bound_keeps_it_and_says_it_is_a_bound(setup, monkeypatch):
    prob, ref, w = setup
    _force_censored(monkeypatch)
    out = composite_evaluation(prob.w0, prob, ref, w, censored="lower_bound")
    assert out["composite"] is not None
    assert np.isfinite(out["composite"])
    assert out["feasible"] is True
    assert out["clmax_lower_bound"] is True
    assert out["score"] == out["f"] == out["composite"]


def test_an_uncensored_evaluation_is_not_flagged_as_a_bound(setup):
    prob, ref, w = setup
    out = composite_evaluation(prob.w0, prob, ref, w, censored="lower_bound")
    assert out["clmax_lower_bound"] is False
    assert out["composite"] == pytest.approx(SEED_J, abs=1e-9)


# --------------------------------------------------- the default is unchanged

def test_the_default_path_is_bit_for_bit(setup):
    """An unstated caller must get the number every frozen study was run at."""
    prob, ref, w = setup
    implicit = composite_evaluation(prob.w0, prob, ref, w)
    explicit = composite_evaluation(prob.w0, prob, ref, w, censored="refuse")
    assert implicit["composite"] == explicit["composite"]
    assert implicit["composite"] == SEED_J
    j_i, g_i = composite_objective(prob.w0, prob, ref, w)
    j_e, g_e = composite_objective(prob.w0, prob, ref, w, censored="refuse")
    assert j_i == j_e == SEED_J
    assert np.array_equal(g_i, g_e)


def test_the_factory_forwards_the_mode(setup, monkeypatch):
    prob, ref, w = setup
    _force_censored(monkeypatch)
    j_default, _ = make_composite_fg(prob, ref, w)(prob.w0)
    j_bound, _ = make_composite_fg(prob, ref, w, censored="lower_bound")(prob.w0)
    assert j_default == PENALTY
    assert j_bound > PENALTY and np.isfinite(j_bound)


def test_the_factory_validates_the_mode_at_bind_time(setup):
    """Fail where the caller can see it, not on the first evaluation."""
    prob, ref, w = setup
    with pytest.raises(ValueError, match="unknown censored mode"):
        make_composite_fg(prob, ref, w, censored="nonsense")


# ------------------------------------------------------- the soundness claim

def test_J_is_monotone_in_both_censored_criteria(setup):
    """Why scoring at the bound is a BOUND and not a guess.

    ``clmax`` and ``astall`` are the only two criteria the stall sweep
    censors, and both are higher-better. So raising either can only raise J:
    the value computed at the observed bound cannot exceed the value the
    section would score if XFOIL had marched further.
    """
    _prob, ref, _w = setup
    assert "clmax" in A.HIGHER_BETTER and "astall" in A.HIGHER_BETTER
    base = {"name": "c", "eligible": True, "path": "", "status": "ok",
            "tc": 0.12, "clmax": 1.2, "astall": 14.0, "ldmax": 100.0,
            "ldcr": 70.0, "cm_at": -0.02, "clmax_censored": True}
    for crit, bump in (("clmax", 0.3), ("astall", 3.0)):
        for weights in (PRESETS["gdp-sweep"], PRESETS["gdp-fwd"], PRESETS["gdp-rear"]):
            lo = A.score_candidates([dict(base)], weights, reference=ref)
            hi_rec = dict(base)
            hi_rec[crit] = base[crit] + bump
            hi = A.score_candidates([hi_rec], weights, reference=ref)
            assert hi[0]["composite"] >= lo[0]["composite"] - 1e-12, (
                f"J must be non-decreasing in {crit} under {weights}")


def test_the_bound_never_exceeds_the_resolved_value(setup, monkeypatch):
    """The same shape, scored at a censored bound and then resolved."""
    prob, ref, w = setup
    resolved = composite_evaluation(prob.w0, prob, ref, w)["composite"]

    real = A.polar_metrics

    def truncated(*a, **kw):
        rec = dict(real(*a, **kw))
        rec["clmax"] = rec["clmax"] - 0.25      # as if the march stopped early
        rec["clmax_censored"] = True
        return rec

    monkeypatch.setattr(A, "polar_metrics", truncated)
    bound = composite_evaluation(prob.w0, prob, ref, w,
                                 censored="lower_bound")["composite"]
    assert bound < resolved, "a bound below the true clmax must score below J"


# ---------------------------------------------------------- the api flag surface
#
# Until now the ONLY way to select the arm was to patch the module attribute
# ``airfoil_select.composite_evaluation`` (which works because
# ``composite_objective`` calls it as a module global). That is a study
# harness, not an interface: it cannot be stated in a RunConfig, cannot be
# stored beside a result, and cannot be reached from the shells at all. These
# pin the real flag.

def test_the_flag_is_declared_and_travels_only_when_it_is_not_the_default():
    """A default call must send the LEGACY flag set, byte for byte.

    Every stored composite result cell is keyed on the flag dict, so a flag
    that appears with its own default value would miss every cached run.
    """
    from aerobo import api

    assert "airfoil_censored" in api.AIRFOIL_OBJECTIVE_FLAG_KEYS

    plain = api.airfoil_run_config(objective="composite")
    assert "airfoil_censored" not in plain.flags

    stated_default = api.airfoil_run_config(objective="composite",
                                            censored="refuse")
    assert stated_default.flags == plain.flags

    bound = api.airfoil_run_config(objective="composite",
                                   censored="lower_bound")
    assert bound.flags["airfoil_censored"] == "lower_bound"
    assert set(bound.flags) - set(plain.flags) == {"airfoil_censored"}


def test_an_unknown_mode_is_refused_where_the_caller_asked():
    from aerobo import api

    with pytest.raises(ValueError, match="unknown censored mode"):
        api.airfoil_run_config(objective="composite", censored="sample_it")


def test_the_policy_is_refused_on_a_cd_run_rather_than_ignored():
    """``-cd`` never reads ``cl_max``, so accepting the flag there would
    promise a policy no evaluation applies."""
    from aerobo import api

    with pytest.raises(ValueError, match="COMPOSITE policy"):
        api.airfoil_run_config(objective="cd", censored="lower_bound")
    # ... but stating the default explicitly on a -cd run is harmless
    assert api.airfoil_run_config(objective="cd", censored="refuse").flags == \
        api.airfoil_run_config(objective="cd").flags


def test_the_built_problem_carries_the_mode(monkeypatch):
    """The flag has to reach the EVALUATION, not just the config."""
    from aerobo import api

    seen: list = []
    real_fg = A.make_composite_fg

    def spy(prob, ref, weights, censored=A.DEFAULT_CENSORED):
        seen.append(censored)
        return real_fg(prob, ref, weights, censored=censored)

    monkeypatch.setattr(A, "make_composite_fg", spy)
    spec = api.PROBLEM_SPECS[api.AIRFOIL_PROBLEM]
    for mode in ("refuse", "lower_bound"):
        cfg = api.airfoil_run_config(objective="composite", censored=mode)
        spec.build(cfg.mission_kwargs or {}, cfg.flags or {},
                   cfg.bounds_overrides)
    assert seen == ["refuse", "lower_bound"]


def test_the_default_build_still_scores_the_frozen_seed(setup):
    """The whole api path, default flags, on the frozen anchor."""
    from aerobo import api

    prob, _ref, _w = setup
    cfg = api.airfoil_run_config(
        objective="composite", score_weights="gdp-sweep",
        anchor=[prob.w0[:prob.n_cst].tolist(), prob.w0[prob.n_cst:].tolist()])
    spec = api.PROBLEM_SPECS[api.AIRFOIL_PROBLEM]
    built = spec.build(cfg.mission_kwargs or {}, cfg.flags or {},
                       cfg.bounds_overrides)
    assert built.evaluate(built.problem.w0)["composite"] == SEED_J
