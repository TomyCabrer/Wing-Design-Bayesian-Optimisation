"""The screen's composite as a SEARCH TARGET, not just a ranking.

Stage 2 ranks the library on six weighted criteria and stage 3 then maximised
one number — so five of the six stopped counting the moment the search started
(report §15.4). ``objective="composite"`` closes that: the same weighted J,
against the same frozen band, as the thing being maximised.

What these tests pin:

* the objective is reachable through the ordinary run config, and the built
  problem really evaluates J (not -cd wearing J's name);
* the two refusals that keep it honest — wing mode (a 2-D score cannot see a
  twist law) and a missing frozen band (a live min-max is a moving target);
* the default path is untouched, flag for flag, so every stored -cd run and
  its cached cells still mean what they meant;
* J from a run and J from ``api.score_sections`` are the SAME number for the
  same shape, which is what makes the seed-vs-optimised block readable.
"""

from __future__ import annotations

import numpy as np
import pytest

from aerobo import api, xfoil_run
from aerobo.airfoil import PENALTY, AirfoilProblem
from aerobo.airfoil_select import (
    CRITERIA,
    PRESETS,
    composite_evaluation,
    composite_objective,
    load_screen_reference,
)
from aerobo.xfoil_run import XfoilPolarResult
from gui.v3 import session
from gui.v3.stages import airfoil as stage

WING = {"mass_kg": 12.0, "v_ms": 20.0, "altitude_m": 0.0,
        "s_ref_m2": 1.0, "aspect_ratio": 8.0, "taper": 0.6}


def _install_fake_xfoil(monkeypatch, *, cm=-0.04, censored=False):
    """Cruise sweep vs the wide stall sweep, dispatched on the alpha range —
    the composite needs both (tests/test_airfoil_pipeline.py's convention)."""
    calls = {"cruise": 0, "stall": 0}

    def fake(coords, re, mach, alphas, **kw):
        a = np.asarray(alphas, dtype=float)
        if a.max() > 10.5:
            calls["stall"] += 1
            cl = (0.3 + 0.09 * a if censored
                  else 1.6 - 0.01 * (a - 14.0) ** 2)
        else:
            calls["cruise"] += 1
            cl = 0.25 + 0.11 * a
        cd = 0.006 + 1e-4 * (a - 1.0) ** 2
        return XfoilPolarResult(alpha_deg=a, cl=cl, cd=cd,
                                cm=np.full_like(a, cm), n_requested=a.size)

    monkeypatch.setattr(xfoil_run, "run_xfoil_polar", fake)
    return calls


# ------------------------------------------------- reachable from a config


def test_the_config_carries_the_objective_and_its_band():
    cfg = api.airfoil_run_config(objective="composite",
                                 score_weights="gdp-sweep")
    assert cfg.flags["airfoil_objective"] == "composite"
    assert cfg.flags["airfoil_score_weights"] == "gdp-sweep"


def test_the_default_run_is_untouched_flag_for_flag():
    """A -cd run must send the LEGACY flag set: the objective keys appear only
    when they are not the default, so stored runs and cached result cells keep
    meaning exactly what they meant."""
    cfg = api.airfoil_run_config()
    assert not any(k.startswith("airfoil_objective")
                   or k.startswith("airfoil_score") for k in cfg.flags)


def test_the_built_problem_evaluates_J_not_cd(monkeypatch):
    _install_fake_xfoil(monkeypatch)
    cfg = api.airfoil_run_config(objective="composite",
                                 score_weights="gdp-sweep")
    built = api.PROBLEM_SPECS[cfg.problem_name].build(
        cfg.mission_kwargs or {}, cfg.flags or {}, cfg.bounds_overrides)
    prob = AirfoilProblem()
    ref, w = load_screen_reference(), PRESETS["gdp-sweep"]

    J, g = built.callable(prob.w0)
    J_ref, g_ref = composite_objective(prob.w0, prob, ref, w)
    assert J == pytest.approx(J_ref) and np.allclose(g, g_ref)
    # …and it is a SCORE, not a drag: the -cd objective is a small negative
    assert 0.0 < J < 200.0


def test_evaluate_reports_the_objective_the_optimiser_saw(monkeypatch):
    """``score``/``f`` on the evaluation dict is J. A shell reads that key to
    draw "best objective", and -0.0055 under a run labelled composite is
    exactly the drift this contract exists to stop."""
    _install_fake_xfoil(monkeypatch)
    prob = AirfoilProblem()
    ref, w = load_screen_reference(), PRESETS["gdp-sweep"]
    out = composite_evaluation(prob.w0, prob, ref, w)

    assert out["composite"] is not None
    assert out["score"] == out["f"] == out["composite"]
    assert out["f_cd"] == pytest.approx(-out["cd"])     # still available
    assert set(out["scores"]) == set(CRITERIA)


def test_an_unscoreable_candidate_scores_the_penalty(monkeypatch):
    """A censored cl_max is a solver failure for scoring, so the dict says
    PENALTY rather than reporting a J built on a lower bound."""
    _install_fake_xfoil(monkeypatch, censored=True)
    prob = AirfoilProblem()
    ref, w = load_screen_reference(), PRESETS["gdp-sweep"]
    out = composite_evaluation(prob.w0, prob, ref, w)

    assert out["composite"] is None
    assert out["score"] == PENALTY
    assert "censored" in out["reason"]


# ------------------------------------------------- the two refusals


def test_wing_mode_and_the_composite_are_refused_together():
    """J scores a 2-D section. Searching a twist law against it would be a
    freedom the objective cannot see, so the pair is refused where it is
    asked for rather than silently ignored."""
    with pytest.raises(ValueError, match="2-D SECTION"):
        api.airfoil_run_config(objective="composite", wing=WING)


def test_a_missing_band_refuses_the_run(monkeypatch):
    from aerobo import airfoil_select

    def broken(*a, **kw):
        raise ValueError("no band here")

    monkeypatch.setattr(airfoil_select, "load_screen_reference", broken)
    with pytest.raises(ValueError, match="FROZEN normalisation band"):
        api.airfoil_run_config(objective="composite")


def test_an_unknown_objective_is_named_not_ignored():
    with pytest.raises(ValueError, match="unknown airfoil objective"):
        api.airfoil_run_config(objective="lift")


# ------------------------------------------------- one number, one meaning


def test_the_runs_J_is_the_score_blocks_J(monkeypatch):
    """The optimiser's best objective and what ``score_sections`` reports for
    the same shape are ONE number — the seed-vs-optimised block is only
    readable beside the objective if the two maps are the same map."""
    _install_fake_xfoil(monkeypatch)
    rep = api.optimize_airfoil(objective="composite", score_weights="gdp-sweep",
                               optimiser="random", budget=4, seed=0)

    assert rep["conditions"]["objective"] == "composite"
    assert rep["conditions"]["score"]["reference_sha"]
    best = rep["result"]["best_score"]

    scored = api.score_optimised_section(rep, "gdp-sweep")
    assert scored["optimised"]["composite"] == pytest.approx(best)


def test_the_report_names_the_weights_it_was_run_under(monkeypatch):
    """A partial dict MERGES over the GDP preset (screen_weights' documented
    rule, shared with the screen) and the report states the resolved,
    normalised set — so a stored run says what it maximised, not what was
    typed."""
    _install_fake_xfoil(monkeypatch)
    rep = api.optimize_airfoil(objective="composite",
                               score_weights={"clmax": 1.0, "ldcr": 1.0},
                               optimiser="random", budget=3, seed=0)
    w = rep["conditions"]["score"]["weights"]
    assert sum(w.values()) == pytest.approx(1.0)
    assert w["clmax"] == pytest.approx(w["ldcr"])
    assert w["clmax"] > w["thick"] > 0.0        # the preset's rest survives
    assert w == api.screen_weights({"clmax": 1.0, "ldcr": 1.0}).normalised()


# ------------------------------------------------- the card's own menu


def test_the_menu_offers_the_surfaces_own_physical_objective():
    assert stage.OBJECTIVE_CHOICES(True)["cd"].startswith("wing L/D")
    assert stage.OBJECTIVE_CHOICES(False)["cd"].startswith("2-D L/D")
    for wing_mode in (True, False):
        assert "composite" in stage.OBJECTIVE_CHOICES(wing_mode)


def test_the_composite_never_travels_with_a_wing_guess():
    """One place decides the three coupled arguments, so a wing-capable
    surface picking the composite drops wing mode instead of sending a pair
    the api refuses."""
    kw = stage.objective_kwargs("composite", wing_guess=WING,
                                weights={"clmax": 1.0}, reference=None)
    assert kw["wing"] is None
    assert kw["objective"] == "composite"
    assert kw["score_weights"] == {"clmax": 1.0}
    # …and it builds, which is the assertion that actually binds
    api.airfoil_run_config(**{k: v for k, v in kw.items()
                              if k != "score_reference"})


def test_a_cd_run_sends_the_legacy_argument_set():
    kw = stage.objective_kwargs("cd", wing_guess=WING, weights={"clmax": 1.0},
                                reference={"bounds": {}})
    assert kw == {"objective": "cd", "wing": WING}   # no score arguments


def test_the_live_panel_names_J_as_J():
    """``f`` / ``score`` are the harness's names for whatever was maximised.
    On a composite run they ARE J, so the panel must offer it once, under its
    own name — not a third of the plot showing J labelled "L/D"."""
    from gui import metrics as mc

    have = ["composite", "f", "score", "cd", "clmax", "astall", "ldcr",
            "ldmax", "tc", "cm"]
    short, rest = mc.live_metric_menu(have)
    labels = dict(short)
    assert labels["composite"] == "composite score J"
    assert "f" not in labels and "score" not in labels
    assert "f" not in rest and "score" not in rest
    # the criteria J is made of are offered beside it, under the screen's names
    assert labels["clmax"] == "c_l max" and labels["ldcr"].startswith("L/D at")
    assert mc.live_metric_defaults(have)[0] == "composite"


def test_a_non_composite_run_keeps_its_objective_label():
    from gui import metrics as mc

    for have in (["LoD", "f", "score", "cd_counts", "cl_max"],
                 ["f", "score", "cd", "cl_max_branch"]):
        short = dict(mc.live_metric_menu(have)[0])
        assert "L/D (the objective)" in short.values()
        assert "composite score J" not in short.values()
        assert mc.live_metric_defaults(have)          # still ticks something


def test_every_surface_opens_on_the_composite_with_the_seed_under_it():
    """Every section stage OPENS on the number its own ranking is built from —
    and, since session 45, with the SEED as a floor under it.

    The weights on the screening tab chose the section out of the library on
    six criteria. A search that then maximised drag alone threw five of them
    away the moment the user pressed Optimise — one stage giving two answers
    that disagree (report §15.4). But the weighted SUM is free to sell those
    criteria back, and over 42 paired seeds of the frozen case it does: the
    plain composite returns a section DRAGGIER at the design lift than the one
    the user started from in 34 of 42 runs, against 1 of 42 for the goal
    composite, at a measured price of 1.36 J median. `RESULTS_GOAL_ARM_STUDY.md`
    (pre-registered in `PREREG_SESSION44.md`) is the measurement and report
    §16.3b the argument.

    Both of the other answers stay one click away and `OBJECTIVE_CHOICES` still
    offers all of them.
    """
    S = session.make_session("air")
    for surface in ("main", "aft"):
        assert session.airfoil_state(S, surface)["opt"]["objective"] \
            == "composite_goal"
        # whatever the default is, it must be a COMPOSITE one: the stage may
        # not open on the physical objective while the ranking above it is
        # built from six criteria
        assert stage.is_composite(
            session.airfoil_state(S, surface)["opt"]["objective"])
    # ...and every objective is a real option of the menu on either kind of
    # surface, not a value the select cannot show
    for wing_mode in (True, False):
        choices = stage.OBJECTIVE_CHOICES(wing_mode)
        for name in ("cd", "composite", "composite_goal", "composite_asf"):
            assert name in choices
