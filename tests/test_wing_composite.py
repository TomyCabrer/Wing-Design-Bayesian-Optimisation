"""The WING's criteria as a search target, not a post-mortem.

Stage 3 maximised exactly one number — L/D, or payload L/D under the size
modifier — so everything else a wing is judged on (where it stalls, what the
spar carries, whether the tip chord can be built) was something a user read
afterwards and could not ask for. ``wing_objective="composite"`` closes that
the same way ``airfoil_objective="composite"`` closed it for the section:
the same weighted J, against a FROZEN band, as the thing being maximised.

What these tests pin:

* the six criteria come off the breakdown the family ALREADY returns — one
  surface or two, lifting line or panel solver, devices excluded;
* the band is measured over the design box, is frozen, and refuses a box
  nothing flies in;
* the default path is untouched, flag for flag and value for value, so every
  stored L/D run still means what it meant;
* the constraint channel is the family's own, so a composite arm and an L/D
  arm search one feasible region;
* the number a run reports and the number the score block reports are ONE
  number;
* and the weights are the user's: two weight sets over the SAME evaluated
  designs pick different winners.
"""

from __future__ import annotations

import numpy as np
import pytest

from aerobo import api, objective
from aerobo import wing_score as wsc
from gui.v3 import config, session
from gui.v3.stages import wing as stage

PLAIN = "trim wing"          # lifting line, milliseconds per evaluation
TAIL = "tail"                # two surfaces, and a constraint
SMALL = 12                   # band samples: above MIN_REFERENCE_SAMPLES


def _band(problem: str = PLAIN, n: int = SMALL, seed: int = 0) -> dict:
    return api.wing_score_reference(problem, n=n, seed=seed)


def _cfg(problem: str = PLAIN, weights=None, band=None, **kw) -> api.RunConfig:
    flags = {}
    if weights is not None or band is not None:
        flags = {"wing_objective": "composite",
                 "wing_score_weights": weights or {"lod": 1.0},
                 "wing_score_reference": band if band is not None
                 else _band(problem)}
    return api.RunConfig(problem_name=problem, flags=flags,
                         optimiser=kw.pop("optimiser", "random"),
                         budget=kw.pop("budget", 8), seed=kw.pop("seed", 0))


# ------------------------------------------------- the criteria themselves


def test_the_failure_contract_is_the_packages():
    """Restated in wing_score.py to keep the module importable without the
    physics stack; a drift here would put a different penalty in front of
    the optimiser than every other family uses."""
    assert wsc.PENALTY == objective.PENALTY
    assert wsc.G_FAIL == objective.G_FAIL


@pytest.mark.parametrize("problem", [PLAIN, TAIL, "tandem", "winglet"])
def test_every_family_measures_every_criterion_it_can(problem):
    """Wing, wing + tail, two wings, and a tip device — no XFOIL.

    Every criterion the family's own breakdown can support, and a ``None``
    for the ones it cannot. The blanket "all of them, everywhere" this test
    used to assert stopped being the contract when a criterion arrived that
    needs a quantity only SOME families trim for: a static margin needs a
    neutral point, and a neutral point needs a second surface the solve
    balances against. Scoring the others as zero would rank a family that
    cannot answer BELOW one that answered badly, which is the whole reason
    ``design_metrics`` returns None and ``composite`` refuses it.
    """
    built = api.PROBLEM_SPECS[problem].build({}, {}, None)
    raw = built.evaluate(built.bounds.mean(axis=1))
    assert raw["feasible"]
    metrics = wsc.design_metrics(raw, built.problem, {})
    assert set(metrics) == set(wsc.CRITERIA)

    # WHICH criterion is allowed to be missing is read off the breakdown,
    # not off a list of family names — so a family that starts reporting a
    # margin starts being held to it here with no edit.
    optional = set()
    if raw.get("SM") is None:
        optional.add("stab")          # no second surface, so no neutral point
    if raw.get("spiral_margin") is None:
        optional.add("spiral")        # no lateral deck was asked for
    for key, value in metrics.items():
        if key in optional:
            assert value is None, \
                f"{problem} reports no SM but measured {key}={value}"
            continue
        assert value is not None and np.isfinite(value), (key, metrics)


@pytest.mark.parametrize("problem", [PLAIN, "tandem", "winglet"])
def test_a_family_with_no_neutral_point_refuses_to_be_scored_on_one(problem):
    """The consequence, at the objective rather than the metric.

    A user who weights the margin on a family that does not trim one must
    be told, not handed a number built from a zero. This is the criterion
    half of the None-is-not-zero contract.
    """
    built = api.PROBLEM_SPECS[problem].build({}, {}, None)
    raw = built.evaluate(built.bounds.mean(axis=1))
    assert wsc.design_metrics(raw, built.problem, {})["stab"] is None


def test_the_margin_is_the_one_the_run_is_GATED_on():
    """Not a second, differently-derived copy of it.

    The criterion has to be the same number the constraint uses, or a
    search would buy margin against one definition while being refused
    against another.
    """
    built = api.PROBLEM_SPECS[TAIL].build({}, {}, None)
    raw = built.evaluate(built.bounds.mean(axis=1))
    metrics = wsc.design_metrics(raw, built.problem, {})
    # BELOW the ceiling the criterion IS the gated number, exactly — a
    # separately-derived margin would let a search buy against one
    # definition while being refused against another.
    assert metrics["stab"] == min(raw["SM"], wsc.STAB_TARGET_MAC)
    low = dict(raw, SM=0.5 * wsc.STAB_TARGET_MAC)
    assert wsc.design_metrics(low, built.problem, {})["stab"] == low["SM"]
    # ...and ABOVE it the criterion saturates while the CONSTRAINT does not:
    # the gate keeps reading the real margin, so nothing about which designs
    # are admissible changed when the ceiling was added.
    assert raw["SM"] > wsc.STAB_TARGET_MAC
    assert raw["g"] == pytest.approx(raw["SM"] - raw["SM_min"])
    assert raw["SM"] > raw["SM_min"], "pick a design inside the gate"


def _gated_sample(problem: str, flags: dict, n: int = 1200, seed: int = 7):
    """Designs the OPTIMISER could actually choose: feasible AND inside the
    family's own constraint.

    Sampling on ``feasible`` alone is the mistake that produced this file's
    first, wrong measurement of the margin criterion. ``feasible`` means the
    solve converged; the static-margin gate is a separate channel, and the
    L/D optimum sits ON that gate — so an ungated sweep reports a trade
    between designs the search would have refused.
    """
    built = api.PROBLEM_SPECS[problem].build({}, flags, None)
    bounds = np.array(built.bounds)
    lo, hi = bounds.T
    rng = np.random.default_rng(seed)
    rows = []
    for x in lo + (hi - lo) * rng.random((n, len(lo))):
        raw = built.evaluate(x)
        if not raw.get("feasible") or float(np.min(raw["g"])) < 0.0:
            continue
        m = wsc.design_metrics(raw, built.problem, {})
        if m["lod"] is not None and m["stab"] is not None:
            rows.append((raw, m))
    return built, bounds, rows


def test_the_margin_criterion_saturates_so_the_dial_is_not_bang_bang():
    """The composite is a weighted SUM, which can only reach vertices of its
    frontier's convex hull — so an unbounded criterion gives a dial with two
    positions.

    MEASURED before the ceiling, on the `tail` box with a fuselage charged:
    the argmax sat at SM 0.119 for every weight up to 0.42 and then jumped
    to SM 1.143, buying 9.6x the margin for 18.8 % of L/D in a single step.
    This asserts the ceiling that stops it.
    """
    assert wsc.STAB_TARGET_MAC > 0.08, "must be above the SM_min floor"
    fake = {"SM": 5.0, "score": 30.0}
    assert wsc.design_metrics(fake)["stab"] == wsc.STAB_TARGET_MAC
    fake["SM"] = 0.12
    assert wsc.design_metrics(fake)["stab"] == pytest.approx(0.12)


def test_weighting_the_margin_moves_the_winner_and_the_price_is_bounded():
    """Over designs the search could really choose, and with the fuselage
    charged so the arm is not free.

    MEASURED: L/D alone lands on L/D 29.8456 at SM 0.1071; the `stable`
    preset on L/D 28.7493 at SM 0.2521 — 2.35x the margin for 3.67 % of
    L/D.
    """
    flags = {"fuselage_diameter_m": 0.35}
    built, bounds, rows = _gated_sample(TAIL, flags)
    assert len(rows) >= 400, f"only {len(rows)} gate-passing draws"
    ref = wsc.sample_reference(built.evaluate, bounds, problem=TAIL,
                               n=64, seed=0)

    def win(weights):
        js = [wsc.composite(m, ref, weights)[0] for _, m in rows]
        return rows[int(np.argmax(js))][0]

    plain = win(wsc.PRESETS["efficiency"])
    stable = win(wsc.PRESETS["stable"])
    assert stable["SM"] > 2.0 * plain["SM"], "the weight bought no margin"
    assert stable["score"] < plain["score"], "it must be BOUGHT, not free"
    # ...and the price is bounded by the ceiling, not by luck
    assert stable["score"] > 0.90 * plain["score"], (
        f"the margin cost {100 * (1 - stable['score'] / plain['score']):.1f} % "
        f"of L/D — the saturation is not holding")
    assert stable["SM"] <= wsc.STAB_TARGET_MAC + 0.05


def test_a_free_fuselage_makes_the_margin_criterion_buy_LENGTH():
    """Why the measurement above charges one.

    The tail arm raises the static margin at almost no cost while the body's
    drag is unmodelled, so without a fuselage the criterion buys a longer
    aeroplane rather than a better-balanced one. Pinned so the numbers in
    the criterion's help text cannot quietly stop being true.
    """
    labels = list(api.PROBLEM_SPECS[TAIL].param_labels)
    i_arm = labels.index("l_t_m")
    arms = {}
    for tag, flags in (("free", {}), ("charged", {"fuselage_diameter_m": 0.35})):
        built = api.PROBLEM_SPECS[TAIL].build({}, flags, None)
        bounds = np.array(built.bounds)
        lo, hi = bounds.T
        rng = np.random.default_rng(7)
        kept = []
        for x in lo + (hi - lo) * rng.random((1200, len(lo))):
            raw = built.evaluate(x)
            if not raw.get("feasible") or float(np.min(raw["g"])) < 0.0:
                continue
            m = wsc.design_metrics(raw, built.problem, {})
            if m["lod"] is not None and m["stab"] is not None:
                kept.append((x, m))          # ONE list: scored and indexed
        assert len(kept) >= 400, f"{tag}: only {len(kept)} gate-passing draws"
        ref = wsc.sample_reference(built.evaluate, bounds, problem=TAIL,
                                   n=64, seed=0)
        js = [wsc.composite(m, ref, wsc.PRESETS["efficiency"])[0]
              for _, m in kept]
        arms[tag] = float(kept[int(np.argmax(js))][0][i_arm])
    assert arms["charged"] < arms["free"], (
        f"charging the fuselage did not shorten the L/D-optimal arm {arms}")
    assert arms["free"] - arms["charged"] > 1.0, arms


@pytest.mark.parametrize("problem", [PLAIN, TAIL, "tandem"])
def test_the_stall_and_structure_criteria_need_the_problem(problem):
    """A polar is not in the breakdown and a fixed-table family reports no
    thickness — both live on the PROBLEM. Without it those four criteria are
    not measured, which is a None, not a zero."""
    built = api.PROBLEM_SPECS[problem].build({}, {}, None)
    raw = built.evaluate(built.bounds.mean(axis=1))
    bare = wsc.design_metrics(raw)
    assert bare["clmax"] is None and bare["astall"] is None
    assert bare["mass"] is None
    assert bare["lod"] is not None and bare["bend"] is not None


def test_two_surfaces_are_two_surfaces():
    """A tail's panels are not extra span on the wing: the pair comes back as
    two blocks, which is what keeps one spar from being charged the bending
    moment of two wings."""
    built = api.PROBLEM_SPECS[TAIL].build({}, {}, None)
    surfaces = wsc.surfaces_of(built.evaluate(built.bounds.mean(axis=1)))
    assert len(surfaces) == 2
    assert all(s["cl"] is not None and s["y"].size for s in surfaces)


def test_a_tip_device_is_not_charged_to_the_wing():
    """A winglet's panels sit at the tip with a small chord and a long arm.
    Counting them would report the wing's narrowest chord as the device's and
    make every winglet look structurally ruinous."""
    built = api.PROBLEM_SPECS["winglet"].build({}, {}, None)
    raw = built.evaluate(built.bounds.mean(axis=1))
    surf = wsc.surfaces_of(raw)[0]
    assert surf["device"] is not None and surf["device"].any()
    assert wsc.design_metrics(raw)["build"] > float(
        np.min(surf["c"][surf["device"]]))


def test_the_pinned_section_is_where_the_thickness_comes_from():
    """A fixed-table polar states no t/c, so volume and mass would be
    unmeasurable on a wing+tail — except that the section stages 2 and 2.5
    pinned is in the flags, and its SHAPE has a thickness."""
    assert api.section_thickness("hg40") == pytest.approx(0.150, abs=5e-3)
    assert api.section_thickness("not-a-section") is None
    assert api.section_thickness(None) is None

    flags = {api.SECTION_KEY: "hg40"}
    hints = api.wing_thickness_hints(flags)
    assert hints["wing"] == pytest.approx(0.150, abs=5e-3)

    built = api.PROBLEM_SPECS[TAIL].build({}, flags, None)
    raw = built.evaluate(built.bounds.mean(axis=1))
    thick = api._wing_metrics_fn(built, flags)(raw)
    # …against the same family flying its OWN table, whose 4-digit name is
    # the only thickness a fixed-section problem ever states
    plain = api.PROBLEM_SPECS[TAIL].build({}, {}, None)
    thin = api._wing_metrics_fn(plain, {})(
        plain.evaluate(plain.bounds.mean(axis=1)))
    assert thin["vol"] is not None                   # NACA2412 -> 12 %
    assert thick["vol"] > thin["vol"]                # 15 % encloses more
    assert thick["mass"] < thin["mass"]              # …and Raymer's t/c^-0.3

    # a section pinned on the SECOND surface names that surface only
    aft = api.wing_thickness_hints({api.SECTION_AFT_KEY: "hg40"})
    assert set(aft) == {"tail", "rear"}


def test_a_naca_polar_states_its_own_thickness():
    """Read off the DATA's name (a 4-digit designation IS its thickness),
    never off a problem name."""
    assert wsc._tc_from_name("NACA2412 Re1e6 (XFOIL)") == pytest.approx(0.12)
    assert wsc._tc_from_name("naca 0009") == pytest.approx(0.09)
    assert wsc._tc_from_name("hg40") is None
    assert wsc._tc_from_name(None) is None


def test_the_stall_criteria_move_with_the_loading():
    """Both come off the critical strip: load the wing harder and the usable
    CL and the angle margin both fall."""
    built = api.PROBLEM_SPECS[PLAIN].build({}, {}, None)
    x = built.bounds.mean(axis=1)
    base = wsc.design_metrics(built.evaluate(x), built.problem, {})

    from dataclasses import replace as dc_replace
    heavy = dc_replace(built.problem, CL_target=0.9)
    from aerobo import objective as obj
    loaded = wsc.design_metrics(obj.evaluate(x, heavy), heavy, {})
    assert loaded["astall"] < base["astall"]
    assert loaded["cl_peak"] > base["cl_peak"]
    # the usable CL is a property of the SECTION and the loading shape, so it
    # stays the same order — what changed is how much of it is left
    assert loaded["clmax"] > 0.0 and base["clmax"] > 0.0


def test_a_criterion_nobody_reports_is_none_not_zero():
    """None is "not measured". Scoring it as zero would read as "scored
    badly" for something that was never measured."""
    metrics = wsc.design_metrics({"feasible": True, "score": 12.0})
    assert metrics["lod"] == 12.0
    assert metrics["bend"] is None and metrics["cl_peak"] is None


# ------------------------------------------------------------- the band


def test_the_band_is_measured_over_the_box_and_reproducible():
    a, b = _band(), _band()
    assert a["sha"] == b["sha"]                  # same box, same seed
    assert a["n_feasible"] >= wsc.MIN_REFERENCE_SAMPLES
    assert set(a["bounds"]) <= set(wsc.CRITERIA) and a["bounds"]
    for lo, hi in a["bounds"].values():
        assert hi > lo


def test_a_box_nothing_flies_in_is_refused():
    """A band measured over three feasible samples is two points and a line
    through the middle — and every J in the run would be scored against it."""
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match="too few"):
        wsc.sample_reference(lambda x: {"feasible": False}, bounds, n=SMALL)


def test_an_edited_band_does_not_load():
    """The bands themselves are inside the SHA (reference_sha), so a hand-
    edited number cannot pass as the population it claims."""
    payload = _band()
    key = next(iter(payload["bounds"]))
    payload["bounds"][key] = [0.0, 1.0]
    ref = wsc.reference_from_payload(payload)      # loads: it is well-formed
    assert ref.sha != wsc.reference_sha(
        ref.problem, {}, ref.n_samples, ref.seed, ref.bounds)


@pytest.mark.parametrize("payload,match", [
    ({}, "missing"),
    ({"version": 99, "bounds": {"lod": [0, 1]}, "sha": "x"}, "version"),
    ({"version": wsc.REFERENCE_VERSION, "bounds": {}, "sha": "x"}, "bounds"),
    ({"version": wsc.REFERENCE_VERSION, "bounds": {"lod": [1.0]},
      "sha": "x"}, "2-element"),
])
def test_a_malformed_band_raises_rather_than_loading(payload, match):
    with pytest.raises(ValueError, match=match):
        wsc.reference_from_payload(payload)


# ------------------------------------------- reachable, and only when asked


def test_the_default_run_is_untouched_flag_for_flag():
    """A default V3 session sends the LEGACY flag set: the objective keys
    appear only when the composite is chosen, so stored runs and cached
    result cells keep meaning exactly what they meant."""
    S = session.make_session("air")
    assert not any(k.startswith("wing_objective")
                   or k.startswith("wing_score") for k in config.flags(S))
    built = api.PROBLEM_SPECS[PLAIN].build({}, {}, None)
    x = built.bounds.mean(axis=1)
    assert built.callable(x) == pytest.approx(built.evaluate(x)["LoD"])


def test_the_composite_is_offered_where_there_is_a_wing():
    assert all(k in api.PROBLEM_SPECS[PLAIN].flags
               for k in api.WING_OBJECTIVE_FLAG_KEYS)
    assert all(k in api.PROBLEM_SPECS[TAIL].flags
               for k in api.WING_OBJECTIVE_FLAG_KEYS)
    # …and never on the 2-D section problem, which has no wing and has its
    # own composite over the six SECTION criteria
    assert not any(k in api.PROBLEM_SPECS["airfoil (section)"].flags
                   for k in api.WING_OBJECTIVE_FLAG_KEYS)


def test_the_built_problem_maximises_J_not_LoD():
    band = _band()
    flags = {"wing_objective": "composite", "wing_score_weights": "cruise",
             "wing_score_reference": band}
    built = api.PROBLEM_SPECS[PLAIN].build({}, flags, None)
    plain = api.PROBLEM_SPECS[PLAIN].build({}, {}, None)
    x = built.bounds.mean(axis=1)

    j = built.callable(x)
    out = built.evaluate(x)
    assert out["composite"] == pytest.approx(j)
    assert out["score"] == out["f"] == pytest.approx(j)
    # …and the family's own objective is still there, under its own name
    assert out["f_lod"] == pytest.approx(plain.callable(x))
    assert j != pytest.approx(out["f_lod"])
    assert set(out["scores"]) <= set(wsc.CRITERIA)


def test_the_constraint_channel_is_the_familys_own():
    """A composite arm and an L/D arm must search ONE feasible region: the
    margins come from the family's own evaluation, untouched."""
    band = _band(TAIL)
    flags = {"wing_objective": "composite", "wing_score_weights": "cruise",
             "wing_score_reference": band}
    built = api.PROBLEM_SPECS[TAIL].build({}, flags, None)
    plain = api.PROBLEM_SPECS[TAIL].build({}, {}, None)
    x = built.bounds.mean(axis=1)

    j, g = built.callable(x)
    _, g_plain = plain.callable(x)
    assert np.allclose(np.atleast_1d(g), np.atleast_1d(g_plain))
    assert j > 0.0


def test_a_solver_failure_is_the_penalty_and_the_failed_margin():
    band = _band(TAIL)
    flags = {"wing_objective": "composite", "wing_score_reference": band}
    built = api.PROBLEM_SPECS[TAIL].build({}, flags, None)
    out = wsc.composite_objective(
        built.bounds.mean(axis=1), lambda x: {"feasible": False},
        wsc.reference_from_payload(band), wsc.WingScoreWeights(),
        n_constraints=2)
    assert out[0] == wsc.PENALTY
    assert np.allclose(out[1], wsc.G_FAIL)


# ------------------------------------------------------- the three refusals


def test_an_unknown_objective_is_named_not_ignored():
    with pytest.raises(ValueError, match="unknown wing objective"):
        api.check_wing_objective({"wing_objective": "drag"})


def test_a_composite_without_a_band_is_refused():
    with pytest.raises(ValueError, match="FROZEN normalisation band"):
        api.check_wing_objective({"wing_objective": "composite"})


def test_a_weight_the_band_cannot_cover_is_refused_by_name():
    band = _band()
    band["bounds"].pop("bend")
    with pytest.raises(ValueError, match="bend"):
        api.check_wing_objective({"wing_objective": "composite",
                                  "wing_score_weights": {"bend": 1.0},
                                  "wing_score_reference": band})


# ------------------------------------------------------ one number, one map


def test_the_runs_J_is_the_score_blocks_J():
    cfg = _cfg(weights="cruise")
    res = api.run(cfg)
    scored = api.score_optimised_design(cfg, res.best_x, weights="cruise")
    assert scored["optimised"]["composite"] == pytest.approx(res.best_score)
    assert scored["objective"] == "composite"


def test_the_baseline_is_the_centre_of_the_box():
    cfg = _cfg()
    built = api.PROBLEM_SPECS[PLAIN].build({}, {}, None)
    assert np.allclose(api.baseline_x(cfg), built.bounds.mean(axis=1))


def test_the_block_reads_an_ordinary_LoD_run_too():
    """The criteria are read off the breakdown, so a run that never heard of
    the composite still gets its six numbers — which is how a user finds out
    what maximising one scalar cost."""
    cfg = api.RunConfig(problem_name=PLAIN, optimiser="random", budget=6,
                        seed=0)
    res = api.run(cfg)
    rep = api.score_optimised_design(cfg, res.best_x, weights="cruise",
                                     reference=_band())
    assert rep["objective"] == "lod"
    assert rep["optimised"]["composite"] is not None
    assert set(rep["optimised"]["metrics"]) == set(wsc.CRITERIA)
    assert rep["delta"]["metrics"]["lod"] == pytest.approx(
        rep["optimised"]["metrics"]["lod"] - rep["baseline"]["metrics"]["lod"])


def test_without_a_band_the_metrics_still_come_back():
    cfg = api.RunConfig(problem_name=PLAIN, optimiser="random", budget=4,
                        seed=0)
    rep = api.score_optimised_design(cfg, api.baseline_x(cfg))
    assert rep["optimised"]["composite"] is None
    assert rep["optimised"]["metrics"]["lod"] is not None
    assert "band" in rep["optimised"]["reason"]


# --------------------------------------------------- the weights are the map


def test_two_weight_sets_pick_different_designs():
    """The SAME evaluated designs (random search, same seed, same budget)
    scored two ways. If the weights did not reach the objective, the argmax
    would be the same point."""
    band = _band()
    picks = {}
    for label, w in (("lod", {"lod": 1.0}), ("build", {"build": 1.0})):
        cfg = _cfg(weights=w, band=band, budget=16)
        res = api.run(cfg)
        picks[label] = (res.best_x, res.best_score)
    assert not np.allclose(picks["lod"][0], picks["build"][0])

    # …and each really is the best of its own criterion among what was seen
    metrics = {label: api.score_design(cfg, x, weights="cruise",
                                       reference=band)["metrics"]
               for label, (x, _) in picks.items()
               for cfg in (_cfg(weights="cruise", band=band),)}
    assert metrics["build"]["build"] > metrics["lod"]["build"]
    assert metrics["lod"]["lod"] > metrics["build"]["lod"]


def test_the_report_says_where_the_answer_sits_in_the_box():
    """The centre is ONE design. "beats 87% of the box sample" is the
    question a user actually asked, and the population is already stored
    beside the band it produced."""
    band = _band()
    assert len(band["samples"]) == band["n_feasible"]
    cfg = _cfg(weights="cruise", band=band, budget=12)
    res = api.run(cfg)
    rep = api.score_optimised_design(cfg, res.best_x, weights="cruise")
    assert 0.0 <= rep["beats"] <= 1.0

    ref = wsc.reference_from_payload(band)
    w = wsc.weights_of("cruise")
    assert wsc.population_rank(1e9, ref, w) == pytest.approx(1.0)
    assert wsc.population_rank(-1e9, ref, w) == pytest.approx(0.0)
    assert wsc.population_rank(None, ref, w) is None
    # the samples are DISPLAY-only: nothing in J reads them, so dropping
    # them changes no score
    bare = dict(band, samples=[])
    x = api.baseline_x(cfg)
    assert api.score_design(cfg, x, weights="cruise", reference=bare)[
        "composite"] == pytest.approx(
        api.score_design(cfg, x, weights="cruise",
                         reference=band)["composite"])


def test_a_partial_weight_dict_merges_over_the_preset():
    w = api.wing_score_weights({"bend": 0.5})
    assert w.bend == pytest.approx(0.5)
    assert w.lod == pytest.approx(wsc.WingScoreWeights().lod)   # untouched
    assert sum(w.normalised().values()) == pytest.approx(1.0)


def test_an_unknown_criterion_is_refused():
    with pytest.raises(ValueError, match="unknown wing criteria"):
        api.wing_score_weights({"drag": 1.0})


# ------------------------------------------------------------ the shell


def test_the_stage_opens_on_the_familys_own_objective():
    """The composite is a CHOICE, never something a stage inherits."""
    S = session.make_session("air")
    sc = session.wing_score_state(S)
    assert sc["objective"] == "lod"
    assert sc["weights_source"] == "recommended"
    assert session.wing_weights_are_recommended(S)
    assert sc["weights"] == dict(
        (k, getattr(wsc.PRESETS[session.WING_WEIGHT_PRESET], k))
        for k in wsc.CRITERIA)


def test_the_flags_travel_only_on_a_composite_run():
    S = session.make_session("air")
    assert session.wing_score_flags(S) == {}
    session.set_wing_objective(S, "composite")
    session.set_wing_band(S, _band(S["wing"]["problem"], n=SMALL),
                          session.wing_band_box(S))
    flags = session.wing_score_flags(S)
    assert set(flags) == set(api.WING_OBJECTIVE_FLAG_KEYS)
    assert flags["wing_objective"] == "composite"
    assert flags["wing_score_reference"]["sha"]
    # the measured population stays in the session: a run record carries the
    # bands the objective reads, not a few hundred floats of display data
    assert "samples" not in flags["wing_score_reference"]
    assert session.wing_score_state(S)["band"]["samples"]
    assert set(config.flags(S)) >= set(api.WING_OBJECTIVE_FLAG_KEYS)


def test_an_edited_weight_becomes_the_users():
    S = session.make_session("air")
    session.set_wing_weight(S, "bend", 0.4)
    assert session.wing_score_state(S)["weights_source"] == "user"
    assert not session.wing_weights_are_recommended(S)
    session.set_wing_weights(S, session.WING_WEIGHT_PRESET, "recommended")
    assert session.wing_weights_are_recommended(S)


def test_a_band_belongs_to_the_box_it_was_measured_on():
    S = session.make_session("air")
    session.set_wing_band(S, _band(S["wing"]["problem"], n=SMALL),
                          session.wing_band_box(S))
    assert session.wing_band_stale(S) is None
    row = next(iter(session.wing_band_box(S)))
    S["wing"]["bounds"][row] = [0.31, 0.62]
    assert row in (session.wing_band_stale(S) or "")


def test_the_card_offers_both_scalars_and_names_the_six():
    assert set(stage.WING_OBJECTIVE_CHOICES()) == set(api.WING_OBJECTIVES)
    meta = stage.wing_weight_meta()
    assert set(meta) == set(wsc.CRITERIA)
    assert all(label and why for label, why in meta.values())


def test_the_score_rows_field_names_are_the_tables_columns():
    """The column declarations and the row keys are one contract on this
    card — the pairing that has already broken the section card once."""
    cfg = _cfg(weights="cruise")
    res = api.run(cfg)
    rep = api.score_optimised_design(cfg, res.best_x, weights="cruise")
    rows = stage.wing_score_rows(rep)
    assert rows and rows[0]["metric"] == "composite score J"
    # …plus ``dir``, which is not a column but the VERDICT the table's own
    # body-cell slot paints green/red/grey by. It belongs in this contract
    # for the same reason the four columns do: a row without it renders grey
    # whatever happened, which looks exactly like a working colour rule.
    assert all(set(r) == {"metric", "original", "new", "change", "dir"}
               for r in rows)
    assert all(r["dir"] in ("better", "worse", "same", "") for r in rows)
    # a zero-weight criterion is KEPT: "this counted for nothing" is what the
    # composite's own arithmetic hides
    assert any("weight 0.00" in r["metric"] for r in rows)


def test_no_report_is_no_rows_not_a_crash():
    assert stage.wing_score_rows(None) == []
    assert stage.wing_score_rows({}) == []


def test_the_stage_renders_both_objectives_through_its_own_handler(capsys):
    """Every branch of the objective card — no band, a measured band, the
    score block with nothing to show — drawn through the shell's own
    handler, because a card that only builds in one state builds in none."""
    from gui.v3.app import assemble

    ctx = assemble()
    ctx.act("accept_mission")
    capsys.readouterr()

    for view in ("solver", "run"):
        ctx.render("wing", view)
    ctx.act("set_wing_objective", "composite")
    assert session.wing_score_state(ctx.S)["objective"] == "composite"
    for view in ("solver", "run"):
        ctx.render("wing", view)                  # the no-band branch

    session.set_wing_band(ctx.S, _band(ctx.S["wing"]["problem"], n=SMALL),
                          session.wing_band_box(ctx.S))
    ctx.act("set_wing_weight", "bend", 0.25)
    for view in ("solver", "run"):
        ctx.render("wing", view)                  # …and the measured one

    # …and the run tab WITH a finished run under it, which is the only state
    # the score block itself draws in
    import gui.nice_app as v1
    from gui.v3 import config as v3config

    cfg = v3config.build_cfg(ctx.S)
    sc = session.wing_score_state(ctx.S)
    sc["report"] = api.score_optimised_design(
        cfg, api.baseline_x(cfg), weights=dict(sc["weights"]),
        reference=sc["band"])
    ctx.manager.jobs = [v1.RunJob(cfg=cfg, label="test", budget=4,
                                  status="done")]
    ctx.render("wing", "run")
    sc["scoring"], sc["report"] = True, None      # the spinner branch
    ctx.render("wing", "run")
    sc["scoring"], sc["report_error"] = False, "boom"
    ctx.render("wing", "run")

    err = capsys.readouterr().err
    assert "Traceback" not in err, err
    assert session.wing_score_state(ctx.S)["weights_source"] == "user"
