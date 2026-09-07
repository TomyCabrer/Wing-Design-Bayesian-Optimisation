"""A FRONT for the section stage: ``api.pareto_airfoil`` and ``optimize.mobo``.

Every other section objective collapses six criteria to one number before the
search starts. This one does not: it optimises three of them as three, returns
the non-dominated feasible designs, and lets the composite do the job it should
always have had — RANKING a front the search produced, rather than deciding
what the search is allowed to reach.

What these tests pin, as outcomes:

* the front is FEASIBLE and non-dominated, and an infeasible design cannot
  reach it however good its objectives are;
* the objective vector is the three sub-scores on the FROZEN band, so two runs'
  hypervolumes are the same number — a raw-unit vector would not be;
* the hypervolume reference point is the SEED exactly, the trace is monotone
  non-decreasing, and a run that beats the seed on no design sweeps ZERO
  volume rather than reporting a number;
* the constraint channel is the composite's own, margin for margin, which is
  what makes the front and the scalarised arms comparable at all;
* a refusal is ``PENALTY`` in every component and the refusal encoding is
  applied per objective;
* ranking uses the USER's weights restricted to the three criteria and
  renormalised, so the order a user sees is their own preference;
* every row carries all six criteria — a front picked off must not hide the
  three it was not searched on.

The XFOIL sweeps are faked, as in the other objective tests: the branch under
test is the search and scoring policy, not the boundary-layer march.
"""

from __future__ import annotations

import numpy as np
import pytest

from aerobo import api, xfoil_run
from aerobo.airfoil import AirfoilProblem
from aerobo.airfoil_select import (
    CRITERIA,
    PARETO_CRITERIA,
    PENALTY,
    PRESETS,
    ScoreWeights,
    composite_objective,
    load_screen_reference,
    pareto_evaluation,
    pareto_objective,
    pareto_weights,
    seed_pareto_point,
)
from aerobo.optimize import mobo


def _install_fake_xfoil(monkeypatch, *, geometry_sensitive=True):
    """A polar that MOVES with the section, so the front is not degenerate."""

    def fake(coords, re, mach, alphas, **kw):
        a = np.asarray(alphas, dtype=float)
        t = float(np.ptp(np.asarray(coords)[:, 1])) if geometry_sensitive \
            else 0.12
        if a.max() > 10.5:
            cl = 1.2 + 4.0 * t - 0.01 * (a - 14.0) ** 2
        else:
            cl = 0.25 + 0.11 * a
        cd = 0.006 + 1e-4 * (a - 1.0) ** 2 + 0.02 * (t - 0.12) ** 2
        cm = np.full_like(a, -0.02 - 0.4 * (t - 0.12))
        return xfoil_run.XfoilPolarResult(alpha_deg=a, cl=cl, cd=cd, cm=cm,
                                          n_requested=a.size)

    monkeypatch.setattr(xfoil_run, "run_xfoil_polar", fake)


@pytest.fixture()
def setup(monkeypatch):
    _install_fake_xfoil(monkeypatch)
    prob = AirfoilProblem()
    return prob, load_screen_reference(), PRESETS["gdp-sweep"]


# ------------------------------------------------------- the objective vector


def test_the_vector_is_three_sub_scores_on_the_frozen_band(setup):
    """Not raw units. A hypervolume is a PRODUCT of ranges, so raw units would
    make the number mean whatever the units happened to be, and two runs could
    not be compared at all."""
    prob, ref, w = setup
    out = pareto_evaluation(prob.w0, prob, ref, w)

    assert out["pareto_criteria"] == ["ldcr", "clmax", "cm"]
    assert len(out["pareto"]) == 3
    for k, v in zip(PARETO_CRITERIA, out["pareto"]):
        assert v == pytest.approx(out["scores"][k], abs=1e-12)
    # …and the composite J is still there beside it, unchanged
    assert out["composite"] == pytest.approx(
        sum(w.normalised()[k] * out["scores"][k] for k in CRITERIA), abs=1e-9)


def test_cm_enters_higher_better_because_the_band_mirrors_it(setup):
    """|Cm| is lower-better in physics and higher-better in the vector, because
    ``sub_score`` mirrors it. A front that maximised |Cm| would be the sign
    error this asserts against."""
    prob, ref, w = setup
    out = pareto_evaluation(prob.w0, prob, ref, w)
    lo, hi = ref.band("cm")
    got = abs(float(out["cm"]))

    assert out["pareto"][2] == pytest.approx(100.0 * (hi - got) / (hi - lo),
                                             abs=1e-9)
    # a LARGER |Cm| must score LOWER on the objective
    worse = 100.0 * (hi - (got + 0.01)) / (hi - lo)
    assert worse < out["pareto"][2]


def test_the_constraint_channel_is_the_composites_own(setup):
    """Margin for margin. This is what makes the front and every scalarised
    arm one comparison rather than two unrelated searches."""
    prob, ref, w = setup
    y, g = pareto_objective(prob.w0, prob, ref, w)
    _j, g_c = composite_objective(prob.w0, prob, ref, w)

    assert np.allclose(g, g_c)
    assert y.shape == (3,)


def test_a_refusal_is_the_sentinel_in_every_component(setup):
    """The scalar path returns PENALTY; the vector path returns it k times,
    which is what ``mobo`` tests for and what the refusal encoding replaces.
    A partial vector would be a half-measured design scored as a whole one."""
    prob, ref, w = setup
    x_bad = np.full(prob.dim, 1e3)          # far outside anything flyable
    y, g = pareto_objective(x_bad, prob, ref, w)

    assert y.shape == (3,) and np.all(y == PENALTY)
    assert g.size == np.asarray(composite_objective(x_bad, prob, ref, w)[1]).size


def test_the_reference_point_is_the_seed_exactly(setup):
    """Not a nudged-outward nadir: a design earns volume only by beating the
    section the user already has on ALL THREE criteria at once."""
    prob, ref, w = setup
    r = seed_pareto_point(prob, ref, w)
    out = pareto_evaluation(prob.w0, prob, ref, w)

    assert np.allclose(r, out["pareto"])
    # …and therefore the seed itself sweeps no volume
    assert mobo.hv_needed_for(np.asarray(out["pareto"]), r) == 0.0


def test_the_rank_weights_are_the_users_own_renormalised(setup):
    """The order a user sees over a front must be their preference and not a
    fourth one invented for the ranking."""
    _prob, _ref, w = setup
    v = pareto_weights(w)

    assert v.sum() == pytest.approx(1.0)
    # gdp-sweep: ldcr 0.35, clmax 0.20, cm 0.20 -> 0.35/0.75 etc.
    assert v == pytest.approx([0.35 / 0.75, 0.20 / 0.75, 0.20 / 0.75])
    # a preset that weights none of the three falls back to equal, not to nan
    flat = pareto_weights(ScoreWeights(thick=1.0, clmax=0.0, ldmax=0.0,
                                       ldcr=0.0, astall=0.0, cm=0.0))
    assert flat == pytest.approx([1 / 3, 1 / 3, 1 / 3])


# ---------------------------------------------------------------- the harness


def test_an_infeasible_design_never_reaches_the_front():
    """However good its objectives are. A design that violates a gate is not
    on the front at any objective value, and letting it dominate a feasible
    one would hand the user a section the problem already refused."""
    Y = np.array([[10.0, 10.0, 10.0],       # best on everything, INFEASIBLE
                  [1.0, 1.0, 1.0],
                  [2.0, 0.5, 0.5]])
    feas = np.array([False, True, True])
    idx = mobo.pareto_indices(Y, feas)

    assert 0 not in idx
    assert set(idx) == {1, 2}               # neither of the two dominates
    assert mobo.pareto_indices(Y, np.zeros(3, bool)).size == 0


def test_the_front_is_non_dominated():
    Y = np.array([[3.0, 1.0], [2.0, 2.0], [1.0, 3.0], [1.0, 1.0]])
    idx = mobo.pareto_indices(Y, np.ones(4, bool))

    assert set(idx) == {0, 1, 2}            # row 3 is dominated by all three


def test_a_run_that_beats_nothing_sweeps_zero_volume():
    """The honest answer, not a failure: a run whose designs never dominate the
    reference point has swept no volume, and saying 0.0 is what lets a reader
    tell that from a small win."""
    ref = np.array([5.0, 5.0])
    below = np.array([[4.0, 9.0], [9.0, 4.0]])      # neither dominates ref
    assert mobo.dominated_hypervolume(below, ref) == 0.0

    above = np.array([[6.0, 7.0]])
    assert mobo.dominated_hypervolume(above, ref) == pytest.approx(2.0)


@pytest.mark.parametrize("q", [1, 2, 4])
def test_the_acquisition_actually_runs_at_every_batch_size(q):
    """The q > 1 half of the same lesson, and it caught a second live bug.

    `optimize_acqf` returns a SCALAR acquisition value at q = 1 and a (q,)
    TENSOR under sequential batching. A purely diagnostic `float(acq_val)`
    inside the try block therefore raised for every batched iteration, and the
    random fallback swallowed it — the acquisition was working the whole time.
    Parametrising over q is what makes that visible.
    """
    def fg(x):
        x = np.asarray(x, float)
        return np.array([-x[0], -x[1]]), np.array([x[0] + x[1] - 0.5])

    h = mobo.run_mobo_constrained(fg, np.array([[0., 1.], [0., 1.]]),
                                  ref_point=[-1.2, -1.2], n_init=6, n_iter=8,
                                  seed=0, q=q)
    assert h.gp_failures == [], (
        f"q={q} fell back to random on {len(h.gp_failures)} iterations")
    assert h.X.shape[0] == 14


def test_a_parallel_batch_visits_exactly_the_serial_points():
    """Concurrency may move the wall clock and nothing else.

    The acquisition proposes the whole batch before any of it is evaluated, so
    the visited points cannot depend on evaluation order — and the history is
    appended in PROPOSAL order, not completion order, so the running
    hypervolume trace cannot either.
    """
    def fg(x):
        x = np.asarray(x, float)
        return np.array([-x[0], -x[1]]), np.array([x[0] + x[1] - 0.5])

    b = np.array([[0., 1.], [0., 1.]])
    kw = dict(ref_point=[-1.2, -1.2], n_init=6, n_iter=8, seed=0, q=4)
    par = mobo.run_mobo_constrained(fg, b, parallel=True, **kw)
    ser = mobo.run_mobo_constrained(fg, b, parallel=False, **kw)

    assert np.allclose(par.X, ser.X)
    assert np.allclose(par.Y, ser.Y)
    assert np.allclose(par.hv_trace, ser.hv_trace)
    assert par.gp_failures == ser.gp_failures == []


@pytest.mark.parametrize("acqf", ["qnehvi", "qnparego"])
def test_the_acquisition_actually_runs(acqf):
    """THE test this file was missing, and it cost a whole wrong result.

    ``run_mobo_constrained`` catches a GP/acquisition failure and draws at
    random instead — the scalar path does the same, and it is right to, because
    a single failed fit must not end a run. But it means a completely broken
    acquisition looks exactly like a working one from the outside: the first
    real front this module produced had ``gp_failures`` on EVERY iteration
    (botorch refuses a plain callable where an ``MCMultiOutputObjective`` is
    required, and the refusal was swallowed), so the "qNEHVI front" was a
    Sobol front. Nothing else in this file could see that.

    So: assert the fallback never fired. A test that only checks the OUTPUT of
    a function with a silent fallback is testing the fallback.
    """
    def fg(x):
        x = np.asarray(x, float)
        return np.array([-x[0], -x[1]]), np.array([x[0] + x[1] - 0.5])

    h = mobo.run_mobo_constrained(fg, np.array([[0., 1.], [0., 1.]]),
                                  ref_point=[-1.2, -1.2], n_init=6, n_iter=4,
                                  seed=0, acqf=acqf)
    assert h.gp_failures == [], (
        f"{acqf} fell back to random on {len(h.gp_failures)} of 4 iterations")


def test_the_hypervolume_trace_never_falls(setup):
    """Hypervolume is monotone in the observed set by construction, so a
    falling trace would mean the trace was not the observed set's."""
    def fg(x):
        x = np.asarray(x, float)
        return np.array([-x[0], -x[1]]), np.array([x[0] + x[1] - 0.5])

    h = mobo.run_mobo_constrained(fg, np.array([[0., 1.], [0., 1.]]),
                                  ref_point=[-1.2, -1.2], n_init=6, n_iter=6,
                                  seed=0)
    assert np.all(np.diff(h.hv_trace) >= -1e-12)
    assert h.hv_trace.size == h.X.shape[0]
    assert h.hypervolume > 0.0
    assert bool(h.feasible[h.front_idx].all())
    assert h.gp_failures == []


def test_the_warm_start_is_budget_fair():
    """The seed shrinks the Sobol draw by the row it adds, so a warm-started
    run and a cold one cost the same — the same contract the scalar path
    holds."""
    def fg(x):
        return np.array([-float(x[0]), -float(x[1])]), np.array([1.0])

    b = np.array([[0., 1.], [0., 1.]])
    cold = mobo.run_mobo_constrained(fg, b, ref_point=[-2., -2.], n_init=6,
                                     n_iter=4, seed=0)
    warm = mobo.run_mobo_constrained(fg, b, ref_point=[-2., -2.], n_init=6,
                                     n_iter=4, seed=0,
                                     x_init=np.array([[0.5, 0.5]]))
    assert cold.X.shape[0] == warm.X.shape[0] == 10
    assert np.allclose(warm.X[0], [0.5, 0.5])       # …and it went in FIRST


def test_rank_front_orders_by_the_weighted_sum():
    Y = np.array([[1.0, 9.0], [9.0, 1.0], [5.0, 5.0]])

    assert list(mobo.rank_front(Y, [1.0, 0.0])) == [1, 2, 0]
    assert list(mobo.rank_front(Y, [0.0, 1.0])) == [0, 2, 1]
    with pytest.raises(ValueError, match="weights"):
        mobo.rank_front(Y, [1.0, 0.0, 0.0])


def test_an_unknown_acqf_is_refused():
    def fg(x):
        return np.array([0.0, 0.0]), np.array([1.0])

    with pytest.raises(ValueError, match="acqf"):
        mobo.run_mobo_constrained(fg, np.array([[0., 1.]]), ref_point=[-1, -1],
                                  n_init=2, n_iter=0, acqf="nope")
    with pytest.raises(ValueError, match="ref_point"):
        mobo.run_mobo_constrained(fg, np.array([[0., 1.]]),
                                  ref_point=[-1, -1, -1], n_init=2, n_iter=0)


# ------------------------------------------------------------ the api surface


def test_the_nadir_hypervolume_is_a_within_run_signal_only(setup):
    """The seed reference answers "did anything beat what I have, on
    everything?" and is 0.0 for a whole run when nothing did — right as a
    cross-run number, useless as progress. The nadir trace is measured from a
    reference fixed at the run's OWN final feasible nadir, so it is monotone
    and readable; and it is not comparable across runs, which is why the two
    are kept in separate fields and separately named."""
    def fg(x):
        x = np.asarray(x, float)
        return np.array([-x[0], -x[1]]), np.array([x[0] + x[1] - 0.5])

    # a reference point NOTHING can dominate: the seed trace stays flat at 0
    h = mobo.run_mobo_constrained(fg, np.array([[0., 1.], [0., 1.]]),
                                  ref_point=[10.0, 10.0], n_init=6, n_iter=6,
                                  seed=0)
    assert h.hypervolume == 0.0 and np.all(h.hv_trace == 0.0)

    # …while the within-run trace still moves, and never falls
    assert h.hypervolume_nadir > 0.0
    assert np.all(np.diff(h.hv_trace_nadir) >= -1e-12)
    assert h.hv_trace_nadir.size == h.X.shape[0]
    # the nadir sits strictly BELOW every feasible point, or the worst design
    # would sweep exactly zero and the trace would start flat
    assert np.all(h.Y[h.feasible] > h.nadir)


def test_a_four_objective_front_is_reachable_and_states_itself(monkeypatch):
    """`(L/D)max` is left out for DIMENSIONALITY, not redundancy — measured
    over the library it is Pearson +0.506 with cruise L/D, so it carries three
    quarters of its variance independently. "We checked" and "we assumed" are
    different claims; this is what makes the first one available."""
    from aerobo.airfoil_select import PARETO_CRITERIA_4, check_pareto_criteria

    assert check_pareto_criteria(None) == PARETO_CRITERIA
    assert check_pareto_criteria(4) == PARETO_CRITERIA_4
    assert check_pareto_criteria(("ldcr", "cm")) == ("ldcr", "cm")
    for bad, match in ((("ldcr", "nope"), "unknown front criteria"),
                       (("ldcr", "ldcr"), "repeated"),
                       (("ldcr",), "at least two"),
                       (5, "no 5-objective")):
        with pytest.raises(ValueError, match=match):
            check_pareto_criteria(bad)

    _install_fake_xfoil(monkeypatch)
    rep = api.pareto_airfoil(budget=12, seed=0, score_weights="gdp-sweep",
                             criteria=4)
    assert rep["conditions"]["criteria"] == list(PARETO_CRITERIA_4)
    assert rep["result"]["gp_failures"] == []
    # REACHABLE is half the name and it was the unguarded half: `n_front` is
    # just `len(rows)` with no floor, so a regression that left the 4-objective
    # path with an empty non-dominated set made the loop below vacuous and
    # everything else here is CONFIG. Assert the run produced a front first.
    assert rep["result"]["n_front"] >= 1
    assert len(rep["front"]) == rep["result"]["n_front"]
    assert [r["rank"] for r in rep["front"]] == list(range(len(rep["front"])))
    for r in rep["front"]:
        assert set(r["objectives"]) == set(PARETO_CRITERIA_4)
        # ...and they are four scores, not a refusal wearing four components
        assert all(np.isfinite(v) for v in r["objectives"].values())
        assert any(v > PENALTY for v in r["objectives"].values())
    # the rank weights follow the criteria set, not a hard-coded three
    assert len(rep["conditions"]["score"]["rank_weights"]) == 4


def test_pareto_is_an_airfoil_objective_now_that_run_delivers_one():
    """It was deliberately absent while ``run`` could not return a front.

    The rule it was absent under is unchanged: a name in that tuple must be a
    mode an evaluation actually delivers. What changed is that ``run`` now
    delivers this one — so the test that used to assert the name was MISSING
    asserts instead that the promise behind it is kept.
    """
    assert api.PARETO_OBJECTIVE_NAME in api.AIRFOIL_OBJECTIVES
    cfg = api.airfoil_run_config(objective="pareto", score_weights="gdp-sweep")
    assert cfg.flags["airfoil_objective"] == "pareto"
    # the seed IS the hypervolume reference point, so it must be in the
    # training set — and that is what keeps `run` and `pareto_airfoil`
    # (warm_start=True by default) from disagreeing
    assert cfg.flags[api.BO_WARM_START_FLAG] is True
    assert api._pareto_settings(cfg.flags)["warm_start"] is True


def test_a_front_build_refuses_to_be_scalar_optimised():
    """The half-measure this objective was held back to avoid.

    A front has no scalar. If ``_build_airfoil`` handed back the composite as
    the callable, a caller could run a weighted sum while believing it had
    searched a front — so the callable RAISES, and only the multi-objective
    path in ``run`` ever produces a result.
    """
    cfg = api.airfoil_run_config(objective="pareto", score_weights="gdp-sweep")
    built = api.PROBLEM_SPECS[cfg.problem_name].build({}, cfg.flags, None)
    with pytest.raises(TypeError, match="FRONT, not a scalar"):
        built.callable(np.asarray(built.problem.w0, dtype=float))
    # …while `evaluate` still works, because a breakdown at a front point is a
    # composite breakdown and the card needs one
    out = built.evaluate(np.asarray(built.problem.w0, dtype=float))
    assert "composite" in out


@pytest.mark.parametrize("key,value,owner", [
    ("pareto_acqf", "qnparego", "pareto"),
    ("pareto_q", 4, "pareto"),
    ("pareto_criteria", 4, "pareto"),
])
def test_the_front_flags_are_refused_on_a_scalar_objective(key, value, owner):
    """A multi-objective acquisition, a batch size and a criteria set mean
    nothing on a weighted sum. Stated there, they RAISE — the same table and
    the same rule as goal_penalty on an ASF run."""
    with pytest.raises(ValueError, match="does not read"):
        api.airfoil_run_config(objective="composite",
                               score_weights="gdp-sweep", **{key: value})


@pytest.mark.parametrize("key,value", [("goal_penalty", 3.0),
                                       ("asf_rho", 0.5),
                                       ("asf_lambda", "unit")])
def test_the_scalar_reference_flags_are_refused_on_a_front(key, value):
    """And in the other direction: a front has neither a hinge nor a min term,
    so a penalty or an augmentation on it is a term no evaluation applies."""
    with pytest.raises(ValueError, match="does not read"):
        api.airfoil_run_config(objective="pareto", score_weights="gdp-sweep",
                               **{key: value})


def test_an_unknown_front_acqf_is_refused_before_any_xfoil_runs():
    """Refused at the CONFIG, not 40 evaluations into the search."""
    with pytest.raises(ValueError, match="pareto acqf"):
        api.airfoil_run_config(objective="pareto", pareto_acqf="nsga2")


def test_run_returns_the_front_and_a_best_x_chosen_in_the_common_currency(
        monkeypatch):
    """``run(objective="pareto")`` — the whole point of lifting it in.

    Asserted as outcomes, not as a restatement of the assembly:

    * the field carries a report `front_rows` can read WITHOUT adaptation;
    * `best_x` is the front row with the highest PLAIN composite J, recomputed
      here from the rows rather than read back from the field that set it;
    * `eval_y` is one VECTOR per evaluation, not a scalar;
    * `history` is the hypervolume trace and is monotone non-decreasing;
    * and it really was qNEHVI that chose the points.
    """
    from gui.v3.stages import airfoil as stage

    _install_fake_xfoil(monkeypatch)
    cfg = api.airfoil_run_config(objective="pareto", score_weights="gdp-sweep",
                                 budget=14, seed=0)
    res = api.run(cfg)

    assert res.front is not None
    assert res.front["conditions"]["objective"] == "pareto"
    assert res.front["result"]["gp_failures"] == []
    # the tested presenter reads the stored field with no adaptation at all
    rows = stage.front_rows(res.front)
    assert len(rows) == len(res.front["front"]) + 1

    # best_x is the argmax of plain composite J OVER THE FRONT — derived here
    # from the rows, so this fails if the choice rule changes
    scored = [(r["composite"], r["x"]) for r in res.front["front"]
              if r["composite"] is not None]
    j_best, x_best = max(scored, key=lambda t: t[0])
    assert res.best_score == pytest.approx(j_best)
    assert res.best_x == pytest.approx(x_best)
    # …and it is NOT max(eval_y): eval_y is a vector per evaluation
    Y = np.asarray(res.eval_y, dtype=float)
    assert Y.ndim == 2 and Y.shape == (res.n_evals, 3)

    hv = np.asarray(res.history, dtype=float)
    assert hv.size == res.n_evals
    assert np.all(np.diff(hv) >= -1e-12)
    assert res.feasible is True and res.breakdown is not None
    assert res.acqf == "qnehvi"


def test_the_front_run_and_pareto_airfoil_are_the_same_search(monkeypatch):
    """ONE search path. Two entry points that disagreed about the band, the
    reference point or the criteria would be two searches wearing one name."""
    _install_fake_xfoil(monkeypatch)
    rep = api.pareto_airfoil(budget=14, seed=0, score_weights="gdp-sweep")
    res = api.run(api.airfoil_run_config(objective="pareto",
                                        score_weights="gdp-sweep",
                                        budget=14, seed=0))

    assert res.front["conditions"] == rep["conditions"]
    assert res.front["seed"] == rep["seed"]
    assert len(res.front["front"]) == len(rep["front"])
    for a, b in zip(res.front["front"], rep["front"]):
        assert a["x"] == pytest.approx(b["x"])
        assert a["objectives"] == pytest.approx(b["objectives"])
    assert res.front["result"]["hypervolume"] == pytest.approx(
        rep["result"]["hypervolume"])


def test_the_shell_offers_the_front_and_says_what_it_costs():
    """Stage 2's menu. The label is an outcome, not decoration: a user who
    picks a front expecting a better single section has been mis-sold, and
    `RESULTS_FRONT_REACHABILITY.md` measured that it IS a worse single-answer
    search at equal budget."""
    from gui.v3.stages import airfoil as stage

    choices = stage.OBJECTIVE_CHOICES(wing_mode=False)
    assert "pareto" in choices
    label = choices["pareto"].lower()
    assert "front" in label
    # the honest half of the label — it must not read as a strict upgrade
    assert "not one winner" in label and "worse" in label
    # and it travels with the other composite objectives, so it gets the same
    # stall sweep, design vector and score card
    assert stage.is_composite("pareto")
    assert set(stage.COMPOSITE_OBJECTIVES) == set(
        api.AIRFOIL_COMPOSITE_OBJECTIVES)


def test_the_shell_asks_for_a_front_the_same_way_it_asks_for_a_composite():
    """`objective_kwargs` is the one place that decides which scalar is
    maximised, whether a wing is flown and which band J is measured on. A
    front is a SECTION run on the frozen band, so it must come back with the
    composite arguments and `wing=None` — a twist law searched against a
    three-objective front is the same freedom-nobody-sees bug."""
    from gui.v3.stages import airfoil as stage

    kw = stage.objective_kwargs("pareto", wing_guess={"mass_kg": 60.0},
                                weights={"ldcr": 1.0}, reference=None)
    assert kw["objective"] == "pareto"
    assert kw["wing"] is None
    assert kw["score_weights"] == {"ldcr": 1.0}


def test_the_front_survives_the_result_round_trip(monkeypatch):
    """The card reads `rep["result"]`, which is `RunResult.to_dict()`. A front
    dropped there would show ONE answer for a search whose answer is the set —
    so the round trip is asserted, and a scalar run is asserted to carry
    nothing so its stored shape is unchanged."""
    from gui.v3.stages import airfoil as stage

    _install_fake_xfoil(monkeypatch)
    rep = api.optimize_airfoil(objective="pareto", score_weights="gdp-sweep",
                               budget=12, seed=0, with_baseline=False,
                               with_section=False)
    stored = rep["result"]["front"]
    assert stored is not None
    assert len(stage.front_rows(stored)) == len(stored["front"]) + 1

    plain = api.optimize_airfoil(objective="composite",
                                 score_weights="gdp-sweep", budget=6, seed=0,
                                 optimiser="sobol", with_baseline=False,
                                 with_section=False)
    assert plain["result"]["front"] is None


def test_a_front_refuses_a_stop_rule_rather_than_ignoring_one():
    """The multi-objective loop has no early-stop hook. Accepting a stop_rule
    and never consulting it would arm a Stop button that does nothing — a
    silent no-op on a CANCEL, which is the worst place in the program to have
    one. It RAISES, and the message says why.

    Asserted as the refusal itself, not as a property of the output: a run that
    merely finished would be indistinguishable from one that ignored the rule.
    """
    cfg = api.airfoil_run_config(objective="pareto", score_weights="gdp-sweep",
                                 budget=8, seed=0)
    fired = {"n": 0}

    def never_stop(i, best):
        fired["n"] += 1
        return False

    with pytest.raises(ValueError, match="cannot be stopped early"):
        api.run(cfg, stop_rule=never_stop)
    # …and it was refused BEFORE any XFOIL ran, not after a search
    assert fired["n"] == 0


def test_the_shell_does_not_send_a_stop_rule_it_would_be_refused_for():
    """Stage 2 passes a stop rule on every scalar run. If it passed one on a
    front the api would raise, so the runner has to know the difference — and
    the source is the single place that decides it."""
    import inspect

    from gui.v3.stages import airfoil as stage

    src = inspect.getsource(stage)
    # the guard exists and is keyed on the objective, not on something the
    # objective merely correlates with
    assert 'is_front = str(oc.get("objective", "cd")) == "pareto"' in src
    assert "**({} if is_front else dict(" in src


def test_the_front_settings_reach_the_search(monkeypatch):
    """A flag that is accepted must change what runs. Stated criteria and a
    stated acquisition are read back off the report the search wrote."""
    _install_fake_xfoil(monkeypatch)
    cfg = api.airfoil_run_config(objective="pareto", score_weights="gdp-sweep",
                                 pareto_acqf="qnparego", pareto_criteria=4,
                                 budget=12, seed=0)
    res = api.run(cfg)
    assert res.front["conditions"]["acqf"] == "qnparego"
    assert len(res.front["conditions"]["criteria"]) == 4
    assert np.asarray(res.eval_y, dtype=float).shape[1] == 4
    for r in res.front["front"]:
        assert len(r["objectives"]) == 4


def test_the_front_run_reports_a_ranked_front_with_every_criterion(monkeypatch):
    _install_fake_xfoil(monkeypatch)
    rep = api.pareto_airfoil(budget=14, seed=0, score_weights="gdp-sweep")

    assert rep["conditions"]["objective"] == "pareto"
    assert rep["conditions"]["criteria"] == list(PARETO_CRITERIA)
    assert rep["result"]["n_evals"] == 14
    assert rep["result"]["n_front"] >= 1
    # …and it was really qNEHVI that chose the points, not the random fallback
    assert rep["result"]["gp_failures"] == []

    ranks = [r["rank"] for r in rep["front"]]
    assert ranks == sorted(ranks) == list(range(len(ranks)))
    v = np.asarray(rep["conditions"]["score"]["rank_weights"])
    scored = [sum(r["objectives"][k] * vi
                  for k, vi in zip(PARETO_CRITERIA, v))
              for r in rep["front"]]
    assert scored == sorted(scored, reverse=True)

    for r in rep["front"]:
        # ALL SIX criteria on every row: the three it was not searched on are
        # exactly the ones a user picking off a front needs to see
        assert set(r["scores"]) == set(CRITERIA)
        assert r["composite"] is not None
        assert set(r["raw"]) == {"ldcr", "clmax", "cm"}
        assert r["cd_at_cl"] is not None and r["tc"] is not None


def test_the_front_run_states_what_it_measured_against(monkeypatch):
    _install_fake_xfoil(monkeypatch)
    rep = api.pareto_airfoil(budget=12, seed=0, score_weights="gdp-sweep")
    cond, seed = rep["conditions"], rep["seed"]

    assert cond["score"]["reference_sha"]
    assert cond["reference_point"] == pytest.approx(
        [seed["objectives"][k] for k in PARETO_CRITERIA])
    assert cond["acqf"] == "qnehvi" and cond["q"] == 1
    assert len(rep["result"]["hypervolume_trace"]) == rep["result"]["n_evals"]


def test_an_unknown_front_acqf_is_refused_at_the_surface():
    with pytest.raises(ValueError, match="pareto acqf"):
        api.pareto_airfoil(budget=8, acqf="nsga2")


def test_the_front_rows_carry_the_seed_and_every_criterion(monkeypatch):
    """The presenter the stage-2 card reads. Pure, so the ordering and the
    deltas are testable without a browser."""
    from gui.v3.stages import airfoil as stage

    _install_fake_xfoil(monkeypatch)
    rep = api.pareto_airfoil(budget=12, seed=0, score_weights="gdp-sweep")
    rows = stage.front_rows(rep)

    assert len(rows) == len(rep["front"]) + 1
    seed_rows = [r for r in rows if r["is_seed"]]
    assert len(seed_rows) == 1 and seed_rows[0]["rank"] == "seed"
    # the seed's own deltas against itself are zero, which is what makes the
    # column readable at all
    for k in PARETO_CRITERIA:
        assert seed_rows[0][f"d_{k}"] == "+0.0"
    assert [r["rank"] for r in rows if not r["is_seed"]] == \
        [str(i + 1) for i in range(len(rep["front"]))]
    for r in rows:
        assert set(PARETO_CRITERIA) <= set(r)
        assert r["composite"] != "—"
    assert stage.front_rows(None) == [] and stage.front_rows({}) == []


def test_the_front_refuses_a_seed_it_cannot_score(monkeypatch):
    """A hypervolume needs a reference point, and inventing one out of a failed
    evaluation is the mistake the goal composite already refuses to make."""
    def broken(coords, re, mach, alphas, **kw):
        a = np.asarray(alphas, dtype=float)
        return xfoil_run.XfoilPolarResult(
            alpha_deg=a[:1], cl=np.array([0.1]), cd=np.array([0.01]),
            cm=np.array([-0.01]), n_requested=a.size)

    monkeypatch.setattr(xfoil_run, "run_xfoil_polar", broken)
    with pytest.raises(ValueError, match="reference point"):
        api.pareto_airfoil(budget=8)
