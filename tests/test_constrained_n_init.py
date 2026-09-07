"""A constrained initial design is a different question from an unconstrained one.

Unconstrained, the Sobol seed only has to start the surrogate somewhere, and
the wing kind's measured rule (0.5 x dim, 15 cases) is about convergence.
Constrained, the same seed decides whether the run ever observes a FEASIBLE
point — and a run that never does not recover, it rides a box corner
(``optimize.feasible``). Measured on the reported 20-D box, where 4.7 % of
draws fly: 10 points returned NOTHING on 2 of 3 seeds, 21 returned a design on
3 of 3.

So the recommendation carries a constrained arm. These tests hold the two
things that can silently go wrong with it: that it only applies where it was
measured to matter, and that the unconstrained rule is untouched.
"""
import json
from pathlib import Path

from aerobo.optimize import budget

DATA = Path(__file__).resolve().parents[1] / "data" / "search_budget.json"


def _spec():
    return json.loads(DATA.read_text())["kinds"]["wing"]["n_init"]


def test_the_constrained_arm_asks_for_at_least_the_dimension():
    """The regime claim, not a tuned constant: a seed smaller than the design
    vector on a mostly-refused box is the blind case."""
    for dim in (12, 20, 30):
        plan = budget.recommend(dim=dim, kind="wing", constrained=True)
        assert plan.n_init >= dim, (dim, plan.n_init)


def test_the_unconstrained_rule_is_untouched():
    """15 measured cases say 0.5 x dim there, and this change must not have
    moved them — the two arms answer different questions."""
    spec = _spec()
    for dim in (8, 20, 30):
        plan = budget.recommend(dim=dim, kind="wing", constrained=False)
        want = int(round(float(spec["per_dim"]) * dim))
        want = max(int(spec["min"]), min(int(spec["max"]), want))
        assert plan.n_init == min(want, plan.budget - 1), dim


def test_the_two_arms_differ_where_the_measurement_says_they_should():
    """A 20-D constrained wing is the case that was measured; if these ever
    coincide again the constrained arm has been dropped or overridden."""
    con = budget.recommend(dim=20, kind="wing", constrained=True)
    unc = budget.recommend(dim=20, kind="wing", constrained=False)
    assert con.n_init > unc.n_init
    assert con.flags()["bo_n_init"] == con.n_init


def test_the_rule_travels_as_a_flag_the_run_honours():
    """A recommendation with no parameter to pass it through is a
    recommendation nobody follows — this repo has shipped one before."""
    from aerobo import api

    plan = budget.recommend(dim=20, kind="wing", constrained=True)
    cfg = api.RunConfig(problem_name="trim wing + free planform",
                        flags=plan.flags(), optimiser="bo", budget=60, seed=0)
    assert api._bo_n_init(cfg) == plan.n_init
    n_init, n_iter = api._bo_split(60, 20, api._bo_n_init(cfg))
    assert n_init == plan.n_init and n_iter == 60 - plan.n_init


def test_the_arm_is_recorded_with_what_measured_it():
    """A measured entry that cannot say what measured it is an opinion."""
    arm = _spec()["constrained"]
    assert arm["measured"] is True
    assert "feasibility_guide_study" in arm["source"]
    assert arm["seeds"] >= 3 and "0/0/12" in arm["why"]
