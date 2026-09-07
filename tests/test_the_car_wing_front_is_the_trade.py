"""The car-wing PARETO FRONT: the trade itself, instead of one point of it.

Every scalar objective in ``carwing.CAR_OBJECTIVES`` answers the downforce/
drag question by deciding it in advance — a ratio, a weight, or a floor. The
front decides nothing: it searches both objectives at once and hands back the
non-dominated set, and the choosing happens after, against whatever the user
actually has (a circuit, a rule, a budget).

That matters here because the front is CONCAVE. Measured over a 14 803-design
grid of the endplate family whose front has 618 points, a linear weight on
(CZ, -CD) reaches 6.3 % of it and a downforce/efficiency weight 2.6 %; the
biggest gap a weight sweep jumps is 13.9 % of the front in one step. No
normalisation recovers a concave region — only searching both objectives
does.

These assert the PROPERTIES a front has to have, not that the call returns
something:

* the objectives are FORCES, and the sign convention is the maximising one
  ``optimize.mobo`` requires — a drag column that was not negated would make
  the search hunt the draggiest wing it can find;
* the reference point is the INCUMBENT, so the hypervolume answers "did
  anything beat the wing I have, on both?" and is comparable between runs;
* a refusal returns the penalty at the declared margin WIDTH, per family —
  the vector twin of ``fg_car_wing``'s contract;
* every returned row is really non-dominated and really feasible;
* the front spans a genuine range rather than collapsing to one design;
* the order is STATED (downforce, or the lap where there is a circuit) and
  never a weight;
* and the entry point refuses a family that has no downforce to report,
  naming the ones that do.
"""

from __future__ import annotations

import numpy as np
import pytest

from aerobo import api
from aerobo.carwing import (CAR_PARETO_OBJECTIVES, CAR_PARETO_SIGNS, G_FAIL,
                            PENALTY, CarWingProblem, car_pareto_point,
                            car_pareto_reference, evaluate_car_wing,
                            make_car_pareto_fg)
from aerobo.carwing_multi import CarWingMultiProblem, evaluate_car_wing_multi
from aerobo.endplate import CarWingEndplateProblem, evaluate_car_wing_endplate

FAMILIES = [
    ("carwing", CarWingProblem, evaluate_car_wing),
    ("endplate", CarWingEndplateProblem, evaluate_car_wing_endplate),
    ("multi", CarWingMultiProblem, evaluate_car_wing_multi),
]


def _centre(prob):
    b = np.asarray(prob.bounds, dtype=float)
    return 0.5 * (b[:, 0] + b[:, 1])


@pytest.mark.parametrize("fam,P,ev", FAMILIES, ids=[f[0] for f in FAMILIES])
def test_the_front_maximises_downforce_and_minimises_drag(fam, P, ev):
    """The sign convention, asserted against the breakdown it is built from.

    Get the drag's sign wrong and the search maximises it: the run would come
    back with the draggiest wing in the box and every other assertion here
    would still pass.
    """
    prob = P()
    out = ev(_centre(prob), prob)
    y = car_pareto_point(out)
    assert CAR_PARETO_OBJECTIVES == ("downforce_N", "drag_N")
    assert CAR_PARETO_SIGNS == (+1.0, -1.0)
    assert y[0] == pytest.approx(out["downforce_N"])
    assert y[1] == pytest.approx(-out["drag_N"])
    assert y[1] < 0.0, "the drag column is not negated; the search would want more"

    # ...and MORE drag at the same downforce really does score lower
    prob2 = P(V=prob.V)
    x = _centre(prob2)
    i_alpha = 3
    lo, hi = prob2.bounds[i_alpha]
    a_lo, a_hi = lo + 0.2 * (hi - lo), lo + 0.6 * (hi - lo)
    y_lo = car_pareto_point(ev(np.r_[x[:i_alpha], a_lo, x[i_alpha + 1:]], prob2))
    y_hi = car_pareto_point(ev(np.r_[x[:i_alpha], a_hi, x[i_alpha + 1:]], prob2))
    assert y_hi[0] > y_lo[0], "the mutation did not raise the downforce"
    assert y_hi[1] < y_lo[1], "the drag column did not fall as the drag rose"


@pytest.mark.parametrize("fam,P,ev", FAMILIES, ids=[f[0] for f in FAMILIES])
def test_a_refused_design_returns_the_penalty_at_the_declared_width(fam, P, ev):
    """``fg_car_wing``'s contract, componentwise.

    A margin vector at the wrong WIDTH is a shape error inside the optimiser
    rather than a bad score, and a front run has one fg per family — so the
    width has to follow the problem's own declaration, not a constant.
    """
    prob = P()
    fg = make_car_pareto_fg(prob, ev)
    b = np.asarray(prob.bounds, dtype=float)
    # a corner far outside anything flyable: every family refuses it
    y, g = fg(np.where(np.arange(b.shape[0]) % 2 == 0, b[:, 0], b[:, 1]) * 1e3)
    assert len(y) == len(CAR_PARETO_OBJECTIVES)
    assert list(y) == [PENALTY, PENALTY]
    assert len(g) == prob.n_constraints, \
        f"{fam}: {len(g)} margins for {prob.n_constraints} declared"
    assert all(v == G_FAIL for v in g)


@pytest.mark.parametrize("fam,P,ev", FAMILIES, ids=[f[0] for f in FAMILIES])
def test_the_reference_point_is_the_incumbent(fam, P, ev):
    """Fixed before the run and read off the box centre, so the hypervolume
    is "did anything beat what I have, on both?" and two runs compare."""
    prob = P()
    ref = car_pareto_reference(prob, ev)
    centre = car_pareto_point(ev(_centre(prob), prob))
    assert ref == pytest.approx(centre)
    # ...so the incumbent itself contributes exactly zero volume
    assert not (ref[0] > centre[0] or ref[1] > centre[1])


def test_the_front_is_really_a_front_and_really_spans_the_trade():
    """One real search. Every row non-dominated AND feasible, and the set is
    a range rather than one design repeated."""
    r = api.pareto_car_wing("car rear wing", flags={"V": 55.0},
                            budget=32, seed=0)
    rows = r["front"]
    assert rows, "the search returned no front at all"
    Y = np.array([[row["downforce_N"], -row["drag_N"]] for row in rows])
    for i in range(len(Y)):
        others = np.delete(Y, i, axis=0)
        if not len(others):
            continue
        dominated = np.all(others >= Y[i], axis=1) & np.any(others > Y[i], axis=1)
        assert not dominated.any(), (
            f"row {i} (Fz {Y[i, 0]:.1f} N, D {-Y[i, 1]:.2f} N) is dominated")
    for row in rows:
        assert row["breakdown"]["feasible"], "an infeasible design reached the front"
        assert row["downforce_N"] == pytest.approx(
            row["breakdown"]["downforce_N"])
        assert row["drag_N"] == pytest.approx(row["breakdown"]["drag_N"])
    if len(rows) > 1:
        fz = np.array([row["downforce_N"] for row in rows])
        assert fz.max() / fz.min() > 1.2, (
            "the front collapsed to one design — it is not spanning the trade")
    assert r["result"]["n_front"] == len(rows)
    assert r["objectives"] == ["downforce_N", "drag_N"]


def test_the_order_is_stated_and_never_a_weight():
    """Downforce descending with no circuit. An order, not a verdict."""
    r = api.pareto_car_wing("car rear wing", flags={"V": 55.0},
                            budget=24, seed=1)
    assert r["ordered_by"] == "downforce_N"
    fz = [row["downforce_N"] for row in r["front"]]
    assert fz == sorted(fz, reverse=True)


def test_a_family_with_no_downforce_to_report_is_refused_by_name():
    """...and the refusal names the families that DO, plus the section front.

    The defect this closes is the one ``laptime`` shipped with: an objective
    offered to a family that cannot answer it, which constructs and then
    raises somewhere inside the run.
    """
    with pytest.raises(ValueError) as exc:
        api.pareto_car_wing("mission wing")
    msg = str(exc.value)
    assert "car rear wing" in msg
    assert "pareto_airfoil" in msg
    # the offered set is DERIVED from the registry, not listed in the module
    fams = api.car_pareto_families()
    assert fams and all("car_objective" in api.PROBLEM_SPECS[f].flags
                        for f in fams)
    for f in fams:
        assert f in api.PROBLEM_SPECS
