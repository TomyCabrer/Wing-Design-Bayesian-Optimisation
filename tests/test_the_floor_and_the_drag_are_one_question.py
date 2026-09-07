"""Minimising the drag under a downforce FLOOR, and why that pair had to exist.

The car families could state the downforce/drag trade three ways, and every
one of them scalarises it: ``cz`` and ``downforce`` ignore the drag,
``efficiency`` divides by it, ``laptime`` weights it at the circuit's own
exchange rate. All three pick ONE point of the trade and none of them can be
pointed at a different one without changing the question.

The fourth way is the epsilon-constraint: state the downforce you need
(``downforce_min_n``) and minimise the drag that buys it. It needed a scorer,
because ``ENDPLATE_OBJECTIVES`` had none — and the reason it is worth adding
rather than approximating is that the (CZ, CD) Pareto front of these families
is strongly CONCAVE, which no linear weight can see.

The tests assert OUTCOMES, not that the dict has new keys:

* every objective a family DECLARES, that family can SCORE (the exact defect
  ``laptime`` shipped with one family across — a menu entry that constructed
  and then raised ``KeyError`` on the first evaluation);
* the score is strictly decreasing in the drag, which is the whole contract
  of a minus-transform on a maximising harness;
* under one floor, ``drag`` pays less drag than ``efficiency`` and both clear
  the floor — the measurement that justified the pair;
* ``efficiency`` OVERSHOOTS the floor and ``drag`` does not, which is WHY it
  pays more: a ratio under a floor climbs above it chasing a better quotient;
* ``cd`` is refused at a free reference area and ``drag`` is not, on every
  family — the mirror of the ``cz``/``downforce`` refusal that was already
  there;
* and the PENALTY sentinel trap the ``drag`` docstring warns about is REAL at
  a speed inside the offered band, not a hypothetical.
"""

import numpy as np
import pytest

from aerobo.carwing import (CAR_OBJECTIVES, PENALTY, CarWingProblem,
                            evaluate_car_wing)
from aerobo.carwing_multi import (CAR_MULTI_OBJECTIVES, CarWingMultiProblem,
                                  evaluate_car_wing_multi)
from aerobo.cartrack import synthetic_lap
from aerobo.endplate import (ENDPLATE_OBJECTIVES, CarWingEndplateProblem,
                             evaluate_car_wing_endplate)

#: (label, objective set, problem class, evaluator) for every car family that
#: declares an objective menu. Read off the modules so a family that gains an
#: objective joins these tests without editing them.
FAMILIES = [
    ("carwing", CAR_OBJECTIVES, CarWingProblem, evaluate_car_wing),
    ("endplate", ENDPLATE_OBJECTIVES, CarWingEndplateProblem,
     evaluate_car_wing_endplate),
    ("multi", CAR_MULTI_OBJECTIVES, CarWingMultiProblem,
     evaluate_car_wing_multi),
]


def _problem(P, objective, **kw):
    """A box-centre problem on one objective, with a circuit iff it needs one."""
    if objective == "laptime":
        kw.setdefault("track_spec", synthetic_lap())
    return P(objective=objective, **kw)


def _centre(prob):
    b = prob.bounds
    return 0.5 * (b[:, 0] + b[:, 1])


@pytest.mark.parametrize("fam,objs,P,ev", FAMILIES,
                         ids=[f[0] for f in FAMILIES])
def test_every_objective_a_family_offers_is_one_it_can_score(fam, objs, P, ev):
    """A menu entry that cannot be scored is a KeyError with a label on it."""
    for name in objs:
        prob = _problem(P, name)
        out = ev(_centre(prob), prob)
        assert "score" in out, f"{fam}/{name} returned no score"
        assert np.isfinite(out["score"]), f"{fam}/{name} scored {out['score']}"
        assert out["objective"] == name
        assert out["objective_label"] == objs[name]


@pytest.mark.parametrize("fam,objs,P,ev", FAMILIES,
                         ids=[f[0] for f in FAMILIES])
@pytest.mark.parametrize("name", ["cd", "drag"])
def test_the_drag_score_falls_as_the_drag_rises(fam, objs, P, ev, name):
    """The transform is decreasing, so the harness's max IS the drag's min.

    Mutated through the rigging incidence, which every car family carries and
    which raises the drag monotonically over the lower half of its band.
    """
    prob = _problem(P, name)
    x = _centre(prob)
    i_alpha = list(prob.param_names).index("alpha_deg") \
        if hasattr(prob, "param_names") else 3
    lo, hi = prob.bounds[i_alpha]
    scores, drags = [], []
    for a in np.linspace(lo + 0.15 * (hi - lo), lo + 0.65 * (hi - lo), 5):
        y = x.copy()
        y[i_alpha] = a
        out = ev(y, prob)
        if "CD" not in out:
            continue
        scores.append(out["score"])
        drags.append(out["CD"] if name == "cd" else out["drag_N"])
    assert len(scores) >= 4, f"{fam}: too few flyable points to test the sign"
    assert drags == sorted(drags), f"{fam}: the mutation did not raise the drag"
    assert scores == sorted(scores, reverse=True), \
        f"{fam}/{name}: score did not fall as the drag rose: {scores}"
    # and the score IS the quantity, negated — not merely ranked with it
    assert scores[0] == pytest.approx(-drags[0], rel=1e-12)


def _best_under_floor(objective, floor_n, n_alpha=25):
    """Box-centre plate rows, alpha swept: the winner this objective picks.

    A deliberate exhaustive sweep rather than an optimiser call — the claim
    under test is about the OBJECTIVE, and a search would put its own
    convergence between the assertion and the thing asserted.
    """
    prob = CarWingEndplateProblem(objective=objective,
                                  downforce_min_n=float(floor_n))
    x = _centre(prob)
    lo, hi = prob.bounds[3]
    best = None
    for a in np.linspace(lo, hi, n_alpha):
        for h in np.linspace(0.0, 0.44, 12):
            for cr in np.linspace(0.5, 3.0, 11):
                y = x.copy()
                y[3], y[4], y[6] = a, h, cr
                out = evaluate_car_wing_endplate(y, prob)
                if "CD" not in out or out["downforce_N"] < floor_n:
                    continue
                if best is None or out["score"] > best["score"]:
                    best = out
    assert best is not None, f"no design clears {floor_n} N"
    return best


@pytest.mark.parametrize("floor_n", [200.0, 400.0])
def test_min_drag_under_a_floor_pays_less_drag_than_max_efficiency(floor_n):
    """The measurement the pair was added for, re-derived rather than pinned."""
    d = _best_under_floor("drag", floor_n)
    e = _best_under_floor("efficiency", floor_n)
    assert d["downforce_N"] >= floor_n, "the drag winner missed the floor"
    assert e["downforce_N"] >= floor_n, "the efficiency winner missed the floor"
    assert d["drag_N"] <= e["drag_N"], (
        f"at a {floor_n:.0f} N floor 'drag' paid {d['drag_N']:.2f} N and "
        f"'efficiency' paid {e['drag_N']:.2f} N — the pair buys nothing")


def test_a_ratio_under_a_floor_overshoots_it_and_that_is_what_costs():
    """WHY efficiency pays more: it climbs off the floor for a better quotient.

    At the low floor the constraint is slack for the ratio's optimum, so the
    two formulations genuinely disagree; the drag winner sits AT the floor and
    the efficiency winner well above it.
    """
    floor_n = 200.0
    d = _best_under_floor("drag", floor_n)
    e = _best_under_floor("efficiency", floor_n)
    d_over = d["downforce_N"] / floor_n - 1.0
    e_over = e["downforce_N"] / floor_n - 1.0
    assert e_over > d_over, (
        "the ratio did not overshoot further than the drag minimiser "
        f"({e_over:+.1%} vs {d_over:+.1%}) — the mechanism claimed in "
        "carwing.CAR_OBJECTIVES is not the one operating")
    # the overshoot is bought, not free: it costs drag the floor never asked for
    assert e["drag_N"] > d["drag_N"]
    # and the efficiency winner IS more efficient — each wins its own scoreboard
    assert e["CZ"] / e["CD"] > d["CZ"] / d["CD"]


@pytest.mark.parametrize("fam,objs,P,ev", FAMILIES,
                         ids=[f[0] for f in FAMILIES])
def test_a_drag_coefficient_is_refused_at_a_free_area_and_the_force_is_not(
        fam, objs, P, ev):
    """The mirror of the cz/downforce refusal, and for the mirrored reason."""
    band = (0.15, 0.90)
    with pytest.raises(ValueError, match="free reference area"):
        _problem(P, "cd", area_bounds_m2=band)
    prob = _problem(P, "drag", area_bounds_m2=band)
    out = ev(_centre(prob), prob)
    assert np.isfinite(out["score"])
    assert out["score"] == pytest.approx(-out["drag_N"], rel=1e-12)


def test_the_penalty_sentinel_really_does_bite_the_drag_objective():
    """The docstring's warning, re-derived from the model at a speed in band.

    ``PENALTY`` is -100.0 and ``drag`` scores -D in newtons, so a design
    dragging more than 100 N scores WORSE than a refusal. The warning is only
    worth carrying if it is reachable, so this asserts it IS — and asserts the
    published escape (a coefficient objective) is not affected.
    """
    from dataclasses import replace

    base = CarWingEndplateProblem(objective="drag")
    fast = replace(base, V=85.0)
    x = _centre(fast)
    x[3] = fast.bounds[3][1]          # rigging at its ceiling: the draggiest
    x[6] = fast.bounds[6][1]          # widest plate chord
    out = evaluate_car_wing_endplate(x, fast)
    assert "CD" in out, "the corner used to demonstrate the trap does not fly"
    assert out["drag_N"] > -PENALTY, (
        f"the trap is unreachable at V={fast.V} m/s (worst drag "
        f"{out['drag_N']:.1f} N) — carwing.CAR_OBJECTIVES overstates it")
    assert out["score"] < PENALTY, "a >100 N design did not score below refusal"
    # the coefficient half of the pair is nowhere near the sentinel
    cd_out = evaluate_car_wing_endplate(x, replace(fast, objective="cd"))
    assert cd_out["score"] > PENALTY
