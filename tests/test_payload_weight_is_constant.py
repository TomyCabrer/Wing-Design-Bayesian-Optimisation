"""The payload weight is the same constant for every candidate.

``sizing.py`` states the reason the sized problems score PAYLOAD L/D rather
than L/D:

    f = W_fixed / D. With W varying across candidates "maximise L/D" would
    reward a heavier wing that happens to lift better; W_fixed is the same
    constant for every candidate, so maximising W_fixed/D IS minimising drag.

That only holds if ``W_fixed`` really is constant. Five families read its
default — the weight the family's own trim target implies,
``weight_for(CL_target, V, S)`` — AFTER replacing the problem with the
candidate's size, so ``prob.S`` was the CANDIDATE's area and the "fixed"
payload grew with the wing. The objective then rewarded area outright, which
is the exact failure the payload formulation exists to prevent.

The reference is now read from the problem's own published area, before the
size replaces it. Gated here across every sized family, because the bug was
identical in each and a comment in one of them would not have caught it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aerobo import api     # noqa: E402

#: one sized problem per family that carries the size modifier
SIZED = (
    "trim wing + free planform",
    "winglet_capped + free planform",
    "tail + free planform",
    "tail [designed tail] + free planform",
    "tandem + free planform",
    "wing+airfoil (coupled) + free planform",
)


def _span_area_rows(labels) -> tuple[int, int]:
    return labels.index("b_m"), labels.index("S_m2")


@pytest.mark.parametrize("name", SIZED)
def test_the_fixed_weight_does_not_follow_the_candidate_area(name):
    """Two candidates that differ ONLY in area must report the same payload.

    The area row is swept across its own box rather than probed at one
    point, because the families reach feasibility over different parts of
    it — a single probe would have skipped the very families the bug lived
    in.
    """
    built = api.PROBLEM_SPECS[name].build({}, {}, None)
    labels = list(built.param_labels)
    _i_b, i_s = _span_area_rows(labels)
    x = built.bounds.mean(axis=1)
    found = []
    for frac in np.linspace(0.0, 1.0, 21):
        trial = x.copy()
        trial[i_s] = (built.bounds[i_s, 0]
                      + frac * (built.bounds[i_s, 1] - built.bounds[i_s, 0]))
        out = built.evaluate(trial)
        if out["feasible"]:
            found.append(out)
    assert len(found) >= 2, f"{name}: no feasible pair anywhere on the area row"
    a, b = found[0], found[-1]
    assert b["S_m2"] > a["S_m2"]
    assert b["W_fixed_N"] == pytest.approx(a["W_fixed_N"], rel=1e-12), (
        "the payload weight followed the area — payload L/D would then "
        "reward growing the wing rather than reducing drag")


@pytest.mark.parametrize("name", SIZED)
def test_the_score_is_that_constant_over_the_drag(name):
    built = api.PROBLEM_SPECS[name].build({}, {}, None)
    x = built.bounds.mean(axis=1)
    out = built.evaluate(x)
    if not out["feasible"]:
        # walk the area row until something flies (see the sweep above)
        i_s = list(built.param_labels).index("S_m2")
        for frac in np.linspace(0.0, 1.0, 21):
            trial = x.copy()
            trial[i_s] = (built.bounds[i_s, 0] + frac
                          * (built.bounds[i_s, 1] - built.bounds[i_s, 0]))
            out = built.evaluate(trial)
            if out["feasible"]:
                break
    assert out["feasible"], f"{name}: nothing flies on the area row"
    assert out["score"] == pytest.approx(out["W_fixed_N"] / out["D_N"],
                                         rel=1e-9)


def test_a_sized_problem_is_heavier_than_the_one_it_extends():
    """sizing.py's stated consequence: the wing's structure is ADDED to the
    weight the fixed-CL problem already implies."""
    built = api.PROBLEM_SPECS["trim wing + free planform"].build({}, {}, None)
    out = built.evaluate(built.bounds.mean(axis=1))
    assert out["W_total_N"] > out["W_fixed_N"]
    assert out["W_wing_N"] == pytest.approx(out["W_total_N"]
                                            - out["W_fixed_N"])


def test_an_explicit_fixed_weight_is_used_as_given():
    from aerobo import objective

    prob = objective.Problem(mode="trim", size_free=True, W_fixed_N=900.0)
    out = objective.evaluate(np.array([0.6, 0.0, -2.0, 12.0, 12.0]), prob)
    assert out["feasible"], out["reason"]
    assert out["W_fixed_N"] == pytest.approx(900.0)
