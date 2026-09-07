"""What a refused design looks like to the surrogate.

The failure contract (-100.0 for a design the physics cannot score) is a
LABEL. These tests pin the separation between that label and the number the
GP is trained on: the transform may change what the model sees and must
never change what the run recorded, what counts as feasible, or which
design wins.
"""

from __future__ import annotations

import numpy as np
import pytest

from aerobo import api
from aerobo.optimize import refusal as R
from aerobo.optimize.bo import run_bo
from aerobo.optimize.constrained import run_bo_constrained


# ------------------------------------------------------------ the transform

def test_sentinel_mode_is_the_identity():
    """The default must be bit-for-bit the legacy path — the values the GP
    was trained on before this module existed."""
    y = [1.0, -100.0, 2.5, -100.0, 0.25]
    out = R.impute(y, "sentinel")
    assert np.array_equal(out, np.array(y))


def test_worst_puts_a_refusal_at_the_worst_design_that_flew():
    y = [1.0, -100.0, 2.0, 0.5]
    out = R.impute(y, "worst")
    assert out.tolist() == [1.0, 0.5, 2.0, 0.5]


def test_worst_margin_keeps_a_refusal_strictly_worse():
    """A plateau gives the model no gradient out of the refusal region."""
    y = [1.0, -100.0, 2.0, 0.5]
    out = R.impute(y, "worst_margin")
    assert out[1] < min(v for v in y if v != -100.0)
    # ...but on the scale of the feasible spread, not 100 units below it
    assert out[1] == pytest.approx(0.5 - R.MARGIN * (2.0 - 0.5))


def test_the_scale_is_the_point():
    """The defect being fixed, stated as a number: with the sentinel in the
    training set, standardised feasible values collapse to a hair's width."""
    y = np.array([-0.0050, -0.0060, -100.0, -0.0055])     # a Cd case
    raw = (y - y.mean()) / y.std()
    fixed = R.impute(y, "worst")
    fixed = (fixed - fixed.mean()) / fixed.std()
    spread_raw = np.ptp(raw[[0, 1, 3]])
    spread_fixed = np.ptp(fixed[[0, 1, 3]])
    assert spread_raw < 0.02                  # the three real designs, merged
    assert spread_fixed > 1.0                 # ...and told apart again


def test_a_value_that_is_not_the_sentinel_is_never_touched():
    y = [-99.9, -100.5, 3.0, -100.0]
    out = R.impute(y, "worst")
    assert out.tolist() == [-99.9, -100.5, 3.0, -100.5]


def test_nothing_feasible_yet_leaves_the_values_alone():
    """No feasible observation means no scale to be aware of. The degenerate
    training set is the BO loop's problem (it falls back to a random point),
    not something to paper over with an invented number."""
    y = [-100.0, -100.0]
    for mode in R.MODES:
        assert R.impute(y, mode).tolist() == y


def test_a_single_feasible_value_still_gets_a_margin():
    out = R.impute([-100.0, 4.0, -100.0], "worst_margin")
    assert out[0] < 4.0 and np.isfinite(out[0]) and out[0] == out[2]


def test_impute_does_not_mutate_its_input():
    y = np.array([1.0, -100.0])
    R.impute(y, "worst")
    assert y[1] == -100.0


def test_an_unknown_mode_is_refused_by_name():
    with pytest.raises(ValueError, match="unknown refusal mode"):
        R.impute([1.0], "worst_ever")


def test_minimising_is_refused_rather_than_guessed():
    """-100 is only 'the worst' if bigger is better."""
    with pytest.raises(ValueError, match="MAXIMISE"):
        R.impute([1.0, -100.0], "worst", maximize=False)


# ------------------------------------------------------------- the BO loops

def _refusing(d=2, bad=lambda x: x[0] > 0.6):
    """A toy objective on the repo's contract: real values of order 1e-3,
    exactly -100.0 where it refuses."""
    def f(x):
        return -100.0 if bad(x) else -1e-3 * float(np.sum((x - 0.3) ** 2))
    return f, np.tile([0.0, 1.0], (d, 1))


def test_run_bo_records_the_true_sentinel_whatever_the_mode():
    f, b = _refusing()
    res = run_bo(f, b, n_init=6, n_iter=4, seed=0, refusal="worst")
    refused = [i for i, x in enumerate(res.X) if x[0] > 0.6]
    assert refused, "the toy problem must refuse something"
    assert all(res.y[i] == -100.0 for i in refused)
    assert res.refusal == "worst"


def test_run_bo_sentinel_mode_is_bit_for_bit_the_default():
    f, b = _refusing()
    a = run_bo(f, b, n_init=6, n_iter=6, seed=1)
    c = run_bo(f, b, n_init=6, n_iter=6, seed=1, refusal="sentinel")
    assert np.array_equal(a.X, c.X) and np.array_equal(a.y, c.y)


def test_the_stand_in_is_recomputed_before_every_fit(monkeypatch):
    """The imputed value must track the CURRENT history, not be computed once.

    npj Comput. Mater. 8 (2022) 'floor padding' — the published form of this
    trick — is explicitly a running quantity: a failure is set to the worst
    success *so far*, and the padding updates as new data arrives. Hoisting
    ``impute`` out of the BO loop (an easy and plausible optimisation, since
    it looks like a pure preprocessing step) would freeze the stand-in at its
    n_init value and silently break that property, while still passing every
    other test in this file.
    """
    seen: list[tuple[np.ndarray, np.ndarray]] = []
    real = R.impute

    def spy(y, mode=R.DEFAULT_MODE, **kw):
        out = real(y, mode, **kw)
        seen.append((np.array(y, dtype=float), np.array(out, dtype=float)))
        return out

    monkeypatch.setattr(R, "impute", spy)
    f, b = _refusing()
    n_init, n_iter = 6, 5
    run_bo(f, b, n_init=n_init, n_iter=n_iter, seed=3, refusal="worst")

    assert len(seen) == n_iter, "imputed once per FIT, not once per run"
    for i, (y_in, _y_out) in enumerate(seen):
        assert y_in.size == n_init + i, (
            "each fit must see the whole history to date; a frozen stand-in "
            "would be recomputed from a stale array")
    for y_in, y_out in seen:
        bad = y_in == R.PENALTY
        if not bad.any() or (~bad).sum() == 0:
            continue
        assert np.allclose(y_out[bad], y_in[~bad].min()), (
            "the stand-in must be THIS fit's worst feasible value")
        assert np.array_equal(y_out[~bad], y_in[~bad]), "reals untouched"


def test_the_stand_in_follows_a_worsening_history():
    """The running quantity has to actually move when the worst gets worse."""
    y = [-1.0, R.PENALTY, -2.0, -5.0]
    seen = [float(R.impute(y[:n], "worst")[1]) for n in (3, 4)]
    assert seen == [-2.0, -5.0], (
        "a refusal imputed from a 3-point history must be re-imputed when a "
        "worse 4th point arrives")


def test_imputation_changes_where_bo_looks():
    """If it did not, there would be nothing to measure."""
    f, b = _refusing()
    a = run_bo(f, b, n_init=6, n_iter=8, seed=2)
    c = run_bo(f, b, n_init=6, n_iter=8, seed=2, refusal="worst")
    assert np.array_equal(a.X[:6], c.X[:6])        # same Sobol seed design
    assert not np.array_equal(a.X[6:], c.X[6:])    # ...different search


def test_constrained_bo_keeps_the_history_and_the_feasibility_true():
    def f_and_g(x):
        if x[0] > 0.6:
            return -100.0, -1.0
        return -1e-3 * float(np.sum(x ** 2)), float(0.5 - x[1])
    b = np.tile([0.0, 1.0], (2, 1))
    h = run_bo_constrained(f_and_g, b, n_init=6, n_iter=4, seed=0,
                           refusal="worst")
    for i, x in enumerate(h.X):
        refused = x[0] > 0.6
        assert (h.y[i] == -100.0) == refused
        # a refusal reports a clearly-infeasible margin (the contract); every
        # design that flew reports its true one
        assert (h.g[i] >= 0.0) == (not refused and x[1] <= 0.5)
    assert h.best_y > -100.0
    assert h.meta["refusal"] == "worst"


def test_constrained_sentinel_mode_is_bit_for_bit_the_default():
    def f_and_g(x):
        return (-100.0, -1.0) if x[0] > 0.6 else (float(-x[0]), float(x[1]))
    b = np.tile([0.0, 1.0], (2, 1))
    a = run_bo_constrained(f_and_g, b, n_init=6, n_iter=6, seed=3)
    c = run_bo_constrained(f_and_g, b, n_init=6, n_iter=6, seed=3,
                           refusal="sentinel")
    assert np.array_equal(a.X, c.X) and np.array_equal(a.y, c.y)


# ------------------------------------------------------------------ the flag

def test_the_flag_is_validated_at_the_api_boundary():
    cfg = api.RunConfig(problem_name="car rear wing",
                        flags={api.BO_REFUSAL_FLAG: "nonsense"},
                        optimiser="bo", budget=12, seed=0)
    with pytest.raises(ValueError, match="unknown refusal mode"):
        api.run(cfg)


def test_an_unstated_flag_is_the_sentinel():
    cfg = api.RunConfig(problem_name="car rear wing", optimiser="bo",
                        budget=12, seed=0)
    assert api._bo_refusal(cfg) == "sentinel"


def test_the_flag_is_a_search_flag_not_a_physics_one():
    """Stating it must not change the PROBLEM — same dimension, same box,
    same constraint. Only where the search goes."""
    base = api.RunConfig(problem_name="car rear wing", optimiser="bo",
                         budget=12, seed=0)
    with_flag = api.RunConfig(problem_name="car rear wing",
                              flags={api.BO_REFUSAL_FLAG: "worst"},
                              optimiser="bo", budget=12, seed=0)
    a = api.run(base)
    c = api.run(with_flag)
    assert (a.dim, a.is_constrained) == (c.dim, c.is_constrained)
    assert a.param_labels == c.param_labels
