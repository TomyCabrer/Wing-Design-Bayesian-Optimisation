"""Which design-box ROW is the limiting factor — and when that cannot be said.

A gate name ("aspect ratio outside the band") tells a user what the physics
objected to. It does not tell them which control to move, which is the
question they have in front of an editable design box. ``box_refusal_probe``
now answers it from draws it already paid for: of the designs that SURVIVED,
how much of each row did they never come from.

The statistic has one trap and it is the whole reason for this file. The
observed span of k draws understates the interval they came from — E[span] =
(k-1)/(k+1) — so on a box that is 90 % refused, five survivors make every row
look two thirds dead. A fixed threshold would therefore name a row on every
difficult box, which is exactly the class of advice that sends a user to
narrow a row that was never the problem.
"""
import numpy as np
import pytest

from aerobo import api


def _draws(X, kept):
    return [(np.asarray(x, dtype=float), not bool(k)) for x, k in zip(X, kept)]


BOX = np.array([[0.0, 10.0], [0.0, 10.0], [0.0, 10.0]])
LABELS = ["free", "dead_top", "wide"]


def _synthetic(n_kept: int, seed: int = 0):
    """``n_kept`` survivors: uniform in rows 0 and 2, capped at 3 in row 1."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 10.0, size=(n_kept, 3))
    X[:, 1] = rng.uniform(0.0, 3.0, size=n_kept)
    return _draws(X, [True] * n_kept) + _draws(
        rng.uniform(0.0, 10.0, size=(200, 3)), [False] * 200)


def test_the_row_that_is_dead_is_the_row_that_is_named():
    """The signal: one row whose survivors only ever came from its bottom."""
    rows = api._refusal_by_row(_synthetic(40), BOX, LABELS)
    assert [r["label"] for r in rows] == ["dead_top"]
    r = rows[0]
    assert r["end"] == "top"
    assert r["dead_frac"] > 0.6
    assert r["n_kept"] == 40


def test_a_healthy_box_names_nothing_however_few_survived():
    """The trap, held in both directions. Every row here is uniform over its
    whole band, so a correct statistic reports NOTHING — at 5 survivors, at
    12, and at 40. A fixed span threshold reports two of the three at 5."""
    for k in (5, 8, 12, 40):
        X = np.random.default_rng(k).uniform(0.0, 10.0, size=(k, 3))
        draws = _draws(X, [True] * k) + _draws(
            np.random.default_rng(99).uniform(0.0, 10.0, size=(200, 3)),
            [False] * 200)
        assert api._refusal_by_row(draws, BOX, LABELS) == [], (
            f"named a row off {k} survivors of a box with no dead span")


def test_the_span_p_value_is_the_beta_cdf_it_claims_to_be():
    """Checked against simulation, not against itself: a wrong exponent gives
    a fluent, monotone, entirely mis-calibrated number."""
    rng = np.random.default_rng(3)
    for k in (3, 6, 20):
        draws = rng.uniform(size=(20000, k))
        span = draws.max(axis=1) - draws.min(axis=1)
        for r in (0.3, 0.6, 0.9):
            empirical = float((span <= r).mean())
            assert api._span_p_value(r, k) == pytest.approx(empirical,
                                                            abs=0.015), (k, r)
    assert api._span_p_value(1.0, 9) == pytest.approx(1.0)
    assert api._span_p_value(0.0, 9) == 0.0


def test_too_few_survivors_says_nothing_at_all():
    """Below the floor the answer is silence, not a guess."""
    assert api._refusal_by_row(_synthetic(3), BOX, LABELS) == []


def test_the_reported_box_names_its_span_row():
    """End to end on the family that provoked all of this: the box the user
    searched had a span row of 6-40 m and nothing above 17.3 m ever survived.

    Cheap because a refusal costs no physics, and it is the one integration
    that proves the analysis rides on the probe's own draws.
    """
    cfg = api.RunConfig(
        problem_name=("tail + winglet (span-capped) [designed tail + tip "
                      "device] + free planform + free chord law"),
        flags={"winglet_chord_follows": True, "tail_winglet_dir": "follow",
               "tail_winglet_blend_frac": 0.5, "winglet_blend_frac": 0.5,
               "blend_shape": "spiral",
               "wing_loading_limit_pa": 75.202848},
        optimiser="bo", budget=72, seed=0)
    payload = api.box_refusal_probe(cfg, n=192)
    assert payload["n_refused"] > payload["n"] // 2
    named = [r for r in payload["rows"] if r["label"] == "b_m"]
    assert named, [r["label"] for r in payload["rows"]]
    row = named[0]
    assert row["end"] == "top"
    assert row["kept"][1] < 0.6 * row["bounds"][1]
