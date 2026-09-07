"""A design box RECOMMENDED for a mission, measured rather than asserted.

The family's published box is the box its solvers were calibrated over, and
its size rows already follow the aeroplane. What none of it follows is THIS
MISSION: the same 0.2-1 taper and the same +-0.5 chord coefficients are
searched whether the wing carries 4.9 N at 12 m/s or 653 N at 14.6.

``aerobo.recommend`` answers that by measuring — draw the box, evaluate, keep
what flew, bound the best-scoring quarter of it — so every band it returns is
this codebase's own physics at this mission. What these tests hold is the
three properties a recommendation has to have to be usable: it is
REPRODUCIBLE (same mission, same answer), it is MISSION-SPECIFIC (a different
mission gets a different box, or the feature does nothing), and it is SAFE (it
never widens a search and never returns a band of width zero, both of which
the api refuses downstream for good reasons).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

PROBLEM = "tail"
BIG = {"W_N": 652.8022140185119, "V": 14.6}
MODEL = {"W_N": 4.905, "V": 12.0}
MODEL_WING = {"b_m": 1.2, "S_m2": 0.18}

N = 32          # enough to bound a box of this dimension, cheap enough to run


def _built(mission, flags=None):
    from aerobo import api

    return api.PROBLEM_SPECS[PROBLEM].build(mission, flags or {}, None)


# ------------------------------------------------------ the pure geometry
def test_the_box_is_the_smallest_one_around_the_points():
    from aerobo import recommend

    box = np.array([[0.0, 10.0], [0.0, 10.0]])
    pts = [np.array([2.0, 4.0]), np.array([6.0, 5.0])]
    got = recommend.box_around(pts, box, pad=0.0, min_width=0.0)
    assert got[0].tolist() == [2.0, 6.0]
    assert got[1].tolist() == [4.0, 5.0]


def test_the_pad_widens_by_a_share_of_the_published_width():
    from aerobo import recommend

    box = np.array([[0.0, 10.0]])
    got = recommend.box_around([np.array([4.0]), np.array([6.0])], box,
                               pad=0.1, min_width=0.0)
    assert got[0].tolist() == [3.0, 7.0]        # +-0.1 * 10


def test_it_never_widens_the_published_row():
    """A recommendation may narrow a search and may never licence a wider
    one — the rule every derived number in this shell follows."""
    from aerobo import recommend

    box = np.array([[0.0, 1.0]])
    # points AT both edges, with a pad that would run outside
    got = recommend.box_around([np.array([0.0]), np.array([1.0])], box,
                               pad=0.5, min_width=0.0)
    assert got[0].tolist() == [0.0, 1.0]


def test_a_row_whose_draws_all_agree_is_not_a_band_of_width_zero():
    """The samplers raise on one and constrained BO quietly degrades to
    random draws while still calling itself BO, so a collapsed row is
    widened about its own centre rather than returned."""
    from aerobo import recommend

    box = np.array([[0.0, 10.0]])
    got = recommend.box_around([np.array([5.0]), np.array([5.0])], box,
                               pad=0.0, min_width=0.02)
    assert got[0][1] > got[0][0]
    assert got[0][1] - got[0][0] == pytest.approx(0.2)      # 0.02 * 10
    assert 0.5 * (got[0][0] + got[0][1]) == pytest.approx(5.0)


# ------------------------------------------------------ the best-scoring share
def test_the_best_share_keeps_the_top_quantile():
    from aerobo import recommend

    pts = [np.array([float(i)]) for i in range(10)]
    scores = [float(i) for i in range(10)]           # 9 is best
    keep, cut = recommend.best_share(pts, scores, 0.2)
    assert cut == 8.0
    assert [float(p[0]) for p in keep] == [8.0, 9.0]


def test_a_family_with_no_scalar_objective_is_not_ranked_on_one():
    """Falling back to the whole set is the honest answer; ranking on a
    number half the draws do not have is not."""
    from aerobo import recommend

    pts = [np.array([1.0]), np.array([2.0])]
    keep, cut = recommend.best_share(pts, [1.0, None], 0.5)
    assert cut is None and len(keep) == 2


# ---------------------------------------------------------- the measurement
def test_the_same_mission_gives_the_same_recommendation():
    """Reproducibility is a property of the problem, not of the session: a
    recommendation that moved under a user would make the run that took it
    unreproducible."""
    from aerobo import recommend

    a = recommend.recommend_box(_built(BIG), n=N, verify=False)
    b = recommend.recommend_box(_built(BIG), n=N, verify=False)
    assert a["rows"] == b["rows"] and a["rows"]


def test_a_different_mission_gives_a_different_box():
    """The whole point. Measured on the row a mission most obviously moves —
    the tail's area, which is a fraction of a wing that changed by 55x."""
    from aerobo import recommend

    big = recommend.recommend_box(_built(BIG), n=N, verify=False)
    small = recommend.recommend_box(_built(MODEL, MODEL_WING), n=N,
                                    verify=False)
    assert big["rows"] and small["rows"]
    assert big["rows"] != small["rows"]
    assert big["rows"]["S_t_m2"][1] > 10.0 * small["rows"]["S_t_m2"][1]


def test_every_recommended_row_is_inside_the_published_one():
    from aerobo import recommend

    built = _built(BIG)
    got = recommend.recommend_box(built, n=N, verify=False)
    assert got["rows"]
    for i, lab in enumerate(built.param_labels):
        lo, hi = got["rows"][lab]
        pub_lo, pub_hi = (float(v) for v in built.bounds[i])
        assert pub_lo - 1e-12 <= lo < hi <= pub_hi + 1e-12, lab


def test_it_recommends_over_the_designs_that_actually_flew():
    """Not over every draw. The distinction is the one ``box_refusal_probe``
    exists for: a design can solve and still miss a limit, and a box drawn
    around those is a box drawn around designs the search would reject.
    """
    from aerobo import recommend

    built = _built(BIG)
    drawn, kept, scores = recommend.admissible_draws(
        built, np.asarray(built.bounds, dtype=float), N, 0)
    assert drawn == N and 0 < len(kept) < drawn, "expected a mixed box"
    assert len(scores) == len(kept)
    for x in kept:
        flew, f = recommend._admissible(built, x)
        assert flew and f is not None


def test_a_box_that_contains_nothing_is_reported_and_not_recommended_over():
    """No narrowing helps a box with no answer in it, and a recommendation
    that invented one would send a user further from the fix."""
    from aerobo import recommend

    built = _built(BIG)

    class _Nothing:
        """The same problem with every draw refused."""
        def __getattr__(self, name):
            return getattr(built, name)

        def evaluate(self, x):
            return {"feasible": False, "reason": "bounds violation",
                    "score": -100.0}

    got = recommend.recommend_box(_Nothing(), n=8, verify=False)
    assert got["empty"] is True
    assert got["rows"] == {}
    assert got["n_admissible"] == 0


def test_the_verified_fraction_is_measured_and_not_assumed():
    """``frac_after`` is a re-probe of the recommendation, so it is free to
    come out WORSE than the box it narrowed — the best-scoring region is not
    the most-feasible one. A number that could only ever improve would be an
    assertion dressed as a measurement.
    """
    from aerobo import recommend

    built = _built(BIG)
    got = recommend.recommend_box(built, n=N, verify=True)
    assert got["frac_after"] is not None
    assert 0.0 <= got["frac_after"] <= 1.0
    # and it really is a measurement OF THE RECOMMENDATION: probing the same
    # box by hand gives the same answer
    rec = np.array([got["rows"][lab] for lab in built.param_labels],
                   dtype=float)
    frac, _best = recommend.probe_box(built, rec, got["n"], got["seed"])
    assert frac == got["frac_after"]
