"""The result read against its own design box — pure, no server, no physics.

A finished run used to report its winning design as a list of numbers, so
"AR = 12" said nothing about whether 12 was an aerodynamic answer or simply
the top of the box the search was given. :func:`gui.metrics.design_box`
places every variable inside the box the run ACTUALLY searched
(``RunResult.bounds``) and names the two things a reader needs from it:

* RIDING — the optimum sits within 2 % of an edge, so the box decided it
  (§13 boundary-riding law);
* NARROWED — that row is not the family's published bound, because this run
  overrode it (a design-box edit, or the section carried from stage 2).

The box must come from the RECORD: ``ProblemSpec.default_bounds`` is built
with no flags, so a flag-dependent row (winglet cant, chord law, planform
size) would otherwise be compared against bounds the run never had.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gui import metrics                                         # noqa: E402


def test_a_value_is_placed_between_its_own_low_and_high():
    rows = metrics.design_box(["AR", "taper"], [9.0, 0.45],
                              [[6.0, 12.0], [0.3, 0.9]])
    ar, taper = rows
    assert (ar["lo"], ar["hi"], ar["value"]) == (6.0, 12.0, 9.0)
    assert ar["frac"] == 0.5 and ar["span"] == 6.0
    assert not ar["riding"] and not ar["outside"] and not ar["narrowed"]
    assert taper["frac"] == 0.25


def test_an_optimum_on_an_edge_is_reported_as_riding_that_edge():
    rows = metrics.design_box(["lo_edge", "hi_edge", "inside"],
                              [6.05, 11.95, 9.0],
                              [[6.0, 12.0], [6.0, 12.0], [6.0, 12.0]])
    assert [r["riding"] for r in rows] == ["low", "high", ""]
    # the 2 % band is the tolerance, and it is the caller's to widen
    wide = metrics.design_box(["x"], [9.0], [[6.0, 12.0]], tol=0.6)
    assert wide[0]["riding"] == "low"


def test_a_value_outside_its_box_is_flagged_not_clamped_away():
    """A design vector that does not lie in the box it was scored against is
    a defect, not a rounding: it must be visible, and the value must still be
    the one the run recorded."""
    rows = metrics.design_box(["x"], [13.0], [[6.0, 12.0]])
    assert rows[0]["outside"] and rows[0]["value"] == 13.0
    assert rows[0]["frac"] > 1.0
    assert rows[0]["riding"] == "high"


def test_a_row_this_run_narrowed_says_so():
    rows = metrics.design_box(["AR", "tc"], [9.0, 0.12],
                              [[6.0, 12.0], [0.10, 0.14]],
                              overrides={"tc": [0.10, 0.14]})
    assert [r["narrowed"] for r in rows] == [False, True]


def test_a_record_with_no_box_still_reports_its_values():
    """Nothing is invented: no bounds means no bounds, not a zero box."""
    rows = metrics.design_box(["AR", "taper"], [9.0, 0.45])
    assert [r["value"] for r in rows] == [9.0, 0.45]
    assert all(r["lo"] is None and r["frac"] is None and not r["riding"]
               for r in rows)


def test_a_collapsed_or_unnamed_row_degrades_without_raising():
    rows = metrics.design_box([], [1.0, 2.0, None],
                              [[1.0, 1.0], "not a pair", [0.0, 1.0]])
    assert [r["label"] for r in rows] == ["x0", "x1", "x2"]
    assert rows[0]["span"] == 0.0 and rows[0]["frac"] is None
    assert rows[1]["lo"] is None
    assert rows[2]["value"] is None and rows[2]["frac"] is None


def test_the_box_of_a_real_run_is_the_flown_one_not_the_published_default():
    """The record's own bounds carry the run's overrides; the spec's default
    box does not. Comparing against the spec would call a narrowed row
    'riding' whenever the user moved a bound onto the optimum."""
    from aerobo import api

    name = "trim wing"
    spec = api.PROBLEM_SPECS[name]
    label, (lo, hi) = next(iter(spec.default_bounds.items()))
    mid = 0.5 * (float(lo) + float(hi))
    built = spec.build({}, {}, {label: [mid, float(hi)]})
    flown = [[float(a), float(b)] for a, b in built.bounds]
    x = [0.5 * (a + b) for a, b in flown]
    i = list(built.param_labels).index(label)
    x[i] = mid                                   # the low edge of the OVERRIDE

    rows = metrics.design_box(built.param_labels, x, flown,
                              overrides={label: [mid, float(hi)]})
    assert rows[i]["lo"] == mid and rows[i]["riding"] == "low"
    assert rows[i]["narrowed"]
    # against the family's published box the same design is mid-box, which is
    # exactly the answer that would have been wrong to show
    published = [list(spec.default_bounds[lbl])
                 for lbl in built.param_labels]
    assert metrics.design_box(built.param_labels, x,
                              published)[i]["riding"] == ""
