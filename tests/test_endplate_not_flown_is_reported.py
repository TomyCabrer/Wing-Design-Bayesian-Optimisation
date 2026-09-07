"""A plate the search declined to build has to be SAID.

The endplate's height is a design variable whose lower bound is zero, so
"no plate" is an answer the optimiser is allowed to give — and it gave it: a
run of the designed-endplate family came back with ``endplate_h_m = 0.000``,
which the VLM drops (``vlm.MIN_WINGLET_FRAC``). Downstream that is invisible
in exactly the way that matters: the front view is not drawn at all and the
3-D view falls back to the bare wing, so four geometry views quietly lost a
surface the user had switched on two stages earlier.

The shell already had a sentence for this — and it could never fire here,
because it keys on the ``winglet`` block that only the winglet families
export. This file holds the endplate's own version of it.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aerobo import api                                          # noqa: E402

#: a mid-box design with NO plate: the endplate HEIGHT at its lower bound.
#: Built off the registered vector's own labels rather than a hand-counted
#: index — the family gained its reference AREA as a design row, and a
#: positional literal would have gone on flying whatever landed in slot 4.
_LABELS = list(api.PROBLEM_SPECS["car rear wing + endplates"].param_labels)
_VALUES = {"taper": 0.7, "twist_root_deg": 0.0, "twist_tip_deg": -2.0,
           "alpha_deg": 6.0, "endplate_h_m": 0.0, "ride_height_m": 0.55,
           "endplate_chord_ratio": 3.0, "endplate_tc": 0.12,
           "endplate_toe_deg": 0.0, "S_m2": 0.40, "b_m": 1.6}
X_NO_PLATE = np.array([_VALUES[k] for k in _LABELS], dtype=float)
X_PLATE = X_NO_PLATE.copy()
X_PLATE[_LABELS.index("endplate_h_m")] = 0.30


def _report(x):
    # the ride height sits well ABOVE the deck here, so a wing with no plate
    # is REFUSED by the reach margin — the plates carry the wing on every
    # layout now, and a wing whose plates do not get down to the car is not
    # attached to it. The refusal is the correct answer; what this file is
    # about is that the design is still REPORTED honestly either way, because
    # the geometry views are drawn from a report whether or not it was
    # feasible.
    cfg = api.RunConfig(problem_name="car rear wing + endplates",
                        flags={"mount": "tips"})
    return api.design_report(cfg, x)


def test_a_zero_height_plate_is_not_flown_and_leaves_no_geometry():
    rep = _report(X_NO_PLATE)
    geom = rep["geometry"]
    assert not geom.get("is_winglet")          # nothing to draw, honestly
    assert rep["breakdown"]["endplate_h_m"] == pytest.approx(0.0)


def test_a_plate_that_is_flown_still_exports_its_panels():
    rep = _report(X_PLATE)
    assert sum(rep["geometry"]["is_winglet"]) > 0
    assert rep["breakdown"]["endplate_h_m"] == pytest.approx(0.30)


def test_the_front_view_declines_a_design_with_no_device():
    from gui import nice_app as v1
    assert v1.fig_frontview(_report(X_NO_PLATE)["geometry"]) is None
    assert v1.fig_frontview(_report(X_PLATE)["geometry"]) is not None


def test_the_results_stage_says_so_instead_of_drawing_nothing():
    from gui.v3.stages import results
    src = inspect.getsource(results)
    assert "endplate_h_m" in src
    assert "carries NO endplate" in src
    # ...and it is NOT gated on the winglet block, which this family never
    # exports — that gate is why the existing sentence could not fire
    hint = src[src.index("carries NO endplate") - 900:
               src.index("carries NO endplate")]
    assert 'bd_geo.get("endplate_h_m")' in hint
    assert 'not geom.get("is_winglet")' in hint
