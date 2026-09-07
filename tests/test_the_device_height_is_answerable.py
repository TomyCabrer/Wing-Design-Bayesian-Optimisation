"""The tip device's height is a number the user may STATE.

``winglet_h_frac`` rides its ceiling in four of five families that carry it
(``RESULTS_BOUND_RIDING.md`` §0), and the sweep says why twice over:

* nothing opposes it. ``sizing.sized_state`` takes no winglet argument, so
  ``W_wing_N`` and ``sigma_root_Pa`` come back BIT-IDENTICAL at
  h = 0.00 / 0.05 / 0.10 / 0.15 while the score climbs;
* and the ceiling is three times too low. Flown past it at the published box
  design the solver is smooth, finite and feasible with an INTERIOR optimum
  near h_frac 0.45 (L/D 39.233 against 37.768 at 0.15) — so 0.15 is where
  the nonplanar VLM was MEASURED, not a cap on the device being designed.

The chosen fix is the repo's own idiom rather than a modelled mass penalty:
the user owns the number. That needs the height row to TRAVEL, and it did
not. Stating a band wider than the calibration moved the sampler's box while
``prob.bounds`` stayed at the published one, so every draw outside came back
"bounds violation" and the run reported ``best_score = -100.0`` — the ninth
instance of the bug class ``api._arm_band_kwargs`` names, and a calibration
acting as a ban in the one direction where it is not one.

A calibration is a default, not a ban: the height clamps nothing and refuses
nothing above zero. The CANT still refuses outside the solver's own validity,
because that one really is a physical limit.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aerobo import api, geometry, objective                      # noqa: E402

_TAIL = "tail + winglet (span-capped) [designed tail + tip device]"


# ------------------------------------------------------- the validator

def test_the_published_band_is_the_default_bit_for_bit():
    """``None`` in, published band out — an untouched run is unchanged."""
    assert geometry.winglet_h_row(None) == tuple(
        float(v) for v in geometry.WINGLET_H_FRAC_BOUNDS)
    assert np.array_equal(geometry.bounds("winglet"),
                          geometry.bounds("winglet", h_bounds=None))


def test_a_stated_band_above_the_calibration_is_flown_not_refused():
    """The whole point: 0.15 is a measurement, not a cap."""
    row = geometry.winglet_h_row((0.0, 0.6))
    assert row == (0.0, 0.6)
    box = geometry.bounds("winglet", h_bounds=(0.0, 0.6))
    assert tuple(box[3]) == (0.0, 0.6)


@pytest.mark.parametrize("bad,why", [
    ((0.2, 0.2), "zero width"),
    ((-0.1, 0.3), "below zero"),
    ((0.5, 0.2), "hi < lo"),
    ((float("nan"), 1.0), "not finite"),
    (5, "must be a pair"),
])
def test_only_what_the_samplers_need_is_refused(bad, why):
    """A pair, finite, non-collapsed, non-negative, ordered. Nothing else —
    and in particular no ceiling."""
    with pytest.raises(ValueError, match=why):
        geometry.winglet_h_row(bad)


def test_a_collapsed_row_is_diagnosed_as_a_pin():
    """A bound of width zero is a pin written the wrong way, and the message
    has to say so: a collapsed box breaks the Sobol engine and degrades BO to
    random draws while still calling itself BO."""
    with pytest.raises(ValueError, match="RunConfig.pinned"):
        geometry.winglet_h_row((0.08, 0.08))


# ------------------------------------------------- the band reaches the box

def test_the_band_travels_to_the_problem():
    """The defect this closes: the sampler's box moved and the problem's did
    not, so every draw outside the calibration was a "bounds violation"."""
    spec = api.PROBLEM_SPECS[_TAIL]
    labels = list(spec.param_labels)
    i = labels.index("winglet_h_frac")
    j = labels.index("winglet_h_frac_t")

    base = np.asarray(spec.build({}, {}, None).bounds)
    assert tuple(base[i]) == (0.0, 0.15)

    wide = np.asarray(spec.build({}, {}, {
        "winglet_h_frac": [0.0, 0.6],
        "winglet_h_frac_t": [0.0, 0.5]}).bounds)
    assert tuple(wide[i]) == (0.0, 0.6)
    assert tuple(wide[j]) == (0.0, 0.5)
    # every OTHER row is untouched — the band is one row, not a new box
    others = [k for k in range(len(labels)) if k not in (i, j)]
    assert np.array_equal(base[others], wide[others])


def test_a_device_past_the_old_ceiling_actually_flies():
    """Not merely admitted by the box — evaluated, feasible, and built at
    the height that was asked for."""
    spec = api.PROBLEM_SPECS[_TAIL]
    labels = list(spec.param_labels)
    i = labels.index("winglet_h_frac")
    prob = spec.build({}, {}, {"winglet_h_frac": [0.0, 0.6]})
    box = np.asarray(prob.bounds)
    x = (box[:, 0] + box[:, 1]) / 2.0
    x[i] = 0.45
    out = prob.evaluate(x)
    assert out["feasible"], out.get("reason")
    assert np.isfinite(out["LoD"])
    # STATED IS NOT FLOWN unless it is: assert on the built geometry
    assert out["winglet"]["h_frac"] == pytest.approx(0.45)
    # ...and it is genuinely PAST the old cap, which is the whole point.
    # Not asserted as 0.45 * b_wing / 2: this family span-caps, so the flown
    # fraction is rescaled (h_frac * prob.b / wing.b) and the metre height is
    # set by the REFERENCE span. The cap is the thing under test, not the
    # rescale.
    old_ceiling_m = geometry.WINGLET_H_FRAC_BOUNDS[1] * out["b_wing"] / 2.0
    assert out["winglet"]["h_m"] > old_ceiling_m
    assert out["winglet"]["S_planform"] > 0.0


def test_the_unsized_winglet_family_carries_the_band_too():
    """``objective.Problem`` is the other half of the same row."""
    a = objective.Problem(mode="winglet").bounds
    b = objective.Problem(mode="winglet",
                          winglet_h_bounds=(0.0, 0.6)).bounds
    assert tuple(a[3]) == (0.0, 0.15)
    assert tuple(b[3]) == (0.0, 0.6)
    assert np.array_equal(np.delete(a, 3, axis=0), np.delete(b, 3, axis=0))


# ------------------------------------------------------ the asymmetry

def test_the_cant_still_refuses_outside_the_solvers_validity():
    """The height is a calibration and travels; the cant is a physical limit
    and does not. That asymmetry is the difference between the two, and it
    must survive this patch."""
    with pytest.raises(ValueError):
        geometry.bounds("winglet", cant_bounds=(5.0, 120.0))


def test_the_height_is_a_magnitude_and_the_cant_carries_the_sign():
    """A negative height would mirror the device twice — which way it points
    is the cant's sign, and the message must send the reader there."""
    with pytest.raises(ValueError, match="CANT"):
        geometry.winglet_h_row((-0.05, 0.2))
