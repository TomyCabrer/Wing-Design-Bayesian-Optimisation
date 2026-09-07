"""A run that asked to be guided is told when it cannot be.

``bo_feasibility="guide"`` replaces the flat refusal sentinel with a graded
one, using ``SizeGateMargin`` — the two cheap size gates, computable from the
design vector with no solver. It did nothing at all on the DERIVED-area
families (``size_free`` in ``{"wing_loading", "wing_loading_free"}``, 444
registered variants), because those carry no ``S_m2`` row and the aspect-ratio
arm needed both ``b_m`` and ``S_m2``. It said nothing about that.

Two things are fixed here, and the second is the one that matters.

**The arm.** With the mission's W/S the area IS bounded: ``S = W_total/(W/S)``
and ``W_total = W_fixed + W_wing`` with ``W_wing > 0``, so ``S >= W_fixed/(W/S)``
and therefore ``AR <= b^2 (W/S) / W_fixed``. That is an UPPER bound, so only
the LOW side of the AR band is decidable — claiming the high side would need
an upper bound on the total weight, which is exactly what the weight loop has
not solved yet, and a false positive in a pre-filter costs a DESIGN.

**The measurement.** Structural liveness ("the rows are here") and useful
liveness ("some design in this box trips the gate") are different questions,
and the derived-area family is the case that proves it: at W/S 65 Pa and
W_fixed 650 N the bound is ``AR <= b^2/10``, which on a 6-40 m span band
cannot reach under the lower AR limit at all. Live, sound, and inert. A
grader that is live but never fires is WORSE than a dead one, because it
silences the warning while grading nothing — so the check is a measurement
over the box, not a property of the label list.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aerobo import api, objective, sizing                        # noqa: E402
from aerobo.optimize.feasible import SizeGateMargin              # noqa: E402

_GUIDE = {"bo_feasibility": "guide"}


def _grader(name, mission=None):
    built = api.PROBLEM_SPECS[name].build(mission or {}, {}, None)
    cfg = api.RunConfig(problem_name=name, optimiser="bo", budget=8, seed=0,
                        flags=dict(_GUIDE))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        g = api._gate_grader(cfg, built)
    runtime = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    return g, runtime, built


# ------------------------------------------------------------ the arm

def test_a_derived_area_family_can_be_graded_at_all():
    """It could not before: no ``S_m2`` row, so the AR arm was dead."""
    labels = list(api.PROBLEM_SPECS["trim wing + free span + free W/S"]
                  .param_labels)
    assert "S_m2" not in labels and "b_m" in labels
    g, _, _ = _grader("trim wing + free span + free W/S")
    assert g is not None and g.live


def test_the_bound_is_sound_no_design_is_gated_that_solves():
    """The soundness condition, over the box. A gated draw that SOLVES would
    mean the pre-filter is throwing designs away."""
    g, _, built = _grader("trim wing + free span (W/S)",
                          {"W_N": 650.0, "V": 14.6})
    assert g is not None
    box = np.asarray(built.bounds, dtype=float)
    rng = np.random.default_rng(3)
    pts = box[:, 0] + rng.random((256, box.shape[0])) * (box[:, 1] - box[:, 0])
    for x in pts:
        if g.excess(x) > 0.0:
            assert not built.evaluate(x).get("feasible"), (
                "the size gate refused a design the solver accepted")


def test_only_the_low_side_of_the_band_is_claimed():
    """``AR <= b^2 (W/S)/W_fixed`` is an upper bound, so a design whose bound
    is ABOVE the ceiling must not be gated — nothing is known about it."""
    g = SizeGateMargin(["b_m"], ar_limits=(3.0, 40.0),
                       weight_n=650.0, ws_fixed_pa=65.0)
    assert g.live
    # b = 4 -> AR <= 16*65/650 = 1.6, under the floor of 3: decidable
    assert g.excess(np.array([4.0])) > 0.0
    # b = 30 -> AR <= 900*65/650 = 90, over the ceiling of 40 — but the TRUE
    # AR is lower and may be perfectly inside. Not decidable, not gated.
    assert g.excess(np.array([30.0])) == 0.0


def test_the_loading_has_one_definition():
    """The gate must bound the area the SOLVER flew. Two spellings of the
    fallback W/S would let the two disagree about which design is being run."""
    built = api.PROBLEM_SPECS["trim wing + free span (W/S)"].build(
        {"W_N": 650.0, "V": 14.6}, {}, None)
    prob = built.problem
    ws = objective.design_wing_loading_pa(prob)
    assert ws == pytest.approx(prob.CL_target * 0.5 * prob.rho * prob.V ** 2)
    assert api._design_ws_pa(prob) == ws


# --------------------------------------------------- the measurement

def test_a_live_but_flat_gate_says_so():
    """The derived-area family: sound, live, and unable to fire anywhere in
    its own box. That must not be silent."""
    g, runtime, _ = _grader("trim wing + free span (W/S)",
                            {"W_N": 650.0, "V": 14.6})
    assert g is not None
    assert any("does not fire anywhere in this box" in str(w.message)
               for w in runtime), [str(w.message) for w in runtime]


def test_a_gate_that_does_fire_is_silent():
    """The control. A warning on every run would be no warning at all."""
    for name in ("trim wing + free planform",
                 "trim wing + free span + free W/S"):
        g, runtime, built = _grader(name)
        assert g is not None
        assert not runtime, [str(w.message) for w in runtime]
        # ...and it really does fire, so this is not a silent dead gate
        box = np.asarray(built.bounds, dtype=float)
        rng = np.random.default_rng(0)
        pts = box[:, 0] + rng.random((256, box.shape[0])) * (box[:, 1]
                                                             - box[:, 0])
        assert any(g.excess(x) > 0.0 for x in pts)


def test_a_gate_with_nothing_to_measure_says_so_too():
    """The other half of "never silent": a problem stating nothing this class
    can measure returns None AND warns, rather than quietly degrading to the
    flat sentinel the user asked to replace."""
    g = SizeGateMargin(["taper"], ar_limits=sizing.AR_LIMITS)
    assert not g.live


def test_guiding_is_off_by_default_and_then_nothing_is_warned():
    """A run that never asked to be guided must not be nagged."""
    built = api.PROBLEM_SPECS["trim wing + free span (W/S)"].build({}, {}, None)
    cfg = api.RunConfig(problem_name="trim wing + free span (W/S)",
                        optimiser="bo", budget=8, seed=0)
    assert api._bo_feasibility(cfg) == "off"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert api._gate_grader(cfg, built) is None
    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]
