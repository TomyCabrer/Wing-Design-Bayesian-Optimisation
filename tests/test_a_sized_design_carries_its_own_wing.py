"""A sized design must be trimmed to carry ITSELF, not just the payload.

Free the span or the area and the wing starts weighing itself: that closed
loop is the entire reason ``payload L/D`` exists as an objective. It was open.

``sizing.sized_state`` computes the right target — ``CL_target = W_total /
(q S)`` (``sizing.py``) — and the evaluator passes it down with
``dataclasses.replace``. But ``replace`` RE-RUNS ``__post_init__``, and
``objective.Problem.__post_init__`` re-derives ``CL_target`` from the mission
whenever one is set. The weight-coupled target was therefore overwritten with
``W_N / (q S)`` — the PAYLOAD weight — and the wing's own weight dropped out
of the force balance.

Measured before the fix, on ``trim wing + free planform`` with a stated
mission of 650 N: lift 650.0 N against W_total 2079.4 N, an aircraft trimmed
to carry **31 %** of itself, in **13 registered base families**. It flattered
the score 3.4x (14.88 -> 50.17) because CD was read at a third of the CL the
design actually needs.

It fired ONLY when a mission was stated — that is, only in real use, never in
a bare ``spec.build({}, ...)`` probe, which is why it survived this long.
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

from aerobo import api                                           # noqa: E402

_MISSION = {"W_N": 650.0, "V": 14.6}


def _at_centre(name: str, mission: dict):
    built = api.PROBLEM_SPECS[name].build(mission, {}, None)
    prob = built.problem
    box = np.asarray(built.bounds, dtype=float)
    out = built.evaluate((box[:, 0] + box[:, 1]) / 2.0)
    q = 0.5 * prob.rho * prob.V ** 2
    return out, q, prob


def _lift(out, q, prob) -> float:
    """Lift at the FLOWN flight point.

    With the flight modifier on, V and altitude are design variables, so the
    built problem's rho and V are not the ones that flew — a q taken from
    ``prob`` measures a different aircraft and reports a balance failure that
    is the test's, not the code's. The result carries what actually flew.
    """
    rho = out.get("rho")
    v = out.get("V")
    if rho is not None and v is not None:
        q = 0.5 * float(rho) * float(v) ** 2
    return float(out["CL"]) * q * float(out.get("S_m2", prob.S))


@pytest.mark.parametrize("name", [
    "trim wing + free planform",
    "winglet + free planform",
    "winglet_capped + free planform",
    "wing t/c + sweep + free planform",
    "winglet, blended (span-capped) + free planform",
    "winglet + airfoil (XFOIL) + free planform",
])
def test_lift_equals_the_total_weight_not_the_payload(name):
    """The force balance closes on W_total. Asserted as a closed form, at a
    real design, with a mission stated — the configuration the defect needed."""
    out, q, prob = _at_centre(name, _MISSION)
    assert out["feasible"], out.get("reason")
    lift, w_total, w_fixed = _lift(out, q, prob), out["W_total_N"], out["W_fixed_N"]
    assert w_total > w_fixed > 0.0          # the wing has weight at all
    assert lift == pytest.approx(w_total, rel=2e-2), (
        f"trimmed to carry {lift / w_total:.1%} of itself")
    # ...and it is NOT the payload-only balance the defect produced
    assert abs(lift - w_fixed) > 0.02 * w_fixed


def test_the_score_identity_still_holds_after_the_fix():
    """``payload L/D = L/D x W_fixed/W_total`` is what the objective MEANS;
    the fix must move the trim without breaking the definition."""
    out, _, _ = _at_centre("trim wing + free planform", _MISSION)
    assert out["score"] == pytest.approx(
        out["LoD"] * out["W_fixed_N"] / out["W_total_N"], rel=1e-12)


def test_a_stated_mission_and_no_mission_agree_about_the_balance():
    """The defect's signature was that stating a mission changed the physics.
    A run with no mission was always balanced; one with a mission was not."""
    for mission in ({}, _MISSION):
        out, q, prob = _at_centre("trim wing + free planform", mission)
        assert out["feasible"], out.get("reason")
        assert _lift(out, q, prob) == pytest.approx(out["W_total_N"], rel=2e-2)


def test_every_sized_family_that_flies_carries_its_own_weight():
    """The sweep, not a spot check: no registered sized family may trim to
    anything but its own total weight."""
    warnings.filterwarnings("ignore")
    seen, offenders, checked = set(), [], 0
    for name in api.PROBLEM_SPECS:
        if not any(k in name for k in ("free planform", "free span")):
            continue
        # dedupe on the FAMILY only, never on " + free": splitting there
        # collapsed every flight variant onto its sizeless base, so the
        # modifier whose ordering carries its own defect was never checked
        base = name.split(" [")[0].split(" + free planform")[0] \
                   .split(" + free span")[0]
        key = (base, "flight" in name, "chord law" in name)
        if key in seen:
            continue
        seen.add(key)
        try:
            out, q, prob = _at_centre(name, _MISSION)
            if not out.get("feasible") or "CL" not in out \
                    or "W_total_N" not in out:
                continue
            checked += 1
            lift, w_total = _lift(out, q, prob), out["W_total_N"]
            if abs(lift - w_total) > 0.02 * w_total:
                offenders.append((name, lift / w_total))
        except Exception:                    # noqa: BLE001 — a census
            continue
    assert checked >= 40, f"only {checked} families evaluated"
    assert not offenders, offenders[:8]


# ------------------------------------------------- the weight, at altitude

_HIGH = {"W_N": 2000.0, "V": 30.0, "altitude_m": 3000.0}


@pytest.mark.parametrize("name", [
    "trim wing + free planform",
    "winglet + free planform",
    "winglet_capped + free planform",
])
@pytest.mark.parametrize("alt", [0.0, 3000.0, 8000.0])
def test_a_stated_payload_weight_is_flown_at_every_altitude(name, alt):
    """``weight_for`` back-derives a weight at SEA-LEVEL density, which is
    right for a problem with no mission and wrong for one with a mission:
    ``CL_target`` was derived at the MISSION's altitude, so un-deriving it at
    sea level leaves ``W = W_N x rho_0/rho(h)``. Measured before the fix at
    W_N = 2000 N: 2694.91 N at 3000 m (x1.3475, the density ratio exactly)
    and 4665.18 N at 8000 m (x2.333) — and that weight was FLOWN, not just
    reported."""
    out, q, prob = _at_centre(name, {"W_N": 2000.0, "V": 30.0,
                                     "altitude_m": alt})
    if not out.get("feasible"):
        pytest.skip(f"refused at {alt} m: {out.get('reason')}")
    assert out["W_fixed_N"] == pytest.approx(2000.0, rel=1e-9)
    # ...and the force balance still closes on the TOTAL at that altitude
    assert _lift(out, q, prob) == pytest.approx(out["W_total_N"], rel=2e-2)


def test_the_density_ratio_is_not_hiding_in_the_weight():
    """The signature of the defect, stated as a closed form: the ratio of the
    weights at two altitudes must be 1, not rho_0/rho(h)."""
    from aerobo.mission import isa_density
    lo, _, _ = _at_centre("trim wing + free planform",
                          {"W_N": 2000.0, "V": 30.0, "altitude_m": 0.0})
    hi, _, _ = _at_centre("trim wing + free planform", _HIGH)
    assert hi["W_fixed_N"] / lo["W_fixed_N"] == pytest.approx(1.0, rel=1e-9)
    ratio = isa_density(0.0) / isa_density(3000.0)
    assert ratio > 1.3                      # the defect had a real size
    assert hi["W_fixed_N"] / lo["W_fixed_N"] != pytest.approx(ratio, rel=1e-3)


def test_a_problem_with_no_mission_still_derives_its_weight():
    """The control: ``weight_for``'s sea-level derivation is CORRECT where
    there is no mission — it is what makes switching the flight modifier on
    reproduce the fixed-state problem. The fix must not break that."""
    from aerobo.mission import design_weight_n, weight_for
    built = api.PROBLEM_SPECS["trim wing + free planform"].build({}, {}, None)
    prob = built.problem
    prob.mission = None
    prob.W_fixed_N = None
    assert design_weight_n(prob) == pytest.approx(
        weight_for(prob.CL_target, prob.V, prob.S), rel=1e-12)
