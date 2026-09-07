"""A spec must not lie about its own vector, or about its own gates.

Two defects of one shape — the registry stating something the physics does
not do.

**The vector.** ``_variant_labels`` stacks ``[family][size][flight][chord]``,
but the flight modifier on ``trim wing`` is implemented as
``objective.py``'s ``"mission"`` MODE, and ``geometry.bounds`` puts
``MISSION_V_BOUNDS`` / ``MISSION_ALT_BOUNDS_M`` inside the FAMILY block, with
the size rows vstacked after. The real order is therefore
``[family][V, alt][size][chord]`` and the labels were transposed against
their own boxes: ``b_m`` carried the (10, 25) V band while ``V_ms`` carried
the (6, 40) span band. Not cosmetic — the same labels drive
``bounds_overrides`` and ``pinned``, so ``pinned={"V_ms": 12.0}`` flew a SPAN
of 12 m. 6 of 772 buildable specs were affected.

**The gate** is a second defect of the same shape, found and measured but
NOT fixed here: ``evaluate_aircraft`` calls neither ``check_ar`` nor
``check_ar_limit``, so the aircraft family returns ``feasible=True`` at
AR 136-197 — far outside the (3, 40) band the package declares it models,
~45 % of its feasible draws — while ``api.size_box_conflicts`` reports that
same box empty. The fix is a design question rather than a patch, for the
reason recorded below the vector tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aerobo import api                                           # noqa: E402

_FLIGHT = "trim wing + free planform + free flight state"


# ------------------------------------------------------------- the vector

def test_a_row_set_by_name_is_the_row_that_flies():
    """The whole contract in one assertion, on the spec that broke it."""
    spec = api.PROBLEM_SPECS[_FLIGHT]
    labels = list(spec.param_labels)
    built = spec.build({"W_N": 2000.0, "V": 20.0, "altitude_m": 1000.0},
                       {}, None)
    box = np.asarray(built.bounds, dtype=float)
    want = {"taper": 0.6, "twist_root_deg": 0.0, "twist_tip_deg": -2.0,
            "b_m": 22.0, "S_m2": 15.0, "V_ms": 20.0, "altitude_m": 1500.0}
    x = np.array([np.clip(want[k], *box[i]) for i, k in enumerate(labels)])
    out = built.evaluate(x)
    assert out["feasible"], out.get("reason")
    for row, flown in (("b_m", "b_m"), ("S_m2", "S_m2"),
                       ("V_ms", "V"), ("altitude_m", "altitude_m")):
        assert out[flown] == pytest.approx(want[row], rel=1e-9), row


def test_no_registered_spec_transposes_its_flight_and_size_rows():
    """The sweep. A V band never spans the span's 6-40 m, and a span band is
    never the V block's (10, 25) m/s — so a swap is detectable from the boxes
    alone, without evaluating 772 problems."""
    offenders = []
    for name, spec in api.PROBLEM_SPECS.items():
        labels = list(spec.param_labels)
        if not {"V_ms", "altitude_m", "b_m", "S_m2"} <= set(labels):
            continue
        try:
            box = np.asarray(spec.build({}, {}, None).bounds, dtype=float)
        except Exception:                       # noqa: BLE001 — a census
            continue
        v_lo, v_hi = box[labels.index("V_ms")]
        alt_lo, alt_hi = box[labels.index("altitude_m")]
        b_lo, _ = box[labels.index("b_m")]
        # an ALTITUDE band reaches thousands of metres; a span/area one does
        # not, and a span never starts below a metre
        if alt_hi < 100.0 or b_lo < 1.0 or v_hi <= v_lo:
            offenders.append((name, (v_lo, v_hi), (alt_lo, alt_hi)))
    assert not offenders, offenders[:6]


def test_the_flight_rows_come_before_the_size_rows_on_a_mission_family():
    """The stacking itself, stated once. ``trim wing``'s flight modifier IS
    the mission mode, so its rows sit in the family block."""
    labels = list(api.PROBLEM_SPECS[_FLIGHT].param_labels)
    assert labels.index("V_ms") < labels.index("b_m")
    assert labels.index("altitude_m") < labels.index("S_m2")
    # ...and a family whose flight block really is trailing keeps its order
    other = "winglet + free planform + free flight state"
    if other in api.PROBLEM_SPECS:
        lo = list(api.PROBLEM_SPECS[other].param_labels)
        assert lo.index("b_m") < lo.index("V_ms")


# -------------------------------------------------------------- the gate
#
# The aircraft family's missing aspect-ratio gate is REAL and measured
# (feasible=True at AR 136-197, ~45 % of its feasible draws, while
# api.size_box_conflicts reports that same box empty) — but the fix is NOT
# landed and these tests are therefore not written yet.
#
# Adding `check_ar` to `evaluate_aircraft` refuses the LOW end too: the
# family's own documented span sweep starts at b = 6.0 m with S = 14 m^2,
# i.e. AR 2.57, under the band's 3.0 floor. That broke four physics tests in
# `tests/test_aircraft.py` which pin a MEASURED result (the weight/span trap,
# `aircraft.py:165-176`). Re-ranging those sweeps to match a gate added in
# the same change would be fixing the test to match the code.
#
# So the open question is a design one, for a human: does the (3, 40)
# validity band apply to this family at all, and if so does its documented
# sweep need re-ranging? Until that is answered the contradiction stands and
# is recorded in RESULTS_SIZED_TRIM_DEFECT.md.
