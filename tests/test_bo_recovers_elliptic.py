"""M5 CORRECTNESS GATE: BO must recover the known physics optimum.

Problem: rectangular wing (AR = 10), trimmed to CL = 0.5. Design variables
are 3 twist knots at eta = 0.5, 0.85, 1.0 RELATIVE to the root (root knot
fixed at 0; a constant twist offset is absorbed by trim, so the relative
parameterization removes that degeneracy). Objective: minimise CDi.

Known optimum: elliptic loading, CDi_min = CL^2/(pi*AR), e = 1. Scoping
(Nelder-Mead on the same parameterization) shows the 3-knot law reaches
CDi/CDi_min ~ 1.002, so demanding <= 1.02 tests the OPTIMISER, not the
parameterization. Untwisted rectangle sits at ~1.086 (e ~ 0.921) — the gap
BO must close. If this test fails the framework is wrong: stop and diagnose.
"""

import numpy as np
import pytest

from aerobo.llt import cosine_stations, solve_llt_trim
from aerobo.optimize.bo import run_bo

B = 10.0
S = 10.0
N = 60
CL_TARGET = 0.5
AR = B**2 / S
CDI_MIN = CL_TARGET**2 / (np.pi * AR)

KNOT_ETA = np.array([0.0, 0.5, 0.85, 1.0])
KNOT_BOUNDS = np.array([[-6.0, 1.0]] * 3)  # relative twist knots [deg]

_, _Y = cosine_stations(N, B)
_C = np.full(N, S / B)
_ETA = np.abs(2 * _Y / B)


def cdi_of_knots(k_rel_deg: np.ndarray) -> float:
    """CDi of the rectangular wing at trim CL, twist from relative knots."""
    knots = np.concatenate([[0.0], np.asarray(k_rel_deg, float)])
    twist = np.deg2rad(np.interp(_ETA, KNOT_ETA, knots))
    try:
        _, res = solve_llt_trim(B, _C, twist, CL_TARGET)
    except ValueError:
        return 1.0  # untrimmable: huge CDi penalty (minimisation convention)
    return res.CDi


def e_of_knots(k_rel_deg: np.ndarray) -> float:
    knots = np.concatenate([[0.0], np.asarray(k_rel_deg, float)])
    twist = np.deg2rad(np.interp(_ETA, KNOT_ETA, knots))
    _, res = solve_llt_trim(B, _C, twist, CL_TARGET)
    return res.e


def test_bo_recovers_elliptic_loading():
    res = run_bo(cdi_of_knots, KNOT_BOUNDS, n_init=10, n_iter=35,
                 seed=0, maximize=False)
    ratio = res.best_y / CDI_MIN
    e_best = e_of_knots(res.best_x)

    cdi_untwisted = cdi_of_knots(np.zeros(3))
    assert res.best_y < cdi_untwisted, "BO failed to beat the untwisted baseline"
    assert ratio <= 1.02, (
        f"CORRECTNESS GATE FAILED: BO best CDi/CDi_min = {ratio:.4f} > 1.02 "
        f"(e = {e_best:.4f}) — framework wrong, stop and diagnose"
    )
    assert e_best >= 0.98, f"span efficiency {e_best:.4f} < 0.98"


def test_gate_reference_values():
    """Pin the physics the gate relies on (analytic vs simulated, labelled)."""
    # analytic: Glauert minimum
    assert CDI_MIN == pytest.approx(0.5**2 / (np.pi * 10.0))
    # simulated: untwisted rectangle has ~8-9% excess induced drag, e ~ 0.92
    ratio_untw = cdi_of_knots(np.zeros(3)) / CDI_MIN
    assert 1.05 < ratio_untw < 1.12
    assert 0.90 < e_of_knots(np.zeros(3)) < 0.94
