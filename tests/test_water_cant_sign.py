"""Why the water family is scored span-free — the two measured facts.

Session 47, `PREREG_SESSION47_REGISTRY.md` S47-2. The last open refusal in the §17.8
audit (span-capped accounting under water, 57 configurations) was recorded as
blocked on a geometric question — "what does projected span mean for a
SURFACE-PIERCING foil". These tests pin what was actually measured:

1. **It never pierces.** Over the whole design box the tip device stays
   submerged, so the premise is false and the lateral projection is as well
   defined as it is in air.
2. **A lateral cap is blind to the trade.** The lateral projection is EVEN in
   the cant angle; the physics is ODD in it, and worth 4.4 % of L/D at the
   tall-and-shallow corner. That — not an ill-posed geometry — is why a span
   cap is the wrong constraint here.

Both are pinned because both are load-bearing for a decision recorded in the
report, and because fact 2 is invisible at the box centre (0.28 %), which is
where a careless probe would look.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aerobo import geometry, hydrofoil                         # noqa: E402

CANTS = (15.0, 30.0, 45.0, 60.0, 75.0, 90.0)


def _lod(prob, cant_deg, h_frac, depth_m):
    labels = list(prob.param_labels)
    x = np.array([0.5 * (lo + hi) for lo, hi in prob.bounds])
    x[labels.index("winglet_cant_deg")] = cant_deg
    x[labels.index("winglet_h_frac")] = h_frac
    x[labels.index("depth_m")] = depth_m
    out = hydrofoil.evaluate_hydrofoil_winglet(x, prob)
    assert out["feasible"], out.get("reason")
    return float(out["LoD"])


# ------------------------------------------------- 1. it does not pierce

def test_the_tip_device_never_breaks_the_free_surface():
    """The worst case in the box is still 0.06 m under.

    Asserted from the BOUNDS rather than from a hard-coded 0.06, so widening
    the height band or shallowing the depth band fails this test instead of
    silently invalidating the report's reasoning.
    """
    prob = hydrofoil.HydrofoilWingletProblem()
    semi = float(prob.b) / 2.0
    h_max_arc = float(prob.WINGLET_H_FRAC_BOUNDS[1]) * semi
    depth_min = float(prob.DEPTH_BOUNDS[0])
    clearance = depth_min - h_max_arc          # cant +90 = straight up
    assert clearance > 0.0, (
        "the tip device now pierces the free surface somewhere in its own "
        "box — report §17.8's reasoning assumes it does not")
    assert clearance == pytest.approx(0.060, abs=5e-4)


# --------------------------------- 2. the cap is even, the physics is odd

@pytest.mark.parametrize("cant", CANTS)
def test_the_lateral_projection_is_EVEN_in_the_cant_angle(cant):
    """Which is precisely why a lateral cap cannot see the sign.

    Asked of the PACKAGE's own reach (:func:`geometry.winglet_projection`)
    and of the cap that would be written against it — the CAPPED_COSINE_MODES
    branch of :func:`geometry.wing_from_x`. The old form recomputed
    ``cos(+c) == cos(-c)`` from ``math`` inside the test body, which is true
    of IEEE floats whatever the package does: rewriting the cap as a
    sign-aware reach, or capping the water family on the developed line as
    the blended modes already are, could never have turned it red.
    """
    prob = hydrofoil.HydrofoilWingletProblem()
    h_frac = float(prob.WINGLET_H_FRAC_BOUNDS[1])
    b = float(prob.b)
    h_arc = h_frac * b / 2.0

    up = geometry.winglet_projection(h_arc, +cant)
    dn = geometry.winglet_projection(h_arc, -cant)
    assert up == pytest.approx(dn, rel=0, abs=0.0)
    # ...while the vertical extent is equal and OPPOSITE
    assert geometry.winglet_tip_height(h_arc, +cant) == pytest.approx(
        -geometry.winglet_tip_height(h_arc, -cant), rel=1e-12)

    def capped_b(c):
        """Wing panel a span cap leaves behind at cant ``c``."""
        return float(geometry.wing_from_x(
            np.array([0.5, 0.0, -2.0, h_frac, float(c)]),
            b=b, S=float(prob.S), mode="winglet_capped").b)

    #: the cap SPENDS the projection — one per side — and so is blind to the
    #: sign for exactly the reason the report records
    assert capped_b(+cant) == capped_b(-cant)
    assert b - capped_b(+cant) == pytest.approx(2.0 * up, rel=1e-12)
    if abs(cant) < 90.0:
        # ...and it is a cap that BITES, so a cap accidentally switched off
        # (trivially even) fails here instead of passing
        assert capped_b(+cant) < b
        assert up > 0.0


def test_canting_DOWN_beats_canting_up_at_every_angle():
    """One-signed, so the asymmetry is structure and not scatter."""
    prob = hydrofoil.HydrofoilWingletProblem()
    h_max = float(prob.WINGLET_H_FRAC_BOUNDS[1])
    d_min = float(prob.DEPTH_BOUNDS[0])
    for c in CANTS:
        assert _lod(prob, -c, h_max, d_min) > _lod(prob, +c, h_max, d_min), c


def test_the_sign_asymmetry_is_LARGE_at_the_corner_and_invisible_at_the_centre():
    """The measurement the decision rests on — and the blind probe it caught.

    A centre-only probe returns 0.28 % and would have said "close it by
    analogy with air". The corner says 4.4 %. Both are asserted, because the
    *contrast* is the finding.
    """
    prob = hydrofoil.HydrofoilWingletProblem()
    labels = list(prob.param_labels)
    centre = np.array([0.5 * (lo + hi) for lo, hi in prob.bounds])
    h_mid = float(centre[labels.index("winglet_h_frac")])
    d_mid = float(centre[labels.index("depth_m")])
    h_max = float(prob.WINGLET_H_FRAC_BOUNDS[1])
    d_min = float(prob.DEPTH_BOUNDS[0])

    def worst_gap(h, d):
        return max(abs(_lod(prob, +c, h, d) - _lod(prob, -c, h, d))
                   / _lod(prob, -c, h, d) for c in CANTS)

    centre_gap = worst_gap(h_mid, d_mid)
    corner_gap = worst_gap(h_max, d_min)

    assert centre_gap == pytest.approx(0.00282, abs=2e-4)
    assert corner_gap == pytest.approx(0.04426, abs=2e-4)
    # the contrast: the centre is BELOW the 1 % line the pre-registration
    # used as its kill criterion, and the corner is well above it
    assert centre_gap < 0.01 < corner_gap
    assert corner_gap / centre_gap > 10.0
