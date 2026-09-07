"""Gates for the span-capped winglet mode (phase 3).

The cap: total projected span fixed at b — the wing shrinks (constant S)
to pay for the winglet's horizontal projection, so a raked tip is no
longer a free span extension.
"""

import numpy as np
import pytest

from aerobo import geometry
from aerobo.objective import Problem, evaluate
from aerobo.vlm import VLM


def test_bounds_match_winglet_mode():
    assert np.array_equal(geometry.bounds("winglet_capped"),
                          geometry.bounds("winglet"))


def test_vertical_winglet_reproduces_free_span_mode_exactly():
    """At cant = 90 deg the horizontal projection vanishes: the capped mode
    must reproduce the free-span winglet mode BIT-FOR-BIT."""
    x = np.array([0.4, -1.0, -2.0, 0.12, 90.0])
    out_free = evaluate(x, Problem(mode="winglet"))
    out_cap = evaluate(x, Problem(mode="winglet_capped"))
    assert out_free["feasible"] and out_cap["feasible"]
    assert out_cap["score"] == out_free["score"]
    assert out_cap["wing"].b == out_free["wing"].b


def test_projected_span_capped_exactly():
    """At any cant the outermost panel edge sits at exactly +-b/2."""
    b = 10.0
    for cant in (60.0, 75.0, 89.0):
        x = np.array([0.4, -1.0, -2.0, 0.15, cant])
        prob = Problem(mode="winglet_capped")
        wing = geometry.wing_from_x(x, b=b, S=prob.S, mode="winglet_capped")
        h_frac = float(x[3]) * b / wing.b
        model = VLM(wing, N=prob.N_vlm, winglet_h_frac=h_frac,
                    winglet_cant_deg=cant, n_winglet=prob.n_winglet)
        y_max = float(np.abs(model.y).max())
        # stations sit inside the edges; reconstruct the outermost EDGE
        y_edge = wing.b / 2.0 + h_frac * wing.b / 2.0 * np.cos(np.deg2rad(cant))
        assert y_edge == pytest.approx(b / 2.0, rel=1e-12)
        assert y_max <= b / 2.0 + 1e-9


def test_arc_length_preserved_under_cap():
    """The winglet arc length stays x[3] * (b/2) of the ORIGINAL span."""
    # The old form computed `h_frac_eff = x3 * b / wing.b` in the TEST and
    # then asserted `h_frac_eff * wing.b / 2 == x3 * b / 2` — an algebraic
    # identity in wing.b, with the only production value cancelling out. It
    # could not see the solver hand the VLM the RAW x[3] instead of the
    # rescaled one (objective.py's `h_frac *= b_use / wing.b`), which flies a
    # 0.694 m winglet where the cap contract promises 0.75 m. So ask the RUN
    # what arc it flew, not the test's own arithmetic.
    b, x3, cant = 10.0, 0.15, 60.0
    # the twist pair is the one the sibling capped tests fly (the shrink law
    # reads x[3] and x[4] only, so wing.b is the same either way) — the point
    # here is a design that gets all the way THROUGH the solver
    x = np.array([0.4, -1.0, -2.0, x3, cant])
    wing = geometry.wing_from_x(x, b=b, S=10.0, mode="winglet_capped")
    assert wing.b == pytest.approx(b * (1 - x3 * np.cos(np.deg2rad(cant))))

    out = evaluate(x, Problem(b=b, S=10.0, mode="winglet_capped"))
    assert out["feasible"], f"the capped design must fly: {out['reason']}"
    assert out["wing"].b == pytest.approx(wing.b)
    # h_m is h_frac * wing.b / 2 with the h_frac the VLM was BUILT on, so it
    # reads the rescale straight off the run
    assert out["winglet"]["h_m"] == pytest.approx(x3 * b / 2.0)
    # ...and that arc is exactly what buys the cap back: shrunk semi-span
    # plus the device's horizontal projection returns the nominal b/2
    assert out["winglet"]["projected_semispan_m"] == pytest.approx(b / 2.0)


def test_cap_costs_performance_at_raked_cant():
    """A raked winglet under the cap must be WORSE than the same design with
    free span (the free-span variant genuinely extends the span), and the
    penalty must vanish at cant = 90."""
    x_raked = np.array([0.4, -1.0, -2.0, 0.15, 60.0])
    free = evaluate(x_raked, Problem(mode="winglet"))["score"]
    capped = evaluate(x_raked, Problem(mode="winglet_capped"))["score"]
    assert capped < free - 0.1        # measurable span-cap cost
