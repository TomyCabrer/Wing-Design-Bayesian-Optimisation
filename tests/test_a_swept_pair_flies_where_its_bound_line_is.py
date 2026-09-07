"""A swept tandem was drawn with its rear wing's control points square.

``vlm.VLM`` builds the second wing by panelising it with the SAME
constructor as the first and then offsetting it — ``A3``/``B3`` take
``sub_vlm.A3 + off``, so the bound vortex carries the sweep. The STATIONS
did not: they were rebuilt as ``np.full(n2, second.x)``, a constant. Control
points are ``st3 + d*xhat``, so on a swept pair the point where the tangency
condition is enforced sat up to (b/2)tan(Lambda) AHEAD of the very vortex it
was enforcing it on — 1.82 m on a 10 m pair at 20 deg.

The pair then trimmed into nonsense, and the failure was reported against
the SECTION: a stated ``wing_sweep_deg`` of 6 deg or more came back
"effective AoA outside polar validity", and 12 deg came back "trim alpha
-30.16 deg outside bracket". Both are true statements about a lattice that
was drawn wrong.

The same defect sat on the designed-tail path (``sub_t.A3[:, 0]`` against
``np.full(n_t, tail.x)``), where it is latent only because no registered
family hands that path a swept tail today.
"""
from __future__ import annotations

import numpy as np
import pytest

from aerobo import geometry
from aerobo.tandemvlm import TandemVLMProblem, evaluate_tandem_vlm
from aerobo.vlm import VLM, SecondWing


def _centre(**kw):
    prob = TandemVLMProblem(winglets=True, **kw)
    return prob, np.asarray(prob.bounds, dtype=float).mean(axis=1)


@pytest.mark.parametrize("sweep", [0.0, 3.0, 6.0, 12.0, 20.0, 30.0])
def test_the_whole_sweep_band_flies_instead_of_blaming_the_aerofoil(sweep):
    """The outcome a user meets: state a sweep, get an answer.

    Before the stations carried the sweep, 6/12/20/30 deg all refused, two
    of them naming the section polar for a geometry error.
    """
    prob, x = _centre(wing_sweep_deg=sweep)
    out = evaluate_tandem_vlm(x, prob)
    assert out["feasible"], (sweep, out["reason"])
    # ...and it is a SMALL effect, as a quarter-chord sweep on a pair
    # trimmed in lift should be — not the cliff the broken lattice showed
    prob0, x0 = _centre()
    flat = evaluate_tandem_vlm(x0, prob0)
    assert out["LoD"] == pytest.approx(flat["LoD"], rel=0.05), (
        sweep, out["LoD"], flat["LoD"])


def test_the_rear_wings_control_points_sit_behind_its_own_bound_line():
    """The geometry, asserted directly, so the fix cannot silently revert.

    A control point is ``a*c/(4 pi)`` aft of the bound vortex AT ITS OWN
    STATION. On a swept wing that station moves with |y|, so the difference
    must stay of order the panel chord — never of order the semi-span.
    """
    w = geometry.Wing(b=10.0, S=10.0, taper=0.6, sweep_deg=20.0)
    m = VLM(w, N=20, V=14.6, second=SecondWing(wing=w, x=5.0, z=1.0, N=20))
    sec = m.is_second
    assert sec.any()
    # the stations SPAN a range, they are not one number
    xs = m.st3[sec, 0]
    assert xs.max() - xs.min() > 1.0, xs.max() - xs.min()
    # ...and each sits within a chord of its own vortex, not a semi-span
    gap = np.abs(m.st3[sec, 0] - m.A3[sec, 0])
    assert gap.max() < float(m.c[sec].max()), (gap.max(), m.c[sec].max())


def test_an_unswept_pair_is_bit_for_bit_what_it_always_was():
    """``tan(0)`` is exactly 0.0 and ``0.0 + x`` is exactly ``x``, so every
    published run — all of which are unswept — cannot have moved."""
    w = geometry.Wing(b=10.0, S=10.0, taper=0.6, twist_tip_deg=-2.0)
    m = VLM(w, N=24, V=14.6, second=SecondWing(wing=w, x=5.0, z=1.0, N=24))
    sec = m.is_second
    assert np.array_equal(m.st3[sec, 0], np.full(int(sec.sum()), 5.0))
