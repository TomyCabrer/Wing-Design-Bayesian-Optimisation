"""The empennage weighs something, and each layout weighs its own.

Before this the tail's mass sat inside ``W_fixed`` as a constant — the
limitation ``wingtail.WingTailProblem.size_free`` documented in as many
words: "sizing the tail moves its drag and its trim, not its weight". So a
V-tail, whose whole purpose is to delete a surface, weighed exactly what a
conventional tail weighed, and a T-tail — whose fin carries the tailplane —
weighed the same again.

Raymer's GA horizontal- and vertical-tail equations now join the same weight
fixed point the wing is closed in (both exponents on ``N_z W_dg`` are below
1, so it is still a contraction), and the LAYOUT decides which surfaces are
in it:

* conventional / canard — a tailplane and a fin;
* T-tail — the same two, and Raymer's ``1 + 0.2 H_t/H_v`` charges 20 % on the
  fin for carrying the tailplane;
* V-tail — the two panels at their TRUE area, and no fin at all.

Nothing without an empennage moves: an empty description weighs 0.0 N and
the loop is the one it always ran.
"""
from __future__ import annotations

import numpy as np
import pytest

from aerobo import sizing, tail as tailmod, weights

N_ULT = weights.N_ULT_DEFAULT
W_DG = 9500.0

HT = weights.EmpennageSurface(S=1.5, AR=4.0, taper=0.6, tc=0.10)
FIN = weights.EmpennageSurface(S=0.9, AR=1.5, taper=1.0, tc=0.10,
                               vertical=True)
FIN_T = weights.EmpennageSurface(S=0.9, AR=1.5, taper=1.0, tc=0.10,
                                 vertical=True, tailplane_on_fin=True)


# ------------------------------------------------------- the correlation

def test_a_t_tails_fin_is_the_twenty_percent_heavier_one():
    """The connection, priced. It is the only term in the weight book that
    knows one surface stands on another."""
    conv = weights.tail_weight_raymer(FIN, N_ULT, W_DG)
    tee = weights.tail_weight_raymer(FIN_T, N_ULT, W_DG)
    assert tee == pytest.approx(1.2 * conv, rel=1e-12)


def test_both_surfaces_grow_with_the_weight_they_stabilise():
    for surf in (HT, FIN):
        light = weights.tail_weight_raymer(surf, N_ULT, 5000.0)
        heavy = weights.tail_weight_raymer(surf, N_ULT, 15000.0)
        assert heavy > light
        # ...and sub-linearly, which is what keeps the fixed point a
        # contraction when these join it
        assert heavy / light < 3.0


def test_an_empty_empennage_weighs_nothing():
    assert weights.empennage_weight((), N_ULT, W_DG) == 0.0


# -------------------------------------------------- the layout's surfaces

def test_a_v_tail_carries_no_fin_and_a_conventional_one_does():
    kw = dict(S_t=1.5, b=10.0, S=10.0, l_t=5.5, dz=0.5)
    conv = tailmod.empennage_surfaces(tail_type="conventional", **kw)
    tee = tailmod.empennage_surfaces(tail_type="t_tail", **kw)
    vee = tailmod.empennage_surfaces(tail_type="v_tail", **kw)
    assert [s.vertical for s in conv] == [False, True]
    assert [s.vertical for s in vee] == [False]
    assert tee[1].tailplane_on_fin and not conv[1].tailplane_on_fin


def test_the_v_tails_panels_are_weighed_at_their_true_area():
    """The ``cos^2 gamma`` equivalent flat tail is an AERODYNAMIC device:
    cant costs pitch effectiveness, it does not remove structure."""
    surfaces = tailmod.empennage_surfaces(
        tail_type="v_tail", S_t=1.5, b=10.0, S=10.0, l_t=5.5, dz=0.5)
    assert surfaces[0].S == pytest.approx(1.5)
    assert tailmod.lifting_area("v_tail", 35.0, 1.5) < 1.5   # the aero area


def test_no_second_surface_means_no_empennage():
    assert tailmod.empennage_surfaces(
        tail_type="conventional", S_t=0.0, b=10.0, S=10.0, l_t=5.5,
        dz=0.5) == ()


# ------------------------------------------------------- in the weight loop

def _sized(empennage):
    return sizing.sized_state(
        W_fixed_N=2000.0, b=10.0, S=10.0, taper=0.6, tc=0.12,
        q_Pa=weights.Q_PA_DEFAULT, empennage=empennage)


def test_the_loop_still_closes_and_reports_what_the_tail_cost():
    kw = dict(S_t=1.5, b=10.0, S=10.0, l_t=5.5, dz=0.5)
    bare = _sized(())
    conv = _sized(tailmod.empennage_surfaces(tail_type="conventional", **kw))
    tee = _sized(tailmod.empennage_surfaces(tail_type="t_tail", **kw))
    vee = _sized(tailmod.empennage_surfaces(tail_type="v_tail", **kw))

    assert bare.W_tail_N == 0.0
    assert bare.W_total_N == pytest.approx(bare.W_fixed_N + bare.W_wing_N)
    # every layout weighs more than no empennage, and they differ from
    # each other — which is the whole point
    assert vee.W_tail_N < conv.W_tail_N < tee.W_tail_N
    assert bare.W_total_N < vee.W_total_N < conv.W_total_N < tee.W_total_N
    # the wing's own weight is still reported apart from the tail's, and
    # the three add up to the gross weight the design flies
    for st in (conv, tee, vee):
        assert st.W_total_N == pytest.approx(
            st.W_fixed_N + st.W_wing_N + st.W_tail_N, rel=1e-9)
        assert st.report()["W_tail_N"] == pytest.approx(st.W_tail_N)


def test_a_family_that_passes_none_is_unchanged():
    """The compatibility contract: every caller that never had a tail gets
    the number it always got, to the last bit."""
    before = sizing.sized_state(
        W_fixed_N=2000.0, b=10.0, S=10.0, taper=0.6, tc=0.12,
        q_Pa=weights.Q_PA_DEFAULT)
    assert before.W_tail_N == 0.0
    assert before.W_total_N == pytest.approx(
        weights.total_weight(2000.0, 10.0, 10.0, 0.6, 0.0, 0.12,
                             q_Pa=weights.Q_PA_DEFAULT)[0], rel=1e-12)


# ----------------------------------------------- ...as the FAMILIES fly it

def _sized_tail_family():
    from aerobo import api

    for name in api.PROBLEM_SPECS:
        if name.startswith("tail + free planform"):
            return name
    pytest.skip("no sized tail family in the registry")


def test_a_sized_tail_run_carries_its_empennage_into_the_loop():
    """The wiring, at the call the solver makes."""
    from aerobo import api

    name = _sized_tail_family()
    seen = {}
    real = sizing.sized_state

    def spy(**kw):
        seen.update(kw)
        return real(**kw)

    sizing.sized_state = spy
    try:
        built = api.PROBLEM_SPECS[name].build(
            {}, {"tail_type": "t_tail"}, None)
        built.evaluate(np.asarray(built.bounds.mean(axis=1), dtype=float))
    finally:
        sizing.sized_state = real
    surfaces = seen.get("empennage")
    assert surfaces, "the sized tail family weighed no empennage"
    assert any(s.vertical and s.tailplane_on_fin for s in surfaces), surfaces
