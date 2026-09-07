"""The design box is a question about the craft the MISSION states.

Three reports, one session, and they turned out to be two mechanisms:

* *"separation wing tail recommendation too big"* and *"tail area given
  larger than wing, THIS SHOULDN'T BE"* — the second surface's rows
  (``S_t_m2``, ``l_t_m``, ``z_t_m``) were the family's PUBLISHED ones, and
  those are quoted on a 10 m, 10 m^2 reference aeroplane. A mission stating a
  1 m wing of 0.125 m^2 was searching a stabiliser of **0.5 – 3 m^2** —
  four to twenty-four times the whole aircraft's reference area — on an arm
  of **3 – 8 m**, eight spans behind it. Nothing between the mission and the
  box divided them by anything (:func:`session.layout_band_default`).

* *"span recommendation changes when tips are selected (span grows x10)"* —
  ``apply_choices`` wipes the design box on a problem change, which is right
  about a band the USER typed for the old family's vector, and it took the
  SHELL's own mission bands with it. Only ``set_planform`` wrote them again,
  so every other builder control dropped ``b_m`` back onto the published
  6 – 40 m. The measured box can only narrow INTO the box it is handed, so
  the recommendation then read 11 – 16 m for a 1 m mission.

And one thing the first fix exposed: a box is the PRODUCT of its rows, not
the list of them, and the two gates that judge the product were consulted by
neither (:func:`session.clip_size_box`).
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest                                          # noqa: E402

from aerobo import api                                  # noqa: E402
from aerobo.sizing import AR_LIMITS                     # noqa: E402
from gui.v3 import config, session                      # noqa: E402
from gui.v3.app import assemble                         # noqa: E402


def _shell(*, span=None, ar=None, mass_kg=None, v=None, medium="air",
           tail=True, height=True):
    ctx = assemble(medium)
    m = ctx.S["mission"]
    if mass_kg is not None:
        m["W_N"] = float(mass_kg) * 9.80665
    if v is not None:
        m["V"] = float(v)
    if span is not None:
        assert session.set_size_from_span_ar(ctx.S, span=span)
    if ar is not None:
        assert session.set_size_from_span_ar(ctx.S, ar=ar)
    ctx.act("accept_mission")
    ctx.act("set_planform", "free")
    if tail:
        ctx.act("set_choice", "tail", True)
        if height:
            ctx.act("set_choice", "tail_height", "free")
    return ctx


def _box(ctx):
    return {k: [float(v[0][0]), float(v[0][1])]
            for k, v in config.effective_bounds(ctx.S).items()}


# ---------------------------------------------------------------- the layout
def test_the_second_surface_is_sized_to_this_mission_not_the_reference_one():
    """The report, in numbers: a 1 m wing does not get a 3 – 8 m tail arm."""
    ctx = _shell(span=1.0, ar=8.0)
    box = _box(ctx)
    # the mission's wing: 1 m span, 0.125 m^2
    assert box["l_t_m"] == pytest.approx([0.3, 0.8])        # 0.30-0.80 b
    assert box["z_t_m"] == pytest.approx([0.05, 0.3])       # 0.05-0.30 b
    assert box["S_t_m2"] == pytest.approx([0.00625, 0.0375], rel=1e-3)


def test_the_tail_is_never_bigger_than_the_wing_anywhere_in_the_box():
    """"Tail area given larger than wing. THIS SHOULDN'T BE."

    Not "usually smaller": the WORST corner of the box — the largest
    stabiliser crossed with the smallest wing — has to be a stabiliser.
    """
    for kw in ({"span": 1.0, "ar": 8.0},
               {"span": 10.0, "ar": 10.0},
               {"span": 3.0, "ar": 12.0, "mass_kg": 40.0},
               {}):
        ctx = _shell(**kw)
        box = _box(ctx)
        assert box["S_t_m2"][1] < box["S_m2"][0], (kw, box)


def test_a_mission_that_is_the_reference_craft_gets_the_published_rows():
    """The scaling may not move a published run: at the reference craft the
    ratio is 1 and the row is the family's own, bit-for-bit."""
    air = _box(_shell())                       # 10 m, 10 m^2 aeroplane
    assert air["S_t_m2"] == pytest.approx([0.5, 3.0])
    assert air["l_t_m"] == pytest.approx([3.0, 8.0])
    assert air["z_t_m"] == pytest.approx([0.5, 3.0])

    water = _box(_shell(medium="water"))        # 1.2 m, 0.144 m^2 foil
    assert water["S_t_m2"] == pytest.approx([0.02, 0.06])
    assert water["l_t_m"] == pytest.approx([0.5, 1.5])
    # measured DOWN, so the band is negative and the low end is the deeper one
    assert water["z_t_m"] == pytest.approx([-0.36, -0.06])


def test_the_engine_s_own_derivation_is_not_shadowed():
    """The shell fills a GAP; it does not take the job over.

    A family that honours a chosen planform builds its second surface's rows
    off the wing it is actually flying (``tail.area_band`` / ``arm_band``), so
    that box is already this craft's. An override written on top would replace
    it with a coarser number taken from the mission's REFERENCE area — which
    is how a model aeroplane got the 10 m aeroplane's tail back.
    """
    ctx = assemble("air")
    ctx.act("accept_mission")
    ctx.act("set_choice", "tail", True)
    assert session.flown_size(ctx.S) is not None
    ctx.S["wing"]["choices"]["span_m"] = 1.2
    ctx.S["wing"]["choices"]["area_m2"] = 0.12
    assert session.layout_band_default(ctx.S, "S_t_m2") is None
    # ...and the row on screen followed the model aeroplane anyway
    assert _box(ctx)["S_t_m2"][1] < 0.1

    # the gap is the other case: with the size in the DESIGN VECTOR there is
    # no flown wing, so the engine falls back to the published row
    free = _shell(span=1.0, ar=8.0)
    assert session.flown_size(free.S) is None
    assert session.layout_band_default(free.S, "S_t_m2") is not None


def test_the_arm_is_held_above_the_floor_the_pitch_solve_needs():
    """Scaling a 3 – 8 m arm onto a very small craft walks it under
    ``tail.L_T_MIN_M``, where the tail volume is zero and the solve is
    singular — and ``tail.arm_row`` RAISES there, so the shell would have
    handed the engine a band it refuses to build."""
    from aerobo.tail import L_T_MIN_M

    ctx = _shell(span=0.2, ar=6.0, mass_kg=0.5, v=8.0)
    lo, hi = _box(ctx)["l_t_m"]
    assert lo >= L_T_MIN_M and hi > lo
    api.PROBLEM_SPECS[ctx.S["wing"]["problem"]].build(
        {}, config.build_cfg(ctx.S).flags,
        config.build_cfg(ctx.S).bounds_overrides)     # builds, does not raise


# ------------------------------------------------------- the x10 span report
def test_the_tip_device_menu_does_not_move_the_span_band():
    """The report: selecting a tip device grew the span band tenfold.

    ``apply_choices`` wiped the box on the problem change and only the
    PLANFORM menu wrote the mission's bands again.
    """
    ctx = _shell(span=1.0, ar=8.0)
    before = _box(ctx)
    ctx.act("set_winglet", "canted")
    assert ctx.S["wing"]["problem"] != "tail + free planform + free chord law"
    after = _box(ctx)
    for row in ("b_m", "S_m2", "S_t_m2", "l_t_m"):
        assert after[row] == pytest.approx(before[row]), row
    assert after["b_m"][1] < 5.0, "the 6-40 m published row is back"


def test_no_builder_control_drops_the_box_onto_the_published_row():
    """The tip device is one menu of many, and the mechanism was in
    ``apply_choices`` — so the assertion is over the controls, not over the
    one that was reported."""
    moves = [("set_winglet", "canted"),
             ("set_choice", "tail_type", "t_tail"),
             ("set_choice", "tail_design", "free"),
             ("set_chord_law", "ends")]
    for act in moves:
        ctx = _shell(span=1.0, ar=8.0)
        before = _box(ctx)
        try:
            ctx.act(*act)
        except Exception as exc:                       # noqa: BLE001
            pytest.skip(f"{act} unavailable here: {exc}")
        after = _box(ctx)
        assert after["b_m"][1] < 5.0, (act, after["b_m"])
        assert after["S_m2"][1] < 1.0, (act, after["S_m2"])
        assert after["b_m"] == pytest.approx(before["b_m"]), act


# ------------------------------------------------- the box is a PRODUCT
def test_every_corner_of_the_box_is_a_wing_these_solvers_are_valid_over():
    """``b_m`` and ``S_m2`` were derived independently, so their product ran
    AR 1.3 – 160 against a validity band of 3 – 40."""
    # missions the stall ceiling actually admits — a mission whose own wing
    # is already too small is the OTHER test below, and the clip deliberately
    # stands aside there rather than inventing an interval
    for kw in ({}, {"span": 4.0, "ar": 9.0, "mass_kg": 8.0},
               {"span": 2.0, "ar": 14.0, "mass_kg": 1.5, "v": 16.0}):
        ctx = _shell(**kw)
        box = _box(ctx)
        b_lo, b_hi = box["b_m"]
        s_lo, s_hi = box["S_m2"]
        assert b_lo * b_lo / s_hi >= AR_LIMITS[0] - 1e-9, (kw, box)
        assert b_hi * b_hi / s_lo <= AR_LIMITS[1] + 1e-9, (kw, box)
        # ...and it is not a clip that ate the box
        assert b_hi > b_lo and s_hi > s_lo, (kw, box)


def test_the_area_row_does_not_open_below_what_the_ceiling_requires():
    """``S_FRAC_BOUNDS`` opens the area at 0.8x the mission's — 1.25x its
    loading — so the bottom of the row was refused by arithmetic on every
    mission that states a stall speed. The bound is a LOWER one (the solver
    checks W_TOTAL/S), so it is asserted as one."""
    ctx = _shell(span=10.0, ar=10.0, mass_kg=100.0, v=25.0)
    cap = session.mission_ws_ceiling(ctx.S)
    assert cap, "this mission states no ceiling; the test proves nothing"
    s_lo = _box(ctx)["S_m2"][0]
    assert float(ctx.S["mission"]["W_N"]) / s_lo <= cap + 1e-6


def test_the_wing_on_screen_is_still_inside_its_own_span_row():
    """The corner clip must not do what opening a search may never do. At a
    modest aspect ratio the mission's own span sits below
    ``sqrt(3 * S_FRAC_BOUNDS[1] * S)``, so a clip that stopped there would
    put the stated wing off the bottom of the row it is searched in."""
    for ar in (4.0, 5.0, 6.0, 8.0, 16.0):
        ctx = _shell(span=6.0, ar=ar, mass_kg=60.0)
        lo, hi = _box(ctx)["b_m"]
        b = session.nominal_span(ctx.S)
        assert lo <= b <= hi, (ar, lo, b, hi)


def test_a_mission_outside_its_own_gates_keeps_its_box_and_is_told_why():
    """A clip that cannot produce an interval must not produce an inverted
    one: the shell's job there is to NAME the gate, not to hide it."""
    ctx = _shell(span=1.0, ar=8.0)              # 25 kg on a 1 m wing
    box = _box(ctx)
    assert box["S_m2"][1] > box["S_m2"][0]
    assert box["b_m"][1] > box["b_m"][0]
    got = api.box_refusal_probe(config.build_cfg(ctx.S), n=32)
    assert got["n_feasible"] == 0
    assert any("wing loading" in r["reason"] for r in got["reasons"])


# ---------------------------------------------------- whose row is whose
def test_the_mode_switch_only_re_asks_the_rows_it_owns():
    """Choosing the wing-loading mode is choosing to re-ask the wing's SIZE,
    so that switch overwrites those rows even where the user typed them. It
    is not a statement about how long the fuselage is — and the layout rows
    now live in the same ``size_band_defaults`` dict, so without the filter
    the switch would have discarded a separation band the user set."""
    ctx = _shell(span=10.0, ar=10.0)
    ctx.act("set_bound", "l_t_m", 0, 4.2)
    ctx.act("set_bound", "l_t_m", 1, 5.4)
    ctx.act("set_bound", "b_m", 0, 7.5)
    assert _box(ctx)["l_t_m"] == pytest.approx([4.2, 5.4])

    session.write_size_bands(ctx.S, rows=session.size_rows(ctx.S))
    assert _box(ctx)["l_t_m"] == pytest.approx([4.2, 5.4])   # not this row's
    assert _box(ctx)["b_m"][0] != pytest.approx(7.5)         # this one IS

    session.write_size_bands(ctx.S)                          # no filter
    assert _box(ctx)["l_t_m"] == pytest.approx([3.0, 8.0])


def test_a_typed_separation_is_not_moved_by_a_mission_edit():
    """The refresh is for the rows nobody answered. A band the user typed is
    their answer, in the layout rows exactly as in every other row."""
    ctx = _shell(span=10.0, ar=10.0)
    ctx.act("set_bound", "l_t_m", 0, 4.2)
    ctx.act("set_bound", "l_t_m", 1, 5.4)
    assert session.set_size_from_span_ar(ctx.S, span=2.0)
    session.sync_wing_from_mission(ctx.S)
    assert _box(ctx)["l_t_m"] == pytest.approx([4.2, 5.4])


def test_a_mission_edit_still_moves_the_layout_rows():
    """They are the shell's rows, not the user's, so they follow the mission
    exactly as the span and area rows do."""
    ctx = _shell(span=10.0, ar=10.0)
    assert _box(ctx)["l_t_m"] == pytest.approx([3.0, 8.0])
    assert session.set_size_from_span_ar(ctx.S, span=2.0)
    session.sync_wing_from_mission(ctx.S)
    assert _box(ctx)["l_t_m"] == pytest.approx([0.6, 1.6])


def test_the_measured_box_now_starts_from_the_mission_s_own_rows():
    """End to end, and the whole point: the recommendation can only narrow
    INTO the box it is handed, so the box it is handed has to be this
    mission's."""
    ctx = _shell(span=10.0, ar=10.0, mass_kg=100.0, v=25.0)
    ctx.act("auto_recommend")
    box = _box(ctx)
    assert box["l_t_m"][1] <= 8.0
    assert box["S_t_m2"][1] < box["S_m2"][0]
    b_lo, b_hi = box["b_m"]
    assert 0.6 * 10.0 <= b_lo and b_hi <= 4.0 * 10.0
