"""Which way the SECOND surface's tip device may point — and the two
numbers that decide it.

The geometry has always been signed (``geometry.winglet_path``: + up, −
down) and the water families fly the signed band, because near a free
surface the direction IS the trade. The air tail inherited the WING's band,
(60, 90) — the aircraft convention, which is about ground clearance — so a
downward device on a stabiliser was not merely unlikely, it was outside the
design box and every evaluation there came back "bounds violation".

That matters because mirroring a surface mirrors its lift: the device
direction that pays on a surface carrying a DOWNLOAD is the opposite of the
one that pays on a lifting wing, and a tail's own load is whatever trims the
aircraft. So the direction is now a stated choice
(``api.TAIL_WINGLET_DIR_KEY``) with the same three meanings in both media —
and every family's own published band is untouched when nobody asks.

The direction is only half the answer, though: WHICH way pays follows the
SIGN of what the trim solve puts on the surface, and that sign is set by the
LAYOUT — where the CG sits, and how far the surface is from the wing. Both
were calibrated constants no shell could state (``tail.X_CG_BY_TYPE``,
``hydrotail.X_CG_FRAC``, the clearance floor), so the tail carried whatever
the calibration implied and the device direction followed. They are now
stated values too (``api.TAIL_CG_KEY`` / ``api.TAIL_HEIGHT_KEY``), and the
engine test in the middle of this file is the one that shows it: move the CG
forward of the wing's AC and the tail's load crosses zero, after which the
mirrored device is the better one.

WHAT THE SHELL SENDS is a separate question from what the engine can do, and
it has changed. V3 sent ``follow``, which let the device read the trimmed
load while ``config.TAIL_MOUNT`` had already pinned the surface to push
DOWN — two answers to one question, and the geometry view drew the
disagreement: an elevator mounted upside down wearing a device pointing up.
V3 now states ``down`` with the mounting (``config.TAIL_TIP_DIRECTION``);
in air that is bit-for-bit what ``follow`` already chose, and in water it
wins 64/68. The engine keeps all four directions, and the tests for
``follow`` below keep testing them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aerobo import api, geometry, wingtail          # noqa: E402

AIR = "tail [designed tail + tip device]"
WATER = "hydrofoil + elevator [designed elevator + tip device]"
ROW = "winglet_cant_t_deg"


def _band(name: str, flags: dict | None = None) -> tuple:
    built = api.PROBLEM_SPECS[name].build({}, flags or {}, None)
    i = list(built.param_labels).index(ROW)
    return (float(built.bounds[i][0]), float(built.bounds[i][1]))


# ----------------------------------------------------- the bands themselves
def test_the_three_directions_are_one_band_mirrored_and_their_span():
    up = wingtail.TAIL_WINGLET_DIRECTIONS["up"]
    down = wingtail.TAIL_WINGLET_DIRECTIONS["down"]
    either = wingtail.TAIL_WINGLET_DIRECTIONS["either"]
    assert tuple(up) == tuple(geometry.WINGLET_CANT_BOUNDS_DEG)
    assert tuple(down) == (-up[1], -up[0])          # the EXACT mirror
    assert tuple(either) == tuple(geometry.WINGLET_CANT_LIMITS_SIGNED_DEG)
    assert either[0] <= down[0] and down[1] <= 0.0 <= up[0] <= either[1]


def test_an_unnamed_direction_is_refused_and_none_means_nobody_asked():
    assert wingtail.tail_cant_bounds_for(None) is None
    assert wingtail.tail_cant_bounds_for("down") == (-90.0, -60.0)
    with pytest.raises(ValueError, match="unknown tail winglet direction"):
        wingtail.tail_cant_bounds_for("sideways")


# ------------------------------------------------ every published box stands
def test_no_family_moves_its_own_band_when_nobody_asks():
    """The whole point of a flag: every registry family that carries a tail
    tip device keeps the band it published."""
    seen = 0
    for name, spec in api.PROBLEM_SPECS.items():
        box = spec.default_bounds.get(ROW)
        if box is None:
            continue
        seen += 1
        want = (geometry.WINGLET_CANT_BOUNDS_DEG if spec.medium == "air"
                else geometry.WINGLET_CANT_LIMITS_SIGNED_DEG)
        assert (float(box[0]), float(box[1])) == tuple(want), name
    assert seen > 10, "no family with a tail tip device was checked"


@pytest.mark.parametrize("name,own", [(AIR, "up"), (WATER, "either")])
def test_the_family_says_which_way_its_device_points(name, own):
    """Read off the family's OWN box, so a shell never restates the pair."""
    assert api.tail_winglet_direction(name) == own
    assert _band(name) == tuple(wingtail.TAIL_WINGLET_DIRECTIONS[own])
    # a family with no device of its own has no direction to report
    assert api.tail_winglet_direction("tail") is None
    assert api.tail_winglet_direction("hydrofoil + elevator") is None


@pytest.mark.parametrize("name", [AIR, WATER])
@pytest.mark.parametrize("direction", ["up", "down", "either"])
def test_a_stated_direction_means_the_same_band_in_both_media(name, direction):
    assert _band(name, {api.TAIL_WINGLET_DIR_KEY: direction}) == \
        tuple(wingtail.TAIL_WINGLET_DIRECTIONS[direction])


def test_the_flag_is_declared_exactly_where_the_device_exists():
    for name, spec in api.PROBLEM_SPECS.items():
        has_row = ROW in spec.default_bounds
        assert (api.TAIL_WINGLET_DIR_KEY in spec.flags) == has_row, name


# ------------------------------------------------------- it actually flies
def test_the_air_tail_can_now_fly_a_device_that_points_down():
    """Before: every candidate with a negative cant was refused by the
    problem's own box ("bounds violation"), so the mirror could not even be
    evaluated."""
    spec = api.PROBLEM_SPECS[AIR]
    box = spec.default_bounds
    x = [0.5 * (box[l][0] + box[l][1]) for l in spec.param_labels]
    x[spec.param_labels.index("winglet_h_frac_t")] = 0.12
    x[spec.param_labels.index(ROW)] = -75.0

    plain = api.design_report(api.RunConfig(problem_name=AIR), x)
    assert plain["breakdown"]["reason"] == "bounds violation"

    cfg = api.RunConfig(problem_name=AIR,
                        flags={api.TAIL_WINGLET_DIR_KEY: "down"})
    rep = api.design_report(cfg, x)
    assert rep["breakdown"]["feasible"] is True
    assert rep["breakdown"]["LoD"] > 0.0
    ts = rep["geometry"]["tail_surface"]
    z = np.asarray(ts["z"], float)
    wl = np.asarray(ts["is_winglet"], bool)
    assert wl.any()
    assert z[wl].min() < z[~wl].mean()      # the device hangs BELOW the tail
    assert z[wl].max() <= z[~wl].mean() + 1e-9


def test_the_mirror_is_a_real_difference_not_a_relabelling():
    """Up and down are not the same design: the tail carries whatever trims
    the aircraft, so mirroring the device changes the drag it makes."""
    spec = api.PROBLEM_SPECS[AIR]
    box = spec.default_bounds
    x = [0.5 * (box[l][0] + box[l][1]) for l in spec.param_labels]
    x[spec.param_labels.index("winglet_h_frac_t")] = 0.12
    i = spec.param_labels.index(ROW)

    x_up = list(x)
    x_up[i] = 75.0
    x_dn = list(x)
    x_dn[i] = -75.0
    up = api.design_report(api.RunConfig(problem_name=AIR), x_up)
    dn = api.design_report(
        api.RunConfig(problem_name=AIR,
                      flags={api.TAIL_WINGLET_DIR_KEY: "down"}), x_dn)
    assert up["breakdown"]["LoD"] != pytest.approx(dn["breakdown"]["LoD"],
                                                   rel=1e-9)
    # ...and both are real designs, trimmed to the same lift
    assert up["breakdown"]["CL"] == pytest.approx(dn["breakdown"]["CL"],
                                                  rel=1e-6)


# ---------------------------------------------------------- the guard rails
def test_a_band_outside_the_signed_range_is_refused_on_both_surfaces():
    from aerobo import hydrotail

    with pytest.raises(ValueError, match="outside the physical"):
        wingtail.WingTailProblem(tail_free=True, tail_winglet=True,
                                 tail_cant_bounds=(-120.0, -60.0)).bounds
    with pytest.raises(ValueError, match="outside the physical"):
        hydrotail.HydrofoilTailProblem(tail_free=True, tail_winglet=True,
                                       tail_cant_bounds=(-95.0, 0.0)).bounds


def test_the_wings_own_device_keeps_the_positive_only_rule():
    """Only the TRIMMING surface's band went signed. A wing's winglet is
    still validated against the aircraft range — nothing about ground
    clearance changed."""
    with pytest.raises(ValueError, match="outside the physical"):
        wingtail.WingTailProblem(winglet=True,
                                 winglet_cant_bounds=(-90.0, -60.0)).bounds
    assert tuple(geometry.WINGLET_CANT_LIMITS_DEG) == (5.0, 90.0)


# --------------------------------------------------------- through the shell
def test_the_wing_stage_states_the_direction_and_nothing_can_outvote_it():
    """V3 answers this one (``config.TAIL_TIP_DIRECTION``), so the flag is
    there from the moment the surface carries a device — and it is PINNED,
    which is the half a card-level default would get wrong."""
    from gui.v3 import config
    from gui.v3.app import assemble

    ctx = assemble("air")
    ctx.act("accept_mission")
    ctx.act("set_second_surface", True)
    ctx.act("set_choice", "tail_design", "planform+tip")
    S = ctx.S
    assert ROW in api.PROBLEM_SPECS[S["wing"]["problem"]].default_bounds

    assert config.flags(S)[api.TAIL_WINGLET_DIR_KEY] == "down"
    built = api.PROBLEM_SPECS[S["wing"]["problem"]].build(
        {}, config.flags(S), None)
    i = list(built.param_labels).index(ROW)
    # the WHOLE band is below zero: no candidate in this box points up
    assert float(built.bounds[i][1]) < 0.0
    assert built.problem.tail_winglet_follow is False

    # the read-out is on screen, with the reason it exists
    ctx.render("wing", "type")
    texts = [getattr(e, "text", "") or ""
             for e in ctx.views[("wing", "type")].descendants()]
    assert any(t == "points" for t in texts)
    assert any("the side this surface works" in t for t in texts)

    # ...and a choice that says otherwise does not win: one place owns it
    ctx.act("set_choice", "tail_winglet_dir", "up")
    assert config.flags(S)[api.TAIL_WINGLET_DIR_KEY] == "down"


def test_the_direction_never_travels_to_a_family_that_has_no_device():
    """A flag a family does not declare would raise inside its builder."""
    from gui.nice_app import tail_flags

    assert tail_flags({"tail": True, "tail_design": "planform",
                       "tail_winglet_dir": "down"}) == {}
    assert tail_flags({"tail": True, "tail_design": "planform+tip",
                       "tail_winglet_dir": "down"}) == \
        {"tail_winglet_dir": "down"}


# ================================================== the layout that sets it
def test_the_cg_and_the_separation_are_stated_values_not_calibrations():
    """Both travel as flags, in each family's own frame, and neither moves
    a design variable."""
    from aerobo import tail as tailmod

    air = api.PROBLEM_SPECS[AIR]
    assert api.TAIL_CG_KEY in air.flags and api.TAIL_HEIGHT_KEY in air.flags
    built = air.build({}, {}, None)
    # untouched: the family's own calibrated CG, and the layout's own height
    assert built.problem.x_cg == tailmod.X_CG_BY_TYPE["conventional"]
    assert built.problem.z_t_fixed is None
    assert len(built.param_labels) == len(air.param_labels)

    stated = air.build({}, {api.TAIL_CG_KEY: -0.30,
                            api.TAIL_HEIGHT_KEY: 1.4}, None)
    assert stated.problem.x_cg == -0.30
    assert stated.problem.z_t_fixed == 1.4
    # a VALUE, not a variable: the design vector is the same length
    assert len(stated.param_labels) == len(air.param_labels)


def test_the_water_elevator_takes_the_same_two_in_its_own_frame():
    water = api.PROBLEM_SPECS[WATER]
    assert api.TAIL_CG_KEY in water.flags
    assert api.TAIL_HEIGHT_KEY in water.flags
    built = water.build({}, {api.TAIL_CG_KEY: 0.08,
                             api.TAIL_HEIGHT_KEY: -0.25}, None)
    assert built.problem.x_cg == 0.08
    assert built.problem.z_t_fixed == -0.25      # negative: BELOW the foil


def test_a_height_that_is_already_a_design_variable_is_not_also_stated():
    """Freeing it and stating it is one question answered twice — the
    problem refuses the pair, and the free-height families do not even
    declare the flag."""
    from aerobo import wingtail as wt

    free = [n for n, v in api.WING_TAIL_VARIANTS.items()
            if v["height"] == "free"]
    assert free
    for name in free:
        assert api.TAIL_HEIGHT_KEY not in api.PROBLEM_SPECS[name].flags, name
        assert api.TAIL_CG_KEY in api.PROBLEM_SPECS[name].flags, name
    with pytest.raises(ValueError, match="not both"):
        wt.WingTailProblem(free_height=True, z_t_fixed=1.4).bounds


def test_a_separation_inside_the_trailing_sheet_is_built_and_flown():
    """It used to be refused BOTH ways, and that was the ban this test now
    guards against coming back.

    ``DZ_FRAC*b`` is where these families' second surfaces were MEASURED —
    0.5 m on a 10 m span, 0.06 m on a 1.2 m foil — not the smallest one that
    exists, and a tailplane at or near the wing plane is the ordinary
    layout. Both cores were re-measured smooth through it to zero
    (``tail.DZ_GRID_FRAC``, tests/test_height_band_is_the_users.py).
    """
    from aerobo import hydrotail, tail as tailmod, wingtail as wt

    floor = tailmod.DZ_FRAC * 10.0
    assert wt.WingTailProblem(b=10.0, z_t_fixed=0.5 * floor).z_t_fixed \
        == pytest.approx(0.5 * floor)
    # ...and in water, where the separation is measured DOWN, a stabiliser
    # ABOVE the foil is a layout the user may ask for and get
    assert hydrotail.HydrofoilTailProblem(z_t_fixed=+0.20).z_t_fixed \
        == pytest.approx(0.20)


def _tail_load(x_cg: float) -> float:
    spec = api.PROBLEM_SPECS[AIR]
    box = spec.default_bounds
    x = [0.5 * (box[l][0] + box[l][1]) for l in spec.param_labels]
    x[spec.param_labels.index("winglet_h_frac_t")] = 0.12
    rep = api.design_report(
        api.RunConfig(problem_name=AIR, flags={api.TAIL_CG_KEY: x_cg}), x)
    return float(rep["breakdown"]["CL_t"])


def test_the_cg_sets_the_sign_of_what_the_tail_carries():
    """THE MECHANISM. Moving the CG aft moves the tail's load towards LIFT
    and moving it forward towards DOWNLOAD — monotonically, one for one.

    What it does NOT do is cross zero at the wing AC. The tail carries the
    wing's own nose-down section couple as well as the CG offset
    (tail.section_moment), so with the CG exactly at the AC there is still
    a moment to hold and the tail still pushes DOWN. The crossing sits
    where the two cancel,

        x_cg = -M_ac / (CL_target Sref) = +0.123 m

    aft of the AC on this aircraft — i.e. at 37 % MAC, aft of any normal
    loading, which is the quantitative form of "a conventional tail
    downloads"."""
    assert _tail_load(+0.25) > 0.0                     # aft of the crossing
    assert _tail_load(+0.123) == pytest.approx(0.0, abs=2e-3)
    assert _tail_load(0.0) < 0.0                       # at the wing AC
    assert _tail_load(-0.30) < 0.0
    assert _tail_load(-0.60) < _tail_load(-0.30)      # further forward, more
    # monotone in between, and it is the CG that does it
    loads = [_tail_load(c) for c in (-0.60, -0.30, 0.0, +0.123, +0.25)]
    assert loads == sorted(loads)


def test_and_that_sign_is_what_makes_the_mirrored_device_the_better_one():
    """Which way the device pays is not a matter of taste: with the tail
    carrying LIFT the up device wins, and once the CG puts a DOWNLOAD on it
    the down device does. That is why the direction had to become a choice
    and why the CG had to become stateable."""
    spec = api.PROBLEM_SPECS[AIR]
    box = spec.default_bounds
    base = [0.5 * (box[l][0] + box[l][1]) for l in spec.param_labels]
    base[spec.param_labels.index("winglet_h_frac_t")] = 0.12
    i = spec.param_labels.index(ROW)

    def lod(x_cg: float, cant: float) -> float:
        flags = {api.TAIL_CG_KEY: x_cg}
        if cant < 0.0:
            flags[api.TAIL_WINGLET_DIR_KEY] = "down"
        x = list(base)
        x[i] = cant
        return float(api.design_report(
            api.RunConfig(problem_name=AIR, flags=flags), x)["breakdown"]["LoD"])

    lifting = +0.25          # the family's own CG: the tail LIFTS
    assert _tail_load(lifting) > 0.0
    assert lod(lifting, +75.0) > lod(lifting, -75.0)

    loading = -0.30          # CG forward: the tail carries a DOWNLOAD
    assert _tail_load(loading) < 0.0
    assert lod(loading, -75.0) > lod(loading, +75.0)


def test_the_wing_stage_offers_both_numbers_and_states_the_rule():
    from gui.v3 import config
    from gui.v3.app import assemble

    ctx = assemble("air")
    ctx.act("accept_mission")
    ctx.act("set_second_surface", True)
    ctx.act("set_choice", "tail_design", "planform+tip")
    S = ctx.S
    assert api.TAIL_CG_KEY not in config.flags(S)     # untouched: published

    ctx.act("set_choice", "tail_cg_m", -0.30)
    assert config.flags(S)[api.TAIL_CG_KEY] == -0.30
    built = api.PROBLEM_SPECS[S["wing"]["problem"]].build(
        {}, config.flags(S), None)
    assert built.problem.x_cg == -0.30

    ctx.render("wing", "type")
    texts = [getattr(e, "text", "") or ""
             for e in ctx.views[("wing", "type")].descendants()]
    assert any(t == "CG" for t in texts)
    assert any(t == "separation" for t in texts)
    assert any("DOWNLOAD" in t for t in texts)

    # clearing it hands the number back to the family
    ctx.act("set_choice", "tail_cg_m", None)
    assert api.TAIL_CG_KEY not in config.flags(S)


def test_a_stated_height_never_travels_to_a_family_that_frees_it():
    from gui.nice_app import tail_flags

    assert tail_flags({"tail": True, "tail_height": "free",
                       "tail_height_m": 1.4}) == {}
    assert tail_flags({"tail": True, "tail_height": "fixed",
                       "tail_height_m": 1.4}) == {"z_t_m": 1.4}


# ============================== the direction that is DERIVED, not asked
def test_follow_mode_leaves_the_magnitude_as_the_variable():
    """The band is the PUBLISHED one: in follow mode the design vector
    carries the cant's magnitude, and the side is not in it at all."""
    built = api.PROBLEM_SPECS[AIR].build(
        {}, {api.TAIL_WINGLET_DIR_KEY: "follow"}, None)
    i = list(built.param_labels).index(ROW)
    assert (float(built.bounds[i][0]), float(built.bounds[i][1])) == \
        tuple(geometry.WINGLET_CANT_BOUNDS_DEG)
    assert built.problem.tail_winglet_follow is True
    # ...and nobody gets it by accident
    assert api.PROBLEM_SPECS[AIR].build({}, {}, None) \
        .problem.tail_winglet_follow is False


def _flown(x_cg: float, follow: bool, cant: float = 75.0) -> dict:
    spec = api.PROBLEM_SPECS[AIR]
    box = spec.default_bounds
    x = [0.5 * (box[l][0] + box[l][1]) for l in spec.param_labels]
    x[spec.param_labels.index("winglet_h_frac_t")] = 0.12
    x[spec.param_labels.index(ROW)] = cant
    flags = {api.TAIL_CG_KEY: x_cg}
    if follow:
        flags[api.TAIL_WINGLET_DIR_KEY] = "follow"
    return api.design_report(
        api.RunConfig(problem_name=AIR, flags=flags), x)["breakdown"]


def test_the_device_takes_its_side_from_the_load_without_being_told():
    """THE POINT. Nobody states a direction: the surface's own trim load
    does, per candidate, and it picks the better side in both regimes."""
    lifting = _flown(+0.25, follow=True)
    assert lifting["CL_t"] > 0.0
    assert lifting["tail_winglet_side"] == "up"
    assert lifting["tail_winglet_follows_load"] is True
    # ...and where the load is positive that IS the published design
    assert lifting["LoD"] == pytest.approx(_flown(+0.25, follow=False)["LoD"],
                                           rel=1e-12)

    loading = _flown(-0.30, follow=True)
    assert loading["CL_t"] < 0.0
    assert loading["tail_winglet_side"] == "down"
    assert loading["tail_winglet_cant_deg"] < 0.0
    # the mirror it chose is the better one, and the run that was NOT told
    # to follow keeps the worse side
    assert loading["LoD"] > _flown(-0.30, follow=False)["LoD"]


def test_following_never_moves_a_published_run():
    """Off by default, and off it is bit-for-bit the run that shipped."""
    for x_cg in (0.25, -0.30):
        plain = _flown(x_cg, follow=False)
        assert plain["tail_winglet_follows_load"] is False
        assert plain["tail_winglet_side"] == "up"      # where the vector put it
        assert plain["tail_winglet_cant_deg"] == pytest.approx(75.0)


def test_the_water_elevator_follows_its_own_load_too():
    """Same rule, and the report says which side it landed on. Under water
    the free surface breaks the mirror symmetry, so the side the load implies
    is not always the lower-drag one — the rule is about the surface's
    orientation, and the search (`either`) is what optimises it."""
    from aerobo import hydrotail

    def fly(x_cg, follow):
        prob = hydrotail.HydrofoilTailProblem(
            tail_free=True, tail_winglet=True, x_cg=x_cg,
            tail_winglet_follow=follow)
        labels = list(prob.param_labels)
        box = prob.bounds
        x = 0.5 * (box[:, 0] + box[:, 1])
        x[labels.index("winglet_h_frac_t")] = 0.12
        x[labels.index(ROW)] = 70.0
        return hydrotail.evaluate_hydrofoil_tail(x, prob)

    lifting = fly(0.15, follow=True)
    assert lifting["CL_stab"] > 0.0
    assert lifting["tail_winglet"]["side"] == "up"

    loading = fly(-0.05, follow=True)
    assert loading["CL_stab"] < 0.0
    assert loading["tail_winglet"]["side"] == "down"
    assert loading["tail_winglet"]["follows_load"] is True
    assert loading["tail_winglet"]["cant_deg"] < 0.0


def test_the_shell_does_not_ask_which_way_the_device_points():
    """V3 states the direction; it does not offer it.

    It was a four-entry toggle, then ``follow`` (the side read off each
    candidate's own trim load). It is now ``down``, with the MOUNTING: this
    shell has already answered which way the second surface works, and
    letting the device answer the same question a second way is what put a
    device pointing up on a surface built to push down.
    """
    from gui.v3 import config
    from gui.v3.app import assemble

    ctx = assemble("air")
    ctx.act("accept_mission")
    ctx.act("set_second_surface", True)
    S = ctx.S
    # no tip device yet -> nothing about one travels
    assert api.TAIL_WINGLET_DIR_KEY not in config.flags(S)

    ctx.act("set_tail_tip", "canted")
    assert S["wing"]["choices"]["tail_design"] == "planform+tip"
    assert config.flags(S)[api.TAIL_WINGLET_DIR_KEY] == "down"
    built = api.PROBLEM_SPECS[S["wing"]["problem"]].build(
        {}, config.flags(S), None)
    assert built.problem.tail_winglet_follow is False
    # ...and it is the SAME answer the mounting gives
    assert config.flags(S)[api.TAIL_MOUNT_KEY] == "inverted"

    ctx.render("wing", "type")
    texts = [getattr(e, "text", "") or ""
             for e in ctx.views[("wing", "type")].descendants()]
    assert any("downwards — the side this surface works" in t for t in texts)
    assert any("Not a choice." in t for t in texts)
    # ...and the toggle's four entries are gone from the card
    assert not any("either (searched)" in t for t in texts)

    # taking the device off takes the direction with it
    ctx.act("set_tail_tip", "none")
    assert api.TAIL_WINGLET_DIR_KEY not in config.flags(S)


# ================================================== and what is on the screen
@pytest.mark.parametrize("medium", ["air", "water"])
def test_the_device_is_drawn_below_the_surface_it_hangs_off(medium):
    """THE PICTURE, end to end: shell -> flags -> solve -> lofted surface.

    Every assertion above this one is about a band, a flag or a label, and a
    tip device pointing the wrong way is none of those — it is a coordinate.
    The one that was shipped had a correct band, a correct flag and a label
    reading ``side: up`` on a surface the same shell had mounted upside down
    to push DOWN, and only the geometry view said so. So this asserts the z
    the viewer actually sees: the device's panels, and the drawn surface
    lofted through them, are BELOW the plane of the surface they hang off.
    """
    import numpy as np

    from aerobo import cad
    from gui.v3 import config
    from gui.v3.app import assemble

    ctx = assemble(medium)
    ctx.act("accept_mission")
    ctx.act("set_second_surface", True)
    ctx.act("set_tail_tip", "canted")
    S = ctx.S
    name = S["wing"]["problem"]
    flags = config.flags(S)
    assert flags[api.TAIL_WINGLET_DIR_KEY] == "down"

    built = api.PROBLEM_SPECS[name].build({}, flags, None)
    labels = list(built.param_labels)
    assert ROW in labels
    x = np.array([0.5 * (lo + hi) for lo, hi in built.bounds])
    rep = api.design_report(
        api.RunConfig(problem_name=name, flags=flags), x)
    bd, geom = rep["breakdown"], rep["geometry"]
    assert bd.get("feasible"), bd.get("reason")

    # what the SOLVER says it flew — the two families report it differently
    wl = bd.get("tail_winglet")
    side = wl.get("side") if isinstance(wl, dict) else bd.get(
        "tail_winglet_side")
    assert side == "down"

    # ...the PANELS the run handed the shell
    sf = geom["tail_surface"]
    z = np.asarray(sf["z"], dtype=float)
    on_device = np.asarray(sf["is_winglet"], dtype=bool)
    assert on_device.any() and (~on_device).any()
    plane = float(np.mean(z[~on_device]))
    assert z[on_device].min() < plane - 1e-6      # it reaches DOWN
    assert z[on_device].max() <= plane + 1e-9     # ...and never up

    # ...and the surface those panels are LOFTED into, which is the picture
    xc, zc = cad.section_path(geom, x, labels)
    drawn = cad.tail_surfaces(geom.get("tail") or {}, sf, xc, zc)
    assert drawn, "the tail drew nothing at all"
    lo = min(float(np.min(s.Z)) for s in drawn)
    assert lo < plane - 1e-6
    # the deepest point is the device's, not the section's own thickness
    assert lo < float(np.min(z[~on_device])) - 1e-6
