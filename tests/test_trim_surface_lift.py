"""What a TRIMMING surface actually flies at — and why it is not zero.

The second surface of a tailed aircraft (and of a foiling craft) used to be
screened at Cl = 0, on the argument that what a stabiliser carries is
"whatever trims" and therefore unstateable. It is stateable, exactly: the
two equations every one of these solvers imposes —

    Cm about the CG = 0        and        total CL = the target

— fix each surface's load without any aerodynamics at all. Eliminating the
wing's share leaves ``tail.trim_lift_coefficient``:

    CL_t = CL_target * Sref * x_cg / (S_t * l_t)

which is 0.13 on the published aircraft tail and 0.30 on the hydrofoil's
elevator. Screening at zero lift instead made "L/D at the design Cl" read
cl/cd = 0 for every candidate in the database, so the ranking fell back on
|Cm| — i.e. on camber — and returned a symmetric section by construction
rather than by argument.

This file gates the three things that fix has to keep true:

1. the closed form SATISFIES the two trim equations (algebra, no solver);
2. it agrees with what the solvers actually solve, on every layout —
   conventional, canard, V-tail, T-tail, free-height and the hydrofoil;
3. the V3 shell screens that surface at it, and the screen's cruise-L/D
   criterion ranks something again as a result.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aerobo import api                                          # noqa: E402
from aerobo import polar as _polar                              # noqa: E402
from aerobo.tail import (orient_section,                        # noqa: E402
                         trim_lift_coefficient)


# ------------------------------------------------------- 1. the algebra
def test_the_closed_form_satisfies_both_trim_equations():
    """It is not a correlation: given the CL it returns, the wing's share
    follows from the total, and the moment about the CG is then zero."""
    CL_target, Sref, mac = 0.5, 10.0, 1.02
    for x_cg, S_t, l_t in ((0.25, 1.75, 5.5), (-0.40, 1.75, -5.5),
                           (0.25, 1.10, 8.0), (0.05, 0.04, 1.0)):
        CL_t = trim_lift_coefficient(CL_target, Sref, x_cg, S_t, l_t)
        CL_w_S_w = CL_target * Sref - CL_t * S_t          # from the total
        cm = (x_cg * CL_w_S_w + (x_cg - l_t) * CL_t * S_t) / (Sref * mac)
        assert cm == pytest.approx(0.0, abs=1e-12)


def test_a_surface_with_no_area_or_no_arm_says_nothing():
    assert np.isnan(trim_lift_coefficient(0.5, 10.0, 0.25, 0.0, 5.5))
    assert np.isnan(trim_lift_coefficient(0.5, 10.0, 0.25, 1.75, 0.0))
    assert np.isnan(trim_lift_coefficient(0.5, 0.0, 0.25, 1.75, 5.5))


def test_the_wing_ac_offset_moves_the_denominator_too():
    """With the wing AC away from the origin BOTH the CG and the arm are
    measured from it — a version that only shifted the numerator would be
    wrong wherever the convention is not x_w = 0."""
    a = trim_lift_coefficient(0.5, 10.0, 0.35, 1.75, 5.6, x_w=0.1)
    b = trim_lift_coefficient(0.5, 10.0, 0.25, 1.75, 5.5)
    assert a == pytest.approx(b, rel=1e-12)


# --------------------------------------- 2. against what the solvers solve
@pytest.mark.parametrize("name,flags", [
    ("tail", None),
    ("tail", {"tail_type": "canard"}),
    ("tail", {"tail_type": "v_tail", "dihedral_deg": 35.0}),
    ("tail", {"tail_type": "t_tail"}),
    ("tail (fixed arm)", None),
    ("tail [free height]", None),
    ("tail + winglet", None),
    ("hydrofoil + elevator", None),
])
def test_the_estimate_is_the_load_the_solver_trims_to(name, flags):
    """Mid-box design, one real evaluation, and the surface's own solved
    lift coefficient. The residual is the numerical trapezoid area against
    the analytic one, not a modelling gap."""
    est = api.trim_surface_cl(name, flags=flags)
    assert est is not None
    built = api.PROBLEM_SPECS[name].build({}, flags or {}, None)
    x = np.array([0.5 * (a + b) for a, b in built.bounds])
    bd = built.evaluate(x)
    solved = bd.get("CL_t", bd.get("CL_stab"))
    assert solved is not None
    assert est["cl"] == pytest.approx(float(solved), rel=2e-3)
    # ...and it is nowhere near the zero it used to be screened at. The
    # threshold is 0.02 rather than 0.05 because the load is no longer the
    # CG offset alone: the sections' own couple is in the balance now
    # (tail.section_moment), the CG sits at a real 30 % MAC instead of the
    # 49.5 % it was calibrated to, and the two together put the published
    # aircraft tail at -0.038 — a DOWNLOAD, small and non-zero, which is
    # what a conventional stabiliser actually carries in cruise.
    assert abs(est["cl"]) > 0.02


def test_the_layout_moves_it_because_the_layout_moves_the_balance():
    """A canard trims from upstream (its CG is ahead of the wing AC) and a
    V-tail carries its share on cos^2 of its area: both change the load,
    and neither is a correction bolted on afterwards."""
    plain = api.trim_surface_cl("tail")["cl"]
    canard = api.trim_surface_cl("tail", flags={"tail_type": "canard"})["cl"]
    vee = api.trim_surface_cl("tail", flags={"tail_type": "v_tail",
                                             "dihedral_deg": 35.0})["cl"]
    # the aft layouts carry a DOWNLOAD and the canard carries LIFT: it sits
    # ahead of the CG, so it holds the nose UP where a tail holds it down.
    assert plain < 0.0 < canard
    # a V-tail carries its share on S cos^2 G, so the same balance is closed
    # by a smaller equivalent area — MORE load, in the same direction
    assert vee < plain < 0.0 and abs(vee) > abs(plain)
    # a bigger surface on the same arm carries less of it
    small = api.trim_surface_cl("tail", s_t_m2=1.0)["cl"]
    big = api.trim_surface_cl("tail", s_t_m2=3.0)["cl"]
    assert abs(small) > abs(big) > 0.0 and small < 0.0 and big < 0.0
    # and a longer arm needs less lift for the same moment
    assert abs(api.trim_surface_cl("tail", arm_m=8.0)["cl"]) < \
        abs(api.trim_surface_cl("tail", arm_m=4.0)["cl"])


def test_the_mission_moves_it_because_the_target_lift_is_the_missions():
    """A heavier aircraft trims at a higher CL and its tail follows — but
    AFFINELY, not proportionally, and that distinction is the section
    couple.

    CL_t = [CL_target Sref x_cg + M_ac] / (S_t l_t): the first term scales
    with the mission's load and the second does not, because a section's
    zero-lift moment is a property of its shape and not of how hard the
    aircraft is working. So what doubles when the weight doubles is the
    lift term alone — and on the published aircraft, where the couple is
    the bigger of the two, doubling the weight makes the download SMALLER
    rather than twice as large. That is a real, checkable prediction of
    carrying the couple, so it is gated as one."""
    base = api.trim_surface_cl("tail")
    heavier = api.trim_surface_cl("tail",
                                  mission_kwargs={"W_N": 2.0 * 652.8022140185})
    assert heavier["cl_target"] > base["cl_target"]
    denom = base["s_lift_m2"] * base["arm_m"]
    lift_term = base["cl_target"] * base["sref_m2"] * base["x_cg_m"] / denom
    couple_term = base["m_ac_m3"] / denom
    assert base["cl"] == pytest.approx(lift_term + couple_term, rel=1e-12)
    # only the lift term follows the mission; the couple stays put
    assert heavier["cl"] == pytest.approx(
        2.0 * lift_term + couple_term, rel=1e-9)
    # ...and on this aircraft the couple dominates at the design weight, so
    # a heavier one needs LESS download — far enough, it crosses over and
    # the surface starts lifting instead. Doubling the weight is already
    # past that crossing, which is the point: the two terms scale
    # differently, so the SIGN is a property of the operating point.
    assert couple_term < 0.0 and abs(couple_term) > lift_term > 0.0
    assert base["cl"] < 0.0 < heavier["cl"]
    assert heavier["cl"] - base["cl"] == pytest.approx(lift_term, rel=1e-9)


def test_a_family_with_no_trimming_surface_is_not_given_one():
    assert api.trim_surface_cl("trim wing") is None
    assert api.trim_surface_cl("hydrofoil") is None


def test_every_trimming_family_in_the_registry_can_say_what_it_trims_at():
    """The shell falls back to zero lift when a family will not answer. No
    family in the registry needs that fallback — and if a new one does, it
    fails here rather than silently screening a surface at a lift it never
    flies."""
    trimming = [n for n, s in api.PROBLEM_SPECS.items()
                if "S_t_m2" in s.default_bounds]
    assert trimming
    silent = [n for n in trimming if api.trim_surface_cl(n) is None]
    assert silent == []


# ------------------------------------------------- 3. through the V3 shell
def _tailed_session(medium: str = "air"):
    import gui.nice_app as v1
    from gui.v3 import session as ses

    S = ses.make_session(medium)
    S["wing"]["choices"]["tail"] = True
    v1.normalise_choices(S["wing"]["choices"], keep="tail")
    ses.apply_choices(S)
    S["mission"]["accepted"] = True
    return S


def _alpha_for_cl(pol, cl: float) -> float:
    """The incidence at which ``pol`` makes ``cl``, off its own valid range.

    A monotone read of the linear range, so it is the operating point and
    not a fit: the polars here are tables, and the inverse is only needed to
    ask WHERE on the curve a given lift sits.
    """
    lo, hi = pol.alpha_valid
    a = np.linspace(float(lo), float(hi), 2001)
    return float(np.interp(float(cl), np.asarray(pol.cl(a), dtype=float), a))


@pytest.mark.parametrize("medium", ["air", "water"])
def test_the_second_stage_screens_that_surface_at_its_trim_lift(medium):
    """The stage screens the surface at the point it ACTUALLY flies.

    The catalogue is stored and ranked UPRIGHT; the surface mounts its
    section the way :data:`gui.v3.config.TAIL_MOUNT` says, which is now
    inverted on every family and is NOT a function of the load. So the lift
    the stage screens at is that surface's load restated in the catalogue's
    frame, and the mirror states it exactly (``polar.InvertedPolar``:
    ``cl(a) = -cl0(-a)``, ``cd(a) = cd0(-a)``) — an inverted surface
    carrying ``cl`` IS the upright catalogue at ``-cl``, whichever sign
    ``cl`` has.

    That last clause is the whole test. Written as ``|cl|`` the two agree
    only while a down-mounted surface carries a DOWNLOAD, which was true of
    every family while the mounting FOLLOWED the load. It is not true now:
    the mount is pinned and the water elevator trims to an UP-load at its
    default CG (asserted below, so a moved default cannot silently drop the
    case), and there ``|cl|`` sends the screen to the other side of the
    polar — NACA 2412 at the published water design point: alpha +0.47 deg
    and cd 0.00553 where the elevator is really at -4.83 deg and cd 0.00833,
    a third of its profile drag missing from the ranking.

    So it is gated on the OPERATING POINT and the drag there, not on an
    expression: the point of screening at all is to rank the drag the
    surface makes.
    """
    from gui.v3 import session as ses

    S = _tailed_session(medium)
    assert ses.aft_surface(S)
    trim = ses.trim_lift(S)
    assert trim is not None and abs(trim["cl"]) > 0.02
    geo = ses.surface_geometry(S, "aft")
    screen_cl = geo["cl"]

    # the premise this medium is here to cover, stated out loud
    assert trim["inverted"] is True and geo["inverted"] is True, \
        "the shell pins the mounting (gui.v3.config.TAIL_MOUNT)"
    wanted = "a DOWN-load" if medium == "air" else "an UP-load"
    assert (trim["cl"] < 0.0 if medium == "air" else trim["cl"] > 0.0), (
        f"{medium} is here to cover an inverted surface carrying {wanted}; "
        f"its default has moved (cl {trim['cl']:+.4g}) and that case now "
        f"needs a new home")

    # the catalogue section, and the same section AS MOUNTED
    pol = _polar.default_polar()
    assert float(pol.cl(0.0)) > 0.05, \
        "a symmetric section is its own mirror and would prove nothing here"
    mounted = orient_section(pol, None, trim["inverted"])
    a_flown = _alpha_for_cl(mounted, trim["cl"])
    assert float(mounted.cl(a_flown)) == pytest.approx(trim["cl"], rel=1e-6)
    # ...and the lift the stage screens at puts the catalogue on that very
    # point: the same place on the curve, and so the same drag
    a_screen = _alpha_for_cl(pol, screen_cl)
    assert a_screen == pytest.approx(
        -a_flown if trim["inverted"] else a_flown, abs=1e-6)
    assert float(pol.cd(a_screen)) == pytest.approx(
        float(mounted.cd(a_flown)), rel=1e-9)
    # which is the mirror, in one line
    assert screen_cl == pytest.approx(
        -trim["cl"] if trim["inverted"] else trim["cl"])

    assert ses.surface_design_point(S, "aft")["cl_design"] == screen_cl
    S["ui"]["selected"] = ses.SURFACE_STAGES["aft"]
    assert ses.section_conditions(S)["cl_design"] == screen_cl
    # the wing's own stage is unaffected: it carries the mission's lift
    S["ui"]["selected"] = ses.SURFACE_STAGES["main"]
    assert ses.section_conditions(S)["cl_design"] == \
        pytest.approx(ses.design_point(S)["cl_design"])


def test_the_trim_lift_is_reported_with_the_balance_it_came_from():
    """A number this important may not appear from nowhere: the surface's
    geometry carries the balance that produced it, for the stage to show."""
    from gui.v3 import session as ses

    geo = ses.surface_geometry(_tailed_session(), "aft")
    assert geo is not None and not geo["lifting"]
    assert "trim balance" in geo["cl_source"]
    # BOTH numbers travel, because they are both true and they differ in
    # sign: the load the surface carries, and the lift its section sees the
    # way it is mounted (inverted, on a surface that pushes down)
    trim = ses.trim_lift(_tailed_session())
    assert geo["cl_flown"] == pytest.approx(trim["cl"])
    assert geo["inverted"] is bool(trim["inverted"])
    assert geo["cl"] == pytest.approx(
        -trim["cl"] if trim["inverted"] else trim["cl"])


def test_ranking_at_the_trim_lift_ranks_on_drag_again():
    """The point of the fix, at the screen itself: at zero lift "L/D at the
    design Cl" is 0 for every section, so 25 % of the composite ranks
    nothing; at the trim lift it is the drag that surface actually makes."""
    from gui.v3 import session as ses

    pt = api.screen_library_point()
    if not pt:
        pytest.skip("screen library not built")
    cl = ses.trim_lift(_tailed_session())["cl"]
    kw = dict(weights=dict(ses.TRIM_WEIGHTS), re=float(pt["re"]),
              mach=float(pt.get("mach", 0.0)), tc_min=0.08, cm_max=0.08,
              top_n=6, with_shape=False)
    dead = api.screen_airfoils(cl_design=0.0, **kw)
    alive = api.screen_airfoils(cl_design=cl, **kw)
    assert all(r["ldcr"] == 0.0 for r in dead["ranked"])
    assert all(r["ldcr"] > 1.0 for r in alive["ranked"])
    # and it is a different answer: the zero-lift table is ordered by the
    # criteria that survive there, camber first
    assert [r["name"] for r in alive["ranked"]] != \
        [r["name"] for r in dead["ranked"]]
    # the winner's drag AT the lift it flies is the better one, which is the
    # whole claim
    assert alive["ranked"][0]["cd_at"] < dead["ranked"][0]["cd_at"]
