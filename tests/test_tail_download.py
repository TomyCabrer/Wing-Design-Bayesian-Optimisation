"""What a conventional tail actually carries — and why it used to be wrong.

The stabiliser of every tailed family in this package came out LIFTING. Two
things put it there, and both are quotable:

1. the pitch balance summed lift times arm and nothing else, so the wing's
   own nose-down section couple — the term that makes a real tailplane
   push DOWN — was simply absent (``tail.py`` said so: "Section cm_ac is
   omitted from Cm (lift moments only, per the phase spec)"). The helper
   that reads it, ``polar.section_cm_ac``, existed and had no callers;

2. ``X_CG_BY_TYPE`` put the CG 0.25 m aft of the wing AC — 49.5 % MAC,
   aft of any real aft-CG limit — because it had been calibrated for
   CONSTRAINT ACTIVITY (make the SM = SM_min boundary cross the design box)
   rather than from a loading. And with the couple missing,
   sign(CL_t) = sign(x_cg) exactly, so that choice alone set the sign.

Both are fixed: the couple is carried (``tail.section_moment``, and
``vlm.VLM.solve``'s ``cm_ac`` for the two lattice families) and the CG sits
at 30 % MAC. This file gates the five things that has to keep true:

1. the couple is the section's, computed as cm_ac S mac, and zero for a
   symmetric section — which reproduces the pre-2026-08-01 numbers exactly;
2. it is a CONSTANT: it moves the trim load, and does NOT move the neutral
   point or the static margin;
3. the published aft-tail layouts carry a DOWNLOAD and the canard LIFTS,
   in all three solvers, closed form agreeing with each;
4. the CG's calibration duty survives its move (the SM boundary still
   crosses the design box, and every corner still trims);
5. the aerofoil screen ranks a NEGATIVE design lift the right way up.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aerobo import api, hydrotail, polar, tail, wingtail       # noqa: E402


def _mid(prob):
    b = prob.bounds
    return 0.5 * (b[:, 0] + b[:, 1])


class _Symmetric:
    """A polar with no camber: cm = 0 at every alpha, alpha_L0 = 0."""

    a_lin = 2.0 * np.pi
    alpha_L0 = 0.0
    alpha_valid = (-10.0, 14.0)
    tc = 0.12

    def cm(self, alpha_deg):
        return np.zeros_like(np.asarray(alpha_deg, dtype=float))

    def cl(self, alpha_deg):
        return self.a_lin * np.deg2rad(np.asarray(alpha_deg, dtype=float))

    def cd(self, alpha_deg):
        return np.full_like(np.asarray(alpha_deg, dtype=float), 0.008)


# --------------------------------------------------- 1. what the couple is
def test_the_couple_is_the_sections_own_and_it_is_cm_ac_times_S_mac():
    """M/q = cm_ac S mac per surface, MAC being (1/S) integral c^2 dy — so
    the identity is exact on the solved planform, not a mean-chord
    approximation."""
    prob = tail.TailProblem()
    cfg = tail._config(_mid(prob), prob)
    pol, pol_t = prob.polar, tail.tail_polar(prob)
    got = tail.section_moment(cfg, pol, pol_t)

    cm_w = polar.section_cm_ac(pol)
    cm_t = polar.section_cm_ac(pol_t)
    S_w = float(np.trapezoid(cfg.c_w, cfg.y_w))
    mac_w = float(np.trapezoid(cfg.c_w ** 2, cfg.y_w)) / S_w
    S_t = float(np.trapezoid(cfg.c_t, cfg.y_t))
    mac_t = float(np.trapezoid(cfg.c_t ** 2, cfg.y_t)) / S_t
    assert got == pytest.approx(cm_w * S_w * mac_w + cm_t * S_t * mac_t,
                                rel=1e-12)
    assert cm_w < 0.0                     # NACA 2412: nose-down, as it must
    assert got < 0.0


def test_a_symmetric_section_carries_no_couple_and_restores_the_old_numbers():
    """The back-compat path is not a flag: it is the physics. With cm_ac = 0
    the balance is lift-times-arm again, so a symmetric aircraft at the OLD
    CG reproduces the pre-2026-08-01 published load exactly."""
    sym = _Symmetric()
    prob = tail.TailProblem(polar=sym, polar_tail=sym, x_cg=0.25)
    cfg = tail._config(_mid(prob), prob)
    assert tail.section_moment(cfg, sym, sym) == 0.0

    # CL_t = CL_target Sref x_cg / (S_t l_t) — the pre-couple closed form
    assert tail.trim_lift_coefficient(0.5, 10.0, 0.25, 1.75, 5.5) == \
        pytest.approx(0.5 * 10.0 * 0.25 / (1.75 * 5.5), rel=1e-12)
    out = tail.evaluate_tail(_mid(prob), prob)
    assert out["M_ac_m3"] == 0.0 and out["Cm_ac"] == 0.0
    assert out["CL_t"] > 0.0             # a symmetric wing at an aft CG DOES
    assert out["CL_t"] == pytest.approx(0.5 * 10.0 * 0.25 / (1.75 * 5.5),
                                        rel=2e-3)


# ------------------------------------------- 2. a constant moves what it may
def test_the_couple_moves_the_load_and_not_the_neutral_point():
    """It is constant in (alpha, i_t), so the Jacobian of the trim map — and
    therefore x_np, SM and Cm_alpha — cannot see it. Only the constant of
    the map moves, which is the load."""
    # the ORIENTATION is pinned on both sides: it follows the load, the load
    # is what the couple moves, and this test is about the couple alone
    prob = tail.TailProblem(tail_inverted=False)
    x = _mid(prob)
    with_couple = tail.evaluate_tail(x, prob)

    real = tail.section_moment
    try:
        tail.section_moment = lambda *a, **k: 0.0
        without = tail.evaluate_tail(x, prob)
    finally:
        tail.section_moment = real

    assert with_couple["x_np"] == pytest.approx(without["x_np"], rel=1e-12)
    assert with_couple["SM"] == pytest.approx(without["SM"], rel=1e-12)
    assert with_couple["CL_alpha"] == pytest.approx(without["CL_alpha"],
                                                    rel=1e-12)
    # ...and it moves the load DOWN, by exactly the couple over the volume
    d_cl = with_couple["CL_t"] - without["CL_t"]
    cfg = tail._config(x, prob)
    assert d_cl == pytest.approx(
        tail.section_moment(cfg, prob.polar, tail.tail_polar(prob))
        / (cfg.S_lift * cfg.l_t), rel=2e-3)
    assert d_cl < 0.0


def test_the_trim_is_still_exactly_affine_with_the_couple_in():
    """The 2x2 elimination's affinity gate, re-asserted: a solve at the
    returned (alpha, i_t) reproduces CL and Cm to machine zero."""
    prob = tail.TailProblem()
    x = _mid(prob)
    out = tail.evaluate_tail(x, prob)
    CL, Cm, _ = tail.solve_tail_point(x, out["alpha_rad"], out["i_t_rad"],
                                      prob)
    assert CL == pytest.approx(prob.CL_target, abs=1e-10)
    assert Cm == pytest.approx(0.0, abs=1e-10)


# ------------------------------------------------- 3. the sign, everywhere
@pytest.mark.parametrize("tail_type,kw", [
    ("conventional", {}),
    ("t_tail", {}),
    ("v_tail", {"dihedral_deg": 35.0}),
])
def test_every_aft_layout_carries_a_download(tail_type, kw):
    prob = tail.TailProblem(tail_type=tail_type, **kw)
    out = tail.evaluate_tail(_mid(prob), prob)
    assert out["CL_t"] < 0.0, f"{tail_type} still lifts"
    assert out["Cm_cg"] == pytest.approx(0.0, abs=1e-12)
    # the CG is a real loading: 30 % MAC, the wing AC being at 25 %
    assert prob.x_cg == pytest.approx(0.05)
    assert 0.25 < 0.25 + prob.x_cg / out["mac"] < 0.35


def test_a_canard_lifts_because_it_is_ahead_of_the_cg():
    """Not a bug and not an exception: a canard holds the nose UP, so the
    same balance that downloads a tail lifts a canard."""
    prob = tail.TailProblem(tail_type="canard")
    out = tail.evaluate_tail(_mid(prob), prob)
    assert out["CL_t"] > 0.0
    assert prob.x_cg < 0.0                       # CG ahead of the wing AC


@pytest.mark.parametrize("name", ["tail", "tail (fixed arm)",
                                  "tail [free height]", "tail + winglet"])
def test_the_closed_form_and_the_solver_agree_on_the_download(name):
    """Both cores, one answer: the LLT pair (tail.py) and the nonplanar
    lattice (wingtail.py) trim to the load the closed form states."""
    est = api.trim_surface_cl(name)
    built = api.PROBLEM_SPECS[name].build({}, {}, None)
    x = np.array([0.5 * (a + b) for a, b in built.bounds])
    solved = built.evaluate(x)["CL_t"]
    assert est["cl"] < 0.0 and solved < 0.0
    assert est["cl"] == pytest.approx(float(solved), rel=2e-3)
    assert est["m_ac_m3"] < 0.0


def test_the_lattice_families_carry_the_couple_too():
    """wingtail.py and hydrotail.py hand it to the VLM, which cannot make it
    for itself: one chordwise panel per strip puts every bound vortex on the
    quarter-chord line, so the lattice yields lift-times-arm only."""
    prob = wingtail.WingTailProblem()
    out = wingtail.evaluate_wing_tail(_mid(prob), prob)
    assert out["CL_t"] < 0.0
    assert out["Cm_cg"] == pytest.approx(0.0, abs=1e-10)

    wp = hydrotail.HydrofoilTailProblem()
    wout = hydrotail.evaluate_hydrofoil_tail(_mid(wp), wp)
    assert wout["Cm_cg"] == pytest.approx(0.0, abs=1e-10)
    # the water craft's CG sits BETWEEN its foils (X_CG_FRAC of the arm), so
    # its stabiliser lifts — and the couple still moves it, downwards
    assert wout["CL_stab"] > 0.0
    assert wout["CL_stab"] < 0.30488          # the pre-couple published value


def test_the_zero_crossing_is_where_the_two_terms_cancel():
    """CL_t = 0 at x_cg = -M_ac/(CL_target Sref), not at the wing AC."""
    prob = tail.TailProblem()
    cfg = tail._config(_mid(prob), prob)
    M_ac = tail.section_moment(cfg, prob.polar, tail.tail_polar(prob))
    x_zero = -M_ac / (prob.CL_target * prob.S)
    assert x_zero == pytest.approx(0.123, abs=0.005)
    out = tail.evaluate_tail(_mid(prob), tail.TailProblem(x_cg=x_zero))
    assert out["CL_t"] == pytest.approx(0.0, abs=2e-3)
    # and the published CG is FORWARD of it, which is why the tail downloads
    assert tail.X_CG_BY_TYPE["conventional"] < x_zero


# ------------------------------------------- 4. the calibration duty holds
@pytest.mark.parametrize("tail_type,kw", [
    ("conventional", {}),
    ("t_tail", {}),
    ("canard", {}),
])
def test_the_sm_boundary_still_crosses_the_box_at_the_new_cg(tail_type, kw):
    """The reason the old CG existed. A CG that is realistic but leaves the
    stability constraint vacuous would trade one wrong answer for another,
    so the duty is re-gated here at the moved value."""
    prob = tail.TailProblem(tail_type=tail_type, **kw)
    x0 = _mid(prob)
    sms = []
    for taper, S_t, l_t in itertools.product((0.2, 1.0), (0.5, 3.0),
                                             (3.0, 8.0)):
        x = x0.copy()
        x[0], x[1], x[2], x[3], x[4] = taper, 0.0, 0.0, S_t, l_t
        out = tail.evaluate_tail(x, prob)
        if out.get("reason"):
            continue
        sms.append((out["x_np"] - out["x_cg"]) / out["mac"])
    assert sms, "no corner of the box solves"
    assert min(sms) < prob.SM_min < max(sms)


def test_every_corner_of_the_conventional_box_still_trims():
    """A forward CG asks the stabiliser for more, and past a point the box
    corners stop trimming at all (that is what x_cg = 0.00 does). At the
    calibrated 0.05 m every corner still solves."""
    prob = tail.TailProblem()
    x0 = _mid(prob)
    for taper, S_t, l_t in itertools.product((0.2, 1.0), (0.5, 3.0),
                                             (3.0, 8.0)):
        x = x0.copy()
        x[0], x[1], x[2], x[3], x[4] = taper, 0.0, 0.0, S_t, l_t
        out = tail.evaluate_tail(x, prob)
        assert not out.get("reason"), f"({taper}, {S_t}, {l_t}): {out}"
        assert out["CL_t"] < 0.0
        assert abs(out["i_t_deg"]) <= prob.i_t_max_deg


# --------------------------------------- 5. the screen, the right way up
def test_the_screen_ranks_a_negative_design_lift_the_right_way_up():
    """``ldcr`` is |cl|/cd. Signed, every candidate would score negative and
    the min-max normalisation would award 100 to the LEAST negative — i.e.
    elect the draggiest section in the database. The efficiency of a surface
    pushing down is |L|/D exactly as it is for one pushing up."""
    from aerobo.airfoil import AirfoilProblem
    from aerobo.airfoil_select import polar_metrics

    class _Pol:
        alpha_deg = np.arange(-6.0, 10.5, 0.5)
        cl = 0.11 * alpha_deg + 0.22
        cd = 0.008 + 0.004 * (cl - 0.2) ** 2
        cm = np.full_like(alpha_deg, -0.05)
        n_converged = alpha_deg.size

    pol = _Pol()
    coords = np.array([[1.0, 0.0], [0.5, 0.06], [0.0, 0.0],
                       [0.5, -0.06], [1.0, 0.0]])
    lo = polar_metrics(pol, coords, AirfoilProblem(cl_design=-0.20))
    hi = polar_metrics(pol, coords, AirfoilProblem(cl_design=+0.20))
    assert lo["ldcr"] > 0.0 and hi["ldcr"] > 0.0
    assert lo["ldcr"] == pytest.approx(0.20 / lo["cd_at"], rel=1e-12)
    # a positive-lift screen is untouched by the change
    assert hi["ldcr"] == pytest.approx(0.20 / hi["cd_at"], rel=1e-12)
    # ...and (L/D)max stays the section's own, point-independent number,
    # because it is SHA-cached across screening points (airfoil_select)
    assert lo["ldmax"] == pytest.approx(hi["ldmax"], rel=1e-12)


# ------------------------------- 6. and it flies its section UPSIDE DOWN
def test_the_mirror_is_exact_and_undoes_itself():
    """cl(a) = -cl0(-a), cd(a) = cd0(-a), cm(a) = -cm0(-a): reflecting the
    coordinates in the chord line maps the whole polar, so alpha_L0 and
    cm_ac flip, the slope and thickness do not, and the validity window
    mirrors — an inverted section stalls where the upright one stalls
    NEGATIVELY."""
    base = polar.default_polar()
    inv = polar.inverted(base)
    a = np.linspace(-5.0, 9.0, 29)
    assert np.allclose(inv.cl(a), -base.cl(-a))
    assert np.allclose(inv.cd(a), base.cd(-a))
    assert np.allclose(inv.cm(a), -base.cm(-a))
    assert inv.alpha_L0 == pytest.approx(-base.alpha_L0)
    assert inv.a_lin == pytest.approx(base.a_lin)
    assert inv.tc == pytest.approx(base.tc if hasattr(base, "tc") else 0.12)
    lo, hi = base.alpha_valid
    assert inv.alpha_valid == (-hi, -lo)
    assert polar.section_cm_ac(inv) == pytest.approx(
        -polar.section_cm_ac(base), rel=1e-9)
    assert polar.inverted(inv) is base            # un-inverts, no wrapping


def test_the_coordinates_mirror_with_the_polar():
    """Whatever draws or lofts the surface has to flip with it, or the
    picture contradicts the polar the answer was solved on."""
    xy = np.array([[1.0, 0.0], [0.5, 0.08], [0.0, 0.0],
                   [0.5, -0.04], [1.0, 0.0]])
    got = polar.invert_coords(xy)
    assert np.allclose(got[:, 0], xy[::-1, 0])    # same chordwise stations
    assert np.allclose(got[:, 1], -xy[::-1, 1])   # mirrored camber
    assert np.allclose(polar.invert_coords(got), xy)


def test_a_downloading_surface_flies_its_section_inverted():
    """The rule, and what it buys: the incidence stops fighting the camber.

    Forced upright is the pre-rule behaviour exactly, so the two orientations
    are comparable rather than merely different."""
    prob = tail.TailProblem()
    follow = tail.evaluate_tail(_mid(prob), prob)
    upright = tail.evaluate_tail(_mid(prob),
                                 tail.TailProblem(tail_inverted=False))
    forced = tail.evaluate_tail(_mid(prob),
                                tail.TailProblem(tail_inverted=True))

    assert follow["CL_t"] < 0.0                   # it downloads...
    assert follow["tail_section_inverted"] is True    # ...so it is mirrored
    assert follow["tail_section_follows_load"] is True
    assert upright["tail_section_inverted"] is False
    # the choice is the load's, so stating it reproduces the same numbers
    assert forced["LoD"] == pytest.approx(follow["LoD"], rel=1e-12)

    # what it buys, on the surface's OWN profile drag and its rigging angle
    assert follow["CDp_tail"] < upright["CDp_tail"]
    assert follow["LoD"] > upright["LoD"]
    assert abs(follow["i_t_deg"]) < 1.0 < abs(upright["i_t_deg"])
    assert follow["Cm_cg"] == pytest.approx(0.0, abs=1e-12)


def test_a_lifting_surface_is_left_alone():
    """A canard holds the nose UP, so there is nothing to mirror."""
    prob = tail.TailProblem(tail_type="canard")
    out = tail.evaluate_tail(_mid(prob), prob)
    assert out["CL_t"] > 0.0
    assert out["tail_section_inverted"] is False

    wp = hydrotail.HydrofoilTailProblem()
    wout = hydrotail.evaluate_hydrofoil_tail(_mid(wp), wp)
    assert wout["CL_stab"] > 0.0                  # the CG is between the foils
    # ...but the rule is asked, not assumed: a forward CG downloads it
    fwd = hydrotail.HydrofoilTailProblem(x_cg=-0.05)
    fout = hydrotail.evaluate_hydrofoil_tail(_mid(fwd), fwd)
    assert fout["CL_stab"] < 0.0 and not fout.get("reason")


def test_a_symmetric_section_cannot_be_mirrored_into_a_different_answer():
    """The rule is a no-op on a section that is its own mirror — which is
    what makes it safe to apply by default."""
    sym = _Symmetric()
    x = _mid(tail.TailProblem())
    up = tail.evaluate_tail(x, tail.TailProblem(polar=sym, polar_tail=sym,
                                                tail_inverted=False))
    inv = tail.evaluate_tail(x, tail.TailProblem(polar=sym, polar_tail=sym,
                                                 tail_inverted=True))
    assert up["LoD"] == pytest.approx(inv["LoD"], rel=1e-12)
    assert up["CL_t"] == pytest.approx(inv["CL_t"], rel=1e-12)


def test_all_three_cores_agree_on_the_mirrored_surface():
    """The closed form, the lifting-line pair and the nonplanar lattice."""
    est = api.trim_surface_cl("tail")
    assert est["inverted"] is True
    assert est["cl"] < 0.0 and est["cl_section"] == pytest.approx(-est["cl"])

    llt = tail.evaluate_tail(_mid(tail.TailProblem()), tail.TailProblem())
    assert est["cl"] == pytest.approx(llt["CL_t"], rel=2e-3)

    wt_prob = wingtail.WingTailProblem()
    wt = wingtail.evaluate_wing_tail(_mid(wt_prob), wt_prob)
    assert est["cl"] == pytest.approx(wt["CL_t"], rel=2e-3)
    assert wt["Cm_cg"] == pytest.approx(0.0, abs=1e-10)


def test_the_geometry_payload_tells_the_drawing_which_way_up():
    """A surface flown inverted has to be DRAWN inverted — the loft reads
    the flag off the same payload the caption does."""
    from aerobo import cad

    rep = api.design_report(api.RunConfig(problem_name="tail"),
                            [0.6, 0.0, -2.0, 1.75, 5.5])
    t = (rep.get("geometry") or {}).get("tail") or {}
    assert t.get("section_inverted") is True

    xc = np.array([1.0, 0.5, 0.0, 0.5, 1.0])
    zc = np.array([0.0, 0.09, 0.0, -0.03, 0.0])          # camber UP
    sf = {"y": [-1.0, 0.0, 1.0], "chord": [0.66, 0.66, 0.66],
          "x_offset": 5.5, "z_offset": 0.5}
    up = cad.tail_surfaces({**t, "section_inverted": False}, sf, xc, zc)
    dn = cad.tail_surfaces({**t, "section_inverted": True}, sf, xc, zc)
    assert up and dn
    # the mirror moves the thick side across the chord line and nothing else
    assert (up[0].Z.max() - up[0].Z.min()) == pytest.approx(
        dn[0].Z.max() - dn[0].Z.min(), rel=1e-9)
    assert (up[0].Z.max() - np.median(up[0].Z)) > \
        (dn[0].Z.max() - np.median(dn[0].Z))


def test_the_v3_shell_screens_the_second_surface_at_that_download():
    """End to end: the shell asks for the load the balance says, that load
    is negative, and the SECTION is screened the way up it is mounted.

    Those are two different numbers and both are right: the surface carries
    -0.025, and the section it flies — mounted upside down — sees +0.025.
    An upright catalogue read at |cl| IS the mirrored section read at cl.
    """
    import gui.nice_app as v1
    from gui.v3 import session as ses

    S = ses.make_session("air")
    S["wing"]["choices"]["tail"] = True
    v1.normalise_choices(S["wing"]["choices"], keep="tail")
    ses.apply_choices(S)
    S["mission"]["accepted"] = True
    trim = ses.trim_lift(S)
    assert trim is not None and trim["cl"] < 0.0 and trim["inverted"] is True
    geo = ses.surface_geometry(S, "aft")
    assert geo["inverted"] is True
    assert geo["cl_flown"] == pytest.approx(trim["cl"])
    assert geo["cl"] == pytest.approx(-trim["cl"])        # mounted-side lift
    assert ses.surface_design_point(S, "aft")["cl_design"] == geo["cl"]


@pytest.mark.parametrize("name", ["tail",
                                  "tail + winglet",
                                  "tail + winglet [designed tail + tip device]",
                                  "hydrofoil + elevator"])
def test_every_core_says_out_loud_which_way_up_it_flew_the_section(name):
    """Applying the mirror is not enough — the core has to REPORT it.

    Nothing downstream re-derives the orientation: ``api.design_report``
    copies ``tail_section_inverted`` into the geometry payload, and
    ``cad.tail_surfaces`` (which draws the 3-D view AND writes the STL and
    the OpenVSP script) flips the lofted section on that key alone. A core
    that mirrors the polar in SILENCE therefore flies one section and draws
    the other — the lattice families did exactly that, and the picture said
    "lifting" while the answer said "download".
    """
    spec = api.PROBLEM_SPECS[name]
    built = spec.build({}, {}, None)
    box = spec.default_bounds
    x = [0.5 * (box[k][0] + box[k][1]) for k in built.param_labels]
    raw = built.evaluate(np.asarray(x, dtype=float))

    assert isinstance(raw.get("tail_section_inverted"), (bool, np.bool_))
    cl = raw.get("CL_t", raw.get("CL_stab"))
    assert bool(raw["tail_section_inverted"]) is bool(cl < 0.0)
    # the couple that decided it is quotable in both of its units
    assert raw["Cm_ac"] == pytest.approx(
        raw["M_ac_m3"] / (built.problem.S * raw["mac"]), rel=1e-9)


@pytest.mark.parametrize("name", ["tail", "tail + winglet"])
def test_the_exported_tail_has_its_thick_side_where_the_load_is(name):
    """The loft that leaves this package, measured — not the flag."""
    from aerobo import cad

    spec = api.PROBLEM_SPECS[name]
    built = spec.build({}, {}, None)
    box = spec.default_bounds
    x = [0.5 * (box[k][0] + box[k][1]) for k in built.param_labels]
    rep = api.design_report(api.RunConfig(problem_name=name), x)
    geom = rep["geometry"]
    assert (geom.get("tail") or {}).get("section_inverted") is True

    panel = [s for s in cad.surfaces(geom, x, built.param_labels,
                                     rep.get("section_report"))
             if s.name == "tail"]
    assert panel, "no tail surface was exported"
    z = panel[0].Z
    mid = z[z.shape[0] // 2]
    mid = mid - mid.mean()
    # a downloading surface carries its camber on the side it is loaded on
    assert abs(mid.min()) > abs(mid.max())


def test_openvsp_is_handed_the_tail_section_the_way_up_it_flies():
    """The `.dat` the VSP script feeds to ``AIRFOIL_AFT``.

    The trap this gates is the FALLBACK: the generated script uses the
    wing's file when no aft one exists, so a tail that flies the wing's
    section — the published default — was built in VSP with the camber the
    wrong way and a VSPAERO case run on it. "The tail has no section of its
    own" and "the tail is mounted the same way up as the wing" are two
    different statements, and only the first was being asked.
    """
    import tempfile
    from pathlib import Path

    from aerobo import cad

    name = "tail + free chord law"
    spec = api.PROBLEM_SPECS[name]
    built = spec.build({}, {}, None)
    box = spec.default_bounds
    x = [0.5 * (box[k][0] + box[k][1]) for k in built.param_labels]
    rep = api.design_report(api.RunConfig(problem_name=name), x)
    geom = rep["geometry"]
    assert geom["tail"]["section_inverted"] is True
    # ...and it flies the WING's section: there is no aft section report
    files = cad.export(geom, Path(tempfile.mkdtemp()), stem="t", x_best=x,
                       labels=built.param_labels,
                       section=rep.get("section_report"))
    assert "dat_aft" in files, "the aft .dat is what stops the wing's being used"

    def _z(path):
        rows = [ln.split() for ln in Path(path).read_text().splitlines()[1:]
                if ln.strip()]
        return np.array([float(r[1]) for r in rows])

    wing, aft = _z(files["dat"]), _z(files["dat_aft"])
    assert abs(wing.max()) > abs(wing.min())          # wing camber UP
    assert abs(aft.min()) > abs(aft.max())            # tail camber DOWN
    # the exact mirror of the wing's, not some other section
    assert sorted(aft) == pytest.approx(sorted(-wing))


def test_the_geometry_box_draws_a_view_where_the_mounting_is_legible():
    """A true-scale 3-D view cannot show camber, so it does not count.

    The tail is ~0.08 m thick on a 10 m model, i.e. under 1 % of a frame
    drawn with ``aspectmode="data"`` — correct for span and arm, useless for
    "which way up is this section". The sections view normalises each
    surface on its OWN chord, off the same loft the STL is written from.
    """
    import gui.nice_app as v1

    name = "tail + free chord law"
    spec = api.PROBLEM_SPECS[name]
    built = spec.build({}, {}, None)
    box = spec.default_bounds
    x = [0.5 * (box[k][0] + box[k][1]) for k in built.param_labels]
    rep = api.design_report(api.RunConfig(problem_name=name), x)

    fig = v1.fig_sections_as_flown(rep["geometry"], x, built.param_labels,
                                   None, None, rep["breakdown"])
    assert fig is not None
    by = {str(t.name).split(" ·")[0]: t for t in fig.data}
    assert {"wing", "tail"} <= set(by)

    def _camber(tr):
        z = np.asarray(tr.y, float)
        return "up" if abs(z.max()) > abs(z.min()) else "down"

    assert _camber(by["wing"]) == "up"
    assert _camber(by["tail"]) == "down"
    # normalised on its OWN chord, so both are legible at the same zoom
    for tr in fig.data:
        xs = np.asarray(tr.x, float)
        assert xs.min() == pytest.approx(0.0, abs=1e-9)
        assert xs.max() == pytest.approx(1.0, abs=1e-9)
    # ...and the label says which way the load goes, so the picture and the
    # sign are read together rather than one being checked against the other
    assert "DOWN" in str(by["tail"].name)
    assert "(up)" in str(by["wing"].name)


def test_a_wing_with_a_tail_is_not_offered_a_tailless_aerofoil():
    """The screen must not hand a tailed aeroplane a REFLEXED section.

    |Cm| is a lower-better CRITERION, so ranking a wing on it rewards
    reflex — and reflex is what an aerofoil does INSTEAD of having a tail.
    Measured on the published wing screen: five of the top six were
    reflexed or near-zero, four of them Horten (hg*) and Hepperle (mh*)
    FLYING-WING sections. Under a stabiliser they invert its job.
    """
    from gui.v3 import session as ses
    import gui.nice_app as v1

    # the gate itself: -cm >= 0, i.e. nose-down only
    rep = api.screen_airfoils(weights="gdp-sweep", cl_design=0.5, top_n=6,
                              floors={api.SCREEN_NOSE_DOWN_KEY: 0.0})
    ranked = rep["ranked"]
    assert ranked, "the gate must not empty the eligible set"
    assert all(r["cm_at"] <= 0.0 for r in ranked)
    best = ranked[0]["name"]
    trim = api.trim_surface_cl("tail [designed tail] + free chord law",
                               flags={api.SECTION_KEY: best})
    assert trim["cl"] < 0.0 and trim["inverted"] is True   # a DOWNLOAD

    # ...and the shell asks for it exactly where it belongs: on a wing that
    # HAS a trimming surface, never on the trimming surface itself, never
    # on a tandem pair, and never on a lone wing
    S = ses.make_session("air")
    ch = S["wing"]["choices"]
    ch["tail_design"] = v1.tail_design_start(ch)
    ch["tail"] = True
    v1.normalise_choices(ch, keep="tail")
    ses.apply_choices(S)
    assert ses.surface_job(S, "main") == "wing-trimmed"
    assert ses.nose_down_required(S, "main") is True
    assert ses.nose_down_required(S, "aft") is False
    assert ses.recommended_weights(S, "main")[0]["cm"] == 0.0

    plain = ses.make_session("air")
    ses.apply_choices(plain)
    assert ses.surface_job(plain, "main") == "wing"
    assert ses.nose_down_required(plain, "main") is False

    # a DEFAULT, not a ban: a stated answer wins (repo rule)
    ses.airfoil_state(S, "main")["nose_down"] = False
    assert ses.nose_down_required(S, "main") is False


def test_a_reflexed_wing_section_turns_the_download_into_an_uploaD():
    """The stabiliser lifting is not always a bug — sometimes it is the
    aeroplane the wing's SECTION asked for.

    The couple that makes a tail push down is the WING's ``cm_ac``, and it
    is only nose-down for a normally cambered section. Screening a wing on
    |Cm| (weight 0.20, gated at 0.08) can select a REFLEXED one — ``hg40``,
    a Horten section, has cm_ac > 0 — and then the wing holds its own
    pitching moment, nothing is left for the tail to hold down, and the
    trim balance genuinely asks it for UP-load with an upright section.

    Gated because it looks exactly like the sign error this file exists to
    stop, and the shell now has to say WHY rather than leave it inferred.
    """
    from aerobo import polar as _p

    assert _p.section_cm_ac(api.library_section_polar("naca2412")) < 0.0
    assert _p.section_cm_ac(api.library_section_polar("hg40")) > 0.0

    name = "tail [designed tail] + free chord law"
    base = api.trim_surface_cl(name)
    assert base["m_ac_m3"] < 0.0 and base["cl"] < 0.0
    assert base["inverted"] is True

    flexed = api.trim_surface_cl(name, flags={api.SECTION_KEY: "hg40"})
    assert flexed["m_ac_m3"] > 0.0            # nose-UP couple
    assert flexed["cl"] > 0.0                 # so the tail LIFTS
    assert flexed["inverted"] is False        # ...and mounts upright


def test_a_shared_section_draws_ONE_outline_the_wings():
    """The wing's Shape panel draws the WING's aerofoil, and only that.

    The stabiliser with no section of its own still flies this one mirrored,
    and the hint under the panel still says so — but the mirrored outline is
    no longer overlaid as a second dashed curve. Asked for by the user; the
    panel answers one question (what does the wing fly?) in one place.

    This test exists to fail if that trace is ever put back: it asserts the
    figure is UNCHANGED whether or not a second surface shares the section.
    """
    import gui.nice_app as v1
    from gui.v3.stages.airfoil import shares_section_with

    rep = {"cl_design": 0.5,
           "design": {"name": "naca2412", "tc": 0.12, "tc_max_xc": 0.3,
                      "coords": [[1.0, 0.0], [0.5, 0.09], [0.0, 0.0],
                                 [0.5, -0.03], [1.0, 0.0]]}}
    base = v1.fig_section_shape(rep)
    assert base is not None
    n0 = len(base.data)

    # neither a sharing surface nor an empty one adds a trace
    assert len(shares_section_with(v1.fig_section_shape(rep), "").data) == n0
    fig = shares_section_with(v1.fig_section_shape(rep), "tail")
    assert len(fig.data) == n0
    assert not any("INVERTED" in str(getattr(t, "name", "") or "")
                   for t in fig.data)


def test_the_shape_and_the_polar_are_shown_in_the_same_frame():
    """Flipping the picture and leaving the curves upright is worse than
    flipping neither — the reader sees an aerofoil arching DOWN whose lift
    curve still climbs through positive cl, and reads the contradiction as a
    sign error. So the whole report changes frame together."""
    from gui.v3.stages.airfoil import mirror_section_report

    a = [-4.0, 0.0, 4.0, 8.0]
    rep = {"cl_design": 0.0249,
           "design": {"name": "hg40", "tc": 0.12, "tc_max_xc": 0.3,
                      "coords": [[1.0, 0.0], [0.5, 0.09], [0.0, 0.0],
                                 [0.5, -0.03], [1.0, 0.0]],
                      "polar": {"alpha_deg": a, "cl": [0.05, 0.25, 0.7, 1.1],
                                "cd": [0.009, 0.007, 0.008, 0.013],
                                "branch": [0, 4],
                                "cm": [-0.05, -0.05, -0.06, -0.07]}}}

    assert mirror_section_report(rep, False) is rep      # lifting: untouched

    m = mirror_section_report(rep, True)
    d, p = m["design"], m["design"]["polar"]
    assert m["cl_design"] == pytest.approx(-0.0249)      # DOWN, in the plot
    assert [z for _x, z in d["coords"]] == pytest.approx(
        [0.0, -0.09, 0.0, 0.03, 0.0])                    # camber DOWN
    assert [x for x, _z in d["coords"]] == [1.0, 0.5, 0.0, 0.5, 1.0]
    # cl(a) = -cl0(-a), cm(a) = -cm0(-a) ... and cd is EVEN in alpha
    assert p["alpha_deg"] == pytest.approx([4.0, 0.0, -4.0, -8.0])
    assert p["cl"] == pytest.approx([-0.05, -0.25, -0.7, -1.1])
    assert p["cm"] == pytest.approx([0.05, 0.05, 0.06, 0.07])
    assert p["cd"] == rep["design"]["polar"]["cd"]
    assert p["branch"] == [0, 4]                         # indices still index
    # the source report is not mutated — the ranking still reads it upright
    assert rep["cl_design"] == pytest.approx(0.0249)
    assert rep["design"]["polar"]["cl"][1] == pytest.approx(0.25)
    # and the figures built from it agree with each other
    import gui.nice_app as v1
    shape, polars = v1.fig_section_shape(m), v1.fig_section_polars(m)
    assert shape is not None and polars is not None
    z = np.asarray(shape.data[0].y, float)
    assert abs(z.min()) > abs(z.max())                   # picture: DOWN
    lift = polars.data[0]                                # curve: DOWN too
    assert min(np.asarray(lift.y, float)) < 0.0


def test_the_shell_previews_the_section_the_way_it_is_mounted():
    """Stage 2's shape preview obeys the same rule the loft does.

    Drawing the catalogue's stored shape next to a caption that says the
    surface pushes DOWN is what makes the whole thing read as a sign error.
    """
    import plotly.graph_objects as go

    from gui.v3.stages.airfoil import mirror_section_fig

    def _fig():
        f = go.Figure(go.Scatter(x=[1.0, 0.5, 0.0], y=[0.0, 0.09, 0.0]))
        f.add_annotation(x=0.3, y=0.0, text="max t/c", yshift=-28)
        f.update_layout(title_text="Section shape")
        return f

    up = _fig()
    assert mirror_section_fig(up, False) is up          # lifting: untouched
    assert list(up.data[0].y) == [0.0, 0.09, 0.0]

    dn = mirror_section_fig(_fig(), True)
    assert list(dn.data[0].y) == [0.0, -0.09, 0.0]      # camber DOWN
    assert list(dn.data[0].x) == [1.0, 0.5, 0.0]        # and nothing else
    assert dn.layout.annotations[0].yshift == 28
    assert "AS MOUNTED" in dn.layout.title.text
