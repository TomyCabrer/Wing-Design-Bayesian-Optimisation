"""Wing + tip device + tail in ONE nonplanar solve (wingtail.py).

The claim this module makes is that a winglet and a tail can be designed
together honestly, so the tests are about the things that would make that
claim false:

  1. TRIM is exact and affine — CL and Cm are linear in (alpha, i_t), so the
     closed-form 2x2 solve must reproduce CL_target and Cm = 0 to machine
     zero when re-evaluated at the answer.
  2. STABILITY is the same statement as the moment: Cm_alpha = -CL_alpha*SM
     must hold identically, or the reported static margin is decoration.
  3. It must be THE SAME AEROPLANE as tail.py, within a measured tolerance:
     a different aero core (Weissinger VLM vs coupled lifting lines) may
     disagree, but not by more than the difference between the models.
  4. Every switch must reduce to the thing it extends: no winglet, t/c at
     0.12, no chord law, arm fixed at its own value — each has to reproduce
     the simpler problem EXACTLY, so a combination cannot hide a change.
  5. The new freedoms must move the right things in the right direction (a
     higher tail sees less downwash; a span-capped device pays for its
     projection; a blend draws its own wake).
  6. The failure contract (-100, -1), never NaN.

Grid: N_vlm = 40 / N_t = 24 is converged — L/D moves 0.007 % and SM 1e-4
between it and N_vlm = 120 / N_t = 60 (measured 2026-07-26).
"""

import itertools

import numpy as np
import pytest

from aerobo import geometry, tail as T
from aerobo.polar import default_polar
from aerobo.vlm import VLM, TailSurface
from aerobo.wingtail import WingTailProblem, evaluate_wing_tail, fg_wing_tail

X_MID = np.array([0.6, 1.0, -2.0, 1.5, 5.0])     # taper, twists, S_t, l_t


# ------------------------------------------------------------ 1. trim/2. SM

def test_trim_is_exact_in_both_equations():
    prob = WingTailProblem()
    out = evaluate_wing_tail(X_MID, prob)
    assert out["feasible"], out["reason"]
    assert out["CL_total"] == pytest.approx(prob.CL_target, rel=1e-12)
    assert abs(out["Cm_cg"]) < 1e-12
    assert abs(out["cl_residual"]) < 1e-12


def test_cl_and_cm_are_affine_in_the_two_angles():
    """The whole trim rests on it: sample the map at four points and check
    the bilinear identity f(a,i) = f(0,0) + [f(a,0)-f(0,0)] + [f(0,i)-f(0,0)]."""
    model, mac = _model()
    x_cg = 0.25
    a, i = np.deg2rad(3.0), np.deg2rad(-2.0)
    f00 = model.solve(0.0, 0.0, x_cg=x_cg, mac=mac)
    fa0 = model.solve(a, 0.0, x_cg=x_cg, mac=mac)
    f0i = model.solve(0.0, i, x_cg=x_cg, mac=mac)
    fai = model.solve(a, i, x_cg=x_cg, mac=mac)
    for attr in ("CL", "Cm", "CDi"):
        lin = (getattr(fa0, attr) + getattr(f0i, attr) - getattr(f00, attr))
        if attr == "CDi":            # quadratic in the circulation — NOT affine
            assert getattr(fai, attr) != pytest.approx(lin, rel=1e-9)
        else:
            assert getattr(fai, attr) == pytest.approx(lin, rel=1e-11)


def test_cm_alpha_equals_minus_cl_alpha_times_static_margin():
    model, mac = _model()
    x_cg, d = 0.25, 1e-5
    Cma = (model.solve(d, 0.0, x_cg=x_cg, mac=mac).Cm
           - model.solve(0.0, 0.0, x_cg=x_cg, mac=mac).Cm) / d
    CLa = (model.solve(d, 0.0).CL - model.solve(0.0, 0.0).CL) / d
    SM = (model.neutral_point() - x_cg) / mac
    assert Cma == pytest.approx(-CLa * SM, rel=1e-9)


def _model(tail=True, **kw):
    pol = default_polar()
    wing = geometry.Wing(b=10.0, S=10.0, taper=0.6, twist_root_deg=1.0,
                         twist_tip_deg=-2.0)
    ts = TailSurface(S=1.5, x=5.0, z=0.5, AR=T.TAIL_AR, N=24,
                     a=pol.a_lin, alpha_L0=pol.alpha_L0) if tail else None
    return VLM(wing, N=40, a=pol.a_lin, alpha_L0=pol.alpha_L0, V=14.6,
               tail=ts, **kw), wing.mac


# --------------------------------------------------- 3. same aeroplane as tail.py

def test_it_is_the_same_aeroplane_as_the_lifting_line_tail():
    """MEASURED over the 108-point (taper, S_t, l_t, twist) box corners+mid
    grid: L/D agrees to within 2.21 % (mean +0.66 %), the static margin to
    within 0.072 (mean -0.026, the VLM always the lower — its tail lift-curve
    slope at AR 4 is below the lifting line's, which is where a Weissinger
    method is the better of the two), trim angles within 0.28 deg of alpha and
    0.73 deg of i_t, and the two agree on FEASIBILITY at every single point.

    The gates below are those measurements ROUNDED OUT, and the difference
    matters: an earlier version of this docstring quoted the rounded gate
    values (2.4 %, 0.8 %, 0.3 deg, 0.9 deg) as though they were the
    measurement, and the report then published them. Quote the numbers above,
    or re-derive them from this loop; never read a measurement off an
    assertion, which is a bound and not a value."""
    prob, ref = WingTailProblem(), T.TailProblem()
    dl, dsm, feas = [], [], 0
    for taper, S_t, l_t, twr, twt in itertools.product(
            (0.4, 0.7, 1.0), (0.6, 1.5, 2.8), (3.5, 5.5, 7.5),
            (0.0, 2.0), (-4.0, 0.0)):
        x = np.array([taper, twr, twt, S_t, l_t])
        a = evaluate_wing_tail(x, prob)
        b = T.evaluate_tail(x, ref)
        assert a["feasible"] == b["feasible"], x
        if not a["feasible"]:
            continue
        feas += 1
        dl.append((a["LoD"] - b["LoD"]) / b["LoD"])
        dsm.append(a["SM"] - b["SM"])
        assert abs(a["alpha_deg"] - b["alpha_deg"]) < 0.5
        assert abs(a["i_t_deg"] - b["i_t_deg"]) < 1.5
    assert feas == 108
    assert max(abs(np.array(dl))) < 0.03
    assert max(abs(np.array(dsm))) < 0.08
    assert abs(np.mean(dl)) < 0.02


def test_the_lift_split_uses_the_same_convention_as_tail_py():
    """Per-surface CL on its OWN area, and the share that makes them add up
    to the system CL. Getting this wrong reads as a tail carrying 5x its
    load."""
    out = evaluate_wing_tail(X_MID, WingTailProblem())
    ref = T.evaluate_tail(X_MID, T.TailProblem())
    assert out["CL_t"] == pytest.approx(ref["CL_t"], rel=0.05)
    assert out["lift_share_tail"] == pytest.approx(ref["lift_share_tail"],
                                                   rel=0.05)
    share = out["CL_t"] * out["S_lift"] / (out["CL_total"] * 10.0)
    assert out["lift_share_tail"] == pytest.approx(share, rel=1e-12)


# ------------------------------------------------------------ 4. reductions

def test_a_zero_height_winglet_is_exactly_the_no_winglet_problem():
    plain = evaluate_wing_tail(X_MID, WingTailProblem())
    withw = evaluate_wing_tail(np.append(X_MID, [0.0, 80.0]),
                               WingTailProblem(winglet=True))
    assert withw["LoD"] == pytest.approx(plain["LoD"], rel=1e-15)


def test_a_free_thickness_at_the_default_reproduces_the_fixed_polar():
    plain = evaluate_wing_tail(X_MID, WingTailProblem())
    free = evaluate_wing_tail(np.append(X_MID, 0.12),
                              WingTailProblem(tc_free=True))
    assert free["LoD"] == pytest.approx(plain["LoD"], rel=1e-12)


def test_a_zero_order_chord_law_changes_nothing():
    plain = evaluate_wing_tail(X_MID, WingTailProblem())
    law = evaluate_wing_tail(np.append(X_MID, [0.0, 0.0, 0.0]),
                             WingTailProblem(chord_order=3))
    assert law["LoD"] == pytest.approx(plain["LoD"], rel=1e-12)
    # ...and a real law does move the planform
    shaped = evaluate_wing_tail(np.append(X_MID, [0.3, -0.4, 0.2]),
                                WingTailProblem(chord_order=3))
    assert shaped["chord_dev"] > 0.05
    assert shaped["LoD"] != plain["LoD"]


def test_a_fixed_arm_at_the_same_distance_is_the_same_design():
    free = evaluate_wing_tail(X_MID, WingTailProblem())
    fixed = evaluate_wing_tail(X_MID[:4], WingTailProblem(l_t_fixed=5.0))
    assert fixed["LoD"] == pytest.approx(free["LoD"], rel=1e-15)
    assert fixed["l_t"] == pytest.approx(5.0)


def test_the_default_height_is_the_measured_regularisation_floor():
    free = evaluate_wing_tail(np.append(X_MID, 0.5),
                              WingTailProblem(free_height=True))
    fixed = evaluate_wing_tail(X_MID, WingTailProblem())
    assert fixed["dz_tail"] == pytest.approx(T.DZ_FRAC * 10.0)
    assert free["LoD"] == pytest.approx(fixed["LoD"], rel=1e-15)


# --------------------------------------------------------- 5. the freedoms

def test_a_higher_tail_sees_less_downwash_and_stabilises_more():
    """The inviscid part of the T-tail argument, now as a design variable:
    out of the wing's sheet the tail works harder, so the neutral point
    moves aft and the trim incidence has to grow to hold Cm = 0."""
    prob = WingTailProblem(free_height=True)
    low = evaluate_wing_tail(np.append(X_MID, 0.5), prob)
    high = evaluate_wing_tail(np.append(X_MID, 2.5), prob)
    assert high["SM"] > low["SM"]
    assert high["x_np"] > low["x_np"]
    assert abs(high["i_t_deg"]) > abs(low["i_t_deg"])


def test_the_tail_height_box_scales_with_the_span_and_floors_at_the_sheet():
    prob = WingTailProblem(free_height=True)
    lo, hi = prob.z_t_bounds
    assert (lo, hi) == pytest.approx((T.DZ_FRAC * 10.0, 0.30 * 10.0))
    big = WingTailProblem(free_height=True, b=20.0, S=40.0)
    assert big.z_t_bounds == pytest.approx((T.DZ_FRAC * 20.0, 6.0))
    # inside the sheet is refused rather than reported
    out = evaluate_wing_tail(np.append(X_MID, 0.2),
                             WingTailProblem(free_height=True))
    assert not out["feasible"]


def test_a_t_tail_cannot_also_have_a_free_height():
    with pytest.raises(ValueError, match="T-tail"):
        WingTailProblem(free_height=True, tail_type="t_tail")


def test_the_winglet_helps_and_the_span_cap_charges_its_projection():
    free = WingTailProblem(winglet=True)
    capped = WingTailProblem(winglet=True, capped=True)
    x = np.append(X_MID, [0.12, 75.0])
    none = evaluate_wing_tail(np.append(X_MID, [0.0, 75.0]), free)
    on = evaluate_wing_tail(x, free)
    cap = evaluate_wing_tail(x, capped)
    assert on["e"] > none["e"] and on["LoD"] > none["LoD"]
    # the capped wing is SHORTER by twice the device's true reach
    h = 0.12 * 10.0 / 2.0
    assert cap["b_wing"] == pytest.approx(
        10.0 - 2.0 * geometry.winglet_projection(h, 75.0), rel=1e-12)
    assert cap["LoD"] < on["LoD"]      # it pays for the span it borrowed


def test_the_blended_transition_and_its_shape_reach_the_wake():
    prob = WingTailProblem(winglet=True, capped=True, blended=True,
                           junction_drag=True)
    smooth = WingTailProblem(winglet=True, capped=True, blended=True,
                             junction_drag=True, blend_shape="smooth")
    x = np.append(X_MID, [0.12, 75.0, 0.8])
    sharp = evaluate_wing_tail(np.append(X_MID, [0.12, 75.0, 0.0]), prob)
    arc = evaluate_wing_tail(x, prob)
    smo = evaluate_wing_tail(x, smooth)
    assert sharp["CD_junction"] > arc["CD_junction"]     # the fillet is paid for
    assert arc["LoD"] != smo["LoD"]                      # different wake
    assert smo["winglet"]["blend_shape"] == "smooth"


def test_the_layouts_are_tail_py_s_and_not_restated():
    x = X_MID
    canard = evaluate_wing_tail(x, WingTailProblem(tail_type="canard"))
    assert canard["l_t"] == pytest.approx(-5.0)          # upstream
    assert canard["x_cg"] == pytest.approx(T.X_CG_BY_TYPE["canard"])
    v = evaluate_wing_tail(x, WingTailProblem(tail_type="v_tail",
                                              dihedral_deg=35.0))
    assert v["S_lift"] == pytest.approx(1.5 * np.cos(np.deg2rad(35.0)) ** 2)
    t = evaluate_wing_tail(x, WingTailProblem(tail_type="t_tail"))
    assert t["dz_tail"] == pytest.approx(
        T.tail_height("t_tail", 5.0, 10.0, 10.0))
    # the fin's parasite drag is charged always (no switch), and a V-tail —
    # which has no fin — is the only layout that pays none
    fin = evaluate_wing_tail(x, WingTailProblem())
    assert fin["cd0_fin"] > 0.0
    assert v["cd0_fin"] == 0.0


def test_an_elevator_flies_the_same_design_with_a_bigger_deflection():
    plain = evaluate_wing_tail(X_MID, WingTailProblem())
    elev = evaluate_wing_tail(X_MID, WingTailProblem(control="elevator"))
    assert elev["LoD"] == pytest.approx(plain["LoD"], rel=1e-15)
    assert elev["tau_elevator"] == pytest.approx(0.661, abs=1e-3)
    assert abs(elev["delta_e_deg"]) > abs(elev["i_t_deg"])


def test_a_resized_wing_is_flown_and_the_reference_follows():
    prob = WingTailProblem(b=12.0, S=12.0)
    out = evaluate_wing_tail(X_MID, prob)
    assert out["b_wing"] == pytest.approx(12.0)
    assert out["AR"] == pytest.approx(12.0, rel=1e-3)


# ------------------------------------------------------- 6. failure contract

def test_the_failure_contract_is_the_package_s():
    prob = WingTailProblem()
    assert fg_wing_tail(np.array([0.6, 1.0, -2.0, 1.5, 99.0]), prob) == \
        (-100.0, -1.0)                                   # outside the box
    assert fg_wing_tail(np.array([0.6, 1.0]), prob) == (-100.0, -1.0)
    f, g = fg_wing_tail(X_MID, prob)
    assert np.isfinite(f) and np.isfinite(g)
    # a solvable-but-unstable design keeps its TRUE L/D and a signed margin
    unstable = evaluate_wing_tail(np.array([0.6, 1.0, -2.0, 0.5, 3.0]), prob)
    assert unstable["feasible"] and unstable["g"] < 0.0
    assert unstable["LoD"] > 0.0


def test_a_collapsed_chord_law_is_a_penalty_not_an_exception():
    prob = WingTailProblem(chord_order=3, chord_max_frac=2.0)
    out = evaluate_wing_tail(np.append(X_MID, [-2.0, 0.0, 0.0]), prob)
    assert not out["feasible"] and "planform" in out["reason"]


def test_the_design_vector_layout_is_stable_under_every_combination():
    """Each block keeps its position whatever else is switched on — that is
    what lets a ProblemSpec carry a STATIC label list per combination."""
    for winglet, blended, tc_free, free_h, order in itertools.product(
            (False, True), (False, True), (False, True), (False, True),
            (0, 3)):
        if blended and not winglet:
            continue
        prob = WingTailProblem(winglet=winglet, blended=blended,
                               tc_free=tc_free, free_height=free_h,
                               chord_order=order)
        labels = prob.param_labels
        assert len(labels) == prob.dim
        assert labels[:4] == ("taper", "twist_root_deg", "twist_tip_deg",
                              "S_t_m2")
        assert labels[4] == "l_t_m"
        if order:
            assert labels[-order:] == geometry.chord_labels(order)
        x = np.array([0.5 * (lo + hi) for lo, hi in prob.bounds])
        assert evaluate_wing_tail(x, prob)["feasible"]
