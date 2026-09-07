"""Stability-derivative deck — dynamics.py.

Every test here asserts an OUTCOME the implementation could get wrong: a sign
worked out by hand, a closed form, or an identity the solver already satisfies
by another route. None of them restates the code.

The frame conversion gets its own block because it is the defect this module
was written around: the first working version flipped the output moment and
left the input rate in lattice axes, which produced Cl_p = +0.53 — roll
ANTI-damping. Nothing else in the deck looked wrong.
"""

import numpy as np
import pytest

from aerobo import dynamics as dyn
from aerobo.geometry import Wing
from aerobo.vlm import VLM, TailSurface

TWO_PI = 2.0 * np.pi


def _wing_tail(taper=0.5, N=60):
    return VLM(Wing(b=10.0, S=10.0, taper=taper), N=N, V=30.0,
               tail=TailSurface(S=2.0, x=4.0, z=0.5, AR=4.0, N=20))


def _plain(taper=1.0, N=80, **kw):
    return VLM(Wing(b=10.0, S=10.0, taper=taper), N=N, V=30.0, **kw)


# ---------------------------------------------------- riding the same solver

def test_alpha_column_is_the_lattice_own_lift_basis():
    """The deck's alpha column must be the lattice's own lift basis.

    The claim is asserted where it is decidable: the RIGHT-HAND SIDE the deck
    builds is bit-for-bit the one :class:`vlm.VLM` builds for itself
    (``-V * nrm[:, 2]``), which is pure numpy arithmetic. If the deck were
    riding a different boundary condition, that vector would differ.

    The resulting CIRCULATION is compared to machine precision instead,
    deliberately: ``lu_solve`` is LAPACK, and on a loaded machine the BLAS
    thread count — and with it the summation order — changes between calls.
    An exact-equality assertion here passed alone and failed under a full
    run three separate times in this module's history. That is a flaky test,
    not a discovered defect.
    """
    m = _wing_tail()
    r_cp = m.cp - np.array([0.3, 0.0, 0.0])
    u = dyn._disturbance_wind(m, "alpha", r_cp)
    rhs = -np.einsum("ij,ij->i", u, m.nrm)
    assert np.array_equal(rhs, -m.V * m.nrm[:, 2]), \
        "the deck is not writing the lattice's own alpha boundary condition"
    Gam = dyn.rhs_circulation(m, u)
    assert Gam == pytest.approx(m._Gam1, rel=1e-12, abs=1e-15)


def test_static_margin_agrees_with_the_neutral_point():
    """Cm_alpha = -CL_alpha * SM, with SM from VLM.neutral_point().

    Two independent routes to the same number: the deck integrates a moment,
    neutral_point() takes a lift-weighted station. They agree to machine
    precision or one of them is wrong.
    """
    m = _wing_tail()
    x_cg, mac = 0.3, 1.0
    D = dyn.deck(m, x_cg=x_cg, mac=mac, b=10.0)
    sm_vlm = (m.neutral_point() - x_cg) / mac
    assert D.static_margin == pytest.approx(sm_vlm, abs=1e-10)
    assert D.Cm_alpha == pytest.approx(-D.CL_alpha * sm_vlm, abs=1e-10)


def test_pitching_moment_reduces_to_the_lattice_own_cm():
    """The deck's three-axis moment must contain VLM._cm as its y component."""
    m = _wing_tail()
    x_cg, mac = 0.3, 1.0
    _F, M = dyn.panel_loads(m, m._Gam1, x_cg)
    q = 0.5 * m.V ** 2
    assert M.sum(axis=0)[1] / (q * m.S * mac) == pytest.approx(
        m._cm(m._Gam1, x_cg, mac), rel=1e-12)


def test_cl_alpha_matches_the_lattice_value():
    m = _wing_tail()
    D = dyn.deck(m, x_cg=0.3, mac=1.0, b=10.0)
    assert D.CL_alpha == pytest.approx(m._CLa, rel=1e-12)


# ------------------------------------------------------------ the sign rails

def test_roll_damping_is_negative_and_near_strip_theory():
    """Cl_p < 0, within 15 % of the strip-theory -a/12 for a rectangle.

    A wing that rolls must resist rolling. The closed form for a rectangular
    wing under strip theory is -a/12 = -0.5236 at a = 2*pi; the lattice adds
    the induced field, so agreement is close but not exact.
    """
    D = dyn.deck(_plain(taper=1.0), x_cg=0.0, mac=1.0, b=10.0)
    strip = -TWO_PI / 12.0
    assert D.Cl_p < 0.0
    assert abs(D.Cl_p - strip) / abs(strip) < 0.15


def test_roll_damping_falls_as_the_tip_chord_falls():
    """Taper removes area where the roll arm is longest, so |Cl_p| drops."""
    vals = [dyn.deck(_plain(taper=t), x_cg=0.0, mac=1.0, b=10.0).Cl_p
            for t in (1.0, 0.75, 0.5)]
    assert all(v < 0.0 for v in vals)
    assert vals[0] < vals[1] < vals[2]          # increasingly less negative


def test_pitch_damping_is_negative_and_needs_a_tail():
    """Cm_q < 0, and it is the TAIL that makes it big.

    An unswept wing alone with the CG on its quarter-chord line has every
    bound vortex at the same station, so it has no pitch arm at all.
    """
    D = dyn.deck(_wing_tail(), x_cg=0.3, mac=1.0, b=10.0)
    assert D.Cm_q < -5.0
    bare = dyn.deck(_plain(taper=0.5), x_cg=0.0, mac=1.0, b=10.0)
    assert abs(bare.Cm_q) < abs(D.Cm_q) / 100.0


def test_a_canted_tip_device_makes_side_force_and_dihedral_effect():
    """Winglets carry side load, so beta stops being invisible.

    Signs, worked out by hand: sideslip to starboard pushes the (vertical)
    device to port, so CY_beta < 0; the device is ABOVE the wing, so that side
    force rolls away from the sideslip and Cl_beta < 0 (dihedral effect).
    """
    m = _plain(taper=0.5, N=60, winglet_h_frac=0.12,
               winglet_cant_deg=90.0, n_winglet=10)
    D = dyn.deck(m, x_cg=0.3, mac=1.0, b=10.0)
    assert D.CY_beta < 0.0
    assert D.Cl_beta < 0.0


def test_tip_winglets_are_directionally_DEstabilising():
    """Cn_beta < 0 for a tip device ahead of the CG.

    This is not a defect to be fixed by tuning: a winglet sits on the wing's
    quarter-chord line, ahead of the CG, so its side force yaws the nose AWAY
    from the sideslip. It is the measurement that says a tip device is not a
    fin, and it is why V4 has to add a vertical surface aft rather than lean
    on the winglets it already has.
    """
    m = _plain(taper=0.5, N=60, winglet_h_frac=0.12,
               winglet_cant_deg=90.0, n_winglet=10)
    D = dyn.deck(m, x_cg=0.3, mac=1.0, b=10.0)
    assert D.Cn_beta < 0.0


# ------------------------------------------------------- the frame conversion

def test_axis_flip_is_a_proper_rotation():
    """det = +1, so moments and rates transform like positions.

    If this were a reflection, every pseudovector in the deck would need its
    own extra sign and the module's invariance argument would be false.
    """
    assert np.linalg.det(np.diag(dyn.AXIS_FLIP)) == pytest.approx(1.0)


@pytest.mark.parametrize("idx,name,flips", [
    (0, "CX / x-force", True),    # x aft -> x forward
    (1, "CY / y-force", False),   # starboard in both frames
    (2, "CZ / z-force", True),    # z up -> z down
])
def test_to_body_flips_exactly_x_and_z(idx, name, flips):
    """Hand-worked, component by component, for force AND moment alike."""
    e = np.zeros(3)
    e[idx] = 1.0
    CF, CM = dyn.to_body(e, e)
    expect = -1.0 if flips else 1.0
    assert CF[idx] == pytest.approx(expect), name
    assert CM[idx] == pytest.approx(expect), name


def test_rate_columns_flip_their_INPUT_as_well_as_their_output():
    """The double flip, asserted where it can actually fail.

    A body roll rate is a NEGATIVE lattice roll rate, so the wind field the
    deck builds for "p" must point the other way from the lattice's own
    +x rotation. Assert it on the wind field itself: the sign error that
    produced roll anti-damping lived here, and every downstream number stayed
    plausible while it did.
    """
    m = _plain(taper=1.0, N=20)
    r_cp = m.cp - np.array([0.0, 0.0, 0.0])
    u_p = dyn._disturbance_wind(m, "p", r_cp)
    # unit BODY roll rate = omega_lattice (-1, 0, 0); relative wind is
    # -(omega x r), so for a panel out at +y the wind is -(-x_hat x y r_y)
    # = +z_hat * r_y ... assert against the closed form directly
    omega_lat = np.array([-1.0, 0.0, 0.0])
    assert np.allclose(u_p, -np.cross(omega_lat, r_cp))
    # ...and it is NOT the un-flipped version
    assert not np.allclose(u_p, -np.cross(np.array([1.0, 0.0, 0.0]), r_cp))


def test_pitch_rate_does_not_flip():
    """y is starboard in both frames, so the q column is unconverted."""
    m = _plain(taper=1.0, N=20)
    r_cp = m.cp - np.array([0.0, 0.0, 0.0])
    u_q = dyn._disturbance_wind(m, "q", r_cp)
    assert np.allclose(u_q, -np.cross(np.array([0.0, 1.0, 0.0]), r_cp))


# ------------------------------------------------- what the deck refuses to fake

def test_a_wing_and_tail_have_no_yaw_stiffness_AND_say_why():
    """Cn_beta is exactly zero with no vertical surface, and it is EXPLAINED.

    Reporting a bare 0.0 would describe a neutrally stable aeroplane. The deck
    has to carry the reason, because a consumer that prints the number without
    it is the honesty defect this repo keeps meeting.
    """
    D = dyn.deck(_wing_tail(), x_cg=0.3, mac=1.0, b=10.0)
    assert D.Cn_beta == pytest.approx(0.0, abs=1e-12)
    assert "Cn_beta" in D.zeros
    assert "geometry" in D.zeros["Cn_beta"].lower()


# ------------------------------------- the local-velocity force (item 2)

def test_the_induced_velocity_half_factor_reproduces_the_trefftz_drag():
    """The measurement that licenses the whole force model.

    The induced velocity at the lifting line is taken as HALF the Trefftz
    wash, acting along the panel normal. If that factor is right, near-field
    induced drag built from it must equal the published Trefftz CDi — the
    classical statement that the two drag routes are one integral. Measured
    at a ratio of 1.000000 across taper and incidence, so it is asserted
    tightly rather than loosely.
    """
    for taper in (1.0, 0.5, 0.35):
        m = _plain(taper=taper, N=60)
        for a_deg in (2.0, 6.0):
            res = m.solve(np.deg2rad(a_deg))
            w = dyn.induced_velocity(m, res.Gamma)
            # the normal component, against the same quadrature CDi uses
            wn = np.einsum("ij,ij->i", w, m.nrm)
            CDi_near = -float((m.width * res.Gamma) @ wn) / (
                0.5 * m.V ** 2 * m.S)
            assert CDi_near == pytest.approx(res.CDi, rel=1e-10)


def test_roll_due_to_yaw_rate_matches_its_closed_form():
    """Cl_r = 4 * integral(y^2 Gamma dy) / (V S b^2), exactly.

    A yaw rate does not change any panel's incidence — the extra wind is
    in-plane — so it cannot change the circulation. All it does is speed one
    wing up and slow the other, and the rolling moment is that differential
    dynamic pressure against the loading that is already there. That makes
    a closed form available, and it agrees to a ratio of 1.00000.
    """
    m = _plain(taper=0.5, N=80)
    for a_deg in (2.0, 5.0, 8.0):
        D = dyn.deck(m, x_cg=0.0, mac=1.0, b=10.0, alpha=np.deg2rad(a_deg))
        Gam = m._gamma(np.deg2rad(a_deg), 0.0)
        closed = 4.0 * float(np.sum(m.y ** 2 * Gam * m.width)) / (
            m.V * m.S * 10.0 ** 2)
        assert D.Cl_r == pytest.approx(closed, rel=1e-9)


def test_cl_r_is_positive_and_within_a_factor_of_two_of_CL_over_4():
    m = _plain(taper=0.5, N=80)
    D = dyn.deck(m, x_cg=0.0, mac=1.0, b=10.0, alpha=np.deg2rad(6.0))
    CL = -D.const["CZ"]
    assert D.Cl_r > 0.0
    assert 0.5 <= D.Cl_r / (CL / 4.0) <= 2.0


def test_adverse_yaw_is_negative_and_within_a_factor_of_two_of_minus_CL_over_8():
    """Cn_p < 0 — a rolling wing yaws AWAY from the roll.

    This is induced drag being asymmetric: the down-going wing's lift vector
    tilts further back. The freestream-only force could not produce it at all.
    """
    m = _plain(taper=0.5, N=80)
    D = dyn.deck(m, x_cg=0.0, mac=1.0, b=10.0, alpha=np.deg2rad(6.0))
    CL = -D.const["CZ"]
    assert D.Cn_p < 0.0
    assert 0.5 <= D.Cn_p / (-CL / 8.0) <= 2.0


def test_both_are_exactly_proportional_to_the_base_lift():
    """The bilinear claim, asserted as a RATIO that must not drift.

    Base circulation times perturbation field, plus the reverse — so both
    derivatives are linear in the base CL and their ratio to it is a property
    of the planform alone.
    """
    m = _plain(taper=0.5, N=80)
    ratios = []
    for a_deg in (2.0, 4.0, 6.0, 8.0):
        D = dyn.deck(m, x_cg=0.0, mac=1.0, b=10.0, alpha=np.deg2rad(a_deg))
        CL = -D.const["CZ"]
        ratios.append((D.Cl_r / CL, D.Cn_p / CL))
    for i in (1, 2, 3):
        assert ratios[i][0] == pytest.approx(ratios[0][0], rel=1e-6)
        assert ratios[i][1] == pytest.approx(ratios[0][1], rel=1e-6)


def test_the_central_difference_is_exact_in_eps():
    """The force is QUADRATIC in the disturbance, so a central difference is
    exact and the answer must not depend on the step across four decades."""
    m = _plain(taper=0.5, N=60)
    vals = [dyn.deck(m, x_cg=0.0, mac=1.0, b=10.0,
                     alpha=np.deg2rad(5.0), eps=e).Cn_p
            for e in (1e-3, 1e-4, 1e-5, 1e-6, 1e-7)]
    for v in vals[1:]:
        assert v == pytest.approx(vals[0], rel=1e-9)


def test_an_unloaded_deck_still_has_no_cl_r_and_says_it_is_the_BASE_STATE():
    """At alpha = 0 both are zero — and the reason is now the state, not the
    force model. A yaw rate can only redistribute lift that already exists.

    This is the test that changed when item 2 landed: it used to assert the
    same zeros for the opposite reason.
    """
    D = dyn.deck(_wing_tail(), x_cg=0.3, mac=1.0, b=10.0, alpha=0.0)
    assert D.Cl_r == pytest.approx(0.0, abs=1e-12)
    assert D.Cn_p == pytest.approx(0.0, abs=1e-12)
    assert "unloaded" in D.zeros["Cl_r"].lower()
    assert "alpha" in D.zeros["Cn_p"].lower()


def test_the_freestream_force_is_still_reachable_and_still_says_so():
    """local_velocity=False must restore the old model exactly, and the
    explanation must change with it."""
    D = dyn.deck(_wing_tail(), x_cg=0.3, mac=1.0, b=10.0,
                 alpha=np.deg2rad(6.0), local_velocity=False)
    assert D.Cl_r == pytest.approx(0.0, abs=1e-12)
    assert "local_velocity=False" in D.zeros["Cl_r"]


def test_the_new_force_leaves_a_PLANAR_configuration_untouched():
    """Every derivative of item 1 is bit-unchanged when nothing has a
    vertical offset — because the only thing the new force adds to a planar
    wing is a streamwise force with no arm to act through."""
    m = _plain(taper=0.5, N=60)
    a = np.deg2rad(4.0)
    old = dyn.deck(m, x_cg=0.3, mac=1.0, b=10.0, alpha=a,
                   local_velocity=False)
    new = dyn.deck(m, x_cg=0.3, mac=1.0, b=10.0, alpha=a)
    for name in ("CL_alpha", "Cm_alpha", "Cl_p", "Cm_q", "Cn_r", "Cl_beta"):
        assert getattr(new, name) == pytest.approx(getattr(old, name),
                                                   rel=1e-9), name


def test_the_pitching_moment_shift_IS_a_drag_arm_and_scales_with_height():
    """On a non-planar layout Cm_alpha DOES move, by 8.8 % on the reference
    case, and that is not an error to be tolerated: induced drag acts at the
    tail's own height, and a streamwise force at a vertical offset makes a
    pitching moment the lift-only force had no way to produce. Asserted as a
    scaling law — zero at zero height, monotone in it.
    """
    shifts = []
    for z_t in (0.0, 0.25, 0.5, 1.0):
        m = VLM(Wing(b=10.0, S=10.0, taper=0.5), N=60, V=30.0,
                tail=TailSurface(S=2.0, x=4.0, z=z_t, AR=4.0, N=20))
        a = np.deg2rad(4.0)
        kw = dict(x_cg=0.3, mac=1.0, b=10.0, alpha=a)
        shifts.append(dyn.deck(m, **kw).Cm_alpha
                      - dyn.deck(m, local_velocity=False, **kw).Cm_alpha)
    # 1e-9, not 1e-12: a central difference at eps = 1e-5 amplifies rounding
    # by 5e4, so ~1e-12 is this measurement's own noise floor. The real
    # shifts below are ~1e-2, ten orders clear of it.
    assert shifts[0] == pytest.approx(0.0, abs=1e-9)    # no height, no arm
    assert all(s < 0.0 for s in shifts[1:])
    assert shifts[1] > shifts[2] > shifts[3]            # grows with height


def test_the_neutral_point_identity_belongs_to_the_LIFT_ONLY_force():
    """VLM.neutral_point() is a lift-weighted station, so it cannot carry the
    drag-arm term. The identity Cm_alpha = -CL_alpha * SM therefore holds
    against the freestream force and NOT against the local one on a
    non-planar layout. Both halves asserted, so neither can rot.
    """
    m = _wing_tail()
    x_cg, mac, a = 0.3, 1.0, np.deg2rad(6.0)
    sm = (m.neutral_point() - x_cg) / mac
    lift_only = dyn.deck(m, x_cg=x_cg, mac=mac, b=10.0, alpha=a,
                         local_velocity=False)
    assert lift_only.Cm_alpha == pytest.approx(
        -lift_only.CL_alpha * sm, abs=1e-10)
    local = dyn.deck(m, x_cg=x_cg, mac=mac, b=10.0, alpha=a)
    assert abs(local.Cm_alpha - (-local.CL_alpha * sm)) > 1e-3


def test_deck_is_affine_not_linear():
    """A cambered/twisted wing has a non-zero constant column."""
    m = VLM(Wing(b=10.0, S=10.0, taper=0.5, twist_root_deg=2.0,
                 twist_tip_deg=-3.0), N=60, V=30.0)
    D = dyn.deck(m, x_cg=0.0, mac=1.0, b=10.0)
    assert abs(D.const["CZ"]) > 1e-6            # lift at alpha = 0


def test_rhs_refuses_a_wind_field_of_the_wrong_shape():
    m = _plain(N=20)
    with pytest.raises(ValueError, match="one 3-vector per panel"):
        dyn.rhs_circulation(m, np.zeros((m.n_panels, 2)))


def test_unknown_disturbance_raises():
    m = _plain(N=20)
    with pytest.raises(ValueError, match="unknown disturbance"):
        dyn._disturbance_wind(m, "delta_a", m.cp)


def test_induced_tilt_is_OFF_by_default_and_cn_p_does_not_depend_on_it():
    """Cn_p comes from the ROLL-RATE FLOW TILT, which is exact.

    The distinction matters: the disturbance velocity is known in closed
    form, while the induced field's spanwise distribution is not trustworthy
    here. If Cn_p needed the induced term it would inherit that
    untrustworthiness. It does not — it moves by ~20 %, which is the size of
    the uncertainty, not the size of the effect.
    """
    m = _wing_tail()
    kw = dict(x_cg=0.3, mac=1.0, b=10.0, alpha=np.deg2rad(6.0))
    off = dyn.deck(m, **kw)
    on = dyn.deck(m, induced_tilt=True, **kw)
    assert off.Cn_p < 0.0 and on.Cn_p < 0.0
    assert abs(on.Cn_p - off.Cn_p) < 0.4 * abs(off.Cn_p)
    # ...and Cl_r does not use the induced field AT ALL
    assert on.Cl_r == pytest.approx(off.Cl_r, rel=1e-9)


def test_the_induced_angle_DISTRIBUTION_is_measured_unreliable():
    """The measurement behind induced_tilt's default.

    A rectangular wing's loading is close to elliptic, so its induced angle
    should be close to uniform. The lattice's half-Trefftz wash gives a
    spread of more than 50 % across the inner span, while its MEAN lands
    within 1 % of the closed form CL/(pi AR). Total right, distribution
    wrong — which is the near-field limitation vlm.py documents, and the
    reason adverse yaw is not claimed.
    """
    m = _plain(taper=1.0, N=60)
    a = np.deg2rad(5.0)
    G = m._gamma(a, 0.0)
    ai = -np.einsum("ij,ij->i", dyn.induced_velocity(m, G), m.nrm) / m.V
    inner = np.abs(m.y) < 0.85 * 5.0
    res = m.solve(a)
    assert ai[inner].mean() == pytest.approx(res.CL / (np.pi * m.AR),
                                             rel=0.02)          # mean: right
    spread = (ai[inner].max() - ai[inner].min()) / ai[inner].mean()
    assert spread > 0.5                                         # shape: wrong


def test_the_force_wind_is_evaluated_AT_THE_BOUND_SEGMENT():
    """With the CG on the bound line, CX_q = -alpha * CZ_q exactly.

    A pitch rate puts no velocity perturbation at a bound segment that sits
    ON the pitch axis, so the only streamwise force it can make is the
    forward lean of the extra lift it generates — which is alpha times that
    lift, in closed form. Evaluating the disturbance wind at the CONTROL
    POINT instead (which lies d = a c / 4pi aft) would put the segment off
    the axis and invent an extra term proportional to d times the base
    loading. The boundary condition belongs at the control point and the
    FORCE belongs at the bound segment; this is what pins the difference.
    """
    m = _plain(taper=0.5, N=60)
    a = np.deg2rad(5.0)
    D = dyn.deck(m, x_cg=float(m.x[0]), mac=1.0, b=10.0, alpha=a)
    assert D.columns["q"]["CX"] == pytest.approx(
        -a * D.columns["q"]["CZ"], rel=1e-6)


def test_induced_tilt_really_is_off_unless_asked_for():
    """The DEFAULT itself, not just that the flag has an effect."""
    m = _wing_tail()
    kw = dict(x_cg=0.3, mac=1.0, b=10.0, alpha=np.deg2rad(6.0))
    assert dyn.deck(m, **kw).Cn_p == pytest.approx(
        dyn.deck(m, induced_tilt=False, **kw).Cn_p, rel=1e-12)
    assert dyn.deck(m, **kw).Cn_p != pytest.approx(
        dyn.deck(m, induced_tilt=True, **kw).Cn_p, rel=1e-6)
