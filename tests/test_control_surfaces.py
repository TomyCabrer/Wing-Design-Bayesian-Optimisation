"""Control surfaces — dynamics.aileron / flap / elevator / rudder.

Each deflection is one more right-hand side through the lattice's existing LU
factors, exactly as the tail incidence already is. The tests below pin the
sign conventions (stated in the builders' docstrings), the closed form for the
section couple, and the two places where the model must be honest about what
it cannot produce.
"""

import numpy as np
import pytest

from aerobo import dynamics as dyn
from aerobo.geometry import Wing
from aerobo.tail import flap_effectiveness
from aerobo.vlm import VLM, TailSurface, VerticalSurface

X_CG = 0.3


def _model(fin=True, N=60):
    return VLM(Wing(b=10.0, S=10.0, taper=0.5), N=N, V=30.0,
               tail=TailSurface(S=2.0, x=4.0, z=0.5, AR=4.0, N=20),
               vertical=(VerticalSurface(height=1.2, chord=0.7, x=4.0,
                                         z_root=0.5, N=12) if fin else None))


def _deck(m, controls, x_cg=X_CG, alpha=0.0):
    return dyn.deck(m, x_cg=x_cg, mac=1.0, b=10.0, controls=controls,
                    alpha=alpha)


# ------------------------------------------------------------------ aileron

def test_positive_aileron_rolls_to_port():
    """Starboard trailing edge DOWN -> more lift to starboard -> roll to port.

    Body axes put positive roll at starboard-wing-down, so Cl_da < 0. This is
    the convention every other sign in the shell will be read against.
    """
    m = _model()
    assert _deck(m, [dyn.aileron(m)]).columns["aileron"]["Cl"] < -0.05


def test_aileron_makes_no_net_lift():
    """It is antisymmetric: what one wing gains the other loses."""
    m = _model()
    assert _deck(m, [dyn.aileron(m)]).columns["aileron"]["CZ"] == \
        pytest.approx(0.0, abs=1e-9)


def test_aileron_power_scales_with_the_flap_effectiveness():
    """Doubling tau doubles the roll power — the gain enters linearly.

    Asserted as a RATIO so it cannot be satisfied by any monotone response.
    """
    m = _model()
    a, b = 0.15, 0.40
    cl_a = _deck(m, [dyn.aileron(m, chord_frac=a)]).columns["aileron"]["Cl"]
    cl_b = _deck(m, [dyn.aileron(m, chord_frac=b)]).columns["aileron"]["Cl"]
    assert cl_b / cl_a == pytest.approx(
        flap_effectiveness(b) / flap_effectiveness(a), rel=1e-9)


def test_aileron_power_grows_with_span_reach():
    m = _model()
    short = _deck(m, [dyn.aileron(m, span_frac=(0.85, 0.98))])
    long = _deck(m, [dyn.aileron(m, span_frac=(0.50, 0.98))])
    assert long.columns["aileron"]["Cl"] < short.columns["aileron"]["Cl"] < 0


def test_a_deflected_aileron_yaws_PROVERSELY_and_that_is_a_known_limit():
    """The measured answer, and it is the WRONG one for a real aeroplane.

    A real aileron yaws adversely: the wing that gains lift gains more induced
    drag and falls back. This model cannot produce that, because the only yaw
    an aileron can make here is the body-axis forward lean of the extra lift
    vector at a positive incidence — which is proverse. The induced-drag
    asymmetry that would dominate needs the SPANWISE drag distribution, and
    that is exactly what one chordwise panel cannot resolve (measured: the
    induced angle varies 85 % across the inner span of a rectangular wing,
    whose loading is nearly elliptic and whose induced angle should therefore
    be nearly uniform).

    Pinned as measured, with its sign, so nobody reads roll-coupling out of
    this deck believing it is adverse yaw. Turning ``induced_tilt`` on
    reduces the magnitude by 43 % and does not flip it.
    """
    m = _model()
    col = _deck(m, [dyn.aileron(m)], alpha=np.deg2rad(6.0)).columns["aileron"]
    assert col["Cl"] < 0.0            # rolls to port
    assert col["Cn"] < 0.0            # ...and yaws that way too: PROVERSE
    tilt = dyn.deck(m, x_cg=X_CG, mac=1.0, b=10.0, alpha=np.deg2rad(6.0),
                    induced_tilt=True,
                    controls=[dyn.aileron(m)]).columns["aileron"]
    assert tilt["Cn"] < 0.0           # the induced term does not rescue it


def test_aileron_yaw_vanishes_on_an_unloaded_wing():
    """Whatever its sign, the coupling is proportional to lift: at zero
    incidence the lift vector has nothing to lean."""
    m = _model()
    col = _deck(m, [dyn.aileron(m)], alpha=0.0).columns["aileron"]
    assert col["Cl"] < 0.0            # the roll is still there
    assert abs(col["Cn"]) < 0.02 * abs(col["Cl"])


# --------------------------------------------------------------------- flap

def test_flap_raises_lift():
    m = _model()
    assert _deck(m, [dyn.flap(m)]).columns["flap"]["CZ"] < -0.5   # CZ is down


def test_flap_makes_no_roll():
    m = _model()
    assert _deck(m, [dyn.flap(m)]).columns["flap"]["Cl"] == \
        pytest.approx(0.0, abs=1e-9)


def test_the_section_couple_pushes_the_pitching_moment_NOSE_DOWN():
    """The couple's own contribution, isolated by turning it off.

    Whether a flap ends up nose-up or nose-down overall is a property of the
    CONFIGURATION — on this reference geometry the tail sits four chords aft,
    so the flap's downwash unloads it and that wins. What must always be true
    is the DIRECTION the couple pushes, so that is what is asserted; a test
    demanding a universally nose-down flap would be asserting a rule of thumb
    rather than the model.
    """
    m = _model()
    with_couple = _deck(m, [dyn.flap(m)]).columns["flap"]["Cm"]
    # the couple is the ONLY difference between these two
    Gam = dyn.control_column(m, dyn.flap(m))
    CF, CM = dyn.to_body(*dyn._coefficients(m, Gam, X_CG, 0.0, 10.0, 1.0))
    del CF
    assert with_couple < float(CM[1])


def test_hinge_couple_is_zero_at_both_ends_and_negative_between():
    """Closed form: a whole-chord flap is an incidence change and makes no
    couple about the quarter chord; no flap makes none either."""
    assert dyn.hinge_couple(1.0) == pytest.approx(0.0, abs=1e-12)
    # the vanishing-flap end approaches zero as sqrt(c_e/c), and the shared
    # 1e-6 clip with flap_effectiveness floors it at -2e-3 rather than 0
    assert dyn.hinge_couple(1e-9) == pytest.approx(0.0, abs=3e-3)
    assert abs(dyn.hinge_couple(1e-4)) < abs(dyn.hinge_couple(1e-2))
    for r in (0.1, 0.25, 0.3, 0.5, 0.75):
        assert dyn.hinge_couple(r) < -0.2


def test_hinge_couple_matches_the_thin_aerofoil_expression():
    for r in (0.15, 0.3, 0.62):
        th = np.arccos(2.0 * r - 1.0)
        assert dyn.hinge_couple(r) == pytest.approx(
            -0.5 * np.sin(th) * (1.0 - np.cos(th)), rel=1e-12)


# ----------------------------------------------------------------- elevator

def test_positive_elevator_pitches_nose_down():
    """Trailing edge down on a tail four chords aft: unambiguous."""
    m = _model()
    assert _deck(m, [dyn.elevator(m)]).columns["elevator"]["Cm"] < -1.0


def test_a_whole_chord_elevator_IS_the_lattice_all_moving_tail():
    """chord_frac = 1 reproduces VLM._CLi exactly.

    flap_effectiveness(1.0) == 1, so the hinged surface degenerates to the
    all-moving stabiliser the solver already trims with. Bit-level agreement
    is the check that the control channel and the trim channel are the same
    boundary condition.
    """
    m = _model()
    Gam = dyn.control_column(m, dyn.elevator(m, chord_frac=1.0))
    CL = 2.0 * float(Gam @ m.lvec[:, 1]) / (m.V * m.S)
    assert CL == pytest.approx(m._CLi, rel=1e-12)


# ------------------------------------------------------------------- rudder

def test_rudder_makes_side_force_and_yaw():
    m = _model()
    col = _deck(m, [dyn.rudder(m)]).columns["rudder"]
    assert col["CY"] < -0.05          # force to port...
    assert col["Cn"] > 0.02           # ...aft of the CG, so nose to starboard


def test_rudder_also_rolls_because_the_fin_is_above_the_cg():
    """A fin's side force acts high, so a rudder rolls as well as yaws."""
    m = _model()
    assert _deck(m, [dyn.rudder(m)]).columns["rudder"]["Cl"] < 0.0


def test_a_rudder_without_a_fin_is_all_zeros_not_an_error():
    """The honest answer to "what does the rudder do" on a finless aeroplane
    is nothing, reported as nothing — not an exception, and not a number."""
    m = _model(fin=False)
    r = dyn.rudder(m)
    assert r.n_panels == 0
    col = _deck(m, [r]).columns["rudder"]
    assert all(col[k] == pytest.approx(0.0, abs=1e-12) for k in col)


# --------------------------------------------------------- shared machinery

def test_tau_is_one_for_a_level_wing_panel_AND_for_a_fin_panel():
    """The reason a rudder needs no code path of its own.

    (t_hat x n) . x_hat is 1 for a wing panel (t = y, n = z) and 1 for a fin
    panel (t = z, n = -y). One formula, two surfaces.
    """
    m = _model()
    tau = dyn._tau_vector(m)
    assert np.allclose(tau[m.is_tail], 1.0, atol=1e-12)
    assert np.allclose(tau[m.is_vertical], 1.0, atol=1e-12)


def test_control_columns_do_not_disturb_the_disturbance_columns():
    """Adding controls must not change the alpha/beta/rate answers.

    To machine precision, NOT bit-for-bit. Two separate deck() calls sum the
    same panel loads through BLAS, and the thread count OpenBLAS picks
    depends on what else is running on the machine — under a loaded box the
    summation order changes and the last ulp with it. An exact-equality
    assertion here passed alone and failed in a full run, which is a flaky
    test rather than a discovered defect.
    """
    m = _model()
    bare = dyn.deck(m, x_cg=X_CG, mac=1.0, b=10.0)
    with_c = _deck(m, [dyn.aileron(m), dyn.rudder(m)])
    for k in dyn.DISTURBANCES:
        for comp, value in bare.columns[k].items():
            assert with_c.columns[k][comp] == pytest.approx(
                value, rel=1e-12, abs=1e-15), f"{k}/{comp}"
