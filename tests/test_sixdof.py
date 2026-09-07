"""Six-degree-of-freedom flight — sixdof.py.

The tests that matter here are the ones a wrong sign or a dropped term would
break: closed-form inertia, a trim that actually holds, energy conservation
with the dissipative terms off, and a stall that takes lift AWAY.
"""

from dataclasses import replace

import numpy as np
import pytest

from aerobo import dynamics as dyn, sixdof as sd
from aerobo.geometry import Wing
from aerobo.vlm import VLM, TailSurface, VerticalSurface


def _deck():
    m = VLM(Wing(b=10.0, S=10.0, taper=0.5), N=40, V=30.0,
            tail=TailSurface(S=2.0, x=4.0, z=0.5, AR=4.0, N=16),
            vertical=VerticalSurface(height=1.2, chord=0.7, x=4.0,
                                     z_root=0.5, N=10))
    return dyn.deck(m, x_cg=0.3, mac=1.0, b=10.0,
                    controls=[dyn.aileron(m), dyn.flap(m), dyn.elevator(m),
                              dyn.rudder(m)])


DECK = _deck()


def _ac(**kw):
    base = dict(deck=DECK, inertia=sd.Inertia.from_layout(1200.0, 10.0, 8.0),
                CD0=0.028, oswald_e=0.85)
    return sd.Aircraft(**(base | kw))


# ----------------------------------------------------------------- inertia

def test_wing_roll_inertia_is_the_uniform_bar_closed_form():
    """Ixx = m_wing b^2 / 12, checkable by hand."""
    I = sd.Inertia.from_layout(1000.0, b=12.0, length_m=8.0,
                               wing_mass_frac=0.25)
    assert I.Ixx == pytest.approx(250.0 * 144.0 / 12.0)


def test_doubling_the_span_quadruples_roll_inertia():
    a = sd.Inertia.from_layout(1000.0, b=6.0, length_m=8.0)
    b = sd.Inertia.from_layout(1000.0, b=12.0, length_m=8.0)
    assert b.Ixx / a.Ixx == pytest.approx(4.0)


def test_yaw_inertia_is_the_perpendicular_axis_sum():
    I = sd.Inertia.from_layout(1200.0, 10.0, 8.0)
    assert I.Izz == pytest.approx(I.Ixx + I.Iyy)


def test_the_inertia_says_where_it_came_from():
    """An estimate that cannot be traced is a hidden constant."""
    assert "wing bar" in sd.Inertia.from_layout(1200.0, 10.0, 8.0).basis
    assert "radii" in sd.Inertia.from_radii(1200.0, 10.0, 8.0).basis


@pytest.mark.parametrize("kw", [dict(mass_kg=0.0),
                                dict(wing_mass_frac=0.8, tail_mass_frac=0.5)])
def test_impossible_layouts_are_refused(kw):
    with pytest.raises(ValueError):
        sd.Inertia.from_layout(**(dict(mass_kg=1000.0, b=10.0,
                                       length_m=8.0) | kw))


def test_roll_mode_time_constant_is_seconds_not_milliseconds():
    """tau = -Ixx / (qbar S b^2 Cl_p / 2V) — the sanity check that the
    inertia and the deck are in compatible units at all."""
    ac = _ac()
    V, rho = 45.0, 1.225
    tau = -ac.inertia.Ixx / (0.5 * rho * V ** 2 * DECK.S * DECK.b ** 2
                             * DECK.Cl_p / (2 * V))
    assert 0.05 < tau < 5.0


# -------------------------------------------------------------------- trim

def test_trim_leaves_no_residual_acceleration():
    ac = _ac()
    st, trimmed = sd.trim_level(ac, V=45.0)
    d = sd.derivative(trimmed, st)
    assert np.linalg.norm(d[10:13]) < 1e-9        # moments balance exactly
    assert np.linalg.norm(d[7:10]) < 0.1          # forces to within 0.1 m/s2


def test_trim_needs_POSITIVE_thrust():
    """The defect this caught: balancing drag alone, and forgetting that
    weight leans aft along the body x-axis when the aircraft is pitched up,
    asked for -1136 N — an aeroplane towed backwards to stay level."""
    for V in (40.0, 45.0, 60.0):
        _st, t = sd.trim_level(_ac(), V=V)
        assert t.prop.thrust_n > 0.0


def test_trimmed_flight_holds_altitude_for_a_minute():
    st, t = sd.trim_level(_ac(), V=45.0, altitude_m=1000.0)
    hist = sd.integrate(t, st, dt=0.02, n=3000)
    drift = abs(hist[-1].altitude_m - 1000.0)
    assert drift < 10.0                            # < 1 % of 1000 m
    assert abs(hist[-1].V - 45.0) < 0.5


def test_trim_does_not_mutate_the_aircraft_it_was_given():
    ac = _ac()
    _st, t = sd.trim_level(ac, V=45.0)
    assert ac.prop.thrust_n == 0.0 and not ac.controls
    assert t.prop.thrust_n != 0.0 and "elevator" in t.controls


def test_trim_refuses_a_speed_it_cannot_fly():
    with pytest.raises(ValueError, match="cannot fly level"):
        sd.trim_level(_ac(), V=12.0)


def test_trim_refuses_an_unknown_control():
    with pytest.raises(ValueError, match="no control column"):
        sd.trim_level(_ac(), V=45.0, control="thrust_vector")


# ------------------------------------------------------------------- stall

def test_the_soft_clip_leaves_cruise_alone():
    """The tanh version removed 13 % of the lift at CL = 0.95 and made the
    aircraft untrimmable. The p-norm clip must cost under 1 % there."""
    s = sd.Stall(CL_max=1.4)
    assert s.lift(0.95, 0.0) == pytest.approx(0.95, rel=0.01)


def test_lift_never_exceeds_clmax():
    s = sd.Stall(CL_max=1.4)
    assert all(abs(s.lift(x, 0.0)) <= 1.4 + 1e-12
               for x in (2.0, 5.0, 50.0, -5.0))


def test_pulling_past_the_stall_DROPS_lift():
    """Not merely a ceiling — a stall a user can feel takes lift away."""
    ac = _ac()
    def CL(a_deg):
        a = np.deg2rad(a_deg)
        return ac.coefficients(
            sd.State(vel=45.0 * np.array([np.cos(a), 0.0, np.sin(a)])))["CL"]
    peak = CL(16.0)
    assert CL(13.0) < peak                        # still climbing before it
    assert CL(22.0) < peak                        # ...and falling after
    assert CL(30.0) < CL(22.0)


def test_separated_drag_rises_past_the_stall():
    ac = _ac()
    def CD(a_deg):
        a = np.deg2rad(a_deg)
        return ac.coefficients(
            sd.State(vel=45.0 * np.array([np.cos(a), 0.0, np.sin(a)])))["CD"]
    assert CD(25.0) > 5.0 * CD(10.0)


def test_the_stall_can_be_switched_off_for_mode_analysis():
    s = sd.Stall(enabled=False)
    assert s.lift(9.0, np.deg2rad(40.0)) == 9.0
    assert s.extra_drag(np.deg2rad(40.0)) == 0.0


# ---------------------------------------------------------------- kinematics

def test_quaternion_to_dcm_is_orthonormal():
    q = sd._quat_from_euler(0.3, -0.2, 1.1)
    R = sd.quat_to_dcm(q)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(R) == pytest.approx(1.0)


def test_euler_round_trip():
    for ang in [(0.0, 0.0, 0.0), (0.3, -0.2, 1.1), (-1.0, 0.4, -2.5)]:
        got = sd.dcm_to_euler(sd.quat_to_dcm(sd._quat_from_euler(*ang)))
        assert np.allclose(got, ang, atol=1e-9)


def test_gravity_is_straight_down_when_level_and_leans_aft_when_pitched_up():
    level = sd.body_axis_gravity(np.array([1.0, 0.0, 0.0, 0.0]))
    assert np.allclose(level, [0.0, 0.0, sd.G])
    up = sd.body_axis_gravity(sd._quat_from_euler(0.0, np.deg2rad(20.0), 0.0))
    assert up[0] < 0.0                # weight pulls BACKWARD along body x
    assert up[2] == pytest.approx(sd.G * np.cos(np.deg2rad(20.0)))


def test_alpha_and_beta_come_off_the_velocity():
    st = sd.State(vel=np.array([100.0, 10.0, 20.0]))
    assert st.alpha == pytest.approx(np.arctan2(20.0, 100.0))
    assert st.beta == pytest.approx(np.arcsin(10.0 / st.V))


def test_integration_renormalises_the_quaternion():
    st, t = sd.trim_level(_ac(), V=45.0)
    hist = sd.integrate(t, st, dt=0.02, n=500)
    assert np.linalg.norm(hist[-1].quat) == pytest.approx(1.0, abs=1e-12)


# --------------------------------------------------------------- the physics

def test_energy_is_conserved_with_thrust_and_drag_removed():
    """The integrator's own correctness, isolated from the aerodynamics.

    With no thrust, no drag and no induced drag, total energy must be
    constant: anything else is the RK4 or the equations of motion, not the
    model.
    """
    ac = _ac(CD0=0.0, oswald_e=1e9, stall=sd.Stall(enabled=False))
    st, t = sd.trim_level(ac, V=45.0, altitude_m=1000.0)
    t = replace(t, prop=sd.Propulsion(thrust_n=0.0))
    m = t.inertia.mass_kg
    def E(s):
        return 0.5 * m * s.V ** 2 + m * sd.G * s.altitude_m
    hist = sd.integrate(t, st, dt=0.01, n=2000)
    E0 = E(hist[0])
    assert max(abs(E(s) - E0) for s in hist) / E0 < 1e-4


def test_aileron_rolls_the_aircraft_to_port():
    """Sign continuity from the deck all the way to an attitude."""
    st, t = sd.trim_level(_ac(), V=45.0)
    t = replace(t, controls={**t.controls, "aileron": np.deg2rad(10.0)})
    hist = sd.integrate(t, st, dt=0.01, n=200)
    assert hist[-1].rates[0] < 0.0                 # roll rate to port
    assert hist[-1].euler[0] < 0.0                 # ...and it banks that way


def test_roll_rate_reaches_the_closed_form_steady_state():
    """p_ss = -Cl_da delta / Cl_p * (2V/b), within 10 %.

    The closed form ignores the fin's contribution to roll damping, which is
    why this is 10 % and not 1 %.
    """
    V = 45.0
    st, t = sd.trim_level(_ac(), V=V)
    d = np.deg2rad(10.0)
    t = replace(t, controls={**t.controls, "aileron": d})
    hist = sd.integrate(t, st, dt=0.01, n=300)
    p_ss = -DECK.columns["aileron"]["Cl"] * d / DECK.Cl_p * (2 * V / DECK.b)
    assert hist[-1].rates[0] == pytest.approx(p_ss, rel=0.10)


def test_a_thrust_line_below_the_cg_pitches_nose_up():
    """Throttle as a pitch input — a real handling property.

    Against the CLOSED FORM and against the FLOWN outcome, not against the
    implementation: this test used to assert ``M[1] < 0``, which is a
    restatement of a minus sign that was itself the bug. The moment of a
    force applied at a point is ``r x F`` and there is nothing to have an
    opinion about.
    """
    F, M = sd.Propulsion(thrust_n=1000.0, z_offset_m=0.5).force_moment()
    assert F[0] == 1000.0
    r = np.array([0.0, 0.0, 0.5])          # z is DOWN, so +z is BELOW the CG
    assert M == pytest.approx(np.cross(r, F))
    assert M[1] > 0.0                      # ...and +My takes +x toward -z: up

    # ...and the aeroplane must actually pitch UP when the throttle opens on
    # a low thrust line. This is the half a sign error survives on its own.
    st, t = sd.trim_level(_ac(), V=45.0)
    hot = replace(t, prop=replace(t.prop, z_offset_m=1.0,
                                  thrust_n=1.5 * t.prop.thrust_n))
    flown = sd.integrate(hot, st, dt=0.005, n=400)[-1]
    assert flown.euler[1] > st.euler[1] + np.deg2rad(1.0)


def test_pitch_damping_actually_damps():
    """Release the aircraft with a pitch rate and it must decay."""
    st, t = sd.trim_level(_ac(), V=45.0)
    st = replace(st, rates=np.array([0.0, np.deg2rad(10.0), 0.0]))
    hist = sd.integrate(t, st, dt=0.01, n=300)
    assert abs(hist[-1].rates[1]) < abs(st.rates[1])


def test_linearise_returns_the_modes():
    """Eigenvalues of the Jacobian: at least one oscillatory pair with a
    period in the seconds-to-minutes range (phugoid / short period)."""
    st, t = sd.trim_level(_ac(), V=45.0)
    ev = np.linalg.eigvals(sd.linearise(t, st))
    osc = [e for e in ev if abs(e.imag) > 1e-4]
    assert osc, "no oscillatory mode at all"
    periods = sorted(2 * np.pi / abs(e.imag) for e in osc)
    assert 0.3 < periods[0] < 200.0
