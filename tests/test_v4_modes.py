"""The two pure modules stage 6 grew: naming the modes, and flying them.

:mod:`gui.v4.modes` turns a Jacobian into named modes. :mod:`gui.v4.manoeuvre`
turns a mode into an excitation. Both are arithmetic with no nicegui and no
session in them, which is why they can be asserted against closed forms here
rather than against a screenshot.

The one claim worth stating twice: an aeroplane that passes every static sign
test can still be a divergent spiral, because the spiral is a RATIO and not a
sign. Two independent routes to that number are checked against each other
below — the eigenvalue of the full nonlinear Jacobian, and the textbook
criterion on the derivative deck — because a stability verdict that only one
calculation supports is a verdict nobody should fly on.
"""

import math

import numpy as np
import pytest

from gui.v4 import manoeuvre as mv, modes as md


# ---------------------------------------------------------------- fixtures

@pytest.fixture(scope="module")
def flown():
    """A real design, armed: its Jacobian, its deck and its state."""
    from aerobo import api, sixdof as sd
    from gui.v4 import app as v4app

    ctx = v4app.assemble()
    cfg = api.RunConfig(problem_name="tail", budget=4, seed=0)
    built = api.PROBLEM_SPECS["tail"].build({}, {}, None)
    ctx.S["run"]["record"] = {"pretend": "a completed run"}
    ctx.S["run"]["report"] = api.design_report(cfg, built.bounds.mean(axis=1))
    ctx.render("controls", "derivatives")
    ctx.render("flight", "fly")
    ctx.act("flight_arm")
    F = ctx.S["flight"]
    return {"J": sd.linearise(F["ac"], F["state"]), "st": F["state"],
            "deck": F["ac"].deck, "ctx": ctx}


# ------------------------------------------------------ naming the modes

def test_all_five_modes_are_named(flown):
    got = md.classify(flown["J"], float(flown["st"].V))
    assert set(got) == {"phugoid", "short period", "dutch roll",
                        "roll subsidence", "spiral"}


def test_the_split_is_by_eigenvector_and_it_is_clean(flown):
    """Longitudinal and lateral do not mix on a symmetric aeroplane in
    level flight — so if the participation is not near 0 or 1, the labels
    above are being handed out on a coin toss."""
    got = md.classify(flown["J"], float(flown["st"].V))
    for key in ("phugoid", "short period"):
        assert got[key].lateral < 0.02, f"{key} came out lateral"
    for key in ("dutch roll", "roll subsidence", "spiral"):
        assert got[key].lateral > 0.98, f"{key} came out longitudinal"


def test_the_phugoid_is_the_SLOW_longitudinal_oscillation(flown):
    got = md.classify(flown["J"], float(flown["st"].V))
    ph, sp = got["phugoid"], got["short period"]
    assert ph.oscillatory and sp.oscillatory
    assert ph.period_s > sp.period_s, \
        "the phugoid was named as the faster of the two"
    # ...and it is the LIGHTLY damped one, which is what makes it visible
    assert ph.damping < sp.damping


def test_roll_subsidence_is_the_FAST_lateral_root_and_the_spiral_the_slow(
        flown):
    got = md.classify(flown["J"], float(flown["st"].V))
    roll, spiral = got["roll subsidence"], got["spiral"]
    assert not roll.oscillatory and not spiral.oscillatory
    assert abs(roll.real) > abs(spiral.real) * 10, \
        "the two real lateral roots were named the wrong way round"
    assert roll.tau_s < 1.0 < (spiral.tau_s or 1e9)


def test_a_conjugate_pair_is_ONE_mode(flown):
    """Three oscillations in a thirteen-state system is six eigenvalues.
    Counting each member separately would report the Dutch roll twice under
    two names — and it did, until the partner was marked."""
    got = md.classify(flown["J"], float(flown["st"].V))
    osc = [m for m in got.values() if m.oscillatory]
    assert len(osc) == 3
    for m in osc:
        assert m.imag > 0, "a mode was reported at its negative frequency"


def test_the_structurally_zero_eigenvalues_are_not_modes(flown):
    """Four position states nothing depends on, plus the quaternion norm
    direction: five of the thirteen are structurally zero and are not
    anything the aeroplane does."""
    got = md.classify(flown["J"], float(flown["st"].V))
    assert len(got) <= 5
    assert all(abs(m.real) + abs(m.imag) > 1e-8 for m in got.values())


def test_a_design_with_no_lateral_oscillation_reports_no_dutch_roll():
    """A gap is reported as a gap. Inventing a mode from whatever real root
    is left over would put a number on a panel with nothing behind it."""
    # a purely longitudinal 2x2 embedded in the 13-state layout: only u and
    # w are coupled, so there is exactly one longitudinal oscillation
    J = np.zeros((13, 13))
    J[7, 7] = J[9, 9] = -0.1
    J[7, 9], J[9, 7] = -1.0, 1.0
    got = md.classify(J, 20.0)
    assert "dutch roll" not in got
    assert "spiral" not in got
    assert set(got) <= {"phugoid", "short period"}


# ---------------------------------------------------- what a Mode reports

@pytest.mark.parametrize("re,im", [(-0.5, 2.0), (-0.03, 0.43), (0.07, 0.0)])
def test_the_mode_numbers_are_the_closed_forms(re, im):
    m = md.Mode(name="x", real=re, imag=im, lateral=0.5)
    assert m.oscillatory is (abs(im) > 1e-6)
    if m.oscillatory:
        assert m.period_s == pytest.approx(2 * math.pi / im)
        assert m.damping == pytest.approx(-re / math.hypot(re, im))
        assert m.tau_s is None
    else:
        assert m.period_s is None and m.damping is None
        assert m.tau_s == pytest.approx(1.0 / abs(re))
    assert m.double_or_half_s == pytest.approx(math.log(2.0) / abs(re))
    assert m.stable is (re < 0)


def test_participation_of_a_pure_axis_is_exact():
    lat, lon = np.zeros(13), np.zeros(13)
    lat[8] = 1.0                       # sideslip velocity v
    lon[9] = 1.0                       # heave velocity w
    assert md.participation(lat, 20.0) == pytest.approx(1.0)
    assert md.participation(lon, 20.0) == pytest.approx(0.0)
    # and the body RATES are deliberately not in it: an eigenvector's rates
    # are its angles times its eigenvalue, so counting them would weight a
    # fast mode by its own frequency
    rates = np.zeros(13)
    rates[10] = 1.0
    assert md.participation(rates, 20.0) == pytest.approx(0.5)


# ---------------------------------------- the two routes to the same answer

def test_the_deck_criterion_agrees_with_the_eigenvalue(flown):
    """THE CROSS-CHECK. ``Cl_beta.Cn_r - Cn_beta.Cl_r`` is a criterion on
    four numbers in the derivative deck; the spiral eigenvalue comes out of
    a numerical Jacobian of the full nonlinear equations. They are computed
    by different code from different inputs and they must agree on the sign,
    or one of them is wrong and the panel is quoting whichever it happens to
    have reached for.
    """
    D = flown["deck"]
    margin = md.spiral_margin(Cl_beta=D.Cl_beta, Cn_r=D.Cn_r,
                              Cn_beta=D.Cn_beta, Cl_r=D.Cl_r)
    spiral = md.classify(flown["J"], float(flown["st"].V))["spiral"]
    assert (margin > 0) == spiral.stable, (
        f"the deck says the spiral {'converges' if margin > 0 else 'diverges'}"
        f" (margin {margin:+.6f}) and the eigenvalue says the opposite "
        f"({spiral.real:+.5f})")


def test_the_design_is_pitch_and_roll_stable_but_the_spiral_is_not(flown):
    """The measured state of the tail design, pinned so a change to any of
    it is deliberate.

        Cm_alpha  -2.119   pitch stiffness      STABLE
        Cl_beta   -0.020   dihedral effect      STABLE
        Cn_beta   +0.137   yaw stiffness        STABLE
        spiral    +0.0708  time to double 9.8 s DIVERGENT

    The last line is not a contradiction of the first three: an aeroplane
    with a weak dihedral effect and a large fin has every static sign right
    and still rolls off with the stick free.
    """
    D = flown["deck"]
    assert D.Cm_alpha < 0
    assert D.Cl_beta < 0
    assert D.Cn_beta > 0
    spiral = md.classify(flown["J"], float(flown["st"].V))["spiral"]
    assert not spiral.stable
    assert 5.0 < spiral.double_or_half_s < 20.0


# ------------------------------------------------------- the excitations

def test_every_manoeuvre_names_a_mode_that_exists():
    for m in mv.MANOEUVRES:
        assert m.mode in {"phugoid", "short period", "dutch roll",
                          "roll subsidence", "spiral"}
    assert {m.key for m in mv.MANOEUVRES} == set(mv.BY_KEY)


def test_the_released_stick_manoeuvres_have_NO_input_at_any_time():
    """The spiral and the phugoid are answered by what the aeroplane does
    when nothing is touching it. An input at any point in either one is
    answering a different question."""
    for key in ("spiral", "phugoid"):
        m = mv.BY_KEY[key]
        assert m.script == () and not m.has_input
        assert mv.duration_s(m) == 0.0
        for t in (0.0, 0.1, 1.0, 30.0, 1000.0):
            assert mv.inputs_at(m, t) == {}


@pytest.mark.parametrize("key,axis", [("short_period", "elevator"),
                                      ("dutch_roll", "rudder")])
def test_a_doublet_REVERSES_and_then_stops(key, axis):
    """One way, the other way, then nothing. A pulse that does not come back
    leaves a trim change behind and the mode is measured about the wrong
    equilibrium."""
    m = mv.BY_KEY[key]
    first = mv.inputs_at(m, 0.01)[axis]
    second = mv.inputs_at(m, mv.duration_s(m) - 0.01)[axis]
    assert first * second < 0, "the doublet did not change sign"
    assert first == pytest.approx(-second)
    assert mv.inputs_at(m, mv.duration_s(m)) == {}
    assert mv.inputs_at(m, mv.duration_s(m) + 5.0) == {}


def test_the_roll_step_is_HELD_and_does_not_reverse():
    m = mv.BY_KEY["roll"]
    held = {mv.inputs_at(m, t)["aileron"] for t in (0.01, 0.5, 1.0, 1.9)}
    assert len(held) == 1, "the step is not a step"
    assert mv.inputs_at(m, 2.5) == {}


def test_the_phugoid_changes_ONLY_the_speed():
    """+10 % along the flight path at the same incidence. A pitch pulse
    instead would put the short period into the first four seconds of the
    trace, which is somebody else's mode."""
    V0, a0 = 20.0, math.radians(3.0)
    vel = np.array([V0 * math.cos(a0), 0.0, V0 * math.sin(a0)])
    q = np.array([1.0, 0.0, 0.0, 0.0])
    v2, q2 = mv.perturbation(vel, q, mv.BY_KEY["phugoid"])
    assert float(np.linalg.norm(v2)) == pytest.approx(1.10 * V0)
    assert math.atan2(v2[2], v2[0]) == pytest.approx(a0)
    assert v2[1] == pytest.approx(0.0, abs=1e-12)
    assert q2 == pytest.approx(q)


def test_the_spiral_banks_and_changes_nothing_else():
    V0 = 20.0
    vel = np.array([V0, 0.0, 0.0])
    q = np.array([1.0, 0.0, 0.0, 0.0])
    v2, q2 = mv.perturbation(vel, q, mv.BY_KEY["spiral"])
    assert v2 == pytest.approx(vel)
    # 10 degrees of bank, read back through the same euler convention the
    # simulation uses
    from aerobo import sixdof as sd

    roll = sd.State(quat=q2, vel=v2).euler[0]
    assert math.degrees(roll) == pytest.approx(10.0)


def test_the_bank_is_about_the_AEROPLANES_nose_not_the_earths_north():
    """A body-axis roll, i.e. ``q * q_x``. Pre-multiplying rolls about the
    earth axis instead, which is the same manoeuvre only in level flight and
    a different one the moment there is any pitch attitude — so the test
    puts the aeroplane 30 degrees nose-up, where the two disagree.
    """
    from aerobo import sixdof as sd

    p = math.radians(30.0)
    q = np.array([math.cos(p / 2), 0.0, math.sin(p / 2), 0.0])   # pitch up
    _v, q2 = mv.perturbation(np.array([20.0, 0.0, 0.0]), q,
                             mv.BY_KEY["spiral"])
    roll, pitch, _yaw = sd.State(quat=q2).euler
    # a BODY roll leaves the pitch ATTITUDE about where it was and puts the
    # whole 10 degrees on the roll axis...
    assert math.degrees(roll) == pytest.approx(10.0, abs=0.6)
    assert math.degrees(pitch) == pytest.approx(30.0, abs=0.6)
    # ...and it is NOT what an earth-axis roll would have produced
    h = math.radians(10.0) / 2
    r = np.array([math.cos(h), math.sin(h), 0.0, 0.0])
    earth = np.array([
        r[0] * q[0] - r[1] * q[1] - r[2] * q[2] - r[3] * q[3],
        r[0] * q[1] + r[1] * q[0] + r[2] * q[3] - r[3] * q[2],
        r[0] * q[2] - r[1] * q[3] + r[2] * q[0] + r[3] * q[1],
        r[0] * q[3] + r[1] * q[2] - r[2] * q[1] + r[3] * q[0]])
    assert not np.allclose(q2, earth, atol=1e-6), \
        "the body-axis roll and the earth-axis roll came out identical — " \
        "this test cannot see the difference it exists to check"


# ------------------------------------------------ a real part that is zero

def test_a_mode_the_jacobian_cannot_resolve_is_NEUTRAL_not_divergent():
    """``real < 0`` has no tolerance, and the Jacobian is a finite difference.

    Measured on the shipped tail design the phugoid lands at ``+6e-6`` — the
    same value at every finite-difference step from 1e-8 to 1e-3, and a 400 s
    time-domain fit of the speed envelope agrees at ``+1.7e-5``. The Modes
    panel called that DIVERGENT and offered a time to double of 115 795 s.
    Thirty-two hours is not a divergence; it is zero.
    """
    tiny = md.Mode(name="phugoid", real=+6e-6, imag=0.2594, lateral=0.0)
    assert tiny.neutral
    assert tiny.stable, "a neutral mode is not something to colour red"
    assert not tiny.diverges
    assert tiny.double_or_half_s is None, \
        "a mode that never doubles must not be given a time to double"

    # ...and the tolerance is a RESOLUTION floor, not a handling threshold:
    # a spiral that doubles in a minute is still a divergence and must say so
    real = md.Mode(name="spiral", real=math.log(2.0) / 60.0, imag=0.0,
                   lateral=1.0)
    assert real.diverges and not real.stable and not real.neutral
    assert real.double_or_half_s == pytest.approx(60.0)
    # the floor itself, stated as the time it corresponds to
    assert math.log(2.0) / md.NEUTRAL_1_PER_S > 6000.0

    # a genuinely convergent mode is not swept up in it
    good = md.Mode(name="roll subsidence", real=-4.7, imag=0.0, lateral=1.0)
    assert good.stable and not good.neutral and not good.diverges
    assert good.double_or_half_s == pytest.approx(math.log(2.0) / 4.7)


def test_the_neutral_band_is_symmetric():
    """A mode decaying at 1e-6 is as unresolvable as one growing at 1e-6."""
    for re in (+5e-5, -5e-5):
        m = md.Mode(name="phugoid", real=re, imag=0.26, lateral=0.0)
        assert m.neutral and m.stable and not m.diverges

