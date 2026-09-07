"""The car wing's mission: a lap in place of a drag-coefficient allowance.

What is gated here, and why each one is a gate rather than a check:

* THE CLOSED-FORM LIMITS. A car with no aero must reproduce the skidpad
  ``V^2 = mu g R`` and the drag-limited top speed ``(2 P_eff / rho CdA)^(1/3)``
  — both are asserted with ``==``, because the module's own algebra is
  supposed to collapse onto them and "close" would hide a wrong grouping.
* THE TWO DIRECTIONS. More downforce at zero drag cost is never slower; more
  drag at zero downforce is never faster. These are the only reason a lap
  time is allowed to be an objective at all, and the drag one is checked on a
  BRAKING-dominated layout as well, because braking is the one mechanism in
  the model that pulls the other way (drag helps a car stop) — measured, not
  assumed away.
* THE SOLUTION SATISFIES ITS OWN EQUATION. The corner speed is fed back into
  ``mu_eff(N) N = m V^2 / R`` and the residual has to be round-off.
* STEP INDEPENDENCE, with its ORDER. Halving the step must divide the change
  by ~4 (the kink at the braking point makes it second order, not RK4's
  fourth), and the default resolution must sit inside a tolerance this file
  measures rather than one the module asserts.
* THE INERT DEFAULTS. ``k_load = 0`` and a rigid heave law must be bit-for-bit
  no-ops, and a constant coefficient must give the same lap as a callable
  returning it.
* THE BRAKING MECHANISM, ON ITS OWN. A wing shortening the braking zone is
  half of what this module exists to say against a CD budget, and it is worth
  a tenth of a second against a corner-speed term worth more than one — so it
  is gated alone, against an integral computed in this file, and not left to
  the lap-level monotonicity gates that outvote it.
* THE PROVENANCE OF THE LOAD-SENSITIVITY EXPONENT. No verifiable source
  publishes one; every published form that could be checked is LINEAR in load.
  So there is a text tripwire here that fails if any sentence of cartrack.py
  ever attributes an exponent to anybody again.
* WHAT PROMISES NOT TO RAISE. ``lap_time`` says it returns a reason for every
  in-contract failure, so every way of not having a lap is put through it —
  a drag area at or below zero, a law with no root, a coefficient that is not
  a number, a law that returns one — and the single documented raise
  (``n_steps < 2``) is asserted to still be a raise.

Where a physical trend is asserted, its DIRECTION and its closed-form limit
are asserted together, and every quoted number is re-derived here from the
module rather than pinned — except where a published docstring number is
deliberately pinned as a TRIPWIRE, which is marked as such where it happens.

Two tests here carry a "MUTATION THIS TEST IS BUILT TO CATCH" line in their
own docstring. That is not decoration: each one names a one-line change to
cartrack.py that the whole suite used to survive, and each was verified by
making that change, watching the test go red, and restoring the file.
"""

import inspect
import re

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import brentq

from aerobo import cartrack
from aerobo.cartrack import (
    BALANCE_HALF_WIDTH,
    CABRERA_2018_HANKOOK_205_65_R15,
    EXAMPLE_CD_EXPONENT,
    K_LOAD_REFERENCE_LATERAL,
    K_LOAD_REFERENCE_LONGITUDINAL,
    N_STEPS_DEFAULT,
    CarSpec,
    Corner,
    HeaveLaw,
    Straight,
    TrackSpec,
    aero_balance,
    balance_margin,
    corner_speed,
    example_reynolds_cd_law,
    lap_time,
    representative_points,
    single_point_error,
    speed_time_histogram,
    straight_profile,
    synthetic_lap,
    time_weighted_mean_speed,
    top_speed,
    top_speed_result,
)
from aerobo.mission import G0

# The published carwing operating point, so the mission is exercised on the
# wing it exists for: CarWingProblem's own default S = 0.4 m^2 and the CZ/CD
# MISSION_DESIGN_WATER_TRACK.md section 3 measured on `car rear wing`.
CZ_PUB, CD_PUB, S_PUB = 0.655860500048, 0.023739403946, 0.4
CZ_A = CZ_PUB * S_PUB
CD_A = CD_PUB * S_PUB

CAR = CarSpec()
LAP = synthetic_lap()


# ------------------------------------------------------------- provenance

# The rule this file enforces on cartrack.py's prose, and the reason for it.
#
# A literature check found that NO verifiable peer-reviewed source publishes a
# general power-law exponent for mu(Fz). Every published functional form that
# could be checked is LINEAR in load (the Magic Formula's peak-friction term,
# mu = PD1 + PD2 dfz), and every numeric coefficient in one belongs to one
# specific tested tyre. The widely repeated "Fz^0.7-0.9" traces back to a
# Wikipedia sentence that carries its own [citation needed] tag.
#
# So cartrack.py may name a source for the EXISTENCE, the SIGN and the ORDER
# of load sensitivity, and may never name one for an exponent. The mechanical
# rule that makes it testable: **no sentence may contain both a source and an
# exponent**. Converting a published linear slope into an exponent is this
# repo's own arithmetic, and the sentence that says so must not also name whose
# slope it was.
#
# Proper nouns are matched CASE-SENSITIVELY on purpose, so that an ALL-CAPS
# identifier naming one of this module's own constants
# (CABRERA_2018_HANKOOK_205_65_R15) is not read as a claim about a source.
_SOURCE_NAMES = ("Cabrera", "Pacejka", "Magic Formula", "Sensors 18",
                 "Wikipedia", "DOI", "10.3390")
_HEARSAY = ("literature", "handbook", "textbook", "well-known", "well known",
            "known to be", "typical value", "typically quoted",
            "typically reported", "commonly quoted", "commonly reported",
            "widely quoted", "widely reported", "usually quoted",
            "usually taken", "standard value", "rule of thumb", "et al")
_EXPONENT_TOKENS = ("exponent", "power law", "power-law")
#: the retracted claim itself, in the shapes it takes
_RETRACTED = (r"0\.7\s*(?:to|-|–|and|,)\s*0\.9",
              r"0\.9\s*(?:to|-|–|and|,)\s*0\.7")


def _module_prose(mod) -> str:
    """Every docstring and every comment block of ``mod``, and nothing else.

    Code is left out because an identifier is not a claim, and because a run
    of code carries no full stops, so scanning it would glue two unrelated
    docstrings into one 'sentence'.
    """
    import ast

    src = inspect.getsource(mod)
    parts = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node)
            if doc:
                parts.append(doc)
    block: list = []
    for line in src.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            block.append(stripped.lstrip("#:").lstrip("#").strip())
        elif block:
            parts.append(" ".join(block))
            block = []
    if block:
        parts.append(" ".join(block))
    return "\n\n".join(parts)


def _sentences(text: str) -> list:
    """Split on a full stop followed by whitespace, so decimals do not split."""
    return re.split(r"\.[ \t\r\n]+", text)


def test_no_exponent_in_this_module_is_attributed_to_a_source():
    """The provenance tripwire: an attribution must not creep back.

    MUTATION THIS TEST IS BUILT TO CATCH: writing an exponent back onto a
    source anywhere in cartrack.py — "the literature gives k = 0.7-0.9", "a
    typical exponent is 0.8", "Cabrera's exponent is 0.14". Each of those puts
    a source token and an exponent token in one sentence, and each turns this
    test red. The retracted 0.7-0.9 range is banned outright, wherever it
    appears and however it is punctuated.

    It also asserts the positive half, because deleting the attribution is not
    the same as labelling the number: the phrase "THIS REPO'S OWN CALIBRATION"
    has to be in the module, in those words, or the exponent is unowned again.
    """
    prose = _module_prose(cartrack)

    assert "THIS REPO'S OWN CALIBRATION" in prose, (
        "the load-sensitivity exponent has to be labelled as this repo's own "
        "calibration, in those words: it belongs to nobody else")
    assert "Cabrera" in prose and "10.3390/s18030896" in prose, (
        "the one source read in full is cited for the existence, the sign and "
        "the order of load sensitivity; dropping it leaves the effect itself "
        "unsourced")

    for pat in _RETRACTED:
        assert re.search(pat, prose) is None, (
            f"{pat!r} matched: the 0.7-0.9 range is uncited and must not "
            f"appear")

    offenders = []
    for sentence in _sentences(prose):
        low = sentence.lower()
        named = ([t for t in _SOURCE_NAMES if t in sentence]
                 + [t for t in _HEARSAY if t in low])
        exps = [t for t in _EXPONENT_TOKENS if t in low]
        if named and exps:
            offenders.append((named, exps, " ".join(sentence.split())[:220]))
    assert offenders == [], offenders


def test_the_reference_exponents_are_this_repos_arithmetic_on_a_linear_law():
    """What the source publishes, what we compute from it, and the gap.

    The four coefficients are a LINEAR law's; the exponent is the local
    logarithmic slope that law has at its own reference load, which is
    ``-PD2/PD1`` and is arithmetic done here. Both halves are re-derived from
    the transcribed coefficients rather than read off the module's constants,
    so a constant edited by hand fails.
    """
    c = CABRERA_2018_HANKOOK_205_65_R15
    assert (c["PDX1"], c["PDX2"]) == (1.10206790, -0.18524061)
    assert (c["PDY1"], c["PDY2"]) == (0.932775, -0.128085)
    # the sign is the finding: friction FALLS with load, so PD2 < 0 and k > 0
    assert c["PDX2"] < 0.0 and c["PDY2"] < 0.0

    assert K_LOAD_REFERENCE_LONGITUDINAL == -c["PDX2"] / c["PDX1"]
    assert K_LOAD_REFERENCE_LATERAL == -c["PDY2"] / c["PDY1"]
    assert 0.0 < K_LOAD_REFERENCE_LATERAL < K_LOAD_REFERENCE_LONGITUDINAL < 1.0

    # the local slope is what it says it is: match the two laws at N_ref and
    # their log-slopes agree there, to a finite difference taken here
    for pd1, pd2, k in ((c["PDX1"], c["PDX2"], K_LOAD_REFERENCE_LONGITUDINAL),
                        (c["PDY1"], c["PDY2"], K_LOAD_REFERENCE_LATERAL)):
        eps = 1e-6

        def affine(r):
            return pd1 * (1.0 + (pd2 / pd1) * (r - 1.0))

        def power(r):
            return pd1 * r ** (-k)

        slope_a = (np.log(affine(1 + eps)) - np.log(affine(1 - eps))) / (
            np.log(1 + eps) - np.log(1 - eps))
        assert slope_a == pytest.approx(-k, rel=1e-6)
        assert power(1.0) == pytest.approx(affine(1.0), rel=1e-15)


def test_the_power_law_and_the_published_linear_law_part_company():
    """The price of the modelling choice, measured over the lap's own loads.

    The module says the two shapes agree only to first order at ``N_ref`` and
    finish 1.23 % / 0.97 % of mu apart at the top of the reference lap's load
    range. Both the load range and the gap are re-derived here, so if the lap
    or the reference car moves, this fails rather than going quietly stale."""
    lap = lap_time(CZ_A, CD_A, CAR, LAP)
    r = CAR.normal_load(lap["V_max"], CZ_A) / CAR.weight_n
    assert r == pytest.approx(1.3854, abs=5e-4)

    c = CABRERA_2018_HANKOOK_205_65_R15
    gaps = []
    for pd1, pd2, k in ((c["PDX1"], c["PDX2"], K_LOAD_REFERENCE_LONGITUDINAL),
                        (c["PDY1"], c["PDY2"], K_LOAD_REFERENCE_LATERAL)):
        affine = 1.0 + (pd2 / pd1) * (r - 1.0)
        power = r ** (-k)
        gaps.append((power - affine) / affine)
    assert gaps[0] == pytest.approx(0.0123, abs=5e-4)
    assert gaps[1] == pytest.approx(0.0097, abs=5e-4)
    # the power law is the OPTIMISTIC one of the two at high load, which is
    # the direction the module's bias section claims for everything in it
    assert all(g > 0.0 for g in gaps)


def test_a_reference_exponent_is_not_a_default():
    """A calibration is a default only where the thing it calibrates is
    present. This package has no tyre, so ``k_load`` stays 0 and the reference
    order has to be asked for by name."""
    assert CarSpec().k_load == 0.0
    assert CarSpec().mu_eff(1e6) == CarSpec().mu0
    assert K_LOAD_REFERENCE_LATERAL != 0.0


# ---------------------------------------------------------------- the specs


def test_load_sensitivity_at_or_above_one_is_refused_and_says_why():
    """k >= 1 makes the grip FORCE ``N^(1-k)`` fall with load, i.e. a tyre
    that grips less in absolute terms the harder you push it. That is not a
    tyre, so it is a validation error and not a slow answer."""
    CarSpec(k_load=0.999)                       # allowed, however extreme
    with pytest.raises(ValueError, match="N\\^\\(1-k\\)"):
        CarSpec(k_load=1.0)
    with pytest.raises(ValueError):
        CarSpec(k_load=-0.01)


def test_the_frame_is_validated_where_it_has_to_be():
    with pytest.raises(ValueError):
        CarSpec(x_cg_frac=1.0)                  # CG on the rear axle
    with pytest.raises(ValueError):
        CarSpec(drivetrain_eta=1.5)             # more out than in
    with pytest.raises(ValueError):
        CarSpec(mass_kg=0.0)
    with pytest.raises(ValueError):
        CarSpec(balance_window=(0.6, 0.4))
    # the wing's station is NOT bounded: a rear wing overhangs the rear axle,
    # so x > 1 is the normal case and refusing it would refuse the family
    CarSpec(x_wing_frac=1.6)


def test_a_lap_needs_a_corner_and_a_straight_is_its_length():
    with pytest.raises(ValueError, match="at least one corner"):
        TrackSpec(segments=(Straight(length_m=100.0),))
    with pytest.raises(ValueError, match="one straight of 300"):
        TrackSpec(segments=(Corner(radius_m=50.0, arc_m=40.0),
                            Straight(length_m=100.0),
                            Straight(length_m=200.0)))
    with pytest.raises(ValueError):
        TrackSpec(segments=())
    with pytest.raises(ValueError):
        TrackSpec(segments=(Corner(radius_m=50.0, arc_m=40.0), 3.0))
    # the wrap is checked too: a leading and a trailing straight are adjacent
    with pytest.raises(ValueError):
        TrackSpec(segments=(Straight(length_m=100.0),
                            Corner(radius_m=50.0, arc_m=40.0),
                            Straight(length_m=200.0),
                            Straight(length_m=50.0)))


def test_rotation_puts_a_corner_first_and_maps_back():
    t = TrackSpec(segments=(Straight(length_m=100.0),
                            Corner(radius_m=50.0, arc_m=40.0),
                            Straight(length_m=200.0),
                            Corner(radius_m=30.0, arc_m=20.0)))
    segs, shift = t.rotated_to_start_at_a_corner()
    assert isinstance(segs[0], Corner)
    assert shift == 1
    assert [t.segments[(i + shift) % 4] for i in range(4)] == list(segs)
    assert t.length_m == 360.0


def test_the_reference_lap_has_the_three_things_it_claims_to_have():
    """A low-speed corner, a fast corner and a drag-limited straight — the
    three questions a car-wing mission has to be able to put. Each is asserted
    as a PROPERTY re-derived from the run, never as a pinned speed."""
    lap = lap_time(CZ_A, CD_A, CAR, LAP)
    assert lap["feasible"], lap["reason"]
    corners = [r for r in lap["segments"] if r["kind"] == "corner"]
    straights = [r for r in lap["segments"] if r["kind"] == "straight"]
    assert len(corners) == 3 and len(straights) == 3

    # the slow corner is one where the grip is nearly all WEIGHT, so downforce
    # cannot help; the fast one is where the aero share is large
    def weight_share(V):
        N = CAR.normal_load(V, CZ_A)
        return CAR.weight_n / N

    slow, fast = min(corners, key=lambda r: r["V_in"]), max(
        corners, key=lambda r: r["V_in"])
    assert weight_share(slow["V_in"]) > 0.95
    assert weight_share(fast["V_in"]) < 0.85

    # the long straight ends drag-limited: at its peak the drag is (V/V_top)^3
    # of the tractive force available, and that has to be most of it
    longest = max(straights, key=lambda r: r["length_m"])
    assert longest["drag_frac_at_peak"] > 0.85
    assert longest["V_peak_over_top"] > 0.95
    # ... and no corner on this lap is an "aero_unbounded" statement
    assert all(r["limited_by"] == "grip" for r in corners)
    assert lap["warnings"] == []


# ---------------------------------------------------------------- friction


def test_constant_friction_is_bit_for_bit_the_default():
    """k = 0 must take no power and touch no float: mu_eff IS mu0."""
    for N in (1.0, CAR.weight_n, 1e6):
        assert CAR.mu_eff(N) == CAR.mu0
        assert CAR.grip_force(N) == CAR.mu0 * N


def test_load_sensitivity_lowers_mu_but_never_the_force():
    """The direction AND the closed form. mu_eff falls as N^-k; the FORCE
    rises as N^(1-k), which is why k < 1 is the validated bound."""
    car = CarSpec(k_load=0.15)
    assert car.mu_eff(car.n_ref) == pytest.approx(car.mu0, rel=1e-15)
    loads = np.array([0.5, 1.0, 2.0, 8.0]) * car.n_ref
    mus = np.array([car.mu_eff(n) for n in loads])
    fs = np.array([car.grip_force(n) for n in loads])
    assert np.all(np.diff(mus) < 0.0)          # coefficient falls with load
    assert np.all(np.diff(fs) > 0.0)           # force still rises
    closed = car.mu0 * car.n_ref ** 0.15 * loads ** 0.85
    assert fs == pytest.approx(closed, rel=1e-14)
    assert car.mu_eff(0.0) == 0.0 and car.grip_force(-1.0) == 0.0


# ---------------------------------------------------------------- top speed


def test_top_speed_is_the_closed_form_exactly_and_scales_as_the_cube_root():
    """MUTATION THIS TEST IS BUILT TO CATCH: deleting ``drivetrain_eta`` from
    ``CarSpec.power_eff_w``, i.e. returning ``power_w`` alone.

    The previous version of this test computed its expected value FROM
    ``car.power_eff_w`` — the property under test — so the efficiency could be
    dropped entirely and every assertion still held, though it multiplies the
    only power the lap ever uses. Everything here is written in terms of
    ``power_w`` and ``drivetrain_eta`` SEPARATELY, and the last two assertions
    are behavioural rather than algebraic: the efficiency is a plain factor on
    the power, so halving it must divide the top speed by exactly 2^(1/3), and
    it must cost the reference lap 8.6 s.
    """
    car = CarSpec(cza_car_m2=0.0)
    v = top_speed(0.0, car)
    assert v == (2.0 * car.power_w * car.drivetrain_eta
                 / (car.rho * car.cda_car_m2)) ** (1 / 3)
    # doubling the total drag area divides the top speed by 2^(1/3), exactly
    v2 = top_speed(car.cda_car_m2, car)
    assert v2 / v == pytest.approx(2.0 ** (-1 / 3), rel=1e-14)
    # a callable is the numeric path and must land on the same root
    assert top_speed(lambda V: 0.0, car) == pytest.approx(v, rel=1e-12)

    # halving the efficiency IS doubling the drag area, as far as (4) can see
    half = CarSpec(cza_car_m2=0.0, drivetrain_eta=0.5 * car.drivetrain_eta)
    assert top_speed(0.0, half) / v == pytest.approx(2.0 ** (-1 / 3), rel=1e-14)
    # ... and the lap spends it: P_eff is the only power in _accel as well
    t_full = lap_time(CZ_A, CD_A, CAR, LAP)["lap_time_s"]
    t_half = lap_time(CZ_A, CD_A,
                      CarSpec(drivetrain_eta=0.5 * CAR.drivetrain_eta),
                      LAP)["lap_time_s"]
    assert t_half - t_full == pytest.approx(8.63, abs=0.05)


def test_top_speed_falls_with_drag_and_ignores_downforce():
    vs = [top_speed(c, CAR) for c in np.linspace(0.0, 0.5, 11)]
    assert np.all(np.diff(vs) < 0.0)
    # rolling resistance is not modelled, so downforce cannot enter here
    assert top_speed(CD_A, CarSpec(cza_car_m2=0.0)) == top_speed(
        CD_A, CarSpec(cza_car_m2=5.0))


def test_the_callable_top_speed_has_no_search_cap():
    """The regression this is here for: the callable path used to solve over a
    fixed ``(0, 200] m/s`` bracket and RAISE above it — an undocumented cap on
    a path carwing.py takes on every evaluation, since it hands ``lap_time``
    ``np.interp`` callables. A slippery car whose top speed is 711 m/s has to
    come back on the callable path exactly where the closed form puts it."""
    slippery = CarSpec(cda_car_m2=0.001, cza_car_m2=0.0)
    closed = top_speed(0.0, slippery)
    assert closed > 200.0                       # above the old bracket
    assert top_speed(lambda V: 0.0, slippery) == pytest.approx(closed,
                                                               rel=1e-11)
    assert top_speed_result(0.0, slippery)["path"] == "closed_form"
    assert top_speed_result(lambda V: 0.0, slippery)["path"] == "bracketed"
    # a whole lap on the callable path, which is what carwing.py runs
    lap = lap_time(lambda V: CZ_A, lambda V: CD_A, slippery, LAP)
    assert lap["feasible"], lap["reason"]
    assert lap["V_top"] == pytest.approx(top_speed(CD_A, slippery), rel=1e-11)


def test_top_speed_returns_a_reason_where_there_is_no_top_speed():
    """Three ways not to have one, all of them statements. The bare-float
    :func:`top_speed` raises the SAME statement, and says in its docstring
    that this is the price of returning a number."""
    dead = top_speed_result(-CAR.cda_car_m2, CAR)          # total drag area 0
    assert dead["feasible"] is False and dead["V_top"] is None
    assert "not positive" in dead["reason"]
    assert dead["cda_total_m2"] == 0.0

    negative = top_speed_result(lambda V: -1.0, CAR)       # and on the law
    assert negative["feasible"] is False
    assert "still accelerating" in negative["reason"]

    nan = top_speed_result(float("nan"), CAR)
    assert nan["feasible"] is False and "not finite" in nan["reason"]

    with pytest.raises(ValueError, match="not positive"):
        top_speed(-CAR.cda_car_m2, CAR)


# -------------------------------------------------------------- corner speed


def test_a_car_with_no_aero_is_the_skidpad_exactly():
    car = CarSpec(cza_car_m2=0.0)
    for R in (25.0, 80.0, 180.0):
        out = corner_speed(R, 0.0, CD_A, car)
        assert out["limited_by"] == "grip"
        assert out["V"] == np.sqrt(car.mu0 * G0 * R)


def test_the_corner_speed_satisfies_its_own_equation_to_round_off():
    """The gate that no amount of algebra can fake: put the answer back."""
    for car in (CAR, CarSpec(k_load=0.12)):
        for R in (25.0, 80.0, 180.0, 400.0):
            out = corner_speed(R, CZ_A, CD_A, car)
            if out["limited_by"] != "grip":
                continue
            V = out["V"]
            demand = car.mass_kg * V * V / R
            grip = car.grip_force(car.normal_load(V, CZ_A))
            assert abs(grip - demand) / demand < 1e-12
            assert abs(out["residual"]) / demand < 1e-12


def test_the_aero_unbounded_corner_is_a_statement_and_names_its_threshold():
    """Equation (2)'s denominator goes non-positive at
    ``R = m / (1/2 mu rho CzA_tot)``. Above it there is no grip-limited speed
    at all, and the module has to SAY so and cap at the top speed rather than
    return a large number."""
    cza_tot = CAR.cza_car_m2 + CZ_A
    r_star = CAR.mass_kg / (0.5 * CAR.mu0 * CAR.rho * cza_tot)
    above = corner_speed(r_star * 1.01, CZ_A, CD_A, CAR)
    assert above["limited_by"] == "aero_unbounded"
    assert above["V"] == above["V_top"]
    assert above["V_grip"] is None
    assert f"{r_star:.4g}" in above["reason"]
    # just below it the equation does have a root — far above the top speed,
    # so the car is capped for a DIFFERENT and separately-named reason
    below = corner_speed(r_star * 0.99, CZ_A, CD_A, CAR)
    assert below["limited_by"] == "top_speed"
    assert below["V_grip"] > below["V_top"]
    assert below["V"] == below["V_top"]


def test_load_sensitivity_bounds_the_corner_only_as_algebra():
    """The module docstring claims a load-sensitive tyre has a finite corner
    speed at every radius, and then refuses to pretend that is useful. Both
    halves are checked: the root EXISTS (found here independently, not by the
    module), and it is astronomically out of reach, so the module still
    reports the corner as aero-unbounded."""
    car = CarSpec(k_load=0.05)
    R = 5000.0

    def resid(V):
        return car.grip_force(car.normal_load(V, CZ_A)) - car.mass_kg * V * V / R

    assert resid(1e3) > 0.0                      # still gripping at 1000 m/s
    root = brentq(resid, 1e3, 1e12, xtol=1.0)
    assert root > 1e8                            # measured ~3.5e9 m/s
    out = corner_speed(R, CZ_A, CD_A, car)
    assert out["limited_by"] == "aero_unbounded"
    assert out["V"] == out["V_top"]


def test_the_numeric_corner_path_lands_on_the_closed_form():
    """A CONSTANT callable is routed to the scan-and-Brent path by type, so
    this compares two genuinely different solvers on one answer."""
    for R in (25.0, 80.0, 180.0):
        closed = corner_speed(R, CZ_A, CD_A, CAR)["V"]
        numeric = corner_speed(R, lambda V: CZ_A, CD_A, CAR)["V"]
        assert numeric == pytest.approx(closed, rel=1e-13)


def test_corner_speed_rises_with_downforce_and_ignores_drag():
    for R in (25.0, 80.0, 180.0):
        vs = [corner_speed(R, c, CD_A, CAR)["V"]
              for c in np.linspace(0.0, 1.0, 9)]
        assert np.all(np.diff(vs) > 0.0)
    # drag reaches a corner only through the top-speed cap, and at 25 m the
    # cap is nowhere near — so the corner speed must not move at all
    a = corner_speed(25.0, CZ_A, CD_A, CAR)["V"]
    b = corner_speed(25.0, CZ_A, 4.0 * CD_A, CAR)["V"]
    assert a == b


# ---------------------------------------------------------------- straights


def test_the_straight_starts_and_ends_where_the_corners_put_it():
    pr = straight_profile(1400.0, 20.0, 30.0, CZ_A, CD_A, CAR)
    assert pr["feasible"], pr["reason"]
    assert pr["V"][0] == pytest.approx(20.0, rel=1e-12)
    assert pr["V"][-1] == pytest.approx(30.0, rel=1e-12)
    assert pr["V_peak"] > 60.0                   # it accelerates in between
    assert pr["brake_point_m"] is not None
    assert not pr["brake_limited_entry"]
    assert pr["V"].max() <= pr["V_top"] * (1 + 1e-12)


def test_a_straight_too_short_to_brake_on_is_reported_never_silent():
    """Corner speeds are not propagated backwards, so this case understates
    the lap. The contract is that it says so, in the warnings, with the entry
    speed the next corner actually implies."""
    pr = straight_profile(30.0, 56.0, 19.0, CZ_A, CD_A, CAR)
    assert pr["brake_limited_entry"]
    assert pr["V_entry_implied"] < 56.0
    track = TrackSpec(segments=(Corner(radius_m=180.0, arc_m=50.0),
                                Straight(length_m=30.0),
                                Corner(radius_m=25.0, arc_m=50.0),
                                Straight(length_m=800.0)))
    lap = lap_time(CZ_A, CD_A, CAR, track)
    assert lap["feasible"]
    assert any("too short to brake" in w for w in lap["warnings"])
    assert any("understated" in w for w in lap["warnings"])
    # ... and the reference lap does NOT trip it
    assert lap_time(CZ_A, CD_A, CAR, LAP)["warnings"] == []


def _braking_quadrature(cz_a, v_hi, v_lo, car=CAR):
    """``(distance, time)`` braking from ``v_hi`` to ``v_lo`` with ``cd_a = 0``.

    Written out here from ``CarSpec.grip_force`` and ``CarSpec.q`` alone, so it
    shares no integrator with the module under test:

        a(V) = (mu_eff(N(V)) N(V) + q CdA_car) / m,
        x = int_{v_lo}^{v_hi} V dV / a(V),   t = int_{v_lo}^{v_hi} dV / a(V).
    """
    def a_of(V):
        q = car.q(V)
        f_grip = car.grip_force(car.weight_n + q * (car.cza_car_m2 + cz_a))
        return (f_grip + q * car.cda_car_m2) / car.mass_kg

    d = quad(lambda V: V / a_of(V), v_lo, v_hi, epsabs=1e-13, epsrel=1e-13)[0]
    t = quad(lambda V: 1.0 / a_of(V), v_lo, v_hi, epsabs=1e-13, epsrel=1e-13)[0]
    return d, t


def test_downforce_shortens_the_braking_zone_on_its_own():
    """The braking half of the module's case against a CD budget, gated alone.

    MUTATION THIS TEST IS BUILT TO CATCH: deleting the wing's downforce from
    ``_brake``'s grip term — ``car.grip_force(weight_n + q * cza_car_m2)`` in
    place of ``... + q * (cza_car_m2 + cz_law(V))`` — so that a wing shortens
    no braking zone at all. Before this test the whole suite stayed green
    under it. Measured, that mutation moves the reference lap only 61.3717 ->
    61.3991 s with the published wing and 60.3681 -> 60.4646 s with a 1.0 m^2
    one, i.e. 0.03-0.10 s, while the downforce gate above only requires two
    m^2 of CzA to be worth more than 1 s — which the corner-speed term supplies
    on its own. The mechanism was invisible because it was outvoted.

    So it is gated three ways here, all with ``cd_a = 0`` so that drag is
    identical between the arms and the ONLY channel a wing has is ``_brake``:

    1. against an integral this file computes from ``CarSpec.grip_force``;
    2. by the braking DISTANCE, which a 1.2 m^2 wing must cut by 8.9 %;
    3. on a fixed-length straight, where the brake point must move later
       (102 -> 111 m) and the time must fall — and must not be bit-identical,
       which is what the mutation makes it.
    """
    v_hi, v_lo = 60.0, 20.0
    d0, t0 = _braking_quadrature(0.0, v_hi, v_lo)
    d1, t1 = _braking_quadrature(1.2, v_hi, v_lo)

    # (2) the direction and the size of the effect, from the integral alone
    assert d1 < d0
    assert (d0 - d1) / d0 == pytest.approx(0.0889, abs=2e-3)

    # (1) the module's own integrator reproduces both, on a straight whose
    # length IS the braking distance, so the whole segment is braking
    for cz, d, t in ((0.0, d0, t0), (1.2, d1, t1)):
        pr = straight_profile(d, v_hi, v_lo, cz, 0.0, CAR)
        assert pr["feasible"], pr["reason"]
        assert pr["V"][0] == pytest.approx(v_hi, rel=1e-9)
        assert pr["V"][-1] == pytest.approx(v_lo, rel=1e-9)
        assert pr["t_s"] == pytest.approx(t, rel=1e-4)

    # the isolation is PROVED, not assumed: over the whole speed range these
    # straights visit, the traction limit is the ENGINE, so _accel's grip term
    # (which also carries the downforce) cannot be what moved
    for cz in (0.0, 1.2):
        for V in np.linspace(v_lo, 63.0, 40):
            assert (CAR.power_w * CAR.drivetrain_eta / V
                    < CAR.grip_force(CAR.normal_load(V, cz)))

    # (3) on a fixed straight: later braking, less time, and not the same
    bare = straight_profile(200.0, v_hi, v_lo, 0.0, 0.0, CAR)
    winged = straight_profile(200.0, v_hi, v_lo, 1.2, 0.0, CAR)
    assert winged["brake_point_m"] > bare["brake_point_m"]
    assert winged["t_s"] < bare["t_s"]
    assert winged["t_s"] != bare["t_s"]


def test_drag_shortens_the_braking_zone_the_countervailing_mechanism():
    """This is the one place the drag gate could fail, so it is measured
    head-on rather than left implicit: on a segment that is pure braking,
    more drag really is FASTER, because the tyres are not doing all the work.
    The lap gate below then has to hold in spite of it."""
    slow = straight_profile(60.0, 56.0, 19.0, 0.0, 0.0, CAR)["t_s"]
    fast = straight_profile(60.0, 56.0, 19.0, 0.0, 1.0, CAR)["t_s"]
    assert fast < slow


def test_downforce_buys_acceleration_where_the_traction_limit_binds():
    """``F_traction = min(power, grip)`` is a claim, and on the REFERENCE car
    it is never exercised — 250 kW on 1200 kg of slicks outruns the tyres only
    below 12.32 m/s, and the slowest corner on the reference lap is 19.45. So
    the traction branch is gated on a car where it does bind, with the
    straight's two ends PINNED so that no corner speed can leak into the
    comparison: downforce then has to buy acceleration, and to buy an order of
    magnitude more of it than it does on the power-limited car."""
    hp = CarSpec(power_w=900.0e3)
    p_hp = hp.power_w * hp.drivetrain_eta
    p_ref = CAR.power_w * CAR.drivetrain_eta
    assert hp.grip_force(hp.normal_load(20.0, CZ_A)) < p_hp / 20.0
    lap = lap_time(CZ_A, CD_A, CAR, LAP)
    v_slow = min(r["V_in"] for r in lap["segments"])
    assert CAR.grip_force(CAR.normal_load(v_slow, CZ_A)) > p_ref / v_slow

    def t(car, cz):
        return straight_profile(400.0, 20.0, 45.0, cz, CD_A, car)["t_s"]

    czs = (0.0, 0.5, 1.5, 3.0)
    hp_ts = [t(hp, c) for c in czs]
    ref_ts = [t(CAR, c) for c in czs]
    assert np.all(np.diff(hp_ts) < 0.0)
    assert (hp_ts[0] - hp_ts[-1]) > 10.0 * (ref_ts[0] - ref_ts[-1])


def test_a_pathological_coefficient_law_returns_a_reason_not_a_raise():
    """The in-contract failure path. It cannot be reached by the physics (the
    module docstring derives that), only by a caller's callable."""
    out = straight_profile(1000.0, 20.0, 30.0, 0.0, lambda V: 1e9, CAR)
    assert out["feasible"] is False
    assert "non-positive speed" in out["reason"]
    assert out["lap_time_s"] is None


@pytest.mark.parametrize("cz_a, cd_a, marker", [
    # a total drag area at or below zero: reachable in-contract, because cd_a
    # may be negative and CdA_car is finite
    (CZ_A, -CAR.cda_car_m2, "not positive"),
    (CZ_A, -1.0, "not positive"),
    (CZ_A, lambda V: -1.0, "still accelerating"),
    # coefficients that are not numbers
    (CZ_A, float("nan"), "not finite"),
    (float("nan"), CD_A, "not finite"),
    (float("inf"), CD_A, "not finite"),
    (CZ_A, "big", "must be a number"),
    (CZ_A, None, "must be a number"),
    # laws that return one
    (CZ_A, lambda V: float("nan"), "non-finite drag area"),
    (lambda V: float("nan"), CD_A, "non-finite downforce area"),
])
def test_the_lap_returns_a_reason_where_it_promises_not_to_raise(cz_a, cd_a,
                                                                 marker):
    """``lap_time`` says it never raises for an in-contract failure. Each row
    here RAISED out of it before: three of them out of the top-speed solve,
    the rest out of ``_as_law``'s finiteness guard, and one out of ``brentq``
    with a message about NaN function values.

    A degenerate coefficient is exactly what an optimiser hands a mission
    when its own solver has failed, so a raise here comes out of the middle
    of a search rather than as a refused design."""
    out = lap_time(cz_a, cd_a, CAR, LAP)
    assert out["feasible"] is False
    assert out["lap_time_s"] is None
    assert marker in out["reason"], out["reason"]


def test_the_only_thing_the_lap_raises_for_is_a_call_site_error():
    """``n_steps < 2`` is not a design a caller could have meant — it is a
    request for an integrator that cannot integrate — so it raises, and the
    docstring says so. Nothing about the COEFFICIENTS does."""
    with pytest.raises(ValueError, match="n_steps"):
        lap_time(CZ_A, CD_A, CAR, LAP, n_steps=1)
    with pytest.raises(ValueError, match="radius_m"):
        corner_speed(0.0, CZ_A, CD_A, CAR)
    # and the two dict-returning helpers keep the same promise as the lap
    assert corner_speed(50.0, float("nan"), CD_A, CAR)["feasible"] is False
    assert corner_speed(50.0, CZ_A, -1.0, CAR)["limited_by"] == "no_top_speed"
    assert straight_profile(100.0, 20.0, 30.0, CZ_A, -1.0, CAR)[
        "feasible"] is False


# --------------------------------------------------------------- the lap


def test_more_downforce_at_zero_drag_cost_is_never_slower():
    """The first reason a lap time may be an objective. Checked with constant
    friction and with a load-sensitive tyre, because k > 0 gives back part of
    the grip the downforce buys and could in principle reverse it — it cannot,
    for k < 1, which is exactly what the validated bound is protecting."""
    for car in (CAR, CarSpec(k_load=0.25)):
        ts = [lap_time(c, CD_A, car, LAP)["lap_time_s"]
              for c in np.linspace(0.0, 2.0, 15)]
        assert np.all(np.diff(ts) < 0.0), car
    # and it is not a rounding-scale effect: two m^2 of CzA is worth seconds
    t0 = lap_time(0.0, CD_A, CAR, LAP)["lap_time_s"]
    t2 = lap_time(2.0, CD_A, CAR, LAP)["lap_time_s"]
    assert t0 - t2 > 1.0


def test_more_drag_at_zero_downforce_is_never_faster():
    """The second reason, and the harder one: braking pulls the other way
    (measured directly above). So it is gated on the reference lap AND on a
    layout built to give braking every advantage — short straights between a
    fast corner and a hairpin."""
    braking = TrackSpec(name="braking-dominated",
                        segments=(Corner(radius_m=180.0, arc_m=200.0),
                                  Straight(length_m=60.0),
                                  Corner(radius_m=25.0, arc_m=60.0),
                                  Straight(length_m=400.0)))
    for track in (LAP, braking):
        ts = [lap_time(0.0, c, CAR, track)["lap_time_s"]
              for c in np.linspace(0.0, 0.4, 17)]
        assert np.all(np.diff(ts) > 0.0), track.name
        # ... and with downforce present too, which is the real design space
        ts = [lap_time(CZ_A, c, CAR, track)["lap_time_s"]
              for c in np.linspace(0.0, 0.4, 17)]
        assert np.all(np.diff(ts) > 0.0), track.name


def test_the_lap_time_is_step_independent_and_second_order():
    """The kink where the acceleration profile meets the braking profile is
    what sets the order, not RK4. Both the ORDER and the residual at the
    shipped resolution are measured here; the module quotes these numbers."""
    ns = [50, 100, 200, 400, 800, 1600]
    ts = [lap_time(CZ_A, CD_A, CAR, LAP, n_steps=n)["lap_time_s"] for n in ns]
    diffs = np.abs(np.diff(ts))
    ratios = diffs[:-1] / diffs[1:]
    assert np.all(ratios > 3.4) and np.all(ratios < 4.4), ratios
    # the shipped default, against a 8x finer grid
    t_def = lap_time(CZ_A, CD_A, CAR, LAP, n_steps=N_STEPS_DEFAULT)["lap_time_s"]
    assert abs(t_def / ts[-1] - 1.0) < 3e-4
    with pytest.raises(ValueError):
        straight_profile(100.0, 20.0, 20.0, CZ_A, CD_A, CAR, n_steps=1)


def test_a_constant_coefficient_and_a_callable_returning_it_are_one_lap():
    a = lap_time(CZ_A, CD_A, CAR, LAP)["lap_time_s"]
    b = lap_time(lambda V: CZ_A, lambda V: CD_A, CAR, LAP)["lap_time_s"]
    assert b == pytest.approx(a, rel=1e-12)


def test_the_breakdown_is_in_the_callers_order_and_sums_to_the_lap():
    """A lap that starts on a straight is rotated internally; the breakdown
    must come back indexed the way it was asked for, or a caller reading
    ``segments[0]`` gets a different segment than they wrote."""
    track = TrackSpec(segments=(Straight(length_m=600.0),
                                Corner(radius_m=40.0, arc_m=80.0),
                                Straight(length_m=900.0),
                                Corner(radius_m=120.0, arc_m=150.0)))
    lap = lap_time(CZ_A, CD_A, CAR, track)
    assert lap["feasible"], lap["reason"]
    kinds = [r["kind"] for r in lap["segments"]]
    assert kinds == ["straight", "corner", "straight", "corner"]
    assert [r["index"] for r in lap["segments"]] == [0, 1, 2, 3]
    assert lap["segments"][1]["radius_m"] == 40.0
    assert sum(r["t_s"] for r in lap["segments"]) == pytest.approx(
        lap["lap_time_s"], rel=1e-14)
    assert sum(r["length_m"] for r in lap["segments"]) == track.length_m
    assert lap["V_mean"] == pytest.approx(track.length_m / lap["lap_time_s"],
                                          rel=1e-14)


def test_a_lap_of_one_corner_is_its_arc_over_its_speed():
    """The degenerate closed form: no straight anywhere, so the lap time is
    exactly arc / V_corner and nothing about the integrator can enter it."""
    track = TrackSpec(segments=(Corner(radius_m=45.0, arc_m=282.7),))
    lap = lap_time(CZ_A, CD_A, CAR, track)
    V = corner_speed(45.0, CZ_A, CD_A, CAR)["V"]
    assert lap["lap_time_s"] == 282.7 / V
    # a corner is not adjacent to ITSELF, however the lap wraps
    assert lap["warnings"] == []


def test_adjacent_corners_are_allowed_and_flagged_as_the_transient_they_are():
    track = TrackSpec(segments=(Corner(radius_m=25.0, arc_m=40.0),
                                Corner(radius_m=200.0, arc_m=150.0),
                                Straight(length_m=700.0)))
    lap = lap_time(CZ_A, CD_A, CAR, track)
    assert lap["feasible"]
    assert any("instantaneously" in w for w in lap["warnings"])


def test_the_lap_ranks_what_a_drag_coefficient_allowance_could_not():
    """MISSION_DESIGN_WATER_TRACK.md measured that CD_budget = 0.11 admitted
    1857 of 1857 draws of the car-wing box — it decides nothing. The lap has
    to decide: two wings that both SATISFY that allowance, one with more
    downforce and more drag, must come back with different lap times, and the
    exchange rate has to be a property of the track and not a constant."""
    budget_cd_a = 0.11 * S_PUB
    cheap = (0.30, 0.35 * budget_cd_a)
    grippy = (0.95, budget_cd_a)
    t_cheap = lap_time(*cheap, CAR, LAP)["lap_time_s"]
    t_grippy = lap_time(*grippy, CAR, LAP)["lap_time_s"]
    assert abs(t_cheap - t_grippy) > 0.05        # the allowance cannot rank them
    # the exchange rate is the TRACK's: a layout of nothing but slow corners
    # and one of nothing but fast ones must not order the pair the same way
    slow = TrackSpec(name="slow", segments=(Corner(radius_m=22.0, arc_m=900.0),
                                            Straight(length_m=300.0)))
    fast = TrackSpec(name="fast", segments=(Corner(radius_m=400.0, arc_m=900.0),
                                            Straight(length_m=300.0)))
    d_slow = (lap_time(*cheap, CAR, slow)["lap_time_s"]
              - lap_time(*grippy, CAR, slow)["lap_time_s"])
    d_fast = (lap_time(*cheap, CAR, fast)["lap_time_s"]
              - lap_time(*grippy, CAR, fast)["lap_time_s"])
    assert d_slow * d_fast < 0.0, (d_slow, d_fast)


# ------------------------------------------------------- the multi-point part


def test_the_histogram_is_the_lap_and_not_a_second_one():
    lap = lap_time(CZ_A, CD_A, CAR, LAP)
    V, dt = speed_time_histogram(lap)
    assert float(np.sum(dt)) == pytest.approx(lap["lap_time_s"], rel=1e-13)
    assert V.min() >= lap["V_min"] * (1 - 1e-12)
    assert V.max() <= lap["V_max"] * (1 + 1e-12)
    # the mean is TIME weighted, so it must not equal the distance-weighted
    # average speed the lap also reports
    tw = time_weighted_mean_speed(lap)
    assert tw != lap["V_mean"]
    assert lap["V_min"] < tw < lap["V_max"]


@pytest.mark.parametrize("binning", ["speed", "time"])
def test_the_representative_points_are_a_distribution_over_the_lap(binning):
    out = representative_points(CZ_A, CD_A, CAR, LAP, n_points=4,
                                binning=binning)
    assert out["feasible"], out["reason"]
    pts = out["points"]
    assert 1 <= len(pts) <= 4
    assert sum(p["weight"] for p in pts) == pytest.approx(1.0, rel=1e-13)
    assert sum(p["time_s"] for p in pts) == pytest.approx(out["lap_time_s"],
                                                          rel=1e-13)
    # the weighted mean of the points IS the lap's time-weighted mean speed:
    # a set of points that does not preserve it is not a summary of this lap
    assert sum(p["weight"] * p["V"] for p in pts) == pytest.approx(
        out["V_mean_time_weighted"], rel=1e-12)
    assert [p["weight"] for p in pts] == sorted(
        (p["weight"] for p in pts), reverse=True)
    for p in pts:
        assert p["q_Pa"] == pytest.approx(0.5 * CAR.rho * p["V"] ** 2, rel=1e-14)
    with pytest.raises(ValueError):
        representative_points(CZ_A, CD_A, CAR, LAP, binning="quantile")


def test_the_two_binnings_disagree_about_the_weights_and_agree_about_the_mean():
    by_speed = representative_points(CZ_A, CD_A, CAR, LAP, n_points=4,
                                     binning="speed")
    by_time = representative_points(CZ_A, CD_A, CAR, LAP, n_points=4,
                                    binning="time")
    ws = sorted(p["weight"] for p in by_speed["points"])
    wt = sorted(p["weight"] for p in by_time["points"])
    assert max(ws) - min(ws) > 0.2               # the histogram says something
    assert max(wt) - min(wt) < 0.1               # the quantiles nearly cannot
    assert by_speed["V_mean_time_weighted"] == by_time["V_mean_time_weighted"]


def test_the_single_point_shortcut_is_free_when_nothing_varies():
    out = single_point_error(CZ_A, CD_A, CAR, LAP)
    assert out["rel_error"] == 0.0
    assert out["t_single_s"] == out["t_multi_s"]
    assert out["V_ref"] == time_weighted_mean_speed(lap_time(CZ_A, CD_A, CAR,
                                                             LAP))


def test_the_single_point_error_grows_with_the_wings_share_of_the_drag():
    """The mechanism the module states: the error is the product of the wing's
    share of total drag and the coefficient's variation. Both halves are
    swept, and the reference car's own number is re-derived rather than
    pinned."""
    ref = single_point_error(CZ_A, example_reynolds_cd_law(CD_A), CAR, LAP)
    assert 0.0 < abs(ref["rel_error"]) < 1e-5    # a fifth of a millisecond
    car = CarSpec(cda_car_m2=0.30, cza_car_m2=0.30)
    errs = [abs(single_point_error(0.60, example_reynolds_cd_law(c), car,
                                   LAP)["rel_error"])
            for c in (0.02, 0.10, 0.30)]
    assert errs[0] < errs[1] < errs[2]
    assert errs[2] > abs(ref["rel_error"])
    # a coefficient that really moved would cost an order of magnitude more
    strong = abs(single_point_error(
        0.60, example_reynolds_cd_law(0.30, exponent=0.5), car,
        LAP)["rel_error"])
    assert strong > 10.0 * errs[2]


def test_the_published_single_point_table_is_what_the_shipped_law_produces():
    """The tripwire on ``single_point_error``'s own five-row table.

    That table was measured with the example law's exponent at 0.042 while the
    shipped :data:`EXAMPLE_CD_EXPONENT` is 0.0412 — 1.0194x apart, showing up
    as rows 1.0193x too large. Nothing in the suite noticed, because every
    other assertion about the error is an inequality.

    Two gates, and the first is the re-derivation. To first order the error is
    PROPORTIONAL to the exponent, since the exponent is the whole reason the
    coefficient varies at all; so the reference row measured at the shipped
    constant has to equal the row measured at half of it, doubled — which ties
    the table to :data:`EXAMPLE_CD_EXPONENT` without writing the constant down
    a second time. The second gate is the five pinned rows: if the shipped
    exponent moves, the first gate still passes and these fail, which is
    exactly when the docstring needs re-measuring."""
    def err(cz, cd_ref, car, exponent=EXAMPLE_CD_EXPONENT):
        law = example_reynolds_cd_law(cd_ref, exponent=exponent)
        out = single_point_error(cz, law, car, LAP)
        assert out["feasible"], out["reason"]
        return out

    ref = err(CZ_A, CD_A, CAR)
    half = err(CZ_A, CD_A, CAR, exponent=0.5 * EXAMPLE_CD_EXPONENT)
    assert ref["rel_error"] / half["rel_error"] == pytest.approx(2.0, rel=1e-2)

    car = CarSpec(cda_car_m2=0.30, cza_car_m2=0.30)
    rows = [
        (ref, 3.467e-6, 0.000213),
        (err(0.60, 0.02, car), 7.685e-6, 0.000454),
        (err(0.60, 0.10, car), 3.815e-5, 0.002267),
        (err(0.60, 0.30, car), 1.123e-4, 0.006787),
        (err(0.60, 0.30, car, exponent=0.5), 1.236e-3, 0.074617),
    ]
    for out, rel_pub, delta_pub in rows:
        assert out["rel_error"] == pytest.approx(rel_pub, rel=2e-3)
        assert out["delta_s"] == pytest.approx(delta_pub, rel=2e-3)

    # the two ratios the docstring names as the mechanism
    assert rows[3][0]["rel_error"] / rows[0][0]["rel_error"] == pytest.approx(
        32.0, rel=0.05)
    assert rows[4][0]["rel_error"] / rows[3][0]["rel_error"] == pytest.approx(
        11.0, rel=0.05)


def test_the_example_law_reproduces_the_measurement_it_was_fitted_to():
    """MISSION_DESIGN_WATER_TRACK.md section 3 measured CD -2.591 % at 55 m/s
    against 30, and -4.784 % at 100 against 30, on `car rear wing +
    endplates`. The two points imply DIFFERENT exponents — the drift is not
    exactly a power law — so the gate is the bracket, re-derived here: the
    shipped exponent has to lie between the two single-point fits, and
    reproduce each measurement to a tenth of a point of CD."""
    r55, r100 = 1.0 - 0.02591, 1.0 - 0.04784
    p55 = -np.log(r55) / np.log(55.0 / 30.0)
    p100 = -np.log(r100) / np.log(100.0 / 30.0)
    assert min(p55, p100) < EXAMPLE_CD_EXPONENT < max(p55, p100)
    law = example_reynolds_cd_law(1.0, v_ref=30.0)
    assert law(55.0) - r55 == pytest.approx(0.0, abs=1.5e-3)
    assert law(100.0) - r100 == pytest.approx(0.0, abs=1.5e-3)
    assert law(30.0) == 1.0
    # the law is a pure scaling of its reference value, at every speed
    assert example_reynolds_cd_law(0.25)(80.0) == 0.25 * example_reynolds_cd_law(
        1.0)(80.0)


# ---------------------------------------------------------------- heave


def test_a_rigid_heave_law_is_bit_for_bit_inert():
    """Default OFF means default INVISIBLE: the stated height comes back
    untouched at every speed, and even a height below the law's own floor is
    left alone, because with a rigid car that is the caller's statement."""
    law = HeaveLaw()
    assert law.rigid
    for h0 in (0.30, 0.02):
        for V in (0.0, 20.0, 80.0):
            assert law.height_at(h0, CAR.q(V)) == h0
            assert CAR.ride_height_at(h0, V) == h0
    a = representative_points(CZ_A, CD_A, CAR, LAP, ride_height_m=0.30)
    assert all(p["ride_height_m"] == 0.30 for p in a["points"])
    assert a["heave"] is False


def test_the_heave_law_sinks_the_wing_and_stops_at_its_bump_stop():
    """Direction and closed form: h falls linearly in q until the stop."""
    k = 5.0e-5                      # m/Pa: 0.10 m at q = 2000 Pa
    car = CarSpec(heave=HeaveLaw(k_m_per_pa=k, h_min_m=0.18))
    assert car.ride_height_at(0.30, 0.0) == 0.30
    q40 = car.q(40.0)
    assert car.ride_height_at(0.30, 40.0) == pytest.approx(0.30 - k * q40,
                                                           rel=1e-14)
    hs = [car.ride_height_at(0.30, V) for V in (0.0, 20.0, 40.0, 60.0, 80.0)]
    assert np.all(np.diff(hs) <= 0.0)
    assert hs[-1] == 0.18                        # clamped at the bump stop
    # the clamp's ORDER: a car stated BELOW its own bump stop is left where it
    # was put, because a stated ride height is the caller's design variable
    below = CarSpec(heave=HeaveLaw(k_m_per_pa=k, h_min_m=0.20))
    assert [below.ride_height_at(0.15, V) for V in (0.0, 40.0, 80.0)] == [0.15] * 3
    pts = representative_points(CZ_A, CD_A, car, LAP, n_points=4,
                                ride_height_m=0.30)
    assert pts["heave"] is True
    fastest = max(pts["points"], key=lambda p: p["V"])
    slowest = min(pts["points"], key=lambda p: p["V"])
    assert fastest["ride_height_m"] < slowest["ride_height_m"]
    with pytest.raises(ValueError):
        HeaveLaw(k_m_per_pa=-1e-6)               # a car does not rise
    with pytest.raises(ValueError):
        HeaveLaw(h_min_m=0.0)


# ---------------------------------------------------------------- balance


def test_the_front_share_of_a_single_force_is_the_statics_identity():
    """Equation (5), checked against a moment balance written out here rather
    than against the module's own arithmetic. An OVERHUNG wing must come back
    NEGATIVE: it levers the front axle up."""
    car = CarSpec(cza_car_m2=0.0, x_wing_frac=1.15)
    out = aero_balance(1.0, car)
    L = car.wheelbase_m
    x = car.x_wing_frac * L
    assert out["aero_balance"] == pytest.approx((L - x) / L, rel=1e-14)
    assert out["aero_balance"] < 0.0
    # a force ON the front axle is all of it; on the rear axle, none of it
    assert aero_balance(1.0, CarSpec(cza_car_m2=0.0,
                                     x_wing_frac=0.0))["aero_balance"] == 1.0
    assert aero_balance(1.0, CarSpec(cza_car_m2=0.0,
                                     x_wing_frac=1.0))["aero_balance"] == 0.0


def test_the_aero_balance_does_not_move_with_speed_when_the_wing_does_not():
    """q cancels out of equation (6) exactly. This is what makes the heave
    law (or a Reynolds-dependent CZ) the ONLY mechanism that can move it, and
    it is asserted with == because 'nearly cancels' would hide a q left in."""
    a = aero_balance(CZ_A, CAR, V=10.0)["aero_balance"]
    b = aero_balance(CZ_A, CAR, V=90.0)["aero_balance"]
    c = aero_balance(CZ_A, CAR)["aero_balance"]
    assert a == b == c
    # the LOAD balance does move: static at rest, towards the aero balance
    # as q grows
    assert aero_balance(CZ_A, CAR, V=1e-6)["load_balance"] == pytest.approx(
        CAR.front_weight_frac, rel=1e-6)
    far = aero_balance(CZ_A, CAR, V=1e5)["load_balance"]
    assert far == pytest.approx(a, rel=1e-6)
    mid = aero_balance(CZ_A, CAR, V=55.0)["load_balance"]
    assert a < mid < CAR.front_weight_frac


def test_a_rear_wing_drives_the_balance_rearward_monotonically():
    bals = [aero_balance(c, CAR)["aero_balance"] for c in np.linspace(0.0, 1.5, 13)]
    assert np.all(np.diff(bals) < 0.0)
    # ... and a wing mounted exactly where the car's own downforce acts moves
    # it not at all: the closed-form limit of the same identity
    car = CarSpec(x_wing_frac=CAR.x_cp_car_frac)
    base = aero_balance(0.0, car)["aero_balance"]
    assert aero_balance(3.0, car)["aero_balance"] == pytest.approx(base,
                                                                   rel=1e-14)


def test_the_default_balance_window_is_the_speed_invariant_one():
    """Equation (7): the front/rear grip split is speed-independent iff the
    aero balance equals the static front weight fraction, and the
    load-sensitivity exponent cancels out of that condition entirely. Both
    halves are checked — the window's centre, and the cancellation."""
    lo, hi = CAR.balance_window_used()
    assert (lo + hi) / 2.0 == pytest.approx(CAR.front_weight_frac, rel=1e-14)
    assert hi - lo == pytest.approx(2.0 * BALANCE_HALF_WIDTH, rel=1e-14)
    assert CarSpec(balance_window=(0.3, 0.4)).balance_window_used() == (0.3, 0.4)

    # the cancellation, measured: put a car exactly ON equation (7) and check
    # that its front grip share is the same at rest and at speed, for k = 0
    # AND for k > 0
    czw = 0.5
    car0 = CarSpec(cza_car_m2=0.0, x_wing_frac=1.0 - CarSpec().front_weight_frac)
    assert aero_balance(czw, car0)["aero_balance"] == pytest.approx(
        car0.front_weight_frac, rel=1e-14)
    for k in (0.0, 0.2):
        car = CarSpec(cza_car_m2=0.0, k_load=k,
                      x_wing_frac=1.0 - CarSpec().front_weight_frac)
        shares = []
        for V in (1e-6, 40.0, 90.0):
            q = car.q(V)
            n_f = car.weight_n * car.front_weight_frac + q * czw * (
                1.0 - car.x_wing_frac)
            n_r = car.weight_n * (1.0 - car.front_weight_frac) + q * czw * (
                car.x_wing_frac)
            g_f, g_r = car.grip_force(n_f), car.grip_force(n_r)
            shares.append(g_f / (g_f + g_r))
        assert shares[0] == pytest.approx(shares[-1], rel=1e-12), k


def test_the_cancellation_in_equation_7_belongs_to_the_power_law():
    """The narrow reading of (7), which the module docstring used to overstate.

    The exponent cancelling is not a property of load sensitivity. It is a
    property of the SHAPE this module assumed: the grip force ``mu0 N_ref^k
    N^(1-k)`` is homogeneous of degree ``1-k``, so a factor common to
    ``N_f`` and ``N_r`` leaves their ratio alone. The published form is affine
    in the load, whose grip force is quadratic in N and does NOT have that
    property.

    Both halves are measured on the SAME car — one exactly on equation (7),
    with the two friction laws matched at ``N_ref`` — so the only difference
    is the shape. The power law holds the front grip share bit-for-bit
    constant across the speed range; the affine one moves it by 4.4e-4, four
    hundredths of a point of balance. Small, and not zero, which is the whole
    claim."""
    czw = 0.5
    k = K_LOAD_REFERENCE_LATERAL
    car = CarSpec(cza_car_m2=0.0, k_load=k,
                  x_wing_frac=1.0 - CarSpec().front_weight_frac)
    assert aero_balance(czw, car)["aero_balance"] == pytest.approx(
        car.front_weight_frac, rel=1e-14)          # the car IS on (7)

    def front_grip_shares(mu_of_N):
        out = []
        for V in (1e-6, 40.0, 90.0):
            q = car.q(V)
            n_f = car.weight_n * car.front_weight_frac + q * czw * (
                1.0 - car.x_wing_frac)
            n_r = car.weight_n * (1.0 - car.front_weight_frac) + q * czw * (
                car.x_wing_frac)
            g_f, g_r = mu_of_N(n_f) * n_f, mu_of_N(n_r) * n_r
            out.append(g_f / (g_f + g_r))
        return out

    power = front_grip_shares(lambda N: car.mu0 * (N / car.n_ref) ** (-k))
    # matched at N_ref in value AND in log-slope, so nothing but the shape
    # separates the two laws
    affine = front_grip_shares(
        lambda N: car.mu0 * (1.0 - k * (N / car.n_ref - 1.0)))
    assert max(power) - min(power) == 0.0
    assert max(affine) - min(affine) == pytest.approx(4.4e-4, rel=0.1)
    # the two laws are close but not equal even at rest, because the AXLES do
    # not sit at N_ref (the car is 53/47) and that is already off the point
    # where the shapes were matched — 0.4 %, an order above the speed drift,
    # which is the same "a local exponent is local" statement in miniature
    assert power[0] == pytest.approx(affine[0], rel=5e-3)


def test_the_balance_margin_says_which_end_binds():
    """Two margins, not one: 'too rearward' and 'too forward' are different
    instructions and a collapsed scalar cannot tell them apart."""
    # the reference car with NO wing is too FORWARD of the window
    bare = balance_margin(0.0, CAR)
    assert bare["g_high"] < 0.0 and bare["g_low"] > 0.0
    assert bare["binding"] == "high" and bare["g"] == bare["g_high"]
    # a large wing drives it out the other end
    big = balance_margin(1.0, CAR)
    assert big["g_low"] < 0.0 and big["g_high"] > 0.0
    assert big["binding"] == "low" and big["g"] == big["g_low"]
    # and there is a wing in between that satisfies it — solved here from the
    # identity, not read off the module
    lo, hi = CAR.balance_window_used()
    czc, fc = CAR.cza_car_m2, 1.0 - CAR.x_cp_car_frac
    fw = 1.0 - CAR.x_wing_frac
    def cz_at(b):                                   # from equation (6)
        return czc * (fc - b) / (b - fw)

    inside = 0.5 * (cz_at(lo) + cz_at(hi))
    ok = balance_margin(inside, CAR)
    assert ok["binding"] is None and ok["g"] >= 0.0
    assert lo <= ok["aero_balance"] <= hi
    # the balance FALLS with cz_a, so the window maps to a cz_a interval with
    # the ends swapped: [cz_at(hi), cz_at(lo)] = [0.0329, 0.2286] m^2 here.
    # The published wing (0.2623 m^2) sits just ABOVE it — a real finding, and
    # the reason this margin exists at all
    assert cz_at(hi) < cz_at(lo) < CZ_A
    assert balance_margin(CZ_A, CAR)["binding"] == "low"
    assert -0.05 < balance_margin(CZ_A, CAR)["g_low"] < 0.0   # only just


def test_the_balance_refuses_the_two_questions_it_cannot_answer():
    """Both are in-contract failures with reasons, not exceptions."""
    nowt = balance_margin(0.0, CarSpec(cza_car_m2=0.0))
    assert nowt["feasible"] is False
    assert "no net downforce" in nowt["reason"]
    assert nowt["g"] is None
    speed_law = aero_balance(lambda V: 0.5 * (V / 55.0) ** 0.1, CAR)
    assert speed_law["feasible"] is False
    assert "has to be asked AT a speed" in speed_law["reason"]
    assert aero_balance(lambda V: 0.5, CAR, V=55.0)["feasible"] is True


def test_representative_points_reports_a_lap_that_did_not_close():
    out = representative_points(0.0, lambda V: 1e9, CAR, LAP)
    assert out["feasible"] is False
    assert "no lap to sample" in out["reason"]
