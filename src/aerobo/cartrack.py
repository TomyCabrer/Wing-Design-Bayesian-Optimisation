"""The car wing's MISSION: a lap, in place of a drag-coefficient calibration.

Why a track spec at all — the same argument mission.py makes for air
-------------------------------------------------------------------
mission.py exists because "maximise L/D at CL = 0.5" is an arbitrary
target until a weight and a speed are stated, at which point it becomes
EXACTLY "minimise the drag of sustaining W at V". carwing.py is in the
position mission.py's docstring describes as the one before the mission
layer: it maximises a downforce COEFFICIENT against ``CD_budget = 0.11``,
and that budget's own docstring calls itself an allowance. The design
document MISSION_DESIGN_WATER_TRACK.md measured what the allowance does:
over 1857 geometrically-feasible draws of the car-wing box at 55 m/s it
admitted **1857 of 1857** — it decides nothing — while a two-point
requirement pair admitted 5. A calibration that refuses nothing is not a
budget, it is a constant.

The physical task a rear wing is bought for is a LAP. Downforce is not
free: every newton of it is bought with drag on the straights, and the
exchange rate is not a number a designer can state — it is an integral
over the circuit. So the equivalent of mission.py's "sustain W at V with
minimum drag" is here

    lap_time(cz_a, cd_a; car, track)  ->  seconds,

with ``cz_a`` and ``cd_a`` the WING's ``CZ*S`` and ``CD*S`` in m^2, added
to a REST-OF-CAR baseline that must be declared. Minimising it is the
objective; everything else in this module exists to make that number
honest, or to say when it is not.

This is item **E5/E8** of MISSION_DESIGN_WATER_TRACK.md section 4, which
deferred exactly two things: evaluating the wing at more than one point
of the lap (E5), and a car FRAME in which a balance can be computed (E8).
Both are here, and both are answered the way that document argues they
must be — E5 by evaluating at the mission's own speeds rather than
referring one point to another by a q ratio, E8 by making the frame a
STATED property of :class:`CarSpec` rather than something a rear wing
could ever derive on its own.

The quasi-steady point-mass lap
-------------------------------
The car is a point mass at one normal load. A lap is an ordered, CYCLIC
list of segments: corners (radius, arc length) taken at constant speed,
and straights on which the car accelerates and then brakes.

**Corner speed — derived.** Flat, unbanked corner of radius R. The
vertical load on the tyres is the weight plus the aerodynamic downforce,

    N(V) = m g + q (CzA_car + cz_a),      q = 1/2 rho V^2

and the lateral force available is the friction law below applied to that
load. Steady cornering demands the centripetal force m V^2 / R, so

    mu_eff(N) N = m V^2 / R.                                    (1)

With CONSTANT friction (k = 0, the default) and constant coefficients
this is linear in V^2 and solves in closed form:

    mu (m g + 1/2 rho V^2 CzA_tot) = m V^2 / R
    mu m g = V^2 (m/R - 1/2 mu rho CzA_tot)
    V_corner^2 = mu m g / (m/R - 1/2 mu rho CzA_tot).           (2)

At CzA_tot = 0 that is V^2 = mu g R, the skidpad — the closed-form limit
this module is gated against.

**The aero-unbounded case is a statement, not a number.** The denominator
of (2) is the corner's demand rate minus the downforce's grip rate, both
per unit V^2. When ``1/2 mu rho CzA_tot >= m/R`` the aero adds lateral
capacity at least as fast as the corner consumes it, and (2) has no
positive root: grip NEVER runs out. The honest reading is that this
corner does not limit this car — the limit is somewhere else (top speed,
or the driver, or the track's width). :func:`corner_speed` returns
``limited_by='aero_unbounded'`` with a reason and caps the speed at the
car's top speed. It does not return a large number and let it propagate.
For the reference car this happens at ``R >= m / (1/2 mu rho CzA_tot)``,
which is measured in the tests.

**Load sensitivity.** A real tyre's friction coefficient FALLS as its
normal load rises; the module carries it as

    mu_eff(N) = mu0 (N / N_ref)^(-k),     k >= 0,               (3)

so the grip FORCE is ``mu0 N_ref^k N^(1-k)``. ``k = 0`` is the default
and means constant friction; ``N_ref`` defaults to the car's static
weight ``m g``, so ``mu0`` is then the friction at rest and ``k`` describes
only how much of the aero load is given back. ``k`` is required to be < 1,
and that is a derivation rather than taste: the grip force goes as
``N^(1-k)``, so ``k >= 1`` would mean a tyre that loses absolute grip when
you push down on it, which no tyre does.

**Where k comes from — and what it is not.** That the coefficient falls
with load at all, and roughly how big the effect is, is measured and
citable: Cabrera, Castillo, Perez, Velasco, Guerra and Hernandez, "A
Procedure for Determining Tire-Road Friction Characteristics Using a
Modification of the Magic Formula Based on Experimental Results", Sensors
18(3), 2018, article 896, DOI 10.3390/s18030896 (open access), Table 2,
Hankook 205/65 R15 — carried here as
:data:`CABRERA_2018_HANKOOK_205_65_R15`. What is published there is a
LINEAR law, its equations (4) and (14),

    mu = (PD1 + PD2 dfz) lambda,      dfz = (Fz - Fz0) / Fz0,

i.e. the friction at a nominal load plus a slope in fractional load change.
Two things follow, and both have to be said out loud.

First, (3) is a power law and the published one is affine in the load. They
are different shapes; matched at ``N_ref`` they agree only to first order,
and over the load range the reference lap actually visits — ``N/N_ref``
from 1 at rest to 1.3854 at the fastest point of the fastest straight —
they finish 1.23 % of mu apart on the longitudinal pair and 0.97 % apart on
the lateral one (measured in tests/test_cartrack.py). The power law was
chosen anyway, and for a stated reason: it is homogeneous, ``N^(1-k)``, so
the exponent cancels out of the balance condition (7) below EXACTLY, which
an affine law does not do. That is a modelling choice with a price, not a
finding, and the price is the 1.23 % just quoted.

Second, no numeric exponent here is quoted from anyone. Reading a value of
``k`` off a linear law is arithmetic this repo did: ``k`` is the local
logarithmic slope ``-d ln mu / d ln N`` at the reference load, which for an
affine law is exactly ``-PD2 / PD1``. Evaluated on the four coefficients
above that is 0.168 longitudinally and 0.137 laterally, for one tested tyre
at its own temperature and pressure on its own road, and those two numbers
are **THIS REPO'S OWN CALIBRATION** — they are a reference ORDER for a
caller who has no tyre data of their own, they are re-derived in code as
:data:`K_LOAD_REFERENCE_LONGITUDINAL` and
:data:`K_LOAD_REFERENCE_LATERAL`, and they are NOT the default. ``k_load``
defaults to 0, because this package has no tyre and a made-up load
sensitivity is worse than none.

With ``k > 0`` equation (1) is no longer closed-form, and something
falls out of it that is true as algebra and misleading as engineering:
the grip force then grows as ``V^(2(1-k))`` while the corner's demand
grows as ``V^2``, so the demand always wins in the end and a
load-sensitive tyre has a finite corner speed at EVERY radius. But "in
the end" can be arbitrarily far away. Measured on the reference car at
``k = 0.05`` and ``R = 5000 m``, the root sits at **3.46e9 m/s** — so
load sensitivity removes the aero-unbounded case as a matter of algebra
and not as a matter of engineering, and this module treats the two the
same way: it scans to 1.5x the car's top speed, and reports
``aero_unbounded`` when nothing crosses, which means "grip does not run
out at any speed this car can reach". Uniqueness of the root is also
derived: the
residual ``g(V) = mu0 N_ref^k N(V)^(1-k) - m V^2 / R`` has
``g'(V) = V [mu0 N_ref^k (1-k) N^(-k) rho CzA_tot - 2 m / R]``, whose
bracket is monotone decreasing in V, so g rises then falls and — with
``g(0) > 0`` and ``g(inf) = -inf`` — crosses zero exactly once. A
speed-DEPENDENT ``cz_a`` law breaks that argument, so there the module
takes the FIRST sign change on a scan and says so.

**Top speed — derived.** Power at the wheels against drag:

    P_eff = P eta = D V = q CdA_tot V = 1/2 rho CdA_tot V^3
    V_top = (2 P_eff / (rho CdA_tot))^(1/3).                    (4)

Rolling resistance is NOT in (4) (see the omissions), so V_top is a
closed-form cube root and the drag-vs-speed gate is exact.

**Straights.** Integrated in the SPACE domain, which is what turns a
two-point boundary problem (leave corner i at its speed, arrive at corner
i+1 at its speed) into two initial-value problems:

    dV/dx = a(V) / V,   t = int dx / V

with, accelerating,

    a_acc(V) = ( min(F_power(V), F_grip(V)) - D(V) ) / m
    F_power = P_eff / V,  F_grip = mu_eff(N) N,  D = q CdA_tot

and, braking, the grip limit again with drag HELPING (it does; a car
stops shorter at speed than the tyres alone explain):

    a_brk(V) = -( F_grip(V) + D(V) ) / m.

Both grip terms depend on the downforce, which is the whole point: a
wing shortens the braking zone and lengthens the traction-limited part of
the exit, and neither effect exists in a CD budget. The scheme is
classical RK4 in x, forward from the entry speed and BACKWARD from the
exit speed, with

    V(x) = min( V_acc(x), V_brake(x), V_top )

and the time by the trapezoidal rule on 1/V. The braking point is
therefore located by the crossing of the two profiles rather than
guessed. Convergence is limited by that crossing, not by RK4: V(x) has a
kink there, so the time integral converges at SECOND order in the step —
measured on the reference lap, halving the step divides the change by
3.63, 3.79, 3.86, 3.98 over n = 50 to 1600, i.e. the ratio 4 that h^2
predicts. At the default n = 200 per straight the lap time is 61.3717 s
against 61.3602 s at n = 1600, a relative error of 1.9e-4. The tests
measure the order and the residual rather than asserting a tolerance
nobody checked.

**When the straight is too short to slow down.** If the backward braking
profile is already below the entry speed at the start of the straight,
the car cannot have left the previous corner that fast — properly, the
previous corner's speed is then set by the next corner, and corner speeds
become a fixed point. That backward propagation is NOT modelled. The
segment is flagged ``brake_limited_entry`` with the implied entry speed,
and the lap time is computed from the profile that is consistent with
ARRIVING correctly (the car is assumed to have already been slow in the
corner), which understates the lap time. It is reported, never silent.

Bias direction of the whole model: **OPTIMISTIC**. Every simplification
below removes a reason the real car is slower — instantaneous transitions
between braking and cornering, full lateral grip the moment the corner
starts, no combined-slip ellipse, no gear shifts, no rolling resistance,
no tyre degradation, all-wheel traction. The absolute lap time is
therefore not a prediction. It is a RANKING instrument for wings on ONE
car and ONE track, where the shared optimism largely cancels; the tests
gate the rankings (more downforce is never slower, more drag is never
faster), not the seconds.

Evaluating the wing at the lap's own speeds (E5)
------------------------------------------------
MISSION_DESIGN_WATER_TRACK.md section 3 measured that on ``car rear
wing`` the coefficients are bit-identical across 30-100 m/s, so a
single-point evaluation is exact there — and that on ``car rear wing +
endplates``, whose plate friction is Reynolds-based, CD drifts -2.591 %
at 55 m/s relative to 30 and -4.784 % at 100. A single power-law fit to
those two points gives an exponent of -0.043 and -0.041 respectively, i.e.
``CD ~ V^-0.042`` over that range, consistent with about a fifth of the
drag being turbulent plate friction going as ``Re^-0.2``. That law is
used as the module's stated example of a speed-dependent coefficient
(:func:`example_reynolds_cd_law`) and nothing else; it is a fit to this
repo's own measurement, not a literature correlation.

So :func:`lap_time` accepts each coefficient as EITHER a float or a
callable of speed, and :func:`representative_points` hands a caller the
speeds the lap actually spends its time at, with those time weights, so
the wing can be evaluated at each. :func:`single_point_error` measures
what the shortcut costs for a given law rather than asserting it is
small — and what it costs on the REFERENCE car, under that law, is
3.47e-6 of lap time (0.21 ms in 61 s), because the wing's 0.0095 m^2 of
drag is one percent of the car's 0.90 m^2 and a four-percent drift on one
percent is nothing. That is a finding, not a disappointment: it says the
single-point shortcut is safe HERE and names the two quantities that
would make it unsafe elsewhere (the wing's share of the drag, and the
coefficient's variation), both tabulated on :func:`single_point_error`.

Ride height is not constant on a lap (the optional heave law)
-------------------------------------------------------------
A rear wing's height above the track falls with dynamic pressure, because
the suspension and the mounts deflect under the downforce they are
carrying. carwing.py makes ``ride_height_m`` a design variable and the
ground image is built at that height, so a lap that changes it changes
the aerodynamics. The law offered is the simplest one that is a
STATEMENT rather than a fit:

    h(V) = clamp( h0 - k_h q,  h_min, h0 )

with ``k_h`` in m/Pa — an inverse stiffness, and again a CALIBRATION the
caller owns, defaulting to 0 = rigid. At ``k_h = 0`` the law is
bit-for-bit inert (it short-circuits and returns the stated height; the
clamp is not applied, because with a rigid car a stated height below the
floor is the caller's statement and not this module's to correct).

Balance, and why it needs a frame (E8)
--------------------------------------
E8's argument for deferring balance was that "the rear wing's
contribution to the CoP is dominated by its mounting station, which is
not in the design vector" — a balance window would be a mission statement
over a frame that does not exist. So the frame is STATED, on
:class:`CarSpec`: a wheelbase, a CG fraction, the rest of the car's own
downforce AND the station that downforce acts at, and the wing's own
station. None of it is derived, and none of it could be: **a rear wing
alone cannot compute the front axle's share of anything**, because the
share is a ratio and it has no access to the denominator.

Given the frame, the arithmetic is a statics identity. Take x measured
from the front-axle contact patch, rearwards, as a fraction of the
wheelbase. Moments about the REAR axle for a single downward force F at
station x:

    F_front L = F (L - x)   =>   F_front / F = 1 - x,          (5)

so a wing OVERHUNG behind the rear axle (x > 1) has a NEGATIVE front
share: it levers the front axle up. That is correct and it is the reason
a rear wing on its own drives the balance rearward. Summing (5) over the
car's own downforce and the wing's:

    aero_balance = [ CzA_car (1 - x_cp_car) + cz_a (1 - x_wing) ]
                   / ( CzA_car + cz_a ).                        (6)

Note what (6) does not contain: q. The aero balance is speed-INDEPENDENT
whenever the coefficients are, which is exactly why a heave law (or a
Reynolds-dependent CD) is the thing that makes it move, and why
:func:`aero_balance` accepts the same float-or-callable coefficients as
the lap.

The DEFAULT window is derived rather than chosen. Front and rear grip are
``mu0 N_ref^k N_f^(1-k)`` and ``mu0 N_ref^k N_r^(1-k)``, so the front's
share of total grip depends on speed only through ``N_f / N_r``; that
ratio is constant in speed iff the aero downforce splits in the same
proportion as the static weight, i.e. iff

    aero_balance = 1 - x_cg     (the static front weight fraction).  (7)

The load-sensitivity exponent cancels out of (7) entirely, and it is worth
being precise about WHY, because the reason is narrower than it looks. It
is not that load sensitivity does not matter to balance. It is that the
particular law this module assumed, equation (3), makes the grip force
``mu0 N_ref^k N^(1-k)`` — HOMOGENEOUS of degree ``1-k`` in N — so a common
factor in ``N_f`` and ``N_r`` leaves ``N_f^(1-k) / N_r^(1-k)`` alone and k
drops out. An affine law does not have that property: ``mu = PD1 + PD2 dfz``
gives a grip force quadratic in N, whose front/rear ratio moves with the
load level, so (7) would be the leading term of a condition rather than the
condition. (7) is therefore exact under THIS module's friction law and
approximate under the published one — which is the price named for the
modelling choice at equation (3), showing up a second time. A car
satisfying (7) has handling that does not change with speed, to that
accuracy. The default window is (7) plus/minus a half-width which IS a
calibration the caller owns (:data:`BALANCE_HALF_WIDTH`, 0.05 = five
points of balance), and the reference car does NOT sit inside it with the
published wing — reported in the tests rather than tuned away, because
"a rear wing alone drives the balance rearward" is the finding, not a bug.

What is NOT modelled
--------------------
* TRANSIENTS of every kind: corner entry and exit are instantaneous
  changes of state, lateral force appears the moment the arc does, and
  there is no combined-slip friction ellipse — the car brakes at 100 % of
  the circle up to the geometric start of the corner and turns at 100 % of
  it thereafter. Optimistic.
* TYRE THERMAL state, pressure, wear, and any variation of mu around the
  lap or between axles. One mu0 and one k for the whole car. Optimistic.
* GEAR SHIFTS, torque curve, launch, clutch: the engine is a constant
  power source through a constant efficiency, available at every speed.
  Optimistic on acceleration, and it is also why F_power blows up as
  V -> 0 (the grip limit is what actually caps traction at low speed).
* ROLLING RESISTANCE and driveline drag beyond the single efficiency.
  Optimistic on top speed and on acceleration.
* WEIGHT TRANSFER, longitudinal or lateral. A point mass carries one
  normal load, so with k > 0 the grip is that of a car whose four tyres
  are always equally loaded — which is the best case. Optimistic. It is
  also why (7) is exact here and only approximate on a real car.
* WHICH WHEELS DRIVE. Traction-limited acceleration uses the whole car's
  grip; a two-wheel-drive car has less. Optimistic.
* THE DRIVER: no line choice, no reaction time, no error, no lift-and-
  coast, no fuel or tyre management. Optimistic.
* BANKING, camber, ELEVATION change, kerbs, surface grip variation, wind.
  Banking and downhill braking would both RAISE the modelled speeds, so
  omitting them is conservative; a flat model of a banked circuit is the
  one direction of error this module has that is not optimistic.
* THE INTERACTION between the wing and the rest of the car's aero map.
  ``CzA_car`` and ``CdA_car`` are constants ADDED to the wing's, so the
  wing is assumed not to change the floor's or the diffuser's work. On a
  real car it does — that coupling is the single largest reason a rear
  wing is fitted at the height it is — and it needs an aero map this
  package does not carry. The additivity is stated on :class:`CarSpec`
  and is the reason the wing's contribution is separately visible at all.
* FUEL BURN, so the mass is constant around the lap.
* SEGMENT-LEVEL geometry: a corner is a radius and an arc, with no
  entry/exit radius variation, no camber, and no track width to use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import brentq

from .carwing import RHO_AIR
from .mission import G0

#: half-width of the DEFAULT aero-balance window, in balance fraction. A
#: CALIBRATION and nothing else: equation (7) of the module docstring derives
#: where the window's CENTRE must be (the static front weight fraction, so the
#: front/rear grip split does not move with speed), and says nothing about how
#: far a car may sit from it before the driver minds. 0.05 = five points of
#: balance. Owned by the caller through ``CarSpec.balance_window``.
BALANCE_HALF_WIDTH = 0.05

#: default number of RK4 steps per straight. The step is L/n, so the reference
#: lap's longest straight (1400 m) is integrated at 7 m. Chosen against the
#: measured convergence in tests/test_cartrack.py, not by taste.
N_STEPS_DEFAULT = 200

#: speeds scanned when a corner's equation has to be solved numerically (a
#: load-sensitive tyre, or a speed-dependent coefficient law). The scan finds
#: the FIRST sign change; two roots closer together than V_top/this are not
#: resolved, which cannot happen for the constant-coefficient case (the module
#: docstring derives that its root is unique).
N_SCAN_DEFAULT = 256

#: how far :func:`top_speed_result` may expand its bracket when the drag law is
#: a CALLABLE, as a number of DOUBLINGS from 1 m/s. This is not a cap on the
#: answer — the bracket grows until it contains the root, so the callable path
#: answers wherever the closed form does — it is only what makes the search
#: terminate on a law that never crosses. 64 doublings reach 1.8e19 m/s; a law
#: with no crossing below that is not describing a car, and the failure names
#: the total drag area it found up there, which is what a non-positive or
#: vanishing drag area looks like from inside the search.
N_DOUBLINGS_MAX = 64

#: the four load coefficients of the Magic Formula's peak-friction term for
#: ONE tested tyre, transcribed from Cabrera, Castillo, Perez, Velasco, Guerra
#: and Hernandez, "A Procedure for Determining Tire-Road Friction
#: Characteristics Using a Modification of the Magic Formula Based on
#: Experimental Results", Sensors 18(3), 2018, article 896,
#: DOI 10.3390/s18030896 (open access), Table 2, Hankook 205/65 R15.
#:
#: They are the coefficients of a LINEAR law — the paper's equations (4) and
#: (14) read ``mu = (PD1 + PD2 dfz) lambda`` with ``dfz = (Fz - Fz0)/Fz0``, so
#: PD1 is the friction at the nominal load and PD2 its slope in fractional
#: load change. ``X`` is longitudinal (braking and traction), ``Y`` lateral
#: (cornering); this module carries ONE mu for both, which is an omission
#: listed below.
#:
#: These are the only numbers in this module taken from anywhere outside it.
#: They belong to that one tyre at its own temperature and pressure on its own
#: road, and they are here to fix the EXISTENCE, the SIGN and the ORDER of
#: load sensitivity — nothing else about it is cited, and nothing here is a
#: general property of tyres.
CABRERA_2018_HANKOOK_205_65_R15 = {
    "PDX1": 1.10206790,
    "PDX2": -0.18524061,
    "PDY1": 0.932775,
    "PDY2": -0.128085,
}


def _k_load_from_affine_slope(pd1: float, pd2: float) -> float:
    """``-PD2 / PD1`` — the local exponent an affine friction law implies.

    THIS REPO'S OWN CALIBRATION, and the arithmetic that makes it one. A power
    law ``mu = mu0 (N/N_ref)^-k`` has ``-d ln mu / d ln N = k`` everywhere; an
    affine law ``mu = PD1 (1 + (PD2/PD1) dfz)`` with ``dfz = N/N_ref - 1`` has

        -d ln mu / d ln N  =  -(N/mu) dmu/dN  =  -PD2 / PD1     at N = N_ref,

    because at the reference load ``mu = PD1`` and ``dmu/dN = PD2 / N_ref``.
    So the two laws share a slope at ONE point, and this is the value of k
    that makes them share it.

    What is NOT modelled by that number: anything away from ``N_ref``. The
    exponent is a LOCAL slope, and the module docstring measures what the two
    shapes cost each other over the reference lap's own load range (1.23 % of
    mu at ``N/N_ref = 1.3854``). Integrating a local exponent as if it were a
    global one is a mistake this repo has made before and now measures.
    """
    return -float(pd2) / float(pd1)


#: reference ORDER for the load-sensitivity exponent ``k`` of equation (3),
#: longitudinal and lateral. **THIS REPO'S OWN CALIBRATION**: the value is
#: computed here, by :func:`_k_load_from_affine_slope`, from the four
#: coefficients of :data:`CABRERA_2018_HANKOOK_205_65_R15`, and no exponent is
#: taken from anywhere.
#:
#: They exist so that a caller with no tyre data of their own has an order to
#: reason with instead of a guess. They are NOT the default and must not
#: become one: ``CarSpec.k_load`` defaults to 0, and a calibration is a
#: default only when the thing it calibrates is present, which here it is not
#: (this package has no tyre).
K_LOAD_REFERENCE_LONGITUDINAL = _k_load_from_affine_slope(
    CABRERA_2018_HANKOOK_205_65_R15["PDX1"],
    CABRERA_2018_HANKOOK_205_65_R15["PDX2"])
K_LOAD_REFERENCE_LATERAL = _k_load_from_affine_slope(
    CABRERA_2018_HANKOOK_205_65_R15["PDY1"],
    CABRERA_2018_HANKOOK_205_65_R15["PDY2"])


def _fail(reason: str, **extra) -> dict:
    """In-contract failure, the package's contract: a reason, never a raise."""
    d = {"feasible": False, "reason": reason, "lap_time_s": None}
    d.update(extra)
    return d


Coefficient = float | Callable[[float], float]


def _coefficient_reason(value, name: str) -> str:
    """``""`` if ``value`` can serve as a coefficient, else the reason it cannot.

    This is the guard that keeps the module's ``_fail`` contract: every public
    entry point runs it BEFORE :func:`_as_law` — directly, or through the one
    it delegates to (:func:`representative_points` and
    :func:`single_point_error` go through :func:`lap_time`,
    :func:`balance_margin` through :func:`aero_balance`) — so a coefficient
    that is not a number comes back as a reason rather than as an exception
    out of the middle of a solve.

    A float has to be finite. A CALLABLE is deliberately not sampled here — a
    law is only known where it is asked, and sampling it at a speed the lap
    never visits would refuse laws that are perfectly usable. Its values are
    checked at every speed the solvers actually evaluate it at instead, and the
    reason then names that speed.
    """
    if callable(value):
        return ""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return (f"{name} must be a number in m^2 or a callable of speed, got "
                f"{value!r}")
    if not np.isfinite(v):
        return (f"{name} = {value!r} is not finite, so there is nothing here "
                f"to compute: a coefficient that is NaN or infinite is an "
                f"answer some solver failed to produce, not a design to fly")
    return ""


def _as_law(value: Coefficient, name: str) -> tuple[Callable[[float], float], bool]:
    """``(law(V), is_constant)`` for a coefficient given as a float or callable.

    The constancy flag is taken from the TYPE, not sniffed by sampling: a
    caller who hands over a callable gets the numeric path even if the
    callable happens to be constant, and the tests use exactly that to check
    the numeric path against the closed form.

    The ``ValueError`` on a non-finite float is a last-resort guard on a
    PRIVATE helper and not part of any public contract: every public entry
    point runs :func:`_coefficient_reason` first and returns the same statement
    as a reason, so an in-contract caller never reaches the raise.
    """
    if callable(value):
        return value, False
    v = float(value)
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return (lambda V: v), True


# ---------------------------------------------------------------- the car


@dataclass(frozen=True)
class HeaveLaw:
    """Ride height falling with dynamic pressure: h(V) = h0 - k q, clamped.

    ``k_m_per_pa`` is an inverse stiffness in m/Pa — how far the wing sinks
    towards the track per pascal of dynamic pressure, lumping the suspension,
    the tyre sidewall and the mount's compliance into one number. It is a
    CALIBRATION of a specific car, and the only honest default is 0 (rigid).

    At ``k = 0`` the law is bit-for-bit inert, and that is DERIVED rather than
    special-cased: ``h0 - 0.0 * q`` is exactly ``h0`` in IEEE-754 for any
    finite q, and the clamp below returns ``h0`` for an input of ``h0``
    whatever ``h_min_m`` is. There is no branch on ``rigid`` in
    :meth:`height_at`, because a branch nothing can distinguish is a branch
    no test can hold honest.

    ``h_min_m`` is a bump-stop, not an aerodynamic limit: it says the car
    cannot sink past it however fast it goes. The VLM's own clearance floor
    (vlm.MIN_PLANE_CLEARANCE_FRAC) is a separate and stricter matter, checked
    where the wing is built.

    The clamp is ``min(h0, max(h_min, h0 - k q))``, and the ORDER is the
    statement: the law may never raise the car above the height it was given,
    and may never sink it below the stop; where the two disagree — a caller
    who states a height already below their own bump stop — the STATED height
    wins and the law does nothing. That is deliberate. A stated ride height is
    a design variable the caller owns, and a heave law is not the place to
    quietly move it.
    """

    k_m_per_pa: float = 0.0
    h_min_m: float = 0.05

    def __post_init__(self):
        if not float(self.k_m_per_pa) >= 0.0:
            raise ValueError(
                f"k_m_per_pa must be >= 0 (a car sinks under downforce, it "
                f"does not rise), got {self.k_m_per_pa!r}")
        if not float(self.h_min_m) > 0.0:
            raise ValueError(f"h_min_m must be > 0, got {self.h_min_m!r}")

    @property
    def rigid(self) -> bool:
        return float(self.k_m_per_pa) == 0.0

    def height_at(self, h0_m: float, q_pa: float) -> float:
        """Ride height [m] at dynamic pressure ``q_pa``, from the stated h0."""
        h = float(h0_m) - float(self.k_m_per_pa) * float(q_pa)
        return float(min(float(h0_m), max(float(self.h_min_m), h)))


@dataclass(frozen=True)
class CarSpec:
    """The car the wing is bolted to: everything the lap needs and the wing
    cannot know.

    Every field is a STATEMENT about a particular car. None of it is derived
    from the wing, and the module docstring's E8 argument is exactly that the
    front axle's share of the downforce is a ratio whose denominator lives
    here.

    Fields, and what each one IS:

    ``mass_kg``
        the car with driver and fuel, constant around the lap.
    ``mu0``
        the tyre's peak friction coefficient at the reference load
        ``n_ref_n``. A calibration of a tyre, not a physical constant.
    ``k_load``
        the load-sensitivity exponent of equation (3): ``mu_eff = mu0
        (N/N_ref)^-k``. That a tyre's coefficient of friction falls as you
        push down on it is real and measured — the module docstring cites
        one tested tyre for the sign and the order of it — but the SIZE
        belongs to a specific tyre at a specific temperature and pressure,
        so ``k`` is a calibration THE CALLER OWNS and the default is 0,
        meaning constant friction. :data:`K_LOAD_REFERENCE_LATERAL` and
        :data:`K_LOAD_REFERENCE_LONGITUDINAL` are an order to start from,
        derived in this module and belonging to it, never a default.
        Required < 1: the grip force goes as ``N^(1-k)``, so k >= 1 would
        describe a tyre that loses absolute grip when loaded.
    ``n_ref_n``
        the load ``mu0`` was stated at [N]; ``None`` derives it as the static
        weight ``m g``, so ``mu0`` is then the friction at rest.
    ``power_w``
        engine power, treated as constant and available at every speed.
    ``drivetrain_eta``
        fraction of it that reaches the road. ``power_w * drivetrain_eta`` is
        the only thing the lap ever uses.
    ``cda_car_m2``
        the drag area of EVERYTHING BUT THE WING. The wing's ``CD*S`` is
        ADDED to it, which is what makes the wing's contribution separately
        visible in the breakdown — and which assumes the two do not interact
        (see the module's omissions).
    ``cza_car_m2``
        likewise the downforce area of everything but the wing: splitter,
        floor, diffuser, bodywork. May be negative — a car without a floor
        makes LIFT — in which case the total normal load can reach zero and
        the lap reports it rather than returning an imaginary speed.
    ``wheelbase_m``, ``x_cg_frac``
        the frame. ``x_cg_frac`` is the CG's distance behind the FRONT axle
        as a fraction of the wheelbase, so the static front weight fraction
        is ``1 - x_cg_frac``.
    ``x_wing_frac``
        the wing's own longitudinal station in the same frame. Typically
        ``> 1``: a rear wing overhangs the rear axle, and equation (5) then
        gives it a NEGATIVE front share.
    ``x_cp_car_frac``
        where the rest of the car's downforce acts, in the same frame. This
        is the number a rear wing can never derive and the reason E8 called
        balance a large item.
    ``rho``
        air density at the circuit. A track is an elevation and a
        temperature, not an ISA flight level, so it is stated here rather
        than computed from an altitude (MISSION_DESIGN_WATER_TRACK.md, E3).
    ``balance_window``
        the aero-balance requirement, or ``None`` for the DERIVED default of
        equation (7): the static front weight fraction plus/minus
        :data:`BALANCE_HALF_WIDTH`.
    ``heave``
        an optional :class:`HeaveLaw`; ``None`` is rigid and inert.

    The defaults are ONE coherent car at the scale carwing.py already
    publishes (b 1.6 m, S 0.4 m^2, 55 m/s): a front-engined racing touring
    car of about 1200 kg on slicks, with a splitter and a flat floor. They
    are this package's own choice, chosen to be mutually consistent — the
    top speed they imply (measured in the tests) is the sanity check on
    them — and not taken from any published car.
    """

    mass_kg: float = 1200.0
    mu0: float = 1.5              # slick, dry, at the static load
    k_load: float = 0.0           # constant friction unless the caller says
    n_ref_n: float | None = None  # None -> m g
    power_w: float = 250.0e3
    drivetrain_eta: float = 0.88
    cda_car_m2: float = 0.90      # everything but the wing
    cza_car_m2: float = 1.20      # splitter + floor + diffuser
    wheelbase_m: float = 2.65
    x_cg_frac: float = 0.47       # 53 % static front
    x_wing_frac: float = 1.15     # overhung behind the rear axle
    x_cp_car_frac: float = 0.40   # splitter-dominated: 60 % of it on the front
    rho: float = RHO_AIR
    balance_window: tuple | None = None
    heave: HeaveLaw | None = None

    def __post_init__(self):
        for name in ("mass_kg", "mu0", "power_w", "cda_car_m2",
                     "wheelbase_m", "rho"):
            v = float(getattr(self, name))
            if not v > 0.0:
                raise ValueError(f"{name} must be > 0, got {getattr(self, name)!r}")
        if not 0.0 <= float(self.k_load) < 1.0:
            raise ValueError(
                f"k_load must satisfy 0 <= k < 1: the grip FORCE goes as "
                f"N^(1-k), so k >= 1 describes a tyre that loses absolute "
                f"grip when it is loaded. Got {self.k_load!r}")
        if not 0.0 < float(self.drivetrain_eta) <= 1.0:
            raise ValueError(
                f"drivetrain_eta must be in (0, 1], got {self.drivetrain_eta!r}")
        if self.n_ref_n is not None and not float(self.n_ref_n) > 0.0:
            raise ValueError(f"n_ref_n must be > 0 or None, got {self.n_ref_n!r}")
        if not 0.0 < float(self.x_cg_frac) < 1.0:
            raise ValueError(
                f"x_cg_frac must lie strictly between the axles, got "
                f"{self.x_cg_frac!r}")
        for name in ("x_wing_frac", "x_cp_car_frac"):
            if not np.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite, got {getattr(self, name)!r}")
        if not np.isfinite(float(self.cza_car_m2)):
            raise ValueError(f"cza_car_m2 must be finite, got {self.cza_car_m2!r}")
        if self.balance_window is not None:
            lo, hi = (float(v) for v in self.balance_window)
            if not hi > lo:
                raise ValueError(
                    f"balance_window must satisfy lo < hi, got "
                    f"{self.balance_window!r}")
        if self.heave is not None and not isinstance(self.heave, HeaveLaw):
            raise ValueError("heave must be a HeaveLaw or None")

    # ---- derived, all of it one line and none of it stored

    @property
    def weight_n(self) -> float:
        """Static weight m g [N] (mission.G0, the package's one gravity)."""
        return float(self.mass_kg) * G0

    @property
    def n_ref(self) -> float:
        """Reference normal load for the friction law [N]; m g by default."""
        return (self.weight_n if self.n_ref_n is None else float(self.n_ref_n))

    @property
    def power_eff_w(self) -> float:
        """Power at the road [W] = power_w * drivetrain_eta."""
        return float(self.power_w) * float(self.drivetrain_eta)

    @property
    def front_weight_frac(self) -> float:
        """Static front-axle weight fraction, 1 - x_cg_frac (equation 5)."""
        return 1.0 - float(self.x_cg_frac)

    @property
    def heave_law(self) -> HeaveLaw:
        """The heave law in force; a rigid one when none was stated."""
        return self.heave if self.heave is not None else HeaveLaw()

    def q(self, V: float) -> float:
        """Dynamic pressure 1/2 rho V^2 [Pa] at speed ``V``."""
        return 0.5 * float(self.rho) * float(V) ** 2

    def mu_eff(self, N: float) -> float:
        """Effective friction coefficient at normal load ``N`` [N]; eq. (3).

        Returns 0 for a non-positive load: a car whose tyres are unloaded has
        no grip, and that is a statement the lap has to be able to make (a
        negative ``cza_car_m2`` at speed can produce it).
        """
        N = float(N)
        if N <= 0.0:
            return 0.0
        k = float(self.k_load)
        if k == 0.0:
            return float(self.mu0)      # bit-for-bit: no power is taken
        return float(self.mu0) * (N / self.n_ref) ** (-k)

    def grip_force(self, N: float) -> float:
        """Available tyre force ``mu_eff(N) N`` [N] = ``mu0 N_ref^k N^(1-k)``."""
        N = float(N)
        if N <= 0.0:
            return 0.0
        return self.mu_eff(N) * N

    def normal_load(self, V: float, cza_wing_m2: float) -> float:
        """Total normal load ``m g + q (CzA_car + cz_a)`` [N] at speed ``V``."""
        return self.weight_n + self.q(V) * (float(self.cza_car_m2)
                                            + float(cza_wing_m2))

    def ride_height_at(self, h0_m: float, V: float) -> float:
        """Wing ride height [m] at speed ``V`` under the stated heave law."""
        return self.heave_law.height_at(h0_m, self.q(V))

    def balance_window_used(self) -> tuple:
        """The balance window in force: the caller's, or equation (7)'s."""
        if self.balance_window is not None:
            return (float(self.balance_window[0]), float(self.balance_window[1]))
        c = self.front_weight_frac
        return (c - BALANCE_HALF_WIDTH, c + BALANCE_HALF_WIDTH)


# ---------------------------------------------------------------- the track


@dataclass(frozen=True)
class Corner:
    """A constant-radius arc, taken at a constant speed.

    ``radius_m`` is the path radius the car follows and ``arc_m`` the
    distance travelled along it — a length, not an angle, because the lap is
    an integral over distance and the angle never enters. The turn's
    DIRECTION is irrelevant to a symmetric point mass.
    """

    radius_m: float
    arc_m: float

    def __post_init__(self):
        if not float(self.radius_m) > 0.0:
            raise ValueError(f"radius_m must be > 0, got {self.radius_m!r}")
        if not float(self.arc_m) > 0.0:
            raise ValueError(f"arc_m must be > 0, got {self.arc_m!r}")

    @property
    def length_m(self) -> float:
        return float(self.arc_m)


@dataclass(frozen=True)
class Straight:
    """A length of track with no lateral demand. A straight IS its length."""

    length_m: float

    def __post_init__(self):
        if not float(self.length_m) > 0.0:
            raise ValueError(f"length_m must be > 0, got {self.length_m!r}")


@dataclass(frozen=True)
class TrackSpec:
    """An ordered, CYCLIC list of segments — the lap.

    The list wraps: the last segment is followed by the first, because a lap
    time is only well posed on a closed circuit (an open sector would need a
    stated entry speed, which is a different question).

    Two validation rules, both of them arguments rather than preferences:

    * AT LEAST ONE CORNER. A lap with no corner puts no upper bound on the
      speed anywhere, so the whole thing degenerates to ``length / V_top``
      and the wing's downforce cannot appear in the answer at all. That is
      not a lap; it is a top-speed run, and :func:`top_speed` is the function
      for it.
    * NO TWO STRAIGHTS IN A ROW. A straight is defined only by its length, so
      splitting one carries no information and merging is exact — but the
      merge would then have to be undone to report a per-segment breakdown
      against the caller's own indices. Refusing the split keeps the
      breakdown one-to-one with what was asked for.

    Adjacent CORNERS are allowed and are a real thing (a compound corner, a
    chicane), but the model changes speed between them instantaneously,
    which is a transient it does not carry. :func:`lap_time` reports it in
    ``warnings`` rather than refusing it.
    """

    segments: tuple = ()
    name: str = "unnamed"

    def __post_init__(self):
        segs = tuple(self.segments)
        object.__setattr__(self, "segments", segs)
        if not segs:
            raise ValueError("a track needs at least one segment")
        for i, s in enumerate(segs):
            if not isinstance(s, (Corner, Straight)):
                raise ValueError(
                    f"segment {i} is {type(s).__name__}; a track is made of "
                    f"Corner and Straight")
        if not any(isinstance(s, Corner) for s in segs):
            raise ValueError(
                "a lap needs at least one corner: with no corner nothing "
                "limits the speed, the lap time is length / top speed and the "
                "downforce cannot enter the answer. Use top_speed() instead")
        n = len(segs)
        for i in range(n):
            j = (i + 1) % n
            if isinstance(segs[i], Straight) and isinstance(segs[j], Straight):
                raise ValueError(
                    f"segments {i} and {j} are both straights; a straight is "
                    f"defined only by its length, so give one straight of "
                    f"{segs[i].length_m + segs[j].length_m} m")

    @property
    def length_m(self) -> float:
        """Lap distance [m]."""
        return float(sum(s.length_m for s in self.segments))

    @property
    def corners(self) -> tuple:
        return tuple(s for s in self.segments if isinstance(s, Corner))

    def rotated_to_start_at_a_corner(self) -> tuple:
        """``(segments, shift)`` rotated so index 0 is a Corner.

        Every straight is then bracketed by corners (no two straights are
        adjacent, by validation), which is what makes each straight a
        two-point boundary problem with both ends known. ``shift`` maps back:
        rotated index i is the caller's index ``(i + shift) % n``.
        """
        segs = self.segments
        shift = next(i for i, s in enumerate(segs) if isinstance(s, Corner))
        return (segs[shift:] + segs[:shift], shift)


def synthetic_lap() -> TrackSpec:
    """The reference lap. **SYNTHETIC — it is not any real circuit.**

    Deliberately not a real circuit's numbers: this package has no verified
    survey of one, and inventing a name for a made-up layout would be a
    citation nobody could check. What it IS is a layout chosen so that the
    three things a car-wing mission has to be able to express are all present
    and all bind on the reference car:

    * a genuine LOW-SPEED corner (the 25 m hairpin), where downforce buys
      almost nothing — the grip is nearly all weight — so a wing paid for
      here is paid for badly;
    * a genuine FAST corner (the 180 m radius), where downforce buys a great
      deal, because the aero term of equation (2)'s denominator is a large
      fraction of it;
    * a straight long enough to be DRAG-LIMITED rather than power-limited at
      its end (the 1400 m one), so the wing's drag is charged against
      something real rather than against a coefficient allowance.

    Every number below is DERIVED by the module from the layout and the
    reference car, at the published carwing point (CZ 0.655860500048, CD
    0.023739403946 on S = 0.4 m^2, i.e. cz_a 0.26234 m^2 and cd_a 0.0094958
    m^2), and re-derived rather than pinned in tests/test_cartrack.py:

    ========  ======  ==========  =====================================
    segment    L [m]   V [m/s]     what it is for
    ========  ======  ==========  =====================================
    hairpin      60    19.45      97.2 % of the grip here is WEIGHT:
                                  downforce buys almost nothing
    straight   1000    peak 65.9  accelerate, then brake for the last
                                  80 m
    medium      120    35.95      91.0 % weight
    straight    500    peak 60.3  brakes for its last 7.5 m
    fast        200    57.59      79.8 % weight, so a fifth of the grip
                                  is bought: downforce is worth a lot
    straight   1400    peak 71.2  0.970 of the 73.37 m/s top speed, so
                                  91 % of the engine is spent on drag
                                  at the end of it — drag-limited
    ========  ======  ==========  =====================================

    Lap distance 3280 m; lap time 61.372 s, mean speed 53.44 m/s. The
    reference car's aero-unbounded radius (module docstring, equation 2) is
    893.2 m, comfortably outside the layout, so no corner on this lap is a
    statement instead of a number.
    """
    return TrackSpec(
        name="synthetic reference lap (not a real circuit)",
        segments=(
            Corner(radius_m=25.0, arc_m=60.0),     # hairpin
            Straight(length_m=1000.0),
            Corner(radius_m=80.0, arc_m=120.0),    # medium
            Straight(length_m=500.0),
            Corner(radius_m=180.0, arc_m=200.0),   # fast
            Straight(length_m=1400.0),             # long: ends drag-limited
        ),
    )


# ------------------------------------------------------- top and corner speed


def top_speed_result(cd_a: Coefficient, car: CarSpec | None = None) -> dict:
    """Drag-limited top speed, equation (4), as an in-contract RESULT.

    Returns ``{"feasible", "reason", "V_top", "cda_total_m2", "path"}``. It
    never raises for anything a caller can put in it, which is what lets
    :func:`lap_time` keep the same promise: the lap needs a top speed before
    it can integrate anything, so every way of not having one has to arrive
    here as a reason.

    Constant ``cd_a`` takes the closed form ``(2 P_eff / (rho CdA_tot))^(1/3)``
    exactly (``path = "closed_form"``). A callable takes a bracketed solve of
    ``P_eff = 1/2 rho CdA(V) V^3`` (``path = "bracketed"``), which agrees with
    the closed form to round-off for a constant callable (gated in the tests).

    **The bracket is grown, not stated.** There is deliberately no
    ``v_max_search`` argument. A search cap is invisible in the answer and
    visible only in a raise, and carwing.py's production path hands this
    function ``np.interp`` callables on every evaluation — so a cap would be
    a refusal nobody chose, fired from inside an optimiser. Instead the upper
    end doubles from 1 m/s until the residual changes sign, at most
    :data:`N_DOUBLINGS_MAX` times (1.8e19 m/s); the answer is therefore
    wherever the closed form would put it, and the doubling limit only makes
    a law that NEVER crosses terminate, with a reason that names the drag
    area found up there.

    The three ways there is no top speed, all of them reasons:

    * a total drag area <= 0 — a car with no drag has no drag-limited speed
      at all, and this is reachable in-contract because ``cd_a`` may be
      negative (a wing that is a net thrust) and ``cda_car_m2`` is finite;
    * a coefficient that is not a finite number, or a law returning one;
    * a law under which the residual never changes sign.

    Rolling resistance is not in the balance — see the module's omissions —
    so this is exactly the cube root and not a cubic.
    """
    reason = _coefficient_reason(cd_a, "cd_a")
    if reason:
        return _fail(reason, V_top=None, cda_total_m2=None, path=None)
    car = car or CarSpec()
    law, is_const = _as_law(cd_a, "cd_a")
    p_eff = car.power_eff_w

    def cda_tot_at(V):
        return float(car.cda_car_m2) + float(law(V))

    if is_const:
        cda_tot = cda_tot_at(0.0)
        if not cda_tot > 0.0:
            return _fail(
                f"total drag area CdA_car + cd_a = {cda_tot:g} m^2 is not "
                f"positive, so equation (4) has no root: a car with no drag "
                f"has no drag-limited top speed",
                V_top=None, cda_total_m2=float(cda_tot), path="closed_form")
        return {
            "feasible": True, "reason": "",
            "V_top": float((2.0 * p_eff
                            / (float(car.rho) * cda_tot)) ** (1.0 / 3.0)),
            "cda_total_m2": float(cda_tot), "path": "closed_form",
        }

    def resid(V):
        return p_eff - 0.5 * float(car.rho) * cda_tot_at(V) * V ** 3

    lo = 1e-6
    try:
        r_lo = resid(lo)
    except (TypeError, ValueError) as exc:                  # a law that is
        return _fail(f"cd_a law raised at V = {lo:g} m/s: {exc}",  # not one
                     V_top=None, cda_total_m2=None, path="bracketed")
    if not np.isfinite(r_lo):
        return _fail(
            f"cd_a law gives a non-finite drag area {float(law(lo)):g} m^2 at "
            f"V = {lo:g} m/s, so equation (4) has no residual to solve",
            V_top=None, cda_total_m2=None, path="bracketed")
    if r_lo <= 0.0:
        return _fail(
            f"the car is already drag-limited at {lo:g} m/s (total drag area "
            f"{cda_tot_at(lo):g} m^2 against {p_eff:g} W at the road), so "
            f"there is no top speed above it to find",
            V_top=None, cda_total_m2=float(cda_tot_at(lo)), path="bracketed")

    hi = 1.0
    for _ in range(int(N_DOUBLINGS_MAX)):
        r_hi = resid(hi)
        if not np.isfinite(r_hi):
            return _fail(
                f"cd_a law gives a non-finite drag area at V = {hi:g} m/s, so "
                f"equation (4) has no residual to solve there",
                V_top=None, cda_total_m2=None, path="bracketed")
        if r_hi <= 0.0:
            break
        hi *= 2.0
    else:
        return _fail(
            f"the drag law leaves the car still accelerating at {hi:g} m/s "
            f"(total drag area {cda_tot_at(hi):g} m^2), which is what a "
            f"vanishing or non-positive drag area looks like from inside the "
            f"bracket search; there is no drag-limited top speed",
            V_top=None, cda_total_m2=float(cda_tot_at(hi)), path="bracketed")

    v = float(brentq(resid, lo, hi, xtol=1e-12, rtol=1e-14, maxiter=200))
    return {
        "feasible": True, "reason": "", "V_top": v,
        "cda_total_m2": float(cda_tot_at(v)), "path": "bracketed",
    }


def top_speed(cd_a: Coefficient, car: CarSpec | None = None) -> float:
    """Drag-limited top speed [m/s] — :func:`top_speed_result` as a number.

    The convenience form, for a caller who wants the speed and nothing else.
    It RAISES ``ValueError`` where :func:`top_speed_result` would return a
    reason, and that is the price of returning a bare float: a function whose
    whole answer is a number has nowhere to put a statement. Nothing in this
    module that promises not to raise goes through here — :func:`lap_time`,
    :func:`straight_profile` and :func:`corner_speed` all call
    :func:`top_speed_result` and hand the reason on.
    """
    out = top_speed_result(cd_a, car)
    if not out["feasible"]:
        raise ValueError(out["reason"])
    return float(out["V_top"])


def corner_speed(radius_m: float, cz_a: Coefficient, cd_a: Coefficient,
                 car: CarSpec | None = None,
                 n_scan: int = N_SCAN_DEFAULT,
                 v_top: float | None = None) -> dict:
    """Quasi-steady corner speed, and WHICH limit produced it.

    Returns ``{"feasible", "V", "limited_by", "reason", "V_grip", "V_top",
    "residual"}``. A corner with no answer — there is no top speed to cap it
    against, or a coefficient is not a number — comes back
    ``feasible=False``, ``V=None``, ``limited_by="no_top_speed"`` with the
    reason, never as a raise. ``radius_m <= 0`` is the one exception and it
    IS a raise, because it is a call-site error and not a design: a corner
    with no radius is not a corner a car could be asked to take.

    ``limited_by`` is one of

        ``"grip"``             equation (1) has a root below the top speed;
                               ``V`` is that root.
        ``"top_speed"``        it has a root, but above the car's top speed;
                               the car simply cannot arrive that fast, so
                               ``V`` is the top speed.
        ``"aero_unbounded"``   grip does not run out at any speed this car can
                               reach. In the closed-form case that is exact —
                               the denominator of equation (2) is <= 0, so
                               there is no positive root at all, and the
                               threshold radius ``R = m / (1/2 mu rho
                               CzA_tot)`` is a number the reason states. In
                               the numeric case it means no sign change was
                               found up to 1.5x the top speed, which is the
                               same engineering statement (the module
                               docstring measures where a load-sensitive
                               tyre's root actually lies: 3.46e9 m/s at
                               k = 0.05, R = 5000 m). ``V`` is capped at the
                               car's top speed and ``reason`` says which of
                               the two happened.

    There is no "no grip at all" case to handle, and that is a derivation
    rather than an omission: at V -> 0 the residual is ``mu0 m g > 0`` for
    every car, because a car has weight; and at any root ``mu_eff(N) N =
    m V^2 / R > 0`` forces ``N > 0``, so a solved corner always has loaded
    tyres however much lift ``cza_car_m2`` describes.

    ``residual`` is ``mu_eff(N) N - m V^2 / R`` at the returned speed when
    grip is what limited it — zero to round-off, and asserted as such rather
    than assumed.

    ``v_top`` lets a caller that has already solved equation (4) hand it in.
    That is not micro-optimisation: with a speed-dependent ``cd_a`` the top
    speed costs a bracketed solve, i.e. tens of calls into the CALLER's own
    aerodynamic model, and :func:`lap_time` would otherwise pay for one per
    corner and one per straight.
    """
    R = float(radius_m)
    if not R > 0.0:
        raise ValueError(f"radius_m must be > 0, got {radius_m!r}")
    for value, name in ((cz_a, "cz_a"), (cd_a, "cd_a")):
        reason = _coefficient_reason(value, name)
        if reason:
            return {"feasible": False, "reason": reason, "V": None,
                    "V_grip": None, "V_top": None,
                    "limited_by": "no_top_speed", "residual": None}
    car = car or CarSpec()
    cz_law, cz_const = _as_law(cz_a, "cz_a")
    if v_top is None:
        top = top_speed_result(cd_a, car)
        if not top["feasible"]:
            return {"feasible": False, "V": None, "V_grip": None,
                    "V_top": None, "limited_by": "no_top_speed",
                    "residual": None,
                    "reason": (f"no corner speed at R = {R:g} m without a top "
                               f"speed to cap it: {top['reason']}")}
        v_top = float(top["V_top"])
    v_top = float(v_top)
    m = float(car.mass_kg)

    def residual(V):
        return car.grip_force(car.normal_load(V, cz_law(V))) - m * V * V / R

    if cz_const and float(car.k_load) == 0.0:
        # closed form, equation (2)
        cza_tot = float(car.cza_car_m2) + cz_law(0.0)
        denom = m / R - 0.5 * float(car.mu0) * float(car.rho) * cza_tot
        if denom <= 0.0:
            return {
                "feasible": True,
                "V": float(v_top), "V_grip": None, "V_top": float(v_top),
                "limited_by": "aero_unbounded", "residual": None,
                "reason": (
                    f"no grip-limited corner speed exists at R = {R:g} m: the "
                    f"downforce adds lateral capacity at 1/2 mu rho CzA_tot = "
                    f"{0.5 * float(car.mu0) * float(car.rho) * cza_tot:.4g} N "
                    f"per (m/s)^2 against the corner's demand m/R = "
                    f"{m / R:.4g}, so equation (2)'s denominator is "
                    f"{denom:.4g} <= 0 and grip never runs out. Every radius "
                    f"at or above "
                    f"{m / (0.5 * float(car.mu0) * float(car.rho) * cza_tot):.4g}"
                    f" m does this on this car. Capped at the car's top speed "
                    f"{v_top:.4g} m/s"),
            }
        v_grip = float(np.sqrt(float(car.mu0) * m * G0 / denom))
    else:
        # numeric: scan for the FIRST sign change, then Brent on that bracket.
        # The scan starts positive for every car (residual -> mu0 m g > 0 as
        # V -> 0), so the first crossing is the corner speed.
        grid = np.linspace(1e-9, max(v_top * 1.5, 1.0), int(n_scan))
        vals = np.array([residual(float(v)) for v in grid])
        if not np.all(np.isfinite(vals)):
            # a cz_a law returning NaN would otherwise leave every comparison
            # False and be read as "grip never runs out", which is the exact
            # opposite of what is known: nothing is known
            bad = float(grid[int(np.nonzero(~np.isfinite(vals))[0][0])])
            return {
                "feasible": False, "V": None, "V_grip": None,
                "V_top": float(v_top), "limited_by": "no_top_speed",
                "residual": None,
                "reason": (
                    f"the cz_a law gives a non-finite downforce area at "
                    f"V = {bad:.4g} m/s, so equation (1) has no residual to "
                    f"solve at R = {R:g} m"),
            }
        idx = np.nonzero(vals <= 0.0)[0]
        if idx.size == 0:
            return {
                "feasible": True,
                "V": float(v_top), "V_grip": None, "V_top": float(v_top),
                "limited_by": "aero_unbounded", "residual": None,
                "reason": (
                    f"no grip-limited corner speed found at R = {R:g} m below "
                    f"{grid[-1]:.4g} m/s: grip still exceeds the corner's "
                    f"demand there. Capped at the car's top speed "
                    f"{v_top:.4g} m/s"),
            }
        j = int(idx[0])
        v_grip = float(brentq(residual, float(grid[j - 1]), float(grid[j]),
                              xtol=1e-12, rtol=1e-14, maxiter=200))

    if v_grip >= v_top:
        return {
            "feasible": True,
            "V": float(v_top), "V_grip": float(v_grip), "V_top": float(v_top),
            "limited_by": "top_speed", "residual": None,
            "reason": (
                f"grip would allow {v_grip:.4g} m/s at R = {R:g} m but the car "
                f"cannot exceed {v_top:.4g} m/s"),
        }
    return {
        "feasible": True,
        "V": float(v_grip), "V_grip": float(v_grip), "V_top": float(v_top),
        "limited_by": "grip", "residual": float(residual(v_grip)), "reason": "",
    }


# ---------------------------------------------------------------- straights


def _accel(V: float, cz_law, cd_law, car: CarSpec, p_eff: float) -> float:
    """Longitudinal acceleration [m/s^2] under power, grip and drag."""
    q = car.q(V)
    f_power = p_eff / V
    f_grip = car.grip_force(car.weight_n + q * (float(car.cza_car_m2)
                                                + float(cz_law(V))))
    drag = q * (float(car.cda_car_m2) + float(cd_law(V)))
    return (min(f_power, f_grip) - drag) / float(car.mass_kg)


def _brake(V: float, cz_law, cd_law, car: CarSpec) -> float:
    """Braking deceleration MAGNITUDE [m/s^2]: grip plus drag, both at V.

    Drag helps: a car at speed stops shorter than its tyres alone explain.
    The brakes themselves are assumed able to reach the grip limit at every
    speed (no fade, no pedal, no brake balance).
    """
    q = car.q(V)
    f_grip = car.grip_force(car.weight_n + q * (float(car.cza_car_m2)
                                                + float(cz_law(V))))
    drag = q * (float(car.cda_car_m2) + float(cd_law(V)))
    return (f_grip + drag) / float(car.mass_kg)


def _rk4_profile(V0: float, h: float, n: int, deriv) -> np.ndarray:
    """RK4 on ``dV/dx = deriv(V)`` from ``V0``, ``n`` steps of ``h``.

    Returns ``n + 1`` speeds. A step that would drive the speed to zero or
    below returns ``nan`` from there on, which the caller turns into a reason
    — a negative speed in a lap simulator is a failure to report, not a value
    to propagate.

    Note what this guard is FOR. It cannot fire on the physics: accelerating,
    the speed decays towards a positive terminal speed and never through it;
    braking backwards, the speed only rises. It fires on a pathological
    caller-supplied coefficient law (a callable returning an enormous drag
    area, say), which is exactly the case where returning a reason beats
    returning a negative speed that would look like an answer.
    """
    out = np.empty(n + 1, dtype=float)
    out[0] = V = float(V0)
    for i in range(n):
        if not (V > 0.0) or not np.isfinite(V):
            out[i + 1:] = np.nan
            return out
        k1 = deriv(V)
        k2 = deriv(max(V + 0.5 * h * k1, 1e-9))
        k3 = deriv(max(V + 0.5 * h * k2, 1e-9))
        k4 = deriv(max(V + h * k3, 1e-9))
        V = V + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        out[i + 1] = V
    return out


def straight_profile(length_m: float, v_in: float, v_out: float,
                     cz_a: Coefficient, cd_a: Coefficient,
                     car: CarSpec | None = None,
                     n_steps: int = N_STEPS_DEFAULT,
                     v_top: float | None = None) -> dict:
    """Speed profile and time over one straight; the scheme is in the module
    docstring.

    ``v_in`` is the speed leaving the previous corner and ``v_out`` the speed
    the next corner demands. Returns ``x``, ``V``, ``t_s``, ``V_peak``, the
    braking-point distance, and ``brake_limited_entry`` — the flag for a
    straight too short to slow down on, which is reported and never silent.

    ``n_steps < 2`` raises: that is a call-site error, a request for an
    integrator that cannot integrate, and not a design the caller could have
    meant. Everything about the COEFFICIENTS comes back as a reason.
    """
    n = int(n_steps)
    if n < 2:
        raise ValueError(f"n_steps must be >= 2, got {n_steps!r}")
    for value, name in ((cz_a, "cz_a"), (cd_a, "cd_a")):
        reason = _coefficient_reason(value, name)
        if reason:
            return _fail(reason)
    car = car or CarSpec()
    cz_law, _ = _as_law(cz_a, "cz_a")
    cd_law, _ = _as_law(cd_a, "cd_a")
    L = float(length_m)
    h = L / n
    p_eff = car.power_eff_w
    if v_top is None:
        top = top_speed_result(cd_a, car)
        if not top["feasible"]:
            return _fail(f"no straight without a top speed: {top['reason']}")
        v_top = float(top["V_top"])
    v_top = float(v_top)

    v_acc = _rk4_profile(v_in, h, n,
                         lambda V: _accel(V, cz_law, cd_law, car, p_eff) / V)
    # backwards from the exit: in the reversed coordinate s = L - x the
    # braking car SPEEDS UP, so the same RK4 runs with a positive derivative
    v_brk_rev = _rk4_profile(v_out, h, n,
                             lambda V: _brake(V, cz_law, cd_law, car) / V)
    v_brk = v_brk_rev[::-1]

    if not np.all(np.isfinite(v_acc)) or not np.all(np.isfinite(v_brk)):
        return _fail(
            "straight integration reached a non-positive speed, or a speed "
            "that is not a number: the car cannot hold this straight (drag "
            "exceeds the traction available at the entry speed), or a "
            "coefficient law returned something that is not a drag area")

    V = np.minimum(np.minimum(v_acc, v_brk), v_top)
    if np.any(V <= 0.0):
        return _fail("straight integration produced a non-positive speed")
    x = np.linspace(0.0, L, n + 1)
    t = float(np.trapezoid(1.0 / V, x))
    brake_from = np.nonzero(v_brk < np.minimum(v_acc, v_top))[0]
    return {
        "feasible": True, "reason": "",
        "x": x, "V": V, "t_s": t, "length_m": L,
        "V_in": float(V[0]), "V_out": float(V[-1]),
        "V_peak": float(V.max()), "V_top": float(v_top),
        "brake_point_m": (float(x[int(brake_from[0])]) if brake_from.size
                          else None),
        "brake_limited_entry": bool(v_brk[0] < v_in - 1e-9),
        "V_entry_implied": float(min(v_brk[0], v_in)),
        "n_steps": n,
    }


# ---------------------------------------------------------------- the lap


def lap_time(cz_a: Coefficient, cd_a: Coefficient,
             car: CarSpec | None = None, track: TrackSpec | None = None,
             n_steps: int = N_STEPS_DEFAULT) -> dict:
    """Lap time [s] for a wing of ``CZ*S = cz_a`` and ``CD*S = cd_a`` [m^2].

    Both coefficients may be a float or a callable of speed; a callable is
    how the multi-point interface (:func:`representative_points`) is spent,
    and a float is bit-for-bit the single-point case.

    Never raises for anything a caller can put in a coefficient. Each of
    these comes back ``feasible=False`` with a reason instead, and each is
    gated in tests/test_cartrack.py:

    * a coefficient that is not a finite number (NaN, infinity, a string);
    * a TOTAL drag area ``CdA_car + cd_a`` at or below zero, which is
      reachable because ``cd_a`` may be negative;
    * a drag LAW under which equation (4) has no root — there is no search
      cap to run into, see :func:`top_speed_result`;
    * a downforce law that returns something that is not a number;
    * a straight the car cannot hold, or a corner with no grip.

    ``n_steps`` is the exception and it raises, because an integrator with
    fewer than two steps is a call-site error rather than a design.

    The returned dict carries ``lap_time_s``, a per-segment ``segments`` list
    in the CALLER's own order, ``warnings`` (adjacent corners, straights too
    short to brake on), and the derived ``V_top``.
    """
    for value, name in ((cz_a, "cz_a"), (cd_a, "cd_a")):
        reason = _coefficient_reason(value, name)
        if reason:
            return _fail(reason)
    car = car or CarSpec()
    track = track or synthetic_lap()
    segs, shift = track.rotated_to_start_at_a_corner()
    n = len(segs)

    top = top_speed_result(cd_a, car)
    if not top["feasible"]:
        return _fail(f"no lap without a top speed: {top['reason']}")
    v_top = float(top["V_top"])
    warnings: list[str] = []

    # every corner's speed first: a straight needs the corners at both ends
    corner_out: dict[int, dict] = {}
    for i, s in enumerate(segs):
        if isinstance(s, Corner):
            cs = corner_speed(s.radius_m, cz_a, cd_a, car, v_top=v_top)
            if not cs["feasible"]:
                return _fail(f"segment {(i + shift) % n}: {cs['reason']}")
            corner_out[i] = cs
            if cs["limited_by"] == "aero_unbounded":
                warnings.append(
                    f"segment {(i + shift) % n}: {cs['reason']}")

    rows: list[dict] = [None] * n
    for i, s in enumerate(segs):
        if isinstance(s, Corner):
            cs = corner_out[i]
            V = cs["V"]
            rows[i] = {
                "kind": "corner", "radius_m": float(s.radius_m),
                "length_m": float(s.arc_m), "t_s": float(s.arc_m) / V,
                "V_in": V, "V_out": V, "V_mean": V, "V_peak": V,
                "limited_by": cs["limited_by"], "reason": cs["reason"],
                "profile_V": np.array([V, V]),
                "profile_x": np.array([0.0, float(s.arc_m)]),
            }
            nxt = segs[(i + 1) % n]
            if n > 1 and isinstance(nxt, Corner):
                warnings.append(
                    f"segments {(i + shift) % n} and {(i + 1 + shift) % n} are "
                    f"adjacent corners: the model changes speed between them "
                    f"instantaneously, a transient it does not carry")
        else:
            v_in = corner_out[i - 1]["V"]                 # i >= 1 by rotation
            v_out = corner_out[(i + 1) % n]["V"]
            pr = straight_profile(s.length_m, v_in, v_out, cz_a, cd_a, car,
                                  n_steps=n_steps, v_top=v_top)
            if not pr["feasible"]:
                return _fail(f"segment {(i + shift) % n}: {pr['reason']}")
            if pr["brake_limited_entry"]:
                warnings.append(
                    f"segment {(i + shift) % n}: {s.length_m:g} m is too short "
                    f"to brake from {v_in:.4g} to {v_out:.4g} m/s; the entry "
                    f"speed implied by the next corner is "
                    f"{pr['V_entry_implied']:.4g} m/s. Corner speeds are NOT "
                    f"propagated backwards, so this lap time is understated")
            rows[i] = {
                "kind": "straight", "radius_m": None,
                "length_m": float(s.length_m), "t_s": pr["t_s"],
                "V_in": pr["V_in"], "V_out": pr["V_out"],
                "V_mean": float(pr["length_m"] / pr["t_s"]),
                "V_peak": pr["V_peak"],
                "V_peak_over_top": float(pr["V_peak"] / v_top),
                # at the peak the tractive force is P_eff/V and the drag is
                # 1/2 rho CdA V^2, whose ratio is exactly (V/V_top)^3 — so
                # this IS the fraction of the engine spent on drag there, and
                # it says how drag-limited the straight got without a
                # threshold anywhere
                "drag_frac_at_peak": float((pr["V_peak"] / v_top) ** 3),
                "limited_by": ("top_speed" if pr["V_peak"] >= v_top * (1 - 1e-9)
                               else "distance"),
                "reason": "", "brake_point_m": pr["brake_point_m"],
                "profile_V": pr["V"], "profile_x": pr["x"],
            }

    total = float(sum(r["t_s"] for r in rows))
    # back to the caller's segment order: a breakdown indexed by the module's
    # internal rotation would hand ``segments[0]`` to someone who wrote a
    # different segment there
    out_rows = [None] * n
    for i in range(n):
        rows[i]["index"] = (i + shift) % n
        out_rows[(i + shift) % n] = rows[i]
    return {
        "feasible": True, "reason": "",
        "lap_time_s": total,
        "segments": out_rows,
        "length_m": track.length_m,
        "V_mean": float(track.length_m / total),
        "V_top": float(v_top),
        "V_min": float(min(r["V_in"] for r in rows)),
        "V_max": float(max(r["V_peak"] for r in rows)),
        "warnings": warnings,
        "track": track.name,
        "n_steps": int(n_steps),
    }


# -------------------------------------------------- the multi-point interface


def speed_time_histogram(lap: dict) -> tuple:
    """``(V, dt)`` — every cell of the lap and how long the car spends in it.

    Corners contribute one cell at their constant speed; straights contribute
    one cell per integration step, at the cell's mean speed and with the
    trapezoidal time of that cell, so the cells sum EXACTLY to the segment
    times :func:`lap_time` reported (asserted in the tests, because a
    histogram that does not sum to the lap is a different lap).
    """
    Vs, dts = [], []
    for r in lap["segments"]:
        if r["kind"] == "corner":
            Vs.append(r["V_in"])
            dts.append(r["t_s"])
        else:
            x, V = r["profile_x"], r["profile_V"]
            dx = np.diff(x)
            inv = 1.0 / V
            dt = 0.5 * (inv[:-1] + inv[1:]) * dx          # the trapezoid cells
            Vs.append(0.5 * (V[:-1] + V[1:]))
            dts.append(dt)
    return (np.concatenate([np.atleast_1d(v) for v in Vs]),
            np.concatenate([np.atleast_1d(t) for t in dts]))


def time_weighted_mean_speed(lap: dict) -> float:
    """The one speed a single-point evaluation would have to be made at.

    The TIME-weighted mean, not the distance-weighted one (which is just the
    lap's average speed): a coefficient evaluated once is being asked to stand
    in for every instant of the lap, and instants are what time weights.
    """
    V, dt = speed_time_histogram(lap)
    return float(np.sum(V * dt) / np.sum(dt))


def representative_points(cz_a: Coefficient, cd_a: Coefficient,
                          car: CarSpec | None = None,
                          track: TrackSpec | None = None,
                          n_points: int = 4, binning: str = "speed",
                          ride_height_m: float | None = None,
                          n_steps: int = N_STEPS_DEFAULT) -> dict:
    """The speeds the lap weights most, with their TIME weights.

    This is E5's interface: a caller evaluates the wing AT each returned
    speed (and, with a heave law stated, at each returned ride height) and
    passes the results back as callables, instead of evaluating once and
    referring the answer to other speeds by a q ratio. The q-ratio referral
    is exact only when the coefficients are speed-independent, which
    MISSION_DESIGN_WATER_TRACK.md section 3 measured to be true of ``car rear
    wing`` and false by up to 4.8 % of CD on ``car rear wing + endplates``.

    ``binning`` is stated because the two honest choices give different
    answers and neither is the obvious one:

        ``"speed"``  (default) equal-width bins between the lap's slowest and
                     fastest speeds. The weights are then the genuine
                     histogram of lap time against speed, so they SAY where
                     the time goes; empty bins are dropped and the rest
                     renormalised, so the returned count may be < n_points.
        ``"time"``   equal-TIME quantiles. The weights are then approximately
                     1/n and carry almost no information, but the speeds are
                     placed where the time is, which is the better sampling if
                     all you want is n points to evaluate at. Only
                     approximately: the split falls on whole integration
                     cells, so each bin gets whatever the cell boundaries
                     allow — measured 0.212 to 0.288 at n = 4 on the reference
                     lap, against 0.25 exactly.

    Each point's speed is the TIME-weighted mean speed within its bin, so the
    weighted mean of the returned speeds equals :func:`time_weighted_mean_speed`
    exactly (gated in the tests).

    ``ride_height_m`` opts into the heave law: give the wing's stated ride
    height and each point carries the height ``h(V)`` implies. With no heave
    law on the car (or ``k_m_per_pa = 0``) every returned height is the stated
    one, bit-for-bit.
    """
    if binning not in ("speed", "time"):
        raise ValueError(f"unknown binning {binning!r}; choose speed | time")
    if int(n_points) < 1:
        raise ValueError(f"n_points must be >= 1, got {n_points!r}")
    car = car or CarSpec()
    lap = lap_time(cz_a, cd_a, car, track, n_steps=n_steps)
    if not lap["feasible"]:
        return _fail(f"no lap to sample: {lap['reason']}")

    V, dt = speed_time_histogram(lap)
    n = int(n_points)
    order = np.argsort(V)
    Vs, dts = V[order], dt[order]
    total = float(np.sum(dts))

    groups: list[np.ndarray] = []
    if binning == "speed":
        lo, hi = float(Vs[0]), float(Vs[-1])
        edges = np.linspace(lo, hi, n + 1)
        idx = np.clip(np.searchsorted(edges, Vs, side="right") - 1, 0, n - 1)
        groups = [np.nonzero(idx == b)[0] for b in range(n)]
    else:
        cum = np.cumsum(dts)
        cuts = np.searchsorted(cum, np.linspace(0.0, total, n + 1)[1:-1],
                               side="left")
        groups = [g for g in np.split(np.arange(Vs.size), cuts)]

    points = []
    for g in groups:
        if g.size == 0:
            continue
        w = float(np.sum(dts[g]))
        if w <= 0.0:
            continue
        v_bin = float(np.sum(Vs[g] * dts[g]) / w)
        pt = {"V": v_bin, "weight": w / total, "time_s": w,
              "q_Pa": car.q(v_bin)}
        if ride_height_m is not None:
            pt["ride_height_m"] = car.ride_height_at(ride_height_m, v_bin)
        points.append(pt)
    # renormalise after any empty bin was dropped, so the weights are a
    # distribution whatever the binning left behind
    wsum = sum(p["weight"] for p in points)
    for p in points:
        p["weight"] = p["weight"] / wsum
    points.sort(key=lambda p: p["weight"], reverse=True)
    return {
        "feasible": True, "reason": "",
        "points": points, "binning": binning,
        "V_mean_time_weighted": time_weighted_mean_speed(lap),
        "lap_time_s": lap["lap_time_s"],
        "heave": (car.heave_law.rigid is False),
    }


#: exponent of :func:`example_reynolds_cd_law`. DERIVED, not chosen: the
#: least-squares fit in log space to the two ratios
#: MISSION_DESIGN_WATER_TRACK.md section 3 measured, i.e. the p minimising
#: ``sum_i (ln r_i + p ln(V_i/30))^2`` over (55 m/s, -2.591 %) and
#: (100 m/s, -4.784 %). Written out: p = 0.0412.
EXAMPLE_CD_EXPONENT = 0.0412


def example_reynolds_cd_law(
        cd_a_ref: float, v_ref: float = 55.0,
        exponent: float = EXAMPLE_CD_EXPONENT) -> Callable[[float], float]:
    """``cd_a(V) = cd_a_ref (V/v_ref)^-exponent`` — the module's stated example.

    NOT a literature correlation and not a model of anything: a two-point fit
    to THIS repo's own measurement in MISSION_DESIGN_WATER_TRACK.md section 3,
    where ``car rear wing + endplates`` was measured at CD -2.591 % at 55 m/s
    relative to 30 and -4.784 % at 100 m/s relative to 30.

    Solving ``(V2/V1)^-p`` for each point separately gives p = 0.04331 and
    p = 0.04072, which do not agree — the drift is not exactly a power law —
    so the shipped :data:`EXAMPLE_CD_EXPONENT` is the least-squares fit to
    both, 0.0412, and it reproduces the two measurements as -2.466 % (against
    -2.591 %) and -4.839 % (against -4.784 %), i.e. to about a tenth of a
    percentage point of CD. The SIZE of it is what a turbulent plate friction
    going as ``Re^-0.2`` would give if about a fifth of the drag were plate
    friction, which is what the endplate family charges — that is a
    consistency check on the fit, not its derivation.

    It exists so that :func:`single_point_error` and the tests have ONE
    speed-dependent law that is traceable to a measurement, rather than a
    made-up one. A real caller passes their own solver's answers.
    """
    cd0 = float(cd_a_ref)
    v0 = float(v_ref)
    p = float(exponent)
    return lambda V: cd0 * (float(V) / v0) ** (-p)


def single_point_error(cz_a: Coefficient, cd_a: Coefficient,
                       car: CarSpec | None = None,
                       track: TrackSpec | None = None,
                       n_steps: int = N_STEPS_DEFAULT) -> dict:
    """What the single-point shortcut costs, MEASURED for the given laws.

    The shortcut is: evaluate the wing once, at the lap's time-weighted mean
    speed, and use those two numbers everywhere. This runs the lap both ways
    and returns the relative lap-time error

        (t_single - t_multi) / t_multi

    signed, so a positive number means the shortcut reports a SLOWER lap than
    the honest one. With constant coefficients the error is exactly 0.0 and
    the two lap times are bit-identical — asserted with ``==`` in the tests,
    because "the shortcut is free when nothing varies" is the property that
    makes it safe to leave on by default.

    What it is measured to be, on the reference car and lap, so that a caller
    knows what order of thing they are trading away. Every row runs
    :func:`example_reynolds_cd_law` at the SHIPPED
    :data:`EXAMPLE_CD_EXPONENT` (0.0412) except the last, which states its
    own exponent, and every row is re-derived from the module in
    tests/test_cartrack.py rather than pinned here as a literal:

    =================================  ==========  ==========
    case                               rel. error  lap error
    =================================  ==========  ==========
    reference car, published wing       3.467e-6    0.00021 s
    CdA_car 0.30, wing cd_a 0.02        7.685e-6    0.00045 s
    CdA_car 0.30, wing cd_a 0.10        3.815e-5    0.0023 s
    CdA_car 0.30, wing cd_a 0.30        1.123e-4    0.0068 s
    the same, but with exponent 0.5     1.236e-3    0.0746 s
    =================================  ==========  ==========

    (The first four rows are 1.0193x smaller than the table this docstring
    carried before, which had been measured with the exponent at 0.042
    against a shipped 0.0412 — a ratio of 1.0194. To first order the error is
    proportional to the exponent, because the exponent is what makes the
    coefficient vary at all, and the two ratios agreeing to four digits is
    that statement checked.)

    The mechanism the table shows is that the error is the product of two
    small things — the wing's SHARE of the car's total drag, and the
    coefficient's VARIATION over the lap's speed range — so on the reference
    car (a 0.0095 m^2 wing against a 0.90 m^2 car, drifting 4 % over the lap)
    the shortcut costs a fifth of a millisecond and the multi-point interface
    buys nothing. It is not therefore useless: the same table says the error
    grows by a factor of 32 (an order and a half) on a car whose wing IS its
    drag, and by 11 more if the coefficient really moved by the 0.5 power.
    The honest default is the shortcut, measured — which is what this
    function is for — and not the shortcut, assumed.
    """
    car = car or CarSpec()
    lap_multi = lap_time(cz_a, cd_a, car, track, n_steps=n_steps)
    if not lap_multi["feasible"]:
        return _fail(f"no multi-point lap: {lap_multi['reason']}")
    v_ref = time_weighted_mean_speed(lap_multi)
    cz_law, _ = _as_law(cz_a, "cz_a")
    cd_law, _ = _as_law(cd_a, "cd_a")
    lap_single = lap_time(float(cz_law(v_ref)), float(cd_law(v_ref)),
                          car, track, n_steps=n_steps)
    if not lap_single["feasible"]:
        return _fail(f"no single-point lap: {lap_single['reason']}")
    t_m = lap_multi["lap_time_s"]
    t_s = lap_single["lap_time_s"]
    return {
        "feasible": True, "reason": "",
        "V_ref": float(v_ref),
        "cz_a_ref": float(cz_law(v_ref)), "cd_a_ref": float(cd_law(v_ref)),
        "t_multi_s": float(t_m), "t_single_s": float(t_s),
        "rel_error": float((t_s - t_m) / t_m),
        "delta_s": float(t_s - t_m),
    }


# ---------------------------------------------------------------- balance


def aero_balance(cz_a: Coefficient, car: CarSpec | None = None,
                 V: float | None = None) -> dict:
    """Fraction of the TOTAL aerodynamic downforce carried by the front axle.

    Equation (6) of the module docstring. The frame is STATED by
    :class:`CarSpec` — the wheelbase, the wing's station, and the station of
    the rest of the car's own downforce — and is not derived, because it
    cannot be: a rear wing has no access to the denominator of a ratio over
    the whole car. That is exactly the argument
    MISSION_DESIGN_WATER_TRACK.md's E8 makes for why balance was deferred,
    and stating the frame is what un-defers it.

    ``V`` is optional and only matters when ``cz_a`` is speed-dependent: q
    cancels out of (6), so with constant coefficients the aero balance does
    not move with speed at all. A heave law that changes the wing's CZ with
    ride height is the mechanism that makes it move; passing the speed is how
    that arrives.

    ``load_balance`` is the OTHER balance and the one a driver feels: the
    front share of the total vertical axle load, weight included. It does
    depend on speed, running from the static ``1 - x_cg_frac`` at rest
    towards the aero balance as q grows. It needs ``V``.
    """
    reason = _coefficient_reason(cz_a, "cz_a")
    if reason:
        return _fail(reason, aero_balance=None)
    car = car or CarSpec()
    cz_law, cz_const = _as_law(cz_a, "cz_a")
    if not cz_const and V is None:
        # evaluating a speed-dependent law "at no speed" would silently pick
        # V = 0, where a Reynolds law is singular and every law is a guess
        return _fail(
            "cz_a is a speed-dependent law, so the balance has to be asked AT "
            "a speed: pass V. (With a constant cz_a, equation (6) has no q in "
            "it and the balance is the same at every speed.)",
            aero_balance=None)
    czw = float(cz_law(0.0 if V is None else float(V)))
    czc = float(car.cza_car_m2)
    total = czc + czw
    front = czc * (1.0 - float(car.x_cp_car_frac)) + czw * (
        1.0 - float(car.x_wing_frac))
    if total == 0.0:
        return _fail(
            "the car makes no net downforce (CzA_car + cz_a == 0), so the "
            "front axle's SHARE of it is not defined; state a car with a "
            "floor, or ask for the load balance instead",
            aero_balance=None)
    bal = float(front / total)
    out = {
        "feasible": True, "reason": "",
        "aero_balance": bal,
        "cza_total_m2": float(total),
        "cza_front_m2": float(front),
        "front_share_wing": float(1.0 - float(car.x_wing_frac)),
        "front_share_car": float(1.0 - float(car.x_cp_car_frac)),
        "static_front_frac": float(car.front_weight_frac),
        "load_balance": None, "V": (None if V is None else float(V)),
    }
    if V is not None:
        q = car.q(float(V))
        n_front = car.weight_n * car.front_weight_frac + q * front
        n_total = car.weight_n + q * total
        out["load_balance"] = (float(n_front / n_total) if n_total != 0.0
                               else None)
    return out


def balance_margin(cz_a: Coefficient, car: CarSpec | None = None,
                   V: float | None = None) -> dict:
    """The aero balance as CONSTRAINABLE margins against a stated window.

    Two margins in the package's convention (feasible iff ``>= 0``), because
    which END binds is the whole information — MISSION_DESIGN_WATER_TRACK.md
    section 2 makes the same point about the water card's area window, and a
    single collapsed margin cannot say "too rearward" versus "too forward":

        ``g_low  = balance - window_lo``   too REARWARD when negative
        ``g_high = window_hi - balance``   too FORWARD when negative

    ``g`` is their minimum (what a scalar-constrained harness wants) and
    ``binding`` names which one it was. Both are already dimensionless
    fractions of the total downforce, so nothing is normalised and a margin
    of -0.12 means twelve points of balance.

    The window is :meth:`CarSpec.balance_window_used`: the caller's, or the
    derived default of equation (7) — the static front weight fraction, where
    the front/rear grip split does not change with speed — plus/minus
    :data:`BALANCE_HALF_WIDTH`, which is the only calibration in it.
    """
    car = car or CarSpec()
    bal = aero_balance(cz_a, car, V)
    if not bal["feasible"]:
        return _fail(bal["reason"], g=None, g_low=None, g_high=None)
    lo, hi = car.balance_window_used()
    b = bal["aero_balance"]
    g_low = float(b - lo)
    g_high = float(hi - b)
    g = min(g_low, g_high)
    return {
        "feasible": True, "reason": "",
        "aero_balance": b, "window": (float(lo), float(hi)),
        "g_low": g_low, "g_high": g_high, "g": float(g),
        "binding": (None if g >= 0.0 else ("low" if g_low < g_high else "high")),
        "load_balance": bal["load_balance"], "V": bal["V"],
        "static_front_frac": bal["static_front_frac"],
    }
