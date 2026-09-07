"""Mission layer: design weight + speed + altitude/depth -> flow state + trim CL.

Why a mission spec at all — the fixed-lift equivalence
------------------------------------------------------
Every Tier objective in this framework trims to a CL target and maximises
L/D. The mission layer grounds that convention in the physical task
"sustain a weight W at speed V with minimum drag":

    steady level flight:   L = W          (weight fixed by the mission)
    trim target:           CL_target = W / (q S),   q = 1/2 rho V^2
    drag at trim:          D = W / (L/D)
    power at fixed V:      P = D V = W V / (L/D)

With W and V fixed, D and P are both monotone DECREASING in L/D, so
"maximise L/D at CL_target = W/(q S)" is EXACTLY "minimise drag" AND
"minimise power at fixed V" for the mission — the existing trim-mode
objective is the mission objective, it just had CL_target hard-coded.
MissionSpec closes the loop: the single physical input is the design
weight W_N, and CL_target follows from wherever the craft flies (V, h)
or sails (V, depth). This is what makes V and altitude legitimate DESIGN
VARIABLES in objective mode="mission": a candidate that flies higher/
slower carries the SAME weight at a different (rho, q), hence a
different CL_target — the classic speed/loading coupling (same pattern
as HydrofoilProblem.CL_target(V) in hydrofoil.py).

Air branch: ISA troposphere + lower stratosphere
------------------------------------------------
International Standard Atmosphere (ISO 2533 / US Standard Atmosphere
1976, first two layers):

    troposphere  (0 <= h <= 11 km):  T(h) = T0 - L h,
                                     p(h) = p0 (T/T0)^(g/(R L))
    lower stratosphere (11-20 km):   T = 216.65 K (isothermal),
                                     p(h) = p11 exp(-g (h - h11)/(R T))
    both layers:                     rho = p / (R T)

with T0 = 288.15 K, p0 = 101325 Pa, L = 6.5 K/km, R = 287.053 J/(kg K),
g = 9.80665 m/s^2. Valid to 20 km; a ValueError is raised beyond (the
next ISA layer has a different lapse rate — extend here if ever needed).
Pressure and temperature are continuous at the 11 km tropopause by
construction (p11 is the troposphere formula evaluated at 11 km).
Altitudes are treated as GEOPOTENTIAL (the variable the USSA-1976 layer
formulas are defined in); the geometric/geopotential distinction is
neglected — error < 0.05% below 20 km, ~0.014% at the 3 km mission bound.

Dynamic viscosity via Sutherland's law (Sutherland 1893; White,
*Viscous Fluid Flow*, air constants):

    mu(T) = 1.716e-5 * (T / 273.15)^1.5 * (273.15 + 110.4) / (T + 110.4)

Water branch
------------
rho and mu come from hydrofoil.py (``water_properties`` over
``WATER_KINDS`` — the single source of truth for the water constants, sea
or fresh, carried together with the vapour pressure; they are NOT redefined
here). At Tier level both are CONSTANT WITH DEPTH: water
compressibility (~0.5% per 100 m) and temperature stratification are
neglected — a documented reduced-order choice. ``depth_m`` is carried on
the spec because depth-dependent PHYSICS (free-surface image, cavitation
static head) lives in hydrofoil.py, not in the fluid properties.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

# ---------------------------------------------------------------- ISA constants

T0_ISA = 288.15          # K,  sea-level standard temperature
P0_ISA = 101325.0        # Pa, sea-level standard pressure
LAPSE_TROPO = 0.0065     # K/m, tropospheric lapse rate
R_AIR = 287.053          # J/(kg K), specific gas constant of air
GAMMA_AIR = 1.4          # ratio of specific heats, air (calorically perfect)
G0 = 9.80665             # m/s^2, standard gravity
H_TROPOPAUSE_M = 11000.0  # troposphere / lower-stratosphere boundary
T_STRATO = 216.65        # K, isothermal lower stratosphere (11-20 km)
H_ISA_MAX_M = 20000.0    # model validity ceiling (next layer not implemented)

# Sutherland's law, air (Sutherland 1893; White, Viscous Fluid Flow)
MU_REF_SUTH = 1.716e-5   # Pa s at T_REF_SUTH
T_REF_SUTH = 273.15      # K
S_SUTH = 110.4           # K, Sutherland constant for air


# ---------------------------------------------------------------- ISA functions


def isa_temperature(h_m: float) -> float:
    """ISA temperature [K] at (geopotential) altitude h_m [m]; valid to 20 km."""
    if h_m > H_ISA_MAX_M:
        raise ValueError(
            f"altitude {h_m} m above ISA model validity ({H_ISA_MAX_M:.0f} m)"
        )
    if h_m <= H_TROPOPAUSE_M:
        return T0_ISA - LAPSE_TROPO * h_m
    return T_STRATO


def isa_pressure(h_m: float) -> float:
    """ISA pressure [Pa]: barometric in the troposphere, exponential above."""
    T = isa_temperature(h_m)          # also enforces the 20 km validity limit
    exponent = G0 / (R_AIR * LAPSE_TROPO)
    if h_m <= H_TROPOPAUSE_M:
        return P0_ISA * (T / T0_ISA) ** exponent
    p11 = P0_ISA * (T_STRATO / T0_ISA) ** exponent   # continuous at 11 km
    return p11 * float(np.exp(-G0 * (h_m - H_TROPOPAUSE_M) / (R_AIR * T_STRATO)))


def isa_density(h_m: float) -> float:
    """ISA density [kg/m^3] via the ideal gas law rho = p / (R T)."""
    return isa_pressure(h_m) / (R_AIR * isa_temperature(h_m))


def speed_of_sound(h_m: float) -> float:
    """Speed of sound [m/s] at ISA altitude ``h_m``: a = sqrt(gamma R T).

    Calorically perfect air, the same ISA temperature the density and the
    viscosity above are taken at — so a Mach number formed with it cannot
    disagree with the rest of the operating point. It exists because a
    shell that asks for a speed and an altitude already knows the Mach
    number, and making the user type it would let the two drift apart.
    """
    return float(np.sqrt(GAMMA_AIR * R_AIR * isa_temperature(h_m)))


def sutherland_mu(T_K: float) -> float:
    """Dynamic viscosity of air [Pa s] by Sutherland's law (module docstring)."""
    return MU_REF_SUTH * (T_K / T_REF_SUTH) ** 1.5 * (T_REF_SUTH + S_SUTH) / (T_K + S_SUTH)


# ---------------------------------------------------------------- mission spec


@dataclass
class MissionSpec:
    """Design weight + operating point -> (rho, mu, q, CL_target).

    Fields:
        W_N        design weight [N] the surface must sustain (L = W_N)
        V          speed [m/s]
        altitude_m altitude [m] (air branch; ISA, valid 0-20 km)
        depth_m    submergence depth [m] (water branch; properties are
                   depth-independent at Tier level — see module docstring)
        medium     "air" | "water"
    """

    W_N: float
    V: float
    altitude_m: float = 0.0
    depth_m: float | None = None
    medium: str = "air"
    water: str = "sea"    # which water (hydrofoil.WATER_KINDS); sea = the
    #                       published default, so rho/mu are unchanged

    def __post_init__(self):
        if self.medium not in ("air", "water"):
            raise ValueError(f"unknown medium {self.medium!r} (air | water)")
        if self.medium == "water":
            from .hydrofoil import water_properties
            water_properties(self.water)      # refuse an unknown water here

    @property
    def rho(self) -> float:
        """Density [kg/m^3]: ISA at altitude_m (air) or the chosen water."""
        if self.medium == "water":
            from .hydrofoil import water_properties
            return water_properties(self.water)["rho"]
        return isa_density(self.altitude_m)

    @property
    def mu(self) -> float:
        """Dynamic viscosity [Pa s]: Sutherland (air) or the chosen water."""
        if self.medium == "water":
            from .hydrofoil import water_properties
            return water_properties(self.water)["mu"]
        return sutherland_mu(isa_temperature(self.altitude_m))

    @property
    def q(self) -> float:
        """Dynamic pressure 1/2 rho V^2 [Pa] at the mission state."""
        return 0.5 * self.rho * self.V**2

    def cl_target(self, S: float) -> float:
        """Trim lift coefficient CL_target = W_N / (q S) for reference area S."""
        return self.W_N / (self.q * S)


def flight_state(spec: MissionSpec, V: float, altitude_m: float,
                 S_ref: float) -> dict:
    """Flow state + trim target a design weight implies at ``(V, altitude)``.

    The ONE place the flight modifier's physics lives (geometry.py holds its
    vector layout): every family that carries the modifier calls this with
    its own reference area and replaces ``V / rho / mu / CL_target`` on a
    COPY of itself. Keeping it here is what stops five families from
    drifting into five slightly different definitions of "flying faster".

    Air-only, and it says so: the second variable is an ALTITUDE fed to the
    ISA model and the section polars are air tables. A water mission raises
    — the water families already carry speed AND depth as design variables
    (hydrofoil.py), so there is nothing to add there and silently accepting
    one would report an air atmosphere under the sea.
    """
    if spec.medium != "air":
        raise ValueError(
            "a free flight state is air-only (altitude design variable, air "
            "polars); water problems already fly speed and depth as design "
            "variables — see hydrofoil.HydrofoilProblem")
    cand = replace(spec, V=float(V), altitude_m=float(altitude_m))
    return {"V": cand.V, "rho": cand.rho, "mu": cand.mu,
            "CL_target": cand.cl_target(float(S_ref))}


def weight_for(CL_target: float, V: float, S_ref: float) -> float:
    """Design weight a family's own fixed trim point already implies.

    ``W = CL_target * q * S`` at SEA-LEVEL ISA density — the density the
    MissionSpec itself will use at altitude 0, not the RHO_SL literal — so
    that :func:`flight_state` at ``(V, 0)`` returns the same CL_target back:
    exactly for the ubiquitous 0.5 default (halving is exact in IEEE-754),
    to within an ulp otherwise. This is what makes switching the flight
    modifier ON reproduce the fixed-state problem at its own design point.
    """
    q0 = 0.5 * isa_density(0.0) * float(V) ** 2
    return float(CL_target) * (q0 * float(S_ref))


def design_weight_n(prob) -> float:
    """The design weight a problem flies — STATED first, derived only as a
    fallback.

    :func:`weight_for` back-derives a weight from a fixed-CL trim point at
    SEA-LEVEL density, which is exactly right for a problem that has no
    mission: it is what makes switching the flight modifier on reproduce the
    fixed-state problem. It is exactly WRONG for a problem that has one.

    ``Problem.__post_init__`` derives ``CL_target`` from the mission at the
    MISSION's altitude, so un-deriving it at sea level does not cancel: the
    two densities leave ``W = W_N * rho_0 / rho(h)``. Measured at
    ``W_N = 2000 N``: 2694.91 N at 3000 m (x1.3475, the density ratio to the
    last bit) and 4665.18 N at 8000 m (x2.333). That weight is FLOWN, not
    merely reported — the trim target is built from it — so a mission stated
    at altitude flew an aircraft heavier than the one the user asked for, and
    scored it against the inflated payload.

    Order: the explicit ``W_fixed_N``, then the mission's own ``W_N``, then
    the sea-level derivation. The first two are statements; only the third is
    an inference.
    """
    w = getattr(prob, "W_fixed_N", None)
    if w is not None:
        return float(w)
    spec = getattr(prob, "mission", None)
    if spec is not None and getattr(spec, "W_N", None) is not None:
        return float(spec.W_N)
    return weight_for(prob.CL_target, prob.V, prob.S)


def default_mission(S: float = 10.0) -> MissionSpec:
    """MissionSpec reproducing the legacy Problem defaults bit-for-bit.

    Chosen so cl_target(10.0) == 0.5 EXACTLY at sea level with V = 14.6
    (the Problem defaults). W_N is the exact expression

        W_N = 0.5 * q_sl * S,   q_sl = 1/2 rho_ISA(0) * 14.6^2

    using this module's OWN sea-level ISA density (~= 1.22499946, i.e.
    1.225 to 5e-7 relative — R = 287.053 vs the ISA-defining 287.05287),
    NOT the rounded literal 1.225: cl_target then evaluates as
    (0.5 * (q S)) / (q S), which is exactly 0.5 in IEEE-754 (halving is
    exact), so Problem(mission=default_mission()) trims to the identical
    CL_target = 0.5 and the legacy objective is preserved bit-for-bit
    (regression-gated in tests/test_mission.py). Numerically
    W_N ~= 652.80 N (~66.6 kg at g0).
    """
    V = 14.6
    q_sl = 0.5 * isa_density(0.0) * V**2
    return MissionSpec(W_N=0.5 * (q_sl * S), V=V, altitude_m=0.0)
