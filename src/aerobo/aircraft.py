"""Tier B: weight-coupled aircraft wing sizing — span and area become design
variables, and structure closes the span trap.

The trap this module exists to avoid
------------------------------------
Tier A fixes b and S, so L/D depends only on shape. Free the span in a
pure-aero model and the optimiser runs away: at fixed lift L the induced
drag is (Prandtl lifting-line)

    D_i = L^2 / (q pi b^2 e)   ~  1 / b^2,

monotone decreasing in span, while the profile drag ~ q S cd is span-blind
— so "max L/D with free b" slams the span upper bound no matter where the
box is put. The bound, not the physics, would pick the answer.

The classical fix (Prandtl 1933, "Über Tragflügel kleinsten induzierten
Widerstandes"; Raymer, *Aircraft Design: A Conceptual Approach*, Ch. 15)
is that span costs STRUCTURE: the wing bending material grows with span,
so the wing gets heavier, the total lift requirement rises, and the
induced drag — quadratic in weight — climbs back up:

    W_total(b) = W_fixed + W_wing(b, ...)        (weights.total_weight,
                 fixed point: Raymer's statistical wing weight takes the
                 design gross weight W_dg = W_total as an input)
    D_i(b)  ~  W_total(b)^2 / b^2.

W^2 against 1/b^2 produces a strictly INTERIOR optimum span b* (gated
numerically in tests/test_aircraft.py: a 25-point span sweep must peak
away from both box edges, and f at the top bound must be worse than the
peak — the trap is closed by physics, not by the box).

Honesty label on WHERE the trap closes: Raymer's statistical GA wing
weight is span-soft (W_wing ~ b^1.2 at fixed S through the A^0.6 term),
so the drag-vs-weight balance only turns over at sailplane-like
proportions — the calibrated unconstrained optimum sits at b* ~ 30 m,
AR ~ 60 (open-class-glider territory; gliders ARE what pure minimum-drag
aerodynamics builds). That is a physically correct interior optimum,
reached inside the deliberately generous B_BOUNDS below. What pulls the
design back to GA proportions is the STRUCTURE: root bending stress
grows ~ b^4 at fixed S, so the g = 0 boundary cuts the span back to
~11-13 m and the constrained optimum RIDES the boundary — the
hydrofoil-style active-constraint story, with a clean physical reading
(aerodynamics wants a glider; the spar makes it an aircraft).

Objective — why "payload L/D", not L/D (a derived decision)
-----------------------------------------------------------
With W varying across candidates, "maximise L/D" is NO LONGER "minimise
drag": a heavier wing that raises L/D would be rewarded even though the
aircraft needs more thrust to fly. The mission objective (mission.py) is
minimum drag = minimum thrust = minimum power at fixed V:

    L = W_total  (trim)  =>  D = W_total / (L/D),   P = D V.

We report and maximise the dimensionless

    f = W_fixed / D_total       ("payload L/D"),

which is a monotone transform of 1/D because W_fixed is the SAME constant
for every candidate — maximising f IS minimising D IS minimising thrust
and power at fixed V. Equivalence chain to the legacy objective:

    f = W_fixed / D = (W_fixed / W_total) * (L/D)  -->  L/D  as W_wing -> 0,

so f reduces EXACTLY to the Tier A L/D when the wing is weightless
(regression-gated bit-for-bit in tests/test_aircraft.py by solving the
mission weight so that CL_target = 0.5 at the legacy b = S = 10 probe).

D here is the WING's drag only (cd0_extra = 0 in the reused Problem —
deliberate): W_fixed carries the fuselage/tail/systems WEIGHT but their
parasite drag is not modelled at this tier. The asymmetry (weight counted,
drag not) is a documented reduced-order choice; hook non-wing parasite
drag through cd0_extra when a fuselage model arrives.

Constraint — root bending, LOG-normalised (a measured calibration choice)
--------------------------------------------------------------------------
The spar must carry the ultimate-load root bending moment
(weights.root_bending_stress, n_ult = limit x 1.5). The margin reported
to the constrained optimiser is the LOG stress ratio

    g = ln(sigma_allow / sigma_root),        feasible iff g >= 0,

NOT the linear normalised margin (sigma_allow - sigma_root)/sigma_allow.
Both share the boundary, the sign, and the first-order behaviour AT the
boundary (ln(sigma_allow/sigma) = -ln(1 - g_lin) ~ g_lin as g_lin -> 0),
and both decrease monotonically with span (longer arm, heavier wing,
thinner root). They differ in the tail — and the tail is what the
constraint GP in optimize/constrained.py has to regress: sigma ~ b^4
spans two decades over the span box, so the linear margin spans
[-2000, +0.9] and its GP standardisation flattens the feasible island
into numerical noise. Measured on this exact problem (2026-07-12):
constrained BO with the linear margin found 0 feasible points in 32
evaluations (seed 0); with the log margin it finds 3 within the 12-eval
smoke budget (tests/test_aircraft.py). The log margin spans a genuinely
GP-friendly [-7.6, +1.5] over the box — the same O(1) scale as the
hydrofoil cavitation margin. The linear margin is still reported in the
breakdown as ``g_sigma_lin`` for engineering readability.

``sigma_allow`` defaults to the raw material allowable
weights.SIGMA_ALLOW_PA (2024-T3, 345 MPa yield / 1.5 = 230 MPa). No
problem-level derating is needed: with the generous span box the
unconstrained optimum lives at b ~ 30 m where sigma_root is ~ 100x the
allowable (sigma ~ b^4), so the constraint is emphatically ACTIVE — the
2000-point Sobol calibration scan (class docstring) shows the best
feasible design sitting essentially ON g = 0. Override per-problem via
the dataclass field.

Design vector and conventions
-----------------------------
    x = [taper, twist_root_deg, twist_tip_deg, b_m, S_m2, tc]      (d = 6)

Twist is local incidence, + = nose-up, washout = negative tip twist
(geometry.py convention). Sweep is FIXED at 0 deg here — sweep remains a
tier_a_plus design variable; the Raymer weight formula still receives
sweep_deg = 0 explicitly so the tier_a_plus merge is a one-line change.
g >= 0 feasible; f is MAXIMISED.

Default mission: GA-scale MissionSpec(W_N=12000 N, V=50 m/s, sea level).
W_N is the FIXED (non-wing) weight — payload + fuselage + tail + systems
(~1223 kg, light-twin class), NOT the total: the total
W(x) = W_fixed + W_wing(x) comes from weights.total_weight per candidate.
W_N was CALIBRATED UP from the 6000 N starting point: at 6 kN the area
optimum S* sat on the S lower bound (too little lift-dependent profile
drag to price wing area); at 12 kN both b* and S* are strictly interior
(class docstring). Re_mac spans ~0.7-13e6 across the box (~1.6e6 at the
calibrated optimum region) while the section polars are XFOIL at
Re = 1e6 — flagged limitation, same precedent as the hydrofoil
(Re 0.9-1.6e6 vs 1e6, hydrofoil.py docstring).

Failure contract (objective.py / hydrofoil.py): solver failure -> exactly
(PENALTY, G_FAIL); solvable-but-overstressed -> TRUE f with signed g < 0
(constrained BO must see the violation magnitude, never a penalty).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from . import geometry, materials, weights
from .sizing import PAYLOAD_LOD_UNITS
from .mission import MissionSpec
from .objective import PENALTY, Problem, _fail, evaluate_geometry
from .weights import N_ULT_DEFAULT, SIGMA_ALLOW_PA

G_FAIL = -1.0        # finite infeasible margin reported on solver failure

SWEEP_FIXED_DEG = 0.0   # sweep stays a tier_a_plus variable; the weight
#                         formula still takes sweep_deg explicitly (see above)


@dataclass
class AircraftProblem:
    """Tier B weight-coupled wing-sizing problem (module docstring).

    Design vector (d = 6):

        x = [taper, twist_root_deg, twist_tip_deg, b_m, S_m2, tc]

    Objective (MAXIMISE): f = W_fixed / D_total ("payload L/D").
    Constraint: g = ln(sigma_allow / sigma_root) >= 0 (module docstring).

    Calibration (all numbers re-verified numerically 2026-07-12):
    * Span sweep at [taper 0.45, tw 1.0/-2.0, b, S=14, tc=0.12] over
      B_BOUNDS, 25 points: strict interior maximum at b* = 30.08 m
      (index 17/24), f* = 47.57; f(6) = 11.18, f(40) = 44.27 — the weight
      loop closes the span trap inside the box (sailplane-like optimum,
      module docstring honesty label). g_sigma strictly decreases along
      the sweep (+1.63 -> -6.44 in log units).
    * 2000-point Sobol scan of f (g ignored): best f = 49.72 at
      b = 31.9 m, S = 12.1 m2 with sigma_root = 134x sigma_allow
      (g = -4.90) — the unconstrained optimum is deep in the infeasible
      zone, i.e. the constraint is emphatically ACTIVE. 165/2000 points
      feasible (8.2%; 44 solver failures). Best FEASIBLE scan point:
      f = 27.16 at b = 13.2 m with g = +0.36; runners-up at g = +0.098
      and +0.044 — the good feasible designs RIDE the g = 0 boundary,
      hydrofoil-style.
    * W_N = 12000 N (calibrated up from 6000: see module docstring);
      at 12 kN both b* and S* (~14 m2) are strictly interior.
    """

    mission: MissionSpec = field(
        default_factory=lambda: MissionSpec(W_N=12000.0, V=50.0, altitude_m=0.0)
    )
    n_ult: float = N_ULT_DEFAULT
    N: int = 60
    sigma_allow: float = SIGMA_ALLOW_PA
    material: object = None   # WHAT THE WING IS BUILT OF
    #   (materials.Material, or a key from materials.MATERIALS).
    #   None = 2024-T3 aluminium, the material Raymer's correlation
    #   was regressed over, so every published run is bit-identical.
    #   It sets BOTH the allowable the spar is sized to and the
    #   factor the statistical wing weight is scaled by — the second
    #   is the one the span answer actually rides (materials.py).
    polar_family: Any = None      # t/c-indexed NACA 24XX family; lazy
    chord_order: int = 0          # free chord law (geometry.py); 0 = straight
    #   taper, no extra design variables, bit-for-bit the published problem.
    #   NOTE the weight model is planform-blind beyond (b, S, taper, t/c):
    #   Raymer's statistical wing weight has no chord-distribution term, so a
    #   chord law changes the AERODYNAMICS and the trim target, not W_wing.
    chord_max_frac: float = geometry.CHORD_COEFF_BOUND
    chord_law: str = geometry.DEFAULT_CHORD_LAW   # WHICH SHAPE the
    #   coefficients above describe (geometry.CHORD_LAWS). Changes what
    #   the design vector DRAWS, never how long it is.
    chord_limits: "geometry.ChordLimits | None" = None   # what the chord
    #   DISTRIBUTION has to obey, in metres and degrees (geometry.ChordLimits:
    #   a minimum chord, a maximum, a maximum local taper ANGLE, and which end
    #   is the wide one). None = unconstrained, which is every published run.
    #   Checked where the planform is built, so a violation is an in-contract
    #   failure (the penalty contract), never an exception at the optimiser.
    #: the aspect-ratio band the USER will accept b^2/S in, as
    #: ``(ar_min, ar_max)`` with either end None (api.AR_LIMIT_KEYS).
    #: A SECOND band beside ``sizing.AR_LIMITS``, never a replacement
    #: for it, and None — no limit at all — is every published run.
    ar_limits: "tuple | None" = None
    flight_free: bool = False     # speed + altitude as design variables
    #   (geometry.py's flight modifier), on TOP of the free span and area.
    #   This is the combination the sizing loop is actually about: the wing
    #   is weighed at the candidate's own dynamic pressure (Raymer's q^0.006
    #   term), the trim target is W_total/(q S) at that state, and the
    #   structural constraint therefore moves with the flight state as well
    #   as with the planform.

    #: cruise-speed box for the flight modifier. NOT the Tier A (10, 25)
    #: band: this problem's calibrated design point is 50 m/s, which that
    #: band does not even contain. Chosen around it so the published point
    #: is strictly interior; Re_mac then spans the same order the module
    #: docstring already flags against the fixed-Re section polars.
    FLIGHT_V_BOUNDS = (30.0, 80.0)   # [m/s]

    B_BOUNDS = (6.0, 40.0)        # [m]  span; generous ON PURPOSE — the
    #   unconstrained optimum (b* ~ 30 m, sailplane-like) must be interior
    #   so the box never picks the answer; gated in tests
    S_BOUNDS = (8.0, 22.0)        # [m2] area; S* ~ 14 interior at W_N = 12 kN

    def __post_init__(self):
        if self.mission.medium != "air":
            raise ValueError(
                "AircraftProblem is air-only (water lifting surfaces belong "
                "to hydrofoil.HydrofoilProblem)"
            )
        if self.polar_family is None:
            from .polar import default_polar_family
            self.polar_family = default_polar_family()

    @property
    def bounds(self) -> np.ndarray:
        box = geometry.with_flight_bounds(np.array([
            geometry.TAPER_BOUNDS,
            geometry.TWIST_ROOT_BOUNDS_DEG,
            geometry.TWIST_TIP_BOUNDS_DEG,
            self.B_BOUNDS,
            self.S_BOUNDS,
            geometry.TC_BOUNDS,
        ], dtype=float), self.flight_free, v_bounds=self.FLIGHT_V_BOUNDS)
        return geometry.with_chord_bounds(box, self.chord_order,
                                          self.chord_max_frac,
                                          self.chord_law)

    @property
    def dim(self) -> int:
        return int(self.bounds.shape[0])


def evaluate_aircraft(x: np.ndarray, prob: AircraftProblem | None = None) -> dict:
    """Full Tier B evaluation with breakdown; never raises for in-contract
    failures. ``feasible`` refers to the SOLVER (PENALTY contract); the
    structural constraint is reported separately as signed ``g_sigma``.

    Pipeline: bounds check -> weight fixed point (weights.total_weight) ->
    CL_target = W_total / (q S) -> t/c-selected polar -> objective.
    evaluate_geometry on a locally-built trim Problem — the aero core is
    the IDENTICAL Tier A code path (bit-for-bit, regression-gated), only
    the trim target now carries the candidate's own wing weight.
    """
    prob = prob or AircraftProblem()
    x = np.asarray(x, dtype=float)
    bnds = prob.bounds
    if x.shape != (bnds.shape[0],):
        return _fail(f"design vector shape {x.shape} != ({bnds.shape[0]},)")
    if np.any(x < bnds[:, 0] - 1e-12) or np.any(x > bnds[:, 1] + 1e-12):
        return _fail("bounds violation")

    taper, tw_root, tw_tip, b, S, tc = (float(v) for v in x[:6])

    # ---- flight state (geometry.py's flight modifier): the candidate's own
    # (V, altitude) becomes THE mission this design is weighed, trimmed and
    # scored at — one spec, so the weight loop's q and the trim target
    # cannot disagree about where the aeroplane is flying.
    flight = geometry.flight_from_x(x, prob.flight_free, prob.chord_order)
    if flight is not None:
        try:
            prob = replace(prob, flight_free=False,
                           mission=replace(prob.mission, V=flight[0],
                                           altitude_m=flight[1]))
        except ValueError as exc:      # ISA validity; unreachable in the box
            return _fail(f"flight state: {exc}")

    try:
        coeffs = geometry.chord_coeffs_from_x(x, prob.chord_order,
                                              prob.chord_law)
        geometry.Wing(b=b, S=S, taper=taper, chord_coeffs=coeffs,
                      chord_limits=prob.chord_limits)
    except ValueError as exc:      # collapsed chord law — penalty contract
        return _fail(f"planform: {exc}")

    try:
        mat = materials.resolve(prob.material)
        W_total, W_wing = weights.total_weight(
            prob.mission.W_N, b, S, taper, SWEEP_FIXED_DEG, tc, prob.n_ult,
            q_Pa=prob.mission.q,   # weigh at the ACTUAL mission q (q^0.006 term)
            k_w=mat.k_w,           # ...and of the MATERIAL it is built of
        )
    except (ValueError, RuntimeError, OverflowError) as exc:
        return _fail(f"weight fixed point: {exc}")
    if not (np.isfinite(W_total) and W_total > 0.0):
        return _fail("non-finite/non-positive total weight")

    q = prob.mission.q
    CL_target = W_total / (q * S)

    try:
        pol = prob.polar_family.at(tc)
    except ValueError as exc:
        return _fail(f"polar family: {exc}")

    wing = geometry.Wing(b=b, S=S, taper=taper, twist_root_deg=tw_root,
                         twist_tip_deg=tw_tip, tc=tc, chord_coeffs=coeffs,
                         chord_limits=prob.chord_limits)
    _, c, twist_rad = wing.sample(prob.N)

    # Local trim Problem = the Tier A aero core with this candidate's flow
    # state and weight-coupled trim target (mission=None: CL_target is set
    # HERE from W_total, not from the fixed weight — Problem.__post_init__
    # would overwrite it with W_fixed/(qS), which is the wrong lift).
    aero_prob = Problem(
        b=b, S=S, N=prob.N, V=prob.mission.V, rho=prob.mission.rho,
        mu=prob.mission.mu, CL_target=CL_target, mode="trim", polar=pol,
    )
    out = evaluate_geometry(c, twist_rad, aero_prob, alpha_deg=None,
                            mac=wing.mac, sweep_deg=SWEEP_FIXED_DEG, polar=pol)
    if not out["feasible"]:
        return out

    # the spar box is sized on the FLOWN root chord: with a chord law the
    # root section is no longer the trapezoid's (weights.py docstring)
    sigma = weights.root_bending_stress(
        b, S, taper, tc, W_total, prob.n_ult,
        c_root=float(wing.chord(np.array([0.0]))[0]))
    # LOG stress margin (module docstring: measured GP-scale choice); the
    # linear margin is kept alongside for engineering readability.
    # the material owns the allowable; ``prob.sigma_allow`` is the override
    # a study can still set, exactly as in sizing.sized_state
    sigma_allow = (mat.sigma_allow_Pa if prob.material is not None
                   else prob.sigma_allow)
    g_sigma = float(np.log(sigma_allow / sigma))
    g_sigma_lin = (sigma_allow - sigma) / sigma_allow

    D_N = q * S * out["CD"]                  # drag force at trim [N]
    f = prob.mission.W_N / D_N               # payload L/D (module docstring)
    if not (np.isfinite(f) and np.isfinite(g_sigma)):
        return _fail("non-finite payload L/D or stress margin")

    out.update({
        "score": float(f),
        "f": float(f),
        # ...and WHAT that number is. This family has no unsized twin, so
        # there is no configuration in which its score is an aero L/D — and
        # ``out`` already carries "LoD" from evaluate_geometry, higher by
        # exactly W_fixed/W_total.
        "score_units": PAYLOAD_LOD_UNITS,
        "W_total_N": float(W_total),
        "W_wing_N": float(W_wing),
        "W_fixed_N": float(prob.mission.W_N),
        "CL_target": float(CL_target),
        "sigma_root_Pa": float(sigma),
        "sigma_allow_Pa": float(sigma_allow),
        **mat.report(),
        "g_sigma": float(g_sigma),
        "g_sigma_lin": float(g_sigma_lin),
        "g": float(g_sigma),
        "D_N": float(D_N),
        "wing": wing,
        "sweep_deg": SWEEP_FIXED_DEG,
        "tc": tc,
    })
    if flight is not None:
        out.update({"V": float(prob.mission.V),
                    "altitude_m": float(flight[1]),
                    "rho": float(prob.mission.rho),
                    "mu": float(prob.mission.mu)})
    return out


def fg_aircraft(x: np.ndarray, prob: AircraftProblem | None = None
                ) -> tuple[float, float]:
    """Constrained-harness callable: (payload L/D, log stress margin).

    Solvable-but-overstressed designs return their TRUE f with g < 0;
    solver failures return exactly (PENALTY, G_FAIL) — the
    hydrofoil.fg_hydrofoil contract."""
    out = evaluate_aircraft(x, prob)
    if not out["feasible"]:
        return PENALTY, G_FAIL
    return float(out["f"]), float(out["g_sigma"])
