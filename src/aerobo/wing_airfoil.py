"""Coupled wing + airfoil design: two couplings of section shape into the
wing loop, built to measure WHAT DECOMPOSITION BUYS.

The engineering question
------------------------
A wing optimiser wants to command the section it flies. The classic
industrial answer is HIERARCHICAL DECOMPOSITION (Sobieszczanski-Sobieski &
Haftka 1997): the wing level commands only the two quantities it actually
needs from a section — structural depth t/c and design lift cl — and a
section-level team (here: the constrained section BO of
scripts/gen_cst_library.py) delivers a pre-optimised airfoil for each
(t/c, cl) demand. The alternative is the FLAT coupling: put every section
shape parameter straight into the wing design vector and let one optimiser
fight the combined dimensionality with the full-fidelity section solver in
the loop. This module implements BOTH so the experiments
(experiments/coupled_bo.py) can price the difference in budget and wall
time — the crossover-thesis question asked on a coupled problem.

A) WingAirfoilProblem — summary-variable coupling (d = 5, DEFAULT tool path)
----------------------------------------------------------------------------
    x = [taper, twist_root_deg, twist_tip_deg, tc_sec, cl_sec]

Tier A operating point (b = S = 10, V = 14.6 m/s, trim to CL_target = 0.5)
so the score is DIRECTLY comparable to the tier_a_plus 5-D headline
L/D = 37.57 (PROGRESS.md item 2: [taper, twists, sweep, tc] on the NACA
24XX family) — same aero core, same trim target, same d; only the two
extra dimensions differ (sweep + NACA-thickness there, section summary
variables into the BO-optimised CST library here).

``tc_sec`` and ``cl_sec`` index the pre-optimised CST section library
through polar.PolarFamily2D (bounds = the library hull, 0.09-0.15 x
0.3-0.7). Semantics of ``cl_sec``: it selects WHICH pre-optimised section
the wing gets (each library member was drag-minimised AT its own design
cl), it does not force the wing to operate there — the wing still trims to
CL_target, and a section flown off its design point pays its off-design
drag through the polar. That mismatch pricing is exactly the coupling the
problem exposes to the optimiser.

Honesty labels: intermediate (tc_sec, cl_sec) polars are bilinear
INTERPOLATIONS of the 4 surrounding members (PolarFamily2D docstring),
never XFOIL runs; and alpha_valid of a blend is the INTERSECTION of its
corners' converged ranges, so the two low-cl members whose HIGH-alpha
branch stalls ((0.09, 0.3) above +7 deg, (0.15, 0.3) above +6.5 deg — XFOIL
will not converge past the pre-stall knee there; both NEGATIVE branches are
complete to -4 deg) cap the high-alpha feasible window of every blend
touching them. The grid-centre member (0.12, 0.5) was RE-OPTIMISED under the
generator's sweep-completeness gate: its 64-eval drag optimum (cd 30.1 ct)
was convergence-hostile at negative alpha (1/9 converged even marched from
0), so its stored polar started at alpha 0 and, via the alpha_valid
intersection, poisoned every interior blend; seed 1 of the ladder gives an
equally-good cd 56.3-ct section that sweeps -4..12 cleanly
(scripts/gen_cst_library.py "Cell-acceptance gate"). Measured feasibility of
the 5-D box (1024 scrambled Sobol, seed 0) AFTER the fix: 1024/1024 (100%) —
every trim stays inside the members' converged range (min tip alpha_eff
-2.3 deg > the -4 floor, max root alpha_eff +5.9 deg < the +6.5 low-cl
stall). With the starved centre member (alpha_valid (0, 12)) it was 0/1024,
the (0, 12) window poisoning ALL interior blends.

B) FlatCoupledProblem — flat coupling (d = 17, NOT the default tool path)
--------------------------------------------------------------------------
    x = [taper, twist_root_deg, twist_tip_deg, b_m, S_m2,
         w_upper_0..5, w_lower_0..5]                    (n_cst = 6)

The dimension-scaling study arm: Tier B aircraft physics (aircraft.py —
REUSED, not reimplemented: weights.total_weight fixed point, payload
L/D f = W_fixed / D, log stress margin g = ln(sigma_allow / sigma_root)),
but the section polar comes from a per-candidate VISCOUS XFOIL sweep of
the candidate's own CST section (xfoil_run.run_xfoil_polar, cached).
The t/c that feeds the Raymer weight formula and the spar-stress model is
the REAL geometric thickness of that section, cst_thickness(w_u, w_l) —
the aero-structural coupling is closed through the actual shape, not a
free parameter. Every evaluation costs a real XFOIL sweep (~1-2 s;
alphas -4..12 deg step 1.0, deliberately coarser than the 2-D section
problem's 0.5-deg grid to hold the per-eval cost down — the wing-level
trim only needs the polar's linear region and a cd(alpha) table).

CST design box (recorded derivation, 2026-07-13, AirfoilProblem style)
----------------------------------------------------------------------
Anchor = cst_anchor("2412", n_cst=6) (NACA 2412 least-squares refit,
t/c = 0.1201):
    w_u ~ [0.194, 0.198, 0.233, 0.173, 0.221, 0.207]
    w_l ~ [-0.152, -0.107, -0.093, -0.095, -0.073, -0.080]
Same asymmetric half-widths and floors as the 8-D section box (airfoil.py:
0.35 in the thickness-REDUCING directions, 0.15 in the thickness-
INCREASING ones, upper floor +0.05 / lower cap -0.02 keep each surface on
its own side of the chord line, which PROVES in-box non-self-intersection).
Pre-registered validity check (>= 80% bar, same criterion as the 8-D
derivation: non-self-intersecting AND t/c in (0.06, 0.20)): 200 scrambled
Sobol points -> 200/200 valid (100%), sampled t/c in [0.087, 0.184],
median 0.134.

Failure contract (objective.py / hydrofoil.py / airfoil.py)
-----------------------------------------------------------
Solver/tool failure -> exactly (PENALTY, G_FAIL), finite, never NaN or an
exception to the optimiser. XFOIL starvation (< 3 converged points, or a
pre-stall monotone branch too short to define the linear region) is a
failed DESIGN -> PENALTY; XfoilError (binary missing/unexecutable) is an
INFRASTRUCTURE fault and propagates (airfoil.py convention — silently
scoring every design -100 on a broken install would fabricate a study).
Solvable-but-overstressed flat designs return their TRUE f with signed
g < 0 (constrained BO must see the violation magnitude).

References
----------
Sobieszczanski-Sobieski, J., and Haftka, R. T., "Multidisciplinary
    aerospace design optimization: survey of recent developments,"
    Structural Optimization, Vol. 14, 1997, pp. 1-23. doi:10.1007/BF01197554
Kulfan, B. M., "Universal Parametric Geometry Representation Method,"
    Journal of Aircraft, Vol. 45, No. 1, 2008, pp. 142-158 (CST).
Raymer, D. P., "Aircraft Design: A Conceptual Approach," 5th ed., AIAA,
    2012, Ch. 15 (statistical wing weight — via weights.py).
Drela, M., "XFOIL: An Analysis and Design System for Low Reynolds Number
    Airfoils," Conf. on Low Reynolds Number Airfoil Aerodynamics, 1989
    (via xfoil_run.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import geometry, weights
from .sizing import PAYLOAD_LOD_UNITS
from .airfoil import (
    W_HALF_THICK,
    W_HALF_THIN,
    W_LOWER_CAP,
    W_UPPER_FLOOR,
    _longest_increasing_run,
    _surface_y,
    cst_anchor,
    cst_coords,
    cst_thickness,
)
from . import materials
from .aircraft import SWEEP_FIXED_DEG, AircraftProblem
from .mission import MissionSpec
from .objective import MU_SL, PENALTY, RHO_SL, Problem, _fail, evaluate_geometry
from .polar import TablePolar
from .weights import N_ULT_DEFAULT, SIGMA_ALLOW_PA

G_FAIL = -1.0        # finite infeasible margin reported on solver failure

# Library hull (the 3 x 3 grid of scripts/gen_cst_library.py) — the summary
# variables may not command sections the library cannot deliver. The numbers
# live in geometry.py because the WINGLET+coupled families build their design
# box there (geometry.bounds) and geometry may not import this module; these
# names stay because they are the published ones.
TC_SEC_BOUNDS = geometry.TC_SEC_BOUNDS
CL_SEC_BOUNDS = geometry.CL_SEC_BOUNDS

# Comparability anchor: tier_a_plus 5-D BO result at the same operating
# point (PROGRESS.md item 2; results context in experiments/coupled_bo.py).
TIER_A_PLUS_REF_LOD = 37.57


# =========================================================================
# A) Summary-variable coupled problem (d = 5) — the default tool path
# =========================================================================


@dataclass
class WingAirfoilProblem:
    """Summary-variable coupled wing + airfoil problem (module docstring A).

    Design vector (d = 5):

        x = [taper, twist_root_deg, twist_tip_deg, tc_sec, cl_sec]

    Objective (MAXIMISE): trim L/D at the Tier A operating point — b = S =
    10 m, V = 14.6 m/s, CL_target = 0.5 — via the IDENTICAL aero core as
    every other tier (objective.evaluate_geometry on a locally-built trim
    Problem, the tier_a_plus polar-override pattern), so scores compare
    1:1 with the tier_a_plus 5-D headline L/D 37.57.

    ``family`` is the (t/c, cl_design)-indexed CST section library
    (polar.PolarFamily2D), lazily loaded from data/airfoils/cst_library —
    pass a pre-built family to share it across evaluations (the
    experiment harness does; loading is one-off JSON I/O, evaluation is
    ~0.5 ms). FileNotFoundError at construction means the library has not
    been generated on this machine (scripts/gen_cst_library.py).
    """

    b: float = 10.0
    S: float = 10.0
    N: int = 60
    V: float = 14.6           # m/s -> Re_mac ~ 1e6 (library polars' Re)
    rho: float = RHO_SL
    mu: float = MU_SL
    CL_target: float = 0.5
    family: Any = None        # PolarFamily2D; lazily loaded
    chord_order: int = 0      # free chord law (geometry.py); 0 = taper only
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
    material: object = None   # WHAT THE WING IS BUILT OF
    #   (materials.Material, or a key from materials.MATERIALS).
    #   None = 2024-T3 aluminium, the material Raymer's correlation
    #   was regressed over, so every published run is bit-identical.
    #   It sets BOTH the allowable the spar is sized to and the
    #   factor the statistical wing weight is scaled by — the second
    #   is the one the span answer actually rides (materials.py).
    size_free: bool = False    # SIZE modifier (sizing.py): span and area
    #   free, weight-coupled, payload L/D with a root-bending stress margin
    #   as the problem's first constraint.
    span_bounds_m: tuple | None = None     # (min, max) SPAN row [m] of the
    #   size modifier; None -> the fractional band around this problem's own
    #   span (sizing.span_bounds)
    area_bounds_m2: tuple | None = None    # (min, max) AREA row [m^2]; None
    #   -> the fractional band around its own reference area. Stating it is
    #   what lets a search reach a mission of a different weight at all
    wing_loading_max_Pa: float | None = None   # the MISSION's ceiling on W/S
    #   (constraint_diagram): any candidate above it is refused, in-contract
    W_fixed_N: float | None = None    # non-wing weight for the size modifier
    flight_free: bool = False  # speed + altitude as design variables
    #   (geometry.py's flight modifier). Note what it does NOT move: the
    #   library polars are a single-Re set, so a candidate that flies faster
    #   gets the same section data at a different Re — the fixed-Re
    #   simplification this tier already carries, now visible in the vector.
    mission: Any = None        # mission.MissionSpec (design weight); defaults
    #   to the weight this problem's own CL_target implies at its own speed

    def __post_init__(self):
        if self.family is None:
            from .polar import load_cst_polar_family
            self.family = load_cst_polar_family()
        if self.flight_free and self.mission is None:
            from .mission import MissionSpec, isa_density
            q0 = 0.5 * isa_density(0.0) * self.V**2
            self.mission = MissionSpec(W_N=self.CL_target * (q0 * self.S),
                                       V=self.V, altitude_m=0.0)

    @property
    def bounds(self) -> np.ndarray:
        from .sizing import with_size_bounds
        box = with_size_bounds(np.array([
            geometry.TAPER_BOUNDS,
            geometry.TWIST_ROOT_BOUNDS_DEG,
            geometry.TWIST_TIP_BOUNDS_DEG,
            TC_SEC_BOUNDS,
            CL_SEC_BOUNDS,
        ], dtype=float), self.size_free, self.b, self.S,
            span_bounds_m=self.span_bounds_m,
            area_bounds_m2=self.area_bounds_m2,
            wing_loading_max_Pa=self.wing_loading_max_Pa)
        box = geometry.with_flight_bounds(box, self.flight_free)
        return geometry.with_chord_bounds(box, self.chord_order,
                                          self.chord_max_frac,
                                          self.chord_law)

    @property
    def dim(self) -> int:
        return int(self.bounds.shape[0])


def evaluate_wing_airfoil(x: np.ndarray,
                          prob: WingAirfoilProblem | None = None) -> dict:
    """Full summary-variable evaluation with breakdown; never raises for
    in-contract failures.

    Pipeline: bounds check -> family.at(tc_sec, cl_sec) (exact member at
    grid nodes, bilinear blend inside — polar.PolarFamily2D) -> locally-
    built trim Problem -> objective.evaluate_geometry (the tier_a_plus
    polar-override pattern). The alpha_valid check inside
    evaluate_geometry is the stall/extrapolation failure proxy; blends
    inherit the INTERSECTION of their corners' converged ranges (module
    docstring honesty label).
    """
    prob = prob or WingAirfoilProblem()
    x = np.asarray(x, dtype=float)
    bnds = prob.bounds
    if x.shape != (bnds.shape[0],):
        return _fail(f"design vector shape {x.shape} != ({bnds.shape[0]},)")
    if np.any(x < bnds[:, 0] - 1e-12) or np.any(x > bnds[:, 1] + 1e-12):
        return _fail("bounds violation")

    taper, tw_root, tw_tip, tc_sec, cl_sec = (float(v) for v in x[:5])

    # ---- size (sizing.py's modifier)
    size_req = bool(prob.size_free)
    W_fixed_ref = None
    if size_req:
        from dataclasses import replace as _replace0

        from .mission import weight_for as _weight_for0
        from .sizing import check_ar, size_from_x
        # the payload weight is the SAME CONSTANT for every candidate — that
        # is what makes payload L/D = W_fixed/D a minimum-drag objective
        # (sizing.py). Read from the problem's OWN published area, BEFORE the
        # candidate size replaces it: taken afterwards it scaled with the
        # area, so a bigger wing carried a bigger "fixed" payload and the
        # score rewarded growing the wing.
        W_fixed_ref = (prob.W_fixed_N if prob.W_fixed_N is not None
                       else _weight_for0(prob.CL_target, prob.V, prob.S))
        b_use, S_use = size_from_x(x, True, prob.flight_free,
                                   prob.chord_order)
        reason = check_ar(b_use, S_use)
        if reason is not None:
            return _fail(f"size: {reason}")
        prob = _replace0(prob, b=b_use, S=S_use, size_free=False)

    # ...and the ASPECT-RATIO LIMIT THE USER SET. A different question from
    # the validity band above, so it is asked whether or not the size is a
    # design variable: a limit that only applied to searched sizes would go
    # quiet exactly when the user typed the number it is about.
    if prob.ar_limits is not None:
        from .sizing import check_ar_limit as _check_ar_limit
        why_ar = _check_ar_limit(float(prob.b), float(prob.S), prob.ar_limits)
        if why_ar is not None:
            return _fail(f"size: {why_ar}")

    # ---- flight state (geometry.py's flight modifier)
    flight = geometry.flight_from_x(x, prob.flight_free, prob.chord_order)
    if flight is not None:
        from dataclasses import replace as _replace

        from .mission import flight_state
        try:
            prob = _replace(prob, flight_free=False, mission=None,
                            **flight_state(prob.mission, flight[0], flight[1],
                                           prob.S))
        except ValueError as exc:  # water medium, or ISA validity
            return _fail(f"flight state: {exc}")

    try:
        pol = prob.family.at(tc_sec, cl_sec)
    except ValueError as exc:
        return _fail(f"polar family: {exc}")

    try:
        wing = geometry.Wing(
            b=prob.b, S=prob.S, taper=taper, twist_root_deg=tw_root,
            twist_tip_deg=tw_tip, tc=tc_sec,
            chord_limits=prob.chord_limits,
            chord_coeffs=geometry.chord_coeffs_from_x(x, prob.chord_order,
                                                      prob.chord_law))
    except ValueError as exc:
        return _fail(f"planform: {exc}")
    _, c, twist_rad = wing.sample(prob.N)

    sized = None
    if size_req:
        from dataclasses import replace as _replace1

        from .sizing import sized_state
        try:
            sized = sized_state(
                W_fixed_N=W_fixed_ref,
                b=prob.b, S=prob.S, taper=wing.taper, tc=wing.tc,
                q_Pa=0.5 * prob.rho * prob.V**2,
                c_root=float(wing.chord(np.array([0.0]))[0]),
                wing_loading_max_Pa=prob.wing_loading_max_Pa,
                material=prob.material)
        except (ValueError, RuntimeError, OverflowError) as exc:
            return _fail(f"size: {exc}")
        prob = _replace1(prob, CL_target=sized.CL_target)
        # ...and again after the copy: ``replace`` re-runs
        # ``objective.Problem.__post_init__``, which re-derives
        # CL_target from the mission and would put the PAYLOAD
        # weight back in the force balance while leaving the
        # wing's own weight out of it. See objective.py's note.
        prob.CL_target = float(sized.CL_target)

    # Local trim Problem = the Tier A aero core with the library polar
    # override (tier_a_plus pattern; aircraft.py evaluate_aircraft analogue).
    aero_prob = Problem(
        b=prob.b, S=prob.S, N=prob.N, V=prob.V, rho=prob.rho, mu=prob.mu,
        CL_target=prob.CL_target, mode="trim", polar=pol,
    )
    out = evaluate_geometry(c, twist_rad, aero_prob, alpha_deg=None,
                            mac=wing.mac, sweep_deg=0.0, polar=pol)
    if out["feasible"]:
        out.update({
            "wing": wing,
            "sweep_deg": 0.0,
            "tc_sec": tc_sec,
            "cl_sec": cl_sec,
            "tc": tc_sec,
        })
        if flight is not None:
            out.update({"V": float(prob.V), "altitude_m": float(flight[1]),
                        "rho": float(prob.rho), "mu": float(prob.mu)})
        if sized is not None:
            q = 0.5 * prob.rho * prob.V**2
            f = sized.payload_lod(q, out["CD"])
            if not np.isfinite(f):
                return _fail("non-finite payload L/D")
            out.update(sized.report())
            out.update({"f": float(f), "score": float(f),
                        "g": float(sized.g_sigma),
                        "D_N": float(q * sized.S * out["CD"])})
    return out


#: finite infeasible margin reported on solver failure by the SIZED variant
G_FAIL = -1.0


def fg_wing_airfoil(x: np.ndarray, prob: WingAirfoilProblem | None = None
                    ) -> tuple[float, float]:
    """Constrained-harness callable for the SIZE modifier: (payload L/D,
    signed root-bending stress margin)."""
    out = evaluate_wing_airfoil(x, prob)
    if not out["feasible"]:
        return PENALTY, G_FAIL
    if "g" not in out:
        raise ValueError("fg_wing_airfoil() is for the size-modifier variant")
    return float(out["score"]), float(out["g"])


def objective_wing_airfoil(x: np.ndarray,
                           prob: WingAirfoilProblem | None = None) -> float:
    """Scalar objective (MAXIMISE) for the unconstrained runners
    (optimize/baselines.py REGISTRY / optimize/bo.py): L/D, or exactly
    PENALTY on any in-contract failure."""
    out = evaluate_wing_airfoil(x, prob)
    return float(out["score"]) if out["feasible"] else PENALTY


# =========================================================================
# B) Flat coupled problem (d = 17) — dimension-scaling study arm
# =========================================================================


@dataclass
class FlatCoupledProblem:
    """Flat wing + airfoil coupling: Tier B aircraft physics with the
    candidate's own CST section evaluated by XFOIL (module docstring B).

    Design vector (d = 5 + 2 n_cst = 17):

        x = [taper, twist_root_deg, twist_tip_deg, b_m, S_m2,
             w_upper_0..5, w_lower_0..5]

    Objective (MAXIMISE): f = W_fixed / D_total (payload L/D — aircraft.py
    derivation: monotone in total drag = thrust = power at fixed V).
    Constraint: g = ln(sigma_allow / sigma_root) >= 0 (aircraft.py's
    measured GP-scale choice), with sigma_root fed by the REAL section
    thickness cst_thickness(w_u, w_l).

    Wing-level box: the aircraft.py planform box (span/area bounds are
    AircraftProblem's own class constants — same generous-span story,
    same active structural constraint closing it). Section box: the
    n_cst = 6 NACA 2412 anchor refit with the recorded asymmetric
    half-widths (module docstring; 200/200 Sobol validity).

    Cost model: one cached viscous XFOIL sweep per candidate (~1-2 s
    fresh, ~ms cached) — 17 alphas, -4..12 deg step 1.0. This is the
    "expensive black box" arm of the decomposition experiment; DO NOT put
    it inside another loop without checking the budget arithmetic.
    """

    mission: MissionSpec = field(
        default_factory=lambda: MissionSpec(W_N=12000.0, V=50.0,
                                            altitude_m=0.0)
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
    re: float = 1e6
    mach: float = 0.0
    n_cst: int = 6                # weights PER SURFACE; d = 5 + 2 n_cst
    dz_te: float = 0.0            # sharp TE, fixed (airfoil.py convention)
    alphas: np.ndarray = field(
        default_factory=lambda: np.arange(-4.0, 12.5, 1.0))
    cache_dir: Any = None         # None -> xfoil_run default cache
    timeout_s: float = 20.0
    n_panel: int = 200

    B_BOUNDS = AircraftProblem.B_BOUNDS      # reuse the Tier B planform box
    S_BOUNDS = AircraftProblem.S_BOUNDS      # (calibration story there)

    def __post_init__(self):
        if self.mission.medium != "air":
            raise ValueError("FlatCoupledProblem is air-only")
        w_u, w_l = cst_anchor("2412", self.n_cst)
        self._w0 = np.concatenate([w_u, w_l])
        lo = np.concatenate([
            np.maximum(w_u - W_HALF_THIN, W_UPPER_FLOOR),
            w_l - W_HALF_THICK,
        ])
        hi = np.concatenate([
            w_u + W_HALF_THICK,
            np.minimum(w_l + W_HALF_THIN, W_LOWER_CAP),
        ])
        self._cst_bounds = np.column_stack([lo, hi])

    @property
    def w0(self) -> np.ndarray:
        """NACA 2412 anchor weights (n_cst per surface) — the baseline
        section; prepend a planform to get an 'anchor-ish' design."""
        return self._w0.copy()

    @property
    def bounds(self) -> np.ndarray:
        return np.vstack([
            np.array([
                geometry.TAPER_BOUNDS,
                geometry.TWIST_ROOT_BOUNDS_DEG,
                geometry.TWIST_TIP_BOUNDS_DEG,
                self.B_BOUNDS,
                self.S_BOUNDS,
            ], dtype=float),
            self._cst_bounds,
        ])

    @property
    def dim(self) -> int:
        return 5 + 2 * self.n_cst


def evaluate_flat_coupled(x: np.ndarray,
                          prob: FlatCoupledProblem | None = None) -> dict:
    """Full flat-coupled evaluation with breakdown; never raises for
    in-contract failures. ``feasible`` refers to the SOLVER (PENALTY
    contract); the structural constraint is the signed ``g_sigma``.

    Pipeline: bounds check -> geometric precheck (positive thickness,
    airfoil.py) -> tc = cst_thickness -> weight fixed point
    (weights.total_weight) -> CL_target = W_total / (q S) -> per-candidate
    XFOIL polar of the candidate's own section (cached; the pre-stall
    monotone branch becomes an in-memory TablePolar, whose __post_init__
    extracts a_lin / alpha_L0 exactly as polar.py does for file-backed
    tables) -> objective.evaluate_geometry on a locally-built trim
    Problem -> root bending stress -> payload L/D.

    XFOIL starvation (< 3 converged points, or < 3 points on the
    pre-stall monotone branch — no bracket for the linear-region fit) is
    a failed DESIGN -> PENALTY dict; XfoilError propagates (infrastructure
    fault, airfoil.py convention).
    """
    from . import xfoil_run  # late import: tests monkeypatch run_xfoil_polar

    prob = prob or FlatCoupledProblem()
    x = np.asarray(x, dtype=float)
    bnds = prob.bounds
    if x.shape != (bnds.shape[0],):
        return _fail(f"design vector shape {x.shape} != ({bnds.shape[0]},)")
    if np.any(x < bnds[:, 0] - 1e-12) or np.any(x > bnds[:, 1] + 1e-12):
        return _fail("bounds violation")

    taper, tw_root, tw_tip, b, S = (float(v) for v in x[:5])
    w_u = x[5:5 + prob.n_cst]
    w_l = x[5 + prob.n_cst:]

    # Geometric precheck (airfoil.py): guaranteed in-box by the floor/cap,
    # guards out-of-box callers before XFOIL is asked to panel garbage.
    psi = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, 401)))[1:-1]
    gap = _surface_y(psi, w_u, +0.5 * prob.dz_te) - _surface_y(
        psi, w_l, -0.5 * prob.dz_te)
    if not np.all(gap > 0.0):
        return _fail("self-intersecting / non-positive-thickness geometry")

    tc = cst_thickness(w_u, w_l, dz_te=prob.dz_te)

    try:
        mat = materials.resolve(prob.material)
        W_total, W_wing = weights.total_weight(
            prob.mission.W_N, b, S, taper, SWEEP_FIXED_DEG, tc, prob.n_ult,
            q_Pa=prob.mission.q, k_w=mat.k_w,
        )
    except (ValueError, RuntimeError, OverflowError) as exc:
        return _fail(f"weight fixed point: {exc}")
    if not (np.isfinite(W_total) and W_total > 0.0):
        return _fail("non-finite/non-positive total weight")

    q = prob.mission.q
    CL_target = W_total / (q * S)

    # --- the candidate's OWN section polar: one cached XFOIL sweep -------
    coords = cst_coords(w_u, w_l, dz_te=prob.dz_te)
    pol_x = xfoil_run.run_xfoil_polar(
        coords, prob.re, prob.mach, prob.alphas,
        timeout_s=prob.timeout_s, cache_dir=prob.cache_dir,
        n_panel=prob.n_panel,
    )   # XfoilError (infrastructure fault) intentionally propagates
    if pol_x.n_converged < 3:
        return _fail(
            "XFOIL starvation (< 3 converged points) = failed design",
            n_converged=pol_x.n_converged)

    # Pre-stall monotone branch only (airfoil.py): past cl_max the lift
    # curve folds over and would corrupt both the linear-region extraction
    # and the wing-level cd(alpha) interpolation.
    sl = _longest_increasing_run(pol_x.cl)
    if (sl.stop - sl.start) < 3:
        return _fail(
            "XFOIL starvation (pre-stall monotone branch < 3 points — "
            "no bracket for the linear region) = failed design",
            n_converged=pol_x.n_converged)
    try:
        pol = TablePolar(
            alpha_deg=pol_x.alpha_deg[sl], CL=pol_x.cl[sl],
            CD=pol_x.cd[sl], CM=pol_x.cm[sl],
            name=f"CST candidate t/c={tc:.3f} Re{prob.re:.0e} (XFOIL)",
            Re=prob.re,
        )
    except (ValueError, np.linalg.LinAlgError, TypeError) as exc:
        return _fail(f"degenerate polar (linear-region extraction): {exc}")

    wing = geometry.Wing(b=b, S=S, taper=taper, twist_root_deg=tw_root,
                         twist_tip_deg=tw_tip, tc=tc)
    _, c, twist_rad = wing.sample(prob.N)

    aero_prob = Problem(
        b=b, S=S, N=prob.N, V=prob.mission.V, rho=prob.mission.rho,
        mu=prob.mission.mu, CL_target=CL_target, mode="trim", polar=pol,
    )
    out = evaluate_geometry(c, twist_rad, aero_prob, alpha_deg=None,
                            mac=wing.mac, sweep_deg=SWEEP_FIXED_DEG,
                            polar=pol)
    if not out["feasible"]:
        return out

    sigma = weights.root_bending_stress(b, S, taper, tc, W_total, prob.n_ult)
    sigma_allow = (mat.sigma_allow_Pa if prob.material is not None
                   else prob.sigma_allow)
    g_sigma = float(np.log(sigma_allow / sigma))
    g_sigma_lin = (sigma_allow - sigma) / sigma_allow

    D_N = q * S * out["CD"]
    f = prob.mission.W_N / D_N               # payload L/D (aircraft.py)
    if not (np.isfinite(f) and np.isfinite(g_sigma)):
        return _fail("non-finite payload L/D or stress margin")

    out.update({
        "score": float(f),
        "f": float(f),
        "score_units": PAYLOAD_LOD_UNITS,      # W_fixed/D, not the aero L/D
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
        "tc": float(tc),
        "wing": wing,
        "sweep_deg": SWEEP_FIXED_DEG,
        "n_converged": pol_x.n_converged,
        "n_branch": int(sl.stop - sl.start),
        "from_cache": pol_x.from_cache,
    })
    return out


def fg_flat(x: np.ndarray, prob: FlatCoupledProblem | None = None
            ) -> tuple[float, float]:
    """Constrained-harness callable: (payload L/D, log stress margin) —
    the aircraft.fg_aircraft contract.

    Solvable-but-overstressed designs return their TRUE f with g < 0;
    solver/tool failures (including XFOIL starvation) return exactly
    (PENALTY, G_FAIL)."""
    out = evaluate_flat_coupled(x, prob)
    if not out["feasible"]:
        return PENALTY, G_FAIL
    return float(out["f"]), float(out["g_sigma"])
