"""Design vector x -> L/D, with the -100.0 penalty-on-failure contract.

Modes (see README "Design vector"):
  trim (default): x = [taper, twist_root_deg, twist_tip_deg]; the root AoA is
      solved so CL = CL_target (brentq). Documented default — every candidate
      is compared at the same lift, so L/D differences are purely geometric.
  soft: x = [taper, twist_root_deg, twist_tip_deg, alpha_deg]; alpha is free
      and the score is L/D - soft_weight*(CL - CL_target)^2.
  tier_a_plus: x = [taper, twist_root_deg, twist_tip_deg, sweep_deg, tc];
      trim-mode scoring on a 5-D vector. sweep enters as reduced-order
      simple sweep theory (a = a_lin cos Λ, llt.swept_section_slope — the
      LLT stays geometrically unswept); tc selects the section polar from
      the NACA 24XX XFOIL PolarFamily (linear-in-t/c blend between members,
      exact at members). At sweep = 0, tc = 0.12 this reproduces the Tier A
      trim objective EXACTLY (regression-gated in tests/test_tier_a_plus.py).
  winglet (Tier A+): x = trim vector + [winglet_h_frac in [0, 0.15],
      winglet_cant_deg in [60, 90]]; the physics kernel switches from the LLT
      to the nonplanar VLM (vlm.py), trimmed to CL_target (closed form — the
      VLM is linear in alpha). CDi comes from the TREFFTZ plane; profile drag
      integrates the SAME section polar over the main-wing strips AND the
      winglet strips at the alpha-method effective angle (documented
      approximation: the winglet reuses the wing airfoil polar) — this is how
      the winglet's extra wetted area is charged. Trim-only (no soft variant).
  winglet_capped: same vector and physics as winglet, but the TOTAL
      PROJECTED SPAN is capped at b — the wing panel shrinks (constant S)
      to pay for the winglet's horizontal projection, so raking the tip is
      no longer a free span extension (see geometry.wing_from_x). At
      cant = 90 deg it reproduces "winglet" exactly (regression-gated).
  mission: x = [taper, twist_root_deg, twist_tip_deg, V_ms, altitude_m];
      requires Problem.mission (a mission.MissionSpec carrying the design
      weight W_N). Speed and altitude are bounded design variables: each
      candidate's flow state (rho, mu from ISA/Sutherland) and trim target
      CL_target = W_N / (q S) are rebuilt from ITS (V, h) — the same
      speed-to-loading coupling as HydrofoilProblem.CL_target(V). At the
      default-mission sea-level point (V = 14.6, h = 0) this reproduces
      the Tier A trim objective exactly (regression-gated in
      tests/test_mission.py). V/h affect only q, CL_target and the Re_mac
      diagnostic; the section polar remains the fixed Re = 1e6 table
      (Tier A simplification). Air-only: water missions belong to
      hydrofoil.HydrofoilProblem (guarded with _fail).

Failure contract: `objective()` returns exactly PENALTY (-100.0) whenever the
solver fails, trim cannot bracket, any effective AoA leaves the polar's valid
range (stall / extrapolation proxy), or the result is non-finite. BO sees a
finite, very-bad value instead of NaN. `evaluate()` returns the full
breakdown dict with `feasible` / `reason`.

Drag model: CD = CDi (LLT) + CDp (section polar integrated over span)
[+ cd0_extra for future non-wing components — see drag.py double-count guard].
Re note: the polar is a single-Re table (Re_mac ~ 1e6); spanwise chord-Re
variation is neglected (Tier A simplification, flagged here).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from .mission import isa_density, isa_temperature, sutherland_mu

from . import geometry
from .llt import pg_beta, solve_llt, solve_llt_trim, swept_section_slope
from .polar import default_polar, default_polar_family
from .vlm import VLM

# NOTE: ground_effect and tandem are imported LAZILY inside evaluate_geometry
# (only when prob.ground_h_m is set). tandem.py imports names from THIS module,
# so a top-level import here would be a circular import at package load.

PENALTY = -100.0

# SEA-LEVEL AIR HAS ONE AUTHOR, and it is the ISA model — these are read
# from it rather than written here. They used to be the literals 1.225 and
# 1.789e-5, which is where the ISA model lands to 4.4e-7 and 1.7e-4
# respectively, and that gap was a real defect hiding behind a zero: a family
# defaults to THESE numbers while its own MissionSpec derives rho/mu from ISA,
# so switching the flight modifier on at the family's own (V, 0) rebuilt the
# flow state slightly different. Nothing read the difference while the only
# Reynolds-based charge in the book (the fin's parasite drag) was switched
# off; charging it unconditionally made
# `test_flight_state.py::test_the_published_state_reproduces_the_base_problem_exactly`
# fail on four families at once. Derived here, the modifier is the identity
# at the published point again, in every term.
RHO_SL = isa_density(0.0)     # kg/m^3, ISA sea level
MU_SL = sutherland_mu(isa_temperature(0.0))      # Pa s


# ---------------------------------------------------------------- winglet types
#
# The nonplanar VLM (vlm.py) models a winglet as ONE straight canted tip ray:
# height fraction h/(b/2) and cant angle from the wing plane are the only two
# geometric freedoms it honours (chord = tip chord, incidence = tip twist,
# zero toe — documented simplifications). Winglet SWEEP would move the bound
# vortices in x only, and the Trefftz-plane induced drag depends on the (y, z)
# wake trace alone, so sweep is invisible to this model — it is deliberately
# NOT offered as a design variable (adding a knob with no effect would
# fabricate a trade). Given that, the honest "type" axis is the CANT BAND
# crossed with the span-accounting (free vs capped):
#
#   vertical (fence):  cant ~ 90 deg, near-zero horizontal projection — the
#                      pure nonplanar case; span-free is honest (nothing is
#                      raked outboard).
#   canted:            cant 60-90 deg — the classic winglet; the legacy band.
#   raked (wingtip):   cant 20-60 deg — mostly a planar span extension. Raking
#                      MUST be scored span-capped: with free span the optimiser
#                      just adds projected span for free and reports an inflated
#                      L/D (verified — free-span L/D rises monotonically as
#                      cant -> 0). The capped mode shrinks the wing to pay for
#                      the projection, which is the only honest raked-tip trade.
#
# ``force_capped`` records that constraint; the API refuses a free-span raked
# run. Blended winglets are intentionally absent: their defining feature is a
# curved root transition the single straight ray cannot represent, so offering
# one would be a fabricated geometry.
WINGLET_TYPES: dict[str, dict] = {
    "canted": {
        "cant_bounds": (60.0, 90.0), "force_capped": False,
        "label": "canted winglet",
        "note": "classic canted winglet (legacy 60-90 deg cant band)",
    },
    "vertical": {
        "cant_bounds": (84.0, 90.0), "force_capped": False,
        "label": "vertical fence",
        "note": "near-vertical tip fence: pure nonplanar, ~zero projection",
    },
    "raked": {
        "cant_bounds": (20.0, 60.0), "force_capped": True,
        "label": "raked wingtip",
        "note": "low-cant planar extension; span-capped so the rake pays for "
                "its own projection (a free-span rake reports fake L/D)",
    },
}


@dataclass
class Problem:
    """Tier A single-wing L/D problem definition.

    ``mission`` (mission.MissionSpec, optional): when set, the mission
    OVERRIDES V, rho, mu and CL_target in __post_init__ — the flow state
    comes from ISA/Sutherland (or the water constants) at the mission
    operating point and CL_target = W_N / (q S), so the design weight
    becomes the single physical source of the trim target. The default
    ``mission=None`` preserves the legacy hard-coded values bit-for-bit
    (regression-gated in tests/test_mission.py). In ``mode="mission"``
    the spec additionally supplies W_N for the per-candidate (V, h)
    states (see evaluate).

    Phase-5 physics flags (ADDITIVE — every default is OFF and reproduces the
    Tier A objective BIT-FOR-BIT; regression-gated in test_physics_flags.py).
    They compose (mach + ground + slipstream in any combination) and ``prob``
    is never mutated per-eval — the mission-mode ``replace`` copies carry them
    through unchanged:

    ``mach`` (float, default 0.0): free-stream Mach. Threaded into
        llt.swept_section_slope(a_lin, sweep_deg, mach=) as a SECTION-slope
        compressibility correction (Prandtl-Glauert on the normal Mach; see
        llt.pg_beta). ``mach == 0.0`` short-circuits to the exact legacy
        expression a0 cos Λ. The free-stream transonic ceiling PG_MACH_MAX
        (0.7) is enforced via pg_beta: M < 0 or M >= 0.7 raises ValueError,
        which the objective maps to PENALTY (never NaN to the optimiser).

    ``ground_h_m`` (float | None, default None): wing height above a rigid
        ground plane [m]. When set, the non-winglet solve routes through
        ground_effect.solve_wing_ige_trim / solve_wing_ige (the rigid-wall
        positive-image LLT) in place of solve_llt_trim / solve_llt — same
        interface, and the effective-AoA / profile-drag integration downstream
        is identical; only the induced drag (CDi_total, ground-reduced) and the
        upwash-shifted trim alpha change. Winglet modes use the nonplanar VLM
        (evaluate_winglet), where the image system is not implemented, so
        ground_h_m there is an in-contract failure (PENALTY, reason set).

    ``slipstream`` (slipstream.SlipstreamSpec | None, default None): prop-
        slipstream footprint. Its per-station modifiers(y, c) -> (q_ratio,
        dalpha) enter the strip solve in TWO physically distinct ways
        (compose order documented in evaluate_geometry): (1) the swirl angle
        dalpha(y) [rad] is ADDED to the geometric twist array BEFORE the solve
        (rigorous, twist-like — it shifts the trim alpha and the loading);
        (2) the dynamic-pressure ratio q_ratio(y) = mu(y)^2 multiplies the
        PROFILE-DRAG integrand only, CDp = trapz(cd c q_ratio)/S. The lift-side
        spanwise-q variation is DELIBERATELY NEGLECTED (the LLT circulation
        solve stays at q_inf) — a documented formulation limit whose error is
        bounded by the cruise-dormancy finding (mu ~ 1.002 => q_ratio ~ 1.004,
        < 0.5%). The OFF spec (mu_inf = 1, dalpha_ref_deg = 0) returns
        (ones, zeros) bit-for-bit, so slipstream=OFF is exactly the legacy
        result.
    """

    b: float = 10.0
    S: float = 10.0
    N: int = 60
    V: float = 14.6           # m/s -> Re_mac ~ 1e6 for mac = 1 m at sea level
    rho: float = RHO_SL
    mu: float = MU_SL
    CL_target: float = 0.5
    mode: str = "trim"   # trim | soft | tier_a_plus | winglet | winglet_capped
    #                      | winglet_tc | winglet_capped_tc | mission
    #                      | winglet_coupled | winglet_capped_coupled
    soft_weight: float = 20.0
    cd0_extra: float = 0.0    # non-wing parasite hook (fuselage etc.)
    polar: Any = field(default_factory=default_polar)
    polar_family: Any = None  # section family; lazily loaded. t/c-indexed
    #                          (polar.PolarFamily, the NACA 24XX bank) in the
    #                          TC_MODES; (t/c x cl)-indexed
    #                          (polar.PolarFamily2D, the pre-optimised CST
    #                          library) in the COUPLED_MODES. One field, two
    #                          arities — the mode says which, and the two
    #                          mode lists are disjoint by test.
    N_vlm: int = 40           # winglet mode: main-wing VLM panels (full span)
    n_winglet: int = 8        # winglet mode: panels per winglet
    mission: Any = None       # mission.MissionSpec; overrides V/rho/mu/CL_target
    mach: float = 0.0         # Phase 5: free-stream Mach (PG section correction)
    ground_h_m: float | None = None   # Phase 5: height AGL [m]; None = free air
    slipstream: Any = None    # Phase 5: slipstream.SlipstreamSpec; None = OFF
    winglet_cant_bounds: tuple | None = None   # winglet-TYPE cant band; None =
    #                          legacy (60, 90). Only read in the winglet modes;
    #                          None keeps every non-winglet run bit-for-bit.
    winglet_h_bounds: tuple | None = None      # tip-device HEIGHT band,
    #                          as a fraction of the semi-span. None = the
    #                          published (0, 0.15), which is where this
    #                          model was MEASURED and not a cap: the
    #                          solver is smooth and feasible past it, with
    #                          an interior optimum near 0.45. A stated
    #                          band is flown, never refused.
    chord_order: int = 0       # polynomial chord-law order (geometry.py).
    #                          0 = straight taper, no extra design variables,
    #                          bit-for-bit legacy. > 0 APPENDS that many
    #                          coefficients to the design vector, in every
    #                          mode (LLT and VLM alike — the VLM samples
    #                          wing.chord(y) like the lifting line does).
    chord_max_frac: float = geometry.CHORD_COEFF_BOUND   # coefficient box
    chord_law: str = geometry.DEFAULT_CHORD_LAW   # WHICH SHAPE those
    #   coefficients describe (geometry.CHORD_LAWS): the published polynomial,
    #   the interior-only law that holds the root and tip chords exactly, a
    #   cranked planform, or the elliptic blend. It changes what the same
    #   design vector draws, never how long it is — the order is still the
    #   number of trailing entries (geometry.chord_param_count).
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
    winglet_chord_follows: bool = False   # the tip device's chord CONTINUES
    #   the wing's own chord law past the tip instead of holding the tip
    #   chord (vlm.py). False = the rectangular device every published
    #   winglet run flew, bit-for-bit. A VALUE choice, never a design
    #   variable: it changes the planform a given (h_frac, cant) draws, not
    #   the dimension of the vector — and it is a statement about the DESIGN,
    #   so it reaches every designed surface's tip device, exactly as one
    #   ChordLimits reaches every wing of a problem.
    material: object = None   # WHAT THE WING IS BUILT OF
    #   (materials.Material, or a key from materials.MATERIALS).
    #   None = 2024-T3 aluminium, the material Raymer's correlation
    #   was regressed over, so every published run is bit-identical.
    #   It sets BOTH the allowable the spar is sized to and the
    #   factor the statistical wing weight is scaled by — the second
    #   is the one the span answer actually rides (materials.py).
    size_free: bool | str = False   # SIZE MODIFIER (sizing.py): the wing's
    #                          size becomes part of the design, the wing
    #                          WEIGHS what that size implies (Raymer, closed
    #                          as a fixed point), the trim target follows
    #                          that weight, the score becomes payload
    #                          L/D = W_fixed/D and a root-bending stress
    #                          margin joins as a CONSTRAINT. Freeing the span
    #                          without those three would just slam the box
    #                          (aircraft.py).
    #                          TWO modes (sizing.size_mode):
    #                            True                — span AND area are
    #                              design variables (2 rows);
    #                            "wing_loading"      — the AREA follows the
    #                              chosen W/S (``wing_loading_Pa``) through
    #                              the same weight loop, so the vector
    #                              carries the SPAN alone (1 row) inside a
    #                              band the user states. W/S is the mission's
    #                              answer (constraint_diagram.py) and the
    #                              span is the aerodynamicist's; this mode is
    #                              what keeps the optimiser from re-opening
    #                              the mission's question.
    wing_loading_Pa: float | None = None   # W/S for the wing-loading mode.
    #                          None -> the loading this problem already
    #                          flies, CL_target * q, so the mode opens on the
    #                          family's own operating point and the trim CL
    #                          is unchanged by construction.
    span_bounds_m: tuple | None = None     # (min, max) span for that mode's
    #                          single row. None -> the same fractional band
    #                          the two-variable mode uses. Read by the FREE
    #                          mode too: a span band is a hangar or a trailer
    #                          whichever mode is asking.
    area_bounds_m2: tuple | None = None    # (min, max) AREA for the free
    #                          mode's second row. None -> the fractional band
    #                          around this problem's own reference area,
    #                          which is what every published run flew — and
    #                          which is blind to the mission, so a heavier
    #                          aircraft could not reach its own design point
    #                          until this could be stated (sizing.py).
    ws_bounds_pa: tuple | None = None      # (min, max) WING LOADING [Pa] for
    #                          the searched-loading mode's row. None ->
    #                          sizing.WS_FRAC_BOUNDS around the loading this
    #                          problem already flies.
    wing_loading_max_Pa: float | None = None   # the MISSION's own ceiling on
    #                          W/S — the binding ws_max line of its
    #                          constraint diagram (stall / landing field in
    #                          air, fly-up / cavitation in water). It CLIPS
    #                          the searched band and refuses any candidate
    #                          above it in every sized mode (in-contract
    #                          failure, like the aspect-ratio band), so a
    #                          search can no longer buy payload L/D with a
    #                          loading the aircraft cannot land at. None =
    #                          the mission states no ceiling, every published
    #                          run.
    W_fixed_N: float | None = None   # non-wing weight for the size modifier;
    #                          None -> the weight this problem's own
    #                          CL_target implies (sizing.py documents why a
    #                          sized problem is then heavier than its parent)
    flight_free: bool = False  # flight-state MODIFIER (geometry.py): speed
    #                          and altitude become design variables in ANY
    #                          mode — the LLT trim modes and the nonplanar
    #                          VLM winglet modes alike — because they change
    #                          only the flow state (rho, mu) and the trim
    #                          target CL = W/(q S), never the geometry. The
    #                          rows sit between the mode block and the chord
    #                          block; ``mode="mission"`` IS this modifier on
    #                          the plain trim wing and keeps its own vector,
    #                          so the two are refused together (geometry.py).
    #                          Needs a design WEIGHT: ``mission`` supplies it,
    #                          and defaults to mission.default_mission(S), the
    #                          weight the legacy CL_target = 0.5 already
    #                          implies, so the modifier's own (V, h) =
    #                          (14.6, 0) point reproduces the base problem.
    blend_shape: str = "arc"   # how a blended winglet turns out of the wing
    #                          plane (geometry.BLEND_SHAPES): "arc" is the
    #                          constant-radius fillet (legacy, bit-for-bit),
    #                          "smooth" the curvature-continuous smoothstep,
    #                          "spiral" the clothoid pair (also G2, and the
    #                          gentler of the two: peak curvature 4/3 of the
    #                          mean against the smoothstep's 3/2).
    #                          A VALUE choice, never a design variable: it
    #                          changes the shape a given blend_frac draws,
    #                          not the dimension of the vector.
    blend_frac_fixed: float = 0.0  # the winglet's OWN blend, as a VALUE:
    #                          the fraction of the device's arc length spent
    #                          turning out of the wing plane (geometry.
    #                          winglet_path), applied in the plain winglet
    #                          modes — whose vector is [taper, twist_root,
    #                          twist_tip, h_frac, cant] and stays that way.
    #                          0.0 = the sharp corner every published winglet
    #                          run flies, bit-for-bit.
    #                          Why a value at all, when three modes already
    #                          DESIGN this number: a blended tip is a SHAPE
    #                          decision ("build it blended"), and asking the
    #                          optimiser to choose the blend as well costs a
    #                          dimension on every winglet problem that wants
    #                          one. Refused in the BLENDED_MODES, which read
    #                          the same quantity out of x[5]: one number, one
    #                          owner.
    wing_blend_frac: float = 0.0   # fraction of the SEMI-SPAN over which the
    #                          transition starts INBOARD of the tip, i.e. how
    #                          much of the turn the WING does (geometry.py:
    #                          span_path). 0 = the published geometry, the
    #                          whole turn on the winglet. Also a VALUE choice
    #                          in every mode but "winglet_capped_blended_wing",
    #                          which reads it out of the design vector (x[6])
    #                          and ignores this field.
    #                          It exists because a turn confined to the device
    #                          has at most blend_frac * h of arc: at the blend
    #                          fraction a span-capped optimiser settles on the
    #                          radius is ~0.13 tip chords, which is a corner
    #                          whatever turn law draws it. A real blended
    #                          winglet starts turning on the wing.
    junction_drag: bool = False   # charge the wing/winglet corner's
    #                          interference drag (junction.py — a reduced-
    #                          order ADD-ON, and the only thing that makes a
    #                          smooth blend worth anything to a Trefftz-plane
    #                          method). OFF keeps every published winglet
    #                          result bit-for-bit; the blended-winglet
    #                          problems turn it on, because without it
    #                          "blend" is just a differently drawn wake.

    def __post_init__(self):
        if self.blend_shape not in geometry.BLEND_SHAPES:
            raise ValueError(
                f"unknown blend_shape {self.blend_shape!r}; "
                f"choose from {list(geometry.BLEND_SHAPES)}")
        lo, hi = geometry.WING_BLEND_FRAC_BOUNDS
        if not (lo <= float(self.wing_blend_frac) <= hi):
            raise ValueError(
                f"wing_blend_frac {self.wing_blend_frac} outside the design "
                f"band {geometry.WING_BLEND_FRAC_BOUNDS}")
        blo, bhi = geometry.WINGLET_BLEND_BOUNDS
        if not (blo <= float(self.blend_frac_fixed) <= bhi):
            raise ValueError(
                f"blend_frac_fixed {self.blend_frac_fixed} outside the "
                f"design band {geometry.WINGLET_BLEND_BOUNDS}")
        if float(self.blend_frac_fixed) > 0.0:
            if self.mode in BLENDED_MODES:
                raise ValueError(
                    f"mode {self.mode!r} DESIGNS the winglet's blend "
                    f"fraction (x[5]); passing blend_frac_fixed as well "
                    f"would give one number two owners — use a plain "
                    f"winglet mode for a fixed blend, or leave it at 0")
            if self.mode not in WINGLET_MODES:
                raise ValueError(
                    f"blend_frac_fixed needs a winglet to blend into the "
                    f"wing; mode {self.mode!r} carries none")
        if self.flight_free and self.mission is None:
            # the weight the legacy CL_target = 0.5 already implies, so the
            # box point (14.6 m/s, 0 m) reproduces the base problem exactly
            # (lazy import: mission -> hydrofoil -> tandem -> this module)
            from .mission import default_mission
            self.mission = default_mission(S=self.S)
        if self.mission is not None:
            self.V = self.mission.V
            self.rho = self.mission.rho
            self.mu = self.mission.mu
            self.CL_target = self.mission.cl_target(self.S)
        if self.mode in TC_MODES and self.polar_family is None:
            self.polar_family = default_polar_family()
        if self.mode in COUPLED_MODES and self.polar_family is None:
            # the (t/c x cl) CST library, not the NACA 24XX bank: a different
            # family object with a different arity, which is exactly why the
            # coupled modes are not in TC_MODES. FileNotFoundError here means
            # the library has not been generated on this machine
            # (scripts/gen_cst_library.py) — the same contract
            # wing_airfoil.WingAirfoilProblem already has.
            from .polar import load_cst_polar_family
            self.polar_family = load_cst_polar_family()
        if self.polar_family is not None and self.mode in (TC_MODES
                                                           + COUPLED_MODES):
            # ...and the ARITY has to match the mode, checked HERE.
            #
            # One field carries two kinds of family, and the mode says which.
            # A caller who hands the wrong one gets a bare TypeError out of
            # the polar lookup, hundreds of lines into evaluate() — which
            # breaks this module's failure contract (an in-contract failure
            # is a PENALTY dict, never an exception at the optimiser) and
            # says nothing about what was actually wrong. Refuse at
            # construction, where the two objects are still distinguishable.
            from .polar import PolarFamily2D
            want2d = self.mode in COUPLED_MODES
            got = self.polar_family
            if want2d and not isinstance(got, PolarFamily2D):
                raise ValueError(
                    f"mode {self.mode!r} indexes the pre-optimised section "
                    f"library by (t/c, cl) and needs a PolarFamily2D "
                    f"(polar.load_cst_polar_family()); got "
                    f"{type(got).__name__}")
            if not want2d and isinstance(got, PolarFamily2D):
                raise ValueError(
                    f"mode {self.mode!r} selects its section by THICKNESS "
                    f"alone and needs a 1-D PolarFamily "
                    f"(polar.default_polar_family()); got a PolarFamily2D — "
                    f"use a coupled mode ({', '.join(COUPLED_MODES)}) to "
                    f"search the library's two summary variables")

    @property
    def bounds(self) -> np.ndarray:
        rows = None
        if self.size_free:
            from .sizing import size_bounds, size_mode, span_bounds, \
                ws_bounds, SIZE_MODE_WS, SIZE_MODE_WS_FREE
            mode = size_mode(self.size_free)
            if mode == SIZE_MODE_WS:
                rows = span_bounds(self.b, self.span_bounds_m)
            elif mode == SIZE_MODE_WS_FREE:
                # the band opens around the loading this problem ALREADY
                # flies (CL_target x q), for the same reason the span band
                # opens around its own span: a default that is not the
                # current design is a default that has chosen something
                rows = np.vstack([
                    span_bounds(self.b, self.span_bounds_m),
                    ws_bounds(self.CL_target * 0.5 * self.rho * self.V ** 2,
                              self.ws_bounds_pa, self.wing_loading_max_Pa)])
            else:
                rows = size_bounds(self.b, self.S, 1, self.span_bounds_m,
                                   self.area_bounds_m2)
        return geometry.bounds(self.mode, cant_bounds=self.winglet_cant_bounds,
                               h_bounds=self.winglet_h_bounds,
                               chord_order=self.chord_order,
                               chord_max_frac=self.chord_max_frac,
                               chord_law=self.chord_law,
                               flight_free=self.flight_free, size_rows=rows)

    @property
    def dim(self) -> int:
        return self.bounds.shape[0]


def _fail(reason: str, **extra) -> dict:
    d = {"feasible": False, "reason": reason, "LoD": None, "score": PENALTY}
    d.update(extra)
    return d


def design_wing_loading_pa(prob) -> float | None:
    """The W/S a DERIVED-area family flies, stated once.

    ``wing_loading_Pa`` when the mission set one, else the loading implied by
    the design point itself (CL_target * q). Extracted because
    :func:`api._gate_grader` needs exactly this number to bound the area — and
    therefore the aspect ratio — of a family with no ``S_m2`` row, and two
    spellings of it would let the cheap gate and the solver disagree about
    which design the run is flying.

    ``None`` only when the problem states neither, which is the one case where
    nothing can be bounded.
    """
    ws = getattr(prob, "wing_loading_Pa", None)
    if ws is not None:
        return float(ws)
    try:
        return float(prob.CL_target * 0.5 * prob.rho * prob.V ** 2)
    except (AttributeError, TypeError, ValueError):
        return None


def evaluate_geometry(
    c: np.ndarray,
    twist_rad: np.ndarray,
    prob: Problem,
    alpha_deg: float | None = None,
    mac: float | None = None,
    sweep_deg: float = 0.0,
    polar: Any = None,
) -> dict:
    """Physics core: sampled chord + twist arrays -> L/D breakdown.

    ``alpha_deg=None`` trims to prob.CL_target; otherwise alpha is fixed
    (soft mode / studies). Shared by the Tier A design vector (evaluate)
    and the crossover experiments (per-station twist knots, any dimension).

    Tier A+ hooks (defaults preserve Tier A behaviour exactly):
    ``sweep_deg`` applies simple sweep theory a = a_lin cos(Λ) to the section
    slope (llt.swept_section_slope — the LLT geometry stays unswept);
    ``polar`` overrides prob.polar (used for the t/c-selected family member).

    Phase-5 flags on ``prob`` (all OFF by default -> bit-for-bit legacy):
    ``prob.mach`` feeds the compressibility term of swept_section_slope (guarded
    by pg_beta -> PENALTY at M >= 0.7); ``prob.ground_h_m`` swaps the free-air
    solve for the rigid-ground image LLT (ground_effect); ``prob.slipstream``
    composes as: (1) dalpha(y) ADDED to the twist BEFORE the solve, then
    (2) q_ratio(y) scaling the profile-drag integrand. Compose order is
    slipstream-twist -> ground/free solve -> slipstream-q on CDp.
    """
    pol = polar if polar is not None else prob.polar

    # --- compressibility gate: pg_beta enforces 0 <= M < PG_MACH_MAX and maps
    # the transonic/negative ValueError to PENALTY. At M = 0 (default) this is a
    # no-op returning 1.0 and swept_section_slope short-circuits to the exact
    # legacy a0 cos Λ, so the whole path stays bit-for-bit.
    try:
        pg_beta(prob.mach)
    except ValueError as exc:
        return _fail(f"compressibility: {exc}")
    a_sec = float(swept_section_slope(pol.a_lin, sweep_deg, mach=prob.mach))

    yv = geometry.cosine_stations(prob.N, prob.b)[1]

    # --- slipstream compose step 1: swirl dalpha(y) is twist-like, so it is
    # ADDED to the geometric twist BEFORE the (free-air OR ground) solve. The
    # dynamic-pressure ratio q_ratio(y) is deferred to the profile-drag integral
    # below (lift-side spanwise-q variation is neglected — see Problem docstring).
    # OFF spec -> (ones, zeros) bit-for-bit, so twist_used IS twist_rad and
    # q_ratio stays None (the integrand executes the exact legacy expression).
    twist_used = twist_rad
    q_ratio = None
    if prob.slipstream is not None:
        q_ratio, dalpha_ss = prob.slipstream.modifiers(yv, c)
        twist_used = np.asarray(twist_rad, dtype=float) + dalpha_ss

    ground = prob.ground_h_m is not None
    if ground:
        # lazy import to avoid the objective <-> tandem circular import
        from .ground_effect import solve_wing_ige, solve_wing_ige_trim
        from .tandem import Surface
    try:
        if alpha_deg is None:
            if ground:
                alpha, res = solve_wing_ige_trim(
                    prob.b, c, twist_used, prob.CL_target, prob.ground_h_m,
                    a=a_sec, alpha_L0=pol.alpha_L0, V=prob.V,
                )
            else:
                alpha, res = solve_llt_trim(
                    prob.b, c, twist_used, prob.CL_target,
                    a=a_sec, alpha_L0=pol.alpha_L0, V=prob.V,
                )
        else:
            alpha = float(np.deg2rad(alpha_deg))
            if ground:
                res = solve_wing_ige(
                    Surface(b=prob.b, c=c, alpha_geo=alpha + twist_used,
                            a=a_sec, alpha_L0=pol.alpha_L0),
                    prob.ground_h_m, V=prob.V,
                )
            else:
                res = solve_llt(
                    prob.b, c, alpha + twist_used,
                    a=a_sec, alpha_L0=pol.alpha_L0, V=prob.V,
                )
    except (ValueError, np.linalg.LinAlgError) as exc:
        return _fail(f"solver failure: {exc}")

    # Normalise the free-air (LLTResult) and ground (GroundEffectResult) outputs
    # to a common set of scalars. Ground effect carries its induced-drag
    # reduction in CDi_total and its upwash in foil.alpha_eff_y; the profile-drag
    # integration and validity check downstream are IDENTICAL to free air.
    if ground:
        alpha_eff_y = res.foil.alpha_eff_y
        S_sol, CDi, CL, e_span, AR = (
            res.foil.S, res.CDi_total, res.CL, res.e_total, res.foil.AR
        )
    else:
        alpha_eff_y = res.alpha_eff_y
        S_sol, CDi, CL, e_span, AR = res.S, res.CDi, res.CL, res.e, res.AR

    alpha_eff_deg = np.rad2deg(alpha_eff_y)
    lo, hi = pol.alpha_valid
    if alpha_eff_deg.min() < lo or alpha_eff_deg.max() > hi:
        return _fail(
            "effective AoA outside polar validity (stall/extrapolation proxy)",
            alpha_eff_range=(float(alpha_eff_deg.min()), float(alpha_eff_deg.max())),
        )

    # --- slipstream compose step 2: q_ratio(y) scales the profile-drag
    # integrand only. q_ratio is None when OFF -> the exact legacy integrand.
    cd_y = pol.cd(alpha_eff_deg)
    integrand = cd_y * c
    if q_ratio is not None:
        integrand = integrand * q_ratio
    CDp = float(np.trapezoid(integrand, yv) / S_sol)
    CD = CDi + CDp + prob.cd0_extra
    LoD = CL / CD

    if not np.isfinite(LoD):
        return _fail("non-finite L/D")

    score = LoD
    if prob.mode == "soft":
        score = LoD - prob.soft_weight * (CL - prob.CL_target) ** 2

    mac = mac if mac is not None else float(np.mean(c))
    return {
        "feasible": True,
        "reason": "",
        "score": float(score),
        "LoD": float(LoD),
        "CL": CL,
        "CL_target": float(prob.CL_target),
        "CDi": CDi,
        "CDp": CDp,
        "cd0_extra": prob.cd0_extra,
        "CD": float(CD),
        "e": e_span,
        "AR": AR,
        "alpha_deg": float(np.rad2deg(alpha)),
        "alpha_eff_range_deg": (float(alpha_eff_deg.min()), float(alpha_eff_deg.max())),
        "Re_mac": float(prob.rho * prob.V * mac / prob.mu),
        "polar": getattr(pol, "name", "unknown"),
        "llt": res,
    }


#: every mode whose vector carries a winglet (and therefore runs the VLM)
WINGLET_MODES = ("winglet", "winglet_capped", "winglet_tc",
                 "winglet_capped_tc", "winglet_blended",
                 "winglet_capped_blended", "winglet_capped_blended_wing",
                 "winglet_coupled", "winglet_capped_coupled")
#: winglet modes whose vector carries the blend fraction at x[5]
BLENDED_MODES = ("winglet_blended", "winglet_capped_blended",
                 "winglet_capped_blended_wing")
#: winglet modes whose wing span shrinks to hold the PROJECTED span
CAPPED_MODES = ("winglet_capped", "winglet_capped_tc",
                "winglet_capped_blended", "winglet_capped_blended_wing",
                "winglet_capped_coupled")
#: modes that SELECT their section by thickness off the NACA 24XX polar
#: family (``polar_family``) rather than flying the single ``polar`` table —
#: so there is no fixed section for a chosen one to replace. api.TC_MODES
#: mirrors this list (and a test gates the two together).
#:
#: The COUPLED modes are DELIBERATELY NOT here even though they too select a
#: section rather than fly a fixed one: membership routes to the ONE-argument
#: ``family.at(tc)``, and their library is two-dimensional. They are listed
#: in COUPLED_MODES instead, and a test gates the two lists disjoint.
TC_MODES = ("tier_a_plus", "winglet_tc", "winglet_capped_tc")
#: modes that select their section from the pre-optimised (t/c x cl) CST
#: library through the TWO-argument ``PolarFamily2D.at(tc_sec, cl_sec)`` —
#: the summary-variable coupling wing_airfoil.py flies, given a tip device.
#: Re-exported from geometry, which owns the three places that must agree
#: (the design box, the thickness index, the span cap).
COUPLED_MODES = geometry.COUPLED_MODES


def _flow_state(prob: "Problem", x) -> tuple:
    """``(rho, V)`` a candidate flies at, BEFORE anything needs an area.

    Only the wing-loading size mode needs this: the area it derives depends
    on the dynamic pressure, and with the flight modifier on that pressure
    is itself a design variable. Density and speed do not depend on the
    area, so they can be read early — the trim target, which does, is still
    built once, later, from the resolved size (mission.flight_state).
    """
    from .sizing import flow_state_for

    return flow_state_for(x, mission=prob.mission,
                          flight_free=prob.flight_free,
                          n_trailing=prob.chord_order,
                          rho=prob.rho, V=prob.V)


def evaluate_winglet(
    wing: geometry.Wing,
    h_frac: float,
    cant_deg: float,
    prob: Problem,
    polar: Any = None,
    blend_frac: float = 0.0,
    wing_blend_frac: float = 0.0,
) -> dict:
    """Tier A+ physics core: wing + winglet -> L/D breakdown via the VLM.

    Mirrors evaluate_geometry but with the nonplanar solver: closed-form trim
    to prob.CL_target, TREFFTZ-plane CDi, profile drag from the section polar
    integrated over main-wing AND winglet strips (winglet reuses the wing
    polar at the alpha-method effective angle — documented approximation).
    Span efficiency e is referenced to the planar span, so e > 1 is possible
    (Munk 1921). ``polar`` overrides prob.polar (the winglet_tc modes pass
    the t/c-selected family member — same pattern as tier_a_plus); the
    default None keeps the legacy path bit-for-bit.

    ``wing_blend_frac`` extends the transition INBOARD of the tip, so the
    outer wing turns into the device (geometry.span_path). It is a fraction
    of the semi-span the VLM is built on, exactly as ``h_frac`` is, so the
    caller rescales both together when a span cap has shrunk the wing.
    """
    pol = polar if polar is not None else prob.polar
    try:
        beta = pg_beta(prob.mach)   # 1.0 exactly at M = 0 (bit-for-bit)
    except ValueError as exc:
        return _fail(f"compressibility: {exc}")
    try:
        model = VLM(
            wing, N=prob.N_vlm, winglet_h_frac=h_frac,
            winglet_cant_deg=cant_deg, n_winglet=prob.n_winglet,
            winglet_blend_frac=blend_frac,
            winglet_blend_shape=prob.blend_shape,
            winglet_wing_blend_frac=wing_blend_frac,
            winglet_chord_follows=prob.winglet_chord_follows,
            a=pol.a_lin / beta, alpha_L0=pol.alpha_L0, V=prob.V,
        )
        alpha, res = model.solve_trim(prob.CL_target)
    except (ValueError, np.linalg.LinAlgError) as exc:
        return _fail(f"solver failure: {exc}")

    alpha_eff_deg = np.rad2deg(res.alpha_eff)
    lo, hi = pol.alpha_valid
    if alpha_eff_deg.min() < lo or alpha_eff_deg.max() > hi:
        return _fail(
            "effective AoA outside polar validity (stall/extrapolation proxy)",
            alpha_eff_range=(float(alpha_eff_deg.min()), float(alpha_eff_deg.max())),
        )

    cd_y = pol.cd(alpha_eff_deg)
    CDp = float(np.sum(cd_y * res.c * res.width) / res.S)

    # wing/winglet corner: invisible to the Trefftz plane, so it is a
    # reduced-order add-on and never on by default (junction.py)
    jr = None
    if prob.junction_drag and float(h_frac) > 0.0:
        from . import junction
        c_tip = float(wing.chord(np.array([wing.b / 2.0]))[0])
        jr = junction.report(
            tc=wing.tc, chord=c_tip, s_ref=wing.S,
            h=float(h_frac) * wing.b / 2.0, cant_deg=cant_deg,
            blend_frac=blend_frac, n_junctions=2,
            blend_shape=prob.blend_shape,
            wing_arc=float(wing_blend_frac) * wing.b / 2.0)
    CD_junction = float(jr["CD_junction"]) if jr else 0.0

    CD = res.CDi + CDp + prob.cd0_extra + CD_junction
    LoD = res.CL / CD

    if not np.isfinite(LoD):
        return _fail("non-finite L/D")

    S_wl = float(np.sum((res.c * res.width)[res.is_winglet]))
    c_wl = res.c[res.is_winglet]
    return {
        "feasible": True,
        "reason": "",
        "score": float(LoD),
        "LoD": float(LoD),
        "CL": res.CL,
        "CL_target": float(prob.CL_target),
        "CDi": res.CDi,
        "CDp": CDp,
        "cd0_extra": prob.cd0_extra,
        "CD": float(CD),
        "e": res.e,
        "AR": res.AR,
        "alpha_deg": float(np.rad2deg(alpha)),
        "alpha_eff_range_deg": (float(alpha_eff_deg.min()), float(alpha_eff_deg.max())),
        "Re_mac": float(prob.rho * prob.V * wing.mac / prob.mu),
        "polar": getattr(pol, "name", "unknown"),
        "winglet": {
            "h_frac": float(h_frac),
            "cant_deg": float(cant_deg),
            # the height that reached the lattice; h_frac above stays the
            # REQUESTED number, so the pair reads asked / flown
            "h_m": float(model.winglet_h_m),
            "S_planform": S_wl,          # both winglets, one side counted
            # what the device's chord DID: the narrowest panel chord on it,
            # and whether it was drawn by continuing the wing's chord law
            # (vlm.winglet_chord_follows) or held at the tip chord. The
            # narrowest chord is the buildability number a rectangular device
            # never had to report — there it is the tip chord by construction.
            **({"chord_min_m": float(np.min(c_wl))} if c_wl.size else {}),
            "chord_follows": bool(prob.winglet_chord_follows),
            "blend_frac": float(blend_frac),
            "blend_shape": str(prob.blend_shape),
            "wing_blend_frac": float(wing_blend_frac),
            "wing_blend_arc_m": float(wing_blend_frac) * wing.b / 2.0,
            "tip_height_m": geometry.winglet_tip_height(
                model.winglet_h_m, cant_deg, blend_frac,
                prob.blend_shape,
                float(wing_blend_frac) * wing.b / 2.0),
            "projection_m": geometry.winglet_projection(
                model.winglet_h_m, cant_deg, blend_frac,
                prob.blend_shape,
                float(wing_blend_frac) * wing.b / 2.0),
            "projected_semispan_m": geometry.span_projection(
                wing.b / 2.0, model.winglet_h_m, cant_deg,
                blend_frac, prob.blend_shape, float(wing_blend_frac)),
            **({"junction": jr} if jr else {}),
        },
        "CD_junction": CD_junction,
        "vlm": res,
    }


def evaluate(x: np.ndarray, prob: Problem | None = None) -> dict:
    """Full evaluation with breakdown. Never raises for in-contract failures."""
    prob = prob or Problem()
    x = np.asarray(x, dtype=float)
    bnds = prob.bounds

    if x.shape != (bnds.shape[0],):
        return _fail(f"design vector shape {x.shape} != ({bnds.shape[0]},)")
    if np.any(x < bnds[:, 0] - 1e-12) or np.any(x > bnds[:, 1] + 1e-12):
        return _fail("bounds violation")

    # ---- size (sizing.py's modifier): the size comes out of the vector
    # BEFORE the wing is built, so everything downstream — the planform, the
    # trim target, the weight and the spar — describes the same aircraft.
    size = None
    b_use, S_use = prob.b, prob.S
    if prob.size_free:
        from .sizing import WS_MODES, check_ar, resolve_size, size_mode
        kw = {"wing_loading_max_Pa": prob.wing_loading_max_Pa}
        if size_mode(prob.size_free) in WS_MODES:
            # the AREA follows the chosen wing loading, and a loading is
            # quoted at the flow state the candidate flies in — so the
            # flight block is read first (it needs no area), and the area's
            # own fixed point is closed against that q. The trim target
            # falls out of it: CL = W_total/(q S) = (W/S)/q.
            from .mission import design_weight_n
            rho_ws, V_ws = _flow_state(prob, x)
            kw.update(
                # ignored by the searched-loading mode, which reads its own
                # row out of the vector (sizing.resolve_spans)
                wing_loading_Pa=design_wing_loading_pa(prob),
                # STATED first. weight_for un-derives at SEA LEVEL, and
                # CL_target was derived at the MISSION's altitude, so the two
                # densities do not cancel: a mission at 3000 m flew
                # W_N x 1.3475. See mission.design_weight_n.
                W_fixed_N=design_weight_n(prob),
                taper=float(x[0]), tc=geometry.tc_from_x(x, prob.mode),
                q_Pa=0.5 * rho_ws * V_ws ** 2)
        try:
            size = resolve_size(x, prob.size_free, prob.flight_free,
                                prob.chord_order, **kw)
        except (ValueError, OverflowError) as exc:
            return _fail(f"size: {exc}")
        b_use, S_use = size
        reason = check_ar(b_use, S_use)
        if reason is not None:
            return _fail(f"size: {reason}")

    # ...and the ASPECT-RATIO LIMIT THE USER SET, which is a different
    # question from the validity band above and is therefore asked
    # whether or not the size is a design variable: a limit that only
    # applied to searched sizes would be a limit that went quiet exactly
    # when the user typed the number it is about.
    if prob.ar_limits is not None:
        from .sizing import check_ar_limit
        reason = check_ar_limit(b_use, S_use, prob.ar_limits)
        if reason is not None:
            return _fail(f"size: {reason}")

    try:
        wing = geometry.wing_from_x(x, b=b_use, S=S_use, mode=prob.mode,
                                    chord_order=prob.chord_order,
                                    chord_law=prob.chord_law,
                                    blend_shape=prob.blend_shape,
                                    wing_blend_frac=prob.wing_blend_frac,
                                    blend_frac=prob.blend_frac_fixed,
                                    chord_limits=prob.chord_limits)
    except ValueError as exc:
        # a chord law that collapses the chord is an in-contract failure of a
        # VALID configuration (the box allows coefficient sums the positivity
        # floor does not) -> penalty, never an exception into the optimiser.
        return _fail(f"planform: {exc}")

    # ---- flight state: mode="mission" reads its own (x[3], x[4]); the
    # flight MODIFIER reads the pair ahead of the chord block (geometry.py).
    # Both rebuild the SAME thing — the flow state and the trim target the
    # design weight implies at that (V, h) — so they share one code path and
    # the modifier works in the VLM winglet modes too, which the mode never
    # could. prob is never mutated: replace() constructs copies.
    eval_prob = prob
    flight = ((float(x[3]), float(x[4])) if prob.mode == "mission"
              else geometry.flight_from_x(x, prob.flight_free,
                                          prob.chord_order))
    if flight is not None:
        from .mission import flight_state
        if prob.mission is None:
            return _fail(
                "a free flight state requires prob.mission "
                "(mission.MissionSpec carrying the design weight W_N)")
        try:
            eval_prob = replace(
                prob, mission=None, flight_free=False,
                **flight_state(prob.mission, flight[0], flight[1], S_use))
        except ValueError as exc:  # water medium, or ISA validity
            return _fail(f"flight state: {exc}")

    sized = None
    if size is not None:
        from .mission import design_weight_n
        from .sizing import sized_state
        q = 0.5 * eval_prob.rho * eval_prob.V**2
        W_fixed = design_weight_n(prob)
        try:
            sized = sized_state(
                W_fixed_N=W_fixed, b=b_use, S=S_use, taper=wing.taper,
                tc=wing.tc, q_Pa=q, sweep_deg=wing.sweep_deg,
                c_root=float(wing.chord(np.array([0.0]))[0]),
                wing_loading_max_Pa=prob.wing_loading_max_Pa,
                material=prob.material)
        except (ValueError, RuntimeError, OverflowError) as exc:
            return _fail(f"size: {exc}")
        # b and S also move the REFERENCE the solver is built on, so the
        # copy carries them alongside the weight-coupled trim target
        eval_prob = replace(eval_prob, b=b_use, S=S_use, size_free=False,
                            CL_target=sized.CL_target)
        # ...and AGAIN, after the copy. ``dataclasses.replace`` re-runs
        # ``__post_init__``, which re-derives CL_target from the mission
        # whenever one is set (see the mission branch above) — so the
        # weight-coupled target just passed in was silently overwritten with
        # W_N/(q S). That put the PAYLOAD weight in the force balance and
        # left the wing's own weight out of it: measured lift = 650.0 N
        # against W_total = 2079.4 N, i.e. the aircraft trimmed to carry 31 %
        # of itself, across 13 registered base families. It flattered the
        # score by 3.4x (14.88 -> 50.17) because CD was read at a third of
        # the CL the design actually needs. Assigned last, so nothing can
        # re-derive it.
        eval_prob.CL_target = float(sized.CL_target)

    if prob.mode in WINGLET_MODES:
        if prob.ground_h_m is not None:
            # winglet modes use the nonplanar VLM (evaluate_winglet); the rigid-
            # ground image system lives in the LLT path (evaluate_geometry) only.
            return _fail("ground effect not implemented for VLM winglet modes")
        if prob.slipstream is not None:
            # the slipstream skin composes into the LLT strip integration
            # (evaluate_geometry) only — failing loudly beats ignoring it.
            return _fail("slipstream not implemented for VLM winglet modes")
        h_frac = float(x[3])
        wbf = geometry.wing_blend_from_x(x, prob.mode, prob.wing_blend_frac)
        if prob.mode in CAPPED_MODES:
            # wing span shrank to hold projected span = b_use (geometry.py);
            # rescale so the winglet ARC LENGTH stays x[3] * (b_use / 2) —
            # and, for the same reason, so the WING-side arc of the
            # transition stays wbf * (b_use / 2). Both are fixed against the
            # nominal span, which is what makes the cap a closed-form solve
            # rather than a fixed point (geometry.developed_semispan).
            h_frac *= b_use / wing.b
            wbf *= b_use / wing.b
        pol = None
        if prob.mode in ("winglet_tc", "winglet_capped_tc"):
            try:
                pol = prob.polar_family.at(wing.tc)  # exact member or blend
            except ValueError as exc:
                return _fail(f"polar family: {exc}")
        elif prob.mode in COUPLED_MODES:
            # the SUMMARY-VARIABLE lookup: two arguments, structural depth
            # and design lift, into the pre-optimised CST library
            # (wing_airfoil.py's module docstring A carries the semantics —
            # cl_sec selects WHICH pre-optimised member, it does not force
            # the wing to operate there; the wing still trims to CL_target
            # and pays the mismatch through the polar). x[5] is already on
            # the Wing as its thickness (geometry.wing_from_x), so read it
            # from there and keep one owner for the row.
            try:
                pol = prob.polar_family.at(wing.tc, float(x[6]))
            except ValueError as exc:
                return _fail(f"polar family: {exc}")
        # the blend fraction has ONE owner per mode: the design vector where
        # the mode designs it, the problem's own value everywhere else
        # (0.0 = the sharp corner, so every published winglet run is
        # bit-for-bit). __post_init__ refuses the pair.
        blend = (float(x[5]) if prob.mode in BLENDED_MODES
                 else float(prob.blend_frac_fixed))
        out = evaluate_winglet(wing, h_frac, float(x[4]), eval_prob, polar=pol,
                               blend_frac=blend, wing_blend_frac=wbf)
    else:
        _, c, twist = wing.sample(prob.N)
        alpha_deg = float(x[3]) if prob.mode == "soft" else None

        pol = None
        if prob.mode == "tier_a_plus":
            try:
                pol = prob.polar_family.at(wing.tc)  # exact member or blend
            except ValueError as exc:
                return _fail(f"polar family: {exc}")

        out = evaluate_geometry(c, twist, eval_prob, alpha_deg=alpha_deg,
                                mac=wing.mac, sweep_deg=wing.sweep_deg,
                                polar=pol)
    if sized is not None and out["feasible"]:
        # payload L/D (sizing.py): W varies across candidates, so "maximise
        # L/D" would reward a heavier wing that lifts better. W_fixed is the
        # same constant for every candidate, so this IS minimum drag.
        q = 0.5 * eval_prob.rho * eval_prob.V**2
        f = sized.payload_lod(q, out["CD"])
        if not np.isfinite(f):
            return _fail("non-finite payload L/D")
        out.update(sized.report())
        out["f"] = float(f)
        out["score"] = float(f)
        out["g"] = float(sized.g_sigma)
        out["D_N"] = float(q * sized.S * out["CD"])
    if out["feasible"]:
        # THE POINT THIS WAS FLOWN AT, on every exit and not only under the
        # flight modifier — same reason as ``wingtail``: a report that
        # states no speed is rebuilt at ``flightmodel.DEFAULT_V_MS`` and has
        # its mass back-solved there, which is a different aeroplane.
        out["V"] = eval_prob.V
        out["rho"] = eval_prob.rho
        out["mu"] = eval_prob.mu
    if flight is not None and out["feasible"]:
        out["altitude_m"] = float(flight[1])
    if out["feasible"]:
        out["wing"] = wing
        out["sweep_deg"] = wing.sweep_deg
        out["tc"] = wing.tc
        if prob.mode in COUPLED_MODES:
            # the DEMAND the wing level made on the section library, under
            # the same two keys wing_airfoil.evaluate_wing_airfoil reports
            # them: a breakdown carrying only "tc" cannot say which member
            # was flown, because the design lift is half the index
            out["tc_sec"] = float(wing.tc)
            out["cl_sec"] = float(x[6])
        if wing.chord_coeffs:
            out["chord_coeffs"] = list(wing.chord_coeffs)
            # ...and WHICH SHAPE they are the parameters of: the same three
            # numbers draw four different planforms, so a breakdown carrying
            # only the numbers cannot be redrawn
            out["chord_law"] = wing.chord_law
            out["chord_dev"] = wing.chord_dev
            out["taper"] = float(wing.taper)   # BASELINE taper; the flown tip
            #   chord is taper * multiplier(1) * area factor, so the trapezoid
            #   ratio alone no longer describes the planform
    return out


def objective(x: np.ndarray, prob: Problem | None = None) -> float:
    """Scalar objective (MAXIMISE): L/D, or PENALTY on any failure."""
    out = evaluate(x, prob)
    return float(out["score"]) if out["feasible"] else PENALTY


#: finite infeasible margin reported on solver failure by the SIZED variants
#: (aircraft.py's constant — the same contract, since it is the same
#: root-bending constraint).
G_FAIL = -1.0


def fg(x: np.ndarray, prob: Problem | None = None) -> tuple[float, float]:
    """Constrained-harness callable for the SIZE modifier: (payload L/D,
    signed root-bending stress margin).

    Solvable-but-overstressed designs return their TRUE payload L/D with a
    negative margin (a constrained optimiser has to see the violation's
    magnitude); solver failures return exactly (PENALTY, G_FAIL).
    """
    out = evaluate(x, prob)
    if not out["feasible"]:
        return PENALTY, G_FAIL
    if "g" not in out:
        raise ValueError("fg() is for the size-modifier variants — this "
                         "problem has no constraint (use objective())")
    return float(out["score"]), float(out["g"])
