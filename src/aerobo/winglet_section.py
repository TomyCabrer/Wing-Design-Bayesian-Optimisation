"""13-D winglet + free-form CST section co-design, real XFOIL in the loop.

The full-fidelity composition of two existing tiers:

    x = [taper, twist_root_deg, twist_tip_deg,          (wing/winglet block)
         winglet_h_frac, winglet_cant_deg,
         w_upper_0..3, w_lower_0..3]                    (CST section block)

Per evaluation: the candidate CST section gets a live viscous XFOIL alpha
sweep (aerobo.xfoil_run — cached on disk by coordinate hash, so repeated
sections are free), the converged table becomes a :class:`polar.TablePolar`
(a_lin / alpha_L0 extracted from the linear range), and that polar drives
the SAME nonplanar-VLM wing+winglet trim evaluation as the winglet tiers
(``objective.evaluate_winglet(polar=...)``).

Objective (MAXIMISE): trimmed wing+winglet L/D.
Constraints (identical semantics to airfoil.AirfoilProblem, feasible >= 0):

    g0 = (t/c - tc_min) / tc_min          structural-depth proxy
    g1 = (cm_max - |cm(cl_design)|)/cm_max  trim-drag proxy

Failure contract: fg returns exactly (PENALTY, [G_FAIL, G_FAIL]) for any
in-contract failure (bad geometry, unconvergeable section, VLM/trim
failure); XfoilError (broken install) propagates — a study scored -100 on
a missing binary would be fabricated.

Block structure (the "portfolio" decomposition the optimize.blocks runner
consumes): the wing/winglet block sees millisecond VLM evals once the
section polar is cached; the section block pays the XFOIL sweep. Different
costs -> different optimisers per block (report §13's prescription).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import geometry
from .airfoil import (
    G_FAIL,
    AirfoilProblem,
    cst_coords,
    cst_thickness,
    norm_margin,
    _longest_increasing_run,
    _surface_y,
)
from .objective import MU_SL, PENALTY, RHO_SL, Problem, _fail, evaluate_winglet
from .polar import TablePolar


@dataclass
class WingletSectionProblem:
    """Wing + winglet planform with a free-form CST section (d = 13).

    ``section`` carries the CST box/anchor, XFOIL sweep settings and the
    two constraint parameters (tc_min, cm_max) — the exact
    AirfoilProblem machinery, re-used. ``capped`` selects the span-capped
    winglet geometry (winglet_capped rules) instead of the free span
    extension.
    """

    section: AirfoilProblem = field(default_factory=AirfoilProblem)
    capped: bool = False
    winglet: bool = True        # False -> plain trim wing + CST section: the
    #   same section co-design with NO tip device (d = 11, or 14 with a chord
    #   law). Not the same thing as a winglet whose height optimises to zero:
    #   the two winglet variables are GONE, so the optimiser never spends
    #   budget on them and the design box does not offer them.
    b: float = 10.0
    S: float = 10.0
    N_vlm: int = 40
    n_winglet: int = 8
    V: float = 14.6
    rho: float = RHO_SL
    mu: float = MU_SL
    CL_target: float = 0.5
    cd0_extra: float = 0.0
    chord_order: int = 0        # free chord law (geometry.py). The rows are
    #   appended to the WING block, so the section block is untouched and the
    #   portfolio decomposition below still holds: shaping the planform is
    #   cheap (cached polar + VLM), shaping the section is not.
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
    winglet_chord_follows: bool = False   # the tip device's chord CONTINUES
    #   the wing's own chord law past the tip instead of holding the tip
    #   chord (vlm.py). Forwarded to the wing/VLM state below, which is where
    #   objective.Problem carries it. False = every published run,
    #   bit-for-bit.
    # --- the tip device's ROOT, as VALUES (objective.Problem's own field
    #     names, since that is what this family's wing state IS). They are
    #     here because a blend is a SHAPE of tip device, not a family: the
    #     shell offers "blended winglet" beside every other shape, and this
    #     family — the one that designs the section WITH the device — was the
    #     last one drawing that corner as a corner. It cost nothing to draw:
    #     the inner Problem's mode is already "winglet"/"winglet_capped",
    #     which is exactly where a fixed blend lives.
    blend_frac_fixed: float = 0.0   # share of the device's arc spent turning
    #   out of the wing plane. 0.0 = the sharp corner every published run of
    #   this family flew, bit-for-bit.
    blend_shape: str = "arc"        # which turn law draws it (geometry.
    #   BLEND_SHAPES); "arc" is the constant-radius fillet the library
    #   defaults to everywhere.
    wing_blend_frac: float = 0.0    # ...and how much of the turn the WING
    #   does, as a fraction of the semi-span (geometry.span_path).
    junction_drag: bool = False     # charge the corner's interference drag
    #   (junction.py). The api turns it ON with a blend for the same reason
    #   every other family does: without the charge a blend is only a
    #   differently drawn wake.
    material: object = None   # WHAT THE WING IS BUILT OF
    #   (materials.Material, or a key from materials.MATERIALS).
    #   None = 2024-T3 aluminium, the material Raymer's correlation
    #   was regressed over, so every published run is bit-identical.
    #   It sets BOTH the allowable the spar is sized to and the
    #   factor the statistical wing weight is scaled by — the second
    #   is the one the span answer actually rides (materials.py).
    size_free: bool = False     # SIZE modifier (sizing.py): the rows join
    #   the WING block like the other modifiers, so the portfolio split is
    #   untouched. With it on the score is payload L/D and the root-bending
    #   stress margin becomes a THIRD constraint beside t/c and |Cm| — and it
    #   is owned by the wing block, not the section block, which is what
    #   keeps optimize.blocks' "who can repair infeasibility" honest.
    span_bounds_m: tuple | None = None     # (min, max) SPAN row [m] of the
    #   size modifier; None -> the fractional band around this problem's own
    #   span (sizing.span_bounds)
    area_bounds_m2: tuple | None = None    # (min, max) AREA row [m^2]; None
    #   -> the fractional band around its own reference area. Stating it is
    #   what lets a search reach a mission of a different weight at all
    wing_loading_max_Pa: float | None = None   # the MISSION's ceiling on W/S
    #   (constraint_diagram): any candidate above it is refused, in-contract
    W_fixed_N: float | None = None    # non-wing weight for the size modifier
    flight_free: bool = False   # speed + altitude as design variables
    #   (geometry.py's flight modifier). The rows join the WING block, ahead
    #   of the chord coefficients and BEFORE the section weights, so the
    #   portfolio split below is untouched: shaping the planform and choosing
    #   where to fly it are both cheap (cached polar + VLM), shaping the
    #   section is not. What the modifier buys here is the honest version of
    #   "design the aerofoil for the mission": the section is shaped at the
    #   CL the candidate's own (V, altitude) demands.
    mission: Any = None         # mission.MissionSpec carrying the design
    #   weight; defaults to the weight this problem's own CL_target implies

    def __post_init__(self):
        if not self.winglet:
            mode = "trim"
        else:
            mode = "winglet_capped" if self.capped else "winglet"
        if self.flight_free and self.mission is None:
            from .mission import MissionSpec, weight_for
            self.mission = MissionSpec(
                W_N=weight_for(self.CL_target, self.V, self.S), V=self.V,
                altitude_m=0.0)
        # the VLM/trim state container; the polar is supplied per-eval.
        # rho/mu are carried so a mission (weight + altitude) sets the whole
        # operating point consistently: they do not enter the VLM, but they
        # DO set the reported Re_mac, which must not disagree with the
        # CL_target the same mission implies.
        self._wing_prob = Problem(
            b=self.b, S=self.S, mode=mode, V=self.V, rho=self.rho,
            mu=self.mu, CL_target=self.CL_target, N_vlm=self.N_vlm,
            n_winglet=self.n_winglet, cd0_extra=self.cd0_extra,
            chord_order=self.chord_order, chord_max_frac=self.chord_max_frac,
            chord_law=self.chord_law, chord_limits=self.chord_limits,
            # ...the aspect-ratio limit too, so the NO-DEVICE branch — which
            # hands the whole wing sub-vector to ``objective.evaluate`` — is
            # judged against the same limit as the tip-device branch below
            # rather than silently dropping it
            ar_limits=self.ar_limits,
            winglet_chord_follows=self.winglet_chord_follows,
            blend_frac_fixed=self.blend_frac_fixed,
            blend_shape=self.blend_shape,
            wing_blend_frac=self.wing_blend_frac,
            junction_drag=self.junction_drag,
            flight_free=self.flight_free, mission=self.mission,
            size_free=self.size_free, W_fixed_N=self.W_fixed_N,
            span_bounds_m=self.span_bounds_m,
            area_bounds_m2=self.area_bounds_m2,
            wing_loading_max_Pa=self.wing_loading_max_Pa)
        from .sizing import size_bounds
        self._wing_bounds = geometry.bounds(   # (5 + size + flight + chord, 2)
            mode, chord_order=self.chord_order,
            chord_max_frac=self.chord_max_frac,
            chord_law=self.chord_law,
            flight_free=self.flight_free,
            size_rows=(size_bounds(self.b, self.S, 1, self.span_bounds_m,
                                   self.area_bounds_m2)
                       if self.size_free else None))

    @property
    def n_wing(self) -> int:
        return self._wing_bounds.shape[0]

    @property
    def bounds(self) -> np.ndarray:
        return np.vstack([self._wing_bounds, self.section.bounds])

    @property
    def dim(self) -> int:
        return self.n_wing + self.section.dim

    @property
    def x0(self) -> np.ndarray:
        """Warm-start anchor: wing-block box midpoint + the section anchor
        (NACA 2412 by default — feasible on both margins)."""
        wb = self._wing_bounds
        return np.concatenate([0.5 * (wb[:, 0] + wb[:, 1]),
                               self.section.w0])

    @property
    def blocks(self) -> tuple:
        """Portfolio block metadata for :mod:`aerobo.optimize.blocks`.

        Both the optimiser and the budget share per block are MEASURED
        choices, not preferences:

        * CST section (8-D, one XFOIL sweep per new shape) -> ``bo``.
          ``results/airfoil.json`` benchmarks this exact sub-problem — same
          parametrisation, same two constraints, same solver — at budget 60
          over 5 seeds: BO 54.6 drag counts, GA 56.1, random 59.0, SLSQP
          64.6 (worse than the NACA 2412 baseline's 59.6). Gradients are a
          liability once the solver is expensive and slightly noisy.
        * wing/winglet (5-D VLM) -> ``bo``.
          ``results/block_optimiser_bench.json`` measures precisely the
          sub-problem this block poses (section frozen at the anchor, so
          the polar is a cache hit) at the per-block budgets the runner
          actually hands out: BO wins at every one — 38.07 / 37.92 / 38.99
          median L/D at budgets 8 / 12 / 20, against SLSQP 37.82 / 37.82 /
          38.33 and sobol last. Without that measurement this block's
          optimiser was a judgement call.

        The share is cost-driven: a wing-block evaluation costs 2.5 ms
        (cached polar + VLM) against ~1.37 s for a section evaluation that
        needs a fresh sweep — a factor of ~550 — so spending twice as many
        evaluations on the planform is very nearly free in wall-clock terms.

        ``owns_constraints`` records that BOTH margins (t/c and |Cm|) are
        functions of the section weights alone: the wing block cannot repair
        an infeasible design, so the runner must not send it to try.
        """
        nw = self.n_wing
        return (
            {"name": "wing/winglet", "indices": tuple(range(nw)),
             "optimiser": "bo", "share": 2.0,
             "cost": "cheap (VLM + cached polar, ~2.5 ms/eval)",
             "owns_constraints": False},
            {"name": "CST section", "indices": tuple(range(nw, self.dim)),
             "optimiser": "bo", "share": 1.0,
             "cost": "expensive (live XFOIL sweep, ~1.4 s/new section)",
             "owns_constraints": True},
        )


def evaluate_winglet_section(x: np.ndarray,
                             prob: WingletSectionProblem | None = None
                             ) -> dict:
    """Full 13-D evaluation with breakdown; in-contract failures never raise.

    ``feasible`` refers to the SOLVER; the two design margins are reported
    separately as ``g`` (constrained BO needs true f at infeasible designs).
    """
    from .xfoil_run import run_xfoil_polar

    prob = prob or WingletSectionProblem()
    sec = prob.section
    x = np.asarray(x, dtype=float)
    bnds = prob.bounds
    if x.shape != (bnds.shape[0],):
        return _fail(f"design vector shape {x.shape} != ({bnds.shape[0]},)")
    if np.any(x < bnds[:, 0] - 1e-12) or np.any(x > bnds[:, 1] + 1e-12):
        return _fail("bounds violation")

    nw = prob.n_wing
    xw, xs = x[:nw], x[nw:]
    w_u, w_l = xs[: sec.n_cst], xs[sec.n_cst:]

    # ---- section: geometry prechecks (airfoil.py contract)
    psi = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, 401)))[1:-1]
    gap = _surface_y(psi, w_u, +0.5 * sec.dz_te) - _surface_y(
        psi, w_l, -0.5 * sec.dz_te)
    if not np.all(gap > 0.0):
        return _fail("self-intersecting / non-positive-thickness geometry")
    tc = cst_thickness(w_u, w_l, dz_te=sec.dz_te)
    coords = cst_coords(w_u, w_l, dz_te=sec.dz_te)

    # ---- section: cached viscous XFOIL sweep -> TablePolar
    pol_raw = run_xfoil_polar(
        coords, sec.re, sec.mach, sec.alphas,
        timeout_s=sec.timeout_s, cache_dir=sec.cache_dir,
        n_panel=sec.n_panel,
    )   # XfoilError (infrastructure fault) intentionally propagates
    if pol_raw.n_converged < 3:
        return _fail("unconvergeable section (< 3 converged XFOIL points) "
                     "= failed design", n_converged=pol_raw.n_converged)
    polar = TablePolar(alpha_deg=pol_raw.alpha_deg, CL=pol_raw.cl,
                       CD=pol_raw.cd, CM=pol_raw.cm,
                       name="cst-live", Re=sec.re)

    # ---- section margins at the design lift (airfoil.py semantics)
    sl = _longest_increasing_run(pol_raw.cl)
    cl_b, cm_b = pol_raw.cl[sl], pol_raw.cm[sl]
    if cl_b.size < 2 or cl_b[-1] < sec.cl_design or cl_b[0] > sec.cl_design:
        return _fail("cl_design not bracketed by the pre-stall monotone "
                     "branch = failed design",
                     n_converged=pol_raw.n_converged)
    cm_at = float(np.interp(sec.cl_design, cl_b, cm_b))
    g = np.array([norm_margin(tc - sec.tc_min, sec.tc_min),
                  norm_margin(sec.cm_max - abs(cm_at), sec.cm_max)])

    # ---- wing + winglet VLM trim with the live polar
    # the blend travels into the planform as well as into the panelisation:
    # a blended device reaches further outboard than h cos(cant), so a
    # span-capped wing has to be capped on the whole developed line
    # (geometry.wing_from_x; objective.evaluate does exactly this).
    blend_kw = dict(blend_frac=prob.blend_frac_fixed,
                    blend_shape=prob.blend_shape,
                    wing_blend_frac=prob.wing_blend_frac)
    try:
        wing = geometry.wing_from_x(xw, b=prob.b, S=prob.S,
                                    mode=prob._wing_prob.mode,
                                    chord_order=prob.chord_order,
                                    chord_law=prob.chord_law,
                                    chord_limits=prob.chord_limits,
                                    **blend_kw)
    except ValueError as exc:          # collapsed chord law
        return _fail(f"planform: {exc}")
    if prob.winglet:
        h_frac = float(xw[3])
        wbf = float(prob.wing_blend_frac)
        if prob.capped:
            h_frac *= prob.b / wing.b   # arc length stays x[3] * b/2
            wbf *= prob.b / wing.b      # ...and so does the WING-side arc
        # the tip-device branch calls the VLM core directly (it has already
        # built the wing), so the flight modifier is resolved HERE; the
        # branch below hands the whole wing sub-vector to objective.evaluate,
        # which resolves it itself.
        wing_prob = prob._wing_prob
        flight = geometry.flight_from_x(xw, prob.flight_free,
                                        prob.chord_order)
        # FLIGHT BEFORE SIZE, the order every sibling uses
        # (objective.py, wing_airfoil.py, tail.py). Run the other way round,
        # ``flight_state`` splices its own CL_target over the weight-coupled
        # one that ``sized_state`` had just produced — the same clobber
        # objective.py's note describes, re-entered by statement order — and
        # ``sized_state`` was handed a PRE-flight q (130.56 Pa against the
        # 162.02 Pa actually flown). Measured: the design flew carrying 46.9 %
        # of its own weight, and the score was inflated 2.305x
        # (15.21 -> 35.07) because CD was read at 43 % of the needed CL.
        if flight is not None:
            from dataclasses import replace as _replace

            from .mission import flight_state
            try:
                wing_prob = _replace(
                    wing_prob, mission=None, flight_free=False,
                    **flight_state(prob.mission, flight[0], flight[1],
                                   prob.S))
            except ValueError as exc:   # water medium, or ISA validity
                return _fail(f"flight state: {exc}")
        sized = None
        if prob.size_free:
            from dataclasses import replace as _replace0

            from .mission import design_weight_n
            from .sizing import check_ar, size_from_x, sized_state
            b_use, S_use = size_from_x(xw, True, prob.flight_free,
                                       prob.chord_order)
            reason = check_ar(b_use, S_use)
            if reason is not None:
                return _fail(f"size: {reason}")
            wing_prob = _replace0(wing_prob, b=b_use, S=S_use,
                                  size_free=False)
            wing = geometry.wing_from_x(xw, b=b_use, S=S_use,
                                        mode=wing_prob.mode,
                                        chord_order=prob.chord_order,
                                        chord_law=prob.chord_law,
                                        chord_limits=prob.chord_limits,
                                        **blend_kw)
            if prob.capped:
                h_frac = float(xw[3]) * b_use / wing.b
                wbf = float(prob.wing_blend_frac) * b_use / wing.b
            try:
                sized = sized_state(
                    # STATED first — see mission.design_weight_n
                    W_fixed_N=design_weight_n(prob),
                    b=b_use, S=S_use, taper=wing.taper, tc=wing.tc,
                    q_Pa=0.5 * wing_prob.rho * wing_prob.V**2,
                    c_root=float(wing.chord(np.array([0.0]))[0]),
                    wing_loading_max_Pa=prob.wing_loading_max_Pa,
                    material=prob.material)
            except (ValueError, RuntimeError, OverflowError) as exc:
                return _fail(f"size: {exc}")
            wing_prob = _replace0(wing_prob, CL_target=sized.CL_target)
            # ...and again after the copy: ``replace`` re-runs
            # ``objective.Problem.__post_init__``, which re-derives
            # CL_target from the mission and would put the PAYLOAD
            # weight back in the force balance while leaving the
            # wing's own weight out of it. See objective.py's note.
            wing_prob.CL_target = float(sized.CL_target)
        # ...and the ASPECT-RATIO LIMIT THE USER SET, off the wing that was
        # actually built. A different question from the validity band above,
        # so it is asked whether or not the size is a design variable: a
        # limit that only applied to searched sizes would go quiet exactly
        # when the user typed the number it is about.
        if prob.ar_limits is not None:
            from .sizing import check_ar_limit as _check_ar_limit
            why_ar = _check_ar_limit(float(wing.b), float(wing.S),
                                     prob.ar_limits)
            if why_ar is not None:
                return _fail(f"size: {why_ar}")
        out = evaluate_winglet(wing, h_frac, float(xw[4]), wing_prob,
                               polar=polar,
                               blend_frac=float(prob.blend_frac_fixed),
                               wing_blend_frac=wbf)
        if sized is not None and out["feasible"]:
            q = 0.5 * wing_prob.rho * wing_prob.V**2
            f = sized.payload_lod(q, out["CD"])
            if not np.isfinite(f):
                return _fail("non-finite payload L/D")
            out.update(sized.report())
            out.update({"f": float(f), "score": float(f),
                        "g": float(sized.g_sigma),
                        "D_N": float(q * sized.S * out["CD"])})
        if flight is not None and out["feasible"]:
            out.update({"V": float(wing_prob.V),
                        "altitude_m": float(flight[1]),
                        "rho": float(wing_prob.rho),
                        "mu": float(wing_prob.mu)})
    else:
        # no tip device: the SAME trim evaluation the Tier A wing uses, with
        # the live section polar substituted (dataclasses.replace keeps every
        # other operating-point field identical)
        from dataclasses import replace as _replace

        from .objective import evaluate as _evaluate_wing
        out = _evaluate_wing(xw, _replace(prob._wing_prob, polar=polar))
    if not out["feasible"]:
        return out

    if prob.size_free:
        # the wing solve already closed the weight loop and scored payload
        # L/D (objective.evaluate with size_free on); its stress margin joins
        # the two section margins as the third constraint
        g = np.concatenate([g, [float(out["g"])]])
    out.update({
        "g": g, "g_tc": float(g[0]), "g_cm": float(g[1]),
        "tc": float(tc), "cm_at_cl_design": cm_at,
        "cl_design": float(sec.cl_design),
        "n_converged": pol_raw.n_converged,
        "from_cache": pol_raw.from_cache,
        "wing": wing,
    })
    return out


def fg_winglet_section(x: np.ndarray,
                       prob: WingletSectionProblem | None = None
                       ) -> tuple[float, np.ndarray]:
    """Constrained-harness callable: (L/D, [g_tc, g_cm]).

    Solvable-but-infeasible designs return their TRUE L/D with signed
    margins; solver failures return exactly (PENALTY, [G_FAIL, G_FAIL])."""
    out = evaluate_winglet_section(x, prob)
    if not out["feasible"]:
        return PENALTY, np.array([G_FAIL, G_FAIL])
    return float(out["score"]), np.asarray(out["g"], dtype=float)
