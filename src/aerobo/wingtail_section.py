"""Wing + tip device + tail + a free-form CST SECTION, real XFOIL in the loop.

``winglet_section.py`` shapes a section together with a wing and its tip
device; ``wingtail.py`` puts a tail into the same nonplanar solve. This is the
composition of the two — the aeroplane problem with its aerofoil designed
rather than selected:

    x = [ wing + tail block (wingtail.WingTailProblem.param_labels) ]
        [ w_upper_0..3, w_lower_0..3 ]                (CST section block)

Per evaluation the candidate section gets a live viscous XFOIL alpha sweep
(cached on disk by coordinate hash), the converged table becomes a
:class:`polar.TablePolar`, and THAT polar drives the same coupled wing+tail
VLM trim wingtail.py already does — wing, winglet AND tail read the one
section, which is the honest version of "design the aerofoil for the
aeroplane" (the tail's own section is a separate question this model does not
ask: it flies the wing's).

Objective (MAXIMISE): L/D at the 2-D trim point — or payload L/D if the size
modifier is on, exactly as in the wing+tail family.

Constraints, in this order (feasible >= 0):

    g0  static margin - SM_min                (the family's own)
    [g1 root-bending stress margin            if the size modifier is on]
    g_tc  (t/c - tc_min)/tc_min               structural-depth proxy
    g_cm  (cm_max - |cm(cl_design)|)/cm_max   trim-drag proxy

The family's constraints come FIRST so that a variant's margin vector is its
base problem's with the section's two appended — the same rule the size
modifier follows.

``tc_free`` is refused: the CST weights ARE the thickness, and offering a
separate t/c variable would let the polar family and the live section
disagree about the same number.

Block structure (optimize.blocks): the wing+tail block is milliseconds once
the section polar is cached; the section block pays the XFOIL sweep. The
section block OWNS the two section margins — the wing block cannot repair an
infeasible aerofoil — while the static margin (and the stress margin) belong
to the wing block, so neither is listed as owned by the section.

Failure contract: the package's. XfoilError (a broken install) propagates;
every in-contract failure returns the penalty with a full margin vector.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .airfoil import (
    G_FAIL,
    AirfoilProblem,
    _longest_increasing_run,
    _surface_y,
    cst_coords,
    cst_thickness,
    norm_margin,
)
from .objective import PENALTY, _fail
from .polar import TablePolar
from .tandemvlm import TandemVLMProblem, evaluate_tandem_vlm
from .wingtail import WingTailProblem, evaluate_wing_tail

#: air family -> (evaluator, the field a designed polar is passed through,
#: how many margins the family owns before the section's two). The tandem
#: pair owns none unless the size modifier is on, which adds its own.
_FAMILIES = {
    WingTailProblem: (evaluate_wing_tail, "polar", 1),
    TandemVLMProblem: (evaluate_tandem_vlm, "section_polar", 0),
}


@dataclass
class WingTailSectionProblem:
    """Wing (+ tip device) + tail with a free-form CST section.

    ``wing_tail`` carries every wing/tail freedom (tip device, span
    accounting, blended transition, tail height, fixed arm, chord law,
    flight state, size) exactly as the family defines them; ``section``
    carries the CST box, the XFOIL sweep settings and the two design gates.
    """

    section: AirfoilProblem = field(default_factory=AirfoilProblem)
    wing_tail: object = field(default_factory=WingTailProblem)

    def __post_init__(self):
        if type(self.wing_tail) not in _FAMILIES:
            raise ValueError(
                f"unsupported family {type(self.wing_tail).__name__}; "
                f"choices: {[t.__name__ for t in _FAMILIES]}")
        if getattr(self.wing_tail, "tc_free", False):
            raise ValueError(
                "tc_free is refused here: the CST weights ARE the section, so "
                "a separate thickness variable would let the polar family and "
                "the live section disagree about t/c")

    @property
    def n_wing(self) -> int:
        return int(self.wing_tail.bounds.shape[0])

    @property
    def param_labels(self) -> tuple:
        n = self.section.n_cst
        return tuple(self.wing_tail.param_labels) + tuple(
            [f"w_upper_{i}" for i in range(n)]
            + [f"w_lower_{i}" for i in range(n)])

    @property
    def bounds(self) -> np.ndarray:
        return np.vstack([self.wing_tail.bounds, self.section.bounds])

    @property
    def dim(self) -> int:
        return int(self.bounds.shape[0])

    @property
    def n_constraints(self) -> int:
        """The family's own margins (+ stress, if sized) + the section's two."""
        own = _FAMILIES[type(self.wing_tail)][2]
        return own + 2 + (1 if self.wing_tail.size_free else 0)

    @property
    def x0(self) -> np.ndarray:
        """Warm-start anchor: wing-block box midpoint + the section anchor."""
        wb = self.wing_tail.bounds
        return np.concatenate([0.5 * (wb[:, 0] + wb[:, 1]), self.section.w0])

    @property
    def blocks(self) -> tuple:
        nw = self.n_wing
        return (
            {"name": "wing/tail", "indices": tuple(range(nw)),
             "optimiser": "bo", "share": 2.0,
             "cost": "cheap (coupled VLM + cached polar, ~5 ms/eval)",
             "owns_constraints": False},
            {"name": "CST section", "indices": tuple(range(nw, self.dim)),
             "optimiser": "bo", "share": 1.0,
             "cost": "expensive (live XFOIL sweep, ~1.4 s/new section)",
             "owns_constraints": True},
        )


def evaluate_wing_tail_section(x: np.ndarray,
                               prob: WingTailSectionProblem | None = None
                               ) -> dict:
    """Full evaluation with breakdown; in-contract failures never raise."""
    from .xfoil_run import run_xfoil_polar

    prob = prob or WingTailSectionProblem()
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

    # ---- section geometry prechecks (airfoil.py contract)
    psi = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, 401)))[1:-1]
    gap = _surface_y(psi, w_u, +0.5 * sec.dz_te) - _surface_y(
        psi, w_l, -0.5 * sec.dz_te)
    if not np.all(gap > 0.0):
        return _fail("self-intersecting / non-positive-thickness geometry")
    tc = cst_thickness(w_u, w_l, dz_te=sec.dz_te)
    coords = cst_coords(w_u, w_l, dz_te=sec.dz_te)

    # ---- live (cached) viscous sweep -> TablePolar
    pol_raw = run_xfoil_polar(coords, sec.re, sec.mach, sec.alphas,
                              timeout_s=sec.timeout_s,
                              cache_dir=sec.cache_dir, n_panel=sec.n_panel)
    if pol_raw.n_converged < 3:
        return _fail("unconvergeable section (< 3 converged XFOIL points) "
                     "= failed design", n_converged=pol_raw.n_converged)
    polar = TablePolar(alpha_deg=pol_raw.alpha_deg, CL=pol_raw.cl,
                       CD=pol_raw.cd, CM=pol_raw.cm, name="cst-live",
                       Re=sec.re)

    # ---- the section's own margins at the design lift (airfoil.py semantics)
    sl = _longest_increasing_run(pol_raw.cl)
    cl_b, cm_b = pol_raw.cl[sl], pol_raw.cm[sl]
    if cl_b.size < 2 or cl_b[-1] < sec.cl_design or cl_b[0] > sec.cl_design:
        return _fail("cl_design not bracketed by the pre-stall monotone "
                     "branch = failed design",
                     n_converged=pol_raw.n_converged)
    cm_at = float(np.interp(sec.cl_design, cl_b, cm_b))
    g_sec = np.array([norm_margin(tc - sec.tc_min, sec.tc_min),
                      norm_margin(sec.cm_max - abs(cm_at), sec.cm_max)])

    # ---- the family's own solve, on THAT section
    evaluate, polar_field, n_own = _FAMILIES[type(prob.wing_tail)]
    kw = {polar_field: polar}
    if polar_field == "polar":
        kw["polar_family"] = None      # a live section replaces the family
    out = evaluate(xw, replace(prob.wing_tail, **kw))
    if not out["feasible"]:
        return out

    g_family = (np.atleast_1d(np.asarray(out["g"], dtype=float))
                if "g" in out else np.empty(0, dtype=float))
    out.update({
        "g": np.concatenate([g_family, g_sec]),
        "g_tc": float(g_sec[0]), "g_cm": float(g_sec[1]),
        "tc": float(tc), "cm_at_cl_design": cm_at,
        "cl_design": float(sec.cl_design),
        "n_converged": pol_raw.n_converged,
        "from_cache": pol_raw.from_cache,
    })
    return out


def fg_wing_tail_section(x: np.ndarray,
                         prob: WingTailSectionProblem | None = None
                         ) -> tuple[float, np.ndarray]:
    """Constrained-harness callable: (score, [family margins…, g_tc, g_cm])."""
    prob = prob or WingTailSectionProblem()
    out = evaluate_wing_tail_section(x, prob)
    if not out["feasible"]:
        return PENALTY, np.full(prob.n_constraints, G_FAIL, dtype=float)
    return float(out["score"]), np.asarray(out["g"], dtype=float)
