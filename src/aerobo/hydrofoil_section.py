"""Hydrofoil (+ tip device, + elevator) with a DESIGNED CST section.

Why this needed new machinery
-----------------------------
Shaping a section for a wing needs cl, cd and cm. Shaping one for a foil
needs Cp_min as well: the constraint that decides a hydrofoil is

    sigma_cav(depth, V) + Cp_min(alpha_eff) >= 0,

and XFOIL reports the minimum surface pressure only through its ``CPMN``
command, one angle at a time — which is why the polar sweep grew a
``with_cpmin`` mode (xfoil_run.py) that drives the alphas individually and
pairs each CPMN output with the converged row it belongs to. The CST section
library carries no Cp_min table at all (polar.py says so and raises), so
before this the water families could only fly the NACA 24XX family members.

With that in place the composition is the same shape as the air one
(winglet_section.py / wingtail_section.py):

    x = [ foil block (its own param_labels, WITHOUT the t/c row) ]
        [ w_upper_0..3, w_lower_0..3 ]

The t/c row leaves the vector because the CST weights ARE the thickness;
``section_polar`` on the foil problems is the switch, and the resulting
thickness is REPORTED (and constrained) rather than designed twice.

Constraints, in this order (feasible >= 0): the foil family's own margins
first — cavitation, the static margin too when an elevator is carried, and
the DRAUGHT margin after those when the family is flown under a cap
(``draught_max_m``, report §17.11) — then the section's two,
``(t/c - tc_min)/tc_min`` and ``(cm_max - |cm|)/cm_max``. The foil's block is
therefore not a fixed width, which is why both the count and the labels are
asked of the foil problem rather than read from a per-type table.

Failure contract: the package's. A section whose sweep converges but carries
no Cp_min pairing is an in-contract FAILURE, not a design flown without a
cavitation check.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

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
from .hydrofoil import (
    HydrofoilProblem,
    HydrofoilWingletProblem,
    evaluate_hydrofoil,
    evaluate_hydrofoil_winglet,
)
from .hydrotail import HydrofoilTailProblem, evaluate_hydrofoil_tail
from .objective import PENALTY, _fail
from .polar import TablePolar

#: foil problem type -> (evaluator, how many margins the family owns, what
#: they are called). The count and the names are the UNCAPPED family's, and
#: they are a fallback only: both are asked of the foil PROBLEM wherever it
#: can answer, because neither is a property of the type once a draught cap
#: can add a margin (see :attr:`HydrofoilSectionProblem.n_constraints`).
_FAMILIES = {
    HydrofoilProblem: (evaluate_hydrofoil, 1, ("cavitation margin",)),
    HydrofoilWingletProblem: (evaluate_hydrofoil_winglet, 1,
                              ("cavitation margin",)),
    HydrofoilTailProblem: (evaluate_hydrofoil_tail, 2,
                           ("cavitation margin", "static margin - SM_min")),
}

#: this problem's OWN margins, appended after the foil family's — in this
#: order, and last, which is what ``api._SIZE_MARGIN_INSERT`` relies on.
SECTION_CONSTRAINT_LABELS = ("t/c margin", "|Cm| margin")


@dataclass
class HydrofoilSectionProblem:
    """A water family + a free-form CST section, live XFOIL with Cp_min."""

    section: AirfoilProblem = field(default_factory=AirfoilProblem)
    foil: Any = field(default_factory=HydrofoilWingletProblem)

    def __post_init__(self):
        if type(self.foil) not in _FAMILIES:
            raise ValueError(
                f"unsupported foil problem {type(self.foil).__name__}; "
                f"choices: {[t.__name__ for t in _FAMILIES]}")
        if self.foil.section_polar is not None:
            raise ValueError("the foil problem must be built WITHOUT a "
                             "section_polar — this problem designs it")

    @property
    def _probe(self):
        """The foil problem as it will be flown: t/c out of the vector."""
        return replace(self.foil, section_polar=_PLACEHOLDER_POLAR)

    @property
    def n_wing(self) -> int:
        return int(self._probe.bounds.shape[0])

    @property
    def n_constraints(self) -> int:
        """The foil's margins plus this problem's own two (t/c, |Cm|).

        The foil's count is ASKED FOR rather than read out of ``_FAMILIES``
        wherever the problem can answer, because that number is no longer a
        property of the TYPE: a foil carrying a draught cap owns one margin
        more than the same class without one. The table stays as the fallback
        for foil problems that expose no ``n_constraints``.

        Getting this from the table alone was a live width mismatch — success
        assembles the vector from ``out["g"]`` (which grows with the cap)
        while failure sizes itself from here (which did not), so a capped
        co-design foil would report 5 margins on success and 4 on failure.
        Found by adversarial review in session 48, when the flag was not yet
        wired into this builder; session 49 wired it, so the path is live.
        """
        n_foil = getattr(self.foil, "n_constraints", None)
        if n_foil is None:
            n_foil = _FAMILIES[type(self.foil)][1]
        return int(n_foil) + len(SECTION_CONSTRAINT_LABELS)

    @property
    def constraint_labels(self) -> tuple:
        """Display names for the margins this problem actually returns.

        Asked of the FOIL for the same reason the count is: with a draught
        cap the foil owns one margin more than its class does uncapped, so
        the names stopped being a property of the type at the same moment
        the width did. ``api.design_report`` prefers this over the static
        ``ProblemSpec.constraint_labels``, which describes the uncapped
        family — without it a capped co-design run would print its draught
        margin under the label "t/c margin", which is exactly the failure
        mode ``hydrofoil.HydrofoilWingletProblem.constraint_labels`` exists
        to prevent one level down.
        """
        foil = getattr(self.foil, "constraint_labels", None)
        if foil is None:
            foil = _FAMILIES[type(self.foil)][2]
        return tuple(foil) + SECTION_CONSTRAINT_LABELS

    @property
    def param_labels(self) -> tuple:
        n = self.section.n_cst
        return tuple(self._probe.param_labels) + tuple(
            [f"w_upper_{i}" for i in range(n)]
            + [f"w_lower_{i}" for i in range(n)])

    @property
    def bounds(self) -> np.ndarray:
        return np.vstack([self._probe.bounds, self.section.bounds])

    @property
    def dim(self) -> int:
        return int(self.bounds.shape[0])

    @property
    def x0(self) -> np.ndarray:
        b = self._probe.bounds
        return np.concatenate([0.5 * (b[:, 0] + b[:, 1]), self.section.w0])

    @property
    def blocks(self) -> tuple:
        nw = self.n_wing
        return (
            {"name": "foil", "indices": tuple(range(nw)), "optimiser": "bo",
             "share": 2.0,
             "cost": "cheap (imaged VLM + cached polar, ~5 ms/eval)",
             "owns_constraints": False},
            {"name": "CST section", "indices": tuple(range(nw, self.dim)),
             "optimiser": "bo", "share": 1.0,
             "cost": "expensive (live XFOIL sweep WITH Cp_min, ~2 s/section)",
             "owns_constraints": True},
        )


class _PlaceholderPolar:
    """Stands in for the designed section while the VECTOR is being described.

    Only ``tc`` is read at that point (the labels and the box do not depend on
    the polar's numbers), so this carries the CST anchor's thickness and
    nothing else — and raises if anything tries to FLY it.
    """

    tc = 0.12
    name = "cst-live (placeholder)"

    #: read by ``hydrofoil.require_cp_min``. That check refuses a section with
    #: no Cp_min table at CONSTRUCTION time, and this object legitimately has
    #: none — it is never flown, and the REAL designed section built in
    #: ``evaluate_hydrofoil_section`` carries ``alpha_cpmin``/``CPMIN`` from a
    #: ``with_cpmin=True`` sweep (and returns an in-contract failure if the
    #: sweep came back without one). Declared as a plain class attribute, not
    #: left to ``__getattr__`` below, which would raise AttributeError on the
    #: probe instead of answering it.
    is_placeholder = True

    def __getattr__(self, item):      # pragma: no cover - guard
        raise AttributeError(
            f"the placeholder section polar has no {item!r}: it exists only "
            f"to describe the design vector, never to be flown")


_PLACEHOLDER_POLAR = _PlaceholderPolar()


def evaluate_hydrofoil_section(x: np.ndarray,
                               prob: HydrofoilSectionProblem | None = None
                               ) -> dict:
    """Full evaluation with breakdown; in-contract failures never raise."""
    from .xfoil_run import run_xfoil_polar

    prob = prob or HydrofoilSectionProblem()
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

    psi = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, 401)))[1:-1]
    gap = _surface_y(psi, w_u, +0.5 * sec.dz_te) - _surface_y(
        psi, w_l, -0.5 * sec.dz_te)
    if not np.all(gap > 0.0):
        return _fail("self-intersecting / non-positive-thickness geometry")
    tc = cst_thickness(w_u, w_l, dz_te=sec.dz_te)
    coords = cst_coords(w_u, w_l, dz_te=sec.dz_te)

    # ---- the sweep, WITH the minimum surface pressure per angle
    pol_raw = run_xfoil_polar(coords, sec.re, sec.mach, sec.alphas,
                              timeout_s=sec.timeout_s,
                              cache_dir=sec.cache_dir, n_panel=sec.n_panel,
                              with_cpmin=True)
    if pol_raw.n_converged < 3:
        return _fail("unconvergeable section (< 3 converged XFOIL points) "
                     "= failed design", n_converged=pol_raw.n_converged)
    if not pol_raw.has_cp_min:
        # a foil flown without a cavitation check would be a fabricated
        # result, so this is a design failure rather than a fallback
        return _fail("no Cp_min pairing for this section (XFOIL run "
                     "truncated) — a cavitation constraint cannot be read",
                     n_converged=pol_raw.n_converged)
    polar = TablePolar(alpha_deg=pol_raw.alpha_deg, CL=pol_raw.cl,
                       CD=pol_raw.cd, CM=pol_raw.cm,
                       alpha_cpmin=pol_raw.alpha_deg, CPMIN=pol_raw.cp_min,
                       name="cst-live", Re=sec.re)
    polar.tc = float(tc)          # what the foil families report as t/c

    sl = _longest_increasing_run(pol_raw.cl)
    cl_b, cm_b = pol_raw.cl[sl], pol_raw.cm[sl]
    if cl_b.size < 2 or cl_b[-1] < sec.cl_design or cl_b[0] > sec.cl_design:
        return _fail("cl_design not bracketed by the pre-stall monotone "
                     "branch = failed design",
                     n_converged=pol_raw.n_converged)
    cm_at = float(np.interp(sec.cl_design, cl_b, cm_b))
    g_sec = np.array([norm_margin(tc - sec.tc_min, sec.tc_min),
                      norm_margin(sec.cm_max - abs(cm_at), sec.cm_max)])

    evaluate = _FAMILIES[type(prob.foil)][0]
    out = evaluate(xw, replace(prob.foil, section_polar=polar))
    if not out["feasible"]:
        return out

    g_family = np.atleast_1d(np.asarray(out["g"], dtype=float))
    out.update({
        "g": np.concatenate([g_family, g_sec]),
        "g_tc": float(g_sec[0]), "g_cm": float(g_sec[1]),
        "tc": float(tc), "cm_at_cl_design": cm_at,
        "cl_design": float(sec.cl_design),
        "n_converged": pol_raw.n_converged,
        "from_cache": pol_raw.from_cache,
    })
    return out


def fg_hydrofoil_section(x: np.ndarray,
                         prob: HydrofoilSectionProblem | None = None
                         ) -> tuple[float, np.ndarray]:
    """Constrained-harness callable: (L/D, [family margins…, g_tc, g_cm])."""
    prob = prob or HydrofoilSectionProblem()
    out = evaluate_hydrofoil_section(x, prob)
    if not out["feasible"]:
        return PENALTY, np.full(prob.n_constraints, G_FAIL, dtype=float)
    return float(out["score"]), np.asarray(out["g"], dtype=float)
