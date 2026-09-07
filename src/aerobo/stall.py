"""Wing CL_max and the spanwise stall origin — the CRITICAL-SECTION method.

Where the wing stalls, and how much lift it makes when it does, in CLOSED
FORM: two lifting-line solves, no root-finding, and ZERO XFOIL calls at
evaluation time (the 2-D cl_max ceiling comes from the pre-computed
wide-alpha polar family, polar.stall_polar_family).

The method says: sweep the root incidence up, and the wing stalls when the
FIRST strip reaches its own 2-D cl_max. That strip is the stall origin; the
wing CL at that instant is CL_max.

PROVENANCE (corrected session 47 -- the previous version of this docstring
cited a paper that does not exist). The lifting-line solve underneath is
Anderson, *Fundamentals of Aerodynamics*, 5th ed., McGraw-Hill 2011, SS5.3
("Prandtl's Classical Lifting-Line Theory"); note that SS5.3 is the
CIRCULATION solve and its published table of contents lists no stall
criterion among its subsections, so do not attribute the criterion to it.
(That is a table-of-contents check, not a reading of the section text.) The criterion itself is classical and is not
claimed as ours. The published lifting-line CL_max method in the same family
is Phillips, W. F. & Alley, N. R., "Predicting Maximum Lift Coefficient for
Twisted Wings Using Lifting-Line Theory", J. Aircraft 44(3), 2007, 898-910,
doi:10.2514/1.25640 -- **verified at bibliographic level only (paywalled);
its twist- and sweep-corrections are NOT implemented here.** The title and
issue number this docstring carried before session 47 matched no publication
at all; the coordinates above are the checked ones. The bad string is
deliberately NOT reproduced here, so that it cannot be copied back out of a
comment -- it is quoted once, in the report's own correction note, and
`tests/test_citation_provenance.py` sweeps the source tree to keep it out.

Why it is closed form
---------------------
The lifting line is AFFINE in root incidence. llt.solve_llt builds

    M @ A = alpha_geo - alpha_L0,        alpha_geo = alpha + twist(y),

with M (llt.fourier_system) depending only on b, c and the section slope —
never on alpha. So A(alpha) = A_twist + alpha A_ones, and every linear
functional of A (CL, Cl_y) is affine in alpha too. Two solves therefore span
the whole alpha family; the same trick already replaces the brentq trim with
one division in adjoint.py (lines 129-134). With

    p(y) = Cl_y at alpha = 0            (the twist-only loading)
    q(y) = Cl_y at alpha = 1 rad  -  p(y)   ( = dCl_y/dalpha, per rad)

strip y reaches its ceiling at root incidence (cl_max_sec(y) - p(y)) / q(y),
and the wing stalls at the smallest of those:

    alpha*   = min over strips with q > 0 of (cl_max_sec(y) - p(y)) / q(y)
    CL_max   = CL_0 + alpha* (CL_1 - CL_0)
    y*       = the argmin strip,   eta_stall = |2 y* / b|   (0 root, 1 tip)

Strips with q <= 0 are excluded: they never approach their ceiling from
below as the root incidence rises, so they cannot be the first to stall.
(q > 0 everywhere on any sane planform; q <= 0 needs pathological twist.)

LIMITATIONS — read before quoting a number
------------------------------------------
1. 2-D cl_max with NO 3-D relief. Each strip is charged its own section
   ceiling as if it were an infinite wing at the same effective angle.
   Spanwise pressure gradients and the crossflow that feeds the separated
   region are not modelled, so a real wing usually carries a little past the
   first section's 2-D limit, and reaching cl_max at ONE station is not the
   same event as a CL break in the wing's lift curve. The predicted CL_max
   is a lower bound in that specific sense — not an error bar.
2. LINEAR section slope carried all the way up to cl_max. The real cl(alpha)
   rounds over well before its peak, so the linear line hits cl_max EARLY
   and alpha* is under-predicted. Measured on this very polar family
   (Re = 1e6): the linear slope crosses cl_max at alpha_eff = 10.65 / 11.64 /
   12.12 / 12.00 deg for t/c = 0.09 / 0.12 / 0.15 / 0.18 while the tabulated
   2-D stall sits at 14.0 / 16.0 / 17.0 / 17.5 deg — a 3.4 to 5.5 deg gap.
   Read alpha_star_deg as "the incidence at which the linearised loading
   first touches the ceiling", never as a stall-warning setting.
3. ONE Reynolds number. cl_max_sec comes from the Re = 1e6, M = 0, Ncrit = 9
   XFOIL family, while Re_mac varies across the design boxes it is used on
   (Tier A is ~1e6 by construction; the free-planform aircraft box spans
   ~0.7-13e6, aircraft.py). cl_max rises with Re, so a large-Re candidate is
   under-credited; Mach and sweep effects on cl_max are absent entirely (the
   LLT itself stays geometrically unswept, llt.swept_section_slope).
4. Grid quantisation. eta_stall can only take one of the N cosine-station
   values, so tip stall reports eta_stall = cos(pi/2N) (0.99966 at N = 60),
   never exactly 1, and root stall reports the innermost station
   (0.02618 at N = 60), never exactly 0.

CONSEQUENCE: the outputs of this module are a RELATIVE RANKING WITHIN ONE
DESIGN BOX (which taper stalls further outboard, how much CL_max washout
buys), never an absolute prediction of a real wing's maximum lift.

This module is a DIAGNOSTIC, deliberately outside the BO objective:
objective.py does not import it, its cost is not charged to an evaluation,
and it RAISES on bad input instead of returning objective.PENALTY.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .llt import TWO_PI, pg_beta, solve_llt, swept_section_slope
from .polar import PolarFamily, stall_points, stall_polar_family

# One-shot loader cache for the wide-alpha family (5 small .pol files) —
# same pattern as tandem._GAUSS_CACHE. Keyed by data_dir so a test can load
# a fixture family without poisoning the default.
_STALL_FAMILY_CACHE: dict[str, PolarFamily] = {}

# Stall-origin degeneracy: strips whose per-strip stall incidence lies within
# this many degrees of alpha* are treated as tied for "first to stall". 0.05
# deg is far below the ~2 deg fidelity of the method (see LIMITATION 2) yet
# far above float roundoff, so it separates a genuine margin from a numerical
# tie. STALL_TIE_FRAC: if more than this fraction of the span is tied, the
# origin is a roundoff artefact (elliptic / near-uniform margin) and
# ``stall_degenerate`` is set.
STALL_TIE_TOL_DEG = 0.05
STALL_TIE_FRAC = 0.10


def default_stall_family(data_dir: str | Path | None = None) -> PolarFamily:
    """Cached polar.stall_polar_family() (the file parse happens once)."""
    key = str(data_dir)
    if key not in _STALL_FAMILY_CACHE:
        _STALL_FAMILY_CACHE[key] = stall_polar_family(data_dir)
    return _STALL_FAMILY_CACHE[key]


def section_clmax(tc, family: PolarFamily | None = None,
                  drop_tol: float = 0.01):
    """2-D cl_max at thickness ratio ``tc`` from the wide-alpha family.

    Interpolates in t/c by the SAME rule as PolarFamily.at: exact at a
    member, linear between the two bracketing members, ValueError outside
    the anchor range (no extrapolation of polar data). ``tc`` may be a
    scalar or an array (per-strip thickness); the return shape follows it.

    Only members whose cl peak is actually RESOLVED anchor the interpolation
    — a censored member's "peak" is just where XFOIL's viscous march gave
    up (polar.StallPoint.censored), i.e. a lower bound, and blending it in
    would silently drag the ceiling down. At Re = 1e6 that excludes the
    t/c = 0.06 member, so the usable range here is [0.09, 0.18] — NARROWER
    than geometry.TC_BOUNDS = (0.08, 0.16), and a t/c = 0.08 wing has no
    cl_max at this Reynolds number rather than a guessed one.
    """
    family = family if family is not None else default_stall_family()
    pts = stall_points(family, drop_tol=drop_tol)
    anchors = np.array(sorted(t for t, sp in pts.items() if not pts[t].censored),
                       dtype=float)
    if anchors.size == 0:
        raise ValueError(
            f"no member of {family.name} resolved a cl peak (all censored): "
            "no section cl_max is available")
    values = np.array([pts[float(t)].cl_max for t in anchors], dtype=float)

    tc_arr = np.asarray(tc, dtype=float)
    lo, hi = float(anchors[0]), float(anchors[-1])
    if np.any(tc_arr < lo - 1e-12) or np.any(tc_arr > hi + 1e-12):
        raise ValueError(
            f"t/c outside the RESOLVED stall-family range [{lo}, {hi}]: "
            f"{np.min(tc_arr):.4f}..{np.max(tc_arr):.4f} (censored members "
            "are excluded as anchors — see section_clmax)")
    out = np.interp(np.clip(tc_arr, lo, hi), anchors, values)
    return float(out) if tc_arr.ndim == 0 else out


@dataclass(frozen=True)
class WingStall:
    """Critical-section stall summary of ONE wing at ONE flow state.

    Headline: ``CL_max`` at root incidence ``alpha_star_deg``, first stall at
    ``eta_stall`` = |2 y*/b| (0 = root, 1 = tip). Everything else is there to
    plot the spanwise margin: ``margin_y`` = cl_max_sec - Cl_y at alpha*, so
    it is >= 0 across the span and EXACTLY 0 at the critical strip, and
    ``alpha_stall_y_deg`` is the root incidence at which each strip would
    reach its own ceiling (+inf where q <= 0, i.e. never).

    ``CL_max`` and ``alpha_star_deg`` are robust; the stall *origin*
    (``eta_stall`` / ``y_stall`` / ``i_stall``) is not always meaningful. When
    many strips reach their ceiling within a hair of alpha* — the elliptic /
    near-uniform-margin case — which strip is nominally "first" is decided by
    roundoff. ``n_near_stall`` counts strips within ``STALL_TIE_TOL_DEG`` of
    alpha*, and ``stall_degenerate`` is True when that count exceeds a small
    fraction of the span: read it before quoting eta_stall as a location.
    """

    CL_max: float
    alpha_star_deg: float
    eta_stall: float
    y_stall: float               # [m], signed spanwise position of the origin
    i_stall: int                 # index of the critical strip
    n_near_stall: int            # strips within STALL_TIE_TOL_DEG of alpha*
    stall_degenerate: bool       # True => eta_stall is a roundoff tie, not a
    #                              location (near-uniform spanwise margin)
    CL_0: float                  # wing CL at zero root incidence (twist only)
    CL_alpha: float              # d CL / d alpha [1/rad]
    S: float
    AR: float
    y: np.ndarray                # (N,) cosine stations [m]
    eta: np.ndarray              # (N,) 2y/b, SIGNED (-1 .. 1)
    c: np.ndarray                # (N,) chord [m]
    cl_max_sec: np.ndarray       # (N,) 2-D section ceiling
    p: np.ndarray                # (N,) Cl_y at alpha = 0
    q: np.ndarray                # (N,) dCl_y/dalpha [1/rad]
    Cl_y_stall: np.ndarray       # (N,) loading at alpha*
    margin_y: np.ndarray         # (N,) cl_max_sec - Cl_y_stall (0 at i_stall)
    alpha_stall_y_deg: np.ndarray  # (N,) per-strip stall incidence [deg]


def wing_clmax(
    b: float,
    c: np.ndarray,
    twist_rad: np.ndarray,
    cl_max_sec: float | np.ndarray,
    a: np.ndarray | float = TWO_PI,
    alpha_L0: np.ndarray | float = 0.0,
) -> WingStall:
    """Critical-section CL_max and stall origin — two LLT solves, no march.

    Kernel, mirroring llt.solve_llt / llt.solve_llt_trim argument order and
    conventions: ``c`` and ``twist_rad`` are sampled at
    ``llt.cosine_stations(N, b)`` (geometry.Wing.sample gives exactly that),
    twist is LOCAL INCIDENCE relative to the root, + = nose-up, so washout is
    a negative tip twist. ``a`` [1/rad] and ``alpha_L0`` [rad] are the section
    slope and zero-lift angle, scalar or per-station.

    ``cl_max_sec`` is the 2-D ceiling: a scalar for a uniform section, or an
    (N,) array for a spanwise-varying one (e.g. section_clmax(tc_y)).

    Raises ValueError if the ceiling is non-positive/non-finite or if no
    strip gains lift with root incidence (q <= 0 everywhere). Linear-algebra
    failures propagate from solve_llt — this is a diagnostic, not the
    penalty-contracted objective.
    """
    c = np.asarray(c, dtype=float)
    N = c.size
    twist_rad = np.broadcast_to(np.asarray(twist_rad, dtype=float), (N,))
    cl_max_y = np.array(np.broadcast_to(np.asarray(cl_max_sec, dtype=float), (N,)))
    if not np.all(np.isfinite(cl_max_y)) or np.any(cl_max_y <= 0.0):
        raise ValueError("cl_max_sec must be finite and positive at every strip")

    # The two solves that span the affine family (module docstring): root
    # incidence 0 and 1 rad on the SAME twist distribution and the SAME M.
    res0 = solve_llt(b, c, twist_rad, a, alpha_L0)
    res1 = solve_llt(b, c, twist_rad + 1.0, a, alpha_L0)
    p = res0.Cl_y
    q = res1.Cl_y - p

    rising = q > 0.0
    if not np.any(rising):
        raise ValueError(
            "no strip gains lift with root incidence (dCl_y/dalpha <= 0 "
            "everywhere): the critical-section sweep has no stall point")
    # +inf on the non-rising strips keeps them out of the argmin without
    # perturbing the finite entries (the division is masked, not clipped).
    alpha_stall_y = np.where(rising, (cl_max_y - p) / np.where(rising, q, 1.0),
                             np.inf)
    i = int(np.argmin(alpha_stall_y))
    alpha_star = float(alpha_stall_y[i])

    # Degeneracy: how many strips reach their ceiling within a whisker of
    # alpha*. When that is a large fraction of the span (near-uniform margin,
    # e.g. elliptic loading against a constant ceiling) the argmin above is
    # decided by roundoff and eta_stall is not a physical location.
    n_near = int(np.count_nonzero(
        np.abs(alpha_stall_y - alpha_star) <= np.deg2rad(STALL_TIE_TOL_DEG)))
    degenerate = bool(n_near > max(1, int(STALL_TIE_FRAC * N)))

    CL_alpha = res1.CL - res0.CL
    Cl_y_stall = p + alpha_star * q

    return WingStall(
        CL_max=float(res0.CL + alpha_star * CL_alpha),
        alpha_star_deg=float(np.rad2deg(alpha_star)),
        eta_stall=float(abs(2.0 * res0.y[i] / b)),
        y_stall=float(res0.y[i]),
        i_stall=i,
        n_near_stall=n_near,
        stall_degenerate=degenerate,
        CL_0=float(res0.CL),
        CL_alpha=float(CL_alpha),
        S=float(res0.S),
        AR=float(res0.AR),
        y=res0.y,
        eta=2.0 * res0.y / b,
        c=c,
        cl_max_sec=cl_max_y,
        p=p,
        q=q,
        Cl_y_stall=Cl_y_stall,
        margin_y=cl_max_y - Cl_y_stall,
        alpha_stall_y_deg=np.rad2deg(alpha_stall_y),
    )


def evaluate_stall(
    c: np.ndarray,
    twist_rad: np.ndarray,
    prob: Any,
    cl_max_sec: float | np.ndarray | None = None,
    tc: float | np.ndarray | None = None,
    sweep_deg: float = 0.0,
    polar: Any = None,
    stall_family: PolarFamily | None = None,
) -> WingStall:
    """Problem-level entry point: the objective.evaluate_geometry signature.

    Takes the wing exactly the way the rest of the codebase passes it —
    ``c`` and ``twist_rad`` sampled at the prob.N cosine stations plus a
    Problem carrying b / polar / mach — so BOTH callers work unchanged:

        Tier A (objective.evaluate):
            _, c, twist = wing.sample(prob.N)
            evaluate_stall(c, twist, prob, tc=wing.tc, sweep_deg=wing.sweep_deg)
        free planform (aircraft.evaluate_aircraft), where b and S are design
        variables and ``aero_prob`` is the locally built trim Problem:
            evaluate_stall(c, twist_rad, aero_prob, tc=tc)

    Exactly one of ``cl_max_sec`` (an explicit ceiling — scalar for a uniform
    section, (N,) for a varying one) and ``tc`` (look the ceiling up in the
    wide-alpha family, section_clmax) must be given.

    The section SLOPE and zero-lift angle come from the polar the wing is
    being flown on (``polar``, else prob.polar) with the same reduced-order
    sweep/compressibility correction the objective applies
    (llt.swept_section_slope), while the cl_max CEILING comes from the stall
    family. For the NACA 24XX sections those are the same XFOIL solve at the
    same Re/M/Ncrit — the cruise and stall tables agree to XFOIL's own print
    resolution on the overlapping alphas (tests/test_stall_polar.py), so
    a_lin differs by < 1e-4 between them.
    """
    if (cl_max_sec is None) == (tc is None):
        raise ValueError("give exactly one of cl_max_sec (explicit ceiling) "
                         "or tc (look it up in the stall polar family)")

    # Phase-5 physics skins change the loading evaluate_geometry sees, but
    # this module solves the PLAIN free-air lifting line. Returning free-air
    # stall numbers for a Problem that carries a skin would be a silent
    # mismatch between the L/D and the CL_max reported for the same design —
    # exactly the failure this module's raise-don't-penalise contract exists
    # to prevent. Refuse instead, until each skin's effect on the ceiling is
    # derived rather than assumed.
    for flag, why in (("ground_h_m", "ground effect changes the induced "
                                     "field and so the spanwise loading"),
                      ("slipstream", "a non-uniform slipstream changes both "
                                     "the local q and the local incidence")):
        if getattr(prob, flag, None) is not None:
            raise NotImplementedError(
                f"evaluate_stall does not model prob.{flag}: {why}. Compute "
                "the stall margin on a Problem with the skin OFF, or extend "
                "wing_clmax to carry the skin's loading.")

    if len(np.atleast_1d(c)) != int(getattr(prob, "N", len(np.atleast_1d(c)))):
        raise ValueError(
            f"chord sampled at {len(np.atleast_1d(c))} stations but prob.N is "
            f"{prob.N}; b comes from the Problem and the grid from c, so a "
            "mismatch silently solves a different wing")

    pol = polar if polar is not None else prob.polar
    mach = float(getattr(prob, "mach", 0.0))
    pg_beta(mach)          # same 0 <= M < 0.7 gate as evaluate_geometry
    a_sec = float(swept_section_slope(pol.a_lin, sweep_deg, mach=mach))
    if cl_max_sec is None:
        cl_max_sec = section_clmax(tc, family=stall_family)
    return wing_clmax(prob.b, c, twist_rad, cl_max_sec,
                      a=a_sec, alpha_L0=pol.alpha_L0)
