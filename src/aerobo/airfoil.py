"""CST (Kulfan) airfoil parametrisation + NACA 4-digit generator for Tier B.

CST — Class/Shape function Transformation [Kulfan 2008]
-------------------------------------------------------
A surface (upper or lower, each with its OWN weight vector) is

    y(psi) = C(psi) * S(psi) + psi * dz_te / 2        (upper; lower: - dz_te/2)

with psi = x/c in [0, 1] and

    class function   C(psi) = psi^N1 * (1 - psi)^N2,   N1 = 0.5, N2 = 1.0
    shape function   S(psi) = sum_{i=0}^{n} w_i B_i(psi)
    Bernstein basis  B_i(psi) = C(n, i) psi^i (1 - psi)^(n-i),  n = n_w - 1.

N1 = 0.5 gives the round-nose sqrt(psi) leading edge (finite LE radius,
r_LE/c = w_0^2 / 2 in Kulfan's derivation — ``cst_le_radius``, with the
buildable floor ``R_LE_MIN``); N2 = 1.0 gives a sharp
(or, with dz_te > 0, finite-gap) trailing edge with finite boat-tail
angle. The Bernstein weights are the SAME (unnegated) convention on both
surfaces — Kulfan's Eq. (6)-(9) — so a conventional airfoil has
w_upper > 0 and w_lower < 0. The TE-gap term psi * dz_te/2 is Kulfan's
zeta_T ridge line: + on the upper surface, - on the lower.

Because y is LINEAR in the weights, fitting CST to a given airfoil is a
plain linear least-squares problem (``fit_cst``) — no iteration, exact
global optimum in the L2 sense.

NACA 4-digit [Abbott & von Doenhoff 1959, §6]
---------------------------------------------
Code "MPXX": max camber m = M/100 at position p = P/10, thickness
t = XX/100. Half-thickness distribution (CLOSED-TE variant: last
coefficient -0.1036 instead of the open-TE -0.1015, which zeroes y_t at
x = 1 so the loop closes exactly):

    y_t = 5 t (0.2969 sqrt(x) - 0.1260 x - 0.3516 x^2
               + 0.2843 x^3 - 0.1036 x^4)

Camber line (parabolic arcs, slope-continuous at x = p):

    y_c   = m/p^2 (2 p x - x^2)                      x <  p
          = m/(1-p)^2 ((1 - 2p) + 2 p x - x^2)       x >= p

and the thickness is applied PERPENDICULAR to the camber line
(theta = atan dy_c/dx):

    x_u = x - y_t sin(theta),   y_u = y_c + y_t cos(theta)
    x_l = x + y_t sin(theta),   y_l = y_c - y_t cos(theta).

Loop convention (both generators, frozen inter-agent contract)
--------------------------------------------------------------
Single closed loop in XFOIL LOAD order: TE-upper -> LE -> TE-lower,
cosine-clustered abscissae (dense at LE *and* TE), ``n_pts`` total
points, LE appears exactly once (no duplicate point).

References
----------
Kulfan, B. M., "Universal Parametric Geometry Representation Method,"
    Journal of Aircraft, Vol. 45, No. 1, 2008, pp. 142-158.
    doi:10.2514/1.29958
Abbott, I. H., and von Doenhoff, A. E., "Theory of Wing Sections,"
    Dover, New York, 1959, §6 (families of wing sections).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import comb
from pathlib import Path

import numpy as np

# Class-function exponents: round-nose, sharp-TE airfoil class (Kulfan 2008).
N1 = 0.5
N2 = 1.0


# ---------------------------------------------------------------- CST core


def class_fn(psi: np.ndarray) -> np.ndarray:
    """Kulfan class function C(psi) = psi^N1 (1-psi)^N2 (N1=0.5, N2=1.0)."""
    psi = np.asarray(psi, dtype=float)
    return psi**N1 * (1.0 - psi) ** N2


def bernstein_matrix(psi: np.ndarray, n_w: int) -> np.ndarray:
    """(len(psi), n_w) Bernstein design matrix, order n = n_w - 1.

    B[j, i] = C(n, i) psi_j^i (1 - psi_j)^(n - i).  Rows sum to 1
    (partition of unity), which is what makes the weights an intuitive,
    well-scaled BO design vector.
    """
    if n_w < 1:
        raise ValueError(f"need at least one Bernstein weight (got n_w={n_w})")
    psi = np.asarray(psi, dtype=float)
    n = n_w - 1
    return np.column_stack(
        [comb(n, i) * psi**i * (1.0 - psi) ** (n - i) for i in range(n_w)]
    )


def _surface_y(psi: np.ndarray, w: np.ndarray, dz_half: float) -> np.ndarray:
    """One CST surface: y = C(psi) [B(psi) w] + psi * dz_half."""
    w = np.atleast_1d(np.asarray(w, dtype=float))
    return class_fn(psi) * (bernstein_matrix(psi, w.size) @ w) + psi * dz_half


def _loop_psi(n_pts: int) -> tuple[np.ndarray, np.ndarray]:
    """Cosine-clustered psi arrays for the closed loop, LE shared once.

    Upper: psi = (1 + cos theta)/2, theta in [0, pi] -> 1 .. 0 (TE -> LE);
    lower: mirror image 0 .. 1 with the LE point dropped, so the two
    pieces concatenate to exactly ``n_pts`` points with a single LE.
    """
    if n_pts < 5:
        raise ValueError(f"n_pts={n_pts} too small for a closed loop")
    n_u = (n_pts + 1) // 2              # TE-upper .. LE inclusive
    n_l = n_pts - n_u + 1               # LE .. TE-lower inclusive (LE dropped)
    psi_u = 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, n_u)))
    psi_l = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, n_l)))
    return psi_u, psi_l[1:]


def cst_coords(
    w_upper: np.ndarray,
    w_lower: np.ndarray,
    n_pts: int = 160,
    dz_te: float = 0.0,
) -> np.ndarray:
    """CST airfoil as one closed (n_pts, 2) loop in XFOIL LOAD order.

    TE-upper (1, +dz_te/2) -> LE (0, 0) -> TE-lower (1, -dz_te/2), cosine
    x-spacing on both surfaces. ``w_lower`` is the lower surface's own
    weight vector in the unnegated Kulfan convention (negative entries
    for a conventional airfoil); the vectors may have different lengths.
    """
    psi_u, psi_l = _loop_psi(n_pts)
    y_u = _surface_y(psi_u, w_upper, +0.5 * dz_te)
    y_l = _surface_y(psi_l, w_lower, -0.5 * dz_te)
    return np.column_stack(
        [np.concatenate([psi_u, psi_l]), np.concatenate([y_u, y_l])]
    )


def cst_thickness(
    w_upper: np.ndarray, w_lower: np.ndarray, dz_te: float = 0.0
) -> float:
    """Max thickness-to-chord: max over psi of y_upper(psi) - y_lower(psi).

    Dense cosine-clustered sampling (2001 stations) — thickness is a
    smooth C^1 function of psi away from the LE, so grid-max error is
    O(1e-7) t/c, far below the geometric tolerances in play. Used by the
    Tier B structural constraint g = t/c - t/c_min.
    """
    psi = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, 2001)))
    gap = _surface_y(psi, w_upper, +0.5 * dz_te) - _surface_y(
        psi, w_lower, -0.5 * dz_te
    )
    return float(np.max(gap))


# Buildable-nose floor on r_LE/c, derived from the NACA 4-digit nose radius
#
#     r_LE/c = 1.1019 t^2          [Abbott & von Doenhoff 1959, §6]
#
# (the sqrt(x) term of the half-thickness polynomial is 5 t * 0.2969, so
# y -> 1.4845 t sqrt(x) at the nose; matching y = sqrt(2 r x) gives
# 2 r = (1.4845 t)^2 = 2.2038 t^2). Evaluated at the Tier B thickness floor
# tc_min = 0.10 (AirfoilProblem.tc_min) that is r_LE/c = 0.011019 — the nose
# of a NACA 0010, the bluntness of the thinnest section the structural
# constraint admits. The 0.5 keeps the floor deliberately conservative: it
# flags only a nose sharper than HALF that, i.e. a genuinely knife-edge
# leading edge that no metal or composite wing skin could be laid up over
# and that XFOIL's inviscid panel + integral-BL model resolves poorly.
R_LE_MIN = 0.5 * 1.1019 * 0.10**2     # = 0.0055095


def cst_le_radius(w: np.ndarray) -> float:
    """Leading-edge radius r_LE/c of ONE CST surface: r_LE/c = w_0^2 / 2.

    Closed form for the N1 = 0.5 class function (module docstring, Kulfan
    2008). At the nose C(psi) = psi^0.5 (1-psi) -> sqrt(psi) and every
    Bernstein basis function except B_0 vanishes at psi = 0, so the shape
    function collapses onto its first weight:

        y(psi) -> w_0 sqrt(psi)                  (psi -> 0).

    A circle of radius r tangent to the chord line at the origin is
    x = y^2 / (2 r), i.e. y = sqrt(2 r psi) + O(psi^{3/2}); matching the
    sqrt(psi) coefficients gives 2 r = w_0^2, hence r_LE/c = w_0^2 / 2.
    ONLY the first weight sets the nose — w_1.. shape the surface aft of
    it and drop out of the limit.

    Sign-blind (w_0 is squared), so the same call returns the lower
    surface's radius from its own (negative-class) weight vector. A
    cambered section therefore carries two different per-surface values.
    Analytically both surfaces of a NACA 4-digit share ONE nose radius —
    the camber line is O(x) at the nose while the thickness form is
    O(sqrt(x)), so camber cannot change the leading-order radius — and the
    split here is an artefact of the least-squares CST fit. Empirically the
    geometric mean sqrt(r_upper r_lower) recovers the analytic NACA value
    to ~0.5 % on the 4-weight NACA 2412 fit (tests/test_airfoil.py); treat
    that as a property of the fit, not as a geometric identity.

    Compare against ``R_LE_MIN`` for the buildable-nose screen.
    """
    w = np.atleast_1d(np.asarray(w, dtype=float))
    if w.size < 1:
        raise ValueError("need at least one Bernstein weight to set the nose")
    return 0.5 * float(w[0]) ** 2


def fit_cst(
    x_coords: np.ndarray,
    y_upper: np.ndarray,
    y_lower: np.ndarray,
    n_cst: int = 4,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Least-squares CST fit of an airfoil given on a common x grid.

    Because the CST surface is linear in the Bernstein weights,

        y -/+ psi dz_te/2 = [C(psi) B(psi)] w,

    each surface is one (unweighted) linear least-squares solve — no
    iteration, no initial guess. ``n_cst`` = number of weights per
    surface (Bernstein order n_cst - 1). dz_te is read off as the
    ordinate gap at the aft-most station (assumed to be the TE, x ~ 1).

    Returns (w_upper, w_lower, dz_te) reproducible via ``cst_coords``.
    """
    x = np.asarray(x_coords, dtype=float)
    yu = np.asarray(y_upper, dtype=float)
    yl = np.asarray(y_lower, dtype=float)
    if not (x.shape == yu.shape == yl.shape):
        raise ValueError(
            f"shape mismatch: x{x.shape}, y_upper{yu.shape}, y_lower{yl.shape}"
        )
    if x.min() < -1e-9 or x.max() > 1.0 + 1e-9:
        raise ValueError("x_coords must lie in [0, 1] (unit chord)")

    i_te = int(np.argmax(x))
    dz_te = float(yu[i_te] - yl[i_te])

    A = class_fn(x)[:, None] * bernstein_matrix(x, n_cst)
    w_u, *_ = np.linalg.lstsq(A, yu - 0.5 * dz_te * x, rcond=None)
    w_l, *_ = np.linalg.lstsq(A, yl + 0.5 * dz_te * x, rcond=None)
    return w_u, w_l, dz_te


# ---------------------------------------------------------------- NACA 4-digit


def naca4_coords(code: str, n_pts: int = 160) -> np.ndarray:
    """Analytic NACA 4-digit airfoil, same closed-loop order as cst_coords.

    Closed-TE half-thickness polynomial (-0.1036 last coefficient) and
    exact perpendicular thickness application about the two-parabola
    camber line [Abbott & von Doenhoff 1959, §6]. Symmetric sections
    ("00XX") have a degenerate camber line handled explicitly.

    Note the exact construction offsets the abscissae (x_u = x - y_t
    sin theta), so output x's are cosine-clustered only to O(y_t theta).
    """
    code = str(code).strip()
    if len(code) != 4 or not code.isdigit():
        raise ValueError(f"NACA 4-digit code must be 4 digits, got {code!r}")
    m = int(code[0]) / 100.0            # max camber
    p = int(code[1]) / 10.0             # max-camber position
    t = int(code[2:]) / 100.0           # thickness/chord

    def half_thickness(x: np.ndarray) -> np.ndarray:
        return 5.0 * t * (
            0.2969 * np.sqrt(x)
            - 0.1260 * x
            - 0.3516 * x**2
            + 0.2843 * x**3
            - 0.1036 * x**4              # closed TE (open-TE value: -0.1015)
        )

    def camber(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if m == 0.0 or p == 0.0:
            return np.zeros_like(x), np.zeros_like(x)
        fwd = x < p
        yc = np.where(
            fwd,
            m / p**2 * (2.0 * p * x - x**2),
            m / (1.0 - p) ** 2 * ((1.0 - 2.0 * p) + 2.0 * p * x - x**2),
        )
        dyc = np.where(
            fwd,
            2.0 * m / p**2 * (p - x),
            2.0 * m / (1.0 - p) ** 2 * (p - x),
        )
        return yc, dyc

    def surface(x: np.ndarray, sign: float) -> tuple[np.ndarray, np.ndarray]:
        yt = half_thickness(x)
        yc, dyc = camber(x)
        theta = np.arctan(dyc)
        return (x - sign * yt * np.sin(theta),
                yc + sign * yt * np.cos(theta))

    psi_u, psi_l = _loop_psi(n_pts)
    x_u, y_u = surface(psi_u, +1.0)
    x_l, y_l = surface(psi_l, -1.0)
    return np.column_stack(
        [np.concatenate([x_u, x_l]), np.concatenate([y_u, y_l])]
    )


# ---------------------------------------------------------------- Tier B problem
#
# Section-level drag minimisation on the CST design vector, evaluated with
# REAL viscous XFOIL polars (xfoil_run.run_xfoil_polar, cached).

PENALTY = -100.0     # objective.py contract (solver/tool failure only)
G_FAIL = -1.0        # finite infeasible margin reported on solver failure

# Design-box constants (derivation recorded in the AirfoilProblem docstring):
W_UPPER_FLOOR = 0.05   # upper-surface weights stay positive-class
W_LOWER_CAP = -0.02    # lower-surface weights stay negative-class
W_HALF_THIN = 0.35     # half-width in the thickness-REDUCING directions
W_HALF_THICK = 0.15    # half-width in the thickness-INCREASING directions

def norm_margin(slack: float, limit: float) -> float:
    """A constraint margin normalised by its own limit, safely.

    Every gate in this package reports its slack as a FRACTION of the limit
    it is measured against ((t/c - tc_min)/tc_min, (cm_max - |cm|)/cm_max,
    ...) so that margins of different units are O(1) and comparable, and so
    the constrained harness's feasibility test (all g >= 0) is
    scale-free.

    A limit of ZERO is a legitimate setting, not an error: ``tc_min = 0`` is
    how a caller says "no thickness floor" and ``cm_max = 0`` says "no
    pitching moment at all". There is no scale to divide by then, so the RAW
    slack is returned — same sign, same feasible set, no ZeroDivisionError.
    A non-finite limit (the "gate off" idiom for an upper cap, cm_max = 1e9)
    divides normally, which is what makes an off gate read as a margin of
    ~1. For every positive, finite limit this is exactly the old expression,
    bit-for-bit.
    """
    lim = float(limit)
    if lim > 0.0 and np.isfinite(lim):
        return float(slack) / lim
    return float(slack)


_ANCHOR_CACHE: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}


def cst_anchor_from_coords(
    coords: np.ndarray, n_cst: int = 4
) -> tuple[np.ndarray, np.ndarray]:
    """CST fit (w_upper, w_lower) of an arbitrary closed coordinate loop.

    The cst_anchor recipe generalised to any loop in XFOIL LOAD order
    (TE-upper -> LE -> TE-lower, unit chord): split at the min-x point,
    each branch sorted by x (cambered noses are slightly non-monotonic in
    x, so the sort matters), interpolated onto a common 201-point cosine
    grid, then one linear least-squares solve per surface (fit_cst). Any
    fitted TE gap is dropped — the Tier B problem fixes dz_te = 0, so an
    open-TE database section is anchored by its sharp-TE CST projection.
    """
    coords = np.asarray(coords, dtype=float)
    x, y = coords[:, 0], coords[:, 1]
    i_le = int(np.argmin(x))
    if i_le == 0 or i_le == x.size - 1:
        raise ValueError("LE (min-x point) at loop end: not a closed "
                         "TE-upper -> LE -> TE-lower loop")
    xu, yu = x[: i_le + 1][::-1], y[: i_le + 1][::-1]
    xl, yl = x[i_le:], y[i_le:]
    su, sl = np.argsort(xu), np.argsort(xl)
    xg = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, 201)))
    w_u, w_l, _dz = fit_cst(
        xg,
        np.interp(xg, xu[su], yu[su]),
        np.interp(xg, xl[sl], yl[sl]),
        n_cst=n_cst,
    )
    return w_u, w_l


def cst_anchor(code: str = "2412", n_cst: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """CST fit (w_upper, w_lower) of a NACA 4-digit section — the box anchor.

    Dense analytic loop (naca4_coords, 4000 pts) through the
    cst_anchor_from_coords recipe. The ~1e-17 fitted TE gap of the
    closed-TE generator is dropped (the Tier B problem fixes dz_te = 0).
    Cached per (code, n_cst) — deterministic.

    For "2412" / n_cst = 4: w_u ~ [0.191, 0.221, 0.190, 0.216],
    w_l ~ [-0.150, -0.083, -0.093, -0.072], t/c = 0.1199 (cst_thickness).
    """
    key = (str(code), int(n_cst))
    if key not in _ANCHOR_CACHE:
        _ANCHOR_CACHE[key] = cst_anchor_from_coords(
            naca4_coords(code, n_pts=4000), n_cst=n_cst)
    w_u, w_l = _ANCHOR_CACHE[key]
    return w_u.copy(), w_l.copy()


def anchor_box(
    w_upper: np.ndarray, w_lower: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """(lo, hi) design box around a CST anchor — the recorded 2026-07-12
    recipe (AirfoilProblem docstring): +/- W_HALF_THIN in the
    thickness-REDUCING directions (upper down, lower up), W_HALF_THICK in
    the thickness-INCREASING ones, with the positive-class floor / negative-
    class cap that PROVE no in-box self-intersection."""
    w_upper = np.asarray(w_upper, dtype=float)
    w_lower = np.asarray(w_lower, dtype=float)
    lo = np.concatenate([
        np.maximum(w_upper - W_HALF_THIN, W_UPPER_FLOOR),
        w_lower - W_HALF_THICK,
    ])
    hi = np.concatenate([
        w_upper + W_HALF_THICK,
        np.minimum(w_lower + W_HALF_THIN, W_LOWER_CAP),
    ])
    return lo, hi


#: the CRUISE polar sweep, as ``(min, max, step)`` in degrees. Stated here
#: rather than left buried in a ``default_factory`` because it is a CEILING on
#: the design lift this family can be ASKED for, and one that reads as physics
#: when it is not: a section whose branch stops at cl 1.2 because the sweep
#: stopped at 10 deg is refused in exactly the same words as one that genuinely
#: cannot reach the lift. Measured 2026-08-09 at re 8.5e5: cl_design 1.0 is
#: 5/14 feasible, 1.3 is 0/14 and >= 1.6 is 0/14 for every optimiser — the
#: cliff is this number, not the aerodynamics. It bites the car rear wing
#: hardest, whose reference lift is 1.0 where every other family's is 0.5.
#:
#: WHY THE DEFAULT DOES NOT MOVE. ``xfoil_run.cache_key`` hashes the sweep, so
#: widening it here would invalidate every cached polar in the repo and re-run
#: every frozen study underneath its own result files. The range is therefore
#: a PARAMETER (``alpha_sweep``, reached through the ``airfoil_sweep_alpha_*``
#: flags) with the published default left alone — the same treatment
#: ``draught_max_m`` and ``bo_refusal`` get, and the opposite of a calibration
#: that has quietly become a gate.
#:
#: Separate from ``airfoil_select.STALL_ALPHAS`` (the wide 0-20 deg sweep that
#: ``clmax``/``astall`` need) and from ``section_wing``'s ``alpha_max_deg``,
#: which is the WING's incidence cap and moves nothing here.
ALPHA_SWEEP_DEG = (-4.0, 10.0, 0.5)


def alpha_sweep(alpha_min_deg: float | None = None,
                alpha_max_deg: float | None = None,
                step_deg: float | None = None) -> np.ndarray:
    """The XFOIL cruise sweep [deg]; each end defaults to :data:`ALPHA_SWEEP_DEG`.

    Giving ONE end is legal — "sweep further up" is a complete sentence, and
    the other end keeps the published value (the contract
    the car's span band is the design box's own ``b_m`` row,
    ``api._car_size_band_kwargs``).

    Bit-for-bit note: called with no arguments this reproduces the historical
    ``np.arange(-4.0, 10.5, 0.5)`` element for element, which is what keeps
    every cached polar's ``xfoil_run.cache_key`` valid.
    """
    lo = ALPHA_SWEEP_DEG[0] if alpha_min_deg is None else float(alpha_min_deg)
    hi = ALPHA_SWEEP_DEG[1] if alpha_max_deg is None else float(alpha_max_deg)
    step = ALPHA_SWEEP_DEG[2] if step_deg is None else float(step_deg)
    if not step > 0.0:
        raise ValueError(f"alpha sweep step must be > 0 deg, got {step}")
    if not hi > lo:
        raise ValueError(
            f"alpha sweep must satisfy min < max, got ({lo}, {hi}) deg")
    # built by COUNT, not by a padded ``arange`` stop: a stated ceiling must
    # never be exceeded, and every rounded stop (``hi + step``, ``hi + step/2``)
    # overshoots for some off-grid ``hi`` — 10.3 deg would be swept to 10.5.
    # Bit-for-bit with ``np.arange(lo, hi + step, step)`` on the grid, which is
    # what the default reproduces (np.arange accumulates ``lo + i*step`` too).
    n = int(np.floor((hi - lo) / step + 1e-9)) + 1
    return lo + step * np.arange(n)


#: how far the sweep is widened when the design lift needs it, and the lift
#: below which it is left alone. Both MEASURED, not chosen — 12 scrambled Sobol
#: sections over the 8-D box at re 8.5e5, each polared on the published sweep
#: and on 0-20 deg, then re-read at each design lift
#: (``results/alpha_sweep_probe.json``):
#:
#:     cl_design   published   widened   rescued by widening
#:      0.5-0.9      10/12       9/12          0
#:      1.0           9/12       9/12          0
#:      1.1           5/12       9/12          4
#:      1.2           2/12       6/12          4
#:      1.3           1/12       5/12          4
#:      1.5           0/12       1/12          1
#:
#: Two readings, and both matter. Above cl 1.0 the published sweep is what
#: refuses the design — 4 of 12 sections in the box are rescued by nothing but
#: more alpha. At and below cl 1.0 widening rescues NOTHING, and it is not
#: free either: one section in twelve is LOST, because extra post-stall rows
#: can shorten the monotone branch ``_longest_increasing_run`` finds. So the
#: threshold is a real edge in the data, not a round number, and the rule only
#: ever widens above it.
ALPHA_SWEEP_SAFE_CL = 1.0
#: ...and where widening stops: past ~20 deg XFOIL's integral-BL model is
#: describing a separated section it cannot be trusted on, so a lift the sweep
#: still cannot reach by then is the SECTION's ceiling and is reported as one.
ALPHA_SWEEP_CEILING_DEG = 20.0


def sweep_ceiling_for(cl_design: float) -> float:
    """The sweep top end a design lift needs [deg] — the published one below
    :data:`ALPHA_SWEEP_SAFE_CL`, the reliability ceiling above it.

    Deliberately a STEP and not a slope. Predicting the incidence at which an
    unknown section reaches a given lift would need that section's lift curve,
    which is what the sweep is being run to find out; a slope estimate that is
    5 % optimistic refuses the design exactly as the old ceiling did. The cost
    of overshooting is a few more cached XFOIL alphas — seconds — against a
    whole search that returns nothing, so the asymmetry decides it.
    """
    cl = float(cl_design)
    if not cl > ALPHA_SWEEP_SAFE_CL:
        return float(ALPHA_SWEEP_DEG[1])
    return float(ALPHA_SWEEP_CEILING_DEG)


@dataclass
class AirfoilProblem:
    """Tier B: 2-D section drag minimisation at fixed design lift, real XFOIL.

    Design vector (d = 2 n_cst = 8, unnegated Kulfan convention):

        x = [w_upper_0 .. w_upper_3,  w_lower_0 .. w_lower_3]

    Objective (MAXIMISE): f = -cd(cl_design) — the viscous profile-drag
    coefficient interpolated at the design lift from a converged XFOIL polar
    sweep (Drela 1989). The sign flip makes the constrained harness's
    maximise convention minimise drag; report drag in counts (1e4 cd).
    Comparing candidates AT THE SAME LIFT (cl-interpolation, not fixed
    alpha) is what makes cd differences meaningful — the 2-D analogue of
    the Tier A trim-mode decision.

    Constraints (signed margins, both normalised O(1), feasible iff >= 0).
    Each is divided by its own limit through :func:`norm_margin`, which
    returns the RAW slack when that limit is zero — ``tc_min = 0`` (no
    thickness floor) and ``cm_max = 0`` (no pitching moment at all) are
    settings, not errors:

        g0 = (t/c - tc_min) / tc_min
             structural-depth proxy: the wing box needs section depth for
             spar caps and fuel volume; without it the drag optimum runs to
             a vanishing-thickness plate.
        g1 = (cm_max - |cm(cl_design)|) / cm_max
             trim-drag proxy: the tail must react the wing pitching moment
             and trim drag grows with |Cm| (Raymer 2018, trim-drag buildup);
             an uncapped cd optimum drifts to strongly aft-loaded
             high-|Cm| shapes. cm_max = 0.08 |Cm| cap is typical of
             moderately cambered GA sections (NACA 2412 at cl = 0.5:
             |Cm| ~ 0.05).
        g2 = (min(r_LE_u, r_LE_l) - r_le_min) / r_le_min   [only if
             r_le_min is not None] buildable-nose proxy: the closed-form
             per-surface leading-edge radius r_LE/c = w_0^2 / 2 (Kulfan
             N1 = 0.5, ``cst_le_radius``) of the BLUNTER-limiting surface
             must clear the floor. The unconstrained cd optimum drives one
             surface's nose to a knife edge (r_LE/c ~ 2e-4 at the box cap,
             ~27x below R_LE_MIN — le_radius_audit) that no wing skin could
             be laid up over and that XFOIL's inviscid-panel + integral-BL
             model resolves poorly. None (default) => the m = 2 problem,
             bit-for-bit the frozen 2412 study.

    Design box (recorded derivation, 2026-07-12)
    --------------------------------------------
    Anchor w0 = cst_anchor("2412"). The spec's symmetric w0 +/- 0.35 box
    FAILED the pre-registered validity check — 200 scrambled Sobol points,
    valid = non-self-intersecting AND t/c in (0.06, 0.20), pass bar >= 80%:
    symmetric +/- 0.35 passed 38.5% (123/200 points thicker than t/c 0.20;
    +/- 0.30 -> 57.5%, +/- 0.25 -> 74.5% — the thickness-increasing corners
    dominate the box). Adopted box, same floors, asymmetric half-widths:

        upper surface:  [max(w0 - 0.35, +0.05),   w0 + 0.15]
        lower surface:  [w0 - 0.15,   min(w0 + 0.35, -0.02)]

    i.e. +/- 0.35 is kept in the thickness-REDUCING directions (upper down,
    lower up), where the tc_min constraint must be reachable and active,
    and 0.15 in the thickness-INCREASING directions. Validity: 200/200
    (100%), sampled t/c in [0.070, 0.190], median 0.132. The floor/cap keep
    each surface on its own side of the chord line, which PROVES no in-box
    self-intersection: the Bernstein basis is positive on (0, 1), so
    w_upper >= +0.05 and w_lower <= -0.02 give y_u > 0 > y_l behind the LE.

    Failure contract: any solver/geometry failure returns the finite
    (PENALTY, [G_FAIL, G_FAIL]) pair via fg_airfoil — never NaN, never an
    exception to the optimiser. XfoilError (binary missing/unexecutable) is
    an INFRASTRUCTURE fault and DOES propagate: silently scoring every
    design -100 on a broken install would fabricate a study.
    """

    re: float = 1e6
    mach: float = 0.0
    cl_design: float = 0.5
    tc_min: float = 0.10
    cm_max: float = 0.08          # |Cm| cap — trim-drag proxy (docstring)
    # Buildable-nose floor on the MIN per-surface r_LE/c (cst_le_radius): the
    # optional third constraint g2 = (min(r_u, r_l) - r_le_min) / r_le_min.
    # None => the m = 2 problem (no LE constraint), bit-for-bit the frozen
    # study; set to e.g. R_LE_MIN to screen out knife-edge noses XFOIL's
    # inviscid-panel + integral-BL model resolves poorly (module docstring).
    r_le_min: float | None = None
    n_cst: int = 4                # weights PER SURFACE; d = 2 n_cst
    dz_te: float = 0.0            # sharp TE, fixed (not a design variable)
    #: the CRUISE sweep this section is polared over (:func:`alpha_sweep`).
    #: Its TOP END caps the design lift that can be asked for — see
    #: :data:`ALPHA_SWEEP_DEG` for why the default does not move.
    alphas: np.ndarray = field(default_factory=alpha_sweep)
    cache_dir: str | Path | None = None    # None -> xfoil_run default cache
    timeout_s: float = 20.0
    n_panel: int = 200
    # Box anchor: a NACA 4-digit code (default — legacy behaviour, box and
    # w0 bit-for-bit the 2412 study), or an explicit (w_upper, w_lower)
    # CST fit of an arbitrary base section (the §15 screen-then-optimise
    # pipeline). w0 is CLIPPED into the box: the floor/cap can bind for
    # unconventional bases (a no-op for "2412").
    anchor: object = "2412"
    #: SYMMETRIC — the vertical stabiliser's mode, and it HALVES the vector.
    #:
    #: A fin at zero sideslip must make no side force, so its section has to
    #: have no camber: ``vlm.VerticalSurface`` pins its zero-lift angle to 0
    #: and a cambered fin would fly a permanent side load with nothing to
    #: trim it against. In CST that is exactly ``w_lower = -w_upper``, so the
    #: lower surface stops being free and the design vector is ``n_cst`` long
    #: instead of ``2 n_cst`` — four variables where the wing has eight.
    #:
    #: This is the whole of "simpler because it is only for a symmetrical
    #: aerofoil": the SAME machinery, the same XFOIL evaluation and the same
    #: constraints on half the box. It is not a different kind of search and
    #: it must not become one.
    #:
    #: Pair it with ``cl_design = 0.0``: a fin's design point is zero lift,
    #: which is the other half of what makes it a fin.
    symmetric: bool = False

    def __post_init__(self):
        if isinstance(self.anchor, str):
            w_u, w_l = cst_anchor(self.anchor, self.n_cst)
        else:
            w_u, w_l = (np.asarray(v, dtype=float).copy()
                        for v in self.anchor)
            if w_u.shape != (self.n_cst,) or w_l.shape != (self.n_cst,):
                raise ValueError(
                    f"anchor weight shapes {w_u.shape}/{w_l.shape} != "
                    f"({self.n_cst},)")
        if self.symmetric:
            # the anchor is symmetrised too: a cambered anchor would put the
            # box's centre on a shape this problem cannot represent
            w_u = 0.5 * (w_u - w_l)
            w_l = -w_u
        lo, hi = anchor_box(w_u, w_l)
        if np.any(lo >= hi):
            raise ValueError(
                "degenerate design box for this anchor (floor/cap crosses "
                f"the half-width band): lo={lo}, hi={hi}")
        self._w0 = np.clip(np.concatenate([w_u, w_l]), lo, hi)
        self._bounds = np.column_stack([lo, hi])
        if self.symmetric:
            # only the UPPER half is searched. The lower is not FIXED, it is
            # MIRRORED — see ``split_weights``, the one place that happens.
            self._w0 = self._w0[:self.n_cst]
            self._bounds = self._bounds[:self.n_cst]

    def split_weights(self, x):
        """``x`` -> ``(w_upper, w_lower)``, the ONE place the vector splits.

        There were two, and they had to agree: a symmetric problem carries
        the upper surface only and mirrors it, so a caller that still wrote
        ``x[:n], x[n:]`` would read the mirror as a lower surface and get a
        shape nobody asked for. Both now ask here.
        """
        x = np.asarray(x, dtype=float)
        if self.symmetric:
            w_u = x[: self.n_cst]
            return w_u, -w_u
        return x[: self.n_cst], x[self.n_cst:]

    @property
    def w0(self) -> np.ndarray:
        """Anchor point (default: NACA 2412) — the experiment's baseline
        design, clipped into the box (no-op for the default anchor)."""
        return self._w0.copy()

    @property
    def bounds(self) -> np.ndarray:
        return self._bounds.copy()

    @property
    def dim(self) -> int:
        return self.n_cst if self.symmetric else 2 * self.n_cst

    @property
    def param_labels(self) -> tuple:
        up = tuple(f"w_upper_{i}" for i in range(self.n_cst))
        if self.symmetric:
            return up
        return up + tuple(f"w_lower_{i}" for i in range(self.n_cst))


def _longest_increasing_run(v: np.ndarray) -> slice:
    """Slice of the longest CONTIGUOUS strictly-increasing run of ``v``.

    Used to isolate the pre-stall monotone branch of cl(alpha): past
    cl_max the lift curve folds over, cl stops being a function of the
    sweep index, and np.interp (which requires ascending abscissae) would
    silently corrupt the cd/cm interpolation. Ties/plateaus break a run.

    HEURISTIC limits (documented, accepted): "longest" is a proxy for
    "pre-stall" — it would pick a post-stall recovery branch if that
    branch both converged at more alphas than the pre-stall one and stall
    sat below ~3 deg (XFOIL essentially never converges long post-stall
    runs, so this is a theoretical corner). A mid-branch cl blip splits
    the run and can conservatively fail a fine section near cl_design
    (finite PENALTY per contract, never a wrong number).
    """
    n = int(v.size)
    if n == 0:
        return slice(0, 0)
    best_s, best_e, s = 0, 1, 0
    for i in range(1, n):
        if v[i] <= v[i - 1]:
            if i - s > best_e - best_s:
                best_s, best_e = s, i
            s = i
    if n - s > best_e - best_s:
        best_s, best_e = s, n
    return slice(best_s, best_e)


def xfoil_precheck(x: np.ndarray, prob: AirfoilProblem | None = None) -> str | None:
    """The reason ``evaluate_airfoil`` would reject ``x`` WITHOUT calling XFOIL,
    or None if it would go on to solve.

    Exactly the shape / bounds / geometry rejections below, factored out so a
    caller that wants to start a sweep speculatively (``airfoil_select.
    composite_evaluation`` prefetches the stall sweep) can ask whether XFOIL is
    going to be asked at all. A speculative sweep that fires for a design
    ``evaluate_airfoil`` rejects on geometry would burn a subprocess on
    coordinates nobody panels — and would break every caller whose contract is
    "an invalid design never reaches the solver".
    """
    prob = prob or AirfoilProblem()
    x = np.asarray(x, dtype=float)
    bnds = prob.bounds
    if x.shape != (bnds.shape[0],):
        return f"design vector shape {x.shape} != ({bnds.shape[0]},)"
    if np.any(x < bnds[:, 0] - 1e-12) or np.any(x > bnds[:, 1] + 1e-12):
        return "bounds violation"
    w_u, w_l = prob.split_weights(x)
    # Geometric prechecks on the interior psi grid (endpoints excluded: for
    # a sharp TE both surfaces meet, 0 = 0): positive thickness everywhere
    # behind the LE <=> upper strictly above lower <=> no self-intersection.
    # Guaranteed in-box by the floor/cap (class docstring) — this guards
    # out-of-box callers before XFOIL is asked to panel garbage.
    psi = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, 401)))[1:-1]
    gap = _surface_y(psi, w_u, +0.5 * prob.dz_te) - _surface_y(
        psi, w_l, -0.5 * prob.dz_te)
    if not np.all(gap > 0.0):
        return "self-intersecting / non-positive-thickness geometry"
    return None


#: what ``evaluate_airfoil`` maximises, under its own name. MINUS a
#: drag coefficient: a negative number on a page next to a positive
#: L/D, and the two must never share an axis.
AIRFOIL_SCORE_UNITS = "-c_d at cl_design"


def evaluate_airfoil(x: np.ndarray, prob: AirfoilProblem | None = None) -> dict:
    """Full Tier B evaluation with breakdown; never raises for in-contract
    failures (bad geometry / unconvergeable section). ``feasible`` refers
    to the SOLVER (PENALTY contract); the two design constraints are
    reported separately as the signed margin array ``g`` — constrained BO
    needs true objective values AT infeasible designs (hydrofoil.py
    convention)."""
    from .objective import _fail
    from .xfoil_run import run_xfoil_polar

    prob = prob or AirfoilProblem()
    x = np.asarray(x, dtype=float)
    # shape / bounds / geometry — the rejections that never reach the solver,
    # in xfoil_precheck so a speculative caller can ask the same question
    reason = xfoil_precheck(x, prob)
    if reason is not None:
        return _fail(reason)

    w_u, w_l = prob.split_weights(x)
    tc = cst_thickness(w_u, w_l, dz_te=prob.dz_te)
    coords = cst_coords(w_u, w_l, dz_te=prob.dz_te)
    pol = run_xfoil_polar(
        coords, prob.re, prob.mach, prob.alphas,
        timeout_s=prob.timeout_s, cache_dir=prob.cache_dir,
        n_panel=prob.n_panel,
    )   # XfoilError (infrastructure fault) intentionally propagates

    if pol.n_converged < 3:
        return _fail(
            "unconvergeable section (< 3 converged XFOIL points) "
            "= failed design", n_converged=pol.n_converged)

    sl = _longest_increasing_run(pol.cl)
    cl_b, cd_b = pol.cl[sl], pol.cd[sl]
    cm_b, al_b = pol.cm[sl], pol.alpha_deg[sl]
    if cl_b.size < 2 or cl_b[-1] < prob.cl_design or cl_b[0] > prob.cl_design:
        # NAME THE CEILING, and name what moves it. The branch can stop short
        # for two completely different reasons — the section stalls, or the
        # SWEEP ran out (ALPHA_SWEEP_DEG) — and the old wording asserted the
        # first in both cases. It is the second that produces a whole search
        # with no feasible design above cl_design ~ 1.3, so the two are told
        # apart here by whether the branch reached the top of the sweep.
        al_lo, al_hi = float(prob.alphas[0]), float(prob.alphas[-1])
        sweep_limited = bool(cl_b.size and float(al_b[-1]) >= al_hi - 1e-9
                             and cl_b[-1] < prob.cl_design)
        why = (f"the alpha sweep stops at {al_hi:g} deg with cl still rising "
               f"— widen it (flags['airfoil_sweep_alpha_max_deg'])"
               if sweep_limited else
               "the section stalls below it")
        return _fail(
            f"cl_design {float(prob.cl_design):.4g} not bracketed by the "
            f"pre-stall monotone branch, which reaches cl "
            f"{float(cl_b[-1]) if cl_b.size else float('nan'):.4g} over the "
            f"sweep [{al_lo:g}, {al_hi:g}] deg: {why} = failed design",
            n_converged=pol.n_converged, cl_max=float(pol.cl.max()),
            cl_branch_max=(float(cl_b[-1]) if cl_b.size else None),
            alpha_sweep_deg=(al_lo, al_hi), sweep_limited=sweep_limited)

    cd_at = float(np.interp(prob.cl_design, cl_b, cd_b))
    cm_at = float(np.interp(prob.cl_design, cl_b, cm_b))
    alpha_at = float(np.interp(prob.cl_design, cl_b, al_b))
    f = -cd_at                       # MAXIMISE f  <=>  minimise cd at cl
    if not np.isfinite(f):
        return _fail("non-finite objective")
    # Nose radius of the blunter-LIMITING surface (the min binds the floor):
    # r_LE/c = w_0^2/2 per surface, cambered sections split into two (docstring).
    r_le = min(cst_le_radius(w_u), cst_le_radius(w_l))
    margins = [norm_margin(tc - prob.tc_min, prob.tc_min),
               norm_margin(prob.cm_max - abs(cm_at), prob.cm_max)]
    if prob.r_le_min is not None:
        # third constraint g2 — normalised by the floor so it is O(1) like
        # the other two margins (feasible iff min r_LE/c >= r_le_min).
        margins.append(norm_margin(r_le - prob.r_le_min, prob.r_le_min))
    g = np.array(margins)

    return {
        "feasible": True, "reason": "", "score": float(f), "f": float(f),
        "score_units": AIRFOIL_SCORE_UNITS,
        "cd": cd_at, "cm": cm_at, "alpha_deg": alpha_at, "tc": float(tc),
        "r_le": float(r_le),
        "g": g, "g_tc": float(g[0]), "g_cm": float(g[1]),
        "g_rle": (float(g[2]) if g.size == 3 else None),
        "cl_design": float(prob.cl_design),
        "n_converged": pol.n_converged, "n_branch": int(cl_b.size),
        "cl_max_branch": float(cl_b[-1]), "from_cache": pol.from_cache,
    }


def fg_airfoil(x: np.ndarray, prob: AirfoilProblem | None = None
               ) -> tuple[float, np.ndarray]:
    """Constrained-harness callable: (f, g) — multi-g contract
    (optimize/constrained.py; feasible iff ALL margins >= 0). The margin
    vector is length m = 2 ([g_tc, g_cm]) by default, or m = 3
    ([g_tc, g_cm, g_rle]) when ``prob.r_le_min`` is set.

    Solvable-but-infeasible designs return their TRUE f with signed
    margins; solver/geometry failures return exactly (PENALTY, [G_FAIL]*m)
    — a finite, clearly-infeasible margin of the correct width so the
    optimiser's constraint block (the pymoo GA sizes it up front) matches."""
    prob = prob or AirfoilProblem()
    out = evaluate_airfoil(x, prob)
    if not out["feasible"]:
        m = 3 if prob.r_le_min is not None else 2
        return PENALTY, np.full(m, G_FAIL)
    return float(out["score"]), np.asarray(out["g"], dtype=float)
