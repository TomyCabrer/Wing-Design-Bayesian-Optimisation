"""Section polars: XFOIL .pol ingestion and a clearly-labelled analytic fallback.

A polar provides, for the lifting-line and profile-drag integration:
  cl(alpha_deg), cd(alpha_deg), cm(alpha_deg)   interpolants
  a_lin      linear-region lift-curve slope [1/rad]
  alpha_L0   zero-lift angle [rad]
  alpha_valid = (lo, hi) [deg]  range outside which the model is untrusted
                (used by the objective as a stall/extrapolation failure proxy)

The linear-region extraction ports `extract_linear` from
~/Desktop/GDP/tandem_wing/core/lift_drag_strip.m: locate the zero-lift
crossing, least-squares fit Cl(alpha) within +-6 deg of it.
Incompressible Tier A: no Prandtl-Glauert correction (beta_PG = 1).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def _extract_linear(alpha_deg: np.ndarray, cl: np.ndarray) -> tuple[float, float]:
    """Return (a_lin [1/rad], alpha_L0 [rad]) from a Cl(alpha) table."""
    sgn = np.sign(cl)
    crossings = np.where(sgn[:-1] * sgn[1:] < 0)[0]
    if crossings.size:
        i = crossings[0]
        a0_guess = alpha_deg[i] + cl[i] * (alpha_deg[i + 1] - alpha_deg[i]) / (
            cl[i] - cl[i + 1]
        )
    else:
        a0_guess = 0.0
    mask = (alpha_deg >= a0_guess - 6.0) & (alpha_deg <= a0_guess + 6.0)
    if mask.sum() < 3:
        i_max = int(np.argmax(cl))
        mask = alpha_deg < alpha_deg[i_max] - 1.0
    coeffs = np.polyfit(alpha_deg[mask], cl[mask], 1)
    slope_per_deg, intercept = coeffs[0], coeffs[1]
    a_lin = slope_per_deg * 180.0 / np.pi
    alpha_L0 = np.deg2rad(-intercept / slope_per_deg)
    return float(a_lin), float(alpha_L0)


@dataclass
class TablePolar:
    """Polar backed by an (alpha, CL, CD, CM) table — e.g. an XFOIL .pol file.

    Optionally carries a companion minimum-surface-Cp table (alpha, Cp_min)
    from scripts/gen_cpmin_family.py (XFOIL CPMN per alpha, viscous minimum)
    for the Tier C cavitation constraint; ``cp_min`` raises if absent.
    """

    alpha_deg: np.ndarray
    CL: np.ndarray
    CD: np.ndarray
    CM: np.ndarray
    name: str = "table"
    Re: float | None = None
    alpha_cpmin: np.ndarray | None = None
    CPMIN: np.ndarray | None = None
    #: the linear-region pair, STATED rather than extracted. A table built by
    #: BLENDING two members is the one case where extraction is wrong: the
    #: blend is carried on the INTERSECTED alpha grid, so re-fitting the
    #: slope there fits it in a window clipped by the OTHER member's
    #: convergence range, and the result can leave the convex hull of the two
    #: slopes it is between. Every blend in this module states the pair
    #: instead (BlendedPolar and BilinearPolar2D always did; polar_at_re does
    #: now). ``None`` = extract from the table, which is the right thing for
    #: every FILE-backed and XFOIL-backed polar and is bit-for-bit what this
    #: class has always done.
    a_lin_override: float | None = None
    alpha_L0_override: float | None = None
    a_lin: float = field(init=False)
    alpha_L0: float = field(init=False)

    def __post_init__(self):
        order = np.argsort(self.alpha_deg)
        self.alpha_deg = np.asarray(self.alpha_deg, float)[order]
        self.CL = np.asarray(self.CL, float)[order]
        self.CD = np.asarray(self.CD, float)[order]
        self.CM = np.asarray(self.CM, float)[order]
        # extracted FIRST even when overridden: a degenerate table still has
        # to raise here, where it always has, rather than silently become
        # valid because a caller happened to state a slope for it
        self.a_lin, self.alpha_L0 = _extract_linear(self.alpha_deg, self.CL)
        if self.a_lin_override is not None:
            self.a_lin = float(self.a_lin_override)
        if self.alpha_L0_override is not None:
            self.alpha_L0 = float(self.alpha_L0_override)
        if self.alpha_cpmin is not None:
            o = np.argsort(self.alpha_cpmin)
            self.alpha_cpmin = np.asarray(self.alpha_cpmin, float)[o]
            self.CPMIN = np.asarray(self.CPMIN, float)[o]

    @property
    def alpha_valid(self) -> tuple[float, float]:
        return float(self.alpha_deg[0]), float(self.alpha_deg[-1])

    @property
    def has_cp_min(self) -> bool:
        return self.alpha_cpmin is not None

    def cl(self, alpha_deg):
        return np.interp(alpha_deg, self.alpha_deg, self.CL)

    def cd(self, alpha_deg):
        return np.interp(alpha_deg, self.alpha_deg, self.CD)

    def cm(self, alpha_deg):
        return np.interp(alpha_deg, self.alpha_deg, self.CM)

    def cp_min(self, alpha_deg):
        """Minimum (most negative) surface Cp at alpha [deg] — XFOIL viscous
        CPMN table, linear interpolation, exact at tabulated alphas."""
        if self.alpha_cpmin is None:
            raise ValueError(f"polar '{self.name}' has no Cp_min table "
                             "(run scripts/gen_cpmin_family.py)")
        return np.interp(alpha_deg, self.alpha_cpmin, self.CPMIN)


def load_xfoil_polar(path: str | Path, name: str | None = None) -> TablePolar:
    """Parse an XFOIL .pol file (v6.9x PACC format, tolerant of extra columns).

    Data rows have >= 5 numeric columns: alpha, CL, CD, CDp, CM, [Xtr...].
    """
    path = Path(path)
    rows = []
    re_num = None
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s:
            continue
        if "Re =" in s:
            # e.g. "Mach =   0.000     Re =     1.000 e 6     Ncrit ..."
            try:
                seg = s.split("Re =")[1].split("Ncrit")[0]
                re_num = float(seg.replace(" ", "").replace("e", "e"))
            except ValueError:
                re_num = None
        parts = s.split()
        try:
            vals = [float(p) for p in parts]
        except ValueError:
            continue
        if len(vals) >= 5:
            rows.append(vals[:5])
    if len(rows) < 5:
        raise ValueError(f"no polar data rows parsed from {path}")
    arr = np.array(rows)

    # companion Cp_min table (gen_cpmin_family.py), if present on disk
    alpha_cp, cpmin = None, None
    cp_path = path.with_suffix(".cpmin")
    if cp_path.exists():
        cp_rows = [
            [float(p) for p in ln.split()]
            for ln in cp_path.read_text().splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")
        ]
        if cp_rows:
            cp_arr = np.array(cp_rows)
            alpha_cp, cpmin = cp_arr[:, 0], cp_arr[:, 1]

    return TablePolar(
        alpha_deg=arr[:, 0], CL=arr[:, 1], CD=arr[:, 2], CM=arr[:, 4],
        name=name or path.stem, Re=re_num,
        alpha_cpmin=alpha_cp, CPMIN=cpmin,
    )


@dataclass
class AnalyticPolar:
    """PLACEHOLDER analytic thin-airfoil polar — NOT real airfoil data.

    cl = 2*pi*(alpha - alpha_L0), cd = cd0 + k*(cl - cl_cd_min)^2, cm const.
    Coefficients loosely resemble a cambered section at Re ~ 1e6; use only
    when no XFOIL polar is available, and label results accordingly.
    """

    alpha_L0_deg: float = -2.0
    cd0: float = 0.006
    k: float = 0.008
    cl_cd_min: float = 0.3
    cm0: float = -0.05
    valid_deg: float = 12.0
    name: str = "analytic-thin-airfoil (PLACEHOLDER)"
    Re: float | None = None

    @property
    def a_lin(self) -> float:
        return 2.0 * np.pi

    @property
    def alpha_L0(self) -> float:
        return float(np.deg2rad(self.alpha_L0_deg))

    @property
    def alpha_valid(self) -> tuple[float, float]:
        return (self.alpha_L0_deg - self.valid_deg, self.alpha_L0_deg + self.valid_deg)

    def cl(self, alpha_deg):
        return 2.0 * np.pi * np.deg2rad(np.asarray(alpha_deg) - self.alpha_L0_deg)

    def cd(self, alpha_deg):
        return self.cd0 + self.k * (self.cl(alpha_deg) - self.cl_cd_min) ** 2

    def cm(self, alpha_deg):
        return np.full_like(np.asarray(alpha_deg, dtype=float), self.cm0)

    def cp_min(self, alpha_deg):
        """PLACEHOLDER minimum surface Cp — thin-foil style correlation
        cp_min = c0 + c1*cl + c2*cl^2, least-squares fit to the REAL
        NACA 2412 XFOIL viscous CPMN data at alpha = 0/2/4/6 deg (Re 1e6):
        fit residual < 0.06 there. NOT real data at other sections; use
        the TablePolar route (gen_cpmin_family.py) whenever available."""
        cl = self.cl(alpha_deg)
        return -0.8043 + 1.7864 * cl - 3.6491 * cl**2


def section_cm_ac(pol, default: float = 0.0) -> float:
    """Section pitching moment about the QUARTER CHORD at zero lift — cm_ac.

    This is the constant every pitch balance in the package needs and none
    of them used to ask for: a cambered section carries a nose-down couple
    that does not vanish with its lift, so an aircraft's tail has to react
    it whatever the CG does. Read at ``alpha_L0`` because that is where
    cm(c/4) IS the moment about the aerodynamic centre by definition (the
    lift term is zero there); over the linear range of a real table it is
    near enough constant to be treated as one — NACA 2412 at Re 1e6 spans
    -0.048 .. -0.057 between alpha = -4 and +6 deg against -0.0541 here,
    and holding it constant is what keeps CL and Cm AFFINE in the trim
    unknowns (tail.py's closed-form elimination).

    ``default`` (0.0) is returned for anything that cannot answer — a polar
    with no ``cm``, or one whose table does not reach its own zero-lift
    angle. Zero is the SYMMETRIC-section value, i.e. the honest "this model
    has no section moment to report", and it reproduces every result taken
    before this term existed.
    """
    cm = getattr(pol, "cm", None)
    if not callable(cm):
        return float(default)
    a0 = float(np.rad2deg(float(getattr(pol, "alpha_L0", 0.0))))
    try:
        val = float(np.asarray(cm(a0), dtype=float).reshape(-1)[0])
    except (TypeError, ValueError, IndexError):
        return float(default)
    return val if np.isfinite(val) else float(default)


@dataclass
class InvertedPolar:
    """A section flown UPSIDE DOWN — the z -> -z mirror of another polar.

    Mounting a cambered aerofoil inverted is what a surface that pushes DOWN
    is for: its camber then works the side the surface actually loads, and
    the trim incidence stops fighting it. On the published tail this is
    worth 3.9 % of the surface's profile drag, and the tell-tale is the
    incidence — i_t goes from -4.93 deg (cranked nose-down against its own
    camber) to -0.31 deg (sitting in the flow).

    The mirror is exact, not a model: reflecting the coordinates in the
    chord line maps the whole polar,

        cl(a) = -cl0(-a),   cd(a) = cd0(-a),   cm(a) = -cm0(-a),

    so alpha_L0 and cm_ac flip sign, the lift-curve slope and thickness are
    unchanged, and the validity window mirrors with everything else — an
    inverted section stalls where the upright one stalls NEGATIVELY. Cp_min
    mirrors too, which is what keeps the cavitation constraint honest on an
    inverted hydrofoil surface.

    Inverting twice returns the original object rather than a stack of
    wrappers (:func:`inverted`).
    """

    base: object
    a_lin: float = field(init=False)
    alpha_L0: float = field(init=False)

    def __post_init__(self):
        self.a_lin = float(self.base.a_lin)
        self.alpha_L0 = -float(self.base.alpha_L0)

    @property
    def name(self) -> str:
        return f"{getattr(self.base, 'name', 'section')} (inverted)"

    @property
    def tc(self) -> float:
        return float(getattr(self.base, "tc", 0.12))

    @property
    def alpha_valid(self) -> tuple[float, float]:
        lo, hi = self.base.alpha_valid
        return (-float(hi), -float(lo))

    @property
    def has_cp_min(self) -> bool:
        return bool(getattr(self.base, "has_cp_min", False))

    def cl(self, alpha_deg):
        return -np.asarray(self.base.cl(-np.asarray(alpha_deg, dtype=float)))

    def cd(self, alpha_deg):
        return np.asarray(self.base.cd(-np.asarray(alpha_deg, dtype=float)))

    def cm(self, alpha_deg):
        return -np.asarray(self.base.cm(-np.asarray(alpha_deg, dtype=float)))

    def cp_min(self, alpha_deg):
        return np.asarray(
            self.base.cp_min(-np.asarray(alpha_deg, dtype=float)))


def inverted(pol):
    """``pol`` mounted upside down (:class:`InvertedPolar`); un-inverts."""
    if isinstance(pol, InvertedPolar):
        return pol.base
    return InvertedPolar(pol)


def invert_coords(coords) -> np.ndarray:
    """Section coordinates mirrored in the chord line — z -> -z.

    The shape that goes with :class:`InvertedPolar`, for anything that
    DRAWS or LOFTS the section (the geometry views, the STL/VSP export):
    a surface flown inverted has to be drawn inverted or the picture
    contradicts the polar it was solved on. The point ORDER is reversed
    with the sign so the contour still runs the way every consumer here
    expects it to (TE -> upper -> LE -> lower -> TE).
    """
    xy = np.asarray(coords, dtype=float)
    out = xy[::-1].copy()
    out[:, 1] = -out[:, 1]
    return out


def default_polar(data_dir: str | Path | None = None):
    """NACA 2412 XFOIL polar at Re 1e6 if present, else the analytic fallback."""
    if data_dir is None:
        data_dir = Path(__file__).resolve().parents[2] / "data" / "airfoils"
    pol = Path(data_dir) / "naca2412_re1e6.pol"
    if pol.exists():
        return load_xfoil_polar(pol, name="NACA2412 Re1e6 (XFOIL)")
    return AnalyticPolar()


@dataclass
class BlendedPolar:
    """Linear t/c blend of two bracketing TablePolars (built by PolarFamily.at).

    cl/cd/cm(alpha) = (1-w)*lo(alpha) + w*hi(alpha) with w the linear weight
    in t/c; a_lin and alpha_L0 blend the same way. alpha_valid is the
    INTERSECTION of the members' ranges (conservative: outside it at least
    one member would be extrapolating).
    """

    lo: TablePolar
    hi: TablePolar
    w: float
    name: str = "blended"
    Re: float | None = None

    @property
    def a_lin(self) -> float:
        return (1.0 - self.w) * self.lo.a_lin + self.w * self.hi.a_lin

    @property
    def alpha_L0(self) -> float:
        return (1.0 - self.w) * self.lo.alpha_L0 + self.w * self.hi.alpha_L0

    @property
    def alpha_valid(self) -> tuple[float, float]:
        lo_lo, lo_hi = self.lo.alpha_valid
        hi_lo, hi_hi = self.hi.alpha_valid
        return max(lo_lo, hi_lo), min(lo_hi, hi_hi)

    def cl(self, alpha_deg):
        return (1.0 - self.w) * self.lo.cl(alpha_deg) + self.w * self.hi.cl(alpha_deg)

    def cd(self, alpha_deg):
        return (1.0 - self.w) * self.lo.cd(alpha_deg) + self.w * self.hi.cd(alpha_deg)

    def cm(self, alpha_deg):
        return (1.0 - self.w) * self.lo.cm(alpha_deg) + self.w * self.hi.cm(alpha_deg)

    def cp_min(self, alpha_deg):
        return (1.0 - self.w) * self.lo.cp_min(alpha_deg) \
            + self.w * self.hi.cp_min(alpha_deg)


class PolarFamily:
    """Section-polar family indexed by thickness ratio t/c.

    Members are REAL XFOIL polars (one airfoil per t/c, same Re/Mach/Ncrit).
    ``at(tc)`` returns the polar model for a design t/c:

      - at a member t/c (within 1e-9): the member TablePolar itself, EXACTLY
        (no interpolation error at the anchors);
      - between members: a BlendedPolar interpolating cl/cd/cm/a_lin/alpha_L0
        LINEARLY in t/c between the two bracketing members. Linear-in-t/c is
        a reduced-order modelling choice, adequate for the 3-percentage-point
        member spacing here (NACA 24XX section properties vary smoothly with
        thickness); it is NOT XFOIL data at intermediate t/c.

    Requesting t/c outside [min, max] member thickness raises ValueError
    (no extrapolation of polar data).
    """

    def __init__(self, members: dict[float, TablePolar]):
        if len(members) < 2:
            raise ValueError("PolarFamily needs at least 2 members")
        self._tc = np.array(sorted(members), dtype=float)
        self._polars = [members[t] for t in self._tc]
        self.name = "PolarFamily[" + ", ".join(p.name for p in self._polars) + "]"

    @property
    def tc_range(self) -> tuple[float, float]:
        return float(self._tc[0]), float(self._tc[-1])

    @property
    def members(self) -> dict[float, TablePolar]:
        return dict(zip(self._tc.tolist(), self._polars))

    def at(self, tc: float):
        """Polar model at thickness ratio ``tc`` (see class docstring)."""
        tc = float(tc)
        lo, hi = self.tc_range
        if tc < lo - 1e-12 or tc > hi + 1e-12:
            raise ValueError(f"t/c = {tc:.4f} outside polar family range [{lo}, {hi}]")
        i_near = int(np.argmin(np.abs(self._tc - tc)))
        if abs(self._tc[i_near] - tc) <= 1e-9:
            return self._polars[i_near]          # exact member — no blending
        i_hi = int(np.searchsorted(self._tc, tc))
        i_lo = i_hi - 1
        w = (tc - self._tc[i_lo]) / (self._tc[i_hi] - self._tc[i_lo])
        p_lo, p_hi = self._polars[i_lo], self._polars[i_hi]
        return BlendedPolar(
            lo=p_lo, hi=p_hi, w=float(w),
            name=f"{p_lo.name} <-> {p_hi.name} @ t/c={tc:.3f} (linear blend)",
            Re=p_lo.Re,
        )


# ---------------------------------------------------------------------------
# 2-D section-polar family over (t/c, cl_design) — the pre-optimised CST
# library (scripts/gen_cst_library.py). ADDITIVE to this module: nothing
# above this banner is touched; the 1-D PolarFamily keeps its behaviour and
# callers.
# ---------------------------------------------------------------------------


@dataclass
class BilinearPolar2D:
    """Bilinear (t/c, cl_design) blend of the 4 cell-corner member polars
    (built by PolarFamily2D.at — not meant to be constructed directly).

    Same reduced-order rule as BlendedPolar, applied along both grid axes:
    the members' cl/cd/cm tables are resampled onto ONE common alpha grid
    (union of the member alpha nodes, clipped to the INTERSECTION of their
    valid ranges — outside it at least one member would be extrapolating)
    and combined with the bilinear corner weights; ``a_lin`` and
    ``alpha_L0`` combine with the SAME weights (blended linearly, not
    re-extracted from the blended table).
    """

    alpha_deg: np.ndarray          # common alpha grid [deg], ascending
    CL: np.ndarray                 # bilinearly blended tables on that grid
    CD: np.ndarray
    CM: np.ndarray
    a_lin: float                   # bilinear blend of member a_lin [1/rad]
    alpha_L0: float                # bilinear blend of member alpha_L0 [rad]
    name: str = "bilinear-2d"
    Re: float | None = None

    @property
    def alpha_valid(self) -> tuple[float, float]:
        return float(self.alpha_deg[0]), float(self.alpha_deg[-1])

    def cl(self, alpha_deg):
        return np.interp(alpha_deg, self.alpha_deg, self.CL)

    def cd(self, alpha_deg):
        return np.interp(alpha_deg, self.alpha_deg, self.CD)

    def cm(self, alpha_deg):
        return np.interp(alpha_deg, self.alpha_deg, self.CM)

    def cp_min(self, alpha_deg):
        raise ValueError(
            f"polar '{self.name}' has no Cp_min table — the CST library "
            "members carry no CPMN data (Tier C only)")


class PolarFamily2D:
    """Section-polar family indexed by (t/c, cl_design) — the pre-optimised
    CST section library.

    Members are REAL XFOIL polars of BO-optimised CST sections, one per node
    of a full rectangular (t/c, cl_design) grid (same Re/Mach/Ncrit;
    scripts/gen_cst_library.py). ``at(tc, cl_sec)`` returns the polar model
    for a wing station commanding structural depth ``tc`` and section design
    lift ``cl_sec``:

      - at a member node (both coordinates within 1e-9): the member
        TablePolar itself, EXACTLY (no interpolation error at the anchors —
        same guarantee as the 1-D PolarFamily);
      - inside the grid: a BilinearPolar2D combining cl/cd/cm tables,
        ``a_lin`` and ``alpha_L0`` BILINEARLY from the 4 surrounding
        members. Bilinear-in-(t/c, cl_design) is a reduced-order modelling
        choice, adequate for the 0.03 x 0.2 cell spacing here (optimised
        section properties vary smoothly across neighbouring cells); it is
        NOT XFOIL data at intermediate (t/c, cl_design) — intermediate
        points are INTERPOLATED, never XFOIL-verified.

    Requesting a point outside the rectangular grid hull raises ValueError
    (no extrapolation of polar data).
    """

    def __init__(self, members: dict[tuple[float, float], TablePolar]):
        tcs = sorted({k[0] for k in members})
        cls_ = sorted({k[1] for k in members})
        if len(tcs) < 2 or len(cls_) < 2:
            raise ValueError(
                "PolarFamily2D needs at least a 2 x 2 (t/c, cl_design) grid")
        gaps = [(t, c) for t in tcs for c in cls_ if (t, c) not in members]
        if gaps:
            raise ValueError(
                f"PolarFamily2D grid is not rectangular — missing {gaps}")
        self._tc = np.array(tcs, dtype=float)
        self._cl = np.array(cls_, dtype=float)
        self._polars = {k: members[k] for k in members}
        self.name = (f"PolarFamily2D[{len(tcs)} t/c x {len(cls_)} cl_design "
                     f"members]")

    @property
    def tc_range(self) -> tuple[float, float]:
        return float(self._tc[0]), float(self._tc[-1])

    @property
    def cl_range(self) -> tuple[float, float]:
        return float(self._cl[0]), float(self._cl[-1])

    @property
    def members(self) -> dict[tuple[float, float], TablePolar]:
        return dict(self._polars)

    @staticmethod
    def _bracket(grid: np.ndarray, v: float) -> tuple[int, int, float]:
        """(i_lo, i_hi, w) with grid[i_lo] <= v <= grid[i_hi], linear w."""
        i_hi = int(np.clip(np.searchsorted(grid, v), 1, grid.size - 1))
        i_lo = i_hi - 1
        w = (v - grid[i_lo]) / (grid[i_hi] - grid[i_lo])
        return i_lo, i_hi, float(np.clip(w, 0.0, 1.0))

    def at(self, tc: float, cl_sec: float):
        """Polar model at (t/c, cl_design) = (tc, cl_sec) (class docstring)."""
        tc, cl_sec = float(tc), float(cl_sec)
        t_lo, t_hi = self.tc_range
        c_lo, c_hi = self.cl_range
        if (tc < t_lo - 1e-12 or tc > t_hi + 1e-12
                or cl_sec < c_lo - 1e-12 or cl_sec > c_hi + 1e-12):
            raise ValueError(
                f"(t/c, cl_design) = ({tc:.4f}, {cl_sec:.4f}) outside the "
                f"polar-family grid hull [{t_lo}, {t_hi}] x [{c_lo}, {c_hi}]")

        i_t = int(np.argmin(np.abs(self._tc - tc)))
        i_c = int(np.argmin(np.abs(self._cl - cl_sec)))
        if (abs(self._tc[i_t] - tc) <= 1e-9
                and abs(self._cl[i_c] - cl_sec) <= 1e-9):
            # exact member node — no blending (anchors are exact)
            return self._polars[(float(self._tc[i_t]), float(self._cl[i_c]))]

        it_lo, it_hi, wt = self._bracket(self._tc, tc)
        ic_lo, ic_hi, wc = self._bracket(self._cl, cl_sec)
        corners = [           # (member, bilinear weight); weights sum to 1
            (self._polars[(float(self._tc[it_lo]), float(self._cl[ic_lo]))],
             (1.0 - wt) * (1.0 - wc)),
            (self._polars[(float(self._tc[it_hi]), float(self._cl[ic_lo]))],
             wt * (1.0 - wc)),
            (self._polars[(float(self._tc[it_lo]), float(self._cl[ic_hi]))],
             (1.0 - wt) * wc),
            (self._polars[(float(self._tc[it_hi]), float(self._cl[ic_hi]))],
             wt * wc),
        ]

        # common alpha grid: union of member nodes on the INTERSECTION of
        # their valid ranges (BlendedPolar's conservative alpha_valid rule)
        a_lo = max(p.alpha_valid[0] for p, _ in corners)
        a_hi = min(p.alpha_valid[1] for p, _ in corners)
        if a_hi <= a_lo:
            raise ValueError(
                f"member polars around ({tc:.4f}, {cl_sec:.4f}) share no "
                "overlapping alpha range")
        nodes = np.concatenate([p.alpha_deg for p, _ in corners])
        grid = np.unique(np.concatenate(
            [nodes[(nodes >= a_lo) & (nodes <= a_hi)], [a_lo, a_hi]]))

        CL = sum(w * p.cl(grid) for p, w in corners)
        CD = sum(w * p.cd(grid) for p, w in corners)
        CM = sum(w * p.cm(grid) for p, w in corners)
        return BilinearPolar2D(
            alpha_deg=grid, CL=CL, CD=CD, CM=CM,
            a_lin=sum(w * p.a_lin for p, w in corners),
            alpha_L0=sum(w * p.alpha_L0 for p, w in corners),
            name=(f"CST library @ t/c={tc:.3f}, cl_design={cl_sec:.3f} "
                  "(bilinear blend)"),
            Re=corners[0][0].Re,
        )


def load_cst_polar_family(data_dir: str | Path | None = None) -> PolarFamily2D:
    """CST section library (scripts/gen_cst_library.py) as a PolarFamily2D.

    Loads every data/airfoils/cst_library/cst_tc*_cl*.json — REAL XFOIL
    polars of BO-optimised sections, never fabricated; the (t/c, cl_design)
    grid node of each member is read from the file's own tc_cell/cl_design
    fields (not parsed from the filename). Raises FileNotFoundError if no
    library is on disk and ValueError (via PolarFamily2D) if the stored
    cells do not form a full rectangular grid.
    """
    if data_dir is None:
        data_dir = (Path(__file__).resolve().parents[2]
                    / "data" / "airfoils" / "cst_library")
    files = sorted(Path(data_dir).glob("cst_tc*_cl*.json"))
    if not files:
        raise FileNotFoundError(
            f"no CST library members in {data_dir}; "
            "run scripts/gen_cst_library.py")
    members: dict[tuple[float, float], TablePolar] = {}
    for f in files:
        blob = json.loads(f.read_text())
        key = (float(blob["tc_cell"]), float(blob["cl_design"]))
        members[key] = TablePolar(
            alpha_deg=np.asarray(blob["alpha_deg"], dtype=float),
            CL=np.asarray(blob["cl"], dtype=float),
            CD=np.asarray(blob["cd"], dtype=float),
            CM=np.asarray(blob["cm"], dtype=float),
            name=(f"CST t/c>={key[0]} cl_design={key[1]} "
                  f"Re{blob['provenance']['re']:.0e} (XFOIL)"),
            Re=float(blob["provenance"]["re"]),
        )
    return PolarFamily2D(members)


def default_polar_family(data_dir: str | Path | None = None) -> PolarFamily:
    """NACA 24XX XFOIL family at Re 1e6 (t/c = 0.06 ... 0.18).

    Loads every data/airfoils/naca24XX_re1e6.pol present (generated by
    scripts/gen_polar_family.py with a local XFOIL binary — real polars,
    never fabricated). Raises FileNotFoundError if fewer than 2 members
    are on disk.
    """
    if data_dir is None:
        data_dir = Path(__file__).resolve().parents[2] / "data" / "airfoils"
    members: dict[float, TablePolar] = {}
    for pol in sorted(Path(data_dir).glob("naca24??_re1e6.pol")):
        code = pol.stem[4:8]
        tc = int(code[2:]) / 100.0
        members[tc] = load_xfoil_polar(pol, name=f"NACA{code} Re1e6 (XFOIL)")
    if len(members) < 2:
        raise FileNotFoundError(
            f"need >= 2 naca24XX_re1e6.pol files in {data_dir}; "
            "run scripts/gen_polar_family.py"
        )
    return PolarFamily(members)


# ---------------------------------------------------------------------------
# Reynolds-number bank of the SAME family — scripts/gen_polar_family_re.py.
#
# ADDITIVE, and deliberately so: `default_polar_family` above globs
# "naca24??_re1e6.pol", which matches none of the files this section reads, so
# every published run keeps flying the Re-1e6 tables byte for byte. What this
# adds is the ability to ASK for another Reynolds number, which until now the
# family simply could not be asked — a hydrofoil elevator at Re 1.5e5 and a
# 12 m wing at 3e6 were looked up in the same table, and nothing downstream
# read `TablePolar.Re`, so no correction was applied anywhere either. Three
# consumers ask now (tests/test_api_config_promises.py declares them): both
# car families, and the WATER families' `flown_reynolds` opt-in — the one
# that needed the Cp_min half of the bank, which is why every (t/c, Re) cell
# carries a `.cpmin` companion rather than only Re 1e6.
#
# Measured on the generated bank (NACA 2412, cd at alpha = 2 deg):
#
#     Re      1e5      3e5      1e6      3e6
#     cd    0.01551  0.00889  0.00578  0.00508
#
# i.e. flying a Re-3e5 wing on the 1e6 table under-predicts section drag by
# 31 counts (35 %), and at 1e5 by 97 counts (63 %).
# ---------------------------------------------------------------------------

#: filename tags of the shipped Reynolds bank, with the Reynolds number each
#: one was actually run at. "1e6" is the original family; the rest come from
#: scripts/gen_polar_family_re.py under the identical XFOIL recipe.
RE_BANK_TAGS: tuple[tuple[float, str], ...] = (
    (1.0e5, "1e5"), (3.0e5, "3e5"), (1.0e6, "1e6"), (3.0e6, "3e6"),
    # the top node, added for the water families: with their size rows open
    # a hydrofoil box reaches Re_mac 7.3e6, past what 3e6 can answer for.
    (1.0e7, "1e7"),
)


def polar_re_bank(data_dir: str | Path | None = None
                  ) -> dict[float, dict[float, TablePolar]]:
    """``{t/c: {Re: TablePolar}}`` for every NACA 24XX member on disk.

    Missing (t/c, Re) cells are simply absent — the generator records a failed
    XFOIL march rather than faking a table, and low Reynolds numbers do fail on
    the thin members. Callers must therefore handle ragged coverage; that is
    what :func:`polar_at_re` does.
    """
    if data_dir is None:
        data_dir = Path(__file__).resolve().parents[2] / "data" / "airfoils"
    out: dict[float, dict[float, TablePolar]] = {}
    for re, tag in RE_BANK_TAGS:
        for pol in sorted(Path(data_dir).glob(f"naca24??_re{tag}.pol")):
            code = pol.stem[4:8]
            tc = int(code[2:]) / 100.0
            out.setdefault(tc, {})[float(re)] = load_xfoil_polar(
                pol, name=f"NACA{code} Re{tag} (XFOIL)")
    return out


def polar_at_re(tc: float, re: float,
                data_dir: str | Path | None = None,
                bank: dict | None = None,
                blend_slope: bool = True) -> TablePolar:
    """The NACA 24XX member at thickness ``tc``, AT Reynolds number ``re``.

    Returns a real :class:`TablePolar` (so every consumer works unchanged),
    built by interpolating the bracketing bank members onto a common alpha
    grid: **log(cd) against log(Re)**, and cl / cm linearly.

    Why log-log for drag: both the laminar and the turbulent flat-plate
    skin-friction laws are power laws in Re (cf ~ Re^-1/2 and ~ Re^-1/5), so
    log cd is close to STRAIGHT against log Re while cd itself is convex and
    a linear interpolant sits above the truth everywhere between two members.
    This is the encoding the tools that do it well use (XFLR5 and RCAIDE bank
    and interpolate; NeuralFoil states the log-log argument explicitly), and
    it is measurably better here, not just tidier: over the spanwise probe's
    bank the worst interpolation error falls from 1.20 to 0.33 counts
    (results/spanwise_re_probe.json), and the residual is one-signed and
    positive, as a convex function under any interpolant must be. The local
    exponent is NOT constant over
    this range — it moves about -0.41 to -0.585 from 1e5 to 3e6 — which is
    why the bank is interpolated rather than replaced by one fitted power law.
    Lift and moment stay linear: they are not power laws in Re and the
    literature's own recommendation is linear in c_l.

    Exact at a bank Reynolds number: the member table is returned as-is, so a
    request at Re 1e6 is byte for byte the shipped polar and nothing that
    reaches this function by asking for the default point can drift.

    ``re`` outside the bank RAISES rather than extrapolating — an extrapolated
    viscous polar is a fabrication, and this is precisely the range where the
    laminar separation bubble makes the trend non-monotone (see the module
    note above: NACA 2418's fitted lift slope is not monotone in Re across
    this bank, which is a real property of the low-Re polar and not noise to
    be smoothed over).

    ``blend_slope`` (default True) states the LINEAR-REGION pair — ``a_lin``
    and ``alpha_L0`` — as the same interpolation of the two members' own
    values that everything else here gets, instead of re-fitting them from
    the blended table. This is the convention the module already declares:
    :class:`BlendedPolar` blends them, and :class:`BilinearPolar2D`'s
    docstring says so in as many words ("blended linearly, not re-extracted
    from the blended table"). ``polar_at_re`` was the only blend in the file
    that re-extracted, and because the blend is carried on the INTERSECTED
    alpha grid the re-fit ran in a window clipped by the *other* member's
    convergence range and left the convex hull of the two slopes it sits
    between — measured 2026-08-07 at t/c 0.06, members 6.98558 (Re 1e5) and
    6.87775 (Re 3e5): the re-fit gives 7.19660 at Re 2.9e5, above BOTH, on
    25 of 25 log-spaced samples across the segment, and is discontinuous
    across the bank node at Re 3e5 by 4.49 % of the value it approaches
    (equivalently 4.70 % of the node's own value — one gap, two
    denominators; everything else in the repo quotes the first).

    ``blend_slope=False`` restores the re-extraction bit-for-bit, and stays
    reachable for anyone re-deriving an old figure.

    UNTIL SESSION 63 this paragraph added that no published number could
    depend on the choice "for a stronger reason than 'no one reads the
    slope': this function has no production consumer in the package at all".
    **That premise is now false and is corrected here rather than left to be
    believed.** Two consumers arrived with the car families: ``carwing.py``'s
    flown-Reynolds opt-in, which feeds ``a_lin`` and ``alpha_L0`` straight
    into the VLM and therefore DOES read the blended slope, and
    ``carwing_multi.py``, which reads ``cd`` only (its two-element section
    fits its own linear pair from the panel-method lift curve, because no
    single-element table has that section's slope). The declared set is
    pinned by ``tests/test_api_config_promises.py``, which fails on a new
    undeclared consumer.

    The DEFAULT is unchanged, because the absence of consumers was never the
    argument for it — it was only the reason the choice was safe. The
    argument is the paragraph above: the re-extraction lands outside the
    convex hull of the two slopes it sits between, on 25 of 25 samples, and
    is discontinuous across a bank node by 4.49 %. That is as true with a
    consumer as without one. What has changed is that it is now
    load-bearing on one path, which is worth saying out loud.

    The Cp_min table travels too, when BOTH members carry one: it is
    interpolated on the same weight as cl. **Today no pair does** — the bank
    ships ``.cpmin`` at Re 1e6 only — so a blended polar still has no Cp_min
    and ``cp_min()`` still raises on one. This is the path becoming correct
    ahead of the data, not a hazard closed; it is gated on synthetic members
    and goes live the day the Cp_min bank widens.
    """
    bank = polar_re_bank(data_dir) if bank is None else bank
    if not bank:
        raise FileNotFoundError(
            "no NACA 24XX Reynolds bank on disk; run "
            "scripts/gen_polar_family_re.py")
    tc = float(tc)
    tcs = np.array(sorted(bank), dtype=float)
    i_tc = int(np.argmin(np.abs(tcs - tc)))
    if abs(float(tcs[i_tc]) - tc) > 1e-9:
        raise ValueError(
            f"t/c = {tc:.4f} is not a bank member ({tcs.tolist()}); blend in "
            "t/c with PolarFamily first, then ask for the Reynolds number")
    members = bank[float(tcs[i_tc])]
    res = np.array(sorted(members), dtype=float)
    re = float(re)
    if re < res[0] - 1e-9 or re > res[-1] + 1e-9:
        raise ValueError(
            f"Re = {re:.3g} outside the bank for t/c = {tc:.3f} "
            f"[{res[0]:.3g}, {res[-1]:.3g}] — no extrapolation of polar data")
    i_near = int(np.argmin(np.abs(res - re)))
    if abs(float(res[i_near]) - re) <= 1e-6 * max(re, 1.0):
        return members[float(res[i_near])]

    j = int(np.searchsorted(res, re))
    j = min(max(j, 1), res.size - 1)
    p0, p1 = members[float(res[j - 1])], members[float(res[j])]
    t = (np.log(re) - np.log(float(res[j - 1]))) / (
        np.log(float(res[j])) - np.log(float(res[j - 1])))
    # the common alpha grid is the INTERSECTION of the two tables' validity
    # windows: outside it one of the two marches did not converge, and
    # extending the grid there would silently duplicate an endpoint
    a_lo = max(p0.alpha_deg[0], p1.alpha_deg[0])
    a_hi = min(p0.alpha_deg[-1], p1.alpha_deg[-1])
    if not (a_hi > a_lo):
        raise ValueError(
            f"bracketing polars for t/c {tc:.3f} at Re {re:.3g} share no "
            "alpha range")
    grid = np.unique(np.concatenate([
        p0.alpha_deg[(p0.alpha_deg >= a_lo) & (p0.alpha_deg <= a_hi)],
        p1.alpha_deg[(p1.alpha_deg >= a_lo) & (p1.alpha_deg <= a_hi)]]))
    cd0, cd1 = np.asarray(p0.cd(grid)), np.asarray(p1.cd(grid))
    if np.all(cd0 > 0.0) and np.all(cd1 > 0.0):
        cd = np.exp((1.0 - t) * np.log(cd0) + t * np.log(cd1))
    else:                       # a non-positive tabulated cd is a solver
        cd = (1.0 - t) * cd0 + t * cd1      # artefact; do not take its log
    # the Cp_min table, when BOTH members carry one — same weight as cl.
    # Interpolated onto the intersection of their OWN cpmin grids, which is
    # not the cl/cd grid: the two tables come from different XFOIL passes.
    cp_grid = cp_vals = None
    if p0.has_cp_min and p1.has_cp_min:
        c_lo = max(p0.alpha_cpmin[0], p1.alpha_cpmin[0])
        c_hi = min(p0.alpha_cpmin[-1], p1.alpha_cpmin[-1])
        if c_hi > c_lo:
            cp_grid = np.unique(np.concatenate([
                p0.alpha_cpmin[(p0.alpha_cpmin >= c_lo)
                               & (p0.alpha_cpmin <= c_hi)],
                p1.alpha_cpmin[(p1.alpha_cpmin >= c_lo)
                               & (p1.alpha_cpmin <= c_hi)]]))
            cp_vals = ((1.0 - t) * np.asarray(p0.cp_min(cp_grid))
                       + t * np.asarray(p1.cp_min(cp_grid)))
    return TablePolar(
        alpha_deg=grid,
        CL=(1.0 - t) * p0.cl(grid) + t * p1.cl(grid),
        CD=cd,
        CM=(1.0 - t) * p0.cm(grid) + t * p1.cm(grid),
        name=f"NACA24{int(round(tc * 100)):02d} Re{re:.3g} (log-log Re blend)",
        Re=re,
        alpha_cpmin=cp_grid,
        CPMIN=cp_vals,
        # the linear-region pair, blended rather than re-fitted on a grid the
        # other member clipped (see the docstring, and the module convention
        # BlendedPolar / BilinearPolar2D already state)
        a_lin_override=((1.0 - t) * p0.a_lin + t * p1.a_lin
                        if blend_slope else None),
        alpha_L0_override=((1.0 - t) * p0.alpha_L0 + t * p1.alpha_L0
                           if blend_slope else None),
    )


def polar_family_at_re(re: float, data_dir: str | Path | None = None
                       ) -> PolarFamily:
    """:class:`PolarFamily` over t/c, with every member flown at ``re``.

    The drop-in Reynolds-aware twin of :func:`default_polar_family`. Members
    whose bank does not cover ``re`` are omitted; fewer than two survivors
    raises, because a family cannot blend in t/c without a bracket.

    ``polar_family_at_re(1e6)`` returns the shipped tables themselves — the
    identity, asserted in tests/test_polar_re_bank.py.
    """
    bank = polar_re_bank(data_dir)
    members: dict[float, TablePolar] = {}
    for tc in sorted(bank):
        try:
            members[tc] = polar_at_re(tc, re, bank=bank)
        except ValueError:
            continue        # this t/c has no bank coverage at this Re
    if len(members) < 2:
        raise ValueError(
            f"the NACA 24XX Reynolds bank covers fewer than 2 thicknesses at "
            f"Re = {re:.3g}; run scripts/gen_polar_family_re.py or stay inside "
            f"{[t for t, _ in RE_BANK_TAGS]}")
    return PolarFamily(members)


# ---------------------------------------------------------------------------
# Wide-alpha ("stall") polar family — scripts/gen_stall_polar_family.py.
# ADDITIVE to this module: nothing above this banner is touched. The cruise
# family loaded by default_polar_family() stops at alpha = +14 deg, below the
# viscous cl peak of every NACA 24XX section at Re = 1e6, so cl_max is
# unobservable there. The stall family repeats the SAME XFOIL solve
# (Re = 1e6, M = 0, Ncrit = 9, same paneling recipe, sweep split at alpha = 0)
# out to alpha = +22 deg in separate naca24XX_re1e6_stall.pol files — the
# glob above ("naca24??_re1e6.pol") matches on the exact "_re1e6.pol" suffix
# and therefore never picks these up.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StallPoint:
    """Maximum-lift summary of ONE wide-alpha section polar.

    ``cl_max`` / ``alpha_stall_deg`` are the largest TABULATED cl and the
    alpha at which it occurs. When ``censored`` is True the viscous march
    ended while cl was still climbing (XFOIL silently omits alphas whose
    Newton/BL solve failed, and separated flow past stall is exactly where
    that happens), so those two numbers are a LOWER BOUND on the true peak,
    not the peak itself — a censored member must not be used to build a wing
    CL_max. ``alpha_max_deg`` is the last converged alpha in the table and
    ``n_points`` its converged-row count.
    """

    tc: float
    cl_max: float
    alpha_stall_deg: float
    n_points: int
    alpha_max_deg: float
    censored: bool


def stall_point(polar: TablePolar, tc: float = float("nan"),
                drop_tol: float = 0.01) -> StallPoint:
    """Locate the cl peak of a wide-alpha section polar (see StallPoint).

    The peak is accepted as a genuine stall only if the table CONTINUES past
    it and cl drops at least ``drop_tol`` below the maximum at some larger
    alpha; a table whose maximum sits at (or within ``drop_tol`` of) its last
    converged row is flagged ``censored`` — the march simply ran out, which
    on thin sections happens well before the section actually stalls.
    """
    a = np.asarray(polar.alpha_deg, float)
    cl = np.asarray(polar.CL, float)
    i = int(np.argmax(cl))
    cl_max = float(cl[i])
    past = cl[i + 1:]
    confirmed = bool(past.size and float(past.min()) <= cl_max - drop_tol)
    return StallPoint(
        tc=float(tc),
        cl_max=cl_max,
        alpha_stall_deg=float(a[i]),
        n_points=int(a.size),
        alpha_max_deg=float(a[-1]),
        censored=not confirmed,
    )


def stall_points(family: PolarFamily, drop_tol: float = 0.01
                 ) -> dict[float, StallPoint]:
    """``{t/c: StallPoint}`` for every member of a wide-alpha family."""
    return {tc: stall_point(p, tc=tc, drop_tol=drop_tol)
            for tc, p in family.members.items()}


def usable_tc_floor(family: PolarFamily, drop_tol: float = 0.01) -> float | None:
    """Thinnest member t/c whose cl peak is actually resolved (not censored).

    Members below this floor have no usable cl_max: at Re = 1e6 the NACA 2406
    boundary layer diverges before the section stalls, so its table is
    right-censored no matter how far the requested sweep goes. Returns None if
    NO member resolved a peak.
    """
    ok = [tc for tc, sp in stall_points(family, drop_tol).items() if not sp.censored]
    return min(ok) if ok else None


def stall_polar_family(data_dir: str | Path | None = None) -> PolarFamily:
    """Wide-alpha NACA 24XX XFOIL family at Re 1e6 (alpha -6 ... +22 deg).

    Loads every data/airfoils/naca24XX_re1e6_stall.pol present (generated by
    scripts/gen_stall_polar_family.py with a local XFOIL binary — real
    polars, never fabricated). Same conditions as default_polar_family(), so
    the two are commensurable on the overlapping alpha range; unlike the
    cruise family this one includes the t/c = 0.12 member (NACA 2412), which
    default_polar() uses as the Tier A default section. Raises
    FileNotFoundError if fewer than 2 members are on disk.

    Note the members may still be right-censored in cl (thin sections); use
    stall_points() / usable_tc_floor() before treating a member's peak as a
    cl_max.
    """
    if data_dir is None:
        data_dir = Path(__file__).resolve().parents[2] / "data" / "airfoils"
    members: dict[float, TablePolar] = {}
    for pol in sorted(Path(data_dir).glob("naca24??_re1e6_stall.pol")):
        code = pol.stem[4:8]
        tc = int(code[2:]) / 100.0
        members[tc] = load_xfoil_polar(pol, name=f"NACA{code} Re1e6 stall (XFOIL)")
    if len(members) < 2:
        raise FileNotFoundError(
            f"need >= 2 naca24XX_re1e6_stall.pol files in {data_dir}; "
            "run scripts/gen_stall_polar_family.py"
        )
    return PolarFamily(members)


# ---------------------------------------------------------------------------
# The wide-alpha family AT A REYNOLDS NUMBER (session 42, item 3a).
#
# Everything above this banner is untouched. `stall_polar_family` still loads
# exactly the Re-1e6 members and `stall.section_clmax` still reads them, so
# every published stall number is bit-for-bit what it was.
#
# What this adds is the OTHER half of the correction session 41 shipped for
# drag. `SectionWingProblem(re_strip="bank")` looks each strip's cd up at that
# strip's own Reynolds number; the cl_max CEILING that the same strip is
# measured against is still read off Re 1e6. The literature review's topic C
# reports the per-strip cl_max error as the LARGER of the two effects (13-36 %
# per strip against 0.3-2.2 % on integrated C_D) and records that no published
# number exists for a strip code -- so the bank has to exist before the
# question can even be asked.
# ---------------------------------------------------------------------------


def stall_re_bank(data_dir: str | Path | None = None
                  ) -> dict[float, dict[float, TablePolar]]:
    """``{t/c: {Re: TablePolar}}`` of the WIDE-ALPHA family on disk.

    The `_stall` twin of :func:`polar_re_bank`, and ragged in exactly the same
    way: a (t/c, Re) cell whose XFOIL march failed is simply ABSENT, never
    faked. Low Reynolds numbers fail on the thin members, which is a fact
    about the boundary layer and not a gap to be filled in.
    """
    if data_dir is None:
        data_dir = Path(__file__).resolve().parents[2] / "data" / "airfoils"
    out: dict[float, dict[float, TablePolar]] = {}
    for re, tag in RE_BANK_TAGS:
        for pol in sorted(Path(data_dir).glob(f"naca24??_re{tag}_stall.pol")):
            code = pol.stem[4:8]
            tc = int(code[2:]) / 100.0
            out.setdefault(tc, {})[float(re)] = load_xfoil_polar(
                pol, name=f"NACA{code} Re{tag} stall (XFOIL)")
    return out


def clmax_at_re(tc: float, re: float,
                data_dir: str | Path | None = None,
                bank: dict | None = None,
                drop_tol: float = 0.01) -> float:
    """The section cl_max at thickness ``tc`` AND Reynolds number ``re``.

    Interpolation, and why it is not the log-log rule `polar_at_re` uses for
    drag:

    * in **t/c**, linearly between the two bracketing members, exactly as
      :meth:`PolarFamily.at` and :func:`stall.section_clmax` already do;
    * in **Re**, linearly in ``log(Re)``. cl_max is not a power law in
      Reynolds number -- it is a saturating curve set by where the laminar
      separation bubble bursts -- so the log-log encoding that is right for
      skin friction has no basis here. Linear-in-log-Re is what the
      measurement literature plots cl_max against and what XFLR5's batch
      analysis blends, and over this bank it is what the data look like.

    **Censored members are excluded at every Reynolds number separately.** A
    member whose march stopped while cl was still climbing reports a LOWER
    BOUND (:class:`StallPoint`), and the set of censored members is different
    at each Re -- thin sections censor first and censor at higher Re than
    thick ones. Blending a censored cell in would drag the ceiling down by an
    amount that is an artefact of XFOIL's convergence, not of the section.

    Raises ``ValueError`` rather than extrapolating outside the bank's
    resolved coverage, in either variable. That refusal is the point: a
    ceiling nobody measured is worse than no ceiling.
    """
    bank = stall_re_bank(data_dir) if bank is None else bank
    if not bank:
        raise ValueError(
            "the wide-alpha Reynolds bank is empty; run "
            "scripts/gen_stall_polar_family.py --re-grid")
    re = float(re)

    # {Re: {t/c: cl_max}} over the RESOLVED cells only
    by_re: dict[float, dict[float, float]] = {}
    for tc_m, cells in bank.items():
        for re_m, pol in cells.items():
            sp = stall_point(pol, tc=tc_m, drop_tol=drop_tol)
            if not sp.censored:
                by_re.setdefault(float(re_m), {})[float(tc_m)] = sp.cl_max

    def _at_tc(cl_by_tc: dict[float, float]) -> float | None:
        """Linear in t/c inside the resolved members at ONE Reynolds number."""
        anchors = np.array(sorted(cl_by_tc), dtype=float)
        if anchors.size == 0:
            return None
        if float(tc) < anchors[0] - 1e-12 or float(tc) > anchors[-1] + 1e-12:
            return None
        vals = np.array([cl_by_tc[float(t)] for t in anchors], dtype=float)
        return float(np.interp(float(tc), anchors, vals))

    # Reynolds numbers at which THIS thickness is resolved. A bank member that
    # censors at 1e5 but not at 1e6 must not silently anchor the blend.
    usable = sorted(r for r in by_re if _at_tc(by_re[r]) is not None)
    if not usable:
        raise ValueError(
            f"no Reynolds number in the wide-alpha bank resolves a cl peak at "
            f"t/c = {float(tc):.4f}; the members that cover it are censored")
    lo, hi = usable[0], usable[-1]
    if re < lo * (1.0 - 1e-9) or re > hi * (1.0 + 1e-9):
        raise ValueError(
            f"Re = {re:.4g} is outside the wide-alpha bank's resolved range "
            f"[{lo:.4g}, {hi:.4g}] at t/c = {float(tc):.4f}; the bank refuses "
            "rather than extrapolating a stall ceiling")
    xs = np.log(np.array(usable, dtype=float))
    ys = np.array([_at_tc(by_re[r]) for r in usable], dtype=float)
    return float(np.interp(np.log(re), xs, ys))
