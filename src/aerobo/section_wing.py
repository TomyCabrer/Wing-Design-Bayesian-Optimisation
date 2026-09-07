"""Tier B section shape + twist law co-design for a GUESSED wing.

The standalone section problem (:mod:`aerobo.airfoil`) minimises 2-D profile
drag at a design lift coefficient the caller types in. That number is not
free: on a real aircraft it follows from how much wing you guessed you would
need. This module closes that loop — the caller states the wing they are
sizing for (mass, speed, altitude, a REFERENCE-AREA GUESS, aspect ratio,
taper) and the design lift coefficient is DERIVED from it,

    CL_design = W / (q S) ,   W = m g0 ,   q = 1/2 rho(h) V^2

with the section Reynolds number taken at the mean aerodynamic chord,
Re = rho V mac / mu (Sutherland), so the polar is flown at the Reynolds
number that wing actually produces instead of an independently typed one.

Because a wing is now in play, twist becomes a design variable rather than
a fixed operating point. The twist law is a polynomial in the normalised
semi-span coordinate eta = |2y/b|, ANCHORED AT THE ROOT:

    theta(eta) = sum_{j=1..k} t_j eta^j        [deg, + = nose-up]

so ``twist_order`` k = 1 is the classic linear law (t_1 IS the tip twist,
washout is negative) and k = 2, 3, 4 add quadratic .. quartic freedom. The
root incidence is deliberately NOT a design variable: it is the trim unknown
solved for so the wing makes exactly CL_design (llt.solve_llt_trim).

The CHORD distribution is a design variable on the same footing. The taper
ratio the caller guesses fixes a straight-taper baseline
c_trap(eta) = c_root (1 - (1 - lambda) eta), which the design vector then
reshapes by a polynomial MULTIPLIER anchored at the root,

    c(eta) = A * c_trap(eta) * (1 + sum_{j=1..m} k_j eta^j)

with the scalar A chosen so the planform integrates back to the SAME area
(see :func:`chord_law`). Area must be held because it is the area guess that
set CL_design in the first place: a chord law free to change S would silently
move the design point it is being scored at. ``chord_order`` m = 0 is the
fixed straight taper (the legacy planform, reproduced bit-for-bit), and
m = 1, 2, 3 buy linear .. cubic reshaping — enough to approach the elliptic
loading a single taper ratio cannot reach, since c_trap is linear in eta for
every lambda and elliptic chord is not.

Two honest limits on that freedom, both reported rather than assumed away:
the section polar is flown at ONE Reynolds number, the one the BASELINE mean
aerodynamic chord implies, so a reshaped planform's true MAC (and the Re it
would imply) are reported as ``mac_true`` / ``re_true`` beside the ``re``
actually flown; and the deviation from straight taper is capped by the
``chord_max_frac`` constraint, which is what keeps that single-Re assumption
from being stretched past the point it means anything.

Objective (MAXIMISE): the WING lift-to-drag ratio at that trimmed state,

    L/D = CL / (CDi + CDp)
    CDi = pi AR sum_n n A_n^2                (lifting line, llt.py)
    CDp = (1/S) int cd(cl(y)) c(y) dy        (strip integral of the SAME
                                              viscous XFOIL polar the section
                                              problem scores with)

which is why twist is nearly free to explore: the polar depends only on the
eight CST weights, so every twist variation of one shape reuses the cached
XFOIL run.

SCOPE / honest labels
---------------------
* WING-ONLY L/D. No fuselage, tail, nacelle or interference drag: CDp is the
  strip integral of section profile drag and CDi is the lifting-line induced
  drag, nothing else. It is an upper bound on the aircraft L/D, and is
  labelled "wing L/D" everywhere it is reported.
* The lifting line is LINEAR: the real polar is reduced to a section slope
  ``a`` [1/rad] and zero-lift angle ``alpha_L0`` by least squares over a
  ``SLOPE_FIT_CL_BAND`` window of the pre-stall branch around CL_design.
  Nonlinear cl(alpha) curvature is therefore not carried into the loading —
  the strip cd(cl) lookup IS the real polar, the loading that feeds it is
  linearised.
* Stations whose local cl falls BELOW the converged polar branch (the tips,
  where cl -> 0) have their cd clamped to the branch endpoint. This is the
  standard flat-bucket strip approximation; the number of clamped stations
  is reported as ``n_clamped_low`` and never hidden. Stations ABOVE the
  branch are LOCAL STALL, where a linear lifting line is simply invalid —
  those designs FAIL the documented penalty contract rather than returning a
  plausible-looking L/D.
* The lifting line normalises on its own trapezoidal area over the midpoint
  cosine grid, which under-reads the exact trapezoidal S by O(1/N^2). Both
  are reported (``S_ref`` requested vs ``S_llt`` flown); the default
  ``n_stations`` keeps the gap below ~0.2%.

Failure contract: identical to :mod:`aerobo.airfoil` — any geometry/solver
failure returns the finite ``(PENALTY, [G_FAIL] * n_constraints)`` pair via
:func:`fg_section_wing`, never NaN and never an exception to the optimiser.
XfoilError (a broken install) still propagates: silently scoring every
design -100 would fabricate a study.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .airfoil import (
    G_FAIL,
    PENALTY,
    AirfoilProblem,
    _longest_increasing_run,
    _surface_y,
    cst_coords,
    cst_thickness,
    norm_margin,
)
from .llt import cosine_stations, solve_llt_trim
from .mission import isa_density, isa_temperature, sutherland_mu

G0 = 9.80665              # m/s^2, standard gravity (weight from mass)

#: alpha sweep for the wing mode. Wider than the section default
#: (arange(-4, 10.5, 0.5)) at BOTH ends: the tip stations of a trimmed wing
#: run down to cl ~ 0, which a cambered section only reaches at negative
#: alpha, and the root stations of a washed-out wing sit above the section
#: design cl. A branch that brackets the whole spanwise cl range is what
#: keeps the strip integral out of the clamped regime.
WING_ALPHAS = np.arange(-8.0, 14.5, 0.5)

#: half-width [cl] of the window the section slope/alpha_L0 line is fitted
#: over, centred on CL_design. Wide enough for a well-conditioned fit,
#: narrow enough to linearise where the wing actually operates.
SLOPE_FIT_CL_BAND = 0.45
SLOPE_FIT_MIN_PTS = 4     # below this the whole pre-stall branch is used

#: polynomial twist orders offered. 1 = linear (t_1 is the tip twist).
TWIST_ORDERS = (1, 2, 3, 4)
TWIST_ORDER_NAMES = {1: "linear", 2: "quadratic", 3: "cubic", 4: "quartic"}

#: polynomial CHORD orders offered. 0 = the straight-taper baseline is left
#: alone (no chord design variables at all), which is the legacy planform.
CHORD_ORDERS = (0, 1, 2, 3)
CHORD_ORDER_NAMES = {0: "fixed (straight taper)", 1: "linear",
                     2: "quadratic", 3: "cubic"}

#: largest chord deviation the cap may be set to. Kept below 1 so that any
#: FEASIBLE chord law still has a strictly positive chord everywhere.
CHORD_MAX_FRAC_LIMIT = 0.9

#: multiplier below which the planform is not a wing any more. A design whose
#: chord law collapses a station to (near) zero chord cannot be flown at all,
#: so it returns the penalty contract rather than a plausible-looking L/D.
CHORD_MULT_FLOOR = 0.05

#: trim bracket for the root incidence search [deg]. Deliberately wider than
#: any sane alpha_max_deg so the bracket failure means "this wing cannot make
#: CL_design at all", not "the cap was hit" (the cap is a CONSTRAINT, g3).
TRIM_BRACKET_DEG = (-15.0, 22.0)


# ------------------------------------------------------------------ the wing


@dataclass
class WingGuess:
    """The wing the section is being designed FOR — the surface-area guess.

    Everything the section needs that is NOT section geometry: the design
    lift coefficient (from weight and the area guess), the planform the
    lifting line loads, and the Reynolds number at the mean aerodynamic
    chord. Trapezoidal planform, unswept, symmetric.

    Fields
    ------
    mass_kg        design mass [kg]; weight W = m g0
    v_ms           cruise speed [m/s]
    altitude_m     altitude [m] (ISA via mission.py, 0-20 km)
    s_ref_m2       REFERENCE-AREA GUESS [m^2] — the knob this module exists
                   for: it sets both CL_design = W/(qS) and the planform.
    aspect_ratio   b^2/S, so b = sqrt(AR S)
    taper          tip chord / root chord, in (0, 1]
    n_stations     lifting-line stations across the FULL span
    """

    mass_kg: float = 60.0
    v_ms: float = 20.0
    altitude_m: float = 0.0
    s_ref_m2: float = 6.0
    aspect_ratio: float = 10.0
    taper: float = 0.6
    n_stations: int = 61

    def __post_init__(self):
        if not (self.mass_kg > 0 and self.v_ms > 0 and self.s_ref_m2 > 0
                and self.aspect_ratio > 0):
            raise ValueError(
                "mass_kg, v_ms, s_ref_m2 and aspect_ratio must all be > 0 "
                f"(got {self.mass_kg}, {self.v_ms}, {self.s_ref_m2}, "
                f"{self.aspect_ratio})")
        if not (0.0 < self.taper <= 1.0):
            raise ValueError(f"taper must be in (0, 1], got {self.taper}")
        if int(self.n_stations) < 11:
            raise ValueError(
                f"n_stations must be >= 11, got {self.n_stations}")

    # --- flow state -----------------------------------------------------
    @property
    def rho(self) -> float:
        """Density [kg/m^3] — ISA at ``altitude_m``."""
        return isa_density(self.altitude_m)

    @property
    def mu(self) -> float:
        """Dynamic viscosity [Pa s] — Sutherland at the ISA temperature."""
        return sutherland_mu(isa_temperature(self.altitude_m))

    @property
    def q(self) -> float:
        """Dynamic pressure 1/2 rho V^2 [Pa]."""
        return 0.5 * self.rho * self.v_ms ** 2

    @property
    def weight_n(self) -> float:
        """Design weight W = m g0 [N]."""
        return self.mass_kg * G0

    @property
    def cl_design(self) -> float:
        """Design lift coefficient CL = W/(q S) — the derived target."""
        return self.weight_n / (self.q * self.s_ref_m2)

    # --- planform -------------------------------------------------------
    @property
    def b(self) -> float:
        """Span [m] = sqrt(AR S)."""
        return float(np.sqrt(self.aspect_ratio * self.s_ref_m2))

    @property
    def S(self) -> float:
        """Reference area [m^2] (alias of ``s_ref_m2``)."""
        return float(self.s_ref_m2)

    @property
    def c_root(self) -> float:
        """Root chord [m] of the trapezoid with this area and taper."""
        return 2.0 * self.s_ref_m2 / (self.b * (1.0 + self.taper))

    @property
    def mac(self) -> float:
        """Mean aerodynamic chord [m] (trapezoidal closed form)."""
        lam = self.taper
        return (2.0 / 3.0) * self.c_root * (1 + lam + lam ** 2) / (1 + lam)

    @property
    def re_mac(self) -> float:
        """Section Reynolds number at the MAC — rho V mac / mu."""
        return self.rho * self.v_ms * self.mac / self.mu

    def stations(self, chord_coeffs=()
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(y, chord, eta) at the ``n_stations`` midpoint-cosine LLT nodes.

        ``chord_coeffs`` reshapes the straight-taper baseline by the
        area-preserving polynomial multiplier of :func:`chord_law`. Empty (the
        default) returns the trapezoid untouched, and an all-zero coefficient
        vector returns it bit-for-bit, because the normalising factor is then
        a ratio of two identical integrals.

        Raises ValueError when the law drives the chord to (near) zero — see
        CHORD_MULT_FLOOR; callers map that to the penalty contract.
        """
        _theta, y = cosine_stations(int(self.n_stations), self.b)
        eta = np.abs(2.0 * y / self.b)
        c = self.c_root * (1.0 - (1.0 - self.taper) * eta)
        coeffs = np.asarray(chord_coeffs, dtype=float).ravel()
        if coeffs.size:
            c = chord_law(eta, c, y, coeffs)
        return y, c, eta

    def to_dict(self) -> dict:
        """JSON-safe inputs + everything derived from them (the GUI view)."""
        return {
            "mass_kg": float(self.mass_kg), "v_ms": float(self.v_ms),
            "altitude_m": float(self.altitude_m),
            "s_ref_m2": float(self.s_ref_m2),
            "aspect_ratio": float(self.aspect_ratio),
            "taper": float(self.taper), "n_stations": int(self.n_stations),
            "rho": float(self.rho), "q": float(self.q),
            "weight_n": float(self.weight_n),
            "cl_design": float(self.cl_design), "b": float(self.b),
            "c_root": float(self.c_root), "mac": float(self.mac),
            "re_mac": float(self.re_mac),
        }


# ----------------------------------------------------------------- the twist


def twist_deg(eta, coeffs) -> np.ndarray:
    """Twist law theta(eta) = sum_j coeffs[j-1] eta^j  [deg].

    Local incidence RELATIVE TO THE ROOT (llt.solve_llt_trim convention,
    + = nose-up), hence theta(0) = 0 by construction and washout is negative.
    With one coefficient this is the linear law and ``coeffs[0]`` is the tip
    twist; higher orders bend the law between root and tip without adding a
    root degree of freedom (that is the trim unknown).
    """
    eta = np.asarray(eta, dtype=float)
    out = np.zeros_like(eta, dtype=float)
    for j, cj in enumerate(coeffs, start=1):
        out = out + float(cj) * eta ** j
    return out


def twist_envelope_deg(coeffs, n: int = 401) -> float:
    """max |theta(eta)| over the semi-span — the "max twist angle" measured.

    Sampled on a uniform eta grid rather than assumed at the tip: for k > 1
    the extremum of the polynomial can sit inboard, and bounding only the
    coefficients would let the true envelope exceed the cap.
    """
    if len(coeffs) == 0:
        return 0.0
    return float(np.max(np.abs(twist_deg(np.linspace(0.0, 1.0, n), coeffs))))


def twist_bounds(order: int, twist_max_deg: float) -> np.ndarray:
    """(order, 2) box for the twist coefficients: each in +/- twist_max_deg.

    The box alone does NOT bound the envelope for order > 1 (coefficients can
    add up); that is what the g_twist constraint is for. Bounding each
    coefficient by the cap instead of cap/order keeps a pure high-order law
    reachable at full amplitude.
    """
    order = int(order)
    if order not in TWIST_ORDERS:
        raise ValueError(
            f"twist_order must be one of {TWIST_ORDERS}, got {order}")
    t = float(twist_max_deg)
    if not (t > 0.0):
        raise ValueError(f"twist_max_deg must be > 0, got {twist_max_deg}")
    return np.tile([-t, t], (order, 1)).astype(float)


# ---------------------------------------------------------------- the chord


def chord_multiplier(eta, coeffs) -> np.ndarray:
    """1 + sum_j coeffs[j-1] eta^j — the chord law BEFORE area normalisation.

    Anchored at 1 at the root by construction, exactly as the twist law is
    anchored at 0 there: the root chord is set by the area guess and the taper
    ratio, so the design vector reshapes the planform OUTBOARD of the root
    rather than re-deriving a number the caller already gave.
    """
    eta = np.asarray(eta, dtype=float)
    out = np.ones_like(eta, dtype=float)
    for j, kj in enumerate(coeffs, start=1):
        out = out + float(kj) * eta ** j
    return out


def chord_law(eta, c_trap, y, coeffs) -> np.ndarray:
    """Straight-taper chord reshaped by ``coeffs``, at UNCHANGED area.

    The multiplier alone would change the planform area, and with it
    CL_design = W/(qS) — the design point the whole run is scored at. So the
    reshaped chord is rescaled by the ratio of the two trapezoidal integrals
    over the SAME station grid the lifting line normalises on, which pins the
    flown area to the baseline's to machine precision (and exactly, when the
    coefficients are zero: the two integrals are then computed from identical
    arrays).

    Raises ValueError if the law collapses the chord (multiplier at or below
    CHORD_MULT_FLOOR) or integrates to a non-positive area.
    """
    mult = chord_multiplier(eta, coeffs)
    if float(np.min(mult)) <= CHORD_MULT_FLOOR:
        raise ValueError(
            f"chord law collapses the chord (min multiplier "
            f"{float(np.min(mult)):.4f} <= {CHORD_MULT_FLOOR})")
    c = np.asarray(c_trap, dtype=float) * mult
    a_ref = float(np.trapezoid(c_trap, y))
    a_new = float(np.trapezoid(c, y))
    if not (a_new > 0.0) or not np.isfinite(a_new) or not (a_ref > 0.0):
        raise ValueError("chord law integrates to a non-positive area")
    return c * (a_ref / a_new)


def chord_dev_envelope(c, c_trap) -> float:
    """max |c/c_trap - 1| over the span — "how far from straight taper".

    Measured on the FLOWN chord (after area normalisation) against the
    baseline trapezoid, so it is the deviation the user can actually see in
    the planform rather than the raw coefficient sum.
    """
    c = np.asarray(c, dtype=float)
    c_trap = np.asarray(c_trap, dtype=float)
    if c.shape != c_trap.shape or not np.all(c_trap > 0.0):
        raise ValueError("chord arrays must match and the baseline be > 0")
    return float(np.max(np.abs(c / c_trap - 1.0)))


def chord_bounds(order: int, chord_max_frac: float) -> np.ndarray:
    """(order, 2) box for the chord coefficients: each in +/- chord_max_frac.

    As with the twist box, the coefficients can add up, so the box alone does
    NOT bound the deviation envelope — that is the g_chord constraint's job.
    Order 0 returns an empty (0, 2) box: no chord design variables.
    """
    order = int(order)
    if order not in CHORD_ORDERS:
        raise ValueError(
            f"chord_order must be one of {CHORD_ORDERS}, got {order}")
    f = float(chord_max_frac)
    if not (0.0 < f <= CHORD_MAX_FRAC_LIMIT):
        raise ValueError(
            f"chord_max_frac must be in (0, {CHORD_MAX_FRAC_LIMIT}], got {f}")
    if order == 0:
        return np.zeros((0, 2), dtype=float)
    return np.tile([-f, f], (order, 1)).astype(float)


def mac_of(c, y) -> float:
    """Mean aerodynamic chord of an arbitrary chord distribution [m].

    int c^2 dy / int c dy — the general definition the trapezoidal closed
    form in :attr:`WingGuess.mac` is the straight-taper special case of.
    """
    c = np.asarray(c, dtype=float)
    y = np.asarray(y, dtype=float)
    den = float(np.trapezoid(c, y))
    if not (den > 0.0):
        raise ValueError("non-positive planform area")
    return float(np.trapezoid(c * c, y) / den)


def section_line_fit(cl_b: np.ndarray, alpha_deg_b: np.ndarray,
                     cl_design: float) -> tuple[float, float]:
    """(a [1/rad], alpha_L0 [rad]) least-squares line through the real polar.

    Fitted over the pre-stall branch points within ``SLOPE_FIT_CL_BAND`` of
    ``cl_design`` (falling back to the whole branch when that window holds
    fewer than ``SLOPE_FIT_MIN_PTS`` points), because the lifting line is
    linear and should be linearised WHERE THE WING FLIES rather than over an
    arbitrary sweep. Raises ValueError on a non-positive slope (an
    unusable/garbage polar), which the caller maps to the penalty contract.
    """
    al = np.deg2rad(np.asarray(alpha_deg_b, dtype=float))
    cl = np.asarray(cl_b, dtype=float)
    m = np.abs(cl - float(cl_design)) <= SLOPE_FIT_CL_BAND
    if int(m.sum()) < SLOPE_FIT_MIN_PTS:
        m = np.ones_like(cl, dtype=bool)
    a, b0 = np.polyfit(al[m], cl[m], 1)
    if not (a > 0.0) or not np.isfinite(a) or not np.isfinite(b0):
        raise ValueError("non-positive / non-finite section lift-curve slope")
    return float(a), float(-b0 / a)


# --------------------------------------------------------------- the problem


@dataclass
class SectionWingProblem:
    """CST section + twist law + chord law, scored on WING L/D at derived CL.

    Design vector (d = 2 n_cst + twist_order + chord_order = 8 + k + m):

        x = [w_upper_0..w_upper_3, w_lower_0..w_lower_3,
             t_1 .. t_k,                    twist-law coefficients [deg]
             k_1 .. k_m]                    chord-law coefficients [-]

    Objective (MAXIMISE): f = L/D of the trimmed wing (module docstring).

    Constraints (signed margins, all normalised O(1), feasible iff >= 0):

        g0 = (t/c - tc_min) / tc_min                 section depth
        g1 = (cm_max - |cm(CL_design)|) / cm_max     trim-drag proxy
        g2 = (twist_max_deg - max|theta|) / twist_max_deg
             the user's "max twist angle": the washout envelope over the
             whole semi-span, not just the tip (see twist_envelope_deg).
        g3 = (alpha_max_deg - max|alpha_root + theta|) / alpha_max_deg
             the user's "max angle": the largest LOCAL GEOMETRIC incidence
             anywhere on the span at the trimmed state. Capping the root
             alone would let a strongly twisted law hide an inboard or
             outboard station at an unflyable incidence.
        g4 = (chord_max_frac - max|c/c_trap - 1|) / chord_max_frac
             ONLY when chord_order >= 1: how far the flown planform may
             depart from the straight-taper baseline. Bounds the error in
             the single-Reynolds-number section polar (module docstring) and
             keeps the chord strictly positive.

    With ``chord_order = 0`` (the default) there are no chord design
    variables, four constraints, and the planform is the legacy trapezoid
    bit-for-bit.

    ``airfoil`` supplies the section box (its ``anchor`` is the base seed
    from the library screen), the two section gates and the panelling; its
    ``re`` / ``cl_design`` / ``alphas`` are OVERRIDDEN from the wing, because
    in this mode they are derived quantities, not inputs. The resulting
    problem is exposed as ``.section`` so every CST-aware helper
    (api._section_weights, section_report, …) keeps working unchanged.
    """

    airfoil: AirfoilProblem = field(default_factory=AirfoilProblem)
    wing: WingGuess = field(default_factory=WingGuess)
    twist_order: int = 1
    twist_max_deg: float = 6.0
    alpha_max_deg: float = 10.0
    chord_order: int = 0
    chord_max_frac: float = 0.5
    alphas: np.ndarray | None = None      # None -> WING_ALPHAS
    #: HOW each spanwise strip's profile drag gets its Reynolds number.
    #:
    #: * ``"mac"`` — the legacy path and the default: one polar at the mean
    #:   aerodynamic chord's Re, every strip looked up in it.
    #: * ``"power"`` — Drela's strip correction, ``cd * (Re_y/Re_mac)**n``
    #:   with n = -0.5 below Re 2e5 (laminar) and -0.2 above (turbulent), the
    #:   split QPROP/XROTOR ship. **Zero extra XFOIL.** Measured against the
    #:   bank below it recovers most of the integrated error for nothing.
    #: * ``"bank"`` — ``re_bank`` real XFOIL sweeps log-spaced over
    #:   [Re(tip), Re(root)], cd interpolated log-log between them. The
    #:   expensive, faithful one.
    #:
    #: NONE of them fixes ``cl_max``: the local stall ceiling moves with Re
    #: too, and this correction is applied to drag alone, so the stall gate
    #: stays optimistic at the tip in every mode. Said here because the
    #: literature says it and it would otherwise be an unstated limitation.
    re_strip: str = "mac"
    re_bank: int = 1                      # sweeps, when re_strip == "bank"

    def __post_init__(self):
        if int(self.twist_order) not in TWIST_ORDERS:
            raise ValueError(
                f"twist_order must be one of {TWIST_ORDERS}, "
                f"got {self.twist_order}")
        if int(self.re_bank) < 1:
            raise ValueError(
                f"re_bank must be >= 1 (1 = the single-Re lookup), "
                f"got {self.re_bank}")
        self.re_bank = int(self.re_bank)
        if self.re_strip not in RE_STRIP_MODES:
            raise ValueError(
                f"unknown re_strip mode {self.re_strip!r}; "
                f"choose from {list(RE_STRIP_MODES)}")
        if self.re_strip == "bank" and self.re_bank < 2:
            raise ValueError(
                "re_strip='bank' needs re_bank >= 2 XFOIL sweeps to "
                "interpolate between; re_bank=1 IS the 'mac' mode")
        if self.re_strip != "bank" and self.re_bank != 1:
            raise ValueError(
                f"re_bank={self.re_bank} only means anything under "
                f"re_strip='bank' (got {self.re_strip!r}) — a sweep count "
                f"that no mode reads is a setting that does nothing")
        if not (self.alpha_max_deg > 0.0):
            raise ValueError(
                f"alpha_max_deg must be > 0, got {self.alpha_max_deg}")
        self.twist_order = int(self.twist_order)
        self.chord_order = int(self.chord_order)
        alphas = (WING_ALPHAS if self.alphas is None
                  else np.asarray(self.alphas, dtype=float))
        # the wing OWNS the operating point in this mode
        self.section = replace(self.airfoil, re=self.wing.re_mac,
                               cl_design=self.wing.cl_design, alphas=alphas)
        self._twist_bounds = twist_bounds(self.twist_order, self.twist_max_deg)
        # chord_bounds validates chord_order / chord_max_frac and returns an
        # empty (0, 2) box at order 0, so the legacy vstack is unchanged
        self._chord_bounds = chord_bounds(self.chord_order,
                                          self.chord_max_frac)
        self._bounds = np.vstack([self.section.bounds, self._twist_bounds,
                                  self._chord_bounds])
        self._x0 = np.concatenate(
            [self.section.w0, np.zeros(self.twist_order, dtype=float),
             np.zeros(self.chord_order, dtype=float)])

    @property
    def x0(self) -> np.ndarray:
        """Baseline design: the anchor section with ZERO twist."""
        return self._x0.copy()

    @property
    def w0(self) -> np.ndarray:
        """Alias of :attr:`x0` (the AirfoilProblem naming)."""
        return self._x0.copy()

    @property
    def bounds(self) -> np.ndarray:
        return self._bounds.copy()

    @property
    def dim(self) -> int:
        return int(self.section.dim + self.twist_order + self.chord_order)

    @property
    def n_constraints(self) -> int:
        return 4 + (1 if self.chord_order else 0)

    @property
    def constraint_labels(self) -> tuple:
        """Display names for g0..g3 (+g4 with a chord law). api.design_report
        prefers these over the static ProblemSpec labels, which describe the
        2-constraint section."""
        if self.chord_order:
            return CONSTRAINT_LABELS + (CHORD_CONSTRAINT_LABEL,)
        return CONSTRAINT_LABELS

    @property
    def param_labels(self) -> tuple:
        n = self.section.n_cst
        return tuple(
            [f"w_upper_{i}" for i in range(n)]
            + [f"w_lower_{i}" for i in range(n)]
            + [f"twist_{j}_deg" for j in range(1, self.twist_order + 1)]
            + [f"chord_{j}" for j in range(1, self.chord_order + 1)])


CONSTRAINT_LABELS = ("t/c margin", "|Cm| margin",
                     "twist-envelope margin", "max-incidence margin")
CHORD_CONSTRAINT_LABEL = "chord-deviation margin"


#: the three spanwise-Reynolds treatments (SectionWingProblem.re_strip)
RE_STRIP_MODES: tuple[str, ...] = ("mac", "power", "bank")

#: Drela's strip-Reynolds drag exponents, and the Reynolds number OUR code
#: switches between them at: laminar skin friction goes as Re^-1/2 and
#: turbulent as Re^-1/5, and a blade-element code applies whichever the local
#: strip is in. The switch is a REGIME boundary, not a fitted parameter, which
#: is why it is two constants and not a regression: the local exponent
#: measured on our own bank drifts from about -0.41 to -0.585 across
#: 1e5 -> 3e6, so one fitted exponent is biased at both ends.
#:
#: PROVENANCE, corrected in session 43 (report SS18.4). The two EXPONENTS are
#: Drela's and are shipped by QPROP and XROTOR. **The 2e5 THRESHOLD between
#: them is ours, not theirs** -- an earlier version of this comment and of the
#: report called it "the split QPROP/XROTOR ship" and that is wrong against
#: the verbatim primary quotes in LITERATURE_REVIEW_S41.md SSA.2.2: XROTOR
#: puts its -0.1..-0.2 TURBULENT band at Re > 2e6 and gives -0.5..-1.5 across
#: Re 2e5..8e5 (the band this wing's tip actually occupies), and QPROP states
#: no Reynolds threshold at all. Our threshold is justified by MEASUREMENT
#: instead: the local two-point exponents on the E387 LTPT drag polar are
#: -0.585 across 1e5->2e5 and -0.41 across 3e5->4.6e5, so 2e5 is where the
#: measured exponent crosses the midpoint of the two flat-plate anchors.
RE_EXP_LAMINAR, RE_EXP_TURBULENT = -0.5, -0.2
RE_EXP_SWITCH = 2.0e5


def _cd_power_law(cd_mac, re_y, re_mac: float):
    """Drela's ``cd * (Re_y/Re_mac)**n`` strip correction. No XFOIL at all.

    The cheap half of the spanwise-Reynolds fix, and the one the literature
    recommends where a polar bank is not affordable (QPROP and XROTOR ship
    the two exponents; the 2e5 switch between them is ours — see
    ``RE_EXP_SWITCH``). It corrects DRAG only — ``cl_max`` moves with Reynolds
    number too and this does nothing about it, so the stall gate stays
    optimistic at the tip.

    It is NOT a substitute for the bank, and that is measured rather than
    assumed (report SS18.4): scored against the dense bank on the same strips
    the free correction recovers 40-313 % of the integrated error depending on
    the row, i.e. it overshoots on some and undershoots on others. The
    "removes essentially all of the integrated error at zero cost" line is
    this project's OWN session-41 review speaking about its own 2-D numbers,
    not a published claim, and this section's own measurement refutes it.

    **The exponent is INTEGRATED, not sampled** (fixed session 42, item 1).
    The model above declares a LOCAL exponent, ``d ln cd / d ln Re = n(Re)``,
    which is exactly why it is two constants and not a regression. Carrying a
    strip from ``Re_mac`` to ``Re_y`` therefore integrates that local exponent
    along the path, and when the path crosses ``RE_EXP_SWITCH`` the integral
    is piecewise:

        cd(Re_y) / cd(Re_ref)
            = (Re_sw/Re_ref)**n_turb * (Re_y/Re_sw)**n_lam     (crossing down)

    The original form picked ``n`` from the STRIP's regime and applied it
    over the whole span, which is the same thing only when both ends sit in
    one regime. When they straddle the switch it is discontinuous: at
    ``Re_ref = 1.13e6`` a strip at Re 199 999 was multiplied by 2.3770 and one
    at Re 200 001 by 1.4139 — a **1.68x jump for a 1e-5 relative change in a
    strip's Reynolds number**, and a step of 393x the local smooth gradient in
    a BO objective as a chord-law coefficient walks one station across the
    boundary.

    Nothing published on the default wing moves: at taper 0.6 the strips run
    Re 7.96e5..1.33e6 and none of them crosses 2e5, which is precisely why the
    defect survived — no test in the suite put a strip on the far side of the
    switch. It bites at model / UAV scale, which is the Reynolds range this
    correction exists for.
    """
    re_y = np.asarray(re_y, dtype=float)
    ref = float(re_mac)
    # log-space path integral of the piecewise-constant local exponent, which
    # is continuous by construction and reduces to the single-regime form
    # whenever the path does not cross.
    lo = np.minimum(re_y, ref)
    hi = np.maximum(re_y, ref)
    sw_ = np.clip(RE_EXP_SWITCH, lo, hi)            # the crossing, where it is
    ln = (RE_EXP_LAMINAR * (np.log(sw_) - np.log(lo))
          + RE_EXP_TURBULENT * (np.log(hi) - np.log(sw_)))
    # the integral above runs low -> high; flip it where the strip is FASTER
    # than the reference, so the correction is antisymmetric as a ratio law
    # must be (going out and back returns the original drag)
    ln = np.where(re_y < ref, -ln, ln)
    return np.asarray(cd_mac, dtype=float) * np.exp(ln)


def _cd_per_strip(coords, sec, n_bank: int, wing, c, cl_y, clip, cd_single):
    """Strip profile drag read at each strip's OWN Reynolds number.

    ``n_bank`` XFOIL sweeps of the SAME section, log-spaced over
    [Re(tip), Re(root)], and cd interpolated between them linearly in log(Re)
    — both flat-plate skin-friction laws are power laws in Re, so cd is close
    to straight against log(Re) and strongly curved against Re.

    Returns ``(cd_y, info)``. ``info`` records what it cost and how well the
    bank resolved the span, because both are part of the answer: the bank is
    ``n_bank`` XFOIL sweeps per evaluation instead of one, and its own
    interpolation error was measured at <= 0.32 counts above Re 3e5 and 1.20
    counts at Re 1.5e5 (results/spanwise_re_probe.json). Below the point where
    that error is the same size as the effect, the bank is buying noise.
    """
    from .xfoil_run import run_xfoil_polar

    lo, hi = clip
    c = np.asarray(c, dtype=float)
    re_y = wing.rho * wing.v_ms * c / wing.mu
    re_lo, re_hi = float(re_y.min()), float(re_y.max())
    if not (re_hi > re_lo * (1.0 + 1e-12)):
        # a rectangular wing: every strip is at the MAC Reynolds number, so
        # the bank has nothing to resolve and the legacy lookup IS the answer.
        # Returning it verbatim (rather than a one-member bank that would
        # re-run XFOIL at the same point) keeps that identity exact.
        return np.asarray(cd_single, dtype=float), {
            "n_polars": 0, "re_lo": re_lo, "re_hi": re_hi, "re_ratio": 1.0,
            "bank_re": [], "note": "uniform chord: the single-Re lookup is exact"}
    bank_re = np.geomspace(re_lo, re_hi, int(n_bank))
    branches = []
    for re_i in bank_re:
        pol_i = run_xfoil_polar(coords, float(re_i), sec.mach, sec.alphas,
                                timeout_s=sec.timeout_s,
                                cache_dir=sec.cache_dir, n_panel=sec.n_panel)
        if pol_i.n_converged < 3:
            raise _BankMiss(f"section unconvergeable at Re {re_i:.3g}")
        sl_i = _longest_increasing_run(pol_i.cl)
        cl_i, cd_i = pol_i.cl[sl_i], pol_i.cd[sl_i]
        if cl_i.size < 2:
            raise _BankMiss(f"no monotone branch at Re {re_i:.3g}")
        branches.append((cl_i, cd_i))

    logs = np.log(bank_re)
    cd_y = np.empty_like(re_y)
    for k, (re_k, clk) in enumerate(zip(re_y, np.asarray(cl_y, dtype=float))):
        j = min(max(int(np.searchsorted(bank_re, re_k)), 1), bank_re.size - 1)
        c0_cl, c0_cd = branches[j - 1]
        c1_cl, c1_cd = branches[j]
        v0 = float(np.interp(np.clip(clk, c0_cl[0], c0_cl[-1]), c0_cl, c0_cd))
        v1 = float(np.interp(np.clip(clk, c1_cl[0], c1_cl[-1]), c1_cl, c1_cd))
        t = (np.log(re_k) - logs[j - 1]) / (logs[j] - logs[j - 1])
        # LOG-log in drag: cd is a power law in Re either side of transition,
        # so log cd is near-straight against log Re while cd itself is convex
        # and a linear interpolant sits above the truth between members. Worth
        # a measured factor of two here (polar.polar_at_re's docstring).
        cd_y[k] = (np.exp((1.0 - t) * np.log(v0) + t * np.log(v1))
                   if v0 > 0.0 and v1 > 0.0 else (1.0 - t) * v0 + t * v1)
    return cd_y, {"n_polars": int(n_bank),
                  "re_lo": re_lo, "re_hi": re_hi,
                  "re_ratio": re_hi / re_lo,
                  "bank_re": [float(r) for r in bank_re]}


class _BankMiss(RuntimeError):
    """A bank member XFOIL could not fly — falls back to the single-Re path."""


def evaluate_section_wing(x: np.ndarray,
                          prob: SectionWingProblem | None = None) -> dict:
    """Full section+twist evaluation with breakdown; never raises in contract.

    ``feasible`` refers to the SOLVER (penalty contract); the four design
    constraints are reported separately as the signed margin array ``g`` —
    constrained BO needs true objective values AT infeasible designs
    (hydrofoil.py convention). The LLT solution is returned under ``llt`` so
    ``api.design_report`` picks up the spanwise arrays with no extra work.
    """
    from .objective import _fail
    from .xfoil_run import run_xfoil_polar

    prob = prob or SectionWingProblem()
    sec, wing = prob.section, prob.wing
    x = np.asarray(x, dtype=float)
    bnds = prob.bounds
    if x.shape != (bnds.shape[0],):
        return _fail(f"design vector shape {x.shape} != ({bnds.shape[0]},)")
    if np.any(x < bnds[:, 0] - 1e-12) or np.any(x > bnds[:, 1] + 1e-12):
        return _fail("bounds violation")

    n = sec.n_cst
    w_u, w_l = x[:n], x[n:2 * n]
    k = int(prob.twist_order)
    tw = x[2 * n:2 * n + k]
    ch = x[2 * n + k:]

    # --- section geometry (identical prechecks to evaluate_airfoil) ------
    psi = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, 401)))[1:-1]
    gap = _surface_y(psi, w_u, +0.5 * sec.dz_te) - _surface_y(
        psi, w_l, -0.5 * sec.dz_te)
    if not np.all(gap > 0.0):
        return _fail("self-intersecting / non-positive-thickness geometry")

    tc = float(cst_thickness(w_u, w_l, dz_te=sec.dz_te))
    coords = cst_coords(w_u, w_l, dz_te=sec.dz_te)
    pol = run_xfoil_polar(
        coords, sec.re, sec.mach, sec.alphas,
        timeout_s=sec.timeout_s, cache_dir=sec.cache_dir,
        n_panel=sec.n_panel,
    )   # XfoilError (infrastructure fault) intentionally propagates

    if pol.n_converged < 3:
        return _fail("unconvergeable section (< 3 converged XFOIL points) "
                     "= failed design", n_converged=pol.n_converged)

    sl = _longest_increasing_run(pol.cl)
    cl_b, cd_b = pol.cl[sl], pol.cd[sl]
    cm_b, al_b = pol.cm[sl], pol.alpha_deg[sl]
    cl_des = float(sec.cl_design)
    if cl_b.size < 2 or cl_b[-1] < cl_des or cl_b[0] > cl_des:
        return _fail(
            "CL_design not bracketed by the pre-stall monotone branch "
            "(section cannot reach the derived design lift) = failed design",
            n_converged=pol.n_converged, cl_max=float(pol.cl.max()),
            cl_design=cl_des)

    cm_at = float(np.interp(cl_des, cl_b, cm_b))
    cd_at = float(np.interp(cl_des, cl_b, cd_b))

    # --- linearise the real polar for the lifting line ------------------
    try:
        a_slope, alpha_L0 = section_line_fit(cl_b, al_b, cl_des)
    except ValueError as exc:
        return _fail(f"section linearisation failed: {exc}",
                     n_converged=pol.n_converged)

    # --- trim the wing to the DERIVED CL --------------------------------
    try:
        y, c, eta = wing.stations(ch)
    except ValueError as exc:
        return _fail(f"chord law is not a planform: {exc}")
    # the straight-taper baseline the deviation margin is measured against.
    # Without a chord law the flown chord IS the baseline, so the legacy path
    # does not pay for a second planform evaluation.
    c_trap = wing.stations()[1] if ch.size else c
    theta_deg = twist_deg(eta, tw)
    try:
        alpha_root, llt = solve_llt_trim(
            wing.b, c, np.deg2rad(theta_deg), cl_des, a=a_slope,
            alpha_L0=alpha_L0,
            alpha_bracket=(np.deg2rad(TRIM_BRACKET_DEG[0]),
                           np.deg2rad(TRIM_BRACKET_DEG[1])))
    except (ValueError, RuntimeError) as exc:
        return _fail(
            f"wing cannot be trimmed to CL_design = {cl_des:.3f} within "
            f"{TRIM_BRACKET_DEG} deg root incidence ({exc})",
            cl_design=cl_des)

    # --- strip profile drag from the SAME polar -------------------------
    cl_y = np.asarray(llt.Cl_y, dtype=float)
    lo, hi = float(cl_b[0]), float(cl_b[-1])
    n_above = int(np.sum(cl_y > hi))
    if n_above:
        return _fail(
            f"local stall: {n_above} station(s) above the section pre-stall "
            f"branch (cl_max ~ {hi:.3f}) — a linear lifting line is invalid "
            "there = failed design",
            cl_design=cl_des, cl_y_max=float(cl_y.max()))
    n_clamped_low = int(np.sum(cl_y < lo))
    cd_y = np.interp(np.clip(cl_y, lo, hi), cl_b, cd_b)
    re_bank_info: dict | None = None
    if prob.re_strip == "power":
        # zero extra XFOIL: the MAC polar's cd, scaled to each strip's own
        # Reynolds number by the flat-plate exponent of the regime it is in
        re_y = wing.rho * wing.v_ms * c / wing.mu
        cd_y = _cd_power_law(cd_y, re_y, wing.re_mac)
        re_bank_info = {"mode": "power", "n_polars": 0,
                        "re_lo": float(re_y.min()), "re_hi": float(re_y.max()),
                        "re_ratio": float(re_y.max() / re_y.min()),
                        "exponents": [RE_EXP_LAMINAR, RE_EXP_TURBULENT],
                        "switch_re": RE_EXP_SWITCH}
    elif prob.re_strip == "bank":
        # Every strip is looked up at ITS OWN Reynolds number instead of the
        # wing's MAC value. Measured cost of not doing this
        # (scripts/spanwise_re_probe.py, results/spanwise_re_probe.json):
        # +0.13 ct of CDp on the published taper-0.6 default, +2.08 ct
        # (+3.9 % of CDp, -1.9 % of L/D) at taper 0.2, where the root/tip
        # Reynolds ratio reaches 5. A rectangular wing returns EXACTLY the
        # single-Re answer, which is the control.
        #
        # Only cd is taken per strip. The lift slope and zero-lift angle stay
        # the MAC polar's, deliberately: the linear fit to a low-Re polar is
        # not trustworthy (the same fit gives NACA 2412 a_lin 6.33/rad at
        # Re 1e6 and 8.66/rad at 1e5 — above 2*pi, a separation-bubble
        # artefact of fitting a curved lift line, not physics), and feeding
        # per-strip slopes would move the circulation on the strength of it.
        try:
            cd_y, re_bank_info = _cd_per_strip(
                coords, sec, prob.re_bank, wing, c, cl_y, (lo, hi), cd_y)
        except _BankMiss as exc:
            # a bank member XFOIL could not fly is NOT a failed design — the
            # design flew fine at the MAC point. Fall back to the single-Re
            # lookup and SAY SO in the breakdown, rather than refusing a
            # candidate for a modelling refinement it did not ask for.
            re_bank_info = {"n_polars": 0, "fell_back": str(exc)}

    S_llt = float(np.trapezoid(c, y))
    CDp = float(np.trapezoid(cd_y * c, y) / S_llt)
    CDi = float(llt.CDi)
    CD = CDp + CDi
    if not (CD > 0.0) or not np.isfinite(CD):
        return _fail("non-positive / non-finite wing drag")
    CL = float(llt.CL)
    LD = CL / CD
    if not np.isfinite(LD):
        return _fail("non-finite objective")

    # --- constraints ----------------------------------------------------
    alpha_root_deg = float(np.rad2deg(alpha_root))
    alpha_geo_deg = alpha_root_deg + theta_deg
    alpha_geo_max = float(np.max(np.abs(alpha_geo_deg)))
    env = twist_envelope_deg(tw)
    g_list = [
        norm_margin(tc - sec.tc_min, sec.tc_min),
        norm_margin(sec.cm_max - abs(cm_at), sec.cm_max),
        norm_margin(prob.twist_max_deg - env, prob.twist_max_deg),
        norm_margin(prob.alpha_max_deg - alpha_geo_max, prob.alpha_max_deg),
    ]
    c_dev = chord_dev_envelope(c, c_trap)
    if prob.chord_order:
        g_list.append(
            norm_margin(prob.chord_max_frac - c_dev, prob.chord_max_frac))
    g = np.array(g_list)

    # the polar was flown at the BASELINE mac's Reynolds number (it is fixed
    # when the problem is built); report what the flown planform's own mac
    # would have implied so a reshaped chord cannot quietly drift off it
    mac_true = mac_of(c, y)
    re_true = float(wing.rho * wing.v_ms * mac_true / wing.mu)
    # root and tip chord of the FLOWN law, evaluated at eta = 0 and 1 rather
    # than read off the nearest station: the cosine grid has no node at either
    # end, so a station reading would understate the true taper.
    if ch.size:
        mult = chord_multiplier(eta, ch)
        scale = float(np.trapezoid(c_trap, y) / np.trapezoid(c_trap * mult, y))
    else:
        scale = 1.0
    c_root_flown = scale * wing.c_root
    c_tip_flown = (scale * wing.c_root * wing.taper
                   * float(chord_multiplier(np.array([1.0]), ch)[0]))

    return {
        "feasible": True, "reason": "", "score": LD, "f": LD, "LoD": LD,
        "LD": LD, "CL": CL, "CD": CD, "CDi": CDi, "CDp": CDp,
        "e": float(llt.e), "AR": float(llt.AR), "S_llt": S_llt,
        "S_ref": float(wing.S), "b": float(wing.b), "mac": float(wing.mac),
        "mac_true": mac_true, "re_true": re_true,
        "cl_design": cl_des, "re": float(sec.re),
        "cd": cd_at, "cm": cm_at, "tc": tc,
        "a_per_rad": a_slope, "alpha_L0_deg": float(np.rad2deg(alpha_L0)),
        "alpha_root_deg": alpha_root_deg,
        "alpha_geo_max_deg": alpha_geo_max,
        "twist_env_deg": env,
        "twist_coeffs_deg": [float(v) for v in tw],
        "twist_tip_deg": float(twist_deg(np.array([1.0]), tw)[0]),
        "chord_coeffs": [float(v) for v in ch],
        "chord_dev": c_dev,
        "chord_root_m": float(c_root_flown),
        "chord_tip_m": float(c_tip_flown),
        "taper_flown": float(c_tip_flown / c_root_flown),
        # the FLOWN chord is already exported as the LLT result's own c array
        # (api.design_report picks it up as geometry["chord"]); only the
        # baseline it departed from is new information
        "chord_trap_m": [float(v) for v in c_trap],
        "cl_y_min": float(cl_y.min()), "cl_y_max": float(cl_y.max()),
        "n_clamped_low": n_clamped_low,
        # None on the legacy single-Re path, so a stored breakdown says which
        # of the two drag models produced its CDp rather than leaving it to
        # be inferred from the problem that is no longer beside it
        "re_bank": re_bank_info,
        "n_converged": pol.n_converged, "n_branch": int(cl_b.size),
        "from_cache": pol.from_cache,
        "g": g, "g_tc": float(g[0]), "g_cm": float(g[1]),
        "g_twist": float(g[2]), "g_alpha": float(g[3]),
        "g_chord": float(g[4]) if prob.chord_order else None,
        "llt": llt,                      # picked up by api.design_report
        "wing": wing,                    # planform scalars for the same
    }


def fg_section_wing(x: np.ndarray, prob: SectionWingProblem | None = None
                    ) -> tuple[float, np.ndarray]:
    """Constrained-harness callable: (f, [g_tc, g_cm, g_twist, g_alpha(, g_chord)]).

    Solvable-but-infeasible designs return their TRUE L/D with signed
    margins; solver/geometry failures return exactly
    ``(PENALTY, [G_FAIL] * n_constraints)`` — five entries once a chord law
    is in play, four without one.
    """
    prob = prob or SectionWingProblem()
    out = evaluate_section_wing(x, prob)
    if not out["feasible"]:
        return PENALTY, np.full(prob.n_constraints, G_FAIL, dtype=float)
    return float(out["f"]), np.asarray(out["g"], dtype=float)
