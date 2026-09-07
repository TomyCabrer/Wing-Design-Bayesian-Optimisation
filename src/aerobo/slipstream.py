"""Propeller-slipstream / wing interaction: finite-jet-height lift ratio +
non-uniform velocity-triangle profiles across the prop-disc footprint.

numpy port of the GDP MATLAB prop-slipstream skin — ported VERBATIM, the way
llt.py was ported from solve_llt.m:

    ~/Desktop/GDP/tandem_wing/core/jet_lift_ratio.m       -> jet_lift_ratio
    ~/Desktop/GDP/tandem_wing/core/slipstream_profiles.m  -> slipstream_profiles
    ~/Desktop/GDP/tandem_wing/scripts/test_jet_lift_ratio.m -> cross-check tests

Two physically distinct effects, both bundled here because they share the same
prop-disc geometry:

1. FINITE-SLIPSTREAM-HEIGHT LIFT RATIO  ``jet_lift_ratio`` (K_l).
   A 2-D wing section immersed on the centreline of a propeller slipstream of
   FINITE height h (surrounded by free stream) does NOT see the naive
   infinite-jet square-law augmentation K_l = (V_j/V_inf)^2 = mu^2 that
   low-order LL/VLM models apply. Nederlof et al. (AIAA J. 63(6), 2025,
   "Fast Numerical Modeling of Propeller-Wing Aerodynamic Interactions",
   Eqs. 19-26) show that square law overpredicts the integral lift
   augmentation by ~8% and up to ~25-66% locally. The finite-height model is
   a 2-D image-vortex construction (Ting & Liu 1969; Prabhu 1984;
   Rethorst 1958): the bound vortex Gamma at x=0 with thin-aerofoil control
   point at x=c/2 is reflected by the two slip lines at +/- h/2 with strength
   factor eps = (mu^2-1)/(mu^2+1), giving image pairs at +/- n h of strength
   eps^n Gamma. Their net downwash at the control point yields the closed form

       K_l = mu^2 / ( 1 + (1/2) SUM_{n=1..N} eps^n / (1/4 + n^2 (h/c)^2) ).

   Exact analytic limits (each a sign/limit sanity check, tested):
       h/c -> inf :  SUM -> 0            => K_l -> mu^2  (infinite-jet law)
       h/c -> 0   :  (1/2)SUM -> mu^2-1  => K_l -> 1     (no augmentation)
       mu  == 1   :  eps = 0             => K_l = 1      (outside the jet)
   K_l multiplies the 2-D lift slope in the strip solver (GDP lift_drag_strip.m).

2. NON-UNIFORM AXIAL + SWIRL PROFILES  ``slipstream_profiles``.
   Replaces the top-hat slipstream (uniform velocity + uniform swirl in a wake
   band) with the physically-representative RADIAL profile of Nederlof et al.
   (Figs. 3 & 10) + Patterson (2016) velocity-triangle: the axial velocity
   peaks in the core and tapers to zero at the shear layer, and the swirl is
   ANTISYMMETRIC about the prop axis. Both shapes are INTEGRAL-PRESERVING
   redistributions of the already-validated scalar means — the full-tube
   average of each shape is normalised to 1, so the band-mean axial velocity
   equals the momentum-theory value w and the band-mean |swirl| equals the
   Bell reference angle. This is a radial redistribution, NOT a new magnitude.

KNOWN FINDING (GDP, encoded in tests) -- in CRUISE this skin is DORMANT and
that small effect is CORRECT behaviour, not a bug:
  * cruise proprotors are lightly loaded => slipstream mu ~ 1.002, so the
    dynamic-pressure augmentation q_ratio = mu^2 ~ 1.004 (< 0.5%) regardless
    of the model;
  * the proprotor is huge vs the local chord (GDP: D_p = 6.19 m, contracted
    h = 5.67 m, tip chord 0.60 m => h/c ~ 9.5), so K_l ~ mu^2 to < 0.1% --
    the finite-height knockdown all but vanishes and the infinite-jet square
    law was already accurate here.
  The slipstream only matters off-design (high thrust / low speed) and for the
  swirl-driven roll asymmetry of co-rotating props; the tests below assert both
  the exact OFF limits and the < 0.5% cruise dormancy.

Sign conventions (DERIVED, not assumed)
---------------------------------------
* eps sign: eps = (mu^2-1)/(mu^2+1) is >0 for an accelerated slipstream
  (mu>1) and =0 at mu=1 (so K_l=1 exactly outside the jet, no rounding
  residue -- multiplication by an exact 0). eps<0 (decelerated, mu<1) is also
  admissible and the series still converges (|eps|<1 for all mu>0).
* swirl sign: the swirl SPEED magnitude is V tan|dalpha_ref|; its up/down
  DIRECTION is set purely by the rotation sign ``spin`` and the inboard/
  outboard geometry sign(y - y_p), NOT by the sign of dalpha_ref. spin = -1
  (CW) / +1 (CCW) viewed FROM BEHIND looking forward; +ve v_swirl = upwash =
  raises the local AoA. A station at y_p+d and its mirror at y_p-d therefore
  carry equal-and-opposite swirl => the swirl-induced angle is antisymmetric
  about each prop axis and integrates to zero over a symmetric station pair.

PHASE INVARIANT: a SlipstreamSpec with mu_inf = 1 (no thrust) and
dalpha_ref_deg = 0 returns q_ratio identically 1.0 and dalpha identically 0.0
(bit-for-bit == float equality) -- the OFF default reproduces the no-slipstream
result exactly. This is the regression-gated default the objective must keep.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

DEFAULT_N_IMG = 40         # image pairs summed in K_l (jet_lift_ratio.m default)
DEFAULT_CORE_FRAC = 0.6    # axial flat-core fraction (slipstream_profiles.m default)


# --------------------------------------------------------------------------- K_l


def jet_lift_ratio(
    mu,
    h_over_c,
    n_img: int = DEFAULT_N_IMG,
    kernel: str = "image",
):
    """Finite-slipstream-height sectional lift-augmentation ratio K_l.

    Verbatim port of jet_lift_ratio.m. Returns the ratio
    ``K_l = l_section_in_jet / l_section_freestream`` for a 2-D section on the
    centreline of a propeller slipstream ("jet") of finite height h surrounded
    by free stream, via the image-vortex closed form (module docstring, model 1):

        eps = (mu^2 - 1) / (mu^2 + 1)                         (paper Eqs. 19,26)
        S   = (1/2) SUM_{n=1..n_img} eps^n / (1/4 + n^2 (h/c)^2)
        K_l = mu^2 / (1 + S)

    Parameters
    ----------
    mu : velocity ratio V_j/V_inf per section (>= 1 in an accelerated
        slipstream; mu < 1 decelerated is also valid). Scalar or array.
    h_over_c : slipstream height / local chord, h = 2 R_W (jet height = wake
        diameter). Scalar or array, broadcastable against ``mu``.
    n_img : number of image pairs summed (default 40, the .m default).
    kernel : ``"image"`` (default) finite-height model, or ``"infinite"`` which
        returns exactly mu^2 -- the infinite-jet square law, the OFF / A-B
        legacy branch.

    Returns
    -------
    K_l : lift-augmentation ratio, broadcast of ``mu`` and ``h_over_c``. A
        Python float for scalar inputs (MATLAB ergonomics), else an ndarray.

    Exact limits (tested): h/c->inf => mu^2 ; h/c->0 => 1 ; mu==1 => 1.
    """
    mu = np.asarray(mu, dtype=float)
    h_over_c = np.asarray(h_over_c, dtype=float)
    mu2 = mu**2

    shape = np.broadcast_shapes(mu2.shape, h_over_c.shape)

    if kernel.lower() == "infinite":
        # infinite-jet square law (legacy behaviour): K_l = mu^2, broadcast so a
        # vector A/B test returns one K_l per station.
        out = np.broadcast_to(mu2, shape).astype(float)
        return out.item() if out.ndim == 0 else out
    if kernel.lower() != "image":
        raise ValueError(f"kernel must be 'image' or 'infinite' (got {kernel!r})")

    eps_r = (mu2 - 1.0) / (mu2 + 1.0)          # image-strength factor; 0 at mu==1
    hoc2 = h_over_c**2

    # Image-pair series S = (1/2) sum_{n=1}^{N} eps^n / (1/4 + n^2 (h/c)^2),
    # accumulated exactly as the .m does (en carries eps^n across iterations).
    eps_b = np.broadcast_to(eps_r, shape)
    hoc2_b = np.broadcast_to(hoc2, shape)
    S = np.zeros(shape, dtype=float)
    en = np.ones(shape, dtype=float)           # eps^0 -> eps^n each step
    for n in range(1, n_img + 1):
        en = en * eps_b                        # eps^n
        S = S + en / (0.25 + (n**2) * hoc2_b)
    S = 0.5 * S

    # Outside the slipstream mu==1 -> eps=0 -> S=0 -> K_l = mu^2/1 = 1 exactly
    # (no guard needed: multiplication/division involving the exact 0 leaves no
    # rounding residue).
    K_l = np.broadcast_to(mu2, shape) / (1.0 + S)
    K_l = np.asarray(K_l, dtype=float)
    return K_l.item() if K_l.ndim == 0 else K_l


# ------------------------------------------------------------- radial shapes


def axial_shape(s, core_frac: float = DEFAULT_CORE_FRAC):
    """Raised-cosine axial shape f_ax(s), s = |rho| in [0, 1] (verbatim .m).

        f_ax(s) = 1                                       s <= core_frac
                  cos( (pi/2)(s-core_frac)/(1-core_frac) )^2   core_frac < s < 1
                  0                                       s >= 1

    Flat core out to ``core_frac`` of the wake radius, smoothly tapering to
    zero at the shear-layer edge s = 1.
    """
    s = np.asarray(s, dtype=float)
    denom = max(1.0 - core_frac, float(np.finfo(float).eps))
    core = (s <= core_frac).astype(float)
    edge_mask = (s > core_frac) & (s < 1.0)
    edge = np.where(
        edge_mask,
        np.cos(0.5 * np.pi * (s - core_frac) / denom) ** 2,
        0.0,
    )
    return core + edge


def swirl_shape(s):
    """Swirl radial shape f_sw(s) = sin(pi s), s = |rho| in [0, 1] (verbatim .m).

    Peaks at mid-radius (s = 1/2) and vanishes at hub (s=0) and edge (s=1);
    combined with the antisymmetric geometry sign it makes the swirl a pure
    up/down couple across the disc.
    """
    return np.sin(np.pi * np.asarray(s, dtype=float))


# ------------------------------------------------------------- profile struct


@dataclass
class SlipstreamProfile:
    """Per-station slipstream field (output of ``slipstream_profiles``).

    Arrays sized like the input stations y (mirrors the .m ``prof`` struct,
    plus the derived swirl angle ``dalpha``):

        w_axial   axial induced velocity profile [m/s]
        v_swirl   swirl vertical velocity profile [m/s] (+ve = upwash)
        mu        local AXIAL velocity ratio (V + w_axial)/V (>=1 in band, 1 out)
        dalpha    swirl-induced local AoA increment [rad] (velocity triangle)
        in_wake   True where the station is immersed in >= 1 prop wake
        frac_wake fraction of stations immersed
    """

    y: np.ndarray
    w_axial: np.ndarray
    v_swirl: np.ndarray
    mu: np.ndarray
    dalpha: np.ndarray
    in_wake: np.ndarray
    frac_wake: float


def slipstream_profiles(
    y,
    y_p_list,
    R_W,
    w: float,
    dalpha_ref: float,
    spin,
    V: float,
    core_frac: float = DEFAULT_CORE_FRAC,
    swirl_fn: Callable | None = None,
    axial_fn: Callable | None = None,
) -> SlipstreamProfile:
    """Non-uniform per-strip axial + swirl profiles (verbatim port of the .m).

    Parameters
    ----------
    y : spanwise stations [m] (vector, e.g. the LLT cosine stations).
    y_p_list : prop spanwise centres [m] (vector; one per prop, symmetric pairs).
    R_W : contracted wake radius [m]; scalar (all props) or per-prop vector.
    w : momentum-theory band-MEAN axial induced velocity at the wing [m/s].
    dalpha_ref : reference Bell swirl angle [deg]; the band-mean |swirl| MAGNITUDE
        (Bell V/STOL Tilt-Rotor Study, wing-pylon wake formula). Its sign does
        NOT set the swirl direction (see below).
    spin : prop rotation sign(s) +/-1: scalar (all props) or vector per prop.
        -1 = CW, +1 = CCW viewed from behind; the .m maps a scalar via
        sign(spin) + (spin==0), so spin==0 becomes +1 (CCW).
    V : freestream speed [m/s].
    core_frac, swirl_fn, axial_fn : optional shape overrides (defaults = the .m
        raised-cosine axial + sin(pi s) swirl).

    Returns
    -------
    SlipstreamProfile.

    Swirl-direction derivation (the sign trap): the swirl SPEED is
    ``vsw_mag = V tan|dalpha_ref|``; the up/down sense is
    ``spin[k] * sign(y - y_p)`` -- rotation sign times inboard/outboard
    geometry, NOT the sign of dalpha_ref. +ve v_swirl = upwash. Hence a station
    and its mirror across a prop axis carry opposite swirl (antisymmetric).

    The angle ``dalpha`` is the swirl term of the unified velocity triangle
    used by the GDP strip solver, alpha_e = atan2(V sin i + v_swirl,
    V cos i + w_axial): evaluated at i = 0 the swirl ADDS
    ``dalpha = atan2(v_swirl, V + w_axial)`` to the local geometric AoA
    (small-angle -> v_swirl/(V+w_axial)). Using the local axial speed V+w_axial
    (not V) keeps the pair-antisymmetry EXACT (odd numerator, common even
    denominator).
    """
    y = np.atleast_1d(np.asarray(y, dtype=float)).ravel()
    y_p_list = np.atleast_1d(np.asarray(y_p_list, dtype=float)).ravel()
    n_prop = y_p_list.size

    ax_fn = axial_fn if axial_fn is not None else (lambda s: axial_shape(s, core_frac))
    sw_fn = swirl_fn if swirl_fn is not None else swirl_shape

    # spin: scalar -> sign(spin) + (spin==0) replicated; vector -> sign per prop
    spin_arr = np.asarray(spin, dtype=float)
    if spin_arr.ndim == 0:
        s0 = float(spin_arr)
        spin_arr = np.full(n_prop, np.sign(s0) + (1.0 if s0 == 0.0 else 0.0))
    else:
        spin_arr = np.sign(spin_arr.ravel().astype(float))

    # R_W: scalar -> replicated; vector -> per prop
    R_W_arr = np.asarray(R_W, dtype=float)
    if R_W_arr.ndim == 0:
        R_W_arr = np.full(n_prop, float(R_W_arr))
    else:
        R_W_arr = R_W_arr.ravel()

    w_axial = np.zeros_like(y)
    v_swirl = np.zeros_like(y)
    in_wake = np.zeros(y.shape, dtype=bool)

    # Swirl SPEED magnitude (band-mean); direction set by spin*geometry below.
    vsw_mag = V * np.tan(np.deg2rad(abs(float(dalpha_ref))))

    # Full-tube (clip-independent) normalisers: integrate each radial shape over
    # the COMPLETE wake radius s in [0, 1] ONCE, so the profile magnitude is the
    # fixed momentum value regardless of how much of the tube lands on the wing
    # (a clipped wake must NOT renormalise to a lower peak).
    s_full = np.linspace(0.0, 1.0, 201)
    Za = float(np.trapezoid(ax_fn(s_full), s_full))
    if Za <= 0.0:
        Za = 1.0
    Zs = float(np.trapezoid(sw_fn(s_full), s_full))
    if Zs <= 0.0:
        Zs = 1.0

    for k in range(n_prop):
        yp = y_p_list[k]
        Rk = R_W_arr[k]
        if Rk <= 0.0:
            continue
        rho = (y - yp) / Rk                     # signed radial coordinate
        band = np.abs(rho) <= 1.0
        if not np.any(band):
            continue
        s = np.abs(rho[band])

        # axial: max-rule where bands overlap (cannot superpose the q rise)
        fa = ax_fn(s)
        w_k = w * fa / Za
        w_axial[band] = np.maximum(w_axial[band], w_k)

        # swirl: antisymmetric about the prop axis, summed across props
        fs = sw_fn(s)
        sgn = np.sign(rho[band])
        sgn[sgn == 0.0] = 1.0
        v_swirl[band] = v_swirl[band] + spin_arr[k] * sgn * vsw_mag * fs / Zs

        in_wake[band] = True

    mu = 1.0 + w_axial / V
    # velocity-triangle swirl angle: exactly antisymmetric because the numerator
    # is odd across a prop axis while the denominator V + w_axial is even.
    dalpha = np.arctan2(v_swirl, V + w_axial)

    return SlipstreamProfile(
        y=y,
        w_axial=w_axial,
        v_swirl=v_swirl,
        mu=mu,
        dalpha=dalpha,
        in_wake=in_wake,
        frac_wake=float(np.sum(in_wake) / y.size),
    )


# ------------------------------------------------------------------ the spec / hook


@dataclass
class SlipstreamSpec:
    """Prop-slipstream footprint on the wing -> per-station modifiers the LLT/
    strip objective integration consumes.

    Design/config inputs (what the caller sets):
        D_p         propeller diameter [m].
        y_centres   prop spanwise centres [m] (list -- one per prop; physical
                    layouts use symmetric +/- pairs).
        mu_inf      developed-slipstream velocity ratio V_j/V_inf (band-mean),
                    >= 1 accelerated. DEFAULT 1.0 = OFF (no thrust).
        CT          OPTIONAL disk thrust coefficient; if given it OVERRIDES
                    mu_inf via actuator-disk momentum theory (below).
        x_station   axial prop -> wing station [m]. Documentation / validity
                    only: this model assumes the slipstream is FULLY DEVELOPED
                    at the wing (far-wake velocity & contraction), which needs
                    x_station a few diameters downstream. It does not (yet) enter
                    the profiles -- the GDP pipeline supplies the developed-wake
                    R_W via a Heyson contraction externally (see ``R_W`` below).
        spin        rotation sign(s) +/-1: scalar or per-prop (-1 CW / +1 CCW
                    from behind). DEFAULT -1.
        dalpha_ref_deg  Bell reference swirl angle magnitude [deg] (band-mean).
                    DEFAULT 0.0 = OFF (no swirl).
        core_frac   axial flat-core fraction (default 0.6).
        R_W         OPTIONAL explicit contracted wake radius [m]; if given it
                    OVERRIDES the momentum-theory contraction (use this to feed
                    a Heyson-model R_W, as the GDP strip pipeline does).
        n_img       image pairs in K_l (default 40).

    The OFF default (mu_inf=1, dalpha_ref_deg=0) makes ``modifiers`` return
    (ones, zeros) BIT-FOR-BIT -- the PHASE INVARIANT.

    Momentum-theory closed forms (derived)
    ---------------------------------------
    Actuator disk with axial induction a: disk speed V(1+a), developed-wake
    speed V(1+2a). So the band-mean developed ratio is mu_inf = 1 + 2a, giving
    the band-mean axial induced velocity w = (mu_inf - 1) V and axial induction
    a = (mu_inf - 1)/2. Mass conservation across the tube,
    pi R_p^2 V(1+a) = pi R_W^2 V(1+2a), gives the far-wake CONTRACTION

        R_W = R_p sqrt((1 + a)/(1 + 2a)) = R_p sqrt((1 + mu_inf)/(2 mu_inf)),

    which -> R_p exactly at mu_inf = 1 (no contraction). Disk thrust coefficient
    (freestream q, disk area) CT = 4 a (1 + a) inverts to a = (-1+sqrt(1+CT))/2,
    hence mu_inf = 1 + 2a = sqrt(1 + CT) (=> mu_inf = 1 at CT = 0).
    """

    D_p: float
    y_centres: Sequence[float]
    mu_inf: float = 1.0
    CT: float | None = None
    x_station: float = 0.0
    spin: object = -1.0
    dalpha_ref_deg: float = 0.0
    core_frac: float = DEFAULT_CORE_FRAC
    R_W: float | None = None
    n_img: int = DEFAULT_N_IMG

    def __post_init__(self):
        if self.CT is not None:
            if self.CT < -1.0:
                raise ValueError(f"CT must be >= -1 (got {self.CT})")
            # thrust coefficient overrides mu_inf (momentum theory, docstring)
            self.mu_inf = float(np.sqrt(1.0 + float(self.CT)))
        if self.mu_inf <= 0.0:
            raise ValueError(f"mu_inf must be > 0 (got {self.mu_inf})")

    # --- geometry ---------------------------------------------------------

    @property
    def R_prop(self) -> float:
        """Geometric prop radius R_p = D_p / 2 [m]."""
        return 0.5 * self.D_p

    @property
    def wake_radius(self) -> float:
        """Contracted (developed-wake) radius R_W [m].

        Uses the explicit ``R_W`` override if supplied (e.g. a Heyson value),
        else the momentum-theory contraction R_p sqrt((1+mu_inf)/(2 mu_inf)).
        """
        if self.R_W is not None:
            return float(self.R_W)
        return self.R_prop * float(np.sqrt((1.0 + self.mu_inf) / (2.0 * self.mu_inf)))

    @property
    def w_mean(self) -> float:
        """Band-mean axial induced velocity per unit freestream speed,
        w/V = mu_inf - 1 (the nondimensional momentum value the profile
        integral preserves). 0 when OFF."""
        return self.mu_inf - 1.0

    # --- profile ----------------------------------------------------------

    def _profile(self, y: np.ndarray) -> SlipstreamProfile:
        """Nondimensional slipstream profile at stations y (computed at V=1, so
        w_axial, v_swirl are per-unit-V => mu and dalpha are V-independent)."""
        return slipstream_profiles(
            y,
            self.y_centres,
            self.wake_radius,
            w=self.w_mean,             # = (mu_inf - 1) * V with V = 1
            dalpha_ref=self.dalpha_ref_deg,
            spin=self.spin,
            V=1.0,
            core_frac=self.core_frac,
        )

    # --- the hook ---------------------------------------------------------

    def modifiers(self, y, c) -> tuple[np.ndarray, np.ndarray]:
        """Per-station slipstream modifiers on the LLT/strip stations.

        Returns ``(q_ratio, dalpha)``, both arrays sized like ``y``:

          q_ratio[i] = mu(y_i)^2   -- LOCAL DYNAMIC-PRESSURE RATIO
              q_local/q_inf at station i. In strip theory the section force is
              q_local * c * coefficient, so q_ratio scales BOTH the section
              lift and the section PROFILE DRAG. mu here is the AXIAL velocity
              ratio (V + w_axial)/V (the swirl's second-order contribution to q
              is neglected -- it is carried as an angle instead). == 1 exactly
              outside every prop wake, and identically 1 when OFF.

          dalpha[i] = atan2(v_swirl_i, V + w_axial_i)  [rad] -- the SWIRL ANGLE
              ADDED to the local geometric AoA (upwash +ve). Antisymmetric about
              each prop axis; identically 0 when OFF.

        These two are the velocity-triangle modifiers and are CHORD-INDEPENDENT.
        ``c`` (per-station chord at the same stations as ``y``) is validated for
        alignment and is the length scale the SEPARATE finite-slipstream-height
        lift correction consumes -- see ``k_lift(y, c)`` / ``jet_lift_ratio``,
        which multiplies the 2-D lift SLOPE (a different, chord-dependent effect
        the objective applies to the lift term only). In the dormant cruise
        regime K_l ~ mu^2 ~ q_ratio, so the two views agree to < 0.5%.
        """
        y = np.asarray(y, dtype=float).ravel()
        c = np.asarray(c, dtype=float).ravel()
        if c.shape != y.shape:
            raise ValueError(
                f"c and y must have the same length (got {c.shape} vs {y.shape})"
            )
        prof = self._profile(y)
        q_ratio = prof.mu ** 2
        return q_ratio, prof.dalpha

    def k_lift(self, y, c, kernel: str = "image") -> np.ndarray:
        """Finite-slipstream-height lift-slope multiplier K_l at each station.

        K_l = jet_lift_ratio(mu(y), h/c) with jet height h = 2 R_W (the wake
        diameter) => h_over_c = 2 R_W / c. Outside the wake mu = 1 => K_l = 1
        exactly, whatever h/c. This is the chord-dependent lift correction that
        multiplies the 2-D slope; ``modifiers`` returns the (chord-independent)
        dynamic-pressure ratio and swirl angle.
        """
        y = np.asarray(y, dtype=float).ravel()
        c = np.asarray(c, dtype=float).ravel()
        if c.shape != y.shape:
            raise ValueError(
                f"c and y must have the same length (got {c.shape} vs {y.shape})"
            )
        prof = self._profile(y)
        h_over_c = 2.0 * self.wake_radius / c
        return np.asarray(
            jet_lift_ratio(prof.mu, h_over_c, n_img=self.n_img, kernel=kernel),
            dtype=float,
        )
