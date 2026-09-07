"""The two-element car-wing SECTION as a design problem, scored on a LAP.

What this module is for
-----------------------
``carwing_multi.py`` can fly a slotted section on a three-dimensional wing and
gives the two elements' ARRANGEMENT four design rows. It cannot design the two
elements' SHAPES: the wing's design vector is a planform, and putting sixteen
aerofoil weights into it would make a 27-row search out of a family the
repository's own optimiser study already records as defeating every method it
has at 7 and 10 rows (``RESULTS_WING_OPTIMISER_REFERENCE_VERDICT.md``).

So the section is designed HERE, in 2-D, where a candidate costs milliseconds,
and the wing is asked afterwards whether the winner helped it. That split is
the entire experimental design, and its weak point is stated rather than
hidden: **a better 2-D score is not a better wing.** This module therefore
ships with the reduction it uses written down, calibrated against the actual
three-dimensional solver, and reported with its own fit error — so that when
the transfer measurement disagrees, the disagreement has a place to live.

WHY THE OBJECTIVE IS A LAP AND NOT A WEIGHTED COMPOSITE
-------------------------------------------------------
A two-element rear wing trades CL_max, which is worth something in a corner,
against L/D, which is worth something on a straight. The obvious way to score
that is a weighted sum, and this repository has already MEASURED what that
does:

* `a-weight-is-not-an-exchange-rate` — a composite sells whatever it weights
  highest; ``dJ/dv = 100 w / span`` says the weights ARE the answer.
* `the-band-is-not-the-lever` — n = 42: the normalising band did nothing, and
  the exact-weights arm was the WORST of them.
* `j-is-not-the-wing` — +12.4 composite points over the seed handed the wing
  nothing at all (24/42 wins, p = 0.44).

A circuit states the same trade WITHOUT anyone choosing a weight: the corner
radii say how much downforce is worth, the straight lengths say how much drag
costs, and ``cartrack.lap_time`` integrates them into one physical scalar in
seconds. That is the objective. The weighted composite is kept — as a
pre-registered CONTROL (``objective="weighted"``), so that "the lap is the
better objective" is a measurement in this package and not an opinion in a
docstring.

The reduction: 2-D section -> wing -> lap
-----------------------------------------
A lap needs the wing's ``CZ*S`` and ``CD*S``. Getting them from a section
needs three-dimensional information, and the whole point of a 2-D study is
that the candidate does not get to consult the three-dimensional solver. So
the reduction is a FIXED bridge (:class:`SectionBridge`), measured once
against ``carwing_multi`` at a reference wing and then held constant for every
candidate:

1. **Lift — the finite-wing slope, with a calibrated span efficiency.** The
   lattice flies a LINEAR section: it is handed ``(a_lin, alpha_L0)`` and
   nothing else about the polar, so its CL is exactly affine in incidence
   (MEASURED: the lattice's CL differences over a 2-degree step at the
   reference wing are identical to six decimals). The bridge is therefore the
   textbook finite-wing slope on the same pair::

       a_3D = a_lin / (1 + a_lin / (pi * AR * e_lift))                    (1)
       CL   = a_3D * (alpha_g - alpha_L0) + dCL                          (2)
       alpha_eff = alpha_L0 + CL / a_lin                                  (3)

   with (3) not a modelling choice but ``vlm.py``'s own definition of a
   strip's effective incidence, copied so the two mean the same thing.

   and ``dCL`` is not a constant but a function of the section's own zero-lift
   angle::

       dCL = c0 + c_sin sin(alpha_L0) + c_cos (cos(alpha_L0) - 1)         (4)

   WHY THAT SHAPE, AND WHY THERE IS A CORRECTION AT ALL. The lattice's CL is
   **not** affine in ``alpha_L0``, and the reason is visible in ``vlm.py``:
   the zero-lift angle enters through the panel NORMALS (``theta = twist -
   alpha_L0``, then a Rodrigues rotation), so it appears as ``sin theta`` and
   ``cos theta`` inside both the right-hand side and the influence matrix.
   MEASURED at the reference wing with the section slope held at 6.347 /rad,
   sweeping ``alpha_L0`` from -20 to 0 deg in 5-degree steps, CL falls in
   steps of 0.4651, 0.4369, 0.4174, 0.4051 — a 15 % spread where an affine
   dependence would give four identical numbers. Equation (4) is therefore
   not a curve fitted to a residual for want of anything better: it is the
   functional form the geometry actually produces, with its coefficients
   measured. ``(cos - 1)`` rather than ``cos`` only so the three columns are
   not near-collinear over a range where the cosine is within 10 % of one.

   ``e_lift``, ``c0``, ``c_sin`` and ``c_cos`` are CALIBRATED over a stated
   SET of sections spanning the design box (:data:`BRIDGE_CALIBRATION_SLOTS`),
   never at one point, and the bridge then VALIDATES ITSELF on a second set it
   was not fitted on (:data:`BRIDGE_HOLDOUT_SLOTS`). MEASURED: 0.209 % mean
   and 0.819 % worst relative error in CL on the 108 fitted points, 0.258 %
   and 0.882 % on 80 held-out points across 7 sections it had never seen. A
   surrogate that is only reported on its own fit is not reported.

2. **Induced drag — measured, not fitted.** ``CDi = CL^2 / (pi AR e_drag)``
   with ``e_drag`` read straight off the lattice (``VLMResult.e``), which is
   what that number IS. Over a ground plane it is not the same as ``e_lift``
   and the two are kept apart for that reason.

3. **Everything the section is not.** ``CD0`` — the pylon, the junction and
   the endplate's own profile drag — is measured over the same set and added.
   It does not move with the section, which is exactly why it can be a
   constant here, and its spread over the set is reported so a reader can see
   the size of that "does not".

WHAT MAKES THIS A 2-D OBJECTIVE, and it is the thing to be suspicious of:
``AR``, ``e_lift``, ``dCL``, ``e_drag``, ``CD0``, ``S`` and the reference
chord are the SAME for every candidate. A section that would change the
lattice's span loading — and every section does, through its own lift-curve
slope and its own zero-lift angle — is scored as though it changed only the
two the bridge carries. That is the approximation the transfer measurement
exists to price, and the bridge reports its own residual against the lattice
it was calibrated on so a reader can see how big it was BEFORE the transfer
result arrives.

The section is flown at the lap's own speeds
--------------------------------------------
Each representative speed has its own Reynolds number, and each element its
own again through its own chord fraction. The section polar is therefore
rebuilt per speed — cheaply, because ``carwing_multi``'s cascade cache is
keyed on GEOMETRY alone and the two panel solves are what costs.

The design vector
-----------------
Per element: ``n_cst`` upper-surface CST weights, ``n_cst`` lower-surface
weights, and ONE thickness row. Then the four arrangement rows and the wing's
rigging incidence::

    x = [w_u_main (n), w_l_main (n), tc_main,
         w_u_flap (n), w_l_flap (n), tc_flap,
         flap_chord_frac, flap_deflection_deg,
         slot_gap_frac, slot_overlap_frac,
         alpha_deg]

WHY THICKNESS IS ITS OWN ROW rather than whatever the weights happen to give.
The stall criterion is built from a MEASURED stall, and the shipped wide-alpha
bank resolves one only for t/c in [0.09, 0.18] — outside that
``carwing_multi.suction_peak_ceiling`` refuses, correctly, because a censored
peak is a lower bound and not a peak. Left to the raw weights, a large part of
the box would be refused for that one reason and the optimiser would spend its
budget discovering a table's limits. So the thickness is stated directly and
BOUNDED to the resolved range, and the weights carry the shape. The rescaling
is exact and linear in the weights (:func:`scaled_weights`): with camber
``(w_u + w_l)/2`` and half-thickness ``(w_u - w_l)/2``, multiplying the
half-thickness by ``k`` is

    w_u' = (1+k)/2 w_u + (1-k)/2 w_l,   w_l' = (1-k)/2 w_u + (1+k)/2 w_l

and since ``y_u' - y_l' = k (y_u - y_l)``, a non-crossing section stays
non-crossing for any ``k > 0`` — so ``airfoil.anchor_box``'s proof that no
point of its box self-intersects survives the rescaling.

It costs ONE redundant direction per element (the overall scale of the
half-thickness part of the weights is undone by the rescaling), which is
stated because a degenerate direction is a real thing to know about a search
space, and accepted because the alternative is a refusal class that eats the
box.

What is NOT modelled
--------------------
* Everything ``carwing_multi`` and ``panel2d`` already list: viscosity in the
  slot, wakes, the cove, spanwise variation of the slot, compressibility.
* THE WING'S RESPONSE TO THE SECTION, by construction (above). The bridge is
  fixed; only the transfer measurement can say what that cost.
* The structural deflection limit and the drag BUDGET. The lap prices drag
  physically — every newton of it is paid for on the straight — and charging
  a coefficient allowance on top would charge the same drag twice, once by
  physics and once by calibration. The three-dimensional transfer is run both
  ways (budget on, budget off) for exactly this reason.
* Any spanwise or planform freedom. The reference wing's planform is FIXED at
  the bridge's own calibration point; the section is the only thing designed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import airfoil, carwing_multi
# Only the two this module READS. The four slot BOUNDS are deliberately not
# imported: the box below is built from ``carwing_multi.slot_rows()``, which
# is that family's single statement of all four, and importing them here as
# well would be a second place for the same question to be answered.
from .carwing_multi import N_SECTION_NODES, SLOT_GAP_TE_NON_MERGING_FRAC

__all__ = [
    "ALPHA_BOUNDS_DEG", "SECTION_OBJECTIVES", "SectionBridge",
    "SectionProblem", "TC_BOUNDS", "WEIGHTED_REFERENCE",
    "BRIDGE_CALIBRATION_SLOTS", "BRIDGE_HOLDOUT_SLOTS", "REFUSAL_SCORE",
    "calibrate_bridge", "default_bridge", "element_from_x",
    "evaluate_section", "f_section", "fg_section", "reference_x",
    "resolved_tc_bounds", "scaled_weights", "section_coords_of",
    "textbook_bridge",
]

#: objectives this problem can score, and what each maximises.
SECTION_OBJECTIVES: dict[str, str] = {
    "laptime": "minus the lap time [-s] (maximise => minimise the lap)",
    "weighted": "a weighted sum of cl_max and (l/d)_max, both normalised "
                "(the pre-registered CONTROL — see the module docstring)",
}

#: the WING's rigging incidence band, and it is deliberately
#: ``carwing_multi.CarWingMultiProblem.ALPHA_BOUNDS_DEG`` verbatim: the same
#: question must be asked in the same place and over the same band in 2-D and
#: in 3-D, or the transfer measurement compares two different searches.
ALPHA_BOUNDS_DEG = (0.0, 12.0)

#: each element's thickness band: the range over which the shipped wide-alpha
#: NACA 24XX bank RESOLVES a stall, which is what the suction-peak ceiling is
#: built from. Not taste and not a structural limit — outside it there is no
#: measured stall to build a ceiling on, and ``carwing_multi`` refuses. Stated
#: as a constant here so the box cannot silently drift off the anchors; the
#: anchors themselves are read from the bank at call time by
#: :func:`resolved_tc_bounds`, which is the tripwire.
TC_BOUNDS = (0.09, 0.18)

#: the weighted CONTROL's weights, fixed BEFORE any run
#: (``PREREG_SESSION64_CARSECTION.md``). Equal is the only choice that is not
#: itself a claim about the exchange rate, which is the point being tested:
#: the lap does not need one.
WEIGHTED_REFERENCE = (0.5, 0.5)


def resolved_tc_bounds() -> tuple:
    """``(lo, hi)`` thicknesses at which the shipped bank resolves a stall.

    Derived from the tables at call time, never pasted. :data:`TC_BOUNDS` is
    the constant the design box uses; this is what it must equal, and
    ``test_the_thickness_band_is_the_resolved_one`` is the tripwire that says
    so — because a bank that grew a thinner resolved member, or lost one,
    would silently make the box wrong in one direction or the other.
    """
    from .polar import stall_point, stall_polar_family

    fam = stall_polar_family()
    ok = sorted(t for t, p in fam.members.items()
                if not stall_point(p, t).censored)
    return (float(ok[0]), float(ok[-1]))


# ------------------------------------------------------------- the geometry


def scaled_weights(w_upper, w_lower, tc: float):
    """CST weights rescaled to an exact maximum thickness/chord.

    DERIVED (module docstring): with camber ``(w_u + w_l)/2`` and
    half-thickness ``(w_u - w_l)/2``, scaling the half-thickness by ``k``
    leaves the camber line alone and multiplies ``y_u - y_l`` by ``k``
    everywhere, so the maximum thickness is ``k`` times what it was and the
    location of the maximum does not move. Hence ``k = tc / tc(w)`` hits the
    target exactly rather than iteratively.

    Returns ``(w_u', w_l')``. Raises ValueError for a degenerate section (one
    with no thickness to scale) — that is a call-site error; a design that
    reaches it has been built wrong upstream.
    """
    wu = np.asarray(w_upper, dtype=float)
    wl = np.asarray(w_lower, dtype=float)
    t0 = float(airfoil.cst_thickness(wu, wl))
    if not t0 > 1e-9:
        raise ValueError(
            f"cannot rescale a section of thickness {t0:.3g} to t/c "
            f"{float(tc):.4f}: there is no thickness to scale")
    k = float(tc) / t0
    return (0.5 * (1.0 + k) * wu + 0.5 * (1.0 - k) * wl,
            0.5 * (1.0 - k) * wu + 0.5 * (1.0 + k) * wl)


def element_from_x(x, i0: int, n_cst: int):
    """``(w_upper, w_lower, tc)`` for one element, read from the vector."""
    xs = np.asarray(x, dtype=float)
    wu = xs[i0:i0 + n_cst]
    wl = xs[i0 + n_cst:i0 + 2 * n_cst]
    return wu, wl, float(xs[i0 + 2 * n_cst])


def section_coords_of(w_upper, w_lower, tc: float, n_nodes: int):
    """The element's closed unit-chord loop at the stated thickness.

    ``dz_te = 0``: ``panel2d`` refuses a blunt trailing edge and measures why
    (an error that GROWS under refinement), so the sections this family
    designs are sharp by construction rather than closed after the fact.
    """
    wu, wl = scaled_weights(w_upper, w_lower, tc)
    return airfoil.cst_coords(wu, wl, int(n_nodes), dz_te=0.0)


# --------------------------------------------------------------- the bridge


@dataclass(frozen=True)
class SectionBridge:
    """The FIXED 2-D section -> 3-D wing reduction, and where it came from.

    Every field is measured (or derived in closed form from something
    measured) at ONE reference wing, and then held constant for every
    candidate section. That constancy is what makes the objective 2-D; see
    the module docstring for the three relations and
    :func:`calibrate_bridge` for how each number is obtained.

    ``fit_report`` carries the residual of the reduction against the lattice
    it was calibrated on — the honest statement of how well it reproduces the
    thing it stands in for, available BEFORE any transfer result.
    """

    S_m2: float
    b_m: float
    mac_m: float
    AR: float
    e_lift: float
    #: equation (4)'s three coefficients: ``(c0, c_sin, c_cos)``.
    dCL_coef: tuple
    e_drag: float
    CD0: float
    ride_height_m: float
    rho: float
    mu: float
    #: the reference wing and the sections it was calibrated over — so the
    #: calibration is reproducible from the record rather than from a memory
    #: of how it was run
    source: str = ""
    fit_report: dict = field(default_factory=dict, repr=False)

    def re_ref(self, V: float) -> float:
        """Reynolds number on the STOWED chord at a stated speed."""
        return float(self.rho) * float(V) * float(self.mac_m) / float(self.mu)

    def a_3d(self, a_lin: float) -> float:
        """Equation (1): the finite-wing slope this section would give."""
        a = float(a_lin)
        return a / (1.0 + a / (np.pi * float(self.AR) * float(self.e_lift)))

    def dCL(self, alpha_L0_rad: float) -> float:
        """Equation (4): the lattice's non-affine response to ``alpha_L0``."""
        c0, cs, cc = (float(v) for v in self.dCL_coef)
        a = float(alpha_L0_rad)
        return c0 + cs * np.sin(a) + cc * (np.cos(a) - 1.0)

    def wing_CL(self, section, alpha_g_deg: float):
        """``(CL, alpha_eff_deg)`` from equations (2)-(4), or
        ``(None, alpha_eff_deg)``.

        ``None`` when the effective incidence leaves the section's own flyable
        window: the section cannot be flown there, which is a refusal and not
        a number to clip. Closed form — there is nothing to iterate, because
        the lattice this stands in for is itself linear in incidence.
        """
        lo, hi = section.alpha_valid
        a_lin = float(section.a_lin)
        aL0 = float(section.alpha_L0)                     # [rad]
        CL = (self.a_3d(a_lin) * (np.deg2rad(float(alpha_g_deg)) - aL0)
              + self.dCL(aL0))
        a_eff = np.rad2deg(aL0 + CL / a_lin)
        if not np.isfinite(CL) or a_eff < lo or a_eff > hi:
            return None, float(a_eff)
        return float(CL), float(a_eff)


def reference_x(prob=None):
    """``(x, prob)`` — the 3-D reference design the bridge is measured at.

    The design box's own CENTRE for ``carwing_multi``'s published family,
    because a bridge calibrated at a corner would be calibrated where the
    wing is unusual.
    """
    prob = prob or carwing_multi.CarWingMultiProblem()
    return prob.bounds.mean(axis=1), prob


#: the SECTIONS the bridge is calibrated over, as ``(flap_chord_frac,
#: flap_deflection_deg, slot_gap_frac, slot_overlap_frac)``.
#:
#: A set, not a point, and that is the whole reason it is a named constant.
#: The two fitted constants have to stand in for the lattice across the range
#: of ``(a_lin, alpha_L0)`` the design box produces, and the section's
#: zero-lift angle is what moves most: it runs from about -3.6 deg at a barely
#: deflected flap to about -21 deg at the top of the deflection band. A bridge
#: fitted at one section would be exact there and would carry that section's
#: geometry into every other candidate's score.
#:
#: The nine points are the corners and centre of the (chord fraction,
#: deflection) plane at the gap/overlap centre, plus the two ends of the gap
#: row — chosen because those are the two rows the measured ``alpha_L0``
#: actually responds to. Infeasible members are DROPPED, not replaced: at the
#: top of the deflection band some flap fractions have no flyable incidence at
#: all, and substituting a nearby feasible point would quietly re-centre the
#: calibration.
BRIDGE_CALIBRATION_SLOTS = (
    (0.18, 2.0, 0.031, 0.02), (0.18, 17.5, 0.031, 0.02),
    (0.18, 32.0, 0.031, 0.02),
    (0.275, 2.0, 0.031, 0.02), (0.275, 17.5, 0.031, 0.02),
    (0.275, 32.0, 0.031, 0.02),
    (0.38, 2.0, 0.031, 0.02), (0.38, 17.5, 0.031, 0.02),
    (0.38, 32.0, 0.031, 0.02),
    (0.275, 17.5, 0.015, 0.02), (0.275, 17.5, 0.045, 0.02),
)

#: sections the bridge is VALIDATED on and never fitted to. Seven riggings
#: that share no row value with :data:`BRIDGE_CALIBRATION_SLOTS`, spread over
#: the whole box including both overlap extremes — which the calibration set
#: deliberately does not vary, so the held-out error also says what holding
#: the overlap fixed during the fit cost.
#:
#: A surrogate reported only on its own fit is not reported. This is the
#: measurement that says the bridge generalises: MEASURED 0.258 % mean and
#: 0.882 % worst relative CL error over 80 held-out points, against 0.209 %
#: and 0.819 % on the 108 it was fitted to.
BRIDGE_HOLDOUT_SLOTS = (
    (0.22, 9.0, 0.020, 0.00), (0.33, 25.0, 0.040, 0.05),
    (0.16, 30.0, 0.025, -0.01), (0.40, 12.0, 0.050, 0.06),
    (0.20, 22.0, 0.035, 0.03), (0.30, 5.0, 0.013, -0.02),
    (0.36, 20.0, 0.018, 0.04),
)


def _sweep_sections(x, prob, alphas, slots):
    """``(CL, alpha_g_rad, a_lin, alpha_L0_rad, e, CD0, n_sections)``.

    One lattice sweep per rigging. A rigging that flies at fewer than three
    incidences contributes NOTHING rather than its two points: three is the
    fewest that can distinguish a slope from a pair of samples, and a section
    represented by two points would weight the fit by how nearly infeasible
    it is.
    """
    rows, n = [], 0
    for slot in slots:
        got = []
        for a in alphas:
            xx = np.asarray(x, dtype=float).copy()
            xx[3] = float(a)
            xx[carwing_multi.SLOT_ROW_OFFSET:
               carwing_multi.SLOT_ROW_OFFSET + 4] = slot
            out = carwing_multi.evaluate_car_wing_multi(xx, prob)
            if out["feasible"]:
                got.append(out)
        if len(got) >= 3:
            n += 1
            rows += got
    if not rows:
        return None, n
    return {
        "rows": rows,
        "CL": np.array([r["CL_model"] for r in rows], dtype=float),
        "alpha_g": np.deg2rad([r["alpha_deg"] for r in rows]),
        "a_lin": np.array([r["section_a_lin"] for r in rows], dtype=float),
        "alpha_L0": np.deg2rad([r["section_alpha_L0_deg"] for r in rows]),
        "e": np.array([r["e"] for r in rows], dtype=float),
        "cd0": np.array([r["cd0_struts"] + r["CD_junction"]
                         + r["CDp_endplate"] for r in rows], dtype=float),
    }, n


def _dCL_basis(alpha_L0: np.ndarray) -> np.ndarray:
    """Equation (4)'s three columns, in ``dCL_coef``'s own order."""
    a = np.asarray(alpha_L0, dtype=float)
    return np.column_stack([np.ones_like(a), np.sin(a), np.cos(a) - 1.0])


def calibrate_bridge(prob=None, alpha_deg=None, x_ref=None,
                     slots=BRIDGE_CALIBRATION_SLOTS,
                     holdout=BRIDGE_HOLDOUT_SLOTS,
                     e_lift: float | None = None,
                     dCL_coef: tuple | None = None) -> SectionBridge:
    """Measure the 2-D -> wing reduction against ``carwing_multi``.

    Lattice sweeps at the reference wing over
    :data:`BRIDGE_CALIBRATION_SLOTS` x the family's own incidence band. From
    them:

    * ``AR``, ``S``, ``mac``, the ride height and the air: read off the wing.
    * ``e_drag``: the lattice's own span efficiency, averaged over the set
      (it is what that number is; the spread is reported).
    * ``CD0``: the pylon + junction + endplate profile drag, averaged over the
      set. Its spread is reported too — it is treated as a constant and a
      reader is owed the size of that assumption.
    * ``e_lift`` and ``dCL_coef``: least squares of equations (1)-(4) against
      every ``(section, alpha)`` pair. ``e_lift`` enters nonlinearly and the
      three ``dCL`` coefficients linearly, so ``e_lift`` is scanned and the
      others are its inner linear solution at each trial. The criterion is
      MEAN RELATIVE error in CL, not sum of squares, because the wing's CL
      spans a factor of two over the set and a squared-error fit would spend
      its accuracy on the high-CL end.
      Both may be PASSED instead, which is how the textbook control bridge is
      built (``e_lift = 1``, all coefficients zero) without a second code
      path.

    ``holdout`` is a second set of sections the fit never sees; the error on
    it is reported beside the error on the fit. Pass ``()`` to skip it (it
    costs one more sweep), and read a bridge with no held-out number as a
    bridge that has not been validated.

    Raises ValueError if fewer than three sections fly: a bridge with no
    reference is a call-site error, not a design failure.
    """
    x, prob = (reference_x(prob) if x_ref is None
               else (np.asarray(x_ref, dtype=float),
                     prob or carwing_multi.CarWingMultiProblem()))
    lo, hi = (ALPHA_BOUNDS_DEG if alpha_deg is None
              else (float(alpha_deg[0]), float(alpha_deg[-1])))
    alphas = (np.linspace(lo, hi, 13) if alpha_deg is None
              else np.asarray(alpha_deg, dtype=float))

    fit, n_fit = _sweep_sections(x, prob, alphas, slots)
    if n_fit < 3:
        raise ValueError(
            f"only {n_fit} of {len(slots)} calibration sections fly at 3 or "
            f"more incidences on this reference wing: there is no set to "
            f"calibrate a bridge on. Give calibrate_bridge a reference "
            f"design (x_ref) or a slot set that flies")

    rows = fit["rows"]
    CL, a2d, aL0 = fit["CL"], fit["a_lin"], fit["alpha_L0"]
    dalpha = fit["alpha_g"] - aL0
    AR = float(rows[0]["AR"])
    basis = _dCL_basis(aL0)

    def _err(el: float):
        """``(mean relative error, coefficients)`` at this ``e_lift``."""
        a3 = a2d / (1.0 + a2d / (np.pi * AR * el))
        coef, *_ = np.linalg.lstsq(basis, CL - a3 * dalpha, rcond=None)
        pred = a3 * dalpha + basis @ coef
        return float(np.mean(np.abs(pred - CL) / np.abs(CL))), coef

    if e_lift is None:
        # a scan, not a descent: the objective is a mean of absolute values
        # and is not smooth, so a bracketed line search can settle in a kink.
        # 3000 points over two and a half decades is 0.2 % in e_lift, which
        # is far finer than the quantity is determined to.
        grid = np.geomspace(0.5, 40.0, 3000)
        errs = np.array([_err(v)[0] for v in grid])
        e_lift = float(grid[int(np.argmin(errs))])
    e_lift = float(e_lift)
    coef = (np.asarray(_err(e_lift)[1], dtype=float) if dCL_coef is None
            else np.asarray(dCL_coef, dtype=float))

    wing = rows[0]["wing"]
    bridge = SectionBridge(
        S_m2=float(rows[0]["S_m2"]), b_m=float(rows[0]["b_m"]),
        mac_m=float(wing.mac), AR=AR,
        e_lift=e_lift, dCL_coef=tuple(float(v) for v in coef),
        e_drag=float(np.mean(fit["e"])), CD0=float(np.mean(fit["cd0"])),
        ride_height_m=float(rows[0]["ride_height_m"]),
        rho=float(prob.rho), mu=float(prob.mu),
        source=(f"carwing_multi at x = box centre, {n_fit} of {len(slots)} "
                f"calibration sections x {alphas.size} incidences in "
                f"[{lo:g}, {hi:g}] deg ({len(rows)} points)"),
    )

    def _residual(pack):
        a3 = pack["a_lin"] / (1.0 + pack["a_lin"] / (np.pi * AR * e_lift))
        pred = (a3 * (pack["alpha_g"] - pack["alpha_L0"])
                + _dCL_basis(pack["alpha_L0"]) @ coef)
        rel = np.abs(pred - pack["CL"]) / np.abs(pack["CL"])
        return {"n": int(pack["CL"].size),
                "rel_mean": float(np.mean(rel)),
                "rel_max": float(np.max(rel)),
                "abs_max": float(np.max(np.abs(pred - pack["CL"])))}

    report = {
        "n_points": int(CL.size), "n_sections": int(n_fit),
        "fit": _residual(fit),
        "a_lin_range": (float(a2d.min()), float(a2d.max())),
        "alpha_L0_deg_range": (float(np.rad2deg(aL0).min()),
                               float(np.rad2deg(aL0).max())),
        "e_lattice_min": float(fit["e"].min()),
        "e_lattice_max": float(fit["e"].max()),
        "CD0_min": float(fit["cd0"].min()), "CD0_max": float(fit["cd0"].max()),
    }
    if holdout:
        held, n_held = _sweep_sections(x, prob, alphas, holdout)
        report["holdout"] = (None if held is None else
                             {**_residual(held), "n_sections": int(n_held)})
    object.__setattr__(bridge, "fit_report", report)
    return bridge


def textbook_bridge(prob=None, **kw) -> SectionBridge:
    """The CONTROL bridge: elliptic loading, no additive term, no fit.

    ``e_lift = 1`` and ``dCL = 0`` — the reduction a textbook would write down
    for a plain wing in free air, with nothing calibrated in it at all. The
    geometry (``AR``, ``S``, ``mac``), the induced-drag efficiency and
    ``CD0`` are still measured, because those are not modelling choices.

    It exists to answer one question and it is a question about this study
    rather than about the wing: does the ANSWER move when the reduction's two
    fitted constants are replaced by the values nobody fitted? A ranking that
    survives it does not rest on the calibration.
    """
    return calibrate_bridge(prob, e_lift=1.0, dCL_coef=(0.0, 0.0, 0.0), **kw)


_BRIDGE_CACHE: dict = {}


def default_bridge() -> SectionBridge:
    """The reference bridge, measured once per process."""
    hit = _BRIDGE_CACHE.get("default")
    if hit is None:
        hit = calibrate_bridge()
        _BRIDGE_CACHE["default"] = hit
    return hit


# -------------------------------------------------------------- the problem


@dataclass
class SectionProblem:
    """Design BOTH elements' shapes and their arrangement, against a lap."""

    n_cst: int = 4
    n_section_nodes: int = N_SECTION_NODES
    #: nodes in each element's generated CST loop. Separate from
    #: ``n_section_nodes`` (which is the NACA generator's count in
    #: ``carwing_multi``) because a supplied section is flown at its OWN
    #: sampling — so THIS is the count the cascade actually panels, and the
    #: count a measured ceiling has to be read on.
    n_coord_nodes: int = N_SECTION_NODES
    bridge: Any = None
    car_spec: Any = None
    track_spec: Any = None
    track_points: int = 4
    objective: str = "laptime"
    weights: tuple = WEIGHTED_REFERENCE
    #: the reference ``(cl_max, ld_max)`` the weighted control normalises by.
    #: ``None`` measures them from :meth:`reference_scores` at construction —
    #: i.e. from the shipped NACA 24XX cascade at the box-centre arrangement,
    #: which is the design this study's improvements are quoted against.
    weighted_reference: tuple | None = None
    #: the non-merging floor on the TRAILING-EDGE gap. ON by default here and
    #: off by default in ``carwing_multi``, and the asymmetry is deliberate:
    #: this module is a SEARCH, and an inviscid search left free rides the
    #: tightest slot the box allows (measured — see the results), while
    #: ``carwing_multi``'s default may not move because every number that
    #: family has published was measured without it.
    slot_gap_te_min_frac: float | None = SLOT_GAP_TE_NON_MERGING_FRAC
    ceiling_scale: float = 1.0
    #: whose loop each element's substituted ceiling is read on
    #: (``carwing_multi.CEILING_SHAPES``). ``"naca"`` here and there: the
    #: alternative is VACUOUS INSIDE A SEARCH — it lets a section's own peak
    #: at a fixed angle set that section's own ceiling, so it rewards a sharp
    #: nose, and both of this study's first-attempt 2-D winners came out
    #: infeasible the moment their stall was measured on them (that constant's
    #: comment carries the numbers). It is a field rather than a constant
    #: because the measurement that shows why needs both.
    ceiling_shape: str = "naca"
    #: MEASURED ceilings ``((cp, a), (cp, a))``, from
    #: ``section_stall.measured_ceilings``. ``None`` is the NACA 24XX
    #: substitution. A measurement costs an XFOIL process per element, so it
    #: belongs on the designs a study VERIFIES and not in this loop — which is
    #: why it is a field to be set, not a mode to be searched under.
    section_ceilings: Any = None
    tc_bounds: tuple = TC_BOUNDS
    anchor: str = "2412"

    def __post_init__(self):
        if self.objective not in SECTION_OBJECTIVES:
            raise ValueError(
                f"unknown objective {self.objective!r}; "
                f"choose from {sorted(SECTION_OBJECTIVES)}")
        if int(self.n_cst) < 1:
            raise ValueError(f"n_cst must be >= 1, got {self.n_cst!r}")
        lo, hi = (float(v) for v in self.tc_bounds)
        rlo, rhi = resolved_tc_bounds()
        if lo < rlo - 1e-12 or hi > rhi + 1e-12:
            raise ValueError(
                f"tc_bounds {self.tc_bounds!r} reaches outside the range the "
                f"shipped wide-alpha bank RESOLVES a stall over "
                f"[{rlo:g}, {rhi:g}]. Outside it there is no measured stall "
                f"to build a suction-peak ceiling on and every candidate "
                f"there is refused for that one reason")
        if self.objective == "laptime" and self.track_spec is None:
            from .cartrack import synthetic_lap
            self.track_spec = synthetic_lap()
        if self.bridge is None:
            self.bridge = default_bridge()
        if self.objective == "weighted" and self.weighted_reference is None:
            self.weighted_reference = self.reference_scores()

    # ---- the box

    @property
    def element_box(self):
        """``(lo, hi)`` for ONE element's ``2 n_cst + 1`` rows."""
        wu, wl = airfoil.cst_anchor(self.anchor, int(self.n_cst))
        lo, hi = airfoil.anchor_box(wu, wl)
        return (np.concatenate([lo, [float(self.tc_bounds[0])]]),
                np.concatenate([hi, [float(self.tc_bounds[1])]]))

    @property
    def bounds(self) -> np.ndarray:
        elo, ehi = self.element_box
        return np.vstack([
            np.column_stack([elo, ehi]),          # main element
            np.column_stack([elo, ehi]),          # flap
            carwing_multi.slot_rows(),
            np.array([ALPHA_BOUNDS_DEG], dtype=float),
        ])

    @property
    def dim(self) -> int:
        return int(self.bounds.shape[0])

    @property
    def n_element_rows(self) -> int:
        return 2 * int(self.n_cst) + 1

    @property
    def param_labels(self) -> tuple:
        n = int(self.n_cst)
        out = []
        for el in ("main", "flap"):
            out += [f"w_upper_{el}_{i + 1}" for i in range(n)]
            out += [f"w_lower_{el}_{i + 1}" for i in range(n)]
            out.append(f"tc_{el}")
        out += ["flap_chord_frac", "flap_deflection_deg",
                "slot_gap_frac", "slot_overlap_frac", "alpha_deg"]
        return tuple(out)

    @property
    def constraint_labels(self) -> tuple:
        """None. Every limit this problem has is a REFUSAL.

        The section's limits — an intersecting flap, a slot below the
        resolution floor, a trailing-edge gap below the non-merging floor, an
        element past its measured suction-peak ceiling — have no meaningful
        "how far past it are we" that an optimiser could walk down, which is
        the shape ``carwing_multi.constraint_labels`` gives for the same
        reason. And the drag BUDGET is deliberately absent: the lap prices
        drag physically (module docstring).
        """
        return ()

    @property
    def n_constraints(self) -> int:
        return 0

    # ---- the reference design, and what the control normalises by

    def reference_x(self) -> np.ndarray:
        """The shipped section at the box-centre arrangement.

        Both elements the CST projection of the anchor NACA at t/c 0.12 —
        which is the ``carwing_multi`` default section — and the four slot
        rows at their own band centres. This is the design every improvement
        in this study is quoted against, so it is constructed rather than
        described.
        """
        wu, wl = airfoil.cst_anchor(self.anchor, int(self.n_cst))
        el = np.concatenate([wu, wl, [0.12]])
        slots = carwing_multi.slot_rows().mean(axis=1)
        return np.concatenate([el, el, slots,
                               [float(np.mean(ALPHA_BOUNDS_DEG))]])

    def reference_scores(self) -> tuple:
        """``(cl_max, ld_max)`` of :meth:`reference_x`, over its own window.

        Evaluated with the normalisation set to ``(1, 1)`` first, because the
        two numbers being measured here are what the normalisation is FOR and
        the evaluator would otherwise need the answer to produce it. Nothing
        else about the evaluation depends on the pair, so the placeholder
        cannot leak into the measurement.
        """
        was, self.weighted_reference = self.weighted_reference, (1.0, 1.0)
        try:
            got = evaluate_section(self.reference_x(), self)
        finally:
            self.weighted_reference = was
        if not got["feasible"]:
            raise ValueError(
                f"the reference section is not flyable, so there is nothing "
                f"to normalise the weighted control by: {got['reason']}")
        return (float(got["cl_max"]), float(got["ld_max"]))


def _element_coords(x, i0: int, prob: SectionProblem):
    wu, wl, tc = element_from_x(x, i0, int(prob.n_cst))
    return section_coords_of(wu, wl, tc, int(prob.n_coord_nodes)), float(tc)


def evaluate_section(x, prob: SectionProblem | None = None) -> dict:
    """Full evaluation with breakdown; never raises for in-contract failures.

    Returns ``feasible=False`` with a ``reason`` for every way a section can
    fail to be a section — a degenerate shape, an intersecting flap, a slot
    below either floor, an element past its ceiling, an incidence the wing
    cannot fly the section at, a lap the car cannot complete.
    """
    from .objective import _fail

    prob = prob or SectionProblem()
    x = np.asarray(x, dtype=float)
    bnds = prob.bounds
    if x.shape != (bnds.shape[0],):
        return _fail(f"design vector shape {x.shape} != ({bnds.shape[0]},)")
    if np.any(x < bnds[:, 0] - 1e-12) or np.any(x > bnds[:, 1] + 1e-12):
        return _fail("bounds violation")

    n_el = prob.n_element_rows
    try:
        main_c, tc_main = _element_coords(x, 0, prob)
        flap_c, tc_flap = _element_coords(x, n_el, prob)
    except ValueError as exc:
        return _fail(f"element shape: {exc}")

    ff, defl, gap, ovl = (float(v) for v in x[2 * n_el:2 * n_el + 4])
    alpha_g = float(x[2 * n_el + 4])
    br = prob.bridge

    def _section_at(V: float):
        return carwing_multi.cascade_polar(
            tc_main, ff, defl, gap, ovl, br.re_ref(V),
            n_nodes=int(prob.n_section_nodes),
            coords=main_c, flap_coords=flap_c,
            ceiling_scale=float(prob.ceiling_scale),
            gap_te_min_frac=prob.slot_gap_te_min_frac,
            ceilings=prob.section_ceilings,
            ceiling_shape=prob.ceiling_shape)

    # --- the base point: the reference speed the bridge was calibrated at
    V0 = _reference_speed(br)
    pol, reason = _section_at(V0)
    if reason is not None:
        return _fail(f"section: {reason}")

    got = _wing_point(pol, alpha_g, br)
    if got is None:
        lo, hi = pol.alpha_valid
        return _fail(
            f"the wing cannot fly this section at {alpha_g:.2f} deg of "
            f"rigging: the effective incidence the reduction settles on "
            f"leaves the section's flyable window [{lo:.2f}, {hi:.2f}] deg",
            alpha_valid=(float(lo), float(hi)))
    CL, CD, a_eff = got

    # --- the section's own headline numbers, over its FLYABLE window only
    lo, hi = pol.alpha_valid
    win = (pol.alpha_deg >= lo - 1e-9) & (pol.alpha_deg <= hi + 1e-9)
    cl_w, cd_w = pol.CL[win], pol.CD[win]
    cl_max = float(np.max(cl_w))
    ld = cl_w / np.where(cd_w > 0.0, cd_w, np.nan)
    ld_max = float(np.nanmax(ld))
    a_clmax = float(pol.alpha_deg[win][int(np.argmax(cl_w))])
    a_ldmax = float(pol.alpha_deg[win][int(np.nanargmax(ld))])

    lap = track_rows = None
    if prob.track_spec is not None:
        from . import cartrack
        car = prob.car_spec or cartrack.CarSpec()
        seed = cartrack.representative_points(
            CL * br.S_m2, CD * br.S_m2, car, prob.track_spec,
            n_points=int(prob.track_points))
        if not seed["feasible"]:
            return _fail(f"track: {seed['reason']}")
        rows = []
        for pt in sorted(seed["points"], key=lambda p: p["V"]):
            V = float(pt["V"])
            pv, why = _section_at(V)
            if why is not None:
                return _fail(f"the lap flies this section at {V:.4g} m/s: "
                             f"{why}")
            gv = _wing_point(pv, alpha_g, br)
            if gv is None:
                return _fail(
                    f"the lap flies this wing at {V:.4g} m/s and the "
                    f"effective incidence there leaves the section's window "
                    f"{pv.alpha_valid}")
            rows.append({"V": V, "weight": float(pt["weight"]),
                         "CZ": gv[0], "CD": gv[1],
                         "cz_a_m2": gv[0] * br.S_m2,
                         "cd_a_m2": gv[1] * br.S_m2,
                         "Re_ref": br.re_ref(V)})
        Vs = np.array([r["V"] for r in rows], dtype=float)
        cza = np.array([r["cz_a_m2"] for r in rows], dtype=float)
        cda = np.array([r["cd_a_m2"] for r in rows], dtype=float)
        lap = cartrack.lap_time(lambda V: float(np.interp(V, Vs, cza)),
                                lambda V: float(np.interp(V, Vs, cda)),
                                car, prob.track_spec)
        if not lap["feasible"]:
            return _fail(f"lap: {lap['reason']}")
        track_rows = rows

    if prob.objective == "laptime":
        score = -float(lap["lap_time_s"])
    else:
        ref_cl, ref_ld = prob.weighted_reference
        w1, w2 = (float(v) for v in prob.weights)
        score = w1 * cl_max / ref_cl + w2 * ld_max / ref_ld

    g_main, g_flap = pol.stall_margin(a_eff)
    return {
        "feasible": True, "reason": "",
        "score": float(score), "objective": prob.objective,
        "objective_label": SECTION_OBJECTIVES[prob.objective],
        "g": [], "constraint_labels": [],
        "lap_time_s": (None if lap is None else float(lap["lap_time_s"])),
        "lap": lap, "track_points": track_rows,
        "CZ": float(CL), "CD": float(CD), "CD_bridge_CD0": float(br.CD0),
        "CDi": float(CL * CL / (np.pi * br.AR * br.e_drag)),
        "cd_section": float(pol.cd(a_eff)),
        "alpha_deg": alpha_g, "alpha_eff_deg": float(a_eff),
        "cl_max": cl_max, "ld_max": ld_max,
        "alpha_cl_max_deg": a_clmax, "alpha_ld_max_deg": a_ldmax,
        "alpha_valid_deg": (float(lo), float(hi)),
        "tc_main": float(pol.tc), "tc_flap": float(pol.tc_flap),
        "flap_chord_frac": ff, "flap_deflection_deg": defl,
        "slot_gap_frac": gap, "slot_overlap_frac": ovl,
        "slot_width_frac": float(pol.slot_width_frac),
        "slot_gap_te_frac": float(pol.slot_gap_te_frac),
        "cp_ceiling_main": float(pol.cp_ceiling_main),
        "cp_ceiling_flap": float(pol.cp_ceiling_flap),
        "ceiling_source": str(pol.ceiling_source),
        "ceiling_scale": float(pol.ceiling_scale),
        "ceiling_shape": str(pol.ceiling_shape),
        "stall_margin_main": float(g_main), "stall_margin_flap": float(g_flap),
        "lift_share_main": float(pol.share(a_eff)),
        "Re_ref": br.re_ref(V0), "Re_main": float(pol.re_main),
        "Re_flap": float(pol.re_flap), "V_ref": V0,
        "section": pol, "main_coords": main_c, "flap_coords": flap_c,
        "bridge": br,
    }


def _reference_speed(bridge: SectionBridge) -> float:
    """The speed the BASE point is flown at.

    ``carwing_multi``'s own published speed, and it is that number rather than
    the lap's mean because the base point exists to be comparable with the
    published family's single point. The lap does not use it: the lap re-flies
    the section at its own representative speeds.
    """
    return float(carwing_multi.CarWingMultiProblem.V)


def _wing_point(section, alpha_g_deg: float, bridge: SectionBridge):
    """``(CZ, CD, alpha_eff)`` for one section at one rigging angle, or None."""
    CL, a_eff = bridge.wing_CL(section, alpha_g_deg)
    if CL is None:
        return None
    cd_sec = float(section.cd(a_eff))
    CDi = float(CL * CL / (np.pi * float(bridge.AR) * float(bridge.e_drag)))
    CD = cd_sec + CDi + float(bridge.CD0)
    if not np.isfinite(CD) or CD <= 0.0:
        return None
    return float(CL), CD, float(a_eff)


#: what a REFUSED candidate scores, per objective — and it is not
#: ``objective.PENALTY``.
#:
#: `refusal-is-scale-aware` in this repository's memory records what the -100
#: sentinel did to a Gaussian process's length scale, and the same rule bites
#: harder here for a plainer reason: **a feasible design must never rank below
#: a refused one.** A lap score is MINUS A LAP TIME, so it is around -61 on
#: this circuit, and -100 happens to sit below it — by 39 seconds, on a
#: quantity whose whole feasible spread is 0.87 s. MEASURED over 600 uniform
#: draws of the design box, the feasible laps run 60.739 to 61.606 s. A
#: circuit half as fast, or a car with a fifth of the power, would put
#: feasible laps past 100 s and the sentinel would start beating them.
#:
#: So the sentinel is stated per objective and derived from what the objective
#: can produce, rather than inherited from a package constant that means "the
#: solver broke" on a scale of order one:
#:
#:   ``laptime``   -1e4 seconds. Four orders below any lap ``cartrack`` can
#:                 return without refusing first (its own top-speed and corner
#:                 solvers refuse rather than return an unbounded time).
#:   ``weighted``  -1.0. The score is a positive-weighted sum of two ratios
#:                 whose numerators are a section's cl_max and (l/d)_max over
#:                 its FLYABLE window; both are positive for anything that
#:                 flies, so any feasible score is > 0.
REFUSAL_SCORE: dict[str, float] = {"laptime": -1.0e4, "weighted": -1.0}


def fg_section(x, prob: SectionProblem | None = None) -> tuple:
    """Constrained-harness callable: ``(score, margins)``. No margins."""
    prob = prob or SectionProblem()
    out = evaluate_section(x, prob)
    if not out["feasible"]:
        return REFUSAL_SCORE[prob.objective], []
    return float(out["score"]), []


def f_section(x, prob: SectionProblem | None = None) -> float:
    """Unconstrained-harness callable: the score, or the refusal sentinel.

    This problem declares no constraints (:attr:`SectionProblem.n_constraints`
    is 0), so the plain single-objective baselines (``optimize.baselines``)
    are the right harness for it and this is their entry point.
    """
    return fg_section(x, prob)[0]
