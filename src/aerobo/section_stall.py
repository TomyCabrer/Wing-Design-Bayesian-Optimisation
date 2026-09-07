"""The stall of THIS section, at THIS Reynolds number — measured, not
substituted.

Why this module exists
----------------------
``carwing_multi.suction_peak_ceiling`` builds a two-element section's stall
criterion in two steps: take the RESOLVED stall incidence of a NACA 24XX of
the element's thickness off the shipped wide-alpha XFOIL bank, then solve the
element inviscidly at that incidence and call its leading-edge suction peak the
ceiling. The second step is exact — it is the same panel method, on the same
grid, that the criterion will judge. The first step is a SUBSTITUTION twice
over:

  * it is a NACA 24XX's stall standing in for the stall of whatever shape is
    actually being flown, which matters the moment either element is a designed
    section rather than a 24XX; and
  * it is a Re-1e6 measurement used at whatever Reynolds number the element
    flies. ``carwing_multi``'s own docstring states the direction and declines
    to quantify it: a section stalls EARLIER at lower Re, so at the ~2e5 a
    small car wing's flap reaches, a 1e6 ceiling is OPTIMISTIC.

This module closes both by measuring the stall on the coordinates being flown,
at the Reynolds number they are flown at, with the tool this repository already
treats as the reference for anything 2-D and single-element: XFOIL. What it
does NOT close is the criterion itself — see "What is still a model" below.

The method, and each step's authority
-------------------------------------
1. **Viscous sweep.** ``xfoil_run.run_xfoil_polar`` over
   :data:`STALL_SWEEP_DEG`, which is the shipped wide-alpha bank's own sweep
   (-6 to +22 deg, step 0.5, the range and step
   ``scripts/gen_stall_polar_family.py`` uses), so a measured ceiling and a
   substituted one differ in the SHAPE and the REYNOLDS NUMBER and in nothing
   else procedural. Ncrit, Mach, the panel count and the split-at-zero
   marching order are that driver's, unchanged.
2. **The peak, with the same censoring rule.** The stall is the cl peak, and
   it counts as a stall only if the table CONTINUES past it and cl falls at
   least :data:`STALL_DROP_TOL` below the maximum somewhere above — exactly
   ``polar.stall_point``'s test, applied here to a fresh polar instead of a
   file. A right-censored table is a march that ran out, which is a LOWER
   BOUND on the stall angle and not a stall (`censored-clmax-is-a-lower-bound`
   in this repository's memory), and it is refused rather than used.
3. **The ceiling.** ``panel2d`` solves the section, isolated, at the measured
   stall incidence, on the SAME panel count the cascade will be flown on, and
   its Cp_min is the ceiling. This is ``carwing_multi``'s own rule and it is
   kept: the criterion and the quantity it judges are always measured on one
   grid, so their shared discretisation error partly cancels in the margin
   (that module measures the residual at 0.48 % optimistic at 120 nodes).

What is still a model, after this
---------------------------------
The criterion is unchanged and it is still ours: *an element in a slot is
unstalled while its inviscid suction peak is no deeper than the peak the same
tool gives that element ALONE at its own stall.* A slot exists precisely so
that an element can hold a peak it could not hold alone, so the criterion is
conservative in a direction nobody here has quantified. What this module
removes is the two things that were not even part of the criterion — the wrong
shape and the wrong Reynolds number — so that a result which moves when the
criterion is swept is a result about the CRITERION and not about a NACA 24XX
that nobody flew.

It also does not make the criterion cheap. One measured ceiling is an XFOIL
process, so this belongs on the handful of designs a study VERIFIES, not in an
optimiser's inner loop. The sweeps are cached by ``xfoil_run``'s own content
key, so re-verifying the same section is free.

What is NOT modelled
--------------------
* Everything XFOIL does not model: three-dimensionality, unsteadiness,
  compressibility above its Karman-Tsien correction's range, and the
  transition physics beyond an e^N envelope at the driver's own Ncrit.
* THE SLOT. Both elements are measured ALONE. That is the point — the ceiling
  is an isolated-section statement by construction — but it means nothing here
  sees the interaction, and a real merged-wake stall is outside this and
  outside ``panel2d``.
* Hysteresis, and any distinction between leading-edge and trailing-edge
  stall. The cl peak is the stall, full stop, which is the same reduction the
  shipped bank makes.
"""

from __future__ import annotations

import numpy as np

from . import panel2d

__all__ = [
    "STALL_DROP_TOL", "STALL_SWEEP_DEG", "measured_ceiling",
    "measured_ceilings", "measured_stall", "stall_sweep_alphas",
]

#: the sweep a stall is measured over: the shipped wide-alpha bank's own
#: ``(lo, hi, step)`` (``scripts/gen_stall_polar_family.py``: -6 to +22 deg,
#: step 0.5). Stated as this module's constant rather than imported from a
#: script so that a change to either is a visible disagreement, and quoted
#: here so a reader can see that a measured ceiling and a substituted one are
#: procedurally the same measurement of different shapes.
STALL_SWEEP_DEG = (-6.0, 22.0, 0.5)

#: how far cl must fall past its peak before the peak is a STALL rather than
#: the end of the data. ``polar.stall_point``'s own default, and it is that
#: number on purpose: a measured ceiling and a substituted one must apply the
#: same censoring test or the two are not comparable.
STALL_DROP_TOL = 0.01

#: XFOIL panel count for the viscous sweep. ``xfoil_run.run_xfoil_polar``'s
#: own default; it is the panelling XFOIL solves on and is unrelated to
#: ``carwing_multi.N_SECTION_NODES``, which is the panelling the INVISCID
#: ceiling is then read on. Two counts, two solvers, two jobs — and the
#: second one has to match the cascade's, which is why it is a parameter here
#: and not a constant.
XFOIL_N_PANEL = 200


def stall_sweep_alphas(sweep: tuple = STALL_SWEEP_DEG) -> np.ndarray:
    """The alpha grid of :data:`STALL_SWEEP_DEG`, endpoint included."""
    lo, hi, step = (float(v) for v in sweep)
    n = int(round((hi - lo) / step))
    return lo + step * np.arange(n + 1, dtype=float)


def measured_stall(coords, re: float, mach: float = 0.0,
                   sweep: tuple = STALL_SWEEP_DEG,
                   n_panel: int = XFOIL_N_PANEL,
                   drop_tol: float = STALL_DROP_TOL,
                   **xfoil_kw) -> dict:
    """XFOIL's stall for one closed loop at one Reynolds number.

    Returns a dict, always. ``feasible`` False with a ``reason`` for anything
    a section can do to a viscous solver — nothing converged, the march ran
    out before the peak, the loop is not a loop — because a section that
    cannot be measured is an infeasible DESIGN and not a programming error
    (``objective._fail``'s contract). ``xfoil_run.XfoilError`` still
    propagates: a missing binary is infrastructure, not a design.

    On success: ``alpha_stall_deg``, ``cl_max``, ``alpha_max_deg`` (the last
    converged row, so a reader can see how much table lies past the peak),
    ``n_converged`` and ``re``.
    """
    from .xfoil_run import run_xfoil_polar

    c = np.asarray(coords, dtype=float)
    if c.ndim != 2 or c.shape[1] != 2 or c.shape[0] < 10:
        return {"feasible": False,
                "reason": (f"a section is a closed (n, 2) loop with at least "
                           f"10 nodes; got shape {c.shape}")}
    if not np.all(np.isfinite(c)):
        return {"feasible": False,
                "reason": "the section's coordinates are not all finite"}
    if not float(re) > 0.0:
        return {"feasible": False,
                "reason": f"Reynolds number must be > 0, got {re!r}"}

    alphas = stall_sweep_alphas(sweep)
    res = run_xfoil_polar(c, float(re), float(mach), alphas,
                          n_panel=int(n_panel), **xfoil_kw)
    if res.n_converged == 0:
        return {"feasible": False,
                "reason": (f"XFOIL converged at no incidence of the "
                           f"{sweep[0]:g}..{sweep[1]:g} deg sweep at "
                           f"Re = {float(re):.3g}: this section has no "
                           f"measurable polar, let alone a measurable stall")}

    a = np.asarray(res.alpha_deg, dtype=float)
    cl = np.asarray(res.cl, dtype=float)
    i = int(np.argmax(cl))
    cl_max = float(cl[i])
    past = cl[i + 1:]
    # polar.stall_point's rule, restated on a fresh polar: the table must
    # CONTINUE past the peak and fall away from it
    if not (past.size and float(past.min()) <= cl_max - float(drop_tol)):
        return {"feasible": False,
                "reason": (f"the viscous march ran out at alpha = "
                           f"{float(a[-1]):.2f} deg with cl still at its "
                           f"maximum {cl_max:.4f} (Re = {float(re):.3g}): "
                           f"that is a right-CENSORED table, i.e. a lower "
                           f"bound on the stall angle, and a ceiling built "
                           f"on one would be a number with nothing behind "
                           f"it"),
                "censored": True, "cl_max": cl_max,
                "alpha_max_deg": float(a[-1]), "re": float(re)}
    return {"feasible": True, "reason": "", "censored": False,
            "alpha_stall_deg": float(a[i]), "cl_max": cl_max,
            "alpha_max_deg": float(a[-1]),
            "n_converged": int(res.n_converged), "re": float(re),
            "from_cache": bool(res.from_cache)}


def measured_ceiling(coords, re: float,
                     mach: float = 0.0, sweep: tuple = STALL_SWEEP_DEG,
                     n_panel: int = XFOIL_N_PANEL,
                     drop_tol: float = STALL_DROP_TOL,
                     **xfoil_kw) -> dict:
    """``(Cp_min ceiling, alpha_stall)`` for one section, both measured on it.

    ``coords`` MUST BE THE LOOP THE CASCADE WILL FLY, node for node — the
    array ``carwing_multi.element_base_coords`` returns, not a denser or
    coarser sampling of the same shape. The reason is not tidiness: the
    ceiling is an inviscid Cp_min and Cp_min is a PANEL-COUNT quantity, so a
    ceiling read on one grid and a peak read on another do not cancel their
    shared discretisation error and their ratio — which is the stall margin —
    is wrong by their difference. MEASURED on a NACA 2412 at Re 1e6: the same
    section's ceiling is -13.9441 on the 200-node loop and -13.8808 on the
    120-node one, a 0.46 % gap that would land straight in the margin.
    (``carwing_multi.N_SECTION_NODES`` prices the same effect the other way
    round: at 120 nodes the peak is 1.25 % shallow, the ceiling 0.81 % shallow
    and the margin they make 0.48 % optimistic — that partial cancellation is
    what a mismatched grid throws away.)

    XFOIL's own panelling is a separate question and is not this one: it
    re-panels whatever it is given to ``n_panel`` before solving, so the
    viscous half is insensitive to the node count of the file.

    Returns the :func:`measured_stall` dict with ``cp_ceiling`` and
    ``ceiling`` (the ``(cp, alpha)`` pair ``carwing_multi.cascade_polar``
    takes) added; an infeasible stall is passed straight through.
    """
    got = measured_stall(coords, re, mach=mach, sweep=sweep, n_panel=n_panel,
                         drop_tol=drop_tol, **xfoil_kw)
    if not got["feasible"]:
        return got
    c = np.asarray(coords, dtype=float)
    x, y = panel2d.close_trailing_edge(c[:, 0], c[:, 1])
    reason = panel2d.check_body(x, y)
    if reason is not None:
        return {"feasible": False,
                "reason": (f"XFOIL measured this section's stall at "
                           f"{got['alpha_stall_deg']:.2f} deg, but the panel "
                           f"method cannot read a suction peak off it: "
                           f"{reason}")}
    body = panel2d.Body(x, y, name="isolated")
    sol = panel2d.solve(body, np.deg2rad(got["alpha_stall_deg"]), V=1.0,
                        c_ref=1.0, ref_point=(0.25, 0.0))
    cp = float(sol.cp_min)
    return {**got, "cp_ceiling": cp, "n_panels": int(body.n_panels),
            "ceiling": (cp, float(got["alpha_stall_deg"]))}


def measured_ceilings(main_coords, flap_coords, re_main: float,
                      re_flap: float, **kw):
    """``(((cp, a), (cp, a)), None)`` for a two-element section, or
    ``(None, reason)``.

    The pair ``carwing_multi.cascade_polar(..., ceilings=...)`` takes. Each
    element is measured on ITS OWN coordinates at ITS OWN Reynolds number,
    which is the whole difference from the substitution: the shipped path
    reads one NACA 24XX bank at Re 1e6 for both.

    Both elements' coordinates are on a UNIT chord and are the loops the
    cascade flies (:func:`measured_ceiling`). That the flap's are unit-chord
    is not a detail: a section's polar is a function of its shape and its
    Reynolds number, and the flap's chord has already been spent in computing
    ``re_flap`` — passing its scaled-down coordinates as well would apply the
    chord twice.
    """
    out = []
    for name, c, re in (("main element", main_coords, float(re_main)),
                        ("flap", flap_coords, float(re_flap))):
        got = measured_ceiling(c, re, **kw)
        if not got["feasible"]:
            return None, f"measured stall ({name}): {got['reason']}"
        out.append(got["ceiling"])
    return tuple(out), None
