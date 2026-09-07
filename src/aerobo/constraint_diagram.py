"""Where the wing loading comes from: the constraint diagram.

Every problem in this package trims to a lift coefficient, and that lift
coefficient is ``CL = W / (q S)`` — so the WING LOADING ``W/S`` decides the
operating point before any aerodynamics happens. Until now the shells asked
for it as a number with no way to answer it, which is the wrong question:
W/S is not a preference, it is what the MISSION allows.

This module is the standard way that question is answered, written for the
three media this package flies in. It is pure functions over plain numbers —
no plotting, no GUI, no ``api`` — so the same numbers can be tested, quoted
in the report, and drawn by whichever shell wants to draw them.

The air diagram (the textbook one)
----------------------------------
Every constraint is written as a relation between the wing loading ``W/S``
[Pa] and the thrust-to-weight ratio ``T/W`` [-], the two numbers that size an
aircraft. Plotted against each other they cut the plane into a feasible
region; the design point is conventionally its lowest T/W at the highest
allowable W/S, because both cost fuel and structure.

Sources, and what is a MODEL CHOICE here:

* stall / approach — ``W/S <= 1/2 rho V_s^2 CL_max``. Definitional.
* landing field length — Raymer's approach-speed correlation, quoted as a
  limit on W/S at the landing weight fraction. The distance-to-approach-speed
  constant is a CALIBRATED value (``K_LAND``), stated, not measured here.
* take-off — the GA take-off parameter ``TOP = (W/S) / (sigma CL_TO (T/W))``
  with a distance correlation ``s_TO = K_TO * TOP`` (Raymer, Roskam). Again a
  correlation with a stated constant.
* cruise matching — at a chosen cruise altitude and speed,
  ``T/W >= q CD0 / (W/S) + (W/S) / (q pi A e)``. Exact for a parabolic polar,
  which is what this package's own drag build-up reduces to; the minimum of
  the right-hand side sits at ``(W/S)* = q sqrt(pi A e CD0)`` — the classic
  minimum-drag wing loading, and the number a pure aerodynamicist would pick.
* climb — ``T/W >= RoC / V + 2 sqrt(CD0 / (pi A e))`` (steady climb at the
  best-climb speed, small-angle).
* sustained turn at load factor n —
  ``T/W >= q CD0 / (W/S) + n^2 (W/S) / (q pi A e)``.

The water diagram (the same idea, the foil's constraints)
---------------------------------------------------------
A foiling craft has no "stall speed" in the runway sense; what bounds its
wing loading at the bottom is FLY-UP — the foil must lift the craft at the
take-off speed, at the highest lift coefficient the section can hold — and
what bounds it at the top is CAVITATION: at the maximum speed the local
suction peak must stay above vapour pressure, and the suction peak grows with
the lift coefficient the loading demands. So

    W/S <= 1/2 rho_w V_takeoff^2 CL_max            (it has to fly at all)
    W/S <= 1/2 rho_w V_max^2 CL_cav(depth, V_max)  (it must not cavitate)

with ``CL_cav`` the lift coefficient at which the section's own ``-Cp_min``
reaches the cavitation number ``sigma`` at that depth and speed
(:func:`cl_cavitation_limited`, which takes the section's ``Cp_min(cl)``
behaviour as the one number it cannot invent).

The tandem case
---------------
A pair carries one weight on two surfaces, so the diagram is stated on the
PAIR's total area — exactly the reference area ``tandem.py`` trims on — and
the front/rear split stays a design variable. What changes is the effective
aspect ratio: for the same total area and span the pair's induced drag is not
one wing's, so :func:`air_diagram` is given the pair's own ``A_eff``.

What this module does NOT do
----------------------------
It does not choose for you. :func:`recommend` returns the binding constraint
and the wing loading it allows, with every curve beside it, so the choice is
made in the open — and leaving W/S as a DESIGN VARIABLE (the free-planform
modifier, sizing.py) stays a first-class answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

G = 9.80665                # m/s^2

#: Raymer's landing-distance correlation constant, s_land = K_LAND * V_a^2
#: with V_a the approach speed [m/s] and s_land [m]. CALIBRATED CONSTANT, in
#: the middle of the GA range quoted in the source; it is here so the number
#: is visible rather than buried in a formula.
K_LAND = 0.5

#: GA take-off correlation, s_TO = K_TO * TOP with TOP in SI
#: ((W/S)[Pa] / (sigma CL_TO (T/W))). CALIBRATED CONSTANT.
K_TO = 0.0806

#: ISA sea-level density [kg/m^3] — the density the field-length constraints
#: are quoted at, whatever the cruise altitude is.
RHO_SL = 1.225


@dataclass(frozen=True)
class Constraint:
    """One line of the diagram.

    ``kind`` says how to read it:

    * ``"ws_max"`` — a vertical line: the wing loading may not exceed
      ``ws_pa``, whatever the thrust is (stall, landing, cavitation, fly-up);
    * ``"twr_curve"`` — a curve ``T/W >= f(W/S)``: feasible ABOVE it
      (cruise, climb, turn, take-off).
    """

    name: str
    kind: str
    note: str = ""
    ws_pa: float | None = None
    twr: np.ndarray | None = None      # aligned with the caller's W/S grid

    def __post_init__(self):
        if self.kind not in ("ws_max", "twr_curve"):
            raise ValueError(f"unknown constraint kind {self.kind!r}")


@dataclass
class AirMission:
    """The mission numbers an air constraint diagram needs.

    Everything is SI and everything is the USER's: this module invents no
    aircraft. ``cd0`` and ``e`` are the parabolic-polar pair the rest of the
    package's drag build-up reduces to; ``aspect_ratio`` is the wing's (or a
    tandem pair's effective one).
    """

    v_stall_ms: float                  # 1-g stall / approach speed
    cl_max: float                      # at that speed, in the landing config
    v_cruise_ms: float
    rho_cruise: float = RHO_SL
    aspect_ratio: float = 10.0
    cd0: float = 0.025
    oswald_e: float = 0.85
    # optional constraints — each is skipped when its input is None
    climb_rate_ms: float | None = None
    v_climb_ms: float | None = None    # defaults to 1.2 * v_stall
    turn_load_factor: float | None = None
    takeoff_distance_m: float | None = None
    cl_takeoff: float | None = None    # defaults to 0.8 * cl_max
    landing_distance_m: float | None = None
    sigma_density_ratio: float = 1.0   # field elevation, rho/rho_sl
    #: the thrust-to-weight the aircraft HAS, against the T/W its constraints
    #: DEMAND. Optional, and None on every published mission, but it is the
    #: half of the matching diagram that closes it: without a supply number
    #: the diagram can only report the upper LIMIT on W/S (stall, landing),
    #: and a limit is not a design point. With it, the design point is the
    #: textbook one — the highest wing loading whose required T/W the engine
    #: can actually deliver (:attr:`Diagram.ws_matched_pa`).
    #:
    #: This matters well beyond the picture. A searched wing loading has no
    #: interior optimum in any objective in this package (measured: 128 runs,
    #: ``results/ws_band_study.json``) — L/D and the composite both rise
    #: monotonically with W/S and park at the top of whatever band they are
    #: given. That is not a search defect: a smaller wing at the same weight
    #: IS more efficient, and the thing that stops it in reality is the thrust
    #: needed to fly it, which no objective in this package prices. So the
    #: honest stopping point is a thrust one, and this is where it enters.
    twr_available: float | None = None

    def q_cruise(self) -> float:
        return 0.5 * float(self.rho_cruise) * float(self.v_cruise_ms) ** 2


@dataclass
class WaterMission:
    """The mission numbers a FOILING craft's diagram needs."""

    v_takeoff_ms: float                # the speed it must fly at
    v_max_ms: float
    cl_max: float                      # the section's usable maximum
    depth_m: float = 0.4               # foil submergence at speed
    rho: float = 1025.0
    p_atm: float = 101325.0
    p_vap: float = 2340.0
    #: the section's suction peak as a function of lift coefficient,
    #: ``-Cp_min ~ cp_a + cp_b * cl`` — a LINEARISED description of a real
    #: polar's Cp_min branch, whose two numbers come from the section that is
    #: actually chosen (aerobo.polar.TablePolar.cp_min). Defaults are a
    #: 12%-thick section's rough shape and are meant to be replaced.
    cp_min_a: float = 0.6
    cp_min_b: float = 2.2


@dataclass
class Diagram:
    """The computed diagram: the grid, the constraints, and the answer."""

    ws_grid_pa: np.ndarray
    constraints: list = field(default_factory=list)
    ws_max_pa: float | None = None     # the binding upper limit
    binding: str = ""                  # which constraint set it
    twr_required: np.ndarray | None = None   # the upper envelope of curves
    ws_min_drag_pa: float | None = None      # cruise minimum-drag loading
    #: the MATCHING POINT: the highest wing loading whose required T/W the
    #: mission's available thrust can deliver, already capped by
    #: :attr:`ws_max_pa`. None when the mission states no available thrust
    #: (``AirMission.twr_available``) — which is not the same as no limit, and
    #: is reported as its own note rather than silently read as one.
    ws_matched_pa: float | None = None
    #: what the matching point is limited BY: ``"thrust"`` when the engine
    #: runs out first, or the name of the W/S cap when that binds first.
    matched_binding: str = ""
    notes: list = field(default_factory=list)

    def twr_at(self, ws_pa: float) -> float | None:
        """Thrust-to-weight the curves demand at a wing loading."""
        if self.twr_required is None:
            return None
        return float(np.interp(float(ws_pa), self.ws_grid_pa,
                               self.twr_required))

    @property
    def ws_design_pa(self) -> float | None:
        """The wing loading this diagram actually recommends.

        The matching point when the mission stated its thrust, and the upper
        LIMIT otherwise. The distinction is the whole point: a limit says
        where the wing stops being legal, a matching point says where it stops
        being flyable, and only the second is a design point.

        None when the mission is INFEASIBLE — no wing loading is both allowed
        and flyable. Falling back to the limit there would hand back a design
        point for an aircraft that cannot be built, which is the one answer
        worse than no answer.
        """
        if self.matched_binding == "infeasible":
            return None
        return self.ws_matched_pa if self.ws_matched_pa is not None \
            else self.ws_max_pa

    def recommend(self) -> dict:
        """The design point this diagram implies, stated with its reason."""
        ws = self.ws_design_pa
        return {
            "wing_loading_pa": ws,
            "binding_constraint": (self.matched_binding
                                   if self.ws_matched_pa is not None
                                   else self.binding),
            "twr_required": None if ws is None else self.twr_at(ws),
            "wing_loading_min_drag_pa": self.ws_min_drag_pa,
            # kept separate and unconditional, so a caller can always tell
            # which of the two questions was answered
            "wing_loading_max_pa": self.ws_max_pa,
            "wing_loading_matched_pa": self.ws_matched_pa,
            "notes": list(self.notes),
        }


# --------------------------------------------------------------------- air
def air_diagram(m: AirMission, n_grid: int = 200) -> Diagram:
    """The classic (W/S, T/W) matching diagram for an air mission."""
    if not (m.v_stall_ms > 0.0 and m.cl_max > 0.0):
        raise ValueError("stall speed and CL_max must both be > 0")
    if not (m.v_cruise_ms > m.v_stall_ms):
        raise ValueError("cruise speed must exceed the stall speed")

    ws_stall = 0.5 * RHO_SL * m.v_stall_ms ** 2 * m.cl_max
    k = 1.0 / (np.pi * float(m.aspect_ratio) * float(m.oswald_e))
    q = m.q_cruise()
    ws_min_drag = float(q * np.sqrt(m.cd0 / k))
    # the grid has to SHOW the trade, so it spans both the allowed band and
    # the aerodynamic optimum — which for a light aircraft usually sits well
    # outside it (that is the point the diagram is making)
    grid = np.linspace(0.05 * ws_stall,
                       1.3 * max(1.25 * ws_stall, ws_min_drag), int(n_grid))

    cons: list[Constraint] = [Constraint(
        name="stall / approach",
        kind="ws_max", ws_pa=float(ws_stall),
        note=(f"W/S <= ½ρV²CL_max at V_s = {m.v_stall_ms:g} m/s and "
              f"CL_max = {m.cl_max:g}. Definitional, and usually the binding "
              f"one for a light aircraft."))]

    if m.landing_distance_m is not None:
        # s_land = K_LAND * V_a^2, V_a the approach speed
        v_app = float(np.sqrt(float(m.landing_distance_m) / K_LAND))
        ws_land = (0.5 * RHO_SL * m.sigma_density_ratio * v_app ** 2
                   * m.cl_max)
        cons.append(Constraint(
            name="landing distance", kind="ws_max", ws_pa=float(ws_land),
            note=(f"s_land = {m.landing_distance_m:g} m implies an approach "
                  f"speed of {v_app:.1f} m/s (K_LAND = {K_LAND}), and that "
                  f"speed with CL_max = {m.cl_max:g} caps W/S.")))

    # cruise matching: T/W >= q CD0/(W/S) + k (W/S)/q
    twr_cruise = q * m.cd0 / grid + k * grid / q
    cons.append(Constraint(
        name="cruise", kind="twr_curve", twr=twr_cruise,
        note=(f"T/W >= q·CD0/(W/S) + (W/S)/(q·πAe) at V = "
              f"{m.v_cruise_ms:g} m/s. Its minimum is the minimum-drag wing "
              f"loading — the number a pure aerodynamicist would choose.")))

    if m.climb_rate_ms is not None:
        v_climb = float(m.v_climb_ms or 1.2 * m.v_stall_ms)
        twr_climb = np.full_like(
            grid, m.climb_rate_ms / v_climb + 2.0 * np.sqrt(m.cd0 * k))
        cons.append(Constraint(
            name="climb", kind="twr_curve", twr=twr_climb,
            note=(f"T/W >= RoC/V + 2√(CD0/πAe) for {m.climb_rate_ms:g} m/s "
                  f"at {v_climb:.1f} m/s. Independent of W/S at the "
                  f"best-climb speed, so it is a floor on thrust.")))

    if m.turn_load_factor is not None:
        n = float(m.turn_load_factor)
        twr_turn = q * m.cd0 / grid + n * n * k * grid / q
        cons.append(Constraint(
            name=f"sustained turn (n = {n:g})", kind="twr_curve",
            twr=twr_turn,
            note=("the induced term carries n², so a manoeuvre requirement "
                  "pushes the design to LOWER wing loading.")))

    if m.takeoff_distance_m is not None:
        cl_to = float(m.cl_takeoff or 0.8 * m.cl_max)
        top_max = float(m.takeoff_distance_m) / K_TO
        # TOP = (W/S) / (sigma CL_TO (T/W))  =>  T/W >= (W/S)/(sigma CL_TO TOP)
        twr_to = grid / (m.sigma_density_ratio * cl_to * top_max)
        cons.append(Constraint(
            name="take-off distance", kind="twr_curve", twr=twr_to,
            note=(f"s_TO = {m.takeoff_distance_m:g} m through the GA "
                  f"take-off parameter (K_TO = {K_TO}, CL_TO = {cl_to:g}).")))

    caps = [c for c in cons if c.kind == "ws_max"]
    binding = min(caps, key=lambda c: c.ws_pa)
    curves = [c.twr for c in cons if c.kind == "twr_curve"]
    envelope = np.max(np.vstack(curves), axis=0) if curves else None

    notes = [
        "Every curve is a REQUIREMENT: feasible designs sit above the T/W "
        "curves and to the left of the W/S limits.",
        f"The minimum-drag wing loading at the cruise point is "
        f"{ws_min_drag:.0f} Pa ({ws_min_drag / G:.1f} kg/m²); the mission "
        f"allows at most {binding.ws_pa:.0f} Pa "
        f"({binding.ws_pa / G:.1f} kg/m²).",
    ]
    if ws_min_drag > binding.ws_pa:
        notes.append(
            "The aerodynamic optimum is OUTSIDE the allowed band, so the "
            "field-length/stall limit is what sets the wing — this is the "
            "usual case, and it is why W/S is a mission answer rather than "
            "an aerodynamic one.")

    ws_matched, matched_binding = _matching_point(
        grid, envelope, float(binding.ws_pa), binding.name, m.twr_available,
        notes)
    return Diagram(ws_grid_pa=grid, constraints=cons,
                   ws_max_pa=float(binding.ws_pa), binding=binding.name,
                   twr_required=envelope, ws_min_drag_pa=ws_min_drag,
                   ws_matched_pa=ws_matched, matched_binding=matched_binding,
                   notes=notes)


def _matching_point(grid, envelope, ws_cap: float, cap_name: str,
                    twr_available, notes: list):
    """Close the diagram: the highest W/S the available thrust can fly.

    Returns ``(ws_matched_pa, what_limits_it)``, or ``(None, "")`` when the
    mission stated no thrust — in which case the caller keeps reporting the
    upper LIMIT and says so, rather than presenting a limit as a design point.

    The required-T/W envelope is not monotone (the cruise term falls with W/S
    while the induced and take-off terms rise), so the powered set is an
    interval and the design point is its right-hand end. Taken off the grid
    and not solved analytically because the envelope is a max of curves whose
    membership changes along it, which has no closed form.
    """
    if twr_available is None:
        notes.append(
            "No available thrust was stated (AirMission.twr_available), so "
            "this diagram reports the upper LIMIT on W/S and not a matching "
            "point. A limit says where the wing stops being legal; only the "
            "thrust closes where it stops being flyable — and since no "
            "objective in this package prices thrust, an unclosed diagram is "
            "why a SEARCHED wing loading has nothing to stop it below the "
            "top of its band.")
        return None, ""
    twr = float(twr_available)
    if envelope is None:              # no T/W curve at all: only caps exist
        return float(ws_cap), cap_name
    powered = np.asarray(envelope) <= twr
    inside = powered & (np.asarray(grid) <= ws_cap)
    if not inside.any():
        notes.append(
            f"NO wing loading is both allowed and flyable at T/W = {twr:g}: "
            f"the cheapest point on the envelope needs T/W "
            f"{float(np.min(envelope)):.3f}. The mission is infeasible as "
            f"stated — raise the thrust, or relax the constraint that sets "
            f"the envelope.")
        return None, "infeasible"
    ws_matched = float(np.max(np.asarray(grid)[inside]))
    # which of the two ran out first, told apart on the grid step so a
    # matching point sitting exactly on the cap is not blamed on thrust
    step = float(grid[1] - grid[0]) if len(grid) > 1 else 0.0
    limited_by = cap_name if ws_matched >= ws_cap - step else "thrust"
    if limited_by == cap_name:
        # the cap is an exact number and the grid is not; reporting the last
        # grid point below it would quantise the answer and read as a
        # thrust-limited one a step lower than the limit it actually is
        ws_matched = float(ws_cap)
    notes.append(
        f"Matching point: W/S = {ws_matched:.0f} Pa ({ws_matched / G:.1f} "
        f"kg/m²), limited by {limited_by}, at T/W available {twr:g}. This — "
        f"not the {ws_cap:.0f} Pa limit — is the design point, and it is the "
        f"number a searched W/S band should stop at.")
    return ws_matched, limited_by


# ------------------------------------------------------------------- water
def cavitation_number(m: WaterMission, v_ms: float) -> float:
    """``sigma = (p_atm + rho g h - p_v) / (½ rho V²)``.

    hydrofoil.sigma_cav, restated here so this module imports nothing. It is
    its own function because TWO things need it and they must not drift: the
    cavitation LINE (:func:`cl_cavitation_limited`) and the linearisation of
    a real section's suction peak taken AT that line (:func:`cp_min_line`).
    """
    return ((m.p_atm + m.rho * G * float(m.depth_m) - m.p_vap)
            / (0.5 * m.rho * float(v_ms) ** 2))


def cl_cavitation_limited(m: WaterMission, v_ms: float) -> float:
    """The lift coefficient at which the section starts to cavitate.

    Cavitation begins where the suction peak reaches the cavitation number
    (:func:`cavitation_number`):

        -Cp_min(cl) = sigma   with   -Cp_min ~ cp_min_a + cp_min_b · cl

    so ``cl_cav = (sigma - cp_min_a) / cp_min_b``. Returns 0.0 when even zero
    lift cavitates at that speed and depth — which is the honest answer: no
    wing loading is admissible there.
    """
    sigma = cavitation_number(m, v_ms)
    return float(max(0.0, (sigma - m.cp_min_a) / m.cp_min_b))


#: the alpha sweep :func:`cp_min_line` reads a section's suction peak over.
#: Wide enough to contain the whole cavitation bucket and both its walls at
#: every thickness in the water bank, fine enough that the crossing it
#: brackets is within ~0.003 in cl of the tabulated one.
CP_MIN_ALPHA_DEG = np.arange(-4.0, 12.0 + 1e-9, 0.25)


def cp_min_line(polar, m: WaterMission, v_ms: float | None = None,
                alpha_deg=None) -> tuple | None:
    """``(cp_min_a, cp_min_b)`` read off a REAL section, or ``None``.

    :class:`WaterMission` describes a section's suction peak with a straight
    line, ``-Cp_min ~ a + b·cl``, and its defaults are a 12%-thick section's
    ROUGH shape — the field comment has said "meant to be replaced" since the
    class was written, and nothing replaced them. A real ``-Cp_min(cl)`` is
    not a line at all: it is the cavitation BUCKET, falling to a minimum near
    the section's ideal lift and rising steeply on both walls. Fitting one
    line to the whole of it is the worst of both, so this fits the line where
    the diagram actually reads it — the TANGENT at the crossing
    ``-Cp_min(cl) = sigma``. :func:`cl_cavitation_limited` then returns that
    crossing exactly, and the line's slope is the section's own there.

    Why it matters: the placeholder pair put the published hydrofoil design
    point (6 kN on 0.144 m², 12 m/s, 0.575 m down) 34 % INSIDE the cavitation
    limit's refusal, while the engine that scores that same family measures
    its Cp_min from the polar and finds it uncavitated. The diagram was
    refusing the design point its own solver flies.

    ``polar`` is anything with ``cl(alpha_deg)`` and ``cp_min(alpha_deg)``
    (aerobo.polar.TablePolar and the family members). ``None`` when it cannot
    answer — no Cp_min table, or a sweep that does not bracket the crossing —
    so the caller keeps whatever pair it already had rather than being handed
    a fitted number with no data under it.
    """
    v = float(m.v_max_ms if v_ms is None else v_ms)
    if not v > 0.0:
        return None
    sigma = cavitation_number(m, v)
    al = np.asarray(CP_MIN_ALPHA_DEG if alpha_deg is None else alpha_deg,
                    dtype=float)
    try:
        cl = np.array([float(polar.cl(a)) for a in al], dtype=float)
        mcp = np.array([float(-polar.cp_min(a)) for a in al], dtype=float)
    except (AttributeError, TypeError, ValueError, IndexError):
        return None
    if cl.size < 3 or not np.all(np.isfinite(cl)) or not np.all(
            np.isfinite(mcp)):
        return None
    floor = int(np.argmin(mcp))
    if mcp[floor] >= sigma:
        # the bucket's own FLOOR is already at vapour pressure: this section
        # cavitates at every lift it can make at this speed and depth. The
        # line that says so is the one whose crossing is at zero lift — and
        # its slope has to come off the RISING wall, because the slope AT the
        # floor is zero by definition and a zero slope has no crossing at all.
        b = _rising_slope(cl, mcp, floor)
        return (float(sigma), float(b)) if b > 0.0 else None
    above = np.nonzero(mcp[floor + 1:] > sigma)[0]
    if above.size == 0:
        # the section never reaches vapour pressure anywhere in the sweep:
        # cavitation does not bind, and the honest line is the one whose
        # crossing sits at the top of what this section can do
        j = int(np.argmax(cl))
        b = _local_slope(cl, mcp, max(j, 1))
        cl_star = float(cl[j])
    else:
        j = floor + 1 + int(above[0])
        b = _local_slope(cl, mcp, j)
        cl_star = float(np.interp(sigma, [mcp[j - 1], mcp[j]],
                                  [cl[j - 1], cl[j]]))
    if not b > 0.0:
        return None
    return (float(sigma - b * cl_star), float(b))


def _local_slope(cl, mcp, j: int) -> float:
    """d(-Cp_min)/dcl across the interval ending at index ``j``."""
    dcl = float(cl[j] - cl[j - 1])
    return float((mcp[j] - mcp[j - 1]) / dcl) if abs(dcl) > 1e-12 else 0.0


def _rising_slope(cl, mcp, floor: int) -> float:
    """The first positive d(-Cp_min)/dcl above the bucket's floor."""
    for j in range(max(floor, 0) + 1, int(len(cl))):
        b = _local_slope(cl, mcp, j)
        if b > 0.0:
            return b
    return 0.0


def water_diagram(m: WaterMission, n_grid: int = 200) -> Diagram:
    """The foiling craft's version: fly-up at the bottom, cavitation at the
    top. Both are limits on W/S alone — a foil's thrust matching is a
    propulsion question this package does not model — so the diagram is a
    band rather than a region."""
    if not (m.v_takeoff_ms > 0.0 and m.v_max_ms > m.v_takeoff_ms):
        raise ValueError("need 0 < v_takeoff < v_max")

    ws_flyup = 0.5 * m.rho * m.v_takeoff_ms ** 2 * m.cl_max
    cl_cav = cl_cavitation_limited(m, m.v_max_ms)
    ws_cav = 0.5 * m.rho * m.v_max_ms ** 2 * cl_cav

    grid = np.linspace(0.05 * ws_flyup, 1.6 * max(ws_flyup, ws_cav),
                       int(n_grid))
    cons = [
        Constraint(name="fly-up", kind="ws_max", ws_pa=float(ws_flyup),
                   note=(f"the foil has to lift the craft at "
                         f"{m.v_takeoff_ms:g} m/s with CL_max = "
                         f"{m.cl_max:g}: W/S <= ½ρV²CL_max.")),
        Constraint(name="cavitation", kind="ws_max", ws_pa=float(ws_cav),
                   note=(f"at {m.v_max_ms:g} m/s and {m.depth_m:g} m down "
                         f"the section can hold CL = {cl_cav:.3f} before its "
                         f"suction peak reaches vapour pressure.")),
    ]
    binding = min(cons, key=lambda c: c.ws_pa)
    notes = [
        "A foiling craft's wing loading is bounded at BOTH ends by the same "
        "kind of statement — it must fly slowly enough to take off and stay "
        "un-cavitated fast enough to cruise — and the tighter of the two is "
        "the design driver.",
        f"fly-up allows {ws_flyup:.0f} Pa, cavitation allows {ws_cav:.0f} Pa "
        f"({binding.name} binds).",
    ]
    if ws_cav <= 0.0:
        notes.append(
            "Cavitation allows NO positive wing loading at that speed and "
            "depth: go deeper, slower, or to a section with a flatter "
            "suction peak.")
    return Diagram(ws_grid_pa=grid, constraints=cons,
                   ws_max_pa=float(binding.ws_pa), binding=binding.name,
                   notes=notes)


# ------------------------------------------------------------------ tandem
def tandem_effective_aspect_ratio(b: float, s_total: float,
                                  span_efficiency: float = 1.0) -> float:
    """A tandem pair's effective aspect ratio on its TOTAL area.

    Both wings share the span, so the pair's geometric aspect ratio on the
    reference area is ``b²/S_total`` — NOT either wing's own. The mutual
    induced drag is what ``span_efficiency`` carries: 1.0 says "treat the
    pair as one wing of that aspect ratio", which is the optimistic reading;
    the honest number for a given gap and stagger comes out of the tandem
    solver itself (tandem.py), and this is the placeholder the diagram uses
    until it has one.
    """
    if not (b > 0.0 and s_total > 0.0):
        raise ValueError("span and total area must both be > 0")
    return float(span_efficiency * b * b / s_total)


def wing_loading_to_area(wing_loading_pa: float, weight_n: float) -> float:
    """``S = W / (W/S)`` — the whole point of choosing a wing loading."""
    if not (wing_loading_pa > 0.0 and weight_n > 0.0):
        raise ValueError("wing loading and weight must both be > 0")
    return float(weight_n / wing_loading_pa)


def cl_at(wing_loading_pa: float, rho: float, v_ms: float) -> float:
    """The trim lift coefficient a wing loading implies: ``CL = (W/S)/q``.

    This is the identity that makes W/S the mission's real answer: fix it and
    the operating lift coefficient is fixed with it, whatever the area turns
    out to be.
    """
    q = 0.5 * float(rho) * float(v_ms) ** 2
    if q <= 0.0:
        raise ValueError("dynamic pressure must be > 0")
    return float(wing_loading_pa / q)
