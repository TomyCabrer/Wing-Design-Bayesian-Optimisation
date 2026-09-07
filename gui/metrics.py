"""What a result MEANS, as presentation data — pure, no UI, no physics.

The Results page used to show one number (the objective) and bury every
aerodynamic quantity in a collapsed "all breakdown scalars" grid, so a
finished run said almost nothing on sight.  This module turns a run's
breakdown into named, formatted, unit-carrying metrics:

* :func:`headline` — the handful of numbers that ARE the answer, in
  reading order, with the problem's own speciality metrics appended
  (static margin for a tail, lift share for a tandem, cavitation margin
  for a hydrofoil, …);
* :func:`groups`   — every metric worth naming, grouped for the Aero tab;
* :func:`family`   — which speciality a breakdown belongs to.

Families are detected from the BREAKDOWN KEYS, not the problem name: a
breakdown that carries ``SM`` is a tail layout whatever the run was
called, and new problems inherit the right cards for free.

Values come only from what the run recorded; anything absent is simply
not shown (never invented, never zero-filled).
"""

from __future__ import annotations

import math
from typing import NamedTuple


class Spec(NamedTuple):
    """One presentable quantity: where to find it and how to say it."""

    keys: tuple           # breakdown keys to try, in order
    label: str
    unit: str = ""
    digits: int = 3
    tip: str = ""
    scale: float = 1.0    # multiply before formatting (e.g. fractions -> %)


# --------------------------------------------------------------- catalogue
PERFORMANCE = (
    Spec(("LoD",), "L/D", "", 2,
         "Lift-to-drag ratio of the best design — the objective for every "
         "wing-like problem."),
    Spec(("CL", "CL_total"), "C_L", "", 4,
         "Trimmed lift coefficient actually flown."),
    Spec(("CD",), "C_D", "", 5, "Total drag coefficient."),
    Spec(("CDi", "CDi_total"), "C_Di", "", 5,
         "Induced (vortex) drag — the part span and loading control."),
    Spec(("CDp",), "C_Dp", "", 5,
         "Profile + parasite drag from the section polar and any extras."),
    Spec(("e", "e_total"), "span efficiency e", "", 3,
         "Oswald/span efficiency: 1.0 is the elliptic ideal."),
    Spec(("alpha_deg",), "α", "deg", 2, "Trim angle of attack."),
    Spec(("AR",), "aspect ratio", "", 2, "b² / S of the reference wing."),
    Spec(("Re_mac",), "Re (MAC)", "", 0,
         "Chord Reynolds number at the mean aerodynamic chord."),
)

SPECIALITY: dict[str, tuple] = {
    "tail": (
        Spec(("SM",), "static margin", "", 3,
             "(x_np − x_cg)/MAC. The constrained quantity: the run is only "
             "feasible while this stays above SM_min."),
        Spec(("SM_min",), "SM required", "", 3,
             "Lower bound the layout must clear."),
        Spec(("CL_w",), "C_L wing", "", 4, "Lift carried by the wing."),
        Spec(("CL_t",), "C_L tail", "", 4,
             "Lift carried by the tail — negative means it is pushing down "
             "to trim."),
        Spec(("S_t",), "tail area", "m²", 3, "Tail planform area."),
        Spec(("b_t",), "tail span", "m", 3, "Tail span."),
        Spec(("l_t",), "tail arm", "m", 3,
             "Signed moment arm: negative is a canard, ahead of the wing."),
        Spec(("i_t_deg",), "tail incidence", "deg", 2,
             "All-moving stabilator setting the trim solve landed on."),
        Spec(("delta_e_deg",), "elevator δe", "deg", 2,
             "Elevator deflection required to trim."),
        Spec(("Cm_cg",), "C_m about CG", "", 5,
             "Residual pitching moment — a trimmed design sits at ~0."),
        Spec(("CDi_mut",), "mutual C_Di", "", 5,
             "Induced drag from wing/tail interference alone."),
    ),
    "tandem": (
        Spec(("lift_share_front",), "front lift share", "%", 1,
             "Fraction of total lift carried by the front wing.", 100.0),
        Spec(("dx",), "stagger aft", "m", 2,
             "How far aft of the front wing's quarter-chord the rear one "
             "sits. STATED, not searched: the mutual induction below is a "
             "function of exactly this and the height under it."),
        Spec(("dz",), "stagger up", "m", 2,
             "How far above the front wing the rear one sits. Vertical gap "
             "is what decouples a pair — streamwise separation alone never "
             "does — and with a tip device on each wing it is also what "
             "keeps the two devices apart."),
        Spec(("CDi_mut",), "mutual C_Di", "", 5,
             "Interference drag between the two surfaces — the whole point "
             "of the tandem trade."),
        Spec(("CDi_self_front",), "front self C_Di", "", 5, ""),
        Spec(("CDi_self_rear",), "rear self C_Di", "", 5, ""),
        Spec(("e_front",), "front e", "", 3, "Front-surface span efficiency."),
        Spec(("e_rear",), "rear e", "", 3, "Rear-surface span efficiency."),
        Spec(("S_front",), "front area", "m²", 3, ""),
        Spec(("S_rear",), "rear area", "m²", 3, ""),
        Spec(("eps_rear_mean_deg",), "downwash on rear", "deg", 2,
             "Mean induced downwash the front wing imposes on the rear."),
    ),
    "hydrofoil": (
        Spec(("sigma_cav",), "cavitation margin σ", "", 3,
             "Cavitation number minus the worst suction peak: feasible "
             "while positive."),
        Spec(("cp_min_worst",), "min C_p", "", 3,
             "Deepest suction peak on the foil — the most negative pressure "
             "coefficient, where cavitation starts."),
        Spec(("Fn_h",), "Froude (depth)", "", 3,
             "Depth-based Froude number — sets the free-surface wave loss."),
        Spec(("CDi_surf",), "free-surface C_Di", "", 5,
             "Extra induced drag from the free surface (wave making)."),
        Spec(("CDi_self",), "self C_Di", "", 5,
             "Induced drag the foil would have deep underwater."),
        Spec(("depth",), "depth", "m", 3, "Submergence of the foil."),
        Spec(("V",), "speed", "m/s", 2, "Design speed."),
        Spec(("cd0_mast",), "mast C_D0", "", 5,
             "Mast parasite drag charged: wetted area at the strut's own "
             "Reynolds number and form factor."),
        Spec(("CD_strut",), "strut C_D", "", 5,
             "The strut's WHOLE charge — its parasite drag, the two "
             "junctions where it meets the fuselage, and the induced drag "
             "of the leeway it flies to carry the rig's side load. Equal to "
             "the mast C_D0 above only where the published wetted-area "
             "charge is being reproduced (strut_model off)."),
        Spec(("CDi_strut_side",), "leeway C_Di", "", 5,
             "What carrying the rig's side force costs. It goes as 1/V^4 at "
             "a fixed side load, so it is a rounding error at top speed and "
             "the strut's largest term at take-off."),
        Spec(("strut_leeway_deg",), "leeway", "deg", 2,
             "The angle the strut flies at to make the side force the rig "
             "demands. Zero without a rig, or with no side force stated."),
    ),
    "aircraft": (
        Spec(("W_total_N",), "total weight", "N", 1,
             "Fixed weight plus the sized wing structure."),
        Spec(("wing_loading_Pa",), "wing loading", "N/m²", 1,
             "W/S the design actually flies — an OUTPUT where the area is "
             "searched, the number the search picked where the loading is, "
             "and the mission's own where it states one. It fixes the trim "
             "lift coefficient at (W/S)/q and the stall speed at "
             "sqrt(2(W/S)/(rho CL_max)), which is why the mission's "
             "constraint diagram is about this number."),
        Spec(("W_wing_N",), "wing weight", "N", 1,
             "Structural weight the planform pays for its span and area."),
        Spec(("W_tail_N",), "empennage weight", "N", 1,
             "Tailplane + fin, by Raymer's GA correlations at THIS layout: "
             "a T-tail's fin carries the tailplane (charged 20 %), and a "
             "V-tail has two panels and no fin at all. 0 where the family "
             "carries no empennage."),
        Spec(("D_N",), "drag", "N", 2, "Dimensional drag at the trim point."),
        Spec(("sigma_root_Pa",), "root stress", "MPa", 1,
             "Root bending stress, against the allowable below.", 1e-6),
        Spec(("sigma_allow_Pa",), "allowable stress", "MPa", 1, "", 1e-6),
    ),
    "winglet": (
        Spec(("CDi",), "C_Di", "", 5,
             "Induced drag — what the winglet is bought to reduce."),
    ),
    "carwing": (
        Spec(("CZ",), "downforce C_Z", "", 4,
             "Downforce coefficient on the wing area, positive downwards — "
             "the quantity the run maximises."),
        Spec(("efficiency",), "C_Z / C_D", "", 2,
             "Downforce per unit drag: the aerodynamic efficiency of the "
             "device, and what the drag budget is really trading against."),
        Spec(("downforce_N",), "downforce", "N", 0,
             "Dimensional load at the design speed."),
        Spec(("M_max_Nm",), "peak bending", "N·m", 1,
             "Largest bending moment in the spar for THIS mount layout."),
        Spec(("deflection_m",), "deflection", "mm", 2,
             "Movement of the wing relative to its mount under load — "
             "ride-height and incidence change at speed.", 1000.0),
        Spec(("ride_height_m",), "ride height", "m", 3,
             "Wing height above the track."),
        Spec(("b_m",), "span", "m", 3,
             "The span this design flew — a design variable here: at the "
             "fixed reference area it IS the aspect ratio, so it trades "
             "induced drag against the bending the mount carries."),
        Spec(("overall_width_m",), "overall width", "m", 3,
             "The car's width: the wing plus whatever its endplates project. "
             "This is what the span band bounds, because a regulation "
             "measures the car, not the wing. A plate at cant 90 projects "
             "nothing; a blended one reaches outboard and the wing shortens "
             "to pay for it."),
        Spec(("endplate_projection_m",), "plate projection", "m", 3,
             "How far one plate reaches outboard of the wing tip — zero "
             "unless its root is blended."),
        Spec(("endplate_h_m",), "endplate height", "m", 3,
             "Arc length of the plate towards the track. A length, not a "
             "fraction of the span — the plate has to reach the car."),
        Spec(("cd0_struts",), "pylon C_D0", "", 5,
             "Parasite drag of the mounting struts (zero when the wing is "
             "endplate-mounted)."),
        Spec(("CD_junction",), "junction C_D", "", 5,
             "Interference drag of the mount's corners (junction.py — a "
             "reduced-order add-on, not solver output)."),
        # --- designed endplates (endplate.py). Absent keys are skipped, so
        # the plain car wing still renders exactly the rows above.
        Spec(("endplate_chord_m",), "endplate chord", "m", 3,
             "Streamwise chord of the plate — its own, not the wing's. It "
             "answers to two bands: a ratio of the wing's tip chord (the "
             "design box's row) and a band in metres (the card's), and this "
             "is the first clipped into the second."),
        Spec(("endplate_chord_requested_m",), "chord asked for", "m", 3,
             "What the design vector's ratio row asked for, before the metre "
             "band was applied. Equal to the chord flown unless the band bit "
             "— and then the difference is the whole story of that design."),
        Spec(("endplate_chord_ratio",), "chord / tip chord", "", 3,
             "The plate's chord as a multiple of the WING's tip chord, as "
             "flown. Above 1 the plate is the bigger surface at the corner, "
             "which is what a blend then has to ramp away."),
        Spec(("endplate_chord_corner_m",), "chord at the corner", "m", 3,
             "The chord where the plate meets the wing. Equal to the plate's "
             "chord at a sharp corner — that IS the step the two surfaces "
             "join at — and ramped back towards the wing's tip chord by a "
             "blend, which is what makes a blend a blend."),
        Spec(("endplate_chord_mean_m",), "mean plate chord", "m", 3,
             "Averaged over the plate's height: what the plate's own drag "
             "build-up is charged on, since a blended root flies less chord "
             "than the plate's nominal one."),
        Spec(("endplate_tc",), "endplate t/c", "", 4,
             "Thickness ratio of the plate's section: up is drag, down is "
             "a plate that bends sideways (stiffness goes as t³)."),
        Spec(("endplate_toe_deg",), "endplate toe", "deg", 2,
             "Plate incidence. Positive loads the plate inboard, i.e. turns "
             "the flow outboard (outwash) — which unloads the tip vortex "
             "and raises C_Z, at the cost of a side load the plate has to "
             "carry."),
        Spec(("CD_endplate",), "endplate C_D", "", 5,
             "Parasite drag of the plates: friction × form factor on both "
             "faces, plus the edge penalty a constant-thickness plate pays "
             "(four times as much square-cut as radiused)."),
        Spec(("endplate_side_load_N",), "plate side load", "N", 1,
             "Lateral load sizing the plate: its own toe side force plus "
             "the yaw load case, worst-case in the same sense."),
        Spec(("endplate_deflection_m",), "plate deflection", "mm", 2,
             "Sideways movement of the wing relative to the plate's built-in "
             "end — cantilevered from the car deck when endplate-mounted, "
             "from the wing when the pylons carry the load.", 1000.0),
        Spec(("endplate_tip_z_m",), "plate reach", "m", 3,
             "How far the plate gets from the wing towards the car. It must "
             "reach the attachment deck when the wing is endplate-mounted."),
        Spec(("reach_m",), "gap to the deck", "m", 3,
             "Ride height minus the car's attachment height: the distance "
             "the plate (or the pylons) has to span."),
        # --- the SLOTTED two-element section (carwing_multi.py). Absent keys
        # are skipped, so a single-element run renders exactly the rows above.
        Spec(("n_elements",), "elements", "", 0,
             "How many parts the section has. Two means a slot: the flap is "
             "in the SECTION (solved in 2-D and flown as a polar), not in the "
             "lattice, and ONE endplate carries the whole assembly."),
        Spec(("flap_chord_frac",), "flap chord / chord", "", 3,
             "The flap's share of the stowed chord. The two element chords "
             "SUM to the reference chord, so every coefficient above is on "
             "that same reference."),
        Spec(("flap_deflection_deg",), "flap deflection", "deg", 2,
             "Trailing edge towards the track — the slot's own incidence, "
             "kept out of the wing's alpha row so the two families stay "
             "comparable at the same nominal design point."),
        Spec(("slot_gap_frac",), "slot gap (placement)", "", 4,
             "Where the flap's leading edge was PLACED below the main "
             "element's trailing edge. Not the width the flow sees."),
        Spec(("slot_overlap_frac",), "slot overlap", "", 4,
             "Placement in x, positive when the flap's leading edge is "
             "upstream of the main element's trailing edge. Negative is the "
             "Fowler direction."),
        Spec(("slot_width_frac",), "slot width", "", 4,
             "The MEASURED minimum surface-to-surface distance of the built "
             "geometry, as a fraction of the reference chord — smaller than "
             "the placement gap, and it OPENS as the flap deflects."),
        Spec(("slot_gap_te_frac",), "slot gap (trailing edge)", "", 4,
             "Main element's TRAILING EDGE to the flap's surface — the "
             "quantity a wind-tunnel report calls the gap, on the same "
             "flap-stowed chord those reports use. Shown beside the other "
             "two because all three differ (0.0310 / 0.0242 / 0.0240 at the "
             "box centre) and a bound taken from a paper is stated against "
             "THIS one."),
        Spec(("lift_share_main",), "main element lift share", "%", 1,
             "Fraction of the assembly's lift carried by the main element; "
             "the rest is the flap's.", 100.0),
        Spec(("stall_margin_main",), "main stall margin", "", 3,
             "1 − (suction peak / the peak this section is measured to hold "
             "ALONE), at the critical strip. Negative is refused. "
             "CONSERVATIVE: a slot exists precisely to let an element hold a "
             "peak it could not hold alone."),
        Spec(("stall_margin_flap",), "flap stall margin", "", 3,
             "The same criterion for the flap, at its own chord."),
        Spec(("cp_min_main",), "main C_p min", "", 3,
             "Deepest inviscid suction on the main element — the variable the "
             "profile drag is looked up against, because it is what sets the "
             "adverse gradient the boundary layer has to survive."),
        Spec(("cp_min_flap",), "flap C_p min", "", 3, ""),
        Spec(("cp_ceiling_main",), "suction ceiling (main)", "", 3,
             "The peak the MAIN element's section is measured to sustain "
             "ALONE at its own resolved stall — the ceiling its margin is "
             "stated against."),
        Spec(("cp_ceiling_flap",), "suction ceiling (flap)", "", 3,
             "The flap's own, which is a different number as soon as the "
             "flap is a different shape or thickness. Judging the flap "
             "against the main element's would refuse or admit it for a "
             "shape it does not have."),
        Spec(("tc_flap",), "flap thickness", "", 4,
             "The FLAP's own maximum thickness/chord, measured on the shape "
             "it flies rather than asked for."),
        # ``ceiling_source`` / ``ceiling_section`` are deliberately NOT here:
        # this catalogue is numeric by construction (``resolve`` reads a
        # number or nothing), and the two of them are strings. They are in
        # the breakdown and in the design report, which is where a reader
        # asks "which stall data produced this" — gated by
        # test_the_slot_is_presentable_on_the_results_page.
        Spec(("Re_main",), "Re (main element)", "", 0,
             "Reynolds number on the main element's OWN chord, at the MAC."),
        Spec(("Re_flap",), "Re (flap)", "", 0,
             "Reynolds number on the flap's own chord — a few times smaller, "
             "which is why the flap chord band has a floor."),
        # --- the mount as a continuum, the lap, and the ride-height
        # sensitivity (carwing.py's opt-in blocks; all absent by default)
        Spec(("lap_time_s",), "lap time", "s", 3,
             "Time round the stated circuit (cartrack.py, a quasi-steady "
             "point mass). The score is minus this, so a difference of 0.1 IS "
             "a tenth of a second."),
        # ...and the three speeds the lap DERIVES that a scalar grid can
        # hold. Flattened out of the ``lap`` sub-dict by ``enrich``, the same
        # way the tip device's height is flattened out of ``winglet``.
        #
        # NOT ``V_mean``: the lap already reports its own length, and
        # ``V_mean`` is ``length_m / lap_time_s`` exactly (3280 / 61.4288 =
        # 53.395, which is the number it prints). It is the objective
        # restated in another unit, and this catalogue's own rule is one
        # entry per question — LIVE_METRICS' comment is where that rule is
        # written down. The panel says the distance instead, once.
        Spec(("lap_V_top_ms",), "top speed (drag-limited)", "m/s", 2,
             "Where the car's power and its total drag balance — the wing's "
             "drag included. This is the number a rear wing costs on the "
             "straight, and the lap charges it: it is not a bound the "
             "search was given, it is what this design leaves the car."),
        Spec(("lap_V_max_ms",), "fastest point reached", "m/s", 2,
             "The quickest the car actually gets on this layout. Close to "
             "the top speed above means the longest straight ENDS "
             "drag-limited rather than running out first — which is what "
             "makes the wing's drag cost real seconds here rather than "
             "being paid for by a coefficient allowance."),
        Spec(("lap_V_min_ms",), "slowest corner", "m/s", 2,
             "The slowest point of the lap. With the fastest point above it "
             "is the speed RANGE this one section is being asked to work "
             "over — which is the range the per-speed rows sample."),
        Spec(("dCZ_dh_per_m",), "dC_Z/dh", "1/m", 3,
             "How much downforce coefficient the wing loses per metre it "
             "rises. Negative (ground proximity makes downforce); a large "
             "magnitude is a knife-edge platform whose balance moves with "
             "every bump and braking event."),
        Spec(("windup_root_deg",), "elastic windup (root)", "deg", 3,
             "Incidence the wing twists itself to under its own load, at the "
             "root. Zero unless the mount is off the aerodynamic centre."),
        Spec(("windup_contraction",), "windup feedback gain", "", 3,
             "Ratio of successive corrections in the aeroelastic fixed point "
             "— the MEASURED amplification eigenvalue. At or above 1 the wing "
             "is diverging and the design is refused."),
        Spec(("pylon_length_m",), "pylon length", "m", 3,
             "From the wing down to the car's attachment DECK — not to the "
             "track. Absent unless a deck was stated."),
        Spec(("aero_balance",), "aero balance", "", 3,
             "Front axle's share of the total downforce."),
    ),
    "section": (
        Spec(("cd_counts",), "c_d", "counts", 1,
             "Section drag at the design lift coefficient (1 count = 1e-4)."),
        Spec(("cl_design",), "design c_l", "", 3, ""),
        Spec(("tc",), "t/c", "", 4, "Maximum thickness / chord."),
    ),
}

GEOMETRY = (
    Spec(("b",), "span b", "m", 3, "Reference span."),
    Spec(("S",), "area S", "m²", 3, "Reference planform area."),
    Spec(("taper",), "taper λ", "", 3,
         "Tip chord / root chord of the straight-taper BASELINE. With a free "
         "chord law the flown planform is that baseline reshaped — read it "
         "with the chord deviation below."),
    Spec(("chord_dev",), "chord deviation", "", 3,
         "max |c / c_trapezoid − 1| over the span: how far the free chord "
         "law moved the planform off straight taper (area is held fixed)."),
    Spec(("sweep_deg",), "sweep", "deg", 2, "Quarter-chord sweep."),
    Spec(("tc",), "t/c", "", 4, "Section thickness ratio."),
    Spec(("twist_root_deg",), "twist root", "deg", 2, ""),
    Spec(("twist_tip_deg",), "twist tip", "deg", 2, ""),
)

#: family -> (breakdown key that proves it, human name)
_FAMILY_MARKERS = (
    ("carwing", "CZ", "car rear wing"),
    ("tail", "SM", "wing + tail"),
    ("tandem", "lift_share_front", "tandem pair"),
    ("hydrofoil", "sigma_cav", "hydrofoil"),
    ("aircraft", "W_total_N", "free planform"),
    ("section", "cd_counts", "airfoil section"),
    ("winglet", "winglet", "winglet"),
)

#: what a run MAXIMISED -> the catalogue key that IS that quantity.
#:
#: A headline that does not lead with the number the search was scored on is a
#: headline about a different question. The car families made that visible:
#: with ``objective='laptime'`` the answer IS the lap time, and until this
#: table existed the headline led with C_Z (the first of the family's
#: specialities in catalogue order) while the lap time sat six rows down and
#: the shells' hero readout showed the raw score — MINUS 61.429, unlabelled,
#: beside a lap time of 61.429 s further down the same panel. Two spellings of
#: one number, one of them negative, is worse than showing it once.
#:
#: Keyed off the breakdown's own ``objective`` field, which the car families
#: record beside ``objective_label``. A family that does not record one, or
#: records a value with no named row, simply gets the old order — this
#: promotes a row that is already in the catalogue and never invents one.
OBJECTIVE_LEAD = {
    "laptime": "lap_time_s",
    "downforce": "downforce_N",
    "efficiency": "efficiency",
    "cz": "CZ",
}


#: how many speciality metrics ride along in the headline row
_HEADLINE_SPECIALS = 3
_HEADLINE_PERF = ("LoD", "CL", "CD", "CDi", "e", "alpha_deg")


def family(bd: dict) -> str:
    """Which speciality this breakdown belongs to (``""`` = plain wing)."""
    for name, marker, _ in _FAMILY_MARKERS:
        if bd.get(marker) is not None:
            return name
    return ""


def family_label(bd: dict) -> str:
    for name, marker, label in _FAMILY_MARKERS:
        if bd.get(marker) is not None:
            return label
    return "wing"


def _number(v):
    """Finite float, or None for anything that cannot be shown as a number."""
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        f = float(v)
        return f if math.isfinite(f) else None
    return None


def fmt(value: float, digits: int = 3) -> str:
    """Readable fixed/scientific text for a metric value."""
    a = abs(value)
    if a < 1e-9:
        # a trim residual solved to 1e-16 is ZERO to a reader; printing
        # "-1.77e-16" only invites the question of what went wrong
        return "0"
    if a >= 1e6 or a < 1e-4:
        return f"{value:.3g}"
    if digits <= 0:
        return f"{value:,.0f}"
    return f"{value:.{digits}f}"


def resolve(spec: Spec, bd: dict):
    """(value, text) for ``spec`` against a breakdown, or None if absent."""
    for k in spec.keys:
        v = _number(bd.get(k))
        if v is not None:
            return v * spec.scale, fmt(v * spec.scale, spec.digits)
    return None


def _rows(specs, bd: dict) -> list[dict]:
    out = []
    for sp in specs:
        got = resolve(sp, bd)
        if got is None:
            continue
        value, text = got
        out.append({"label": sp.label, "value": value, "text": text,
                    "unit": sp.unit, "tip": sp.tip, "key": sp.keys[0]})
    return out


def enrich(bd: dict, geom: dict | None = None) -> dict:
    """Breakdown + a few DERIVED presentables (drag counts, section c_d…).

    Kept out of the physics: these are restatements of numbers the run
    already recorded, in the units an aerodynamicist reads them in.
    """
    out = dict(bd or {})
    geom = geom or {}
    cd = _number(out.get("CD"))
    if cd is not None:
        out.setdefault("CD_counts", cd * 1e4)
    for src, dst in (("CDi", "CDi_counts"), ("CDi_total", "CDi_counts"),
                     ("CDp", "CDp_counts")):
        v = _number(out.get(src))
        if v is not None:
            out.setdefault(dst, v * 1e4)
    pol = out.get("polar")
    if isinstance(pol, dict):
        cdd = _number(pol.get("cd_at_cl_design"))
        if cdd is not None:
            out.setdefault("cd_counts", cdd * 1e4)
        cld = _number(pol.get("cl_design"))
        if cld is not None:
            out.setdefault("cl_design", cld)
    wl = geom.get("winglet") if isinstance(geom.get("winglet"), dict) else None
    if isinstance(out.get("winglet"), dict):
        wl = wl or out["winglet"]
    if wl:
        for k, dst in (("h_m", "winglet_h_m"), ("cant_deg", "winglet_cant_deg"),
                       ("h_frac", "winglet_h_frac")):
            v = _number(wl.get(k))
            if v is not None:
                out.setdefault(dst, v)
    # THE LAP's own derived speeds, out of its sub-dict and into the grid —
    # the same lift the tip device's height gets above, and for the same
    # reason: a number computed on every run of this family and reachable
    # only by opening a nested dict is a number nobody reads. ``lap_time_s``
    # is NOT among them: the solvers already promote it themselves, and
    # promoting it twice would put two spellings of one number in the
    # catalogue.
    lap = out.get("lap")
    if isinstance(lap, dict):
        for k, dst in (("V_top", "lap_V_top_ms"), ("V_max", "lap_V_max_ms"),
                       ("V_min", "lap_V_min_ms")):
            v = _number(lap.get(k))
            if v is not None:
                out.setdefault(dst, v)
    for k in ("b", "S", "taper", "sweep_deg", "tc", "twist_root_deg",
              "twist_tip_deg"):
        v = _number(geom.get(k))
        if v is not None:
            out.setdefault(k, v)
    # a tail reports BOTH the stabilator incidence and the equivalent
    # elevator deflection whatever the control is; showing the one that was
    # not flown reads as two contradictory settings
    if out.get("control") == "elevator":
        out.pop("i_t_deg", None)
    elif out.get("control") is not None:
        out.pop("delta_e_deg", None)
    return out


WINGLET_EXTRA = (
    Spec(("winglet_h_m",), "winglet height", "m", 3,
         "Height of the winglet actually flown."),
    Spec(("winglet_cant_deg",), "cant", "deg", 1,
         "0° = a planar span extension, 90° = fully vertical."),
    Spec(("winglet_h_frac",), "h / (b/2)", "", 4,
         "Winglet height as a fraction of the semi-span."),
)


def objective_row(bd: dict, geom: dict | None = None):
    """The metric that IS what this run maximised, or None.

    ``None`` for every family that does not record an ``objective`` the
    catalogue has a row for, which is most of them — a caller then falls back
    to whatever it showed before, and no family gets a hero readout invented
    for it.
    """
    full = enrich(bd, geom)
    key = OBJECTIVE_LEAD.get(str(full.get("objective") or ""))
    if key is None:
        return None
    specs = SPECIALITY.get(family(full), ()) + PERFORMANCE + GEOMETRY
    spec = next((sp for sp in specs if sp.keys[0] == key), None)
    if spec is None:
        return None
    rows = _rows((spec,), full)
    return rows[0] if rows else None


def headline(bd: dict, geom: dict | None = None) -> list[dict]:
    """The few numbers that ARE the answer, in reading order.

    Performance first (L/D, C_L, C_D, α), then this family's own headline
    quantities — a tail run leads with static margin, a hydrofoil with its
    cavitation margin, because that is the number the run was about.
    """
    full = enrich(bd, geom)
    perf = [r for r in _rows(PERFORMANCE, full)
            if r["key"] in _HEADLINE_PERF]
    fam = family(full)
    specials = SPECIALITY.get(fam, ())
    if fam == "winglet":
        specials = WINGLET_EXTRA + specials
    rows = _rows(specials, full)
    # ...led by the one the run was SCORED on, where the family records it
    # (OBJECTIVE_LEAD). Hoisted rather than appended, and hoisted BEFORE the
    # truncation below, or the row that is the answer would be the one the
    # truncation dropped.
    lead = objective_row(full)
    if lead is not None:
        rows = ([lead] + [r for r in rows if r["key"] != lead["key"]])
    rows = rows[:_HEADLINE_SPECIALS]
    if fam == "carwing":
        # a downforce run is not answered by L/D: its own numbers lead
        return rows + perf
    return perf + rows


def groups(bd: dict, geom: dict | None = None) -> list[tuple[str, list]]:
    """Every named metric, grouped for the aerodynamics tab.

    A design can belong to more than one family now that the tip device and
    the tail are solved together (wingtail.py): the family MARKERS pick the
    leading one, and a tip device flown alongside it gets its own group
    rather than disappearing because something more specific matched first.
    """
    full = enrich(bd, geom)
    fam = family(full)
    out = [("Performance", _rows(PERFORMANCE, full))]
    specials = SPECIALITY.get(fam, ())
    if fam == "winglet":
        specials = WINGLET_EXTRA + specials
    if specials:
        out.append((f"{family_label(full).capitalize()} specifics",
                    _rows(specials, full)))
    if fam != "winglet" and full.get("winglet") is not None:
        # WINGLET_EXTRA only: the induced drag in SPECIALITY["winglet"] is
        # already in Performance, and repeating it reads as two numbers
        out.append(("Tip device", _rows(WINGLET_EXTRA, full)))
    out.append(("Geometry", _rows(GEOMETRY, full)))
    return [(title, rows) for title, rows in out if rows]


#: an optimum this close to a box edge (as a fraction of the row's width) is
#: RIDING it: the BOX is setting that variable, not the physics (§13
#: boundary-riding law). Same 2 % the V1 results page has always used.
RIDING_TOL = 0.02


def design_box(labels, x, bounds=None, overrides=None,
               tol: float = RIDING_TOL) -> list[dict]:
    """One row per design variable: the winning value INSIDE its own box.

    ``bounds`` is the box the run actually searched — ``RunResult.bounds``,
    i.e. the family's published box with this run's overrides already
    applied, in design-vector order. It is the only honest source: the
    spec's ``default_bounds`` is built with no flags, so a winglet cant or
    a chord-law row could be compared against a box the run never had.

    Each row carries ``label``, ``value``, ``lo``, ``hi``, ``span``,
    ``frac`` (0 at the low bound, 1 at the high one), ``riding``
    (``""`` / ``"low"`` / ``"high"``), ``outside`` and ``narrowed`` (the row
    appears in ``overrides``, so this box is not the published one).

    Anything missing stays None rather than being invented: a record with no
    box gives rows with values and no bounds, which is exactly what it
    knows.
    """
    labels = list(labels or [])
    values = list(x or [])
    box = list(bounds or [])
    ov = overrides or {}
    rows = []
    for i, raw in enumerate(values):
        label = str(labels[i]) if i < len(labels) else f"x{i}"
        value = _number(raw)
        pair = box[i] if i < len(box) else None
        lo = hi = None
        if isinstance(pair, (list, tuple)) and len(pair) == 2:
            lo, hi = _number(pair[0]), _number(pair[1])
        span = frac = None
        riding, outside = "", False
        if lo is not None and hi is not None:
            span = hi - lo
            if value is not None and span > 0:
                frac = (value - lo) / span
                outside = not 0.0 <= frac <= 1.0
                if frac <= tol:
                    riding = "low"
                elif frac >= 1.0 - tol:
                    riding = "high"
        rows.append({"label": label, "value": value, "lo": lo, "hi": hi,
                     "span": span, "frac": frac, "riding": riding,
                     "outside": outside, "narrowed": label in ov})
    return rows


def drag_split(bd: dict) -> list[dict]:
    """Induced / profile drag split in counts, for the drag bar."""
    full = enrich(bd)
    parts = []
    for key, label in (("CDi_counts", "induced"), ("CDp_counts", "profile")):
        v = _number(full.get(key))
        if v is not None:
            parts.append({"label": label, "counts": v})
    total = _number(full.get("CD_counts"))
    if total is not None and parts:
        known = sum(p["counts"] for p in parts)
        rest = total - known
        if rest > 0.5:                       # 0.05 counts of rounding noise
            parts.append({"label": "other", "counts": rest})
    return parts


# --------------------------------------------------------------- the LAP
#
# A lap is not one number. ``lap_time_s`` has had a Spec since cartrack.py
# was written, and it was the ONLY thing about the lap a reader could see:
# the per-segment breakdown, the representative speeds the wing was actually
# re-flown at, the drag-limited top speed and the aero balance window were all
# computed on every run and printed nowhere. Two of those are TABLES and a
# scalar grid cannot hold a table, which is why they need their own reader
# rather than three more Specs.
#
# Everything here is a restatement of what the run recorded — no physics, no
# UI — so the wording and the arithmetic are assertable without a browser,
# the same split the rest of this module keeps.

#: how a segment's ``limited_by`` reads on the page. cartrack sets it per
#: segment and the word is the whole point of the row: a corner taken at the
#: GRIP limit is one a wing can still buy speed in, and a corner reported as
#: ``aero_unbounded`` is one where the downforce already outruns the radius,
#: so buying more of it there is buying nothing.
LAP_LIMIT_WORDS = {
    "grip": "grip",
    "aero_unbounded": "aero-unbounded — grip never runs out here",
    "distance": "how long the straight is",
    "power": "power",
    "top_speed": "top speed",
}


def lap_report(bd: dict) -> dict | None:
    """The lap as presentation data, or None when this run had no circuit.

    The three parts a reader of a lap needs and could not previously see:

    ``summary``
        the circuit's own name and distance, and the integration step the
        straights were solved at. The NAME is carried because every number
        below belongs to that layout and to no other — the shipped one says
        "not a real circuit" in its own name, and a page that dropped the
        name would be quoting a lap time as though it were a property of the
        wing.
    ``segments``
        one row per segment of the lap, in the caller's own order, with the
        time it took, the speeds at its ends and WHAT limited it. Straights
        additionally carry ``v_peak_over_top`` and ``drag_frac_at_peak`` —
        how close to the drag-limited top speed the car got, and what
        fraction of the engine was going into drag when it got there. Those
        two are what say whether a straight actually CHARGES the wing's drag
        or merely runs out first, which is the difference between a lap that
        prices a rear wing and one that does not.
    ``points``
        the representative speeds the wing was re-flown at, with the weight
        each carries in the lap and the coefficients it produced there. This
        is what ``track_points`` bought: without it the table is one row and
        the section's speed dependence is invisible.
    ``balance``
        the front axle's share of the downforce against the window it is
        judged in — the number alone is unreadable, because 0.49 is only
        "nose-light" relative to a window whose centre cartrack DERIVES from
        the car's static weight split.

    The spanwise speed/position PROFILES each segment carries are deliberately
    not here. They are ~200-point arrays per straight, they are a plot and not
    a table, and nothing on the results page draws them yet.
    """
    lap = bd.get("lap") if isinstance(bd, dict) else None
    if not isinstance(lap, dict) or _number(lap.get("lap_time_s")) is None:
        return None
    total = _number(lap.get("lap_time_s"))

    segments = []
    for i, seg in enumerate(lap.get("segments") or ()):
        if not isinstance(seg, dict):
            continue
        t = _number(seg.get("t_s"))
        radius = _number(seg.get("radius_m"))
        kind = str(seg.get("kind") or "")
        segments.append({
            "index": int(_number(seg.get("index")) or i),
            "kind": kind,
            "label": (f"corner R = {fmt(radius, 0)} m" if radius is not None
                      else kind or "segment"),
            "length_m": _number(seg.get("length_m")),
            "t_s": t,
            # what share of the LAP this segment is. The reason a corner
            # matters is how long the car spends in it, and 3.1 s on its own
            # does not say that.
            "share": (None if t is None or not total else t / total),
            "v_in": _number(seg.get("V_in")),
            "v_out": _number(seg.get("V_out")),
            "v_peak": _number(seg.get("V_peak")),
            "limited_by": LAP_LIMIT_WORDS.get(str(seg.get("limited_by")),
                                              str(seg.get("limited_by") or "")),
            "v_peak_over_top": _number(seg.get("V_peak_over_top")),
            "drag_frac_at_peak": _number(seg.get("drag_frac_at_peak")),
        })

    points = []
    for row in (bd.get("track_points") or ()):
        if not isinstance(row, dict):
            continue
        points.append({k: _number(row.get(src)) for k, src in
                       (("v", "V"), ("weight", "weight"), ("cz", "CZ"),
                        ("cd", "CD"), ("cz_a_m2", "cz_a_m2"),
                        ("cd_a_m2", "cd_a_m2"),
                        ("ride_height_m", "ride_height_m"),
                        ("re_flown", "Re_flown"))})

    bal = bd.get("balance")
    balance = None
    if isinstance(bal, dict) and _number(bal.get("aero_balance")) is not None:
        window = bal.get("window")
        window = (tuple(float(w) for w in window)
                  if isinstance(window, (list, tuple)) and len(window) == 2
                  else None)
        balance = {"value": _number(bal.get("aero_balance")),
                   "window": window,
                   "static_front_frac": _number(bal.get("static_front_frac")),
                   "load_balance": _number(bal.get("load_balance")),
                   "feasible": bool(bal.get("feasible", True)),
                   "reason": str(bal.get("reason") or "")}

    return {
        "summary": {
            "track": str(lap.get("track") or "the stated circuit"),
            "length_m": _number(lap.get("length_m")),
            "lap_time_s": total,
            "n_steps": _number(lap.get("n_steps")),
        },
        "warnings": [str(w) for w in (lap.get("warnings") or ())],
        "segments": segments,
        "points": points,
        "balance": balance,
    }


# ------------------------------------------------- what a LIVE search shows
#
# The incumbent sampler (gui/v3/sampler.py) reports EVERY numeric key the
# breakdown carries: 13 on a 2-D section and 46 the moment the section is
# scored on a wing. That is not a menu, it is a dump — and most of it is the
# same number said twice. The redundancies, measured on the wing-mode
# breakdown:
#
#   LoD  = LD = f = score          the objective, four times
#   CD   / CD_counts               the same drag, x 1e4
#   CDi  / CDi_counts / CDi_total  ...and again
#   CDp  / CDp_counts              ...and again
#   S    / S_llt / S_ref           the reference area, three ways
#   taper / taper_flown            the baseline and the flown one (equal
#                                  without a chord law)
#   mac  / mac_true, re / re_true  the same pair twice more
#   cl_design, tc_min...           CONSTANTS of the run, flat by definition
#   n_branch / n_converged         XFOIL bookkeeping, not aerodynamics
#
# So the panel offers a SHORTLIST: one entry per question, the preferred
# spelling of each, in the order an aerodynamicist reads them. Everything
# else is still there behind a disclosure — nothing is hidden, it is ranked.
#
#: ordered (canonical key, alternates, label). The FIRST key present in the
#: sample is the one offered; the alternates are the spellings it replaces,
#: which is what makes this a de-duplication rather than a filter.
LIVE_METRICS: tuple = (
    # A composite run's objective is J. It reaches the panel three times —
    # ``composite`` plus the harness's generic ``f`` / ``score`` — and
    # ``live_metric_menu`` drops the generic pair when ``composite`` is
    # present, so the number is offered once, under the name it has, instead
    # of a third of the panel being J labelled "L/D".
    (("composite",), "composite score J"),
    (("LoD", "LD", "f", "score"), "L/D (the objective)"),
    (("cd_counts", "cd"), "section c_d"),
    (("CD_counts", "CD"), "C_D"),
    (("CDi_counts", "CDi", "CDi_total"), "C_Di"),
    (("CDp_counts", "CDp"), "C_Dp"),
    (("e",), "span efficiency e"),
    # ``clmax`` (no underscore) and ``astall`` / ``ldcr`` / ``ldmax`` are the
    # SCREEN's spellings (airfoil_select.polar_metrics), which reach the panel
    # on a composite run — the criteria J is made of belong beside J.
    (("cl_max", "cl_max_branch", "clmax"), "c_l max"),
    (("astall",), "stall angle [deg]"),
    (("ldcr",), "L/D at design c_l"),
    (("ldmax",), "(L/D) max"),
    # ...and the SCREEN's drag at the design lift, which is a criterion now
    # (airfoil_select's ``cdcr``) and is the one a zero-lift surface is
    # ranked on. Spelled ``cd_at`` because that is the field polar_metrics
    # writes; the plain ``cd`` above is the -cd problem's own.
    (("cd_at",), "c_d at design c_l"),
    (("cm",), "c_m"),
    (("tc",), "t/c"),
    (("r_le",), "leading-edge radius"),
    (("alpha_deg", "alpha_root_deg"), "α"),
    (("twist_env_deg",), "max |twist|"),
    (("chord_dev",), "chord deviation"),
    (("g_tc",), "margin: t/c"),
    (("g_cm",), "margin: |c_m|"),
    (("g_alpha",), "margin: local α"),
)

#: what the panel TICKS when nothing has been chosen. Three: the objective,
#: the drag it is made of, and the one number a shape search moves that the
#: objective does not see. More than three on one pair of axes is a
#: spaghetti plot.
LIVE_METRIC_DEFAULTS = ("LoD", "cd_counts", "cl_max")

#: keys that are CONSTANT for a run (its operating point and its gates) or
#: pure solver bookkeeping. Never offered as a live series: a flat line and
#: an evaluation counter are not results.
LIVE_METRIC_NEVER = frozenset({
    "cl_design", "re", "re_true", "Re_mac", "mach", "S", "S_ref", "S_llt",
    "b", "AR", "n_branch", "n_converged", "n_clamped_low", "i", "n",
})


def live_metric_menu(available) -> tuple:
    """``(shortlist, rest)`` for a live-metric panel.

    ``shortlist`` is ``[(key, label)]`` — the catalogue's order, one entry per
    question, each spelled the way :data:`LIVE_METRICS` prefers and only
    where this run actually reported it. ``rest`` is every other key the
    sample carried, minus the constants and the bookkeeping, so nothing is
    lost — it is ranked.
    """
    have = list(available)
    # ``f`` / ``score`` are the harness's names for WHATEVER was maximised, so
    # on a composite run they are the same number as ``composite``. Drop the
    # generic pair there: offered under the L/D entry they would plot J
    # labelled "L/D", and offering one number twice is worse than once.
    if "composite" in have:
        have = [k for k in have if k not in ("f", "score")]
    seen: set = set()
    shortlist = []
    for keys, label in LIVE_METRICS:
        chosen = next((k for k in keys if k in have), None)
        seen.update(keys)
        if chosen is not None:
            shortlist.append((chosen, label))
    rest = [k for k in have if k not in seen and k not in LIVE_METRIC_NEVER]
    return tuple(shortlist), tuple(rest)


def live_metric_defaults(available) -> list:
    """Which keys a fresh panel ticks, filtered to what this run reports.

    Falls back to the first shortlist entry (the objective, under whatever
    name this family spells it) so the plot is never empty while the sampler
    has data.
    """
    # THE DEFAULTS TICK WHAT THE MENU OFFERS. A composite run's objective is
    # ``composite``, and the menu deliberately stops offering the generic
    # ``f`` / ``score`` there — ticking one of those would draw a series with
    # no checkbox beside it.
    offered = {k for k, _ in live_metric_menu(available)[0]}
    have = set(available)
    picked = []
    for want in LIVE_METRIC_DEFAULTS:
        keys = next((ks for ks, _ in LIVE_METRICS if want in ks), (want,))
        chosen = next((k for k in keys if k in have and k in offered), None)
        if chosen is not None and chosen not in picked:
            picked.append(chosen)
    if "composite" in offered and "composite" not in picked:
        picked.insert(0, "composite")
    if not picked:
        short, _ = live_metric_menu(available)
        picked = [short[0][0]] if short else []
    return picked


def live_metric_panel(available, selected=()) -> tuple:
    """``(primary, more)`` for a panel that LEADS WITH A FEW.

    :func:`live_metric_menu` already drops the duplicate spellings, but on a
    wing breakdown its shortlist is still fifteen or more entries — a wall of
    checkboxes above a plot that can carry two axes. So the panel leads with
    the handful a reader wants on sight and puts the rest one press away:

    * ``primary`` — ``[(key, label)]``: this run's defaults
      (:func:`live_metric_defaults`), plus anything the user has already
      ticked, so a chosen metric never hides inside a collapsed expansion;
    * ``more`` — every other key the sample carried, ranked, labelled by the
      catalogue where it has a name and by its raw key where it does not.

    Nothing is dropped: ``primary + more`` is the whole menu.
    """
    short, rest = live_metric_menu(available)
    labels = dict(short)
    rest_keys = list(rest)
    lead: list = []
    for key in list(live_metric_defaults(available)) + list(selected or ()):
        if key not in lead and (key in labels or key in rest_keys):
            lead.append(key)
    primary = tuple((k, labels.get(k, k)) for k in lead)
    more = tuple([(k, labels[k]) for k, _ in short if k not in lead]
                 + [(k, k) for k in rest_keys if k not in lead])
    return primary, more


# ------------------------------------------------- the refusal on an axis
#: What every objective in this repo returns for a design the physics could
#: not score — ``aerobo.objective.PENALTY``. Imported rather than re-typed,
#: because a convergence plot that clipped at a number the solver had since
#: moved would silently stop hiding the cliff it exists to hide.
def refusal_sentinel() -> float:
    """The failure sentinel, read off the solver (never a local constant)."""
    from aerobo.objective import PENALTY

    return float(PENALTY)


def _percentile(sorted_vals: list, pct: float) -> float:
    """Linear-interpolated percentile of an already sorted list (no numpy —
    this module carries no array dependency and is imported by the shell)."""
    if not sorted_vals:
        raise ValueError("percentile of an empty sample")
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = (len(sorted_vals) - 1) * max(0.0, min(100.0, float(pct))) / 100.0
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return float(sorted_vals[lo]) * (1.0 - frac) + float(sorted_vals[hi]) * frac


def convergence_yrange(values, must_show=(), *, pad_frac: float = 0.06,
                       floor_pct: float = 2.0):
    """A y range for an objective axis that a REFUSAL cannot squash.

    A design the physics could not score returns exactly the sentinel
    (:func:`refusal_sentinel`, -100). Plotted on the same axis as the thing
    the run is actually doing — an L/D of 40, a 0-100 composite, a drag
    coefficient of 1e-3 — one refusal owns the whole axis and the
    convergence the plot exists to show is a flat line at the top of it.

    So the axis is bounded by the designs that FLEW:

    * the sentinel is dropped outright (it is a label, not a value);
    * with enough scored points, the bottom is their ``floor_pct``
      percentile rather than their minimum, so a constrained family's own
      large penalties do not do the sentinel's job for it;
    * ``must_show`` (the best-so-far trace) is always inside, whatever the
      percentile says — the curve the user is reading is never cropped.

    Returns ``(lo, hi)``, or **None** when nothing is off scale. None means
    "let plotly autoscale": a clean run is drawn exactly as it always was,
    and the caller can use the None to decide whether to say anything about
    points below the axis at all.
    """
    sentinel = refusal_sentinel()

    def _finite(seq):
        out = []
        for v in seq or ():
            try:
                f = float(v)
            except (TypeError, ValueError):
                continue
            if math.isfinite(f):
                out.append(f)
        return out

    every = _finite(values) + _finite(must_show)
    if not every:
        return None
    scored = sorted(v for v in _finite(values) if v != sentinel)
    keep = sorted(v for v in _finite(must_show) if v != sentinel)
    base = scored or keep
    if not base:
        return None                      # everything was refused: no scale
    hi = max(max(base), max(keep) if keep else max(base))
    lo = (_percentile(scored, floor_pct) if len(scored) >= 8
          else min(base))
    if keep:
        lo = min(lo, min(keep))
    span = hi - lo
    pad = span * pad_frac if span > 0.0 else max(abs(hi) * 0.05, 1e-9)
    lo, hi = lo - pad, hi + pad
    if min(every) >= lo:
        return None                      # nothing is off scale
    return (lo, hi)


def offscale(values, lo: float) -> list:
    """``[(index, value)]`` for the points a range of ``lo`` crops away.

    A non-finite value counts as off scale too: it is a point that was
    evaluated and cannot be drawn, which is the same thing to a reader
    counting how much of the run is missing from the picture.
    """
    out = []
    for i, v in enumerate(values or ()):
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(f) or f < lo:
            out.append((i, f))
    return out


# ---------------------------------------------------------------- verdicts
def verdict(delta, tol: float = 1e-12) -> str:
    """The verdict of a change on an UP-IS-BETTER quantity.

    Four values, because a comparison table has four things to say:
    ``"better"``, ``"worse"``, ``"same"`` (judged, and it did not move) and
    ``""`` (nothing to judge — not measured, or no preferred direction).
    ``same`` and ``""`` are kept apart because "the search did not move this"
    and "nobody can say which way is better" are different sentences; a card
    is free to paint both grey, but it must not be forced to.

    Same vocabulary as ``nice_app._cmp_row``'s ``dir``, so one colour rule
    (:func:`gui.v3.widgets.verdict_slots`) serves every seed-vs-optimised
    table in the shell. Sub-scores and composites are 0-100 band numbers, so
    the sign of the change IS the verdict; the tolerance is there to stop
    float noise being painted green.
    """
    if delta is None:
        return ""
    try:
        d = float(delta)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(d):
        return ""
    if abs(d) <= tol:
        return "same"
    return "better" if d > 0.0 else "worse"
