"""Tier C: hydrofoil under a free surface — negative-image lifting line.

Model: a lifting surface at depth ``h`` below the free surface in the HIGH
depth-Froude-number regime (planing / foiling speeds),

    Fn_h = V / sqrt(g h)  >>  1.

In that limit the linearised free-surface condition degenerates to the
zero-pressure ("negative image") plane, and the surface effect is captured
by an image vortex system mirrored across the surface plane. The image's
circulation is LINEAR in the real foil's circulation, so — exactly as in
the tandem coupling blocks (tandem.py) — it folds into the influence
matrix and the problem stays ONE linear system with no new unknowns.

Image sign (the classic trap — derived, not assumed)
----------------------------------------------------
Linearised steady free-surface condition on z = z_fs:

    U^2 phi_xx + g phi_z = 0
      Fn -> 0   :  phi_z = 0  (rigid wall)      -> SYMMETRIC extension of phi
      Fn -> inf :  phi_xx = 0 => phi = 0        -> ANTI-SYMMETRIC extension

Anti-symmetric extension (phi -> -phi across the plane): u, v are odd in
(z - z_fs), w is even. Vorticity therefore keeps its omega_x and omega_y
components: the image of a lifting wing is an IDENTICAL wing (same
circulation sense in the fixed frame, i.e. also "lifting up") at the
mirror point z = 2 z_fs - z_foil. The real foil consequently sits in the
DOWNWASH of a same-sign biplane partner at gap 2h:

    CL_alpha drops and CDi rises as h -> 0,  CDi = CDi_iso * (1 + sigma(2h/b))

with sigma Prandtl's biplane interference factor — the anti-ground-effect.
(The rigid-wall / Fn -> 0 image is the OPPOSITE one: the mirror wing lifts
down, giving upwash and the familiar ground-effect drag reduction. The
word "negative image" refers to the sign flip of the POTENTIAL, not of the
image filaments' fixed-frame circulation sense.)

Only the REAL foil is physical, so the surface-induced ("mutual") drag is
counted once — on the real foil — unlike the tandem case where both
surfaces carry forces.

Validity and limitations (stated per plan; see report §11)
----------------------------------------------------------
* High-Fn limit: results apply for Fn_h >~ 2-3. At the Tier C design point
  (V ~ 8-12 m/s, h <= ~1 m) Fn_h = V/sqrt(g h) >= 2.6 — inside the regime.
  The finite-Fn wave-making part of the surface effect (and wave drag) is
  NOT modelled.
* Lifting-line scope: the image captures the 3-D induced part of the
  surface effect (trailing-vortex downwash). The 2-D section-level lift
  loss from the image BOUND vortex's streamwise velocity at the foil
  (relevant for h <~ 1 chord) is a chordwise effect outside lifting-line
  resolution — flagged as a limitation, not modelled.
* Water properties below; section polars are XFOIL runs at Re = 1e6 —
  chord/speed choices in the Tier C problem keep Re close to that (the
  residual Re sensitivity is flagged in the report).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from . import fin as _fin
from . import geometry
from .llt import LLTResult, TWO_PI, fourier_system
from .tandem import Surface, VortexSystem, mutual_cdi, w_influence, _panel_quadrature

# ---------------------------------------------------------------- water/air constants

RHO_WATER = 1025.0        # kg/m^3, seawater
MU_WATER = 1.08e-3        # Pa s, seawater ~20 C
G_GRAV = 9.81             # m/s^2
P_ATM = 101325.0          # Pa
P_VAP = 2340.0            # Pa, water vapour pressure ~20 C

#: The two waters a foil is actually flown in, at ~20 C. SEA is the module
#: default and the published one (the constants above, by reference), so
#: nothing changes unless a caller asks for fresh. Three properties move
#: together and all three matter: density and viscosity set the loading and
#: the Reynolds number, and the vapour pressure sets where cavitation
#: starts — carrying only rho into fresh water would quietly keep sea
#: water's cavitation threshold.
WATER_KINDS: dict[str, dict] = {
    "sea": {"label": "sea water (~20 C)", "rho": RHO_WATER, "mu": MU_WATER,
            "p_vap": P_VAP},
    "fresh": {"label": "fresh water (~20 C)", "rho": 998.2, "mu": 1.002e-3,
              "p_vap": 2339.0},
}


def water_properties(kind: str = "sea") -> dict:
    """``{rho, mu, p_vap}`` of a named water (see :data:`WATER_KINDS`)."""
    if kind not in WATER_KINDS:
        raise ValueError(f"unknown water {kind!r}; "
                         f"choose from {tuple(WATER_KINDS)}")
    return {k: float(v) for k, v in WATER_KINDS[kind].items()
            if k != "label"}


def froude_depth(V: float, h: float) -> float:
    """Depth Froude number Fn_h = V / sqrt(g h). Model valid for Fn_h >> 1."""
    return float(V / np.sqrt(G_GRAV * h))


# ---------------------------------------------------------------- cavitation

def sigma_cav(depth: float, V: float, rho: float = RHO_WATER,
              p_vap: float = P_VAP) -> float:
    """Cavitation number at foil depth ``depth`` and speed ``V``:

        sigma_cav = (p_atm + rho g h - p_v) / (1/2 rho V^2)

    No cavitation anywhere on the section requires -Cp_min <= sigma_cav.
    Note the two-way coupling that makes the Tier C problem interesting:
    sigma_cav TIGHTENS toward the surface (less static head) and with
    speed, while the free-surface image ALSO pushes the optimum deep.
    """
    return float((P_ATM + rho * G_GRAV * depth - p_vap) / (0.5 * rho * V**2))


#: display name of the draught margin, in one place because three modules
#: report it (hydrofoil, hydrotail, and the api spec descriptions).
DRAUGHT_CONSTRAINT_LABEL = "draught margin"

#: display name of the IMMERSION margin, beside the draught's for the same
#: reason. The two are opposite ends of the same craft: the draught cap asks
#: how DEEP the assembly may reach, the immersion margin asks whether the
#: highest part of it is still in the water at all.
IMMERSION_CONSTRAINT_LABEL = "immersion margin"


# ---------------------------------------------------------------- heel

#: The heel angle at which this model stops describing a foiling craft.
#:
#: NOT a calibration and not a taste, which is why it is a refusal and not a
#: band (``a-calibration-is-a-default-not-a-ban``): at |phi| = 90 deg the
#: foil stands on edge, its lift carries none of the weight (the trim target
#: goes as ``1/cos phi``, i.e. to infinity), and the free-surface image —
#: a mirror in a HORIZONTAL plane — is being asked about a vertical surface.
#: Everything strictly inside is accepted, priced and flown, including heel
#: angles no rider would hold: a windfoil carving at 40 deg is a real state
#: and this package's job is to price it, not to have an opinion about it.
HEEL_LIMIT_DEG = 90.0


def heel_angle(value) -> float:
    """Validate a stated heel angle [deg] -> float, or refuse it.

    HEEL IS A STATED FLIGHT CONDITION HERE, not a derived one, and the
    reason is the same one ``rig.RigLoads.z_ce_m`` has no default: closing
    it needs a model this package does not have. In steady straight flight
    the rig's side force and the appendage's lateral reaction form a ROLL
    couple, and what balances it is the RIDER — a mass on a lever the rider
    chooses, moment by moment. A heel angle derived from the side force
    alone would therefore be a rider model wearing a physics hat, and it
    would set the single number that decides how much span the craft can
    carry. ``rig.righting_lever_required_m`` reports what lateral offset a
    stated side force demands, so the closure is on the page instead of
    invented here.

    ZERO IS THE DEFAULT AND IT IS A REAL STATE — a foil under a tow boat, a
    dinghy sailed flat, and every water run this package has published. At
    zero every quantity below reduces exactly to the flat-water expression
    it replaced (``sin 0 = 0``, ``cos 0 = 1``), which is what makes the
    whole feature bit-for-bit invisible until it is asked for.
    """
    try:
        phi = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"heel_deg {value!r} is not a number; it is the craft's roll "
            f"angle in DEGREES, positive lifting the +y (starboard) tip "
            f"towards the free surface.") from None
    if not np.isfinite(phi):
        raise ValueError(f"heel_deg {value!r} is not finite.")
    if abs(phi) >= HEEL_LIMIT_DEG:
        raise ValueError(
            f"heel_deg {phi} deg is at or past {HEEL_LIMIT_DEG} deg, where "
            f"the foil stands on edge: it carries none of the weight (the "
            f"trim target goes as 1/cos phi) and the free-surface image is "
            f"a mirror in a horizontal plane. Anything strictly inside is "
            f"accepted and priced, including angles no rider would hold.")
    return phi


def _clearance(value, where: str) -> float | None:
    """Validate a stated tip-immersion clearance [m] -> float or None."""
    if value is None:
        return None
    try:
        c = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"{where}: tip_clearance_m {value!r} is not a number; it is the "
            f"depth of water the shallowest part of the foil must keep, in "
            f"metres.") from None
    if not (np.isfinite(c) and c >= 0.0):
        raise ValueError(
            f"{where}: tip_clearance_m {value!r} must be a finite number >= "
            f"0 m. Zero is allowed and means the bare geometric requirement "
            f"(stay wet), which is enforced whether or not it is stated.")
    return c


def heeled_z(y, z=0.0, heel_deg: float = 0.0):
    """Height above the foil plane [m] of points ``(y, z)`` at heel ``phi``.

    A roll is a rotation about the craft's own x axis, so a point at span
    station ``y`` and height ``z`` ends up at ``z' = y sin(phi) + z cos(phi)``
    — which is the whole of the geometry, and the whole of why SPAN finally
    has a price. Every other term in this package is blind to how wide the
    foil is (``_mast_cd0`` reads depth and area, the section polars read
    ``alpha_eff``, and there is no structural weight model); ``y sin(phi)``
    is not. At heel the windward tip climbs towards the surface in direct
    proportion to the span, and the static head it loses is what the
    cavitation margin and :func:`immersion_margin` charge for.

    ``y`` is SIGNED and stays signed: one tip rises and the other dives, and
    collapsing that to ``|y|`` would make a heeled craft symmetric, which is
    exactly what it is not.

    WHAT HEEL DOES **NOT** DO HERE, said out loud because it is a limit and
    not an oversight. The roll enters the STATIC book — the head at each
    station, the immersion, the draught, and the ``1/cos`` on the lift — and
    it does NOT enter the induced-flow model: the free-surface image is
    still built at the foil's own depth (``build_image_operators``), i.e. at
    the craft's MEAN submergence, and the lattice still flies an upright
    planform. The image strength varies as the local ``1/(2h)``, so the
    error is second order in ``phi`` about a mean that heel does not move,
    while the terms this function feeds are FIRST order in it — which is why
    the static book is where the span finally gets a price and the induced
    book is not where it was missing one.

    Closing it is a named next step and a real one: the nonplanar families
    already carry an arbitrary ``(y, z)`` path and an image plane at
    ``z = depth``, so rolling the geometry itself is exactly representable
    there. The planar family cannot follow — its whole lifting-line
    treatment rests on the image being a pure TRANSLATION of a horizontal
    system — so doing it would split the two families' physics, which is a
    decision and not a fix.
    """
    phi = np.deg2rad(float(heel_deg))
    return np.asarray(y, dtype=float) * np.sin(phi) + np.asarray(
        z, dtype=float) * np.cos(phi)


def submergence(depth: float, y, z=0.0, heel_deg: float = 0.0):
    """Local submergence [m] below the free surface, heel included.

    ``depth - z'`` with ``z'`` from :func:`heeled_z`. Positive is wet;
    zero and below is a part of the foil in the air, which this package
    refuses per candidate rather than modelling (a partially emerged
    lifting surface is a different problem, not a worse design).
    """
    return float(depth) - heeled_z(y, z, heel_deg)


def heel_lift_factor(heel_deg: float = 0.0) -> float:
    """What a heeled foil must multiply its lift by: ``1 / cos(phi)``.

    The vertical force balance, and nothing more. A foil rolled by ``phi``
    points its lift ``phi`` off the vertical, so carrying a weight ``W``
    needs ``L = W / cos(phi)`` and the sideways component ``L sin(phi)``
    goes where a wind foiler needs it anyway — into the side force the rig
    is pulling against, which is why heel is not merely a cost.

    That last part is deliberately NOT credited back to the design. The
    lateral balance is longitudinal-solver-invisible in this package
    (``rig.py`` says so about ``side_n``), so the lift penalty is charged
    and the side force it produces is reported, not scored. A run that
    scored it would be claiming a lateral solve this engine has not got.
    """
    return float(1.0 / np.cos(np.deg2rad(float(heel_deg))))


def immersion_margin(depth: float, y, z=0.0, heel_deg: float = 0.0,
                     clearance_m: float = 0.0) -> dict:
    """How much water is left over the SHALLOWEST part of the foil.

        g_imm = min_over_panels(depth - z') - clearance_m

    Signed, like every other margin here: a constrained optimiser needs the
    size of the violation and not a boolean.

    WHY THIS IS A SEPARATE MARGIN AND NOT PART OF THE CAVITATION ONE. They
    fail differently and at different places. Cavitation is a PRESSURE
    condition — it can bite in deep water at high speed — and it is already
    charged per panel at the local head. Emergence is a GEOMETRIC one, and
    the cavitation number does not see it: at ``h = 0`` the static term
    ``rho g h`` merely vanishes and ``sigma_cav`` is still a comfortable
    ~1.4 at 12 m/s, so a foil breaking the surface reports a healthy
    cavitation margin right up to the moment it ventilates. The two numbers
    have to be two numbers.

    ``clearance_m`` is the tip immersion the craft is required to keep, and
    it defaults to ZERO — which is not an allowance at all but the bare
    geometric statement "stay in the water". A ventilation clearance is a
    real number (it scales with the local chord and with how hard the
    suction side is working) and it is the user's to state; defaulting one
    would put a limit nobody chose between the craft and its span.
    """
    h = submergence(depth, y, z, heel_deg)
    h = np.atleast_1d(np.asarray(h, dtype=float))
    yy = np.broadcast_to(np.asarray(y, dtype=float), h.shape)
    i = int(np.argmin(h))
    return {
        "g": float(h[i] - float(clearance_m)),
        "h_min_m": float(h[i]),
        "y_at_h_min_m": float(yy.flat[i]),
        "clearance_m": float(clearance_m),
        "h_y": h,
    }


class MissingCpMinError(ValueError):
    """A hydrofoil was given a section that cannot answer ``cp_min``.

    Its own class so a caller (the shells, ``api``) can catch exactly this
    and report a configuration problem rather than a solver crash.
    """


def _re_bank_tags() -> tuple:
    """The Reynolds nodes the shipped Cp_min bank covers, read from the bank
    itself (``polar.RE_BANK_TAGS``) rather than spelt into a message that
    then goes stale — which is exactly what happened when the bank grew from
    one node to five."""
    from .polar import RE_BANK_TAGS
    return tuple(t for _, t in RE_BANK_TAGS)


def require_cp_min(pol, where: str) -> None:
    """Refuse, LEGIBLY and at construction time, a section with no Cp_min.

    Every hydrofoil family evaluates the cavitation constraint on every
    evaluation -- ``cavitation_margin`` calls ``polar.cp_min`` unconditionally,
    because a foil that cavitates is not a design, it is a different flow. So
    a section that cannot answer it is not a run that degrades; it is a run
    that cannot start.

    It used to *start*, and then die inside the objective on the first
    evaluation with ``ValueError: polar '...' has no Cp_min table``, which
    surfaces from three frames below an optimiser as a crash rather than as a
    refusal. This project's own convention is that refusals happen at CONFIG
    time, not mid-search (cf. the three in ``api.wing_score_reference``), and
    a mid-search raise also throws away every XFOIL sweep already paid for
    (memory: ``stop-is-a-stop-rule``).

    BOTH AXES ARE COVERED NOW, and the second one only just. The THICKNESS
    axis was always complete: all five NACA 24XX members ship a ``.cpmin``
    companion and ``BlendedPolar`` blends it, so every ``polar_family.at(tc)``
    answers. The REYNOLDS axis was not — only Re 1e6 shipped one, because
    ``scripts/gen_cpmin_family.py`` had its Reynolds number as a module
    CONSTANT and could regenerate no other node — so ``polar.polar_at_re``
    returned a table with no Cp_min at three of its four bank nodes and at
    every blend between them. That is what kept the water families off their
    own Reynolds number: the drag half was available all along
    (``carwing``/``carwing_multi`` are declared consumers), and the
    CAVITATION half had nothing to read. All twenty (t/c, Re) cells carry a
    Cp_min table now, and ``_check_flown_reynolds`` asserts it at config time
    so this refusal cannot come back as a mid-search surprise.

    WHY THE BANK IS NOT SIMPLY GENERATED. Measured, pre-registered as S47-1
    (``PREREG_SESSION47_REGISTRY.md``,
    ``figures/cpmin_re_probe_s47.json``): between
    Re 3e5 and Re 1e6 on NACA 2412, Cp_min moves by up to **0.147** over the
    cruise window alpha -2..+8 deg and **1.355** by alpha 14 deg. Since the
    margin is ``g = sigma_cav + cp_min``, that is a 1:1 error in a
    FEASIBILITY GATE. The sign also has two regimes -- the low-Re peak is
    slightly deeper below alpha 4 deg and markedly BLUNTER above it -- so the
    missing tables cannot be faked from the 1e6 table with a correction
    factor. They are real XFOIL work, and shipping a guess in their place
    would be the "a calibration became a rule" failure this project has
    recorded twice.
    """
    try:
        pol.cp_min(0.0)
    except MissingCpMinError:
        raise
    except (ValueError, TypeError, AttributeError) as exc:
        name = getattr(pol, "name", type(pol).__name__)
        re_num = getattr(pol, "Re", None)
        at_re = f" at Re {re_num:.3g}" if isinstance(re_num, (int, float)) else ""
        raise MissingCpMinError(
            f"{where}: the section '{name}'{at_re} carries no Cp_min table, "
            f"and a hydrofoil cannot be scored without one -- the cavitation "
            f"constraint g = sigma_cav + Cp_min is evaluated on EVERY "
            f"evaluation, so this is refused here rather than raised inside "
            f"the objective.\n"
            f"  Underlying: {exc}\n"
            f"  Why: the Cp_min bank covers the THICKNESS axis and the "
            f"REYNOLDS nodes "
            f"{_re_bank_tags()} "
            f"(data/airfoils/naca24??_re*.cpmin) and nothing else. A polar "
            f"lands here when it is a section supplied WITHOUT a .cpmin "
            f"companion -- a designed shape, or a .dat loaded on its own -- "
            f"or when it is off the ends of that bank.\n"
            f"  Two ways out: (a) fly a section the bank carries; or (b) "
            f"extend it -- scripts/gen_cpmin_family.py takes a Reynolds tag "
            f"and costs about 2 s per member. Do NOT substitute a table from "
            f"another Reynolds number: it is wrong by up to 0.147 over "
            f"alpha -2..+8 deg and 1.355 by alpha 14 deg, in a feasibility "
            f"gate, and the sign flips at about alpha 4 deg "
            f"(measured on NACA 2412 between Re 3e5 and 1e6)."
        ) from exc


def _check_cp_min_available(prob, where: str) -> None:
    """Run :func:`require_cp_min` over the sections ``prob`` can actually fly.

    A designed section is one polar. A family is a whole thickness axis, and
    the row that selects it is a DESIGN VARIABLE -- so checking one thickness
    would pass a family that fails at the other end of its own box. Both ends
    of ``geometry.TC_BOUNDS`` are checked, which is where any missing member
    shows up first.

    Deliberately NOT behind an opt-out flag. It changes no published number:
    the whole shipped NACA 24XX thickness axis carries ``.cpmin`` and
    ``BlendedPolar`` blends it, so every family this repo ships passes and
    every frozen hydrofoil result is bit-for-bit what it was (pinned by
    ``tests/test_hydrofoil_cpmin_refusal.py``). The only configurations it
    turns away are ones that would have crashed a few milliseconds later.
    """
    sec = getattr(prob, "section_polar", None)
    if sec is not None:
        # NEVER-FLOWN SENTINELS ARE NOT THIS CHECK'S BUSINESS, and there is
        # more than one of them: hydrofoil_section._PlaceholderPolar (which
        # describes the design vector while the real section is swept per
        # evaluation, with_cpmin=True) and api's "this problem flies the
        # section you chose, and none was given" guard. Both raise on
        # ATTRIBUTE ACCESS by design, so even probing a marker on them throws
        # — and both already refuse legibly on their own terms. Anything that
        # will not answer a plain getattr is therefore left alone rather than
        # reported as a missing Cp_min table, which it is not.
        try:
            if bool(getattr(sec, "is_placeholder", False)):
                return
        except Exception:
            return
        require_cp_min(sec, f"{where}.section_polar")
        return
    fam = getattr(prob, "polar_family", None)
    if fam is None:
        return
    lo, hi = geometry.TC_BOUNDS
    for tc in (float(lo), float(hi)):
        try:
            pol = fam.at(tc)
        except Exception:            # a family that cannot build is not this
            continue                 # check's business -- let it fail its way
        require_cp_min(pol, f"{where}.polar_family at t/c = {tc:g}")


def _check_draught_cap(prob, where: str) -> None:
    """Refuse an unsatisfiable draught cap at CONFIG time, not by searching.

    A cap below the shallowest depth the family can fly cannot be met by ANY
    design in the box — the foil alone draws ``DEPTH_BOUNDS[0]`` before the
    tip device adds a millimetre. That is a configuration error, and a search
    that returns "no feasible design found" after paying for a full budget is
    a much worse way to learn it (same convention as ``require_cp_min``).

    The cap is NOT clamped into range: refusing a number the user typed is
    right here because there is no nearby value that means what they asked
    for (cf. ``a-calibration-is-a-default-not-a-ban`` — this is not a measured
    band being enforced as a gate, it is an arithmetic impossibility).
    """
    cap = getattr(prob, "draught_max_m", None)
    if cap is None:
        return
    d = float(cap)
    if not (d > 0.0):
        raise ValueError(f"{where}: draught_max_m must be positive, got {cap}")
    # The floor is the family's MINIMUM ACHIEVABLE draught, not merely its
    # shallowest depth. Those differ the moment anything else sits below the
    # foil: the elevator families carry a stabiliser at -DZ_FRAC*b, so their
    # floor is 0.21 m where DEPTH_BOUNDS[0] is 0.15 m, and every cap in
    # [0.15, 0.21) used to be accepted and then found infeasible only by
    # searching. (Adversarial review of this change; the original guard
    # compared against DEPTH_BOUNDS[0] for every family.)
    lo = float(prob.min_draught_m)
    if d < lo:
        raise ValueError(
            f"{where}: draught_max_m {d} m is below the shallowest draught "
            f"this family can reach ({lo:.4g} m), so no design in its box can "
            f"satisfy it -- the assembly draws {lo:.4g} m at its shallowest "
            f"before any tip device cants down. Raise the cap, or lower the "
            f"family's depth band.")


def _operating_row(band, label: str, unit: str, why_positive: str) -> tuple:
    """One OPERATING-POINT design-box row ``(lo, hi)``, validated.

    The shared body of :func:`depth_row` and :func:`speed_row`, which is
    where the prose explaining each row lives; this is only the arithmetic
    the two have in common, kept in one place so the two refusals read the
    same and cannot drift apart.

    What is checked is exactly what the samplers and the solve genuinely
    need, and nothing else: a PAIR, two FINITE numbers, non-collapsed, low
    end strictly positive, low end first. There is deliberately NO ceiling
    and no floor above zero. The published bands are where the water
    families were MEASURED, and this repo's rule is that a calibration is a
    default and never a ban (``tail.arm_row`` says the same sentence about
    the tail arm) — so a band wider than the published one is the user's to
    ask for, and the cavitation margin, the draught cap and the Froude
    diagnostics are what price it, at the design where it actually fails.

    An infinite or NaN end is refused before anything else is read off the
    pair, and it is a DECISION rather than an oversight: ``inf`` is not a
    band a Sobol sequence can be scaled into and NaN silently poisons every
    comparison below it (``lo > 0.0`` is False for NaN, so without this the
    user would be told their band "starts at or below zero" when what they
    actually handed over was not a number at all).

    WHY THE COLLAPSED ROW IS DIAGNOSED FIRST, ahead of the positivity test.
    ``(0.0, 0.0)`` is both non-positive and zero-width, and the two messages
    send the user to different places: "starts at or below zero" invites
    them to raise the low end, which leaves the row still collapsed and
    still wrong. The mistake that actually happened is that they wrote a PIN
    as a bound, and this repo's recorded rule is that a pin is
    ``RunConfig.pinned`` and never a collapsed bound — so a row that
    collapsed is reported as a pin WHEREVER it collapsed, at zero, below
    zero or at 12 m/s, and only a genuinely two-ended row goes on to be
    asked whether its low end is positive.
    """
    try:
        lo, hi = float(band[0]), float(band[1])
    except (TypeError, IndexError, ValueError, KeyError):
        # KeyError is in the list because a MAPPING is the shape a caller
        # reaches for when they half-remember the interface — {"lo": .., "hi":
        # ..} indexes as band[0] and raises KeyError, not IndexError, and
        # without it the one sentence written for this mistake never reaches
        # the person making it.
        raise ValueError(
            f"the {label} design-box row must be a pair (lo, hi) in {unit}; "
            f"got {band!r}") from None
    if not (np.isfinite(lo) and np.isfinite(hi)):
        raise ValueError(
            f"the {label} design-box row ({lo}, {hi}) {unit} is not finite")
    if hi == lo:
        raise ValueError(
            f"the {label} design-box row ({lo}, {hi}) {unit} has zero width. "
            f"That is a PIN written as a bound, and this repo does pins the "
            f"one honest way: RunConfig.pinned takes {label} OUT of the "
            f"design vector, where a collapsed box instead breaks the Sobol "
            f"engine and silently degrades BO to random draws on every "
            f"iteration while still calling itself BO (api._apply_overrides "
            f"refuses the same thing for the same reason).")
    if not (lo > 0.0):
        # NOT relaxed to an epsilon, unlike the SIZE rows (`sizing.MIN_POSITIVE`
        # and `api._apply_overrides`). "As close to zero as this allows" is a
        # question the geometry answers honestly — a vanishing tail area comes
        # back `untrimmable: |i_t| exceeds limit`, a vanishing span comes back
        # with an aspect ratio outside its band. These two rows do not.
        # Measured at 1e-9: `V_ms` returns `solver failure: untrimmable:
        # alpha = 9122722` — the solve blowing up, reported as if it were a
        # finding — and `depth_m` returns **feasible**, a foil flying happily
        # at the free surface it would ventilate through. A refusal that says
        # why beats an answer that is wrong.
        raise ValueError(
            f"the {label} design-box row ({lo}, {hi}) {unit} starts at or "
            f"below zero. {why_positive} No upper bound is imposed — this is "
            f"a refusal of a number the solve cannot mean, not a cap on the "
            f"craft you are designing.")
    if hi < lo:
        raise ValueError(
            f"the {label} design-box row ({lo}, {hi}) {unit} is INVERTED: a "
            f"box row is (lo, hi), low end first, and read the other way "
            f"round it is an empty interval no sampler can draw from. Swap "
            f"the two numbers.")
    return (lo, hi)


def depth_row(band) -> tuple:
    """The ``depth_m`` design-box row [m] — the USER's band where one is set.

    The SEVENTH row to learn what six already know (``tail.arm_row``,
    ``tail.height_row``, ``sizing.span_bounds``): a band a shell widens has
    to reach the PROBLEM, not merely the sampler it draws from. Until it
    did, ``DEPTH_BOUNDS`` was a bare class attribute with no annotation — so
    it was not a dataclass field and no builder keyword could set it — and
    ``api._apply_overrides`` moved ``built.bounds`` while ``prob.bounds``
    stayed at (0.15, 1.0) m. Every draw outside that came straight back as
    ``"bounds violation"`` from ``fg_hydrofoil``, and ``RunConfig.pinned``
    refused a pin against a box the run was no longer searching. It is the
    identical failure ``arm_row`` was written for, one row along.

    WHY THE PUBLISHED BAND IS ONLY A DEFAULT. 0.15 m is the shallow end of a
    foiling dinghy's ride height, and the free-surface image this module is
    built on is STRONGEST there — take-off, touchdown and a marginal-lull
    ride all happen with the foil closer to the surface than the calibration
    allows, and those are the conditions a small-craft study is about.
    Nothing in the solve changes at 0.10 m: the image system is a function
    of depth throughout and the ventilation/cavitation margins are what
    price the shallow end, which is the honest place for it to be priced.
    (The deep end is a mast length, not a physics limit —
    ``LITERATURE_REVIEW_FOILINGBO.md`` records that the 1.0 m end is itself
    unreachable on a 0.85-0.95 m windfoil mast, so the published band is
    neither a floor nor a ceiling for any real craft.)
    """
    return _operating_row(
        band, "depth_m", "m",
        "Submergence is measured from the free surface down to the foil, so "
        "zero puts the foil IN the surface and the negative image coincides "
        "with the real one — the influence the whole family is built on is "
        "singular there, and a negative depth asks for a foil in the air.")


#: default SPAN band, as multiples of the family's own published span —
#: used when a caller asks for a searched span without saying how wide.
#: Half to double is the widest bracket whose corners stay mostly inside
#: ``sizing.AR_LIMITS`` against the matching area band; what falls outside
#: is refused per candidate, not banned, because a band is a question and
#: a validity limit is a property of one design.
SIZE_BAND_FRAC = (0.5, 2.0)


def _size_band(band, nominal: float, row) -> tuple | None:
    """A stated size band, validated — or the default bracket around
    ``nominal`` when the caller asked for the row but not for its width.

    ``True`` is how a shell says "search this, you choose the band"; a pair
    is the band itself; ``None`` is the published fixed size.
    """
    if band is None or band is False:
        return None
    if band is True:
        lo, hi = SIZE_BAND_FRAC
        return row((lo * float(nominal), hi * float(nominal)))
    return row(band)


#: WHAT A SEARCHED SPAN COSTS WHEN NOTHING PRICES IT. One sentence, in one
#: place, because three families and two shells have to say the same thing
#: (``api._wet_size_kwargs``, ``gui.v3.session``).
#:
#: THIS USED TO BE A REFUSAL, and the refusal was the wrong shape. It raised
#: whenever a craft with a vertical surface asked for the span row at zero
#: heel, which is the DEFAULT state of every water family this package ships
#: — so "search the span and the area" was unreachable on the hydrofoil
#: without first turning the mast off (a lie about the craft: a foiler hangs
#: off its strut) or typing a heel angle nobody had asked for. A bound that
#: refuses the default configuration is a ban, and this package's own rule is
#: that a measurement is a DEFAULT and not a ban: the honest answer is to fly
#: the row and say what it will do, which is what this sentence is for.
#:
#: What it will do is measured, not feared: at zero heel the answer rides the
#: TOP of whatever band it is given and stops on the aspect-ratio limit the
#: lifting line is honest to, because nothing on the craft grows with the
#: span. That makes ``b_m`` a bound rather than an optimum — worth having, and
#: worth reading as "as wide as I allowed" rather than "as wide as it should
#: be". :func:`span_pricing_note` is what puts it in front of whoever asked.
SPAN_IS_UNPRICED_AT_ZERO_HEEL = (
    "a searched span has nothing to trade against AT ZERO HEEL, so expect "
    "the answer at the TOP of the band you gave it, stopped by a LIMIT and "
    "not by an optimum. Nothing on a foil flown flat reads how WIDE it is: "
    "every station sits at one depth, there is no structural weight in this "
    "package, and where the craft carries a mast that mast's size is the "
    "submergence depth times its chord, so its drag (hydrofoil._mast_cd0) "
    "reads the depth and the reference AREA and never the span. Taking the "
    "mast off does not price it either — it makes the ratchet STEEPER. "
    "MEASURED on the shipped foil (S = 0.144 m^2, depth 0.575 m, V 12 m/s), "
    "L/D climbs monotonically 20.02 -> 46.33 from b = 0.9 m to 2.4 m with "
    "the mast on and 22.01 -> 58.63 with it off, and the next station "
    "(2.5 m) is refused at aspect ratio 43.4, outside the (3, 40) band "
    "these solvers are honest over. Read the answer as 'as wide as I "
    "allowed' and argue about the BAND — the beam, the foil case or the "
    "class rule that sets it. On that planform the default span band's top, "
    "2.4 m (SIZE_BAND_FRAC's 2.0 x 1.2 m), is aspect ratio 40 to within "
    "floating point, so the band and the ceiling land on the same number; "
    "state a different size and they part, because the band scales with "
    "your span while the ceiling reads b^2/S. HEEL is the one term that "
    "reads the span (hydrofoil.heeled_z: the rising tip climbs "
    "(b/2) sin(phi) towards the surface, and hydrofoil.immersion_margin "
    "refuses the design that leaves it) — but it takes a REAL angle: the "
    "aspect-ratio ceiling binds first until about 29 deg, where the 2.4 m "
    "top of the band first runs out of immersion "
    "(asin(2 x 0.575 / 2.4) = 28.63 deg). Below that the span is stopped by "
    "the solvers' validity and not by the water.")

#: the name this sentence carried while it was a refusal. Kept as an alias
#: because it is cited from ``api``, ``hydrotail`` and both shells, and a
#: rename that lands in four files under a concurrent writer is a rename that
#: half-lands. The behaviour it named is gone; the reading is not.
SPAN_NEEDS_NO_FIN = SPAN_IS_UNPRICED_AT_ZERO_HEEL


def heel_prices_span(heel_deg) -> bool:
    """Does THIS heel angle price a span? See :func:`span_is_priced`.

    Split out so the shells can ask the question before there is a problem
    to ask it of — a size menu decides which entries exist from stage 1's
    answers, and a menu with its own copy of the rule is a menu free to
    disagree with the builder about what is offered.
    """
    return heel_angle(heel_deg) != 0.0


def span_is_priced(prob) -> bool:
    """Is there anything in THIS problem that reads how WIDE the foil is?

    The one question behind :data:`SPAN_IS_UNPRICED_AT_ZERO_HEEL`, asked in
    one place so the engine's caution and the shells' menus cannot disagree
    about it. It no longer decides whether the row EXISTS — every water
    family may search its span — only whether the answer is an optimum or a
    bound (:func:`span_pricing_note`).

    Only heel answers it. A foil flown flat sits at one depth from root to
    tip, so at fixed area a wider one is purely a higher aspect ratio and
    purely cheaper: there is no structural weight in this package, and the
    mast is sized by DEPTH x chord, so nothing on the craft grows with the
    span. Heeled, the rising tip climbs ``(b/2) sin(phi)`` towards the free
    surface (:func:`heeled_z`) — the span buys static head at one tip and
    spends it at the other, and :func:`immersion_margin` refuses the design
    that runs out. That is a real two-sided trade, and it is the ONLY one.

    A stated ``tip_clearance_m`` is deliberately NOT counted. It tightens
    the same margin rather than creating it, so at zero heel on a planar
    foil it still reads one depth for every station and prices nothing.
    """
    return heel_prices_span(getattr(prob, "heel_deg", 0.0))


def span_pricing_note(prob) -> str | None:
    """What has to be said about THIS problem's searched span, or ``None``.

    The one author of the caution, so a shell, a script and a report cannot
    each invent their own wording for it. Non-``None`` exactly when the span
    row is being searched and nothing on the craft reads it — which is a
    statement about the ANSWER (a bound, not an optimum), not a reason to
    refuse the question. Silent where the span is fixed, and silent where a
    heel angle prices it.
    """
    if getattr(prob, "span_bounds_m", None) is None:
        return None
    if span_is_priced(prob):
        return None
    return SPAN_IS_UNPRICED_AT_ZERO_HEEL


def _size_check(prob) -> None:
    """Validate a problem's size bands in place. Call from ``__post_init__``.

    THE ONE PLACE the size rows' bands are resolved, because all three water
    problems call it and a rule restated per family is a rule that drifts.

    IT USED TO REFUSE, and what it refused was the default: a searched span
    beside a vertical surface at zero heel. Every water family ships with the
    mast on and the craft flown flat, so the row a foil designer starts from
    — "how wide should this foil be?" — raised before it could be asked. The
    reason behind it is real and is kept (:func:`span_is_priced`); what is
    gone is the ban. The row flies, and :func:`span_pricing_note` says what
    the answer will be worth. Nothing published moves: the combination this
    used to refuse could not appear in any recorded run, because it raised.

    THERE USED TO BE A SECOND ONE — no searched span beside a searched
    stabiliser DEPTH, because that row's band is ``Z_T_FRAC_BOUNDS`` times
    the span. It is gone, and what it was really about was the depth row's
    UNITS rather than the pair: with both free the row is asked as a
    FRACTION of the candidate's own span (``hydrotail.Z_T_FRAC_LABEL``), so
    nothing moves anything else's bounds and the freedom is real. It mattered
    because the family it refused — a hydrofoil with an elevator whose depth
    is designed — is the one the shells open water on, so "search the span"
    was unreachable there without changing family first.
    """
    prob.span_bounds_m = _size_band(prob.span_bounds_m, prob.b, span_row)
    prob.area_bounds_m2 = _size_band(prob.area_bounds_m2, prob.S, area_row)


def size_labels(prob) -> tuple:
    """The SIZE rows this problem searches, in design-vector order."""
    return ((("b_m",) if prob.span_bounds_m is not None else ())
            + (("S_m2",) if prob.area_bounds_m2 is not None else ()))


def size_rows(prob) -> list:
    """The bounds those rows carry, in the same order."""
    return [r for r in (prob.span_bounds_m, prob.area_bounds_m2)
            if r is not None]


def size_from_x(prob, x) -> tuple:
    """``(b, S)`` for one candidate: the searched rows, else the fixed size.

    Read by LABEL and not by index. The size rows sit after the operating
    point and before the chord coefficients, and both of those blocks are
    already optional — an index arithmetic here would be a fourth place that
    has to agree with the other three about how long the vector is.
    """
    lab = prob.param_labels
    xs = np.asarray(x, dtype=float)
    b = float(xs[lab.index("b_m")]) if "b_m" in lab else float(prob.b)
    S = float(xs[lab.index("S_m2")]) if "S_m2" in lab else float(prob.S)
    return b, S


def size_refusal(prob, b: float, S: float) -> str | None:
    """Why this candidate's SIZE is not describable, or None.

    Only where the size is SEARCHED. A fixed planform was validated once,
    at build (``api._planform_size``), and gating it again here would let a
    published run start refusing itself; a searched one reaches corners
    nobody typed, so it is checked per candidate — which is where a validity
    limit belongs, because it is a property of one design and not of the
    band the user asked for.
    """
    from . import sizing

    if not size_labels(prob):
        return sizing.check_ar_limit(b, S, prob.ar_limits)
    return sizing.check_ar(b, S, prob.ar_limits)


def span_row(band) -> tuple:
    '''The ``b_m`` design-box row [m] — the SPAN, where the user searches it.

    The NINTH row to learn what eight already know (:func:`depth_row` tells
    the story), and the first of the two that make a water craft's SIZE a
    design variable instead of a calibration.

    WHY THE SIZE IS OPTIONAL AND EVERY OTHER ROW IS NOT. The published water
    families fly ``b = 1.2 m, S = 0.144 m2`` — a real dinghy foil, the
    geometry the cavitation margins, the Froude band and the mast model were
    all calibrated on. Making that a design variable unconditionally would
    move every water number this package has published, so a band ABSENT is
    the fixed published planform, bit-for-bit, and a band PRESENT is what
    puts ``b_m`` in the design vector. The switch is the band: there is no
    second control that has to agree with it.

    WHAT PAYS FOR SPAN, and what does not. The foil carries a FIXED design
    lift, so a wider foil at the same area is straight induced-drag relief
    with nothing to trade against it — no structural weight model, and
    ``_mast_cd0`` does not read the span. Expect the answer at the top of
    the band (this repo has the same finding recorded for a searched wing
    loading). That is not a bug in the search: it is the statement that on
    this physics the band IS the design decision, and the number to argue
    about is the beam, the foil case or the class rule that sets it.
    '''
    return _operating_row(
        band, "b_m", "m",
        "A span is a length: zero span is no foil at all, and the lifting "
        "line divides by it.")


def area_row(band) -> tuple:
    '''The ``S_m2`` design-box row [m2] — the AREA, where the user searches it.

    The twin of :func:`span_row` in every respect, and the row that makes
    the trim target a design variable too: the foil carries a fixed lift, so
    ``CL_target = L/(q S)`` and the area is the loading. It is the one size
    row with a genuine two-sided trade in this physics — a bigger foil trims
    at a lower CL (cavitation margin up, the suction peak down) and pays
    wetted area and, through the fixed-Re polar, a section quoted further
    from where it was measured.
    '''
    return _operating_row(
        band, "S_m2", "m2",
        "An area is an area: at zero the trim CL is infinite and the foil "
        "has no surface to make lift on.")


def speed_row(band) -> tuple:
    """The ``V_ms`` design-box row [m/s] — the USER's band where one is set.

    The EIGHTH row, and the twin of :func:`depth_row` in every respect
    including the bug it exists to close (read that docstring first).

    WHY THIS ONE IS A BLOCKER AND NOT A CONVENIENCE. The published band is
    (8, 16) m/s, calibrated on a ~600 kg foiling dinghy. Every craft this
    engine has since been pointed at flies outside it at BOTH ends, and the
    numbers are in ``LITERATURE_REVIEW_FOILINGBO.md``: take-off is 4.73 m/s
    for a race windfoil and 3.75-4.13 m/s for a wingfoil, wingfoil cruise
    starts at 5 m/s, and a kitefoil's design speed range is quoted as
    5-18 m/s. So with the row frozen the engine could not evaluate a
    wingfoil AT ITS OWN CRUISE SPEED, could not reach a kitefoil's top
    speed, and could not be asked the take-off question at all — the
    question that, in these disciplines, is the one that sizes the wing.
    A band is a calibration; this one had become the shape of the craft.

    What still refuses a speed is the physics that should: the cavitation
    margin at the fast end (``cavitation_margin``), and the free-surface
    model's own validity at the slow end, which is a Froude-number statement
    about the design and not a number this function is entitled to guess.
    """
    return _operating_row(
        band, "V_ms", "m/s",
        "The dynamic pressure is 0.5*rho*V^2, so a zero speed makes "
        "CL_target = L/(q S) infinite and there is no trim to solve; a "
        "negative speed is the same flow backwards, which this steady "
        "solver does not model.")


def cavitation_margin(hr: HydrofoilResult, polar, V: float,
                      rho: float = RHO_WATER,
                      p_vap: float = P_VAP,
                      heel_deg: float = 0.0) -> dict:
    """Per-strip no-cavitation constraint for a solved hydrofoil.

    Each strip operates at its effective angle alpha_eff(y) (exactly the
    angle the profile-drag integration uses); the section data map it to
    the minimum surface pressure coefficient Cp_min(y). The scalar
    constraint takes the WORST (most cavitation-prone) station:

        g = min_y [ sigma_cav(h(y), V) + Cp_min(y) ]   >= 0.

    Returns {g, sigma_cav, cp_min_worst, y_worst}. g is SIGNED — negative
    means cavitation onset; constrained optimisers need to see the
    magnitude of the violation, so this never collapses to a penalty.

    THE STATIC HEAD IS PER STATION, and at zero heel that is a distinction
    without a difference: a flat foil is at one depth, every ``sigma_cav``
    is the same number, and the worst station is still the one with the
    lowest ``Cp_min`` — bit-for-bit the scalar this used to be. HEELED it is
    the whole point. ``h(y) = depth - y sin(phi)`` (:func:`submergence`), so
    the windward tip loses static head in proportion to the SPAN and this
    constraint becomes the first thing in the planar family that reads how
    wide the foil is. Before it, the only spanwise variation here was
    ``Cp_min(alpha_eff(y))``, which is a loading question and scale-free.
    """
    alpha_eff_deg = np.rad2deg(hr.foil.alpha_eff_y)
    cp_min_y = polar.cp_min(alpha_eff_deg)
    h_y = submergence(hr.depth, hr.foil.y, 0.0, heel_deg)
    sig_y = np.array([sigma_cav(h, V, rho, p_vap) for h in np.atleast_1d(h_y)])
    g_y = sig_y + cp_min_y
    i_worst = int(np.argmin(g_y))
    return {
        "g": float(g_y[i_worst]),
        "sigma_cav": float(sig_y[i_worst]),
        "sigma_cav_root": float(sigma_cav(hr.depth, V, rho, p_vap)),
        "cp_min_worst": float(cp_min_y[i_worst]),
        "y_worst": float(hr.foil.y[i_worst]),
        "depth_worst": float(np.atleast_1d(h_y)[i_worst]),
        "cp_min_y": cp_min_y,
    }


# ---------------------------------------------------------------- image operators


@dataclass
class ImageOperators:
    """Cached influence of the (same-sign) image system on the real foil.

    W  : (N, N)  image panel circulations -> w at the foil's collocation pts
    Wq : (N*n_gauss, N)  same, at the foil's drag-quadrature points
    """

    img_sys: VortexSystem
    W: np.ndarray
    Wq: np.ndarray
    n_gauss: int = 8


def build_image_operators(sys: VortexSystem, depth: float,
                          n_gauss: int = 8) -> ImageOperators:
    """Image system for a foil ``depth`` below the free surface.

    The free surface lies at z = sys.z + depth; the image is geometrically
    identical (the vortex system is planar-horizontal, so the mirror is a
    pure translation) at z = sys.z + 2*depth, with circulation +Gamma
    (same-sign, see module docstring).
    """
    if depth <= 0.0:
        raise ValueError(f"depth must be > 0 (got {depth})")
    img = VortexSystem(b=sys.b, N=sys.N, x=sys.x, z=sys.z + 2.0 * depth)
    pts = np.column_stack([np.full(sys.N, sys.x), sys.y_col,
                           np.full(sys.N, sys.z)])
    qpts, _ = _panel_quadrature(sys, n_gauss)
    return ImageOperators(
        img_sys=img,
        W=w_influence(pts, img),
        Wq=w_influence(qpts, img),
        n_gauss=n_gauss,
    )


# ---------------------------------------------------------------- coupled solve


@dataclass
class HydrofoilResult:
    foil: LLTResult           # per-foil result; alpha_eff_y includes eps_surf
    eps_surf: np.ndarray      # (N,) image-induced angle at the foil [rad]
    #                           (downwash: eps_surf < 0 for a lifting foil)
    CL: float
    CDi_self: float           # Fourier self-induced drag (isolated-wing term)
    CDi_surf: float           # image-induced drag, counted ONCE (real foil)
    CDi_total: float          # CDi_self + CDi_surf
    e_total: float            # CL^2 / (pi AR CDi_total) — surface-degraded e
    depth: float
    Sref: float


def solve_hydrofoil(
    surf: Surface,
    depth: float,
    V: float = 1.0,
    Sref: float | None = None,
    ops: ImageOperators | None = None,
) -> HydrofoilResult:
    """Solve the monoplane equation with the free-surface image folded in.

    With G = 2 b V sin(j theta) mapping Fourier coefficients to panel
    circulations, the image adds eps_surf = (1/V) W G A to the effective
    angle, so the single linear system is

        (M - (1/V) W G) A = alpha_geo - alpha_L0

    — the same structure as one off-diagonal tandem coupling block, with
    the "other surface" being the foil's own image (same-sign circulation).
    """
    n = surf.N
    a0 = np.broadcast_to(np.asarray(surf.alpha_L0, float), (n,))
    ag = np.broadcast_to(np.asarray(surf.alpha_geo, float), (n,))

    theta, y, sin_jt, nsin_jt, M = fourier_system(surf.b, surf.c, surf.a)
    G = 2.0 * surf.b * V * sin_jt                  # A -> panel Gamma

    if ops is None:
        ops = build_image_operators(surf.system, depth)

    K = M - (1.0 / V) * ops.W @ G
    A = np.linalg.solve(K, ag - a0)
    Gam = G @ A
    eps = (ops.W @ Gam) / V                        # image-induced angle

    c = np.asarray(surf.c, float)
    S = np.trapezoid(c, y)
    AR = surf.b**2 / S
    nvec = np.arange(1, n + 1)
    sum_nA2 = float(np.sum(nvec * A**2))
    alpha_i = nsin_jt @ A
    CL = float(np.pi * AR * A[0])
    CDi_self = float(np.pi * AR * sum_nA2)

    foil = LLTResult(
        CL=CL, CDi=CDi_self,
        e=float(A[0] ** 2 / sum_nA2) if sum_nA2 > 0 else np.nan,
        AR=float(AR), S=float(S), A=A, theta=theta, y=y, c=c,
        Gamma=Gam, Cl_y=2.0 * Gam / (V * c), alpha_i_y=alpha_i,
        alpha_eff_y=ag + eps - alpha_i,
    )

    Sref = Sref if Sref is not None else float(S)
    # image drag on the REAL foil only (the image carries no physical force):
    cdi_surf, _ = mutual_cdi(
        surf.system, ops.img_sys, Gam, Gam, V, Sref,
        n_gauss=ops.n_gauss, W_a=ops.Wq,
    )
    CL_ref = CL * S / Sref
    CDi_self_ref = CDi_self * S / Sref
    CDi_total = CDi_self_ref + cdi_surf
    e_total = float(CL_ref**2 / (np.pi * (surf.b**2 / Sref) * CDi_total)) \
        if CDi_total > 0 else np.nan

    return HydrofoilResult(
        foil=foil, eps_surf=eps, CL=CL_ref,
        CDi_self=CDi_self_ref, CDi_surf=float(cdi_surf),
        CDi_total=float(CDi_total), e_total=e_total,
        depth=float(depth), Sref=float(Sref),
    )


# ------------------------------------------- the section at its own Reynolds

@lru_cache(maxsize=1)
def _re_bank() -> dict:
    """The NACA 24XX Reynolds bank, read from disk ONCE.

    ``polar.polar_re_bank`` re-reads every ``.pol`` file on every call (7.6 ms
    measured), which is far more than a water evaluation costs. Cached here
    rather than in polar.py for the reason ``carwing._re_bank`` gives for
    caching it there: it is static data on disk and polar.py is shared.
    """
    from .polar import polar_re_bank

    return polar_re_bank()


def flown_reynolds_number(prob, mac: float, V: float) -> float:
    """``rho V mac / mu`` — the Reynolds number this candidate actually flies.

    The same expression the breakdown has always REPORTED as ``Re_mac``. It
    was reported and not used: the section came from ``polar_family.at(tc)``,
    which globs ``naca24??_re1e6.pol``, so every water design was looked up
    in the Re = 1e6 table whatever it was flying.
    """
    return float(prob.rho) * float(V) * float(mac) / float(prob.mu)


def polar_for(prob, tc: float, mac: float, V: float, polar=None):
    """The section table a candidate flies — at its own Re where asked for.

    ``polar`` short-circuits to a CHOSEN section (one table at one Reynolds
    number), which is why ``flown_reynolds`` is refused beside one at config
    time rather than silently ignored here.

    WHY THIS IS OPT-IN. Every published water number was flown on the Re-1e6
    tables, and the mismatch is not small: with the size rows open the box
    now spans Re_mac 2.28e5-7.29e6, and at the bottom of that the 1e6 table
    under-predicts section cd by 30-43 % (measured on the shipped bank,
    t/c 0.12, alpha 0-6 deg). Turning this on therefore MOVES every water
    number, which is a decision and not a fix — so the flag defaults False
    and an untouched run is bit-for-bit the published one.

    Raises ``ValueError`` (the in-contract callers catch it) when the flown
    Reynolds number is outside the bank. Not clamped and not extrapolated:
    an extrapolated viscous polar is a fabrication, and the whole point of
    this route is that the table is the one the design was measured on.
    """
    if polar is not None:
        return polar
    if not getattr(prob, "flown_reynolds", False):
        return prob.polar_family.at(tc)
    from .polar import polar_at_re

    return polar_at_re(float(tc), flown_reynolds_number(prob, mac, V),
                       bank=_re_bank())


def _check_flown_reynolds(prob, where: str) -> None:
    """Refuse a ``flown_reynolds`` this problem cannot honour, at CONFIG time.

    Two refusals, both of them the shape this package prefers — said once,
    before a budget is spent, rather than as a mid-search exception:

    * a CHOSEN section is one table at one Reynolds number, so "read it at
      the flown Re" has nothing to read. ``carwing`` refuses the same pair
      for the same reason;
    * the bank must carry a Cp_min companion at EVERY node, because the
      water families' constraint is ``g = sigma_cav + Cp_min`` and a node
      without one would refuse mid-search at whatever Reynolds number the
      search happened to reach. Until this session only Re 1e6 shipped one
      (``scripts/gen_cpmin_family.py`` was hardcoded to it); all four nodes
      carry one now, and this is what says so out loud if that regresses.
    """
    if not getattr(prob, "flown_reynolds", False):
        return
    if getattr(prob, "section_polar", None) is not None:
        raise ValueError(
            f"{where}: flown_reynolds needs the NACA 24XX Reynolds bank, and "
            f"this problem flies a CHOSEN section — one table, at one "
            f"Reynolds number. Fly the family's sections to read them at the "
            f"flown Re, or keep the chosen section and accept the Re it was "
            f"measured at (its own polar states it).")
    missing = [(tc, re) for tc, byre in sorted(_re_bank().items())
               for re, pol in sorted(byre.items())
               if not getattr(pol, "has_cp_min", False)]
    if missing:
        raise ValueError(
            f"{where}: flown_reynolds needs a Cp_min companion at every node "
            f"of the Reynolds bank (the cavitation margin is "
            f"sigma_cav + Cp_min), and {len(missing)} are missing, e.g. "
            f"t/c {missing[0][0]:.2f} at Re {missing[0][1]:.0e}. Run "
            f"scripts/gen_cpmin_family.py <xfoil> "
            f"{' '.join(t for _, t in _RE_TAGS())}.")


def _RE_TAGS() -> list:
    from .polar import RE_BANK_TAGS

    return list(RE_BANK_TAGS)


def _mast_cd0(depth: float, S: float, c_mast: float = 0.08,
              cf_mast: float = 0.004) -> float:
    """Reduced-order mast (strut) parasite drag, referenced to foil area S.

    Wetted area = 2 sides x depth x mast chord; flat-plate turbulent
    Cf ~ 0.004 at Re_mast ~ 1e6 (GDP drag-buildup analogue, no form factor
    — a documented reduced-order choice). Linear in depth: this is the
    practical penalty that opposes the free-surface image's "go deep"
    gradient and creates an interior depth optimum.
    """
    return cf_mast * (2.0 * depth * c_mast) / S


# --------------------------------------------------------------- THE STRUT
# ``_mast_cd0`` above is the PUBLISHED charge and stays exactly what it was:
# wetted area times a constant flat-plate Cf, no form factor, no junction and
# no side force. It is what every recorded water number was scored on.
#
# It is also, measurably, not a strut. FoilingBO states the gap on its own
# card (``gui/foiling/layout.py``, ``Mast``): the like-for-like under-charge
# against published strut data is 1.85-2.35x, and the largest missing term is
# not a friction correction at all — it is that NOTHING IN THE MODEL RESISTED
# THE RIG'S SIDE LOAD. ``rig.RigLoads.side_n`` is 274 N on the measured
# windfoil and 615 N on the measured kitefoil, and it was carried as a value
# that no surface flew. A windfoil's mast IS its daggerboard; a model in
# which it makes no side force has no upwind case and no leeway.
#
# The four functions below are that surface, priced. They are used where a
# family switches ``strut_model`` on and are BYPASSED where it is off, so the
# published numbers have one author and the new ones another.

#: Span efficiency of the strut working at leeway. A symmetric, untwisted,
#: constant-chord surface with one end at a free surface and the other in a
#: junction: 0.90 is the ordinary rectangular-planform value, taken as a
#: DEFAULT rather than a measurement, and stated so the number a leeway drag
#: rides is visible instead of buried.
STRUT_E = 0.90

#: Chordwise station of maximum thickness, as a fraction of chord — the
#: ``(x/c)_m`` :func:`drag.wing_form_factor` reads. 0.30 is the NACA
#: four-digit family's, and a strut section IS one: martinez2026 records a
#: real windfoil mast as a NACA 0009 "so that it can generate lift equally in
#: both directions" (quoted in ``rig.py``).
STRUT_XC_M = 0.30

#: THE LINEAR RANGE the side-force slope above is valid over [deg]. A strut
#: is a symmetric section at a small angle and :func:`strut_cl_alpha` is a
#: straight line: asked for enough side force it will happily return a
#: thirty-degree leeway and the induced drag of a lift the section cannot
#: make. 12 deg is a conservative reading of where a symmetric NACA
#: four-digit section at Re ~ 1e6 stops being linear, and it is a REFUSAL
#: rather than a clamp — a craft that cannot hold its rig's side load at a
#: speed is not a craft flying with extra drag, it is a craft sliding
#: sideways, and this package has no model of that.
STRUT_LINEAR_DEG = 12.0

#: How many corners the strut makes where it meets the craft. Two: the
#: fuselage runs fore and aft of it and the boundary layers meet on BOTH
#: sides, which is the same count ``junction.report`` uses for a wing with a
#: tip device on each side.
STRUT_N_JUNCTIONS = 2


def strut_aspect_ratio(depth: float, chord: float) -> float:
    """Effective aspect ratio of a SURFACE-PIERCING strut: ``depth / chord``.

    NOT ``2 * depth / chord``. A surface against a RIGID wall gets its own
    reflection and flies at twice its geometric aspect ratio — that is the
    ground-plane result, and it is the one a strut does not get. The upper
    end of a mast is the FREE SURFACE, where the boundary condition is p = 0
    rather than no-flow, so the image system carries the OPPOSITE sign and
    the reflection buys nothing: the strut behaves as a wing with a free end
    at the waterline (Hoerner, *Fluid-Dynamic Lift*, surface-piercing
    struts). It is the same high-Froude free-surface limit
    :func:`build_image_operators` builds for the horizontal foil, seen by a
    surface at ninety degrees to it.

    The FOIL end is closed by the foil itself, which acts as an end plate.
    That credit is not taken here — it would need the foil's own thickness
    and planform at the junction, and taking it would flatter the strut.
    """
    h, c = abs(float(depth)), float(chord)
    if not (h > 0.0 and c > 0.0):
        return 0.0
    return h / c


def strut_cl_alpha(AR: float) -> float:
    """Side-force curve slope [1/rad] of a strut of aspect ratio ``AR``.

    Helmbold: ``2 pi AR / (2 + sqrt(AR^2 + 4))`` — the low-aspect-ratio form
    that reduces to the Prandtl finite-wing answer at large AR (4.78 against
    4.80 per rad at AR = 7.2, the shipped foil's strut) and stays finite as
    AR -> 0, which a short deep-chord mast on a shallow craft can approach.
    """
    AR = float(AR)
    if AR <= 0.0:
        return 0.0
    return float(TWO_PI * AR / (2.0 + np.sqrt(AR * AR + 4.0)))


def strut_report(*, depth: float, chord: float, tc: float, S_ref: float,
                 V: float, rho: float, mu: float, side_n: float = 0.0,
                 e: float = STRUT_E, junction: bool = True,
                 x_qc: float = 0.0, x_cg: float = 0.0) -> dict:
    """The strut as a SURFACE: its wetted drag, its junction and its leeway.

    Every coefficient returned is referenced to ``S_ref`` (the foil area the
    rest of the drag book is on), so ``CD`` is addable to ``CDi`` and
    ``CDp`` exactly as ``cd0_mast`` was.

    Three terms where the published charge had one:

    ``cd0``
        wetted area x ``skin_friction_cf`` AT THE STRUT'S OWN REYNOLDS
        NUMBER x ``wing_form_factor`` at its own thickness. The published
        charge used a constant Cf = 0.004 and no form factor at all, which
        is why ``fin_tc`` — a section the shell asks for on stage 2.7 —
        moved no number in the whole package.
    ``CD_junction``
        the two corners where the strut meets the craft, on the SAME Hoerner
        correlation (``junction.junction_cd``) every other junction in this
        repo is charged on. Sharp: nothing here draws a fillet.
    ``CDi_side``
        THE ONE THAT MATTERS. In steady flight the strut carries the rig's
        side load, so it flies at a leeway angle and pays induced drag for
        it: ``Cy^2 / (pi AR e)``, on the strut's own area, converted to
        ``S_ref``. It scales as 1/V^4 at fixed side force, so it is a
        rounding error at top speed and the dominant strut term at
        take-off — which is the physics a wetted-area-only mast could not
        express.

    ``side_n`` of zero is a craft with no rig (or a rig that states no side
    force): the leeway terms are exactly zero and the answer is the
    parasite book alone.

    NO DERIVATIVES ARE RETURNED. ``Cn_beta`` and ``Cy_beta`` have ONE author
    in this package — the lattice ``flightmodel`` builds from the reported
    fin block — and a second, reduced-order copy here is how a surface comes
    to have two sizes. What this returns instead is the GEOMETRY that author
    reads: the quarter-chord station and its arm about the CG.
    """
    from .drag import skin_friction_cf, wing_form_factor

    h, c = abs(float(depth)), float(chord)
    S_ref = float(S_ref)
    S_mast = h * c
    out = {"S_m2": float(S_mast), "height_m": float(h), "chord_m": float(c),
           "tc": float(tc), "AR": float(strut_aspect_ratio(h, c)),
           "x_qc_m": float(x_qc), "arm_m": float(x_qc) - float(x_cg),
           "side_n": float(side_n)}
    if not (S_mast > 0.0 and S_ref > 0.0 and float(V) > 0.0):
        out.update({"Re": 0.0, "Cf": 0.0, "FF": 1.0, "cd0": 0.0,
                    "CD_junction": 0.0, "Cy": 0.0, "CL_alpha": 0.0,
                    "leeway_deg": 0.0, "CDi_side": 0.0, "CD": 0.0})
        return out

    Re = float(rho) * float(V) * c / float(mu)
    Cf = skin_friction_cf(Re)
    FF = wing_form_factor(float(tc), xc_m=STRUT_XC_M)
    cd0 = Cf * FF * (2.0 * S_mast) / S_ref

    CD_j = 0.0
    if junction and 0.0 < float(tc) < 1.0:
        from .junction import junction_cd
        CD_j = junction_cd(float(tc), c, S_ref, radius=0.0,
                           n_junctions=STRUT_N_JUNCTIONS)

    q = 0.5 * float(rho) * float(V) ** 2
    Cy = float(side_n) / (q * S_mast) if side_n else 0.0
    a_side = strut_cl_alpha(out["AR"])
    leeway = (Cy / a_side) if a_side > 0.0 else 0.0
    CDi_side = ((Cy * Cy / (np.pi * out["AR"] * float(e)) * S_mast / S_ref)
                if (Cy and out["AR"] > 0.0) else 0.0)

    leeway_deg = float(np.rad2deg(leeway))
    out.update({"Re": float(Re), "Cf": float(Cf), "FF": float(FF),
                "cd0": float(cd0), "CD_junction": float(CD_j),
                "Cy": float(Cy), "CL_alpha": float(a_side),
                "leeway_deg": leeway_deg,
                "CDi_side": float(CDi_side),
                # ...and whether the number above is inside the straight
                # line it was read off (:data:`STRUT_LINEAR_DEG`). Reported
                # rather than clamped: the caller decides what a strut that
                # cannot hold its side load means for the design, and
                # ``hydrotail`` refuses it through the penalty contract.
                "linear": bool(abs(leeway_deg) <= STRUT_LINEAR_DEG),
                "CD": float(cd0 + CD_j + CDi_side)})
    return out


def solve_hydrofoil_trim(
    b: float,
    c: np.ndarray,
    twist_y: np.ndarray,
    CL_target: float,
    depth: float,
    a: np.ndarray | float = TWO_PI,
    alpha_L0: np.ndarray | float = 0.0,
    V: float = 1.0,
    alpha_bracket_deg: tuple[float, float] = (-10.0, 15.0),
    ops: ImageOperators | None = None,
) -> tuple[float, HydrofoilResult]:
    """Trim root AoA so CL = CL_target (closed form — CL is affine in alpha).

    Mirrors the tandem trim trick: the coupled system is linear, alpha only
    enters the RHS, so two solves give the exact trim point.
    """
    twist_y = np.asarray(twist_y, dtype=float)
    N = np.asarray(c).size
    sys0 = VortexSystem(b=b, N=N)
    if ops is None:
        ops = build_image_operators(sys0, depth)

    def _solve(alpha: float) -> HydrofoilResult:
        return solve_hydrofoil(
            Surface(b=b, c=c, alpha_geo=alpha + twist_y, a=a, alpha_L0=alpha_L0),
            depth, V=V, ops=ops,
        )

    r0 = _solve(0.0)
    r1 = _solve(np.deg2rad(1.0))
    slope = (r1.CL - r0.CL) / np.deg2rad(1.0)
    if abs(slope) < 1e-9:
        raise ValueError("degenerate CL_alpha in hydrofoil trim")
    alpha = (CL_target - r0.CL) / slope
    lo, hi = np.deg2rad(alpha_bracket_deg[0]), np.deg2rad(alpha_bracket_deg[1])
    if not (lo <= alpha <= hi):
        raise ValueError(
            f"untrimmable: alpha = {np.rad2deg(alpha):.2f} deg outside bracket"
        )
    return float(alpha), _solve(alpha)


# ---------------------------------------------------------------- Tier C problem


@dataclass
class HydrofoilProblem:
    """Tier C flagship: cavitation-constrained hydrofoil under a free surface.

    Design vector (d = 6):

        x = [taper, twist_root_deg, twist_tip_deg, tc, depth_m, V_ms]

    The foil carries a FIXED design lift (boat weight share) at whatever
    speed the optimiser chooses, so the trim target couples speed to
    loading:  CL_target(V) = L_design / (1/2 rho V^2 S).  That coupling is
    what makes the cavitation constraint a genuine two-sided trade (the
    classic "cavitation bucket"): fast -> low CL but large dynamic pressure
    scales the zero-lift suction peak up against a shrinking sigma_cav
    (both ~ 1/V^2 in Cp units, with the static head rho g h breaking the
    tie); slow -> sigma_cav relaxes but the high-cl suction peak grows
    quadratically. Depth enters three ways: the free-surface image
    penalises shallow (CDi up, CL_alpha down), the static head rewards
    deep (sigma_cav up), and the mast wetted area charges deep linearly
    (_mast_cd0) -- the interior-optimum trade documented in report §11.

    Objective (MAXIMISE): L/D at trim = CL_target / CD,
        CD = CDi_total (self + image) + CDp (section polar at alpha_eff)
             + mast cd0.
    Constraint: g = sigma_cav(h, V) + min_y Cp_min(y) >= 0 (cavitation_margin).

    Contract (constrained harness, optimize/constrained.py):
    ``fg(x)`` returns (L/D, g) for every SOLVABLE design, including
    infeasible ones (true value + signed violation — constrained BO needs
    to see both); solver failures return exactly (PENALTY, G_FAIL) with a
    finite, clearly-infeasible margin.

    Scale: b = 1.2 m, S = 0.144 m^2 (AR = 10, c_mean = 0.12 m), seawater,
    L_design = 3.7 kN (~380 kg foiling dinghy at the mid-speed CL ~ 0.5).
    Re_mac = 0.9-1.8e6 across the V range against a section table at a
    fixed Re = 1e6 — which was a flagged limitation and is now a CHOICE:
    ``flown_reynolds`` reads the section at the Reynolds number the design
    actually flies (:func:`polar_for`). Off by default, because turning it
    on moves every published water number; with the size rows open as well
    the box spans Re_mac 2.3e5-7.3e6 and the two answers differ by -6.1 %
    to +6.1 % of L/D at its corners.
    Fn_h = V/sqrt(gh) >= 2.6 over the whole box (high-Fn image model valid,
    module docstring).
    """

    b: float = 1.2
    S: float = 0.144
    N: int = 40
    L_design: float = 6000.0      # [N]
    rho: float = RHO_WATER
    mu: float = MU_WATER
    p_vap: float = P_VAP          # water vapour pressure (sea vs fresh)
    n_twist_stations: int = 2     # linear twist law (root/tip), Tier A style
    c_mast: float = 0.08
    cf_mast: float = 0.004
    #: THE STRUT, ASKED. ``fin`` is the presence key every family answers
    #: through ``fin.has_fin``: True (the default) charges the mast exactly
    #: as every published water run was charged, and False is a craft with
    #: no vertical surface at all — no mast drag, nothing to choose a
    #: section for, and the honest zero yaw stiffness that goes with it.
    #: ``fin_tc`` is its SECTION'S thickness, chosen on stage 2.7 and
    #: reported, so the loft and the flight rebuild carry the shape that was
    #: picked. It does NOT enter ``_mast_cd0``, which charges wetted area
    #: with no form factor — so a thicker strut section moves no published
    #: number, and that is said here rather than implied.
    fin: bool = True
    fin_tc: float | None = None
    alpha_bracket_deg: tuple[float, float] = (-10.0, 15.0)
    section_polar: object = None   # a DESIGNED section
    #   (hydrofoil_section.py): the CST weights ARE the thickness, so the t/c
    #   row leaves the vector and this polar — which must carry a Cp_min
    #   table — drives the solve.
    #: READ THE SECTION AT THE FLOWN REYNOLDS NUMBER
    #: (``polar.polar_at_re``) instead of the family's Re = 1e6 tables. The
    #: speed has always been a design variable here and the breakdown has
    #: always reported ``Re_mac``; what did not happen was the table
    #: FOLLOWING it. Off by default, because turning it on moves every
    #: published water number — and it moves them by a lot: with the size
    #: rows open the box spans Re_mac 2.28e5-7.29e6, and at the bottom of
    #: that the Re-1e6 table under-predicts section cd by 30-43 %.
    #: Refused beside a chosen ``section_polar`` (one table, one Reynolds
    #: number) — ``_check_flown_reynolds`` says so at config time.
    flown_reynolds: bool = False
    polar_family: object = None   # lazily loaded NACA 24XX family
    chord_order: int = 0          # free chord law (geometry.py); 0 = straight
    #                               taper, no extra variables, bit-for-bit
    chord_max_frac: float = geometry.CHORD_COEFF_BOUND
    chord_law: str = geometry.DEFAULT_CHORD_LAW   # WHICH SHAPE the
    #   coefficients above describe (geometry.CHORD_LAWS). Changes what
    #   the design vector DRAWS, never how long it is.
    chord_limits: "geometry.ChordLimits | None" = None   # what the chord
    #   DISTRIBUTION has to obey, in metres and degrees (geometry.ChordLimits:
    #   a minimum chord, a maximum, a maximum local taper ANGLE, and which end
    #   is the wide one). None = unconstrained, which is every published run.
    #   Checked where the planform is built, so a violation is an in-contract
    #   failure (the penalty contract), never an exception at the optimiser.
    #: the aspect-ratio band the USER will accept b^2/S in, as
    #: ``(ar_min, ar_max)`` with either end None (api.AR_LIMIT_KEYS).
    #: A SECOND band beside ``sizing.AR_LIMITS``, never a replacement
    #: for it, and None — no limit at all — is every published run.
    ar_limits: "tuple | None" = None

    #: THE SIZE, WHERE THE USER SEARCHES IT. ``None`` — every published run —
    #: is the fixed planform above, bit-for-bit: no extra row, no extra
    #: variable, the same design vector every water study was run on. A band
    #: (or ``True`` for the default bracket, :data:`SIZE_BAND_FRAC`) puts
    #: ``b_m`` / ``S_m2`` in the design vector as ordinary design-box rows,
    #: appended after the operating point and before the chord coefficients
    #: so every existing index keeps its meaning.
    #:
    #: They are the ONLY rows here that can leave the solvers' honest
    #: aspect-ratio band (``sizing.AR_LIMITS``), because they are the only
    #: two that set it. A corner outside it is refused PER CANDIDATE with
    #: its reason, never banned at the band — the same rule the draught cap
    #: and the cavitation margin are priced by.
    span_bounds_m: tuple | None = None
    area_bounds_m2: tuple | None = None

    #: THE CRAFT'S ROLL, and the only term in this family that reads the
    #: SPAN. ``0.0`` — every published run — is a foil flown flat and every
    #: expression below reduces to the one it replaced, bit-for-bit. Set, it
    #: does three things: the trim target grows as ``1/cos(phi)``
    #: (:func:`heel_lift_factor`), the static head becomes a function of the
    #: span station (:func:`cavitation_margin`), and the rising tip has to
    #: stay in the water (:func:`immersion_margin`). The third is what turns
    #: a searched span from a ratchet into a question with an answer.
    #:
    #: NOT derived from ``rig.RigLoads.side_n``, and ``heel_angle`` says why:
    #: what balances a rig's roll couple is the rider, and this package has
    #: no rider.
    heel_deg: float = 0.0
    #: the tip immersion the craft must KEEP [m] — a ventilation clearance,
    #: stated by whoever knows the craft. ``None`` (every published run) adds
    #: no margin and no constraint; the bare geometric requirement that the
    #: foil stay in the water is enforced per candidate either way, because
    #: an emerged foil is not a worse design, it is a different problem.
    tip_clearance_m: float | None = None

    # ---- the OPERATING POINT's two design-box rows. FIELDS, not the bare
    # class attributes they used to be: the defaults below are exactly the
    # published bands (shallow end: image + cavitation active; fast end:
    # sigma_cav tightens), so an untouched problem is bit-for-bit the one
    # every water study was run on — but a shell that widens either row can
    # now say so to the PROBLEM as well as to the sampler, which is what
    # ``depth_row`` / ``speed_row`` exist for.
    DEPTH_BOUNDS: tuple[float, float] = (0.15, 1.0)     # [m]
    V_BOUNDS: tuple[float, float] = (8.0, 16.0)         # [m/s]

    def __post_init__(self):
        # first, because everything below reads them: ``min_draught_m`` is
        # DEPTH_BOUNDS[0] and the draught check refuses a cap against it
        self.DEPTH_BOUNDS = depth_row(self.DEPTH_BOUNDS)
        self.V_BOUNDS = speed_row(self.V_BOUNDS)
        if self.polar_family is None:
            from .polar import default_polar_family
            self.polar_family = default_polar_family()
        _check_cp_min_available(self, "HydrofoilProblem")
        _size_check(self)
        self.heel_deg = heel_angle(self.heel_deg)
        self.tip_clearance_m = _clearance(self.tip_clearance_m,
                                          "HydrofoilProblem")
        _check_flown_reynolds(self, "HydrofoilProblem")
        self._ops_cache: dict = {}

    @property
    def param_labels(self) -> tuple:
        labels = ["taper", "twist_root_deg", "twist_tip_deg"]
        if self.section_polar is None:
            labels.append("tc")
        labels += ["depth_m", "V_ms"]
        # ...then the SIZE, where it is searched. LAST but for the chord
        # coefficients, so every published index keeps its meaning.
        labels += list(size_labels(self))
        return tuple(labels) + geometry.chord_labels(self.chord_order,
                                                    self.chord_law)

    @property
    def bounds(self) -> np.ndarray:
        rows = [geometry.TAPER_BOUNDS,
                geometry.TWIST_ROOT_BOUNDS_DEG,
                geometry.TWIST_TIP_BOUNDS_DEG]
        if self.section_polar is None:
            rows.append(geometry.TC_BOUNDS)
        rows += [self.DEPTH_BOUNDS, self.V_BOUNDS] + size_rows(self)
        return geometry.with_chord_bounds(np.array(rows, dtype=float),
                                          self.chord_order,
                                          self.chord_max_frac,
                                          self.chord_law)

    @property
    def dim(self) -> int:
        return int(self.bounds.shape[0])

    def CL_target(self, V: float, S: float | None = None) -> float:
        """Trim CL for the fixed design lift at speed ``V`` on area ``S``.

        ``S`` defaults to the problem's own, so every published caller is
        unchanged; a run that SEARCHES the area passes the candidate's, and
        must — the area is the loading, and reading the nominal one here
        would trim the design at a lift it is not carrying.
        """
        area = float(self.S if S is None else S)
        return float(self.L_design * heel_lift_factor(self.heel_deg)
                     / (0.5 * self.rho * V**2 * area))

    @property
    def immersion_capped(self) -> bool:
        """Does this problem carry the IMMERSION margin as a constraint?

        Tied to the STATED clearance and not to the heel, deliberately, and
        for the reason ``draught_max_m`` is: the width of the margin vector
        is part of the harness contract, so it must hang off one opt-in
        field that a caller sets on purpose. A heeled run with no clearance
        stated still cannot fly an emerged foil — that is refused per
        candidate, in contract, with its reason — it simply does not report
        a second margin nobody asked for.
        """
        return self.tip_clearance_m is not None

    @property
    def n_constraints(self) -> int:
        return 2 if self.immersion_capped else 1

    @property
    def constraint_labels(self) -> tuple:
        return (("cavitation margin",)
                + ((IMMERSION_CONSTRAINT_LABEL,) if self.immersion_capped
                   else ()))

    def ops(self, depth: float, b: float | None = None) -> ImageOperators:
        """Image operators for a foil of span ``b`` at ``depth`` -> cached.

        The cache key is BOTH, not just the depth. It was the depth alone
        while the span was a constant, which is exactly the kind of cache
        that goes wrong silently the moment the constant stops being one:
        the second candidate of a searched span would have been handed the
        first one's image system and flown a wing it is not.
        """
        span = float(self.b if b is None else b)
        key = (round(float(depth), 12), round(span, 12))
        if key not in self._ops_cache:
            sys0 = VortexSystem(b=span, N=self.N)
            self._ops_cache[key] = build_image_operators(sys0, depth)
        return self._ops_cache[key]


PENALTY = -100.0     # objective.py contract (solver failure only)
G_FAIL = -1.0        # finite infeasible margin reported on solver failure


def evaluate_hydrofoil(x: np.ndarray, prob: HydrofoilProblem | None = None) -> dict:
    """Full Tier C evaluation with breakdown; never raises for in-contract
    failures. ``feasible`` refers to the SOLVER (PENALTY contract);
    the cavitation constraint is reported separately as signed ``g``."""
    from . import geometry
    from .objective import _fail

    prob = prob or HydrofoilProblem()
    x = np.asarray(x, dtype=float)
    bnds = prob.bounds
    if x.shape != (bnds.shape[0],):
        return _fail(f"design vector shape {x.shape} != ({bnds.shape[0]},)")
    if np.any(x < bnds[:, 0] - 1e-12) or np.any(x > bnds[:, 1] + 1e-12):
        return _fail("bounds violation")

    if prob.section_polar is None:
        taper, tw_root, tw_tip, tc, depth, V = (float(v) for v in x[:6])
    else:
        taper, tw_root, tw_tip, depth, V = (float(v) for v in x[:5])
        tc = float(getattr(prob.section_polar, "tc", 0.12))
    # THE SIZE THE CANDIDATE FLIES — its own rows where they are searched,
    # the problem's published planform where they are not.
    b_x, S_x = size_from_x(prob, x)
    why = size_refusal(prob, b_x, S_x)
    if why is not None:
        return _fail(f"size: {why}")
    try:
        wing = geometry.Wing(
            b=b_x, S=S_x, taper=taper, twist_root_deg=tw_root,
            twist_tip_deg=tw_tip, tc=tc,
            chord_limits=prob.chord_limits,
            chord_coeffs=geometry.chord_coeffs_from_x(x, prob.chord_order,
                                                      prob.chord_law))
    except ValueError as exc:
        return _fail(f"planform: {exc}")
    yv, c, twist_rad = wing.sample(prob.N)

    # THE SECTION THIS CANDIDATE FLIES — at its own Reynolds number where the
    # design asked for that (``polar_for``). The mac is the candidate's, so
    # with the size rows open the table follows the planform as well as the
    # speed; with ``flown_reynolds`` off it is the family's Re-1e6 table and
    # every published number is unchanged.
    try:
        pol = polar_for(prob, tc, wing.mac, V, prob.section_polar)
    except ValueError as exc:
        return _fail(f"polar family: {exc}")

    CLt = prob.CL_target(V, S_x)
    try:
        alpha, hr = solve_hydrofoil_trim(
            b_x, c, twist_rad, CL_target=CLt, depth=depth,
            a=pol.a_lin, alpha_L0=pol.alpha_L0, V=V,
            alpha_bracket_deg=prob.alpha_bracket_deg,
            ops=prob.ops(depth, b_x),
        )
    except (ValueError, np.linalg.LinAlgError) as exc:
        return _fail(f"solver failure: {exc}")

    alpha_eff_deg = np.rad2deg(hr.foil.alpha_eff_y)
    lo, hi = pol.alpha_valid
    if alpha_eff_deg.min() < lo or alpha_eff_deg.max() > hi:
        return _fail(
            "effective AoA outside polar validity (stall/extrapolation proxy)",
            alpha_eff_range=(float(alpha_eff_deg.min()), float(alpha_eff_deg.max())),
        )

    CDp = float(np.trapezoid(pol.cd(alpha_eff_deg) * c, yv) / hr.foil.S)
    # THE STRUT IS A SURFACE THE CRAFT CAN BE ASKED ABOUT. Charged exactly
    # as it always was where there is one (``fin.has_fin``, True unless the
    # design says otherwise), and zero where the design states none — the
    # same three-state contract every other family's vertical surface has,
    # so "no strut" cannot mean "a strut nobody pays for".
    cd0_mast = (_mast_cd0(depth, S_x, prob.c_mast, prob.cf_mast)
                if _fin.has_fin(prob) else 0.0)
    CD = hr.CDi_total + CDp + cd0_mast
    LoD = hr.CL / CD
    if not np.isfinite(LoD):
        return _fail("non-finite L/D")

    # THE FOIL HAS TO BE IN THE WATER. Checked at the geometric span
    # extremes rather than at the lifting line's collocation stations: the
    # tip is what emerges, and no panel midpoint is ever at +/- b/2.
    imm = immersion_margin(depth, np.array([-0.5 * b_x, 0.5 * b_x]),
                           0.0, prob.heel_deg,
                           prob.tip_clearance_m or 0.0)
    if imm["h_min_m"] <= 0.0:
        return _fail(
            f"tip emerged: at {prob.heel_deg:g} deg of heel a {b_x:.3g} m "
            f"span reaches {-imm['h_min_m']:.3g} m above a free surface "
            f"{depth:.3g} m over the foil. A partly emerged foil is a "
            f"different problem, not a worse design.")

    cav = cavitation_margin(hr, pol, V=V, rho=prob.rho,
                            p_vap=prob.p_vap, heel_deg=prob.heel_deg)

    g_out = ([float(cav["g"]), float(imm["g"])] if prob.immersion_capped
             else cav["g"])
    return {
        "feasible": True, "reason": "", "score": float(LoD), "LoD": float(LoD),
        "g": g_out, "g_cav": float(cav["g"]),
        "heel_deg": float(prob.heel_deg),
        "immersion_m": float(imm["h_min_m"]),
        "y_at_immersion_m": float(imm["y_at_h_min_m"]),
        "tip_clearance_m": prob.tip_clearance_m,
        "g_immersion": (float(imm["g"]) if prob.immersion_capped else None),
        "sigma_cav": cav["sigma_cav"],
        "cp_min_worst": cav["cp_min_worst"], "y_worst": cav["y_worst"],
        "CL": hr.CL, "CL_target": CLt,
        "CDi_self": hr.CDi_self, "CDi_surf": hr.CDi_surf,
        "CDi_total": hr.CDi_total, "CDp": CDp, "cd0_mast": cd0_mast,
        "CD": float(CD), "e_total": hr.e_total,
        "alpha_deg": float(np.rad2deg(alpha)),
        "depth": depth, "V": V, "Fn_h": froude_depth(V, depth),
        "Re_mac": float(prob.rho * V * wing.mac / prob.mu),
        "polar": getattr(pol, "name", "unknown"),
        # ...and the Reynolds number that table was MEASURED at, beside the
        # one the design is flying. Two numbers, because for every published
        # run they differ (Re_mac 0.9-1.8e6 against a table at 1e6) and a
        # card that showed only the flown one would imply a table that
        # followed it.
        "section_re": (float(pol.Re) if getattr(pol, "Re", None) is not None
                       else None),
        # the SIZE actually flown, and the Wing object it was flown as.
        # ``api.design_report`` reads ``wing`` first for its planform
        # scalars, so a searched span reaches the report, the loft, the
        # 3-D view and the flight rebuild instead of the nominal one.
        "b": float(b_x), "S": float(S_x), "AR": float(b_x * b_x / S_x),
        "wing": wing,
        "hydrofoil": hr,
    }


def fg_hydrofoil(x: np.ndarray, prob: HydrofoilProblem | None = None
                 ) -> tuple[float, float]:
    """Constrained-harness callable: (L/D, signed cavitation margin g).

    Solvable-but-cavitating designs return their TRUE L/D with g < 0;
    solver failures return exactly (PENALTY, G_FAIL).

    With a stated tip clearance the margin is the PAIR ``[g_cav, g_imm]``
    and the failure value widens with it — the winglet family's contract,
    followed here rather than restated, so a harness that can read one can
    read the other."""
    prob = prob or HydrofoilProblem()
    out = evaluate_hydrofoil(x, prob)
    n = prob.n_constraints
    if not out["feasible"]:
        return PENALTY, (G_FAIL if n == 1 else [G_FAIL] * n)
    g = out["g"]
    return float(out["score"]), (g if n > 1 else float(g))


# =====================================================================
# Nonplanar variant: hydrofoil + tip device (winglet) under the surface
# =====================================================================
#
# The planar solver above mirrors a horizontal vortex system, where the
# image is a pure TRANSLATION — which is why it can stay a lifting line.
# A canted tip device breaks that, so this variant runs the nonplanar VLM
# with an image plane (vlm.ImagePlane, free_surface = same-sign image).
#
# Two things change physically, and both are the point of the variant:
#   1. the tip device works against the surface image, not just the wing's
#      own trailing system;
#   2. CAVITATION becomes a per-panel question. Submergence is depth - z,
#      so a device canted UP sits in less static head and cavitates first,
#      while one canted DOWN buys margin. The planar problem cannot see
#      this because there the depth is a single number.
# Cant is therefore free over (-90, +90) deg here — the air winglet band
# (geometry.WINGLET_CANT_LIMITS_DEG) is about raked-tip span accounting and
# does not apply.


def cavitation_margin_panels(alpha_eff_deg, z, depth: float, polar,
                             V: float, rho: float = RHO_WATER,
                             p_vap: float = P_VAP, y=None,
                             heel_deg: float = 0.0) -> dict:
    """Per-PANEL no-cavitation margin for a nonplanar foil.

    Same constraint as :func:`cavitation_margin` — sigma_cav + Cp_min >= 0 at
    the worst station — but each panel carries its OWN submergence
    ``depth - z'`` (positive z is up, towards the surface), so the static-head
    term varies over the surface. Returns the worst (smallest) margin and
    where it occurs.

    ``y`` and ``heel_deg`` are the craft's ROLL, and they are optional for
    one reason only: every published run of this family is upright, and an
    absent ``y`` with zero heel gives ``z' = z`` exactly. Heeled, the tip
    device's own cant and the roll add: a device canted DOWN on the rising
    side buys back head the heel took away, which is a trade the planar
    family cannot express and this one now can.
    """
    alpha_eff_deg = np.asarray(alpha_eff_deg, dtype=float)
    z = np.asarray(z, dtype=float)
    h_local = (float(depth) - z if float(heel_deg) == 0.0
               else submergence(depth, np.zeros_like(z) if y is None
                                else np.asarray(y, dtype=float), z, heel_deg))
    if np.any(h_local <= 0.0):
        raise ValueError("panel at or above the free surface")
    sig = np.array([sigma_cav(h, V, rho, p_vap) for h in h_local])
    cp_min = polar.cp_min(alpha_eff_deg)
    g_y = sig + cp_min
    i = int(np.argmin(g_y))
    return {
        "g": float(g_y[i]),
        "sigma_cav": float(sig[i]),
        "sigma_cav_root": float(sigma_cav(depth, V, rho, p_vap)),
        "cp_min_worst": float(cp_min[i]),
        "depth_worst": float(h_local[i]),
        "g_y": g_y,
    }


@dataclass
class HydrofoilWingletProblem:
    """Tier C + tip device: cavitation-constrained foil WITH a winglet (8-D).

    Design vector::

        x = [taper, twist_root_deg, twist_tip_deg, tc, depth_m, V_ms,
             winglet_h_frac, winglet_cant_deg]

    the 6-D planar vector plus the tip device's arc-length fraction and cant.
    Cant is signed: +90 deg points the device straight UP (towards the
    surface), -90 deg straight DOWN. That sign is a real trade here —
    upward buys nonplanar span for less cavitation margin, downward the
    reverse — which is why the band is symmetric rather than the air
    problem's (60, 90).

    Same operating point, weight, mast model and objective as
    :class:`HydrofoilProblem`; the solver is the imaged VLM instead of the
    imaged lifting line, and the cavitation constraint is evaluated per
    panel at the LOCAL submergence.
    """

    b: float = 1.2
    S: float = 0.144
    N_vlm: int = 40               # main-foil panels (full span)
    n_winglet: int = 8            # panels per tip device
    L_design: float = 6000.0      # [N]
    rho: float = RHO_WATER
    mu: float = MU_WATER
    p_vap: float = P_VAP          # water vapour pressure (sea vs fresh)
    c_mast: float = 0.08
    cf_mast: float = 0.004
    #: THE STRUT, ASKED. ``fin`` is the presence key every family answers
    #: through ``fin.has_fin``: True (the default) charges the mast exactly
    #: as every published water run was charged, and False is a craft with
    #: no vertical surface at all — no mast drag, nothing to choose a
    #: section for, and the honest zero yaw stiffness that goes with it.
    #: ``fin_tc`` is its SECTION'S thickness, chosen on stage 2.7 and
    #: reported, so the loft and the flight rebuild carry the shape that was
    #: picked. It does NOT enter ``_mast_cd0``, which charges wetted area
    #: with no form factor — so a thicker strut section moves no published
    #: number, and that is said here rather than implied.
    fin: bool = True
    fin_tc: float | None = None
    alpha_bracket_deg: tuple[float, float] = (-10.0, 15.0)
    #: READ THE SECTION AT THE FLOWN REYNOLDS NUMBER
    #: (``polar.polar_at_re``) instead of the family's Re = 1e6 tables. The
    #: speed has always been a design variable here and the breakdown has
    #: always reported ``Re_mac``; what did not happen was the table
    #: FOLLOWING it. Off by default, because turning it on moves every
    #: published water number — and it moves them by a lot: with the size
    #: rows open the box spans Re_mac 2.28e5-7.29e6, and at the bottom of
    #: that the Re-1e6 table under-predicts section cd by 30-43 %.
    #: Refused beside a chosen ``section_polar`` (one table, one Reynolds
    #: number) — ``_check_flown_reynolds`` says so at config time.
    flown_reynolds: bool = False
    polar_family: object = None
    section_polar: object = None        # a DESIGNED section
    #   (hydrofoil_section.py): the CST weights ARE the thickness, so the t/c
    #   row leaves the vector and this polar — which must carry a Cp_min
    #   table — drives the solve.
    chord_order: int = 0                # free chord law (geometry.py)
    chord_max_frac: float = geometry.CHORD_COEFF_BOUND
    chord_law: str = geometry.DEFAULT_CHORD_LAW   # WHICH SHAPE the
    #   coefficients above describe (geometry.CHORD_LAWS). Changes what
    #   the design vector DRAWS, never how long it is.
    chord_limits: "geometry.ChordLimits | None" = None   # what the chord
    #   DISTRIBUTION has to obey, in metres and degrees (geometry.ChordLimits:
    #   a minimum chord, a maximum, a maximum local taper ANGLE, and which end
    #   is the wide one). None = unconstrained, which is every published run.
    #   Checked where the planform is built, so a violation is an in-contract
    #   failure (the penalty contract), never an exception at the optimiser.
    #: the aspect-ratio band the USER will accept b^2/S in, as
    #: ``(ar_min, ar_max)`` with either end None (api.AR_LIMIT_KEYS).
    #: A SECOND band beside ``sizing.AR_LIMITS``, never a replacement
    #: for it, and None — no limit at all — is every published run.
    ar_limits: "tuple | None" = None

    #: THE SIZE, WHERE THE USER SEARCHES IT. ``None`` — every published run —
    #: is the fixed planform above, bit-for-bit: no extra row, no extra
    #: variable, the same design vector every water study was run on. A band
    #: (or ``True`` for the default bracket, :data:`SIZE_BAND_FRAC`) puts
    #: ``b_m`` / ``S_m2`` in the design vector as ordinary design-box rows,
    #: appended after the operating point and before the chord coefficients
    #: so every existing index keeps its meaning.
    #:
    #: They are the ONLY rows here that can leave the solvers' honest
    #: aspect-ratio band (``sizing.AR_LIMITS``), because they are the only
    #: two that set it. A corner outside it is refused PER CANDIDATE with
    #: its reason, never banned at the band — the same rule the draught cap
    #: and the cavitation margin are priced by.
    span_bounds_m: tuple | None = None
    area_bounds_m2: tuple | None = None
    winglet_chord_follows: bool = False   # the tip device's chord CONTINUES
    #   the foil's own chord law past the tip instead of holding the tip
    #   chord (vlm.py; objective.Problem carries the air twin of this field).
    #   False = the rectangular device every published run flew, bit-for-bit.
    blend_frac: float = 0.0             # the tip device's root TRANSITION, as
    #   a VALUE (objective.Problem.blend_frac_fixed is the air twin of this
    #   field): the fraction of the device's arc length spent turning out of
    #   the foil plane. 0.0 = the sharp corner every published water run
    #   flies, bit-for-bit. It is not a design variable here for the same
    #   reason it is not one in air — how the corner is BUILT is a shop
    #   decision, and the device's own freedoms are its height and its cant.
    blend_shape: str = "arc"            # which turn law draws it
    #   (geometry.BLEND_SHAPES); "arc" is the constant-radius fillet.
    wing_blend_frac: float = 0.0        # ...and how much of the turn the FOIL
    #   does, as a fraction of the semi-span (geometry.span_path). Confined to
    #   the device the turn has at most blend_frac * h of arc, so the radius
    #   stays a fraction of a tip chord.
    junction_drag: bool = False         # charge the corner's interference
    #   drag (junction.py). The Hoerner correlation is a drag AREA — D/q —
    #   so it carries into water unchanged: what it is a function of is the
    #   corner's geometry (thickness ratio, chord, fillet radius), not the
    #   fluid. Its Reynolds-number blindness is the same documented limit
    #   here as in air. OFF by default; a blend switches it on, because
    #   without it a blend is only a differently drawn wake.
    draught_max_m: float | None = None  # the DRAUGHT CAP [m]: how deep the
    #   whole assembly is allowed to reach below the free surface,
    #   ``depth - min(z)`` over every panel. ``None`` = uncapped, which is
    #   every published run, bit-for-bit (the constraint vector stays width 1).
    #
    #   WHY THIS AND NOT A SPAN CAP (session 47, report SS17.10). The air
    #   families cap PROJECTED SPAN, and porting that here was measured to be
    #   the wrong constraint: the lateral projection is ``h*cos(cant)``, which
    #   is EVEN in the cant angle, while this family's dominant tip-device
    #   trade is ODD in it -- canting down sinks the device into more static
    #   head and is worth 4.4 % of L/D at the tall-and-shallow corner (12/12
    #   pairs). A lateral cap therefore holds fixed a quantity that cannot see
    #   the trade. Draught is the quantity a foiling craft is actually limited
    #   by (shallow water, launching, the trailer), and it is ODD in cant, so
    #   it binds on exactly the freedom the family is exploiting.
    #
    #   It is a CONSTRAINT, not a bound, and deliberately so: it is a limit on
    #   ``depth + the device's downward reach``, i.e. on a SUM of two searched
    #   quantities (``depth_m`` is a design variable), so no box on either row
    #   can express it. Measured off the solved geometry rather than from
    #   ``h*sin(cant)`` so that a blended or raked device is priced by the
    #   same rule as a straight one.
    #
    #   RESOLUTION CAVEAT. ``vlm`` reports panel MIDPOINTS, so this is the
    #   deepest panel centre, not the deepest point: at full height and -90
    #   deg of cant it reads 0.2391 m against a true tip at 0.2400 m, and
    #   that 0.9 mm is DISCRETISATION, not blending (the default config has
    #   blend_frac = 0, where the tip IS the arc end). It converges from
    #   below with panel count -- 0.2366 / 0.2391 / 0.2398 / 0.23999 at
    #   4 / 8 / 16 / 64 device panels -- so the cap is permissive by about
    #   0.9 mm at the shipped n_winglet = 8, one-signed. Stated because the
    #   study that prices this cap turns on millimetres.

    #: THE CRAFT'S ROLL and the tip immersion it must keep — the planar
    #: twin's fields of the same name, same defaults, same validators, and
    #: the same reason for existing (``HydrofoilProblem.heel_deg``). They
    #: bite harder here: this family already carries panels at their own
    #: submergence, so a heel and a cant compose instead of being two
    #: descriptions of one number, and the shallowest panel can be a tip
    #: DEVICE rather than the wing tip.
    heel_deg: float = 0.0
    tip_clearance_m: float | None = None

    # ---- the OPERATING POINT's two rows, as FIELDS (HydrofoilProblem's
    # fields of the same name, same defaults, same validators — the planar
    # twin's docstring is the one to read). The two device rows below stay
    # bare class attributes on purpose: a tip device's height and cant are
    # SHAPE, and their bands are geometric limits of the VLM, not an
    # operating point a user states about their own craft.
    DEPTH_BOUNDS: tuple[float, float] = (0.15, 1.0)     # [m]
    V_BOUNDS: tuple[float, float] = (8.0, 16.0)         # [m/s]
    #: tip-device HEIGHT band, arc length / semi-span. An ANNOTATED
    #: field, not a bare class attribute: without the annotation no
    #: builder keyword could reach it, which is exactly the state
    #: DEPTH_BOUNDS/V_BOUNDS were in before the speed row travelled.
    #: None = the published band; a stated band is flown, not refused.
    winglet_h_bounds: tuple | None = None
    #: ...and its CANT band, an annotated field for exactly the same reason
    #: (a bare class attribute is unreachable by any builder keyword). It is
    #: THE FOIL'S OWN CANT AND SWEEP [deg], as stated values.
    #:
    #: Lattice geometry (``geometry.dihedral_rotate`` and the quarter-chord
    #: offset in ``vlm.VLM``), so they are declared on THIS family and not on
    #: the planar :class:`HydrofoilProblem`: that one solves a monoplane
    #: Fourier equation whose quarter-chord line is straight along y, and a
    #: dihedral there would have nothing to act on.
    #:
    #: WHAT PRICES THEM, measured at this family's box centre. Both trades
    #: are two-sided, and which side wins depends on how hard the foil is
    #: working:
    #:
    #: * SWEEP spreads the load, lowering the peak suction, so it BUYS
    #:   cavitation margin: on the shipped 6 kN foil the margin runs
    #:   0.3496 -> 0.3975 from 0 to 20 deg for 1.9 % of L/D. On a lightly
    #:   loaded 1030 N wind foil, where cavitation is nowhere near binding,
    #:   it buys nothing and costs 9 % of L/D by 30 deg.
    #: * DIHEDRAL moves the tips in SUBMERGENCE. Up (positive) they sit in
    #:   less static head and cavitate sooner (0.3496 -> 0.2131 at 20 deg);
    #:   down (anhedral) they gain margin and cost DRAUGHT — 0.575 m flat
    #:   against 0.795 m at -20 deg, which is what ``draught_max_m`` caps.
    #:   L/D itself peaks slightly anhedral (-5 deg: 29.83 against 29.77).
    #:
    #: NOT A SPIRAL LEVER, and that is the one thing they must not be sold
    #: as. This craft's only vertical surface is the mast, whose quarter
    #: chord stands AHEAD of the CG, so ``Cn_beta`` is negative and
    #: ``wing_score.spiral_refusal`` refuses the design rather than scoring
    #: it — which is also why these are STATED values here and not searched
    #: rows: the criterion that would price a searched cant in air reads
    #: nothing at all under water. See
    #: tests/test_the_refusal_names_the_solver_it_is_running.py.
    #:
    #: 0.0 is every published run, bit-for-bit.
    wing_dihedral_deg: float = 0.0
    wing_sweep_deg: float = 0.0

    #: what a device TYPE narrows: a vertical fence is |cant| 84-90 of the
    #: signed band below, and under water that band has a SIDE — so a fence
    #: here is (-90, -84), the DOWN one, which is the side that carries
    #: cavitation margin (a device canted up sits in less static head and
    #: cavitates first — see the note beside the free-surface image). None =
    #: the published signed band, so every published run is unchanged.
    winglet_cant_bounds: tuple | None = None
    WINGLET_H_FRAC_BOUNDS = (0.0, 0.15)     # arc length / semi-span
    WINGLET_CANT_BOUNDS_DEG = (-90.0, 90.0)  # signed: + up, - down

    def __post_init__(self):
        # first: ``min_draught_m`` reads DEPTH_BOUNDS[0] and the draught cap
        # below is refused against it
        self.DEPTH_BOUNDS = depth_row(self.DEPTH_BOUNDS)
        self.V_BOUNDS = speed_row(self.V_BOUNDS)
        if self.polar_family is None:
            from .polar import default_polar_family
            self.polar_family = default_polar_family()
        _check_cp_min_available(self, "HydrofoilWingletProblem")
        _size_check(self)
        self.heel_deg = heel_angle(self.heel_deg)
        self.tip_clearance_m = _clearance(self.tip_clearance_m,
                                          "HydrofoilWingletProblem")
        _check_flown_reynolds(self, "HydrofoilWingletProblem")
        _check_draught_cap(self, "HydrofoilWingletProblem")
        if self.blend_shape not in geometry.BLEND_SHAPES:
            raise ValueError(
                f"unknown blend_shape {self.blend_shape!r}; "
                f"choose from {list(geometry.BLEND_SHAPES)}")
        lo, hi = geometry.WINGLET_BLEND_BOUNDS
        if not (lo <= float(self.blend_frac) <= hi):
            raise ValueError(
                f"blend_frac {self.blend_frac} outside the design band "
                f"{geometry.WINGLET_BLEND_BOUNDS}")
        wlo, whi = geometry.WING_BLEND_FRAC_BOUNDS
        if not (wlo <= float(self.wing_blend_frac) <= whi):
            raise ValueError(
                f"wing_blend_frac {self.wing_blend_frac} outside the design "
                f"band {geometry.WING_BLEND_FRAC_BOUNDS}")

    @property
    def param_labels(self) -> tuple:
        labels = ["taper", "twist_root_deg", "twist_tip_deg"]
        if self.section_polar is None:
            labels.append("tc")
        labels += ["depth_m", "V_ms", "winglet_h_frac", "winglet_cant_deg"]
        labels += list(size_labels(self))       # see HydrofoilProblem
        return tuple(labels) + geometry.chord_labels(self.chord_order,
                                                    self.chord_law)

    @property
    def bounds(self) -> np.ndarray:
        rows = [geometry.TAPER_BOUNDS,
                geometry.TWIST_ROOT_BOUNDS_DEG,
                geometry.TWIST_TIP_BOUNDS_DEG]
        if self.section_polar is None:
            rows.append(geometry.TC_BOUNDS)
        rows += [self.DEPTH_BOUNDS, self.V_BOUNDS,
                 geometry.winglet_h_row(self.winglet_h_bounds,
                                        self.WINGLET_H_FRAC_BOUNDS),
                 self._cant_box()] + size_rows(self)
        return geometry.with_chord_bounds(np.array(rows, dtype=float),
                                          self.chord_order,
                                          self.chord_max_frac,
                                          self.chord_law)

    def _cant_box(self) -> tuple:
        """The tip device's cant band: the published one, or the caller's.

        Validated against the SIGNED limits, because under water the side is
        part of the answer — the twin of ``wingtail._cant_box(signed=True)``.
        """
        if self.winglet_cant_bounds is None:
            return self.WINGLET_CANT_BOUNDS_DEG
        lo, hi = (float(v) for v in self.winglet_cant_bounds)
        cmin, cmax = geometry.WINGLET_CANT_LIMITS_SIGNED_DEG
        if not (cmin <= lo < hi <= cmax):
            raise ValueError(
                f"winglet cant_bounds {(lo, hi)} outside the physical VLM "
                f"range {geometry.WINGLET_CANT_LIMITS_SIGNED_DEG}")
        return (lo, hi)

    @property
    def dim(self) -> int:
        return int(self.bounds.shape[0])

    @property
    def min_draught_m(self) -> float:
        """The shallowest draught any design in this box can reach [m].

        Here it is just the shallowest depth: the only thing below the foil
        plane is the tip device, and the device can be zero-height or canted
        UP, in which case it adds nothing. Families with a second surface
        override this (see ``hydrotail.HydrofoilTailProblem``).
        """
        return float(self.DEPTH_BOUNDS[0])

    @property
    def immersion_capped(self) -> bool:
        """See :meth:`HydrofoilProblem.immersion_capped` — one field, on
        purpose, because the margin vector's width is a contract."""
        return self.tip_clearance_m is not None

    @property
    def n_constraints(self) -> int:
        return (1 + (0 if self.draught_max_m is None else 1)
                + (1 if self.immersion_capped else 0))

    @property
    def constraint_labels(self) -> tuple:
        """Display names for the margins this problem actually returns.

        ``api.design_report`` prefers these over the static ProblemSpec
        labels, which describe the UNCAPPED family (the precedent is
        ``section_wing.SectionWingProblem``). Without this, a capped run would
        report its second margin under the first one's name.
        """
        return (("cavitation margin",)
                + (() if self.draught_max_m is None
                   else (DRAUGHT_CONSTRAINT_LABEL,))
                + ((IMMERSION_CONSTRAINT_LABEL,) if self.immersion_capped
                   else ()))

    def CL_target(self, V: float, S: float | None = None) -> float:
        """Trim CL on area ``S`` — see :meth:`HydrofoilProblem.CL_target`."""
        area = float(self.S if S is None else S)
        return float(self.L_design * heel_lift_factor(self.heel_deg)
                     / (0.5 * self.rho * V**2 * area))


def evaluate_hydrofoil_winglet(
        x: np.ndarray, prob: HydrofoilWingletProblem | None = None) -> dict:
    """Full evaluation of the nonplanar hydrofoil; never raises in contract."""
    from . import geometry
    from .objective import _fail
    from .vlm import VLM, ImagePlane

    prob = prob or HydrofoilWingletProblem()
    x = np.asarray(x, dtype=float)
    bnds = prob.bounds
    if x.shape != (bnds.shape[0],):
        return _fail(f"design vector shape {x.shape} != ({bnds.shape[0]},)")
    if np.any(x < bnds[:, 0] - 1e-12) or np.any(x > bnds[:, 1] + 1e-12):
        return _fail("bounds violation")

    if prob.section_polar is None:
        taper, tw_root, tw_tip, tc, depth, V, h_frac, cant = (
            float(v) for v in x[:8])
    else:
        # the designed section fixes t/c: the row is gone from the vector
        taper, tw_root, tw_tip, depth, V, h_frac, cant = (
            float(v) for v in x[:7])
        tc = float(getattr(prob.section_polar, "tc", 0.12))
    b_x, S_x = size_from_x(prob, x)          # see evaluate_hydrofoil
    why = size_refusal(prob, b_x, S_x)
    if why is not None:
        return _fail(f"size: {why}")
    try:
        wing = geometry.Wing(
            b=b_x, S=S_x, taper=taper, twist_root_deg=tw_root,
            twist_tip_deg=tw_tip, tc=tc,
            sweep_deg=prob.wing_sweep_deg,
            dihedral_deg=prob.wing_dihedral_deg,
            chord_limits=prob.chord_limits,
            chord_coeffs=geometry.chord_coeffs_from_x(x, prob.chord_order,
                                                      prob.chord_law))
    except ValueError as exc:
        return _fail(f"planform: {exc}")
    # THE SECTION THIS CANDIDATE FLIES — at its own Reynolds number where the
    # design asked for that (``polar_for``). The mac is the candidate's, so
    # with the size rows open the table follows the planform as well as the
    # speed; with ``flown_reynolds`` off it is the family's Re-1e6 table and
    # every published number is unchanged.
    try:
        pol = polar_for(prob, tc, wing.mac, V, prob.section_polar)
    except ValueError as exc:
        return _fail(f"polar family: {exc}")

    CLt = prob.CL_target(V, S_x)
    try:
        # foil in the z = 0 plane, free surface `depth` above it
        model = VLM(wing, N=prob.N_vlm, winglet_h_frac=h_frac,
                    winglet_cant_deg=cant, n_winglet=prob.n_winglet,
                    winglet_blend_frac=prob.blend_frac,
                    winglet_blend_shape=prob.blend_shape,
                    winglet_wing_blend_frac=prob.wing_blend_frac,
                    winglet_chord_follows=prob.winglet_chord_follows,
                    a=pol.a_lin, alpha_L0=pol.alpha_L0, V=V,
                    image=ImagePlane(z=depth, kind="free_surface"))
        lo, hi = (np.deg2rad(d) for d in prob.alpha_bracket_deg)
        alpha, res = model.solve_trim(CLt, alpha_bracket=(lo, hi))
    except (ValueError, np.linalg.LinAlgError) as exc:
        return _fail(f"solver failure: {exc}")

    alpha_eff_deg = np.rad2deg(res.alpha_eff)
    lo_v, hi_v = pol.alpha_valid
    if alpha_eff_deg.min() < lo_v or alpha_eff_deg.max() > hi_v:
        return _fail(
            "effective AoA outside polar validity (stall/extrapolation proxy)",
            alpha_eff_range=(float(alpha_eff_deg.min()),
                             float(alpha_eff_deg.max())),
        )

    CDp = float(np.sum(pol.cd(alpha_eff_deg) * res.c * res.width) / res.S)
    # THE STRUT IS A SURFACE THE CRAFT CAN BE ASKED ABOUT. Charged exactly
    # as it always was where there is one (``fin.has_fin``, True unless the
    # design says otherwise), and zero where the design states none — the
    # same three-state contract every other family's vertical surface has,
    # so "no strut" cannot mean "a strut nobody pays for".
    cd0_mast = (_mast_cd0(depth, S_x, prob.c_mast, prob.cf_mast)
                if _fin.has_fin(prob) else 0.0)

    # foil/device corner: invisible to a lifting-surface method, so it is the
    # same reduced-order add-on the air problem charges (junction.py) and it
    # is never on by default. The correlation is a drag AREA, a function of
    # the corner's geometry alone, so it needs no water properties.
    jr = None
    if prob.junction_drag and h_frac > 0.0:
        from . import junction
        c_tip = float(wing.chord(np.array([wing.b / 2.0]))[0])
        jr = junction.report(
            tc=wing.tc, chord=c_tip, s_ref=wing.S,
            h=h_frac * wing.b / 2.0, cant_deg=cant,
            blend_frac=prob.blend_frac, n_junctions=2,
            blend_shape=prob.blend_shape,
            wing_arc=prob.wing_blend_frac * wing.b / 2.0)
    CD_junction = float(jr["CD_junction"]) if jr else 0.0

    CD = res.CDi + CDp + cd0_mast + CD_junction
    LoD = res.CL / CD
    if not np.isfinite(LoD):
        return _fail("non-finite L/D")

    # THE ASSEMBLY HAS TO BE IN THE WATER — checked before the cavitation
    # book, because an emerged panel makes that book meaningless rather than
    # pessimistic. Measured over the same panel MIDPOINTS the draught is
    # (the resolution caveat on ``draught_max_m`` applies here in the other
    # direction, and is one-signed the same way).
    imm = immersion_margin(depth, res.y, res.z, prob.heel_deg,
                           prob.tip_clearance_m or 0.0)
    if imm["h_min_m"] <= 0.0:
        return _fail(
            f"panel emerged: at {prob.heel_deg:g} deg of heel the assembly "
            f"reaches {-imm['h_min_m']:.3g} m above a free surface "
            f"{depth:.3g} m over the foil (span {b_x:.3g} m, at "
            f"y = {imm['y_at_h_min_m']:.3g} m). A partly emerged foil is a "
            f"different problem, not a worse design.")

    try:
        cav = cavitation_margin_panels(alpha_eff_deg, res.z, depth, pol,
                                       V=V, rho=prob.rho,
                                       p_vap=prob.p_vap, y=res.y,
                                       heel_deg=prob.heel_deg)
    except ValueError as exc:
        return _fail(f"cavitation: {exc}")

    S_wl = float(np.sum((res.c * res.width)[res.is_winglet]))

    # DRAUGHT: how deep the assembly reaches below the free surface. z is up
    # positive and the surface is z = depth, so the deepest panel is the one
    # with the smallest z — the exact mirror of the "tip_z_m" (shallowest)
    # already reported below. Taken off the SOLVED geometry, so a blended
    # device, a canted one and a straight one are all measured the same way.
    # ...and it is the HEELED height that decides which panel is deepest:
    # rolling the craft swaps which tip is the low one, so a draught read off
    # the upright geometry would cap a shape the craft is not flying.
    z_heeled = heeled_z(res.y, res.z, prob.heel_deg)
    draught_m = float(depth - float(z_heeled.min()))
    g_draught = (None if prob.draught_max_m is None
                 else float(prob.draught_max_m) - draught_m)
    # ``g`` is the FULL margin vector whenever there is more than one margin,
    # exactly as hydrotail.py does it, and a bare scalar when there is only
    # the cavitation one — which is every uncapped run, bit-for-bit.
    #
    # It must NOT stay a scalar when capped. Consumers assemble the
    # optimiser's constraint vector from this value on the success path
    # (``wing_score.composite_objective``) while sizing the FAILURE path from
    # ``n_constraints``; a scalar here means success width 1 against failure
    # width 2, which either raises deep inside the optimiser or — worse —
    # silently drops the draught margin so a design over the cap is reported
    # feasible. Found by adversarial review of this change, not by a test.
    g_list = ([float(cav["g"])]
              + ([] if g_draught is None else [float(g_draught)])
              + ([float(imm["g"])] if prob.immersion_capped else []))
    g_out = g_list[0] if len(g_list) == 1 else g_list
    return {
        "feasible": True, "reason": "", "score": float(LoD), "LoD": float(LoD),
        "g": g_out, "g_cav": float(cav["g"]), "sigma_cav": cav["sigma_cav"],
        "heel_deg": float(prob.heel_deg),
        "immersion_m": float(imm["h_min_m"]),
        "y_at_immersion_m": float(imm["y_at_h_min_m"]),
        "tip_clearance_m": prob.tip_clearance_m,
        "g_immersion": (float(imm["g"]) if prob.immersion_capped else None),
        "draught_m": draught_m,
        "draught_max_m": (None if prob.draught_max_m is None
                          else float(prob.draught_max_m)),
        "g_draught": g_draught,
        "sigma_cav_root": cav["sigma_cav_root"],
        "cp_min_worst": cav["cp_min_worst"],
        "depth_worst": cav["depth_worst"],
        "CL": res.CL, "CL_target": CLt,
        "CDi_total": res.CDi, "CDp": CDp, "cd0_mast": cd0_mast,
        "CD_junction": CD_junction,
        "CD": float(CD), "e_total": res.e, "CDi": res.CDi,
        "alpha_deg": float(np.rad2deg(alpha)),
        "depth": depth, "V": V, "Fn_h": froude_depth(V, depth),
        "Re_mac": float(prob.rho * V * wing.mac / prob.mu),
        "polar": getattr(pol, "name", "unknown"),
        # ...and the Reynolds number that table was MEASURED at, beside the
        # one the design is flying. Two numbers, because for every published
        # run they differ (Re_mac 0.9-1.8e6 against a table at 1e6) and a
        # card that showed only the flown one would imply a table that
        # followed it.
        "section_re": (float(pol.Re) if getattr(pol, "Re", None) is not None
                       else None),
        "winglet": {
            "h_frac": h_frac,
            "cant_deg": cant,
            # the height that reached the lattice; h_frac stays the
            # REQUESTED number, so the pair reads asked / flown
            "h_m": float(model.winglet_h_m),
            "S_planform": S_wl,
            # what the device's chord did (objective.evaluate_winglet reports
            # the same pair): the narrowest panel chord on it, and whether it
            # continued the foil's chord law or held the tip chord
            **({"chord_min_m": float(np.min(res.c[res.is_winglet]))}
               if res.is_winglet.any() else {}),
            "chord_follows": bool(prob.winglet_chord_follows),
            "tip_z_m": float(res.z.max()),
            "tip_depth_m": float(depth - res.z.max()),
            "blend_frac": float(prob.blend_frac),
            "blend_shape": str(prob.blend_shape),
            "wing_blend_frac": float(prob.wing_blend_frac),
            "wing_blend_arc_m": float(prob.wing_blend_frac * b_x / 2.0),
            **({"junction": jr} if jr else {}),
        },
        # the SIZE actually flown — see evaluate_hydrofoil
        "b": float(b_x), "S": float(S_x), "AR": float(b_x * b_x / S_x),
        "wing": wing,
        "vlm": res,
    }


def fg_hydrofoil_winglet(x: np.ndarray,
                         prob: HydrofoilWingletProblem | None = None
                         ) -> tuple[float, float | np.ndarray]:
    """Constrained-harness callable: (L/D, margins).

    Width follows the problem, in the shape :func:`airfoil.fg_airfoil` already
    uses: a SCALAR cavitation margin uncapped — bit-for-bit every published
    run — and ``[g_cav, g_draught]`` when ``prob.draught_max_m`` is set. A
    failed design returns a margin of the matching width, because the
    optimiser's constraint block is sized up front.
    """
    prob = prob or HydrofoilWingletProblem()
    n = prob.n_constraints
    out = evaluate_hydrofoil_winglet(x, prob)
    if not out["feasible"]:
        return (PENALTY, G_FAIL) if n == 1 else (PENALTY, np.full(n, G_FAIL))
    if n == 1:
        return float(out["score"]), float(out["g"])
    return float(out["score"]), np.asarray(out["g"], dtype=float)
