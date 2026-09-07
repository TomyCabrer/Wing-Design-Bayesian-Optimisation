"""Tandem pair in the NONPLANAR solver — two wings that can carry winglets.

``tandem.py`` couples two PLANAR lifting lines: excellent for the mutual-
induction question it was built for, and structurally unable to carry a tip
device, because every filament in it is horizontal. That is why the tandem
family was the last one with no winglet, no designed section and no image
plane. This module puts the pair into the nonplanar VLM instead
(``vlm.SecondWing``): one influence matrix over front wing + its tip device +
rear wing + its tip device, one Trefftz plane over the combined wake.

What that buys, beyond "the same answer with more panels":

* a TIP DEVICE on either wing, or both, with its own height and cant — the
  nonplanar trade the planar pair cannot represent at all;
* the pair's induced drag from ONE far-field sum, so the mutual term and the
  self terms are counted together (Munk's stagger theorem collapses the
  streamwise placement, exactly as in the lifting-line version);
* every modifier the rest of the package has — the chord law (one per wing),
  the flight state, the weight-coupled size — and a DESIGNED CST section
  through ``section_polar``.

Design vector (blocks; modifiers last, package rule)
----------------------------------------------------
    [taper_front, taper_rear,
     twist_root_front_deg, twist_tip_front_deg,
     twist_root_rear_deg,  twist_tip_rear_deg,
     area_split_front, decalage_deg]
    [+ winglet_h_front, cant_front, winglet_h_rear, cant_rear   if winglets]
    [+ b_m, b_rear_m, S_m2  if the size modifier is on — a span per WING,
                             because the two wings are two wings]
    [+ V_ms, altitude_m     if the flight modifier is on]
    [+ chord_front_k1..k3, chord_rear_k1..k3   if the chord law is on]

Deliberate difference from ``tandem.py``: the twist of each wing is the
package's ROOT/TIP linear law (geometry.Wing) rather than that module's
3-knot spline. Two reasons — the Wing object is what the VLM samples, and it
is what carries the chord law — and one consequence: the two problems are not
the same design space, so their numbers are their own (the planar family
keeps its published results). The decalage is the rear wing's incidence
offset, added to both its twist ends, so it means exactly what it means
there.

Objective (MAXIMISE): system L/D at the trimmed CL on the TOTAL area, or
payload L/D with the size modifier on. Failure contract: the package's.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from . import dynamics as dyn
from . import fin as _fin
from . import geometry
from .objective import MU_SL, PENALTY, RHO_SL, _fail
from .polar import default_polar
from .tandem import DECALAGE_BOUNDS_DEG, SPLIT_BOUNDS
from .vlm import (MIN_PLANE_CLEARANCE_FRAC, VLM, SecondWing,
                  VerticalSurface, device_is_flown)

G_FAIL = -1.0

#: tip-device box, shared by both wings (geometry.py's air bands)
WINGLET_H_FRAC_BOUNDS = geometry.WINGLET_H_FRAC_BOUNDS
WINGLET_CANT_BOUNDS_DEG = geometry.WINGLET_CANT_BOUNDS_DEG


def _device_clearance(b: float, dz: float, h_frac_front: float,
                      cant_front: float, h_frac_rear: float,
                      cant_rear: float, b_rear: float | None = None,
                      blend_frac: float = 0.0, blend_shape: str = "arc",
                      wing_blend_frac: float = 0.0) -> str | None:
    """Reason string if the pair's two tip devices run into each other.

    Both devices point UP (the cant band is 60-90 deg from the wing plane),
    so the front wing's device occupies ``z`` in [0, tip_z_front] and the
    rear's [dz, dz + tip_z_rear]. Those bands must not meet, and there is
    nothing subtle about why:

    * GEOMETRICALLY, a front device that reaches the rear wing's plane and a
      rear device rising out of it are one continuous vertical surface joining
      the two wings. That is a BOX WING (or a joined wing) — a different
      aircraft, with a closed lifting system this pair's panelisation does not
      model: two separate wings with two separate devices is what is built
      here, and drawing them on top of each other is what made the pair look
      as if its winglets were joined.
    * NUMERICALLY, the front device's trailing wake is a vertical sheet at the
      tip running straight downstream. Put the rear device inside it and its
      control points sit ON the shed vorticity: the horseshoe kernel is
      singular there and the near-field influence is not merely inaccurate,
      it is regularised away to zero (vlm._seminf). In the Trefftz plane the
      two devices' wake traces coincide outright, so the far-field sum sees
      ONE sheet of doubled circulation — Munk's stagger theorem collapsing the
      streamwise separation that is the only thing keeping them apart.

    The clearance floor is :data:`vlm.MIN_PLANE_CLEARANCE_FRAC` of the
    semi-span — the same "these two surfaces are too close for this method"
    number the imaged solvers use against their own image plane.

    A pair with no device on either wing is not checked: two PLANAR wings are
    separated by the streamwise stagger, which is the layout this family was
    built to study, and the vertical gap there is a physics parameter rather
    than a clash (the module docstring's "pure streamwise separation never
    decouples the pair").
    """
    # each device's height is a fraction of ITS OWN wing's semi-span (that is
    # what the h_frac variable means where the wing is panelised), so a pair
    # with two spans has two semi-spans here
    semi = 0.5 * float(b)
    semi_r = semi if b_rear is None else 0.5 * float(b_rear)
    # a device shorter than the panel builder's floor is DROPPED, so it
    # cannot clash with anything — read the same rule, not a second one
    tz = []
    for h_frac, cant, s in ((h_frac_front, cant_front, semi),
                            (h_frac_rear, cant_rear, semi_r)):
        h = float(h_frac) * s
        # ...at the tip height the device ACTUALLY reaches: a blend trades
        # tip height for projected span, so reading the sharp-corner height
        # here would refuse a pair whose blended devices are in fact clear
        tz.append(geometry.winglet_tip_height(
            h, float(cant), float(blend_frac), str(blend_shape),
            float(wing_blend_frac) * s)
            if device_is_flown(h_frac) else 0.0)
    tip_z_front, tip_z_rear = tz
    if tip_z_front <= 0.0 and tip_z_rear <= 0.0:
        return None
    # the floor is a resolution statement about the WAKE the two devices
    # share, so it is set by the coarser of the two panelisations — the wider
    # wing's. Equal spans leave it exactly the number it always was.
    floor = MIN_PLANE_CLEARANCE_FRAC * max(semi, semi_r)
    lo_f, hi_f = 0.0, float(tip_z_front)
    lo_r, hi_r = float(dz), float(dz) + float(tip_z_rear)
    gap = max(lo_f, lo_r) - min(hi_f, hi_r)      # > 0 when the bands are clear
    if gap >= floor:
        return None
    return (
        f"the two tip devices are not separated: the front device spans z = "
        f"[{lo_f:.3g}, {hi_f:.3g}] m and the rear's [{lo_r:.3g}, {hi_r:.3g}] "
        f"m, leaving {gap:.3g} m against a floor of {floor:.3g} m. Two wings "
        f"whose devices meet are a JOINED wing, not a tandem pair — raise the "
        f"stagger dz, or shorten a device")


@dataclass
class TandemVLMProblem:
    """Two coupled wings in one nonplanar solve (module docstring)."""

    #: THE PAIR'S CANT AND SWEEP [deg] — one answer for both wings, because
    #: it is a statement about how the aeroplane is built. Lattice geometry
    #: (geometry.dihedral_rotate, and the quarter-chord offset in vlm.VLM),
    #: so this family can SCORE them where the lifting-line `tandem` cannot.
    wing_dihedral_deg: float = 0.0
    wing_sweep_deg: float = 0.0
    #: ...SEARCHED INSTEAD OF STATED, and asked ONE ROW AT A TIME. Each of
    #: these puts its own row into the design vector, and the matching field
    #: above must then stay 0.0 — one question, one place, and
    #: ``api.check_flags`` refuses a flag the vector already carries. They
    #: are separate because they are priced against different things: the
    #: dihedral is the pair's only source of ``Cl_beta`` that holds at any
    #: lift, and the sweep moves the neutral point at a real cost in L/D
    #: while its own dihedral effect goes as CL and vanishes at cruise.
    #:
    #: A DIMENSION CHANGE, so this is a variant AXIS and not a flag
    #: (``api.TANDEM_VLM_VARIANTS``), the same rule ``wingtail.WingTailProblem
    #: .dihedral_free`` exists for. It belongs on THIS family and not on the
    #: lifting-line ``tandem``: each surface's quarter-chord line there is
    #: straight along y, so there is no out-of-plane geometry for a dihedral
    #: to act on.
    dihedral_free: bool = False
    sweep_free: bool = False
    #: BOTH, as one word — what every published free-cant run was built with
    #: and what the registry's ``free`` state spells. :meth:`__post_init__`
    #: expands it into the pair above; nothing below reads it again.
    cant_free: bool = False
    #: MEASURE THE LATERAL HALF. Off (every published run) the pair is solved
    #: exactly as it always was and the vertical surface below is a DRAG
    #: charge only; on, that same surface joins the lattice so a sideslip has
    #: something to push on, and the breakdown gains the six lateral
    #: derivatives and the ``spiral`` criterion the composite scores.
    #:
    #: Opted into by WEIGHTING ``spiral`` and by nothing else
    #: (``api._arm_lateral`` sets it), because it is not free: the vertical's
    #: panels enlarge the influence matrix and ``dynamics.deck`` costs six
    #: more solves on top. Without it a searched cant is a row nothing
    #: prices, which is worse than no row at all.
    lateral: bool = False
    #: GATE THE MODES, at a MIL-F-8785C level — 3, 2, 1, or ``None`` for the
    #: shipped behaviour (no handling rows at all, every published run
    #: bit-for-bit). ``wingtail.WingTailProblem.handling_level``'s twin, and
    #: deliberately the same field on the same terms: a pair is an aeroplane
    #: and the standard does not care how many wings it has.
    #:
    #: Setting it adds :data:`handling.N_ROWS` signed margins AFTER whatever
    #: this problem already declared, so the size modifier's stress margin
    #: keeps ``g[0]`` for every caller that reads it. It IMPLIES
    #: :attr:`lateral` — a Dutch-roll requirement on a pair with no vertical
    #: surface in the scored lattice would be gating a number that is exactly
    #: zero for a geometric reason — and it is armed through the same switch
    #: the ``spiral`` weight is (``api._needs_lateral``).
    #:
    #: WHY IT BELONGS HERE AND NOT ON THE LIFTING-LINE ``tandem``: that
    #: family charges the fin's drag and weighs it but never flies it, so its
    #: ``Cn_beta`` is not a measurement. The same reason the published
    #: ``tail`` family is not offered the gate either.
    handling_level: int | None = None
    #: WHERE THE YAW AND ROLL MOMENTS ARE TAKEN [m], x aft of the front
    #: wing's quarter chord.
    #:
    #: FIXED AT THE FRONT WING'S QUARTER CHORD, and NOT derived per
    #: candidate. A tandem is trimmed in LIFT ALONE — the decalage is a
    #: design variable, not a moment unknown — so the pair has no balance
    #: point the solve finds, and a reference read off each candidate's lift
    #: centroid would make ``spiral`` buyable with ``area_split_front`` and
    #: ``decalage_deg`` instead of with the cant. x = 0 is also the reference
    #: ``fin.tandem_fin_station`` sizes the fin's arm against and the one
    #: ``flightmodel.build_flight_model`` assumes when it rebuilds a pair, so
    #: the scored deck and the flown one agree by construction. The lift
    #: centroid is REPORTED beside it (``x_lift``) rather than scored.
    x_cg: float = 0.0
    #: THE PAIR'S VERTICAL SURFACE — the same four keys the lifting-line
    #: ``tandem.TandemProblem`` carries, and for the same reason: the fin is
    #: sized against the STAGGER, reported, lofted and flown, so it is paid
    #: for here. ``fin=False`` is a pair with no vertical surface at all and
    #: reproduces every number this family published before it was charged.
    fin: bool = True
    fin_volume_coeff: float | None = None
    fin_ar: float | None = None
    fin_tc: float | None = None
    #: HOW FAR AFT OF THE REAR WING THE FIN STANDS [m] — the boom.
    #:
    #: ``None`` is ``fin.TANDEM_FIN_BOOM_FRAC`` of the stagger, which is
    #: where a vertical tail belongs on a three-view and where this pair's
    #: spiral goes (the measured table is on that constant). A number
    #: travels verbatim; ``0.0`` puts the fin back ON the rear wing's root.
    #: It is a LENGTH the builder states, not a design variable: how long a
    #: boom is, is a decision about the airframe, and nothing in this
    #: objective would pay for it — a searched boom runs to the top of
    #: whatever band it is given, because the fin's area falls as 1/l_t.
    fin_boom_m: float | None = None
    b: float = 10.0                # the FRONT wing's span
    #: THE REAR WING'S SPAN, when it is not the front's — tandem.TandemProblem's
    #: ``b_rear``, same contract: None flies both wings at ``b`` (bit-for-bit
    #: the published pair), a length in metres travels verbatim, and under the
    #: SIZE modifier both spans are design variables instead (one row each).
    #: Here it carries a second job the planar pair does not have: a tip
    #: device's height is a fraction of ITS OWN wing's semi-span, so the two
    #: devices are sized on the two spans and :func:`_device_clearance` reads
    #: each one on the wing it belongs to.
    b_rear: float | None = None
    S_total: float = 20.0          # pair reference area
    #: THE STAGGER, IN METRES, AND IT IS THE USER'S — tandem.TandemProblem's
    #: dx/dz, same contract: api.py derives them from a span the user types,
    #: and nothing downstream (the size modifier included) moves them again.
    #: Here they carry a second job the planar pair does not have: with a tip
    #: device on each wing, ``dz`` is what keeps the two devices apart, and
    #: :func:`_device_clearance` refuses the pair when it does not.
    dx: float = 5.0                # rear quarter-chord aft of the front
    dz: float = 1.0                # rear above the front
    N: int = 40                    # panels per wing (full span)
    n_winglet: int = 8
    V: float = 14.6
    rho: float = RHO_SL
    mu: float = MU_SL
    CL_target: float = 0.5         # SYSTEM CL on S_total
    cd0_extra: float = 0.0
    polar: Any = field(default_factory=default_polar)
    section_polar: Any = None      # a DESIGNED section (hydrofoil_section-
    #                                style): overrides ``polar`` when set
    polar_rear: Any = None         # the REAR wing's own section
    #   (tandem.TandemProblem.polar_rear's twin). None keeps the documented
    #   default — the rear wing flies the front's — so every published run is
    #   bit-for-bit unchanged. The two wings of a tandem carry different
    #   shares of the weight at different local Reynolds numbers, so "one
    #   aerofoil for both" is a choice, not a law.
    alpha_bracket_deg: tuple[float, float] = (-10.0, 15.0)
    # ---- design freedoms (each changes the design vector)
    winglets: bool = False
    #: the two tip-device HEIGHT bands, front then rear, each a fraction
    #: of ITS OWN wing's semi-span (a pair can have two spans). None =
    #: the published (0, 0.15) — where the model was MEASURED, not a cap.
    winglet_h_bounds_front: tuple | None = None
    winglet_h_bounds_rear: tuple | None = None
    #: ...and the CANT band both devices search, which is what a tip-device
    #: TYPE narrows (objective.WINGLET_TYPES: a vertical fence is 84-90 deg
    #: of the published 60-90). An ANNOTATED field for the reason
    #: ``winglet_h_bounds`` is one in hydrofoil.py: a bare class attribute
    #: is unreachable by any builder keyword, and this pair was the family
    #: where "vertical fence (90 deg)" could not be asked for at all — the
    #: shell dropped the flag (api.sanitise_flags) and searched the full
    #: band under a label that said 90. One band for BOTH wings: the shape
    #: is one question about the pair's tip device, and the shell asks it
    #: once. None = the published band, so every published run is
    #: bit-for-bit unchanged.
    winglet_cant_bounds: tuple | None = None
    chord_order: int = 0
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
    winglet_chord_follows: bool = False   # the tip device's chord CONTINUES
    #   each wing's own chord law past its tip instead of holding the tip
    #   chord (vlm.py; objective.Problem carries the twin of this field).
    #   False = the rectangular device every published run flew, bit-for-bit.
    #   Both wings of the pair, for the usual reason: it is a statement about
    #   the design, not about one surface.
    # ---- the tip devices' root TRANSITION, as VALUES (never dimensions).
    # Same three fields and the same meanings as objective.Problem's, applied
    # to BOTH wings of the pair for the same reason winglet_chord_follows is:
    # it is a statement about how the design is built, not about one surface.
    # 0.0 is the sharp corner every published pair flies, bit-for-bit.
    blend_frac_fixed: float = 0.0
    wing_blend_frac: float = 0.0
    blend_shape: str = "arc"
    junction_drag: bool = False     # charge each corner's interference drag
    #   (junction.py) — the add-on a blend exists to buy, so a blend defaults
    #   it on. Charged on BOTH wings' devices, each on its own tip chord.
    flight_free: bool = False
    material: object = None   # WHAT THE WING IS BUILT OF
    #   (materials.Material, or a key from materials.MATERIALS).
    #   None = 2024-T3 aluminium, the material Raymer's correlation
    #   was regressed over, so every published run is bit-identical.
    #   It sets BOTH the allowable the spar is sized to and the
    #   factor the statistical wing weight is scaled by — the second
    #   is the one the span answer actually rides (materials.py).
    size_free: bool = False
    #: the wing-loading size mode's two knobs, tandem.TandemProblem's twins:
    #: the W/S the pair's TOTAL area follows through the weight loop, and the
    #: band BOTH span rows are searched in. Both None = the published pair.
    wing_loading_Pa: float | None = None
    span_bounds_m: tuple | None = None
    area_bounds_m2: tuple | None = None    # (min, max) AREA row of the free
    #   planform mode [m^2]; None -> the fractional band around this
    #   problem's own reference area (sizing.size_bounds)
    ws_bounds_pa: tuple | None = None      # (min, max) of the SEARCHED
    #   wing-loading row [Pa] (sizing.SIZE_MODE_WS_FREE); None ->
    #   sizing.WS_FRAC_BOUNDS around the loading this problem already flies
    wing_loading_max_Pa: float | None = None   # the MISSION's ceiling on W/S
    #   (constraint_diagram): clips that band and refuses any candidate above
    #   it, in every sized mode
    W_fixed_N: float | None = None
    mission: Any = None

    #: the searched cant rows, as ``(field, label, band name in geometry)``.
    #: The wing+tail family's table, spelled again here rather than imported,
    #: because these are the PAIR's rows — but in the same order, so the two
    #: families' vectors are read the same way.
    CANT_ROWS = (("dihedral_free", "wing_dihedral_deg", "DIHEDRAL_BOUNDS_DEG"),
                 ("sweep_free", "wing_sweep_deg", "SWEEP_BOUNDS_DEG"))

    def __post_init__(self):
        # the pair, said in one word — expanded once, idempotently, because
        # ``dataclasses.replace`` re-runs this
        if self.cant_free:
            self.dihedral_free = self.sweep_free = True
        if self.blend_shape not in geometry.BLEND_SHAPES:
            raise ValueError(
                f"unknown blend_shape {self.blend_shape!r}; "
                f"choose from {list(geometry.BLEND_SHAPES)}")
        blo, bhi = geometry.WINGLET_BLEND_BOUNDS
        if not (blo <= float(self.blend_frac_fixed) <= bhi):
            raise ValueError(
                f"blend_frac_fixed {self.blend_frac_fixed} outside the "
                f"design band {geometry.WINGLET_BLEND_BOUNDS}")
        wlo, whi = geometry.WING_BLEND_FRAC_BOUNDS
        if not (wlo <= float(self.wing_blend_frac) <= whi):
            raise ValueError(
                f"wing_blend_frac {self.wing_blend_frac} outside the design "
                f"band {geometry.WING_BLEND_FRAC_BOUNDS}")
        if float(self.blend_frac_fixed) > 0.0 and not self.winglets:
            raise ValueError(
                "blend_frac_fixed needs a tip device to blend into the wing "
                "it sits on; this problem carries none")
        # ...asked PER ROW: a family that searches only one of them may be
        # told the other, and refusing that would be a ban with no
        # measurement behind it
        for _attr, _lbl, _band in self.CANT_ROWS:
            if getattr(self, _attr) and getattr(self, _lbl):
                raise ValueError(
                    f"the pair's cant is answered twice: {_attr}=True "
                    f"searches {_lbl}, and {_lbl}={getattr(self, _lbl)} "
                    f"states it. Use the family that searches this row and "
                    f"leave the field at 0.0, or the family that states it")
        if self.b_rear is not None and not float(self.b_rear) > 0.0:
            raise ValueError(
                f"the rear wing's span must be a positive length in metres "
                f"(got {self.b_rear!r}); leave it None for a pair whose two "
                f"wings are the same width")
        if self.flight_free and self.mission is None:
            from .mission import MissionSpec, weight_for
            self.mission = MissionSpec(
                W_N=weight_for(self.CL_target, self.V, self.S_total),
                V=self.V, altitude_m=0.0)
        if self.handling_level is not None:
            # VALIDATED HERE, once, so a level the gate does not implement is
            # refused where the problem is built rather than deep inside an
            # evaluation — and the IMPLICATION is stated here too, for a
            # caller that constructs the problem directly instead of coming
            # through ``api._arm_lateral``. Gating modes off a lattice with
            # no vertical surface in it would be gating Cn_beta = 0.
            from . import handling as _hq

            self.handling_level = _hq.level_of(self.handling_level)
            self.lateral = True

    @property
    def constraint_labels(self) -> tuple[str, ...]:
        """The margins this problem declares, in ``g``'s own order.

        The size modifier's stress margin is index 0 where it exists and
        stays there; the handling rows go LAST. Declared rather than
        counted, because ``api`` reads a built problem's own declaration in
        preference to the registry's static count
        (:func:`api._with_wing_objective`) — which is what lets one flag
        change the width without every generated ``ProblemSpec`` knowing.

        THE PAIR HAS NO STATIC-MARGIN ROW, which is the one place this is
        not the wing+tail's list with a name changed: a tandem is trimmed in
        LIFT ALONE (the decalage is a design variable, not a moment
        unknown), so there is no pitch balance for a margin to be measured
        against. An unsized pair with no level therefore declares NOTHING,
        and that is exactly the unconstrained problem this family shipped as.
        """
        out: tuple[str, ...] = ()
        if self.size_free:
            out += ("root-bending stress margin",)
        if self.handling_level is not None:
            from . import handling as _hq

            out += _hq.constraint_labels(self.handling_level)
        return out

    @property
    def n_constraints(self) -> int:
        return len(self.constraint_labels)

    @property
    def b_r(self) -> float:
        """The rear wing's span [m] — its own where it has one, else the
        front's (tandem.TandemProblem.b_r's twin)."""
        return float(self.b if self.b_rear is None else self.b_rear)

    # ---------------------------------------------------------- the vector

    @property
    def param_labels(self) -> tuple:
        labels = ["taper_front", "taper_rear",
                  "twist_root_front_deg", "twist_tip_front_deg",
                  "twist_root_rear_deg", "twist_tip_rear_deg",
                  "area_split_front", "decalage_deg"]
        if self.winglets:
            labels += ["winglet_h_front", "winglet_cant_front_deg",
                       "winglet_h_rear", "winglet_cant_rear_deg"]
        # ...then the PAIR'S OWN CANT, where it is searched. LAST in the
        # family block and ahead of the size, flight and chord blocks, all
        # three of which are read BACKWARD from the end of the vector
        # (sizing.resolve_spans, geometry.flight_from_x, the chord slices in
        # unpack), so no existing index moves.
        labels += [lbl for attr, lbl, _b in self.CANT_ROWS
                   if getattr(self, attr)]
        from .sizing import size_labels
        return (tuple(labels) + size_labels(self.size_free, n_spans=2)
                + geometry.flight_labels(self.flight_free)
                + tuple(f"chord_front_k{j}"
                        for j in range(1, self.chord_order + 1))
                + tuple(f"chord_rear_k{j}"
                        for j in range(1, self.chord_order + 1)))

    def _cant_box(self) -> tuple:
        """The pair's tip-device cant band: the published one, or the
        caller's, refused outside the physical VLM range.

        The twin of ``wingtail.WingTailProblem._cant_box`` and validated
        against the same unsigned limits — a pair's wings are lifting
        surfaces, so their devices point the way a wing's does.
        """
        if self.winglet_cant_bounds is None:
            return WINGLET_CANT_BOUNDS_DEG
        lo, hi = (float(v) for v in self.winglet_cant_bounds)
        cmin, cmax = geometry.WINGLET_CANT_LIMITS_DEG
        if not (cmin <= lo < hi <= cmax):
            raise ValueError(
                f"winglet cant_bounds {(lo, hi)} outside the physical VLM "
                f"range {geometry.WINGLET_CANT_LIMITS_DEG}")
        return (lo, hi)

    @property
    def bounds(self) -> np.ndarray:
        rows = [geometry.TAPER_BOUNDS, geometry.TAPER_BOUNDS,
                geometry.TWIST_ROOT_BOUNDS_DEG, geometry.TWIST_TIP_BOUNDS_DEG,
                geometry.TWIST_ROOT_BOUNDS_DEG, geometry.TWIST_TIP_BOUNDS_DEG,
                SPLIT_BOUNDS, DECALAGE_BOUNDS_DEG]
        if self.winglets:
            cant = self._cant_box()
            rows += [
                geometry.winglet_h_row(self.winglet_h_bounds_front),
                cant,
                geometry.winglet_h_row(self.winglet_h_bounds_rear),
                cant]
        rows += [getattr(geometry, band) for attr, _l, band in self.CANT_ROWS
                 if getattr(self, attr)]
        from .sizing import with_size_bounds, ws_of
        box = with_size_bounds(np.array(rows, dtype=float), self.size_free,
                               self.b, self.S_total,
                               span_bounds_m=self.span_bounds_m, n_spans=2,
                               area_bounds_m2=self.area_bounds_m2,
                               ws0=ws_of(self.CL_target, self.rho, self.V),
                               ws_bounds_pa=self.ws_bounds_pa,
                               wing_loading_max_Pa=self.wing_loading_max_Pa)
        box = geometry.with_flight_bounds(box, self.flight_free)
        chord = geometry.chord_bounds(self.chord_order, self.chord_max_frac,
                                      self.chord_law)
        if chord.size:
            box = np.vstack([box, chord, chord])   # front, then rear
        return box

    @property
    def dim(self) -> int:
        return int(self.bounds.shape[0])

    def unpack(self, x: np.ndarray) -> dict:
        """Design vector -> named values (the ONE place the layout is read)."""
        x = np.asarray(x, dtype=float)
        out = {
            "taper_f": float(x[0]), "taper_r": float(x[1]),
            "tw_root_f": float(x[2]), "tw_tip_f": float(x[3]),
            "tw_root_r": float(x[4]), "tw_tip_r": float(x[5]),
            "split": float(x[6]), "decalage": float(x[7]),
        }
        i = 8
        if self.winglets:
            out["h_f"], out["cant_f"] = float(x[i]), float(x[i + 1])
            out["h_r"], out["cant_r"] = float(x[i + 2]), float(x[i + 3])
            i += 4
        else:
            out["h_f"] = out["h_r"] = 0.0
            out["cant_f"] = out["cant_r"] = 90.0
        # the pair's cant, stated or searched, read in ONE place so that
        # everything downstream takes it from the same key whichever way the
        # question was answered
        out["wing_dihedral_deg"] = float(self.wing_dihedral_deg)
        out["wing_sweep_deg"] = float(self.wing_sweep_deg)
        for attr, lbl, _band in self.CANT_ROWS:
            if getattr(self, attr):
                out[lbl] = float(x[i]); i += 1
        m = int(self.chord_order)
        out["coeffs_f"] = geometry.ChordCoeffs(x[-2 * m:-m],
                                               self.chord_law) if m else ()
        out["coeffs_r"] = geometry.ChordCoeffs(x[-m:],
                                               self.chord_law) if m else ()
        return out


def evaluate_tandem_vlm(x: np.ndarray, prob: TandemVLMProblem | None = None
                        ) -> dict:
    """Full evaluation with breakdown; penalty contract on failure."""
    prob = prob or TandemVLMProblem()
    x = np.asarray(x, dtype=float)
    bnds = prob.bounds
    if x.shape != (bnds.shape[0],):
        return _fail(f"design vector shape {x.shape} != ({bnds.shape[0]},)")
    if np.any(x < bnds[:, 0] - 1e-12) or np.any(x > bnds[:, 1] + 1e-12):
        return _fail("bounds violation")

    v = prob.unpack(x)
    m2 = 2 * int(prob.chord_order)

    # ---- size: the pair's span and TOTAL area. The stagger stays put: it is
    # the layout the user stated (see the dx/dz fields)
    size_req = bool(prob.size_free)
    W_fixed_ref = None
    if size_req:
        from .mission import weight_for as _weight_for0
        from .sizing import (WS_MODES, check_ar, flow_state_for,
                             resolve_spans, size_mode)
        # the payload weight is the SAME CONSTANT for every candidate — that
        # is what makes payload L/D = W_fixed/D a minimum-drag objective
        # (sizing.py). Read from the problem's OWN published area, BEFORE the
        # candidate size replaces it: taken afterwards it scaled with the
        # area, so a bigger wing carried a bigger "fixed" payload and the
        # score rewarded growing the wing.
        W_fixed_ref = (prob.W_fixed_N if prob.W_fixed_N is not None
                       else _weight_for0(prob.CL_target, prob.V,
                                         prob.S_total))
        size_kw: dict = {"wing_loading_max_Pa": prob.wing_loading_max_Pa}
        if size_mode(prob.size_free) in WS_MODES:
            # the TOTAL area follows the loading, closed against the q this
            # candidate flies at (the flight block needs no area, so it is
            # read first) and split by the split this candidate flies — the
            # planar pair's own path, same order, same reasons
            rho_ws, V_ws = flow_state_for(
                x, mission=prob.mission, flight_free=prob.flight_free,
                n_trailing=m2, rho=prob.rho, V=prob.V)
            size_kw.update(
                # ignored by the SEARCHED-loading mode, which reads its own
                # row out of the vector (sizing.resolve_spans)
                wing_loading_Pa=(prob.wing_loading_Pa
                                 if prob.wing_loading_Pa is not None
                                 else prob.CL_target * 0.5 * prob.rho
                                 * prob.V ** 2),
                W_fixed_N=W_fixed_ref,
                taper=0.5 * (v["taper_f"] + v["taper_r"]), tc=0.12,
                q_Pa=0.5 * rho_ws * V_ws ** 2,
                area_fracs=(v["split"], 1.0 - v["split"]))
        try:
            (b_front, b_rear), S_use = resolve_spans(
                x, prob.size_free, prob.flight_free, m2, n_spans=2, **size_kw)
        except (ValueError, OverflowError) as exc:
            return _fail(f"size: {exc}")
        for b_use in (b_front, b_rear):            # ONE wing's aspect ratio,
            reason = check_ar(b_use, 0.5 * S_use)  # asked of each wing
            if reason is not None:
                return _fail(f"size: {reason}")
    # ...and the ASPECT-RATIO LIMIT THE USER SET. A different question
    # from the validity band, so it is asked whether or not the size is
    # a design variable: a limit that only applied to searched sizes
    # would go quiet exactly when the user typed the number it is about.
    if prob.ar_limits is not None:
        from .sizing import check_ar_limit as _check_ar_limit
        # per WING, as the validity gate above is: the pair's total area
        # reports half the aspect ratio either surface actually flies at
        S_pair = float(S_use if prob.size_free else prob.S_total)
        for b_wing in ((b_front, b_rear) if prob.size_free
                       else (float(prob.b), prob.b_r)):
            why_ar = _check_ar_limit(b_wing, 0.5 * S_pair, prob.ar_limits)
            if why_ar is not None:
                return _fail(f"size: {why_ar}")
    if prob.size_free:
        prob = replace(prob, b=b_front, b_rear=b_rear, S_total=S_use,
                       size_free=False)

    # ---- flight state
    flight = geometry.flight_from_x(x, prob.flight_free, m2)
    if flight is not None:
        from .mission import flight_state
        try:
            prob = replace(prob, flight_free=False, mission=None,
                           **flight_state(prob.mission, flight[0], flight[1],
                                          prob.S_total))
        except ValueError as exc:
            return _fail(f"flight state: {exc}")

    # ---- the two tip devices are TWO devices. Checked before anything is
    # built, because a pair whose devices meet is a different aircraft and
    # the panel method would answer for it silently (see the helper)
    reason = _device_clearance(prob.b, prob.dz, v["h_f"], v["cant_f"],
                               v["h_r"], v["cant_r"], b_rear=prob.b_r,
                               blend_frac=prob.blend_frac_fixed,
                               blend_shape=prob.blend_shape,
                               wing_blend_frac=prob.wing_blend_frac)
    if reason is not None:
        return _fail(reason)

    S_f = v["split"] * prob.S_total
    S_r = prob.S_total - S_f
    try:
        # the cant and sweep reach BOTH wings: they are a statement about
        # how the aeroplane is built, not about one surface — the same rule
        # the chord limits and the blend already follow here
        wing_f = geometry.Wing(b=prob.b, S=S_f, taper=v["taper_f"],
                               twist_root_deg=v["tw_root_f"],
                               twist_tip_deg=v["tw_tip_f"],
                               sweep_deg=v["wing_sweep_deg"],
                               dihedral_deg=v["wing_dihedral_deg"],
                               chord_limits=prob.chord_limits,
                               chord_coeffs=v["coeffs_f"])
        wing_r = geometry.Wing(b=prob.b_r, S=S_r, taper=v["taper_r"],
                               twist_root_deg=v["tw_root_r"] + v["decalage"],
                               twist_tip_deg=v["tw_tip_r"] + v["decalage"],
                               sweep_deg=v["wing_sweep_deg"],
                               dihedral_deg=v["wing_dihedral_deg"],
                               chord_limits=prob.chord_limits,
                               chord_coeffs=v["coeffs_r"])
    except ValueError as exc:
        return _fail(f"planform: {exc}")

    sized = None
    CL_trim = prob.CL_target
    if size_req:
        from .sizing import sized_state
        try:
            sized = sized_state(
                W_fixed_N=W_fixed_ref,
                b=prob.b, S=prob.S_total,
                taper=0.5 * (v["taper_f"] + v["taper_r"]),
                tc=wing_f.tc, q_Pa=0.5 * prob.rho * prob.V**2,
                wing_areas=(S_f, S_r), wing_spans=(prob.b, prob.b_r),
                wing_loading_max_Pa=prob.wing_loading_max_Pa,
                material=prob.material)
        except (ValueError, RuntimeError, OverflowError) as exc:
            return _fail(f"size: {exc}")
        CL_trim = sized.CL_target

    pol = prob.section_polar if prob.section_polar is not None else prob.polar
    pol_r = pol if prob.polar_rear is None else prob.polar_rear
    try:
        # THE SURFACE THE PAIR IS ALREADY CHARGED FOR, in the lattice that
        # scores it — but only where the lateral half was asked for. A fin
        # perturbs the symmetric solve in the last ulp, so an unasked-for one
        # would move every published L/D by a rounding. Sized by the ONE fin
        # law and stationed by the ONE station rule
        # (``fin.tandem_fin_station``), which is what the drag charge below,
        # ``api.design_report``, the loft and the rebuild all use — so the
        # surface flown here is the surface flown everywhere else.
        vertical = None
        if prob.lateral and _fin.has_fin(prob):
            arm_v, z_root_v = _fin.tandem_fin_station(prob.dx, prob.dz,
                                                      prob.fin_boom_m)
            g_v = _fin.size_fin(b=prob.b, S=prob.S_total, l_t=arm_v,
                                tail_type="conventional", z_root=z_root_v,
                                **_fin.fin_law_kwargs(prob))
            if g_v is not None:
                vertical = VerticalSurface(height=g_v.height,
                                           chord=g_v.chord, x=g_v.x_qc,
                                           z_root=g_v.z_root, N=12)
        model = VLM(
            wing_f, N=prob.N, winglet_h_frac=v["h_f"],
            winglet_cant_deg=v["cant_f"], n_winglet=prob.n_winglet,
            winglet_blend_frac=prob.blend_frac_fixed,
            winglet_blend_shape=prob.blend_shape,
            winglet_wing_blend_frac=prob.wing_blend_frac,
            winglet_chord_follows=prob.winglet_chord_follows,
            a=pol.a_lin, alpha_L0=pol.alpha_L0, V=prob.V, S_ref=prob.S_total,
            second=SecondWing(wing=wing_r, x=prob.dx, z=prob.dz, N=prob.N,
                              winglet_h_frac=v["h_r"],
                              winglet_cant_deg=v["cant_r"],
                              n_winglet=prob.n_winglet,
                              winglet_blend_frac=prob.blend_frac_fixed,
                              winglet_blend_shape=prob.blend_shape,
                              winglet_wing_blend_frac=prob.wing_blend_frac,
                              a=pol_r.a_lin, alpha_L0=pol_r.alpha_L0),
            vertical=vertical)
        lo, hi = (np.deg2rad(d) for d in prob.alpha_bracket_deg)
        alpha, res = model.solve_trim(CL_trim, alpha_bracket=(lo, hi))
    except (ValueError, np.linalg.LinAlgError) as exc:
        return _fail(f"solver failure: {exc}")

    alpha_eff_deg = np.rad2deg(res.alpha_eff)
    rear = model.is_second
    # THE FIN IS NOT A LIFTING SURFACE OF THIS PAIR, and the lattice does not
    # know that. Where the lateral deck put a vertical in the solve, its
    # panels must stay out of the horizontal book-keeping, for the two
    # reasons wingtail.py records: its parasite drag has exactly one author
    # (``cd0_fin`` below), so counting its panel area again would charge the
    # same surface twice; and a symmetric fin at zero sideslip flies at
    # c_l ~ 0, which is a fact about the trim state and not about the
    # section's validity range — letting it into the alpha gate makes a
    # surface that is doing nothing able to refuse a design. With no vertical
    # this mask is all-True and every line below is the one it always was.
    solid = ~np.asarray(getattr(model, "is_vertical",
                                np.zeros(alpha_eff_deg.shape, bool)), bool)
    ae_all = alpha_eff_deg[solid]
    # each wing is judged against ITS OWN polar's validity range; with one
    # section they are the same range, so this is the previous test verbatim
    lo_p, hi_p = pol.alpha_valid
    lo_r, hi_r = pol_r.alpha_valid
    ae_f, ae_r = alpha_eff_deg[solid & ~rear], alpha_eff_deg[solid & rear]
    if (ae_f.min() < lo_p or ae_f.max() > hi_p
            or (ae_r.size and (ae_r.min() < lo_r or ae_r.max() > hi_r))):
        return _fail(
            "effective AoA outside polar validity (stall/extrapolation proxy)",
            alpha_eff_range=(float(ae_all.min()), float(ae_all.max())))

    strip = np.where(rear, pol_r.cd(alpha_eff_deg),
                     pol.cd(alpha_eff_deg)) * res.c * res.width
    CDp_f = float(np.sum(strip[solid & ~rear]) / prob.S_total)
    CDp_r = float(np.sum(strip[solid & rear]) / prob.S_total)
    # the corners where a blended device leaves its wing — one report per
    # wing, each on ITS OWN tip chord and semi-span (a pair may have two
    # spans, so a single charge would quote the wrong chord for one of them)
    jr_f = jr_r = None
    if prob.junction_drag and prob.winglets:
        from . import junction
        for key, wg, h_frac, cant in (("f", wing_f, v["h_f"], v["cant_f"]),
                                      ("r", wing_r, v["h_r"], v["cant_r"])):
            if float(h_frac) <= 0.0:
                continue
            rep = junction.report(
                tc=wg.tc,
                chord=float(wg.chord(np.array([wg.b / 2.0]))[0]),
                s_ref=prob.S_total, h=float(h_frac) * wg.b / 2.0,
                cant_deg=float(cant), blend_frac=prob.blend_frac_fixed,
                n_junctions=2, blend_shape=prob.blend_shape,
                wing_arc=prob.wing_blend_frac * wg.b / 2.0)
            if key == "f":
                jr_f = rep
            else:
                jr_r = rep
    CD_junction = float((jr_f or {}).get("CD_junction", 0.0)) \
        + float((jr_r or {}).get("CD_junction", 0.0))
    # THE FIN'S PARASITE DRAG, on the arm the surface actually stands at
    # (``fin.tandem_fin_station``: on a boom aft of the rear wing). One law
    # (``fin.size_fin``) sizes the surface the report states, the loft
    # exports and the rebuild flies, so charging it here is what makes those
    # four the same surface — see tandem.evaluate_tandem, which does the
    # identical thing for the lifting-line pair.
    cd0_fin = 0.0
    if _fin.has_fin(prob):
        from . import tail as _tail       # see tandem.evaluate_tandem
        try:
            arm, _z = _fin.tandem_fin_station(prob.dx, prob.dz,
                                              prob.fin_boom_m)
            cd0_fin = float(_tail.vtail_cd0(
                prob.S_total, arm, b=prob.b, S=prob.S_total,
                V=prob.V, rho=prob.rho, mu=prob.mu,
                **_fin.fin_drag_kwargs(prob)))
        except ValueError:
            cd0_fin = 0.0
    CD = res.CDi + CDp_f + CDp_r + prob.cd0_extra + CD_junction + cd0_fin
    LoD = res.CL / CD
    if not (np.isfinite(LoD) and CD > 0.0):
        return _fail("non-finite L/D")

    CL_f = 2.0 * float(res.Gamma[~rear] @ res.ly[~rear]) / (prob.V * S_f)
    CL_r = 2.0 * float(res.Gamma[rear] @ res.ly[rear]) / (prob.V * S_r)
    S_wl = float(np.sum((res.c * res.width)[res.is_winglet]))

    out = {
        "feasible": True, "reason": "", "score": float(LoD),
        "LoD": float(LoD), "CL_total": float(res.CL),
        "CL_target": float(CL_trim),
        "CL_front": CL_f, "CL_rear": CL_r,
        "lift_share_front": float(CL_f * S_f / (res.CL * prob.S_total))
        if res.CL else float("nan"),
        "CDi_total": float(res.CDi), "CDi": float(res.CDi),
        "CDp": float(CDp_f + CDp_r), "CDp_front": CDp_f, "CDp_rear": CDp_r,
        "cd0_extra": prob.cd0_extra, "CD_junction": CD_junction,
        "cd0_fin": float(cd0_fin),
        "CD": float(CD),
        "e": float(res.e), "AR": float(res.AR),
        "alpha_deg": float(np.rad2deg(alpha)),
        "alpha_eff_range_deg": (float(ae_all.min()), float(ae_all.max())),
        "S_front": float(S_f), "S_rear": float(S_r),
        # ``b`` is the FRONT wing's span (every reader means that by it); the
        # two named keys say what a pair with two spans actually flew
        "Sref": float(prob.S_total), "b": float(prob.b),
        "b_front": float(prob.b), "b_rear": float(prob.b_r),
        "dx": float(prob.dx), "dz": float(prob.dz),
        # so a rebuild FLIES what was scored (flightmodel.build_flight_model)
        "wing_dihedral_deg": float(v["wing_dihedral_deg"]),
        "sweep_deg": float(v["wing_sweep_deg"]),
        "decalage_deg": float(v["decalage"]),
        "Re_mac": float(prob.rho * prob.V * wing_f.mac / prob.mu),
        "Re_mac_rear": float(prob.rho * prob.V * wing_r.mac / prob.mu),
        "polar": getattr(pol, "name", "unknown"),
        "polar_rear": getattr(pol_r, "name", "unknown"),
        "wing": wing_f, "wing_rear": wing_r, "vlm": res,
        "tc": float(wing_f.tc),
        # THE POINT THIS PAIR WAS FLOWN AT, on every exit. Written only under
        # the flight modifier before, so a report from every other run stated
        # no speed — and ``flightmodel.build_flight_model``, with nothing to
        # read, fell back to 45 m/s and back-solved a mass there. The same
        # defect wingtail.py records, in the other family.
        "V": float(prob.V), "rho": float(prob.rho), "mu": float(prob.mu),
        # ...AND WHERE THE MOMENTS ARE TAKEN. ``x_cg`` is the stated
        # reference (the front wing's quarter chord); ``x_lift`` below is
        # where this candidate's lift actually acts. The second is REPORTED
        # and not scored: a tandem is trimmed in lift alone, so it has no
        # balance point the solve finds, and reading the reference off the
        # candidate would let the lateral criterion be bought with the area
        # split instead of with the cant.
        "x_cg": float(prob.x_cg),
        "mac": float(prob.S_total / prob.b),
    }
    if prob.lateral:
        # the deck is linearised about the state that was just TRIMMED, not
        # about zero: Cl_r and Cn_p go as the base CL and do not exist on an
        # unloaded pair (dynamics.deck). ``i_t`` stays 0.0 and means it — the
        # pair has no elevator, its longitudinal trim IS the decalage, and
        # that is already in the geometry the lattice was built from.
        dk = dyn.deck(model, prob.x_cg, out["mac"], prob.b, alpha=alpha)
        out["Cl_beta"] = float(dk.Cl_beta)
        out["Cn_beta"] = float(dk.Cn_beta)
        out["Cl_r"] = float(dk.Cl_r)
        out["Cn_r"] = float(dk.Cn_r)
        out["Cl_p"] = float(dk.Cl_p)
        out["Cn_p"] = float(dk.Cn_p)
        # the attitude the criterion was evaluated at, stated beside it: the
        # classical theta0 = 0 form crosses early, so a margin quoted without
        # its attitude cannot be reproduced (dynamics.spiral_margin)
        out["spiral_theta0_deg"] = float(np.rad2deg(dk.alpha_ref))
        out["spiral_margin"] = dyn.spiral_margin_of(dk)
        gam_w = res.Gamma * res.ly
        tot = float(np.sum(gam_w))
        out["x_lift"] = (float(np.sum(gam_w * model.st3[:, 0]) / tot)
                         if tot != 0.0 else None)
    if prob.winglets:
        out["winglet"] = {
            "h_frac_front": v["h_f"], "cant_front_deg": v["cant_f"],
            "h_frac_rear": v["h_r"], "cant_rear_deg": v["cant_r"],
            # the heights that reached the lattice: the h_frac pair above
            # stays REQUESTED, so each surface reads asked / flown
            "h_m_front": float(model.winglet_h_m),
            "h_m_rear": float(model.winglet_h_m_second),
            "S_planform": S_wl,
            "blend_frac": float(prob.blend_frac_fixed),
            "blend_shape": str(prob.blend_shape),
            "wing_blend_frac": float(prob.wing_blend_frac),
            **({"junction_front": jr_f} if jr_f else {}),
            **({"junction_rear": jr_r} if jr_r else {}),
        }
    if flight is not None:
        out["altitude_m"] = float(flight[1])
    if sized is not None:
        q = 0.5 * prob.rho * prob.V**2
        f = sized.payload_lod(q, CD)
        if not np.isfinite(f):
            return _fail("non-finite payload L/D")
        out.update(sized.report())
        out.update({"f": float(f), "score": float(f),
                    "g": float(sized.g_sigma),
                    "D_N": float(q * sized.S * CD)})
    if prob.lateral and prob.handling_level is not None:
        # THE MODES, ON THE LATTICE THAT WAS SCORED — the wing+tail's
        # own block (``wingtail.evaluate_wing_tail``), reading this
        # family's numbers. Everything the gate needs is already in
        # hand: the deck above, which ``handling_level`` armed through
        # the same switch the spiral WEIGHT uses, and this pair's own
        # weight, span efficiency and profile drag.
        from . import handling as _hq

        _vert = getattr(model, "vertical", None)
        # THE WEIGHT THIS PAIR FLIES AT, from the design and never from
        # a default. Sized, it is the weight loop's own total; unsized,
        # the pair is trimmed to ``CL_trim`` on its OWN reference area,
        # so lift equals weight there by construction — taken from the
        # solve rather than from a dict, the same identity and the same
        # reason as on the wing+tail family.
        _W = (float(sized.W_total_N) if sized is not None
              else float(CL_trim * 0.5 * prob.rho * prob.V ** 2
                         * prob.S_total))
        # HOW LONG THE AEROPLANE IS: a pair owns no fuselage, so its
        # length is the boom — the station the fin actually stands at
        # (``fin.tandem_fin_station``), which is the same number the
        # drag charge and the report read. ``drag.body_length_for_arm``
        # is the wing+tail's answer to the same question and takes the
        # same argument, so the two families measure inertia off one
        # rule.
        from . import drag as _drag_mod

        _arm_hq, _ = _fin.tandem_fin_station(prob.dx, prob.dz,
                                             prob.fin_boom_m)
        hq = _hq.gate(
            dk, prob.handling_level,
            mass_kg=_W / 9.80665, b_m=float(prob.b),
            body_length_m=float(
                _drag_mod.body_length_for_arm(_arm_hq)),
            rho=float(prob.rho), CD0=float(CDp_f + CDp_r),
            oswald_e=float(res.e),
            AR_vertical=(None if _vert is None else
                         float(abs(_vert.height) ** 2
                               / max(abs(_vert.height) * _vert.chord,
                                     1e-12))),
            breakdown=out)
        out.update(hq.report())
        out["handling_margins"] = hq
        # ...AND ONTO THE CONSTRAINT VECTOR, LAST. Whatever this problem
        # already declared keeps its index, so the size modifier's
        # stress margin is still ``g[0]`` for every caller that reads it
        # (``constraint_labels``, ``fg_tandem_vlm``). An unsized pair had
        # no ``g`` at all — it is an unconstrained problem — so the rows
        # ARE the vector there.
        out["g"] = (np.concatenate(
            [np.atleast_1d(np.asarray(out["g"], dtype=float)), hq.g])
            if "g" in out else hq.g)
    if wing_f.chord_coeffs or wing_r.chord_coeffs:
        out["chord_coeffs"] = list(wing_f.chord_coeffs)
        out["chord_coeffs_rear"] = list(wing_r.chord_coeffs)
    return out


def objective_tandem_vlm(x: np.ndarray,
                         prob: TandemVLMProblem | None = None) -> float:
    """Scalar objective (MAXIMISE): system L/D, or PENALTY on any failure."""
    out = evaluate_tandem_vlm(x, prob)
    return float(out["score"]) if out["feasible"] else PENALTY


def fg_tandem_vlm(x: np.ndarray, prob: TandemVLMProblem | None = None
                  ):
    """Constrained-harness callable.

    With the SIZE modifier alone: (payload L/D, signed root-bending stress
    margin) — one scalar margin, exactly as this family has always returned.
    ``handling_level`` adds :data:`handling.N_ROWS` more rows to whatever
    that was, so an unsized pair with a level returns five and a sized one
    six.

    The FAILURE vector follows the same width, and it comes from
    ``prob.constraint_labels`` — the same declaration ``api`` reads — never
    from a literal counted out here: a penalty returned at the wrong width is
    a shape error inside the optimiser rather than a bad score, and it must
    not reach one silently. A scalar ``g`` is kept ONLY in the case that has
    always returned one, so no existing caller sees its margin become a
    length-1 array (``wingtail.fg_wing_tail`` is the twin of this rule).
    """
    n_g = len(prob.constraint_labels) if prob is not None else 1
    out = evaluate_tandem_vlm(x, prob)
    if not out["feasible"]:
        return (PENALTY, G_FAIL) if n_g == 1 else (PENALTY,
                                                   np.full(n_g, G_FAIL))
    if "g" not in out:
        raise ValueError(
            "fg_tandem_vlm() is for a CONSTRAINED variant — the size "
            "modifier, a handling level, or both. This problem declares no "
            "margins, so its harness callable is objective_tandem_vlm()")
    g = out["g"]
    if n_g == 1:
        return float(out["score"]), float(g)
    g = np.atleast_1d(np.asarray(g, dtype=float))
    if g.size != n_g:
        raise ValueError(
            f"this problem declares {n_g} margins "
            f"{prob.constraint_labels} but its evaluation returned "
            f"{g.size} — a constraint vector whose width does not match its "
            f"own declaration is a shape error in the harness, not a bad "
            f"design, and it must not reach the optimiser silently")
    return float(out["score"]), g
