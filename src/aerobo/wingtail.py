"""Wing + tip device + tail in ONE nonplanar solve — the aeroplane problem.

Why this module exists
----------------------
``tail.py`` couples a wing and a tail through tandem.py's planar lifting
lines: excellent for the layout question it was built for, but every surface
in it is horizontal, so a WINGLET cannot be represented at all. The winglet
problems, conversely, run the nonplanar VLM (vlm.py) and have no tail, so
they are trimmed in lift only and say nothing about stability. Between them
sat the thing anybody designing an aeroplane actually wants: shape the wing
— its planform, its chord law, its section, its tip device — WITH the tail
that has to trim and stabilise it.

That gap is closed here by putting the tail INTO the nonplanar solve
(vlm.TailSurface) rather than bolting a second solver on:

* one influence matrix over wing + winglet + tail panels, so the wing-on-tail
  downwash, the tail's upwash back-reaction and the winglet's own effect on
  both come OUT of the solve. There is no d(eps)/d(alpha) model anywhere.
* induced drag in the TREFFTZ plane over the combined wake (Munk's stagger
  theorem: the streamwise placement drops out), so the tail's induced drag
  and its interference with the wing's wake are counted once, together.
* trim in (alpha, i_t) on (CL, Cm) as a closed-form 2x2 solve, because the
  linearised boundary condition keeps both exactly affine in the two angles
  (vlm.VLM.solve_trim_moment) — the same elimination tail.py does, with the
  Jacobian read off the basis solutions instead of finite differences.
* static margin from the SAME basis solutions: x_np is the lift-weighted
  station of the alpha-derivative, SM = (x_np - x_cg)/mac, and
  Cm_alpha = -CL_alpha * SM identically (gated).

Everything about the tail's LAYOUT — conventional / T-tail / V-tail / canard,
the calibrated CG per layout, the elevator-vs-stabilator control gate, the
fin drag build-up — is imported from tail.py, not restated, so the two
modules cannot describe different aeroplanes.

What is new here as a DESIGN FREEDOM
------------------------------------
* the tip device (height, cant, and optionally the blended transition) beside
  the tail, span-capped or free exactly as the winglet problems define it;
* the tail's VERTICAL distance from the wing plane, ``z_t``, as a design
  variable rather than a fixed 0.05 b. Raising the tail lifts it out of the
  wing's trailing sheet and it sees less downwash — an INVISCID effect only.
  The wake here is rigid, planar and inviscid: no dynamic-pressure deficit,
  no roll-up, no deep-stall pitch-up, all of which are real considerations
  for a high tail and none of which this model can speak to (the same
  honesty label tail.py's T-tail carries). The box OPENS on DZ_FRAC*b and is
  not bounded by it: any band the user types is the band searched, down to a
  tail in the wing plane (tail.height_row). What DZ_FRAC*b buys is the
  published comparison, and what is genuinely fragile is much lower —
  measured here, the L/D spread over three grids is 0.098 % at 0.05 b, still
  0.273 % at 0.01 b, and 6.6 % at zero (tail.DZ_GRID_FRAC).
* the section: thickness t/c off the NACA 24XX polar family (``tc_free``),
  composed freely with the above;
* the chord law, inherited from geometry.py like every other family.

Design vector (blocks, in this order — the chord law LAST, package rule)
-----------------------------------------------------------------------
    [taper, twist_root_deg, twist_tip_deg, S_t_m2]
    [+ l_t_m           unless the arm is fixed by a flag]
    [+ z_t_m           if the tail height is free]
    [+ winglet_h_frac, winglet_cant_deg      if a tip device is carried]
    [+ winglet_blend_frac                    if the transition is designed]
    [+ tc                                    if the section thickness is free]
    [+ chord_k1..k3                          if the chord law is free]

Relation to the published problems
----------------------------------
This is a DIFFERENT aero core from tail.py (Weissinger VLM vs coupled
lifting lines), so its numbers are its own. MEASURED with no winglet over a
108-point (taper, S_t, l_t, twist) grid: L/D within 2.4 % (mean 0.8 %),
static margin within 0.072 (mean 0.026, this solver always the lower — its
tail lift-curve slope at AR 4 sits below the lifting line's, which is where
a Weissinger method is the better of the two), trim angles within 0.3 deg in
alpha and 0.9 deg in i_t, and the two agree on FEASIBILITY at every point.
Close enough to be the same aeroplane, far enough apart that mixing the two
in one comparison would be dishonest. tests/test_wingtail.py measures and
gates that agreement; ``tail`` / ``tail (fixed arm)`` keep their published
LLT solver untouched.

Failure contract: the package's. evaluate_wing_tail never raises for
in-contract failures; fg_wing_tail returns exactly (PENALTY, G_FAIL) on
solver failure or an untrimmable design, and a solvable-but-unstable design
returns its TRUE L/D with a signed negative margin.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from . import dynamics as dyn
from . import fin as _fin
from . import geometry, junction, tail as tailmod
from .objective import MU_SL, PENALTY, RHO_SL, WINGLET_TYPES, _fail
from .polar import default_polar, default_polar_family
from .polar import InvertedPolar as _InvertedPolar
from .polar import section_cm_ac as _section_cm_ac
from .vlm import VLM, TailSurface, VerticalSurface

G_FAIL = tailmod.G_FAIL

#: tail height above the wing plane, as a FRACTION of the span, when it is a
#: design variable — the box this row OPENS on, and not a limit on it
#: (tail.height_row carries the user's own band into the problem). The low
#: end is tail.DZ_FRAC, the height every published tail here was measured
#: at; the high end is 0.30 b, past which the rigid-wake assumption is being
#: asked to describe a surface in a different flow regime, and where a fin
#: that tall is really a T-tail whose height its own sizing derives
#: (tail.tail_height).
Z_T_FRAC_BOUNDS = (tailmod.DZ_FRAC, 0.30)

#: aspect-ratio box for a DESIGNED tail. The published surface is fixed at
#: tail.TAIL_AR = 4, and the conventional H-tail range is AR_t ~ 3-5
#: (Raymer, tail-geometry guidance); the box is opened wider than that on
#: both sides so the constraint is the physics and not the box — a stubby
#: AR 2 surface (cheap to build, poor lift slope) and a slender AR 8 one
#: (efficient, structurally awkward) are both reachable, and the optimiser
#: has to trade span efficiency against the wetted area it costs.
TAIL_AR_BOUNDS = (2.0, 8.0)

#: washout box for a designed tail [deg]: the TIP twist relative to the
#: root. Only the DIFFERENCE is a design variable — the uniform part of a
#: tail's incidence IS the trim unknown i_t, so carrying a root twist as
#: well would give the search a flat direction (two variables, one effect).
TAIL_WASHOUT_BOUNDS_DEG = (-6.0, 2.0)


@dataclass
class WingTailProblem:
    """Wing (+ tip device) + tail, nonplanar, trimmed in lift AND pitch.

    Objective (MAXIMISE): L/D at the 2-D trim point.
    Constraint: g = (SM - SM_min)/1.0 >= 0.

    Which design variables exist is fixed at construction (``winglet``,
    ``blended``, ``tc_free``, ``free_height``, ``l_t_fixed``,
    ``chord_order``) because the dimension is part of the problem's identity
    — the package registers one ProblemSpec per combination rather than
    letting a flag change the vector's length underneath a static label list.
    """

    # ---- wing size (api.PLANFORM_KEYS may choose these)
    b: float = 10.0
    S: float = 10.0
    # ---- resolution
    N_vlm: int = 40            # wing panels (full span)
    n_winglet: int = 8         # panels per winglet
    N_t: int = 24              # tail panels
    # ---- flow state / references
    V: float = 14.6
    rho: float = RHO_SL
    mu: float = MU_SL
    CL_target: float = 0.5
    cd0_extra: float = 0.0
    #: THE BODY, and it is the same question ``tail.TailProblem`` asks — see
    #: its note for the measurement. This family DECLARED the flag in the
    #: registry (``api.FUSELAGE_KEYS`` is on every wing+tail spec) and could
    #: not take it: ``check_flags`` accepted ``fuselage_diameter_m`` and the
    #: build then raised ``TypeError``, so the one configuration that both
    #: searches an arm AND can carry a wing dihedral was the one that could
    #: not charge a body for the arm it was searching.
    #:
    #: None charges nothing, which is every published nonplanar run
    #: bit-for-bit.
    fuselage_diameter_m: float | None = None
    polar: Any = field(default_factory=default_polar)
    #: the TAIL's own section (tail.TailProblem.polar_tail's twin). None
    #: keeps the documented "the tail reuses the wing polar" default, so
    #: every published wing+tail run is bit-for-bit unchanged.
    polar_tail: Any = None
    polar_family: Any = None
    # ---- wing design freedoms (each changes the design vector)
    winglet: bool = False
    capped: bool = False        # winglet span accounting
    blended: bool = False       # the root transition is a design variable
    tc_free: bool = False       # section thickness off the polar family
    #: THE WING'S DIHEDRAL AS A DESIGN VARIABLE (row ``wing_dihedral_deg``),
    #: instead of the stated value below.
    #:
    #: It is the only WING-side source of ``Cl_beta`` this package has that
    #: holds at any lift — without it a design's whole dihedral effect comes
    #: from the fin and the tip device, and the spiral mode of a fin-only
    #: aeroplane cannot be made convergent by sizing that fin (a bigger fin
    #: raises ``Cn_beta`` and ``Cl_beta`` together). Searching it needs a
    #: LATERAL term in the objective — ``wing_score``'s ``spiral`` criterion
    #: — or an L/D-only score puts it at whichever bound costs least
    #: projected span. That is a measurement, not a rule: both bounds stay
    #: reachable, and the run reports which limit it rides.
    dihedral_free: bool = False
    #: THE WING'S QUARTER-CHORD SWEEP AS A DESIGN VARIABLE (row
    #: ``wing_sweep_deg``), instead of the stated value below.
    #:
    #: A SEPARATE freedom, because it is priced against a different thing.
    #: Sweep moves the neutral point aft and costs lift-curve slope
    #: (measured on the wing+tail box centre: 15 deg costs 12.7 % of L/D and
    #: takes x_np from 0.443 to 1.014 m), and its dihedral effect goes as CL
    #: and has all but vanished at cruise — so the spiral argument above is
    #: NOT an argument for this row, and a design that wants a static margin
    #: rather than a roll lever wants this one and not that one.
    sweep_free: bool = False
    #: BOTH, as one word. Kept because it is what every published free-cant
    #: run was built with and what the registry's ``free`` state still
    #: spells; :meth:`__post_init__` expands it into the pair above and
    #: nothing below this line reads it again.
    cant_free: bool = False
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
    #   the surface's own chord law past the tip instead of holding the tip
    #   chord (vlm.py; objective.Problem carries the twin of this field).
    #   False = the rectangular device every published run flew, bit-for-bit.
    #   It reaches the TAIL's tip device too: it is a statement about the
    #   design, not about one surface.
    wing_loading_Pa: float | None = None   # W/S for the wing-loading size
    #                             mode (sizing.SIZE_MODE_WS): the WING's area
    #                             follows it and the vector carries the span
    #                             alone. The tail's own area stays its own
    #                             design variable. None -> the loading this
    #                             problem already flies, CL_target x q.
    span_bounds_m: tuple | None = None     # (min, max) span for that row
    area_bounds_m2: tuple | None = None    # (min, max) AREA row of the free
    #   planform mode [m^2]; None -> the fractional band around this
    #   problem's own reference area (sizing.size_bounds)
    ws_bounds_pa: tuple | None = None      # (min, max) of the SEARCHED
    #   wing-loading row [Pa] (sizing.SIZE_MODE_WS_FREE); None ->
    #   sizing.WS_FRAC_BOUNDS around the loading this problem already flies
    wing_loading_max_Pa: float | None = None   # the MISSION's ceiling on W/S
    #   (constraint_diagram): clips that band and refuses any candidate above
    #   it, in every sized mode
    material: object = None   # WHAT THE WING IS BUILT OF
    #   (materials.Material, or a key from materials.MATERIALS).
    #   None = 2024-T3 aluminium, the material Raymer's correlation
    #   was regressed over, so every published run is bit-identical.
    #   It sets BOTH the allowable the spar is sized to and the
    #   factor the statistical wing weight is scaled by — the second
    #   is the one the span answer actually rides (materials.py).
    size_free: bool | str = False   # SIZE modifier (sizing.py): span and area
    #                             become design variables, the wing weighs
    #                             what its size implies, the trim target
    #                             follows that weight, the score becomes
    #                             payload L/D and a root-bending stress
    #                             margin joins the static margin as a SECOND
    #                             constraint. Documented limit: the Raymer
    #                             weight is the WING's — the tail's own mass
    #                             sits inside W_fixed, so sizing the tail
    #                             moves its drag and its trim, not its weight.
    W_fixed_N: float | None = None    # non-wing weight for the size modifier
    flight_free: bool = False   # speed + altitude are design variables
    #                             (geometry.py's flight modifier): the flow
    #                             state and the trim target CL = W/(q S) are
    #                             rebuilt per candidate from the design
    #                             weight, exactly as in objective.py. The
    #                             STATIC MARGIN constraint is unaffected —
    #                             x_np comes from the same inviscid solve at
    #                             any speed — so what the modifier really adds
    #                             here is the tail sizing at a chosen cruise
    #                             point rather than a fixed one.
    mission: Any = None         # mission.MissionSpec carrying the design
    #                             weight; defaults (flight_free only) to the
    #                             weight this problem's own CL_target implies
    free_height: bool = False   # tail height above the wing plane is free
    z_t_fixed: float | None = None  # ...or STATED [m above the wing plane].
    #                             Only meaningful while free_height is off:
    #                             the two would be one question asked twice,
    #                             so __post_init__ refuses the pair. None
    #                             keeps the LAYOUT's own height
    #                             (tail.tail_height), which is what every
    #                             published run flies.
    l_t_fixed: float | None = None
    l_t_bounds_m: tuple | None = None  # (min, max) the SEARCHED arm is boxed
    #                             by — the user's band, checked against the
    #                             pitch solve's own floor and nothing else
    #                             (tail.arm_row). None -> tail.arm_band(b).
    s_t_bounds_m2: tuple | None = None  # ...and the SEARCHED tail AREA's own
    #                             band [m^2] (tail.area_row). None ->
    #                             tail.area_band(S), a fraction of the wing.
    tail_limits: "tailmod.TailLimits | None" = None   # the SECOND surface's
    #                             own span and chord limits, in metres
    #                             (tail.TailLimits). None = unconstrained.
    z_t_bounds_m: tuple | None = None  # ...and the same for the VERTICAL
    #                             separation when free_height searches it:
    #                             the user's band in metres (tail.height_row).
    #                             None -> Z_T_FRAC_BOUNDS on this span.
    # ---- the TAIL as a designed surface (each changes the design vector).
    # Off, the tail is the published rectangle at AR 4 whose only freedoms
    # are its area and its arm. On, it is a wing in its own right — taper,
    # aspect ratio and washout, its own chord law when the wing carries one,
    # and (tail_winglet) its own tip device — panelised by the same VLM
    # constructor as the main wing (vlm.TailSurface). Its INCIDENCE stays
    # the trim unknown either way.
    tail_free: bool = False
    #: which way up the tail's SECTION is mounted — tail.TailProblem's field
    #: and tail.tail_polar's rule, so the two cores cannot fly the same
    #: surface differently. None (default) follows the load: inverted where
    #: the surface pushes down, which is the section half of the same
    #: statement ``tail_winglet_follow`` makes about the tip device.
    tail_inverted: bool | None = None
    tail_winglet: bool = False
    tail_cant_bounds: tuple | None = None
    winglet_h_bounds_t: tuple | None = None
    #                          tip-device HEIGHT band for this surface, as a
    #                          fraction of ITS semi-span. None = the published
    #                          (0, 0.15) — where the model was MEASURED, not a
    #                          cap. Validated by geometry.winglet_h_row.
    tail_winglet_follow: bool = False   # the tip device on the TAIL takes its
    #                             SIGN from the load the trim solve puts on
    #                             that surface, per candidate: the design
    #                             variable is then the cant's magnitude and
    #                             which side it sits on is derived, not
    #                             chosen. Costs one extra coupled solve on
    #                             the candidates where the sign disagrees
    #                             with where the device already is. Off by
    #                             default, so every published run is
    #                             bit-for-bit what it was.
    # ---- winglet configuration (values)
    winglet_cant_bounds: tuple | None = None
    winglet_h_bounds: tuple | None = None
    #                          tip-device HEIGHT band for this surface, as a
    #                          fraction of ITS semi-span. None = the published
    #                          (0, 0.15) — where the model was MEASURED, not a
    #                          cap. Validated by geometry.winglet_h_row.
    blend_frac_fixed: float = 0.0   # the tip device's root TRANSITION, as a
    #                             VALUE rather than a design variable — the
    #                             air winglet family's own field name
    #                             (objective.Problem.blend_frac_fixed) and the
    #                             same meaning: how much of the device's arc
    #                             length is spent turning out of the surface's
    #                             plane. 0 = the sharp corner every published
    #                             wing+tail run flies, bit-for-bit. Refused
    #                             beside ``blended``, which DESIGNS the same
    #                             number. It reaches BOTH tip devices — the
    #                             wing's and the tail's — for the reason
    #                             ``winglet_chord_follows`` does: it is a
    #                             statement about how the design is built, not
    #                             about one surface.
    wing_blend_frac: float = 0.0    # ...and how much of the turn the SURFACE
    #                             does, as a fraction of its own semi-span.
    #                             Confined to the device the turn has at most
    #                             blend_frac * h of arc, so the fillet radius
    #                             is tied to the device's height; letting it
    #                             start inboard of the tip is what makes a
    #                             blend look like a blend (geometry.span_path).
    tail_blend_frac_fixed: float | None = None   # ...and the TAIL's own device
    #                             where it differs. None = the design's blend
    #                             above, which is the statement a single
    #                             control makes ("build the tip devices
    #                             blended"); a number here is the second
    #                             surface answering the shape question for
    #                             itself, which is what a per-surface tip
    #                             device menu asks.
    blend_shape: str = "arc"
    junction_drag: bool = False
    # ---- THE WING'S OWN CANT AND SWEEP (values, not design variables here).
    # Both are lattice geometry: geometry.dihedral_rotate turns the whole
    # semi-span rigidly, and the quarter-chord line takes an |y| tan(Lambda)
    # offset. They are stated on THIS family and not on the lifting-line
    # `tail` one because only a lattice can score them — an LLT has no
    # out-of-plane geometry to give a dihedral effect to, so a flag there
    # would be a flag the objective does not read.
    #
    # ``dihedral_deg`` below is a different quantity: the TAIL's cant, i.e.
    # what makes a V-tail. It is folded into the tail's equivalent flat area
    # (tail.lifting_area) rather than into the lattice, which is this
    # family's published convention.
    wing_dihedral_deg: float = 0.0
    wing_sweep_deg: float = 0.0
    # ---- tail configuration (values; tail.py's semantics verbatim)
    tail_type: str = "conventional"
    dihedral_deg: float = 0.0
    #: THE FIN'S OWN SHAPE, where a user states one. ``None`` on all three
    #: means the published law (``fin.V_V_DEFAULT`` 0.04, ``AR_VT_DEFAULT``
    #: 1.5, ``FIN_TC_DEFAULT`` 0.10), so every stored run is bit-for-bit.
    #:
    #: They travel TOGETHER to every consumer — the drag book that charges
    #: the fin, the lateral deck that flies it, the report that states it,
    #: the exporter that lofts it — because the defect this whole surface
    #: was built out of was a fin sized one way in one place and another way
    #: somewhere else. A stated volume coefficient that reached the drawing
    #: but not the drag would be the same bug wearing a control.
    fin_volume_coeff: float | None = None
    fin_ar: float | None = None
    fin_tc: float | None = None
    #: IS THERE A FIN AT ALL — see :class:`tail.TailProblem.fin` for the
    #: whole account. ``True`` reproduces every published run bit-for-bit.
    #: Read through ``fin.has_fin``, never directly, because a V-tail is the
    #: other way of answering no.
    fin: bool = True
    #: MEASURE THE LATERAL DECK, at a price. Off by default and off in every
    #: published run: switching it on puts a :class:`vlm.VerticalSurface`
    #: into the SCORED lattice and runs :func:`dynamics.deck` on it, which
    #: costs +44 % of an evaluation (measured: 2.04 -> 2.94 ms at N_vlm 60)
    #: and perturbs the symmetric solve by 1-8 ulp, so it is not something
    #: to leave on "because it is nearly free". It is switched on by the
    #: composite objective when, and only when, the ``roll`` criterion
    #: carries a weight (``api._with_wing_objective``).
    #:
    #: Why it must be measured HERE and not from the flight rebuild: the
    #: rebuild reconstructs the wing from the report's planform and drops
    #: the tip device (``flightmodel._planform_from_report``), so its
    #: ``Cl_beta`` comes back bit-identical for every winglet height from
    #: 0.00 to 0.15 while L/D moves 5 %. A roll criterion scored off that
    #: would be blind to the wing's single largest out-of-plane surface.
    lateral: bool = False
    #: GATE THE MODES, at a MIL-F-8785C level — 3, 2, 1, or ``None`` for the
    #: shipped behaviour (no handling rows at all, every published run
    #: bit-for-bit). Setting it adds :data:`handling.N_ROWS` signed margins to
    #: this problem's constraint vector, AFTER whatever it already declared,
    #: so ``g[0]`` stays the static margin for every caller that reads it.
    #:
    #: It implies :attr:`lateral` and is armed through the same switch the
    #: spiral WEIGHT is (``api._needs_lateral``): a yaw-stiffness requirement
    #: on an aeroplane with no fin in the scored lattice would be gating a
    #: number that is exactly zero for a geometric reason.
    #:
    #: WHAT IT COSTS. The deck is the +44 % the spiral criterion already
    #: prices; on top of that, trimming, linearising and naming the modes is
    #: pure algebra on an AFFINE deck — no second lattice solve — and measures
    #: 2.2 ms against the 1.7 ms an evaluation of this family costs.
    handling_level: int | None = None
    control: str = "stabilator"
    elevator_chord_frac: float = 0.30
    delta_e_max_deg: float = 25.0
    x_cg: float | None = None
    SM_min: float = 0.08
    i_t_max_deg: float = 15.0
    alpha_bracket_deg: tuple[float, float] = (-10.0, 15.0)

    #: THE SEARCHED CANT ROWS, as ``(field, design-vector label, the name of
    #: the band in :mod:`~aerobo.geometry`)``. ONE table, read by
    #: :attr:`param_labels`, :attr:`bounds` and :meth:`unpack`, so a label
    #: and the band under it cannot come to disagree — the same shape
    #: ``endplate.CarWingEndplateProblem.FREE_ROWS`` already uses for its two
    #: freedoms. The ORDER is load-bearing: dihedral first, so a family that
    #: searches both has exactly the vector it has always had.
    CANT_ROWS = (("dihedral_free", "wing_dihedral_deg", "DIHEDRAL_BOUNDS_DEG"),
                 ("sweep_free", "wing_sweep_deg", "SWEEP_BOUNDS_DEG"))

    def __post_init__(self):
        # ``cant_free`` is the pair, said in one word. Expanded HERE and
        # never read again, so there is one representation below this line —
        # and idempotently, because ``dataclasses.replace`` re-runs this.
        if self.cant_free:
            self.dihedral_free = self.sweep_free = True
        if self.tail_type not in tailmod.TAIL_TYPES:
            raise ValueError(f"unknown tail_type {self.tail_type!r}; "
                             f"choices: {list(tailmod.TAIL_TYPES)}")
        if self.control not in ("stabilator", "elevator"):
            raise ValueError(f"unknown control {self.control!r}; "
                             f"choices: ['stabilator', 'elevator']")
        if self.blend_shape not in geometry.BLEND_SHAPES:
            raise ValueError(
                f"unknown blend_shape {self.blend_shape!r}; "
                f"choose from {list(geometry.BLEND_SHAPES)}")
        if self.blended and not self.winglet:
            raise ValueError("a blended transition needs a tip device to "
                             "blend into (winglet=True)")
        # ONE QUESTION, ONE PLACE — and asked PER ROW, because the two are
        # separately searchable now: a sweep-only family may perfectly well
        # be told a dihedral, and refusing it because its neighbour is a
        # design variable would be a ban with no measurement behind it. The
        # searched row IS the answer, so a stated value beside THAT row is a
        # second answer the box would silently overwrite — the shape of
        # defect `vlm.TailSurface` already refuses for a cant stated twice.
        for _attr, _lbl, _band in self.CANT_ROWS:
            if getattr(self, _attr) and getattr(self, _lbl):
                raise ValueError(
                    f"the wing's cant is stated twice: {_attr}=True searches "
                    f"{_lbl}, and {_lbl}={getattr(self, _lbl)} states it. "
                    f"Use the design box (bounds_overrides / pinned) on the "
                    f"searched row, or the family that states it")
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
        if self.tail_blend_frac_fixed is not None:
            if not (blo <= float(self.tail_blend_frac_fixed) <= bhi):
                raise ValueError(
                    f"tail_blend_frac_fixed {self.tail_blend_frac_fixed} "
                    f"outside the design band "
                    f"{geometry.WINGLET_BLEND_BOUNDS}")
            if float(self.tail_blend_frac_fixed) > 0.0 \
                    and not self.tail_winglet:
                raise ValueError(
                    "tail_blend_frac_fixed needs a tip device ON THE TAIL to "
                    "blend into it (tail_winglet=True)")
        if float(self.blend_frac_fixed) > 0.0:
            if self.blended:
                raise ValueError(
                    "this problem DESIGNS the tip device's blend fraction "
                    "(winglet_blend_frac is a design variable), so a stated "
                    "blend_frac_fixed would be the same number twice — drop "
                    "one of them")
            if not (self.winglet or self.tail_winglet):
                raise ValueError(
                    "blend_frac_fixed needs a tip device to blend into the "
                    "surface it sits on; this problem carries none")
        if self.capped and not self.winglet:
            raise ValueError("span-capped accounting needs a tip device")
        if self.free_height and self.tail_type == "t_tail":
            raise ValueError(
                "a T-tail's height is DERIVED from its fin sizing "
                "(tail.tail_height), so it cannot also be a design variable "
                "— use the conventional layout to make the height free")
        if self.z_t_fixed is not None:
            if self.free_height:
                raise ValueError(
                    "the tail height cannot be both a design variable and a "
                    "stated value (free_height with z_t_fixed) — free it, or "
                    "state it, not both")
            if self.tail_type == "t_tail":
                raise ValueError(
                    "a T-tail's height is DERIVED from its fin sizing "
                    "(tail.tail_height), so it cannot be stated either")
            # ...and NO floor on the value — see tail.height_row: 0.05 b is
            # the MEASURED height, and a tailplane in the wing plane is the
            # ordinary layout, not an out-of-contract input.
        if self.x_cg is None:
            # the sized twin's (b, S) are two design rows' seeds, not the
            # aeroplane — the CG is calibrated on the wing the BOX can build
            # (tail.cg_reference_size, and the "no solution" it closes)
            self.x_cg = tailmod.default_x_cg(
                self.tail_type, *tailmod.cg_reference_size(
                    self.b, self.S, size_free=self.size_free,
                    span_bounds_m=self.span_bounds_m,
                    area_bounds_m2=self.area_bounds_m2,
                    ws_bounds_pa=self.ws_bounds_pa,
                    wing_loading_Pa=self.wing_loading_Pa,
                    W_design_N=tailmod._cg_design_weight(self)))
        if self.tail_type == "v_tail":
            lo, hi = tailmod.DIHEDRAL_BOUNDS_DEG
            if not (lo <= float(self.dihedral_deg) <= hi):
                raise ValueError(
                    f"V-tail dihedral {self.dihedral_deg} deg outside the "
                    f"supported range {tailmod.DIHEDRAL_BOUNDS_DEG}")
        if self.l_t_bounds_m is not None:
            if self.l_t_fixed is not None:
                raise ValueError(
                    "the arm cannot be both a stated value (l_t_fixed) and a "
                    "searched band (l_t_bounds_m) — state it, or box it")
            tailmod.arm_row(self.l_t_bounds_m)      # validated once, here
        if self.z_t_bounds_m is not None:
            if self.z_t_fixed is not None:
                raise ValueError(
                    "the tail height cannot be both a stated value "
                    "(z_t_fixed) and a searched band (z_t_bounds_m) — state "
                    "it, or box it")
            tailmod.height_row(self.z_t_bounds_m, (0.0, 1.0))   # validated
        if self.l_t_fixed is not None:
            if float(self.l_t_fixed) < tailmod.L_T_MIN_M:
                raise ValueError(
                    f"stated arm {self.l_t_fixed} m is below the floor "
                    f"{tailmod.L_T_MIN_M} m: with no moment arm the tail "
                    f"volume is zero and the pitch trim is singular. (The "
                    f"CALIBRATED band is {tailmod.L_T_BOUNDS} m — the box "
                    f"the search opens on — but a longer aeroplane is a "
                    f"design decision, not an out-of-contract input.)")
        if self.tc_free and self.polar_family is None:
            self.polar_family = default_polar_family()
        if self.flight_free and self.mission is None:
            # The weight THIS problem already flies, so the box point
            # (V, h) = (self.V, 0) returns its own trim target: q is built
            # from the ISA sea-level density the MissionSpec will itself use
            # (NOT the RHO_SL literal), which makes CL_target round-trip
            # exactly at the default 0.5 — halving is exact — and to within
            # an ulp otherwise. The only quantities that move are the ones
            # rho/mu feed as DIAGNOSTICS (Re_mac) plus the fin's parasite
            # drag, both at the 4e-7 relative level; mission.default_mission
            # documents the same trade for objective.py's mission mode.
            from .mission import MissionSpec, isa_density
            q0 = 0.5 * isa_density(0.0) * self.V**2
            self.mission = MissionSpec(W_N=self.CL_target * (q0 * self.S),
                                       V=self.V, altitude_m=0.0)

    # ---------------------------------------------------------- the vector

        # LAST, so the checks above have already refused a malformed
        # configuration before this one turns a flag into a deck.
        if self.handling_level is not None:
            from . import handling as _hq

            # VALIDATED HERE, so a bad level is a construction error with the
            # choices named rather than five G_FAIL rows on every evaluation
            # of a run that will never produce anything.
            self.handling_level = _hq.level_of(self.handling_level)
            # ...AND IT IMPLIES THE DECK. Asking for a Dutch roll on a
            # lattice with no vertical surface is asking about a mode that is
            # exactly zero for a geometric reason, so the flag that requests
            # the gate is also the one that puts the fin in — the same rule
            # the spiral WEIGHT follows (``api._needs_lateral``), and stated
            # in BOTH places because a caller can build this problem directly.
            self.lateral = True

    @property
    def constraint_labels(self) -> tuple[str, ...]:
        """The margins this problem declares, in ``g``'s own order.

        The static margin is index 0 and stays there; the size modifier's
        stress margin keeps index 1 where it exists; the handling rows go
        LAST. Declared rather than counted, because ``api`` reads a built
        problem's own declaration in preference to the registry's static
        count (``_with_wing_objective``) — which is what lets one flag change
        the width without thirty ``ProblemSpec`` entries having to know.
        """
        out = ("static margin - SM_min",)
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
    def tail_blend_frac(self) -> float:
        """The blend the TAIL's own tip device is drawn with.

        The design's blend unless this surface answered for itself — one
        place, so the solver, the junction charge and the breakdown cannot
        disagree about which number the second surface flew.
        """
        return float(self.blend_frac_fixed
                     if self.tail_blend_frac_fixed is None
                     else self.tail_blend_frac_fixed)

    @property
    def z_t_bounds(self) -> tuple[float, float]:
        """Tail-height box [m] — the USER's band, else the span fractions on
        THIS wing's span (``tail.height_row``)."""
        lo, hi = Z_T_FRAC_BOUNDS
        return tailmod.height_row(self.z_t_bounds_m,
                                  (lo * self.b, hi * self.b))

    @property
    def param_labels(self) -> tuple:
        labels = ["taper", "twist_root_deg", "twist_tip_deg", "S_t_m2"]
        if self.l_t_fixed is None:
            labels.append("l_t_m")
        if self.free_height:
            labels.append("z_t_m")
        if self.tail_free:
            labels += ["taper_t", "AR_t", "washout_t_deg"]
            if self.tail_winglet:
                labels += ["winglet_h_frac_t", "winglet_cant_t_deg"]
        if self.winglet:
            labels += ["winglet_h_frac", "winglet_cant_deg"]
        if self.blended:
            labels.append("winglet_blend_frac")
        if self.tc_free:
            labels.append("tc")
        labels += [lbl for attr, lbl, _b in self.CANT_ROWS
                   if getattr(self, attr)]
        from .sizing import size_labels
        chord = geometry.chord_labels(self.chord_order,
                                      self.chord_law)
        if self.tail_free and chord:
            # one chord law per SURFACE, wing first then tail — the tandem
            # pair's convention (tandem.py), so the trailing block is read
            # the same way wherever two surfaces are designed together
            chord = chord + tuple(f"{lbl}_t" for lbl in chord)
        return (tuple(labels) + size_labels(self.size_free)
                + geometry.flight_labels(self.flight_free) + chord)

    @property
    def bounds(self) -> np.ndarray:
        # the tail's SIZE rows, narrowed first to whatever the user stated
        # in metres: a span or chord limit IS an area/aspect-ratio band, and
        # tail.TailLimits.narrow is that change of variables
        s_row = tailmod.area_row(self.s_t_bounds_m2,
                                 tailmod.area_band(self.S))
        ar_row = TAIL_AR_BOUNDS if self.tail_free else None
        if self.tail_limits is not None:
            s_row, ar_row = self.tail_limits.narrow(s_row, ar_row,
                                                    tailmod.TAIL_AR)
        rows = [geometry.TAPER_BOUNDS,
                geometry.TWIST_ROOT_BOUNDS_DEG,
                geometry.TWIST_TIP_BOUNDS_DEG,
                s_row]
        if self.l_t_fixed is None:
            rows.append(tailmod.arm_row(self.l_t_bounds_m,
                                        tailmod.arm_band(self.b)))
        if self.free_height:
            rows.append(self.z_t_bounds)
        if self.tail_free:
            rows += [geometry.TAPER_BOUNDS, ar_row,
                     TAIL_WASHOUT_BOUNDS_DEG]
            if self.tail_winglet:
                rows += [
                    geometry.winglet_h_row(self.winglet_h_bounds_t),
                    self._cant_box(self.tail_cant_bounds, signed=True)]
        if self.winglet:
            rows += [geometry.winglet_h_row(self.winglet_h_bounds),
                     self._cant_box(self.winglet_cant_bounds)]
        if self.blended:
            rows.append(geometry.WINGLET_BLEND_BOUNDS)
        if self.tc_free:
            rows.append(geometry.TC_BOUNDS)
        rows += [getattr(geometry, band) for attr, _l, band in self.CANT_ROWS
                 if getattr(self, attr)]
        from .sizing import with_size_bounds, ws_of
        box = with_size_bounds(np.array(rows, dtype=float), self.size_free,
                               self.b, self.S, self.span_bounds_m,
                               area_bounds_m2=self.area_bounds_m2,
                               ws0=ws_of(self.CL_target, self.rho, self.V),
                               ws_bounds_pa=self.ws_bounds_pa,
                               wing_loading_max_Pa=self.wing_loading_max_Pa)
        box = geometry.with_flight_bounds(box, self.flight_free)
        box = geometry.with_chord_bounds(box, self.chord_order,
                                         self.chord_max_frac, self.chord_law)
        if self.tail_free and self.chord_order:
            # the tail's own chord law, appended after the wing's (the
            # labels above name them ``*_t``)
            box = geometry.with_chord_bounds(box, self.chord_order,
                                             self.chord_max_frac,
                                             self.chord_law)
        return box

    def _cant_box(self, override, signed: bool = False) -> tuple:
        """Cant band for a tip device: the published one, or the caller's
        (refused outside the physical VLM range).

        ``signed`` widens the ACCEPTED range to
        :data:`geometry.WINGLET_CANT_LIMITS_SIGNED_DEG`, and is passed for a
        device on the TAIL. The default band is the same (60, 90) either
        way, so nothing published moves; what it allows is asking for the
        mirror. A tail is the surface where that question is real: mirroring
        a surface mirrors its lift, so the device direction that pays on a
        stabiliser carrying a DOWNLOAD is the opposite of the one that pays
        on a lifting wing — and the wing's band cannot express it.
        """
        if override is None:
            return geometry.WINGLET_CANT_BOUNDS_DEG
        lo, hi = float(override[0]), float(override[1])
        limits = (geometry.WINGLET_CANT_LIMITS_SIGNED_DEG if signed
                  else geometry.WINGLET_CANT_LIMITS_DEG)
        cmin, cmax = limits
        if not (cmin <= lo < hi <= cmax):
            raise ValueError(
                f"winglet cant_bounds {(lo, hi)} outside the physical "
                f"VLM range {limits}")
        return (lo, hi)

    @property
    def dim(self) -> int:
        return int(self.bounds.shape[0])

    @property
    def n_chord_rows(self) -> int:
        """Trailing chord-coefficient rows: one block per DESIGNED surface,
        which is what the size and flight blocks have to step over."""
        return int(self.chord_order) * (2 if self.tail_free else 1)

    @property
    def tau_elevator(self) -> float:
        """Control effectiveness: 1.0 for an all-moving stabilator."""
        if self.control != "elevator":
            return 1.0
        return tailmod.flap_effectiveness(self.elevator_chord_frac)

    def unpack(self, x: np.ndarray) -> dict:
        """Design vector -> named values (the ONE place the layout is read).

        Chord coefficients are the TRAILING entries, as everywhere in the
        package; every other block keeps its position whatever is switched
        on, which is what lets the labels above be static per problem.
        """
        x = np.asarray(x, dtype=float)
        i = 4
        out = {"taper": float(x[0]), "twist_root_deg": float(x[1]),
               "twist_tip_deg": float(x[2]), "S_t": float(x[3])}
        if self.l_t_fixed is None:
            out["dist"] = float(x[i]); i += 1
        else:
            out["dist"] = float(self.l_t_fixed)
        if self.free_height:
            out["z_t"] = float(x[i]); i += 1
        else:
            # the layout's own height, unless the caller stated one
            out["z_t"] = (None if self.z_t_fixed is None
                          else float(self.z_t_fixed))
        out["taper_t"], out["AR_t"], out["washout_t"] = None, None, 0.0
        out["h_frac_t"], out["cant_t_deg"] = 0.0, 90.0
        if self.tail_free:
            out["taper_t"] = float(x[i])
            out["AR_t"] = float(x[i + 1])
            out["washout_t"] = float(x[i + 2])
            i += 3
            if self.tail_winglet:
                out["h_frac_t"] = float(x[i])
                out["cant_t_deg"] = float(x[i + 1])
                i += 2
        if self.winglet:
            out["h_frac"], out["cant_deg"] = float(x[i]), float(x[i + 1])
            i += 2
        else:
            out["h_frac"], out["cant_deg"] = 0.0, 90.0
        # the blend fraction has ONE owner: the design vector where the
        # problem designs it, the stated VALUE everywhere else (objective.py
        # reads the same pair the same way)
        out["blend_frac"] = float(self.blend_frac_fixed)
        if self.blended:
            out["blend_frac"] = float(x[i]); i += 1
        out["tc"] = None
        if self.tc_free:
            out["tc"] = float(x[i]); i += 1
        # the wing's cant and sweep have ONE reader below whichever way they
        # were answered: the searched rows where this family carries them,
        # the stated fields otherwise. Spelled ``wing_*`` here as everywhere
        # else, because ``prob.dihedral_deg`` is a DIFFERENT angle — the
        # tail's cant, i.e. what makes a V-tail — and the two have been
        # confused in this package before.
        out["wing_dihedral_deg"] = float(self.wing_dihedral_deg)
        out["wing_sweep_deg"] = float(self.wing_sweep_deg)
        for attr, lbl, _band in self.CANT_ROWS:
            if getattr(self, attr):
                out[lbl] = float(x[i]); i += 1
        m = int(self.chord_order)
        if self.tail_free and m:
            # two trailing chord blocks, wing then tail (tandem's convention)
            out["chord_coeffs"] = geometry.ChordCoeffs(x[-2 * m:-m],
                                                       self.chord_law)
            out["chord_coeffs_t"] = geometry.ChordCoeffs(x[-m:],
                                                         self.chord_law)
        else:
            out["chord_coeffs"] = geometry.chord_coeffs_from_x(
                x, m, self.chord_law)
            out["chord_coeffs_t"] = ()
        return out


# ---------------------------------------------------------------- evaluate


def _ae_range(alpha_eff_deg, solid) -> tuple:
    """The effective-angle range OF THE LIFTING SURFACES, fin excluded.

    The gate that uses it is already masked (a symmetric fin at zero sideslip
    flies at c_l ~ 0 — "a surface that is doing nothing able to refuse a
    design"), and ``tandemvlm`` reports its own range off exactly that mask.
    This was the one place still quoting the raw array, so a refusal named a
    range wider than the range it had actually tested.

    NOT a number that moves: over 128 Sobol draws of `tail [free cant]`'s box
    the armed and unarmed reports agree at 102/102 comparable draws, because
    the fin's angle sits inside the wing's. Aligned because one quantity
    should have one author, not because a published value was wrong.
    """
    ae = alpha_eff_deg if solid is None else alpha_eff_deg[solid]
    if not ae.size:                       # nothing but a fin: say nothing new
        ae = alpha_eff_deg
    return (float(ae.min()), float(ae.max()))


def evaluate_wing_tail(x: np.ndarray, prob: WingTailProblem | None = None
                       ) -> dict:
    """Full evaluation with breakdown; -100.0 penalty contract on failure."""
    prob = prob or WingTailProblem()
    x = np.asarray(x, dtype=float)
    bnds = prob.bounds
    if x.shape != (bnds.shape[0],):
        return _fail(f"design vector shape {x.shape} != ({bnds.shape[0]},)")
    if np.any(x < bnds[:, 0] - 1e-12) or np.any(x > bnds[:, 1] + 1e-12):
        return _fail("bounds violation")

    v = prob.unpack(x)

    # ---- size (sizing.py's modifier): the candidate span and area replace
    # the problem's own BEFORE anything is built, so the planform, the tail
    # height floor, the trim target, the weight and the spar all describe one
    # aircraft.
    size_req = bool(prob.size_free)
    if size_req:
        from dataclasses import replace as _replace

        from .mission import weight_for as _weight_for
        from .sizing import (WS_MODES, check_ar, flow_state_for,
                             resolve_size, size_mode)
        # the payload weight is the SAME CONSTANT for every candidate — that
        # is what makes payload L/D = W_fixed/D a minimum-drag objective
        # (sizing.py). Read from the problem's OWN published area, BEFORE the
        # candidate size replaces it: taken afterwards it scaled with the
        # area, so a bigger wing carried a bigger "fixed" payload and the
        # score rewarded growing the wing.
        W_fixed_ref = (prob.W_fixed_N if prob.W_fixed_N is not None
                       else _weight_for(prob.CL_target, prob.V, prob.S))
        kw = {"wing_loading_max_Pa": prob.wing_loading_max_Pa}
        if size_mode(prob.size_free) in WS_MODES:
            # the WING's area follows the mission's wing loading: read the
            # flow state first (it needs no area), then close the area's own
            # fixed point against that dynamic pressure
            rho_ws, V_ws = flow_state_for(
                x, mission=prob.mission, flight_free=prob.flight_free,
                n_trailing=prob.n_chord_rows, rho=prob.rho, V=prob.V)
            kw.update(
                # ignored by the SEARCHED-loading mode, which reads its own
                # row out of the vector (sizing.resolve_spans)
                wing_loading_Pa=(prob.wing_loading_Pa
                                 if prob.wing_loading_Pa is not None
                                 else prob.CL_target * 0.5 * prob.rho
                                 * prob.V ** 2),
                W_fixed_N=W_fixed_ref,
                taper=float(v["taper"]),
                tc=(geometry.TC_DEFAULT if v["tc"] is None
                    else float(v["tc"])),
                q_Pa=0.5 * rho_ws * V_ws ** 2)
        try:
            b_use, S_use = resolve_size(x, prob.size_free, prob.flight_free,
                                        prob.n_chord_rows, **kw)
        except (ValueError, OverflowError) as exc:
            return _fail(f"size: {exc}")
        reason = check_ar(b_use, S_use)
        if reason is not None:
            return _fail(f"size: {reason}")
        prob = _replace(prob, b=b_use, S=S_use, size_free=False)

    # ...and the ASPECT-RATIO LIMIT THE USER SET. A different question from
    # the validity band above, so it is asked whether or not the size is a
    # design variable: a limit that only applied to searched sizes would go
    # quiet exactly when the user typed the number it is about.
    if prob.ar_limits is not None:
        from .sizing import check_ar_limit as _check_ar_limit
        why_ar = _check_ar_limit(float(prob.b), float(prob.S), prob.ar_limits)
        if why_ar is not None:
            return _fail(f"size: {why_ar}")

    # ---- flight state (geometry.py's flight modifier): rebuild the flow
    # state and the trim target this candidate's (V, altitude) imply, then
    # carry on with a COPY that flies it. Nothing downstream changes: the
    # trim, the static margin and the drag build-up all read the same fields.
    flight = geometry.flight_from_x(x, prob.flight_free, prob.n_chord_rows)
    if flight is not None:
        from dataclasses import replace as _replace

        from .mission import flight_state
        try:
            prob = _replace(prob, flight_free=False, mission=None,
                            **flight_state(prob.mission, flight[0], flight[1],
                                           prob.S))
        except ValueError as exc:      # water medium, or ISA validity
            return _fail(f"flight state: {exc}")

    pol = prob.polar
    if v["tc"] is not None:
        try:
            pol = prob.polar_family.at(v["tc"])   # exact member or blend
        except ValueError as exc:
            return _fail(f"polar family: {exc}")

    # ---- wing geometry (span-capped devices pay for their projection)
    b_wing = prob.b
    h = float(v["h_frac"]) * prob.b / 2.0
    if prob.winglet and prob.capped and h > 0.0:
        # the wing-side arc is measured on the semi-span the DEVICE hangs off,
        # which is what the span cap is solving for — so the projection is
        # read on prob.b/2 (the pre-cap semi-span), exactly as hydrofoil.py
        # measures it on its own
        proj = geometry.winglet_projection(h, v["cant_deg"], v["blend_frac"],
                                           prob.blend_shape,
                                           wing_arc=(prob.wing_blend_frac
                                                     * prob.b / 2.0))
        b_wing = prob.b - 2.0 * proj
        if b_wing <= 0.0:
            return _fail("span cap leaves no wing")
    try:
        wing = geometry.Wing(
            b=b_wing, S=prob.S, taper=v["taper"],
            twist_root_deg=v["twist_root_deg"],
            twist_tip_deg=v["twist_tip_deg"],
            # t/c is a design variable only when the section is free; the
            # fixed case keeps Wing's own default, which IS the default
            # polar's NACA 2412 thickness (it feeds the junction charge and
            # the reported geometry, never the polar lookup)
            tc=0.12 if v["tc"] is None else float(v["tc"]),
            sweep_deg=v["wing_sweep_deg"],
            dihedral_deg=v["wing_dihedral_deg"],
            chord_limits=prob.chord_limits,
            chord_coeffs=v["chord_coeffs"])
    except ValueError as exc:
        return _fail(f"planform: {exc}")
    # ---- the weight the candidate SIZE implies (sizing.py): the trim
    # target stops being a constant and becomes W_total/(q S) for this wing.
    sized = None
    if size_req:
        from .sizing import sized_state
        try:
            sized = sized_state(
                W_fixed_N=W_fixed_ref,
                b=prob.b, S=prob.S, taper=wing.taper, tc=wing.tc,
                q_Pa=0.5 * prob.rho * prob.V**2,
                c_root=float(wing.chord(np.array([0.0]))[0]),
                # ...AND THE EMPENNAGE, priced by layout (tail.py's one
                # reader): the tailplane's own area, and a fin unless the
                # layout is a V-tail. The height it is sized against is the
                # layout's, computed the same way the tail geometry below
                # computes it — one rule, read twice, never two rules.
                empennage=tailmod.empennage_surfaces(
                    tail_type=prob.tail_type, S_t=v["S_t"], b=prob.b,
                    S=prob.S, l_t=v["dist"],
                    dz=(tailmod.tail_height(prob.tail_type, v["dist"],
                                            prob.b, prob.S)
                        if v["z_t"] is None else float(v["z_t"])),
                    AR_t=v["AR_t"], taper_t=v["taper_t"],
                    tc=(None if v["tc"] is None else float(v["tc"])),
                    fin=_fin.has_fin(prob),
                    fin_kwargs=_fin.fin_law_kwargs(prob)),
                wing_loading_max_Pa=prob.wing_loading_max_Pa,
                material=prob.material)
        except (ValueError, RuntimeError, OverflowError) as exc:
            return _fail(f"size: {exc}")
        CL_trim = sized.CL_target
    else:
        CL_trim = prob.CL_target

    # the winglet's ARC LENGTH is the design variable, so a shrunken wing
    # must not shrink the device with it (objective.evaluate's rule)
    h_frac_flown = float(v["h_frac"]) * prob.b / wing.b if wing.b > 0 else 0.0

    # ---- tail geometry: layout decides where, unless the height is free
    S_lift = tailmod.lifting_area(prob.tail_type, prob.dihedral_deg, v["S_t"])
    x_t = tailmod.surface_x(prob.tail_type, v["dist"])
    # ...and the height is NOT re-checked against DZ_FRAC*b here. It used to
    # be, which made the floor a third, independent ban: even with the box
    # widened and the constructor's guard cleared, every draw below 0.05 b
    # came back a penalty. The box is the statement about where to search
    # (tail.height_row) and this solve is smooth through it to zero.
    z_t = (tailmod.tail_height(prob.tail_type, v["dist"], prob.b, prob.S)
           if v["z_t"] is None else float(v["z_t"]))
    pol_t = pol if prob.polar_tail is None else prob.polar_tail
    tail_wing = None
    if prob.tail_free:
        # the tail as a wing in its own right: its aspect ratio sets the
        # span the same area is spread over, and its washout is measured
        # from a root at zero — the uniform part IS the trim incidence
        try:
            tail_wing = geometry.Wing(
                b=float(np.sqrt(float(v["AR_t"]) * S_lift)), S=S_lift,
                taper=float(v["taper_t"]),
                twist_root_deg=0.0, twist_tip_deg=float(v["washout_t"]),
                tc=0.12 if v["tc"] is None else float(v["tc"]),
                # the TAIL's chord limits are the tail's (tail.TailLimits),
                # never the WING's metres — see tail._config. The two that
                # are NOT lengths stay: ``trend`` ("the root is the largest
                # chord") and ``rate_max_deg`` (a local taper ANGLE) are
                # invariant under a uniform rescale, so they mean the same
                # thing on a 0.6 m stabiliser as on a 10 m wing, and the
                # shell says so on screen ("it holds on every designed
                # surface"). Written as ``chord_limits=None`` this threw
                # them out with the metres, and a chord that grew outboard
                # was refused on the wing and flew on the tail — the same
                # defect tail.py fixed for the lifting-line twin.
                chord_limits=(None if prob.chord_limits is None else
                              replace(prob.chord_limits,
                                      c_min_m=None, c_max_m=None)),
                chord_coeffs=v["chord_coeffs_t"])
        except ValueError as exc:
            return _fail(f"tail planform: {exc}")
    mac = wing.mac

    # ...and what the user said the surface may MEASURE, in metres. Checked
    # on the tail actually drawn: the box narrowing above is exact only with
    # the aspect ratio fixed, and with it free the box is the smallest one
    # CONTAINING the allowed set — these are the corners it added.
    if prob.tail_limits is not None:
        _b_t = (tail_wing.b if tail_wing is not None
                else float(np.sqrt(tailmod.TAIL_AR * S_lift)))
        _c_t = (tail_wing.sample(prob.N_t)[1] if tail_wing is not None
                else (S_lift / _b_t if _b_t > 0.0 else 0.0))
        _why = prob.tail_limits.violation(_b_t, _c_t)
        if _why is not None:
            return _fail(_why)

    # the sections' own couple, on this model's (S, mac). The lattice cannot
    # make it — one chordwise panel per strip puts every bound vortex on the
    # quarter-chord line — so it is handed in (vlm.VLM.solve_trim_moment),
    # exactly as tail.section_moment derives it for the lifting-line core.
    # M/q = cm_ac * S * mac per surface, the second factor being MAC's own
    # definition. TIP DEVICES are left out: a near-vertical panel's section
    # couple is a YAW couple, not a pitch one, to the same first order in
    # cant that this planar-strip drag model already assumes.
    _mac_t = (tail_wing.mac if tail_wing is not None
              else float(np.sqrt(S_lift / tailmod.TAIL_AR))
              if S_lift > 0.0 else 0.0)
    _m_ac_w = float(_section_cm_ac(pol) * wing.S * wing.mac)
    _m_ac_t = float(_section_cm_ac(pol_t) * S_lift * _mac_t)
    cm_ac_model = float((_m_ac_w + _m_ac_t) / (prob.S * mac))

    # ...and WHICH WAY UP the surface flies its section: inverted where it
    # pushes down (tail.tail_polar — the rule the tip device already obeys),
    # read off the load the balance implies with the WING's couple ALONE.
    # That exclusion is tail.stabiliser_load's, and api.trim_surface_cl's,
    # and it is the whole point of both: mirroring a section mirrors THAT
    # surface's own couple, so a reading that included it would be letting
    # the choice vote on itself — the download that justified inverting the
    # surface is destroyed by the inversion, and nothing converges because
    # nothing is iterated.
    #
    # Invisible while the stabiliser is small, which is why it survived:
    # over the published 10 m aeroplane's whole (S_t, l_t) box the two
    # readings choose the same orientation at 121/121 points. It is decisive
    # when the surface is not small. On a 0.5 kg model flown through the
    # same box — S_T_BOUNDS is in absolute m^2, so the tail comes out ~10x
    # the wing — the stabiliser's own couple is 43x the wing's, the two
    # readings differ at 121/121 points, and the self-referring one mounted
    # the section inverted and then trimmed it to lift UP (CL_t +0.0066
    # here against the lifting-line core's -0.0064 at the same design).
    cl_stab = tailmod.trim_lift_coefficient(
        CL_trim, prob.S, prob.x_cg, S_lift, x_t, M_ac=_m_ac_w)
    pol_t = tailmod.tail_polar(prob, cl_stab)
    if isinstance(pol_t, _InvertedPolar):
        # the couple mirrors with the section, so the balance the SOLVE is
        # handed is re-read on the section actually flown. The orientation
        # above is NOT re-decided on it.
        cm_ac_model = float(
            (_m_ac_w + _section_cm_ac(pol_t) * S_lift * _mac_t)
            / (prob.S * mac))

    def _fly(cant_t: float):
        """One coupled build + trim solve at a stated tail-device cant."""
        surf = TailSurface(S=S_lift, x=x_t, z=z_t, AR=tailmod.TAIL_AR,
                           N=prob.N_t, a=pol_t.a_lin,
                           alpha_L0=pol_t.alpha_L0, wing=tail_wing,
                           winglet_h_frac=float(v["h_frac_t"]),
                           winglet_cant_deg=float(cant_t),
                           n_winglet=prob.n_winglet,
                           # the blend is a statement about how the DESIGN is
                           # built, so the tail's own device is drawn with the
                           # same transition the wing's is
                           winglet_blend_frac=prob.tail_blend_frac,
                           winglet_blend_shape=prob.blend_shape,
                           winglet_wing_blend_frac=prob.wing_blend_frac)
        m = VLM(wing, N=prob.N_vlm, winglet_h_frac=h_frac_flown,
                winglet_cant_deg=v["cant_deg"], n_winglet=prob.n_winglet,
                winglet_blend_frac=v["blend_frac"],
                winglet_blend_shape=prob.blend_shape,
                winglet_wing_blend_frac=prob.wing_blend_frac,
                winglet_chord_follows=prob.winglet_chord_follows,
                a=pol.a_lin, alpha_L0=pol.alpha_L0, V=prob.V, tail=surf,
                vertical=_lateral_fin())
        al, it, rr = m.solve_trim_moment(
            CL_trim, prob.x_cg, mac,
            alpha_bracket=(np.deg2rad(prob.alpha_bracket_deg[0]),
                           np.deg2rad(prob.alpha_bracket_deg[1])),
            cm_ac=cm_ac_model)
        return m, surf, al, it, rr

    def _lateral_fin():
        """The surface that makes yaw stiffness, or None.

        ``None`` unless the lateral deck was asked for — the fin perturbs
        the symmetric solve in the last ulp, so an unasked-for one would
        move every published L/D by a rounding. Sized by the ONE fin law
        (``fin.fin_for_layout``), which is also what the drag book, the
        report, the CAD export and the flight rebuild use, so the surface
        scored here is the surface flown everywhere else.

        A V-TAIL gets None from that law and needs nothing else: its own
        canted panels are already in this lattice and they carry the yaw.
        AN AEROPLANE WITH NO FIN gets None too, and then ``Cn_beta`` is
        exactly zero rather than 0.1133 — which is what stage 1's card has
        been claiming all along.
        """
        if not prob.lateral:
            return None
        g = _fin.fin_for_layout(b=prob.b, S=prob.S, l_t=v["dist"],
                                tail_type=prob.tail_type, dz=z_t,
                                fin=_fin.has_fin(prob),
                                **_fin.fin_law_kwargs(prob))
        if g is None:
            return None
        return VerticalSurface(height=g.height, chord=g.chord, x=g.x_qc,
                               z_root=g.z_root, N=12)

    def _tail_lift(rr) -> float:
        """The tail's own lift coefficient, on ITS own area (the report's
        ``CL_t``), read straight off a solved circulation."""
        m = rr.is_tail
        return (2.0 * float(rr.Gamma[m] @ rr.ly[m]) / (prob.V * S_lift)
                if (m.any() and S_lift > 0.0) else 0.0)

    cant_t = float(v["cant_t_deg"])
    followed = None                  # None = nobody asked it to follow
    try:
        model, tail_surf, alpha, i_t, res = _fly(cant_t)
        # THE DEVICE FOLLOWS THE LOAD. Mirroring a surface mirrors its lift,
        # so the tip device that pays on a surface carrying a download is the
        # mirror of the one that pays on a lifting surface — and what this
        # surface carries is not an input, it is the trim outcome. In follow
        # mode the cant's MAGNITUDE stays the design variable and its SIGN is
        # read off that outcome: solve once, look at CL_t, and re-fly
        # mirrored if the sign disagrees with where the device is. One extra
        # solve, and only when it disagrees.
        if prob.tail_winglet_follow and float(v["h_frac_t"]) > 0.0:
            want = -1.0 if _tail_lift(res) < 0.0 else 1.0
            cant_t = float(np.copysign(abs(cant_t), want))
            followed = "up" if cant_t >= 0.0 else "down"
            if cant_t != float(v["cant_t_deg"]):
                model, tail_surf, alpha, i_t, res = _fly(cant_t)
        x_np = model.neutral_point()
    except (ValueError, np.linalg.LinAlgError) as exc:
        return _fail(f"solver failure: {exc}")

    if abs(i_t) > np.deg2rad(prob.i_t_max_deg):
        return _fail("untrimmable: |i_t| exceeds limit",
                     i_t_deg=float(np.rad2deg(i_t)))
    delta_e = float(i_t) / prob.tau_elevator
    if abs(delta_e) > np.deg2rad(prob.delta_e_max_deg):
        return _fail("untrimmable: |elevator deflection| exceeds limit",
                     delta_e_deg=float(np.rad2deg(delta_e)),
                     i_t_deg=float(np.rad2deg(i_t)))

    SM = (x_np - prob.x_cg) / mac
    g = (SM - prob.SM_min) / 1.0

    alpha_eff_deg = np.rad2deg(res.alpha_eff)
    tail_mask = res.is_tail
    # THE FIN IS NOT A LIFTING SURFACE OF THIS DESIGN, and the lattice does
    # not know that. When the lateral deck is on, the vertical's panels are
    # in the solve so that a sideslip has something to push on — but they
    # must not join the horizontal book-keeping below, for two separate
    # reasons and both were measured:
    #
    # * PROFILE DRAG. The strip sum is over panel wetted area, so the fin
    #   joining it moved CDp +5.48 % and L/D -2.7 % the moment the deck was
    #   asked for. The fin's parasite drag has exactly one author — the
    #   Raymer ``cd0_fin`` below — so counting the panels too would charge
    #   the same surface twice.
    # * THE STALL GATE. A symmetric fin at zero sideslip flies at c_l ~ 0,
    #   which is a fact about the trim state and not about the section's
    #   validity range; letting it into the wing's alpha check makes a
    #   surface that is doing nothing able to refuse a design.
    #
    # With this mask the deck is free of charge to the objective: CDi comes
    # back bit-identical and CL, e, alpha and SM move by at most one ulp.
    vert_mask = getattr(model, "is_vertical", None)
    if vert_mask is None or not np.any(vert_mask):
        solid = np.ones_like(tail_mask, dtype=bool)
    else:
        solid = ~np.asarray(vert_mask, dtype=bool)
    wing_mask = (~tail_mask) & solid
    tail_only = tail_mask & solid
    # each surface is judged against ITS OWN polar's validity range; with one
    # section they are the same range, so this is the previous test verbatim
    lo_p, hi_p = pol.alpha_valid
    lo_t, hi_t = pol_t.alpha_valid
    ae_w, ae_t = alpha_eff_deg[wing_mask], alpha_eff_deg[tail_only]
    if (ae_w.min() < lo_p or ae_w.max() > hi_p
            or (ae_t.size and (ae_t.min() < lo_t or ae_t.max() > hi_t))):
        return _fail(
            "effective AoA outside polar validity (stall/extrapolation proxy)",
            alpha_eff_range=_ae_range(alpha_eff_deg, solid))

    # ---- drag: one Trefftz plane, strip profile drag, the labelled add-ons
    strip = np.where(tail_mask, pol_t.cd(alpha_eff_deg),
                     pol.cd(alpha_eff_deg)) * res.c * res.width
    # a V-tail's dihedral costs pitch effectiveness but wets exactly as much
    # surface as before, so the profile term is charged on the PANEL area
    # (tail.py's rule; wet_ratio is exactly 1 for every other layout)
    wet_ratio = float(v["S_t"]) / S_lift if S_lift > 0.0 else 1.0
    CDp_wing = float(np.sum(strip[wing_mask]) / res.S)
    CDp_tail = float(np.sum(strip[tail_only]) * wet_ratio / res.S)
    CDp = CDp_wing + CDp_tail

    # ALWAYS, like the tailplane's — see tail.py's note. A V-tail carries no
    # fin and so pays nothing, which is the layout's own argument, and an
    # aeroplane the user gave no fin pays nothing for the same reason
    # reached the other way (``fin.has_fin``).
    cd0_fin = 0.0
    if _fin.has_fin(prob):
        cd0_fin = tailmod.vtail_cd0(prob.S, v["dist"], b=prob.b, S=prob.S,
                                    V=prob.V, rho=prob.rho, mu=prob.mu,
                                    **_fin.fin_drag_kwargs(prob))

    # the corners where a tip device leaves its surface — ONE report per
    # surface that carries one, each on ITS OWN tip chord, thickness and
    # semi-span. Both are charged on the same reference area, so they add.
    # Charging only the wing's would have made a blend on the TAIL a pure
    # wake-geometry benefit with no cost, which is exactly the "differently
    # drawn wake" this add-on exists to prevent.
    jr = jr_t = None
    if prob.junction_drag and h_frac_flown > 0.0:
        c_tip = float(wing.chord(np.array([wing.b / 2.0]))[0])
        jr = junction.report(tc=wing.tc, chord=c_tip, s_ref=wing.S,
                             h=h_frac_flown * wing.b / 2.0,
                             cant_deg=v["cant_deg"],
                             blend_frac=v["blend_frac"], n_junctions=2,
                             blend_shape=prob.blend_shape,
                             wing_arc=prob.wing_blend_frac * wing.b / 2.0)
    if prob.junction_drag and prob.tail_winglet \
            and float(v["h_frac_t"]) > 0.0 and tail_wing is not None:
        c_tip_t = float(tail_wing.chord(np.array([tail_wing.b / 2.0]))[0])
        jr_t = junction.report(
            tc=tail_wing.tc, chord=c_tip_t, s_ref=prob.S,
            h=float(v["h_frac_t"]) * tail_wing.b / 2.0,
            # the cant FLOWN is signed under "follow the load"; the fillet
            # radius is a magnitude, so the correlation reads |cant|
            cant_deg=abs(float(cant_t)),
            blend_frac=prob.tail_blend_frac, n_junctions=2,
            blend_shape=prob.blend_shape,
            wing_arc=prob.wing_blend_frac * tail_wing.b / 2.0)
    CD_junction = float((jr or {}).get("CD_junction", 0.0)) \
        + float((jr_t or {}).get("CD_junction", 0.0))

    # ...and the FUSELAGE the arm implies, where a diameter was stated: the
    # same ``tail._fuselage_charge`` the lifting-line family uses, so the two
    # cannot drift. Without it a searched arm is free — its drag unmodelled
    # while its static margin and tail volume are fully counted.
    cd0_fus, fus_L, fus_fr = tailmod._fuselage_charge(prob, v["dist"])
    CD = (res.CDi + CDp + prob.cd0_extra + cd0_fin + cd0_fus
          + CD_junction)
    LoD = CL_trim / CD
    if not (np.isfinite(LoD) and np.isfinite(g)) or CD <= 0.0:
        return _fail("non-finite L/D or SM margin")

    d_alpha, _ = model.lift_derivatives()
    # per-surface lift coefficients follow tail.py's convention: each is
    # referenced to ITS OWN area, and lift_share_tail below is what makes
    # them add up (CL_w S + CL_t S_t = CL_total S)
    CL_w = 2.0 * float(res.Gamma[~tail_mask] @ res.ly[~tail_mask]) \
        / (prob.V * prob.S)
    CL_t = 2.0 * float(res.Gamma[tail_mask] @ res.ly[tail_mask]) \
        / (prob.V * S_lift) if S_lift > 0.0 else 0.0
    # each surface's tip device is its own (hydrotail.py's rule): counting
    # the stabiliser's panels in the WING's device area reported a wing
    # winglet whose planform area grew when the TAIL's device grew
    wl_wing = res.is_winglet & ~tail_mask
    wl_tail = res.is_winglet & tail_mask
    S_wl = float(np.sum((res.c * res.width)[wl_wing]))

    out = {
        "feasible": True, "reason": "", "score": float(LoD),
        "LoD": float(LoD), "g": float(g),
        "SM": float(SM), "SM_min": prob.SM_min,
        "x_np": float(x_np), "x_cg": float(prob.x_cg),
        "CL": float(res.CL), "CL_total": float(res.CL),
        "CL_target": float(CL_trim),
        # each on its OWN area (tail.py's convention); the share below is
        # what makes them add up to the system CL
        "CL_w": CL_w, "CL_t": CL_t,
        # the sections' own couple, both ways round: as M/q [m^3] and as the
        # Cm it contributes on (Sref, mac). Reported because it is what sets
        # the SIGN of CL_t, and a reader comparing a download against a CG
        # alone will not otherwise find where it came from.
        "M_ac_m3": float(cm_ac_model * prob.S * mac),
        "Cm_ac": float(cm_ac_model),
        # which way up the stabiliser's section is mounted, and why — a
        # surface that pushes down flies its camber the other way, and the
        # geometry views and the CAD export have to be told (they read this
        # key, so a core that mirrors the section in SILENCE gets drawn and
        # exported the right way up while flying the other one)
        "tail_section_inverted": bool(isinstance(pol_t, _InvertedPolar)),
        "tail_section_follows_load": prob.tail_inverted is None,
        # which way the tip device on the tail ended up pointing, and
        # whether the surface's own load chose it (follow mode) or the
        # design vector did
        "tail_winglet_cant_deg": float(cant_t),
        # gated on the device that was FLOWN, not the one that was asked
        # for: below vlm.MIN_WINGLET_FRAC the stabiliser's device never
        # enters the lattice, and reporting a side for it drew a surface
        # nothing flew
        "tail_winglet_side": (("up" if cant_t >= 0.0 else "down")
                              if model.winglet_h_m_tail > 0.0 else None),
        "tail_winglet_h_m": float(model.winglet_h_m_tail),
        "tail_winglet_h_frac": float(v["h_frac_t"]),
        "tail_winglet_S_planform": float(
            np.sum((res.c * res.width)[wl_tail])),
        "tail_winglet_follows_load": followed is not None,
        # ...and the SHAPE it was built with: its own blend (the design's
        # unless this surface answered for itself) and the corner that blend
        # bought, charged on the same reference area as the wing's
        "tail_winglet_blend_frac": (prob.tail_blend_frac
                                    if prob.tail_winglet else None),
        "CD_junction_tail": float((jr_t or {}).get("CD_junction", 0.0)),
        "lift_share_tail": (float(CL_t * S_lift / (res.CL * prob.S))
                            if res.CL else float("nan")),
        "alpha_rad": float(alpha), "alpha_deg": float(np.rad2deg(alpha)),
        "i_t_rad": float(i_t), "i_t_deg": float(np.rad2deg(i_t)),
        "delta_e_deg": float(np.rad2deg(delta_e)),
        "tau_elevator": float(prob.tau_elevator),
        "control": prob.control,
        "Cm_cg": float(res.Cm), "cm_residual": float(res.Cm),
        "cl_residual": float(res.CL - CL_trim),
        "CL_alpha": float(np.sum(d_alpha)),
        "CDi": float(res.CDi), "CDi_total": float(res.CDi),
        "CDp": float(CDp), "CDp_wing": CDp_wing, "CDp_tail": CDp_tail,
        "cd0_extra": prob.cd0_extra, "cd0_fin": float(cd0_fin),
        "cd0_fuselage": float(cd0_fus),
        "fuselage_diameter_m": (None if prob.fuselage_diameter_m is None
                                else float(prob.fuselage_diameter_m)),
        "fuselage_length_m": float(fus_L),
        "fuselage_fineness": float(fus_fr),
        # the wing's own cant and sweep, so a rebuild FLIES what was scored
        # rather than an unswept flat wing (flightmodel.build_flight_model)
        "wing_dihedral_deg": float(v["wing_dihedral_deg"]),
        "sweep_deg": float(v["wing_sweep_deg"]),
        "CD_junction": CD_junction, "CD": float(CD),
        "e": float(res.e), "AR": float(res.AR),
        "alpha_eff_range_deg": _ae_range(alpha_eff_deg, solid),
        "Re_mac": float(prob.rho * prob.V * mac / prob.mu),
        "polar": getattr(pol, "name", "unknown"),
        "polar_tail": getattr(pol_t, "name", "unknown"),
        "tail_type": prob.tail_type, "dihedral_deg": float(prob.dihedral_deg),
        "S_t": float(v["S_t"]), "S_lift": float(S_lift),
        "b_t": float(tail_surf.span), "l_t": float(x_t),
        "dist_from_wing_m": float(v["dist"]), "dz_tail": float(z_t),
        "mac": float(mac), "b_wing": float(wing.b), "taper": float(wing.taper),
        "tc": float(wing.tc), "moment_trim": True,
        "wing": wing, "vlm": res,
    }
    if prob.winglet:
        out["winglet"] = {
            "h_frac": float(v["h_frac"]), "cant_deg": float(v["cant_deg"]),
            # the height that reached the lattice: h_frac above stays the
            # REQUESTED number, so the pair reads asked / flown
            "h_m": float(model.winglet_h_m),
            "S_planform": S_wl,
            "blend_frac": float(v["blend_frac"]),
            "blend_shape": str(prob.blend_shape),
            "wing_blend_frac": float(prob.wing_blend_frac),
            "wing_blend_arc_m": float(prob.wing_blend_frac * wing.b / 2.0),
            "tip_height_m": geometry.winglet_tip_height(
                model.winglet_h_m, v["cant_deg"], v["blend_frac"],
                prob.blend_shape,
                prob.wing_blend_frac * wing.b / 2.0),
            "projection_m": geometry.winglet_projection(
                model.winglet_h_m, v["cant_deg"], v["blend_frac"],
                prob.blend_shape,
                prob.wing_blend_frac * wing.b / 2.0),
            **({"junction": jr} if jr else {}),
        }
    if wing.chord_coeffs:
        out["chord_coeffs"] = list(wing.chord_coeffs)
        out["chord_dev"] = wing.chord_dev
    if sized is not None:
        # payload L/D and the SECOND constraint (sizing.py): the static
        # margin keeps index 0, the stress margin joins it
        q = 0.5 * prob.rho * prob.V**2
        f = sized.payload_lod(q, out["CD"])
        if not np.isfinite(f):
            return _fail("non-finite payload L/D")
        out.update(sized.report())
        out["f"] = float(f)
        out["score"] = float(f)
        out["g_sm"] = float(g)
        out["g"] = np.array([float(g), float(sized.g_sigma)])
        out["D_N"] = float(q * sized.S * out["CD"])
    if prob.lateral:
        # the deck is linearised about the state that was just TRIMMED, not
        # about zero: Cl_r and Cn_p are proportional to the base CL and do
        # not exist on an unloaded wing (dynamics.deck's own docstring).
        dk = dyn.deck(model, prob.x_cg, mac, prob.b, alpha=alpha, i_t=i_t)
        out["Cl_beta"] = float(dk.Cl_beta)
        out["Cn_beta"] = float(dk.Cn_beta)
        out["Cl_r"] = float(dk.Cl_r)
        out["Cn_r"] = float(dk.Cn_r)
        out["Cl_p"] = float(dk.Cl_p)
        out["Cn_p"] = float(dk.Cn_p)
        # THE ATTITUDE THE CRITERION IS EVALUATED AT, stated beside it.
        # The deck was linearised at the trimmed ``alpha`` and the flight
        # path is level, so the pitch attitude IS that alpha — which is what
        # ``spiral_margin_of`` reads off ``dk.alpha_ref``. Written out
        # because the criterion is not attitude-free: the classical form
        # (theta0 = 0) crosses zero 2.13 deg of dihedral EARLY on this
        # family, so a margin quoted without the attitude it was measured at
        # cannot be reproduced.
        out["spiral_theta0_deg"] = float(np.rad2deg(dk.alpha_ref))
        out["spiral_margin"] = dyn.spiral_margin_of(dk)
    if prob.handling_level is not None:
        # THE MODES, ON THE LATTICE THAT WAS SCORED. Everything the gate
        # needs is already in hand at this point: the deck above (which the
        # level's own criterion has already paid for — ``handling_level``
        # arms ``lateral`` through the same switch the spiral WEIGHT does),
        # the design's weight, its own span efficiency and profile drag, and
        # the fin that actually reached the lattice rather than one a rebuild
        # sized. ``handling.gate`` says what it shares with
        # ``flightmodel.build_flight_model`` and what it refuses to guess.
        from . import handling as _hq
        from . import drag as _drag_mod

        _vert = getattr(model, "vertical", None)
        # THE WEIGHT THIS DESIGN FLIES AT, from the design and never from a
        # default. The size modifier makes it a real total; otherwise the
        # aeroplane is trimmed to ``CL_trim`` on the lattice's OWN reference
        # area, so lift equals weight there by construction — which is the
        # same identity ``build_flight_model`` back-solves when a report
        # carries no weight, taken here from the solve rather than from a
        # dict. ``prob.mission`` is not asked: this family leaves it None
        # unless a mission was applied, and a mass that silently defaulted is
        # exactly how a 66.6 kg design came to be flown at 632 kg once.
        _W = (float(sized.W_total_N) if sized is not None
              else float(CL_trim * 0.5 * prob.rho * prob.V ** 2 * dk.S))
        hq = _hq.gate(
            dk, prob.handling_level,
            mass_kg=_W / 9.80665, b_m=float(wing.b),
            body_length_m=float(_drag_mod.body_length_for_arm(x_t)),
            rho=float(prob.rho), CD0=float(CDp), oswald_e=float(res.e),
            AR_vertical=(None if _vert is None else
                         float(abs(_vert.height) ** 2
                               / max(abs(_vert.height) * _vert.chord, 1e-12))),
            breakdown=out)
        out.update(hq.report())
        out["handling_margins"] = hq
        # ...AND ONTO THE CONSTRAINT VECTOR. The static margin keeps index 0
        # and whatever else this problem already declared keeps its place, so
        # a caller that read ``g[0]`` before this flag existed still reads the
        # same margin (``fg_wing_tail``, ``constraint_labels``).
        out["g"] = np.concatenate([np.atleast_1d(np.asarray(out["g"],
                                                            dtype=float)),
                                   hq.g])
    # THE POINT THIS DESIGN WAS FLOWN AT, on every exit and not only when
    # the flight state was SEARCHED. It was written only under the modifier,
    # so a report from every other run stated no speed at all — and
    # ``flightmodel.build_flight_model``, having nothing to read, fell back
    # to 45 m/s and back-solved a mass there: 632 kg for a design that
    # weighs 66.6, the same aeroplane at 9.5x the dynamic pressure. Every
    # rate derivative and every time constant in stages 5 and 6 moved with
    # it. ``prob`` is the flown state by construction here — the flight
    # modifier REPLACES it above — so this is the point, not a restatement.
    out["V"] = float(prob.V)
    out["rho"] = float(prob.rho)
    out["mu"] = float(prob.mu)
    if flight is not None:
        out["altitude_m"] = float(flight[1])
    return out


def fg_wing_tail(x: np.ndarray, prob: WingTailProblem | None = None
                 ):
    """Constrained-harness callable.

    Without the size modifier: (L/D at trim, signed static-margin margin).
    With it: (payload L/D, [static margin, root-bending stress margin]) —
    the objective and the constraint COUNT both change, which is why the
    sized variants are their own registered problems.

    ``handling_level`` adds five more rows to whichever of those it is, and
    the FAILURE vector follows: a penalty returned at the wrong width is a
    shape error inside the optimiser rather than a bad score, so the width
    comes from ``prob.constraint_labels`` — the same declaration ``api``
    reads — and never from a literal counted out here. A scalar ``g`` is
    kept ONLY in the case that has always returned one, so no existing
    caller sees its margin become a length-1 array.
    """
    sized = prob is not None and prob.size_free
    n_g = len(prob.constraint_labels) if prob is not None else 1
    out = evaluate_wing_tail(x, prob)
    if not out["feasible"]:
        return (PENALTY, G_FAIL) if n_g == 1 else (PENALTY,
                                                   np.full(n_g, G_FAIL))
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


#: which way a tip device ON A TRIMMING SURFACE may point, as a named band.
#: The direction that pays is not the wing's: mirroring a surface mirrors
#: its lift, so a stabiliser carrying a DOWNLOAD wants the mirror of what a
#: lifting wing wants — and a tail's load changes sign with the trim state,
#: which is why "either" exists at all. ``up`` is the published band
#: (bit-for-bit; the aircraft convention is ground clearance, not
#: aerodynamics), ``down`` is its exact mirror, ``either`` is the signed
#: span the water families have always flown.
TAIL_WINGLET_DIRECTIONS: dict[str, tuple] = {
    "up": geometry.WINGLET_CANT_BOUNDS_DEG,
    "down": (-geometry.WINGLET_CANT_BOUNDS_DEG[1],
             -geometry.WINGLET_CANT_BOUNDS_DEG[0]),
    "either": geometry.WINGLET_CANT_LIMITS_SIGNED_DEG,
    # ...and the one that does not ASK. The design variable is then the
    # cant's MAGNITUDE (the published band) and the side is derived per
    # candidate from the load the trim solve puts on the surface — see
    # ``tail_winglet_follow`` and :func:`evaluate_wing_tail`. It is a band
    # here only because every direction has to be one; what makes it
    # different is the flag that travels with it.
    "follow": geometry.WINGLET_CANT_BOUNDS_DEG,
}

#: the direction that is DERIVED rather than stated
FOLLOW_DIRECTION = "follow"


def tail_cant_bounds_for(direction: str | None, wtype: str | None = None):
    """Cant band for a named tail tip-device DIRECTION and TYPE, or ``None``.

    ``None`` means NOBODY ASKED — the family keeps its own published band,
    so a run that does not set either flag is bit-for-bit unchanged. The
    named directions each mean the same thing in every family, which is the
    point of naming them: the aircraft tail's own default is ``up`` and the
    hydrofoil elevator's is ``either`` (under water the direction is a
    cavitation trade), and without an explicit band "up" would have meant
    two different things in the two media.

    ``wtype`` is the SHAPE, in the same vocabulary the wing's tip device
    uses (:data:`WINGLET_TYPES` — ``canted``, ``vertical``, ``raked``). It
    narrows the direction's band to that type's own MAGNITUDE range, keeping
    the direction's sign: a vertical fence on a surface that points down is
    (-90, -84), on one that points up (84, 90), and on one that follows the
    load the magnitude band (84, 90) with the side derived per candidate.

    The one pair that has no band is ``either`` with a type narrower than
    the full range: "near-vertical, on whichever side" is two intervals with
    a hole in the middle, and one bounds row cannot say that. It is refused
    by name rather than silently widened, and the answer it points at —
    "follow the load" — is the same statement without the hole.
    """
    if direction is None and wtype is None:
        return None
    if direction is not None and direction not in TAIL_WINGLET_DIRECTIONS:
        raise ValueError(
            f"unknown tail winglet direction {direction!r}; "
            f"choose from {sorted(TAIL_WINGLET_DIRECTIONS)}")
    band = TAIL_WINGLET_DIRECTIONS[FOLLOW_DIRECTION if direction is None
                                   else direction]
    if band is None:
        return None
    lo, hi = (float(v) for v in band)
    if wtype in (None, "canted"):
        # "canted" IS the legacy band, so it narrows nothing — asking for it
        # has to leave a published run bit-for-bit alone
        return None if direction is None else (lo, hi)
    if wtype not in WINGLET_TYPES:
        raise ValueError(f"unknown winglet_type {wtype!r}; "
                         f"choose from {sorted(WINGLET_TYPES)}")
    t_lo, t_hi = (float(v) for v in WINGLET_TYPES[wtype]["cant_bounds"])
    if lo < 0.0 < hi:
        raise ValueError(
            f"a {wtype!r} tip device on a surface whose cant may take EITHER "
            f"sign is two bands with a hole between them (|cant| in "
            f"[{t_lo:g}, {t_hi:g}] deg), which one bounds row cannot say — "
            f"state a side (up/down), or use "
            f"{FOLLOW_DIRECTION!r}, which is the same statement with the "
            f"side read off each candidate's own load")
    if hi <= 0.0:                      # a band that points DOWN: mirror it
        return (-t_hi, -t_lo)
    return (t_lo, t_hi)


def winglet_cant_bounds_for(wtype: str | None, capped: bool):
    """Cant band for a winglet TYPE (objective.WINGLET_TYPES), validated.

    Mirrors api._winglet_cant_bounds' rule so the tail family cannot become
    the one place a free-span raked tip slips through.
    """
    if wtype is None or wtype == "canted":
        return None
    if wtype not in WINGLET_TYPES:
        raise ValueError(f"unknown winglet_type {wtype!r}; "
                         f"choose from {sorted(WINGLET_TYPES)}")
    spec = WINGLET_TYPES[wtype]
    if spec["force_capped"] and not capped:
        raise ValueError(
            f"winglet_type {wtype!r} must be span-capped (raking a free-span "
            f"tip reports a fabricated L/D)")
    return tuple(spec["cant_bounds"])
