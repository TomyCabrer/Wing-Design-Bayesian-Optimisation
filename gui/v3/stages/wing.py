"""Stage 3 — WING: compose the wing type, then optimise it.

The wing TYPE is not a list of names: it is a set of choices (tip device,
planform freedom, chord law, flight state) that the registry either has a
combined solver for or does not. WHETHER there is a second surface is not
among them — a tail, an elevator or a tandem pair is a property of the
vehicle, decided in stage 1, and this stage shows it the way it shows the
medium: the value it holds, and the way back to the stage that owns it.
What IS here is that surface's GEOMETRY, asked with the wing's own three
questions in the wing's own words (``_surface_design_controls``): its
planform, its tip device, its chord law. Every menu here is built from
``gui.nice_app.option_available`` / ``airfoil_options`` /
``winglet_options``, i.e. by applying the choice, deriving the problem and
reading the problem's own inverse choices back — so a menu can never offer
something with no solver behind it, nor hide something the registry can
solve. Anything missing gets its honest reason.

This stage asks NOTHING about the section. Stage 2 chose it, and stage 3
shows only the value it holds and the way back to the stage that owns it
(the ``airfoil`` row of the configuration list). The "Section carried from
stage 2" panel that used to sit on the right — the flown-section readouts,
the treatment switches, the design-box link and its half-width — was
deleted at the user's request: it re-asked, in stage 3's words, questions
stage 2 had already answered, and reading it was the price of every visit
to this stage. The treatment choices themselves survive as ACTIONS
(``set_choice("airfoil", ...)`` / ``set_choice("fly_section", ...)``) with
no control drawing them, so a family that reshapes the section has to be
selected some other way.

Stage 2's section is still carried in here by ``config.section_link_rows``,
now always on (``W["section_link"]`` stays at its "auto" default): pinned
CST weights where the family designs its own section, a narrowed thickness
where it flies t/c, advisory where the section is fixed. The design-box
view is where that is stated — it colours the rows the link touched and
says so above them — and a row typed by hand still wins over the link.

The SIZE is this stage's too, and the PLANFORM menu is where it is decided:
one control, asking who decides the wing's size, with the size card under it
answering in whichever way that control selected.

* ``fixed`` — you choose the size, and the SPAN is the whole of that choice.
  A wing is specified by its span before anything else (a hangar, a trailer,
  a class rule, a spar), and the AREA is already the mission's: stage 1 types
  a wing loading and S = W/(W/S). So the card is one number, in metres
  (``_size_controls`` / ``session.set_span_m``), with the area and the aspect
  ratio beside it as readouts. There is NO ``b_m`` row in the design box — a
  span that is not searched has no band to state — and an untouched card
  flies √(AR_estimate · S), so the default run is bit-for-bit the published
  planform of the problem the builder selected.
* ``wing_loading`` — the span joins the design vector and the AREA leaves it
  (it follows the mission's W/S through the weight loop), which is the only
  pairing in which a span BAND is an honest question. The band is then the
  ``b_m`` row of the design box, opened around the span the card held, and
  the card becomes a readout of what that row implies.
* ``free`` — span and area are both searched, weight-coupled, and both are
  ordinary design-box rows.

Nobody types an ASPECT RATIO anywhere: it is b²/S, so it is a consequence of
the span, either the chosen one or the one the optimiser picks in the band.
Stage 2 guesses one to have a chord to design a section for, and the size
card carries the one button that re-points that guess at the wing actually
flown when the two have drifted apart.

The published aircraft-sizing FAMILY is not offered here (V1/V2 still offer
it). It is a family of its own, with no tip-device, tail or section solver,
so choosing it in a pipeline whose stage 2 designed a section would throw
that section away; ``free`` is the same freedom and composes with everything.
The menu says so rather than silently dropping the entry.

The problem opens WITH the chord law on (``nice_app.BUILDER_START``); the
chord-law control here is the one that takes it back off.

Launching builds an ``aerobo.api.RunConfig`` (gui/v3/config.py) and hands it
to the shared ``RunManager`` — the same background worker and the same
``api.run`` entry point the experiment harnesses use.
"""

from __future__ import annotations

import copy
import re
import threading
import time

import plotly.graph_objects as go

from gui import diagnose, metrics

from .. import config, figstyle, relax as relax_mod, sampler as sampler_mod, \
    session, theme, widgets

#: decimals the design-box inputs SHOW. Six is finer than any bound the
#: solvers meaningfully distinguish, and it keeps a CST weight readable
#: instead of spelling out 0.1473055393570988 in a table cell.
BOX_DIGITS = 6

#: choices that carry a NUMBER rather than select a family: they are typed
#: into a field, and no menu anywhere depends on their value. Writing one
#: must therefore NOT redraw the view its own field lives in — a rebuilt
#: input loses the focus and swallows the rest of the number being typed
#: (typing 7.5 into the arm stored 7.0; the "0.25" of an elevator chord —
#: a field this shell no longer has — stored 0.2). The same trap the size
#: card and the design box are already
#: built around; these were the choices that went through ``set_choice``.
#: Everything after the four tail rows is ``gui.nice_app.NUMBER_CHOICE_KEYS``
#: — the same list every shell keeps, written out rather than imported
#: because ``gui.nice_app`` pulls in ``aerobo.cad`` at import time and
#: importing ``gui.v3.*`` has to stay side-effect free. Stated as a COUNT
#: nowhere: this list is the one that grew (the car's area band, its force
#: budget and its downforce floor were added to the V1 list and not to this
#: one, and every field of the track card's size block then rebuilt itself
#: under the cursor). ``test_tandem_stagger`` holds the two together.
#: ...and WHICH VIEW each of them is typed into. ``VALUE_KEYS`` says "writing
#: this must not rebuild the view its own field lives in"; it used to be
#: enough to name the keys, because every one of them was a field on the TYPE
#: card and the box was safe to redraw. The car's two LIMITS broke that
#: assumption by moving: they are asked UNDER THE DESIGN BOX now (beside the
#: chord and second-surface limits, where this shell keeps limits), so
#: redrawing the box from their own handler is the very bug the list exists
#: to prevent, one view across. Named rather than inferred, because the
#: alternative is a rule that guesses which view a key was typed into.
BOX_VALUE_KEYS = (
                  # NO "car_drag_budget_n" / "car_downforce_min_n". They were
                  # the reason this list exists, and they have MOVED: both are
                  # drawn under the Maximise select on the TYPE card now,
                  # beside the objective whose other half they are. So the
                  # view that must not be rebuilt from their handler is the
                  # type card — which is what VALUE_KEYS alone already means,
                  # and naming them here would skip the box repaint they now
                  # genuinely want (the box's own read-outs follow a limit).
                  # ...and the PLATE CHORD's two metre fields, which
                  # moved under the box beside the ratio row they clip.
                  # Same trap, same list: they are typed into the box
                  # view now, so rebuilding that view from their own
                  # handler would replace the field mid-number.
                  "car_endplate_chord_min_m",
                  "car_endplate_chord_max_m")

VALUE_KEYS = ("tail_arm_m", "tail_dihedral_deg",
              "tail_cg_m", "tail_height_m", "mast_station_frac",
              "car_drag_budget_n", "car_downforce_min_n",
              # NO "car_track_points". The circuit left the card with the
              # lap-time objective, so there is no field here to protect —
              # the lap is still in the engine and still reachable through
              # api (nice_app, above _car_limit_rows, says how).
              "car_endplate_blend_frac", "car_endplate_cant_deg",
              "car_endplate_chord_min_m", "car_endplate_chord_max_m",
              "tandem_dx_m", "tandem_dz_m", "tandem_b_rear_m",
              "tandem_fin_boom_m")

#: the car's two LIMITS, as the block under the design box asks them:
#: ``(flag, label, unit, step, note)`` — the same shape ``TAIL_LIMIT_ROWS``
#: uses, and gated the same way (a row is drawn iff the derived problem
#: DECLARES the flag), so a family that stops honouring one loses its field
#: without a list here being edited.
#:
#: Both are FORCES. A coefficient allowance is referenced to the reference
#: area, which is a design variable on every car family, so it is not an
#: allowance on anything the car can feel.
CAR_LIMIT_ROWS = (
    ("drag_budget_n", "car_drag_budget_n", "Drag, no more than", "N", 5.0,
     "A ceiling on the drag this wing may make, at the stated speed. Blank "
     "is the default and means no drag constraint: the score is already "
     "downforce per unit drag, so nothing runs away without one. State a "
     "number when the car's drag budget is decided elsewhere and this wing "
     "has been given a share of it."),
    ("downforce_min_n", "car_downforce_min_n", "Downforce, at least", "N",
     25.0,
     "A floor on the downforce, which is what turns “the most efficient "
     "rear wing” into the question a race engineer actually has. "
     "Without one the answer is a small wing making little downforce — "
     "measured over this family's own area band, CZ/CD peaks near 0.14 "
     "m² at about 193 N, against 486 N at 0.4 m². With one, the "
     "answer sits ON the floor: it spends exactly the area the downforce "
     "demands."),
)

#: design-box rows that are a SPAN, in surface order — the wing's, then a
#: second surface's where the family has one of its own (a tandem pair's rear
#: wing). They lead the box together: a pair that searches two spans is being
#: asked one question twice, and the two bands belong side by side.
#: ``aerobo.sizing.span_labels`` is where the names come from.
SPAN_ROWS = ("b_m", "b_rear_m")

#: WHAT THE FOUR CANT STATES ARE CALLED on the card, keyed by
#: ``api.WING_CANTS``. Four and not two because the dihedral and the sweep
#: are two questions: one buys Cl_beta at any lift and is what the `spiral`
#: criterion reads, the other buys a neutral point and costs L/D while its
#: own dihedral effect goes as CL and is gone at cruise. With one toggle a
#: design that wanted the second had to accept the first, and the searched
#: dihedral of an L/D-only run rides the ANHEDRAL bound.
WING_CANT_LABELS = {"fixed": "you state them",
                    "dihedral": "optimise the dihedral",
                    "sweep": "optimise the sweep",
                    "free": "optimise both"}

#: the smallest gap the two aspect-ratio limits may be switched on with.
#: ``api.ar_limits_of`` refuses ar_max <= ar_min outright, and a pair a
#: rounding apart is an empty search rather than a limit, so a limit switched
#: on beside a live one is placed at least this far from it (or refused, and
#: said so — see ``_ar_limit_default``). In aspect-ratio units, on a solver
#: band of 3-40.
AR_LIMIT_GAP = 0.5

#: how a design-box row NAMES the surface it belongs to, so the box can be
#: one table PER SURFACE. The registry has one convention for it and it is a
#: SEGMENT of the label, ``_t``: ``S_t_m2``, ``AR_t``, ``l_t_m``, ``z_t_m``,
#: ``taper_t``, ``washout_t_deg``, ``chord_k1_t``, ``winglet_cant_t_deg``,
#: ``winglet_h_frac_t``. Matched as a segment and never as a substring —
#: ``tc``, ``taper``, ``tc_front`` and ``endplate_tc`` all contain a ``t``
#: and not one of them is a second-surface row. Verified against every label
#: the registry declares (``test_v3_two_design_boxes``), because a rule that
#: silently misfiles one row puts a tail constraint in the wing's table.
#:
#: A TANDEM pair is deliberately NOT split by this. Its rear wing is marked
#: ``_rear``, and the pair's two spans belong side by side — reading them
#: apart is what hid the one comparison the pair is for (``_box_items``), so
#: a pair stays one table.
AFT_ROW = re.compile(r"(?:^|_)t(?:$|_)")

#: draws a mission recommendation is measured over (``aerobo.recommend``).
#: A refusal returns before its solver, so most of this is cheap exactly on
#: the boxes worth recommending for; 128 is what leaves enough admissible
#: draws to bound a box of this dimension after the best quarter is kept.
#: It is a button, not a keystroke — the cost is paid when it is asked for.
RECOMMEND_DRAWS = 128

#: how much narrower a row has to get before it is worth PROPOSING. A finite
#: sample bounds almost every row a hair inside its own edges — the reference
#: tail's arm came back 3-7.98 against a published 3-8, which is 99.6 % of the
#: width and says nothing — and a card that lists those beside a row that
#: genuinely halved buries the one finding it had. Rows under the threshold
#: are counted, not hidden: the card says how many held their full width.
RECOMMEND_MIN_SHRINK = 0.05

#: the chord-distribution limits, as the card asks them (api.CHORD_LIMIT_KEYS):
#: ``(flag, label, unit, step, note)``. These are the questions the
#: coefficient rows of the design box cannot ask — a chord law's minimum
#: chord, its maximum and its steepest local taper angle are all nonlinear in
#: the coefficients once the area rescale is applied, so no box on ``chord_k*``
#: can state them.
CHORD_LIMIT_ROWS = (
    ("chord_min_m", "minimum chord", "m", 0.05,
     "No station may be narrower than this. It is what a spar depth, a "
     "control-surface hinge or the section's own Reynolds number really "
     "asks for — the solver's only chord floor otherwise is the 5% "
     "collapse guard."),
    ("chord_max_m", "maximum chord", "m", 0.05,
     "No station may be wider than this: the mould, the billet, the "
     "trailer."),
    ("chord_rate_max_deg", "max chord rate", "deg", 2.0,
     "The steepest LOCAL taper angle, atan(|dc/dy|), anywhere on the span. "
     "A straight taper has one value of it; a free chord law has a "
     "distribution, and its peak is what a buildable mould line limits."),
)

#: the wing's ASPECT-RATIO limit, as the card asks it:
#: ``(flag, label, unit, step, note)`` — api.AR_LIMIT_KEYS.
#:
#: It sits with the chord limits because it is the same kind of statement: a
#: quantity an engineer holds that no ROW of the design box can hold. The box
#: states a span and an area, and b²/S is a diagonal across the two — every
#: span x area rectangle spans a RANGE of aspect ratios (the shipped 3.6-24 m
#: span against a 0.8-2.2x area band reaches AR 1.3-160) — so a limit on the
#: ratio can only be a limit, never a bound.
#:
#: It does BOTH things the card can do: the size box is clipped so no corner
#: of it leaves the band (:func:`session.clip_size_box`, which already does
#: this for the solvers' own band), and every candidate is checked against it
#: where the size is resolved (``sizing.check_ar_limit``), because a clip on
#: two rows cannot make a rectangle out of a region that is not one.
AR_LIMIT_ROWS = (
    ("ar_min", "minimum aspect ratio", "b²/S", 0.5,
     "No design narrower in span than this for its area. It is what a "
     "glide ratio target, a class rule or a structural span limit asks — "
     "and below it the induced drag the whole search is trading against "
     "stops being the number that decides the design."),
    ("ar_max", "maximum aspect ratio", "b²/S", 0.5,
     "No design longer in span than this for its area: the spar, the "
     "hangar, the trailer, the roll rate. The solvers' own ceiling is 40 "
     "(sizing.AR_LIMITS) and this can only narrow it."),
)

#: the SECOND surface's own size, in the units a builder has:
#: ``(flag, label, unit, step, note)``. The design box states that surface as
#: an area (``S_t_m2``) and, where its planform is designed, an aspect ratio
#: (``AR_t``) — the solver's variables, and neither is a question anybody
#: asks. Span and chord are. Both directions are exact
#: (b = sqrt(AR·S), c = sqrt(S/AR)), so stating these NARROWS the searched
#: box rather than only refusing draws out of it (api.TAIL_LIMIT_KEYS →
#: tail.TailLimits).
#:
#: The SPAN is not here. A low and a high on a span is a BAND, which is what
#: every row of the design box already is — and asking it in a card of its
#: own put the one number a reader compares it against (``b_m``, the wing's
#: own span, the first row of that box) two panels away. It is asked as
#: :data:`TAIL_SPAN_ROW`, in the box, directly under the wing's.
TAIL_LIMIT_ROWS = (
    ("tail_chord_min_m", "minimum chord", "m", 0.02,
     "No station of it narrower than this. On a small aeroplane it is the "
     "Reynolds number that asks: a 60 mm chord at 12 m/s is Re 5e4, where "
     "the section tables this solver reads no longer describe the flow."),
    ("tail_chord_max_m", "maximum chord", "m", 0.02,
     "No station wider than this — the mould, or the room left between the "
     "wing's trailing edge and the back of the aeroplane."),
)

#: the second surface's SPAN, as the design box asks it: ``(low flag, high
#: flag)``. Not a design-vector row — that surface is searched as an area
#: (``S_t_m2``) and, where its planform is designed, an aspect ratio
#: (``AR_t``) — but b = sqrt(AR·S) is exact in BOTH directions, so a band
#: typed here narrows the box the sampler draws from
#: (``tail.TailLimits.narrow``) exactly as a band on a design variable does.
#: It leads the box beside :data:`SPAN_ROWS`, because "the tail came out
#: wider than the wing" is a comparison between two rows of one table.
TAIL_SPAN_ROW = ("tail_span_min_m", "tail_span_max_m")

#: a design-box row that is a chord-law COEFFICIENT: ``chord_k2``,
#: ``chord_front_k1`` (a tandem carries one law per wing), ``chord_k3_t`` (a
#: designed tail carries its own). The law is a property of a SURFACE, so the
#: rows are grouped by ``(wing prefix, surface suffix)`` and each group is
#: asked as one question.
CHORD_ROW = re.compile(r"chord_(?:(front|rear)_)?k(\d+)(_t)?$")

#: per chord-law group: ``(surface, the design-box row carrying the
#: straight-taper baseline that law multiplies)``. The baseline matters: the
#: area rescale integrates against the trapezoid, so the same coefficients
#: bend a rectangular wing and a sharply tapered one by different amounts.
CHORD_GROUPS = {
    ("", ""): ("wing", "taper"),
    ("", "_t"): ("second surface", "taper_t"),
    ("front", ""): ("front wing", "taper_front"),
    ("rear", ""): ("rear wing", "taper_rear"),
}

#: the colour of a design-box row's provenance chip, by where the band came
#: from. Lifted out of ``_box_row`` so that a typed bound — which may not
#: rebuild the row its own input sits in — can still repaint the chip.
_SOURCE_COLOR = {"default": theme.INK_FAINT, "section": theme.ACCENT,
                 "mission": theme.ACCENT, "recommended": theme.GOOD,
                 "user": theme.WARN, "released": theme.INK_FAINT,
                 # ...and the one the SHELL moved on the user's behalf. It
                 # had no entry, so a floored dihedral row drew the same
                 # grey as "default" and the box said a family published a
                 # low bound of 0 where it publishes -10.
                 "unpriced": theme.WARN}

#: what the chord law IS, said once. The coefficients are the solver's
#: variables and nobody's design intent, so the panel says what they do and
#: then asks its question in the planform instead.
CHORD_LAW_NOTE = (
    "The law reshapes the straight-taper chord — c(η) = c_trap(η) · m(η), "
    "η = 0 at the root and 1 at the tip — and rescales it to hold the area "
    "exactly, so it moves chord ALONG the span without resizing the surface. "
    "WHICH shape m(η) is is the first question below; the coefficients of it "
    "are what the solver searches, and what a design box has to say about "
    "them is how far they may bend the planform, which is asked here in the "
    "planform itself and stored in the coefficients underneath."
)

CHORD_TREND_LABELS = {
    "free": "unconstrained — the law may bulge where the loading wants it",
    "root_largest": "the root is the largest chord (never grows outboard)",
    "root_smallest": "the root is the smallest chord (never falls outboard)",
}

CHORD_TREND_NOTES = {
    "free": "The published behaviour: the chord distribution is judged by "
            "the loading it draws, not by its shape.",
    "root_largest": "The conventional planform, stated as a constraint: the "
                    "chord never grows outboard, so the widest section is at "
                    "the root where the bending moment is.",
    "root_smallest": "The inverse: the chord never falls outboard. Asked for "
                     "where the TIP has to carry chord — a tip device's root, "
                     "a surface whose outboard Reynolds number is the binding "
                     "one.",
}

#: propeller keys that change WHICH controls exist — the enable switch and
#: the single/pair toggle. Only those two need the solver view rebuilt, and
#: neither is a field with a caret in it; every other propeller key is a
#: number typed into that same view (see ``VALUE_KEYS``).
PROP_SHAPE_KEYS = ("enabled", "layout")

#: what the wing search may do with the section stage 2 chose, named for
#: THAT SECTION rather than for the solver family behind it. Ordered: fly
#: it, reshape it, then the two that throw it away. ``section_only`` (the
#: 2-D section problem) is deliberately absent — it has no wing to fly on,
#: and designing a section is what stage 2 is for.
TREATMENTS = {
    "fixed": "fly it — one section table, this one",
    "section_wing": "reshape it — CST weights searched with the wing "
                    "(live XFOIL per candidate)",
    "tc_sweep": "discard it — search thickness t/c on the NACA 24XX family "
                "instead",
    "coupled": "discard it — pick from the pre-optimised (t/c × cl) section "
               "library instead",
}

#: why V3's planform menu does not offer the aircraft-sizing FAMILY, in the
#: same "not offered here" note every other missing entry uses
#: (nice_app.PLANFORM_OPTION_WHY, which V1/V2 read unchanged — they still
#: offer it). V1's reason is about the tip device and the tail; V3's is one
#: step stronger, because this shell is a pipeline: the family has no
#: designed-section solver either, so stage 2's answer would not be flown.
PLANFORM_AIRCRAFT_WHY = (
    "it is a family of its own — no tip-device, tail or designed-section "
    "solver — so in a pipeline that has already chosen a section it can only "
    "take things away: the section stage 2 designed would not be the one "
    "flown. 'free span + area (weight-coupled)' is the same freedom, sizes "
    "the wing the same way, and composes with everything on this stage"
)

#: the V-tail's note, as THIS shell has to tell it. V3 does not offer the
#: fin-drag switch V1/V2 carry, because the fin is not a modelled surface
#: anywhere in the package: it has no panels, makes no lift (the solvers are
#: symmetric, so there is no yaw for it to answer), and appears in neither
#: geometry.py nor cad.py. It exists as two Raymer scalars only —
#: ``tail.vtail_cd0`` and the T-tail's height — so charging its drag is a
#: guess dressed as physics, and V3 declines to make it. The consequence has
#: to be said out loud rather than left as a condition on a switch this
#: shell does not have: with the fin free, deleting it buys nothing, so the
#: V-tail here can only ever pay the cost of dihedral.
V_TAIL_NOTE_V3 = (
    "Modelled as the equivalent flat tail of area S_t·cos²Γ (a panel at "
    "dihedral Γ both sees and returns cos Γ), while profile drag is still "
    "charged on the FULL panel area — dihedral costs pitch effectiveness "
    "without saving wetted area. The fin this layout exists to delete IS "
    "charged on every other layout now, so the comparison is finally a fair "
    "one: measured on the tail family, a V-tail gives up 0.11 of static "
    "margin and 0.04 % of L/D to the dihedral, and takes 6.4 % of L/D back "
    "by having no fin to drag. Read it as a verdict on the configuration, "
    "which is what it now is."
)


def WING_OBJECTIVE_CHOICES() -> dict:
    """The scalars this stage's search can maximise, as a select's options.

    ``lod`` is THE FAMILY'S OWN objective and it is not a question: L/D at
    the trimmed design lift, payload L/D where the size modifier makes the
    weight a variable, downforce over drag for a car wing. Which of those is
    honest here is decided by the configuration, not by preference.

    ``composite`` is the question that IS the user's. A wing has ten things
    worth having — cruise L/D, the CL and the angle it stalls at, span
    efficiency, spar load, mass, internal volume, a chord the shop can build,
    a flat deck angle — and maximising the first alone spends the other nine
    silently. Picking this maximises the weighted composite of all ten
    (``aerobo.wing_score``), against a band measured over this design box.
    """
    return {"lod": "the family's own objective (L/D)",
            "composite": "composite score of your criteria"}


def ws_ratchet_note(S: dict) -> tuple | None:
    """What a SEARCHED wing loading will do, said before the run does it.

    ``(warning, follow_up_or_None)``, or ``None`` when the loading is not
    searched and there is nothing to warn about.

    Not a refusal. The row is legal and the physics behind it is sound: a
    searched W/S has no interior optimum in this package (128 runs,
    ``results/ws_band_study.json``) because a smaller wing at the same weight
    really is more efficient and nothing scored here prices the thrust that
    costs. What the user cannot see without being told is the CONSEQUENCE —
    the row is not being traded, it is being driven to its ceiling, so the
    number that decides the answer is the band's top, which is the mission's
    and not the optimiser's.

    Deliberately NOT a nudge towards the other objective. The measurement
    says the composite parks HARDER than L/D (16/16 against 12/16 on the trim
    wing), so "pick the composite instead" is advice its own data refutes.
    """
    from gui.v3 import session

    if not session.loading_is_searched(S):
        return None
    cap = session.mission_ws_ceiling(S)
    d = session.ws_diagram(S)
    matched = d is not None and d.ws_matched_pa is not None
    stated = session.ws_cap_source(S) == session.WS_CAP_STATED
    if stated and cap:
        # the ceiling is the user's own number here, so naming the diagram
        # would credit this band to a card that is binding nothing
        where = f"the W/S you stated ({cap:.4g} Pa), which you made the ceiling"
    elif matched and cap:
        where = (f"the matching point of your constraint diagram "
                 f"({cap:.4g} Pa)")
    elif cap:
        where = f"the mission's {d.binding or 'own'} limit ({cap:.4g} Pa)"
    else:
        where = "the top of the band you set"
    warn = (
        f"The wing loading is SEARCHED, and it has no interior optimum in "
        f"this package: measured over 128 runs, both objectives rise "
        f"monotonically with W/S and finish at the top of their band (the "
        f"composite parks harder than L/D, 16/16 against 12/16). That is "
        f"honest physics against an incomplete objective — a smaller wing at "
        f"the same weight really is more efficient, and nothing scored here "
        f"pays for the thrust it costs. So expect this row to end at {where}, "
        f"and read the ceiling as the answer rather than the search.")
    if matched or stated:
        # nothing to advise on a stated ceiling: closing the diagram would
        # change a card that is not deciding this band
        return warn, None
    return warn, (
        "State the thrust-to-weight you actually have, in stage 1, and the "
        "diagram closes: the ceiling becomes the loading the engine can fly "
        "rather than the one the stall allows.")


def wing_weight_meta() -> dict:
    """``{key: (label, why)}`` for every wing criterion, from the physics.

    Read off ``wing_score.WING_CRITERIA`` rather than restated here: the
    label a user weights and the label the report prints are the same
    sentence, and a criterion added there appears on this card with no edit.
    """
    from aerobo import wing_score as wsc

    return {c.key: (c.label + (f" [{c.unit}]" if c.unit else "")
                    + ("  (lower is better)" if c.sense == "min" else ""),
                    c.help)
            for c in wsc.WING_CRITERIA}


def _cell(v, nd: int = 2, signed: bool = False) -> str:
    """One score cell: a number, or an em dash when it was not measured."""
    if v is None:
        return "—"
    return f"{float(v):+.{nd}f}" if signed else f"{float(v):.{nd}f}"


def wing_score_rows(report: dict | None) -> list[dict]:
    """BASELINE vs OPTIMISED on every wing criterion, as table rows.

    Two rows per criterion would be two tables, so each row carries the
    0-100 SUB-SCORE (what J is built from) and states the raw metric with its
    unit beside the name — the number a user recognises, and the number the
    weight actually multiplies, in one line.

    A criterion the band does not cover is dropped rather than shown as a
    zero; a criterion with zero weight is kept, because "this counted for
    nothing" is exactly what the composite's arithmetic hides.

    Every row carries ``dir`` — the verdict the change earns on a 0-100
    up-is-better band (:func:`gui.metrics.verdict`) — so the table can be
    painted green/red/grey instead of leaving the reader to work the sign of
    each row out of a column of signed numbers. A criterion weighted 0 is
    left UNJUDGED for the same reason it is left in: it is not in J, so
    calling its fall a loss would be a complaint about a search that was
    never defending it.
    """
    if not report:
        return []
    from aerobo import wing_score as wsc

    base = report.get("baseline") or {}
    opt = report.get("optimised") or {}
    delta = report.get("delta") or {}
    weights = report.get("weights") or {}
    rows = [{"metric": "composite score J",
             "original": _cell(base.get("composite")),
             "new": _cell(opt.get("composite")),
             "change": _cell((delta.get("composite")), signed=True),
             "dir": metrics.verdict(delta.get("composite"))}]
    d_scores = delta.get("scores") or {}
    for crit in wsc.WING_CRITERIA:
        key = crit.key
        b_val = (base.get("scores") or {}).get(key)
        o_val = (opt.get("scores") or {}).get(key)
        if b_val is None and o_val is None:
            continue
        b_raw = (base.get("metrics") or {}).get(key)
        o_raw = (opt.get("metrics") or {}).get(key)
        unit = f" {crit.unit}" if crit.unit else ""
        w = float(weights.get(key, 0.0))
        rows.append({
            "metric": f"{crit.label} · weight {w:.2f}",
            "original": f"{_cell(b_val, 1)}  ({_cell(b_raw, 3)}{unit})",
            "new": f"{_cell(o_val, 1)}  ({_cell(o_raw, 3)}{unit})",
            "change": _cell(d_scores.get(key), 1, signed=True),
            "dir": metrics.verdict(d_scores.get(key)) if w > 0.0 else ""})
    return rows


def build(ctx):     # noqa: PLR0915  (one stage, built whole)
    from nicegui import ui

    from aerobo import api, materials
    from gui import nice_app as v1

    S = ctx.S
    W = S["wing"]
    seen = {"version": -1, "sampler": -1,
            "band": None, "score": None, "relax": None, "probe": None,
            "auto": None,
            # the QUEUE that has been reported on, and per-job state: which
            # jobs have already been logged, and the record each landed with
            # (see the publication rule in ``poll``)
            "queue": None, "published": set(), "records": {}}
    #: containers (and the two size widgets) a control can redraw WITHOUT
    #: redrawing the view it sits in — a rebuilt input loses the focus and
    #: swallows the rest of the number being typed
    boxes: dict = {}
    #: the value label beside each criterion-weight slider, so a slider can
    #: write its own number without redrawing the card it sits in
    weight_labels: dict = {}
    #: ``{box row label: its provenance chip}``. The chip ("default", "user",
    #: "recommended", …) sits in the same row as the two number inputs, so
    #: the only thing that can redraw it is ``_render_box`` — which a typed
    #: bound may not call, because it would destroy the input the digits are
    #: arriving in. Kept here so the ONE thing a typed bound changes about
    #: the chip, "this band is yours now", can be written into it in place.
    box_tags: dict = {}
    #: ``{row label: [low input, high input]}`` — the design box's own number
    #: fields, kept for the same reason ``box_tags`` is: a control OUTSIDE a
    #: row can move that row (the aspect-ratio limit clips the span; the
    #: pair's two areas write the split), and rebuilding the table to show it
    #: would destroy whatever field is being typed into
    #: (:func:`_repaint_row_values`).
    box_inputs: dict = {}
    #: live named metrics of the incumbent (see gui/v3/sampler.py). The
    #: sampler costs one design_report per IMPROVEMENT, not per evaluation.
    live = sampler_mod.IncumbentSampler()
    #: ``picked`` is whether the USER has ever ticked a metric. Until they
    #: have, the panel plots the catalogue's answer for this run
    #: (``metrics.live_metric_defaults``) rather than a stored list, so a
    #: weighted run opens on ``composite`` and an L/D run on ``LoD`` without
    #: either name being hard-coded here.
    live_cfg = {"on": True, "keys": [], "picked": False}
    #: the reasons the planform menu gives for an entry it does not offer —
    #: the registry's own, with V3's stronger one for the aircraft family
    PLANFORM_V3_WHY = dict(v1.PLANFORM_OPTION_WHY,
                           aircraft=PLANFORM_AIRCRAFT_WHY)
    #: the layout notes, with V3's own V-tail entry (see V_TAIL_NOTE_V3):
    #: V1/V2's wording is conditional on a switch ("with fin drag left
    #: off..."), and there is no switch any more — the charge is
    #: unconditional, so this shell states the measured verdict instead
    TAIL_TYPE_V3_NOTES = dict(v1.TAIL_TYPE_NOTES, v_tail=V_TAIL_NOTE_V3)

    def sp():
        return api.PROBLEM_SPECS[W["problem"]]

    # ------------------------------------------------------- choice edits
    def set_choice(key: str, value, notify: bool = True):
        ch = W["choices"]
        if ch.get(key) == value:
            return
        ch[key] = value
        if key in v1.SPECIAL_KEYS and value != v1.BUILDER_DEFAULTS[key]:
            losers = [k for k in v1.SPECIAL_KEYS
                      if k != key and ch[k] != v1.BUILDER_DEFAULTS[k]
                      and not v1.specials_compatible(key, k, ch)]
            for k in losers:
                ch[k] = v1.BUILDER_DEFAULTS[k]
            if losers and notify:
                names = ", ".join(v1._SPECIAL_LABEL.get(k, k) for k in losers)
                ui.notify(f"Reset {names} — no combined solver with "
                          f"{v1._SPECIAL_LABEL.get(key, key)}", type="info")
                ctx.log(f"reset {names}: no combined solver with "
                        f"{v1._SPECIAL_LABEL.get(key, key)}", "warn")
        dropped = v1.normalise_choices(ch, keep=key)
        if dropped and notify:
            names = ", ".join(v1._SPECIAL_LABEL.get(k, k) for k in dropped)
            ctx.log(f"reset {names}: no solver for it in this configuration",
                    "warn")
        before = W["problem"]
        notes = session.apply_choices(S)
        if W["problem"] != before:
            ctx.log(f"solver: {W['problem']}", "info")
            for n in notes:
                ctx.log(n, "warn")
        # a typed NUMBER that selected no new problem changes no menu, so the
        # TYPE view stands: rebuilding it from the handler of a field INSIDE
        # it destroys that field mid-number (VALUE_KEYS). The other views are
        # redrawn as usual — they hold no field being typed, and one of them
        # genuinely follows the number (the car's span BAND bounds the design
        # box's b_m row).
        if key in VALUE_KEYS and W["problem"] == before:
            # ...and not the view the field is IN. For most of these that is
            # the type card; for the car's limits it is the box (see
            # BOX_VALUE_KEYS), and rebuilding it here replaced every number
            # input under the cursor mid-digit
            if key not in BOX_VALUE_KEYS:
                _render_box()
            _render_solver()
            # ...but the fin's own read-out IS rebuilt, from its own
            # container inside the type card. Four of these keys move the
            # surface the Vertical tail card quotes — the pair's stagger
            # (both ends of it place the fin's foot and its arm), the
            # tailplane's arm and its height — and the skip above is exactly
            # what left that card describing the fin of the previous number.
            # It is a sub-container, never the container a field is in, so
            # the focus trap this branch exists for is not reopened.
            _render_fin_derived()
            ctx.refresh()
            return
        _render_type()
        _render_box()
        _render_solver()
        ctx.refresh()

    def _set_planform(value: str):
        """WHO DECIDES THE WING'S SIZE — the one control that asks it.

        Not ``set_choice``: entering or leaving the wing-loading mode also
        opens or removes the ``b_m`` row of the design box (the band is the
        question that mode asks, and it is nobody's question in the others),
        and that pairing lives in ``session.set_planform`` so a preset or a
        test reaches the same state as the menu does.

        The whole stage is redrawn because this changes the PROBLEM: the size
        card asks a different question, the design box gains or loses a row,
        and — for the two sized modes — the score becomes payload L/D with a
        root-bending stress constraint (sizing.py), which the solver panel
        reports.
        """
        ch = W["choices"]
        if ch.get("planform") == value:
            return
        before = W["problem"]
        notes = session.set_planform(S, value)
        if W["problem"] != before:
            ctx.log(f"solver: {W['problem']}", "info")
        for n in notes:
            ctx.log(n, "warn")
        if ch.get("planform") != value:
            # derive_problem refused it (no such twin in this configuration);
            # the notes above say why, and the menu falls back with the state
            ui.notify("this family has no solver for that planform",
                      type="warning")
        if session.span_is_searched(S):
            box = session.span_box(S)
            if box is not None:
                ctx.log(f"span: searched over {box[0]:.4g}–{box[1]:.4g} m, "
                        f"area from the mission's wing loading", "info")
        elif api.resizable(W["problem"]):
            size = session.flown_size(S)
            if size:
                ctx.log(f"span: fixed at {size[0]:.4g} m "
                        + ("(chosen)" if session.chosen_span(S) is not None
                           else "(stage 2's aspect-ratio estimate)"), "info")
        _render_type()
        _render_box()
        _render_solver()
        # the sized modes change the OBJECTIVE (payload L/D) and the trim
        # target, which stages 1 and 2 both quote
        ctx.render_when_shown("mission")
        ctx.render_when_shown("airfoil")
        ctx.refresh()

    #: the water SIZE select, re-bound on every render of the type card.
    #: Held because the heel field can change its options without the card
    #: being redrawn — see where it is created.
    _wet_size_el: dict = {}

    def _set_wet_size(value: str):
        """WHO DECIDES A WATER CRAFT'S SIZE (session.WET_SIZE_MODES).

        The water twin of :func:`_set_planform`, and it does the same one
        thing that matters: the mode opens or removes the ``b_m`` / ``S_m2``
        rows of the design box, so the box has to be redrawn with it. It
        does NOT change the family — the rows are flags on the same problem,
        not a different solver — which is why there is no ``derive_problem``
        here and no note about a twin that might not exist.
        """
        # ...unless this select is being re-optioned rather than answered:
        # see the guard in _set_wet_heel for why a display write must not
        # store an answer or rebuild this card.
        if _wet_size_el.get("quiet"):
            return
        if not session.set_wet_size_mode(S, value):
            return
        session.apply_choices(S)
        rows = [r for r in ("b_m", "S_m2")
                if r in session.searched_labels(S)]
        ctx.log("size: " + ("fixed at the stated span and area"
                            if not rows else
                            f"searched — {' and '.join(rows)} "
                            f"{'is' if len(rows) == 1 else 'are'} now a row "
                            f"of the design box"), "info")
        _render_type()
        _render_box()
        _render_solver()
        ctx.render_when_shown("airfoil")
        ctx.refresh()

    def set_winglet(shape: str):
        """The tip-device SHAPE (v1.WINGLET_SHAPES) — one question, three
        answers plus none. It writes the span accounting AND the blend value
        the shape implies, so nothing else on screen has to be asked."""
        ch = W["choices"]
        before = ch.get("winglets", "none")
        v1.set_winglet_shape(ch, shape)
        wl = ch["winglets"]
        if before != wl:
            # set_choice re-applies the family rules for the new speciality;
            # it compares against the value already written, so the choice is
            # restored first and set through the one function that owns them
            ch["winglets"] = before
            set_choice("winglets", wl)
        else:
            v1.normalise_choices(ch, keep="winglets")
            session.apply_choices(S)
            _render_type()
            _render_box()
            ctx.refresh()

    # ------------------------------------------------------ view: type
    def _row(label: str):
        row = ui.row().classes("w-full items-center gap-3 no-wrap")
        with row:
            ui.label(label).classes("field-label").style(
                f"color:{theme.INK_MUTED};min-width:118px")
        return row

    def _missing(offered, labels, why):
        note = v1.missing_options_note(offered, labels, why)
        if note:
            widgets.hint(note)

    def _planform_control(pf_opts):
        """WHO DECIDES THE SIZE — the menu, for the families that have one.

        Split out of :func:`_render_type` when the track stopped having one:
        a rear wing searches both of its dimensions, so the single entry this
        select offered it ("fixed span + area (you choose the size)") named a
        mode that does not exist on that family and contradicted the hint
        directly under it.
        """
        ch = W["choices"]
        with _row("planform"):
            sel = ui.select(
                pf_opts,
                value=(ch["planform"] if ch["planform"] in pf_opts
                       else "fixed"),
                on_change=lambda e: _set_planform(e.value)) \
                .props("outlined dense").classes("grow min-w-0")
            if len(pf_opts) <= 1:
                sel.disable()
                # ...and say WHY, in this family's own terms. "calibrated
                # size" was true of the water families until they became
                # resizable and was never true of the car, whose span is a
                # design VARIABLE with a band of its own.
                #
                # The question is whether the span is a ROW OF THE BOX, so it
                # is asked of the box (session.span_box) — not of
                # session.span_rows, which counts the surfaces that HAVE a
                # span in either state and so answers "searched" for every
                # family that can merely be given one.
                sel.tooltip(
                    "this family SEARCHES its span — its min and max are a "
                    "row of the design box, so there is no sizing mode to "
                    "pick"
                    if session.span_box(S) is not None else
                    "this family has one sizing mode: you state the span and "
                    "area below")
        _missing(pf_opts, v1.PLANFORM_CHOICE_LABELS, PLANFORM_V3_WHY)
        widgets.hint(v1.PLANFORM_CHOICE_NOTES.get(ch["planform"], ""))
        _size_controls()

    def _render_type():     # noqa: PLR0915
        box = ctx.views[("wing", "type")]
        box.clear()
        # ...AND EVERY HANDLE INTO IT GOES WITH IT. These five containers
        # are created inside this card and REPAINTED from outside it —
        # ``set_choice``'s VALUE_KEYS branch and ``_render_derived_geo`` both
        # reach them without rebuilding the card — and each is created by a
        # block several configurations never draw (no tail, no fin, no cant
        # panel; a car, a water craft). Left behind, the handle points into
        # the subtree this ``clear()`` has just deleted and the next repaint
        # calls ``clear()`` on it: "an element has been deleted but is still
        # being used", reproduced by switching a wing+tail off and then
        # typing any number. A card that is not drawn owns no box; the
        # blocks below re-add the ones they draw.
        for _stale in ("fin_derived", "cg", "stability",
                       "cant_note", "cant_note_kw"):
            boxes.pop(_stale, None)
        ch = W["choices"]
        water = ch["medium"] == "water"
        track = ch["medium"] == "track"
        tandem = ch["system"] == "tandem"
        with box:
            with ui.row().classes("w-full items-start gap-3 no-wrap"):
                with ui.column().classes("gap-3").style("flex:3 1 0;"
                                                        "min-width:0"):
                    with widgets.group_box("Configuration"):
                        # THE MOUNT IS THE FIRST QUESTION. On a rear wing it
                        # is the question that decides the others — it picks
                        # the solver family, it says whether the plate is a
                        # designed structure or a tip device, and everything
                        # the card asks about the plate below follows from
                        # it. Asked last (which is where it used to sit, at
                        # the foot of a list under the tip device it decides)
                        # the reader answered four questions before the one
                        # that changes their meaning.
                        if track:
                            v1._car_mount_controls(ch, set_choice,
                                                   W["problem"])
                        with _row("medium"):
                            ui.label(session.MEDIA[S["medium"]][0]) \
                                .classes("readout")
                            ui.button("change in stage 1", icon="edit",
                                      on_click=lambda: ctx.select(
                                          "mission", "operating")) \
                                .props("flat dense size=sm no-caps")
                        # THE TRACK DOES NOT REPEAT STAGE 1. A car has one
                        # lifting surface and no tail, and stage 1 now says
                        # so in one sentence; a second read-out of the same
                        # fact here (and a third in the "tail" row at the
                        # foot of this list) was the redundancy that made
                        # this card read as a list of things the car cannot
                        # do. What stage 3 owns for a rear wing is below:
                        # the mount, the size box, the objective, the
                        # elements and the plates.
                        if not track:
                            with _row("lifting system"):
                                # one owner, stage 1: how many surfaces carry
                                # the weight is a property of the aircraft,
                                # and it moves the reference area the mission
                                # is stated on — so it is decided with the
                                # mission
                                ui.label("tandem pair (front + rear)"
                                         if tandem else
                                         "one lifting surface") \
                                    .classes("readout")
                                if water:
                                    widgets.hint("water → hydrofoil solver "
                                                 "(single surface)")
                                else:
                                    ui.button(
                                        "change in stage 1", icon="edit",
                                        on_click=lambda: ctx.select(
                                            "mission", "operating")) \
                                        .props("flat dense size=sm no-caps")
                        # ...and if that system IS a pair, WHERE the second
                        # wing sits is asked here, under the row that declares
                        # it. The separation is what makes a tandem a tandem —
                        # the mutual induction the family exists to model is a
                        # function of exactly these two lengths — so it is
                        # configuration, stated beside the configuration it
                        # belongs to, and never a design variable. (It used to
                        # sit at the foot of this list, below the tail and the
                        # car rows, which put the pair's defining dimension
                        # further from the pair than the chord law.)
                        if tandem and not (water or track):
                            # ...but NOT the rear wing's span: a span is a
                            # span whichever wing carries it, and both are
                            # asked together on the size card below
                            # (_size_controls). Asked here as well it was the
                            # same question in two cards, and the two answers
                            # could disagree on screen.
                            # ...and NOT the fin's boom either: the pair's
                            # fin station is a question about the FIN, and
                            # this shell has a card for the fin. Asked here
                            # it sat between a paragraph about the rear
                            # wing's stagger and one about the rear wing's
                            # span, with no sentence of its own, while the
                            # surface it places was drawn nowhere at all.
                            v1._tandem_controls(ch, set_choice,
                                                with_span=False,
                                                with_fin=False)

                        # the airfoil is stage 2's answer, so it appears the
                        # way the medium does — the value it holds and the
                        # way back to the stage that owns it, never a second
                        # menu to choose it again
                        with _row("airfoil"):
                            ui.label(session.section_summary(S)) \
                                .classes("readout")
                            ui.button("change in stage 2", icon="edit",
                                      on_click=_back_to_section) \
                                .props("flat dense size=sm no-caps")

                        # ONE question about the tip device: what SHAPE is
                        # it? The span accounting (does it extend the span or
                        # is the wing shrunk to pay for it?) is answered by
                        # the honest rule rather than by the user, and the
                        # blend is a build decision carried as a VALUE — so a
                        # blended tip costs the same two design variables a
                        # canted one does, its height and its cant.
                        wl_opts = v1.winglet_shapes(ch)
                        if track:
                            # ...and on the track only where the plate is a
                            # DEVICE. Under the endplate mount the plate is
                            # the load path: its section and its root blend
                            # were asked under the Mount above, and a
                            # "shape" row here would be the same question in
                            # a second vocabulary. `car_tip_shapes` returns
                            # nothing there, and nothing is drawn.
                            if v1.car_tip_shapes(ch, W["problem"]):
                                _car_tip_control()
                        else:
                            with _row("tip device"):
                                key = v1.winglet_shape_key(ch)
                                sel = ui.select(
                                    wl_opts,
                                    value=key if key in wl_opts else "none",
                                    on_change=lambda e: set_winglet(e.value)) \
                                    .props("outlined dense") \
                                    .classes("grow min-w-0")
                                if len(wl_opts) <= 1:
                                    sel.disable()
                                    sel.tooltip("no tip-device solver for "
                                                "this configuration")
                            widgets.hint(v1.WINGLET_SHAPE_NOTES.get(
                                key if key in wl_opts else "none", ""))
                            # ...and WHICH SIDE a fence sits on, where the
                            # band has one. In air the cant band is unsigned
                            # and a 90 deg device points up because a wing's
                            # does; under water it is signed (+ up towards
                            # the free surface, - down), so narrowing the
                            # magnitude has to name a side and this family
                            # names DOWN (api.WATER_TIP_DIRECTION). Said
                            # here because it is the one thing the shape's
                            # own note cannot know.
                            if water and key == "vertical":
                                widgets.hint(
                                    "Under water the fence hangs DOWN — the "
                                    "side that carries cavitation margin, "
                                    "since a device canted up sits in less "
                                    "static head and cavitates first. The "
                                    "canted shape keeps the whole signed "
                                    "band and lets the search choose the "
                                    "side.")
                            _missing(wl_opts, v1.WINGLET_SHAPE_LABELS,
                                     v1.option_why(ch, v1.WINGLET_SHAPE_WHY))
                            _tip_chord_control(
                                key if key in wl_opts else "none")

                        # NO section menu here, and none anywhere else in
                        # this stage. The section was decided in stage 2;
                        # asking again in the configuration list — in the
                        # vocabulary of solver families, next to the tip
                        # device — made the pipeline read as though the
                        # earlier answer had not counted. The row above is
                        # the whole of what stage 3 says about it: the value
                        # it holds, and the way back.

                        # ONE control for the wing's SIZE: who decides it.
                        # Every entry the registry has a solver for is here,
                        # including the wing-loading one — a mode the user can
                        # only reach by discovering a switch in another card
                        # is a mode nobody chooses. What the entry selects is
                        # then answered under it: a typed span for the fixed
                        # planform, a b_m BAND in the design box for the two
                        # that search one (_size_controls, _box_items).
                        #
                        # The aircraft FAMILY is dropped (PLANFORM_V3_WHY says
                        # so, in the same note every other missing entry uses):
                        # it flies its own section and carries no tip-device or
                        # tail solver, so in a pipeline whose stage 2 designed
                        # a section it can only take things away.
                        pf_opts = {k: v for k, v
                                   in v1.planform_options(ch).items()
                                   if k != "aircraft"}
                        if track:
                            # THE MENU WAS SAYING THE OPPOSITE OF THE TRUTH.
                            # `planform_options` offers a car exactly one
                            # entry, "fixed", whose label reads "fixed span +
                            # area (you choose the size)" — and a rear wing
                            # chooses neither: b_m and S_m2 are both rows of
                            # the design box. The disabled select therefore
                            # displayed a sizing mode that does not exist
                            # here, directly above a hint (`_size_controls`)
                            # correctly saying both dimensions are searched.
                            # Stated once now, as what it is.
                            with _row("planform"):
                                ui.label("span AND area are searched") \
                                    .classes("readout")
                            widgets.hint(
                                "There is no sizing mode to pick: a rear "
                                "wing's two dimensions are design variables, "
                                "because what bounds them is a regulation or "
                                "the bodywork rather than a wing loading. "
                                "Say how wide and how big it may be on the "
                                "design box's b_m and S_m2 rows.")
                            _size_controls()
                        elif water and session.wet_size_available(S):
                            # WATER ASKS THE SAME QUESTION WITH ITS OWN
                            # ANSWERS. `planform_options` hands a water
                            # family exactly one entry — "fixed span + area
                            # (you choose the size)" — so the select was a
                            # label with nothing to select, and the entry it
                            # was missing is the one a foil designer starts
                            # from. A foil's span is not set by a hangar or
                            # a class rule; it is the design.
                            with _row("planform"):
                                # KEPT, because the heel field below changes
                                # which entries this menu has (a strut prices
                                # no span at zero heel) and a NUMBER FIELD
                                # MUST NOT REBUILD THE CARD IT SITS IN. So
                                # _set_wet_heel re-options this select in
                                # place instead of re-rendering; the handle
                                # is re-bound on every render, which is what
                                # keeps it from driving a destroyed element.
                                _wet_size_el["sel"] = ui.select(
                                    session.wet_size_modes(S),
                                    value=session.wet_size_mode(S),
                                    on_change=lambda e: _set_wet_size(
                                        e.value)) \
                                    .props("dense outlined options-dense") \
                                    .classes("w-full")
                            widgets.hint(session.WET_SIZE_HINTS[
                                session.wet_size_mode(S)])
                            # ...and what the entry the user TOOK is worth.
                            # No entry is ever missing now, so this is not
                            # "why the menu shrank" any more: it is the
                            # caution that a span nothing prices comes back
                            # as the top of its band. THE HANDLE IS KEPT for
                            # the same reason the select's is — the heel
                            # field below changes this sentence, and a typed
                            # number must not rebuild the card it sits in,
                            # so _set_wet_heel rewrites it in place.
                            why = session.wet_size_why(S) or ""
                            # split=False: this label's TEXT is rewritten in
                            # place by _set_wet_heel, and a ? built from the
                            # words it used to say is worse than a plain line
                            _wet_size_el["why"] = widgets.hint(why, split=False)
                            _wet_size_el["why"].set_visibility(bool(why))
                            _wet_heel_control()
                            _size_controls()
                        elif water:
                            # ...and a water family with no size row at all.
                            with _row("planform"):
                                ui.label("fixed span + area") \
                                    .classes("readout")
                            why = session.wet_size_why(S)
                            widgets.hint(why or
                                         session.WET_SIZE_UNAVAILABLE_WHY)
                            _wet_heel_control()
                            _size_controls()
                        else:
                            _planform_control(pf_opts)

                        with _row("chord law"):
                            chd = ui.toggle(
                                {"fixed": "straight taper",
                                 "free": "polynomial chord law"},
                                value=ch.get("chord", "fixed"),
                                on_change=lambda e: set_choice("chord",
                                                               e.value)) \
                                .props("dense no-caps unelevated "
                                       "toggle-color=primary")
                            if not v1.chord_available(ch):
                                chd.disable()
                                chd.tooltip("this family has no chord-law "
                                            "twin")

                        # NO flight-state control. Speed and altitude are the
                        # MISSION's, typed in stage 1 — the design point, the
                        # section's Reynolds number and the trim target are
                        # all quoted at them — so a toggle here that handed
                        # those same two numbers to the optimiser made the
                        # answer given three stages earlier conditional on a
                        # switch nobody would look back at. The modifier is
                        # still in the registry and still offered by V1/V2
                        # (session.V3_PINNED_CHOICES).

                        # WHETHER there is a second surface is stage 1's
                        # question — it decides how many surfaces the design
                        # has, which sections stage 2 must choose, and what
                        # this stage is configuring. Its GEOMETRY is this
                        # stage's, and that is what the card below holds.
                        # ...and NOT on the track: a rear wing is the last
                        # surface on the car, stage 1 says so, and a row here
                        # reading "tail: none" beside a "change in stage 1"
                        # button offered a trip to a switch that is disabled
                        # when you get there. The car's second element is a
                        # SLOTTED FLAP and it is switched on below.
                        if not track:
                            with _row("elevator" if water else "tail"):
                                ui.label(
                                    session.second_surface_name(S)
                                    or "none").classes("readout")
                                ui.button("change in stage 1", icon="edit",
                                          on_click=lambda: ctx.select(
                                              "mission", "operating")) \
                                    .props("flat dense size=sm no-caps")
                            if ch["tail"]:
                                _tail_controls(water)
                            elif not water:
                                # A PAIR CARRIES A FIN AND HAD NO CARD FOR
                                # IT. ``_fin_controls`` was nested inside
                                # ``_tail_controls``, which is drawn only
                                # under ``ch["tail"]`` — and a tandem's
                                # second surface is its REAR WING, so that
                                # switch is off and disabled. Stage 1 said
                                # "add a vertical stabiliser ... stage 3
                                # sizes it", stage 3 drew nothing, and the
                                # one field that did exist (the boom) sat on
                                # the pair's layout block describing itself
                                # as a rear-wing length. The card is
                                # self-gating — it returns at once for a
                                # family that declares none of the fin's
                                # flags, which is every plain wing — so it is
                                # called for anything in air without a tail
                                # card rather than for the pair by name.
                                #
                                # WATER IS EXCLUDED for the reason
                                # ``_tail_controls`` already excludes it: a
                                # craft's vertical is the MAST, which is not
                                # a Raymer volume-coefficient tail and is not
                                # sized, placed or described by this card.
                                _fin_controls()
                        if track:
                            # limits=False: the drag ceiling and the
                            # downforce floor are drawn under the DESIGN BOX
                            # in this shell, beside the chord and tail limits
                            # that are already there. One home per shell
                            # size_note=False: this shell states the size
                            # under its own planform row (_size_controls),
                            # and saying it twice on one card was the
                            # redundancy this stage was cleaned up for
                            # mount=False: drawn at the TOP of this list,
                            # which is where the question belongs
                            # plate_chord=False: the plate's metre band is
                            # drawn under the DESIGN BOX, beside the ratio
                            # row it clips
                            # limits_below=True: the two LIMITS are drawn
                            # immediately under this block by
                            # _car_limit_controls, so the per-objective notes
                            # point DOWN rather than across to another tab
                            v1._car_controls(ch, set_choice, limits=False,
                                             limits_below=True,
                                             size_note=False, mount=False,
                                             plate_chord=False)
                            # THE OTHER HALF OF THE LAW, beside the law.
                            # These two used to be drawn under the DESIGN BOX
                            # with the chord and tail limits — a defensible
                            # place for a limit and the wrong place for THIS
                            # one, because it is not an independent bound: a
                            # car objective is one half of a trade and the
                            # limit is the other half. Choosing "efficiency"
                            # and looking for where to say how much downforce
                            # is worth having found a SENTENCE pointing at
                            # another tab, which is not a way to state a
                            # number. Moved, never duplicated: the box's
                            # Limits block no longer draws them, and the two
                            # keys left BOX_VALUE_KEYS with them.
                            _car_limit_controls()

                    # THE BODY, AND THE WING'S OWN GEOMETRY. Both are
                    # configuration questions and both are asked here now,
                    # under the card that says what this aeroplane is,
                    # because that is where a reader looks for "does it have
                    # a fuselage" and "does the wing have dihedral".
                    #
                    # The body used to be asked three cards down, inside the
                    # TRIM LAYOUT, and only when the arm happened to be
                    # searched — so an aeroplane's most obvious component
                    # appeared and vanished with a toggle about the tail. The
                    # cant used to be in the right-hand column, beside the
                    # derived-solver read-out; that is where its
                    # ANSWERABILITY is decided, not where the answer belongs.
                    if not water:
                        _fuselage_controls()
                    _wing_cant_panel()
                    # ...AND HOW STABLE IT MUST BE, where there is no
                    # second-surface card to ask it in.
                    #
                    # ``_handling_level_control`` is drawn inside the TRIM
                    # LAYOUT card, beside the static margin, which is the
                    # right place for it and is argued for in its own
                    # docstring — on a family that HAS one. That card is
                    # ``_tail_controls``, drawn only under ``ch["tail"]``, so
                    # a TANDEM never reached it: the pair now flies a fin,
                    # declares ``api.HANDLING_LEVEL_KEY`` like every other
                    # family with a lateral deck, and had no control anywhere
                    # able to write it. A flag with no control is the same
                    # unreachable-freedom class as the greyed tip device.
                    #
                    # Not moved for everyone, and not drawn twice: the call
                    # is skipped exactly where the trim card will make it,
                    # and the control self-guards on the flag, so a family
                    # with no lateral deck still draws nothing here.
                    if not ch["tail"]:
                        _handling_level_control()

                boxes["right"] = ui.column().classes("gap-3").style(
                    "flex:2 1 0;min-width:0")
                _render_right()

    def _render_right():
        """The derived-solver panel — everything on the right of the wing
        type view. Redrawable on its own, so a size edit can update the
        planform it reports without rebuilding the field it was typed
        into."""
        box = boxes.get("right")
        if box is None:
            return
        box.clear()
        with box:
            _derived_panel()
            # the cant card is NOT here any more: what this column says is
            # which family the configuration derived, and the cant is a
            # question about the aeroplane rather than about the solver. It
            # is asked in the configuration column, under the body.

    # -------------------------------------------- the tip device's own chord
    def _car_tip_control():
        """A REAR WING'S TIP DEVICE IS ITS ENDPLATE — and it is a menu.

        Drawn only under the PYLON mount, which is the only layout where the
        plate is a device at all: with the struts carrying the car the plate
        goes back to being a fence of free height, and a fence has a shape.
        Under the endplate mount `v1.car_tip_shapes` returns nothing and this
        is not called — there the plate is the load path, and its section and
        root blend are asked under the Mount instead.

        It used to be a sentence ("endplates — always fitted"), which was
        true and useless: the plate has a SHAPE, and three of the four shapes
        the aircraft menu offers are things a rear wing's plates really do.
        So this offers the same four keys (``v1.CAR_TIP_SHAPE_LABELS``), and
        the entries a given configuration cannot fly are missing with the
        reason printed rather than silently absent.

        All four are honest here, and only because the family charges what
        they cost: a cant or a blend projects the plate OUTBOARD, and
        ``carwing.py`` now takes that projection out of the wing's own span
        before anything is flown (the span row is the OVERALL width). While
        it did not, a blend bought +12.3 % CZ by flying 1.92 m of hardware
        inside a stated 1.6 m band.

        Deliberately NOT a menu that switches families for you: picking an
        entry never moves the Mount above it, because answering one more
        question must never widen a search.
        """
        ch = W["choices"]
        opts = v1.car_tip_shapes(ch, W["problem"])
        key = v1.car_tip_shape_key(ch, W["problem"])
        with _row("tip device"):
            ui.select(opts, value=key,
                      on_change=lambda e: _set_car_tip(e.value)) \
                .props("outlined dense").classes("grow min-w-0")
        widgets.hint(v1.CAR_TIP_SHAPE_NOTES.get(key, ""))
        _missing(opts, v1.CAR_TIP_SHAPE_LABELS, v1.CAR_TIP_SHAPE_WHY)
        if key == "canted":
            # the band is the LATTICE's own validity, read from the one
            # helper the Mount card's copy of this field reads. It was 1-90
            # here and 5-90 there: the same key, the same solver, two
            # accepted ranges, and one of them was wrong.
            _lo, _hi = v1._car_plate_cant_limits()
            v1._num_row("leaning at", "car_endplate_cant_deg", ch, set_choice,
                        suffix="° from the wing plane", step=5.0,
                        placeholder=f"{v1.car_tip_cant_deg({}):g} (vertical)",
                        lo=_lo, hi=_hi)
            widgets.hint(
                "90° is vertical and projects nothing outboard. Lower leans "
                "the plate out, and the height row is an ARC LENGTH along "
                "it, so the plate keeps its size and trades reach for "
                "projection — which this family pays for out of the wing's "
                "own span, because b_m here is the OVERALL width. BLANK IS "
                "90°, i.e. still the vertical plate: this entry is where you "
                "STATE a lean, and no number in this repo says what a rear "
                "wing's plates lean by, so it does not pick one for you.")
        # NO blend FIELD here. The aircraft menu's rule, and the reason it
        # has one: a blend is a VALUE the shape implies, not a second
        # question — `nice_app.set_winglet_shape` installs
        # BLEND_FRAC_DEFAULT when the reader picks "blended" and clears it
        # otherwise, and `_set_car_tip` now does exactly the same. (Under the
        # ENDPLATE mount the same number IS asked as a field, because there
        # the plate is a designed structure and its root blend is one of the
        # two things it is built from.)
        _car_tip_chord_control(key)

    def _set_car_tip(shape: str):
        """The plate's shape, and the one design-box row it decides.

        "none" is not a flag — there is no such thing as a plate of no
        height, only a height of zero — so it is a PIN on ``endplate_h_m``
        (``api.RunConfig.pinned`` through ``config.fixed_rows``), which takes
        the row out of the design vector rather than collapsing its band to
        zero width. Leaving "none" releases it back to the box it came from.
        """
        fixed = S["wing"].setdefault("fixed", {})
        if shape == "none":
            fixed["endplate_h_m"] = 0.0
            ctx.log("tip device: none — endplate_h_m pinned at 0 and out of "
                    "the design vector", "info")
        elif fixed.pop("endplate_h_m", None) is not None:
            ctx.log("tip device: the endplate height is a design variable "
                    "again", "info")
        # ...and the BLEND the shape implies, written the way the aircraft
        # menu writes it (nice_app.set_winglet_shape): a default turn when
        # the reader picks "blended", and nothing at all otherwise, because a
        # vertical or canted plate meets the wing in a crease.
        W["choices"]["car_endplate_blend_frac"] = (
            v1.BLEND_FRAC_DEFAULT if shape == "blended" else None)
        set_choice("car_tip_shape", shape)

    def _car_tip_chord_control(shape: str):
        """Does the PLATE's chord continue the wing's, or hold the tip's?

        The car's half of :func:`_tip_chord_control`, and gated the same way
        — on the registry's own declaration (``endplate_chord_follows``),
        which the designed-endplate family deliberately does not carry: there
        the plate's chord is a design row of its own, so continuing the
        wing's law onto it would be a second answer to a question the search
        is already answering.
        """
        if shape == "none" or "endplate_chord_follows" not in sp().flags:
            return
        on = bool(W["choices"].get("car_endplate_chord_follows"))
        with _row("its chord"):
            ui.switch("follows the wing's chord distribution", value=on,
                      on_change=lambda e: set_choice(
                          "car_endplate_chord_follows",
                          True if e.value else None)) \
                .props("dense")
        widgets.hint(
            "On: the plate is sampled from the wing's own chord law "
            "continued past the tip, so a tapered wing gets a tapered plate "
            "and there is no step at the junction. Off — the published plate "
            "— it holds the wing's TIP chord the whole way, which is a "
            "rectangle bolted to a tapered wing. Measured at the box centre: "
            "+0.05 on CZ/CD, by taking drag out rather than putting "
            "downforce in.")

    def _tip_chord_control(shape: str):
        """Does the tip device's CHORD continue the wing's, or hold the tip?

        The second question about the device, and the only other one that
        changes the shape it is built in: its height and cant are design
        variables, its blend is folded into the shape menu above, and its
        chord was — until this switch — always the wing's tip chord, i.e. a
        rectangle bolted to a tapered wing. On it continues the chord
        distribution past the tip instead, so the planform is one curve from
        root to device tip.

        Offered only where the family flies a device AND declares the flag
        (api.WINGLET_CHORD_KEY — the car endplate is excluded there: its
        chord is already a design variable of its own).
        """
        if shape == "none" or api.WINGLET_CHORD_KEY not in sp().flags:
            return
        on = bool(W["flags"].get(api.WINGLET_CHORD_KEY))
        with _row("its chord"):
            ui.switch("follows the wing's chord distribution", value=on,
                      on_change=lambda e: _set_tip_chord(bool(e.value))) \
                .props("dense")
        widgets.hint(
            "On: the device's chord is the wing's own chord law, continued "
            "past the tip along the device (a tapered wing gets a tapered "
            "device, and there is no step at the junction). Off — the "
            "published behaviour — the device holds the wing's TIP chord "
            "the whole way, which is a rectangle."
            if not on else
            "The chord law does not stop at the tip: the device is sampled "
            "from the same distribution, so wing and device are one planform. "
            "A law that runs the device to a knife edge is refused (it scores "
            "the penalty) rather than quietly squared off.")

    #: NO "how tall" ROW HERE, deliberately. The device's height was asked
    #: twice: once as a switch under the tip-device menu ("state it myself",
    #: with a number beside it) and once as ``winglet_h_frac`` in the design
    #: box, where every other searched-or-stated row is asked. Both wrote the
    #: same pin (``config.fixed_rows``), so nothing could disagree — but a
    #: question with two homes is still two questions, and this shell asks
    #: each one in ONE place. The design box is that place: it is where
    #: searched/fixed is chosen for the taper, the twist and the sweep, and a
    #: height is not a different kind of number.
    #:
    #: What the deleted control had to say survives where it is actionable:
    #: nothing in this model charges a tip device for its own weight or its
    #: root bending (``sizing.sized_state`` takes no winglet argument), so the
    #: height row rides the top of whatever band it is given. That is the
    #: design box's note about the row, not a second control.

    def _set_flag_bool(key: str, on: bool):
        """A BOOLEAN flag: present means yes, absent means the default.

        Not ``_set_flag``, which casts to float — an on/off written as 1.0
        is a different value from the True the builder checks for, and a
        flag written as 0.0 is PRESENT and therefore "answered no" rather
        than unanswered. Redraws the type view for its own read-out and
        touches nothing else: this is a value, so no problem is re-derived.
        """
        if on:
            W["flags"][key] = True
        else:
            W["flags"].pop(key, None)
        _render_type()

    def _set_tip_chord(on: bool):
        if on:
            W["flags"][api.WINGLET_CHORD_KEY] = True
        else:
            W["flags"].pop(api.WINGLET_CHORD_KEY, None)
        # a VALUE, so no problem is re-derived and no menu below it changes;
        # the type view is redrawn for its own note, and the design box is
        # not touched at all
        _render_type()
        ctx.log("tip device chord: "
                + ("follows the wing's chord law" if on
                   else "the wing's tip chord (rectangular)"), "info")
        ctx.refresh()

    def _back_to_section():
        """Stage 2, on the view that can actually change the answer: the
        ranking where one has been screened, the screening form where none
        has (an empty ranking table is not somewhere to send anyone)."""
        ctx.select("airfoil",
                   "ranking" if S["airfoil"]["screen"]["report"] else "screen")

    def _set_wet_heel(value):
        """HOW FAR OVER THE CRAFT IS SAILED [deg].

        Stored and logged and NOTHING ELSE IS REDRAWN, deliberately. The
        heel creates no row of the design box — it is a flight condition,
        like the water it is flown in — so the card that holds this field
        has no reason to be rebuilt, and rebuilding it under the cursor is
        the defect this shell has a standing rule against (a typed number
        must not rebuild its own field).
        """
        if not session.set_wet_heel_deg(S, 0.0 if value in (None, "")
                                        else value):
            ctx.log("heel: refused — see hydrofoil.heel_angle", "warn")
            return
        phi = session.wet_heel_deg(S)
        ctx.log("heel: " + ("flat — every station at one depth, and the "
                            "span has nothing to trade against"
                            if phi == 0.0 else
                            f"{phi:g} deg — the rising tip is "
                            f"(b/2)·sin({phi:g}°) nearer the surface, so "
                            f"the span is now priced"), "info")
        # ...and the FIRST thing it changes is the note, which is now the
        # only thing it changes: the menu no longer gains or loses entries
        # with the heel (every water family offers all four), so an early
        # return on "the modes are unchanged" would return every time — and
        # the sentence the user is reading says TYPE A HEEL ANGLE. A remedy
        # that stays on screen after it has been taken is a remedy nobody
        # believes twice. Rewritten in place, like the select below and for
        # the same reason: re-rendering the card would destroy the field
        # this value was just typed into.
        why_el = _wet_size_el.get("why")
        if why_el is not None:
            text = session.wet_size_why(S) or ""
            why_el.set_text(text)
            why_el.set_visibility(bool(text))
        # ...and that is what USED TO open the span rows beside a strut
        # (session.wet_size_modes). Kept because the menu is still built
        # from the session, and a family that one day removes a mode would
        # need it; today the options are heel-independent and this is a
        # no-op.
        sel = _wet_size_el.get("sel")
        if sel is None:
            return
        modes = session.wet_size_modes(S)
        if list(modes) == list(sel.options or {}):
            session.apply_choices(S)
            _render_box()
            return
        was = session.wet_size_mode(S)
        # QUIETLY. ``set_options`` assigns ``value`` FIRST (nicegui's
        # ChoiceElement), so where the clamp has moved the mode the write
        # fires the select's own on_change -> _set_wet_size, and that does
        # two things this must not do: it calls _render_type, which destroys
        # the very field the heel was typed into (the rest of "12" goes with
        # it), and it STORES the clamped mode, turning a read-time clamp into
        # a destructive edit -- a stated "free SPAN" would not come back when
        # the heel did. The guard makes the write a display change only, and
        # the stored answer stays the user's (session.wet_size_mode).
        _wet_size_el["quiet"] = True
        try:
            sel.set_options(modes, value=was)
        finally:
            _wet_size_el["quiet"] = False
        ctx.log("size: " + ("the span may now be searched — a heeled craft "
                            "prices it" if "span" in modes else
                            f"flat again, so the span rows are gone and the "
                            f"size is '{was}'"), "info")
        # the mode may have been CLAMPED back by the same edit (session.
        # wet_size_mode), which takes b_m out of the design box, so the box
        # is redrawn — a different card, so nothing under the cursor moves.
        session.apply_choices(S)
        _render_box()

    def _wet_heel_control():
        """The heel field, asked beside the size because that is what it
        prices. One question, one place."""
        widgets.number_field(
            "heel", widgets.shown(session.wet_heel_deg(S)),
            lambda e: _set_wet_heel(e.value), unit="deg", step=1.0,
            tip="how far over the craft is sailed. 0 is flat and every "
                "published water run; anything else gives each span "
                "station its own static head, and is the only term in "
                "these solvers that reads how wide the foil is.")
        widgets.hint(session.WET_HEEL_HINT)
        _wet_reynolds_control()

    def _set_wet_flown_re(value):
        """WHICH TABLE the sections are read from. Stored, logged, nothing
        redrawn: it creates no design-box row (see :func:`_set_wet_heel`)."""
        session.set_wet_flown_re(S, bool(value))
        ctx.log("section tables: "
                + ("read at each surface's own Reynolds number "
                   "(rho·V·mac/mu) — the speed and the chord now choose the "
                   "table" if session.wet_flown_re(S) else
                   "the family's Re = 1e6 tables, whatever is flown"), "info")

    def _wet_reynolds_control():
        """...and which Reynolds number the section is read at.

        Beside the heel because they are the same kind of answer — what the
        water is doing to this craft — and not beside the section, which is
        about SHAPE. Absent on a family flying a chosen section, where the
        question has no answer (one table, one Reynolds number).
        """
        if not session.wet_re_available(S):
            return
        with _row("section Re"):
            ui.switch("read at the flown Reynolds number",
                      value=session.wet_flown_re(S),
                      on_change=lambda e: _set_wet_flown_re(e.value)) \
                .props("dense")
        widgets.hint(session.WET_RE_HINT)

    # ------------------------------------------------------- view: the size
    def _size_controls():
        """The size question, and then what the wing is BUILT of.

        Split from :func:`_size_question` because that one answers a
        different question per planform mode and RETURNS EARLY from three of
        them — including the free planform, which is the mode a material
        matters most on. A card appended to the end of its body would have
        been on screen for one mode out of four.
        """
        _size_question()
        _material_controls()

    def _size_question():
        """The SIZE this stage flies, asked the way the planform menu above
        selected.

        With the planform FIXED this is a QUESTION, and it is one number: the
        SPAN, in metres. The area is not asked here — the mission states a
        load and a wing loading, and S = W/(W/S) — and the ASPECT RATIO is not
        asked anywhere, because it is b²/S: a consequence of the span, not an
        input. Stage 2 guesses one to have a chord to design a section for,
        and this card reports how far the wing actually flown has moved from
        that guess (``_ar_disagreement``).

        With the span SEARCHED the same card is a readout of what the design
        box's ``b_m`` row implies — the question moved to the band, which is
        where a searched variable is asked.
        """
        name = W["problem"]
        if not api.resizable(name) and not session.span_is_searched(S):
            labels = api.PROBLEM_SPECS[name].param_labels
            if "b_m" in labels:
                # the free planform (and the car families): the size IS the
                # search, so the question is a band and it is asked where
                # every other band is
                spans = "the two spans (one per wing)" \
                    if "b_rear_m" in labels else "the span"
                # WHAT THE SIZE IS SPENT ON — and it is not one sentence for
                # both families. The aircraft's size closes against a Raymer
                # wing weight, so growing it costs payload and root stress.
                # A rear wing is bolted to a car: nothing weighs it in the
                # score, which maximises downforce (or CZ/CD, or the force),
                # and what growing it costs is DRAG against the budget and
                # DEFLECTION against the limit. Read off the declared design
                # vector — a ride height is a car — rather than off a family
                # list, so the twins and the chord variants inherit it.
                car = "ride_height_m" in labels
                widgets.hint(
                    "Size: "
                    + (f"{spans} and the area are "
                       + ("all" if "b_rear_m" in labels else "both")
                       + " design variables"
                       if "S_m2" in labels else
                       f"{spans} " + ("are" if "b_rear_m" in labels else "is")
                       + " a design variable, against this family's "
                       "own reference area")
                    + " — their bands are rows of the design box. "
                    + ("Nothing weighs the wing here: the score is the "
                       "downforce it makes (or CZ/CD), and what its size "
                       "costs is the DRAG budget and the DEFLECTION limit, "
                       "which are the two constraints."
                       if car else
                       "The wing weighs what its size implies, so the score "
                       "is payload L/D and a root-bending stress margin is a "
                       "constraint."))
                return
            size = api.planform_size(name)
            widgets.hint(
                "Size: this family carries calibrated geometry"
                + (f" — b = {size[0]:.3f} m, S = {size[1]:.3f} m²." if size
                   else ".")
                + " The mission's area sets the section design point only.")
            return
        # the INPUT is built once and never redrawn by its own handler: a
        # rebuilt ui.number loses the focus and swallows the rest of the
        # number being typed (VALUE_KEYS, the mission card's W/S field). Only
        # the readouts under it follow the value, in their own container.
        #
        # ONE FIELD PER SURFACE THAT HAS A SPAN (session.span_rows): a tandem
        # pair's two wings are two wings, and asking for the pair's width once
        # made the rear wing's span a number nobody could state (or, before
        # that, a number stated three cards away, beside the stagger). The
        # question is the same question, so it is asked in the same place,
        # twice.
        if api.resizable(name):
            rows = session.span_rows(S)
            _size_mode_control(rows)
            if session.size_mode(S) == "chords":
                _chord_size_fields()
                # a second surface still has its own span: the two END CHORDS
                # are the wing's planform, and the pair's other wing is not
                # the same wing
                for row in rows[1:]:
                    _span_field(row, named=True)
            else:
                for row in rows:
                    _span_field(row, named=len(rows) > 1)
        boxes["size"] = ui.column().classes("w-full gap-0")
        _render_size_note()

    # -------------------------------------------------- what it is BUILT of
    def _material_key() -> str:
        """Which material the flags currently say, as a menu value.

        The three custom NUMBERS win over the preset key, exactly as
        ``api._material_kwargs`` resolves them: a typed density is a more
        specific statement than a picked preset, and a menu that disagreed
        with the builder about which one applies would be showing a material
        the run is not using.
        """
        fl = W.get("flags") or {}
        # EITHER number, not both: a half-cleared pair is still an unfinished
        # custom material, and a menu that flipped back to the reference
        # would leave the other number in the flags for the builder to refuse
        # at launch — with nothing on screen having said so.
        if (fl.get(api.MATERIAL_RHO_KEY) is not None
                or fl.get(api.MATERIAL_SIGMA_KEY) is not None):
            return "custom"
        return str(fl.get(api.MATERIAL_KEY) or materials.REFERENCE_KEY)

    def _material_now():
        """The Material the run would be built of, from the flags on screen."""
        fl = W.get("flags") or {}
        rho, sig = fl.get(api.MATERIAL_RHO_KEY), fl.get(api.MATERIAL_SIGMA_KEY)
        if rho is not None or sig is not None:
            # the card answers the same way the BUILDER does (api's
            # ``_material_kwargs`` refuses half a material), so what is on
            # screen can never disagree with what a launch would do
            if rho is None or sig is None:
                return None
            try:
                return materials.custom(float(rho), float(sig) * 1e6,
                                        fl.get(api.MATERIAL_SKIN_RHO_KEY))
            except ValueError:
                return None          # mid-edit: a zero typed into a field
        return materials.resolve(fl.get(api.MATERIAL_KEY))

    def _material_options() -> dict:
        """The table as a select's options, grouped in its own order."""
        short = {"model / RC / UAV": "model", "homebuilt / recreational":
                 "homebuilt", "certified GA / transport": "GA"}
        out = {}
        for group, keys in materials.MATERIAL_GROUPS.items():
            for key in keys:
                out[key] = (f"{short.get(group, group)} · "
                            f"{materials.MATERIALS[key].label}")
        out["custom"] = "custom — type a density and an allowable"
        return out

    def _set_material(key: str):
        """Answer it. The REFERENCE writes no flag at all.

        Aluminium 2024-T3 is what Raymer's correlation was regressed over, so
        it is the default in the physics as well as in this menu: writing a
        flag for it would put a key in every stored config that changes
        nothing, and the one thing this shell must keep true is that a run
        with no material flag is the published calibration bit-for-bit.
        """
        fl = W["flags"]
        for k in api.MATERIAL_FLAG_KEYS:
            fl.pop(k, None)
        if key == "custom":
            # opens on the entry a custom material is most often a variant of
            fl[api.MATERIAL_RHO_KEY] = 1600.0
            fl[api.MATERIAL_SIGMA_KEY] = 500.0
        elif key != materials.REFERENCE_KEY:
            fl[api.MATERIAL_KEY] = key
        m = _material_now()
        if m is not None:
            ctx.log(f"material: {m.label} — k_w {m.k_w:.3f}, allowable "
                    f"{m.sigma_allow_Pa / 1e6:.0f} MPa", "info")
        _render_material()
        ctx.refresh()

    def _set_material_number(key: str, value):
        """A typed custom property. The FIELD is never rebuilt by its own
        handler (VALUE_KEYS' rule) — only the readout under it follows."""
        fl = W["flags"]
        if value in (None, ""):
            fl.pop(key, None)
        else:
            fl[key] = float(value)
        _render_material_readout()
        ctx.refresh()

    def _render_material_readout():
        box = boxes.get("material_readout")
        if box is None:
            return
        box.clear()
        m = _material_now()
        with box:
            if m is None:
                widgets.hint(
                    "A custom material needs BOTH a density and an allowable, "
                    "and both must be > 0 — one without the other is half a "
                    "material, and the run refuses it.", "bad")
                return
            with ui.row().classes("items-center gap-3 w-full"):
                widgets.readout(
                    "weight factor", f"{m.k_w:.3f}", "×",
                    tip="what the statistical wing weight is multiplied by. "
                        "This is the number the span answer rides — the "
                        "allowable beside it moves the aeroplane far less "
                        "(measured: d ln f / d ln k_w ≈ -1 against ≈ +0.01)")
                widgets.readout(
                    "allowable", f"{m.sigma_allow_Pa / 1e6:.0f}", "MPa",
                    tip="the design stress the root spar is sized to — the "
                        "number g = ln(σ_allow/σ_root) is measured against")
                if m.rho_skin != m.rho_cap_kgm3:
                    widgets.readout("core ρ", f"{m.rho_skin:.0f}", "kg/m³",
                                    tip="skin/core density: the caps carry "
                                        "the bending, this carries the shape")
            if m.note:
                widgets.hint(m.note)

    def _material_controls():
        """WHAT THE WING IS BUILT OF — asked only where a wing is WEIGHED.

        The gate is the family's own flag declaration and not a mode test: a
        fixed-size family computes no weight and no stress at all (its
        breakdown carries neither a W_wing_N nor a σ_root), so a material
        there would be a control that changes nothing. ``MODIFIER_FLAGS``
        declares these four keys on the three sized modes and nowhere else,
        which is the same authority ``api.check_flags`` refuses them by.

        Why it is on the SIZE card rather than a card of its own: "how big"
        and "what of" are one question here. Freeing the size is what makes
        the wing weigh something, and the material is what it weighs — until
        the size is a variable there is nothing for a material to change.
        """
        if api.MATERIAL_KEY not in api.PROBLEM_SPECS[W["problem"]].flags:
            return
        boxes["material"] = ui.column().classes("w-full gap-2")
        _render_material()

    def _render_material():
        box = boxes.get("material")
        if box is None:
            return
        box.clear()
        key = _material_key()
        with box:
            widgets.select_field(
                "built of", _material_options(), key,
                lambda e: _set_material(str(e.value)),
                tip="the material sets BOTH the allowable the spar is sized "
                    "to and the factor the wing's statistical weight is "
                    "scaled by")
            if key == "custom":
                fl = W["flags"]
                widgets.number_field(
                    "density", fl.get(api.MATERIAL_RHO_KEY),
                    lambda e: _set_material_number(api.MATERIAL_RHO_KEY,
                                                   e.value),
                    unit="kg/m³", step=50.0,
                    tip="of the SPAR CAPS — the strength-sized part")
                widgets.number_field(
                    "allowable", fl.get(api.MATERIAL_SIGMA_KEY),
                    lambda e: _set_material_number(api.MATERIAL_SIGMA_KEY,
                                                   e.value),
                    unit="MPa", step=10.0,
                    tip="DESIGN allowable, safety factor already applied — "
                        "not a handbook ultimate")
                widgets.number_field(
                    "core density", fl.get(api.MATERIAL_SKIN_RHO_KEY),
                    lambda e: _set_material_number(api.MATERIAL_SKIN_RHO_KEY,
                                                   e.value),
                    unit="kg/m³", step=25.0,
                    tip="skin/core, if it is not the cap material — leave "
                        "empty for a homogeneous build")
            boxes["material_readout"] = ui.column().classes("w-full gap-1")
            _render_material_readout()
            widgets.hint(
                "The weight factor is an EMPIRICAL fit, not a calculation: "
                "one parameter anchored so aluminium is exactly 1.000 and "
                "carbon prepreg lands on the ~0.87 composite wings actually "
                "weigh. It carries buckling, minimum gauge and joints as a "
                "single number, and Raymer's exponents stay aluminium's — so "
                "read it as a ranking between materials, not as a structural "
                "estimate of one.")

    def _size_mode_control(rows):
        """WHICH TWO NUMBERS state the planform.

        A trapezoid has three (area, span, taper) and this shell states two:
        the area is stage 1's, through the wing loading, and the span is the
        card's — which leaves the taper to the search, and with it the root
        and tip chords. That is the wrong way round for a wing whose CHORDS
        are the stated thing: a spar depth, a hinge line, a mould, a rib kit
        or a class rule states a chord in millimetres and lets the span be
        whatever those imply. So the same card can be asked either way.
        """
        if not session.chord_span_available(S):
            return
        with _row("state the planform by"):
            ui.toggle({"span": "the span",
                       "chords": "the root & tip chord"},
                      value=session.size_mode(S),
                      on_change=lambda e: _set_size_mode(e.value)) \
                .props("dense no-caps unelevated toggle-color=primary")
        widgets.hint(
            "The span, and the taper is searched — the published question."
            if session.size_mode(S) == "span" else
            "The two end chords, in metres. They state the TAPER "
            "(c_tip/c_root), which is then fixed rather than searched, and "
            "the SPAN follows from the mission's area: b = 2S/(c_root + "
            "c_tip).")

    def _chord_size_fields():
        """The two end chords, in metres."""
        ch = session.chosen_chords(S)
        widgets.number_field(
            "root chord", widgets.shown(ch[0] if ch else None),
            lambda e: _set_size_chord("root", e.value), unit="m", step=0.01,
            tip="the chord at the centreline. With the interior-only chord "
                "law it is exactly what flies; under any other law the chord "
                "law reshapes the ends too, and this is the straight-taper "
                "baseline it starts from.")
        widgets.number_field(
            "tip chord", widgets.shown(ch[1] if ch else None),
            lambda e: _set_size_chord("tip", e.value), unit="m", step=0.01,
            tip="the chord at the tip. Above the root chord is an inverse "
                "taper — allowed: the family's taper band is a default, not "
                "a ban, and a fixed row is not searched.")

    def _set_size_mode(mode: str):
        session.set_size_mode(S, str(mode))
        ctx.log("planform stated by "
                + ("the two end chords — the taper is fixed at their ratio "
                   "and the span follows the area"
                   if session.size_mode(S) == "chords" else
                   "the span — the taper is searched"), "info")
        _render_type()
        _render_box()
        _render_solver()
        ctx.refresh()

    def _set_size_chord(which: str, value):
        """One end chord [m]. Clearing either gives the planform back to the
        span: half an answer would leave a taper nobody chose."""
        ch = session.chosen_chords(S)
        cur = None if ch is None else (ch[0] if which == "root" else ch[1])
        if widgets.is_echo(value, cur):
            return
        if not session.set_size_chord(S, which, value):
            ui.notify("a chord is a positive length in metres",
                      type="negative")
            return
        ch = session.chosen_chords(S)
        ctx.log(f"{which} chord: "
                + (f"{(ch[0] if which == 'root' else ch[1]):.4g} m"
                   if ch else "cleared")
                + (f" — taper {session.chord_taper(S):.4g}, span "
                   f"{session.chord_span(S):.4g} m"
                   if session.chord_taper(S) is not None
                   and session.chord_span(S) else ""), "info")
        # NOT _render_type(): it holds the field being typed into
        _render_size_note()
        _render_box()
        _render_solver()
        ctx.refresh()

    def _span_field(row: str, named: bool):
        """One typed SPAN, in metres, for the surface ``row`` belongs to."""
        surface = session.SPAN_ROW_LABELS.get(row, "wing")
        fld = widgets.number_field(
            f"span b — {surface}" if named else "span b",
            widgets.shown(session.chosen_span(S, row)),
            lambda e, r=row: _set_span(e.value, r), unit="m", step=0.1,
            # "the craft", not "the aircraft": this field is the water
            # families' too now, and a hydrofoil's span is bounded by a class
            # rule and a spar exactly as a wing's is by a hangar
            tip=("the width the craft occupies — a hangar, a trailer, "
                 "a class rule, a spar. Clear the field to fly stage 2's "
                 "aspect-ratio estimate instead."
                 if row == "b_m" else
                 "this surface's OWN span. The two wings share a fuselage, "
                 "not a span, and how far out this one reaches decides how "
                 "much of the other's downwash it sits in — the trade the "
                 "pair exists to study. Clear the field to fly it as wide "
                 "as the front wing."))
        # empty means "nobody chose one", and the placeholder is what the
        # run flies then — the estimate (or, for a second surface, the front
        # wing's span), shown without being stored: a stored copy would go
        # stale the moment stage 2 or the other field moved
        fld.props(f'placeholder="{session.nominal_span(S, row):.3f}"')
        return fld

    def _set_span(value, row: str = "b_m"):
        """A chosen span [m]. Clearing it gives the size back to what it was
        derived from — stage 2's aspect-ratio estimate for the wing, the
        front wing's span for a second surface — which is what keeps an
        untouched session bit-for-bit the published planform."""
        if widgets.is_echo(value, session.chosen_span(S, row)):
            return
        if not session.set_span_m(S, value, row):
            ui.notify("the span must be a positive length in metres",
                      type="negative")
            return
        chosen = session.chosen_span(S, row)
        surface = session.SPAN_ROW_LABELS.get(row, "wing")
        ctx.log(f"span ({surface}): "
                + (f"{chosen:.4g} m (chosen)" if chosen is not None
                   else "cleared — "
                   + ("stage 2's aspect-ratio estimate" if row == "b_m"
                      else "as wide as the front wing")), "info")
        # NOT _render_type(): that would rebuild the field this handler is
        # inside. Everything else that quotes the size is redrawn — including
        # the right-hand column, which is the container ``_render_right``
        # exists for and which this handler was the one size edit that forgot:
        # the Derived solver card went on reporting "planform flown: b =
        # 10.000 m" beside a span field reading 11.
        _render_size_note()
        # ...and the SECOND SURFACE's two placement notes, whose bands are
        # calibrated ON THE SPAN: typing 11 m left "the solver places the
        # surface 3 - 8 m aft" and "searches 0.5 - 3 m above the wing plane"
        # on screen when both bands had become other ones. Unreachable while
        # the shell opened without a tail, because nothing drew the card.
        _render_arm_note()
        _render_height_note(W["choices"].get("medium") == "water")
        _render_right()
        _render_box()
        _render_solver()
        ctx.refresh()

    def _span_readout(row: str, named: bool):
        """One surface's SPAN, and the aspect ratio it implies.

        The value it was given where the span is typed, the BAND where it is
        searched — the same two states the card itself is in. The aspect
        ratio is read on that surface's OWN share of the reference area
        (session.wing_area_share), because that is the number the solver
        checks: a pair quoting b²/S_total would report half the aspect ratio
        the run refuses a wing for.
        """
        surface = session.SPAN_ROW_LABELS.get(row, "wing")
        box = session.span_box(S, row)
        band = session.wing_ar_band(S, row)
        ar = session.wing_aspect_ratio(S, row)
        if box is None:
            chosen = session.chosen_span(S, row)
            widgets.readout(
                f"span, {surface}" if named else "span",
                f"{session.nominal_span(S, row):.3f}", "m",
                tip="chosen above" if chosen is not None
                else ("as wide as the front wing" if row != "b_m"
                      else "√(AR · S) at stage 2's estimate"))
        else:
            widgets.readout(f"span, {surface}" if named else "span",
                            f"{box[0]:.3g}–{box[1]:.3g}", "m",
                            tip=f"the design box's {row} row")
        widgets.readout(
            f"aspect ratio, {surface}" if named else "aspect ratio",
            (f"{band[0]:.3g}–{band[1]:.3g}" if band
             else (f"{ar:.4g}" if ar is not None else "—")), "b²/S",
            tip="a consequence of the span, not an input"
            + (" — on this wing's own share of the area" if named else ""))

    def _chord_size_note():
        """What the two stated chords buy — and, honestly, what they do not.

        A chord law reshapes the planform the taper draws, so under the
        published polynomial the flown root and tip chords are NOT the ones
        typed above: the law moves them and the area rescale moves them
        again. The interior-only law is the one that holds them exactly, so
        the card says which one is on and offers the other.
        """
        from aerobo import geometry

        ch = session.chosen_chords(S)
        if session.size_mode(S) != "chords" or ch is None:
            return
        lam, b = session.chord_taper(S), session.chord_span(S)
        if lam is None or not b:
            return
        with ui.row().classes("items-center gap-3 w-full"):
            widgets.readout("root chord", f"{ch[0]:.4g}", "m",
                            tip="stated above")
            widgets.readout("tip chord", f"{ch[1]:.4g}", "m",
                            tip="stated above")
            widgets.readout("taper", f"{lam:.4g}", "c_tip/c_root",
                            tip="fixed, not searched")
        lo, hi = api.PLANFORM_AR_LIMITS
        ar = b * b / float(S["mission"]["s_ref_m2"])
        if not (lo <= ar <= hi):
            widgets.hint(
                f"Those chords imply a span of {b:.4g} m on the mission's "
                f"area — aspect ratio {ar:.4g}, and these solvers are honest "
                f"over {lo:g}–{hi:g}. Outside it a candidate is refused, not "
                f"scored: state wider chords (a shorter span), or move the "
                f"wing loading in stage 1.", "bad")
        law = _chord_law()
        if api.CHORD_LAW_KEY not in sp().flags or law == "ends":
            return
        if W["choices"].get("chord") != "free":
            return
        with ui.row().classes("items-center gap-2"):
            widgets.hint(
                f"The chord law on this design is "
                f"{geometry.CHORD_LAW_NAMES[law]}, which reshapes the ends "
                f"too — so the chords above are the straight-taper BASELINE "
                f"it starts from, not the chords that fly. The interior-only "
                f"law holds both of them exactly.", "warn")
            ui.button("hold the ends", icon="straighten",
                      on_click=lambda: _set_chord_law("ends")) \
                .props("flat dense size=sm no-caps")

    def _render_wet_size_note(size, span, area_box):
        """...and the same note for a craft whose SIZE ROWS ARE THE BOX.

        A water craft has no wing loading to derive an area from — it carries
        a stated load through a size it is given or asked to search — so the
        air note's whole subject ("the area follows the mission's W/S") is a
        sentence about machinery this family does not have.
        """
        free_span = span is not None
        free_area = area_box is not None
        with ui.row().classes("items-center gap-3 w-full"):
            widgets.readout("area", f"{size[1]:.4g}", "m²",
                            tip="the middle of the S_m2 row the run will "
                                "search — the optimiser picks one inside it"
                                if free_area else "stated: this mode does "
                                "not search the area")
            widgets.readout("span", f"{size[0]:.4g}", "m",
                            tip=("the middle of the b_m row the run will "
                                 "search" if free_span else
                                 "stated: this mode does not search the span"))
            ui.button("set the bands", icon="tune",
                      on_click=lambda: ctx.select("wing", "box")) \
                .props("flat dense size=sm no-caps")
        band = session.wing_ar_band(S)
        if free_span and free_area and band:
            note = ("Both size rows are design variables: the area over "
                    f"{area_box[0]:.4g}–{area_box[1]:.4g} m² and the span "
                    f"over {span[0]:.4g}–{span[1]:.4g} m, so the numbers "
                    f"above are the middle of each and not an answer. The "
                    f"corners of that box reach aspect ratio "
                    f"{band[0]:.3g}–{band[1]:.3g}"
                    + (", and the part outside 3–40 is refused per design."
                       if band[0] < 3.0 or band[1] > 40.0 else "."))
        elif free_area:
            note = ("The AREA is the design variable, over "
                    f"{area_box[0]:.4g}–{area_box[1]:.4g} m²; the span is "
                    f"the one you stated. The area is the LOADING here — "
                    f"the craft carries a fixed lift, so CL = L/qS — which "
                    f"is why it has a two-sided optimum where the span "
                    f"does not.")
        elif free_span:
            # THE MODE THAT USED TO FALL THROUGH TO THE AIR BRANCH, where it
            # was told to "choose 'area from the mission's W/S, span
            # optimised' above" — an entry this menu has never had. A water
            # craft has no wing loading to derive an area from, so that
            # whole sentence is about machinery this family does not carry.
            note = ("The SPAN is the design variable, over "
                    f"{span[0]:.4g}–{span[1]:.4g} m; the area is the one "
                    f"you stated. At fixed area a wider foil is a higher "
                    f"aspect ratio and nothing else, so watch the ceiling: "
                    + (f"this band reaches aspect ratio "
                       f"{band[0]:.3g}–{band[1]:.3g}, and the part outside "
                       f"3–40 is refused per design."
                       if band else "the corners outside 3–40 are refused "
                       "per design."))
        else:
            note = ("Neither size row is searched: the design vector "
                    "RESHAPES this planform (taper, twist, chord law) and "
                    "never resizes it. The span and area above are the ones "
                    "this craft is given — choose a free SPAN or a free "
                    "AREA in the planform menu to search either instead, "
                    "inside a band you state.")
        widgets.hint(note)

    def _render_size_note():
        """What the run will fly, and where each number came from."""
        box = boxes.get("size")
        if box is None:
            return
        box.clear()
        size = session.flown_size(S)
        if not size:
            return
        span = session.span_box(S)
        ws = session.wing_loading(S)
        # THE AREA IS A DESIGN-BOX ROW ON THE WATER SIZE MODES, and both
        # branches below are written for the air ones: each of them states
        # in words that the area is NOT searched (it "follows the mission's
        # W/S through the weight loop"), while the design box one click away
        # prints "area S 0.072-0.288 m^2 - searched". Two controls in the
        # same stage answering one question opposite ways, with the wrong
        # one on the card the mode is chosen on.
        area_box = session.span_box(S, "S_m2")
        # ...and the branch is chosen by the MEDIUM, not by whether the area
        # happens to be a box row. Routing on `area_box is not None` sent the
        # two water modes that do NOT free the area — "fixed" and "free SPAN"
        # — into the air branch, whose closing sentence tells the reader to
        # "choose 'area from the mission's W/S, span optimised' above": an
        # entry the water menu has never had, on the very card a foil
        # designer is hunting for a free span. A water craft has no wing
        # loading to derive an area from, so the whole air note is about
        # machinery this family does not carry, in all four modes.
        spec = api.PROBLEM_SPECS.get(S["wing"]["problem"])
        if area_box is not None or (spec is not None
                                    and spec.medium == "water"):
            _render_wet_size_note(size, span, area_box)
            return
        # one line per surface that has a span: with two of them a single
        # "span" readout would be reporting one wing of a pair
        rows = session.span_rows(S)
        pair = len(rows) > 1
        with box:
            if span is None:
                chosen = session.chosen_span(S)
                with ui.row().classes("items-center gap-3 w-full"):
                    widgets.readout("area", f"{size[1]:.4g}", "m²",
                                    tip="stage 1's: S = W/(W/S)"
                                    + (" — the PAIR's total" if pair else ""))
                    for row in rows:
                        _span_readout(row, pair)
                widgets.hint(
                    ("The span is the one above and the area is the mission's, "
                     "so the aspect ratio is whatever the two imply."
                     if chosen is not None else
                     "No span chosen, so the run flies √(AR · S) at the aspect "
                     "ratio stage 2 guessed, on the area the mission states — "
                     "the family's own published planform, bit-for-bit.")
                    + " Neither is searched: the design vector RESHAPES this "
                    "planform (taper, twist, chord law). Choose 'area from "
                    "the mission's W/S, span optimised' above to search the "
                    "span instead, inside a band you state."
                    + (" Each wing's aspect ratio is stated on its OWN share "
                       "of that area, which is what the solver checks."
                       if pair else "")
                    if session.size_mode(S) != "chords" else
                    "The two END CHORDS are the stated planform: the taper is "
                    "fixed at their ratio (a pinned row of the design box, "
                    "not searched) and the span above is what the mission's "
                    "area then implies.")
                _chord_size_note()
                _ar_disagreement()
                return
            with ui.row().classes("items-center gap-3 w-full"):
                widgets.readout("area", f"{size[1]:.4g}", "m²",
                                tip=("where the band opens: the MISSION's "
                                     "wing loading, not this run's — the "
                                     "area follows whichever loading each "
                                     "candidate flies"
                                     if session.loading_is_searched(S)
                                     else "from the mission's wing loading")
                                + (" — the PAIR's total" if pair else ""))
                for row in rows:
                    _span_readout(row, pair)
                # the span is the question this mode asks, so the way to the
                # place it is asked belongs beside the number it produces
                ui.button("set the band", icon="tune",
                          on_click=lambda: ctx.select("wing", "box")) \
                    .props("flat dense size=sm no-caps")
            subject = ("The two spans are the design variables, each inside "
                       "its own band in the design box"
                       if pair else
                       "The span is the design variable, inside the band the "
                       "design box holds")
            ws_band = (session.ws_band(S) if session.loading_is_searched(S)
                       else None)
            if ws_band is not None:
                # the OTHER loading mode: W/S is a row of the design box, so
                # saying "it follows the mission's W/S" would name a number
                # this run does not use
                cap = session.mission_ws_ceiling(S)
                widgets.hint(
                    f"{subject}, and so is the WING LOADING: "
                    f"{ws_band[0]:.4g}–{ws_band[1]:.4g} N/m² "
                    f"({ws_band[0] / 9.80665:.3g}–{ws_band[1] / 9.80665:.3g} "
                    f"kg/m²). The area follows whichever loading a candidate "
                    f"flies, through the same weight loop, so the trim lift "
                    f"coefficient is (W/S)/q for ITS loading. The mission's "
                    f"own W/S = {ws:.4g} N/m² is where that band opens, not "
                    f"what the run flies."
                    + ((" The W/S you stated is the ceiling"
                        if session.ws_cap_source(S) == session.WS_CAP_STATED
                        else " The mission's constraint diagram caps it")
                       + f" at {cap:.4g} N/m², and any candidate above that "
                         f"is refused."
                       if cap else ""))
                if session.device_reaches_past_the_span(S):
                    widgets.hint(
                        "This family's tip device is NOT span-capped: it "
                        "projects outboard of the span above.", "warn")
                _ar_disagreement()
                return
            widgets.hint(
                (f"{subject}. The area is not searched: it follows the "
                 f"mission's W/S = {ws:.4g} N/m² through the weight loop, so "
                 f"the trim lift coefficient is fixed at (W/S)/q and the "
                 f"aspect ratio is whatever the chosen span implies."
                 if ws else
                 f"{subject}; the area follows the "
                 "mission's wing loading, so the aspect ratio is whatever the "
                 "chosen span implies.")
                + (" It is the PAIR's total area, split between the two wings "
                   "by the split the search itself chooses, so each wing's "
                   "aspect ratio is read on its own share." if pair else ""))
            if session.device_reaches_past_the_span(S):
                widgets.hint(
                    "This family's tip device is NOT span-capped: it projects "
                    "outboard of the span above, so that band bounds the wing "
                    "panel and not the width the aircraft occupies.", "warn")
            else:
                widgets.hint(
                    "The tip device is span-capped, so this band is the "
                    "PROJECTED span — the wing panel shrinks to pay for the "
                    "device's own projection, and the number above is the "
                    "width the aircraft occupies."
                    if W["choices"].get("winglets", "none") != "none" else
                    "No tip device, so the span above is the width the "
                    "aircraft occupies.")
            _ar_disagreement()

    def _ar_disagreement():
        """Stage 2 designed the section for a chord. Does the flown wing
        still have it?

        The estimate is a GUESS and the flown aspect ratio is a consequence,
        so they drift apart the moment a span is chosen (or a span band
        narrowed). Saying so is half the job; the button is the other half —
        the SECTION follows the WING, never the reverse.
        """
        flown = session.flown_aspect_ratio(S)
        est = session.section_aspect_ratio(S)
        if abs(flown - est) <= 1e-9 * max(1.0, abs(flown), abs(est)):
            return
        widgets.hint(
            f"Stage 2 designed its section for AR {est:.3g}; the wing this "
            f"box implies is AR {flown:.3g}, so the section's Reynolds "
            f"number is that other wing's.", "warn")
        ui.button(f"re-design the section for AR {flown:.3g}",
                  icon="sync", on_click=_adopt_in_section) \
            .props("flat dense size=sm no-caps")

    def _adopt_in_section():
        if not session.set_section_aspect_ratio(
                S, session.flown_aspect_ratio(S)):
            return
        ctx.log(f"the section's chord now follows the flown wing — AR "
                f"{session.section_aspect_ratio(S):.4g}. Re-screen (or "
                f"re-optimise) to design at it.", "info")
        _render_size_note()
        _render_right()
        ctx.render_when_shown("mission")
        ctx.render_when_shown("airfoil")
        ctx.refresh()
        _back_to_section()

    def _tail_controls(water: bool = False):
        """The second surface's own card.

        The ARM — how far aft of the wing the surface sits — is the tail's
        defining dimension and belongs on screen in every medium (the water
        solver carries it as ``l_t_m`` exactly like the air one, and hiding
        it left the elevator with no horizontal position at all). It is
        asked one card down, as the HORIZONTAL SEPARATION, beside the CG and
        the vertical separation: those three numbers place the surface, and
        the card that asks them is the one that reports the static margin
        they imply — see :func:`_trim_layout_controls`. What water genuinely
        does not have is the air LAYOUT (no fin, no fuselage, no V-tail, all-moving by
        construction — see nice_app._derive_hydro_tail), so those controls
        are the ones that stay behind.
        """
        ch = W["choices"]
        with ui.column().classes("w-full gap-2 pl-3").style(
                f"border-left:2px solid {theme.RULE_SOFT}"):
            if not water:
                # WHICH EMPENNAGE is stage 1's question now, not this card's:
                # it decides how many surfaces the aircraft has (a V-tail has
                # no separate fin, a T-tail connects the two) and therefore
                # which section stages exist, which is a configuration
                # question rather than a design-box one. Reported here —
                # every note this card carried about the layout is about the
                # box under it — with the way back to where it is answered.
                ttype = str(ch.get("tail_type", "conventional"))
                with _row("empennage"):
                    widgets.tag(v1.TAIL_TYPE_LABELS.get(ttype, ttype),
                                theme.ACCENT)
                    ui.button("change it on stage 1", icon="arrow_back",
                              on_click=lambda: ctx.select("mission",
                                                          "operating")) \
                        .props("flat dense size=sm no-caps")
                widgets.hint(TAIL_TYPE_V3_NOTES.get(ttype, ""))
                if ch.get("tail_type") == "v_tail":
                    widgets.number_field(
                        "dihedral", ch.get("tail_dihedral_deg", 35.0),
                        lambda e: set_choice("tail_dihedral_deg",
                                             float(e.value or 35.0)),
                        unit="deg", step=1.0)
                _fin_controls()
                # NO _fuselage_controls() HERE. The body's diameter is only
                # a question where the HORIZONTAL SEPARATION is optimised,
                # so it is asked beside that toggle (_trim_layout_controls)
                # rather than two blocks above it under a card about the fin.
            # NO arm control here — not because the arm is not a question,
            # but because of WHERE it is asked. The arm IS the horizontal
            # separation, and it is asked one card down (searched or stated,
            # exactly like the vertical one) in the place that also reports
            # the static margin it moves (_trim_layout_controls). What was
            # wrong before was the DISTANCE between the two controls: a
            # "fixed arm?" toggle here and "where is the CG?" two cards down
            # produced a margin the user could not attribute to either.
            # NO height control here either, and for the same reason as the
            # arm: the VERTICAL separation is the other half of where the
            # surface sits, so it is asked beside the horizontal one, in the
            # card that reports what the layout does to the static margin
            # (_trim_layout_controls). Freeing it there is what puts z_t_m in
            # the design vector.
            # the second surface's OWN design freedoms, asked the way the
            # WING's are asked — one control per freedom, same words, same
            # order. A single three-valued "design it" menu described the
            # same three answers, but it made the surface read as a fitting
            # with a detail level rather than as a surface with freedoms.
            _surface_design_controls(water)
            # NO fin-drag switch and NO control-type toggle. V1/V2 offer both;
            # this shell offers neither, for the two reasons in
            # session.V3_PINNED_CHOICES: the fin is not a surface this package
            # models, and the elevator/stabilator choice cannot reach the
            # answer. Every V3 session therefore runs an all-moving surface
            # with the fin uncharged, and no control here can write either.
            if water:
                widgets.hint(
                    "Under water the surface is all-moving and there is no "
                    "fin: the air LAYOUT menu (T-tail, V-tail, canard) is "
                    "what does not appear here.")

    def _trim_layout_controls(water: bool, surface: str):
        """WHERE the second surface is, and where the CG is — the two numbers
        that decide the SIGN of what it carries.

        Two questions, both asked here, in the card that reports what they do
        to the static margin between them.

        The CG is ALWAYS determined: a field in every state, in metres aft
        (negative = FORWARD) of the wing's own aerodynamic centre, opening on
        the value this family flies when nobody states one. The HORIZONTAL
        SEPARATION beside it has the same two answers the vertical one has —
        STATE it (``l_t_m`` travels as a flag and the arm leaves the design
        vector) or OPTIMISE it (the arm stays a design variable and the
        design box's ``l_t_m`` row is its min and max).

        They used to be one either/or control ("stated as"), which made the
        arm unaskable whenever the user wanted to place the CG. What that
        control was really guarding against was DISTANCE — the arm asked in
        the configuration card and the CG two cards down, so the margin that
        came out belonged to neither — and both questions living here is what
        actually fixes that.

        A CG ahead of the wing's own centre of lift has to be balanced by a
        DOWNLOAD aft; behind it, by lift. That sign is not a detail: it sets
        which way the tip device pays (``_tip_direction_control``), which way
        the section should be cambered, and which way the elevator deflects.
        Each family ships a CALIBRATED default (chosen so the static-margin
        boundary crosses its own design box), which is a sensible starting
        point and not a statement about the aircraft being designed.

        Underneath, the STABILITY the pair implies, because neither number
        states it on its own: the neutral point of the whole aircraft and the
        static margin about it (``session.pitch_stability``).
        """
        sp = api.PROBLEM_SPECS[W["problem"]]
        ch = W["choices"]
        if api.TAIL_CG_KEY not in sp.flags:
            return
        widgets.hairline()
        own = _published_layout()
        arm_fixed = ch.get("tail_arm") == "fixed"
        aft = "foil" if water else "wing"
        # THE CG, in every state. It is the number that decides the SIGN of
        # what the surface carries, so a card that could not show it while
        # the arm was stated was hiding the design's own question behind a
        # layout choice. In its own container: in water the value it opens on
        # is a FRACTION OF THE ARM, so the field beside it moves this one —
        # and the arm's field may not rebuild the view it is being typed into
        # (the focus trap), which is what a container is for.
        boxes["cg"] = ui.column().classes("w-full gap-0")
        _render_cg(water)
        # ...and WHERE the surface sits behind it, asked with the same two
        # answers as the vertical separation below (_vertical_separation_
        # controls): state it, or give it to the optimiser.
        arm_box = session.published_arm_box(S)
        with _row("separation"):
            tg = ui.toggle({"fixed": "you state it", "free": "optimise it"},
                           value="fixed" if arm_fixed else "free",
                           on_change=lambda e: _set_arm(e.value)) \
                .props("dense no-caps unelevated toggle-color=primary")
            other = "free" if arm_fixed else "fixed"
            if not v1.option_available(ch, "tail_arm", other):
                tg.disable()
                tg.tooltip(f"no {other}-arm solver for this configuration")
        if arm_fixed:
            widgets.number_field(
                f"aft of the {aft}", widgets.shown(ch.get("tail_arm_m")),
                lambda e: _set_trim_number("tail_arm_m", _num(e.value)),
                unit="m", step=0.25,
                tip=f"the {aft}'s quarter chord to the {surface}'s own — "
                    f"the arm, stated instead of searched")
            widgets.hint(
                "Stated, so the arm LEAVES the design vector — there is no "
                "l_t_m row in the design box while this is on, which is the "
                "honest form of the answer: a zero-width bound would break "
                "the samplers instead."
                + (f" Any positive length: this family's results were "
                   f"MEASURED over {arm_box[0]:g} – {arm_box[1]:g} m, which "
                   f"is the box the search opens on, but a longer aeroplane "
                   f"is a design decision and the solver takes it."
                   if arm_box else ""))
            # the extrapolation note is a READ-OUT of the number in the field
            # above it, so it lives in its own container for the same reason
            # the margin does: typing the arm may not rebuild the type view,
            # and a note that only redraws with the whole view is a note that
            # never appears while the number is being typed.
            boxes["arm_warn"] = ui.column().classes("w-full gap-0")
            _render_arm_warn()
        else:
            # the sentence QUOTES the design-box row, which is edited in
            # another view — so it lives in a container of its own and is
            # redrawn when that row moves (``_set_bound``). Without that it
            # is a read-out that goes stale the moment the band it describes
            # is typed, which is the trap the CG and the margin above are
            # already in their own containers for.
            boxes["arm_note"] = ui.column().classes("w-full gap-0")
            _render_arm_note(water)
            with ui.row().classes("items-center gap-2"):
                # the band IS the design box's row, and the way to the place
                # it is asked belongs beside the sentence that quotes it —
                # the same route the size card offers for the span band
                ui.button("set the band", icon="tune",
                          on_click=lambda: ctx.select("wing", "box")) \
                    .props("flat dense size=sm no-caps")
        # ...and the ONE number that makes a searched separation a well-posed
        # question, asked here because this is the toggle that decides
        # whether it is a question at all (_fuselage_controls). In water
        # there is no body: the air-only card is skipped with the rest.
        # THE BODY IS NOT ASKED HERE ANY MORE. It is a configuration
        # question — "does this aeroplane have a fuselage" — and it is asked
        # in the configuration card, where the reader is already answering
        # what the aeroplane is. Asked here it was a component that appeared
        # and disappeared with the separation toggle two rows above it.
        # the margin READ-OUT gets its own container: the three numbers that
        # move it are VALUE_KEYS, and typing one must not rebuild the view
        # its own field is in (the focus trap) — so the readout is redrawn
        # on its own instead, and cannot go stale under a typed CG.
        boxes["stability"] = ui.column().classes("w-full gap-0")
        _render_stability(water)
        _handling_level_control()
        _vertical_separation_control(water, own)
        _strut_station_control(water)
        widgets.hint(
            f"This is what the {surface} CARRIES — but the CG is only half "
            f"of it. The other half is the {aft}'s own SECTION COUPLE: a "
            f"cambered aerofoil pitches nose-down at every lift, including "
            f"zero, and the {surface} has to hold that too. So the load "
            f"crosses zero not at the {aft} AC but 0.12 m behind it, and any "
            f"normal loading forward of that gives a DOWNLOAD. The value the "
            f"CG field opens on is a real cruise CG (30 % MAC on the air "
            f"families), chosen to be one and checked to keep the "
            f"static-margin boundary inside this family's design box. The "
            f"run reports the lift it landed on (CL_t) and the incidence "
            f"that trims it.")

    # NO MOUNTING BLOCK. The second surface is always built to push DOWN
    # (``tail_mount = "inverted"``, sent from ``gui.v3.config.flags``), so
    # there was never a question here — first a toggle, then a read-out chip
    # and a paragraph explaining it. Both are gone: the card states the
    # layout's DECISIONS, and this is not one.
    #
    # The engine flag is untouched and still takes all three values
    # (``api.TAIL_MOUNT_KEY``, ``tail.orient_section``); the shell simply
    # sends the one and says nothing about it. What that costs is stated
    # rather than hidden: the geometry views draw this section mirrored and
    # stage 2 screens it at the lift it sees mounted that way, and no card
    # now accounts for the flip.

    def _fin_geometry():
        """The fin this configuration implies, or None if it cannot be said.

        Through ``fin.fin_for_layout`` — the ONE sizing law, the same call
        the drag book, the report, the CAD export and the flight rebuild
        make — so the surface described here is the surface that flies. A
        searched arm has no single answer, so the caller is handed the arm
        it was computed at and says so.

        THE THREE INPUTS ARE READ WHERE THE RUN READS THEM, and each of them
        was being invented here instead:

        * the ARM came from ``W["flags"][api.TAIL_ARM_KEY]``, and there is no
          ``api.TAIL_ARM_KEY`` — the ``hasattr`` beside it turned that into a
          silent ``None``, so every fin on this card was drawn at the middle
          of the SEARCH BAND even when the user had stated an arm. Measured:
          arm typed 3.0 m, card still printing the 5.5 m fin (0.727 m2 where
          the law says 1.333). The arm now comes from ``tail_flags``, which
          is the one translation from this card's ``tail_arm_m`` to the
          ``l_t_m`` the solver is handed;
        * the HEIGHT was always the LAYOUT's (``tail.tail_height``), so a
          stated vertical separation moved the tailplane and left the fin's
          foot where it was. ``TailProblem.dz_for`` prefers a stated
          ``z_t_m`` and so does this;
        * the SHAPE was not passed at all, so the card printed the 0.04/1.5
          fin under two fields the user had just typed 0.08 and 3.0 into.

        A PAIR IS THE OTHER ARRANGEMENT and was not answered here at all.
        Every input above is a WING+TAIL input: a tandem sends no ``l_t_m``
        (``tail_flags`` returns ``{}`` for one) and has no published arm box,
        so both branches fell through and this returned ``(None, None)`` for
        every configuration of the family — "state the size and the
        separation and this fills in", forever, on a card whose separations
        are the STAGGER and are asked three rows up. Its station has one
        author, ``fin.tandem_fin_station``, which is the same function the
        two tandem engines charge the drag on and ``api.design_report`` sizes
        and lofts at, so the fin drawn here is the fin flown.
        """
        from aerobo import fin as finmod, tail as tailmod
        from gui.nice_app import tail_flags

        size = session.flown_size(S)
        if not size:
            return None, None
        b, area = float(size[0]), float(size[1])
        # WHICH LAYOUT, asked of the REGISTRY and not of the menu: the boom
        # key is declared by the pair and by nothing else, so a family that
        # has one is a family whose fin stands on one (the rule
        # ``shell-must-not-infer-a-mode-from-a-vector`` keeps).
        if api.TANDEM_FIN_BOOM_KEY in sp().flags:
            dx, dz_pair = v1.tandem_stagger(W["choices"])
            boom = W["choices"].get("tandem_fin_boom_m")
            try:
                x_qc, z_root = finmod.tandem_fin_station(
                    float(dx), float(dz_pair),
                    None if boom is None else float(boom))
                # ``fin_for_layout`` with the pair's own station: not a
                # T-tail, so the foot is ``dz`` and the height is the sizing
                # law's — which is exactly the ``size_fin`` call
                # ``api.design_report`` makes for a pair, arguments and all.
                g = finmod.fin_for_layout(
                    b=b, S=area, l_t=float(x_qc), tail_type="tandem",
                    dz=float(z_root), **finmod.fin_law_kwargs(W["flags"]))
            except (ValueError, KeyError, ZeroDivisionError):
                return None, None
            # the station is STATED on this card, never searched, so there is
            # no band to quote it at
            return g, None
        sent = tail_flags(W["choices"])
        arm = sent.get("l_t_m")
        searched = arm is None
        if searched:
            box = session.published_arm_box(S)
            if not box:
                return None, None
            arm = 0.5 * (float(box[0]) + float(box[1]))
        try:
            stated_dz = sent.get(api.TAIL_HEIGHT_KEY)
            dz = (float(stated_dz) if stated_dz is not None else
                  tailmod.tail_height(str(ch_tail_type()), float(arm), b, area))
            g = finmod.fin_for_layout(
                b=b, S=area, l_t=float(arm), tail_type=str(ch_tail_type()),
                dz=dz, **finmod.fin_law_kwargs(W["flags"]))
        except (ValueError, KeyError, ZeroDivisionError):
            return None, None
        return g, (float(arm) if searched else None)

    def ch_tail_type() -> str:
        """The EMPENNAGE this design is arranged as — and the conventional
        one wherever that question is not asked.

        WHICH empennage is stage 1's AIR question: a water craft has no fin,
        no V-tail and no T-tail (its vertical is the mast, and the card here
        says so), and the track has no empennage at all. The CHOICE, like
        every other, survives a medium switch — so a session that picked a
        T-tail in air and then moved to water went on reading ``t_tail``
        here, which disabled the elevator's DEPTH toggle (and displayed it
        as "fixed") on a family whose FREE-DEPTH twin it was already flying,
        and drew "the fin's own span" as the separation of a craft that has
        no fin. A layout is answered where the layout is asked.
        """
        if S["medium"] != "air":
            return "conventional"
        # ...AND NOR IS IT ASKED WITHOUT A SECOND SURFACE. An empennage is
        # how that surface is ARRANGED (stage 1 disables the select and says
        # "add a tail first"), so a family with none has no answer to give —
        # and a TANDEM is the case that made the stale one visible: its
        # second surface is the rear wing, and a session that had chosen a
        # V-tail earlier went on reading ``v_tail`` here, which printed "a
        # V-tail has NO separate fin" over the card of a pair whose fin the
        # run sizes, charges, lofts and flies. Same rule as the medium
        # above, one family across; ``session.fin_surface`` states it too.
        if not W["choices"].get("tail"):
            return "conventional"
        return str(W["choices"].get("tail_type", "conventional"))

    # NO FIN_SHAPE_ROWS. The card used to ask for the fin's VOLUME
    # COEFFICIENT and its ASPECT RATIO, and the user's rule for this surface
    # is the one already recorded against its drag switch — *"I don't want it
    # to be an option... do as the horizontal tail"*. The horizontal surface
    # is never asked for a volume coefficient or an aspect ratio: its area is
    # the solver's and its shape is the planform question below. So neither
    # is the vertical one. The sizing law (``fin.size_fin``) keeps its
    # published constants, the read-out says what they are, and the three
    # flags stay in the registry for a library caller or a stored record —
    # ``_fin_geometry`` reads them, so a session that HAS one still draws it.
    #
    # NO THICKNESS ROW either, and for the older reason: t/c is a property of
    # the SECTION, chosen on stage 2.7, so asking for it here made one number
    # answerable in two places.

    def _fin_is_on_a_boom() -> bool:
        """Does this family's fin stand on a BOOM aft of a rear wing?

        True for the tandem pair and for nothing else, and asked of the
        REGISTRY: ``api.TANDEM_FIN_BOOM_KEY`` is declared by the two tandem
        engines alone, so a family that carries the flag is a family whose
        fin station is the stagger plus a boom (``fin.tandem_fin_station``).
        Read here rather than off ``choices["system"]`` for the rule
        ``shell-must-not-infer-a-mode-from-a-vector`` states: the menu says
        what was asked for, the spec says what will be flown.
        """
        return api.TANDEM_FIN_BOOM_KEY in sp().flags

    def _set_fin_boom(value):
        """The PAIR'S BOOM [m] — how far aft of the rear wing the fin stands.

        A CHOICE and not a flag (it travels through
        ``nice_app.tandem_flags``), so it goes through ``set_choice`` like
        the two stagger lengths beside it — and then the derived half is
        redrawn by hand, exactly as :func:`_set_fin_number` does. It has to
        be: the key is in :data:`VALUE_KEYS`, which is what stops
        ``set_choice`` rebuilding the type card and taking the cursor out of
        this very field mid-number, and that same skip is what would
        otherwise leave the four read-outs above quoting the fin of the
        previous boom.

        Blank CLEARS it, back to ``fin.TANDEM_FIN_BOOM_FRAC`` of the stagger
        — the field's placeholder — so an untouched card is bit-for-bit the
        published pair.
        """
        set_choice("tandem_fin_boom_m",
                   None if value in (None, "") else float(value))
        _render_fin_derived()

    def _set_fin_number(key: str, value):
        """A stated VALUE flag of the empennage. Rebuilds the DERIVED
        read-out only.

        Not the whole card: these are typed numbers, and a handler that
        rebuilt its own field would take the cursor out of it on every
        keystroke. The fin's area/height/chord are the derived half and DO
        have to follow, or the card would quote a fin the flags no longer
        describe.

        Its only caller left is the BODY DIAMETER — the fin's own shape rows
        are withdrawn — and the redraw still matters there for the day the
        fin's drag reaches the read-out. Clearing the field REMOVES the flag,
        which is what makes the body's own "hidden means unanswered" rule
        reachable from the keyboard.
        """
        if value in (None, ""):
            W["flags"].pop(key, None)
        else:
            W["flags"][key] = float(value)
        _render_fin_derived()

    def _render_fin_derived():
        """The fin the stated shape implies — from the sizing law itself."""
        box = boxes.get("fin_derived")
        if box is None:
            return
        box.clear()
        g, arm_at = _fin_geometry()
        with box:
            if g is None:
                widgets.hint(
                    "The fin follows from the span, the area and the arm, "
                    "and one of those is not settled yet — state the size "
                    "and the separation and this fills in.")
                return
            widgets.hint(
                f"Sized by VOLUME COEFFICIENT, the way the drag book has "
                f"always sized it: S_vt = V_v · b · S / arm, at "
                f"V_v = {g.V_v:g} and aspect ratio {g.AR:.3g}."
                + (f" Shown at the middle of your {arm_at:.3g} m arm band, "
                   f"because a searched arm has no one answer — the fin "
                   f"shrinks as the arm grows." if arm_at is not None else ""))
            with ui.row().classes("items-center gap-3 w-full"):
                widgets.readout("area", f"{g.S:.3g}", "m²")
                widgets.readout("height", f"{abs(g.height):.3g}", "m")
                widgets.readout("chord", f"{g.chord:.3g}", "m")
                widgets.readout("t/c", f"{session.fin_thickness(S):.3g}")
            # ...and WHERE it sits. On a WING+TAIL that is not a second
            # question: both of its separations are the SECOND SURFACE's,
            # answered once below (`_trim_layout_controls` /
            # `_vertical_separation_control`), and reporting them here is
            # what lets this card be read on its own without becoming a
            # place they can be answered differently.
            #
            # ON A PAIR THEY ARE NOT THE SECOND SURFACE'S. The rear wing is
            # the second surface, and the fin stands on a BOOM aft of it —
            # a third length, asked by the field above this box, which is
            # the only place on the aeroplane it is answered.
            pair = _fin_is_on_a_boom()
            with ui.row().classes("items-center gap-3 w-full"):
                widgets.readout("quarter chord", f"{g.x_qc:.3g}", "m aft",
                                tip=("the yaw arm, measured from the FRONT "
                                     "wing's quarter chord — the stagger "
                                     "plus the boom above"
                                     if pair else
                                     "the yaw arm — the HORIZONTAL "
                                     "separation, answered below"))
                widgets.readout("foot", f"{g.z_root:.3g}", "m up",
                                tip=("where it meets the rear wing's root — "
                                     "the pair's vertical stagger, asked on "
                                     "the configuration card"
                                     if pair else
                                     "where it meets the body — the VERTICAL "
                                     "separation, answered below"))
                # ...AND HOW LONG AN AEROPLANE THAT MAKES. The station is
                # the only length on a pair that nothing bounds — the fin's
                # area falls as 1/l_t, so the score rewards a longer boom
                # monotonically and no structure is weighed or charged for
                # it. There is no bound to add (a calibration is a default,
                # not a ban), so the number the reader would have to work
                # out for themselves is stated instead: this is the SAME
                # length the handling gate already takes the pair's inertia
                # off (``tandemvlm``: "a pair owns no fuselage, so its length
                # is the boom"), so the card and the gate cannot disagree.
                if pair:
                    from aerobo import drag as dragmod

                    widgets.readout(
                        "aeroplane length",
                        f"{dragmod.body_length_for_arm(g.x_qc):.3g}", "m",
                        tip="what a boom this long makes the whole vehicle "
                            "— drag.body_length_for_arm, the length the "
                            "handling gate takes this pair's inertia off")
            if pair:
                dx_p, dz_p = v1.tandem_stagger(W["choices"])
                widgets.hint_help(
                    f"The quarter chord is the {dx_p:g} m stagger plus the "
                    f"boom above; the foot stays on the rear wing's root "
                    f"however long the boom is.",
                    "WHAT THE BOOM BUYS, measured on this family at its own "
                    "box centre. The fin's area is a VOLUME COEFFICIENT — "
                    "S_vt = V_v·b·S/l_t — so lengthening the boom shrinks "
                    "the surface, and both the drag and the yaw stiffness "
                    "fall with it:\n\n"
                    "  boom 0 m (on the rear wing): S_vt 1.600 m², "
                    "L/D 24.59, Cn_beta +0.1031, spiral −0.00975, and it "
                    "takes 10.5° of dihedral to turn that spiral;\n"
                    "  boom 5 m (the default, one stagger aft): S_vt "
                    "0.800 m², L/D 25.11, Cn_beta +0.0665, spiral "
                    "−0.00125, and 3.5° of dihedral turns it.\n\n"
                    "So the ON-THE-WING station buys yaw STIFFNESS (the "
                    "rear wing works as an end plate) and the boom buys yaw "
                    "DAMPING, which is what the `spiral` criterion reads. "
                    "Stiffness sits on the wrong side of that product, so a "
                    "pair asked to be stable wants the arm and a pair asked "
                    "for the smallest fin wants the wing.\n\n"
                    "NOTHING CHARGES THE BOOM ITSELF. It is drawn, and its "
                    "length is what the handling gate takes the pair's "
                    "inertia off, but no structure weight and no body drag "
                    "are billed for it — so on L/D alone a longer boom is "
                    "monotonically better (25.46 at 20 m, 25.69 at 200 m) "
                    "and there is no interior optimum for the station. "
                    "State the boom your airframe actually has.",
                    title="Where a pair's fin stands")
            else:
                widgets.hint(
                    "Both of those are the second surface's own separations, "
                    "asked once, below — state them or give them to the "
                    "optimiser there and this follows.")
            # WHERE the thickness came from — the one property of the fin
            # this card reports rather than asks
            own = session.section_of(S, "fin") or {}
            own_name = str(own.get("name") or "its own default")
            widgets.hint(
                f"Its thickness is its SECTION's: "
                f"{own_name}, chosen on "
                f"“{session.stage_label(S, 'airfoil_fin')}”. That is the t/c "
                f"the form factor charges its parasite drag on and the t/c "
                f"the export lofts."
                if session.section_is_own(S, "fin") else
                "No section has been chosen for it yet, so it flies its own "
                f"symmetric stand-in — {own_name} — and "
                f"that section's thickness is what is charged and lofted. "
                f"Choose or search one on "
                f"“{session.stage_label(S, 'airfoil_fin')}”.")
            if ch_tail_type() == "t_tail":
                widgets.hint(
                    "A T-TAIL puts the tailplane on the fin's TIP, so the "
                    "fin runs from the body up to it and its span IS the "
                    "tailplane's height. That is what makes it a T-tail "
                    "rather than a conventional tail mounted higher.")

    def _set_fuselage(on: bool):
        """The body's on/off switch.

        ON writes the diameter the shell can defend
        (:func:`session.fuselage_diameter_default` — the slenderness measured
        to leave the searched arm an interior optimum) rather than an empty
        field the user has to fill before anything is charged: a switch that
        turns something on and then charges nothing is a switch that lies.
        A diameter already stated is the user's and is kept.

        OFF removes the flag outright, which is what "no body" IS in the
        api — not a zero, which ``tail._fuselage_charge`` refuses.
        """
        key = api.FUSELAGE_KEYS[0]
        if on:
            if W["flags"].get(key) in (None, ""):
                W["flags"][key] = session.fuselage_diameter_default(S)
        else:
            W["flags"].pop(key, None)
        _render_type()
        _render_solver()
        ctx.refresh()

    def _fuselage_controls():
        """THE BODY — because without it a longer aeroplane is free.

        The arm raises the static margin and the tail volume at almost no
        cost while the fuselage's weight is carried as a constant and its
        DRAG is not modelled at all. Measured on this family: L/D rose
        monotonically with the arm all the way to 100 m, no interior
        optimum — the search would have made an aeroplane nobody can build.
        Charging ``drag.fuselage_cd0`` on a body 1.6 arms long puts the
        turning point back at 3.0 m.

        ASKED WHERE IT DECIDES SOMETHING, which is where the separation is
        OPTIMISED — the user's rule, and the plan's: *"the user will only
        have to state the diameter of the fuselage if selected to optimise
        separation"*. With the arm searched and no body the problem is
        ill-posed and the answer is always the longest aeroplane the box
        allows; with the arm STATED a constant added to CD cannot move a
        variable nobody is varying, so the question is not one. It used to be
        drawn in every state, under the Vertical tail card, two blocks above
        the toggle that decides whether it matters.

        It is still drawn on a stated arm WHEN A VALUE IS ALREADY THERE: a
        stored record, a preset or a session that switched the toggle back
        must not have a number it cannot see or clear. Hidden means
        unanswered, never silently answered.

        The api accepts absence so stored records still reproduce; the SHELL
        is where the user is told.
        """
        keys = [k for k in api.FUSELAGE_KEYS if k in sp().flags]
        if not keys:
            return
        key = keys[0]
        searched = api.arm_is_searched(W["problem"])
        stated = W["flags"].get(key)
        with widgets.group_box("Fuselage"):
            # THE SWITCH, because "has a body" is a fact about the aeroplane
            # and an empty number field is not a way to say no. It used to be
            # answered by the field's own emptiness AND by whether the arm
            # happened to be searched, so the question was reachable in one
            # state and invisible in the other — hidden means unanswered, and
            # this is what asks it.
            ui.switch("this aeroplane has a fuselage", value=stated is not None,
                      on_change=lambda e: _set_fuselage(bool(e.value))) \
                .props("dense")
            if stated is None:
                widgets.hint(
                    "No body: nothing is charged for one" + (
                        ". And the tail arm is SEARCHED on this run, so its "
                        "drag is unmodelled while its static margin and tail "
                        "volume are fully counted — the search runs to the "
                        "top of the arm band every time. Switch it on and "
                        "the trade closes."
                        if searched else
                        ", which is every published run. The arm is stated "
                        "on this run, so a constant added to CD cannot move "
                        "anything; switch it on to fly a body anyway."),
                    "bad" if searched else "")
                return
            widgets.number_field(
                "body diameter" + (" (required)" if searched else ""),
                stated,
                lambda e: _set_fin_number(key, e.value),
                unit="m", step=0.05,
                tip="the maximum body diameter; its LENGTH follows the arm "
                    "(1.6 × arm, drag.BODY_LENGTH_FRAC)")
            if not searched:
                widgets.hint(
                    "The separation is STATED on this run, so the body "
                    "changes no answer — a constant added to CD cannot move "
                    "an arm nobody is searching. It is still flown, charged "
                    "and exported; switch it off to fly without one.")
            else:
                widgets.hint(
                    "Charged as a Raymer body (drag.fuselage_cd0) on a "
                    "length of 1.6 × the arm. The report carries the "
                    "fineness ratio it implies, which says 'not an "
                    "aeroplane' faster than a drag count does.")

    def _fin_controls():
        """THE VERTICAL TAIL: what it is, and whether it is paid for.

        This card used to say, in a comment, that there was no fin control
        because "the fin is not a surface this package models: no panels in
        any solver ... nothing in geometry.py or cad.py". Every clause of
        that is now false — ``fin.py`` is one sizing law, ``vlm`` panels it,
        ``cad.fin_surface`` lofts it, the 3-D view draws it and the flight
        rebuild flies it. What survives is the LAST clause, and it is the
        one worth a control: the scored objective does not charge its drag
        unless asked, so the design flies a surface it did not pay for.

        The SIZE IS NOT ASKED, exactly as the horizontal surface's is not.
        This card REPORTS the fin the one law implies from the span, the
        area and the separation — the same call the drag book, the lateral
        deck, the report, the CAD export and the flight rebuild make — and
        every number on it is a read-out. See the note where
        ``FIN_SHAPE_ROWS`` used to be for why the two fields went.
        """
        # DOES THIS FAMILY HAVE A FIN TO DESCRIBE? Asked of the shape flags
        # it would be described with. It used to ask for ``charge_fin_drag``,
        # which was the drag SWITCH — a control that no longer exists, since
        # the fin's parasite drag is charged the way the tailplane's is.
        if not any(k in sp().flags for k in api.FIN_SHAPE_KEYS):
            return
        g, arm_at = _fin_geometry()
        with widgets.group_box("Vertical tail"):
            # ...AND DOES THIS AEROPLANE HAVE ONE? A different question from
            # the one above, and it was not asked at all: with the stage-1
            # switch off this card went on drawing a sized, quoted fin —
            # byte-identical to the switch-on text — for a surface that is
            # now neither built, charged, weighed nor flown.
            #
            # AND WHICH "no" IT IS. ``fin_surface`` answers both — the
            # switch and the LAYOUT — and it answers the layout FIRST, so
            # the V-tail's own paragraph, which used to sit below this
            # branch, could never be reached: every V-tail read "you said so
            # on stage 1" about a decision the layout had made. Two
            # sentences, one branch, and the reason is named.
            if not session.fin_surface(S):
                widgets.hint(
                    "A V-TAIL HAS NO SEPARATE FIN — its two canted panels "
                    "carry the yaw, which is the whole point of the layout. "
                    "Nothing below applies and nothing draws one: the sizing "
                    "law itself returns none here, so the export, the 3-D "
                    "view and the flight model all agree. Choose a "
                    "conventional or T-tail on stage 1 to design a fin of "
                    "your own."
                    if ch_tail_type() == "v_tail" else
                    "This aeroplane carries NO vertical stabiliser — you "
                    "said so on stage 1. Nothing is drawn, nothing is "
                    "charged for it and the empennage weighs the tailplane "
                    "alone; the flight stage reports Cn_beta as exactly "
                    "zero. Turn “add a vertical stabiliser (fin and "
                    "rudder)” back on in stage 1 to design one.")
                return
            # WHERE THE PAIR'S FIN STANDS — the one question this card ASKS.
            #
            # It is asked HERE and not on the pair's layout block, where it
            # used to be, for the reason the withdrawn shape rows have at the
            # top of this card: the question and the four numbers it moves
            # belong together. There it sat between a paragraph about the
            # rear wing's stagger and one about the rear wing's span, with no
            # sentence of its own — a "Fin," field under two "Rear wing,"
            # fields, reading as a third rear-wing length — while the surface
            # it places was drawn on no card at all.
            #
            # OUTSIDE ``boxes["fin_derived"]`` deliberately. Its handler
            # redraws that box, and a field inside a container rebuilt from
            # its own ``on_change`` loses the focus and swallows the rest of
            # the number (the trap ``VALUE_KEYS`` exists for). So the field
            # is drawn once with the card and the read-outs follow it from
            # underneath.
            if _fin_is_on_a_boom():
                v1._num_row(
                    "Fin, aft of the rear wing by", "tandem_fin_boom_m",
                    W["choices"], lambda _k, v: _set_fin_boom(v),
                    suffix="m", step=0.5, lo=0.0,
                    placeholder=f"{v1._fin_boom_default(W['choices']):g}")
                widgets.hint(
                    "Blank is one stagger aft of the rear wing "
                    "(fin.TANDEM_FIN_BOOM_FRAC) — a fraction and not a "
                    "length, so it follows the layout instead of imposing a "
                    "5 m boom on a pair staggered 2 m. 0 stands the fin ON "
                    "the rear wing's root, which is the station this family "
                    "shipped with and is kept reachable rather than taken "
                    "away. A pair has no fuselage to hang a fin off, so "
                    "unlike every other layout here this length is a real "
                    "choice about the airframe and nothing in the run makes "
                    "it for you.")
            boxes["fin_derived"] = ui.column().classes("w-full gap-1")
            _render_fin_derived()
            widgets.hint(
                "Its parasite drag is CHARGED, the way the tailplane's is — "
                "there is no switch, because there is no version of this "
                "aeroplane that flies a fin without one. It was optional "
                "and off, and that allowance was worth +5.5 % L/D on this "
                "family: the design flew a surface it did not pay for while "
                "the same surface carried all of its yaw stiffness in the "
                "Controls and Flight stages.")

    def _fixed_height_is_stated() -> bool:
        """Would picking "you state it" give the user a FIELD?

        The question the height toggle's label has to answer, and it is about
        the family the option SELECTS, not the one the session is on. Reached
        through ``derive_problem`` on the session's own choices — never by
        editing a problem NAME — for the reason ``session.published_arm_box``
        gives: the names differ per family and a helper that matches one
        spelling silently answers for the other.
        """
        from gui.nice_app import derive_problem

        if ch_tail_type() == "t_tail":
            # the registry says yes and the LAYOUT says no: a T-tail's
            # tailplane sits on the fin's tip, so the height is the fin's
            # span and there is no field under this toggle to type it into
            return False
        try:
            name, _ = derive_problem(dict(W["choices"], tail_height="fixed"))
        except Exception:                   # noqa: BLE001 — a label
            return api.TAIL_HEIGHT_KEY in api.PROBLEM_SPECS[W["problem"]].flags
        spec = api.PROBLEM_SPECS.get(name)
        return bool(spec) and api.TAIL_HEIGHT_KEY in spec.flags

    def _strut_station_control(water: bool):
        """WHERE THE STRUT STANDS — the third number that places a surface.

        Asked HERE because this is the card that places surfaces: the CG, the
        arm and the vertical separation are the other three, and the strut's
        station is the same kind of answer. It is a water question and only a
        water question — an aeroplane's vertical is sized by a volume
        coefficient and stands at the tail by construction — so the row is
        drawn off the registry (``x_mast_frac`` is declared on the elevator
        families alone) rather than off a medium test.

        WHAT IT DOES, and why it is worth a control at all. The strut used to
        be reported standing AT the main foil, which is the station every arm
        on this craft is measured from: a vertical surface with no lever.
        Placed on the fuselage it becomes the craft's only yaw lever, and the
        sign of its moment is decided here — forward of the CG it is
        destabilising (which is what a real windfoil is: the rider stands
        over the mast and steers it), aft of the CG it weathercocks. The
        drag does NOT read the station, so this control moves stability and
        nothing else; the read-out under it says which side of the CG the
        answer landed on.
        """
        if not water:
            return
        sp = api.PROBLEM_SPECS.get(W["problem"])
        if not sp or "x_mast_frac" not in (sp.flags or ()):
            return
        if not session.fin_surface(S):
            return                       # no strut: nothing to place
        ch = W["choices"]
        free = ch.get("mast_station") == "free"
        with _row("strut station"):
            ui.toggle({"fixed": "you state it", "free": "optimise it"},
                      value="free" if free else "fixed",
                      on_change=lambda e: set_choice("mast_station", e.value)) \
                .props("dense no-caps unelevated toggle-color=primary")
            if not free:
                widgets.number_field(
                    "fraction of the fuselage",
                    (ch.get("mast_station_frac")
                     if ch.get("mast_station_frac") is not None
                     else _published_mast_frac()),
                    lambda e: _set_trim_number("mast_station_frac",
                                               _num(e.value)),
                    unit="", step=0.05,
                    tip="0 is the front wing's quarter chord and 1 is the "
                        "stabiliser's — the strut is bolted to the fuselage "
                        "between them")
        # its own container: ``mast_station_frac`` is a VALUE_KEY, so the
        # read-out under the field may not be redrawn by rebuilding the view
        # the field itself lives in (the focus trap)
        boxes["strut_station"] = ui.column().classes("w-full gap-0")
        _render_strut_station()

    def _published_mast_frac() -> float:
        """The station this family flies when nobody states one — read off a
        BUILT problem, never restated here."""
        from aerobo import hydrotail as _ht
        try:
            prob = api.PROBLEM_SPECS[W["problem"]].build({}, {}, None).problem
        except Exception:                       # noqa: BLE001 — a read-out
            return float(_ht.X_MAST_FRAC)
        foil = getattr(prob, "foil", None) or prob
        frac = getattr(foil, "x_mast_frac", None)
        return float(_ht.X_MAST_FRAC if frac is None else frac)

    def _arm_now():
        """The fuselage length this design flies — the STATED arm where there
        is one, else the middle of the box the search covers, which is the
        design every other read-out on this stage is quoted at."""
        from gui.nice_app import tail_flags
        arm = tail_flags(W["choices"]).get("l_t_m")
        if isinstance(arm, (int, float)):
            return float(arm)
        arm = _published_layout().get("l_t")
        if isinstance(arm, (int, float)):
            return float(arm)
        box = _bounds_of("l_t_m")
        return 0.5 * (float(box[0]) + float(box[1])) if box else None

    def _render_strut_station():
        """WHICH SIDE OF THE CG the stated station landed on, in metres.

        The fraction is the well-posed way to ASK the question (the arm is
        itself a design variable), and it is the wrong way to READ the
        answer: what decides the sign of the strut's yaw moment is the
        station against the CG, both in metres on this craft.
        """
        box = boxes.get("strut_station")
        if box is None:
            return
        box.clear()
        ch = W["choices"]
        if ch.get("mast_station") == "free":
            with box:
                widgets.hint(
                    "The solver chooses where the strut stands. It is a "
                    "stability row, not a drag one: the strut's wetted area "
                    "and the side force it carries do not read its station, "
                    "so what the search is buying here is yaw stiffness.")
            return
        frac = ch.get("mast_station_frac")
        frac = _published_mast_frac() if frac is None else float(frac)
        arm, cg = _arm_now(), _cg_now()
        with box:
            if arm is None or cg is None:
                widgets.hint(
                    "0 is the front wing's quarter chord, 1 the "
                    "stabiliser's.")
                return
            x = frac * float(arm)
            gap = x - float(cg)
            widgets.hint(
                f"That puts the strut {x:.3g} m aft of the front wing on a "
                f"{float(arm):.3g} m fuselage, with the CG at "
                f"{float(cg):.3g} m — so it stands {abs(gap):.3g} m "
                + ("AFT of the CG and weathercocks (positive yaw stiffness)."
                   if gap > 0 else
                   "FORWARD of the CG, so its own yaw moment is "
                   "DESTABILISING. That is what a real windfoil is — the "
                   "rider stands over the mast and steers it — and moving "
                   "the strut aft of the CG is what changes the sign."))

    def _vertical_separation_control(water: bool, own: dict):
        """The VERTICAL separation — stated, or searched.

        Where the surface sits has two components and this is the second one,
        so it is asked beside the first rather than in the configuration card
        two blocks up. Both answers exist in the registry and both are real:
        STATED sends ``z_t_m`` as a flag, OPTIMISED selects the family's free
        twin (``tail [free height]`` / ``hydrofoil + elevator [free depth]``)
        and puts ``z_t_m`` in the design vector with its own box row — 0.5–3
        m above the wing in air, −0.36…−0.06 m below the foil in water. It is
        not cosmetic: out of the wing's trailing sheet the surface sees less
        downwash and the neutral point moves aft, which is the margin the
        card above reports.

        A T-TAIL is the one layout that cannot answer it: its height IS the
        fin span its own Raymer sizing implies, so the toggle is disabled
        rather than offered and then refused — and no field is drawn under it
        either, which was the defect: ``dz_for`` honours a stated ``z_t_m``
        over the layout's own, so a typed height moved the tailplane off the
        fin it is supposed to be sitting on.

        A family that takes NEITHER answer keeps the "the layout's own"
        wording — but that is now decided by the family the option would
        SELECT, not by the one the session is on (see the note beside
        ``can_state``). Every fact here comes from the registry, never from a
        list in this file.
        """
        ch = W["choices"]
        sp = api.PROBLEM_SPECS[W["problem"]]
        free = ch.get("tail_height") == "free"
        can_state = api.TAIL_HEIGHT_KEY in sp.flags
        # ...and the LABEL is not `can_state`, which is a fact about the
        # family the user is on RIGHT NOW. While the height is optimised that
        # family is the free twin, which by construction has no ``z_t_m``
        # flag — so the option read "the layout's own" in exactly the state
        # where PICKING it hands you a typed field. Measured over every
        # shell-reachable configuration (air conventional / V-tail, water):
        # the fixed twin takes ``z_t_m`` in all of them, so the label was
        # never true where it appeared. It asks the family the option would
        # SELECT (:func:`_fixed_height_is_stated`), which is what makes the
        # two separations read the same way — the user's ask.
        can_free = free or v1.option_available(ch, "tail_height", "free")
        # ...and the T-TAIL keeps the row even though it can answer neither:
        # "there is no height control here" is not the same statement as "a
        # T-tail's height IS its fin span", and the second one is the true
        # one. A layout that cannot answer a question still has to be seen
        # being asked it.
        t_tail = ch_tail_type() == "t_tail"
        if not (free or can_state or can_free or t_tail):
            return                       # this family has no such question
        with _row("height" if not water else "depth"):
            ht = ui.toggle({"fixed": ("you state it" if _fixed_height_is_stated()
                                      else "the layout's own"),
                            "free": "optimise it"},
                           value="fixed" if t_tail else ("free" if free
                                                         else "fixed"),
                           on_change=lambda e: set_choice("tail_height",
                                                          e.value)) \
                .props("dense no-caps unelevated toggle-color=primary")
            if t_tail or not can_free:
                ht.disable()
                ht.tooltip("a T-tail's height IS its fin span" if t_tail else
                           "no free-height solver for this configuration")
        if free:
            # the sentence QUOTES the design-box row, edited in another view,
            # so it lives in its own container and is redrawn from
            # ``_set_bound`` — the arm's band is in one for the same reason
            boxes["height_note"] = ui.column().classes("w-full gap-0")
            _render_height_note(water)
            with ui.row().classes("items-center gap-2"):
                ui.button("set the band", icon="tune",
                          on_click=lambda: ctx.select("wing", "box")) \
                    .props("flat dense size=sm no-caps")
        elif t_tail:
            # THE ONE LAYOUT THAT CANNOT ANSWER IT, and it was being offered
            # a field anyway: the toggle beside this said "a T-tail's height
            # IS its fin span" and then a typed box wrote ``z_t_m``, which
            # ``TailProblem.dz_for`` honours over the layout's own. So the
            # tailplane flew at whatever was typed while the fin was still
            # sized to sqrt(AR_vt · S_vt) — the T came apart, and the card
            # showed neither half moving. One question, one place: the
            # height is REPORTED here, off the same law that sizes the fin.
            g, _ = _fin_geometry()
            if g is not None:
                with ui.row().classes("items-center gap-3 w-full"):
                    widgets.readout(
                        "height", f"{abs(g.height):.3g}", "m",
                        tip="the fin's own span — a T-tail's tailplane sits "
                            "on its tip, so this IS the vertical separation")
            widgets.hint(
                "Not a field, because it is not a free number: move it by "
                "moving the fin, and the fin follows the span, the area and "
                "the separation. A conventional tail is the layout that can "
                "be mounted at a height of your choosing.")
        elif not can_state:
            pass                         # tail_height_note says what it flies
        else:
            widgets.number_field(
                "below the foil" if water else "above the wing plane",
                ch.get("tail_height_m") if ch.get("tail_height_m") is not None
                else own.get("z_t"),
                lambda e: _set_trim_number("tail_height_m", _num(e.value)),
                unit="m", step=0.05,
                tip="the VERTICAL separation the surface is flown at "
                    + ("(negative: it hangs below)" if water else ""))
            # ...and a READ-OUT of that field, so it gets its own container:
            # ``tail_height_m`` is a VALUE_KEY and may not rebuild the view
            # its own input lives in (the focus trap).
            boxes["height_warn"] = ui.column().classes("w-full gap-0")
            _render_height_warn(water)
        # ...through the same medium-aware reader the toggle above uses: the
        # note's T-tail branch is the one sentence that would still have said
        # "a T-tail's height IS its fin span" to a boat (see ch_tail_type)
        widgets.hint(v1.tail_height_note(dict(ch, tail_type=ch_tail_type())))

    def _render_height_note(water: bool):
        """The SEARCHED vertical separation's band, and where it leaves the
        one this family was measured over.

        The band is the user's — ``tail.height_row`` carries the design box's
        ``z_t_m`` row into the problem's own bounds, so widening it downwards
        searches the wider band instead of collecting refused draws. What the
        family's own band still means is where its results were MEASURED, and
        saying so is the whole content of this sentence.
        """
        box = boxes.get("height_note")
        if box is None:
            return
        box.clear()
        row = _bounds_of(api.TAIL_HEIGHT_KEY)
        bands = session.height_bands(S)
        where = "below the foil" if water else "above the wing plane"
        with box:
            widgets.hint(
                "The vertical separation is a design variable"
                + (f": the solver searches {row[0]:g} – {row[1]:g} m {where}."
                   if row else "."))
            if not (row and bands):
                return
            lo, hi = bands["measured"]
            lo, hi = min(abs(lo), abs(hi)), max(abs(lo), abs(hi))
            near = min(abs(row[0]), abs(row[1]))
            if near < bands["grid"]:
                widgets.hint(
                    f"That band reaches within {bands['grid']:g} m of the "
                    f"{'foil' if water else 'wing'} plane, where the induced "
                    f"downwash stops being grid-converged — the static "
                    f"margin of a candidate that close moves with the panel "
                    f"count by enough to flip the margin gate "
                    f"(tail.DZ_GRID_FRAC). It is searched as asked; read "
                    f"those candidates as indicative.", "warn")
            elif not (lo <= near and max(abs(row[0]), abs(row[1])) <= hi):
                widgets.hint(
                    f"That band reaches outside the {lo:g} – {hi:g} m this "
                    f"family's results were MEASURED over. It searches as "
                    f"asked — the separation is a design decision, not an "
                    f"out-of-contract input — and the solve stays "
                    f"grid-converged; it is the published comparison that "
                    f"is being extrapolated.", "warn")

    def _render_height_warn(water: bool):
        """The STATED vertical separation, against the measured band.

        Any separation is flown, including zero: the tailplane in the wing
        plane is the ordinary layout, and all three solvers were measured
        smooth through the old 0.05 b floor to zero and past it
        (``tail.height_row``). The floor that used to refuse this was a
        calibration, and it bit hardest on the small aircraft it was least
        entitled to refuse — 0.5 m of clearance is nothing on a 10 m span
        and everything on a 2 m one.
        """
        box = boxes.get("height_warn")
        if box is None:
            return
        box.clear()
        held = W["choices"].get("tail_height_m")
        bands = session.height_bands(S)
        if held is None or bands is None:
            return
        z = abs(float(held))
        lo, hi = bands["measured"]
        lo, hi = min(abs(lo), abs(hi)), max(abs(lo), abs(hi))
        plane = "foil" if water else "wing"
        with box:
            # the two sentences are about different things and only one can
            # be true at a time: how far the SOLVE can be trusted, and how
            # far the family's published numbers reach
            if z < bands["grid"]:
                widgets.hint(
                    f"{held:g} m is within {bands['grid']:g} m of the "
                    f"{plane} plane, where the induced downwash stops being "
                    f"grid-converged: the static margin moves with the panel "
                    f"count there (±0.035 at zero, against ±0.0002 at the "
                    f"measured height), which is enough to flip whether a "
                    f"design passes the margin gate at all "
                    f"(tail.DZ_GRID_FRAC). It is built and flown as stated "
                    f"— a surface in the {plane} plane is the ordinary "
                    f"layout — but read a marginal result as indicative.",
                    "warn")
            elif not lo <= z <= hi:
                widgets.hint(
                    f"{held:g} m is outside the {lo:g} – {hi:g} m band this "
                    f"family was calibrated over. It is built and solved as "
                    f"stated, and the solve is grid-converged there; what is "
                    f"extrapolated is the comparison with the published "
                    f"results.", "warn")

    def _cg_now():
        """The CG this family flies when NOBODY states one — what the field
        opens on, and what "empty" means beside it.

        Read off a built, mid-box problem (``session.family_cg``) rather than
        off the published one: the family's own CG is not one number. In air
        it follows the LAYOUT (``tail.X_CG_BY_TYPE``: aft on a conventional
        tail, FORWARD on a canard) and in water it is a fraction of the ARM
        (``hydrotail.X_CG_FRAC``), so a card that opened on the published
        problem's value showed a canard the conventional station and every
        water session half the one it flew.

        The family's, NOT the run's: with a CG typed, the run's own is that
        typed number, and a sentence saying "empty = 0.900 m" beside the 0.900
        the user just typed states the opposite of what it means.
        """
        cg = session.family_cg(S)
        return cg if cg is not None else _published_layout().get("x_cg")

    def _render_cg(water: bool):
        """Draw the CG field and what empty means, alone.

        Redrawn when the ARM moves, because in water the value it opens on is
        a fraction of that arm — and never while the CG itself is being typed
        into, which would destroy the field mid-number.
        """
        box = boxes.get("cg")
        if box is None:
            return
        box.clear()
        aft = "foil" if water else "wing"
        ch = W["choices"]
        cg = _cg_now()
        with box:
            widgets.number_field(
                "CG", widgets.shown(ch["tail_cg_m"]
                                    if ch.get("tail_cg_m") is not None
                                    else cg),
                lambda e: _set_trim_number("tail_cg_m", _num(e.value)),
                unit="m", step=0.05,
                tip="metres aft of the " + aft + " AC; NEGATIVE is forward "
                    "of it — the CG the trim solve balances about")
            widgets.hint(
                "Empty = this family's calibrated value"
                + (f", {cg:.3f} m aft of the {aft} AC"
                   if cg is not None else "")
                + (" — a fraction of the arm here, so it follows a searched "
                   "separation until you state one."
                   if water else ", which does not move with the separation.")
                + " It travels whether the separation is stated or searched.")

    def _render_arm_warn():
        """Draw the stated arm's EXTRAPOLATION note alone — see ``_render_cg``.

        The separation is completely free: any positive length is a length
        the solver will fly, because how long an aeroplane is, is a design
        decision and not an out-of-contract input (tail.L_T_MIN_M is the
        only floor, and it is there because a zero arm makes the pitch trim
        singular). What the family's own band still means is where its
        results were MEASURED, so a number outside it is honest
        extrapolation — said once, here, beside the number it is about.
        """
        box = boxes.get("arm_warn")
        if box is None:
            return
        box.clear()
        arm_box = session.published_arm_box(S)
        held = W["choices"].get("tail_arm_m")
        if arm_box is None or held is None:
            return
        with box:
            if float(held) <= 0.0:
                widgets.hint(
                    "the separation has to be a real distance — at zero arm "
                    "there is no moment to trim with and the pitch solve is "
                    "singular.", "bad")
            elif not arm_box[0] <= float(held) <= arm_box[1]:
                widgets.hint(
                    f"{held:g} m is outside the {arm_box[0]:g} – "
                    f"{arm_box[1]:g} m band this family was calibrated over. "
                    f"It will fly — the geometry is built and solved as "
                    f"stated — but the result is an extrapolation of the "
                    f"published measurements, so read the static margin it "
                    f"reports rather than trusting the family's own.", "warn")

    def _set_arm(value: str):
        """Whether the horizontal separation is SEARCHED or STATED.

        The twin selection — and the snap that keeps a stated arm inside the
        box its own family is calibrated over — lives in
        ``session.set_tail_arm``, so a preset or a test reaches the same
        state this toggle does. It does not touch the CG: that is the other
        question on this card, and it is asked in both states.
        """
        ch = W["choices"]
        if ch.get("tail_arm", "free") == value:
            return
        before = W["problem"]
        notes = session.set_tail_arm(S, value)
        if W["problem"] != before:
            ctx.log(f"solver: {W['problem']}", "info")
        for n in notes:
            ctx.log(n, "warn")
        _render_type()
        _render_box()
        _render_solver()
        ctx.refresh()

    def _set_trim_number(key: str, value):
        """One of the three numbers that place the surface (the CG, the
        horizontal separation, the vertical one).

        They are VALUE_KEYS: typing one may NOT rebuild the type view, or the
        field being typed into is destroyed mid-number. But every one of them
        moves the static margin, so the READ-OUT under them has to follow —
        in its own container, which is what ``boxes["stability"]`` is for.
        The bug this closes: a CG typed into the field moved the run and left
        the neutral point and margin beside it quoting the old one.

        The CG field is the one that draws a DERIVED value while nothing is
        stored (the family's own, :func:`_cg_now`), so the browser echoing
        that value back must not be recorded as a statement: an untouched
        card sends no ``x_cg_m`` at all, and in water a pinned CG severs the
        ``x_cg = X_CG_FRAC · l_t`` rule the family is calibrated on.

        The stated ARM is the one number here that may not be BLANK. The
        others mean "leave it to the family" empty, and the flags honour that
        (``nice_app.tail_flags`` sends them only when they are not None); the
        arm is sent whenever it is the stated one, so a cleared field reached
        ``float(None)`` and took the whole stage down with it — with the
        field it would have been retyped into gone from the page. Clearing it
        therefore keeps the held value, which is the one the family can fly.
        """
        if key == "tail_arm_m" and value is None:
            return
        if key == "tail_cg_m" and W["choices"].get(key) is None \
                and widgets.is_echo(value, _cg_now()):
            return
        set_choice(key, value)
        water = W["choices"].get("medium") == "water"
        if key == "tail_arm_m":
            # the arm moves both read-outs beside it: the margin, and — where
            # the CG is a fraction of the arm — what "empty" means for the CG
            _render_arm_warn()
            if W["choices"].get("tail_cg_m") is None:
                _render_cg(water)
        if key == "tail_height_m":
            _render_height_warn(water)
        # ...and the FIN, which is sized against the arm and stood on the
        # height: both of the numbers this handler writes. It was not in
        # this list, so a typed separation left the Vertical tail card above
        # quoting the fin of the arm before it — 0.727 m2 at 5.5 m still on
        # screen after 3.0 m was typed, where the law says 1.333.
        if key in ("tail_arm_m", "tail_height_m"):
            _render_fin_derived()
        _render_stability(water)

    def _handling_level_control():
        """HOW STABLE, as a standard rather than as a taste.

        Asked HERE because the question above it is the static margin, and
        this is the same question asked of the other five modes: the margin
        gates the pitch stiffness, and MIL-F-8785C Class I / Category B gates
        the spiral, the phugoid, the Dutch roll, the short period and the
        roll. One place, one question — "how stable must it be" — with the
        one the run has always had beside the five it did not.

        IT IS A CARD AND NOT A ROW because of what it decides. Every other
        control in this view says what the aeroplane IS; this one says what
        it must be able to DO, it is the only place the four lateral modes
        are asked about at all, and its levels cost real L/D
        (:data:`handling.MEASURED_COST_PCT`) — a reader who scrolls past a
        bare toggle has not declined it, they have not seen it.

        THE LEVELS ARE NAMED, not just numbered. The standard counts DOWN to
        its strictest rung, so "Level 3" reads like the best of three when it
        is the floor. The words come from ``handling.LEVEL_NAMES`` — the
        standard's own definitions — and the number stays beside each one,
        because the number is what the flag carries and what the report
        prints.

        Off by default, which is every published run: the levels add five
        signed constraint rows and they cost the deck that measures them.
        Only the families that HAVE a lateral deck declare the flag, so on
        the rest this card does not appear rather than offering a level that
        would be dropped (``api._declare_handling_flags``).

        WHAT TO EXPECT WHEN YOU PICK ONE. Measured over 256 draws of this
        shell's own box on `tail [free height]`, the clause that actually
        binds is the PHUGOID on essentially every design — not the spiral the
        ladder is named for, which removes one draw in three levels. The run
        reports which clause bound each candidate, so this is checkable on
        your own box rather than taken from that measurement.
        """
        sp_ = api.PROBLEM_SPECS[W["problem"]]
        if api.HANDLING_LEVEL_KEY not in sp_.flags:
            return
        from aerobo import handling as hq

        widgets.hairline()
        with widgets.group_box("Handling qualities"):
            widgets.hint(
                "The static margin above gates ONE thing: pitch stiffness. "
                "A level gates the aeroplane's five OTHER modes too — the "
                "spiral, the phugoid, the Dutch roll, the short period and "
                "the roll — as signed margins, so the optimiser is pushed "
                "out of an unacceptable design rather than stopped at it.")
            now = W["flags"].get(api.HANDLING_LEVEL_KEY)
            # the WORD first and the number with it. Both halves, always:
            # the word is the only one a reader who has not met the standard
            # can act on, and the number is what the flag and the report say.
            opts = {"off": "static margin only"}
            opts.update({str(v): f"{hq.LEVEL_NAMES[v]} (L{v})"
                         for v in api.handling_levels()})

            def _set(value):
                if value == "off":
                    W["flags"].pop(api.HANDLING_LEVEL_KEY, None)
                else:
                    W["flags"][api.HANDLING_LEVEL_KEY] = int(value)
                _render_handling_note()

            with _row("how stable"):
                ui.toggle(opts, value="off" if now is None else str(now),
                          on_change=lambda e: _set(e.value)) \
                    .props("dense no-caps unelevated toggle-color=primary")
            boxes["handling"] = ui.column().classes("w-full gap-1")
            _render_handling_note()

    def _handling_line(head: str, body: str, *, mono: bool = True,
                       tip: str = ""):
        """One `threshold — what it means` line.

        Not :func:`widgets.kv`: that renders its value in the READOUT face
        (monospace, no-wrap), which is right for a number and wrong for a
        sentence — a clause's explanation would run off the card instead of
        wrapping under it. The head keeps the monospace face where it IS a
        threshold, and the body wraps as a hint.
        """
        # ...and the body obeys the shell's own length rule. It is drawn as
        # a raw label rather than through widgets.hint (it has to flex inside
        # this row), so it splits itself: Level 3's sentence is 45 words.
        lead, rest = widgets.split_hint(body)
        with ui.row().classes("w-full items-baseline gap-2"):
            h = ui.label(head).classes("readout" if mono else "field-label") \
                .style("min-width:176px")
            ui.label(lead).classes("hint").style("flex:1 1 0;min-width:0")
            if rest:
                # no popup title: the head is on the same line, and a popup
                # that opens by repeating the row it hangs off is a line of
                # the reader's attention spent to learn nothing
                widgets.help_dot(rest)
        if tip:
            widgets.explain(h, tip, title=head)

    def _render_handling_note():
        """What the chosen level ASKS FOR, in the aeroplane's own terms.

        Redrawn on its own — the toggle that changes it is in the row above
        and rebuilding that row would take the toggle out from under the
        click that moved it.

        Every string here is read off :mod:`aerobo.handling`: the names, the
        sentences, the five thresholds and the measured cost. A shell that
        restates a threshold is a second place for the standard to live, and
        the two would disagree the first time either moved.
        """
        box = boxes.get("handling")
        if box is None:
            return
        box.clear()
        from aerobo import handling as hq

        lvl = W["flags"].get(api.HANDLING_LEVEL_KEY)
        with box:
            if lvl is None:
                widgets.hint(
                    "Nothing refuses a design for those five modes right "
                    "now, so a wing that wallows after a gust, wags in yaw "
                    "or rolls off slowly with the stick free can win the "
                    "search on L/D alone.", "warn")
                widgets.hint("What the three rungs ask for, loosest "
                             "first — which is not the order of their "
                             "numbers:")
                for v in api.handling_levels():
                    # the standard's own grades, cheapest first, each with
                    # what it costs a SEARCH — the number a reader is
                    # actually trading L/D against.
                    cost = hq.MEASURED_COST_PCT[v]
                    priced = (f"[measured {cost[1]:+.1f} to {cost[0]:+.1f} % "
                              f"of L/D]" if cost else
                              "[cost not measured for this rung]")
                    _handling_line(
                        f"Level {v} · {hq.LEVEL_NAMES[v]}",
                        f"{hq.LEVEL_MEANING[v]}  {priced}",
                        mono=False)
                return

            # ONE coercion, at the edge: a flag arrives from a widget, a
            # saved session or a URL, so it may be "1" as easily as 1 — and
            # every dict below is keyed by the int.
            lvl = hq.level_of(lvl)
            rows = hq.clause_rows(lvl)
            cost = hq.MEASURED_COST_PCT[lvl]
            with ui.row().classes("w-full items-center gap-2 no-wrap"):
                widgets.tag(f"LEVEL {lvl}", theme.ACCENT)
                ui.label(hq.LEVEL_NAMES[lvl].upper()).classes("sect-head")
            widgets.hint(hq.LEVEL_MEANING[lvl])
            # WHERE THE NUMBERS COME FROM, per rung: two of the three are a
            # citation and one is this package's own, and a card that said
            # MIL-F-8785C over all three would be citing a document for a
            # requirement it does not make.
            widgets.hint(f"{hq.level_source(lvl)}. Five signed margins join "
                         f"the static one:")
            for clause, label, meaning in rows:
                # threshold AND physics: a row that reads "dutch roll zeta
                # >= <number>" tells a reader who does not already know what
                # a Dutch roll IS nothing at all, and this card exists for
                # exactly that reader.
                _handling_line(label, meaning, tip=clause)
            if cost:
                widgets.hint(
                    f"Measured cost of this rung on two wing+tail boxes: "
                    f"{cost[1]:+.1f} to {cost[0]:+.1f} % of L/D against the "
                    f"static margin alone (24 paired seeds, "
                    f"scripts/gate_costs_a_search.py). It did not empty the "
                    f"box. The run reports which clause bound each candidate "
                    f"— on the boxes measured so far that is the phugoid, "
                    f"not the spiral.")
            else:
                widgets.hint(
                    "What this rung costs a search has NOT been measured: "
                    "its published price was measured against the standard's "
                    "Level 3, which is not what it asks any more. The run "
                    "reports which clause bound each candidate, so your own "
                    "box answers it.", "warn")

    def _render_stability(water: bool):
        """Redraw the margin read-out alone."""
        box = boxes.get("stability")
        if box is None:
            return
        box.clear()
        with box:
            _stability_note(water)

    def _stability_note(water: bool):
        """Where the NEUTRAL POINT is, and whether this CG is in front of it.

        The question the CG field cannot answer on its own, and the reason
        "the CG must be ahead of the wing AC" is the wrong rule for an
        aft-tail aircraft: the surface behind the wing drags the whole
        aircraft's neutral point AFT of the wing AC, and stability is
        x_cg < x_np — the margin, not the sign of x_cg. A canard moves x_np
        forward instead, and its calibrated CG with it.

        The LOAD is reported beside the margin and IN WORDS, because a
        signed coefficient is the one thing on this card a reader can take
        the wrong way round: a stabiliser that carries -0.038 is pushing
        DOWN, and which way it pushes decides which way its section should
        be cambered, which way its tip device pays and which way the
        elevator deflects.
        """
        st = session.pitch_stability(S)
        if st is None:
            # a family with no CG has nothing to report and says nothing;
            # one that HAS a CG and still cannot be read has to say why, or
            # the three numbers simply vanish at the CG that removed them
            why = session.stability_unevaluable(S)
            if why:
                widgets.hint(
                    f"No margin to report at this CG: {why}. The usual cause "
                    f"is the mounting below — the surface is built to push "
                    f"DOWN, so a CG far enough aft asks an inverted section "
                    f"to LIFT and the incidence that trims it runs off the "
                    f"end of the polar. Move the CG forward and the three "
                    f"numbers come back. (On the published air tail the "
                    f"boundary is about 0.8 m, and every CG that far back is "
                    f"already statically unstable.)", "warn")
            return
        aft = "foil" if water else "wing"
        with ui.row().classes("items-center gap-3 w-full"):
            widgets.readout("CG", f"{st['x_cg']:.3f}", "m",
                            tip=f"aft of the {aft} AC (negative = forward)")
            widgets.readout("neutral point", f"{st['x_np']:.3f}", "m",
                            tip="of the WHOLE aircraft, on the coupled "
                                "lift-curve slopes — the station the CG has "
                                "to stay in front of")
            widgets.readout(
                "static margin", f"{st['SM']:+.3f}", "mac",
                color=(theme.WARN if not st["accepted"] else ""),
                tip="(x_np − x_cg)/mac. Cm_alpha = −CL_alpha · SM")
            trim = session.trim_lift(S)
            if trim is not None:
                widgets.readout(
                    "carries", f"{trim['cl']:+.4f}", "CL",
                    tip="the lift coefficient the trim balance leaves this "
                        "surface — NEGATIVE is a download")
        if trim is not None:
            way = "DOWNWARDS" if trim["cl"] < 0.0 else "UPWARDS"
            name = session.second_surface_name(S) or "second surface"
            widgets.hint(
                f"The {name} pushes {way} here: CL {trim['cl']:+.4f} on "
                f"its own {trim['s_lift_m2']:.4g} m². "
                + ("So it flies its section UPSIDE DOWN — its camber works "
                   "the side the surface is actually loaded on, and the "
                   "trim incidence stops fighting it. Stage 2's ranking is "
                   "read at the lift the section sees the way it is "
                   "mounted, and the views draw it mirrored. "
                   if trim.get("inverted") else
                   "It is a lifting surface, so it is screened like one — "
                   "and mounted the RIGHT WAY UP, like the wing. ")
                + f"({trim['source']})")
            # WHY it goes that way, which is the question a reader actually
            # has when a stabiliser comes out lifting. Two authors, and the
            # section's is the one nobody expects: a REFLEXED wing section
            # (cm_ac > 0) pitches nose-UP at zero lift, so there is no
            # nose-down couple left for the tail to hold and the balance
            # asks it for up-load. Screening a wing on |Cm| can select one
            # without the reader ever deciding to build a reflexed aeroplane.
            m_ac = float(trim.get("m_ac_m3") or 0.0)
            if trim["cl"] >= 0.0 and m_ac >= 0.0:
                sec = (session.section_of(S, "main") or {}).get("name")
                widgets.hint(
                    f"It pushes UP because the wing's own section couple is "
                    f"NOSE-UP (M_ac {m_ac:+.4g} m³"
                    + (f", section {sec}" if sec else "")
                    + f"): a reflexed aerofoil holds its own pitching moment, "
                    f"so nothing is left for the {name} to hold down. Change "
                    f"the wing's section on stage 2 — one with the usual "
                    f"nose-down cm_ac puts the {name} back on a download — "
                    f"or move the CG aft, which does the same thing.", "info")
        if not st["stable"]:
            widgets.hint(
                f"UNSTABLE at the mid-box design: the CG is {st['x_cg']:.3f} "
                f"m and the neutral point is {st['x_np']:.3f} m, so the CG "
                f"is BEHIND it (SM {st['SM']:+.3f} mac, Cm_alpha > 0). Move "
                f"the CG forward, or the surface further aft / bigger — the "
                f"run scores such a design honestly and the static-margin "
                f"constraint refuses it.", "warn")
        elif not st["accepted"]:
            widgets.hint(
                f"Stable but inside the margin the run requires: SM "
                f"{st['SM']:+.3f} mac against SM_min {st['SM_min']:.2f}. The "
                f"constraint is signed, so the optimiser is pushed out of "
                f"here rather than stopped.", "warn")
        else:
            widgets.hint(
                f"Stable: the CG is {st['x_np'] - st['x_cg']:.3f} m in front "
                f"of the neutral point, SM {st['SM']:+.3f} mac against the "
                f"SM_min {st['SM_min']:.2f} the run requires. Note the "
                f"criterion is the NEUTRAL POINT, not the {aft} AC — the "
                f"surface behind the {aft} moves x_np aft of it, which is "
                f"why a CG behind the {aft} AC is still stable here. "
                f"Quoted at the mid-box design; the run reports its own.")

    def _num(value):
        """A typed number, or None for "leave it to the family"."""
        if value in (None, ""):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _published_layout() -> dict:
        """The layout this family flies when nobody states one — read off a
        BUILT problem, never restated here."""
        try:
            prob = api.PROBLEM_SPECS[W["problem"]].build({}, {}, None).problem
        except Exception:                       # noqa: BLE001 — a read-out
            return {}
        foil = getattr(prob, "foil", None) or getattr(prob, "wing_tail", prob)
        out = {}
        arm = getattr(foil, "l_t_fixed", None)
        if isinstance(arm, (int, float)):
            out["l_t"] = float(arm)
        x_cg = getattr(foil, "x_cg", None)
        if x_cg is None and hasattr(foil, "x_cg_for"):
            # a family whose CG is a FRACTION of the arm (hydrotail): with the
            # arm searched the arm to quote is the MIDDLE OF THE BOX the run
            # searches, which is the design every other read-out on this stage
            # is quoted at. 0.5 m was a stand-in that halved the water CG.
            box = _bounds_of("l_t_m")
            x_cg = foil.x_cg_for(arm if arm is not None
                                 else 0.5 * (box[0] + box[1]) if box
                                 else 1.0)
        if isinstance(x_cg, (int, float)):
            out["x_cg"] = float(x_cg)
        z = getattr(foil, "z_t_fixed", None)
        if z is None:
            # the layout's OWN height, where the family derives one rather
            # than carrying a stated field (the coupled LLT's measured
            # 0.05 b clearance, a T-tail's fin span). The field pre-fills
            # with what the run flies, so stating it starts from the truth.
            z = getattr(foil, "dz", None)
        if isinstance(z, (int, float)):
            out["z_t"] = float(z)
        return out

    def _tip_direction_note(surface: str):
        """WHICH WAY that tip device points — STATED, never asked.

        It used to be a four-entry toggle (up / down / either / follow the
        load), and then for a while it was ``follow``: the design variable
        was the cant's MAGNITUDE and the side was read off each candidate's
        own trim solution.

        It is now ``down``, with the mounting (``config.TAIL_MOUNT``), for
        the reason the mounting itself is stated — this shell has already
        answered which way this surface works. Leaving the device to read
        the trimmed load let the two answers disagree, and the disagreement
        was on screen: a surface mounted upside down to push down, wearing a
        device placed as though it lifted. Free in air and better in water;
        the numbers are on ``config.TAIL_TIP_DIRECTION``. What is left here
        is the read-out saying so.
        """
        if api.tail_winglet_direction(W["problem"]) is None:
            return
        with _row("points"):
            ui.label("downwards — the side this surface works")\
                .classes("readout")
        widgets.hint(
            f"Not a choice. The {surface} is built to push DOWN (its section "
            f"is mounted upside down for it), and a tip device pays on the "
            f"side its surface is loaded — so it points down too. The design "
            f"variable is the cant's MAGNITUDE; only the side is answered "
            f"here.")
        widgets.hint(
            "Measured against reading the side off each candidate's trimmed "
            "load, at matched cant magnitude: in AIR the load is a download "
            "everywhere in the box and the two are bit-for-bit identical at "
            "141/141 designs; in WATER the trimmed load points UP at this "
            "craft's own CG, and pointing the device down anyway wins 64/68 "
            "(+0.22 % of score on average, never worse) — under water the "
            "direction is a free-surface trade, not a lift-mirroring one.")

    def _surface_design_controls(water: bool):
        """The second surface's own design freedoms — the wing's questions.

        Three, in the wing's own order and vocabulary: its PLANFORM, its TIP
        DEVICE, its CHORD LAW. They compose the way the registry says they
        do (``api.TAIL_DESIGNS``): a tip device on this surface needs the
        surface to be designed at all, because a rectangle drawn from its
        area alone has no tip to hang one on.

        The CHORD LAW is deliberately the same control as the wing's, not a
        second one that could disagree with it: the free chord law is a
        MODIFIER of the whole design (api.MODIFIERS), and it gives every
        DESIGNED surface its own coefficient block. So this switch says what
        it really does — turn the law on for the aircraft, and this surface
        gets one of its own as soon as its planform is designed.
        """
        ch = W["choices"]
        surface = session.second_surface_name(S) or "second surface"
        held = ch_design()
        designed = held != "fixed"
        can_design = v1.option_available(ch, "tail_design", "planform") \
            or designed
        can_tip = v1.option_available(ch, "tail_design", "planform+tip") \
            or held == "planform+tip"
        why = v1.option_why(ch, v1.TAIL_DESIGN_WHY)
        widgets.hairline()
        widgets.hint(f"The {surface} as a surface of its own — the same "
                     f"three freedoms the wing above has.")

        with _row("planform"):
            pf = ui.toggle({"fixed": "published rectangle (AR 4)",
                            "planform": "designed (taper, AR, washout)"},
                           value="planform" if designed else "fixed",
                           on_change=lambda e: _set_tail_design(
                               "planform" if e.value == "planform"
                               else "fixed")) \
                .props("dense no-caps unelevated toggle-color=primary")
            if not can_design:
                pf.disable()
                pf.tooltip(why["planform"])
        widgets.hint(v1.TAIL_DESIGN_NOTES["planform" if designed else "fixed"]
                     if can_design else why["planform"])

        # ONE question about this surface's tip device, in the WING's own
        # words: what SHAPE is it? A yes/no switch asked a smaller question
        # about the same kind of fitting — the wing above is asked for a
        # shape, and "the same three freedoms the wing has" has to mean the
        # same three questions.
        tip_opts = v1.tail_tip_shapes(ch)
        tip_key = v1.tail_tip_shape_key(ch)
        with _row("tip device"):
            tp = ui.select(tip_opts,
                           value=tip_key if tip_key in tip_opts else "none",
                           on_change=lambda e: _set_tail_tip(e.value)) \
                .props("outlined dense").classes("grow min-w-0")
            # DISABLED ONLY WHERE THE REGISTRY HAS NOTHING TO SELECT. It used
            # to be greyed out whenever the planform above was still the
            # published rectangle — "design this surface's planform first" —
            # which made a freedom the registry HAS ('tail [designed tail +
            # tip device]', 'hydrofoil + elevator [designed elevator + tip
            # device]') unreachable until an unrelated toggle was found and
            # moved. It is an ORDERING, not a refusal: ``_set_tail_tip``
            # writes the composite ``tail_design`` the family is selected by,
            # so choosing a shape here designs the planform WITH it and the
            # toggle above follows. The hint below says so before the click.
            if len(tip_opts) <= 1:
                tp.disable()
                tp.tooltip(why["planform+tip"])
            elif not designed:
                tp.tooltip("choosing one designs this surface's planform "
                           "too — the toggle above follows")
        widgets.hint(v1.TAIL_TIP_SHAPE_NOTES.get(
            tip_key if tip_key in tip_opts else "none", ""))
        if len(tip_opts) > 1 and not designed:
            widgets.hint(
                "This surface is still the published rectangle, and a tip "
                "device comes with a designed planform: the device's height "
                "is a fraction of ITS semispan and the solver panels the two "
                "together (the tip-device rows live in the designed-tail "
                "branch of wingtail/hydrotail). Choosing a shape here "
                "therefore switches the planform above to “designed” as "
                "well — one click, both rows in the box.")
        _missing(tip_opts, v1.TAIL_TIP_SHAPE_LABELS,
                 v1.option_why(ch, v1.TAIL_TIP_SHAPE_WHY))
        if held == "planform+tip":
            widgets.hint(v1.TAIL_DESIGN_NOTES["planform+tip"])
            _tip_direction_note(surface)
        elif designed and not can_tip:
            widgets.hint(why["planform+tip"])
        # the LAYOUT this surface is trimmed in — where it sits, and where
        # the CG it balances about is. Not a tip-device question: it decides
        # the sign of what the surface carries whether or not it has one.
        _trim_layout_controls(water, surface)

        with _row("chord law"):
            cl = ui.toggle({"fixed": "straight taper",
                            "free": "polynomial chord law"},
                           value=ch.get("chord", "fixed"),
                           on_change=lambda e: set_choice("chord", e.value)) \
                .props("dense no-caps unelevated toggle-color=primary")
            if not v1.chord_available(ch):
                cl.disable()
                cl.tooltip("this family has no chord-law twin")
        widgets.hint(
            "The chord law is one modifier for the whole design — the same "
            "switch as the wing's above, shown here because it is one of "
            "this surface's freedoms too. "
            + (f"On, and with its planform designed, the {surface} carries "
               f"its OWN cubic law (chord_k*_t) beside the wing's."
               if ch.get("chord") == "free" and designed else
               f"On, the {surface} gets a law of its own as soon as its "
               f"planform is designed above."
               if ch.get("chord") == "free" else
               "Off: both surfaces fly straight taper."))

    def ch_design() -> str:
        held = W["choices"].get("tail_design", "fixed")
        from aerobo import api as _api
        return held if held in _api.TAIL_DESIGNS else "fixed"

    def _set_tail_design(value: str):
        """Write the composite ``tail_design`` the registry speaks.

        The two controls above are the two independent questions; this is
        the one value the family is selected by, so the mapping happens in
        exactly one place.
        """
        set_choice("tail_design", value)

    def _set_tail_tip(shape: str):
        """The second surface's tip-device SHAPE — one control, three
        answers written together.

        A shape writes the ``tail_design`` the FAMILY is selected by, the
        cant band its type narrows to, and its own blend. All three go
        through ``v1.set_tail_tip_shape`` so a preset or a test reaches the
        same state this menu does, and then through ``set_choice`` so the
        normalisation, the derived problem and the design box all follow —
        exactly the path the wing's own shape menu takes.
        """
        ch = W["choices"]
        held = ch.get("tail_design")
        v1.set_tail_tip_shape(ch, shape)
        # ...and the DIRECTION, which V3 does not ask and does not write
        # here either: it is one of the answers this shell STATES, so it
        # lives with the others in `config.TAIL_TIP_DIRECTION` and travels
        # through `config.stated_physics`. Writing it into the choices as
        # well would be the same question answered twice, and the copy that
        # lost would be invisible (_tip_direction_note).
        ch["tail_winglet_dir"] = None
        if ch["tail_design"] != held:
            # a different FAMILY: set_choice re-normalises, re-derives the
            # problem and redraws everything that quotes it
            new, ch["tail_design"] = ch["tail_design"], held
            set_choice("tail_design", new)
            return
        # same family, a different VALUE — nothing re-derives, so this is the
        # narrow redraw the type view's own handlers use
        ctx.log(f"{session.second_surface_name(S) or 'second surface'} tip "
                f"device: {v1.TAIL_TIP_SHAPE_LABELS.get(shape, shape)}",
                "info")
        _render_type()
        ctx.refresh()

    def _render_arm_note(water: bool | None = None):
        """Draw the SEARCHED arm's band sentence alone — see ``_render_cg``."""
        box = boxes.get("arm_note")
        if box is None:
            return
        if water is None:
            water = api.PROBLEM_SPECS[W["problem"]].medium == "water"
        box.clear()
        with box:
            widgets.hint(_arm_note(water))

    def _arm_note(water: bool) -> str:
        """What the SEARCHED arm's band is, and whose band it is.

        The band is the user's: the design-box row travels into the problem's
        own bounds (``api._arm_band_kwargs`` -> ``tail.arm_row``), so a wider
        row is a longer aeroplane rather than a page of refused draws. The
        family's published interval survives as what it always was — where
        the results were MEASURED — so a band outside it says so here, in the
        same words the stated arm's note uses (``_render_arm_warn``).
        """
        box = _bounds_of("l_t_m")
        where = ("aft of the main foil" if water else "aft of the wing")
        if box is None:
            return ("the arm is a design variable — the solver places the "
                    f"surface {where}.")
        lo, hi, source = box
        tail = {"user": " — the box you typed in the design-box view, not "
                        "the family's published one.",
                "mission": " — the band this mission implies, which follows "
                           "the mission until you type in it.",
                "recommended": " — MEASURED for this mission: the band the "
                               "designs that actually flew it used, always "
                               "opened far enough to hold the wing this "
                               "mission states. It follows the mission until "
                               "you type in it.",
                "section": " — narrowed by the section chosen in stage 2.",
                }.get(source, " (edit the box in the design-box view).")
        published = session.published_arm_box(S)
        if published and not (published[0] <= lo and hi <= published[1]):
            tail += (f" Outside the {published[0]:g} – {published[1]:g} m "
                     f"this family was CALIBRATED over, so read the static "
                     f"margin the run reports rather than the family's own — "
                     f"the geometry is built and flown as asked.")
        return (f"the arm is a design variable: the solver places the surface "
                f"{lo:g} – {hi:g} m {where}{tail}")

    def _bounds_of(label: str):
        """``(low, high, source)`` of the row the RUN will search.

        The EFFECTIVE box, not the family's published one: a row typed in
        the design box is what ``config.bounds_overrides`` sends to
        ``api.run``, so a read-out quoting ``default_bounds`` states an
        interval the search was told not to use — one question answered in
        two places, which is the disagreement this stage exists to prevent.
        """
        row = config.effective_bounds(S).get(label)
        if row is None:
            return None
        (lo, hi), source = row
        return [float(lo), float(hi), source]

    def _derived_panel():
        name, notes = v1.derive_problem(W["choices"])
        spec = api.PROBLEM_SPECS[name]
        with widgets.group_box("Derived solver"):
            ui.label(spec.display).classes("sect-head")
            widgets.hint(spec.description)
            with ui.row().classes("items-center gap-2 flex-wrap"):
                widgets.tag(spec.medium.upper(), theme.INK_MUTED)
                widgets.tag(f"{len(spec.param_labels)}-D", theme.ACCENT)
                if spec.is_constrained:
                    widgets.tag("CONSTRAINED", theme.WARN)
                if spec.slow:
                    widgets.tag("XFOIL · SLOW", theme.WARN)
            widgets.hint(f"twist: {v1.twist_description(name)}")
            # what this RUN flies, not what the family opens on: a chosen
            # size is exactly the case where the two differ
            size = session.flown_size(S)
            if size:
                widgets.hint(f"planform flown: b = {size[0]:.3f} m, "
                             f"S = {size[1]:.3f} m²")
            else:
                # ...and a family that SEARCHES its size flies no single
                # planform to report. The car falls here: `flown_size` is
                # None for it, so this line quoted `api.planform_size` — the
                # family's own published 1.6 m x 0.4 m² — under the heading
                # "planform flown", on a run whose b_m and S_m2 rows the
                # optimiser was about to move anywhere inside their bands.
                # The band IS the answer here, so the band is what is shown.
                eb = config.effective_bounds(S)
                rows = [(lbl, unit, eb[lbl][0]) for lbl, unit
                        in (("b_m", "m"), ("S_m2", "m²")) if lbl in eb]
                if rows:
                    widgets.hint(
                        "planform SEARCHED: "
                        + ", ".join(f"{lbl} {float(r[0]):.3g}–"
                                    f"{float(r[1]):.3g} {unit}"
                                    for lbl, unit, r in rows)
                        + " — the design box's own rows, so the planform "
                          "this run flies is part of the answer.")
                else:
                    own = api.planform_size(name)
                    if own:
                        widgets.hint(f"planform flown: b = {own[0]:.3f} m, "
                                     f"S = {own[1]:.3f} m²")
            for n in notes:
                widgets.hint(n, "warn")

    def _fix_section_point(surface: str):
        """Point this surface's section stage at the surface's own Re, ARM the
        re-screen, and go there.

        It still does NOT start the sweep. A screen at a surface's own
        Reynolds number is real XFOIL — minutes on the first visit — and
        launching that from another stage is exactly the invisible expensive
        work this shell refuses to do. What changed is what the user is handed
        on arrival: a primed button that states the price, instead of a log
        line naming a control to go and find. The decision was already taken
        here; only the cost has to be shown there.
        """
        a = session.airfoil_state(S, surface)
        a["re_source"] = "mission"
        a["override_point"] = False
        name = ("wing" if surface == "main"
                else session.second_surface_name(S) or "the second surface")
        own = session.section_conditions(S, surface).get("re")
        session.arm_rescreen(
            S, surface,
            reason=(f"stage 3 found that the {name} flies its section at a "
                    f"different Reynolds number from the one it was screened "
                    f"at, and re-pointed this stage at the surface's own"),
            re=own)
        ctx.log(f"the {name} section stage now screens at its own Reynolds "
                f"number — the re-screen is waiting there with its price", "info")
        ctx.render(session.SURFACE_STAGES[surface])
        ctx.select(session.SURFACE_STAGES[surface], "screen")

    # ------------------------------------- the box against the mission
    def _size_conflicts() -> list:
        """What this mission and these size rows cannot BOTH allow.

        Closed form and free (measured 0.8 ms on the family that provoked
        it): both cheap gates are inequalities in numbers the box already
        states, so nothing is evaluated. Never fatal — a view that cannot
        build the problem shows a card rather than breaking the page.

        A BUILD THAT RAISES IS ITSELF A CONFLICT, and is the one this card
        exists to explain: a ``ws_pa`` row typed above the mission's own
        ceiling makes ``api._size_band_kwargs`` raise before any gate runs,
        so swallowing it drew nothing at all — not the loading conflict, not
        the aspect-ratio one, not the area one, because the raise precedes
        every gate. The wing stage then said nothing until Run, where the job
        came back ``status="error"`` with this same message. It is said here
        instead, in the words the api already uses: they name the row and
        both ways out.
        """
        try:
            return api.size_box_conflicts(config.build_cfg(S))
        except (ValueError, KeyError) as exc:
            return [{"kind": "the box cannot be built", "empty": True,
                     "text": f"This design box and this mission cannot both "
                             f"be stated: {exc}",
                     # no row and no band: the api's message says what to
                     # move, and this card must never offer a press whose
                     # destination it had to guess
                     "row": None, "suggest": None, "frac": None}]
        except Exception:      # noqa: BLE001 — a view, never fatal
            return []

    def _widen_row(label: str, band) -> None:
        """Take the suggested band for a row the mission empties.

        Written through ``W["bounds"]`` — the same place a typed bound
        lands — so the row becomes a USER row and says so in its source
        column, rather than a number that appeared from nowhere.

        The RUN view is repainted too, and that is the whole reported bug and
        not a nicety: this card is rendered from ``_render_no_feasible`` as
        well as from the design box, ``ctx.refresh`` paints the shell chrome
        and no view at all, and a press from the run view therefore moved the
        row and left the same warning and the same button on screen. Nothing
        the user could see changed, so the recommendation read as broken.
        ``_relax_adopt`` — the only other writer of ``W["bounds"]`` — has
        always repainted all three.
        """
        W["bounds"][label] = [float(band[0]), float(band[1])]
        session.bounds_source(S).pop(label, None)   # now the user's row
        # ...and a row cannot be RELEASED and reached at the same time: the
        # overrides drop a released row, so the band would never reach the
        # run and the card would keep asking for the press it just got.
        W["bounds_off"] = [k for k in (W.get("bounds_off") or [])
                           if str(k) != str(label)]
        # ...nor FIXED and widened at the same time, which is the whole of
        # the reported bug. A pinned row leaves the design vector, and
        # ``config.bounds_overrides`` widens its band back down to contain
        # the pin — measured on the reported box, the freshly written
        # [93.08, 533.3] came back out as [20.0, 533.3] and the gate went on
        # judging the row at [20, 20]. So the press repainted byte-identical
        # text and the identical button, and changed nothing in the run
        # either. Taking a band for a row IS the decision to search it, so
        # the pin goes — said out loud in the log, because a value the user
        # typed is not something to drop silently.
        if str(label) in (W.get("fixed") or {}):
            was = (W["fixed"] or {}).pop(str(label))
            ctx.log(f"{label} was held fixed at {float(was):.4g} — that pin "
                    f"is released, because a row cannot be searched over a "
                    f"band and held at one value at the same time", "warn")
        _render_box()
        _render_solver()
        _render_run()
        ctx.refresh()

    def _render_size_conflicts() -> bool:
        """The pre-flight, and the reason this card exists.

        A user ran three searches on a mission whose weight had gone up and
        whose wing rows had not: at 7000 N against a 75.2 Pa ceiling off its
        own constraint diagram the wing has to be 93 m^2, the area row
        stopped at 22, and every one of the 72 draws was refused before its
        solver. What the shell said, 20 s later each time, was "no solution
        was found" — which reads as a failed search and was a box that could
        not contain an answer. Both cheap gates are inequalities in (b, S),
        so this is decidable BEFORE the run and is said here.

        A warning, never a ban: the numbers are the user's, and a box that
        looks empty to these two gates is still theirs to search.
        """
        found = _size_conflicts()
        # ...and one PRESS per band. Two gates can want the same row moved to
        # the same place (the reported box drew "set S_m2 to 93.09 – 533.3"
        # twice, from the loading gate and from the joint one), and a second
        # button that does exactly what the first did reads as a second
        # recommendation that also failed.
        offered: set = set()
        for f in found:
            widgets.hint(f["text"], "warn" if f["empty"] else "")
            band, row = f.get("suggest"), f.get("row")
            if not (f["empty"] and band and row):
                continue
            key = (str(row), round(float(band[0]), 6), round(float(band[1]), 6))
            if key in offered:
                continue
            offered.add(key)
            ui.button(f"set {row} to {band[0]:.4g} – {band[1]:.4g}",
                      icon="straighten",
                      on_click=lambda _=None, r=row, b=band:
                      _widen_row(r, b)) \
                .props("outline dense no-caps")
        # widening the box is one way out of an empty box; the other is that
        # the ceiling emptying it is not the user's. Said here as a POINTER
        # and not as a second control — the loading is asked at stage 1, and
        # a question answerable in two places is answered twice.
        if any(f["empty"] and f["kind"].startswith("wing loading")
               for f in found) and \
                session.ws_cap_source(S) == session.WS_CAP_MISSION:
            widgets.hint(
                "That ceiling is this mission's own stall/landing "
                "requirement. If you have DETERMINED your wing loading "
                "elsewhere, say so at stage 1 (“Wing loading” → “The W/S I "
                "typed”) and it becomes the ceiling instead — the box stops "
                "being empty without a row moving.")
        return bool(found)

    # ------------------------------------------------------- view: box
    def _effective_clabels() -> tuple:
        """The margins THIS configuration will report, in order.

        Not ``sp().constraint_labels``, which is the family's static
        declaration and only its DEFAULT margin set. A car family budgets
        nothing unless the user states a limit, so the same problem reports
        one margin or three depending on the Limits block above — and
        ``diagnose.infeasibility_report`` names margins BY INDEX, so a static
        list of the wrong length either leaves a margin unnamed or, worse,
        prints one margin under another's name and sends the relax planner
        after the wrong constraint.
        """
        return api.constraint_labels_of(W["problem"], W["flags"],
                                        config.mission_kwargs(S)
                                        if sp().uses_mission else None)

    def _render_box():     # noqa: PLR0915
        box = ctx.views[("wing", "box")]
        box.clear()
        # every chip in the old table has just been deleted; a reference kept
        # across the rebuild would have _retag_source write into a dead
        # element instead of the one on screen. Same for the fixed-row note's
        # container: on a problem with no editable rows it is never re-made,
        # and a handler clearing the deleted one is nicegui's "an element has
        # been deleted but is still being used".
        box_tags.clear()
        box_inputs.clear()          # the fields are about to be rebuilt
        boxes.pop("fixed_note", None)
        eff = config.effective_bounds(S)
        _, link_notes = config.section_link_rows(S)
        groups = _chord_groups(eff)
        chord_rows = {lab for labs in groups.values() for lab in labs}
        with box:
            if not eff:
                widgets.hint("This problem exposes no editable box bounds.")
                return
            off = config.released_rows(S)
            items = _box_items(eff, chord_rows)
            # ONE TABLE PER SURFACE. Every constraint on the wing in the
            # first, every constraint on the second surface in the second —
            # because that is how a reader holds them: "how big may the tail
            # be" is a question about one object, and it was answered by
            # four rows scattered through a single table of eleven between
            # the wing's twists and its chord law. The split is by label
            # (AFT_ROW), so a family that grows a row gets it filed without
            # a list here being updated.
            aft = [it for it in items if AFT_ROW.search(it[0])]
            wing_items = [it for it in items if not AFT_ROW.search(it[0])]
            aft_name = session.second_surface_name(S) if aft else None
            split = bool(aft) or _tail_span_asked()
            with widgets.group_box("Wing design box" if split
                                   else "Design box"):
                # TWO LINES, AND THE PARAGRAPHS BEHIND THE MARK. This
                # opened with about a hundred and eighty words in four
                # stacked paragraphs above the first row of a table the
                # reader had come here to set — and the two that explain the
                # switches now sit on the switches' own column headers,
                # which is where somebody wondering what one does is
                # looking.
                widgets.hint_help(
                    ("The box the optimiser searches — the WING's half; the "
                     "second surface has its own below."
                     if split else "The box the optimiser searches."),
                    "Rows in BLUE were pinned by the section chosen in "
                    "stage 2, rows in GREEN were measured for this mission, "
                    "and rows you type win over both.\n\n"
                    "\u201cReset to the solver\u2019s box\u201d gives back "
                    "the family\u2019s own published box, bit-for-bit, and "
                    "stops the measurement.\n\n"
                    "The two switches on each row — constrain and fix — are "
                    "explained on their own column headings.",
                    title="What the colours mean")
                floored = session.unpriced_cant_floor(S)
                if floored:
                    widgets.hint(
                        "A row tagged “unpriced” had its low end moved by "
                        "the shell, not by the family: nothing in this "
                        "objective prices the SIGN of "
                        + " and ".join(floored)
                        + f", so it is floored at "
                          f"{session.CANT_FLOOR_DEG:g} rather than searched "
                          f"down into the anhedral half nothing wants. "
                          f"Type the row to take the whole band back — this "
                          f"is a default, not a limit — or weight “spiral "
                          f"divergence” on the objective card and it lifts "
                          f"itself.", "warn")

                # its OWN container: the sentence quotes the value in every
                # fixed row's input, and those inputs are in this same view —
                # so the one handler that moves it (`_set_fixed_value`) can
                # redraw the sentence without rebuilding the field the digits
                # are arriving in. Left inline it read "Fixed: taper = 0.6"
                # over a field the user had just typed 0.5 into.
                boxes["fixed_note"] = ui.column().classes("w-full gap-1")
                _render_fixed_note()
                if groups:
                    widgets.hint(
                        "The chord law's own rows (chord_k…) are asked "
                        "BELOW, as the planform they draw rather than as "
                        "coefficients nobody can picture.")
                for note in link_notes:
                    widgets.hint(note)
                _box_grid(wing_items, off)
                # ...and whether the rows above and the MISSION can both be
                # satisfied at all. Drawn here as well as beside Launch
                # because this is the view that can fix it.
                _render_size_conflicts()
                _span_note()
                _render_recommendation("wing")
            if split:
                name = aft_name or session.second_surface_name(S) \
                    or "second surface"
                with widgets.group_box(f"{name.capitalize()} design box"):
                    widgets.hint(
                        f"Every constraint on the {name} — how big it may "
                        f"be, where it sits, and the shape of it where its "
                        f"planform is designed. The three states are the "
                        f"ones above: searched over a band, released, or "
                        f"fixed and out of the design vector.")
                    _box_grid(aft, off, spans=True)
                    _tail_span_note()
                    _render_recommendation("aft")
            with ui.row().classes("items-center gap-2"):
                ui.button("Reset to the solver's box", icon="restart_alt",
                          on_click=_reset_box) \
                    .props("outline dense no-caps")

            for key in groups:
                _chord_law_panel(key)
            # the trend belongs to the DRAWING where there is one (a chord-law
            # panel asks it beside the band it clips); with no law there is no
            # such panel, and it is asked here with the other limits
            _pair_area_controls()
            _chord_limit_controls(owns_trend=not groups)
            _tail_limit_controls()
            # NO _car_limit_controls() here any more — the car's drag ceiling
            # and downforce floor are drawn under the Maximise select on the
            # Wing type card, which is where the objective that needs them is
            # chosen. One home, moved rather than copied.
            _plate_chord_controls()

            # derived FROM the box above, so it follows every input in this
            # view — and therefore lives in a container of its own, or a
            # typed bound would leave it quoting the box that was replaced
            boxes["geo"] = ui.column().classes("w-full gap-0")
        _render_derived_geo()

    def _render_derived_geo():
        """The planform the current box implies, redrawn on its own.

        ...and with it the measured dihedral, which is taken on the same box
        CENTRE and therefore moves with every row of it. Hooked HERE rather
        than in each handler because this is the call every box edit already
        makes: a read-out that follows the box, refreshed from the one place
        that knows the box moved, cannot be left behind by a handler somebody
        adds tomorrow (``test_v3_act_render_fixpoint``).
        """
        _render_cant_note()
        # ...and the MARGIN, for the same reason and the same way. It is
        # quoted at the mid-box design ("Quoted at the mid-box design; the
        # run reports its own"), so pinning a row or moving a bound moves
        # it — and it was left behind by every one of those edits. Reachable
        # since the shell started opening WITH a second surface, which is
        # what draws the card it lives on.
        water = W["choices"].get("medium") == "water"
        if boxes.get("stability") is not None:
            _render_stability(water)
        # ...and the two read-outs of the SECOND SURFACE that follow the same
        # box: the fin the sizing law implies (its area, height and chord all
        # follow the span and the arm) and what an EMPTY CG field means (the
        # family's calibrated value, which moves with the aspect-ratio
        # limits). Neither was reachable while the shell opened without a
        # tail, because nothing drew the cards they live on; both were left
        # behind by a span or a limit typed two cards above them.
        if boxes.get("fin_derived") is not None:
            _render_fin_derived()
        if boxes.get("cg") is not None and W["choices"].get("tail_cg_m") is None:
            _render_cg(water)
        holder = boxes.get("geo")
        if holder is None:
            return
        holder.clear()
        geo = v1.geometry_summary(
            W["problem"],
            {k: list(v[0]) for k, v in config.effective_bounds(S).items()},
            # the CHORD LAW alone, so the rows describe the shape this
            # session carries rather than the family's published one. NOT
            # the whole flag dict: this panel builds the problem to read its
            # size, and a chosen section that cannot be loaded would take
            # the entire design box down with it (the same guard
            # ``_chord_baseline`` documents one panel above).
            {api.CHORD_LAW_KEY: _chord_law()},
            # ...and the AREA this mission states, for the modes where the
            # span is a row and the area is not (S = W/(W/S)): the family's
            # published planform is not what those runs fly, so the panel is
            # given the number rather than left to invent one.
            area_m2=session.reference_area(S))
        if not geo:
            return
        with holder, widgets.group_box("Derived geometry"):
            for key, value in geo:
                widgets.kv(key, value)

    def _box_items(eff, chord_rows):
        """The design box's rows, SPAN FIRST where the family searches one.

        A row here is a BAND on a design variable, so there is a span row
        exactly when the span is searched — the wing-loading mode, the free
        planform, the car families. With the planform fixed the span is not
        searched at all, and the honest place for it is then the size card
        (one number, in metres), not a low/high pair pretending to be a band
        nobody is searching over.

        A tandem pair searches a span PER WING (``b_m`` and ``b_rear_m``), so
        "span first" means both of them, in wing order: they are the same
        question asked twice, and reading them apart — front span at the top,
        rear span three rows down among the twists — hid the one comparison
        the pair is for.
        """
        items = [(k, v) for k, v in eff.items() if k not in chord_rows]
        spans = [it for it in items if it[0] in SPAN_ROWS]
        spans.sort(key=lambda it: SPAN_ROWS.index(it[0]))
        rest = [it for it in items if it[0] not in SPAN_ROWS]
        return spans + rest

    def _span_note():
        """What the span rows mean in the state they are in."""
        eff = config.effective_bounds(S)
        rows = [r for r in session.span_rows(S) if r in eff]
        if not rows:
            return
        if not session.span_is_searched(S):
            # the free planform and the car families: b_m is an ordinary
            # design-box row there, searched beside an area (or against a
            # fixed reference one), and it says so where the row is
            return
        bands = {r: session.wing_ar_band(S, r) for r in rows}
        pair = len(rows) > 1
        opening = (
            f"{' and '.join(rows)} are the SPANS, one per wing, and they are "
            f"design variables"
            if pair else f"{rows[0]} is the SPAN, and it is a design variable")
        widgets.hint(
            opening
            + ": the area is not searched beside "
            + ("them" if pair else "it")
            + " — it follows the mission's W/S through the weight loop. "
            + ("A pair's two wings need not be the same width, and how far "
               "the rear one reaches decides how much of the front wing's "
               "downwash it sits in — so each has a band of its own. "
               if pair else
               "This row is the band it is searched in. ")
            + "The planform menu in the type view is where the mode itself is "
            "switched off, back to a span you choose."
            + "".join(f" {r}: AR {b[0]:.3g}–{b[1]:.3g}."
                      for r, b in bands.items() if b))
        # a span band and a wing loading can disagree, and the aspect ratio is
        # where that shows up: b and S are not independent once W/S is stated,
        # so a band the solvers are not valid over is a refused RUN, not a
        # narrower search (sizing.check_ar — the penalty contract). Asked per
        # WING, on that wing's own share of the area: check_ar is called once
        # per surface, so one wing can leave the band while the other holds.
        size = session.flown_size(S)
        if not size:
            return
        lo_ar, hi_ar = api.PLANFORM_AR_LIMITS
        for row, band in bands.items():
            if not band or (band[0] >= lo_ar and band[1] <= hi_ar):
                continue
            area = float(size[1]) * session.wing_area_share(S, row)
            widgets.hint(
                f"{row} is AR {band[0]:.3g}–{band[1]:.3g}, and these solvers "
                f"are honest over {lo_ar:g}–{hi_ar:g} — outside it a candidate "
                f"is refused, not scored. On that surface's S = {area:.4g} m² "
                f"the valid spans are {(lo_ar * area) ** 0.5:.3g}–"
                f"{(hi_ar * area) ** 0.5:.3g} m; a narrower wing than that "
                f"needs a higher wing loading (stage 1), not a smaller span.",
                "warn")

    def _box_grid(items, off, spans: bool = False):
        """The design box's table: one row per bound, in the order the
        family declares them.

        ``spans`` is True only for the table that OWNS the second surface's
        span band — the aft design box, where that row leads. A chord-law
        panel draws its own coefficients through this same grid and the wing
        table draws the wing's rows through it, and drawing the band in every
        caller put the row on screen three times.

        Three states per row, because they are three different statements
        about one design variable: SEARCHED over a band (the published
        behaviour), released (still searched, but nothing this session says
        constrains it), and FIXED — decided, and taken out of the design
        vector entirely (``config.fixed_rows``).
        """
        fixed = config.fixed_rows(S)
        # ...and the SECOND surface's span, which is a band on a length like
        # every row here — so it is a row of that surface's own table, and
        # it leads it, the way the wing's span leads the wing's
        tail_span = spans and _tail_span_asked()
        n_span = sum(1 for lab, _ in items if lab in SPAN_ROWS)
        with ui.element("div").classes("w-full").style(
                "display:grid;grid-template-columns:1.6fr auto auto 1fr "
                "1fr auto;gap:4px 10px;align-items:center"):
            #: what each column ASKS, for the two that are decisions and
            #: not read-outs. They were six bare words over a table whose
            #: switches change what the optimiser is allowed to see.
            HEAD_HELP = {
                "constrain": (
                    "ON: the row is searched inside the band you type.\n\n"
                    "OFF: the variable is still DESIGNED — the solver's own "
                    "published box stands and any pin stage 2 put on it is "
                    "released — but nothing this session constrains it. "
                    "Your numbers are kept and come back with the switch."),
                "fix": (
                    "Fix a row you have already decided. The optimiser "
                    "never sees the variable, the search is one dimension "
                    "smaller, and everything the run reports is still the "
                    "whole wing.\n\n"
                    "A fixed row is NOT a band of width zero — that breaks "
                    "the samplers and is refused."),
                "source": (
                    "Who set this band. \"default\" is the family's own "
                    "published bound; \"section\" is pinned by the section "
                    "chosen in stage 2; \"mission\" is narrowed by what "
                    "stage 1 states; \"recommended\" was measured for this "
                    "mission; \"user\" is your number, and it wins over "
                    "the rest.\n\n"
                    "\"released\" means the row is not constrained here at "
                    "all. \"unpriced\" means the shell floored it because "
                    "nothing in this objective prices the sign."),
            }
            for head in ("parameter", "constrain", "fix", "low", "high",
                         "source"):
                if head in HEAD_HELP:
                    with ui.row().classes("items-center gap-1 no-wrap"):
                        ui.label(head).classes("readout-label")
                        widgets.help_dot(HEAD_HELP[head],
                                         title=f"the {head} column")
                else:
                    ui.label(head).classes("readout-label")
            if tail_span and not n_span:
                # the ordinary case now that the tables are split: no span
                # row of this surface's own is in the design vector, and the
                # band leads its table
                _tail_span_row()
            for idx, (label, (row, source)) in enumerate(items):
                _box_row(label, row, source, label not in off,
                         fixed.get(label))
                if tail_span and n_span and idx + 1 == n_span:
                    _tail_span_row()

    def _box_row(label: str, row, source: str, held: bool,
                 fixed_at: float | None = None):
        # NAMED, not just coded. Every card upstream calls this row by what
        # it IS — "the separation", "the tail area" — and the box called it
        # ``l_t_m``, so the one place the band is set was the one place its
        # own name did not appear: a user sent here by "set the band" read
        # the table, found no separation in it, and concluded the row was
        # missing. The api label stays underneath, because that is the
        # column name in the evaluations log and in every export.
        help_text = v1.param_help(label)
        name = help_text[0] if help_text and help_text[0] else label
        with ui.column().classes("gap-0 min-w-0"):
            with ui.row().classes("items-center gap-1 no-wrap"):
                ui.label(name).classes("readout")
                # THE ROW'S OWN EXPLANATION, VISIBLY. ``param_help`` has
                # carried a written paragraph per row all along and the only
                # way to reach it was to hover the name — an affordance
                # nothing on screen announced, on the one table where
                # setting a band wrong returns "no solution".
                if help_text and help_text[1]:
                    widgets.help_dot(help_text[1], title=name)
            if name != label:
                ui.label(label).classes("readout-label")
        # NO SECOND COPY ON HOVER. This paragraph is behind the ? beside the
        # name, one line above; a tooltip carrying the same words is the
        # same answer in a channel nothing on screen announces, and it was
        # the longest text on this view (158 words on z_t_m).
        # the SPAN of the wing-loading mode cannot be released: the mode
        # exists to search a span the user bounds, and the family's published
        # band (6–40 m on the 10 m trim wing) is nobody's opinion about the
        # wing on screen. Switching the mode off is the planform menu's job,
        # and the tooltip says so rather than offering a switch that would
        # quietly search another band.
        locked = label in SPAN_ROWS and session.span_is_searched(S)
        pinned = fixed_at is not None
        sw = ui.switch(value=(held or locked) and not pinned,
                       on_change=lambda e, l=label:
                       _set_row_on(l, bool(e.value))) \
            .props("dense") \
            .tooltip("fixed — no band to constrain" if pinned else
                     "the mode searches this span" if locked else
                     "off: the published box is searched")
        if locked or pinned:
            sw.disable()
        # ...and the row's third state. NOT a bound of width zero: the api
        # refuses one, because scipy's Sobol engine raises on it and
        # constrained BO quietly degrades to random draws while still
        # calling itself BO. Fixing takes the variable OUT of the design
        # vector (api.RunConfig.pinned), so the search is a smaller search
        # and every number it reports is still the whole wing.
        fix = ui.switch(value=pinned,
                        on_change=lambda e, l=label:
                        _set_row_fixed(l, bool(e.value))) \
            .props("dense color=warning") \
            .tooltip("hold it at the value you type"
                     if not pinned else
                     "give it back to the optimiser")
        if locked:
            # the wing-loading mode exists to SEARCH the span; holding it
            # fixed is the fixed-planform mode, and that is the planform
            # menu's question rather than a switch buried in a table
            fix.disable()
            fix.tooltip("this mode searches the span")
        # NO Quasar display format on these inputs: nicegui posts the
        # rendered text back, so a "%.4g" would silently shrink the CST box
        # (the V1 bounds editor's bug). The value is instead ROUNDED for
        # display and the same rounding is recognised coming back
        # (_set_bound), so an untouched row keeps the solver's own exact
        # bound.
        if pinned:
            # one field, in the LOW column, and the high column says what
            # happened: a fixed row has no band, and drawing it as one (two
            # equal numbers) is exactly the picture the api refuses
            ui.number(value=widgets.shown(fixed_at, BOX_DIGITS), step=0.01,
                      on_change=lambda e, l=label:
                      _set_fixed_value(l, e.value)) \
                .props("outlined dense hide-bottom-space") \
                .classes("w-full")
            ui.label("not searched").classes("readout-label")
        else:
            pair_of_inputs = []
            for idx in (0, 1):
                num = ui.number(value=widgets.shown(row[idx], BOX_DIGITS),
                                step=0.01,
                                on_change=lambda e, l=label, i=idx:
                                _set_bound(l, i, e.value)) \
                    .props("outlined dense hide-bottom-space") \
                    .classes("w-full")
                if not held:
                    num.disable()
                pair_of_inputs.append(num)
            # KEPT, so a control that CLIPS this row — the aspect-ratio limit,
            # the pair's two areas — can write the new numbers into the fields
            # already on screen. The alternative is `_render_box`, which
            # rebuilds the table and takes the input the digits are arriving
            # in with it (the focus trap this shell keeps closing).
            box_inputs[label] = pair_of_inputs
        if pinned:
            box_tags[label] = widgets.tag("fixed", theme.WARN)
            return
        box_tags[label] = widgets.tag(source, _SOURCE_COLOR.get(
            source, theme.INK_FAINT))

    # ------------------------------------------ the second surface's span
    def _tail_span_asked() -> bool:
        """Can this family be handed a band on the second surface's span?

        Asked of the REGISTRY (the flags the problem accepts) and of the
        CONFIGURATION (whether a second surface was selected at all), never
        of a problem name — the row exists exactly where ``TailLimits`` can
        reach the build.
        """
        return (all(k in sp().flags for k in TAIL_SPAN_ROW)
                and session.second_surface_name(S) is not None)

    def _tail_span_row():
        """The second surface's SPAN, drawn as a row of the design box.

        It is not a design-vector row: the solver searches that surface as
        an area (``S_t_m2``) and, where its planform is designed, an aspect
        ratio (``AR_t``). But b = sqrt(AR*S) is exact both ways, so a low
        and a high typed here NARROW the box the sampler draws from
        (``tail.TailLimits.narrow``) rather than only refusing the draws
        that break them — which is what every other row of this table does.
        It is therefore a row of this table, one line under ``b_m``: "the
        tail came out wider than the wing" is a comparison between two spans
        and it was being asked two panels apart.

        The FIX column is disabled and says why. A span is not a variable on
        this surface, so there is nothing to take out of the design vector;
        the rows that can be fixed are the area and the aspect ratio it
        follows from, and they are in this same table.
        """
        lo_key, hi_key = TAIL_SPAN_ROW
        name = session.second_surface_name(S) or "second surface"
        held = {k: W["flags"].get(k) for k in TAIL_SPAN_ROW}
        on = any(v is not None for v in held.values())
        with ui.column().classes("gap-0 min-w-0"):
            with ui.row().classes("items-center gap-1 no-wrap"):
                ui.label(f"{name} span").classes("readout")
                # the mark every other row of this table carries, for the
                # one row that is not a design variable — which is exactly
                # the row whose explanation a reader needs most
                widgets.help_dot(
                    f"How wide the {name} may be, in metres.\n\n"
                    f"The solver searches this surface as an AREA and an "
                    f"aspect ratio; a band here converts to both exactly "
                    f"and narrows what the sampler draws, rather than only "
                    f"refusing the draws that break it.\n\n"
                    f"So there is nothing to FIX here: the rows that can be "
                    f"held are S_t_m2 and AR_t, in this same table.",
                    title=f"the {name} span")
            ui.label(f"{lo_key} / {hi_key}").classes("readout-label")
        ui.switch(value=on,
                  on_change=lambda e: _set_tail_span_on(bool(e.value))) \
            .props("dense") \
            .tooltip("off: the published box is searched"
                     if not on else
                     "held inside the band beside it")
        fix = ui.switch(value=False).props("dense color=warning")
        fix.disable()
        fix.tooltip("a span is not a variable here")
        for key in TAIL_SPAN_ROW:
            num = ui.number(
                value=(widgets.shown(held[key]) if held[key] is not None
                       else None),
                step=0.05,
                on_change=lambda e, k=key: _set_tail_limit(k, e.value)) \
                .props("outlined dense hide-bottom-space") \
                .classes("w-full")
            if not on:
                num.disable()
        widgets.tag("user" if on else "default",
                    theme.WARN if on else theme.INK_FAINT)

    def _tail_span_of_draw(x, labels) -> float | None:
        """The span the second surface would be DRAWN at, for one design.

        ``b = sqrt(AR_t * S_t)`` — the convention ``session.surface_geometry``
        and ``tail.TailLimits.narrow`` both use, so a band measured here is a
        band the row it is written into really narrows. ``AR_t`` is only a
        design variable where the surface's planform is designed; elsewhere
        it is the published fixed one, exactly as ``surface_geometry`` falls
        back.
        """
        try:
            s_t = float(x[labels.index("S_t_m2")])
        except (ValueError, IndexError, TypeError):
            return None
        if not s_t > 0.0:
            return None
        if "AR_t" in labels:
            ar_t = float(x[labels.index("AR_t")])
        else:
            from aerobo.tail import TAIL_AR

            ar_t = float(TAIL_AR)
        return (ar_t * s_t) ** 0.5 if ar_t > 0.0 else None

    def _tail_span_is_the_user_s() -> bool:
        """Did a PERSON set the second surface's span band?

        The flags alone cannot answer it once the shell is allowed to write
        them, which is why the measurement records itself in
        ``bounds_source`` under ``session.TAIL_SPAN_KEY`` — the same place
        every other measured row says so, and the same place the mission
        reads to take them all back.
        """
        held = [W["flags"].get(k) for k in TAIL_SPAN_ROW]
        if not all(v is not None for v in held):
            return False
        return session.bounds_source(S).get(session.TAIL_SPAN_KEY) \
            != "recommended"

    def _tail_span_recommendation(got: dict):
        """A band on the second surface's SPAN, from the best draws.

        The one quantity this card could not propose. ``recommend_box``
        returns bands per DESIGN-VECTOR ROW, and the tail span is not one:
        the solver searches that surface as an area and (where its planform
        is designed) an aspect ratio, and the span is a FLAG PAIR
        (``TAIL_SPAN_ROW``) narrowed through ``tail.TailLimits``. A band
        built from the corners of the two rows it derives from is far wider
        than the span the best designs actually use — measured on the
        reference tail, sqrt of the corner rows gives 1.00-3.90 m where the
        kept designs live in a fraction of that.

        So it is measured on the POINTS (``got["best_x"]``), padded the same
        way ``recommend.box_around`` pads a row, and offered only where it
        genuinely narrows what the row offers today.
        """
        if not _tail_span_asked():
            return None
        pts = got.get("best_x") or []
        labels = list(got.get("labels") or [])
        spans = [b for b in (_tail_span_of_draw(x, labels) for x in pts)
                 if b is not None]
        if len(spans) < 2:
            return None
        from aerobo.recommend import PAD_FRAC

        lo, hi = min(spans), max(spans)
        pad = PAD_FRAC * max(hi - lo, 1e-9)
        lo, hi = max(0.0, lo - pad), hi + pad
        # ...and only where it narrows. The baseline is the band the row
        # holds today where it is switched on, and the band the CURRENT box
        # implies where it is not — RECOMMEND_MIN_SHRINK has no meaning
        # against an absent row.
        held = [W["flags"].get(k) for k in TAIL_SPAN_ROW]
        if all(v is not None for v in held):
            was_lo, was_hi = float(held[0]), float(held[1])
        else:
            eff = config.effective_bounds(S)
            s_row, ar_row = eff.get("S_t_m2"), eff.get("AR_t")
            if s_row is None:
                return None
            s_lo, s_hi = float(s_row[0][0]), float(s_row[0][1])
            if ar_row is not None:
                a_lo, a_hi = float(ar_row[0][0]), float(ar_row[0][1])
            else:
                from aerobo.tail import TAIL_AR

                a_lo = a_hi = float(TAIL_AR)
            was_lo, was_hi = (a_lo * s_lo) ** 0.5, (a_hi * s_hi) ** 0.5
        was = was_hi - was_lo
        if not (was > 0.0 and (hi - lo) / was <= 1.0 - RECOMMEND_MIN_SHRINK):
            return None
        return round(lo, 4), round(hi, 4)

    def _set_tail_span_on(on: bool):
        """Switch the band on or off. BOTH ends together: a row of this
        table is a band, and half of one drawn in a two-column grid is the
        picture that made the pair unreadable in its old card. Off pops the
        flags, which is the solver's published box bit-for-bit."""
        for key in TAIL_SPAN_ROW:
            if on:
                W["flags"][key] = _tail_limit_default(key)
            else:
                W["flags"].pop(key, None)
        _render_box()
        ctx.log((session.second_surface_name(S) or "second surface")
                + " span: " + ("constrained" if on else "off"), "info")
        ctx.refresh()

    def _tail_span_note():
        """What the row means where it is, and where the surface is NOW."""
        if not _tail_span_asked():
            return
        name = session.second_surface_name(S) or "second surface"
        geo = session.surface_geometry(S, "tail")
        held = [W["flags"].get(k) for k in TAIL_SPAN_ROW]
        widgets.hint(
            f"The {name} span is a LIMIT, not a design variable: the box "
            f"searches that surface as an area (and an aspect ratio where "
            f"its planform is designed), and a band typed on the row above "
            f"converts to both exactly — so it narrows what is drawn rather "
            f"than only refusing it."
            + (f" Mid-box now: {geo['span']:.3g} m." if geo else ""))
        size = session.flown_size(S)
        if all(v is not None for v in held) and size:
            b = float(size[0])
            if float(held[1]) > b:
                widgets.hint(
                    f"That maximum ({float(held[1]):.3g} m) is wider than "
                    f"the wing ({b:.3g} m).", "warn")

    # -------------------------------------------------- the chord law's box
    def _chord_groups(eff: dict) -> dict:
        """``{group key: [chord_k… in coefficient order]}`` of ``eff``.

        One entry per DESIGNED SURFACE (CHORD_GROUPS), because the law is a
        property of a surface: a tandem carries one per wing and a designed
        tail one of its own, and a single panel over all of them would ask
        about a planform that does not exist.
        """
        out: dict = {}
        for label in eff:
            m = CHORD_ROW.fullmatch(label)
            if m and (m.group(1) or "", m.group(3) or "") in CHORD_GROUPS:
                out.setdefault((m.group(1) or "", m.group(3) or ""),
                               []).append(label)
        return {k: sorted(v, key=lambda s: int(CHORD_ROW.fullmatch(s)
                                               .group(2)))
                for k, v in out.items()}

    def _mid(label: str, eff: dict):
        """Mid-point of a design-box row, or None where there is no such row.

        A baseline quantity the law is READ against (the taper it multiplies,
        the tail area it is drawn on) is usually itself a design variable, so
        the honest single number is the middle of the box the run searches —
        the same box, not the family's published one.
        """
        row = eff.get(label)
        return None if row is None else 0.5 * (float(row[0][0])
                                               + float(row[0][1]))

    def _chord_baseline(key, eff: dict):
        """``(taper, root chord [m], semi-span [m])`` of the surface a law
        reshapes. The two lengths are ``None`` where this shell does not know
        the surface's size — a tandem's wings, or an aircraft whose span and
        area are themselves searched — and the panel then speaks in chords
        rather than in metres.
        """
        lam = _mid(CHORD_GROUPS[key][1], eff)
        lam = 1.0 if lam is None else float(lam)
        span = area = None
        if key == ("", ""):
            # the same two-step the derived-solver card uses: a size this
            # session chose, else the family's own. NOT built with the run's
            # flags — that would rebuild the problem with the chosen section
            # in it, and a section that cannot be loaded would take the whole
            # design box down with it
            size = session.flown_size(S) or api.planform_size(W["problem"])
            if size:
                span, area = float(size[0]), float(size[1])
        elif key == ("", "_t"):
            area, ar = _mid("S_t_m2", eff), _mid("AR_t", eff)
            if area and ar and area > 0.0 and ar > 0.0:
                span = (ar * area) ** 0.5
        if not span:
            return lam, None, None
        return lam, 2.0 * area / (span * (1.0 + lam)), 0.5 * span

    def _chord_trend() -> str:
        """The chord TREND this session asks for.

        The default is ``config.CHORD_TREND`` — "the root is the largest
        chord" — and NOT the absence of the flag: the shell opens on the
        constraint, and `config.flags` sends the same answer, so the band
        drawn here and the box searched cannot disagree about it.
        """
        return W["flags"].get(api.CHORD_TREND_KEY) or config.CHORD_TREND

    def _chord_law() -> str:
        """Which SHAPE this session's chord coefficients describe."""
        from aerobo import geometry

        return (W["flags"].get(api.CHORD_LAW_KEY)
                or geometry.DEFAULT_CHORD_LAW)

    def _chord_reach(key, labels, eff: dict, trend: str = "free",
                     limits=None):
        """``(ChordReach, root chord, semi-span)`` of one law's rows.

        No ``trend`` and no ``limits`` is the BOX's own reach — what the
        deviation field asks for and what ``chord_bound_for_dev`` inverts,
        neither of which knows about either. The live ones are passed when
        the band is DRAWN, because a law they forbid never flies, so it has
        no business in the picture of what this box buys.
        """
        from aerobo import geometry

        lam, c_root, semi = _chord_baseline(key, eff)
        rows = [list(eff[label][0]) for label in labels]
        return geometry.chord_reach(rows, lam, trend=trend, limits=limits,
                                    c_root=c_root, semi=semi,
                                    law=_chord_law()), c_root, semi

    def _live_limits(c_root):
        """``(the ChordLimits this panel may DRAW, what it cannot)``.

        A limit in metres is measured against a length, and this shell knows
        the surface's root chord for some laws and not others — a tandem's
        wings are sized inside the solver, and nothing here can put a metre
        on them. So either the panel draws every live limit, or it draws the
        trend alone and NAMES the ones it left out. It never draws a band
        that quietly ignores one: a picture that contains planforms the run
        refuses is the thing this whole panel exists to stop.
        """
        from aerobo import geometry

        lim = api.chord_limits_of(W["flags"])
        if lim is None:
            return None, ""
        missing = [name for name, value in (
            ("minimum chord", lim.c_min_m), ("maximum chord", lim.c_max_m),
            ("max chord rate", lim.rate_max_deg)) if value is not None]
        if c_root or not missing:
            return lim, ""
        return ((geometry.ChordLimits(trend=lim.trend)
                 if lim.trend != "free" else None), ", ".join(missing))

    def _chord_law_panel(key):
        """The chord law asked as a PLANFORM.

        The coefficient rows are the right variables for the solver and the
        wrong ones for a person — a box on ``chord_k2`` states nothing anyone
        can picture, and widening it says nothing about the wing it buys. So
        the question is asked once, in the drawing: how far may the law bend
        the chord away from the straight taper it starts from. That single
        number IS the coefficient box (geometry.chord_bound_for_dev), the
        band it reaches is drawn underneath it, and the coefficients stay one
        disclosure away for anyone who wants them.

        The chord's TREND is asked here too — which end of the surface is the
        wide one is a question about this same planform, and the band drawn
        below answers it in the drawing. It is one statement over the whole
        design (one ChordLimits reaches every wing of a problem), so the
        first law's panel asks it and the others quote it.
        """
        eff = config.effective_bounds(S)
        labels = _chord_groups(eff).get(key) or []
        if not labels:
            return
        # the BOX's own deviation, trend or no trend: it is what the field
        # asks for and what _set_chord_dev inverts (geometry.chord_bound_for_dev
        # knows nothing about a trend), so reading it off the trend-clipped
        # reach would make the number typed in and the number shown disagree
        reach, _, _ = _chord_reach(key, labels, eff)
        with widgets.group_box(f"Chord law · {CHORD_GROUPS[key][0]}"):
            widgets.hint(CHORD_LAW_NOTE)
            _chord_law_control(key)
            with _row("may bend the chord"):
                ui.number(value=round(100.0 * reach.dev_max, 1), step=5,
                          on_change=lambda e, k=key:
                          _set_chord_dev(k, e.value)) \
                    .props("outlined dense hide-bottom-space") \
                    .classes("w-24 shrink-0")
                ui.label("% away from straight taper").classes("field-unit")
            _chord_trend_control(key)
            # TWO containers, and which redraws which is the whole trap
            # (VALUE_KEYS): the BAND follows either input, so it may never
            # hold one; the coefficient rows follow the deviation field only,
            # because a row that redrew itself would swallow the number being
            # typed into it.
            boxes[f"law_{key}"] = ui.column().classes("w-full gap-1")
            with ui.expansion("the coefficients themselves").classes(
                    "w-full").props("dense"):
                widgets.hint(
                    "The rows the design vector actually carries. Typing one "
                    "by hand is allowed — the band above follows — and an "
                    "asymmetric box is a perfectly good answer; the question "
                    "above just cannot be written as one.")
                boxes[f"coef_{key}"] = ui.column().classes("w-full gap-0")
        _render_chord_law(key)
        _render_chord_rows(key)

    def _chord_law_control(key):
        """WHICH SHAPE the coefficients describe.

        The question the panel below could not ask: the band, the deviation
        field and the coefficients all describe how FAR the law may bend the
        chord, and none of them says what it bends it INTO. A cubic that
        fills the tip in also lifts the root chord; an interior-only law
        cannot touch either end; a cranked law draws straight panels. Those
        are different design decisions, not different amounts of one.

        Asked ONCE for the whole design, like the trend beside it: one law
        reaches every designed surface (api.CHORD_LAW_KEY is a single flag),
        so the first panel asks it and the others quote it.
        """
        from aerobo import geometry

        if api.CHORD_LAW_KEY not in sp().flags:
            return
        law = _chord_law()
        if key != _trend_owner():
            widgets.hint(
                f"Law: {geometry.CHORD_LAW_NAMES[law]} — one statement for "
                f"the whole design, asked with the "
                f"{CHORD_GROUPS[_trend_owner()][0]}'s law.")
            return
        with _row("law"):
            ui.select(dict(geometry.CHORD_LAW_NAMES), value=law,
                      on_change=lambda e: _set_chord_law(e.value)) \
                .props("outlined dense").classes("grow min-w-0")
        widgets.hint(geometry.CHORD_LAW_NOTES[law])
        n = len(_chord_groups(config.effective_bounds(S)).get(key) or [])
        if law == "elliptic":
            widgets.hint(
                f"It is ONE number, so the design vector is shorter than the "
                f"cubic's: this surface searches {n} chord row, not 3.",
                "warn")

    def _set_chord_law(law: str):
        """Choose the chord law. The coefficient ROWS change with it — a law
        with fewer parameters searches fewer of them, and every law's box
        means something of its own — so this session's own numbers on those
        rows go with the change, exactly as a family change drops them."""
        from aerobo import geometry

        law = str(law)
        if law == _chord_law():
            return
        if law == geometry.DEFAULT_CHORD_LAW:
            W["flags"].pop(api.CHORD_LAW_KEY, None)
        else:
            W["flags"][api.CHORD_LAW_KEY] = law
        rows = [k for k in list(W["bounds"]) if config._is_chord_row(k)]
        for k in rows:
            W["bounds"].pop(k, None)
        W["bounds_off"] = [k for k in (W.get("bounds_off") or [])
                           if not config._is_chord_row(str(k))]
        W["fixed"] = {k: v for k, v in (W.get("fixed") or {}).items()
                      if not config._is_chord_row(str(k))}
        ctx.log(f"chord law: {geometry.CHORD_LAW_NAMES[law]}"
                + (" — the coefficient rows are the new law's, so any band "
                   "typed for the old one is dropped" if rows else ""), "ok")
        _render_box()
        _render_type()
        ctx.refresh()

    def _trend_owner() -> tuple | None:
        """Which chord-law panel ASKS for the trend (the rest quote it).

        The wing's, where there is one; otherwise the first law the problem
        declares. There is one trend flag for the whole design — it reaches
        every surface's planform through one ChordLimits — so exactly one
        panel may own the control.
        """
        groups = _chord_groups(config.effective_bounds(S))
        if not groups:
            return None
        return ("", "") if ("", "") in groups else next(iter(groups))

    def _chord_trend_control(key):
        """Which end of THIS planform is the wide one.

        The other three chord limits are lengths and angles and live in their
        own card; this one is a statement about the shape the panel above is
        drawing, and about which candidates the band below may contain — so
        it is asked here, once, beside the drawing it changes.
        """
        if "chord_trend" not in sp().flags:
            return
        trend = _chord_trend()
        if key != _trend_owner():
            widgets.hint(f"Trend: {CHORD_TREND_LABELS[trend]} — one statement "
                         f"for the whole design, asked with the "
                         f"{CHORD_GROUPS[_trend_owner()][0]}'s law.")
            return
        with _row("trend"):
            ui.select(CHORD_TREND_LABELS, value=trend,
                      on_change=lambda e: _set_chord_trend(e.value)) \
                .props("outlined dense").classes("grow min-w-0")
        widgets.hint(CHORD_TREND_NOTES[trend])
        if trend != "free":
            widgets.hint("It holds on every designed surface, and it is "
                         "enforced on the planform each candidate draws: one "
                         "that breaks it scores the penalty. The band below "
                         "shows only the laws that keep it.")

    def _baseline_violation(key, eff: dict, lim):
        """Why the STRAIGHT TAPER this band is drawn on breaks the limits.

        Worth saying on its own, because it is the one empty band no chord
        law can fix: if the trapezoid at the middle of the taper box already
        falls below the minimum chord, every law that lifts the tip has to
        drop something else, and the answer is a different taper (or a
        different limit), not a wider law. The words are ChordLimits' own, so
        they are the words the run would have used.
        """
        import numpy as np

        lam, c_root, semi = _chord_baseline(key, eff)
        if lim is None or not c_root:
            return None
        eta = np.linspace(0.0, 1.0, int(lim.n_check))
        return lim.violation(eta * semi, c_root * (1.0 - (1.0 - lam) * eta))

    def _empty_band_note(reach, trend: str, why: str | None = None) -> str:
        """Why there is no band at all, in the terms it was asked in."""
        lim = reach.limits
        asked = []
        if lim is not None:
            if lim.c_min_m is not None:
                asked.append(f"a chord below {lim.c_min_m:.3g} m")
            if lim.c_max_m is not None:
                asked.append(f"a chord above {lim.c_max_m:.3g} m")
            if lim.rate_max_deg is not None:
                asked.append(f"a taper steeper than {lim.rate_max_deg:.3g}°")
        if why:
            return (f"The straight taper this band is drawn on already breaks "
                    f"the limits — {why} — and no law in this box lifts it "
                    f"back: the area is held, so a law that widens one end "
                    f"narrows the other. Widen the law, move the limit, or "
                    f"narrow the taper row until the baseline itself obeys "
                    f"it.")
        if asked:
            return ("No planform in this box keeps those limits: every "
                    "candidate draws " + " or ".join(asked)
                    + (", or turns the chord the wrong way"
                       if trend != "free" else "")
                    + ", and every one of them scores the penalty. Widen the "
                      "law, or move the limits to where the wing already is.")
        if trend != "free":
            return ("No planform in this box keeps that trend: every "
                    "candidate either collapses the chord or turns it the "
                    "wrong way, and every one of them scores the penalty. "
                    "Widen the law, or ask for a different trend.")
        return ("No planform in this box is flyable: every candidate "
                "collapses the chord somewhere, and every one of them "
                "scores the penalty. Narrow the coefficients below.")

    def _render_chord_law(key):
        """What this law's box reaches, redrawn without any of its inputs."""
        holder = boxes.get(f"law_{key}")
        if holder is None:
            return
        holder.clear()
        eff = config.effective_bounds(S)
        labels = _chord_groups(eff).get(key) or []
        if not labels:
            return
        trend = _chord_trend()
        lim, unmeasured = _live_limits(_chord_baseline(key, eff)[1])
        reach, c_root, semi = _chord_reach(key, labels, eff, trend=trend,
                                           limits=lim)
        why = _baseline_violation(key, eff, lim)
        with holder:
            if reach.empty:
                widgets.hint(_empty_band_note(reach, trend, why), "bad")
            else:
                _chord_reach_readout(reach, c_root, semi,
                                     boxed=key != ("", ""))
                figstyle.show(_reach_fig(reach, c_root, semi),
                              f"chord_law_{key[0]}{key[1]}", 240)
                if why:
                    # the band survives, but the dashed line through it does
                    # not: only the laws that lift the baseline back over the
                    # limit are in the picture, and the picture should say so
                    widgets.hint(
                        f"The dashed straight taper is itself refused — "
                        f"{why}. Only the laws that lift it back over the "
                        f"limit are in the band, so the law here is not a "
                        f"freedom the design can give up.", "warn")
            if unmeasured:
                widgets.hint(
                    f"The band above is NOT holding the {unmeasured}: those "
                    f"are limits in metres, and this surface's size is "
                    f"searched rather than stated, so nothing here can put a "
                    f"metre on its chord. They are enforced on every "
                    f"candidate all the same — a planform that breaks one "
                    f"scores the penalty, and the band cannot show you "
                    f"which.", "warn")
            sources = {eff[label][1] for label in labels}
            if "released" in sources:
                widgets.hint("Released: this session constrains the law not "
                             "at all — the solver's own published box is "
                             "searched.", "warn")
            elif sources == {"default"}:
                widgets.hint("The solver's own published box, bit-for-bit.")

    def _render_chord_rows(key):
        """The coefficient rows themselves. Redrawn from the DEVIATION field
        (which writes them) and from nowhere else — a row redrawn by its own
        handler loses the focus and swallows the rest of the number."""
        holder = boxes.get(f"coef_{key}")
        if holder is None:
            return
        holder.clear()
        eff = config.effective_bounds(S)
        labels = _chord_groups(eff).get(key) or []
        off = config.released_rows(S)
        with holder:
            _box_grid([(label, eff[label]) for label in labels], off)

    def _chord_reach_readout(reach, c_root, semi, boxed: bool = False):
        """The band in numbers: what the widest and narrowest candidate in
        this box put at the root and at the tip.

        ``boxed`` says the surface's SIZE is itself a design variable, so the
        lengths are quoted on the middle of the box the run searches rather
        than on a planform this shell chose. The baseline taper always is —
        it is a design variable everywhere — so it is quoted as its own row.
        """
        unit = "m" if c_root else "× root chord (straight taper)"
        root_base = c_root or 1.0
        tip_base = root_base * reach.taper
        lo, hi = reach.root
        widgets.kv("root chord",
                   f"{lo * root_base:.3g} – {hi * root_base:.3g} {unit}")
        lo, hi = reach.tip
        widgets.kv("tip chord",
                   f"{lo * tip_base:.3g} – {hi * tip_base:.3g} {unit}")
        widgets.kv("baseline taper", f"λ = {reach.taper:.3g}",
                   tip="the straight taper the law multiplies. It is a "
                       "design variable, so the band is drawn on the middle "
                       "of the box the run searches — narrow the taper row "
                       "to sharpen this picture")
        if semi:
            widgets.kv("semi-span",
                       f"{semi:.3g} m" + (" · mid of its box" if boxed
                                          else ""),
                       tip="the law holds the AREA, so it moves chord along "
                           "this span without changing it")
        widgets.kv("flyable share of the box",
                   f"{100.0 * reach.flyable_frac:.0f}%",
                   color=theme.WARN if reach.flyable_frac < 0.5 else "",
                   tip="what is left after the chord collapses, the trend "
                       "turns the wrong way and the limits in metres are "
                       "broken; the rest score the penalty instead of an L/D")
        # ...and when the trend is what emptied it, the two ways out, priced.
        # This shell OPENS on "the root is the largest chord", which is the
        # right default for the law it opens on (measured 85 of 128 draws fly
        # on `poly`) and a very expensive one for two of the others (`ends`
        # 6/128, `kinked` 4/128): those laws bend the chord about the MIDDLE
        # of the span, so almost every coefficient in their published box
        # turns it up again somewhere outboard. A search over 5 % of a box is
        # a search that reports "no solution", and it must not be a surprise.
        if reach.flyable_frac < 0.5 and _chord_trend() != "free":
            widgets.hint(
                f"Most of this law's box is refused by the chord TREND, not "
                f"by the physics: only "
                f"{100.0 * reach.flyable_frac:.0f}% of it draws a wing whose "
                f"chord never grows outboard. Either narrow the '% away from "
                f"straight taper' above — a smaller deviation bends the "
                f"chord less and turns it up less — or set the trend to "
                f"'either end may be the wide one' beside this drawing. A "
                f"box this thin will report no solution at any budget.",
                "warn")
        held = _limits_held(reach)
        if held:
            widgets.kv("band holds", held,
                       tip="the band above contains only the planforms these "
                           "allow — a candidate outside them is refused, not "
                           "scored, so it has no business in the picture")

    def _limits_held(reach) -> str:
        """The live limits the band was drawn under, as one line.

        Read off the ChordReach rather than off the flags, so it names what
        was APPLIED: a panel that cannot measure a metre draws the band
        without it, and this line has to agree with the band, not with the
        card above it.
        """
        lim = reach.limits
        if lim is None:
            return ""
        said = []
        if lim.c_min_m is not None:
            said.append(f"c ≥ {lim.c_min_m:.3g} m")
        if lim.c_max_m is not None:
            said.append(f"c ≤ {lim.c_max_m:.3g} m")
        if lim.rate_max_deg is not None:
            said.append(f"|dc/dy| ≤ {lim.rate_max_deg:.3g}°")
        if lim.trend != "free":
            said.append(CHORD_TREND_LABELS[lim.trend])
        return " · ".join(said)

    def _reach_fig(reach, c_root, semi) -> go.Figure:
        """The chord distribution this box draws: the straight-taper baseline
        and the band every flyable candidate in the box stays inside.

        Under a chord TREND, or under a chord limit in metres, the band is
        already theirs (the candidates they forbid are not in it,
        geometry.chord_reach), so the drawing says so rather than leaving a
        reader to notice that both edges now fall from the root. The limits
        in metres are drawn as the lines the band now touches: they are the
        one thing on this picture that is a NUMBER the user typed, and seeing
        the band stop on one is what says the design box and the limit are
        talking about the same wing."""
        eta = reach.eta
        base = (c_root or 1.0) * (1.0 - (1.0 - reach.taper) * eta)
        x = eta * semi if semi else eta
        fig = go.Figure()
        fig.add_scatter(x=x, y=reach.hi * base, mode="lines", name="widest",
                        line=dict(width=1, color=theme.ACCENT))
        fig.add_scatter(x=x, y=reach.lo * base, mode="lines",
                        name="narrowest", fill="tonexty",
                        fillcolor="rgba(29,95,164,0.16)",
                        line=dict(width=1, color=theme.ACCENT))
        fig.add_scatter(x=x, y=base, mode="lines", name="straight taper",
                        line=dict(width=2, color=theme.INK_MUTED,
                                  dash="dash"))
        notes = []
        if reach.trend != "free":
            notes.append(CHORD_TREND_LABELS[reach.trend])
        lim = reach.limits
        if lim is not None and c_root:
            for value, name in ((lim.c_min_m, "minimum chord"),
                                (lim.c_max_m, "maximum chord")):
                if value is not None:
                    fig.add_hline(y=float(value), line_width=1,
                                  line_dash="dot", line_color=theme.WARN,
                                  annotation_text=name,
                                  annotation_position="top left",
                                  annotation_font_size=10)
            # the rate is a SLOPE, so it is not a line on this picture; the
            # band already holds it, and the number is what is missing
            if lim.rate_max_deg is not None:
                notes.append(f"|dc/dy| ≤ {lim.rate_max_deg:.3g}°")
        if notes:
            fig.add_annotation(
                xref="paper", yref="paper", x=0.99, y=0.02,
                xanchor="right", yanchor="bottom", showarrow=False,
                font=dict(size=10, color=theme.WARN),
                text=" · ".join(notes))
        fig.update_layout(
            height=240, legend=dict(orientation="h", y=1.14),
            xaxis_title="span station y [m]" if semi
            else "η = |2y/b| — root to tip",
            yaxis_title="chord [m]" if c_root
            else "chord ÷ straight-taper root chord")
        return fig

    def _set_chord_dev(key, value):
        """Ask the law for a DEVIATION and store it in the coefficients.

        The band is monotone in the coefficient bound — a wider box contains
        a narrower one — so the deviation inverts to exactly one symmetric
        box, which is what the run then searches. Releasing is undone here:
        asking for a limit is asking for the row to be constrained.

        The field is not redrawn (it is the one being typed into); everything
        that follows the number lives in ``boxes["law_…"]`` underneath it.
        """
        if value in (None, ""):
            return
        from aerobo import geometry

        eff = config.effective_bounds(S)
        labels = _chord_groups(eff).get(key) or []
        if not labels:
            return
        reach, _, _ = _chord_reach(key, labels, eff)
        if widgets.is_echo(value, 100.0 * reach.dev_max, 1):
            return
        want = float(value) / 100.0
        if not want > 0.0:
            # 0 % IS a question — "let the chord be the straight taper and
            # nothing else" — and it used to be answered by returning in
            # silence, leaving the field showing 0 beside a box it had not
            # touched. The law's own floor is the honest answer: the smallest
            # deviation `geometry.chord_bound_for_dev` can express (it clamps
            # its own result to 1e-4 for the same reason), which searches a
            # band as close to the straight taper as the coefficients allow.
            want = 1e-4
            ctx.log("0 % away from straight taper is the straight taper "
                    "itself — the band was set as close to it as the "
                    "coefficients can express. To search no law at all, "
                    "switch the chord-law rows off.", "warn")
        lam = _chord_baseline(key, eff)[0]
        # inverted UNDER THIS LAW: the same coefficient box bends a cubic and
        # an interior-only law by quite different amounts, so a box computed
        # for the published law would answer a question nobody asked
        f = geometry.chord_bound_for_dev(want, len(labels), lam,
                                         law=_chord_law())
        W["bounds_off"] = [label for label in (W.get("bounds_off") or [])
                           if label not in labels]
        # ...and any coefficient this session had FIXED goes back into the
        # search: asking for a band is asking for the law to be searched, and
        # a coefficient left pinned would sit outside the band drawn under
        # the field that just wrote it
        W["fixed"] = {k: v for k, v in (W.get("fixed") or {}).items()
                      if k not in labels}
        # ...and written as the LAW's own kind of box, not as a symmetric
        # pair: the elliptic blend's parameter is a fraction in [0, f], and
        # writing -f into it would search planforms bent AWAY from elliptic
        # that nobody asked for
        rows = geometry.chord_bounds(len(labels), f, _chord_law())
        for label, row in zip(labels, rows):
            W["bounds"][label] = [float(row[0]), float(row[1])]
        ctx.log(f"chord law ({CHORD_GROUPS[key][0]}): the chord may bend "
                f"{100.0 * want:.3g}% away from straight taper — "
                f"|k| ≤ {f:.4g} on {len(labels)} row(s)", "info")
        _render_chord_law(key)
        _render_chord_rows(key)
        _render_derived_geo()
        ctx.refresh()

    def _stated_ar() -> float | None:
        """The aspect ratio of the wing ON SCREEN, or None.

        ``session.flown_size`` answers only where the shell decides the size
        and the planform is FIXED — with the size searched it returns None,
        which is exactly the configuration this card is most used in. So the
        fallback is the same pair every other band in this shell is taken
        around: the span stage 2 flies (:func:`session.nominal_span`) and the
        mission's own area.
        """
        size = session.flown_size(S)
        if size is None:
            span = session.nominal_span(S)
            area = float((S.get("mission") or {}).get("s_ref_m2") or 0.0)
            size = (span, area) if span and area > 0.0 else None
        if size is None:
            size = api.planform_size(W["problem"])
        if not size or not float(size[1]) > 0.0:
            return None
        return float(size[0]) ** 2 / float(size[1])

    def _ar_limit_default(key: str) -> float | None:
        """A starting value for an aspect-ratio limit just switched on, or
        ``None`` where the limit already live leaves no room for one.

        Off the planform this run actually flies, exactly as
        :func:`_chord_limit_default` is: switching a limit on may not, by
        itself, refuse the design on screen. +-20% of the flown ratio, then
        clipped into the solvers' own band, because a limit wider than that
        band would be a number the card shows and the run cannot honour.

        ...AND INSIDE THE OTHER END. The clip alone wrote pairs the run
        refuses: on a wing whose stated ratio is past the solvers' ceiling
        both +-20 % clip to 40, so with "minimum aspect ratio" already on at
        40, switching the maximum on stored (40, 40) — which
        ``api.ar_limits_of`` refuses outright (it wants ar_max > ar_min), and
        the refusal came out of the card that draws the limit, replacing
        stage 3's whole Design box with a traceback. The room left by the
        live end is the band this may pick from, and where there is none the
        answer is no number at all rather than an invalid pair.
        """
        from aerobo.sizing import AR_LIMITS

        ar = _stated_ar() or 10.0
        lo, hi = float(AR_LIMITS[0]), float(AR_LIMITS[1])
        other = W["flags"].get("ar_max" if key == "ar_min" else "ar_min")
        if other not in (None, ""):
            # a gap, not a touch: the pair is strict (>), and two numbers a
            # rounding apart is not a band anyone asked for
            if key == "ar_min":
                hi = min(hi, float(other) - AR_LIMIT_GAP)
            else:
                lo = max(lo, float(other) + AR_LIMIT_GAP)
        if not hi >= lo:
            return None
        want = 0.8 * ar if key == "ar_min" else 1.25 * ar
        return round(min(max(want, lo), hi), 2)

    def _set_ar_limit_on(key: str, on: bool):
        value = _ar_limit_default(key) if on else None
        if on and value is None:
            # the other end is already at the solvers' own edge: there is no
            # value this end could take that the run would accept, so the
            # switch does not move and the card says why (a limit that
            # refuses everything is not a limit — it is an empty search)
            other = "ar_max" if key == "ar_min" else "ar_min"
            msg = (f"{key} cannot be switched on: {other} is at "
                   f"{float(W['flags'][other]):.4g} and the two must differ "
                   f"by at least {AR_LIMIT_GAP:g}. Move {other} first.")
            ctx.log(msg, "warn")
            ui.notify(msg, type="warning")
            _render_box()
            ctx.refresh()
            return
        W["flags"][key] = value
        if not on:
            W["flags"].pop(key, None)
        # the CLIP follows the live band; a row the user typed is
        # still their answer, so only the shell's own rows move
        session.write_size_bands(S, only_shell_owned=True)
        _render_box()
        ctx.log(f"{key}: " + (f"aspect ratio limited to "
                              f"{W['flags'][key]:.4g}" if on else "off"),
                "info")
        ctx.refresh()

    def _set_ar_limit(key: str, value):
        """A typed aspect-ratio limit.

        Validated through ``api.ar_limits_of`` — the same reader the problem
        builder calls — so a max below the min is refused HERE, with the old
        pair put back, instead of raising out of a build inside a NiceGUI
        handler where nothing on screen would move. The field is not rebuilt
        from its own handler (a rebuilt input swallows the rest of the number
        being typed).
        """
        if value in (None, ""):
            return
        if W["flags"].get(key) is None:
            return                      # the row is off; the field is disabled
        if widgets.is_echo(value, float(W["flags"][key])):
            return
        held = dict(W["flags"])
        W["flags"][key] = float(value)
        try:
            api.ar_limits_of(W["flags"])
            session.write_size_bands(S, only_shell_owned=True)
            config.effective_bounds(S)
        except Exception as exc:      # noqa: BLE001 — the two ends conflict
            W["flags"] = held
            session.write_size_bands(S, only_shell_owned=True)
            ctx.log(f"{key} {value}: {exc}", "warn")
            ui.notify(str(exc), type="warning")
            _ar_limit_repaint()
            return
        _ar_limit_repaint()

    def _ar_limit_repaint():
        """Everything a typed aspect-ratio limit moves, without rebuilding
        the field it was typed into.

        The limit CLIPS the size box (`session.clip_size_box`), so the span
        row's own numbers move; the size card in the type view quotes them;
        and the derived-geometry panel under the box is drawn from them.
        Each of those lives in a container of its own, so each is repainted
        by name and the box view itself is skipped.
        """
        _repaint_row_values()
        _render_ar_note()
        _render_size_note()
        _render_derived_geo()
        ctx.refresh(("wing", "box"))

    def _ar_limit_rows():
        """The wing's aspect-ratio band, in the units it is measured in.

        Drawn inside the size-limits box beside the chord limits, because it
        answers the same question in the same units-of-the-real-world way —
        and because the two are read together: a minimum chord and a maximum
        aspect ratio are the two ends of "how thin may this wing get".
        """
        if not any(k in sp().flags for k in api.AR_LIMIT_KEYS):
            return
        from aerobo.sizing import AR_LIMITS

        live = api.ar_limits_of(W["flags"])
        now = _stated_ar()
        widgets.hint(
            f"The wing's ASPECT RATIO, b²/S — the one size question the box "
            f"above cannot ask, because it is a ratio of two of its rows and "
            f"every rectangle of span x area covers a range of them. Off is "
            f"the solvers' own band {AR_LIMITS[0]:.0f}–{AR_LIMITS[1]:.0f}, "
            f"which a limit here can only narrow."
            + (f" Mid-box now: AR {now:.3g}." if now else ""))
        for key, label, unit, step, tip in AR_LIMIT_ROWS:
            held = W["flags"].get(key)
            with _row(label):
                ui.switch(value=held is not None,
                          on_change=lambda e, k=key:
                          _set_ar_limit_on(k, bool(e.value))) \
                    .props("dense")
                boxes[f"lim_{key}"] = ui.number(
                    value=widgets.shown(held) if held is not None else None,
                    step=step,
                    on_change=lambda e, k=key: _set_ar_limit(k, e.value)) \
                    .props("outlined dense hide-bottom-space") \
                    .classes("w-28 shrink-0")
                if held is None:
                    boxes[f"lim_{key}"].disable()
                ui.label(unit).classes("field-unit")
            widgets.hint(tip)
        # ...and what the live limit is doing — including a stated wing that
        # breaks it — in a container of its own, so a typed limit can move it
        # without rebuilding the field (:func:`_render_ar_note`).
        boxes["ar_note"] = ui.column().classes("w-full gap-0")
        _render_ar_note()

    def _render_ar_note():
        """What the live aspect-ratio limit is doing, in its own container."""
        note = boxes.get("ar_note")
        if note is None:
            return
        note.clear()
        live = api.ar_limits_of(W["flags"])
        now = _stated_ar()
        with note:
            _ar_limit_note_body(live, now)

    def _ar_limit_note_body(live, now):
        if live is not None and now is not None:
            band = session.ar_band(S)
            if not (band[0] <= now <= band[1]):
                widgets.hint(
                    f"The wing this mission states is AR {now:.3g}, outside "
                    f"{band[0]:.3g}–{band[1]:.3g}. The size box still holds "
                    f"it — a limit may not push the stated wing out of its "
                    f"own box — so the run will refuse it per candidate "
                    f"until the mission or the limit moves.", "warn")
        if live is not None:
            eff_now = config.effective_bounds(S)
            clips = "b_m" in eff_now and session.AREA_ROW in eff_now
            widgets.hint(
                ("The size box above is clipped so no corner of it leaves "
                 "this band, AND every candidate is checked against it where "
                 "its size is resolved — a clip on two rows cannot make a "
                 "rectangle out of a region that is not one."
                 if clips else
                 "This box does not search a span and an area as two rows, "
                 "so there is no corner to clip: the limit is checked per "
                 "candidate where the size is resolved, and a design outside "
                 "it is refused."), "info")

    # ------------------------------------------- the PAIR's two areas
    #
    # The box states a tandem as a TOTAL area and a SPLIT. Nobody designs a
    # pair that way: a pair is two wings, and what a builder, a rule or a
    # tunnel states is each wing's own area. Both directions are exact
    # (S = S_f + S_r, k = S_f/S), so the card can be asked either way round
    # and the two rows follow (``session.pair_area_rows``).
    def _set_pair_mode(mode: str):
        session.set_pair_area_mode(S, str(mode))
        _render_box()
        _render_size_note()
        ctx.render("airfoil", "ranking")
        ctx.render("airfoil_aft", "ranking")
        ctx.log(f"the pair's area: "
                + ("stated per wing" if str(mode) == "wings"
                   else "stated as the pair's total"), "info")
        ctx.refresh()

    def _set_pair_exact(wing: str, on: bool):
        session.set_pair_area(S, wing, exact=bool(on))
        _render_box()
        _render_size_note()
        ctx.render("airfoil", "ranking")
        ctx.render("airfoil_aft", "ranking")
        ctx.refresh()

    def _set_pair_value(wing: str, value):
        """A typed area for ONE wing. The card is NOT rebuilt from inside the
        field's own handler — that swallows the rest of the number being
        typed — so only the read-outs below it are redrawn."""
        if value in (None, ""):
            return
        st = session.pair_area_state(S)[wing]
        if st.get("value") is not None and widgets.is_echo(
                value, float(st["value"])):
            return
        if not session.set_pair_area(S, wing, value=value):
            ctx.log(f"{wing} area {value}: not an area", "warn")
            return
        _pair_area_repaint()

    def _set_pair_band(wing: str, end: str, value):
        if value in (None, ""):
            return
        st = session.pair_area_state(S)[wing]
        band = list(st.get("band") or [])
        if len(band) != 2:
            return
        i = 0 if end == "lo" else 1
        if widgets.is_echo(value, float(band[i])):
            return
        band[i] = value
        if not session.set_pair_area(S, wing, band=band):
            ctx.log(f"{wing} area {end} {value}: the band's ends cross", "warn")
            return
        _pair_area_repaint()

    def _pair_area_repaint():
        """Everything two stated areas move, without rebuilding the field.

        They write the SPLIT row (and the total where the family searches
        one), so the box's own numbers move; the size card in the type view
        quotes the pair's areas; stage 2 designs each surface's section at
        its share of the area, so its ranking quotes the split too. Each of
        those is a different container from the one being typed into.
        """
        _repaint_row_values()
        _render_pair_note()
        # the PIN two exact areas write is quoted under the table, in its own
        # container — a typed area moves the value it names
        _render_fixed_note()
        _render_size_note()
        _render_derived_geo()
        ctx.render("airfoil", "ranking")
        ctx.render("airfoil_aft", "ranking")
        ctx.refresh(("wing", "box"))

    def _adopt_pair_total():
        """Make the mission's wing area the one the two stated wings add up
        to. A BUTTON and not a side effect: the mission's card owns that
        number, and a stage-3 control that rewrote it silently would be the
        same question answered in two places."""
        got = session.pair_area_total(S)
        if got is None:
            return
        want = 0.5 * (got[0] + got[1])
        if not session.set_reference_area(S, want):
            ui.notify(f"{want:.4g} m² is not a size this mission can fly",
                      type="warning")
            return
        session.sync_wing_from_mission(S)
        session.write_size_bands(S, only_shell_owned=True)
        ctx.log(f"the mission's wing area is now {want:.4g} m² — the two "
                f"wings you stated", "info")
        _render_box()
        # a BUTTON, not a typed field: the whole type view may be rebuilt,
        # and it must be — it quotes the area this just moved
        ctx.render("wing", "type")
        ctx.render_when_shown("mission")
        ctx.render("airfoil", "ranking")
        ctx.render("airfoil_aft", "ranking")
        ctx.refresh()

    def _pair_area_controls():
        if not session.pair_area_available(S):
            return
        mode = session.pair_area_mode(S)
        with widgets.group_box("The pair's two areas"):
            widgets.hint(
                "The box states this pair as a TOTAL area and a SPLIT, which "
                "is what the solver searches. A pair is two wings, so it can "
                "be stated that way instead: give each wing its own area — "
                "exactly, or as a band — and the total and the split follow "
                "(S = S_front + S_rear, split = S_front / S).")
            with _row("stated as"):
                ui.select({"total": "the pair's total area and its split",
                           "wings": "each wing's own area"},
                          value=mode,
                          on_change=lambda e: _set_pair_mode(e.value)) \
                    .props("outlined dense").classes("grow min-w-0")
            if mode != "wings":
                return
            st = session.pair_area_state(S)
            for wing in ("front", "rear"):
                side = st[wing]
                with _row(f"{wing} wing"):
                    ui.switch(value=bool(side["exact"]),
                              on_change=lambda e, w=wing:
                              _set_pair_exact(w, bool(e.value))) \
                        .props("dense")
                    if side["exact"]:
                        boxes[f"pair_{wing}"] = ui.number(
                            value=widgets.shown(side["value"]), step=0.5,
                            on_change=lambda e, w=wing:
                            _set_pair_value(w, e.value)) \
                            .props("outlined dense hide-bottom-space") \
                            .classes("w-28 shrink-0")
                    else:
                        band = list(side["band"] or [0.0, 0.0])
                        for i, end in enumerate(("lo", "hi")):
                            boxes[f"pair_{wing}_{end}"] = ui.number(
                                value=widgets.shown(band[i]), step=0.5,
                                on_change=lambda e, w=wing, k=end:
                                _set_pair_band(w, k, e.value)) \
                                .props("outlined dense hide-bottom-space") \
                                .classes("w-24 shrink-0")
                    ui.label("m²").classes("field-unit")
                widgets.hint("Exactly this area (the switch), or a band the "
                             "search may move it in."
                             if wing == "front" else
                             "The rear wing's own area, asked the same way.")
            boxes["pair_note"] = ui.column().classes("w-full gap-0")
            _render_pair_note()

    def _render_pair_note():
        """The pair's read-out, in a container of its own."""
        note = boxes.get("pair_note")
        if note is None:
            return
        note.clear()
        with note:
            _pair_area_readout()

    def _pair_area_readout():
        """What the two stated wings imply, against what the run will fly."""
        rows = session.pair_area_rows(S)
        if not rows:
            widgets.hint("Both wings, or neither: one area alone states "
                         "neither the total nor the split.", "warn")
            return
        total = session.pair_area_total(S)
        kind, val = rows[session.PAIR_SPLIT_ROW]
        said = (f"split {float(val):.4g}" if kind == "pin"
                else f"split {float(val[0]):.4g}–{float(val[1]):.4g}")
        widgets.hint(
            f"That is a total of "
            + (f"{total[0]:.4g} m²" if total[0] == total[1]
               else f"{total[0]:.4g}–{total[1]:.4g} m²")
            + f" at {said}."
            + ("" if session.AREA_ROW in rows else
               " The total is not a row this family searches, so what the "
               "run flies is the mission's own area — only the split is "
               "written into the box."))
        # the published SPLIT band, which a stated pair may leave: a
        # calibration is a default and not a ban, so it is said and not
        # refused
        published = config.default_bounds(S).get(session.PAIR_SPLIT_ROW)
        ends = (val, val) if kind == "pin" else val
        if published and not (float(published[0]) <= float(ends[0])
                              and float(ends[1]) <= float(published[1])):
            widgets.hint(
                f"That split is outside the band this family was calibrated "
                f"over ({float(published[0]):.3g}–{float(published[1]):.3g}). "
                f"It is your answer, so it stands — but the pair's solvers "
                f"have not been measured there.", "warn")
        area = session.reference_area(S)
        if session.AREA_ROW in rows or area is None or total is None:
            return
        if abs(0.5 * (total[0] + total[1]) - float(area)) <= 1e-9:
            return
        widgets.hint(
            f"The mission states {float(area):.4g} m² for the pair, and the "
            f"two wings above add up to "
            + (f"{total[0]:.4g} m²" if total[0] == total[1]
               else f"{total[0]:.4g}–{total[1]:.4g} m²")
            + ". Only the split is written into the box, so the run still "
              "flies the mission's area unless you take the total over.",
            "warn")
        ui.button("Use the two wings as the mission's area",
                  icon="straighten", on_click=_adopt_pair_total) \
            .props("outline dense no-caps")

    def _chord_limit_controls(owns_trend: bool = True):
        """The chord distribution in METRES and DEGREES.

        The rows above are the design box, and for the chord law they are
        COEFFICIENTS: nobody knows what ``chord_k2 = -0.31`` draws. What an
        engineer actually has to hold is a minimum chord, a maximum, a limit
        on how fast the chord may change, and which end of the wing is the
        wide one — none of which is a box on any coefficient, because the
        area rescale makes all four nonlinear in them.

        So they are stated here in their own units and checked on the chord
        distribution each candidate actually draws (api.CHORD_LIMIT_KEYS →
        geometry.ChordLimits). A candidate that breaks a live limit is an
        in-contract failure, exactly like a law that collapses the chord: it
        scores the penalty rather than being quietly clipped into something
        the user did not ask for.

        ``owns_trend`` is False when a chord-law panel is showing the trend
        already — it is the one limit that is a statement about the SHAPE the
        law draws, so where there is a law it is asked in that drawing and
        appears here only for a straight-taper problem, which has no such
        panel and where the trend is still a real constraint (it decides
        whether the taper may exceed 1).
        """
        asks_chord = any(k in sp().flags for k in api.CHORD_LIMIT_KEYS)
        asks_ar = any(k in sp().flags for k in api.AR_LIMIT_KEYS)
        if not (asks_chord or asks_ar):
            return
        with widgets.group_box("Wing size limits"):
            _ar_limit_rows()
            if not asks_chord:
                return
            widgets.hint(
                "The wing's chord distribution in the units it is built in. "
                "Each limit is off until you switch it on, and off is the "
                "solver's own behaviour exactly. A planform that breaks a "
                "live limit is refused (it scores the penalty), never "
                "quietly reshaped."
                + (" The band each chord law draws is already holding them, "
                   "so switching one on narrows the picture as well as the "
                   "run." if _chord_groups(config.effective_bounds(S))
                   else ""))
            # the fourth size question this card does NOT ask, said once so
            # a reader looking for it stops looking here
            widgets.hint(
                "The wing's SPAN is not asked here: it is the first row of "
                "the wing design box above."
                if "b_m" in config.effective_bounds(S) else
                "The wing's SPAN is not asked here: this family does not "
                "search it — it is stated on the size card in the type "
                "view.")
            for key, label, unit, step, tip in CHORD_LIMIT_ROWS:
                held = W["flags"].get(key)
                with _row(label):
                    ui.switch(value=held is not None,
                              on_change=lambda e, k=key:
                              _set_chord_limit_on(k, bool(e.value))) \
                        .props("dense")
                    boxes[f"lim_{key}"] = ui.number(
                        value=widgets.shown(held) if held is not None else None,
                        step=step,
                        on_change=lambda e, k=key:
                        _set_chord_limit(k, e.value)) \
                        .props("outlined dense hide-bottom-space") \
                        .classes("w-28 shrink-0")
                    if held is None:
                        boxes[f"lim_{key}"].disable()
                    ui.label(unit).classes("field-unit")
                widgets.hint(tip)
            if not owns_trend:
                return
            trend = _chord_trend()
            with _row("trend"):
                ui.select(CHORD_TREND_LABELS, value=trend,
                          on_change=lambda e: _set_chord_trend(e.value)) \
                    .props("outlined dense").classes("grow min-w-0")
            widgets.hint(CHORD_TREND_NOTES[trend])

    def _tail_limit_controls():
        """The SECOND surface's span and chord, in metres.

        Same shape as the chord limits above and for the same reason: the
        rows in the box are an area and an aspect ratio, and a builder's
        constraint is a length. The difference is that these two convert
        EXACTLY — b = sqrt(AR·S), c = sqrt(S/AR) — so a limit typed here
        narrows the box the sampler draws from (``tail.TailLimits.narrow``)
        instead of only refusing the draws that break it. With the aspect
        ratio fixed the narrowing is the whole constraint; with it searched
        the box is the smallest one CONTAINING what the limits allow, and
        the corners it adds are refused per candidate, exactly as a broken
        chord limit is.
        """
        if not any(k in sp().flags for k in
                   [r[0] for r in TAIL_LIMIT_ROWS]):
            return
        name = session.second_surface_name(S) or "second surface"
        with widgets.group_box(f"{name.capitalize()} size limits"):
            geo = session.surface_geometry(S, "tail")
            widgets.hint(
                f"How big the {name} is allowed to be, in metres. The box "
                f"above states it as an AREA (and an aspect ratio where its "
                f"planform is designed), which is what the solver searches "
                f"and not what anybody measures — so state it here instead "
                f"and the two rows follow."
                + (f" Mid-box now: span {geo['span']:.3g} m, chord "
                   f"{geo['mac']:.3g} m, area {geo['area']:.3g} m²."
                   if geo else "")
                + " Each limit is off until you switch it on, and off is "
                  "the solver's own box exactly.")
            widgets.hint(
                f"The {name} SPAN is not asked here: it is a band, so it is "
                f"the first row of the {name} design box above, where the "
                f"rest of this surface's constraints are.")
            for key, label, unit, step, tip in TAIL_LIMIT_ROWS:
                held = W["flags"].get(key)
                with _row(label):
                    ui.switch(value=held is not None,
                              on_change=lambda e, k=key:
                              _set_tail_limit_on(k, bool(e.value))) \
                        .props("dense")
                    boxes[f"tlim_{key}"] = ui.number(
                        value=widgets.shown(held) if held is not None else None,
                        step=step,
                        on_change=lambda e, k=key:
                        _set_tail_limit(k, e.value)) \
                        .props("outlined dense hide-bottom-space") \
                        .classes("w-28 shrink-0")
                    if held is None:
                        boxes[f"tlim_{key}"].disable()
                    ui.label(unit).classes("field-unit")
                widgets.hint(tip)

    def _car_limit_controls():
        """The car's two LIMITS: a drag ceiling and a downforce floor, in N.

        Under the box rather than on the wing-type card because that is where
        this shell keeps its limits — beside the chord limits and the second
        surface's — and because both of them are statements about the ANSWER
        rather than about the design vector. A box row is a band on a variable
        the optimiser moves; a ceiling on drag is a bound on something the
        solver reports, so it cannot be a row and should not pretend to be.

        NEITHER IS ON BY DEFAULT, and that is the change. The family used to
        ship a drag ALLOWANCE (CD 0.11, restated as 81.5 N once the area
        moved) and the answer spent it: a budget handed to a maximiser is the
        number that picks the answer, not one that bounds it — on both
        certified optima the drag margin sits at 0.048 and 0.035 while
        binding almost nowhere else in the box. So the score prices drag
        (CZ/CD) and a ceiling is yours to state or leave alone.

        The FLOOR is offered beside it because the default score is a ratio,
        and a ratio has an interior peak that is a small wing: measured over
        the family's own area band, CZ/CD peaks at 35.5 near 0.14 m² where
        the wing makes about 193 N, against 486 N at 0.4 m². The floor is how
        a user says how much downforce is worth having, and the answer then
        sits ON it.
        """
        rows = [r for r in CAR_LIMIT_ROWS if r[0] in sp().flags]
        if not rows:
            return
        with widgets.group_box("Limits"):
            widgets.hint(
                "What the ANSWER has to satisfy, as opposed to the "
                "DESIGN BOX on the next tab, which is only where the "
                "optimiser may look. Both are "
                "forces in newtons — the unit that still means something "
                "when the reference area is itself designed, since a "
                "COEFFICIENT allowance is referenced to the very area being "
                "searched. Blank is no constraint at all, not a constraint "
                "set to zero.")
            # NO CIRCUIT PAIRING NOTE. It used to say here that a chosen
            # lap already prices drag physically, so a ceiling typed in this
            # block is a SECOND, independent limit on the same drag — true,
            # and unreachable now that no card asks for a circuit
            # (`car_track_of` reads "off" for every card-built state). The
            # pairing is still honoured by the engine: api accepts a drag
            # ceiling and a lap together and refuses neither.
            wanted = _limit_the_objective_wants()
            for key, choice, label, unit, step, tip in rows:
                held = W["choices"].get(choice)
                with _row(label):
                    # A SWITCH, matching the tail limits one block up. Blank
                    # already meant "off", but a field whose off-state is an
                    # empty box does not read as a limit you can turn on —
                    # and the objective that NEEDS one has no way to point at
                    # it. The switch is the affordance; the number behind it
                    # stays disabled until it is thrown.
                    ui.switch(value=held is not None,
                              on_change=lambda e, k=choice:
                              _set_car_limit_on(k, bool(e.value))) \
                        .props("dense")
                    boxes[f"clim_{choice}"] = ui.number(
                        value=widgets.shown(held) if held is not None else None,
                        step=step, placeholder="none",
                        on_change=lambda e, k=choice:
                        _set_car_limit(k, e.value)) \
                        .props("outlined dense hide-bottom-space") \
                        .classes("w-28 shrink-0")
                    if held is None:
                        boxes[f"clim_{choice}"].disable()
                    ui.label(unit).classes("field-unit")
                    if choice == wanted and held is None:
                        # WHICH of the two this run's objective is missing.
                        # Every car objective is one half of a trade whose
                        # other half is one of these, and the select that
                        # chose it is on a different card — so the pairing
                        # has to be visible from the side that can act on it.
                        ui.label("← this objective needs it") \
                            .classes("text-[11px] text-amber-700 shrink-0")
                widgets.hint(tip)

    def _plate_chord_controls():
        """THE PLATE'S CHORD, IN METRES — the second of its two bands.

        Drawn HERE, under the design box, because this is a bound on a design
        ROW and the row is right above it. The plate's chord is a design
        variable stated as a multiple of the wing's TIP chord
        (``endplate_chord_ratio``), which is the right way to keep a plate
        proportioned to its wing and useless as a bound: the tip chord moves
        with the span, the taper and the chord law, so the same ratio is a
        different number at every point of the box. These two state it in
        METRES, which is how a regulation or a piece of bodywork is written.

        It used to be typed on the wing-TYPE card, three cards away from the
        ratio row it clips — the reader had to hold a band from one screen
        against a row on another to know what would fly. What flies is the
        ratio's chord clipped into this band: the metre band wins where they
        disagree, every point of the box stays flyable, and a design whose
        chord was clipped says so on the results page.

        Gated on the registry's own declaration, exactly as the tail and car
        limit blocks are — only the designed-endplate family has a plate
        chord to bound, so under the pylon mount this draws nothing.
        """
        if "endplate_chord_min_m" not in sp().flags:
            return
        with widgets.group_box("Plate chord limits"):
            widgets.hint(
                "The endplate's chord answers to two bands at once. Its "
                "design row above (endplate chord ratio) states it as a "
                "multiple of the wing's TIP chord; these two state it in "
                "metres. Blank is unbounded — the published problem — and a "
                "number is a hard edge the flown chord is clipped into, not "
                "a refusal.")
            for choice, label in (("car_endplate_chord_min_m",
                                   "Plate chord, at least"),
                                  ("car_endplate_chord_max_m",
                                   "Plate chord, no more than")):
                held = W["choices"].get(choice)
                with _row(label):
                    boxes[f"pchord_{choice}"] = ui.number(
                        value=widgets.shown(held) if held is not None else None,
                        step=0.05, placeholder="unbounded",
                        on_change=lambda e, k=choice:
                        _set_plate_chord(k, e.value)) \
                        .props("outlined dense hide-bottom-space") \
                        .classes("w-28 shrink-0")
                    ui.label("m").classes("field-unit")

    def _set_plate_chord(choice: str, value):
        """A typed plate-chord bound. Blank CLEARS it — an unbounded chord is
        not a chord bounded at zero.

        Writes the CHOICE key, so the bound reaches the run down the one
        channel every other car control uses (``nice_app.car_flags``), and
        goes through ``set_choice``, which knows not to rebuild the view a
        BOX_VALUE_KEY was typed into.
        """
        if value in (None, ""):
            if W["choices"].get(choice) is not None:
                set_choice(choice, None)
                ctx.log(f"{choice}: unbounded", "info")
            return
        try:
            new = float(value)
        except (TypeError, ValueError):
            return
        if not new > 0.0:
            ctx.log(f"{choice} {value}: a chord in metres is positive; clear "
                    f"the field for no bound at all", "warn")
            return
        held = W["choices"].get(choice)
        if held is not None and widgets.is_echo(value, float(held)):
            return
        set_choice(choice, new)

    #: which LIMIT each car objective is the other half of. Not a
    #: preference and not a default: an objective that maximises a quantity
    #: is bounded by the OTHER one or it runs to an edge, and which edge is a
    #: property of the objective. ``laptime`` is absent on purpose — the lap
    #: prices downforce against drag physically, at the circuit's own
    #: exchange rate, so neither limit is its missing half (nice_app's
    #: OBJECTIVE_ALREADY_PRICES says the same thing from the failure side).
    CAR_OBJECTIVE_WANTS_LIMIT = {
        "efficiency": "car_downforce_min_n",
        "drag": "car_downforce_min_n",
        "cd": "car_downforce_min_n",
        "downforce": "car_drag_budget_n",
        "cz": "car_drag_budget_n",
        # ...and the one that adds the two forces rather than trading them.
        # It wants the CEILING for the same reason 'downforce' does, and the
        # measurement is stronger here: without one the answer takes 100 % of
        # the area row and 100 % of the alpha sweep, and the ceiling is what
        # turns the span into an answer (58 % of band at 81.5 N).
        "downforce_plus_drag": "car_drag_budget_n",
    }

    def _limit_the_objective_wants() -> str | None:
        """The limit this run's objective is missing, or None."""
        obj = (W["choices"].get("car_objective")
               or api.CAR_DEFAULT_OBJECTIVE)
        return CAR_OBJECTIVE_WANTS_LIMIT.get(str(obj))

    def _car_limit_default(choice: str):
        """A starting value for a car limit just switched on, or None.

        The force the INCUMBENT wing already makes, read off the centre of
        the box the user is looking at — the same rule
        :func:`_tail_limit_default` follows, and for the same reason:
        switching a limit on must not, by itself, refuse the design on
        screen. A floor that opens at the incumbent's own downforce starts
        SATISFIED and does nothing until it is moved.

        Re-derived from the model on every throw rather than pinned here, so
        it cannot drift away from the family it bounds. Returns None when
        the box centre does not fly, and the caller then refuses to switch
        the limit on rather than inventing a number for it.
        """
        try:
            built = api.PROBLEM_SPECS[W["problem"]].build(
                {}, config.flags(S), config.bounds_overrides(S))
            b = built.problem.bounds
            out = built.evaluate(0.5 * (b[:, 0] + b[:, 1]))
            got = {"car_downforce_min_n": out.get("downforce_N"),
                   "car_drag_budget_n": out.get("drag_N")}[choice]
            return round(float(got), 1) if got and float(got) > 0.0 else None
        except Exception:
            return None

    def _set_car_limit_on(choice: str, on: bool):
        """Throw a car limit on or off.

        ON opens it at the incumbent's own force, so the limit starts
        satisfied; OFF clears the choice entirely, because a limit you have
        switched off is not a limit of zero.
        """
        if not on:
            set_choice(choice, None)
            ctx.log(f"{choice}: off", "info")
            _render_type()
            ctx.refresh()
            return
        start = _car_limit_default(choice)
        if start is None:
            ctx.log(f"{choice}: not switched on — the centre of this box "
                    f"does not fly, so there is no incumbent force to open "
                    f"the limit at. Type one, or widen the box.", "warn")
            _render_type()
            return
        set_choice(choice, start)
        ctx.log(f"{choice}: on at {start:g} N (what this box's centre "
                f"already makes — it starts satisfied)", "info")
        _render_type()
        ctx.refresh()

    def _set_car_limit(choice: str, value):
        """A typed car limit. Blank CLEARS it — a limit you have deleted is
        not a limit of zero, and clearing the field is the only way back to
        "no constraint" once one has been typed.

        Writes the CHOICE key, not ``W["flags"]`` directly, so the limit
        reaches the run down the one channel every other car control uses
        (``nice_app.car_flags``). Both worked; two channels for one question
        is what this session removed everywhere else, and leaving one here
        would have been the same defect in a new place.

        Goes through ``set_choice``, which already knows not to rebuild the
        view a VALUE_KEY was typed into (BOX_VALUE_KEYS).
        """
        if value in (None, ""):
            if W["choices"].get(choice) is not None:
                set_choice(choice, None)
                ctx.log(f"{choice}: no limit", "info")
            return
        try:
            new = float(value)
        except (TypeError, ValueError):
            return
        if not new > 0.0:
            ctx.log(f"{choice} {value}: a limit in newtons is positive; "
                    f"clear the field for no limit at all", "warn")
            return
        held = W["choices"].get(choice)
        if held is not None and widgets.is_echo(value, float(held)):
            return
        set_choice(choice, new)

    def _tail_limit_default(key: str) -> float:
        """A starting value for a limit just switched on — read off the
        surface this run actually flies, so switching one on cannot by
        itself refuse the incumbent."""
        geo = session.surface_geometry(S, "tail") or {}
        span = float(geo.get("span") or 1.0)
        chord = float(geo.get("mac") or 0.25)
        return {"tail_span_min_m": round(0.5 * span, 3),
                "tail_span_max_m": round(1.5 * span, 3),
                "tail_chord_min_m": round(0.5 * chord, 3),
                "tail_chord_max_m": round(1.5 * chord, 3)}[key]

    def _set_tail_limit_on(key: str, on: bool):
        if on:
            W["flags"][key] = _tail_limit_default(key)
        else:
            W["flags"].pop(key, None)
        # touching either end of the span pair makes the pair the USER'S:
        # the measurement no longer owns it and will not write it again
        if key in TAIL_SPAN_ROW:
            session.bounds_source(S).pop(session.TAIL_SPAN_KEY, None)
        _render_box()
        ctx.log(f"{key}: " + ("constrained" if on else "off"), "info")
        ctx.refresh()

    def _set_tail_limit(key: str, value):
        """A typed limit. The view is NOT rebuilt — rebuilding the input
        from its own handler swallows the rest of the number being typed —
        but the DERIVED read-out under the box is, because these two move
        the rows it quotes."""
        if value in (None, ""):
            return
        if W["flags"].get(key) is None:
            return                      # the row is off; the field is disabled
        if widgets.is_echo(value, float(W["flags"][key])):
            return
        held = dict(W["flags"])
        W["flags"][key] = float(value)
        try:
            # the SAME constructor the problem builder calls, so a pair this
            # card accepts is a pair the run can build. ``effective_bounds``
            # alone does not ask it — it narrows S_t_m2 through the limits
            # and never constructs them — so an inverted span band used to
            # be taken here and raise a ValueError at launch instead.
            api.tail_limits_of(W["flags"])
            config.effective_bounds(S)
        except Exception as exc:      # noqa: BLE001 — the limits can conflict
            W["flags"] = held
            ctx.log(f"{key} {value}: {exc}", "warn")
            # THE SAME REDRAW AS THE SUCCESS PATH, and for the same reason.
            # ``_render_box`` starts with ``box.clear()``, and this handler
            # belongs to a ``ui.number`` ``_render_box`` itself created — so
            # refusing a keystroke replaced all 16 number inputs in the box
            # under the cursor, exactly the failure this function's own
            # docstring says it avoids. A transient min > max while retyping
            # a value is the ordinary way to reach here.
            _render_derived_geo()
            ctx.refresh()
            return
        # ...and a typed end makes the pair theirs (see `_set_tail_limit_on`)
        if key in TAIL_SPAN_ROW:
            session.bounds_source(S).pop(session.TAIL_SPAN_KEY, None)
        _render_derived_geo()
        ctx.refresh()

    def _chord_limit_default(key: str) -> float:
        """A starting value for a limit the user has just switched on.

        Read off the planform this run actually flies, so the limit opens
        where the design already is — switching it on cannot, by itself,
        refuse the incumbent.
        """
        size = session.flown_size(S) or api.planform_size(W["problem"])
        c_mean = (float(size[1]) / float(size[0])) if size else 1.0
        return {"chord_min_m": round(0.5 * c_mean, 3),
                "chord_max_m": round(2.0 * c_mean, 3),
                "chord_rate_max_deg": 30.0}[key]

    def _set_chord_limit_on(key: str, on: bool):
        W["flags"][key] = _chord_limit_default(key) if on else None
        if not on:
            W["flags"].pop(key, None)
        _render_box()
        ctx.log(f"{key}: " + ("constrained" if on else "off"), "info")
        ctx.refresh()

    def _set_chord_limit(key: str, value):
        """A typed limit. The view is NOT rebuilt: rebuilding the input from
        its own handler swallows the rest of the number being typed.

        VALIDATED through the same constructor the problem builder calls, the
        way ``_set_tail_limit`` already was. Without it, typing 0 into
        "minimum chord" stored 0.0, and the next render raised
        ``ValueError: chord_min_m must be > 0`` out of ``_live_limits`` —
        which ``Ctx.render`` contains by replacing the WHOLE design box with
        a traceback panel. The flag stayed 0.0, so every later render did it
        again and the box never came back.
        """
        if value in (None, ""):
            return
        if W["flags"].get(key) is None:
            return                      # the row is off; the field is disabled
        if widgets.is_echo(value, float(W["flags"][key])):
            return
        held = dict(W["flags"])
        W["flags"][key] = float(value)
        try:
            api.chord_limits_of(W["flags"])
            config.effective_bounds(S)
        except Exception as exc:      # noqa: BLE001 — the limits can conflict
            W["flags"] = held
            ctx.log(f"{key} {value}: {exc}", "warn")
            # the SUCCESS path's redraw, not ``_render_box``: measured,
            # typing a bare "0" into "minimum chord" replaced all 16 number
            # inputs in the box (ids 1854… → 2179…) from inside the handler
            # of one of them, while a valid 0.45 left all 16 untouched. The
            # flags are already rolled back above, so what these containers
            # redraw is the box as it still stands.
            _render_derived_geo()
            for group in _chord_groups(config.effective_bounds(S)):
                _render_chord_law(group)
            ctx.refresh()
            return
        # ONE ChordLimits reaches every designed surface, so every chord-law
        # band follows this number — not only the wing's — and each of them
        # lives in a container of its own, never this field's
        for group in _chord_groups(config.effective_bounds(S)):
            _render_chord_law(group)
        ctx.refresh()

    def _set_chord_trend(value: str):
        if _chord_trend() == value:
            return
        # STORED for every value, "free" included. The default is no longer
        # the absence of the flag (it is `config.CHORD_TREND`), so popping the
        # key to mean "free" would have made "free" unselectable — the read
        # would fall straight back to the default that was just left.
        W["flags"][api.CHORD_TREND_KEY] = str(value)
        # the whole box: the trend clips the band EVERY chord-law panel draws
        # (not only the one whose select was touched), and the flyable share
        # each of them quotes moves with it
        _render_box()
        ctx.log(f"chord trend: {CHORD_TREND_LABELS[value]}", "info")
        ctx.refresh()

    def _render_fixed_note():
        """Which rows have left the design vector, and at what value."""
        holder = boxes.get("fixed_note")
        if holder is None:
            return
        holder.clear()
        fixed_now = config.fixed_rows(S)
        if not fixed_now:
            return
        with holder:
            widgets.hint(
                "Fixed: "
                + ", ".join(f"{k} = {v:.6g}" for k, v in fixed_now.items())
                + f". The search is {len(fixed_now)} dimension"
                + ("s" if len(fixed_now) != 1 else "")
                + " smaller than this family's published one.", "warn")
            # ...AND A PINNED ANHEDRAL SAYS SO. The searched row is floored
            # at 0 while nothing prices its sign, and a PIN goes round the
            # floor by construction — it is not a band. It is still the
            # user's number and is not refused; it is named.
            _anhedral = api.anhedral_note(
                fixed_now.get(api.WING_CANT_KEYS[0]))
            if _anhedral:
                widgets.hint(_anhedral + " It is pinned, so the floor that "
                             "holds the SEARCHED row at 0 does not apply — "
                             "this is the value the run will fly.", "bad")

    def _repaint_row_values(labels=None):
        """Write the box's own numbers into the fields already on screen.

        The in-place twin of :func:`_retag_source`, and it exists for the
        same reason: a control that moves a row it does not live in — the
        aspect-ratio limit clips the span row, the pair's two areas write the
        split — may not call ``_render_box``, because that rebuilds the table
        and destroys the ``ui.number`` being typed into. Silent otherwise:
        the run would search the clipped box while the table showed the old
        one, which is this repo's own "the box shown is the box searched"
        failure one card over.
        """
        eff = config.effective_bounds(S)
        for label, fields in list(box_inputs.items()):
            if labels is not None and label not in labels:
                continue
            row = eff.get(label)
            if row is None or len(fields) != 2:
                continue
            for idx, field in enumerate(fields):
                want = widgets.shown(row[0][idx], BOX_DIGITS)
                if field.value != want:
                    field.value = want
            _retag_source(label)

    def _retag_source(label: str):
        """Move one row's provenance chip without rebuilding the row.

        ``_render_box`` is the only thing that draws the chip, and a typed
        bound may not call it — that would destroy the ``ui.number`` the
        digits are arriving in. So the one transition a typed bound causes,
        "default"/"recommended"/"mission" → "user", is written into the chip
        that is already on screen.
        """
        el = box_tags.get(label)
        row = config.effective_bounds(S).get(label)
        if el is None or row is None:
            return
        if label in config.fixed_rows(S):
            # a pinned row has no provenance to show: the band it carries is
            # whatever ``bounds_overrides`` widened to contain the pin, and
            # ``_box_row`` says "fixed" rather than naming a source
            widgets.retag(el, "fixed", theme.WARN)
            return
        source = str(row[1])
        widgets.retag(el, source,
                      _SOURCE_COLOR.get(source, theme.INK_FAINT))

    def _set_bound(label: str, idx: int, value):
        if value in (None, ""):
            return
        eff = config.effective_bounds(S)
        if label not in eff:
            return
        cur = [float(v) for v in eff[label][0]]
        # the rounded value we drew, coming back = nothing was typed; taking
        # it would turn a default row into an override that differs from the
        # solver's own box in the seventh decimal
        if widgets.is_echo(value, cur[idx], BOX_DIGITS):
            return
        cur[idx] = float(value)
        W["bounds"][label] = cur
        # a band the user typed is theirs: the mission may no longer move it
        session.bounds_source(S).pop(label, None)
        # ...and the row's own chip has to say so. It cannot be redrawn (it
        # shares the row with the input being typed into), so it is written
        # in place: without this the box went on tagging a band the user had
        # just typed as "default" until some unrelated edit rebuilt the table.
        _retag_source(label)
        # a coefficient typed by hand moves the band drawn above it, which
        # lives in a container of its own — so it can be redrawn without
        # taking the input the digits are arriving in with it.
        #
        # ...and so does every row the law is READ AGAINST: the taper it
        # multiplies (``_chord_baseline`` takes λ from the middle of that
        # row's band) and, for the second surface, the area and aspect ratio
        # that put a metre on it. Measured before this line existed: typing
        # taper's upper bound 1 -> 0.9 left the panel in the SAME view
        # drawing λ = 0.6 over a 64 % flyable band while the box underneath
        # had already moved to λ = 0.55 / 74 %.
        for group, rows in _chord_groups(config.effective_bounds(S)).items():
            if label in rows or label == CHORD_GROUPS[group][1] or (
                    group == ("", "_t") and label in ("S_t_m2", "AR_t")):
                _render_chord_law(group)
        if label == "b_m":
            # the size card quotes this row (span, area, the aspect ratio it
            # implies), and it lives in the TYPE view — a different container
            # from the input being typed into, so redrawing it cannot swallow
            # the rest of the number
            _render_size_note()
        if label == "l_t_m":
            # ...and the placement card quotes THIS one: where the surface is
            # placed, and whether that band has left the interval the family
            # was calibrated over. Same view, same reason it is safe.
            _render_arm_note()
        if label == api.TAIL_HEIGHT_KEY:
            _render_height_note(W["choices"].get("medium") == "water")
        _render_derived_geo()
        # the box view is the one being TYPED into — its derived containers
        # are redrawn by name above and the row's own provenance chip sits
        # in the same grid cell as the input, so it may not be rebuilt here
        ctx.refresh(("wing", "box"))

    def _set_row_on(label: str, on: bool):
        """Constrain this row, or release it.

        Releasing keeps the typed numbers — they are the user's answer to a
        question they may switch back on — and re-renders the whole box,
        because releasing a row also releases the section's pin on it and
        the derived-geometry panel underneath quotes the box.

        The SPAN of the wing-loading mode is the one row that cannot be
        released (its switch is disabled, ``_box_row``): "search the span
        over the solver's own box" is not an answer anybody wants — the
        published band is a statement about the family's own 10 m wing — and
        giving the span up entirely is the planform menu's question, not a
        switch buried in a table. Belt and braces, it is refused here too, so
        no caller can reach that state through the action registry.
        """
        if label == "b_m" and not on and session.span_is_searched(S):
            ui.notify("the span is what this mode searches — change the "
                      "planform to give it up", type="warning")
            return
        off = [str(k) for k in (W.get("bounds_off") or [])]
        if on and label in off:
            off.remove(label)
        elif not on and label not in off:
            off.append(label)
        else:
            return
        W["bounds_off"] = off
        ctx.log(f"{label}: "
                + ("constrained by this session again" if on else
                   "released — the solver's own box is searched"), "info")
        _render_box()
        ctx.refresh()

    def _set_row_fixed(label: str, on: bool):
        """Take this variable out of the search, or give it back.

        It opens at the MIDDLE of the band the row is searched in, which is
        the one value the session has already said something about — the
        published centre where nothing was typed, the centre of the user's
        own band where something was. A fixed row cannot also be released:
        "nothing constrains it" and "it is exactly this" are contradictory,
        so switching the fix on takes the release off.
        """
        if label == "b_m" and on and session.span_is_searched(S):
            ui.notify("the span is what this mode searches — choose the "
                      "fixed planform to state it instead", type="warning")
            return
        fixed = dict(W.get("fixed") or {})
        if on:
            row = config.effective_bounds(S).get(label)
            if row is None:
                return
            mid = 0.5 * (float(row[0][0]) + float(row[0][1]))
            fixed[label] = config.fixed_value(S, label, mid)
            W["bounds_off"] = [k for k in (W.get("bounds_off") or [])
                               if str(k) != label]
        elif label in fixed:
            del fixed[label]
        else:
            return
        W["fixed"] = fixed
        ctx.log(f"{label}: "
                + (f"fixed at {fixed[label]:.6g} — the optimiser searches "
                   f"one dimension fewer" if on else
                   "searched again"), "info")
        _render_box()
        ctx.refresh()

    def _set_fixed_value(label: str, value):
        """The value a fixed row is held at.

        Taken as typed, band or no band: a fixed row is not searched, so its
        band has stopped being a statement about anything, and the value is
        the user's answer to a real question. The band travels widened to
        contain it (``config.bounds_overrides``) so the run still gets one
        consistent statement. Redraws NOTHING that contains this input — a
        rebuilt ui.number loses the focus and swallows the rest of the
        number being typed.
        """
        if value in (None, ""):
            return
        fixed = dict(W.get("fixed") or {})
        if label not in fixed:
            return
        if widgets.is_echo(value, fixed[label], BOX_DIGITS):
            return
        fixed[label] = config.fixed_value(S, label, float(value))
        W["fixed"] = fixed
        m = CHORD_ROW.fullmatch(label)
        if m:
            # the band drawn above the coefficients is a picture of what the
            # search may reach, and a fixed coefficient is a point in it
            _render_chord_law((m.group(1) or "", m.group(3) or ""))
        # the "Fixed: taper = …" sentence at the head of the table quotes
        # exactly this number, in a container of its own for that reason
        _render_fixed_note()
        _render_derived_geo()
        ctx.refresh(("wing", "box"))

    def _set_span_searched(on: bool):
        """The wing-loading mode, as a switch: put the span in the design
        vector at the mission's W/S, or give the size back to the card.

        The planform menu is where a user answers this; the switch stays
        because that IS the question the mode asks, and everything that is
        not the menu (a preset, a test, another stage) asks it this way.
        """
        _set_planform("wing_loading" if on else "fixed")

    # ------------------------------------ a box RECOMMENDED for this mission
    #
    # The family's published box is the box its solvers were calibrated over,
    # and its size rows already follow the aeroplane. What none of it follows
    # is THIS MISSION: the same 0.2-1 taper and the same +-0.5 chord
    # coefficients are searched whether the wing carries 4.9 N at 12 m/s or
    # 653 N at 14.6, and the designs that fly one of those do not use the
    # same part of the box as the designs that fly the other.
    #
    # So the shell does not assert a recommendation from a table of rules. It
    # MEASURES one (``aerobo.recommend``): draw this box, evaluate, keep what
    # flew, and bound the best-scoring quarter of it. Every number offered is
    # this codebase's own physics at this mission, and it is reproducible —
    # same mission, same recommendation, because the Sobol seed is fixed and
    # quoted.
    #
    # It is OFFERED, never applied: the proposal is drawn beside the rows it
    # would replace and a second click takes it, which is the rule every
    # derived number in this shell follows.
    def _recommend_stamp() -> str:
        """WHAT a recommendation was measured over, as one comparable value.

        The proposal is a statement about a mission and a box, and both move
        under it: accept a new weight at stage 1, type a bound, choose a
        section that pins a row, and the bands on screen were measured over a
        problem that no longer exists. Stamping the measurement and comparing
        the stamp at draw time is what stops the card quoting it — the same
        rule the derived read-outs follow, applied to a number that is
        expensive enough to be worth keeping between renders.
        """
        try:
            d = config.cfg_dict(S)
        except Exception:      # noqa: BLE001 — a view, never fatal
            return ""
        # the SEARCH POLICY is not part of it: the optimiser and the budget
        # decide how the box is searched and not which designs fly this
        # mission, so changing them must not throw a measurement away.
        return repr(sorted((k, repr(v)) for k, v in d.items()
                           if k not in ("optimiser", "budget", "seed")))

    def _recommendation(scope: str):
        got = (W.get("recommend") or {}).get(scope)
        if got is None:
            return None
        if got.get("stamp") != _recommend_stamp():
            # measured over a problem this session has since changed
            W.setdefault("recommend", {}).pop(scope, None)
            return None
        return got

    def _measure_box() -> dict:
        """The raw measurement for THIS problem and mission, cached by stamp.

        One measurement, however many tables read it. The wing's table and the
        second surface's are two views of the same draws — the designs that
        flew are jointly admissible, which is the whole reason a per-surface
        probe would be wrong — so measuring once per table would pay twice for
        the same answer and, on a slow family, pay it in XFOIL sweeps.

        Raises. The caller decides what a configuration that cannot be
        measured looks like on ITS card.
        """
        stamp = _recommend_stamp()
        cache = (W.get("recommend") or {}).get("_raw")
        if cache and cache.get("stamp") == stamp:
            return copy.deepcopy(cache["got"])
        cfg = config.build_cfg(S)
        # ...through the api, not around it. Building the problem by hand
        # here skipped `sanitise_flags`, the wing objective's own refusal
        # and the PINS, and the composite branch then raised into the
        # caller's `except` — which logged one grey line and returned BEFORE
        # `_render_box()`, so the button moved not one pixel on screen.
        draws = 48 if sp().slow else RECOMMEND_DRAWS
        got = api.recommend_box(cfg, n=draws, points=_run_points())
        W.setdefault("recommend", {})["_raw"] = {
            "stamp": stamp, "got": copy.deepcopy(got)}
        return got

    def _unnarrowed_cfg():
        """``(cfg, stamp)`` — this configuration with the shell’s OWN
        measured bands taken back, WITHOUT touching the state on screen.

        The measurement may not be a ratchet: every pass has to measure over
        a box with no measured rows in it, or each pass narrows what the last
        pass narrowed, every step defensible on its own, and a session that
        changed one menu three times is searching a box nobody chose.

        The old way of guaranteeing that was ``drop_recommended_bounds``
        BEFORE measuring — which meant the box on screen fell back to the
        mission’s own band for the whole time the worker ran. Every menu
        toggle therefore read ``mission``, then ``recommended``, then a
        different pair of numbers, and a user reasonably read that as the
        shell changing its mind under them. Measuring over a cfg built here
        keeps the box on screen still until the answer lands, and then swaps
        it once.
        """
        d = config.cfg_dict(S)
        ov = dict(d.get("bounds_overrides") or {})
        want = session.size_band_defaults(S)
        for row, source in list(session.bounds_source(S).items()):
            if source != "recommended":
                continue
            if row == session.TAIL_SPAN_KEY:
                # a FLAG pair, not a band (session.TAIL_SPAN_FLAGS)
                flags = dict(d.get("flags") or {})
                for key in session.TAIL_SPAN_FLAGS:
                    flags.pop(key, None)
                d["flags"] = flags
            elif row in want:
                # a SIZE row returns to the MISSION’s band, not to the
                # family’s published one — the same rule
                # `drop_recommended_bounds` follows, for the same reason
                ov[row] = [float(want[row][0]), float(want[row][1])]
            else:
                ov.pop(row, None)
        d["bounds_overrides"] = ov or None
        stamp = repr(sorted((k, repr(v)) for k, v in d.items()
                            if k not in ("optimiser", "budget", "seed")))
        return api.RunConfig(**d), stamp

    def _run_points():
        """The FEASIBLE designs this session's own run already found, or None.

        A search is a better instrument than a uniform sample wherever the
        feasible set is a per-cent of the box — measured on the reported
        tandem session: 0 of 96 draws admissible, while the run that had
        already been paid for found 50 designs that fly. So the box can be
        measured from the run, and this is where the shell hands those
        designs over (``api.recommend_box(points=…)``).

        Only where the record is about THIS search: ``relax.banner`` is the
        same drift test the run card uses, and a non-empty banner means the
        record and the session have parted company — its designs are then
        about a different box and would be a recommendation for that one.
        """
        record = (S.get("run") or {}).get("record")
        if not record:
            return None
        try:
            if relax_mod.banner(S, record):
                return None
            return api.feasible_points_of(record)
        except Exception:          # noqa: BLE001 — a read-out, never fatal
            return None

    def _measure_unnarrowed() -> dict:
        """:func:`_measure_box`, over :func:`_unnarrowed_cfg`. Raises.

        Its own cache slot: ``_raw`` is keyed on the box ON SCREEN, which is
        the box the per-table path measures over, and the two are not the
        same measurement once a band has been written.
        """
        cfg, stamp = _unnarrowed_cfg()
        cache = (W.get("recommend") or {}).get("_auto_raw")
        if cache and cache.get("stamp") == stamp:
            return copy.deepcopy(cache["got"])
        got = api.recommend_box(cfg, n=48 if sp().slow else RECOMMEND_DRAWS,
                                points=_run_points())
        W.setdefault("recommend", {})["_auto_raw"] = {
            "stamp": stamp, "got": copy.deepcopy(got)}
        return got

    def _make_recommendation(scope: str):
        """Measure a box for this mission, for ONE surface's table."""
        try:
            cfg = config.build_cfg(S)
            got = _measure_box()
        except Exception as exc:      # noqa: BLE001 — a view, never fatal
            # SAID, not swallowed. An offer that cannot be measured is an
            # answer about this configuration and belongs on the card that
            # asked for it, beside the button.
            W.setdefault("recommend", {})[scope] = {
                "error": f"{type(exc).__name__}: {exc}",
                "stamp": _recommend_stamp()}
            ctx.log(f"recommendation: {exc}", "warn")
            ui.notify(f"the box could not be measured: {exc}", type="warning")
            _render_box()
            ctx.refresh()
            return
        eff = config.effective_bounds(S)
        # ...and only the rows THIS table owns. The measurement is over the
        # whole design vector, because the draws that flew are jointly
        # admissible and a per-surface probe could not know that — but the
        # proposal a table shows is the proposal a table can take.
        # the pair's two spans are unioned first, for the reason
        # `_auto_apply` gives: they are one question asked twice, and the
        # offer a table shows has to be the one the box would take
        measured = session.union_pair_spans(
            S, got["rows"], {lab: bnd for lab, (bnd, _s) in eff.items()})
        mine = {lab: band for lab, band in measured.items()
                if lab in eff and (bool(AFT_ROW.search(lab)) == (scope == "aft"))
                and not CHORD_ROW.fullmatch(lab)}
        # ...and only where the measurement actually SAYS something. A finite
        # sample bounds nearly every row a hair inside its own edges, and a
        # proposal listing those beside a row that halved is a proposal whose
        # one finding cannot be seen.
        rows, held = {}, 0
        for lab, band in mine.items():
            now = eff[lab][0]
            was = float(now[1]) - float(now[0])
            if was > 0.0 and (band[1] - band[0]) / was <= 1.0 - \
                    RECOMMEND_MIN_SHRINK:
                rows[lab] = band
            else:
                held += 1
        got["rows"] = rows
        got["n_held"] = held
        # ...and the second surface's SPAN, which is a flag pair rather than
        # a row and so could never appear in `rows`. Offered under the AFT
        # table only, because that is the table that owns it.
        if scope == "aft":
            band = _tail_span_recommendation(got)
            if band is not None:
                got["tail_span"] = [float(band[0]), float(band[1])]
            else:
                got["n_held"] = held + (1 if _tail_span_asked() else 0)
        got["stamp"] = _recommend_stamp()
        # ...and what THIS TABLE's proposal is worth, measured on the box it
        # would actually produce. The whole-vector figure is the wrong number
        # under a table that is offering two of its rows: it quotes a box
        # nobody is being offered, and it moves when the OTHER surface's rows
        # move. Same seed, same draw count, so it is comparable with the
        # figure for the box as it stands.
        if rows:
            got["frac_after"], got["best_after"] = api.recommendation_probe(
                cfg, rows, n=got["n"], seed=got["seed"])
        got.setdefault("stamp", _recommend_stamp())
        W.setdefault("recommend", {})[scope] = got
        _render_box()
        ctx.log(f"recommendation ({scope}): {got['n_admissible']} of "
                f"{got['n']} draws flew this mission; "
                f"{len(rows)} row(s) proposed", "info")
        ctx.refresh()

    def _take_recommendation(scope: str):
        """Take it. Written through ``W['bounds']`` — the same place a typed
        bound lands — and tagged ``"recommended"`` in the source column, so a
        number that appeared from nowhere still says where it came from.

        TAGGED, NOT UNTAGGED. This used to POP the row's source, which made a
        measured band indistinguishable from a typed one — and a typed row is
        never refreshed. So taking the recommendation for a 10 m wing and then
        stating a 1 m one in the mission left the box searching 8.95 – 14.79 m
        against a mission that said 1 m, with nothing on screen saying why,
        and pressing the button again re-measured a box the mission had
        already emptied and could only answer "no region". The tag is what
        makes :func:`session.drop_recommended_bounds` able to take it back.
        """
        got = _recommendation(scope)
        if not got:
            return
        for lab, band in got["rows"].items():
            W["bounds"][lab] = [float(band[0]), float(band[1])]
            session.bounds_source(S)[lab] = "recommended"
        # the second surface's span is a FLAG pair, so it lands where the
        # switch on its row writes (`_set_tail_span_on`) rather than in
        # `W["bounds"]` — and taking it switches the row on, because a band
        # on a row nobody enabled would be a number with no control holding it
        span = got.get("tail_span")
        if span:
            for key, value in zip(TAIL_SPAN_ROW, span):
                W["flags"][key] = float(value)
            # ...and marked, for the same reason every band above is: it was
            # measured over a mission, and a mission that moves must be able
            # to take it back rather than leave a stale pair nobody can see
            # the origin of
            session.bounds_source(S)[session.TAIL_SPAN_KEY] = "recommended"
        W.setdefault("recommend", {}).pop(scope, None)
        _render_box()
        _render_solver()
        ctx.log(f"took the recommended box ({scope}): "
                + ", ".join(
                    [f"{k} {v[0]:.4g}–{v[1]:.4g}"
                     for k, v in got["rows"].items()]
                    + ([f"{TAIL_SPAN_ROW[0]}/{TAIL_SPAN_ROW[1]} "
                        f"{span[0]:.4g}–{span[1]:.4g} m"] if span else [])),
                "info")
        ctx.refresh()

    # -------------------------------- the recommendation, APPLIED by default
    #
    # A box the mission empties is not a search. Measured on the free planform
    # at the shipped air mission, 11 of 128 draws over the published box fly
    # it (8.6 %) and 94 of 128 fly the box this measurement recommends
    # (73.4 %); the span row alone goes 6 - 40 m to 8.95 - 14.79 m, which is
    # the same row about six times narrower. All of that was behind a BUTTON,
    # so the 8.6 % box is what anyone who did not press it searched. It is now
    # the default: measured on arrival at this stage, written into the rows
    # nobody has answered themselves.
    #
    # THREE RULES, so "default" never means "decided for you":
    #
    # * a row you TYPED, a row the SECTION pinned, a row that is FIXED and a
    #   row that is switched OFF are answers already given. A default does not
    #   overrule an answer, so only default/mission/recommended rows are
    #   written, and the source column says which is which;
    # * "Reset to the solver's box" MEANS it — pressing it switches the
    #   automatic measurement off for this problem and mission, or the box
    #   would re-narrow under the cursor. The button below brings it back, and
    #   any change to the problem or the mission re-arms it;
    # * a SLOW family is not measured on arrival. 48 draws each holding an
    #   XFOIL sweep is minutes, and a stage that takes minutes to open is not
    #   a default anybody asked for. Those keep the button, and say so.
    #
    # AND IT IS NOT A RATCHET. Every pass measures over a box with no measured
    # rows in it (``session.drop_recommended_bounds`` first, always): without
    # that, each pass narrows what the last pass narrowed, every step of it
    # defensible on its own, and a session that changed one menu three times
    # would be searching a box nobody chose.
    def _auto_state() -> dict:
        return W.setdefault("auto_rec", {})

    def _auto_stamp() -> str:
        """WHAT the automatic measurement is a statement about.

        The recommendation's stamp MINUS the box, PLUS the pins. A band is
        part of the configuration, so stamping on one would re-measure every
        time a number was typed into the design box: a second of solver per
        keystroke, and — far worse — a repaint of the very table the cursor
        is in, which is how this shell loses half a typed number.

        What must re-measure is the PROBLEM, the MISSION and the rows that
        have left the design vector (a pinned row is not measured as free).
        What must not is a BOUND: that is the user narrowing the box
        themselves, and the answer to it is their row, not a new measurement.
        """
        try:
            d = config.cfg_dict(S)
        except Exception:      # noqa: BLE001 — a view, never fatal
            return ""
        return repr((sorted((k, repr(v)) for k, v in d.items()
                            if k not in ("optimiser", "budget", "seed",
                                         "bounds_overrides")),
                     sorted(config.fixed_rows(S))))

    def _auto_ready() -> bool:
        """Is this a moment to spend the measurement?

        The wing stage has to be REACHABLE — the mission is accepted and
        nothing about it is refused — because everything the measurement is
        about comes from the mission. Beyond that the answer is yes wherever
        the user is standing, so the box is already measured by the time they
        arrive; a slow family is the exception and waits to be looked at.
        """
        if session.stage_states(S)["wing"][0] == "locked":
            return False
        return S["ui"]["selected"] == "wing" or not sp().slow

    def _auto_wanted() -> bool:
        """Is this a configuration the shell measures without being asked?

        Every one of them, including the SLOW families. There is no button to
        fall back on any more, so a family that is not measured here is a
        family whose box is nobody's opinion about the aeroplane on screen —
        and the measurement is a background worker, so a slow one costs a
        spinner and a later repaint rather than a stage that will not paint.
        """
        a = _auto_state()
        if a.get("busy") or a.get("off"):
            return False
        return a.get("stamp") != _auto_stamp()

    def _empty_box_reasons(n: int = 64) -> list:
        """``[(count, reason)]`` — WHY nothing flew. Never raises.

        "There is no region to recommend" is a measurement, not an
        explanation, and the explanation is one probe away: the same kind of
        draws, split by the gate that refused each one. A user whose box is
        empty needs the name of the limit, not the fact of the emptiness.
        """
        try:
            got = api.box_refusal_probe(config.build_cfg(S), n=int(n))
        except Exception:          # noqa: BLE001 — a read-out, never fatal
            return []
        return [(int(r["n"]), str(r["reason"]))
                for r in (got.get("reasons") or [])[:3]]

    def _auto_apply(got: dict) -> list:
        """Write the measured bands into the rows nobody has answered.

        Through ``session.hold_the_stated_design`` first, always: the
        measurement is the min/max of the draws that flew and owes the wing
        on screen nothing, so on the shipped air mission it came back
        11.509 – 13.925 m over a mission stating 10 m — a box that refuses
        its own aeroplane. A row the widening then no longer narrows by
        ``RECOMMEND_MIN_SHRINK`` is simply not written, which is the honest
        outcome: the measurement had nothing usable to say about it.
        """
        eff = config.effective_bounds(S)
        fixed_now = config.fixed_rows(S)
        src = session.bounds_source(S)
        written = []
        # A PAIR GETS ONE SPAN BAND, the union of the two the measurement
        # drew (`session.union_pair_spans`): the front and the rear wing are
        # the same question asked twice, and a recommendation should contain
        # everything reasonable rather than cut each row down to the draws
        # that happened to land on it. Then the stated wing is held, as
        # always — the union runs first so the hold widens the band that is
        # actually going to be written.
        rows = session.union_pair_spans(
            S, got.get("rows") or {},
            {lab: bnd for lab, (bnd, _src) in
             config.effective_bounds(S).items()})
        rows = session.hold_the_stated_design(S, rows)
        # ...and WHY a row this mission was measured for was not written. A
        # span the user is asking about, measured and then held back by one of
        # the three rules below, reads on screen as "there is no
        # recommendation for the span" — which is the opposite of what
        # happened. Recorded per row, quoted by the card.
        held = _auto_state().setdefault("held", {})
        held.clear()
        spans = set(session.span_rows(S))
        for lab, band in rows.items():
            if lab in fixed_now or CHORD_ROW.fullmatch(lab):
                if lab in spans:
                    held[lab] = (list(band), "you have fixed this row")
                continue
            now = eff.get(lab)
            if now is None or now[1] not in ("default", "mission",
                                             "recommended"):
                if lab in spans and now is not None:
                    held[lab] = (list(band),
                                 f"the band on screen is yours ({now[1]}), "
                                 f"and a measurement never overwrites a "
                                 f"number you typed")
                continue
            was = float(now[0][1]) - float(now[0][0])
            if not (was > 0.0 and (float(band[1]) - float(band[0])) / was
                    <= 1.0 - RECOMMEND_MIN_SHRINK):
                if lab in spans:
                    held[lab] = (list(band),
                                 f"it is not narrower than the "
                                 f"{now[0][0]:.4g}–{now[0][1]:.4g} row you "
                                 f"already have — once widened to hold the "
                                 f"wing this mission states, there is nothing "
                                 f"left to narrow")
                continue          # the measurement says nothing about this row
            W["bounds"][lab] = [float(band[0]), float(band[1])]
            src[lab] = "recommended"
            written.append(lab)
        return written

    # ...and the SECOND SURFACE'S SPAN, which is REPORTED and not imposed.
    #
    # It is one of the four numbers this measurement is for — wing span, wing
    # area, tail span, tail area — and the first version of this wrote it into
    # the flag pair with the rest. Measured, that is a mistake: b_t is
    # sqrt(AR_t x S_t), so a span band is a DIAGONAL across the two rows the
    # box already bounds, and all it can do is cut their corners. On the
    # reference tail, writing it took the admissible fraction 0.727 -> 0.531
    # with the tail's planform fixed and 0.258 -> 0.180 with it designed, and
    # the best design over the box was IDENTICAL to fifteen digits in both
    # (17.85283560154081 and 17.855572726197906). It costs feasible draws and
    # buys nothing.
    #
    # Where AR_t is not a design variable it is worse than useless: with the
    # aspect ratio fixed, a span band is the AREA band restated, and restating
    # a constraint more tightly than it was measured is the silent extra
    # narrowing this shell exists not to do.
    #
    # So the number is measured and QUOTED — the span the designs that flew
    # this mission are actually drawn at — and the row stays the user's to
    # set. `_take_recommendation` still writes it, because that is a person
    # asking for it by name.
    def _auto_tail_span(got: dict):
        if not _tail_span_asked():
            return None
        try:
            band = _tail_span_recommendation(got)
        except Exception:          # noqa: BLE001 — a read-out, never fatal
            return None
        return None if band is None else [float(band[0]), float(band[1])]

    def _auto_recommend(force: bool = False) -> dict:
        """Measure this mission's box and make it the box. Never raises."""
        a = _auto_state()
        if force:
            a["off"] = False
        a.update(error=None, rows=[], empty=False, stripped=False,
                 reasons=[], n=None, n_admissible=None, tail_span=None,
                 frac_before=None, frac_after=None, best_after=None,
                 # WHICH measurement the bands came from (api.recommend_box):
                 # designs that fly, designs that merely solved, a gate's own
                 # closed-form suggestion, or a number stated outside the box
                 # that rules all of it out. A band quoted without this is a
                 # box the card cannot stand behind.
                 basis=None, escalated=False, n_solved=None, conflicts=[],
                 n_points=None)
        try:
            got = _measure_unnarrowed()
        except Exception as exc:   # noqa: BLE001 — a view, never fatal
            session.drop_recommended_bounds(S)
            a["error"] = f"{type(exc).__name__}: {exc}"
            a["stamp"] = _auto_stamp()
            ctx.log(f"the recommended box could not be measured: {exc}",
                    "warn")
            return a
        a.update(n=got.get("n"), n_admissible=got.get("n_admissible"),
                 frac_before=got.get("frac_before"),
                 empty=bool(got.get("empty")),
                 basis=got.get("basis"),
                 n_points=got.get("n_points"),
                 escalated=bool(got.get("escalated")),
                 n_solved=got.get("n_solved"),
                 conflicts=[str(c.get("text") or "")
                            for c in (got.get("conflicts") or [])],
                 stripped=bool(got.get("objective_stripped")))
        # NOT "empty" ANY MORE, but "nothing to write". A box where nothing
        # flew still has an answer — where the solver at least ran, or which
        # gate's own suggestion would clear it (api.recommend_box's ladder) —
        # and this used to return before any of it could be written, which is
        # the "there is no recommendation" the user was reading.
        # WHY nothing flew, asked whenever nothing did — including on the
        # rungs that still produced a band (`solved`, `gates`). The reasons
        # and the band answer different questions, and hanging the reasons off
        # "did we get rows" made a gate suggestion silence the explanation.
        if a["empty"]:
            a["reasons"] = _empty_box_reasons()
        if not (got.get("rows") or {}):
            session.drop_recommended_bounds(S)
            a["stamp"] = _auto_stamp()
            ctx.log(f"none of {got.get('n')} draws over this box fly this "
                    f"mission, and nothing measurable is left to narrow it "
                    f"with — the box is not narrowed", "warn")
            return a
        # the bands the LAST pass wrote go back now — one tick before the
        # new ones are written, so the swap is atomic on screen
        session.drop_recommended_bounds(S)
        written = _auto_apply(got)
        a["rows"] = written
        a["tail_span"] = _auto_tail_span(got)
        if written:
            # the probe re-searches BOX ROWS; the tail span is a flag pair
            # and reaches the build through `TailLimits` instead, so it is
            # already in the box this measures and is not a row to pass in
            rows = {k: W["bounds"][k] for k in written if k in W["bounds"]}
            try:
                a["frac_after"], a["best_after"] = api.recommendation_probe(
                    config.build_cfg(S), rows, n=int(got["n"]),
                    seed=int(got["seed"]))
            except Exception:      # noqa: BLE001 — a read-out, never fatal
                pass
            ctx.log("design box measured for this mission: "
                    + ", ".join(
                        f"{k} {W['bounds'][k][0]:.4g}–{W['bounds'][k][1]:.4g}"
                        if k in W["bounds"] else
                        f"{k} " + "–".join(
                            f"{float(W['flags'][f]):.4g}"
                            for f in TAIL_SPAN_ROW) + " m"
                        for k in written),
                    "info")
        a["stamp"] = _auto_stamp()
        return a

    def _auto_kick():
        """Start the measurement in the background, once per configuration."""
        a = _auto_state()
        if not _auto_wanted():
            return
        a["busy"] = True
        # stamped BEFORE the thread starts: the 0.5 s heartbeat repaints while
        # the worker runs, and a second kick would measure the same box twice
        a["stamp"] = _auto_stamp()
        threading.Thread(target=_auto_worker, daemon=True).start()

    def _auto_worker():
        try:
            _auto_recommend()
        finally:
            a = _auto_state()
            a["busy"] = False
            a["tick"] = time.time()

    def _render_auto(scope: str):
        """What the automatic measurement did, under the WING table only.

        One statement, not two: it writes rows in both tables from one set of
        draws, so printing it under each would read as two measurements that
        happen to agree.
        """
        if scope != "wing":
            return
        a = _auto_state()
        if a.get("busy"):
            with ui.row().classes("items-center gap-2"):
                ui.spinner(size="sm")
                ui.label(
                    "measuring the box this mission flies"
                    + (" — this family holds an XFOIL sweep, so it takes a "
                       "while; the box below is the family's published one "
                       "until it lands" if sp().slow else "")
                    + ("; the rows marked “recommended” below are "
                       "the LAST measurement, opened at the bottom onto this "
                       "mission\u2019s own band, until this one lands"
                       if session.measured_bands(S) else "")) \
                    .classes("hint")
            return
        if a.get("off"):
            widgets.hint(
                "You reset to the solver's published box, so this mission's "
                "own box is not being measured. It comes back on its own when "
                "the problem or the mission changes.")
            ui.button("Measure this mission's box", icon="auto_awesome",
                      on_click=lambda: (_auto_recommend(force=True),
                                        _render_box(), _render_solver(),
                                        ctx.refresh())) \
                .props("outline dense no-caps")
            return
        if a.get("error"):
            widgets.hint(f"the box could not be measured for this mission: "
                         f"{a['error']}. The solver's own published box "
                         f"stands.", "warn")
            return
        # BEFORE every branch below, because each of them returns: a span
        # this mission was measured for and the shell may not write is the
        # thing the "no recommendation for the free span" report was about,
        # and hanging it off the success path is how it stayed invisible on
        # exactly the sessions that needed it.
        _render_held_spans()
        if a.get("empty") and not a.get("rows"):
            widgets.hint(
                f"None of {a.get('n')} draws over this box fly this mission"
                + (" (measured twice, the second time with three times the "
                   "draws)" if a.get("escalated") else "")
                + ", so there is nothing to measure a narrower box FROM — the "
                  "box is left exactly as the family publishes it. This is a "
                  "conflict between the box and the mission, not a missing "
                  "recommendation.", "warn")
            for count, reason in (a.get("reasons") or []):
                widgets.hint(f"{count} of the draws: {reason}")
            # ...and where the two cheap gates rule the box out in CLOSED
            # FORM, the finding IS the recommendation: it names the number to
            # change and what it would have to be. No sampling can improve on
            # an inequality, and a card that stayed silent here was leaving
            # the one actionable sentence unsaid.
            for text in (a.get("conflicts") or []):
                widgets.hint(text, "warn")
            if a.get("basis") == "mission":
                cap = session.ws_over_ceiling(S)
                if cap is not None:
                    ui.button("Open the mission's wing loading",
                              icon="north_east",
                              on_click=lambda: (ctx.select("mission",
                                                           "operating"),
                                                ctx.refresh())) \
                        .props("outline dense no-caps")
            return
        if not a.get("rows"):
            if a.get("n_admissible") is None:
                return
            widgets.hint(
                f"{a.get('n_admissible')} of {a.get('n')} draws over this box "
                f"fly this mission, and the best of them use every row over "
                f"essentially its full width — so the published box already "
                f"IS this mission's box. Nothing was narrowed.")
            return
        if a.get("basis") == "run":
            widgets.hint(
                f"Not one of {a.get('n')} draws over this box flew this "
                f"mission — but the run you have already paid for did, and "
                f"the rows marked “recommended” below are drawn around the "
                f"{a.get('n_points')} best designs IT found. That is a "
                f"stronger measurement than the sample, not a weaker one: "
                f"every point in it was evaluated. Re-running from this box "
                f"searches where your own answer lives.", "ok")
            return
        if a.get("basis") == "solved":
            widgets.hint(
                f"NOTHING in this box flies this mission — but {a.get('n_solved')} "
                f"of {a.get('n')} draws reached their solver and missed a "
                f"margin, and the rows marked “recommended” below are drawn "
                f"around THOSE. It is where this mission's physics at least "
                f"runs, offered as a starting point and not as a feasible "
                f"box; the reach and the constraint cards say what is still "
                f"being missed.", "warn")
            return
        if a.get("basis") == "gates":
            widgets.hint(
                "Not one draw over this box reached a solver, so the box was "
                "not measured but SOLVED: the rows below are what the gate "
                "that refused everything can accept, in closed form.", "warn")
            for text in (a.get("conflicts") or []):
                widgets.hint(text)
            return
        before, after = a.get("frac_before"), a.get("frac_after")
        widgets.hint(
            f"Measured for this mission and APPLIED: {len(a['rows'])} row(s) "
            f"marked “recommended” below were narrowed around the "
            f"designs that actually flew it, each one still open far enough "
            f"to hold the wing the mission states"
            + (f" — {after:.0%} of draws over the box you now have fly this "
               f"mission, against {before:.0%} over the family's published "
               f"one." if (before is not None and after is not None)
               else ".")
            + " Type in any row and it becomes yours; “Reset to the "
              "solver\u2019s box” puts the published box back and "
              "stops this.", "ok")
        # ...and the caveat this project OWES its own study. It moved here
        # from the offer card with the numbers: a narrower box is a smaller
        # search, and over 42 seeds no objective gain was measured from
        # narrowing a band — the tightest arm was the worst of them. What is
        # claimed above is FEASIBILITY (a box most of whose draws are refused
        # before their solver spends a budget on nothing), and that is a
        # different claim from a better answer.
        widgets.hint(
            "A narrower box is a SMALLER SEARCH, not a better answer. What "
            "this buys is that the draws are spent on designs that fly: over "
            "42 seeds this project measured no objective gain from narrowing "
            "a band, and the tightest arm of that study was the worst of "
            "them.")
        if a.get("stripped"):
            widgets.hint(
                "Ranked on this family\u2019s own L/D, not on your composite "
                "J: the composite needs a frozen normalisation band and none "
                "has been measured yet. Measure the band on the objective "
                "card to rank on J instead.", "warn")

    def _render_held_spans():
        """A SPAN this mission was measured for, and why it is not on the row.

        The measurement bounds every row it draws, including the span, and
        three rules can then keep that band off the table: the row is fixed,
        the band on it is the user's own, or the band is no narrower than
        what is already there. Each is correct and each is invisible, and the
        three together are read as "there is no recommendation for the span".
        So the number is said anyway — quoted, never imposed.
        """
        for lab, got in (_auto_state().get("held") or {}).items():
            try:
                band, why = got
            except (TypeError, ValueError):      # a state written by an
                continue                         # older shell
            widgets.hint(
                f"Measured for this mission: the designs that flew it draw "
                f"{lab} at {float(band[0]):.4g} – {float(band[1]):.4g} m. It "
                f"is quoted and not applied, because {why}.")

    def _render_auto_tail_span():
        """The second surface's SPAN over the measured box — quoted, not set.

        Drawn under the table that owns the row, so the number the user asked
        for is beside the control that could impose it, with the measurement
        that says imposing it is not worth doing.
        """
        a = _auto_state()
        band = a.get("tail_span")
        if not band or a.get("busy") or a.get("off"):
            return
        name = session.second_surface_name(S) or "second surface"
        widgets.hint(
            f"Measured: the designs that flew this mission draw the {name} "
            f"at {band[0]:.4g} \u2013 {band[1]:.4g} m across "
            f"(b = \u221a(AR\u00b7S), off the area row above). It is quoted "
            f"and NOT written into the span row: a span band is a diagonal "
            f"across the area and aspect-ratio rows this box already bounds, "
            f"and measured here it took the admissible fraction 73 % to 53 % "
            f"while leaving the best design identical to fifteen digits. "
            f"Switch the row on below if you want it as a constraint anyway.")

    def _render_recommendation(scope: str):
        """What this mission's own box is, drawn under the table it is for.

        THERE IS NO BUTTON. A measurement offered behind one is a measurement
        most sessions never see: the numbers that matter here — the wing's
        span and area, the second surface's span and area, and how far apart
        the two are — were on the family's published rows, which are a
        statement about the family's own 10 m reference aeroplane and about no
        mission at all. So the box measures itself for the aeroplane on
        screen, every design gets its own, and the card says what it did.

        `_make_recommendation` and `_take_recommendation` survive as ACTIONS
        (the shell registers both) because they are the same machinery with
        the per-table filtering the aft card needs, and because a test drives
        the offer path. Nothing on screen presses them.
        """
        _render_auto(scope)
        if scope == "aft":
            _render_auto_tail_span()

    def _reset_box():
        W["bounds"] = {}
        W["bounds_off"] = []
        # ...and any recommendation with them: it was measured over the box
        # that just went away, and a proposal quoting a box nobody is
        # searching is the stale-read-out trap this shell keeps falling into
        W["recommend"] = {}
        # ...and the AUTOMATIC one is switched off for this problem and
        # mission. "Reset to the solver's box" is a request for the published
        # box, and a shell that measured a narrower one back over it half a
        # second later would be arguing with the button that was just pressed.
        # Either change re-arms it (`_auto_wanted` compares the stamp), and
        # the card carries a button that turns it back on here and now.
        _auto_state().update(off=True, stamp=None, rows=[], busy=False,
                             error=None, empty=False, reasons=[])
        # ...and every fixed row with them: "the solver's box, bit-for-bit"
        # is a statement about the whole design vector, and a variable this
        # session took out of it is the largest departure from that box there
        # is.
        W["fixed"] = {}
        # ...except the span, which has no solver's-own box worth resetting
        # to: the family's published band is a statement about the family's
        # own 10 m wing. Reset it to the fractional band around the wing this
        # session flies, which is where switching the row on put it.
        W["bounds_source"] = {}
        session.write_size_bands(S)
        _render_box()
        ctx.log("design box reset", "info")
        ctx.refresh()

    # ---------------------------------------------------- view: solver
    # =============================================== what the search maximises
    def _score():
        return session.wing_score_state(S)

    def _forget_score() -> None:
        """Drop the criteria report the moment a NEW search starts.

        Nothing else clears it: ``_score_worker`` is the only writer and it
        only runs on a record that carries a design. So a run launched over
        a finished one kept showing the finished one's rows — "beats 93 % of
        the box" beside "best objective —" — for its whole duration. The
        band and the weights are untouched; they are the user's settings,
        not this run's answer.
        """
        _score().update(report=None, report_error=None, report_for=None,
                        scoring=False)

    def _set_wing_objective(name: str):
        session.set_wing_objective(S, str(name))
        # the live panel follows the objective: plotting "L/D" as the trace
        # of a run maximising J is the same drift the ``f_lod`` split exists
        # to stop, one panel further down. It follows it by FORGETTING the
        # selection rather than by naming a key — the catalogue leads with
        # whatever this run spells its objective (``metrics.live_metric_
        # defaults`` puts ``composite`` first wherever the breakdown reports
        # it), so a new objective needs no second place to be listed.
        live_cfg.update(keys=[], picked=False)
        ctx.log("this search will maximise "
                + ("the composite of your criteria"
                   if str(name) == "composite"
                   else "the family's own objective (L/D)"), "ok")
        _render_solver()
        # WHAT is maximised changes the design vector stage 1 estimates over
        # (the composite adds no variable but does add the band it is
        # normalised against, and the plan's own label follows it), so the
        # Search tab's estimate has to follow. Without this the objective
        # radio left stage 1 quoting the previous objective's plan.
        ctx.refresh()

    def _set_wing_weight(key: str, value):
        session.set_wing_weight(S, key, value)
        lab = weight_labels.get(key)
        if lab is not None:
            lab.set_text(f"{_score()['weights'][key]:.2f}")
        # a moved slider makes the set the USER's (session.set_wing_weight),
        # and the two sentences under the card say whose it is and whether
        # the run would refuse it. They live in their own container, so they
        # can follow the slider without the slider being rebuilt under it.
        _render_weights_source()

    def _use_recommended_wing_weights():
        # the preset with whatever this family cannot report already zeroed,
        # so "use the recommended weights" cannot hand back a set the run
        # refuses (session.recommended_wing_weights)
        session.set_wing_weights(S, session.recommended_wing_weights(S),
                                 "recommended")
        dropped = session.wing_band_uncovered(S)
        ctx.log(f"criterion weights set to the recommended "
                f"“{session.WING_WEIGHT_PRESET}” set"
                + (f" (without {', '.join(dropped)} — not reported here)"
                   if dropped else ""), "ok")
        _render_solver()

    def _band_worker(n: int, seed: int):
        """Measure the band over the box, in the background.

        The one part of the composite that costs evaluations, so it reports
        its progress and never blocks the shell.
        """
        sc = _score()
        sc.update(measuring=True, error=None, progress=0.0)
        try:
            d = config.cfg_dict(S)
            box = session.wing_band_box(S)

            def progress(i, total):
                sc["progress"] = float(i) / max(1, int(total))

            payload = api.wing_score_reference(
                d["problem_name"], flags=d["flags"],
                mission_kwargs=d["mission_kwargs"],
                bounds_overrides=d["bounds_overrides"],
                # a FIXED row does not vary in the population the band is
                # measured over, exactly as it does not vary in the search
                pinned=d.get("pinned"),
                # ...and keep drawing until n samples have FLOWN, rather than
                # taking whatever flies out of n draws. Without it the band
                # REFUSED boxes the search solves: on the family a user
                # reported, 2 of 32 draws flew and the composite could not be
                # measured at all, while a screened search on the same box
                # found a design in 39 of 72 evaluations
                # (aerobo.wing_score.sample_reference).
                screen=True,
                n=int(n), seed=int(seed), progress=progress)
            session.set_wing_band(S, payload, box)
        except Exception as exc:      # noqa: BLE001 — a form, never fatal
            sc["error"] = f"{type(exc).__name__}: {exc}"
            session.set_wing_band(S, None, None)
        finally:
            sc["measuring"] = False
            sc["progress"] = None
            sc["stamp"] = time.time()

    def _measure_band():
        sc = _score()
        if sc.get("measuring"):
            ui.notify("the band is already being measured", type="info")
            return
        n = int(sc.get("samples") or 32)
        ctx.log(f"measuring the normalisation band — {n} evaluations of "
                f"{sp().display}", "ok")
        threading.Thread(target=_band_worker, args=(n, int(W["seed"])),
                         daemon=True).start()
        _render_solver()

    def _set_band_samples(value):
        try:
            n = max(8, int(float(value)))
        except (TypeError, ValueError):
            return
        _score()["samples"] = n

    def _band_cost_note(n: int) -> str:
        """What the sweep will cost, in the units the user already knows."""
        slow = sp().slow
        return (f"{n} evaluations of {sp().display}"
                + (" — this family holds an XFOIL sweep per evaluation, so "
                   "budget minutes, not seconds"
                   if slow else " — milliseconds each for this family"))

    def _ratchet_note():
        note = ws_ratchet_note(S)
        if note is None:
            return
        warn, follow_up = note
        widgets.hint(warn, "warn")
        if follow_up:
            widgets.hint(follow_up)

    def _render_objective():     # noqa: PLR0915
        sc = _score()
        # every label ``_set_wing_weight`` writes into belongs to the card
        # about to be rebuilt (or, on the L/D branch, to one that is about to
        # stop existing), so the old references are dead. Kept alive across a
        # repaint they wrote the typed number into a deleted element —
        # nicegui's "an element has been deleted but is still being used" —
        # and the number beside the slider never moved. The provenance
        # container under the sliders goes the same way — on the L/D branch
        # it is not re-made at all, so a slider driven while L/D is selected
        # cleared a container that no longer exists.
        weight_labels.clear()
        boxes.pop("weights_source", None)
        with widgets.group_box("Objective"):
            widgets.select_field(
                "maximise", WING_OBJECTIVE_CHOICES(),
                str(sc.get("objective") or "lod"),
                lambda e: _set_wing_objective(str(e.value)),
                tip="the ONE number the optimiser improves")
            _ratchet_note()
            if str(sc.get("objective")) != "composite":
                widgets.hint(
                    "The family's own objective: L/D at the trimmed design "
                    "lift (payload L/D where the size is searched). Your "
                    "criteria are still measured after the run — the block "
                    "under the trace says what maximising this one number "
                    "cost the other nine.")
                return
            _render_wing_weights()
            _render_band()

    def _render_wing_weights():
        sc = _score()
        # criteria THIS family never reports. Their sliders are shown and
        # disabled rather than hidden: the ten criteria are the vocabulary
        # the whole report is written in, and a card that silently dropped
        # two of them would read as "this wing has no stall lift" instead of
        # "nothing here measures one". A weight on one of these is what the
        # composite raises on, so the slider that cannot be honoured is the
        # slider that cannot be moved.
        uncovered = set(session.wing_band_uncovered(S))
        with ui.column().classes("gap-1 w-full"):
            for key, (label, why) in wing_weight_meta().items():
                dead = key in uncovered
                with ui.row().classes("w-full items-center gap-2 no-wrap"):
                    lab = ui.label(label).classes("field-label") \
                        .style("min-width:190px"
                               + (";opacity:0.45" if dead else ""))
                    lab.tooltip(f"{why}\n\nNot reported by this family, so "
                                "it cannot be weighted here." if dead else why)
                    sl = ui.slider(min=0.0, max=1.0, step=0.05,
                                   value=float(sc["weights"].get(key, 0.0)),
                                   on_change=lambda e, k=key:
                                   _set_wing_weight(k, e.value)) \
                        .props("dense").classes("grow")
                    if dead:
                        sl.props("disable")
                    out = ui.label("—" if dead
                                   else f"{sc['weights'].get(key, 0.0):.2f}") \
                        .classes("readout").style(
                            "min-width:34px;text-align:right")
                    weight_labels[key] = out
        if uncovered:
            names = ", ".join(wing_weight_meta()[k][0].split(" [")[0]
                              .replace("  (lower is better)", "")
                              for k in wing_weight_meta() if k in uncovered)
            widgets.hint(f"This family reports no {names}, so the band has "
                         f"nothing to normalise them against and they are not "
                         f"weighable here. The other criteria are unaffected — "
                         f"weights are normalised over the ones that count.")
        widgets.hint("Weights are normalised before scoring, so only their "
                     "ratios matter. A criterion at zero is not scored at "
                     "all — it is not a floor, and nothing refuses a design "
                     "for it.")
        # ...and the two things a MOVED SLIDER changes go in a container of
        # their own. A slider may not redraw the card it sits in (it would be
        # destroyed under the cursor mid-drag), so without this the card went
        # on saying "These are the recommended “cruise” weights … move any
        # slider and they become yours" after a slider had been moved and the
        # set had already become the user's.
        boxes["weights_source"] = ui.column().classes("w-full gap-1")
        _render_weights_source()

    def _render_weights_source():
        """Whose weight set this is, and whether the run would refuse it."""
        holder = boxes.get("weights_source")
        if holder is None:
            return
        holder.clear()
        with holder:
            refused = session.wing_weights_refused(S)
            if refused:
                widgets.hint(
                    "These weights would be refused by the run: "
                    + ", ".join(refused)
                    + " carry weight but this family does not report them. "
                      "Set them to zero, or re-measure the band on the box "
                      "you are searching.", "warn")
            if session.wing_weights_are_recommended(S):
                widgets.hint(f"These are the recommended “"
                             f"{session.WING_WEIGHT_PRESET}” weights: cruise "
                             f"L/D first, then the CL it stalls at, the load "
                             f"path, span efficiency and a buildable chord. "
                             f"Move any slider and they become yours.")
            else:
                widgets.hint("These are YOUR weights — nothing in this shell "
                             "moves them again.")
                ui.button("use the recommended weights", icon="tune",
                          on_click=_use_recommended_wing_weights) \
                    .props("flat dense no-caps")

    def _render_band():     # noqa: PLR0915
        sc = _score()
        with widgets.group_box("Normalisation band"):
            if sc.get("measuring"):
                with ui.row().classes("items-center gap-2"):
                    ui.spinner(size="sm")
                    pct = sc.get("progress")
                    ui.label("sweeping the design box"
                             + (f" — {100.0 * float(pct):.0f}%"
                                if pct else "")).classes("hint")
                return
            band = sc.get("band") or {}
            if band:
                with ui.row().classes("w-full items-start gap-4 no-wrap"):
                    widgets.readout("measured",
                                    f"{band.get('n_feasible', 0)}/"
                                    f"{band.get('n_samples', 0)} flew")
                    widgets.readout("criteria",
                                    str(len(band.get("bounds") or {})))
                    widgets.readout("band id", str(band.get("sha", ""))[:8])
                stale = session.wing_band_stale(S)
                if stale:
                    widgets.hint(f"This band is stale — {stale}. J stays a "
                                 f"fixed function of the design, but it is "
                                 f"normalised against a region this search "
                                 f"no longer visits: measure it again.",
                                 "warn")
                with ui.expansion("the band, criterion by criterion") \
                        .props("dense").classes("w-full"):
                    ui.table(columns=[
                        {"name": "metric", "label": "criterion",
                         "field": "metric", "align": "left"},
                        {"name": "lo", "label": "p5", "field": "lo",
                         "align": "right"},
                        {"name": "hi", "label": "p95", "field": "hi",
                         "align": "right"}],
                        rows=[{"metric": k, "lo": f"{v[0]:.4g}",
                               "hi": f"{v[1]:.4g}"}
                              for k, v in (band.get("bounds") or {}).items()],
                        row_key="metric") \
                        .classes("w-full").props("dense flat bordered")
            else:
                widgets.hint("No band yet. The composite needs one before it "
                             "can run: a wing has no library to normalise "
                             "against, so the 0-100 scale is measured over "
                             "YOUR design box — its centre plus a "
                             "low-discrepancy sweep of it — and then frozen "
                             "for the whole search.", "warn")
            if sc.get("error"):
                widgets.hint(sc["error"], "bad")
            with ui.row().classes("items-center gap-2 no-wrap"):
                ui.button("measure the band", icon="straighten",
                          on_click=_measure_band) \
                    .props("outline dense no-caps")
                widgets.number_field(
                    "samples", int(sc.get("samples") or 32),
                    lambda e: _set_band_samples(e.value), step=8,
                    width="w-24",
                    tip="how many box samples the band is measured over")
            widgets.hint(_band_cost_note(int(sc.get("samples") or 32))
                         + ". Paid once per box, not once per run.")
            widgets.hint(
                "Frozen is the point: a band recomputed as the search goes "
                "would move under the optimiser — best-so-far would stop "
                "being monotone and two seeds would not be comparable. A "
                "design outside the band scores past its end rather than "
                "saturating.")

    def _render_recommended_search(eff: dict):
        """The strategy and the budget when the MISSION owns them (V3.5).

        Read-outs, not fields. The numbers are a function of this stage's own
        design vector and of what it is maximising, so a control here would
        be a second answer to a question stage 1 has already asked — and the
        one thing a user does want to do from here, take the numbers over, is
        the button at the bottom rather than a silent edit.
        """
        plan = eff.get("plan")
        with widgets.group_box("Search strategy · recommended"):
            widgets.kv("optimiser",
                       api.OPTIMISER_SPECS[eff["optimiser"]].display)
            if eff["optimiser"] == "bo" and not sp().is_constrained:
                widgets.kv("acquisition", str(eff.get("acqf") or "logei"))
            if eff.get("n_init"):
                widgets.kv("Sobol seed", f"{int(eff['n_init'])} evaluations",
                           tip="how much of the budget is spent mapping the "
                               "box before the surrogate starts choosing")
            if plan is not None and eff["optimiser"] == "bo_slsqp" \
                    and plan.optimiser == "bo":
                # NOT the "family does not offer it" case below: the family
                # offers plain BO perfectly well and a later measurement beat
                # it. Said out loud, because a recommendation quietly replaced
                # is a recommendation nobody can audit.
                widgets.hint(
                    "The budget study recommends “bo”, and it was fitted "
                    "before this arm existed. A later study on certified "
                    "optima put BO-then-SLSQP above "
                    "pure BO on 42 of 54 independent runs, so that is what "
                    "flies; the budget is still the recommended one.", "info")
            elif plan is not None and plan.optimiser != eff["optimiser"]:
                widgets.hint(
                    f"The study's winner for this class is "
                    f"“{plan.optimiser}”, which this family does not offer "
                    f"(it is constrained); the budget is still the "
                    f"recommended one.", "warn")
        with widgets.group_box("Budget · recommended"):
            with ui.row().classes("w-full items-start gap-4 flex-wrap"):
                widgets.readout("budget", str(int(eff["budget"])), "evals")
                widgets.readout("design variables",
                                str(plan.dim) if plan else "—")
                widgets.readout("expected",
                                plan.est_text if plan else "—")
            widgets.number_field(
                "seed", W["seed"], lambda e: _set_int("seed", e.value, 0),
                step=1, width="w-24",
                tip="the recommendation sizes the search; WHICH search it is "
                    "is still yours")
            if plan is not None:
                widgets.hint(plan.why + ".")
            if session.search_state(S).get("stop_when_converged") \
                    and plan is not None and plan.patience:
                reach = session.convergence_rule_reach(S)
                if reach and not reach["can_fire"]:
                    # ...and where it CANNOT fire, say that instead. The
                    # published patience is 40 against recommended budgets of
                    # 17-53, so at these efforts the plateau test never has
                    # 40 evaluations behind it to compare against and the
                    # switch is inert. Promising a stop that cannot happen is
                    # how a user reads a full-budget run as "it never
                    # converged".
                    widgets.hint(
                        f"At this budget that switch cannot fire: the rule "
                        f"compares against the best design "
                        f"{reach['patience']} evaluations ago, so it needs "
                        f"more than {reach['needs'] - 1} and this run has "
                        f"{reach['budget']}. Raise the effort (or the "
                        f"budget) past {reach['needs']} to give it something "
                        f"to compare against.", "warn")
                else:
                    widgets.hint(
                        f"It will stop before that if the best design has not "
                        f"improved by {plan.tol:.1%} of the span it has "
                        f"covered in {plan.patience} evaluations — the "
                        f"partial result is kept.")
            with ui.row().classes("items-center gap-2 no-wrap"):
                ui.button("use my own values", icon="edit",
                          on_click=lambda: ctx.act("adopt_search_values")) \
                    .props("flat dense no-caps")
                ui.button("stage 1 · Search & budget", icon="north_east",
                          on_click=lambda: ctx.select("mission", "search")) \
                    .props("flat dense no-caps")

    def _render_own_strategy(names):
        """The strategy the USER picked — the pre-V3.5 card, unchanged."""
        with widgets.group_box("Search strategy"):
            for n in names:
                o = api.OPTIMISER_SPECS[n]
                selected = n == W["optimiser"]
                row = ui.row().classes(
                    "w-full items-start gap-2 no-wrap").style(
                    "padding:3px 4px;border-radius:2px;"
                    + (f"background:{theme.ACCENT_FILL}" if selected else ""))
                mark = theme.ACCENT if selected else theme.INK_FAINT
                with row:
                    ui.icon("radio_button_checked" if selected
                            else "radio_button_unchecked") \
                        .style(f"color:{mark}")
                    with ui.column().classes("gap-0"):
                        ui.label(o.display).classes("readout")
                        if o.note:
                            widgets.hint(o.note)
                row.on("click", lambda _, nn=n: _set_opt(nn))
            # a list that silently got shorter is the kind of thing nobody
            # notices — and since the shell now OPENS on the chord law, the
            # exact-gradient solver is missing before anyone has chosen
            # anything. Say what the freedom costs and how to get it back.
            _lost_note = v1.chord_law_optimiser_note(W["problem"])
            if _lost_note:
                widgets.hint(_lost_note)
            # The same rule, for the other reason a name can be absent: this
            # MACHINE has no PyTorch. api.compatible_optimisers drops what
            # cannot run; this is the sentence that says so, so the user who
            # came for Bayesian optimisation reads why here rather than
            # meeting it as a failure part-way into a search.
            _no_torch = api.torch_optimiser_note()
            if _no_torch:
                widgets.hint(_no_torch)
            if W["optimiser"] == "bo" and not sp().is_constrained:
                widgets.select_field(
                    "acquisition", list(v1.ACQF_CHOICES), W["acqf"],
                    lambda e: W.__setitem__("acqf", e.value),
                    tip="how the surrogate proposes the next design to "
                        "evaluate")
            if sp().has_blocks and W["optimiser"] == "blocks":
                widgets.hint(
                    "The portfolio runs each declared block with its own "
                    "sub-optimiser; the defaults are the published ones.")
            widgets.hint("These are YOUR search settings. The measured "
                         "recommendation is at stage 1 · Search & budget.")

    def _render_own_budget():
        with widgets.group_box("Budget"):
            widgets.number_field(
                "budget", W["budget"],
                lambda e: _set_int("budget", e.value, 2),
                unit="evals", step=1, width="w-24")
            widgets.number_field(
                "seed", W["seed"],
                lambda e: _set_int("seed", e.value, 0), step=1, width="w-24")
            widgets.number_field(
                "repeat seeds", W["n_seeds"],
                lambda e: _set_int("n_seeds", e.value, 1),
                unit="runs", step=1, width="w-24",
                tip="queue seeds base…base+N−1 for a median and a band "
                    "instead of a single trace")
            est = _estimate()
            if est:
                widgets.hint(est)

    def _render_solver():     # noqa: PLR0915
        box = ctx.views[("wing", "solver")]
        box.clear()
        names = api.compatible_optimisers(W["problem"])
        if W["optimiser"] not in names:
            W["optimiser"] = session.default_optimiser(W["problem"])
        eff = session.effective_wing_search(S)
        recommended = eff["source"] == "recommended"
        with box:
            with ui.row().classes("w-full items-start gap-3 no-wrap"):
                with ui.column().classes("gap-3").style("flex:3 1 0;"
                                                        "min-width:0"):
                    # WHAT is being maximised comes before HOW it is searched:
                    # a strategy is a means, and the scalar is the question
                    _render_objective()
                    if recommended:
                        _render_recommended_search(eff)
                    else:
                        _render_own_strategy(names)
                with ui.column().classes("gap-3").style("flex:2 1 0;"
                                                        "min-width:0"):
                    if not recommended:
                        _render_own_budget()
                    # NO cant card here any more. It is GEOMETRY — the shape
                    # the wing is built in — and it was on the Solver tab
                    # because that is where it was written, not because that
                    # is where it is asked. A user looking for "can I give
                    # this wing dihedral?" opens Wing type, finds a tail
                    # layout, a tip device and a planform, concludes the
                    # answer is no, and never reaches a third tab whose
                    # heading is about optimisers. It is now in the Wing
                    # type card's right column, beside the derived solver
                    # that decides whether it can be scored at all.
                    _physics_panel()

            # BEFORE the button, because it is about whether pressing it can
            # return anything: a box the mission provably empties spends the
            # whole budget on draws refused before their solver and comes
            # back with "no solution was found"
            _render_size_conflicts()

            with ui.row().classes("w-full items-center gap-2"):
                ui.button("Launch", icon="play_arrow", on_click=launch) \
                    .props("unelevated dense no-caps color=primary")
                widgets.explain(
                    ui.button("Cancel", icon="stop", on_click=cancel)
                    .props("outline dense no-caps"),
                    "Stops after the evaluation in flight and KEEPS the best "
                    "design found so far — it is scored, drawn and "
                    "exportable like any other result, and labelled as the "
                    "partial search it is.", title="Cancel")
                ui.space()
                ui.button("Copy the reproduce snippet", icon="content_copy",
                          on_click=lambda: ctx.act("snippet")) \
                    .props("flat dense no-caps")

            with widgets.group_box("What will run"):
                d = config.cfg_dict(S)
                widgets.kv("problem", sp().display)
                widgets.kv("optimiser",
                           api.OPTIMISER_SPECS[d["optimiser"]].display)
                widgets.kv("mission fields sent",
                           ", ".join(f"{k}={v:g}"
                                     for k, v in d["mission_kwargs"].items())
                           or "none (published defaults, bit-for-bit)")
                widgets.kv("flags", ", ".join(sorted(d["flags"])) or "none")
                # ...and WHOSE those rows are. The count alone reads as "you
                # typed N bounds" on a box whose size rows the shell derived
                # and whose narrowed rows a measurement wrote.
                srcs = config.effective_bounds(S)
                measured = sum(1 for lab in (d["bounds_overrides"] or {})
                               if srcs.get(lab, (None, ""))[1] == "recommended")
                widgets.kv("bound overrides",
                           f"{len(d['bounds_overrides'] or {})} row(s)"
                           + (f", {measured} measured for this mission"
                              if measured else ""))
                # a run that STARTS FROM a design is a different run, and it
                # has to say so where it says everything else it will do —
                # otherwise a seed armed on one page silently anchors every
                # search from then on
                if d.get("x_seed"):
                    widgets.kv("starts from",
                               "a design you kept — evaluated first, and the "
                               "answer can only be at least as good")
                    ui.button("forget it and start fresh", icon="backspace",
                              on_click=_forget_seed) \
                        .props("flat dense no-caps")
                ui.code(v1.reproduce_snippet(d), language="python") \
                    .classes("w-full")

    # NO REFUSAL CARD. A family that cannot score a cant now draws NOTHING
    # here — no heading, no sentence, no route button. What stood here was a
    # "Wing cant and sweep" box that named the measured reason (the mast
    # ahead of the CG under water, the lifting line in air) and then offered
    # buttons that CHANGED THE FAMILY to one that could answer: add a tip
    # device, free the tail height, search the cant, design the tail's
    # planform. Asked for by the user after reading exactly that card on a
    # hydrofoil. A control whose only action is to become a different
    # aeroplane is not the question its own heading asks, and a heading that
    # appears only to say "not here" reads as a button that does not work.
    #
    # The routes are deleted rather than moved: every one of them is a
    # choice the user can already make, three of them on this same card (tip
    # device, tail height, tail design) and the fourth on the cant card
    # itself wherever the family declares the rows. The measured reason is
    # not lost either — ``session.spiral_dihedral`` still owns it and the
    # spiral read-out still states it, on the designs that have a spiral
    # answer to state it about.

    def _wing_cant_panel():
        """THE WING'S OWN CANT AND SWEEP, where the family can score them.

        Both are lattice geometry (``geometry.dihedral_rotate`` and the
        quarter-chord offset in ``vlm.VLM``), so only a lattice-backed family
        declares them — a lifting line has no out-of-plane geometry to give a
        dihedral effect to, and ``api.check_flags`` refuses the keys there.
        The card is therefore absent rather than disabled: it is not a
        setting this family happens to leave at zero, it is a question it
        cannot answer.

        STATED OR SEARCHED, the same either/or the two separations offer —
        because a dihedral is the same kind of question: a number somebody
        decides, or a row the optimiser rides. Free is a different DIMENSION
        and therefore a different registered problem (``api.WING_CANTS``),
        which is why the toggle changes the family rather than a flag.

        What a searched cant needs to be worth anything is a LATERAL term in
        the objective. Neither row buys lift, so on L/D alone the dihedral
        does not settle at zero — it goes to a BOUND, measured over 42
        paired searches as the anhedral one (median exactly -10 deg, riding
        it in 23 of 42 seeds), and the design comes back divergent 30 times
        in 42 (the other 12 converge on their tip device, by accident). The
        composite's ``spiral`` criterion is the term that changes that, so
        the free side of this card says so and links to it
        (RESULTS_SESSION67_WING_CANT.md).
        """
        flags = sp().flags
        mode = api.searched_cant(W["problem"])
        searched = mode != "fixed"
        # WHICH ROW IS WHICH. Two questions now, answered one at a time:
        # ``searched_keys`` are the design-box rows this family carries and
        # ``keys`` are the ones it still takes as stated flags, so a family
        # that optimises the dihedral draws the searched band for it AND the
        # typed buttons for the sweep, on the same card.
        searched_keys = [k for k in api.searched_cant_keys(mode)]
        keys = [k for k in api.WING_CANT_KEYS if k in flags]
        ch = W["choices"]
        offered = session.cant_states_offered(W["problem"], ch)
        can_free = offered.get("free", False)
        # ...AND WHETHER ANYTHING CAN ANSWER AT ALL — including a TWIN this
        # configuration can be switched to, which is a different question
        # from whether THIS family takes a cant as a stated number. A
        # lifting line has no out-of-plane geometry, so the published
        # TANDEM declares neither cant key and searches neither row, and
        # this card used to return here — but the pair has a nonplanar twin
        # for every cant state and this card is the control that selects
        # one. Absent, the pair's dihedral was unreachable in BOTH
        # directions: no field to type into, because the lifting line
        # cannot fly one, and no toggle to search with, because the toggle
        # was on the card that had gone. Asked through
        # ``session.cant_is_answerable`` and not decided here, because
        # stage 5's spiral advice branches on the same question and the two
        # must not disagree about whether this card exists. The card is
        # still ABSENT where nothing can answer.
        if not session.cant_is_answerable(W["problem"], ch):
            # nothing at all: see NO REFUSAL CARD above
            return
        rows = {"wing_dihedral_deg": ("dihedral", 0.5,
                                      "tips UP, degrees. Negative is "
                                      "anhedral. This is the only wing-side "
                                      "source of Cl_beta that holds at any "
                                      "lift: without it the fin carries the "
                                      "dihedral effect and the yaw "
                                      "stiffness at once, and no fin size "
                                      "makes the spiral converge."),
                "wing_sweep_deg": ("quarter-chord sweep", 1.0,
                                   "aft positive, degrees. Moves the "
                                   "neutral point aft and costs lift-curve "
                                   "slope; it is NOT a roll lever — its "
                                   "dihedral effect goes as CL and "
                                   "vanishes at cruise.")}
        on = searched or any(W["flags"].get(k) not in (None, "")
                             for k in keys)
        with widgets.group_box("Wing cant and sweep"):
            # THE SWITCH, for the same reason the body has one: "does this
            # wing have a dihedral" is a fact about the aeroplane, and an
            # empty pair of fields is not a way to say no. It is also the
            # question with the largest consequence on this card — with the
            # switch off the only wing-side Cl_beta left is the swept one,
            # which goes as CL and is gone at cruise, so in practice the
            # fin carries the dihedral effect and the yaw stiffness at once,
            # and no fin size makes the spiral converge.
            ui.switch("this wing has dihedral and/or sweep", value=on,
                      on_change=lambda e: _set_wing_cant(bool(e.value))) \
                .props("dense")
            if not on and sp().medium == "water":
                # the water craft's OWN reason. The air sentence below is
                # about a spiral this craft has no vertical surface to have.
                widgets.hint(
                    "Flat and unswept — every published water run. Under "
                    "water the cant is not a roll lever: this craft's only "
                    "vertical surface is the MAST, and on the measured "
                    "windfoil layout its quarter chord stands AHEAD of the "
                    "CG, so its yaw stiffness is negative and a dihedral has "
                    "nothing to race against. (That sign is a station now, "
                    "not a fact: the strut-station control above moves the "
                    "mast along the fuselage, and aft of the CG it "
                    "weathercocks.) "
                    "What the cant does buy here is CAVITATION MARGIN and "
                    "the neutral point — sweep spreads the load (margin "
                    "0.350 → 0.397 at 20 deg for 1.9 % of L/D) and moves "
                    "x_np aft; dihedral trades the tips' submergence "
                    "against DRAUGHT. Switch it on to state either.", "warn")
                boxes["cant_note"] = ui.column().classes("w-full gap-1")
                boxes["cant_note_kw"] = {}
                _render_cant_note()
                return
            if not on:
                widgets.hint(
                    "Planar and unswept — every published run. That is a "
                    "wing with NO dihedral effect of its own: every bit of "
                    "this aeroplane's Cl_beta then comes from the fin and "
                    "the tip device, which is a spiral mode no fin SIZE can "
                    "converge (stage 5 measures ten of them).", "warn")
                # ...AND THE NUMBER, with the switch off. The measurement is
                # a diagnosis, not a setting: the user who has not answered
                # this question is exactly the one who needs to be told that
                # their spiral diverges and by how much dihedral. Hiding it
                # behind the switch would put back the complaint this card
                # was rebuilt for — the shell knowing the lever and not the
                # setting. Its button states the value, which turns the
                # switch on.
                boxes["cant_note"] = ui.column().classes("w-full gap-1")
                boxes["cant_note_kw"] = {}
                _render_cant_note()
                return
            if sp().medium == "water":
                # THE WATER CARD IS A DIFFERENT SENTENCE, because the water
                # cant buys a different thing. Quoting the air family's
                # spiral crossing here would sell the row as a roll lever on
                # a craft whose only vertical surface is DEstabilising.
                widgets.hint(
                    "Geometry the imaged lattice flies and the objective "
                    "prices — measured at this family's own box centre. "
                    "SWEEP spreads the load and so buys CAVITATION MARGIN: "
                    "0.350 → 0.397 from 0 to 20 deg on `hydrofoil + "
                    "winglet`, for 1.9 % of L/D. On `hydrofoil + elevator` "
                    "20 deg also moves the neutral point aft far enough to "
                    "turn that box centre's static margin from −0.308 to "
                    "+0.474 — an infeasible design made feasible — while "
                    "GAINING 3.1 % of L/D. DIHEDRAL moves the tips in "
                    "submergence: up they sit in less static head and "
                    "cavitate sooner (0.350 → 0.213 at 20 deg), down they "
                    "buy margin and cost DRAUGHT (0.575 m flat, 0.795 m at "
                    "−20 deg, which is what the draught cap is about).")
                widgets.hint(
                    "NOT a roll or spiral lever, and this is the one thing "
                    "not to read into it: this craft's only vertical "
                    "surface is the MAST, and on the measured windfoil "
                    "layout its quarter chord stands AHEAD of the CG, so its "
                    "yaw stiffness is negative and there is nothing for a "
                    "dihedral to race against. Move the strut aft of the CG "
                    "(the strut-station control) and it weathercocks. That "
                    "is "
                    "also why these are numbers you STATE here and not rows "
                    "the optimiser rides — the criterion that would price a "
                    "searched cant in air reads nothing at all under "
                    "water.", "warn")
                for key in keys:
                    label, step, why = rows[key]
                    _cant_buttons(key, label, step)
                return
            widgets.hint(
                "Geometry the lattice flies and the objective prices. "
                "Measured on the box centre of `tail [free cant]`, the "
                "family the study was run on: 5.7 deg of dihedral turns a "
                "divergent spiral MODE into a convergent one for 0.43 % of "
                "L/D, and 15 deg of sweep costs 12.7 % while moving the "
                "neutral point 0.57 m aft. The closed-form criterion alone "
                "crosses at 3.56 deg — 2.13 deg optimistic, because it "
                "drops the nose-up trim attitude — so 3.56 buys the number "
                "and not the aeroplane. THAT IS ONE FAMILY'S ANSWER: the "
                "crossing runs from 3.07 deg to 6.13 deg over the families "
                "this shell derives, so the number for the configuration "
                "you are actually on is measured below.")
            boxes["cant_note"] = ui.column().classes("w-full gap-1")
            boxes["cant_note_kw"] = {}
            _render_cant_note()
            # ...AND WHICH OF THE TWO IS OPTIMISED, asked as two questions
            # in one row. They are separately searchable because they are
            # priced against separate things: the dihedral buys Cl_beta at
            # any lift and is what the `spiral` criterion reads, while the
            # sweep buys a neutral point and its own dihedral effect goes as
            # CL and is gone at cruise. "Optimise both" was the only free
            # answer this card had, so a design that wanted a static margin
            # had to hand the optimiser a roll lever to get one.
            with _row("dihedral and sweep"):
                tg = ui.toggle(WING_CANT_LABELS,
                               value=mode,
                               on_change=lambda e: set_choice("wing_cant",
                                                              e.value)) \
                    .props("dense no-caps unelevated toggle-color=primary")
                missing = [WING_CANT_LABELS[k] for k in api.WING_CANTS
                           if k != "fixed" and not offered.get(k)]
                if not can_free and not any(offered.get(k)
                                            for k in ("dihedral", "sweep")):
                    tg.disable()
                    tg.tooltip("no free-cant solver for this configuration")
                elif missing:
                    tg.tooltip("no solver here for: " + ", ".join(missing))
            if searched:
                _cant_box_note()
            # ...and the rows this family still STATES, which on a half
            # variant is the other one. Not an else: a family that searches
            # the sweep takes a typed dihedral, and that is the whole point
            # of splitting the freedom.
            for key in keys:
                label, step, why = rows[key]
                _cant_buttons(key, label, step)
                widgets.hint(why)

    def _set_wing_cant(on: bool):
        """The wing cant's on/off switch.

        ON with nothing stated writes the MEASURED angle — the smallest
        dihedral that turns this configuration's spiral
        (:func:`session.spiral_dihedral`) — because that is the answer the
        card exists to give, and a switch that turns the question on and
        leaves a blank field has answered nothing. Where the measurement
        cannot say (a family that yaws away from the airflow, or one whose
        spiral no cant turns) the fields are opened empty and the card says
        why, which is the honest version of the same switch.

        OFF clears BOTH flags, and takes the family off its free-cant twin
        if that is where the rows live: with the cant SEARCHED the answer is
        in the design vector, so clearing a flag would leave the switch
        saying "off" over two rows the optimiser is still riding.
        """
        if not on:
            for k in api.WING_CANT_KEYS:
                W["flags"].pop(k, None)
            if api.cant_is_searched(W["problem"]):
                set_choice("wing_cant", "fixed")
                return
            _render_type()
            _render_solver()
            ctx.refresh()
            return
        # A FAMILY THAT CANNOT STATE ONE ANSWERS BY SEARCHING IT. The
        # published tandem is a lifting line: it declares neither cant key,
        # so there is no flag for the branch below to write and the switch
        # would have flipped straight back off. Its nonplanar twin carries
        # the rows, and selecting that twin IS the answer to "this wing has
        # dihedral" here — the dihedral first, which is the row the stated
        # branch below writes and the only wing-side Cl_beta that holds at
        # any lift.
        if not any(k in sp().flags for k in api.WING_CANT_KEYS) \
                and not api.cant_is_searched(W["problem"]):
            for want in ("dihedral", "free", "sweep"):
                if v1.option_available(W["choices"], "wing_cant", want):
                    set_choice("wing_cant", want)
                    return
            return
        key = api.WING_CANT_KEYS[0]
        # ...and only where the dihedral is still a STATED value. On a
        # family that searches it there is no flag to write — the answer is
        # in the design box — and ``api.check_flags`` refuses one.
        if key in sp().flags and W["flags"].get(key) in (None, ""):
            info = session.spiral_dihedral(S)
            if info.get("status") == "found" and info.get("gamma_deg"):
                W["flags"][key] = round(float(info["gamma_deg"]), 2)
        _render_type()
        _render_solver()
        ctx.refresh()

    def _render_cant_note():
        """Redraw the measured dihedral read-out ALONE.

        It is a read-out that follows the DESIGN BOX — measured on the box
        centre, so pinning a row, moving a bound or re-sizing a tandem's
        surfaces all move it — and this shell's rule for those is a named
        container the handlers redraw, never a whole view rebuilt from under
        a field being typed. ``boxes["cant_note"]`` is that container; the
        stability read-out two cards up works the same way, for the same
        reason.
        """
        box = boxes.get("cant_note")
        if box is None:
            return
        box.clear()
        with box:
            _spiral_dihedral_note(**(boxes.get("cant_note_kw") or {}))

    def _spiral_dihedral_note(*, offer_button: bool = True):
        """THE ANGLE THIS CONFIGURATION NEEDS, measured, beside the field
        that sets it.

        The card shipped one constant — 5.7 deg, offered as a step at
        every configuration, and it is `tail [free cant]`'s answer. Measured
        with :func:`aerobo.api.dihedral_for_spiral` over the
        families this shell derives (RESULTS_SESSION69_SPIRAL_RECOMMENDATION
        .md): 3.07 deg on `tail + winglet`, 5.57 on `tail [free height]`,
        6.13 on `tail` — and the shell's own DEFAULT wing+tail family is
        that last one. Pressing the 5.7 deg step there left the 6-DOF spiral
        eigenvalue at +0.0026, which is an aeroplane that rolls off with the
        stick free, so the step did not clear the gate it cited.

        The number below is the smallest dihedral whose margin is not
        negative, measured on this family's box centre by the same
        instrument stage 5 prints, and at it the spiral MODE is stable on
        every family checked (-0.0005 to -0.0014) while half a degree less
        is not (+0.003 to +0.010).
        """
        info = session.spiral_dihedral(S)
        st = info.get("status")
        if st in ("not_applicable", "no_fin"):
            # SILENCE, deliberately. A car's rear wing and a hydrofoil have
            # no spiral to turn, and a wing with no vertical surface has
            # nothing for a dihedral to race against — a number here would
            # be about an aeroplane the rebuild invented.
            return
        if st == "slow":
            ui.button("measure the dihedral this design needs", icon="straighten",
                      on_click=lambda: (session.spiral_dihedral(S, force=True),
                                        ctx.render("wing", "type"))) \
                .props("flat dense size=sm no-caps")
            widgets.hint(
                "Not measured automatically here: this family solves a live "
                "section, so the one design report it costs is minutes "
                "rather than milliseconds.")
            return
        if st == "error":
            widgets.hint(f"the dihedral this design needs could not be "
                         f"measured: {info.get('error')}", "warn")
            return
        # the design's OWN fin, guaranteed by the gate above — a report
        # with none is not answered here at all
        assumed = ""
        if st == "refused":
            widgets.hint(
                f"NOT ANSWERED: this design's Cn_beta is "
                f"{info.get('Cn_beta', 0.0):+.5f}, so it yaws AWAY from the "
                f"airflow and the spiral criterion is a difference of two "
                f"products with the wrong sign in it. A dihedral is not the "
                f"question until the fin is." + assumed, "warn")
            return
        if st == "converges":
            widgets.hint(
                f"This configuration's spiral ALREADY converges at the "
                f"{info.get('flown_deg', 0.0):+.3g} deg it flies (margin "
                f"{info.get('margin_now', 0.0):+.6f}), measured on its box "
                f"centre." + assumed)
            return
        if st == "out_of_reach":
            band = info.get("band") or [0.0, 15.0]
            widgets.hint(
                f"MEASURED, and no dihedral up to {band[1]:g} deg converges "
                f"this configuration's spiral (margin "
                f"{info.get('margin_now', 0.0):+.6f} as it stands). The "
                f"lever is somewhere else — a tip device carries most of "
                f"Cl_beta where there is one, and stage 5 measures what the "
                f"fin does." + assumed, "warn")
            return
        gam = float(info.get("gamma_deg") or 0.0)
        cost = info.get("lod_cost_pct")
        price = ("" if cost is None else
                 f" It costs {cost:.2f} % of L/D on this box centre.")
        widgets.hint(
            f"MEASURED ON THIS CONFIGURATION: {gam:.2f} deg of dihedral is "
            f"the least that turns its spiral (margin "
            f"{info.get('margin_now', 0.0):+.6f} flat, "
            f"{info.get('margin_at', 0.0):+.6f} there)." + price + assumed)
        # ...AND WHETHER THE RUN CAN REACH IT. With the cant SEARCHED the
        # answer comes out of the design box, and a box typed into an
        # interval that excludes this angle makes the recommendation
        # unreachable — measured: a band typed as [-10, -2] left the card
        # still offering 5.23 deg, which the run could never return. The
        # box shown is the box searched, and so is the advice beside it.
        row = _bounds_of(api.WING_CANT_KEYS[0])
        if api.dihedral_is_searched(W["problem"]) and row and not (
                row[0] - 1e-9 <= gam <= row[1] + 1e-9):
            widgets.hint(
                f"THE RUN CANNOT REACH IT: the design box searches "
                f"{row[0]:g} – {row[1]:g} deg, which does not contain "
                f"{gam:.2f}. Widen the row, or state the cant instead of "
                f"searching it.", "bad")
        if offer_button and api.WING_CANT_KEYS[0] in sp().flags:
            ui.button(f"state {gam:.2f} deg", icon="check",
                      on_click=_take_spiral_dihedral) \
                .props("outline dense no-caps") \
                .tooltip("writes the dihedral field below — the same flag "
                         "the buttons write")

    def _take_spiral_dihedral():
        """Press: the measured angle becomes the stated one.

        Re-READ rather than closed over, so the value written is the one the
        measurement holds now and not the one a button was drawn with — a
        card can outlive the configuration it was rendered for, and this is
        the same rule the recommended-box buttons follow. Registered as an
        action so the press is drivable without a browser.
        """
        info = session.spiral_dihedral(S)
        if info.get("status") != "found" or not info.get("gamma_deg"):
            return None
        gam = round(float(info["gamma_deg"]), 2)
        _set_flag(api.WING_CANT_KEYS[0], gam)
        _render_type()
        # ...and the SOLVER view, which lists the flags the run will carry:
        # this press adds one to that list
        _render_solver()
        ctx.refresh()
        return gam

    def _cant_box_note():
        """THE SEARCHED ROWS — their bands, and what has to be in the
        objective for a band to mean anything.

        The bands are QUOTED off the effective design box (the same source
        the searched separation quotes), never off ``geometry``\'s published
        constants: a row widened in the box is the row the run searches, and
        a read-out of the published pair would state an interval the search
        was told not to use.

        ONE ROW OR TWO. ``_bounds_of`` answers None for a row this family
        does not carry, so a half variant quotes only its own — and the
        warning below is about the DIHEDRAL's sign specifically, so it is
        gated on that row rather than on "something is searched".
        """
        for key, name in (("wing_dihedral_deg", "dihedral"),
                          ("wing_sweep_deg", "quarter-chord sweep")):
            row = _bounds_of(key)
            if row:
                widgets.hint(f"{name}: the solver searches "
                             f"{row[0]:g} – {row[1]:g} deg.")
        with ui.row().classes("items-center gap-2"):
            ui.button("set the bands", icon="tune",
                      on_click=lambda: ctx.select("wing", "box")) \
                .props("flat dense size=sm no-caps")
        # the two sentences are MEASURED, and the measurement is
        # RESULTS_SESSION67_WING_CANT.md (scripts/cant_arms.py, 42 paired
        # searches at the recommended budget). Re-derive it before editing
        # either number.
        if api.dihedral_is_searched(W["problem"]) \
                and not api.wants_spiral(config.flags(S)):
            floored = session.unpriced_cant_floor(S)
            widgets.hint(
                "NOTHING IN THIS OBJECTIVE PRICES THE SIGN OF THIS ROW. It "
                "is the flattest of the eleven rows this family trades — "
                "2.1 % of L/D end to end, against 41.6 % for the sweep — so "
                "the search resolves it last or not at all: measured over "
                "32 seeds of this configuration, 11 came back ANHEDRAL and "
                "only 12 had a convergent spiral. More budget does not fix "
                "it (7 of 16 negative at this stage's own budget).",
                "warn")
            if floored:
                widgets.hint(
                    f"So the band below is FLOORED at "
                    f"{session.CANT_FLOOR_DEG:g} deg while nothing prices "
                    f"it, and that costs nothing: L/D peaks at +5 to +6 deg "
                    f"on every free-cant family measured and the anhedral "
                    f"bound is 2.0-2.5 % below the best, so the search was "
                    f"spending draws on a half-box nothing wants. Measured, "
                    f"8 seeds: anhedral 4/6 before, 0/8 after, and the "
                    f"median L/D went UP. Type the row to take the band "
                    f"back — this is a default, not a limit.")
            widgets.hint(
                "It stops the ANHEDRAL answers and it does not buy "
                "stability: with the floor and no price on the criterion, "
                "3 of 8 winners still have a convergent spiral. Weighting "
                "'spiral divergence' is what buys it — 27 of 32 convergent, "
                "for about 1.1 % of L/D and 0.12 of Dutch-roll damping "
                "ratio, which is a preference and not a violation (no "
                "winner of either arm is under MIL-F-8785C Level 1's 0.19). "
                "Weight it, or state the cant instead of searching it.",
                "warn")
            ui.button("weight it in the objective", icon="tune",
                      on_click=lambda: ctx.select("wing", "solver")) \
                .props("flat dense size=sm no-caps")

    #: WHAT EACH BUTTON MEANS, in degrees. Buttons and not a typed box
    #: because a cant is CHOSEN from a handful of engineering answers rather
    #: than dialled, and the numbers below are the ones the measurements
    #: landed on. The typed field beside them takes any value, so a user who
    #: wants 3.7 is not refused — the same contract every recommended value
    #: in this shell carries.
    #:
    #: EACH BUTTON STATES ITS OWN NUMBER. It used to read "a little",
    #: "enough", "a lot" / "mild", "moderate", "hard", and a word is not an
    #: angle: "enough" was one family's 5.7 deg offered at every
    #: configuration, and it left the shell's own default wing+tail with a
    #: divergent spiral (+0.0026) while still saying it was enough. A number
    #: cannot make that claim. Only "none" survives as a word, and it is not
    #: a value — it is the branch that stores NO FLAG AT ALL
    #: (:func:`_set_cant_step`), which is what keeps every published run
    #: bit-for-bit.
    #:
    #: The label is DERIVED from the value in :func:`_cant_label` rather than
    #: typed beside it, so a button cannot come to disagree with the number
    #: it sets. The first element below is kept as the same string only so
    #: the table still reads as a table.
    CANT_STEPS = {
        "wing_dihedral_deg": (
            ("none", 0.0, "planar — every published run"),
            ("2 deg", 2.0, "not enough to converge the spiral alone"),
            ("5.7 deg", 5.7, "where the spiral MODE crosses on the box "
                             "centre of `tail [free cant]`, the family the "
                             "study was run on. NOT every family: measured "
                             "5.43-6.13 across the wing+tail families with "
                             "no tip device and 3.07 with one, so on the "
                             "bare pair this step leaves the mode divergent "
                             "(+0.0026) and the number above is the one for "
                             "the configuration on screen. The criterion "
                             "alone says 3.56, which is 2.13 deg optimistic"),
            ("6 deg", 6.0, "convergent on every family measured except the "
                           "bare wing+tail, which turns at 6.13; "
                           "costs 0.49 % of L/D"),
        ),
        "wing_sweep_deg": (
            ("none", 0.0, "unswept — every published run"),
            ("10 deg", 10.0, "x_np about 0.4 m aft"),
            ("20 deg", 20.0, "x_np about 0.8 m aft, and it is costing "
                             "real L/D by here"),
            ("30 deg", 30.0, "x_np 1.3 m aft; a 12 % L/D bill"),
        ),
    }

    def _cant_label(value: float) -> str:
        """The BUTTON's text, from the number it sets.

        Derived and not stored: the one thing this ladder must never do
        again is offer a word that outlives the measurement behind it. Zero
        is the exception and it is not a value — it is "no flag at all", so
        it keeps the only word on the row.
        """
        return "none" if float(value) == 0.0 else f"{float(value):g} deg"

    def _cant_buttons(key: str, label: str, step: float):
        """The choice as BUTTONS, with the typed box as the escape hatch.

        The stored value is a plain number either way — nothing downstream
        knows a button was pressed — so a session restored from a typed 3.7
        selects no button and says so under the field rather than snapping
        to a step.

        The option KEYS are ``str(value)`` and the option LABELS come from
        :func:`_cant_label`. Those are deliberately different shapes —
        ``str(6.0)`` is "6.0" and the label is "6 deg" — because the key is
        what :func:`_set_cant_step` parses back and what a test drives.
        """
        held = W["flags"].get(key)
        steps = CANT_STEPS[key]
        opts = {str(v): _cant_label(v) for _n, v, _why in steps}
        matched = next((v for _n, v, _w in steps
                        if held is not None and abs(float(held) - v) < 1e-9),
                       None)
        if held is None:
            matched = 0.0          # unstated IS planar/unswept
        with _row(label):
            ui.toggle(opts, value=(None if matched is None
                                   else str(float(matched))),
                      on_change=lambda e: _set_cant_step(key, e.value)) \
                .props("dense no-caps unelevated toggle-color=primary")
        # the number itself, always visible: a button that hides what it set
        # is a button whose effect the user cannot check
        widgets.number_field(
            "…or state it", held, lambda e, k=key: _set_flag(k, e.value),
            unit="deg", step=step,
            tip="the buttons above are the measured answers; this is the "
                "same flag and takes any value. Empty is planar/unswept, "
                "which is what every published run flew")
        if matched is None and held is not None:
            widgets.hint(f"{float(held):+.4g} deg — your own number, not one "
                         f"of the steps above.")
        else:
            why = next((w for _n, v, w in steps
                        if matched is not None and abs(v - matched) < 1e-9),
                       "")
            if why:
                widgets.hint(why)
        # ...AND A STATED ANHEDRAL IS NAMED, whichever of those two branches
        # drew. "-60 deg — your own number, not one of the steps above" is
        # true and says nothing: the searched row is floored at 0 while
        # nothing prices its sign, and the STATED one had no equivalent at
        # all. Not refused — a stated number is the user's answer — but it
        # does not pass in silence.
        if key == api.WING_CANT_KEYS[0]:
            note = api.anhedral_note(held)
            if note:
                widgets.hint(note, "bad")

    def _set_cant_step(key: str, value):
        """A button. 0 stores NOTHING, so a planar wing sends no flag at all
        and every published run reproduces bit-for-bit."""
        if value in (None, ""):
            return
        v = float(value)
        _set_flag(key, None if v == 0.0 else v)

    def _physics_panel():
        flags = sp().flags
        # mach is NOT in this list: it is derived from the mission and
        # offered as an opt-in switch above, never typed
        scalars = [("ground_h_m", "ride height", 0.05, None)]
        offered = [row for row in scalars if row[0] in flags]
        if not offered and "slipstream" not in flags:
            return
        with widgets.group_box("Physics"):
            if "mach" in flags:
                _mach_control()
            for key, label, step, default in offered:
                widgets.number_field(
                    label, W["flags"].get(key, default),
                    lambda e, k=key: _set_flag(k, e.value), step=step,
                    tip="left empty, the solver's own default is used")
            if "slipstream" in flags:
                p = W["prop"]
                ui.switch("propeller slipstream",
                          value=bool(p["enabled"]),
                          on_change=lambda e: _set_prop("enabled",
                                                        bool(e.value))) \
                    .props("dense")
                if p["enabled"]:
                    widgets.number_field(
                        "disk diameter", p["D_p"],
                        lambda e: _set_prop_number("D_p", e.value),
                        unit="m", step=0.1)
                    ui.toggle({"single": "centreline", "pair": "pair"},
                              value=p["layout"],
                              on_change=lambda e: _set_prop("layout",
                                                            e.value)) \
                        .props("dense no-caps unelevated "
                               "toggle-color=primary")
                    if p["layout"] == "pair":
                        widgets.number_field(
                            "pair centre y", p["y_p"],
                            lambda e: _set_prop_number("y_p", e.value),
                            unit="m", step=0.1)
                    widgets.number_field(
                        "thrust coefficient", p["CT"],
                        lambda e: _set_prop_number("CT", e.value),
                        step=0.05)

    def _mach_control():
        """The mission already implies a Mach number; applying it is still a
        DECISION, because the section tables are incompressible."""
        dp = session.design_point(S)
        mach = None if "error" in dp else dp.get("mach")
        if mach is None:
            return
        ui.switch(f"apply the mission's Mach number ({mach:.4f})",
                  value=bool(W.get("apply_mach")),
                  on_change=lambda e: _set_apply_mach(bool(e.value))) \
            .props("dense")
        if W.get("apply_mach"):
            widgets.hint(
                "Sent as the mach flag: the wing solver applies its "
                "Prandtl-Glauert correction. The SECTION data stay "
                "incompressible either way.",
                "warn" if mach > session.MACH_WARN else "")
        else:
            widgets.hint("Off: the run is incompressible, as the published "
                         "studies are. The number above is derived from the "
                         "mission speed and altitude.")

    def _set_apply_mach(on: bool):
        W["apply_mach"] = bool(on)
        _render_solver()
        ctx.refresh()

    def _set_flag(key: str, value):
        if value in (None, ""):
            W["flags"].pop(key, None)
        else:
            W["flags"][key] = float(value)
        # the physics numbers are typed INTO the solver view, so it is the
        # one derived view this handler may not rebuild
        ctx.refresh(("wing", "solver"))

    def _set_prop(key: str, value):
        W["prop"][key] = value
        # only the two keys that change WHICH controls exist redraw the
        # view; a number typed into one of those controls must not (rule:
        # a rebuilt input loses the focus and swallows the rest of it)
        if key in PROP_SHAPE_KEYS:
            _render_solver()
        ctx.refresh(("wing", "solver"))

    def _set_prop_number(key: str, value):
        """A typed propeller number — and empty is not zero.

        ``<input type=number>`` reports a half-typed "2." as an empty
        string, which arrives here as ``None``: ``float(value or 0.0)`` then
        wrote a 0 m disk (a physically meaningless propeller) between two
        keystrokes. Empty leaves the stored value alone, exactly as every
        other numeric handler in this file does.
        """
        if value in (None, ""):
            return
        _set_prop(key, float(value))

    def _set_opt(name: str):
        W["optimiser"] = name
        _render_solver()
        ctx.refresh()

    def _set_int(key: str, value, lo: int):
        if value in (None, ""):
            return
        W[key] = max(lo, int(value))
        # the budget/seed fields live in the solver view: it may not be
        # rebuilt from inside one of their own on_change handlers
        ctx.refresh(("wing", "solver"))

    def _estimate() -> str:
        eff = session.effective_wing_search(S)
        n = int(eff["budget"]) * max(1, int(eff["n_seeds"]))
        if sp().slow:
            return (f"{n} evaluations of a live XFOIL sweep — expect minutes "
                    f"to hours, not seconds.")
        return (f"{n} evaluations of a reduced-order solve — seconds "
                f"to minutes.")

    # ------------------------------------------------------- run control
    def launch():
        if ctx.manager.running:
            ui.notify("a run is already in progress", type="warning")
            return
        if ((S.get("run") or {}).get("relax") or {}).get("busy"):
            ui.notify("the reach is still searching — stop it first",
                      type="warning")
            return
        for label, (lo, hi) in W["bounds"].items():
            if not float(lo) < float(hi):
                ui.notify(f"bound “{label}”: low must be below high",
                          type="negative")
                return
        eff = session.effective_wing_search(S)
        n_seeds = max(1, int(eff["n_seeds"]))
        base = int(eff["seed"])
        stop = session.stop_rule_factory(S, eff)
        jobs = []
        try:
            for k in range(n_seeds):
                cfg = config.build_cfg(S, seed=base + k)
                # the objective's own refusals land HERE, beside the button
                # that caused them, instead of in a queued job's error field
                api.check_wing_objective(cfg.flags)
                jobs.append(v1.RunJob(
                    cfg=cfg,
                    label=f"{sp().display} · seed {base + k}",
                    budget=int(eff["budget"]),
                    stop_rule=stop,
                    # …and MEMOISED, so the continuation this run may be given
                    # later re-flies its prefix off disk instead of buying it
                    # twice (session.EVAL_CACHE)
                    eval_cache=session.EVAL_CACHE))
        except (ValueError, KeyError) as exc:
            ui.notify(f"{type(exc).__name__}: {exc}", type="negative")
            ctx.log(f"launch refused: {exc}", "error")
            return
        try:
            ctx.manager.start(jobs)
        except RuntimeError as exc:
            ui.notify(str(exc), type="negative")
            return
        S["run"]["error"] = None
        _forget_score()
        live.reset()
        if live_cfg["on"]:
            base_cfg = jobs[0].cfg
            live.start(cfg_fn=lambda c=base_cfg: c,
                       incumbent_fn=_incumbent,
                       running_fn=lambda: ctx.manager.running)
        ctx.log(f"launched {n_seeds} run(s) of {sp().display} — "
                f"{int(eff['budget'])} evaluations, seed {base}"
                + (" (budget and optimiser from the measured recommendation)"
                   if eff["source"] == "recommended" else ""), "ok")
        ctx.status("running…", "busy", 0.0)
        ctx.select("wing", "run")
        ctx.refresh()

    def continue_run(record=None, extra=None):
        """Give a FINISHED run more evaluations, and say what that means.

        Not a warm start and not a handoff: the record's own configuration is
        re-launched at a bigger budget on the SAME seed, so for every
        optimiser in ``api.CONTINUABLE_OPTIMISERS`` the evaluations already
        paid for come back bit-for-bit and the new ones carry on from them
        (measured, ``api.continue_run_config``). That matters because this
        repo has measured the alternative — seeding BO with a previous run's
        best — and BO is the optimiser it is worst to hand off TO.

        The config comes off the RECORD, never off the session: the box may
        have moved since, and a longer run of a DIFFERENT search is not what
        the button says. Where they differ, the card says so first.
        """
        if ctx.manager.running:
            ui.notify("a run is already in progress", type="warning")
            return
        if ((S.get("run") or {}).get("relax") or {}).get("busy"):
            ui.notify("the reach is still searching — stop it first",
                      type="warning")
            return
        rd = record if record is not None else (S.get("run") or {}).get(
            "record")
        out = session.continue_run(S, rd, extra)
        if out["error"]:
            ui.notify(out["error"], type="negative")
            ctx.log(f"cannot continue: {out['error']}", "warn")
            return
        cfg, note = out["cfg"], out["note"]
        try:
            api.check_wing_objective(cfg.flags)
            # THE CONTINUATION'S OWN BUDGET sizes its stop rule. Bare,
            # ``stop_rule_factory`` reads the CURRENT plan's budget: a fresh
            # air session gives max_evals=29 with patience=40, so a
            # continuation to 43 evaluations had plateau detection
            # structurally impossible — patience 40 is unreachable inside 29,
            # and the wrapper's ``fired and rule.n < rule.max_evals`` is
            # False for every evaluation past 29. The longer the run, the
            # more the adaptive stop is worth, and this was the one place it
            # was switched off.
            eff = dict(session.effective_wing_search(S))
            eff["budget"] = int(cfg.budget)
            job = v1.RunJob(cfg=cfg,
                            label=f"{sp().display} · seed {cfg.seed} · "
                                  f"{note['was']}+{note['added']}",
                            budget=int(cfg.budget),
                            stop_rule=session.stop_rule_factory(S, eff),
                            # THE REASON THE MEMO EXISTS. This run re-flies
                            # every evaluation the one it continues already
                            # paid for; with the cache on, those come back off
                            # disk and what it spends is the new ones.
                            eval_cache=session.EVAL_CACHE,
                            # ...and the run view SAYS that, instead of
                            # showing a counter that starts at 1 again with
                            # no explanation. Only where the api has judged
                            # the longer run to contain the shorter one: an
                            # uncontained continuation flies a different set
                            # of designs, and calling its opening evaluations
                            # a replay would be false.
                            # THE EVALUATIONS IT INHERITS. On a resume the
                            # previous run's points are this run's training
                            # set and nothing is re-flown, so `replay` — the
                            # count of repeated evaluations — is zero and the
                            # counter simply starts where the last run
                            # stopped.
                            resume=note.get("resume"),
                            replay=(0 if note.get("resume") else
                                    (int(note["was"])
                                     if note.get("exact") else 0)),
                            # ...and the CURVE it continues, so the view does
                            # not go blank while the prefix re-flies
                            prior_history=list(rd.get("history") or []))
            ctx.manager.start([job])
        except (ValueError, KeyError, RuntimeError) as exc:
            ui.notify(f"{type(exc).__name__}: {exc}", type="negative")
            ctx.log(f"continuation refused: {exc}", "error")
            return
        S["run"]["error"] = None
        _forget_score()
        live.reset()
        # …and START the sampler, as launch does. Without this the live panel
        # showed the switch ON, "0 samples" and "waiting for the first
        # improvement…" for the whole continuation — the run that is by
        # definition the long one, where watching the incumbent's metrics is
        # worth most.
        if live_cfg["on"]:
            live.start(cfg_fn=lambda c=cfg: c,
                       incumbent_fn=_incumbent,
                       running_fn=lambda: ctx.manager.running)
        ctx.log(f"continuing: {note['was']} -> {note['budget']} evaluations. "
                + note["why"]
                # WHY THE COUNTER STARTS AT 1 AGAIN. It re-flies the prefix by
                # design — that is what makes the longer run contain the
                # shorter one — and with the memo on it re-flies it off disk,
                # so the seconds are the new evaluations only. Said here
                # because the progress bar going back to the start is the one
                # thing about this button a user reads as a fault.
                # ...and WHICH of the two continuations this is. A resume
                # counts on from where the last run stopped and flies only
                # the new designs; a re-flight starts at 1 again, which is
                # the thing users read as a fault, so it is the one that has
                # to be said out loud.
                + (f" It RESUMES: the {note['was']} evaluations already paid "
                   f"for are this run's training set, the counter opens at "
                   f"{note['was']}, and the only designs flown are the "
                   f"{note['added']} new ones."
                   if note.get("resume") else
                   f" The counter restarts because the search does; the "
                   f"{note['was']} evaluations already paid for come back out "
                   f"of the on-disk memo, so what this spends is the "
                   f"{note['added']} new ones."
                   if session.EVAL_CACHE and note["exact"] else ""),
                "ok" if note["exact"] else "warn")
        if out["drift"]:
            ctx.log("this runs the RECORD's configuration, not the one on "
                    "screen — " + ", ".join(out["drift"]) + " changed since",
                    "warn")
        ctx.status("running…", "busy", 0.0)
        ctx.select("wing", "run")
        ctx.refresh()

    def _incumbent():
        """(evaluations so far, best design vector) of the running job."""
        job = ctx.manager.current
        if job is None:
            return 0, None
        best_x, best_f = None, None
        for rec in job.records:
            f, x = rec.get("f"), rec.get("x")
            feasible = rec.get("feasible")
            if x is None or f is None or feasible is False:
                continue
            if best_f is None or float(f) > best_f:
                best_f, best_x = float(f), list(x)
        return len(job.records), best_x

    def cancel():
        if not ctx.manager.running:
            ui.notify("nothing is running", type="info")
            return
        ctx.manager.cancel()
        ctx.log("cancel requested — the current evaluation finishes first, "
                "and the best design found so far is kept and scored", "warn")
        ctx.status("cancelling…", "warn", None)

    # ------------------------------------------------------- view: run
    #: WHAT THE RUN VIEW'S STRUCTURE DEPENDS ON — and nothing else.
    #:
    #: While a search runs, only numbers move: the same read-outs, the same
    #: figures, the same metric menu. So the view is BUILT when this tuple
    #: moves and UPDATED IN PLACE on every other tick (:func:`_tick_run`).
    #:
    #: Rebuilding it per evaluation is what threw the page back to the top.
    #: The work area is one scroll pane; clearing the view collapses its
    #: content height, the browser clamps the scroll position to 0, and a
    #: user reading the live trace of a running search was returned to the
    #: top of the page twice a second — which is also why an expansion
    #: inside it could never be left open.
    def _run_sig():
        job = ctx.manager.current
        return (id(job) if job is not None else None,
                getattr(job, "status", None),
                len(ctx.manager.jobs),
                bool(getattr(job, "records", ())),
                bool(getattr(job, "iters", ())),
                bool(getattr(job, "error", None)),
                bool(getattr(job, "wall", None)),
                bool(sp().is_constrained),
                bool(live_cfg["on"]),
                tuple(live.keys()),
                bool(live.error),
                # the "keep going" total is written into the button in place
                # (``_set_extra``), but a view rebuilt for any OTHER reason
                # must not come back advertising the default again
                (S.get("run") or {}).get("continue_extra"))

    # ---------------------------------------------- a continuation, on screen
    #
    # WHY THE COUNTER MAY GO BACK TO 1 — and why, on a BO arm, it no longer
    # does. "Keep going" on a resumable optimiser (``api.RESUMABLE_OPTIMISERS``
    # — ``bo`` and ``bo_slsqp``, which is what the recommended policy flies on
    # a constrained wing) hands the record's own evaluations to the new run as
    # its TRAINING SET: nothing is re-flown, the counter opens at what was
    # inherited, and a continuation of 53 by 4 flies 4. Everything else — and
    # any record too old to carry ``eval_x``/``eval_y`` — falls back to
    # re-launching the record's configuration on the same seed at a bigger
    # budget, where the prefix IS re-flown and that re-flight is precisely
    # what makes the longer run contain the shorter one. Warm-starting from
    # the previous best instead is the arm this repo measured and lost with
    # (``RESULTS_HANDOFF.md`` — BO is the optimiser it is worst to hand off
    # TO); a resume is not that, because a GP opened on 53 observations is in
    # the same state as one that flew them.
    #
    # What was wrong was the SILENCE. With the memo on (``session.EVAL_CACHE``)
    # the re-flight is a disk read — measured on ``trim wing``, bo, 10 -> 14:
    # 10 hits, 4 misses, 0.6 s against the 1.4 s the first run cost — so the
    # bar races through the prefix and stops where the last run stopped. Read
    # without a label, that is indistinguishable from starting over. These
    # three helpers put the label on it.
    def _replay_of(job) -> int:
        """How many leading evaluations of this job are a re-flight."""
        return max(0, int(getattr(job, "replay", 0) or 0))

    def _resumed_of(job) -> int:
        """How many evaluations this job INHERITED and never flew.

        A resumed continuation (``api.run(resume=…)``) opens with the
        previous run's points as its training set, so its own records start
        at evaluation ``resumed + 1``. Every count on this card adds it back,
        because the run is one search of ``budget`` evaluations and the user
        asked for the last few of them.
        """
        return max(0, int((getattr(job, "resume", None) or {}).get("n") or 0))

    def _flown_text(job) -> str:
        """``evaluations`` as the card shows it: inherited + this run's."""
        return f"{_resumed_of(job) + len(job.records)}/{job.budget}"

    def _run_state_text(job) -> str:
        n, replay = len(job.records), _replay_of(job)
        if job.status == "running" and replay and n < replay:
            return "replaying"
        return job.status

    def _run_conv_fig(job):
        """The convergence curve, with the replayed prefix marked.

        The prefix is the SAME evaluations the previous run flew, so the
        curve over it is the previous run's curve: shading it says where the
        old run ended and the new search began, which is the question "did
        continuing buy anything?" asks.

        THE PLOT DOES NOT GO BLANK. A continuation owns no records until its
        first evaluation lands, so this used to draw an empty panel and then
        redraw from evaluation 1 — the same thing the counter does, and the
        thing users read as "it started over". The run being lengthened
        carries its own curve (``RunJob.prior_history``), so that curve is
        drawn first and stays drawn, as its own trace: over the prefix the
        two are the same numbers by construction, and past it they are
        honestly two different runs.
        """
        prior = list(getattr(job, "prior_history", None) or [])
        fig = v1.fig_convergence(job.records,
                                 history=(prior if prior else None))
        replay = _replay_of(job)
        if fig is None:
            return fig
        if prior and job.records:
            # ...as a SECOND trace once this run has points of its own. Muted
            # and dashed: it is context, and the live curve is the answer.
            import plotly.graph_objects as _go

            # NaN where the prior run had no feasible design yet. A history
            # stores None for -inf (v1._nan_history exists for exactly this,
            # and fig_convergence above already reads the same array through
            # it); floating it raw raised TypeError the moment the
            # continuation drew its second trace — i.e. on every "Keep going"
            # from a run whose first evaluation was refused. Common on the car
            # families, where the box has real refusals in it.
            fig.add_trace(_go.Scatter(
                x=list(range(1, len(prior) + 1)),
                y=v1._nan_history(prior),
                mode="lines", name="the run this continues",
                line=dict(color=theme.INK_MUTED, width=1.5, dash="dot"),
                hovertemplate="eval %{x}<br>was %{y:.5g}<extra></extra>"))
        if replay < 1:
            return fig
        fig.add_vrect(x0=0.5, x1=replay + 0.5, line_width=0,
                      fillcolor=theme.INK_MUTED, opacity=0.12,
                      annotation_text=f"re-flown ({replay})",
                      annotation_position="top left",
                      annotation_font_size=10)
        return fig

    def _tick_run():
        """Update a run view already on screen: numbers and traces only.

        Falls back to the full build the moment the structure moves — a job
        finished, the first record arrived, BO started iterating, the
        sampler learned a metric it had not reported before. Everything
        else is written into the elements that are already there, so the
        page never changes height and the scroll position survives.
        """
        if boxes.get("run_sig") != _run_sig():
            _render_run()
            return
        job = ctx.manager.current
        if job is None:
            return
        if boxes.get("run_state") is not None:
            boxes["run_state"].set_text(_run_state_text(job))
        if boxes.get("run_evals") is not None:
            boxes["run_evals"].set_text(_flown_text(job))
        if boxes.get("run_replay") is not None:
            replay = _replay_of(job)
            boxes["run_replay"].set_text(
                f"{min(len(job.records), replay)}/{replay}")
        if boxes.get("run_best") is not None:
            best = next((r["best"] for r in reversed(job.records)
                         if r.get("best") is not None), None)
            boxes["run_best"].set_text(
                f"{best:.5g}" if best is not None else "—")
        if boxes.get("run_wall") is not None and job.wall:
            boxes["run_wall"].set_text(f"{job.wall:.1f}")
        if boxes.get("run_samples") is not None:
            boxes["run_samples"].set_text(f"{len(live.samples)} samples")
        figstyle.update(boxes.get("run_conv"), _run_conv_fig(job),
                        "convergence")
        if boxes.get("run_cons") is not None:
            fig = v1.fig_constraints(
                job.records,
                list(_effective_clabels()))
            if fig is not None:
                figstyle.update(boxes["run_cons"], fig, "constraints")
        figstyle.update(boxes.get("run_live"), _live_fig(), "live_metrics")
        if boxes.get("run_surr") is not None:
            figstyle.update(boxes["run_surr"],
                            v1.fig_surrogate(job.records, job.iters),
                            "surrogate")

    def _render_run():
        box = ctx.views[("wing", "run")]
        box.clear()
        # every handle below belongs to THIS build of the view: a stale one
        # would be written into a pane that is no longer on screen
        for key in [k for k in boxes if k.startswith("run_")]:
            boxes.pop(key, None)
        jobs = ctx.manager.jobs
        with box:
            if not jobs:
                widgets.hint("No run yet. Configure the solver on the "
                             "previous tab and launch.")
                boxes["run_sig"] = _run_sig()
                return
            job = ctx.manager.current
            with ui.row().classes("w-full items-start gap-4 no-wrap"):
                boxes["run_state"] = widgets.readout(
                    "state", _run_state_text(job))
                boxes["run_evals"] = widgets.readout(
                    "evaluations", _flown_text(job))
                if _resumed_of(job):
                    widgets.readout(
                        "inherited", f"{_resumed_of(job)}",
                        tip="evaluations this continuation was HANDED by the "
                            "run it lengthens — they are its training set, "
                            "not physics it bought, and none of them is "
                            "flown again")
                if _replay_of(job):
                    replay = _replay_of(job)
                    boxes["run_replay"] = widgets.readout(
                        "re-flown",
                        f"{min(len(job.records), replay)}/{replay}",
                        tip="evaluations this continuation is repeating from "
                            "the run it lengthens — the same designs, on the "
                            "same seed, read back out of the on-disk memo")
                best = next((r["best"] for r in reversed(job.records)
                             if r.get("best") is not None), None)
                boxes["run_best"] = widgets.readout(
                    "best objective",
                    f"{best:.5g}" if best is not None else "—")
                if job.wall:
                    boxes["run_wall"] = widgets.readout(
                        "wall", f"{job.wall:.1f}", "s")
            # ...and whether the numbers above are even about the box on the
            # Design box tab. DERIVED by comparing the record's own overrides
            # against this session's, so it is true after a reach, after an
            # edit that was never launched, and after a session is loaded — a
            # boolean somebody had to remember to set would go stale the first
            # time somebody forgot.
            drift = relax_mod.banner(S, (S.get("run") or {}).get("record"))
            if drift:
                widgets.hint(drift, "warn")
            if len(jobs) > 1:
                widgets.hint("queue: " + ", ".join(
                    f"{j.label.split('·')[-1].strip()} {j.status}"
                    for j in jobs))
            if job.error:
                widgets.hint(job.error, "bad")
            with widgets.group_box("What the search bought"):
                _render_score_block()
            if _resumed_of(job):
                got = _resumed_of(job)
                widgets.hint(
                    f"This is a CONTINUATION, and it starts at {got}: the "
                    f"{got} evaluations the run it lengthens paid for are "
                    f"this run's training set, so nothing is flown twice. "
                    f"What it buys is the "
                    f"{max(0, int(job.budget) - got)} new designs after them.")
            if _replay_of(job):
                replay = _replay_of(job)
                widgets.hint(
                    f"This is a CONTINUATION, so the counter starts at 1 "
                    f"again: the first {replay} evaluations are the ones the "
                    f"run it lengthens already flew, re-flown on the same "
                    f"seed — which is what makes this longer run contain the "
                    f"shorter one. They come back out of the on-disk memo "
                    f"rather than being bought twice, so what this spends is "
                    f"the {max(0, int(job.budget) - replay)} new ones."
                    if session.EVAL_CACHE else
                    f"This is a CONTINUATION: the first {replay} evaluations "
                    f"are the ones the run it lengthens already flew, "
                    f"re-flown on the same seed. The evaluation memo is off, "
                    f"so they are being paid for again.")
            with widgets.group_box("Convergence", pad=False):
                boxes["run_conv"] = figstyle.show(_run_conv_fig(job),
                                                  "convergence")
            _render_keep_going()
            if job.records and sp().is_constrained:
                labels = list(_effective_clabels())
                fig = v1.fig_constraints(job.records, labels)
                if fig is not None:
                    with widgets.group_box("Constraint margins", pad=False):
                        boxes["run_cons"] = figstyle.show(fig, "constraints")
            _live_panel()
            if job.iters:
                with widgets.group_box("Surrogate", pad=False):
                    boxes["run_surr"] = figstyle.show(
                        v1.fig_surrogate(job.records, job.iters), "surrogate")
        boxes["run_sig"] = _run_sig()

    def _keep_going_extra(rd: dict) -> int:
        """How many evaluations the "keep going" field is showing.

        Held in session state, not in a widget: the field must survive the
        0.5 s heartbeat repainting the run view under the cursor.
        """
        R = S.setdefault("run", {})
        want = R.get("continue_extra")
        if want is None:
            want = session.continue_extra_default(rd)
            R["continue_extra"] = int(want)
        return max(1, int(want))

    def _render_keep_going():
        """Did this run flatten out, and can it simply be given more?

        The two halves of one question. A budget that ran out tells the user
        nothing about whether it was ENOUGH, and until now the shell answered
        it nowhere — so "it did not converge" was a guess. The verdict here is
        the run's OWN stopping rule replayed over its own history
        (``session.run_convergence``), and the control beside it re-launches
        the same search longer rather than starting a different one.
        """
        rd = (S.get("run") or {}).get("record")
        if not rd or ctx.manager.running:
            return
        conv = session.run_convergence(S, rd)
        out = session.continue_run(S, rd, _keep_going_extra(rd))
        with widgets.group_box("Did it converge?"):
            kinds = {"converged": ("flattened out", ""),
                     "climbing": ("still climbing", "warn"),
                     "too_short": ("cannot be said", "warn"),
                     "unmeasured": ("no measured rule", ""),
                     "empty": ("nothing finite was scored", "warn")}
            word, kind = kinds.get(conv["verdict"], (conv["verdict"], ""))
            widgets.readout("verdict", word,
                            tip="replayed from this run's own best-so-far "
                                "log through the same rule the search policy "
                                "would have stopped it with")
            widgets.hint(conv["text"], kind)
            if out["error"]:
                widgets.hint(out["error"], "warn")
                return
            note = out["note"]
            held: dict = {}

            def _set_extra(value, r=rd) -> None:
                """The typed number moves the BUTTON, not just the state.

                Two halves of one control. The state half was already here,
                but the button ran ``continue_run(r)`` with no extra, so
                session.py substituted ``continue_extra_default`` and the
                field bought nothing. The label half is re-derived HERE, in
                place, rather than by repainting the view: this input is one
                ``_render_run`` built, and clearing the pane from inside its
                own handler replaces it under the cursor mid-number.
                """
                if value in (None, ""):
                    return
                S["run"]["continue_extra"] = max(1, int(value))
                nxt = session.continue_run(S, r, _keep_going_extra(r))
                if nxt["note"] and held.get("btn") is not None:
                    held["btn"].set_text(
                        f"Keep going — {nxt['note']['budget']} in total")
                if nxt["note"] and held.get("why") is not None:
                    held["why"].set_text(nxt["note"]["why"])

            with ui.row().classes("w-full items-center gap-2 no-wrap"):
                widgets.number_field(
                    "more evaluations", _keep_going_extra(rd),
                    lambda e: _set_extra(e.value),
                    step=1, width="w-24",
                    tip="added to the budget this run already spent")
                held["btn"] = ui.button(
                    f"Keep going — {note['budget']} in total",
                    icon="play_arrow",
                    on_click=lambda _=None, r=rd: continue_run(
                        r, _keep_going_extra(r))) \
                    .props("unelevated dense no-caps color=primary")
            # split=False, for the reason the wet-size line carries: the
            # "keep going" note is rewritten in place as the extra evaluations
            # are typed (held["why"].set_text, above)
            held["why"] = widgets.hint(note["why"],
                                       "" if note["exact"] else "warn",
                                       split=False)
            if out["drift"]:
                widgets.hint(
                    "this runs the configuration the RECORD carries, not the "
                    "one on screen — " + ", ".join(out["drift"])
                    + " changed since. Launch instead to search what the "
                      "form now says.", "warn")

    # -------------------------------------- baseline vs optimised, scored
    def _score_worker(rd: dict):
        """Score the box centre and the returned design on every criterion.

        Two physics evaluations, after the run rather than during it, so a
        search never pays for its own report. Runs on ANY objective: on an
        L/D run these numbers are what maximising one scalar cost the other
        five, which is the whole reason the block exists.
        """
        sc = _score()
        # ``report_for`` is the record this report is ABOUT, written before
        # the first physics call so the spinner is attributed too. The view
        # ignores a report stamped with anybody else's record (D2).
        sc.update(scoring=True, report=None, report_error=None,
                  report_for=rd)
        try:
            cfg = config.build_cfg(S)
            best_x = rd.get("best_x")
            if not best_x:
                raise ValueError("the run returned no feasible design")
            sc["report"] = api.score_optimised_design(
                cfg, best_x, weights=dict(sc["weights"]),
                reference=sc.get("band"))
        except Exception as exc:      # noqa: BLE001 — a view, never fatal
            sc["report_error"] = f"{type(exc).__name__}: {exc}"
        finally:
            sc["scoring"] = False
            sc["report_stamp"] = time.time()

    def _forget_seed() -> None:
        session.clear_start_design(S)
        ctx.log("the next run starts from its own initial design again",
                "info")
        _render_solver()
        ctx.refresh()

    def _start_from(x, note: str) -> None:
        """Arm the next run to begin at design ``x``, then run it."""
        err = session.start_from_design(S, x)
        if err:
            ui.notify(err, type="warning")
            ctx.log(f"cannot start from that design: {err}", "warn")
            return
        ctx.log(note, "info")
        _render_solver()
        ctx.refresh()
        launch()

    def _render_start_from_nearest(rd: dict, labels) -> bool:
        """"It found nothing" is not the end of the log — take the design that
        came CLOSEST and run again from there.

        A constrained run that returns no design still evaluated designs, and
        the least-violating one is a real wing that missed a limit. Given back
        as the next run's FIRST evaluation it is not another roll of the same
        dice: the surrogate starts on the constraint boundary instead of on a
        flat prior, and the incumbent rule (a max over evaluated points with
        every margin >= 0) means nothing can be lost by having it.

        Silent where there is nothing to offer — a box in which every draw was
        refused before its solver has no nearest design, only a nearest
        REFUSAL, and offering that would be offering a wing nobody flew.
        """
        rep = diagnose.infeasibility_report(rd, constraint_labels=labels)
        near = (rep or {}).get("nearest_miss")
        if not near or not near.get("x"):
            return False
        with ui.row().classes("w-full items-center gap-2"):
            ui.button("Start again from the closest design", icon="restart_alt",
                      on_click=lambda _=None, x=list(near["x"]): _start_from(
                          x, f"starting from the design that came closest — "
                             f"it missed {near['label']} by "
                             f"{abs(float(near['violation'])):.4g}")) \
                .props("unelevated dense no-caps color=primary")
            widgets.hint(
                f"evaluation {int(near['index']) + 1}, short of "
                f"{near['label']} by {abs(float(near['violation'])):.4g} — "
                f"it is evaluated first and never screened away, so the next "
                f"run cannot do worse than knowing about it.")
        return True

    def _render_no_feasible() -> bool:
        """Why the finished run returned no design — True when it said so.

        The design box on this stage IS editable row by row, so the measured
        recommendation (:func:`gui.diagnose.box_moves`: which variable
        correlates with the binding margin, and whether the best attempt was
        already riding its bound) lands on a control the user can actually
        move — and the button below opens it.
        """
        rd = (S.get("run") or {}).get("record")
        if not isinstance(rd, dict) or rd.get("best_x") is not None:
            return False
        labels = _effective_clabels()
        lines = diagnose.infeasibility_lines(
            diagnose.infeasibility_report(rd, constraint_labels=labels))
        # the CLOSED-FORM answer first, where there is one: "your area row
        # stops 4.2x below what this mission needs" is a proof, and the
        # measured lines below it are evidence. A run over a box the mission
        # empties has no margins to report — every draw was refused before
        # its solver — so without this the card could only say that nothing
        # came close, which is true and useless.
        _render_size_conflicts()
        _render_start_from_nearest(rd, labels)
        for text, level in lines:
            widgets.hint(text, level)
        # the lines above are EVIDENCE; everything below is an OFFER, and the
        # two are not conditional on each other. A record with no margins at
        # all — every draw refused before its solver — has no evidence to
        # print and is precisely the failure the reach and the probe were
        # built for, so returning early on an empty `lines` put the offers
        # out of reach of the run that needed them most.
        _render_relax()
        _render_box_probe()
        ui.button("Open the design box", icon="crop_free",
                  on_click=lambda: ctx.select("wing", "box")) \
            .props("outline dense no-caps")
        # ...and it is always True from here: this function is only reached
        # for a run that FINISHED and returned no design, and "the criteria
        # appear here once a run finishes" is the one sentence that state
        # must never get.
        return True

    def _probe_worker():
        """Measure how much of the design box is refused BEFORE its solver.

        The failure this answers looked like a solver problem and was a box
        problem: on the run that motivated ``aerobo.optimize.feasible``, 30 of
        64 draws left the aspect-ratio band the span x area rows overshoot and
        26 more sat above the mission's own wing-loading ceiling — 88 % of the
        box in two gates, and no number on this page said so. The margins
        cannot say it (a refused design has no margin to report), so it is
        measured, in a thread, only when the user asks: refusals are ~134x
        cheaper than a design that flies, but "cheap" is not "free" on a
        family holding an XFOIL sweep.
        """
        if not isinstance(S["run"].get("box_probe"), dict):
            S["run"]["box_probe"] = {}          # declared, so it may be None
        pr = S["run"]["box_probe"]
        # the stamp goes on at the START as well as the end: the poll below
        # repaints on a CHANGED stamp, so a worker that only stamped when it
        # finished left its own spinner unpainted for the whole measurement
        pr.update(busy=True, payload=None, error=None, stamp=time.time())
        try:
            # 64 draws answer "how much of this box is refused"; naming a ROW
            # needs the draws that SURVIVE, and on a box that is 90 % refused
            # 64 of them leave about five — too few to tell a dead span from
            # the span five points happen to cover (api._span_p_value). So a
            # family whose refusals are cheap draws enough to answer both,
            # and one holding an XFOIL sweep keeps the small probe.
            n = 64 if sp().slow else 192
            pr["payload"] = api.box_refusal_probe(config.build_cfg(S), n=n)
        except Exception as exc:      # noqa: BLE001 — a view, never fatal
            pr["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            pr["busy"] = False
            pr["stamp"] = time.time()

    def _render_box_probe():
        """The probe's own block: a button, a spinner, or what it measured."""
        pr = (S.get("run") or {}).get("box_probe") or {}
        if pr.get("busy"):
            with ui.row().classes("items-center gap-2"):
                ui.spinner(size="sm")
                ui.label("measuring how much of the design box is refused "
                         "before it is solved").classes("hint")
            return
        if pr.get("error"):
            widgets.hint(f"the design box could not be measured: "
                         f"{pr['error']}", "warn")
            return
        payload = pr.get("payload")
        if not payload:
            ui.button("Measure the design box", icon="rule",
                      on_click=lambda: threading.Thread(
                          target=_probe_worker, daemon=True).start()) \
                .props("outline dense no-caps")
            return
        n, nr = int(payload["n"]), int(payload["n_refused"])
        nf = int(payload["n_feasible"])
        ni = int(payload.get("n_solved_inadmissible") or 0)
        widgets.hint(
            f"Of {n} draws over this box, {nr} were REFUSED before their "
            f"solver ran, {ni} flew and then missed a limit, and {nf} were "
            f"ADMISSIBLE. A refusal is not a bad design — it is a design the "
            f"box allows and the physics does not accept, so a box made mostly "
            f"of them spends the search on nothing. The two take opposite "
            f"fixes: a refusal is a box to narrow, a missed limit is a design "
            f"trade.",
            "warn" if nr > n // 2 else "")
        for row in (payload.get("reasons") or [])[:4]:
            widgets.hint(f"    · {row['n']} x  {row['reason']}", "")
        # ...and WHICH ROW OF THE BOX that leaves dead. The gate names what
        # the physics objected to; this names the control to move, which is
        # the question a user standing in front of the design box has.
        rows = payload.get("rows") or []
        if rows:
            widgets.hint(
                "The design box itself is the limiting factor here — every "
                "draw that survived came from a part of these rows:", "warn")
            for r in rows[:3]:
                lo, hi = r["bounds"]
                klo, khi = r["kept"]
                widgets.hint(
                    f"    · {r['label']}: all {r['n_kept']} survivors sat "
                    f"between {klo:.4g} and {khi:.4g}, of a row searched over "
                    f"{lo:.4g}–{hi:.4g} — {100 * r['dead_frac']:.0f} % of it "
                    f"(the {r['end']}) produced nothing. Narrowing it there "
                    f"spends the budget where the designs are.", "")
            widgets.hint(
                "That is a MARGINAL: the gates are joint conditions in "
                "several rows at once, so a dead span means nothing survived "
                "there WITH THE REST OF THIS BOX — not that the row is wrong.")

    # ------------------------------- reaching for the box nearest this one
    #
    # Every other sentence on this card ends by handing the user a control.
    # This one answers instead: which rows would have to move, by how little,
    # and what design is out there. Three properties make it a reach rather
    # than "search a bigger box until something turns up":
    #
    #   · only rows this run's own evidence implicates move — a closed-form
    #     gate that PROVES the box empty first, a fitted slope of the binding
    #     margin second, and nothing at all where neither speaks;
    #   · one press is one search, at the budget the user already paid, on a
    #     COPY of the problem. It is not a RunJob and never touches the
    #     manager: ``RunManager.start`` REPLACES its queue, so routing this
    #     through it would delete the run the user is looking at and repoint
    #     the readouts and the convergence trace at a box nobody asked for;
    #   · the answer reported is the closest FEASIBLE design the reached run
    #     saw to the box the user drew — not its best-scoring one. The search
    #     maximises the user's objective (so the wing is a good wing); the
    #     ranking is by distance (so the wing is a near one).
    #
    # Nothing here writes a bound. ``_relax_adopt`` is the only writer, it
    # runs on a press, and it moves only the rows the ANSWER is outside — not
    # the rows the search was given.

    def _relax() -> dict:
        """This stage's reach state.

        ``session.make_session`` DECLARES the key (holding None) so that File
        ▸ New session clears it, which means ``setdefault`` is not enough —
        the key exists and holds None, and the caller would get None back.
        """
        if not isinstance(S["run"].get("relax"), dict):
            S["run"]["relax"] = {}
        return S["run"]["relax"]

    def _relax_plan(step: int | None = None) -> dict:
        """The plan, re-derived on every render — never cached.

        Two builds and a closed-form gate, measured in milliseconds, and
        derived so it cannot describe a session the user has since edited
        (the staleness comparison is inside it). Never fatal: a view that
        cannot build shows no card, the same contract ``_size_conflicts``
        keeps.
        """
        rd = (S.get("run") or {}).get("record")
        if not isinstance(rd, dict):
            return {}
        n = int(_relax().get("step") or 0) if step is None else int(step)
        try:
            return relax_mod.plan(S, rd, step=n)
        except Exception as exc:      # noqa: BLE001 — a view, never fatal
            return {"verdict": "blocked", "moves": [], "blocked": [],
                    "labels": [], "bounds": None,
                    "reason": f"the reach could not be planned: "
                              f"{type(exc).__name__}: {exc}"}

    def _relax_answer(plan: dict, rrd: dict) -> dict:
        """What the reached run found, read against the USER'S box."""
        B, labels = plan.get("bounds"), plan.get("labels")
        near = diagnose.closest_feasible(rrd, B, labels)
        if near is not None:
            return dict(near, kind="found", n_evals=rrd.get("n_evals"))
        rep = diagnose.infeasibility_report(
            rrd, constraint_labels=_effective_clabels())
        return {"kind": "refused", "n_evals": rrd.get("n_evals"),
                "refused_fraction": diagnose.refused_fraction(rrd),
                "own_refused_fraction": diagnose.refused_fraction(
                    (S.get("run") or {}).get("record")),
                "nearest": (rep or {}).get("nearest_miss")}

    def _relax_worker(plan: dict):
        """One search, on a copy of the problem, in a thread (no ``ui.*``)."""
        st = _relax()
        # the stamp goes on at the START as well as on every progress tick:
        # poll repaints on a CHANGED stamp, and api.run's first callback does
        # not arrive until the first evaluation has been PAID FOR — which on a
        # slow family is a minute of a button that looks like it did nothing
        st.update(busy=True, stop=False, answer=None, error=None, plan=plan,
                  n=0, budget=int(plan.get("budget") or 0), record=None,
                  stamp=time.time())
        try:
            cfg = api.RunConfig(**plan["cfg_dict"])
            api.check_wing_objective(cfg.flags)

            def progress(i, best, **kw):
                st["n"] = int(i)
                st["stamp"] = time.time()

            # STOPPING is a stop rule and never an exception out of the
            # progress callback: raising there throws away every evaluation
            # already paid for, which on an XFOIL family is the whole run.
            res = api.run(cfg, progress_cb=progress,
                          stop_rule=lambda i, best: bool(st.get("stop")),
                          results_dir=str(v1.RESULTS_DIR),
                          # the reach searches a WIDENED box, which is a
                          # different problem key — so it shares nothing with
                          # the run that prompted it, and only repeats of the
                          # reach itself come back off disk
                          eval_cache=session.EVAL_CACHE)
            rrd = res.to_dict()
            st["record"] = rrd
            st["answer"] = _relax_answer(plan, rrd)
        except Exception as exc:      # noqa: BLE001 — surfaced on the card
            st["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            st["busy"] = False
            st["stop"] = False
            st["stamp"] = time.time()

    def _relax_run(step: int = 0):
        if ctx.manager.running or _relax().get("busy"):
            ui.notify("a run is already in progress", type="warning")
            return
        plan = _relax_plan(step)
        if plan.get("verdict") != "ready":
            # the REASON, on the card, and not only as a toast: a button that
            # offered "one more search of 35 evaluations" and answers with a
            # disappearing "nothing to reach for" has told the user nothing
            _relax().update(step=int(step), plan=plan, busy=False,
                            answer=None, stamp=time.time())
            ui.notify(plan.get("reason") or "there is nothing here to reach "
                                            "for", type="info")
            _render_run()
            return
        _relax()["step"] = int(step)
        ctx.log(f"reaching for the nearest box that holds a design — "
                f"{len(plan['moves'])} row(s) moved, "
                f"{plan.get('budget')} evaluations, on a COPY of this "
                f"problem. Your design box is not changed.", "info")
        threading.Thread(target=_relax_worker, args=(plan,),
                         daemon=True).start()
        _render_run()          # ctx.refresh paints the shell chrome, not views
        ctx.refresh()

    def _relax_stop():
        st = _relax()
        if not st.get("busy"):
            ui.notify("nothing is running", type="info")
            return
        st["stop"] = True
        ctx.log("stop requested — the current evaluation finishes first, and "
                "everything already paid for is kept", "warn")

    def _relax_adopt():
        """Move the rows the ANSWER needed, and start the next run from it.

        The only writer in this whole feature, and it writes where a TYPED
        bound writes (``W["bounds"]``), so the box view reports the row's
        source as the user's. A number that came out of a search does not get
        to look like a default.
        """
        st = _relax()
        plan, ans = st.get("plan") or {}, st.get("answer") or {}
        x = ans.get("x")
        if ans.get("kind") != "found" or not x:
            ui.notify("there is no design to adopt", type="warning")
            return
        rows = relax_mod.minimal_box(plan.get("bounds"), plan.get("labels"), x)
        before = {k: W["bounds"].get(k) for k in rows}
        had = session.start_design(S)
        for label, band in rows.items():
            W["bounds"][label] = [float(band[0]), float(band[1])]
            session.bounds_source(S).pop(label, None)
        err = session.start_from_design(S, [float(v) for v in x])
        if err:
            for label, was in before.items():
                if was is None:
                    W["bounds"].pop(label, None)
                else:
                    W["bounds"][label] = was
            ui.notify(err, type="warning")
            ctx.log(f"nothing adopted: {err}", "warn")
            return
        ctx.log(("no row had to move — that design was already inside your "
                 "box; " if not rows else
                 "moved " + ", ".join(f"{k} to {v[0]:.4g}–{v[1]:.4g}"
                                      for k, v in rows.items())
                 + " (yours now, in the design box); ")
                + "the next run starts from this design"
                + (" — it replaced the one you had armed" if had else "")
                + ". Nothing has launched.", "ok")
        _render_box()
        _render_solver()
        _render_run()
        ctx.refresh()

    def _render_relax() -> bool:     # noqa: PLR0915
        """The reach, in whichever of its states it is in. True when it said
        something."""
        st = _relax()
        fresh = _relax_plan()
        # an ANSWER is about the session it was reached from. If that session
        # has since changed, the answer is not wrong — it is about a different
        # question, and leaving it on screen beside the new box is how a user
        # ends up adopting rows for a mission they no longer have.
        if st.get("answer") is not None and fresh.get("verdict") == "stale":
            st.clear()
        plan = (st.get("plan") if (st.get("busy") or st.get("answer")
                                   is not None) else fresh)
        if not plan:
            return False

        if st.get("busy"):
            with ui.row().classes("items-center gap-2"):
                ui.spinner(size="sm")
                ui.label(f"{int(st.get('n') or 0)}/"
                         f"{int(st.get('budget') or 0)} evaluations — "
                         f"searching the reached box").classes("hint")
                ui.button("Stop", icon="stop", on_click=_relax_stop) \
                    .props("outline dense no-caps")
            return True

        for text, level in relax_mod.verdict(dict(st, plan=plan)):
            widgets.hint(text, level)

        ans = st.get("answer") if st.get("plan") else None
        if plan.get("verdict") != "ready":
            return True

        if ans is None:
            step = int(st.get("step") or 0)
            ui.button("Reach for the nearest box that holds a design",
                      icon="open_in_full",
                      on_click=lambda _=None, k=step: _relax_run(k)) \
                .props("unelevated dense no-caps color=primary")
            wall = plan.get("wall")
            widgets.hint(
                f"Works out which rows are provably what stopped this run, "
                f"moves only those, by the smallest amount the evidence "
                f"requires — then searches that box ONCE, at this run's own "
                f"{plan.get('budget')} evaluations and seed {plan.get('seed')}"
                + (f", which took {float(wall):.1f} s"
                   if isinstance(wall, (int, float)) else "")
                + ". It runs on a COPY of this problem: your design box is "
                  "not changed, and this page keeps describing the run you "
                  "launched, unless you adopt the rows afterwards.")
            return True

        if ans.get("kind") == "found":
            with ui.row().classes("w-full items-center gap-2"):
                n = len(ans.get("rows") or [])
                ui.button(("Start the next run from this design" if not n else
                           "Move that row and start from this design" if n == 1
                           else f"Move those {n} rows and start from this "
                                f"design"),
                          icon="crop_free", on_click=_relax_adopt) \
                    .props("unelevated dense no-caps color=primary")
                if n:
                    widgets.hint(
                        "Only those rows move, and only far enough to hold "
                        "this design — not the amount the search was given. "
                        "They become YOUR rows in the design box and say so "
                        "in its source column. Nothing launches: press Launch "
                        "when you are ready.")
        # ...and "reach further" only where the next press is a DIFFERENT
        # search. The scale multiplies a MEASURED move; a gate move is a proof
        # and is taken whole, so on the emptied-mission case — the one this
        # feature was built for — step 2 is bit-identical to step 1 and the
        # button would sell a second full budget for the same answer.
        step = int(st.get("step") or 0)
        if step + 1 < len(relax_mod.STEPS):
            nxt = _relax_plan(step + 1)
            if nxt.get("verdict") == "ready" \
                    and nxt.get("cfg_dict") != plan.get("cfg_dict"):
                ui.button(f"Reach further — one more search of "
                          f"{nxt.get('budget')} evaluations",
                          icon="open_in_full",
                          on_click=lambda _=None, k=step + 1: _relax_run(k)) \
                    .props("outline dense no-caps")
        return True

    def _render_score_block():     # noqa: PLR0915
        sc = _score()
        # A REPORT BELONGS TO THE RECORD IT WAS SCORED FROM. The scorer is
        # started only when a record carries a ``best_x``, and it is the only
        # thing that ever writes these keys — so a run that came back with
        # nothing left the PREVIOUS run's report standing, and both branches
        # below return early on it. The whole no-solution card underneath
        # (the size-conflict proof, the infeasibility lines, "start again
        # from the closest design", the reach, "measure the design box") was
        # then unreachable, and the user read "beats 93 % of the box" beside
        # "best objective —". The stamp is the record's own IDENTITY, not a
        # clock: a loaded session and a re-published seed both arrive without
        # one, and both must not inherit somebody else's rows.
        own = sc.get("report_for") is (S.get("run") or {}).get("record")
        if own and sc.get("scoring"):
            with ui.row().classes("items-center gap-2"):
                ui.spinner(size="sm")
                ui.label("scoring the baseline and the optimised design on "
                         "your criteria").classes("hint")
            return
        if own and sc.get("report_error"):
            widgets.hint(f"the design could not be scored on your criteria: "
                         f"{sc['report_error']}", "warn")
            return
        rep = sc.get("report") if own else None
        rows = wing_score_rows(rep)
        if not rows:
            # A FINISHED RUN WITH NOTHING IN IT is not "no run yet". The
            # score block is never launched without a ``best_x``, so this
            # branch is where a search that satisfied no constraint lands —
            # and "the criteria appear here once a run finishes" is exactly
            # the wrong sentence for a run that has finished. The record
            # carries every margin it measured, so the reason is read off it
            # rather than guessed at (:mod:`gui.diagnose`).
            if not _render_no_feasible():
                widgets.hint("The criteria of the design appear here once a "
                             "run finishes.")
            return
        base = rep.get("baseline") or {}
        opt = rep.get("optimised") or {}
        d = (rep.get("delta") or {}).get("composite")
        with ui.row().classes("w-full items-start gap-4 no-wrap"):
            widgets.readout(
                "baseline J", _cell(base.get("composite")),
                tip="the CENTRE of your design box — the design you get by "
                    "drawing the box and not searching it")
            widgets.readout("optimised J", _cell(opt.get("composite")))
            widgets.readout(
                "change", _cell(d, signed=True),
                color=("" if d is None
                       else theme.GOOD if d >= 0.0 else theme.WARN))
            beats = rep.get("beats")
            if beats is not None:
                widgets.readout(
                    "beats", f"{100.0 * float(beats):.0f}", "% of the box",
                    tip="where this design sits in the population the band "
                        "was measured over — one design against the whole "
                        "box sample, not against its centre")
        if base.get("composite") is None and opt.get("composite") is None:
            widgets.hint(
                "No composite: " + str(opt.get("reason")
                                       or base.get("reason") or "no band")
                + ". The raw criteria are still measured, one row each.",
                "warn")
        with ui.expansion("every criterion, one row each") \
                .props("dense").classes("w-full"):
            widgets.verdict_slots(ui.table(columns=[
                {"name": "metric", "label": "criterion", "field": "metric",
                 "align": "left"},
                {"name": "original", "label": "baseline (raw)",
                 "field": "original", "align": "right"},
                {"name": "new", "label": "optimised (raw)", "field": "new",
                 "align": "right"},
                {"name": "change", "label": "change", "field": "change",
                 "align": "right"}],
                rows=rows, row_key="metric")
                .classes("w-full").props("dense flat bordered"))
            widgets.hint(
                "▲ green is a criterion this design improved on, ▼ red one "
                "it gave up, = grey one that did not move. A criterion you "
                "weighted 0 stays grey either way: it is not in J, so the "
                "search was never holding it.")
        was_target = str(rep.get("objective")) == "composite"
        ref = rep.get("reference") or {}
        widgets.hint(
            ("This IS what the search maximised — the same weighted J, "
             "restated for the baseline so the run's own improvement is "
             "visible. "
             if was_target else
             "These are your criteria under your weights — NOT what the "
             "search maximised (that was the family's own L/D). A criterion "
             "that went backwards is what maximising one number cost. ")
            + "Both designs are scored on one frozen band"
            + (f" ({ref.get('n_feasible', 0)} box samples)"
               if ref.get("sha") else "")
            + ", so a sub-score can go past either end — that is the band "
              "being honest about a design outside the sample's spread, not "
              "an error.")
        if d is not None and d < 0.0:
            widgets.hint(
                "The optimised design scores BELOW the box centre on your "
                "weights. On an L/D run that is the trade being visible. On "
                "a composite run it means the band is stale or the search "
                "ran out of budget before it beat the centre.", "warn")

    # ------------------------------------------------ live named metrics
    def _live_selected() -> list:
        """The metric keys plotted right now.

        Until the user ticks something this is the catalogue's answer for
        this run rather than a stored list, so the panel opens on the
        objective under whatever name this family spells it.
        """
        have = live.keys()
        if not live_cfg["picked"]:
            return metrics.live_metric_defaults(have)
        return [k for k in live_cfg["keys"] if k in have]

    def _live_labels() -> dict:
        """key -> the name the catalogue gives it (metrics.LIVE_METRICS)."""
        primary, more = metrics.live_metric_panel(live.keys(),
                                                  _live_selected())
        return dict(primary) | dict(more)

    def _live_fig() -> go.Figure:
        keys = _live_selected()
        if not keys or len(live.samples) < 2:
            return figstyle.empty(
                "the incumbent's named metrics appear here as it improves",
                260)
        labels = _live_labels()
        fig = go.Figure()
        for i, key in enumerate(keys):
            xs, ys = live.series(key)
            fig.add_scatter(x=xs, y=ys, mode="lines+markers",
                            name=labels.get(key, key),
                            line=dict(width=2,
                                      color=theme.SERIES[i % len(theme.SERIES)]
                                      ),
                            yaxis="y" if i == 0 else "y2")
        fig.update_layout(
            xaxis_title="evaluation",
            yaxis_title=labels.get(keys[0], keys[0]), height=280,
            legend=dict(orientation="h", y=1.12))
        if len(keys) > 1:
            fig.update_layout(yaxis2=dict(
                title=", ".join(labels.get(k, k) for k in keys[1:]),
                overlaying="y", side="right", showgrid=False))
        return fig

    def _live_panel():
        """The incumbent's named metrics: a few offered, the rest one press
        away.

        A wing breakdown reports several dozen numeric keys, and offering
        every one of them as a checkbox put a wall of forty names above a
        plot that can carry two axes — most of them the same number in a
        second spelling. The catalogue de-duplicates and ranks them
        (:func:`gui.metrics.live_metric_panel`): this row is the handful
        that answer a question on sight plus anything already ticked, and
        the expansion under it holds everything else, still selectable.
        """
        with widgets.group_box("Live metrics", pad=False):
            with ui.column().classes("group-pad w-full gap-2"):
                with ui.row().classes("items-center gap-3 no-wrap"):
                    ui.switch("sample the incumbent",
                              value=bool(live_cfg["on"]),
                              on_change=lambda e: _set_live(bool(e.value))) \
                        .props("dense")
                    boxes["run_samples"] = widgets.tag(
                        f"{len(live.samples)} samples", theme.INK_MUTED)
                    if live.error:
                        widgets.tag("sampling failed", theme.BAD)
                widgets.hint(
                    "The progress log records the objective and the margins "
                    "only. These are the NAMED metrics of the best design so "
                    "far, re-evaluated once each time it improves — not once "
                    "per evaluation, which would double the cost of the "
                    "search.")
                picked = _live_selected()
                primary, more = metrics.live_metric_panel(live.keys(), picked)
                if primary:
                    with ui.row().classes("items-center gap-2 flex-wrap"):
                        for key, label in primary:
                            ui.checkbox(
                                label, value=key in picked,
                                on_change=lambda e, k=key:
                                _toggle_metric(k, bool(e.value))) \
                                .props("dense size=xs")
                    if more:
                        with ui.expansion(
                                f"every other reported number ({len(more)})") \
                                .props("dense").classes("w-full"):
                            widgets.hint(
                                "The rest of this run's breakdown — the "
                                "second spelling of a number above, the "
                                "planform read-outs, the remaining "
                                "constraint margins.")
                            with ui.row().classes(
                                    "items-center gap-2 flex-wrap"):
                                for key, label in more:
                                    ui.checkbox(
                                        label, value=key in picked,
                                        on_change=lambda e, k=key:
                                        _toggle_metric(k, bool(e.value))) \
                                        .props("dense size=xs")
                elif live_cfg["on"]:
                    widgets.hint("waiting for the first improvement…")
                if live.error:
                    widgets.hint(live.error, "warn")
            boxes["run_live"] = figstyle.show(_live_fig(), "live_metrics")

    def _set_live(on: bool):
        live_cfg["on"] = bool(on)
        if on and ctx.manager.running:
            job = ctx.manager.current
            if job is not None:
                live.start(cfg_fn=lambda c=job.cfg: c,
                           incumbent_fn=_incumbent,
                           running_fn=lambda: ctx.manager.running)
        elif not on:
            live.stop()
        # the switch is IN the panel this rebuilds, and the panel's shape
        # depends on it — so this goes through the signature (a full build)
        # rather than calling the builder directly
        _tick_run()

    def _toggle_metric(key: str, on: bool):
        """Add or drop one series — WITHOUT rebuilding the view it sits in.

        The first tick takes ownership: until then the plot shows the
        catalogue's defaults, and toggling one of them off has to leave the
        other two rather than resetting to them on the next render. Only
        the figure is redrawn; rebuilding the panel would scroll the page
        back to the top on every click, and close the expansion the click
        may have come from.
        """
        keys = list(_live_selected())
        if on and key not in keys:
            keys.append(key)
        if not on and key in keys:
            keys.remove(key)
        live_cfg["keys"] = keys
        live_cfg["picked"] = True
        figstyle.update(boxes.get("run_live"), _live_fig(), "live_metrics")

    # ------------------------------------------------------------ polling
    def poll() -> bool:
        mgr = ctx.manager
        sc = _score()
        # the two background jobs this stage owns besides the search itself:
        # the band sweep (solver tab) and the criteria report (run tab)
        if sc.get("stamp") != seen["band"]:
            seen["band"] = sc.get("stamp")
            if sc.get("error"):
                ctx.log(f"the band could not be measured: {sc['error']}",
                        "error")
            elif sc.get("band"):
                b = sc["band"]
                ctx.log(f"band measured over {b.get('n_feasible', 0)} of "
                        f"{b.get('n_samples', 0)} box samples "
                        f"({len(b.get('bounds') or {})} criteria)", "ok")
            if S["ui"]["selected"] == "wing" \
                    and S["ui"]["tab"]["wing"] == "solver":
                _render_solver()
        if sc.get("report_stamp") != seen["score"]:
            seen["score"] = sc.get("report_stamp")
            if S["ui"]["selected"] == "wing" \
                    and S["ui"]["tab"]["wing"] == "run":
                _render_run()
        # the REACH and the box probe: both write into S["run"] from their
        # own threads and neither is a RunJob, so the manager's version never
        # moves for them and nothing would repaint without this
        moved = False
        rx = (S.get("run") or {}).get("relax") or {}
        if rx.get("stamp") != seen["relax"]:
            seen["relax"] = rx.get("stamp")
            if rx.get("error"):
                ctx.log(f"the reach failed: {rx['error']}", "error")
            elif rx.get("answer") is not None and not rx.get("busy"):
                a = rx["answer"]
                if a.get("kind") == "found":
                    ctx.log(
                        "the reach found a design that meets every limit"
                        + (" INSIDE the box you already drew — your box is "
                           "not what stopped this run"
                           if not a.get("rows") else
                           f", {a['dinf']:.3g} of a row-width outside your "
                           f"box on {len(a['rows'])} row(s)")
                        + ". Your design box is unchanged.", "ok")
                else:
                    ctx.log("the reach found nothing either — the limit, not "
                            "the box, is what has no solution inside it",
                            "warn")
            moved = True
        # the design box is MEASURED for this mission BEFORE this stage is
        # opened, and the measurement is started from here rather than from
        # `_render_box`: a render fires on every keystroke in the box, and a
        # worker started from a paint would be started by the paint the last
        # worker caused.
        #
        # NOT on arrival, which is what it used to be. The measurement takes
        # about a second on a fast family, so a session that walked into this
        # stage watched the box open on the mission's own band and then
        # narrow under them — and every menu on this card does the same thing
        # again, because the problem it measured is gone. The user's own
        # words: "when the user finishes the airfoil the design box should
        # already be generated". So it runs as soon as this stage is
        # REACHABLE, while stage 2 is still on screen.
        #
        # A SLOW family still waits for arrival. 48 draws each holding an
        # XFOIL sweep is minutes of CPU, and stage 2 is itself running XFOIL
        # on those families — starting both is a session that competes with
        # itself for cores, to fill in a box the user may never walk to.
        if _auto_ready() and _auto_wanted():
            _auto_kick()
            moved = True
        auto_tick = _auto_state().get("tick")
        if auto_tick != seen.get("auto"):
            seen["auto"] = auto_tick
            _render_box()
            _render_solver()
            moved = True
        pr_stamp = ((S.get("run") or {}).get("box_probe") or {}).get("stamp")
        if pr_stamp != seen["probe"]:
            seen["probe"] = pr_stamp
            moved = True
        if moved and S["ui"]["selected"] == "wing" \
                and S["ui"]["tab"]["wing"] == "run":
            _render_run()
        if live.version != seen["sampler"]:
            seen["sampler"] = live.version
            if S["ui"]["selected"] == "wing" \
                    and S["ui"]["tab"]["wing"] == "run":
                # IN PLACE. A new sample is a new point on one trace, not a
                # new page: rebuilding the view here is what scrolled the
                # user away from the metrics they were reading
                _tick_run()
        if mgr.version == seen["version"]:
            return False
        seen["version"] = mgr.version
        job = mgr.current
        if job is None:
            return False
        if job.status == "running":
            # ...COUNTING WHAT THIS RUN INHERITED. The run view's own
            # read-out already adds it (``_flown_text``); the shell-wide
            # status bar did not, so a continuation of 53 by 4 announced
            # "1/57 … 4/57" across the top of every stage — which is
            # exactly what "it started from the beginning" looks like, on
            # the one widget that is visible from everywhere.
            done = _resumed_of(job) + len(job.records)
            ctx.status(f"{job.label} — {done}/{job.budget}",
                       "busy", done / max(1, job.budget))
        if S["ui"]["selected"] == "wing" and S["ui"]["tab"]["wing"] == "run":
            # every evaluation lands here. ``_tick_run`` rebuilds the view
            # when its STRUCTURE moved (a job finished, the first record
            # arrived) and otherwise writes the new numbers into the
            # elements already on screen
            _tick_run()
        # EVERY TERMINAL JOB, not the manager's cursor. ``launch`` queues one
        # job per repeat seed, and ``mgr.current`` returns the running job
        # first and otherwise the LAST terminal one — while the worker flips
        # job k done and job k+1 running within a few bytecodes, so a 0.5 s
        # heartbeat effectively never saw an intermediate job terminal.
        # Measured over three seeds scoring 1.111 / 2.222 / 0.333, the shell
        # logged one completion and published 0.333: stage 4 and the
        # Properties pane presented an ARBITRARY member of the queue as "the
        # design the search found", against this repo's own measurement that
        # identical BO-on-XFOIL runs land 3–4 % apart. So each job is logged
        # exactly once as it lands, and what is published is the BEST
        # finished one. ``mgr.current`` is untouched — it is the running
        # view's cursor, not the publication rule.
        #
        # The seen-set is keyed by id() and RESET with the queue: a RunJob is
        # an ``eq=True`` dataclass and therefore unhashable, and an id from a
        # freed job of a previous run could otherwise collide with a new one
        # and swallow its publication.
        if seen.get("queue") is not mgr.jobs:
            seen["queue"] = mgr.jobs
            seen["published"] = set()
            seen["records"] = {}
        landed = [j for j in mgr.jobs
                  if j.status in ("done", "error", "cancelled")
                  and id(j) not in seen["published"]]
        for j in landed:
            seen["published"].add(id(j))
            if j.status == "error":
                S["run"]["error"] = j.error
                ctx.log(f"run failed: {j.error}", "error")
                ctx.status("run failed", "error", None)
                continue
            if j.result is None:
                continue
            rec = j.result.to_dict()
            seen["records"][id(j)] = rec
            # a search that found nothing admissible is an OUTCOME, not
            # a missing number: ``to_dict`` always emits best_score, so
            # the key is present and None (a constrained family stopped
            # before its first feasible design) and formatting it with
            # ":.6g" raised TypeError
            got = rec.get("best_score")
            ctx.log(f"{j.label} {j.status} — "
                    + (f"best objective {got:.6g}"
                       if isinstance(got, (int, float))
                       else "no feasible incumbent")
                    + f" after {rec.get('n_evals', '?')} evaluations",
                    "ok")
        if landed:
            best_rd, best_f = None, None
            for j in mgr.jobs:
                rec = seen["records"].get(id(j))
                if rec is None:
                    continue
                got = rec.get("best_score")
                if isinstance(got, (int, float)) \
                        and (best_f is None or float(got) > best_f):
                    best_f, best_rd = float(got), rec
            # a queue in which NOTHING was feasible still has an outcome to
            # publish: the whole no-solution card is read off a record, so
            # the most recent one stands in for a best that does not exist
            if best_rd is None:
                best_rd = next(
                    (seen["records"][id(j)] for j in reversed(mgr.jobs)
                     if id(j) in seen["records"]), None)
            if best_rd is not None \
                    and (S.get("run") or {}).get("record") is not best_rd:
                S["run"]["error"] = None
                ctx.act("set_result", best_rd)
                # the STATUS first, then the sentence: anything that raises
                # while wording the log line would otherwise leave the shell
                # claiming a finished run is still in progress, for the rest
                # of the session, with no way back
                if not mgr.running:
                    ctx.status("run complete", "ok", None)
                if len(seen["records"]) > 1:
                    ctx.log(f"showing the best of {len(seen['records'])} "
                            f"finished seed(s)"
                            + (f" — objective {best_f:.6g}"
                               if best_f is not None else "")
                            + ". Every seed's own record is on disk.", "ok")
                # …and what it bought on the OTHER five criteria, scored in
                # the background: a search maximised one number, and the
                # weights named ten
                if best_rd.get("best_x"):
                    # claimed for this record HERE rather than on the worker
                    # thread: the repaint below can beat the thread's first
                    # statement, and a block that has not been claimed yet
                    # draws as "no report" for one heartbeat
                    _score().update(scoring=True, report=None,
                                    report_error=None, report_for=best_rd)
                    threading.Thread(target=_score_worker, args=(best_rd,),
                                     daemon=True).start()
            # ...and repaint, because the record lands AFTER the render above.
            # The whole no-design card — the closed-form proof, the margins,
            # the nearest miss, the reach — is read off S["run"]["record"], so
            # the view painted on the tick a run FINISHES was painted without
            # it. Measured in the running shell: a run that came back with
            # nothing showed "the criteria appear here once a run finishes"
            # and only became a card when the user happened to switch tabs.
            if S["ui"]["selected"] == "wing" \
                    and S["ui"]["tab"]["wing"] == "run":
                _render_run()
        return True

    ctx.add_poll(poll)

    # ------------------------------------------------------------ wiring
    ctx.on_render("wing", "type", _render_type)
    ctx.on_render("wing", "box", _render_box)
    ctx.on_render("wing", "solver", _render_solver)
    ctx.on_render("wing", "run", _render_run)
    # the Solver card is the stage's own read-out of the design vector: the
    # budget, the dimension, the expected wall clock, the flag list and the
    # box-vs-mission conflicts. Every box control on the neighbouring view
    # moves it, and each of them used to have to remember to.
    ctx.on_derived("wing", "solver")
    ctx.register("launch", launch)
    ctx.register("cancel", cancel)
    ctx.register("continue_run", continue_run)
    # the same handler the menus call — registered so the shell (and the
    # smoke tests) can drive a configuration change through the REAL path,
    # normalisation and all, instead of poking the choices dict
    ctx.register("set_choice", set_choice)
    ctx.register("set_winglet", set_winglet)
    # ...and the SECOND SURFACE's tip device, asked as the same kind of
    # question in the same vocabulary
    ctx.register("set_tail_tip", _set_tail_tip)
    ctx.register("take_spiral_dihedral", _take_spiral_dihedral)
    ctx.register("set_wing_cant", _set_wing_cant)
    ctx.register("set_fuselage", _set_fuselage)
    # …and the objective's own three, for the same reason: a smoke test (and
    # the shell) drives them through the REAL handler, normalisation and all
    ctx.register("set_wing_objective", _set_wing_objective)
    ctx.register("set_wing_weight", _set_wing_weight)
    ctx.register("measure_band", _measure_band)
    ctx.register("set_row_on", _set_row_on)
    ctx.register("set_row_fixed", _set_row_fixed)
    ctx.register("set_fixed_value", _set_fixed_value)
    # the mission recommendation, per surface table ("wing" | "aft"): asking
    # for one and taking it are two actions, because they are two decisions
    ctx.register("reset_box", _reset_box)
    ctx.register("recommend_box", _make_recommendation)
    ctx.register("take_recommendation", _take_recommendation)
    # ...and the one that MEASURES AND APPLIES, which is what the stage does
    # on arrival. Registered so the shell, the smoke tests and a user pressing
    # "measure it again" all drive the same path.
    # the readiness GATE, registered so a test can drive the real predicate
    # rather than restate it: "the box is measured before this stage opens"
    # is a claim about `_auto_ready`, and a test that asserts the source text
    # of the caller would pass over any predicate at all
    ctx.register("auto_ready", _auto_ready)
    ctx.register("auto_recommend", lambda force=True: (
        _auto_recommend(force=force), _render_box(), _render_solver(),
        ctx.refresh())[0])
    ctx.register("set_bound", _set_bound)
    ctx.register("reset_box", _reset_box)
    ctx.register("set_span_searched", _set_span_searched)
    ctx.register("set_planform", _set_planform)
    ctx.register("set_span", _set_span)
    ctx.register("set_size_mode", _set_size_mode)
    ctx.register("set_size_chord", _set_size_chord)
    ctx.register("set_tail_arm", _set_arm)
    ctx.register("set_trim_number", _set_trim_number)
    ctx.register("set_chord_limit_on", _set_chord_limit_on)
    ctx.register("set_chord_limit", _set_chord_limit)
    ctx.register("set_pair_area_mode", _set_pair_mode)
    ctx.register("set_pair_area_exact", _set_pair_exact)
    ctx.register("set_pair_area_value", _set_pair_value)
    ctx.register("set_pair_area_band", _set_pair_band)
    ctx.register("adopt_pair_area_total", _adopt_pair_total)
    ctx.register("set_ar_limit_on", _set_ar_limit_on)
    ctx.register("set_ar_limit", _set_ar_limit)
    # …and the SECOND SURFACE's half of the same question. Registered for the
    # same reason: the tail's limits carry the same refuse-and-roll-back path
    # as the wing's, and until now no test could reach it through the real
    # handler at all
    ctx.register("set_tail_limit_on", _set_tail_limit_on)
    ctx.register("set_tail_limit", _set_tail_limit)
    ctx.register("set_chord_trend", _set_chord_trend)
    ctx.register("set_chord_dev", _set_chord_dev)
    ctx.register("set_chord_law", _set_chord_law)
    ctx.register("set_tip_chord", _set_tip_chord)
    ctx.register("fix_section_point", _fix_section_point)
    # the reach: offered on the Convergence view when a run returns nothing,
    # registered so the shell and the tests drive it through the REAL path
    ctx.register("relax_reach", _relax_run)
    ctx.register("relax_stop", _relax_stop)
    ctx.register("relax_adopt", _relax_adopt)
