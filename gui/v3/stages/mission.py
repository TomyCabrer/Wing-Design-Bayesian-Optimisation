"""Stage 1 — MISSION: the operating point everything downstream derives from.

What the user states here is small on purpose: the medium, the design
weight, the speed, the altitude or depth, the SIZE — and HOW MANY
LIFTING SURFACES the vehicle has: a tandem pair, and whether there is a tail
(an elevator, under water). That last one belongs here and not in stage 3
because it is not a wing setting: it decides how many surfaces exist to be
designed at all, and therefore how many SECTIONS stage 2 has to choose. The
second surface's geometry — its arm, its height, its planform, its tip
device — stays in stage 3 with the rest of the design box.

NO PLANFORM is stated — neither taper nor a flown aspect ratio. Both are
things the design process answers: taper is a design variable of every wing
family, and the aspect ratio a run flies is b²/S with the span stated on
stage 3, so this stage reports what it can derive without one and hands the
rest on:

    density, dynamic pressure  ->  CL design = W / (q S)   [needs no planform]
    area                                                   [the trim reference]

The size is ONE stored number — the reference area — asked in whichever of
its three faces the user actually has (``session.SIZE_STATEMENTS``): the
wing loading a specification carries (S = W/(W/S)), the area itself, or a
span with a rough aspect ratio (S = b²/AR). They are three labels on one
quantity, not three quantities, so no two of them can disagree.

The span in that third face is a SKETCH OF THE AREA and nothing more. It
writes no span: the span is stage 3's constraint, stated there once, and
where stage 3 has been given one that differs this card says so
(``session.mission_span_disagrees``) rather than showing stage 3's number in
a field that cannot set it. Its aspect ratio is stage 2's existing estimate,
written through stage 2's own setter, so there is still exactly one.

The chord — and so the Reynolds number the section is screened at — needs
an aspect ratio, and that number is an ESTIMATE owned by stage 2
(``session.section_aspect_ratio``). It is quoted here, attributed, so the
mission still shows the point it leads to without pretending to have chosen
a wing.

The load and the area ARE the wing's trim target in stage 3, so a mission
accepted here cannot silently disagree with what is later flown. Accepting
the mission is what unlocks the rest of the pipeline; editing it afterwards
un-accepts it, because a section screened at the old Reynolds number is no
longer the section this mission asked for.
"""

from __future__ import annotations

import plotly.graph_objects as go

from .. import config, figstyle, session, theme, widgets

#: what the design load IS, per medium — a rear wing's load is downforce,
#: not weight, and calling it weight would misname the only number the car
#: mission actually states
LOAD_LABEL = {"air": "design weight", "water": "design lift",
              "track": "design downforce"}

#: what each FLUID face of the airfoil-only flow form means. Air and water
#: derive their own state and refuse a typed density (``api.flow_point``);
#: the custom face is the one that takes rho and mu verbatim.
FLUID_NOTES = {
    "air": "ISA air: the altitude sets the density, the viscosity and the "
           "speed of sound, so the Mach number follows the speed rather "
           "than being typed beside it.",
    "water": "Water at ~20 C: density and viscosity come from the named "
             "water and do not vary with depth at this tier, so there is no "
             "depth to state — a 2-D section carries no cavitation "
             "constraint either.",
    "custom": "Your fluid, stated outright: density, viscosity and Mach "
              "number are used exactly as typed. This is the face for a "
              "tunnel, a scaled test, or a fluid this repo has no model of.",
}


#: WHAT EACH EMPENNAGE IS, said in terms of the surfaces it creates — which
#: is why the question is asked on this stage at all. The aerodynamic notes
#: (downwash relief, the cos²Γ equivalent tail, the wake model) stay on stage
#: 3 beside the design box they qualify; these say how many surfaces there
#: are and which stages therefore exist.
#: the ONE LINE under the empennage select, per layout. The paragraph that
#: used to sit there in full is :data:`TAIL_TYPE_MISSION_NOTES`, and it is
#: now behind the ``?`` beside the select.
TAIL_TYPE_MISSION_SHORT = {
    "conventional": "tailplane and fin — stages 2.5 and 2.7",
    "t_tail": "tailplane on the fin's tip — sized together",
    "v_tail": "one canted pair — no fin, no stage 2.7",
    "canard": "trimmer ahead of the wing — carries UP load",
}

TAIL_TYPE_MISSION_NOTES = {
    "conventional": "Two separate surfaces on the body: a tailplane aft and "
                    "a fin under it. Each is designed on its own stage — 2.5 "
                    "for the tailplane, 2.7 for the fin — and each carries "
                    "its own section, at its own chord and its own Reynolds "
                    "number.",
    "t_tail": "The two surfaces are CONNECTED: the tailplane stands on the "
              "fin's tip, so the fin runs from the body up to it and its "
              "span IS the tailplane's height. Sizing one sizes the other "
              "(fin.size_fin), and the tailplane sits out of the wing's "
              "trailing sheet. Both still carry their own section.",
    "v_tail": "ONE surface, not two: the panels are canted, so the same pair "
              "makes pitch authority and yaw stiffness. There is no separate "
              "fin — stage 2.7 does not exist for this layout — and no "
              "separate tailplane either: what stage 2.5 designs is the V's "
              "own panel. The cant is asked on stage 3, where it is a "
              "geometry of the surface rather than a count of surfaces.",
    "canard": "The trimming surface sits AHEAD of the wing, so it carries UP "
              "load rather than a download and the balance runs the other "
              "way. The fin is a separate surface as on a conventional "
              "layout, and both are still designed on their own stages.",
}


def build(ctx):     # noqa: PLR0915  (one stage, built whole)
    from nicegui import ui

    from aerobo import api
    from aerobo.mission import G0
    from gui import nice_app as v1

    S = ctx.S
    boxes: dict = {}

    # ------------------------------------------------------------- helpers
    def touch(*, size_field: bool = True):
        """A mission edit invalidates the acceptance and every derived view.

        ``size_field`` is False ONLY when the edit came from one of the size
        fields themselves: rebuilding a number field from inside its own
        on_change swallows the rest of the number being typed, which is the
        focus trap those fields were split into their own container for.
        Every OTHER edit has to repaint it, because the size fields all show
        a number derived from the mission — the weight is the numerator of
        the loading — and the field is the only place that number is shown
        as a control. Left stale it does not merely
        read wrong: type W = 1000 N and the box still says 65.2802 while the
        hint under it says S = W/(W/S) = 10 m²; one step-arrow press then
        submits 70.2802, ``is_echo`` measures it against the CURRENT 100
        N/m², finds no echo, and the loading is dropped to 70.28 — growing
        the reference area 10 -> 14.229 m² from a single arrow key.
        """
        S["mission"]["accepted"] = False
        session.sync_wing_from_mission(S)
        if size_field:
            _render_size_fields()
        _render_load_note()
        _render_area_note()
        # the stated-vs-allowed comparison moves with BOTH of them, so it is
        # redrawn on every mission edit rather than only when the diagram is
        _render_ws_cap()
        _render_ws_conflict()
        # the diagram is DRAWN at the operating point and at the loading
        # this mission states, so a weight, a speed or an area typed on the
        # left moves the picture too. The fields it is asked through are not
        # rebuilt — only the point it quotes and the answer it derives.
        _render_ws_point()
        _render_ws_derived()
        _render_derived()
        ctx.render("mission", "point")
        ctx.render_when_shown("wing")
        # the operating view is this handler's OWN view: every container in
        # it that quotes the mission was just redrawn by name above, and
        # rebuilding the whole view would take the field being typed into
        # with it. Everything else derived (the Search tab, both screen
        # forms) follows.
        ctx.refresh(("mission", "operating"))

    def setter(key: str):
        def _set(e):
            if e.value in (None, ""):
                return
            # the field SHOWS a rounded number (the published weight is an
            # exact 652.8022140185119 that no 60px box can render); when
            # that rounded number is what comes back, nothing was typed and
            # the exact value must survive, or an untouched mission would
            # start sending W_N to the solver
            if widgets.is_echo(e.value, S["mission"][key]):
                return
            try:
                S["mission"][key] = float(e.value)
            except (TypeError, ValueError):
                return
            touch()
        return _set

    def _set_loading(e):
        """W/S -> reference area. The area stays the stored quantity (every
        solver speaks area); this is the control the user actually thinks
        in, and the derived area is shown right under it."""
        if e.value in (None, ""):
            return
        if widgets.is_echo(e.value, session.wing_loading(S)):
            return
        if not session.set_wing_loading(S, e.value):
            ui.notify("wing loading must be positive", type="negative")
            return
        # the field this came FROM must not be rebuilt under the cursor
        touch(size_field=False)

    def _set_area(e):
        """The reference area, typed. The SAME stored number the loading
        field writes — this is the other face of it, not a second size."""
        if e.value in (None, ""):
            return
        if widgets.is_echo(e.value, session.reference_area(S)):
            return
        if not session.set_reference_area(S, e.value):
            ui.notify("reference area must be positive", type="negative")
            return
        touch(size_field=False)

    def _set_sketch_span(e):
        """A span typed HERE sizes the AREA (S = b²/AR) and nothing else.

        It is not the wing's span — stage 3 states that, once — and the note
        under the field says so, loudly, whenever the two differ.
        """
        if e.value in (None, ""):
            return
        if widgets.is_echo(e.value, session.mission_span(S)):
            return
        if not session.set_size_from_span_ar(S, span=e.value):
            ui.notify("span must be positive", type="negative")
            return
        touch(size_field=False)

    def _set_sketch_ar(e):
        """The aspect ratio typed here is stage 2's estimate — the session's
        only one — and moving it holds the span and moves the area."""
        if e.value in (None, ""):
            return
        if widgets.is_echo(e.value, session.section_aspect_ratio(S)):
            return
        if not session.set_size_from_span_ar(S, ar=e.value):
            ui.notify("aspect ratio must be positive", type="negative")
            return
        touch(size_field=False)
        # the estimate is stage 2's field as much as this one's, and a value
        # changed in one place must not read as the old one in the other
        ctx.render_when_shown("airfoil")

    def _set_size_statement(e):
        """Switch which face of the size is asked. Stores no size at all."""
        if not session.set_size_statement(S, e.value):
            return
        _render_size_fields()
        _render_area_note()

    def _render_size_fields():
        """The one control for the size, in whichever unit was chosen.

        Exactly one of the three forms is on screen at a time. Showing two of
        them would be two fields for one number, which is the pair that can
        disagree that this stage has spent its whole life not having.
        """
        box = boxes.get("size_fields")
        if box is None:
            return
        box.clear()
        mode = session.size_statement(S)
        with box:
            if mode == session.SIZE_AS_AREA:
                widgets.number_field(
                    "reference area S",
                    widgets.shown(session.reference_area(S)),
                    _set_area, unit="m²", step=0.5, width="w-32",
                    tip="the area the design lift coefficient is defined "
                        "on; the wing loading follows from it and the load")
            elif mode == session.SIZE_AS_SPAN_AR:
                widgets.number_field(
                    "span (sketch)", widgets.shown(session.mission_span(S)),
                    _set_sketch_span, unit="m", step=0.1, width="w-32",
                    tip="how wide this area is at the aspect ratio below — "
                        "it sizes the AREA; the span the run flies is "
                        "stage 3's to state")
                widgets.number_field(
                    "aspect ratio (estimate)",
                    widgets.shown(session.section_aspect_ratio(S)),
                    _set_sketch_ar, step=0.5, width="w-32",
                    tip="stage 2's estimate — the same number the section's "
                        "chord and Reynolds number are derived from")
            else:
                widgets.number_field(
                    "wing loading W/S",
                    widgets.shown(session.wing_loading(S)),
                    _set_loading, unit="N/m²", step=5.0, width="w-32",
                    tip="what a wing is actually specified by; the "
                        "reference area follows from it and the design load")

    def _render_area_note():
        """Every face of the one stored size, whichever one was typed.

        The mode chooses where the cursor goes, not what the user is allowed
        to see: a session that states an area still has a wing loading, and a
        ceiling is applied to it further down this same card.
        """
        box = boxes.get("area")
        if box is None:
            return
        box.clear()
        mode = session.size_statement(S)
        area = session.reference_area(S)
        ws = session.wing_loading(S)
        span = session.mission_span(S)
        ar = session.nominal_aspect_ratio(S)
        with box:
            if area is None or ws is None:
                widgets.hint("The design load and the area do not yet "
                             "describe a wing.", "warn")
                _render_planform_note()
                return
            if mode == session.SIZE_AS_AREA:
                widgets.hint(f"wing loading = W/S = {ws:.4g} N/m² — it "
                             f"follows from the design load and this area, "
                             f"and it is what the ceiling below is applied "
                             f"to.")
            elif mode == session.SIZE_AS_SPAN_AR:
                widgets.hint(f"reference area S = b²/AR = {area:.4g} m², and "
                             f"the wing loading W/S = {ws:.4g} N/m² follows "
                             f"from it and the design load.")
            else:
                widgets.hint(f"reference area S = W/(W/S) = {area:.4g} m² — "
                             f"the area the design lift coefficient is "
                             f"defined on.")
            if mode != session.SIZE_AS_SPAN_AR and span is not None:
                widgets.hint(f"At stage 2's aspect-ratio estimate {ar:.3g} "
                             f"that area is √(AR·S) = {span:.3g} m across — "
                             f"a sketch of the size, not the span: stage 3 "
                             f"states the span.")
            gap = session.mission_span_disagrees(S)
            if gap:
                widgets.hint(f"Stage 3 flies b = {gap[1]:.3g} m, not the "
                             f"{gap[0]:.3g} m this area and estimate imply. "
                             f"The span is stage 3's to state and it has "
                             f"been stated; the aspect ratio this run flies "
                             f"is b²/S = {gap[1] * gap[1] / area:.3g}.",
                             "warn")
        _render_planform_note()

    #: what each diagram input is, in the order the card asks them. The
    #: OPERATING POINT is not in here — speed, density and depth come from
    #: the mission fields on the left, and the aspect ratio from the wing —
    #: so what is left is what the diagram alone needs.
    WS_LABELS = {
        "v_stall_ms": ("stall / approach speed", "m/s"),
        "cl_max": ("CL_max (no section chosen yet)", "—"),
        "cd0": ("zero-lift drag CD0", "—"),
        "oswald_e": ("span efficiency e", "—"),
        "climb_rate_ms": ("climb rate", "m/s"),
        "turn_load_factor": ("sustained turn", "g"),
        "takeoff_distance_m": ("take-off distance", "m"),
        "landing_distance_m": ("landing distance", "m"),
        "v_takeoff_ms": ("fly-up speed", "m/s"),
        # the only CAPABILITY on the card — everything else above is a
        # requirement. It is what closes the diagram into a matching point.
        "twr_available": ("T/W available", "—"),
    }

    #: the six-word line under each requirement. The card asked nine
    #: questions in the vocabulary of the answer ("sustained turn", "g") and
    #: none in the vocabulary of the ask, and they all shared one tip
    #: ("clear the field to drop this requirement") that says what CLEARING
    #: does and nothing about what filling it in means.
    WS_NOTES = {
        "v_stall_ms": "the slowest you require it to fly",
        "cd0": "drag at zero lift, whole aircraft",
        "oswald_e": "how close to elliptic the lift is",
        "climb_rate_ms": "climb you require at this weight",
        "turn_load_factor": "g you require it to sustain",
        "takeoff_distance_m": "ground roll you require",
        "landing_distance_m": "ground roll you require",
        "v_takeoff_ms": "speed it must fly away at",
        "twr_available": "the thrust you have, over weight",
    }

    #: ...and the ones that need a paragraph, behind a ``?``.
    WS_HELP = {
        "v_stall_ms": ("The slowest speed you REQUIRE the aircraft to fly "
                       "at. It is the whole upper bound on wing loading: "
                       "W/S <= ½ρV²CL_max.\n\n"
                       "It opens at a fraction of the cruise speed, but "
                       "never below the speed the family's published wing "
                       "actually flies at plus a small margin — a default "
                       "that refused the mission it opened on is a bug "
                       "about the default, not a requirement."),
        "v_takeoff_ms": ("The speed the foil must lift the craft away at. "
                         "It is the upper bound on wing loading: "
                         "W/S <= ½ρV²CL_max, and it is one END of a "
                         "foiling craft's band — cavitation is the other."
                         "\n\n"
                         "It opens at the speed the family's PUBLISHED foil "
                         "actually flies at plus 5 %, not at a fraction of "
                         "the cruise speed: half of 12 m/s asked a 0.144 m² "
                         "foil to carry 6 kN, which needs CL 2.26 against a "
                         "usable 0.9. Type a slower one and you are asking "
                         "for a bigger foil."),
        "cd0": ("CD0 is the WHOLE AIRCRAFT's drag coefficient at zero "
                "lift, on the same reference area the wing loading uses.\n\n"
                "It fixes where the cruise and climb curves sit, so a guess "
                "here moves the matching point and the W/S ceiling with it. "
                "A clean light aircraft is 0.02-0.03."),
        "oswald_e": ("How close to elliptic the spanwise lift is: 1.0 is "
                     "elliptic and nothing real is above it.\n\n"
                     "It enters the induced drag as CL^2/(pi AR e), so it "
                     "is what the climb and turn curves cost. The wing the "
                     "search produces has its own measured value; this one "
                     "is the estimate the DIAGRAM is drawn with."),
        "twr_available": ("Every other row here is a REQUIREMENT — a limit "
                          "the wing loading has to stay under. This one is "
                          "a capability.\n\n"
                          "Without it the card can only report the highest "
                          "legal W/S. With it the cruise-matching curve "
                          "crosses the limits and the diagram returns an "
                          "actual design point, which is also where a "
                          "searched wing loading stops instead of running "
                          "to the top of its band."),
    }

    #: HOW BIG ONE CLICK OF THE SPINNER IS, per row. Every field on this
    #: card used to step by 0.5, over rows whose defaults are 0.025 (CD0)
    #: and 0.85 (span efficiency) — so one click of the arrow took CD0 to
    #: 0.525, twenty-one times any real aircraft's, and e to 1.35, which is
    #: not physical. Both then move the W/S ceiling that a sized search
    #: REFUSES designs against, silently. A step is not a limit: anything
    #: can still be typed.
    WS_STEPS = {
        "v_stall_ms": 0.5, "cd0": 0.002, "oswald_e": 0.01,
        "climb_rate_ms": 0.5, "turn_load_factor": 0.1,
        "takeoff_distance_m": 10.0, "landing_distance_m": 10.0,
        "v_takeoff_ms": 0.5, "twr_available": 0.02,
    }

    def _render_ws_cap():
        """WHO CAPS THE WING LOADING — the switch, and what it costs.

        A user who has DETERMINED their W/S is not making a mistake when a
        default stall speed disagrees with it, and until this switch existed
        the shell had no way to be told so: the diagram's ceiling travelled to
        every sized run whatever the user had decided, and the card saying so
        could only be read, never answered. This is the answer.

        It is one control in one place because it is one question, and it
        moves the SOLVER and not just the card — `session.mission_ws_ceiling`
        is what both read. A switch that hid the warning and left the refusal
        would be strictly worse than the warning.
        """
        box = boxes.get("ws_cap")
        if box is None:
            return
        box.clear()
        if S["medium"] not in session.WS_DIAGRAM_MEDIA:
            return                      # a car wing has no loading to cap
        src = session.ws_cap_source(S)
        with box:
            with ui.row().classes("w-full items-center gap-1 no-wrap"):
                ui.toggle({session.WS_CAP_MISSION:
                           "This mission's requirements",
                           session.WS_CAP_STATED: "The W/S I typed"},
                          value=src,
                          on_change=lambda e: _set_ws_cap(e.value)) \
                    .props("dense no-caps unelevated toggle-color=primary")
                widgets.help_dot(
                    "The ceiling travels to a SIZED search: designs above "
                    "it are refused before their solver runs, so an "
                    "over-tight answer here comes back with no design at "
                    "all rather than a slower one.\n\n"
                    "\"This mission's requirements\" takes the tightest "
                    "limit from the constraint diagram below. \"The W/S I "
                    "typed\" makes your own number the ceiling and stops "
                    "the diagram binding anything — it still draws, to "
                    "price your number.",
                    title="What the ceiling does")
            widgets.hint("Which wing loading the solver refuses above.")
            if src == session.WS_CAP_STATED:
                mission_cap = session.diagram_ws_ceiling(S)
                widgets.hint(
                    "Your number is the ceiling: it is what the solver "
                    "refuses designs above, and nothing below overrules it."
                    + (f" This mission's own stall/landing limit "
                       f"({mission_cap:.0f} N/m²) is still drawn under "
                       f"“Where does this come from?”, and now binds nothing "
                       f"— a design that exceeds it will not be refused, and "
                       f"will not be flyable at the speeds stated there."
                       if mission_cap else ""))
            else:
                cap = session.mission_ws_ceiling(S)
                widgets.hint(
                    (f"This mission caps W/S at {cap:.0f} N/m², and that cap "
                     f"reaches the solver: a sized search refuses every "
                     f"design above it. Switch to the other answer if you "
                     f"own this number and the requirements below are not "
                     f"yours."
                     if cap else
                     "Nothing you have stated below bounds W/S yet, so no "
                     "ceiling reaches the solver."))

    def _set_ws_cap(value):
        if not session.set_ws_cap_source(S, value):
            return
        cap = session.mission_ws_ceiling(S)
        ctx.log(
            f"wing loading capped by "
            + ("the W/S you stated" if value == session.WS_CAP_STATED
               else "this mission's own requirements")
            + (f" — {cap:.0f} N/m² reaches the solver" if cap else
               " — no ceiling reaches the solver"), "info")
        _render_ws_cap()
        _render_ws_diagram()
        touch()

    def _render_ws_conflict():
        """The loading you STATED against the one this mission ALLOWS.

        Both are answers to one question and both are already in this form;
        until now only one of them travelled to the solver, as a refusal.
        A user who had decided their wing loading, raised the weight, and left
        the stall requirement alone got three runs that returned nothing in
        12–21 s and no sentence anywhere saying why. Neither number is
        overruled here — which one is wrong is theirs to say — but they are
        put beside each other with both ways out.
        """
        box = boxes.get("ws_conflict")
        if box is None:
            return
        box.clear()
        c = session.ws_over_ceiling(S)
        if c is None:
            return
        with box:
            widgets.hint(
                f"You have stated W/S = {c['stated']:.0f} N/m², and this "
                f"mission's own requirements allow at most {c['cap']:.0f} "
                f"N/m²"
                + (f" ({c['binding']})" if c["binding"] else "")
                + f" — {c['over']:.3g}x. That ceiling travels to the solver: "
                f"a sized search REFUSES every design above it before its "
                f"solver runs, which is a run that comes back with no design "
                f"at all rather than a slower one.", "bad")
            widgets.hint(
                "Two ways out, and they are different decisions: take the "
                "ceiling as your loading (the wing gets bigger), or change "
                "the requirement that sets it — the stall/approach speed, the "
                "landing field, the fly-up — under “Where does this come "
                "from?” below. A stall speed left at its default while the "
                "weight went up is the usual cause.")
            ui.button(f"Use {c['cap']:.0f} N/m²", icon="check",
                      on_click=_adopt_ws_cap).props("outline dense no-caps")

    def _adopt_ws_cap():
        """Take the CEILING as the loading.

        Not :func:`_adopt_ws`: that adopts the diagram's DESIGN POINT, which
        is None on a diagram that does not close — and a diagram that does not
        close is exactly the state this card fires in, so the button would
        have shown a number and then refused to write it.
        """
        c = session.ws_over_ceiling(S)
        if c is None or not session.set_wing_loading(S, c["cap"]):
            ui.notify("nothing to adopt: this mission states no ceiling",
                      type="warning")
            return
        ctx.log(f"wing loading set to {c['cap']:.0f} N/m² — the ceiling "
                f"{c['binding'] or 'this mission'} puts on it; reference "
                f"area S = {float(S['mission']['s_ref_m2']):.4g} m²", "info")
        touch()

    def _render_ws_diagram():
        """Where the wing loading COMES FROM.

        W/S is not a preference: it is what the mission allows, and the
        standard way of answering it is the constraint diagram
        (aerobo.constraint_diagram) — stall and field length and cruise
        matching in air, fly-up and cavitation in water. The card states
        those requirements, shows the band they leave and the aerodynamic
        optimum beside it, and writes the answer into the field above in one
        click. Typing W/S directly stays exactly as it was: this is an
        answer to the same question, not a second question.

        THE CARD IS TWO PARTS, and the split is what makes it live. The
        requirement FIELDS are built here and only here — rebuilding a field
        from inside its own on_change swallows the rest of the number being
        typed (the focus trap) — while everything the diagram DERIVES from
        them (:func:`_render_ws_point`, :func:`_render_ws_derived`) sits in
        containers of its own that are cleared and redrawn on every edit.
        Before that split the figure was drawn once, at build time: a user
        who raised the weight, or typed a stall speed, went on reading a
        diagram of the mission they no longer had.
        """
        box = boxes.get("ws_diagram")
        # the two derived containers belong to THIS build of the card; a
        # medium with no diagram builds neither, and a stale handle would be
        # cleared into a pane that is no longer on screen
        boxes.pop("ws_point", None)
        boxes.pop("ws_derived", None)
        if box is None:
            return
        box.clear()
        if S["medium"] not in session.WS_DIAGRAM_MEDIA:
            with box:
                widgets.hint("A car's rear wing carries no weight, so it has "
                             "no wing loading to derive. Its equivalent — "
                             "how much downforce, at what drag, on how big a "
                             "wing — is asked on the track card in stage 3, "
                             "where the size and the objective live.")
            return
        with box, ui.expansion("Constraint diagram — what caps W/S",
                               icon="rule").classes("w-full").props("dense"):
            widgets.hint(
                "The constraint diagram: every requirement below is a limit "
                "on the wing loading (or on the thrust it needs). The "
                "tightest one decides the wing.")
            if session.ws_cap_source(S) == session.WS_CAP_STATED:
                # the card still DRAWS — it is how a user checks what their
                # own number costs — but it must say it is not deciding
                # anything, or it reads as the limit the solver applies
                widgets.hint(
                    "You have said the W/S above is the ceiling, so nothing "
                    "on this card reaches the solver: it is here to price "
                    "your number, not to overrule it.", "warn")
            boxes["ws_point"] = ui.column().classes("w-full gap-0")
            _render_ws_point()
            clmax = session.section_cl_max(S)
            for key, value in session.ws_inputs(S).items():
                label, unit = WS_LABELS.get(key, (key, ""))
                if key == "cl_max" and clmax is not None:
                    # the SECTION measures this; asking for it as well would
                    # be the tool disagreeing with its own stage 2
                    widgets.kv("CL_max (wing)",
                               f"{clmax[0]:.3g} — {clmax[1]}")
                    continue
                widgets.number_field(
                    label, widgets.shown(value) if value is not None else None,
                    (lambda e, k=key: _set_ws(k, e.value)), unit=unit,
                    step=WS_STEPS.get(key, 0.5), width="w-28",
                    note=WS_NOTES.get(key, "clear to drop it"),
                    help=WS_HELP.get(key, ""),
                    help_title=label)
            boxes["ws_derived"] = ui.column().classes("w-full gap-1")
            _render_ws_derived()

    def _render_ws_point():
        """The operating point the diagram is DRAWN AT.

        Speed, density, depth and the aspect ratio are not asked for on this
        card — they are the mission's and the wing's, stated once each — so
        the sentence that quotes them has to move when they do, or the card
        claims to be drawn at a point the session left behind.
        """
        box = boxes.get("ws_point")
        if box is None:
            return
        box.clear()
        point = session.design_point(S)
        with box:
            widgets.hint(
                f"It is drawn at THIS mission's operating point — "
                f"{float(point['v_ms']):.4g} m/s, ρ = "
                f"{float(point['rho']):.4g} kg/m³"
                + (f", {float(point['depth_m']):.3g} m down"
                   if point.get("depth_m") else "")
                + f" — and at the aspect ratio the session already owns, "
                f"{session.nominal_aspect_ratio(S):.4g}. None of those is asked "
                f"for again below.")

    def _render_ws_derived():
        """What the diagram ANSWERS — redrawn on every edit that moves it.

        The band, the notes, the figure and the adopt button. Nothing here
        is typed into, so this container can be cleared as often as the
        state changes: it is rebuilt when a requirement above is typed
        (:func:`_set_ws`) and when the mission itself moves (``touch``),
        which are exactly the two ways the picture can go stale.
        """
        box = boxes.get("ws_derived")
        if box is None:
            return
        box.clear()
        d = session.ws_diagram(S)
        with box:
            v_s = session.stall_speed_at(S)
            if v_s is not None:
                widgets.hint(
                    f"You do not have to guess the stall speed: at the wing "
                    f"loading this mission already states "
                    f"({session.wing_loading(S):.0f} N/m²) this section "
                    f"stalls at {v_s:.3g} m/s. The field above is the speed "
                    f"you REQUIRE it not to exceed — that requirement is "
                    f"what caps W/S, and the two move together.")
            if d is None:
                widgets.hint("These numbers do not describe a mission yet — "
                             "a cruise speed above the stall speed, and a "
                             "positive CL_max.", "warn")
                return
            r = d.recommend()
            ws_rec = r["wing_loading_pa"]
            if ws_rec is None:
                # an INFEASIBLE diagram: no loading is both allowed and
                # flyable. There is nothing to adopt, and offering the ws_max
                # line here would hand back a design point for an aircraft
                # that cannot be built — the notes below say which way out.
                widgets.hint("No wing loading is both allowed and flyable at "
                             "the thrust you stated.", "bad")
                for note in r["notes"]:
                    widgets.hint(note)
                figstyle.show(v1.fig_constraint_diagram(d), "ws_diagram", 320)
                return
            matched = r.get("wing_loading_matched_pa") is not None
            widgets.hint(
                (f"Design point: W/S = {ws_rec:.0f} N/m² "
                 if matched else
                 f"Allowed: W/S ≤ {ws_rec:.0f} N/m² ")
                + f"({ws_rec / 9.80665:.1f} kg/m²) — set by "
                f"{r['binding_constraint']}."
                + (f" It needs T/W ≥ {r['twr_required']:.3f}."
                   if r.get("twr_required") else ""))
            if not matched:
                # say what is MISSING, not just what is known: this is the
                # difference between a limit and a design point, and it is
                # the reason a searched W/S has nothing to stop it
                widgets.hint(
                    "That is a LIMIT, not a design point — it says where the "
                    "wing stops being legal, not where it stops being "
                    "flyable. State the T/W you have above and the diagram "
                    "closes on a matching point, which is also where a "
                    "SEARCHED wing loading will then stop.", "warn")
            for note in r["notes"]:
                widgets.hint(note)
            figstyle.show(v1.fig_constraint_diagram(d), "ws_diagram", 320)
            ui.button(f"Use {ws_rec:.0f} N/m²", icon="check",
                      on_click=_adopt_ws) \
                .props("outline dense no-caps")

    def _set_ws(key: str, value):
        """A typed diagram input, and everything it moves.

        The card is NOT rebuilt from inside its own field — that would
        swallow the rest of the number being typed — but the picture the
        number changes is: the figure, the band, the notes and the adopt
        button live in a container of their own, and so do the two lines
        above the expansion that quote the ceiling this requirement sets.
        """
        if not session.set_ws_input(S, key, value):
            return
        _render_ws_derived()
        # the ceiling is what this requirement moves, and it is read out
        # ABOVE the expansion — a stall speed typed here changes which
        # number the switch and the conflict card are arguing about
        _render_ws_cap()
        _render_ws_conflict()

    def _adopt_ws():
        """Take the DIAGRAM's recommendation as the loading.

        This writes a size, so it ends in :func:`touch` exactly as
        :func:`_adopt_ws_cap` does. It used to hand-roll its own render list
        instead, and the one thing that list left out was the only thing that
        reaches the solver: ``session.sync_wing_from_mission`` publishes the
        stored area into ``S["wing"]["flags"]`` as ``b_m``/``S_m2``, and
        without it the run flew the family's PUBLISHED 10 m / 10 m² while the
        mission card, stage 3's design box and this very log line all quoted
        the adopted 8.6806 m².
        """
        ws = session.adopt_ws_recommendation(S)
        if ws is None:
            ui.notify("nothing to adopt: the diagram has no band",
                      type="warning")
            return
        ctx.log(f"wing loading set to {ws:.0f} N/m² from the constraint "
                f"diagram — reference area S = "
                f"{float(S['mission']['s_ref_m2']):.4g} m²", "info")
        touch()
        return ws

    def _render_planform_note():
        """Neither taper nor aspect ratio is asked for here, so the stage
        says who does own them and what they are worth right now."""
        box = boxes.get("taper")
        if box is None:
            return
        box.clear()
        lam = session.taper_box(S)
        band = session.taper_re_band(S)
        ar = session.section_aspect_ratio(S)
        size = session.flown_size(S)
        span = session.span_box(S)
        sketching = session.size_statement(S) == session.SIZE_AS_SPAN_AR
        with box:
            widgets.hairline()
            # A SHORT LINE AND THE PARAGRAPH BEHIND A MARK. Both of these
            # answer "why can I not type an aspect ratio here?", which is a
            # question the reader has only if they have it — so the answer
            # goes where they can reach it, not across three lines above the
            # field they were reading.
            widgets.hint_help(
                ("The aspect ratio above is stage 2's ESTIMATE (AR "
                 f"{ar:.3g})." if sketching else
                 "Aspect ratio is not a mission input — it is b²/S."),
                # in the SKETCH face an aspect ratio is on this very card, so
                # the flat "not a mission input" would be read as a denial of
                # a field the user is looking at. The claim that matters is
                # unchanged in both: the FLOWN aspect ratio is b²/S and
                # nobody types it — what stage 2 carries, and what the sketch
                # borrows, is an estimate of a chord.
                (f"The aspect ratio above is stage 2's ESTIMATE (AR "
                 f"{ar:.3g}) — what the section's chord is derived from, and "
                 f"here what turns a width into an area. The aspect ratio "
                 f"the run FLIES is b²/S and nobody states it"
                 if sketching else
                 f"ASPECT RATIO is not a mission input — and it is nobody "
                 f"else's either: it is b²/S. The section is designed for a "
                 f"chord, so stage 2 carries an estimate (currently AR "
                 f"{ar:.3g})")
                + f"; the SPAN is the constraint, and it is stated on "
                  f"stage 3"
                + (f" — searched over {span[0]:.3g}–{span[1]:.3g} m, the b_m "
                   f"row of its design box." if span
                   else f" — its size card, currently b = {size[0]:.3f} m at "
                        f"this area ("
                        + ("chosen" if session.chosen_span(S) is not None
                           else "the span that estimate implies, nobody "
                                "having chosen one")
                        + ")." if size
                   else " (this family carries its own planform)."),
                title="Where the aspect ratio comes from")
            if not lam:
                widgets.hint("Taper: this family flies a fixed planform.")
                return
            short = (f"Taper λ ∈ [{lam[0]:g}, {lam[1]:g}] — the solver "
                     f"chooses it.")
            long = ("The optimiser picks the taper, not you: it is a design "
                    "variable of the family, and stage 3's design box is "
                    "where its band can be narrowed.")
            if band:
                long += (f"\n\nRe at the MAC is quoted at the mean chord "
                         f"S/b; across that box it spans {band[0]:.3g} – "
                         f"{band[1]:.3g}, which is the range the section is "
                         f"actually screened over.")
            widgets.hint_help(short, long, title="Who chooses the taper")

    def _through_the_builder(key: str, value):
        """Apply a stage-1 builder choice the way stage 3 applies its own.

        The medium and the lifting system are the two builder keys this
        stage owns, and they are the two that move the FAMILY hardest: air
        → track loses the designed-section solver, single → tandem loses
        the thickness sweep. ``nice_app.normalise_choices`` is what drops a
        speciality the newly derived family cannot solve — session 24c
        wired it into every V1/V2 builder change for exactly this reason,
        and it is what keeps state from holding a value its own menu no
        longer offers. Writing the choice here and calling ``apply_choices``
        directly skipped it, so a raked tip device asked for in air survived
        into the hydrofoil: the tip-device select fell back to "none" while
        ``winglet_type='raked'`` stayed in the state and still went out in
        the run's flags, against a spec that declares no such flag.

        Routed through the wing stage's own ``set_choice`` — as
        :func:`set_tail` already is — so ONE function knows the family
        rules (mutual exclusion, normalisation, re-derivation) and stage 1
        and stage 3 can never apply the same change differently. The
        fallback is for a headless or partial shell where stage 3 was not
        built; it does the same two steps in the same order.
        """
        if "set_choice" in ctx.actions:
            ctx.act("set_choice", key, value)
            return
        ch = S["wing"]["choices"]
        ch[key] = value
        dropped = v1.normalise_choices(ch, keep=key)
        if dropped:
            names = ", ".join(v1._SPECIAL_LABEL.get(k, k) for k in dropped)
            ctx.log(f"reset {names}: no solver for it in this configuration",
                    "warn")
        for note in session.apply_choices(S):
            ctx.log(note, "warn")

    def _drop_a_system_this_family_cannot_fly(**pending) -> str | None:
        """Re-derive the LIFTING SYSTEM, and say so if it had to go.

        ``system`` is not one of ``nice_app.NORMALISED_KEYS`` — it is not a
        speciality, it is how many surfaces carry the load — so
        ``normalise_choices`` never sees it, and "tandem" used to survive a
        switch to water or to the track, where ``derive_problem`` quietly
        ignores it. Nothing downstream then agreed with anything else: the
        disabled second-surface switch gave the car's reason in its tooltip
        and the tandem's reason in the hint underneath, and
        ``session.second_surface_name`` reads ``system`` before ``tail``, so
        the hydrofoil's ELEVATOR was called the "rear wing" in stage 2's
        surface toggle, its design-point heading and its adopt log, while
        stage 3's configuration list read "lifting system: tandem pair"
        beside "medium: Hydrofoil (water)".

        Availability is asked of the REGISTRY, never decided here:
        ``option_available`` applies the choice to the configuration being
        moved to, derives the problem and reads the problem's own inverse
        choices back. Called BEFORE the change is applied, so the family is
        derived once, from a state that is already honest.
        """
        ch = S["wing"]["choices"]
        held = ch.get("system", v1.BUILDER_DEFAULTS["system"])
        if held == v1.BUILDER_DEFAULTS["system"]:
            return None
        if v1.option_available(dict(ch, **pending), "system", held):
            return None
        ch["system"] = v1.BUILDER_DEFAULTS["system"]
        return (f"no {held} solver in this configuration — back to one "
                f"lifting surface")

    def set_medium(value: str):
        if value == S["medium"]:
            return
        # ONE owner: the choices dict is the state, and apply_choices does
        # the rest (re-open the mission on the new family's design point,
        # drop a section screened for the old one)
        left = _drop_a_system_this_family_cannot_fly(medium=value)
        _through_the_builder("medium", value)
        ctx.log(f"medium set to {session.MEDIA[value][0]} — "
                f"solver family: {S['wing']['problem']}", "info")
        if left:
            ctx.log(left, "warn")
        _render_operating()
        ctx.render("mission", "point")
        ctx.render_when_shown("airfoil")
        ctx.render_when_shown("airfoil_aft")
        ctx.render_when_shown("wing")
        ctx.refresh()

    def set_system(value: str):
        if value == S["wing"]["choices"]["system"]:
            return
        _through_the_builder("system", value)
        ctx.log(f"lifting system: {value} — solver family: "
                f"{S['wing']['problem']}", "info")
        _render_operating()
        ctx.render("mission", "point")
        ctx.render_when_shown("airfoil")
        ctx.render_when_shown("airfoil_aft")
        ctx.render_when_shown("wing")
        ctx.refresh()

    def set_tail(value: bool):
        """Does this vehicle HAVE a second surface?

        A mission question, not a wing one: whether the aircraft carries a
        tail (or the craft an elevator) decides how many surfaces there are
        to design, which section stage 2 has to choose and what stage 3 is
        even configuring. Its GEOMETRY — arm, height, planform, tip device —
        stays in stage 3, where the rest of the design box lives.

        Routed through the wing stage's own ``set_choice`` so the family
        rules (mutual exclusion, normalisation, re-derivation) are applied
        by the one function that owns them.
        """
        ch = S["wing"]["choices"]
        if bool(ch.get("tail")) == bool(value):
            return
        # A surface that has just appeared opens DESIGNED where a solver can
        # design it (v1.tail_design_start): taper, aspect ratio and washout
        # are what a stabiliser is, and the published rectangle at AR 4 is a
        # fitting. Written BEFORE the switch is applied so the family is
        # derived once, from the configuration that will actually be flown.
        if value:
            ch["tail_design"] = v1.tail_design_start(ch)
        if "set_choice" in ctx.actions:
            ctx.act("set_choice", "tail", bool(value))
        else:       # the wing stage is not built (a headless/partial shell)
            ch["tail"] = bool(value)
            v1.normalise_choices(ch, keep="tail")
            session.apply_choices(S)
        aft = session.second_surface_name(S) or "second surface"
        stage_25 = session.aft_surface(S) is not None
        ctx.log((f"{aft} added — solver family: {S['wing']['problem']}"
                 + ("; stage 2.5 now chooses its own aerofoil"
                    if stage_25 else "")
                 if value else
                 f"second surface removed — solver family: "
                 f"{S['wing']['problem']}"), "info")
        for note in session.refresh_recommended_weights(S):
            ctx.log(note, "info")
        _render_operating()
        ctx.render("mission", "point")
        ctx.render_when_shown("airfoil")
        ctx.render_when_shown("airfoil_aft")
        ctx.render_when_shown("wing")
        ctx.refresh()

    def _second_surface_control():
        """The switch, and the honest reason where there is nothing to add."""
        ch = S["wing"]["choices"]
        water = S["medium"] == "water"
        can = v1.option_available(ch, "tail", True) or bool(ch.get("tail"))
        sw = ui.switch("add an elevator (rear stabiliser, below the foil)"
                       if water else "add a tail (trims and stabilises)",
                       value=bool(ch.get("tail")),
                       on_change=lambda e: set_tail(bool(e.value))) \
            .props("dense")
        if not can:
            # ONE branch decides BOTH texts. They used to be written
            # separately and tested in opposite orders — the tooltip asked
            # the medium first, the hint asked the system first — so a
            # configuration answering yes to both (a tandem that had leaked
            # into the track medium) put two different reasons on the same
            # disabled switch, one of them about a family that was not on
            # screen. That leak is re-derived away now, but a refusal that
            # can contradict itself is the bug, not the leak that exposed it.
            track = S["medium"] == "track"
            sw.disable()
            sw.tooltip("a car's rear wing has nothing behind it" if track else
                       "the tandem pair's rear wing IS the second surface")
            widgets.hint(
                "A car's rear wing is the last surface on the car: there is "
                "nothing behind it to trim with."
                if track else
                "The tandem pair already carries two lifting surfaces — its "
                "rear wing is the second one, and stage 2 chooses a section "
                "for it.")
            return
        if ch.get("tail"):
            aft = session.aft_surface(S)
            widgets.hint_help(
                "Trimmed in lift AND pitch — one solve, not two.",
                ("The stabiliser sits below and aft of the main foil in the "
                 "same imaged solve, and it carries its own cavitation "
                 "margin at its own (deeper) submergence."
                 if water else
                 "There is a static-margin constraint, and the tail is in "
                 "the same solve as the wing rather than bolted on after "
                 "it.")
                + "\n\n"
                + (f"Stage 2.5 — “{session.stage_label(S, 'airfoil_aft')}” "
                   f"in the tree — chooses this surface's own aerofoil, at "
                   f"its own chord and its own design lift; stage 3 sizes "
                   f"and shapes it (arm, height, planform, tip device)."
                   if aft else
                   f"Stage 3 sizes and shapes this surface (arm, height, "
                   f"planform, tip device). This solver family selects its "
                   f"section by thickness, so there is no separate aerofoil "
                   f"to choose for it: it flies the wing's. It is currently "
                   f"the "
                   f"{session.second_surface_name(S) or 'second surface'}."),
                title="What adding a tail changes")
        else:
            widgets.hint(
                "Off: one surface carries the load and nothing trims it — the "
                "run is trimmed in lift alone and says nothing about "
                "stability.")

    def set_tail_type(value: str):
        """WHICH EMPENNAGE this aircraft has.

        A MISSION question, and for the same reason the two switches around
        it are: the layout decides how many surfaces there are and which
        stages exist, not merely how one of them is shaped. A V-tail has no
        separate vertical surface at all — its two canted panels carry the
        yaw, so stage 2.7 does not exist for it — and a T-tail CONNECTS the
        two, standing the tailplane on the fin's tip so the fin's span IS the
        tailplane's height (``fin.size_fin``). Asking that on stage 3, under
        the design box, put a question about the aircraft's configuration two
        stages after the stages it decides.

        Routed through the wing stage's ``set_choice``, the one function that
        owns the family rules, exactly as :func:`set_tail` is.
        """
        ch = S["wing"]["choices"]
        value = str(value or "conventional")
        if str(ch.get("tail_type", "conventional")) == value:
            return
        if "set_choice" in ctx.actions:
            ctx.act("set_choice", "tail_type", value)
        else:       # the wing stage is not built (a headless/partial shell)
            ch["tail_type"] = value
            v1.normalise_choices(ch, keep="tail_type")
            session.apply_choices(S)
        ctx.log(
            f"empennage: {v1.TAIL_TYPE_LABELS.get(value, value)}"
            + ("; its two canted panels ARE the vertical surface, so there "
               "is no separate fin and no stage 2.7"
               if value == "v_tail" else
               "; the tailplane stands on the fin's tip, so the fin's span "
               "is the tailplane's height and the two are sized together"
               if value == "t_tail" else
               "; a separate fin and tailplane, each with its own section"),
            "info")
        _render_operating()
        ctx.render("mission", "point")
        ctx.render_when_shown("airfoil_aft")
        ctx.render_when_shown("airfoil_fin")
        ctx.render_when_shown("wing")
        ctx.refresh()

    def _empennage_control():
        """Which empennage — asked beside the two switches it interacts with.

        Only in air: under water there is no layout menu at all (no fin, no
        fuselage, all-moving by construction — see
        ``nice_app._derive_hydro_tail``), and only where there IS a second
        surface, because an empennage is what that surface is arranged as.
        """
        ch = S["wing"]["choices"]
        if S["medium"] != "air":
            return
        # THE LABEL IS BESIDE THE FIELD, NOT INSIDE IT. This was the one
        # ``label=`` in the whole shell, and Quasar's floating label cannot
        # work here: theme.py pins every field at 26 px
        # (``.q-field__control-container{padding-top:0!important;height:26px}``)
        # so there is no room above the value for the label to float into.
        # It floated 4 px and landed ON the value — "empennage" printed
        # through "conventional (aft, on the fuselage)", which reads as a
        # struck-out, unselectable field. ``widgets.select_field`` is the
        # house answer and says so in its own docstring; use it.
        kind = str(ch.get("tail_type", "conventional"))
        sel = widgets.select_field(
            "empennage", v1.TAIL_TYPE_LABELS, kind,
            lambda e: set_tail_type(str(e.value)),
            width="grow min-w-0",
            help="\n\n".join(
                f"{v1.TAIL_TYPE_LABELS[k]} — {TAIL_TYPE_MISSION_NOTES[k]}"
                for k in v1.TAIL_TYPE_LABELS
                if k in TAIL_TYPE_MISSION_NOTES),
            help_title="What each layout changes")
        if not ch.get("tail"):
            sel.disable()
            sel.tooltip("add a tail first — an empennage is how the second "
                        "surface is arranged")
            widgets.hint("With no second surface there is no empennage to "
                         "arrange. The fin below is still a question: a "
                         "tailless aircraft can carry one.")
            return
        widgets.hint(TAIL_TYPE_MISSION_SHORT.get(kind, ""))

    def set_fin(value: bool):
        """Does this vehicle HAVE a vertical tail?

        A mission question for the same reason the second surface is one:
        whether there IS a fin decides how many surfaces there are to design
        and which stages exist. Its geometry — volume coefficient, aspect
        ratio, station, thickness — stays where the rest of the design box
        is, on stage 3.

        It was not askable at all before V5, and the reason recorded in
        ``session.V3_PINNED_CHOICES`` was that the fin "is not a surface this
        package models". That is no longer true in any of its clauses: the
        lattice carries its panels, the drag book charges it, the CAD exports
        it and the flight model's whole yaw stiffness comes from it. So the
        question is real, and it is asked here.
        """
        ch = S["wing"]["choices"]
        if bool(ch.get("fin", session.FIN_DEFAULT)) == bool(value):
            return
        ch["fin"] = bool(value)
        ctx.log(("vertical tail added — stage 2.7 now chooses its section, "
                 "the run charges its drag and weighs it, and the flight "
                 "stage flies it"
                 if value else
                 "vertical tail removed: the run no longer charges its drag "
                 "or weighs it, nothing on this aircraft makes yaw "
                 "stiffness, and Cn_beta is exactly zero"), "info")
        _render_operating()
        ctx.render("mission", "point")
        # ...AND THE TWO STAGES THIS DECIDES. The fin is a surface like the
        # others now: stage 2.7 chooses its section and stage 3 draws it,
        # so both have to be repainted the way ``set_tail_type`` repaints
        # them. Neither is a DERIVED view, so ``ctx.refresh`` cannot reach
        # them — and the Wing type card went on drawing a sized, quoted fin,
        # byte-identical whether the switch was on or off.
        ctx.render_when_shown("airfoil_fin")
        ctx.render_when_shown("wing")
        ctx.refresh()

    def _fin_control():
        """The switch, and the honest reason where the answer is not the
        user's."""
        ch = S["wing"]["choices"]
        water = S["medium"] == "water"
        if S["medium"] not in ("air", "water"):
            return          # a car's rear wing carries no vertical surface
        v_tail = str(ch.get("tail_type", "conventional")) == "v_tail"
        on = bool(ch.get("fin", session.FIN_DEFAULT))
        # A SWITCH THAT REACHES NOTHING MUST NOT BE OFFERED. The answer
        # travels as a flag (``api.FIN_PRESENCE_KEY``) and a flag travels
        # only to a family that declares it, so on a family with no fin in
        # its solver at all — a single wing, a car's rear wing — this switch
        # was live, defaulted ON, and changed nothing: no drag, no weight,
        # no stage 2.7, no yaw stiffness. Asked of the SPEC, which is where
        # "does this family have a fin" is already asked
        # (:func:`session.fin_surface`).
        from aerobo import api as _api
        _sp = _api.PROBLEM_SPECS.get(S["wing"].get("problem"))
        family_has_one = _sp is None or any(
            k in _sp.flags for k in (*_api.FIN_SHAPE_KEYS,
                                     _api.FIN_PRESENCE_KEY))
        sw = ui.switch("carry a strut (the mast that holds the foil)" if water
                       else "add a vertical stabiliser (fin and rudder)",
                       value=on and not v_tail and family_has_one,
                       on_change=lambda e: set_fin(bool(e.value))) \
            .props("dense")
        if not family_has_one:
            sw.disable()
            sw.tooltip("this family carries no vertical surface")
            widgets.hint(
                "This configuration has no fin to design: its solver builds "
                "one lifting surface and charges, weighs and flies exactly "
                "that. Nothing here makes yaw stiffness, so Cn_beta is zero "
                "— add a tail or a tandem rear wing above and the question "
                "becomes a real one.")
            return
        if v_tail:
            sw.disable()
            sw.tooltip("a V-tail IS the vertical surface")
            widgets.hint(
                "This layout answers the question by construction: a V-tail's "
                "panels are canted, so they make yaw stiffness and pitch "
                "authority at once — carrying a separate fin as well is the "
                "one thing the layout exists to avoid. Choose a conventional "
                "or T-tail above to design a fin of your own.")
            return
        if on and water:
            widgets.hint_help(
                "The mast — the run already charges its drag.",
                f"Its wetted area is two sides of the submergence depth by "
                f"the mast chord, and that drag is what stops the search "
                f"from flying the foil arbitrarily deep.\n\n"
                f"Stage 2.7 — “{session.stage_label(S, 'airfoil_fin')}” in "
                f"the tree — chooses its section (symmetric: a strut at "
                f"zero leeway must make no side force).",
                title="What the mast is")
        elif on:
            widgets.hint_help(
                "Its section is stage 2.7; stage 3 sizes it.",
                f"“{session.stage_label(S, 'airfoil_fin')}” in the tree "
                f"chooses the fin's section, and that is a smaller question "
                f"than a wing's: a fin must make no side force at zero "
                f"sideslip, so the shape is symmetric and only its "
                f"thickness is a choice.",
                title="Where the fin is designed")
        elif water:
            widgets.hint(
                "Off: no mast. The run stops charging its wetted area — "
                "worth 14.5 % of L/D on the shipped foil, and the term that "
                "creates the interior depth optimum, so the search will now "
                "want to fly as deep as its box allows. Nothing carries the "
                "foil and nothing makes yaw stiffness. Choose this to see "
                "what the strut costs, not to describe a craft.", "warn")
        else:
            widgets.hint(
                "Off: no vertical surface is designed, drawn, charged or "
                "weighed. The run pays no fin drag (0.09 drag counts here, "
                "about 5.8 % of L/D) and the empennage weighs the tailplane "
                "alone, which is where most of that number went. The fin is "
                "the only thing on this aircraft DESIGNED to make yaw "
                "stiffness, so Cn_beta is zero unless a canted tip device "
                "is fitted — that carries side force too, at whatever arm "
                "its station gives. The flight stage flies exactly this "
                "aeroplane and says so.", "warn")

    def set_water(value: str):
        if value == S.get("water", "sea"):
            return
        S["water"] = value
        # the water sets rho/mu, so the design point (and every screened
        # Reynolds number) moves with it
        session.sync_wing_from_mission(S)
        ctx.log(f"water: {api.water_kinds()[value]} — the design point and "
                f"the cavitation threshold both follow it", "info")
        _render_operating()
        ctx.render("mission", "point")
        ctx.render_when_shown("wing")
        ctx.refresh()

    def accept():
        # ONE action ("accept_mission"), because the toolbar's Run button,
        # the tree and the tests all call it and stage 1 has one verb
        # whatever it is asking about
        if session.airfoil_only(S):
            accept_flow()
            return
        dp = session.design_point(S)
        if "error" in dp:
            ui.notify(dp["error"], type="negative")
            ctx.log(dp["error"], "error")
            return
        S["mission"]["accepted"] = True
        session.sync_wing_from_mission(S)
        ctx.log(f"mission accepted — CL {dp['cl_design']:.4f}, "
                f"Re at MAC {dp['re_mac']:.4g}, MAC {dp['mac']:.4f} m", "ok")
        ui.notify("Mission accepted", type="positive")
        ctx.refresh()
        ctx.select("airfoil", "screen")

    def reset_defaults():
        if session.airfoil_only(S):
            reset_flow_defaults()
            return
        name = S["wing"]["problem"]
        # which UNIT the size is typed in is not a published default — it is
        # the user's, and a reset of the operating point must not move their
        # cursor into a field they did not choose
        stated_as = session.size_statement(S)
        S["mission"].update(session.mission_defaults(name, S["medium"]))
        # ...and the CAR's own stated number, which mission_defaults knows
        # nothing about: clearing it puts the reference CZ back on
        # session.REFERENCE_CL, and track_point_sync rebuilds the stored
        # weight/area/altitude mirror from there.
        S["mission"].pop("cz_design", None)
        session.set_size_statement(S, stated_as)
        # the planform estimate is part of "the published defaults": leaving
        # a typed aspect ratio behind would reset the mission onto a chord
        # nobody asked for
        S["airfoil"]["ar"] = session.default_aspect_ratio(name)
        session.sync_wing_from_mission(S)
        if S["medium"] == "track":
            # the track states neither a design load nor an aspect ratio, so
            # the aircraft sentence would quote a `w_source` about a weight
            # this card does not show and an estimate nobody owns
            dp = session.design_point(S)
            ctx.log(f"mission reset to {name}'s own point — "
                    f"{float(S['mission']['V']):.4g} m/s at the reference CZ "
                    f"{session.track_design_cz(S):g}, so the section is "
                    f"screened at Re {dp.get('re_mac', float('nan')):.3e}",
                    "info")
        else:
            ctx.log(f"mission reset to the operating point of {name} — "
                    f"{S['mission']['w_source']} — and the aspect ratio to "
                    f"that family's own "
                    f"{session.default_aspect_ratio(name):.4g}", "info")
        _render_operating()
        ctx.render("mission", "point")
        ctx.render_when_shown("airfoil")
        ctx.render_when_shown("wing")
        ctx.refresh()

    # ------------------------------------------------- the mode, and the flow
    #
    # AIRFOIL-ONLY (``session.MODES``). Stage 1 asks a different question in
    # this mode and stages 3 and 4 are not part of the session at all: there
    # is no vehicle, so there is no weight, no reference area and no wing
    # loading — and no way to derive a chord either, which is why the chord
    # is asked outright. What comes out is the same three numbers the section
    # search has always taken (Re, Mach, design Cl), reached without stating
    # an aircraft nobody chose.
    def _mode_control():
        """WHICH question this session is answering. Drawn in both modes, at
        the top of stage 1, because it is the first thing a session decides
        and the only control that can undecide it."""
        mode = session.session_mode(S)
        with widgets.group_box("What this session designs"):
            ui.toggle({k: v[0] for k, v in session.MODES.items()},
                      value=mode,
                      on_change=lambda e: _set_mode(str(e.value))) \
                .props("dense no-caps unelevated toggle-color=primary")
            widgets.hint(session.MODES[mode][1])

    def _set_mode(value: str):
        if not session.set_mode(S, value):
            return
        # the shell may be sitting on a stage this mode does not have — the
        # switch can be driven from anywhere, and Ctx.select refuses a locked
        # stage rather than opening it, so a session left pointing at the
        # wing would show a stage it cannot leave
        if not session.stage_visible(S, S["ui"]["selected"]):
            S["ui"]["selected"] = "mission"
            S["ui"]["tab"]["mission"] = "operating"
        ctx.log(f"{session.MODES[value][0]} — "
                + ("stage 1 states the flow the section works in, and the "
                   "wing stages are not part of this session"
                   if value == "airfoil" else
                   "the mission sizes a surface again, and the wing stages "
                   "are back"), "info")
        _render_operating()
        for stage in ("mission", "airfoil", "airfoil_aft", "wing", "results"):
            ctx.render_when_shown(stage)
        ctx.refresh()

    #: decimals each flow field SHOWS, where the shell's default of four is
    #: wrong for it. Viscosity is the reason this exists: ``round(1.79e-5,
    #: 4)`` is 0.0, so the field would display zero for the number it holds
    #: and ``is_echo`` would read every keystroke as the echo of that zero.
    FLOW_DIGITS = {"mu": 10, "rho": 6, "mach": 5, "chord_m": 5}

    def _flow_digits(key: str) -> int:
        return FLOW_DIGITS.get(key, widgets.FIELD_DIGITS)

    def _flow_touch():
        """A flow edit un-accepts the point (``session.set_flow``) and moves
        every view derived from it.

        The operating view is this handler's OWN — the read-outs in it are
        redrawn by name above, and rebuilding it would take the field being
        typed into with it (the focus trap this stage is shaped around)."""
        _render_flow_derived()
        _render_flow_mach()
        ctx.render("mission", "point")
        ctx.render_when_shown("airfoil")
        ctx.refresh(("mission", "operating"))

    def flow_setter(key: str):
        def _set(e):
            if e.value in (None, ""):
                return
            if widgets.is_echo(e.value, session.flow_state(S)[key],
                               _flow_digits(key)):
                return
            if not session.set_flow(S, key, e.value):
                return
            _flow_touch()
        return _set

    def set_flow_value(key: str, value):
        """The same edit, without the field's echo guard — the registered
        twin, so the shell and the tests drive the real path."""
        if not session.set_flow(S, key, value):
            return
        _flow_touch()

    def set_fluid(value: str):
        """Switch which FACE the flow is stated in (or which water it is).

        Rebuilds the view on purpose: the face decides WHICH fields exist,
        and a toggle is not a field being typed into."""
        if not session.set_fluid(S, value):
            return
        f = session.flow_state(S)
        ctx.log(f"flow stated as {session.FLUID_LABELS[f['fluid']]}"
                + (f" ({api.water_kinds()[f['water']]})"
                   if f["fluid"] == "water" else ""), "info")
        _render_operating()
        ctx.render("mission", "point")
        ctx.render_when_shown("airfoil")
        ctx.refresh()

    def set_flow_mach(on: bool):
        """Fly the stated Mach number, or design incompressible."""
        if not session.set_flow_apply_mach(S, bool(on)):
            return
        ctx.log("the section is designed at M "
                + (f"{session.flow_point(S).get('mach', 0.0):.4f}"
                   if on else "0 (incompressible)")
                + (" — no cached polar is at that Mach, so the library "
                   "screen is a real sweep" if on else ""),
                "warn" if on else "info")
        _render_operating()
        ctx.render("mission", "point")
        ctx.render_when_shown("airfoil")
        ctx.refresh()

    def accept_flow():
        pt = session.flow_point(S)
        if "error" in pt:
            ui.notify(pt["error"], type="negative")
            ctx.log(pt["error"], "error")
            return
        session.set_stage1_accepted(S, True)
        ctx.log(f"flow accepted — Re {pt['re']:.4g} at c = "
                f"{pt['chord_m']:.4f} m, M {session.flown_mach(S):.4f}, "
                f"design Cl {pt['cl_design']:.4f}", "ok")
        ui.notify("Flow conditions accepted", type="positive")
        ctx.refresh()
        ctx.select("airfoil", "screen")

    def reset_flow_defaults():
        f = session.flow_state(S)
        f.clear()
        f.update(session.flow_defaults(S))
        ctx.log("flow reset to the point this session's published mission "
                "implies — "
                f"{f['V']:.4g} m/s, c = {f['chord_m']:.4g} m, "
                f"Cl {f['cl_design']:.4g}", "info")
        _render_operating()
        ctx.render("mission", "point")
        ctx.render_when_shown("airfoil")
        ctx.refresh()

    def _flow_cost_note(pt: dict):
        """What screening at THIS point costs, said where the point is set.

        The library checkpoint is one (Re, Mach) pair; anything else is a
        real XFOIL sweep, and in this mode the user can move both of them
        with one keystroke. Stage 2 says the same thing beside its own
        button — this says it before the button is pressed."""
        lib = session.library_point()
        if lib is None:
            widgets.hint("The screening cache has not been built on this "
                         "machine, so the first screen runs XFOIL over the "
                         "database whatever this point is.", "warn")
            return
        same = (abs(float(pt["re"]) - float(lib["re"]))
                <= 1e-9 * float(lib["re"])
                and float(session.flown_mach(S)) == float(lib.get("mach", 0.0)))
        if same:
            widgets.hint(f"This is the cached library point "
                         f"(Re {lib['re']:.3g}), so screening it is instant.")
            return
        # HOW FAR off, not just "not it": the cache is keyed on the point
        # EXACTLY, so a chord 0.04 % away costs the same sweep as one twice
        # the size — and a user reading "not it" beside 999554 vs 1e+06
        # deserves to be told which of those two facts is doing the work.
        gap = abs(float(pt["re"]) - float(lib["re"])) / float(lib["re"])
        widgets.hint(
            f"The library is cached at Re {lib['re']:.3g}, M "
            f"{float(lib.get('mach', 0.0)):.3g}; this point is "
            f"{gap * 100.0:.2g} % away in Re"
            + (" and at a different Mach"
               if float(session.flown_mach(S))
               != float(lib.get("mach", 0.0)) else "")
            + ". The cache is keyed on the point exactly, so stage 2's "
              "screen sweeps its shortlist for real — tens of seconds to "
              "minutes on the first visit, instant afterwards. The shape "
              "optimisation runs live XFOIL at this point either way.",
            "warn")

    def _render_flow_derived():
        """What the stated flow IS — the derived half of the card.

        Its own container, repainted by name, because the fields above it
        are typed into: this is the read-out side of the focus split every
        card in this stage uses."""
        box = boxes.get("flow_derived")
        if box is None:
            return
        box.clear()
        pt = session.flow_point(S)
        with box:
            if "error" in pt:
                widgets.hint(pt["error"], "bad")
                return
            widgets.kv("density rho", f"{pt['rho']:.6g} kg/m³",
                       tip="ISA at the altitude, the named water, or yours")
            widgets.kv("viscosity mu", f"{pt['mu']:.6g} Pa·s",
                       tip="Sutherland at the ISA temperature, the named "
                           "water, or yours")
            widgets.kv("kinematic nu", f"{pt['nu']:.6g} m²/s", tip="mu / rho")
            widgets.kv("dynamic pressure q", f"{pt['q']:.6g} Pa",
                       tip="½ rho V²")
            widgets.kv("Reynolds number", f"{pt['re']:.6g}",
                       color=theme.ACCENT,
                       tip="rho V c / mu — the section is screened AND "
                           "optimised at this")
            if pt["a_ms"] is not None:
                widgets.kv("speed of sound", f"{pt['a_ms']:.5g} m/s",
                           tip="√(gamma R T) at the ISA temperature, or the "
                               "one your speed and Mach number imply")
            flown = session.flown_mach(S)
            widgets.kv("Mach number",
                       f"{pt['mach']:.4f}"
                       + ("" if flown else "   (not flown — M 0)"),
                       color=(theme.WARN if pt["mach"] > session.MACH_WARN
                              else ""),
                       tip="what the speed and the fluid imply; the switch "
                           "beside the design point decides whether XFOIL "
                           "is given it")
            widgets.kv("lift per span",
                       f"{pt['lift_per_span_n_m']:.5g} N/m",
                       tip="Cl q c — what choosing that lift coefficient "
                           "means in newtons, per metre of span")
            _flow_cost_note(pt)
            if session.stage1_accepted(S):
                widgets.hint("Accepted. Editing any field above un-accepts "
                             "it: a section screened at the old Reynolds "
                             "number is not this flow's section.")

    def _render_flow_mach():
        """Fly the stated Mach number, or design incompressible.

        Drawn only where there IS one: water is incompressible at this tier
        and a custom fluid may state M = 0, and a switch that decides
        nothing is worse than no switch."""
        box = boxes.get("flow_mach")
        if box is None:
            return
        box.clear()
        f = session.flow_state(S)
        pt = session.flow_point(S)
        mach = 0.0 if "error" in pt else float(pt["mach"])
        if not mach > 0.0:
            return
        with box:
            ui.switch(f"fly the Mach number ({mach:.4f})",
                      value=bool(f.get("apply_mach")),
                      on_change=lambda e: set_flow_mach(bool(e.value))) \
                .props("dense")
            widgets.hint(
                "On: XFOIL is given this Mach and applies its "
                "compressibility correction. NO cached polar is at it — the "
                "whole screening checkpoint is M 0 — so the library screen "
                "becomes a real sweep of the database."
                if f.get("apply_mach") else
                "Off: the section is designed incompressible, as every "
                "cached polar and every published study here is. The number "
                "above is what the flow really is.",
                "warn" if (f.get("apply_mach") and mach > session.MACH_WARN)
                else "")

    # ------------------------------------------------- the track's own view
    def _set_track_cz_value(value):
        """Set it from a value (the registered action; the field wraps it)."""
        if value in (None, ""):
            return
        if not session.set_track_design_cz(S, value):
            ui.notify("the reference CZ must be positive", type="negative")
            return
        touch(size_field=False)

    def _set_track_cz(e):
        """The reference CZ — the ONE coefficient the track card states."""
        if e.value in (None, ""):
            return
        if widgets.is_echo(e.value, session.track_design_cz(S)):
            return
        # `touch` (inside) repaints the derived block through _render_derived,
        # which routes to this view's own on the track. The FIELD is not
        # rebuilt, which is the focus trap this stage is built around.
        _set_track_cz_value(e.value)

    def _render_track_view(box):
        """Stage 1 for a CAR REAR WING: the speed, the CZ, and nothing else.

        Everything this card used to ask besides those two was an aircraft
        question the track problem has no answer to — a design weight, an
        altitude, a wing loading, a reference area, and the constraint
        diagram that derives one from the other. ``session.TRACK_POINT_NOTE``
        carries the reasoning; what it means on screen is that a user is no
        longer asked to state four numbers so that the shell can back one
        out of them, then shown a warning under each saying the run does not
        fly it.
        """
        m = S["mission"]
        size = session.track_reference_size(S)
        with box:
            _mode_control()
            with widgets.group_box("Vehicle"):
                ui.toggle({k: v[0] for k, v in session.MEDIA.items()},
                          value=S["medium"],
                          on_change=lambda e: set_medium(e.value)) \
                    .props("dense no-caps unelevated toggle-color=primary")
                widgets.hint(session.MEDIA["track"][1])
                # ONE STATEMENT, not two disabled switches. The aircraft
                # card asks "how many lifting surfaces" and then "is there a
                # tail" — two questions a car answers the same way for the
                # same reason, and answering it twice in greyed-out controls
                # made the track card look like it was refusing things
                # rather than describing a rear wing.
                with ui.row().classes("w-full items-center gap-2 no-wrap"):
                    ui.label("lifting surfaces") \
                        .classes("field-label").style(
                            f"color:{theme.INK_MUTED};min-width:120px")
                    ui.label("one — the rear wing").classes("readout")
                widgets.hint(
                    "A rear wing is the LAST surface on the car: there is "
                    "nothing behind it to trim with, and a tandem pair has "
                    "no car-wing solver. The only second surface a rear "
                    "wing can have is a SLOTTED FLAP — a chordwise problem "
                    "inside its own section — and that is a switch in "
                    "stage 3, not a second wing here.")

            with ui.row().classes("w-full items-start gap-3 no-wrap"):
                with ui.column().classes("gap-3").style("flex:1 1 0;"
                                                        "min-width:0"):
                    with widgets.group_box("Operating point"):
                        widgets.number_field(
                            "speed", widgets.shown(m["V"]), setter("V"),
                            unit="m/s", step=1.0, width="w-32",
                            tip="the one number on this card the car-wing "
                                "solver reads — it sets the dynamic "
                                "pressure, every force in newtons, the "
                                "deflection and the section Reynolds number")
                        widgets.hint(
                            f"Reaches the solver as the car problem's own V "
                            f"flag — the family takes no mission spec at "
                            f"all. "
                            f"Its published point is "
                            f"{v1.car_default_v_ms():.4g} m/s.")
                        widgets.number_field(
                            "reference CZ", widgets.shown(
                                session.track_design_cz(S)),
                            _set_track_cz, step=0.05, width="w-32",
                            tip="the downforce coefficient the SECTION is "
                                "designed at — it is not a target the wing "
                                "is trimmed to, because nothing trims a "
                                "rear wing")
                        widgets.hint(
                            f"Stage 2 screens and optimises its aerofoil at "
                            f"this lift coefficient. The RUN does not aim "
                            f"for it: stage 3 maximises what you tell it to "
                            f"(downforce, CZ/CD, or a lap) and the CZ it "
                            f"reaches is an answer. Opens at "
                            f"{session.REFERENCE_CL:g}, this shell's own "
                            f"reference.")
                    with widgets.group_box("Air"):
                        widgets.hint(
                            "Fixed, and not asked: carwing.py flies its "
                            "module constant RHO_AIR = 1.225 kg/m³ (ISA at "
                            "sea level) whatever a card says, so an altitude "
                            "field here would move the density stage 2 "
                            "designs against while stage 3 flew 1.225 "
                            "regardless. A track is a track.")

                with ui.column().classes("gap-3").style("flex:1 1 0;"
                                                        "min-width:0"):
                    with widgets.group_box("Size — stage 3's, "
                                           "not this card's"):
                        widgets.hint(
                            "A rear wing's SPAN and its reference AREA are "
                            "both design variables: their bands are the "
                            "b_m and S_m2 rows of stage 3's design box, "
                            "because what bounds a rear wing is a "
                            "regulation or the bodywork, not a wing loading. "
                            "There is no weight to divide by one either.")
                        if size:
                            widgets.hint(
                                f"The section is therefore designed at the "
                                f"MIDDLE of that box — b = {size[0]:.3g} m, "
                                f"S = {size[1]:.4g} m², so AR = "
                                f"{size[0] ** 2 / size[1]:.3g} and the mean "
                                f"chord is {size[1] / size[0]:.4g} m. "
                                f"Narrow either row and this follows it.")
                        else:
                            widgets.hint(
                                "This car family carries its own planform, "
                                "so the section is designed at that.",
                                "warn")
                        ui.button("open the design box", icon="tune",
                                  on_click=lambda: ctx.select("wing", "box")) \
                            .props("flat dense no-caps")

            with widgets.group_box("Implied design point"):
                boxes["derived"] = ui.column().classes("w-full gap-2")

            with ui.row().classes("w-full items-center gap-2"):
                ui.button("Accept mission", icon="check", on_click=accept) \
                    .props("unelevated dense no-caps color=primary")
                ui.button("Published defaults", icon="restart_alt",
                          on_click=reset_defaults) \
                    .props("outline dense no-caps")
                ui.space()
                widgets.hint("Accepting unlocks the airfoil stage.")
        _render_track_derived()

    def _render_track_derived():
        """What the two stated numbers imply, in the car's own terms."""
        box = boxes.get("derived")
        if box is None:
            return
        box.clear()
        dp = session.design_point(S)
        size = session.track_reference_size(S)
        with box:
            if "error" in dp:
                widgets.hint(dp["error"], "bad")
                return
            with ui.row().classes("w-full items-start gap-4 flex-wrap"):
                widgets.readout("q", f"{dp['q']:.4g}", "Pa",
                                tip="½ rho V² at RHO_AIR")
                widgets.readout("rho", f"{dp['rho']:.5g}", "kg/m³")
                if dp.get("mach") is not None:
                    widgets.readout(
                        "Mach", f"{dp['mach']:.4f}",
                        color=(theme.WARN if dp["mach"] > session.MACH_WARN
                               else ""),
                        tip="derived from the speed — the car solver is "
                            "incompressible, so this is a check, not an input")
                fz = (float(dp["cl_design"]) * float(dp["q"])
                      * float(dp["s_ref_m2"]))
                widgets.readout(
                    "downforce at CZ", f"{fz:.4g}", "N",
                    tip="CZ q S at the box's mid area — what the reference "
                        "coefficient is worth in newtons, so the drag "
                        "ceiling and downforce floor in stage 3 can be "
                        "stated beside a number")
            widgets.hairline()
            with ui.row().classes("w-full items-start gap-4 flex-wrap"):
                widgets.readout("screen at CZ", f"{dp['cl_design']:.4f}")
                widgets.readout("MAC", f"{dp['mac']:.4f}", "m")
                widgets.readout("Re at MAC", f"{dp['re_mac']:.3e}")
                if size:
                    widgets.readout("AR", f"{size[0] ** 2 / size[1]:.3g}",
                                    "b²/S", tip="off the design box's own "
                                                "b_m and S_m2 rows")
            widgets.hint(
                "The chord comes from the design box, so this is the "
                "Reynolds number of a wing in the MIDDLE of the size you "
                "allowed — not of a wing anyone has chosen. Stage 3 reports "
                "the Reynolds number it actually flew.")
            if dp.get("mach") is not None and dp["mach"] > session.MACH_WARN:
                widgets.hint(
                    f"M = {dp['mach']:.3f}: every section polar here is "
                    f"INCOMPRESSIBLE, so past about {session.MACH_WARN:g} "
                    f"the section data are no longer the right data.", "warn")
            if S["mission"]["accepted"]:
                widgets.hint("Accepted. Editing either field above "
                             "un-accepts it: a section screened at the old "
                             "Reynolds number is not this mission's section.")

    def _render_flow_view(box):
        """Stage 1, in airfoil-only mode: the flow, and nothing else."""
        f = session.flow_state(S)
        fluid = str(f["fluid"])
        with box:
            _mode_control()
            with ui.row().classes("w-full items-start gap-3 no-wrap"):
                with ui.column().classes("gap-3").style("flex:1 1 0;"
                                                        "min-width:0"):
                    with widgets.group_box("The flow"):
                        ui.toggle(dict(session.FLUID_LABELS), value=fluid,
                                  on_change=lambda e: set_fluid(str(e.value))) \
                            .props("dense no-caps unelevated "
                                   "toggle-color=primary")
                        widgets.hint(FLUID_NOTES[fluid])
                        if fluid == "water":
                            ui.toggle(api.water_kinds(), value=str(f["water"]),
                                      on_change=lambda e:
                                      set_fluid(str(e.value))) \
                                .props("dense no-caps unelevated "
                                       "toggle-color=primary")
                            widgets.hint(
                                "Density and viscosity move together, and so "
                                "does the vapour pressure — which this mode "
                                "does not use: a 2-D section carries no "
                                "cavitation constraint, that lives in the "
                                "hydrofoil families.")
                        widgets.number_field(
                            "speed", widgets.shown(f["V"]), flow_setter("V"),
                            unit="m/s", step=0.5, width="w-32",
                            tip="sets the dynamic pressure, the Reynolds "
                                "number and (in air) the Mach number")
                        if fluid == "air":
                            widgets.number_field(
                                "altitude", widgets.shown(f["altitude_m"]),
                                flow_setter("altitude_m"), unit="m",
                                step=100.0, width="w-32",
                                tip="ISA density, viscosity and speed of "
                                    "sound all come from it")
                        if fluid == "custom":
                            widgets.number_field(
                                "density rho",
                                widgets.shown(f["rho"], _flow_digits("rho")),
                                flow_setter("rho"), unit="kg/m³", step=0.05,
                                width="w-32",
                                tip="the fluid's density, verbatim")
                            widgets.number_field(
                                "viscosity mu",
                                widgets.shown(f["mu"], _flow_digits("mu")),
                                flow_setter("mu"), unit="Pa·s", step=1e-6,
                                width="w-32",
                                tip="DYNAMIC viscosity (Pa·s), not the "
                                    "kinematic one — nu is derived beside it")
                            widgets.number_field(
                                "Mach number",
                                widgets.shown(f["mach"], _flow_digits("mach")),
                                flow_setter("mach"), step=0.01, width="w-32",
                                tip="0 states an incompressible flow; the "
                                    "speed of sound it implies is derived "
                                    "beside it")
                    with widgets.group_box("The section's design point"):
                        widgets.number_field(
                            "reference chord",
                            widgets.shown(f["chord_m"],
                                          _flow_digits("chord_m")),
                            flow_setter("chord_m"), unit="m", step=0.05,
                            width="w-32",
                            tip="a Reynolds number is not a property of a "
                                "flow alone — this is the length in "
                                "rho V c / mu")
                        widgets.number_field(
                            "design Cl", widgets.shown(f["cl_design"]),
                            flow_setter("cl_design"), step=0.05,
                            width="w-32",
                            tip="the lift coefficient the section is scored "
                                "at: its L/D, its margins and its ranking "
                                "are all read there")
                        # ...and the Mach DECISION, in its own container:
                        # its label quotes a derived number and its very
                        # presence follows one, so typing a density beside
                        # it has to repaint it — and a view that rebuilt
                        # itself from a typed field would swallow the number
                        # being typed (the trap this whole stage is built
                        # around).
                        boxes["flow_mach"] = ui.column() \
                            .classes("w-full gap-1")
                with ui.column().classes("gap-3").style("flex:1 1 0;"
                                                        "min-width:0"):
                    with widgets.group_box("What that flow is"):
                        boxes["flow_derived"] = ui.column() \
                            .classes("w-full gap-1")

            with ui.row().classes("w-full items-center gap-2"):
                ui.button("Accept flow conditions", icon="check",
                          on_click=accept_flow) \
                    .props("unelevated dense no-caps color=primary")
                ui.button("Published defaults", icon="restart_alt",
                          on_click=reset_flow_defaults) \
                    .props("outline dense no-caps")
                ui.space()
                widgets.hint("Accepting unlocks the airfoil stage.")
        _render_flow_derived()
        _render_flow_mach()

    def _render_flow_point():
        """Stage 1's second view in airfoil-only mode: the point, and how it
        moves with the two numbers that are not the fluid's."""
        box = ctx.views[("mission", "point")]
        box.clear()
        pt = session.flow_point(S)
        with box:
            if "error" in pt:
                widgets.hint(pt["error"], "bad")
                return
            with widgets.group_box("The stated flow"):
                for key, value, why in (
                        ("fluid", session.FLUID_LABELS[str(pt["fluid"])]
                         + (f" — {api.water_kinds()[pt['water']]}"
                            if pt["water"] else ""),
                         "air and water derive their state; a custom fluid "
                         "is stated outright"),
                        ("speed V", f"{pt['v_ms']:.6g} m/s", "yours"),
                        ("reference chord c", f"{pt['chord_m']:.6g} m",
                         "the length the Reynolds number is formed on"),
                        ("density rho", f"{pt['rho']:.6g} kg/m³", ""),
                        ("viscosity mu", f"{pt['mu']:.6g} Pa·s", ""),
                        ("dynamic pressure q", f"{pt['q']:.6g} Pa",
                         "½ rho V²"),
                        ("Reynolds number", f"{pt['re']:.6g}",
                         "rho V c / mu"),
                        ("Mach number", f"{pt['mach']:.4f}"
                         + ("" if session.flown_mach(S) else
                            "   (not flown — the search runs at M 0)"), ""),
                        ("design Cl", f"{pt['cl_design']:.6f}",
                         "stage 2 screens, ranks and optimises here"),
                        ("lift per span", f"{pt['lift_per_span_n_m']:.6g} N/m",
                         "Cl q c")):
                    with ui.row().classes("w-full items-center gap-2 "
                                          "no-wrap"):
                        ui.label(key).classes("field-label").style(
                            f"color:{theme.INK_MUTED};min-width:150px")
                        ui.label(value).classes("readout").style(
                            "min-width:170px")
                        if why:
                            widgets.hint(why)
                widgets.hint("No vehicle is stated anywhere above, which is "
                             "the point of this mode: these are the three "
                             "numbers api.optimize_airfoil and "
                             "api.screen_airfoils take (Re, Mach, design "
                             "Cl), and the chord that carries the first one.")
            with widgets.group_box("Sensitivity to speed", pad=False):
                figstyle.show(_flow_sweep_fig(), "flow_design_point")
                with ui.column().classes("group-pad w-full"):
                    widgets.hint(
                        "Every point is one api.flow_point call at the same "
                        "fluid and the same chord — the dashed line is the "
                        "stated speed. The Reynolds number is linear in it; "
                        "the design Cl is NOT on the curve, because nothing "
                        "here carries a weight for the speed to trade "
                        "against. That trade is what the mission mode is "
                        "for.")

    def _flow_sweep_fig() -> go.Figure:
        """Re (and Mach, where there is one) against speed, at the stated
        chord — the same shape as the mission's own sweep, off
        ``api.flow_point`` alone."""
        f = session.flow_state(S)
        v0 = float(f["V"])
        speeds = [v0 * (0.5 + 0.05 * i) for i in range(21)]
        re, mach = [], []
        for v in speeds:
            kw = {"fluid": str(f["fluid"]), "V": v,
                  "chord_m": float(f["chord_m"]),
                  "cl_design": float(f["cl_design"])}
            if f["fluid"] == "air":
                kw["altitude_m"] = float(f["altitude_m"])
            elif f["fluid"] == "water":
                kw["water"] = str(f["water"])
            else:
                kw.update(rho=float(f["rho"]), mu=float(f["mu"]),
                          mach=float(f["mach"]))
            try:
                pt = api.flow_point(**kw)
            except ValueError:
                continue
            re.append(pt["re"])
            # a custom fluid's Mach is a STATED number, not a function of
            # the speed, so only the derived one is drawn against it
            mach.append(pt["mach"] if f["fluid"] == "air" else None)
        fig = go.Figure()
        fig.add_scatter(x=speeds[:len(re)], y=re, name="Re at the chord",
                        line=dict(color=theme.ACCENT, width=2))
        if any(m is not None for m in mach):
            fig.add_scatter(x=speeds[:len(mach)], y=mach, name="Mach",
                            yaxis="y2", line=dict(color=theme.GOOD, width=2,
                                                  dash="dot"))
        fig.add_vline(x=v0, line=dict(color=theme.INK_MUTED, width=1,
                                      dash="dash"))
        fig.update_layout(
            xaxis_title="speed [m/s]", yaxis_title="Reynolds number",
            yaxis2=dict(title="Mach", overlaying="y", side="right",
                        showgrid=False),
            height=320, legend=dict(orientation="h", y=1.08))
        return fig

    # ------------------------------------------------------ operating view
    def _render_load_note():
        """The line under the design load: what it is in mass, and where the
        number came from. It has to re-render with the field, or it goes on
        claiming the family's own point after the user has typed over it."""
        box = boxes.get("load")
        if box is None:
            return
        box.clear()
        m = S["mission"]
        own = session.mission_defaults(S["wing"]["problem"],
                                       S["medium"])["W_N"]
        with box:
            note = f"= {m['W_N'] / G0:.1f} kg × g₀"
            if float(m["W_N"]) == float(own):
                note += f" · {m.get('w_source', '')}"
                widgets.hint(note)
            else:
                widgets.hint(f"{note} · edited — this family's own point is "
                             f"{own:.6g} N")
            # ...and whether the number above is FLOWN. The water families
            # carry the load as a stated value rather than a mission field,
            # so nothing on this card ever contradicted a load the run did
            # not take (:func:`config.stated_load_note`).
            unflown = config.stated_load_note(S)
            if unflown:
                widgets.hint(unflown, "warn")

    def _render_derived():
        # the track builds its OWN derived block (_render_track_derived): a
        # car has no CL design, no wing loading and no aspect-ratio estimate
        # to attribute to stage 2, so the aircraft one would be four
        # read-outs of numbers this session does not have
        if S["medium"] == "track" and not session.airfoil_only(S):
            _render_track_derived()
            return
        box = boxes.get("derived")
        if box is None:
            return
        box.clear()
        dp = session.design_point(S)
        with box:
            if "error" in dp:
                widgets.hint(dp["error"], "bad")
                return
            # wraps: the derived row grew a Mach read-out, and a no-wrap
            # row of seven put the last one off the edge of the pane
            with ui.row().classes("w-full items-start gap-4 flex-wrap"):
                widgets.readout("CL design", f"{dp['cl_design']:.4f}", "",
                                tip="W / (q S) — the lift coefficient this "
                                    "mission trims to. No planform needed.")
                widgets.readout("q", f"{dp['q']:.4g}", "Pa")
                widgets.readout("rho", f"{dp['rho']:.4g}", "kg/m³")
                if dp.get("mach") is not None:
                    widgets.readout(
                        "Mach", f"{dp['mach']:.4f}",
                        color=(theme.WARN if dp["mach"] > session.MACH_WARN
                               else ""),
                        tip="V / sqrt(gamma R T) at the ISA temperature of "
                            "this altitude — derived, never typed")
            widgets.hairline()
            # the chord-dependent half of the point: real numbers, but ones
            # that exist only because stage 2 estimated an aspect ratio, so
            # they are drawn apart from the mission's own and labelled with
            # the number they came from
            ar = session.section_aspect_ratio(S)
            with ui.row().classes("w-full items-start gap-4 flex-wrap"):
                widgets.readout("Re at MAC", f"{dp['re_mac']:.3e}", "",
                                tip="rho V MAC / mu — the section's Reynolds "
                                    "number, at the estimated chord")
                widgets.readout("MAC", f"{dp['mac']:.4f}", "m")
                widgets.readout("span", f"{dp['b']:.3f}", "m")
                widgets.readout("AR estimate", f"{ar:.3g}", "b²/S",
                                tip="stage 2 owns this number — it decides "
                                    "which section is screened, not what is "
                                    "flown")
            widgets.hint(
                "The three above follow the aspect-ratio ESTIMATE, which "
                "lives in stage 2 with the section it is for. Change it "
                "there; stage 3 decides the span the run actually flies.")
            if dp.get("mach") is not None and dp["mach"] > session.MACH_WARN:
                widgets.hint(
                    f"M = {dp['mach']:.3f}: every section polar in this "
                    f"framework is INCOMPRESSIBLE — the cached library and "
                    f"every live CST sweep run at M = 0 — so past about "
                    f"{session.MACH_WARN:g} the section data are no longer "
                    f"the right data. The wing solver's Prandtl-Glauert "
                    f"correction can still be applied on the Solver tab; it "
                    f"corrects the wing, not the tables.", "warn")
            widgets.hairline()
            name = S["wing"]["problem"]
            size = session.flown_size(S)
            span = session.span_box(S)
            if size and span:
                widgets.hint(
                    f"The wing family selected so far ({name}) SEARCHES its "
                    f"span: stage 3's design box constrains it to "
                    f"{span[0]:.3g}–{span[1]:.3g} m, and the area follows the "
                    f"wing loading above through the weight loop rather than "
                    f"being this fixed {size[1]:.4g} m².")
            elif size:
                widgets.hint(
                    f"The wing family selected so far ({name}) honours a "
                    f"chosen size. It will fly this area and the span stage "
                    f"3's size card holds — b = {size[0]:.3f} m, S = "
                    f"{size[1]:.3f} m²"
                    + ("" if session.chosen_span(S) is not None else
                       ", the span the stage-2 aspect-ratio estimate implies, "
                       "nobody having chosen one")
                    + ". Its planform menu is where the span becomes a design "
                    "variable instead.")
            else:
                # the one family left here is the CAR, whose span is an
                # ordinary design-box row against a FIXED reference area — so
                # it neither takes a chosen size nor follows a wing loading.
                # "carries calibrated geometry" was written when the water
                # families shared this branch; they take a chosen span now,
                # and the car never had a calibration to protect.
                searched = "b_m" in api.PROBLEM_SPECS[
                    S["wing"]["problem"]].param_labels
                widgets.hint(
                    (f"{name} SEARCHES its span against its own fixed "
                     f"reference area, so the area above sets the SECTION "
                     f"design point only — the span's min and max are a row "
                     f"of stage 3's design box."
                     if searched else
                     f"{name} carries calibrated geometry (its published "
                     f"margins are quoted on it), so the area above sets the "
                     f"SECTION design point only — stage 3 flies the "
                     f"family's own planform."), "warn")
            if S["mission"]["accepted"]:
                widgets.hint("Accepted. Editing any field above un-accepts "
                             "it: a section screened at the old Reynolds "
                             "number is not this mission's section.")

    def _render_operating():
        box = ctx.views[("mission", "operating")]
        box.clear()
        # every named container in this view belonged to the build just
        # cleared, and half of them do not exist in the other mode: a stale
        # handle would have the mission's own read-out helpers clearing into
        # a column that is no longer on screen (the rule ``_render_ws_diagram``
        # already follows for its two)
        for name in ("load", "area", "taper", "ws_cap", "ws_conflict",
                     "ws_diagram", "size_fields", "derived", "flow_derived",
                     "flow_mach"):
            boxes.pop(name, None)
        if session.airfoil_only(S):
            _render_flow_view(box)
            return
        if S["medium"] == "track":
            # A CAR STATES TWO NUMBERS. The aircraft card below asks six and
            # the track problem reads one of them; the rest were asked so
            # that a lift coefficient could be backed out of them, and that
            # coefficient is now asked directly (session.TRACK_POINT_NOTE).
            _render_track_view(box)
            return
        m = S["mission"]
        water = S["medium"] == "water"
        spec = api.PROBLEM_SPECS[S["wing"]["problem"]]
        with box:
            # WHICH question this session answers, in both modes and in the
            # same place: the switch that reaches the airfoil-only mode has
            # to be reachable FROM the pipeline, or the mode is a state with
            # no control
            _mode_control()
            with widgets.group_box("Vehicle"):
                ui.toggle({k: v[0] for k, v in session.MEDIA.items()},
                          value=S["medium"],
                          on_change=lambda e: set_medium(e.value)) \
                    .props("dense no-caps unelevated toggle-color=primary")
                widgets.hint(session.MEDIA[S["medium"]][1])
                if S["medium"] == "air":
                    # a tandem is a different AIRCRAFT, not a different wing
                    # setting: it carries the weight on two surfaces, so it
                    # belongs to the mission and it changes the reference
                    # area the same weight is spread over
                    ui.toggle({"single": "One lifting surface",
                               "tandem": "Tandem pair (front + rear)"},
                              value=S["wing"]["choices"]["system"],
                              on_change=lambda e: set_system(e.value)) \
                        .props("dense no-caps unelevated "
                               "toggle-color=primary")
                    widgets.hint(
                        "The tandem pair is solved as one system (mutual "
                        "induction), references its TOTAL area, and opens on "
                        "its own design weight — twice the single wing's, "
                        "for twice the area at the same lift coefficient."
                        if S["wing"]["choices"]["system"] == "tandem" else
                        "One wing carries the load; the tail, if you add one "
                        "below, trims rather than lifts.")
                else:
                    # ...and where the pair has no solver the control does not
                    # simply vanish. It disappeared silently, directly above a
                    # second-surface switch that is greyed WITH a reason —
                    # the one asymmetry this stage was rebuilt to remove. The
                    # registry agrees the gate is right (option_available is
                    # False for both media); what was missing was saying so.
                    tog = ui.toggle({"single": "One lifting surface",
                                     "tandem": "Tandem pair (front + rear)"},
                                    value="single") \
                        .props("dense no-caps unelevated "
                               "toggle-color=primary")
                    tog.disable()
                    why = ("a tandem pair has no water solver — the imaged "
                           "foil carries one lifting surface, with an "
                           "elevator below if you add one"
                           if water else
                           "a tandem has no car-wing solver — a two-element "
                           "rear wing is a different, chordwise problem")
                    tog.tooltip(why)
                    widgets.hint(why)
                _second_surface_control()
                _empennage_control()
                _fin_control()
                if S["medium"] == "water":
                    ui.toggle(api.water_kinds(),
                              value=S.get("water", "sea"),
                              on_change=lambda e: set_water(e.value)) \
                        .props("dense no-caps unelevated "
                               "toggle-color=primary")
                    widgets.hint(
                        "Density, viscosity AND vapour pressure move "
                        "together: fresh water is lighter, slightly less "
                        "viscous, and cavitates at its own threshold, so a "
                        "lake foil is not a sea foil scaled.")

            with ui.row().classes("w-full items-start gap-3 no-wrap"):
                with ui.column().classes("gap-3").style("flex:1 1 0;"
                                                        "min-width:0"):
                    with widgets.group_box("Operating point"):
                        widgets.number_field(
                            LOAD_LABEL[S["medium"]], widgets.shown(m["W_N"]),
                            setter("W_N"), unit="N", step=10.0,
                            width="w-32",
                            tip="the load the surface must sustain; with "
                                "speed and area it sets the design lift "
                                "coefficient")
                        boxes["load"] = ui.column().classes("w-full gap-0")
                        widgets.number_field(
                            "speed", widgets.shown(m["V"]), setter("V"),
                            unit="m/s", step=0.5,
                            tip="sets dynamic pressure and the section "
                                "Reynolds number")
                        if "V" not in spec.mission_fields:
                            row = "V_ms" in session.operating_rows(S)
                            band = (session.operating_band_default(S, "V_ms")
                                    if row else None)
                            widgets.hint(
                                "This solver does not take a speed as a "
                                "mission field: "
                                + v1.mission_field_note(S["wing"]["problem"],
                                                        "V")
                                + (". The number you type here IS that box "
                                   "row's centre — stage 3 opens it on "
                                   f"{band[0]:.3g}–{band[1]:.3g} m/s, the "
                                   "family's own band width moved onto your "
                                   "speed — and it sets the section's design "
                                   "point."
                                   if band else
                                   ". Untouched it opens on the family's own "
                                   "published band; type a speed and stage "
                                   "3's row follows it. It also sets the "
                                   "section's design point."
                                   if row else
                                   ". It still sets the section's design "
                                   "point."),
                                "" if row else "warn")
                        if water:
                            widgets.number_field(
                                "depth", widgets.shown(m["depth_m"]),
                                setter("depth_m"), unit="m", step=0.05,
                                tip="submergence depth — sets the cavitation "
                                    "margin in the water solver")
                        else:
                            widgets.number_field(
                                "altitude", widgets.shown(m["altitude_m"]),
                                setter("altitude_m"), unit="m", step=100.0,
                                tip="ISA density and viscosity come from it")

                with ui.column().classes("gap-3").style("flex:1 1 0;"
                                                        "min-width:0"):
                    with widgets.group_box("Size"):
                        # ONE question — how big is the surface — asked in
                        # whichever unit the user actually has. A wing
                        # loading is what a specification carries, an AREA is
                        # what a rule or a mould gives you, and a span with a
                        # rough aspect ratio is how one is first sketched.
                        # All three write the same stored area.
                        with ui.row().classes("w-full items-center gap-1 "
                                              "no-wrap"):
                            ui.toggle({k: session.SIZE_STATEMENT_LABELS[k]
                                       for k in session.SIZE_STATEMENTS},
                                      value=session.size_statement(S),
                                      on_change=_set_size_statement) \
                                .props("dense no-caps unelevated "
                                       "toggle-color=primary")
                            widgets.help_dot(
                                "Wing loading, area, and span with an "
                                "aspect ratio are three faces of ONE stored "
                                "quantity — the reference area — so no two "
                                "of them can disagree.\n\n"
                                "Whichever you pick is the field you type "
                                "into; the other two are derived and read "
                                "out underneath it. The span here only "
                                "sketches the area: the span the run "
                                "actually flies is stated on stage 3.",
                                title="Why three buttons for one number?")
                        widgets.hint("State the size in whatever you have.")
                        boxes["size_fields"] = ui.column() \
                            .classes("w-full gap-0")
                        _render_size_fields()
                        boxes["area"] = ui.column().classes("w-full gap-0")
                        boxes["taper"] = ui.column().classes("w-full gap-0")
                        # WHO CAPS IT — asked before the two answers are
                        # compared, because on "the W/S I typed" there is
                        # nothing to compare and the comparison must not be
                        # the only way to reach the switch
                        boxes["ws_cap"] = ui.column().classes("w-full gap-1")
                        _render_ws_cap()
                        # the two answers to this one question, compared —
                        # ABOVE the expansion, because the ceiling inside it
                        # is what refuses every design of a run that states a
                        # loading over it, and a limit nobody opened is a
                        # limit nobody was told about
                        boxes["ws_conflict"] = ui.column().classes(
                            "w-full gap-1")
                        _render_ws_conflict()
                        boxes["ws_diagram"] = ui.column().classes(
                            "w-full gap-1")
                        _render_ws_diagram()

            with widgets.group_box("Implied design point"):
                boxes["derived"] = ui.column().classes("w-full gap-2")

            with ui.row().classes("w-full items-center gap-2"):
                ui.button("Accept mission", icon="check",
                          on_click=accept) \
                    .props("unelevated dense no-caps color=primary")
                ui.button("Published defaults", icon="restart_alt",
                          on_click=reset_defaults) \
                    .props("outline dense no-caps")
                ui.space()
                widgets.hint("Accepting unlocks the airfoil stage.")
        _render_load_note()
        _render_area_note()
        _render_derived()

    # --------------------------------------------------------- point view
    def _sweep_fig() -> go.Figure:
        """CL and Re against speed, from ``api.design_point`` alone.

        The ONE place a stage rebuilds a design-point call by hand, so it
        has to carry the same arguments :func:`session.design_point` does —
        ``water`` included. It did not, and took the argument's "sea"
        default: in fresh water the curves (rho 1025) contradicted the
        read-outs in the group box directly above them (rho 998.2) by 2.6 %
        in CL and 4.7 % in Re, and the dashed line marking the mission speed
        crossed neither of the numbers the same page states.
        """
        m = S["mission"]
        v0 = float(m["V"])
        speeds = [v0 * (0.5 + 0.05 * i) for i in range(21)]
        cl, re = [], []
        for v in speeds:
            try:
                dp = api.design_point(
                    medium=S["medium"], W_N=float(m["W_N"]), V=v,
                    s_ref_m2=float(m["s_ref_m2"]),
                    aspect_ratio=session.section_aspect_ratio(S),
                    taper=float(m["taper"]),
                    altitude_m=float(m["altitude_m"]),
                    water=S.get("water", "sea"),
                    depth_m=(float(m["depth_m"]) if S["medium"] == "water"
                             else None))
            except ValueError:
                continue
            cl.append(dp["cl_design"])
            re.append(dp["re_mac"])
        fig = go.Figure()
        fig.add_scatter(x=speeds[:len(cl)], y=cl, name="CL design",
                        line=dict(color=theme.ACCENT, width=2))
        fig.add_scatter(x=speeds[:len(re)], y=re, name="Re at MAC",
                        yaxis="y2", line=dict(color=theme.GOOD, width=2,
                                              dash="dot"))
        fig.add_vline(x=v0, line=dict(color=theme.INK_MUTED, width=1,
                                      dash="dash"))
        fig.update_layout(
            xaxis_title="speed [m/s]", yaxis_title="CL design",
            yaxis2=dict(title="Re at MAC", overlaying="y", side="right",
                        showgrid=False),
            height=320, legend=dict(orientation="h", y=1.08))
        return fig

    def _track_sweep_fig() -> go.Figure:
        """Re and downforce against speed, at the STATED CZ.

        The aircraft sweep plots CL against speed, because an aeroplane
        holds a weight and its lift coefficient is whatever trims it. A car
        holds nothing: the coefficient is what the section is designed at
        and it does not move with the speed, so the two things that DO move
        are drawn instead — the Reynolds number, and what the reference
        coefficient is worth in newtons.
        """
        m = S["mission"]
        dp0 = session.design_point(S)
        v0 = float(m["V"])
        cz = session.track_design_cz(S)
        rho, mu = float(dp0["rho"]), float(dp0["mu"])
        mac, s_ref = float(dp0["mac"]), float(dp0["s_ref_m2"])
        speeds = [v0 * (0.4 + 0.06 * i) for i in range(21)]
        re = [rho * v * mac / mu for v in speeds]
        fz = [cz * 0.5 * rho * v * v * s_ref for v in speeds]
        fig = go.Figure()
        fig.add_scatter(x=speeds, y=fz, name=f"downforce at CZ {cz:g}",
                        line=dict(color=theme.ACCENT, width=2))
        fig.add_scatter(x=speeds, y=re, name="Re at MAC", yaxis="y2",
                        line=dict(color=theme.GOOD, width=2, dash="dot"))
        fig.add_vline(x=v0, line=dict(color=theme.INK_MUTED, width=1,
                                      dash="dash"))
        fig.update_layout(
            xaxis_title="speed [m/s]", yaxis_title="downforce [N]",
            yaxis2=dict(title="Re at MAC", overlaying="y", side="right",
                        showgrid=False),
            height=320, legend=dict(orientation="h", y=1.08))
        return fig

    def _render_track_point():
        """Stage 1's second view for a car: what the two stated numbers are,
        and what they imply at the size stage 3 is allowed to search."""
        box = ctx.views[("mission", "point")]
        box.clear()
        dp = session.design_point(S)
        size = session.track_reference_size(S)
        with box:
            if "error" in dp:
                widgets.hint(dp["error"], "bad")
                return

            def _table(rows):
                for key, value, why in rows:
                    with ui.row().classes("w-full items-center gap-2 "
                                          "no-wrap"):
                        ui.label(key).classes("field-label").style(
                            f"color:{theme.INK_MUTED};min-width:150px")
                        ui.label(value).classes("readout").style(
                            "min-width:150px")
                        widgets.hint(why)

            with widgets.group_box("What this card states"):
                _table([
                    ("speed V", f"{float(S['mission']['V']):.6g} m/s",
                     "the only number here the car-wing solver reads"),
                    ("reference CZ", f"{session.track_design_cz(S):.4f}",
                     "the lift coefficient stage 2 designs the section at"),
                ])
                widgets.hint(session.TRACK_POINT_NOTE)

            with widgets.group_box("The air, which is not asked"):
                _table([
                    ("density rho", f"{dp['rho']:.6g} kg/m³",
                     "carwing.RHO_AIR — ISA at sea level, fixed in the "
                     "solver"),
                    ("viscosity mu", f"{dp['mu']:.6g} Pa·s",
                     "Sutherland at the same state"),
                    ("dynamic pressure q", f"{dp['q']:.6g} Pa", "½ rho V²"),
                    ("Mach", f"{dp['mach']:.4f}" if dp.get("mach") is not None
                     else "—",
                     "derived; every polar in this framework is "
                     "incompressible"),
                ])

            title = ("At the middle of stage 3's design box"
                     if size else "At this family's own planform")
            with widgets.group_box(title):
                _table([
                    ("reference area S", f"{dp['s_ref_m2']:.4f} m²",
                     "the S_m2 row — a band on a design variable, not a "
                     "number derived from a weight"),
                    ("span b", f"{dp['b']:.4f} m", "the b_m row"),
                    ("MAC", f"{dp['mac']:.4f} m", "mean aerodynamic chord"),
                    ("Re at MAC", f"{dp['re_mac']:.6g}",
                     "rho V MAC / mu — stage 2 screens at this Re"),
                    ("downforce at the reference CZ",
                     f"{dp['cl_design'] * dp['q'] * dp['s_ref_m2']:.4g} N",
                     "CZ q S — what the coefficient is worth, so a drag "
                     "ceiling or downforce floor in stage 3 has a scale"),
                ])
                widgets.hint(
                    "Both dimensions are SEARCHED, so this is a mid-box "
                    "wing and not a chosen one. Narrow either row and every "
                    "number here follows it; the run reports the chord and "
                    "the Reynolds number it actually flew.")

            with widgets.group_box("Sensitivity to speed", pad=False):
                figstyle.show(_track_sweep_fig(), "mission_track_point")
                with ui.column().classes("group-pad w-full"):
                    widgets.hint(
                        "The dashed line is the stated speed. The "
                        "COEFFICIENT does not move with it — that is the "
                        "difference between a car and an aeroplane: nothing "
                        "trims a rear wing, so the section keeps the design "
                        "point you stated and what changes is its Reynolds "
                        "number and the force the same coefficient makes.")

    def _render_point():
        if session.airfoil_only(S):
            _render_flow_point()
            return
        if S["medium"] == "track":
            _render_track_point()
            return
        box = ctx.views[("mission", "point")]
        box.clear()
        dp = session.design_point(S)
        with box:
            if "error" in dp:
                widgets.hint(dp["error"], "bad")
                return
            ar = session.section_aspect_ratio(S)

            def _table(rows):
                for key, value, why in rows:
                    with ui.row().classes("w-full items-center gap-2 "
                                          "no-wrap"):
                        ui.label(key).classes("field-label").style(
                            f"color:{theme.INK_MUTED};min-width:150px")
                        ui.label(value).classes("readout").style(
                            "min-width:150px")
                        widgets.hint(why)

            with widgets.group_box("Derived from the mission alone"):
                _table([
                    ("density rho", f"{dp['rho']:.6g} kg/m³",
                     "ISA at the altitude, or sea water"),
                    ("viscosity mu", f"{dp['mu']:.6g} Pa·s",
                     "Sutherland at the ISA temperature, or sea water"),
                    ("dynamic pressure q", f"{dp['q']:.6g} Pa",
                     "½ rho V²"),
                    ("reference area S", f"{dp['s_ref_m2']:.4f} m²",
                     "W / (W/S) — what the lift coefficient is defined on"),
                    ("CL design", f"{dp['cl_design']:.6f}",
                     "W / (q S) — stage 2 screens at this lift coefficient"),
                ])
                widgets.hint("None of these needs a planform, which is why "
                             "the mission can state them.")

            with widgets.group_box(f"At the stage-2 aspect-ratio estimate "
                                   f"(AR {ar:.4g})"):
                _table([
                    ("span b", f"{dp['b']:.4f} m", "√(AR · S)"),
                    ("root chord", f"{dp['c_root']:.4f} m",
                     "reference trapezoid at the mean chord (taper is the "
                     "solver's variable, not a mission input)"),
                    ("MAC", f"{dp['mac']:.4f} m",
                     "mean aerodynamic chord, closed form"),
                    ("Re at MAC", f"{dp['re_mac']:.6g}",
                     "rho V MAC / mu — stage 2 screens at this Re"),
                ])
                widgets.hint(
                    "An estimate is all the section needs: it decides which "
                    "chord the polars are flown at. The span the RUN flies is "
                    "stage 3's, and may differ — in which case the section "
                    "was designed for a slightly different Reynolds number, "
                    "and both stages say so.")
            with widgets.group_box("Sensitivity to speed", pad=False):
                figstyle.show(_sweep_fig(), "mission_design_point")
                with ui.column().classes("group-pad w-full"):
                    widgets.hint(
                        "Every point is one api.design_point call at the "
                        "same weight and area — the dashed line is the "
                        "mission speed. Flying slower raises the lift "
                        "coefficient the section must work at and lowers its "
                        "Reynolds number; both change which section wins in "
                        "stage 2.")

    # ------------------------------------------------- view: search policy
    def _set_search_mode(mode: str):
        if not session.set_search_mode(S, str(mode)):
            return
        ctx.log("search settings: "
                + ("the measured recommendation is live — every stage's "
                   "budget follows its own design vector"
                   if session.search_is_recommended(S)
                   else "your own values are live — nothing moves them"),
                "info")
        ctx.render("mission", "search")
        ctx.render_when_shown("airfoil")
        ctx.render_when_shown("airfoil_aft")
        ctx.render_when_shown("wing")
        ctx.refresh()

    def _set_effort(value: str):
        if not session.set_search_effort(S, str(value)):
            return
        ctx.render("mission", "search")
        ctx.render_when_shown("airfoil")
        ctx.render_when_shown("airfoil_aft")
        ctx.render_when_shown("wing")
        ctx.refresh()

    def _set_stop(on: bool):
        session.set_search_stop(S, bool(on))
        ctx.render("mission", "search")
        ctx.render_when_shown("wing")
        ctx.refresh()

    def _take_over():
        """Hand the recommendation's numbers to the user and switch to own."""
        session.adopt_recommendation(S)
        ctx.log("the recommended numbers were copied into stages 2 and 3 — "
                "they are yours now, and nothing in this shell moves them "
                "again", "info")
        ctx.render("mission", "search")
        ctx.render_when_shown("airfoil")
        ctx.render_when_shown("airfoil_aft")
        ctx.render_when_shown("wing")
        ctx.refresh()

    def _plan_row(label: str, plan, what: str):
        k = max(1, int(plan.n_restarts))
        with ui.row().classes("w-full items-start gap-4 flex-wrap"):
            widgets.readout("stage", label)
            widgets.readout("design variables", str(plan.dim))
            widgets.readout("optimiser", plan.label
                            + (f" × {k} restarts" if k > 1 else ""))
            widgets.readout("expected",
                            plan.est_text if k == 1
                            else f"{plan.est_text} × {k}",
                            tip="physics plus the optimiser's own cost, at "
                                "the speed measured on this machine")
        widgets.hint(f"{what} — {plan.why}.")

    def _render_search():     # noqa: PLR0915
        box = ctx.views[("mission", "search")]
        box.clear()
        st = session.search_state(S)
        study = session.search_study()
        recommended = session.search_is_recommended(S)
        with box:
            with widgets.group_box("Where the search settings come from"):
                for mode, title, why in (
                        ("recommended", "The measured recommendation",
                         "the optimiser and the budget follow a study of "
                         "these problems — sized from the number of design "
                         "variables and what is being optimised, per stage"),
                        ("own", "My own values",
                         "the budget and optimiser fields on stages 2, 2.5 "
                         "and 3 are yours; nothing here moves them")):
                    selected = (mode == st["mode"])
                    row = ui.row().classes(
                        "w-full items-start gap-2 no-wrap").style(
                        "padding:3px 4px;border-radius:2px;"
                        + (f"background:{theme.ACCENT_FILL}"
                           if selected else ""))
                    with row:
                        ui.icon("radio_button_checked" if selected
                                else "radio_button_unchecked") \
                            .style(f"color:{theme.ACCENT if selected else theme.INK_FAINT}")
                        with ui.column().classes("gap-0"):
                            ui.label(title).classes("readout")
                            widgets.hint(why)
                    row.on("click", lambda _, m=mode: _set_search_mode(m))
                if study is None:
                    widgets.hint(
                        "The study is not installed in this checkout "
                        "(data/search_budget.json), so there is nothing to "
                        "recommend and your own values are what runs. "
                        "Rebuild it with aerobo.experiments.budget_study.",
                        "warn")

            if recommended and study is not None:
                with widgets.group_box("How converged"):
                    widgets.select_field(
                        "aim for", session.SEARCH_EFFORT_LABELS,
                        st["effort"], lambda e: _set_effort(str(e.value)),
                        tip="how close to the best design the study ever "
                            "found the budget is sized to get")
                    widgets.hint(
                        "“Converged” is a target, not an event: the budget "
                        "is the number of evaluations at which the study's "
                        "runs had reached this fraction of the best design "
                        "they ever found, measured on the same class of "
                        "problem at the same number of design variables.")
                    ui.switch("stop early if it stops improving",
                              value=bool(st["stop_when_converged"]),
                              on_change=lambda e: _set_stop(bool(e.value))) \
                        .props("dense")
                    widgets.hint(
                        "The budget stays the ceiling — this only ends a run "
                        "that has flattened out, and the partial result is "
                        "kept exactly as a cancelled run's is.")

                with widgets.group_box("What that means, stage by stage"):
                    # the WING's plan, where this session has a wing stage
                    # at all: an airfoil-only session's search policy is the
                    # section's, and a row for a stage it does not have is a
                    # budget nobody will spend
                    plan = (None if session.airfoil_only(S)
                            else session.wing_plan(S))
                    if plan is not None:
                        _plan_row("3 · Wing", plan,
                                  f"“{api.PROBLEM_SPECS[S['wing']['problem']].display}”")
                    for stage, surface in (("airfoil", "main"),
                                           ("airfoil_aft", "aft")):
                        if not session.stage_visible(S, stage):
                            continue
                        ap = session.airfoil_plan(S, surface)
                        if ap is None:
                            continue
                        _plan_row(session.STAGE_LABELS[stage].strip(), ap,
                                  "CST section + live XFOIL")
                    if st.get("error"):
                        widgets.hint(st["error"], "bad")
                    widgets.hint(
                        "These follow the configuration: open another design "
                        "freedom and the vector grows, so the budget does "
                        "too. The stages show the same numbers and say they "
                        "came from here.")
                    ui.button("use these as my own values instead",
                              icon="edit", on_click=_take_over) \
                        .props("flat dense no-caps")

            if not recommended:
                with widgets.group_box("What that means, stage by stage"):
                    if not session.airfoil_only(S):
                        eff = session.effective_wing_search(S)
                        widgets.kv(
                            "3 · Wing",
                            f"{api.OPTIMISER_SPECS[eff['optimiser']].display}"
                            f", {int(eff['budget'])} evaluations"
                            + (f" × {int(eff['n_seeds'])} seeds"
                               if int(eff["n_seeds"]) > 1 else ""))
                    for stage, surface in (("airfoil", "main"),
                                           ("airfoil_aft", "aft")):
                        if not session.stage_visible(S, stage):
                            continue
                        a = session.effective_airfoil_search(S, surface)
                        widgets.kv(session.STAGE_LABELS[stage].strip(),
                                   f"{int(a['budget'])} evaluations")
                    widgets.hint(
                        "Your numbers, edited on the stages themselves. This "
                        "card still shows them, because what the search will "
                        "do is a stage-1 question either way.")

            if study is not None:
                with widgets.group_box("Provenance"):
                    widgets.kv("measured", str(study.get("measured") or "—"))
                    widgets.kv("runs", str(study.get("n_runs") or "—"))
                    widgets.kv("problems", str(study.get("cases") or "—"))
                    if study.get("machine"):
                        widgets.kv("machine", str(study["machine"]))
                    if study.get("note"):
                        widgets.hint(str(study["note"]))

    # ------------------------------------------------------------ wiring
    ctx.on_render("mission", "operating", _render_operating)
    ctx.on_render("mission", "point", _render_point)
    ctx.on_render("mission", "search", _render_search)
    # both of stage 1's read-out views follow the DESIGN VECTOR, which stages
    # 2 and 3 own: the Search tab quotes the dimension, the optimiser and the
    # wall clock every stage will fly, and the operating view quotes the
    # family sentence, the flown span and the taper interval the section is
    # screened over. Declared here so any stage's setter repaints them
    # (Ctx.refresh) instead of each one remembering to.
    ctx.on_derived("mission", "operating")
    ctx.on_derived("mission", "search")
    # ...and the derived-point view, which is read-outs end to end (density,
    # viscosity, vapour pressure, the reference area, the design Cl). The
    # medium can be changed from stage 3's Configuration card as well as
    # from here, and doing so left this table quoting air's 1.225 kg/m³ for
    # a water session until something else happened to rebuild it.
    # ...with its own contents as the signature: ``design_point`` is 0.01 ms
    # against a 22 ms repaint, and this view is a table of exactly it
    ctx.on_derived("mission", "point",
                   sig=lambda: (tuple(sorted(
                       (k, str(v)) for k, v in session.design_point(S).items()
                   )), session.section_aspect_ratio(S)))
    ctx.register("set_search_mode", _set_search_mode)
    ctx.register("set_search_effort", _set_effort)
    ctx.register("set_search_stop", _set_stop)
    ctx.register("adopt_search_values", _take_over)
    ctx.register("accept_mission", accept)
    # the RESET, as an action and not only a button handler: it is the one
    # place that re-derives the whole operating point, and on the track it
    # takes its own branch (no design load, no aspect-ratio estimate), so it
    # has to be reachable by something other than a click
    ctx.register("reset_mission", reset_defaults)
    # ...and the car's own two numbers, for the same reason
    ctx.register("set_track_cz", _set_track_cz_value)
    # ...and the airfoil-only mode's own four: the switch that changes what
    # the session designs, and the three controls of the flow it states
    ctx.register("set_mode", _set_mode)
    ctx.register("set_flow", set_flow_value)
    ctx.register("set_fluid", set_fluid)
    ctx.register("set_flow_mach", set_flow_mach)
    # the same handlers the switches call — registered so the shell (and the
    # smoke tests) can move the mission's own three choices through the REAL
    # path, family rules and re-renders included. Driving
    # ``set_choice("medium", …)`` instead exercises stage 3's handler, which
    # is not the path any user takes: stage 3 only shows the medium and the
    # lifting system, with a button back to here.
    ctx.register("set_second_surface", set_tail)
    # ...and the VERTICAL one, the same kind of question one surface
    # further out: does this aircraft have a fin at all?
    ctx.register("set_fin", set_fin)
    # ...and WHICH EMPENNAGE they are arranged as. Registered because the
    # layout decides which stages exist — a V-tail has no stage 2.7 — so it
    # has to be drivable without a browser, like every other question on this
    # stage that changes the shape of the pipeline.
    ctx.register("set_empennage", set_tail_type)
    ctx.register("set_medium", set_medium)
    # the constraint diagram's two verbs, registered so the shell (and the
    # tests) drive the REAL path rather than poking the mission dict
    ctx.register("set_ws_input", _set_ws)
    ctx.register("adopt_ws", _adopt_ws)
    ctx.register("set_lifting_system", set_system)
