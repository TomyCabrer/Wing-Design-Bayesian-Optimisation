"""Stage 5 — CONTROLS: what the design is flown WITH.

Stages 1-4 size and score a lifting surface against a mission. Nothing in
that mission ever says how the aircraft is to be rolled, or what stops it
weathercocking, so the design arrives here with no control surfaces and — the
measured fact this stage exists for — with a yaw stiffness of EXACTLY zero.
That is a hole in the geometry, not a stable aeroplane, and the derivatives
view says so in those words until a fin is added.

Three questions, asked once each:

* **Ailerons** — cut out of the planform the search produced, as a span
  band and a hinge-chord fraction (``dynamics.aileron``), and the
  **elevator**'s hinge chord, which at 1.0 IS the all-moving stabiliser
  the solver already trims with.
* **The vertical surface** — a fin and its rudder. Where it goes is ONE
  NUMBER: the station of its LEADING EDGE, drawn on a side elevation
  (:mod:`gui.v4.sideview`) beside the wing's and the stabiliser's, so the
  question is answered in a coordinate the user can see rather than chosen
  from a menu of two words. A hydrofoil answers it differently — the strut
  already sits between the wing and the tail, so that is where its vertical
  goes, and it hangs DOWN because the foil does not pierce the free surface
  — but that is a different NUMBER, not a different control.
* **How much rudder** — recommended from the sideslip full deflection has
  to hold, measured by rebuilding rather than scaled, and never applied.
* **Is the result actually stable** — static margin, yaw stiffness, dihedral
  effect and the roll mode, each beside the sign it has to have.

The deck this stage builds is what stage 6 flies. Nothing is re-derived
there.
"""

from __future__ import annotations

import traceback

import numpy as np

from aerobo import fin as _fin
from aerobo.flightmodel import QUARTER
from gui.v3 import session as v3s, theme, widgets
from gui.v4 import (fields as fl, modes as md, session as v4s,
                    sideview as sv, stick as stk)

#: the four stages V4 inherits from V3, named once. Everything ``ctx``
#: knows about as a DERIVED view lives in one of them, and none of those
#: views can quote a control surface — see ``edit`` for the measurement.
V3_STAGES: tuple = tuple(v3s.STAGES)


def build(ctx):
    from nicegui import ui

    S = ctx.S
    C = S["controls"]

    # ------------------------------------------------------------- helpers
    def _spec():
        from aerobo.flightmodel import ControlsSpec
        a, v = C["aileron"], C["vertical"]
        t = C.get("thrust") or {"through_cg": True, "z_below_cg_m": 0.0}
        return ControlsSpec(
            aileron=bool(a["on"]),
            aileron_span=(float(a["span_from"]), float(a["span_to"])),
            aileron_chord_frac=float(a["chord_frac"]),
            # THE SWITCH SAYS "off = all-moving stabiliser", SO OFF MUST SEND
            # ONE. ``ControlsSpec.elevator=False`` means the deck is built
            # with no elevator column at all (flightmodel.py:632), which is
            # not a stabilator — it is an aeroplane with no pitch control.
            # ``sixdof.trim_level`` solves for alpha AND one pitch control,
            # so it raised "no control column named 'elevator' in this deck"
            # before stage 6's first frame, and the Fly view answered that
            # with a warning and nothing else: the aeroplane became
            # unflyable because a switch was moved to the setting its own
            # caption describes. A WHOLE-CHORD HINGE IS THE ALL-MOVING
            # SURFACE, bit for bit — this stage's own hint says so and
            # tests/test_control_surfaces.py::
            # test_a_whole_chord_elevator_IS_the_lattice_all_moving_tail
            # pins ``chord_frac=1.0`` to ``VLM._CLi`` at rel=1e-12.
            elevator=True,
            elevator_chord_frac=(float(C["elevator"]["chord_frac"])
                                 if C["elevator"]["on"] else 1.0),
            fin=bool(v["on"]),
            fin_height_m=v["height_m"], fin_chord_m=v["chord_m"],
            fin_x_le_m=v["x_le_m"], fin_ventral=bool(v["ventral"]),
            rudder_chord_frac=float(v["rudder_chord_frac"]),
            thrust_through_cg=bool(t["through_cg"]),
            thrust_z_below_cg_m=float(t["z_below_cg_m"] or 0.0))

    def _report():
        return S["run"].get("report")

    def _point() -> dict:
        """THE MISSION'S OWN FLUID STATE — one funnel.

        Speed, density AND viscosity, because the fin has a Reynolds number
        as well as a dynamic pressure and a fin sized in the wrong fluid is
        not the fin that flies. Same source as every other stage
        (:func:`gui.v3.session.design_point`), so a mission change moves all
        of them together.
        """
        try:
            dp = v3s.design_point(S)
        except Exception:                           # noqa: BLE001
            dp = {}
        if not isinstance(dp, dict) or dp.get("error"):
            dp = {}
        return dp

    def _air() -> tuple[float, float]:
        """The speed and density every deck here is built in."""
        dp = _point()
        return (float(dp.get("v_ms") or (S.get("mission") or {}).get("V")
                      or 45.0),
                float(dp.get("rho") or 1.225))

    def _reynolds(chord: float) -> float | None:
        """``rho.V.c/mu`` at the mission point, or None if it has no state."""
        dp = _point()
        mu = dp.get("mu")
        V, rho = _air()
        if not mu or not chord or chord <= 0.0:
            return None
        return float(rho) * float(V) * float(chord) / float(mu)

    def rebuild():
        """Rebuild the deck from the CURRENT answers. Cheap — the lattice is
        one build and the whole deck is a handful of back-substitutions — so
        it runs inline on every edit rather than behind a button."""
        rep = _report()
        # WHICH DESIGN AND WHICH AIR THIS DECK IS FOR, recorded whether the
        # build works or not: :func:`_ensure_deck` reads both to tell
        # "already current" from "already tried and failed", and stage 6
        # reaches it through the same function. Written BEFORE the build so
        # a rebuild that raises is not re-paid on every repaint of every
        # view that asks.
        #
        # THE POINT IS HALF THE KEY. The deck is built from the spec AND
        # from ``_air()``, which reads the mission live; keyed on the report
        # alone, a stage-1 speed edit left C["deck"] at the old air while
        # every comparison lattice on this stage (the Reynolds block, the
        # rudder solve, the fin defaults) called ``_air()`` fresh and quoted
        # the new. Two answers on one panel, for two different aeroplanes.
        C["deck_report"] = rep
        C["deck_point"] = _air()
        # ...AND EVERY DERIVED ANSWER GOES WITH THE DECK. The spiral scan
        # and the dihedral scan are verdicts about a particular aeroplane
        # ("6.13 deg of wing dihedral turns this margin"), and a new report
        # left them on screen unchanged — measured bit-identical across two
        # different designs. The session's own comment says they are cleared
        # whenever the design changes; this is the only place that runs when
        # it does.
        C["spiral_scan"] = None
        C["dihedral_scan"] = None
        C["size_note"] = None
        if not rep:
            # ...AND THE MODEL WITH IT. Nulling only the deck left ``C["fm"]``
            # pointing at the previous design's lattice, which is the object
            # stage 6 flies (flight.py::_fm) — so a session whose report had
            # gone (a re-run in progress, a reloaded window) armed and flew an
            # aeroplane that no stage on screen described.
            C["fm"] = C["deck"] = None
            C["rec_cache"] = {}
            C["error"] = ("stage 4 has not finished re-evaluating the "
                          "design yet — its geometry is what the control "
                          "surfaces are cut out of")
            return
        try:
            from aerobo.flightmodel import build_flight_model

            # THE MISSION'S OWN AIR, not sea level. A hydrofoil is designed
            # in water at 1000 kg/m3 and an aeroplane at altitude is not at
            # 1.225: a deck built at the wrong density is a different
            # aeroplane, and stage 6 would fly that one.
            V, rho = _air()
            fm = build_flight_model(rep, _spec(), V=V, rho=rho)
            C["fm"] = fm
            C["deck"] = fm.deck
            C["error"] = None
            # EVERY DERIVED ANSWER BELONGS TO THIS DECK. The two
            # recommendations and the all-blank fin each cost real lattices
            # (1 + 4 + 1 = six builds) and they were paid on every PAINT of
            # the Fin & rudder view — the view that repaints on every edit,
            # on every tab click and once at assembly. They depend on
            # nothing but the answers this rebuild just read, so they are
            # memoised here and thrown away with the deck they describe.
            C["rec_cache"] = {}
        except Exception as exc:                    # noqa: BLE001
            C["fm"] = C["deck"] = None
            C["error"] = f"{type(exc).__name__}: {exc}"
            C["rec_cache"] = {}
            print("[v4] controls rebuild failed:\n" + traceback.format_exc())

    #: THE STAGES THIS ONE CANNOT MOVE, so their derived views are never
    #: repainted for it.
    #:
    #: ``ctx.refresh()`` repaints every view that declared itself derived —
    #: mission/operating, mission/point, mission/search, airfoil/screen,
    #: airfoil/optimise and wing/solver. Not one of them can quote a control
    #: surface: an aileron band is not in the design vector, not in the
    #: mission and not in the design box, and stage 5 runs entirely AFTER
    #: the search. Repainting them was pure cost, and it was not small.
    #: Measured in a browser on a seeded shell, one switch on this stage:
    #:
    #:     before   1360 ms   6829 DOM mutations   1264 nodes added
    #:     after     253 ms    168 DOM mutations     50 nodes added
    #:
    #: ...and the AFTER row was taken with a side elevation and a
    #: three-rebuild rudder recommendation already added to the same
    #: view, so the comparison is if anything unkind to the fix.
    #:
    #: The CHROME still repaints (``refresh`` always calls it first), which
    #: is what takes the stage from "ready" to "done" in the tree.
    def _cached(name: str, make, *on):
        """``make()``, once per rebuild AND per value of ``on``.

        The rebuild is most of the key: everything these answers read comes
        off the deck, and a new deck gets a new cache. ``on`` is for the
        rest — an input the recommendation reads that is NOT in the spec the
        lattice was built from. There is exactly one today (the sideslip the
        rudder is asked to hold) and it caught this the honest way: keyed on
        the rebuild alone, three different targets in a row returned the
        first one's answer, because moving a target that the LATTICE does
        not depend on does not make a new deck.

        A miss on a cache that no rebuild has created yet still computes —
        this must never be the reason a recommendation is missing, only the
        reason it is fast.
        """
        box = C.get("rec_cache")
        key = (name, *on)
        if box is None:
            return make()
        if key not in box:
            box[key] = make()
        return box[key]

    def edit(path, value):
        node = C
        for k in path[:-1]:
            node = node[k]
        node[path[-1]] = value
        # a recommendation belongs to the design it was computed for, and
        # ``rebuild`` — which every edit runs — is what throws it away. The
        # sweep costs 28 ms and would happily go on showing a verdict about
        # a fin that is no longer fitted.
        rebuild()
        # ...and only the view being LOOKED AT. The other three are
        # re-rendered by ``Ctx.select`` the moment their tab is clicked
        # (``app._select_view``), so building them now builds them twice
        # and shows the second one.
        ctx.render("controls", S["ui"]["tab"]["controls"])
        ctx.refresh(*V3_STAGES, "controls")

    #: how long the browser waits after the last keystroke before the number
    #: is sent. ``ui.number`` fires its handler on EVERY keystroke, and this
    #: stage's setters rebuild the panel the field lives in — the whole panel
    #: is derived from the fin, so there is nothing to show if it does not.
    #: Measured in a browser without this: typing "1.85" into the fin height
    #: committed "1", rebuilt the view under the cursor, and threw ".85" away;
    #: the field read 1 and the session stored 1.0.
    FIELD_DEBOUNCE_MS = 600

    def num(*a, **kw):
        """:func:`gui.v3.widgets.number_field`, committing when you STOP
        typing. Same row, same look — a stage does not get its own widget,
        only its own timing."""
        el = widgets.number_field(*a, **kw)
        el.props(f"debounce={FIELD_DEBOUNCE_MS}")
        return el

    #: returned by :func:`_number` when what arrived is not an answer, so a
    #: field whose blank means "keep what you had" can tell that apart from a
    #: field whose blank means "let the model choose" (fin height, chord,
    #: station, and the two sizing targets).
    _KEEP = object()

    def _number(e, blank=_KEEP):
        """THE NUMBER, off a nicegui CHANGE EVENT.

        ``ui.number(on_change=...)`` hands its handler a
        ``ValueChangeEventArguments``, not a value, and every numeric field
        in this stage used to treat it as one. That had a quiet half and a
        loud half. Quiet: ``float(event)`` raises inside the handler, where
        nicegui logs it and the user sees the field simply not take — no
        number typed into stage 5 has EVER committed. Loud: the three fields
        that stored the argument straight (``lambda x: edit(..., x)``) put
        the event object into the session, so the next render handed it to
        ``ui.number`` as a value and the whole view died in nicegui's own
        formatter with ``int() argument must be ... not
        'ValueChangeEventArguments'``.

        One funnel, so the next field added here cannot get it wrong on its
        own; a test fires every field's handler and checks the session took
        a float.
        """
        v = getattr(e, "value", e)
        if v is None or v == "":
            return blank
        try:
            return float(v)
        except (TypeError, ValueError):
            return _KEEP

    def edit_num(path, e, *, blank=_KEEP):
        """``edit`` for a numeric field: nothing is stored unless a number
        (or an intended blank) arrived."""
        v = _number(e, blank)
        if v is not _KEEP:
            edit(path, v)

    def set_target(key, e):
        """A sizing TARGET, stored on the stage with no rebuild — the fin is
        not resized until the button beside the field is pressed.

        Goes through the same funnel because the sentinel must not reach the
        session either: stored, it would come back as ``ui.number``'s value
        on the next render, which is the crash this all started with. The
        mapping is total — there is no branch that leaves a target holding
        something that cannot be drawn again.
        """
        v = _number(e, blank=None)
        C[key] = None if v is _KEEP else v

    def _band(label, group, note="", help="", enabled=True):
        """The three numbers a hinged band is: where it starts, where it
        ends, and how much of the chord moves.

        ``note`` is the short line under them; ``help`` is the paragraph
        behind the ``?`` beside it. Nothing here is a limit — a band that
        overlaps its neighbour is reported by the deck, not refused.
        """
        g = C[group]
        with ui.row().classes("items-end gap-3"):
            num(
                "from (frac of semi-span)", g["span_from"],
                lambda e, gr=group: edit_num((gr, "span_from"), e),
                step=0.05, enabled=enabled)
            num(
                "to", g["span_to"],
                lambda e, gr=group: edit_num((gr, "span_to"), e),
                step=0.05, enabled=enabled)
            num(
                "hinge chord (frac)", g["chord_frac"],
                lambda e, gr=group: edit_num((gr, "chord_frac"), e),
                step=0.05, enabled=enabled)
        if not enabled:
            # A FIELD THAT TAKES AN ANSWER NOTHING READS. With the switch
            # above off, ``_spec()`` sends the surface as absent and
            # ``build_flight_model`` cuts nothing — measured: the deck's
            # columns are identical before and after moving span_from from
            # 0.60 to 0.10 — while every keystroke here still paid a full
            # lattice rebuild. Disabled rather than hidden, so the numbers
            # the surface WOULD have are still readable.
            widgets.hint("Switched off above — these are what it would be "
                         "cut to.")
        if note and help:
            widgets.hint_help(note, help, title=f"{label}: the three numbers")
        elif note:
            widgets.hint(note)

    # ------------------------------------------------------- view: surfaces
    def _render_surfaces():
        box = ctx.views[("controls", "surfaces")]
        # ...AND THIS ONE TOO, even though nothing on it reads the deck.
        #
        # Entering stage 5 opens its FIRST tab, which is this one. With the
        # build left to the three tabs that quote a derivative, opening the
        # stage on a finished run built nothing: stage 5 stayed "ready"
        # rather than "done", and stage 6 — which is locked until stage 5
        # has produced a deck — stayed LOCKED until the user happened to
        # click Fin & rudder. From the outside that is "I clicked Controls,
        # the stages were grey, and then flight and control became
        # available", with the second half arriving for no reason the user
        # can see.
        #
        # Entering the stage is what makes the deck, so every one of its
        # views asks for it.
        _ensure_deck()
        box.clear()
        with box:
            if not _report():
                widgets.hint("No design to cut control surfaces out of yet. "
                             "Stage 3 sizes the wing; stage 4 re-evaluates "
                             "it; this stage hinges it.", "warn")
                return
            with widgets.group_box("Ailerons"):
                with ui.row().classes("w-full items-center gap-1 no-wrap"):
                    ui.switch("fit ailerons", value=C["aileron"]["on"],
                              on_change=lambda e: edit(("aileron", "on"),
                                                       bool(e.value)))
                    widgets.help_dot(
                        "Off removes the aileron column from the deck "
                        "entirely, so stage 6 offers no roll stick and the "
                        "roll manoeuvres are not offered either — a script "
                        "written into a missing axis flies nothing.\n\n"
                        "The deflection is antisymmetric: right down, left "
                        "up. On this lattice a positive right-aileron-down "
                        "gives Cl_da < 0, and the yaw it makes is PROVERSE "
                        "(Cn_da = -0.019 against Cl_da = -0.288) — a "
                        "one-chordwise-panel lattice cannot produce adverse "
                        "yaw, and dynamics.py says why in one place.",
                        title="Fitting ailerons")
                _band("Ailerons", "aileron", enabled=bool(
                          C["aileron"]["on"]),
                      note="Outboard band, deflected antisymmetrically.",
                      help=
                      "The two numbers are fractions of the SEMI-SPAN the "
                      "search produced, measured from the root. Reaching "
                      "further inboard buys roll power.\n\n"
                      "The hinge chord fraction sets how much of a "
                      "whole-surface incidence change the deflection is "
                      "worth — thin-aerofoil tau, the same curve the "
                      "stabiliser's elevator and the fin's rudder "
                      "are read on.")
            with widgets.group_box("Elevator"):
                ui.switch("hinged elevator (off = all-moving stabiliser)",
                          value=C["elevator"]["on"],
                          on_change=lambda e: edit(("elevator", "on"),
                                                   bool(e.value)))
                num(
                    "hinge chord (frac)", C["elevator"]["chord_frac"],
                    lambda e: edit_num(("elevator", "chord_frac"), e),
                    step=0.05, enabled=bool(C["elevator"]["on"]),
                    note="fraction of the stabiliser chord",
                    help="At a hinge chord of 1.0 this IS the all-moving "
                         "stabiliser the solver already trims with: the two "
                         "are the same boundary condition, and a test pins "
                         "them equal to rel 1e-12.\n\n"
                         "Below 1.0 the deflection is worth tau times a "
                         "whole-surface incidence change, and the pitch "
                         "power Cm_de on the Derivatives tab follows it.",
                    help_title="What a hinge chord of 1.0 means")

    # --------------------------------------------------- sizing the fin
    #: the height band the sizing search works in, as a fraction of span.
    #: Wide on purpose: it is the range a SEARCH covers, not a range the
    #: user is held to — a height typed into the field is accepted whatever
    #: this says.
    FIN_BAND = (0.01, 0.35)
    #: full rudder, for the sideslip the fin can be held at. Read off the
    #: STICK's own stop rather than restated, because stage 6 is what
    #: actually applies it and two numbers called "full rudder" that
    #: disagree is a lie in whichever place is not flown.
    RUDDER_MAX_DEG = float(stk.LIMITS["rudder"])

    #: the sideslip full rudder is asked to HOLD, when nobody has said [deg].
    #:
    #: A STATED DEFAULT, and it is not a measurement: nothing in this
    #: package derives a crosswind requirement, and no report carries one.
    #: What makes it answerable is beside it on the panel — the crosswind
    #: component it corresponds to at THIS design's own speed, V.tan(beta) —
    #: so the number can be replaced with one that means something for the
    #: aircraft. Nothing refuses any value, including zero.
    #:
    #: READ off the session's own defaults rather than written twice: the
    #: field is seeded from ``CONTROLS_DEFAULTS`` and falls back to this
    #: only once somebody has cleared the box, so two literals here would
    #: have disagreed exactly where nobody looks.
    RUDDER_BETA_DEFAULT = float(
        v4s.CONTROLS_DEFAULTS["vertical"]["rudder_beta_deg"])

    #: how many real rebuilds the rudder recommendation costs. The yaw power
    #: is very nearly proportional to the thin-aerofoil effectiveness tau —
    #: measured on the tail design, 0.7 % at a hinge fraction of 0.3 and
    #: 2.8 % at 0.9 — so the closed-form inverse of tau is a good first
    #: guess and NOT an answer. Each pass re-solves for tau from a MEASURED
    #: beta, which lands inside 0.05 % by the second: 5.14 -> 5.0005 deg for
    #: a 5 deg target. Three, at 2.3 ms each, so the number on the panel is
    #: one the lattice actually produced.
    RUDDER_PASSES = 3

    def _fin_geometry() -> dict | None:
        """THE FIN THAT WAS BUILT, not the three fields that describe it.

        Every one of height, chord and station may be blank, and blank means
        a default computed inside :func:`aerobo.flightmodel.build_flight_model`
        — since V5 the DESIGN'S own fin where the report states one. Reading
        the fields would therefore report ``None`` for the fin the lattice
        actually flew, which is the whole "stated is not flown" mistake in
        one panel. So this reads the ``VerticalSurface`` off the model.
        """
        fm, D = C.get("fm"), C.get("deck")
        v = getattr(getattr(fm, "model", None), "vertical", None)
        w = getattr(getattr(fm, "model", None), "wing", None)
        if v is None or w is None or D is None:
            return None
        h, c = abs(float(v.height)), float(v.chord)
        S_v = h * c
        arm = float(v.x) - float(D.x_cg)
        S_w, b_w = float(w.S), float(w.b)
        denom = S_w * b_w
        return {"h": h, "c": c, "S_v": S_v, "x": float(v.x),
                # the QUARTER CHORD is the arm; the LEADING EDGE is what the
                # field asks for and what the side elevation is drawn in
                "x_le": float(v.x) - QUARTER * c, "arm": arm,
                "AR": (h / c) if c > 1e-12 else 0.0,
                "S_w": S_w, "b_w": b_w,
                "V_v": (S_v * arm / denom) if denom > 1e-12 else 0.0,
                "ventral": float(v.height) < 0.0}

    def _uncached_fin_defaults() -> dict | None:
        """The fin the model builds when all three fields are BLANK.

        ASKED, not restated. What the three defaults ARE lives inside
        :func:`aerobo.flightmodel.build_flight_model` — and since V5 it is
        the DESIGN'S fin (``geometry["fin"]``, sized by volume coefficient,
        the same surface the drag book charges) wherever the report carries
        one, falling back to the old span-and-mac rule where it does not. A
        copy of either here would have gone stale on exactly that change. So
        this builds a model with the three fields cleared and reads the
        surface back.

        One extra lattice (2.3 ms), and only when at least one field is
        PINNED: with nothing pinned the fin as built already IS the default,
        and :func:`_fin_geometry` has it for nothing.
        """
        from dataclasses import replace as _replace

        from aerobo.flightmodel import build_flight_model
        rep = _report()
        if rep is None:
            return None
        V, rho = _air()
        try:
            fm = build_flight_model(
                rep, _replace(_spec(), fin_height_m=None, fin_chord_m=None,
                              fin_x_le_m=None), V=V, rho=rho)
        except Exception:                              # noqa: BLE001
            return None
        v = getattr(fm.model, "vertical", None)
        if v is None:
            return None
        c = float(v.chord)
        return {"h": abs(float(v.height)), "c": c,
                "x_le": float(v.x) - QUARTER * c}

    def _deck_at(height_m: float):
        """The deck this design would have with a fin of that height.

        A real rebuild, 2.8 ms, rather than a scaling law: Cn_beta is close
        to linear in fin AREA but Cl_r and Cn_r are not, and the spiral
        criterion is a product of all four.
        """
        from dataclasses import replace as _replace

        from aerobo.flightmodel import build_flight_model
        V, rho = _air()
        return build_flight_model(
            _report(), _replace(_spec(), fin_height_m=float(height_m)),
            V=V, rho=rho).deck

    def _deck_at_fin(height_m: float, chord_m: float):
        """The deck with a fin of that height AND that chord — a real build.

        The chord matters on its own and not only through the area: at fixed
        fin VOLUME a fatter chord is a lower aspect ratio, and a low-aspect
        fin makes markedly less side force per unit area. Measured on the
        tail design, holding V_v at 0.0434 while the chord goes 0.664 ->
        1.000 m takes the aspect ratio 1.81 -> 0.80 and ``Cn_beta`` +0.13696
        -> +0.08016. So the Reynolds recommendation below prices itself with
        a build rather than asserting that equal volume is equal stiffness.
        """
        from dataclasses import replace as _replace

        from aerobo.flightmodel import build_flight_model
        V, rho = _air()
        return build_flight_model(
            _report(), _replace(_spec(), fin_height_m=float(height_m),
                                fin_chord_m=float(chord_m)),
            V=V, rho=rho).deck

    def _re_reference() -> tuple[float, str] | None:
        """The Reynolds number this design's SECTION DATA is at, and where
        that number came from.

        The fin does not have a section in the deck at all: the lattice gives
        it a lift-curve slope with no Reynolds number anywhere in it, so
        there is nothing here to compare against on its own terms. What the
        design does have is the point its aerofoil data was screened at —
        stage 2's own Reynolds number where a section was chosen, and the
        wing's Re at MAC otherwise, which is what stage 2 screens at by
        default.
        """
        try:
            flown = v3s.section_flown_re(S, "main")
        except Exception:                              # noqa: BLE001
            flown = None
        if flown:
            return float(flown), "the section's own table"
        re_mac = (_point() or {}).get("re_mac")
        if re_mac:
            return float(re_mac), "the wing at its MAC"
        return None

    def _uncached_re_recommendation() -> dict | None:
        """What the fin's Reynolds number IS, what it would take to match the
        design's own, and what that costs.

        Returns ``needed=False`` when the fin chord is already at or above
        the reference: there is nothing to recommend, and shrinking a fin to
        bring its Reynolds number DOWN is not a thing anybody wants.
        """
        g, D = _fin_geometry(), C.get("deck")
        ref = _re_reference()
        if g is None or D is None or ref is None:
            return None
        re_ref, ref_label = ref
        re_now = _reynolds(g["c"])
        if re_now is None:
            return None
        out = {"re_now": re_now, "re_rudder": re_now * v_rudder_frac(),
               "re_ref": re_ref, "ref_label": ref_label,
               "c_now": g["c"], "h_now": g["h"], "AR_now": g["AR"],
               "V_v": g["V_v"], "cnb_now": float(D.Cn_beta),
               "needed": re_now < re_ref}
        if not out["needed"]:
            return out
        # the chord that puts the fin AT the reference, and the height that
        # holds the fin volume while it gets there (V_v is linear in h.c)
        c_rec = g["c"] * re_ref / re_now
        h_rec = g["h"] * g["c"] / c_rec
        out["c_rec"], out["h_rec"] = c_rec, h_rec
        out["AR_rec"] = h_rec / c_rec if c_rec > 1e-12 else 0.0
        try:
            out["cnb_rec"] = float(_deck_at_fin(h_rec, c_rec).Cn_beta)
        except Exception:                              # noqa: BLE001
            out["cnb_rec"] = None
        return out

    # -------------------------------------------------- sizing the rudder
    def _rudder_target_deg() -> float:
        """The sideslip full rudder is asked to hold [deg]."""
        v = C["vertical"].get("rudder_beta_deg")
        if v in (None, ""):
            return RUDDER_BETA_DEFAULT
        return abs(float(v))

    def _beta_held(D) -> float | None:
        """The steady sideslip full rudder can hold [deg], off a DECK.

        ``Cn_dr.d_max / Cn_beta``. The fin is in both: growing it buys yaw
        stiffness and spends the authority to sideslip against it, which is
        why this is the number a rudder is sized by and the fin area is not.

        ``None`` WHERE THE AEROPLANE DOES NOT WEATHERCOCK. The quotient was
        taken on ``abs(Cn_beta)``, so a design whose fin stands ahead of the
        CG — Cn_beta negative, the aeroplane yawing AWAY from the airflow —
        was told it holds a tidy 12.4 deg of sideslip, and the block below
        then offered it a SMALLER rudder. There is no steady sideslip to
        hold at all: the equilibrium is unstable, and the number is the
        distance to a balance the aeroplane runs away from. The Stability
        check tab was calling the same design directionally FAILED on the
        same deck. One predicate answers it — :func:`dynamics.weathercocks`
        — the same one the spiral scan already refuses on.
        """
        from aerobo.dynamics import weathercocks

        col = (getattr(D, "columns", None) or {}).get("rudder")
        if col is None or not weathercocks(float(D.Cn_beta)):
            return None
        return abs(float(col["Cn"])) * RUDDER_MAX_DEG / float(D.Cn_beta)

    def _deck_at_rudder(frac: float):
        """The deck with that rudder hinge fraction — a real rebuild."""
        from dataclasses import replace as _replace

        from aerobo.flightmodel import build_flight_model
        V, rho = _air()
        return build_flight_model(
            _report(), _replace(_spec(), rudder_chord_frac=float(frac)),
            V=V, rho=rho).deck

    def _tau_inverse(t: float) -> float:
        """The hinge chord fraction whose thin-aerofoil tau is ``t``.

        ``flap_effectiveness`` is strictly increasing from 0 at 0 to 1 at 1,
        so a bisection on the CLOSED FORM inverts it exactly and costs no
        lattice at all.
        """
        from aerobo.tail import flap_effectiveness as _tau
        lo, hi = 1e-6, 1.0
        if t >= _tau(1.0):
            return 1.0
        if t <= _tau(lo):
            return lo
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if _tau(mid) < t:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def _uncached_rudder_recommendation() -> dict | None:
        """THE RUDDER, AS A REQUIREMENT: the hinge chord that holds a stated
        sideslip at full deflection.

        Two-sided on purpose. Under the target it says grow, over it it says
        what the SMALLEST rudder that still meets it would be — and warns
        that shrinking one takes its Reynolds number down with it, which the
        block above will not recommend on its own. Neither is applied, and
        neither refuses a fraction typed into the field.

        Out of reach is reported as out of reach: at a hinge fraction of 1.0
        the "rudder" IS the whole fin, all-moving, and no rudder holds more
        sideslip than that. Clamping to the edge of the search would be a
        recommendation that lies.
        """
        from aerobo.tail import flap_effectiveness as _tau

        D, g = C.get("deck"), _fin_geometry()
        if D is None or g is None:
            return None
        f0 = v_rudder_frac()
        b0 = _beta_held(D)
        if b0 is None or not b0 > 0.0:
            return None
        target = _rudder_target_deg()
        V, _rho = _air()
        out = {"f_now": f0, "beta_now": b0, "target": target,
               "cross_ms": V * np.tan(np.deg2rad(target)), "V": V,
               "cn_now": float(D.columns["rudder"]["Cn"]),
               "d_max": RUDDER_MAX_DEG,
               "meets": b0 >= target}
        # the ceiling FIRST, so "unreachable" is a measurement and not a
        # failed search
        try:
            D_max = _deck_at_rudder(1.0)
        except Exception:                              # noqa: BLE001
            return out
        b_max = _beta_held(D_max)
        out["beta_max"] = b_max
        if b_max is None or target > b_max:
            out["reachable"] = False
            return out
        out["reachable"] = True
        f, b, D_f = f0, b0, D
        for _ in range(RUDDER_PASSES):
            t_req = _tau(f) * target / b
            f = _tau_inverse(t_req)
            try:
                D_f = _deck_at_rudder(f)
            except Exception:                          # noqa: BLE001
                return out
            b_new = _beta_held(D_f)
            if b_new is None or not b_new > 0.0:
                return out
            b = b_new
        out["f_rec"], out["beta_rec"] = f, b
        out["cn_rec"] = float(D_f.columns["rudder"]["Cn"])
        out["re_now"] = _reynolds(g["c"] * f0)
        out["re_rec"] = _reynolds(g["c"] * f)
        return out

    def _apply_rudder():
        rec = _rudder_recommendation()
        if rec is None or rec.get("f_rec") is None:
            return
        edit(("vertical", "rudder_chord_frac"), float(rec["f_rec"]))

    def _apply_re():
        """Take the recommendation: both fields at once, one rebuild.

        Both, because the chord alone would silently move the fin volume the
        user sized for — a Reynolds fix that changes the thing you chose the
        fin by is a trap, not a recommendation.
        """
        rec = _re_recommendation()
        if rec is None or not rec.get("needed"):
            return
        C["vertical"]["chord_m"] = float(rec["c_rec"])
        C["vertical"]["height_m"] = float(rec["h_rec"])
        C["spiral_scan"] = None
        C["dihedral_scan"] = None
        rebuild()
        ctx.render("controls", S["ui"]["tab"]["controls"])
        ctx.refresh(*V3_STAGES, "controls")

    def _solve_height(target: float, read) -> float | None:
        """Bisect the fin height until ``read(deck)`` reaches ``target``.

        ``read`` is monotone increasing in height for both things this is
        used for (yaw stiffness, and fin area through it), so a bisection is
        enough and no derivative has to be trusted. Returns None when the
        target lies outside the band, which is an answer — "no fin between
        1 % and 35 % of the span gets there" — and not a failure.
        """
        g = _fin_geometry()
        if g is None:
            return None
        lo, hi = FIN_BAND[0] * g["b_w"], FIN_BAND[1] * g["b_w"]
        try:
            f_lo, f_hi = read(_deck_at(lo)), read(_deck_at(hi))
        except Exception:                              # noqa: BLE001
            return None
        if not (min(f_lo, f_hi) <= target <= max(f_lo, f_hi)):
            return None
        for _ in range(24):
            mid = 0.5 * (lo + hi)
            try:
                f_mid = read(_deck_at(mid))
            except Exception:                          # noqa: BLE001
                return None
            if (f_mid < target) == (f_lo < target):
                lo, f_lo = mid, f_mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def _apply_height(h):
        if h is None:
            return
        C["size_note"] = None
        edit(("vertical", "height_m"), float(h))

    def _refuse_size(key, text):
        """Say why a sizing button did nothing, where it was pressed.

        Both buttons could return silently — an out-of-band target, a fin
        ahead of the CG — leaving the panel byte-identical to before the
        click. From the outside that is a broken button, and the panel's own
        help promises the opposite ("reported as OUT OF REACH rather than
        clamped"). The note is cleared by the next rebuild, so it belongs to
        the design it was measured on.
        """
        C["size_note"] = (key, text)
        ctx.render("controls", S["ui"]["tab"]["controls"])

    def _size_to_Vv():
        """Height from a target fin volume coefficient — CLOSED FORM.

        V_v = S_v.l_v/(S.b) and S_v = h.c with the chord and the arm both
        fixed, so the height is one division. No search, and no rebuild
        needed to find it — only to show what it bought.
        """
        g, tgt = _fin_geometry(), C.get("size_Vv")
        if g is None or tgt in (None, ""):
            return
        # THE ARM HAS A SIGN, and it decides whether the question has an
        # answer. l_v is measured from the CG, so a fin AHEAD of it makes
        # the volume coefficient negative: the division then returned a
        # negative height, which the model reads as a VENTRAL fin — a 4.92 m
        # surface hung below a design whose ventral switch still read off,
        # achieving minus the target.
        if g["arm"] <= 0.0 or g["c"] <= 0.0:
            _refuse_size(
                "Vv", "the fin's quarter chord stands AHEAD of the CG "
                      f"(arm {g['arm']:+.3f} m), so there is no positive "
                      "fin volume to solve for. Move the leading edge aft "
                      "first.")
            return
        h = abs(float(tgt)) * g["S_w"] * g["b_w"] / (g["c"] * g["arm"])
        # ...and the same band its sibling searches in, for the same reason:
        # a height is a recommendation, and 132 % of the span is not one.
        lo, hi = FIN_BAND[0] * g["b_w"], FIN_BAND[1] * g["b_w"]
        if not (lo <= h <= hi):
            _refuse_size(
                "Vv", f"V_v = {abs(float(tgt)):.4f} needs a fin {h:.3f} m "
                      f"tall on this chord and arm — outside "
                      f"{FIN_BAND[0]:.0%}-{FIN_BAND[1]:.0%} of the span "
                      f"({lo:.3f}-{hi:.3f} m). Nothing is clamped: the "
                      f"answer is a different chord or a longer arm.")
            return
        _apply_height(h)

    def _size_to_Cnb():
        tgt = C.get("size_Cnb")
        if tgt in (None, ""):
            return
        h = _solve_height(float(tgt), lambda d: float(d.Cn_beta))
        if h is None:
            g = _fin_geometry()
            band = ("" if g is None else
                    f" ({FIN_BAND[0] * g['b_w']:.3f}-"
                    f"{FIN_BAND[1] * g['b_w']:.3f} m)")
            _refuse_size(
                "Cnb", f"no fin between {FIN_BAND[0]:.0%} and "
                       f"{FIN_BAND[1]:.0%} of the span{band} reaches "
                       f"Cn_beta {float(tgt):+.5f}. That is the answer, not "
                       f"a failed search — nothing is clamped to the edge "
                       f"of the band.")
            return
        _apply_height(h)

    def _flown_dihedral_deg() -> float:
        """The dihedral THE LATTICE FLEW, off the built wing.

        Not the report's number and not a field: since the cant became a
        design variable a wing may carry one because the SEARCH chose it,
        and the advice above is opposite in the two cases — "give it a
        dihedral" is wrong-headed on a wing that already has 9 degrees.
        """
        w = getattr(getattr(C.get("fm"), "model", None), "wing", None)
        return float(getattr(w, "dihedral_deg", 0.0) or 0.0)

    def _scan_spiral():
        """Does ANY fin size in the band make the spiral converge?

        Asked rather than assumed, because the intuitive answer is wrong on
        at least one design in this package. The spiral criterion is
        ``Cl_beta.Cn_r - Cn_beta.Cl_r`` and shrinking the fin lowers the
        yaw stiffness — which looks like the fix — but it lowers the
        dihedral effect and the yaw damping with it, because on a lattice
        wing with no dihedral the fin is where ALL of Cl_beta comes from.
        Ten builds, 28 ms, and the answer is a measurement.
        """
        g = _fin_geometry()
        if g is None:
            C["spiral_scan"] = None
            return
        rows = []
        for frac in (0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.16,
                     0.22, 0.30):
            h = frac * g["b_w"]
            try:
                d = _deck_at(h)
            except Exception:                          # noqa: BLE001
                continue
            m = md.spiral_margin_of(d)
            rows.append((frac, h, float(m), float(d.Cn_beta)))
        # A FIN IS ONLY AN ANSWER WHILE THE AEROPLANE STILL WEATHERCOCKS.
        # The criterion is Cl_beta.Cn_r - Cn_beta.Cl_r, so driving Cn_beta
        # NEGATIVE flips the second term and "converges" the spiral on
        # paper: measured on the box-centre `tail + winglet`, a fin at 1 %
        # of span scores +0.000566 with Cn_beta -0.00707 — an aeroplane
        # that yaws away from the airflow. That was the recommendation this
        # panel made, and it is worse than no recommendation, so the
        # candidates are filtered to the directionally stable ones and the
        # rejected ones are REPORTED rather than dropped.
        stable = [r for r in rows if r[3] > 0.0]
        best = max(stable, key=lambda r: r[2]) if stable else None
        unstable = [r for r in rows if r[3] <= 0.0 and r[2] > 0.0]
        # ...AND IF NO FIN DOES, THE WING'S ANSWER IN THE SAME PRESS. The
        # panel's verdict is "the spiral is a WING answer here", and making
        # the user find a second button for the number that verdict points
        # at is the recommendation-you-have-to-press-for again.
        if best is None or best[2] <= 0.0:
            _scan_dihedral()
        C["spiral_scan"] = {"rows": rows, "best": best,
                            "unstable_wins": unstable,
                            # ...and the fallback the "No" branch quotes: the
                            # least-bad STABLE fin, so the sentence is about
                            # a fin somebody could actually fit
                            "least_bad": (max(stable, key=lambda r: r[2])
                                          if stable else None)}

    def _cant_answer_route() -> str:
        """HOW this configuration can be given a dihedral, as one clause.

        Both places that advise a dihedral read this, because the ACTION is
        different in each of four states and naming the wrong one is the
        same failure in both: a user sent to a control that is not there
        concludes the shell is broken.

        * nothing can answer — no card is drawn at all, and this says so
          rather than pointing at one (``v3s.cant_is_answerable``);
        * the row is already SEARCHED — the card is on 'optimise …' and the
          action is not a switch, it is a WEIGHT. Nothing else in the
          objective prices the sign of a dihedral, so an L/D-only search
          settles it on the anhedral bound;
        * the family STATES a cant (``v3s.cant_is_stated``) — a field to
          type the measured number into, or the switch to searching it;
        * it does NEITHER, which is the tandem pair: its lifting-line
          solver flies no stated cant, so there is no field on the card and
          the only answer is to switch the question, which moves the run
          onto the nonplanar twin that carries the row.

        The card is NAMED in all three answerable states, which is the
        contract stage 3 is held to from the other side.
        """
        from aerobo import api as _api

        name = S["wing"]["problem"]
        card = "stage 3 (Wing type ▸ Wing cant and sweep)"
        if not v3s.cant_is_answerable(name, S["wing"]["choices"]):
            return ("This family has no field to give it one: stage 3 asks "
                    "the cant only of the families whose solver has "
                    "out-of-plane geometry. The number is what one of those "
                    "would need.")
        buy = ("weight 'spiral divergence' in the objective so the search "
               "buys the convergent one")
        if _api.dihedral_is_searched(name):
            return (f"S{card[1:]} already carries the row as a design "
                    f"variable, "
                    f"so the action is not a switch: {buy}. Nothing else in "
                    f"that objective prices the SIGN of a dihedral, and on "
                    f"L/D alone the row settles on the anhedral bound.")
        if v3s.cant_is_stated(name):
            return (f"Give it one in {card}: state a value, or switch that "
                    f"question to 'optimise them' and {buy}.")
        return (f"Give it one in {card}. This family flies no STATED cant, "
                f"so there is no field to type it into — switch that "
                f"question to 'optimise the dihedral', which moves the run "
                f"onto the twin that carries the row, and {buy}.")

    def _scan_dihedral():
        """AND HOW MUCH WING DIHEDRAL DOES? The other half of the answer.

        The fin scan above could already say a divergent spiral is a WING
        answer here and not a fin one. It could not say HOW MUCH wing: the
        advice was "give it one in stage 3", and stage 3 offered a 5.7 deg
        step that was one family's number, measured on `tail [free cant]`. On the shell's own default wing+tail family
        the crossing is 6.13 deg, so that step left the 6-DOF spiral
        eigenvalue at +0.0026 and the aeroplane still rolled off.

        Twenty-odd lattice rebuilds of THIS design at successive cants
        (``flightmodel.dihedral_for_spiral``), scored by the same
        ``spiral_margin_of`` printed above it. It changes the cant the
        rebuild FLIES; it does not re-search or re-price the design, exactly
        as the fin scan beside it does not.
        """
        rep = _report()
        if not rep:
            C["dihedral_scan"] = None
            return
        try:
            from aerobo.flightmodel import dihedral_for_spiral

            V, rho = _air()
            fix = dihedral_for_spiral(rep, _spec(), V=V, rho=rho)
        except Exception as exc:                       # noqa: BLE001
            C["dihedral_scan"] = {"status": "error", "error": str(exc)}
            return
        C["dihedral_scan"] = {
            "status": fix.status, "gamma_deg": fix.gamma_deg,
            "margin_now": fix.margin_now, "margin_at": fix.margin_at,
            "flown_deg": fix.flown_deg, "Cn_beta": fix.Cn_beta,
            "band": list(fix.band), "extra_deg": fix.extra_deg}

    def _render_dihedral_scan():
        """What the scan found, in the same box as the fin's answer."""
        s = C.get("dihedral_scan")
        if not s:
            return
        st = s.get("status")
        if st == "error":
            widgets.hint(f"the dihedral could not be measured: "
                         f"{s.get('error')}", "warn")
            return
        flown = float(s.get("flown_deg") or 0.0)
        if st == "refused":
            widgets.hint(
                f"Not answered: Cn_beta is {s.get('Cn_beta', 0.0):+.5f}, so "
                f"this aeroplane yaws AWAY from the airflow and a positive "
                f"spiral margin on it would be arithmetic. The fin is the "
                f"question, not the wing.", "warn")
            return
        if st == "converges":
            widgets.hint(
                f"Its spiral already converges at the {flown:+.3g} deg of "
                f"dihedral it flies (margin "
                f"{s.get('margin_now', 0.0):+.6f}).")
            return
        if st == "out_of_reach":
            band = s.get("band") or [0.0, 15.0]
            widgets.hint(
                f"No dihedral up to {band[1]:g} deg converges this design's "
                f"spiral — measured, not assumed. Neither surface on this "
                f"aeroplane is the lever.", "warn")
            return
        gam = float(s.get("gamma_deg") or 0.0)
        extra = s.get("extra_deg")
        more = ("" if not extra or abs(extra) < 1e-9 else
                f" ({extra:+.2f} deg more than the {flown:.2f} deg it "
                f"flies)")
        where = (_cant_answer_route()
                 + (" — this stage flies the design, it does not reshape "
                    "it." if v3s.cant_is_answerable(
                        S["wing"]["problem"], S["wing"]["choices"]) else ""))
        widgets.hint(
            f"{gam:.2f} deg of wing dihedral does{more}: the margin goes "
            f"{s.get('margin_now', 0.0):+.6f} to "
            f"{s.get('margin_at', 0.0):+.6f}, and it is the SMALLEST cant "
            f"that turns it. " + where)

    # ---- the three DERIVED answers, memoised against the deck they describe
    def _fin_defaults():
        return _cached("fin_defaults", _uncached_fin_defaults)

    def _re_recommendation():
        return _cached("re_rec", _uncached_re_recommendation)

    def _rudder_recommendation():
        # the TARGET is not in the spec, so it is not in the deck, so it has
        # to be in the key
        return _cached("rudder_rec", _uncached_rudder_recommendation,
                       _rudder_target_deg())

    # ------------------------------------------------------- the layout
    def _layout() -> dict | None:
        """Every x station in the design, off the LATTICE, or None.

        The stations are what makes the fin's own station answerable: "put
        the leading edge at 4.5 m" is a sentence with no meaning until the
        wing is known to be at 0 and the stabiliser at 4.6.
        """
        fm = C.get("fm")
        D = C.get("deck")
        m = getattr(fm, "model", None)
        if m is None or D is None:
            return None
        try:
            x_np = float(m.neutral_point())
        except Exception:                              # noqa: BLE001
            x_np = None
        return sv.stations(m, x_cg=float(D.x_cg), x_np=x_np)

    def _render_layout():
        """THE SIDE ELEVATION, and the same stations as numbers.

        Both, deliberately. The picture answers "is the fin near the tail?"
        at a glance and cannot be typed into; the table answers "what do I
        type?" and cannot be read at a glance.
        """
        st = _layout()
        if st is None:
            return
        with widgets.group_box("Where everything is (side elevation, "
                               "x AFT from the wing's quarter chord)"):
            ui.html(sv.svg(st)).classes("w-full")
            for key, label, _role, _vert in sv.GROUPS:
                b = st["boxes"].get(key)
                if b is None:
                    continue
                widgets.kv(f"{label} x", f"LE {b['x_le']:+.3f}   "
                                         f"c/4 {b['x_qc']:+.3f}   "
                                         f"TE {b['x_te']:+.3f} m")
            widgets.kv("CG", f"{st['x_cg']:+.3f} m", color=theme.WARN)
            if st.get("x_np") is not None:
                widgets.kv("neutral point", f"{st['x_np']:+.3f} m",
                           color=theme.ACCENT)
            widgets.hint_help(
                "x = 0 is the wing's quarter chord; x runs AFT.",
                "So a station bigger than the stabiliser's puts the fin "
                "behind the tail.\n\n"
                "Every number here is measured off the panels the LATTICE "
                "built, not off the fields below. A blank chord is a "
                "default the model owns, so drawing the fields would draw a "
                "fin that is not there.",
                title="Reading the side elevation")

    # ------------------------------------------------------- view: vertical
    def _render_vertical():
        box = ctx.views[("controls", "vertical")]
        # the sizing block below reads the fin off the BUILT model, so this
        # view needs a deck for the same reason Derivatives and Check do —
        # and for the same reason must not wait to be edited into existence
        _ensure_deck()
        box.clear()
        v = C["vertical"]
        with box:
            widgets.hint_help(
                "Without this surface Cn_beta is exactly zero.",
                "Not neutral — ABSENT. Before a vertical surface exists the "
                "design has no panel that carries side force at all, so "
                "every lateral number the deck reports is a zero with a "
                "reason rather than a measurement.\n\n"
                "Tip winglets do not fix it: measured at h/(b/2) = 0.12 and "
                "90 degrees of cant they give CY_beta -0.155, Cl_beta "
                "+0.066 and Cn_beta +0.0047 — they sit ahead of the CG and "
                "read DEstabilising.",
                title="Why a fin is a geometry gap, not a setting",
                kind="warn")
            _render_layout()
            with widgets.group_box("The vertical surface"):
                # THE DESIGN CAN VETO THIS SWITCH, the way a V-tail already
                # does. Stage 1 asks "add a vertical stabiliser" and the run
                # honours it — no fin is built, charged, weighed or reported
                # — so a switch here that says "fit one" would be the same
                # question answered twice in two directions, and this one
                # cannot win: ``flightmodel.build_flight_model`` refuses to
                # invent a surface the report states the design does not
                # have (``fin.states_no_fin``).
                none_by_design = _fin.states_no_fin(
                    (_report() or {}).get("geometry"))
                sw = ui.switch("fit a vertical surface", value=v["on"]
                               and not none_by_design,
                               on_change=lambda e: edit(("vertical", "on"),
                                                        bool(e.value)))
                if none_by_design:
                    sw.disable()
                    sw.tooltip("this design carries no fin — stage 1")
                    widgets.hint_help(
                        "No fin on this design — you said so in stage 1.",
                        "The run neither charged its drag nor weighed it, "
                        "so fitting one here would fly an aeroplane the "
                        "search never scored.\n\n"
                        "The answer stays where it was given: turn the "
                        "vertical stabiliser back on in stage 1 and run the "
                        "design again.",
                        title="Why this switch is disabled", kind="warn")
                    return
                with ui.row().classes("w-full items-center gap-1 no-wrap"):
                    ui.switch("ventral (hangs below)", value=v["ventral"],
                              on_change=lambda e: edit(("vertical",
                                                        "ventral"),
                                                       bool(e.value)))
                    widgets.help_dot(
                        "Above or below changes the ROLL sign, not the yaw "
                        "one. Both make Cn_beta — the arm is the same — but "
                        "a fin ABOVE the CG rolls the craft away from a "
                        "sideslip (stable Cl_beta) and one below rolls it "
                        "into the sideslip.\n\n"
                        "So a ventral fin buys directional stiffness and "
                        "spends dihedral effect, which is the pair the "
                        "spiral mode is made of.",
                        title="What hanging it below changes")
                _render_fin_fields()
            _render_sizing()

    #: (session key, label, key in the geometry dicts). One table, so a
    #: fourth fin field cannot be added with a different idea of what blank
    #: means. WHAT BLANK MEANS is NOT in here: it depends on whether the
    #: report states a fin (V5's ``geometry["fin"]``, sized by volume
    #: coefficient — the same surface the drag book charges) or predates that
    #: block, and a restatement here would have gone stale the moment the
    #: default moved. :func:`_fin_blank_rules` reads it off the design.
    FIN_FIELDS = (("height_m", "height [m]", "h"),
                  ("chord_m", "chord [m]", "c"),
                  ("x_le_m", "leading edge at x [m]", "x_le"))

    def _fin_blank_rules() -> dict:
        """What "blank" means for each field, FROM THE DESIGN.

        A report that carries a fin block hands the rebuild a real surface;
        one that does not gets the old span-and-mac fallback. Both live in
        ``flightmodel.build_flight_model`` and neither is restated here — the
        report is asked which case it is in.
        """
        rep = _report() or {}
        blk = ((rep.get("geometry") or {}).get("fin")) or {}
        if blk:
            v_v = float(blk.get("V_v", 0.0))
            arm = float(blk.get("l_t_m", 0.0))
            return {"h": f"the design's fin (volume coefficient {v_v:g} "
                         f"against its {arm:.3g} m arm)",
                    "c": f"the design's fin at aspect ratio "
                         f"{float(blk.get('AR', 0.0)):g}",
                    "x_le": "the design's fin, a quarter chord ahead of "
                            "its arm"}
        return {"h": "12 % of the span", "c": "0.65 of the mean chord",
                "x_le": "the tail station"}

    def _set_dim(key, e):
        """A DIMENSION, or a blank. Never a zero.

        ``fields.is_pinned`` counts 0.0 as an answer, and
        ``build_flight_model`` tests the height and the chord for
        TRUTHINESS — so a typed 0 pinned the field, flipped its caption
        to "YOURS, not the model's", and was then dropped by the model,
        which went on flying its own 1.044 m fin. Stated is not flown,
        inside the panel written to stop exactly that.

        A fin of zero height is the switch above, so a zero is refused
        here with a reason rather than stored somewhere that will not
        honour it.
        """
        val = _number(e, blank=None)
        if val is not None and float(val) <= 0.0:
            # blank FIRST — ``edit`` rebuilds, and a rebuild throws the
            # note away with every other answer that belonged to the
            # previous deck — then say why, which repaints again.
            edit(("vertical", key), None)
            _refuse_size(
                "fin", "a fin dimension of zero or less is not a fin — "
                       "the switch above is how a design flies without "
                       "one. The field is back on the model's own "
                       "number.")
            return
        edit(("vertical", key), val)

    def _render_fin_fields():
        """THE THREE FIELDS, OPENING ON THE FIN THAT IS BEING FLOWN.

        They used to open EMPTY on a design that has a fin 1.200 m tall with
        a 0.664 m chord at x = 5.334 m, because blank means "the model
        chooses" and nothing put the model's choice in the box. The number
        existed, the lattice was flying it, and the one place a user looks
        for it said nothing.

        What is shown is read off the BUILT surface (:func:`_fin_geometry`),
        never recomputed from the rule — showing a restatement of the default
        is how the box and the aeroplane drift apart. Nothing is stored until
        somebody types, so a blank field goes on following the design.
        """
        v = C["vertical"]
        g = _fin_geometry()
        # the fallback the hint quotes, and it is bought only when it is
        # needed: with nothing pinned the fin AS BUILT already is the default
        dflt = (_fin_defaults()
                if any(fl.is_pinned(v[k]) for k, _l, _g in FIN_FIELDS)
                else g)
        rules = _fin_blank_rules()

        for key, label, gkey in FIN_FIELDS:
            why = rules[gkey]
            fl.defaulted(
                num, label,
                stored=v[key],
                flown=(None if g is None else g[gkey]),
                default=(None if dflt is None else dflt[gkey]),
                on_change=(lambda e, k=key: _set_dim(k, e)),
                unit="m", why=why, dp=4, width="w-32",
                note="blank = the model's own",
                help=("Blank means " + why + ".\n\n"
                      + ("This is the LEADING EDGE, not the quarter chord. "
                         "The side elevation above is drawn in the same "
                         "coordinate, so a number typed here lands where "
                         "you can see it."
                         if key == "x_le_m" else
                         "Type a number and the model flies yours; clear "
                         "the box and it goes back to following the "
                         "design.")),
                help_title=f"What blank means for the {label.split(' [')[0]}")
        note = C.get("size_note")
        if note and note[0] == "fin":
            widgets.hint(note[1], "warn")
        if g is None:
            widgets.hint(
                ("The vertical surface is switched off above, so there are "
                 "no dimensions to open on."
                 if C.get("deck") is not None else
                 "No fin has been built yet, so these three have nothing to "
                 "open on. They fill in as soon as the lattice runs."),
                "warn")

    def _render_sizing():
        """WHAT THE FIN AND THE RUDDER ACTUALLY ARE, and how to choose them.

        The three fields above say what the fin is ALLOWED to be; every one
        of them can be blank. This says what it IS — read off the surface the
        lattice built — and puts the two numbers a fin is normally chosen by
        beside it: the fin volume coefficient, and the yaw stiffness it
        buys. Both can be solved for, because "make it 12 % of the span" is
        a shape and "make V_v 0.03" is a requirement.
        """
        g = _fin_geometry()
        D = C.get("deck")
        if D is None:
            widgets.hint("No deck yet — the sizing is read off the surface "
                         "the lattice built, not off the fields above.",
                         "warn")
            return
        if g is None:
            # ...and this is NOT the same sentence. The deck is fine; there
            # is simply no vertical surface on it, because the switch above
            # is off. Saying "the lattice has not run" about a stage whose
            # Derivatives tab is printing a full set of rows sends the
            # reader looking for a fault that is not there.
            widgets.hint("The vertical surface is switched off above — "
                         "switch it on to size one. Everything else on this "
                         "design is built and the Derivatives tab has it.",
                         "warn")
            return
        with widgets.group_box("Fin, as built"):
            widgets.kv("height", f"{g['h']:.3f} m   "
                                 f"({g['h'] / g['b_w']:.1%} of span)"
                                 + ("   VENTRAL" if g["ventral"] else ""))
            widgets.kv("chord", f"{g['c']:.3f} m")
            widgets.kv("area S_v", f"{g['S_v']:.4f} m2   "
                                   f"({g['S_v'] / g['S_w']:.2%} of the wing)")
            widgets.kv("aspect ratio", f"{g['AR']:.2f}   (geometric — a fin "
                                       f"rooted on a body behaves like more)")
            widgets.kv("leading edge", f"{g['x_le']:.3f} m")
            widgets.kv("arm l_v", f"{g['arm']:.3f} m   (quarter chord at "
                                  f"{g['x']:.3f} m, CG at {D.x_cg:.3f} m)")
            widgets.kv("volume V_v", f"{g['V_v']:.4f}",
                       tip="S_v.l_v / (S.b)")
            widgets.kv("yaw stiffness", f"{D.Cn_beta:+.5f} /rad",
                       color=(theme.GOOD if D.Cn_beta > 0 else theme.BAD))

        rec = _re_recommendation()
        if rec is not None:
            with widgets.group_box("Reynolds number"):
                widgets.kv("fin chord Re", f"{rec['re_now']:.3g}",
                           tip="rho.V.c/mu at the mission point — the same "
                               "fluid state stage 1 states and stage 2 "
                               "screens at")
                widgets.kv("rudder chord Re", f"{rec['re_rudder']:.3g}",
                           tip="the rudder is a fraction of an already "
                               "small chord")
                widgets.kv("this design's own", f"{rec['re_ref']:.3g}   "
                                                 f"({rec['ref_label']})")
                if not rec["needed"]:
                    widgets.hint_help(
                        "Nothing to recommend — the fin is already there.",
                        "The fin runs at or above the Reynolds number this "
                        "design's own section data was taken at.\n\n"
                        "Nothing here would ever ask you to make it "
                        "SMALLER: a lower Reynolds number is not a design "
                        "goal.",
                        title="Why there is no recommendation")
                else:
                    cost = ""
                    if rec.get("cnb_rec") is not None and \
                            abs(rec["cnb_now"]) > 1e-9:
                        drop = rec["cnb_rec"] / rec["cnb_now"] - 1.0
                        cost = (f"   Cn_beta {rec['cnb_now']:+.5f} -> "
                                f"{rec['cnb_rec']:+.5f} ({drop:+.1%})")
                    widgets.kv("recommended chord",
                               f"{rec['c_rec']:.3f} m   (now "
                               f"{rec['c_now']:.3f} m)",
                               color=theme.WARN)
                    widgets.kv("...holding V_v",
                               f"height {rec['h_now']:.3f} -> "
                               f"{rec['h_rec']:.3f} m, aspect ratio "
                               f"{rec['AR_now']:.2f} -> {rec['AR_rec']:.2f}")
                    if cost:
                        widgets.kv("...and it COSTS", cost.strip(),
                                   color=theme.BAD)
                    ui.button("set the chord and hold V_v",
                              on_click=_apply_re) \
                        .props("dense flat no-caps")
                widgets.hint_help(
                    "A recommendation, not a number to trust.",
                    "The lattice gives the fin a lift-curve slope with no "
                    "Reynolds number in it at all, and the profile drag it "
                    "never sees is not a weak function of Re: this "
                    "package's own screen has hg40 at L/D 80.6 at Re 1e6, "
                    "42.2 at 3e5 and 19.5 at 1.5e5. So a fin whose chord "
                    "runs well below the point the section data was taken "
                    "at is one whose Cn_beta above is optimistic — and its "
                    "rudder is worse, because the rudder chord is a "
                    "fraction of the fin's.\n\n"
                    "The cost line is a real rebuild, not an assertion: fin "
                    "VOLUME is not a sufficient statistic for yaw "
                    "stiffness. Holding V_v while the chord grows lowers "
                    "the fin's aspect ratio, and a low-aspect fin makes "
                    "less side force per unit area.\n\n"
                    "Which of the two you want is a decision, so nothing "
                    "here is applied for you and nothing refuses a chord "
                    "you type.",
                    title="Why this is a recommendation", kind="warn")

        col = (D.columns or {}).get("rudder")
        with widgets.group_box("Rudder"):
            num(
                "hinge chord (frac)", v_rudder_frac(),
                lambda e: edit_num(("vertical", "rudder_chord_frac"), e),
                step=0.05,
                tip="of the FIN chord. The area below follows from it, and "
                    "so does the yaw power.")
            S_r = g["S_v"] * v_rudder_frac()
            widgets.kv("area S_r", f"{S_r:.4f} m2")
            denom = g["S_w"] * g["b_w"]
            widgets.kv("volume V_r",
                       f"{(S_r * g['arm'] / denom) if denom > 1e-12 else 0:.4f}")
            if col:
                widgets.kv("yaw power Cn_dr", f"{col['Cn']:+.5f} /rad")
                widgets.kv("side force CY_dr", f"{col['CY']:+.5f} /rad")
                # the number a rudder is really sized by: how much sideslip
                # full deflection can HOLD against the fin's own stiffness
                beta = _beta_held(D)
                if beta is not None:
                    widgets.kv("holds sideslip",
                               f"{beta:.1f} deg at {RUDDER_MAX_DEG:.0f} deg "
                               f"of rudder")
                    widgets.hint_help(
                        "Cn_dr·d_max / Cn_beta — the crosswind number.",
                        "The steady sideslip full rudder can hold against "
                        "the fin's own weathercock stiffness.\n\n"
                        "It is the reason a bigger fin is not automatically "
                        "a better one: the fin appears in the DENOMINATOR "
                        "as well as the numerator, so growing it buys yaw "
                        "stiffness and spends control authority.",
                        title="What 'holds sideslip' means")
                _render_rudder_rec()
            else:
                widgets.hint("No rudder column in the deck — the fin is "
                             "switched off above.", "warn")

        with widgets.group_box("Size it"):
            with ui.row().classes("items-end gap-2 no-wrap"):
                num(
                    "target V_v", C.get("size_Vv"),
                    lambda e: set_target("size_Vv", e),
                    step=0.005, width="w-24",
                    note="fin volume S_v·l_v/(S·b)")
                ui.button("set the height", on_click=_size_to_Vv) \
                    .props("dense flat no-caps")
            with ui.row().classes("items-end gap-2 no-wrap"):
                num(
                    "target Cn_beta", C.get("size_Cnb"),
                    lambda e: set_target("size_Cnb", e),
                    step=0.01, width="w-24",
                    note="yaw stiffness [/rad], positive is stable")
                ui.button("set the height", on_click=_size_to_Cnb) \
                    .props("dense flat no-caps")
            note = C.get("size_note")
            if note and note[0] in ("Vv", "Cnb"):
                widgets.hint(
                    ("target V_v: " if note[0] == "Vv"
                     else "target Cn_beta: ") + note[1], "warn")
            widgets.hint_help(
                "Both write the height above; chord and station stay.",
                "Solve the fin's HEIGHT for a target you state: a fin "
                "volume coefficient (a shape), or a yaw stiffness (a "
                "requirement). V_v is closed form; Cn_beta is a small "
                "search over rebuilt lattices.\n\n"
                f"A target outside {FIN_BAND[0]:.0%}-{FIN_BAND[1]:.0%} of "
                "the span is reported as OUT OF REACH rather than clamped "
                "to the edge of the search — a recommendation is not a "
                "limit, and a silent clamp is a recommendation that lies.",
                title="What 'set the height' does")

            m = md.spiral_margin_of(D)
            widgets.kv("spiral margin", f"{m:+.6f}   "
                                        f"({'converges' if m > 0 else 'DIVERGES'})",
                       color=(theme.GOOD if m > 0 else theme.BAD),
                       tip="Cl_beta.(Cn_r - t.Cn_p) - Cn_beta.(Cl_r - "
                           "t.Cl_p), t = tan(trim attitude)")
            # ONE BUTTON, BECAUSE IT IS ONE QUESTION. There were two —
            # "does any fin size fix the spiral?" and "how much wing
            # dihedral fixes it?" — and the first already runs the second
            # whenever no fin height scores a positive margin, which is
            # exactly the case in which the dihedral answer is wanted. So
            # the second button re-ran work the first had done and the pair
            # asked the reader to know which half of the answer they needed
            # before they had either.
            with ui.row().classes("items-center gap-2 no-wrap"):
                ui.button("what fixes the spiral?",
                          on_click=lambda: (_scan_spiral(),
                                            ctx.render("controls",
                                                       "vertical"))) \
                    .props("dense flat no-caps")
                widgets.help_dot(
                    "Ten rebuilt lattices over fin heights from "
                    f"{FIN_BAND[0]:.0%} to {FIN_BAND[1]:.0%} of the span, "
                    "asking whether any of them makes Cl_beta·Cn_r − "
                    "Cn_beta·Cl_r positive — and, when none does, a second "
                    "sweep over WING DIHEDRAL, which is the lever that "
                    "actually turns it.\n\n"
                    "About 28 ms. Nothing is applied: both halves report a "
                    "number for you to type.",
                    title="What this measures")
            _render_dihedral_scan()
            scan = C.get("spiral_scan")
            if scan and scan.get("unstable_wins"):
                frac, h, m_u, cnb_u = max(scan["unstable_wins"],
                                          key=lambda r: r[2])
                widgets.hint_help(
                    f"{len(scan['unstable_wins'])} heights 'pass' with "
                    f"Cn_beta negative — none offered.",
                    f"Best of them: margin {m_u:+.6f} at {h:.3f} m, with "
                    f"Cn_beta {cnb_u:+.5f}.\n\n"
                    "The criterion is Cl_beta·Cn_r − Cn_beta·Cl_r. Take the "
                    "yaw stiffness below zero and its sign flips without "
                    "anything having got better — the aeroplane yaws AWAY "
                    "from the airflow.\n\n"
                    "A convergent spiral on a directionally unstable "
                    "aircraft is arithmetic, not stability.",
                    title="Why these heights are not offered", kind="warn")
            if scan and scan.get("best"):
                frac, h, best_m, cnb = scan["best"]
                if best_m > 0:
                    widgets.hint(
                        f"Yes — the best of ten heights is {h:.3f} m "
                        f"({frac:.0%} of span), margin {best_m:+.6f}, "
                        f"Cn_beta {cnb:+.5f}. Set the height above to it.",
                        "")
                else:
                    gam = _flown_dihedral_deg()
                    widgets.hint_help(
                        f"No — the least negative margin is {best_m:+.6f} "
                        f"at {h:.3f} m.",
                        f"Every fin from {FIN_BAND[0]:.0%} to "
                        f"{FIN_BAND[1]:.0%} of the span leaves the margin "
                        "negative. Shrinking the fin takes the dihedral "
                        "effect down with the yaw stiffness, so the ratio "
                        "does not improve.\n\n"
                        "The spiral is a WING answer here, not a fin one"
                        + (f" — and this wing is flying {gam:+.3g} deg of "
                           f"dihedral already, so the answer is MORE of it."
                           if abs(gam) > 1e-9 else
                           " — and this wing carries NO dihedral, which is "
                           "where all of its Cl_beta went missing. "
                           + _cant_answer_route()),
                        title="Can a fin size fix the spiral?", kind="warn")

    def _render_rudder_rec():
        """THE RECOMMENDED RUDDER, beside the field that sets it.

        A number, not only a button: the ask was for the rudder to be GIVEN
        a recommended value, and a recommendation you have to press
        something to see is not one.
        """
        rec = _rudder_recommendation()
        if rec is None:
            return
        num(
            "hold this much sideslip [deg]",
            C["vertical"].get("rudder_beta_deg"),
            lambda e: edit_num(("vertical", "rudder_beta_deg"), e,
                               blank=None),
            step=1.0, width="w-28",
            note=f"= {rec['cross_ms']:.2f} m/s of crosswind",
            help=f"Blank is {RUDDER_BETA_DEFAULT:.0f} deg, and that is a "
                 f"STATED default rather than a measurement: nothing in "
                 f"this package derives a crosswind requirement and no "
                 f"report carries one.\n\n"
                 f"What makes it answerable is the crosswind it corresponds "
                 f"to at this design's own {rec['V']:.1f} m/s — "
                 f"{rec['cross_ms']:.2f} m/s — which is the form to argue "
                 f"with. Nothing refuses any value, including zero.",
            help_title="Where 10 degrees comes from")
        if not rec.get("reachable", True):
            bm = rec.get("beta_max")
            widgets.kv("recommended hinge chord", "OUT OF REACH",
                       color=theme.BAD)
            widgets.hint_help(
                f"No rudder holds {rec['target']:.1f} deg on this fin.",
                "At a hinge fraction of 1.00 the rudder IS the fin — "
                "all-moving — and that holds "
                + (f"{bm:.1f} deg" if bm is not None else "less") + ".\n\n"
                "Nothing is clamped to the edge of the search and nothing "
                "is applied: the answer is a bigger fin ARM or a lower "
                "target, and both are decisions.",
                title="Out of reach, measured at the ceiling", kind="warn")
            return
        f_rec = rec.get("f_rec")
        if f_rec is None:
            return
        widgets.kv("recommended hinge chord",
                   f"{f_rec:.3f}   (now {rec['f_now']:.3f})",
                   color=(theme.WARN if abs(f_rec - rec["f_now"]) > 5e-3
                          else theme.GOOD))
        widgets.kv("...which HOLDS",
                   f"{rec['beta_rec']:.2f} deg   (target "
                   f"{rec['target']:.1f}, now {rec['beta_now']:.2f})")
        widgets.kv("...and COSTS",
                   f"Cn_dr {rec['cn_now']:+.5f} -> {rec['cn_rec']:+.5f} /rad")
        if rec.get("re_rec") and rec.get("re_now"):
            widgets.kv("...at rudder Re",
                       f"{rec['re_now']:.3g} -> {rec['re_rec']:.3g}",
                       color=(theme.BAD if rec["re_rec"] < rec["re_now"]
                              else theme.INK))
        ui.button("set the hinge chord", on_click=_apply_rudder) \
            .props("dense flat no-caps")
        if rec["meets"]:
            widgets.hint_help(
                f"Already holds {rec['beta_now']:.1f} deg — this is the "
                f"SMALLEST rudder that still does.",
                "So the recommendation shrinks the surface, and a smaller "
                "rudder runs at a lower Reynolds number than the block "
                "above is trying to raise.\n\n"
                "Which of the two matters is a decision: nothing here is "
                "applied for you, and nothing refuses the fraction you "
                "type.",
                title="What the recommendation is trading")
        else:
            widgets.hint_help(
                f"Short of the target: holds {rec['beta_now']:.1f} of "
                f"{rec['target']:.1f} deg.",
                f"The recommended chord is a MEASUREMENT, not a scaling: "
                f"{RUDDER_PASSES} rebuilds, each one re-solving for the "
                f"hinge chord from the sideslip the lattice actually "
                f"produced.\n\n"
                "That is necessary because Cn_dr is only NEARLY "
                "proportional to the thin-aerofoil effectiveness — 0.7 % "
                "out at a hinge fraction of 0.3, 2.8 % at 0.9.",
                title="How the recommended chord was found")

    def v_rudder_frac() -> float:
        return float(C["vertical"]["rudder_chord_frac"])

    # ---------------------------------------------------- view: derivatives
    #: (attribute, symbol, what it is, unit, WANTED SIGN, what fixes it).
    #:
    #: The sign is in the table rather than in a second view. There was a
    #: "Stability check" tab that printed four of these rows again, off the
    #: same deck object, with the same values — one question ("is this
    #: stable?") answered in two places, one tab-click apart, and neither
    #: of them the place a reader lands on first. What that tab uniquely had
    #: was the static margin and a colour, so both moved HERE: every row is
    #: painted by its own sign test, and a row with no preferred sign
    #: (``0``) is painted as the plain measurement it is.
    ROWS = (("static_margin", "SM", "static margin (POSITIVE = stable)",
             "mac", +1, "the CG is ahead of the neutral point"),
            ("CL_alpha", "CL_a", "lift-curve slope", "/rad", 0, ""),
            ("Cm_alpha", "Cm_a", "pitch stiffness (negative = stable)",
             "/rad", -1, ""),
            ("Cl_p", "Cl_p", "roll damping (negative)", "/rad", -1, ""),
            ("Cm_q", "Cm_q", "pitch damping (negative)", "/rad", -1, ""),
            ("Cn_r", "Cn_r", "yaw damping (negative)", "/rad", -1, ""),
            ("CY_beta", "CY_b", "side force from sideslip (negative)",
             "/rad", -1, ""),
            ("Cl_beta", "Cl_b", "dihedral effect (negative = stable)",
             "/rad", -1, "wing dihedral, or a fin above the CG"),
            ("Cn_beta", "Cn_b", "yaw stiffness (POSITIVE = stable)", "/rad",
             +1, "a fin aft of the CG is what makes this positive"),
            ("Cl_r", "Cl_r", "roll from yaw rate", "/rad", 0, ""),
            ("Cn_p", "Cn_p", "adverse yaw from roll rate", "/rad", 0, ""))

    def _ensure_deck():
        """Build the deck if a design exists and nothing has built one yet.

        The deck used to appear only as a side effect of edit(), so opening
        this stage and going straight to Derivatives showed "no deck" for
        ever — the user had to nudge an unrelated field to make the stage
        work. A view must not depend on having been edited first.

        KEYED ON THE REPORT, not on "is there a deck and no error". Two
        things went wrong with that test and both ended at stage 6:

        * it LATCHED. ``C["error"]`` is cleared only by a successful
          ``rebuild``, and this refused to call one while an error was
          present, so a single failed rebuild — the no-report branch during
          a re-run is enough — left the stage broken for the session. The
          only escape was to nudge an unrelated field, because ``edit``
          rebuilds unconditionally.
        * it went STALE. A new run replaces ``S["run"]["report"]`` and
          leaves ``C["deck"]`` non-None, so nothing rebuilt and stage 6 went
          on flying the previous design with no sign on screen.

        Both are the same question — "is this deck the one for the report on
        screen?" — so it is asked once, by identity. A rebuild that raised
        still stamped ``deck_report``, so a design that cannot build is not
        re-attempted on every repaint.
        """
        rep = _report()
        if rep and (C.get("deck_report") is not rep
                    or C.get("deck_point") != _air()):
            rebuild()

    def _render_derivatives():
        box = ctx.views[("controls", "derivatives")]
        _ensure_deck()
        box.clear()
        with box:
            if C.get("error"):
                widgets.hint(C["error"], "warn")
                return
            D = C.get("deck")
            if D is None:
                widgets.hint("No deck yet.", "warn")
                return
            fm = C.get("fm")
            failed = []
            with widgets.group_box("Stability derivatives (body axes: "
                                   "x forward, y starboard, z down)"):
                for attr, sym, meaning, unit, want, fix in ROWS:
                    val = float(getattr(D, attr))
                    why = D.zeros.get({"Cn_b": "Cn_beta", "Cl_r": "Cl_r",
                                       "Cn_p": "Cn_p"}.get(sym, ""), None)
                    ok = (want == 0) or (val * want > 0.0)
                    if not ok:
                        failed.append(sym)
                    widgets.kv(f"{sym}  ({meaning})",
                               f"{val:+.5f} {unit}",
                               color=("" if want == 0 else
                                      theme.GOOD if ok else theme.BAD))
                    if why:
                        widgets.hint(why, "warn")
                    elif not ok:
                        widgets.hint(fix or f"{sym} is the wrong sign for "
                                            f"stability.", "warn")
                widgets.hint_help(
                    ("Every sign test passes." if not failed else
                     "Wrong sign: " + ", ".join(failed) + "."),
                    "A coloured row is one derivative tested against the "
                    "sign stability asks of it; a plain one has no "
                    "preferred direction.\n\n"
                    "They are SIGNS, not a certificate. A design that "
                    "passes every one of them can still be unpleasant to "
                    "fly — a sign says nothing about how FAST a mode "
                    "returns, only that it does. Stage 6's Modes tab reads "
                    "the eigenvalues of this same deck, with periods and "
                    "times to half.",
                    title="What passing every row means",
                    kind="" if not failed else "warn")
            if D.columns.get("aileron") or D.columns.get("rudder"):
                with widgets.group_box("Control power"):
                    for nm, key, lbl in (
                            ("aileron", "Cl", "Cl_da  roll per rad aileron"),
                            ("aileron", "Cn", "Cn_da  yaw per rad aileron"),
                            ("elevator", "Cm", "Cm_de  pitch per rad elev."),
                            ("rudder", "Cn", "Cn_dr  yaw per rad rudder"),
                            ("rudder", "CY", "CY_dr  side force per rad")):
                        col = D.columns.get(nm)
                        if col:
                            widgets.kv(lbl, f"{col[key]:+.5f} /rad")
            else:
                # SILENCE READS AS A MISSING PANEL.
                widgets.hint(
                    "No control-power rows: this design has no aileron and "
                    "no rudder in its deck.", "warn")
            if fm is not None:
                with widgets.group_box("Is this the aeroplane that was "
                                       "scored?"):
                    # THE HONESTY SURFACE, and it is long by nature: every
                    # assumption the bridge had to make is a sentence with a
                    # number in it. On screen that was four to six
                    # paragraphs stacked under the derivative table nobody
                    # had finished reading. Behind one mark, counted, they
                    # are findable instead of skipped.
                    fid = fm.fidelity.as_text()
                    widgets.hint_help(
                        fid.split(".")[0].strip() + ".",
                        fid, title="How close rebuilt is to scored",
                        kind="" if fm.fidelity.ok else "warn")
                    if not fm.fidelity.ok:
                        widgets.hint_help(
                            "Rebuilt and scored disagree — shown, not hidden.",
                            "The simulation rebuilds a vortex-lattice model "
                            "from the stored report. Where the design was "
                            "SCORED by a different solver — the "
                            "lifting-line path, for instance — the two "
                            "legitimately disagree.\n\n"
                            "The gap is displayed rather than tolerated: "
                            "read the flight behaviour as THIS model's, not "
                            "as the optimiser's.",
                            title="Why the two numbers differ", kind="warn")
                    if fm.assumptions:
                        widgets.hint_help(
                            f"{len(fm.assumptions)} stated assumption"
                            + ("s." if len(fm.assumptions) != 1 else "."),
                            "\n\n".join(str(a) for a in fm.assumptions),
                            title="What the bridge had to assume")

    # ---------------------------------------------------------- view: check
    # ----------------------------------------------------- view: propulsion
    def _render_propulsion():
        """WHERE THE THRUST VECTOR STARTS — one question, asked once, here.

        It was a field in stage 6, beside the flight condition, which put a
        property of the AIRFRAME among the levers of a particular flight:
        the deck this stage builds knew nothing about it, and the same
        design had a different thrust line depending on which stage had last
        been opened.

        Asked as a PAIR, because the interesting answer is not a number.
        "Through the CG" is thrust with no pitching moment at all, which is
        the honest default for propulsion this package does not model; it is
        a real answer, not a zero somebody forgot to type. Turning it off is
        what makes the throttle a pitch input.
        """
        box = ctx.views[("controls", "propulsion")]
        _ensure_deck()
        box.clear()
        T = C["thrust"]
        with box:
            with widgets.group_box("Thrust line"):
                with ui.row().classes("w-full items-center gap-1 no-wrap"):
                    ui.switch("thrust acts THROUGH the CG (no pitching "
                              "moment)",
                              value=bool(T["through_cg"]),
                              on_change=lambda e: edit(("thrust",
                                                        "through_cg"),
                                                       bool(e.value)))
                    widgets.help_dot(
                        "ON is a real answer, not a zero somebody forgot to "
                        "type: thrust through the CG makes no pitching "
                        "moment at all, which is the honest default for "
                        "propulsion this package does not model.\n\n"
                        "Turning it OFF is what makes the throttle a pitch "
                        "input — and that is the whole reason the question "
                        "is asked here, where the deck can see it, rather "
                        "than among stage 6's flight levers.",
                        title="Through the CG, or offset")
                if not T["through_cg"]:
                    num("thrust line below the CG [m]",
                        T["z_below_cg_m"],
                        lambda e: edit(("thrust", "z_below_cg_m"),
                                       _number(e, blank=0.0)),
                        step=0.05,
                        note="+ is BELOW the CG",
                        help="A thrust line below the CG pitches the nose "
                             "UP when the throttle opens, because the "
                             "moment of a force is r x F.\n\n"
                             "Type a negative number for a thrust line "
                             "ABOVE the CG; nothing here refuses a sign.",
                        help_title="Which way the offset points")
                    z = float(T["z_below_cg_m"] or 0.0)
                    widgets.kv(
                        "at full throttle",
                        (f"pitching moment = thrust x {abs(z):.3g} m, nose "
                         f"{'UP' if z > 0 else 'DOWN'}") if z
                        else "nothing — the offset is zero, which is the "
                             "switch above")
                else:
                    widgets.kv("pitching moment from thrust", "none")
            widgets.hint_help(
                "One number, and it is the vertical one.",
                "A thrust line ahead of or behind the CG makes no moment at "
                "all — the moment of a force is r x F, and both vectors lie "
                "along the body x-axis.\n\n"
                "A sideways offset is asymmetric thrust, which needs a "
                "second engine before it is a question.",
                title="Why there is no x or y offset")
            widgets.hint_help(
                "Thrust is a scalar along body x. Not modelled: propeller.",
                "No propeller, no slipstream over the wing, no torque "
                "reaction. Modelling those would put a second aerodynamic "
                "model beside the lattice, and this stage's whole claim is "
                "that there is only one.\n\n"
                "HOW MUCH thrust is available is answered in stage 6, at "
                "the throttle, because nothing here knows what the design "
                "has.",
                title="What the propulsion model is", kind="warn")

    ctx.on_render("controls", "surfaces", _render_surfaces)
    ctx.on_render("controls", "propulsion", _render_propulsion)
    ctx.on_render("controls", "vertical", _render_vertical)
    ctx.on_render("controls", "derivatives", _render_derivatives)
    ctx.register("controls_rebuild", rebuild)
    # ...and the one stage 6 asks before it flies anything: "is the deck the
    # one for the design on screen?" See flight.py::_fm.
    ctx.register("controls_ensure_deck", _ensure_deck)
    ctx.register("controls_fin", _fin_geometry)
    ctx.register("controls_size_vv", _size_to_Vv)
    ctx.register("controls_size_cnb", _size_to_Cnb)
    ctx.register("controls_deck_at", _deck_at)
    ctx.register("controls_scan_spiral", _scan_spiral)
    # ...and the fin's three dimensions, through the SAME guard the field
    # uses, so a test drives what the user drives
    ctx.register("controls_edit_fin", lambda key, value: _set_dim(
        key, type("E", (), {"value": value})()))
    ctx.register("controls_scan_dihedral", _scan_dihedral)
    ctx.register("controls_reynolds", _re_recommendation)
    ctx.register("controls_apply_re", _apply_re)
    ctx.register("controls_layout", _layout)
    ctx.register("controls_rudder_rec", _rudder_recommendation)
    ctx.register("controls_apply_rudder", _apply_rudder)
    ctx.register("controls_deck_at_rudder", _deck_at_rudder)
