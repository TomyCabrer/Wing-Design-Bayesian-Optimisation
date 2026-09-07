"""Session -> ``aerobo.api.RunConfig``: the only place V3 builds a run.

Same contract the other shells honour, restated for the V3 session shape:

* an UNTOUCHED card sends NOTHING — no mission kwargs, no bound overrides —
  so a default V3 run is bit-for-bit the run its PROBLEM publishes. Note
  which problem that is: the session opens on the polynomial chord law
  (``nice_app.BUILDER_START``), so an untouched air session assembles
  ``wing (free chord law)``, not ``trim wing``. The chord law is a problem
  NAME, chosen in the builder — everything this module could add on top is
  still empty, and switching the chord law back to straight taper assembles
  the published trim wing itself;
* overrides are SPARSE: a row equal to the default box is dropped, so the
  same run assembled through different paths collapses to the same config;
* the physics flags come from the V1 builder helpers (``tail_flags``,
  ``winglet_flags``, ``car_flags``, ``planform_flags``), never from a table
  restated here.

The one thing V3 adds is the SECTION LINK: stage 2 chose a section, and
:func:`section_link_rows` turns that choice into bound overrides on
whichever rows of the wing problem can actually carry it — the CST weights
where the family designs its own section, the thickness row where it flies
t/c, and nothing at all where the family's section is fixed. Every row it
cannot honour is reported rather than dropped silently.
"""

from __future__ import annotations

import math

#: (height row, cant row) for every tip device the registry can put in a
#: design vector — the four names ``api._WINGLET_H_ROWS`` knows. Paired
#: because the cant of a device that does not exist is not a weak row, it is
#: a DEAD one, and the pair is what makes that derivable.
DEVICE_ROWS = (("winglet_h_frac", "winglet_cant_deg"),
               ("winglet_h_frac_t", "winglet_cant_t_deg"),
               ("winglet_h_front", "winglet_cant_front_deg"),
               ("winglet_h_rear", "winglet_cant_rear_deg"))

#: how far either side of a chosen CST weight the wing's box is opened when
#: the section is pinned. Small enough that the wing search stays near the
#: section stage's answer, wide enough that the optimiser is not searching a
#: sliver (a collapsed dimension is refused outright by api._apply_overrides).
CST_HALF_WIDTH = 0.06

#: half-width used when the link lands on a thickness variable instead
TC_HALF_WIDTH = 0.008


def spec(S: dict):
    from aerobo import api

    return api.PROBLEM_SPECS[S["wing"]["problem"]]


def mission_kwargs(S: dict) -> dict:
    """Operating-point fields the wing problem honours (already filtered to
    the ones that differ from its own defaults — see session.sync_wing)."""
    sp = spec(S)
    return {k: float(v) for k, v in S["wing"]["mission_edits"].items()
            if k in sp.mission_fields}


def default_bounds(S: dict) -> dict:
    """``{label: [lo, hi]}`` the run opens on — the box BEFORE this session
    narrows anything.

    The registered spec's own box, with the rows a FLAG moves read off the
    built problem instead.

    The SPAN is the one that started this. A sized family's span box is a
    fraction of the nominal span it is given (sizing.span_bounds), and this
    shell gives it one — the area the mission states and the aspect ratio
    stage 2 guessed. So the static ``default_bounds["b_m"]`` is the family's
    published band (6–40 m on the 10 m trim wing) while the run searches
    1.44–9.6 m around a 2.4 m wing: the design box would have painted a row
    "default" and searched another.

    The TIP DEVICES' CANT rows are the same thing one control further out.
    A tip-device SHAPE narrows the cant band by flag — a vertical fence is
    84–90° rather than the family's 60–90° (objective.WINGLET_TYPES), and on
    the second surface the type's magnitude with the direction's sign
    (wingtail.tail_cant_bounds_for) — so with the fence chosen the table
    said 60–90 while the run searched 84–90. Read off the BUILT problem, in
    a guard: a build can fail for reasons that have nothing to do with the
    box (a chosen section that cannot be loaded), and a design box that
    disappears because of one is worse than one row quoting the family's
    published band.

    Everything that asks what the box IS goes through here, so the view, the
    sparse overrides and the run cannot disagree about it (api.span_box).
    """
    from aerobo import api, sizing

    out = {k: [float(a), float(b)]
           for k, (a, b) in spec(S).default_bounds.items()}
    # ...for EVERY span row: a tandem pair searches one per wing, and the
    # second one drifts from the static box exactly as the first does
    for key in sizing.span_labels(2):
        if key not in out:
            continue
        row = api.span_box(S["wing"]["problem"], flags(S), key)
        if row is not None:
            out[key] = [float(row[0]), float(row[1])]
    # a None means the row does not EXIST under this session's flags (a chord
    # law with fewer parameters than the registered twin), so it leaves the
    # box rather than sitting in it with a band nothing searches
    for key, row in _flag_moved_rows(S, out).items():
        if row is None:
            out.pop(key, None)
        else:
            out[key] = row
    return out


#: design-vector rows a VALUE flag can narrow. Each is a tip device's cant:
#: the wing's, the second surface's, and a tandem pair's two.
CANT_ROWS = ("winglet_cant_deg", "winglet_cant_t_deg",
             "winglet_cant_front_deg", "winglet_cant_rear_deg")

#: ...and the SECOND SURFACE'S HEIGHT, for the same reason one step further
#: in: its box is a FRACTION OF THE SPAN (wingtail.Z_T_FRAC_BOUNDS — 0.05 b
#: at the bottom, the clearance the family's results were measured at, and
#: since session 36 a DEFAULT rather than a floor). The static band is
#: the 10 m trim wing's 0.5–3 m, so on a 23.6 m span the table said 0.5–3
#: while the run searched 1.18–7.08 — and a user who typed the 0.5 they were
#: shown turned it into a real override every draw below 1.18 then failed on.
#: ...and the SEARCHED WING LOADING, which the mission's own ceiling moves.
#: The ceiling travels as the flag ``wing_loading_limit_pa`` off stage 1's
#: constraint diagram, and it cuts the top off the ``ws_pa`` row: measured on
#: ``tail + free span + free W/S``, the static box says 32.64-130.56 Pa while
#: the run with the ceiling searches 32.64-75.20. Omitted from this list, the
#: design box painted the un-flagged band as "default", so FIXING the row at a
#: value inside the range shown — 100 Pa, say — was refused outright by
#: ``api._pin_of``: the mirror image of the failure ``feasible.py`` exists for,
#: and the same rule broken (the box SHOWN must be the box SEARCHED).
#: ...and the SECOND SURFACE'S SIZE — its area, its arm and (where its
#: planform is designed) its aspect ratio. All three are now FRACTIONS of
#: the wing (tail.area_band / tail.arm_band: 5-30 % of its area, 0.3-0.8 of
#: its span), so the static box is the 10 m aeroplane's 0.5-3 m^2 and 3-8 m
#: while a 0.5 kg model searches 0.009-0.054 m^2 on a 0.36-0.96 m arm. Two of
#: them move again for a stated span or chord limit, which is an area and an
#: aspect-ratio band exactly (tail.TailLimits.narrow). Left out of this list
#: the design box would paint the 10 m aeroplane's band "default" over a
#: model aeroplane's search — the same rule broken as for the span itself.
#: ...and the WING'S OWN REFERENCE AREA, wherever it is a design variable.
#: The sized air families search ``S_m2`` over a band their flags carry, and
#: the static spec box is the published one, so it belongs in this list for
#: the reason every row here does.
#:
#: The CAR no longer moves this row by flag — its ``S_m2`` and ``b_m`` bands
#: ARE the design box's rows and reach the solver through
#: ``api._car_size_band_kwargs``, so a rebuild reads back the family's own
#: published band, which is exactly what a "default" row should say. It was
#: not always: with the card's area band typed as 0.2-0.3 m² the box SHOWN
#: read [0.10, 0.48] while the run SEARCHED [0.20, 0.30] — this list's own
#: rule ("the box SHOWN must be the box SEARCHED") broken on the row the card
#: had just been used to state, and its mirror image, a PIN inside the band
#: shown but outside the band searched, refused outright by ``api._pin_of``.
#: Both are gone with the card fields; the entry stays for the air families.
FLAG_MOVED_ROWS = CANT_ROWS + ("z_t_m", "ws_pa", "S_m2",
                               "S_t_m2", "l_t_m", "AR_t")

#: ...and the SIZE rows a flag ADDS rather than moves. Every entry above is
#: a row the family already has, whose BAND a flag narrows; these two do not
#: exist at all until the flag is on. The water families are where that
#: matters: a hydrofoil's span and reference area are STATED numbers, and
#: ``free_span`` / ``free_area`` turn each into a design variable. Read only
#: through FLAG_MOVED_ROWS the box missed them completely — with "free AREA"
#: chosen the run searched a seventh variable (``S_m2``, 0.072-0.288 m^2 on
#: the published 0.144 m^2 foil) that the design box never drew, so the band
#: could not be read, could not be typed and could not be pinned, and the
#: card said "fixed span + area (you choose the size)" over a search that
#: chose it instead. Same rule as the list above, one case further out: the
#: box SHOWN must be the box SEARCHED.
#: ...and ``z_t_frac``, which is one step stranger: it does not merely
#: appear, it REPLACES ``z_t_m``. On a family that searches its stabiliser's
#: depth, freeing the SPAN changes that row's units — a depth in metres is a
#: statement about one span, and with the span searched the same number means
#: two different layouts at the two ends of its band, so the row is asked as
#: a fraction of the candidate's own span instead (hydrotail.Z_T_FRAC_LABEL).
#: The static box still has the metre row, so this list ADDS the fraction and
#: the loop below DROPS any row the built problem turns out not to have.
FLAG_ADDED_ROWS = ("b_m", "S_m2", "z_t_frac")


def _flags_without_the_objective(fl: dict) -> dict:
    """The same flags with every OBJECTIVE key removed.

    Used only to READ a design box that the objective's own refusal would
    otherwise hide (:func:`flag_moved_bounds`). Sound because an objective is
    not a dimension: it says what is maximised, never which rows exist or how
    wide they are, so the box it reads back is the box the run will search.
    Asked of ``api``'s own ``*_OBJECTIVE_FLAG_KEYS`` tuples rather than a
    list kept here, so a new scored quantity that joins one of those tuples
    cannot re-open the hole by arriving under a new name.

    ``car_objective`` IS the exception and is named literally, because the
    car families publish no ``CAR_OBJECTIVE_FLAG_KEYS`` tuple to ask. That
    is the one family a future ``car_score_weights`` would slip past; the
    fix when it lands is the tuple in ``api``, not another literal here.
    (``wing_objective``/``foil_objective`` are in the literal for symmetry
    only — their own tuples already carry them.)
    """
    from aerobo import api

    drop = set()
    for name in dir(api):
        if name.endswith("OBJECTIVE_FLAG_KEYS"):
            drop |= set(getattr(api, name) or ())
    # the car's scored quantity is spelled without the shared suffix
    drop |= {"car_objective", "foil_objective", "wing_objective"}
    return {k: v for k, v in fl.items() if k not in drop}

#: ...and the CHORD LAW's rows, which a flag can move differently again: the
#: LAW itself is a flag (api.CHORD_LAW_KEY), and a law does not merely narrow
#: its rows' bands — it can carry a different NUMBER of them (the elliptic
#: blend is one number where the cubic is three) and a different kind of box
#: (a fraction in [0, 0.9] rather than a symmetric +/-). So these rows are not
#: patched one by one like the cant rows: the whole chord block comes off the
#: built problem, and the static one is dropped.
CHORD_ROW_PREFIX = "chord_k"


def _is_chord_row(label: str) -> bool:
    """Is this design-box row a chord-law parameter?

    The names a family gives them: ``chord_k1``, ``chord_front_k2`` (one law
    per wing on a tandem), ``chord_k1_t`` (a designed tail's own).
    """
    import re

    return bool(re.fullmatch(r"chord_(?:(?:front|rear)_)?k\d+(?:_t)?",
                             str(label)))


def _chord_rows_the_law_carries(S: dict, chord: list) -> dict:
    """``{row: None}`` for the chord rows this session's LAW cannot carry.

    The build is what normally decides which coefficient rows exist, and
    where it fails this is the same answer read off the law itself: every
    law but the elliptic blend carries its order, and the blend is ONE number
    whatever order the registered twin declares
    (:data:`aerobo.geometry.CHORD_LAW_ORDERS`, clamped for the run in
    :func:`aerobo.api.chord_order_for`). Counted PER BLOCK, because a problem
    can carry several — a pair's two wings, a designed tail's own.
    """
    import re

    from aerobo import api, geometry

    law = str(flags(S).get(api.CHORD_LAW_KEY) or geometry.DEFAULT_CHORD_LAW)
    orders = geometry.CHORD_LAW_ORDERS.get(law)
    if not orders:
        return {}
    keep = max(orders)
    out: dict = {}
    for key in chord:
        m = re.fullmatch(r"chord_(?:(?:front|rear)_)?k(\d+)(?:_t)?", str(key))
        if m and int(m.group(1)) > keep:
            out[key] = None
    return out


def _flag_moved_rows(S: dict, static: dict) -> dict:
    """``{label: [lo, hi]}`` for the rows this session's FLAGS have moved.

    Guarded: a build failure leaves the static box, because a design box
    that vanishes tells the user less than one row quoting the published
    band does.

    The chord block is handled wholesale rather than row by row, because a
    chord LAW changes which rows exist (see CHORD_ROW_PREFIX): the caller
    drops every static chord row and takes the built problem's instead.
    """
    from aerobo import api

    wanted = [k for k in FLAG_MOVED_ROWS if k in static]
    # ...and the rows a flag ADDS, which are by definition NOT in the static
    # box, so they cannot be found by looking there (FLAG_ADDED_ROWS).
    wanted += [k for k in FLAG_ADDED_ROWS if k not in wanted]
    chord = [k for k in static if _is_chord_row(k)]
    if not wanted and not chord:
        return {}
    try:
        spc = api.PROBLEM_SPECS[S["wing"]["problem"]]
        # THE SAME MISSION THE RUN FLIES. Building with ``{}`` here reads the
        # rows off the FAMILY'S PUBLISHED mission while ``cfg_dict`` sends
        # ``mission_kwargs(S)`` — the exact failure this table exists to
        # prevent, one level in. ``objective.Problem`` derives ``ws_pa`` as a
        # fraction band around W/S, so every weight edit moved the searched
        # centre while the shown band stayed put: measured on ``trim wing +
        # free span + free W/S`` at W_N = 700 N, the box showed
        # 32.640125-75.202848 tagged "default" while the run searched
        # 35.0-75.202848, so a pin at 33 Pa — inside the band on screen — was
        # refused by ``api.size_box_conflicts``. At W_N = 7000 N the same
        # build raises instead, which the except below turns back into the
        # static box (a read-out that quotes the published band still tells
        # the user more than a design box that vanishes).
        built = spc.build(mission_kwargs(S) if spc.uses_mission else None,
                          flags(S), None)
    except Exception:                       # noqa: BLE001 — a read-out
        # ...AND THE FIRST THING TO TRY IS THE SAME BUILD WITHOUT THE
        # OBJECTIVE. The commonest way this raises is not a broken box at
        # all: the composite objective refuses to build until its
        # normalisation band is measured (``api.wing_score_reference``), and
        # that refusal is about WHAT IS MAXIMISED, not about which rows exist
        # or how wide they are. An objective is not a dimension — it moves no
        # bound — so a build with those keys stripped reads exactly the same
        # box, and reading it there is honest rather than a guess.
        #
        # It matters because of what the blanket fallback below drops:
        # FLAG_ADDED_ROWS are by definition absent from the static box, so
        # they do not fall back to a published band, they VANISH — while
        # ``flags(S)`` still sends ``free_span``/``free_area`` and the run
        # still searches them. Measured on a water session with "free span +
        # area" chosen and a heel typed: objective ``lod`` showed b_m and
        # S_m2, objective ``composite`` showed neither, and both runs
        # searched both. A box that hides a row the optimiser is riding is
        # worse than a box quoting a band one step behind.
        #
        # THIS CAN MOVE A STORED SESSION'S SEARCH, and it is the one part of
        # the change it belongs to that can — so it is written down rather
        # than claimed away. ``bounds_overrides`` emits a row only where the
        # stored value differs from ``defaults``, so widening what
        # ``defaults`` knows changes which rows are SENT. Measured on an AIR
        # family, no water anywhere near it: ``tail + winglet [free height,
        # designed tail] + free chord law`` with a vertical device and the
        # composite objective moves the drawn box from the static 84-90 to
        # the built 60-90, and a user's typed 60-90 stops being silently
        # discarded and starts being searched. That is the correct
        # direction — the old behaviour dropped the edit — but it is a
        # change of behaviour and not a no-op.
        try:
            built = spc.build(
                mission_kwargs(S) if spc.uses_mission else None,
                _flags_without_the_objective(flags(S)), None)
        except Exception:                   # noqa: BLE001 — a read-out
            # A FALLBACK MAY NOT LEAVE THE BOX CONTRADICTING A FLAG. Quoting
            # the published band for a row whose band a flag moved is a
            # read-out one step behind; keeping a row the session's LAW DOES
            # NOT HAVE is a box that cannot be drawn at all —
            # ``geometry.chord_reach`` raises on three coefficients under the
            # elliptic blend (one number), and the panel that draws the law
            # took stage 3's whole Design box down with it.
            return _chord_rows_the_law_carries(S, chord)
    labels = list(built.param_labels)
    out = {}
    for key in wanted:
        if key not in labels:
            # ...and a row the session's flags took OUT of the design vector
            # leaves the box (None -> the caller pops it), rather than sitting
            # in it with a band nothing searches. The chord block below has
            # always done this; the row that made it general is ``z_t_m``,
            # which a searched span replaces with ``z_t_frac``.
            if key in static:
                out[key] = None
            continue
        lo, hi = built.bounds[labels.index(key)]
        out[key] = [float(lo), float(hi)]
    if chord:
        for key in chord:                   # the static block goes entirely
            out[key] = None
        for i, key in enumerate(labels):    # ...and the built one replaces it
            if _is_chord_row(key):
                lo, hi = built.bounds[i]
                out[key] = [float(lo), float(hi)]
    return out


#: HOW THE SECOND SURFACE IS MOUNTED, for every family that has one. The
#: shell used to ask (follow the load / upright / upside down) and now
#: states it: a stabiliser or elevator is built to push DOWN, which is what
#: mounting a cambered aerofoil upside down is for — its camber works the
#: side the layout loads it on instead of the trim incidence being cranked
#: to fight it (``tail.tail_polar``: 3.9 % off the published stabiliser's
#: profile drag, i_t -4.93 -> -0.31 deg). The engine still takes all three
#: (``api.TAIL_MOUNTS``); this is the one the shell sends.
#:
#: What it costs, measured at each family's mid-box design and its own CG:
#: the AIR tail is unchanged to the last bit (the balance already asked for
#: a download there, so "follow the load" was already sending this), while
#: the WATER elevator trims to an UP-load at its default CG and now flies
#: against its own camber for it — L/D 24.67 -> 23.77 and i_t -0.10 ->
#: +4.51 deg. Far enough aft in air the same thing stops being a cost and
#: becomes a refusal: past about 0.8 m of CG the incidence leaves the polar
#: and there is no design to read (``session.stability_unevaluable`` is what
#: says so on the card).
TAIL_MOUNT = "inverted"

#: WHICH WAY THE SECOND SURFACE'S TIP DEVICE POINTS — the other half of the
#: same statement, and the half that is on screen. The shell used to send
#: ``follow``: the cant's magnitude was the design variable and its SIGN was
#: read off the load the trim solve put on that surface
#: (``wingtail.tail_winglet_follow``). That is the right rule for a surface
#: whose load is not stated — and this shell states it. :data:`TAIL_MOUNT`
#: pins the surface to push DOWN, so leaving the device to read the trimmed
#: load meant the two answers could disagree: an elevator mounted upside
#: down to push down, carrying a device placed as though it lifted. That is
#: what the geometry view drew, and it is one question answered twice.
#:
#: What it costs, MEASURED at matched cant MAGNITUDE against ``follow`` over
#: each family's own design box (uniform draws, both arms feasible):
#:
#: * AIR (``tail [designed tail + tip device]``, n = 141): the load is a
#:   download everywhere in the box, ``follow`` already mirrored the device,
#:   and stating it is **bit-for-bit identical at 141/141** designs. No cost
#:   and no change to any published air run.
#: * WATER (``hydrofoil + elevator [designed elevator + tip device]``,
#:   n = 68): ``follow`` reads an UP load at this craft's own CG and points
#:   the device up; pointing it DOWN instead **wins 64/68**, mean +0.221 %
#:   of score, best +1.117 %, and is never worse (min +0.000 %). Under water
#:   the direction is a free-surface trade rather than a lift-mirroring one
#:   (``the-water-foil-does-not-pierce``: the trade is ODD in cant, and
#:   canting down keeps the device away from the surface), so the load's
#:   sign was answering a question it does not decide.
#:
#: The engine still takes all four answers (``wingtail.TAIL_WINGLET_DIRECTIONS``
#: — ``up``, ``down``, ``either``, ``follow``); this is the one the shell
#: sends. PINNED, not defaulted, for the reason ``tail_mount`` is: the band
#: the card draws, the box the run searches and the surface the geometry view
#: lofts all read it here, and a card-level default would leave them
#: disagreeing.
TAIL_TIP_DIRECTION = "down"

#: WHICH END OF THE WING IS THE WIDE ONE — the chord trend this shell opens
#: on. ``geometry.CHORD_TRENDS`` still holds all of them and the library
#: default is ``"free"``, so every published run and every scripted study is
#: untouched; this is the V3 form's opening answer, the way ``TAIL_MOUNT`` is.
#:
#: What it costs, MEASURED over the joint published box (taper U(0.2, 1.0) x
#: chord_k1..k3 U(-0.5, 0.5), 20 000 draws through ``geometry.Wing``): 97.3 %
#: of designs build under ``"free"`` and 66.9 % under ``"root_largest"``, and
#: the band the chord-law card draws at the taper box's mid-point falls from
#: a flyable fraction of 0.966 to 0.639. That is a real budget cost on every
#: untouched session, and it is the right way round: the third of the box it
#: refuses is planforms whose chord grows towards the tip, which is a shape
#: an airframe almost never wants and which nobody was asking to search.
#: One click on the chord-law card gives it back.
CHORD_TREND = "root_largest"


def stated_physics(problem_name: str) -> dict:
    """The physics THIS SHELL ANSWERS instead of asking, for one family.

    A key earns a place here only when V3 deliberately stopped asking a
    question and stated the answer, so V2 (which still asks, or has no
    control at all) can never agree with it — and the value is pinned too,
    because V3 quietly changing its mind is a divergence as much as V3
    sending a key V2 does not.

    One declaration, read by :func:`flags` (which sends it) AND by the drift
    guards in the tests (which allow exactly it). Two copies of this list is
    how a shell comes to send something nothing is checking.
    """
    from aerobo import api

    declared = api.PROBLEM_SPECS[problem_name].flags
    return {k: v for k, v in stated_physics_all().items() if k in declared}


def stated_physics_all() -> dict:
    """Every key this shell states, before any family filters it.

    Separate from :func:`stated_physics` because a guard needs the WHOLE set
    to test membership against — filtered to one family it would silently
    allow a key on the family that declares it and refuse the same key on the
    next one.
    """
    from aerobo import api

    return {api.TAIL_MOUNT_KEY: TAIL_MOUNT,
            api.TAIL_WINGLET_DIR_KEY: TAIL_TIP_DIRECTION,
            api.CHORD_TREND_KEY: CHORD_TREND}


def flags(S: dict) -> dict:
    from aerobo import api
    from gui.nice_app import (car_flags, planform_flags, slipstream_dict,
                              tail_flags, tandem_flags, winglet_flags)

    from . import session

    W = S["wing"]
    sp = spec(S)
    # `v not in (None, False)` dropped every flag whose value is 0.0, because
    # `0.0 in (None, False)` is True in Python. A card could therefore show a
    # zero the run never received — a coplanar tail, a zero ride height, a
    # zero-floored band — and the family's own default was used instead with
    # nothing on screen saying so. Identity, not equality.
    out = {k: v for k, v in W["flags"].items()
           if v is not None and v is not False}
    if "slipstream" in sp.flags:
        ss = slipstream_dict(W["prop"])
        if ss is not None:
            out["slipstream"] = ss
    out.update(tail_flags(W["choices"]))
    # THE PHYSICS THIS SHELL ANSWERS INSTEAD OF ASKING (:func:`stated_physics`
    # — one declaration, and the drift guards read the same one).
    #
    #   tail_mount    WHICH WAY UP the second surface is built. It is always
    #                 mounted to push DOWN, so its camber works the side the
    #                 layout loads it on rather than the trim incidence
    #                 fighting it. PINNED, not defaulted: the solver,
    #                 `session.trim_lift` (and so the lift stage 2 screens the
    #                 section at) and the geometry views all take their answer
    #                 off these flags, and a card-level default would leave
    #                 three of them following the load while the run did not.
    #
    #   tail_winglet_dir
    #                 WHICH WAY THAT SURFACE'S TIP DEVICE POINTS, and the
    #                 same answer: DOWN, the side the surface is mounted to
    #                 work. Pinned for the same reason and sent AFTER
    #                 `tail_flags`, which is what makes this the authority
    #                 rather than a second opinion beside the tip-device
    #                 card. Free in air (bit-identical at 141/141 designs,
    #                 where the load was already a download) and a gain in
    #                 water (64/68) — see TAIL_TIP_DIRECTION.
    #
    #   chord_trend   WHICH END OF THE WING IS THE WIDE ONE. A chord that
    #                 GROWS towards the tip moves the structure outboard and
    #                 worsens the tip stall margin; it is not what a designer
    #                 means by "let the law be free". DEFAULTED, not pinned —
    #                 the chord-law card can state "free" and it must win —
    #                 which is why this one goes through `setdefault`. Stated
    #                 here all the same, because the band DRAWN on that card
    #                 and the box SEARCHED by the run both read it from here.
    for key, value in stated_physics(W["problem"]).items():
        if key == api.CHORD_TREND_KEY:
            out.setdefault(key, value)
        else:
            out[key] = value
    wl = winglet_flags(W["choices"])
    # the FIXED BLEND is a value, and a family whose solver draws the corner
    # as a corner declares no such flag — sending it anyway would either
    # raise or be dropped inside a builder. ``derive_problem`` already says
    # so in its notes; here it simply does not travel.
    if api.WINGLET_BLEND_KEY not in sp.flags:
        wl.pop(api.WINGLET_BLEND_KEY, None)
    out.update(wl)
    # ...and the chord limits, for the same reason: the 2-D section problem
    # has no wing, so a chord it does not draw cannot be limited. The tip
    # device's chord rides the same rule: a family with no device (or whose
    # device carries its own chord variable — the car endplate) declares no
    # such flag, and would raise on the kwarg.
    for key in (*api.CHORD_LIMIT_KEYS, api.WINGLET_CHORD_KEY,
                api.TAIL_CG_KEY, api.TAIL_HEIGHT_KEY,
                # the handling gate rides the same rule, and it matters more
                # here than for most: only the families with a lateral deck
                # declare it, and a level sent to one without would ask for a
                # Dutch roll on an aeroplane whose yaw stiffness is exactly
                # zero for a geometric reason.
                api.HANDLING_LEVEL_KEY,
                *api.TAIL_TIP_KEYS, *api.TAIL_LIMIT_KEYS):
        if key not in sp.flags:
            out.pop(key, None)
    # THE CAR'S SPEED. A mission field on a family that refuses a mission
    # spec: ``api._make_car_wing_builder`` raises on mission_kwargs and points
    # at ``flags={'V': ...}``, so stage 1's speed reached the solver through
    # nothing at all and every car run flew ``carwing.CarWingProblem.V`` = 55
    # whatever the card said. Sent here, off the same ``S["mission"]["V"]``
    # every other family's operating point comes from, and dropped by
    # ``car_flags`` when it equals the family's own default so an untouched
    # card is still bit-for-bit the published run.
    out.update(car_flags(W["choices"], S["mission"].get("V")))
    out.update(tandem_flags(W["choices"], W["problem"]))
    out.update(planform_flags(W["choices"], W["problem"]))
    # the pair's STAGGER — and the rear wing's own SPAN — only travel to a
    # family that has two surfaces to stagger, the same rule as the chord
    # limits above and the same reason. (The span additionally leaves the
    # flags of a SIZED pair, which searches a span per wing instead; the
    # registry strips it there, so this loop drops it.)
    for key in api.TANDEM_STAGGER_KEYS + api.TANDEM_SPAN_KEYS:
        if key not in sp.flags:
            out.pop(key, None)
    if sp.has_blocks and W["optimiser"] == "blocks":
        chosen = {k: v for k, v in (W.get("block_optimisers") or {}).items()
                  if v}
        if chosen:
            out["block_optimisers"] = chosen
    if W["optimiser"] == "bo" and not sp.is_constrained \
            and W["acqf"] != "logei":
        out["acqf"] = W["acqf"]
    # the water a hydrofoil flies in: sea is the published default, so an
    # untouched run still sends nothing
    if api.WATER_KEY in sp.flags and S.get("water", "sea") != "sea":
        out[api.WATER_KEY] = S["water"]
    # THE SECTION STAGE 2 CHOSE, per surface. This is what makes the
    # pipeline a pipeline: on a family that flies a fixed table polar the
    # chosen section replaces the family's NACA anchor outright, so the run
    # flies the aerofoil the user picked instead of merely being seeded by
    # it. Nothing is sent when no section was chosen, so the published run
    # is untouched.
    for key, surface in ((api.SECTION_KEY, "main"),
                         (api.SECTION_AFT_KEY, "aft"),
                         # ...and the car ENDPLATE's, on its own key. A third
                         # slot rather than a reuse of the aft one: the plate
                         # is a third surface, and a car can carry a chosen
                         # wing section beside a chosen plate section.
                         (api.SECTION_PLATE_KEY, "plate")):
        if key not in sp.flags:
            continue
        if surface in ("aft", "plate") \
                and not session.section_is_own(S, surface):
            continue            # the aft surface follows the main one, and
            #                     the plate flies its build-up until asked
        value = session.section_flag_value(S, surface)
        if value:
            out[key] = value
    # ...AND THE FIN'S, which travels as a THICKNESS rather than as a shape:
    # the fin's section is symmetric and the sizing law, the drag book and
    # the CAD loft take t/c from one flag (`fin.fin_shape_kwargs`). Stage 2.7
    # is where that section is chosen, so this is where its thickness leaves
    # the pipeline — the Wing type card used to ask for the number a second
    # time, and the run could then be charged at one t/c while flying
    # another.
    if "fin_tc" in sp.flags and session.fin_surface(S):
        out["fin_tc"] = float(session.fin_thickness(S))
    # ...AND WHERE THAT SURFACE STANDS, on the families that have somewhere
    # to put it. A strut only has a station between two wings, so the keys
    # are declared on the elevator families alone (``api.STRUT_STATION_KEYS``)
    # and this asks the registry rather than the medium.
    #
    # SENT ONLY WHEN ANSWERED, the rule the load, the car's speed and the fin
    # switch already follow: an untouched session sends neither key and flies
    # the family's own measured windfoil station, so every stored run is
    # bit-for-bit. And never both — the problem refuses a stated station
    # beside a searched one, which is a refusal a shell must not be able to
    # trigger.
    if "x_mast_frac" in sp.flags and session.fin_surface(S):
        _mast = W["choices"]
        if _mast.get("mast_station") == "free":
            out["free_mast_station"] = True
        elif _mast.get("mast_station_frac") is not None:
            out["x_mast_frac"] = float(_mast["mast_station_frac"])
    # ...AND WHETHER THERE IS A FIN AT ALL. Until this line the switch on
    # stage 1 reached NOTHING: the line above is all it did, so "no fin" was
    # implemented as "a fin at fin.FIN_TC_DEFAULT" — the run still charged
    # cd0_fin 0.000885 (5.79 % of L/D here), still weighed a surface that is
    # 82 % of the empennage book, still drew one in the report and still flew
    # one at Cn_beta 0.1133, while the card that asked said that number was
    # exactly zero.
    #
    # SENT ONLY TO SAY NO, the rule the load and the car's speed already
    # follow: the flag's default is True, so an untouched session sends
    # exactly the flags it always sent and every published run is
    # bit-for-bit. It cannot travel through ``W["flags"]`` above either —
    # that comprehension drops any value that IS False.
    #
    # Off the CHOICE, not off ``session.fin_surface``, which answers a wider
    # question than this one — it also says no for a V-tail, which needs no
    # flag at all (``fin.has_fin`` reads the layout itself), and a switch
    # that is not on screen must not send an answer.
    #
    # WATER IS INCLUDED NOW. It was excluded on the argument that "a water
    # craft's vertical is the strut, not a Raymer volume-coefficient fin",
    # which was true and is not a reason to drop the answer: the strut is a
    # reported, charged, flyable surface since V5 (``fin.mast``), the water
    # families declare the presence key for exactly that, and sending False
    # is how a design says it has no mast — which zeroes ``cd0_mast``
    # (+14.5 % L/D on the shipped foil) instead of quietly keeping it.
    # Nothing moves unless the switch is moved: the flag is sent ONLY to
    # say no.
    # ...AND WHETHER THE SIZE IS SEARCHED. One control on stage 3
    # (`session.WET_SIZE_MODES`) turns the `b_m` / `S_m2` rows on, and this
    # is where its answer leaves the pipeline. Empty for the fixed mode, so
    # an untouched water session sends exactly what it always sent.
    out.update(session.wet_size_flags(S))
    # ...AND HOW FAR OVER IT IS FLYING. Beside the size on purpose: the heel
    # is the only term any water solver has that reads the SPAN, so it is
    # what a searched span is traded against (`session.wet_heel_flags`).
    # Empty at zero heel, so an untouched water session is unchanged.
    out.update(session.wet_heel_flags(S))
    # ...and WHICH TABLE each surface's section is read from: the family's
    # Re-1e6 one, or the Reynolds number the design actually flies.
    out.update(session.wet_re_flags(S))
    if (api.FIN_PRESENCE_KEY in sp.flags
            and str(W["choices"].get("medium", "air")) in ("air", "water")
            and not bool(W["choices"].get("fin", session.FIN_DEFAULT))):
        out[api.FIN_PRESENCE_KEY] = False
    # the LOAD the mission states, on a family that carries it as a stated
    # VALUE instead of a mission field (api.WEIGHT_KEY -> the hydrofoil's
    # L_design). Every water family declares this flag and NONE of them
    # declares a `W_N` mission field, so until this line the stated load
    # crossed no boundary at all: stage 1 took "5 N", stage 2 designed its
    # section for CL = W/(qS) off that 5 N, and stage 3 flew the family's
    # published 6 kN. On the mission that reported it (5 N over 0.08 m2) the
    # consequence was not a wrong number but NO number — every draw in the
    # box cavitates carrying 6 kN on a foil sized for 5, so the measurement
    # came back empty and the box card said there was no solution.
    #
    # Dropped when it EQUALS the family's own published design lift, the
    # rule the size flags and the mission edits already follow
    # (`session.sync_wing_from_mission`): the mission form opens on
    # `api.family_design_point`, so an untouched water session still sends
    # exactly the flags it always sent and reproduces the published run.
    #
    # A non-positive or non-finite load is NOT sent. `api._weight_kwargs`
    # refuses it — correctly, a foil trimmed to 0 N has no design point —
    # but it refuses by raising out of the BUILD, and this function is on
    # every repaint's path, so sending one would replace a card with a
    # traceback. The mission card says which load is being flown instead
    # (`mission._render_load_note`), so the fallback is never silent.
    if api.WEIGHT_KEY in sp.flags:
        try:
            w = float(S["mission"]["W_N"])
        except (TypeError, ValueError, KeyError):
            w = 0.0
        if math.isfinite(w) and w > 0.0:
            published = api.family_design_point(W["problem"]).get("W_N")
            if published is None or abs(w - float(published)) > 1e-9 * max(
                    1.0, abs(float(published))):
                out[api.WEIGHT_KEY] = w
    # the WING LOADING the mission states, on a family that sizes itself from
    # it (api's "size_ws" modifier). This is what makes the mode the mission's
    # answer rather than a second one: stage 1 types W/S, the solver derives
    # the area from it per candidate, and the trim CL = (W/S)/q is fixed by
    # the same number. Nothing is sent when the family does not size that way.
    if "wing_loading_pa" in sp.flags:
        ws = session.wing_loading(S)
        if ws:
            out["wing_loading_pa"] = float(ws)
    # ...and the CEILING that same mission puts on any wing loading: the
    # binding ws_max line of its own constraint diagram (stall or landing
    # field in air, fly-up or cavitation in water). Sent to every sized
    # family, because every sized family can fly past it — the free planform
    # reaches a loading through its area row, and the two loading modes
    # state or search one. Until it was sent, that diagram was a card at
    # stage 1 and nothing else: the search could buy payload L/D with a
    # loading the aircraft could not land at, and did. Nothing is sent when
    # the mission states no requirement that bounds W/S.
    if "wing_loading_limit_pa" in sp.flags:
        cap = session.mission_ws_ceiling(S)
        if cap:
            out["wing_loading_limit_pa"] = float(cap)
    # the Mach number the mission implies, sent ONLY when the user asks for
    # it: the section tables are incompressible, so applying the correction
    # is a decision, not a derivation (see session.MACH_WARN)
    if "mach" in sp.flags and W.get("apply_mach"):
        mach = session.design_point(S).get("mach")
        if mach:
            out["mach"] = float(mach)
    # WHICH SCALAR this search maximises. Empty on the family's own
    # objective, so an untouched run sends exactly the flags it always sent;
    # a composite run carries its weights AND the frozen band, because both
    # are part of the objective rather than a display choice
    # (api.WING_OBJECTIVE_FLAG_KEYS).
    score = session.wing_score_flags(S)
    if score and all(k in sp.flags for k in score):
        out.update(score)
    # …and the rule this function applies key by key above, applied ONCE to
    # whatever is left. Every pop above is an instance of "a flag travels only
    # to a family that declares it", and the list of pops has to be kept in
    # step with the registry by hand — which is how `winglet_type` and
    # `blend_shape` were still travelling to families that ignore them.
    # `api.check_flags` refuses those now, so this is what keeps the Run
    # button working; it is a backstop for the loops above, not a replacement.
    #
    # WHAT DISCARDING THE DROPPED LIST HERE COSTS, and it is not nothing. The
    # first version of this comment argued that a key reaches this line only
    # when the shell has no control for it on THIS problem, so what is dropped
    # is always a stale choice rather than a live request. THAT ARGUMENT IS
    # RETRACTED — report §17.12's closing paragraph and HANDOVER's session-49
    # entry both record it as false, and `winglet_type` is the counter-example
    # in both: it is a live menu on the families that declare it and is
    # discarded by the ones that do not, so a shape chosen on screen was flown
    # as a different one. Because the shell sanitises BEFORE storing, the key
    # never reaches the config and `api.unhonoured_flags` cannot report it
    # either. The cure is at the MENU, not here: a control is offered only
    # where the derived family honours its flag (nice_app._shape_flag_available
    # gates both tip-device shapes that are carried as flags), which leaves
    # this line dropping stale state and nothing else.
    out, _dropped = api.sanitise_flags(W["problem"], out)
    return out



def stated_load_note(S: dict) -> str | None:
    """Why the load on the mission card is NOT the load the run flies, or
    ``None`` when it is.

    The mission card asks for a design load on every family, because stage 2
    screens its section at CL = W/(qS) whatever stage 3 turns out to be. Two
    families do not then fly it, and both used to say nothing:

    * the solver takes no load at all (the 2-D section problem, and the car
      rear wing, which maximises downforce under a drag budget and has no
      lift target to trim to) — the number is real, and it reaches the
      section stage and nothing further;
    * the family carries it as a stated VALUE (:data:`api.WEIGHT_KEY`) but
      the number cannot be trimmed to. ``api._weight_kwargs`` refuses a
      non-positive or non-finite load out of the build, so :func:`flags`
      cannot send one without replacing a card with a traceback; it sends
      nothing and the family's published design lift is flown. That is a
      fallback, and a fallback nobody is told about is the failure this
      repo keeps re-finding, so it is said here and drawn under the field.

    A family that honours the load — through a ``W_N`` mission field or
    through a usable weight flag — returns ``None`` and draws no line.
    """
    from aerobo import api

    from gui.nice_app import mission_field_note

    sp = spec(S)
    name = S["wing"]["problem"]
    try:
        w = float(S["mission"]["W_N"])
    except (TypeError, ValueError, KeyError):
        w = float("nan")
    carries = api.WEIGHT_KEY in sp.flags
    if "W_N" in sp.mission_fields:
        return None
    if not carries:
        return ("the run does not fly this load: "
                + mission_field_note(name, "W_N")
                + ". It is still the load stage 2 designs its section for.")
    if math.isfinite(w) and w > 0.0:
        return None
    published = api.family_design_point(name).get("W_N")
    flown = f"{float(published):.6g} N" if published is not None \
        else "its own published design lift"
    return (f"a design load is the lift the surface is trimmed to carry, so "
            f"it must be a positive number of newtons — this one cannot be "
            f"trimmed to, and the run flies this family's {flown} instead.")

# ------------------------------------------------------------ section link
def section_link_rows(S: dict) -> tuple[dict, list[str]]:
    """``(bound overrides, notes)`` carrying stage 2's section into stage 3.

    Three cases, and the notes say which one happened:

    * the family DESIGNS its section (``w_upper_*`` / ``w_lower_*`` in the
      design vector) — every weight is pinned to a narrow box around the
      chosen section, so the wing search starts from that shape instead of
      from the family's NACA anchor;
    * the family FLIES A THICKNESS (``tc``) — the thickness row is narrowed
      onto the chosen section's t/c;
    * the family's section is FIXED — nothing is sent, and the note says the
      choice is advisory here (the chosen section informed the mission-level
      picture, not this solver's polars).

    A weight that lies outside the family's own box is left at the family
    default with a note: silently clipping it would report a section the run
    did not fly.

    OUTSIDE is decided by CONTAINMENT — is the chosen value inside the
    family's own interval? — never by "did the clipped interval survive?".
    A weight far outside collapses the clip and is caught either way, but a
    weight just outside does not: v43015's w_upper_0 (0.36999) against a box
    ending at 0.34128 clips to a perfectly usable [0.30999, 0.34128], and
    the row would be sent, painted as pinned to v43015 and searched — a
    shape v43015 cannot take. goe741 is the sharp version: its w_lower_1
    (+0.02084) clips onto a box that is entirely negative, so the "pinned"
    row forces the opposite SIGN of the section that was chosen. The same
    holds for the thickness row (s8037's t/c 0.16004 against [0.08, 0.16]).
    """
    from . import session

    W = S["wing"]
    if W.get("section_link") != "auto" \
            or S["airfoil"].get("decision") is None:
        return {}, []
    weights = session.section_weights(S)
    sec = S["airfoil"].get("section") or {}
    defaults = spec(S).default_bounds
    half = float(W.get("section_half_width", CST_HALF_WIDTH))
    name = sec.get("name", "the chosen section")
    out: dict = {}
    notes: list[str] = []

    cst_rows = [k for k in defaults if k.startswith(("w_upper_", "w_lower_"))]
    if cst_rows and weights is not None:
        w_u, w_l = weights
        chosen = {f"w_upper_{i}": v for i, v in enumerate(w_u)}
        chosen.update({f"w_lower_{i}": v for i, v in enumerate(w_l)})
        missed = []
        for row in cst_rows:
            if row not in chosen:
                continue
            lo_d, hi_d = defaults[row]
            w = chosen[row]
            if not float(lo_d) <= w <= float(hi_d):
                missed.append(row)      # the run could never reach it
                continue
            lo = max(float(lo_d), w - half)
            hi = min(float(hi_d), w + half)
            if not hi > lo:
                missed.append(row)      # the family pins this row already
                continue
            out[row] = [lo, hi]
        if out:
            notes.append(
                f"CST box pinned to {name} (±{half:g} per weight) — the wing "
                f"search starts from that shape, not from the family's own "
                f"anchor.")
        if missed:
            notes.append(
                "outside this family's CST box, left at the family default: "
                + ", ".join(missed))
        return out, notes

    tc = sec.get("tc")
    if "tc" in defaults and tc:
        lo_d, hi_d = defaults["tc"]
        inside = float(lo_d) <= float(tc) <= float(hi_d)
        lo = max(float(lo_d), float(tc) - TC_HALF_WIDTH)
        hi = min(float(hi_d), float(tc) + TC_HALF_WIDTH)
        if inside and hi > lo:
            notes.append(f"thickness narrowed onto {name}: "
                         f"t/c ∈ [{lo:.3f}, {hi:.3f}].")
            return {"tc": [lo, hi]}, notes
        notes.append(
            f"t/c {float(tc):.3f} of {name} lies outside this family's "
            f"thickness box [{float(lo_d):.3f}, {float(hi_d):.3f}] — the row "
            f"is left at the family default.")
        return {}, notes

    if flies_chosen_section(S):
        notes.append(
            "this family flies ONE section table, and it is the section "
            "chosen in stage 2 — its polar replaces the family's NACA "
            "anchor, so no design-box row is needed to carry it.")
    elif "tc_sec" in spec(S).param_labels:
        # WHICH family cannot hold the section is read off the DECLARED
        # design vector, not off the problem name: the coupled family alone
        # has eight registered twins (chord law × flight state × planform),
        # so an exact-name branch would tell seven of them the wrong story.
        notes.append(
            "this family picks its section from a pre-optimised (t/c × cl) "
            "CST section library as it searches (tc_sec, cl_sec are the "
            "index, not a shape), so a chosen aerofoil cannot be flown here "
            "at all. The choice from stage 2 stays advisory: it set the "
            "mission-level design point. Pick a designed-section or "
            "fixed-section wing type to fly it.")
    else:
        notes.append(
            "this family selects its section by THICKNESS off the NACA 24XX "
            "polar family, so a chosen shape cannot be flown here without "
            "disabling the t/c variable it searches. The choice from stage 2 "
            "stays advisory: it set the mission-level design point. Pick a "
            "designed-section or fixed-section wing type to fly it.")
    return {}, notes


def flies_chosen_section(S: dict, surface: str = "main") -> bool:
    """Will the run actually FLY the section stage 2 chose for ``surface``?

    True when the family declares that surface's chosen-section flag and a
    section travels on it — i.e. when the polar behind the run is the picked
    aerofoil's, not the published NACA 2412 table. The second surface is asked
    with its own flag (``api.SECTION_AFT_KEY``): a family can fly a chosen
    wing section and select its tail's by thickness.
    """
    from aerobo import api

    from . import session

    key = {"main": api.SECTION_KEY, "plate": api.SECTION_PLATE_KEY}.get(
        surface, api.SECTION_AFT_KEY)
    return (key in spec(S).flags
            and session.section_flag_value(S, surface) is not None)


def released_rows(S: dict) -> set:
    """Design-box rows the user has switched OFF.

    "Off" means UNCONSTRAINED, not "not searched": a variable the design
    still carries, whose box is nobody's opinion but the solver's. The row is
    dropped from the overrides entirely — which is exactly what the family's
    own published box is — and any section pin on it is released with it, so
    a row cannot be off and pinned at the same time.

    The numbers the user typed are kept (``S["wing"]["bounds"]``), so
    switching the row back on restores them rather than making them retype.
    """
    return {str(k) for k in (S["wing"].get("bounds_off") or ())}


def fixed_rows(S: dict) -> dict:
    """``{label: value}`` — design-box rows the user has DECIDED.

    Three states, and they are three different statements about one variable:

    * searched — a band, the published behaviour;
    * released (``bounds_off``) — still searched, but nothing this session
      says constrains it, so the solver's own box stands;
    * FIXED — not searched at all. The variable leaves the design vector
      (``api.RunConfig.pinned``), which is the only honest way to say it: a
      box of width zero is refused by the api because the samplers cannot
      draw from it and constrained BO degrades to random search on it.

    Fixed wins over released — a value is a stronger statement than "do not
    constrain it" — and a row whose label the family no longer declares is
    dropped, so a stale pin can never hold a variable nobody can see.
    """
    from . import session

    out = {}
    known = default_bounds(S)
    for k, v in (S["wing"].get("fixed") or {}).items():
        if str(k) in known and v is not None:
            out[str(k)] = float(v)
    # ...and the one this session fixes on the user's behalf: stating both END
    # CHORDS states the taper (c_tip/c_root), and a taper the optimiser could
    # still move is not a stated chord. It is a pin like any other — same
    # mechanism, same reduced search — so it is reported the same way.
    lam = session.chord_taper(S)
    if lam is not None and "taper" in known:
        out["taper"] = float(lam)
    # ...and the PAIR's two areas, for the same reason and by the same
    # mechanism: stating "the front wing is 6 m² and the rear 4 m²" states
    # the total (10 m²) and the split (0.6), and a row the optimiser could
    # still move is not a stated area. Only where BOTH are exact — a band on
    # either wing derives bands, which belong in `bounds_overrides` and not
    # here.
    for row, (kind, val) in session.pair_area_rows(S).items():
        if kind == "pin" and row in known:
            out[row] = float(val)
    # ...and the CANT of a device that does not exist. vlm.MIN_WINGLET_FRAC
    # drops a tip device shorter than 1 % of the semi-span, so a stated
    # height below it flies exactly the planar surface, and the cant row
    # beside it is then not a small effect but IDENTICALLY no effect —
    # measured flat to 3.6e-15 across its whole band, against 0.016-0.456
    # L/D at h = 0.05. A dead row is worse than an absent one: the GP spends
    # a dimension of its budget fitting noise on it. A pin derived from a
    # pin, exactly like the taper above, and reported the same way.
    from aerobo.vlm import MIN_WINGLET_FRAC
    bands = default_bounds(S)
    for h_lbl, c_lbl in DEVICE_ROWS:
        if h_lbl not in out or c_lbl not in known or c_lbl in out:
            continue
        if float(out[h_lbl]) < MIN_WINGLET_FRAC:
            # what will FLY, not what was typed: the device is dropped, so
            # the honest record of the height is zero
            out[h_lbl] = 0.0
            row = bands.get(c_lbl)
            if row is not None:
                out[c_lbl] = float(0.5 * (float(row[0]) + float(row[1])))
    return out


def fixed_value(S: dict, label: str, value: float) -> float:
    """A value a row is to be FIXED at, as typed.

    Not clamped into the row's band, and that is deliberate. A band is a
    statement about a SEARCH, and a fixed row is not searched — so the band
    has stopped being a statement about anything, while the value is the
    user's answer to a real question ("the tip chord is 180 mm"). A published
    band is a default, never a ban: clamping here would have turned the
    family's calibrated taper box into a refusal of every inverse-tapered
    wing anybody could state.

    What keeps this consistent with the api — which DOES refuse a pin outside
    the box it is handed, because two contradictory statements about one
    variable is a bug in a caller — is :func:`bounds_overrides`, which widens
    the row it belongs to so that the box sent contains the pin.
    """
    return float(value)


def bounds_overrides(S: dict) -> dict | None:
    """Sparse ``{label: [lo, hi]}`` — user edits plus the section link."""
    from . import session

    defaults = default_bounds(S)
    merged: dict = {}
    linked, _ = section_link_rows(S)
    # THE SEARCHED CANT'S FLOOR, first — before the section link, the pair's
    # areas and the user's own rows, every one of which outranks it. It is a
    # default the shell writes because nothing in the objective can pay for
    # the sign of that row (:func:`session.unpriced_cant_floor`).
    merged.update(session.unpriced_cant_floor(S))
    merged.update(linked)
    # the PAIR's two areas, where they were stated per wing: the smallest box
    # containing every (S_front, S_rear) the two answers allow. Before the
    # user's own row edits, which win — the design box is where the run is
    # finally decided (see the note below).
    for row, (kind, val) in session.pair_area_rows(S).items():
        if kind == "band" and row in defaults:
            merged[row] = [float(val[0]), float(val[1])]
    # a row the user typed WINS over the linked row: the design box view is
    # where the run is finally decided, and a control that can be overruled
    # by an earlier stage is a control that lies
    for k, v in S["wing"]["bounds"].items():
        merged[k] = [float(v[0]), float(v[1])]
    off = released_rows(S)
    out = {k: v for k, v in merged.items()
           if k in defaults and v != defaults[k] and k not in off}
    # a FIXED row's band must contain its pin, or the api refuses the pair as
    # two contradictory statements. The band is not searched — nothing draws
    # from it — so widening it costs no design freedom and keeps the two
    # halves of the same answer agreeing (see :func:`fixed_value`).
    for key, val in fixed_rows(S).items():
        row = out.get(key) or defaults.get(key)
        if row is None:
            continue
        lo, hi = float(row[0]), float(row[1])
        if not (lo <= val <= hi):
            out[key] = [min(lo, float(val)), max(hi, float(val))]
    return out or None


def search_flags(S: dict, base: dict) -> dict:
    """``base`` with the SEARCH decisions of the live policy applied.

    Kept apart from :func:`flags` for one structural reason: the measured
    recommendation is derived from the BUILT problem, and building it needs
    the physics flags — so a ``flags`` that already carried the
    recommendation would have to build the problem to know what it is. The
    physics flags are what the problem is; these two are how it is searched
    (``acqf``, and the Sobol seed size ``api.BO_N_INIT_FLAG``).
    """
    from aerobo import api

    from . import session

    eff = session.effective_wing_search(S)
    out = dict(base)
    out.pop("acqf", None)
    out.pop(api.BO_N_INIT_FLAG, None)
    out.pop(api.BO_REFUSAL_FLAG, None)
    out.pop(api.BO_FEASIBILITY_FLAG, None)
    if eff["optimiser"] != "blocks":
        # the per-block portfolio is an option OF the blocks optimiser; left
        # behind by a policy that chose another one it would ride along in
        # every flag list and every reproduce snippet, meaning nothing
        out.pop("block_optimisers", None)
    if eff["optimiser"] == "blocks":
        # the blocks runner's own sub-searches are BO runs and honour it
        out[api.BO_FEASIBILITY_FLAG] = FEASIBILITY_DEFAULT
    if eff["optimiser"] not in api.BO_ITER_OPTIMISERS:
        return out
    # ...which is `bo` and `bo_slsqp`: the two arms that run a BO loop
    # DIRECTLY, so the initial-design size, the refusal imputation and the
    # feasibility policy all reach it. `bo_slsqp` reaches it for a quarter of
    # the budget rather than all of it, and that changes nothing about which
    # flags the loop reads. Stripping them there would have been the
    # silently-dropped-flag failure, wearing the disguise of a name test.
    #
    # `handoff_phi` is deliberately NOT sent: api owns the measured default
    # (0.25) and a shell that echoed it would be a second place the number
    # lives, free to drift from the study it came from.
    if not spec(S).is_constrained and (eff.get("acqf") or "logei") != "logei":
        out["acqf"] = eff["acqf"]
    if eff.get("n_init"):
        out[api.BO_N_INIT_FLAG] = int(eff["n_init"])
    if eff.get("refusal") and eff["refusal"] != "sentinel":
        out[api.BO_REFUSAL_FLAG] = str(eff["refusal"])
    # ...and the shell's OWN default, which is not a recommendation and has no
    # control: a run whose row is SEARCHED must never come back with nothing
    # where the same row PINNED inside that band comes back with a design.
    # That is what a user hit — the wing+tail air family, the stabiliser's arm
    # free, 0 of 72 evaluations feasible; the arm fixed at 5.5 m, 53 of 69 —
    # and the cause was the search, not the physics: 92.6 % of that design box
    # is refused before any solver runs, so a 10-point Sobol seed was blind
    # more often than not, and a blind constrained BO run rides a box corner
    # for the rest of its budget (aerobo.optimize.feasible measures all three).
    # The api default stays legacy so the frozen studies keep their numbers;
    # here, where a person is waiting for an answer, the default is to find one.
    out[api.BO_FEASIBILITY_FLAG] = FEASIBILITY_DEFAULT
    return out


#: the flags the SEARCH POLICY owns, as opposed to the physics
#: (:func:`search_flags`). Kept as a name so a caller can ask "what did this
#: shell send about the PROBLEM?" without knowing which keys are search.
SEARCH_FLAG_KEYS = ("acqf", "bo_n_init", "bo_refusal", "bo_feasibility")

#: what THIS shell asks of a BO run that has not found a feasible design —
#: ``aerobo.optimize.feasible``'s strongest mode. Named here rather than
#: written into :func:`search_flags` so a reader can see the shell's default
#: beside the contract it is part of.
#:
#: ``guide`` over ``rescue``: the feasibility phase climbs the min-margin
#: surface, and a refused design reports a FLAT sentinel however far outside
#: the gate it is — so on a box of refusals the phase has nothing to climb and
#: falls back to a uniform draw every iteration. Measured on the box the
#: mission empties: 24 of 24 phase iterations blind under ``rescue``, 4 of 24
#: under ``guide`` (``RunResult.n_rescue_blind``). On the reported box the two
#: score the same at n=3 seeds (feasible 39/57/56 against 51/54/53, best
#: 19.05/18.34/18.41 against 18.72/19.52/18.44) — because there the SCREEN
#: already fixes the initial design and the phase rarely runs at all. The
#: default is the one that also works when it does.
FEASIBILITY_DEFAULT = "guide"


def physics_flags(d: dict) -> dict:
    """A built config's flags WITHOUT the ones the search policy added.

    The V3 assembly contract — an untouched session sends nothing the problem
    did not already publish — is a statement about the PHYSICS, and V3.5 put
    two search decisions in the same dict. This is how a caller states the
    contract it actually means.
    """
    return {k: v for k, v in (d.get("flags") or {}).items()
            if k not in SEARCH_FLAG_KEYS}


def section_cfg_dict(S: dict, surface: str = "main") -> dict:
    """The SECTION run this session would launch, in :func:`cfg_dict`'s shape.

    An airfoil-only session has no wing run to reproduce — stage 3 is not
    part of it — but it does have a run: the CST + live-XFOIL search stage 2
    launches. Built through ``api.airfoil_run_config`` (the same call the
    stage's own worker makes) rather than by assembling flags here, so a
    snippet cannot describe a different search from the one the button runs.
    """
    from aerobo import api

    from . import session
    from .stages.airfoil import shape_kwargs

    A = session.airfoil_state(S, surface)
    eff = session.effective_airfoil_search(S, surface)
    shape = shape_kwargs(S, surface, A["opt"], A["weights"], None)
    cfg = api.airfoil_run_config(
        **shape, optimiser=str(eff["optimiser"]), budget=int(eff["budget"]),
        seed=int(eff["seed"]), refusal=eff.get("refusal"),
        n_init=eff.get("n_init"), feasibility=FEASIBILITY_DEFAULT)
    return cfg.to_dict()


def cfg_dict(S: dict) -> dict:
    from . import session

    W = S["wing"]
    eff = session.effective_wing_search(S)
    out = {"problem_name": W["problem"],
           "mission_kwargs": mission_kwargs(S),
           "flags": search_flags(S, flags(S)),
           "optimiser": eff["optimiser"],
           "budget": int(eff["budget"]),
           "seed": int(W["seed"]),
           "bounds_overrides": bounds_overrides(S)}
    # ...and the FIXED rows only where there are any. The key is absent, not
    # None, on an untouched session: this dict is compared field-for-field
    # against the V2 shell's to hold the "same choices, same config" contract
    # (tests/test_v3_pipeline.py), and a key V2 has never heard of would
    # break it while meaning nothing.
    pinned = fixed_rows(S)
    if pinned:
        out["pinned"] = pinned
    # ...and the DESIGN THIS RUN STARTS FROM, where the session kept one.
    # Absent on every untouched session for the same reason ``pinned`` is: an
    # unarmed key would differ from the V2 config field-for-field while
    # meaning nothing. A seed is evaluated FIRST and never screened away
    # (api.RunConfig.x_seed), which is what makes "continue from the design
    # that came closest" a continuation rather than a fresh roll of the dice.
    seed_x = S["wing"].get("x_seed")
    if seed_x:
        out["x_seed"] = [float(v) for v in seed_x]
    return out


def build_cfg(S: dict, seed: int | None = None):
    from aerobo import api

    d = cfg_dict(S)
    if seed is not None:
        d["seed"] = int(seed)
    return api.RunConfig(**d)


def effective_bounds(S: dict) -> dict:
    """``{label: ([lo, hi], source)}`` — what the run will actually search.

    ``source`` is ``"default"``, ``"section"`` (pinned by stage 2),
    ``"mission"`` (a SIZE row the shell derived from the mission — the same
    place a typed row is stored, but nobody typed it, and the mission may
    still move it), ``"recommended"`` (a band MEASURED for this mission —
    ``aerobo.recommend`` — which nobody typed either, and which the mission
    also moves), ``"user"`` (typed in the design box) or ``"released"``
    (the row was switched off, so nothing constrains it and the solver's own
    box stands), which is what the box view colours its rows by.
    """
    from . import session

    defaults = default_bounds(S)
    linked, _ = section_link_rows(S)
    off = released_rows(S)
    written = S["wing"].get("bounds_source") or {}
    # the PAIR's two areas, where they were stated per wing: the band they
    # derive is the user's answer one step removed, and it has to appear here
    # as well as in `bounds_overrides` — this is what the box VIEW draws, and
    # a view that showed the published row while the run searched another
    # would be this repo's own "the box shown is the box searched" failure.
    pair = {row: band for row, (kind, band)
            in session.pair_area_rows(S).items() if kind == "band"}
    floor = session.unpriced_cant_floor(S)
    out = {}
    for k, dv in defaults.items():
        row, source = dv, "default"
        if k in floor:
            # ...and it says which it is: the box shown is the box searched,
            # so a row the shell floored may not read "default"
            row, source = list(floor[k]), "unpriced"
        if k in linked:
            row, source = linked[k], "section"
        if k in pair:
            row, source = [float(pair[k][0]), float(pair[k][1])], "user"
        if k in S["wing"]["bounds"]:
            v = S["wing"]["bounds"][k]
            row = [float(v[0]), float(v[1])]
            source = {"shell": "mission",
                      "recommended": "recommended"}.get(written.get(k),
                                                        "user")
        if k in off:
            row, source = dv, "released"
        out[k] = (row, source)
    return out
