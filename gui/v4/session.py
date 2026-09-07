"""V4's session state — V3's, plus stage 5 and stage 6.

Every name :mod:`gui.v3.session` defines is re-exported here unchanged, so a
stage module written against ``session.X`` does not care which shell mounted
it. The tables are then OVERRIDDEN — the stage list, the view table, the
labels — along with the functions that read them (:func:`make_session`,
:func:`stage_states`, :func:`stage_label`, :func:`stage_visible`).

:func:`stage_visible` is the newest of those, and it is here rather than in
V3 because V3's version cannot express the rule: it hides both stages in an
airfoil-only session only as a side effect of that mode being two stages
long, and it has no way to say that a CAR REAR WING has no stage 5 and no
stage 6 either. The medium predicate it asks
(:func:`gui.v3.session.car_wing`) does live in V3, where the car's other
helpers are; only the stage rule is V4's.

Copying the namespace rather than editing V3's is the whole point. V3 keeps
four stages — open the V3 shell and there is no Controls tab and no Flight
tab anywhere in it — and V4 gets every fix V3's session ever receives,
because it is the same module object underneath.

The V3 functions copied in below still close over V3's OWN globals, which is
why the two that read ``STAGES``/``VIEWS``/``STAGE_LABELS`` are wrapped
rather than merely shadowed: ``_v3.make_session`` builds a five-stage ``ui``
sub-dict no matter which module you call it from.
"""

from __future__ import annotations

import copy
import sys as _sys

from gui.v3 import session as _v3

# ---- re-export V3's whole namespace (functions, constants, private helpers
# alike: a stage module is entitled to reach for anything V3's own stages
# can reach for). Dunders are excluded so this module keeps its own identity.
_SELF = _sys.modules[__name__]
for _name, _value in vars(_v3).items():
    if not (_name.startswith("__") and _name.endswith("__")):
        setattr(_SELF, _name, _value)
del _name, _value

#: the stages, in pipeline order. The first five are V3's, unchanged and in
#: the same order; 5 and 6 are what V4 is.
STAGES = (*_v3.STAGES, "controls", "flight")

#: stage -> its views (work-area tabs), each ``(key, label, tree icon)``.
#: V3's table is copied, not mutated: ``dict(_v3.VIEWS)`` shares the tuples
#: but not the mapping, so adding a key here cannot add a tab to V3.
VIEWS: dict = dict(_v3.VIEWS)

# Stage 5. The design has been SIZED and SCORED by now; what it has never
# had is anything to control it with. Ailerons are cut out of the wing
# the search produced, and the vertical surface is ADDED — the lattice
# had no side-force-carrying surface at all before this, so ``Cn_beta`` was
# exactly zero for every configuration this package builds.
#
# FOUR TABS, NOT FIVE. There was a "Stability check" that printed four of
# the Derivatives table's own rows again — same deck object, same values,
# one tab-click apart — plus the static margin and a red/green colour. The
# colour and the static margin are what it uniquely had, so both moved into
# the table, and the question "is this stable?" is answered once, where the
# numbers it is asked about already are.
VIEWS["controls"] = (("surfaces", "Ailerons & elevator", "swap_horiz"),
                     ("propulsion", "Propulsion", "rocket_launch"),
                     ("vertical", "Fin & rudder", "vertical_align_center"),
                     ("derivatives", "Derivatives & stability",
                      "table_rows"))

# ...and stage 6, where it is flown. Note what is NOT a tab here: mass,
# thrust and CG. They are not set-up questions asked before the simulation —
# they are levers beside the moving picture, because the whole point of
# flying a design is feeling what happens when you move one.
VIEWS["flight"] = (("fly", "Fly", "flight"),
                   ("setup", "Flight condition", "tune"),
                   ("traces", "Traces", "show_chart"),
                   ("modes", "Modes", "vibration"))

STAGE_LABELS: dict = dict(_v3.STAGE_LABELS)
STAGE_LABELS["controls"] = "5  Controls"
STAGE_LABELS["flight"] = "6  Flight"

#: the two stages V4 adds, as a set, for anything that needs to ask "is this
#: one of mine" without hard-coding the pair twice.
V4_STAGES = ("controls", "flight")


# --------------------------------------------------------------- V4 defaults
#: Stage 5's starting answers. Every one of them is a QUESTION the earlier
#: stages never asked, which is why they get a stage rather than a card
#: bolted onto Results: the wing was sized against a mission, and nothing in
#: that mission said anything about rolling it.
CONTROLS_DEFAULTS: dict = {
    # ailerons: the outboard band, as a fraction of the semi-span the search
    # actually produced, and the hinge line as a fraction of local chord
    "aileron": {"on": True, "span_from": 0.60, "span_to": 0.98,
                "chord_frac": 0.25},
    # NO FLAP. It was a switch that was off in every session anybody
    # opened, and turning it on bought a lift column the stall clip ate
    # and a drag it never paid, so the honest thing it could do was move
    # the stall to a LOWER incidence. A control that cannot help is one
    # more thing on four screens to read past.
    "elevator": {"on": True, "chord_frac": 0.40},
    # WHERE THE THRUST VECTOR STARTS. A stage-5 question because it is a
    # property of the airframe, not of a particular flight, and asked as a
    # pair because the interesting answer is not a number: "through the CG"
    # means NO pitching moment at all, which is the honest default for a
    # design whose propulsion this package does not model. Turning it off
    # exposes the offset, in metres BELOW the CG, and the throttle becomes a
    # pitch input. There is no x or y offset to ask about —
    # ``sixdof.Propulsion`` says why in one place.
    "thrust": {"through_cg": True, "z_below_cg_m": 0.0},
    # the vertical surface. WHERE IT GOES IS ONE NUMBER, and that number is
    # the LEADING EDGE station in metres — the same coordinate the wing and
    # the stabiliser are drawn at on this stage's side elevation, so "at the
    # tail" and "between the wing and the tail" are places the user can see
    # and type rather than words in a menu. (There WAS such a menu. It stored
    # "tail" or "between" and nothing anywhere read it: the fin went where
    # the station said, or to the tail station when the station was blank.
    # A control that cannot move the model is worse than no control.)
    "vertical": {"on": True, "height_m": None,
                 "chord_m": None, "x_le_m": None, "rudder_chord_frac": 0.40,
                 # the sideslip full rudder is asked to HOLD [deg]. The
                 # rudder recommendation is solved against it; blank means
                 # the stated default, and nothing refuses a chord.
                 "rudder_beta_deg": 10.0,
                 # a fin BELOW the craft is how the strut is asked for; in
                 # water that is the only place it can go, because the foil
                 # does not pierce the surface
                 "ventral": False},
    # the two sizing TARGETS the fin can be solved for, and the answer to
    # "does any fin size fix the spiral". None each: nothing is solved for
    # until the user asks, and every one of them is cleared whenever the
    # design changes, because a stale recommendation is worse than none.
    "size_Vv": None,
    "size_Cnb": None,
    "spiral_scan": None,
    "dihedral_scan": None,
    # ...and why a sizing button did nothing, when it did nothing. A target
    # outside the band the search covers is REPORTED, not clamped, and a
    # report needs somewhere to live between the press and the repaint.
    "size_note": None,

    # ---- the worker keys, declared rather than made on first use. They are
    # what the stage BUILT, not what the user answered, but they are listed
    # here for the reason every other worker dict in this shell is: this is
    # the file a reader goes to for "what is in S['controls']", and a key
    # that only exists once a view has been opened is not in it.
    #
    # ``deck_report`` and ``deck_point`` together are the identity of the
    # aeroplane the deck describes — the design, and the air it is flown in
    # — and :func:`gui.v4.stages.controls.build`'s ``_ensure_deck`` rebuilds
    # when either moves.
    "deck": None,
    "fm": None,
    "error": None,
    "deck_report": None,
    "deck_point": None,
    "rec_cache": {},
}

#: Stage 6's starting answers.
#:
#: THE SPLIT IN THIS DICT IS THE STAGE'S ONE IDEA. ``live`` holds the three
#: numbers that are not questions to answer before flying — weight, thrust
#: and where the CG sits — because the only way to learn what they do is to
#: move one while the aeroplane is in the air and watch the picture change.
#: They start as ``None`` and are filled at arming time from the design
#: itself, so the aeroplane always begins as the one the search produced.
#: Everything else is the flight CONDITION: it is set before the run,
#: because changing it means re-trimming, which is a new flight.
FLIGHT_DEFAULTS: dict = {
    "live": {"thrust_n": None,    # None until armed -> the trim thrust
             "mass_kg": None,     # None until armed -> the design's weight
             "x_cg_m": None},     # None until armed -> the design's CG
    # NO INERTIA ANSWERS HERE. There were six — a mode, two mass fractions
    # and the three moments — and nothing read one of them. What the
    # simulation flies is ``sixdof.estimate_inertia``'s tensor, built inside
    # ``build_flight_model`` and scaled by the mass lever, and stage 6
    # REPORTS it with its basis named. State declared before a control and a
    # path into the model exist is state that survives File > New session
    # saying something that is not true.
    # NO "thrust_z_m". The thrust LINE is a property of the airframe and it
    # is answered in stage 5 (``CONTROLS_DEFAULTS["thrust"]``), which is
    # where the fin, the ailerons and the elevator are answered. It was a
    # field here as well, so the same design had two thrust lines depending
    # on which stage had last been opened — and stage 5's deck knew nothing
    # about the one stage 6 held.
    # THE STOP W RUNS INTO. None = 2.5x the trim thrust — a default, not a
    # ban, so an over-powered design is a number the pilot types rather than
    # a limit they cannot reach. It is drawn beside the throttle bar, and
    # the current thrust is drawn beside IT.
    #
    # There is no matching floor: the throttle bottoms out at zero
    # (``flight.THRUST_MIN_N``), because a stopped engine makes no thrust
    # and this stage has no propeller to reverse.
    "thrust_max_n": None,
    # THE TWO CONDITION LEVERS. They are not in ``live`` because moving one
    # means a new TRIM — the aeroplane is re-trimmed at the new speed or
    # altitude rather than merely shoved to it — but they are beside the
    # picture with the live ones, because "fly it at 20 m/s" is a thing a
    # pilot asks in the air and not a form to go back to.
    "V_trim": None,               # None = the mission's own speed
    "altitude_m": None,           # None = the mission's own altitude
    # the canned excitation that is running, by key in gui.v4.manoeuvre.
    # It stays set after the script has run out, because the trace being
    # looked at is still that manoeuvre's.
    "manoeuvre": None,
    #: named modes of the linearised motion, from gui.v4.modes. Recomputed
    #: when a lever moves, NOT once a frame: it is a 13x13 Jacobian and an
    #: eigensolve, 1.58 ms, which is a fifth of a frame at 120 Hz.
    "modes": None,
    "CD0": None,                  # None = the design report's own CD0
    "oswald_e": None,
    "stall_on": True,
    # WHICH VIEW. "game" is the chase camera over a moving world; the
    # aeroplane is identical in both, and so is the deck it flies —
    # "engineering" only turns the world off, puts the fixed camera back and
    # shows the CG and neutral-point markers again.
    "mode": "game",
    "hud": True,
    "crashed": False,
    # ...and the sentence the crash left on screen, which outlives the
    # frame that wrote it and is cleared by a RE-TRIM. The frame path owns
    # the banner, so this is where a stop rule's message has to live for the
    # engineering view — where the HUD's own warning line is switched off —
    # to say anything at all.
    "crash_banner": None,
    # NO "dt". The simulation reads a WALL CLOCK and integrates in fixed
    # substeps (:data:`gui.v4.stages.flight.SUBSTEP_S`); a stored frame time
    # is what made it fly at half speed, because it was 1/60 while the timer
    # fired at 1/30 and nothing connected the two.
    "speed": 1.0,                 # wall-clock multiplier: 1.0 IS real time
    # live stick, in DEGREES, so the view and the state agree on units
    "stick": {"elevator": 0.0, "aileron": 0.0, "rudder": 0.0},
    # what each axis is being asked for, in [-1, 1]. DERIVED, never written
    # directly: it is the sum of every device holding that axis — the
    # keyboard's ±1 and the gamepad's fraction — clipped. The stick springs
    # back to centre on release, so what a press sets is a demand and not a
    # deflection (:mod:`gui.v4.stick`).
    "hold": {"elevator": 0.0, "aileron": 0.0, "rudder": 0.0, "thrust": 0.0},
    # the GAMEPAD's half of that sum, kept apart from the keyboard's for the
    # same reason the keys are kept as a set: two hands on one control are
    # not a last writer. A pad released to centre must not centre an axis
    # whose key is still down (:mod:`gui.v4.pad`).
    "pad": {"elevator": 0.0, "aileron": 0.0, "rudder": 0.0, "thrust": 0.0},
    # is one actually talking to us? Reported by the browser on the connect
    # and disconnect edges only, and shown beside the keys so a pad that is
    # not being seen says so instead of feeling broken.
    "pad_on": False,
    # the KEYS physically down, by name. ``hold`` is derived from this and
    # not written directly: two keys on one axis are opposite demands on ONE
    # control, and a last-writer-wins sign centred the stick when the pilot
    # released the key they were no longer using. Declared here rather than
    # made on first use, like every other worker key, so File > New session
    # cannot leak the previous flight's keyboard.
    "keys_down": set(),
    "running": False,
}


def make_session(medium: str = "air") -> dict:
    """V3's session, plus the two stage workspaces and their ui entries.

    The sub-dicts are declared HERE rather than created on first use for the
    same reason every V3 worker dict is: File > New session refills the
    top-level sub-dicts in place, and a key that only existed once a stage
    had been opened would survive a reset carrying the previous design's
    controls.
    """
    S = _v3.make_session(medium)
    S["controls"] = copy.deepcopy(CONTROLS_DEFAULTS)
    S["flight"] = copy.deepcopy(FLIGHT_DEFAULTS)
    for stage in V4_STAGES:
        S["ui"]["tab"][stage] = VIEWS[stage][0][0]
        S["ui"]["expanded"][stage] = False
    return S


def stage_label(S: dict, stage: str) -> str:
    """V3's label rules, extended to the two stages V3 has no name for."""
    if stage in V4_STAGES:
        return STAGE_LABELS[stage]
    return _v3.stage_label(S, stage)


def views_of(S: dict, stage: str) -> tuple:
    """V3's table for V3's stages, V4's own for the two it adds.

    Stage 5's tab list used to depend on the MEDIUM: a foiler was offered
    three of the five, because nothing on it is hinged. It is now offered
    none of them — :func:`free_flight` hides the whole stage — so there is
    one list again and no session can be looking at a tab its medium does
    not own.
    """
    if stage not in V4_STAGES:
        return _v3.views_of(S, stage)      # V3 owns it and its table
    return VIEWS[stage]


def free_flight(S: dict) -> bool:
    """Does this session design something that FLIES, free, on its own?

    Three sessions out of four do not, and each for its own reason:

    * an **airfoil-only** session has no vehicle at all;
    * a **car rear wing** is bolted to a car — it makes a load rather than
      carrying one, and has no free-flight degrees of freedom;
    * a **foiling craft** is held by the water and driven by a rig. Nothing
      on it is hinged (its pitch is an all-moving stabiliser set at design
      time, its only vertical surface is the mast), and the mast's quarter
      chord stands AHEAD of the CG, which measures ``Cn_beta = -0.196`` —
      so a six-degree-of-freedom aeroplane simulation of it would be
      answering a question the craft has not got, with a number
      :func:`aerobo.wing_score.spiral_refusal` already refuses.

    Stages 5 and 6 are exactly the stages that only make sense when this is
    true, which is why they ask it here rather than each testing a medium.
    """
    return not (_v3.airfoil_only(S) or _v3.car_wing(S)
                or S.get("medium") == "water")


def stage_visible(S: dict, stage: str) -> bool:
    """V3's rules, plus the one V3 has no vocabulary for.

    V3's own :func:`gui.v3.session.stage_visible` already hides both of
    these in an airfoil-only session, but only as a side effect of its first
    branch (that mode is two stages long), and it ends in ``return True`` —
    so every other configuration gets them. A CAR REAR WING and a FOILING
    CRAFT are the configurations that must not: see :func:`free_flight`.

    The V4 branch comes FIRST and does not delegate, because delegating
    would land on that ``return True``. Everything else does delegate, and
    that is load-bearing: on the track V3 still hides the second surface's
    and the fin's section stages and SHOWS the endplate's, and a rule
    written one branch too wide would take the endplate's stage with it.

    The car's medium predicate lives in V3
    (:func:`gui.v3.session.car_wing`), where the car's other helpers are;
    only the stage rule is V4's, because a V3 file may not name these two
    stages — see the leak test in ``tests/test_v4_stages.py``.
    """
    if stage in V4_STAGES:
        return free_flight(S)
    return _v3.stage_visible(S, stage)


def stage_states(S: dict) -> dict:
    """V3's gates, plus stage 5's and stage 6's.

    Stage 5 needs a DESIGN to cut control surfaces out of, so it gates on
    the run record exactly as Results does. Stage 6 needs stage 5 to have
    produced a derivative deck: flying a design with no controls is not a
    degraded simulation, it is a different (and dishonest) one.
    """
    out = _v3.stage_states(S)

    # an AIRFOIL-ONLY session has no vehicle at all, so it has neither.
    # Hidden (:func:`stage_visible` already returns False for both) AND
    # locked, because hiding is only cosmetic — the shell mounts every stage
    # and ``Ctx.select`` is what actually refuses to open one.
    if _v3.airfoil_only(S):
        why = ("this session designs a section only — a control surface and "
               "a flight simulation belong to a vehicle, and this one has "
               "none")
        out["controls"] = out["flight"] = ("locked", why)
        return out

    # ...and a CAR REAR WING, for the same reason in a different shape: it
    # has a vehicle, and the vehicle is a car. BEFORE the run-record branch
    # below, or the lock would be defeated by the very run that branch's
    # reason invites — a finished car run would report "ready" and open a
    # stage that rebuilds the downforce surface right way up as an
    # 18 kg aeroplane at a speed nobody chose.
    if _v3.car_wing(S):
        why = ("this design is a car rear wing — it is bolted to a car, "
               "carries no weight and has no free-flight degrees of "
               "freedom, so there is nothing here to roll, to trim or to "
               "fly. Stage 4 is where the track pipeline ends")
        out["controls"] = out["flight"] = ("locked", why)
        return out

    # ...and a FOILING CRAFT, which is the third shape of the same thing and
    # the one that used to get a reduced stage 5 instead of none. It has a
    # vehicle and the vehicle does not fly free: the water holds it and a
    # rig drives it. Nothing on it is hinged, and its one vertical surface
    # is the mast, whose quarter chord stands AHEAD of the CG — measured
    # Cn_beta = -0.196, a directional stiffness of the wrong SIGN that no
    # criterion in this package will buy back
    # (``RESULTS_SESSION74_LATERAL_EVERYWHERE.md``). So a six-degree-of-
    # freedom aeroplane simulation of it would not be a rough answer; it
    # would be an answer to a different craft.
    if S.get("medium") == "water":
        why = ("this design is a foiling craft — the water holds it and a "
               "rig drives it, nothing on it is hinged, and its only "
               "vertical surface is the mast, which stands ahead of the CG "
               "(Cn_beta = -0.196). There is no free-flight aeroplane here "
               "to roll, to trim or to fly. Stage 4 is where the water "
               "pipeline ends")
        out["controls"] = out["flight"] = ("locked", why)
        return out

    if (S.get("run") or {}).get("record") is None:
        why = ("size and fly a wing first — a control surface is cut out of "
               "a planform, and there is no planform until stage 3 has run")
        out["controls"] = out["flight"] = ("locked", why)
        return out

    C, F = S.get("controls") or {}, S.get("flight") or {}
    # DONE MEANS "THERE IS A DECK FOR THIS DESIGN", and the second half is
    # the load-bearing one. ``C["deck"]`` outlives the run that produced it,
    # so a re-run left stage 5 reading "done" and stage 6 unlocked on a deck
    # built for the PREVIOUS aeroplane — the tree's two most confident
    # words, about a design that no longer exists. The stage rebuilds when
    # it is opened (``controls.py::_ensure_deck``), so the honest state
    # before that is "ready", with the re-run named as the reason.
    fresh = (C.get("deck") is not None
             and C.get("deck_report") is (S.get("run") or {}).get("report"))
    # WHY THE STAGE IS WORTH OPENING — a tree node that is merely "ready"
    # says nothing about whether opening it would change anything.
    why = ("the design has no ailerons and no fin yet "
           "— with no vertical surface its yaw stiffness is "
           "exactly zero, which is a hole in the geometry rather "
           "than a stable aeroplane")
    if C.get("deck") is not None and not fresh:
        why = ("the design changed — stage 5 rebuilds its deck for the new "
               "one when you open it, and stage 6 flies what it builds")
    out["controls"] = ("done" if fresh else "ready", "" if fresh else why)
    if not fresh:
        out["flight"] = ("locked",
                         "build the control surfaces first — stage 6 flies "
                         "the derivative deck stage 5 produces")
    elif F.get("running"):
        out["flight"] = ("running", "")
    elif F.get("history"):
        out["flight"] = ("done", "")
    else:
        out["flight"] = ("ready", "")
    return out


def extra_running(S: dict) -> tuple:
    """Stage 6's flight loop, for the toolbar.

    V3's Stop button counts the run manager, the reach and the four section
    searches; none of them is a six-degree-of-freedom integration at 30 Hz.
    So while the aeroplane was flying, the tree painted ``6 Flight —
    running`` and the one button whose job is to stop things was greyed out,
    and ``Solution > Stop`` fell through to "nothing is running" and left it
    flying. One list answers both.
    """
    return ((bool((S.get("flight") or {}).get("running")),
             "flight_run", (False,)),)


def stage_badge(S: dict, stage: str) -> str | None:
    """What the two V4 stages HOLD, in the tree, in one short reading.

    V3's badge falls through to the wing search's objective for any stage it
    does not name, so stages 4, 5 and 6 all painted the same ``f -1.234`` —
    two of them about a search that has nothing to do with what they show.

    Stage 5 holds a DECK, and the one number that says whether building it
    changed anything is the yaw stiffness the design had none of. Stage 6
    holds a FLIGHT, and what it has is how long it lasted.
    """
    if stage == "controls":
        D = (S.get("controls") or {}).get("deck")
        return "" if D is None else f"Cn_b {float(D.Cn_beta):+.4f}"
    if stage == "flight":
        F = S.get("flight") or {}
        t = F.get("history_t") or []
        if F.get("error"):
            return "down"
        return f"{float(t[-1]):.0f} s" if t else ""
    return None


def stage_available(S: dict, stage: str) -> bool:
    return stage_states(S)[stage][0] != "locked"
