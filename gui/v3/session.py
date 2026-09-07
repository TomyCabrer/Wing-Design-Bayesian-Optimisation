"""V3 session state — the four pipeline stages and what flows between them.

The V3 shell is a PIPELINE: a mission is stated, a section is chosen for
that mission, a wing type is chosen and optimised around that section, and
the result is reported. This module owns the state each stage writes and
the rules for what the next stage may read, so no stage has to reach into
another's widgets:

    mission  ->  load, speed, fluid state, area  ->  CL design
    mission  ->  how many surfaces (tandem? tail?)  ->  how many sections
    airfoil  ->  aspect-ratio estimate -> MAC, Re  ->  chosen section
    airfoil  ->  a section PER SURFACE (:func:`surface_design_point`)
    airfoil  ->  chosen section (CST weights, t/c)  ->  wing stage
    wing     ->  aerobo.api.RunConfig (gui/v3/config.py)  ->  results

Note where the aspect ratio sits: the mission does not carry one at all
(:data:`DEFAULT_ASPECT_RATIO`). Stage 2 estimates one because a section
needs a chord; stage 3 decides the one that is flown.

Everything derived is derived HERE, from ``aerobo.api`` — never in a stage
and never twice, so the Reynolds number the screen runs at is the same
number the properties grid shows.

The wing sub-state deliberately keeps V1's key names (``choices``,
``problem``, ``bounds``, ``flags``, ``mission_edits``, ``prop`` …) because
the V1 builder helpers (``derive_problem``, ``winglet_flags``, …) are the
single source of what combinations exist; V3 renders them, it does not
restate them.

Import-safe: aerobo / gui.nice_app imported lazily inside the functions.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results" / "gui_runs"

#: where a CAD export lands when the user has not said otherwise. Inside the
#: repo's own results tree rather than in a browser download folder, because
#: an export is one of THIS session's outputs and because the OpenVSP bridge
#: has to be handed a real path (``aerobo.vsp``) — a download is a file the
#: shell can neither name nor open afterwards.
DEFAULT_EXPORT_DIR = REPO_ROOT / "results" / "cad_export"

#: THE SHELL MEMOISES ITS EVALUATIONS (:mod:`aerobo.eval_cache`), and the
#: studies do not. "Keep going" is one longer run of the same search — the
#: prefix is genuinely re-flown, which is what makes it contain the run it
#: lengthens — so without this a continuation to 60 pays for the first 40
#: evaluations a second time and the user watches the progress bar start at 1
#: with nothing to show for the wait. The objective is a pure function of the
#: design vector, so nothing the run reports changes; only the seconds do,
#: and the record says how many evaluations came off disk
#: (``RunResult.eval_cache``). ``AEROBO_EVAL_CACHE=0`` in the environment
#: turns it off for a whole process.
EVAL_CACHE: bool = True

#: DOES THE VEHICLE HAVE A VERTICAL TAIL — the default, and it is True.
#:
#: Every air configuration this package builds has flown one since V5: the
#: drag book charges it, the lattice carries its panels, the CAD exports it
#: and the flight model's whole yaw stiffness comes from it. Defaulting it
#: off would make the published aeroplane the exception. What the switch
#: buys is the ability to say NO — a flying wing, a design whose directional
#: stability comes from sweep, or a V-tail (which answers it by construction
#: and cannot be overridden).
FIN_DEFAULT: bool = True

#: the stages, in pipeline order. ``airfoil_aft`` is stage 2 asked a SECOND
#: time, for the second surface: it appears only where the configuration has
#: one (:func:`stage_visible`) and is never a gate on stage 3, because a
#: second surface with no section of its own flies the wing's.
#: ``airfoil_fin`` is stage 2 asked a THIRD time, for the VERTICAL surface.
#: It appears only where the vehicle HAS a fin — a mission question since V5,
#: because whether there is a vertical tail decides how many surfaces there
#: are to design, exactly as the second surface does. Its section is the
#: simplest of the three: a fin at zero sideslip must make no side force, so
#: the shape is symmetric and only its THICKNESS is a choice.
#: ...and the CAR ENDPLATE's, on the one family where the plate is a designed
#: part rather than a fence. Same argument as the fin's, one vehicle across: a
#: rear wing's plates are a surface with their own chord, their own Reynolds
#: number and their own job, and until this stage existed the only thing that
#: could be said about their aerofoil was which CONSTRUCTION family they were
#: built in — flat, rounded or shaped — under which every section of a given
#: thickness scores identically. Symmetric, like the fin's, and for the same
#: physics: a cambered plate at zero toe carries a side force.
STAGES = ("mission", "airfoil", "airfoil_aft", "airfoil_fin", "airfoil_plate",
          "wing", "results")

#: stage -> the views (work-area tabs) it owns, in tab order, each as
#: ``(key, label, tree icon)``
VIEWS: dict[str, tuple[tuple[str, str, str], ...]] = {
    "mission": (("operating", "Operating point", "tune"),
                ("point", "Design point", "speed"),
                # WHICH optimiser and HOW MANY evaluations — one question,
                # one place, for every stage that searches (V3.5)
                ("search", "Search & budget", "bolt")),
    "airfoil": (("screen", "Library screening", "tune"),
                ("ranking", "Ranking", "format_list_numbered"),
                ("section", "Section", "gesture"),
                ("optimise", "Shape optimisation", "auto_graph")),
    "wing": (("type", "Wing type", "category"),
             ("box", "Design box", "crop_free"),
             ("solver", "Solver", "settings"),
             ("run", "Convergence", "show_chart")),
    # NO section view here. Stage 2 (and 2.5) is where a section is chosen,
    # shown and shaped; repeating its shape and its polar as a tab of the
    # RESULTS read as though the result contained a second answer about the
    # aerofoil. The designed shape is still computed (``section_report``) —
    # the geometry view draws its outline and the section ``.dat`` export is
    # built from it — it simply has no tab of its own.
    "results": (("summary", "Summary", "summarize"),
                ("geometry", "Geometry", "view_in_ar"),
                ("loading", "Loading", "ssid_chart"),
                ("log", "Evaluations", "table_rows")),
}

#: the second surface's stage is stage 2 AGAIN — the same four views, built
#: from the same module against its own workspace. Sharing the tuple is the
#: point: one question asked twice cannot drift into two different forms.
VIEWS["airfoil_aft"] = VIEWS["airfoil"]

#: ...and the FIN's stage is stage 2 a third time, with one view fewer. There
#: is no "Ranking" for it: ranking screens a LIBRARY against a lift-carrying
#: criterion, and a symmetric fin section is chosen on thickness and drag
#: alone. Offering a ranking tab would be offering a question the surface
#: does not ask.
#: ...and the VERTICAL STABILISER's is stage 2 a third time, the SAME four
#: views. It was briefly one cut-down view; the argument for that was half
#: right (a fin has no design lift coefficient) and half wrong (it does have
#: a shape to search — a symmetric CST section at cl = 0, four variables
#: instead of eight). Sharing the tuple is the point, for the same reason
#: the aft surface shares it.
VIEWS["airfoil_fin"] = VIEWS["airfoil"]

#: ...and the ENDPLATE's is stage 2 a fourth time, the same four views again.
#: A plate has a library to screen (the symmetric members) and a shape to
#: search (a symmetric CST section at cl = 0), which is exactly the fin's set
#: of questions — so it gets the fin's set of views, by identity.
VIEWS["airfoil_plate"] = VIEWS["airfoil"]

STAGE_LABELS = {"mission": "1  Mission", "airfoil": "2  Airfoil",
                "airfoil_aft": "2.5  Airfoil",
                "airfoil_fin": "2.7  Vertical stabiliser",
                "airfoil_plate": "2.8  Endplate", "wing": "3  Wing",
                "results": "4  Results"}

#: surface -> the stage that chooses ITS section, and back. The stage owns
#: the surface (one question, one place): there is no target switch to leave
#: pointing at a surface the shell is not showing.
SURFACE_STAGES = {"main": "airfoil", "aft": "airfoil_aft",
                  "fin": "airfoil_fin", "plate": "airfoil_plate"}

#: ...and where each SECONDARY surface's chosen section is stored, under the
#: main workspace. The wing is absent on purpose: its section is ``section``
#: and it has no fallback, which is what makes it the one the others inherit.
#: One table, so a fourth surface cannot arrive with a key only half the
#: readers know about.
SECTION_KEYS = {"aft": "section_aft", "fin": "section_fin",
                "plate": "section_plate"}
STAGE_SURFACE = {v: k for k, v in SURFACE_STAGES.items()}


def stage_label(S: dict, stage: str) -> str:
    """The tree/breadcrumb label, which NAMES the surface where there are two.

    "2  Airfoil" and "2.5  Airfoil" say nothing about which surface is being
    designed, and the whole reason the second stage exists is that the two
    answers differ. So both grow a suffix as soon as a second surface is on
    the vehicle — and the suffix is the configuration's own word for it
    ("tail", "elevator", "rear wing"), never a generic "aft".
    """
    label = STAGE_LABELS[stage]
    if airfoil_only(S):
        # stage 1 asks a different question in this mode, so it does not
        # keep a name that describes the other one
        return "1  Flow" if stage == "mission" else label
    if stage not in STAGE_SURFACE:
        return label
    # THE VERTICAL SURFACE'S STAGE IS NAMED AFTER THE VERTICAL SURFACE. It
    # fell through to the rule below, which suffixes the SECOND surface's
    # name, so on any configuration with both it read "2.7  Vertical
    # stabiliser · rear wing" — the fin's stage labelled with the tailplane's
    # name. And a boat's is a strut (:func:`surface_name`), so the number is
    # kept and the noun is asked for rather than restated.
    if STAGE_SURFACE[stage] in ("fin", "plate"):
        # ...and the same rule for the ENDPLATE: it is not the second
        # surface, so suffixing it with the second surface's noun would
        # label the plate's stage "rear wing".
        number = label.split("  ")[0]
        surface = STAGE_SURFACE[stage]
        return f"{number}  {surface_name(S, surface).capitalize()}"
    aft = aft_surface(S)
    if not aft:
        return label
    return f"{label} · {'wing' if stage == 'airfoil' else aft}"


def views_of(S: dict, stage: str) -> tuple:
    """The views this SESSION offers on ``stage``, in tab order.

    :data:`VIEWS` is the total table — every container is mounted from it, so
    a view can never be missing when a session changes shape under it — and
    this is what the tree, the tab strip and the breadcrumb read. Here it
    answers the whole table: nothing in this pipeline offers a different set
    of tabs for a different vehicle. A shell that does overrides it, the way
    :func:`stage_visible` is overridden for a stage that some vehicles have
    not got.
    """
    return VIEWS[stage]


def extra_running(S: dict) -> tuple:
    """Loops a SHELL mounts that the toolbar has to see and be able to stop.

    Each entry is ``(is_running, action, args)``. V3 mounts none: every
    search it can start is one the toolbar already counts by name.

    It exists because a shell that adds a stage with a loop of its own has
    two chrome controls to keep honest — the Stop button's enabled state and
    ``Solution > Stop`` — and neither of them should have to learn a stage
    list. A shell overrides this; nothing here knows what such a stage is.
    """
    return ()


def stage_badge(S: dict, stage: str) -> str | None:
    """The tree badge for a stage this module's own rules cannot answer for.

    ``None`` means "the shell's rules decide", which is every stage V3 has.
    The badge is the one place the tree says what a stage HOLDS, so a stage
    a later shell adds must either answer for itself here or show nothing —
    falling through to the wing search's objective paints the same number on
    three different nodes.
    """
    return None


def stage_visible(S: dict, stage: str) -> bool:
    """Is this stage part of THIS configuration's pipeline at all?

    Two ways a stage can be absent. The second surface's stage is absent
    whenever the vehicle has no second surface that can be handed a section
    of its own — a stage for a surface that does not exist is worse than no
    stage. And an AIRFOIL-ONLY session (:func:`airfoil_only`) has no vehicle
    at all, so the wing and its results are not stages it is part way
    through: they are not its work.

    Hiding is only half of it — the shell mounts every stage either way, and
    ``Ctx.select`` gates on :func:`stage_states` — so both halves are said
    here and there, and a hidden stage always carries a locked reason.
    """
    if airfoil_only(S):
        return stage in ("mission", "airfoil")
    if stage == "airfoil_aft":
        return aft_surface(S) is not None
    if stage == "airfoil_fin":
        return fin_surface(S)
    if stage == "airfoil_plate":
        return plate_surface(S)
    return True

#: screening criterion weights — V1/V2 default, restated as the form's
#: starting point (api.screen_weights normalises whatever is sent).
#: ``airfoil_select.PRESETS["gdp-sweep"]``, the bulk-sweep preset the whole
#: screen was ported from; a test pins the equality so this cannot drift.
DEFAULT_WEIGHTS = {"ldcr": 0.35, "clmax": 0.20, "cm": 0.20,
                   "ldmax": 0.15, "thick": 0.10, "astall": 0.0}

#: A TANDEM pair's two wings are NOT screened alike, and the numbers for
#: that are not invented here: ``airfoil_select.PRESETS`` carries the GDP
#: TANDEM-WING project's own front and rear presets, ported verbatim with
#: the tunnel-trust terms dropped. They differ where the two jobs differ —
#: the front wing keeps a Cl-max weight (it sets the pair's stall), the rear
#: wing spends four times as much on STALL ANGLE, because it flies in the
#: front wing's downwash and the local angle it sees moves with the front
#: wing's loading. Both weight cruise L/D above everything else.
TANDEM_FRONT_WEIGHTS = {"thick": 0.10, "clmax": 0.25, "ldmax": 0.10,
                        "ldcr": 0.40, "cm": 0.10, "astall": 0.05}
TANDEM_REAR_WEIGHTS = {"thick": 0.10, "clmax": 0.10, "ldmax": 0.05,
                       "ldcr": 0.40, "cm": 0.10, "astall": 0.20}

#: ...and what a TRIMMING surface is screened on: GDP's symmetric preset
#: (``gdp-sym``), verbatim. It spends less on cruise L/D than a wing's
#: preset and more on stall angle and thickness — a stabiliser is asked for
#: control authority and a margin to keep it, not for the last count of
#: cruise drag — but it DOES spend on cruise L/D, because that criterion
#: measures the drag this surface makes at the lift it actually flies
#: (:func:`trim_lift`).
#:
#: It used to zero that criterion and hand its weight to |Cm| and (L/D) max.
#: That was a workaround for screening the surface at ZERO lift, where "L/D
#: at the design Cl" is cl/cd = 0 for every candidate in the database and
#: the ranking silently collapsed onto |Cm| — i.e. onto camber, which made
#: a symmetric section the answer by default rather than by argument. The
#: lift a trimming surface flies is not zero and never was: it is what the
#: family's own moment balance says, and the shell now asks for it.
TRIM_WEIGHTS = {"ldcr": 0.25, "clmax": 0.20, "cm": 0.10,
                "ldmax": 0.15, "thick": 0.10, "astall": 0.10}

#: THE VERTICAL STABILISER's preset, and it is not the trim surface's.
#:
#: THREE criteria stop meaning anything on it, and the weight goes to the one
#: that does:
#:
#: * ``cm`` is ZERO. A symmetric section's pitching moment about the
#:   quarter chord is zero identically (XFOIL returns -0.0 on one), so
#:   ranking a library on it would be ranking noise.
#: * ``ldcr`` — "L/D at the design Cl" — is zero for the same reason a fin has
#:   no design lift: at zero side force there is no L to divide by D. It is 0
#:   for EVERY candidate, so it cannot rank anything.
#: * ``ldmax`` is the same objection one step further out. It is max(cl/cd)
#:   over the linear band — an L/D read at whatever lift happens to maximise
#:   it, which on a fin is a lift the surface never carries. It used to hold
#:   0.35 here on the argument that it "stands in for" drag at zero lift. It
#:   does not: over the 226 eligible symmetric library sections its rank
#:   correlation with -cd at zero lift is **+0.077** (Pearson +0.359), and the
#:   section this preset used to elect (sibnia_s-16) carries **61.9 % more
#:   zero-lift drag** than the lowest-drag section it was ranked against.
#:
#: So that weight moved to ``cdcr`` — the DRAG at the design lift, which at
#: cl = 0 is the zero-lift drag itself. Nothing new is measured: it is the
#: number the ranking already showed in its unweighted "cd @Cl" column
#: (``airfoil_select.LOWER_BETTER``).
#:
#: What is left is: the drag it costs, the thickness that has to house a
#: spar and a rudder hinge, and — the one that matters for control — the
#: angle it still works at. A rudder earns its section at DEFLECTION, where
#: separation decides the side force it can still make, so stall angle and
#: Cl max carry real weight even though the cruise point does not.
FIN_WEIGHTS = {"ldcr": 0.0, "clmax": 0.25, "cm": 0.0, "ldmax": 0.0,
               "cdcr": 0.35, "thick": 0.20, "astall": 0.20}

#: The one thing the screen still cannot say about a trim surface: it is
#: ranked at the lift that trims the DESIGN point, and a stabiliser is asked
#: for lift on BOTH sides of it (that is what the control does). So the
#: ranking carries the raw cd at the design Cl as a column, unweighted, and
#: says that a section chosen here is chosen for one point of a range.
TRIM_DRAG_NOTE = (
    "This surface is ranked at the ONE lift that trims the design point; a "
    "stabiliser is asked for lift either side of it, which no single-point "
    "screen measures. The “cd @Cl” column is the drag at that trim lift, "
    "unweighted — read it beside the score.")

#: A family that declares no design lift — the car rear wing MAXIMISES
#: downforce under a drag budget — still needs the mission form to open on a
#: number. It opens on the load CL = 1.0 implies at that family's own track
#: point: a plain reference chosen by this shell, labelled as such, and not a
#: published target. Everything else opens on what the family itself carries.
REFERENCE_CL = 1.0

#: TAPER IS NOT A MISSION INPUT: every wing family carries taper as a design
#: VARIABLE (``taper`` is the first entry of nearly every param vector), so
#: asking the user to state one would be asking them to guess at the answer
#: the optimiser is about to produce. The mission's Reynolds number is
#: therefore quoted at the MEAN chord S/b — the taper-free chord — which is
#: exactly the MAC of the unity-taper reference trapezoid. What the search
#: can do to it is reported instead (:func:`taper_re_band`).
REFERENCE_TAPER = 1.0

#: NEITHER IS ASPECT RATIO. The mission states a LOAD and a wing loading;
#: those two give the reference area and the design lift coefficient
#: (CL = W/(qS)) without any planform at all. A chord — and so a Reynolds
#: number — needs one more number, and that number is not a mission
#: commitment: it is an ESTIMATE, owned by stage 2 (:func:`section_aspect_ratio`),
#: because the only thing it decides there is which section is screened.
#:
#: NOBODY TYPES ONE ANYWHERE ELSE EITHER. Stage 3 asks for the SPAN — the
#: number a hangar, a trailer or a class rule actually states — as a length
#: on its size card (:func:`chosen_span`) or, where the planform menu made it
#: a design variable, as the b_m band of its design box (:func:`span_box`).
#: The aspect ratio the run flies is then b²/S: a consequence, reported
#: (:func:`flown_aspect_ratio`), never asked. The estimate and the flown one
#: are allowed to differ; the estimate then just says which chord the section
#: was designed for, and stage 3 offers to re-point it.
DEFAULT_ASPECT_RATIO = 10.0

#: what each medium calls its vehicle, and the one-line honest scope note
MEDIA = {
    "air": ("Aircraft wing (air)",
            "Cruise in air: the wing is trimmed to carry the design weight "
            "at the operating point below."),
    "water": ("Hydrofoil (water)",
              "Sea water, cavitation-constrained: speed and depth are "
              "design variables of the water solvers, so the mission below "
              "sets the section design point and the craft weight."),
    "track": ("Car rear wing (track)",
              "Inverted surface in air over the ground: the mission sets "
              "the track speed, and the wing stage owns the rest — the span, "
              "the mount, whether the reference AREA is designed too, what "
              "is maximised (downforce coefficient, downforce in newtons, or "
              "efficiency) and how its drag is budgeted."),
}


# ------------------------------------------------------------------ state
def default_aspect_ratio(problem_name: str) -> float:
    """The aspect ratio a family's OWN planform has (b²/S), or 10.

    This is what the stage-2 estimate opens on, so an untouched session
    designs its section for the chord the family actually flies — and so the
    span stage 3 derives from it is that family's own span, which is what
    keeps the default run bit-for-bit ITS problem's published one. (Which
    problem that is, a fresh session decides through the builder: it opens
    on the chord law, so the air default is ``wing (free chord law)``. A
    chord-law twin carries its base's planform size exactly, so this number
    does not move with it.)
    """
    from aerobo import api

    size = api.planform_size(problem_name)
    if size:
        b, s = float(size[0]), float(size[1])
        if b > 0.0 and s > 0.0:
            return b * b / s
    return DEFAULT_ASPECT_RATIO


def mission_defaults(problem_name: str, medium: str,
                     water: str = "sea") -> dict:
    """The mission form values a family opens on, and where they came from.

    Read from the family itself (``api.family_design_point``): the hydrofoil
    opens on its own 6 kN design lift and its own speed box, not on an air
    wing's 652.8 N at 14.6 m/s. ``w_source`` records what produced the design
    load so the form can say it out loud — including the one case where the
    family declares none and this shell supplies a reference (see
    :data:`REFERENCE_CL`).

    No aspect ratio: see :data:`DEFAULT_ASPECT_RATIO`. The mission is the
    load, the speed, the state of the fluid and the area — nothing that
    describes a planform.
    """
    from aerobo import api

    fdp = api.family_design_point(problem_name)
    area = float(fdp["s_ref_m2"] or 10.0)
    span = float(fdp["b_m"] or (area * 10.0) ** 0.5)
    speed = float(fdp["V"] or 14.6)
    altitude = float(fdp["altitude_m"] or 0.0)
    depth = float(fdp["depth_m"] or 0.6)
    weight, source = fdp["W_N"], fdp["W_source"]
    if weight is None:
        # q is medium/altitude physics, so it is taken from api rather than
        # recomputed here; it does not depend on the weight passed in
        q = api.design_point(medium=medium, W_N=1.0, V=speed, s_ref_m2=area,
                             aspect_ratio=span * span / area, taper=0.6,
                             altitude_m=altitude, water=water,
                             depth_m=depth if medium == "water" else None)["q"]
        weight = REFERENCE_CL * q * area
        source = (f"no published design load for this family — opened at "
                  f"CL = {REFERENCE_CL:g} as a reference")
    return {"W_N": float(weight), "V": speed, "altitude_m": altitude,
            "depth_m": depth, "s_ref_m2": area,
            "taper": REFERENCE_TAPER, "ws_cap_source": WS_CAP_MISSION,
            "size_stated_as": SIZE_AS_LOADING,
            "w_source": source, "accepted": False}


def _airfoil_workspace(weights: dict) -> dict:
    """One surface's airfoil-stage workspace — screening form, ranking, and
    shape optimiser.

    Built twice, from the same factory, because the second surface is asked
    the SAME question at its OWN design point: its own screening report, its
    own criterion weights, its own optimiser trace. Sharing one workspace is
    what made a ranking for the wing sit on screen labelled as the tail's.
    """
    from gui.nice_app import OPT_CHORD_DEFAULT

    return {
        # mission (this surface's own Re) | library (cached, instant).
        # See RE_SOURCE_DEFAULT for why the default is the honest one.
        "re_source": RE_SOURCE_DEFAULT,
        "override_point": False,     # type Re/Cl instead of deriving them
        # how many of the cached ranking's leaders get a real XFOIL sweep at
        # this surface's own Reynolds number (``re_source`` = mission only)
        "shortlist": SHORTLIST_DEFAULT,
        "cond": {"re": 1.0e6, "mach": 0.0, "cl_design": 0.5,
                 "tc_min": 0.15, "cm_max": 0.08},
        "weights": dict(weights),
        # whether those weights are still the RECOMMENDED set for this
        # surface's job (:func:`recommended_weights`) or the user's own. A
        # recommended set follows the configuration; an edited one is never
        # overwritten.
        "weights_source": "recommended",
        # THE THIRD GATE, and the key that makes it a default rather than a
        # ban. None = follow the configuration (:func:`nose_down_required`:
        # on for a wing that HAS a trimming surface, off everywhere else);
        # True/False = the user answered, and the shell never moves it again.
        # It was documented as "the screen form's gate row turns it off"
        # while no key existed and nothing in gui/ wrote one, so 40 reflexed
        # and near-zero-Cm sections left the shipped wing ranking (165 → 125
        # eligible) with nothing on the form or in the header saying a third
        # gate was live.
        "nose_down": None,
        "floors": {"clmax": None, "ldcr": None, "astall": None},
        "top_n": 14,
        "screen": {"report": None, "running": False, "error": None,
                   "progress": None, "stamp": None, "selected": 0,
                   "candidates": None, "surface": None,
                   # WHAT THE SWEEP IS DOING WHILE IT DOES IT. A shortlist
                   # re-screen is minutes of live XFOIL, one section at a
                   # time, and a spinner that says only "screening…" hides
                   # both halves of it: which PASS is running (the cached
                   # library ranking costs nothing; the sweep costs the
                   # minutes) and which sections have already landed.
                   # ``phase`` is None | "library" | "sweep"; ``swept`` is one
                   # row per finished section, in the order they finished.
                   "phase": None, "swept": []},
        # A PENDING RE-SCREEN, set by another stage that discovered this
        # surface's operating point had moved (:func:`arm_rescreen`). It is
        # NOT a queued job: nothing here starts XFOIL on its own. It is the
        # request, carried to the one stage that can price it, so the user
        # arrives at a primed button with the cost stated instead of a log
        # line telling them which control to go and press.
        "rescreen": None,
        "opt": {"budget": 24, "seed": 0, "optimiser": "bo",
                # WHICH scalar the search maximises: ``composite`` — the
                # weighted score of the six criteria the screen ranks on — or
                # ``cd``, this surface's own physical objective (a wing L/D
                # where that is meaningful, the section's 2-D L/D otherwise).
                #
                # OPENS ON THE COMPOSITE WITH THE SEED UNDER IT, because that
                # is the question the user has already answered and because
                # the plain sum answers it badly. The weights above chose the
                # section out of the library on six criteria; a search that
                # then maximises drag alone silently throws five of them away
                # the moment stage 2 starts (report §15.4). But the weighted
                # SUM is free to sell those criteria back, and over 42 paired
                # seeds of the frozen case it does: the plain composite hands
                # back a section with MORE drag at the design lift than the
                # one the user started from in 34 of 42 runs, and it pays for
                # that by selling cruise L/D in 34 and the stall angle in 27.
                # `composite_goal` takes that to 1 of 42 draggier, at a
                # measured price of 1.36 J median (1.9 % of J) — a number only
                # the optimiser sees. A shell exists so a non-specialist gets
                # a section they can fly, and "the score went up while the
                # drag went up" is the one outcome that destroys trust in it.
                #
                # RESULTS_GOAL_ARM_STUDY.md (pre-registered in
                # PREREG_SESSION44.md) is the whole measurement, including the
                # two qualifications that ship with this default: the floor is
                # BREACHED in 21 of 42 runs at budget 48, so nothing here may
                # promise a guarantee; and `composite` is one click away and
                # frozen bit-for-bit, because every published number is on it.
                "objective": "composite_goal",
                "twist_order": 1, "twist_max_deg": 6.0,
                "alpha_max_deg": 10.0,
                # both halves of the chord law come from the ONE constant
                # V1 opens on: seeding the cap from it while hard-coding
                # order 0 beside it left this stage designing sections for
                # a straight taper the wing stage no longer flies
                "chord_order": int(OPT_CHORD_DEFAULT["order"]),
                "chord_max_frac": float(OPT_CHORD_DEFAULT["chord_max_frac"]),
                "tc_min": 0.10, "cm_max": 0.08,
                "report": None, "running": False, "error": None,
                "records": [], "progress": None, "stamp": None,
                # the run ended because the STOP BUTTON fired its stop rule,
                # not because the budget ran out. It is not an error: the
                # report is a partial one carrying the best section the
                # search reached (stages.airfoil.STOP_REASON).
                "stopped": False,
                # THE ARGUMENTS THE LAST RUN ACTUALLY FLEW
                # (:func:`stages.airfoil.shape_kwargs`) beside its budget,
                # optimiser and seed. Kept because "Continue" re-flies THIS
                # and never the form: a form can move between a run and the
                # button, and a longer run of a different search is not a
                # continuation of anything (:func:`continue_section`).
                "launch": None,
                # the budget of the run IN FLIGHT — the continuation's when
                # there is one, so the RUNNING tag counts against the run
                # rather than against a policy that no longer describes it.
                "run_budget": None,
                # SEED vs OPTIMISED on the SCREEN's six criteria under this
                # surface's own weights (api.score_optimised_section). The
                # search maximises one number (cd, or wing L/D); the weights
                # asked for six, so what the other five did is a result, not
                # a footnote — and it is computed after the run rather than
                # during it, because it costs a wide stall sweep per shape.
                "score": None, "scoring": False, "score_error": None},
    }


#: what the wing-level weight form OPENS on when a user first switches the
#: objective to the composite: ``wing_score.PRESETS["cruise"]`` — cruise L/D
#: first, with a fifth of the weight spread over the stall proxy, the load
#: path and the narrowest chord. A recommendation, not a rule: every weight
#: is editable, and an edited set is never overwritten.
WING_WEIGHT_PRESET = "cruise"


def _wing_score_state() -> dict:
    """Stage 3's objective workspace: the scalar, the weights, the band.

    The band is the part that costs something. It is p5/p95 of each criterion
    over a sweep of the design box (``api.wing_score_reference``), measured
    once, and it belongs to the box it was measured on — ``box``/``problem``
    are kept beside it so :func:`wing_band_stale` can say when the user has
    since drawn a different box.
    """
    from aerobo import wing_score as wsc

    return {
        # "lod" (the family's own objective) | "composite"
        "objective": "lod",
        "weights": dict(asdict(wsc.PRESETS[WING_WEIGHT_PRESET])),
        "weights_source": "recommended",
        "samples": int(wsc.REFERENCE_SAMPLES),
        "band": None,          # the measured payload (api.wing_score_reference)
        "box": None,           # the design box it was measured over
        "problem": None,       # …and the family
        "measuring": False, "progress": None, "error": None, "stamp": None,
        # BASELINE vs OPTIMISED on the six criteria after a run
        # (api.score_optimised_design)
        "report": None, "scoring": False, "report_error": None,
        "report_stamp": None,
    }


def wing_score_state(S: dict) -> dict:
    """Stage 3's objective workspace (see :func:`_wing_score_state`)."""
    W = S["wing"]
    if not isinstance(W.get("score"), dict):
        W["score"] = _wing_score_state()
    return W["score"]


def recommended_wing_weights(S: dict) -> dict:
    """The preset a fresh card opens on, minus what this family cannot score.

    The preset is one dict for every family; which of its criteria can be
    MEASURED is a property of the family, and only a measured band knows
    (:func:`wing_band_uncovered`). So the recommendation is the preset with
    the unreportable criteria zeroed — the same set the card shows and the
    run accepts, rather than a set the run would refuse.
    """
    from aerobo import wing_score as wsc

    want = dict(asdict(wsc.PRESETS[WING_WEIGHT_PRESET]))
    for key in wing_band_uncovered(S):
        want[key] = 0.0
    return want


def wing_weights_are_recommended(S: dict) -> bool:
    """Is the wing weight set still the preset it opened on (unedited)?"""
    sc = wing_score_state(S)
    if sc.get("weights_source") != "recommended":
        return False
    want = recommended_wing_weights(S)
    return all(abs(float(sc["weights"].get(k, 0.0)) - float(v)) < 1e-12
               for k, v in want.items())


def set_wing_weight(S: dict, key: str, value) -> None:
    """Write one wing criterion weight — and record that it is the user's.

    A band stays valid across a weight change (it normalises the criteria,
    it does not weight them), so nothing is invalidated here.
    """
    from aerobo import wing_score as wsc

    if key not in wsc.CRITERIA:
        raise ValueError(f"unknown wing criterion {key!r}")
    sc = wing_score_state(S)
    sc["weights"][key] = max(0.0, float(value))
    sc["weights_source"] = "user"


def set_wing_weights(S: dict, weights, source: str = "user") -> None:
    """Write the whole wing weight set (a preset name or a dict)."""
    from aerobo import wing_score as wsc

    sc = wing_score_state(S)
    sc["weights"] = dict(asdict(wsc.weights_of(weights)))
    sc["weights_source"] = source


def wing_band_uncovered(S: dict) -> tuple:
    """Criteria the MEASURED band does not cover, in criterion order.

    A family reports what its solver exports, and not every family exports
    everything: a car rear wing has no ``clmax`` (nothing trims it, so there
    is no lift it stalls at) and no ``mass`` (Raymer is a light-aircraft
    correlation), a hydrofoil has neither of those nor ``astall``/``vol``.
    ``api.wing_score_reference`` measures a band only for what the sweep
    actually saw, and the composite then REFUSES a weighted criterion the
    band does not cover — loudly, in the optimiser, on a run the user has
    already paid for.

    So the form has to be able to ask the same question the run will ask.
    Empty while no band has been measured yet: nothing is known to be
    missing then, and answering "all of them" would grey out every slider on
    a fresh card.
    """
    from aerobo import wing_score as wsc

    band = (wing_score_state(S) or {}).get("band") or {}
    bounds = band.get("bounds")
    if not bounds:
        return ()
    return tuple(c for c in wsc.CRITERIA if c not in bounds)


def wing_weights_refused(S: dict) -> tuple:
    """Criteria this run would be REFUSED for: weighted, and not covered.

    The pair that makes a composite run raise. Checked where the weights are
    typed so the answer lands beside the slider that caused it, which is the
    same rule :func:`api.check_wing_objective` follows for the objective.
    """
    sc = wing_score_state(S)
    return tuple(c for c in wing_band_uncovered(S)
                 if float(sc["weights"].get(c, 0.0)) > 0.0)


def set_wing_objective(S: dict, name: str) -> None:
    """Choose the scalar stage 3 maximises (``lod`` | ``composite``)."""
    from aerobo import api

    if name not in api.WING_OBJECTIVES:
        raise ValueError(f"unknown wing objective {name!r}")
    wing_score_state(S)["objective"] = str(name)


def set_wing_band(S: dict, payload: dict | None, box: dict | None = None
                  ) -> None:
    """Store a measured band together with the box and family it belongs to.

    Measuring the band is also the moment this shell first LEARNS which
    criteria the family reports, so it is the moment a RECOMMENDED weight set
    has to be brought into line with them. The shipped preset weights the
    stall lift; a car rear wing has no stall lift to weight, so a track
    session that took the recommendation and switched to the composite was
    refused by its own default — the user had chosen nothing wrong and the
    message named a criterion they never asked for.

    Only a recommended set is touched. An edited one is the user's answer and
    is left exactly as typed, with :func:`wing_weights_refused` saying what it
    would cost — the same split every other recommendation in this shell
    follows (:func:`refresh_recommended_weights`).
    """
    sc = wing_score_state(S)
    sc["band"] = payload
    sc["box"] = box
    sc["problem"] = S["wing"]["problem"] if payload else None
    if payload and sc.get("weights_source") == "recommended":
        dropped = wing_band_uncovered(S)
        if dropped:
            for key in dropped:
                sc["weights"][key] = 0.0


def wing_band_box(S: dict) -> dict:
    """The design box a band would be measured over, as plain numbers.

    Exactly what the run will search (``config.effective_bounds``, with every
    FIXED row collapsed to its value the way :func:`_searched_box` does), so
    the staleness check compares like with like — fixing a row after
    measuring changes the population and has to show up as stale.
    """
    return {str(k): [float(v[0]), float(v[1])]
            for k, v in _searched_box(S).items()}


def wing_band_stale(S: dict) -> str | None:
    """Why the stored band no longer describes this search, or None.

    A band is a population, and the population is THE BOX. Changing the
    family, or narrowing a bound after measuring, leaves J perfectly
    well-defined but normalised against a region the search no longer visits
    — which is worth saying out loud, and not worth refusing over.
    """
    sc = wing_score_state(S)
    if not sc.get("band"):
        return None
    if sc.get("problem") and sc["problem"] != S["wing"]["problem"]:
        return (f"measured on “{sc['problem']}”, and this stage now flies "
                f"“{S['wing']['problem']}”")
    old = sc.get("box")
    if not isinstance(old, dict):
        return None
    new = wing_band_box(S)
    if set(old) != set(new):
        return "the design box has different rows than when it was measured"
    moved = [k for k, v in new.items()
             if abs(float(v[0]) - float(old[k][0])) > 1e-9
             or abs(float(v[1]) - float(old[k][1])) > 1e-9]
    if moved:
        return ("the design box moved since it was measured: "
                + ", ".join(moved[:4])
                + (f" (+{len(moved) - 4} more)" if len(moved) > 4 else ""))
    return None


def wing_score_flags(S: dict) -> dict:
    """The objective flags stage 3's run travels with (empty on ``lod``).

    Empty is the point: a run on the family's own objective sends exactly
    the flags it always sent, so every stored run and every cached result
    still means what it meant.
    """
    sc = wing_score_state(S)
    if str(sc.get("objective") or "lod") != "composite":
        return {}
    band = sc.get("band")
    if isinstance(band, dict) and band.get("samples"):
        # the measured POPULATION stays in the session, not in the run
        # record: the objective reads the (lo, hi) bands only, and every
        # stored run — and the reproduce snippet — would otherwise carry a
        # few hundred floats of display data that change no score. The
        # score block gets the full band handed to it directly.
        band = {k: v for k, v in band.items() if k != "samples"}
    return {"wing_objective": "composite",
            "wing_score_weights": {k: float(v)
                                   for k, v in sc["weights"].items()},
            "wing_score_reference": band}


# =====================================================================
# V3.5 — the SEARCH POLICY: the measured recommendation, or the user's own
# =====================================================================
#
# Which optimiser and how many evaluations were, until V3.5, two numbers the
# user was asked for on two different stages with no way of knowing the
# answer. They are now measured (``aerobo.optimize.budget``, fitted by
# ``aerobo.experiments.budget_study``) and asked ONCE, here, as a policy:
#
#     recommended   every stage's search follows the study — the budget is a
#                   function of that stage's own design-vector length and of
#                   what it is optimising, so it changes when the freedoms do
#     own           the numbers on stages 2 and 3 are the user's, exactly as
#                   they were before, and nothing moves them
#
# It lives at the TOP of the session and not inside ``wing`` because it is
# not a wing setting: the section stage searches too, and one question asked
# in two places is the bug class this shell keeps out.

#: mode -> what it means on the stages
SEARCH_MODES = ("recommended", "own")

#: how converged the recommendation aims for (``optimize.budget.EFFORTS``)
SEARCH_EFFORTS = ("quick", "balanced", "thorough")

SEARCH_EFFORT_LABELS = {
    "quick": "quick — 90% of the way",
    "balanced": "balanced — 95%",
    "thorough": "thorough — 99%",
}


def _search_state() -> dict:
    return {
        "mode": "recommended",
        "effort": "balanced",
        # stop as soon as the search stops improving, instead of spending the
        # whole budget. The budget stays the backstop — an adaptive stop with
        # no ceiling is an unbounded run.
        "stop_when_converged": True,
        "error": None,
        "cache": {},
    }


def search_state(S: dict) -> dict:
    """The policy, created on demand (a session pickled before V3.5 has none)."""
    st = S.get("search")
    if not isinstance(st, dict):
        st = _search_state()
        S["search"] = st
    st.setdefault("cache", {})
    return st


def search_is_recommended(S: dict) -> bool:
    return search_state(S)["mode"] == "recommended"


def set_search_mode(S: dict, mode: str) -> bool:
    """Choose the policy. ``False`` on a name that is not one, or no change.

    A mode chosen on the RADIO is a fresh statement, so it drops whatever
    ``adopt_recommendation`` left behind (:data:`ADOPTED_SEARCH_KEY`): the
    two carries are what "use these as my own values" was holding, and a
    user who has since picked "my own values" from the control itself is
    saying something else. Adopt writes its stash after calling this.
    """
    if mode not in SEARCH_MODES:
        return False
    st = search_state(S)
    if st["mode"] == mode:
        return False
    st["mode"] = mode
    clear_adopted_search(S)
    return True


def set_search_effort(S: dict, effort: str) -> bool:
    if effort not in SEARCH_EFFORTS:
        return False
    st = search_state(S)
    if st["effort"] == effort:
        return False
    st["effort"] = effort
    st["cache"] = {}            # the budget is a function of it
    return True


def set_search_stop(S: dict, on: bool) -> None:
    search_state(S)["stop_when_converged"] = bool(on)


def search_study() -> dict | None:
    """Provenance of the frozen study, or None if it is not installed."""
    from aerobo.optimize import budget as B

    try:
        return B.study_stamp()
    except B.MissingStudyError:
        return None


def _plan_key(*parts) -> str:
    import hashlib

    return hashlib.sha1("|".join(repr(p) for p in parts).encode()).hexdigest()[:16]


def _cached_plan(S: dict, key: str, build):
    """Plans are cached per (problem, flags, effort): building one costs a
    problem build and a few timed evaluations, and every re-render of a stage
    would otherwise pay for it again."""
    st = search_state(S)
    cache = st["cache"]
    if key not in cache:
        cache[key] = build()
        if len(cache) > 16:                     # a session, not a database
            for stale in list(cache)[:-8]:
                cache.pop(stale, None)
    return cache[key]


def wing_plan(S: dict):
    """The measured plan for stage 3's problem, or None (with ``error`` set).

    None is not a failure of the run: it means the frozen study is not
    installed (or this problem could not be built), and the shell falls back
    to the user's own numbers and says so.
    """
    from aerobo import api

    from . import config

    st = search_state(S)
    try:
        flags = config.flags(S)
    except Exception as exc:            # noqa: BLE001 — a read-out
        st["error"] = f"{type(exc).__name__}: {exc}"
        return None
    # WHICH SCALAR is being maximised is part of the question: the study
    # measured the composite's own cost against the family's objective on the
    # same problems (:func:`aerobo.optimize.budget.recommend`)
    objective = str(wing_score_state(S).get("objective") or "lod")
    # the FIXED rows belong in the key: the budget law is a law in the
    # dimension actually searched, so a plan cached before a row was fixed is
    # a plan for a bigger problem than the one about to run
    fixed = config.fixed_rows(S)
    key = _plan_key("wing", S["wing"]["problem"], sorted(flags), flags,
                    st["effort"], objective, repr(sorted(fixed.items())))

    def _build():
        return api.recommended_search(
            S["wing"]["problem"], flags=flags,
            mission_kwargs=config.mission_kwargs(S),
            bounds_overrides=config.bounds_overrides(S),
            effort=st["effort"], objective=objective, pinned=fixed or None)

    try:
        plan = _cached_plan(S, key, _build)
    except Exception as exc:            # noqa: BLE001 — a read-out
        st["error"] = f"{type(exc).__name__}: {exc}"
        return None
    st["error"] = None
    return plan


def _states_its_own_vertical(report: dict) -> bool:
    """Does the DESIGN carry the surface its yaw stiffness comes from?

    Not "is there a fin block": a V-TAIL has none, and it is the one layout
    in this package whose vertical surface is stated, charged and flown
    with nothing invented anywhere in the chain. Its ``geometry["fin"]`` is
    an explicit ``None`` — "no SEPARATE fin" — and the yaw comes from the
    cant of its own two panels, which is stated geometry
    (``geometry["tail"]["dihedral_deg"]``). Measured on `tail` at the box
    centre: 35 deg of panel cant is Cn_beta +0.0666, moving monotonically
    with the cant, and ``cd0_fin`` is 0.0 because there is no fin to charge.
    Read as "no fin block, therefore no vertical surface", that design was
    refused an answer it can give correctly — and told it had no yaw
    stiffness when it has the third-highest in the family.

    The two cases that DO have to be refused look identical at the ``fin``
    key and are separated here:

    * the fin switched OFF on a conventional layout — ``fin`` is None and
      the tail's own cant is 0, and the deck's Cn_beta is then exactly
      -0.000000. Nothing carries yaw, which is what the user asked for;
    * a CANARD — the ``fin`` key is ABSENT rather than None, because the
      aft fin's arm is not the (upstream) surface station the block carries
      (api.py). The design is charged cd0_fin for a surface it does not
      describe, so the rebuild invents one, and the invented fin is 21 %
      stiffer than the one the design pays for. A number measured on it is
      about an aeroplane nobody designed.
    """
    geom = report.get("geometry") or {}
    if geom.get("fin"):
        return True
    if "fin" not in geom:
        return False                    # canard: absent, not "no fin"
    tail = geom.get("tail") or {}
    try:
        return abs(float(tail.get("dihedral_deg") or 0.0)) > 1e-9
    except (TypeError, ValueError):
        return False


#: what the searched wing cant is floored at while NOTHING PRICES ITS SIGN.
#:
#: The row is the FLATTEST of the eleven this family trades — 2.14 % of L/D
#: end to end, against 41.6 % for the sweep and 27.6 % for the tail arm — so
#: a search resolves it last or not at all. Measured over 32 seeds of the
#: shell's own configuration at budget 24: 11 came back ANHEDRAL, and only
#: 12 of 32 had a convergent spiral. More budget does not fix it (7 of 16
#: negative at the shell's own 44).
#:
#: AND NOTHING IN THIS PACKAGE PREFERS ANHEDRAL. Sweeping the row at each
#: free-cant family's box centre, L/D peaks at +5 to +6 deg on 6 of 6
#: families and the anhedral bound is 2.0-2.5 % BELOW the best; over 8 Sobol
#: box points, 0 have a negative L/D argmax; at all 16 measured winners the
#: local optimum in this row is positive and the winner is giving up a
#: median 0.52 % of L/D by not being there. The one quantity anhedral
#: maximises is Dutch-roll damping, which no criterion scores.
#:
#: The published band's own justification — a high wing whose effective
#: dihedral is too large — is a phenomenon this package cannot represent:
#: there is no wing vertical position in the physics and no fuselage in the
#: lattice, so nothing in a model here can ask for anhedral.
#:
#: SO IT IS A DEFAULT AND NOT A BAN, which is this repo's standing rule. It
#: is lifted the moment the objective can pay for the sign (weight
#: ``spiral``), and typing the row takes it back — the box view says which
#: of the two you are looking at. Measured: 6 seeds at budget 44, 4/6
#: anhedral before and 0/6 after, median L/D 28.095 -> 28.286 (the search
#: stops spending draws on a half-box nothing wants).
CANT_FLOOR_DEG = 0.0


def unpriced_cant_floor(S: dict) -> dict:
    """``{row: [lo, hi]}`` flooring the searched cant, or ``{}``.

    Three conditions, all read rather than remembered: the family SEARCHES
    the cant (a stated one is the user's own number and is not touched), the
    objective does NOT price ``spiral``, and the row is one this family
    actually has. A row the user has typed is not special-cased here — their
    band is applied after this one in both consumers and simply wins.
    """
    from aerobo import api

    from . import config

    name = S["wing"]["problem"]
    row = api.WING_CANT_KEYS[0]
    # THE DIHEDRAL specifically, not "a cant is searched": the floor is a
    # statement about that row's SIGN, and since the freedom split a family
    # can search the sweep and state the dihedral.
    if not api.dihedral_is_searched(name):
        return {}
    try:
        if api.wants_spiral(config.flags(S)):
            return {}
    except Exception:                   # noqa: BLE001 — a read-out
        return {}
    band = (api.PROBLEM_SPECS[name].default_bounds or {}).get(row)
    if not band:
        return {}
    lo, hi = float(band[0]), float(band[1])
    if lo >= CANT_FLOOR_DEG or hi <= CANT_FLOOR_DEG:
        return {}
    return {row: [float(CANT_FLOOR_DEG), hi]}


def fuselage_diameter_default(S: dict) -> float | None:
    """The body diameter this configuration OPENS with, or None.

    A shape statement, not a number somebody guessed: the diameter that
    puts the body at :data:`V3_FUSELAGE_FINENESS` for the arm this session
    is actually searching — read off the EFFECTIVE design box, so widening
    the arm band moves the body the shell offers with it. Measured off the
    published band it would quote a diameter for an aeroplane the run was
    told not to build.

    See :data:`V3_FUSELAGE_FINENESS` for the two measurements behind the
    ratio, and :data:`V3_START_FLAGS` for why a body is offered at all.
    """
    from aerobo import api, drag

    from . import config

    spec = api.PROBLEM_SPECS.get(S["wing"]["problem"])
    if spec is None or api.FUSELAGE_KEYS[0] not in tuple(spec.flags):
        return None
    stated = (S["wing"].get("flags") or {}).get("l_t_m")
    if stated not in (None, ""):
        arm = float(stated)
    else:
        row = config.effective_bounds(S).get("l_t_m")
        if row is None:
            arm = _default_arm_m(spec)
        else:
            (lo, hi), _src = row
            arm = 0.5 * (float(lo) + float(hi))
    if not arm:
        return None
    return round(drag.diameter_for_fineness(arm, V3_FUSELAGE_FINENESS), 3)


def cant_states_offered(problem_name: str, choices: dict) -> dict:
    """``{state: is it reachable from here}`` over :data:`api.WING_CANTS`.

    WHICH FREEDOMS THIS CONFIGURATION CAN ACTUALLY BE GIVEN, asked of the
    registry one state at a time rather than as a single yes/no: the two
    halves and the pair are three different registered problems, and a
    family can perfectly well carry one and not another. The state the
    problem is already IN is offered without a round trip — it is on
    screen, so no menu has to agree that it exists.
    """
    from aerobo import api
    from gui.nice_app import option_available

    mode = api.searched_cant(problem_name)
    return {k: (mode == k or option_available(choices, "wing_cant", k))
            for k in api.WING_CANTS}


def cant_is_stated(problem_name: str) -> bool:
    """Can this family be TOLD a cant — the two flags, not the two rows?

    The narrow half of :func:`cant_is_answerable`, split out because the
    advice that sends a user to the cant card has to say which of the two
    answers is available there. A lifting-line pair can be given a searched
    dihedral (through its nonplanar twin) and cannot be given a stated one
    at all, so "state a value" is a sentence about a field it has not got.
    """
    from aerobo import api

    spec = api.PROBLEM_SPECS.get(problem_name)
    return spec is not None and set(api.WING_CANT_KEYS) <= set(spec.flags)


def cant_is_answerable(problem_name: str, choices: dict | None = None
                       ) -> bool:
    """Does this family have a wing cant to state or search AT ALL?

    Every way of answering counts — the STATED pair as flags, the two
    SEARCHED rows, or a TWIN this configuration can be switched to —
    because the question is "does this wing have a dihedral", not "is there
    a field". Read off the registry every time.

    THE THIRD WAY IS THE TANDEM PAIR'S ONLY ONE, and leaving it out is what
    made the pair's dihedral unreachable in both directions. A lifting line
    has no out-of-plane geometry, so the published ``tandem`` declares
    neither cant key and searches neither row — but every cant state has a
    registered NONPLANAR twin (``api.tandem_vlm_problem``), and the card
    this predicate guards is the control that selects one. Without
    ``choices`` there is nothing to ask that question of, so the answer
    stays the narrow one: pass them wherever a session is in hand.

    ONE AUTHOR, because two places branch on it and they must not disagree:
    stage 3 draws the "Wing cant and sweep" card only where this is true (a
    family that cannot score a cant draws nothing there, no heading and no
    button), and stage 5's spiral advice may only send a user to that card
    when it exists. Advice naming a card that is not drawn is worse than no
    advice: the user goes looking and concludes the shell is broken. The
    pair got the WORSE half of that trade — its spiral is a wing answer no
    fin size can turn, and the advice that says so was replaced by "this
    family has no field for it" on the one family that most needed it.
    """
    from aerobo import api

    if problem_name not in api.PROBLEM_SPECS:
        return False
    if cant_is_stated(problem_name) or api.cant_is_searched(problem_name):
        return True
    if choices is None:
        return False
    offered = cant_states_offered(problem_name, choices)
    return any(offered.get(k) for k in api.WING_CANTS if k != "fixed")


def spiral_dihedral(S: dict, *, force: bool = False) -> dict:
    """How much WING DIHEDRAL turns THIS configuration's spiral, measured.

    The card used to quote ONE constant — 5.7 deg — at every configuration,
    and it is one family's answer: measured over the families this shell
    derives, the crossing runs 3.07 deg (``tail + winglet``, whose tip
    device already carries most of ``Cl_beta``) to 6.13 deg (``tail``, and
    the shell's own default ``tail + free chord law`` with it). So the step
    that shipped at 5.7 deg was 86 % too much on one family and too small on
    the one a user lands on first: press it there and the aeroplane still rolls
    off (6-DOF spiral eigenvalue +0.0026 at 5.7 deg, -0.0005 at 6.13).

    The box CENTRE is what is measured, which is what the constant claimed
    to be and what the shell can know before a search has run. It is one
    ``api.design_report`` plus about twenty lattice rebuilds — ~0.1 s on a
    wing+tail — and it is cached per (problem, flags, mission, box) like
    every other measured read-out on this stage.

    ``force`` measures a family the registry marks ``slow`` (a live-XFOIL
    section twin), where the report alone can cost minutes. Without it those
    return ``status="slow"`` and the card offers the measurement as a button
    rather than paying for it in a render.

    Returns a plain dict — ``status`` first, and the number is only
    meaningful under it (:func:`aerobo.api.dihedral_for_spiral`), plus
    ``lod_cost_pct`` where the family can price the cant it would fly.
    """
    from aerobo import api

    from . import config

    name = S["wing"]["problem"]
    spec = api.PROBLEM_SPECS.get(name)
    if spec is None:
        return {"status": "error", "error": f"no such problem: {name!r}",
                "problem": name}
    if spec.slow and not force:
        return {"status": "slow", "problem": name}
    # A SPIRAL IS A QUESTION A FREE-FLYING AEROPLANE HAS. The measurement is
    # a rebuild, and a rebuild will happily fly anything: asked about a car's
    # rear wing it invents a fin, trims the assembly as an aircraft and
    # reports that 6.92 deg of dihedral would turn "its" spiral — a number
    # about an aeroplane nobody designed. So the card asks only where the
    # design IS one: air, and a vertical surface the DESIGN carries rather
    # than one the rebuild made up (with no vertical surface in the report
    # it sizes one off the span, and every lateral number then rides three
    # assumed dimensions). Everything else gets no answer instead of a
    # fabricated one, and the ``why`` says which of the two it was.
    if spec.medium != "air":
        # NOT "it has no spiral mode": it has one, and 148 of 150 box draws
        # of `hydrofoil + elevator` classify it (74 of them divergent). The
        # reason is that nothing about it would be trustworthy — the only
        # vertical surface a foiling craft has here is the MAST, which
        # stands at the foil with no yaw arm (Cn_beta +0.005 with no
        # elevator, and -0.196 with one, because the CG moves aft of it),
        # the hull and rudder that would carry directional stability are
        # not modelled at all, and no water breakdown states a density, so
        # a rebuild flies the craft in AIR.
        return {"status": "not_applicable", "problem": name,
                "why": f"a {spec.medium} design's only vertical surface is "
                       f"the mast that carries the foil, and the hull and "
                       f"rudder that would make it weathercock are not "
                       f"modelled — so a spiral answer here would be about "
                       f"a craft this package has not described"}
    try:
        flags = config.flags(S)
        mission = config.mission_kwargs(S)
        overrides = config.bounds_overrides(S)
    except Exception as exc:            # noqa: BLE001 — a read-out
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}",
                "problem": name}
    key = _plan_key("spiral", name, sorted(flags), flags, mission, overrides)

    def _build():
        import numpy as _np

        built = spec.build(mission, flags, overrides)
        x = _np.asarray(built.bounds, dtype=float).mean(axis=1)
        cfg = api.RunConfig(problem_name=name, budget=4, seed=0, flags=flags,
                            mission_kwargs=mission,
                            bounds_overrides=overrides)
        report = api.design_report(cfg, x)
        # ASKED OF THE API, which is also what keeps this stage clear of the
        # machinery that flies a design: the measurement is a rebuild, the
        # question is the wing's geometry, and the api is where the two
        # meet. ``fin_stated`` comes back with it — a report with no
        # vertical surface has one INVENTED for the measurement, and every
        # lateral number then rides three assumed dimensions, which the card
        # has to be able to say rather than quoting an angle as though the
        # aeroplane had been measured with a fin on it.
        if not _states_its_own_vertical(report):
            return {"status": "no_fin", "problem": name,
                    "why": "this design states no vertical surface of its "
                           "own, so any yaw stiffness measured here would "
                           "come from a fin the rebuild invented"}
        out = dict(api.dihedral_for_spiral(report), problem=name,
                   lod_cost_pct=None)
        # ...AND WHAT IT COSTS, where the family can be asked. A cant is
        # geometry a lifting line cannot score, so only the lattice-backed
        # families have a price to quote — and quoting one family's price
        # beside another family's angle is the same mistake the constant
        # made, in the other column.
        if (out["gamma_deg"] and out["status"] == "found"
                and api.WING_CANT_KEYS[0] in tuple(spec.flags)):
            try:
                flat = built.evaluate(x)
                canted = spec.build(
                    mission,
                    dict(flags,
                         **{api.WING_CANT_KEYS[0]:
                            float(out["gamma_deg"])}),
                    overrides).evaluate(x)
                if flat.get("LoD") and canted.get("LoD"):
                    out["lod_cost_pct"] = float(
                        100.0 * (flat["LoD"] - canted["LoD"]) / flat["LoD"])
            except Exception:           # noqa: BLE001 — a read-out
                out["lod_cost_pct"] = None
        return out

    try:
        return _cached_plan(S, key, _build)
    except Exception as exc:            # noqa: BLE001 — a read-out
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}",
                "problem": name}


def airfoil_plan(S: dict, surface: str = "main"):
    """The measured plan for ONE surface's section search (stage 2 / 2.5).

    The dimension is the section problem's own — 8 CST weights plus whatever
    twist and chord coefficients the objective opened — so a surface that
    designs a twist law is recommended a bigger budget than one that does
    not, which is the whole point of asking the study per stage.
    """
    from aerobo import api

    st = search_state(S)
    A = airfoil_state(S, surface)
    oc = A["opt"]
    wing = wing_guess(S, surface) if wing_objective(S, surface) else None
    # EVERY composite objective, asked of the api rather than listed here: the
    # plain J, the same J with the seed as a floor, and the Tchebycheff of the
    # same six are one 2-D section problem with one dimension and one measured
    # budget factor. Reading only "composite" here planned a composite_goal run
    # as a -cd run — wing guess, twist law, the wrong vector and the wrong
    # budget — which is the same stale-branch bug class as any other menu that
    # lists a name a helper does not know. A literal tuple here would have
    # reopened it the moment a fourth objective was added, so it asks.
    objective = str(oc.get("objective", "cd"))
    composite = objective in api.AIRFOIL_COMPOSITE_OBJECTIVES
    kw = dict(twist_order=int(oc["twist_order"]),
              chord_order=int(oc["chord_order"]))
    if composite or wing is None:
        # a composite section run is a SECTION run: no wing, so no twist or
        # chord law joins the vector (api.airfoil_run_config refuses the mix)
        kw = {}
    key = _plan_key("airfoil", surface, bool(wing), objective, kw, st["effort"])

    def _build():
        name = objective if composite else "cd"
        cfg = api.airfoil_run_config(
            objective=name, wing=(None if composite else wing), **kw)
        return api.recommended_search(
            cfg.problem_name, flags=cfg.flags, effort=st["effort"],
            objective=name)

    try:
        plan = _cached_plan(S, key, _build)
    except Exception as exc:            # noqa: BLE001 — a read-out
        st["error"] = f"{type(exc).__name__}: {exc}"
        return None
    st["error"] = None
    return plan


def effective_wing_search(S: dict) -> dict:
    """What stage 3 will ACTUALLY run with, and where each number came from.

    ``source`` is ``"recommended"`` or ``"own"``; every consumer — the run
    config, the card, the reproduce snippet — reads this one function, so the
    thing shown and the thing launched cannot drift apart.
    """
    from aerobo import api

    W = S["wing"]
    # the two decisions the stage has no FIELD for travel from the adoption
    # that made this the user's own plan, and are the shell's published
    # defaults where nothing was adopted (:func:`adopted_search_carries`)
    carries = adopted_search_carries(S, WING_SEARCH)
    own = {"optimiser": str(W["optimiser"]), "acqf": str(W["acqf"]),
           "budget": int(W["budget"]), "n_init": carries["n_init"],
           "n_seeds": max(1, int(W.get("n_seeds") or 1)),
           "seed": int(W["seed"]), "refusal": carries["refusal"],
           "source": "own", "plan": None}
    if not search_is_recommended(S):
        return own
    plan = wing_plan(S)
    if plan is None:
        return own
    names = api.compatible_optimisers(W["problem"])
    optimiser = plan.optimiser if plan.optimiser in names else own["optimiser"]
    if optimiser == "bo" and "bo_slsqp" in names:
        # THE RECOMMENDER CANNOT NAME THIS ARM. Its law is fitted from
        # `results/budget_study/wing.jsonl`, which contains pure arms only —
        # `bo_slsqp` did not exist when those rows were collected, so "bo" is
        # the best thing it can say and not a finding that bo BEAT the
        # handoff. The later, separately certified measurement
        # (`RESULTS_HANDOFF.md`, `RESULTS_HANDOFF_SPLIT.md`) puts
        # `bo -> slsqp` above pure BO on 42 of 54 independent runs.
        #
        # The recommended BUDGET is kept, on the reasoning this function
        # already applies one branch below: a budget is a property of the
        # problem's SIZE, not of the method that spends it.
        #
        # `n_init` is deliberately NOT kept, and that is not tidying. The
        # recommended Sobol seed is sized for a BO run that owns the whole
        # budget; the handoff's BO phase owns a quarter of it, and
        # `_bo_split` clamps n_init to `n_a - 1` — so forwarding a
        # full-budget seed leaves the phase with ONE acquisition step. That
        # is a Sobol draw followed by a polish, which is not the arm the
        # study measured, wearing the arm's name.
        return {**own, "optimiser": "bo_slsqp", "budget": int(plan.budget),
                "n_seeds": max(1, int(plan.n_restarts)),
                "refusal": plan.refusal,
                "n_init": None, "source": "recommended", "plan": plan}
    if optimiser != plan.optimiser:
        # the study's winner is not offered for this family (the gradient
        # baselines are unconstrained-only). Keep the recommended BUDGET —
        # that is a property of the problem's size, not of the optimiser —
        # and say which optimiser is actually flying.
        return {**own, "budget": int(plan.budget), "source": "recommended",
                "plan": plan, "optimiser": optimiser,
                "n_init": None, "refusal": "sentinel"}
    return {"optimiser": optimiser,
            "acqf": (plan.acqf or "logei"),
            "budget": int(plan.budget),
            "n_init": plan.n_init,
            "n_seeds": max(1, int(plan.n_restarts)),
            "seed": int(W["seed"]),
            # what a design the physics REFUSED is shown to the surrogate as
            # (aerobo.optimize.refusal). A search decision like the split: the
            # objective still returns -100 and the run still records it.
            "refusal": plan.refusal,
            "source": "recommended", "plan": plan}


def effective_airfoil_search(S: dict, surface: str = "main") -> dict:
    """The same, for one surface's section search."""
    from aerobo import api

    oc = airfoil_state(S, surface)["opt"]
    carries = adopted_search_carries(S, surface)
    own = {"optimiser": str(oc["optimiser"]), "budget": int(oc["budget"]),
           "seed": int(oc["seed"]), "n_restarts": 1,
           "refusal": carries["refusal"], "n_init": carries["n_init"],
           "source": "own", "plan": None}
    if not search_is_recommended(S):
        return own
    plan = airfoil_plan(S, surface)
    if plan is None:
        return own
    # the section problem is constrained (t/c and |Cm| margins), so not every
    # optimiser in the study can fly it; the BUDGET is still the recommended
    # one either way — it is a property of the vector's length, not of the
    # method that spends it
    names = api.compatible_optimisers(api.AIRFOIL_PROBLEM)
    optimiser = plan.optimiser if plan.optimiser in names else own["optimiser"]
    return {"optimiser": optimiser, "budget": int(plan.budget),
            "seed": int(oc["seed"]),
            # a restart splits the budget rather than multiplying it: k runs
            # of ``budget`` each, and the best one is the answer
            "n_restarts": max(1, int(plan.n_restarts)),
            # What a candidate XFOIL refused is shown to the surrogate as
            # (aerobo.optimize.refusal) — on this class 7-14 % of evaluations
            # come back as the -100 sentinel, and showing the GP that number
            # wrecks its scale. The study measured the alternative winning on
            # 20/20 seed pairs, so the recommended plan carries "worst".
            "refusal": plan.refusal,
            # The Sobol seed size. It has to be CARRIED, not left to the
            # runner's default: the section's own split arms rank a four-point
            # seed first (0.5 d), against a default of 2 d capped at 16. While
            # those two happened to agree the omission was invisible, which is
            # exactly why it survived.
            "n_init": plan.n_init,
            "source": "recommended", "plan": plan}


def stop_rule_factory(S: dict, eff: dict | None = None):
    """A FACTORY for the adaptive stop, or None if nothing asked for one.

    A factory because a stop rule holds the run's own history: handing the
    same object to every seed of a queue would stop the second run on the
    first one's plateau.

    ``eff`` is either stage's effective search (:func:`effective_wing_search`
    / :func:`effective_airfoil_search`) — both carry the plan the patience
    and the tolerance come from, and both are bounded by their own budget.
    """
    from aerobo.optimize.budget import ConvergenceStop

    st = search_state(S)
    if not (search_is_recommended(S) and st.get("stop_when_converged")):
        return None
    eff = eff or effective_wing_search(S)
    plan = eff.get("plan")
    if plan is None or plan.patience is None or plan.tol is None:
        return None
    patience, tol, budget = (int(plan.patience), float(plan.tol),
                             int(eff["budget"]))

    def _factory():
        rule = ConvergenceStop(patience=patience, tol=tol, max_evals=budget)

        def _stop(i, best):
            fired = rule.update(best)
            # ...but ONLY for the plateau. ``ConvergenceStop`` also returns
            # True at ``n >= max_evals``, and max_evals here is the run's own
            # budget — which the optimiser stops at anyway. Passing that
            # through fired the rule on the LAST evaluation of every run: a
            # complete search came back ``partial=True``, was written as
            # ``<ts>_partial.json``, lost ``n_screened``/``n_rescue``/
            # ``failures`` (``partial_result`` never sets them), paid one
            # extra objective evaluation to rebuild its breakdown, and the
            # Results page told the user "stopped early - 29 of 29
            # evaluations - the stop rule fired". An adaptive stop that
            # cannot be exhausted is still impossible, because the budget
            # remains the optimiser's ceiling; what is gone is the redundant
            # second statement of it.
            return bool(fired and rule.n < rule.max_evals)

        return _stop

    return _factory


def convergence_rule_reach(S: dict, eff: dict | None = None) -> dict | None:
    """Can the adaptive stop fire at THIS budget? ``None`` when there is no
    rule at all.

    ``{"can_fire", "patience", "needs", "budget"}``. The rule needs more than
    ``patience`` evaluations before its plateau test has anything to compare
    against, and the study publishes ``patience = 40`` for the wing against
    recommended budgets of 17-53 — so at the quick and balanced efforts the
    switch is on, the card says it will stop a run that flattens out, and it
    cannot. A switch that does nothing must say so rather than be left as a
    control the user believes they have used.
    """
    eff = eff or effective_wing_search(S)
    plan = eff.get("plan")
    if plan is None or plan.patience is None or plan.tol is None:
        return None
    patience, budget = int(plan.patience), int(eff["budget"])
    return {"can_fire": budget > patience + 1, "patience": patience,
            "needs": patience + 2, "budget": budget}


def adopt_recommendation(S: dict) -> None:
    """Copy the live recommendation into the user's own fields, then switch.

    What "switch to my own values" has to do: leaving stage 3 on a budget of
    40 the moment the user takes control would silently undo the
    recommendation they were looking at.
    """
    eff = effective_wing_search(S)
    W = S["wing"]
    W["optimiser"] = eff["optimiser"]
    W["acqf"] = eff.get("acqf") or W["acqf"]
    W["budget"] = int(eff["budget"])
    W["n_seeds"] = int(eff.get("n_seeds") or 1)
    section = {}
    # EVERY SURFACE, from the table — not a literal pair. This read
    # ("main", "aft"), so the vertical stabiliser's search budget was already
    # missed by "use these as my own", and a fourth surface would have been
    # missed the same way. `airfoil_state` raises KeyError for a surface with
    # no workspace, which is the guard below.
    for surface in SURFACES:
        try:
            oc = airfoil_state(S, surface)["opt"]
        except KeyError:
            continue
        a = effective_airfoil_search(S, surface)
        oc["optimiser"] = a["optimiser"]
        oc["budget"] = int(a["budget"])
        section[surface] = a
    # ...AND THE TWO DECISIONS WITH NO FIELD TO COPY THEM INTO. The mode
    # switch is read after this line, so the stash is written after it too.
    set_search_mode(S, "own")
    _stash_adopted_search(S, WING_SEARCH, eff)
    for surface, a in section.items():
        _stash_adopted_search(S, surface, a)


#: where :func:`adopt_recommendation` leaves the search decisions it has no
#: field to copy into: ``refusal`` (what a design the physics refused is shown
#: to the surrogate as) and ``n_init`` (the Sobol seed size). Stage 3's own
#: sits on ``S["wing"]``; each surface's on that surface's optimise state.
ADOPTED_SEARCH_KEY = "adopted_search"

#: :func:`adopted_search_carries`'s name for STAGE 3's search. Deliberately
#: not a surface name (``SURFACES``), so the two can never be confused.
WING_SEARCH = "wing"


def _adopted_home(S: dict, surface: str) -> dict | None:
    """The dict :data:`ADOPTED_SEARCH_KEY` lives in for one search."""
    if surface == WING_SEARCH:
        return S.get("wing")
    try:
        return airfoil_state(S, surface)["opt"]
    except (KeyError, TypeError):
        return None


def _stash_adopted_search(S: dict, surface: str, eff: dict) -> None:
    """Keep one search's fieldless decisions, if there were any to keep."""
    home = _adopted_home(S, surface)
    if home is None or eff.get("source") != "recommended":
        return
    home[ADOPTED_SEARCH_KEY] = {"refusal": eff.get("refusal"),
                                "n_init": eff.get("n_init")}


def clear_adopted_search(S: dict) -> None:
    """Forget every stashed carry — the mode was chosen by hand instead."""
    for surface in (WING_SEARCH, *SURFACES):
        home = _adopted_home(S, surface)
        if home is not None:
            home.pop(ADOPTED_SEARCH_KEY, None)


def adopted_search_carries(S: dict, surface: str = "main") -> dict:
    """The search decisions "use these as my own" cannot express, and keeps.

    ``{"refusal", "n_init"}`` for one search — :data:`WING_SEARCH` for stage
    3's, a surface name for that surface's section search — as the two OWN
    branches should report them.

    ``adopt_recommendation`` copies the numbers the stage has FIELDS for.
    ``refusal`` and ``n_init`` have none, so copying only the fields silently
    reverted the two decisions the study actually measured the moment a user
    took the budget over: the wing's flags went ``{acqf, bo_n_init: 4,
    bo_refusal: 'worst', bo_feasibility}`` -> ``{acqf, bo_feasibility}``, so
    every refused design was shown to the GP as the -100 sentinel again
    (worst beat sentinel on 20 of 24 seed pairs, p = 0.0015) and the Sobol
    seed grew 4 -> 12 of a 29-evaluation budget. The section search lost the
    same two on the same button (20/20 pairs, p = 2e-06).

    A HAND-PICKED "my own values" still drops them, which is the published
    behaviour and is asserted as such: nothing is stashed until the adopt
    button is pressed, and :func:`set_search_mode` clears the stash. This
    reads the stash and nothing else — asking the recommender again would
    make the two carries reappear on a plan the user never adopted.
    """
    got = (_adopted_home(S, surface) or {}).get(ADOPTED_SEARCH_KEY) or {}
    return {"refusal": got.get("refusal") or "sentinel",
            "n_init": got.get("n_init")}


def airfoil_state(S: dict, surface: str = "main") -> dict:
    """The airfoil workspace of one surface (``main`` / ``aft`` / ``fin``)."""
    return S[SURFACE_STAGES.get(surface, "airfoil")]


#: builder choices V3 PINS rather than asks about. Three different reasons:
#:
#: * ``flight`` — another stage already owns the question. The operating
#:   point is stage 1's: speed and altitude are typed there, the design point
#:   and every Reynolds number downstream are quoted at them, and a stage-3
#:   control that could hand the SAME two numbers to the optimiser made the
#:   mission's answer conditional on a toggle three stages later.
#: * ``tail_fin_drag`` — pinned here, and no longer WITHHOLDING anything,
#:   which are two different facts. There is nothing left to withhold: the
#:   fin's parasite drag is charged unconditionally now, the way the
#:   tailplane's always was, so no shell offers a switch and no flag carries
#:   one. The V1/V2 choice survives as a dead key this table keeps False.
#:
#:   The reason it was pinned is kept because every clause of it has since
#:   become false. It read:
#:   "the vertical fin is not a surface this package models: no panels in any
#:   solver, no lift, and nothing in geometry.py or cad.py — it survives as
#:   two Raymer scalars only, so a switch pricing it asks the user to
#:   underwrite a guess." That was true when written and V5 falsified it in
#:   four places at once: ``vlm.VerticalSurface`` puts the fin in the lattice
#:   with panels that carry side force, ``aerobo.fin`` gives it ONE size
#:   instead of the two it had, ``cad.fin_surface`` lofts and exports it, and
#:   ``design_report`` carries it so a rebuild flies the surface that was
#:   scored. There is now an honest answer to charge, so the shell asks it.
#:
#:   The lesson is not about the fin. A pin is a statement about the ENGINE
#:   at a moment in time, and nothing makes one re-read itself when the
#:   engine moves: this text went on justifying a withheld control for a
#:   session after its premise died. A pin should name what would falsify
#:   it — this one now does.
#: * ``tail_control`` — the choice cannot reach the answer. A hinged elevator
#:   enters the surface's boundary condition as the same uniform RHS shift as
#:   whole-surface incidence, so it trims to the SAME state with a larger
#:   deflection: ``tail.flap_effectiveness`` is a re-parameterisation, and
#:   L/D is equal exactly (test_api's ``==``, not ``approx``). Its one real
#:   effect is the deflection gate, and that gate never fires — over the
#:   whole 5-D tail box |i_t| reaches 7.0 deg (400/400 feasible) and the
#:   composed wing+tail 6.8 deg, so at the 25 deg limit it would take an
#:   elevator of c_e/c_t < 0.050 to fail anything the card can build. A
#:   toggle that changes no number the user will ever see is a question
#:   asked for nothing, so V3 flies the all-moving surface every time.
#:
#: The other two modifiers are untouched — the registry still carries every
#: ``+ free flight state`` twin and ``control="elevator"`` still works, and
#: V1/V2 still offer both; V3 simply does not ask. The fin's drag is the one
#: that stopped being a modifier at all. The elevator's one visible consequence, the hinged
#: surface in the CAD export (``cad.elevator_grid``), goes with it: what V3
#: exports is the surface it actually flew.
V3_PINNED_CHOICES: dict = {"flight": "fixed", "tail_fin_drag": False,
                           "tail_control": "stabilator"}


#: WHAT A FRESH AIR SESSION OPENS ON, over ``nice_app.BUILDER_START``.
#:
#: BUILDER_START is V1's opening wish and it is a WING ALONE. A wing alone
#: has no vertical surface, so it has no yaw stiffness, no spiral mode and
#: nothing for the Controls and Flight stages to fly: the shell opened on an
#: aeroplane whose lateral half did not exist and then reported "no spiral
#: stability" the moment a tail was switched on.
#:
#: So V3 opens on a whole aeroplane, and on the CHEAPEST configuration that
#: can carry the answer:
#:
#: * ``tail`` — a second surface, which is what makes a fin, a trim state
#:   and an empennage exist at all;
#: * ``tail_height`` free — the surface comes off the wing plane, which is
#:   what makes the family LATTICE-backed. That matters because a lifting
#:   line has no out-of-plane geometry and therefore cannot score a wing
#:   dihedral, and the dihedral is the ONLY lever that turns this
#:   configuration's spiral (no fin size does — ``controls._scan_spiral``).
#:   It costs one design variable and no surface, which is the cheapest
#:   answer that reaches a cant-capable family at all (the others are a tip
#:   device, a designed tail, or searching the cant). Stage 3 used to offer
#:   those four as BUTTONS on a refusal card; the card is gone, so this
#:   start choice is now the whole reason the shell opens on a family that
#:   can be asked the question.
#:
#: It is a starting POINT, not a pin: every one of these is a control the
#: user can move, and V3_PINNED_CHOICES above is the list of questions V3
#: does not ask at all.
V3_START_CHOICES: dict = {"tail": True, "tail_height": "free"}

#: ...and the two NUMBERS that opening aeroplane flies, applied only where
#: the derived family declares them.
#:
#: ``wing_dihedral_deg`` 7.0 — MEASURED on the family above (its box centre,
#: ``scripts/spiral_dihedral_table.py``). Its spiral criterion crosses zero
#: at 5.57 deg, but the crossing is a NEUTRAL aeroplane: the 6-DOF spiral
#: root there is -0.0013 and the mode halves in 548 s. At 7.0 it is -0.0323
#: and halves in 21 s, which is a spiral a pilot would call stable, and the
#: whole of it costs 0.67 % of L/D. Opening on the crossing would have been
#: the same mistake the 5.7 deg step made — a number that clears the sign
#: and not the question (RESULTS_SESSION69_SPIRAL_RECOMMENDATION.md).
#:
#: ``fuselage_diameter_m`` — the diameter that puts the body at
#: :data:`V3_FUSELAGE_FINENESS` for the arm this family's box centres on.
#: Without a body the searched arm is FREE — its drag unmodelled while its
#: static margin and tail volume are fully counted — and the search runs to
#: the top of the arm band every time.
V3_START_FLAGS: dict = {"wing_dihedral_deg": 7.0}

#: how slender the body a fresh session opens with is, as a fineness ratio.
#:
#: TWO MEASUREMENTS PICK IT, and neither is a taste.
#:
#: * it leaves the searched arm an INTERIOR optimum, which is the whole
#:   reason a body is charged at all. Swept on the opening family's box
#:   centre (``tail [free height]``, arm band 3-8 m): with no body the best
#:   arm is 8.00 m — the top bound — and at fineness 40 it is 3.60 m. Go the
#:   other way and the body over-penalises: at 16.4 (the form factor's own
#:   minimum, ``drag.SLENDER_FINENESS``) the arm rides the BOTTOM bound;
#: * and it is slender enough that the shape penalty has stopped mattering:
#:   ``drag.fuselage_cd0``'s form factor is 1.101 here against its minimum
#:   1.055, so 4.4 % — inside 5 %.
#:
#: It costs L/D and it is meant to: 30.92 uncharged to 27.39 at the box
#: centre. That is the price of a well-posed arm, not a tax — the run that
#: does not pay it answers "the longest aeroplane the box allows".
V3_FUSELAGE_FINENESS: float = 40.0


def default_optimiser(problem_name: str) -> str:
    """The optimiser the WING stage opens a family on.

    ``bo_slsqp`` where the family has it, plain ``bo`` where it does not.
    That is not a preference: ``RESULTS_HANDOFF.md`` scored 1,558 budget-fair
    runs against certified optima and `bo -> slsqp` beat pure BO on 42 of 54
    independent runs (p = 5.2e-05), the only pair measured that beats both its
    parents pairwise. On the 53 runs carrying every constrained arm it takes
    the three best median gaps of nineteen arms.

    It is CONSTRAINED-ONLY, because ``slsqp`` is, so an unconstrained family
    silently keeps plain ``bo`` — and that is the right answer there anyway:
    the unconstrained analogue the study ran, `bo -> gradient`, LOSES to pure
    BO at the two coarse epsilon rungs (-1.7 and -10.8 points). The default
    goes exactly as wide as the evidence and no wider.

    Read by every site that has to pick one, so the answer cannot drift
    between the opening session, a family change and the solver card.
    """
    from aerobo import api
    names = api.compatible_optimisers(problem_name)
    for pick in ("bo_slsqp", "bo"):
        if pick in names:
            return pick
    return names[0]


def _start_flags(problem: str, medium: str) -> dict:
    """:data:`V3_START_FLAGS` for the family a fresh session derives.

    Filtered by the registry, never by a name: a flag the family does not
    declare is one ``api.check_flags`` refuses, and a shell that wrote it
    anyway would fail at launch rather than on the card that set it.

    The fuselage diameter is COMPUTED here rather than stored as a number,
    because the honest statement is about the body's SHAPE: it is whatever
    puts this family's own arm at ``drag.SLENDER_FINENESS``. Stored as a
    constant it would be a diameter measured on one family's arm band and
    quoted at every other — the defect the dihedral recommendation exists to
    close, in the other column.
    """
    from aerobo import api, drag

    if medium != "air":
        return {}
    spec = api.PROBLEM_SPECS.get(problem)
    if spec is None:
        return {}
    out = {k: v for k, v in V3_START_FLAGS.items() if k in tuple(spec.flags)}
    if api.FUSELAGE_KEYS[0] in tuple(spec.flags):
        arm = _default_arm_m(spec)
        if arm:
            out[api.FUSELAGE_KEYS[0]] = round(
                drag.diameter_for_fineness(arm, V3_FUSELAGE_FINENESS), 3)
    return out


def _default_arm_m(spec) -> float | None:
    """The wing-to-tail arm this family opens on: its own flag, or the
    centre of the design box row it searches."""
    row = (spec.default_bounds or {}).get("l_t_m")
    if row:
        return 0.5 * (float(row[0]) + float(row[1]))
    return None


def make_session(medium: str = "air") -> dict:
    """A fresh session, opened on ``medium``'s published operating point."""
    from gui.nice_app import (PROP_DEFAULTS, derive_problem, start_choices,
                              tail_design_start)

    # BUILDER_START, not BUILDER_DEFAULTS: the session OPENS on the polynomial
    # chord law, and start_choices demotes it to straight taper on the one
    # family that has no chord-law twin, so no stage ever renders a state its
    # own menu does not offer.
    # ...and V3_START_CHOICES over that, in AIR only: the opening aeroplane
    # is a whole one (a tail, and the lattice family that can carry a wing
    # dihedral), because a wing alone has no spiral to be stable. Water and
    # the track open where they always did — a hydrofoil has no fuselage and
    # no dihedral question, and a rear wing is not a free-flying aeroplane.
    start = dict(V3_PINNED_CHOICES)
    if medium == "air":
        start.update(V3_START_CHOICES)
    choices = start_choices(medium=medium, **start)
    # ...AND THAT SURFACE OPENS DESIGNED, exactly as one the user switches on
    # does. ``mission.set_tail`` applies ``nice_app.tail_design_start`` the
    # moment a second surface appears — "a stabiliser is a SURFACE, not a
    # fitting" — but the opening aeroplane carries one from its first paint,
    # so it never went through that handler: every fresh air session held
    # ``tail_design="fixed"``, which is the published AR-4 rectangle. It cost
    # the box the surface's own three rows (taper_t, AR_t, washout_t_deg) and
    # it greyed the tip-device menu out ("design this surface's planform
    # first"), which is what the user saw. Asked of the registry, never
    # assumed: a family with no designed-tail twin still opens on its
    # rectangle, and re-normalised through the one entry point so the opening
    # state cannot hold a value its own menus do not offer.
    if choices.get("tail"):
        start["tail_design"] = tail_design_start(choices)
        choices = start_choices(medium=medium, **start)
    problem = derive_problem(choices)[0]

    S_state = {
        # WHAT THIS SESSION DESIGNS (:data:`MODES`). A fresh session is the
        # published pipeline; the airfoil-only mode is a deliberate choice,
        # made on stage 1, and it keeps its own stated flow beside the
        # mission rather than instead of it.
        "mode": MODE_DEFAULT,
        "medium": medium,
        "water": "sea",              # which water (api.water_kinds())
        # WHICH optimiser and HOW MANY evaluations — asked once, at stage 1,
        # for every stage that searches (:func:`search_state`)
        "search": _search_state(),
        "mission": mission_defaults(problem, medium),
        "airfoil": {
            **_airfoil_workspace(DEFAULT_WEIGHTS),
            # None = nobody has chosen one, which IS the statement "the wing
            # family flies its own published section". There is no separate
            # "solver" value any more: it declared exactly what None already
            # meant, and every reader had to spell both.
            "decision": None,            # None | library | optimised
            # the aspect-ratio ESTIMATE the section's chord (and so its
            # Reynolds number) is derived from — stage 2 owns it, stage 3
            # owns the span that is actually flown
            "ar": default_aspect_ratio(problem),
            "section": None,             # the chosen section (see below)
            # the SECOND surface's own section (tail / elevator, or a tandem
            # pair's rear wing). None = it flies the wing's, which is what
            # the solvers do by default (tail.polar_tail / tandem.polar_rear)
            "section_aft": None,
        },
        # stage 2.5's workspace. The DECISION keys stay on the wing's, above:
        # what unlocks stage 3 is the wing's section, and this surface's is an
        # addition to it (:func:`set_section`).
        "airfoil_aft": _airfoil_workspace(DEFAULT_WEIGHTS),
        # the FIN's own workspace. Same shape as the other two so the one
        # stage module builds all three; what differs is the question it is
        # allowed to ask, not the state it keeps.
        "airfoil_fin": _airfoil_workspace(DEFAULT_WEIGHTS),
        # ...and the ENDPLATE's, on the same footing: its own screen, its own
        # weights and its own shape search, so a section chosen for the plate
        # cannot land on the wing (`airfoil_state` keys the workspace by the
        # STAGE, which is why a fourth surface needs both a SURFACE_STAGES
        # entry and a same-named dict here).
        "airfoil_plate": _airfoil_workspace(DEFAULT_WEIGHTS),
        "wing": {
            "choices": choices,
            "problem": problem,
            # NO aspect ratio of its own. It was state with no control the
            # moment the span became stage 3's question: nothing on this stage
            # asks for an aspect ratio, because b²/S is not a question — the
            # SPAN is (:func:`chosen_span`, :func:`span_box`), and the aspect
            # ratio is whatever it implies (:func:`flown_aspect_ratio`).
            #
            # The SPAN the size card holds, in metres. None = nobody chose
            # one, so the run flies √(AR_estimate · S) and is bit-for-bit the
            # published planform. Only the FIXED planform reads it (a family
            # whose span is a design variable is not handed one), but it is
            # kept across a mode change, exactly like a typed design-box
            # bound: it is the user's answer to a question they may switch
            # back on.
            "span_m": None,
            # ...or the same size stated as the two END CHORDS, in metres
            # (SIZE_MODES). "span" is the published question and the default;
            # in "chords" the taper is pinned at c_tip/c_root and the span is
            # derived from the mission's area (:func:`chord_span`).
            "size_mode": "span",
            "chord_root_m": None,
            "chord_tip_m": None,
            "bounds": {},
            # which of those bands the SHELL derived from the mission (rather
            # than the user typing them) — see :func:`bounds_source`
            "bounds_source": {},
            # design-box rows switched OFF: not constrained by anything this
            # session says, so the solver's own published box stands. The
            # typed numbers stay in "bounds" and come back with the switch
            # (config.released_rows).
            "bounds_off": [],
            # design-box rows the user has DECIDED: {label: value}. The
            # variable leaves the design vector — the optimiser never sees it
            # — and travels as api.RunConfig.pinned. Empty is the published
            # behaviour: everything the family declares is searched.
            "fixed": {},
            # a DESIGN this stage starts its next run from, in full design-
            # vector coordinates (api.RunConfig.x_seed). None is every
            # published run: a seed is something the session was GIVEN — the
            # design that came closest on a run that found nothing, or a
            # previous answer — never something it invents.
            "x_seed": None,
            # THE OPENING AEROPLANE'S OWN NUMBERS (:data:`V3_START_FLAGS`),
            # filtered to what this family actually declares — a flag it
            # does not honour would be accepted and silently ignored, which
            # ``api.check_flags`` refuses outright.
            "flags": _start_flags(problem, medium),
            "mission_edits": {},
            "prop": dict(PROP_DEFAULTS),
            "optimiser": default_optimiser(problem),
            "acqf": "logei",
            "budget": 40,
            "seed": 0,
            "n_seeds": 1,
            "block_optimisers": {},
            "apply_mach": False,         # send the derived Mach to the run
            "section_link": "auto",      # auto | none
            "section_half_width": 0.06,  # CST box half-width when pinning
            # WHICH SCALAR this stage's search maximises, and the band it is
            # measured against (:func:`wing_score_state`). Opens on the
            # family's own objective: the composite is a deliberate choice,
            # never something a stage inherits.
            "score": _wing_score_state(),
        },
        # ...and the two BACKGROUND workers stage 3 owns beside the search
        # itself, declared here rather than created on first use: File > New
        # session refills these sub-dicts in place, and a key that only ever
        # existed because a worker had run once would survive a new session
        # carrying the previous one's answer.
        "run": {"record": None, "report": None, "section_report": None,
                "error": None, "stamp": None,
                "relax": None, "box_probe": None},
        # WHERE a CAD export lands. A folder the user owns, asked once and
        # remembered for the session: an export the browser downloads goes
        # somewhere the shell cannot name, and "open it in OpenVSP" needs a
        # path on this machine, not a download.
        "export": {"dir": str(DEFAULT_EXPORT_DIR), "stem": "aerobo_design",
                   "last": None},
        "ui": {"selected": "mission",
               "tab": {s: VIEWS[s][0][0] for s in STAGES},
               "expanded": {s: (s == "mission") for s in STAGES}},
    }
    sync_wing_from_mission(S_state)
    refresh_recommended_weights(S_state)
    # ...and the FLOW the airfoil-only mode states, built last because it
    # opens on the point the mission just built implies (:func:`flow_defaults`)
    # — the two modes then start on the same aerodynamics. Created here, not
    # on first use, for the reason every other worker dict is: File > New
    # session refills the top-level sub-dicts in place, and a key that only
    # existed once something had asked for it would survive a reset carrying
    # the previous session's answer.
    S_state["flow"] = flow_defaults(S_state)
    return S_state


# ------------------------------------------------------ aspect ratio (two)
def section_aspect_ratio(S: dict) -> float:
    """Stage 2's ESTIMATE — what the section's chord is derived from.

    Everything this number reaches is a section decision: the MAC, the
    Reynolds number the library is screened at, and the wing the wing-L/D
    objective flies its candidate sections on. It is not a commitment to a
    planform (:data:`DEFAULT_ASPECT_RATIO`).
    """
    try:
        ar = float(S["airfoil"].get("ar"))
    except (TypeError, ValueError):
        ar = 0.0
    return ar if ar > 0.0 else default_aspect_ratio(S["wing"]["problem"])


def set_section_aspect_ratio(S: dict, value) -> bool:
    """Set the estimate. False (and nothing stored) if it is unusable."""
    try:
        ar = float(value)
    except (TypeError, ValueError):
        return False
    if not ar > 0.0:
        return False
    S["airfoil"]["ar"] = ar
    return True


def nominal_aspect_ratio(S: dict) -> float:
    """The aspect ratio the session SIZES a fixed-span wing at.

    Stage 2's estimate, and nothing else — there is no second, wing-owned
    aspect ratio any more, because nothing on screen asks for one. Stage 3
    constrains the SPAN (:func:`span_box`) and the aspect ratio it flies is
    b²/S, reported by :func:`flown_aspect_ratio`. This is the number that
    turns the mission's AREA into the span of a wing whose span is NOT
    searched, which is the only remaining job for a guess.

    Named apart from :func:`section_aspect_ratio` because the two questions
    are different even where the answer is the same one number: that one is
    "which chord was the section designed for", this one is "how wide is the
    wing we have not constrained".
    """
    return section_aspect_ratio(S)


def section_point_is_stale(S: dict, surface: str = "main"):
    """``(screened Re, current Re)`` when the ranking on screen was produced
    at a different design point, else ``None``.

    A ranking is a list of sections ordered at ONE Reynolds number and one
    design lift. Move either — by editing the mission, or the aspect-ratio
    estimate the chord comes from — and the order on screen is an answer to
    a question nobody is asking any more. The shell reports that rather than
    silently re-labelling the table.
    """
    rep = airfoil_state(S, surface)["screen"].get("report") or {}
    try:
        was = float((rep.get("conditions") or {}).get("re"))
    except (TypeError, ValueError):
        return None
    if not was > 0.0:
        return None
    now = float(section_conditions(S, surface)["re"])
    if abs(now - was) <= 1e-9 * max(abs(now), abs(was)):
        return None
    return was, now


def arm_rescreen(S: dict, surface: str, reason: str, re: float | None = None,
                 cost_hint: str = "") -> dict:
    """Record that this surface's section stage OUGHT to be re-screened.

    The last open half of the stage-2 <-> stage-3 loop. A disagreement between
    the point the section was screened at and the point the surface actually
    flies was already detected and reported on both stages, and resolving it
    re-pointed the estimate — but the user was then handed back a *log line*
    naming a control to go and press. Two clicks and a sentence to follow, for
    a decision they had already taken.

    What this does NOT do is start the sweep. That is deliberate and it is the
    same rule the rest of the shell obeys: a re-screen at a surface's own
    Reynolds number is real XFOIL, minutes on the first visit, and this shell
    does not start expensive work the user cannot see the price of. What it
    does is carry the REQUEST to the one place that can state that price, so
    the destination opens on a primed button instead of an instruction.

    Cleared by :func:`disarm_rescreen` — on completion, on dismissal, or when
    the point moves again and the pending request is no longer the right one.
    """
    a = airfoil_state(S, surface)
    a["rescreen"] = {"reason": str(reason),
                     "re": (None if re is None else float(re)),
                     "cost_hint": str(cost_hint),
                     "stamp": time.time()}
    return a["rescreen"]


def disarm_rescreen(S: dict, surface: str = "main") -> None:
    """Forget a pending re-screen request (done, dismissed, or superseded)."""
    airfoil_state(S, surface)["rescreen"] = None


def pending_rescreen(S: dict, surface: str = "main") -> dict | None:
    """The pending request, if it is still the right one.

    A request that names a Reynolds number the surface has since left is
    stale and reads as absent: it would otherwise offer to re-screen at a
    point that is no longer this surface's own, which is the very failure the
    request exists to fix.
    """
    a = airfoil_state(S, surface)
    req = a.get("rescreen")
    if not req:
        return None
    want = req.get("re")
    if want is None:
        return req
    try:
        now = float(section_conditions(S, surface)["re"])
    except Exception:                        # noqa: BLE001 — never crash a view
        return req
    if abs(now - float(want)) > 1e-6 * max(abs(now), abs(float(want))):
        a["rescreen"] = None
        return None
    return req


def section_lift_is_stale(S: dict, surface: str = "main"):
    """``(screened Cl, current Cl)`` when the table was ordered at another
    design lift, else ``None``.

    The Reynolds check above cannot see this one on the CACHED library point:
    both surfaces are screened at the same cached Re, so a ranking produced
    for a lifting surface and then read for a trimming one has an identical
    Reynolds number and a completely different order. The design lift is what
    actually moved — a tandem's rear wing changing into a tail, or the
    mission's own CL being edited — so it is checked in its own right.
    """
    rep = airfoil_state(S, surface)["screen"].get("report") or {}
    cond = rep.get("conditions") or {}
    if "cl_design" not in cond:
        return None
    try:
        was = float(cond["cl_design"])
    except (TypeError, ValueError):
        return None
    now = float(section_conditions(S, surface)["cl_design"])
    if abs(now - was) <= 1e-9 * max(1.0, abs(now), abs(was)):
        return None
    return was, now


def flown_size(S: dict) -> tuple[float, float] | None:
    """``(span, area)`` the run will fly, or ``None`` where it is not ours.

    Only a family that honours a chosen size (``api.resizable``) has a size
    this shell decides; everything else carries calibrated geometry, and the
    honest answer is that the planform is the family's own.

    With the FIXED planform the span is the one the size card holds
    (:func:`chosen_span`), and √(AR_estimate · S) only where nobody has
    chosen one — a guess is what a shell falls back to, never what it
    overwrites an answer with.

    With the SPAN CONSTRAINED (:func:`span_is_searched`) there is no single
    span to report — the optimiser decides one inside the box — so the span
    quoted here is the MID-BOX one, the same convention every other derived
    number in this shell follows (:func:`surface_geometry`, :func:`taper_box`).
    It moves when the box does.

    A span-searching family does NOT declare the planform flags — a wing is
    sized one way, so a family whose span is in the vector may not also be
    handed a chosen one (tests/test_wing_loading_size.py) — which is why
    ``api.resizable`` is not the whole question here: its area is the
    mission's, through the wing loading, and its span is the box.
    """
    from aerobo import api

    if not (api.resizable(S["wing"]["problem"]) or span_is_searched(S)):
        return None
    # ...and the AREA IS A BOX ROW WHEREVER IT IS SEARCHED, read the same way
    # and quoted by the same convention as the span above it: the middle of
    # the row the run will search. The mission's area is what a family that
    # STATES its planform flies; the water size modes put S_m2 in the design
    # vector (api.WET_SIZE_KEYS), and reading the mission there made every
    # number derived from this one a statement about a wing the run does not
    # fly. Measured on the water default with both size rows open: the card
    # quoted AR 15.6 against the solver's own 12.5 at the same box mid, and
    # the aspect-ratio band 2.5-40 against corners the solver refuses at 1.25
    # and 80 -- and then offered to re-design the section for the 15.6.
    area_box = span_box(S, "S_m2")
    if area_box is not None:
        area = 0.5 * (area_box[0] + area_box[1])
    else:
        try:
            area = float(S["mission"]["s_ref_m2"])
        except (TypeError, ValueError):
            return None
    if not area > 0.0:
        return None
    box = span_box(S)
    if box is not None:
        return 0.5 * (box[0] + box[1]), area
    return nominal_span(S), area


# --------------------------------------------------------------- the SPAN
#
# A SPAN BELONGS TO A SURFACE, and a family may carry more than one that has
# a span of its own: a tandem pair's two wings are two wings, and tying the
# rear one to the front was a modelling shortcut, not a property of the
# aeroplane (sizing.span_labels -> ``b_m``, ``b_rear_m``). So every question
# this shell asks about the span is asked per ROW, and the two states stay
# the two states they were:
#
#   FIXED     one typed LENGTH per surface, on the size card;
#   SEARCHED  one BAND per surface, in the design box.
#
# WHICH rows exist is asked of the registry (:func:`span_rows`), never of a
# problem name: the same pair appears under a dozen names once the section,
# tip-device and size twins are generated.
#
#: where a typed span is STORED, per design-box row. The front wing's is the
#: shell's own (``S["wing"]["span_m"]``, pushed into the builder choices by
#: :func:`sync_wing_from_mission`); the rear wing's is a builder CHOICE
#: outright, because ``gui.nice_app.tandem_flags`` is the one emitter of the
#: ``b_rear_m`` flag and V1/V2 ask the same question through it. One flag,
#: one path to it.
SPAN_CHOICE_KEYS = {"b_rear_m": "tandem_b_rear_m"}

#: how a span row is NAMED where a surface has to be named. Single-surface
#: families never see these — there is one span and it is "the span".
SPAN_ROW_LABELS = {"b_m": "front wing", "b_rear_m": "rear wing"}


def span_rows(S: dict) -> tuple:
    """The span rows this family carries, in surface order.

    Read off the REGISTRY — a row counts when the family either declares it
    as a design-vector label (the span is searched) or accepts it as a flag
    (the span is typed) — so the answer is the same question in both states
    and no helper has to branch on a problem name. Every family has ``b_m``;
    the tandem pair adds ``b_rear_m``.
    """
    from aerobo import api
    from aerobo.sizing import span_labels

    spec = api.PROBLEM_SPECS[S["wing"]["problem"]]
    known = set(spec.param_labels) | set(spec.flags)
    return tuple(r for r in span_labels(2) if r in known)


# ----------------------------------------------------- the size, as CHORDS
#
# A trapezoid has three numbers and the shell states two of them: the area
# (stage 1, through the wing loading) and the span. The third — the taper — is
# a design variable, so the ROOT and TIP chords are outputs of the search
# rather than anything a user can state.
#
# That is the wrong way round for a great many real wings. A spar depth, a
# control-surface hinge, a mould, a rib kit or a class rule states a CHORD, in
# millimetres, at the root and at the tip; the span is then whatever those
# imply. So the size card can be asked the other way: give the two chords, and
#
#     taper = c_tip / c_root                (pinned — not searched)
#     b     = 2 S / (c_root + c_tip)        (the area is still the mission's)
#
# Nothing about the area moves: stage 1 still owns it, which is why this is a
# re-parameterisation of the size card rather than a second place to state it.
#: how the size card is being ASKED. "span" is the published question.
SIZE_MODES = ("span", "chords")


def size_mode(S: dict) -> str:
    return "chords" if str(S["wing"].get("size_mode")) == "chords" else "span"


def chosen_chords(S: dict) -> tuple[float, float] | None:
    """``(root chord, tip chord)`` [m] the size card holds, or None.

    Both or neither: one chord alone says nothing about a planform that the
    span does not already say, and half an answer would silently fall back to
    a taper the user did not choose.
    """
    W = S["wing"]
    try:
        c_r = float(W.get("chord_root_m"))
        c_t = float(W.get("chord_tip_m"))
    except (TypeError, ValueError):
        return None
    if not (c_r > 0.0 and c_t > 0.0):
        return None
    return c_r, c_t


def chord_taper(S: dict) -> float | None:
    """The taper the two stated chords imply, or None where none are stated.

    Gated on :func:`chord_span_available` as well as on the mode, because
    this is the number that becomes a PIN (``config.fixed_rows``) and a pin
    has to be removable. The chords survive a family change — the card that
    offered them may not — and on a family whose size block is not even
    drawn ("car rear wing + free chord law": ``api.resizable`` is False) the
    stale pair still pinned ``taper`` at 0.3333 on a design nobody had typed
    a taper for, with no control anywhere able to take it off again.
    """
    ch = chosen_chords(S)
    if ch is None or size_mode(S) != "chords" or not chord_span_available(S):
        return None
    return ch[1] / ch[0]


def chord_span(S: dict) -> float | None:
    """The SPAN the two stated chords imply, at the mission's area.

    b = 2S / (c_root + c_tip), which is the trapezoid's own area identity
    read for b. None where the chords are not the question being asked, or
    where this family's area is not the mission's to give.

    THIS SURFACE'S SHARE of the area, the way :func:`_span_readout` already
    reads it: the mission states the PAIR's total, so dividing by all of it
    on a tandem doubled the span the chords implied — a 13.33 m front wing
    became 26.67 m at aspect ratio 71, past ``api.PLANFORM_AR_LIMITS``, while
    the card's own honesty note computed b²/S_total = 35.6 and stayed silent.
    The gate below already keeps every pair out of this mode today (a tandem
    names its rows ``taper_front``/``taper_rear``, so it has no ``taper`` to
    pin and the share is 1.0 wherever the chords can be asked at all); the
    share is read here anyway so that the two can never disagree if one does.

    Gated on :func:`chord_span_available` for the reason :func:`chord_taper`
    gives: the chords outlive the card that offered them, and a span derived
    from a pair of numbers nothing on screen can show is not a size anybody
    stated. Measured before the gate: switching the planform to wing loading
    left the chords setting ``nominal_span`` at 16.67 m, so ``b_m`` opened on
    10–66.7 m instead of the mission's 6–40 m.
    """
    ch = chosen_chords(S)
    if ch is None or size_mode(S) != "chords" or not chord_span_available(S):
        return None
    try:
        area = float(S["mission"]["s_ref_m2"]) * wing_area_share(S, "b_m")
    except (TypeError, ValueError):
        return None
    total = ch[0] + ch[1]
    if not (area > 0.0 and total > 0.0):
        return None
    return 2.0 * area / total


def chord_span_available(S: dict) -> bool:
    """Can this family's planform be stated as two CHORDS?

    It needs both halves of the arithmetic: an area this shell gives it (so
    the span can be derived) and a ``taper`` row to pin (so the ratio is a
    statement rather than a wish). A family that SEARCHES its span has
    neither — its size is the design vector's — so the card keeps asking the
    only question that family answers.
    """
    from aerobo import api

    from . import config

    name = S["wing"]["problem"]
    if not api.resizable(name) or span_is_searched(S):
        return False
    return "taper" in config.default_bounds(S)


def set_size_mode(S: dict, mode: str) -> str:
    """Ask the size card for the span, or for the two end chords."""
    S["wing"]["size_mode"] = "chords" if str(mode) == "chords" else "span"
    # the SIZE the run is handed is pushed through the same path a typed span
    # takes (``choices["span_m"]``/``["area_m2"]``, sync_wing_from_mission),
    # so there is one definition of "the size this session flies" and the
    # chords cannot become a second one
    sync_wing_from_mission(S)
    return S["wing"]["size_mode"]


def clear_size_chords(S: dict) -> None:
    """Take the size card back to the SPAN question and forget the chords.

    NO STATE OUTLIVES ITS OWN CONTROL — the rule the span, the section and
    the fixed rows already follow in :func:`apply_choices`. Two chords in
    metres are a statement about the wing that was on screen when they were
    typed: a 1.2 m hydrofoil is not that wing (every air↔water carry-over is
    refused outright — the admissible ``c_root + c_tip`` intervals at 10 m²
    and 0.144 m² are disjoint), and neither is the next family, which may not
    draw the size block at all.

    Called with the span, so the two halves of one answer fall together.
    """
    W = S["wing"]
    W["size_mode"] = "span"
    W["chord_root_m"] = None
    W["chord_tip_m"] = None


def set_size_chord(S: dict, which: str, value) -> bool:
    """Store one end chord [m]; empty clears it. False = not a length."""
    key = "chord_root_m" if str(which) == "root" else "chord_tip_m"
    if value in (None, ""):
        S["wing"][key] = None
        sync_wing_from_mission(S)
        return True
    try:
        c = float(value)
    except (TypeError, ValueError):
        return False
    if not c > 0.0:
        return False
    # the SPAN this chord implies is the span the run flies, so it faces the
    # same refusal a typed one does — before it is kept, or the sync below
    # raises out of the handler with the number already stored. Written as
    # store-probe-restore because the answer is ``chord_span`` itself: one
    # definition of the derived span, not a second copy of the arithmetic.
    prev = S["wing"].get(key)
    S["wing"][key] = c
    derived = chord_span(S)
    if derived is not None and planform_size_refusal(
            S, span=derived) is not None:
        S["wing"][key] = prev
        return False
    sync_wing_from_mission(S)
    return True


def chosen_span(S: dict, row: str = "b_m") -> float | None:
    """The span the size card holds for ``row`` [m], or ``None`` where nobody
    chose one.

    A stored number, not a derived one: it is the answer to the one question
    a fixed planform leaves open once the mission has stated the area. The
    aspect ratio is NOT that question — it is b²/S, and a shell that asked
    for it would be asking for the span in a unit nobody's hangar is
    measured in (:func:`flown_aspect_ratio`).

    ...unless the card is being asked in CHORDS (:data:`SIZE_MODES`), in
    which case the wing's span IS derived — from the two chords and the
    mission's area — and this returns that, because it is still the span the
    run flies and every caller here wants the one that flies.
    """
    if row == "b_m":
        derived = chord_span(S)
        if derived is not None:
            return derived
    W = S["wing"]
    key = SPAN_CHOICE_KEYS.get(row)
    raw = W["choices"].get(key) if key else W.get("span_m")
    try:
        b = float(raw)
    except (TypeError, ValueError):
        return None
    return b if b > 0.0 else None


def nominal_span(S: dict, row: str = "b_m") -> float:
    """The span a run flies with the planform FIXED [m].

    The chosen one where there is one, and √(AR_estimate · S) otherwise —
    stage 2's guess, which exists so the section has a chord to be designed
    for. Every band this shell opens is taken around this number, so it may
    not itself depend on any band (see :func:`span_band_default`).

    A SECOND surface with no span of its own typed falls back to the FIRST
    one's, which is what its solver does: a tandem's rear wing is as wide as
    the front wing until someone says otherwise (``tandem.TandemProblem.b_r``).
    """
    b = chosen_span(S, row)
    if b is not None:
        return b
    if row != "b_m":
        return nominal_span(S)
    try:
        area = float(S["mission"]["s_ref_m2"])
    except (TypeError, ValueError):
        return 0.0
    return (nominal_aspect_ratio(S) * area) ** 0.5 if area > 0.0 else 0.0


def planform_size_refusal(S: dict, span=None, area=None) -> str | None:
    """Why the ENGINE would refuse this (span, area) pair, or ``None``.

    The same test ``api._planform_size`` applies to the size flags this shell
    sends — b²/S against ``api.PLANFORM_AR_LIMITS`` — asked BEFORE the number
    is stored, because there it is a ``ValueError`` raised out of
    ``api.default_mission_values`` inside a NiceGUI event handler: no notify,
    no log line, no card repainted, and the out-of-band number left sitting
    in the state. Type 25 m on the default 10 m² wing (the band is 5.48-20 m)
    and nothing on screen moved, while ``build_cfg`` still passed and the job
    died in the worker.

    A REFUSAL and not a warning, which this repo does not do lightly ("a
    calibration is a default, not a ban"). The difference here is that the
    band is not a preference the shell holds: below ~3 the lifting-line
    reduction and the one-chordwise-panel Weissinger VLM have stopped
    describing a wing and above ~40 the fixed single-Re section polar has,
    so the api refuses the build outright rather than scoring it. A shell
    that stored the number anyway would not be respecting the user's answer,
    it would be holding a session that cannot produce one.

    ``None`` on a family whose size this shell does not send (``api.resizable``
    — the car and the fixed-geometry water families), where nothing is
    validated because nothing travels.
    """
    from aerobo import api

    if not api.resizable(S["wing"]["problem"]):
        return None
    try:
        b = float(nominal_span(S) if span is None else span)
        s = float(reference_area(S) if area is None else area)
    except (TypeError, ValueError):
        return None
    if not (b > 0.0 and s > 0.0):
        return None
    lo, hi = api.PLANFORM_AR_LIMITS
    ar = b * b / s
    if lo <= ar <= hi:
        return None
    return (f"a {b:.4g} m span on {s:.4g} m² is aspect ratio {ar:.3g}, "
            f"outside the {lo:g}–{hi:g} these solvers are valid over — "
            f"{(lo * s) ** 0.5:.4g}–{(hi * s) ** 0.5:.4g} m at this area, "
            f"or {b * b / hi:.4g}–{b * b / lo:.4g} m² at this span")


def set_span_m(S: dict, value, row: str = "b_m") -> bool:
    """Choose the span of ``row`` [m]; ``False`` (and nothing stored) if it
    is unusable.

    Clearing the field gives the span back to what it was derived from — the
    aspect-ratio estimate for the wing, the front wing's span for a second
    surface — which the size card says, and which is what makes an untouched
    session still the published run, bit-for-bit.
    """
    W = S["wing"]
    key = SPAN_CHOICE_KEYS.get(row)
    if value in (None, ""):
        if key:
            W["choices"][key] = None
        else:
            W["span_m"] = None
            sync_wing_from_mission(S)
        return True
    try:
        b = float(value)
    except (TypeError, ValueError):
        return False
    if not b > 0.0:
        return False
    # ...and it has to be a span this AREA can carry: the pair is what the
    # size flags state, and an out-of-band aspect ratio raises out of the
    # sync below rather than being reported (:func:`planform_size_refusal`).
    # The rear row is not checked here because it does not travel through
    # ``api._planform_size`` — it is a tandem flag of its own.
    if not key and planform_size_refusal(S, span=b) is not None:
        return False
    if key:
        W["choices"][key] = b
    else:
        W["span_m"] = b
        sync_wing_from_mission(S)
    return True



# The span is a CONSTRAINT, not a derived number, and it is asked in exactly
# one place: the ``b_m`` row of stage 3's design box. Everything else about
# the planform follows it — the aspect ratio the run flies is b²/S, with S
# fixed by the mission's wing loading, so AR stops being a question anyone is
# asked and becomes a consequence to report.
#
# Two states, one switch:
#
#   OFF  the span is the constructor value √(AR_estimate · S). Nothing
#        constrains it because nothing searches it: stage 2's aspect-ratio
#        ESTIMATE is a guess, and the run flies the wing that guess implies.
#   ON   the span joins the design vector inside the band typed in that row,
#        and the AREA leaves it: it follows the mission's W/S through the
#        weight loop (the ``size_ws`` modifier — sizing.py). That is the pair
#        the constraint needs. Freeing the span alone against a FIXED area
#        would be freeing the aspect ratio while pretending the mission had
#        not already answered the loading.
def span_row_available(S: dict) -> bool:
    """Can this family search its span at the mission's wing loading?

    Asked of the REGISTRY (``api.with_modifiers``), never of a problem-name
    list kept here: the families wired for the mode are declared once, in
    ``api._WING_LOADING_FAMILIES``, and a shell with its own copy would offer
    the row on a family whose evaluate() cannot read it.
    """
    from aerobo import api

    name = S["wing"]["problem"]
    if span_is_searched(S):
        return True
    return api.with_modifiers(
        name, api.modifiers_of(name) | {"size_ws"}) is not None


def span_is_searched(S: dict) -> bool:
    """Is the span a design variable AGAINST A WING LOADING?

    That is either LOADING modifier — ``size_ws``, where the mission states
    the loading, or ``size_ws_free``, where the search picks it inside a band
    — and this asks the registry for them rather than reading the design
    vector. Two other families put ``b_m`` in their vector without being in
    this state, and both would be misreported by a label test: the
    FREE-PLANFORM family searches the AREA beside the span (so no loading
    sizes the wing), and the CAR families search the span against a FIXED
    reference area (they have no mission at all — their operating point is a
    track speed). Both of those b_m rows are ordinary design-box rows and
    behave like ones.
    """
    from aerobo import api

    return bool({"size_ws", "size_ws_free"}
                & api.modifiers_of(S["wing"]["problem"]))


#: WHO DECIDES A WATER CRAFT'S SIZE — the four answers, in one control.
#:
#: The air shell asks this in the planform menu (fixed / wing-loading /
#: free), and a water family reached that menu with exactly ONE entry in it:
#: "fixed span + area (you choose the size)", a select with nothing to
#: select. It was not a menu, it was a label — and the thing it ruled out is
#: the question a foil designer starts from, because a foil's span is not
#: set by a hangar or a class rule the way a wing's often is.
#:
#: The two rows are offered SEPARATELY and not as one "free size" switch,
#: because they behave differently and the difference is measured (see
#: RESULTS_SESSION71): the AREA has a genuine two-sided optimum — it is the
#: loading, so a bigger foil trims at a lower CL and buys cavitation margin
#: while paying wetted area — and the SPAN does not. Nothing in this physics
#: pays for span (no structural weight, and the mast drag does not read it),
#: so a searched span runs to the top of whatever it is allowed and stops on
#: the aspect-ratio limit the lifting line is honest to. Both are worth
#: offering; pretending they are the same question is not.
WET_SIZE_MODES = {
    "fixed": "fixed span + area (you choose the size)",
    "area": "free AREA (the span is yours, the loading is searched)",
    "span": "free SPAN (the area is yours, the aspect ratio is searched)",
    "both": "free span + area (both are design-box rows)",
}

#: which api flag each mode turns on (api.WET_SIZE_KEYS).
_WET_SIZE_FLAGS = {"fixed": (), "area": ("free_area",),
                   "span": ("free_span",),
                   "both": ("free_span", "free_area")}


#: what each mode COSTS, said where it is chosen. Two of the four carry a
#: measured warning rather than a description, because the measurement is
#: the useful part: a searched span on this physics has no interior optimum.
WET_SIZE_HINTS = {
    "fixed": "The published planform: b = 1.2 m over 0.144 m² unless you "
             "type otherwise. Speed and depth are still design variables — "
             "they are rows of the design box, not mission fields.",
    "area": "The area is the LOADING (the foil carries a fixed lift, so "
            "CL = L/qS), and this is the size row with a real two-sided "
            "trade: bigger buys cavitation margin at a lower trim CL and "
            "pays wetted area. Measured over the default band it settles "
            "inside it rather than at an end.",
    "span": "Nothing in this model pays for span — there is no structural "
            "weight and the mast drag does not read it — so the search "
            "will take all of it and stop on the aspect-ratio limit the "
            "lifting line is honest to (40). Read the answer as 'as wide "
            "as you allowed': the number to argue about is the band, i.e. "
            "the beam, the foil case or the class rule that sets it.",
    "both": "Both rows, and they behave differently: the area finds an "
            "interior optimum, the span runs to the top of what you allow. "
            "Corners where b²/S leaves the solvers' honest band (3–40) are "
            "refused per design, with the aspect ratio quoted.",
}


#: what a SPAN row costs on a craft with a vertical surface flown FLAT. Said
#: beside the entry the user just took, in the shell's own words, with
#: ``hydrofoil.SPAN_IS_UNPRICED_AT_ZERO_HEEL`` carrying the same reading to
#: anyone who reaches the solver directly.
#:
#: THIS USED TO BE THE REASON THE TWO SPAN ENTRIES WERE MISSING. They are
#: not missing any more. Every water family ships with the mast on and the
#: craft flown flat, so the menu removed "free span" and "free span + area"
#: from the default configuration of every hydrofoil this shell can build —
#: and the row a foil designer starts from is exactly "how wide should this
#: foil be?". Removing an entry from the default state is a ban wearing a
#: menu's clothes. The measurement behind it is kept and is now said as what
#: it is: a statement about the ANSWER, not a reason to withhold the
#: question.
#: the shell's own half: the lead the user reads first, the ACTION only a
#: shell can offer, and the extra clause the ``both`` mode needs. The
#: MEASURED half is not restated here — it is
#: ``hydrofoil.SPAN_IS_UNPRICED_AT_ZERO_HEEL``, spliced in by
#: :func:`wet_size_why`, so the numbers have ONE author. Two copies of a
#: measurement drift, and these two already had: they disagreed about where
#: the climb stops and by 11 degrees about when heel starts to bind.
WET_SIZE_UNPRICED_SPAN_LEAD = (
    "THE SPAN YOU GET BACK WILL BE THE TOP OF YOUR BAND, not an optimum — "
)

#: ...and what to do about it, which is the part that belongs to a screen.
WET_SIZE_UNPRICED_SPAN_ACT = (
    " TYPE A HEEL ANGLE IN THE FIELD BELOW to make the water price it "
    "instead, remembering it takes a real angle rather than a token one."
)

#: ...and the clause the ``both`` mode needs, because with the AREA searched
#: as well the ceiling is not a fixed number any more: the span may only
#: reach sqrt(40 S), so a wider AREA band buys span, and the answer rides
#: whichever of the two bounds it meets first.
WET_SIZE_UNPRICED_SPAN_BOTH = (
    " You are searching the AREA too, so the aspect-ratio ceiling moves "
    "with it: the span can only reach sqrt(40 x S), which is 2.4 m at the "
    "published 0.144 m2 and 3.39 m at the top of the default area band. "
    "Widening the AREA band is therefore a way of buying span, and the "
    "answer rides whichever of the two bounds it reaches first."
)

def unpriced_span_why(mode: str = "span") -> str:
    """The whole caution, assembled — shell words around ONE measurement.

    The measured half is ``hydrofoil.SPAN_IS_UNPRICED_AT_ZERO_HEEL`` rather
    than a copy of it, because a copy is what these two sentences already
    were and they had already drifted apart on both numbers that matter.
    """
    from aerobo import hydrofoil as _hf

    return (WET_SIZE_UNPRICED_SPAN_LEAD
            + _hf.SPAN_IS_UNPRICED_AT_ZERO_HEEL
            + (WET_SIZE_UNPRICED_SPAN_BOTH if mode == "both" else "")
            + WET_SIZE_UNPRICED_SPAN_ACT)


#: the name this sentence carried while it explained an ABSENCE. Kept as an
#: alias so a caller that has not caught up reads the new sentence rather
#: than a NameError. The ``span``-mode wording, which is what it always was.
WET_SIZE_UNPRICED_SPAN_WHY = unpriced_span_why("span")
WET_SIZE_NO_SPAN_WITH_FIN_WHY = WET_SIZE_UNPRICED_SPAN_WHY

#: ...and what a family that searches its stabiliser DEPTH is told when it
#: searches its span as well: the depth row changes UNITS. It used to be a
#: refusal — the depth band is a fraction of the span, so freeing both was
#: said to let one box row move another's bounds — and the families it
#: refused are the ones this shell opens water on, which made "search the
#: span" unreachable there. The row is a fraction of the candidate's own
#: span now (``hydrotail.Z_T_FRAC_LABEL``), so nothing moves anything, and
#: this sentence says so where the design box would otherwise appear to have
#: lost a row and gained a stranger.
WET_SIZE_FREE_DEPTH_IS_A_FRACTION_WHY = (
    "This family searches its stabiliser's DEPTH as well, so with the span "
    "searched too that row changes units: the design box shows z_t_frac "
    "(0.01–0.30 of the span) instead of z_t_m. A depth in metres is a "
    "statement about one span — 0.36 m under a 0.6 m foil is 0.6 of it and "
    "under a 2.4 m foil 0.15 of it — and as a fraction it means the same "
    "layout for every candidate the search draws."
)

#: ...and why there is no size menu at all. Kept for the families that
#: declare neither row; every water family declares the AREA row, so in
#: practice this is what a non-water family would see.
WET_SIZE_UNAVAILABLE_WHY = (
    "This family's size is a stated planform, not a design-box row: it "
    "declares neither of the two size flags, so there is nothing here to "
    "choose between."
)


def _wet_size_strut_on(S: dict) -> bool:
    """Does this craft have a vertical surface at all?

    The stage-1 answer, read the way ``fin_surface`` reads it — the switch
    plus whether the family declares the presence key — and NOT a second
    copy of the rule: a menu that disagreed with the switch about whether
    there is a mast would offer a row the builder refuses.
    """
    from aerobo import api as _api

    sp = _api.PROBLEM_SPECS.get(S["wing"]["problem"])
    if sp is None or _api.FIN_PRESENCE_KEY not in sp.flags:
        return False
    return bool(S["wing"]["choices"].get("fin", FIN_DEFAULT))


def wet_size_modes(S: dict) -> dict:
    """The size modes THIS session may choose between, in menu order.

    Asked of the REGISTRY and never of a name list: BOTH rows are on every
    water family (``api.wet_size_keys``), so every water session is offered
    all four modes. A control that offered a mode the builder refuses is the
    defect this whole file is arranged against — and the builder refuses
    none of them now.

    NOTHING REMOVES THE SPAN ENTRIES ANY MORE. They used to go whenever the
    craft had a vertical surface and was flown flat, which is the DEFAULT
    state of every water family here, so "free span + area" was missing from
    the first screen of every hydrofoil. The measurement that motivated it —
    at zero heel nothing on the craft reads the span, so the answer rides the
    band (``hydrofoil.span_is_priced``) — is true and is still said, beside
    the entry the user takes rather than in place of it
    (:func:`wet_size_why`, ``WET_SIZE_UNPRICED_SPAN_WHY``). Stating what an
    answer is worth is this shell's job; deciding the user may not have it is
    not.
    """
    from aerobo import api as _api

    sp = _api.PROBLEM_SPECS.get(S["wing"]["problem"])
    if sp is None:
        return {"fixed": WET_SIZE_MODES["fixed"]}
    have = set(sp.flags)
    span = "free_span" in have
    area = "free_area" in have
    return {k: v for k, v in WET_SIZE_MODES.items()
            if k == "fixed"
            or all((span if f == "free_span" else area)
                   for f in _WET_SIZE_FLAGS[k])}


def wet_size_why(S: dict) -> str | None:
    """What this session needs told about its size menu, or None.

    NOTHING IS MISSING FROM THE MENU ANY MORE, so this no longer explains an
    absence. It explains what the entry the user TOOK is worth. Two kinds of
    sentence, in the order they matter:

    * the span is being searched and nothing PRICES it — a vertical surface
      on flat water — so the answer will ride the top of the band rather
      than settle inside it. The measurement that used to remove the entry,
      said beside it instead (``WET_SIZE_UNPRICED_SPAN_WHY``), and it names
      the two actions that change the answer;
    * the span the user just asked for CHANGES ANOTHER ROW'S UNITS on a
      family that searches its stabiliser depth (``hydrotail.Z_T_FRAC_LABEL``).
      Said here because the design box would otherwise appear to have lost a
      row and gained a stranger.

    The first wins where both apply: a row whose answer is a bound is a
    bigger thing to know than a row that changed units.
    """
    from aerobo import api as _api

    sp = _api.PROBLEM_SPECS.get(S["wing"]["problem"])
    if sp is None:
        return WET_SIZE_UNAVAILABLE_WHY
    have = set(sp.flags)
    if not have & set(_api.WET_SIZE_KEYS):
        return WET_SIZE_UNAVAILABLE_WHY
    if "free_span" not in have:
        return WET_SIZE_UNAVAILABLE_WHY
    if "free_span" not in set(_WET_SIZE_FLAGS[wet_size_mode(S)]):
        # the span is not being searched, so neither sentence is about this
        # session. The mode menu says what the other entries are.
        return None
    # ...the span IS being searched. Is anything reading it? The rule is the
    # solver's own (``hydrofoil.heel_prices_span``) rather than a second copy
    # of it here, asked of the stage-1 answers because the problem the user
    # will run does not exist yet.
    #
    # HEEL ALONE, exactly as the solver asks it. This used to AND in the
    # strut, which suppressed the caution on the one configuration where it
    # is most true: with the mast OFF nothing reads the span either, and
    # measured on the shipped foil the ratchet is STEEPER without it (L/D to
    # the top of the band 58.63 against 46.33). A warning withheld where it
    # applies harder is worse than no warning.
    from aerobo import hydrofoil as _hf

    said = []
    if not _hf.heel_prices_span(wet_heel_deg(S)):
        said.append(unpriced_span_why(wet_size_mode(S)))
    # ...and what the SPAN does to the stabiliser-depth row on the families
    # that SEARCH it. Read off the registered design vector — a family that
    # searches its depth is one with a ``z_t_m`` row — and not off a name or
    # a flag: the flag of that name is the STATED height, carried by the
    # families that do not.
    #
    # BOTH, not the first of the two. They are about different rows, and the
    # configuration that used to lose the units sentence to the caution is
    # this shell's OWN DEFAULT — mast on, flown flat, span searched — so the
    # one session guaranteed to see a metre row turn into a fraction was the
    # one guaranteed never to be told why.
    if _api.TAIL_HEIGHT_KEY in sp.param_labels:
        said.append(WET_SIZE_FREE_DEPTH_IS_A_FRACTION_WHY)
    return " ".join(said) or None


def wet_size_available(S: dict) -> bool:
    """Is there a size menu at all — more than one thing to choose?"""
    return len(wet_size_modes(S)) > 1


def wet_size_mode(S: dict) -> str:
    """Which of :data:`WET_SIZE_MODES` the water size is being asked in.

    CLAMPED to what this session may actually choose (:func:`wet_size_modes`),
    so the card, the flags and the run cannot disagree.

    THE CLAMP NO LONGER FIRES. It was written for a menu that removed the
    span entries beside a strut on flat water, so a stored "free span" could
    stop being offered under the user; every water family declares both size
    keys now and no answer is ever withdrawn, which leaves this a guard
    against a FUTURE family that removes one rather than a live behaviour.
    Kept because that is exactly the case a shell must not get wrong, and
    because a mode the builder refuses is the defect this file is arranged
    against.
    """
    got = str(S["wing"]["choices"].get("wet_size", "fixed"))
    offered = wet_size_modes(S)
    if got in offered:
        return got
    return "area" if got == "both" and "area" in offered else "fixed"


def set_wet_size_mode(S: dict, value) -> bool:
    """Choose it. False (and nothing stored) on a mode that is not one."""
    if value not in WET_SIZE_MODES:
        return False
    S["wing"]["choices"]["wet_size"] = str(value)
    return True


def wet_size_flags(S: dict) -> dict:
    """``{flag: True}`` for the size rows this session searches.

    Empty for the fixed mode and for any family that does not declare the
    keys, which is what keeps an untouched water session — and every family
    that never grew the rows — sending exactly the flags it always sent.
    """
    if not wet_size_available(S):
        return {}
    return {k: True for k in _WET_SIZE_FLAGS[wet_size_mode(S)]}


#: HOW FAR OVER THE CRAFT IS FLYING — the one row that gives a searched span
#: something to trade against, so it is asked beside the size and nowhere
#: else. Zero is a foil flown flat and every published water number.
WET_HEEL_KEY = "heel_deg"
WET_HEEL_DEFAULT = 0.0

WET_HEEL_HINT = (
    "Heel is a FLIGHT CONDITION, not a shape: state how far over the craft "
    "is sailed and every span station gets its own static head. It is the "
    "only quantity in these solvers that reads how WIDE the foil is — the "
    "rising tip climbs (b/2)·sin(heel) towards the surface — so with the "
    "span searched this row is what stops the answer at a span instead of "
    "at the top of the band you typed. It is asked and not derived because "
    "what balances a rig's roll couple is the rider, and this package has "
    "no rider; the run reports the lateral CG offset your stated side force "
    "would need (rig.righting_lever_required_m)."
)


def wet_heel_deg(S: dict) -> float:
    """The stated heel [deg] for this session (0.0 = flat, the default)."""
    try:
        return float(S["wing"]["choices"].get(WET_HEEL_KEY,
                                              WET_HEEL_DEFAULT))
    except (TypeError, ValueError):
        return WET_HEEL_DEFAULT


def set_wet_heel_deg(S: dict, value) -> bool:
    """Store it. False (and nothing stored) on a value the solver refuses.

    Validated HERE by the solver's own validator rather than by a copy of
    its band, so the shell and the run refuse the same numbers with the same
    sentence (``hydrofoil.heel_angle``).
    """
    from aerobo import hydrofoil as _hf

    try:
        phi = _hf.heel_angle(value)
    except ValueError:
        return False
    S["wing"]["choices"][WET_HEEL_KEY] = float(phi)
    return True


#: READ EACH SURFACE'S SECTION AT ITS OWN REYNOLDS NUMBER. Asked beside the
#: heel because it is the other answer about the water this craft is actually
#: in, and because both are FLIGHT CONDITIONS rather than shape.
WET_RE_KEY = "flown_reynolds"

WET_RE_HINT = (
    "Off (every published water number): both surfaces are read off the "
    "NACA 24XX tables at Re = 1e6, whatever they are flying. On: each "
    "surface's section is read at its OWN Reynolds number — rho·V·mac/mu, "
    "the number this run has always reported and never used. The speed is "
    "a design variable here and the chord moves with the size rows, so the "
    "gap is not small: over a box with the span and area open it is worth "
    "−6.1 % of L/D at a thin fast foil (Re 6.6e5) and +6.1 % at a fat one "
    "(Re 3.7e6). A design outside the bank (Re 1e5–1e7) is refused rather "
    "than extrapolated."
)


def wet_re_available(S: dict) -> bool:
    """Does this family take the flown-Reynolds switch, and can it use it?

    False beside a CHOSEN section: that is one table at one Reynolds number,
    so there is nothing to re-read — the solver refuses the pair at config
    time (``hydrofoil._check_flown_reynolds``) and a shell that offered the
    switch anyway would be offering a run that cannot be built.
    """
    from aerobo import api as _api

    sp = _api.PROBLEM_SPECS.get(S["wing"]["problem"])
    if sp is None or WET_RE_KEY not in sp.flags:
        return False
    return section_of(S, "main") is None or not section_is_own(S, "main")


def wet_flown_re(S: dict) -> bool:
    """Is the section read at the flown Re? (False = the Re-1e6 tables.)"""
    return bool(S["wing"]["choices"].get(WET_RE_KEY, False))


def set_wet_flown_re(S: dict, value) -> bool:
    """Store it. Always accepted — it is a switch, not a band."""
    S["wing"]["choices"][WET_RE_KEY] = bool(value)
    return True


def wet_re_flags(S: dict) -> dict:
    """``{flown_reynolds: True}`` where it is on and offered, else empty.

    Empty when OFF, so an untouched water session sends exactly what it
    always sent — and empty when the family cannot honour it, so a stored
    answer left behind by a section chosen afterwards cannot reach a builder
    that would refuse it.
    """
    if not wet_re_available(S) or not wet_flown_re(S):
        return {}
    return {WET_RE_KEY: True}


def wet_heel_flags(S: dict) -> dict:
    """``{heel_deg: phi}`` where a heel is stated, else empty.

    Empty at zero on purpose: a flag whose value is the problem's own
    default would put a "rig-less craft heeled 0 deg" line in every water
    report and make every published run send a flag it never sent.
    """
    from aerobo import api as _api

    sp = _api.PROBLEM_SPECS.get(S["wing"]["problem"])
    phi = wet_heel_deg(S)
    if sp is None or WET_HEEL_KEY not in sp.flags or phi == 0.0:
        return {}
    return {WET_HEEL_KEY: float(phi)}


def searched_labels(S: dict) -> tuple:
    """The design vector this session would actually launch, by label.

    Off the BUILT problem and not the registry's static ``param_labels``:
    rows that a flag creates (the water families' size) exist only once the
    flags are assembled, and the static list describes the family without
    them.

    AND IT RETRIES WITHOUT THE OBJECTIVE, for the reason
    ``config.default_bounds`` does (``config._flags_without_the_objective``):
    the composite objective refuses to build until its normalisation band is
    measured, and an empty tuple here reads as "this run searches nothing" on
    every card that quotes it — while the run itself searches the same vector
    it always did. An objective is not a dimension, so the labels the
    stripped build reports are the labels the run will fly.
    """
    from aerobo import api as _api

    from . import config

    spc = _api.PROBLEM_SPECS.get(S["wing"]["problem"])
    if spc is None:
        return ()
    mission = {} if spc.uses_mission else None
    for fl in (config.flags(S),
               config._flags_without_the_objective(config.flags(S))):
        try:
            return tuple(spc.build(mission, fl, None).param_labels)
        except Exception:                   # noqa: BLE001 — a read-out
            continue
    return ()


def loading_is_searched(S: dict) -> bool:
    """Is the WING LOADING itself a design variable (``size_ws_free``)?

    The narrower question :func:`span_is_searched` deliberately does not
    ask: in that mode W/S is a row of the design box like any other, so the
    mission's own W/S field is a starting point rather than the answer, and
    the cards that quote "the loading this wing flies" have to say which of
    the two it is.
    """
    from aerobo import api

    return "size_ws_free" in api.modifiers_of(S["wing"]["problem"])


def span_box(S: dict, row: str = "b_m") -> tuple | None:
    """``(min, max)`` span the run searches for ``row`` [m], or ``None`` when
    it is not searched at all.

    The box the RUN uses — the design-box row, section pins and releases
    already in it (``config.effective_bounds``) — so this cannot drift from
    what the optimiser is handed.
    """
    from . import config

    got = config.effective_bounds(S).get(row)
    if got is None:
        return None
    return float(got[0][0]), float(got[0][1])


def span_band_default(S: dict, row: str = "b_m") -> tuple | None:
    """The band the span row OPENS on when it is switched on, or ``None``.

    The family's own fractional band (sizing.B_FRAC_BOUNDS) around the
    NOMINAL span (:func:`nominal_span`) — the span the fixed planform flies,
    chosen or estimated. That is the one band that cannot refuse the design
    on screen: switching the search on must not, by itself, exclude the wing
    the user was looking at.

    The nominal span, NOT the flown one: with the span already searched,
    :func:`flown_size` reports the middle of the current box, so a band taken
    around that would drift outwards every time it was recomputed (2.3x per
    reset — the fractional band is not centred on its own midpoint).

    ONE BAND, WHICHEVER ROW ASKS. ``sizing.size_bounds`` opens every span row
    of a pair on the same fractional band for a reason worth repeating here:
    the two wings are alternatives for the same job, so opening the rear one
    narrower — around the narrower wing somebody happened to type — would
    answer the question the search is being asked. The band is then WIDENED
    to hold every span that IS typed, because the other rule still stands:
    switching the search on may not, by itself, exclude the pair the user was
    looking at.

    ``row`` is therefore accepted and deliberately unused; it keeps the call
    site honest (a caller asks for the band of a row) and leaves room for a
    family whose surfaces are not alternatives.
    """
    from aerobo import api
    from aerobo.sizing import B_FRAC_BOUNDS

    name = S["wing"]["problem"]
    # ...and the SIZED families, which is where this was silently absent.
    # ``span_is_searched`` deliberately excludes the ``size`` modifier (it
    # asks "is the span searched AGAINST A LOADING"), and ``resizable`` is
    # False for every sized family, so the free planform — the one mode whose
    # span IS a design variable with no loading behind it — fell through both
    # tests and got no band at all. Its b_m row then stayed at the family's
    # published 6-40 m, which is a fraction of the 10 m reference wing
    # ``objective.Problem`` carries, for every mission.
    if not (api.resizable(name) or span_is_searched(S)
            or api.is_sized(api.modifiers_of(name))):
        return None
    b = nominal_span(S)
    if not b > 0.0:
        return None
    lo, hi = B_FRAC_BOUNDS[0] * b, B_FRAC_BOUNDS[1] * b
    for other in span_rows(S):
        typed = chosen_span(S, other)
        if typed is not None:
            lo, hi = min(lo, typed), max(hi, typed)
    return round(lo, 3), round(hi, 3)


#: the design-box row a sized family searches its AREA in
AREA_ROW = "S_m2"

#: the PAIR's split row: the front wing's share of the total area
#: (``tandem.TandemProblem``). Named beside :data:`AREA_ROW` because the two
#: are one answer on a tandem — see the per-wing area question below.
PAIR_SPLIT_ROW = "area_split_front"

#: rows whose band the SHELL derives from the mission rather than the user
#: typing it. Kept as a set of names so :func:`size_band_defaults` and the
#: refresh in :func:`sync_wing_from_mission` cannot disagree about which rows
#: are the shell's to move.
def size_rows(S: dict) -> tuple:
    """The SIZE rows of this family's design box — spans, then the area."""
    from aerobo import api

    name = S["wing"]["problem"]
    rows = list(span_rows(S)) if (span_is_searched(S)
                                  or api.is_sized(api.modifiers_of(name))) \
        else []
    if AREA_ROW in (api.PROBLEM_SPECS[name].param_labels or ()):
        rows.append(AREA_ROW)
    return tuple(rows)


def area_band_default(S: dict) -> tuple | None:
    """The band the AREA row opens on for THIS MISSION, or ``None``.

    The area twin of :func:`span_band_default`, and it exists for the same
    reason that one does: a design box is a question about the wing on
    screen, and the published ``S_m2`` row is a fraction of
    ``objective.Problem.S`` — a 10 m^2 reference the mission cannot reach
    (``api._make_variant_spec`` strips ``b_m``/``S_m2`` from a sized spec's
    flags, deliberately, because a typed size beside a searched one is two
    answers to one question). The consequence was not a cosmetic one: a 25 kg
    aeroplane whose mission states 2.0 m^2 was searched over 8-22 m^2, so the
    SMALLEST wing the optimiser could return was four times the one the
    mission asked for, and a 64-draw probe put the best payload L/D in the
    published box at 14.6 against 36.2 in the mission-sized one.

    ``sizing.S_FRAC_BOUNDS`` is the same fractional band the engine uses
    around its own reference, so this changes WHICH wing the fraction is
    taken about and nothing else. Widened to hold an area the user typed, for
    the reason :func:`span_band_default` gives: opening the search may not,
    by itself, exclude the wing that was on screen.
    """
    from aerobo import api
    from aerobo.sizing import S_FRAC_BOUNDS

    name = S["wing"]["problem"]
    if AREA_ROW not in (api.PROBLEM_SPECS[name].param_labels or ()):
        return None
    if not api.is_sized(api.modifiers_of(name)):
        # the CAR searches an area against no mission at all
        return None
    area = float((S.get("mission") or {}).get("s_ref_m2") or 0.0)
    if not area > 0.0:
        return None
    lo, hi = S_FRAC_BOUNDS[0] * area, S_FRAC_BOUNDS[1] * area
    typed = (S["wing"].get("choices") or {}).get("area_m2")
    if typed:
        lo, hi = min(lo, float(typed)), max(hi, float(typed))
    return round(lo, 4), round(hi, 4)


#: the SECOND SURFACE's own design-box rows, and what each is a fraction OF.
#: These are not sizes of the wing: they are the LAYOUT — how big the
#: stabiliser is, how far behind the wing it sits, and how far above or below
#: it. Every one is PUBLISHED as a length or an area on the family's own
#: reference craft (``tail.S_T_BOUNDS`` / ``L_T_BOUNDS`` are the fractions
#: times a 10 m, 10 m^2 aeroplane; ``hydrotail``'s are the same times a 1.2 m,
#: 0.144 m^2 foil), so on any OTHER mission they are a statement about
#: somebody else's craft.
LAYOUT_ROWS = {"S_t_m2": "area", "l_t_m": "span", "z_t_m": "span"}


#: significant figures a band the shell DERIVES is quoted to
_TIDY_SIG = 6


def _tidy(v: float, toward: int = 0) -> float:
    """Round a derived band end to :data:`_TIDY_SIG` significant figures.

    SIGNIFICANT, not decimal: these rows are derived by scaling, so they run
    from a 22 m^2 wing to a 0.006 m^2 stabiliser in the same expression, and
    ``round(v, 4)`` turns the second one into a band whose ends have two
    digits of resolution — 0.00625 m^2 became 0.0063, which is a 0.8 % move
    in a number nobody typed.

    ``toward`` is +1 to round UP and -1 to round DOWN, and it is not a
    nicety: a clipped end sits exactly ON the gate it was derived from
    (``b_hi = sqrt(AR_max * S_lo)`` gives that corner ``AR_max`` to the last
    bit), so rounding it to nearest lands it outside half the time — and then
    the one corner the clip exists to remove is the one corner refused.
    """
    v = float(v)
    if not (v and math.isfinite(v)):
        return 0.0 if not v else v
    if not toward:
        return float(f"{v:.6g}")
    q = 10.0 ** (math.floor(math.log10(abs(v))) - (_TIDY_SIG - 1))
    n = v / q
    # ...and a value already ON the grid stays there: v/q for an exact 6.0
    # comes back 599999.9999999999, and a ceil of that is a band end one part
    # in 1e15 above the number it is supposed to equal — which is enough to
    # put the mission's own span outside its own row.
    near = round(n)
    if abs(n - near) < 1e-9:
        # ...and through the FORMATTER, not through ``near * q``: 6.0 / 1e-5
        # is 599999.9999999999 and 600000 * 1e-5 is 6.000000000000001, so
        # multiplying back would return a band end one part in 1e15 above the
        # number it is meant to equal — enough to put the mission's own span
        # outside its own row.
        return float(f"{v:.6g}")
    return (math.ceil(n) if toward > 0 else math.floor(n)) * q


def reference_wing(S: dict) -> tuple | None:
    """``(span [m], area [m^2])`` this medium's PUBLISHED rows are quoted on.

    The one number :func:`layout_band_default` needs and the only one the
    published boxes do not carry: ``l_t_m`` is 3 – 8 m because the aeroplane
    those results were measured on has a 10 m span, and nothing in the row
    itself says so.

    Read off the engine's own constants rather than by building the problem
    (``api.planform_size``), for two reasons: building is a solver call in
    the middle of a state write, and it returns ``None`` for exactly the
    families whose size is a design vector — which is most of the ones a
    layout row appears on.
    """
    if S["medium"] == "air":
        from aerobo.tail import B_REF_M, S_REF_M2
        return float(B_REF_M), float(S_REF_M2)
    if S["medium"] == "water":
        from aerobo.hydrofoil import HydrofoilProblem
        prob = HydrofoilProblem()
        return float(prob.b), float(prob.S)
    return None                          # the car has no second surface


def layout_rows(S: dict) -> tuple:
    """The :data:`LAYOUT_ROWS` THIS family actually searches."""
    from aerobo import api

    spec = api.PROBLEM_SPECS[S["wing"]["problem"]]
    labels = set(spec.param_labels or ())
    return tuple(row for row in LAYOUT_ROWS
                 if row in labels and (spec.default_bounds or {}).get(row))


def layout_band_default(S: dict, row: str) -> tuple | None:
    """The band a LAYOUT row opens on for THIS MISSION, or ``None``.

    :func:`span_band_default`'s twin for the second surface, and it exists
    for the third time in this module for the same reason: a design box is a
    question about the craft on screen. A mission stating a 1 m wing was
    searching the stabiliser's arm over **3 – 8 m** and its area over
    **0.5 – 3 m^2** — a tail eight spans behind a wing, four times the
    aircraft's whole reference area — because those are the numbers the 10 m
    reference aeroplane publishes and no code between the mission and the box
    ever divided them by anything.

    The rule is the family's PUBLISHED row scaled by the mission's wing over
    the reference wing (:func:`reference_wing`) — spans for the two
    separations, area for the area. Written that way rather than from the
    fractions directly so that BOTH media are one code path and so that a
    mission which IS the reference craft reproduces the published box
    bit-for-bit; and it can only ever be a DEFAULT, because the engine
    carries any band the user types straight into the problem
    (``tail.arm_row`` / ``area_row`` / ``height_row``).

    The one clamp is ``tail.L_T_MIN_M``: at zero arm the tail volume is zero
    and the pitch solve is singular, so the arm's low end is held above the
    floor the solver would refuse rather than handed to it.
    """
    from aerobo import api
    from aerobo.tail import L_T_MIN_M

    which = LAYOUT_ROWS.get(row)
    if which is None:
        return None
    if flown_size(S) is not None:
        # THE ENGINE ALREADY DID IT. A family that honours a chosen planform
        # builds its second surface's rows off the wing it is actually flying
        # (``tail.area_band`` / ``arm_band``), so the box on screen is already
        # this craft's — writing an override here would SHADOW that derivation
        # with a coarser one taken from the mission's reference area, and a
        # model aeroplane would get the 10 m aeroplane's tail back.
        #
        # The gap this function fills is the other case, and it is the one the
        # user hit: with the size in the DESIGN VECTOR there is no flown wing
        # to derive from, so the engine falls back to the family's published
        # row and the box reads 0.5 – 3 m^2 of stabiliser for a 0.125 m^2
        # aircraft.
        return None
    spec = api.PROBLEM_SPECS[S["wing"]["problem"]]
    published = (spec.default_bounds or {}).get(row)
    ref = reference_wing(S)
    if not published or ref is None:
        return None
    if which == "area":
        have = float((S.get("mission") or {}).get("s_ref_m2") or 0.0)
        want = ref[1]
    else:
        have, want = nominal_span(S), ref[0]
    if not (have > 0.0 and want > 0.0):
        return None
    k = have / want
    lo, hi = float(published[0]) * k, float(published[1]) * k
    if row == "l_t_m" and lo < L_T_MIN_M:
        # the solve's own floor, not a taste: below it there is no moment arm
        lo = L_T_MIN_M
        hi = max(hi, lo * 2.0)
    return _tidy(lo), _tidy(hi)


# --------------------------------------------------- the PAIR's two areas
#
# A tandem's design box states the pair as a TOTAL area (``S_m2``) and a
# SPLIT (``area_split_front``, the front wing's share of it). That is what the
# solver searches, and it is not what anybody designs: a pair is two wings,
# and the number a builder, a rule or a wind tunnel states is each wing's own
# area. Asked as a total and a fraction, "the front wing is 6 m² and the rear
# 4 m²" is a pair of simultaneous equations the user has to solve on paper —
# and the answer to one of them (the split) moves when the other (the total)
# does, which is exactly the shape of question this shell exists not to ask.
#
# So the pair's area may be stated the other way round, per wing, EXACTLY or
# as a band, and the two rows the solver searches are derived:
#
#     S_total          = S_front + S_rear
#     area_split_front = S_front / (S_front + S_rear)
#
# Both directions are exact for a stated pair; for a pair of BANDS the box is
# the smallest one containing every (S_front, S_rear) the two bands allow —
# the split's ends are the corners S_f_lo/(S_f_lo + S_r_hi) and
# S_f_hi/(S_f_hi + S_r_lo) — and the corners it adds are refused per
# candidate, exactly as a broken chord limit is. The card says so.

#: how the pair's area is stated: the published total + split, or per wing.
PAIR_AREA_MODES = ("total", "wings")

#: where the per-wing answer lives, under ``S["wing"]["choices"]``. One key,
#: holding one dict, because half of this answer is not an answer: a front
#: area with no rear area states neither of the two rows it derives.
PAIR_AREA_KEY = "tandem_areas"


def pair_area_available(S: dict) -> bool:
    """Does THIS family have a pair of areas to state?

    Read off the design vector — a SPLIT row — and not off the system menu,
    for the reason every other reader here gives: the same pair appears under
    many names, and a family that stopped searching its split would still
    have been offered the card.

    The TOTAL is not required to be a searched row. On most tandem families
    it is not: the mission states the pair's area and the box searches only
    how it is divided. Both cases are answerable per wing — what changes is
    where the total lands (:func:`pair_area_rows`), and a card that refused
    the common one would be a card nobody saw.
    """
    from aerobo import api

    labels = set(api.PROBLEM_SPECS[S["wing"]["problem"]].param_labels or ())
    if PAIR_SPLIT_ROW not in labels:
        return False
    return AREA_ROW in labels or reference_area(S) is not None


def pair_area_state(S: dict) -> dict:
    """The per-wing answer, defaulted. Never None, never partial."""
    got = dict((S["wing"].get("choices") or {}).get(PAIR_AREA_KEY) or {})
    out = {"mode": str(got.get("mode") or "total")}
    for wing in ("front", "rear"):
        side = dict(got.get(wing) or {})
        out[wing] = {"exact": bool(side.get("exact")),
                     "value": side.get("value"),
                     "band": list(side.get("band") or [])}
    if out["mode"] not in PAIR_AREA_MODES:
        out["mode"] = "total"
    return out


def pair_area_mode(S: dict) -> str:
    """``"total"`` (the published question) or ``"wings"``."""
    return pair_area_state(S)["mode"] if pair_area_available(S) else "total"


def _pair_area_band(side: dict) -> tuple | None:
    """One wing's area as ``(lo, hi)`` — a stated value is a degenerate band.

    None where the wing has not been answered, which is what makes "half an
    answer is not an answer" a property of the reader rather than a rule each
    caller has to remember.
    """
    if side.get("exact"):
        try:
            v = float(side.get("value"))
        except (TypeError, ValueError):
            return None
        return (v, v) if v > 0.0 else None
    band = side.get("band") or []
    if len(band) != 2:
        return None
    try:
        lo, hi = float(band[0]), float(band[1])
    except (TypeError, ValueError):
        return None
    return (lo, hi) if hi > lo > 0.0 else None


def pair_areas(S: dict) -> dict | None:
    """``{"front": (lo, hi), "rear": (lo, hi)}`` as stated, or None.

    Both wings or neither: one area alone states neither the total nor the
    split, and half an answer written into one of those rows would be a
    number nobody typed sitting in the box.
    """
    if not pair_area_available(S) or pair_area_mode(S) != "wings":
        return None
    st = pair_area_state(S)
    front, rear = _pair_area_band(st["front"]), _pair_area_band(st["rear"])
    if front is None or rear is None:
        return None
    return {"front": front, "rear": rear}


def pair_area_rows(S: dict) -> dict:
    """The two rows the pair's stated areas imply.

    ``{row: ("pin", value)}`` where both wings are stated EXACTLY — the total
    and the split are then both single numbers, and a row the optimiser could
    still move is not a stated area — and ``{row: ("band", (lo, hi))}``
    otherwise. Empty where the pair is stated the published way.

    The split's band is the corner arithmetic, not the ratio of the two
    bands' ends: f = S_f/(S_f + S_r) falls with S_r, so its lowest value is
    the smallest front wing against the LARGEST rear one and its highest is
    the largest front against the smallest rear.
    """
    from aerobo import api

    got = pair_areas(S)
    if got is None:
        return {}
    (f_lo, f_hi), (r_lo, r_hi) = got["front"], got["rear"]
    s_lo, s_hi = f_lo + r_lo, f_hi + r_hi
    k_lo = f_lo / (f_lo + r_hi)
    k_hi = f_hi / (f_hi + r_lo)
    exact = (f_hi == f_lo and r_hi == r_lo)
    out = {PAIR_SPLIT_ROW: ("pin", k_lo) if exact
           else ("band", (k_lo, k_hi))}
    # the TOTAL only where the family SEARCHES it. Where it does not, the
    # pair's area is the mission's and this card cannot quietly restate it —
    # it reports the total the two answers imply and offers to adopt it
    # (:func:`pair_area_total`), which keeps one question in one place.
    labels = set(api.PROBLEM_SPECS[S["wing"]["problem"]].param_labels or ())
    if AREA_ROW in labels:
        out[AREA_ROW] = ("pin", s_lo) if exact else ("band", (s_lo, s_hi))
    return out


def pair_area_total(S: dict) -> tuple | None:
    """``(lo, hi)`` — the pair's total area the two stated wings imply.

    None where the pair has not been stated per wing. Reported beside the
    mission's own area wherever the family does not search the total, so
    "the two wings I typed do not add up to the aeroplane the mission
    describes" is said on the card rather than discovered in a refusal.
    """
    got = pair_areas(S)
    if got is None:
        return None
    return (got["front"][0] + got["rear"][0], got["front"][1] + got["rear"][1])


def set_pair_area_mode(S: dict, mode: str) -> None:
    """Ask the pair's area the published way, or per wing.

    Switching TO the per-wing question seeds both wings from the box on
    screen, so the card opens on the pair the session already describes and
    turning it on cannot, by itself, move the search.
    """
    if not pair_area_available(S):
        return
    mode = str(mode)
    if mode not in PAIR_AREA_MODES:
        return
    st = pair_area_state(S)
    st["mode"] = mode
    if mode == "wings" and pair_areas(S) is None:
        seed = _pair_area_seed(S)
        if seed is not None:
            for wing in ("front", "rear"):
                st[wing] = {"exact": False, "value": None,
                            "band": [seed[wing][0], seed[wing][1]]}
    S["wing"].setdefault("choices", {})[PAIR_AREA_KEY] = st


def _pair_area_seed(S: dict) -> dict | None:
    """Each wing's area band, read off the box the session already has.

    The product of the total row and the split row, corner by corner — the
    same arithmetic :func:`pair_area_rows` inverts, so switching the question
    over and straight back is a no-op on the search.
    """
    from . import config

    eff = config.effective_bounds(S)
    split = eff.get(PAIR_SPLIT_ROW)
    if split is None:
        return None
    total = eff.get(AREA_ROW)
    if total is not None:
        (s_lo, s_hi) = total[0]
    else:
        # the mission's own area, which is what this family flies when it
        # does not search one — the pair opens on the aeroplane on screen
        area = reference_area(S)
        if area is None:
            return None
        s_lo = s_hi = float(area)
    (k_lo, k_hi) = split[0]
    return {"front": (float(s_lo) * float(k_lo), float(s_hi) * float(k_hi)),
             "rear": (float(s_lo) * (1.0 - float(k_hi)),
                      float(s_hi) * (1.0 - float(k_lo)))}


def set_pair_area(S: dict, wing: str, *, exact: bool | None = None,
                  value=None, band=None) -> bool:
    """Answer ONE wing's area. False where the number is not an area.

    Refused rather than clamped: an area is the user's answer to a real
    question, and the published split band is a calibration and not a ban
    (the card says when a stated pair leaves it). What IS refused is a
    number that is not a positive area, or a band whose ends cross — those
    are not answers, and storing one leaves the box unbuildable.
    """
    if not pair_area_available(S) or str(wing) not in ("front", "rear"):
        return False
    st = pair_area_state(S)
    side = dict(st[str(wing)])
    if exact is not None:
        side["exact"] = bool(exact)
        if bool(exact) and side.get("value") in (None, "") and side.get("band"):
            side["value"] = 0.5 * (float(side["band"][0])
                                   + float(side["band"][1]))
    if value is not None:
        try:
            v = float(value)
        except (TypeError, ValueError):
            return False
        if not v > 0.0:
            return False
        side["value"] = v
    if band is not None:
        try:
            lo, hi = float(band[0]), float(band[1])
        except (TypeError, ValueError, IndexError):
            return False
        if not (hi > lo > 0.0):
            return False
        side["band"] = [lo, hi]
    st[str(wing)] = side
    S["wing"].setdefault("choices", {})[PAIR_AREA_KEY] = st
    return True


def union_pair_spans(S: dict, bands: dict, limits: dict | None = None
                     ) -> dict:
    """One span band for BOTH wings of a pair: the widest of the two.

    A tandem is TWO wings and the shell asks their spans as two rows, but a
    measured recommendation bounds each row on the draws that flew — and the
    draws that flew put the front wing's span in one part of its row and the
    rear wing's in another. Reported per row, that reads as an instruction
    ("the front wing must be 5.2-14.3 m, the rear 4.0-13.8 m") about a pair
    whose two wings are the same question asked twice, and it cuts off the
    span the mission itself states more often than either row alone does.

    THE RULE THE USER ASKED FOR: *take the largest and the smallest* — a
    recommendation should contain everything reasonable. So the two bands
    become their union, and both rows carry it.

    ``limits`` (``{row: (lo, hi)}``) is what each row may not leave — the
    published or effective bound — because a union may not widen a row past
    the box it is a recommendation FOR. Rows absent from ``limits`` are left
    unclipped; rows absent from ``bands`` are left alone entirely, which is
    what makes this safe to call on a single-wing family (it returns its
    argument).
    """
    rows = [r for r in span_rows(S) if r in (bands or {})]
    if len(rows) < 2:
        return bands
    lo = min(float(bands[r][0]) for r in rows)
    hi = max(float(bands[r][1]) for r in rows)
    out = dict(bands)
    for row in rows:
        band = (lo, hi)
        cap = (limits or {}).get(row)
        if cap is not None:
            band = (max(lo, float(cap[0])), min(hi, float(cap[1])))
            if not (band[1] > band[0]):      # a clip that empties the row
                band = (float(bands[row][0]), float(bands[row][1]))
        out[row] = band
    return out


def hold_the_stated_design(S: dict, bands: dict) -> dict:
    """Widen the SIZE rows of ``bands`` so the wing on screen is inside them.

    THE RULE, and the only one that outranks every derivation in this file:
    a band the SHELL wrote may not exclude the design the user is looking at.
    Two things wrote such a band and both could break it —

    * :func:`clip_size_box`, whose corner clip can push the span row past the
      mission's own span (at AR 5 the stated span sits below
      ``sqrt(3 * 2.2 * S)``);
    * the MEASURED box (``aerobo.recommend``), which is the min/max of the
      draws that flew and owes the stated wing nothing at all. Measured on
      the shipped air mission: the mission states 10 m and the measurement
      came back **11.509 – 13.925 m** — a design box that refuses the
      aeroplane the mission on screen describes, with the row still reading
      "recommended" as though it were advice.

    So both go through here. It is not the same as widening a recommendation
    into a wider search: the numbers held are the mission's own and a typed
    one, every one of them already inside the family's published row, and
    "the measurement prefers a longer wing than you stated" is a FINDING to
    report — never a reason to drop the stated wing out of its own box.

    The AREA is held only where a number was TYPED, matching
    :func:`clip_size_box`: the aspect-ratio band is a validity range for
    these solvers, while the loading ceiling is a requirement the mission
    itself stated, and a mission whose own wing lands too fast should see
    that said rather than have the box quietly re-opened under it.

    Rows this does not recognise are returned untouched.
    """
    out = dict(bands)
    for row in span_rows(S):
        band = out.get(row)
        if band is None:
            continue
        lo, hi = float(band[0]), float(band[1])
        # ...per ROW, which is the typed span where one was typed and stage
        # 2's estimate otherwise (:func:`nominal_span`), so a tandem's rear
        # wing is held against its own number and not the front wing's
        value = nominal_span(S, row)
        if value and value > 0.0:
            lo, hi = min(lo, value), max(hi, value)
        out[row] = (lo, hi)
    typed_area = (S["wing"].get("choices") or {}).get("area_m2")
    if typed_area and AREA_ROW in out:
        lo, hi = float(out[AREA_ROW][0]), float(out[AREA_ROW][1])
        out[AREA_ROW] = (min(lo, float(typed_area)),
                         max(hi, float(typed_area)))
    return out


def ar_band(S: dict) -> tuple:
    """The aspect-ratio band this configuration is judged against.

    ``sizing.AR_LIMITS`` — where these solvers describe a wing at all —
    narrowed by the limit the user set on the wing-limits card
    (:data:`api.AR_LIMIT_KEYS`, read through ``api.ar_limits_of`` so the
    shell and the problem builder cannot disagree about what the flags say).
    A limit can only ever narrow: answering one more question must not widen
    a search, and outside the solvers' band there is nothing to widen into.

    Malformed flags return the solvers' band. This is read while a number is
    being TYPED — "1" on the way to "12" — and a card that threw there would
    replace the design box with a traceback; the typed value is validated
    where it is stored (``_set_ar_limit``).
    """
    from aerobo import api
    from aerobo.sizing import AR_LIMITS

    lo, hi = float(AR_LIMITS[0]), float(AR_LIMITS[1])
    try:
        want = api.ar_limits_of((S.get("wing") or {}).get("flags") or {})
    except (ValueError, TypeError):
        return (lo, hi)
    if want is None:
        return (lo, hi)
    if want[0] is not None:
        lo = max(lo, float(want[0]))
    if want[1] is not None:
        hi = min(hi, float(want[1]))
    return (lo, hi) if hi > lo else (float(AR_LIMITS[0]), float(AR_LIMITS[1]))


def clip_size_box(S: dict, bands: dict) -> dict:
    """Clip the SPAN and AREA rows to the corners this mission can fly.

    The two rows are derived independently — a fraction of the nominal span,
    a fraction of the mission's area — and a box is not the pair of rows, it
    is their PRODUCT. Nothing consulted the two gates the corners then walk
    straight into, and the result was a design box that mostly could not be
    flown:

    * ``sizing.check_ar`` refuses ``b^2/S`` outside :data:`sizing.AR_LIMITS`
      (3 – 40), and a 0.6 – 4 m span crossed with a 0.1 – 0.275 m^2 area
      spans AR **1.3 – 160**;
    * ``sizing.check_wing_loading`` refuses ``W/S`` above the mission's own
      ceiling, and ``S_FRAC_BOUNDS`` opens the area at **0.8x** the mission's
      — i.e. 1.25x its loading — so the bottom of the area row is refused by
      arithmetic, on every mission that states a stall speed.

    Measured on a 2 kg, 1 m-span mission: **128 of 128** draws refused, 55 on
    the aspect ratio and 51 on the loading. Both are IN-CONTRACT refusals,
    so nothing raised and nothing was wrong — the box was simply mostly
    outside the mission.

    The clip is the conservative one: every CORNER of the box must pass, so
    the span ends at ``sqrt(AR_max * S_lo)`` and starts at
    ``sqrt(AR_min * S_hi)``. That is deliberately stricter than "the box
    contains flyable designs", because a box whose corners are refused is a
    search spending its draws on penalties.

    Rules that outrank the clip, in order:

    * A NUMBER THE USER TYPED IS ALWAYS INSIDE. The widening the two band
      functions do for a stated span or area is re-applied afterwards, so
      opening the search still cannot exclude the wing that was on screen.
    * A BOX IS NEVER INVERTED. If the mission's own wing is outside these
      gates the clip cannot produce anything, and the unclipped band is
      returned rather than an empty one: the shell's job there is to say
      what refused the design (``api.box_refusal_probe``), not to hide it.

    A PAIR IS CLIPPED ON ITS OWN CONVENTION. A tandem's area row is the
    pair's TOTAL, so ``b^2/S_total`` is not either wing's aspect ratio — and
    for a long time that was read as "leave a pair alone", which left the
    span rows uncrossed with the area row and the box mostly outside the
    gate. Measured on a reported session: spans 3 – 40 m against an area row
    of 16 – 44 m^2 spans **AR 0.41 – 200** per wing, and 36 of 48 draws were
    refused on the aspect ratio before their solver — so the box could not be
    measured, no band could be recommended for the free span, and the search
    spent its budget on refusals.

    What the pair needed was not "no clip" but the gate's OWN arithmetic:
    ``tandem.py`` and ``tandemvlm.py`` check each wing at ``0.5 * S_total``
    (a nominal half, not the candidate's split — see
    :func:`wing_area_share`), so each span row is clipped against half the
    area row. On that session the clip gives 8.12 – 17.89 m, which contains
    the design the run actually found (12.4 m front, 8.1 m rear).
    """
    from aerobo.sizing import AR_LIMITS

    spans = [r for r in bands if r in span_rows(S)]
    if not spans:
        return bands
    if AREA_ROW not in bands:
        # THE AREA IS NOT ALWAYS A ROW — and where it is not, it is still a
        # NUMBER OR A RANGE this shell knows, so the span row can still be
        # crossed with it. Two modes and they are different:
        #
        #   free span (W/S)        S = W / (W/S) with the loading STATED, so
        #                          the area is one number;
        #   free span + free W/S   the loading is a design row, so the area
        #                          SWEEPS W/ws over that row — 8.68 – 17.36 m^2
        #                          on the shipped air mission, a factor of two.
        #                          Crossing the span with the stated area
        #                          instead left a box whose corners the gate
        #                          refuses at both ends (b = 6 m at the big
        #                          area is AR 2.1, b = 20 m at the small one
        #                          is AR 46, against a 3 – 40 band).
        lo_area = hi_area = None
        #: WHY THE CEILING IS DROPPED WHERE THE LOADING IS SEARCHED. The
        #: area a candidate gets is ``W_total / (W/S)`` and ``W_total`` is the
        #: fixed weight PLUS the wing's own structure, which ``sizing``
        #: closes as a fixed point — so ``W_N / ws`` is a LOWER bound on the
        #: area and nothing here bounds it above. Measured on the tandem at
        #: this mode: draws at b = 12.05 m came back at AR 2.39, i.e. an area
        #: of ~121 m^2 against the 25.9 m^2 this arithmetic predicts. A
        #: FLOOR derived from an under-estimated area is still safe (it can
        #: only be too low, so it excludes nothing the gate accepts); a
        #: CEILING derived from one is not — it would cut long spans the gate
        #: would have taken.
        area_is_lower_bound = False
        ws = bands.get(WS_ROW) or ws_band(S)
        if loading_is_searched(S) and ws:
            try:
                weight = float(S["mission"]["W_N"])
            except (KeyError, TypeError, ValueError):
                weight = 0.0
            ws_lo, ws_hi = float(ws[0]), float(ws[1])
            if weight > 0.0 and ws_lo > 0.0 and ws_hi > 0.0:
                lo_area, hi_area = weight / ws_hi, weight / ws_lo
                area_is_lower_bound = True
        if lo_area is None:
            try:
                area = float(S["mission"].get("s_ref_m2") or 0.0)
            except (KeyError, TypeError, ValueError):
                area = 0.0
            if not area > 0.0:
                got = flown_size(S)
                area = float(got[1]) if got and float(got[1]) > 0.0 else 0.0
            if not area > 0.0:
                return bands
            lo_area = hi_area = area
        bands = {**bands, AREA_ROW: (lo_area, hi_area)}
        stated_area = True
    else:
        stated_area = False
        area_is_lower_bound = False
    #: the area each span row's own wing carries, as the GATE reads it: the
    #: whole reference area on a single surface, a nominal half on a pair.
    #: Never the candidate's own split — the refusal this clip exists to
    #: avoid is written against the nominal half, and a clip stricter than
    #: the gate would refuse designs the run accepts.
    share = 0.5 if len(spans) > 1 else 1.0
    s_lo, s_hi = bands[AREA_ROW]

    # 1. the LOADING, on the area row: a wing smaller than this lands too
    #    fast whatever else the design does. A LOWER BOUND, not the exact
    #    area: the loading the solver checks is W_TOTAL/S, and W_total is the
    #    fixed weight PLUS the structural weight of the wing, which grows
    #    with span (``sizing`` closes that as a fixed point). So this removes
    #    only area that is refused for certain — a long-span draw can still
    #    be refused inside the clipped row, and the refusal probe is what
    #    says so.
    cap = mission_ws_ceiling(S)
    if cap:
        try:
            need = float(S["mission"]["W_N"]) / float(cap)
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            need = 0.0
        if need > 0.0:
            s_lo = max(s_lo, need)

    # 2. the ASPECT RATIO, on the span row, over the area row as just clipped
    #    — over the band the RUN will judge candidates against, which is the
    #    solvers' own (:data:`sizing.AR_LIMITS`) narrowed by whatever limit
    #    the user set (``api.AR_LIMIT_KEYS``). Reading only the solvers' band
    #    here would have drawn a box whose corners the user's own limit
    #    refuses, and the refusal would arrive per candidate instead of in
    #    the row that could have shown it.
    ar_lo, ar_hi = ar_band(S)
    want = {}
    for row_b in spans:
        b_lo, b_hi = bands[row_b]
        if s_lo > 0.0 and s_hi > 0.0:
            b_lo = max(b_lo, (ar_lo * share * s_hi) ** 0.5)
            if not area_is_lower_bound:
                b_hi = min(b_hi, (ar_hi * share * s_lo) ** 0.5)
        want[row_b] = (b_lo, b_hi)

    # 3. THE WING ON SCREEN IS ALWAYS INSIDE. Opening a search may not, by
    #    itself, exclude the design the user is looking at — the rule
    #    :func:`span_band_default` is written around — and the corner clip
    #    above can do exactly that: at AR 5 the mission's own span sits below
    #    sqrt(3 * 2.2 * S), so a box that clipped and stopped would put the
    #    stated wing off the bottom of its own span row.
    #
    #    The AREA is not widened back to the mission's, and that asymmetry is
    #    deliberate: the aspect-ratio band is a VALIDITY range for these
    #    solvers, so a box outside it is arithmetic, while the loading
    #    ceiling is a REQUIREMENT the mission itself stated. A mission whose
    #    own wing lands too fast should see that said (the refusal probe
    #    names the gate), not have the box quietly re-opened under it.
    held = hold_the_stated_design(S, {**want, AREA_ROW: (s_lo, s_hi)})
    s_lo, s_hi = held[AREA_ROW]
    # ZERO WIDTH IS NOT INVERTED. Where the area is STATED rather than
    # searched the row arrives as (S, S) — the wing-loading mode, which is
    # the one the "no recommendation for the free span" report was about —
    # and demanding a positive width there threw the whole clip away, so the
    # span row was crossed with nothing and kept a band the gate refuses.
    if not (s_hi >= s_lo):
        return bands                     # this mission is outside its own gates
    out = dict(bands)
    for row_b in spans:
        b_lo, b_hi = held[row_b]
        if not (b_hi > b_lo):
            # THIS ROW ONLY. A pair has two span rows and they are clipped
            # separately (a rear wing may be stated where the front is not),
            # so one row the gates leave nothing of must not throw away the
            # other's clip — nor its own published band.
            continue
        # rounded INWARDS, so that a corner sitting exactly on a gate cannot
        # be rounded back out through it (:func:`_tidy`)
        out[row_b] = (_tidy(b_lo, +1), _tidy(b_hi, -1))
    if out == dict(bands):
        return bands
    if stated_area:
        # borrowed, never written: this box has no area row, and inventing
        # one would hand the run a variable the family does not search
        out.pop(AREA_ROW, None)
        return out
    out[AREA_ROW] = (_tidy(s_lo, +1), _tidy(s_hi, -1))
    return out


#: THE OPERATING POINT, where a family carries it as a design-box ROW
#: instead of as a mission field: box row -> the mission field that states it.
#:
#: This is the other half of a contract that only ever had one half. Stage 1
#: asks for a load, a speed and an altitude-or-depth; ``sync_wing_from_mission``
#: pushes each of them at ``spec.mission_fields``, and a field the family does
#: not carry there is DROPPED. For the water families that is every one of
#: them — ``mission_fields = ()``, because their speed and depth are design
#: VARIABLES — so a stated speed reached nothing at all: type 5 m/s for a
#: wingfoil take-off and the run still searches the family's published
#: 8-16 m/s. The card said so out loud ("The wing solver does not take a
#: speed here ... set its box bounds instead"), which is an honest warning
#: about a question the shell was asking twice and answering once.
#:
#: The SIZE has worked this way for a while (:func:`write_size_bands` — the
#: mission's area and span write the ``S_m2``/``b_m`` rows), so this is that
#: rule applied to the rows next to them rather than a new mechanism.
OPERATING_ROWS = {"V_ms": "V", "depth_m": "depth_m", "altitude_m": "altitude_m"}


def operating_rows(S: dict) -> tuple:
    """The operating-point rows THIS family states in its design box.

    Only the ones the family does NOT also take as a mission field: where it
    does, the mission already reaches the solver directly and writing a band
    as well would be two answers to one question.
    """
    from aerobo import api

    spec = api.PROBLEM_SPECS[S["wing"]["problem"]]
    labels = set(spec.param_labels or ())
    return tuple(row for row, fld in OPERATING_ROWS.items()
                 if row in labels and fld not in (spec.mission_fields or ()))


def operating_band_default(S: dict, row: str) -> tuple | None:
    """The band an operating-point row opens on FOR THIS MISSION, or None.

    The family's own published band, moved to sit on the stated value and
    keeping its RATIO — a published (8, 16) m/s is a factor of two, so a
    stated 5 m/s opens on (3.54, 7.07). Three things that buys:

    * the stated value is always inside the band, so stating an operating
      point cannot by itself empty the search — the rule
      :func:`span_band_default` already follows for the span;
    * the band's WIDTH is still the family's own calibration rather than a
      new number invented here. A factor of two in speed is what the water
      families were calibrated over, and it stays a factor of two;
    * it is geometric and not additive, because these are quantities whose
      sensible neighbourhood scales with them: +/- 4 m/s is most of a
      wingfoil take-off and nothing to a kitefoil.

    ``None`` where the family states no band for the row, where the mission
    has nothing to say about it, or — the load-bearing one — WHERE THE
    MISSION IS STILL THE FAMILY'S OWN. An untouched session must reproduce
    the published run bit-for-bit, and the mission opens on the family's own
    design point, which is NOT the centre of the family's own band: the
    water default is 12 m/s against a published (8, 16) whose geometric
    centre is 11.31, and 0.575 m against a (0.15, 1.0) whose centre is
    0.387. Centring on it regardless moved every untouched water box —
    8.49-16.97 m/s and 0.223-1.485 m — which is the published band replaced
    by an arithmetic accident. Nobody stated those numbers, so nothing is
    written and the family's own band stands. This is the same rule
    ``sync_wing_from_mission`` uses one field further out for the mission
    EDITS themselves.
    """
    from aerobo import api

    fld = OPERATING_ROWS.get(row)
    if fld is None:
        return None
    name = S["wing"]["problem"]
    spec = api.PROBLEM_SPECS[name]
    band = (spec.default_bounds or {}).get(row)
    if not band:
        return None
    lo, hi = float(band[0]), float(band[1])
    try:
        want = float(S["mission"][fld])
    except (KeyError, TypeError, ValueError):
        return None
    if not (want > 0.0 and lo > 0.0 and hi > lo):
        return None
    # THE FAMILY'S OWN DESIGN POINT is the "untouched" marker, and it is the
    # same source the mission OPENS on (:func:`_default_mission`), so the two
    # cannot disagree about what untouched means. ``api.default_mission_values``
    # is the wrong question here: it answers for MISSION FIELDS, and the
    # families this function is for have none — it returns {} for every water
    # problem, which made every untouched water box move.
    own = api.family_design_point(name).get(fld)
    if own is not None and float(want) == float(own):
        return None
    ratio = (hi / lo) ** 0.5
    return (want / ratio, want * ratio)


def size_band_defaults(S: dict) -> dict:
    """``{row: (lo, hi)}`` — the band each row the SHELL owns opens on.

    The wing's own size (:func:`size_rows`) and the second surface's layout
    (:func:`layout_rows`) together, in one place, so the mode switch that
    writes them and the mission edit that refreshes them cannot derive
    different numbers — and then :func:`clip_size_box`, because the box is
    the PRODUCT of these rows and not the list of them.

    ...and the SEARCHED WING LOADING, on the one mode that searches it. It
    belongs here for the same reason the spans do — it is derived from the
    mission (:func:`ws_band_default` centres it on the mission's own ceiling)
    and nobody typed it — and it was the one size row this function could
    not produce, so ``write_size_bands`` could neither stamp it nor bring it
    back. Two measured consequences of that omission: ``apply_choices``
    wiped ``W["bounds"]`` on a problem change and its compensating
    ``write_size_bands`` left ``ws_pa`` on the family's published band, so
    the searched TOP fell 167.58 -> 130.56 Pa while the box still painted it
    "default"; and the row was "user" from birth, so the mission refresh
    (``only_shell_owned=True``) skipped it forever and tightening the stall
    speed afterwards killed the run in the worker on a row nobody had typed.

    NOT part of :func:`clip_size_box`: that clip is the span/area PRODUCT
    against the aspect-ratio gate, and the loading is neither of them.
    """
    out = {}
    for row in size_rows(S):
        band = (area_band_default(S) if row == AREA_ROW
                else span_band_default(S, row))
        if band is not None:
            out[row] = (float(band[0]), float(band[1]))
    out = clip_size_box(S, out)
    if loading_is_searched(S):
        band = ws_band_default(S)
        if band is not None:
            out[WS_ROW] = (float(band[0]), float(band[1]))
    for row in layout_rows(S):
        band = layout_band_default(S, row)
        if band is not None:
            out[row] = (float(band[0]), float(band[1]))
    # ...and the OPERATING POINT, on the families that carry it as a row.
    # Last, and not inside ``clip_size_box``: that clip is the span/area
    # PRODUCT against the aspect-ratio gate, and a speed is neither.
    for row in operating_rows(S):
        band = operating_band_default(S, row)
        if band is not None:
            out[row] = (float(band[0]), float(band[1]))
    return out


#: ``bounds_source`` values the SHELL owns — a band nobody typed, which the
#: mission is therefore still allowed to move. ``"shell"`` is the mission's own
#: size band (:func:`write_size_bands`); ``"recommended"`` is a band a
#: MEASUREMENT wrote (the recommended box, ``aerobo.recommend``).
#:
#: Both are the shell's answer and not the user's, and the distinction is
#: load-bearing: taking a recommendation used to POP the row's source, which
#: made it a user row, and a user row is never refreshed. So a session that
#: took the recommendation for a 10 m wing and then stated a 1 m one in the
#: mission kept searching 8.95 – 14.79 m — the mission said 1 m and the box
#: said the old wing, for the rest of the session, with nothing on screen
#: saying why.
SHELL_BOUNDS_SOURCES = ("shell", "recommended")

#: the SECOND SURFACE'S SPAN is not a design-vector row — the solver searches
#: that surface as an area and, where its planform is designed, an aspect
#: ratio, and the span is this FLAG PAIR narrowed through ``tail.TailLimits``.
#: It is still one of the four numbers the measured box is for (wing span,
#: wing area, tail span, tail area), so it is tracked in ``bounds_source``
#: under :data:`TAIL_SPAN_KEY` like any other row, and taken back with them.
TAIL_SPAN_FLAGS = ("tail_span_min_m", "tail_span_max_m")

#: the ``bounds_source`` key that owns :data:`TAIL_SPAN_FLAGS`. Not a label in
#: any box, deliberately: it must never collide with a design-vector row.
TAIL_SPAN_KEY = "tail span"


def bounds_source(S: dict) -> dict:
    """``{row: "shell"}`` for every band the SHELL derived, not the user.

    The provenance a value comparison cannot supply. ``set_ws_cap_source``
    infers "still the default" by comparing the stored row against the band
    the shell would write, which is exact only until a user types a number
    that happens to equal it — and, worse, cannot work at all for a row
    written BEFORE the mission moved, because the band the shell "would
    write" is already the new one by the time anybody asks.
    """
    return S["wing"].setdefault("bounds_source", {})


def write_size_bands(S: dict, *, only_shell_owned: bool = False,
                     rows=None) -> list:
    """Write the mission's size bands into the design box. Returns the rows.

    ``only_shell_owned`` is the REFRESH: rows the user has typed into are the
    user's answer and are left exactly alone, and every other row follows the
    mission. Without it (a planform mode change) the row is rewritten,
    because choosing the mode is choosing to ask the question.

    ``rows`` limits WHICH rows the overwrite applies to, and exists because
    that last sentence is only true of the rows the mode owns. Choosing the
    wing-loading mode is choosing to re-ask the wing's SIZE; it is not a
    statement about how long the fuselage is, so a planform switch that
    rewrote the whole of :func:`size_band_defaults` would silently discard a
    separation band the user had typed.
    """
    W = S["wing"]
    src = bounds_source(S)
    want = size_band_defaults(S)
    written = []
    for row, (lo, hi) in want.items():
        if rows is not None and row not in rows:
            continue
        if only_shell_owned and src.get(row) not in SHELL_BOUNDS_SOURCES:
            continue
        W["bounds"][row] = [float(lo), float(hi)]
        src[row] = "shell"
        written.append(row)
    return written


def measured_bands(S: dict) -> dict:
    """``{row: (lo, hi)}`` — the bands a MEASUREMENT currently holds."""
    W = S["wing"]
    return {row: (float(W["bounds"][row][0]), float(W["bounds"][row][1]))
            for row, source in bounds_source(S).items()
            if source == "recommended" and row in (W.get("bounds") or {})}


def relax_measured_bands(S: dict, carried: dict) -> list:
    """Carry a measurement through a BUILDER change. Returns the rows written.

    A builder control is not a mission. Ticking the tail or choosing a tip
    device changes the design VECTOR, so the measurement taken over the old
    vector is not an answer about the new one — but throwing it away puts the
    box back on the mission's full band, and the user watches the span row go
    8.1 – 18.6 m, then 11.5 – 13.9 m, then 8.1 – 18.6 m again, twice per
    menu. The shell looks like it is changing its mind about the aeroplane
    when all it did was re-measure.

    So the band is RELAXED instead of dropped, and asymmetrically:

    * the BOTTOM goes back to the mission's own band. A tip device is bought
      precisely to make a shorter wing do the same job, so the low end is the
      end the new vector is most likely to move, and opening it is the safe
      direction — a search is never wrong for having room.
    * the TOP is KEPT. "Nothing above 13.9 m paid for itself on this mission"
      is a statement about the mission's weight, speed and area, none of
      which a builder menu touches.

    On the shipped air mission that reads 8.124 – 13.925 m after a tip-device
    toggle, against 8.124 – 18.634 m before this existed.

    It is not a ratchet and cannot become one: the next pass measures over
    :func:`size_band_defaults`, not over this (``_unnarrowed_cfg``), and it
    overwrites what this wrote. A MISSION change still drops the lot
    (:func:`drop_recommended_bounds`) — there the mission's own band has
    moved, so ``[mission_lo, measured_hi]`` is two aeroplanes spliced
    together.

    A row the user has since typed into is theirs and is not restored.
    """
    from aerobo import api

    if not carried:
        return []
    W = S["wing"]
    src = bounds_source(S)
    want = size_band_defaults(S)
    published = api.PROBLEM_SPECS[W["problem"]].default_bounds or {}
    proposed = {}
    for row, band in carried.items():
        if row == TAIL_SPAN_KEY:
            continue                 # a FLAG pair, reported and not imposed
        base = want.get(row) or published.get(row)
        if base is None:
            continue                 # a row this family does not have
        if row in (W.get("bounds") or {}) \
                and src.get(row) not in SHELL_BOUNDS_SOURCES:
            # ...the user has since answered it. Asked as "is the row WRITTEN
            # and not the shell's", because a typed row has NO source at all
            # (`_set_bound` pops it) — testing `src.get(row) is None` for
            # "nobody has written this" would let the carry overwrite every
            # typed band in the box.
            continue
        lo, hi = float(base[0]), min(float(band[1]), float(base[1]))
        if hi > lo:
            proposed[row] = (lo, hi)
    written = []
    for row, (lo, hi) in hold_the_stated_design(S, proposed).items():
        if not hi > lo:
            continue
        W["bounds"][row] = [float(lo), float(hi)]
        src[row] = "recommended"
        written.append(row)
    return written


def drop_recommended_bounds(S: dict) -> list:
    """Take back every band a MEASUREMENT wrote. Returns the rows dropped.

    A recommendation is a statement about a MISSION and a box: draw this box,
    evaluate, keep what flew. Move the mission and the bands it produced are
    an answer to a question nobody is asking any more — so they go, and the
    rows they were holding return to what the shell would have written
    anyway: the mission's own band for a SIZE row, the family's published one
    for everything else.

    A row the user typed is theirs and is not here to be dropped: only
    ``"recommended"`` is taken back, which is exactly the set the shell wrote
    without being asked.
    """
    W = S["wing"]
    src = bounds_source(S)
    want = size_band_defaults(S)
    dropped = []
    for row, source in list(src.items()):
        if source != "recommended":
            continue
        dropped.append(row)
        if row == TAIL_SPAN_KEY:
            # a FLAG pair, not a band: taking it back switches the row off
            # again, which is where it was before anything measured it
            for key in TAIL_SPAN_FLAGS:
                W["flags"].pop(key, None)
            src.pop(row, None)
            continue
        if row in want:
            # a SIZE row is not deleted, it is REWRITTEN, and the one place
            # that knows what to is the call below — deleting it here would
            # leave the row on the family's published 6–40 m, which is the
            # bug this function exists to close, moved one line down
            continue
        W["bounds"].pop(row, None)
        src.pop(row, None)
    if dropped:
        write_size_bands(S, only_shell_owned=True)
    return dropped


def set_planform(S: dict, value: str) -> list[str]:
    """Answer the planform question; returns the derivation notes.

    One control owns WHO DECIDES THE WING'S SIZE, and this is what it writes.
    Choosing the wing-loading mode selects the family's ``size_ws`` twin and
    WRITES the opening span band into the design box as a user row — written
    rather than left implicit because the implicit band is a fraction of the
    nominal span, so a box nobody had typed into would drift every time the
    chosen span or stage 2's estimate moved, and the constraint the user set
    would move under them. Leaving the mode takes the row back out: a span
    band is a question only where the span is searched.

    ONE ROW PER SURFACE THAT HAS A SPAN (:func:`span_rows`). A tandem pair
    searches two, and a mode that opened a band for the front wing only would
    leave the rear one on the family's published 6–40 m — nobody's opinion
    about the pair on screen, and a silent asymmetry between two wings the
    search is meant to compare.
    """
    from gui.nice_app import _SPECIAL_LABEL, normalise_choices

    W = S["wing"]
    if W["choices"].get("planform") == value:
        return []
    W["choices"]["planform"] = value
    # the same normalisation every other builder change goes through: a
    # planform VALUE that selects a family (the aircraft one, which V3 does
    # not offer but a preset may hold) has no tip-device or section solver,
    # and the state may not keep a choice its own menus cannot show. ``keep``
    # makes the newest answer outrank the older ones.
    dropped = normalise_choices(W["choices"], keep="planform")
    notes = apply_choices(S)
    if dropped:
        names = ", ".join(_SPECIAL_LABEL.get(k, k) for k in dropped)
        notes = [f"reset {names}: no combined solver with this planform"] \
            + notes
    # every SIZE row this mode searches opens on the MISSION's own band —
    # the spans and, on the free planform, the area. The free planform used
    # to take the other branch and POP b_m instead, because
    # ``span_is_searched`` excludes the ``size`` modifier: choosing the one
    # mode whose span is genuinely a design variable deleted the span band
    # and left the row on the family's published 6-40 m.
    from aerobo import api as _api

    asked = span_is_searched(S) or _api.is_sized(_api.modifiers_of(
        W["problem"]))
    src = bounds_source(S)
    if asked:
        # the rows THIS MODE owns are re-asked; the second surface's layout
        # rows are only refreshed where nobody has typed into them
        write_size_bands(S, rows=size_rows(S))
        write_size_bands(S, only_shell_owned=True)
    else:
        for row in ("b_m", "b_rear_m", AREA_ROW):
            W["bounds"].pop(row, None)
            src.pop(row, None)
    # ...and the LOADING's own row, on the mode that searches it, for the
    # same reason and with the same lifetime: the band is the question that
    # mode asks, so it belongs in the design box where every other band is,
    # and it comes back out when the mode is left.
    #
    # Through :func:`write_size_bands`, not by hand, so it is STAMPED like
    # every other band the shell derived. Written straight into
    # ``W["bounds"]`` it was a "user" row from birth: the mission could never
    # refresh it and a problem change could never restore it.
    if loading_is_searched(S):
        write_size_bands(S, rows=(WS_ROW,))
    else:
        W["bounds"].pop(WS_ROW, None)
        src.pop(WS_ROW, None)
    return notes


def published_arm_box(S: dict) -> tuple | None:
    """``(lo, hi)`` the HORIZONTAL SEPARATION is calibrated over, or ``None``.

    The family's published ``l_t_m`` row — read off the FREE-arm twin, which
    is the one that carries the arm as a design variable, so this answers in
    either arm mode. It is the interval the family's results were MEASURED
    over (``tail.L_T_BOUNDS`` = 3–8 m in air, ``hydrotail.L_T_BOUNDS`` =
    0.5–1.5 m in water) and the box a searched arm opens on — NOT a limit:
    a STATED arm is free above ``tail.L_T_MIN_M``, because how long an
    aeroplane is, is a design decision. Quoting it is how the shell says
    where the extrapolation starts.

    The twin is reached through ``derive_problem`` on the session's own
    choices, never by editing the problem's NAME: the names differ per family
    ("tail (fixed arm)" against "hydrofoil + elevator [fixed arm]"), and a
    helper that matches one spelling silently answers ``None`` for the other.
    """
    from gui.nice_app import derive_problem

    from aerobo import api

    ch = S["wing"]["choices"]
    try:
        name, _ = derive_problem(dict(ch, tail_arm="free"))
        row = api.PROBLEM_SPECS[name].default_bounds.get("l_t_m")
    except Exception:                       # noqa: BLE001 — a read-out
        return None
    if row is None:
        return None
    return float(row[0]), float(row[1])


def height_bands(S: dict) -> dict | None:
    """What this session's VERTICAL SEPARATION means, or ``None``.

    ``{"measured": (lo, hi), "grid": z}`` — the band the family's results
    were MEASURED over, and the height below which the second surface's
    downwash stops being grid-converged (``tail.DZ_GRID_FRAC`` * span).
    Neither is a limit; both are things the shell has to be able to say.

    :func:`published_arm_box`'s twin, with one difference that matters: the
    arm's band is absolute metres, so the family's STATIC box is the right
    answer, while the height's is a FRACTION OF THE SPAN
    (``wingtail.Z_T_FRAC_BOUNDS`` = 0.05 b … 0.30 b). Quoting the static box
    would quote the 10 m trim wing's 0.5–3 m at a user flying a 23.6 m span,
    which is the drift ``config.FLAG_MOVED_ROWS`` exists for — so this
    builds with the session's FLAGS and reads the row off the result.

    Built with NO bounds overrides, deliberately: with them the row that
    comes back is the user's own band, and a sentence that quotes it as "the
    band this family was calibrated over" tells them their own number is the
    measurement.
    """
    from aerobo import api, hydrotail as _ht, wingtail as _wt
    from aerobo.tail import DZ_GRID_FRAC

    from gui.nice_app import derive_problem

    from . import config

    ch = S["wing"]["choices"]
    try:
        name, _ = derive_problem(dict(ch, tail_height="free"))
        spc = api.PROBLEM_SPECS[name]
        built = spc.build({} if spc.uses_mission else None,
                          config.flags(S), None)
        labels = list(built.param_labels)
        if api.TAIL_HEIGHT_KEY not in labels:
            return None
        lo, hi = built.bounds[labels.index(api.TAIL_HEIGHT_KEY)]
    except Exception:                       # noqa: BLE001 — a read-out
        return None
    lo, hi = float(lo), float(hi)
    # the band's own low end IS its family's shallow FRACTION times the span,
    # whichever sign convention this medium uses, so the span comes back out
    # of it rather than being re-derived from a size the second surface may
    # not share. THE FRACTION IS THE FAMILY'S: air opens the row at the
    # measurement height (0.05 b) and water opens it at the grid line
    # (0.01 b), because a foiling craft is flat — one divisor for both would
    # report a water span five times too large.
    frac = float((_ht.Z_T_FRAC_BOUNDS if spc.medium == "water"
                  else _wt.Z_T_FRAC_BOUNDS)[0])
    span = min(abs(lo), abs(hi)) / frac if frac else 0.0
    return {"measured": (lo, hi), "grid": DZ_GRID_FRAC * span}


def family_cg(S: dict) -> float | None:
    """The CG this family flies when NOBODY states one, or ``None``.

    Not a constant this shell can restate: in air it is the LAYOUT's
    (``tail.X_CG_BY_TYPE``, aft on a tail and forward on a canard) and in
    water it is a FRACTION OF THE ARM (``hydrotail.X_CG_FRAC``), so the only
    honest source is a built problem — the same mid-box solve the margin
    read-out is quoted from.

    Asked by clearing a stated CG for the length of that solve, because
    :func:`pitch_stability` reports what the RUN flies and the run flies the
    stated one. The distinction is the whole point of the sentence this
    number appears in: "empty = this family's calibrated value, X" beside a
    typed X says the opposite of what it means.
    """
    ch = S["wing"]["choices"]
    held = ch.get("tail_cg_m")
    if held is None:
        st = pitch_stability(S)
        return None if st is None else float(st["x_cg"])
    ch["tail_cg_m"] = None
    try:
        st = pitch_stability(S)
    finally:
        ch["tail_cg_m"] = held
    return None if st is None else float(st["x_cg"])


def set_tail_arm(S: dict, value: str) -> list[str]:
    """Whether the HORIZONTAL SEPARATION is SEARCHED or STATED: ``"free"``
    leaves the arm a design variable (``l_t_m`` stays a row of the design
    box, which is where its min and max are set), ``"fixed"`` selects the
    family's fixed-arm twin and sends the typed metres as a flag.

    It says nothing about the CG. The CG is its own question, asked on the
    same card in every state and travelling as ``x_cg_m`` on BOTH twins — so
    nothing is cleared here. (It used to be: the card asked WHICH END of the
    lever was stated, and stating the separation wiped a typed CG so the
    hint promising the family's calibrated one stayed true. The hint is gone
    with the either/or, and with it the reason to throw the number away.)

    The arm the flag carries has to be one that MEANS something on this
    family: the air default (5.5 m) is four times the water box's ceiling,
    i.e. a hull four times as long as the craft the water family models.
    So a value NOBODY TYPED is re-defaulted onto the middle of the box this
    family was measured over (:func:`published_arm_box`) — the same duty
    ``set_planform`` does for the span band.

    **A value the user typed is never touched.** That distinction is the
    whole rule. The snap used to fire on "outside the calibrated band",
    which quietly deleted every arm a user had a reason to state: type 1.5 m
    for a small aeroplane, touch the toggle, and it was 5.5 m again — the
    calibration acting as a ban one control further out, after the solvers
    had stopped enforcing it (``tail.arm_row``). A short arm is exactly the
    case the model exists to study, and both media build and solve one: the
    water family flies a 5.5 m arm and the air family a 0.5 m one, so there
    is no refusal left to protect either. What survives is the floor the
    pitch solve genuinely needs (``tail.L_T_MIN_M``) and a NOTE saying where
    the measurements end.

    Lives here rather than in the toggle's handler so a preset, a test or
    another stage reaches the same state the menu does.
    """
    from gui.nice_app import BUILDER_DEFAULTS, normalise_choices

    from aerobo.tail import L_T_MIN_M

    W = S["wing"]
    ch = W["choices"]
    if ch.get("tail_arm", "free") == value:
        return []
    box = published_arm_box(S) if value == "fixed" else None
    ch["tail_arm"] = value
    normalise_choices(ch, keep="tail_arm")
    notes = apply_choices(S)
    if box is not None:
        lo, hi = box
        try:
            held = float(ch.get("tail_arm_m"))
        except (TypeError, ValueError):
            held = None
        # nobody's number: the shell's own default, carried in from whatever
        # family was on screen before this one
        untouched = held is not None and held == BUILDER_DEFAULTS.get(
            "tail_arm_m")
        unflyable = held is None or held < L_T_MIN_M
        if unflyable or (untouched and not lo <= held <= hi):
            ch["tail_arm_m"] = round(0.5 * (lo + hi), 3)
            notes = notes + [
                f"separation set to {ch['tail_arm_m']:g} m: the middle of "
                f"the {lo:g}–{hi:g} m band this family was calibrated over. "
                f"Type any positive length from there — the number is a "
                f"starting point, not a limit"]
        elif not lo <= held <= hi:
            notes = notes + [
                f"separation kept at {held:g} m — outside the {lo:g}–{hi:g} m "
                f"band this family was calibrated over, so the run is an "
                f"extrapolation of the published measurements. It is built "
                f"and flown as stated"]
    return notes


def set_span_searched(S: dict, on: bool) -> list[str]:
    """The wing-loading mode, as a switch: ON searches the span at the
    mission's W/S, OFF gives the size back to the card.

    Kept as its own name because that IS the question the mode answers, and
    because everything outside the planform menu (a preset, a test, another
    stage) asks it this way.
    """
    if bool(on) == span_is_searched(S):
        return []
    return set_planform(S, "wing_loading" if on else "fixed")


def flown_aspect_ratio(S: dict) -> float:
    """The aspect ratio the RUN flies — b²/S, never a typed number.

    With the span searched this is the mid-box wing's; with it fixed it is
    the estimate's own, because the span was derived from that estimate.
    """
    size = flown_size(S)
    if not size:
        return nominal_aspect_ratio(S)
    b, area = float(size[0]), float(size[1])
    return b * b / area if area > 0.0 else nominal_aspect_ratio(S)


def flown_ar_band(S: dict) -> tuple | None:
    """``(AR_lo, AR_hi)`` the span box implies, or ``None`` when the span is
    not searched (there is one aspect ratio then, :func:`flown_aspect_ratio`).
    """
    box = span_box(S)
    size = flown_size(S)
    if box is None or not size:
        return None
    area = float(size[1])
    if not area > 0.0:
        return None
    return box[0] ** 2 / area, box[1] ** 2 / area


# TWO AREAS, AND THE DIFFERENCE MATTERS ON A PAIR. The two functions above
# report b²/S_ref — the REFERENCE area, which is the aspect ratio stage 2's
# estimate is stated in and the one a single-surface family flies anyway. The
# three below report the aspect ratio of ONE WING on ITS OWN share of that
# area, which is what the solvers actually enforce (``sizing.check_ar`` is
# called per wing on half the pair's total, so a card quoting b²/S_total
# beside a run refusing the wing would be reporting half the number the
# refusal is about).
def wing_area_share(S: dict, row: str = "b_m") -> float:
    """The share of the reference area the surface of ``row`` carries.

    1.0 wherever there is one lifting surface. On a pair it is the AREA SPLIT
    — itself a design variable (``area_split_front``), so the honest single
    number is the middle of the box the run searches, the same convention
    every other derived number in this shell follows.

    ONE KNOWN DISAGREEMENT, recorded rather than papered over: the solvers
    VALIDATE a pair's aspect ratio on a nominal half of the total
    (``check_ar(b, 0.5 * S_total)`` in tandem.py / tandemvlm.py) while
    REPORTING each wing's true one. With the split box untouched the two
    agree exactly — its midpoint is 0.5 — and they part company only if
    somebody narrows ``area_split_front`` off centre, where this reports the
    aspect ratio the wing really flies and the run refuses on the nominal
    one. Making the check use the candidate's own split would move which
    candidates are feasible in the free-planform pair as well, so it is a
    physics decision, not a view one.
    """
    from . import config

    rows = span_rows(S)
    if len(rows) < 2 or row not in rows:
        return 1.0
    got = config.effective_bounds(S).get("area_split_front")
    split = 0.5 if got is None else 0.5 * (float(got[0][0]) + float(got[0][1]))
    return split if row == rows[0] else 1.0 - split


def wing_aspect_ratio(S: dict, row: str = "b_m") -> float | None:
    """The aspect ratio the surface of ``row`` flies, on its own area."""
    size = flown_size(S)
    if not size:
        return None
    area = float(size[1]) * wing_area_share(S, row)
    if not area > 0.0:
        return None
    box = span_box(S, row)
    b = 0.5 * (box[0] + box[1]) if box else nominal_span(S, row)
    return b * b / area if b > 0.0 else None


def wing_ar_band(S: dict, row: str = "b_m") -> tuple | None:
    """``(AR_lo, AR_hi)`` the box of ``row`` implies for that wing alone.

    THE CORNERS OF THE BOX, not the ends of the span row at one area. Where
    the AREA is a design-box row too (the water size modes), the widest wing
    the run can draw sits over the SMALLEST area and the narrowest over the
    largest, so quoting both against a single area understates the reach at
    both ends. Measured on the water default with both rows open: 2.0-32.0
    against corners the solver itself refuses at 1.25 and 80, on a card whose
    own hint promises that corners outside 3-40 are refused per design.
    """
    box = span_box(S, row)
    size = flown_size(S)
    if box is None or not size:
        return None
    share = wing_area_share(S, row)
    area_box = span_box(S, "S_m2")
    if area_box is not None:
        lo_a, hi_a = area_box[0] * share, area_box[1] * share
        if not (lo_a > 0.0 and hi_a > 0.0):
            return None
        return box[0] ** 2 / hi_a, box[1] ** 2 / lo_a
    area = float(size[1]) * share
    if not area > 0.0:
        return None
    return box[0] ** 2 / area, box[1] ** 2 / area


def device_reaches_past_the_span(S: dict) -> bool:
    """Is a tip device reaching BEYOND the span box?

    A span limit is a limit on the WIDTH THE AIRCRAFT OCCUPIES, so what it
    has to bound is the PROJECTED span. On the span-capped families
    (``api.CAPPED_MODES``) it already does: the wing panel shrinks to pay for
    the device's horizontal projection, so ``b_m`` is that width exactly, and
    V3 picks the capped variant wherever one exists (nice_app.WINGLET_SHAPES
    lists it first). Where only a FREE-span device exists the device projects
    outboard of ``b_m`` and the box is the wing panel alone — true here, and
    the wing stage says so rather than quoting a width the run exceeds.
    """
    return S["wing"]["choices"].get("winglets", "none") == "free"


# ------------------------------------------------------------- the car
#: WHAT A CAR REAR WING ACTUALLY STATES, and what it does not.
#:
#: ``carwing.CarWingProblem`` reads exactly ONE number off this stage: the
#: speed, and it reads it as a flag because the family refuses a mission spec
#: outright (``api._make_car_wing_builder``: "weight and altitude do not
#: enter"). Its density is the module constant ``carwing.RHO_AIR``, its span
#: and its reference area are ``b_m`` and ``S_m2`` — two ordinary rows of
#: stage 3's design box — and nothing weighs it, because it is bolted to a
#: car.
#:
#: So the aircraft mission's other three questions have no answer here. A
#: design WEIGHT is a load this wing does not carry (it MAKES a load); an
#: ALTITUDE moves a density the solver does not read; and a WING LOADING is
#: W/S with no W. All three were asked all the same, because one number
#: downstream needed them: stage 2 screens its section at CL = W/(qS), so a
#: weight and an area were invented to imply a lift coefficient.
#:
#: They are not invented any more. The track card asks the coefficient
#: ITSELF — the reference CZ the section is designed at — and the chord it is
#: designed for comes off the design box's own two size rows. The stored
#: ``W_N`` / ``s_ref_m2`` / ``altitude_m`` are then a MIRROR of those
#: answers (:func:`track_point_sync`), kept in step so every reader that
#: speaks the aircraft vocabulary — ``api.design_point``, :func:`wing_guess`,
#: :func:`taper_re_band` — keeps working without being taught a second one.
TRACK_POINT_NOTE = (
    "A rear wing carries no weight and flies at one altitude: the car's. So "
    "this card asks the two numbers the track problem actually has — the "
    "SPEED, which reaches the solver, and the reference CZ the section is "
    "designed at. The wing's size is not asked here either: its span and its "
    "reference area are rows of stage 3's design box, and the chord this "
    "section is screened at comes from them.")


def car_wing(S: dict) -> bool:
    """Is this session designing a CAR REAR WING?

    One predicate for a question ten call sites were asking by hand, three
    of them with exactly this body (``S["medium"] == "track" and not
    airfoil_only(S)``) and the rest with the bare medium — which is not the
    same question, because a section-only session in the track medium is a
    SECTION session and has to explain itself as one.

    What it is for: a rear wing is bolted to a car. It makes a load rather
    than carrying one, nothing weighs it, and it has no free-flight degrees
    of freedom at all — so a whole class of question that is well posed for
    an aeroplane (how it is trimmed, how it is rolled, how it behaves when
    it is let go) has no answer here, and a shell that asks it is inventing
    a vehicle. The twin of :func:`airfoil_only`, and read the same way.
    """
    return not airfoil_only(S) and S.get("medium") == "track"


def track_reference_size(S: dict) -> tuple[float, float] | None:
    """``(b, S)`` the CAR's section is designed at — mid design box.

    The car's two dimensions are design VARIABLES (``b_m``, ``S_m2``), so
    there is no single planform to quote and no wing loading to derive one
    from. The middle of the box the run will actually search
    (:func:`_searched_box`, so a narrowed row moves it) is the honest
    stand-in, and every card that shows it says which box it came from.

    ``None`` where the family carries neither row — the fixed-geometry car
    variants — and the caller falls back to that family's own planform.
    """
    box = _searched_box(S)
    got = []
    for row in ("b_m", "S_m2"):
        band = box.get(row)
        if band is None:
            return None
        mid = 0.5 * (float(band[0]) + float(band[1]))
        if not mid > 0.0:
            return None
        got.append(mid)
    return (got[0], got[1])


def track_design_cz(S: dict) -> float:
    """The reference downforce coefficient the section is screened at.

    Opens on :data:`REFERENCE_CL`, which is what this shell already
    back-derived a weight from — so an untouched track session designs its
    section at exactly the point it always did.
    """
    try:
        cz = float(S["mission"].get("cz_design"))
    except (TypeError, ValueError):
        cz = 0.0
    return cz if cz > 0.0 else REFERENCE_CL


def set_track_design_cz(S: dict, value) -> bool:
    """Set it. False (and nothing stored) if it is unusable."""
    try:
        cz = float(value)
    except (TypeError, ValueError):
        return False
    if not cz > 0.0:
        return False
    S["mission"]["cz_design"] = cz
    track_point_sync(S)
    return True


def track_point_sync(S: dict) -> bool:
    """Hold the stored mission in step with what the CAR states.

    The track card states a speed and a CZ; the design box states the size.
    This writes the three aircraft-vocabulary numbers those imply, so that
    the one funnel every stage reads (:func:`design_point`) returns the car's
    own point without any caller learning a second vocabulary:

        ``altitude_m``  0 — ``carwing`` flies ``carwing.RHO_AIR``, which IS
                        the ISA sea-level density, and an altitude the solver
                        does not read must not move the Reynolds number the
                        section IS screened at (it did: 1000 m took 9 % off
                        the density stage 2 designed against while stage 3
                        flew 1.225 regardless).
        ``s_ref_m2``    the design box's ``S_m2`` row, mid-box.
        ``W_N``         CZ q S — chosen so that ``CL = W/(qS)`` comes back
                        as exactly the stated CZ. It is a mirror, not a
                        load: :func:`config.stated_load_note` already says
                        the run does not fly it, and the track card no
                        longer shows it at all.

    ...and the aspect-ratio ESTIMATE, for the same reason: b²/S off the same
    two rows. On every other family that number is a guess the user owns,
    because the mission states no planform. The car states one — two bands —
    so a guess beside them would be a second answer, and stage 2 shows it as
    a read-out there.

    Returns True when something moved, so a caller can decide to repaint.
    """
    from aerobo import api

    if S.get("medium") != "track" or airfoil_only(S):
        return False
    m = S["mission"]
    size = track_reference_size(S) or api.planform_size(S["wing"]["problem"])
    if not size:
        return False
    b, s_ref = float(size[0]), float(size[1])
    if not (b > 0.0 and s_ref > 0.0):
        return False
    try:
        v = float(m["V"])
    except (TypeError, ValueError):
        return False
    cz = track_design_cz(S)
    try:
        # W = 1 N only to READ rho and q: neither depends on the load, and
        # the load is what this call exists to compute
        probe = api.design_point(
            medium="track", W_N=1.0, V=v, s_ref_m2=s_ref,
            aspect_ratio=(b * b / s_ref), taper=REFERENCE_TAPER,
            altitude_m=0.0)
    except (ValueError, TypeError):
        return False
    want = {"altitude_m": 0.0, "s_ref_m2": s_ref,
            "W_N": cz * float(probe["q"]) * s_ref}
    moved = any(float(m.get(k, float("nan"))) != v2 for k, v2 in want.items())
    m.update(want)
    m.setdefault("cz_design", cz)
    if abs(section_aspect_ratio(S) - b * b / s_ref) > 0.0:
        set_section_aspect_ratio(S, b * b / s_ref)
        moved = True
    return moved


# --------------------------------------------------------- derived mission
def design_point(S: dict) -> dict:
    """The SECTION design point of the current mission, or ``{"error": …}``.

    One call, one definition: the airfoil stage screens at this Reynolds
    number and this lift coefficient, the properties grid quotes them, and
    the wing stage's trim target comes from the same weight and area.

    The lift coefficient is a pure mission quantity (W / qS). The chord — and
    therefore ``b``, ``mac`` and ``re_mac`` — is quoted at stage 2's
    aspect-ratio ESTIMATE, so every one of those keys is a statement about
    the section, not about the planform stage 3 will fly. Use
    :func:`flown_size` for that.
    """
    from aerobo import api

    # THE CAR STATES ITS POINT DIFFERENTLY, and the difference is resolved
    # here rather than in every caller: a rear wing has no weight and no
    # wing loading, so the numbers below are the mirror of the two it does
    # state (:func:`track_point_sync`). Idempotent, and cheap — one
    # api.design_point call on a track session, none on any other.
    track_point_sync(S)
    m = S["mission"]
    try:
        return api.design_point(
            medium=S["medium"], W_N=float(m["W_N"]), V=float(m["V"]),
            s_ref_m2=float(m["s_ref_m2"]),
            aspect_ratio=section_aspect_ratio(S), taper=float(m["taper"]),
            altitude_m=float(m["altitude_m"]),
            water=S.get("water", "sea"),
            depth_m=(float(m["depth_m"]) if S["medium"] == "water"
                     else None))
    except (ValueError, TypeError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def mission_valid(S: dict) -> bool:
    return "error" not in design_point(S)


# ------------------------------------------------------------ the two modes
#: WHAT THIS SESSION IS FOR. Two answers, because the pipeline is asked to
#: do two jobs and only one of them has a vehicle in it:
#:
#: * ``pipeline`` — the shipped four stages. A mission sizes a surface, the
#:   section is designed at the point that mission implies, and the wing is
#:   flown at it.
#: * ``airfoil`` — a SECTION and nothing else. The flow is stated directly
#:   (:func:`flow_point`) and stage 2 is the whole of the work.
#:
#: The second one exists because the section search always WAS standalone:
#: ``api.optimize_airfoil`` and ``api.screen_airfoils`` take
#: ``(re, mach, cl_design)`` and never see a mission. What the shell had no
#: way to do was ASK for that point — every route into stage 2 ran through a
#: design weight and a reference area that somebody designing an aerofoil
#: does not have, and inventing them to get past stage 1 states a vehicle
#: nobody chose (and screens the section at ITS chord).
MODES = {
    "pipeline": ("Vehicle — mission, section, wing",
                 "State a mission. The section is designed at the Reynolds "
                 "number and lift coefficient it implies, and the wing is "
                 "sized and flown at the same point."),
    "airfoil": ("Airfoil only — state the flow",
                "No vehicle. State the flow the section works in — a fluid, "
                "a speed and a chord, or a density, a viscosity, a Mach "
                "number and a speed — and design the aerofoil in it. The "
                "wing stages are not part of this session."),
}

MODE_DEFAULT = "pipeline"


def session_mode(S: dict) -> str:
    """Which of :data:`MODES` this session is in."""
    mode = str(S.get("mode") or MODE_DEFAULT)
    return mode if mode in MODES else MODE_DEFAULT


def airfoil_only(S: dict) -> bool:
    """Is this session designing a SECTION and nothing else?

    Asked by everything that would otherwise reach for a vehicle: the stage
    gating, the second surface, the wing-mode objective and the section's
    own operating point. One predicate, so a mode cannot be half applied.
    """
    return session_mode(S) == "airfoil"


def set_mode(S: dict, mode: str) -> bool:
    """Switch what the session designs. Keeps BOTH sides' answers.

    A mission and a flow point are separate stored answers, so switching is
    not destructive in either direction: come back to the vehicle and its
    mission is exactly where it was left. What the switch does move is the
    UI's own selection, because two of the stages do not exist on the
    airfoil-only side and a shell left pointing at one of them would open on
    a locked stage (the caller does that — see ``mission._set_mode``).
    """
    if mode not in MODES or mode == session_mode(S):
        return False
    S["mode"] = mode
    return True


# ------------------------------------------------------------- the FLOW
#: the fluid faces stage 1 offers in airfoil-only mode. Air and water DERIVE
#: their state (ISA at an altitude; the named water), custom takes it
#: verbatim — see :func:`aerobo.api.flow_point`, which refuses a typed
#: density on a derived face rather than ignoring it.
FLUID_LABELS = {"air": "Air (ISA)", "water": "Water",
                "custom": "Custom fluid"}

#: the numbers the flow form stores. ``rho``/``mu``/``mach`` are the CUSTOM
#: face's own three; the derived faces never read them.
FLOW_NUMBERS = ("V", "chord_m", "cl_design", "altitude_m",
                "rho", "mu", "mach")

#: what an airfoil-only session opens on when the session's own mission
#: cannot be derived. It always can (``make_session`` builds one from the
#: family's published design point), so this is the guard and not the
#: answer: sea-level ISA at the published air speed, one metre of chord.
FLOW_FALLBACK = {"V": 14.6, "chord_m": 1.0, "cl_design": 0.5}


def flow_defaults(S: dict) -> dict:
    """The flow an airfoil-only session OPENS on.

    Not a set of literals: the point the session's OWN mission already
    implies — its speed, its altitude, the mean chord that mission's area
    and stage 2's aspect-ratio estimate give, and the lift coefficient it
    trims to. So the two modes open on the same aerodynamics and a section
    designed either way is comparable; every number here is the user's to
    replace, and nothing follows the mission after this.

    The CUSTOM face's three numbers (rho, mu, Mach) open on what the named
    fluid derives, so switching faces states the same flow rather than
    teleporting to somebody else's.
    """
    from aerobo import api

    m = S["mission"]
    fluid = "water" if S["medium"] == "water" else "air"
    dp = design_point(S)
    out = {"fluid": fluid, "water": str(S.get("water", "sea")),
           "altitude_m": 0.0,
           "V": float(FLOW_FALLBACK["V"]),
           "chord_m": float(FLOW_FALLBACK["chord_m"]),
           "cl_design": float(FLOW_FALLBACK["cl_design"]),
           "rho": 0.0, "mu": 0.0, "mach": 0.0,
           # ...and whether that Mach number is FLOWN (:func:`flown_mach`).
           # Off, like the wing stage's own twin, and for a reason that is
           # this mode's rather than that one's — see the docstring there.
           "apply_mach": False,
           "accepted": False}
    if "error" not in dp:
        out["V"] = float(dp["v_ms"])
        out["chord_m"] = float(dp["mac"])
        out["cl_design"] = float(dp["cl_design"])
        if fluid == "air":
            out["altitude_m"] = float(m["altitude_m"])
    try:
        base = api.flow_point(fluid=fluid, V=out["V"],
                              chord_m=out["chord_m"],
                              altitude_m=out["altitude_m"],
                              water=out["water"])
    except (ValueError, TypeError):
        base = api.flow_point(fluid="air", V=float(FLOW_FALLBACK["V"]),
                              chord_m=float(FLOW_FALLBACK["chord_m"]))
    out["rho"], out["mu"] = float(base["rho"]), float(base["mu"])
    out["mach"] = float(base["mach"])
    return out


def flow_state(S: dict) -> dict:
    """The stored flow answers, created on first ask.

    Created lazily as well as in ``make_session`` because a session dict
    that predates this mode (a stored one, a test's) must not KeyError the
    moment something asks what flow it states.
    """
    flow = S.get("flow")
    if not isinstance(flow, dict) or "fluid" not in flow:
        flow = flow_defaults(S)
        S["flow"] = flow
    return flow


def flow_point(S: dict) -> dict:
    """The stated flow, derived — or ``{"error": …}``.

    The twin of :func:`design_point`, and the same contract: one call, one
    definition. In airfoil-only mode this IS the section's operating point
    (:func:`section_conditions` reads it), the stage-1 read-outs quote it,
    and the tree badge is off it — so no two of them can disagree.

    The physics is ``api.flow_point``'s, not this shell's: the ISA, the
    Sutherland viscosity, the named waters and the Reynolds number all live
    in ``aerobo`` beside the mission's, because a second definition of "the
    Reynolds number of this flow" in the GUI is exactly the drift this
    module exists to prevent.
    """
    from aerobo import api

    f = flow_state(S)
    kw: dict = {"fluid": str(f["fluid"]), "V": float(f["V"]),
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
        return api.flow_point(**kw)
    except (ValueError, TypeError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def flow_valid(S: dict) -> bool:
    return "error" not in flow_point(S)


def flown_mach(S: dict) -> float:
    """The Mach number the SEARCH runs at, which is a decision.

    ``flow_point``'s ``mach`` is a fact about the stated flow — 14.6 m/s at
    sea level IS M 0.043, and an altitude cannot imply anything else. Flying
    it is still a choice, and an expensive one:

    * ON — XFOIL is given ``MACH`` and applies its compressibility
      correction, so the polars are the ones that flow really has. But no
      cached polar in this repo is at that Mach (the whole screening
      checkpoint is M = 0, and ``screen_at_point`` compares the two
      exactly), so the library screen becomes a real sweep of the database
      rather than a cache read.
    * OFF — the section is designed incompressible, which is what every
      cached polar, every library ranking and every published study here
      already is. At the speeds this shell opens on the difference is a
      fraction of a percent; by M 0.3 (:data:`MACH_WARN`) it is not.

    Off by default, and said out loud on the card, for the same reason the
    wing stage's ``apply_mach`` is: a correction nobody asked for should not
    arrive with a session, and a number the user typed must not be applied
    behind their back either.
    """
    pt = flow_point(S)
    if "error" in pt or not flow_state(S).get("apply_mach"):
        return 0.0
    return float(pt["mach"])


def set_flow_apply_mach(S: dict, on: bool) -> bool:
    """Fly the stated Mach number, or design incompressible."""
    f = flow_state(S)
    if bool(f.get("apply_mach")) == bool(on):
        return False
    f["apply_mach"] = bool(on)
    f["accepted"] = False
    return True


def set_flow(S: dict, key: str, value) -> bool:
    """Type one flow number. Un-accepts the point, like a mission edit.

    The number is stored EVEN IF it makes the point underivable (a zero
    chord, a negative density): the field holds what was typed and stage 1
    says what is wrong with it, which is the same contract every other
    typed field in this shell has. What is refused here is a value that is
    not a number at all.
    """
    if key not in FLOW_NUMBERS:
        return False
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(v):
        return False
    f = flow_state(S)
    f[key] = v
    f["accepted"] = False
    return True


def set_fluid(S: dict, value: str) -> bool:
    """Switch which FACE the flow is stated in (or which water it is).

    Switching to the custom face carries the derived state across, so the
    three fields open on the flow the user was already in rather than on
    somebody else's constants — the same rule the size statements on the
    mission side follow (one quantity, several faces, no two of which can
    disagree).
    """
    from aerobo import api

    f = flow_state(S)
    if value in api.water_kinds():
        if str(f["water"]) == value:
            return False
        f["water"] = str(value)
        f["accepted"] = False
        return True
    if value not in FLUID_LABELS or str(f["fluid"]) == value:
        return False
    if value == "custom":
        pt = flow_point(S)
        if "error" not in pt:
            f["rho"], f["mu"] = float(pt["rho"]), float(pt["mu"])
            f["mach"] = float(pt["mach"])
    if value == "water":
        # water has no altitude, and api.flow_point refuses one rather than
        # ignoring it: the stated height belongs to the air face, so it is
        # dropped HERE, out loud, instead of failing the derivation
        f["altitude_m"] = 0.0
    f["fluid"] = str(value)
    f["accepted"] = False
    return True


def fluid_phrase(S: dict) -> str:
    """The stated fluid, as it reads INSIDE a sentence.

    ``FLUID_LABELS`` are control captions ("Air (ISA)"); lower-casing one
    mid-sentence produced "in air (isa)". One phrase, so every card that
    names the fluid in prose names it the same way.
    """
    from aerobo import api

    f = flow_state(S)
    if f["fluid"] == "water":
        return str(api.water_kinds()[str(f["water"])])
    return "air" if f["fluid"] == "air" else "the stated fluid"


def flow_summary(S: dict) -> str:
    """The stage-1 tree badge in airfoil-only mode."""
    pt = flow_point(S)
    if "error" in pt:
        return "invalid"
    return (f"Re {pt['re']:.3g}" +
            (f" · M {pt['mach']:.3f}" if pt["mach"] > 0.0 else ""))


# -------------------------------------------------- what stage 1 answered
def stage1_valid(S: dict) -> bool:
    """Is stage 1's point derivable — whichever point this mode asks for?"""
    return flow_valid(S) if airfoil_only(S) else mission_valid(S)


def stage1_accepted(S: dict) -> bool:
    """Has stage 1 been accepted? The one gate stage 2 hangs off."""
    return bool((flow_state(S) if airfoil_only(S)
                 else S["mission"])["accepted"])


def set_stage1_accepted(S: dict, value: bool) -> None:
    (flow_state(S) if airfoil_only(S) else S["mission"])["accepted"] = \
        bool(value)


def stage1_error(S: dict) -> str:
    """Why it is not derivable, or ""."""
    pt = flow_point(S) if airfoil_only(S) else design_point(S)
    return str(pt.get("error") or "")


def _is_second_surface_row(label: str) -> bool:
    """Is this design-vector row the SECOND surface's?

    The package's own naming convention: the tail / elevator's rows carry a
    ``_t`` suffix (``taper_t``, ``AR_t``, ``washout_t_deg``, ``chord_k1_t``,
    ``winglet_h_frac_t``), while a tandem's two WINGS are ``_front`` /
    ``_rear``. One reader, because the rule is a fact about the vector and
    not about any one card.
    """
    return label.endswith("_t") or label.endswith("_t_deg")


def taper_box(S: dict) -> list | None:
    """The taper interval the current family will actually search, if any.

    Tandem carries one per surface (``taper_front`` / ``taper_rear``); the
    union of them is what the section has to cover. The SECOND surface's own
    taper (``taper_t``) is NOT in it: this band is quoted as the interval the
    WING's mean chord — and so its Reynolds number — moves over, and a
    stabiliser's taper does not move the wing's chord. It went unnoticed
    while the designed tail was something the user had to switch on; the
    opening aeroplane designs its tail, so the default air session quoted a
    Reynolds band widened by another surface's row.

    The box the RUN searches (:func:`_searched_box`), not the family's
    published one: narrowing taper in the design box and then reading stage 1
    still say "λ ∈ [0.2, 1] is a design variable of the solver" was the
    mission quoting a Reynolds band over an interval nobody was searching.
    """
    bounds = _searched_box(S)
    rows = [v for k, v in bounds.items()
            if k == "taper" or (k.startswith("taper_")
                                and not _is_second_surface_row(k))]
    if not rows:
        return None
    return [min(float(r[0]) for r in rows), max(float(r[1]) for r in rows)]


def taper_re_band(S: dict) -> tuple[float, float] | None:
    """``(Re_lo, Re_hi)`` at the MAC over that taper box.

    The mission quotes Re at the mean chord (:data:`REFERENCE_TAPER`); this
    is how far the optimiser's own taper can move it, which is the honest
    way to state a Reynolds number for a planform that is not decided yet.

    TAPER ONLY. It is a trapezoid band: a free CHORD LAW reshapes the
    planform on top of that trapezoid, so the flown MAC can leave this
    interval, and the caller must say so wherever the law is on (which,
    since ``nice_app.BUILDER_START``, is the default). The run reports its
    own ``mac_true`` / ``re_true`` beside the Reynolds number it flew.
    """
    from aerobo import api

    box = taper_box(S)
    if not box:
        return None
    m = S["mission"]
    res = []
    for lam in box:
        try:
            dp = api.design_point(
                medium=S["medium"], W_N=float(m["W_N"]), V=float(m["V"]),
                s_ref_m2=float(m["s_ref_m2"]),
                aspect_ratio=section_aspect_ratio(S), taper=float(lam),
                altitude_m=float(m["altitude_m"]),
                water=S.get("water", "sea"),
                depth_m=(float(m["depth_m"]) if S["medium"] == "water"
                         else None))
        except (ValueError, TypeError):
            return None
        res.append(float(dp["re_mac"]))
    return (min(res), max(res))


#: above this Mach the incompressible section data stop being the right
#: data: every cached library polar and every live CST sweep is run at
#: M = 0, so a design point past it is being scored on the wrong tables
#: whether or not the wing solver applies its Prandtl-Glauert correction.
MACH_WARN = 0.3


def wing_loading(S: dict) -> float | None:
    """W/S [N/m²] — the quantity a wing is actually specified by.

    The session stores the AREA (every solver speaks area); the form shows
    the loading and derives the area from it, so there is still exactly one
    stored number and no pair that can disagree.
    """
    m = S["mission"]
    try:
        return float(m["W_N"]) / float(m["s_ref_m2"])
    except (TypeError, ValueError, ZeroDivisionError):
        return None


#: what the constraint diagram is asked, per medium. These are MISSION
#: numbers — the ones that decide a wing loading before any aerodynamics
#: happens — and they live beside the mission because that is whose answer
#: W/S is (aerobo.constraint_diagram). None means "this requirement is not
#: part of my mission", and the diagram simply leaves that line out.
WS_DIAGRAM_DEFAULTS: dict = {
    # What is NOT asked here, and why: the OPERATING POINT. The mission
    # above already states the speed, the altitude (and so the density) and
    # the depth, and the session already owns an aspect ratio (stage 3's
    # flown one, or stage 2's estimate while stage 3 follows it). The
    # diagram needs all four, and it takes them — asking again would put the
    # same number in two places, which is the one thing this shell does not
    # do. What is left here is what the diagram alone needs: the speeds and
    # distances the mission does NOT state, and the drag pair a wing loading
    # is matched against.
    # ``twr_available`` is the one input here that is not a REQUIREMENT but a
    # capability: the thrust-to-weight the aircraft HAS. Optional, and None
    # keeps every existing session's diagram unchanged — but it is what closes
    # the diagram. Without it the card can only report the ws_max LIMIT, and a
    # limit adopted as a design point is what leaves a searched wing loading
    # with nothing to stop it below the top of its band (results/
    # ws_band_study.json: no interior optimum in W/S, in any objective).
    "air": {"v_stall_ms": None, "cl_max": 1.6,
            "cd0": 0.025, "oswald_e": 0.85,
            "climb_rate_ms": 4.0, "turn_load_factor": None,
            "takeoff_distance_m": None, "landing_distance_m": None,
            "twr_available": None},
    "water": {"v_takeoff_ms": None, "cl_max": 0.9},
}

#: finite-wing CL_max as a fraction of the SECTION's cl_max. MODEL CHOICE,
#: the standard preliminary-design allowance for an unswept wing: the tips
#: stall first, so a wing never reaches its own aerofoil's two-dimensional
#: maximum. It is stated here rather than buried because the stall line of
#: the constraint diagram is the whole upper bound on W/S in air.
WING_CLMAX_FRAC = 0.9


def section_cl_max(S: dict, surface: str = "main"):
    """``(CL_max, why)`` the chosen section gives the WING, or ``None``.

    This is the answer to "how would I know CL_max": you would not — the
    section does, and stage 2 already screened it. Every library pick
    carries its measured ``clmax`` (airfoil_select's stall sweep), and the
    wing's is that times :data:`WING_CLMAX_FRAC`. A designed CST section
    has no screened cl_max, so it returns None and the card asks.
    """
    sec = section_of(S, surface) or {}
    try:
        cl = float(sec.get("clmax"))
    except (TypeError, ValueError):
        return None
    if not cl > 0.0:
        return None
    return (WING_CLMAX_FRAC * cl,
            f"{WING_CLMAX_FRAC:g} x cl_max {cl:.3g} of "
            f"{sec.get('name', 'the chosen section')}")


def stall_speed_at(S: dict, wing_loading_pa: float | None = None,
                   cl_max: float | None = None) -> float | None:
    """The speed this design ACTUALLY stalls at (or flies up at), or None.

    ``V_s = sqrt(2 (W/S) / (rho CL_max))`` — the same relation the stall
    constraint is written from, read the other way round. It is what makes
    the stall speed a REQUIREMENT rather than a guess: the card shows where
    the mission already sits, and the field states where it may not go.

    ``cl_max`` overrides the SECTION's, for the one caller that has to answer
    before a section exists: the diagram's own opening speed
    (:func:`ws_inputs`) is derived from the CL_max the diagram itself will be
    drawn with (``WS_DIAGRAM_DEFAULTS``), which on a fresh session is the only
    one there is. Without the override this returned None there and the
    opening requirement was a fraction of the cruise speed with nothing under
    it — see :data:`OPENING_SPEED_MARGIN`.
    """
    cl = ((float(cl_max), "stated") if cl_max is not None
          else section_cl_max(S))
    ws = wing_loading(S) if wing_loading_pa is None else float(wing_loading_pa)
    if cl is None or not cl[0] > 0.0 or not ws or ws <= 0.0:
        return None
    rho = float(design_point(S).get("rho") or 0.0)
    if rho <= 0.0:
        return None
    return float((2.0 * ws / (rho * cl[0])) ** 0.5)


#: the stall / fly-up speed a fresh card opens on, as a fraction of the
#: MISSION's speed. Both are speeds the mission does not state, and both are
#: tied to the one it does: 0.6 puts cruise at 1.67 x the stall speed (a
#: normal light-aircraft margin), and a foiling craft flies up at roughly
#: half its cruise. Starting points only — the field is there to be typed in.
STALL_SPEED_FRAC = 0.6
FLYUP_SPEED_FRAC = 0.5

#: ...AND THE OPENING REQUIREMENT MUST NOT REFUSE THE MISSION IT OPENS ON.
#: A fraction of the cruise speed is a fine starting point only where the
#: craft can actually meet it, and on the published foiling family it could
#: not: 0.5 x 12 m/s asked a 0.144 m² foil to carry 6 kN at 6 m/s, which
#: needs CL = 2.26 against the 0.9 the card is drawn with. Every fresh water
#: session therefore opened 2.51x over its own ceiling, and that ceiling
#: travels to the solver — a sized run refused every design before its
#: optimiser started.
#:
#: So the opening speed is the higher of the fraction and the speed the
#: family's PUBLISHED craft actually flies at, with 5 % on top (10 % in W/S,
#: since W/S goes as V²). It is a DEFAULT, not a limit: the field is there to
#: be typed into, and typing a slower one is how you ask for a bigger wing.
#: What it may not do is open on a requirement the mission cannot meet and
#: call that the mission's fault.
#:
#: THE PUBLISHED LOADING, NEVER THE TYPED ONE. The ceiling this speed sets
#: travels to the solver and centres the searched ``ws_pa`` band, and a
#: searched W/S has no interior optimum — the answer IS the top of that band
#: (results/ws_band_study.json). Derived from the loading in the form, the
#: ceiling would move with the area somebody happened to type and the answer
#: with it, which is the exact defect ``ws_band_default`` was written to
#: remove. ``mission_defaults`` is the one place that says what a family
#: opens on, so it is the one place this reads.
OPENING_SPEED_MARGIN = 1.05
#: ...and it must stay a take-off/stall speed: never at or above the cruise
#: speed, which would leave the diagram with no band at all (water_diagram
#: requires ``v_takeoff < v_max``) and take the whole card down to "these
#: numbers do not describe a mission". A craft that only just flies at its
#: cruise speed IS in conflict, and the conflict card is where that is said.
OPENING_SPEED_CEILING_FRAC = 0.95
#: the track has no diagram: a car's rear wing carries no weight, so it has
#: no wing loading to derive — its "loading" is a downforce target.
WS_DIAGRAM_MEDIA = ("air", "water")


def published_wing_loading(S: dict) -> float | None:
    """W/S of the family's OWN published design point [Pa], or ``None``.

    Not ``wing_loading`` — that is the number in the form, which the user
    moves. This is the one the form OPENED on (:func:`mission_defaults`), and
    it is a property of the family: it does not change when an area is typed.
    See :data:`OPENING_SPEED_MARGIN` for why that distinction is the whole
    point.
    """
    try:
        d = mission_defaults(S["wing"]["problem"], S["medium"],
                             S.get("water", "sea"))
        ws = float(d["W_N"]) / float(d["s_ref_m2"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None
    return ws if ws > 0.0 else None


def opening_speed(S: dict, frac: float, cl_max=None) -> float:
    """The stall / fly-up speed a fresh diagram card opens on [m/s].

    ``frac`` of the mission's cruise speed, but never below the speed the
    family's PUBLISHED loading actually flies at plus
    :data:`OPENING_SPEED_MARGIN`, and never above
    :data:`OPENING_SPEED_CEILING_FRAC` of cruise. See those two constants for
    why — in one line, a requirement that opens already broken is not a
    requirement, it is a bug report about the default.
    """
    v = float(S["mission"].get("V") or 0.0)
    fromfrac = frac * v
    v_min = stall_speed_at(S, published_wing_loading(S), cl_max=cl_max)
    if v_min is None or not v_min > 0.0:
        return fromfrac
    return min(max(fromfrac, OPENING_SPEED_MARGIN * float(v_min)),
               OPENING_SPEED_CEILING_FRAC * v)


def ws_inputs(S: dict) -> dict:
    """The diagram's inputs for this session's medium (created on first ask).

    The one speed the card opens with — the stall speed in air, the fly-up
    speed in water — is derived from the MISSION's speed rather than being a
    number out of the air: they are the same aircraft, and a card that
    opened at 18 m/s stall beside a 14.6 m/s cruise would describe no
    mission at all.
    """
    medium = S["medium"] if S["medium"] in WS_DIAGRAM_MEDIA else "air"
    store = S["mission"].setdefault("ws_diagram", {})
    if medium not in store:
        fresh = dict(WS_DIAGRAM_DEFAULTS[medium])
        v = float(S["mission"].get("V") or 0.0)
        key, frac = (("v_stall_ms", STALL_SPEED_FRAC) if medium == "air"
                     else ("v_takeoff_ms", FLYUP_SPEED_FRAC))
        if fresh.get(key) is None and v > 0.0:
            fresh[key] = round(opening_speed(S, frac, fresh.get("cl_max")), 2)
        store[medium] = fresh
    return store[medium]


def set_ws_input(S: dict, key: str, value) -> bool:
    """Type one diagram input. Clearing it drops the requirement."""
    inputs = ws_inputs(S)
    if key not in inputs:
        return False
    if value in (None, ""):
        inputs[key] = None
        return True
    try:
        inputs[key] = float(value)
    except (TypeError, ValueError):
        return False
    return True


def cavitation_polar(S: dict):
    """The section whose suction peak the cavitation line is read off.

    Stage 2's pick where there is one, and the WATER FAMILY's own section at
    the thickness the run will search where there is not
    (``api.water_family_polar``) — which is the section
    ``hydrofoil.HydrofoilProblem`` evaluates its cavitation constraint on, so
    an unanswered stage 2 draws the line the solver will use rather than a
    stand-in. ``None`` when neither can be had.
    """
    from aerobo import api

    chosen = section_flag_value(S)
    if chosen is not None:
        try:
            return api.section_polar_for(chosen)
        except (FileNotFoundError, TypeError, ValueError, KeyError):
            pass          # ...and fall through to the family's own section
    try:
        return api.water_family_polar(_mid(_searched_box(S), "tc"))
    except (FileNotFoundError, TypeError, ValueError, KeyError):
        return None


def cavitation_line(S: dict, m) -> tuple | None:
    """``(cp_min_a, cp_min_b)`` for this session's section, or ``None``.

    THE DIAGRAM WAS REFUSING THE DESIGN POINT ITS OWN SOLVER FLIES.
    ``WaterMission``'s placeholder pair (0.6, 2.2) is a 12 %-thick section's
    rough shape, and its own field comment has said "meant to be replaced"
    since it was written. On the published foiling family — 6 kN on 0.144 m²
    at 12 m/s, 0.575 m down, i.e. cl = 0.565 — it puts the cavitation ceiling
    at 27 493 Pa against a stated 41 667 Pa, while the real NACA 2412 section
    the solver evaluates that family on holds -Cp_min = 0.96 there against a
    cavitation number of 1.42 and does not cavitate at all (ceiling
    54 326 Pa, i.e. the placeholder was 34 % low).

    ``None`` where the section cannot answer — no Cp_min table, no polar —
    and the placeholder stands, said out loud rather than silently fitted.
    """
    from aerobo import constraint_diagram as cd

    pol = cavitation_polar(S)
    return None if pol is None else cd.cp_min_line(pol, m)


def ws_diagram(S: dict):
    """The constraint diagram this mission implies, or ``None``.

    ``None`` for a medium that has no wing loading to derive (the track) or
    for inputs that do not describe a mission — a diagram that cannot be
    built says so through the caller rather than by raising into a render.
    """
    from aerobo import constraint_diagram as cd

    if S["medium"] not in WS_DIAGRAM_MEDIA:
        return None
    inputs = {k: v for k, v in ws_inputs(S).items() if v is not None}
    point = design_point(S)
    # CL_max is the SECTION's, where stage 2 has chosen one: a stall line
    # drawn from a typed-in CL_max beside a screened aerofoil that measures
    # its own would be the tool disagreeing with itself.
    chosen = section_cl_max(S)
    if chosen is not None:
        inputs["cl_max"] = chosen[0]
    try:
        if S["medium"] == "water":
            from aerobo.hydrofoil import water_properties

            water = water_properties(S.get("water", "sea"))
            # the SPEED, the DEPTH and the water itself are the mission's;
            # cavitation is therefore judged at the point being designed for
            m = cd.WaterMission(
                v_max_ms=float(point["v_ms"]),
                depth_m=float(point["depth_m"] or 0.0),
                rho=float(water["rho"]), p_vap=float(water["p_vap"]),
                **inputs)
            # ...and the SUCTION PEAK is the section's, on the same rule
            # cl_max is (above): a cavitation line drawn from a placeholder
            # beside a section that measures its own is the tool disagreeing
            # with itself. It was, by 34 % — see cavitation_line.
            pair = cavitation_line(S, m)
            if pair is not None:
                m = replace(m, cp_min_a=pair[0], cp_min_b=pair[1])
            return cd.water_diagram(m)
        # the speed and the density are the mission's, the aspect ratio is
        # the wing's — each from the one place that owns it
        return cd.air_diagram(cd.AirMission(
            v_cruise_ms=float(point["v_ms"]),
            rho_cruise=float(point["rho"]),
            aspect_ratio=nominal_aspect_ratio(S), **inputs))
    except (ValueError, TypeError, KeyError):
        # KeyError is ``point`` with no speed in it: ``design_point`` answers
        # ``{"error": ...}`` for a mission it cannot solve, and reading
        # ``point["v_ms"]`` off that raised THROUGH the render. Recorded in
        # results/v3_render_errors.log twice on 2026-09-01 (a water session),
        # where it took stage 1's whole Operating point view down and
        # replaced it with a traceback panel. The contract this function
        # already states — "a diagram that cannot be built says so through
        # the caller rather than by raising into a render" — now covers the
        # case where the MISSION is the thing that is not built yet.
        return None


#: WHO OWNS THE WING LOADING'S CEILING. One question, two honest answers,
#: and until it was asked the shell held both at once: the diagram derived a
#: ceiling from the stall/landing fields and sent it to every sized run, while
#: the user typed a loading of their own into the field above it. A user who
#: has DETERMINED their W/S — from a sister aircraft, a rule set, a customer
#: spec, an analysis this tool does not do — is not making a mistake when the
#: default stall speed disagrees, and a card that keeps saying so is the tool
#: overruling them slowly.
#:
#: ``"mission"`` (the default) is what this shell has always done: the binding
#: ``ws_max`` line of the constraint diagram is a REQUIREMENT and travels to
#: the solver. ``"stated"`` makes the number in the W/S field the ceiling
#: instead. It is deliberately NOT "no ceiling at all": every sized family can
#: fly past whatever loading it is given, and with none at all the search buys
#: payload L/D with a wing that lands at any speed it likes (which is what the
#: ceiling was added for). It moves the limit to the user's number; it does
#: not delete the limit.
WS_CAP_MISSION = "mission"
WS_CAP_STATED = "stated"
WS_CAP_SOURCES = (WS_CAP_MISSION, WS_CAP_STATED)


def ws_cap_source(S: dict) -> str:
    """Which answer to :data:`WS_CAP_SOURCES` this session is using."""
    got = S["mission"].get("ws_cap_source")
    return got if got in WS_CAP_SOURCES else WS_CAP_MISSION


def set_ws_cap_source(S: dict, value) -> bool:
    """Choose who caps the wing loading. False on a value that is neither.

    RE-OPENS THE SEARCHED W/S ROW. On the mode that searches the loading, the
    ceiling is both the centre and the top of the band
    (:func:`ws_band_default`), so a band left at the old ceiling would be a
    box nobody set — the run would search up to a limit the user has just
    said is not theirs. Only when the row is still the one this shell wrote:
    a band the user typed is an answer of its own and is not overwritten.
    """
    if value not in WS_CAP_SOURCES:
        return False
    was_default = ws_band_default(S) if loading_is_searched(S) else None
    S["mission"]["ws_cap_source"] = value
    if was_default is None:
        return True
    row = S["wing"]["bounds"].get(WS_ROW)
    if row is not None and len(row) == 2 and all(
            abs(float(a) - float(b)) <= 1e-9 * max(1.0, abs(float(b)))
            for a, b in zip(row, was_default)):
        band = ws_band_default(S)
        if band is not None:
            S["wing"]["bounds"][WS_ROW] = [float(band[0]), float(band[1])]
    return True


def diagram_ws_ceiling(S: dict) -> float | None:
    """The highest wing loading this mission's REQUIREMENTS allow [Pa].

    The binding ``ws_max`` line of the mission's own constraint diagram —
    the stall/approach speed and the landing field length in air, fly-up and
    cavitation in water. It is a REQUIREMENT, not a preference, which is why
    it reaches the solver (``config.flags`` -> ``wing_loading_limit_pa``)
    rather than only the card that draws it: a sized search moves the
    loading, and a limit that only appears on a diagram is not a limit.

    ``None`` when the mission states nothing that bounds W/S — the whole
    diagram is drawn from numbers the user gave, so an unanswered stall
    speed leaves the search exactly as free as it was.

    THE MATCHING POINT WINS WHERE THERE IS ONE. When the mission states its
    available thrust, the diagram closes and the honest ceiling is the highest
    loading the engine can actually fly (``Diagram.ws_design_pa``), which is
    at or below the ``ws_max`` line. This is what stops a SEARCHED wing
    loading: measured over 128 runs, no objective in this package has an
    interior optimum in W/S — L/D and the composite both park at the top of
    whatever band they are given, because a smaller wing at the same weight
    genuinely is more efficient and nothing here prices the thrust it costs.
    An unclosed diagram leaves that band ending at an arbitrary multiple of
    the seed; a closed one ends it where the aircraft stops flying.

    AN INFEASIBLE DIAGRAM STILL HAS A LIMIT. ``ws_design_pa`` is None there,
    and rightly: no loading is both allowed and flyable, so there is nothing
    to RECOMMEND. But the ``ws_max`` line is a REQUIREMENT — the stall speed,
    the landing field, the fly-up — and an engine that cannot meet it does
    not repeal it. Reading the recommendation as the ceiling made stating a
    capability WIDEN the search: on the default air mission, typing a T/W of
    0.30 (against the 0.44 the climb demands) took the band from
    32.6–75.2 Pa to 32.6–130.6 Pa, i.e. answering one more question deleted
    the stall limit. So the ceiling falls back to the limit, and only the
    DESIGN POINT (``adopt_ws_recommendation``) stays None.

    This is the DERIVATION, always drawn. Whether it BINDS is
    :func:`mission_ws_ceiling`.
    """
    d = ws_diagram(S)
    if d is None:
        return None
    cap = d.ws_design_pa
    if cap is None:
        cap = d.ws_max_pa          # the requirement, not the recommendation
    try:
        cap = float(cap)
    except (TypeError, ValueError):
        return None
    return cap if cap > 0.0 else None


def mission_ws_ceiling(S: dict) -> float | None:
    """The wing loading ceiling that actually BINDS this session [Pa].

    :func:`diagram_ws_ceiling` on the default (:data:`WS_CAP_MISSION`), and
    the loading the user typed on :data:`WS_CAP_STATED` — a user who has
    determined their W/S elsewhere, whose stall/landing fields are therefore
    not the requirement that decides it.

    THE BRANCH IS HERE, in the one function every consumer already asks: the
    flag that reaches the solver (``config.cfg_dict`` ->
    ``wing_loading_limit_pa``), the band a searched W/S opens on
    (:func:`ws_band_default`), its cap (:func:`ws_band`), the emptiness proof
    (``api.size_box_conflicts``) and the card that compares the two
    (:func:`ws_over_ceiling`). Put anywhere else, switching the source would
    silence a warning while the solver went on refusing every design — the
    exact failure the warning exists for, made invisible.

    On ``"stated"`` it is the user's number rather than NO number, because
    every sized family can fly past whatever loading it is given: with no
    ceiling the search buys payload L/D with a wing that lands at any speed
    it likes, which is what the ceiling was added for. The limit moves; it
    does not disappear.
    """
    if ws_cap_source(S) == WS_CAP_STATED:
        stated = wing_loading(S)
        return float(stated) if stated and stated > 0.0 else None
    return diagram_ws_ceiling(S)


def start_from_design(S: dict, x) -> str | None:
    """Arm the next wing run to start from design ``x``. Returns an error.

    THE ANSWER TO "IT FOUND NOTHING". A constrained run that returns no design
    still evaluated designs, and the least-violating one is a real wing — it
    missed a limit, it is not noise. Handing that design back as the next
    run's FIRST evaluation is the one mechanism in this literature that makes
    a continuation monotone rather than another roll: the incumbent is a max
    over evaluated points, and a seed is never screened away, so a run told a
    design that flies returns one at least as good. Told a design that nearly
    flies, it starts its surrogate on the constraint boundary instead of on a
    flat prior — which is where the answer is.

    Refused rather than repaired when it does not fit the box the run will
    search: a clipped seed is a DIFFERENT design from the one that was
    measured, and starting from it while calling it "the closest design" would
    be a claim about a wing nobody evaluated. ``None`` means armed.
    """
    from aerobo import api

    from . import config

    try:
        row = [float(v) for v in (x or [])]
    except (TypeError, ValueError):
        return "that is not a design vector"
    if not row:
        return "that run kept no design vector to start from"
    W = S["wing"]
    before = W.get("x_seed")
    W["x_seed"] = row
    try:                      # the api's own validation, not a second copy
        api.check_x_seed(config.build_cfg(S))
    except Exception as exc:                      # noqa: BLE001
        W["x_seed"] = before
        return str(exc)
    return None


#: how many evaluations "keep going" adds when nothing else is said — half
#: the run's own budget, so the offer scales with the search rather than with
#: a number someone typed once.
def continue_extra_default(record: dict) -> int:
    """How many evaluations to offer adding to ``record``.

    A run STOPPED BY HAND is offered the rest of the budget it was launched
    with: the user already decided that number and then interrupted it, so
    "keep going" means finish what was started. A run that spent its whole
    budget is offered half of it again.
    """
    cfgd = (record or {}).get("config") or {}
    budget = int(cfgd.get("budget") or 0)
    spent = int((record or {}).get("n_evals") or 0)
    if record.get("partial") and budget > spent:
        return int(budget - spent)
    return max(1, budget // 2)


def run_convergence(S: dict, record: dict) -> dict:
    """Did this finished run flatten out? — the verdict, in the run's own rule.

    The question "expand the budget" exists to answer, and the one the shell
    never answered: a user looking at a score has no way to tell whether more
    evaluations would buy anything. The patience and the tolerance come from
    the same measured plan the search policy uses, so the verdict on the
    results page and the rule that would have stopped the run are the same
    arithmetic (:func:`aerobo.optimize.budget.converged_report`).
    """
    from aerobo.optimize.budget import converged_report

    plan = (effective_wing_search(S) or {}).get("plan")
    patience = None if plan is None else plan.patience
    tol = None if plan is None else plan.tol
    return converged_report((record or {}).get("history") or [],
                            patience, tol)


def continue_run(S: dict, record: dict, extra: int | None = None) -> dict:
    """Arm "give this finished run N more evaluations". Never launches.

    Returns ``{"cfg", "note", "drift", "error"}``. ``cfg`` is the record's own
    configuration at a bigger budget — NOT the session's current one, because
    the session may have moved since (a widened row, a different objective),
    and a longer run of a DIFFERENT search is not the thing the button says it
    is. ``drift`` names what has moved, so the card can say that taking this
    offer runs the old configuration rather than the one on screen.

    ``api.continue_run_config`` owns the honesty about whether the longer run
    actually contains the shorter one (``note["exact"]``).
    """
    from aerobo import api

    from . import config

    out = {"cfg": None, "note": None, "drift": [], "error": None}
    if not record or not (record.get("config") or {}).get("problem_name"):
        out["error"] = "there is no finished run to continue"
        return out
    extra = int(continue_extra_default(record) if extra is None else extra)
    if extra < 1:
        out["error"] = "a continuation must add at least one evaluation"
        return out
    # WHAT THE NUMBER ON THE BUTTON MEANS: evaluations that have not been
    # flown yet, exactly as on the section (:func:`continue_section`). A run
    # STOPPED at 12 of 40 has 28 unflown, so finishing it is a TOTAL of 40 —
    # not 68, which is what adding to the BUDGET instead of to what was SPENT
    # produces, and which hands back a bigger search than the one that was
    # interrupted while the offer says "the rest of the budget you asked for".
    cfgd = record.get("config") or {}
    old = int(cfgd.get("budget") or 0)
    spent = int(record.get("n_evals") or 0)
    total = spent + extra
    # CAN THIS RUN BE RESUMED? A BO record carries its own evaluations, and
    # they are a training set: the continuation inherits them and buys only
    # the new ones (``api.can_resume`` -> ``api.run(resume=…)``). Then the
    # arithmetic is against what was FLOWN — `total - spent` is the number of
    # designs this run will actually evaluate — where a re-flight has to add
    # to the BUDGET instead, because it repeats everything first.
    resumes = api.can_resume(record)
    try:
        if total <= old and not resumes:
            # RE-FLYING THE SAME CONFIGURATION IS THE CONTINUATION. The budget
            # does not move, so nothing about the search changes and the
            # containment question does not arise for ANY optimiser — not even
            # the ones a longer budget would re-shape (``ga``'s population,
            # BO's Sobol block).
            cfg = api.run_config_of(record)
            note = {"exact": True, "budget": old, "added": old - spent,
                    "was": spent,
                    "why": (f"the same search, finished: it flew {spent} of "
                            f"the {old} evaluations you asked for, and this "
                            f"re-flies those {spent} identically and reaches "
                            f"the {old - spent} it never got to")}
        else:
            cfg, note = api.continue_run_config(
                record, (total - spent) if resumes else (total - old))
            note = {**note, "was": spent, "added": total - spent}
    except (ValueError, KeyError, TypeError) as exc:
        out["error"] = str(exc)
        return out
    out["cfg"], out["note"] = cfg, note
    # what has moved under the record since it was made. Compared field by
    # field rather than by equality of the whole dict, so the card can NAME
    # the difference instead of saying "something changed".
    try:
        now = config.build_cfg(S).to_dict()
    except Exception:                                 # noqa: BLE001
        return out
    was = record.get("config") or {}
    for key in ("problem_name", "mission_kwargs", "flags", "optimiser",
                "bounds_overrides", "pinned"):
        if (was.get(key) or None) != (now.get(key) or None):
            out["drift"].append(key)
    return out


#: what a SECTION continuation compares, field by field, to say whether the
#: form still describes the run it is about to lengthen. The value is the name
#: the user knows the field by, because "score_weights changed" names a
#: variable and "the weights changed" names a decision.
SECTION_SHAPE_NAMES = {
    "re": "the Reynolds number", "mach": "the Mach number",
    "cl_design": "the design lift coefficient",
    "tc_min": "the minimum thickness", "cm_max": "the pitching-moment cap",
    "anchor": "the seed section", "twist_order": "the twist law",
    "twist_max_deg": "the twist cap", "alpha_max_deg": "the incidence cap",
    "chord_order": "the chord law", "chord_max_frac": "the chord cap",
    "objective": "the objective", "score_weights": "the weights",
    "score_reference": "the band the score is measured on",
    "censored": "the censoring policy", "wing": "the wing the section flies on",
}


def continue_section(record: dict, launch: dict,
                     shape_now: dict | None = None,
                     extra: int | None = None) -> dict:
    """Arm "give this finished SECTION search more evaluations". Never runs.

    The stage-2 twin of :func:`continue_run`, and the same statement: one
    longer run, not a handoff. ``launch`` is the snapshot the worker kept of
    the arguments it actually flew — the search is re-flown from THAT, never
    from the form, because the form may have moved since and a longer run of
    a different search is not what the button says. Where they differ,
    ``drift`` names the difference in the user's words (:data:`SECTION_SHAPE_NAMES`).

    A continuation costs less here than the budget suggests: the evaluations
    already paid for come back out of the on-disk XFOIL polar cache
    (``xfoil_run.cache_key``), so what is bought is the NEW evaluations. What
    it cannot buy is containment for every optimiser —
    :func:`api.continue_run_config` owns that honesty (``note["exact"]``), and
    it is the same arithmetic the wing's "Keep going" is judged by.

    Returns ``{"cont", "note", "extra", "drift", "error"}``. ``cont`` is what
    the stage hands back to its worker: the launch snapshot at the new budget.
    """
    from aerobo import api

    out = {"cont": None, "note": None, "extra": None, "drift": [],
           "error": None}
    res = (record or {}).get("result") or {}
    cfgd = (record or {}).get("config") or {}
    if not cfgd.get("problem_name"):
        out["error"] = "there is no finished section search to continue"
        return out
    if not (launch or {}).get("shape"):
        # a run whose snapshot is gone (the shell was restarted under it) can
        # still be READ; it just cannot be re-flown from itself, and saying so
        # is better than re-flying the form and calling it the same search.
        out["error"] = ("this run was not launched by this shell, so the "
                        "search it flew cannot be repeated exactly — "
                        "optimise again to start a run that can be continued")
        return out
    if int(launch.get("n_restarts", 1) or 1) > 1:
        out["error"] = ("this run was several independent searches, and "
                        "continuing one of them is not the same search")
        return out
    # WHAT THE NUMBER ON THE BUTTON MEANS: evaluations that have not been
    # flown yet. A run stopped by hand flew `spent` of the `old` its user
    # asked for, so finishing it is `old - spent` more and a TOTAL of `old` —
    # not `old + (old - spent)`, which is the arithmetic that comes out of
    # adding to a budget instead of to what was spent, and which hands back a
    # bigger search than the one that was interrupted.
    old = int(cfgd.get("budget") or 0)
    spent = int(res.get("n_evals") or 0)
    if extra is None:
        extra = int(continue_extra_default(
            {"config": cfgd, "n_evals": spent,
             "partial": bool(res.get("partial"))}))
    extra = int(extra)
    if extra < 1:
        out["error"] = "a continuation must add at least one evaluation"
        return out
    out["extra"] = extra
    total = spent + extra
    # the same question the wing's continuation asks (:func:`continue_run`):
    # a BO section search carries its evaluations, so the continuation
    # inherits them instead of re-flying them through XFOIL
    rec_for_resume = {"config": cfgd, "eval_x": res.get("eval_x"),
                      "eval_y": res.get("eval_y"), "eval_g": res.get("eval_g"),
                      "is_constrained": res.get("is_constrained")}
    resume = (api.resume_payload(rec_for_resume)
              if api.can_resume(rec_for_resume) else None)
    if total <= old and resume is None:
        # RE-FLYING THE SAME CONFIGURATION IS THE CONTINUATION. Nothing about
        # the search changes, so the containment question does not arise for
        # any optimiser — including the ones a longer budget would change
        # (``api.CONTINUABLE_OPTIMISERS``) — and the note can say so plainly.
        pinned_init = None          # the budget does not move; nothing to pin
        note = {"exact": True, "budget": old, "added": old - spent,
                "was": spent,
                "why": (f"the same search, finished: the {spent} evaluations "
                        f"it flew come back out of the XFOIL cache and the "
                        f"{old - spent} it never reached follow them")}
    elif resume is not None:
        # NOTHING IS RE-FLOWN. The evaluations already bought are handed to
        # the new search as its training set, so a section continued by 8
        # spends 8 XFOIL polars — not its whole prefix again, which on this
        # stage is the expensive half of the run.
        pinned_init = None                  # no initial design is drawn
        note = {"exact": True, "budget": total, "added": extra, "was": spent,
                "resume": resume, "resumed": spent,
                "why": (f"a resume, not a re-run: the {spent} evaluations "
                        f"already paid for are this search's training set, "
                        f"so it starts at {spent} and buys {extra} NEW "
                        f"sections")}
    else:
        try:
            # THE SPLIT THE RUN ACTUALLY FLEW travels with the record.
            # ``continue_run_config`` pins BO's Sobol block back to it when a
            # small budget clamped it, and it can only do that if it is told
            # what was flown — passing the dimension alone left it re-deriving
            # the split, and a re-derivation is the thing a continuation must
            # not depend on.
            cont_cfg, note = api.continue_run_config(
                {"config": cfgd, "dim": res.get("dim"),
                 "searched_dim": res.get("searched_dim"),
                 "bo_split": res.get("bo_split")}, total - old)
        except (ValueError, KeyError, TypeError) as exc:
            out["error"] = str(exc)
            return out
        note = {**note, "was": spent, "added": total - spent}
        pinned_init = (cont_cfg.flags or {}).get(api.BO_N_INIT_FLAG)
        if str((launch.get("shape") or {}).get("objective") or "") == "pareto":
            # A FRONT'S CONTAINMENT IS UNMEASURED. `continue_run_config`
            # judges by optimiser name, and a front run's name is the scalar
            # optimiser it was launched under — but what it actually flies is
            # a batch acquisition over the whole Pareto set, which nothing in
            # this repo has re-flown at two budgets. Unproven is not refused;
            # it is simply not promised.
            note = {**note, "exact": False,
                    "why": (note["why"] + " — except that this is a FRONT run, "
                            "whose batches have not been measured at two "
                            "budgets, so the evaluations already paid for are "
                            "not promised back")}
    out["note"] = note
    # ...and the pinned Sobol block, where the api set one. The stage sends
    # this straight to ``api.optimize_airfoil(n_init=…)``, so without it the
    # section's continuation drew the DEFAULT seed size for its new, bigger
    # budget and re-flew none of what was paid for — the wing's bug, one
    # stage over.
    search = {**dict(launch.get("search") or {}), "budget": int(note["budget"])}
    if pinned_init is not None:
        search["n_init"] = int(pinned_init)
    # ...and the evaluations the new run INHERITS, straight to
    # ``api.optimize_airfoil(resume=…)``
    search["resume"] = note.get("resume")
    out["cont"] = {"shape": dict(launch["shape"]),
                   "seed": int(launch.get("seed") or 0),
                   "search": search,
                   # WHAT THE STAGE SHOWS WHILE IT RUNS. The re-flown prefix
                   # travels with the snapshot so the progress chip can name
                   # it — a counter that goes back to 1 and races through 12
                   # evaluations reads as a restart, which is the one thing
                   # about this button a user takes for a fault. Zero where
                   # the api judged the longer run uncontained: those
                   # evaluations are a different set of designs, and calling
                   # them a re-flight would be false.
                   # ZERO on a RESUME: nothing is re-flown at all, so
                   # there is no prefix to name. What that run shows instead
                   # is where its counter starts (``resumed``).
                   "was": (0 if note.get("resume")
                           else (int(note["was"]) if note.get("exact") else 0)),
                   "resumed": int(note.get("resumed") or 0),
                   "exact": bool(note.get("exact"))}
    own = _own_section_anchor(record)
    for key, was in (launch.get("shape") or {}).items():
        if shape_now is None:
            break
        now = shape_now.get(key)
        if (now or None) == (was or None):
            continue
        if key == "anchor" and own is not None and _same_anchor(now, own):
            # NOT DRIFT — THE RUN DID THIS. A finished search adopts its own
            # winner as the surface's section, so the form's anchor differs
            # from the launch BECAUSE of the run being continued. Reporting
            # that would make every continuation warn about itself, and the
            # cure it points at (optimise again, from the winner) is the warm
            # start this repo measured as the worst arm for BO.
            continue
        out["drift"].append(SECTION_SHAPE_NAMES.get(key, key))
    return out


def _own_section_anchor(record: dict):
    """``(w_upper, w_lower)`` of the section THIS report produced, or None."""
    design = ((record or {}).get("section") or {}).get("design") or {}
    w_u, w_l = design.get("w_upper"), design.get("w_lower")
    if not w_u or not w_l:
        return None
    return [float(v) for v in w_u], [float(v) for v in w_l]


def _same_anchor(a, b) -> bool:
    """Two CST anchors, compared as numbers rather than as containers."""
    try:
        au, al = a
        bu, bl = b
        return ([float(v) for v in au] == [float(v) for v in bu]
                and [float(v) for v in al] == [float(v) for v in bl])
    except (TypeError, ValueError):
        return False


def clear_start_design(S: dict) -> None:
    """Forget the design the next run would have started from."""
    S["wing"]["x_seed"] = None


def start_design(S: dict):
    """The design the next wing run starts from, or ``None``."""
    return S["wing"].get("x_seed")


def ws_over_ceiling(S: dict) -> dict | None:
    """The wing loading THIS SESSION STATES against the one it allows.

    Two answers to one question can coexist in this form and only one of them
    reaches the solver as a refusal. The user types a wing loading (or an
    area, which is the same number); the constraint diagram derives a CEILING
    from a different set of fields — the stall or approach speed, the landing
    field, the fly-up — and that ceiling travels to every sized run as
    ``wing_loading_limit_pa``. Leave the stall requirement at its default and
    raise the weight, and the two silently disagree: the reported session
    stated 7000 N against a 75.2 Pa ceiling drawn from an untouched ~9.4 m/s
    stall speed, so every design the search could reach was refused before its
    solver and the run reported "no solution was found".

    Returns ``None`` when they agree (or when there is no ceiling), else
    ``{"stated", "cap", "binding", "over"}`` — the loading stated, the ceiling,
    the requirement that sets it, and the ratio. This is a DISAGREEMENT, never
    a refusal: which of the two numbers is wrong is the user's call, and the
    card offers both ways out.
    """
    stated = wing_loading(S)
    cap = mission_ws_ceiling(S)
    if not stated or not cap or stated <= cap:
        return None
    d = ws_diagram(S)
    binding = ""
    if d is not None:
        try:
            binding = str(d.recommend().get("binding_constraint") or "")
        except Exception:                       # a card, never fatal
            binding = ""
    return {"stated": float(stated), "cap": float(cap), "binding": binding,
            "over": float(stated) / float(cap)}


#: the SEARCHED wing loading's design-box row (sizing.WS_LABEL), and the
#: band a shell opens it on: the same fractional band the physics would use
#: (sizing.WS_FRAC_BOUNDS), written down rather than left implicit for the
#: reason the span band is — an implicit band moves whenever the mission's
#: W/S moves, so the constraint the user set would drift under them.
WS_ROW = "ws_pa"


def ws_band(S: dict) -> tuple | None:
    """``(min, max)`` wing loading the RUN searches [Pa], or ``None``.

    The band as the optimiser will get it — the design-box row, with every
    pin and release ``config.effective_bounds`` applies — so a card quoting
    it cannot drift from the box (:func:`span_box`'s twin, same reason).
    """
    from . import config

    got = config.effective_bounds(S).get(WS_ROW)
    if got is None:
        return None
    return float(got[0][0]), float(got[0][1])


def ws_band_default(S: dict) -> tuple | None:
    """``(min, max)`` the searched-W/S row opens on [Pa], or ``None``.

    Around the mission's own DESIGN POINT where the diagram gives one, and
    around the loading the aircraft already flies where it does not.

    WHICH CENTRE is the whole question, and it is not a presentation detail.
    A searched W/S has no interior optimum in any objective in this package
    (128 runs, ``results/ws_band_study.json``): the answer is the top of the
    band, whatever the band is. So the band does not merely FRAME the answer,
    it IS the answer, and a band centred on the seed makes the result a
    function of the area the user happened to type — double the seed and the
    search returns a different aeroplane for the same mission. Centred on the
    ceiling instead, the answer is the mission's own matching point, which is
    a number the mission determines and the seed does not.

    Two things keep it a band and not a point:

    * it reaches DOWN by ``sizing.WS_FRAC_BOUNDS[0]``, so the search still
      has somewhere to go if a criterion the objective prices does prefer a
      bigger wing (the composite's stall and buildability terms can);
    * it is WIDENED to contain the loading flown today, because the mode
      exists to re-open that question and a band that cannot return to the
      current answer is a mode that cannot be cancelled.

    The one case where the seed is deliberately left OUTSIDE is a seed above
    the ceiling: that aircraft does not meet its own mission, and a band
    stretched up to hold it would be the shell agreeing with it.
    """
    from aerobo.sizing import WS_FRAC_BOUNDS

    ws = wing_loading(S)
    if not ws:
        return None
    ws = float(ws)
    cap = mission_ws_ceiling(S)
    centre = ws if cap is None else float(cap)
    lo, hi = WS_FRAC_BOUNDS[0] * centre, WS_FRAC_BOUNDS[1] * centre
    if cap is not None:
        hi = min(hi, float(cap))
        lo = min(lo, ws)              # ...but never past the ceiling
        if not lo < hi:
            lo = WS_FRAC_BOUNDS[0] * hi
    return (lo, hi)


def adopt_ws_recommendation(S: dict) -> float | None:
    """Write the diagram's wing loading into the mission. Returns what it
    wrote, or None if there was nothing to write.

    The diagram's own recommendation (``ws_design_pa``), which is the matching
    point when the mission states its thrust and the ``ws_max`` limit
    otherwise — never the limit when a matching point exists, because
    adopting a limit as a design point is what put the seed on the upper wall
    of its own searched band.
    """
    d = ws_diagram(S)
    if d is None:
        return None
    ws = d.ws_design_pa
    if not ws:
        return None
    if not set_wing_loading(S, ws):
        return None
    return float(ws)


def set_wing_loading(S: dict, value) -> bool:
    """Set the area from a wing loading. False if the value is unusable."""
    try:
        ws = float(value)
    except (TypeError, ValueError):
        return False
    if not ws > 0.0:
        return False
    S["mission"]["s_ref_m2"] = float(S["mission"]["W_N"]) / ws
    return True


# ------------------------------------------- how the SIZE is stated (three)
#: ONE NUMBER, THREE FACES. The session stores the reference AREA and nothing
#: else about the size: :func:`wing_loading` derives W/S from it and the
#: design load, and :func:`nominal_span` derives the span of an unconstrained
#: wing from it and stage 2's aspect-ratio estimate. This says which of those
#: faces the mission card puts a cursor in — a PRESENTATION choice over one
#: stored quantity, not a fourth number that can disagree with the other
#: three. Switching modes stores nothing about the size, so a session that
#: switches back and forth is bit-for-bit the one that never switched.
#:
#: ``span_ar`` is the one that needs a word, because this shell spent two
#: sessions taking exactly those two questions OUT of the form. It does not
#: put them back:
#:
#: * it does NOT give the mission a span. The span is stage 3's constraint
#:   and is asked there, once (:func:`span_box`, :func:`set_span_m`). Nothing
#:   here writes one. What the field offers is the area entered the way a
#:   wing is first sketched — "about 3 m across, about AR 8" — and √(AR·S)
#:   read back out, which is the same arithmetic :func:`nominal_span` already
#:   does. Where stage 3 has been given a span of its own and it differs, the
#:   card says so (:func:`mission_span_disagrees`) instead of showing a
#:   number the field cannot set.
#: * it does NOT create a second aspect ratio. The AR it takes and shows is
#:   stage 2's existing estimate (:func:`section_aspect_ratio`), written
#:   through the same setter stage 2 writes it with.
SIZE_AS_LOADING = "loading"
SIZE_AS_AREA = "area"
SIZE_AS_SPAN_AR = "span_ar"
SIZE_STATEMENTS = (SIZE_AS_LOADING, SIZE_AS_AREA, SIZE_AS_SPAN_AR)

#: mode -> the label on its button in the mission card
SIZE_STATEMENT_LABELS = {SIZE_AS_LOADING: "W/S",
                         SIZE_AS_AREA: "area S",
                         SIZE_AS_SPAN_AR: "span x AR"}


def size_statement(S: dict) -> str:
    """Which of :data:`SIZE_STATEMENTS` the mission card is asking in."""
    got = S["mission"].get("size_stated_as")
    return got if got in SIZE_STATEMENTS else SIZE_AS_LOADING


def set_size_statement(S: dict, value) -> bool:
    """Choose how the size is stated. False on a value that is none of them."""
    if value not in SIZE_STATEMENTS:
        return False
    S["mission"]["size_stated_as"] = value
    return True


def reference_area(S: dict) -> float | None:
    """The reference area S [m²] — the one stored size number."""
    try:
        area = float(S["mission"]["s_ref_m2"])
    except (TypeError, ValueError):
        return None
    return area if area > 0.0 else None


def set_reference_area(S: dict, value) -> bool:
    """Set the area directly [m²]. False (and nothing stored) if unusable.

    "Unusable" is the PAIR, not the number: with a span already chosen at
    stage 3 the area is the other half of an aspect ratio, and 3 m² under a
    typed 12 m span is AR 48 — stored silently before this guard, warned
    about nowhere (plain span mode draws no aspect-ratio read-out at all),
    and refused by the solver only once the run reached the worker.
    """
    try:
        area = float(value)
    except (TypeError, ValueError):
        return False
    if not area > 0.0:
        return False
    # probed against the span the wing would then fly, which is why the value
    # goes in first: with no span chosen, ``nominal_span`` is √(AR·S) and
    # follows the area, so an untouched session can never refuse itself
    prev = S["mission"]["s_ref_m2"]
    S["mission"]["s_ref_m2"] = area
    if planform_size_refusal(S) is not None:
        S["mission"]["s_ref_m2"] = prev
        return False
    return True


def mission_span(S: dict) -> float | None:
    """The span the mission's own area and AR estimate imply, √(AR·S) [m].

    A VIEW of the stored area, deliberately NOT :func:`nominal_span`: that
    one answers "what span does this run fly" and prefers a span chosen on
    stage 3. This one answers "how wide is the area I am looking at", which
    is the only one of the two the mission card can set.
    """
    area = reference_area(S)
    if area is None:
        return None
    ar = nominal_aspect_ratio(S)
    return (ar * area) ** 0.5 if ar > 0.0 else None


def mission_span_disagrees(S: dict) -> tuple | None:
    """``(mission span, span stage 3 flies)`` in metres when they differ,
    else ``None``.

    Not an error: stage 3 owning the span is the design. It is reported so
    the mission's span field is never read as the wing's span.
    """
    mine = mission_span(S)
    theirs = chosen_span(S)
    if mine is None or theirs is None:
        return None
    if abs(mine - theirs) <= 1e-9 * max(1.0, abs(theirs)):
        return None
    return (float(mine), float(theirs))


def set_size_from_span_ar(S: dict, span=None, ar=None) -> bool:
    """Set the area as b²/AR, from whichever of the two the user typed.

    Typing a SPAN holds the aspect ratio and moves the area; typing an
    ASPECT RATIO holds the span and moves the area. Either way exactly one
    number is stored, and it is the area — plus the aspect ratio itself when
    one was typed, which goes to :func:`set_section_aspect_ratio` because
    that is the session's only aspect-ratio estimate and there must not be a
    second.

    False (and nothing stored) if the pair does not describe a wing.
    """
    if span is None and ar is None:
        return False
    # the span is read BEFORE any aspect ratio is written, so typing an AR
    # holds the span that is on screen rather than the one the new AR would
    # have implied against the old area
    b = mission_span(S) if span is None else span
    a = nominal_aspect_ratio(S) if ar is None else ar
    try:
        b, a = float(b), float(a)
    except (TypeError, ValueError):
        return False
    if not (b > 0.0 and a > 0.0):
        return False
    if ar is not None and not set_section_aspect_ratio(S, a):
        return False
    S["mission"]["s_ref_m2"] = b * b / a
    return True


def wing_guess(S: dict, surface: str | None = None) -> dict:
    """WingGuess kwargs for the section+twist (wing-mode) airfoil design.

    The wing it flies the candidate sections on is the ESTIMATE's wing —
    which is what that estimate is for. Air only, and the caller must check
    :func:`wing_objective` first: ``section_wing.WingGuess`` reads the ISA
    at an altitude, so handing it a water mission would design the foil
    against an air atmosphere.

    A SECOND lifting surface flies its own: a tandem pair's rear wing carries
    its share of the reference area on the same span, so scoring its section
    on the whole pair's wing would design it for twice its own chord — the
    same error :func:`surface_geometry` exists to stop in the screen.
    """
    from aerobo.mission import G0

    m = S["mission"]
    out = {"mass_kg": float(m["W_N"]) / G0, "v_ms": float(m["V"]),
           "altitude_m": float(m["altitude_m"]),
           "s_ref_m2": float(m["s_ref_m2"]),
           "aspect_ratio": section_aspect_ratio(S),
           "taper": float(m["taper"])}
    geo = surface_geometry(S, target_surface(S) if surface is None
                           else surface)
    if geo is not None and geo["area"] > 0.0 and geo["span"] > 0.0:
        out["s_ref_m2"] = float(geo["area"])
        out["aspect_ratio"] = float(geo["span"]) ** 2 / float(geo["area"])
    return out


def wing_objective(S: dict, surface: str | None = None) -> bool:
    """Is the shape optimiser scoring WING L/D rather than 2-D L/D?

    Not a user choice: the MEDIUM decides it, and so does the SURFACE. Wing
    mode flies the candidate on a lifting line at the ISA state of the
    mission altitude, so it is air-only — handing it a water mission would
    design the foil against an air atmosphere. Water and track get the 2-D
    objective, and their own solver carries the 3-D physics
    (:data:`TWO_D_NOTE`).

    A TRIMMING surface is excluded on its own account: a tail's job is a
    pitching moment at whatever lift trims the aircraft, so scoring its
    section by the L/D of a wing carrying the design weight would be
    optimising it for something it does not do (:data:`TRIM_CL_NOTE`). A
    tandem pair's rear wing DOES carry lift, so it keeps wing mode.

    An AIRFOIL-ONLY session is 2-D by construction: it states a flow and a
    chord and never a wing, so there is no lifting line to fly a candidate
    on — and :func:`wing_guess` would answer with a mission this mode never
    asked for.
    """
    if airfoil_only(S) or S["medium"] != "air":
        return False
    geo = surface_geometry(S, target_surface(S) if surface is None
                           else surface)
    return geo is None or bool(geo.get("lifting"))


TRIM_CL_NOTE = (
    "This surface TRIMS the aircraft: what it carries is whatever balances "
    "the pitching moment, never the design weight. That load is not zero and "
    "it is not a guess — the family's own two trim equations (moment about "
    "the CG = 0, total CL = the target) fix it in closed form, and this "
    "surface is screened at THAT lift. It is scored on its own 2-D L/D all "
    "the same: the wing-L/D objective would be designing it for a job it "
    "does not do.")


#: ...and the same statement for a session that has no vehicle at all. The
#: reason matters: the medium note below blames the MEDIUM, which reads as
#: false on an air session designing a section in air.
SECTION_ONLY_NOTE = (
    "This session designs a SECTION and nothing else, so the score is the "
    "section's own 2-D L/D. The wing-L/D objective flies each candidate on a "
    "lifting line sized from a mission, and this mode states no vehicle to "
    "size one from — switch stage 1 to the vehicle pipeline to score a "
    "section by the wing it will fly on.")


TWO_D_NOTE = (
    "This medium optimises the section on its own 2-D L/D: the wing-L/D "
    "objective flies a lifting line at an ISA altitude, which only the air "
    "mission has. The water and track missions screen and optimise the "
    "section at their own Reynolds number instead, and the medium's own "
    "solver carries the 3-D physics.")


def library_point() -> dict | None:
    """The (Re, Mach) the screening cache covers, or None if unbuilt."""
    from aerobo import api

    return api.screen_library_point()


LIBRARY_RE_NOTE = (
    "The library's polars are cached at ONE Reynolds number, and every "
    "lift-dependent metric can be re-derived exactly at any design Cl from "
    "them — so the cached point answers instantly and honours the design lift "
    "exactly, but it ranks (and flies) sections at ITS Reynolds number, not "
    "this surface's. Screening at this surface's own Re — the DEFAULT — runs "
    "a real XFOIL sweep: the whole database would take hours, so the "
    "shortlist below is "
    "re-screened there instead — the leaders of the cached ranking, swept at "
    "the point this surface actually flies. Whichever is chosen, the section "
    "is FLOWN in stage 3 at the point it was screened at.")


#: how many of the cached ranking's leaders are re-screened at the surface's
#: own Reynolds number. A live viscous sweep each (cached after the first
#: visit), so this is the knob that trades minutes for coverage; the whole
#: database at one new point is hours, which is what made the honest option
#: unusable before it was a shortlist.
SHORTLIST_DEFAULT = 24


#: WHERE A SECTION IS RANKED, BY DEFAULT: this surface's own Reynolds number.
#:
#: The cached library point is instant and exact in the design Cl, but it ranks
#: — and, because a library polar is measured data at ONE Reynolds number,
#: FLIES — every section at the point the cache was built at (Re 1e6). A wing
#: whose own MAC Reynolds number is somewhere else was therefore handed a
#: section chosen on the wrong polars by default: hg40 is L/D 80.6 at cl 0.5
#: at Re 1e6 and 42.2 at Re 3e5, and the reordering that follows is not a
#: rounding effect. The honest point is the one the surface actually flies, so
#: that is what the shell opens on; the cached point stays one click away
#: (``re_source = "library"``) for a quick look or a machine with no XFOIL.
#:
#: The price is real and is stated wherever it is paid: the first screen at a
#: new point sweeps :data:`SHORTLIST_DEFAULT` sections for real (minutes),
#: instant on every later visit because the sweeps cache like any other polar.
RE_SOURCE_DEFAULT = "mission"


#: fallback design lift for a trimming surface whose family will not say
#: what it trims at (:func:`api.trim_surface_cl` returns None). Zero is the
#: only honest stand-in there — it states no load rather than inventing one
#: — but it is a LAST RESORT: screening at zero lift makes "L/D at the
#: design Cl" identically 0 for every candidate, so the ranking collapses
#: onto |Cm| and picks a symmetric section by default rather than by
#: argument. Every family the registry carries answers, so this is unused
#: in practice and stays for the next family that does not.
TRIM_SURFACE_CL = 0.0


def _mid(box: dict, label: str) -> float | None:
    row = box.get(label)
    return None if row is None else 0.5 * (float(row[0]) + float(row[1]))


def _searched_box(S: dict) -> dict:
    """``{label: [lo, hi]}`` — the box the optimiser will ACTUALLY search.

    The family's published box with this session's own narrowing already in
    it: a row typed in the design box, or one pinned by the section chosen in
    stage 2. There is exactly ONE definition of that (``config.bounds_overrides``
    is what the run is built with, and ``config.effective_bounds`` is what the
    design-box view colours), and this is it — read here rather than restated,
    because two derivations of "the box" drift.

    Why it matters beyond the view: everything :func:`surface_geometry` and
    :func:`taper_box` quote is a MID-BOX number, and their docstrings say so
    ("it moves when the box does"). Reading ``default_bounds`` instead left a
    tail whose ``S_t_m2`` had been narrowed to [0.4, 0.6] being screened at
    the published 1.75 m² — a Reynolds number 1.87x the one it would fly —
    under a hint asserting that the wrong number was mid-box.

    A FIXED row (``config.fixed_rows``) collapses to its value here, which is
    the same statement one step further: the variable is not searched at all,
    so the mid-box number every readout downstream quotes is the number the
    user typed rather than the middle of a band nothing will draw from.
    """
    from . import config

    fixed = config.fixed_rows(S)
    return {k: ([fixed[k], fixed[k]] if k in fixed
                else [float(row[0]), float(row[1])])
            for k, (row, _src) in config.effective_bounds(S).items()}


#: memo for :func:`pitch_stability` — one solve per (problem, flags, box).
#: The solve itself is ~2 ms on every family that has a CG, but the type view
#: is redrawn on every choice, and a readout may not cost a solve per redraw.
_STABILITY_CACHE: dict = {}

#: ...and WHY there is no margin, keyed the same way. A family with no CG at
#: all is not in here — that is "nothing to report" and the card says
#: nothing. What is in here is a family that HAS a CG whose mid-box design
#: does not evaluate, which since the second surface became permanently
#: down-mounted (:data:`gui.v3.config.TAIL_MOUNT`) is reachable from the CG
#: field itself: an inverted section asked to lift runs the trim incidence
#: off the end of its polar. A read-out that just vanished there would be
#: the card going quiet at exactly the number that silenced it.
_STABILITY_WHY: dict = {}


def _stability_key(name, flags, over, mission, fixed) -> tuple:
    return (name, repr(sorted(flags.items(), key=lambda kv: kv[0])),
            repr(over), repr(sorted(mission.items(), key=lambda kv: kv[0])),
            repr(sorted(fixed.items())))


def stability_unevaluable(S: dict) -> str | None:
    """WHY this family has no margin to report, in the solver's own words.

    ``None`` when there is a margin, and ``None`` on a family that carries no
    CG at all — that one is not a failure, it is a question the family does
    not have. What this catches is the third case: a family with a CG whose
    mid-box design does not evaluate, which the card has to SAY rather than
    quietly drawing nothing where three numbers were.
    """
    from aerobo import api

    from . import config

    name = S["wing"]["problem"]
    if api.TAIL_CG_KEY not in api.PROBLEM_SPECS[name].flags:
        return None
    if pitch_stability(S) is not None:
        return None
    key = _stability_key(name, config.flags(S), config.bounds_overrides(S),
                         config.mission_kwargs(S), config.fixed_rows(S))
    return _STABILITY_WHY.get(key) or "the mid-box design does not evaluate"


def pitch_stability(S: dict) -> dict | None:
    """Where the NEUTRAL POINT is, and whether the CG is in front of it.

    ``None`` on a family that models no pitch balance at all — the wing-alone
    and tandem problems carry no CG, no Cm equation and no static-margin
    constraint, so there is nothing here to report and the shell says that
    rather than inventing a margin.

    The criterion is ``x_cg < x_np``, NOT "the CG is ahead of the wing's
    aerodynamic centre". A surface behind the wing drags the whole aircraft's
    neutral point aft of the wing AC (``x_np = (x_w a_w + x_t a_t)/(a_w +
    a_t)`` on the COUPLED slopes, tail.py), which is exactly why a
    conventional aft-tail aircraft is loaded with its CG behind the wing AC
    and is still stable. Where the surface is in FRONT — the canard — the
    neutral point moves ahead of the wing AC instead, and the family's
    calibrated CG is ahead of it too (``tail.X_CG_BY_TYPE["canard"]`` =
    −0.40 m). The one honest answer is therefore the margin, not the sign of
    ``x_cg``: SM = (x_np − x_cg)/mac, stable while it is positive and
    accepted by the run while it clears ``SM_min``.

    Quoted at the MID-BOX design — the same convention every other derived
    number in this shell follows — through the same flags and bound
    overrides the run is built with, so this cannot drift from what is flown.
    """
    import numpy as np

    from aerobo import api

    from . import config

    W = S["wing"]
    name = W["problem"]
    if api.TAIL_CG_KEY not in api.PROBLEM_SPECS[name].flags:
        return None
    flags = config.flags(S)
    over = config.bounds_overrides(S)
    mission = config.mission_kwargs(S)
    fixed = config.fixed_rows(S)
    key = _stability_key(name, flags, over, mission, fixed)
    if key in _STABILITY_CACHE:
        return _STABILITY_CACHE[key]
    try:
        built = api.PROBLEM_SPECS[name].build(mission, flags, over)
        # the centre of the box, with every FIXED row at its own value: the
        # margin quoted here is the margin of a design this session could
        # actually return, and the centre of a row nobody searches is not one
        x = np.asarray(built.bounds.mean(axis=1), dtype=float)
        labels = list(built.param_labels)
        for lbl, val in fixed.items():
            if lbl in labels:
                x[labels.index(lbl)] = float(val)
        out = built.evaluate(x)
    except Exception as exc:                # noqa: BLE001 — a read-out
        _STABILITY_CACHE[key] = None
        _STABILITY_WHY[key] = str(exc) or exc.__class__.__name__
        return None
    if not isinstance(out, dict) or out.get("SM") is None:
        _STABILITY_CACHE[key] = None
        # the SOLVER's own words for it, never a guess restated here
        _STABILITY_WHY[key] = str((out or {}).get("reason") or "") \
            if isinstance(out, dict) else ""
        return None
    row = {"x_cg": float(out["x_cg"]), "x_np": float(out["x_np"]),
           "SM": float(out["SM"]), "SM_min": float(out.get("SM_min", 0.0)),
           "mac": float(out.get("mac") or 0.0),
           "stable": float(out["SM"]) > 0.0,
           "accepted": float(out["SM"]) >= float(out.get("SM_min", 0.0))}
    _STABILITY_CACHE[key] = row
    return row


def _fin_geometry(S: dict) -> dict | None:
    """The vertical stabiliser's own size, or None where there is not one."""
    from aerobo import api, fin as _fin

    ch = S["wing"]["choices"]
    # ONE PREDICATE OWNS "IS THERE A FIN". This asked three of
    # :func:`fin_surface`'s four questions and not the fourth — whether the
    # FAMILY has a fin at all — so a plain wing, which carries none and gets
    # no stage 2.7, still had a 0.727 m2 fin quoted for it here. The size of
    # a surface is not a different question from whether it exists.
    if not fin_surface(S):
        return None
    spec = api.PROBLEM_SPECS.get(S["wing"].get("problem"))
    prob = None
    if spec is not None:
        try:
            prob = spec.build({}, S["wing"].get("flags") or {}, None).problem
        except Exception:                                  # noqa: BLE001
            prob = None
    b = float(getattr(prob, "b", 10.0) or 10.0)
    S_w = float(getattr(prob, "S", 10.0) or 10.0)
    arm = float(getattr(prob, "l_t_fixed", None) or 0.0)
    # A PAIR'S FIN IS SIZED ON THE PAIR. A tandem has no ``S`` and no tail
    # arm: its reference area is the two wings' TOTAL and its arm is the
    # STAGGER the user typed — which is exactly what ``api.design_report``
    # sizes the reported fin against, and what ``tandem.evaluate_tandem``
    # charges. Reading the wing+tail's names on a pair quoted a fin of
    # 0.727 m2 on a 5.5 m arm beside a design flying 1.6 m2 on a 5.0 m one:
    # the card would have described a surface less than half the size of the
    # one being paid for.
    #
    # ...ON THE ARM THE SURFACE STANDS AT: the fin sits on a BOOM AFT OF THE
    # REAR WING (``fin.tandem_fin_station``, the one author both engines and
    # the report read). The card must not hard-code where that is — reading
    # the station function, and handing it this design's own boom, is what
    # keeps this card describing the fin that is actually flown.
    if getattr(prob, "S_total", None) is not None:
        S_w = float(prob.S_total)
        arm, _z = _fin.tandem_fin_station(
            float(getattr(prob, "dx", 0.0) or 0.0),
            float(getattr(prob, "dz", 0.0) or 0.0),
            getattr(prob, "fin_boom_m", None))
    # A BOAT'S IS THE MAST, and it is sized by where the foil is rather than
    # by a tail volume: one constructor (``fin.mast``), the same one the
    # report and the loft use, so the card cannot describe a different strut
    # from the one being charged.
    if str(ch.get("medium", "air")) == "water":
        depth = _mid(_searched_box(S), "depth_m")
        # WHERE IT STANDS, off the problem the card is describing. The
        # station used to be absent here as it was everywhere: a strut
        # reported at the front foil, which is the station every arm on the
        # craft is measured from. It is the yaw arm, so a card that dropped
        # it would describe a surface with no lever while the run flies one
        # with a real one — the two-authors defect this constructor exists
        # to have fixed, one field along.
        arm_m = float(getattr(prob, "l_t_fixed", None)
                      or _mid(_searched_box(S), "l_t_m") or 0.0)
        frac = getattr(prob, "x_mast_frac", None)
        if frac is None and hasattr(prob, "strut_model"):
            from aerobo import hydrotail as _ht
            frac = _ht.X_MAST_FRAC
        x_qc = float(frac or 0.0) * arm_m
        geo = _fin.mast(depth=float(depth or 0.0),
                        chord=float(getattr(prob, "c_mast", 0.08) or 0.08),
                        tc=fin_thickness(S), x_qc=x_qc)
        if geo is None:
            return None
        where = (f", standing {geo.x_qc:.3g} m aft of the front foil on a "
                 f"{arm_m:.3g} m fuselage" if geo.x_qc else "")
        return {"area": geo.S, "span": abs(geo.height), "mac": geo.chord,
                "cl": 0.0, "lifting": False,
                "source": f"the strut that carries the foil: "
                          f"{abs(geo.height):.3g} m of submergence "
                          f"(depth_m, mid-box) by a {geo.chord:.3g} m mast "
                          f"chord → {geo.S:.4g} m²{where}"}
    if not arm:
        from aerobo import tail as _tail
        arm = _tail.default_arm(b)
    try:
        geo = _fin.size_fin(b=b, S=S_w, l_t=arm,
                            tail_type=str(ch.get("tail_type", "conventional")),
                            # its SECTION's thickness (fin_thickness), which
                            # is where the answer lives once stage 2.7 has
                            # chosen one
                            tc=fin_thickness(S))
    except ValueError:
        return None
    if geo is None:
        return None
    return {"area": geo.S, "span": abs(geo.height), "mac": geo.chord,
            # ZERO, and meant: a fin's design point is no side force at all
            "cl": 0.0, "lifting": False,
            "source": f"volume coefficient {geo.V_v:g} on a {geo.l_t:.3g} m "
                      f"arm → {geo.S:.4g} m² at aspect ratio {geo.AR:g}"}


def plate_surface(S: dict) -> bool:
    """Does this vehicle carry an ENDPLATE the user can give an aerofoil to?

    :func:`fin_surface`'s twin, and asked the same way — of the registry,
    never of a name. Three things make it False:

    * an AIRFOIL-ONLY session has no vehicle to bolt a plate to;
    * the FAMILY's plate is not a designed part. Only the designed-endplate
      family declares :data:`api.SECTION_PLATE_KEY`; on the plain car wing
      the plate is a fence of free height carrying the WING's own chord and
      the wing's own section, so there is nothing there to give a section
      to, and on an aircraft there is no plate at all;
    * the plate has been PINNED OUT — the tip-device menu's "none" entry
      pins ``endplate_h_m`` to zero, which is how this shell deletes a
      surface, and a stage for a surface of zero height is a stage for
      nothing.
    """
    if airfoil_only(S):
        return False
    from aerobo import api as _api
    sp = _api.PROBLEM_SPECS.get(S["wing"].get("problem"))
    if sp is None or _api.SECTION_PLATE_KEY not in sp.flags:
        return False
    h = _mid(_searched_box(S), "endplate_h_m")
    return h is None or h > 0.0


def _plate_geometry(S: dict) -> dict | None:
    """The ENDPLATE's own size, from the box the run will actually search.

    NO SEED EVALUATION IS NEEDED, and that is worth saying because the
    obvious way to get the plate's chord is to fly the box centre and read
    ``endplate_chord_m`` off the breakdown. The plate's chord is a RATIO of
    the wing's tip chord and the trapezoid closes in one line: with area S,
    overall width b and taper λ the root chord is 2S/(b(1+λ)) and the tip is
    λ times it, so

        c_ep = endplate_chord_ratio · 2 S λ / (b (1 + λ))

    — which agrees with the flown lattice exactly at the published cant,
    where the plates project nothing and the wing spans the whole width.
    It drifts by the projection once a plate leans or blends, and that is
    the same mid-box approximation every other surface here quotes.

    Its lift is ZERO by construction, exactly as the fin's is: a plate at
    zero toe is an end fence, its section is symmetric, and its drag is what
    the section is chosen for.
    """
    if not plate_surface(S):
        return None
    box = _searched_box(S)
    S_w, b, lam = (_mid(box, "S_m2"), _mid(box, "b_m"), _mid(box, "taper"))
    ratio = _mid(box, "endplate_chord_ratio")
    h = _mid(box, "endplate_h_m")
    if None in (S_w, b, lam, ratio) or not (S_w > 0.0 and b > 0.0
                                            and 1.0 + lam > 0.0):
        return None
    c_tip = 2.0 * S_w * lam / (b * (1.0 + lam))
    c_ep = float(ratio) * c_tip
    h = 0.0 if h is None else float(h)
    if not c_ep > 0.0:
        return None
    return {"area": c_ep * h, "span": h, "mac": c_ep,
            # ZERO, and meant: a plate at zero toe makes no side force, so
            # there is no design lift to screen a section at
            "cl": 0.0, "lifting": False,
            "source": f"chord ratio {ratio:g} on a {c_tip:.4g} m tip chord "
                      f"→ {c_ep:.4g} m plate chord, {h:.3g} m tall "
                      f"(mid design box)"}


def surface_geometry(S: dict, surface: str = "main") -> dict | None:
    """One surface's OWN size — ``None`` where it is the reference wing.

    The mission states an area and stage 2 estimates an aspect ratio, which
    between them give A chord — but not every surface's. Two families carry
    surfaces whose chord is not that one:

    * a TAIL / ELEVATOR is a small surface of its own area (``S_t_m2``) at
      its own aspect ratio, and it does not carry the weight: what it lifts
      is whatever trims the aircraft (:data:`TRIM_SURFACE_CL`);
    * a TANDEM pair splits ONE reference area between two wings of the same
      span (``area_split_front``), so the front and the rear fly different
      chords — and therefore different Reynolds numbers — at the same lift
      coefficient. Screening both at the whole pair's chord was the reason
      the two wings kept being handed the same section.

    Everything here is read off the family's OWN design box (mid-point of
    the interval the optimiser will search), never from a table restated in
    the shell — so it moves when the box does, and ``source`` says so.

    The span it quotes is stage 2's ESTIMATE (:func:`section_aspect_ratio`),
    not the aspect ratio stage 3 flies. Every consumer of this function is a
    stage-2 quantity — the Reynolds number the library is screened at, the
    wing the shape optimiser flies its candidates on, the hints beside them
    — and :func:`design_point` already quotes the estimate for every family
    that has no surface of its own. Reading the FLOWN aspect ratio here made
    the tandem the one family whose section was designed for stage 3's wing
    while both stages went on saying it was designed for stage 2's.
    """
    if surface == "plate":
        # THE ENDPLATE'S OWN SIZE. Its chord is a design ROW of its own (a
        # ratio of the wing's tip chord), so it is neither the reference
        # wing's chord nor a share of it — screening the plate at the wing's
        # Reynolds number was the whole reason it could not be designed.
        return _plate_geometry(S)
    if surface == "fin":
        # THE VERTICAL STABILISER'S OWN SIZE, from ``fin.size_fin`` — the one
        # law that sizes it, so this cannot describe a surface the design
        # does not have (a V-tail returns None) or a different one from the
        # drag book's. Its lift is ZERO by construction and that is not a
        # placeholder: a fin at zero sideslip must make no side force, which
        # is why its section is symmetric.
        return _fin_geometry(S)
    box = _searched_box(S)
    if "area_split_front" in box:
        split = _mid(box, "area_split_front")
        area = _mission_area(S)
        b = (section_aspect_ratio(S) * area) ** 0.5
        if not (b > 0.0 and area > 0.0) or split is None:
            return None
        frac = split if surface == "main" else 1.0 - split
        s_own = frac * area
        return {"area": s_own, "span": b, "mac": s_own / b,
                "cl": None, "lifting": True,
                "source": f"{frac:.0%} of the pair's {area:.4g} m² "
                          f"(area_split_front, mid-box) on the same "
                          f"{b:.3f} m span"}
    if surface == "main":
        return None                  # the reference wing: the mission's own
    s_t = _mid(box, "S_t_m2")
    if s_t is None or not s_t > 0.0:
        return None
    ar_t = _mid(box, "AR_t")
    if ar_t is None:
        from aerobo.tail import TAIL_AR
        ar_t = float(TAIL_AR)
        ar_src = f"aspect ratio {ar_t:g} (the published surface's)"
    else:
        ar_src = f"aspect ratio {ar_t:.3g} (mid-box — it is a variable)"
    span = (ar_t * s_t) ** 0.5
    trim = trim_lift(S)
    # The lift the section is SCREENED at is the one it sees the way it is
    # MOUNTED. A surface mounted inverted (api.trim_surface_cl -> "inverted")
    # flies its section upside down, and `polar.InvertedPolar` fixes the map
    # exactly: cl(a) = -base.cl(-a). So a surface carrying aircraft-frame C
    # reads the upright catalogue at **-C**, not at |C|.
    #
    # This comment used to say |cl|, which is the same number whenever the
    # surface carries a DOWNLOAD — 1920 of the 2064 families with a second
    # surface, and every case the identity was written against. It is NOT the
    # same for a surface mounted inverted that trims to an UP-load: the water
    # elevator carries +0.2864 and is screened at -0.2864, 50.7 % apart in
    # profile drag and 3.4x apart in cp_min, which feeds a cavitation gate 1:1.
    # `api.py` returns -cl; this reads it. The signed load it actually carries
    # travels beside it as `cl_flown`.
    return {"area": s_t, "span": span, "mac": s_t / span,
            "cl": (TRIM_SURFACE_CL if trim is None
                   else float(trim.get("cl_section", trim["cl"]))),
            "cl_flown": None if trim is None else float(trim["cl"]),
            "inverted": bool(trim is not None and trim.get("inverted")),
            "cl_source": None if trim is None else trim["source"],
            "lifting": False,
            "source": f"{s_t:.4g} m² (S_t_m2, mid-box) at {ar_src}"}


def trim_lift(S: dict) -> dict | None:
    """What the second surface TRIMS at, from the family's own balance.

    :func:`api.trim_surface_cl` closes the two trim equations the solver
    itself imposes, so this is the lift that surface will actually fly —
    not a stand-in. It is sent the mission and the flags the session
    carries, because both move it: the layout (a canard trims from
    upstream, a V-tail on cos² of its area) and the load the mission
    states.

    ``None`` where the family declares no trimming surface, or will not say
    what it trims at.

    The AREA and the ARM come from the box this session will actually search
    (:func:`_searched_box`), so a narrowed ``S_t_m2`` moves the trim lift the
    same way it moves the chord :func:`surface_geometry` quotes — the two are
    read off one box or they disagree on screen. Untouched, the mid-point of
    the searched box IS the family's own mid-box, which is what
    ``api.trim_surface_cl`` defaults to, so nothing published moves.
    """
    from aerobo import api

    from . import config

    name = S["wing"]["problem"]
    if "S_t_m2" not in api.PROBLEM_SPECS[name].default_bounds:
        return None
    box = _searched_box(S)
    try:
        return api.trim_surface_cl(name, s_t_m2=_mid(box, "S_t_m2"),
                                   arm_m=_mid(box, "l_t_m"),
                                   mission_kwargs=config.mission_kwargs(S),
                                   flags=config.flags(S))
    except Exception:      # noqa: BLE001 — a read-out, never fatal
        return None


def _mission_area(S: dict) -> float:
    try:
        return float(S["mission"]["s_ref_m2"])
    except (TypeError, ValueError):
        return 0.0


def surface_design_point(S: dict, surface: str = "main") -> dict:
    """:func:`design_point`, re-quoted at ONE surface's own chord and lift.

    The fluid state, the speed and the weight are the mission's — those are
    properties of the flight, not of a surface. What a surface owns is its
    chord (and so its Reynolds number) and, where it trims rather than
    lifts, its design lift coefficient.
    """
    dp = design_point(S)
    geo = surface_geometry(S, surface)
    if "error" in dp or geo is None:
        return dp
    out = dict(dp)
    out["mac"] = float(geo["mac"])
    out["b"] = float(geo["span"])
    out["s_ref_m2"] = float(geo["area"])
    out["re_mac"] = float(dp["rho"] * dp["v_ms"] * geo["mac"] / dp["mu"])
    if geo["cl"] is not None:
        out["cl_design"] = float(geo["cl"])
    out["surface_source"] = geo["source"]
    return out


def section_conditions(S: dict, surface: str | None = None) -> dict:
    """The (re, mach, cl_design, tc_min, cm_max) the airfoil stage runs at.

    ``surface`` defaults to the one the shell is CHOOSING for (the stage on
    screen, :func:`target_surface`), so the screen, the ranking, the
    staleness check and the shape optimiser all speak about the same surface
    without any of them having to ask which it is. The Re source, the
    override and the gates come from THAT surface's own workspace
    (:func:`airfoil_state`) — the two forms are separate, and each screens on
    what its own form says.

    The design lift coefficient comes from that surface's own design point —
    free to honour, since the cached polars can be re-scored at any Cl. The
    Reynolds number follows ``re_source``: the surface's own Re at ITS MAC
    (the default, :data:`RE_SOURCE_DEFAULT` — a shortlist sweep) or the cached
    library point (instant, but it ranks and flies at ITS Re; see
    :data:`LIBRARY_RE_NOTE`).

    ``override_point`` hands both numbers to the user verbatim. That exists
    because a section is sometimes designed for a point the mission does not
    imply (a scaled test, a published comparison) — and doing it through an
    explicit switch is what stops a typed-in field from silently disagreeing
    with the mission everything else is derived from.
    """
    surface = target_surface(S) if surface is None else surface
    a = airfoil_state(S, surface)
    cond = dict(a["cond"])
    # AIRFOIL-ONLY: stage 1 IS the point. The flow was stated there — a
    # fluid, a speed and a chord, or a density, a viscosity and a Mach
    # number — so the derivation below (which needs a vehicle) is not the
    # one that applies, and the per-surface override switch is not drawn:
    # two controls for one point is the disagreement this function exists
    # to prevent. The GATES stay the surface's own; they are refusals, not
    # an operating point. Note the MACH survives here and nowhere else in
    # this shell: it is a number the user stated (or the altitude implied),
    # and api.optimize_airfoil / screen_airfoils honour it into XFOIL.
    if airfoil_only(S):
        pt = flow_point(S)
        if "error" not in pt:
            cond["re"] = float(pt["re"])
            cond["mach"] = flown_mach(S)
            cond["cl_design"] = float(pt["cl_design"])
        return cond
    if a["override_point"]:
        return cond
    dp = surface_design_point(S, surface)
    if "error" not in dp:
        cond["cl_design"] = float(dp["cl_design"])
        cond["re"] = float(dp["re_mac"])
    cond["mach"] = 0.0
    if a.get("re_source", RE_SOURCE_DEFAULT) == "library":
        pt = library_point()
        if pt:
            cond["re"] = float(pt["re"])
            cond["mach"] = float(pt.get("mach", 0.0))
    return cond


def surface_name(S: dict, surface: str) -> str:
    """What THIS surface is called, in the user's own words.

    One table, because the name is asked for in a dozen places — the stage's
    heading, the "designing" tag, the button that walks to the next surface,
    every log line — and a second copy is how "the tail" and "the second
    surface" came to be two names for one thing on one screen.

    The vertical stabiliser is named as such rather than "fin": the mission
    asks for a "vertical tail", and the shell should not rename the answer.
    """
    if airfoil_only(S) and surface == "main":
        # no vehicle, so there is no "wing": there is one section and it is
        # the whole answer
        return "section"
    if surface == "fin":
        # ...and a BOAT'S is the STRUT, for the same reason: stage 1 asks it
        # as "carry a strut (the mast that holds the foil)", the drag book
        # charges a mast, and a craft whose vertical surface is called a
        # vertical stabiliser on one screen and a mast on the next is two
        # surfaces to the reader.
        return ("strut" if str(S["wing"]["choices"].get("medium", "air"))
                == "water" else "vertical stabiliser")
    if surface == "plate":
        return "endplate"
    if surface == "aft":
        return aft_surface(S) or second_surface_name(S) or "second surface"
    return "wing"


def next_section_stage(S: dict, stage: str) -> str | None:
    """The next surface's section stage after ``stage``, or None if it is the
    last one this configuration has.

    THE PIPELINE ORDER, read off :data:`STAGES` and :func:`stage_visible`
    rather than restated: "done here" on the wing's section used to name
    ``airfoil_aft`` literally and every other surface used to fall through to
    stage 3. With a vertical stabiliser on the vehicle that skipped it — the
    horizontal tail's Next went to the wing stage, and the fin then quietly
    flew whatever section it was left with, which is exactly the defect the
    aft hand-off exists to prevent.

    A third surface must not need a third literal, so this walks the stage
    list. Anything added to :data:`SURFACE_STAGES` joins the chain by being
    in it.
    """
    order = [st for st in STAGES if st in STAGE_SURFACE]
    if stage not in order:
        return None
    for nxt in order[order.index(stage) + 1:]:
        if stage_visible(S, nxt):
            return nxt
    return None


def target_surface(S: dict) -> str:
    """Which surface the shell is currently choosing a section for.

    ONE source: the stage on screen. Each surface has its own stage
    (:data:`SURFACE_STAGES`) and its own workspace, so there is no target
    setting that can be left pointing at a surface the user is not looking
    at — the bug that switch existed to have.

    Falls back to the wing everywhere else, and wherever the configuration
    has no second surface at all.
    """
    surface = STAGE_SURFACE.get(S["ui"]["selected"], "main")
    if surface == "aft" and not aft_surface(S):
        return "main"
    # ...and the same guard for the vertical stabiliser: a surface the
    # vehicle does not have cannot be the one being designed
    if surface == "fin" and not fin_surface(S):
        return "main"
    if surface == "plate" and not plate_surface(S):
        return "main"
    return surface


# ---------------------------------------------------------- chosen section
def release_chosen_section(S: dict) -> bool:
    """Switch the "[chosen section]" twin off when there is no section left.

    RULE 6 — state can never disagree with its own menus. ``fly_section`` is
    not a flag on a run: on the families that SELECT their section by
    thickness (``api.CHOSEN_SECTION_TWINS``, the water ones) it selects a
    DIFFERENT PROBLEM, one whose design vector has no t/c because the polar
    and the Cp_min table come from one cached sweep of the shape stage 2
    picked. Clearing the section used to leave that twin selected with nothing
    to give it: the run then died inside ``api.run`` with
    ``MissingSectionError`` — deliberately not a ``ValueError``, so the
    penalty contract does not absorb it and the job is killed — and nothing on
    screen could show or undo the state, because stage 3's own switch was not
    drawn while no section is chosen. That switch is now gone entirely (the
    "Section carried from stage 2" panel was deleted), so this is the ONLY
    thing standing between a cleared section and a killed job.

    It lives HERE, beside the state it protects, rather than in the one
    handler that used to clear a section: there are three such paths (the
    clear-section action, a medium change, and stage 2 never being answered),
    and a rule enforced in one of them is not enforced.

    Returns True when it changed something, so the caller can re-derive the
    problem and say so.
    """
    ch = S["wing"]["choices"]
    if not ch.get("fly_section"):
        return False
    ch["fly_section"] = False
    return True


def set_section(S: dict, section: dict | None, decision: str | None,
                surface: str = "main"):
    """Record stage 2's outcome for one surface (or clear everything).

    Choosing for the AFT surface never changes the pipeline's decision
    state: the wing's section is what unlocks stage 3, and the second
    surface's is an addition to it. Clearing (``section is None`` with no
    decision) clears both — that is the medium-change path, where a section
    screened for the old vehicle must not survive.
    """
    a = S["airfoil"]
    key = SECTION_KEYS.get(surface)
    if key:
        a[key] = section
        return
    a["section"] = section
    a["decision"] = decision
    if section is None and decision is None:
        # a full clear takes EVERY secondary surface with it: a section
        # screened for the old vehicle must not survive a medium change,
        # and listing them by hand is how one gets left behind
        for k in SECTION_KEYS.values():
            a[k] = None


def section_summary(S: dict, surface: str = "main") -> str:
    """One line for the tree badge / status bar."""
    a = S["airfoil"]
    if surface == "aft":
        sec = a.get("section_aft")
        if not sec:
            return "as the wing"
        return str(sec.get("name", "section"))
    if surface in ("fin", "plate"):
        # ...and the FIN, which fell through to the wing's branch and
        # reported the WING's section on the vertical tail's own badge. Its
        # unanswered state is not "as the wing" either: a stabiliser flies a
        # SYMMETRIC section at its own thickness until one is chosen
        # (:func:`fin_default_section`), which is a different sentence. The
        # ENDPLATE is the same surface type and the same sentence, at its own
        # default (:func:`plate_default_section`).
        sec = a.get(SECTION_KEYS[surface])
        if not sec:
            own = (plate_default_section(S) if surface == "plate"
                   else fin_default_section(S))
            return f"{own['name']} (its own default)"
        return str(sec.get("name", "section"))
    sec = a["section"]
    if not sec:
        # ...and in a session with no vehicle there is no family whose own
        # section it could be flying: nothing has been chosen, and that is
        # the whole statement
        return ("nothing chosen yet" if airfoil_only(S)
                else "the family's own section")
    tc = sec.get("tc")
    tag = sec.get("name", "section")
    return f"{tag}" + (f" · t/c {tc:.3f}" if tc else "")


def section_weights(S: dict, surface: str = "main"):
    """``(w_upper, w_lower)`` of a surface's chosen section, or ``None``."""
    sec = section_of(S, surface) or {}
    w_u, w_l = sec.get("w_upper"), sec.get("w_lower")
    if not w_u or not w_l:
        return None
    return [float(v) for v in w_u], [float(v) for v in w_l]


#: the surfaces a section can be chosen FOR. "main" is the wing (the foil,
#: the car's rear wing, a tandem pair's front wing); "aft" is the second
#: surface where the configuration has one — the tail/elevator, or the rear
#: wing of a tandem pair. Families with no second surface never show it.
SURFACES = ("main", "aft", "fin", "plate")


def second_surface_name(S: dict) -> str | None:
    """What this configuration's SECOND surface is CALLED, or None.

    A naming question, so it is answered from the configuration: a craft
    with an elevator has an elevator whether or not the solver family it
    selected can be handed a section for it (:func:`aft_surface`).
    """
    ch = S["wing"]["choices"]
    if ch.get("system") == "tandem":
        return "rear wing"
    if not ch.get("tail"):
        return None
    return "elevator" if S["medium"] == "water" else "tail"


def fin_surface(S: dict) -> bool:
    """Does this vehicle carry a VERTICAL SURFACE the user can design?

    A mission question since V5 — whether there is a fin at all decides how
    many surfaces there are, which is the same kind of question as whether
    there is a second surface. Its GEOMETRY (volume coefficient, aspect
    ratio, station) stays on stage 3's Wing type card, where the rest of the
    design box lives; this answers only "is there one".

    Four things make it False, and only one of them is the user's:

    * an AIRFOIL-ONLY session has no vehicle to bolt a fin to;
    * the FAMILY carries no fin. A plain wing does not — 439 of the 4279
      registered problems declare none of the fin keys, and the DEFAULT
      fresh session is one of them — so stage 2.7 was offering to choose a
      section for a surface that would never be built, charged or flown.
      Asked of the spec, which is where "does this family have a fin" is
      already asked on stage 3;
    * a V-TAIL carries none by construction — its cant is its yaw stiffness,
      and ``fin.size_fin`` returns None for one. That rule is ASKED of the
      engine rather than restated here, so the shell cannot offer a stage
      for a surface the design does not have;
    * the user says no.

    A WATER craft is excluded for the reason ``_derive_hydro_tail`` gives:
    its vertical is the strut, which is not a Raymer volume-coefficient tail
    and is not asked for here.
    """
    if airfoil_only(S):
        return False
    ch = S["wing"]["choices"]
    # A WATER CRAFT IS NOT EXCLUDED ANY MORE, and the reason it was is the
    # reason it is not: "its vertical is the strut, not a Raymer
    # volume-coefficient fin". True, and the strut is now a surface in its
    # own right — ``fin.mast`` sizes it from the submergence depth and the
    # mast chord, ``hydrofoil._mast_cd0`` has always charged exactly that
    # wetted area, ``api.design_report`` states it and the flight rebuild
    # flies it. What was missing was never the surface; it was the ASK.
    if not bool(ch.get("fin", FIN_DEFAULT)):
        return False
    from aerobo import api as _api
    sp = _api.PROBLEM_SPECS.get(S["wing"]["problem"])
    if sp is not None and not any(k in sp.flags for k in
                                  (*_api.FIN_SHAPE_KEYS,
                                   _api.FIN_PRESENCE_KEY)):
        return False
    # the LAYOUT's own answer, from the one function that owns it: a V-tail
    # has no separate fin whatever the switch says.
    #
    # ...ASKED ONLY WHERE THE LAYOUT IS ASKED. An empennage is how a SECOND
    # SURFACE is arranged — stage 1 disables the select without one ("add a
    # tail first") — and the stored answer survives every configuration
    # change, so a session that chose a V-tail and then moved to a TANDEM
    # went on reading ``v_tail`` here. That answered "this craft has no
    # vertical surface" for a pair, which took away its Vertical tail card
    # and stage 2.7, while the run built, charged, lofted and flew a fin
    # (``api.design_report`` sizes a pair's at ``tail_type="tandem"``, and
    # ``fin.has_fin`` reads no ``tail_type`` off a tandem problem at all
    # because the flag is not in the family's spec and never travels).
    # The same rule ``ch_tail_type`` states for water, one family across.
    from aerobo import fin as _fin
    layout = (str(ch.get("tail_type", "conventional")) if ch.get("tail")
              else "conventional")
    return _fin.size_fin(b=10.0, S=10.0, l_t=5.5,
                         tail_type=layout) is not None


def aft_surface(S: dict) -> str | None:
    """The second surface that can be handed a SECTION OF ITS OWN, or None.

    Read off the DERIVED PROBLEM, never from the menu state: a family that
    selects its sections by thickness has a second surface but no table for
    a chosen one to replace, so the stage must not offer to choose it.

    An AIRFOIL-ONLY session has none: there is no vehicle for a second
    surface to be part of. Answered here rather than at each of the dozen
    callers (the stage list, the labels, the badges, the screen's surface
    row, the weights recommendation) — one predicate, so the mode cannot be
    half applied.
    """
    from aerobo import api

    if airfoil_only(S):
        return None
    if api.SECTION_AFT_KEY not in api.PROBLEM_SPECS[S["wing"]["problem"]] \
            .flags:
        return None
    return second_surface_name(S)


def section_of(S: dict, surface: str = "main") -> dict | None:
    """The section chosen for one surface (the main one's, by default).

    An aft surface with no section of its own flies the main one's — the
    same default the solvers carry, said once here so every view agrees.
    """
    a = S["airfoil"]
    key = SECTION_KEYS.get(surface)
    if key:
        own = a.get(key)
        if own:
            return own
        if surface in ("fin", "plate"):
            # A FIN DOES NOT INHERIT THE WING'S SECTION. Every other surface
            # does — that is the default the solvers carry (polar_tail /
            # polar_rear = None) — but the wing's is CAMBERED, and a
            # cambered fin flies a permanent side load with nothing to trim
            # it against. Falling through here handed a stabiliser a
            # GOE741 by default, which is the same defect as letting one
            # WIN its screen, one layer down and switched on out of the box.
            #
            # Its default is its own symmetric stand-in, at the thickness
            # its drag is charged at — the section ``cad.fin_surface`` lofts
            # and the OpenVSP script writes, so the shell, the export and
            # the drag book agree without being told twice.
            return (plate_default_section(S) if surface == "plate"
                    else fin_default_section(S))
        return a.get("section")
    return a.get("section")


def fin_thickness(S: dict) -> float:
    """How thick the vertical stabiliser IS — from its own section.

    THE SECTION DECIDES, not a number typed on stage 3. Stage 2.7 chooses (or
    searches) the fin's aerofoil, and an aerofoil has a thickness; asking for
    one again on the Wing type card made the same property answerable in two
    places, and the two could disagree — the card charged drag and lofted CAD
    at one t/c while the section flown was another.

    Falls back to the stated ``fin_tc`` flag, and then to
    ``fin.FIN_TC_DEFAULT``, for the case where no section has been chosen
    yet: the fin's stand-in section IS that thickness
    (:func:`fin_default_section`), so the two agree by construction.
    """
    from aerobo import fin as _fin

    own = S["airfoil"].get(SECTION_KEYS["fin"])
    tc = (own or {}).get("tc") if own else None
    if tc in (None, ""):
        tc = (S["wing"].get("flags") or {}).get("fin_tc")
    try:
        tc = float(_fin.FIN_TC_DEFAULT if tc in (None, "") else tc)
    except (TypeError, ValueError):
        return float(_fin.FIN_TC_DEFAULT)
    return tc if tc > 0.0 else float(_fin.FIN_TC_DEFAULT)


def fin_default_section(S: dict) -> dict:
    """The symmetric section a vertical stabiliser flies until one is chosen.

    Named as a function rather than a constant because its THICKNESS is the
    user's (``fin_tc``), and the name has to follow it: a card saying
    "NACA 0010" beside a 14 % fin would be the shell describing a section
    nobody is flying.
    """
    from aerobo import fin as _fin

    tc = (S["wing"].get("flags") or {}).get("fin_tc")
    tc = float(_fin.FIN_TC_DEFAULT if tc in (None, "") else tc)
    return {"name": f"NACA 00{round(tc * 100):02d}",
            "tc": tc, "symmetric": True, "source": "the fin's own default"}


def plate_default_section(S: dict) -> dict:
    """The symmetric section a car ENDPLATE flies until one is chosen.

    :func:`fin_default_section`'s twin, and its thickness follows the same
    rule: the plate's t/c is a DESIGN ROW (``endplate_tc``), so the stand-in
    is named at the middle of the band the run will actually search rather
    than at a constant this file would have to keep in step with
    ``endplate.TC_BOUNDS``.

    Symmetric for the physics ``endplate.py``'s own docstring gives: a
    vertical panel built at theta = twist − alpha_L0 carries a side force at
    zero toe unless alpha_L0 is zero, and the two plates' loads cancel in CY
    — so a cambered default would be invisible in the totals and paid for in
    induced drag on both plates.
    """
    tc = _mid(_searched_box(S), "endplate_tc")
    tc = 0.12 if tc is None or not tc > 0.0 else float(tc)
    return {"name": f"NACA 00{round(tc * 100):02d}",
            "tc": tc, "symmetric": True,
            "source": "the endplate's own default"}


def section_is_own(S: dict, surface: str) -> bool:
    """Was a section chosen for this surface specifically?"""
    key = SECTION_KEYS.get(surface)
    return key is None or bool(S["airfoil"].get(key))


# ------------------------------------------- criterion weights per surface
def surface_job(S: dict, surface: str = "main") -> str:
    """What this surface is FOR, as far as choosing a section goes.

    ``wing`` (it carries the design load), ``tandem-front`` / ``tandem-rear``
    (it carries a share of it, in or ahead of the other wing's wake), or
    ``trim`` (it carries whatever balances the pitching moment). The three
    jobs want different sections, which is the whole reason this stage is
    asked twice.
    """
    # ...and in a session with no vehicle it is for nothing but itself
    if airfoil_only(S):
        return "section"
    if surface in ("fin", "plate"):
        # its own job, and not "trim": a stabiliser's section is chosen for
        # zero side force and low drag, where a trimming surface's is chosen
        # for the download it has to make. AN ENDPLATE'S JOB IS THE SAME ONE
        # — it is a vertical surface at nominally zero incidence whose whole
        # contribution is the nonplanar benefit, and what its section buys is
        # drag and stiffness — so it is scored by the same weights rather
        # than given a fourth set nothing here measured.
        return "fin"
    if S["wing"]["choices"].get("system") == "tandem":
        return "tandem-front" if surface == "main" else "tandem-rear"
    geo = surface_geometry(S, surface)
    if geo is not None and not geo["lifting"]:
        return "trim"
    if surface == "main" and _has_trimming_surface(S):
        return "wing-trimmed"
    return "wing"


def _has_trimming_surface(S: dict) -> bool:
    """Is there a second surface whose job is to TRIM this one?"""
    if S["wing"]["choices"].get("system") == "tandem":
        return False
    geo = surface_geometry(S, "aft")
    return geo is not None and not geo["lifting"]


def nose_down_required(S: dict, surface: str = "main") -> bool:
    """Should this surface's screen REFUSE reflexed (nose-up) sections?

    True only for a wing that has a trimming surface, and that is the whole
    argument: reflex is what an aerofoil does INSTEAD of having a tail, so
    on an aeroplane that HAS one it inverts the stabiliser's job — the wing
    holds its own pitching moment, the trim balance asks the tail for
    up-load, and a "stabiliser" comes out lifting. The trimming surface
    itself is unaffected (it is screened as mounted, either way up), and so
    is a tandem pair, where neither wing is trimming the other in this
    sense.

    A DEFAULT, not a ban — the screen form's gate row turns it off, and the
    repo rule is that a calibration never refuses a user's number.
    """
    if surface != "main":
        return False
    a = airfoil_state(S, surface)
    stated = a.get("nose_down")
    if stated is not None:
        return bool(stated)
    return _has_trimming_surface(S)


#: job -> (weights, one line saying why those and where they come from)
JOB_WEIGHTS = {
    # AIRFOIL-ONLY: the section is not for a surface of anything, so the
    # recommendation is the general-purpose one and it says so rather than
    # describing a wing that carries a design load nobody stated
    "section": (DEFAULT_WEIGHTS,
                "the GDP bulk-sweep preset: cruise L/D first, then Cl max "
                "and the pitching moment — the general-purpose ranking, and "
                "this session states no surface for the section to be FOR"),
    "wing": (DEFAULT_WEIGHTS,
             "the GDP bulk-sweep preset: cruise L/D first, then Cl max and "
             "the pitching moment — a wing that carries the design load"),
    # A wing with a TAIL is the one case where |Cm| should not be ranked:
    # the tail exists to carry the wing's pitching moment, so scoring the
    # wing on it is asking one aeroplane to solve the same problem twice —
    # and, because |Cm| is lower-better, it rewards REFLEX, which is what a
    # section does instead of having a tail. The weight it frees goes back
    # to cruise L/D, the thing this wing is actually for.
    "wing-trimmed": ({**DEFAULT_WEIGHTS, "cm": 0.0,
                      "ldcr": float(DEFAULT_WEIGHTS.get("ldcr", 0.35))
                      + float(DEFAULT_WEIGHTS.get("cm", 0.2))},
                     "the GDP bulk-sweep preset with the |Cm| criterion "
                     "moved to cruise L/D: this wing HAS a tail, and the "
                     "tail is what carries its pitching moment — ranking "
                     "the wing on |Cm| too would reward a REFLEXED section, "
                     "which is what an aerofoil does instead of having a "
                     "tail (and it leaves the stabiliser lifting)"),
    "tandem-front": (TANDEM_FRONT_WEIGHTS,
                     "GDP's own FRONT-wing preset: cruise L/D first, and a "
                     "Cl-max weight, because the front wing of a tandem sets "
                     "where the pair stalls"),
    "tandem-rear": (TANDEM_REAR_WEIGHTS,
                    "GDP's own REAR-wing preset: cruise L/D first, and four "
                    "times the stall-angle weight — the rear wing flies in "
                    "the front wing's downwash, so the angle it sees moves "
                    "with the front wing's loading"),
    # THE PLATE IS SCORED BY THE FIN'S WEIGHTS, and shares the entry rather
    # than copying it. Same surface type and the same three dead criteria: a
    # plate at zero toe makes no side force, so |Cm| is identically zero and
    # both L/D criteria are read at a lift it never carries. What is left —
    # the drag it costs, the thickness it needs, the angle it still works at
    # — is exactly what an endplate's section is chosen for. A fourth preset
    # would be four numbers nothing in this repository measured.
    "fin": (FIN_WEIGHTS,
            "a SYMMETRIC preset with the three criteria a fin cannot be "
            "ranked on set to zero: |Cm| is identically zero on a symmetric "
            "section, and BOTH L/D criteria need a lift this surface does "
            "not carry — “L/D at the design Cl” is 0 for every candidate, "
            "and “(L/D) max” is read at whatever lift maximises it, which "
            "ranks these sections almost independently of the drag they "
            "cost (Spearman +0.077). Its weight went to the DRAG at the "
            "design Cl, which here is the zero-lift drag itself. What is "
            "left is the drag it costs, the thickness that has to house a "
            "spar and a rudder hinge, and the angle it still works at — "
            "because a rudder earns its section at DEFLECTION, not at zero "
            "sideslip"),
    "trim": (TRIM_WEIGHTS,
             "GDP's SYMMETRIC preset: less on cruise L/D than a wing's and "
             "more on stall angle and thickness — a stabiliser is asked for "
             "control authority and the margin to keep it — but it still "
             "ranks cruise L/D, at the lift the trim balance says this "
             "surface carries"),
}


def recommended_weights(S: dict, surface: str = "main") -> tuple[dict, str]:
    """``(weights, why)`` for one surface — the answer to "what should this
    form open on?".

    Not a table of the shell's own invention: three of the four sets are
    ``airfoil_select.PRESETS`` verbatim (the GDP bulk-sweep, front-wing and
    rear-wing presets), and the fourth is the symmetric preset with the one
    criterion that cannot rank anything at zero lift moved aside
    (:data:`TRIM_WEIGHTS`).
    """
    return JOB_WEIGHTS[surface_job(S, surface)]


def weights_are_recommended(S: dict, surface: str = "main") -> bool:
    """Is this surface's weight set still the recommended one (unedited)?"""
    a = airfoil_state(S, surface)
    if a.get("weights_source") != "recommended":
        return False
    want = recommended_weights(S, surface)[0]
    return all(abs(float(a["weights"].get(k, 0.0)) - float(v)) < 1e-12
               for k, v in want.items())


def set_weights(S: dict, surface: str, weights: dict, source: str):
    """Write one surface's criterion weights and say where they came from."""
    a = airfoil_state(S, surface)
    a["weights"] = {k: float(v) for k, v in dict(weights).items()}
    a["weights_source"] = source


def refresh_recommended_weights(S: dict) -> list[str]:
    """Re-open every UNEDITED weight form on what its surface is now for.

    The configuration decides the job (a tandem's rear wing, a tail that
    trims), and the configuration can change under a form that is already
    open — adding a tail turns stage 2.5's surface from nothing into a
    trimming one. So the recommendation is re-applied wherever the user has
    not overridden it, and NEVER where they have: an edited weight set is an
    answer, and nothing in this shell resets an answer.
    """
    notes = []
    for surface in SURFACES:
        a = airfoil_state(S, surface)
        if a.get("weights_source") != "recommended":
            continue
        want, why = recommended_weights(S, surface)
        if all(abs(float(a["weights"].get(k, 0.0)) - float(v)) < 1e-12
               for k, v in want.items()):
            continue
        set_weights(S, surface, want, "recommended")
        # ...named as the surface it IS. "second surface" was the name given
        # to every surface that is not the wing, so the fin's and the
        # endplate's notes both claimed to be about the tailplane.
        name = surface_name(S, surface)
        notes.append(f"criterion weights for the {name} now follow its job "
                     f"({surface_job(S, surface)}) — {why}")
    return notes


def section_point(sec: dict | None) -> dict | None:
    """``{"re", "mach"}`` a LIBRARY pick must be flown at, or ``None``.

    ``None`` means "the cached library point" — the section was screened
    there, the sidecar already holds that sweep, and the flag can travel as a
    bare name exactly as it always did. A point is returned only when the
    section was screened somewhere else, which is the case the run has to be
    told about.
    """
    if not sec:
        return None
    cond = sec.get("conditions") or {}
    try:
        re_sec = float(cond["re"])
    except (KeyError, TypeError, ValueError):
        return None
    mach = float(cond.get("mach") or 0.0)
    lib = library_point()
    if lib and abs(re_sec - float(lib["re"])) <= 1e-9 * abs(re_sec) \
            and abs(mach - float(lib.get("mach", 0.0))) <= 1e-12:
        return None
    return {"re": re_sec, "mach": mach}


def section_flown_re(S: dict, surface: str = "main") -> float | None:
    """The Reynolds number the RUN will fly this surface's section at.

    Resolved by ``api.section_polar_point`` from the very value the flag will
    carry, so the number shown is the number the solver uses — restating the
    fallback rules here is exactly how the two would drift apart. The wing
    stage shows it beside the surface's own design Re, because those being
    different is a real modelling statement and used to be invisible.
    """
    from aerobo import api

    value = section_flag_value(S, surface)
    if value is None:
        return None
    try:
        return float(api.section_polar_point(value)["re"])
    except (TypeError, ValueError, KeyError):
        return None


def section_flies_off_point(S: dict, surface: str = "main"):
    """``(flown_re, surface_re)`` when those disagree, else ``None``.

    The question the wing stage has to answer out loud: is the section this
    surface carries a section AT the Reynolds number this surface flies? A
    library pick screened at the cached point is not, whenever the surface's
    own chord puts it somewhere else — and profile drag is not a weak function
    of Re (hg40: L/D 80.6 at 1e6, 42.2 at 3e5, 19.5 at 1.5e5).
    """
    flown = section_flown_re(S, surface)
    dp = surface_design_point(S, surface)
    own = dp.get("re_mac") if "error" not in dp else None
    if not flown or not own:
        return None
    if abs(float(own) - float(flown)) <= POINT_MOVE_TOL * abs(float(flown)):
        return None
    return float(flown), float(own)


def section_point_penalty(S: dict, surface: str = "main") -> dict | None:
    """What the CURRENT section costs at the point the surface actually flies.

    A detection is not a diagnosis. Reporting "these two Reynolds numbers
    disagree" asks the user to price the disagreement themselves, in a
    quantity — profile drag against Reynolds number — that is exactly the
    thing they came here for the tool to know. This answers it in the user's
    own variables: the section they already chose, its cruise L/D at the point
    it was screened at, and its cruise L/D where the surface actually flies.

    Returns ``{"name", "cl", "re_screened", "re_own", "ld_screened",
    "ld_own", "loss_pct"}`` or ``None`` when the two points agree, the pick is
    not a library one, or either polar cannot be had **from cache**. It never
    starts an XFOIL sweep — this runs inside a render, and a view that can
    block for minutes is worse than a view that says nothing.
    """
    from aerobo import api

    off = section_flies_off_point(S, surface)
    if not off:
        return None
    flown, own = off
    a = airfoil_state(S, surface)
    pick = (a.get("section") or {}) if isinstance(a.get("section"), dict) else {}
    name = pick.get("name") or (a.get("screen", {}).get("report") or {}).get(
        "winner", {}).get("name") if isinstance(pick, dict) else None
    value = section_flag_value(S, surface)
    if isinstance(value, dict):
        name = value.get("name") or name
    elif isinstance(value, str):
        name = value
    if not isinstance(name, str) or not name:
        return None                      # a DESIGNED section, not a library one
    cl = float(section_conditions(S, surface).get("cl_design") or 0.5)

    def _ld(re: float):
        try:
            # cache_only: this runs inside a render, and the fallback branch
            # of library_section_polar is a synchronous XFOIL sweep
            pol = api.library_section_polar(name, re=float(re),
                                            cache_only=True)
        except Exception:                # noqa: BLE001 — a view must not crash
            return None
        if pol is None:
            return None
        try:
            import numpy as _np
            clv = _np.asarray(pol.CL, dtype=float)
            cdv = _np.asarray(pol.CD, dtype=float)
            if clv.size < 2 or not (clv.min() <= cl <= clv.max()):
                return None
            order = _np.argsort(clv)
            cd_at = float(_np.interp(cl, clv[order], cdv[order]))
            return None if cd_at <= 0.0 else cl / cd_at
        except Exception:                # noqa: BLE001
            return None

    ld_s, ld_o = _ld(flown), _ld(own)
    if ld_s is None or ld_o is None or ld_s <= 0.0:
        return None
    return {"name": name, "cl": cl,
            "re_screened": float(flown), "re_own": float(own),
            "ld_screened": float(ld_s), "ld_own": float(ld_o),
            "loss_pct": 100.0 * (ld_o - ld_s) / ld_s}


def section_flag_value(S: dict, surface: str = "main"):
    """What ``api.SECTION_KEY`` should carry for a surface, or ``None``.

    A LIBRARY pick travels as its name, WITH the point it was screened at
    whenever that is not the cached library point; a DESIGNED section travels
    as its CST weights and the point it was designed at. Both shapes are what
    ``api.section_polar_for`` accepts. Nothing travels when the stage handed
    the section to the solver.

    WHY THE POINT TRAVELS. A library polar is measured data at ONE Reynolds
    number. Sending the name alone made the run fly the cached library sweep
    (Re 1e6) whatever the mission was — hg40's L/D at cl 0.5 is 80.6 there and
    42.2 at Re 3e5, so a small chord flew a section that does not exist at its
    own Reynolds number, and the stage-2 toggle that offered to screen at the
    mission's Re changed nothing downstream even once it worked. A pick made
    AT the library point still travels as a bare name — which is now the
    NON-default route (:data:`RE_SOURCE_DEFAULT`), so a shell-default run
    carries a point and is byte-for-byte a run screened at the library point
    only where the surface's own Re happens to be the cached one.

    A designed section with no weights sends NOTHING. Its ``name`` is a
    display label ("CST section (optimised)"), not a library entry: sending
    it made the solver look the label up in the UIUC directory and raise
    ``FileNotFoundError`` mid-run.

    The DECISION gates the wing's section only. An unanswered stage 2 is a
    statement about the wing — it flies the family's own published section;
    a section chosen for the second surface on its own stage still travels,
    because ``polar_tail`` / ``polar_rear`` is an independent flag and
    dropping it silently would be exactly the kind of quiet loss this shell
    refuses to do.
    """
    a = S["airfoil"]
    if surface in ("aft", "plate"):
        # ...and the ENDPLATE on the same rule: its section is its own stage's
        # answer, so an unanswered stage sends nothing and the plate flies its
        # construction family's build-up. Its DEFAULT is a symmetric stand-in
        # named for the card (`plate_default_section`) and is not a library
        # entry, so it must never travel as one — the same trap a designed
        # section's display label is.
        if not section_is_own(S, surface):
            return None
    elif a.get("decision") is None:
        return None
    sec = section_of(S, surface)
    if not sec:
        return None
    if sec.get("source") == "library" and sec.get("name"):
        pt = section_point(sec)
        if pt is None:
            return str(sec["name"])
        return {"name": str(sec["name"]), **pt}
    w_u, w_l = sec.get("w_upper"), sec.get("w_lower")
    if not w_u or not w_l:
        return None
    cond = sec.get("conditions") or {}
    out = {"w_upper": [float(v) for v in w_u],
           "w_lower": [float(v) for v in w_l],
           "name": sec.get("name") or "designed CST"}
    # the point the section was DESIGNED at travels with it: its polar has
    # to be flown at the Reynolds number it was optimised for, and that
    # sweep is already in the XFOIL cache from stage 2
    if cond.get("re"):
        out["re"] = float(cond["re"])
    if cond.get("mach"):
        out["mach"] = float(cond["mach"])
    return out


# ------------------------------------------------------ mission -> wing
def sync_wing_from_mission(S: dict):
    """Push the mission into the wing sub-state.

    Two things cross this boundary, and only these two:

    * the OPERATING POINT — written into ``mission_edits`` for whichever of
      ``W_N`` / ``V`` / ``altitude_m`` / ``depth_m`` the derived problem
      actually honours (a field the problem does not carry is dropped
      rather than sent, which would raise);
    * the SIZE — area from the mission, span from the size card
      (:func:`nominal_span`: the chosen span, or the aspect-ratio estimate
      where nobody chose one), and only onto a family that declares the
      planform flags (``api.resizable``). That is the FIXED-planform case: a
      family whose span is CONSTRAINED does not declare those flags at all (a
      wing is sized one way), and gets its span from the design box and its
      area from the wing loading instead. The CAR is the one family left with
      neither — its span is an ordinary design-box row against a fixed
      reference area — so its size stays its own and the mission area is used
      for the section design point alone. The hydrofoil used to be in that
      sentence; it is not any more (api._RESIZABLE_PROBLEMS), chosen section
      and designed section included.

    A value equal to the problem's own default is dropped, so an untouched
    mission reproduces the published run bit-for-bit.
    """
    from aerobo import api

    W = S["wing"]
    name = W["problem"]
    spec = api.PROBLEM_SPECS[name]
    m = S["mission"]
    # ...and BEFORE either of them on the track, because the car's stored
    # operating point is derived from its design box rather than typed
    # (:func:`track_point_sync`). Pushing a stale mirror would send the
    # weight and area of the box the session had before this edit.
    track_point_sync(S)

    if api.resizable(name):
        b = nominal_span(S)
        own = api.planform_size(name)
        same = own is not None and (float(own[0]), float(own[1])) == (
            float(b), float(m["s_ref_m2"]))
        # a size EQUAL to what the family already flies is not a resize:
        # sending it would put b_m/S_m2 in the flags of a run that is
        # otherwise the published one, and every reproduce snippet, saved
        # record and history diff would carry that noise for no change
        W["choices"]["span_m"] = None if same else float(b)
        W["choices"]["area_m2"] = None if same else float(m["s_ref_m2"])
    else:
        W["choices"]["span_m"] = None
        W["choices"]["area_m2"] = None

    from gui.nice_app import planform_flags

    size_flags = planform_flags(W["choices"], name)
    try:
        defaults = api.default_mission_values(name, size_flags)
    except ValueError:
        # THE BACKSTOP, not the guard. Every setter that can state a size
        # checks the pair first (:func:`planform_size_refusal`), because a
        # number this shell has already stored is one the user is looking at.
        # This catches the ways a size can arrive without passing one — a
        # preset, a restored session, a family change that re-derives the
        # span — where the alternative is a ValueError out of whatever
        # handler happened to call the sync. The family's OWN size answers
        # instead: these values are only used to decide which mission fields
        # count as edits, so the worst case is one more field sent
        # explicitly, against a run that is refused for its size anyway.
        defaults = api.default_mission_values(name, {})
    wanted = {"W_N": float(m["W_N"]), "V": float(m["V"]),
              "altitude_m": float(m["altitude_m"]),
              "depth_m": float(m["depth_m"])}
    edits = {}
    for fld in spec.mission_fields:
        val = wanted.get(fld)
        if val is None:
            continue
        dv = defaults.get(fld)
        if dv is None or float(val) != float(dv):
            edits[fld] = float(val)
    W["mission_edits"] = edits

    # ...and every band a MEASUREMENT wrote goes back, because it was measured
    # over the mission that has just moved (:func:`drop_recommended_bounds`).
    drop_recommended_bounds(S)

    # ...and the SIZE ROWS follow the mission too, wherever the shell wrote
    # them and the user has not since typed into them. Without this the band
    # was correct only at the instant the planform mode was chosen: a session
    # opens on a complete default mission, so choosing the mode first (the
    # natural order) and stating the real mission afterwards left a box
    # measured about a wing nobody was designing, tagged "user" so nothing
    # would touch it.
    write_size_bands(S, only_shell_owned=True)

    # ...and the OPERATING-POINT rows, which need their own call because they
    # have no "switch it on" moment to write them the first time. A size row
    # is written when the planform MODE is chosen (``write_size_bands`` with
    # an explicit ``rows``), and the refresh above then keeps it in step; a
    # speed row is simply always there, so it has no source at all until
    # something writes one — and ``only_shell_owned`` skips exactly that
    # state. The result was a stated operating point that reached nothing:
    # 5 m/s typed on stage 1, 8-16 m/s searched by the run.
    #
    # A row the USER has typed is still theirs and is left alone; unsourced
    # and shell-owned rows follow the mission.
    src = bounds_source(S)
    own = (None,) + tuple(SHELL_BOUNDS_SOURCES)
    following = tuple(r for r in operating_rows(S) if src.get(r) in own)
    if following:
        write_size_bands(S, rows=following)


def apply_choices(S: dict) -> list[str]:
    """Re-derive the wing problem after a builder choice changed.

    Returns the honest notes ``derive_problem`` produced. Three resets live
    here so they happen whichever control moved:

    * a PROBLEM change drops what belonged to the OLD problem (bound labels,
      released rows, an incompatible optimiser) — a bound label the new
      problem does not have refuses to build. FLAGS are filtered rather than
      wiped: one the new family still declares is still the user's answer,
      and its control is still on screen (see below);
    * a MEDIUM change (the mission stage owns the control, but the choices
      dict is the state) re-opens the mission on the new family's own design
      point and drops the chosen section, because a section screened for an
      air wing is not a hydrofoil's section;
    * an UNTOUCHED mission follows the family. If the stated load is still
      the old family's own design load, moving to another family (single ->
      tandem, say, whose 20 m² carries twice the weight) re-opens it on the
      new one's — FIELD BY FIELD (:func:`_follow_the_family`), so a cruise
      altitude or a submergence depth the user typed is not carried off by a
      load that follows. A number the user has typed is theirs and is kept,
      and the form says it is an edit — the shell never overwrites a stated
      number. The aspect-ratio ESTIMATE follows the same rule on its own
      account: still the old family's own aspect ratio means nobody chose it,
      so it re-opens on the new family's.
    * the mission is then re-pushed into the wing, so the operating point
      sent is always the one on screen.
    """
    from aerobo import api
    from gui.nice_app import derive_problem

    W = S["wing"]
    old_problem, old_medium = W["problem"], S["medium"]
    # ...taken BEFORE anything below can wipe it, and put back relaxed at the
    # end: a builder menu is not a mission, and the box should not flick back
    # to its full width and in again every time one is pressed
    # (:func:`relax_measured_bands`).
    carried = measured_bands(S)
    was_own = _mission_is_the_family_default(S, old_problem, old_medium)
    ar_was_own = (float(S["airfoil"].get("ar") or 0.0)
                  == default_aspect_ratio(old_problem))
    medium_changed = W["choices"]["medium"] != old_medium
    if medium_changed:
        # the chosen section goes below (a section screened for an air wing is
        # not a hydrofoil's), so the twin that exists to FLY one cannot stand
        # — and it has to fall BEFORE the problem is derived from these very
        # choices, or the derived family is the one with nothing to fly.
        release_chosen_section(S)
    name, notes = derive_problem(W["choices"])
    if medium_changed:
        S["medium"] = W["choices"]["medium"]
    if name != W["problem"]:
        W["problem"] = name
        # A FLAG THE NEW FAMILY STILL DECLARES IS STILL THE USER'S ANSWER.
        # Wiping the whole dict deselected controls that had not gone
        # anywhere: turning on "the tip device's chord follows the wing's
        # chord law" and then moving the PLANFORM menu (which is a different
        # registered problem — "tandem (nonplanar) + winglets + free chord
        # law" -> "tandem (nonplanar) + winglets") silently switched it back
        # off, and the same for every chord limit, tail height, CG and
        # stagger the user had typed. Nothing about the answer changed: the
        # new family declares the same flag and the same control is on
        # screen offering it.
        #
        # The rule is the one ``config.flags`` already applies at build time
        # — a flag travels only to a family that declares it — moved to where
        # the state lives, so what is KEPT and what TRAVELS cannot disagree.
        # A medium change still clears everything: a chord limit in metres
        # for a 10 m wing is not an opinion about a 1.2 m hydrofoil, which is
        # the same reason the section and the span go below.
        W["flags"] = ({} if medium_changed else
                      {k: v for k, v in (W.get("flags") or {}).items()
                       if k in api.PROBLEM_SPECS[name].flags})
        W["bounds"] = {}
        W["bounds_source"] = {}
        # ...and the released rows with them: "this row does not have to be
        # contained" is an answer about a design VECTOR, and the new family
        # has a different one. A row released on the old problem would sit
        # here with nothing on screen able to show it (its label may not even
        # exist any more) and would quietly release the new family's row of
        # the same name.
        W["bounds_off"] = []
        # ...and the FIXED rows, for the same reason and more strongly: a
        # value is a stronger statement than a band, and one carried onto a
        # family whose row of that name means something else would hold the
        # new design at a number nobody typed for it.
        W["fixed"] = {}
        # ...and the SIZE CARD's chords face, which is the same rule applied
        # to the same card's other question. The chords are cleared NOWHERE
        # else in the tree, and the control that offers them is gated on
        # ``chord_span_available``, so on a family that does not offer it the
        # pair survived with nothing on screen able to reach it: switching
        # the planform to wing loading left them setting nominal_span at
        # 16.67 m (b_m opened 10-66.7 m against the mission's 6-40 m) and
        # pinning ``taper`` on a design nobody had typed a taper for.
        clear_size_chords(S)
        # (the span row goes with them, because it IS one of them: b_m is a
        # design-box row, and a band typed for the old family's span is not
        # an opinion about the new one's. No state outlives its own control.)
        if api.PROBLEM_SPECS[name].has_blocks:
            W["optimiser"] = "blocks"
        if W["optimiser"] not in api.compatible_optimisers(name):
            W["optimiser"] = default_optimiser(name)
    if medium_changed:
        S["mission"].update(mission_defaults(W["problem"], S["medium"],
                                             S.get("water", "sea")))
        S["airfoil"]["ar"] = default_aspect_ratio(W["problem"])
        # ...and the chosen SPAN with them, for the same reason the section
        # goes: a 10 m air wing is not an opinion about a 1.2 m hydrofoil.
        # A mode change (fixed <-> wing loading) does NOT clear it — that is
        # the same wing, asked a different way.
        W["span_m"] = None
        # ...and the CHORDS the size card may have been asked in instead,
        # which are the same statement in another unit (:func:`chord_span`).
        # Air->water is loud without this — ``api._planform_size`` refuses the
        # aspect ratio the carried-over chords imply against a 0.144 m² foil —
        # and air->track is silent, which is worse.
        clear_size_chords(S)
        set_section(S, None, None)
    elif name != old_problem and (was_own or ar_was_own):
        # the family's own point, and the user never stated one — so following
        # it is not a decision being overturned. The ACCEPTANCE is therefore
        # kept: un-accepting here would lock stage 2 and grey out a section
        # that is still perfectly valid, which is the "adding one thing resets
        # everything" behaviour. If the point genuinely moved, say so instead.
        before = design_point(S)
        if was_own:
            _follow_the_family(S, old_problem, W["problem"])
        if ar_was_own:
            S["airfoil"]["ar"] = default_aspect_ratio(name)
        notes = notes + _point_moved_notes(before, design_point(S))
    if name != old_problem:
        # ...AND THE SHELL'S OWN BANDS BACK. The wipe above is right about
        # the USER's rows — a band typed for the old family's vector is not
        # an opinion about the new one's — but it took the shell's mission
        # bands with them and nothing wrote them again, so every builder
        # control except the planform menu (which does this itself) dropped
        # the box back onto the family's PUBLISHED row. Selecting a tip
        # device on a mission stating a 1 m wing moved b_m from 0.6 – 4 m to
        # 6 – 40 m: the span band grew tenfold because a menu about the wing
        # TIP was pressed, and the measured box, which can only narrow INTO
        # the box it is handed, then recommended 11 – 16 m.
        #
        # These are not the user's rows to lose: they are derived from the
        # mission, they are marked ``"shell"``, and they follow the family by
        # construction (:func:`size_band_defaults` reads the new problem).
        write_size_bands(S)
    sync_wing_from_mission(S)
    # ...and the measurement carried through, relaxed onto the NEW family's
    # own rows. After `sync_wing_from_mission`, so the mission's bands are
    # the floor it relaxes onto; a MEDIUM change carries nothing, on the same
    # rule the section and the span go by — a 10 m air wing is not an opinion
    # about a 1.2 m hydrofoil.
    if not medium_changed:
        relax_measured_bands(S, carried)
    # the JOB of each surface can have changed with the family (a tandem's
    # rear wing is not a tail), and an unedited weight form should open on
    # what its surface is now for
    return notes + refresh_recommended_weights(S)


#: relative move in the section's own two conditions (Re and design CL) worth
#: telling the user about — below this the chosen section is unquestionably
#: still the right one, above it the screen was run at a different point.
POINT_MOVE_TOL = 0.02


def _point_moved_notes(before: dict, after: dict) -> list[str]:
    """Notes for a family-following mission whose design point actually moved.

    The section from stage 2 was screened at ``before``'s Reynolds number and
    design lift. Silently re-opening on another family's point would leave a
    section chosen for conditions the run no longer flies.
    """
    out = []
    for key, label, fmt in (("re_mac", "Re at MAC", "{:.3g}"),
                            ("cl_design", "design CL", "{:.4g}")):
        old, new = before.get(key), after.get(key)
        if not old or not new:
            continue
        if abs(float(new) - float(old)) > POINT_MOVE_TOL * abs(float(old)):
            out.append(
                f"this family opens on its own operating point: {label} "
                f"{fmt.format(float(old))} → {fmt.format(float(new))}. The "
                f"section from stage 2 was chosen at the old one — re-screen "
                f"if that matters.")
    return out


def _follow_the_family(S: dict, old_problem: str, new_problem: str):
    """Re-open on the new family's design point ONLY the numbers nobody stated.

    :func:`_mission_is_the_family_default` asks one question — is the LOAD
    (with the speed and the area it is stated against) still the family's own?
    — and the answer used to re-open the WHOLE mission dict on the new
    family's defaults. That took the atmosphere with it: a cruise altitude
    typed in stage 1 matches no field that test compares, so freeing the
    winglets or ticking the tail in stage 3 silently put the run back at ISA
    sea level (``mission_kwargs`` went ``{'altitude_m': 3000.0}`` → ``{}``)
    while the mission node still read ACCEPTED — and the two notes it logged
    blamed the new family for a point it had not moved. The water twin lost a
    stated submergence the same way, moving the cavitation margin and stage
    2's screening point under the user.

    So the follow is per FIELD, and the rule is the one the load already
    obeys: a number still equal to the OLD family's own default was nobody's
    decision and re-opens on the new family's; a number that differs is the
    user's and is kept. ADDITIVE: on an untouched mission every field follows,
    which is exactly what the wholesale update did, so the published run is
    unmoved. The acceptance is not a mission number and is never touched here.
    """
    medium, water = S["medium"], S.get("water", "sea")
    try:
        was = mission_defaults(old_problem, medium, water)
        now = mission_defaults(new_problem, medium, water)
    except (ValueError, KeyError):
        return
    m = S["mission"]
    followed = set()
    for key, value in now.items():
        if key in ("accepted", "w_source"):
            continue
        try:
            stated = float(m[key]) != float(was[key])
        except (KeyError, TypeError, ValueError):
            stated = True
        if not stated:
            m[key] = value
            followed.add(key)
    # the LABEL travels with the load it describes: saying "the problem's own
    # design lift" over a weight the user typed would be a false citation
    if "W_N" in followed:
        m["w_source"] = now["w_source"]


def _mission_is_the_family_default(S: dict, problem: str,
                                   medium: str) -> bool:
    """Is the stated mission still exactly what this family opens on?"""
    try:
        own = mission_defaults(problem, medium, S.get("water", "sea"))
    except (ValueError, KeyError):
        return False
    m = S["mission"]
    return all(float(m[k]) == float(own[k])
               for k in ("W_N", "V", "s_ref_m2"))


# ------------------------------------------------------------- stage gating
def stage_states(S: dict) -> dict:
    """``{stage: (state, reason)}`` — what the tree draws and what it says.

    ``state`` is one of ``done | active | ready | locked | running | error``.
    A locked stage keeps the reason it is locked, because a greyed-out node
    with no explanation is the thing this shell exists to avoid.
    """
    out: dict[str, tuple[str, str]] = {}
    only = airfoil_only(S)
    stage1_ok = stage1_valid(S)
    accepted = stage1_accepted(S)
    out["mission"] = (("done" if accepted and stage1_ok
                       else "error" if not stage1_ok else "ready"),
                      "" if stage1_ok else stage1_error(S))

    a = S["airfoil"]
    if not accepted:
        out["airfoil"] = ("locked",
                          ("accept the flow conditions first — the section "
                           "is designed at the Reynolds number, Mach number "
                           "and lift coefficient they state" if only else
                           "accept the mission first — the section is "
                           "designed at the Reynolds number and lift "
                           "coefficient the mission implies"))
    elif a["screen"]["running"] or a["opt"]["running"]:
        out["airfoil"] = ("running", "")
    elif a["decision"] is not None:
        out["airfoil"] = ("done", "")
    else:
        out["airfoil"] = ("ready",
                          ("screen the library at the stated flow, then "
                           "shape-optimise from what it picks — this stage "
                           "is the whole of an airfoil-only session"
                           if only else
                           "the wing flies the family's own published "
                           "section until one is chosen here"))

    # stage 2.5 — the second surface's own section. It is DONE when that
    # surface has a section of its own and READY otherwise: never a gate on
    # anything, because a second surface with no section flies the wing's.
    aft = aft_surface(S)
    b = airfoil_state(S, "aft")
    if aft is None:
        out["airfoil_aft"] = ("locked",
                              ("this session designs one section — a second "
                               "surface belongs to a vehicle, and this one "
                               "has none" if only else
                               "this configuration has one lifting surface — "
                               "add a tail or a tandem rear wing in stage 1 "
                               "and its own section is chosen here"))
    elif not accepted:
        out["airfoil_aft"] = ("locked", out["airfoil"][1])
    elif b["screen"]["running"] or b["opt"]["running"]:
        out["airfoil_aft"] = ("running", "")
    elif a.get("section_aft"):
        out["airfoil_aft"] = ("done", "")
    else:
        out["airfoil_aft"] = ("ready", f"the {aft} flies the wing's section "
                                       f"until one is chosen here")

    # ...and the FIN's stage, on the same rules. Its locked reason names the
    # two ways a vehicle has no fin, because "add one in stage 1" is wrong
    # advice for a V-tail — that layout answers the question by construction
    # and the switch cannot override it.
    f = S.get("airfoil_fin") or {}
    if not fin_surface(S):
        ch = S["wing"]["choices"]
        out["airfoil_fin"] = ("locked", (
            "this session designs one section — a fin belongs to a vehicle, "
            "and this one has none" if only else
            "a V-tail carries no separate fin: its panels' cant IS its yaw "
            "stiffness, which is the whole point of the layout. Choose a "
            "conventional or T-tail in stage 1 to design one"
            if str(ch.get("tail_type")) == "v_tail" else
            "a water craft's vertical is the strut, not a volume-coefficient "
            "fin, and this shell does not design it"
            if str(ch.get("medium", "air")) != "air" else
            "this vehicle has no vertical tail — turn one on in stage 1 and "
            "its own section is chosen here"))
    elif not accepted:
        out["airfoil_fin"] = ("locked", out["airfoil"][1])
    elif (f.get("screen", {}).get("running")
          or f.get("opt", {}).get("running")):
        out["airfoil_fin"] = ("running", "")
    elif a.get("section_fin"):
        out["airfoil_fin"] = ("done", "")
    else:
        out["airfoil_fin"] = ("ready",
                              "the fin flies a symmetric section at its own "
                              "thickness until one is chosen here")

    # ...and the ENDPLATE's stage, on exactly those rules. Its locked reason
    # names the three ways a vehicle has no plate to design, because "turn it
    # on" is wrong advice for two of them: an aircraft has no endplate at all,
    # and the plain car wing's plate is a fence carrying the WING's chord and
    # the wing's section, which is a different part.
    pl = S.get("airfoil_plate") or {}
    if not plate_surface(S):
        from aerobo import api as _api
        _sp = _api.PROBLEM_SPECS.get(S["wing"].get("problem"))
        _designed = (_sp is not None
                     and _api.SECTION_PLATE_KEY in _sp.flags)
        out["airfoil_plate"] = ("locked", (
            "this session designs one section — an endplate belongs to a "
            "vehicle, and this one has none" if only else
            "the plate has been taken off this design: its height is pinned "
            "at zero, so there is no surface here to give an aerofoil to"
            if _designed else
            "this wing's plates are a FENCE rather than a designed part — "
            "they carry the wing's own chord and the wing's own section, so "
            "there is nothing here to give an aerofoil to. Choose the "
            "endplate mount on stage 3's Wing type card and the plate "
            "becomes a part with a section of its own"))
    elif not accepted:
        out["airfoil_plate"] = ("locked", out["airfoil"][1])
    elif (pl.get("screen", {}).get("running")
          or pl.get("opt", {}).get("running")):
        out["airfoil_plate"] = ("running", "")
    elif a.get("section_plate"):
        out["airfoil_plate"] = ("done", "")
    else:
        out["airfoil_plate"] = ("ready",
                                "the plate flies a build-up on its section "
                                "FAMILY until an aerofoil is chosen here — "
                                "and a family knows nothing about a shape")

    # STAGE 2 IS NOT A GATE — for the same reason stage 2.5 never was. A wing
    # with no chosen section flies the family's own published one, which is a
    # complete, runnable design; locking stage 3 until the user pressed a
    # button that only RESTATED that made the pipeline read as three routes
    # where there are two, and (now that the default screen sweeps live XFOIL)
    # would have priced the way through at minutes of sweeps nobody asked for.
    # What the shell owes instead is a statement, and stage 3's configuration
    # list makes it: the ``airfoil`` row reads "the family's own section"
    # (:func:`section_summary`) when stage 2 was never answered. The MISSION
    # is still a gate: the wing is sized and flown at the point stage 1
    # states.
    # ...and the two stages an AIRFOIL-ONLY session does not have. They are
    # hidden (:func:`stage_visible`) AND locked, because hiding is only
    # cosmetic — the shell mounts every stage, and ``Ctx.select`` is what
    # actually refuses to open one, quoting the reason stored here.
    section_only_why = ("this session designs a section only — switch stage "
                        "1 back to the vehicle pipeline to size and fly a "
                        "wing at the same point")
    if only:
        out["wing"] = ("locked", section_only_why)
        out["results"] = ("locked", "this session designs a section only — "
                                    "the search, its convergence and the "
                                    "shape it found are stage 2's own view")
        return out

    out["wing"] = (("ready", "") if accepted else
                   ("locked", "accept the mission first — the wing is sized "
                              "and flown at the point it states"))

    rec = S["run"]["record"]
    if S["run"]["error"]:
        out["results"] = ("error", S["run"]["error"])
    elif rec is None:
        out["results"] = ("locked", "no completed run yet")
    else:
        out["results"] = ("done", "")
    return out


def stage_available(S: dict, stage: str) -> bool:
    return stage_states(S)[stage][0] != "locked"


# ------------------------------------------------------------ CAD export
def export_dir(S: dict) -> Path:
    """The folder a CAD export lands in — the user's answer, or the default.

    Expanded (``~``) and made absolute here, in one place, so the field, the
    button that writes and the message that names the path all mean the same
    directory.
    """
    raw = (S.get("export") or {}).get("dir") or DEFAULT_EXPORT_DIR
    p = Path(str(raw)).expanduser()
    return p if p.is_absolute() else (REPO_ROOT / p)


def export_dir_raw(S: dict) -> str:
    """WHAT THE USER TYPED for the folder — not what it resolves to.

    The field on the card renders this one. :func:`export_dir` expands ``~``
    and anchors a relative answer at the repo, which is what the WRITER
    needs; painting that back into the input replaces "~/Desktop/wing" with
    "/Users/…/Desktop/wing" the next time the card redraws, and the card
    redraws on its own (the background VSP build publishes progress through
    the heartbeat). Same rule as every other typed field in this shell —
    the field holds the answer, a readout holds what it derives to.
    """
    return str((S.get("export") or {}).get("dir") or DEFAULT_EXPORT_DIR)


def export_stem_raw(S: dict) -> str:
    """...and the same for the file NAME.

    :func:`export_stem` strips anything a filesystem or OpenVSP's script
    would choke on, which is right for the file and wrong for the field: a
    redraw between two keystrokes turned "wing v2" into "wingv2" under the
    cursor. What was typed stays typed; the card quotes the sanitised name
    beside it, so the difference is visible rather than silent.
    """
    return str((S.get("export") or {}).get("stem") or "")


def export_plan(S: dict) -> dict:
    """WHERE and UNDER WHAT NAMES this export would land, before it runs.

    ``{"dir": Path, "stem": str, "typed": str, "names": {kind: Path},
    "existing": [name, …]}`` — the answer to both halves of the question the
    card asks, quoted back as files rather than as a folder. ``existing``
    is what would be REPLACED, because "choose the name" is not a real
    choice if the shell will not say when the name is already taken.

    The names come from ``cad.export_name`` — the writers' own table — so
    the card cannot promise a file the export does not write.
    """
    from aerobo import cad

    d, stem = export_dir(S), export_stem(S)
    names = {k: d / cad.export_name(stem, k)
             for k in ("stl", "dat", "vsp", "vsp3")}
    existing = sorted(p.name for p in names.values() if p.exists())
    return {"dir": d, "stem": stem, "typed": export_stem_raw(S),
            "names": names, "existing": existing}


def set_export_dir(S: dict, value) -> str:
    """Point the export at a folder. Returns the resolved path as a string.

    Nothing is created here: a folder is made when something is written to
    it, so typing a path halfway through does not litter the filesystem.
    """
    S.setdefault("export", {})["dir"] = (str(value).strip()
                                         or str(DEFAULT_EXPORT_DIR))
    return str(export_dir(S))


def export_stem(S: dict) -> str:
    """The base FILENAME an export uses (``<stem>.vsp3``, ``<stem>.stl``…).

    Sanitised to something a filesystem and OpenVSP's own script both take:
    an empty or path-like answer would write outside the folder the user
    chose, which is the one thing this control must not do.
    """
    raw = str((S.get("export") or {}).get("stem") or "").strip()
    safe = "".join(c for c in raw if c.isalnum() or c in "-_.")
    return safe or "aerobo_design"


def export_payload(S: dict) -> dict | None:
    """Everything an export needs, off the run that was flown — or None.

    ``{"geom", "breakdown", "x", "labels", "section", "section_aft", "cfg"}``.
    Read from the SAME design report the result page draws, so a file that
    leaves the shell cannot disagree with the numbers on screen.
    """
    from aerobo import api

    R = S["run"]
    rd = R.get("record") or {}
    rep = R.get("report") or {}
    geom = rep.get("geometry") or {}
    x = rd.get("best_x")
    if not geom or x is None:
        return None
    cfg = api.RunConfig(**{k: (rd.get("config") or {}).get(k) for k in
                           ("problem_name", "mission_kwargs", "flags",
                            "optimiser", "budget", "seed",
                            "bounds_overrides")})
    sr = R.get("section_report") or {}
    section = ((sr.get("design") or {}).get("coords")
               if isinstance(sr, dict) else None)
    if section is None:
        section = api.flown_section_coords(cfg, x)
    return {"cfg": cfg, "geom": geom, "breakdown": rep.get("breakdown") or {},
            "x": x, "labels": rd.get("param_labels") or [],
            "section": section,
            "section_aft": api.flown_section_coords(cfg, x, aft=True)}


# ------------------------------------------------- the SECTION, as files
#
# Stage 4 exports the AIRCRAFT: an STL, an OpenVSP script, and the sections
# the run flew beside them. Two things that never reached is what this block
# is for.
#
# An aerofoil-only session (:func:`airfoil_only`) has no stage 4 at all — the
# whole session is a section, and until now the only way out of the shell
# with it was a screenshot. And even in a vehicle session the section is a
# deliverable of its own, chosen (or designed) at stage 2 and worth having
# hours before the wing has been searched.
#
# Three files, because a section is three separate things to whoever
# receives it:
#
#   section_dat    the SHAPE, Selig/XFOIL order at unit chord — what XFOIL,
#                  OpenVSP, a CAD loft or another optimiser reads. It is the
#                  outline the card DRAWS (``stages.airfoil.section_outline``),
#                  never a second derivation of it: a database section keeps
#                  its own blunt trailing edge, and only a section with no
#                  stored coordinates falls back to its CST refit.
#   section_polar  the SWEEP that was flown, one row per angle of attack,
#                  with the pre-stall monotone branch MARKED. The cd and cm
#                  this shell quotes at the design lift are interpolated on
#                  that branch only, and a reader handed a bare table cannot
#                  tell which rows they came from.
#   section_json   the DESIGN — where it came from, the point it was designed
#                  at, its CST weights, its coordinates and the numbers on
#                  screen. The record that makes the other two reproducible.
#
# All three are written from the SECTION THAT WAS CHOSEN (:func:`section_of`),
# which is the same object stage 3 flies, so a file that leaves the shell
# cannot disagree with what the wing is about to be searched with.

#: the export kinds this block writes, in the order the card offers them.
#: Names come from ``cad.FILE_SUFFIXES`` — the one table (:func:`cad.export_name`).
SECTION_EXPORT_KINDS = ("section_dat", "section_polar", "section_json")

#: what each kind is called on the card, and what it is for.
SECTION_EXPORT_LABELS = {
    "section_dat": ("coordinates (.dat)",
                    "Selig/XFOIL order, unit chord — XFOIL, OpenVSP, CAD"),
    "section_polar": ("polar (.csv)",
                      "the alpha sweep that was flown, one row per point"),
    "section_json": ("design (.json)",
                     "the point, the weights, the coordinates and the "
                     "numbers — everything needed to repeat it"),
}


def section_export_stem(S: dict, surface: str = "main") -> str:
    """The base filename a SECTION export uses on this surface.

    The design's own stem (:func:`export_stem`, shared with the CAD export so
    one answer names every file a session writes), with the second surface's
    files marked: a session that designs two sections must not write one over
    the other.
    """
    stem = export_stem(S)
    return f"{stem}_aft" if surface == "aft" else stem


def section_polar_of(sec: dict | None) -> dict | None:
    """The alpha sweep a chosen section carries, or None.

    One accessor for both sources. A LIBRARY pick carries the screening
    report's winner block and an OPTIMISED one carries its own design block,
    but they are the same shape (``api._polar_payload``) — which is why the
    section card can draw either with one figure, and why one export can
    write either with one writer. A ranked row that did not win carries its
    metrics and not its curves, and that is a None here rather than an empty
    file.
    """
    rep = (sec or {}).get("report") or {}
    pol = ((rep.get("design") or {}).get("polar")) or None
    if not pol:
        return None
    # LENGTH, never truthiness: a report written in-process carries numpy
    # arrays here (only the stored one has been through ``_json_safe``), and
    # ``if pol.get("alpha_deg")`` raises on an array of more than one element
    # — an export that dies on the live report and works on the reloaded one.
    return pol if len(pol.get("alpha_deg") or ()) else None


def section_dat_text(sec: dict | None) -> str | None:
    """The chosen section as Selig/XFOIL ``.dat`` text, or None.

    THE OUTLINE THE CARD DREW. ``section_outline`` is imported from the stage
    rather than copied because the file and the picture must be the same
    aerofoil — a second derivation here is how the drawing came to close a
    trailing edge the section really has (see its docstring).
    """
    import numpy as np

    from aerobo import cad

    from .stages.airfoil import section_outline

    xy = section_outline(sec or {})
    if xy is None:
        return None
    arr = np.asarray(xy, dtype=float)
    name = str((sec or {}).get("name") or "aerobo section")
    return cad.airfoil_dat(arr[:, 0], arr[:, 1], name)


def section_polar_csv(sec: dict | None) -> str | None:
    """The flown sweep as CSV, or None when the section carries no curves.

    ``on_branch`` is the column that stops the table lying by omission: 1
    where the point is on the pre-stall monotone branch the design-lift
    readouts were interpolated on, 0 where it is past it. The operating point
    rides in ``#`` comment lines above the header — ``numpy.loadtxt`` and
    ``pandas.read_csv(comment="#")`` both skip them, and a polar with no
    Reynolds number on it is a table nobody can reuse.
    """
    import numpy as np

    pol = section_polar_of(sec)
    if pol is None:
        return None
    a = np.asarray(pol.get("alpha_deg") or [], dtype=float)
    cl = np.asarray(pol.get("cl") or [], dtype=float)
    cd = np.asarray(pol.get("cd") or [], dtype=float)
    cm = np.asarray(pol.get("cm") or [], dtype=float)
    n = int(min(a.size, cl.size, cd.size, cm.size))
    if n < 1:
        return None
    on = np.zeros(n, dtype=int)
    br = pol.get("branch")
    if br and len(br) == 2:
        on[int(br[0]):int(br[1])] = 1
    else:
        # no branch recorded is not "every point is on it": say so with a
        # column of -1 rather than claiming a confidence nothing measured
        on[:] = -1
    rep = (sec or {}).get("report") or {}
    head = [f"# aerobo section polar: {(sec or {}).get('name', 'section')}",
            f"# source: {(sec or {}).get('source', 'unknown')}",
            f"# re: {float(pol.get('re', 0.0)):.6g}",
            f"# mach: {float(pol.get('mach', 0.0)):.6g}"]
    if rep.get("cl_design") is not None:
        head.append(f"# cl_design: {float(rep['cl_design']):.6g}")
    head.append(f"# converged: {int(pol.get('n_converged', n))} of "
                f"{int(pol.get('n_requested', n))} requested")
    head.append("# on_branch: 1 = pre-stall monotone branch (the cd/cm at "
                "cl_design are interpolated on these rows only), "
                "-1 = not recorded")
    rows = ["alpha_deg,cl,cd,cm,on_branch"]
    rows += [f"{a[i]:.6f},{cl[i]:.6f},{cd[i]:.7f},{cm[i]:.6f},{on[i]:d}"
             for i in range(n)]
    return "\n".join(head + rows) + "\n"


def section_export_record(S: dict, surface: str = "main") -> dict | None:
    """The JSON-safe record of the chosen section, or None if there is none.

    Everything the shell knows about it in one object: which surface it is
    for, where it came from, the POINT it was designed at (its own stored
    conditions — not the form's, which may have moved since), the flow when
    the session states one directly, the CST weights stage 3 would fly, the
    coordinates and the sweep.
    """
    import numpy as np

    from .stages.airfoil import section_outline

    sec = section_of(S, surface)
    if not sec:
        return None
    xy = section_outline(sec)
    out = {
        "shell": {"mode": session_mode(S), "surface": surface,
                  "surface_has_its_own": section_is_own(S, surface)},
        "section": {k: sec.get(k) for k in
                    ("name", "source", "rank", "tc", "ldcr", "clmax", "cm",
                     "te_gap", "best_score") if sec.get(k) is not None},
        "designed_at": dict(sec.get("conditions") or {}),
        "weights": ({"upper": [float(v) for v in sec["w_upper"]],
                     "lower": [float(v) for v in sec["w_lower"]]}
                    if sec.get("w_upper") and sec.get("w_lower") else None),
        "coords": (np.asarray(xy, dtype=float).tolist()
                   if xy is not None else None),
        "polar": section_polar_of(sec),
    }
    rep = sec.get("report") or {}
    for key in ("cl_design", "tc_min", "cm_max"):
        if rep.get(key) is not None:
            out["designed_at"].setdefault(key, rep[key])
    if airfoil_only(S):
        # THE FLOW IS THE SESSION, so it travels with the file: an
        # aerofoil-only session states rho, mu, a speed and a chord, and a
        # Reynolds number with none of them beside it is not reproducible.
        pt = flow_point(S)
        out["flow"] = None if "error" in pt else pt
    return out


def _jsonable(value):
    """Last-resort coercion for :func:`json.dumps` — numpy in, python out."""
    import numpy as np

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"{type(value).__name__} is not JSON-serialisable")


def section_export_files(S: dict, surface: str = "main") -> dict:
    """``{"stem", "files": {kind: {"name", "text"}}, "absent": {kind: why}}``.

    Pure: nothing is written and nothing is computed that the section does
    not already carry, so a card may call it on every repaint. A kind that
    cannot be written lands in ``absent`` WITH ITS REASON rather than being
    dropped — a button that is simply missing tells the user nothing.
    """
    import json

    stem = section_export_stem(S, surface)
    sec = section_of(S, surface)
    files: dict = {}
    absent: dict = {}
    if not sec:
        why = ("no section is chosen for this surface yet — screen the "
               "library, or optimise a shape, and adopt one")
        return {"stem": stem, "files": {},
                "absent": {k: why for k in SECTION_EXPORT_KINDS}}

    from aerobo import cad

    def put(kind: str, text: str | None, why: str):
        if text is None:
            absent[kind] = why
        else:
            files[kind] = {"name": cad.export_name(stem, kind), "text": text}

    put("section_dat", section_dat_text(sec),
        "this section carries neither coordinates nor CST weights, so there "
        "is no outline to write")
    put("section_polar", section_polar_csv(sec),
        "this section carries its metrics but not its curves — the screening "
        "report keeps the full sweep for its WINNER only. Re-weight the "
        "criteria so this section wins, or optimise from it, to get a polar")
    rec = section_export_record(S, surface)
    # ``default`` and not a pre-pass: a screening report's conditions block
    # can carry numpy scalars, and a file that raises TypeError at the last
    # step is worse than one that writes the number.
    put("section_json",
        None if rec is None else json.dumps(rec, indent=1, default=_jsonable),
        "there is no section to describe")
    return {"stem": stem, "files": files, "absent": absent}


def section_export_plan(S: dict, surface: str = "main") -> dict:
    """WHERE a section export would land and what it would replace.

    The stage-2 twin of :func:`export_plan`, and the same contract: the names
    come from the writers' own table, and ``existing`` is what a Save would
    overwrite — "choose the name" is not a real choice if the shell will not
    say when the name is already taken.
    """
    d = export_dir(S)
    got = section_export_files(S, surface)
    names = {k: d / v["name"] for k, v in got["files"].items()}
    return {"dir": d, "stem": got["stem"], "typed": export_stem_raw(S),
            "names": names, "absent": got["absent"],
            "existing": sorted(p.name for p in names.values() if p.exists())}


def write_section_export(S: dict, surface: str = "main",
                         kinds=None) -> list:
    """Write the section's files into the chosen folder. Returns the paths.

    The folder is created here and nowhere else — typing half a path into the
    field must not litter the filesystem (:func:`set_export_dir`). A ``kind``
    that cannot be written raises rather than being skipped silently: the
    caller asked for a named file and is entitled to hear that it does not
    exist.
    """
    got = section_export_files(S, surface)
    want = tuple(SECTION_EXPORT_KINDS if kinds is None else kinds)
    missing = [k for k in want if k not in got["files"]]
    if missing and kinds is not None:
        raise ValueError(got["absent"].get(missing[0], f"cannot write "
                                                       f"{missing[0]}"))
    d = export_dir(S)
    d.mkdir(parents=True, exist_ok=True)
    out = []
    for kind in want:
        f = got["files"].get(kind)
        if f is None:
            continue
        path = d / f["name"]
        path.write_text(f["text"], encoding="utf-8")
        out.append(path)
    return out
