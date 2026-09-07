"""aerobo — NiceGUI desktop shell ("AeroBO Studio").

THIN presentation layer, same architecture rule as the Streamlit shell
(``gui/app.py``): ZERO physics and ZERO optimiser logic live here. Every
action builds an ``aerobo.api.RunConfig`` and goes through ``aerobo.api.run``
— the exact API the experiment harnesses and the pure-API twin test drive.
Presentation-only geometry (drawing a trapezoid, a NACA 4-digit section for
the 3-D view, rotating a chord line by a twist angle) is the only "math"
allowed in this file.

What this shell adds over the Streamlit one:

* an AIRCRAFT-BUILDER design page: choose the lifting system (single wing /
  tandem), winglets, airfoil treatment, planform freedom, CHORD LAW,
  flight-state freedom and an H-tail — the shell derives the matching solver
  problem and is explicit about which combinations exist (capability notes).
  The chord law is the one freedom that is ON before anything is touched
  (:data:`BUILDER_START`), because a taper ratio alone can only draw a
  straight chord;
* an ALWAYS-EDITABLE mission card (exact legacy defaults pre-filled; only
  the fields you actually change are sent, so an untouched card is
  bit-for-bit the legacy operating point);
* a propeller-slipstream panel with real parameters (diameter, placement,
  thrust CT or velocity ratio, swirl, rotation sense) building the
  ``SlipstreamSpec`` dict the physics consumes;
* true background runs with LIVE streaming plots (per-eval objective,
  GP surrogate residual + uncertainty, acquisition decay, constraint
  margins) fed by the rich ``progress_cb``/``iter_cb`` payloads;
* a run QUEUE with seed batches, cancel, ETA and per-job progress;
* a Results page with bound-riding indicators, planform + 3-D wing view
  WITH real section thickness, spanwise distributions, airfoil section,
  constraint tables, the FULL evaluation log (table + CSV), the raw result
  JSON and one-click reproduce-snippet export;
* a Compare page with seed-band overlays and run management;
* presets, dark/light toggle, native macOS window + completion notification.

Run (native window)::

    .venv/bin/python gui/nice_app.py

Run (browser)::

    .venv/bin/python gui/nice_app.py --browser
"""

from __future__ import annotations

import csv
import io
import json
import math
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --------------------------------------------------------------- constants

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:            # editable-install safety
    sys.path.insert(0, str(REPO_ROOT / "src"))

# The 3-D views and the CAD export must draw the SAME surface or one of them
# is lying about what was flown, so the loft lives in the package
# (``aerobo.cad``) and this shell only adds colour. Imported here, after the
# path fix above, rather than inside each helper: these are called per frame.
from aerobo import cad                                # noqa: E402

# WHICH WAY UP a design was solved. A car family models the vehicle mirrored
# (``geometry.MIRRORED_FRAME``), so a picture of its model frame stands the
# car on its roof; the views below ask this and flip the PICTURE. One
# predicate, in the package, because the solvers declare the frame and a
# shell that guessed it from a problem NAME would be a second author.
from aerobo import geometry as _geom_frame            # noqa: E402

# What a number MEANS as presentation data (names, units, and the
# refusal sentinel a convergence axis must not be scaled by). Pure —
# no UI, no physics — so the V1 shell and the V3 stages read one
# catalogue instead of two.
from gui import metrics as _metrics                   # noqa: E402

RESULTS_DIR = REPO_ROOT / "results" / "gui_runs"
PRESETS_PATH = RESULTS_DIR / "presets.json"

ACCENT = "#0ea5e9"        # sky-500  (primary / best-so-far)
GOOD = "#10b981"          # emerald  (feasible)
BAD = "#ef4444"           # red      (infeasible / violated)
WARN = "#f59e0b"          # amber    (bound-riding, warnings)
MUTED = "#94a3b8"         # slate-400 (axis text — readable on dark AND light)
GRID = "rgba(148,163,184,0.18)"
BAND = "rgba(14,165,233,0.18)"

ACQF_CHOICES = ("logei", "ucb", "qmes", "qlognei")


# ================================================== aircraft-builder mapping
# Pure, testable choice -> solver-problem mapping. The builder controls are
# honest about the solver capability matrix: exactly ONE "speciality" beyond
# the plain trim wing can be active (each speciality IS its own problem), and
# derive_problem states what was ignored if choices ever conflict.

BUILDER_DEFAULTS: dict = {
    "medium": "air",        # air | water | track (car rear wing: inverted
    #                         wing over a rigid ground plane — carwing.py)
    "system": "single",     # single | tandem
    "winglets": "none",     # none | free | capped  (span accounting)
    "winglet_type": "canted",   # canted | vertical | raked  (cant band; see
    #                             objective.WINGLET_TYPES) | blended |
    #                             blended_wing (transition families — they
    #                             select a PROBLEM, not a sub-band). Paired
    #                             with ``winglets`` through WINGLET_OPTIONS.
    "winglet_blend_frac": None,  # the tip device's OWN blend, as a VALUE
    #                             (api.WINGLET_BLEND_KEY): how much of the
    #                             device's arc turns out of the wing plane.
    #                             None = the sharp corner every published
    #                             winglet run flies. NOT a speciality key: it
    #                             changes the shape a winglet problem draws,
    #                             never the design vector, which is what lets
    #                             "blended" be a tip-device SHAPE costing the
    #                             same two variables as a canted one
    #                             (WINGLET_SHAPES). The three problems that
    #                             DESIGN the blend keep their own vectors and
    #                             are selected by ``winglet_type`` instead.
    "blend_shape": "spiral",    # blended winglet only: how the device turns
    #                             out of the wing plane (geometry.BLEND_SHAPES).
    #                             The clothoid is the GUI default: it is the
    #                             crease-free law with the gentlest elbow
    #                             (peak curvature 4/3 of the mean, against
    #                             the smoothstep's 3/2). The library default
    #                             stays "arc" so every published run is
    #                             bit-for-bit, which is why this travels as
    #                             an EXPLICIT flag.
    "planform": "fixed",    # fixed | free | wing_loading | aircraft.
    #                         "free" is the SIZE MODIFIER (sizing.py): span
    #                         and area become design variables on WHATEVER
    #                         family the other choices select, weight-coupled,
    #                         with a root-bending stress constraint.
    #                         "wing_loading" is the OTHER size modifier
    #                         (api "size_ws"): the area is not a design
    #                         variable at all — it follows the mission's
    #                         chosen W/S through the same weight loop — so
    #                         the SPAN alone is searched, inside a band the
    #                         user states. Same three costs, one fewer
    #                         question re-opened.
    #                         "aircraft" selects the published, calibrated
    #                         6-D weight-coupled sizing problem instead (it
    #                         frees t/c as well and carries the frozen §
    #                         results), which is why it is a family value
    #                         rather than the modifier.
    "span_m": None,         # fixed planform only: the wing SIZE the user
    "area_m2": None,        # chose (api.PLANFORM_KEYS). None = the solver's
    #                         own published size, and no flag is sent, so an
    #                         untouched card is bit-for-bit. NOT speciality
    #                         keys: a size is a value, never a design variable
    #                         (the vector reshapes the planform, it never
    #                         resizes it).
    "chord": "fixed",       # fixed (straight taper) | free (cubic chord law
    #                         on top of the taper, area held — geometry.py).
    #                         "fixed" HERE is the neutral reset value, not
    #                         the opening one: it has to be a value every
    #                         family can honour (choices_consistent trusts it
    #                         without asking the registry, and the 2-D
    #                         section problem has no chord-law twin). What a
    #                         page OPENS on is BUILDER_START, which is this
    #                         dict with the chord law switched ON.
    "airfoil": "fixed",     # fixed | tc_sweep | coupled | section_only |
    #                         section_wing (CST section + wing planform)
    "flight": "fixed",      # fixed | free  (free = V + altitude design vars)
    # --- car rear wing (medium = track). NOT speciality keys: they refine
    # the car problem rather than selecting a different solver family.
    # WHERE THE LOAD LEAVES THE WING, in one boolean — and therefore which
    # family flies. True: the plates carry the car, at the tips, so they are
    # DESIGNED (chord, thickness, toe, and the reach-the-car constraint) and
    # the card asks their section and their root blend. False: a swan-neck
    # pylon pair carries near the centreline and the plate is a fence of free
    # height, which is when the plate becomes a TIP DEVICE with a shape menu
    # of its own (CAR_TIP_SHAPE_LABELS).
    #
    # It opens on True because that is what a rear wing IS: the thing bolted
    # to the car is the plate. The consequence, stated once here because it
    # is the one invariant this breaks, is that an untouched track card no
    # longer reproduces this repo's published plain-family car numbers — the
    # published runs are the pylon answer without its strut flags, and they
    # are reachable through api.
    "car_endplates": True,
    # NOTE there is no "car_free_area". The reference area is a design
    # variable on every car family — how big the wing is is a design question
    # exactly as how wide it is — so there is nothing to switch, and its band
    # is the design box's own S_m2 row.
    "car_two_element": False,   # fly a SLOTTED two-element section instead
    #                             of a single one (carwing_multi.py). A
    #                             different FAMILY rather than a flag: the
    #                             flap's chord fraction, its deflection and
    #                             the gap and overlap that place it are four
    #                             new rows of the design vector. The flap is
    #                             in the SECTION, not in the lattice — one
    #                             chordwise panel per strip cannot represent
    #                             a slot — so it costs a 2-D panel solve per
    #                             candidate and about 6x the evaluation time.
    # WHICH plate-borne layout (carwing.MOUNTS): "tips" (the endplates
    # carry, at the tip) | "inboard" (a second pair of sheets carries, 35%
    # out — about 4.5x stiffer for about +13% drag). There is no pylon
    # layout: a rear wing is bolted to the car by the sheets it already has.
    # Left as a plain string rather than read off the family, because
    # BUILDER_DEFAULTS is a literal table a test diffs against the physics.
    "car_mount": "tips",
    # THE PLATE AS A TIP DEVICE (CAR_TIP_SHAPE_LABELS): the same four keys
    # the aircraft's winglet menu uses. None = the family's own, which is
    # "vertical" — the published plate, at 90 deg towards the track.
    "car_tip_shape": None,
    # the CANT the plate leans outboard at [deg from the wing plane].
    # None = the plate's own 90. Every car family charges the outboard
    # projection against its span row, so it is honest on all of them, and it
    # is now asked on BOTH mounts: as the "canted" tip device's own number
    # under the pylons, and as a typed field beside the section and the blend
    # under the plates (_car_plate_controls). This comment used to say "a
    # plate that is carrying the car does not lean", which is not true of any
    # rear wing and was never true of this package — endplate.py has flown
    # the cant since the flag was declared. What WAS true is that its reach
    # constraint read the cant as 90 whatever was flown, so a leaning plate
    # was credited with reach it did not have; that is fixed at the same time
    # (endplate.py), and the field would have been dishonest without it.
    "car_endplate_cant_deg": None,
    # does the plate carry the wing's CHORD LAW past the tip (True) or the
    # wing's TIP chord the whole way (None/False, the published rectangle)?
    # The car's spelling of the aircraft's winglet_chord_follows, offered on
    # the two families whose plate has no chord row of its own.
    "car_endplate_chord_follows": None,
    # NO "car_mount_layout" KEY. The mount is one question and it has one
    # answer stored in one place, which is ``car_endplates`` above: a second
    # key holding the same fact is how two controls come to disagree on
    # screen. :func:`car_mount_layout_of` spells that boolean as the two
    # words the menu shows.
    # THE CIRCUIT this wing is bought for (api.CAR_TRACKS, and "off" for the
    # off mode api.CAR_TRACK_OFF spells). A rear wing's task is a LAP —
    # cartrack.py's docstring makes the same argument mission.py makes for
    # air — and until one is stated there is no mission here at all, only a
    # figure of merit. "off" is the published behaviour: every car run ever
    # published in this repo was scored without a circuit, so an untouched
    # card is bit-for-bit that run.
    #
    # It is a CHOICE and not a value because it decides what the Maximise
    # menu beside it may offer: `laptime` needs a lap to time, and both car
    # problem classes refuse the objective by name without one. Choosing a
    # circuit is therefore a menu-changing act, and writing it must rebuild
    # the card (which is why it is NOT in NUMBER_CHOICE_KEYS).
    "car_track": "off",
    # ...and how many representative speeds that lap is sampled at. Each one
    # is a full re-solve of the wing, so this is the lap's own cost knob and
    # not a modelling preference. None = cartrack's own default, read off the
    # dataclass field by `car_default_track_points`.
    "car_track_points": None,
    "car_endplate_section": "shaped",   # endplate.SECTIONS
    # ...and WHO ANSWERS the plate's two shape numbers: the reader, or the
    # optimiser. "fixed" is the published behaviour and reads the typed field
    # beside it; "free" selects the registry twin that carries the number as a
    # DESIGN ROW instead (api.PLATE_FREEDOMS, api.plate_freedom_name), which
    # is why they are choices and not switches — each one changes the problem
    # and therefore the design box.
    #
    # BOTH default to "fixed", so an untouched card derives the same family it
    # always did, sends the same flags and runs bit-for-bit.
    "car_plate_cant": "fixed",
    "car_plate_blend": "fixed",
    "car_endplate_blend_frac": None,    # the wing/PLATE corner, as a value
    #                             (api CAR_ENDPLATE_KEYS "blend_frac"): how
    #                             much of the plate's height is spent turning
    #                             out of the wing plane. This is the track's
    #                             answer to the tip-device blend every other
    #                             family has — the plate IS this wing's tip
    #                             device — and the solver has priced it since
    #                             endplate.py was written; nothing asked for
    #                             it. Only the DESIGNED-endplate family
    #                             charges the corner (carwing's plain fence
    #                             counts no junctions), so a blend on the
    #                             other one would be lift for free.
    #                             None = the sharp corner every published car
    #                             run flies, bit-for-bit.
    # the PLATE's chord band in metres (api CAR_ENDPLATE_KEYS
    # "endplate_chord_*_m"). Its other band is the design box's own
    # endplate_chord_ratio row — a ratio of the wing's TIP chord, which moves
    # with the span, the taper and the chord law, so it cannot state a box the
    # part has to fit. Both are live; the metre band wins, and the breakdown
    # says when it bit. None = unconstrained, the published problem.
    "car_endplate_chord_min_m": None,
    "car_endplate_chord_max_m": None,
    # NOTE there are no span or area BAND keys here either. Both are bands on
    # DESIGN VARIABLES, so they are the design box's b_m and S_m2 rows and are
    # stated only there. They used to be asked in both places, which was a
    # live defect in both directions (api._CAR_SIZE_ROWS carries the two
    # measurements), and api.check_flags refuses the old flags now.
    # WHAT THE RUN MAXIMISES (carwing.CAR_OBJECTIVES minus "cz", which is
    # refused against a designed area). None = api.CAR_DEFAULT_OBJECTIVE.
    "car_objective": None,
    # THE DRAG CEILING [N], and there is no menu beside it: the family
    # budgets nothing by default, because an allowance handed to a maximiser
    # is a number the answer rides rather than a limit it respects. None = no
    # drag constraint at all.
    "car_drag_budget_n": None,
    "car_downforce_min_n": None,   # a downforce FLOOR [N]; what makes
    #                             "efficiency" the question a race engineer
    #                             has rather than one answered by a small wing
    # --- tandem pair (system = tandem). The STAGGER, in metres: where the
    # rear wing sits relative to the front one. Configuration, never a design
    # variable — a layout is a decision, not something a run discovers.
    # None = derived from the span in play (the published b/2 and 0.1 b).
    "tandem_dx_m": None,        # rear quarter-chord aft of the front's [m]
    "tandem_dz_m": None,        # rear quarter-chord above the front's [m]
    # ...and how WIDE the rear wing is, in metres. Same kind of answer: the
    # two wings share a fuselage, not a span. None = as wide as the front
    # one, which is the pair every published run flew. Only asked while the
    # planform is fixed — freeing it puts a span row per wing in the box.
    "tandem_b_rear_m": None,    # the rear wing's own span [m]
    # ...and how far AFT OF THE REAR WING the fin stands — the boom. The
    # third length of the same layout, and the one a pair cannot avoid
    # answering: a conventional aeroplane's fin hangs off a body that ends
    # somewhere, and a pair has two wings and a gap. None = the package's
    # measured default (api.TANDEM_FIN_BOOM_KEY / fin.TANDEM_FIN_BOOM_FRAC),
    # 0 = a fin standing on the rear wing's own root.
    "tandem_fin_boom_m": None,  # fin quarter-chord aft of the rear wing [m]
    "tail": False,          # H-tail / elevator on
    # --- tail/elevator configuration. NOT speciality keys: they refine the
    # tail problem rather than selecting a different solver family, so they
    # never participate in the mutual-exclusion reset.
    "tail_type": "conventional",   # conventional | t_tail | v_tail | canard
    "tail_arm": "free",            # free (design variable) | fixed (chosen)
    "tail_arm_m": 5.5,             # chosen distance from the wing [m]
    "tail_height": "fixed",        # fixed (the layout's own height) | free
    "tail_cg_m": None,             # where the CG sits, in the family's own
    #                                frame: metres AFT of the wing's own
    #                                aerodynamic centre (air) or of the
    #                                foil's (water). None = the family's
    #                                calibrated value (tail.X_CG_BY_TYPE /
    #                                hydrotail.X_CG_FRAC). This is the number
    #                                that sets the SIGN of what the second
    #                                surface carries: a CG ahead of the wing
    #                                AC has to be balanced by a DOWNLOAD aft.
    "tail_height_m": None,         # the vertical separation as a stated
    #                                value [m]: above the wing plane in air,
    #                                BELOW the foil (negative) in water. Only
    #                                where the height is not already a design
    #                                variable — None keeps the layout's own.
    #                                (a design variable — wingtail.py). Free
    #                                height selects the NONPLANAR wing+tail
    #                                solver, like every other tail freedom.
    "tail_span_min_m": None,       # what the SECOND surface may MEASURE, in
    "tail_span_max_m": None,       # metres: its span and its chord. The box
    "tail_chord_min_m": None,      # states that surface as an area and an
    "tail_chord_max_m": None,      # aspect ratio, which are the solver's
    #                                variables and not a builder's — a
    #                                fuselage is only so wide and a mould
    #                                only so long. None = off, which is the
    #                                published behaviour (api.TAIL_LIMIT_KEYS
    #                                -> tail.TailLimits).
    "tail_mount": "auto",          # auto | upright | inverted — WHICH WAY UP
    #                                the second surface flies its section.
    #                                auto (the published default) follows the
    #                                load the balance asks for; the other two
    #                                state it, and the run reports the load
    #                                it landed on either way. Named for the
    #                                MOUNTING and not the load, because the
    #                                surface's own camber couple is one of
    #                                the two authors of that load — mirroring
    #                                the section can mirror what it is then
    #                                asked to carry (api.TAIL_MOUNT_KEY).
    "tail_dihedral_deg": 35.0,     # V-tail only
    "fly_section": False,          # fly the section chosen upstream (stage 2
    #                                in V3) instead of searching a thickness.
    #                                On the fixed-table families it is a flag
    #                                that changes no problem; where the
    #                                family SELECTS its section by t/c it
    #                                selects the twin that flies one
    #                                (api.chosen_section_problem).
    "tail_design": "fixed",        # fixed | planform | planform+tip — how
    #                                much of the TAIL is designed. "fixed" is
    #                                the published rectangle at AR 4 (area and
    #                                arm only); "planform" frees its taper,
    #                                aspect ratio and washout (and its own
    #                                chord law where the wing has one);
    #                                "planform+tip" adds a tip device ON THE
    #                                TAIL, which needs the nonplanar solver.
    "wing_cant": "fixed",          # fixed | dihedral | sweep | free — which
    #                                of the WING's own dihedral and
    #                                quarter-chord sweep are STATED values
    #                                (api.WING_CANT_KEYS, default 0/0 — the
    #                                planar unswept wing every published run
    #                                flew) and which are DESIGN-BOX ROWS
    #                                (api.WING_CANTS). Four states because
    #                                they are two questions: the halves let a
    #                                design ask for a neutral point without
    #                                also handing the optimiser a roll lever,
    #                                and the reverse. Each is a different
    #                                dimension and therefore a different
    #                                registered problem, which is why it is a
    #                                choice here and not a flag. The dihedral
    #                                is the only wing-side source of Cl_beta
    #                                that
    #                                holds at any lift: with it fixed at zero
    #                                what is left is sweep's, which goes as
    #                                CL, so the fin and the tip device carry
    #                                the dihedral effect at cruise and no fin
    #                                SIZE makes that spiral mode converge.
    "tail_winglet_dir": None,      # up | down | either — which way the TAIL's
    #                                own tip device may point (a cant BAND,
    #                                api.TAIL_WINGLET_DIR_KEY). None = the
    #                                family's own published band, which is not
    #                                the same in both media: an aircraft tail
    #                                opens on "up" (the aircraft convention),
    #                                a hydrofoil's elevator on the signed band
    #                                (under water the direction is a
    #                                cavitation trade). Mirroring a surface
    #                                mirrors its lift, so which way pays
    #                                follows the sign of the load the trim
    #                                solve puts on it — hence a choice.
    "tail_winglet_type": None,     # vertical | canted(None) — WHAT SHAPE the
    #                                second surface's tip device is, in the
    #                                wing's own vocabulary
    #                                (api.TAIL_WINGLET_TYPE_KEY). None = the
    #                                canted band the family publishes, so an
    #                                untouched card still sends nothing.
    "tail_winglet_blend_frac": None,   # ...and whether it is BLENDED, as this
    #                                surface's OWN value
    #                                (api.TAIL_WINGLET_BLEND_KEY) rather than
    #                                the wing's — shaping one surface's device
    #                                may not reshape the other's.
    "tail_control": "stabilator",  # stabilator | elevator
    "tail_elevator_chord": 0.30,   # elevator chord / tail chord
    "tail_fin_drag": False,        # charge the fin's parasite drag
}

#: What a fresh page OPENS on — deliberately NOT the same dict.
#:
#: ``BUILDER_DEFAULTS`` has a second job that constrains what may go in it: it
#: is the NEUTRAL value every control resets to, which is why
#: :func:`choices_consistent` treats a control holding it as legal without
#: asking the registry. That shortcut is only sound while each default value
#: exists on EVERY family — "straight taper" always does; a chord law does
#: not (the pure 2-D section problem has no planform to reshape). So the
#: opening state travels separately, and goes through
#: :func:`normalise_choices` like any other state (:func:`start_choices`).
#:
#: It opens on the POLYNOMIAL CHORD LAW, for the same reason the airfoil
#: optimiser opens on one (:data:`OPT_CHORD_DEFAULT`): a taper ratio alone can
#: only ever draw a STRAIGHT chord, whatever ratio is chosen, and the loading
#: that minimises induced drag is not straight — so opening on "fixed" left
#: the biggest planform lever switched off behind a control most sessions
#: never touched. Like the clothoid ``blend_shape``, this is a GUI starting
#: point ONLY: every ``aerobo.api`` default is untouched, so scripted studies
#: and published numbers still reproduce bit-for-bit.
BUILDER_START: dict = dict(BUILDER_DEFAULTS, chord="free")

# One winglet control folds the span-accounting (free/capped) and the cant
# band (objective.WINGLET_TYPES) into a single honest menu: a raked wingtip is
# ALWAYS span-capped (a free-span rake reports a fabricated L/D), and a vertical
# fence has ~zero horizontal projection so free span is the honest accounting.
# Each option key decodes to (winglets, winglet_type).
WINGLET_OPTIONS: dict[str, tuple[str, str]] = {
    "none": ("none", "canted"),
    "canted_free": ("free", "canted"),
    "canted_capped": ("capped", "canted"),
    "vertical": ("free", "vertical"),
    "raked": ("capped", "raked"),
    # blended is a TRANSITION shape, not a cant band: the arc's length is a
    # design variable of its own and the corner's interference drag is
    # charged, so it selects a different problem rather than a sub-band.
    "blended": ("capped", "blended"),
    # ...and the wing-side blend frees a SECOND transition variable: how far
    # inboard of the tip the turn starts. Confined to the winglet the turn
    # has at most blend_frac * h of arc, so its radius stays a fraction of a
    # tip chord and the junction still reads as a corner.
    "blended_wing": ("capped", "blended_wing"),
}

WINGLET_OPTION_LABELS = {
    "none": "none",
    "canted_free": "canted — free span extension",
    "canted_capped": "canted — span-capped",
    "vertical": "vertical fence (near-90° cant)",
    "raked": "raked wingtip (low cant, span-capped)",
    "blended": "blended root transition (span-capped)",
    "blended_wing": "blended INTO THE WING (span-capped)",
}

#: winglet types whose problem designs a root TRANSITION (and so offers the
#: transition-shape menu). Both are span-capped and charge junction drag.
BLENDED_TYPES = ("blended", "blended_wing")


def winglet_option_key(ch: dict) -> str:
    """Current (winglets, winglet_type) -> the composite menu key."""
    pair = (ch.get("winglets", "none"), ch.get("winglet_type", "canted"))
    for key, val in WINGLET_OPTIONS.items():
        if val == pair:
            return key
    return "none" if pair[0] == "none" else "canted_free"


# ------------------------------------------------ the SHAPE-only tip menu
# The seven-entry menu above is the STUDY menu: it exposes the span
# accounting (does the tip device extend the span, or is the wing shrunk to
# pay for it?) and the two transition families as separate answers, because
# comparing them IS the winglet study.
#
# A design session is asking a different question — what shape is the tip
# device? — and there are three answers: a vertical fence, a canted winglet,
# or a blended one. This table says which (winglets, winglet_type) pair each
# shape means, in preference order, so the SPAN ACCOUNTING is chosen by the
# honest rule rather than by the user:
#
#   * a vertical fence has ~zero horizontal projection, so free span is the
#     honest accounting and there is nothing to cap;
#   * a canted or blended device DOES project, so it is span-capped wherever
#     a capped variant exists — that is the comparison in which a tip device
#     has to earn its keep. The water families are scored span-free on
#     purpose (their trade is the cant's SIGN, not its projection), so they
#     fall through to the free-span pair.
#
# ...and the BLEND is a value, not a dimension (api.WINGLET_BLEND_KEY): the
# blended entry is the canted problem with the transition drawn on it, so it
# costs the same two design variables — the height and the cant — that item
# "only winglet h frac and cant deg" asks for.
WINGLET_SHAPES: dict[str, tuple] = {
    "none": (("none", "canted"),),
    "vertical": (("free", "vertical"),),
    "canted": (("capped", "canted"), ("free", "canted")),
    "blended": (("capped", "canted"), ("free", "canted")),
}

WINGLET_SHAPE_LABELS = {
    "none": "none",
    "vertical": "vertical fence (90°)",
    "canted": "canted winglet",
    "blended": "blended winglet",
}

WINGLET_SHAPE_NOTES = {
    "none": "A plain tip. Everything a tip device could buy shows up as the "
            "difference from this.",
    "vertical": "A near-90° fence: almost no horizontal projection, so it "
                "adds height without adding span. Two design variables — how "
                "tall it is, and its cant.",
    "canted": "A winglet at a free cant angle, span-capped where the family "
              "has a capped variant: the wing shrinks to pay for the "
              "device's horizontal projection, so the device has to earn its "
              "keep rather than being a longer wing in disguise.",
    "blended": "The same two design variables, with the root corner replaced "
               "by a real transition: the device turns out of the wing plane "
               "over part of its arc, and the corner's interference drag is "
               "charged (junction.py) — without that charge a blend is only "
               "a differently drawn wake. The blend FRACTION is a build "
               "decision here, not a design variable.",
}

#: how much of the device's arc length the fixed blend spends turning out of
#: the wing plane (geometry.WINGLET_BLEND_BOUNDS: 0 = sharp corner, 1 = pure
#: arc). Half is the shape a blended winglet is usually drawn with — enough
#: arc for the fillet radius to be a real fraction of the tip chord, short
#: enough that the device still ends at its own cant angle.
BLEND_FRAC_DEFAULT = 0.5

#: why a shape is not offered. Same contract as WINGLET_OPTION_WHY: the
#: reason has to be true of whatever configuration is on screen.
WINGLET_SHAPE_WHY = {
    "vertical": "this family's tip-device solver chooses its own cant band — "
                "it has no winglet_type flag, so a 90° fence cannot be asked "
                "for here",
    "canted": "no tip-device solver for this configuration",
    "blended": "this family has no blended tip device — its solver draws the "
               "wing/device corner as a corner",
}

#: WHAT SHAPE the tip device is, as a flag: ``objective.WINGLET_TYPES``
#: narrows the cant row onto the shape's own band (``vertical`` is 84-90°
#: against a family's full -90…90). A family that does not declare the key
#: picks its own band, so the word never reaches the solver — which is why
#: the SHAPE menu has to ask the registry for it, exactly as the blend does.
WINGLET_TYPE_KEY = "winglet_type"


def _shape_flag_available(ch: dict, pair: tuple, key: str) -> bool:
    """Does the problem these choices derive HONOUR flag ``key``?

    Asked of the derived family rather than of the pair, because both shape
    flags are value flags: the pair chooses the SOLVER and the flag then
    narrows something inside it, so a family that never declares the key runs
    the pair's default and the shape on screen is not the shape flown.
    ``api.accepted_flags`` is the same test ``api.sanitise_flags`` applies on
    the way to the Run button — a key it drops there is a control the menu
    should not have offered.
    """
    from aerobo import api

    trial = dict(ch)
    trial["winglets"], trial["winglet_type"] = pair
    trial.pop("winglet_blend_frac", None)
    name, _ = derive_problem(trial)
    if name not in api.PROBLEM_SPECS:
        return False
    return key in api.accepted_flags(name)


def _blend_flag_available(ch: dict, pair: tuple) -> bool:
    """Does the problem these choices derive DECLARE the fixed-blend flag?"""
    from aerobo import api

    return _shape_flag_available(ch, pair, api.WINGLET_BLEND_KEY)


def winglet_shape_pair(ch: dict, shape: str) -> tuple | None:
    """The (winglets, winglet_type) pair ``shape`` means here, or None.

    Asked of the registry entry by entry (:func:`option_available`), so a
    family that carries only a free-span device gets the free-span pair and
    one that carries none at all gets nothing — the menu can then leave the
    entry out with a reason instead of offering a shape no solver draws.
    """
    for pair in WINGLET_SHAPES.get(shape, ()):
        key = next((k for k, v in WINGLET_OPTIONS.items() if v == pair), None)
        if key is None or not option_available(ch, "winglets", key):
            continue
        if shape == "blended" and not _blend_flag_available(ch, pair):
            continue
        # ...and the FENCE gets the same test, for the same reason. The pair
        # ``("free", "vertical")`` selects an ordinary free-span winglet
        # solver everywhere a tip device exists, and the 90° is carried by
        # the ``winglet_type`` FLAG on top of it. On the families that do not
        # declare the key the flag is dropped by ``api.sanitise_flags`` and
        # the run searches the family's full cant band (-90…90 on the
        # hydrofoil) instead of WINGLET_TYPES['vertical']'s 84-90 — a
        # SUPERSET of the band asked for, with the select still reading
        # "vertical fence (90°)" because winglet_shape_key reads the word
        # back out of the state. Measured over the stage-3 sweep: 14 derived
        # families offered the entry and dropped the flag (every hydrofoil
        # one and every nonplanar tandem one).
        if shape == "vertical" and not _shape_flag_available(
                ch, pair, WINGLET_TYPE_KEY):
            continue
        return pair
    return None


def winglet_shapes(ch: dict) -> dict:
    """The tip-device SHAPES this configuration actually has a solver for."""
    return {k: v for k, v in WINGLET_SHAPE_LABELS.items()
            if k == "none" or winglet_shape_pair(ch, k) is not None}


def winglet_shape_key(ch: dict) -> str:
    """Which SHAPE the current choices hold.

    The blend is read off the VALUE, not off the winglet type: the blended
    entry selects the canted problem plus ``winglet_blend_frac``, so a
    configuration built in V1 with one of the blend-DESIGNING types maps onto
    the same word — the shape on screen is the shape that will be flown.

    The FENCE is read the same way, and for the same reason: the word
    ``vertical`` sits in the state whether or not the derived family declares
    ``winglet_type``, and where it does not the flag is dropped on the way to
    the run and the family's own cant band is searched. Reading it back out of
    the state regardless is what let the select go on saying "vertical fence
    (90°)" over a search that was flying -90…90; a configuration carried into
    such a family (choose the fence in air, then switch the medium to water)
    holds the word but flies a canted device, so that is what this says.
    """
    if ch.get("winglets", "none") == "none":
        return "none"
    if ch.get("winglet_type") in BLENDED_TYPES or ch.get(
            "winglet_blend_frac"):
        return "blended"
    if ch.get("winglet_type") == "vertical" \
            and winglet_shape_pair(ch, "vertical") is not None:
        return "vertical"
    return "canted"


def set_winglet_shape(ch: dict, shape: str) -> None:
    """Write a SHAPE into the choices (the value + the pair it implies)."""
    if shape not in WINGLET_SHAPES:
        raise ValueError(
            f"unknown tip-device shape {shape!r}; choose from "
            f"{list(WINGLET_SHAPES)}. (The seven-entry study menu speaks a "
            f"different vocabulary — see WINGLET_OPTIONS.)")
    pair = winglet_shape_pair(ch, shape) or WINGLET_SHAPES[shape][0]
    ch["winglets"], ch["winglet_type"] = pair
    ch["winglet_blend_frac"] = (BLEND_FRAC_DEFAULT if shape == "blended"
                                else None)


BLEND_SHAPE_LABELS = {
    "spiral": "clothoid (smooth, gentlest elbow)",
    "smooth": "smoothstep (smooth, tightest middle)",
    "arc": "constant-radius fillet",
}

BLEND_SHAPE_NOTES = {
    "spiral": "The turn's curvature ramps up from zero, holds, and ramps "
              "back to zero — the road/rail transition spiral. Crease-free "
              "like the smoothstep, but it does not dump the whole turn into "
              "the middle: peak curvature is 4/3 of the mean against the "
              "smoothstep's 3/2, so the tightest point of the elbow is a "
              "third gentler at the same arc length and cant.",
    "smooth": "The turn out of the wing plane starts and ends with ZERO "
              "curvature, so the wing runs into the winglet with no crease "
              "at either end — a genuinely smooth horizontal-to-vertical "
              "transition. Same arc length, same cant: only the shape moves. "
              "It pays for the crease-free ends with a TIGHTER middle (peak "
              "curvature 3/2 of the mean), which is why a short blend drawn "
              "this way can still look abrupt.",
    "arc": "A circular fillet: the tangent is continuous but the curvature "
           "jumps 0 → 1/R at the junction and back at the top, so both ends "
           "are still creases. This is the legacy blend shape.",
}


def winglet_flags(ch: dict) -> dict:
    """Builder choices -> the winglet flags the api reads.

    Empty for the legacy canted band, so an untouched winglet run stays
    bit-for-bit. The BLENDED entries send their transition SHAPE explicitly
    (geometry.BLEND_SHAPES) — a value flag, not a cant band and not a design
    variable — because the GUI defaults it to the clothoid while the library
    default stays "arc".
    """
    if ch.get("winglets", "none") == "none":
        return {}
    wtype = ch.get("winglet_type", "canted")
    if wtype in BLENDED_TYPES:
        # both blended types are carried by the problem NAME (each has its
        # own design variables), so they must not travel as a cant-band flag
        # the API would reject — see derive_problem. The SHAPE does travel.
        return {"blend_shape": ch.get("blend_shape", "spiral")}
    out: dict = {}
    if wtype != "canted":
        out["winglet_type"] = wtype
    # the FIXED blend (api.WINGLET_BLEND_KEY) — a value on a problem whose
    # vector has no blend row, so the turn law travels with it. Absent, the
    # dict is empty and the run is the published one, bit-for-bit.
    blend = ch.get("winglet_blend_frac")
    if blend:
        from aerobo import api

        out[api.WINGLET_BLEND_KEY] = float(blend)
        out["blend_shape"] = ch.get("blend_shape", "spiral")
    return out


def objective_winglet_note(wtype: str) -> str:
    """One-line physics note for a winglet type (objective.WINGLET_TYPES)."""
    try:
        from aerobo import objective
        spec = objective.WINGLET_TYPES.get(wtype)
        return spec["note"] if spec else ""
    except Exception:       # noqa: BLE001 — a note is cosmetic, never fatal
        return ""


TAIL_HEIGHT_LABELS = {
    "fixed": "the layout's own",
    "free": "optimise it",
}

#: how much of the TAIL the search designs (api.TAIL_DESIGNS)
TAIL_DESIGN_LABELS = {
    "fixed": "area and arm only (rectangle, AR 4)",
    "planform": "its planform too (taper, AR, washout)",
    "planform+tip": "its planform and its own tip device",
}

TAIL_DESIGN_NOTES = {
    "fixed": "The published stabiliser: a rectangle at aspect ratio 4 whose "
             "only freedoms are its area and how far aft it sits.",
    "planform": "The tail becomes a surface in its own right — taper, aspect "
                "ratio and washout join the design vector, and it takes its "
                "own chord law wherever the wing has one. Its INCIDENCE "
                "stays the trim unknown, so the twist freedom is the "
                "washout: a root twist would be that same angle twice.",
    "planform+tip": "...and it carries a tip device of its own (height and "
                    "cant), panelised by the same VLM code as the wing's. "
                    "Needs the nonplanar wing+tail solver — a lifting line "
                    "is planar and cannot hold one.",
}


def tail_design_options(ch: dict) -> dict:
    """The tail-design entries this configuration can actually reach."""
    return {k: v for k, v in TAIL_DESIGN_LABELS.items()
            if k == "fixed" or option_available(ch, "tail_design", k)}


# ------------------------------------ the SECOND SURFACE's tip device, shaped
#
# The wing's tip device is asked as a SHAPE (WINGLET_SHAPES above): none, a
# near-vertical fence, a canted winglet, or the same device with its root
# corner replaced by a real transition. The second surface's was a yes/no,
# which asked a different — and smaller — question about the same kind of
# fitting. It is asked the same way here, in the same words, and it maps onto
# the same three kinds of answer:
#
#   * WHETHER there is one is ``tail_design`` (the registry selects the
#     family: "planform+tip" against "planform");
#   * WHAT SHAPE it is narrows the cant band (api.TAIL_WINGLET_TYPE_KEY,
#     wingtail.tail_cant_bounds_for) — a VALUE, exactly like the wing's;
#   * whether it is BLENDED is a VALUE too (api.TAIL_WINGLET_BLEND_KEY), and
#     its own rather than the wing's, so shaping one surface's device does
#     not silently reshape the other's.
#
#: shape -> the winglet_type it means for the surface (None = the family's
#: own canted band, i.e. nothing travels)
TAIL_TIP_TYPES: dict[str, str | None] = {
    "none": None,
    "vertical": "vertical",
    "canted": None,
    "blended": None,
}

TAIL_TIP_SHAPE_LABELS = {
    "none": "none",
    "vertical": "vertical fence (90°)",
    "canted": "canted winglet",
    "blended": "blended winglet",
}

TAIL_TIP_SHAPE_NOTES = {
    "none": "A plain tip on the second surface. Everything a device could "
            "buy there shows up as the difference from this.",
    "vertical": "A near-vertical fence: height without horizontal "
                "projection. Two design variables — how tall it is and its "
                "cant — with the cant band narrowed to 84–90° of whichever "
                "side the surface's own load asks for.",
    "canted": "A winglet at a free cant, panelised by the same VLM code as "
              "the wing's. Which SIDE it sits on is not asked: the surface "
              "carries whatever lift trims the aircraft, and a mirrored "
              "surface makes mirrored lift, so the side is read off each "
              "candidate's own trim solution.",
    "blended": "The same two design variables with the root corner replaced "
               "by a real transition, and the corner's interference drag "
               "charged (junction.py). The blend FRACTION is a build "
               "decision here, not a design variable — and it is this "
               "surface's own, so the wing's device keeps whatever shape it "
               "was given.",
}

TAIL_TIP_SHAPE_WHY = {
    "vertical": "a tip device on the second surface needs the nonplanar "
                "wing+tail solver, which this configuration has no variant "
                "of",
    "canted": "a tip device on the second surface needs the nonplanar "
              "wing+tail solver, which this configuration has no variant of",
    "blended": "this family's solver draws the surface/device corner as a "
               "corner, so there is no transition for a blend to shape",
}


def tail_tip_shape_key(ch: dict) -> str:
    """Which SHAPE the second surface's tip device currently holds."""
    if ch.get("tail_design") != "planform+tip":
        return "none"
    if ch.get("tail_winglet_blend_frac"):
        return "blended"
    return ("vertical" if ch.get("tail_winglet_type") == "vertical"
            else "canted")


def tail_tip_shapes(ch: dict) -> dict:
    """The shapes the second surface's tip device can actually be built as.

    Asked of the registry entry by entry, like every other menu here: the
    device itself needs a ``planform+tip`` solver, and the blend needs that
    solver to declare the blend flag.
    """
    from aerobo import api

    out = {"none": TAIL_TIP_SHAPE_LABELS["none"]}
    if not (option_available(ch, "tail_design", "planform+tip")
            or ch.get("tail_design") == "planform+tip"):
        return out
    out["vertical"] = TAIL_TIP_SHAPE_LABELS["vertical"]
    out["canted"] = TAIL_TIP_SHAPE_LABELS["canted"]
    trial = dict(ch, tail_design="planform+tip")
    trial.pop("tail_winglet_blend_frac", None)
    spec = api.PROBLEM_SPECS.get(derive_problem(trial)[0])
    if spec is not None and api.TAIL_WINGLET_BLEND_KEY in spec.flags:
        out["blended"] = TAIL_TIP_SHAPE_LABELS["blended"]
    return out


def set_tail_tip_shape(ch: dict, shape: str) -> None:
    """Write a SHAPE for the second surface's tip device into the choices."""
    if shape not in TAIL_TIP_SHAPE_LABELS:
        raise ValueError(
            f"unknown tip-device shape {shape!r} for the second surface; "
            f"choose from {list(TAIL_TIP_SHAPE_LABELS)}")
    if shape == "none":
        if ch.get("tail_design") == "planform+tip":
            ch["tail_design"] = "planform"
        ch["tail_winglet_type"] = None
        ch["tail_winglet_blend_frac"] = None
        return
    ch["tail_design"] = "planform+tip"
    ch["tail_winglet_type"] = TAIL_TIP_TYPES[shape]
    ch["tail_winglet_blend_frac"] = (BLEND_FRAC_DEFAULT if shape == "blended"
                                     else None)


#: what a second surface OPENS on the moment it is added. A stabiliser is a
#: SURFACE, not a fitting: wherever a solver can design its taper, aspect
#: ratio and washout, that is the honest default, and the published rectangle
#: at AR 4 is the fallback for a family that has no designed-tail variant.
#:
#: Deliberately NOT in :data:`BUILDER_DEFAULTS` — that dict is the NEUTRAL
#: reset every control falls back to, and ``choices_consistent`` trusts a
#: control holding a default without asking the registry, which is only sound
#: while the value exists on every family (see :data:`BUILDER_START`). Nor is
#: it in ``BUILDER_START``: with no second surface on the vehicle there is no
#: tail to design, so a starting value would be normalised away before the
#: user ever switched the surface on. It is applied WHEN the surface appears.
TAIL_DESIGN_START = "planform"


def tail_design_start(ch: dict) -> str:
    """The design freedom a newly added second surface opens on.

    Asked of the registry (:func:`option_available`), like every other menu
    in this module, so a family with no designed-tail solver opens on its
    published rectangle instead of holding a value its own control cannot
    offer.
    """
    trial = dict(ch, tail=True)
    return (TAIL_DESIGN_START
            if option_available(trial, "tail_design", TAIL_DESIGN_START)
            else BUILDER_DEFAULTS["tail_design"])


TAIL_DESIGN_WHY = {
    "planform": "no solver designs the tail's own planform in this "
                "configuration",
    "planform+tip": "a tip device on the TAIL needs the nonplanar wing+tail "
                    "solver, which this configuration has no variant of",
}


def tail_height_note(ch: dict) -> str:
    """Honest one-liner under the tail-height control."""
    if ch.get("tail_type") == "t_tail":
        return ("A T-tail's height IS the fin span implied by its own Raymer "
                "sizing, so it cannot also be a design variable — switch to "
                "the conventional layout to optimise the height.")
    if ch.get("tail_height") == "free":
        return ("Adds the vertical distance to the design vector. Out of the "
                "wing's trailing sheet the tail sees less downwash and the "
                "neutral point moves aft; the box floors at the measured "
                "0.05 b where the tail is clear of the sheet. INVISCID only "
                "— this wake is rigid, so no dynamic-pressure deficit and no "
                "deep-stall pitch-up.")
    return ("The tail sits at the height its layout implies: 0.05 b (the "
            "measured kernel-regularisation clearance) for an aft tail or a "
            "canard, the fin span for a T-tail.")


TAIL_TYPE_LABELS = {
    "conventional": "conventional (aft, on the fuselage)",
    "t_tail": "T-tail (raised onto the fin top)",
    "v_tail": "V-tail / butterfly (no separate fin)",
    "canard": "canard (surface ahead of the wing)",
}

TAIL_TYPE_NOTES = {
    "conventional": "Baseline: tail 0.05 b above the wing plane — the "
                    "measured kernel-regularisation height.",
    "t_tail": "Height comes from the Raymer fin sizing (h = √(AR_vt·S_vt), "
              "0.87–1.41 m over the arm range), so the tail sits out of the "
              "wing's trailing sheet and sees less downwash. Inviscid "
              "relief ONLY: this wake is rigid and planar, so no dynamic-"
              "pressure deficit, no roll-up and no deep-stall pitch-up.",
    "v_tail": "Modelled as the equivalent flat tail of area S_t·cos²Γ (a "
              "panel at dihedral Γ both sees and returns cos Γ), while "
              "profile drag is still charged on the FULL panel area — "
              "dihedral costs pitch effectiveness without saving wetted "
              "area. The fin it deletes is charged on every other layout "
              "now, so the benefit is in the number too: measured on the "
              "tail family, −0.04 % L/D to the dihedral and +6.4 % from "
              "having no fin.",
    "canard": "Surface ahead of the wing (x < 0); the kernel handles it "
              "natively. The CG moves to the separately calibrated "
              "x_cg = −0.40 m — with the aft-tail CG every canard would "
              "report a plausible L/D while being unconditionally unstable.",
}

# Speciality keys select a SOLVER FAMILY, so they are mutually exclusive
# unless a combined solver exists (specials_compatible). "chord" and "flight"
# are NOT among them: they are MODIFIERS (api.MODIFIERS). The chord law only
# reshapes Wing.chord(y) and the flight state only moves the flow state and
# the trim target CL = W/(q S) — both of which every solver in the package
# handles the same way — so each composes with whatever family the other
# choices select (api.CHORD_TWINS / api.FLIGHT_TWINS). They therefore never
# reset anything and are never reset.
#
# "planform" is in the list because ONE of its values ("aircraft") selects a
# family. Its sizing values are modifiers like the two above — see
# PLANFORM_MODIFIER_VALUES, which is what specials_compatible reads.
SPECIAL_KEYS = ("tail", "winglets", "planform", "airfoil")

#: builder choice key + the value that switches each api MODIFIER on. The
#: planform control is three-valued (see BUILDER_DEFAULTS), so the modifier
#: is ON for exactly one of its values — which is why this is a table of
#: (key, value) rather than "anything but the default".
MODIFIER_ON = {"size": ("planform", "free"), "flight": ("flight", "free"),
               "chord": ("chord", "free"),
               "size_ws": ("planform", "wing_loading"),
               "size_ws_free": ("planform", "wing_loading_free")}

#: the planform values that are api MODIFIERS ('size', 'size_ws') rather than
#: a family of their own. Read out of MODIFIER_ON so a new sizing modifier
#: needs no second list to be kept in step with it.
PLANFORM_MODIFIER_VALUES = frozenset(
    val for key, val in MODIFIER_ON.values() if key == "planform")

_SPECIAL_LABEL = {
    # "planform sizing", not "free planform": the value this key is reset
    # FROM is now only ever the aircraft family (see specials_compatible),
    # while normalise_choices can still drop a sizing modifier a family has
    # no twin for — one label has to be true of both
    "tail": "H-tail", "winglets": "winglets", "planform": "planform sizing",
    "airfoil": "airfoil optimisation", "flight": "free flight state",
    "chord": "free chord law",
    # not a speciality either — the LIFTING SYSTEM, which is normalised for
    # the same reason (only the air families fly a pair) and is indexed here
    # bare by the reset notice in V1 and V2
    "system": "tandem pair",
    # not a speciality — a VALUE that normalise_choices drops with the rest
    # of the state when the derived family cannot draw it
    "winglet_blend_frac": "blended tip device",
}

PROBLEM_TO_CHOICES: dict[str, dict] = {
    # the nonplanar pair's entries are GENERATED (tandem_vlm_choices), not
    # listed: the family is a product of three axes now, so a hand-kept
    # table would be seven chances for the mapping to drift and fourteen
    # the day a fourth axis lands.
    "trim wing": {},
    "wing t/c + sweep": {"airfoil": "tc_sweep"},
    "free planform (aircraft)": {"planform": "aircraft"},
    "winglet": {"winglets": "free"},
    "winglet_capped": {"winglets": "capped"},
    "winglet, blended (span-capped)": {"winglets": "capped",
                                       "winglet_type": "blended"},
    "winglet, blended into the wing (span-capped)":
        {"winglets": "capped", "winglet_type": "blended_wing"},
    "winglet + t/c": {"winglets": "free", "airfoil": "tc_sweep"},
    "winglet_capped + t/c": {"winglets": "capped", "airfoil": "tc_sweep"},
    # ...and the pair that demands its section from the pre-optimised
    # (t/c × cl) library instead. Both halves of the round trip matter: this
    # table AND the forward branch in _derive_family. With only one of them
    # the name inverts to the DEFAULT choices, option_available decides every
    # menu entry about the wrong family, and the Airfoil menu would go on
    # saying the coupled library "has no combined solver with this
    # configuration" about the very solver it was holding — the same failure
    # the winglet + section entries above carry a paragraph about.
    "winglet + airfoil (coupled)": {"winglets": "free", "airfoil": "coupled"},
    "winglet_capped + airfoil (coupled)": {"winglets": "capped",
                                           "airfoil": "coupled"},
    # ...and the two that carry a tip device BESIDE the designed section.
    # They invert to "section_wing", not "section_only", because that is what
    # they are: _make_winglet_section_builder builds all three of these from
    # one factory (the wingless one only drops the two tip-device rows), so
    # the vector carries the wing's planform as well as the CST weights. They
    # used to invert to "section_only", and since option_available decides a
    # menu entry by that round trip, every configuration with a tip device
    # was told "no solver shapes a CST section with this configuration" —
    # about the very solver it was holding. V3's "let the wing search reshape
    # it" switch was disabled by it (stages/wing.py), which made the tip
    # device and the designed section read as mutually exclusive.
    "winglet + airfoil (XFOIL)": {"winglets": "free",
                                  "airfoil": "section_wing"},
    "winglet_capped + airfoil (XFOIL)": {"winglets": "capped",
                                         "airfoil": "section_wing"},
    "tail": {"tail": True},
    "tail (fixed arm)": {"tail": True, "tail_arm": "fixed"},
    # the lifting-line family's designed-tail twins (the tail's own planform
    # in the vector; a tip device on it needs the nonplanar family)
    "tail [designed tail]": {"tail": True, "tail_design": "planform"},
    "tail (fixed arm) [designed tail]": {"tail": True, "tail_arm": "fixed",
                                         "tail_design": "planform"},
    "tandem": {"system": "tandem"},
    # the pair with its SECTION THICKNESS searched, one row per wing. Listed
    # here for the reason the block above exists at all: a name this table
    # does not know inverts to the DEFAULT choices, so "tandem + t/c" would
    # round-trip to a single wing and every menu entry decided by that round
    # trip would be answered about the wrong family (the "helper matches a
    # problem NAME exactly" bug class).
    "tandem + t/c": {"system": "tandem", "airfoil": "tc_sweep"},
    "hydrofoil": {"medium": "water"},
    "hydrofoil + winglet": {"medium": "water", "winglets": "free"},
    # ...and car_endplates FALSE, explicitly. It is the MOUNT key now — the
    # plates carry the car or the pylons do — and it opens on True, so a
    # table entry that left it unstated would round-trip the plain family
    # back as the designed-endplate one.
    "car rear wing": {"medium": "track", "car_endplates": False},
    # the SLOTTED pair. Keyed on its own choice and NOT on system="tandem":
    # a two-element wing is one lifting surface with a two-part section, so
    # inverting it to the tandem switch would make stage 2 offer a section
    # for a second surface that does not exist. The track branch of
    # _derive_family still READS system="tandem" as a request for it, and
    # says so, because that is the question a user asking for two surfaces on
    # a car actually means.
    "car rear wing (two-element)": {"medium": "track",
                                    "car_endplates": False,
                                    "car_two_element": True},
    "car rear wing + endplates": {"medium": "track", "car_endplates": True},
    # ...and the same family with the PLATE's own numbers searched instead of
    # stated. Listed rather than derived because they are not modifiers: each
    # is a different base name, and inverting one has to restore the toggle
    # that selected it as well as the family it belongs to.
    "car rear wing + endplates [free cant]": {
        "medium": "track", "car_endplates": True, "car_plate_cant": "free"},
    "car rear wing + endplates [free blend]": {
        "medium": "track", "car_endplates": True, "car_plate_blend": "free"},
    "car rear wing + endplates [free cant, free blend]": {
        "medium": "track", "car_endplates": True,
        "car_plate_cant": "free", "car_plate_blend": "free"},
    "airfoil (section)": {"airfoil": "section_only"},
    "wing + airfoil (XFOIL)": {"airfoil": "section_wing"},
    "wing+airfoil (coupled)": {"airfoil": "coupled"},
}
# NOTE: the free-chord-law twins are NOT listed here. Every one of them is
# "its base problem's choices + chord: free", so they are derived from
# api.CHORD_TWINS (chord_base_of / chord_twin_of) — one mechanism, no list to
# keep in step with the registry.


#: the two plate-borne layouts, in the card's own words. Read off
#: ``carwing.MOUNTS`` rather than pasted, so a layout the family renames or
#: re-stations cannot leave a stale sentence on the card — the same rule
#: ``_car_span_band`` followed for the span band before the design box took
#: that question over.
def car_mount_labels() -> dict:
    """``{key: label}`` for the mount select, off the physics."""
    from aerobo.carwing import MOUNTS

    return {k: v["label"] for k, v in MOUNTS.items()}


def car_default_mount() -> str:
    """The layout a card opens on — the family's own field default."""
    from aerobo.carwing import CarWingProblem

    return str(CarWingProblem.mount)

#: THE MOUNT, AS TWO THINGS — the whole of what any card asks about it.
#:
#: ``carmount.py`` models a continuum: any station on the semi-span, either
#: load path, either side of the wing, any chordwise attachment, any pylon
#: count, any torsional stiffness. All of it is still REACHABLE — ``api``
#: declares every one of those flags and ``api._car_mount_kwargs`` reads them,
#: so a script or a saved run can state the lot. What is narrowed here is the
#: CARD: nine controls to reach two answers is not a question, it is a form.
#:
#: So the shell asks the one question a rear wing actually poses — WHERE DOES
#: THE LOAD LEAVE THE WING: out through the plates it already has, or down
#: through struts near the middle? Everything each answer implies is DERIVED
#: from it (see :func:`car_mount_layout_flags`), not asked again.
#:
#: Both answers keep the endplates. What the choice changes is whether they
#: also CARRY the car — and that is not a refinement, it is what the plate is
#: for. A plate holding a wing up is a designed structure (its chord, its
#: thickness, its toe, and a constraint that it reach the deck); a plate on a
#: pylon-borne wing is a fence, which is exactly a tip device. So this one
#: answer picks the solver family as well as the load path, and everything
#: else the card asks about the plate follows from it.
CAR_MOUNT_LAYOUT_LABELS = {
    "tips": "through the endplates, at the tips",
    "pylons": "on pylons, near the centreline",
}

CAR_MOUNT_LAYOUT_NOTES = {
    "tips": "The plates ARE the load path, gripping at the very tip. They "
            "are there anyway, so this costs no extra wetted area, no new "
            "corner and no footprint on the wing — and the whole span "
            "between the grips sags. A plate that carries the car is a "
            "designed structure rather than a fence, so its chord, its "
            "thickness and its toe become design rows, its section and its "
            "root blend are asked below, and the span row becomes the "
            "OVERALL width with the plates included.",
    "pylons": "A pair of SWAN-NECK struts from the wing down to the car's "
              "deck, gripping at {frac:g} of the semi-span — the same "
              "“near the middle” station the published inboard "
              "layout uses, so the two answers differ in the load path and "
              "not in a second invented number. Swan neck means they come "
              "over the top and land on the PRESSURE side, leaving the "
              "suction surface — which on an inverted wing faces the track, "
              "and is where the downforce is made — clean. That is a "
              "GEOMETRY statement, which is why this layout needs no "
              "suction-loss calibration: no method in this package can "
              "produce one honestly, and an underslung strut would need it. "
              "It pays two wetted struts and two wing/strut corners, and it "
              "holds the wing close to the middle, so the deflection "
              "collapses — the span outboard of each grip hogs while the "
              "middle sags, instead of one long sagging span. The plates "
              "stay at the tips; they just stop carrying anything, which is "
              "what makes them a TIP DEVICE — so their shape is asked as one "
              "below, in the aircraft menu's own four words.",
}


def car_mount_continuum_available(ch: dict,
                                  problem_name: str | None = None) -> bool:
    """Can the family these choices derive to take a MountSpec at all?

    Read off ``api.accepted_flags`` — the registry's own declaration — for
    the same reason :func:`car_lap_available` is: which car families carry a
    mount continuum is a fact about ``CarWingProblem`` having a ``mount_spec``
    field and ``CarWingMultiProblem`` / ``CarWingEndplateProblem`` not having
    one. Offering the pylon layout where it is not declared would produce a
    run ``api.check_flags`` refuses at the Run button.
    """
    from aerobo import api

    if ch.get("medium") != "track":
        return False
    name = problem_name or derive_problem(ch)[0]
    return "car_mount_model" in api.accepted_flags(name)


def car_default_mount_layout() -> str:
    """The layout a card opens on — the plates, because that is what a rear
    wing is bolted to the car by."""
    return "tips" if BUILDER_DEFAULTS["car_endplates"] else "pylons"


def car_mount_layout_of(ch: dict) -> str:
    """Which of the two layouts this configuration is bolted on by.

    DERIVED, from the one key that stores the answer. The mount decides
    whether the plates carry the car, and a plate that carries the car is a
    designed plate — chord, thickness, toe and the reach-the-car constraint —
    which is a different solver family (``_derive_family``). So the mount IS
    ``car_endplates``: one question, one key, and no second control that can
    hold a stale copy of it.
    """
    return "tips" if ch.get("car_endplates") else "pylons"


def car_mount_layout_flags(layout: str) -> dict:
    """What one layout IS, in flags, off ``carmount``'s own numbers.

    ``tips`` sends NOTHING. It is ``CarWingProblem.mount`` untouched, which
    is the rule every control on this card obeys: state nothing, send
    nothing, and the run is bit-for-bit the published one.

    ``pylons`` sends four statements and no others. Every remaining
    ``MountSpec`` field keeps its own default, so the pylon count, the deck
    height, the chordwise attachment (the aerodynamic centre, which winds
    the wing up by exactly nothing) and the torsional stiffness are the
    MODULE's numbers rather than this card's inventions. The station is
    ``carmount.INBOARD_STATION_FRAC`` for the same reason.

    The SIDE is the one place this makes a judgement, and it is deliberate:
    a card with no loss knob can only honestly offer the strut whose zero
    loss is a geometry statement. An underslung pylon sits in the suction
    peak, and no lifting-surface method or reduced-order correlation in this
    package prices that; sending it from a two-entry menu would quietly ship
    an unpriced advantage. ``api`` still accepts ``mount_side='suction'``
    with a ``suction_loss`` the user supplies.
    """
    from aerobo import carmount

    if layout != "pylons":
        return {}
    return {"car_mount_model": "continuum",
            "mount_kind": "pylon",
            "mount_station_frac": float(carmount.INBOARD_STATION_FRAC),
            "mount_side": "pressure"}


def car_mount_flags(ch: dict, problem_name: str | None = None) -> dict:
    """The MOUNT half of :func:`car_flags`.

    Nothing travels where the family declares no ``mount_spec``:
    ``api._car_mount_kwargs`` refuses every refinement there by name (they
    describe a MountSpec and are read by nothing without one), which is a
    refusal the card must never provoke — so the pylon layout is not drawn
    in that case and not sendable from it either.
    """
    if not car_mount_continuum_available(ch, problem_name):
        return {}
    return car_mount_layout_flags(car_mount_layout_of(ch))


def _car_mount_controls(ch: dict, set_choice,
                        problem_name: str | None = None) -> None:
    """WHERE THE LOAD LEAVES THE WING — the first question the card asks.

    This block used to be nine controls deep: a mode select, a load-path
    select, a spelling select for the station, the station itself, a side, a
    pylon count, a pylon drag model, a deck height, a suction loss, a
    chordwise attachment and a torsional stiffness. Every one of them was
    real — ``carmount.py`` reads all of them — and together they asked the
    reader to design a mount before they could choose one. The two entries
    below are the two mounts a rear wing has; the rest is derived from them.

    IT IS ASKED FIRST BECAUSE IT DECIDES THE REST. The answer picks the
    solver family (``_derive_family`` reads the same key), and with it
    whether the plate is a designed structure or a tip device:

    * ``tips`` — the plates carry the car, so they are DESIGNED. Their
      chord, thickness and toe become design rows, and the two things a
      designed plate is built from are asked right here, under the answer
      that created them: its SECTION and its ROOT BLEND. There is no tip
      device menu, because the plate is not a device — it is the mount.
    * ``pylons`` — a swan-neck pair carries near the centreline and the plate
      goes back to being a fence of free height. That is exactly a tip
      device, so the shape menu appears instead (:func:`car_tip_shapes`).

    Neither answer is a switch the reader has to find: they are the same one
    question, and everything each implies follows it on the card.
    """
    from nicegui import ui

    if _car_is_two_element(ch):
        # a SLOTTED wing carries no carmount.MountSpec and has no designed
        # plate solver either (one sheet spans both elements), so neither of
        # the two answers can be flown. The question that IS answerable there
        # is the older two-valued lookup, and it is drawn in its own words —
        # this is not the same menu with an entry missing.
        with ui.row().classes("w-full items-center gap-3 no-wrap"):
            ui.label("Mount").classes("text-xs w-20 shrink-0 opacity-70")
            ui.select(car_mount_labels(),
                      value=ch.get("car_mount", car_default_mount()),
                      on_change=lambda e: set_choice("car_mount", e.value)) \
                .props("outlined dense").classes("grow min-w-0")
        ui.label("A slotted wing carries no carmount.MountSpec and has no "
                 "designed-plate solver, so its mount is the two-valued "
                 "lookup and both of its layouts are plate-borne: the "
                 "endplates grip at the tips, or a second pair grips "
                 "inboard. A pylon has nothing here to reach.") \
            .classes("text-[11px] opacity-60")
        return

    from aerobo import carmount

    layout = car_mount_layout_of(ch)
    with ui.row().classes("w-full items-center gap-3 no-wrap"):
        ui.label("Mount").classes("text-xs w-20 shrink-0 opacity-70")
        ui.select(CAR_MOUNT_LAYOUT_LABELS, value=layout,
                  on_change=lambda e: set_choice("car_endplates",
                                                 e.value == "tips")) \
            .props("outlined dense").classes("grow min-w-0")
    ui.label(CAR_MOUNT_LAYOUT_NOTES[layout].format(
        frac=float(carmount.INBOARD_STATION_FRAC))) \
        .classes("text-[11px] opacity-60")
    if layout == "tips":
        _car_plate_controls(ch, set_choice)


def _plate_answer_row(ch: dict, set_choice, key: str, label: str,
                      row: str) -> bool:
    """WHO ANSWERS one of the plate's two shape numbers — and draw the toggle.

    Returns True where the reader answers it, so the caller draws the field
    beside it; False where the OPTIMISER does, in which case there is no
    field at all and a line saying which design-box row took its place.

    A TOGGLE AND NOT A SWITCH, because the two sides are different PROBLEMS:
    searching the number adds a row to the design vector, so it is a registry
    variant (:data:`api.PLATE_FREEDOMS`) and the shell selects a family
    rather than sending a flag. That is also why the field VANISHES rather
    than greying out — a stated value beside a searched row is one question
    answered twice, and `api.check_flags` refuses it on that family.

    The typed value is NOT cleared when the optimiser takes over: it lives in
    ``choices``, which a family change does not filter, so switching back
    restores the number the reader typed. Answering one more question must
    never lose an answer already given.
    """
    from nicegui import ui

    free = ch.get(key) == "free"
    with ui.row().classes("w-full items-center gap-3 no-wrap"):
        ui.label(label).classes("text-xs w-20 shrink-0 opacity-70")
        ui.toggle({"fixed": "I state it", "free": "optimise it"},
                  value="free" if free else "fixed",
                  on_change=lambda e: set_choice(key, e.value)) \
            .props("dense unelevated no-caps").classes("grow min-w-0")
    if free:
        ui.label(f"Searched, not stated: this is the {row} row of the design "
                 f"box, and its band is edited there like every other. What "
                 f"makes it worth searching is that it is priced at BOTH "
                 f"ends — a plate that leans or turns projects outboard, "
                 f"which comes out of the overall width the span row bounds, "
                 f"and it loses the vertical reach the attachment deck "
                 f"demands — so there is an answer to find rather than a "
                 f"bound to ride. Nothing in this repository records a "
                 f"number for it, which is the case for asking.") \
            .classes("text-[11px] opacity-60")
    return not free


def _plate_airfoil_options() -> dict:
    """The symmetric library sections a PLATE may be given, as a select's
    options. ``None`` heads the list and is the published behaviour: the
    section FAMILY above and its build-up.

    Restricted to the symmetric members for the reason
    ``endplate.CarWingEndplateProblem`` refuses the rest: a cambered section
    has a zero-lift angle, and a vertical panel built at theta = twist -
    alpha_L0 carries a side force at zero toe. Measured from the coordinate
    sidecar (``api.symmetric_section_names``), never from a list here — and
    where the sidecar is missing the answer is "cannot restrict", so the row
    says so instead of offering an unrestricted menu.
    """
    from aerobo import api

    try:
        names = api.symmetric_section_names()
    except Exception:                       # noqa: BLE001 — reported below
        names = ()
    return {None: "the section family above (a build-up, no aerofoil)",
            **{n: n for n in names}}


def _car_plate_airfoil_row(ch: dict, set_choice) -> None:
    """WHERE THE PLATE'S AEROFOIL IS ASKED — which is not here.

    A pointer and deliberately not a select. Choosing a section is a STAGE
    per surface in this shell (``session.SURFACE_STAGES``), and the endplate
    has one: stage 2.8, with its own screen, its own weights and its own
    shape search, at the PLATE's own Reynolds number. A second menu on this
    card would be the same question in two vocabularies — and worse than
    that, it would have LOST: ``config.flags`` writes ``car_flags`` first and
    the per-surface section loop second, so the stage's answer overwrote the
    card's silently.

    What this row does say is that the SECTION FAMILY above stops deciding
    the two things it decides, once an aerofoil is chosen. That is not
    obvious and it is not cosmetic: with a section the plate's profile drag
    is the section's own cd (no form factor, no edge lump) and its bending
    inertia is the integral of its own coordinates (no family constant).
    """
    from nicegui import ui

    ui.label("The plate's AEROFOIL is not asked here — it is stage 2.8, "
             "which screens the symmetric library and searches a symmetric "
             "CST section at the PLATE's own Reynolds number, not the "
             "wing's. Choose one there and the family above stops deciding "
             "both of the things it decides: the profile drag becomes that "
             "section's own cd at the angles each panel flies (no form "
             "factor, no edge lump) and the bending inertia becomes the "
             "integral of its own coordinates. Leave it and the plate is "
             "the published build-up, which is every run in this "
             "repository.") \
        .classes("text-[11px] opacity-70")


def _car_plate_controls(ch: dict, set_choice) -> None:
    """What a LOAD-BEARING plate is made of: its section, root blend and cant.

    Drawn only under the ``tips`` mount, and drawn there automatically. It
    used to sit behind an "Endplates — design them" switch at the foot of the
    card, which asked a second time what the mount had already decided: a
    plate that is carrying the car is designed, and one that is not cannot
    be. The switch is gone and these two follow the answer that creates them.

    All three are the designed-endplate family's own flags (``section``,
    ``blend_frac``, ``endplate_cant_deg``), so none is drawn — or sent —
    anywhere else.

    THE CANT IS ASKED HERE AND NOT IN A SHAPE MENU. A leaning plate is a real
    rear wing and the family has flown one since ``endplate_cant_deg`` was
    declared; the card simply never asked, because the number rode on the
    tip-device menu and :func:`car_tip_shapes` correctly offers no menu where
    the plate is the mount. Re-opening that menu here would have asked one
    question in two vocabularies twice over — its ``blended`` entry writes the
    same ``car_endplate_blend_frac`` this block already types, and its
    ``none`` entry pins the plate to zero height, which deletes the load path
    the mount just chose. A cant is not a shape. It is a dimension of a
    designed part, like the section and the blend above it, so it is asked
    beside them.
    """
    from nicegui import ui

    with ui.row().classes("w-full items-center gap-3 no-wrap"):
        ui.label("Section").classes("text-xs w-20 shrink-0 opacity-70")
        ui.select(CAR_ENDPLATE_SECTION_LABELS,
                  value=ch.get("car_endplate_section", "shaped"),
                  on_change=lambda e: set_choice("car_endplate_section",
                                                 e.value)) \
            .props("outlined dense").classes("grow min-w-0")
    ui.label("A flat plate is stiffer per unit thickness but pays for its "
             "square edges; a shaped section has ~47% of that bending "
             "inertia and no edge penalty — and beats it on drag at every "
             "thickness offered, so the flat plate is only ever chosen for "
             "stiffness. Rounding its edges is the middle those two ends "
             "could not express: all of the stiffness, a quarter of the edge "
             "charge, and the part most rear wings actually carry. Symmetric "
             "in every case — the plate's incidence is the toe variable.") \
        .classes("text-[11px] opacity-60")
    _car_plate_airfoil_row(ch, set_choice)
    if _plate_answer_row(ch, set_choice, "car_plate_blend", "Root blend",
                         "endplate_blend_frac"):
        _num_row("Root blend (0–1)", "car_endplate_blend_frac", ch,
                 set_choice, suffix="of plate height", placeholder="0",
                 lo=0.0, hi=1.0)
    ui.label("The wing/plate CORNER, softened over this fraction of the "
             "plate's height. What it buys is the corner's interference "
             "charge; what it costs is a plate that ends SHORTER and reaches "
             "OUTBOARD, which the wing pays for out of its span band — the "
             "span row is the OVERALL width, so a blend never buys reach the "
             "regulation did not grant. Blank is a crease, and is every "
             "published run.").classes("text-[11px] opacity-60")
    lo_cant, hi_cant = _car_plate_cant_limits()
    if _plate_answer_row(ch, set_choice, "car_plate_cant", "Leaning at",
                         "endplate_cant_deg"):
        _num_row("Leaning at", "car_endplate_cant_deg", ch, set_choice,
                 suffix="° from the wing plane", step=5.0,
                 placeholder=f"{_car_plate_cant_default():g}",
                 lo=lo_cant, hi=hi_cant)
    ui.label(f"90° is upright, and is every published plate. Less leans it "
             f"outboard: the plate keeps its LENGTH, so it reaches less far "
             f"DOWN (h·sin of this angle) and further OUT — and both ends of "
             f"that are charged. The outboard reach comes out of the span "
             f"row, which is the overall width; the lost vertical reach comes "
             f"out of the reach margin, because a plate that leans has to be "
             f"longer to still get to the deck. The band is the nonplanar "
             f"lattice's own validity ({lo_cant:g}–{hi_cant:g}°), not a "
             f"recommendation: below about {_car_plate_cant_reach_floor():.0f}"
             f"° the TOP of the ride band stops being reachable at any plate "
             f"height this box offers, and the run says so as a refusal "
             f"rather than being stopped from asking.") \
        .classes("text-[11px] opacity-60")
    ui.label("The plates carry the wing, so a plate has to REACH the car: it "
             "must span from the wing down to the attachment deck, which is "
             "a signed constraint — raise the wing and the plate carrying it "
             "has to grow. That is what this mount means, and it is why the "
             "plate is designed rather than assumed. A LEANING plate spans "
             "that gap along its own line rather than straight down, so it "
             "is both a longer member and a more loaded one, and its "
             "deflection margin says so.") \
        .classes("text-[11px] opacity-70")


#: THE PLATE AS A TIP DEVICE — the car's answer to the aircraft's
#: :data:`WINGLET_SHAPES`, in the SAME four keys so the two menus read alike.
#:
#: What differs is which of the four a given car family can actually fly, and
#: that is not a UI decision — it is which family COUNTS the plate's outboard
#: projection. ``carwing``/``carwing_multi`` report ``b_m`` as the WING's
#: span, so a plate that leans or curves outboard flies wider than the band
#: says: measured, cant 60 scores 39.79 against vertical's 38.28 and buys it
#: by flying 1.725 m of wing inside a 1.6 m band, which on a car is a
#: regulation or a piece of bodywork. ``endplate.CarWingEndplateProblem``
#: reports ``b_m`` as the OVERALL width and pays for the projection out of the
#: wing's own span (its ``developed_semispan`` call), so there the same cant
#: costs rather than pays — 27.83 at 60 against 28.97 vertical, which is why
#: real rear-wing plates are near-vertical.
CAR_TIP_SHAPE_LABELS = {
    "none": "none — no plate at all",
    "vertical": "endplates, vertical (towards the track)",
    "canted": "endplates, canted outboard",
    "blended": "endplates, blended into the wing",
}

CAR_TIP_SHAPE_NOTES = {
    "none": "No plate. The wing's tips are open, so it loses the fence AND "
            "the nonplanar benefit — offered only where the PYLONS carry the "
            "load, because with the plates gone a tip-borne mount has "
            "nothing to grip. The height row leaves the design vector "
            "(pinned at 0), so this is a smaller search, not a wider one.",
    "vertical": "The published plate: straight towards the track, at 90° to "
                "the wing. Its HEIGHT is a design variable (design box: "
                "endplate_h_m) and it projects nothing outboard, so the "
                "wing spans the whole width band. Every published car result "
                "in this repo was flown this way.",
    "canted": "The plate leans OUTBOARD. ``endplate_h_m`` is an arc length "
              "along the plate, so canting trades its vertical reach for "
              "outboard projection at constant plate — and on this family "
              "that projection comes out of the WING's span, because b_m is "
              "the overall width. It is priced, not free: the wing gets "
              "narrower as the plate leans, and the refusal for a projection "
              "that eats the whole width is the solver's own.",
    "blended": "The wing/plate CORNER is rounded over a fraction of the "
               "plate's height instead of being a crease. What it buys is "
               "the corner's interference charge; what it costs is a plate "
               "that ends SHORTER and reaches OUTBOARD, paid for out of the "
               "wing's span on the same developed line.",
}

#: why an entry is not offered, in the vocabulary :func:`missing_options_note`
#: prints.
#: ...and it is ALL OR NOTHING today, which is why every entry carries the same
#: sentence. There is exactly one reason a shape can be unavailable — the plates
#: are the load path, so they are the mount rather than a device — and in that
#: state the row is not drawn at all (:func:`car_tip_shapes` returns ``{}``), so
#: nothing here reaches the screen as things stand. It is kept, and the
#: ``_missing`` call that reads it is kept, because a per-entry gate is exactly
#: what a new family would add: the net is what makes such a gate explain itself
#: instead of silently dropping an entry.
CAR_TIP_SHAPE_WHY = {
    key: ("the plates are the load path here, so they are the MOUNT and not a "
          "device to shape — their section and their root blend are asked "
          "under the Mount above. Set the Mount to the pylons and the plate "
          "becomes a fence with a shape of its own.")
    for key in CAR_TIP_SHAPE_LABELS
}


def car_tip_shapes(ch: dict, problem_name: str | None = None) -> dict:
    """``{key: label}`` — the plate shapes THIS configuration can fly.

    ONE gate, and it is the Mount. A tip device is a thing bolted to a tip;
    a plate that is carrying the car is not a device, it is the mount — its
    section and its root blend are asked under the Mount answer that created
    them (:func:`_car_plate_controls`), and asking for its "shape" as well
    would be the same question in two places with two vocabularies. So where
    the plates carry the load this returns nothing at all and the card draws
    no tip-device row.

    Under the PYLON mount the plate goes back to being a fence of free
    height, which is exactly a tip device, and all four shapes are offered —
    the aircraft menu's own keys. ``canted`` and ``blended`` are honest here
    only because the family charges the plate's outboard projection against
    its span row (carwing.py); while it did not, a lean or a blend flew
    wider than the band and bought downforce the regulation never granted.

    Deliberately NOT a menu that switches families for you: picking an entry
    never changes the Mount, and this repo's rule is that answering one more
    question never widens a search.

    THE CANT NO LONGER RIDES ON THIS MENU ALONE. It used to be reachable only
    through the ``canted`` entry, so the gate below withheld a real question —
    a load-bearing plate can lean, and ``endplate.py`` has always flown one —
    rather than only withholding a vocabulary. The number is now typed beside
    the section and the blend under the tips mount
    (:func:`_car_plate_controls`); this menu keeps its single gate unchanged,
    because a SHAPE is still not a thing a mount has.
    """
    if _car_is_two_element(ch):
        # one sheet spans both elements and there is no designed-plate
        # solver; the fence is what it is, so the shape menu has nothing to
        # offer that is not already true
        return {}
    if car_mount_layout_of(ch) == "tips":
        return {}
    return dict(CAR_TIP_SHAPE_LABELS)


def car_tip_shape_key(ch: dict, problem_name: str | None = None) -> str:
    """The shape this configuration holds, clipped to what it can fly."""
    offered = car_tip_shapes(ch, problem_name)
    held = str(ch.get("car_tip_shape") or "vertical")
    return held if held in offered else "vertical"


def car_tip_cant_deg(ch: dict) -> float:
    """The cant a canted plate flies at, or the vertical plate's own 90.

    Read off ``carwing.CarWingProblem`` — the family the shape menu is
    offered on (the pylon mount), rather than the designed-endplate one it
    used to be read from. Both ship 90.0; taking it from the class that will
    fly it is what stops the two drifting apart silently.
    """
    from aerobo.carwing import CarWingProblem

    v = ch.get("car_endplate_cant_deg")
    return float(v) if v is not None \
        else float(CarWingProblem.endplate_cant_deg)


def _car_plate_cant_default() -> float:
    """The cant a DESIGNED plate flies when the field is blank.

    Off ``endplate.CarWingEndplateProblem`` — the class that will fly it.
    :func:`car_tip_cant_deg` reads ``carwing.CarWingProblem`` for the same
    number under the PYLON mount, and its docstring gives the rule both
    obey: take it from the class that flies it, so the two cannot drift
    apart silently. Both ship 90.0 today, which is the point — this is a
    live contract, not a value.
    """
    from aerobo.endplate import CarWingEndplateProblem

    return float(CarWingEndplateProblem.endplate_cant_deg)


def _car_plate_cant_limits() -> tuple[float, float]:
    """The band the field accepts — the LATTICE's validity, nothing narrower.

    ``geometry.WINGLET_CANT_LIMITS_DEG`` is the full physical span the
    nonplanar VLM stays valid over. A shell must not refuse what the family
    can fly (a shallow plate is a real answer at a low ride height), so the
    only bar here is the method's own.
    """
    from aerobo.geometry import WINGLET_CANT_LIMITS_DEG

    lo, hi = WINGLET_CANT_LIMITS_DEG
    return float(lo), float(hi)


def _car_plate_cant_reach_floor() -> float:
    """Below this cant the TOP of the ride band is unreachable [deg].

    DERIVED from the family's own three numbers, never written down: the
    tallest plate the box offers is ``max(ENDPLATE_H_M_BOUNDS)``, the
    furthest it has to reach is ``max(RIDE_HEIGHT_BOUNDS_M) - deck``, and a
    plate at cant c closes ``h sin(c)`` of that. So

        sin(c) >= reach_max / h_max

    is the condition for the whole ride band to stay reachable. This is
    INFORMATION on the card, not a bound on the field: a shallower plate is
    perfectly flyable lower down, and the run refuses the corner where it is
    not — which is where a refusal belongs.
    """
    import numpy as np

    from aerobo.endplate import CarWingEndplateProblem as P

    h_max = float(max(P.ENDPLATE_H_M_BOUNDS))
    reach_max = float(max(P.RIDE_HEIGHT_BOUNDS_M)) - float(P.deck_height_m)
    if h_max <= 0.0 or reach_max <= 0.0:
        return 0.0
    return float(np.rad2deg(np.arcsin(min(1.0, reach_max / h_max))))


CAR_ENDPLATE_SECTION_LABELS = {
    "shaped": "streamlined symmetric section",
    "rounded": "flat plate, rounded edges",
    "flat": "flat plate (square edges)",
}


def _car_is_two_element(ch: dict) -> bool:
    """Does this track configuration fly a SLOTTED two-element section?

    One reader for the two ways of asking, so ``car_flags`` and
    ``_derive_family`` can never disagree about which family is being built:
    the dedicated switch, and the tandem switch — which on the track means
    "a second surface", and the only second surface a rear wing can have is a
    flap in its own section.
    """
    return bool(ch.get("car_two_element")) or ch.get("system") == "tandem"


def car_default_v_ms() -> float:
    """The speed the car families fly with no flag — read, never pasted.

    Off the dataclass field itself, so a family that re-times its published
    design point cannot leave a constant here saying otherwise.
    """
    from aerobo.carwing import CarWingProblem

    return float(CarWingProblem.V)


def car_flags(ch: dict, v_ms: float | None = None,
              problem_name: str | None = None) -> dict:
    """Builder choices -> the car-wing configuration flags api accepts.

    Emitted only for the track medium, so these keys can never reach a
    problem that would not know what to do with them. The mount is always
    sent (its default IS carwing's own, so a untouched card stays
    bit-for-bit); the endplate section only when the endplates are designed.

    ``problem_name`` is the family these choices derive to, and it is here
    for the LAP: two of the six registered car families have no ``track_spec``
    field, so ``api.CAR_ENDPLATE_KEYS`` declares neither lap key and sending
    one there is refused by ``check_flags`` at the Run button. Optional, and
    derived when absent, exactly as :func:`tandem_flags` does it — a caller
    that already knows the name (V3 holds it on the wing state) saves the
    derivation, and a caller that does not is not made to care.

    ``v_ms`` is the SPEED the mission card states, and it is a parameter
    rather than a choice because it is a mission field. Until session 62 it
    reached the solver through nothing: the car families declare a ``V`` flag
    (``api.CAR_WING_KEYS``) AND refuse a mission spec outright, so a card set
    to 40 m/s still flew ``carwing.CarWingProblem.V`` = 55 and every
    coefficient, Reynolds number, drag force and deflection in the answer was
    quoted at a speed nobody asked for. Sent only when it differs from that
    default, so an untouched card is still bit-for-bit the published run.
    """
    if ch.get("medium") != "track":
        return {}
    out = {"mount": ch.get("car_mount", car_default_mount())}
    # THE MOUNT AS A CONTINUUM, where the family declares it and the card
    # asked for it. `api._car_wing_kwargs` drops the plain ``mount`` kwarg
    # when a spec is built, so the two never reach the problem together.
    out.update(car_mount_flags(ch, problem_name))
    if v_ms is not None and float(v_ms) != car_default_v_ms():
        # gated by the medium above, which IS the family gate: every
        # registered track family declares "V" (api._CAR_CORE_KEYS), and
        # nothing else does
        out["V"] = float(v_ms)
    # DESIGNED ENDPLATES have no two-element solver, so their flags must not
    # travel to the family a two-element choice derives to — they would be
    # refused by check_flags at the Run button. ``_derive_family`` says the
    # same thing in a note; here they simply do not travel.
    plates = bool(ch.get("car_endplates")) and not _car_is_two_element(ch)
    if plates:
        out["section"] = ch.get("car_endplate_section", "shaped")
        # ...and the plate's CANT, where one was typed. Only where one was:
        # blank sends nothing and the run is bit-for-bit the published
        # upright plate, which is the rule every control on this card obeys.
        # The same key carries the number under the PYLON mount, where the
        # canted TIP DEVICE implies it instead of typing it — one question,
        # one key, and the Mount makes the two sites mutually exclusive on
        # screen.
        if (ch.get("car_endplate_cant_deg") is not None
                and ch.get("car_plate_cant") != "free"):
            # ...and NOT where the cant is a design row. The family that
            # searches it does not declare the flag at all
            # (api._plate_variant_spec drops it), so sending one would fail
            # at `check_flags` — which is the right failure, and this is why
            # it never fires: a searched question has no field to state.
            out["endplate_cant_deg"] = float(ch["car_endplate_cant_deg"])
        # NO PLATE AEROFOIL WRITE HERE. The plate's section is stage 2.8's
        # answer and travels through `config.flags`'s per-surface loop with
        # every other chosen section. A second writer on this card would be
        # overwritten by that loop without a word, because `car_flags` is
        # merged first — the silent-override failure, not a near miss.
        # ...and the plate's chord in METRES, where one was stated. Gated on
        # the designed-endplate family for the same reason the blend is: only
        # that solver has a plate chord to bound (carwing's fence carries the
        # wing's own chord), so the flag would be meaningless on the other.
        for key, flag in (("car_endplate_chord_min_m", "endplate_chord_min_m"),
                          ("car_endplate_chord_max_m",
                           "endplate_chord_max_m")):
            if ch.get(key) is not None:
                out[flag] = float(ch[key])
    # NO SPAN BAND AND NO AREA BAND. Both are BANDS ON DESIGN VARIABLES, so
    # they are the design box's ``b_m`` and ``S_m2`` rows and reach the
    # problem through ``api._car_size_band_kwargs`` — never as a flag from
    # here. ``api.check_flags`` refuses the old spellings outright now, which
    # is what stops a shell quietly re-opening the second channel.
    #
    # WHAT IS MAXIMISED, and the two LIMITS. All value flags on every car
    # family. Sent only where the user stated one, so an untouched card is
    # the family's own default configuration exactly.
    if ch.get("car_objective"):
        out["car_objective"] = str(ch["car_objective"])
    # the DRAG CEILING, in newtons, and only where one was typed. There is no
    # menu and no default: the family ships with no drag constraint at all
    # (api.CAR_DEFAULT_OBJECTIVE says why an allowance handed to a maximiser
    # is a number the answer rides), so a blank field sends nothing and the
    # run carries no drag margin.
    if ch.get("car_drag_budget_n") is not None:
        out["drag_budget_n"] = float(ch["car_drag_budget_n"])
    if ch.get("car_downforce_min_n") is not None:
        out["downforce_min_n"] = float(ch["car_downforce_min_n"])
    # THE CIRCUIT, and only to a family that has one to set. Two things are
    # gated here and they are separate refusals, both of which api makes and
    # neither of which the card may reach:
    #
    #   * a lap key on the DESIGNED-ENDPLATE family. That class carries no
    #     track_spec/car_spec/track_points at all, so the registry declares
    #     neither key on it (api.CAR_ENDPLATE_KEYS) and check_flags refuses
    #     the run outright. The gate is the registry's own declaration, never
    #     a copy of "which families have a lap" kept here;
    #   * ``track_points`` WITHOUT a circuit, which api._car_track_kwargs
    #     refuses by name ("sampling a lap that does not exist is not a
    #     request"). The count is only ever sent inside the branch that sends
    #     the track, so the card cannot express the pair at all.
    #
    # An untouched card holds "off" and sends neither, which is every
    # published car run bit-for-bit.
    track = car_track_of(ch)
    if track is not None and car_lap_available(ch, problem_name):
        out["car_track"] = track
        if ch.get("car_track_points") is not None:
            out["track_points"] = int(ch["car_track_points"])
    # ...and the wing/plate CORNER, where one was asked for. Gated on the
    # DESIGNED-endplate family, not on the medium: only that solver counts
    # the junctions (carwing's plain fence is a free-height plate with
    # n_junctions = 0), so a blend on the other family would soften the
    # corner's wake while paying nothing for the corner.
    if plates and ch.get("car_plate_blend") == "free":
        # the corner is a design row, so only the LAW it follows is still a
        # statement — `blend_shape` names which shape the turn is, never how
        # much of one there is, and the free family keeps declaring it
        out["blend_shape"] = ch.get("blend_shape", "spiral")
    elif plates and ch.get("car_endplate_blend_frac"):
        out["blend_frac"] = float(ch["car_endplate_blend_frac"])
        out["blend_shape"] = ch.get("blend_shape", "spiral")
    # ...and the PLATE'S SHAPE, where the plate is a TIP DEVICE rather than
    # the mount — i.e. under the pylon layout, which is the only place
    # `car_tip_shapes` offers a shape at all. Each statement is gated on the
    # family that declares it, and `api.check_flags` refuses the wrong way
    # round, which is what makes these gates checkable rather than polite.
    #
    # A CANT is honest on every car family now: all three charge the plate's
    # outboard projection against their span row, so a lean shortens the
    # wing instead of flying wider than the band. A CHORD LAW travels only to
    # the families whose plate has no chord row of its own (the designed
    # plate's chord IS a design variable, so continuing the wing's law onto
    # it would answer a question the search is already answering).
    shape = car_tip_shape_key(ch, problem_name) \
        if car_tip_shapes(ch, problem_name) else None
    if shape == "canted":
        out["endplate_cant_deg"] = car_tip_cant_deg(ch)
    if shape == "blended":
        # the aircraft menu's own rule (set_winglet_shape): the blend is a
        # VALUE the shape implies, not a field, because the library has a
        # default turn and the reader has already said "blended"
        out["blend_frac"] = float(ch.get("car_endplate_blend_frac")
                                  or BLEND_FRAC_DEFAULT)
        out["blend_shape"] = ch.get("blend_shape", "spiral")
    if shape is not None and ch.get("car_endplate_chord_follows"):
        out["endplate_chord_follows"] = True
    return out


def tandem_flags(ch: dict, problem_name: str | None = None) -> dict:
    """Builder choices -> the tandem pair's STAGGER and REAR-SPAN flags.

    Emitted only for a configuration that actually SELECTS a tandem family —
    the pair in air; water and the track derive to a hydrofoil or a car wing,
    which have no second surface to stagger — and only where the user stated
    a number: untouched, the stagger follows the published fractions of
    whatever span is in play (api._tandem_planform_kwargs), so an untouched
    card is bit-for-bit the published pair.

    The REAR SPAN travels under one extra rule, the same one
    :func:`planform_flags` obeys: only to a family that is not already
    SEARCHING its spans. Sized, the pair carries a span row per wing, and a
    typed span beside a searched one is two answers to one question — the
    registry strips the flag there (api.TANDEM_SPAN_KEYS), so sending it
    would be silently dropped rather than honoured.
    """
    if ch.get("system") != "tandem" or ch.get("medium", "air") != "air":
        return {}
    out: dict = {}
    for key, flag in (("tandem_dx_m", "dx_m"), ("tandem_dz_m", "dz_m"),
                      ("tandem_fin_boom_m", "fin_boom_m")):
        if ch.get(key) is not None:
            out[flag] = float(ch[key])
    if ch.get("tandem_b_rear_m") is not None:
        name = problem_name or derive_problem(ch)[0]
        if planform_resizable(name):
            out["b_rear_m"] = float(ch["tandem_b_rear_m"])
    return out


PLANFORM_CHOICE_LABELS = {
    "fixed": "fixed span + area (you choose the size)",
    "wing_loading": "area from the mission's W/S, span optimised",
    "wing_loading_free": "span AND wing loading optimised (W/S searched in "
                         "N/m², inside the mission's own ceiling)",
    "free": "free span + area (weight-coupled) — composes with everything",
    "aircraft": "the published aircraft-sizing problem (also frees t/c)",
}

#: why a planform entry is not offered — same contract as WINGLET_OPTION_WHY,
#: read by missing_options_note so a menu never just loses an entry.
PLANFORM_OPTION_WHY = {
    "wing_loading_free": "this family has no wing-loading sizing variant; "
                         "the same families carry the searched-W/S mode and "
                         "the fixed-W/S one, because both derive the area "
                         "from a loading through the same weight loop",
    "wing_loading": "this family has no wing-loading sizing variant; the "
                    "families that carry it are the AIR ones that close a "
                    "weight loop — the single wing, the wing with a tail, "
                    "and the tandem pair (one span row per wing). Elsewhere "
                    "there is no weight loop for an area to follow, so the "
                    "size is stated on the size card or already in the "
                    "design vector",
    "free": "this family has no weight-coupled sizing variant. Freeing the "
            "size honestly means the surface WEIGHS what its size implies, "
            "and the Raymer wing-weight loop these families close is an "
            "aircraft one — there is no calibrated structural weight for a "
            "hydrofoil or a rear wing, so a weight-coupled size here would "
            "be a number with nothing behind it. Their size is still yours: "
            "state it on the size card (the car's span is a design variable "
            "of its own)",
    "aircraft": "it is a family of its own, with no tip-device, tail or "
                "section-optimisation solver, so offering it here could "
                "only take one of those away. 'free span + area "
                "(weight-coupled)' is the same freedom and composes with "
                "them — clear the tip device and the tail to reach the "
                "published problem itself",
}

PLANFORM_CHOICE_NOTES = {
    "fixed": "The design vector reshapes the planform; the size card below "
             "sets what it reshapes.",
    "free": "Span and area join the design vector on whatever family the "
            "choices above select. Freeing them honestly costs three "
            "changes at once: the wing WEIGHS what its size implies "
            "(Raymer, closed as a fixed point), the trim target follows "
            "that weight, and the score becomes PAYLOAD L/D = W_fixed/D — "
            "with the weight varying, 'maximise L/D' would reward a heavier "
            "wing that lifts better. A root-bending stress margin joins as a "
            "constraint; without it the span just slams the box.",
    "aircraft": "The calibrated 6-D problem that shipped: taper, twist, "
                "span, area AND section thickness, weight-coupled, with the "
                "frozen published results. Its own family, so it does not "
                "compose with the tip device or the tail — use 'free span + "
                "area' for that.",
    "wing_loading": "The area is NOT a design variable: it follows the wing "
                    "loading the mission states, through the same weight "
                    "loop (S = W_total/(W/S), closed as a fixed point), so "
                    "the search carries the SPAN alone — inside the band you "
                    "type below. Everything the free planform costs is paid "
                    "here too (the wing weighs what its size implies, the "
                    "score is payload L/D, the root-bending stress margin is "
                    "a constraint). What is different is the question: W/S "
                    "is the MISSION's — a constraint diagram answers it, and "
                    "it fixes the trim lift coefficient outright, CL = "
                    "(W/S)/q — while the span is the aerodynamicist's. Like "
                    "the free planform it is a modifier, so the tip device "
                    "and the tail stay exactly as chosen above.",
}


def _opt_float(value) -> float | None:
    """A cleared number field posts None/'' — that means "solver default"."""
    if value is None or value == "":
        return None
    return float(value)


def _opt_float_in(value, lo: float | None = None,
                  hi: float | None = None) -> float | None:
    """:func:`_opt_float`, held inside the range its solver can answer.

    A cleared field still means "solver default" and clamping must not invent
    a number for it. Everything else is held to [lo, hi]: a blend fraction
    typed as 1.5 reaches geometry.span_path, raises, and returns as a whole
    run of penalties reading "solver failure: blend_frac must be in [0, 1]" —
    a card that can only ask answerable questions is cheaper than a run that
    fails for a reason nobody reads.
    """
    v = _opt_float(value)
    if v is None:
        return None
    if lo is not None:
        v = max(float(lo), v)
    if hi is not None:
        v = min(float(hi), v)
    return v


def planform_size_note(problem_name: str, ch: dict) -> str:
    """Honest one-liner under the size card: what a chosen size does, and
    what it does NOT move."""
    from aerobo import api

    base = ("Span and area are configuration, not design variables: the "
            "vector reshapes the planform (taper, twist, chord law), the "
            "size sets what it reshapes. The trim target follows it — "
            "CL = W/(qS) — so the mission card's default weight scales with "
            "the area you type, and an aspect ratio outside "
            f"{api.PLANFORM_AR_LIMITS} is refused rather than scored.")
    if problem_name.startswith("tail"):
        base += (" The tail's CG and arm box were calibrated on the 10 m / "
                 "10 m² wing, so resizing moves where the static-margin "
                 "boundary sits inside the box.")
    if ch.get("span_m") is None and ch.get("area_m2") is None:
        base += " Untouched, this run is bit-for-bit the published one."
    return base


def planform_size_defaults(problem_name: str) -> tuple[float, float] | None:
    """The (span, area) a problem flies untouched — what the size card
    pre-fills with. None when the problem has no fixed planform."""
    from aerobo import api
    return api.planform_size(problem_name)


def planform_resizable(problem_name: str) -> bool:
    """Does the derived problem honour a chosen span/area?"""
    from aerobo import api
    return api.resizable(problem_name)


def planform_flags(ch: dict, problem_name: str | None = None) -> dict:
    """Builder choices -> the wing-SIZE flags (api.PLANFORM_KEYS).

    Emitted ONLY for a problem that declares them, so a size typed into the
    card can never be silently ignored by a solver that does not resize
    (the hydrofoil and the car wing carry calibrated geometry; the
    free-planform aircraft has b and S in its design vector). Untouched
    entries send nothing at all — bit-for-bit.
    """
    name = problem_name or derive_problem(ch)[0]
    if not planform_resizable(name):
        return {}
    out: dict = {}
    for key, flag in (("span_m", "b_m"), ("area_m2", "S_m2")):
        val = ch.get(key)
        if val is not None:
            out[flag] = float(val)
    return out


def tail_flags(ch: dict) -> dict:
    """Builder choices -> the tail/elevator configuration flags api accepts.

    Only non-default entries are emitted, so an untouched tail card sends
    nothing and the 5-D problem stays bit-for-bit its published self.
    """
    if not ch.get("tail"):
        return {}
    out: dict = {}
    ttype = ch.get("tail_type", "conventional")
    if ttype != "conventional":
        out["tail_type"] = ttype
    if ttype == "v_tail":
        out["dihedral_deg"] = float(ch.get("tail_dihedral_deg", 35.0))
    if ch.get("tail_control", "stabilator") != "stabilator":
        out["control"] = ch["tail_control"]
        out["elevator_chord_frac"] = float(
            ch.get("tail_elevator_chord", 0.30))
    if ch.get("tail_arm") == "fixed":
        out["l_t_m"] = float(ch.get("tail_arm_m", 5.5))
    # the LAYOUT the surface is trimmed in, where the user stated one. The
    # height only travels while it is not a design variable: freeing it and
    # stating it would be one question answered twice, and the problem
    # refuses the pair outright.
    if ch.get("tail_cg_m") is not None:
        out["x_cg_m"] = float(ch["tail_cg_m"])
    if ch.get("tail_height_m") is not None and ch.get("tail_height") != "free":
        out["z_t_m"] = float(ch["tail_height_m"])
    # ...and WHICH WAY UP it flies its section. "auto" is the published rule
    # (follow the load), so it travels as nothing at all.
    from aerobo import api as _api
    if ch.get("tail_mount", _api.TAIL_MOUNT_AUTO) != _api.TAIL_MOUNT_AUTO:
        out[_api.TAIL_MOUNT_KEY] = str(ch["tail_mount"])
    # ...and what it may MEASURE. Each limit is off until it is switched on,
    # and off travels as nothing at all.
    for key in _api.TAIL_LIMIT_KEYS:
        if ch.get(key) is not None:
            out[key] = float(ch[key])
    # which way the SECOND surface's tip device may point. Only a family
    # that designs that device declares the flag, and only a stated
    # direction travels — untouched, each family keeps its own published
    # band (api.tail_winglet_direction reads what that is)
    if ch.get("tail_design") == "planform+tip" and ch.get("tail_winglet_dir"):
        out["tail_winglet_dir"] = str(ch["tail_winglet_dir"])
    # ...and WHAT SHAPE that device is: the cant band its type narrows to,
    # and its own blend. Same rule as the direction — only a stated answer
    # travels, so an untouched card leaves every published run alone.
    if ch.get("tail_design") == "planform+tip":
        if ch.get("tail_winglet_type"):
            out["tail_winglet_type"] = str(ch["tail_winglet_type"])
        if ch.get("tail_winglet_blend_frac"):
            out["tail_winglet_blend_frac"] = float(
                ch["tail_winglet_blend_frac"])
    return out


#: why a family does not offer a given Winglet entry. Each reason has to be
#: true of WHATEVER configuration is on screen — a note that answers by
#: naming some other family ("the water tip device is…" under a tandem pair)
#: reads as a non-answer. Where a family's own reason is genuinely different,
#: it goes in OPTION_WHY_BY_MEDIUM below rather than into the general text.
WINGLET_OPTION_WHY = {
    "canted_free": "no tip-device solver for this configuration",
    "canted_capped": "no span-capped tip-device variant for this "
                     "configuration",
    "vertical": "no tip-device solver for this configuration",
    "raked": "a raked tip must be span-capped, and this configuration has no "
             "span-capped variant",
    "blended": "this configuration's solver draws the surface/device corner "
               "as a corner, so there is no transition for a blend to shape",
    "blended_wing": "the wing-side blend is its own problem — it exists on "
                    "the single wing only",
}

#: the Airfoil menu, in the order it is offered.
AIRFOIL_OPTION_LABELS = {
    "fixed": "fixed section (NACA 2412 polar)",
    "tc_sweep": "optimise thickness t/c + sweep",
    "coupled": "co-optimise section (t/c, cl library)",
    "section_wing": "shape section + wing (CST + XFOIL)",
    "section_only": "section only, 2-D (CST + XFOIL)",
}

#: why a family does not offer a given Airfoil entry. Shown under the menu so
#: a missing option is an explanation rather than a mystery.
AIRFOIL_OPTION_WHY = {
    # both of these are EXTENSION POINTS, and each used to name the wrong
    # obstacle. The NACA 24XX family and the pre-optimised library are plain
    # section lookups — any solver that reads a polar can read them, and the
    # tandem pair demonstrably flies a 24XX member today. What is missing is
    # the DESIGN VARIABLE that selects one, in this family's vector.
    "tc_sweep": "this family flies one section table — the one chosen, or "
                "its own default. A thickness that SELECTS a member of the "
                "NACA 24XX family is not yet one of its design variables",
    "coupled": "the pre-optimised (t/c × cl) section library has no combined "
               "solver with this configuration yet — the library itself is "
               "two numbers, indexed by nothing about the wing",
    "section_wing": "no solver shapes a CST section with this configuration",
    "section_only": "the pure 2-D section problem has no wing to fly on; "
                    "with a wing selected, use 'shape section + wing'",
}

#: the handful of refusals a family answers DIFFERENTLY from the general
#: reason — because the option is not missing so much as already granted.
#: ``{medium: {menu_key: reason}}``, consulted by :func:`option_why`.
OPTION_WHY_BY_MEDIUM = {
    "water": {
        # the water family is scored span-free BY CONSTRUCTION — there is no
        # capped water variant to select, and the reason there is none is
        # that under water the trade a tip device is judged on is the cant's
        # SIGN (which end sees more static head), not its projection. Said
        # as one sentence, because a reason that names only the intent reads
        # as though the capped solver were there and withheld.
        "canted_capped": "no span-capped water variant exists: the water tip "
                         "device is scored span-free, because its trade is "
                         "the cant's SIGN, not its projection",
        "raked": "a raked tip must be span-capped, and no span-capped water "
                 "variant exists — the water tip device is scored span-free",
        "tc_sweep": "the water families already carry t/c as a design "
                    "variable, so there is nothing for a t/c sweep to add",
    },
    "track": {
        "canted_free": "the endplates ARE this wing's tip device, and they "
                       "have their own card below",
        "canted_capped": "the endplates ARE this wing's tip device, and they "
                         "have their own card below",
        "vertical": "the endplates ARE this wing's tip device, and they have "
                    "their own card below",
        # ...and the SHAPE menu's own key for the same answer: the seven-entry
        # study menu spells it canted_free / canted_capped, the shape menu
        # simply "canted" (WINGLET_SHAPE_WHY), and a reason table that
        # answers one and not the other leaves the honest sentence beside a
        # generic "no solver" for the same missing entry
        "canted": "the endplates ARE this wing's tip device, and they have "
                  "their own card below",
        # ...and the BLENDED entry, in the SAME words. Not because a plate
        # has no blend — it has one, endplate.py has priced that corner
        # since it was written and the card now asks for it ("Root blend") —
        # but because the answer is the same sentence: the question belongs
        # to the plate, so it is asked where the plate is. Word-for-word
        # identical on purpose — missing_options_note groups entries by their
        # reason, so a differently-worded version of the same answer would
        # print the endplate sentence twice.
        "blended": "the endplates ARE this wing's tip device, and they have "
                   "their own card below",
    },
}


def option_why(ch: dict, table: dict) -> dict:
    """``table`` specialised to this configuration (OPTION_WHY_BY_MEDIUM)."""
    override = OPTION_WHY_BY_MEDIUM.get(ch.get("medium", "air"), {})
    return {k: override.get(k, v) for k, v in table.items()}


def option_available(ch: dict, key: str, value) -> bool:
    """Would this family HONOUR ``ch[key] = value``?

    Asked of the registry rather than answered from a table: the choice is
    applied, the problem derived, and the problem's own inverse choices read
    back. If the value survives that round trip a solver exists for it; if it
    does not, the builder would have quietly ignored it. This is what stopped
    the menus from greying out combinations that had since become real
    problems (tandem + a designed section was the one that got noticed).
    """
    trial = dict(ch)
    if key == "winglets":                     # composite menu key
        wl, wtype = WINGLET_OPTIONS.get(value, ("none", "canted"))
        trial["winglets"], trial["winglet_type"] = wl, wtype
    else:
        trial[key] = value
    name, _ = derive_problem(trial)
    back = choices_from_problem(name)
    if key == "winglets":
        # the SPAN ACCOUNTING (none / free / capped) is what a solver has to
        # support; the cant BAND (canted / vertical / raked) is a value flag
        # the same problem carries, so it round-trips as "canted" and must not
        # be read as unavailable. A blended transition IS its own problem, so
        # that one is compared in full.
        wl, wtype = WINGLET_OPTIONS.get(value, ("none", "canted"))
        if back.get("winglets") != wl:
            return False
        if wtype in BLENDED_TYPES:
            return back.get("winglet_type") == wtype
        return True
    return back.get(key) == value


def airfoil_options(ch: dict) -> dict:
    """Airfoil menu entries this configuration actually has a solver for."""
    return {k: v for k, v in AIRFOIL_OPTION_LABELS.items()
            if k == "fixed" or option_available(ch, "airfoil", k)}


def winglet_options(ch: dict) -> dict:
    """Winglet menu entries this configuration actually has a solver for."""
    return {k: v for k, v in WINGLET_OPTION_LABELS.items()
            if k == "none" or option_available(ch, "winglets", k)}


def planform_options(ch: dict) -> dict:
    """Planform entries this configuration actually has a solver for.

    Two rules, because the control asks two different kinds of question.
    The SIZING values are modifiers (:data:`PLANFORM_MODIFIER_VALUES`), so
    they are offered wherever the family declares them and take nothing
    away — a tip device and a tail are still exactly as chosen after one is
    picked. ``"aircraft"`` is a FAMILY, and the only entry in this menu that
    can delete another choice: it has no combined solver with a tip device,
    a tail or a section optimisation, so choosing it used to silently reset
    the surface the user had just asked for. It is therefore offered only
    while there is nothing to lose (:func:`active_specials` is empty), with
    :data:`PLANFORM_OPTION_WHY` saying so and pointing at the composable
    freedom — which is what its own note has always recommended.
    """
    out = {}
    for key, label in PLANFORM_CHOICE_LABELS.items():
        if key == "fixed":
            out[key] = label
        elif not option_available(ch, "planform", key):
            continue
        elif key in PLANFORM_MODIFIER_VALUES or not [
                s for s in active_specials(ch) if s != "planform"]:
            out[key] = label
    return out


def missing_options_note(offered: dict, all_labels: dict,
                         why: dict) -> str:
    """One line naming the entries this family does NOT offer, and why.

    Entries that share a reason are named TOGETHER — the car wing refuses
    three winglet entries for the one reason (its endplates are the tip
    device), and repeating that sentence three times reads as noise rather
    than as an answer.
    """
    missing = [k for k in all_labels if k not in offered and k in why]
    if not missing:
        return ""
    groups: dict[str, list[str]] = {}
    for k in missing:                       # insertion order = menu order
        groups.setdefault(why[k], []).append(all_labels[k])
    bits = [f"{', '.join(labels)} — {reason}"
            for reason, labels in groups.items()]
    return "Not offered here: " + "; ".join(bits) + "."


#: builder keys whose availability is decided from the registry
#: (option_available). Their controls fall back to the default — or go flat —
#: when the held value is not offered, so the STATE has to fall back with
#: them: otherwise the select reads "fixed" while the choices still say
#: "section_wing", and the solver banner reports an "ignored" feature the
#: user cannot see selected anywhere.
#:
#: The two MODIFIERS are in here as well. They are still never reset by
#: another CHOICE (that is what SPECIAL_KEYS governs, and they compose with
#: every family that declares them); this is the other axis — a family that
#: has no variant at all cannot hold one, and a greyed control reading "free"
#: over a banner saying the free chord law was ignored is the same lie the
#: menus were fixed to stop telling. ``derive_problem`` keeps its refusal
#: notes for every caller that does NOT come through the builder (a preset, a
#: saved record, the API).
#: ``system`` goes FIRST, and it is in here at all because a lifting SYSTEM
#: is a capability like any other: only the air families fly a pair, so
#: leaving it out let "tandem" survive a switch to water or the track as dead
#: state. Dead is the charitable reading — it was read back: the V1/V2 pages
#: compute ``tail_live = not (track or tandem)`` from it, so a hydrofoil
#: reached by way of an air tandem had its ELEVATOR switch disabled with the
#: tooltip "the tandem pair's rear wing IS the second surface", about a
#: hydrofoil that has no rear wing, while ``option_available(ch, "tail",
#: True)`` said yes and hydrotail.py stood there ready to solve it.
#:
#: FIRST rather than appended, because the restore loop keeps the earliest
#: key it can: with ``system`` last, a state holding both the pair and (say)
#: a t/c sweep resolves the other way round and 404 air combinations lose the
#: pair. First reproduces ``_derive_family``'s own priority exactly — no air
#: combination changes, and water and track both drop it.
NORMALISED_KEYS = ("system", "tail", "airfoil", "winglets", "planform",
                   "chord", "flight", "wing_cant")


def _held_option(ch: dict, key: str):
    """The menu key ``ch`` currently holds for a normalised control."""
    return (winglet_option_key(ch) if key == "winglets"
            else ch.get(key, BUILDER_DEFAULTS[key]))


def _set_option(ch: dict, key: str, value) -> None:
    """Write a menu key back into the choices (winglets is a composite)."""
    if key == "winglets":
        ch["winglets"], ch["winglet_type"] = WINGLET_OPTIONS[value]
    else:
        ch[key] = value


def choices_consistent(ch: dict) -> bool:
    """Is every normalised control holding a value its own menu offers?"""
    return all(_held_option(ch, k) == BUILDER_DEFAULTS[k]
               or option_available(ch, k, _held_option(ch, k))
               for k in NORMALISED_KEYS)


def normalise_choices(ch: dict, keep: str | None = None) -> list[str]:
    """Drop, in place, every speciality this configuration has no solver for.

    Called after any builder change (the medium, the system and the tail
    switch are not speciality keys, so the mutual-exclusion reset in
    ``set_choice`` never saw them): switching to the car after asking for a
    designed section used to leave ``airfoil = "section_wing"`` in the state
    while the menu — which offers only what the registry can solve — showed
    "fixed".

    The keys are cleared TOGETHER and then restored one at a time, keeping a
    restore only while the WHOLE state stays consistent. Restoring them
    independently is not enough: they constrain each other, so clearing a tip
    device can make the aircraft-sizing family legal again and that family in
    turn has no t/c solver — an independent restore would put back both and
    leave the state inconsistent in a new way. ``keep`` (the key the user has
    just set) is restored first, so their newest choice outranks the older
    ones. Returns the keys that stayed cleared, for the caller to notify about.
    """
    # THE VALUE-LEVEL FALLBACK RUNS ON BOTH PATHS. ``choices_consistent`` asks
    # only about :data:`NORMALISED_KEYS` — the family-SELECTING controls — and
    # the circuit selects no family: switching the medium off the track, or
    # switching the designed endplates on, leaves every normalised key holding
    # a value its own menu offers, so the loop below is skipped entirely and a
    # circuit left in the state would survive onto a family with no
    # ``track_spec``. Returning early without it is exactly the stale-menu bug
    # this function exists to prevent, one level down.
    #
    # ``_normalise_winglet_blend`` is deliberately NOT moved up here with it.
    # It has the same shape and looks like the same latent gap, and changing
    # where it runs would move a shipped behaviour on a control this session
    # did not touch — recorded in the handover instead of fixed in passing.
    if choices_consistent(ch):
        return _normalise_car_lap(ch)
    order = ([keep] if keep in NORMALISED_KEYS else []) \
        + [k for k in NORMALISED_KEYS if k != keep]
    held = {k: _held_option(ch, k) for k in order}
    for k in order:
        _set_option(ch, k, BUILDER_DEFAULTS[k])
    dropped = []
    for k in order:
        if held[k] == BUILDER_DEFAULTS[k]:
            continue
        _set_option(ch, k, held[k])
        if not choices_consistent(ch):
            _set_option(ch, k, BUILDER_DEFAULTS[k])
            dropped.append(k)
    dropped += _normalise_winglet_blend(ch)
    dropped += _normalise_car_lap(ch)
    return dropped


def _normalise_car_lap(ch: dict) -> list[str]:
    """Drop a CIRCUIT the newly derived family has no field for, and the
    objective and sampling count that only exist beside one.

    Three values, none of them in :data:`NORMALISED_KEYS` — they select no
    family, so the loop above never sees them — and all three have to fall
    back with the state for the same reason the winglet blend does. The three
    ways the state goes stale, each of them a menu that would then be showing
    a value it does not offer:

    * the medium leaves the track, or the DESIGNED-ENDPLATE family is
      selected. Neither has a ``track_spec``, so the Circuit row is not drawn
      at all and a circuit left in the state would be sent to a family
      ``check_flags`` refuses it on;
    * the circuit is switched OFF while ``laptime`` is selected. Both car
      problem classes refuse that pair by name ("objective 'laptime' needs a
      circuit"), so the run would die at the Run button on a combination the
      menu had shown as available;
    * ``car_track_points`` survives the circuit. ``api._car_track_kwargs``
      refuses a sampling count with no lap to sample, by name.

    The objective falls back to ``None``, which is "the family's own default"
    (``api.CAR_DEFAULT_OBJECTIVE``) and not to a spelling of it — a card that
    wrote ``efficiency`` into the state would be answering a question the
    user had not been asked, and the two would then drift the first time that
    default moved.
    """
    dropped: list[str] = []
    if not car_lap_available(ch):
        if ch.get("car_track", BUILDER_DEFAULTS["car_track"]) \
                != BUILDER_DEFAULTS["car_track"]:
            ch["car_track"] = BUILDER_DEFAULTS["car_track"]
            dropped.append("car_track")
    if car_track_of(ch) is None:
        if ch.get("car_objective") == "laptime":
            ch["car_objective"] = BUILDER_DEFAULTS["car_objective"]
            dropped.append("car_objective")
        if ch.get("car_track_points") is not None:
            ch["car_track_points"] = None
            dropped.append("car_track_points")
    return dropped


def _normalise_winglet_blend(ch: dict) -> list[str]:
    """Drop a FIXED BLEND the newly derived family cannot draw.

    The blend is a value, so it is not one of :data:`NORMALISED_KEYS` — it
    selects nothing and the loop above would never see it. It still has to
    fall back with the rest of the state: a family whose solver draws the
    corner as a corner leaves the menu with no "blended" entry, so a blend
    left in the state would put the tip-device select on a shape its own
    menu does not offer. Exactly the stale-menu bug class, one level down.
    """
    if not ch.get("winglet_blend_frac"):
        return []
    pair = (ch.get("winglets", "none"), ch.get("winglet_type", "canted"))
    if pair[0] != "none" and _blend_flag_available(ch, pair):
        return []
    ch["winglet_blend_frac"] = None
    return ["winglet_blend_frac"]


def start_choices(**overrides) -> dict:
    """The configuration a fresh page opens on, already normalised.

    :data:`BUILDER_START` is a WISH, not a state: the polynomial chord law it
    asks for exists on nearly every family but not on the pure 2-D section
    problem, so the opening configuration is put through
    :func:`normalise_choices` exactly like a configuration the user built. A
    page therefore never opens holding a value its own menus do not offer —
    the one rule that keeps builder state and the controls describing it from
    drifting apart.

    ``overrides`` (the medium a shell opens on, a preset a test exercises)
    outrank the starting point, so asking for straight taper still gets it.
    """
    ch = dict(BUILDER_START, **overrides)
    normalise_choices(ch)
    return ch


def specials_compatible(k1: str, k2: str, ch: dict) -> bool:
    """Which speciality PAIRS have a combined solver.

    Winglets + airfoil (thickness family, or the 13-D live-XFOIL CST
    section) is the pair that shipped. The TAIL now joins it: the nonplanar
    wing+tail family (api.WING_TAIL_VARIANTS) carries a tip device and the
    section THICKNESS beside the tail, so those combinations no longer reset
    each other. Section SHAPING (CST/XFOIL, section library) still has no
    wing+tail solver, so that pair stays exclusive.

    The free chord law is not in this matrix at all — it is a modifier, not
    a family (SPECIAL_KEYS).
    """
    pair = {k1, k2}
    if "planform" in pair:
        # Sizing is a MODIFIER, so it composes with the family the other
        # choices select — 'winglet + free span (W/S)', 'tail + free
        # planform' and friends are in the registry, and picking a size used
        # to wipe the tip device that had just been asked for. The one
        # exception is the legacy "aircraft" value, which IS a family (its
        # own calibrated 6-D problem, with no combined solver here). Where
        # the modifier itself is missing for that family — 'winglet +
        # airfoil (XFOIL)' has no W/S twin — the menu does not offer it and
        # normalise_choices is the net, both reading the registry rather
        # than this hand-kept matrix.
        return ch.get("planform") in PLANFORM_MODIFIER_VALUES
    if pair == {"winglets", "airfoil"}:
        return ch.get("airfoil") in ("tc_sweep", "section_only",
                                     "section_wing")
    if pair == {"tail", "winglets"}:
        return True
    if pair == {"tail", "airfoil"}:
        # thickness selection AND live-XFOIL section shaping both have a
        # wing+tail solver now (wingtail.py / wingtail_section.py); only the
        # pre-optimised section LIBRARY does not
        return ch.get("airfoil") in _TAIL_AIRFOILS + _TAIL_SECTION_AIRFOILS
    return False


#: builder choice key -> the api MODIFIER it switches on. Both are plain
#: "fixed"/"free" selects that never participate in the family reset.
MODIFIER_KEYS = {"chord": "chord", "flight": "flight"}


def chord_twin_of(name: str) -> str | None:
    """Free-chord-law twin of a problem, or None if it has no chord variant."""
    from aerobo import api
    return api.CHORD_TWINS.get(name)


def modifier_twin_of(name: str, mod: str) -> str | None:
    """The same problem with one more MODIFIER on, or None if unavailable."""
    from aerobo import api
    return api.add_modifier(name, mod)


def active_modifiers(ch: dict) -> list[str]:
    """Modifier keys the builder choices have switched on, in vector order.

    Read through :data:`MODIFIER_ON`, the same table :func:`derive_problem`
    uses, because a modifier's NAME is not its choice key: the size modifier
    is one value of the three-valued ``planform`` control, so there is no
    ``ch["size"]`` to compare a default against.
    """
    from aerobo import api
    return [m for m in api.MODIFIERS
            if ch.get(MODIFIER_ON[m][0], BUILDER_DEFAULTS[MODIFIER_ON[m][0]])
            == MODIFIER_ON[m][1]]


#: NOTE the three helpers that used to live here are gone with the controls
#: they served: ``_car_span_band`` and ``_car_area_band`` read the card's own
#: span/area band back off the physics, and those bands are the design box's
#: ``b_m`` / ``S_m2`` rows now; ``_car_drag_budget_default`` restated the
#: published coefficient allowance as a force, and no car family carries a
#: default allowance any more. ``carwing.SPAN_BOUNDS_M`` /
#: ``AREA_BOUNDS_M2`` and ``carwing.published_drag_budget_n`` are still where
#: those numbers live if a caller needs one.

#: what a car run may maximise, in the card's own words. NO ``cz``: the area
#: is a design variable on every car family now, and a downforce COEFFICIENT
#: is referenced to the very area being searched — ``carwing`` refuses it
#: outright there, so offering it would be offering a run that cannot be
#: built. What is left is a FORCE, a RATIO and a TASK, and the wording says
#: what each one costs.
#:
#: ``laptime`` is the task, and it is the only entry here that is CONDITIONAL
#: — :func:`car_objective_options` drops it wherever no circuit is stated.
#: Both car problem classes refuse it by name without one ("objective
#: 'laptime' needs a circuit"), so a menu that could send it there would be a
#: menu that promises a run the registry declines at the Run button.
CAR_OBJECTIVE_LABELS = {
    "efficiency": "efficiency CZ/CD (= downforce per unit drag)",
    "downforce": "downforce, in newtons",
    # the epsilon-constraint half of the same trade, and the entry to reach
    # for beside "Downforce, at least": with a floor stated, "drag" asks the
    # question a race engineer has — make at least F newtons and pay the
    # least drag doing it. carwing.CAR_OBJECTIVES carries the measurement
    # (98.5 % of the Pareto front against efficiency's 38.5 % under the SAME
    # floor) and the reason the ratio is the worse half of the pair.
    "drag": "drag, in newtons (minimised — state a downforce floor below)",
    "laptime": "lap time round the chosen circuit (minimised)",
    # the one entry that ADDS the two forces instead of trading them. It is
    # the total load the wing puts into the car, and it is a RATCHET —
    # carwing.CAR_OBJECTIVES carries the measurement, and the note under the
    # select says so before it is picked rather than after it has answered.
    "downforce_plus_drag": "downforce + drag, in newtons (the total load)",
}

#: why an objective is not offered — the same contract as
#: :data:`WINGLET_OPTION_WHY`, read by :func:`missing_options_note`, so a menu
#: never just loses an entry.
CAR_OBJECTIVE_WHY = {
    "laptime": "a lap time is a property of a wing ON A CIRCUIT — choose a "
               "circuit above and this becomes the last entry here",
}


#: WHICH spelling of "no circuit" the card holds. ``api.CAR_TRACK_OFF`` is a
#: TUPLE of accepted spellings (its helper lower-cases and strips), and a card
#: has to store exactly one of them. Written as a literal for the reason
#: :data:`BUILDER_DEFAULTS` gives for the mount — this is a label table a test
#: diffs against the physics, and a table that computes its own keys cannot be
#: diffed against anything. ``test_car_lap_control`` asserts membership.
_CAR_TRACK_OFF_VALUE = "off"

#: the circuits this shell can name, in the card's own words. ``off`` is the
#: mode string ``api.CAR_TRACK_OFF`` spells, not an absent key: a flag loop
#: that skips ``None`` cannot carry "no mission" as a value, and the card has
#: to be able to say it out loud so the state can hold it.
#:
#: One shipped lap, and its label carries the word SYNTHETIC because the
#: layout is invented and every number the lap produces is a property of it
#: (RESULTS_SESSION64_CARSECTION section 8 opens on exactly that limitation).
CAR_TRACK_LABELS = {
    "off": "no circuit — score the wing on its own",
    "synthetic": "synthetic reference lap (3280 m; hairpin, medium, fast, "
                 "three straights) — NOT a real circuit",
}


def car_track_options(ch: dict) -> dict:
    """The Circuit menu for these choices, or ``{}`` where there is none.

    Empty means DO NOT DRAW THE ROW: the designed-endplate family has no
    ``track_spec``, ``car_spec`` or ``track_points`` field at all, so
    ``api.CAR_ENDPLATE_KEYS`` declares neither lap key and ``check_flags``
    refuses both outright. Asked from the registry rather than from the
    choices, for the reason `a-shell-must-not-infer-a-mode-from-a-vector`
    gives: which families can time a lap is the registry's answer, and a
    second copy of it here is a copy that goes stale.

    The ORDER is the label table's, which is "off" first: a card opens on the
    published behaviour, and a circuit is a mission the user adds.
    """
    if not car_lap_available(ch):
        return {}
    from aerobo import api

    keys = [_CAR_TRACK_OFF_VALUE, *api.CAR_TRACKS]
    return {k: CAR_TRACK_LABELS[k] for k in keys if k in CAR_TRACK_LABELS}


def car_track_of(ch: dict) -> str | None:
    """The circuit these choices state, or None for "no circuit".

    One reader, so the card, :func:`car_flags` and
    :func:`_normalise_car_lap` can never disagree about whether a lap was
    asked for. Anything the registry does not name as a circuit reads as OFF
    rather than being passed on to be refused: a stale value in the state is
    the shell's problem to fall back from, not the solver's to raise about.
    """
    from aerobo import api

    v = ch.get("car_track")
    if v is None or str(v).strip().lower() in api.CAR_TRACK_OFF:
        return None
    return str(v) if str(v) in api.CAR_TRACKS else None


def car_lap_available(ch: dict, problem_name: str | None = None) -> bool:
    """Can the family these choices derive to be given a circuit at all?

    Read off ``api.accepted_flags`` — the registry's own declaration — and
    not off the choices, because "which car families carry a lap" is a fact
    about ``CarWingProblem`` / ``CarWingMultiProblem`` having ``track_spec``
    and ``CarWingEndplateProblem`` not having it. Four families declare the
    two lap keys; the two designed-endplate ones do not.
    """
    from aerobo import api

    if ch.get("medium") != "track":
        return False
    name = problem_name or derive_problem(ch)[0]
    return api.CAR_LAP_KEYS[0] in api.accepted_flags(name)


def car_default_track_points() -> int:
    """How many speeds a lap samples with no flag — read, never pasted.

    Off the dataclass field, so a family that re-chooses its sampling cannot
    leave a constant here saying otherwise (the same rule
    :func:`car_default_v_ms` follows).
    """
    from aerobo.carwing import CarWingProblem

    return int(CarWingProblem.track_points)


def car_objective_options(ch: dict, problem_name: str | None = None) -> dict:
    """The Maximise menu for these choices — the entries that can be RUN.

    THE CIRCUIT UNLOCKS THE OBJECTIVE, and not the other way round. Both
    orders were available and this one is chosen, for four reasons:

    * a circuit is a MISSION, and a mission is stated before it is scored.
      ``cartrack.py``'s own docstring makes the argument ``mission.py`` makes
      for air — "maximise CZ against an allowance" is arbitrary until a car
      and a lap are stated, at which point it becomes "go round faster". The
      shell already asks every other family for its mission before asking
      what to maximise;
    * the other order answers a question nobody asked. Picking ``laptime``
      would have to INSTALL a circuit, and every number in the answer would
      then be a property of ``cartrack.synthetic_lap`` — a synthetic layout
      the user never chose. RESULTS_SESSION64_CARSECTION section 8 opens with
      "THE ANSWER IS THE CIRCUIT'S", and on a layout with the straights
      doubled the same study's winner is 0.79 s SLOWER than the section it
      beat. A mission that appears as a side effect of a scoring menu is the
      one thing that finding forbids;
    * de-selecting would then be ambiguous. If the objective installed the
      circuit, moving off ``laptime`` would have to decide whether to remove
      it — a second silent act, on a question the user had not been asked;
    * and a circuit is worth having WITHOUT the objective. With a track
      stated and ``efficiency`` maximised the run still reports the lap time,
      the per-speed rows and the aero balance, so the wing is still measured
      against the task even when it is not scored on it. That is not true in
      reverse: the objective is useless without the circuit.

    The consequence the card owes the user is that switching the circuit OFF
    while ``laptime`` is selected must not leave a menu holding a value it no
    longer offers — :func:`_normalise_car_lap` is the net, and it is the same
    stale-menu rule ``_normalise_winglet_blend`` obeys one control across.
    """
    out = dict(CAR_OBJECTIVE_LABELS)
    if car_track_of(ch) is None \
            or not car_lap_available(ch, problem_name):
        out.pop("laptime", None)
    return out

#: There is no drag-budget MENU any more. Drag was a three-way choice
#: (coefficient / force / none) whose default handed the optimiser an
#: allowance to spend, and an allowance handed to a maximiser is what picks
#: the answer. It is now one optional number in ``_car_limit_rows``: blank is
#: no drag constraint, and a number is a ceiling in newtons.


#: choices that carry a NUMBER rather than select a family — every one of
#: them is typed into a ``_num_row`` field, becomes a flag, and no MENU
#: anywhere depends on its value. Writing one must therefore NOT rebuild the
#: card its own field lives in: a rebuilt input loses the focus and swallows
#: the rest of the number (typing "7.5" stored 7, then nothing; "1.25" stored
#: 1.2). Every shell that owns a builder card reads this list — V3 mirrors it
#: in ``gui.v3.stages.wing.VALUE_KEYS`` (which also carries its own tail
#: numbers), and a test holds the two together.
NUMBER_CHOICE_KEYS = ("car_drag_budget_n", "car_downforce_min_n",
                      # the plate's CANT. It changes no menu — it refines one
                      # shape already chosen — so typing into it must not
                      # rebuild the card underneath the cursor.
                      "car_endplate_cant_deg",
                      # the LAP's sampling count. A number, and deliberately
                      # not the circuit beside it: the circuit decides what
                      # the Maximise menu offers and so must rebuild the
                      # card, while this only makes the same run cost more.
                      "car_track_points",
                      "car_endplate_blend_frac",
                      "car_endplate_chord_min_m", "car_endplate_chord_max_m",
                      "tandem_dx_m", "tandem_dz_m", "tandem_b_rear_m",
                      "tandem_fin_boom_m")


def _num_row(label: str, key: str, ch: dict, set_choice, *, suffix: str,
             step: float = 0.05, placeholder: str = "",
             lo: float | None = None, hi: float | None = None) -> None:
    """One optional NUMBER a card asks for, in the card's own units.

    Blank means "not stated" and sends no flag at all, which is what keeps an
    untouched card bit-for-bit the published problem — so the field's empty
    state has to survive a re-render, and clearing it has to clear the choice.

    The key belongs in ``NUMBER_CHOICE_KEYS``: that is what stops the shell
    rebuilding this field from this field's own handler, mid-number.

    ``lo``/``hi`` bound a value whose SOLVER has a range — the blend fraction
    is a fraction, and typing 1.5 into it used to reach geometry.span_path,
    raise, and come back as a whole run of penalties reading "solver failure:
    blend_frac must be in [0, 1]". A field that can only be asked a question
    the model can answer is the fix; the range is stated in the label as well,
    because a silently-clamped number is its own surprise.
    """
    from nicegui import ui

    with ui.row().classes("w-full items-center gap-3 no-wrap"):
        ui.label(label).classes("text-xs w-32 shrink-0 opacity-70")
        num = ui.number(value=ch.get(key), step=step, placeholder=placeholder,
                        on_change=lambda e: set_choice(
                            key, _opt_float_in(e.value, lo, hi))) \
            .props("outlined dense").classes("grow min-w-0")
        if lo is not None:
            num.props(f'min="{lo:g}"')
        if hi is not None:
            num.props(f'max="{hi:g}"')
        ui.label(suffix).classes("text-xs opacity-60 shrink-0")


def _car_controls(ch: dict, set_choice, *, limits: bool = True,
                  limits_below: bool = False,
                  size_note: bool = True, mount: bool = True,
                  plate_chord: bool = True) -> None:
    """Car rear-wing configuration: the mount, and what the run maximises.

    WHAT IS NOT HERE, and why. This card used to ask four numbers the design
    box also asks — the span band and the area band — and it used to carry a
    switch deciding whether the area was designed at all. Both are gone:

    * HOW WIDE and HOW BIG are BANDS ON DESIGN VARIABLES, so they are the
      ``b_m`` and ``S_m2`` rows of the design box and are stated only there.
      Asking them twice was not a duplication, it was a defect in both
      directions — widening the box row gave a run made entirely of "bounds
      violation" refusals, and typing a band here left the box on screen
      showing one band while the run searched another (api._CAR_SIZE_ROWS
      carries both measurements).
    * WHETHER the area is designed is not a question any more. It always is.
      A rear wing's size is a design variable exactly as its width is.

    ...and four more that used to be here and are not:

    * the ENDPLATES switch. The Mount already answers it: a plate that
      carries the car is designed, a plate on a pylon-borne wing is a fence.
      Its two remaining questions (the section, the root blend) are asked
      under the Mount answer that creates them, in
      :func:`_car_plate_controls`.
    * the CIRCUIT, and with it the lap-time objective. The lap is the only
      score here that needs no weights, and it is still in the engine —
      ``cartrack.py``, the ``car_track`` / ``track_points`` flags and every
      lap row of the report are untouched, so a script or a saved session
      still flies one (:func:`car_flags` still sends it). The CARD stopped
      asking. With no circuit settable, :func:`car_objective_options` drops
      ``laptime`` on its own.
    * the ELEMENTS switch. A slotted two-element wing is a different solver
      family (``carwing_multi``) and four more design rows; it is still
      registered, still derivable (``_car_is_two_element``) and still flown
      by anything that sets the choice, but the card does not offer it.
    * the PLATE CHORD's metre band, which is a bound on a design row and
      belongs beside that row. V3 draws it under its design box and passes
      ``plate_chord=False``; V1 and V2 have no such place and keep it here.

    ``limits`` draws the drag ceiling and the downforce floor. V3 passes
    False and draws its own richer pair (a switch that opens each limit at
    the incumbent's own force) immediately below this block, so it passes
    ``limits_below=True`` as well. One shell, one home — never both.

    ``limits_below`` is only about WHERE THE NOTES POINT. Each objective's
    note ends by naming the limit that is its other half, and that sentence
    has to be true: with the rows drawn by this function it says "directly
    below", with them drawn by the caller directly below it says the same,
    and it would say "on another card" only for a shell that really put them
    on one. None does any more — a car objective is one half of a trade, and
    the number that is its other half must be typable where the objective is
    chosen. It used to be a tab away in V3, and "state a downforce floor"
    then resolved to a sentence rather than to a field.

    ``size_note`` is the same rule for the SIZE paragraph: V3 states it under
    its planform row, where the size is actually asked, so it passes False.

    ``mount`` is the same rule again, for position rather than for
    duplication: the mount is the first question the card asks, and V3 draws
    it at the TOP of its configuration list itself, so it passes False here.
    """
    from nicegui import ui

    from aerobo import api

    with ui.column().classes("w-full gap-1"):
        if mount:
            _car_mount_controls(ch, set_choice)
        # THE SIZE, said ONCE PER SHELL. V3 asks it under the planform row
        # (`gui.v3.stages.wing._size_controls`) and passes ``size_note=False``
        # here, because printing the same paragraph under the mount as well
        # told the reader the same thing twice on one screen, three
        # paragraphs apart. V1 and V2 have no such row, so they keep it.
        if size_note:
            ui.label("Both dimensions of this wing are designed — its "
                     "reference AREA and its SPAN. Their bands are rows of "
                     "the design box (S_m2 and b_m), in metres and m², "
                     "because what bounds a rear wing is a regulation or "
                     "the bodywork. The endplate's height is a length for "
                     "the same reason: a plate reaches the car or it does "
                     "not.").classes("text-[11px] opacity-60")

        # --- what the run maximises. NOT a coefficient: CZ is referenced to
        # the very area being searched, so maximising it just shrinks the
        # wing, and carwing refuses it outright against a free area.
        offered = car_objective_options(ch)
        # CLAMPED TO WHAT IS OFFERED, and that is not defensive padding — a
        # select handed a value outside its options raises, and nicegui's
        # ValueError takes the whole card down ("this view failed to
        # render"). It happens for real: a state holding ``laptime`` reaches
        # a family with no lap — a saved session, or a shell whose
        # apply_choices does not run `_normalise_car_lap` — and the reader
        # then loses every control on the screen, not just this one. Same
        # rule the tip-device and planform selects already obey one card up.
        objective = ch.get("car_objective") or api.CAR_DEFAULT_OBJECTIVE
        if objective not in offered:
            objective = api.CAR_DEFAULT_OBJECTIVE
        with ui.row().classes("w-full items-center gap-3 no-wrap"):
            ui.label("Maximise").classes("text-xs w-20 shrink-0 opacity-70")
            ui.select(offered, value=objective,
                      on_change=lambda e: set_choice("car_objective", e.value)) \
                .props("outlined dense").classes("grow min-w-0")
        if objective == "efficiency":
            ui.label("Efficiency is downforce per unit drag, and the area "
                     "cancels out of it — which is what makes it the one "
                     "score that is well posed with the size designed and "
                     "nothing budgeted. It has a real interior peak (around "
                     "0.14 m² of area) rather than running to a bound. What "
                     "it does NOT do is make much downforce: at that peak "
                     "the wing makes about 193 N against 486 N at 0.4 m². "
                     "State a downforce floor and the answer becomes the "
                     "most efficient wing that still makes what you need. "
                     + _limit_home_note("Downforce, at least", limits or limits_below)) \
                .classes("text-[11px] opacity-60")
        elif objective == "laptime":
            ui.label("The lap is the only score here that is a TASK rather "
                     "than a figure of merit, and it is the only one that "
                     "needs no weights: downforce is worth exactly what it "
                     "saves in the corners minus what its drag costs on the "
                     "straights, and the circuit supplies that exchange rate "
                     "instead of you choosing it. The run maximises minus "
                     "the lap time, so a score difference of 0.1 IS a tenth "
                     "of a second. Everything the answer says is a property "
                     "of the circuit — pick a different layout and a "
                     "different wing wins.").classes("text-[11px] opacity-60")
        elif objective in ("drag", "cd"):
            ui.label("Drag, minimised — the other half of the same trade, "
                     "and the half that needs a DOWNFORCE FLOOR beside it or "
                     "it is not a question: with nothing to make, the "
                     "least-drag wing is the smallest one (measured over "
                     "3000 draws of this box, 43 N at CZ 0.058). State the "
                     "floor and this becomes 'make at least that much and "
                     "pay the least drag doing it' — which reaches 98.5 % of "
                     "the downforce/drag trade against efficiency's 38.5 % "
                     "under the SAME floor, because a ratio under a floor "
                     "climbs off it chasing a better quotient. "
                     + _limit_home_note("Downforce, at least", limits or limits_below)) \
                .classes("text-[11px] opacity-60")
        elif objective == "downforce_plus_drag":
            ui.label("Downforce PLUS drag, added rather than traded — the "
                     "total aerodynamic load the wing puts into the car, "
                     "which is what the mount carries and what a "
                     "braking-limited case wants (the drag retards the car "
                     "directly, the downforce through the tyres). Read the "
                     "rest before picking it: this one RIDES ITS BOUNDS. "
                     "Measured on the registered box, the answer takes 100 % "
                     "of the area row and 100 % of the alpha sweep on both "
                     "families, with every margin slack — more area at the "
                     "same CZ makes more of both terms, so it spends all the "
                     "area you allow it. That is not the “cz” defect (which "
                     "moves opposite to the physics); it is the physics, and "
                     "the area band is your regulation. A drag ceiling is "
                     "what turns the SPAN into an answer instead of a bound: "
                     "at 81.5 N the drag sits exactly on the ceiling and the "
                     "span moves to 58 % of its band. "
                     + _limit_home_note("Drag, no more than", limits or limits_below)) \
                .classes("text-[11px] opacity-60")
        else:
            ui.label("Downforce in newtons, which is a force and not a "
                     "coefficient — the right target once the reference area "
                     "is designed. With no drag ceiling it is maximised by "
                     "spending drag until the section stalls, so it is worth "
                     "pairing with one. "
                     + _limit_home_note("Drag, no more than", limits or limits_below)) \
                .classes("text-[11px] opacity-60")

        if limits:
            _car_limit_rows(ch, set_choice)
        if plate_chord:
            _car_plate_chord_rows(ch, set_choice)


def _car_plate_chord_rows(ch: dict, set_choice) -> None:
    """The designed plate's chord, in METRES — its second band.

    The plate's chord answers to two bands at once and this is the one a
    regulation is written in. Its DESIGN row states it as a multiple of the
    wing's TIP chord, which is the right way to keep a plate proportioned to
    its wing and useless as a bound, since the tip chord moves with the span,
    the taper and the chord law.

    Drawn only where the plate is designed, because only that family has a
    plate chord to bound (a fence carries the wing's own chord). V3 draws
    these two under its DESIGN BOX, beside the ratio row they clip, and
    passes ``plate_chord=False`` here; V1 and V2 have no box view.
    """
    from nicegui import ui

    if not (bool(ch.get("car_endplates")) and not _car_is_two_element(ch)):
        return
    _num_row("Plate chord ≥", "car_endplate_chord_min_m", ch,
             set_choice, suffix="m", placeholder="unbounded", lo=0.0)
    _num_row("Plate chord ≤", "car_endplate_chord_max_m", ch,
             set_choice, suffix="m", placeholder="unbounded", lo=0.0)
    ui.label("What flies is the ratio row's chord clipped into this band: "
             "the metre band wins where they disagree, every point of the "
             "design box stays flyable, and a design whose chord was clipped "
             "says so on the results page (chord asked for, chord flown). "
             "Blank = unbounded, the published problem.") \
        .classes("text-[11px] opacity-60")


#: NO CIRCUIT CONTROL, and no lap-budget note. Both used to be drawn here
#: and both are gone from every card: a rear wing's circuit is the one score
#: in this repo that needs no weights, and it is still in the ENGINE —
#: ``cartrack.py``, ``api``'s ``car_track`` / ``track_points`` flags, every
#: lap row of the report and :func:`car_flags`'s own branch that sends them
#: are all untouched, so a script or a saved session still flies a lap and
#: still scores on it. What was removed is the question. With no circuit
#: settable from a card, :func:`car_objective_options` drops ``laptime`` on
#: its own, and :func:`car_track_of` reads "off" for every card-built choice
#: dict. To put it back, draw a select over :func:`car_track_options` writing
#: ``car_track`` (a MENU-changing key, so it must rebuild its card) and a
#: ``_num_row`` over ``car_track_points``.


def _limit_home_note(field: str, limits: bool) -> str:
    """Where the limit this objective wants is actually typed.

    THE DEFECT THIS CLOSED, and how it was finally closed properly. Every
    car objective is one half of a trade whose other half is a LIMIT, and the
    two were asked in different views: V3 drew the Maximise select on its
    configuration card and both limit rows under its DESIGN BOX, a whole tab
    away, so choosing "efficiency" and looking for where to say how much
    downforce is worth having found nothing. A POINTER was the first fix and
    it was not enough — "state a downforce floor" then resolved to a sentence
    about another tab rather than to a field. The rows have MOVED (V3 draws
    them under the select itself and passes ``limits_below=True``), so every
    shell now has them adjacent and this note only ever says "directly
    below". It is kept because the sentence still has to be written, and
    because a shell that puts them elsewhere again gets the honest wording
    rather than a lie.
    Reported as "no min downforce appears when drag", and it is not a missing
    field — the field is one view away and nothing said so.

    Duplicating the field is the wrong fix (one question, one home), so what
    the objective carries is a POINTER, and only when the fields are not
    directly below it.
    """
    if limits:
        return f"The “{field}” box is directly below."
    return (f"“{field}” is in the LIMITS block under the design box — the "
            f"same place this shell keeps every other limit.")


def _car_limit_rows(ch: dict, set_choice) -> None:
    """The two car LIMITS: a drag ceiling and a downforce floor, both in N.

    Neither is on by default, and that is the point. This family used to ship
    a drag ALLOWANCE (CD 0.11, restated as 81.5 N once the area moved) and the
    answer spent it: a budget handed to a maximiser is a number the answer
    rides, not a limit it respects — measured on the certified optima, the
    drag margin sits at 0.048 and 0.035 while binding almost nowhere else in
    the box. So nothing is budgeted unless you say so.

    Blank means no constraint at all, which is why they are typed rather than
    switched: a limit you have not stated is not a limit set to zero.
    """
    from nicegui import ui

    _num_row("Drag, no more than", "car_drag_budget_n", ch, set_choice,
             suffix="N", step=5.0, placeholder="no limit", lo=0.0)
    ui.label("A ceiling in newtons, which is the allowance that means "
             "something when the reference area is designed: a COEFFICIENT "
             "budget is referenced to the very area being searched. Blank is "
             "no drag constraint — the score already prices drag, so nothing "
             "runs away without one.").classes("text-[11px] opacity-60")
    _num_row("Downforce, at least", "car_downforce_min_n", ch, set_choice,
             suffix="N", step=25.0, placeholder="no floor", lo=0.0)
    ui.label("A floor in newtons. Efficiency alone answers a question most "
             "people do not mean: its peak is a small wing making little "
             "downforce (about 193 N at 0.14 m², against 486 N at 0.4 m²). "
             "A floor says how much is worth having, and the answer then "
             "sits ON the floor — spend exactly the area the downforce "
             "demands.").classes("text-[11px] opacity-60")



def tandem_stagger(ch: dict) -> tuple:
    """The stagger the pair will actually fly [m], stated or derived.

    Derived exactly the way api._tandem_planform_kwargs derives it — the
    published fractions of whatever span is in play — so the card can show
    the two numbers the run will use even before anyone types one.
    """
    from aerobo import api

    b = float(ch.get("span_m") or 10.0)
    fx, fz = api._TANDEM_STAGGER_FRACS
    dx = ch.get("tandem_dx_m")
    dz = ch.get("tandem_dz_m")
    return (fx * b if dx is None else float(dx),
            fz * b if dz is None else float(dz))


def _fin_boom_default(ch: dict) -> float:
    """The boom this pair flies with nothing typed [m].

    Read off ``fin.tandem_fin_station`` — the one author of the station —
    against the stagger the card is already showing, rather than restated
    here: the package's default is a fraction of the stagger, so a card that
    printed a metre constant would go stale the first time either moved.
    """
    from aerobo import fin as _fin

    dx, _dz = tandem_stagger(ch)
    x_qc, _z = _fin.tandem_fin_station(dx, 0.0)
    return float(x_qc) - float(dx)


def _tandem_controls(ch: dict, set_choice, with_span: bool = True,
                     with_fin: bool = True) -> None:
    """The pair's LAYOUT: where the rear wing sits, and how wide it is.

    Configuration, not design variables — which is the whole point of the
    card. The mutual induction this family exists to model is a function of
    exactly these numbers, so leaving them to be derived from an
    optimiser-chosen span meant the run answered a question about a layout it
    had picked itself.

    The rear SPAN is asked only while the planform is fixed. Free it and both
    spans become design variables — one design-box row per wing — and this
    field would be a second answer to that question, so it is not shown
    (:func:`tandem_flags` refuses to send it there for the same reason).

    ``with_span`` is how a shell that asks for the SPANS somewhere else says
    so. V3 asks both of them together on its size card — a span is a span
    whichever wing carries it, and the two belong side by side — so it takes
    the field out of here rather than asking the same question twice, three
    cards apart. V1 and V2 have no such card and keep it.

    ``with_fin`` is the same rule for the BOOM. The pair's fin station is a
    question about the FIN, and V3 has a card for the fin — it draws the
    surface the boom sizes and places, so the field and the four numbers it
    moves belong on one card. Here the field sat under a paragraph about the
    rear wing's stagger and above one about the rear wing's span, with no
    sentence of its own and nothing anywhere reporting the fin it moved. V1
    and V2 have no vertical-tail card and keep it.
    """
    from nicegui import ui

    dx, dz = tandem_stagger(ch)
    b = float(ch.get("span_m") or 10.0)
    searched = not planform_resizable(derive_problem(ch)[0])
    with ui.column().classes("w-full gap-1 pl-1"):
        _num_row("Rear wing, aft by", "tandem_dx_m", ch, set_choice,
                 suffix="m", step=0.5, placeholder=f"{dx:g}")
        _num_row("Rear wing, above by", "tandem_dz_m", ch, set_choice,
                 suffix="m", step=0.1, placeholder=f"{dz:g}")
        if not searched and with_span:
            _num_row("Rear wing span", "tandem_b_rear_m", ch, set_choice,
                     suffix="m", step=0.5, placeholder=f"{b:g}")
        # ...AND WHERE THE FIN STANDS. The third length of this layout, asked
        # here because it is the same kind of answer as the two above — a
        # boom, a decision about the airframe — and because this is the card
        # that already draws the pair. It was a package CONSTANT until now,
        # and one that moved twice (between the wings, then onto the rear
        # wing) without the builder ever being asked.
        if with_fin:
            _num_row("Fin, aft of the rear wing by", "tandem_fin_boom_m", ch,
                     set_choice, suffix="m", step=0.5,
                     placeholder=f"{_fin_boom_default(ch):g}")
        ui.label(f"Where the rear wing's quarter-chord sits relative to the "
                 f"front one's — a fuselage length, a boom, a wing box. It is "
                 f"YOURS to state: nothing in the run moves it, not even "
                 f"freeing the span. Blank leaves the published layout for "
                 f"the span in play (currently {dx:g} m aft, {dz:g} m up). "
                 f"With a tip device on each wing the vertical gap is also "
                 f"what keeps the two devices apart — close it and the pair "
                 f"is a joined wing, which this solver refuses rather than "
                 f"answers for.").classes("text-[11px] opacity-60")
        ui.label(
            ("The two wings share a fuselage, not a span: the rear one can be "
             "the narrower wing, and blank leaves it as wide as the front "
             f"({b:g} m). Its area is the pair's own split, so a shorter rear "
             "wing is a stubbier one — which is the trade, since it is the "
             "span that sets how much of the front wing's downwash it sits "
             "in." if with_span else
             "The two wings share a fuselage, not a span, and both spans are "
             "asked together on the size card — a span is a span whichever "
             "wing carries it.")
            if not searched else
            "Both spans are design variables here — one row per wing in the "
            "design box below — so neither is typed."
        ).classes("text-[11px] opacity-60")


def modifier_available(ch: dict, mod: str) -> bool:
    """Does the family these choices select accept MODIFIER ``mod``?

    The GUI asks this to decide whether a modifier control stays live. The
    chord law composes with water, the track, winglets, tails, tandem pairs
    and free planforms alike (only the pure 2-D section problem refuses it,
    having no wing planform to reshape); the flight state composes with every
    AIR family that trims to a weight — the water families already fly speed
    and depth as design variables, and the car's speed is a track condition.
    """
    from aerobo import api
    name = _derive_family(ch)[0]
    # asked of the FAMILY, so the answer does not depend on which other
    # modifier happens to be on (they are independent by construction)
    return api.with_modifiers(name, {mod}) is not None


def chord_available(ch: dict) -> bool:
    """Does the family these choices select have a free-chord-law twin?"""
    return modifier_available(ch, "chord")


def flight_available(ch: dict) -> bool:
    """Does the family these choices select accept a free flight state?"""
    return modifier_available(ch, "flight")


def size_available(ch: dict) -> bool:
    """Does the family these choices select accept free span + area?"""
    return modifier_available(ch, "size")


def chord_base_of(name: str) -> str | None:
    """Inverse of :func:`chord_twin_of`: the base problem of a chord twin."""
    from aerobo import api
    for base, twin in api.CHORD_TWINS.items():
        if twin == name:
            return base
    return None


def chord_law_optimiser_note(problem_name: str) -> str:
    """What the chord law costs in SEARCH STRATEGIES, as a sentence, or "".

    The exact-gradient adjoint has no derivative through the area-preserving
    chord law, so a chord-law twin offers one optimiser fewer than the base
    it extends. That was a fair trade while the law was opt-in; now that
    every shell OPENS on it (:data:`BUILDER_START`) the entry is missing
    before the user has chosen anything, and a list that silently got shorter
    is exactly the kind of thing nobody notices. Asked of the registry, not
    from a table, so it stays right if the gate ever moves.
    """
    from aerobo import api

    base = chord_base_of(problem_name)
    if not base:
        return ""
    lost = [o for o in api.compatible_optimisers(base)
            if o not in api.compatible_optimisers(problem_name)]
    if not lost:
        return ""
    return ("The chord law costs "
            + ", ".join(api.OPTIMISER_SPECS[o].display for o in lost)
            + " — there is no exact gradient through the area-preserving "
              "chord law. Straight taper brings it back.")


def active_specials(ch: dict) -> list[str]:
    """Speciality keys whose value selects a FAMILY other than the plain wing.

    ``planform = "free"`` is deliberately not one of them: that value is the
    SIZE MODIFIER, which composes with every family that declares it. Only
    the legacy ``"aircraft"`` value selects a family of its own.
    """
    return [k for k in SPECIAL_KEYS
            if ch.get(k, BUILDER_DEFAULTS[k]) != BUILDER_DEFAULTS[k]
            and not (k == "planform"
                     and ch.get(k) in PLANFORM_MODIFIER_VALUES)]


#: why a family cannot take a modifier — shown instead of dropping it
_MODIFIER_REFUSALS = {
    "size_ws_free": ("free span + searched W/S ignored — this family has "
                     "no wing-loading sizing variant yet: the area follows "
                     "the loading through Raymer's weight loop, and the "
                     "families wired for that are the air ones "
                     "(api._WING_LOADING_FAMILIES)."),
    "size_ws": ("free span at fixed W/S ignored — this family has no "
                "wing-loading sizing variant yet. The families that carry it "
                "are the single-wing air ones (api._WING_LOADING_FAMILIES); "
                "everywhere else the size is either calibrated geometry or "
                "already in the design vector."),
    "size": ("free span + area ignored — this family has no weight-coupled "
             "sizing variant. What the modifier costs is a STRUCTURAL "
             "weight the size implies, and the loop these families close is "
             "Raymer's aircraft one: there is no calibrated structural "
             "weight for a hydrofoil or a rear wing, so a weight-coupled "
             "size there would be a number with nothing behind it. The size "
             "itself is still yours — state it on the size card (the car's "
             "span is a design variable of its own) — and the published "
             "aircraft-sizing problem already carries span and area in its "
             "design vector."),
    "chord": ("free chord law ignored — the 2-D section problem has no wing "
              "planform to reshape. To co-design the SECTION and the chord "
              "distribution, switch winglets on: that is the coupled "
              "CST + planform problem, and the tip device may optimise to "
              "zero height."),
    "flight": ("free flight state ignored — this family does not take one. "
               "The water solvers already fly SPEED and DEPTH as design "
               "variables (that IS their flight state), the car rear wing's "
               "speed is a track condition its drag and deflection budgets "
               "are quoted at, and the 2-D section problem has no trim "
               "target to move."),
}


def derive_problem(ch: dict) -> tuple[str, list[str]]:
    """Map builder choices -> (problem_name, capability notes).

    Two stages. The speciality keys select a solver FAMILY (deterministic
    priority, :func:`_derive_family`); then the MODIFIERS — the free chord
    law and the free flight state, each of which composes with any family
    that supports it — are switched on one at a time through
    ``api.add_modifier``, which is the registry's own composition rule rather
    than a table kept here. Every selected-but-unavailable feature produces a
    note naming what is missing (extension point), never a silent drop.
    """
    from aerobo import api

    name, notes = _derive_family(ch)
    # FLY THE CHOSEN SECTION. On the families whose section is a single
    # table this is a flag and changes no name (gui/v3/config.py sends it);
    # on the ones that SELECT their section by thickness it drops the t/c
    # variable, which makes it a different problem — so the choice is made
    # here, in the one place choices become a problem name.
    if ch.get("fly_section"):
        twin = api.chosen_section_problem(name)
        if twin is not None:
            name = twin
            notes.append(
                "Flies the section chosen upstream instead of searching a "
                "thickness: its polar (and, in water, its Cp_min table) come "
                "from one cached XFOIL sweep of that shape, so t/c leaves "
                "the design vector.")
    for mod in api.MODIFIERS:
        key, on_value = MODIFIER_ON[mod]
        if ch.get(key, BUILDER_DEFAULTS[key]) != on_value:
            continue
        twin = api.add_modifier(name, mod)
        if twin is not None:
            name = twin
        else:
            notes.append(_MODIFIER_REFUSALS[mod])
    # THE WING'S CANT, asked of the registry and not of a list of families
    # that can search it: every wing+tail configuration has a free-cant twin
    # and nothing else does yet, so a user who asks for one anywhere else
    # gets the STATED pair and a note naming what would carry it.
    if ch.get("wing_cant", "fixed") != "fixed" \
            and api.searched_cant(name) != ch.get("wing_cant"):
        notes.append(
            "the wing's dihedral and sweep are not design variables in this "
            "family — they stay stated values (0 deg unless you type one). "
            "The LATTICE-backed families search them (every wing+tail "
            "configuration, and the nonplanar tandem pair): a dihedral is "
            "out-of-plane geometry, and a lifting line has no out-of-plane "
            "shape for it to act on.")
    # the FIXED BLEND is a value, so it cannot select a problem — which means
    # a family with no blended tip device would simply not draw it. Say so:
    # a shape that quietly does not happen is the one thing these notes exist
    # to prevent.
    if ch.get("winglet_blend_frac") and ch.get("winglets", "none") != "none":
        spec = api.PROBLEM_SPECS.get(name)
        if spec is not None and api.WINGLET_BLEND_KEY not in spec.flags:
            notes.append(
                "blended tip device ignored — this family's solver draws the "
                "wing/device corner as a corner, so there is no transition "
                "for a blend fraction to shape. The tip device itself (its "
                "height and its cant) is designed as usual.")
    return name, notes


#: airfoil menu entries the tail family has a combined solver for.
#: ``fixed`` / ``tc_sweep`` map onto the nonplanar wing+tail problem (t/c
#: selects the NACA 24XX polar family member; sweep is not modelled by the
#: VLM, exactly as in the winglet family). The live-XFOIL CST entries select
#: the SECTION co-design family (wingtail_section.py), where wing, tip device
#: and tail all fly the one designed aerofoil. The pre-optimised section
#: LIBRARY (``coupled``) is the remaining extension point and says so.
_TAIL_AIRFOILS = ("fixed", "tc_sweep")
_TAIL_SECTION_AIRFOILS = ("section_only", "section_wing")


def _derive_tail(ch: dict, notes: list[str]) -> tuple[str, list[str]]:
    """Solver problem for a configuration WITH a tail.

    The tail used to reset every other feature: it is now the nonplanar
    wing+tail family (api.WING_TAIL_VARIANTS), so the tip device, the
    section thickness, the chord law and the tail's own height compose with
    it. What still has no combined solver says so.
    """
    from aerobo import api

    wl = None
    if ch.get("winglets") in ("free", "capped"):
        wl = ("blended" if ch.get("winglet_type") in BLENDED_TYPES
              else ch["winglets"])
        if wl == "capped" and ch.get("winglet_type") == "raked":
            pass                      # raked IS the capped band (a cant flag)
        if ch.get("winglet_type") == "blended_wing":
            # the wing+tail family carries the 6-D blended tip device, whose
            # turn is confined to the winglet — there is no wing+tail solver
            # for the wing-side transition, so say so rather than draw a
            # transition the solve did not fly
            notes.append(
                "The wing-side transition has no wing+tail solver — the tip "
                "device is flown with its blend confined to the winglet "
                "(deliberate extension point).")
    airfoil = ch.get("airfoil", "fixed")
    tc = airfoil == "tc_sweep"
    section = airfoil in _TAIL_SECTION_AIRFOILS
    if not section and airfoil not in _TAIL_AIRFOILS:
        notes.append(
            "The pre-optimised section LIBRARY has no wing+tail solver — to "
            "design the aerofoil beside the tail, choose the live-XFOIL CST "
            "section instead (that IS a wing+tail problem); to select one by "
            "thickness, use the t/c family.")
        tc = False
    if section:
        # the CST weights ARE the thickness, so the t/c variable is dropped
        # rather than offered alongside them
        tc = False
    if ch.get("planform") == "aircraft":
        notes.append(
            "The published aircraft-sizing problem is its own family and has "
            "no tail — for a weight-coupled span and area BESIDE the tail, "
            "choose 'free span + area' instead (that is the size modifier, "
            "and it composes here).")
    height = "free" if ch.get("tail_height") == "free" else "fixed"
    if height == "free" and ch.get("tail_type") == "t_tail":
        height = "fixed"
        notes.append(
            "A T-tail's height is DERIVED from its fin sizing, so it cannot "
            "also be a design variable — switch to the conventional layout "
            "to make the height free.")
    arm = "fixed" if ch.get("tail_arm") == "fixed" else "free"
    design = ch.get("tail_design", "fixed")
    if design not in api.TAIL_DESIGNS:
        design = "fixed"
    cant = ch.get("wing_cant", "fixed")
    if cant not in api.WING_CANTS:
        cant = "fixed"
    name = api.wing_tail_problem(winglets=wl, tc=tc, arm=arm, height=height,
                                 design=design, cant=cant)
    if section:
        # the section co-design family is generated from the SAME wing+tail
        # table (api.WING_TAIL_SECTION_VARIANTS), including the combination
        # the published lifting-line tail covers — with a designed section
        # there is no lifting-line problem to keep
        # (api.wing_tail_section_problem never returns None for that reason,
        # so there is no lifting-line fallback to compute here)
        # ...AND THE WING'S CANT, which this branch used to drop. It is
        # computed twelve lines up and forwarded to the plain wing+tail
        # family; the section twin takes the same argument
        # (api.wing_tail_section_problem) and WING_TAIL_SECTION_VARIANTS
        # generates the full product over _WING_CANTS — so asking for a
        # designed section AND a searched dihedral silently returned the
        # fixed-cant family (two design rows fewer) and then printed the
        # note "the wing's dihedral and sweep are not design variables in
        # this family", which was false of it. 384 registered problems —
        # every free-cant × CST-section combination — were unreachable from
        # any sequence of clicks for want of this one keyword.
        sec_name = api.wing_tail_section_problem(winglets=wl, arm=arm,
                                                 height=height,
                                                 design=design, cant=cant)
        notes.append(
            "Wing, tip device and tail in ONE nonplanar solve, flying a CST "
            "section shaped by a live (cached) XFOIL sweep each evaluation "
            "(wingtail_section.py). SLOW: seconds per new section. The t/c "
            "control is off because the CST weights ARE the thickness.")
        return sec_name, notes
    if name is None:
        # nothing beyond the published problem is being asked for, so keep
        # the published lifting-line solver and its published numbers — in
        # its plain form, or its designed-tail twin (that family carries the
        # tail's own planform; only a tip device ON the tail needs the
        # nonplanar solver)
        base = "tail (fixed arm)" if arm == "fixed" else "tail"
        if design == "planform":
            base += " [designed tail]"
            notes.append(
                "The tail is designed too: its taper, aspect ratio and "
                "washout join the design vector. Its incidence stays the "
                "trim unknown, so the twist freedom is the washout.")
        return base, notes
    notes.append(
        "Wing + tail in ONE nonplanar solve (wingtail.py) — a different aero "
        "core from the published lifting-line 'tail', measured to agree with "
        "it within 2.4 % in L/D and 0.07 in static margin over the box. "
        "Compare within one solver, not across the two.")
    return name, notes


def _derive_hydro_tail(ch: dict, winglet: bool,
                       notes: list[str]) -> tuple[str, list[str]]:
    """Solver problem for a WATER configuration carrying an elevator.

    The craft's stabiliser is a real second surface in the imaged VLM
    (hydrotail.py), so the tail card's SIZE questions (arm, and whether its
    depth below the foil is optimised) map straight across. Its LAYOUT
    questions do not: there is no fin, no fuselage and no V-tail under the
    water, and the CG is calibrated as a fraction of the arm rather than per
    layout, so those controls say so instead of being silently honoured.
    """
    from aerobo import api

    if ch.get("tail_type", "conventional") != "conventional":
        notes.append(
            "Tail LAYOUT (T-tail / V-tail / canard) is an air-aircraft "
            "choice — the foiling craft's stabiliser is a submerged surface "
            "aft of the main foil, so the layout menu does not apply here.")
    if ch.get("tail_control", "stabilator") != "stabilator" \
            or ch.get("tail_fin_drag"):
        notes.append(
            "The stabiliser is all-moving and there is no fin to charge: "
            "the elevator-chord and fin-drag controls are air-only.")
    arm = "fixed" if ch.get("tail_arm") == "fixed" else "free"
    height = "free" if ch.get("tail_height") == "free" else "fixed"
    design = ch.get("tail_design", "fixed")
    if design not in api.TAIL_DESIGNS:
        design = "fixed"
    notes.append(
        "Hydrofoil + elevator in ONE imaged solve (hydrotail.py): main foil, "
        "tip device and stabiliser share an influence matrix and their "
        "free-surface images, trimmed in lift AND pitch, with cavitation "
        "judged at every panel's own submergence.")
    if design != "fixed":
        notes.append(
            "The elevator is designed too: its taper, aspect ratio and "
            "washout join the design vector"
            + (", it carries a tip device of its own (signed cant, like the "
               "foil's)" if design == "planform+tip" else "")
            + ", and it takes its own chord law wherever the foil has one. "
            "Its incidence stays the trim unknown, so the twist freedom is "
            "the washout.")
    return api.hydro_tail_problem(winglet=winglet, arm=arm, height=height,
                                  design=design), notes


def _derive_family(ch: dict) -> tuple[str, list[str]]:
    """Solver-family stage of :func:`derive_problem` (chord law not applied)."""
    notes: list[str] = []
    specials = active_specials(ch)

    def ignored(keep: str | None):
        for k in specials:
            if k != keep:
                notes.append(
                    f"{_SPECIAL_LABEL[k]} ignored — no combined solver for "
                    "this configuration (deliberate extension point).")

    if ch.get("medium", "air") == "track":
        # the car wing has its own solver family (downforce at a drag and
        # deflection budget). Endplates are NOT the winglet menu: they are a
        # design variable of that problem, so selecting winglets here is
        # redundant rather than ignored.
        for k in specials:
            if k == "winglets":
                notes.append("Endplates are already a design variable of the "
                             "car wing (height is optimised), so the winglet "
                             "menu does not apply.")
            else:
                notes.append(
                    f"{_SPECIAL_LABEL[k]} ignored — no combined solver for "
                    "this configuration (deliberate extension point).")
        # a SECOND ELEMENT on a rear wing is a chordwise problem, not a
        # spanwise one, and it now has a solver (carwing_multi.py). So the
        # tandem switch is READ as the request it really is rather than
        # refused: two surfaces on a car means a slot.
        two = _car_is_two_element(ch)
        if ch.get("system") == "tandem":
            notes.append(
                "A car's rear wing has nothing behind it, so a tandem PAIR is "
                "not what a second surface means here — a SLOTTED "
                "two-element wing is, and it has a solver. Selected: the flap "
                "joins the SECTION (chord fraction, deflection, slot gap and "
                "overlap are four new design variables), not the lattice.")
        if two:
            # the slotted section and DESIGNED endplates have no combined
            # solver: carwing_multi carries the free-height fence, which is
            # also the only endplate its own geometry can justify (one sheet
            # spans both elements). Named rather than silently dropped.
            if ch.get("car_endplates"):
                notes.append(
                    "Designed endplates have no two-element solver — the "
                    "slotted wing carries the free-height fence instead (one "
                    "plate spans both elements; a second sheet at the same "
                    "spanwise station makes the influence matrix singular).")
            return "car rear wing (two-element)", notes
        # designing the endplates ADDS three variables (chord, thickness,
        # toe) and a constraint, so it is a different problem, not a flag
        if ch.get("car_endplates"):
            # ...and WHICH designed-plate family: the plate's cant and its
            # root blend are each stated or SEARCHED, and searching one adds
            # a row, so it is a family and not a flag. Built through the api's
            # own name builder so the shell never spells the bracket itself.
            from aerobo import api as _api
            free = {k for k, key in (("cant", "car_plate_cant"),
                                     ("blend", "car_plate_blend"))
                    if ch.get(key) == "free"}
            if free:
                notes.append(
                    "The plate's "
                    + " and ".join(sorted(
                        "cant" if k == "cant" else "root blend" for k in free))
                    + " is a DESIGN VARIABLE here, so the field that states "
                      "it is gone and the design box carries the row "
                      "instead — one question, one place.")
            return _api.plate_freedom_name("car rear wing + endplates",
                                           free), notes
        return "car rear wing", notes
    if ch.get("medium", "air") == "water":
        # winglets ARE available in water: the nonplanar solver runs the VLM
        # against a free-surface image plane, and the tip device's cant is
        # signed there because submergence (and so the cavitation margin)
        # varies over the device. So is the ELEVATOR: hydrotail.py puts a
        # stabiliser in that same imaged solve, trimmed in pitch. Everything
        # else still has no water solver.
        wet_winglet = ch.get("winglets") in ("free", "capped")
        if wet_winglet and ch.get("winglets") == "capped":
            notes.append("Span-capped accounting has no water variant — the "
                         "water tip device is scored span-free (its cant "
                         "sign, not its projection, is the trade).")
        others = [k for k in specials
                  if k not in ("winglets", "tail")
                  and not (k == "airfoil"
                           and ch.get("airfoil") in _TAIL_SECTION_AIRFOILS)] \
            or (["system"] if ch.get("system") == "tandem" else [])
        if others:
            notes.append("Water medium → cavitation-constrained hydrofoil; "
                         "the remaining wing features above apply to the air "
                         "solvers only.")
        section = ch.get("airfoil") in _TAIL_SECTION_AIRFOILS
        if ch.get("tail"):
            name, notes = _derive_hydro_tail(ch, wet_winglet, notes)
        elif wet_winglet:
            name = "hydrofoil + winglet"
        else:
            name = "hydrofoil"
        if section:
            notes.append(
                "The foil's section is DESIGNED here (hydrofoil_section.py): "
                "eight CST weights, a live XFOIL sweep WITH the minimum "
                "surface pressure, so the cavitation margin is read off the "
                "candidate's own Cp_min. The t/c variable is gone — the "
                "weights ARE the thickness — and its margin is a constraint. "
                "SLOW: seconds per new section.")
            return name + " + CST section (XFOIL)", notes
        return name, notes
    if ch.get("system", "single") == "tandem":
        # the pair has a NONPLANAR solver now (tandemvlm.py): tip devices on
        # either wing, and a designed section, both of which the published
        # lifting-line pair cannot represent. Asking for either selects it;
        # asking for neither keeps the published problem and its numbers.
        wl = ch.get("winglets") in ("free", "capped")
        section = ch.get("airfoil") in _TAIL_SECTION_AIRFOILS
        # ...AND THE PAIR'S OWN CANT. A searched dihedral is out-of-plane
        # geometry, which is the third thing the published lifting-line pair
        # cannot represent — so asking for it selects the nonplanar family
        # exactly as a tip device or a designed section does.
        from aerobo import api as _api
        cant = ch.get("wing_cant", "fixed")
        if cant not in _api.WING_CANTS:
            cant = "fixed"
        free_cant = cant != "fixed"
        # the PLANAR pair can now search its section thickness — one row per
        # wing, off the NACA 24XX family. It is a different question from the
        # nonplanar family's DESIGNED section (which shapes the aerofoil), so
        # it is answered on the published lifting-line solver and only when no
        # tip device or designed section has already selected the other core.
        tc = ch.get("airfoil") == "tc_sweep" and not (wl or free_cant)
        for k in specials:
            if k == "winglets" and wl:
                continue
            if k == "airfoil" and (section or tc):
                continue
            notes.append(
                f"{_SPECIAL_LABEL[k]} ignored — no combined solver for "
                "this configuration (deliberate extension point).")
        if free_cant and ch.get("airfoil") == "tc_sweep":
            notes.append(
                "The pair's thickness sweep is off: t/c is searched on the "
                "published LIFTING-LINE pair, and a searched cant needs the "
                "panel solver. Ask for one or the other — the section "
                "thickness there, or a searched cant here.")
        if tc:
            # no note: t/c is the SAME freedom here as on every other
            # "+ t/c" family, and none of those explains itself either. What
            # is particular to this one — two rows rather than one, and no
            # chosen section — is in the ProblemSpec description, where the
            # menu already reads it from.
            return "tandem + t/c", notes
        if not (wl or section or free_cant):
            return "tandem", notes
        name = _api.tandem_vlm_problem(wl, section, cant)
        notes.append(
            "Front wing, rear wing and either tip device in ONE nonplanar "
            "solve (tandemvlm.py) — a different aero core AND a different "
            "design space from the published lifting-line pair (root/tip "
            "twist rather than 3 knots per wing), so compare within one of "
            "them."
            + (" The aerofoil is designed here too: 8 CST weights with a "
               "live XFOIL sweep each evaluation." if section else "")
            + (" The pair's dihedral and quarter-chord sweep are DESIGN "
               "ROWS here rather than numbers you state, so the field that "
               "stated them is gone and the design box carries them "
               "instead. Weight `spiral` on the objective card or neither "
               "row has anything to buy: on L/D alone the pair's cant "
               "settles at whichever bound costs least."
               if free_cant else ""))
        return name, notes
    if ch.get("tail"):
        # FIRST among the air families now: the tail composes with the tip
        # device, the section thickness and the chord law, so it can no
        # longer be the branch that everything else falls through to.
        return _derive_tail(ch, notes)
    if ch.get("winglets") in ("free", "capped") \
            and ch.get("airfoil") in ("tc_sweep", "coupled", "section_only",
                                      "section_wing"):
        # the combined winglet + airfoil solvers: t/c family lookup, the
        # pre-optimised (t/c × cl) library, or the 13-D free-form CST section
        # with live XFOIL
        if ch["airfoil"] == "tc_sweep":
            name = ("winglet + t/c" if ch["winglets"] == "free"
                    else "winglet_capped + t/c")
        elif ch["airfoil"] == "coupled":
            name = ("winglet + airfoil (coupled)" if ch["winglets"] == "free"
                    else "winglet_capped + airfoil (coupled)")
        else:
            name = ("winglet + airfoil (XFOIL)"
                    if ch["winglets"] == "free"
                    else "winglet_capped + airfoil (XFOIL)")
        for k in specials:
            if k not in ("winglets", "airfoil"):
                notes.append(
                    f"{_SPECIAL_LABEL[k]} ignored — no combined solver for "
                    "this configuration (deliberate extension point).")
        return name, notes
    if ch.get("winglets") == "capped" \
            and ch.get("winglet_type") == "blended":
        ignored("winglets")
        return "winglet, blended (span-capped)", notes
    if ch.get("winglets") == "capped" \
            and ch.get("winglet_type") == "blended_wing":
        ignored("winglets")
        return "winglet, blended into the wing (span-capped)", notes
    if ch.get("airfoil") == "section_wing":
        # CST section + wing planform, no tip device: the same live-XFOIL
        # co-design as the winglet variants with the two tip-device
        # variables removed
        ignored("airfoil")
        return "wing + airfoil (XFOIL)", notes
    if ch.get("airfoil") == "section_only":
        ignored("airfoil")
        return "airfoil (section)", notes
    if ch.get("planform") == "aircraft":
        # the published, calibrated sizing problem (its own family: it frees
        # t/c as well, and its numbers are the frozen ones)
        ignored("planform")
        return "free planform (aircraft)", notes
    if ch.get("winglets") == "free":
        ignored("winglets")
        return "winglet", notes
    if ch.get("winglets") == "capped":
        ignored("winglets")
        return "winglet_capped", notes
    if ch.get("airfoil") == "tc_sweep":
        ignored("airfoil")
        return "wing t/c + sweep", notes
    if ch.get("airfoil") == "coupled":
        ignored("airfoil")
        return "wing+airfoil (coupled)", notes
    # NOTE no "flight" branch: a free flight state is a MODIFIER now, applied
    # by derive_problem after this stage. On the plain wing it resolves to
    # "mission wing" — the same problem it always did, reached through
    # api.add_modifier instead of a family branch.
    return "trim wing", notes


def wing_tail_choices(name: str) -> dict | None:
    """Builder choices behind a generated wing+tail problem, or None.

    Generated from api.WING_TAIL_VARIANTS rather than listed: 30 hand-kept
    inverse entries would be 30 chances for the mapping to drift.
    """
    from aerobo import api

    v = api.WING_TAIL_VARIANTS.get(name)
    if v is None:
        return None
    wl = v["winglets"]
    return {
        "tail": True,
        "winglets": {None: "none", "free": "free", "capped": "capped",
                     "blended": "capped"}[wl],
        "winglet_type": "blended" if wl == "blended" else "canted",
        "airfoil": "tc_sweep" if v["tc"] else "fixed",
        "tail_arm": v["arm"],
        "tail_height": v["height"],
        "tail_design": v.get("design", "fixed"),
        "wing_cant": v.get("cant", "fixed"),
    }


def wing_tail_section_choices(name: str) -> dict | None:
    """Builder choices behind a generated wing+tail+CST-section problem."""
    from aerobo import api

    v = api.WING_TAIL_SECTION_VARIANTS.get(name)
    if v is None:
        return None
    wl = v["winglets"]
    return {
        "tail": True,
        "winglets": {None: "none", "free": "free", "capped": "capped",
                     "blended": "capped"}[wl],
        "winglet_type": "blended" if wl == "blended" else "canted",
        "airfoil": "section_wing",
        "tail_arm": v["arm"], "tail_height": v["height"],
        "tail_design": v.get("design", "fixed"),
        # ...AND THE WING'S CANT, which this inverse did not carry while its
        # twin ``wing_tail_choices`` did. All 96 section variants state one
        # and 48 of them are ``free``, so on every one of those the shell
        # read the cant back as FIXED: the round trip
        # choices -> family -> choices did not close, which greys out the
        # very option the user just took and reopens a saved preset as the
        # other family. It also mis-derived 192 of the 2142 chord twins,
        # which is how it was found.
        "wing_cant": v.get("cant", "fixed"),
    }


def hydro_section_choices(name: str) -> dict | None:
    """Builder choices behind a generated water + CST-section problem."""
    from aerobo import api

    v = api.HYDRO_SECTION_VARIANTS.get(name)
    if v is None:
        return None
    base = name[: -len(" + CST section (XFOIL)")]
    ch = {"medium": "water", "airfoil": "section_wing"}
    if v["kind"] == "winglet":
        ch["winglets"] = "free"
    elif v["kind"] == "tail":
        ch.update(hydro_tail_choices(base) or {})
    return ch


def hydro_tail_choices(name: str) -> dict | None:
    """Builder choices behind a generated hydrofoil+elevator problem."""
    from aerobo import api

    v = api.HYDRO_TAIL_VARIANTS.get(name)
    if v is None:
        return None
    return {
        "medium": "water", "tail": True,
        "winglets": "free" if v["winglets"] else "none",
        "tail_arm": v["arm"], "tail_height": v["height"],
        "tail_design": v.get("design", "fixed"),
    }


def tandem_vlm_choices(name: str) -> dict | None:
    """Builder choices behind a generated nonplanar-pair problem, or None.

    Generated from api.TANDEM_VLM_VARIANTS for the reason
    :func:`wing_tail_choices` is: the family is a product over tip devices,
    designed section and the pair's cant, and an inverse table kept by hand
    is a table that drifts from the one that generates the names.
    """
    from aerobo import api

    v = api.TANDEM_VLM_VARIANTS.get(name)
    if v is None:
        return None
    return {
        "system": "tandem",
        "winglets": "free" if v["winglets"] else "none",
        "winglet_type": "canted",
        "airfoil": "section_wing" if v["section"] else "fixed",
        "wing_cant": v.get("cant", "fixed"),
    }


def choices_from_problem(name: str) -> dict:
    """Inverse mapping for presets / the advanced problem picker.

    Reads the family and the modifier set straight off the registry
    (api.base_of / api.modifiers_of), so a problem carrying BOTH modifiers
    inverts without this module keeping a list of the combinations.
    """
    from aerobo import api

    ch = dict(BUILDER_DEFAULTS)
    base = api.base_of(name)
    # a chosen-section twin inverts to its own family's choices plus the
    # switch that selected it
    twin_of = {v: k for k, v in api.CHOSEN_SECTION_TWINS.items()}
    fly = base in twin_of
    if fly:
        base = twin_of[base]
    ch.update(wing_tail_choices(base) or wing_tail_section_choices(base)
              or tandem_vlm_choices(base)
              or hydro_section_choices(base) or hydro_tail_choices(base)
              or PROBLEM_TO_CHOICES.get(base, {}))
    if fly:
        ch["fly_section"] = True
    for mod in api.modifiers_of(name):
        key, on_value = MODIFIER_ON[mod]
        ch[key] = on_value
    return ch


# ------------------------------------------------------------- parameter help
#
# The design box lists design-vector labels, and some of them (the CST section
# weights above all) say nothing to anyone who has not read the source. The
# RAW label stays on screen — it is the key api.run, RunConfig.
# bounds_overrides and every saved run use, so hiding it would break the link
# between the GUI and the artefacts — and a readable name plus a one-line
# explanation is added next to it.
#
# Regular families (CST weights, chord/twist-law coefficients) are generated
# by param_help() rather than listed entry by entry.
#
# Not to be confused with PARAM_HELP further down, which explains the
# OPTIMISER's own inputs (mission fields, wing guess) rather than a problem's
# design vector.

DESIGN_PARAM_HELP: dict[str, tuple[str, str]] = {
    "taper_t": (
        "tail taper",
        "Tip chord / root chord of the TAIL, at its own area: the same "
        "freedom the wing has, on the second surface. 1.0 is the published "
        "rectangular stabiliser."),
    "AR_t": (
        "tail aspect ratio",
        "b_t²/S_t — how much span the tail's area is spread over "
        "(b_t = sqrt(AR_t S_t)). The published surface is fixed at 4, the "
        "conventional H-tail range is 3-5: slender is more efficient per "
        "unit area and structurally harder, stubby the reverse."),
    "washout_t_deg": (
        "tail washout",
        "Tail TIP twist relative to its root, + nose-up. Only the "
        "difference is a design variable: the uniform part of a tail's "
        "incidence IS the trim unknown i_t, so a root twist would be that "
        "same angle a second time."),
    "winglet_h_frac_t": (
        "tail tip-device height",
        "Height of the TAIL's own tip device as a fraction of its "
        "semispan (the wing's winglet_h_frac, on the second surface). It "
        "is panelised by the same nonplanar VLM code as the wing's."),
    "winglet_cant_t_deg": (
        "tail tip-device cant",
        "Cant of the TAIL's tip device from the horizontal: 90° is a "
        "vertical fence, small angles a raked extension."),
    "twist_root_front_deg": (
        "front wing root twist",
        "Incidence of the FRONT wing's root section, + nose-up "
        "(tandemvlm.py's root/tip law — the planar pair uses 3 knots per "
        "wing instead)."),
    "twist_tip_front_deg": (
        "front wing tip twist",
        "Incidence of the FRONT wing's tip section; negative is washout, "
        "which unloads the tip and moves the stall inboard."),
    "twist_root_rear_deg": (
        "rear wing root twist",
        "Incidence of the REAR wing's root section, BEFORE the decalage "
        "offset is added."),
    "twist_tip_rear_deg": (
        "rear wing tip twist",
        "Incidence of the REAR wing's tip section, before the decalage."),
    "winglet_h_front": (
        "front tip device height",
        "Arc length of the FRONT wing's tip device as a fraction of its "
        "semi-span."),
    "winglet_cant_front_deg": (
        "front tip device cant",
        "Angle of the FRONT wing's tip device from the wing plane; 90° is a "
        "vertical fence."),
    "winglet_h_rear": (
        "rear tip device height",
        "Arc length of the REAR wing's tip device as a fraction of its "
        "semi-span."),
    "winglet_cant_rear_deg": (
        "rear tip device cant",
        "Angle of the REAR wing's tip device from the wing plane."),
    "taper": (
        "taper λ = tip chord / root chord",
        "1.0 is rectangular, small is a pointy tip. Span and area are fixed, "
        "so this reshapes the planform rather than resizing it."),
    "twist_root_deg": (
        "twist at the root [°]",
        "Section incidence at the centreline in the linear washout law. The "
        "wing is then trimmed to its lift target on top of this."),
    "twist_tip_deg": (
        "twist at the tip [°]",
        "Section incidence at the tip. Negative (washout) unloads the tip: "
        "it costs a little span efficiency and buys stall margin there."),
    "winglet_h_frac": (
        "winglet height / semi-span",
        "ARC LENGTH of the tip device over b/2. 0 drops the winglet."),
    "wing_dihedral_deg": (
        "wing dihedral [°], tips up",
        "A RIGID rotation of the whole semi-span (tip device included) about "
        "the body x-axis, so span, area, chord and twist are untouched and "
        "only where the surface points changes. Negative is anhedral. It is "
        "the only WING-side source of Cl_beta that holds at any lift (a "
        "swept wing has one too, but it goes as CL): with it at zero the fin "
        "and the tip device carry the dihedral effect at cruise, and a "
        "bigger fin raises the yaw stiffness with it, so no fin SIZE "
        "converges the spiral. Measured on the `tail + winglet` box centre, "
        "on the SCORED lattice (the one the objective reads, tip device "
        "included): 3 deg takes the spiral margin −0.002197 → +0.005139 "
        "for 0.21 % of L/D. The flight rebuild agrees to 4 % on the "
        "divergent end and 0.1 % on the convergent one."),
    "wing_sweep_deg": (
        "quarter-chord sweep [°], aft positive",
        "An |y|·tan Λ offset of the bound vortex line. It moves the neutral "
        "point aft — which BUYS static margin — and costs lift-curve slope "
        "and L/D: measured, 15 deg costs 12.7 % of L/D and moves x_np 0.57 m "
        "aft. Not a roll lever: a swept wing's dihedral effect goes as CL and "
        "vanishes at cruise, so read it against the margin constraint, not "
        "against the spiral."),
    "winglet_cant_deg": (
        "winglet cant [°] from the wing plane",
        "90° is a vertical fence (no extra projected span); lower cant tilts "
        "the device outboard, towards a raked span extension. In water the "
        "sign matters: + points up towards the free surface, − points down "
        "into deeper water (more static head, later cavitation)."),
    "winglet_blend_frac": (
        "blended fraction of the winglet",
        "Share of the device's arc length spent turning out of the wing "
        "plane on a constant-radius arc. 0 is a sharp corner; the arc length "
        "is conserved, so this changes the SHAPE, not the size."),
    "wing_blend_frac": (
        "blended fraction of the semi-span (wing side)",
        "How far INBOARD of the tip the transition starts, as a fraction of "
        "the semi-span: the outer wing turns up into the device instead of "
        "meeting it at a fixed point. This is the knob that sets the blend "
        "RADIUS — confined to the winglet the turn has at most "
        "blend_frac × h of arc, so the radius stays a fraction of a tip "
        "chord however it is drawn. Developed span is conserved, so a "
        "wing-side blend trades projected span for height rather than "
        "deleting wing."),
    "V_ms": ("true speed [m/s]", "Free-stream speed at the design point."),
    "altitude_m": (
        "altitude [m]", "Sets air density and viscosity (ISA)."),
    "tc": (
        "section thickness / chord",
        "Selects the member of the NACA 24XX polar family the whole wing "
        "flies on."),
    # the tandem pair carries ONE PER WING, because that family already gives
    # each surface its own span, chord law and section — so a shared thickness
    # would be a constraint nobody asked for
    "tc_front": (
        "front wing section thickness / chord",
        "Selects the member of the NACA 24XX polar family the FRONT wing "
        "flies on. The rear wing has its own row."),
    "tc_rear": (
        "rear wing section thickness / chord",
        "Selects the member of the NACA 24XX polar family the REAR wing "
        "flies on. The front wing has its own row."),
    "tc_sec": (
        "section thickness / chord",
        "First summary variable of the pre-optimised section library."),
    "cl_sec": (
        "section design lift coefficient",
        "Second summary variable of the section library: the cl the section "
        "was optimised at."),
    "sweep_deg": (
        "quarter-chord sweep [°]",
        "Sweep of the quarter-chord line; enters the form factor and the "
        "compressibility correction."),
    "b_m": (
        "span [m]",
        "Wing span. A DESIGN VARIABLE where the size is free — the published "
        "aircraft-sizing problem, or any family carrying the free-planform "
        "modifier, where the wing WEIGHS what its size implies and a "
        "root-bending stress margin constrains it (sizing.py) — and on the "
        "CAR families, where the reference area is fixed so the span IS the "
        "aspect ratio: more span is less induced drag on a longer arm, and "
        "the deflection limit is what pushes back. On a TANDEM PAIR it is "
        "the FRONT wing's span, and the rear wing has one of its own "
        "(b_rear_m)."),
    "ws_pa": (
        "wing loading W/S [N/m²]",
        "The wing loading itself, as a DESIGN VARIABLE (sizing's "
        "searched-loading mode). The area is not in the vector at all: it "
        "follows this number through the weight loop, S = W_total/(W/S), so "
        "every candidate is a closed aircraft and the trim lift coefficient "
        "is CL = (W/S)/q by construction. It is the mission's own question — "
        "a constraint diagram answers it — reopened here between two "
        "loadings you are willing to fly, and the mission's ceiling (stall "
        "or landing field in air, fly-up or cavitation in water) clips the "
        "band. Divide by 9.81 for kg/m²."),
    "b_rear_m": (
        "rear wing span [m]",
        "The REAR wing of a tandem pair, which shares a fuselage with the "
        "front one and not a span (sizing.span_labels). It is the same "
        "question as b_m asked of the second wing, and it is the pair's "
        "central trade: the rear wing sits in the front wing's downwash, and "
        "how much of it depends on how far out its tip reaches — well "
        "inboard of the front tip vortices is a different flow field from "
        "matching them. It weighs and is stressed as the wing it is, and its "
        "tip device is sized on its own semi-span."),
    "S_m2": (
        "wing area [m²]",
        "Wing (or, for a tandem pair, TOTAL) reference area. Free on the same "
        "terms as the span, and the trim lift target follows the weight it "
        "implies."),
    "depth_m": (
        "submergence [m]",
        "Depth of the foil below the free surface: shallow costs induced "
        "drag through the image, deep buys cavitation margin through the "
        "static head and costs mast wetted area."),
    "alpha_deg": (
        "wing incidence [°]",
        "Angle of the wing to the oncoming flow (not trimmed to a target "
        "here — more angle means more downforce until the section stalls)."),
    "endplate_h_m": (
        "endplate height [m]",
        "Arc length of the endplate, pointing at the track — a LENGTH, not a "
        "fraction of the semi-span, because the span is a design variable "
        "here and what the plate has to do is reach a fixed piece of car. "
        "When the wing is endplate-mounted that reach is a signed "
        "constraint: the plate must span from the wing down to the "
        "attachment deck."),
    "endplate_chord_ratio": (
        "endplate chord / wing tip chord",
        "The plate's own streamwise chord. Real endplates are longer than "
        "the wing's chord; more chord is more fence (and more wetted area). "
        "This band is a RATIO, so it keeps the plate proportioned to a wing "
        "whose tip chord the search is moving — it cannot state a length. "
        "The plate's metre band lives on the wing card beside its section, "
        "and the chord flown is this row's answer clipped into it."),
    "endplate_tc": (
        "endplate thickness / chord",
        "Thickness ratio of the plate's section. Bending stiffness goes as "
        "t³, so this is the plate's structural variable — and its drag one."),
    "endplate_toe_deg": (
        "endplate toe [°]",
        "Incidence of the plate. Positive loads it inboard, i.e. turns the "
        "flow outboard (outwash), which unloads the tip vortex and raises "
        "downforce — paid for with a side load the plate has to carry. To a "
        "linear method a cambered plate at zero toe is the same surface as a "
        "symmetric plate at that toe, so this variable carries camber too."),
    # ...and the plate's two SEARCHED shape numbers, which exist only on the
    # families that free them (api.PLATE_FREEDOMS). On the fixed family the
    # same numbers are typed on the Mount card instead — one question, two
    # ways of answering it, never both at once.
    "endplate_cant_deg": (
        "endplate cant [°]",
        "How far the plate leans out of the wing plane; 90° is upright and "
        "is every published plate. Priced at both ends, which is what makes "
        "it worth searching: leaning it out projects the plate OUTBOARD, and "
        "the span row here is the OVERALL width, so the wing shortens to pay "
        "for it — while the plate's vertical reach falls as h·sin, and it "
        "still has to get down to the attachment deck. Against those it buys "
        "the nonplanar benefit the plate is there for. The band is the "
        "nonplanar lattice's own validity, not a recommendation: below about "
        "48° the top of the ride band stops being reachable at any plate "
        "height in the box, and the run says so as a refusal it can see the "
        "sign of."),
    "endplate_blend_frac": (
        "endplate root blend [0–1]",
        "How much of the plate's height is spent turning out of the wing "
        "plane, i.e. how soft the wing/plate CORNER is. It buys the corner's "
        "interference charge and costs a plate that ends SHORTER (the reach "
        "margin tightens) and reaches OUTBOARD (the span row pays, because "
        "b_m is the overall width). 0 is a crease and is every published "
        "run. WHICH LAW the turn follows is still a statement — blend_shape "
        "on the Mount card — because a law is not a length."),
    "ride_height_m": (
        "ride height [m]",
        "Height of the wing above the track. Closer to the ground means more "
        "downforce and less induced drag — until the wall is too close for "
        "the image model."),
    # --- the SLOTTED two-element rear wing (carwing_multi.py). Four rows, and
    # every one of them is a placement in the SECTION rather than anything the
    # lattice can see: a slot is a chordwise interaction at one spanwise
    # station, so the pair is solved in 2-D and enters the VLM as a polar.
    "flap_chord_frac": (
        "flap chord / stowed chord",
        "How much of the section is the flap. The two element chords SUM to "
        "the reference chord, so this splits a fixed chord rather than adding "
        "to it, and every coefficient stays on that same reference. The band "
        "is the family's own statement of what is still a flap: at the top of "
        "it the main element still carries about three quarters of the lift."),
    "flap_deflection_deg": (
        "flap deflection [°]",
        "Trailing edge towards the track — more deflection is more downforce. "
        "Zero is a real design (an undeflected slotted section) and is also "
        "the nearest thing this vector has to 'no flap'. What stops it is not "
        "the band but the section: each element is refused once its inviscid "
        "suction peak goes deeper than the peak that section is MEASURED to "
        "hold alone."),
    "slot_gap_frac": (
        "slot gap / stowed chord",
        "How far BELOW the main element's trailing edge the flap's leading "
        "edge is placed. Not the slot WIDTH: the flap then rotates about that "
        "placed leading edge, so the true surface-to-surface distance is "
        "measured from the built geometry (breakdown: slot width) and is "
        "smaller than this — and it OPENS as the flap deflects, which is the "
        "opposite of what 'more flap, tighter slot' suggests."),
    "slot_overlap_frac": (
        "slot overlap / stowed chord",
        "How far UPSTREAM of the main element's trailing edge the flap's "
        "leading edge is placed (the sense every wind-tunnel report uses). "
        "Negative is a flap set back — a gap in x, which is the Fowler "
        "direction and a real design."),
    "S_t_m2": ("tail area [m²]", "Area of the horizontal tail panel."),
    "l_t_m": (
        "horizontal separation (tail arm) [m]",
        "Distance from the wing's aerodynamic centre to the tail (negative "
        "station for a canard, handled by the solver). This row exists only "
        "while the separation is SEARCHED: state it on the tail card instead "
        "and the arm leaves the design vector, row and all."),
    "z_t_m": (
        "tail / stabiliser height [m]",
        "Vertical distance of the second surface from the wing plane — "
        "positive UP for an aircraft tail (out of the wake), negative for a "
        "hydrofoil's stabiliser (deeper, where there is more static head). "
        "For the aircraft tail: out of the wing's "
        "trailing sheet it sees less downwash, so it works harder and the "
        "neutral point moves aft — an inviscid effect only: this wake is "
        "rigid, so no dynamic-pressure deficit and no deep-stall pitch-up. "
        "In AIR this row opens on 0.05 b … 0.30 b, the band the family's "
        "results were MEASURED over. In WATER it opens at 0.01 b, because a "
        "foiling craft is FLAT — front wing, strut and stabiliser bolted to "
        "one horizontal fuselage — and 0.05 b is 60 mm under the shipped "
        "foil where builders work in millimetres. Either way a default, not "
        "a limit: type any band and it is the band searched, down to a "
        "surface in the wing plane (tail.height_row)."),
    "x_mast_frac": (
        "strut station (fraction of the fuselage)",
        "WHERE THE MAST STANDS: 0 is the front wing's quarter chord and 1 is "
        "the stabiliser's, so the strut is bolted to the fuselage between "
        "them the way a real one is. Asked as a fraction because the arm is "
        "itself a design variable — a station in metres would mean a "
        "different layout at each end of its band. It is a STABILITY row and "
        "not a drag one: the strut's wetted area, its Reynolds number and "
        "the side force it carries do not read the station, and what does is "
        "the yaw arm about the CG — forward of the CG the strut's own yaw "
        "moment is destabilising (which is what a real windfoil is), aft of "
        "it the craft weathercocks."),
    "taper_front": ("front wing taper λ", "Tip / root chord, front wing."),
    "taper_rear": ("rear wing taper λ", "Tip / root chord, rear wing."),
    "area_split_front": (
        "front wing's share of the area",
        "0.5 splits the total area equally between the two wings."),
    "decalage_deg": (
        "decalage [°]",
        "Front-wing incidence minus rear-wing incidence — the tandem "
        "system's longitudinal trim variable."),
}

#: chordwise station each CST weight controls at n_cst = 4: the cubic
#: Bernstein basis functions peak at x/c = i/3.
_CST_STATIONS = ("nose", "forward", "aft", "trailing edge")


def _cst_help(surface: str, i: int, n: int = 4) -> tuple[str, str]:
    """(name, explanation) for CST weight ``i`` of ``n`` on one surface."""
    where = (_CST_STATIONS[i] if n == len(_CST_STATIONS)
             else f"x/c ≈ {i / max(n - 1, 1):.2f}")
    sign = "positive" if surface == "upper" else "negative"
    return (
        f"{surface} surface, {where}",
        f"CST (Kulfan) shape weight {i + 1} of {n} on the {surface} surface: "
        f"it lifts that surface near the {where} of the section. {surface.capitalize()} "
        f"weights are {sign}. Thickness, camber and Cm are OUTPUTS of the "
        f"eight weights, never inputs — the box is centred on the anchor "
        f"aerofoil's own CST fit.")


def param_help(label: str) -> tuple[str, str]:
    """(readable name, hover explanation) for a design-vector label.

    Falls back to the raw label with no explanation, so an unknown parameter
    still renders — a missing entry must never blank out a design box.
    """
    lbl = str(label)
    if lbl in DESIGN_PARAM_HELP:
        return DESIGN_PARAM_HELP[lbl]
    for surface in ("upper", "lower"):
        if lbl.startswith(f"w_{surface}_"):
            try:
                return _cst_help(surface, int(lbl.rsplit("_", 1)[1]))
            except ValueError:
                break
    if lbl.startswith("chord_"):
        rest = lbl[len("chord_"):]
        wing = ""
        for w in ("front", "rear"):
            if rest.startswith(f"{w}_"):
                wing, rest = f"{w} wing ", rest[len(w) + 1:]
                break
        if rest.endswith("_t"):          # the TAIL's own law (designed tail)
            wing, rest = "tail ", rest[:-2]
        j = rest.lstrip("k")
        if not j.isdigit():
            return lbl, ""
        return (
            f"{wing}chord law: η^{j} coefficient".capitalize()
            if wing else f"chord law: η^{j} coefficient",
            f"Coefficient of η^{j} in the {wing}chord multiplier "
            f"1 + Σ kⱼ ηʲ "
            "(η = |2y/b|, 0 at the root, 1 at the tip). The area is held "
            "exactly by a closed-form rescale, so these reshape the chord "
            "DISTRIBUTION without resizing the wing. All zero = straight "
            "taper; a law that collapses the chord anywhere is refused.")
    if lbl.startswith("twist_") and lbl.endswith("_deg"):
        core = lbl[len("twist_"):-len("_deg")]
        if core.replace("k", "").isdigit():
            return (f"twist law coefficient {core}",
                    "Coefficient of the polynomial twist law (the root is "
                    "held at 0°, so these bend the washout distribution).")
        if core.startswith(("front_", "rear_")):
            wing, knot = core.split("_", 1)
            return (f"{wing} wing twist, knot {knot.lstrip('k')}",
                    f"Knot {knot.lstrip('k')} of the {wing} wing's 3-knot "
                    "spline twist law (root fixed at 0°).")
    return lbl, ""


def bounds_legend(labels) -> str:
    """One-line explanation of an unusual design box, or '' for an ordinary
    one. Only the CST section box needs it — the rest are named quantities."""
    labels = [str(x) for x in labels]
    if any(x.startswith(("w_upper_", "w_lower_")) for x in labels):
        return ("The section shape is carried by CST weights — four per "
                "surface, nose to trailing edge — not by t/c and camber "
                "directly. Widening a weight's bounds widens the family of "
                "sections the optimiser may draw around the anchor aerofoil.")
    return ""


def twist_description(problem_name: str) -> str:
    """What the twist law IS on this problem — asked of the FAMILY.

    A modifier twin flies its base's twist law: matching the name exactly
    told a tandem user under a chord law that their twist was the single
    wing's "linear washout, root → tip".
    """
    from aerobo import api

    family = api.base_of(problem_name)
    if family == "tandem":
        return "3-knot spline twist law per wing (roots fixed at 0°)"
    if family == "airfoil (section)":
        return "no wing — 2-D section only"
    return "linear washout, root → tip (2 variables)"


#: problems whose planform rows are drawn by a branch of their own below
#: (the size is in the design vector, or there are two wings to describe)
_OWN_PLANFORM_ROWS = ("free planform (aircraft)", "tandem")

#: a chord-law coefficient row. The law is per SURFACE, so the name carries
#: which one: bare (the wing), ``front``/``rear`` (a tandem pair), or the
#: ``_t`` suffix (a designed tail/elevator). 448 of the 722 registered
#: chord-law problems carry two blocks, which is why nothing here may assume
#: a single one of three coefficients.
_CHORD_COEFF_ROW = re.compile(r"chord_(?:front_|rear_)?k(\d+)(_t)?$")


def chord_coeff_labels(problem_name: str) -> list[str]:
    """The chord-law rows a problem's design vector actually carries."""
    from aerobo import api

    spec = api.PROBLEM_SPECS.get(problem_name)
    return [lbl for lbl in (spec.param_labels if spec else ())
            if _CHORD_COEFF_ROW.fullmatch(str(lbl))]


def chord_law_note(problem_name: str) -> str:
    """What the chord law ADDS to this problem, counted rather than assumed.

    "Adds 3 coefficients" was written when the law rode on one wing. It is
    per SURFACE now: a tandem pair carries one block per wing and a designed
    tail one on each of wing and tail, so 448 of the 722 chord-law problems
    add six. Returns "" where the problem has no law.
    """
    labels = chord_coeff_labels(problem_name)
    if not labels:
        return ""
    surfaces = {(("front" if "_front_" in lbl else
                  "rear" if "_rear_" in lbl else
                  "tail" if lbl.endswith("_t") else "wing"))
                for lbl in labels}
    # named front-to-back, the order the surfaces sit in and the vector lists
    # them — sorting alphabetically read as "the tail and the wing"
    order = ("wing", "front", "rear", "tail")
    where = ("" if surfaces == {"wing"}
             else " — one law per wing" if surfaces == {"front", "rear"}
             else " — one law each on the "
                  + " and the ".join(s for s in order if s in surfaces))
    return (f"Adds {len(labels)} coefficients{where}, reshaping the taper "
            f"baseline outboard at constant area, so the loading is no longer "
            f"whatever one taper ratio can draw. The gated number "
            f"(e ≈ 0.999 against 0.984 for the best trapezoid) is the AIR "
            f"trim wing's, measured from a rectangular baseline; what it is "
            f"worth on this family is this family's own run. It is "
            f"{len(labels)} dimensions the search has to pay for: the frozen "
            f"study (results/chord_law_bo.json, budget 60 × 5 seeds) puts "
            f"the trim wing's gain at +0.11% in the DEFAULT taper box and "
            f"+1.37% in a near-rectangular one, so at a small budget the law "
            f"can finish below straight taper. Widen the budget, or narrow "
            f"taper to where the trapezoid cannot reach.")


def _chord_law_row(bounds: dict, flags: dict | None = None,
                   reshapes_rows_above: bool = True) -> list[tuple[str, str]]:
    """One row naming the chord law the chord rows above do NOT include.

    Root and tip chord are trapezoid arithmetic — c_root = 2S/(b(1+λ)) — and
    that is the whole planform only while the chord distribution IS the
    taper. A chord law multiplies that baseline by 1 + Σ k_j η^j and
    renormalises to hold the area, so the FLOWN root and tip are not those
    numbers. Now that every shell opens on the law (:data:`BUILDER_START`),
    a panel that went on quoting the trapezoid alone would misdescribe the
    default design box, which is why the two rows above say "straight taper"
    whenever this row is present.

    The row leads with what the box DRAWS rather than with the coefficient
    bound: ``|k| ≤ 0.5`` is the solver's variable and nobody's design intent,
    and the planform it buys — a chord free to reach nearly twice the
    trapezoid's, or a tenth of it, at the tip — is not readable from it
    (:func:`aerobo.geometry.chord_reach`).
    """
    from aerobo import api, geometry

    ks = [(int(m.group(1)), key, v) for m, key, v in
          ((_CHORD_COEFF_ROW.fullmatch(k), k, v) for k, v in bounds.items())
          if m and v]
    if not ks:
        return []
    order = max(j for j, _, _ in ks)
    kmax = max(max(abs(float(v[0])), abs(float(v[1]))) for _, _, v in ks)
    # the law is per SURFACE: a pair carries one per wing, a designed tail one
    # on each of wing and tail. Saying "the two chords above" on either would
    # point at rows that describe only one of them (or that the branch never
    # drew at all).
    per_wing = any("_front_" in k or "_rear_" in k for _, k, _ in ks)
    on_tail = any(k.endswith("_t") for _, k, _ in ks)
    what = ("one law per wing" if per_wing
            else "one law each on the wing and the tail" if on_tail
            else "reshapes the two chords above" if reshapes_rows_above
            else "polynomial in eta = |2y/b|")
    # the widest box any one surface carries, read on the taper the run
    # searches (its mid-point): the same coefficients bend a rectangular wing
    # and a sharply tapered one by different amounts
    widest: dict = {}
    for j, _, v in ks:
        widest[j] = max(widest.get(j, 0.0),
                        abs(float(v[0])), abs(float(v[1])))
    box = [[-widest.get(j, kmax), widest.get(j, kmax)]
           for j in range(1, order + 1)]
    taper = next((0.5 * (float(v[0]) + float(v[1]))
                  for k, v in bounds.items()
                  if k.startswith("taper") and v), 1.0)
    # ...and under the LAW the run carries: the same coefficient box bends a
    # cubic and an interior-only law by quite different amounts, and the
    # elliptic blend's box is not even symmetric (api.CHORD_LAW_KEY)
    law = str((flags or {}).get(api.CHORD_LAW_KEY)
              or geometry.DEFAULT_CHORD_LAW)
    dev = geometry.chord_reach(box, taper, law=law).dev_max
    return [("chord law",
             f"{geometry.CHORD_LAW_NAMES[law]}, order {order}, bends the "
             f"chord up to ±{100 * dev:.0f}% away from straight taper "
             f"(|k| ≤ {kmax:g}) — {what}, area held")]


def _searched_planform_rows(problem_name: str, bounds: dict,
                            flags: dict | None, span_bands: list,
                            area_m2: float | None = None
                            ) -> list[tuple[str, str]]:
    """The planform rows where the SIZE is searched — bands, not a wing.

    Split out of :func:`geometry_summary` because it is a different question
    with the same name: with b_m (and possibly S_m2) in the box there is no
    single planform to report, and the honest summary of a box is its
    corners. The area is a band where the box searches one and the family's
    own number where it does not (the wing-loading modes: S = W/(W/S), which
    is not a design row), and the aspect ratio is read across the corners the
    same way :func:`gui.v3.session.clip_size_box` reads them — a pair on half
    the area, because that is the share each wing is checked on.
    """
    from aerobo import api

    pair = len(span_bands) > 1
    names = {"b_m": "front span" if pair else "span b",
             "b_rear_m": "rear span"}
    rows: list[tuple[str, str]] = []
    for row, band in span_bands:
        lo, hi = float(band[0]), float(band[1])
        rows.append((names.get(row, row), f"{lo:.4g} – {hi:.4g} m · searched"))
    area_band = bounds.get("S_m2")
    if area_band:
        s_lo, s_hi = float(area_band[0]), float(area_band[1])
        rows.append(("area S", f"{s_lo:.4g} – {s_hi:.4g} m² · searched"))
    elif bounds.get("ws_pa"):
        # THE LOADING IS A ROW, so the area sweeps with it and there is no
        # number to quote: S = W_total/(W/S), and W_total is the fixed weight
        # plus the wing's own structure, which sizing closes as a fixed point
        # (the asymmetry gui.v3.session.clip_size_box is written around). The
        # honest row says what sets it and stops.
        s_lo = s_hi = 0.0
        rows.append(("area S", "follows the searched W/S row — "
                               "S = W_total/(W/S), and W_total grows with the "
                               "wing, so it is not b²/AR either"))
        rows.append(("to constrain AR", "narrow b and the W/S row together"))
    else:
        # NOT ``api.planform_size`` here. With the span searched against a
        # STATED loading the area is S = W/(W/S), which is the caller's
        # mission and not the family's published planform — and quoting the
        # published one put "10 m²" under every mission that states another.
        # A caller that does not know it gets no area row rather than a
        # number this panel made up.
        s_lo = s_hi = float(area_m2 or 0.0)
        if not s_hi > 0.0:
            return rows + _chord_law_row(bounds, flags,
                                         reshapes_rows_above=False)
        rows.append(("area S", f"{s_hi:.4g} m² · the mission's, "
                               f"not a design row"))
    share = 0.5 if pair else 1.0
    for row, band in span_bands:
        lo, hi = float(band[0]), float(band[1])
        if not (s_lo > 0.0 and s_hi > 0.0):
            break
        rows.append(("aspect ratio" if not pair else f"{names[row]} AR",
                     f"{lo * lo / (share * s_hi):.3g} – "
                     f"{hi * hi / (share * s_lo):.3g}"))
    if area_band:
        rows.append(("to constrain AR", "narrow b and S — AR = b²/S"))
    split = bounds.get("area_split_front")
    if split:
        k_lo, k_hi = float(split[0]), float(split[1])
        rows.append(("front-wing area",
                     f"{k_lo * s_lo:.3g} – {k_hi * s_hi:.3g} m²" if area_band
                     else f"{k_lo:.3g} – {k_hi:.3g} of the area"))
    return rows + _chord_law_row(bounds, flags, reshapes_rows_above=False)


def geometry_summary(problem_name: str, bounds: dict,
                     flags: dict | None = None,
                     area_m2: float | None = None
                     ) -> list[tuple[str, str]]:
    """Derived planform quantities (AR, chords) from the current design box —
    presentation-only arithmetic so 'constrain AR / chord / span' is visible.

    ``flags`` is the run's flag dict: with a CHOSEN size (api.PLANFORM_KEYS)
    the rows report the wing that will actually be flown, not the published
    default. The word "fixed" here means "not a design variable", which is
    still true of a size the user typed in.
    """
    from aerobo import api

    rows: list[tuple[str, str]] = []
    size = None
    # A ROW OF THE BOX IS A DESIGN VARIABLE, WHATEVER FAMILY CARRIES IT. This
    # branch is taken first for exactly that reason: the size modifier's
    # twins, the two wing-loading modes and the car all SEARCH a span (and
    # some of them an area), and this panel read the branch off the family
    # name instead — so a box searching b_m 9.9 – 11.9 m and S_m2 18 – 21.6 m
    # was summarised, one card below the table showing those bands, as
    # "span b 10 m · fixed, area S 10 m² · fixed, aspect ratio 10 · fixed".
    # That is this repo's own "the box shown is the box searched" failure,
    # and the number it invented was the mission's stated wing, which the
    # run does not fly in any of those modes.
    span_bands = [(r, bounds[r]) for r in ("b_m", "b_rear_m") if bounds.get(r)]
    if span_bands:
        return _searched_planform_rows(problem_name, bounds, flags, span_bands,
                                       area_m2)
    # the FAMILY decides which branch draws the rows, not the problem name:
    # a modifier twin is the same aeroplane. Matching the name exactly sent
    # "tandem + free chord law" down the single-wing branch, where it
    # reported the PAIR's 20 m² over one 10 m span as "aspect ratio 5" and
    # dropped the front-wing-area row, and left the aircraft twin with an
    # empty panel. Both were unreachable while the chord law was opt-in;
    # BUILDER_START made them the default view.
    family = api.base_of(problem_name)
    if family not in _OWN_PLANFORM_ROWS:
        # read off the built problem, so a chosen size and a solver's own
        # published size are reported by the SAME path
        size = api.planform_size(problem_name, flags)
    if size is not None:
        b, S = size
        chosen = " · chosen" if (flags or {}).get("b_m") is not None \
            or (flags or {}).get("S_m2") is not None else " · fixed"
        rows += [("span b", f"{b:g} m{chosen}"),
                 ("area S", f"{S:g} m²{chosen}"),
                 ("aspect ratio", f"{b * b / S:g}{chosen}")]
        tb = bounds.get("taper")
        if tb:
            lo, hi = float(tb[0]), float(tb[1])
            cr = lambda lam: 2.0 * S / (b * (1.0 + lam))   # noqa: E731
            law = _chord_law_row(bounds, flags)
            # the INTERIOR-ONLY law holds both end chords exactly, so under it
            # these two rows are not a baseline that something else reshapes —
            # they are the chords that fly, and saying "straight taper" would
            # under-claim the one law that makes them true
            held = str((flags or {}).get("chord_law") or "") == "ends"
            base = (" · held by the chord law" if held
                    else " · straight taper" if law else "")
            rows.append(("root chord", f"{cr(hi):.3g} – {cr(lo):.3g} m{base}"))
            rows.append(("tip chord",
                         f"{lo * cr(lo):.3g} – {hi * cr(hi):.3g} m{base}"))
            rows += law
    elif family == "free planform (aircraft)":
        bb, sb = bounds.get("b_m"), bounds.get("S_m2")
        if bb and sb:
            rows += [("span b", f"{bb[0]:g} – {bb[1]:g} m"),
                     ("area S", f"{sb[0]:g} – {sb[1]:g} m²"),
                     ("aspect ratio",
                      f"{bb[0] ** 2 / sb[1]:.3g} – {bb[1] ** 2 / sb[0]:.3g}"),
                     ("to constrain AR", "narrow b and S — AR = b²/S")]
            rows += _chord_law_row(bounds, flags, reshapes_rows_above=False)
    elif family == "tandem":
        rows += [("per-wing span", "10 m · fixed"),
                 ("total area", "20 m² · fixed")]
        sp = bounds.get("area_split_front")
        if sp:
            rows.append(("front-wing area",
                         f"{20 * sp[0]:.3g} – {20 * sp[1]:.3g} m²"))
        # the pair carries one law PER WING, so the row reads off both
        rows += _chord_law_row(bounds, flags, reshapes_rows_above=False)
    return rows


# ------------------------------------------------------- propeller builder

PROP_DEFAULTS: dict = {
    "enabled": False,
    "D_p": 2.0,             # propeller diameter [m]
    "layout": "pair",       # single (centreline) | pair (symmetric ±y_p)
    "y_p": 2.5,             # pair spanwise centre [m]
    "thrust_mode": "CT",    # CT | mu
    "CT": 0.6,              # disk thrust coefficient
    "mu_inf": 1.15,         # developed velocity ratio V_j/V_inf
    "swirl_deg": 0.0,       # Bell reference swirl angle [deg]
    "rotation": "counter",  # co_cw | co_ccw | counter (pair only)
}


def slipstream_dict(p: dict) -> dict | None:
    """Propeller UI state -> JSON-safe SlipstreamSpec parameter dict."""
    if not p.get("enabled"):
        return None
    single = p.get("layout") == "single"
    y = [0.0] if single else [-float(p["y_p"]), float(p["y_p"])]
    rot = p.get("rotation", "counter")
    if single:
        spin = -1.0 if rot != "co_ccw" else 1.0
    elif rot == "counter":
        spin = [-1.0, 1.0]
    else:
        spin = -1.0 if rot == "co_cw" else 1.0
    d: dict = {"D_p": float(p["D_p"]), "y_centres": y}
    if p.get("thrust_mode") == "mu":
        d["mu_inf"] = float(p["mu_inf"])
    else:
        d["CT"] = float(p["CT"])
    if float(p.get("swirl_deg", 0.0)) != 0.0:
        d["dalpha_ref_deg"] = float(p["swirl_deg"])
        d["spin"] = spin
    return d


MISSION_FIELD_META = {
    "W_N": ("Weight W [N]", 10.0),
    "V": ("Speed V [m/s]", 0.5),
    "altitude_m": ("Altitude [m]", 100.0),
    "depth_m": ("Depth [m]", 0.1),
}


def mission_field_note(problem_name: str, fld: str) -> str:
    """Why a mission field is not editable on this problem (honest disable).

    Asked of the FAMILY: a modifier twin has its base's mission contract, so
    matching the name exactly made the water page under a chord law say
    "weight not modelled" and "not honoured by this solver" where the
    hydrofoil's own answer is that speed and depth are DESIGN VARIABLES.
    """
    from aerobo import api

    family = api.base_of(problem_name)
    if family == "mission wing" and fld in ("V", "altitude_m"):
        return "design variable here — set its box bounds instead"
    if family == "hydrofoil":
        return ("speed and depth are design variables — set their box "
                "bounds" if fld in ("V", "depth_m")
                else "not part of the cavitation formulation")
    if family == "airfoil (section)":
        return "2-D section at the library Re — no aircraft mission"
    if family.startswith("car rear wing") and fld == "V":
        # the generic answer below would be "not honoured by this solver",
        # and that is FALSE: the speed IS honoured — as a FLAG
        # (api.CAR_WING_KEYS), because this family has no MissionSpec at all
        # and refuses one by name. Saying it is not honoured is what let every
        # car run go off at the family's own 55 m/s with nothing on any card
        # contradicting it.
        return ("a track condition rather than a mission: this family takes "
                "its speed as a flag (api.CAR_WING_KEYS 'V'), not as a "
                "mission field")
    if family.startswith("car rear wing") and fld == "W_N":
        # the generic answer below ("trims to a fixed CL") is the one thing
        # this family does NOT do: it maximises the downforce coefficient
        # under a drag budget, so there is no lift to trim to at all
        return ("no lift target — this wing maximises downforce under a "
                "drag budget rather than carrying a weight")
    if fld == "W_N":
        return "weight not modelled — the solver trims to a fixed CL"
    if fld == "altitude_m":
        return "sea-level flow state fixed by this solver"
    if fld == "depth_m":
        return "air problem — no submergence"
    return "not honoured by this solver"


# Airfoil-ONLY optimiser (section = CST + XFOIL). The tab collects a WING
# GUESS, a twist law and the two section gates, and passes them straight
# through as keyword arguments to api.optimize_airfoil — which maps them onto
# the airfoil_-prefixed api.AIRFOIL_FLAG_KEYS / AIRFOIL_WING_FLAG_KEYS. The
# design Cl and the Reynolds number are NOT collected: both are DERIVED from
# the wing guess (CL = W/(qS), Re at the MAC), so there is no way for a typed
# number to disagree with the geometry it is supposed to describe.
OPT_WING_KEYS = ("mass_kg", "v_ms", "altitude_m", "s_ref_m2",
                 "aspect_ratio", "taper")
OPT_GATE_KEYS = ("tc_min", "cm_max")

#: chord law the Airfoil Optimizer page STARTS on. ON, because a taper ratio
#: alone can only ever draw a straight chord and elliptic is not straight, so
#: order 0 left the biggest planform lever switched off by default. It is a
#: GUI starting point only: ``api.optimize_airfoil`` still defaults to
#: ``chord_order=0``, so every scripted study and published number reproduces
#: bit-for-bit.
OPT_CHORD_DEFAULT = {"order": 2, "chord_max_frac": 0.5}

#: hover help for every editable parameter, keyed by field name. Explains what
#: the number DOES in the physics, not what it is called.
PARAM_HELP = {
    # --- the wing guess ---
    "mass_kg": "Design mass the wing has to carry. Weight W = m·g is the "
               "lift it must make, so this sets the design lift coefficient "
               "CL = W/(qS) together with speed, altitude and area.",
    "v_ms": "Cruise speed. Enters as dynamic pressure q = ½ρV²: flying "
            "faster needs LESS lift coefficient for the same weight, and "
            "raises the section Reynolds number.",
    "altitude_m": "Cruise altitude. Sets air density and viscosity through "
                  "the ISA model — thinner air needs a higher lift "
                  "coefficient and drops the Reynolds number.",
    "s_ref_m2": "YOUR GUESS at the wing planform area. The knob this whole "
                "mode is built around: CL_design = W/(qS), so a smaller "
                "guess forces the section to work at a higher lift "
                "coefficient — and a different section wins.",
    "aspect_ratio": "b²/S. Fixes the span that goes with the area you "
                    "guessed, and with it the induced drag "
                    "CDi ≈ CL²/(π·AR·e).",
    "taper": "Tip chord ÷ root chord. Shapes the chord distribution, which "
             "decides how close the untwisted loading already is to "
             "elliptic — and therefore how much twist can buy. With a chord "
             "law switched on this is only the BASELINE the optimiser "
             "reshapes from.",
    # --- the chord law ---
    #
    # ON by default. A taper ratio can only ever draw a STRAIGHT chord, so
    # leaving the law off hid the single biggest planform lever behind a
    # dropdown most users would never open. The api default stays
    # ``chord_order=0`` so every scripted study and every published number
    # reproduces bit-for-bit; this is a GUI starting point, not a physics
    # change.
    "chord_order": "Shape of the chord law. A taper ratio can only ever give "
                   "a STRAIGHT-TAPER chord, and elliptic chord is not "
                   "straight — so this adds polynomial coefficients that "
                   "reshape the planform at CONSTANT AREA (area is held "
                   "because it is what set CL_design). Fixed = the classic "
                   "trapezoid, i.e. no chord design variables at all.",
    "chord_max_frac": "How far the chord may depart from the straight-taper "
                      "baseline, anywhere on the span, as a fraction of the "
                      "local baseline chord. Also the honest limit on the "
                      "single-Reynolds-number polar: the section is flown at "
                      "ONE Re (the baseline MAC's), so a wildly reshaped "
                      "planform would stretch that assumption past meaning.",
    # --- the twist law ---
    "twist_order": "Shape of the twist law θ(η) = Σ tⱼ·ηʲ along the "
                   "semi-span, pinned to zero at the root. Linear is the "
                   "classic single washout knob; higher orders let the "
                   "optimiser bend the law between root and tip. Root "
                   "incidence is NOT in here — it is solved for so the wing "
                   "trims to the design CL.",
    "twist_max_deg": "Cap on |twist| ANYWHERE on the semi-span, not just at "
                     "the tip: enforced as a constraint on the measured "
                     "envelope, so a high-order law cannot sneak past it "
                     "inboard.",
    "alpha_max_deg": "Cap on the largest LOCAL geometric angle of attack "
                     "(root incidence + local twist) at the trimmed state. "
                     "Keeps the design off a stall-adjacent attitude.",
    # --- the two section gates ---
    "tc_min": "Minimum section thickness ÷ chord. Structural-depth proxy: "
              "without it, drag minimisation runs to a vanishing-thickness "
              "plate with nowhere to put a spar or fuel.",
    "cm_max": "Cap on |Cm| at the design lift. Trim-drag proxy: the tail has "
              "to react the wing pitching moment, and unlimited aft loading "
              "buys drag it never pays back.",
    # --- the seed-screen weights ---
    "w_ldcr": "How much the seed score rewards lift-to-drag AT THE DESIGN "
              "LIFT — efficiency at the point you actually cruise.",
    "w_clmax": "How much the seed score rewards maximum lift coefficient — "
               "stall speed, field length, gust margin.",
    "w_cm": "How much the seed score PENALISES pitching moment (lower is "
            "better) — trim drag and tail size.",
    "w_ldmax": "How much the seed score rewards the best L/D anywhere in the "
               "low-α band — off-design efficiency.",
    "w_thick": "How much the seed score rewards thickness — spar depth, fuel "
               "volume, structural mass.",
    "w_astall": "How much the seed score rewards a late stall — how much α "
                "margin there is before the section lets go.",
    # --- search ---
    "optimiser": "Search strategy. Bayesian optimisation fits a surrogate to "
                 "the expensive XFOIL response and is the right default at "
                 "small budgets; GA / Random / Sobol are baselines.",
    "budget": "Candidate designs evaluated. Every NEW section shape costs a "
              "real viscous XFOIL sweep (seconds); twist-only changes reuse "
              "the cached polar and are nearly free.",
    "seed": "Random seed for the sampler. Same seed + same settings "
            "reproduces the run exactly.",
    "shortlist": "How many top-scoring library sections are flown on the real "
                 "wing before one is chosen as the seed. The weighted score "
                 "is only a 2-D proxy — this picks the seed by the actual "
                 "objective, L/D at the cruise CL. One real XFOIL sweep each, "
                 "so a few seconds per entry.",
    # --- library screen (typed operating point + floors) ---
    "re": "Reynolds number ρVc/μ the polars are flown at. Sets how thick the "
          "boundary layer is relative to the chord, and therefore where the "
          "drag bucket sits — a section that wins at 1e6 need not win at 3e6.",
    "mach": "Free-stream Mach number. Left at 0 the polars are "
            "incompressible; compressibility is a section-level correction "
            "only, and above ~0.7 the model has no validity.",
    "cl_design": "The lift coefficient every section is compared AT. Drag "
                 "only means something at equal lift, so this is what "
                 "'L/D at design Cl' and '|Cm|' are interpolated to.",
    "floor_clmax": "Reject any section whose maximum lift coefficient is "
                   "below this. Blank = no floor.",
    "floor_ldcr": "Reject any section whose L/D at the design Cl is below "
                  "this. Blank = no floor.",
    "floor_astall": "Reject any section that stalls below this angle. "
                    "Blank = no floor.",
}


# --------------------------------------------------------------- utilities

def _screen_needs_sweep(cond: dict) -> bool:
    """Would screening at ``cond`` cost live XFOIL sweeps?

    The library's polars are cached at ONE operating point. Asking for that
    one is instant — the ranking comes off the branch sidecar and the design
    Cl is honoured exactly. Asking for any other Reynolds number (or Mach) is
    a fresh viscous sweep per section, which is hours over the whole database
    and is why those requests go through ``api.screen_at_point``'s shortlist
    instead. Shared by the V1 and V2 library pages, which offer the same
    freely-typed point.
    """
    from aerobo import api

    pt = api.screen_library_point()
    if not pt:
        return False        # nothing cached to shortlist from anyway
    re_want = float(cond.get("re") or 0.0)
    if re_want <= 0.0:
        return False
    return (abs(re_want - float(pt["re"])) > 1e-9 * re_want
            or float(cond.get("mach") or 0.0) != float(pt.get("mach", 0.0)))


def _fmt(v, nd=4) -> str:
    """Compact numeric formatting for cards and tables."""
    if v is None:
        return "—"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not math.isfinite(f):
        return "—"
    if f != 0 and (abs(f) < 1e-3 or abs(f) >= 1e5):
        return f"{f:.{nd}g}"
    return f"{f:.{nd}g}"


def _nan_history(hist) -> np.ndarray:
    """JSON history (None where -inf) -> float array with NaN gaps."""
    return np.array([np.nan if v is None else float(v) for v in (hist or [])],
                    dtype=float)


#: What the vertical axis is CALLED once a mirrored design has been turned
#: the right way up: the numbers are still the model's (+z is the car's down,
#: which is what every margin, height and clearance in the breakdown is in),
#: and the axis is the thing that has been drawn downward.
_MIRROR_AXIS_NOTE = " · +z is the car's DOWN (axis drawn downward)"
#: ...and what the picture is, for the shells that keep figure titles.
_MIRROR_TITLE_NOTE = "  ·  car frame: the wing sits ON TOP of its endplates"


def _mirrored(geom: dict) -> bool:
    """Was this geometry solved in the car's MIRRORED frame?

    Asked of the geometry block the drawers are handed, which
    ``api.design_report`` carries the solver's own ``frame`` string out on —
    never of a problem name. A car wing is modelled upside down (model +z is
    the car's downward direction) so that its lift IS its downforce and the
    track is an image plane; drawn in that frame the endplates stand above
    the wing, which is the one thing about the picture no reader believes.
    """
    return _geom_frame.is_mirrored(geom)


def _vlabel(geom: dict, base: str = "z [m]") -> str:
    """A vertical-axis title that says which way up the axis is drawn."""
    return base + (_MIRROR_AXIS_NOTE if _mirrored(geom) else "")


def _flip_if_mirrored(lay: dict, geom: dict) -> dict:
    """Draw a 2-D view's vertical axis DOWNWARD for a mirrored design.

    The data stays in the frame it was solved in — every number beside these
    figures is in it, and a hover readout that disagreed with the breakdown
    would be a second frame nobody asked for — so what is reversed is the
    AXIS, exactly as :func:`fig_planform` reverses the streamwise one to
    draw the nose up. For a wing symmetric in span, mirroring the picture in
    z is the same rotation as turning the car back over, so this IS the
    vehicle's own orientation and not a trick of the axis.
    """
    if _mirrored(geom):
        lay["yaxis"]["autorange"] = "reversed"
    return lay


def _base_layout(title: str, xlab: str, ylab: str, h: int = 280) -> dict:
    """Theme-agnostic plotly layout: transparent bg, muted axes."""
    return dict(
        title=dict(text=title, font=dict(size=14, color=MUTED), x=0.02),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=MUTED, size=11),
        xaxis=dict(title=xlab, gridcolor=GRID, zerolinecolor=GRID),
        yaxis=dict(title=ylab, gridcolor=GRID, zerolinecolor=GRID),
        margin=dict(l=54, r=14, t=40, b=42),
        height=h,
        legend=dict(orientation="h", y=1.14, x=0.35, font=dict(size=10)),
        showlegend=True,
    )


def _empty_fig(msg: str, h: int = 280) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(**_base_layout("", "", "", h))
    fig.add_annotation(text=msg, showarrow=False, font=dict(color=MUTED))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def _macos_notify(title: str, message: str):
    """Best-effort native notification (silent no-op off macOS)."""
    try:
        subprocess.run(
            ["osascript", "-e",
             f'display notification "{message}" with title "{title}"'],
            capture_output=True, timeout=5)
    except Exception:
        pass


# ================================================================ figures
# All figure builders are pure: state dicts in, go.Figure out.

def fig_constraint_diagram(d, title: str = "Constraint diagram"):
    """The wing loading, drawn: every requirement as a line, the allowed band
    shaded, and the aerodynamic optimum marked where it exists.

    Takes an ``aerobo.constraint_diagram.Diagram``. Air diagrams carry T/W
    curves (feasible ABOVE each) and W/S limits (feasible LEFT of each);
    water diagrams carry limits only — a foil's thrust matching is a
    propulsion question this package does not model — so the figure degrades
    to the band and its two edges.
    """
    fig = go.Figure()
    ws = np.asarray(d.ws_grid_pa, dtype=float)
    curves = [c for c in d.constraints if c.kind == "twr_curve"]
    caps = [c for c in d.constraints if c.kind == "ws_max"]
    for i, c in enumerate(curves):
        fig.add_trace(go.Scatter(
            x=ws, y=np.asarray(c.twr, dtype=float), mode="lines", name=c.name,
            line=dict(width=2, dash=("solid" if i == 0 else "dot")),
            hovertemplate=f"{c.name}<br>W/S %{{x:.0f}} Pa<br>"
                          f"T/W %{{y:.3f}}<extra></extra>"))
    if curves:
        env = np.max(np.vstack([np.asarray(c.twr, dtype=float)
                                for c in curves]), axis=0)
        fig.add_trace(go.Scatter(
            x=ws, y=env, mode="lines", name="required T/W",
            line=dict(color=ACCENT, width=3),
            hovertemplate="required T/W %{y:.3f}<extra></extra>"))
    y_top = 1.0
    if curves:
        y_top = float(np.nanmax([np.nanmax(c.twr) for c in curves])) * 1.1
    for c in caps:
        binding = c.name == d.binding
        fig.add_trace(go.Scatter(
            x=[c.ws_pa, c.ws_pa], y=[0.0, y_top], mode="lines",
            name=f"{c.name} limit",
            line=dict(color=(BAD if binding else WARN), width=2,
                      dash="dash")))
    if d.ws_max_pa:
        fig.add_vrect(x0=float(ws[0]), x1=float(d.ws_max_pa),
                      fillcolor=BAND, line_width=0, layer="below")
    if d.ws_min_drag_pa:
        fig.add_trace(go.Scatter(
            x=[d.ws_min_drag_pa], y=[d.twr_at(d.ws_min_drag_pa) or 0.0],
            mode="markers", name="minimum-drag W/S",
            marker=dict(size=11, symbol="star", color=GOOD)))
    layout = _base_layout(title, "wing loading W/S [N/m²]",
                          "thrust / weight" if curves else "")
    layout["showlegend"] = True
    fig.update_layout(**layout)
    if not curves:
        fig.update_yaxes(visible=False, range=[0, 1])
    return fig


def fig_convergence(records: list[dict], history=None, title="Convergence"):
    """Objective vs evaluation — bounded by the designs that actually FLEW.

    A refused design returns the -100 sentinel (``objective.PENALTY``). Left
    on a shared axis it owns the whole range: an L/D run climbing 38 -> 41,
    or a 0-100 composite, becomes a flat line pinned at the top with a cliff
    below it, and the convergence this plot exists to show cannot be read.

    So the axis is set from the scored points (:func:`metrics.convergence_yrange`)
    and the refusals are drawn as their own trace on the floor — still
    counted, still hoverable at their true value, no longer setting the
    scale. On a run with nothing off scale the range is left to plotly and
    the figure is exactly the one this function has always drawn.
    """
    fig = go.Figure()
    xs: list = []
    ys: list = []
    keep: list = []
    if records:
        n = [r["n"] for r in records]
        f = [r.get("f") for r in records]
        feas = [bool(r.get("feasible", True)) for r in records]
        colors = [GOOD if ok else BAD for ok in feas]
        fig.add_trace(go.Scatter(
            x=n, y=f, mode="markers", name="per-eval",
            marker=dict(size=6, color=colors, opacity=0.75),
            hovertemplate="eval %{x}<br>f = %{y:.5g}<extra></extra>"))
        best = [r.get("best") for r in records]
        bx = [i for i, b in zip(n, best) if b is not None and math.isfinite(b)]
        by = [b for b in best if b is not None and math.isfinite(b)]
        fig.add_trace(go.Scatter(
            x=bx, y=by, mode="lines", name="best feasible so far",
            line=dict(color=ACCENT, width=2.5)))
        xs, ys, keep = list(n), list(f), list(by)
    elif history is not None:
        h = _nan_history(history)
        fig.add_trace(go.Scatter(
            x=np.arange(1, h.size + 1), y=h, mode="lines",
            name="best so far", line=dict(color=ACCENT, width=2.5)))
        xs, ys, keep = list(range(1, h.size + 1)), [float(v) for v in h], []
    fig.update_layout(**_base_layout(title, "evaluation", "objective"))
    _clip_refusals(fig, xs, ys, keep)
    return fig


def _clip_refusals(fig, xs, ys, keep=()) -> None:
    """Bound ``fig``'s y axis to the scored points and mark what that hides.

    Mutates the figure. Does nothing at all when no point falls outside the
    range the scored values ask for — "nothing is off scale" and "the axis
    was widened for a refusal" must not look the same.
    """
    rng = _metrics.convergence_yrange(ys, keep)
    if rng is None:
        return
    lo, hi = rng
    off = _metrics.offscale(ys, lo)
    if not off:
        return
    floor = lo + 0.02 * (hi - lo)
    fig.add_trace(go.Scatter(
        x=[xs[i] for i, _ in off], y=[floor] * len(off), mode="markers",
        name=f"refused / off scale ({len(off)})",
        marker=dict(symbol="x", size=7, color=BAD, opacity=0.85),
        customdata=[v for _, v in off],
        hovertemplate="eval %{x}<br>f = %{customdata:.5g}"
                      "  (below the axis)<extra></extra>"))
    fig.update_yaxes(range=[lo, hi])


def fig_surrogate(records: list[dict], iters: list[dict]):
    """GP one-step-ahead prediction error |f - mu| vs its own sigma."""
    pts = []
    by_idx = {r["n"]: r for r in records if r.get("f") is not None}
    for it in iters or []:
        if it.get("mu") is None or it.get("sigma") is None:
            continue
        rec = by_idx.get(it.get("eval_index"))
        if rec is None:
            continue
        pts.append((it["eval_index"], abs(rec["f"] - it["mu"]),
                    it["sigma"], (rec["f"] - it["mu"]) / max(it["sigma"], 1e-12)))
    fig = go.Figure()
    if pts:
        idx = [p[0] for p in pts]
        fig.add_trace(go.Scatter(
            x=idx, y=[p[1] for p in pts], mode="markers+lines",
            name="|f − μ| (GP residual)",
            line=dict(color=WARN, width=1.2), marker=dict(size=6)))
        fig.add_trace(go.Scatter(
            x=idx, y=[p[2] for p in pts], mode="lines",
            name="GP σ at candidate", line=dict(color=ACCENT, width=1.6,
                                                dash="dot")))
        lay = _base_layout("Surrogate learning — prediction error vs σ",
                           "evaluation", "|f − μ|,  σ")
        lay["yaxis"]["type"] = "log"
        fig.update_layout(**lay)
    else:
        fig = _empty_fig("GP diagnostics appear once BO iterations start "
                         "(BO runs only)")
    return fig


def fig_acquisition(iters: list[dict]):
    fig = go.Figure()
    pts = [(it["eval_index"], it["acq"]) for it in iters or []
           if it.get("acq") is not None]
    pf = [(it["eval_index"], it["p_feasible"]) for it in iters or []
          if it.get("p_feasible") is not None]
    if pts:
        fig.add_trace(go.Scatter(
            x=[p[0] for p in pts], y=[p[1] for p in pts],
            mode="markers+lines", name="acquisition at candidate",
            line=dict(color=ACCENT, width=1.4), marker=dict(size=6)))
    if pf:
        fig.add_trace(go.Scatter(
            x=[p[0] for p in pf], y=[p[1] for p in pf],
            mode="lines", name="P(feasible) at candidate", yaxis="y2",
            line=dict(color=GOOD, width=1.6, dash="dot")))
    if pts or pf:
        lay = _base_layout("Acquisition decay — is BO converging?",
                           "evaluation", "acq value (log-EI)")
        lay["yaxis2"] = dict(title="P(feasible)", overlaying="y", side="right",
                             range=[0, 1.02], gridcolor="rgba(0,0,0,0)")
        fig.update_layout(**lay)
    else:
        fig = _empty_fig("Acquisition trace appears once BO iterations start")
    return fig


def fig_constraints(records: list[dict], labels: list[str]):
    fig = go.Figure()
    rows = [(r["n"], r.get("g")) for r in records if r.get("g")]
    if rows:
        m = len(rows[0][1])
        palette = [ACCENT, WARN, "#a78bfa", "#f472b6"]
        for j in range(m):
            name = labels[j] if j < len(labels) else f"g[{j}]"
            fig.add_trace(go.Scatter(
                x=[n for n, _ in rows], y=[g[j] for _, g in rows],
                mode="markers+lines", name=name,
                line=dict(color=palette[j % len(palette)], width=1.2),
                marker=dict(size=5)))
        fig.add_hline(y=0.0, line=dict(color=BAD, width=1.5, dash="dash"))
        fig.update_layout(**_base_layout(
            "Constraint margins — feasible ⟺ all ≥ 0", "evaluation",
            "signed margin g"))
    else:
        fig = _empty_fig("No constraints on this problem")
    return fig


def fig_calibration(records: list[dict], iters: list[dict]):
    """Standardised residual (f − μ)/σ — GP calibration check (±2 band)."""
    by_idx = {r["n"]: r for r in records if r.get("f") is not None}
    pts = []
    for it in iters or []:
        if it.get("mu") is None or it.get("sigma") is None:
            continue
        rec = by_idx.get(it.get("eval_index"))
        if rec is not None:
            pts.append((it["eval_index"],
                        (rec["f"] - it["mu"]) / max(it["sigma"], 1e-12)))
    fig = go.Figure()
    if pts:
        idx = [p[0] for p in pts]
        fig.add_hrect(y0=-2, y1=2, fillcolor=BAND, line_width=0)
        fig.add_trace(go.Scatter(
            x=idx, y=[p[1] for p in pts], mode="markers",
            name="(f − μ)/σ", marker=dict(size=7, color=ACCENT)))
        fig.add_hline(y=0, line=dict(color=MUTED, width=1, dash="dot"))
        fig.update_layout(**_base_layout(
            "GP calibration — standardised residual (±2σ band)",
            "evaluation", "(f − μ)/σ"))
    else:
        fig = _empty_fig("Calibration appears once BO iterations start")
    return fig


def _planform_arrays(geom: dict, x_best=None, labels=None):
    """(y, chord) full-span arrays — :func:`aerobo.cad.planform_arrays`."""
    return cad.planform_arrays(geom, x_best, labels)


def _planform_surface_traces(sf: dict, tail: dict, colour):
    """Plan-view outline of one surface (+ its elevator strip), x aft +."""
    y, c = _planform_arrays({"y": sf.get("y"), "chord": sf.get("chord")})
    if y is None:
        return []
    x0 = float(sf.get("x_offset", 0.0))
    if sf.get("name") == "tail":
        vee = _v_tail_path(tail, 0.0)
        if vee is not None:          # a V-tail's PROJECTED plan outline
            y, _z, c, _m = vee
    le, te = x0 - 0.25 * c, x0 + 0.75 * c
    out = [go.Scatter(
        x=np.concatenate([y, y[::-1]]),
        y=np.concatenate([le, te[::-1]]),
        fill="toself", mode="lines", name=sf.get("name", "surface"),
        line=dict(color=colour, width=2), fillcolor=BAND, hoverinfo="skip")]
    frac = float(tail.get("elevator_chord_frac") or 0.0)
    if (sf.get("name") == "tail" and frac > 0.0
            and tail.get("control") == "elevator"):
        hinge = x0 + (0.75 - frac) * c
        out.append(go.Scatter(
            x=np.concatenate([y, y[::-1]]),
            y=np.concatenate([hinge, te[::-1]]),
            fill="toself", mode="lines",
            name=f"elevator ({frac:.0%} chord)",
            line=dict(color=WARN, width=1, dash="dot"),
            fillcolor="rgba(240,160,60,0.28)", hoverinfo="skip"))
    return out


def _planform_np_traces(yv, cv, mask, x0: float, name: str, colr: str,
                        dev_name: str):
    """Plan-view outline of ONE nonplanar surface, its tip device separate.

    The device's HORIZONTAL PROJECTION is drawn as its own dotted patch
    because that projection is exactly what a span cap does (or does not)
    pay for. Split per surface so a tandem's two wings — and their two
    devices — are two outlines at two stagger stations rather than one
    polyline running from the front wing's tip into the rear wing's.
    """
    out = []
    first = {True: True, False: True}          # one legend entry per kind
    for seg in np.split(np.arange(yv.size),
                        np.flatnonzero(np.diff(mask.astype(int)) != 0) + 1):
        if seg.size < 2:
            continue
        ys, cs_ = yv[seg], cv[seg]
        is_wl = bool(mask[seg[0]])
        out.append(go.Scatter(
            x=np.concatenate([ys, ys[::-1]]),
            y=np.concatenate([x0 - 0.25 * cs_, (x0 + 0.75 * cs_)[::-1]]),
            fill="toself", mode="lines",
            name=dev_name if is_wl else name,
            showlegend=first[is_wl],
            line=dict(color=WARN if is_wl else colr,
                      width=2, dash="dot" if is_wl else "solid"),
            fillcolor=BAND, hoverinfo="skip"))
        first[is_wl] = False
    return out


def _planform_fin_trace(geom: dict):
    """The vertical surface seen from ABOVE: its chord, on the centreline.

    Edge-on, so it carries no span here — and that is the point. This is the
    view a STATION is read in, and until a tandem's fin was drawn in it there
    was no picture that showed where the surface stands relative to the two
    wings it sits between. Plotted in the pane's own axes (span across,
    streamwise up the y-axis, reversed for nose-up) like every other trace
    here.
    """
    blk = (geom or {}).get("fin")
    if not blk:
        return []
    x_le = float(blk.get("x_le_m", blk.get("x_qc_m", 0.0) - 0.25
                         * float(blk.get("chord_m", 0.0))))
    c = float(blk.get("chord_m", 0.0))
    if not c > 0.0:
        return []
    y0 = float(blk.get("y_m", 0.0))
    name = ("strut" if str(blk.get("kind", "fin")) == "mast" else "fin")
    return [go.Scatter(x=[y0, y0], y=[x_le, x_le + c], mode="lines",
                       name=f"{name} (edge-on)", showlegend=True,
                       line=dict(color=MUTED, width=6))]


def fig_planform(geom: dict, x_best=None, labels=None):
    surfs = geom.get("surfaces")
    fig = go.Figure()
    if surfs:
        # tandem AND tail: draw each surface at its own streamwise offset,
        # in the package's axes (x downstream +), so an aft tail is aft and a
        # canard is ahead — the same frame the arm, CG and neutral point are
        # quoted in.
        tail = geom.get("tail") or {}
        colors = [ACCENT, WARN]
        for i, sf in enumerate(surfs):
            # the elevator strip travels with its surface (see the helper)
            for tr in _planform_surface_traces(sf, tail, colors[i % 2]):
                fig.add_trace(tr)
        for tr in _planform_fin_trace(geom):
            fig.add_trace(tr)
        if not fig.data:
            return _empty_fig("No planform geometry available")
        if tail:
            # the CG and the neutral point ARE the tail problem: the static
            # margin gate is the distance between them, so draw both.
            for key, colr, nm in (("x_cg", WARN, "CG"),
                                  ("x_np", ACCENT, "neutral point")):
                v = tail.get(key)
                if v is None:
                    continue
                fig.add_trace(go.Scatter(
                    x=[0.0], y=[float(v)], mode="markers+text",
                    marker=dict(color=colr, size=11,
                                symbol="circle" if key == "x_cg" else "x"),
                    text=[nm], textposition="middle right",
                    textfont=dict(color=colr, size=10),
                    name=nm, hoverinfo="skip"))
            sm = tail.get("SM")
            bits = [f"{tail.get('type', 'tail')}",
                    f"arm {abs(float(tail.get('l_t', 0.0))):.2f} m",
                    f"S_t {float(tail.get('S_t', 0.0)):.2f} m²",
                    f"h {float(tail.get('dz_m', 0.0)):.2f} m"]
            if tail.get("control") == "elevator":
                bits.append(f"δe {float(tail.get('delta_e_deg', 0.0)):+.1f}°")
            else:
                bits.append(f"i_t {float(tail.get('i_t_deg', 0.0)):+.1f}°")
            if sm is not None:
                bits.append(f"SM {float(sm):.3f}")
            title = "Wing + tail (plan view, arm to scale) · " + " · ".join(bits)
        else:
            title = "Planform (plan view, tandem stagger to scale)"
        lay = _base_layout(title, "span y [m]",
                           "x [m] · downstream + (nose up)", h=340)
        lay["yaxis"]["scaleanchor"] = "x"
        #: The streamwise axis is DRAWN nose-up. The data stays in the
        #: package's axes (x downstream +, LE at -0.25 c), because every
        #: number beside this figure is in them and a test pins it; what
        #: is reversed is the AXIS, not the geometry. Without this the
        #: pane maps (span starboard +) x (streamwise aft +), whose cross
        #: product is -z — a view from BELOW with the nose at the bottom,
        #: and a reader sees the trailing edge where the leading edge
        #: belongs on both the wing and the tail at once.
        lay["yaxis"]["autorange"] = "reversed"
        lay["showlegend"] = True
        fig.update_layout(**lay)
        return fig
    np_geom = _nonplanar(geom)
    title = "Planform (plan view, c/4 line straight)"
    show_legend = False
    if np_geom is not None:
        # winglet designs: the wing panel is the planform; the winglet's
        # HORIZONTAL PROJECTION is drawn separately because that projection
        # is exactly what the span cap does (or does not) pay for.
        yv, zv, cv, mask = np_geom
        for tr in _planform_np_traces(yv, cv, mask, 0.0, "wing", ACCENT,
                                      "winglet (projected)"):
            fig.add_trace(tr)
        # the REAR WING of a nonplanar tandem, at its own stagger and with
        # its own tip device: it is a second surface, not more span on this
        # one, and drawn from the same arrays it was flown on
        second_sf = geom.get("second_surface")
        second_np = _nonplanar(second_sf) if second_sf else None
        if second_np is not None:
            ys2, _zs2, cs2, m2 = second_np
            for tr in _planform_np_traces(
                    ys2, cs2, m2, float(second_sf.get("x_offset", 0.0)),
                    "rear wing", GOOD, "rear tip device (projected)"):
                fig.add_trace(tr)
        elif second_sf:
            for tr in _planform_surface_traces(second_sf, {}, GOOD):
                fig.add_trace(tr)
        # a tail solved in the SAME system (wingtail.py) belongs in the plan
        # view too, at its own arm
        tail_np = geom.get("tail") or {}
        tail_sf = geom.get("tail_surface")
        if tail_np and tail_sf:
            for tr in _planform_surface_traces(tail_sf, tail_np, WARN):
                fig.add_trace(tr)
        b_ref = float(geom.get("b") or 0.0)
        if b_ref > 0:
            for sgn in (-1.0, 1.0):
                fig.add_vline(x=sgn * b_ref / 2.0,
                              line=dict(color=MUTED, width=0.8, dash="dot"))
        span_proj = 2.0 * float(np.max(np.abs(yv)))
        title = (f"Planform · projected span {span_proj:.2f} m"
                 + (f" vs reference b = {b_ref:.2f} m" if b_ref else ""))
        show_legend = True
    else:
        y, c = _planform_arrays(geom, x_best, labels)
        if y is None:
            return _empty_fig("No planform geometry available for this problem")
        le, te = -0.25 * c, 0.75 * c
        fig.add_trace(go.Scatter(
            x=np.concatenate([y, y[::-1]]),
            y=np.concatenate([le, te[::-1]]),
            fill="toself", mode="lines", name="planform",
            line=dict(color=ACCENT, width=2), fillcolor=BAND,
            hoverinfo="skip"))
        # a second surface solved in the SAME system belongs here too, and
        # this is the branch a PLANAR main surface takes — the hydrofoil's
        # elevator (no tip device on the foil) landed here and was drawn
        # nowhere, which said the craft had one surface. A nonplanar tandem
        # whose wings carry no device (the CST-section pair) is the same
        # case: planar front wing, second WING at the stagger.
        second_flat = geom.get("second_surface")
        if second_flat:
            for tr in _planform_surface_traces(second_flat, {}, GOOD):
                fig.add_trace(tr)
            show_legend = True
            title = (f"Planform + rear wing (plan view, stagger to scale) · "
                     f"{float(second_flat.get('x_offset', 0.0)):+.2f} m aft")
        tail_flat, tail_sf_flat = geom.get("tail") or {}, geom.get(
            "tail_surface")
        if tail_flat and tail_sf_flat:
            for tr in _planform_surface_traces(tail_sf_flat, tail_flat, WARN):
                fig.add_trace(tr)
            show_legend = True
            title = (f"Planform + second surface (plan view, arm to scale) · "
                     f"arm {abs(float(tail_flat.get('l_t', 0.0))):.2f} m · "
                     f"S_t {float(tail_flat.get('S_t', 0.0)):.3f} m²")
    fin_tr = _planform_fin_trace(geom)
    for tr in fin_tr:
        fig.add_trace(tr)
        show_legend = True
    fig.add_hline(y=0, line=dict(color=MUTED, width=0.8, dash="dash"))
    lay = _base_layout(title, "span y [m]",
                       "x [m] · downstream + (nose up)", h=300)
    lay["yaxis"]["scaleanchor"] = "x"
    #: The streamwise axis is DRAWN nose-up. The data stays in the
    #: package's axes (x downstream +, LE at -0.25 c), because every
    #: number beside this figure is in them and a test pins it; what
    #: is reversed is the AXIS, not the geometry. Without this the
    #: pane maps (span starboard +) x (streamwise aft +), whose cross
    #: product is -z — a view from BELOW with the nose at the bottom,
    #: and a reader sees the trailing edge where the leading edge
    #: belongs on both the wing and the tail at once.
    lay["yaxis"]["autorange"] = "reversed"
    lay["showlegend"] = show_legend
    fig.update_layout(**lay)
    return fig


def _front_path(sf: dict):
    """(y, z, is_winglet) of ONE surface in the front view, or None.

    The surface's OWN panel path where it exports one — a designed tail or
    elevator carries a tip device, and that device lives entirely in this
    view. Only a surface with no z at all falls back to the flat segment at
    its offset, which is all the published lifting-line pair reports.
    """
    y = np.asarray(sf.get("y") or [], dtype=float)
    if y.size < 2:
        return None
    z0 = float(sf.get("z_offset", 0.0))
    z = np.asarray(sf.get("z") or [], dtype=float)
    if z.size != y.size:
        half = float(np.max(np.abs(y)))
        return (np.array([-half, half]), np.array([z0, z0]),
                np.zeros(2, dtype=bool))
    wl = np.asarray(sf.get("is_winglet") or [], dtype=bool)
    if wl.size != y.size:
        wl = np.zeros(y.size, dtype=bool)
    return y, z, wl


def _folded_vee_path(tail: dict, sf: dict):
    """(y, z, is_winglet) of a V-tail's panels WITH the tip device they carry,
    or None when there is nothing to fold.

    Only for the shape that is BOTH — a V-tail whose stabiliser reports its
    own spanwise path because it flies a device. A V-tail without one keeps
    the dotted :func:`cad.v_tail_path` annotation over the flat surface the
    solver flew, which is all a published lifting-line tail reports.
    """
    if str(tail.get("type")) != "v_tail" \
            or float(tail.get("dihedral_deg") or 0.0) <= 0.0 \
            or cad.nonplanar_arrays(sf) is None:
        return None
    path = cad.tail_path(tail, sf)
    if path is None:
        return None
    y, z, _c, wl, _seg = path
    return y, z, wl


def _add_front_path(fig, y, z, mask, name: str, colr: str,
                    wl_colr: str = WARN, width: int = 4):
    """Draw one spanwise path, its tip device in its own colour.

    Split at the wing/winglet junctions and re-joined by repeating the
    boundary station, so the polyline stays continuous: drawing the segments
    independently left a visible gap exactly at the junction the cant angle
    is measured from.
    """
    mask = np.asarray(mask, dtype=bool)
    cuts = np.flatnonzero(np.diff(mask.astype(int)) != 0) + 1
    first = {True: True, False: True}
    for seg in np.split(np.arange(y.size), cuts):
        if seg.size < 1:
            continue
        lo = int(seg[0])
        hi = int(seg[-1]) + 1
        lo = max(lo - 1, 0) if lo else lo          # share the junction point
        if hi - lo < 2:
            continue
        is_wl = bool(mask[int(seg[0])])
        label = f"{name} tip device" if is_wl else name
        fig.add_trace(go.Scatter(
            x=y[lo:hi], y=z[lo:hi], mode="lines", name=label,
            showlegend=first[is_wl],
            line=dict(color=wl_colr if is_wl else colr, width=width)))
        first[is_wl] = False


def _tail_height_note(geom: dict) -> str:
    """"0.42 m above / 0.06 m below the wing plane" — from the geometry that
    was flown, and never a hard-coded direction.

    The elevator of a hydrofoil sits BELOW the foil, and its offset is a
    design variable; the old caption read a key that family does not report
    and said "0.00 m above the wing plane" about a surface 60 mm under it.
    """
    tail = geom.get("tail") or {}
    ts = geom.get("tail_surface") or geom.get("second_surface") or {}
    if not ts:
        # the published lifting-line pair exports its two surfaces as a list;
        # the second one is still the surface whose height this names
        ts = next((s for s in (geom.get("surfaces") or [])[1:]), {})
    name = str(ts.get("name") or "tail")
    dz = ts.get("z_offset", tail.get("dz_m"))
    if dz is None:
        return ""
    dz = float(dz)
    if abs(dz) < 1e-9:
        return f" · {name} in the wing plane"
    return (f" · {name} {abs(dz):.2f} m "
            f"{'above' if dz > 0.0 else 'below'} the wing plane")


def _front_fin_trace(geom: dict):
    """The VERTICAL surface as one front-view line, or [] when there is none.

    The front view is the projection a vertical surface exists in — its
    height is the whole of what it shows — and this view was drawing every
    surface EXCEPT that one. A tandem's fin, an aeroplane's fin and a
    hydrofoil's strut were all reported, charged, lofted and flown while the
    one picture that shows a height had no line for them.

    Drawn at y = 0 (a centreline surface) from ``z_root`` to
    ``z_root + height``, which is signed: a ventral fin and a hydrofoil's
    mast run DOWN and this draws them downward without a special case.
    """
    blk = (geom or {}).get("fin")
    if not blk:
        return []
    z0 = float(blk.get("z_root_m", 0.0))
    h = float(blk.get("height_m", 0.0))
    if not abs(h) > 0.0:
        return []
    y0 = float(blk.get("y_m", 0.0))
    name = ("strut" if str(blk.get("kind", "fin")) == "mast" else "fin")
    return [go.Scatter(x=[y0, y0], y=[z0, z0 + h], mode="lines",
                       name=name, showlegend=True,
                       line=dict(color=MUTED, width=5))]


def _fig_frontview_tail(geom: dict):
    """Front view of a wing + tail: the height the tail sits at, the tip
    device it carries, and — for a V-tail — the panels the equivalent flat
    surface stands in for.

    Returns None unless the design carries tail geometry.
    """
    tail = geom.get("tail") or {}
    second = geom.get("tail_surface") or geom.get("second_surface")
    surfs = geom.get("surfaces") or []
    if not surfs and second and geom.get("y") is not None:
        # a second surface solved in the SAME system exports its panels under
        # its own key rather than as a two-entry ``surfaces`` list (only the
        # published lifting-line pair writes that). Reading just the list
        # left the height of every nonplanar tail — and the whole hydrofoil
        # elevator, which is the surface whose DEPTH is the design variable —
        # out of the one view that height lives in.
        surfs = [{"name": "wing", "y": geom["y"], "z": geom.get("z"),
                  "is_winglet": geom.get("is_winglet"), "z_offset": 0.0},
                 second]
    # a TANDEM pair names no tail and still has two surfaces at two heights,
    # which is the one thing this view is for. So the trigger is the second
    # SURFACE, not the tail block — and everything tail-specific below reads
    # an empty dict harmlessly when there is no tail.
    #
    # ...and ONE surface plus a VERTICAL is also a front view, because a
    # height is exactly what this projection shows. That case is a plain
    # hydrofoil: a single planar foil under a strut, which had no front view
    # at all while the strut it hangs from was reported, charged and flown.
    if len(surfs) < 2:
        if not (geom.get("fin") and geom.get("y") is not None):
            return None
        surfs = [{"name": "wing", "y": geom["y"], "z": geom.get("z"),
                  "is_winglet": geom.get("is_winglet"), "z_offset": 0.0}]
    folded = (_folded_vee_path(tail, surfs[1]) if len(surfs) > 1 else None)
    fig = go.Figure()
    for i, (sf, colr) in enumerate(zip(surfs, (ACCENT, GOOD))):
        path = folded if (i == 1 and folded is not None) else _front_path(sf)
        if path is None:
            continue
        y, z, wl = path
        _add_front_path(fig, y, z, wl, sf.get("name", "surface"), colr,
                        width=5)
    # a V-tail is FLOWN as the equivalent flat tail of area S_t cos^2(Gamma);
    # the dihedralled panels are drawn dotted so the difference between what
    # was modelled and what would be built is visible rather than implied.
    # Not where they are already drawn SOLID: a stabiliser carrying a tip
    # device reports its own path, so the fold above IS the built shape and
    # a dotted copy of it would only claim the panels twice.
    gam = float(tail.get("dihedral_deg") or 0.0)
    if (tail.get("type") == "v_tail" and gam > 0.0 and folded is None
            and len(surfs) > 1):
        b_t = float(tail.get("b_t") or 0.0)
        z0 = float(surfs[1].get("z_offset", 0.0))
        half = 0.5 * b_t / max(np.cos(np.deg2rad(gam)), 1e-6)
        for sgn in (-1.0, 1.0):
            fig.add_trace(go.Scatter(
                x=[0.0, sgn * half * np.cos(np.deg2rad(gam))],
                y=[z0, z0 + half * np.sin(np.deg2rad(gam))],
                mode="lines", showlegend=(sgn > 0),
                name=f"V panels (Γ = {gam:.0f}°, not modelled geometrically)",
                line=dict(color=WARN, width=2, dash="dot")))
    for tr in _front_fin_trace(geom):
        fig.add_trace(tr)
    lay = _base_layout(
        "Front view (looking upstream)" + _tail_height_note(geom)
        + (_MIRROR_TITLE_NOTE if _mirrored(geom) else ""),
        "span y [m]", _vlabel(geom), h=240)
    lay["yaxis"]["scaleanchor"] = "x"
    lay["showlegend"] = True
    _flip_if_mirrored(lay, geom)
    fig.update_layout(**lay)
    return fig


def fig_frontview(geom: dict):
    """Front view (looking upstream) of a nonplanar design: the solver's own
    (y, z) panel path with the winglet arc highlighted. This is the view the
    cant angle actually lives in. Falls back to the wing + tail view when the
    design is a tail layout, and returns None for a plain planar wing.

    A DEVICE-LESS CANTED WING keeps the fallback deliberately. It synthesises
    its wing from ``geometry["y"]``/``["z"]`` (:func:`_fig_frontview_tail`),
    so this view has always drawn the dihedral — and it is also where a
    V-tail's panels get folded and its unmodelled shape drawn dotted, which
    the path branch above does not do."""
    np_geom = _nonplanar(geom)
    if np_geom is None:
        return _fig_frontview_tail(geom)
    y, z, _c, mask = np_geom
    fig = go.Figure()
    for seg in np.split(np.arange(y.size),
                        np.flatnonzero(np.diff(mask.astype(int)) != 0) + 1):
        if seg.size < 2:
            continue
        is_wl = bool(mask[seg[0]])
        fig.add_trace(go.Scatter(
            x=y[seg], y=z[seg], mode="lines",
            name="winglet" if is_wl else "wing",
            showlegend=bool(is_wl and seg[0] == 0),
            line=dict(color=WARN if is_wl else ACCENT, width=4)))
    # THE REAR WING OF A TANDEM PAIR, at its own height and with its own tip
    # device. This is the view that showed the bug those two blocks exist to
    # fix: read as one array the front wing's starboard device ran straight
    # into the rear wing's port device and the pair was drawn with ONE winglet
    # spanning the aircraft. They are two devices, on two wings, and the
    # vertical gap between them is the stagger the user stated.
    second_sf = geom.get("second_surface")
    if second_sf:
        path = _front_path(second_sf)
        if path is not None:
            _add_front_path(fig, *path, "rear wing", GOOD)
    # a tail solved in the SAME system (wingtail.py) is part of this view:
    # its HEIGHT is a design variable there, and this is the view that height
    # lives in
    tail = geom.get("tail") or {}
    ts = geom.get("tail_surface")
    if tail and ts:
        # the tail's own path, tip device and all — folded into the V where
        # the layout is one. It is a designed surface with a cant angle of
        # its own, and this is the view both angles live in.
        path = cad.tail_path(tail, ts)
        if path is not None:
            yv, zv, _c, wl_t, _seg = path
            _add_front_path(fig, yv, zv, wl_t, "tail", GOOD)
        else:
            path = _front_path(ts)
            if path is not None:
                _add_front_path(fig, *path, "tail", GOOD)
    for tr in _front_fin_trace(geom):
        fig.add_trace(tr)
    wl = geom.get("winglet") or {}
    sub = (f"h = {wl.get('h_m', 0.0):.2f} m, cant {wl.get('cant_deg', 0.0):.0f}°"
           if wl else "")
    if tail and ts:
        sub += _tail_height_note(geom)
    lay = _base_layout(f"Front view (looking upstream) · {sub}"
                       + (_MIRROR_TITLE_NOTE if _mirrored(geom) else ""),
                       "span y [m]", _vlabel(geom), h=240)
    lay["yaxis"]["scaleanchor"] = "x"
    lay["showlegend"] = True
    _flip_if_mirrored(lay, geom)
    fig.update_layout(**lay)
    return fig


def _naca4_section(tc: float, m: float = 0.02, p: float = 0.4, n: int = 21):
    """NACA 4-digit outline — :func:`aerobo.cad.naca4_section`."""
    return cad.naca4_section(tc, m, p, n)


def _tc_for_view(geom: dict, x_best=None, labels=None) -> float:
    """Section t/c to draw — :func:`aerobo.cad.tc_for_view`."""
    return cad.tc_for_view(geom, x_best, labels)


def _loft_surface(y, c, twist_rad, xc, zc, x0=0.0, z0=0.0):
    """Planar span loft — :func:`aerobo.cad.loft_surface`."""
    return cad.loft_surface(y, c, twist_rad, xc, zc, x0=x0, z0=z0)


def _elevator_trace(y, c, x0, z0, frac, delta_deg, nrm=None):
    """The deflected-elevator plate as a plotly surface.

    Geometry from :func:`aerobo.cad.elevator_grid` — a schematic, deliberately:
    the physics models the elevator as a flap EFFECTIVENESS tau, never as a
    geometric hinge, so what the plate carries is the hinge station and the
    sign and size of the deflection the trim landed on.
    """
    X, Y, Z = cad.elevator_grid(y, c, x0, z0, frac, delta_deg, nrm=nrm)
    return go.Surface(
        x=X, y=Y, z=Z, surfacecolor=np.zeros_like(Z), showscale=False,
        colorscale=[[0, WARN], [1, WARN]], opacity=0.95,
        name="elevator", hoverinfo="skip",
        lighting=dict(ambient=0.7, diffuse=0.6, specular=0.1))


def _nonplanar(geom: dict):
    """(y, z, chord, is_winglet) for an out-of-plane solve, or None —
    :func:`aerobo.cad.nonplanar_arrays`."""
    return cad.nonplanar_arrays(geom)


def _wing_nonplanar(geom: dict):
    """The WING's own out-of-plane path: a tip device OR A DIHEDRAL.

    :func:`_nonplanar` asks "did a tip device fly", which is the question the
    SECOND-surface blocks still want (their ``z`` array is absolute and their
    height is applied again as ``z0``). The WING has a second way out of its
    plane now that the cant is a design variable, and asking the device
    question about it drew a design that flies 15 deg of dihedral as a flat
    wing — see :func:`aerobo.cad.canted_arrays`."""
    return cad.nonplanar_arrays(geom) or cad.canted_arrays(geom)


def _path_normals(y, z, mask, smooth: bool = False):
    """Unit (tangent, normal) along a spanwise path —
    :func:`aerobo.cad.path_normals`."""
    return cad.path_normals(y, z, mask, smooth=smooth)


def _loft_path(y, z, c, twist_rad, xc, zc, mask, smooth: bool = False,
               x0: float = 0.0):
    """Nonplanar span loft — :func:`aerobo.cad.loft_path`."""
    return cad.loft_path(y, z, c, twist_rad, xc, zc, mask, smooth=smooth,
                         x0=x0)


def _section_xy(geom: dict, x_best=None, labels=None, section=None):
    """(xc, zc) unit-chord section — :func:`aerobo.cad.section_path`."""
    return cad.section_path(geom, x_best, labels, section)


def _wing_twist(geom: dict, y, x_best=None, labels=None):
    """Linear twist law sampled at |y| — :func:`aerobo.cad.wing_twist`."""
    return cad.wing_twist(geom, y, x_best, labels)


def _v_tail_path(tail: dict, z0: float):
    """A V-tail's two panels, or None — :func:`aerobo.cad.v_tail_path`."""
    return cad.v_tail_path(tail, z0)


def _tail_traces(tail: dict, sf: dict, xc, zc, cl_source=None,
                 showscale: bool = False):
    """Surface traces for the tail that was flown.

    The GEOMETRY is :func:`aerobo.cad.tail_surfaces` — the panel(s) the solver
    put where it put them, plus the elevator plate — and this only colours it:
    the panel by local Cl where the solver reported one, the plate in the
    warning colour so a deflection reads at a glance.
    """
    out = []
    src_y, src_cl = (cl_source or (sf.get("y"), sf.get("Cl_y")))
    # a V-tail says so in the legend: it is TWO panels, and the reader has to
    # know the projection they are drawn from is the flat equivalent's span
    vee = cad.v_tail_path(tail, float(sf.get("z_offset", 0.0))) is not None
    for s in cad.tail_surfaces(tail, sf, xc, zc):
        if s.name == "elevator":
            out.append(go.Surface(
                x=s.X, y=s.Y, z=s.Z, surfacecolor=np.zeros_like(s.Z),
                showscale=False, colorscale=[[0, WARN], [1, WARN]],
                opacity=0.95, name="elevator", hoverinfo="skip",
                lighting=dict(ambient=0.7, diffuse=0.6, specular=0.1)))
            continue
        col = s.Z
        if src_cl is not None and src_y is not None \
                and len(src_cl) == len(src_y) and s.y is not None:
            sy = np.abs(np.asarray(src_y, float))
            order = np.argsort(sy)
            cl_i = np.interp(np.abs(s.y), sy[order],
                             np.asarray(src_cl, float)[order])
            col = np.repeat(cl_i[:, None], s.X.shape[1], axis=1)
        out.append(go.Surface(
            x=s.X, y=s.Y, z=s.Z, surfacecolor=col, colorscale="Viridis",
            showscale=showscale,
            colorbar=dict(title="local Cl", thickness=12, len=0.6,
                          tickfont=dict(color=MUTED)),
            opacity=1.0, name=("V-tail panels" if vee else s.name),
            lighting=dict(ambient=0.55, diffuse=0.7, specular=0.25,
                          roughness=0.55, fresnel=0.1)))
    return out


def _scene_axes(title3d: str, height: int = 440, camera=None,
                mirrored: bool = False) -> dict:
    """Common 3-D layout: package axes (x downstream +), true aspect.

    ``mirrored`` turns the PICTURE over for a design solved in the car's
    frame (:func:`_mirrored`) by drawing the z axis downward, which is the
    same move :func:`fig_planform` makes to put the nose up: the geometry is
    untouched, so a hover still reads the model's own +z — the car's downward
    direction, and the frame every height, clearance and margin in the
    breakdown is quoted in — while the wing draws ON TOP of the endplates
    that reach for the deck.

    NOT the camera's ``up`` vector, which is the obvious way to do it and
    does not work: plotly's default 3-D drag mode is the turntable, which
    holds the z axis vertical on screen and quietly discards an inverted up.
    Reversing the AXIS survives that, and survives the V3 restyler's own
    ``scene`` update on top of it.
    """
    return dict(
        paper_bgcolor="rgba(0,0,0,0)", font=dict(color=MUTED, size=10),
        height=height, margin=dict(l=0, r=0, t=30, b=0),
        title=dict(text=title3d + (_MIRROR_TITLE_NOTE if mirrored else ""),
                   font=dict(size=14, color=MUTED), x=0.02),
        scene=dict(
            aspectmode="data",
            xaxis=dict(title="x [m] · downstream +", gridcolor=GRID,
                       backgroundcolor="rgba(0,0,0,0)"),
            yaxis=dict(title="span y [m]", gridcolor=GRID,
                       backgroundcolor="rgba(0,0,0,0)"),
            zaxis=dict(title=("z [m] · the car's DOWN + (drawn downward)"
                              if mirrored else "z [m] · up +"),
                       gridcolor=GRID,
                       backgroundcolor="rgba(0,0,0,0)",
                       **({"autorange": "reversed"} if mirrored else {})),
            camera=camera or dict(eye=dict(x=-1.5, y=-1.5, z=0.9))))


def _mount_traces(geom: dict, x_best=None, labels=None) -> list:
    """The MOUNT's own bodies, for the 3-D view — struts, where there are any.

    Drawn from ``cad.mount_surfaces``, i.e. from the same geometry an export
    would use, so the picture and any file written from it cannot disagree
    about where the wing is held. An endplate-borne mount contributes nothing:
    its load path is the plate, and the plate is already the wing's outboard
    panels.
    """
    import plotly.graph_objects as go

    out = []
    for sf in cad.mount_surfaces(geom, x_best, labels):
        out.append(go.Surface(
            x=sf.X, y=sf.Y, z=sf.Z, showscale=False, name=sf.name,
            colorscale=[[0.0, MUTED], [1.0, MUTED]], opacity=1.0,
            hoverinfo="skip",
            lighting=dict(ambient=0.5, diffuse=0.8, specular=0.15,
                          roughness=0.7, fresnel=0.05)))
    return out


def _mount_caption(geom: dict) -> str:
    """One clause for the 3-D title, where a mount is drawn.

    States the two numbers that decide the picture — how many wetted members
    are charged, and where along the CHORD they hold the wing — because the
    second is invisible at a glance and is what the divergence sense turns on.
    """
    m = geom.get("mount") or {}
    if str(m.get("kind")) != "pylon" or not (m.get("stations_m") or ()):
        return ""
    n_st = len(m["stations_m"])
    frac = float(m.get("x_attach_frac", 0.25))
    side = "swan neck" if m.get("side") == "pressure" else "underslung"
    return (f"  ·  {side}: {int(m.get('n_pylons', 0))} strut(s) at "
            f"{n_st} station(s), {float(m.get('length_m', 0.0)):.3f} m to the "
            f"deck, holding at x/c {frac:.2f}")


def _vertical_label(geom: dict) -> str:
    """`` + fin`` / `` + strut`` / ``""`` — what to CALL the vertical surface.

    A water craft's vertical is its mast, and captioning it "fin" is the
    same defect one layer up from the one ``fin.FinGeometry.kind`` exists
    for: the picture would name a surface the report does not.
    """
    blk = (geom or {}).get("fin")
    if not blk:
        return ""
    return (" + strut" if str(blk.get("kind", "fin")) == "mast"
            else " + fin")


def _fin_traces(geom: dict, xc, zc) -> list:
    """The VERTICAL surface as a 3-D trace, or [] when the design has none.

    Lofted by :func:`aerobo.cad.fin_surface` — the SAME function the STL and
    the OpenVSP script use — rather than by a second loft here. This view and
    ``cad.surfaces`` are already two independent drawers of the same
    aeroplane, which is how the fin came to be exported and not drawn; giving
    the fin its own copy would have made three.

    A V-tail returns [], because the report carries no fin block for one.
    """
    out = []
    for s in cad.fin_surface(geom, xc, zc):
        out.append(go.Surface(
            x=s.X, y=s.Y, z=s.Z, surfacecolor=s.Z,
            colorscale="Greys", showscale=False, opacity=1.0, name=s.name,
            lighting=dict(ambient=0.55, diffuse=0.7, specular=0.25,
                          roughness=0.55, fresnel=0.1)))
    return out


def fig_wing3d(geom: dict, x_best=None, labels=None, section=None):
    """3-D wing with REAL section thickness: the optimised CST section where
    the design carries one (else a NACA 4-digit section at the solved t/c)
    lofted along the span, rotated by the local linear twist about c/4,
    coloured by local Cl where the solver provided it.

    Drawn in the package's axes (x downstream +, z up), so an aft tail is
    aft, a canard is ahead, and every section's nose points into the flow.

    Nonplanar designs (the VLM winglet modes) are lofted along the solver's
    own (y, z) panel path, so the winglet appears at its true height and
    cant instead of the flat span extension a y-only view would imply — and
    a BLENDED transition is lofted without an artificial crease at the
    junction. A tail in the same solve (wingtail.py) is drawn with them.
    Tandem surfaces render at their stagger offsets. Presentation only.

    A dihedral leaves the wing plane exactly as a tip device does, so the gate
    is the WING's (:func:`_wing_nonplanar`) and the loft carries the design's
    quarter-chord sweep (:func:`aerobo.cad.sweep_offset`) — the two rows the
    free-cant families search, neither of which used to reach any picture."""
    np_geom = _wing_nonplanar(geom)
    if np_geom is not None:
        y, z, c, mask = np_geom
        xc, zc = _section_xy(geom, x_best, labels, section)
        twist = _wing_twist(geom, y, x_best, labels)
        # a blended junction has no crease — see _path_normals. Either side
        # of the transition counts: a wing-side blend leaves the wing plane
        # before the tip, so segmenting the normals at the junction would
        # draw a kink that geometry.span_path did not put there.
        wl_info = geom.get("winglet") or {}
        smooth = (float(wl_info.get("blend_frac", 0.0) or 0.0) > 0.0
                  or float(wl_info.get("wing_blend_frac", 0.0) or 0.0) > 0.0)
        X, Y, Z = _loft_path(y, z, c, twist, xc, zc, mask, smooth=smooth,
                             x0=cad.sweep_offset(geom, y))
        cl = geom.get("Cl_y")
        col = (np.repeat(np.asarray(cl, float)[:, None], xc.size, axis=1)
               if cl is not None and len(cl) == y.size else Z)
        traces = [go.Surface(
            x=X, y=Y, z=Z, surfacecolor=col,
            colorscale="Viridis" if cl is not None else "Blues",
            colorbar=dict(title="local Cl" if cl is not None else "z",
                          thickness=12, len=0.6, tickfont=dict(color=MUTED)),
            showscale=True, opacity=1.0,
            lighting=dict(ambient=0.55, diffuse=0.7, specular=0.25,
                          roughness=0.55, fresnel=0.1),
            # the key light stands at the model's +z, which on a MIRRORED
            # design is the car's UNDERSIDE and, once the axis is drawn
            # downward, is below the picture: the wing would be lit from
            # under the track. Its sign follows the frame, so the surface
            # facing the reader is the lit one either way.
            lightposition=dict(x=0, y=0,
                               z=-2 if _mirrored(geom) else 2))]
        # outline the winglet arc so its height/cant read at a glance
        for seg in np.split(np.arange(y.size),
                            np.flatnonzero(np.diff(mask.astype(int)) != 0) + 1):
            if seg.size < 2 or not mask[seg[0]]:
                continue
            traces.append(go.Scatter3d(
                x=np.zeros(seg.size), y=y[seg], z=z[seg], mode="lines",
                line=dict(color=WARN, width=5), showlegend=False,
                hoverinfo="skip"))
        # ...and the pair's REAR WING, lofted along its own path (its own
        # tip device included) at its own stagger — cad.second_wing_surfaces
        second_sf = geom.get("second_surface")
        if second_sf:
            for s in cad.second_wing_surfaces(geom, second_sf, xc, zc,
                                              x_best, labels):
                traces.append(go.Surface(
                    x=s.X, y=s.Y, z=s.Z, surfacecolor=s.Z,
                    colorscale="Greens", showscale=False, opacity=1.0,
                    name=s.name,
                    lighting=dict(ambient=0.55, diffuse=0.7, specular=0.25,
                                  roughness=0.55, fresnel=0.1)))
        traces += _fin_traces(geom, xc, zc)
        traces += _mount_traces(geom, x_best, labels)
        wl = geom.get("winglet") or {}
        sub = _mount_caption(geom)
        if wl:
            sub += (f"  ·  winglet h = {wl.get('h_m', 0.0):.2f} m, "
                    f"cant {wl.get('cant_deg', 0.0):.0f}°")
            if smooth:
                sub += (f", {wl.get('blend_frac', 0.0):.0%} blended"
                        + (f" + {wl['wing_blend_frac']:.0%} of the semi-span "
                           f"on the wing"
                           if float(wl.get("wing_blend_frac", 0.0) or 0.0) > 0
                           else "")
                        + (f" ({geom.get('winglet_blend_shape')})"
                           if geom.get("winglet_blend_shape") else ""))
        # a tail solved WITH the wing (wingtail.py) is drawn with it
        tail = geom.get("tail") or {}
        tail_sf = geom.get("tail_surface")
        if tail and tail_sf:
            traces += _tail_traces(tail, tail_sf, xc, zc)
            ctrl = tail.get("control", "stabilator")
            sub += (f"  ·  {tail.get('type', 'tail')}, arm "
                    f"{float(tail.get('l_t', 0.0)):+.2f} m, "
                    + (f"δe {float(tail.get('delta_e_deg', 0.0)):+.1f}°"
                       if ctrl == "elevator"
                       else f"i_t {float(tail.get('i_t_deg', 0.0)):+.1f}°"))
        if second_sf:
            sub += (f"  ·  rear wing at "
                    f"{float(second_sf.get('x_offset', 0.0)):+.2f} m aft, "
                    f"{float(second_sf.get('z_offset', 0.0)):+.2f} m up")
        fig = go.Figure(traces)
        head = ("Wing + winglet" + (" + rear wing" if second_sf else "")
                + (" + tail" if tail_sf else "")
                + _vertical_label(geom)
                + (" + mount" if _mount_caption(geom) else ""))
        fig.update_layout(**_scene_axes(f"{head} (3-D, as flown){sub}",
                                mirrored=_mirrored(geom)))
        return fig
    surfs = geom.get("surfaces")
    if surfs:
        xc, zc = _section_xy(geom, x_best, labels, section)
        tail = geom.get("tail") or {}
        traces = []
        for i, sf in enumerate(surfs):
            if sf.get("name") == "tail" and tail:
                # the tail is drawn by its own helper: a V-tail is TWO
                # dihedralled panels, and an elevator is a plate on top of
                # whichever panel shape that is
                traces += _tail_traces(tail, sf, xc, zc, showscale=(i == 0))
                continue
            y, c = _planform_arrays({"y": sf.get("y"),
                                     "chord": sf.get("chord")})
            if y is None:
                continue
            x0 = float(sf.get("x_offset", 0.0))
            z0 = float(sf.get("z_offset", 0.0))
            tw_s = _wing_twist(geom, y, x_best, labels)
            X, Y, Z = _loft_surface(y, c, tw_s, xc, zc,
                                    x0=x0 + cad.sweep_offset(geom, y), z0=z0)
            cl = sf.get("Cl_y")
            col = Z
            if cl and len(cl) * 2 >= y.size:
                cl_i = np.interp(np.abs(y),
                                 np.sort(np.abs(np.asarray(sf["y"], float))),
                                 np.asarray(cl, float)[np.argsort(
                                     np.abs(np.asarray(sf["y"], float)))])
                col = np.repeat(cl_i[:, None], xc.size, axis=1)
            traces.append(go.Surface(
                x=X, y=Y, z=Z, surfacecolor=col, colorscale="Viridis",
                showscale=(i == 0),
                colorbar=dict(title="local Cl", thickness=12, len=0.6,
                              tickfont=dict(color=MUTED)),
                opacity=1.0,
                lighting=dict(ambient=0.55, diffuse=0.7, specular=0.25,
                              roughness=0.55, fresnel=0.1)))
        traces += _fin_traces(geom, xc, zc)
        if not traces:
            return None
        if tail:
            ctrl = tail.get("control", "stabilator")
            defl = (f"δe {float(tail.get('delta_e_deg', 0.0)):+.1f}°"
                    if ctrl == "elevator"
                    else f"i_t {float(tail.get('i_t_deg', 0.0)):+.1f}°")
            title3d = (f"Wing + {tail.get('type', 'tail')}"
                       f"{_vertical_label(geom)} (3-D, arm to "
                       f"scale) · {ctrl} · {defl}")
        else:
            title3d = ("Tandem system" + _vertical_label(geom)
                       + " (3-D, stagger to scale)")
        fig = go.Figure(traces)
        fig.update_layout(**_scene_axes(title3d, height=420,
                                mirrored=_mirrored(geom)))
        return fig
    y, c = _planform_arrays(geom, x_best, labels)
    if y is None:
        return None
    tw_r = geom.get("twist_root_deg")
    tw_t = geom.get("twist_tip_deg")
    if (tw_r is None or tw_t is None) and x_best is not None and labels:
        lab = list(labels)
        if "twist_root_deg" in lab:
            tw_r = float(x_best[lab.index("twist_root_deg")])
        if "twist_tip_deg" in lab:
            tw_t = float(x_best[lab.index("twist_tip_deg")])
    tw_r = float(tw_r or 0.0)
    tw_t = float(tw_t or 0.0)
    b = float(np.max(np.abs(y)) * 2.0) or 1.0
    eta = np.abs(2.0 * y / b)
    twist = np.deg2rad(tw_r + (tw_t - tw_r) * eta)

    xc, zc = _section_xy(geom, x_best, labels, section)
    X, Y, Z = _loft_surface(y, c, twist, xc, zc,
                            x0=cad.sweep_offset(geom, y))

    cl = geom.get("Cl_y")
    if cl and len(cl) != y.size and len(cl) * 2 >= y.size:
        cl_half = np.asarray(cl, float)           # solver gave a half/full span
        cl = np.interp(np.abs(y), np.sort(np.abs(np.asarray(geom["y"], float))),
                       cl_half[np.argsort(np.abs(np.asarray(geom["y"], float)))])
    surf_color = (np.repeat(np.asarray(cl, float)[:, None], xc.size, axis=1)
                  if cl is not None and len(cl) == y.size else Z)
    traces = [go.Surface(
        x=X, y=Y, z=Z, surfacecolor=surf_color,
        colorscale="Blues" if cl is None else "Viridis",
        colorbar=dict(title="local Cl" if cl is not None else "z",
                      thickness=12, len=0.6, tickfont=dict(color=MUTED)),
        showscale=True, opacity=1.0,
        lighting=dict(ambient=0.55, diffuse=0.7, specular=0.25,
                      roughness=0.55, fresnel=0.1),
        # the key light follows the frame — see the twin above: at the
        # model's +z it would light a mirrored design from under the track
        lightposition=dict(x=0, y=0, z=-2 if _mirrored(geom) else 2))]
    # ...and a second surface solved in the SAME system, when the main one is
    # PLANAR. The two branches above draw it (a nonplanar wing, and the
    # published two-surface export); this one is the case a hydrofoil with an
    # elevator but no tip device on the foil takes, and it was drawing the
    # foil alone — the whole point of that family being the second surface.
    #
    # ...AND THE VERTICAL SURFACE, which the other two branches have drawn
    # all along. A PLAIN hydrofoil is planar and carries no listed surfaces,
    # so it lands here — and here was the one branch of three that never
    # called ``_fin_traces``. The strut was reported, charged, lofted into
    # the STL and flown by the rebuild, and was missing from the one picture
    # a user looks at. (The winglet twin drew it, off the branch above, so
    # the same craft had its mast in one view and not the other.)
    traces += _fin_traces(geom, xc, zc)
    traces += _mount_traces(geom, x_best, labels)
    title3d = ("Wing" + _vertical_label(geom)
               + " (3-D, real section thickness)" + _mount_caption(geom))
    tail_p, tail_sf_p = geom.get("tail") or {}, geom.get("tail_surface")
    if tail_p and tail_sf_p:
        traces += _tail_traces(tail_p, tail_sf_p, xc, zc)
        ctrl = tail_p.get("control", "stabilator")
        title3d = (f"Wing + {tail_p.get('type', 'tail')}"
                   f"{_vertical_label(geom)} (3-D, arm to scale) "
                   f"· arm {float(tail_p.get('l_t', 0.0)):+.2f} m · "
                   + (f"δe {float(tail_p.get('delta_e_deg', 0.0)):+.1f}°"
                      if ctrl == "elevator"
                      else f"i_t {float(tail_p.get('i_t_deg', 0.0)):+.1f}°"))
    fig = go.Figure(traces)
    fig.update_layout(**_scene_axes(title3d, height=400,
                                mirrored=_mirrored(geom)))
    return fig


def _spanwise_surface_traces(sf: dict, label: str, colr: str,
                             dev_label: str | None = None,
                             dev_colr: str = WARN):
    """(traces, main-panel mask) for ONE surface's spanwise loading.

    Tip-device panels are NOT spanwise stations: on a vertical device every
    one of them sits at the same y, so charting them against y glues a spike
    to the tip. They come out as their own series, plotted at the tip plus
    the arc height — measured from THIS surface's own plane, because the
    hydrofoil's elevator hangs below the foil and an arc height measured
    from z = 0 would put its device inboard of its own tip.

    Shared by the wing and by a second surface solved in the SAME panel
    array (``geom["tail_surface"]``), which reports its loading exactly the
    way the wing reports the wing's.
    """
    y = np.asarray(sf.get("y") or [], dtype=float)
    cl = np.asarray(sf.get("Cl_y") or [], dtype=float)
    if y.size < 2 or cl.size != y.size:
        return [], None
    wl = sf.get("is_winglet")
    mask = (np.asarray(wl, dtype=bool) if wl is not None and len(wl) == y.size
            else np.zeros(y.size, dtype=bool))
    main = ~mask
    if not main.any():
        return [], None
    traces = [go.Scatter(x=y[main], y=cl[main], mode="lines+markers",
                         name=f"{label} — local Cl",
                         line=dict(color=colr, width=2),
                         marker=dict(size=4))]
    if mask.any():
        dev = dev_label or f"{label} tip device"
        z = sf.get("z")
        z_arr = (np.asarray(z, dtype=float)
                 if z is not None and len(z) == y.size
                 else np.zeros(y.size))
        arc = np.abs(z_arr - float(sf.get("z_offset", 0.0)))
        tip = float(np.abs(y[main]).max())
        for side, tag in ((y > 0, "stbd"), (y < 0, "port")):
            sel = mask & side
            if sel.any():
                traces.append(go.Scatter(
                    x=np.sign(y[sel]) * (tip + arc[sel]),
                    y=cl[sel], mode="markers", name=f"{dev} {tag}",
                    marker=dict(size=5, color=dev_colr, symbol="diamond"),
                    hovertemplate=f"{dev} Cl %{{y:.3f}}<extra></extra>"))
    return traces, main


def fig_sections_as_flown(geom: dict, x_best=None, labels=None, section=None,
                          section_aft=None, breakdown: dict | None = None):
    """Every surface's own section, drawn the way it is MOUNTED.

    The 3-D view is true-scale (``aspectmode="data"``), which is right for
    reading span, arm and dihedral and useless for reading camber: a tail
    0.08 m thick on a 10 m model is under 1 % of the frame, so "which way up
    is this section" — the thing a downloading surface makes a real question
    (:func:`aerobo.tail.orient_section`) — is literally invisible there.

    So this draws each surface at ITS own chord: outlines normalised on x/c
    and z/c, overlaid, with the chord line at zero. A wing arching up beside
    a stabiliser arching down is then one glance rather than a measurement.

    The outlines come from ``cad.surfaces`` — the SAME loft that is written
    to STL and handed to OpenVSP — so this view cannot drift from what was
    exported; if the drawing says camber-down, the .stl does too.

    ...with ONE stated exception: a design solved in the car's mirrored
    frame is DRAWN the car's way up (:func:`_flip_if_mirrored`), because a
    section that makes downforce arches downward on the car and upward in
    the model. The exported model stays in the frame it was solved in, which
    is the frame its numbers are in, so on a car the .stl is this picture
    reflected in the chord line — the axis title says which way it is drawn.
    """
    from aerobo import cad

    try:
        surfs = cad.surfaces(geom, x_best, labels, section, section_aft)
    except Exception:
        return None
    if not surfs:
        return None
    bd = breakdown or {}
    #: what each lofted surface carries, so the picture and the sign agree
    carries = {"wing": bd.get("CL_w", bd.get("CL_foil", bd.get("CL"))),
               "tail": bd.get("CL_t", bd.get("CL_stab")),
               "rear": bd.get("CL_rear")}
    colours = [ACCENT, WARN, GOOD, MUTED]
    fig = go.Figure()
    drawn = 0
    for i, s in enumerate(surfs):
        if s.name == "elevator":            # a flat plate has no section
            continue
        if getattr(s, "kind", "lifting") != "lifting":
            # STRUCTURE, not a section the wing flew. The mount has a real
            # aerofoil shape and would draw perfectly well here — which is
            # exactly the problem: this view asks "which way up is each
            # surface the solver flew", and a strut answers a question nobody
            # asked, normalised on its own chord as though the wing had flown
            # it. Filtered on the CONTRACT (Surface.kind) rather than on the
            # name, because the "elevator" line above is what filtering by
            # name looks like after the second part arrives.
            continue
        j = s.X.shape[0] // 2               # mid-span, where the loft is clean
        xs, zs = np.asarray(s.X[j], float), np.asarray(s.Z[j], float)
        c = float(xs.max() - xs.min())
        if not np.isfinite(c) or c <= 0.0:
            continue
        xc = (xs - xs.min()) / c
        zc = (zs - zs.mean()) / c
        cl = carries.get(s.name)
        way = ("" if cl is None else
               f" · carries CL {float(cl):+.3f}"
               + (" (DOWN)" if float(cl) < 0.0 else " (up)"))
        fig.add_trace(go.Scatter(
            x=xc, y=zc, mode="lines", fill="toself",
            name=f"{s.name}{way}",
            line=dict(color=colours[i % len(colours)], width=2),
            opacity=0.75,
            hovertemplate=f"{s.name}: x/c %{{x:.3f}}, z/c %{{y:.3f}}"
                          f"<extra></extra>"))
        drawn += 1
    if not drawn:
        return None
    fig.add_hline(y=0.0, line=dict(color=GRID, width=1, dash="dash"))
    lay = _base_layout("Sections as flown — each on its own chord, mounted "
                       "the way it is drawn and exported"
                       + (_MIRROR_TITLE_NOTE if _mirrored(geom) else ""),
                       "x/c", _vlabel(geom, "z/c"), h=300)
    lay["yaxis"]["scaleanchor"] = "x"
    # ...and for a CAR, drawn the way the section meets the air on the car:
    # the model's suction side is the car's underside, so a section drawn in
    # the model frame arches the wrong way and reads as an aeroplane's.
    _flip_if_mirrored(lay, geom)
    fig.update_layout(**lay)
    return fig


def fig_spanwise(geom: dict):
    surfs = geom.get("surfaces")
    if surfs:
        fig = go.Figure()
        colors = [ACCENT, WARN]
        for i, sf in enumerate(surfs):
            y, cl = sf.get("y"), sf.get("Cl_y")
            if y and cl and len(y) == len(cl):
                fig.add_trace(go.Scatter(
                    x=y, y=cl, mode="lines+markers",
                    name=f"local Cl — {sf.get('name', i)}",
                    line=dict(color=colors[i % 2], width=2),
                    marker=dict(size=4)))
        if not fig.data:
            return None
        fig.update_layout(**_base_layout(
            "Spanwise distributions (front vs rear)", "span y [m]",
            "local Cl", h=300))
        return fig
    y = geom.get("y")
    cl = geom.get("Cl_y")
    if not y or not cl or len(y) != len(cl):
        return None
    y = np.asarray(y, float)
    # the wing, with its winglet panels split out of the curve (they are arc
    # stations, not spanwise ones — see the helper)
    fig = go.Figure()
    traces, main = _spanwise_surface_traces(geom, "wing", ACCENT,
                                            dev_label="winglet")
    if main is None:
        return None
    for tr in traces:
        fig.add_trace(tr)
    mask = ~main
    # ...and a SECOND surface solved in the same panel array. Only the
    # published lifting-line pair exports a two-entry ``surfaces`` list (the
    # branch above); every family that flies its second surface with the
    # wing — the hydrofoil's elevator, an air tail beside a winglet —
    # reports it as ``tail_surface``, and reading the wing's arrays alone
    # drew a two-surface craft as if it had one. Same two-branch shape that
    # dropped it from the plan, front and 3-D views.
    ts = geom.get("tail_surface") or geom.get("second_surface") or {}
    ts_traces, ts_main = _spanwise_surface_traces(
        ts, str(ts.get("name") or "tail"), GOOD)
    for tr in ts_traces:
        fig.add_trace(tr)
    ae = geom.get("alpha_eff_deg")
    notes = []
    if ts_main is not None:
        notes.append(f"wing + {ts.get('name') or 'tail'}")
    if mask.any() or (ts_main is not None and not ts_main.all()):
        notes.append("tip devices plotted at tip + arc height")
    lay = _base_layout(
        "Spanwise distributions"
        + (f" ({' · '.join(notes)})" if notes else ""),
        "span y [m]", "local Cl", h=300)
    if ae is not None and len(ae) == y.size:
        ae = np.asarray(ae, float)
        fig.add_trace(go.Scatter(x=y[main], y=ae[main], mode="lines",
                                 name="α_eff [deg]", yaxis="y2",
                                 line=dict(color=WARN, width=1.6, dash="dot")))
        lay["yaxis2"] = dict(title="α_eff [deg]", overlaying="y", side="right",
                             gridcolor="rgba(0,0,0,0)")
    fig.update_layout(**lay)
    return fig


def fig_airfoil(coords) -> go.Figure | None:
    if not coords:
        return None
    pts = np.asarray(coords, float)
    fig = go.Figure(go.Scatter(
        x=pts[:, 0], y=pts[:, 1], mode="lines", fill="toself",
        line=dict(color=ACCENT, width=2), fillcolor=BAND))
    lay = _base_layout("Airfoil section (CST)", "x/c", "y/c", h=240)
    lay["yaxis"]["scaleanchor"] = "x"
    lay["showlegend"] = False
    fig.update_layout(**lay)
    return fig


# ------------------------------------------------------- airfoil result view

def fig_section_shape(sr: dict) -> go.Figure | None:
    """Optimised section over its anchor, with the t/c constraint drawn.

    Overlaying the anchor is the whole point: on its own an aerofoil outline
    says nothing about what the optimiser DID.
    """
    des = (sr or {}).get("design")
    if not des or not des.get("coords"):
        return None
    fig = go.Figure()
    base = sr.get("baseline")
    if base and base.get("coords"):
        b = np.asarray(base["coords"], float)
        fig.add_trace(go.Scatter(
            x=b[:, 0], y=b[:, 1], mode="lines",
            name=f"{base.get('name', 'anchor')} · t/c {base['tc']:.4f}",
            line=dict(color=MUTED, width=1.4, dash="dot")))
    p = np.asarray(des["coords"], float)
    fig.add_trace(go.Scatter(
        x=p[:, 0], y=p[:, 1], mode="lines", fill="toself",
        name=f"optimised · t/c {des['tc']:.4f}",
        line=dict(color=ACCENT, width=2), fillcolor=BAND))
    xtc = float(des.get("tc_max_xc", 0.3))
    fig.add_vline(x=xtc, line=dict(color=WARN, width=1, dash="dash"))
    fig.add_annotation(x=xtc, y=0.0, text=f"max t/c @ {xtc:.0%}",
                       showarrow=False, yshift=-28,
                       font=dict(size=9, color=MUTED))
    lay = _base_layout("Section shape (CST) vs anchor", "x/c", "y/c", h=280)
    lay["yaxis"]["scaleanchor"] = "x"
    lay["showlegend"] = True
    fig.update_layout(**lay)
    return fig


def _branch_split(pol: dict):
    """(used, unused) index masks for the pre-stall monotone branch."""
    n = len(pol.get("alpha_deg") or [])
    used = np.zeros(n, dtype=bool)
    br = pol.get("branch")
    if br and len(br) == 2:
        used[int(br[0]):int(br[1])] = True
    else:
        used[:] = True
    return used, ~used


def fig_section_polars(sr: dict) -> go.Figure | None:
    """cl(α), the drag polar and cm(cl) for the optimised section.

    Points OUTSIDE the pre-stall monotone branch are drawn hollow: the
    reported cd/cm at the design lift are interpolated on that branch only,
    and a single confident line would hide which points the numbers came
    from.
    """
    des = (sr or {}).get("design") or {}
    pol = des.get("polar")
    if not pol or not pol.get("alpha_deg"):
        return None
    a = np.asarray(pol["alpha_deg"], float)
    cl = np.asarray(pol["cl"], float)
    cd = np.asarray(pol["cd"], float)
    cm = np.asarray(pol["cm"], float)
    used, unused = _branch_split(pol)
    cl_d = float(sr.get("cl_design", 0.5))

    fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.08,
                        subplot_titles=("lift curve", "drag polar",
                                        "pitching moment"))
    base = (sr.get("baseline") or {}).get("polar")

    def add(col, x, y, name, colour=ACCENT, dash=None, width=2):
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name,
                                 legendgroup=name, showlegend=(col == 1),
                                 line=dict(color=colour, width=width,
                                           dash=dash)),
                      row=1, col=col)

    if base and base.get("alpha_deg"):
        ba = np.asarray(base["alpha_deg"], float)
        bcl = np.asarray(base["cl"], float)
        bu, _ = _branch_split(base)
        add(1, ba[bu], bcl[bu], "anchor", MUTED, "dot", 1.4)
        add(2, np.asarray(base["cd"], float)[bu], bcl[bu], "anchor", MUTED,
            "dot", 1.4)
        add(3, bcl[bu], np.asarray(base["cm"], float)[bu], "anchor", MUTED,
            "dot", 1.4)

    add(1, a[used], cl[used], "optimised")
    add(2, cd[used], cl[used], "optimised")
    add(3, cl[used], cm[used], "optimised")
    if unused.any():
        for col, (x, y) in enumerate(((a[unused], cl[unused]),
                                      (cd[unused], cl[unused]),
                                      (cl[unused], cm[unused])), start=1):
            fig.add_trace(go.Scatter(
                x=x, y=y, mode="markers", name="off-branch (not used)",
                legendgroup="off", showlegend=(col == 1),
                marker=dict(size=5, color="rgba(0,0,0,0)",
                            line=dict(color=WARN, width=1))),
                row=1, col=col)

    # design lift + the two constraint limits
    for col in (1, 2):
        fig.add_hline(y=cl_d, line=dict(color=WARN, width=1, dash="dash"),
                      row=1, col=col)
    cm_max = sr.get("cm_max")
    if cm_max is not None:
        for sgn in (-1.0, 1.0):
            fig.add_hline(y=sgn * float(cm_max),
                          line=dict(color=WARN, width=1, dash="dot"),
                          row=1, col=3)
    fig.add_vline(x=cl_d, line=dict(color=WARN, width=1, dash="dash"),
                  row=1, col=3)

    lay = _base_layout("", "", "", h=300)
    lay.pop("title", None)
    lay["showlegend"] = True
    fig.update_layout(**lay)
    fig.update_xaxes(title_text="α [deg]", row=1, col=1, gridcolor=GRID)
    fig.update_yaxes(title_text="c_l", row=1, col=1, gridcolor=GRID)
    fig.update_xaxes(title_text="c_d", row=1, col=2, gridcolor=GRID)
    fig.update_yaxes(title_text="c_l", row=1, col=2, gridcolor=GRID)
    fig.update_xaxes(title_text="c_l", row=1, col=3, gridcolor=GRID)
    fig.update_yaxes(title_text="c_m", row=1, col=3, gridcolor=GRID)
    for ann in fig.layout.annotations:
        ann.font = dict(size=11, color=MUTED)
    return fig


# --------------------------------------------------------- full-results data

def _num(v, scale: float = 1.0):
    """``v`` as a finite float times ``scale``, or None. One place, because a
    row is now kept on a half-populated comparison and each side has to be
    judged on its own."""
    if v is None:
        return None
    try:
        x = float(v) * scale
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _cmp_row(label: str, old, new, nd: int = 3, scale: float = 1.0,
             lower_better: bool | None = None) -> dict | None:
    """One original-vs-optimised table row, or None when NEITHER side has a
    number.

    A metric measured on one side only is SHOWN, with an em dash where it was
    not measured and an empty change: dropping it is how a comparison table
    ends up quietly shorter than the objective it is reporting on, which is
    exactly what a 2-D section run used to look like. Both sides missing is a
    different statement — the quantity does not exist for this run at all (a
    wing L/D on a section run) — and that row still goes.

    ``lower_better`` None means the quantity has no preferred direction — the
    delta is still shown, just not judged.

    ``dir`` is the VERDICT, and it has four values a reader can be shown in
    four colours: ``"better"``, ``"worse"``, ``"same"`` (judged, and the two
    sides are equal to the printed precision) and ``""`` (not judged at all —
    no preferred direction, or one side never measured). ``"same"`` and ``""``
    are kept apart because "this did not move" and "nobody can say whether
    this moved in a good direction" are different sentences, and a card that
    paints them the same grey is still allowed to.
    """
    o, n = _num(old, scale), _num(new, scale)
    if o is None and n is None:
        return None
    if o is None or n is None:
        return {"metric": label,
                "original": "—" if o is None else f"{o:.{nd}f}",
                "new": "—" if n is None else f"{n:.{nd}f}",
                "change": "not measured on both sides", "dir": ""}
    d = n - o
    pct = (100.0 * d / abs(o)) if abs(o) > 1e-12 else None
    direction = ""
    if lower_better is not None:
        if abs(d) <= 1e-12:
            direction = "same"
        else:
            improved = (d < 0.0) if lower_better else (d > 0.0)
            direction = "better" if improved else "worse"
    change = f"{d:+.{nd}f}" + (f"  ({pct:+.1f} %)" if pct is not None else "")
    return {"metric": label, "original": f"{o:.{nd}f}",
            "new": f"{n:.{nd}f}", "change": change, "dir": direction}


def airfoil_compare_rows(rep: dict, score: dict | None = None) -> list[dict]:
    """Original vs optimised, as table rows.

    ORIGINAL is the run's own starting point: the library winner's CST refit
    (the design-box anchor) flown UNTWISTED at the same derived operating
    point. That makes every row a like-for-like difference attributable to
    the optimisation, not to a change of flight condition.

    ``score`` (an :func:`api.score_optimised_section` payload, optional) fills
    the SCREENING criteria on a run that did not compute them itself. Only a
    composite run evaluates the wide stall sweep, so on a ``-cd`` or wing run
    the breakdown carries no cl_max, stall angle or either L/D — and this table
    used to answer "how did the section change?" with four rows while the card
    beside it listed six criteria. The scoring block measures all six for any
    objective; handing them here puts one comparison on the card instead of two
    that cover different metrics.
    """
    base = (rep.get("baseline") or {}).get("breakdown") or {}
    opt = (rep.get("design") or {}).get("breakdown") or {}
    sec = rep.get("section") or {}
    b_sec = (sec.get("baseline") or {})
    o_sec = (sec.get("design") or {})
    b_pol = b_sec.get("polar") or {}
    o_pol = o_sec.get("polar") or {}
    #: the scoring block's own measurement of the six criteria, per side
    s_seed = ((score or {}).get("seed") or {}).get("metrics") or {}
    s_opt = ((score or {}).get("optimised") or {}).get("metrics") or {}

    def crit(key: str, side: dict, scored: dict, scored_key: str | None = None):
        """A criterion off the run's own breakdown, else off the score block."""
        got = side.get(key)
        if got is None:
            got = scored.get(scored_key or key)
        return got
    rows = [
        _cmp_row("wing L/D", base.get("LD"), opt.get("LD"), nd=2,
                 lower_better=False),
        _cmp_row("CD total [counts]", base.get("CD"), opt.get("CD"), nd=1,
                 scale=1e4, lower_better=True),
        _cmp_row("CD induced [counts]", base.get("CDi"), opt.get("CDi"),
                 nd=1, scale=1e4, lower_better=True),
        _cmp_row("CD profile [counts]", base.get("CDp"), opt.get("CDp"),
                 nd=1, scale=1e4, lower_better=True),
        _cmp_row("span efficiency e", base.get("e"), opt.get("e"), nd=4,
                 lower_better=False),
        _cmp_row("root incidence [deg]", base.get("alpha_root_deg"),
                 opt.get("alpha_root_deg"), nd=2),
        _cmp_row("tip twist [deg]", base.get("twist_tip_deg"),
                 opt.get("twist_tip_deg"), nd=2),
        _cmp_row("max |twist| [deg]", base.get("twist_env_deg"),
                 opt.get("twist_env_deg"), nd=2),
        _cmp_row("max local α [deg]", base.get("alpha_geo_max_deg"),
                 opt.get("alpha_geo_max_deg"), nd=2),
        _cmp_row("root chord [m]", base.get("chord_root_m"),
                 opt.get("chord_root_m"), nd=3),
        _cmp_row("tip chord [m]", base.get("chord_tip_m"),
                 opt.get("chord_tip_m"), nd=3),
        _cmp_row("taper flown", base.get("taper_flown"),
                 opt.get("taper_flown"), nd=3),
        _cmp_row("chord deviation from taper", base.get("chord_dev"),
                 opt.get("chord_dev"), nd=4),
        _cmp_row("true MAC [m]", base.get("mac_true"), opt.get("mac_true"),
                 nd=4),
        _cmp_row("planform area flown [m²]", base.get("S_llt"),
                 opt.get("S_llt"), nd=4),
        _cmp_row("section t/c",
                 base.get("tc") or b_sec.get("tc") or s_seed.get("tc"),
                 opt.get("tc") or o_sec.get("tc") or s_opt.get("tc"), nd=4),
        _cmp_row("max thickness at [% c]", b_sec.get("tc_max_xc"),
                 o_sec.get("tc_max_xc"), nd=1, scale=100.0),
        _cmp_row("c_d at design c_l [counts]",
                 b_pol.get("cd_at_cl_design") or base.get("cd")
                 or s_seed.get("cd_at"),
                 o_pol.get("cd_at_cl_design") or opt.get("cd")
                 or s_opt.get("cd_at"), nd=1,
                 scale=1e4, lower_better=True),
        _cmp_row("|c_m| at design c_l",
                 abs(_num(base.get("cm") if base.get("cm") is not None
                          else s_seed.get("cm_at")) or 0.0)
                 if (base.get("cm") is not None
                     or s_seed.get("cm_at") is not None) else None,
                 abs(_num(opt.get("cm") if opt.get("cm") is not None
                          else s_opt.get("cm_at")) or 0.0)
                 if (opt.get("cm") is not None
                     or s_opt.get("cm_at") is not None) else None,
                 nd=4, lower_better=True),
        _cmp_row("α at design c_l [deg]",
                 b_pol.get("alpha_at_cl_design") or s_seed.get("alpha_at"),
                 o_pol.get("alpha_at_cl_design") or s_opt.get("alpha_at"),
                 nd=2),
        # THE SIX SCREENING CRITERIA, where the run measured them. A composite
        # (or goal) run evaluates the wide stall sweep for every candidate, so
        # its breakdown carries cl_max / the stall angle / both L/Ds — and this
        # table used to show none of them, leaving a 2-D run comparing four
        # rows while the objective was built out of six criteria. Elsewhere
        # they are absent and _cmp_row drops the row, exactly as before.
        _cmp_row("c_l max", crit("clmax", base, s_seed),
                 crit("clmax", opt, s_opt), nd=3, lower_better=False),
        _cmp_row("stall angle [deg]", crit("astall", base, s_seed),
                 crit("astall", opt, s_opt), nd=1, lower_better=False),
        _cmp_row("(L/D) max", crit("ldmax", base, s_seed),
                 crit("ldmax", opt, s_opt), nd=1, lower_better=False),
        _cmp_row("L/D at design c_l", crit("ldcr", base, s_seed),
                 crit("ldcr", opt, s_opt), nd=1, lower_better=False),
        _cmp_row("composite score J",
                 base.get("composite") if base.get("composite") is not None
                 else ((score or {}).get("seed") or {}).get("composite"),
                 opt.get("composite") if opt.get("composite") is not None
                 else ((score or {}).get("optimised") or {}).get("composite"),
                 nd=2, lower_better=False),
        _cmp_row("seed-floor penalty [J]", base.get("goal_penalty"),
                 opt.get("goal_penalty"), nd=2, lower_better=True),
    ]
    return [r for r in rows if r is not None]


def fig_opt_span(rep: dict) -> go.Figure | None:
    """Spanwise loading, twist law and planform of the optimised design."""
    des = rep.get("design") or {}
    geo = des.get("geometry") or {}
    bd = des.get("breakdown") or {}
    y = np.asarray(geo.get("y") or [], dtype=float)
    c = np.asarray(geo.get("chord") or [], dtype=float)
    cl = np.asarray(geo.get("Cl_y") or [], dtype=float)
    if y.size < 2 or c.shape != y.shape or cl.shape != y.shape:
        return None
    b = float(bd.get("b") or geo.get("b") or (2.0 * np.max(np.abs(y))))
    eta = np.abs(2.0 * y / b) if b else np.zeros_like(y)
    coeffs = list(bd.get("twist_coeffs_deg") or [])
    theta = np.zeros_like(y)
    for j, cj in enumerate(coeffs, start=1):      # presentation geometry
        theta = theta + float(cj) * eta ** j
    a_root = float(bd.get("alpha_root_deg") or 0.0)
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07,
        subplot_titles=("local lift coefficient",
                        "twist law and local geometric α [deg]",
                        "planform chord [m]"))
    fig.add_trace(go.Scatter(x=y, y=cl, mode="lines",
                             line=dict(color=ACCENT, width=2),
                             name="Cl(y)"), row=1, col=1)
    CL = bd.get("CL")
    if CL is not None:
        fig.add_trace(go.Scatter(
            x=y, y=np.full_like(y, float(CL)), mode="lines",
            line=dict(color=MUTED, width=1, dash="dot"),
            name="wing CL"), row=1, col=1)
    fig.add_trace(go.Scatter(x=y, y=theta, mode="lines",
                             line=dict(color=WARN, width=2),
                             name="twist θ"), row=2, col=1)
    fig.add_trace(go.Scatter(x=y, y=a_root + theta, mode="lines",
                             line=dict(color=GOOD, width=1.5, dash="dash"),
                             name="α_root + θ"), row=2, col=1)
    # x DOWNSTREAM positive, as in every other view: LE at -0.25 c
    fig.add_trace(go.Scatter(x=y, y=-0.25 * c, mode="lines",
                             line=dict(color=MUTED, width=1),
                             showlegend=False), row=3, col=1)
    fig.add_trace(go.Scatter(x=y, y=0.75 * c, mode="lines",
                             line=dict(color=MUTED, width=1),
                             fill="tonexty",
                             fillcolor="rgba(148,163,184,0.20)",
                             name="planform"), row=3, col=1)
    # ...and drawn nose-up, like every other plan view here. The data
    # keeps the package's sign; only the axis direction is reversed.
    fig.update_yaxes(autorange="reversed", row=3, col=1)
    # with a chord law in play the straight-taper BASELINE is overlaid, so the
    # reshaping the optimiser did is visible instead of merely reported
    c_trap = np.asarray(bd.get("chord_trap_m") or [], dtype=float)
    if c_trap.shape == c.shape and not np.allclose(c_trap, c):
        for sgn, show in ((-0.25, True), (0.75, False)):
            fig.add_trace(go.Scatter(
                x=y, y=sgn * c_trap, mode="lines",
                line=dict(color=WARN, width=1, dash="dot"),
                name="straight-taper baseline", showlegend=show),
                row=3, col=1)
    lay = _base_layout("Spanwise result", "y [m]", "", h=520)
    lay.pop("xaxis", None)
    lay.pop("yaxis", None)
    fig.update_layout(**lay)
    fig.update_xaxes(showgrid=True, gridcolor=GRID, color=MUTED)
    fig.update_yaxes(showgrid=True, gridcolor=GRID, color=MUTED)
    fig.update_xaxes(title_text="y [m]", row=3, col=1)
    fig.update_yaxes(scaleanchor="x", scaleratio=1.0, row=3, col=1)
    for ann in fig.layout.annotations:
        ann.font.color = MUTED
        ann.font.size = 11
    return fig


def _g_row(gi) -> list:
    """One eval's constraint value(s) as a LIST.

    Single-constraint problems (tail, hydrofoil, aircraft…) record ``eval_g``
    as a list of SCALARS, multi-constraint ones as a list of vectors. Treating
    the scalar case as a sequence raised ``TypeError: object of type 'float'
    has no len()`` and took the whole Results render down with it.
    """
    if gi is None:
        return []
    if isinstance(gi, (int, float, np.floating, np.integer)) \
            and not isinstance(gi, bool):
        return [float(gi)]
    try:
        return list(gi)
    except TypeError:
        return []


def eval_log_rows(rd: dict) -> tuple[list[dict], list[dict]]:
    """RunResult dict -> (columns, rows) for the full evaluation-log table."""
    labels = [str(l) for l in (rd.get("param_labels") or [])]
    X = rd.get("eval_x") or []
    Y = rd.get("eval_y") or []
    G = rd.get("eval_g") or []
    cols = [{"name": "i", "label": "#", "field": "i", "sortable": True}]
    cols += [{"name": f"x{j}", "label": lbl, "field": f"x{j}",
              "sortable": True} for j, lbl in enumerate(labels)]
    cols.append({"name": "f", "label": "objective f", "field": "f",
                 "sortable": True})
    n_g = max((len(_g_row(g)) for g in G), default=0)
    cols += [{"name": f"g{j}", "label": f"g[{j}]", "field": f"g{j}",
              "sortable": True} for j in range(n_g)]
    rows = []
    for i, xi in enumerate(X):
        row = {"i": i + 1, "f": _fmt(Y[i] if i < len(Y) else None, 6)}
        for j, v in enumerate(xi):
            row[f"x{j}"] = _fmt(v, 5)
        if n_g and i < len(G):
            for j, gv in enumerate(_g_row(G[i])):
                row[f"g{j}"] = _fmt(gv, 5)
        rows.append(row)
    return cols, rows


def eval_log_csv(rd: dict) -> str:
    """RunResult dict -> CSV of the full evaluation log (exact floats)."""
    labels = [str(l) for l in (rd.get("param_labels") or [])]
    X = rd.get("eval_x") or []
    Y = rd.get("eval_y") or []
    G = rd.get("eval_g") or []
    n_g = max((len(_g_row(g)) for g in G), default=0)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["eval"] + labels + ["objective"]
               + [f"g{j}" for j in range(n_g)])
    for i, xi in enumerate(X):
        row = [i + 1] + list(xi) + [Y[i] if i < len(Y) else ""]
        if n_g and i < len(G):
            row += _g_row(G[i])
        w.writerow(row)
    return buf.getvalue()


# ============================================================== run manager

class _Cancelled(Exception):
    pass


def _resume_rows(resume) -> list:
    """A resume payload as per-evaluation records, the shape a rich
    ``progress_cb`` reports and :func:`api.partial_result` reads."""
    pay = resume or {}
    X = pay.get("x") or []
    Y = pay.get("y") or []
    G = pay.get("g") or None
    rows = []
    for i, x in enumerate(X):
        g = None if not G or i >= len(G) else [float(v) for v in G[i]]
        rows.append({"n": i + 1, "x": [float(v) for v in x],
                     "f": (float(Y[i]) if i < len(Y) else None),
                     "g": g,
                     "feasible": True if g is None else bool(min(g) >= 0.0)})
    return rows


@dataclass
class RunJob:
    """One queued optimisation + its live state (thread-written, UI-read)."""

    cfg: object                      # api.RunConfig
    label: str
    budget: int
    #: optional ZERO-ARG FACTORY returning this run's own stop rule, an
    #: ``(i, best) -> bool`` handed to ``api.run(stop_rule=…)``. A factory
    #: and not a rule, because a rule carries the run's history: one shared
    #: between the seeds of a queue would stop the second run on the first
    #: run's plateau. None (the default) is the legacy path exactly.
    stop_rule: object = None
    #: passed straight to ``api.run(eval_cache=…)`` — True for the default
    #: on-disk memo, a path for another, None (the default) for none at all.
    #: A shell that sets it makes a CONTINUATION cheap: the prefix a longer
    #: run re-flies comes back off disk instead of being bought twice. It
    #: changes nothing the run reports except ``RunResult.eval_cache``, which
    #: is what makes the wall clock beside it readable.
    eval_cache: object = None
    #: how many of this run's leading evaluations are a RE-FLIGHT of a run
    #: already paid for. A continuation re-flies its prefix by design — that
    #: is what makes the longer run contain the shorter one — and with the
    #: memo above those evaluations come back off disk in milliseconds. The
    #: number is carried so a view can SAY so: a progress counter that goes
    #: back to 1 and races to 12 is the one thing about "keep going" that
    #: reads as a fault. 0 on every run that is not a continuation, and 0 on
    #: a continuation the api has judged uncontained (``note["exact"]``),
    #: where the prefix is genuinely a different set of designs.
    replay: int = 0
    #: THE EVALUATIONS THIS RUN INHERITS — a previous run's ``eval_x``/
    #: ``eval_y``/``eval_g`` (``api.resume_payload``), handed to it as its
    #: training set. Nothing in it is flown again: a 51-evaluation run
    #: continued by 8 buys 8 designs, and its counter opens at 51. None on
    #: every run that is not a resumed continuation, which is the legacy
    #: path exactly.
    resume: object = None
    #: the convergence curve of the run this one CONTINUES — best-so-far per
    #: evaluation, straight off that run's record. Carried so the view does
    #: not go blank: a continuation starts with no records of its own, and a
    #: plot that empties and redraws from evaluation 1 is exactly what the
    #: counter does, which is the thing users read as a restart. Drawn as its
    #: own trace, never merged into this run's: over the prefix the two are
    #: the same numbers by construction, and past it they are different runs.
    #: None on every run that is not a continuation.
    prior_history: object = None
    records: list = field(default_factory=list)   # rich per-eval payloads
    iters: list = field(default_factory=list)     # BO iteration diagnostics
    status: str = "queued"           # queued|running|done|error|cancelled
    result: object = None            # api.RunResult
    error: str | None = None
    t0: float | None = None
    wall: float | None = None


class RunManager:
    """Background worker draining a job queue; UI polls state via a timer."""

    def __init__(self):
        self.jobs: list[RunJob] = []
        self.cancel_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.version = 0             # bumped on every state change (UI dirty flag)

    # ---- state ----
    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def current(self) -> RunJob | None:
        for j in self.jobs:
            if j.status == "running":
                return j
        for j in reversed(self.jobs):
            if j.status in ("done", "error", "cancelled"):
                return j
        return self.jobs[0] if self.jobs else None
    # NB "skipped" is deliberately NOT in that set: a job the cancel took out
    # before it ever started produced nothing, and returning it as `current`
    # hid the PARTIAL result of the seed that really did run. See _work.

    # ---- control ----
    def start(self, jobs: list[RunJob]):
        if self.running:
            raise RuntimeError("a run is already in progress")
        self.jobs = jobs
        self.cancel_event.clear()
        self.version += 1
        self._thread = threading.Thread(target=self._work, daemon=True)
        self._thread.start()

    def cancel(self):
        self.cancel_event.set()

    # ---- worker (no ui.* calls in here — thread has no slot context) ----
    def _work(self):
        from aerobo import api
        for job in self.jobs:
            if self.cancel_event.is_set():
                # SKIPPED, not cancelled. This job never started, so it holds
                # no records and no result — and `current` returns the LAST
                # terminal job, so calling it "cancelled" made a multi-seed
                # queue publish a job with `result=None` and silently drop the
                # partial result of the seed that had actually been running.
                # The queue line reads truthfully now too ("seed 2 skipped").
                job.status = "skipped"
                self.version += 1
                continue
            job.status = "running"
            job.t0 = time.time()
            self.version += 1

            def progress(i, best, _job=job, **kw):
                if self.cancel_event.is_set():
                    raise _Cancelled
                rec = {"n": int(i),
                       "best": (float(best) if math.isfinite(best) else None)}
                rec.update({k: kw.get(k) for k in ("f", "g", "feasible", "x")})
                _job.records.append(rec)
                self.version += 1

            def iteration(rec, _job=job):
                _job.iters.append(rec)
                self.version += 1

            try:
                rule = job.stop_rule() if callable(job.stop_rule) else None
                job.result = api.run(job.cfg, progress_cb=progress,
                                     iter_cb=iteration,
                                     results_dir=str(RESULTS_DIR),
                                     stop_rule=rule,
                                     eval_cache=job.eval_cache,
                                     resume=job.resume)
                job.status = "done"
            except _Cancelled:
                # A cancelled run has still PAID for every evaluation it
                # made (on the XFOIL problems, possibly an hour of solver
                # time) and its incumbent is a real evaluated design —
                # rebuild the result from the progress log so the Results
                # page shows it instead of discarding the work.
                try:
                    # a RESUMED run owns the evaluations it inherited as well
                    # as the ones it flew: rebuilding from `records` alone
                    # would hand back a record shorter than the run this one
                    # continues, and "stop" would delete 51 evaluations
                    inherited = _resume_rows(job.resume)
                    job.result = api.partial_result(
                        job.cfg, inherited + job.records,
                        wall_time_s=time.time() - (job.t0 or time.time()),
                        bo_iters=job.iters,
                        resumed=(len(inherited) or None),
                        results_dir=str(RESULTS_DIR))
                except Exception as exc:      # never mask the cancellation
                    job.error = (f"partial result unavailable: "
                                 f"{type(exc).__name__}: {exc}")
                # status AFTER the result, never before: `partial_result`
                # builds the problem and evaluates once at best_x, which is
                # seconds to minutes on an XFOIL family. A watcher that saw
                # "cancelled" while `result` was still None would burn its
                # one publish on an empty job.
                job.status = "cancelled"
            except Exception as exc:  # surfaced in the UI
                job.error = f"{type(exc).__name__}: {exc}"
                job.status = "error"
            job.wall = time.time() - job.t0
            self.version += 1
        done = sum(1 for j in self.jobs if j.status == "done")
        _macos_notify("AeroBO Studio",
                      f"Queue finished — {done}/{len(self.jobs)} runs done")


MANAGER = RunManager()


# ============================================================== presets

def load_presets() -> dict:
    try:
        return json.loads(PRESETS_PATH.read_text())
    except (OSError, ValueError):
        return {}


def save_preset(name: str, cfg_dict: dict):
    presets = load_presets()
    presets[name] = cfg_dict
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PRESETS_PATH.write_text(json.dumps(presets, indent=2))


def delete_preset(name: str):
    presets = load_presets()
    presets.pop(name, None)
    PRESETS_PATH.write_text(json.dumps(presets, indent=2))


# ============================================================== reproduce

def reproduce_snippet(cfg_dict: dict) -> str:
    return (
        "from aerobo.api import RunConfig, run\n\n"
        "cfg = RunConfig(\n"
        f"    problem_name={cfg_dict.get('problem_name')!r},\n"
        f"    mission_kwargs={cfg_dict.get('mission_kwargs') or {}!r},\n"
        f"    flags={cfg_dict.get('flags') or {}!r},\n"
        f"    optimiser={cfg_dict.get('optimiser')!r},\n"
        f"    budget={cfg_dict.get('budget')},\n"
        f"    seed={cfg_dict.get('seed')},\n"
        f"    bounds_overrides={cfg_dict.get('bounds_overrides')!r},\n"
        # a variable the session FIXED is not in the search at all, so a
        # snippet without it reproduces a different (larger) problem
        + (f"    pinned={cfg_dict.get('pinned')!r},\n"
           if cfg_dict.get("pinned") else "")
        + ")\n"
        "res = run(cfg, results_dir='results/gui_runs')\n"
        "print(res.best_score, res.best_x)\n"
    )


def markdown_summary(rd: dict) -> str:
    lines = ["# AeroBO run summary", "",
             f"_generated {datetime.now().isoformat(timespec='seconds')}_", "",
             "## Configuration"]
    for k, v in (rd.get("config") or {}).items():
        lines.append(f"- **{k}**: {v}")
    lines += ["", "## Result",
              f"- **best score**: {rd.get('best_score')}",
              f"- **feasible**: {rd.get('feasible')}",
              f"- **evaluations**: {rd.get('n_evals')}",
              f"- **wall time**: {_fmt(rd.get('wall_time_s'))} s"]
    if rd.get("best_x") is not None:
        for lbl, v in zip(rd.get("param_labels") or [], rd["best_x"]):
            lines.append(f"- **{lbl}**: {_fmt(v, 6)}")
    bd = rd.get("breakdown") or {}
    scal = {k: v for k, v in bd.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)}
    if scal:
        lines += ["", "## Breakdown"]
        lines += [f"- **{k}**: {_fmt(v, 6)}" for k, v in scal.items()]
    return "\n".join(lines) + "\n"


# ============================================================== the app

def main(initial_choices: dict | None = None):   # noqa: PLR0915
    """Assemble the whole page. ``ui.run`` is the CALLER's job (``_run_app``).

    ``initial_choices`` is the builder configuration to open on — the app
    itself always opens on :data:`BUILDER_START` (the trim wing under a
    polynomial chord law), and the argument exists so the assembly can be
    exercised on EVERY configuration: half the builder's branches (the car's
    mount card, the water elevator's note, the tandem banner) are only
    reached by a non-default medium or system.
    """
    from nicegui import ui
    from aerobo import api, compute

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ui.dark_mode(True)               # dense-dashboard style is dark-canonical
    ui.colors(primary=ACCENT, dark="#0e1013", dark_page="#0e1013")
    ui.add_head_html(
        '<link rel="preconnect" href="https://fonts.googleapis.com">'
        '<link href="https://fonts.googleapis.com/css2?family=Fira+Sans:wght@'
        '300;400;500;600&family=Fira+Code:wght@400;500&display=swap" '
        'rel="stylesheet">'
        "<style>"
        ":root{--bg:#0e1013;--panel:#15181d;--panel2:#1a1e25;"
        "--border:#262b33;--txt:#d7dce3;--muted:#8b93a1;--faint:#5c6470;}"
        "body,body.body--dark{background:var(--bg)!important;"
        "color:var(--txt);font-family:'Fira Sans',-apple-system,"
        "'Helvetica Neue',sans-serif;}"
        ".font-mono,code,pre{font-family:'Fira Code',ui-monospace,"
        "SFMono-Regular,monospace;}"
        ".q-drawer{background:#101318!important;"
        "border-right:1px solid var(--border);}"
        ".q-card,.aero-card{background:var(--panel)!important;"
        "border:1px solid var(--border);border-radius:10px;"
        "box-shadow:none!important;}"
        ".q-dialog .q-card{background:var(--panel2)!important;}"
        ".nav-btn{justify-content:flex-start!important;width:100%;"
        "border-radius:8px;color:var(--muted)!important;font-weight:500;}"
        ".nav-btn .q-btn__content{justify-content:flex-start;}"
        ".nav-active{background:rgba(14,165,233,.13)!important;"
        "color:#7dd3fc!important;}"
        ".q-tab{text-transform:none!important;}"
        ".q-btn{text-transform:none;}"
        ".stat-label{font-size:10px;letter-spacing:.12em;color:var(--muted);"
        "text-transform:uppercase;font-weight:500;}"
        ".stat-value{font-size:20px;font-weight:600;"
        "font-variant-numeric:tabular-nums;}"
        ".page-title{font-size:17px;font-weight:600;letter-spacing:-.01em;}"
        ".page-sub{font-size:12px;color:var(--muted);}"
        ".derived-row,input,.q-table{font-variant-numeric:tabular-nums;}"
        ".q-field--outlined .q-field__control{background:var(--panel2);"
        "border-radius:8px;}"
        ".q-field--outlined .q-field__control:before"
        "{border-color:var(--border);}"
        ".q-field--dense .q-field__control{font-size:13px;}"
        ".q-table__card{background:var(--panel)!important;box-shadow:none;}"
        ".q-table thead th{color:var(--muted);font-size:11px;"
        "text-transform:uppercase;letter-spacing:.06em;}"
        ".q-chip{background:transparent;}"
        ".q-separator{background:var(--border)!important;}"
        "</style>")

    # ---------------- shared page state (single local user) ----------------
    _choices = start_choices(**(initial_choices or {}))
    #                                 an opening configuration is a state like
    #                                 any other: it holds only what this
    #                                 family has a solver for (BUILDER_START)
    _problem = derive_problem(_choices)[0]
    S: dict = {
        "choices": _choices,
        "problem": _problem,          # derived from choices
        "bounds": {},                 # label -> [lo, hi] (editing copy)
        "mission_defaults": dict(api.default_mission_values(_problem)),
        "mission_edits": {},          # only fields the user actually changed
        "prop": dict(PROP_DEFAULTS),
        "flags": {},                  # mach / ground_h_m values
        "optimiser": "bo",
        "acqf": "logei",
        "budget": 40,
        "seed": 0,
        "n_seeds": 1,
        "block_optimisers": {},       # block name -> sub-optimiser override
        "machine": "auto",            # compute-budget preset (speed only)
        "result": None,               # dict shown on the Results page
        "report": None,               # api.design_report cache for it
        "section": None,              # airfoil coords cache
        "section_report": None,       # shape + polars cache for the airfoil view
        "compare_selection": [],
        # airfoil LIBRARY SCREEN (weighted multi-criterion selection of a
        # known aerofoil — the fast, deterministic twin of the CST BO).
        "screen": {
            "weights": {"ldcr": 0.35, "clmax": 0.20, "cm": 0.20,
                        "ldmax": 0.15, "thick": 0.10, "astall": 0.0},
            "min": {"tc": 0.15, "cm": 0.08, "clmax": None,
                    "ldcr": None, "astall": None},
            "cond": {"re": 1.0e6, "mach": 0.0, "cl_design": 0.5},
            "report": None, "running": False, "error": None,
            "progress": None, "stamp": None,
        },
        # airfoil-ONLY OPTIMISER (section = CST + XFOIL). Three steps behind
        # one button: score the library with the user's WEIGHTS at the design
        # Cl the AREA GUESS implies, seed the CST box on that winner, then
        # co-optimise the eight section weights and the twist-law coefficients
        # for WING L/D at that same design Cl.
        "opt": {
            "wing": {"mass_kg": 60.0, "v_ms": 20.0, "altitude_m": 0.0,
                     "s_ref_m2": 6.0, "aspect_ratio": 10.0, "taper": 0.6},
            "twist": {"order": 1, "twist_max_deg": 6.0,
                      "alpha_max_deg": 10.0},
            "chord": dict(OPT_CHORD_DEFAULT),
            "weights": {"ldcr": 0.35, "clmax": 0.20, "cm": 0.20,
                        "ldmax": 0.15, "thick": 0.10, "astall": 0.0},
            "gates": {"tc_min": 0.12, "cm_max": 0.08},
            "optimiser": "bo", "budget": 24, "seed": 0, "shortlist": 6,
            "report": None, "screen": None, "seed_info": None, "seeds": None,
            "running": False, "error": None, "phase": "",
            "progress": None, "seed_progress": None,
            "records": [], "stamp": None,
        },
    }

    def spec():
        return api.PROBLEM_SPECS[S["problem"]]

    def _stated_v_ms():
        """The speed the mission card is showing, or None if it is blank."""
        v = S["mission_edits"].get("V")
        if v is None:
            v = S["mission_defaults"].get("V")
        return None if v is None else float(v)

    def mission_kwargs_from_state() -> dict:
        """Only the fields the user changed — untouched card = legacy path."""
        sp = spec()
        out = {}
        for k, v in S["mission_edits"].items():
            if k in sp.mission_fields and v is not None:
                dv = S["mission_defaults"].get(k)
                if dv is None or float(v) != float(dv):
                    out[k] = float(v)
        return out

    def cfg_dict_from_state() -> dict:
        sp = spec()
        flags = {k: v for k, v in S["flags"].items() if v not in (None, False)}
        if "slipstream" in sp.flags:
            ss = slipstream_dict(S["prop"])
            if ss is not None:
                flags["slipstream"] = ss
        # tail/elevator + winglet-type configuration is derived from the
        # builder choices at launch, so it can never be wiped by a physics-card
        # re-render
        flags.update(tail_flags(S["choices"]))
        flags.update(winglet_flags(S["choices"]))
        # ...and the car's SPEED, which is a MISSION field on a family that
        # refuses a mission spec. Read off the same card every other family's
        # V comes from, so the number on screen is the number flown.
        flags.update(car_flags(S["choices"], _stated_v_ms()))
        flags.update(tandem_flags(S["choices"]))
        flags.update(planform_flags(S["choices"], S["problem"]))
        if sp.has_blocks and S["optimiser"] == "blocks":
            chosen = {k: v for k, v in (S.get("block_optimisers") or {}).items()
                      if v}
            if chosen:
                flags["block_optimisers"] = chosen
        if (S["optimiser"] == "bo" and not sp.is_constrained
                and S["acqf"] != "logei"):
            flags["acqf"] = S["acqf"]
        overrides = {k: [float(lo), float(hi)]
                     for k, (lo, hi) in S["bounds"].items()}
        defaults = sp.default_bounds
        if overrides == {k: [float(a), float(b)]
                         for k, (a, b) in defaults.items()}:
            overrides = None          # untouched box -> pure legacy path
        # …and a flag travels ONLY to a family that honours it. The cards
        # above are drawn from the builder CHOICES, which outlive a change of
        # problem, so a tail card left on while a hydrofoil is selected — or a
        # winglet type on a family that picks its own section — used to send a
        # key the builder silently ignored. `api.check_flags` refuses that
        # now, so without this line the Run button would raise on the choices
        # behind five of the configs already stored under results/ (four
        # distinct problem/flag pairs). V3 applies the same
        # rule key by key (gui/v3/config.flags); this applies it once, at the
        # edge, where the whole dict is known.
        #
        # Discarding the dropped list is not the silent drop the rule is
        # about: a key reaches here only when this problem has no control for
        # it, so what is dropped is a stale choice from a previous selection,
        # not a request the user is making now. The failure the rule names — a
        # control on screen whose value is thrown away — is what
        # `api.check_flags` now makes impossible to introduce unnoticed.
        flags, _dropped = api.sanitise_flags(S["problem"], flags)
        return {"problem_name": S["problem"],
                "mission_kwargs": mission_kwargs_from_state(),
                "flags": flags, "optimiser": S["optimiser"],
                "budget": int(S["budget"]), "seed": int(S["seed"]),
                "bounds_overrides": overrides}

    def build_cfg(seed: int | None = None):
        d = cfg_dict_from_state()
        if seed is not None:
            d["seed"] = seed
        return api.RunConfig(**d)

    # ------------------------------------------------------------- sidebar
    # hidden tabs element drives the tab_panels; the sidebar is the visible nav
    with ui.element("div").style("display:none"):
        with ui.tabs() as tabs:
            tab_design = ui.tab("Design")
            tab_live = ui.tab("Live")
            tab_results = ui.tab("Results")
            tab_library = ui.tab("Airfoil Library")
            tab_optimizer = ui.tab("Airfoil Optimizer")
            tab_compare = ui.tab("Compare")

    NAV = [("Design", "tune", tab_design),
           ("Live", "show_chart", tab_live),
           ("Results", "insights", tab_results),
           ("Airfoil Library", "menu_book", tab_library),
           ("Airfoil Optimizer", "auto_graph", tab_optimizer),
           ("Compare", "stacked_line_chart", tab_compare)]
    nav_state = {"active": "Design"}
    _tab_by_name = {n: t for n, _, t in NAV}

    with ui.left_drawer(value=True, fixed=True) \
            .props("width=200 breakpoint=0"):
        with ui.row().classes("items-center gap-2 px-2 pt-3 pb-1 no-wrap"):
            ui.icon("flight_takeoff").classes("text-xl text-primary")
            with ui.column().classes("gap-0"):
                ui.label("AeroBO").classes(
                    "text-base font-semibold leading-none")
                ui.label("STUDIO").classes(
                    "text-[9px] tracking-[0.35em] opacity-40 leading-none")
        nav_col = ui.column().classes("w-full gap-1 px-1 mt-4")
        ui.space()
        with ui.column().classes("w-full px-3 pb-3 gap-1"):
            ui.separator()
            with ui.row().classes("items-center gap-2 no-wrap mt-1"):
                side_dot = ui.element("div").style(
                    "width:8px;height:8px;border-radius:50%;"
                    "background:#3f4753;")
                side_status = ui.label("idle").classes(
                    "text-xs opacity-60")

    def goto(name: str):
        nav_state["active"] = name
        panels.set_value(_tab_by_name[name])
        render_nav()

    def render_nav():
        nav_col.clear()
        with nav_col:
            for name, icon, _tab in NAV:
                b = ui.button(name, icon=icon,
                              on_click=lambda _, n=name: goto(n)) \
                    .props("flat no-caps dense align=left") \
                    .classes("nav-btn px-3")
                if name == nav_state["active"]:
                    b.classes("nav-active")

    render_nav()

    panels = ui.tab_panels(tabs, value=tab_design).classes("w-full")

    def page_header(title: str, sub: str):
        with ui.column().classes("gap-0 px-2 pt-2"):
            ui.label(title).classes("page-title")
            ui.label(sub).classes("page-sub")

    # ================================================================ DESIGN
    with panels:
        with ui.tab_panel(tab_design):
            page_header("Design",
                        "Compose the aircraft, set the mission, launch "
                        "the search")
            with ui.row().classes("w-full no-wrap items-start gap-4 p-2"):

                # ------------- left column: aircraft builder --------------
                with ui.column().classes("w-2/5 gap-4"):
                    builder_card = ui.card().classes("aero-card w-full")
                    solver_card = ui.card().classes("aero-card w-full")
                    mission_card = ui.card().classes("aero-card w-full")
                    physics_card = ui.card().classes("aero-card w-full")

                # ------------- right column: box + optimiser --------------
                with ui.column().classes("grow gap-4"):
                    bounds_card = ui.card().classes("aero-card w-full")

                    with ui.card().classes("aero-card w-full"):
                        ui.label("Optimiser").classes("stat-label")
                        _opt0 = api.compatible_optimisers(S["problem"])
                        opt_select = ui.select(
                            {n: api.OPTIMISER_SPECS[n].display
                             for n in _opt0},
                            value="bo" if "bo" in _opt0 else _opt0[0]) \
                            .classes("w-full").props("outlined dense")
                        opt_lost = ui.label("").classes("text-[11px]") \
                            .style(f"color:{MUTED}")
                        acqf_row = ui.row().classes("w-full")
                        with ui.row().classes("w-full gap-3"):
                            budget_in = ui.number(
                                "Budget (evals)", value=S["budget"], min=2,
                                step=1, precision=0).props("outlined dense") \
                                .classes("w-32")
                            seed_in = ui.number(
                                "Seed", value=S["seed"], min=0, step=1,
                                precision=0).props("outlined dense") \
                                .classes("w-24")
                            seeds_in = ui.number(
                                "Batch seeds", value=1, min=1, max=16, step=1,
                                precision=0).props("outlined dense") \
                                .classes("w-28")
                        ui.label("Batch > 1 queues seeds 0…N−1 for "
                                 "median/band comparison.") \
                            .classes("text-xs opacity-60")
                        blocks_card = ui.column().classes("w-full gap-1")

                    machine_card = ui.card().classes("aero-card w-full")

                    with ui.card().classes("aero-card w-full"):
                        ui.label("Presets").classes("stat-label")
                        with ui.row().classes("w-full items-center gap-2"):
                            preset_select = ui.select(
                                sorted(load_presets()),
                                value=None, label="Load preset") \
                                .props("outlined dense clearable") \
                                .classes("grow")
                            ui.button(icon="save", on_click=lambda:
                                      preset_dialog.open()) \
                                .props("flat round dense")
                            ui.button(icon="delete", on_click=lambda:
                                      _delete_preset()) \
                                .props("flat round dense color=negative")

                    run_btn = ui.button("Run optimisation",
                                        icon="rocket_launch") \
                        .classes("w-full h-14 text-base font-semibold") \
                        .props("push size=lg")

                    with ui.expansion("Advanced — pick the solver problem "
                                      "directly", icon="science") \
                            .classes("w-full aero-card"):
                        prob_select = ui.select(
                            {n: s.display
                             for n, s in api.PROBLEM_SPECS.items()},
                            value=S["problem"], label="Solver problem") \
                            .classes("w-full").props("outlined dense")
                        ui.label("Sets the builder above to the matching "
                                 "configuration.").classes(
                                     "text-xs opacity-60")

            with ui.dialog() as preset_dialog, ui.card():
                ui.label("Save current configuration as preset")
                preset_name = ui.input("Preset name").props("outlined dense")
                with ui.row():
                    ui.button("Save", on_click=lambda: _save_preset())
                    ui.button("Cancel", on_click=preset_dialog.close) \
                        .props("flat")

    # ---- design-page dynamic renderers ------------------------------------

    bound_inputs: dict[str, tuple] = {}

    def set_choice(key: str, val, notify=True):
        ch = S["choices"]
        if ch.get(key) == val:
            return
        ch[key] = val
        # solver capability matrix: the newest choice wins and incompatible
        # specialities reset (visible + notified). Winglets + airfoil t/c is
        # the one pair with a combined solver (specials_compatible).
        if key in SPECIAL_KEYS and val != BUILDER_DEFAULTS[key]:
            losers = [k for k in SPECIAL_KEYS
                      if k != key and ch[k] != BUILDER_DEFAULTS[k]
                      and not specials_compatible(key, k, ch)]
            for k in losers:
                ch[k] = BUILDER_DEFAULTS[k]
            if losers and notify:
                ui.notify("Reset " + ", ".join(_SPECIAL_LABEL[k]
                                               for k in losers)
                          + f" — no combined solver with "
                            f"{_SPECIAL_LABEL[key]}", type="info")
        # ...and the medium / system / tail switch is not a speciality key, so
        # the reset above never sees it: drop whatever the new configuration
        # has no solver for, or the menus (which offer only what the registry
        # can solve) would show a fallback the choices disagree with.
        dropped = normalise_choices(ch, keep=key)
        if dropped and notify:
            ui.notify("Reset " + ", ".join(_SPECIAL_LABEL[k] for k in dropped)
                      + " — no solver for it in this configuration",
                      type="info")
        # a typed NUMBER selects no family and changes no menu, so the card it
        # was typed into is left standing (NUMBER_CHOICE_KEYS): rebuilding it
        # from the field's own handler destroys that field mid-number. The
        # cards that READ the value — the bounds table, the mission defaults —
        # are redrawn as always, and they are not the one being typed into.
        apply_choices(rebuild_builder=key not in NUMBER_CHOICE_KEYS)

    def set_winglet(option_key: str):
        """Composite winglet menu -> (winglets, winglet_type). A type-only
        change (same span accounting, e.g. canted->vertical) still re-renders,
        which set_choice's early-out would otherwise skip."""
        wl, wtype = WINGLET_OPTIONS.get(option_key, ("none", "canted"))
        ch = S["choices"]
        ch["winglet_type"] = wtype
        if ch.get("winglets") != wl:
            set_choice("winglets", wl)
        else:
            # a type-only change (canted -> blended, say) does not go through
            # set_choice, but it still changes the PROBLEM and so what the
            # other menus can solve
            dropped = normalise_choices(ch, keep="winglets")
            if dropped:
                ui.notify("Reset " + ", ".join(_SPECIAL_LABEL[k]
                                               for k in dropped)
                          + " — no solver for it in this configuration",
                          type="info")
            apply_choices()

    def apply_choices(rebuild_builder: bool = True):
        name, _ = derive_problem(S["choices"])
        changed = name != S["problem"]
        S["problem"] = name
        # the default weight is the one that trims THIS wing to the legacy
        # CL, so it has to be recomputed whenever the size changes too — not
        # only when the problem does (planform_flags is empty by default, so
        # an untouched card still gets the published numbers)
        S["mission_defaults"] = dict(api.default_mission_values(
            name, planform_flags(S["choices"], name)))
        if changed:
            S["mission_edits"] = {}
            S["flags"] = {}
            if api.PROBLEM_SPECS[name].has_blocks:
                S["optimiser"] = "blocks"   # portfolio default where declared
        prob_select.set_value(name)
        if rebuild_builder:
            render_builder()
        render_solver_banner()
        render_bounds()
        render_mission()
        render_physics()
        render_optimisers()

    def render_builder():   # noqa: PLR0915
        builder_card.clear()
        ch = S["choices"]
        water = ch["medium"] == "water"
        track = ch["medium"] == "track"
        tandem = ch["system"] == "tandem"
        # Nothing is greyed out by MEDIUM any more: every feature control
        # below offers what the registry can solve for THIS configuration
        # (option_available), and set_choice keeps the state in step
        # (normalise_choices). The MODIFIERS (api.MODIFIERS) compose with every
        # family that declares them: the chord law with water and the track
        # included, the flight state with every air family that trims to a
        # weight (water already flies speed and depth; the car's speed is a
        # track condition).
        chord_live = chord_available(ch)
        flight_live = flight_available(ch)
        # ...and the TAIL, which water now takes as well: hydrotail.py puts a
        # stabiliser in the imaged solve, so "elevator" is a water question
        # too. The track (a rear wing has no tail) and the tandem pair (its
        # rear wing IS the second surface) still refuse it.
        tail_live = not (track or tandem)
        with builder_card:
            ui.label("Aircraft").classes("stat-label")

            with ui.row().classes("w-full items-center gap-6"):
                with ui.column().classes("gap-1"):
                    ui.label("Medium").classes("text-xs opacity-60")
                    ui.toggle({"air": "Air", "water": "Water",
                               "track": "Track (car wing)"},
                              value=ch["medium"],
                              on_change=lambda e:
                              set_choice("medium", e.value)) \
                        .props("dense no-caps unelevated "
                               "toggle-color=primary")
                with ui.column().classes("gap-1"):
                    ui.label("Lifting system").classes("text-xs opacity-60")
                    sys_tgl = ui.toggle(
                        {"single": "Single wing", "tandem": "Tandem"},
                        value=ch["system"],
                        on_change=lambda e: set_choice("system", e.value)) \
                        .props("dense no-caps unelevated "
                               "toggle-color=primary")
                    if water or track:
                        sys_tgl.disable()
                        sys_tgl.tooltip(
                            "water → hydrofoil solver (single surface)"
                            if water else
                            "track → car rear wing (single surface over the "
                            "ground)")

            if water:
                ui.label("Hydrofoil: cavitation-constrained single surface — "
                         "speed and depth are design variables.") \
                    .classes("text-xs opacity-70")
            if track:
                ui.label("Car rear wing: inverted surface over the track. "
                         "Maximise downforce at a drag allowance and a "
                         "deflection limit; incidence, SPAN, endplate height "
                         "and ride height are design variables — the last "
                         "three in metres, because they are lengths a car "
                         "decides.").classes("text-xs opacity-70")
                _car_controls(ch, set_choice)
            if tandem:
                ui.label("Tandem: front + rear wings, 3-knot twist laws, "
                         "area split and decalage (10 variables). Asking for "
                         "a tip device or a designed section switches the "
                         "pair to the NONPLANAR solver (tandemvlm.py), whose "
                         "twist law is root/tip instead — a different design "
                         "space, so its numbers are not the planar pair's. "
                         "A tail is the one refusal: the rear wing already "
                         "IS the second surface.").classes(
                             "text-xs opacity-70")
                _tandem_controls(ch, set_choice)

            ui.separator()

            def feature_row(label: str):
                row = ui.row().classes("w-full items-center gap-3 no-wrap")
                with row:
                    ui.label(label).classes("text-sm w-40 shrink-0 opacity-80")
                return row

            # --- winglets (span accounting + cant-band type in one menu).
            # The menu lists exactly what THIS configuration has a solver for
            # (option_available asks the registry) instead of being greyed out
            # wholesale: water runs the imaged VLM and the tandem pair runs the
            # nonplanar one, so both carry tip devices now.
            wl_opts = winglet_options(ch)
            with feature_row("Winglets"):
                wl = ui.select(
                    wl_opts,
                    value=(winglet_option_key(ch)
                           if winglet_option_key(ch) in wl_opts else "none"),
                    on_change=lambda e: set_winglet(e.value)) \
                    .props("outlined dense").classes("grow min-w-0")
                if len(wl_opts) <= 1:
                    wl.disable()
                    wl.tooltip("no tip-device solver for this configuration")
            _wl_note = missing_options_note(wl_opts, WINGLET_OPTION_LABELS,
                                            option_why(ch, WINGLET_OPTION_WHY))
            if _wl_note:
                ui.label(_wl_note).classes(
                    "text-[11px] opacity-60 ml-2 pl-3 -mt-1").style(
                    f"border-left:2px solid {GRID}")
            if ch.get("winglets", "none") != "none":
                note = objective_winglet_note(ch.get("winglet_type", "canted"))
                if note:
                    ui.label(note).classes(
                        "text-[11px] opacity-60 ml-2 pl-3 -mt-1").style(
                        f"border-left:2px solid {GRID}")
            if ch.get("winglet_type") in BLENDED_TYPES \
                    and ch.get("winglets") != "none":
                shape = ch.get("blend_shape", "spiral")
                with feature_row("Transition"):
                    ui.select(BLEND_SHAPE_LABELS, value=shape,
                              on_change=lambda e: set_choice("blend_shape",
                                                             e.value)) \
                        .props("outlined dense").classes("grow min-w-0")
                ui.label(BLEND_SHAPE_NOTES[shape]).classes(
                    "text-[11px] opacity-60 ml-2 pl-3 -mt-1").style(
                    f"border-left:2px solid {GRID}")

            # --- airfoil treatment. Same rule as the winglet menu: the
            # entries are the ones this configuration HAS a solver for, and
            # the ones it does not get named with the reason.
            af_opts = airfoil_options(ch)
            with feature_row("Airfoil"):
                af = ui.select(
                    af_opts,
                    value=(ch["airfoil"] if ch["airfoil"] in af_opts
                           else "fixed"),
                    on_change=lambda e: set_choice("airfoil", e.value)) \
                    .props("outlined dense").classes("grow min-w-0")
                if len(af_opts) <= 1:
                    af.disable()
                    af.tooltip("no section solver for this configuration")
            _af_note = missing_options_note(af_opts, AIRFOIL_OPTION_LABELS,
                                            option_why(ch, AIRFOIL_OPTION_WHY))
            if _af_note:
                ui.label(_af_note).classes(
                    "text-[11px] opacity-60 ml-2 pl-3 -mt-1").style(
                    f"border-left:2px solid {GRID}")
            if ch.get("airfoil") in ("section_only", "section_wing"):
                with ui.row().classes("w-full items-center gap-1 ml-2 pl-3 -mt-1"
                                      ).style(f"border-left:2px solid {GRID}"):
                    ui.label("Shape-optimises 8 CST weights with a live XFOIL "
                             "sweep per evaluation — minutes per run.").classes(
                        "text-[11px] opacity-60")
                    ui.button("try the Airfoil Library instead",
                              on_click=lambda: goto("Airfoil Library")) \
                        .props("flat dense no-caps").classes(
                        "text-[11px] px-1")

            # --- planform freedom: "free" is the SIZE modifier and composes
            # with every family that declares it, so the menu offers what the
            # registry accepts rather than locking on the medium
            pf_opts = planform_options(ch)
            with feature_row("Planform"):
                pf = ui.select(
                    pf_opts,
                    value=(ch["planform"] if ch["planform"] in pf_opts
                           else "fixed"),
                    on_change=lambda e: set_choice("planform", e.value)) \
                    .props("outlined dense").classes("grow min-w-0")
                if len(pf_opts) <= 1:
                    pf.disable()
                    pf.tooltip("this family carries a calibrated size (its "
                               "margins and budgets are quoted on it)")
            pf_note = missing_options_note(pf_opts, PLANFORM_CHOICE_LABELS,
                                           PLANFORM_OPTION_WHY)
            if pf_note:
                ui.label(pf_note).classes(
                    "text-[11px] opacity-60 ml-2 pl-3 -mt-1").style(
                    f"border-left:2px solid {GRID}")
            if ch.get("planform") in PLANFORM_CHOICE_NOTES:
                ui.label(PLANFORM_CHOICE_NOTES[ch["planform"]]).classes(
                    "text-[11px] opacity-60 ml-2 pl-3 -mt-1").style(
                    f"border-left:2px solid {GRID}")
            # the SIZE of a fixed planform is the user's to choose: it is a
            # value, not a design variable (see planform_flags)
            if ch.get("planform") == "fixed":
                size = planform_size_defaults(S["problem"])
                if planform_resizable(S["problem"]) and size:
                    b_val = ch.get("span_m")
                    S_val = ch.get("area_m2")
                    with ui.row().classes("w-full items-center gap-2 ml-2 pl-3"
                                          ).style(f"border-left:2px solid {GRID}"):
                        ui.number("span b [m]",
                                  value=float(b_val if b_val is not None
                                              else size[0]), step=0.5,
                                  on_change=lambda e: set_choice(
                                      "span_m", _opt_float(e.value))) \
                            .props("outlined dense").classes("w-32")
                        ui.number("area S [m²]",
                                  value=float(S_val if S_val is not None
                                              else size[1]), step=0.5,
                                  on_change=lambda e: set_choice(
                                      "area_m2", _opt_float(e.value))) \
                            .props("outlined dense").classes("w-32")
                        ui.button("reset", icon="restart_alt",
                                  on_click=lambda: (
                                      S["choices"].update(span_m=None,
                                                          area_m2=None),
                                      apply_choices())) \
                            .props("flat dense no-caps size=sm")
                    ui.label(planform_size_note(S["problem"], ch)).classes(
                        "text-[11px] opacity-60 ml-2 pl-3 -mt-1").style(
                        f"border-left:2px solid {GRID}")
                elif size:
                    ui.label(
                        f"This solver flies its own calibrated planform "
                        f"(b = {size[0]:g} m, S = {size[1]:g} m²) — its "
                        f"margins and budgets are quoted on that geometry, "
                        f"so the size is not offered here.").classes(
                            "text-[11px] opacity-60 ml-2 pl-3 -mt-1").style(
                            f"border-left:2px solid {GRID}")

            # --- chord distribution (a MODIFIER: it composes with whatever
            # family the choices above select, so it is greyed out only when
            # that family has no chord-law twin at all)
            with feature_row("Chord"):
                cd = ui.select(
                    {"fixed": "straight taper (one ratio)",
                     "free": "free cubic chord law (area held)"},
                    value=ch.get("chord", "fixed"),
                    on_change=lambda e: set_choice("chord", e.value)) \
                    .props("outlined dense").classes("grow min-w-0")
                if not chord_live:
                    cd.disable()
                    cd.tooltip("this problem has no chord-law variant — "
                               "the 2-D section problem has no wing "
                               "planform to reshape")
            if chord_live and ch.get("chord") == "free":
                ui.label(chord_law_note(S["problem"])).classes(
                    "text-[11px] opacity-60 ml-2 pl-3 -mt-1").style(
                    f"border-left:2px solid {GRID}")

            # --- flight state (a MODIFIER, like the chord law: it moves the
            # flow state and the trim target CL = W/(q S), which every air
            # family handles the same way, so it composes with the winglet,
            # the tail, the section and the chord law rather than resetting
            # them)
            with feature_row("Flight state"):
                fs = ui.select(
                    {"fixed": "fixed operating point (mission card)",
                     "free": "speed + altitude as design variables"},
                    value=ch["flight"],
                    on_change=lambda e: set_choice("flight", e.value)) \
                    .props("outlined dense").classes("grow min-w-0")
                if not flight_live:
                    fs.disable()
                    fs.tooltip(
                        "this family has no free-flight-state variant — the "
                        "water solvers already fly speed and depth as design "
                        "variables, the car's speed is a track condition, "
                        "and a 2-D section has no trim target")
            if flight_live and ch.get("flight") == "free":
                ui.label("Adds speed and altitude to the vector. Each "
                         "candidate's density and viscosity come from ISA at "
                         "ITS altitude and the trim target is CL = W/(qS) for "
                         "the mission WEIGHT, so the mission card keeps the "
                         "weight and stops offering a speed the optimiser is "
                         "choosing. Section polars stay single-Re.").classes(
                    "text-[11px] opacity-60 ml-2 pl-3 -mt-1").style(
                    f"border-left:2px solid {GRID}")

            # --- H-tail
            with feature_row("H-tail / elevator"):
                tl = ui.switch(
                    "add tail (static-margin constrained)"
                    if not water else
                    "add elevator / rear stabiliser (trims the craft)",
                    value=bool(ch["tail"]),
                    on_change=lambda e: set_choice("tail", bool(e.value)))
                if not tail_live:
                    tl.disable()
                    tl.tooltip(
                        "a car's rear wing has nothing behind it" if track
                        else "the tandem pair's rear wing IS the second "
                             "surface")
            if ch["tail"] and water:
                ui.label("The stabiliser goes INTO the imaged solve "
                         "(hydrotail.py): main foil, tip device and "
                         "stabiliser share one influence matrix and their "
                         "free-surface images, the craft is trimmed in lift "
                         "AND pitch, and cavitation is judged at every "
                         "panel's own submergence — the stabiliser sits "
                         "deeper, so it carries more static head. Its area "
                         "and arm are design variables; the LAYOUT menu "
                         "below is an air-aircraft question and does not "
                         "apply.").classes(
                    "text-[11px] opacity-60 ml-2 pl-3 -mt-1").style(
                    f"border-left:2px solid {GRID}")
            # the LAYOUT sub-form is an air-aircraft question (fin, V-tail,
            # canard, elevator chord); the water elevator says so in the note
            # above instead of being offered controls hydrotail.py ignores
            if ch["tail"] and not water:
                with ui.column().classes("w-full gap-2 ml-2 pl-3").style(
                        f"border-left:2px solid {GRID}"):
                    with ui.row().classes("w-full items-center gap-3 no-wrap"):
                        ui.label("Type").classes(
                            "text-xs w-28 shrink-0 opacity-70")
                        ui.select(TAIL_TYPE_LABELS,
                                  value=ch.get("tail_type", "conventional"),
                                  on_change=lambda e:
                                  set_choice("tail_type", e.value)) \
                            .props("outlined dense").classes("grow min-w-0")
                    ttype = ch.get("tail_type", "conventional")
                    ui.label(TAIL_TYPE_NOTES.get(ttype, "")).classes(
                        "text-xs opacity-60")
                    if ttype == "v_tail":
                        with ui.row().classes("items-center gap-3 no-wrap"):
                            ui.label("Dihedral Γ").classes(
                                "text-xs w-28 shrink-0 opacity-70")
                            ui.number(label="deg", value=float(
                                ch.get("tail_dihedral_deg", 35.0)),
                                min=20.0, max=50.0, step=1.0,
                                on_change=lambda e: set_choice(
                                    "tail_dihedral_deg",
                                    float(e.value or 35.0))) \
                                .props("outlined dense").classes("w-32")
                            ui.label("configuration, not a design variable "
                                     "(nothing here charges for directional "
                                     "stability, so a free Γ would run to 0)"
                                     ).classes("text-xs opacity-50")

                    # distance from the wing
                    with ui.row().classes("w-full items-center gap-3 no-wrap"):
                        ui.label("Distance from wing").classes(
                            "text-xs w-28 shrink-0 opacity-70")
                        ui.toggle({"free": "optimise it",
                                   "fixed": "I choose it"},
                                  value=ch.get("tail_arm", "free"),
                                  on_change=lambda e:
                                  set_choice("tail_arm", e.value)) \
                            .props("dense no-caps unelevated "
                                   "toggle-color=primary")
                        if ch.get("tail_arm") == "fixed":
                            ui.number(label="l_t [m]",
                                      value=float(ch.get("tail_arm_m", 5.5)),
                                      min=3.0, max=8.0, step=0.25,
                                      on_change=lambda e: set_choice(
                                          "tail_arm_m",
                                          float(e.value or 5.5))) \
                                .props("outlined dense").classes("w-32")
                    ui.label(
                        "Fixing the distance drops it from the design vector "
                        "(5-D → 4-D) rather than pinning a bound: a "
                        "zero-width bound breaks the samplers and quietly "
                        "turns BO into random search."
                        if ch.get("tail_arm") == "fixed" else
                        "Arm l_t and area S_t are design variables — set "
                        "their bounds in the design box on the right.") \
                        .classes("text-xs opacity-60")

                    # height above the wing plane
                    t_tail = ch.get("tail_type") == "t_tail"
                    with ui.row().classes("w-full items-center gap-3 no-wrap"):
                        ui.label("Height above wing").classes(
                            "text-xs w-28 shrink-0 opacity-70")
                        ht = ui.toggle(TAIL_HEIGHT_LABELS,
                                       value=("fixed" if t_tail else
                                              ch.get("tail_height", "fixed")),
                                       on_change=lambda e: set_choice(
                                           "tail_height", e.value)) \
                            .props("dense no-caps unelevated "
                                   "toggle-color=primary")
                        if t_tail:
                            ht.disable()
                            ht.tooltip("a T-tail's height IS its fin span")
                    ui.label(tail_height_note(ch)).classes(
                        "text-xs opacity-60")

                    # how much of the tail the search designs
                    with ui.row().classes("w-full items-center gap-3 no-wrap"):
                        ui.label("Design the tail").classes(
                            "text-xs w-28 shrink-0 opacity-70")
                        td_opts = tail_design_options(ch)
                        td = ui.select(
                            td_opts,
                            value=(ch.get("tail_design", "fixed")
                                   if ch.get("tail_design") in td_opts
                                   else "fixed"),
                            on_change=lambda e: set_choice("tail_design",
                                                           e.value)) \
                            .props("outlined dense").classes("grow min-w-0")
                        if len(td_opts) <= 1:
                            td.disable()
                            td.tooltip("no solver designs the tail here")
                    ui.label(TAIL_DESIGN_NOTES.get(
                        ch.get("tail_design", "fixed"), "")) \
                        .classes("text-xs opacity-60")
                    note = missing_options_note(td_opts, TAIL_DESIGN_LABELS,
                                                option_why(ch,
                                                           TAIL_DESIGN_WHY))
                    if note:
                        ui.label(note).classes("text-xs opacity-60")

                    # control type
                    with ui.row().classes("w-full items-center gap-3 no-wrap"):
                        ui.label("Control").classes(
                            "text-xs w-28 shrink-0 opacity-70")
                        ui.select({"stabilator": "all-moving stabilator",
                                   "elevator": "fixed stab + hinged elevator"},
                                  value=ch.get("tail_control", "stabilator"),
                                  on_change=lambda e:
                                  set_choice("tail_control", e.value)) \
                            .props("outlined dense").classes("grow min-w-0")
                        if ch.get("tail_control") == "elevator":
                            ui.number(label="c_e/c_t", value=float(
                                ch.get("tail_elevator_chord", 0.30)),
                                min=0.1, max=0.5, step=0.05,
                                on_change=lambda e: set_choice(
                                    "tail_elevator_chord",
                                    float(e.value or 0.30))) \
                                .props("outlined dense").classes("w-28") \
                                .tooltip("elevator chord / tail chord")
                    if ch.get("tail_control") == "elevator":
                        ui.label(
                            "Re-parameterisation, not new physics: a linear "
                            "flap enters the trim exactly as incidence does, "
                            "so L/D is unchanged and only the required "
                            "deflection δ_e = i_t/τ — and therefore which "
                            "designs are reachable — differs.").classes(
                                "text-xs opacity-60")

                    fin_sw = ui.switch(
                        "charge the fin's parasite drag",
                        value=bool(ch.get("tail_fin_drag")),
                        on_change=lambda e:
                        set_choice("tail_fin_drag", bool(e.value)))
                    if ttype == "v_tail":
                        fin_sw.disable()
                        fin_sw.tooltip("a V-tail carries no separate fin")
                    ui.label(
                        "Off by default because it moves the published L/D "
                        "by ≈6 %. Turn it on to compare a V-tail against the "
                        "others fairly — that trade is invisible while the "
                        "fin is free.").classes("text-xs opacity-50")

            ui.separator()
            with ui.row().classes("items-center gap-2"):
                ui.icon("switch_access_shortcut").classes(
                    "text-sm opacity-60")
                ui.label(f"Twist: {twist_description(S['problem'])}") \
                    .classes("text-xs opacity-70")

    def render_solver_banner():
        solver_card.clear()
        name, notes = derive_problem(S["choices"])
        sp = api.PROBLEM_SPECS[name]
        with solver_card:
            with ui.row().classes("w-full items-center gap-2"):
                ui.icon("precision_manufacturing").classes(
                    "text-xl text-primary")
                ui.label("Solver").classes("stat-label")
                ui.space()
                ui.chip(sp.medium,
                        icon="water_drop" if sp.medium == "water" else "air") \
                    .props("dense outline square")
                dims = len(sp.default_bounds) if sp.param_labels else 8
                ui.chip(f"{dims}-D", icon="functions").props(
                    "dense outline square")
                if sp.is_constrained:
                    ui.chip("constrained", icon="rule").props(
                        "dense outline square color=amber")
                if sp.slow:
                    ui.chip("XFOIL — slow", icon="hourglass_top").props(
                        "dense outline square color=deep-orange")
            ui.label(sp.display).classes("text-base font-medium")
            ui.label(sp.description).classes(
                "text-xs opacity-70 leading-snug")
            for n in notes:
                with ui.row().classes("items-center gap-1 no-wrap"):
                    ui.icon("info").classes("text-warning text-sm shrink-0")
                    ui.label(n).classes("text-xs opacity-70")

    def render_bounds():
        bound_inputs.clear()
        bounds_card.clear()
        defaults = spec().default_bounds
        S["bounds"] = {k: list(v) for k, v in defaults.items()}
        with bounds_card:
            with ui.row().classes("w-full items-center"):
                ui.label("Design box").classes("stat-label")
                ui.space()
                ui.button("reset", icon="restart_alt",
                          on_click=render_bounds).props("flat dense size=sm")
            if not defaults:
                ui.label("This problem exposes no editable box bounds "
                         "(fixed CST anchor box).").classes(
                             "text-xs opacity-60")
                return
            legend = bounds_legend(defaults)
            if legend:
                ui.label(legend).classes("text-xs opacity-60 leading-snug")
            with ui.grid(columns=3).classes("w-full gap-x-4 gap-y-1"):
                ui.label("parameter").classes("stat-label")
                ui.label("low").classes("stat-label")
                ui.label("high").classes("stat-label")
                # NO display format here. A format like "%.4g" is not
                # cosmetic in NiceGUI: the rounded text is what the widget
                # posts back, so merely rendering the CST box would shrink
                # w_upper_0 from 0.3412753252661441 to 0.3413 and silently
                # optimise inside a different design box than the problem
                # declares.
                for lbl, (lo, hi) in defaults.items():
                    pretty, tip = param_help(lbl)
                    with ui.column().classes("gap-0 mt-2") as cell:
                        ui.label(pretty).classes("text-sm leading-tight")
                        if pretty != lbl:
                            # the RAW label stays visible: it is the key the
                            # API, the saved run and bounds_overrides use
                            ui.label(lbl).classes(
                                "text-xs font-mono opacity-40")
                    if tip:
                        cell.tooltip(tip)
                    lo_in = ui.number(value=float(lo), step=0.01,
                                      on_change=lambda e, l=lbl:
                                      _set_bound(l, 0, e.value)) \
                        .props("outlined dense").classes("w-full")
                    hi_in = ui.number(value=float(hi), step=0.01,
                                      on_change=lambda e, l=lbl:
                                      _set_bound(l, 1, e.value)) \
                        .props("outlined dense").classes("w-full")
                    bound_inputs[lbl] = (lo_in, hi_in)
            rows = geometry_summary(S["problem"], S["bounds"],
                                    planform_flags(S["choices"],
                                                   S["problem"]))
            if rows:
                ui.separator()
                ui.label("Derived geometry").classes("stat-label")
                geo_box = ui.column().classes("w-full gap-0")
                _render_geo_rows(geo_box, rows)
                S["_geo_box"] = geo_box

    def _render_geo_rows(box, rows):
        box.clear()
        with box:
            for k, v in rows:
                with ui.row().classes(
                        "w-full items-center gap-2 no-wrap derived-row"):
                    ui.label(k).classes("text-xs w-32 shrink-0 opacity-60")
                    ui.label(v).classes("text-xs font-mono")

    def _set_bound(label, idx, value):
        if value is not None:
            S["bounds"][label][idx] = float(value)
            box = S.get("_geo_box")
            if box is not None:
                _render_geo_rows(box,
                                 geometry_summary(
                                     S["problem"], S["bounds"],
                                     planform_flags(S["choices"],
                                                    S["problem"])))

    def render_mission():
        mission_card.clear()
        sp = spec()
        defaults = S["mission_defaults"]
        edited = bool(mission_kwargs_from_state())
        with mission_card:
            with ui.row().classes("w-full items-center"):
                ui.label("Mission").classes("stat-label")
                ui.space()
                if sp.mission_fields:
                    if not edited:
                        ui.chip("legacy defaults", icon="verified").props(
                            "dense outline square color=positive")
                    else:
                        ui.chip("customised", icon="edit").props(
                            "dense outline square color=amber")
                    ui.button("reset", icon="restart_alt",
                              on_click=lambda: (
                                  S.__setitem__("mission_edits", {}),
                                  render_mission())) \
                        .props("flat dense size=sm")
            if not sp.mission_fields:
                ui.label("No mission inputs on this problem — "
                         + mission_field_note(S["problem"], "V") + ".") \
                    .classes("text-xs opacity-60")
                return
            show = ["W_N", "V",
                    "depth_m" if sp.medium == "water" else "altitude_m"]
            with ui.row().classes("w-full gap-3"):
                for fld in show:
                    label, step = MISSION_FIELD_META[fld]
                    if fld in sp.mission_fields:
                        val = S["mission_edits"].get(
                            fld, defaults.get(fld, 0.0))
                        ui.number(label, value=float(val), step=step,
                                  on_change=lambda e, f=fld:
                                  _set_mission(f, e.value)) \
                            .props("outlined dense").classes("w-36")
                    else:
                        n = ui.number(label, value=None) \
                            .props("outlined dense disable") \
                            .classes("w-36 opacity-50")
                        n.tooltip(mission_field_note(S["problem"], fld))
            ui.label("Untouched fields keep the exact legacy operating point "
                     "(bit-for-bit); only edited fields are sent to the "
                     "solver.").classes("text-xs opacity-50")

    def _set_mission(fld, value):
        if value is None:
            S["mission_edits"].pop(fld, None)
        else:
            S["mission_edits"][fld] = float(value)

    def render_physics():   # noqa: PLR0915
        # Only the two flags this card OWNS are kept; everything else in the
        # flags dict is derived at launch (tail configuration, slipstream,
        # block optimisers), so nothing that belongs to another card can be
        # destroyed by re-rendering this one.
        physics_card.clear()
        S["flags"] = {k: v for k, v in S["flags"].items()
                      if k in ("mach", "ground_h_m")}
        sp = spec()
        with physics_card:
            ui.label("Physics & propulsion").classes("stat-label")
            if not sp.flags:
                ui.label("No optional physics on this problem "
                         "(bit-for-bit baseline solver).").classes(
                             "text-xs opacity-60")
                return
            with ui.row().classes("w-full gap-8"):
                if "mach" in sp.flags:
                    with ui.column().classes("gap-1"):
                        mach_sw = ui.switch(
                            "Prandtl–Glauert (Mach)",
                            value=S["flags"].get("mach") is not None)
                        mach_val = ui.number(
                            value=S["flags"].get("mach") or 0.3,
                            min=0.0, max=0.69, step=0.05) \
                            .props("outlined dense").classes("w-24")
                        mach_val.bind_visibility_from(mach_sw, "value")

                        def _mach(_=None):
                            S["flags"]["mach"] = (float(mach_val.value)
                                                  if mach_sw.value else None)
                        mach_sw.on_value_change(_mach)
                        mach_val.on_value_change(_mach)
                if "ground_h_m" in sp.flags:
                    with ui.column().classes("gap-1"):
                        gnd_sw = ui.switch(
                            "Ground effect",
                            value=S["flags"].get("ground_h_m") is not None)
                        gnd_val = ui.number(
                            value=S["flags"].get("ground_h_m") or 1.0,
                            min=0.01, step=0.1) \
                            .props("outlined dense").classes("w-24")
                        gnd_val.bind_visibility_from(gnd_sw, "value")

                        def _gnd(_=None):
                            S["flags"]["ground_h_m"] = (float(gnd_val.value)
                                                        if gnd_sw.value
                                                        else None)
                        gnd_sw.on_value_change(_gnd)
                        gnd_val.on_value_change(_gnd)

            if "slipstream" not in sp.flags:
                return
            ui.separator()
            p = S["prop"]
            ui.switch("Propeller slipstream",
                      value=bool(p["enabled"]),
                      on_change=lambda e: (
                          p.__setitem__("enabled", bool(e.value)),
                          render_physics()))
            if not p["enabled"]:
                ui.label("Finite-jet-height lift ratio + velocity-triangle "
                         "swirl across the prop footprint (Nederlof et al. "
                         "2025). Set the propeller below when enabled.") \
                    .classes("text-xs opacity-50")
                return
            with ui.row().classes("w-full gap-3 items-start"):
                ui.number("Diameter D_p [m]", value=p["D_p"], min=0.1,
                          step=0.1,
                          on_change=lambda e: p.__setitem__(
                              "D_p", float(e.value or 0.1))) \
                    .props("outlined dense").classes("w-32")
                ui.select({"pair": "symmetric pair ±y_p",
                           "single": "single, centreline"},
                          value=p["layout"], label="Layout",
                          on_change=lambda e: (
                              p.__setitem__("layout", e.value),
                              render_physics())) \
                    .props("outlined dense").classes("w-44")
                if p["layout"] == "pair":
                    ui.number("Centre y_p [m]", value=p["y_p"], min=0.0,
                              step=0.1,
                              on_change=lambda e: p.__setitem__(
                                  "y_p", float(e.value or 0.0))) \
                        .props("outlined dense").classes("w-32")
            with ui.row().classes("w-full gap-3 items-start"):
                ui.select({"CT": "thrust coefficient CT",
                           "mu": "velocity ratio μ∞"},
                          value=p["thrust_mode"], label="Thrust input",
                          on_change=lambda e: (
                              p.__setitem__("thrust_mode", e.value),
                              render_physics())) \
                    .props("outlined dense").classes("w-44")
                if p["thrust_mode"] == "CT":
                    ui.number("CT (disk)", value=p["CT"], min=-0.9, step=0.05,
                              on_change=lambda e: p.__setitem__(
                                  "CT", float(e.value or 0.0))) \
                        .props("outlined dense").classes("w-28")
                else:
                    ui.number("μ∞ = V_j/V", value=p["mu_inf"], min=0.1,
                              step=0.05,
                              on_change=lambda e: p.__setitem__(
                                  "mu_inf", float(e.value or 1.0))) \
                        .props("outlined dense").classes("w-28")
                ui.number("Swirl Δα_ref [deg]", value=p["swirl_deg"],
                          step=0.5,
                          on_change=lambda e: p.__setitem__(
                              "swirl_deg", float(e.value or 0.0))) \
                    .props("outlined dense").classes("w-36")
                rot = ui.select(
                    {"counter": "counter-rotating pair",
                     "co_cw": "co-rotating CW",
                     "co_ccw": "co-rotating CCW"},
                    value=p["rotation"], label="Rotation",
                    on_change=lambda e: p.__setitem__("rotation", e.value)) \
                    .props("outlined dense").classes("w-44")
                if p["layout"] == "single":
                    rot.tooltip("single prop: CW/CCW only (counter needs a "
                                "pair)")
            ui.label("CT → μ∞ via actuator-disk momentum theory "
                     "(μ∞ = √(1+CT)); swirl direction follows the rotation "
                     "sense per prop. Cruise props are lightly loaded — "
                     "expect small effects unless CT is large.") \
                .classes("text-xs opacity-50")

    def render_optimisers():
        names = api.compatible_optimisers(S["problem"])
        opts = {n: api.OPTIMISER_SPECS[n].display for n in names}
        opt_select.set_options(opts)
        if S["optimiser"] not in names:
            S["optimiser"] = "bo" if "bo" in names else names[0]
        opt_select.set_value(S["optimiser"])
        # a strategy the current problem cannot offer is NAMED, not dropped
        note = chord_law_optimiser_note(S["problem"])
        opt_lost.set_text(note)
        opt_lost.set_visibility(bool(note))
        render_acqf()
        render_blocks()
        render_machine()

    #: measured evidence behind each block's default optimiser, shown in the
    #: UI so the choice can be argued with rather than taken on faith
    BLOCK_EVIDENCE = {
        "CST section": "results/airfoil.json — same 8-D sub-problem, budget "
                       "60 × 5 seeds: BO 54.6 drag counts, GA 56.1, random "
                       "59.0, SLSQP 64.6.",
        "wing/winglet": "results/block_optimiser_bench.json — this exact "
                        "5-D block: BO wins at every per-block budget "
                        "(38.07 / 37.92 / 38.99 median L/D at 8 / 12 / 20).",
    }

    def render_blocks():
        blocks_card.clear()
        sp = spec()
        if not (sp.has_blocks and S["optimiser"] == "blocks"):
            return
        try:
            declared = sp.build({}, {}, None).problem.blocks
        except Exception:
            return
        choices = api.compatible_optimisers(S["problem"])
        choices = [c for c in choices if c != "blocks"]
        with blocks_card:
            ui.separator()
            ui.label("Per-block optimiser").classes("stat-label")
            ui.label("This problem splits into variable groups with very "
                     "different evaluation costs; each group is searched by "
                     "its own method.").classes("text-xs opacity-60")
            for blk in declared:
                nm = blk.get("name", "block")
                cur = (S.get("block_optimisers") or {}).get(
                    nm, blk.get("optimiser", "bo"))
                with ui.row().classes("w-full items-center gap-2 no-wrap"):
                    ui.label(f"{nm} ({len(blk.get('indices', ()))}-D)") \
                        .classes("text-xs grow opacity-80")
                    sel = ui.select(
                        {c: api.OPTIMISER_SPECS[c].display.split(" — ")[0]
                         for c in choices},
                        value=cur if cur in choices else "bo") \
                        .props("outlined dense").classes("w-40")

                    def _set(e, _nm=nm):
                        S.setdefault("block_optimisers", {})[_nm] = e.value
                    sel.on_value_change(_set)
                ui.label(f"{blk.get('cost', '')} · default "
                         f"{blk.get('optimiser', 'bo')}: "
                         f"{BLOCK_EVIDENCE.get(nm, 'no benchmark yet')}") \
                    .classes("text-xs opacity-50")

    def render_machine():
        machine_card.clear()
        mi = compute.detect()
        budget = compute.resolve(S.get("machine", "auto"), mi)
        est = estimate_for_state(budget)
        with machine_card:
            with ui.row().classes("w-full items-center"):
                ui.label("Machine & runtime").classes("stat-label")
                ui.space()
                ui.chip(mi.label if mi.detected else "not detected",
                        icon="memory").props("dense outline square")
            ui.toggle({"quiet": "Quiet", "balanced": "Balanced",
                       "full": "Full", "auto": "Auto"},
                      value=S.get("machine", "auto"),
                      on_change=lambda e: (S.__setitem__("machine", e.value),
                                           render_machine())) \
                .props("dense no-caps unelevated toggle-color=primary")
            ui.label(f"{budget.xfoil_workers} parallel XFOIL sweeps when a "
                     f"run screens sections. This setting changes SPEED "
                     f"only — never a result.").classes("text-xs opacity-60")
            ui.separator()
            with ui.row().classes("items-baseline gap-2"):
                ui.label("Estimated run time").classes("stat-label")
                ui.label(est["label"]).classes("stat-value")
            ui.label(est["note"]).classes("text-xs opacity-50")

    def estimate_for_state(budget) -> dict:
        """Wall-clock projection for the currently configured run."""
        sp = spec()
        d = len(sp.param_labels) or 5
        n = int(S["budget"]) * max(1, int(S.get("n_seeds", 1) or 1))
        if sp.slow:
            s_eval = compute.REFERENCE["s_xfoil_sweep_fresh"]
            fresh, what = 0.75, "a fresh XFOIL sweep for most designs"
        else:
            s_eval = compute.REFERENCE["s_eval_vlm"]
            fresh, what = 1.0, "a millisecond-scale solver evaluation"
        est = compute.estimate_seconds(n, d, S["optimiser"], s_eval,
                                       fresh_fraction=fresh)
        lo = compute.human_time(est["low"])
        hi = compute.human_time(est["high"])
        share = ""
        if est["optimiser_s"] > est["physics_s"]:
            share = (" — the optimiser, not the aerodynamics, is the "
                     "bottleneck here, so extra cores cannot help")
        return {"label": f"{lo} – {hi}",
                "note": (f"{n} evaluations × {what}, plus "
                         f"{compute.human_time(est['optimiser_s'])} of "
                         f"optimiser overhead{share}. Projected from timings "
                         f"measured on {compute.REFERENCE['machine']}.")}

    def render_acqf():
        acqf_row.clear()
        if S["optimiser"] == "bo" and not spec().is_constrained:
            with acqf_row:
                ui.select(list(ACQF_CHOICES), value=S["acqf"],
                          label="Acquisition",
                          on_change=lambda e: S.__setitem__("acqf", e.value)) \
                    .props("outlined dense").classes("w-44")

    def on_problem_select(e):
        if e.value and e.value != S["problem"]:
            S["choices"] = choices_from_problem(e.value)
            apply_choices()

    prob_select.on_value_change(on_problem_select)
    opt_select.on_value_change(lambda e: (S.__setitem__("optimiser", e.value),
                                          render_acqf()))
    budget_in.on_value_change(lambda e: S.__setitem__(
        "budget", int(e.value or 2)))
    seed_in.on_value_change(lambda e: S.__setitem__("seed", int(e.value or 0)))
    seeds_in.on_value_change(lambda e: S.__setitem__(
        "n_seeds", int(e.value or 1)))

    # ---- presets ----
    def _save_preset():
        name = (preset_name.value or "").strip()
        if not name:
            ui.notify("Give the preset a name", type="warning")
            return
        save_preset(name, cfg_dict_from_state())
        preset_select.set_options(sorted(load_presets()))
        preset_dialog.close()
        ui.notify(f"Preset “{name}” saved", type="positive")

    def _delete_preset():
        if not preset_select.value:
            return
        delete_preset(preset_select.value)
        preset_select.set_options(sorted(load_presets()))
        ui.notify("Preset deleted")

    def _load_preset(e):
        if not e.value:
            return
        d = load_presets().get(e.value)
        if not d:
            return
        name = d.get("problem_name", S["problem"])
        S["choices"] = choices_from_problem(name)
        apply_choices()
        S["optimiser"] = d.get("optimiser", "bo")
        opt_select.set_value(S["optimiser"])
        S["budget"] = int(d.get("budget", 40))
        budget_in.set_value(S["budget"])
        S["seed"] = int(d.get("seed", 0))
        seed_in.set_value(S["seed"])
        mk = d.get("mission_kwargs") or {}
        if mk:
            S["mission_edits"] = {k: float(v) for k, v in mk.items()
                                  if k in MISSION_FIELD_META}
        fl = d.get("flags") or {}
        if "acqf" in fl:
            S["acqf"] = fl["acqf"]
        for k in ("mach", "ground_h_m"):
            if k in fl:
                S["flags"][k] = fl[k]
        ss = fl.get("slipstream")
        if isinstance(ss, dict):
            p = S["prop"]
            p["enabled"] = True
            p["D_p"] = float(ss.get("D_p", p["D_p"]))
            ys = [abs(float(v)) for v in ss.get("y_centres", []) if v]
            p["layout"] = "pair" if len(ss.get("y_centres", [])) > 1 \
                else "single"
            if ys:
                p["y_p"] = ys[0]
            if "mu_inf" in ss:
                p["thrust_mode"], p["mu_inf"] = "mu", float(ss["mu_inf"])
            if "CT" in ss:
                p["thrust_mode"], p["CT"] = "CT", float(ss["CT"])
            p["swirl_deg"] = float(ss.get("dalpha_ref_deg", 0.0))
            spin = ss.get("spin", -1.0)
            p["rotation"] = ("counter" if isinstance(spin, (list, tuple))
                             else ("co_ccw" if float(spin) > 0 else "co_cw"))
        else:
            S["prop"]["enabled"] = False
        ov = d.get("bounds_overrides") or {}
        for lbl, pair in ov.items():
            if lbl in S["bounds"]:
                S["bounds"][lbl] = [float(pair[0]), float(pair[1])]
                if lbl in bound_inputs:
                    bound_inputs[lbl][0].set_value(float(pair[0]))
                    bound_inputs[lbl][1].set_value(float(pair[1]))
        render_mission()
        render_physics()
        ui.notify(f"Preset “{e.value}” loaded", type="positive")

    preset_select.on_value_change(_load_preset)

    # ================================================================== LIVE
    with panels:
        with ui.tab_panel(tab_live):
            page_header("Live", "Streaming optimisation diagnostics — GP "
                                "residuals, acquisition, constraints")
            with ui.column().classes("w-full p-2 gap-3"):
                with ui.card().classes("aero-card w-full"):
                    with ui.row().classes("w-full items-center gap-4"):
                        live_spinner = ui.spinner(size="md")
                        live_title = ui.label("No run yet").classes(
                            "text-base font-medium")
                        live_chiprow = ui.row().classes("gap-2")
                        ui.space()
                        live_eta = ui.label("").classes("text-sm opacity-70")
                        cancel_btn = ui.button(
                            "Cancel", icon="stop_circle",
                            on_click=lambda: (MANAGER.cancel(),
                                              ui.notify("Cancelling at next "
                                                        "evaluation…"))) \
                            .props("outline color=negative dense")
                    live_bar = ui.linear_progress(value=0.0,
                                                  show_value=False) \
                        .classes("w-full")
                    with ui.row().classes("w-full gap-8"):
                        def _stat(label):
                            with ui.column().classes("gap-0"):
                                ui.label(label).classes("stat-label")
                                return ui.label("—").classes("stat-value")
                        stat_eval = _stat("evaluations")
                        stat_best = _stat("best feasible")
                        stat_rate = _stat("evals / s")
                        stat_queue = _stat("queue")

                with ui.grid(columns=2).classes("w-full gap-3"):
                    live_conv = ui.plotly(_empty_fig("Launch a run from the "
                                                     "Design tab")) \
                        .classes("w-full")
                    live_surr = ui.plotly(_empty_fig("")).classes("w-full")
                    live_acq = ui.plotly(_empty_fig("")).classes("w-full")
                    live_g = ui.plotly(_empty_fig("")).classes("w-full")

    _live_seen = {"version": -1, "job": None}

    def refresh_live():
        if MANAGER.version == _live_seen["version"]:
            return
        _live_seen["version"] = MANAGER.version
        job = MANAGER.current
        if job is None:
            return
        # sidebar status pill
        dot = {"running": ACCENT, "done": GOOD, "error": BAD,
               "cancelled": WARN}.get(job.status, "#3f4753")
        side_dot.style(f"width:8px;height:8px;border-radius:50%;"
                       f"background:{dot};")
        if job.status == "running":
            side_status.set_text(
                f"running · {len(job.records)}/{job.budget}")
        else:
            side_status.set_text(job.status)
        sp = api.PROBLEM_SPECS.get(job.cfg.problem_name)
        n = len(job.records)
        live_title.set_text(job.label)
        live_spinner.set_visibility(job.status == "running")
        cancel_btn.set_visibility(job.status in ("running", "queued"))
        live_bar.set_value(min(1.0, n / max(job.budget, 1)))
        stat_eval.set_text(f"{n} / {job.budget}")
        best = next((r["best"] for r in reversed(job.records)
                     if r.get("best") is not None), None)
        stat_best.set_text(_fmt(best))
        if job.t0 and n:
            el = (job.wall or (time.time() - job.t0))
            rate = n / max(el, 1e-9)
            stat_rate.set_text(f"{rate:.2f}")
            if job.status == "running" and rate > 0:
                live_eta.set_text(
                    f"ETA ≈ {max(job.budget - n, 0) / rate:.0f} s")
            else:
                live_eta.set_text(f"wall {el:.1f} s")
        idx = MANAGER.jobs.index(job) + 1 if job in MANAGER.jobs else 1
        stat_queue.set_text(f"{idx} / {len(MANAGER.jobs) or 1}")
        live_chiprow.clear()
        with live_chiprow:
            color = {"running": "primary", "done": "positive",
                     "error": "negative", "cancelled": "warning",
                     "queued": "grey"}.get(job.status, "grey")
            ui.chip(job.status, icon="circle").props(
                f"dense outline square color={color}")
            if job.error:
                ui.chip("see console", icon="error").props(
                    "dense outline square color=negative")

        live_conv.update_figure(fig_convergence(job.records))
        live_surr.update_figure(fig_surrogate(job.records, job.iters))
        live_acq.update_figure(fig_acquisition(job.iters))
        if sp is not None and sp.is_constrained:
            live_g.update_figure(fig_constraints(
                job.records, list(sp.constraint_labels)))
        else:
            live_g.update_figure(fig_calibration(job.records, job.iters))

        # a finished newest job feeds the Results page automatically —
        # including a CANCELLED one, whose partial log is real paid-for work
        if (job.status in ("done", "cancelled") and job.result is not None
                and _live_seen["job"] is not job):
            _live_seen["job"] = job
            set_result(job.result.to_dict())
            if job.status == "cancelled":
                ui.notify(f"Cancelled after {job.result.n_evals} evaluations "
                          f"— best so far {_fmt(job.result.best_score)} "
                          "(Results tab updated)", type="warning")
            else:
                ui.notify(f"Run complete — best {_fmt(job.result.best_score)} "
                          "(Results tab updated)", type="positive")

    ui.timer(0.5, refresh_live)

    # ---- launch ----
    def launch():
        if MANAGER.running:
            ui.notify("A run is already in progress — cancel it first",
                      type="warning")
            return
        for lbl, (lo, hi) in S["bounds"].items():
            if not (float(lo) < float(hi)):
                ui.notify(f"Bound “{lbl}”: low must be < high",
                          type="negative")
                return
        p = S["prop"]
        if p["enabled"] and "slipstream" in spec().flags:
            if float(p["D_p"]) <= 0:
                ui.notify("Propeller diameter must be positive",
                          type="negative")
                return
            if p["thrust_mode"] == "mu" and float(p["mu_inf"]) <= 0:
                ui.notify("Velocity ratio μ∞ must be positive",
                          type="negative")
                return
            if p["layout"] == "pair":
                span_edge = float(p["y_p"]) + float(p["D_p"]) / 2.0
                if span_edge > 5.0:            # b = 10 fixed on these wings
                    ui.notify(f"Prop footprint reaches y = {span_edge:g} m — "
                              "beyond the 5 m half-span; it will be "
                              "clipped by the wing tip", type="warning")
        n_seeds = max(1, int(S["n_seeds"]))
        base_seed = int(S["seed"])
        jobs = []
        for k in range(n_seeds):
            cfg = build_cfg(seed=base_seed + k)
            tag = f" · seed {base_seed + k}" if n_seeds > 1 else \
                  f" · seed {base_seed}"
            jobs.append(RunJob(
                cfg=cfg,
                label=f"{spec().display} — "
                      f"{api.OPTIMISER_SPECS[S['optimiser']].display}{tag}",
                budget=int(S["budget"])))
        try:
            MANAGER.start(jobs)
        except RuntimeError as exc:
            ui.notify(str(exc), type="negative")
            return
        _live_seen["version"] = -1
        goto("Live")
        word = "runs" if n_seeds > 1 else "run"
        ui.notify(f"Launched {n_seeds} {word}", type="positive")

    run_btn.on_click(launch)

    # =============================================================== RESULTS
    with panels:
        with ui.tab_panel(tab_results):
            page_header("Results", "Best design, geometry views and the "
                                   "full run record")
            results_box = ui.column().classes("w-full p-2 gap-3")

    def set_result(rd: dict):
        S["result"] = rd
        S["report"] = None
        S["section"] = None
        S["section_report"] = None
        render_results()
        # geometry needs one extra physics eval — do it off the event loop
        threading.Thread(target=_compute_report, args=(rd,),
                         daemon=True).start()

    def _compute_report(rd: dict):
        cfg = None
        try:
            cfg = api.RunConfig(**{k: rd["config"].get(k) for k in
                                   ("problem_name", "mission_kwargs", "flags",
                                    "optimiser", "budget", "seed",
                                    "bounds_overrides")})
            if rd.get("best_x") is not None:
                S["report"] = api.design_report(cfg, rd["best_x"])
                S["section"] = api.section_coords(cfg, rd["best_x"])
        except Exception as exc:
            S["report"] = {"error": f"{type(exc).__name__}: {exc}"}
        # The section view is a SEPARATE try: an airfoil-panel failure must
        # not take the geometry views down with it. Computed once and cached
        # in state — re-deriving it on every re-render would re-burn the
        # XFOIL timeout for a design whose sweep timed out (those are
        # deliberately never cached to disk).
        if cfg is not None and rd.get("best_x") is not None:
            try:
                S["section_report"] = api.section_report(cfg, rd["best_x"])
            except Exception as exc:
                S["section_report"] = {"error": f"{type(exc).__name__}: {exc}"}
        S["report_ready"] = time.time()

    _report_seen = {"t": None}

    def _poll_report():
        if S.get("report_ready") and S.get("report_ready") != _report_seen["t"]:
            _report_seen["t"] = S["report_ready"]
            render_results()

    ui.timer(1.0, _poll_report)

    def render_results():   # noqa: PLR0915
        results_box.clear()
        rd = S["result"]
        with results_box:
            if rd is None:
                with ui.card().classes("aero-card w-full items-center p-10"):
                    ui.icon("insights").classes("text-5xl opacity-30")
                    ui.label("No result yet — run an optimisation, or load a "
                             "saved run from the Compare tab.") \
                        .classes("opacity-60")
                return

            labels = rd.get("param_labels") or []
            best_x = rd.get("best_x")
            bounds = rd.get("bounds") or []
            cfgd = rd.get("config") or {}

            if rd.get("partial"):
                with ui.card().classes("aero-card w-full").style(
                        f"border-left:3px solid {WARN}"):
                    with ui.row().classes("items-center gap-2"):
                        ui.icon("pause_circle").style(f"color:{WARN}")
                        ui.label("Cancelled run — partial result").classes(
                            "stat-label")
                    ui.label(
                        f"The search was stopped after "
                        f"{rd.get('n_evals', 0)} of {cfgd.get('budget', '?')} "
                        f"evaluations. Everything below is real, evaluated "
                        f"work, but the best design is the best found SO FAR "
                        f"— not a converged search, and not comparable with a "
                        f"completed run at this budget.").classes(
                            "text-xs opacity-70")

            # ---- headline cards
            with ui.row().classes("w-full gap-3"):
                def card(label, value, icon, color="text-primary"):
                    with ui.card().classes("aero-card grow"):
                        with ui.row().classes("items-center gap-3"):
                            ui.icon(icon).classes(f"text-3xl {color}")
                            with ui.column().classes("gap-0"):
                                ui.label(label).classes("stat-label")
                                ui.label(value).classes("stat-value")
                card("best score", _fmt(rd.get("best_score")), "emoji_events")
                card("feasible",
                     "yes" if rd.get("feasible") else "NO",
                     "verified" if rd.get("feasible") else "report",
                     "text-positive" if rd.get("feasible")
                     else "text-negative")
                card("evaluations", str(rd.get("n_evals", "—")), "numbers")
                card("wall time", f"{_fmt(rd.get('wall_time_s'))} s", "timer")
                card("run", f"{cfgd.get('optimiser', '?')} · "
                            f"seed {cfgd.get('seed', '?')}", "settings")

            with ui.row().classes("w-full no-wrap items-start gap-3"):
                # ---- best design vector with bound-riding indicators
                with ui.card().classes("aero-card w-1/3"):
                    ui.label("Best design vector").classes("stat-label")
                    if best_x is None:
                        ui.label("No feasible incumbent found.").classes(
                            "text-sm opacity-60")
                    else:
                        for i, v in enumerate(best_x):
                            lbl = labels[i] if i < len(labels) else f"x{i}"
                            lo, hi = (bounds[i] if i < len(bounds)
                                      else (None, None))
                            with ui.row().classes(
                                    "w-full items-center gap-2 no-wrap"):
                                ui.label(lbl).classes(
                                    "text-xs w-28 shrink-0 opacity-80")
                                ui.label(_fmt(v, 5)).classes(
                                    "text-sm font-mono w-20 shrink-0")
                                if lo is not None and hi > lo:
                                    frac = (float(v) - lo) / (hi - lo)
                                    riding = frac < 0.02 or frac > 0.98
                                    bar = ui.linear_progress(
                                        value=min(max(frac, 0.0), 1.0),
                                        show_value=False,
                                        color="warning" if riding
                                        else "primary") \
                                        .classes("grow").props("rounded")
                                    tip = (f"[{_fmt(lo)}, {_fmt(hi)}]"
                                           + ("  ← riding the bound!"
                                              if riding else ""))
                                    bar.tooltip(tip)
                                    if riding:
                                        ui.icon("priority_high").classes(
                                            "text-warning text-sm")
                        ui.label("Amber bar = optimum riding its box bound "
                                 "(§13 boundary-riding law).") \
                            .classes("text-xs opacity-50 mt-2")

                    feas_rows = []
                    g_last = rd.get("best_g")
                    report = S.get("report") or {}
                    clabels = (report.get("constraint_labels")
                               or [])
                    bd = (report.get("breakdown") or rd.get("breakdown")
                          or {})
                    g_bd = bd.get("g")
                    if isinstance(g_bd, (list, tuple)):
                        for j, gv in enumerate(g_bd):
                            name = (clabels[j] if j < len(clabels)
                                    else f"g[{j}]")
                            feas_rows.append((name, gv))
                    elif g_last is not None:
                        feas_rows.append((clabels[0] if clabels
                                          else "binding margin", g_last))
                    if feas_rows:
                        ui.separator()
                        ui.label("Constraint margins").classes("stat-label")
                        for name, gv in feas_rows:
                            ok = gv is not None and float(gv) >= 0
                            with ui.row().classes("items-center gap-2"):
                                ui.icon("check_circle" if ok else "cancel") \
                                    .classes("text-positive" if ok
                                             else "text-negative")
                                ui.label(f"{name}: {_fmt(gv, 4)}").classes(
                                    "text-sm")

                # ---- geometry / convergence column
                with ui.column().classes("grow gap-3"):
                    ui.plotly(fig_convergence(
                        [], history=rd.get("history"),
                        title="Convergence (best feasible so far)")) \
                        .classes("w-full")
                    report = S.get("report")
                    if report is None:
                        with ui.card().classes("aero-card w-full p-6 "
                                               "items-center"):
                            ui.spinner(size="md")
                            ui.label("Evaluating best design for geometry "
                                     "views…").classes("text-xs opacity-60")
                    elif report.get("error"):
                        ui.label(f"Geometry evaluation failed: "
                                 f"{report['error']}").classes(
                                     "text-xs text-negative")
                    else:
                        geom = report.get("geometry") or {}
                        sect = S.get("section")
                        pf = fig_planform(geom, best_x, labels)
                        ui.plotly(pf).classes("w-full")
                        w3 = fig_wing3d(geom, best_x, labels, section=sect)
                        if w3 is not None:
                            ui.plotly(w3).classes("w-full")
                        fv = fig_frontview(geom)
                        if fv is not None:
                            ui.plotly(fv).classes("w-full")
                        # a winglet the SOLVER dropped must not be implied
                        # anywhere on this page
                        wl_req = geom.get("winglet") or {}
                        if wl_req and not geom.get("is_winglet"):
                            ui.label(
                                f"Winglet requested at h/(b/2) = "
                                f"{wl_req.get('h_frac', 0):.4f} "
                                f"({wl_req.get('h_m', 0):.3f} m) was BELOW "
                                f"the solver's 1 %-of-semi-span cut-off and "
                                f"was not flown — the views above show the "
                                f"plain wing, which is what was scored.") \
                                .classes("text-xs").style(f"color:{WARN}")
                        sw = fig_spanwise(geom)
                        if sw is not None:
                            ui.plotly(sw).classes("w-full")

            # ---- airfoil section view (CST-carrying problems only)
            sr = S.get("section_report")
            if isinstance(sr, dict) and not sr.get("error") and sr.get("design"):
                with ui.expansion("Airfoil section — shape and polars",
                                  icon="airlines", value=True).classes(
                                      "w-full aero-card"):
                    des = sr["design"]
                    pol = des.get("polar") or {}
                    with ui.row().classes("w-full gap-6 items-center"):
                        def stat(k, v):
                            with ui.column().classes("gap-0"):
                                ui.label(k).classes("stat-label")
                                ui.label(v).classes("text-sm font-mono")
                        stat("t/c", f"{des['tc']:.4f} "
                                    f"(min {sr['tc_min']:.2f})")
                        stat("max thickness at", f"{des['tc_max_xc']:.1%} c")
                        stat(f"c_d at c_l = {sr['cl_design']:.2f}",
                             (f"{pol['cd_at_cl_design'] * 1e4:.1f} counts"
                              if pol.get("cd_at_cl_design") is not None
                              else "—"))
                        stat("c_m at design c_l",
                             (f"{pol['cm_at_cl_design']:+.4f} "
                              f"(|cap| {sr['cm_max']:.2f})"
                              if pol.get("cm_at_cl_design") is not None
                              else "—"))
                        stat("XFOIL", f"{pol.get('n_converged', 0)}/"
                                      f"{pol.get('n_requested', 0)} converged "
                                      f"· Re {pol.get('re', 0):.3g}")
                    shp = fig_section_shape(sr)
                    if shp is not None:
                        ui.plotly(shp).classes("w-full")
                    pf2 = fig_section_polars(sr)
                    if pf2 is not None:
                        ui.plotly(pf2).classes("w-full")
                    note = ("Section polars are the live viscous XFOIL sweep "
                            "this design was scored with, at the section's "
                            "fixed design Reynolds number.")
                    if (rd.get("problem_name") or "").startswith("winglet"):
                        note += (" The optimised quantity for this problem is "
                                 "the wing+winglet L/D — the section c_d "
                                 "shown here is a diagnostic, not the "
                                 "objective.")
                    if not sr.get("baseline"):
                        note += (" No anchor overlay: this design IS the "
                                 "anchor section, or the anchor's sweep is "
                                 "not in the cache.")
                    ui.label(note).classes("text-xs opacity-60")
            elif isinstance(sr, dict) and sr.get("error"):
                ui.label(f"Airfoil view unavailable: {sr['error']}").classes(
                    "text-xs opacity-60")

            # ---- BO diagnostics (post-hoc)
            if rd.get("bo_iters"):
                recs = [{"n": i + 1, "f": f}
                        for i, f in enumerate(rd.get("eval_y") or [])]
                with ui.expansion("BO surrogate diagnostics",
                                  icon="query_stats").classes(
                                      "w-full aero-card"):
                    with ui.grid(columns=2).classes("w-full gap-3"):
                        ui.plotly(fig_surrogate(recs, rd["bo_iters"]))
                        ui.plotly(fig_acquisition(rd["bo_iters"]))

            # ---- full breakdown scalars
            bd = ((S.get("report") or {}).get("breakdown")
                  or rd.get("breakdown") or {})
            scal = {k: v for k, v in bd.items()
                    if isinstance(v, (int, float))
                    and not isinstance(v, bool)}
            if scal:
                with ui.expansion("All breakdown scalars",
                                  icon="table_rows").classes(
                                      "w-full aero-card"):
                    with ui.grid(columns=4).classes("w-full gap-1"):
                        for k, v in sorted(scal.items()):
                            ui.label(k).classes("text-xs opacity-60")
                            ui.label(_fmt(v, 6)).classes(
                                "text-xs font-mono col-span-3")

            # ---- FULL results: every evaluation + the raw record
            with ui.expansion(f"Full evaluation log "
                              f"({rd.get('n_evals', 0)} evals)",
                              icon="list_alt").classes("w-full aero-card"):
                cols, rows = eval_log_rows(rd)
                ui.table(columns=cols, rows=rows, row_key="i",
                         pagination=15).classes("w-full") \
                    .props("dense flat")
                ui.button("Download CSV", icon="download",
                          on_click=lambda: ui.download(
                              eval_log_csv(rd).encode(),
                              "aerobo_evals.csv")).props("outline dense")

            with ui.expansion("Raw result JSON (everything the run "
                              "recorded)", icon="data_object").classes(
                                  "w-full aero-card"):
                try:
                    ui.json_editor({"content": {"json": rd},
                                    "readOnly": True}).classes("w-full")
                except Exception:
                    ui.code(json.dumps(rd, indent=2)[:200_000],
                            language="json").classes("w-full")

            # ---- exports
            with ui.row().classes("w-full gap-2"):
                ui.button("Download JSON", icon="download",
                          on_click=lambda: ui.download(
                              json.dumps(rd, indent=2).encode(),
                              "aerobo_run.json")).props("outline dense")
                ui.button("Summary (Markdown)", icon="description",
                          on_click=lambda: ui.download(
                              markdown_summary(rd).encode(),
                              "aerobo_summary.md")).props("outline dense")
                ui.button("Copy reproduce snippet", icon="content_copy",
                          on_click=lambda: _copy_snippet(cfgd)) \
                    .props("outline dense")
                if rd.get("path"):
                    ui.label(f"saved: {Path(rd['path']).name}").classes(
                        "text-xs opacity-50 mt-2")

    def _copy_snippet(cfgd: dict):
        code = reproduce_snippet(cfgd)
        with ui.dialog() as dlg, ui.card().classes("w-[640px]"):
            ui.label("Reproduce this run from Python").classes("stat-label")
            ui.code(code, language="python").classes("w-full")
            with ui.row():
                ui.button("Copy", icon="content_copy",
                          on_click=lambda: (ui.clipboard.write(code),
                                            ui.notify("Copied")))
                ui.button("Close", on_click=dlg.close).props("flat")
        dlg.open()

    render_results()

    # ============================================================ AIRFOIL LIB
    def help_label(text: str, key: str, cls: str = "text-xs grow opacity-70"):
        """A field label that explains what the number DOES, on hover."""
        lab = ui.label(text).classes(cls + " cursor-help")
        tip = PARAM_HELP.get(key)
        if tip:
            with lab:
                ui.tooltip(tip).classes("max-w-[320px] text-[11px] "
                                        "leading-snug whitespace-normal")
        return lab

    SCREEN_WEIGHT_LABELS = {
        "ldcr": "L/D at design Cl", "clmax": "Cl max",
        "cm": "|Cm| (lower better)", "ldmax": "(L/D) max",
        "thick": "thickness t/c", "astall": "stall angle",
    }
    SCREEN_MIN_LABELS = {
        "tc": "min t/c", "cm": "max |Cm|", "clmax": "min Cl max",
        "ldcr": "min L/D @ Cl", "astall": "min stall angle [deg]",
    }

    with panels:
        with ui.tab_panel(tab_library):
            page_header(
                "Airfoil Library",
                "Pick the best KNOWN aerofoil for this Reynolds/Mach by a "
                "weighted score — the fast, deterministic alternative to CST "
                "shape optimisation")
            with ui.column().classes("w-full p-2 gap-3"):
                with ui.card().classes("w-full aero-card"):
                    ui.label("How it works").classes("stat-label")
                    ui.markdown(
                        "Every airfoil in the UIUC database is run through a "
                        "real viscous XFOIL polar at the operating point, then "
                        "scored on six criteria — **L/D at the design Cl**, "
                        "**(L/D)max**, **Cl_max**, **stall angle**, **t/c** and "
                        "**|Cm|**. Set a weight per criterion and a hard "
                        "minimum on any of them; the composite is the "
                        "weight-blend of each criterion's 0–100 rank across "
                        "the sections that clear every minimum. Near-instant: "
                        "the polars are cached, so no XFOIL is re-run."
                    ).classes("text-xs opacity-70")

                with ui.row().classes("w-full gap-3 items-stretch no-wrap"):
                    with ui.card().classes("grow aero-card"):
                        ui.label("Criterion weights").classes("stat-label")
                        for k, lbl in SCREEN_WEIGHT_LABELS.items():
                            with ui.row().classes(
                                    "w-full items-center gap-3 no-wrap"):
                                help_label(lbl, f"w_{k}")
                                ui.number(
                                    value=S["screen"]["weights"][k],
                                    min=0.0, max=1.0, step=0.05,
                                    on_change=lambda e, kk=k: S["screen"]
                                    ["weights"].__setitem__(
                                        kk, float(e.value or 0.0))) \
                                    .props("dense outlined").classes("w-24")
                    with ui.card().classes("grow aero-card"):
                        ui.label("Minimums (hard gates)").classes("stat-label")
                        for k, lbl in SCREEN_MIN_LABELS.items():
                            with ui.row().classes(
                                    "w-full items-center gap-3 no-wrap"):
                                help_label(lbl, {"tc": "tc_min",
                                                 "cm": "cm_max"}.get(
                                    k, f"floor_{k}"))
                                ui.number(
                                    value=S["screen"]["min"][k],
                                    step=(0.01 if k in ("tc", "cm") else 0.1),
                                    on_change=lambda e, kk=k: S["screen"]["min"]
                                    .__setitem__(
                                        kk, (float(e.value)
                                             if e.value is not None else None)))\
                                    .props("dense outlined clearable") \
                                    .classes("w-24")
                        ui.label("blank = no floor on that metric").classes(
                            "text-[11px] opacity-50 mt-1")
                    with ui.card().classes("aero-card").style("min-width:220px"):
                        ui.label("Operating point").classes("stat-label")
                        for k, lbl, st in (("re", "Reynolds", 1e5),
                                           ("mach", "Mach", 0.05),
                                           ("cl_design", "design Cl", 0.05)):
                            with ui.row().classes(
                                    "w-full items-center gap-3 no-wrap"):
                                help_label(lbl, k)
                                ui.number(
                                    value=S["screen"]["cond"][k], step=st,
                                    on_change=lambda e, kk=k: S["screen"]["cond"]
                                    .__setitem__(kk, float(e.value or 0.0))) \
                                    .props("dense outlined").classes("w-28")
                        ui.label("Re/Mach also come from the mission card on "
                                 "the Design page.").classes(
                            "text-[11px] opacity-50 mt-1")

                with ui.row().classes("w-full items-center gap-3"):
                    ui.button("Screen library", icon="travel_explore",
                              on_click=lambda: start_screen()) \
                        .props("unelevated no-caps")
                    screen_status = ui.label("").classes("text-xs opacity-60")

                screen_box = ui.column().classes("w-full gap-3")

    def start_screen():
        sc = S["screen"]
        if sc["running"]:
            return
        sc["running"] = True
        sc["error"] = None
        sc["progress"] = None
        screen_status.set_text("screening…")
        threading.Thread(target=_screen_worker, daemon=True).start()

    def _screen_worker():
        sc = S["screen"]
        try:
            floors = {k: v for k, v in sc["min"].items()
                      if k in api.SCREEN_FLOOR_KEYS and v is not None}
            kw = dict(
                weights=dict(sc["weights"]),
                re=sc["cond"]["re"], mach=sc["cond"]["mach"],
                cl_design=sc["cond"]["cl_design"],
                tc_min=float(sc["min"].get("tc") or 0.0),
                cm_max=float(sc["min"].get("cm") or 1e9),
                floors=floors, top_n=15)
            # AWAY FROM THE CACHED POINT the screen is real XFOIL — the whole
            # database would be hours, so the leaders of the cached ranking are
            # re-swept there instead (api.screen_at_point, which says so in its
            # report). At the cached point this is the same instant screen it
            # always was.
            rep = (api.screen_at_point(**kw)
                   if _screen_needs_sweep(sc["cond"])
                   else api.screen_airfoils(**kw))
            sc["report"] = rep
        except Exception as exc:        # noqa: BLE001 — surface, never crash UI
            sc["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            sc["running"] = False
            sc["stamp"] = time.time()

    _screen_seen = {"t": None}

    def _poll_screen():
        sc = S["screen"]
        if sc["running"]:
            screen_status.set_text("screening…")
        if sc.get("stamp") and sc["stamp"] != _screen_seen["t"]:
            _screen_seen["t"] = sc["stamp"]
            screen_status.set_text("")
            render_screen()

    ui.timer(0.8, _poll_screen)

    def render_screen():
        screen_box.clear()
        sc = S["screen"]
        rep = sc.get("report")
        with screen_box:
            if sc.get("error"):
                ui.label(f"Screen failed: {sc['error']}").classes(
                    "text-sm").style(f"color:{BAD}")
                return
            if not rep:
                return
            cond = rep["conditions"]
            with ui.row().classes("w-full gap-6 items-baseline"):
                ui.label(
                    f"{rep['n_eligible']} of {rep['n_screened']} sections "
                    f"clear the gates").classes("text-sm font-medium")
                ui.label(
                    f"Re {cond['re']:.3g} · M {cond['mach']:.2f} · "
                    f"Cl_design {cond['cl_design']:.2f} · "
                    f"t/c ≥ {cond['tc_min']:.2f} · |Cm| ≤ {cond['cm_max']:.2f}"
                    f" · {rep['wall_time_s'] * 1e3:.0f} ms").classes(
                    "text-xs opacity-60")
            if rep.get("floors"):
                ui.label("floors: " + ", ".join(
                    f"{k} ≥ {v:g}" for k, v in rep["floors"].items())).classes(
                    "text-xs opacity-60")
            # is the TYPED operating point actually the one that was scored?
            pt = rep.get("point") or {}
            if pt.get("matched"):
                ui.label(
                    f"{pt.get('rederived', 0)} sections recomputed at this "
                    f"design Cl from their cached polars"
                    + (f" · {pt['not_rederived']} left as stored"
                       if pt.get("not_rederived") else "")).classes(
                    "text-[11px] opacity-50")
            else:
                cached = pt.get("cached")
                ui.label(
                    "the design Cl was NOT applied: this screen is the stored "
                    "checkpoint, whose polars are indexed at "
                    + (f"Re {cached['re']:.3g} · M {cached['mach']:.2f}"
                       if cached else "an unrecorded operating point")
                    + ". Changing Re, Mach or Cl needs new XFOIL work — the "
                      "ranking below is at the checkpoint's own point."
                ).classes("text-[11px]").style(f"color:{WARN}")

            win = rep.get("winner")
            if not win:
                ui.label("No section clears every gate — loosen a minimum "
                         "or a weight floor.").classes("text-sm").style(
                    f"color:{WARN}")
                return

            des = win.get("design") or {}
            with ui.card().classes("w-full aero-card"):
                with ui.row().classes("w-full items-baseline gap-3"):
                    ui.icon("emoji_events").style(f"color:{WARN}")
                    ui.label(f"Best: {des.get('name', win['metrics']['name'])}"
                             ).classes("text-base font-semibold")
                    ui.label(f"composite {win['metrics']['composite']:.1f}"
                             ).classes("text-xs opacity-60")
                with ui.row().classes("w-full gap-6 items-center"):
                    def wstat(k, v):
                        with ui.column().classes("gap-0"):
                            ui.label(k).classes("stat-label")
                            ui.label(v).classes("text-sm font-mono")
                    m = win["metrics"]
                    wstat("t/c", f"{m['tc']:.4f}")
                    wstat("L/D @ Cl", f"{m['ldcr']:.1f}"
                          if m.get("ldcr") is not None else "—")
                    wstat("(L/D) max", f"{m['ldmax']:.1f}"
                          if m.get("ldmax") is not None else "—")
                    wstat("Cl max", (f"{m['clmax']:.2f}"
                                     + ("+" if m.get("clmax_censored") else ""))
                          if m.get("clmax") is not None else "—")
                    wstat("stall α", f"{m['astall']:.1f}°"
                          if m.get("astall") is not None else "—")
                    wstat("|Cm|", f"{abs(m['cm_at']):.4f}"
                          if m.get("cm_at") is not None else "—")
                shp = fig_section_shape(win)
                if shp is not None:
                    ui.plotly(shp).classes("w-full")
                pf = fig_section_polars(win)
                if pf is not None:
                    ui.plotly(pf).classes("w-full")
                if "w_upper" in win:
                    with ui.row().classes("w-full items-center gap-2"):
                        ui.label(
                            "The Airfoil Optimizer seeds its CST box on this "
                            "kind of pick — it runs the screen itself, so it "
                            "always starts from a library winner.").classes(
                            "text-xs opacity-60")
                        ui.button(
                            "Carry these weights over →",
                            icon="rocket_launch",
                            on_click=lambda: _seed_optimizer_from_library()) \
                            .props("flat dense no-caps").classes(
                            "text-[11px] px-1")

            # ranked table
            rows = []
            for i, r in enumerate(rep["ranked"]):
                rows.append({
                    "rank": i + 1, "name": r["name"],
                    "composite": _fmt(r.get("composite"), 1),
                    "ldcr": _fmt(r.get("ldcr"), 1),
                    "ldmax": _fmt(r.get("ldmax"), 1),
                    "clmax": _fmt(r.get("clmax"), 2),
                    "astall": _fmt(r.get("astall"), 1),
                    "tc": _fmt(r.get("tc"), 3),
                    "cm": _fmt(abs(r["cm_at"]) if r.get("cm_at") is not None
                               else None, 4),
                })
            cols = [
                {"name": "rank", "label": "#", "field": "rank",
                 "align": "left"},
                {"name": "name", "label": "airfoil", "field": "name",
                 "align": "left"},
                {"name": "composite", "label": "score", "field": "composite"},
                {"name": "ldcr", "label": "L/D@Cl", "field": "ldcr"},
                {"name": "ldmax", "label": "(L/D)max", "field": "ldmax"},
                {"name": "clmax", "label": "Cl max", "field": "clmax"},
                {"name": "astall", "label": "stall°", "field": "astall"},
                {"name": "tc", "label": "t/c", "field": "tc"},
                {"name": "cm", "label": "|Cm|", "field": "cm"},
            ]
            with ui.card().classes("w-full aero-card"):
                ui.label("Ranking (top 15)").classes("stat-label")
                ui.table(columns=cols, rows=rows, row_key="rank") \
                    .props("dense flat").classes("w-full")

    def _seed_optimizer_from_library():
        """Carry the criterion weights + the two section gates into the
        optimiser tab (the screen→optimise handoff).

        The operating point is NOT carried: the optimiser derives its design
        Cl and Reynolds number from the wing area guess, so copying a typed-in
        Cl across would put a stale number next to a derived one."""
        S["opt"]["weights"].update(S["screen"]["weights"])
        for src, dst in (("tc", "tc_min"), ("cm", "cm_max")):
            v = S["screen"]["min"].get(src)
            if v is not None:
                S["opt"]["gates"][dst] = float(v)
        goto("Airfoil Optimizer")

    render_screen()

    # ====================================================== AIRFOIL OPTIMIZER
    #
    # Section-only design, three steps behind one button:
    #   1. the AREA GUESS fixes the design lift coefficient CL = W/(qS) and
    #      the Reynolds number at the MAC;
    #   2. the library is screened with the USER'S weights at that design Cl,
    #      and its winner seeds the CST design box (always — there is no
    #      unseeded path here);
    #   3. the eight CST weights and the twist-law coefficients are
    #      co-optimised for WING L/D at that same design Cl.
    OPT_WING_META = {
        "mass_kg": ("mass [kg]", 1.0),
        "v_ms": ("cruise speed [m/s]", 0.5),
        "altitude_m": ("altitude [m]", 100.0),
        "s_ref_m2": ("wing area guess [m²]", 0.25),
        "aspect_ratio": ("aspect ratio b²/S", 0.5),
        "taper": ("taper c_tip/c_root", 0.05),
    }
    OPT_GATE_META = {
        "tc_min": ("min t/c", 0.01),
        "cm_max": ("max |Cm|", 0.01),
    }
    OPT_WEIGHT_LABELS = {
        "ldcr": "L/D at design Cl", "clmax": "Cl max",
        "cm": "|Cm| (lower better)", "ldmax": "(L/D) max",
        "thick": "thickness t/c", "astall": "stall angle",
    }

    with panels:
        with ui.tab_panel(tab_optimizer):
            page_header(
                "Airfoil Optimizer",
                "Section-only design: CST + XFOIL, seeded from the library "
                "winner, co-optimised with a twist law for wing L/D at the "
                "lift your area guess implies")
            with ui.column().classes("w-full p-2 gap-3"):
                with ui.card().classes("w-full aero-card"):
                    ui.label("How it works").classes("stat-label")
                    ui.markdown(
                        "**1 · your area guess sets the target.** Weight "
                        "W = m·g over dynamic pressure and the area you "
                        "guessed gives the design lift coefficient "
                        "**CL = W/(qS)**; the Reynolds number comes from the "
                        "mean aerodynamic chord of the same planform. "
                        "Neither is typed in — both would otherwise drift "
                        "away from the geometry they describe.\n\n"
                        "**2 · the library picks the seed — by the number "
                        "that matters.** Your criterion weights SHORTLIST the "
                        "UIUC database at that design Cl; each shortlisted "
                        "section is then flown untwisted on the real wing and "
                        "the seed is the one with the best **L/D at the "
                        "cruise CL**, not the best composite. (Those disagree: "
                        "the weighted score is a 2-D proxy and knows nothing "
                        "about induced drag.) Its CST refit re-centres the "
                        "eight-weight design box — every run is base-seeded, "
                        "never a generic NACA box.\n\n"
                        "**3 · shape, twist and chord are co-optimised.** Each "
                        "candidate is flown through a **live viscous XFOIL "
                        "sweep**, the wing is trimmed to CL_design on a "
                        "lifting line, and the score is **wing "
                        "L/D = CL/(CDi + CDp)** — induced drag from the "
                        "loading, profile drag from the strip integral of "
                        "that same polar — subject to hard **t/c**, "
                        "**|Cm|**, twist-envelope and incidence margins. Turn "
                        "the **chord law** on and the planform reshapes too, "
                        "at constant area, under its own deviation margin. New "
                        "shapes cost seconds of XFOIL; twist- and chord-only "
                        "moves reuse the cached polar and are nearly free.\n\n"
                        "*Wing-only L/D: no fuselage, tail or interference "
                        "drag is in this number.*"
                    ).classes("text-xs opacity-70")

                with ui.row().classes("w-full gap-3 items-stretch no-wrap"):
                    with ui.card().classes("grow aero-card"):
                        ui.label("The wing you are designing for").classes(
                            "stat-label")
                        for k in OPT_WING_KEYS:
                            lbl, st = OPT_WING_META[k]
                            with ui.row().classes(
                                    "w-full items-center gap-3 no-wrap"):
                                help_label(lbl, k)
                                ui.number(
                                    value=S["opt"]["wing"][k], step=st,
                                    on_change=lambda e, kk=k: (
                                        S["opt"]["wing"].__setitem__(
                                            kk, float(e.value or 0.0)),
                                        _sync_wing_preview())) \
                                    .props("dense outlined").classes("w-28")
                        opt_derived = ui.column().classes(
                            "w-full gap-0 mt-2 pt-2 border-t "
                            "border-slate-500/20")
                    with ui.card().classes("aero-card").style(
                            "min-width:260px"):
                        ui.label("Twist law").classes("stat-label")
                        with ui.row().classes(
                                "w-full items-center gap-3 no-wrap"):
                            help_label("law", "twist_order")
                            ui.select(
                                {int(k): f"{k} · {v}"
                                 for k, v in api.airfoil_twist_orders().items()},
                                value=int(S["opt"]["twist"]["order"]),
                                on_change=lambda e: S["opt"]["twist"]
                                .__setitem__("order", int(e.value))) \
                                .props("dense outlined").classes("w-36")
                        for k, lbl, st in (
                                ("twist_max_deg", "max twist [deg]", 0.5),
                                ("alpha_max_deg", "max angle [deg]", 0.5)):
                            with ui.row().classes(
                                    "w-full items-center gap-3 no-wrap"):
                                help_label(lbl, k)
                                ui.number(
                                    value=S["opt"]["twist"][k], min=0.1,
                                    step=st,
                                    on_change=lambda e, kk=k: S["opt"]["twist"]
                                    .__setitem__(
                                        kk, float(e.value or 0.1))) \
                                    .props("dense outlined").classes("w-24")
                        ui.label(
                            "θ(η) = Σ tⱼηʲ, zero at the root; the root "
                            "incidence is solved for at trim, not chosen."
                        ).classes("text-[11px] opacity-50 mt-1")

                        ui.label("Chord law").classes("stat-label mt-3")
                        with ui.row().classes(
                                "w-full items-center gap-3 no-wrap"):
                            help_label("law", "chord_order")
                            ui.select(
                                {int(k): (f"{k} · {v}" if k else v)
                                 for k, v in
                                 api.airfoil_chord_orders().items()},
                                value=int(S["opt"]["chord"]["order"]),
                                on_change=lambda e: (
                                    S["opt"]["chord"].__setitem__(
                                        "order", int(e.value)),
                                    _sync_chord_note())) \
                                .props("dense outlined").classes("w-44")
                        with ui.row().classes(
                                "w-full items-center gap-3 no-wrap"):
                            help_label("max deviation", "chord_max_frac")
                            ui.number(
                                value=S["opt"]["chord"]["chord_max_frac"],
                                min=0.05, max=0.9, step=0.05,
                                on_change=lambda e: S["opt"]["chord"]
                                .__setitem__(
                                    "chord_max_frac",
                                    float(e.value or 0.05))) \
                                .props("dense outlined").classes("w-24")
                        opt_chord_note = ui.label("").classes(
                            "text-[11px] opacity-50 mt-1")

                        ui.label("Section gates").classes("stat-label mt-3")
                        for k in OPT_GATE_KEYS:
                            lbl, st = OPT_GATE_META[k]
                            with ui.row().classes(
                                    "w-full items-center gap-3 no-wrap"):
                                help_label(lbl, k)
                                ui.number(
                                    value=S["opt"]["gates"][k], step=st,
                                    on_change=lambda e, kk=k: S["opt"]["gates"]
                                    .__setitem__(
                                        kk, float(e.value or 0.0))) \
                                    .props("dense outlined").classes("w-24")

                with ui.row().classes("w-full gap-3 items-stretch no-wrap"):
                    with ui.card().classes("grow aero-card"):
                        ui.label("Seed weights (which library section "
                                 "starts the search)").classes("stat-label")
                        for k, lbl in OPT_WEIGHT_LABELS.items():
                            with ui.row().classes(
                                    "w-full items-center gap-3 no-wrap"):
                                help_label(lbl, f"w_{k}")
                                ui.number(
                                    value=S["opt"]["weights"][k],
                                    min=0.0, max=1.0, step=0.05,
                                    on_change=lambda e, kk=k: S["opt"]
                                    ["weights"].__setitem__(
                                        kk, float(e.value or 0.0))) \
                                    .props("dense outlined").classes("w-24")
                        opt_seed_note = ui.label("").classes(
                            "text-[11px] opacity-50 mt-1")
                    with ui.card().classes("aero-card").style(
                            "min-width:240px"):
                        ui.label("Search").classes("stat-label")
                        with ui.row().classes(
                                "w-full items-center gap-3 no-wrap"):
                            help_label("optimiser", "optimiser")
                            ui.select(
                                {"bo": "Bayesian (BoTorch)", "ga": "GA (pymoo)",
                                 "random": "Random", "sobol": "Sobol DOE"},
                                value=S["opt"]["optimiser"],
                                on_change=lambda e: S["opt"].__setitem__(
                                    "optimiser", e.value)) \
                                .props("dense outlined").classes("w-40")
                        for k, lbl, lo, hi, st in (
                                ("budget", "evaluations", 4, 400, 1),
                                ("shortlist", "seed shortlist", 1, 20, 1),
                                ("seed", "seed", 0, 9999, 1)):
                            with ui.row().classes(
                                    "w-full items-center gap-3 no-wrap"):
                                help_label(lbl, k)
                                ui.number(
                                    value=S["opt"][k], min=lo, max=hi, step=st,
                                    on_change=lambda e, kk=k, lo=lo: S["opt"]
                                    .__setitem__(kk, int(e.value or lo))) \
                                    .props("dense outlined").classes("w-28")

                with ui.row().classes("w-full items-center gap-3"):
                    ui.button("Run optimisation", icon="rocket_launch",
                              on_click=lambda: start_optimize()) \
                        .props("unelevated no-caps")
                    opt_stop = ui.button(
                        "Stop", icon="stop", on_click=lambda: stop_optimize()) \
                        .props("outline dense").classes("hidden")
                    opt_status = ui.label("").classes("text-xs opacity-60")

                opt_conv = ui.plotly(
                    _empty_fig("Run to see the convergence trace")) \
                    .classes("w-full")
                opt_box = ui.column().classes("w-full gap-3")

    def _wing_preview() -> tuple[dict | None, str]:
        """(derived quantities, error) for the current area guess."""
        try:
            return api.wing_guess_preview(**S["opt"]["wing"]), ""
        except (ValueError, TypeError) as exc:
            return None, str(exc)

    def _sync_wing_preview():
        """Redraw the derived read-out under the wing card."""
        opt_derived.clear()
        d, err = _wing_preview()
        with opt_derived:
            if d is None:
                ui.label(f"unusable guess: {err}").classes(
                    "text-xs").style(f"color:{BAD}")
                return
            ui.label("derived from this guess — not editable").classes(
                "stat-label")
            with ui.row().classes("w-full gap-4 flex-wrap"):
                for lab, val in (
                        ("design CL", f"{d['cl_design']:.3f}"),
                        ("Re at MAC", f"{d['re_mac']:.3g}"),
                        ("span", f"{d['b']:.2f} m"),
                        ("MAC", f"{d['mac']:.3f} m"),
                        ("q", f"{d['q']:.0f} Pa"),
                        ("weight", f"{d['weight_n']:.0f} N")):
                    with ui.column().classes("gap-0"):
                        ui.label(lab).classes("stat-label")
                        ui.label(val).classes("text-sm font-mono")
            if d["cl_design"] > 1.2:
                ui.label(
                    "design CL above ~1.2 — that is more lift than a plain "
                    "section holds in cruise; guess a bigger area or fly "
                    "faster.").classes("text-[11px] mt-1").style(
                    f"color:{WARN}")

    def _sync_chord_note():
        """Say what the chord law costs and what it is allowed to do."""
        m = int(S["opt"]["chord"]["order"])
        if not m:
            opt_chord_note.set_text(
                "c(η) = c_root(1 − (1−λ)η): the fixed trapezoid your taper "
                "ratio implies. No chord design variables.")
            return
        opt_chord_note.set_text(
            f"c(η) = A·c_trap(η)·(1 + Σ kⱼηʲ) with {m} coefficient"
            f"{'s' if m > 1 else ''}, A holding the area (and so CL_design) "
            f"fixed. Adds {m} dimension{'s' if m > 1 else ''} to the search. "
            f"At this budget one run is dominated by the luck of the initial "
            f"design — the seed-to-seed spread is several times what any "
            f"chord law is worth — so change the seed and read the spread, "
            f"not one number.")

    def _sync_opt_seed_note():
        """State where the seed screen can run, honestly."""
        pt = api.screen_library_point()
        d, _err = _wing_preview()
        cl = f"{d['cl_design']:.3f}" if d else "—"
        if pt:
            opt_seed_note.set_text(
                f"the seed screen runs at Cl {cl} (your derived design lift) "
                f"on the library's cached polars at Re {pt['re']:.3g} · "
                f"M {pt['mach']:.2f}; the optimisation itself flies the "
                f"derived Re")
        else:
            opt_seed_note.set_text(
                "library polar branches not cached yet — the seed screen "
                "will rank on whatever operating point the screen checkpoint "
                "was built at (build the branch cache to fix the design Cl)")

    def start_optimize():
        oc = S["opt"]
        if oc["running"]:
            return
        d, err = _wing_preview()
        if d is None:
            ui.notify(f"unusable wing guess: {err}", type="negative")
            return
        oc["running"] = True
        oc["error"] = None
        oc["records"] = []
        oc["report"] = None
        oc["screen"] = None
        oc["seed_info"] = None
        oc["seeds"] = None
        oc["seed_progress"] = None
        oc["phase"] = "screen"
        opt_status.set_text("screening the library for a seed…")
        opt_stop.classes(remove="hidden")
        opt_box.clear()
        _opt_cancel["flag"] = False
        threading.Thread(target=_optimize_worker, daemon=True).start()

    _opt_cancel = {"flag": False}

    def stop_optimize():
        _opt_cancel["flag"] = True
        opt_status.set_text("stopping after the current evaluation…")

    def _optimize_worker():
        oc = S["opt"]

        def progress(i, best, **kw):
            if _opt_cancel["flag"]:
                raise _Cancelled
            oc["records"].append(
                {"n": int(i),
                 "best": (float(best) if math.isfinite(best) else None)})
            oc["progress"] = int(i)

        try:
            wing = dict(oc["wing"])
            gates = dict(oc["gates"])
            derived = api.wing_guess_preview(**wing)
            # --- step 1: the library picks the seed, at the DERIVED Cl ---
            pt = api.screen_library_point() or {}
            n_short = max(1, int(oc["shortlist"]))
            scr = api.screen_airfoils(
                weights=dict(oc["weights"]),
                re=float(pt.get("re", 1.0e6)),
                mach=float(pt.get("mach", 0.0)),
                cl_design=float(derived["cl_design"]),
                tc_min=gates["tc_min"], cm_max=gates["cm_max"],
                top_n=max(n_short, 12), with_shape=False)
            oc["screen"] = scr
            cands = api.screen_seed_candidates(scr, n=n_short)
            if not cands:
                raise RuntimeError(
                    "no library section clears the gates at the derived "
                    f"design Cl {derived['cl_design']:.3f} — loosen min t/c "
                    "or max |Cm|, or guess a larger wing area")
            if _opt_cancel["flag"]:
                raise _Cancelled
            # the weighted score only SHORTLISTS; the seed is chosen by the
            # objective itself — wing L/D at the cruise CL
            oc["phase"] = "shortlist"

            def seed_prog(i, n, _row):
                if _opt_cancel["flag"]:
                    raise _Cancelled
                oc["seed_progress"] = (int(i), int(n))

            picked = api.rank_seeds_by_wing_ld(
                cands, wing=wing, tc_min=gates["tc_min"],
                cm_max=gates["cm_max"], twist_order=int(oc["twist"]["order"]),
                twist_max_deg=float(oc["twist"]["twist_max_deg"]),
                alpha_max_deg=float(oc["twist"]["alpha_max_deg"]),
                progress_cb=seed_prog)
            oc["seeds"] = picked["ranked"]
            best = picked.get("best")
            if best is None:
                raise RuntimeError(
                    "none of the shortlisted sections could be flown at the "
                    f"cruise CL {derived['cl_design']:.3f} — "
                    + (picked["ranked"][0]["reason"] if picked["ranked"]
                       else "no candidates"))
            win = next(c for c in cands if c["name"] == best["name"])
            oc["seed_info"] = {
                "name": best["name"], "composite": best.get("composite"),
                "ldcr": best.get("ldcr"), "tc": best.get("tc"),
                "ld": best.get("ld"), "rank": best.get("rank"),
                "n_shortlist": len(picked["ranked"]),
            }
            if _opt_cancel["flag"]:
                raise _Cancelled
            # --- steps 2+3: base-seeded CST + twist + chord co-optimisation
            oc["phase"] = "optimise"
            rep = api.optimize_airfoil(
                wing=wing, twist_order=int(oc["twist"]["order"]),
                twist_max_deg=float(oc["twist"]["twist_max_deg"]),
                alpha_max_deg=float(oc["twist"]["alpha_max_deg"]),
                chord_order=int(oc["chord"]["order"]),
                chord_max_frac=float(oc["chord"]["chord_max_frac"]),
                tc_min=gates["tc_min"], cm_max=gates["cm_max"],
                anchor=[win["w_upper"], win["w_lower"]],
                optimiser=oc["optimiser"], budget=int(oc["budget"]),
                seed=int(oc["seed"]), progress_cb=progress,
                results_dir=str(RESULTS_DIR))
            oc["report"] = rep
        except _Cancelled:
            oc["error"] = "stopped — partial trace kept below"
        except Exception as exc:        # noqa: BLE001 — surface, never crash UI
            oc["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            oc["running"] = False
            oc["phase"] = ""
            oc["stamp"] = time.time()

    def _opt_conv_fig():
        recs = S["opt"].get("records") or []
        if not recs:
            return _empty_fig("Run to see the convergence trace")
        xs = [r["n"] for r in recs]
        ys = [r["best"] for r in recs]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines+markers",
            line=dict(color=ACCENT, width=2),
            marker=dict(size=4), name="best wing L/D"))
        fig.update_layout(**_base_layout(
            "Best-so-far wing L/D at the design CL", "evaluation",
            "L/D  (wing only)", h=300))
        return fig

    _opt_seen = {"t": None}

    def _poll_optimize():
        oc = S["opt"]
        if oc["running"]:
            if oc["phase"] == "screen":
                opt_status.set_text("screening the library for a seed…")
            elif oc["phase"] == "shortlist":
                i, n = oc.get("seed_progress") or (0, 0)
                opt_status.set_text(
                    f"flying the shortlist at the cruise CL… {i}/{n}")
            else:
                n = oc.get("progress") or 0
                opt_status.set_text(
                    f"optimising… {n}/{int(oc['budget'])} evaluations "
                    f"(live XFOIL)")
                opt_conv.update_figure(_opt_conv_fig())
        if oc.get("stamp") and oc["stamp"] != _opt_seen["t"]:
            _opt_seen["t"] = oc["stamp"]
            opt_stop.classes(add="hidden")
            opt_status.set_text("")
            render_optimize()

    ui.timer(0.8, _poll_optimize)
    ui.timer(2.0, _sync_opt_seed_note)

    def render_optimize():
        opt_box.clear()
        oc = S["opt"]
        rep = oc.get("report")
        opt_conv.update_figure(_opt_conv_fig())
        with opt_box:
            if oc.get("error"):
                ui.label(oc["error"]).classes("text-sm").style(
                    f"color:{WARN if 'stopped' in oc['error'] else BAD}")
                if not rep:
                    return
            if not rep:
                return
            res = rep.get("result") or {}
            cond = rep.get("conditions") or {}
            wg = cond.get("wing") or {}
            tw = cond.get("twist") or {}
            chd = cond.get("chord") or {}
            bd = (rep.get("design") or {}).get("breakdown") or {}
            best = res.get("best_score")

            with ui.row().classes("w-full gap-6 items-baseline"):
                ui.label(
                    ("feasible optimum found" if res.get("feasible")
                     else "no feasible design found — loosen a gate")
                ).classes("text-sm font-medium").style(
                    None if res.get("feasible") else f"color:{WARN}")
                ui.label(
                    f"{res.get('n_evals', 0)} evals · "
                    f"CL {cond.get('cl_design', 0):.3f} · "
                    f"Re {cond.get('re', 0):.3g} · "
                    f"{tw.get('name', 'linear')} twist ≤ "
                    f"{tw.get('twist_max_deg', 0):.1f}° · "
                    f"α ≤ {tw.get('alpha_max_deg', 0):.1f}° · "
                    + (f"{chd.get('name')} chord ≤ "
                       f"{chd.get('chord_max_frac', 0):.0%} · "
                       if chd.get("order") else "fixed taper · ")
                    + f"t/c ≥ {cond.get('tc_min', 0):.2f} · "
                    f"|Cm| ≤ {cond.get('cm_max', 0):.2f} · "
                    f"{rep.get('wall_time_s', 0):.0f} s").classes(
                    "text-xs opacity-60")

            # THE number: wing L/D at the cruise CL, stated once, large
            if bd.get("LD") is not None:
                base_ld = ((rep.get("baseline") or {}).get("breakdown")
                           or {}).get("LD")
                with ui.card().classes("w-full aero-card"):
                    with ui.row().classes("w-full items-end gap-4"):
                        with ui.column().classes("gap-0"):
                            ui.label("wing L/D at the cruise CL").classes(
                                "stat-label")
                            ui.label(f"{bd['LD']:.2f}").classes(
                                "text-4xl font-semibold leading-none")
                        with ui.column().classes("gap-0 pb-1"):
                            ui.label(f"at CL {bd.get('CL', 0):.3f} · "
                                     f"CD {bd.get('CD', 0) * 1e4:.1f} counts"
                                     ).classes("text-xs opacity-60")
                            if base_ld:
                                dl = bd["LD"] - float(base_ld)
                                ui.label(
                                    f"{dl:+.2f} ({100 * dl / float(base_ld):+.1f} %) "
                                    f"vs the seed it started from "
                                    f"({float(base_ld):.2f})").classes(
                                    "text-xs font-medium").style(
                                    f"color:{GOOD if dl > 0 else BAD}")

            info = oc.get("seed_info") or {}
            scr = oc.get("screen") or {}
            if info:
                with ui.card().classes("w-full aero-card"):
                    with ui.row().classes("w-full items-baseline gap-3"):
                        ui.icon("eco").style(f"color:{GOOD}")
                        ui.label(f"Seeded from: {info['name']}").classes(
                            "text-sm font-medium")
                        ui.label(
                            f"chosen by wing L/D "
                            f"{_fmt(info.get('ld'), 2)} out of "
                            f"{info.get('n_shortlist', 0)} shortlisted · "
                            f"weighted-score rank #{info.get('rank', '—')} "
                            f"(composite {_fmt(info.get('composite'), 1)}) · "
                            f"{scr.get('n_eligible', 0)} of "
                            f"{scr.get('n_screened', 0)} sections cleared the "
                            f"gates").classes("text-xs opacity-60")
                    seeds = oc.get("seeds") or []
                    if len(seeds) > 1:
                        with ui.element("div").classes(
                                "w-full grid gap-x-5 gap-y-1 mt-2").style(
                                "grid-template-columns:1.4fr .6fr .8fr 1fr "
                                "2.2fr"):
                            for h in ("shortlisted section", "score #",
                                      "section L/D", "WING L/D @ cruise CL",
                                      ""):
                                ui.label(h).classes("stat-label")
                            for s in seeds:
                                chosen = s["name"] == info["name"]
                                cls = ("text-xs font-mono"
                                       + (" font-semibold" if chosen
                                          else " opacity-70"))
                                ui.label(("→ " if chosen else "") + s["name"]
                                         ).classes(cls)
                                ui.label(str(s.get("rank", "—"))).classes(cls)
                                ui.label(_fmt(s.get("ldcr"), 1)).classes(cls)
                                lab = ui.label(_fmt(s.get("ld"), 2)).classes(
                                    cls)
                                if chosen:
                                    lab.style(f"color:{GOOD}")
                                note = ui.label(s.get("reason", "")).classes(
                                    "text-[11px]")
                                note.style(f"color:{WARN}" if s.get("reason")
                                           else "")
                        ui.label(
                            "The weighted score only SHORTLISTS. Each entry "
                            "was then flown untwisted on your wing and the "
                            "seed is the best L/D at the cruise CL — the two "
                            "orders disagree because the score is a 2-D proxy "
                            "that cannot see induced drag.").classes(
                            "text-[11px] opacity-50 mt-1")
                    pt = (scr.get("point") or {})
                    if pt.get("matched"):
                        ui.label(
                            f"ranked at YOUR design Cl "
                            f"{cond.get('cl_design', 0):.3f} "
                            f"({pt.get('rederived', 0)} sections recomputed "
                            f"from cached polars at Re "
                            f"{(pt.get('cached') or {}).get('re', 0):.3g})"
                        ).classes("text-[11px] opacity-50")
                    else:
                        ui.label(
                            "ranked from the screen checkpoint as stored — "
                            "its polars are not indexed at this operating "
                            "point, so the seed pick reflects the Cl the "
                            "checkpoint was screened at, not yours."
                        ).classes("text-[11px]").style(f"color:{WARN}")

            if bd:
                with ui.card().classes("w-full aero-card"):
                    ui.label("Wing at the design lift").classes(
                        "text-base font-semibold")
                    with ui.row().classes("w-full gap-6 items-center "
                                          "flex-wrap"):
                        def wstat(k, v):
                            with ui.column().classes("gap-0"):
                                ui.label(k).classes("stat-label")
                                ui.label(v).classes("text-sm font-mono")
                        wstat("wing L/D", f"{bd.get('LD', 0):.2f}")
                        wstat("CL", f"{bd.get('CL', 0):.3f}")
                        wstat("CD induced",
                              f"{bd.get('CDi', 0) * 1e4:.1f} counts")
                        wstat("CD profile",
                              f"{bd.get('CDp', 0) * 1e4:.1f} counts")
                        wstat("span efficiency e", f"{bd.get('e', 0):.3f}")
                        wstat("root incidence",
                              f"{bd.get('alpha_root_deg', 0):+.2f}°")
                        wstat("tip twist",
                              f"{bd.get('twist_tip_deg', 0):+.2f}°")
                        wstat("max |twist|",
                              f"{bd.get('twist_env_deg', 0):.2f}° "
                              f"(cap {tw.get('twist_max_deg', 0):.1f}°)")
                        wstat("max local α",
                              f"{bd.get('alpha_geo_max_deg', 0):.2f}° "
                              f"(cap {tw.get('alpha_max_deg', 0):.1f}°)")
                        wstat("span · MAC",
                              f"{wg.get('b', 0):.2f} m · "
                              f"{wg.get('mac', 0):.3f} m")
                        if chd.get("order"):
                            wstat("root · tip chord",
                                  f"{bd.get('chord_root_m', 0):.3f} · "
                                  f"{bd.get('chord_tip_m', 0):.3f} m")
                            wstat("taper flown",
                                  f"{bd.get('taper_flown', 0):.3f} "
                                  f"(baseline "
                                  f"{chd.get('taper_baseline', 0):.2f})")
                            wstat("chord deviation",
                                  f"{bd.get('chord_dev', 0):.1%} "
                                  f"(cap {chd.get('chord_max_frac', 0):.0%})")
                    ui.label(
                        "Wing-only L/D: induced drag from the lifting-line "
                        "loading plus the strip integral of the section's own "
                        "viscous polar. No fuselage, tail, nacelle or "
                        "interference drag is included, so this is an upper "
                        "bound on the aircraft L/D."
                        + (f" {bd['n_clamped_low']} tip station(s) sit below "
                           "the converged polar branch and use its endpoint "
                           "cd." if bd.get("n_clamped_low") else "")
                    ).classes("text-xs opacity-60")
                    # the polar is flown at ONE Reynolds number; if the chord
                    # law moved the true MAC far off the baseline, say so
                    if chd.get("order") and bd.get("re_true"):
                        drift = abs(float(bd["re_true"])
                                    / max(float(bd.get("re") or 1.0), 1.0) - 1.0)
                        if drift > 0.05:
                            ui.label(
                                f"the reshaped planform's own MAC implies "
                                f"Re {float(bd['re_true']):.3g}, "
                                f"{drift:.0%} off the Re "
                                f"{float(bd.get('re') or 0):.3g} the section "
                                f"polar was actually flown at — tighten the "
                                f"chord-deviation cap to shrink that gap."
                            ).classes("text-[11px]").style(f"color:{WARN}")
                    sp = fig_opt_span(rep)
                    if sp is not None:
                        ui.plotly(sp).classes("w-full")

            sec = rep.get("section")
            if not isinstance(sec, dict) or not sec.get("design"):
                if best is not None and not bd:
                    ui.label(f"best wing L/D = {best:.3f}").classes(
                        "text-sm font-mono")
                for k in ("section_error", "design_error"):
                    if rep.get(k):
                        ui.label(rep[k]).classes("text-xs").style(
                            f"color:{WARN}")
                return

            des = sec["design"]
            pol = des.get("polar") or {}
            with ui.card().classes("w-full aero-card"):
                ui.label("Optimised section").classes("text-base font-semibold")
                with ui.row().classes("w-full gap-6 items-center flex-wrap"):
                    def ostat(k, v):
                        with ui.column().classes("gap-0"):
                            ui.label(k).classes("stat-label")
                            ui.label(v).classes("text-sm font-mono")
                    ostat("t/c", f"{des['tc']:.4f} "
                                 f"(min {sec['tc_min']:.2f})")
                    ostat("max thickness at", f"{des['tc_max_xc']:.1%} c")
                    ostat(f"c_d at c_l = {sec['cl_design']:.3f}",
                          (f"{pol['cd_at_cl_design'] * 1e4:.1f} counts"
                           if pol.get("cd_at_cl_design") is not None else "—"))
                    ostat("c_m at design c_l",
                          (f"{pol['cm_at_cl_design']:+.4f} "
                           f"(|cap| {sec['cm_max']:.2f})"
                           if pol.get("cm_at_cl_design") is not None else "—"))
                    ostat("XFOIL", f"{pol.get('n_converged', 0)}/"
                                   f"{pol.get('n_requested', 0)} converged "
                                   f"· Re {pol.get('re', 0):.3g}")
                shp = fig_section_shape(sec)
                if shp is not None:
                    ui.plotly(shp).classes("w-full")
                pf = fig_section_polars(sec)
                if pf is not None:
                    ui.plotly(pf).classes("w-full")
                ui.label(
                    "Solid = optimised section; dotted = the library winner "
                    "the box was seeded on. Polars are the live viscous XFOIL "
                    "sweep each candidate was scored with.").classes(
                    "text-xs opacity-60")

            rows = airfoil_compare_rows(rep)
            if rows:
                with ui.card().classes("w-full aero-card"):
                    ui.label("Original vs optimised").classes(
                        "text-base font-semibold")
                    ui.label(
                        "ORIGINAL = the library winner this run was seeded "
                        "on (its CST refit, the design-box anchor), flown "
                        "UNTWISTED at the same derived operating point — so "
                        "every difference below is the optimisation, not a "
                        "change of flight condition.").classes(
                        "text-xs opacity-60 mb-1")
                    with ui.element("div").classes(
                            "w-full grid gap-x-6 gap-y-1").style(
                            "grid-template-columns:2fr 1fr 1fr 1.4fr"):
                        for h in ("", "original (seed)", "optimised",
                                  "change"):
                            ui.label(h).classes("stat-label")
                        for r in rows:
                            ui.label(r["metric"]).classes(
                                "text-xs opacity-70")
                            ui.label(r["original"]).classes(
                                "text-xs font-mono opacity-70")
                            ui.label(r["new"]).classes(
                                "text-xs font-mono font-medium")
                            col = {"better": GOOD, "worse": BAD}.get(r["dir"])
                            lab = ui.label(r["change"]).classes(
                                "text-xs font-mono")
                            if col:
                                lab.style(f"color:{col}")
                            else:
                                lab.classes("opacity-60")

    _sync_wing_preview()
    _sync_chord_note()
    _sync_opt_seed_note()
    render_optimize()

    # =============================================================== COMPARE
    with panels:
        with ui.tab_panel(tab_compare):
            page_header("Compare", "Overlay saved runs — equal configs "
                                   "collapse to median + band")
            with ui.column().classes("w-full p-2 gap-3"):
                with ui.row().classes("w-full items-center gap-2"):
                    ui.button("Refresh", icon="refresh",
                              on_click=lambda: render_compare()) \
                        .props("outline dense")
                    ui.button("Load into Results", icon="north_east",
                              on_click=lambda: _load_selected()) \
                        .props("outline dense")
                    ui.button("Delete selected", icon="delete",
                              on_click=lambda: _confirm_delete()) \
                        .props("outline dense color=negative")
                    ui.label(
                        "Select runs to overlay their convergence; "
                        "same problem+optimiser+budget group into "
                        "median + band.").classes("text-xs opacity-60")
                compare_table_box = ui.column().classes("w-full")
                compare_fig = ui.plotly(
                    _empty_fig("Select at least one run")) \
                    .classes("w-full")

    compare_state = {"table": None}

    def render_compare():
        compare_table_box.clear()
        runs = api.list_runs(str(RESULTS_DIR))
        with compare_table_box:
            if not runs:
                ui.label(f"No saved runs in {RESULTS_DIR} yet — every "
                         "GUI run persists automatically.").classes(
                             "opacity-60 text-sm")
                compare_state["table"] = None
                return
            columns = [
                {"name": "timestamp", "label": "when", "field": "timestamp",
                 "sortable": True, "align": "left"},
                {"name": "problem_name", "label": "problem",
                 "field": "problem_name", "sortable": True, "align": "left"},
                {"name": "optimiser", "label": "optimiser",
                 "field": "optimiser", "sortable": True, "align": "left"},
                {"name": "budget", "label": "budget", "field": "budget",
                 "sortable": True},
                {"name": "seed", "label": "seed", "field": "seed",
                 "sortable": True},
                {"name": "best_score", "label": "best", "field": "best_disp",
                 "sortable": True},
                {"name": "feasible", "label": "feasible", "field": "feasible"},
            ]
            for r in runs:
                r["best_disp"] = _fmt(r.get("best_score"))
            table = ui.table(columns=columns, rows=runs,
                             row_key="path",
                             selection="multiple",
                             pagination=12).classes("w-full aero-card")
            table.on("selection", lambda _: _update_overlay())
            compare_state["table"] = table

    def _selected_paths() -> list[str]:
        t = compare_state.get("table")
        if t is None:
            return []
        return [r["path"] for r in (t.selected or [])]

    def _update_overlay():
        paths = _selected_paths()
        if not paths:
            compare_fig.update_figure(_empty_fig("Select at least one run"))
            return
        groups: dict[tuple, list] = {}
        singles = []
        for p in paths:
            try:
                rd = api.load_run(p)
            except Exception:
                continue
            key = (rd.get("problem_name"), rd.get("optimiser"),
                   rd.get("budget"))
            groups.setdefault(key, []).append(rd)
        fig = go.Figure()
        palette = [ACCENT, WARN, GOOD, "#a78bfa", "#f472b6", "#f87171"]
        ci = 0
        for key, rds in groups.items():
            color = palette[ci % len(palette)]
            ci += 1
            name = f"{key[0]} · {key[1]} (b{key[2]})"
            if len(rds) >= 2:
                n = min(len(_nan_history(r.get("history"))) for r in rds)
                H = np.vstack([_nan_history(r.get("history"))[:n]
                               for r in rds])
                med = np.nanmedian(H, axis=0)
                lo = np.nanmin(H, axis=0)
                hi = np.nanmax(H, axis=0)
                xs = np.arange(1, n + 1)
                fig.add_trace(go.Scatter(
                    x=np.concatenate([xs, xs[::-1]]),
                    y=np.concatenate([hi, lo[::-1]]),
                    fill="toself", fillcolor=color.replace(")", ",0.15)")
                    .replace("rgb", "rgba") if color.startswith("rgb")
                    else BAND,
                    line=dict(width=0), name=f"{name} range",
                    showlegend=False, hoverinfo="skip"))
                fig.add_trace(go.Scatter(
                    x=xs, y=med, mode="lines",
                    name=f"{name} · median of {len(rds)}",
                    line=dict(color=color, width=2.4)))
            else:
                rd = rds[0]
                h = _nan_history(rd.get("history"))
                fig.add_trace(go.Scatter(
                    x=np.arange(1, h.size + 1), y=h, mode="lines",
                    name=f"{name} · seed {rd.get('seed')}",
                    line=dict(color=color, width=2)))
            singles.extend(rds)
        fig.update_layout(**_base_layout(
            "Best-so-far overlay", "evaluation", "best feasible objective",
            h=380))
        compare_fig.update_figure(fig)

    def _load_selected():
        paths = _selected_paths()
        if len(paths) != 1:
            ui.notify("Select exactly one run to load", type="warning")
            return
        try:
            set_result(api.load_run(paths[0]))
            goto("Results")
        except Exception as exc:
            ui.notify(f"Could not load run: {exc}", type="negative")

    def _confirm_delete():
        paths = _selected_paths()
        if not paths:
            ui.notify("Nothing selected", type="warning")
            return
        with ui.dialog() as dlg, ui.card():
            ui.label(f"Delete {len(paths)} saved run file(s)? "
                     "This cannot be undone.")
            with ui.row():
                def _do():
                    for p in paths:
                        try:
                            Path(p).unlink()
                        except OSError:
                            pass
                    dlg.close()
                    render_compare()
                    _update_overlay()
                    ui.notify("Deleted", type="warning")
                ui.button("Delete", color="negative", on_click=_do)
                ui.button("Cancel", on_click=dlg.close).props("flat")
        dlg.open()

    render_compare()

    # ---- initial render of the design page
    render_builder()
    render_solver_banner()
    render_bounds()
    render_mission()
    render_physics()
    render_optimisers()


# ================================================================= entry

def _run_app():
    from nicegui import ui
    native = "--browser" not in sys.argv
    port = 8765
    for i, a in enumerate(sys.argv):
        if a == "--port" and i + 1 < len(sys.argv):
            port = int(sys.argv[i + 1])
    main()
    ui.run(
        title="AeroBO Studio",
        favicon="🛩️",
        dark=True,
        native=native,
        window_size=(1500, 940) if native else None,
        reload=False,
        port=port,
        show=not native,
    )


if __name__ in {"__main__", "__mp_main__"}:
    _run_app()
