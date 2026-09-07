"""Stage 2 (and 2.5) — AIRFOIL: pick a section from the library, then refine it.

Two steps, in that order, because they answer different questions:

1. SCREENING picks the best KNOWN section for the mission's design point.
   The UIUC database is scored on six criteria under the user's weights at
   the mission's design Cl; on the cached operating point it is instant.
2. SHAPE OPTIMISATION designs a NEW section: 8 CST weights (plus a twist
   law, and optionally a chord law, in wing mode) searched with a live
   viscous XFOIL sweep per candidate, seeded from the screening pick so the
   search starts on a section that already works.

Step 2 is optional — that is the point of ordering them this way. A run can
also hand the section decision to the wing solver entirely ("use the wing
family's own section"), which is the honest option for the families whose
polars are fixed.

ONE SECTION PER SURFACE, AND ONE STAGE PER SECTION. Where the configuration
carries a second surface — a tail, an elevator, or a tandem pair's rear wing
— this module is BUILT TWICE: stage 2 for the wing, stage 2.5 for that
surface, each with its own workspace (``session.airfoil_state``) and so its
own screening report, ranking, weights and optimiser trace. The two
questions have different answers because they have different design points:
a tail's chord is not the wing's, and its design lift is not the mission's
(it TRIMS, so it is screened at the lift its family's own moment balance
implies — ``session.trim_lift``), while a tandem pair's two wings
split one reference area over the same span and so fly different chords at
the same lift coefficient. The surface is the STAGE's, not a setting inside
it, so nothing on screen can be an answer about the other one.

The criterion weights each stage OPENS on are the ones recommended for that
surface's job (``session.recommended_weights``) — GDP's own front-wing and
rear-wing presets on a tandem, and its symmetric preset on a trimming
surface. Editing them makes them the user's, and the shell then never moves
them again.

This stage also owns the ASPECT-RATIO ESTIMATE. A section cannot be screened
without a Reynolds number, a Reynolds number needs a chord, and the mission
states no planform — so the one number that turns the mission's area into a
chord lives here, next to the section it exists for, and is never sent to a
solver (``session.section_aspect_ratio``). Stage 3 decides the span the run
actually flies.

Everything numerical comes from ``aerobo.api``; the shapes and polars are
drawn with V1's pure figure builders. The workers are daemon threads that
only ASSIGN into their own surface's workspace; the shell's heartbeat
re-renders.
"""

from __future__ import annotations

import math
import threading
import time

import numpy as np
import plotly.graph_objects as go

from gui import diagnose, metrics as metric_catalogue

from .. import config, figstyle, sampler as sampler_mod, session, theme, \
    widgets

#: screening criterion -> (label, what it means)
WEIGHT_META = {
    "ldcr": ("L/D at design Cl", "the cruise number the wing actually flies"),
    "clmax": ("Cl max", "how much lift the section reaches before stall"),
    "cm": ("|Cm| (lower better)", "pitching moment — trim drag and structure"),
    "ldmax": ("(L/D) max", "the best point of the polar, wherever it sits"),
    "cdcr": ("cd at design Cl (lower better)",
             "the drag this surface actually costs at the lift it flies — "
             "and the only drag criterion left on a surface that flies at "
             "ZERO lift, where both L/D criteria are read at a lift it "
             "never carries"),
    "thick": ("thickness t/c", "structural depth: spar height and volume"),
    "astall": ("stall angle", "how much angle-of-attack margin there is"),
}

#: criteria that CANNOT rank this surface, and the sentence saying why. The
#: rows stay on the form — "nobody can weight this" and "you set this to zero"
#: are different sentences and the shell makes both (UNWEIGHTED_STYLE) — but a
#: user who moves one of these off zero has to be told what they bought.
DEAD_CRITERIA = {
    "fin": {
        "ldcr": "“L/D at design Cl” is cl/cd, and a fin makes no side force "
                "at zero sideslip — so it is 0 for every candidate and this "
                "weight ranks nothing.",
        "ldmax": "“(L/D) max” is read at whatever lift maximises cl/cd, "
                 "which is a lift this surface never carries. Over the "
                 "symmetric library it ranks sections almost independently "
                 "of the drag they cost (Spearman +0.077 against −cd at zero "
                 "lift). Weight “cd at design Cl” instead — at zero lift "
                 "that IS the zero-lift drag.",
        "cm": "a symmetric section's |Cm| about the quarter chord is zero "
              "identically, so this weight ranks noise.",
    },
}

#: THE ENDPLATE IS THE SAME SURFACE, one vehicle across: a vertical panel at
#: nominally zero incidence whose section buys drag and stiffness and nothing
#: else. The same three criteria are dead on it, and for the same reasons — so
#: the entry is SHARED rather than copied, because two copies of one argument
#: drift and the sentences beside a weight are the argument.
DEAD_CRITERIA["plate"] = DEAD_CRITERIA["fin"]

#: what a gate is sent as when it is switched OFF: no minimum thickness,
#: and a moment ceiling nothing can exceed. Both are the idiom the screen
#: already understands, so "no gate" needs no new API.
GATE_OFF = {"tc_min": 0.0, "cm_max": 1.0e9}

#: what a gate comes back ON at — the published screening values
GATE_DEFAULTS = {"tc_min": 0.15, "cm_max": 0.08}

#: the shape optimiser's own published gates (AirfoilProblem's defaults)
OPT_GATE_DEFAULTS = {"tc_min": 0.10, "cm_max": 0.08}

#: what a run stopped by the STOP BUTTON records as its ``stop_reason``. A
#: stopped run is a partial run, not a failed one: the evaluations it paid
#: for are kept, its incumbent IS a real evaluated section, and the record
#: has to say which rule ended it — the plateau or the user.
STOP_REASON = "you stopped it"

#: how many finished sections the live sweep log keeps, and how many of them
#: are on screen at once. The default shortlist is 24, so the cap only bites
#: on a user who asked for a much longer one — and a live log that grows
#: without bound is a leak in a view that redraws twice a second.
SWEEP_LOG_MAX = 400
SWEEP_LOG_SHOWN = 8

FLOOR_META = {"clmax": ("min Cl max", 0.05),
              "ldcr": ("min L/D at Cl", 1.0),
              "astall": ("min stall angle [deg]", 0.5)}

#: ranking table: (report key, column label, significant digits). The digit
#: counts are chosen so an L/D of 130 reads as "130" and not as "1.3e+02" —
#: _fmt counts SIGNIFICANT figures, so 2 was too few for the ratios.
RANK_COLUMNS = [
    ("rank", "#", 0), ("name", "section", None), ("composite", "score", 4),
    ("ldcr", "L/D @Cl", 4), ("ldmax", "(L/D)max", 4), ("clmax", "Cl max", 3),
    ("astall", "α stall", 3), ("tc", "t/c", 4), ("cm_at", "Cm", 3),
    # the raw drag at the design lift — the ``cdcr`` criterion's own metric.
    # It used to be the one column no weight priced, which is why a surface
    # that flies at ZERO lift (a fin: both L/D criteria dead) had nothing to
    # be ranked on but its thickness and its stall angle. On a trimming
    # surface — ranked at ONE point of a range it works either side of — it
    # is still the column to read beside the score (session.TRIM_DRAG_NOTE)
    ("cd_at", "cd @Cl", 4),
]

#: ranking column -> the screening criterion whose weight prices it. Three of
#: them are not spelled the same on both sides — t/c is scored as ``thick``,
#: the drag column as ``cdcr``, and the moment column carries the SIGNED cm at
#: the design lift while the criterion scores |cm| — so this is a table and
#: not a name match. A column missing from it (the rank, the name, the score
#: itself) is not a weighted criterion AT ALL, and is shown with no weight
#: rather than a zero one: "nobody can weight this" and "you set this to
#: zero" are different sentences and the table has to make both.
CRITERION_OF_COLUMN = {"ldcr": "ldcr", "ldmax": "ldmax", "clmax": "clmax",
                       "astall": "astall", "tc": "thick", "cm_at": "cm",
                       # the drag column stopped being unweightable: it is
                       # the ``cdcr`` criterion now, and it is what a surface
                       # flying at zero lift is ranked on
                       "cd_at": "cdcr"}


def _join_names(names) -> str:
    """``a``, ``a and b``, ``a, b and c`` — an English list, for a sentence
    that names a variable number of criteria."""
    names = [str(n) for n in names]
    if len(names) <= 1:
        return names[0] if names else ""
    return f"{', '.join(names[:-1])} and {names[-1]}"


#: how a criterion nobody weighted is drawn: dimmed, column and header. Not
#: dropped — the number is still true and still worth reading — and not left
#: bright, which is what made a zero-weight column look like it had ranked
#: something.
UNWEIGHTED_STYLE = "opacity:0.45"

#: a criterion cell: the raw metric, and faint beside it the POINTS that
#: metric puts into the score column. Two numbers in one cell rather than six
#: more columns, because they are one fact read together — "L/D is 83.9,
#: which is +18.4 of this section's score".
POINTS_TPL = """
<q-td :props="props" class="text-right">
  <span>{{ props.value }}</span>
  <span style="color:__FAINT__;font-size:0.82em;margin-left:6px">{{
    props.row.__FIELD__ }}</span>
</q-td>
"""


def weighted_points(row: dict, weights: dict) -> dict:
    """What each criterion of ONE ranked row contributed to its score.

    The score column is a weighted sum of six 0-100 sub-scores
    (``airfoil_select.score_candidates``) and the columns beside it are the
    raw metrics those sub-scores were built from — so the table said what the
    section IS and what it scored in total, and nothing about WHERE the score
    came from. This is that missing middle: ``w_i * s_i`` per criterion, in
    the score's own units, summing to the score itself.

    ``weights`` are the report's OWN — already normalised, which is what makes
    the six terms add up to the composite exactly. Never the live sliders: the
    user is free to move those after the screen ran, and pricing this table
    with them would explain an order they did not produce.

    Cells, not numbers: ``""`` where the criterion carries no weight (the term
    is exactly zero, and the dimmed column is what says so), ``"—"`` where the
    sub-score was not measured, and SIGNED otherwise — sub-scores are not
    clipped to the frozen band, so a section outside it contributes a negative
    number of points and the sign is the interesting part.
    """
    if not weights:
        # a report that does not carry the weights it was ranked on (an old
        # saved run) gets NO prices rather than zeros: "we cannot say what
        # this column contributed" and "it contributed nothing" are the two
        # sentences this whole cell exists to keep apart.
        return {f"pts_{col}": "" for col in CRITERION_OF_COLUMN}
    scores = row.get("scores") or {}
    out = {}
    for col, crit in CRITERION_OF_COLUMN.items():
        w = float(weights.get(crit, 0.0) or 0.0)
        s = scores.get(crit)
        out[f"pts_{col}"] = ("" if w <= 0.0
                             else "—" if s is None
                             else f"{w * float(s):+.1f}")
    return out


def ranking_columns(cols, weights: dict) -> list[dict]:
    """The ranking table's columns, priced by the weights it was ranked on.

    Each weighted criterion's header carries its weight (normalised, so the
    six read as fractions of one score), and a criterion at zero weight is
    drawn dim — it took no part in the order the rows are in, and a column
    that ranked nothing should not look like one that did.
    """
    out = []
    for key, label, _nd in cols:
        # no weights on the report (see weighted_points): every column is
        # drawn as a plain metric, none of them dimmed. Dimming all six would
        # say the ranking used nothing, which is the one thing that is false.
        crit = CRITERION_OF_COLUMN.get(key) if weights else None
        w = None if crit is None else float(weights.get(crit, 0.0) or 0.0)
        col = {"name": key, "field": key,
               "label": label if w is None else f"{label} · w {w:.2f}",
               "align": "left" if key == "name" else "right",
               "sortable": key not in ("rank",)}
        if w is not None and w <= 0.0:
            col["style"] = UNWEIGHTED_STYLE
            col["headerStyle"] = UNWEIGHTED_STYLE
        out.append(col)
    return out


def points_slots(table, cols):
    """Draw metric + contributed points in every weighted criterion cell."""
    for key, _label, _nd in cols:
        if key in CRITERION_OF_COLUMN:
            table.add_slot(f"body-cell-{key}",
                           widgets.paint(POINTS_TPL)
                           .replace("__FIELD__", f"pts_{key}"))
    return table


def stop_or_converged(converged, cancelled):
    """THE STOP BUTTON AND THE CONVERGENCE RULE ARE ONE STOP RULE.

    ``converged(i, best) -> bool`` is stage 1's adaptive stop, or None when
    nothing asked for one; ``cancelled() -> bool`` reads the Stop button.
    Returns the ``api.run(stop_rule=…)`` the section search flies.

    Stop used to raise ``_Cancelled`` out of the progress callback, which
    unwound :func:`api.run` before it could assemble anything — so pressing it
    threw away every XFOIL sweep the search had already paid for, INCLUDING
    THE BEST SECTION IT HAD FOUND, and left the stage with a trace and no
    aerofoil. As a stop rule instead, ``api.run`` ends the search between
    evaluations and rebuilds the result from its own per-evaluation log
    (:func:`api.partial_result`): the run comes back with ``partial=True``,
    the incumbent as ``best_x`` and therefore a ``section`` block — a real
    evaluated aerofoil the stage can score and adopt like any other.

    The rule is passed on EVERY run, not only when the adaptive stop is on:
    without one the button had nothing to fire. It changes no number — it is
    applied between evaluations and answers False until something asks to
    stop, which is the legacy path exactly.
    """
    def _rule(i, best) -> bool:
        if cancelled():
            return True
        return bool(converged(i, best)) if converged is not None else False
    return _rule


def optimised_weights(rep: dict):
    """``(w_upper, w_lower)`` of an optimised section report, or None.

    Two sources, both the run's OWN numbers — the weights are never re-fitted
    from the drawn coordinates, which would introduce a second, slightly
    different section:

    * the section report, which carries the weights its shape was drawn from
      (``section.design.w_upper`` / ``w_lower``);
    * failing that, the design vector, read through the labels of the BUILT
      problem (``result.param_labels``).

    NOT ``PROBLEM_SPECS[name].param_labels``: the section problem's labels are
    derived at BUILD time from ``n_cst``, so the spec's are empty. Reading
    them there returned no weights at all, which left the optimised section
    travelling to stage 3 as its display name ("CST section (optimised)") and
    raising ``FileNotFoundError`` in the library loader.
    """
    design = (rep.get("section") or {}).get("design") or {}
    w_u, w_l = design.get("w_upper"), design.get("w_lower")
    if w_u and w_l:
        return [float(v) for v in w_u], [float(v) for v in w_l]
    res = rep.get("result") or {}
    x = res.get("best_x") or []
    labels = [str(lbl) for lbl in (res.get("param_labels") or [])]
    if not x or len(labels) != len(x):
        return None
    by = {lbl: float(v) for lbl, v in zip(labels, x)}
    w_u = [by[k] for k in labels if k.startswith("w_upper_")]
    w_l = [by[k] for k in labels if k.startswith("w_lower_")]
    return (w_u, w_l) if w_u and w_l else None


def OBJECTIVE_CHOICES(wing_mode: bool) -> dict:
    """The scalars this surface's search can maximise, as a select's options.

    ``cd`` is THE SURFACE'S OWN PHYSICAL OBJECTIVE — a wing L/D where the
    medium and the surface make that meaningful (``session.wing_objective``),
    the section's own 2-D L/D otherwise. It is one key because it is one
    question the user does not get asked: which of the two is honest here is
    decided by the configuration, not by preference.

    ``composite`` is the other question, and it IS the user's: the screen
    ranks the library on six weighted criteria and the search then maximises
    one number, so five of the six stop counting the moment stage 3 starts
    (report §15.4). Picking it optimises the SAME weighted composite the
    ranking is built from.

    ``composite_goal`` is that same score with the SEED under it as a floor.
    A weighted sum is free to sell one criterion to buy another, and measured
    on the frozen study it does: the winner gains 11 points of J while cruise
    L/D falls 12 % and section drag rises 14 %, because over the shipped band
    0.01 of |Cm| is worth four counts of L/D. This one pays the same J and
    subtracts what falling BELOW the seed costs, so "better score" and "better
    section" stop being two different things.

    ``composite_asf`` goes one further and changes the FUNCTION, not its floor.
    A weighted sum — with or without a penalty under it — can only ever return
    a design on the convex hull of what the box can reach, so whole families of
    compromise sections are invisible to it at every weight the user could
    type. This one scores the WORST of the six criteria relative to the seed
    (an augmented Tchebycheff function), so a criterion cannot be sold to buy
    another at any exchange rate, and the non-convex compromises become
    reachable. Its number is not a J and does not compare with the other two:
    the card re-scores every winner on the plain composite for that.

    ``pareto`` is the one entry that does not return a winner. It searches
    three of the criteria as THREE (constrained qNEHVI) and hands back the set
    of designs none of which beats another everywhere — so the question it
    answers is "what am I giving up?", not "what is best?". It is offered on
    measured grounds and its limits are measured too
    (``RESULTS_FRONT_REACHABILITY.md``, 42 seeds): **41 of 42 seeds carry a
    front design that no scalarised winner dominates** (*p* = 2e-11), which is
    the reachability argument shown rather than argued — but at the SAME
    budget the front is a **worse single-answer search** than
    ``composite_goal`` (less hypervolume against the seed, 8/36, sign
    *p* = 0.0012). Both halves are true, so the menu says both: pick it to see
    the trade, not to get one section. The label carries the warning because a
    user who picks it expecting a better winner has been mis-sold.
    """
    return {
        "cd": ("wing L/D (section + twist law)" if wing_mode
               else "2-D L/D at the design Cl"),
        "composite": "composite score of your six criteria",
        "composite_goal": "composite score, no WEIGHTED criterion below the "
                          "seed",
        "composite_asf": "lift the WEAKEST of your six criteria (Tchebycheff)",
        "pareto": "show the whole trade-off front (not one winner — a worse "
                  "single answer at the same budget)",
    }


#: the objectives that score the SIX CRITERIA rather than one physical number.
#: One name for the set, because every branch that asks "is this a composite
#: run?" is asking about the extra stall sweep, the 2-D design vector and the
#: score card — all of which the goal and ASF objectives share exactly.
#: ``pareto`` is here for exactly the reason the others are: it wants the same
#: extra stall sweep, the same 2-D design vector and the same score card. What
#: it does NOT share is a single winner, and that difference lives on the
#: result card, not in this tuple.
COMPOSITE_OBJECTIVES = ("composite", "composite_goal", "composite_asf",
                        "pareto")

#: …and the subset that carries a REFERENCE POINT, which is what the result
#: card's working block reads.
REFERENCE_OBJECTIVES = ("composite_goal", "composite_asf")


def is_composite(objective) -> bool:
    """True for any objective built out of the screening composite."""
    return str(objective) in COMPOSITE_OBJECTIVES


def objective_kwargs(objective: str, *, wing_guess, weights, reference,
                     censored=None) -> dict:
    """The ``api.optimize_airfoil`` arguments one objective implies.

    ONE place decides three coupled things, because they were three
    opportunities to disagree: which scalar is maximised, whether the run flies
    a wing (a composite run never does — J scores a 2-D section, so a twist law
    would be searched against a number that cannot see it, and the api refuses
    the pair), and which weights + frozen band J is measured on (nothing at all
    for a -cd run, which must keep sending the legacy argument set).

    ``composite_goal`` sends the same three plus nothing else: the goals are
    left unstated so the api measures them off the run's OWN seed, which is the
    only reference point that is still right after the user changes the anchor.

    ``censored`` travels ONLY on a composite run. ``api._airfoil_censored``
    raises if a non-default mode reaches a ``-cd`` run, and rightly: the
    censoring policy is a statement about the SCREENING COMPOSITE's stall
    criteria, and a drag run has none. It is the one lever that turns an
    unscoreable reference design into a scoreable one — a seed whose stall
    march died early is refused outright under the default, and scored at the
    bound (a true lower bound on J for any non-negative weights) under
    ``lower_bound``.
    """
    if is_composite(objective):
        out = {"objective": str(objective), "wing": None,
               "score_weights": dict(weights),
               "score_reference": reference}
        if censored:
            out["censored"] = str(censored)
        return out
    return {"objective": "cd", "wing": wing_guess}


def screen_names_for(surface: str):
    """Which library members THIS surface's screen may rank, or None for all.

    Module level and given only the surface, for the reason
    :func:`shape_kwargs` is: the argument list a run flies has to be
    assertable without a browser. A restriction applied inline at the call
    site is one no test can see, and the defect this exists for — a cambered
    GOE741 winning a vertical stabiliser's screen — was exactly a rule that
    lived in one branch of one handler.

    A fin may only be given a SYMMETRIC section: at zero sideslip it must
    make no side force, so a cambered one is a permanent side load the trim
    solve cannot balance. Measured from the coordinates
    (``api.section_max_camber``), never from the name.

    A CAR ENDPLATE is restricted for the same physics stated one vehicle
    across: ``endplate.py``'s whole "section trap" is that a vertical panel
    is built at theta = twist − alpha_L0, so a cambered plate at zero toe
    carries a side force — and the two plates' loads cancel in CY, which is
    what would have made it invisible. ``CarWingEndplateProblem`` refuses one
    outright; this is the half that stops it being offered.

    Raises where the coordinate sidecar is missing rather than returning
    None, because "cannot restrict" and "no restriction" are opposite
    answers and only one of them is safe.
    """
    from aerobo import api

    if surface not in ("fin", "plate"):
        return None
    names = api.symmetric_section_names()
    if not names:
        raise RuntimeError(
            "the section coordinate cache is not built, so the library "
            "cannot be restricted to symmetric sections — and neither a "
            "vertical stabiliser nor a car endplate may be given a cambered "
            "one. Run the wing's screen once to build it.")
    return names


def shape_kwargs(S: dict, surface: str, opt: dict, weights: dict,
                 reference: dict | None) -> dict:
    """Every argument that says WHICH SEARCH a section run is.

    ONE dict, handed to both api calls a run makes — and kept afterwards as
    that run's LAUNCH SNAPSHOT, because a snapshot is what "Continue" re-flies
    (:func:`session.continue_section`). Two hand-copied argument lists is how
    a live sampler's configuration and the search it samples come to disagree,
    and it is also how a "continuation" would quietly become a different
    search: the form can move between the run and the button.

    Module level, and given its state rather than reading a closure, so the
    argument list a run flies can be built and asserted without a browser.
    """
    cond = session.section_conditions(S, surface)
    anchor = session.section_weights(S, surface)

    def gate(key: str) -> float:
        v = opt.get(key)
        return float(GATE_OFF[key] if v is None else v)

    return dict(
        re=float(cond["re"]), mach=float(cond["mach"]),
        cl_design=float(cond["cl_design"]),
        # A VERTICAL STABILISER SEARCHES A SYMMETRIC FAMILY, and it is stated
        # HERE because this dict is what a run IS — including the run a
        # "Continue" re-flies. Setting it at the call site instead would let
        # a continuation resume a fin's search over cambered shapes.
        symmetric=(surface in ("fin", "plate")),
        tc_min=gate("tc_min"), cm_max=gate("cm_max"),
        anchor=(list(anchor) if anchor else None),
        twist_order=int(opt["twist_order"]),
        twist_max_deg=float(opt["twist_max_deg"]),
        alpha_max_deg=float(opt["alpha_max_deg"]),
        chord_order=int(opt["chord_order"]),
        chord_max_frac=float(opt["chord_max_frac"]),
        **objective_kwargs(
            str(opt.get("objective", "cd")),
            wing_guess=(session.wing_guess(S, surface)
                        if session.wing_objective(S, surface) else None),
            weights=dict(weights), reference=reference,
            censored=opt.get("censored")))


#: what a RANKED ROW takes from the screen's refit pass. The weights are the
#: shape that will FLY; ``coords`` is the section itself, blunt trailing edge
#: and all. Carrying only the weights is how the card came to draw a sharp TE
#: onto an aerofoil that does not have one.
CANDIDATE_KEYS = ("w_upper", "w_lower", "coords", "te_gap")


def section_outline(sec: dict):
    """The loop to DRAW for a chosen section: its own, or its CST refit.

    A database section is drawn from its own coordinates, because a picture of
    an aerofoil should be that aerofoil — and this database has blunt ones.
    ``cst_anchor_from_coords`` drops the fitted TE gap (the design box fixes
    ``dz_te`` at zero), so drawing the refit closed MS3-15Retro's 0.0079c
    trailing edge to a point and showed the user a section nobody has.

    The refit remains the fallback, and remains what stage 3 flies: a section
    with no stored outline (an optimised shape, an older session's pick) is
    drawn from its weights exactly as before. Returns ``None`` when there is
    neither.
    """
    import numpy as np

    from aerobo import airfoil as af

    raw = (sec or {}).get("coords")
    if raw is not None:
        arr = np.asarray(raw, dtype=float)
        if arr.ndim == 2 and arr.shape[0] >= 3 and arr.shape[1] == 2:
            return arr
    try:
        out = np.asarray(af.cst_coords(sec["w_upper"], sec["w_lower"]),
                         dtype=float)
    except (ValueError, TypeError, KeyError):
        return None
    # `cst_coords` does not raise on absent weights — it multiplies None-shaped
    # arithmetic into NaN and hands back a full-length loop of them, which the
    # plot renders as an aerofoil-shaped hole. A shape that cannot be drawn is
    # None here, so the card says "no coordinates" instead.
    return out if out.size and np.isfinite(out).all() else None


def merge_candidate(row: dict, candidates) -> dict:
    """A ranked screening row, plus everything its refit knows about it.

    The two halves are produced by different passes — the ranking is scored
    on the database coordinates, the refit is the CST projection stage 3 will
    fly — and they are joined by name here, once.
    """
    out = dict(row)
    for cand in (candidates or []):
        if cand.get("name") == out.get("name"):
            for key in CANDIDATE_KEYS:
                out[key] = cand.get(key)
            break
    return out


def _score_cell(v, nd: int = 1, signed: bool = False) -> str:
    """One score cell: a number, or an em dash when it was not measured."""
    if v is None:
        return "—"
    return f"{float(v):+.{nd}f}" if signed else f"{float(v):.{nd}f}"


#: the verdict a sub-score change earns, and the slot rule that paints it.
#: Both live outside this stage — stage 3's own seed-vs-optimised table asks
#: the identical question, and two copies of a colour rule is how the two
#: tables on THIS card came to declare different field names.
_verdict = metric_catalogue.verdict
verdict_slots = widgets.verdict_slots


def front_rows(front_report: dict | None) -> list[dict]:
    """An ``api.pareto_airfoil`` front as table rows, ranked, seed included.

    One row per non-dominated design plus the SEED itself, marked, because a
    front the user picks off is only readable against the section they already
    have — the same reference point every objective in this project measures
    from. The seed sits in rank order with the rest rather than in a legend.

    Every row carries all six criteria and the plain composite J. The search
    optimised three of them; the other three are exactly what a user choosing
    a point needs to see, and dropping them here would be the four-row
    comparison bug (§ the compare table) one screen over.

    ``Δ`` columns are against the seed, in sub-score points, so "what does this
    point cost me" is a number rather than an inference. Pure: no NiceGUI, so
    the ordering and the arithmetic are testable without a browser.
    """
    rep = front_report or {}
    rows = rep.get("front") or []
    seed = rep.get("seed") or {}
    if not rows:
        return []
    seed_obj = seed.get("objectives") or {}
    keys = rep.get("conditions", {}).get("criteria") or ["ldcr", "clmax", "cm"]

    def row(src, *, is_seed: bool, rank) -> dict:
        obj = src.get("objectives") or {}
        out = {"rank": ("seed" if is_seed else str(int(rank) + 1)),
               "is_seed": bool(is_seed),
               "composite": _score_cell(src.get("composite"), 2)}
        for k in keys:
            v, s0 = obj.get(k), seed_obj.get(k)
            out[k] = _score_cell(v, 1)
            out[f"d_{k}"] = ("—" if v is None or s0 is None
                             else _score_cell(float(v) - float(s0), 1,
                                              signed=True))
        raw = src.get("raw") or {}
        out["raw"] = ", ".join(
            f"{k} {float(raw[k]):.4g}" for k in keys
            if raw.get(k) is not None)
        return out

    out = [row(r, is_seed=False, rank=r.get("rank", i))
           for i, r in enumerate(rows)]
    out.append(row(seed, is_seed=True, rank=0))
    return out


def score_rows(score: dict | None) -> list[dict]:
    """SEED vs OPTIMISED on the SCREENING criteria, as table rows.

    The shape optimiser maximises ONE number (section cd at the design lift,
    or wing L/D in wing mode). The weights on the screening tab asked for six.
    These rows are the other five's answer: the composite the ranking is built
    from, recomputed for the run's own seed and for what it returned, on the
    SAME frozen band (``api.score_optimised_section``) so the two are
    comparable and neither is normalised against the other.

    Field names match the metric table's (``metric`` / ``original`` / ``new``
    / ``change``) — the column declarations and the row keys are one contract
    on this card, and they have already drifted apart once. A weight of zero is
    kept, because "this counted for nothing" is information the composite's own
    arithmetic hides — and so is a criterion whose SUB-SCORE could not be
    measured but whose raw metric was: it is dropped only when neither section
    produced a number, which is the difference between "not measured" and
    "not shown".

    The number columns are SUB-SCORES (0-100 on the frozen band, which is what
    J is a weighted sum of); the criterion column carries the raw values behind
    them, because "L/D at the design Cl fell 83.9 → 73.6" is the sentence a
    reader needs and "-18.9 points" is the sentence the objective needs. A
    criterion that ended below the seed is marked, since that is the whole
    question this table exists to answer.

    THE MARK IS FOR A WEIGHTED CRITERION ONLY. "▼ below the seed" reads as a
    complaint about the search, and it is only a complaint where the objective
    was ever defending that criterion: J is a weighted sum, ``composite_goal``
    prices a shortfall at ``penalty x weight x shortfall`` (zero at weight 0)
    and ``composite_asf`` drops a zero-weight criterion from its aggregates
    outright. So a criterion nobody weighted gets the hollow "▽ … (not
    weighted)" instead — the fact is still shown, the accusation is not made —
    and its ``dir`` is ``""``, so a card colouring by verdict paints it the
    neutral grey rather than the red it never earned.

    ``dir`` per row is the same four-valued verdict ``_cmp_row`` returns
    (``better`` / ``worse`` / ``same`` / ``""``): a sub-score is on a 0-100
    band where up is always better, so the sign of the change IS the verdict.
    """
    if not score:
        return []
    from aerobo import api

    seed = score.get("seed") or {}
    opt = score.get("optimised") or {}
    delta = score.get("delta") or {}
    weights = score.get("weights") or {}
    rows = [{
        "metric": "composite score",
        "original": _score_cell(seed.get("composite"), 2),
        "new": _score_cell(opt.get("composite"), 2),
        "change": _score_cell(delta.get("composite"), 2, signed=True),
        "dir": _verdict(delta.get("composite")),
    }]
    d_scores = delta.get("scores") or {}
    #: the record field each scored criterion reads, for the raw-value caption
    raw_of = {"thick": "tc", "clmax": "clmax", "ldmax": "ldmax",
              "ldcr": "ldcr", "astall": "astall", "cm": "cm_at",
              "cdcr": "cd_at"}
    nd_of = {"thick": 4, "clmax": 3, "ldmax": 1, "ldcr": 1, "astall": 1,
             "cm": 4, "cdcr": 5}
    for key, label in api.SCREEN_METRICS:
        s_val = (seed.get("scores") or {}).get(key)
        o_val = (opt.get("scores") or {}).get(key)
        s_raw = (seed.get("metrics") or {}).get(raw_of.get(key, key))
        o_raw = (opt.get("metrics") or {}).get(raw_of.get(key, key))
        if s_val is None and o_val is None and s_raw is None and o_raw is None:
            continue
        w = float(weights.get(key, 0.0))
        nd = nd_of.get(key, 3)
        raw_txt = ""
        if s_raw is not None or o_raw is not None:
            raw_txt = (f" ({_score_cell(s_raw, nd)} → "
                       f"{_score_cell(o_raw, nd)})")
        d = d_scores.get(key)
        below = d is not None and d < 0.0
        weighted = w > 0.0
        mark = ("" if not below
                else " ▼ below the seed" if weighted
                else " ▽ below the seed (not weighted)")
        rows.append({
            "metric": f"{label}{raw_txt} · weight {w:.2f}{mark}",
            "original": _score_cell(s_val),
            "new": _score_cell(o_val),
            "change": _score_cell(d, signed=True),
            "dir": _verdict(d) if weighted else "",
        })
    return rows


def shares_section_with(fig, surface_name: str):
    """Identity. The second surface's mirrored outline is NO LONGER DRAWN.

    It used to be overlaid on the wing's Shape panel as a dashed second
    curve — "as the {surface} flies it — INVERTED". Removed on the user's
    instruction: the wing stage draws the WING's aerofoil, and one panel
    carrying two outlines of two different surfaces is the one-question-one-
    place rule broken on the busiest picture in the shell.

    Nothing about the physics moved. The stabiliser still flies this section
    mirrored (``aerobo.polar.InvertedPolar``), the geometry views, the STL
    and the OpenVSP model are still built from the mirrored one, and the
    hint under the panel still SAYS so in words — which is where a fact
    about a different surface belongs.

    Kept as a named no-op rather than deleted at the call site so the reason
    has somewhere to live and a future reader does not re-add the trace.
    """
    return fig


def mirror_section_report(rep: dict, inverted: bool) -> dict:
    """A screening report restated in the AIRCRAFT's axes.

    Mirroring an aerofoil about its chord line mirrors its whole polar
    exactly — ``cl(a) = -cl0(-a)``, ``cd(a) = cd0(-a)``, ``cm(a) = -cm0(-a)``
    (``aerobo.polar.InvertedPolar``, which is what the solvers fly). So the
    shape and the curves have to move TOGETHER: a section drawn arching down
    beside a lift curve that still climbs through positive cl reads as a sign
    error, and that contradiction is worse than leaving both upright.

    Restated here rather than re-screened: the catalogue is stored upright
    and RANKED upright at the same |cl| — an upright section at +cl is this
    one at -cl, which is why the ranking is untouched by any of this. Only
    the presentation changes frame.

    ``inverted`` False returns ``rep`` itself (identity, not a copy).
    """
    if not rep or not inverted:
        return rep

    def _coords(c):
        if not c:
            return c
        a = np.asarray(c, dtype=float)
        return [[float(u), -float(v)] for u, v in a[:, :2]]

    def _polar(p):
        if not p:
            return p
        out = dict(p)
        for key in ("alpha_deg", "cl", "cm"):          # cd is even in alpha
            if p.get(key) is not None:
                out[key] = [-float(v) for v in p[key]]
        return out

    def _side(d):
        if not isinstance(d, dict):
            return d
        out = dict(d)
        if "coords" in out:
            out["coords"] = _coords(out["coords"])
        if "polar" in out:
            out["polar"] = _polar(out["polar"])
        for key in ("cl_design", "cl_max", "clmax", "alpha_L0_deg", "cm_ac"):
            if isinstance(out.get(key), (int, float)) \
                    and not isinstance(out[key], bool):
                out[key] = -float(out[key])
        return out

    out = dict(rep)
    for key in ("design", "baseline"):
        if key in out:
            out[key] = _side(out[key])
    if isinstance(out.get("cl_design"), (int, float)) \
            and not isinstance(out["cl_design"], bool):
        out["cl_design"] = -float(out["cl_design"])
    return out


def mirror_section_fig(fig, inverted: bool):
    """A section preview drawn the way its surface MOUNTS it.

    A surface that pushes down flies its camber the other way up
    (``tail.orient_section``), and the geometry views, the STL and the
    OpenVSP script already loft it mirrored. Leaving this one preview
    upright is the contradiction a reader trips over: the caption says the
    surface pushes down, the picture shows an aerofoil arching up, and the
    two together look like a bug in the physics rather than two conventions.

    The mirror is z -> -z about the chord line, so every curve on the figure
    — the shape, its anchor, the t/c annotation — moves together and nothing
    about the shape itself changes. ``inverted`` False returns the figure
    untouched, which is every lifting surface and every symmetric section.
    """
    if fig is None or not inverted:
        return fig
    for tr in fig.data:
        y = getattr(tr, "y", None)
        if y is not None:
            tr.y = [-float(v) for v in y]
    for ann in fig.layout.annotations:
        if ann.yshift:
            ann.yshift = -ann.yshift
    title = getattr(fig.layout.title, "text", "") or "Section"
    fig.update_layout(title_text=f"{title} — drawn AS MOUNTED (inverted)")
    return fig


def build(ctx, stage: str = "airfoil"):     # noqa: PLR0915  (built whole)
    """Build ONE surface's airfoil stage.

    ``stage`` is ``airfoil`` (the wing) or ``airfoil_aft`` (the second
    surface). Everything below is written against ``SURFACE`` and its own
    workspace ``A``, so the two builds cannot drift into two different forms
    for one question.
    """
    from nicegui import ui

    from aerobo import api
    from gui import nice_app as v1

    S = ctx.S
    SURFACE = session.STAGE_SURFACE[stage]
    # THREE SURFACES, TWO DISTINCTIONS. Most of what used to key off ``AFT``
    # means "not the wing" — a secondary surface inherits the wing's section
    # until one is chosen for it, has its own design point, and offers a way
    # back to flying the wing's. That is SECONDARY, and the vertical
    # stabiliser is one of those.
    #
    # ``AFT`` survives only where the branch is genuinely about the second
    # LIFTING surface: an elevator or a tandem rear wing carries load and has
    # a design lift coefficient, and the fin does neither.
    AFT = SURFACE == "aft"
    # ...and the two VERTICAL surfaces, which ask the same questions: no
    # design lift, a symmetric shape, and a section chosen for drag and
    # thickness. Every branch that was about "the fin" is about both.
    FIN = SURFACE in ("fin", "plate")
    SECONDARY = SURFACE != "main"
    #: where THIS surface's own section is stored under the main workspace
    SECTION_KEY = session.SECTION_KEYS.get(SURFACE)
    A = session.airfoil_state(S, SURFACE)
    seen = {"screen": None, "opt": None}
    cancel = {"screen": False, "opt": False}
    weight_labels: dict = {}       # criterion -> its value read-out
    #: containers a control can redraw WITHOUT redrawing the view it sits in
    #: (see _ar_control): rebuilding a view takes the focus out of the field
    #: the user is typing into
    boxes: dict = {}
    #: live named metrics of the section optimiser's incumbent. One
    #: design_report per IMPROVEMENT — and on this stage that report is
    #: nearly free, because the incumbent's XFOIL sweep is already cached.
    live = sampler_mod.IncumbentSampler()
    # ``keys`` is empty and ``picked`` False until the user ticks something:
    # WHICH metrics open is the catalogue's answer, not a literal here, so a
    # family that spells its objective ``f`` rather than ``LoD`` still opens
    # on the objective (metrics.live_metric_defaults).
    live_cfg = {"on": True, "keys": [], "picked": False, "cfg": None,
                "best": None, "best_f": None}
    #: what the LAST section export did, so the card can say it. Held here
    #: rather than in a widget because the card is rebuilt by the save it
    #: reports on.
    export_state: dict = {"note": None, "level": ""}

    def fmt(v, nd=4) -> str:
        return v1._fmt(v, nd)

    # =================================================== screening worker
    def _re_is_the_surface_own() -> bool:
        """Is the screen being asked for a point the cache does not hold?

        True only when the user asked for this surface's OWN Reynolds number
        (or typed a point) AND that is not where the library is cached — the
        one case that costs live XFOIL sweeps. Asking for the cached point by
        either route is still instant, and must still take the instant path.
        """
        cond = session.section_conditions(S, SURFACE)
        # ...and the toggle decides nothing in AIRFOIL-ONLY mode: the point
        # is stage 1's stated flow and this control is not drawn, so a
        # "library" left in the workspace by an earlier vehicle session
        # would send a point the cache does not hold down the instant path —
        # which is `screen_airfoils` over the WHOLE database at a Reynolds
        # number nothing is cached at: hours, not the shortlist's minutes.
        if not session.airfoil_only(S) \
                and A.get("re_source", session.RE_SOURCE_DEFAULT) == "library" \
                and not A["override_point"]:
            return False
        pt = session.library_point()
        if not pt:
            return False
        return abs(float(cond["re"]) - float(pt["re"])) \
            > 1e-9 * abs(float(cond["re"]))

    def _rescreen_banner():
        """A re-screen ARMED by another stage, priced here.

        Stage 3 can discover that this surface flies its section at a
        different Reynolds number from the one it was ranked at, and re-point
        this stage at the surface's own. That used to end in a log line naming
        this button; the request now travels with the decision
        (:func:`session.arm_rescreen`) and lands here, where the price can be
        stated, because this is the stage that knows whether the sweep is
        cached or real.
        """
        req = session.pending_rescreen(S, SURFACE)
        if not req or A["screen"]["running"]:
            return
        own = _re_is_the_surface_own()
        n = int(A.get("shortlist", session.SHORTLIST_DEFAULT))
        price = (f"{n} sections swept for real at this point — two XFOIL "
                 f"marches each (cruise + stall), six at a time: tens of "
                 f"seconds on the first visit, minutes if the marches "
                 f"struggle to converge; instant afterwards"
                 if own else
                 "the whole library is already cached at this point — instant")
        with widgets.group_box("A re-screen is waiting"):
            widgets.hint(f"{req['reason']}. Ranking again at "
                         f"Re {session.section_conditions(S, SURFACE)['re']:.3e} "
                         f"costs: {price}.", "warn")
            # THE DIAGNOSIS, not just the detection. "Two Reynolds numbers
            # disagree" leaves the user to price the disagreement in exactly
            # the quantity they came here for the tool to know. This states it
            # in their own variables — the section they already picked, and
            # what it is worth where the surface actually flies. Cache-only,
            # so it can never make this render block.
            pen = session.section_point_penalty(S, SURFACE)
            if pen:
                verb = "loses" if pen["loss_pct"] < 0 else "gains"
                ui.label(
                    f"{pen['name']} at cl {pen['cl']:.2f}: "
                    f"L/D {pen['ld_screened']:.1f} at "
                    f"Re {pen['re_screened']:.2e} (where it was ranked) "
                    f"vs {pen['ld_own']:.1f} at Re {pen['re_own']:.2e} "
                    f"(where it flies) — {verb} "
                    f"{abs(pen['loss_pct']):.0f} %"
                ).classes("mono text-xs").style(f"color:{theme.INK_MUTED}")
            with ui.row().classes("w-full items-center gap-2"):
                ui.button("Re-screen now", icon="sync",
                          on_click=_run_armed_rescreen) \
                    .props("unelevated dense no-caps color=primary")
                ui.button("Not now", icon="close",
                          on_click=_dismiss_rescreen) \
                    .props("flat dense no-caps")

    def _run_armed_rescreen():
        session.disarm_rescreen(S, SURFACE)
        start_screen()

    def _dismiss_rescreen():
        """Dismissing does NOT re-point the stage back.

        The operating point really did move; refusing the sweep leaves the
        table on screen honestly labelled stale by
        ``session.section_point_is_stale``, which is the pre-existing
        machinery. What is dropped is only the offer.
        """
        session.disarm_rescreen(S, SURFACE)
        ctx.log("re-screen dismissed — the ranking on screen is still the one "
                "produced at the old point, and is labelled as such", "info")
        ctx.render_when_shown(stage)

    def start_screen():
        if A["screen"]["running"] or A["opt"]["running"]:
            ui.notify("already running", type="warning")
            return
        session.disarm_rescreen(S, SURFACE)
        cond = session.section_conditions(S, SURFACE)
        A["screen"].update(running=True, error=None, progress=None,
                           selected=0, swept=[],
                           # the two-pass screen opens on its FIRST pass; the
                           # cached-point screen has only one
                           phase=("library" if _re_is_the_surface_own()
                                  else "sweep"))
        cancel["screen"] = False
        own = _re_is_the_surface_own()
        ctx.log(f"screening the UIUC library for the {_surface_name()} at "
                f"Re {cond['re']:.4g}, Cl {cond['cl_design']:.4f} …"
                + (f" the cache holds no polars at this Reynolds number, so "
                   f"the {int(A.get('shortlist', session.SHORTLIST_DEFAULT))} "
                   f"leaders of the cached ranking are being swept there for "
                   f"real — two XFOIL marches each, six at a time; instant "
                   f"after."
                   if own else ""), "info")
        ctx.status("screening the library…", "busy", 0.0)
        ctx.refresh()
        threading.Thread(target=_screen_worker, daemon=True).start()

    def _screen_worker():
        sc = A["screen"]
        try:
            cond = session.section_conditions(S, SURFACE)
            floors = {k: v for k, v in A["floors"].items()
                      if k in api.SCREEN_FLOOR_KEYS and v is not None}
            # A WING THAT HAS A TAIL MUST NOT BE HANDED A TAILLESS AEROFOIL.
            # |Cm| is a lower-better criterion, so ranking on it rewards
            # REFLEX — which is what a section does INSTEAD of having a tail.
            # Measured on the published screen: five of the top six were
            # reflexed or near-zero, four of them Horten (hg*) and Hepperle
            # (mh*) flying-wing sections. Flown under a stabiliser they
            # invert its job — the wing holds its own pitching moment, so
            # the trim balance asks the tail for UP-load and the
            # "stabiliser" lifts. Gated by DEFAULT, and only here: the
            # trimming surface itself is free to be anything, and a user who
            # wants a reflexed wing turns the gate off.
            if session.nose_down_required(S, SURFACE):
                floors[api.SCREEN_NOSE_DOWN_KEY] = 0.0

            def progress(i, n, rec):
                # THE SWEEP IS THE THING THE USER IS WAITING FOR, so it
                # reports per section rather than per pass. The first callback
                # is also what says pass 1 is over: `screen_at_point` hands
                # this callback to the SHORTLIST pass only (the library pass
                # is a cache read), so its arrival IS the phase change.
                sc["progress"] = (int(i), int(n))
                sc["phase"] = "sweep"
                rec = rec or {}
                row = {"i": int(i), "name": str(rec.get("name") or "?"),
                       "status": str(rec.get("status") or ""),
                       "eligible": bool(rec.get("eligible")),
                       "ldcr": rec.get("ldcr"), "tc": rec.get("tc")}
                swept = sc.setdefault("swept", [])
                swept.append(row)
                if len(swept) > SWEEP_LOG_MAX:      # a log, not a leak
                    del swept[:-SWEEP_LOG_MAX]

            common = dict(
                weights=dict(A["weights"]),
                re=float(cond["re"]), mach=float(cond["mach"]),
                cl_design=float(cond["cl_design"]),
                tc_min=gate_value("tc_min"), cm_max=gate_value("cm_max"),
                floors=floors, top_n=int(A["top_n"]),
                progress_cb=progress, cancel=lambda: cancel["screen"])
            # WHICH MEMBERS THIS SURFACE MAY RANK. A vertical stabiliser's
            # screen is restricted to symmetric sections — see
            # ``screen_names_for``, which owns the rule and raises rather
            # than silently screening everything when it cannot apply it.
            try:
                only = screen_names_for(SURFACE)
            except RuntimeError as exc:
                sc["error"] = str(exc)
                return
            if only is not None:
                common["names"] = only
            if _re_is_the_surface_own():
                # THE SURFACE'S OWN POINT: no cached polar covers it, so this
                # is real XFOIL — the shortlist is what keeps that minutes
                # rather than hours (session.SHORTLIST_DEFAULT).
                rep = api.screen_at_point(
                    shortlist=int(A.get("shortlist",
                                        session.SHORTLIST_DEFAULT)),
                    **common)
            else:
                rep = api.screen_airfoils(**common)
            sc["report"] = rep
            # WHICH surface this ranking is an answer for: the two are
            # screened at different points, so a table left on screen after
            # the target moved is not this surface's ranking
            sc["surface"] = SURFACE
            # CST refits of every ranked row, so ANY row of the table can be
            # chosen (the report itself only refits its winner)
            sc["candidates"] = api.screen_seed_candidates(
                rep, n=len(rep.get("ranked") or []))
        except Exception as exc:      # noqa: BLE001 — surface, never crash
            sc["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            sc["running"] = False
            sc["stamp"] = time.time()

    # =================================================== optimiser worker
    def _shape_kwargs() -> dict:
        """This surface's launch arguments, off the form (:func:`shape_kwargs`)."""
        return shape_kwargs(S, SURFACE, A["opt"], A["weights"],
                            _screen_reference())

    def _prior_curve(oc: dict) -> list:
        """The best-so-far curve of the run being continued.

        Off the stored REPORT where there is one (``result.history``), and
        off the live records otherwise — a run stopped by hand has records
        and a report, a run still being read back may have only one of them.
        Empty list where neither says anything, which draws nothing.
        """
        hist = ((oc.get("report") or {}).get("result") or {}).get("history")
        if hist:
            # NaN where the run had no feasible design yet. A stored history
            # holds None for -inf, and floating it raw put
            # "TypeError: float() ... not 'NoneType'" on screen in place of
            # the plot the moment a continuation drew its second trace — the
            # same defect the wing's own prior curve had. The records branch
            # below already guarded it; this one did not, which is why the
            # crash needed a run whose report had been stored.
            return [float("nan") if v is None else float(v) for v in hist]
        return [float(r["best"]) for r in (oc.get("records") or [])
                if r.get("best") is not None]

    def start_optimise(cont: dict | None = None):
        """Start the shape search — or CONTINUE the one on screen.

        ``cont`` is a continuation snapshot (:func:`session.continue_section`):
        the arguments and the seed of the run being lengthened, at a bigger
        budget. It replaces BOTH the form and stage 1's budget policy, because
        a continuation has to be the same search — anything else is a new run
        wearing the old one's label.
        """
        if A["screen"]["running"] or A["opt"]["running"]:
            ui.notify("already running", type="warning")
            return
        oc = A["opt"]
        oc.update(running=True, error=None, report=None, records=[],
                  progress=None, stopped=False,
                  # HOW MANY OF THE COMING EVALUATIONS ARE A RE-FLIGHT. A
                  # continuation re-flies its prefix on the same seed — that
                  # is what makes the longer run contain the shorter one —
                  # and on this stage those evaluations come straight back
                  # out of the on-disk XFOIL cache. The number is kept so the
                  # RUNNING chip can say "replaying" rather than showing a
                  # counter that appears to have started over.
                  replay=int((cont or {}).get("was") or 0),
                  # ...AND HOW MANY IT INHERITED. A resumed continuation
                  # (``api.optimize_airfoil(resume=…)``) is handed the
                  # previous search's evaluations as its training set: none
                  # of them is flown, so `replay` above is 0 and the counter
                  # simply opens at this number instead of at 1.
                  resumed=int((cont or {}).get("resumed") or 0),
                  # ...AND THE CURVE THAT RUN DREW. Clearing `records` above
                  # empties the convergence plot, so a continuation used to
                  # blank the chart and redraw it from evaluation 1 — the
                  # same thing the counter does, and the thing a user reads
                  # as "it started over". Kept as its own series: over the
                  # re-flown prefix the two are the same numbers, and past
                  # it they are honestly two different runs.
                  prior=(list(_prior_curve(oc)) if cont else []))
        cancel["opt"] = False
        shape = (cont or {}).get("shape") or {}
        objective = (str(shape.get("objective") or "cd") if cont
                     else str(oc.get("objective", "cd")))
        anchored = (bool(shape.get("anchor")) if cont
                    else bool(session.section_weights(S, SURFACE)))
        # WHICH budget is on screen while it runs. The RUNNING tag reads this
        # rather than the policy, because the policy stops describing the run
        # the moment it is a continuation of an older, smaller one.
        oc["run_budget"] = int(
            cont["search"]["budget"] if cont
            else session.effective_airfoil_search(S, SURFACE)["budget"])
        ctx.log(("continuing the shape optimisation for the " if cont
                 else "shape optimisation for the ")
                + f"{_surface_name()} — "
                + ("maximising the COMPOSITE SCORE of your six criteria, "
                   if is_composite(objective) else "")
                + ("seeded from "
                   + (session.section_of(S, SURFACE) or {}).get("name", "?")
                   if anchored else "unseeded (the family's NACA anchor)")
                + f", {int(oc['run_budget'])} evaluations of a live XFOIL "
                  f"sweep"
                + (" (two: the composite needs the wide stall sweep too)"
                   if is_composite(objective) else ""),
                "info")
        ctx.status("optimising the section…", "busy", 0.0)
        ctx.refresh()
        live.reset()
        live_cfg.update(best=None, best_f=None, cfg=None)
        threading.Thread(target=_opt_worker, args=(cont,),
                         daemon=True).start()
        if live_cfg["on"]:
            live.start(cfg_fn=lambda: live_cfg["cfg"],
                       incumbent_fn=_opt_incumbent,
                       running_fn=lambda: bool(A["opt"]["running"]))

    def _continue_offer(extra=None) -> dict | None:
        """What "Continue" would do to the run on screen, priced — or None.

        ``None`` means there is nothing to continue (no finished run, or one
        already running) and the button is simply dead. A dict carrying an
        ``error`` means there IS a run but it cannot be re-flown as itself,
        and the button says why instead of quietly starting something else.
        Pure — it launches nothing, so the label can carry the real number
        rather than the word "more".
        """
        oc = A["opt"]
        if oc["running"] or not oc.get("report") or not oc.get("launch"):
            return None
        return session.continue_section(oc.get("report") or {},
                                        oc.get("launch") or {},
                                        shape_now=_shape_kwargs(), extra=extra)

    def continue_optimise():
        """Give the search that just ran MORE evaluations — the same search.

        Not a warm start and not a fresh run from the winner: the launch
        snapshot is re-flown on the SAME seed at a bigger budget, so for every
        optimiser in ``api.CONTINUABLE_OPTIMISERS`` the evaluations already
        paid for come back bit-for-bit and the new ones carry on from them.
        Handing a search its own previous best instead is the arm this repo
        has measured and lost with (``RESULTS_HANDOFF.md``).

        On THIS stage the re-flight is nearly free: every polar it needs is
        already in the on-disk XFOIL cache, so what a continuation costs is
        the new evaluations — which is what makes "it had not converged" a
        one-press question here rather than a whole search again.
        """
        out = _continue_offer()
        if out is None:
            ui.notify("there is no finished section search to continue",
                      type="info")
            return
        if out["error"]:
            ui.notify(out["error"], type="negative")
            ctx.log(f"cannot continue: {out['error']}", "warn")
            return
        note = out["note"]
        # the cache sentence belongs to a CONTAINED continuation only. A run
        # the api has judged uncontained flies a different set of designs, so
        # "what this costs is the new ones" would be false — the cache still
        # helps, but only where a shape happens to repeat.
        ctx.log(f"continuing the section search: {note['was']} -> "
                f"{note['budget']} evaluations. {note['why']}."
                + (f" The evaluations already paid for come back out of the "
                   f"XFOIL cache, so what this costs is the {note['added']} "
                   f"new ones." if note["exact"] else
                   " Every polar it has already seen is still cached, but a "
                   "fresh search is not bounded by the evaluations you have "
                   "already paid for."),
                "ok" if note["exact"] else "warn")
        if out["drift"]:
            ctx.log("this re-flies the run's OWN search, not the form on "
                    "screen — " + ", ".join(out["drift"])
                    + " changed since. Optimise instead to search what the "
                      "form now says.", "warn")
        start_optimise(out["cont"])

    def _opt_worker(cont: dict | None = None):
        oc = A["opt"]
        # WHICH budget this run spends, and whether it may stop early: the
        # policy is stage 1's, so it is read here rather than off this
        # stage's own fields (session.effective_airfoil_search) — UNLESS this
        # is a continuation, whose budget, optimiser and seed come off the run
        # being lengthened. Stage 1's recommendation may have moved since that
        # run, and a longer run of a different search is not a continuation.
        eff = session.effective_airfoil_search(S, SURFACE)
        if cont is not None:
            eff = {**eff, **cont["search"], "n_restarts": 1}
        # a FACTORY, not a rule: each restart gets its own history, or the
        # second one would stop on the first one's plateau
        _factory = session.stop_rule_factory(S, eff)
        try:
            # WHAT this run flies, and what a later continuation re-flies.
            # Stored BEFORE the search starts, so a run stopped by hand is as
            # continuable as one that spent its whole budget.
            shape = dict(cont["shape"]) if cont else _shape_kwargs()
            seed = int(cont["seed"] if cont else oc["seed"])
            oc["launch"] = {"shape": dict(shape), "seed": seed,
                            "search": {"optimiser": str(eff["optimiser"]),
                                       "budget": int(eff["budget"]),
                                       "refusal": eff.get("refusal"),
                                       "n_init": eff.get("n_init")},
                            "n_restarts": int(eff.get("n_restarts", 1) or 1)}

            # WHERE this restart's trace starts, and the best any restart
            # has reached: k independent searches are ONE answer, so the
            # convergence trace stays monotone across them and the run that
            # wins is the one whose report is kept
            spent = {"n": 0, "best": None}

            def progress(i, best, **kw):
                # NO raise here — see stop_or_converged. It only records.
                b = float(best) if math.isfinite(best) else None
                if b is not None and (spent["best"] is None
                                      or b > spent["best"]):
                    spent["best"] = b
                oc["records"].append({"n": spent["n"] + int(i),
                                      "best": spent["best"]})
                oc["progress"] = spent["n"] + int(i)
                # the incumbent, for the live metric sampler: the rich
                # payload carries this candidate's own f and x
                f, x = kw.get("f"), kw.get("x")
                if x is not None and f is not None and math.isfinite(f) \
                        and (live_cfg["best_f"] is None
                             or float(f) > live_cfg["best_f"]):
                    live_cfg["best_f"] = float(f)
                    live_cfg["best"] = (spent["n"] + int(i), list(x))

            #: a FRONT run differs from every scalar one in exactly one way
            #: down here: it cannot be interrupted mid-search (see the
            #: stop_rule note at the api call below)
            is_front = str(shape.get("objective") or "cd") == "pareto"
            # the SAME configuration the run is about to use, kept so the
            # live sampler can re-evaluate the incumbent through design_report
            live_cfg["cfg"] = api.airfoil_run_config(
                **shape,
                optimiser=str(eff["optimiser"]), budget=int(eff["budget"]),
                seed=seed, refusal=eff.get("refusal"),
                # the shell's own default for a search that finds nothing
                # feasible (config.FEASIBILITY_DEFAULT): the reported section
                # run found 0 of 164 evaluations feasible and the same config
                # with a screened initial design found 142 of 164
                feasibility=config.FEASIBILITY_DEFAULT,
                n_init=eff.get("n_init"))
            # ...then RUN IT, once per recommended restart. The re-measured
            # study gives k = 1 for this class — restarts "won" only on the
            # contaminated polars — so this loop runs exactly once and the run
            # is the single search, byte for byte.
            #
            # NB `budget` here is PER RESTART: recommend() has already divided
            # the total, so k runs of it spend the total the study fitted. The
            # progress readout below counts across restarts (`spent["n"]`) and
            # is compared against the same per-restart number, which reads
            # correctly only at k = 1 — see `_render_recommended_budget`.
            best_rep, best_y = None, None
            for k in range(max(1, int(eff.get("n_restarts", 1)))):
                if cancel["opt"]:
                    break
                conv = _factory() if _factory else None
                rep = api.optimize_airfoil(
                    **shape,
                    optimiser=str(eff["optimiser"]), budget=int(eff["budget"]),
                    refusal=eff.get("refusal"), n_init=eff.get("n_init"),
                    feasibility=config.FEASIBILITY_DEFAULT,
                    # the evaluations this run INHERITS, on a continuation
                    # that resumes rather than re-flies (None on every other
                    # run, which is the legacy call exactly)
                    resume=eff.get("resume"),
                    seed=seed + k, progress_cb=progress,
                    results_dir=str(session.RESULTS_DIR),
                    # A FRONT CANNOT BE STOPPED EARLY, and the api refuses the
                    # argument rather than accepting one nothing reads. So the
                    # rule is not sent on a front run — sending it would raise,
                    # and pretending otherwise would arm a Stop button that
                    # does nothing. The button still cancels BETWEEN restarts
                    # (`cancel["opt"]` is checked around this loop); what it
                    # cannot do is interrupt one front mid-search.
                    **({} if is_front else dict(
                        stop_rule=stop_or_converged(
                            conv, lambda: cancel["opt"]),
                        stop_reason=("the section stopped improving"
                                     if conv else None))))
                y = ((rep.get("result") or {}).get("best_score")
                     if isinstance(rep, dict) else None)
                if best_rep is None or (y is not None
                                        and (best_y is None or y > best_y)):
                    best_rep, best_y = rep, y
                spent["n"] = oc["progress"]
                if cancel["opt"]:
                    break
            if cancel["opt"]:
                # WHY it is partial, on the record the user keeps: the stop
                # rule that fired was the button, not the plateau, and
                # api.run cannot tell the two apart from inside.
                oc["stopped"] = True
                if isinstance(best_rep, dict):
                    (best_rep.get("result") or {})["stop_reason"] = STOP_REASON
            oc["report"] = best_rep
        except Exception as exc:      # noqa: BLE001 — surface, never crash
            oc["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            oc["running"] = False
            oc["stamp"] = time.time()
        # THE OTHER FIVE CRITERIA, after the run rather than during it: the
        # result table can be read the moment the search ends, and the score
        # (which costs a wide stall sweep per shape — seconds, and nothing
        # else caches it) arrives behind its own stamp.
        #
        # A STOPPED RUN IS SCORED TOO — it has a section, so it has the
        # comparison that says what the search bought. Only a run that never
        # reached a feasible design has nothing to score.
        if (oc.get("report") or {}).get("section") and not oc.get("error"):
            _score_worker(oc["report"])

    # ============================================ score under the weights
    def _screen_reference():
        """The frozen band THIS surface's ranking was scored on, if it had
        one (:func:`api.screen_at_point` measures one over the library pass).
        None falls back to the shipped library band — either way the seed and
        the optimised section are scored on the same fixed map, which is the
        property that makes the two numbers comparable at all."""
        rep = A["screen"]["report"] or {}
        ref = rep.get("reference")
        return ref if isinstance(ref, dict) and ref.get("bounds") else None

    def _score_worker(rep: dict):
        oc = A["opt"]
        oc.update(scoring=True, score=None, score_error=None)
        try:
            oc["score"] = api.score_optimised_section(
                rep, dict(A["weights"]), reference=_screen_reference())
        except Exception as exc:      # noqa: BLE001 — a view, never fatal
            oc["score_error"] = f"{type(exc).__name__}: {exc}"
        finally:
            oc["scoring"] = False
            oc["score_stamp"] = time.time()

    def _opt_incumbent():
        """(evaluations, best design vector) — None until the config and a
        feasible incumbent both exist."""
        if live_cfg["cfg"] is None or live_cfg["best"] is None:
            return int(A["opt"]["progress"] or 0), None
        n, x = live_cfg["best"]
        return n, x

    def stop():
        if A["screen"]["running"]:
            cancel["screen"] = True
            ctx.log("screening will stop after the current section", "warn")
        if A["opt"]["running"]:
            cancel["opt"] = True
            ctx.log("optimisation will stop after the current evaluation — "
                    "the best section found so far is kept, scored and "
                    "adopted", "warn")

    def run_stage():
        """Toolbar Run on this stage: screen, or optimise if that is the
        tab the user is looking at."""
        if S["ui"]["tab"][stage] == "optimise":
            start_optimise()
        else:
            start_screen()

    # ========================================================== polling
    def poll() -> bool:
        dirty = False
        sc, oc = A["screen"], A["opt"]
        if live.version != seen.get("sampler"):
            seen["sampler"] = live.version
            if S["ui"]["selected"] == stage \
                    and S["ui"]["tab"][stage] == "optimise":
                # IN PLACE: a new sample is a point on a trace, not a page
                _tick_optimise()
        if sc["running"]:
            if sc["progress"]:
                i, n = sc["progress"]
                ctx.status(f"screening {i}/{n} sections", "busy",
                           (i / n) if n else None)
            # ...and the same progression IN the stage that asked for it, so
            # the user watching the sweep does not have to read a one-line
            # status strip to know it is alive. Only when it is on screen: the
            # optimise tab redraws its own trace, and a hidden view rebuilt
            # twice a second is pure cost (same rule as the branch below).
            if S["ui"]["selected"] == stage \
                    and S["ui"]["tab"][stage] == "screen":
                _render_sweep()
                dirty = True
        if oc["running"]:
            done = int(oc["progress"] or 0)
            # THE RUNNING BUDGET, not the policy's. A continuation is a
            # search of `spent + extra`, and the policy stops describing it
            # the moment it is one — the tag beside the plot already reads
            # `run_budget` (:func:`_run_tag`) and the shell-wide status bar,
            # which is the one visible from every stage, did not: it showed
            # "54/29" on a run of 57.
            n = int(oc.get("run_budget")
                    or session.effective_airfoil_search(S, SURFACE)["budget"])
            ctx.status(f"XFOIL evaluation {done}/{n}", "busy",
                       min(1.0, done / max(1, n)))
            # redraw the live trace only when it is ON SCREEN: rebuilding a
            # hidden view twice a second for the minutes an XFOIL search
            # takes is pure cost, and it would also blow away the focus of
            # any field the user is typing into on another tab
            if S["ui"]["selected"] == stage \
                    and S["ui"]["tab"][stage] == "optimise":
                _tick_optimise()
        if sc.get("stamp") and sc["stamp"] != seen["screen"]:
            seen["screen"] = sc["stamp"]
            if sc["error"]:
                ctx.log(f"screening failed: {sc['error']}", "error")
                ctx.status("screening failed", "error", None)
            else:
                rep = sc["report"] or {}
                ctx.log(f"screened {rep.get('n_screened', 0)} sections, "
                        f"{rep.get('n_eligible', 0)} clear the gates "
                        f"({rep.get('wall_time_s', 0.0):.1f} s)", "ok")
                win = ((rep.get("winner") or {}).get("metrics") or {})
                if win.get("name"):
                    ctx.log(f"best by your weights: {win['name']} "
                            f"(score {fmt(win.get('composite'), 4)})", "ok")
                ctx.status("screening complete", "ok", None)
                _adopt_row(0, quiet=True)
            _render_screen()
            _render_ranking()
            _render_section()
            dirty = True
        if oc.get("stamp") and oc["stamp"] != seen["opt"]:
            seen["opt"] = oc["stamp"]
            if oc["error"]:
                ctx.log(f"shape optimisation: {oc['error']}",
                        "warn" if "stopped" in oc["error"] else "error")
                ctx.status("optimisation stopped", "warn", None)
            elif oc.get("stopped"):
                # STOPPED IS NOT FAILED. The run kept every evaluation it
                # paid for, and its incumbent is a real evaluated section —
                # so it is reported, scored and adopted exactly as a finished
                # run is, and only the word for it differs.
                rep = oc["report"] or {}
                res = rep.get("result") or {}
                if rep.get("section"):
                    ctx.log(f"shape optimisation stopped by you — keeping the "
                            f"best section of the "
                            f"{res.get('n_evals', '?')} evaluations already "
                            f"paid for (objective "
                            f"{fmt(res.get('best_score'), 5)})", "warn")
                    ctx.status("stopped — best section kept", "warn", None)
                    _adopt_optimised(quiet=True)
                else:
                    ctx.log("shape optimisation stopped before any section "
                            "flew, so there is nothing to keep — the library "
                            "pick still stands", "warn")
                    ctx.status("optimisation stopped", "warn", None)
            else:
                rep = oc["report"] or {}
                res = rep.get("result") or {}
                ctx.log(f"shape optimisation finished — best objective "
                        f"{fmt(res.get('best_score'), 5)} after "
                        f"{res.get('n_evals', '?')} evaluations "
                        f"({rep.get('wall_time_s', 0.0):.0f} s)", "ok")
                ctx.status("optimisation complete", "ok", None)
                _adopt_optimised(quiet=True)
            _render_optimise()
            _render_section()
            dirty = True
        if oc.get("score_stamp") and oc["score_stamp"] != seen.get("score"):
            seen["score"] = oc["score_stamp"]
            if oc.get("score_error"):
                ctx.log(f"the section could not be scored on your weights: "
                        f"{oc['score_error']}", "warn")
            else:
                sco = oc.get("score") or {}
                seed_j = (sco.get("seed") or {}).get("composite")
                opt_j = (sco.get("optimised") or {}).get("composite")
                d = (sco.get("delta") or {}).get("composite")
                if opt_j is not None:
                    ctx.log(
                        "score under your weights: "
                        + (f"seed {seed_j:.2f} → " if seed_j is not None
                           else "")
                        + f"optimised {opt_j:.2f}"
                        + (f" ({d:+.2f})" if d is not None else "")
                        + " — the six criteria the ranking uses, not the "
                          "number the search maximised",
                        "ok" if (d is None or d >= 0.0) else "warn")
            _render_optimise()
            dirty = True
        return dirty

    ctx.add_poll(poll)

    # ================================================== section adoption
    def _candidate(idx: int) -> dict | None:
        rep = A["screen"]["report"] or {}
        ranked = rep.get("ranked") or []
        if not ranked or not 0 <= idx < len(ranked):
            return None
        row = dict(ranked[idx])
        row["rank"] = idx + 1
        return merge_candidate(row, A["screen"]["candidates"])

    def _adopt_row(idx: int, quiet: bool = False):
        row = _candidate(idx)
        if row is None:
            return
        A["screen"]["selected"] = idx
        rep = A["screen"]["report"] or {}
        winner = rep.get("winner") or {}
        sec = {
            "name": row.get("name"),
            "source": "library",
            "rank": row.get("rank"),
            "tc": row.get("tc"),
            "ldcr": row.get("ldcr"),
            "clmax": row.get("clmax"),
            "cm": row.get("cm_at"),
            "w_upper": row.get("w_upper"),
            "w_lower": row.get("w_lower"),
            # THE SECTION'S OWN OUTLINE. The weights above are its sharp-TE
            # projection — what stage 3 flies — and drawing those instead
            # closes a blunt trailing edge that the aerofoil really has
            # (stages.airfoil.section_outline).
            "coords": row.get("coords"),
            "te_gap": row.get("te_gap"),
            "conditions": dict(rep.get("conditions") or {}),
            # the report only carries the WINNER's shape + polar; keep it
            # when the pick IS the winner so the section view can draw both
            "report": winner if row.get("rank") == 1 else None,
        }
        session.set_section(S, sec, "library", surface=SURFACE)
        if not quiet:
            ctx.log(f"{_surface_name()} section: {sec['name']} "
                    f"(rank {sec['rank']}, "
                    f"t/c {fmt(sec['tc'], 4)})", "ok")
        ctx.render_when_shown("wing")
        ctx.refresh()

    def _adopt_optimised(quiet: bool = False):
        rep = A["opt"]["report"] or {}
        srep = rep.get("section")
        res = rep.get("result") or {}
        if not srep:
            return
        design = srep.get("design") or {}
        weights = optimised_weights(rep)
        if weights is None:
            # a DESIGNED section travels as its CST weights and nothing else
            # — adopting one without them would hand stage 3 a section it
            # cannot fly, so refuse here rather than fail inside the run
            ctx.log("the optimised section could not be adopted: the run "
                    "carries no CST weights (re-run stage 2)", "error")
            return
        sec = {
            "name": "CST section (optimised)",
            "source": "optimised",
            "tc": design.get("tc"),
            "ldcr": None,
            "w_upper": weights[0],
            "w_lower": weights[1],
            "conditions": dict(rep.get("conditions") or {}),
            "report": {"design": design,
                       "baseline": srep.get("baseline"),
                       "cl_design": srep.get("cl_design"),
                       "tc_min": srep.get("tc_min"),
                       "cm_max": srep.get("cm_max")},
            "best_score": res.get("best_score"),
        }
        session.set_section(S, sec, "optimised", surface=SURFACE)
        if not quiet:
            ctx.log(f"optimised section adopted for the {_surface_name()}",
                    "ok")
        ctx.render_when_shown("wing")
        ctx.refresh()

    def _next_stage() -> tuple[str, str, str]:
        """Where "done here" goes, and what to call it.

        THE PIPELINE ORDER, not a fixed destination: where the configuration
        carries a second surface, the wing's section is not the last section
        question — sending the user to stage 3 there skips the stage 2.5 they
        have not answered yet, and the second surface then quietly flies the
        wing's aerofoil because nothing asked.

        WHICH surface is next is :func:`session.next_section_stage`, not a
        literal here. This branch named ``airfoil_aft`` and sent every other
        surface to stage 3, so on a vehicle with a vertical stabiliser the
        HORIZONTAL tail's "done here" walked past stage 2.7 to the wing —
        the fin's section stage skipped exactly the way stage 2.5 used to be.
        The last section stage, whichever surface that is, goes on to the
        wing.
        """
        if session.airfoil_only(S):
            # no wing stage exists in this mode, and the section IS the
            # answer: the only place left to go is back to the flow it was
            # designed in
            return "mission", "point", "Back to the flow conditions"
        nxt = session.next_section_stage(S, stage)
        if nxt:
            name = session.surface_name(S, session.STAGE_SURFACE[nxt])
            return nxt, "screen", f"Choose the {name}'s section"
        return "wing", "type", "Go to the wing stage"

    def _go_next():
        stage_key, view, _label = _next_stage()
        ctx.select(stage_key, view)

    # ================================================ view: screening setup
    def _render_screen():     # noqa: PLR0915
        box = ctx.views[(stage, "screen")]
        box.clear()
        pt = session.library_point()
        with box:
            # WHICH surface first, but only where there are two: on a
            # single-surface craft "designing: wing" is noise
            if session.aft_surface(S):
                with widgets.group_box("Surface"):
                    _surface_row()
            with widgets.group_box(f"Design point — {_surface_name()}"):
                boxes["point"] = ui.row().classes(
                    "w-full items-start gap-4 no-wrap")
                _ar_control()
                # THE POINT IS STATED ON STAGE 1 in airfoil-only mode: a
                # fluid, a speed and a chord (session.flow_point). So the two
                # controls that pick or retype it here are not drawn — two
                # controls for one point is exactly the disagreement this
                # stage keeps out. What stays is the SHORTLIST, because that
                # is not the point, it is how much XFOIL is spent reaching it.
                only = session.airfoil_only(S)
                if not only:
                    ui.toggle({"mission": f"this {_surface_name()}'s own Re "
                                          f"(sweeps the shortlist)",
                               "library": "cached library point (instant)"},
                              value=A.get("re_source",
                                          session.RE_SOURCE_DEFAULT),
                              on_change=lambda e: _set_re_source(e.value)) \
                        .props("dense no-caps unelevated "
                               "toggle-color=primary")
                    widgets.hint(session.LIBRARY_RE_NOTE)
                if _re_is_the_surface_own():
                    with ui.row().classes("items-center gap-3 no-wrap"):
                        widgets.number_field(
                            "shortlist", A.get("shortlist",
                                               session.SHORTLIST_DEFAULT),
                            lambda e: _set_shortlist(e.value), step=4,
                            width="w-28",
                            tip="how many of the cached ranking's leaders are "
                                "swept at this Reynolds number")
                    widgets.hint(
                        "TWO live viscous XFOIL marches per section on the "
                        "first visit — the cruise sweep (-4..10 deg) and the "
                        "wider stall sweep Cl max needs (0..20 deg) — six "
                        "sections at a time. Order a second or two each, so "
                        "tens of seconds for the default shortlist and "
                        "minutes where the marches fight to converge. Instant "
                        "thereafter: the sweeps are cached like every other "
                        "polar, keyed by the POINT, so moving the speed or "
                        "the chord stage 1 states pays it again. Sections "
                        "outside "
                        "the shortlist are not swept here and cannot appear "
                        "in the ranking, so the table will say what it was "
                        "drawn from.", "warn")
                if pt is None:
                    widgets.hint("The screening cache has not been built on "
                                 "this machine, so the first screen runs "
                                 "XFOIL over the database.", "warn")
                if not only:
                    with ui.row().classes("items-center gap-3 no-wrap"):
                        ui.switch("type the point myself",
                                  value=bool(A["override_point"]),
                                  on_change=lambda e: _set_override(e.value)) \
                            .props("dense")
                        if A["override_point"]:
                            widgets.number_field(
                                "Re", widgets.shown(A["cond"]["re"], 0),
                                lambda e: _set_cond("re", e.value), step=1e5,
                                width="w-32")
                            widgets.number_field(
                                "design Cl",
                                widgets.shown(A["cond"]["cl_design"]),
                                lambda e: _set_cond("cl_design", e.value),
                                step=0.05)

            with ui.row().classes("w-full items-start gap-3 no-wrap"):
                with ui.column().classes("gap-3").style("flex:3 1 0;"
                                                        "min-width:0"):
                    with widgets.group_box("Criterion weights"):
                        for key, (label, why) in WEIGHT_META.items():
                            # WHY THIS CRITERION CANNOT RANK THIS SURFACE, on
                            # the row itself. The row is never removed: a
                            # user who cannot see the control cannot see that
                            # it is zero, and the three surfaces would stop
                            # sharing one form (DEAD_CRITERIA).
                            dead = DEAD_CRITERIA.get(SURFACE, {}).get(key)
                            with ui.row().classes("w-full items-center gap-2 "
                                                  "no-wrap"):
                                lab = ui.label(label).classes("field-label") \
                                    .style("min-width:150px"
                                           + (";" + UNWEIGHTED_STYLE
                                              if dead else ""))
                                # the ? rather than a tooltip where the
                                # sentence is an argument (widgets.explain):
                                # "cd at design Cl" carries 38 words about
                                # what a zero-lift surface can be ranked on,
                                # and a dead criterion carries the reason it
                                # ranks nothing — both invisible on hover
                                widgets.explain(
                                    lab, why if not dead
                                    else f"{why}\n\n{dead}", title=label)
                                # the value is read off a fixed column, not
                                # off a floating Quasar bubble: a stack of
                                # them sat on top of the labels
                                ui.slider(min=0.0, max=1.0, step=0.05,
                                          value=_weight_of(key),
                                          on_change=lambda e, k=key:
                                          _set_weight(k, e.value)) \
                                    .props("dense").classes("grow")
                                out = ui.label(f"{_weight_of(key):.2f}") \
                                    .classes("readout").style(
                                        "min-width:34px;text-align:right")
                                weight_labels[key] = out
                        widgets.hint("Weights are normalised before scoring, "
                                     "so only their ratios matter.")
                        _dead_weight_note()
                        _recommended_note()
                        _trim_weight_note()
                with ui.column().classes("gap-3").style("flex:2 1 0;"
                                                        "min-width:0"):
                    with widgets.group_box("Hard gates"):
                        _gate("min t/c", "tc_min", 0.01,
                              "sections thinner than this are refused "
                              "outright — structural depth")
                        _gate("max |Cm|", "cm_max", 0.01,
                              "sections with a larger pitching moment are "
                              "refused — trim drag and structure")
                        _nose_down_gate()
                        widgets.hint(
                            "A gate REFUSES a section; the weights rank the "
                            "ones that pass. Switch "
                            + ("them all" if _nose_down_applies() else "one")
                            + " off and there is no minimum (or no maximum) "
                            "at all — the screen then ranks the whole "
                            "eligible database on your weights alone.")
                    with widgets.group_box("Floors (optional)"):
                        for key, (label, step) in FLOOR_META.items():
                            widgets.number_field(
                                label, A["floors"].get(key),
                                lambda e, k=key: _set_floor(k, e.value),
                                step=step)
                        widgets.hint("Leave empty for no floor. A floor is a "
                                     "minimum on a higher-is-better metric.")

            _rescreen_banner()

            with ui.row().classes("w-full items-center gap-2"):
                ui.button("Screen the library", icon="search",
                          on_click=start_screen) \
                    .props("unelevated dense no-caps color=primary")
                # THE SECOND SURFACE KEEPS ITS SKIP AND THE WING DOES NOT, and
                # they were never the same button. "Let the tail fly the
                # wing's section" states a real alternative the user can only
                # express here. The wing's old twin ("use the family's own
                # section") declared what is already true of a stage nobody
                # answered — it bought nothing, gated stage 3 behind a click,
                # and made the pipeline read as three routes where there are
                # two. It is gone; not screening IS that answer, and the wing
                # stage says which polars the family then flies.
                if SECONDARY:
                    ui.button(f"Let the {_surface_name()} fly the wing's "
                              f"section", icon="link_off", on_click=_clear_aft)\
                        .props("outline dense no-caps")
                ui.space()
                if A["screen"]["running"]:
                    widgets.tag("RUNNING", theme.ACCENT)
                elif A["screen"]["error"]:
                    widgets.tag("FAILED", theme.BAD)
                elif A["screen"]["report"]:
                    widgets.tag("SCREENED", theme.GOOD)
            # the live sweep gets its OWN container: it redraws twice a second
            # while the screen runs, and redrawing the card around it would
            # take the focus out of whatever field is being typed into
            boxes["sweep"] = ui.column().classes("w-full gap-1")
            _render_sweep()
            if A["screen"]["error"]:
                widgets.hint(A["screen"]["error"], "bad")

    # ---------------------------------------------- the sweep, while it runs
    def _render_sweep():
        """WHAT THE SCREEN IS DOING, WHILE IT IS DOING IT.

        A shortlist re-screen is a live viscous XFOIL sweep per section —
        minutes on the first visit — and all that said so was one line in the
        status strip at the bottom of the window. Two things belong on the
        stage that asked for the sweep and nowhere else:

        * WHICH PASS is running. Pass 1 ranks the whole cached library to pick
          the shortlist and costs nothing; pass 2 is the sweep and costs the
          minutes. A single spinner covering both reads as a stall.
        * WHICH SECTIONS have come back, and with what. A section that failed
          to converge or was refused by a gate is progress too — and it is the
          honest reason a 24-section shortlist can rank fewer than 24 rows.

        Drawn into its own container so the twice-a-second redraw never
        rebuilds the number fields around it (see :func:`_ar_control`).
        """
        box = boxes.get("sweep")
        if box is None:
            return
        box.clear()
        sc = A["screen"]
        if not sc["running"]:
            return
        with box:
            widgets.hairline()
            phase, prog = sc.get("phase"), sc.get("progress")
            if not prog:
                # nothing has finished yet: on the two-pass route that IS the
                # library pass, and its length is not knowable from here
                ui.label("ranking the whole cached library to choose the "
                         "shortlist — a cache read, no XFOIL yet"
                         if phase == "library" else
                         "screening the library…") \
                    .classes("mono text-xs").style(f"color:{theme.INK_MUTED}")
                ui.linear_progress(value=0.0, show_value=False, size="6px") \
                    .props("indeterminate").classes("w-full")
                return
            i, n = int(prog[0]), int(prog[1])
            frac = (i / n) if n else 0.0
            own = _re_is_the_surface_own()
            with ui.row().classes("w-full items-center gap-2 no-wrap"):
                ui.label(f"{'sweeping the shortlist' if own else 'screening'}"
                         f" · {i}/{n}").classes("readout")
                ui.space()
                ui.label(f"{100.0 * frac:.0f} %").classes("mono text-xs") \
                    .style(f"color:{theme.INK_MUTED}")
            ui.linear_progress(value=max(0.0, min(1.0, frac)),
                               show_value=False, size="6px").classes("w-full")
            for row in reversed((sc.get("swept") or [])[-SWEEP_LOG_SHOWN:]):
                ok = row.get("status") == "ok" and row.get("eligible")
                text = (f"{row['i']:>3}  {row['name']}"
                        + (f"   L/D@Cl {fmt(row.get('ldcr'), 4)}"
                           f"   t/c {fmt(row.get('tc'), 4)}" if ok else
                           f"   {row.get('status') or 'refused'}"))
                ui.label(text).classes("mono text-xs").style(
                    f"color:{theme.INK_MUTED if ok else theme.WARN}")
            # counted off the PROGRESS index, not off the log: the log is
            # capped (SWEEP_LOG_MAX), so a long sweep would otherwise
            # under-report what it had already finished
            if i > SWEEP_LOG_SHOWN:
                ui.label(f"…and {i - SWEEP_LOG_SHOWN} earlier "
                         f"— the full ranking arrives when the sweep ends") \
                    .classes("mono text-xs").style(f"color:{theme.INK_FAINT}")

    # ------------------------------------------- the chord the section is for
    def _ar_control():
        """The aspect-ratio ESTIMATE, and the chord it puts the section at.

        It lives on this stage because that is the only thing it decides: a
        section has to be screened at SOME Reynolds number, and a Reynolds
        number needs a chord. The mission states no planform, and the span
        the run flies is stage 3's — so this number is never sent to a
        solver, and the note says which of the two Re's on screen it moves.

        The read-outs and the note are drawn into their OWN containers, so
        typing in the field re-renders them and not the field: a view that
        rebuilds its own input takes the focus away mid-number, and "12"
        arrives as "1".

        The FIELD is the wing stage's alone. The second surface's chord comes
        off its family's own design box (its area and aspect ratio are the
        run's variables, not an estimate), so an aspect-ratio field here
        would be a control that moves nothing on this stage — the note below
        says whose chord this surface flies instead.
        """
        widgets.hairline()
        # AIRFOIL-ONLY: the chord is STATED on stage 1, so there is no area
        # to turn into one and no estimate to make. The note below says
        # where the point came from instead.
        if session.airfoil_only(S):
            boxes["ar"] = ui.column().classes("w-full gap-1")
            _render_point_row()
            _render_ar_note()
            return
        # THE CAR HAS NO ESTIMATE TO MAKE. On every other family the mission
        # states no planform, so a chord needs a guess and this field is it.
        # A rear wing states TWO — its span and its reference area are the
        # b_m and S_m2 rows of the design box — so b²/S is a derived number,
        # and a field here would be a second answer that
        # `session.track_point_sync` overwrote on the next repaint.
        track = S["medium"] == "track" and not session.airfoil_only(S)
        if track and not SECONDARY:
            size = session.track_reference_size(S)
            widgets.readout("aspect ratio",
                            f"{session.section_aspect_ratio(S):.4g}", "b²/S",
                            tip="b²/S off the design box's own two size rows")
            widgets.hint(
                (f"Derived, not estimated: the design box searches b = "
                 f"{size[0]:.3g} m and S = {size[1]:.4g} m² at its middle, "
                 f"so this section is designed at the mean chord "
                 f"{size[1] / size[0]:.4g} m that implies. Narrow either row "
                 f"in stage 3 and this follows it."
                 if size else
                 "Derived from this car family's own planform."))
        elif not SECONDARY:
            widgets.number_field(
                "aspect ratio estimate",
                widgets.shown(session.section_aspect_ratio(S)), _set_ar,
                unit="b²/S", step=0.5, width="w-24",
                tip="MAC = sqrt(S / AR) at the mission's reference area — the "
                    "chord this section is being designed for")
        boxes["ar"] = ui.column().classes("w-full gap-1")
        _render_point_row()
        _render_ar_note()

    def _render_point_row():
        """The three head-line numbers of the design point."""
        row = boxes.get("point")
        if row is None:
            return
        row.clear()
        cond = session.section_conditions(S, SURFACE)
        dp = session.surface_design_point(S, SURFACE)
        with row:
            widgets.readout("screen at Cl", f"{cond['cl_design']:.4f}")
            widgets.readout("screen at Re", f"{cond['re']:.3e}")
            if session.airfoil_only(S):
                # the third number is the MACH here: it is the one part of
                # the stated flow that a search can be run with or without,
                # and it is otherwise invisible on this stage
                widgets.readout(
                    "screen at M", f"{cond['mach']:.4f}",
                    tip="the Mach number stage 1 states, if it is being "
                        "flown; XFOIL is given it verbatim")
                return
            if "error" not in dp:
                widgets.readout(
                    "mission Re", f"{dp['re_mac']:.3e}",
                    tip=("the mission's flow state at THIS surface's own "
                         "chord" if SECONDARY else
                         "what the mission implies at the MAC of the "
                         "estimated planform"))

    def _render_ar_note():
        box = boxes.get("ar")
        if box is None:
            return
        box.clear()
        if session.airfoil_only(S):
            pt = session.flow_point(S)
            with box:
                if "error" in pt:
                    widgets.hint(pt["error"], "bad")
                    return
                widgets.hint(
                    f"Stage 1 states this point: {pt['v_ms']:.4g} m/s in "
                    f"{session.fluid_phrase(S)} on a {pt['chord_m']:.4g} m "
                    f"chord — rho V c / mu = {pt['re']:.4g}. Nothing on this "
                    f"stage moves it; the flow is where it is asked.")
            return
        geo = session.surface_geometry(S, SURFACE)
        dp = session.surface_design_point(S, SURFACE)
        with box:
            if "error" not in dp and geo is not None and geo["lifting"]:
                # A TANDEM PAIR SHARES THE ESTIMATE'S SPAN, so neither of its
                # two wings gets the sentence below. `surface_geometry`
                # answers for the MAIN surface here (the pair splits one
                # reference area between two wings on one span, session.py's
                # ``area_split_front`` branch, b = sqrt(AR·S)), so the note
                # written for a surface sized out of the family's own box
                # used to fire directly beneath the estimate field and call
                # it inert — while typing 16 instead of 5 moved this wing's
                # quoted chord 1.0000 → 0.5590 m and its Re 9.996e+05 →
                # 5.588e+05. The estimate IS that shared span; what the split
                # decides is only which share of the area sits on it.
                widgets.hint(
                    f"This {_surface_name()}'s share of the pair: "
                    f"{geo['source']} → mean chord {geo['mac']:.4f} m → Re "
                    f"{dp['re_mac']:.3e} at {dp['v_ms']:.4g} m/s. Both wings "
                    f"sit on that one span, which is what stage 2's "
                    f"aspect-ratio estimate sets — so it DOES move this chord "
                    f"and this Reynolds number; the area split decides only "
                    f"which share of the area is on it.")
            elif "error" not in dp and geo is not None:
                # this surface's chord is NOT the mission-area/aspect-ratio
                # one: it comes off the family's own box, so the estimate
                # on the wing's stage does not move it, and the note says
                # whose chord this is
                widgets.hint(
                    f"This surface's own chord: {geo['source']} → mean chord "
                    f"{geo['mac']:.4f} m → Re {dp['re_mac']:.3e} at "
                    f"{dp['v_ms']:.4g} m/s. Stage 2's aspect-ratio estimate is "
                    f"the WING's, and does not move it.")
            elif SECONDARY:
                # no size of its own in the family's box: it is screened on
                # the wing's chord, and saying so is the honest version of a
                # number that would otherwise look like this surface's
                widgets.hint(
                    "This family's design box carries no size for the second "
                    "surface, so its section is screened at the WING's chord "
                    "and Reynolds number. The section is still its own.",
                    "warn")
            elif "error" not in dp:
                band = session.taper_re_band(S)
                txt = (f"MAC {dp['mac']:.4f} m on the mission's "
                       f"{dp['s_ref_m2']:.4g} m² → Re {dp['re_mac']:.3e} at "
                       f"{dp['v_ms']:.4g} m/s.")
                if band:
                    txt += (f" Across the taper the wing search may choose, "
                            f"that spans {band[0]:.3g} – {band[1]:.3g}.")
                    # the band varies TAPER only. A chord law reshapes the
                    # planform on top of the trapezoid, so the flown MAC can
                    # leave that interval — and the shell opens on one, so
                    # this is the default case, not an edge one
                    if S["wing"]["choices"].get("chord") == "free":
                        txt += (" The chord law reshapes the planform on top "
                                "of that trapezoid, so the flown MAC can sit "
                                "outside the interval; the run reports its "
                                "own mac_true / re_true beside the Re it "
                                "actually flew.")
                widgets.hint(txt)
            if A.get("re_source", session.RE_SOURCE_DEFAULT) == "library" \
                    and not A["override_point"]:
                widgets.hint(
                    "The screen below is running on the CACHED library point, "
                    + ("so the Reynolds number above is not the one it ranks "
                       "at (switch the source to use it)." if SECONDARY else
                       "so this estimate does not move it. It sets the "
                       "mission Re (switch the source to use it)"
                       + (" and the wing the wing-L/D objective flies on the "
                          "optimisation tab."
                          if session.wing_objective(S, SURFACE) else ".")))
                # ...and it is not only the RANKING that sits at the cached
                # point: a library polar is measured data at one Reynolds
                # number, so the section chosen here is FLOWN there in stage 3
                # too. Said here because this is where the choice is made.
                lib = session.library_point()
                if lib and "error" not in dp and abs(
                        float(dp["re_mac"]) - float(lib["re"])) \
                        > session.POINT_MOVE_TOL * float(lib["re"]):
                    widgets.hint(
                        f"Stage 3 will also FLY whatever is chosen here at "
                        f"Re {float(lib['re']):.3e} — a library section's "
                        f"polar exists only where it was swept. That is "
                        f"{float(dp['re_mac']) / float(lib['re']):.2g}x this "
                        f"surface's own Reynolds number.", "warn")
            # the estimate-versus-flown question is the WING's: this is the
            # only stage that owns the estimate, and the only surface stage 3
            # flies a chosen span for
            if SECONDARY:
                return
            size = session.flown_size(S)
            band = session.flown_ar_band(S)
            flown = session.flown_aspect_ratio(S)
            agree = abs(flown - session.section_aspect_ratio(S)) <= 1e-9 * max(
                1.0, abs(flown), abs(session.section_aspect_ratio(S)))
            if S["medium"] == "track":
                # the CAR searches BOTH size rows, so "carries its own
                # planform" — the branch this family used to fall into
                # below — was false about the one family it was reached by
                sz = session.track_reference_size(S)
                widgets.hint(
                    (f"Stage 3 SEARCHES both of this wing's dimensions: b "
                     f"over {session.span_box(S)[0]:.3g}–"
                     f"{session.span_box(S)[1]:.3g} m and S over its own "
                     f"row, so the aspect ratio it flies is whatever the "
                     f"answer's b²/S turns out to be. This one is the "
                     f"middle of that box ({sz[0] ** 2 / sz[1]:.3g})."
                     if sz and session.span_box(S) else
                     "Stage 3 flies this car family's own planform."))
                return
            widgets.hint(
                "An estimate, not a commitment: it is never sent to a solver. "
                + (("Stage 3 CONSTRAINS the span instead — searching "
                    f"{session.span_box(S)[0]:.3g}–{session.span_box(S)[1]:.3g}"
                    f" m at S = {size[1]:.4g} m², which is AR "
                    f"{band[0]:.3g}–{band[1]:.3g}"
                    + (". This estimate is the middle of that band."
                       if agree else
                       f", mid-box {flown:.3g} — not this one.")
                    if band else
                    f"Stage 3 flies b = {size[0]:.3f} m"
                    + (" — the wing this estimate implies." if agree else
                       f" at AR {flown:.3g}, which is not this one."))
                   if size else
                   "This family carries its own planform, so stage 3 flies "
                   "that."),
                "" if (agree or not size) else "warn")
            if size and not agree:
                _adopt_button()

    def _adopt_button():
        """Resolve an estimate/flown mismatch the only safe way round: the
        section follows the wing, because the flown planform is a decision
        and this number is not."""
        ui.button(f"design for the flown wing (AR "
                  f"{session.flown_aspect_ratio(S):.3g})",
                  icon="sync", on_click=_adopt_flown) \
            .props("flat dense size=sm no-caps")

    def _adopt_flown():
        if not session.set_section_aspect_ratio(
                S, session.flown_aspect_ratio(S)):
            return
        dp = session.design_point(S)
        ctx.log(f"aspect-ratio estimate now follows stage 3: "
                f"{session.section_aspect_ratio(S):.4g} — MAC "
                f"{dp.get('mac', float('nan')):.4f} m, mission Re "
                f"{dp.get('re_mac', float('nan')):.3e}", "info")
        _render_point_row()
        _render_ar_note()
        _render_ranking()
        _render_optimise()
        ctx.render_when_shown("mission")
        ctx.render_when_shown("wing")
        ctx.refresh()

    def _set_ar(e):
        if e.value in (None, ""):
            return
        if widgets.is_echo(e.value, session.section_aspect_ratio(S)):
            return
        if not session.set_section_aspect_ratio(S, e.value):
            ui.notify("aspect ratio must be positive", type="negative")
            return
        # a wing that is still FOLLOWING the estimate moves with it; one that
        # was given its own aspect ratio in stage 3 does not
        session.sync_wing_from_mission(S)
        dp = session.design_point(S)
        ctx.log(f"aspect-ratio estimate {session.section_aspect_ratio(S):.4g} "
                f"— MAC {dp.get('mac', float('nan')):.4f} m, mission Re "
                f"{dp.get('re_mac', float('nan')):.3e}", "info")
        rep = A["screen"]["report"] or {}
        screened = float((rep.get("conditions") or {}).get("re") or 0.0)
        now = float(session.section_conditions(S, SURFACE)["re"])
        if screened and abs(now - screened) > 1e-9 * max(now, screened):
            ctx.log(f"the ranking on screen was screened at Re {screened:.3e} "
                    f"— re-screen to rank at {now:.3e}", "warn")
        _render_point_row()
        _render_ar_note()
        ctx.render_when_shown("mission")
        ctx.render_when_shown("wing")
        # NOT this surface's own screen view: the estimate is typed into it,
        # and the two containers in it that quote the estimate were redrawn
        # by name two lines up.
        ctx.refresh((stage, "screen"))

    def _gate(label: str, key: str, step: float, why: str):
        """A gate that can be switched OFF entirely.

        Off is not a very small number typed by hand: it is sent as the
        no-gate value the screen understands (0 for a minimum, a huge number
        for a maximum), and the form says so in words.
        """
        on = A["cond"].get(key) is not None
        with ui.row().classes("w-full items-center gap-2 no-wrap"):
            ui.switch(value=on,
                      on_change=lambda e, k=key:
                      _toggle_gate(k, bool(e.value))).props("dense")
            widgets.explain(
                ui.label(label).classes("field-label").style("min-width:96px"),
                why, title=label)
            if on:
                ui.number(value=widgets.shown(A["cond"][key]), step=step,
                          on_change=lambda e, k=key: _set_cond(k, e.value)) \
                    .props("outlined dense hide-bottom-space") \
                    .classes("w-24 shrink-0")
            else:
                ui.label("no limit").classes("field-unit")

    def _nose_down_applies() -> bool:
        """Is the third gate a question on THIS form?

        It is exactly the "wing-trimmed" job: a wing that has a surface whose
        job is to trim it (:func:`session.nose_down_required` defaults on for
        that one case, and off for the trimming surface itself, for a tandem
        pair and for a tailless wing). Asked where the answer is always the
        same it would be a control that decides nothing, so the row appears
        only there — and it stays on screen once it is answered, because the
        job has not moved.
        """
        return session.surface_job(S, SURFACE) == "wing-trimmed"

    def _nose_down_gate():
        """THE THIRD HARD GATE, which had no control anywhere in the shell.

        ``nose_down`` refuses every REFLEXED section — the gated quantity is
        -cm_at, so the floor 0 admits only cm <= 0 (airfoil_select.
        NOSE_DOWN_FLOOR). It is on by default for a wing that has a tail,
        with a measured reason: |Cm| is a lower-better criterion, so ranking
        on it rewards reflex, and reflex is what a section does INSTEAD of
        having a tail — flown under a stabiliser it inverts its job. But it
        was a DEFAULT only in the docstring: no key held it, nothing wrote
        one, and it took 40 sections out of the shipped wing ranking (165 →
        125 eligible) with nothing on this form saying a third gate was live.

        Three states, not two, because "follow the configuration" is a real
        answer and is not the same as "on": adding a tail later turns it on,
        removing one turns it off, and an answered gate is never moved again.
        """
        if not _nose_down_applies():
            return
        state = A.get("nose_down")
        with ui.row().classes("w-full items-center gap-2 no-wrap"):
            widgets.explain(
                ui.label("nose-down only").classes("field-label")
                .style("min-width:96px"),
                "Sections with a nose-UP (reflexed) pitching moment are "
                "refused — this wing has a surface to trim it, and a "
                "reflexed section inverts that surface's job.",
                title="nose-down only")
            ui.toggle({"auto": "automatic", "on": "on", "off": "off"},
                      value=("auto" if state is None
                             else "on" if state else "off"),
                      on_change=lambda e: _set_nose_down(e.value)) \
                .props("dense no-caps unelevated toggle-color=primary")
        widgets.hint(
            "Automatic is ON here: this wing HAS a trimming surface, so a "
            "reflexed section would leave the stabiliser lifting. Switch it "
            "off to rank the reflexed sections too — 25.2 % of the library "
            "is nose-up at the published CG."
            if state is None else
            ("Set by you: reflexed sections are refused."
             if state else
             "Set by you: reflexed sections are ranked like any other, even "
             "though this wing has a surface whose job is to trim it."))

    def _set_nose_down(value: str):
        A["nose_down"] = {"auto": None, "on": True, "off": False}[value]
        ctx.log(f"nose-down gate for the {_surface_name()} set to {value} — "
                f"the next screen "
                + ("refuses" if session.nose_down_required(S, SURFACE)
                   else "ranks")
                + " reflexed (nose-up) sections", "info")
        _render_screen()
        ctx.refresh()

    def _recommended_note():
        """What this surface's weights SHOULD be, and whether they are.

        The recommendation is the surface's JOB, not the shell's taste: three
        of the four sets are ``airfoil_select.PRESETS`` verbatim — including
        GDP's own front-wing and rear-wing presets, which is what a tandem
        pair's two wings are. So it is stated with its reason, and restoring
        it is one click; an edited set is never moved by the shell again.
        """
        want, why = session.recommended_weights(S, SURFACE)
        job = session.surface_job(S, SURFACE)
        if session.weights_are_recommended(S, SURFACE):
            widgets.hint(f"Recommended for this surface ({job}): {why}.", "ok")
            return
        widgets.hint(f"These are YOUR weights — the recommended set for a "
                     f"{job} surface is "
                     + ", ".join(f"{WEIGHT_META[k][0]} {v:.2f}"
                                 for k, v in want.items() if v > 0.0)
                     + f": {why}.")
        ui.button("use the recommended weights", icon="tune",
                  on_click=_use_recommended_weights) \
            .props("flat dense size=sm no-caps")

    def _use_recommended_weights():
        want, why = session.recommended_weights(S, SURFACE)
        session.set_weights(S, SURFACE, want, "recommended")
        ctx.log(f"criterion weights for the {_surface_name()} set to the "
                f"recommended set — {why}", "info")
        _render_screen()
        ctx.refresh()

    def _trim_weight_note():
        """What a TRIMMING surface is ranked at, and the one gap that leaves.

        Its design lift is the trim balance's, not the mission's — stated
        here with the balance it came from, because it is the number every
        criterion on this form is evaluated at. A surface whose family will
        NOT say what it trims at falls back to zero lift, where "L/D at the
        design Cl" is cl/cd = 0 for every candidate: that weight then ranks
        nothing, and the warning says so rather than letting the score look
        like it used all six criteria.
        """
        geo = session.surface_geometry(S, SURFACE)
        if geo is None or geo["lifting"]:
            return
        if FIN:
            # A FIN'S ZERO LIFT IS AN ANSWER, not a missing one. The branch
            # below says "this family will not say what it trims at, so it is
            # screened at zero lift" — true of a stabiliser whose balance the
            # family cannot solve, and false of a vertical surface, which must
            # make NO side force at zero sideslip by construction
            # (session.surface_geometry). It read as a defect on the one
            # surface where the number is exactly right.
            widgets.hint(
                f"This {_surface_name()} makes no side force at zero "
                f"sideslip, so it is screened at Cl 0 — and that is what it "
                f"flies, not a fallback. The drag it costs there is the "
                f"“cd at design Cl” criterion; what it must still do at "
                f"DEFLECTION is Cl max and the stall angle.")
            return
        widgets.hint(session.TRIM_DRAG_NOTE)
        if geo.get("cl_source"):
            flown = geo.get("cl_flown")
            # SCREENED AT and CARRIES are two different numbers, and on an
            # inverted surface they differ in SIGN. `geo["cl"]` is where the
            # UPRIGHT catalogue is read; `geo["cl_flown"]` is the load the
            # surface carries in the aircraft's frame. This card used to call
            # `geo["cl"]` "what it carries" and then say the ranking was read
            # at `+abs(flown)` — three sentences that contradict each other the
            # moment the two stop coinciding, which is exactly the water
            # elevator (carries +0.2864, screened at −0.2864).
            widgets.hint(
                f"Screened at Cl {geo['cl']:+.4f} — where the upright "
                f"catalogue is read for this {_surface_name()} "
                f"({geo['cl_source']}).")
            if geo.get("inverted"):
                _fl = float(flown or 0.0)
                # an inverted MOUNT does not imply a download: a stated
                # mounting plus the trim balance can put the surface on an
                # up-load, and then it flies its camber the wrong way round
                _dir = "pushes DOWN" if _fl < 0.0 else "carries an UP-load"
                widgets.hint(
                    f"This {_surface_name()} is mounted INVERTED and "
                    f"{_dir} — it carries CL {_fl:+.4f} — so it flies its "
                    f"section upside down. The upright database is therefore "
                    f"read at {geo['cl']:+.4f}, the mirror of the load: an "
                    f"inverted section at CL {_fl:+.4f} is the catalogue "
                    f"section at {geo['cl']:+.4f}, the same point of the same "
                    f"curve and so the same drag. The winner is flown "
                    f"mirrored — drawn that way in the geometry views and "
                    f"exported that way to CAD."
                    + ("" if _fl < 0.0 else
                       " Note this surface is mounted inverted yet trims to "
                       "an up-load, so its camber works against it; upright "
                       "may be the better mounting here."))
            elif float(geo["cl"]) < 0.0:
                widgets.hint(
                    f"That is a DOWNLOAD: this {_surface_name()} pushes "
                    f"down, so every criterion here is read on the negative "
                    f"side of the polar. “L/D at the design Cl” is |cl|/cd — "
                    f"the efficiency of a surface pushing down is |L|/D.")
            return
        widgets.hint(
            f"This family will not say what its {_surface_name()} trims at, "
            f"so it is screened at zero lift.", "warn")
        if not float(A["weights"].get("ldcr", 0.0)) > 0.0:
            return
        widgets.hint(
            "“L/D at design Cl” is cl/cd, so at zero lift it is 0 for every "
            "candidate here — that weight ranks nothing.", "warn")

    def _toggle_gate(key: str, on: bool):
        A["cond"][key] = float(GATE_DEFAULTS[key]) if on else None
        _render_screen()
        ctx.refresh()

    def gate_value(key: str) -> float:
        """The number the screen actually runs with (off -> no gate)."""
        v = A["cond"].get(key)
        return float(GATE_OFF[key] if v is None else v)

    def _set_re_source(value: str):
        A["re_source"] = value
        if value == "mission" and _re_is_the_surface_own():
            cond = session.section_conditions(S, SURFACE)
            n = int(A.get("shortlist", session.SHORTLIST_DEFAULT))
            ctx.log(f"screening Reynolds number set to this "
                    f"{_surface_name()}'s own, {cond['re']:.3e} — the next "
                    f"screen sweeps the top {n} of the cached ranking there "
                    f"with live XFOIL, and the section chosen from it is "
                    f"FLOWN at that Reynolds number in stage 3", "warn")
        # a section chosen from the OLD point is still on the wing card, and
        # what it says (which Re the run flies it at) has just changed
        ctx.render_when_shown("wing")
        _render_screen()
        ctx.refresh()

    def _set_shortlist(value):
        if value in (None, ""):
            return
        if widgets.is_echo(value, A.get("shortlist",
                                        session.SHORTLIST_DEFAULT), 0):
            return
        A["shortlist"] = max(1, int(float(value)))
        _render_screen()
        ctx.refresh()

    def _set_override(value: bool):
        A["override_point"] = bool(value)
        if value:
            cond = session.section_conditions(S, SURFACE)
            A["cond"]["re"] = float(cond["re"])
            A["cond"]["cl_design"] = float(cond["cl_design"])
        _render_screen()
        ctx.refresh()

    def _set_cond(key: str, value):
        if value in (None, ""):
            return
        nd = 0 if key == "re" else widgets.FIELD_DIGITS
        if widgets.is_echo(value, A["cond"][key], nd):
            return
        A["cond"][key] = float(value)
        # the overridden point is TYPED into this view, so the view may not
        # be rebuilt from its own field's handler
        ctx.refresh((stage, "screen"))

    def _weight_of(key: str) -> float:
        """This surface's weight on one criterion — 0 where its preset does
        not mention it.

        A preset states the criteria that surface is FOR (session.JOB_WEIGHTS)
        and no more, so a form built from the full criterion set reads keys
        that are simply absent. Absent is zero; it is not a KeyError, and it is
        not a reason to hide the row.
        """
        return float(A["weights"].get(key, 0.0) or 0.0)

    def _dead_weight_note():
        """The criteria that cannot rank THIS surface, said once.

        A criterion at zero because the surface cannot be ranked on it is not
        the same fact as a criterion the user zeroed, and the difference is
        invisible on a slider. It matters most in the other direction: a user
        who drags one of these UP buys nothing, and the score goes on looking
        as though it used every criterion.
        """
        dead = DEAD_CRITERIA.get(SURFACE) or {}
        if not dead:
            return
        widgets.hint(
            "On this surface " + _join_names(
                [WEIGHT_META[k][0] for k in dead if k in WEIGHT_META])
            + " cannot rank anything — the rows stay so you can see that "
              "they are zero, and each says why on hover.")
        for key, why in dead.items():
            if _weight_of(key) > 0.0:
                widgets.hint(f"“{WEIGHT_META[key][0]}” carries "
                             f"{_weight_of(key):.2f} here, and {why}", "warn")

    def _set_weight(key: str, value):
        A["weights"][key] = float(value or 0.0)
        # an edited weight set is the USER's, and the shell never moves one
        # again — not when the family changes, not when the surface's job
        # does (session.refresh_recommended_weights)
        A["weights_source"] = "user"
        lab = weight_labels.get(key)
        if lab is not None:
            lab.set_text(f"{A['weights'][key]:.2f}")

    def _set_floor(key: str, value):
        A["floors"][key] = None if value in (None, "") else float(value)

    # ------------------------------------------------- which surface, if two
    def _surface_name() -> str:
        # ONE table (session.surface_name), because the same name is written
        # by the tag, the buttons, the hints and every log line here
        return session.surface_name(S, SURFACE)

    def _surface_row():
        """What this stage is designing, said once at the top of the form.

        There is no target switch: the SURFACE IS THE STAGE. What this row
        does instead is name the surface, quote the size the design point
        below comes from, and offer the other surface's stage — so the two
        answers are one click apart and never one control apart.
        """
        # WHETHER there is more than one surface to design — which is a
        # vertical stabiliser as much as a horizontal one. Asking
        # ``aft_surface`` alone dropped this whole row (the name of the
        # surface, and the way back to the wing's section) on a vehicle whose
        # only second surface is the fin.
        nxt = session.next_section_stage(S, stage)
        if not (SECONDARY or nxt):
            return
        with ui.row().classes("w-full items-center gap-3 no-wrap"):
            ui.label("designing").classes("field-label").style(
                f"color:{theme.INK_MUTED};min-width:80px")
            widgets.tag(_surface_name().upper(), theme.ACCENT)
            if nxt:
                # the NEXT surface in the pipeline, not "the aft one": on a
                # vehicle with both tails this steps wing -> tail -> fin
                nxt_name = session.surface_name(
                    S, session.STAGE_SURFACE[nxt])
                ui.button(f"the {nxt_name}'s section →", icon="arrow_forward",
                          on_click=lambda n=nxt: ctx.select(n, "screen")) \
                    .props("flat dense size=sm no-caps")
            if SECONDARY:
                ui.button("← the wing's section", icon="arrow_back",
                          on_click=lambda: ctx.select("airfoil", "screen")) \
                    .props("flat dense size=sm no-caps")
        if not SECONDARY:
            return
        if not S["airfoil"].get(SECTION_KEY):
            widgets.hint(f"The {_surface_name()} currently flies the wing's "
                         f"section. Screening here gives it one of its own; "
                         f"nothing else in the pipeline waits on it.")
        geo = session.surface_geometry(S, SURFACE)
        if geo:
            widgets.hint(
                f"Its own size: {geo['source']} → mean chord "
                f"{geo['mac']:.4f} m."
                + ("" if geo["lifting"] else " " + session.TRIM_CL_NOTE))

    def _clear_aft():
        session.set_section(S, None, None, surface=SURFACE)
        ctx.log(f"the {_surface_name()} follows the wing's section again",
                "info")
        _render_ranking()
        _render_section()
        ctx.render_when_shown("wing")
        ctx.refresh()

    # ===================================================== view: ranking
    def _render_ranking():
        box = ctx.views[(stage, "ranking")]
        box.clear()
        rep = A["screen"]["report"]
        with box:
            _surface_row()
            if not rep:
                widgets.hint("Nothing screened yet — run the screening on "
                             "the previous tab.")
                return
            cond = rep.get("conditions") or {}
            point = rep.get("point") or {}
            short = rep.get("shortlist") or {}
            with ui.row().classes("w-full items-start gap-4 no-wrap"):
                widgets.readout(
                    "eligible", str(rep.get("n_eligible", 0)),
                    f"of {rep.get('n_screened', 0)}",
                    help="How many library sections cleared every GATE and "
                         "so could be ranked at all, out of how many were "
                         "measured.\n\nA gate REFUSES; the weights only "
                         "order what is left. A small number here means the "
                         "gates decided the answer, not the weights.")
                widgets.readout(
                    "design Cl", fmt(cond.get("cl_design"), 4),
                    help="The lift coefficient every lift-dependent metric "
                         "in the table was re-read at — this surface's own, "
                         "from the mission and the estimated planform.\n\n"
                         "It is not the polar's best point: a section is "
                         "ranked on what it does at the lift it will fly.")
                widgets.readout(
                    "Re", f"{float(cond.get('re', 0.0)):.3e}",
                    "swept" if short.get("source") == "library"
                    else "cached",
                    help="The Reynolds number the polars were read at. "
                         "\"cached\" is the library's own point (Re 1e6); "
                         "\"swept\" means the shortlist was re-flown "
                         "here.\n\nIt matters: hg40 is L/D 80.6 at Re 1e6 "
                         "and 42.2 at 3e5, so a small chord ranked on "
                         "library polars is ranked on a section that does "
                         "not exist at its own Reynolds number.")
                widgets.readout(
                    "wall", f"{rep.get('wall_time_s', 0.0):.1f}", "s",
                    help="What this screen cost in wall-clock seconds. A "
                         "cached-point screen is a table lookup; a swept "
                         "one is two live viscous XFOIL marches per "
                         "section.")
            # WHICH GATE TOOK THE MISSING SECTIONS. The two numeric gates are
            # quoted on the form and in the table's own columns; the third one
            # refuses on the SIGN of the pitching moment and is otherwise
            # invisible — 40 sections left the shipped wing ranking (165 → 125
            # eligible) with nothing anywhere naming it. Read off the REPORT's
            # own floors, exactly like the weights below: the gate row may
            # have been moved since, and this table is the answer the gate it
            # ran under produced.
            if (rep.get("floors") or {}).get(api.SCREEN_NOSE_DOWN_KEY) \
                    is not None:
                widgets.hint(
                    f"The nose-down gate was live for this screen: every "
                    f"REFLEXED (nose-up) section was refused before ranking, "
                    f"so the {rep.get('n_eligible', 0)} of "
                    f"{rep.get('n_screened', 0)} above is over the sections "
                    f"that cleared it as well as the two gates on the form. "
                    f"It is the default for a wing that has a surface to trim "
                    f"it; the screening form's third gate row turns it off.")
            # WHAT THIS TABLE IS AN ORDER OVER. A shortlist re-screen ranks
            # only the sections that were actually swept at this Reynolds
            # number; saying so is the difference between a shortlist and a
            # silently truncated library.
            if short.get("source") == "library":
                widgets.hint(short.get("note", ""), "ok")
            elif short.get("source") == "none":
                widgets.hint(short.get("note", ""), "warn")
            if point.get("excluded"):
                widgets.hint(
                    f"{int(point['excluded'])} section(s) have no cached "
                    f"polar branch, so their lift-dependent metrics could not "
                    f"be re-scored at this design Cl. They are excluded from "
                    f"the ranking rather than ranked on numbers belonging to "
                    f"another lift coefficient.", "warn")
            # a ranking is an order AT one point; if the point has moved
            # since (the mission, or the chord the estimate implies), the
            # table below is the answer to a question nobody is asking now
            stale = session.section_point_is_stale(S, SURFACE)
            # ...and the DESIGN LIFT moves on its own: on the cached library
            # point both surfaces share a Reynolds number, so the Re test
            # cannot see this surface turning from a lifting one into a
            # trimming one (a tandem rear wing becoming a tail). The lift
            # coefficient did move, and that reorders the whole table.
            lift = session.section_lift_is_stale(S, SURFACE)
            if stale or lift:
                with ui.row().classes("w-full items-center gap-2 no-wrap"):
                    widgets.hint(
                        (f"Screened at Re {stale[0]:.3e}; this design point "
                         f"is now Re {stale[1]:.3e}. " if stale else "")
                        + (f"Screened at Cl {lift[0]:.4f}; this surface's "
                           f"design lift is now {lift[1]:.4f}. " if lift
                           else "")
                        + "The order below is the one that point produced.",
                        "warn")
                    ui.button("re-screen", icon="refresh",
                              on_click=start_screen) \
                        .props("flat dense size=sm no-caps")
            # where the shortlist was re-swept, the column that carries the
            # whole argument: what this section ranked at the cached point,
            # beside where it lands at the point it will actually fly
            cols = list(RANK_COLUMNS)
            reordered = any(r.get("rank_library") is not None
                            for r in (rep.get("ranked") or []))
            if reordered:
                cols = cols[:2] + [("rank_library", "# @cache", 0)] + cols[2:]
            # THE WEIGHTS THE TABLE WAS RANKED ON, which are the report's own
            # and not the sliders': the user may have moved those since, and
            # the order on screen is the one the report's weights produced
            # (the re-screen banner above is what says they have diverged).
            ranked_weights = rep.get("weights") or {}
            rows = []
            for i, r in enumerate(rep.get("ranked") or []):
                row = {"idx": i, "rank": i + 1}
                for key, _lbl, nd in cols:
                    if key == "rank":
                        continue
                    v = r.get(key)
                    row[key] = ("—" if v is None else
                                v if isinstance(v, str)
                                else fmt(v, nd if nd is not None else 4))
                row.update(weighted_points(r, ranked_weights))
                rows.append(row)
            columns = ranking_columns(cols, ranked_weights)
            table = ui.table(columns=columns, rows=rows, row_key="idx") \
                .classes("w-full").props("dense flat bordered "
                                         "virtual-scroll")
            points_slots(table, cols)
            table.on("rowClick", lambda e: _adopt_row(int(e.args[1]["idx"])))
            widgets.hint("Click a row to make it the section this design "
                         "carries forward. The score is the weighted "
                         "composite of the six criteria; the columns beside "
                         "it are the raw numbers it was built from, and the "
                         "faint number after each is what that criterion put "
                         "INTO the score (its weight × its 0-100 sub-score). "
                         "The six add up to the score, so a row can be read "
                         "for where its score came from as well as for how "
                         "big it is. A criterion you gave no weight is drawn "
                         "dim and contributes nothing — its column ranked "
                         "none of these rows.")
            sel = _candidate(int(A["screen"]["selected"] or 0))
            if sel:
                widgets.hint(f"carried forward: {sel.get('name')} "
                             f"(rank {sel.get('rank')})")

    # ===================================================== view: section
    def _render_section():     # noqa: PLR0915
        box = ctx.views[(stage, "section")]
        box.clear()
        # the export card's handle belongs to THIS build: the no-section
        # branch below returns before the card is drawn, and a stale handle
        # would let a keystroke on another stage draw into a dead container
        # (a_page_driver_must_re_resolve_on_rebuild)
        boxes.pop("export", None)
        # the section of THIS surface — and on the second surface that is
        # the wing's until one is chosen here, which is exactly what the
        # solver does with it (polar_tail / polar_rear = None)
        own = session.section_is_own(S, SURFACE)
        sec = (S["airfoil"].get(SECTION_KEY) if SECONDARY
               else A["section"])
        with box:
            if not sec:
                if SECONDARY:
                    inherited = session.section_of(S, SURFACE) or {}
                    widgets.hint(
                        f"The {_surface_name()} has no section of its own, so "
                        f"it flies the wing's"
                        + (f" ({inherited['name']})." if inherited.get("name")
                           else ".")
                        + " Screen the library here to give it one.")
                    return
                widgets.hint("No section chosen yet — screen the library, "
                             "then click a row in the ranking. Leaving it "
                             "unanswered is an answer too: the wing family "
                             "then flies its own published polars, and the "
                             "wing stage says which those are.")
                return
            rep = sec.get("report") or {}
            with ui.row().classes("w-full items-start gap-4 no-wrap"):
                widgets.readout("section", str(sec.get("name", "—")))
                if sec.get("tc"):
                    widgets.readout("t/c", fmt(sec["tc"], 4))
                if sec.get("ldcr") is not None:
                    widgets.readout("L/D @ Cl", fmt(sec["ldcr"], 4))
                if sec.get("clmax") is not None:
                    widgets.readout("Cl max", fmt(sec["clmax"], 3))
                widgets.readout("source", str(sec.get("source", "—")))

            # SHAPE AND POLAR TOGETHER, in the aircraft's axes. Flipping the
            # picture and leaving the curves upright is worse than flipping
            # neither: the reader then sees an aerofoil arching down whose
            # lift curve still says it makes positive lift, and reads that
            # as the sign error it is not.
            mounted = _mounted_report(rep)
            # THE SECTION'S OWN OUTLINE WINS. The report's shape block is
            # built from the CST refit at dz_te = 0, so for the rank-1 pick
            # it draws the same closed trailing edge the ranked rows used to
            # get — on aerofoils that are open by up to 0.0079c. Where the
            # coordinates are stored, they are the picture; the report figure
            # (which also carries the anchor overlay) is the fallback.
            shape = None
            if sec.get("coords") is not None:
                shape = _as_mounted(_shape_from_weights(sec))
            if shape is None and mounted:
                shape = v1.fig_section_shape(mounted)
            if shape is None and sec.get("w_upper") and sec.get("w_lower"):
                shape = _as_mounted(_shape_from_weights(sec))
            shared = _shared_with_inverted_surface()
            shape = shares_section_with(shape, shared)
            with widgets.group_box("Shape", pad=False):
                if shape is None:
                    with ui.column().classes("group-pad w-full"):
                        widgets.hint("No coordinates available for this "
                                     "section.")
                else:
                    figstyle.show(shape, f"section_{sec.get('name', 'x')}")
            if shared:
                widgets.hint(
                    f"This aerofoil is flown TWICE. The wing flies it the "
                    f"way it is drawn; the {shared} has no section of its "
                    f"own, so it flies THIS one — mounted upside down, "
                    f"because it pushes down (CL {_aft_cl():+.4f}). What it "
                    f"flies is this section mirrored about its chord line, "
                    f"and that mirrored one is what the geometry views, the "
                    f"STL and the OpenVSP model are built from — it is not "
                    f"drawn here, because this panel is the wing's. Give the "
                    f"{shared} a section of its own on its own stage if you "
                    f"want the two chosen separately.")
            if _is_inverted():
                widgets.hint(
                    f"Drawn AS MOUNTED, in the aircraft's axes: this "
                    f"{_surface_name()} pushes down, so its section flies "
                    f"upside down and its lift curve goes with it — at the "
                    f"trim angle it makes NEGATIVE lift, which is the whole "
                    f"point. The catalogue stores it the other way up; the "
                    f"ranking is read there, at the same |Cl|, because an "
                    f"upright section at +Cl IS this one at −Cl.")

            polar = v1.fig_section_polars(mounted) if mounted else None
            with widgets.group_box("Viscous polar", pad=False):
                if polar is None:
                    with ui.column().classes("group-pad w-full"):
                        widgets.hint(
                            "The screening report carries the full polar for "
                            "its WINNER only — the ranked rows carry their "
                            "metrics, not their curves. Re-weight the "
                            "criteria so this section wins, or optimise from "
                            "it on the next tab, to see the sweep.")
                else:
                    figstyle.show(polar, "section_polar")

            _render_export(sec)

            next_stage, next_view, next_label = _next_stage()
            section_only = session.airfoil_only(S)
            with widgets.group_box("This section" if section_only
                                   else "Carry into the wing stage"):
                if section_only:
                    widgets.hint(
                        "This session designs a section and nothing else, so "
                        "there is no wing stage for it to travel to. Refine "
                        "the shape with live XFOIL on the next tab, export "
                        "its coordinates from the section view, or state a "
                        "different flow on stage 1 and screen again.")
                if not section_only:
                    widgets.hint(
                        f"This section travels to the run as the "
                        f"{_surface_name()}'s own: the solver flies it on "
                        f"that surface alone, and the wing keeps the section "
                        f"chosen on stage 2."
                        if SECONDARY else
                        "The wing stage links this section into its design "
                        "box wherever the family can hold it: pinned CST "
                        "weights on a designed-section wing, a narrowed t/c "
                        "where the wing flies thickness, and advisory only "
                        "where the family's section is fixed.")
                    if next_stage != "wing":
                        widgets.hint(
                            "This configuration carries a second surface, "
                            "and it has its own section to choose — that "
                            "stage comes next, and the wing stage after it.")
                with ui.row().classes("items-center gap-2"):
                    ui.button(next_label, icon="arrow_forward",
                              on_click=lambda: ctx.select(next_stage,
                                                          next_view)) \
                        .props("unelevated dense no-caps color=primary")
                    ui.button("Refine this shape with XFOIL",
                              icon="auto_graph",
                              on_click=lambda: ctx.select(stage,
                                                          "optimise")) \
                        .props("outline dense no-caps")
                    if SECONDARY and own:
                        ui.button(f"Let the {_surface_name()} fly the wing's "
                                  f"section", icon="link_off",
                                  on_click=_clear_aft) \
                            .props("flat dense no-caps")

    # ------------------------------------------------- the section, as files
    #
    # THE SECTION IS A DELIVERABLE, not only an input to stage 3. Stage 4
    # exports the aircraft, and an aerofoil-only session never reaches it —
    # so the answer this stage produced could not leave the shell at all
    # except as a screenshot. It can now, in the three forms a section is
    # actually asked for (:data:`session.SECTION_EXPORT_KINDS`).
    #
    # WHERE the files go is asked ONCE. A browser download needs no folder,
    # so it is always offered; writing to this machine needs a path, and that
    # path is stage 4's question (``session.export_dir`` / ``export_stem``,
    # shared state). In a vehicle session this card quotes it and stage 4
    # asks it; in an aerofoil-only session there IS no stage 4, so the card
    # asks it here instead. One question, one place, in both.
    def _render_export(sec: dict):
        """The card SHELL — the download buttons, and the two fields when
        this session asks them here. Built with the section view.

        Split from what the fields derive, for the reason in
        a_card_that_draws_once_goes_stale: the folder and the name feed a
        "writes" read-out and a "this would replace" warning, and drawing
        those in the same container as the inputs meant they named the
        PREVIOUS answer until something else redrew the whole view.
        """
        got = session.section_export_files(S, SURFACE)
        section_only = session.airfoil_only(S)
        with widgets.group_box("Export this section"):
            with ui.row().classes("items-center gap-2 flex-wrap"):
                ui.label("download:").classes("field-unit")
                for kind in session.SECTION_EXPORT_KINDS:
                    label, _why = session.SECTION_EXPORT_LABELS[kind]
                    btn = ui.button(
                        label, icon="download",
                        on_click=(lambda k=kind: _download_section(k))) \
                        .props("outline dense no-caps")
                    if kind not in got["files"]:
                        btn.disable()
            widgets.hint(" · ".join(
                f"{session.SECTION_EXPORT_LABELS[k][0]} — "
                f"{session.SECTION_EXPORT_LABELS[k][1]}"
                for k in session.SECTION_EXPORT_KINDS))
            # a kind that cannot be written says WHY, beside the button that
            # is greyed out — a missing control tells the user nothing
            for kind, why in got["absent"].items():
                widgets.hint(f"{session.SECTION_EXPORT_LABELS[kind][0]}: "
                             f"{why}.", "warn")

            if section_only:
                with ui.row().classes("w-full items-center gap-2 no-wrap"):
                    # caption beside the field, not Quasar's floating label,
                    # which lands on the value in a 26 px box
                    # (widgets.text_field) — and the fields hold WHAT WAS
                    # TYPED, never rebuilt by their own handler
                    # (a_typed_number_must_not_rebuild_its_field)
                    widgets.text_field(
                        "save to", session.export_dir_raw(S),
                        lambda e: _set_export_dir(e.value), grow=True)
                    widgets.text_field(
                        "name", session.export_stem_raw(S),
                        lambda e: _set_export_stem(e.value), width="w-48")
            boxes["export"] = ui.column().classes("w-full gap-2")
        _render_export_derived()

    def _render_export_derived():
        """Where those files land and what they are called — re-derived.

        Redrawn on every keystroke in the two fields above, and after a save
        (which changes what would be REPLACED). Never contains an input.
        """
        box = boxes.get("export")
        if box is None:
            return
        box.clear()
        got = session.section_export_files(S, SURFACE)
        section_only = session.airfoil_only(S)
        plan = session.section_export_plan(S, SURFACE)
        with box:
            if not section_only:
                widgets.readout("save to", str(plan["dir"]))
            with ui.row().classes("items-center gap-2 flex-wrap"):
                save = ui.button("Save to this machine", icon="save",
                                 on_click=_save_section_export) \
                    .props("unelevated dense no-caps")
                if not got["files"]:
                    save.disable()
                names = ", ".join(sorted(p.name for p in
                                         plan["names"].values()))
                if names:
                    widgets.readout("writes", names)
            if plan["existing"]:
                widgets.hint("this would replace " + ", ".join(
                    plan["existing"]), "warn")
            if not section_only:
                widgets.hint("The folder and the name are asked once, on "
                             "stage 4 · Results, and every file this session "
                             "writes shares them.")
            if export_state["note"]:
                widgets.hint(export_state["note"], export_state["level"])

    def _set_export_dir(value):
        session.set_export_dir(S, value)
        _export_answer_moved()

    def _set_export_stem(value):
        S.setdefault("export", {})["stem"] = str(value or "")
        _export_answer_moved()

    def _export_answer_moved():
        """Re-derive what the card SAYS, never the field it was typed in.

        The folder is the SESSION's — stage 4's CAD card quotes the same
        answer — so that view is owed a repaint too, deferred until it is
        looked at.
        """
        _render_export_derived()
        ctx.render_when_shown("results", "log")

    def _download_section(kind: str):
        got = session.section_export_files(S, SURFACE)
        f = got["files"].get(kind)
        if f is None:
            ui.notify(got["absent"].get(kind, "there is nothing to export"),
                      type="warning")
            return
        ui.download.content(f["text"], f["name"])
        ctx.log(f"exported {f['name']}", "ok")

    def _save_section_export():
        """Write every file this section can produce into the chosen folder."""
        try:
            paths = session.write_section_export(S, SURFACE)
        except (OSError, ValueError) as exc:
            export_state.update(note=f"could not write: {exc}", level="bad")
            ctx.log(f"section export failed: {exc}", "error")
            _render_export_derived()
            return
        if not paths:
            export_state.update(
                note="there is nothing to write yet", level="warn")
            _render_export_derived()
            return
        export_state.update(
            note=f"wrote {', '.join(p.name for p in paths)} to {paths[0].parent}",
            level="ok")
        ctx.log(f"section export: wrote {len(paths)} file(s) to "
                f"{paths[0].parent}", "ok")
        # the card, not the whole section view: what moved is what would be
        # REPLACED next time, and the two fields must survive the answer
        _render_export_derived()

    def _is_inverted() -> bool:
        """Does THIS stage's surface fly its section upside down?"""
        geo = session.surface_geometry(S, SURFACE)
        return bool(geo and geo.get("inverted"))

    def _as_mounted(fig):
        """This surface's preview, drawn the way the surface mounts it."""
        return mirror_section_fig(fig, _is_inverted())

    def _mounted_report(rep: dict):
        """The screening report restated in the aircraft's axes."""
        return mirror_section_report(rep, _is_inverted()) if rep else None

    def _aft_cl() -> float:
        trim = session.trim_lift(S)
        return 0.0 if trim is None else float(trim["cl"])

    def _shared_with_inverted_surface() -> str:
        """Name of a second surface that flies THIS section, upside down.

        Empty otherwise. This is the case the shell used to say nothing
        about: with no aerofoil of its own the stabiliser flies the WING's
        section (stage 1 says so in words), so the only aerofoil on screen
        anywhere is the wing's — drawn upright, correctly for the wing — and
        a reader looking for "the stabiliser's aerofoil" finds that one and
        reads it as the tail's, the wrong way up. It IS the tail's; the tail
        just mounts it mirrored, and nothing here showed that.
        """
        if SURFACE != "main":
            return ""
        if session.section_is_own(S, "aft"):
            return ""                      # it has its own stage; not shared
        trim = session.trim_lift(S)
        if not (trim and trim.get("inverted")):
            return ""
        return session.second_surface_name(S) or "second surface"

    def _shape_from_weights(sec: dict):
        """Draw a ranked section — its own outline where there is one."""
        import numpy as np

        coords = section_outline(sec)
        if coords is None:
            return None
        # v1.fig_airfoil emptiness-checks with ``if not coords``, which a
        # numpy array answers with a ValueError — hand it a plain list
        fig = v1.fig_airfoil(np.asarray(coords, dtype=float).tolist())
        if fig is not None:
            fig.update_layout(height=260)
        return fig

    # ================================================== view: optimisation
    def _conv_fig() -> go.Figure:
        """The best section so far, on an axis a REFUSAL cannot own.

        The first evaluations of a CST search are routinely shapes XFOIL
        cannot fly, and the best-so-far is the -100 sentinel until one of
        them converges. Drawn on one linear axis, that opening cliff is 100
        units tall and the climb the user is watching — 55 -> 58 — is a flat
        line at the top of it. The axis is therefore set from the scored
        points, and the refused opening is drawn on the floor rather than
        scaled to (:func:`gui.metrics.convergence_yrange`).
        """
        recs = A["opt"]["records"]
        prior = [float(v) for v in (A["opt"].get("prior") or [])]
        if not recs and not prior:
            return figstyle.empty("no evaluations yet", 300)
        xs = [r["n"] for r in recs]
        ys = [r["best"] for r in recs]
        fig = go.Figure()
        if prior:
            # the run being continued, muted and dashed: context, not the
            # answer — and what stops the panel going blank while the
            # re-flown prefix comes back out of the XFOIL cache
            fig.add_scatter(x=list(range(1, len(prior) + 1)), y=prior,
                            mode="lines", name="the run this continues",
                            line=dict(color=theme.INK_MUTED, width=1.5,
                                      dash="dot"))
        if recs:
            fig.add_scatter(x=xs, y=ys, mode="lines",
                            line=dict(color=theme.ACCENT, width=2),
                            name="best so far")
        fig.update_layout(xaxis_title="evaluation",
                          yaxis_title="best objective", height=300)
        # THE PREFIX A CONTINUATION RE-FLEW, marked. Over it the curve is the
        # previous run's curve; the shading is where the old search ended and
        # the new evaluations began, which is what "did continuing buy
        # anything?" is asking.
        replay = int(A["opt"].get("replay") or 0)
        if replay >= 1:
            fig.add_vrect(x0=0.5, x1=replay + 0.5, line_width=0,
                          fillcolor=theme.INK_MUTED, opacity=0.12,
                          annotation_text=f"re-flown ({replay})",
                          annotation_position="top left",
                          annotation_font_size=10)
        # the trace IS the best-so-far, so every scored point on it is
        # a `must_show`: a percentile floor may drop a constrained
        # family's outliers, never the curve the user is reading
        rng = metric_catalogue.convergence_yrange(ys, ys)
        if rng is not None:
            lo, hi = rng
            off = metric_catalogue.offscale(ys, lo)
            if off:
                fig.add_scatter(
                    x=[xs[i] for i, _ in off], y=[lo + 0.02 * (hi - lo)] * len(off),
                    mode="markers", name=f"refused ({len(off)})",
                    marker=dict(symbol="x", size=7, color=theme.BAD,
                                opacity=0.85),
                    customdata=[v for _, v in off],
                    hovertemplate="eval %{x}<br>best = %{customdata:.5g}"
                                  "  (below the axis)<extra></extra>")
                fig.update_yaxes(range=[lo, hi])
        return fig

    def _render_recommended_budget(eff: dict):
        """The budget when stage 1 owns it (V3.5) — a read-out, not a field.

        The number is a function of THIS surface's own design vector: the 8
        CST weights, plus the twist and chord coefficients the wing-L/D
        objective opens. Change the objective and it moves, which is the
        reason it is not a box the user has to re-type.
        """
        plan = eff.get("plan")
        k = max(1, int(eff.get("n_restarts", 1)))
        with ui.row().classes("w-full items-start gap-4 flex-wrap"):
            widgets.readout("budget", str(int(eff["budget"])), "evaluations")
            if k > 1:
                widgets.readout("restarts", f"{k} ×",
                                tip="independent searches at the same total "
                                    "spend; the best one is the answer")
            widgets.readout("design variables", str(plan.dim) if plan else "—")
            widgets.readout("expected",
                            (plan.est_text if k == 1 else
                             f"{plan.est_text} × {k}") if plan else "—")
        if plan is not None:
            widgets.hint(plan.why + ".")
        if session.search_state(S).get("stop_when_converged") \
                and plan is not None and plan.patience:
            widgets.hint(
                f"It stops early if the best section has not improved by "
                f"{plan.tol:.1%} of the span it has covered in "
                f"{plan.patience} evaluations — every XFOIL sweep already "
                f"paid for is kept.")
        with ui.row().classes("items-center gap-2 no-wrap"):
            ui.button("use my own values", icon="edit",
                      on_click=lambda: ctx.act("adopt_search_values")) \
                .props("flat dense no-caps")
            ui.button("stage 1 · Search & budget", icon="north_east",
                      on_click=lambda: ctx.select("mission", "search")) \
                .props("flat dense no-caps")

    def _progress_text() -> str:
        """The RUNNING chip's text — one place, because the tick rewrites it
        without rebuilding the row it sits in.

        A CONTINUATION says so while it is inside its re-flown prefix. Those
        evaluations are the ones the run being lengthened already paid for,
        served out of the XFOIL cache, so they go by in seconds — and a
        counter racing from 1 to 12 with no label on it is exactly what
        "it started over" looks like.
        """
        oc = A["opt"]
        budget = int(oc.get("run_budget")
                     or session.effective_airfoil_search(S, SURFACE)["budget"])
        n = int(oc["progress"] or 0)
        replay = int(oc.get("replay") or 0)
        if replay and n < replay:
            return f"RUNNING · re-flying {n}/{replay} already paid for"
        # a RESUMED run's own counter already starts past what it inherited
        # (``api.run`` counts from there), so this needs no offset of its own
        return f"RUNNING · {n}/{budget}"

    #: WHAT THE OPTIMISE VIEW'S STRUCTURE DEPENDS ON. An XFOIL search takes
    #: minutes and repaints twice a second; while it runs only numbers move
    #: (the chip, the two traces), so the view is BUILT when this tuple moves
    #: and updated in place otherwise (:func:`_tick_optimise`). Rebuilding it
    #: per evaluation collapsed the scroll pane's content height, and the
    #: browser answered by clamping the page back to the top — twice a
    #: second, for the whole run.
    def _opt_sig():
        oc = A["opt"]
        return (bool(oc["running"]), bool(oc.get("stopped")),
                bool(oc.get("error")), oc.get("stamp"),
                oc.get("score_stamp"), bool(oc.get("report")),
                bool(oc.get("records")), bool(live_cfg["on"]),
                tuple(live.keys()), bool(live.error))

    def _tick_optimise():
        """Update an optimise view already on screen: numbers and traces.

        Falls back to the full build the moment the structure moves — the
        run finished, was stopped, failed, or the sampler learned a metric
        it had not reported before.
        """
        if boxes.get("opt_sig") != _opt_sig():
            _render_optimise()
            return
        if boxes.get("opt_chip") is not None:
            boxes["opt_chip"].set_text(_progress_text())
        if boxes.get("opt_samples") is not None:
            boxes["opt_samples"].set_text(f"{len(live.samples)} samples")
        figstyle.update(boxes.get("opt_conv"), _conv_fig(),
                        "section_convergence")
        figstyle.update(boxes.get("opt_live"), _live_fig(),
                        "section_live_metrics")

    def _render_optimise():     # noqa: PLR0915
        box = ctx.views[(stage, "optimise")]
        box.clear()
        # every handle below belongs to THIS build of the view: a stale one
        # would be written into a pane that is no longer on screen
        for key in [k for k in boxes if k.startswith("opt_")]:
            boxes.pop(key, None)
        oc = A["opt"]
        anchor = session.section_weights(S, SURFACE)
        with box:
            with widgets.group_box("What this does"):
                # the VECTOR is counted rather than quoted: in wing mode the
                # twist and chord laws join the 8 weights, so the fixed "8"
                # understated the box the budget below has to cover
                _dim = 8 + (int(oc["twist_order"]) + int(oc["chord_order"])
                            if session.wing_objective(S, SURFACE)
                            and not is_composite(oc.get("objective"))
                            else 0)
                widgets.hint(
                    f"{_dim} design variables — 8 CST weights"
                    + (f", {int(oc['twist_order'])} twist and "
                       f"{int(oc['chord_order'])} chord coefficients"
                       if _dim > 8 else "")
                    + " — searched with a REAL viscous XFOIL sweep per "
                    f"candidate: seconds to a minute each, so a budget of "
                    f"{int(session.effective_airfoil_search(S, SURFACE)['budget'])}"
                    f" is minutes to an hour. Twist and "
                    "chord moves reuse the cached polar (it depends only on "
                    "the weights), so they cost far less than a weight move "
                    "— but the budget is NOT scaled for them. The screening "
                    "pick seeds the design box, so the search starts from a "
                    "section that already works rather than a generic "
                    "anchor.")
                if anchor:
                    widgets.hint(f"seeded from: "
                                 f"{(session.section_of(S, SURFACE) or {}).get('name', '?')}")
                else:
                    widgets.hint("no seed chosen — the run would start from "
                                 "the problem's own NACA anchor", "warn")

            with ui.row().classes("w-full items-start gap-3 no-wrap"):
                with ui.column().classes("gap-3").style("flex:1 1 0;"
                                                        "min-width:0"):
                    with widgets.group_box("Objective"):
                        # WHICH SCALAR the search maximises. The PHYSICAL
                        # objective is still not a question — the medium and
                        # the surface decide whether that is a wing L/D or the
                        # section's own 2-D L/D, and the branches below state
                        # which. What IS a question is whether to optimise it
                        # at all or the SIX CRITERIA the weights above name:
                        # the screen ranks on those and the search then throws
                        # five of them away, which is a choice the user should
                        # be making rather than inheriting.
                        widgets.select_field(
                            "maximise", OBJECTIVE_CHOICES(
                                session.wing_objective(S, SURFACE)),
                            oc.get("objective", "cd"),
                            lambda e: _set_objective(str(e.value)))
                        if is_composite(oc.get("objective")):
                            # ...and what to do with a design whose stall
                            # march stopped before it stalled. The default
                            # REFUSES it outright, which is why a weighted
                            # criterion can silently leave the objective:
                            # censoring is measured at 29.1 % of all refusals
                            # on this population, and a censored SEED takes
                            # every criterion with it.
                            widgets.select_field(
                                "a censored stall",
                                {"refuse": "refuse the design (published)",
                                 "lower_bound": "score it at the bound"},
                                oc.get("censored") or "refuse",
                                lambda e: _set_censored(str(e.value)),
                                tip="a march that stops converging before it "
                                    "stalls gives a LOWER BOUND on cl_max, "
                                    "not a missing value — scoring at the "
                                    "bound is a true lower bound on J for "
                                    "any non-negative weights")
                            _render_composite_objective()
                        elif session.wing_objective(S, SURFACE):
                            g = session.wing_guess(S, SURFACE)
                            widgets.readout("scoring",
                                            "WING L/D (section + twist law)")
                            widgets.hint(
                                f"flown on the ESTIMATED wing: "
                                f"{g['s_ref_m2']:.2f} m² from the mission, AR "
                                f"{g['aspect_ratio']:.1f} from the estimate "
                                f"above, taper {g['taper']:.2f} — the design "
                                f"Cl and Re are derived from it, not typed "
                                f"in.")
                            # the objective IS a wing L/D, so which wing it
                            # is matters more here than anywhere else on
                            # this stage
                            if session.flown_size(S) and abs(
                                    session.flown_aspect_ratio(S)
                                    - session.section_aspect_ratio(S)) > 1e-9:
                                widgets.hint(
                                    f"Stage 3's span box implies AR "
                                    f"{session.flown_aspect_ratio(S):.3g}, not "
                                    f"this one — the section below would be "
                                    f"optimised for the L/D of a wing the run "
                                    f"does not fly.", "warn")
                                _adopt_button()
                            # STATED, not asked. A wing L/D needs a loading,
                            # so the estimate wing carries a twist and a
                            # chord law — but neither answer travels to
                            # stage 3 (only the SECTION does), and stage 3
                            # designs both itself: twist_root/twist_tip sit
                            # in every wing design vector and the chord law
                            # has its own control there. Two selects here
                            # asked the wing stage's question a stage early,
                            # about a wing that is only an estimate, and read
                            # as one answer given twice. The values live on
                            # in the airfoil defaults, which follow the chord
                            # law the shell opens the wing stage on, so the
                            # loading scored here stays the one stage 3 is
                            # most likely to fly.
                            widgets.hint(
                                f"Scored on a realistic loading — "
                                f"{api.airfoil_twist_orders()[int(oc['twist_order'])]}"
                                f" twist, "
                                f"{api.airfoil_chord_orders()[int(oc['chord_order'])]}"
                                f" chord — because a wing L/D has to have "
                                f"one. It reshapes the ESTIMATE wing only: "
                                f"stage 3 designs the twist and chord the run "
                                f"actually flies, and what travels from here "
                                f"is the section.")
                        else:
                            widgets.readout("scoring", "2-D L/D")
                            widgets.hint(f"L/D of the {_surface_name()}'s own "
                                         f"section at its design Cl.")
                            geo = session.surface_geometry(S, SURFACE)
                            # a trimming surface is 2-D for its OWN reason,
                            # which is not the medium's — say the right one
                            widgets.hint(
                                session.SECTION_ONLY_NOTE
                                if session.airfoil_only(S) else
                                session.TRIM_CL_NOTE
                                if geo is not None and not geo["lifting"]
                                else session.TWO_D_NOTE)
                with ui.column().classes("gap-3").style("flex:1 1 0;"
                                                        "min-width:0"):
                    with widgets.group_box("Search"):
                        eff = session.effective_airfoil_search(S, SURFACE)
                        if eff["source"] == "recommended":
                            _render_recommended_budget(eff)
                        else:
                            widgets.number_field(
                                "budget", oc["budget"],
                                lambda e: _set_opt("budget", int(e.value or 2)),
                                unit="evaluations", step=1)
                        widgets.number_field(
                            "seed", oc["seed"],
                            lambda e: _set_opt("seed", int(e.value or 0)),
                            step=1)
                        _opt_gate("min t/c", "tc_min", 0.01)
                        _opt_gate("max |Cm|", "cm_max", 0.01)
                        widgets.hint(
                            "Switched off, the corresponding design "
                            "constraint is slack: the shape optimiser is "
                            "then free to go as thin (or as nose-down) as "
                            "the objective likes, and the margin it reports "
                            "is always satisfied.")

            with ui.row().classes("w-full items-center gap-2"):
                run_btn = ui.button("Optimise the section", icon="auto_graph",
                                    on_click=lambda _=None: start_optimise()) \
                    .props("unelevated dense no-caps color=primary")
                # CONTINUE, PRICED. A budget that ran out says nothing about
                # whether it was enough, and until this button the only answer
                # to "it had not converged" was to search again from scratch
                # and throw away every XFOIL sweep already paid for.
                offer = _continue_offer()
                cont_btn = ui.button(
                    ("Continue" if not offer or offer.get("error")
                     else f"Continue — +{int(offer['extra'])} "
                          f"({int(offer['note']['budget'])} in total)"),
                    icon="play_arrow",
                    on_click=lambda _=None: continue_optimise()) \
                    .props("outline dense no-caps")
                if offer is None:
                    widgets.explain(
                        cont_btn,
                        "Runs the search on screen for longer — available "
                        "once one has finished or been stopped.",
                        title="Continue")
                elif offer.get("error"):
                    widgets.explain(cont_btn, offer["error"],
                                    title="Continue")
                else:
                    # the cache clause belongs ONLY to a continuation that
                    # re-flies what was paid for. On a run the api has judged
                    # NOT contained, saying it in the same breath contradicts
                    # the sentence before it.
                    widgets.explain(
                        cont_btn,
                        offer["note"]["why"]
                        + ("\n\nThe evaluations already paid for come back "
                           "out of the XFOIL cache, so a continuation costs "
                           "the new ones." if offer["note"]["exact"]
                           else "\n\nIt will still re-use every polar it "
                                "has seen before, but as a fresh search."),
                        title="Continue")
                if offer is None or offer.get("error"):
                    cont_btn.disable()
                stop_btn = ui.button("Stop", icon="stop", on_click=stop) \
                    .props("outline dense no-caps")
                widgets.explain(
                    stop_btn,
                    "Ends the search after the current evaluation and KEEPS "
                    "the best section it has reached — every XFOIL sweep "
                    "already paid for stays.", title="Stop")
                if oc["running"]:
                    run_btn.disable()
                else:
                    stop_btn.disable()
                ui.space()
                if oc["running"]:
                    boxes["opt_chip"] = widgets.tag(_progress_text(),
                                                    theme.ACCENT)
                elif oc.get("stopped"):
                    widgets.tag("STOPPED", theme.WARN)
            if oc["error"]:
                widgets.hint(oc["error"],
                             "warn" if "stopped" in oc["error"] else "bad")
            got = int(oc.get("resumed") or 0)
            if got and (oc["running"] or oc.get("report")):
                widgets.hint(
                    f"This is a CONTINUATION, and its counter starts at "
                    f"{got}: the {got} evaluations the search it lengthens "
                    f"paid for are this run's training set, so not one XFOIL "
                    f"polar is bought twice. What it spends is the "
                    f"{max(0, int(oc.get('run_budget') or 0) - got)} new "
                    f"sections after them.")
            replay = int(oc.get("replay") or 0)
            if replay and (oc["running"] or oc.get("report")):
                widgets.hint(
                    f"This is a CONTINUATION, so the counter starts at 1 "
                    f"again: the first {replay} evaluations are the ones the "
                    f"search it lengthens already flew, re-flown on the same "
                    f"seed — which is what makes this longer run contain the "
                    f"shorter one. Every polar they need is already in the "
                    f"XFOIL cache, so re-flying them costs a fraction of "
                    f"what they first did — measured cold at Re 8.5e5, an "
                    f"8-evaluation prefix came back in 6.3 s of the 21.7 s "
                    f"it originally cost, the rest being the surrogate "
                    f"refit. Most of this run is the "
                    f"{max(0, int(oc.get('run_budget') or 0) - replay)} new "
                    f"evaluations after it.")

            with widgets.group_box("Convergence", pad=False):
                boxes["opt_conv"] = figstyle.show(_conv_fig(),
                                                  "section_convergence")

            _live_panel()

            rep = oc["report"]
            if rep:
                _render_opt_result(rep)
        boxes["opt_sig"] = _opt_sig()

    def _live_labels() -> dict:
        """key -> the name the shortlist gives it (metrics.LIVE_METRICS)."""
        short, _ = metric_catalogue.live_metric_menu(live.keys())
        return dict(short)

    def _live_selected() -> list:
        """The keys plotted right now.

        Until the user ticks something this is the catalogue's own answer
        (``metrics.live_metric_defaults``) rather than a stored list, so a
        run whose family spells the objective differently still opens on the
        objective instead of on an empty plot.
        """
        have = live.keys()
        if not live_cfg["picked"]:
            return metric_catalogue.live_metric_defaults(have)
        return [k for k in live_cfg["keys"] if k in have]

    def _live_fig() -> go.Figure:
        keys = _live_selected()
        if not keys or len(live.samples) < 2:
            return figstyle.empty(
                "the incumbent section's named metrics appear here", 240)
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
        fig.update_layout(xaxis_title="evaluation",
                          yaxis_title=labels.get(keys[0], keys[0]),
                          height=260, legend=dict(orientation="h", y=1.14))
        if len(keys) > 1:
            fig.update_layout(yaxis2=dict(
                title=", ".join(labels.get(k, k) for k in keys[1:]),
                overlaying="y", side="right", showgrid=False))
        return fig

    def _live_panel():
        with widgets.group_box("Live metrics", pad=False):
            with ui.column().classes("group-pad w-full gap-2"):
                with ui.row().classes("items-center gap-3 no-wrap"):
                    ui.switch("sample the incumbent",
                              value=bool(live_cfg["on"]),
                              on_change=lambda e: _set_live(bool(e.value))) \
                        .props("dense")
                    boxes["opt_samples"] = widgets.tag(
                        f"{len(live.samples)} samples", theme.INK_MUTED)
                widgets.hint(
                    "The best section so far, re-evaluated once per "
                    "improvement. Its viscous sweep is already in the XFOIL "
                    "cache from the evaluation that produced it, so this "
                    "costs almost nothing — and it shows the numbers the "
                    "objective does not: c_l max, |c_m|, the drag split.")
                available = live.keys()
                # A FEW, THEN A PRESS. The breakdown reports 46 numeric keys
                # the moment the section is scored on a wing, and most of
                # them are the same number twice (L/D four times, every drag
                # in coefficients AND counts, three spellings of the
                # reference area) — a wall of checkboxes nobody reads. The
                # catalogue de-duplicates them, ranks them, and hands back
                # the handful this run should LEAD with plus everything else
                # (metrics.live_metric_panel); the rest is one click away
                # rather than gone.
                picked = _live_selected()
                primary, more = metric_catalogue.live_metric_panel(available,
                                                                   picked)
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
            boxes["opt_live"] = figstyle.show(_live_fig(),
                                              "section_live_metrics")

    def _set_live(on: bool):
        live_cfg["on"] = bool(on)
        if not on:
            live.stop()
        elif A["opt"]["running"]:
            live.start(cfg_fn=lambda: live_cfg["cfg"],
                       incumbent_fn=_opt_incumbent,
                       running_fn=lambda: bool(A["opt"]["running"]))
        _tick_optimise()

    def _toggle_metric(key: str, on: bool):
        # the first tick takes ownership: until then the plot shows the
        # catalogue's defaults, and toggling one of them off has to leave
        # the other two rather than resetting to them next render
        keys = list(_live_selected())
        if on and key not in keys:
            keys.append(key)
        if not on and key in keys:
            keys.remove(key)
        live_cfg["keys"] = keys
        live_cfg["picked"] = True
        # only the FIGURE. Rebuilding the view would scroll the page back to
        # the top on every click, and close the expansion the click came from
        figstyle.update(boxes.get("opt_live"), _live_fig(),
                        "section_live_metrics")

    def _set_objective(name: str):
        A["opt"]["objective"] = (str(name)
                                 if str(name) in OBJECTIVE_CHOICES(True)
                                 else "cd")
        _render_optimise()

    def _set_censored(mode: str):
        """How a CENSORED stall is scored. Composite runs only.

        ``api._airfoil_censored`` raises if a non-default mode reaches a -cd
        run, so the flag is stored here and only sent by
        :func:`objective_kwargs` on the composite branch — the state may hold
        an answer the current objective cannot use, but nothing may SEND one.
        """
        A["opt"]["censored"] = (str(mode)
                                if str(mode) in ("refuse", "lower_bound")
                                else "refuse")
        _render_optimise()

    def _render_composite_objective():
        """What the composite objective is, and the two prices it carries."""
        obj = str(A["opt"].get("objective") or "")
        goal = obj == "composite_goal"
        asf = obj == "composite_asf"
        widgets.readout("scoring",
                        "TCHEBYCHEFF OF SIX, FROM THE SEED" if asf
                        else "COMPOSITE SCORE J, SEED AS A FLOOR" if goal
                        else "COMPOSITE SCORE J")
        if asf:
            widgets.hint(
                "Not a weighted sum at all. Each criterion is measured as "
                "weight x (its sub-score minus the seed's), and the search is "
                "scored on the WORST of those, plus 0.05 of their total to "
                "break ties. So it can only improve by lifting whichever "
                "criterion is currently furthest behind the seed — there is no "
                "exchange rate to sell one at.")
            widgets.hint(
                "Why the floor above is not enough: a weighted sum, penalty or "
                "no penalty, can only ever return a design on the CONVEX HULL "
                "of what this box can reach. Every compromise section on a "
                "non-convex part of the trade-off is invisible to it at every "
                "weight you could type. This function can reach them.")
            widgets.hint(
                "Its number is NOT a composite J — it is roughly zero at the "
                "seed, positive for a section that beats it on every criterion "
                "the weights name. The score card below re-scores both "
                "sections on the plain composite, which is the number that "
                "compares across objectives. A criterion you weighted at zero "
                "is dropped, not held at zero: a zero-weight term would pin "
                "the worst-of at zero and switch the function off.", "warn")
        if goal:
            widgets.hint(
                "J minus what falling BELOW the seed costs: for each "
                "criterion, penalty x weight x (how far under the seed's own "
                "sub-score it ended), and nothing at all for a criterion that "
                "improved. A design that beats the seed everywhere scores "
                "exactly the composite; one that sells a criterion pays ten "
                "times the rate the composite would have paid it for.")
            widgets.hint(
                "Why it exists, measured on the frozen study: the plain "
                "composite's winner gains 11 points of J while cruise L/D "
                "falls 12 % (83.9 → 73.6) and section drag at the design lift "
                "RISES 14 %. Over the shipped band 0.01 of |Cm| buys four "
                "counts of L/D and the stall angle is weighted 0, so the sum "
                "sells both — arithmetic nobody asked for. Across ten runs "
                "(two anchors x five seeds) every winner regressed on one to "
                "three of the six criteria.", "warn")
            widgets.hint(
                "The floor is the SEED — the section this design box is "
                "centred on, scored at this same operating point. A criterion "
                "you weighted at zero is not defended, because it is not in J "
                "either: give it a weight to protect it.")
        widgets.hint(
            "The weighted composite of the six criteria above — the SAME "
            "number the ranking is built from, so what the screen chose the "
            "section for is what the search now improves. Every evaluation "
            "costs one extra XFOIL sweep (0-20 deg) because c_l max and the "
            "stall angle need it; it is fired only after the t/c and |Cm| "
            "gates pass, so an infeasible candidate never pays for it.")
        ref = _screen_reference()
        widgets.hint(
            "Scored against a FROZEN 0-100 band"
            + (" measured on this surface's own screen"
               if ref else " (the shipped library band)")
            + ": a band recomputed each evaluation would move under the "
              "optimiser and no GP could fit it. A section outside the band "
              "scores past the end rather than saturating.")
        if session.wing_objective(S, SURFACE):
            widgets.hint(
                "J scores a 2-D SECTION, so this run leaves wing mode: the "
                "twist and chord laws drop out of the design vector (8 "
                "variables, not 10) because a 2-D score cannot see them. The "
                "wing L/D objective above is the one that flies a loading.",
                "warn")
        live_floors = [k for k, v in (A["floors"] or {}).items()
                       if v is not None]
        if live_floors:
            widgets.hint(
                f"Your screening minimums ({', '.join(sorted(live_floors))}) "
                f"do NOT bind here: they are a filter on a catalogue, and the "
                f"search has only the two design constraints beside it (t/c "
                f"and |Cm|). Those criteria still COUNT — they are terms in "
                f"J — they are just not floors.", "warn")
        _render_exchange_rates()
        widgets.hint(
            "Measured, and worth knowing before trusting a big number: the "
            "composite is as exploitable as drag — a laminar bucket real "
            "only inside XFOIL's transition model scores J past 100 (report "
            "§16.3). The seed-vs-optimised table below is where that shows "
            "up.", "warn")

    def _render_exchange_rates():
        """WHAT THE WEIGHTS ACTUALLY PRICE, in the criteria's own units.

        A weight is not an importance: J moves by ``100 w / (hi - lo)`` per
        unit of a criterion, so the width of the band is half the decision and
        it is nowhere on this page. Under the shipped band and the GDP preset
        the largest weight (0.35, cruise L/D) buys less per natural step than
        |Cm| does at 0.20 — which is why the search sells it. This says so
        before the run rather than after.
        """
        from nicegui import ui as _ui

        try:
            rates = api.score_exchange_rates(dict(A["weights"]),
                                             _screen_reference())
        except (ValueError, OSError, KeyError):
            return                       # a hint, never a failure
        if not rates:
            return
        labels = dict(api.SCREEN_METRICS)
        unit = {"thick": "0.01 of t/c", "clmax": "0.1 of c_l max",
                "ldmax": "10 counts of (L/D)max",
                "ldcr": "10 counts of L/D at the design c_l",
                "astall": "1 deg of stall angle", "cm": "0.01 of |C_m|",
                "cdcr": "10 counts of c_d at the design c_l"}
        # a criterion with no unit line is named rather than dropped: a rate
        # table missing a row the weights panel shows is the same silence the
        # zero-weight rule exists to break
        # A CRITERION WITH NO BAND HAS NO RATE, and says so instead of being
        # dropped or printed as 0.00 J: on a surface that flies at zero lift
        # the screened population measures the same L/D for every section, so
        # the band omits it (api.score_exchange_rates -> "unbanded") and a
        # weight on it is refused before the run starts. Sorted last, because
        # None is not a small number.
        rows = [{"step": unit.get(k, f"one step of {labels.get(k, k)}"),
                 "weight": f"{v['weight']:.2f}",
                 "worth": (f"{v['per_step']:.2f} J"
                           if v.get("per_step") is not None
                           else "no band — this population cannot rank it")}
                for k, v in sorted(
                    rates.items(),
                    key=lambda kv: (kv[1].get("per_step") is None,
                                    -(kv[1].get("per_step") or 0.0)))
                if k in labels]
        with _ui.expansion("what a weight is actually worth") \
                .props("dense").classes("w-full"):
            _ui.table(columns=[
                {"name": "step", "label": "one step of…", "field": "step",
                 "align": "left"},
                {"name": "weight", "label": "weight", "field": "weight",
                 "align": "right"},
                {"name": "worth", "label": "moves J by", "field": "worth",
                 "align": "right"}],
                rows=rows, row_key="step") \
                .classes("w-full").props("dense flat bordered")
            widgets.hint(
                "J moves by 100 x weight / (band width) per unit, so the "
                "BAND is half of every trade and the weight alone cannot tell "
                "you what the search will do. Read this table as the exchange "
                "rate the optimiser is being offered.")

    def _render_unscored_criteria(work: dict):
        """Criteria the user WEIGHTED that the objective could not defend.

        The silent failure this exists for: both reference objectives build
        their per-criterion terms from the SEED's measured criteria, so a
        criterion the seed never produced is not merely unweighted — it is
        absent from the objective's iteration domain entirely. Set `ldcr` to
        0.35 (the largest weight in the shipped preset) against a seed whose
        polar could not bracket the design lift, and nothing in the search
        rewards improving it, while the weights panel goes on showing 0.35.

        Named, with the reason, and priced by the weight that stopped
        binding, because the number the user set is the number they will look
        for afterwards.

        A criterion the ASF put back at the BAND FLOOR
        (``airfoil_asf_missing = "band_floor"``) is a different sentence and
        gets one: its weight did bind, weakly and through the augmentation
        only, so reporting it beside the ones that bought nothing would be the
        same false accusation the zero-weight rule exists to avoid.
        """
        labels = dict(api.SCREEN_METRICS)
        weights = session.airfoil_state(S, SURFACE).get("weights") or {}

        def named(items):
            out = []
            for key, why in sorted(items.items()):
                w = weights.get(key)
                out.append(f"{labels.get(key, key)}"
                           + (f" (weight {float(w):.2f})" if w else "")
                           + f" — {why}")
            return "; ".join(out)

        dropped = work.get("dropped") or {}
        if dropped:
            widgets.hint(
                "These criteria carry a weight and were NOT in the objective: "
                + named(dropped)
                + ". The search neither rewarded nor defended them, so their "
                  "weights bought nothing. Either give the reference design a "
                  "value for them (the censored-stall setting on this card is "
                  "the usual cause and the usual cure) or set their weight to "
                  "zero, so the number on screen is the number that ran.",
                "warn")
        floored = work.get("floored") or {}
        if floored:
            widgets.hint(
                "These criteria carry a weight, the reference design has no "
                "value for them, and they were held to the BOTTOM OF THE "
                "FROZEN BAND rather than dropped: " + named(floored)
                + ". Their weights bind — raising one raises the objective — "
                  "but only through the augmentation term, so such a "
                  "criterion cannot become the one the max-min is held to. "
                  "State a reference value for it if it has to bind that "
                  "hard.")

    def _render_asf_working():
        """WHICH criterion the Tchebycheff was holding — the ASF's working.

        The one number that explains an ASF run is the BINDING criterion: the
        max-min improves only by lifting whichever criterion sits furthest
        behind the seed, so "which one was that, and by how much" is the whole
        story of where the budget went.
        """
        bd = ((A["opt"].get("report") or {}).get("design") or {}) \
            .get("breakdown") or {}
        work = bd.get("asf") or {}
        rows = work.get("rows") or {}
        # THE FALLBACK IS TESTED FIRST, because `fallback` is set precisely
        # WHEN `rows` is empty (`asf_terms` returns one or the other). Below
        # the `if not rows: return` guard the branch could never be reached,
        # so the one warning that says "your weights did not reach this run"
        # has never been drawn on screen.
        if work.get("fallback"):
            widgets.hint(f"the reference point was empty, so this run scored "
                         f"the plain composite: {work['fallback']}", "warn")
            _render_unscored_criteria(work)
            return
        if not rows:
            return
        _render_unscored_criteria(work)
        worst, m = work.get("worst"), work.get("min")
        labels = dict(api.SCREEN_METRICS)
        widgets.readout(
            "binding criterion",
            f"{labels.get(worst, worst)} {float(m):+.2f}" if m is not None
            else "—")
        ahead = sorted(((k, v["term"]) for k, v in rows.items()),
                       key=lambda kv: kv[1])
        widgets.hint(
            "Weight x (sub-score minus the seed's), per criterion: "
            + ", ".join(f"{labels.get(k, k)} {t:+.2f}" for k, t in ahead)
            + f". The search maximised the SMALLEST of these plus "
              f"{float(bd.get('asf_rho') or 0.0):g} of their sum"
            + (f" = {float(bd['composite_asf']):+.3f}."
               if bd.get("composite_asf") is not None else ".")
            + (" Every one is positive, so this section beats the seed on "
               "every criterion the weights name."
               if m is not None and float(m) > 0.0 else
               " The negative ones are criteria this section still ends below "
               "the seed on — the budget ran out before the worst of them "
               "could be lifted."))

    def _render_goal_working():
        """What the SEED FLOOR cost the winner — the goal objective's working.

        Read off the run's own design breakdown (``goal_evaluation`` puts it
        there), not recomputed: the penalty the search actually paid is the
        one that decided the winner, and a second measurement of it would be a
        second opinion nobody asked for.

        ONLY THE WEIGHTED CRITERIA ARE NAMED. ``goal_shortfalls`` keeps a row
        for every criterion in the reference point and prices it at
        ``penalty x weight x shortfall``, so a criterion weighted 0 carries a
        real shortfall and a cost of exactly zero. Listing it beside the ones
        that were paid for reads as "the winner sold this", which the
        objective never claimed: it is not in J, so it was never defended.
        The zero-cost rows are counted rather than dropped, so the sentence
        cannot quietly shorten.
        """
        bd = ((A["opt"].get("report") or {}).get("design") or {}) \
            .get("breakdown") or {}
        work = bd.get("goal") or {}
        rows = work.get("rows") or {}
        _render_unscored_criteria(work)
        if not rows:
            # ...and SAY it, rather than drawing nothing. An empty floor is
            # not "no penalty was paid": it is "there was no floor", which is
            # a different run from the one the objective card promised.
            src = str((bd.get("goals") or {}).get("source") or "")
            widgets.hint(
                "nothing was floored on this run: the reference design "
                + (f"could not be scored ({src})" if src else
                   "produced no criteria to floor against")
                + ", so the search maximised the plain composite J and the "
                  "seed floor defended nothing.", "warn")
            return
        pen = float(work.get("penalty") or 0.0)
        widgets.readout("seed floor",
                        "MET — nothing sold" if pen <= 0.0
                        else f"COST {pen:.2f} J")
        #: below the seed but weighted 0 — a fact, never a charge
        unpriced = sorted(k for k, v in rows.items()
                          if v.get("shortfall", 0.0) > 0.0
                          and float(v.get("weight", 0.0)) <= 0.0)
        if pen > 0.0:
            worst = work.get("worst")
            short = ", ".join(
                f"{k} -{v['shortfall']:.1f} pts ({v['cost']:.2f} J)"
                for k, v in sorted(rows.items(),
                                   key=lambda kv: -kv[1]["cost"])
                if v["shortfall"] > 0.0 and float(v.get("weight", 0.0)) > 0.0)
            widgets.hint(
                f"The winner still ends below the seed on {short}. The search "
                f"paid for it — J {bd.get('composite', float('nan')):.2f} "
                f"scored {bd.get('composite_goal', float('nan')):.2f} — and "
                f"chose it anyway, so nothing in the box did better while "
                f"holding {worst}.", "warn")
        else:
            widgets.hint(
                "Every criterion you gave a non-zero weight ended at or above "
                "the seed's own sub-score, so the penalty is zero and this "
                "run's J is the plain composite. That is the objective doing "
                "its job, not a term switching itself off.")
        if unpriced:
            widgets.hint(
                f"{', '.join(unpriced)} also ended below the seed, and cost "
                f"nothing: you weighted "
                + ("it" if len(unpriced) == 1 else "them")
                + " 0, so the floor does not defend "
                + ("it" if len(unpriced) == 1 else "them")
                + " — give it a weight to protect it.")

    def _render_score_block():
        """SEED vs OPTIMISED on the six criteria the WEIGHTS name.

        The search maximised ONE number. This says what that cost or bought
        on the other five — the same composite the ranking upstairs is built
        from, so "rank 1 of the library, then optimised" can be read as one
        sentence instead of two unrelated ones.
        """
        from nicegui import ui as _ui

        oc = A["opt"]
        if oc.get("scoring"):
            with _ui.row().classes("items-center gap-2"):
                _ui.spinner(size="sm")
                _ui.label("scoring both sections on your criteria — one wide "
                          "stall sweep each").classes("hint")
            return
        if oc.get("score_error"):
            widgets.hint(f"the sections could not be scored on your weights: "
                         f"{oc['score_error']}", "warn")
            return
        sco = oc.get("score")
        rows = score_rows(sco)
        if not rows:
            return
        seed = sco.get("seed") or {}
        opt = sco.get("optimised") or {}
        delta = sco.get("delta") or {}
        d = delta.get("composite")
        with _ui.row().classes("w-full items-start gap-4 no-wrap"):
            widgets.readout(
                "seed score", _score_cell(seed.get("composite"), 2),
                tip=(f"{seed.get('label') or 'the seed'} — the section the "
                     "design box was anchored on, flown at this same point"))
            widgets.readout("optimised score",
                            _score_cell(opt.get("composite"), 2))
            widgets.readout(
                "change", _score_cell(d, 2, signed=True),
                color=("" if d is None
                       else theme.GOOD if d >= 0.0 else theme.WARN))
        with _ui.expansion("the six criteria, one row each") \
                .props("dense").classes("w-full"):
            verdict_slots(_ui.table(columns=[
                {"name": "metric", "label": "criterion", "field": "metric",
                 "align": "left"},
                {"name": "original", "label": "seed", "field": "original",
                 "align": "right"},
                {"name": "new", "label": "optimised", "field": "new",
                 "align": "right"},
                {"name": "change", "label": "change", "field": "change",
                 "align": "right"}],
                rows=rows, row_key="metric")
                .classes("w-full").props("dense flat bordered"))
            widgets.hint(
                "▲ green is a criterion this section improved on, ▼ red one "
                "it gave up, = grey one that did not move. A criterion you "
                "weighted 0 is left grey whichever way it went: it is not in "
                "J, so the search was never defending it.")
        ref = sco.get("reference") or {}
        rep_obj = ((A["opt"].get("report") or {})
                   .get("conditions", {}).get("objective"))
        was_target = is_composite(rep_obj)
        if rep_obj == "composite_goal":
            _render_goal_working()
        elif rep_obj == "composite_asf":
            _render_asf_working()
        widgets.hint(
            ("This IS what the search maximised — the composite of your six "
             "criteria, restated here for the seed as well so the run's own "
             "improvement is visible. "
             if was_target else
             "These are the SCREENING criteria under this surface's weights — "
             "not what the search maximised (that is the objective above). ")
            + "Both sections are scored on one frozen 0-100 band"
            + (f" ({ref.get('source')}, {ref.get('n_records', 0)} sections)"
               if ref.get("sha") else "")
            + ", because two sections normalised against each other would "
              "score 100 and 0 whatever they are. A criterion can therefore "
              "score past either end — that is the band being honest about a "
              "section outside the library's spread, not an error. It will "
              "NOT equal the ranking's own score column, which is normalised "
              "across the screened population: same weights, same criteria, "
              "a scale that moves with the table.")
        if d is not None and d < 0.0:
            widgets.hint(
                ("The optimised section scores LOWER on the very number the "
                 "search was maximising. That is the budget, not the trade: "
                 "the box's best feasible point never beat the seed. Give it "
                 "more evaluations, or keep the library pick."
                 if was_target else
                 "The optimised section scores LOWER on your weights than the "
                 "section it started from. That is a real trade, not a "
                 "failure: the search was told to maximise one number and is "
                 "free to spend the other five criteria on it. Keep the "
                 "library pick if the weights are what you meant."), "warn")
        bad = [k for k, ok in (opt.get("gates") or {}).items() if ok is False]
        if bad:
            widgets.hint(
                f"the optimised section is outside the screening gate on "
                f"{', '.join('t/c' if k == 'tc' else '|Cm|' for k in bad)} — "
                f"scored anyway, and said here rather than hidden by "
                f"dropping the row", "warn")
        if opt.get("clmax_censored") or seed.get("clmax_censored"):
            widgets.hint(
                "c_l max is CENSORED for at least one of the two (XFOIL's "
                "march stopped while the lift was still climbing), so its "
                "criterion is a lower bound and the composite with it.",
                "warn")

    def _render_front_block(res: dict):
        """The FRONT, when the run produced one — nothing at all when it did
        not, so every scalar run's card is unchanged.

        The readouts above still show one design (the highest-plain-J point on
        the front), because a card has to lead with something. This block is
        what stops that being read as "the answer": every row here is a design
        no other row beats everywhere, so the spread between them is
        PREFERENCE and not optimality, and the deltas say what each one costs
        against the section the user already has.
        """
        from nicegui import ui as _ui

        front = res.get("front")
        rows = front_rows(front)
        if not rows:
            return
        keys = (front.get("conditions", {}).get("criteria")
                or ["ldcr", "clmax", "cm"])
        n_front = len(front.get("front") or [])
        widgets.hint(
            f"{n_front} designs, and none of them beats another on all "
            f"{len(keys)} searched criteria. The readouts above are the one "
            f"with the highest plain composite J — a choice made by a scalar, "
            f"which is exactly what a front exists to show you the cost of. "
            f"Rows are ranked by YOUR weights over these criteria; Δ is "
            f"against your seed, in sub-score points.")
        cols = [{"name": "rank", "label": "#", "field": "rank",
                 "align": "left"},
                {"name": "composite", "label": "plain J",
                 "field": "composite"}]
        for k in keys:
            cols.append({"name": k, "label": k, "field": k})
            cols.append({"name": f"d_{k}", "label": f"Δ {k}",
                         "field": f"d_{k}"})
        cols.append({"name": "raw", "label": "in raw units", "field": "raw",
                     "align": "left"})
        _ui.table(columns=cols, rows=rows, row_key="rank") \
            .props("dense flat bordered").classes("w-full")

    def _render_no_feasible(rep: dict):
        """The run came back with NOTHING — say which gate stopped it.

        A card that answers this with "best objective —" and two dead buttons
        is the worst page in the shell: the user paid an hour of XFOIL for a
        blank. Everything needed to explain it is in the run's own log, so it
        is read there (:mod:`gui.diagnose`) and then translated back into the
        two numbers on THIS tab — the t/c floor and the |Cm| ceiling — because
        a margin of ``-0.23`` is not a sentence anyone can act on and
        "the thinnest gate you can set and still keep this seed is 0.077"
        is.
        """
        from aerobo import api

        res = rep.get("result") or {}
        cond = rep.get("conditions") or {}
        spec = api.PROBLEM_SPECS.get(str(res.get("problem_name") or ""))
        clabels = tuple(getattr(spec, "constraint_labels", ()) or ())
        report = diagnose.infeasibility_report(res, constraint_labels=clabels)
        with widgets.group_box("No feasible section"):
            if report is None:
                widgets.hint(
                    "The search returned no section and this run kept no "
                    "per-evaluation margins, so it cannot say which gate "
                    "stopped it. Switch a gate off below and run it again.",
                    "bad")
                return
            for text, level in diagnose.infeasibility_lines(report):
                widgets.hint(text, level)
            # …and the SAME margins in the units of the two switches above.
            # norm_margin divides each by its own limit, so the best point's
            # margin inverts exactly: t/c = tc_min (1 + g0), |Cm| = cm_max
            # (1 - g1). That inversion is what turns the diagnosis into an
            # edit the user can make on this tab.
            reach = []
            for c in report["constraints"]:
                if c["best"] is None:
                    continue
                if c["index"] == 0 and cond.get("tc_min"):
                    reach.append(
                        f"the thickest section the box reached is t/c "
                        f"{float(cond['tc_min']) * (1.0 + c['best']):.4f}, "
                        f"and your floor asks {float(cond['tc_min']):.4f}")
                elif c["index"] == 1 and cond.get("cm_max"):
                    reach.append(
                        f"the flattest-moment section the box reached is "
                        f"|Cm| "
                        f"{float(cond['cm_max']) * (1.0 - c['best']):.4f}, "
                        f"and your ceiling asks {float(cond['cm_max']):.4f}")
            if reach:
                widgets.hint("In the units of the two gates on this tab: "
                             + "; ".join(reach) + ".", "warn")
            widgets.hint(
                "THE DESIGN BOX HERE IS THE SEED'S. The 8 CST weights are "
                "searched over the anchor's own published half-widths, so it "
                "is not a box you widen row by row — you move it by seeding "
                "the run from a different section. The two levers on this "
                "tab are the gates themselves: switch off or relax whichever "
                "one is named above, or go back to the ranking and adopt a "
                "section that already clears it.")
            with ui.row().classes("items-center gap-2"):
                ui.button("Keep the library pick", icon="undo",
                          on_click=lambda: _adopt_row(
                              int(A["screen"]["selected"] or 0))) \
                    .props("outline dense no-caps")

    def _render_opt_result(rep: dict):
        from nicegui import ui as _ui

        res = rep.get("result") or {}
        cond = rep.get("conditions") or {}
        best = res.get("best_score")
        wing_mode = bool(cond.get("wing_mode"))
        composite = is_composite(cond.get("objective"))
        # NO INCUMBENT, NO CARD. A front run has no ``best_x`` of the usual
        # kind but does carry a front, so it is not this case.
        if res.get("best_x") is None and not res.get("front"):
            _render_no_feasible(rep)
            return
        with widgets.group_box("Optimised section"):
            if res.get("partial"):
                # WHAT THIS NUMBER IS NOT: a search that ran to its budget.
                # It is the best of the evaluations that were paid for, which
                # is a real design and a real score — and not comparable with
                # a completed run, so the card says so where the score is
                # read rather than in the log the user has scrolled past.
                widgets.hint(
                    f"PARTIAL — the search ended early ("
                    f"{res.get('stop_reason') or 'a stop rule fired'}) after "
                    f"{res.get('n_evals', '?')} of "
                    f"{res.get('budget', '?')} evaluations. Everything below "
                    f"is the best section it reached, flown and scored like "
                    f"any other; it is simply not budget-comparable with a "
                    f"run that finished. Continue re-flies THIS search with "
                    f"the rest of its budget — the evaluations already paid "
                    f"for come back out of the XFOIL cache.", "warn")
            with _ui.row().classes("w-full items-start gap-4 no-wrap"):
                if composite:
                    widgets.readout(
                        "best objective", _score_cell(best, 2), "score",
                        tip="the weighted composite of your six criteria, on "
                            "the frozen 0-100 band")
                elif wing_mode or best is None or not math.isfinite(best):
                    widgets.readout("best objective", fmt(best, 5))
                else:
                    # THE 2-D OBJECTIVE IS -c_d, and printing it raw put
                    # "-0.00546" under a panel that calls itself 2-D L/D.
                    # Same search, same optimum (c_l is fixed, so -c_d and
                    # c_l/c_d rank designs identically) — stated in the units
                    # the section is actually discussed in.
                    widgets.readout("best objective",
                                    f"{-float(best) * 1e4:.1f}",
                                    "counts c_d")
                    cl_d = float(cond.get("cl_design", 0.0))
                    if float(best) < 0.0 and cl_d:
                        widgets.readout("2-D L/D at design c_l",
                                        f"{abs(cl_d) / -float(best):.1f}")
                widgets.readout("evaluations", str(res.get("n_evals", "—")))
                widgets.readout("wall", f"{rep.get('wall_time_s', 0.0):.0f}",
                                "s")
                widgets.readout("mode", "composite J" if composite
                                else "wing L/D" if wing_mode else "2-D L/D")
            _render_front_block(res)
            _render_score_block()
            # the scoring block's six criteria travel WITH the comparison, so
            # a -cd or wing run's table reports the same metrics a composite
            # run's does instead of the four its own breakdown happens to hold
            rows = v1.airfoil_compare_rows(rep, A["opt"].get("score"))
            if rows:
                # the FIELD names are the ones ``airfoil_compare_rows``
                # actually returns (metric / original / new / change). They
                # used to be base / opt / delta, which no row carries, so
                # Quasar found nothing to put in three of the four columns
                # and drew a table of metric names with empty seed,
                # optimised and change cells.
                verdict_slots(_ui.table(columns=[
                    {"name": "metric", "label": "metric", "field": "metric",
                     "align": "left"},
                    {"name": "original", "label": "seed",
                     "field": "original", "align": "right"},
                    {"name": "new", "label": "optimised", "field": "new",
                     "align": "right"},
                    {"name": "change", "label": "change", "field": "change",
                     "align": "right"}],
                    rows=rows, row_key="metric")
                    .classes("w-full").props("dense flat bordered"))
                widgets.hint(
                    "▲ green improved, ▼ red got worse, = grey unchanged; a "
                    "row in plain grey with no glyph is a quantity with no "
                    "better direction (a twist angle, a chord) or one that "
                    "was measured on only one of the two sections.")
                widgets.hint(
                    "SEED is the section this run started from — the library "
                    "winner's CST refit"
                    + (", flown UNTWISTED at the same derived operating point"
                       if wing_mode else
                       ", flown at the SAME Reynolds number and design lift")
                    + " — so every change is the optimisation and not a "
                      "change of flight condition. It is evaluated for this "
                      "comparison (one XFOIL sweep), never read off the "
                      "screening table, which was measured at the library's "
                      "cached point.")
            with _ui.row().classes("items-center gap-2"):
                _ui.button("Adopt this section", icon="check",
                           on_click=lambda: _adopt_optimised()) \
                    .props("unelevated dense no-caps color=primary")
                _ui.button("Keep the library pick", icon="undo",
                           on_click=lambda: _adopt_row(
                               int(A["screen"]["selected"] or 0))) \
                    .props("outline dense no-caps")
        srep = rep.get("section")
        if srep:
            shp = v1.fig_section_shape(srep)
            pol = v1.fig_section_polars(srep)
            if shp is not None:
                with widgets.group_box("Optimised shape (seed overlaid)",
                                       pad=False):
                    figstyle.show(shp, "optimised_section")
            if pol is not None:
                with widgets.group_box("Polar — optimised vs seed",
                                       pad=False):
                    figstyle.show(pol, "optimised_polar")
                    if (srep.get("baseline") or {}).get("polar"):
                        widgets.hint(
                            "Both sweeps are the SAME viscous XFOIL run at "
                            "this surface's Reynolds number and Mach — the "
                            "dotted curve is the seed, the solid one what the "
                            "search returned, so the three panels read as one "
                            "before-and-after rather than as one design's "
                            "polar.")

    def _opt_gate(label: str, key: str, step: float):
        """The same on/off gate for the SHAPE optimiser's two design
        constraints (airfoil.AirfoilProblem's g_tc and g_cm)."""
        oc = A["opt"]
        on = oc.get(key) is not None
        with ui.row().classes("w-full items-center gap-2 no-wrap"):
            ui.switch(value=on,
                      on_change=lambda e, k=key:
                      _toggle_opt_gate(k, bool(e.value))).props("dense")
            ui.label(label).classes("field-label").style("min-width:96px")
            if on:
                ui.number(value=widgets.shown(oc[key]), step=step,
                          on_change=lambda e, k=key:
                          _set_opt_gate(k, e.value)) \
                    .props("outlined dense hide-bottom-space") \
                    .classes("w-24 shrink-0")
            else:
                ui.label("no limit").classes("field-unit")

    def _toggle_opt_gate(key: str, on: bool):
        A["opt"][key] = float(OPT_GATE_DEFAULTS[key]) if on else None
        _render_optimise()
        ctx.refresh()

    def _set_opt_gate(key: str, value):
        if value in (None, ""):
            return
        if widgets.is_echo(value, A["opt"][key]):
            return
        A["opt"][key] = float(value)
        # a typed gate: its own view may not be rebuilt from its handler
        ctx.refresh((stage, "optimise"))

    def opt_gate_value(key: str) -> float:
        v = A["opt"].get(key)
        return float(GATE_OFF[key] if v is None else v)

    def _set_opt(key: str, value):
        A["opt"][key] = value
        if key in ("chord_order", "twist_order"):
            _render_optimise()
        # the two keys that change WHICH controls exist have just redrawn the
        # view; the rest are typed or selected INTO it, so it is not repainted
        # from underneath them
        ctx.refresh((stage, "optimise"))

    # ------------------------------------------------------------ wiring
    ctx.on_render(stage, "screen", _render_screen)
    ctx.on_render(stage, "ranking", _render_ranking)
    ctx.on_render(stage, "section", _render_section)
    ctx.on_render(stage, "optimise", _render_optimise)
    # the screen form states the POINT this surface is read at — the MAC, the
    # Reynolds number, the design Cl, and the interval the wing search's own
    # taper band spans. Every one of those is stage 3's to move, so the view
    # is declared derived and repainted from whatever moved it.
    ctx.on_derived(stage, "screen")
    # the shape-optimiser form states what the search will fly — whether it
    # leaves WING MODE, and how many variables that is — which follows the
    # medium and the surface's job, both of them stage 1's and stage 3's to
    # change. That single predicate is the whole of what this view takes
    # from elsewhere, and it is 0.55 ms against a 39 ms repaint, so it is
    # given as the signature: the form is rebuilt only when the answer moves.
    ctx.on_derived(stage, "optimise",
                   sig=lambda: session.wing_objective(S, SURFACE))
    # ONE ACTION NAMESPACE PER SURFACE. This read
    # ``"" if SURFACE == "main" else "_aft"``, which was right while there
    # were two surfaces and became a COLLISION the moment there were three:
    # the vertical stabiliser registered its Run and Stop under the aft
    # surface's names, so whichever stage was built last owned them and one
    # surface's button drove the other's search.
    suffix = "" if SURFACE == "main" else f"_{SURFACE}"
    ctx.register(f"run_airfoil{suffix}", run_stage)
    ctx.register(f"stop_airfoil{suffix}", stop)
    # the pipeline's central act — "carry this row forward as the section" —
    # registered so it can be driven through the REAL handler rather than by
    # writing the session's section dict by hand, which is how the fields it
    # records (the POINT the row was screened at, above all) drift unnoticed
    ctx.register(f"adopt_section_row{suffix}", _adopt_row)
    # the third hard gate. Registered because it is the one gate whose state
    # is NOT visible in the report's conditions block (it refuses on the sign
    # of cm before ranking), so "what did the screen actually run with" can
    # only be asserted by driving the control the user presses.
    ctx.register(f"set_nose_down_gate{suffix}", _set_nose_down)
    # the section leaving the shell as FILES. Registered because an
    # aerofoil-only session's whole output is these three files, and a test
    # that writes them by hand would not be testing the button.
    ctx.register(f"save_section_export{suffix}", _save_section_export)
    if not SECONDARY:
        # stage 3 sends the user here to choose the SECOND surface's
        # section. It is a STAGE now, so "choose for that surface" is
        # "show that surface's stage" — there is no target to leave behind.
        ctx.register("choose_section_for", lambda surface="aft":
                     ctx.select(session.SURFACE_STAGES.get(surface, "airfoil"),
                                "screen"))
