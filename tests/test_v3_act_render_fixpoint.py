"""act → render must be a FIXPOINT — the one invariant that closes the
whole "card goes stale" class.

Every stale-card defect this shell has shipped has the same shape: a setter
moves the session, then hand-picks two or three of the ~30 ``_render_*``
helpers to re-run and misses one. Nobody can find those by reading, because
the read-out that goes stale is usually drawn by a DIFFERENT stage from the
control that moved it. So this file asserts the property instead of the
call list:

    apply an action · snapshot every view's text · ``ctx.render`` everything
    · snapshot again · the two snapshots are equal.

A difference means a control changed state that some view displays and
nobody repainted it. The user sees exactly that difference on screen, for
as long as it takes some unrelated edit to rebuild the view by accident.

What it caught when it was first run, on the shipped tree (measured, air
session, before the fixes below it):

* ``set_row_fixed("taper", True)`` — stage 1's Search tab and stage 3's
  Solver card went on quoting "bo (ucb), 29 evaluations" / "6 design
  variables" for a run that flies 26 in 5-D, and stage 2's screen went on
  quoting the un-pinned Reynolds interval "1e+06 – 1.15e+06" instead of the
  pinned 1.02e+06 – 1.02e+06;
* ``set_bound("taper", 1, 0.9)`` — the chord-law panel in the SAME view
  drew λ = 0.6 over a 64 % flyable band against a post-render λ = 0.55 /
  74 %, and the row's provenance chip still said "default" for a band the
  user had just typed;
* ``set_wing_objective("composite")`` — stage 1's estimate was left whole
  stages behind;
* ``set_span(11.0)`` — the Derived solver card beside the field still read
  "planform flown: b = 10.000 m";
* ``set_fixed_value("taper", 0.5)`` — "Fixed: taper = 0.6" over a field
  reading 0.5;
* ``set_choice("medium", "water")`` from stage 3 — stage 1's derived-point
  table still quoted air's 1.225 kg/m³ and 130.56 Pa.

The cure is structural rather than per-setter: a view whose text follows the
DESIGN VECTOR declares itself with ``ctx.on_derived(stage, view)``, and
``Ctx.refresh`` — which every setter already ends in — repaints all of them.
A handler driven by a typed ``ui.number`` passes its own view to
``refresh`` so the field it is being typed into is not destroyed mid-number,
and redraws the derived containers INSIDE that view by name, which is what
this shell has always done and what the two in-place helpers
(``_retag_source``, ``_render_fixed_note``) were added for.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

#: every stage the shell mounts, so "render everything" really is everything
STAGES = ("mission", "airfoil", "airfoil_aft", "wing", "results")

#: (action, args), applied IN ORDER to one shell. The order matters: each
#: case runs against the state the previous ones left, which is what a
#: session looks like, and several of them exist to put the session into the
#: state the next one needs (a second surface before its tail limits, a
#: screened row before it can be adopted). Every entry is a control a user
#: can actually press.
CASES: tuple = (
    # ---- the design box: the three states of a row, and a typed band
    ("set_row_fixed", ("taper", True)),
    ("set_fixed_value", ("taper", 0.5)),
    ("set_bound", ("taper", 1, 0.8)),          # a typed band on a PINNED row
    ("set_row_fixed", ("taper", False)),
    ("set_bound", ("taper", 1, 0.9)),
    ("set_bound", ("twist_root_deg", 0, -3.0)),
    ("set_row_on", ("taper", False)),
    ("set_row_on", ("taper", True)),
    ("reset_box", ()),
    ("take_recommendation", ("wing",)),
    # ---- what is maximised, and on what weights
    ("set_wing_objective", ("composite",)),
    ("set_wing_weight", ("lod", 0.4)),
    ("set_wing_objective", ("lod",)),
    # ---- stage 1's search policy
    ("set_search_mode", ("own",)),
    ("set_search_effort", ("thorough",)),
    ("set_search_stop", (True,)),
    ("adopt_search_values", ()),
    ("set_search_mode", ("recommended",)),
    # ---- the planform and its size
    ("set_span_searched", (False,)),
    ("set_span", (11.0,)),
    ("set_span_searched", (True,)),
    ("set_planform", ("wing loading",)),
    ("set_planform", ("fixed span",)),
    ("set_size_mode", ("chords",)),
    ("set_size_chord", ("root", 1.3)),
    ("set_size_mode", ("span",)),
    # ---- the chord law, its trend and its limits
    ("set_chord_law", ("ends",)),
    ("set_chord_trend", ("root_largest",)),
    ("set_chord_dev", (("", ""), 0.3)),
    ("set_chord_limit_on", ("chord_min_m", True)),
    ("set_chord_limit", ("chord_min_m", 0.45)),
    ("set_chord_limit_on", ("chord_min_m", False)),
    # ---- the wing's aspect-ratio limit, which clips the SIZE box as well as
    # refusing candidates, so several read-outs follow it
    ("set_ar_limit_on", ("ar_max", True)),
    ("set_ar_limit", ("ar_max", 9.0)),
    ("set_ar_limit_on", ("ar_max", False)),
    # ---- the tip device
    ("set_winglet", ("blended",)),
    ("set_tip_chord", (True,)),
    ("set_tip_chord", (False,)),
    # ---- the body and the wing's own cant: the two configuration switches
    # (both write a flag and one of them can change the FAMILY, which is the
    # case that has to leave every view redrawn)
    ("set_fuselage", (False,)),
    ("set_fuselage", (True,)),
    ("set_wing_cant", (False,)),
    ("take_spiral_dihedral", ()),
    ("set_wing_cant", (True,)),
    ("set_wing_cant", (False,)),
    ("set_choice", ("wing_cant", "free")),
    # ...and the two HALF states, which are the configurations where the card
    # draws a searched band AND a typed field at once — the shape most likely
    # to render differently after an act than after a rebuild
    ("set_choice", ("wing_cant", "dihedral")),
    ("set_choice", ("wing_cant", "sweep")),
    ("set_wing_cant", (False,)),
    # ---- the second surface, its placement and its own limits
    ("set_second_surface", (True,)),
    ("set_tail_arm", ("fixed",)),
    ("set_tail_arm", ("free",)),
    ("set_trim_number", ("tail_cg_m", 0.4)),
    ("set_tail_tip", ("none",)),
    ("set_tail_limit_on", ("tail_span_min_m", True)),
    ("set_tail_limit", ("tail_span_min_m", 0.6)),
    ("set_tail_limit_on", ("tail_span_min_m", False)),
    # ---- stage 2, on both surfaces
    ("set_nose_down_gate", ("off",)),
    ("set_nose_down_gate_aft", ("off",)),
    ("fix_section_point", ("aft",)),
    ("choose_section_for", ("aft",)),
    ("adopt_section_row", (0,)),
    ("adopt_section_row_aft", (0,)),
    ("clear_section", ()),
    ("set_second_surface", (False,)),
    # ---- the mission, and the constraint diagram's answer
    ("set_ws_input", ("v_stall_ms", 12.0)),
    ("adopt_ws", ()),
    ("accept_mission", ()),
    # ...and the way back to the family's own point. It rewrites the whole
    # operating card, so every read-out derived from it has to follow in one
    # render — which is exactly what this table is for.
    ("reset_mission", ()),
    # ---- the three questions that change the PROBLEM
    ("set_lifting_system", ("tandem",)),
    # ---- the PAIR's two areas, which only exist while a pair is selected:
    # stating them writes the split (and, where the family searches it, the
    # total), so the box, the derived geometry and the solver card all follow
    ("set_pair_area_mode", ("wings",)),
    ("set_pair_area_exact", ("front", True)),
    ("set_pair_area_value", ("front", 12.0)),
    ("set_pair_area_exact", ("rear", True)),
    ("set_pair_area_value", ("rear", 8.0)),
    ("adopt_pair_area_total", ()),
    ("set_pair_area_exact", ("front", False)),
    ("set_pair_area_band", ("front", "hi", 13.0)),
    ("set_pair_area_mode", ("total",)),
    ("set_lifting_system", ("single wing",)),
    ("set_medium", ("water",)),
    ("set_medium", ("air",)),
    ("set_choice", ("medium", "track")),
    ("set_planform", ("fixed span",)),
    # the car's OPERATING POINT, which only exists on this medium: a rear
    # wing has no weight and no altitude to back a lift coefficient out of,
    # so the coefficient is typed directly and stage 2 screens at it
    ("set_track_cz", (1.2,)),
    ("set_choice", ("medium", "air")),
    # ---- stage 4's export fields, and the read-only actions, which must
    # not move anything at all
    ("set_export_dir", ("runs",)),
    ("set_export_stem", ("probe",)),
    ("auto_ready", ()),
    ("refresh_results", ()),
    ("snippet", ()),
    # ---- the AIRFOIL-ONLY mode, driven last and switched back: stage 1
    # asks for a flow instead of a mission, and two of the five stages stop
    # existing. Every case above is about a vehicle, so this pair brackets
    # them rather than sitting among them.
    ("set_mode", ("airfoil",)),
    ("set_fluid", ("custom",)),
    ("set_flow", ("rho", 1.1)),
    ("set_flow", ("mach", 0.2)),
    ("set_flow_mach", (True,)),
    ("set_fluid", ("air",)),
    ("set_flow", ("chord_m", 0.4)),
    ("accept_mission", ()),
    ("set_mode", ("pipeline",)),
)

#: actions this test does NOT drive, and why. An honest exclusion list is
#: better than a table that passes by testing nothing — and the completeness
#: test below makes it impossible to add a setter without landing in one
#: list or the other.
EXCLUDED: dict = {
    # they START WORK rather than edit state: a real optimiser run, a live
    # XFOIL sweep, a background box measurement or a reach. Their result
    # arrives on the heartbeat, so a snapshot taken straight after the call
    # is a race, not a fixpoint.
    "launch": "queues a real run",
    "continue_run": "queues a real run",
    "run_airfoil": "starts a live XFOIL screen/optimise",
    "run_airfoil_aft": "starts a live XFOIL screen/optimise",
    "run_airfoil_fin": "starts a live XFOIL screen/optimise",
    "auto_recommend": "measures the box in a worker thread",
    "recommend_box": "measures the box in a worker thread",
    "measure_band": "measures the composite's band in a worker thread",
    "relax_reach": "runs the reach in a worker thread",
    # they need something in flight, or something finished, to do anything
    # at all — driven here they are no-ops, which would be a green case that
    # asserts nothing
    "cancel": "no-op unless a run is in flight",
    "stop_airfoil": "no-op unless a sweep is in flight",
    "stop_airfoil_aft": "no-op unless a sweep is in flight",
    "stop_airfoil_fin": "no-op unless a sweep is in flight",
    "relax_stop": "no-op unless a reach is in flight",
    "relax_adopt": "needs a finished reach",
    "set_result": "needs a finished run record",
    # they write to the filesystem or launch another program
    "save_cad": "writes files",
    "save_cad_bundle": "writes files",
    "save_section_export": "writes files",
    "save_section_export_aft": "writes files",
    "save_section_export_fin": "writes files",
    "open_in_vsp": "launches OpenVSP",
}


def _texts(view) -> list:
    return [getattr(e, "text", "") or "" for e in view.descendants()]


def _snapshot(ctx) -> dict:
    return {key: _texts(view) for key, view in ctx.views.items()}


def _render_all(ctx) -> None:
    for stage in STAGES:
        ctx.render(stage)


@pytest.fixture(scope="module")
def shell():
    """One assembled shell for the whole file.

    ``assemble()`` costs ~3.5 s and the table is 60-odd cases, so the shell
    is built once and the cases are driven against the state each other
    leaves. Nothing here depends on a pristine session: the invariant is
    "whatever this action did, a render changes nothing more", which is true
    of every state a session can be in.
    """
    from gui.v3.app import assemble

    ctx = assemble("air")
    _render_all(ctx)
    return ctx


# ------------------------------------------------------- 1. the invariant

def test_act_then_render_is_a_fixpoint(shell):
    """The whole class, in one assertion, over every shipped setter."""
    ctx = shell
    stale: list = []
    for name, args in CASES:
        assert name in ctx.actions, f"{name} is not a registered action"
        ctx.act(name, *args)
        # A view that is OFF SCREEN is owed its paint rather than given one
        # (Ctx.render_when_shown — an invisible rebuild still costs a
        # megabyte on the wire, and the native window chokes on it). The
        # shell pays that debt in `select`, before the view can be looked
        # at; here nothing is ever selected, so it is paid explicitly. The
        # invariant is unchanged: after this line, a difference below still
        # means a setter moved state that some view shows and NOBODY —
        # neither a render nor an owed paint — accounted for it.
        ctx.pay_owed()
        before = _snapshot(ctx)
        _render_all(ctx)
        after = _snapshot(ctx)
        for key in sorted(before):
            if before[key] == after.get(key):
                continue
            diff = [(b, a) for b, a in
                    zip(before[key], after[key] + [""] * len(before[key]))
                    if b != a][:3]
            stale.append(f"{name}{args} left {key} stale: {diff}")
    assert not stale, "\n".join(stale)


def test_no_view_rendered_a_traceback(shell):
    """Sixty state changes, and not one of the ~19 views fell over.

    ``Ctx.render`` contains a raising renderer by replacing the view with a
    traceback panel, so a broken card is silent unless something looks —
    and the fixpoint check above would happily call two identical tracebacks
    a fixpoint.
    """
    bad = [key for key, view in shell.views.items()
           if any("failed to render" in t for t in _texts(view))]
    assert not bad, bad


# --------------------------------------------- 2. the table is complete

def test_every_registered_action_is_either_driven_or_excluded(shell):
    """A setter added tomorrow lands in the table or in the exclusion list.

    Seeded from ``ctx.actions`` after assembly, which is the authoritative
    list of what the shell can be asked to do — so this cannot rot into a
    table that covers the setters that happened to exist when it was
    written.
    """
    driven = {name for name, _ in CASES}
    undecided = sorted(set(shell.actions) - driven - set(EXCLUDED))
    assert not undecided, (
        f"registered actions that are neither driven by the fixpoint table "
        f"nor excluded with a reason: {undecided}")
    # ...and the exclusion list may not name an action that no longer exists
    assert not sorted(set(EXCLUDED) - set(shell.actions))


# ------------------------- 3. the three failures the audit named, by name

def test_pinning_a_row_moves_stage_1_and_the_solver_card():
    """``set_row_fixed`` takes a variable out of the design vector, so the
    budget and the dimension every read-out quotes move with it — including
    the two that stage 3 does not own."""
    from gui.v3 import session
    from gui.v3.app import assemble

    ctx = assemble("air")
    _render_all(ctx)
    was = session.effective_wing_search(ctx.S)
    n_before, dim_before = int(was["budget"]), int(was["plan"].dim)

    ctx.act("set_row_fixed", "taper", True)
    ctx.pay_owed()          # neither of these views is the one on screen

    now = session.effective_wing_search(ctx.S)
    # the pin really did take a dimension (and with it a budget) away: this
    # test would pass vacuously against a plan that never moved
    assert int(now["plan"].dim) == dim_before - 1
    assert int(now["budget"]) != n_before

    for view in (("mission", "search"), ("wing", "solver")):
        texts = _texts(ctx.views[view])
        assert any(f"{int(now['budget'])} evaluations" in t for t in texts), \
            f"{view} does not quote the budget this run will fly"
        assert not any(f"{n_before} evaluations" in t for t in texts), \
            f"{view} still quotes the un-pinned budget"
        assert any(f"{int(now['plan'].dim)} design variables" in t
                   for t in texts), view

    # ...and stage 2's screen, which reads the catalogue over the interval
    # the taper band spans — one number wide once the taper is pinned
    line = next(t for t in _texts(ctx.views[("airfoil", "screen")])
                if "Across the taper" in t)
    lo, hi = line.split("that spans ")[1].split(". ")[0].split(" – ")
    assert lo == hi, line


def test_typing_a_taper_bound_moves_the_chord_law_panel_and_the_chip():
    """``set_bound`` may not rebuild the view its own ``ui.number`` is in, so
    the two things in that view which quote the row — the chord law's λ and
    the row's provenance chip — are written in place instead."""
    from gui.v3 import config
    from gui.v3.app import assemble

    ctx = assemble("air")
    _render_all(ctx)
    box = ctx.views[("wing", "box")]
    assert "λ = 0.6" in _texts(box)            # mid of the published 0.2–1
    assert _texts(box).count("user") == 0

    ctx.act("set_bound", "taper", 1, 0.9)

    (lo, hi), source = config.effective_bounds(ctx.S)["taper"]
    assert [lo, hi] == [0.2, 0.9] and source == "user"
    texts = _texts(box)
    # λ is the middle of the band the run now searches, drawn WITHOUT a
    # repaint of the view; before the fix the panel kept 0.6
    assert f"λ = {0.5 * (lo + hi):.3g}" in texts, \
        [t for t in texts if t.startswith("λ")]
    assert texts.count("user") == 1, "the row is still tagged as it was"


def test_choosing_the_objective_moves_stage_1s_estimate():
    """What stage 3 maximises is part of what stage 1 estimates, and the
    radio that chooses it lives on stage 3."""
    from gui.v3.app import assemble

    ctx = assemble("air")
    _render_all(ctx)
    before = _texts(ctx.views[("mission", "search")])

    ctx.act("set_wing_objective", "composite")
    ctx.pay_owed()          # stage 1 is not the stage on screen

    after = _texts(ctx.views[("mission", "search")])
    assert after != before, "stage 1's estimate did not follow the objective"
    ctx.render("mission", "search")
    assert _texts(ctx.views[("mission", "search")]) == after
