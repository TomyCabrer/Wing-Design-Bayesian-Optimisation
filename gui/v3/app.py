"""V3 shell assembly: the CAE desktop, and the stage mounting.

Layout (fixed to the viewport, nothing scrolls the page itself):

    menu bar
    tool bar                             run · stop · stage breadcrumb
    ┌ tree ────────────┬ tab strip ──────────────────────────────────┐
    │ simulation tree  │ work area (one container per stage view)    │
    ├ properties ──────┤                                             │
    │ facts about the  ├─────────────────────────────────────────────┤
    │ selected node    │ output log                                  │
    └──────────────────┴─────────────────────────────────────────────┘
    status bar                            problem · dim · budget · state

``assemble()`` builds the whole thing WITHOUT starting a server (the smoke
test calls it directly); ``run()`` assembles then serves. A stage that
fails to import or build gets a loud panel in its own work area instead of
taking the shell down.
"""

from __future__ import annotations

import importlib
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:            # editable-install safety
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from . import session, theme, widgets            # noqa: E402
from .context import Ctx                         # noqa: E402

#: stage -> module under gui.v3.stages. The second surface's stage is the
#: SAME module built a second time (``build(ctx, stage)``): one question
#: asked twice must not become two forms that can drift apart.
STAGE_MODULES = {"mission": "mission", "airfoil": "airfoil",
                 "airfoil_aft": "airfoil",
                 # ...and so is the VERTICAL STABILISER's. It was briefly its
                 # own cut-down module, on the argument that a fin has no
                 # design lift coefficient and no shape to search. The first
                 # half is true and the second was wrong: a fin's section is
                 # the same CST-through-XFOIL problem with w_lower = -w_upper
                 # and cl_design = 0, which is FOUR variables instead of
                 # eight (airfoil.AirfoilProblem.symmetric). One question
                 # asked three times must not become three forms.
                 "airfoil_fin": "airfoil",
                 # ...and a fourth time for the car's ENDPLATE, which asks
                 # the fin's questions on a different vehicle: a symmetric
                 # section at cl = 0, screened over the symmetric library and
                 # searched as the same four-variable CST problem.
                 "airfoil_plate": "airfoil",
                 "wing": "wing", "results": "results"}

#: when THIS shell loaded its code. ``ui.run(reload=False)``, so a shell left
#: open serves the modules it imported and nothing tells anyone: a fix landed
#: while a 20-hour-old window was still running, and the run that motivated it
#: was launched from that window AFTER the fix and failed the same way, with
#: the new flag absent from its stored config. The check below is the tripwire
#: for that: source newer than this, and the window is not running it.
_LOADED_AT = time.time()

#: how often the tripwire re-stats the tree (seconds). A repaint is on a
#: 0.5 s heartbeat and the answer changes about once a session.
_STALE_EVERY_S = 5.0

_stale_cache: dict = {"at": 0.0, "files": ()}


def _stale_sources() -> tuple:
    """Source files edited since this shell imported its code.

    Throttled, and never raises: a tripwire that could take the window down
    would be worse than the staleness it reports.
    """
    now = time.time()
    if now - _stale_cache["at"] < _STALE_EVERY_S:
        return _stale_cache["files"]
    found = []
    try:
        for root in (REPO_ROOT / "gui", REPO_ROOT / "src" / "aerobo"):
            for path in root.rglob("*.py"):
                if "__pycache__" in path.parts:
                    continue
                if path.stat().st_mtime > _LOADED_AT:
                    found.append(path.name)
    except OSError:
        found = []
    _stale_cache.update(at=now, files=tuple(sorted(set(found))))
    return _stale_cache["files"]


def _mount(ctx: Ctx, stage: str):
    from nicegui import ui

    try:
        mod = importlib.import_module(f"gui.v3.stages.{STAGE_MODULES[stage]}")
        if stage in session.STAGE_SURFACE:
            mod.build(ctx, stage)
        else:
            mod.build(ctx)
    except Exception:
        tb = traceback.format_exc()
        print(f"[v3] stage '{stage}' failed to build:\n{tb}", file=sys.stderr)
        container = ctx.views.get((stage, session.VIEWS[stage][0][0]))
        if container is None:
            return
        with container:
            ui.label(f"STAGE ERROR — {stage}").classes("sect-head") \
                .style(f"color:{theme.BAD}")
            ui.code(tb).classes("w-full")


def assemble(medium: str = "air", *, lazy: bool = False) -> Ctx:
    """Build the entire shell and return its context (no server started).

    ``lazy`` paints only the view that is on screen and OWES the rest (see
    the loop at the end of this function, and ``Ctx.render_when_shown``).
    The shells pass it; tests do not, so "assemble, then read any view"
    still holds everywhere it is relied on.
    """
    from nicegui import ui

    from gui.nice_app import RunManager

    session.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    theme.apply()
    ui.query(".nicegui-content").classes("p-0 gap-0").style(
        "height:100vh;max-height:100vh;overflow:hidden")

    ctx = Ctx(session.make_session(medium), RunManager())
    S = ctx.S

    # FIXED to the viewport, not merely 100vh tall: the shell sits inside
    # nicegui's own page wrappers, and one of them is a few pixels wider
    # than the window — which pushed the tool bar's right-hand chips and the
    # status bar's last cell off the edge of a shell that cannot scroll.
    root = ui.element("div").style(
        "position:fixed;top:0;left:0;right:0;bottom:0;display:flex;"
        "flex-direction:column;overflow:hidden;background:var(--canvas)")
    with root:
        _menubar(ctx)
        toolbar_state = _toolbar(ctx)
        body = ui.element("div").style(
            "flex:1 1 auto;min-height:0;display:flex;gap:2px;padding:2px;")
        with body:
            # ---- left column: tree over properties
            left = ui.element("div").style(
                "width:276px;flex:0 0 276px;display:flex;"
                "flex-direction:column;gap:2px;min-height:0")
            with left:
                with widgets.pane("Simulation", flex="flex:3 1 0;"
                                                    "min-height:0"):
                    ctx.tree = widgets.Tree(on_select=lambda k: _select_key(
                        ctx, k), on_toggle=lambda k: _toggle(ctx, k))
                with widgets.pane("Properties", flex="flex:2 1 0;"
                                                     "min-height:0"):
                    ctx.props = widgets.PropertyGrid()

            # ---- right column: tabbed work area over the output log
            right = ui.element("div").style(
                "flex:1 1 auto;min-width:0;display:flex;"
                "flex-direction:column;gap:2px;min-height:0")
            with right:
                main = ui.element("div").classes("pane").style(
                    "flex:1 1 auto;min-height:0")
                with main:
                    ctx.tabs = widgets.TabStrip(
                        on_change=lambda v: _select_view(ctx, v))
                    work = ui.element("div").classes("work w-full").style(
                        "flex:1 1 auto;min-height:0")
                    with work:
                        for stage in session.STAGES:
                            for key, _lbl, _icon in session.VIEWS[stage]:
                                col = ui.column().classes(
                                    "work-pad w-full gap-3")
                                col.set_visibility(False)
                                ctx.views[(stage, key)] = col
                with widgets.pane("Output", white=True,
                                  header_extra=lambda: _log_tools(ctx),
                                  flex="flex:0 0 150px;min-height:0"):
                    ctx.output = widgets.OutputLog()
        ctx.statusbar = widgets.StatusBar()

    # ---- shell services the stages call
    def select(stage: str, view: str | None = None):
        state = session.stage_states(S)[stage][0]
        if state == "locked":
            reason = session.stage_states(S)[stage][1]
            ui.notify(reason or f"{stage} is not available yet", type="info")
            return
        S["ui"]["selected"] = stage
        S["ui"]["expanded"][stage] = True
        if view is not None:
            S["ui"]["tab"][stage] = view
        _show(ctx)
        ctx.render(stage, S["ui"]["tab"][stage])
        _paint_shell(ctx, toolbar_state)

    ctx._select = select
    ctx._refresh = lambda: _paint_shell(ctx, toolbar_state)
    ctx.register("snippet", lambda: _snippet(ctx))
    # the Edit menu's own item, registered so the rule it enforces (no
    # chosen-section twin without a chosen section) can be driven through the
    # REAL handler rather than by writing the state a test hopes it writes
    ctx.register("clear_section", lambda: _clear_section(ctx))

    for stage in session.STAGES:
        _mount(ctx, stage)

    _show(ctx)
    for stage in session.STAGES:
        # LAZY is what the running shell uses: ~29 views are mounted and one
        # of them is on screen, so painting all of them costs 1.2-2.8 s of
        # blocked event loop and ~750 KB down the socket for a page the user
        # sees a fiftieth of. Every page request pays it — a reload, a
        # reconnect, a second tab — and in the NATIVE window (a WKWebView,
        # not Chrome) that is the "it takes a while before I can use it".
        # The rest are owed and painted by `select` as they are opened.
        #
        # Eager is the DEFAULT because it is the contract every test in the
        # suite is written against: assemble, then read any view.
        (ctx.render_when_shown if lazy else ctx.render)(stage)
    _paint_shell(ctx, toolbar_state)
    ctx.log("V3.5 session started — state the mission, then the section, "
            "then the wing. The search settings (which optimiser, how many "
            "evaluations) are stage 1's Search & budget tab: they open on "
            "the measured recommendation and can be taken over at any "
            "point.", "info")

    ui.timer(0.5, lambda: _tick(ctx, toolbar_state))
    return ctx


# ------------------------------------------------------------------ chrome
def _menubar(ctx: Ctx):
    from nicegui import ui

    with ui.element("div").classes("cae-menubar w-full"):
        ui.label("AeroBO").classes("cae-appname")
        _menu(ctx, "File", [
            ("New session", "restart_alt", lambda: _new_session(ctx)),
            ("Open results folder path", "folder_open",
             lambda: _show_path(ctx)),
        ])
        _menu(ctx, "Edit", [
            ("Reset design box", "crop_free", lambda: _reset_box(ctx)),
            ("Clear chosen section", "layers_clear",
             lambda: _clear_section(ctx)),
        ])
        _menu(ctx, "Solution", [
            ("Run current stage", "play_arrow", lambda: _run_stage(ctx)),
            ("Stop", "stop", lambda: _stop(ctx)),
        ])
        _menu(ctx, "Tools", [
            ("Copy run snippet to output", "content_copy",
             lambda: _snippet(ctx)),
            ("Clear output log", "clear_all",
             lambda: ctx.output and ctx.output.clear()),
        ])
        _menu(ctx, "Help", [("About this pipeline", "info",
                             lambda: _about(ctx))])


def _menu(ctx: Ctx, title: str, items: list[tuple]):
    from nicegui import ui

    with ui.element("div").classes("cae-menu"):
        ui.label(title)
        with ui.menu().props("auto-close"):
            for label, icon, fn in items:
                with ui.menu_item(on_click=fn):
                    with ui.row().classes("items-center gap-2 no-wrap"):
                        ui.icon(icon).style(f"color:{theme.INK_MUTED}")
                        ui.label(label)


def _toolbar(ctx: Ctx) -> dict:
    from nicegui import ui

    state: dict = {}
    with ui.element("div").classes("cae-toolbar w-full"):
        widgets.toolbar_button("restart_alt", "New session — reset every "
                               "stage", lambda: _new_session(ctx))
        widgets.toolbar_sep()
        state["run"] = widgets.toolbar_button(
            "play_arrow", "Run the current stage", lambda: _run_stage(ctx),
            label="Run", primary=True)
        state["stop"] = widgets.toolbar_button(
            "stop", "Stop what is running", lambda: _stop(ctx), label="Stop")
        widgets.toolbar_sep()
        widgets.toolbar_button("chevron_left", "Previous stage",
                               lambda: _step(ctx, -1))
        widgets.toolbar_button("chevron_right", "Next stage",
                               lambda: _step(ctx, +1))
        widgets.toolbar_sep()
        state["crumb"] = ui.row().classes("items-center gap-1 no-wrap")
        ui.space()
        state["chips"] = ui.row().classes("items-center gap-2 no-wrap")
    return state


def _log_tools(ctx: Ctx):
    from nicegui import ui

    ui.button(icon="clear_all",
              on_click=lambda: ctx.output and ctx.output.clear()) \
        .props("flat dense size=sm").classes("tb-btn") \
        .tooltip("Clear the output log")


# ------------------------------------------------------------- navigation
def _select_key(ctx: Ctx, key: str):
    stage, _, view = key.partition("/")
    ctx.select(stage, view or None)


def _select_view(ctx: Ctx, view: str):
    ctx.select(ctx.S["ui"]["selected"], view)


def _toggle(ctx: Ctx, key: str):
    stage = key.partition("/")[0]
    exp = ctx.S["ui"]["expanded"]
    exp[stage] = not exp.get(stage, False)
    ctx.refresh()


def _step(ctx: Ctx, delta: int):
    # a stage this configuration does not have (2.5 with no second surface)
    # is not a step the arrows can land on
    stages = [s for s in session.STAGES if session.stage_visible(ctx.S, s)]
    if ctx.S["ui"]["selected"] not in stages:
        ctx.select(stages[0])
        return
    i = stages.index(ctx.S["ui"]["selected"])
    j = max(0, min(len(stages) - 1, i + delta))
    ctx.select(stages[j])


def _show(ctx: Ctx):
    """Show exactly the active stage's active view."""
    S = ctx.S
    active = (S["ui"]["selected"], S["ui"]["tab"][S["ui"]["selected"]])
    for key, col in ctx.views.items():
        col.set_visibility(key == active)


# ------------------------------------------------------------- shell paint
def _badge(ctx: Ctx, stage: str) -> str:
    S = ctx.S
    # ...a stage a LATER SHELL added answers for itself, because the
    # fall-through at the bottom of this function is the wing search's
    # objective and would paint it on any stage it has never heard of
    own = session.stage_badge(S, stage)
    if own is not None:
        return own
    if stage == "mission":
        # ...whichever point this mode's stage 1 states: the mission's trim
        # lift coefficient, or the stated flow's Reynolds number
        if session.airfoil_only(S):
            return session.flow_summary(S)
        dp = session.design_point(S)
        return "invalid" if "error" in dp else f"CL {dp['cl_design']:.3f}"
    if stage in session.STAGE_SURFACE:
        return session.section_summary(S, session.STAGE_SURFACE[stage])
    if stage == "wing":
        from aerobo import api

        sp = api.PROBLEM_SPECS[S["wing"]["problem"]]
        return f"{len(sp.param_labels)}-D"
    rec = S["run"]["record"]
    if not rec:
        return ""
    best = rec.get("best_score")
    return f"f {best:.4g}" if isinstance(best, (int, float)) else ""


def _nodes(ctx: Ctx) -> list[dict]:
    S = ctx.S
    states = session.stage_states(S)
    nodes = []
    for stage in session.STAGES:
        # the second surface's stage exists only on a vehicle that HAS one:
        # it appears in the tree the moment stage 1 adds a tail (or a tandem
        # rear wing), and is not a greyed-out node the rest of the time
        if not session.stage_visible(S, stage):
            continue
        state, reason = states[stage]
        if stage == "wing" and ctx.manager.running:
            state = "running"
        if state != "locked" and stage == S["ui"]["selected"] \
                and state != "done":
            state = "active"
        expanded = bool(S["ui"]["expanded"].get(stage))
        nodes.append({"key": stage, "label": session.stage_label(S, stage),
                      "level": 0, "icon": None, "state": state,
                      "badge": _badge(ctx, stage),
                      "enabled": state != "locked", "children": True,
                      "expanded": expanded, "tip": reason})
        if not expanded:
            continue
        for key, label, icon in session.views_of(S, stage):
            nodes.append({"key": f"{stage}/{key}", "label": label,
                          "level": 1, "icon": icon,
                          "state": "ready" if state != "locked" else "locked",
                          "enabled": state != "locked", "children": False,
                          "expanded": False, "badge": "", "tip": ""})
    return nodes


def _paint_shell(ctx: Ctx, tb: dict):
    from nicegui import ui

    from aerobo import api

    S = ctx.S
    stage = S["ui"]["selected"]
    ctx.tree.selected = (f"{stage}/{S['ui']['tab'][stage]}"
                         if S["ui"]["expanded"].get(stage) else stage)
    ctx.tree.render(_nodes(ctx))
    # ...and the TAB the strip lights, normalised first: a session can
    # change shape under its own selection (the medium is switchable in
    # stage 1), and a strip asked to highlight a view it is not drawing
    # highlights nothing at all while ``_show`` goes on displaying that
    # view's container.
    offered = session.views_of(S, stage)
    if S["ui"]["tab"][stage] not in {k for k, _l, _i in offered}:
        S["ui"]["tab"][stage] = offered[0][0]
    ctx.tabs.render([(k, lbl) for k, lbl, _ in offered],
                    active=S["ui"]["tab"][stage])
    ctx.props.set(_properties(ctx))

    tb["crumb"].clear()
    with tb["crumb"]:
        crumbs = [s for s in session.STAGES if session.stage_visible(S, s)]
        for i, st in enumerate(crumbs):
            if i:
                ui.label("▸").classes("tb-crumb")
            lab = ui.label(session.stage_label(S, st).split("  ")[-1]) \
                .classes("tb-crumb")
            if st == stage:
                lab.classes("tb-crumb").style(
                    f"color:{theme.INK};font-weight:600")
            lab.on("click", lambda _, s=st: ctx.select(s))

    # WHAT THIS SESSION IS FLYING. In airfoil-only mode that is the section
    # problem itself (``api.AIRFOIL_PROBLEM`` — the 8-D CST + live-XFOIL
    # family stage 2 already runs), not the wing family the session still
    # carries underneath for the helpers that read a design box: quoting the
    # wing's dimension and the wing's budget in a session with no wing stage
    # described a run nothing was going to make.
    only = session.airfoil_only(S)
    sp = api.PROBLEM_SPECS[api.AIRFOIL_PROBLEM if only
                           else S["wing"]["problem"]]
    tb["chips"].clear()
    with tb["chips"]:
        stale = _stale_sources()
        if stale:
            widgets.tag("CODE CHANGED — RESTART", theme.WARN).tooltip(
                "This window is running the code it imported at start-up "
                "(the shell is served with reload off), and "
                f"{len(stale)} source file"
                + ("s have" if len(stale) != 1 else " has")
                + " changed since: "
                + ", ".join(stale[:6])
                + ("…" if len(stale) > 6 else "")
                + ". Runs launched from here still use the OLD code.")
        if only:
            # the section's registered medium is always "air" (it is a 2-D
            # problem), so the chip says which FLUID was actually stated —
            # and that this session designs a section and nothing else
            widgets.tag("SECTION ONLY", theme.ACCENT)
            widgets.tag(session.FLUID_LABELS[
                str(session.flow_state(S)["fluid"])].upper(),
                theme.INK_MUTED)
        else:
            widgets.tag(sp.medium.upper(), theme.INK_MUTED)
        if sp.is_constrained:
            widgets.tag("CONSTRAINED", theme.WARN)
        if sp.slow:
            widgets.tag("XFOIL", theme.WARN)
    # the budget the RUN will spend, not the raw field: a fresh session opens
    # in "recommended", where `config.cfg_dict` builds from
    # `effective_wing_search`, and the status bar quoting `W["budget"]` said
    # "budget 40" while every family flies its own measured plan (29 for
    # air/car, 78 for water). Same single source the card and the snippet read.
    # the SECTION problem builds its design vector out of flags, so its spec
    # publishes no static labels (``param_labels`` is empty) — the honest
    # dimension is the one the measured plan is sized on, and quoting the
    # spec here put "dim 0" in the status bar of every airfoil-only session
    dim = len(sp.param_labels)
    if only:
        plan = session.airfoil_plan(S, "main")
        if plan is not None:
            dim = int(plan.dim)
    ctx.statusbar.set_cells([
        ("problem", sp.display[:38]),
        ("dim", str(dim)),
        ("budget", str(int((session.effective_airfoil_search(S, "main")
                            if only else session.effective_wing_search(S)
                            )["budget"]))),
    ])
    # ...the REACH included: `_stop` has always handled it, but the button
    # that calls `_stop` was greyed out while it ran, so the one search with
    # no Stop of its own was the one the toolbar could not stop.
    running = ctx.manager.running or bool(
        ((S.get("run") or {}).get("relax") or {}).get("busy")) or any(
        session.airfoil_state(S, sf)[k]["running"]
        for sf in session.SURFACES for k in ("screen", "opt")) or any(
        r for r, _a, _kw in session.extra_running(S))
    tb["stop"].set_enabled(bool(running))


def _flow_properties(ctx: Ctx, stage: str) -> list[tuple]:
    """The facts about the selected node in AIRFOIL-ONLY mode.

    A separate function rather than six branches inside :func:`_properties`:
    every row of that one is about a vehicle (a design weight, a reference
    area, a span flown, a solver's design box), and this session has none.
    What it quotes instead is the stated flow and the section that flies in
    it — the same numbers, read from :func:`session.flow_point` and
    :func:`session.section_conditions`, so the grid cannot disagree with
    stage 1's card or with what the search runs at.
    """
    S = ctx.S
    pt = session.flow_point(S)
    f = session.flow_state(S)
    rows: list[tuple] = [("group", "Flow"),
                         ("stated as", session.FLUID_LABELS[str(f["fluid"])])]
    if "error" in pt:
        rows.append(("state", pt["error"], theme.BAD))
        return rows
    if pt["water"]:
        from aerobo import api

        rows.append(("water", api.water_kinds()[pt["water"]]))
    if str(f["fluid"]) == "air":
        rows.append(("altitude", f"{pt['altitude_m']:.4g} m"))
    flown = session.flown_mach(S)
    rows += [("speed", f"{pt['v_ms']:.2f} m/s"),
             ("reference chord", f"{pt['chord_m']:.4f} m"),
             ("density", f"{pt['rho']:.4g} kg/m³"),
             ("viscosity", f"{pt['mu']:.4g} Pa·s"),
             ("dynamic pressure", f"{pt['q']:.4g} Pa"),
             ("Reynolds number", f"{pt['re']:.4g}"),
             ("Mach", f"{pt['mach']:.4f}"
              + ("" if flown else "  (not flown)"),
              theme.WARN if pt["mach"] > session.MACH_WARN and flown else ""),
             ("CL design", f"{pt['cl_design']:.4f}")]
    if stage in session.STAGE_SURFACE or stage == "airfoil":
        rows.append(("group", "Section"))
        sec = S["airfoil"].get("section") or {}
        rows.append(("source", S["airfoil"]["decision"] or "not chosen"))
        if sec:
            rows.append(("name", sec.get("name", "—")))
            if sec.get("tc"):
                rows.append(("t/c", f"{sec['tc']:.4f}"))
            if sec.get("ldcr") is not None:
                rows.append(("L/D at design Cl", f"{sec['ldcr']:.2f}"))
        cond = session.section_conditions(S, "main")
        rows += [("screened at Re", f"{cond['re']:.4g}"),
                 ("screened at M", f"{cond['mach']:.4f}"),
                 ("screened at Cl", f"{cond['cl_design']:.4f}")]
    return rows


def _properties(ctx: Ctx) -> list[tuple]:
    """The facts about the selected node (read-only, single-source)."""
    from aerobo import api

    from gui import metrics

    S = ctx.S
    stage = S["ui"]["selected"]
    rows: list[tuple] = []
    if session.airfoil_only(S):
        return _flow_properties(ctx, stage)
    dp = session.design_point(S)
    rows.append(("group", "Mission"))
    rows.append(("medium", session.MEDIA[S["medium"]][0]))
    if "error" in dp:
        rows.append(("state", dp["error"], theme.BAD))
    else:
        rows += [("design weight", f"{dp['W_N']:.1f} N"),
                 ("speed", f"{dp['v_ms']:.2f} m/s"),
                 ("density", f"{dp['rho']:.4g} kg/m³"),
                 ("dynamic pressure", f"{dp['q']:.4g} Pa"),
                 ("reference area", f"{dp['s_ref_m2']:.3f} m²"),
                 ("CL design", f"{dp['cl_design']:.4f}")]
    if stage in ("airfoil", "airfoil_aft", "wing", "results"):
        rows.append(("group", "Section"))
        sec = S["airfoil"].get("section") or {}
        rows.append(("source", S["airfoil"]["decision"] or "not chosen"))
        # the chord-dependent numbers sit with the SECTION, because that is
        # whose they are: they follow stage 2's aspect-ratio estimate, not
        # the planform the run flies
        if "error" not in dp:
            rows += [("AR estimate",
                      f"{session.section_aspect_ratio(S):.4g}"),
                     ("MAC (estimate)", f"{dp['mac']:.4f} m"),
                     ("Re at MAC", f"{dp['re_mac']:.4g}")]
        if sec:
            rows.append(("name", sec.get("name", "—")))
            if sec.get("tc"):
                rows.append(("t/c", f"{sec['tc']:.4f}"))
            if sec.get("ldcr") is not None:
                rows.append(("L/D at design Cl", f"{sec['ldcr']:.2f}"))
        cond = session.section_conditions(S, "main")
        rows.append(("screened at Re", f"{cond['re']:.4g}"))
        rows.append(("screened at Cl", f"{cond['cl_design']:.4f}"))
        # the SECOND surface, where there is one: it is a different section
        # at a different chord, so it gets its own rows rather than being
        # folded into the wing's and quietly implying they are the same
        aft = session.aft_surface(S)
        if aft:
            rows.append(("group", aft.capitalize()))
            own = session.section_is_own(S, "aft")
            aft_sec = session.section_of(S, "aft") or {}
            rows.append(("section", (aft_sec.get("name", "—") if own
                                     else "as the wing")))
            geo = session.surface_geometry(S, "aft")
            if geo:
                rows.append(("mean chord", f"{geo['mac']:.4f} m"))
                adp = session.surface_design_point(S, "aft")
                if "error" not in adp:
                    rows.append(("Re at its chord", f"{adp['re_mac']:.4g}"))
                    # a trimming surface's design Cl is the TRIM balance's,
                    # not the mission's — and it is not zero
                    trims = not geo["lifting"]
                    rows.append(("design Cl",
                                 f"{adp['cl_design']:.4f}"
                                 + ("  (it trims)" if trims else ""),
                                 theme.ACCENT if trims else ""))
            acond = session.section_conditions(S, "aft")
            rows.append(("screened at Re", f"{acond['re']:.4g}"))
            rows.append(("screened at Cl", f"{acond['cl_design']:.4f}"))
            rows.append(("weights", session.surface_job(S, "aft")
                         + ("" if session.weights_are_recommended(S, "aft")
                            else " (edited)")))
    if stage in ("wing", "results"):
        sp = api.PROBLEM_SPECS[S["wing"]["problem"]]
        rows.append(("group", "Wing problem"))
        size = session.flown_size(S) or api.planform_size(S["wing"]["problem"])
        if size:
            rows += [("span flown", f"{size[0]:.3f} m"),
                     ("area flown", f"{size[1]:.3f} m²")]
        # what the RUN will fly, not the raw session field. In "recommended"
        # the two differ by construction — the `bo -> bo_slsqp` swap, the
        # measured budget, and an "own → ga → back to recommended" round trip
        # all leave `W` naming a solver and a budget nothing will use. Same
        # source `config.cfg_dict` builds from, so the grid a user reads
        # before pressing Run cannot disagree with the run.
        eff = session.effective_wing_search(S)
        rows += [("solver", sp.display),
                 ("dimension", str(len(sp.param_labels))),
                 ("constrained", "yes" if sp.is_constrained else "no"),
                 ("optimiser",
                  api.OPTIMISER_SPECS[eff["optimiser"]].display
                  .split(" — ")[0]),
                 ("budget", f"{int(eff['budget'])} evaluations"),
                 ("seed", str(int(eff["seed"])))]
    if stage == "results" and S["run"]["record"]:
        rec = S["run"]["record"]
        rows.append(("group", "Result"))
        best = rec.get("best_score")
        rows += [("best objective",
                  f"{best:.6g}" if isinstance(best, (int, float)) else "—"),
                 ("evaluations", str(rec.get("n_evals", "—"))),
                 ("wall time", f"{rec.get('wall_time_s', 0.0):.1f} s")]
        # how much of the answer the BOX decided: a run whose optimum sits on
        # its bounds was stopped by the box, not by the aerodynamics, and the
        # summary table on stage 4 says which rows
        box_rows = metrics.design_box(
            rec.get("param_labels") or [], rec.get("best_x") or [],
            rec.get("bounds"),
            (rec.get("config") or {}).get("bounds_overrides"))
        riding = [r for r in box_rows if r["riding"]]
        if any(r["lo"] is not None for r in box_rows):
            rows.append(("at a box bound",
                         f"{len(riding)} of {len(box_rows)} variables",
                         theme.WARN if riding else ""))
    return rows


# ------------------------------------------------------------------ actions
#: stage -> the action its Run button means. A module global, and read with
#: ``.get``, so a shell that mounts more stages can add to it instead of
#: this function growing a branch for a stage V3 does not have.
RUN_ACTIONS: dict = {"mission": "accept_mission", "airfoil": "run_airfoil",
                     "airfoil_aft": "run_airfoil_aft",
                     "airfoil_fin": "run_airfoil_fin",
                     "airfoil_plate": "run_airfoil_plate", "wing": "launch",
                     "results": "refresh_results"}


def _run_stage(ctx: Ctx):
    """The toolbar Run button: run whatever the ACTIVE stage means by run."""
    from nicegui import ui

    stage = ctx.S["ui"]["selected"]
    # ``.get``, and a TABLE rather than a literal: a shell that mounts a
    # stage this one has never heard of used to reach a KeyError here, which
    # is a traceback on the one button whose failure mode should be a
    # sentence. The table is a module global so such a shell can extend it.
    action = RUN_ACTIONS.get(stage)
    if action is None:
        ui.notify(f"nothing to run on the {stage} stage", type="info")
        return
    if action not in ctx.actions:
        ui.notify(f"nothing to run on the {stage} stage", type="info")
        return
    ctx.act(action)


def _stop(ctx: Ctx):
    from nicegui import ui

    stopped = False
    if ctx.manager.running:
        ctx.act("cancel")
        stopped = True
    if ((ctx.S.get("run") or {}).get("relax") or {}).get("busy"):
        ctx.act("relax_stop")
        stopped = True
    # every surface that can be running, from the one table that names them
    for surface in session.SURFACE_STAGES:
        action = "stop_airfoil" + ("" if surface == "main" else f"_{surface}")
        a = session.airfoil_state(ctx.S, surface)
        if a["screen"]["running"] or a["opt"]["running"]:
            ctx.act(action)
            stopped = True
    # ...and whatever loop a later shell's own stage is running. The button
    # above is enabled off the same list, so the two cannot disagree about
    # whether there is anything to stop.
    for is_running, action, args in session.extra_running(ctx.S):
        if is_running and action in ctx.actions:
            ctx.act(action, *args)
            stopped = True
    if not stopped:
        ui.notify("nothing is running", type="info")


def _new_session(ctx: Ctx):
    from nicegui import ui

    if ctx.manager.running:
        ui.notify("a run is in progress — stop it first", type="warning")
        return
    # ...and the REACH, which is a search like any other but is not a RunJob:
    # the sub-dicts below are refilled IN PLACE, so a worker still writing
    # into S["run"] would repopulate the session that was just reset
    if ((ctx.S.get("run") or {}).get("relax") or {}).get("busy"):
        ui.notify("the reach is still searching — stop it first",
                  type="warning")
        return
    # ...and the two SECTION searches per surface, for the same reason and
    # worse. Each worker captured ``A["screen"]`` / ``A["opt"]`` when it was
    # started; the in-place refill below replaces their contents, so the
    # orphaned run kept writing its `scoring`/`score` into the FRESH
    # session's optimise tab, while the toolbar's Stop went grey (`running`
    # recomputes False) and Run would have launched a second concurrent
    # XFOIL search beside the one still going.
    if any(session.airfoil_state(ctx.S, sf)[k]["running"]
           for sf in session.SURFACES for k in ("screen", "opt")):
        ui.notify("a section search is still running — stop it first",
                  type="warning")
        return
    fresh = session.make_session(ctx.S["medium"])
    # the MODE survives, exactly as the medium does: "new session" resets the
    # answers, not the question being asked. Restarting an airfoil-only
    # session into the vehicle pipeline would open four stages the user did
    # not ask for and lock two of the ones they were using.
    fresh["mode"] = session.session_mode(ctx.S)
    fresh["ui"] = ctx.S["ui"]
    # every stage closed over ITS sub-dict when it was built (``A``, ``W``,
    # ``R``), so the sub-dicts are refilled IN PLACE rather than replaced:
    # swapping the objects left each stage writing into a dict nothing else
    # reads any more, and the shell painting the session it no longer had
    for key, val in list(fresh.items()):
        cur = ctx.S.get(key)
        if isinstance(cur, dict) and isinstance(val, dict) and key != "ui":
            cur.clear()
            cur.update(val)
            fresh[key] = cur
    ctx.S.clear()
    ctx.S.update(fresh)
    ctx.S["ui"]["selected"] = "mission"
    for stage in session.STAGES:
        ctx.render(stage)
    ctx.select("mission", "operating")
    ctx.log("new session — every stage reset to the published defaults",
            "info")


def _reset_box(ctx: Ctx):
    """Edit ▸ Reset design box — the stage's OWN button, not a second copy.

    This used to clear three keys here (``bounds``, ``bounds_off``,
    ``fixed``) while stage 3's button cleared four more. Two of the
    omissions were not cosmetic: clearing ``fixed`` without clearing the
    automatic recommendation's stamp re-armed ``_auto_wanted()``, so the
    next 0.5 s heartbeat re-measured and wrote the bands back with source
    "recommended" — the log said "design box reset" and half a second later
    the twist rows were back. And never calling ``write_size_bands`` handed
    the span back the family's PUBLISHED ``b_m [6.0, 40.0]`` for every
    mission, re-opening the "sized box never heard the mission" defect this
    shell already fixed. One button, one implementation.
    """
    if "reset_box" not in ctx.actions:
        # the wing stage failed to build; ``act`` would be a silent no-op
        ctx.log("the wing stage is not built — there is no design box to "
                "reset", "warn")
        return
    ctx.act("reset_box")


def _clear_section(ctx: Ctx):
    session.set_section(ctx.S, None, None)
    # ...and the twin that exists to fly one goes with it, or the run dies on
    # a MissingSectionError nothing on screen could undo
    if session.release_chosen_section(ctx.S):
        for note in session.apply_choices(ctx.S):
            ctx.log(note, "warn")
        ctx.log(f"solver: {ctx.S['wing']['problem']} — the chosen-section "
                f"family has nothing left to fly, so the thickness is a "
                f"design variable again", "warn")
    ctx.render("airfoil")
    ctx.render("airfoil_aft")
    ctx.render("wing")
    ctx.refresh()
    ctx.log("chosen section cleared — the wing family flies its own published "
            "section until one is chosen", "warn")


def _snippet(ctx: Ctx):
    from gui.nice_app import reproduce_snippet

    from . import config

    # WHICH run this session can make. In airfoil-only mode there is no wing
    # run to reproduce — stage 3 is not part of it — and printing the wing
    # family's config would hand the user a snippet for a search this shell
    # will not launch. The section search is a RunConfig like any other.
    cfg = (config.section_cfg_dict(ctx.S) if session.airfoil_only(ctx.S)
           else config.cfg_dict(ctx.S))
    for line in reproduce_snippet(cfg).splitlines():
        ctx.log(line, "info")


def _show_path(ctx: Ctx):
    ctx.log(f"runs are written to {session.RESULTS_DIR}", "info")


def _about(ctx: Ctx):
    from nicegui import ui

    with ui.dialog(value=True) as dlg, ui.card().classes("p-0").style(
            "width:620px;max-width:94vw"):
        dlg.on("hide", dlg.delete)
        with ui.element("div").classes("pane-title w-full"):
            ui.label("About this pipeline")
            ui.space()
            ui.button(icon="close", on_click=dlg.close) \
                .props("flat dense size=sm")
        with ui.column().classes("w-full gap-2 p-3"):
            ui.label("Four stages, in order, each one feeding the next.") \
                .classes("sect-head")
            for n, text in enumerate([
                "MISSION — the operating point and the size guess. Everything "
                "downstream is derived from it: the section's Reynolds number "
                "and design lift coefficient, and the wing's trim target.",
                "AIRFOIL — the UIUC library is screened at that point and "
                "ranked by your criterion weights; the pick can then seed a "
                "CST + live-XFOIL shape optimisation.",
                "WING — the wing type is composed from what the registry can "
                "actually solve, the chosen section is carried into its "
                "design box where the family can hold it, and the search "
                "runs through aerobo.api.run.",
                "RESULTS — the best design: performance, constraint margins, "
                "geometry, spanwise loading and the full evaluation log.",
            ], start=1):
                with ui.row().classes("items-start gap-2 no-wrap"):
                    ui.label(f"{n}").classes("readout").style(
                        f"color:{theme.ACCENT}")
                    ui.label(text).classes("hint")
            widgets.hairline()
            ui.label("The shell holds no physics and no optimiser logic: it "
                     "builds an aerobo.api.RunConfig and calls the same entry "
                     "point the experiment harnesses use.").classes("hint")
    return dlg


# --------------------------------------------------------------------- tick
def _tick(ctx: Ctx, tb: dict):
    """0.5 s heartbeat: let the stages poll their workers, then repaint."""
    dirty = False
    for fn in list(ctx.polls):
        try:
            dirty = bool(fn()) or dirty
        except Exception as exc:                     # a poll must never kill
            print(f"[v3] poll failed: {exc}", file=sys.stderr)
    if dirty:
        _paint_shell(ctx, tb)


def run(native: bool = True, port: int = 8767):
    from nicegui import ui

    assemble(lazy=True)
    ui.run(
        title="AeroBO — design pipeline",
        favicon="🛩️",
        dark=False,
        native=native,
        window_size=(1600, 1000) if native else None,
        reload=False,
        port=port,
        show=not native,
    )
