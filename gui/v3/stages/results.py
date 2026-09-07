"""Stage 4 — RESULTS: the design the search found, and what it means.

The run record itself carries the objective, the margins and the log. The
GEOMETRY and the SECTION are not in it: both are re-evaluated from the
winning design vector through ``api.design_report`` / ``api.section_report``
in a background thread, because on an XFOIL problem that re-evaluation is a
real solve and must never block the shell (nor take it down when it fails —
a failure becomes a message in the view it belongs to).

Every number shown is one the run recorded or one the re-evaluation
produced; nothing is inferred, zero-filled or rounded into agreement.
Figures come from V1's pure builders, restyled for this shell.
"""

from __future__ import annotations

import threading
import time
import traceback

from gui import diagnose, metrics

from .. import figstyle, session, theme, widgets


#: WHAT EACH PICTURE IS, behind the ``?`` in its title bar. Stage 4 drew
#: five figures and explained one of them: a reader who wanted to know which
#: way up a section is mounted, or what the front view's dashed line is, had
#: nowhere to find out. Every sentence here is about the DRAWING — the
#: numbers are on the summary tab, and are not repeated.
FIG_HELP = {
    "planform": (
        "The wing seen from above, at the size and shape the run returned "
        "— leading edge up the page, flow down it.\n\n"
        "The straight-taper outline is drawn behind a designed chord law, "
        "so the law's effect is the gap between the two. A second surface "
        "is drawn at its own station along the body."),
    "front_view": (
        "The same design looking upstream, which is the view a dihedral, a "
        "cant angle and a tip device are actually visible in.\n\n"
        "It is true-scale in both axes: a surface a fraction of its span "
        "thick draws as a line, which is why the sections have a view of "
        "their own."),
    "sections_as_flown": (
        "Each surface's aerofoil at the ATTITUDE IT FLIES — its own chord, "
        "its own incidence, and the right way up.\n\n"
        "This is the drawing that answers \"is the tail upside down\": a "
        "surface trimmed to a download flies its camber the other way, and "
        "on the planform and the 3-D view (both true-scale) that is "
        "invisible."),
    "wing3d": (
        "The lofted surfaces, true-scale in all three axes — the same loft "
        "the STL export writes.\n\n"
        "Camber is unreadable here for exactly that reason; use the "
        "sections view for shape and this one for arrangement."),
    "spanwise": (
        "Three curves against span for the design that was scored: the "
        "chord, the local lift coefficient, and the effective angle of "
        "attack the section sees.\n\n"
        "The local cl curve is where a stall starts: a peak at the tip is "
        "a wing that drops a wing first, and washout is what moves it "
        "inboard. The alpha curve is the geometric twist plus the induced "
        "angle, so it is what the SECTION flies, not what was typed."),
}


def build(ctx):     # noqa: PLR0915  (one stage, built whole)
    from nicegui import ui

    from aerobo import geometry as _geometry

    from gui import nice_app as v1

    S = ctx.S
    R = S["run"]
    seen = {"ready": None, "cad": 0}
    boxes: dict = {}
    # ``version`` is the dirty flag the 0.5 s heartbeat watches: the OpenVSP
    # build runs on a worker thread and may not touch a nicegui element
    cad_state = {"busy": False, "note": None, "level": "", "last": None,
                 "version": 0}
    fmt = v1._fmt

    # ------------------------------------------------------------ data flow
    def set_result(rd: dict):
        R["record"] = rd
        R["report"] = None
        R["section_report"] = None
        R["stamp"] = time.time()
        threading.Thread(target=_compute, args=(rd,), daemon=True).start()
        _render_all()
        ctx.refresh()

    def _publish(rd: dict, **fields):
        """Newer results win; a stale thread's answer is dropped."""
        if R.get("record") is not rd:
            return
        R.update(fields)
        R["ready"] = time.time()

    def _compute(rd: dict):
        from aerobo import api

        cfg = None
        try:
            # every field the record carries, not seven of nine: dropping
            # `pinned` and `x_seed` rebuilt a DIFFERENT run's config, so the
            # geometry and the section on this page were re-derived for a
            # search with the pinned rows free again.
            cfg = api.RunConfig(**{k: rd["config"].get(k) for k in
                                   ("problem_name", "mission_kwargs",
                                    "flags", "optimiser", "budget", "seed",
                                    "bounds_overrides", "pinned", "x_seed")})
            if rd.get("best_x") is None:
                _publish(rd, report={"error": "this run recorded no design "
                                              "vector (no feasible "
                                              "incumbent), so there is no "
                                              "geometry to draw"})
            else:
                _publish(rd, report=api.design_report(cfg, rd["best_x"]))
        except Exception as exc:      # noqa: BLE001 — a view, never fatal
            _publish(rd, report={"error": f"{type(exc).__name__}: {exc}"})
        if cfg is None or rd.get("best_x") is None:
            # SETTLE the section before leaving, for the same reason the
            # success path below settles it: set_result cleared it to None and
            # None means "still running" to every reader (the geometry view's
            # section outline, the .dat export), so returning without
            # publishing leaves them waiting for the rest of the session on a
            # run where nothing is running. Say what the geometry says.
            _publish(rd, section_report={"error": (
                "this run recorded no design vector (no feasible incumbent), "
                "so there is no section to sweep" if cfg is not None else
                "this run's configuration could not be rebuilt, so its "
                "section cannot be re-evaluated")})
            return
        try:
            # {} (never None) means "asked and answered, no section here":
            # None reads as STILL RUNNING and would spin forever
            _publish(rd, section_report=api.section_report(cfg, rd["best_x"])
                     or {})
        except Exception as exc:      # noqa: BLE001
            _publish(rd, section_report={"error":
                                         f"{type(exc).__name__}: {exc}"})

    def poll() -> bool:
        dirty = False
        if R.get("ready") and R["ready"] != seen["ready"]:
            seen["ready"] = R["ready"]
            rep = R.get("report") or {}
            if rep.get("error"):
                ctx.log(f"geometry re-evaluation: {rep['error']}", "warn")
            else:
                ctx.log("geometry and loading re-evaluated from the winning "
                        "design vector", "info")
            _render_all()
            dirty = True
        # ...and the OpenVSP build, which is a subprocess on a worker thread:
        # it publishes into ``cad_state`` and this is what puts the answer on
        # screen. A card that only redrew with the whole view would leave
        # "BUILDING…" up after the app had already opened.
        if cad_state["version"] != seen["cad"]:
            seen["cad"] = cad_state["version"]
            # the DERIVED half only: this fires every 0.5 s while the build
            # runs, and rebuilding the two inputs would destroy the one
            # under the cursor
            _render_cad_derived()
            dirty = True
        return dirty

    ctx.add_poll(poll)

    # --------------------------------------------------------------- bits
    def _bd(rd: dict) -> dict:
        return ((R.get("report") or {}).get("breakdown")
                or rd.get("breakdown") or {})

    def _geom(rd: dict) -> dict:
        return (R.get("report") or {}).get("geometry") or {}

    def _declared_constraints(rd: dict) -> tuple[bool, tuple]:
        """Does the problem this record flew HAVE constraints, and which?

        Answered from the run's OWN declaration (``is_constrained``, which
        ``RunResult.to_dict`` always emits) and, failing that, from the
        registry spec of the problem it names — never from whether the record
        happens to carry margins. A constrained search that found nothing
        feasible carries none at all, and reading that silence as "there are
        no constraints" states the exact inverse of what happened.

        The spec is LOOKED UP by name, never matched against one: the ~1445
        registered problems are generated by composing modifiers, so a branch
        on a single name would miss every twin of it.
        """
        flag = rd.get("is_constrained")
        name = ((rd.get("config") or {}).get("problem_name")
                or rd.get("problem_name"))
        spec = None
        try:
            from aerobo import api

            spec = api.PROBLEM_SPECS.get(name)
        except Exception:             # noqa: BLE001 — a view, never fatal
            spec = None
        labels = tuple(getattr(spec, "constraint_labels", ()) or ())
        if flag is None:
            flag = bool(getattr(spec, "is_constrained", False))
        return bool(flag), labels

    def _guard(fn, label: str):
        try:
            fn()
        except Exception:
            with widgets.group_box(f"{label} — render error"):
                widgets.hint("This block failed to render; the rest of the "
                             "result is unaffected.", "bad")
                ui.code(traceback.format_exc()).classes("w-full")

    def _empty(view: str):
        widgets.hint("No completed run yet. Configure the wing on stage 3 "
                     "and launch it; the result lands here.")
        ui.button("Go to the wing stage", icon="arrow_back",
                  on_click=lambda: ctx.select("wing", "solver")) \
            .props("outline dense no-caps")
        del view

    # ----------------------------------------------------- view: summary
    def _render_summary():     # noqa: PLR0915
        box = ctx.views[("results", "summary")]
        box.clear()
        rd = R.get("record")
        with box:
            if not rd:
                _empty("summary")
                return
            bd = _bd(rd)
            rows = metrics.headline(bd, _geom(rd))
            with ui.row().classes("w-full items-start gap-4 flex-wrap"):
                # THE HERO IS WHAT THE RUN MAXIMISED. L/D for every wing-like
                # family; otherwise the row that IS the objective, where the
                # family records one (metrics.OBJECTIVE_LEAD). The raw score
                # is the last resort and not the first: on a lap it is MINUS
                # the lap time, so the biggest number on the page read
                # "-61.429" beside a "lap time 61.429 s" further down — one
                # quantity, two spellings, one of them negative.
                # ...and the same trap one turn further round: on a SIZED
                # family ``score`` is payload L/D (W_fixed/D), while "LoD"
                # is still the aero number — 17.0 to 34.7 L/D points apart
                # across the 376 registered sized variants. Picking "LoD"
                # unconditionally put a number no optimiser maximised in the
                # biggest type on the page. The family says which it is
                # (api.record_score_units); only when it says "L/D" is the
                # aero number also the maximised one.
                from aerobo import api as _api
                units = _api.record_score_units(rd)
                hero = (next((r for r in rows if r["key"] == "LoD"), None)
                        if units == _api.LOD_UNITS else None)
                lead = None if hero is not None else metrics.objective_row(
                    bd, _geom(rd))
                if hero is not None:
                    widgets.readout("L/D", hero["text"], hero["unit"],
                                    tip=hero["tip"], color=theme.ACCENT)
                elif lead is not None:
                    widgets.readout(lead["label"], lead["text"], lead["unit"],
                                    tip=lead["tip"], color=theme.ACCENT)
                else:
                    # ...under the family's OWN name for it where there is
                    # one. "objective 23.05" is the same silence in a
                    # different font: on a sized run that number is a payload
                    # L/D and nothing on the page said so.
                    widgets.readout(
                        units or "objective", fmt(rd.get("best_score")),
                        tip=("what this search maximised. Payload L/D is "
                             "W_fixed/D — the FIXED weight over the drag, "
                             "not lift over drag: freeing the span or the "
                             "area makes the wing weigh itself, so the "
                             "aircraft's own L/D is reported separately "
                             "below." if units == _api.PAYLOAD_LOD_UNITS
                             else "what this search maximised"),
                        color=theme.ACCENT)
                # ...and the AERO L/D beside it whenever it is not the hero.
                # It is a real number about the aircraft and the user asked
                # for it by name; what it must never do is stand in for the
                # number the optimiser actually maximised.
                if hero is None:
                    lod_row = next((r for r in rows if r["key"] == "LoD"), None)
                    if lod_row is not None:
                        widgets.readout(
                            "L/D", lod_row["text"], lod_row["unit"],
                            tip="the aircraft's lift-to-drag at the design "
                                "point — NOT what this run maximised (see "
                                "the headline number). Trimmed, and "
                                "including the tail's drag.")
                # a composite run's L/D is real, but it is NOT what was
                # maximised — so the number that was gets its own readout
                # rather than leaving the hero implying it
                if str(((rd.get("config") or {}).get("flags") or {})
                       .get("wing_objective")) == "composite":
                    widgets.readout(
                        "composite J", fmt(rd.get("best_score")),
                        tip="the weighted score of your wing criteria — "
                            "the number this search maximised (stage 3's "
                            "objective card)")
                shown_as_hero = {"LoD"} | ({lead["key"]} if lead else set())
                for r in rows:
                    if r["key"] in shown_as_hero:
                        continue
                    widgets.readout(r["label"], r["text"], r["unit"],
                                    tip=r["tip"])
                widgets.readout("evaluations", str(rd.get("n_evals", "—")),
                                tip="designs the optimiser paid for")
                widgets.readout("wall", f"{rd.get('wall_time_s', 0.0):.1f}",
                                "s")

            _guard(lambda: _constraints(rd), "Constraints")
            _guard(lambda: _lateral(rd), "Lateral stability")
            _guard(lambda: _weight(rd), "Weight")
            _guard(lambda: _drag(rd), "Drag breakdown")
            _guard(lambda: _lap(rd), "Lap")

            _guard(lambda: _design_box(rd), "Design box")

            with widgets.group_box("Run"):
                cfgd = rd.get("config") or {}
                widgets.kv("problem", str(cfgd.get("problem_name", "—")))
                widgets.kv("optimiser", str(cfgd.get("optimiser", "—")))
                widgets.kv("budget / seed",
                           f"{cfgd.get('budget', '—')} / "
                           f"{cfgd.get('seed', '—')}")
                # a run that ENDED EARLY is not a run of that budget, and the
                # only place that can be said without the user having to
                # notice a number is here, beside the budget it did not spend
                if rd.get("partial"):
                    widgets.kv("stopped early",
                               f"{rd.get('n_evals', '—')} of "
                               f"{cfgd.get('budget', '—')} evaluations — "
                               + str(rd.get("stop_reason")
                                     or "cancelled by hand"),
                               color=theme.WARN,
                               tip="the log is what was paid for, not a "
                                   "completed search: never read it as this "
                                   "budget's result")
                _eval_cache(rd)
                _keep_going(rd)
                widgets.kv("mission sent",
                           ", ".join(f"{k}={v:g}" for k, v in
                                     (cfgd.get("mission_kwargs") or {})
                                     .items()) or "published defaults")
                widgets.kv("saved as", str(rd.get("path", "—")))

    def _eval_cache(rd: dict):
        """How many of these evaluations were flown, and how many came off disk.

        Beside the wall clock, because it is the fact that makes the wall clock
        readable: a continuation re-flies the whole prefix of the run it
        lengthens, and with the memo on (``aerobo.eval_cache``) that prefix
        costs a file read instead of a solve. A run that reports 40 hits and
        20 misses took 2 minutes to do 20 designs' worth of physics — reading
        its wall time as "this search is fast" is the error this line prevents.
        """
        cache = rd.get("eval_cache") or {}
        if not cache:
            return
        hits, misses = int(cache.get("hits", 0)), int(cache.get("misses", 0))
        total = hits + misses
        if total <= 0:
            return
        widgets.kv("flown / from cache", f"{misses} flown, {hits} off disk",
                   color=(theme.WARN if hits and misses == 0 else ""),
                   tip="the objective is a pure function of the design "
                       "vector, so a cached evaluation and a flown one are "
                       "the same number — what differs is that the wall time "
                       "above only paid for the flown ones")

    def _keep_going(rd: dict):
        """Did it flatten out, and can it be given more? — beside the budget.

        The same two facts the Wing stage's run tab carries, put where a user
        reading a FINISHED result is actually looking. A budget spent says
        nothing about whether it was enough; this says which.
        """
        conv = session.run_convergence(S, rd)
        words = {"converged": "flattened out", "climbing": "still climbing",
                 "too_short": "cannot be said", "unmeasured": "no rule",
                 "empty": "nothing finite was scored"}
        widgets.kv("converged?", words.get(conv["verdict"], conv["verdict"]),
                   color=("" if conv["verdict"] in ("converged", "unmeasured")
                          else theme.WARN),
                   tip=conv["text"])
        out = session.continue_run(S, rd)
        if out["error"] or ctx.manager.running:
            return
        note = out["note"]
        ui.button(f"Keep going — {note['added']} more "
                  f"({note['budget']} in total)", icon="play_arrow",
                  on_click=lambda _=None, r=rd: ctx.act("continue_run", r)) \
            .props("flat dense no-caps") \
            .tooltip(note["why"])

    def _design_box(rd: dict):     # noqa: PLR0915  (one table, built whole)
        """Every design variable of the winning design, against the box the
        run actually searched.

        The box comes from the RECORD (``RunResult.bounds`` — the family's
        published box with this run's overrides already in it), never from
        the spec's default box: the default box is built with no flags, so a
        cant-limited winglet or a chord-law row would be placed against
        bounds this run never had.
        """
        labels = rd.get("param_labels") or []
        best_x = rd.get("best_x") or []
        overrides = (rd.get("config") or {}).get("bounds_overrides") or {}
        with widgets.group_box("Best design in its box"):
            if not labels or not best_x:
                widgets.hint("this run recorded no design vector")
                return
            rows = metrics.design_box(labels, best_x, rd.get("bounds"),
                                      overrides)
            has_box = any(r["lo"] is not None for r in rows)
            if not has_box:
                # an old or hand-made record: say the values, and say plainly
                # that there is nothing to compare them against
                for r in rows:
                    widgets.kv(r["label"], fmt(r["value"], 6))
                widgets.hint("This record carries no design box, so these "
                             "values cannot be placed inside one.")
                return
            with ui.element("div").classes("w-full").style(
                    "display:grid;grid-template-columns:1.5fr .8fr .9fr .8fr "
                    "1.3fr auto;gap:4px 10px;align-items:center"):
                for head in ("parameter", "low", "best", "high",
                             "low → high", "  "):
                    ui.label(head).classes("readout-label")
                for r in rows:
                    lab = ui.label(r["label"]).classes("readout")
                    help_text = v1.param_help(r["label"])
                    if help_text and help_text[0]:
                        lab.tooltip(f"{help_text[0]} — {help_text[1]}")
                    edge = theme.ACCENT if r["narrowed"] else theme.INK_MUTED
                    ui.label(fmt(r["lo"], 4)).classes("readout") \
                        .style(f"color:{edge}")
                    val = ui.label(fmt(r["value"], 6)).classes("readout")
                    if r["riding"] or r["outside"]:
                        val.style("color:" + (theme.BAD if r["outside"]
                                              else theme.WARN))
                    ui.label(fmt(r["hi"], 4)).classes("readout") \
                        .style(f"color:{edge}")
                    if r["frac"] is None:
                        ui.label("—").classes("hint")
                    else:
                        color = (theme.BAD if r["outside"] else
                                 theme.WARN if r["riding"] else theme.ACCENT)
                        widgets.range_bar(r["frac"], color=color).tooltip(
                            f"{r['frac'] * 100:.1f} % of the way from "
                            f"{fmt(r['lo'], 4)} to {fmt(r['hi'], 4)}")
                    if r["outside"]:
                        widgets.tag("outside the box", theme.BAD)
                    elif r["riding"]:
                        widgets.tag(f"at the {r['riding']} bound", theme.WARN)
                    elif r["narrowed"]:
                        widgets.tag("narrowed", theme.ACCENT)
                    else:
                        ui.label("")
            riding = [r for r in rows if r["riding"]]
            widgets.hint(
                "Amber = the optimum sits within 2 % of its own box edge: "
                "there the BOX is setting that variable, not the physics "
                "(§13 boundary-riding law). Blue bounds are rows this run "
                "narrowed — a design-box edit or the section carried from "
                "stage 2; the rest is the solver's own published box.")
            if riding:
                widgets.hint(
                    f"{len(riding)} of {len(rows)} variables ran into a "
                    f"bound: " + ", ".join(r["label"] for r in riding)
                    + ". Widen those rows and run it again if the wider box "
                      "is buildable — otherwise the bound IS the answer, and "
                      "it belongs in the write-up as one.", "warn")
                ui.button("Open the design box", icon="crop_free",
                          on_click=lambda: ctx.select("wing", "box")) \
                    .props("outline dense no-caps")

    def _constraints(rd: dict):
        report = R.get("report") or {}
        clabels = report.get("constraint_labels") or []
        bd = report.get("breakdown") or rd.get("breakdown") or {}
        g = bd.get("g")
        rows = []
        if isinstance(g, (list, tuple)):
            rows = [(clabels[j] if j < len(clabels) else f"g[{j}]", v)
                    for j, v in enumerate(g)]
        elif rd.get("best_g") is not None:
            rows = [(clabels[0] if clabels else "binding margin",
                     rd.get("best_g"))]
        if not rows:
            constrained, spec_labels = _declared_constraints(rd)
            with widgets.group_box("Constraints"):
                if not constrained:
                    widgets.hint("Unconstrained problem — every evaluated "
                                 "design is admissible.")
                    return
                what = ", ".join(clabels or spec_labels)
                head = f"Constrained problem ({what})" if what \
                    else "Constrained problem"
                n_feas = rd.get("n_feasible")
                nothing = (n_feas == 0 if n_feas is not None
                           else rd.get("best_x") is None)
                if nothing:
                    # WHY, not just that. The run's own log carries the
                    # answer — which constraint was never met, whether any
                    # single design could have met them all, and which box
                    # row the best attempt was pressed against — so "widen
                    # the design box" is replaced by the row to widen.
                    lines = diagnose.infeasibility_lines(
                        diagnose.infeasibility_report(
                            rd, constraint_labels=(clabels or spec_labels)))
                    if lines:
                        widgets.hint(head, "bad")
                        for text, level in lines:
                            widgets.hint(text, level)
                    else:
                        n_evals = rd.get("n_evals")
                        widgets.hint(
                            f"{head} — no evaluated design satisfied it: "
                            f"{n_feas or 0} of "
                            f"{n_evals if n_evals is not None else '—'} "
                            f"evaluations were feasible, so there is no "
                            f"margin to show and no design to take forward. "
                            f"This record kept no per-evaluation margins, so "
                            f"it cannot say which constraint stopped it.",
                            "bad")
                    ui.button("Open the design box", icon="crop_free",
                              on_click=lambda: ctx.select("wing", "box")) \
                        .props("outline dense no-caps")
                else:
                    widgets.hint(
                        f"{head} — this record carries no margins, so they "
                        f"cannot be shown here.", "warn")
            return
        with widgets.group_box("Constraint margins"):
            for name, value in rows:
                ok = value is not None and float(value) >= 0.0
                with ui.row().classes("w-full items-center gap-2 no-wrap"):
                    widgets.tag("OK" if ok else "VIOLATED",
                                theme.GOOD if ok else theme.BAD)
                    ui.label(name).classes("field-label").style("flex:1 1 0")
                    ui.label(fmt(value, 4)).classes("readout")
            widgets.hint("Feasible ⟺ every signed margin ≥ 0.")
            # ...and if this run was flown to a handling STANDARD, say which
            # of its clauses actually held the design. The margins above are
            # already labelled with their thresholds; what a reader cannot
            # get from a list of five numbers is which one was the minimum,
            # and that is the whole answer to "what is this level costing
            # me" (aerobo.handling.Margins.binding_clause).
            lvl = bd.get("handling_level")
            if lvl is not None:
                from aerobo import handling as hq

                bound = bd.get("handling_binding")
                widgets.hint(
                    f"Flown to {hq.level_name(lvl)} — {hq.level_source(lvl)}. "
                    f"Five of the rows above are its clauses"
                    + (f", and the one this design was held by is the "
                       f"{bound}." if bound else "."))

    def _lateral(rd: dict):
        """WHICH WAY THE WING IS CANTED, AND WHETHER THE SPIRAL CONVERGES.

        The answer's own stage named NEITHER. A run that searched the cant
        put ``wing_dihedral_deg`` in the design-vector table as a signed
        number and stopped there — so meeting an anhedral wing meant
        reading "-10.00" out of a table and working out unaided that the
        tips point DOWN and the aeroplane therefore rolls INTO a sideslip.
        The whole lateral half of the result had no read-out anywhere
        outside stage 3's advice card, which is about a design box and not
        about this answer.

        Everything here is read off the BREAKDOWN THE RUN RECORDED —
        ``api.lateral_verdict``, which re-derives nothing and re-flies
        nothing — so it is the aeroplane that was scored. A family that
        states no cant gets no card at all rather than a row of dashes.
        """
        from aerobo import api as _api

        v = _api.lateral_verdict(_bd(rd))
        if v["status"] == "not_applicable":
            return
        with widgets.group_box("Lateral stability"):
            widgets.hint(v["says"], v["level"])
            if v["wing_dihedral_deg"] is not None:
                widgets.kv("wing dihedral",
                           f"{v['wing_dihedral_deg']:+.2f} deg",
                           color=(theme.WARN
                                  if v["wing_dihedral_deg"] < 0 else ""),
                           tip="tips UP positive. The wing's own cant, and "
                               "the only wing-side source of Cl_beta that "
                               "holds at any lift — not to be read as the "
                               "TAIL's cant, which a V-tail states "
                               "separately.")
            if v["sweep_deg"] is not None:
                widgets.kv("wing sweep", f"{v['sweep_deg']:+.2f} deg",
                           tip="quarter-chord, aft positive. Its dihedral "
                               "effect goes as CL and vanishes at cruise, "
                               "so it is not a roll lever.")
            for key, label, tip in (
                    ("Cn_beta", "yaw stiffness Cn_β",
                     "positive weathercocks into the airflow. At or below "
                     "zero the spiral criterion is refused rather than "
                     "scored: the sign of its second product flips."),
                    ("Cl_beta", "dihedral effect Cl_β",
                     "negative rolls the aeroplane OUT of a sideslip."),
                    ("Cl_r", "roll due to yaw Cl_r", ""),
                    ("Cn_r", "yaw damping Cn_r", ""),
                    ("spiral_margin", "spiral margin",
                     "Cl_β(Cn_r − t·Cn_p) − Cn_β(Cl_r − t·Cl_p), "
                     "t = tan(trim attitude). Positive converges.")):
                if v[key] is None:
                    continue
                bad = key == "spiral_margin" and v[key] < 0.0
                widgets.kv(label, fmt(v[key], 6),
                           color=theme.BAD if bad else "", tip=tip)
            if v["status"] == "not_measured":
                # ...to the view that HOLDS the objective card, which is
                # the SOLVER tab and not the geometry one: a button whose
                # label promises a control and lands on another page is the
                # same failure as a recommendation that cites a gate it
                # does not clear. Same destination the cant card's own
                # "weight it in the objective" button uses.
                ui.button("weight the spiral in the objective",
                          icon="tune",
                          on_click=lambda: ctx.select("wing", "solver")) \
                    .props("outline dense no-caps") \
                    .tooltip("stage 3's objective card — the `spiral` "
                             "criterion is what builds the lateral deck")

    def _drag(rd: dict):
        parts = metrics.drag_split(_bd(rd))
        if not parts:
            return
        total = sum(p["counts"] for p in parts)
        colors = {"induced": theme.ACCENT, "profile": theme.GOOD,
                  "other": theme.WARN}
        with widgets.group_box("Drag breakdown"):
            with ui.row().classes("w-full no-wrap gap-0").style(
                    "height:10px;border:1px solid var(--rule-soft)"):
                for p in parts:
                    frac = p["counts"] / total if total else 0.0
                    # height:100% — an empty div has no intrinsic height, so
                    # without it the bar renders as an empty outline
                    ui.element("div").style(
                        f"width:{frac * 100:.2f}%;height:100%;"
                        f"background:{colors.get(p['label'], theme.INK_FAINT)}"
                    ).tooltip(f"{p['label']}: {p['counts']:.1f} counts")
            for p in parts:
                widgets.kv(p["label"], f"{p['counts']:.1f} counts")
            widgets.hint(f"Total {total:.1f} counts "
                         f"(1 count = 1 × 10⁻⁴ of C_D).")

    #: standard gravity, for the kg a Newton total is also worth. One import
    #: rather than 9.81 written here, and it is ``mission.G0`` — the layer
    #: V3 is allowed to see. It was taken from V4's flight machinery, which
    #: this shell must not name at all: the separation is enforced by
    #: STRING SEARCH over these files
    #: (test_v4_stages.test_no_v3_source_file_MENTIONS_the_v4_machinery), so
    #: even a comment naming that module fails it. Same number either way —
    #: the flight layer's constant is itself this one.
    def _weight(rd: dict):
        """WHAT THE AEROPLANE WEIGHS — or why this run does not know.

        Only the SIZE modifier weighs anything (``sizing.sized_state``): with
        the span and area fixed, ``CL_target`` is given and the wing's mass
        never enters the objective, so there is no total to show. That is a
        property of the FAMILY, not a missing readout, and a panel that drew
        a dash would say the opposite.

        So the card is always present and answers one of two ways: the three
        weights where they were computed, or the sized twin of this very
        family — ``api.with_modifiers(name, mods | {"size"})``, asked rather
        than spelled out, so it names a problem that exists.
        """
        from aerobo import api
        from aerobo.mission import G0 as sd_G

        bd = _bd(rd)
        total = bd.get("W_total_N")
        if isinstance(total, (int, float)) and not isinstance(total, bool):
            wing = float(bd.get("W_wing_N") or 0.0)
            fixed = float(bd.get("W_fixed_N") or 0.0)
            total = float(total)
            with widgets.group_box("Weight"):
                widgets.kv("total", f"{total:,.1f} N  "
                                    f"({total / sd_G:,.1f} kg)",
                           tip="what the trim target was solved against: "
                               "CL = W_total / (q S), per candidate")
                widgets.kv("wing", f"{wing:,.1f} N"
                                   + (f"  ({wing / total:.1%} of it)"
                                      if total else ""),
                           tip="the statistical wing weight the SIZE "
                               "modifier makes the span pay for")
                # THE EMPENNAGE, where the layout has one. Without this row
                # the three numbers stopped adding up the moment the tail
                # began to weigh something: "everything else" is the
                # mission's fixed weight, not the remainder.
                tails = float(bd.get("W_tail_N") or 0.0)
                if tails:
                    widgets.kv("empennage", f"{tails:,.1f} N"
                                            + (f"  ({tails / total:.1%} of "
                                               f"it)" if total else ""),
                               tip="the tailplane and the fin, weighed by "
                                   "Raymer's GA correlations at this "
                                   "layout — a T-tail's fin carries the "
                                   "tailplane and is charged 20 % for it, "
                                   "and a V-tail has no fin to charge")
                widgets.kv("everything else", f"{fixed:,.1f} N",
                           tip="the fixed weight the mission stated")
                mat = bd.get("material") or bd.get("material_name")
                if mat:
                    widgets.kv("built of", str(mat),
                               tip="the material factor scaling the wing "
                                   "weight (aerobo.materials)")
                if total > 0.0:
                    # the SIZED families report their area as ``S_m2`` (it is
                    # a design variable, not a reference constant), so read
                    # that first and fall back to the fixed-size names
                    S_ref = (bd.get("S_m2") or bd.get("S") or bd.get("Sref")
                             or ((R.get("report") or {}).get("geometry")
                                 or {}).get("S"))
                    if isinstance(S_ref, (int, float)) and float(S_ref) > 0:
                        widgets.kv("wing loading",
                                   f"{total / float(S_ref):,.1f} Pa",
                                   tip="an OUTPUT here, not a choice: "
                                       "whatever W_total/S the optimiser "
                                       "landed on")
            return

        # ...and where the family does not weigh, say so WITH AN ADDRESS.
        name = str((rd.get("config") or {}).get("problem_name") or "")
        twin = None
        if name:
            try:
                twin = api.with_modifiers(
                    name, set(api.modifiers_of(name)) | {"size"})
            except Exception:                              # noqa: BLE001
                twin = None
        with widgets.group_box("Weight"):
            widgets.hint(
                "This family flies a wing of FIXED size, so nothing weighs "
                "it: the trim target is the stated CL and the mass never "
                "enters the objective. There is no total to show — not a "
                "missing readout, a question this run did not ask."
                + (f" Its sized twin does weigh it: {twin!r}, where the span "
                   f"and area become design variables and the wing pays for "
                   f"its own size." if twin and twin != name else ""))

    def _lap(rd: dict):
        """The circuit this wing was scored on — the whole lap, not its total.

        Drawn only where the run had one (``metrics.lap_report`` returns None
        otherwise), so every family without a circuit is untouched.

        WHY IT IS A PANEL AND NOT THREE MORE READOUTS. A lap time is a scalar
        and the three things that make it readable are not: which segments the
        time was spent in, the speeds the wing was actually re-flown at, and
        the balance against the window it is judged in. All three were computed
        on every lap run and printed nowhere — the defect a registered endpoint
        that never reaches the results file is.
        """
        rep = metrics.lap_report(_bd(rd))
        if rep is None:
            return
        head = rep["summary"]
        with widgets.group_box("Lap"):
            widgets.kv("circuit", head["track"],
                       tip="every number in this panel is a property of THIS "
                           "layout and this car. A different circuit ranks "
                           "wings differently — on a layout with the "
                           "straights doubled, a section that wins here "
                           "loses by most of a second.")
            widgets.kv("distance / time",
                       f"{fmt(head['length_m'], 0)} m in "
                       f"{fmt(head['lap_time_s'], 3)} s")
            for w in rep["warnings"]:
                widgets.hint(w, "warn")

            with ui.element("div").classes("w-full").style(
                    "display:grid;grid-template-columns:1.4fr .8fr .8fr .8fr "
                    ".8fr 1.6fr;gap:4px 10px;align-items:center"):
                for head_cell in ("segment", "length", "time", "share",
                                  "peak V", "limited by"):
                    ui.label(head_cell).classes("readout-label")
                for seg in rep["segments"]:
                    ui.label(seg["label"]).classes("readout")
                    ui.label(f"{fmt(seg['length_m'], 0)} m").classes("readout")
                    ui.label(f"{fmt(seg['t_s'], 2)} s").classes("readout")
                    ui.label("—" if seg["share"] is None
                             else f"{seg['share'] * 100:.1f} %") \
                        .classes("readout")
                    ui.label("—" if seg["v_peak"] is None
                             else f"{fmt(seg['v_peak'], 1)} m/s") \
                        .classes("readout")
                    note = seg["limited_by"]
                    frac = seg["drag_frac_at_peak"]
                    if frac is not None:
                        note += f" · {frac * 100:.0f} % of the engine on drag"
                    lab = ui.label(note).classes("readout")
                    if seg["v_peak_over_top"] is not None:
                        lab.tooltip(
                            f"peak speed is "
                            f"{seg['v_peak_over_top'] * 100:.1f} % of the "
                            f"drag-limited top speed — the closer to 100 %, "
                            f"the more of this straight is charged against "
                            f"the wing's drag rather than against its length")
            widgets.hint(
                "Where the lap time went. A corner reported as GRIP-limited "
                "is one downforce can still buy speed in; a straight whose "
                "peak is near the top speed is one the wing's drag is really "
                "paid for on. This is the exchange rate the lap uses in "
                "place of a weight you would otherwise have to choose.")

            if rep["points"]:
                widgets.hairline()
                with ui.element("div").classes("w-full").style(
                        "display:grid;grid-template-columns:repeat(5, 1fr);"
                        "gap:4px 10px;align-items:center"):
                    for head_cell in ("speed", "weight in the lap", "C_Z",
                                      "C_D", "ride height"):
                        ui.label(head_cell).classes("readout-label")
                    for pt in rep["points"]:
                        ui.label(f"{fmt(pt['v'], 1)} m/s").classes("readout")
                        ui.label("—" if pt["weight"] is None
                                 else f"{pt['weight'] * 100:.1f} %") \
                            .classes("readout")
                        ui.label(fmt(pt["cz"], 4)).classes("readout")
                        ui.label(fmt(pt["cd"], 5)).classes("readout")
                        ui.label("—" if pt["ride_height_m"] is None
                                 else f"{fmt(pt['ride_height_m'], 3)} m") \
                            .classes("readout")
                widgets.hint(
                    f"The {len(rep['points'])} representative "
                    f"{'speed' if len(rep['points']) == 1 else 'speeds'} the "
                    f"wing was RE-FLOWN at, each a full re-solve, with the "
                    f"share of the lap's grip-and-drag work it carries. Rows "
                    f"that are identical mean this design's coefficients do "
                    f"not move with speed — which is the honest answer for a "
                    f"fixed-polar section at a pinned ride height, and stops "
                    f"being it the moment the flown Reynolds number or a "
                    f"heave law is switched on. Ask for more speeds on the "
                    f"wing card; each one multiplies what every candidate in "
                    f"the search costs.")

            bal = rep["balance"]
            if bal is not None:
                widgets.hairline()
                lo_hi = ("" if bal["window"] is None else
                         f" (window {fmt(bal['window'][0], 3)}–"
                         f"{fmt(bal['window'][1], 3)})")
                inside = (bal["window"] is None
                          or bal["window"][0] <= bal["value"]
                          <= bal["window"][1])
                widgets.kv("aero balance",
                           f"{fmt(bal['value'], 3)} front{lo_hi}",
                           color=("" if inside else theme.WARN),
                           tip="the front axle's share of the TOTAL aero "
                               "downforce. The window's centre is derived, "
                               "not chosen: it is the static front weight "
                               "fraction, which is where the front/rear grip "
                               "split stops moving with speed. Its half-width "
                               "is a calibration (cartrack.BALANCE_HALF_WIDTH)"
                               " and the caller owns it.")
                if not inside:
                    widgets.hint(
                        "The balance is outside its window: a rear wing on "
                        "its own drives the balance rearward, and this one "
                        "has. That is a property of the CAR and the wing "
                        "together — it is reported, not designed against, "
                        "because nothing here can move the front axle.",
                        "warn")

    # ---------------------------------------------------- view: geometry
    def _render_geometry():
        box = ctx.views[("results", "geometry")]
        box.clear()
        rd = R.get("record")
        with box:
            if not rd:
                _empty("geometry")
                return
            report = R.get("report")
            if report is None:
                with ui.row().classes("items-center gap-2"):
                    ui.spinner(size="sm")
                    widgets.hint("re-evaluating the winning design for the "
                                 "geometry views…")
                return
            if report.get("error"):
                widgets.hint(str(report["error"]), "bad")
                return
            geom = report.get("geometry") or {}
            labels = rd.get("param_labels") or []
            best_x = rd.get("best_x")
            wl = geom.get("winglet") or {}
            if wl and not geom.get("is_winglet"):
                widgets.hint(
                    f"A winglet was requested at h/(b/2) = "
                    f"{wl.get('h_frac', 0):.4f} but fell below the solver's "
                    f"1 %-of-semi-span cut-off, so it was NOT flown — these "
                    f"views show the plain wing, which is what was scored.",
                    "warn")
            # ...and the same sentence for the family whose tip device is the
            # ENDPLATE. Its height is a design variable with a lower bound of
            # zero, so "none" is an answer the search is allowed to give — and
            # it gave it silently: the endplate solver exports no ``winglet``
            # block, so the warning above could never fire for it and the user
            # got four views quietly missing a surface (the front view is not
            # drawn at all, the 3-D one falls back to the bare wing).
            bd_geo = report.get("breakdown") or {}
            h_ep = bd_geo.get("endplate_h_m")
            if (isinstance(h_ep, (int, float)) and not isinstance(h_ep, bool)
                    and not geom.get("is_winglet")):
                widgets.hint(
                    f"This design carries NO endplate: the search put its "
                    f"height at {float(h_ep):.3f} m, under the solver's "
                    f"1 %-of-semi-span cut-off, so no plate was flown. The "
                    f"views below are the bare wing, and that is what was "
                    f"scored. Plate height is a design variable whose lower "
                    f"bound is zero — where the plate buys little (a thick "
                    f"plate, or a ride height that leaves it no room before "
                    f"it reaches the track) the optimiser is free to answer "
                    f"'none'.", "warn")
            # WHICH WAY UP THE PICTURES ARE. A car family solves the
            # vehicle mirrored (aerobo.geometry.MIRRORED_FRAME) so that the
            # model's lift IS its downforce; the views below turn that back
            # over and draw the car's own orientation — the wing on top of
            # the endplates that reach for the deck — while the NUMBERS
            # beside them stay in the frame they were solved in. Said here
            # because this shell strips figure titles, which is where the
            # drawers say it.
            if _geometry.is_mirrored(geom):
                widgets.hint(
                    "Drawn in the CAR's orientation: the wing sits on top "
                    "of its endplates and the track is below the picture. "
                    "The model is solved upside down (its +z is the car's "
                    "downward direction, which is why its lift is the "
                    "downforce), so every height, clearance and margin in "
                    "the numbers above is positive DOWNWARD — the axes say "
                    "so, and only the picture has been turned over.")
            # the 3-D view is true-scale, which makes camber unreadable on a
            # surface a fraction of the span thick — so the sections get
            # their own view, each on its own chord, off the same loft
            pay = session.export_payload(S) or {}
            figs = [("Planform", v1.fig_planform(geom, best_x, labels),
                     "planform", FIG_HELP["planform"]),
                    ("Front view — looking upstream", v1.fig_frontview(geom),
                     "front_view", FIG_HELP["front_view"]),
                    ("Sections as flown — which way up each surface is "
                     "mounted",
                     v1.fig_sections_as_flown(
                         geom, best_x, labels, pay.get("section"),
                         pay.get("section_aft"), report.get("breakdown")),
                     "sections_as_flown", FIG_HELP["sections_as_flown"]),
                    ("Three-dimensional view",
                     v1.fig_wing3d(geom, best_x, labels,
                                   R.get("section_report")), "wing3d",
                     FIG_HELP["wing3d"])]
            missing = [lbl for lbl, fig, _, _h in figs if fig is None]
            for label, fig, name, why in figs:
                if fig is None:
                    continue
                with widgets.group_box(label, pad=False, help=why):
                    figstyle.show(fig, name, 380)
            if missing:
                widgets.hint("Not drawn for this problem: "
                             + ", ".join(m.lower() for m in missing)
                             + " — the solver reports no geometry for it.")

    # ----------------------------------------------------- view: loading
    def _render_loading():
        box = ctx.views[("results", "loading")]
        box.clear()
        rd = R.get("record")
        with box:
            if not rd:
                _empty("loading")
                return
            report = R.get("report")
            if report is None:
                widgets.hint("still re-evaluating the winning design…")
                return
            if report.get("error"):
                widgets.hint(str(report["error"]), "bad")
                return
            fig = v1.fig_spanwise(report.get("geometry") or {})
            if fig is None:
                widgets.hint("This solver reports no spanwise distribution.")
                return
            with widgets.group_box("Spanwise distributions", pad=False,
                                   help=FIG_HELP["spanwise"]):
                figstyle.show(fig, "spanwise", 420)
            widgets.hint(
                "Chord, local lift coefficient and effective angle of attack "
                "across the span of the design that was scored.")
            recs = rd.get("history") or []
            if recs:
                with widgets.group_box("Convergence", pad=False):
                    figstyle.show(v1.fig_convergence([], history=recs),
                                  "convergence", 320)

    # There is no SECTION view. The aerofoil is stage 2's answer and stage
    # 2.5's for the second surface — its shape, its polar and the ranking it
    # was chosen from all live there. ``section_report`` is still computed
    # above, because the geometry view draws the flown outline from it and the
    # section ``.dat`` export is built from it; what was removed is a second
    # place that answered a question already answered upstream.

    # --------------------------------------------------------- view: log
    def _render_log():
        box = ctx.views[("results", "log")]
        box.clear()
        # the CAD card's handles belong to THIS build of the view: the
        # early return below leaves none, and a stale one would let the
        # heartbeat draw into a container that is no longer on screen
        # (a_page_driver_must_re_resolve_on_rebuild)
        boxes.pop("cad", None)
        boxes.pop("cad_derived", None)
        rd = R.get("record")
        with box:
            if not rd:
                _empty("log")
                return
            columns, rows = v1.eval_log_rows(rd)
            with widgets.group_box("Every evaluation", pad=False):
                if not rows:
                    with ui.column().classes("group-pad w-full"):
                        widgets.hint("this run recorded no per-evaluation "
                                     "log")
                else:
                    ui.table(columns=columns, rows=rows, row_key="i",
                             pagination=20).classes("w-full") \
                        .props("dense flat bordered")
            with widgets.group_box("Export"):
                with ui.row().classes("items-center gap-2"):
                    ui.button("Markdown summary", icon="description",
                              on_click=lambda: _download(
                                  v1.markdown_summary(rd),
                                  "aerobo_result.md")) \
                        .props("outline dense no-caps")
                    ui.button("Evaluation log (CSV)", icon="table_view",
                              on_click=lambda: _download(
                                  v1.eval_log_csv(rd),
                                  "aerobo_evaluations.csv")) \
                        .props("outline dense no-caps")
                    ui.button("Run record (JSON)", icon="data_object",
                              on_click=lambda: _download(
                                  _json(rd), "aerobo_run.json")) \
                        .props("outline dense no-caps")
                widgets.hint(f"The full record was written to "
                             f"{rd.get('path', 'the results directory')} "
                             f"when the run finished.")
            boxes["cad"] = ui.column().classes("w-full gap-0")
            _render_cad()

    def _json(rd: dict) -> str:
        import json

        return json.dumps(rd, indent=1)

    def _download(text: str, filename: str):
        ui.download.content(text, filename)
        ctx.log(f"exported {filename}", "ok")

    # ------------------------------------------------------ take it into CAD
    #
    # Two questions, and the shell used to answer only half of one. A CAD
    # export is a FILE somebody is going to open in another program, so it
    # needs a place on this machine — a browser download lands where the
    # shell cannot name it, and "open this in OpenVSP" cannot be done to a
    # file whose path nobody knows. So: the folder is asked (once, remembered
    # for the session), the files are written there, and the path is quoted
    # back. The browser download is still offered beside it, for the case it
    # is actually for — reading the result on another machine.
    def _render_cad():
        """The card SHELL — the two fields, and the box everything else
        lives in. Built when the Evaluations view is drawn, and NOT redrawn
        after that.

        The fields are separated from what they derive for the reason in
        a_card_that_draws_once_goes_stale: this card redraws itself (the
        background OpenVSP build publishes through the 0.5 s heartbeat, and
        every save publishes its own answer), and a redraw that rebuilt the
        two inputs would destroy the one being typed into. Derived
        containers can be cleared as often as state moves; fields cannot.
        """
        box = boxes.get("cad")
        if box is None:
            return
        box.clear()
        with box, widgets.group_box("Take it into CAD"):
            with ui.row().classes("w-full items-center gap-2 no-wrap"):
                # the CAPTION is beside the field, never Quasar's own
                # floating label: this shell pins a field at 26 px and a
                # floated label lands on the value, so "save to" printed
                # itself over the start of the path (widgets.text_field).
                #
                # ...and the fields hold WHAT WAS TYPED, never what it
                # resolves to: a field rebuilt from the resolved path
                # replaces "~/Desktop/wing" — or half a name — under the
                # cursor. The resolution is a readout, one line below
                # (a_typed_number_must_not_rebuild_its_field).
                widgets.text_field(
                    "save to", session.export_dir_raw(S),
                    lambda e: _set_export_dir(e.value), grow=True)
                widgets.text_field(
                    "file name", session.export_stem_raw(S),
                    lambda e: _set_export_stem(e.value), width="w-40")
            boxes["cad_derived"] = ui.column().classes("w-full gap-2")
        _render_cad_derived()

    def _render_cad_derived():
        """Everything the two fields above DERIVE — redrawn on every
        keystroke, and by the heartbeat when the build publishes.

        This used to be the same container as the fields, drawn once, so
        typing a new folder left the card naming the OLD one: the "writes"
        read-out, the sentence listing each file, the sanitised-name warning
        and the "will be REPLACED" line all went on describing the previous
        answer while the buttons wrote to the new one. The text saying where
        a file goes has to be re-derived from the answer, or it is not a
        read-out of the answer at all.
        """
        box = boxes.get("cad_derived")
        if box is None:
            return
        box.clear()
        from aerobo import vsp as vspmod

        have = vspmod.availability()
        with box:
            plan = session.export_plan(S)
            widgets.readout("writes", str(plan["names"]["stl"]))
            widgets.hint(
                f"The folder every button below writes into, on THIS "
                f"machine — created when something is written to it, and "
                f"the full path of each file is logged. The name above "
                f"becomes “{plan['stem']}”: “Save geometry” writes "
                f"{plan['names']['stl'].name}, “Save section” "
                f"{plan['names']['dat'].name}, the OpenVSP button "
                f"{plan['names']['vsp3'].name}, and the bundle writes those "
                f"plus one STL per surface.")
            if plan["typed"] and plan["typed"] != plan["stem"]:
                widgets.hint(
                    f"“{plan['typed']}” is not a file name a filesystem and "
                    f"OpenVSP's own script both take, so it is written as "
                    f"“{plan['stem']}”. Letters, digits, - _ . survive.",
                    "warn")
            if plan["existing"]:
                widgets.hint(
                    "already in that folder and will be REPLACED: "
                    + ", ".join(plan["existing"]), "warn")

            with ui.row().classes("items-center gap-2 flex-wrap"):
                open_btn = ui.button(
                    "Open in OpenVSP", icon="open_in_new",
                    on_click=lambda: _open_in_vsp()) \
                    .props("unelevated dense no-caps color=primary")
                ui.button("Save the whole bundle", icon="save",
                          on_click=lambda: _save_bundle()) \
                    .props("outline dense no-caps")
                ui.button("Save geometry (STL)", icon="view_in_ar",
                          on_click=lambda: _save_one("stl")) \
                    .props("outline dense no-caps")
                ui.button("Save section (.dat)", icon="show_chart",
                          on_click=lambda: _save_one("dat")) \
                    .props("outline dense no-caps")
                ui.button("Save build script (.py)", icon="terminal",
                          on_click=lambda: _save_one("vsp")) \
                    .props("outline dense no-caps")
                if cad_state["busy"]:
                    open_btn.disable()
                    widgets.tag("BUILDING…", theme.ACCENT)
            if have["python"] is None or have["app"] is None:
                open_btn.disable()
                missing = []
                if have["python"] is None:
                    missing.append(
                        "an interpreter that can import openvsp "
                        "(AEROBO_VSP_PYTHON)")
                if have["app"] is None:
                    missing.append("the OpenVSP application (AEROBO_VSP_APP)")
                open_btn.tooltip("not found: " + "; ".join(missing))
                widgets.hint(
                    "“Open in OpenVSP” needs " + " and ".join(missing)
                    + ". Everything else still works — “Save the whole "
                      "bundle” writes the STL, the sections and the build "
                      "script, and the script builds the .vsp3 on any "
                      "machine that has OpenVSP.", "warn")
            else:
                widgets.hint(
                    f"“Open in OpenVSP” builds the design as a NATIVE "
                    f"OpenVSP wing chain (not an imported mesh), writes this "
                    f"run's own reference quantities and attitude into the "
                    f"model's VSPAERO settings — Sref, bref, cref, the CG, "
                    f"Re and the alpha sweep around the trim point — saves "
                    f"the .vsp3 and launches the app on it. So what opens is "
                    f"the geometry AND the physics it was scored with. "
                    f"({have['app']})")
            widgets.hint(
                "The STL is every surface that was flown — wing, tip device, "
                "tail and the elevator plate — watertight and in metres, so "
                "any viewer will take it. The bundle adds one STL per "
                "surface, both aerofoil .dat files and the OpenVSP build "
                "script.")
            if cad_state["note"]:
                widgets.hint(cad_state["note"], cad_state["level"])
            if cad_state["last"]:
                widgets.readout("last written", cad_state["last"])

            with ui.row().classes("items-center gap-2 flex-wrap"):
                ui.label("or download a single file:").classes("field-unit")
                ui.button("STL", icon="download",
                          on_click=lambda: _cad("stl")) \
                    .props("flat dense no-caps size=sm")
                ui.button("OpenVSP script", icon="download",
                          on_click=lambda: _cad("vsp")) \
                    .props("flat dense no-caps size=sm")
                ui.button("section .dat", icon="download",
                          on_click=lambda: _cad("dat")) \
                    .props("flat dense no-caps size=sm")

    def _set_export_dir(value):
        session.set_export_dir(S, value)
        _export_answer_moved()

    def _set_export_stem(value):
        S.setdefault("export", {})["stem"] = str(value or "")
        _export_answer_moved()

    def _export_answer_moved():
        """Where and under what name — re-derive everything that quotes it.

        NOT the whole card: the field being typed into lives in it, and
        rebuilding that under the cursor is the focus-loss bug
        (a_typed_number_must_not_rebuild_its_field). Only the derived half,
        which is the half that names the files.

        The folder is the SESSION's, not this stage's — stage 2 quotes it on
        its own export card (one question, one place), so any view that
        shows a section is owed a repaint too. ``render_when_shown`` defers
        the ones that are not on screen, so this costs one container.
        """
        _render_cad_derived()
        for st, vw in list(ctx.renderers):
            if vw == "section":
                ctx.render_when_shown(st, vw)

    def _payload():
        """The design to export, or None with a message already logged."""
        data = session.export_payload(S)
        if data is None:
            ctx.log("no geometry re-evaluated yet — open the geometry view "
                    "and let it finish", "warn")
        return data

    def _say(note: str, level: str = "", last: str | None = None):
        """From the UI thread: publish and redraw the card now."""
        _publish_cad(note=note, level=level, last=last)
        _render_cad_derived()

    def _publish_cad(**fields):
        """From ANY thread: publish, and let the heartbeat redraw."""
        for key, value in fields.items():
            if key == "last" and value is None:
                continue
            cad_state[key] = value
        cad_state["version"] += 1

    def _save_bundle():
        """Every file, into the chosen folder — nothing built, nothing run."""
        from aerobo import cad

        data = _payload()
        if data is None:
            return
        try:
            d = session.export_dir(S)
            written = cad.export(data["geom"], d,
                                 stem=session.export_stem(S),
                                 x_best=data["x"], labels=data["labels"],
                                 section=data["section"],
                                 section_aft=data["section_aft"])
        except Exception as exc:      # noqa: BLE001 — a view, never fatal
            _say(f"export failed: {type(exc).__name__}: {exc}", "bad")
            ctx.log(f"CAD export failed: {type(exc).__name__}: {exc}", "error")
            return
        ctx.log(f"wrote {len(written)} files to {d}", "ok")
        _say(f"{len(written)} files written.", "ok", last=str(d))

    def _save_one(kind: str):
        """One file, into the chosen folder, under the chosen name.

        Every branch names its file through ``cad.export_name``, so a single
        file and the same file inside the bundle cannot be called different
        things. An unknown kind is refused rather than falling through: the
        ``else`` this replaces wrote the aerofoil .dat for anything that was
        not "stl", so ``save_cad("vsp")`` silently produced a .dat.
        """
        from aerobo import cad

        data = _payload()
        if data is None:
            return
        d = session.export_dir(S)
        stem = session.export_stem(S)
        try:
            path = d / cad.export_name(stem, kind)   # refuses first
            d.mkdir(parents=True, exist_ok=True)
            if kind == "stl":
                surfs = cad.surfaces(data["geom"], data["x"], data["labels"],
                                     data["section"], data["section_aft"])
                path = cad.write_stl(path, surfs, binary=True, name=stem)
            elif kind == "vsp":
                path.write_text(cad.vsp_script(data["geom"], data["x"],
                                               data["labels"], stem=stem))
            elif kind == "dat":
                xc, zc = cad.section_path(data["geom"], data["x"],
                                          data["labels"], data["section"],
                                          n=61)
                path.write_text(cad.airfoil_dat(xc, zc, f"{stem} wing"))
            else:
                raise ValueError(
                    f"{kind!r} is not a file this card writes on its own — "
                    f"“Save the whole bundle” writes it")
        except Exception as exc:      # noqa: BLE001 — a view, never fatal
            _say(f"export failed: {type(exc).__name__}: {exc}", "bad")
            ctx.log(f"CAD export failed: {type(exc).__name__}: {exc}", "error")
            return
        ctx.log(f"wrote {path}", "ok")
        _say("written.", "ok", last=str(path))

    def _open_in_vsp():
        """Build the .vsp3 — geometry AND physics — and launch OpenVSP on it.

        The build runs OpenVSP's own interpreter in a subprocess (seconds),
        so it goes on a worker thread: this handler is serving a click in a
        web shell and may not block it.
        """
        data = _payload()
        if data is None or cad_state["busy"]:
            return
        cad_state["busy"] = True
        _say("building the OpenVSP model…", "")
        ctx.log("building the OpenVSP model (geometry + VSPAERO setup)…",
                "info")
        threading.Thread(target=_vsp_worker, args=(data,),
                         daemon=True).start()

    def _vsp_worker(data: dict):
        from aerobo import vsp as vspmod

        try:
            out = vspmod.build_model(
                data["geom"], session.export_dir(S),
                stem=session.export_stem(S), x_best=data["x"],
                labels=data["labels"], section=data["section"],
                section_aft=data["section_aft"],
                breakdown=data["breakdown"])
            if out.get("error"):
                _publish_cad(busy=False, note=out["error"], level="bad")
                ctx.log(f"OpenVSP: {out['error']}", "error")
                return
            launched = vspmod.open_in_app(out["vsp3"])
            if launched.get("error"):
                _publish_cad(
                    busy=False, level="warn", last=out["vsp3"],
                    note=f"the model was built ({out['vsp3']}) but the app "
                         f"would not start: {launched['error']}")
                ctx.log(f"OpenVSP: {launched['error']}", "warn")
                return
            _publish_cad(
                busy=False, level="ok", last=out["vsp3"],
                note=f"{out.get('vsp_version', 'OpenVSP')} is opening this "
                     f"design, with its VSPAERO reference quantities and "
                     f"trim attitude already set.")
            ctx.log(f"OpenVSP opened {out['vsp3']}", "ok")
        except Exception as exc:      # noqa: BLE001 — a view, never fatal
            _publish_cad(busy=False, level="bad",
                         note=f"{type(exc).__name__}: {exc}")
            ctx.log(f"OpenVSP: {type(exc).__name__}: {exc}", "error")

    def _cad(kind: str):
        """STL / OpenVSP script / section as a browser DOWNLOAD.

        Built from the SAME geometry report the 3-D view draws, so a file that
        leaves here cannot disagree with what the result page shows.
        """
        from aerobo import cad

        data = _payload()
        if data is None:
            return
        stem = session.export_stem(S)
        try:
            if kind == "stl":
                surfs = cad.surfaces(data["geom"], data["x"], data["labels"],
                                     data["section"], data["section_aft"])
                names = ", ".join(s.name for s in surfs)
                ui.download.content(cad.stl_bytes(surfs),
                                    cad.export_name(stem, "stl"))
                ctx.log(f"exported {cad.export_name(stem, 'stl')} "
                        f"({names})", "ok")
            elif kind == "vsp":
                ui.download.content(
                    cad.vsp_script(data["geom"], data["x"], data["labels"],
                                   stem=stem),
                    cad.export_name(stem, "vsp"))
                ctx.log(f"exported {cad.export_name(stem, 'vsp')} — run "
                        f"it beside the section .dat to build the .vsp3",
                        "ok")
            else:
                xc, zc = cad.section_path(data["geom"], data["x"],
                                          data["labels"], data["section"],
                                          n=61)
                ui.download.content(cad.airfoil_dat(xc, zc, f"{stem} wing"),
                                    cad.export_name(stem, "dat"))
                ctx.log(f"exported {cad.export_name(stem, 'dat')}", "ok")
        except Exception as exc:      # noqa: BLE001 — a view, never fatal
            ctx.log(f"CAD export failed: {type(exc).__name__}: {exc}",
                    "error")

    # ------------------------------------------------------------ wiring
    def _render_all():
        """Redraw the four views — the ones on screen NOW, the rest when
        they are opened.

        All four used to be drawn every time a run landed and again when the
        background re-evaluation published, and three of them were invisible
        every time: the Geometry view alone is ~250 KB of elements down the
        socket, and stage 3 is still the stage on screen when a run finishes.
        ``Ctx.render_when_shown`` owes the paint instead, and ``app.select``
        pays it the moment the view is opened.
        """
        for view in ("summary", "geometry", "loading", "log"):
            ctx.render_when_shown("results", view)

    ctx.on_render("results", "summary", _render_summary)
    ctx.on_render("results", "geometry", _render_geometry)
    ctx.on_render("results", "loading", _render_loading)
    ctx.on_render("results", "log", _render_log)
    ctx.register("set_result", set_result)
    ctx.register("refresh_results", _render_all)
    # the CAD card's own actions, registered so a preset, another stage or a
    # test drives the REAL handlers rather than writing the export dict
    ctx.register("set_export_dir", _set_export_dir)
    ctx.register("set_export_stem", _set_export_stem)
    ctx.register("save_cad_bundle", _save_bundle)
    ctx.register("save_cad", _save_one)
    ctx.register("open_in_vsp", _open_in_vsp)
