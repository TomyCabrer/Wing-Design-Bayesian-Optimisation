"""The box that WOULD have held a design, and how far it is from yours.

A run that comes back with nothing has already been given six sentences by
this shell (``RESULTS_NO_SOLUTION.md``): which limit stopped it, whether the
box was provably empty before it started, which row is the limiting factor,
and the design that came closest. Every one of them ends by handing the user
a control and asking them to move it themselves.

This module answers the next question instead of asking it back: *given
everything this run measured, which rows would have to move, by how little,
and what design is out there?* It plans the move, it does not make it — the
user's design box is not written by anything in here, and the run the plan
describes is searched on a COPY of the problem (the same discipline
:mod:`gui.foiling.reach` keeps for a widened row).

Three rules the plan obeys, and each of them is a failure this project has
already had:

* **Only rows the evidence implicates move.** A uniform dilation of every row
  is a different search, not a nearer one — and on most families most rows do
  not reach the physics at all (``api.rows_outside_validity``), so a uniform
  reach buys refusals and calls them a wider box.
* **One press is one search, at the budget the user already paid.** Climbing a
  ladder of boxes until something turns up is the thing the whole area exists
  to refuse: it answers a question nobody asked, at a price nobody quoted.
* **Nothing is discovered by refusal.** Every band this module proposes is
  BUILT before it is offered — the ceiling clip, the positivity clamp and the
  validity re-check all happen in ``plan``, for free, so a row the family will
  not honour is named on the card instead of eating the budget.

Pure apart from :mod:`aerobo.api` and this package's own config: no NiceGUI,
so every number and every sentence is assertable without a browser.
"""

from __future__ import annotations

from gui import diagnose

#: how far inside its new bound the adopted design is put, as a fraction of
#: the new row's width. The same number ``box_moves`` calls "riding a bound"
#: (``diagnose.AT_BOUND_FRAC``) — a box adopted so tightly that the design it
#: was built for is still on its edge would be recommended for widening by
#: the very card that produced it.
MARGIN = diagnose.AT_BOUND_FRAC

#: a strictly-positive row never crosses zero on the way out. Measured: a
#: symmetric move on ``b_m`` [6, 40] raises ``ValueError: span bounds must
#: satisfy 0 < min < max`` out of ``sizing.span_bounds`` at BUILD time, before
#: any search — a wing with a negative span is not a wider search, it is a
#: crash.
MIN_FLOOR_FRAC = 0.10

#: press 1 reaches by what the evidence asks; press 2 reaches 2.5x further and
#: says so with its own price. There is no press 3 in this shell: past that
#: the answer stops being "closest to your box" and becomes a different
#: question, which is the user's to ask on a control they can see.
STEPS = (1.0, 2.5)

#: the fields of the run's own config that have to still be true for the plan
#: to be about the run the user is looking at.
#:
#: ``flags`` is deliberately NOT one of them, and is compared through
#: :func:`config.physics_flags` in :func:`is_stale` instead. V3.5 folds the
#: SEARCH POLICY (``acqf``, ``bo_n_init``, ``bo_refusal``, ``bo_feasibility``)
#: into the same dict as the physics, so a raw comparison made the stage-1
#: search radio destroy a finished, paid-for reach: flip "recommended" to
#: "own" and ``bo_refusal``/``acqf``/``bo_n_init`` disappear while the design
#: box, the mission, the pins and every physics flag are bit-identical. That
#: is the same reasoning ``_reached_cfg`` already applies to budget, seed and
#: optimiser — they do not change the box — and the reached run never sees the
#: live search flags anyway, because it is searched on the RECORD's config.
_SAME = ("problem_name", "mission_kwargs", "bounds_overrides", "pinned")


# ------------------------------------------------------------------ helpers

def _band_of(bounds, labels) -> dict:
    """``{label: [lo, hi]}`` for a built problem's full box."""
    return {str(name): [float(bounds[k][0]), float(bounds[k][1])]
            for k, name in enumerate(labels)}


def is_stale(S: dict, rd: dict) -> str | None:
    """Why this record is not about the session on screen, or None.

    Relaxing a box the run did not search answers a question about neither
    box. Free, and it is checked first: the record carries the whole config
    it ran (``RunResult.to_dict()["config"]``), so this is a comparison and
    not a guess.
    """
    from . import config

    was = (rd or {}).get("config")
    if not isinstance(was, dict):
        return None                      # an old record says nothing; allow it
    now = config.cfg_dict(S)
    for key in _SAME:
        if was.get(key) != now.get(key):
            return (f"the {key.replace('_', ' ')} has changed since this run, "
                    f"so a box reached from it would not be a box this run "
                    f"ever searched. Launch again first.")
    if config.physics_flags(was) != config.physics_flags(now):
        return ("the flags has changed since this run, so a box reached from "
                "it would not be a box this run ever searched. Launch again "
                "first.")
    return None


def clip_to_validity(spec, mission_kwargs: dict, flags: dict,
                     band: dict) -> tuple[dict, list, str | None]:
    """``(band, capped, error)`` — the band with every row the family does not
    validate put back where it was.

    This is the authority on what a relaxation can reach, and it is a
    MEASUREMENT rather than a list: a widened row is written into the box the
    sampler draws from, and only some rows travel on into the problem's own
    bounds. Measured over the registry, 17 % of rows do. A row that does not
    is drawn from and then refused, so offering it as a widening would be
    selling the user a search that spends its budget being told no.

    ``api.rows_outside_validity`` on the BUILT problem is what says which is
    which, so a family that starts honouring a row it did not honour before
    is picked up without anything here being edited. One rebuild is enough
    (measured on ``tail``), and the result is asserted rather than assumed.
    """
    from aerobo import api

    capped: list = []
    try:
        built = spec.build(mission_kwargs, flags, band or None)
    except Exception as exc:      # noqa: BLE001 — the plan reports it verbatim
        return band, capped, f"{type(exc).__name__}: {exc}"
    for row in api.rows_outside_validity(built):
        label = str(row["label"])
        vlo, vhi = float(row["validated"][0]), float(row["validated"][1])
        slo, shi = float(row["searched"][0]), float(row["searched"][1])
        band[label] = [max(slo, vlo), min(shi, vhi)]
        capped.append({"label": label, "searched": [slo, shi],
                       "validated": [vlo, vhi]})
    if not capped:
        return band, capped, None
    try:
        built = spec.build(mission_kwargs, flags, band or None)
    except Exception as exc:      # noqa: BLE001
        return band, capped, f"{type(exc).__name__}: {exc}"
    left = api.rows_outside_validity(built)
    if left:
        return band, capped, ("this family re-derives its own bounds from the "
                              "ones it is handed, so the widened rows cannot "
                              "be clipped back in one pass: "
                              + ", ".join(str(r["label"]) for r in left))
    return band, capped, None


def minimal_box(bounds, labels, x, margin: float = MARGIN) -> dict:
    """``{label: [lo, hi]}`` — the SMALLEST edit to ``bounds`` that contains
    ``x``, with the design a hair inside rather than on its own new bound.

    Only the rows ``x`` is actually outside appear: a row that already
    contained it is absent, so adopting this dict moves exactly the rows the
    answer needed and not the rows the search was given.

    The ``/(1 - margin)`` is exact, not a fudge. The position of ``x`` is
    measured against the NEW width, so adding ``margin x width`` leaves the
    design at 0.982 of a row it was meant to sit at 0.98 of — close enough to
    keep ``box_moves`` recommending that the user widen the row again.
    """
    out: dict = {}
    d = diagnose.outside_box(bounds, labels, x)
    m = min(max(float(margin), 0.0), 0.5)
    for row in d["rows"]:
        lo, hi, v = row["lo"], row["hi"], row["value"]
        if row["end"] == "top":
            out[row["label"]] = [lo, lo + (v - lo) / (1.0 - m)]
        else:
            out[row["label"]] = [hi - (hi - v) / (1.0 - m), hi]
    return out


def banner(S: dict, rd: dict | None) -> str | None:
    """"This run searched a different box from the one on the Design box tab",
    or None — DERIVED, never flagged.

    A comparison of the record's own overrides against the session's, so it is
    true whenever the two disagree: after a reach, after a user edits a row
    and does not re-launch, after loading a session. A boolean set by whoever
    remembered to set it would go stale the first time somebody forgot.
    """
    from . import config

    was = (rd or {}).get("config")
    if not isinstance(was, dict):
        return None
    old = dict(was.get("bounds_overrides") or {})
    new = dict(config.bounds_overrides(S) or {})
    # ...and where a row was not overridden, the band the record ACTUALLY
    # searched, which it carries. "the family's own band" is true and useless
    # beside a number: the whole point of the line is the two bands, and a
    # user cannot compare one of them against a phrase.
    searched = {}
    for k, name in enumerate(was_labels := list(rd.get("param_labels") or ())):
        row = (rd.get("bounds") or [])
        if k < len(row):
            searched[str(name)] = [float(row[k][0]), float(row[k][1])]
    del was_labels
    diff = []
    for label in sorted(set(old) | set(new)):
        a, b = old.get(label), new.get(label)
        if a == b:
            continue
        a = a or searched.get(label)
        diff.append(f"{label} " + (f"{a[0]:.4g} – {a[1]:.4g}" if a
                                   else "the family's own band")
                    + " → " + (f"{b[0]:.4g} – {b[1]:.4g}" if b
                               else "the family's own band"))
    if not diff:
        return None
    return ("This run searched a different box from the one on the Design "
            "box tab: " + "; ".join(diff) + ". Those rows are not what the "
            "numbers above were measured over.")


# ------------------------------------------------------------------- the plan

def plan(S: dict, rd: dict, *, step: int = 0) -> dict:     # noqa: PLR0912,PLR0915
    """What box to reach for, why, and what it costs — nothing is run here.

    ``verdict`` is one of

    ``"ready"``           a box to search, with ``moves`` and ``overrides``;
    ``"blocked"``         a reason it will not be searched, in the mechanism's
                          own words (a build that raises, a row that would
                          have to move further than its own width);
    ``"no-evidence"``     nothing in this record says which row to move — the
                          honest answer on a box where every draw was refused
                          and the two cheap gates do not rule it out;
    ``"nothing-can-grow"``  every row this run searched is one the family
                          validates only over its published band, so a wider
                          box would be drawn from and refused;
    ``"stale"``           the record is not about the session on screen.
    """
    from aerobo import api

    from . import config

    out: dict = {"verdict": "no-evidence", "reason": "", "moves": [],
                 "blocked": [], "overrides": None, "checks": [],
                 "step": int(step), "scale": STEPS[min(int(step),
                                                       len(STEPS) - 1)],
                 "bounds": None, "labels": [], "budget": None, "seed": None,
                 "wall": (rd or {}).get("wall_time_s")}

    stale = is_stale(S, rd)
    if stale:
        return dict(out, verdict="stale", reason=stale)

    cfg = config.build_cfg(S)
    spec = api.PROBLEM_SPECS[cfg.problem_name]
    try:
        built0 = spec.build(cfg.mission_kwargs, cfg.flags,
                            cfg.bounds_overrides)
    except Exception as exc:      # noqa: BLE001
        return dict(out, verdict="blocked",
                    reason=f"this problem does not build: "
                           f"{type(exc).__name__}: {exc}")
    labels = [str(v) for v in built0.param_labels]
    base = _band_of(built0.bounds, labels)
    out["bounds"] = [list(base[name]) for name in labels]
    out["labels"] = labels
    out["budget"] = int(cfg.budget)
    out["seed"] = int(cfg.seed)

    pinned = set(dict(rd.get("pinned") or {}))
    # ...and the rows this session has RELEASED. ``config.bounds_overrides``
    # drops them, so a band written for one never reaches the run: the reach
    # would plan it, price it, spend the whole budget, and then adopt would
    # silently no-op and the seed be refused for sitting outside a box that
    # never moved. Free to check, so it is checked before anything is spent.
    released = set(config.released_rows(S))
    scale = out["scale"]
    moves: dict = {}
    #: rows a source wanted to move and could not, because the user pinned
    #: them. Filed once each: two gates can name the same row, and the
    #: measurement can name it a third time.
    held: set = set()

    def _hold(name: str, why: str) -> None:
        """A row the plan HAS a move for, held fixed by the user.

        Dropping it silently is the reported bug: the plan then fell through
        to "there is no ROW to move for that — it is answered at stage 1",
        which is a sentence about a gate that names no row at all, and is
        simply false about a pinned ``S_m2`` the gate named and banded. The
        row is in the design box, it is the row the proof is about, and the
        one thing standing between the user and a reach is a pin only they
        can release. ``gui.diagnose`` has always worded this correctly for
        its own report; this is the same sentence in the reach.
        """
        if str(name) in held:
            return
        held.add(str(name))
        val = dict(rd.get("pinned") or {}).get(str(name))
        at = f" at {float(val):.4g}" if isinstance(val, (int, float)) else ""
        out["blocked"].append({
            "label": str(name), "validated": None,
            "searched": ([float(base[str(name)][0]),
                          float(base[str(name)][1])]
                         if str(name) in base else None),
            "why": (f"{name} is held FIXED{at} in your design box, so it is "
                    f"not searched at all and a band written for it would "
                    f"change nothing — and {why} Unpin it: it cannot move at "
                    f"all while it is held fixed.")})

    # ---- source 1: the PROOF. Closed form, ~0.2 ms, evaluates nothing, and
    # on the box a mission empties it is the whole answer — there are no
    # margins to correlate when every draw was refused before its solver.
    try:
        found = api.size_box_conflicts(cfg)
    except Exception:      # noqa: BLE001 — a plan, never fatal
        found = []
    gate_texts = [f["text"] for f in found if f.get("empty")]
    for f in found:
        row, band = f.get("row"), f.get("suggest")
        if not (f.get("empty") and row and band):
            continue
        if str(row) in pinned:
            _hold(row, "the closed-form gate proves this box empty on that "
                       "very row, and names the band that would answer it.")
            continue
        # ``size_box_conflicts`` sorts worst-first and several gates can name
        # the same row, so the FIRST proof about a row is the binding one. A
        # plain assignment let the last one overwrite it, which is the weaker
        # band by the sort's own order.
        if str(row) in moves:
            continue
        lo, hi = float(band[0]), float(band[1])
        was_lo, was_hi = base.get(str(row), [lo, hi])
        # the gate's OWN band, verbatim — floor AND ceiling.
        #
        # The obvious improvement here is to keep the gate's floor and shrink
        # its ceiling to the width the user actually drew, so the reached box
        # is the nearest one rather than a band 31x wider than theirs. It was
        # built, and then measured, and it is wrong: on the reported mission
        # that gives S_m2 93.08 – 107.08 m², and ``box_refusal_probe`` refuses
        # 192 of 192 draws from it (125 still above the wing-loading ceiling),
        # while the gate's own 93.08 – 533.3 returns 28 admissible of 192.
        #
        # The reason is the one the api's own comment gives: ``S >= W/cap``
        # cannot see the WING'S weight, and the sizing loop adds it, so the
        # floor is a floor on a lighter aircraft than the one that gets built.
        # The headroom that fixes it lives in the band's ceiling. A narrowing
        # that throws it away is not a nearer box, it is an empty one — and
        # ``size_box_conflicts`` cannot catch it, because it proves emptiness
        # and never feasibility.
        #
        # "Nearest" is therefore delivered where it can be delivered honestly:
        # by RE-RANKING what the run evaluated (``closest_feasible``) and by
        # adopting only the minimal box around the answer (``minimal_box``),
        # not by narrowing the box the search is given.
        moves[str(row)] = {
            "label": str(row), "lo": float(was_lo), "hi": float(was_hi),
            "new_lo": lo, "new_hi": hi, "source": "gate",
            "why": f["text"], "basis": "proof", "r2": None, "corr": None,
            "capped": False}

    # ---- source 2: the MEASUREMENT. Which row correlates with the binding
    # margin AND is being ridden, from diagnose; how far, from the fitted
    # slope of that same margin.
    rep = diagnose.infeasibility_report(
        rd, constraint_labels=tuple(spec.constraint_labels or ()))
    binding = (rep or {}).get("binding")
    if binding is not None:
        for m in (rep.get("moves") or []):
            name = str(m.get("variable"))
            if name in moves:
                continue
            if m.get("pinned") or name in pinned:
                # the measurement found this row riding the binding limit —
                # and the user has held it. Same statement as the gate's:
                # named, not dropped.
                if name in pinned:
                    _hold(name, "this run's own points say it is what the "
                                "binding limit moves with.")
                continue
            if name not in base:
                continue
            mag = diagnose.move_magnitude(rd, int(binding["index"]), m)
            if mag is None:
                continue
            lo, hi = base[name]
            delta = float(mag["delta"]) * scale
            if m.get("at") == "upper":
                new_lo, new_hi = lo, hi + delta
            elif m.get("at") == "lower":
                new_lo = lo - delta
                if lo > 0.0:
                    # a strictly-positive row is stated multiplicatively on the
                    # way down, or it walks through zero and the build raises
                    new_lo = max(new_lo, MIN_FLOOR_FRAC * lo)
                new_hi = hi
            else:
                continue
            moves[name] = {
                "label": name, "lo": lo, "hi": hi,
                "new_lo": float(new_lo), "new_hi": float(new_hi),
                "source": "measured", "basis": mag["basis"],
                "r2": mag["r2"], "corr": m.get("corr"),
                "capped": bool(mag["capped"]),
                "deficit": mag["deficit"], "slope": mag["slope"],
                "at": m.get("at"), "constraint": binding.get("label"),
                "why": m.get("direction")}

    if not moves:
        reason = ("Nothing in this run says which row to move. No design-box "
                  "row is riding the binding limit, and the two closed-form "
                  "gates do not rule this box out either — so a reach would "
                  "be a guess, and a guess that silently widens your search. "
                  "Measure the design box below: which gate is refusing these "
                  "draws is the missing input.")
        # THE STAGE-1 SENTENCE BELONGS TO A GATE THAT NAMES NO ROW. Said
        # about a gate that named one — and it always was, whenever that row
        # happened to be pinned — it is false twice over: the row exists, it
        # is in the design box, and what is stopping the reach is a pin the
        # user can lift here. The blocked list carries that sentence and
        # ``verdict`` prints it under this one.
        rowless = [f["text"] for f in found
                   if f.get("empty") and not f.get("row")]
        if rowless:
            reason = (rowless[0] + " There is no ROW to move for that — it "
                      "is answered at stage 1, not in the design box.")
        elif held:
            head = (gate_texts[0] if gate_texts
                    else "This run has evidence for which row to move.")
            reason = (head + " Every row this reach could move is one you "
                      "have held fixed, so there is nothing to widen until a "
                      "pin is released:")
        elif gate_texts:
            reason = (gate_texts[0] + " There is no ROW to move for that — it "
                      "is answered at stage 1, not in the design box.")
        return dict(out, verdict="no-evidence", reason=reason,
                    blocked=out["blocked"], gate_texts=gate_texts)

    # ---- the clips, in order. Nothing below is allowed to be discovered by
    # a refusal during the run.
    band = dict(cfg.bounds_overrides or {})
    cap = cfg.flags.get("wing_loading_limit_pa")
    for name, m in moves.items():
        lo, hi = float(m["new_lo"]), float(m["new_hi"])
        if name == "ws_pa" and cap is not None:
            # pre-empt the api's own loud raise: a ws_pa band above the
            # mission's ceiling is refused at BUILD time, on purpose
            if hi > float(cap):
                m["ceiling"] = float(cap)
            hi = min(hi, float(cap))
        if m["lo"] > 0.0:
            lo = max(lo, MIN_FLOOR_FRAC * m["lo"])
        if not (hi > lo):
            m["dropped"] = "the clips left this row with no width"
            continue
        m["new_lo"], m["new_hi"] = lo, hi
        band[name] = [lo, hi]

    for name in sorted(released & set(moves)):
        moves.pop(name, None)
        band.pop(name, None)
        out["blocked"].append({
            "label": name, "validated": None, "searched": None,
            "why": (f"{name} is switched OFF in the design box, so nothing "
                    f"this session says constrains it and the solver's own "
                    f"band stands. A band written for a released row never "
                    f"reaches the run — switch the row back on if you want it "
                    f"reached.")})

    band, capped, err = clip_to_validity(spec, cfg.mission_kwargs, cfg.flags,
                                         band)
    if err:
        return dict(out, verdict="blocked", moves=list(moves.values()),
                    reason=err)
    for row in capped:
        name = row["label"]
        m = moves.pop(name, None)
        vlo, vhi = row["validated"]
        out["blocked"].append({
            "label": name, "validated": [vlo, vhi],
            "searched": row["searched"],
            "why": (f"{name} cannot be reached past {vlo:.4g} – {vhi:.4g}. A "
                    f"widened row is written into a COPY of the box the "
                    f"sampler draws from; this family re-checks every "
                    f"candidate against its own band, so a draw outside that "
                    f"comes back refused rather than flown. Widening this row "
                    f"does not reach the physics, so it is not offered.")})

    # ...and a move the clips reduced to the row it started from. Left in, it
    # printed "S 37.6 - 75.2 -> 37.6 - 75.2. MEASURED on this run's own
    # points", counted itself in "1 row(s) moved", and bought the user a
    # bit-identical, deterministic re-run of the search that had just failed —
    # whose "nothing out there either" the card then read as a fact about the
    # physics. Measured on ``wing_loading_free``, where the mission's ceiling
    # IS the ws_pa row's top by construction, so an upper move is always
    # clipped back to nothing.
    for name in [k for k, m in moves.items()
                 if abs(m["new_lo"] - m["lo"]) <= 1e-12
                 and abs(m["new_hi"] - m["hi"]) <= 1e-12]:
        m = moves.pop(name)
        out["blocked"].append({
            "label": name, "validated": None, "searched": [m["lo"], m["hi"]],
            "why": ((f"{name} is already at this mission's own "
                     f"{m['ceiling']:.4g} Pa ceiling, so there is nothing "
                     f"above it to reach for. Whether that ceiling is yours "
                     f"is asked at stage 1, not in the design box.")
                    if m.get("ceiling") is not None else
                    (f"{name} came back from the clips exactly where it "
                     f"started, so reaching would search the box this run "
                     f"already searched."))})
        band.pop(name, None)

    if not moves:
        return dict(out, verdict="nothing-can-grow", blocked=out["blocked"],
                    reason=("Nothing here can be reached. Every row this run "
                            "searched is one this family validates only over "
                            "its published band, and a draw outside that "
                            "comes back refused rather than flown. Nothing "
                            "was run, and nothing was spent."))

    # ---- the four free pre-flight checks
    checks: list = []
    # ...and it is checked on the EXTRAPOLATED moves only. A fitted slope run
    # far enough out is no longer a measurement, so a measured move stops at
    # one row-width. A PROVED one does not: "at 7000 N this mission allows at
    # most 75.2 Pa, so the wing is at least 93 m²" is a requirement, and
    # refusing it for being 37 widths away would be the shrug this whole card
    # exists to replace — the box really is that far from an answer.
    # scaled, because that is what pressing "reach further" MEANS: step 2 is
    # allowed to reach 2.5 widths. Comparing a move already multiplied by the
    # scale against the unscaled limit blocked every fit-basis move of 0.4
    # widths or more at step 2 — the second press could only ever refuse.
    reach_limit = diagnose.MAX_REACH * scale
    over = [m for m in moves.values()
            if m.get("source") != "gate"
            and (m.get("capped")
                 or max(abs(m["new_hi"] - m["hi"]),
                        abs(m["lo"] - m["new_lo"]))
                 > reach_limit * max(1e-12, m["hi"] - m["lo"]))]
    if over:
        m = over[0]
        # the fitted fix is read off ``move_magnitude``, which CAPS what it
        # returns — so the flag, not the returned distance, is what says the
        # line put the answer further out than one row. And the number PRINTED
        # is the one this press would actually have reached (scale included),
        # against the limit this press is actually held to: a refusal whose own
        # quoted distance sits below the threshold it cites is not a reason.
        need = (m.get("deficit", 0.0) / abs(m.get("slope") or 1.0)
                * (1.0 + diagnose.HEADROOM) * scale
                / max(1e-12, m["hi"] - m["lo"]))
        return dict(out, verdict="blocked", moves=list(moves.values()),
                    blocked=out["blocked"],
                    reason=(f"{m['label']} would have to move {need:.2g} of "
                            f"its own width to clear "
                            f"{m.get('constraint') or 'that limit'}. Past "
                            f"{reach_limit:g} width"
                            f"{'s' if reach_limit != 1 else ''} the answer "
                            f"stops being the box nearest yours, so this is "
                            f"yours to decide on the design box rather than "
                            f"mine to reach for."))
    checks.append(("no EXTRAPOLATED row moves by more than its own "
                   "width", True))

    # ...and the reached box must not be one this mission STILL empties.
    # Free, and necessary rather than sufficient: it proves emptiness and
    # never feasibility, so it is a refusal to launch and not a promise.
    def _empty_under(candidate: dict):
        try:
            relaxed = api.RunConfig(**dict(config.cfg_dict(S),
                                           bounds_overrides=candidate))
        except Exception as exc:      # noqa: BLE001
            return None, f"{type(exc).__name__}: {exc}"
        try:
            return [f for f in api.size_box_conflicts(relaxed)
                    if f.get("empty")], None
        except Exception as exc:      # noqa: BLE001
            return None, f"{type(exc).__name__}: {exc}"

    left, err2 = _empty_under(band)
    if err2:
        return dict(out, verdict="blocked", moves=list(moves.values()),
                    blocked=out["blocked"],
                    reason=f"the reached box does not build: {err2}")
    if left:
        return dict(out, verdict="blocked", moves=list(moves.values()),
                    blocked=out["blocked"],
                    reason=("the reached box is still one this mission "
                            "empties: " + left[0]["text"]))
    checks.append(("the reached box is not one this mission empties", True))
    checks.append(("every reached row is one this family flies", True))

    out.update(verdict="ready", moves=list(moves.values()),
               blocked=out["blocked"], overrides=band, checks=checks,
               reason="", gate_texts=gate_texts,
               cfg_dict=_reached_cfg(S, rd, band))
    out["budget"] = int(out["cfg_dict"]["budget"])
    out["seed"] = int(out["cfg_dict"]["seed"])
    out["optimiser"] = str(out["cfg_dict"]["optimiser"])
    return out


def _reached_cfg(S: dict, rd: dict, band: dict) -> dict:
    """The failed run's OWN search, in the reached box — and nothing else.

    Two things this fixes, both of which made the card's promise false.

    The budget, the seed and the optimiser come off the RECORD, not off the
    live session. ``is_stale`` deliberately does not compare them (they do not
    change the box), so a user who edits the budget on the Solver tab and does
    not relaunch left the card quoting the new number as "this run's own" —
    and the reach then really did spend it. "One search, at the budget you
    already paid" has to be true of the search, not only of the sentence.

    And ``x_seed`` is STRIPPED. A start design is a statement about the box it
    was armed in; the reach searches a different one, so a seed armed by the
    "start again from the closest design" button on this very card would be
    outside the reached box and ``api.run`` would refuse it — after the press,
    which is exactly the class of failure this module refuses to have.
    """
    from . import config

    was = rd.get("config")
    base = dict(was) if isinstance(was, dict) else config.cfg_dict(S)
    base.pop("x_seed", None)
    base["bounds_overrides"] = band
    return base


# ------------------------------------------------------------------ the prose

def verdict(state: dict) -> list[tuple[str, str]]:      # noqa: PLR0912
    """The reach, as sentences with a severity — the same contract
    :func:`gui.diagnose.infeasibility_lines` keeps, so the card renders the
    list and decides nothing, and every word is assertable without a browser.

    ``state`` is ``{"plan": ..., "record": ..., "answer": ..., "error": ...}``
    as the shell stores it.
    """
    p = state.get("plan") or {}
    lines: list[tuple[str, str]] = []
    if state.get("error"):
        return [(f"the reach could not be measured: {state['error']}", "warn")]

    v = p.get("verdict")
    if v in ("stale", "no-evidence", "nothing-can-grow", "blocked"):
        lines.append((p.get("reason") or "", "warn"))
        for b in (p.get("blocked") or []):
            lines.append((f"    · {b['why']}", ""))
        return [ln for ln in lines if ln[0]]

    for m in (p.get("moves") or []):
        head = (f"    · {m['label']} — {m['lo']:.4g} – {m['hi']:.4g} → "
                f"{m['new_lo']:.4g} – {m['new_hi']:.4g}")
        if m.get("source") == "gate":
            lines.append((f"{head}. PROVED: {m['why']} The band offered is "
                          f"wider than the gate strictly needs, because that "
                          f"number cannot see the wing's own weight and the "
                          f"sizing loop adds it.", ""))
        elif m.get("basis") == "fit":
            r2 = float(m.get("r2") or 0.0)
            lines.append((f"{head}. MEASURED on this run's own points: the "
                          f"{m.get('constraint')} moves with it "
                          f"(r = {float(m.get('corr') or 0.0):+.2f}, "
                          f"r² = {r2:.2f}), the closest "
                          f"design was already sitting on that bound and "
                          f"missed by {float(m.get('deficit') or 0.0):.4g}, "
                          f"and a straight-line fit puts the fix there — plus "
                          f"a quarter for headroom, because a straight line "
                          f"through a curved margin lands short.", ""))
        else:
            r2 = float(m.get("r2") or 0.0)
            lines.append((f"{head}. The direction is measured "
                          f"(r = {float(m.get('corr') or 0.0):+.2f}) and the "
                          f"DISTANCE is not: a line through these points "
                          f"explains only r² = {r2:.2f} "
                          f"of that margin, so this is one fixed step of "
                          f"{diagnose.RELAX_STEP:.0%} of the row, not an "
                          f"extrapolation.", ""))
        if m.get("ceiling") is not None:
            lines.append((f"        (clipped at this mission's own "
                          f"{m['ceiling']:.4g} Pa ceiling — above it the api "
                          f"refuses the band outright, and whether that "
                          f"ceiling is yours is asked at stage 1.)", ""))
    for b in (p.get("blocked") or []):
        lines.append((f"    · {b['why']}", ""))

    ans = state.get("answer")
    if ans is None:
        return lines

    if ans.get("kind") == "found":
        n = len(ans.get("rows") or [])
        if not n:
            lines.append((
                "A design that meets every limit sits INSIDE the box you "
                "already drew — not one row had to move. Your box is not what "
                "stopped this run; the search is. Start the next run from it "
                "and the same box will hold an answer.", "ok"))
        else:
            lines.append((
                f"A design that meets every limit sits {ans['dinf']:.3g} of a "
                f"row-width outside your box. {n} of "
                f"{len(p.get('labels') or [])} rows had to move; the rest "
                f"already contained it.", "ok"))
            for r in ans["rows"]:
                top = r["end"] == "top"
                edge = r["hi"] if top else r["lo"]
                # ...and the distance in the SAME unit as the headline. A
                # row the answer sits 13.9 widths outside reads as "1387 % of
                # that row's own width", which is arithmetically right and
                # unreadable beside "13.9 of a row-width" three lines above.
                far = (f"{r['d']:.3g} times that row's own width" if r["d"] >= 1
                       else f"{r['d']:.0%} of that row's own width")
                lines.append((
                    f"    · {r['label']} — your row "
                    f"{'ends at' if top else 'starts at'} {edge:.4g}, this "
                    f"design is at {r['value']:.4g}. That is {r['over']:.4g} "
                    f"{'past the top of' if top else 'below'} it, {far}.", ""))
        if ans.get("y") is not None:
            lines.append((
                f"Its objective is {float(ans['y']):.5g}, from the same "
                f"physics, the same budget and the same weights as the run "
                f"that found nothing — the only thing different about it is "
                f"the box it was allowed to come from.", ""))
        lines.append((
            "This is the closest design this budget FOUND, not the closest "
            "that exists: the search maximised your objective over the "
            "reached box, and this is the nearest feasible design among the "
            f"{ans.get('n_evals', '?')} it evaluated.", ""))
        return lines

    if ans.get("kind") == "refused":
        rf, own = ans.get("refused_fraction"), ans.get("own_refused_fraction")
        if rf is not None and own is not None and rf >= own - 0.05:
            lines.append((
                f"At the reached box {rf:.0%} of the draws were STILL refused "
                f"before their solver — against {own:.0%} of your own. "
                f"Widening is not the lever here. The refusal is: measure the "
                f"design box to see which gate is eating it.", "warn"))
        else:
            lines.append((
                "Nothing out there met every limit either"
                + (f" ({rf:.0%} of the draws were refused before their "
                   f"solver, against {own:.0%} of your own)"
                   if rf is not None and own is not None else "")
                + ". The limit, not the box, is what has no solution inside "
                  "it — and a wider box cannot fix a limit.", "warn"))
        near = ans.get("nearest")
        if near:
            lines.append((
                f"The closest it came was short of {near['label']} by "
                f"{abs(float(near['violation'])):.4g}.", ""))
        return lines

    if ans.get("kind") == "error":
        lines.append((f"the reached run failed: {ans.get('error')}", "warn"))
    return lines
