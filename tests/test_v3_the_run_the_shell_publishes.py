"""What the wing stage says about the run it just ran — and about which run.

Five defects with one shape: a number on screen that is about a DIFFERENT
search than the one the user is looking at.

* the criteria report of the previous run survived a run that found nothing,
  and its rows suppressed the whole no-solution card;
* "Keep going" ignored the field beside it, sized its stop rule by the
  recommended budget rather than by its own, and never started the live
  sampler;
* the offer on an empty box wrote a band over a row the user had PINNED,
  which the config then widened straight back down to contain the pin — and
  the reach, meeting the same pin, sent the user to stage 1 instead;
* with repeat seeds only an arbitrary member of the queue reached stage 4;
* and a design box that cannot be BUILT drew no card at all, because the
  raise happens before the first gate runs.

Driven through the real handlers and the real views: every one of these was
invisible to a test that called the helper underneath.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np                                        # noqa: E402

from aerobo import api                                    # noqa: E402


def _texts(view):
    return [getattr(e, "text", None) or "" for e in view.descendants()]


def _shell(medium: str = "air"):
    from gui.v3.app import assemble

    ctx = assemble(medium)
    ctx.act("accept_mission")
    return ctx


def _heavy():
    """The reported session: a weight raised, the size rows left behind."""
    from gui.v3 import session

    ctx = _shell()
    ctx.S["mission"]["W_N"] = 7000.0
    session.sync_wing_from_mission(ctx.S)
    assert session.set_planform(ctx.S, "free") == []
    return ctx


def _all_refused(ctx, n: int = 4):
    """A finished run in which every draw came back refused before its
    solver — built through ``api.partial_result``, the call the runner makes,
    so the record is the one the shell really stores."""
    from gui.v3 import config

    cfg = config.build_cfg(ctx.S, seed=0)
    dim = len(api.PROBLEM_SPECS[cfg.problem_name].param_labels)
    rows = [{"x": [0.5] * dim, "f": -100.0, "g": [-1.0], "feasible": False}
            for _ in range(n)]
    out = api.partial_result(cfg, rows).to_dict()
    assert out["best_x"] is None
    return out


def _finished(ctx, record):
    """The state a finished run leaves: the record on the session and a
    terminal job on the manager. The run view draws nothing without a job."""
    from gui import nice_app as v1
    from gui.v3 import config

    ctx.S["run"]["record"] = record
    job = v1.RunJob(cfg=config.build_cfg(ctx.S, seed=0),
                    label="probe · seed 0", budget=40)
    job.status = "done"
    ctx.manager.jobs = [job]
    ctx.manager.version += 1
    return ctx


def _press(button):
    next(iter(button._event_listeners.values())).handler(None)


def _button(view, prefix: str):
    return next(e for e in view.descendants()
                if type(e).__name__ == "Button"
                and (getattr(e, "text", "") or "").startswith(prefix))


# ------------------------------------------------------------------ D2
def _a_report(beats: float = 0.93):
    """A criteria report shaped like the one ``api.score_optimised_design``
    returns — run A's answer, of which only the composite row is needed for
    ``wing_score_rows`` to emit one."""
    return {"baseline": {"composite": 1.0}, "optimised": {"composite": 2.0},
            "delta": {"composite": 1.0}, "beats": beats,
            "objective": "composite", "reference": {}, "criteria": {}}


def test_a_previous_runs_report_cannot_speak_for_a_run_that_found_nothing():
    """The scorer runs only on a record that carries a design, and it is the
    only thing that clears these keys — so run B returning nothing left run
    A's rows standing, both early-return branches took them, and the whole
    no-solution card underneath (the size-conflict proof, the margins, the
    reach, "measure the design box") was unreachable. What the user read was
    "beats 93 % of the box" beside "best objective —"."""
    from gui.v3 import session

    ctx = _heavy()
    rd = _all_refused(ctx)
    _finished(ctx, rd)
    sc = session.wing_score_state(ctx.S)
    sc["report"] = _a_report()
    sc["report_stamp"] = 1.0            # …and no ``report_for``: run A's

    ctx.render("wing", "run")
    texts = _texts(ctx.views[("wing", "run")])
    assert not any("beats" in t for t in texts), (
        "a report about another record reached the screen")
    assert any("could not be FLOWN" in t or "refused before its solver" in t
               for t in texts), texts[-6:]

    # the CONTROL: the same report stamped with THIS record is shown, so the
    # assertion above is about provenance and not about the report's shape
    sc["report_for"] = rd
    ctx.render("wing", "run")
    assert any("beats" in t for t in _texts(ctx.views[("wing", "run")]))


def test_a_scoring_error_from_another_run_does_not_suppress_the_card_either():
    """``report_error`` early-returns identically, so it is the same bug in
    the other branch."""
    from gui.v3 import session

    ctx = _heavy()
    _finished(ctx, _all_refused(ctx))
    sc = session.wing_score_state(ctx.S)
    sc["report_error"] = "ValueError: the run returned no feasible design"
    sc["report_stamp"] = 1.0

    ctx.render("wing", "run")
    texts = _texts(ctx.views[("wing", "run")])
    assert not any("could not be scored on your criteria" in t
                   for t in texts)
    assert any("could not be FLOWN" in t for t in texts), texts[-6:]


# ----------------------------------------------------------------- D11
def _finished_at_the_recommended_budget(ctx):
    """A finished run of exactly the budget the plan recommends — the state
    the "Keep going" card is drawn in."""
    from gui.v3 import config, session

    eff = session.effective_wing_search(ctx.S)
    cfg = config.build_cfg(ctx.S, seed=int(eff["seed"]))
    dim = len(api.PROBLEM_SPECS[cfg.problem_name].param_labels)
    rng = np.random.default_rng(0)
    rows = [{"x": [float(v) for v in rng.uniform(size=dim)],
             "f": 1.0 + i, "g": [0.1], "feasible": True}
            for i in range(int(eff["budget"]))]
    return api.partial_result(cfg, rows).to_dict(), eff


def _captured_launch(ctx):
    """Arm the manager to CAPTURE the queue instead of running it."""
    jobs: list = []
    ctx.manager.start = lambda queue: jobs.extend(queue)
    return jobs


def test_keep_going_runs_the_number_in_the_field_beside_it():
    """The button was wired to ``continue_run(r)`` with no extra, so
    session.py substituted ``continue_extra_default`` and the field bought
    nothing — while the label, after any unrelated repaint, advertised a
    total the button would not run."""
    ctx = _shell()
    rd, eff = _finished_at_the_recommended_budget(ctx)
    _finished(ctx, rd)
    queued = _captured_launch(ctx)

    ctx.S["run"]["continue_extra"] = 200
    ctx.render("wing", "run")
    view = ctx.views[("wing", "run")]
    assert any(t == f"Keep going — {int(eff['budget']) + 200} in total"
               for t in _texts(view)), _texts(view)[-6:]
    _press(_button(view, "Keep going"))
    assert queued, "the continuation never reached the manager"
    assert int(queued[-1].budget) == int(eff["budget"]) + 200
    assert int(queued[-1].cfg.budget) == int(eff["budget"]) + 200


def test_the_continuations_stop_rule_is_sized_by_its_own_budget():
    """``stop_rule_factory(S)`` bare reads the CURRENT plan's budget, so a
    continuation to 43 evaluations carried ``max_evals=29`` with
    ``patience=40`` — plateau detection structurally impossible, because the
    wrapper returns ``fired and rule.n < rule.max_evals``. The one run long
    enough for the adaptive stop to be worth something was the one run it was
    switched off in."""
    from gui.v3 import session

    ctx = _shell()
    st = session.search_state(ctx.S)
    assert st.get("stop_when_converged"), "this session asks for the rule"
    rd, eff = _finished_at_the_recommended_budget(ctx)
    _finished(ctx, rd)
    queued = _captured_launch(ctx)
    ctx.S["run"]["continue_extra"] = 200
    ctx.render("wing", "run")
    _press(_button(ctx.views[("wing", "run")], "Keep going"))

    job = queued[-1]
    stop = job.stop_rule()
    # climb PAST the recommended budget and then flatten, so the rule can
    # only fire on the continuation's own budget: it fires ``patience`` + 1
    # evaluations into the plateau, which is inside this job's budget and
    # outside the recommended one. (Sized from the plan rather than pinned:
    # the recommended budget follows the design vector, so a fixed two-point
    # climb stopped proving anything the day the opening aeroplane grew its
    # tail's own planform rows.)
    climb = int(eff["budget"]) + 1
    plateau = ([float(i) for i in range(1, climb + 1)]
               + [float(climb)] * (int(job.budget) - climb))
    fired = next((i for i, best in enumerate(plateau, start=1)
                  if stop(i, best)), None)
    assert fired is not None, (
        "the plateau was never detected inside the continuation's own budget")
    assert fired < int(job.budget)
    assert fired > int(eff["budget"]), (
        "the probe must fire past the recommended budget, or it proves "
        "nothing about which budget sized the rule")


def test_the_continuation_starts_the_live_sampler_launch_starts():
    """``launch`` does ``live.reset()`` + ``live.start(...)``; the
    continuation stopped at the reset, so the panel showed the switch ON,
    "0 samples" and "waiting for the first improvement" for the whole run."""
    from gui.v3 import sampler as sampler_mod

    started: list = []
    real = sampler_mod.IncumbentSampler.start

    def spy(self, cfg_fn, incumbent_fn, running_fn):
        started.append(cfg_fn())
        # NOT started for real: the thread would sample the objective

    sampler_mod.IncumbentSampler.start = spy
    try:
        ctx = _shell()
        rd, eff = _finished_at_the_recommended_budget(ctx)
        _finished(ctx, rd)
        queued = _captured_launch(ctx)
        ctx.S["run"]["continue_extra"] = 7
        ctx.render("wing", "run")
        _press(_button(ctx.views[("wing", "run")], "Keep going"))
    finally:
        sampler_mod.IncumbentSampler.start = real
    assert started, "the live sampler was never started for the continuation"
    # ...and it samples the run that is actually running, not the one before
    assert int(started[-1].budget) == int(queued[-1].budget)


# ----------------------------------------------------------------- D14
def test_taking_the_offer_on_a_pinned_row_releases_the_pin():
    """The card is drawn on a pinned row on purpose (the gate is right about
    the box), but the OFFER was a no-op: ``_widen_row`` wrote the band and
    left ``W["fixed"]`` alone, so ``config.bounds_overrides`` widened the
    band back down to contain the pin and ``size_box_conflicts`` went on
    judging the row at [20, 20]. The press repainted byte-identical text and
    the identical button, and changed nothing in the run either."""
    from gui.v3 import config

    ctx = _heavy()
    ctx.S["wing"]["fixed"]["S_m2"] = 20.0
    assert config.build_cfg(ctx.S).pinned == {"S_m2": 20.0}

    ctx.render("wing", "box")
    view = ctx.views[("wing", "box")]
    offer = _button(view, "set S_m2 to")
    _press(offer)

    assert "S_m2" not in (ctx.S["wing"].get("fixed") or {})
    cfg = config.build_cfg(ctx.S)
    assert not (cfg.pinned or {}).get("S_m2")
    lo, hi = (cfg.bounds_overrides or {})["S_m2"]
    # the band the RUN gets is the band the button named, not one widened
    # back down to hold the pin
    assert lo > 90.0, (lo, hi)
    assert [f for f in api.size_box_conflicts(cfg) if f["empty"]] == []


def test_the_reach_names_the_pin_instead_of_sending_the_user_to_stage_1():
    """``relax.plan`` dropped a pinned row silently and fell to "there is no
    ROW to move for that — it is answered at stage 1", which is a sentence
    about a gate that names no row at all. The gate named ``S_m2``, gave a
    band, and the row is in the design box."""
    from gui.v3 import relax

    ctx = _heavy()
    ctx.S["wing"]["fixed"]["S_m2"] = 20.0
    rd = _all_refused(ctx, n=6)
    assert (rd.get("pinned") or {}).get("S_m2") == 20.0

    p = relax.plan(ctx.S, rd)
    assert p["moves"] == []                     # a pinned row cannot move…
    assert "S_m2" not in (p.get("overrides") or {})
    blocked = {b["label"]: b for b in p["blocked"]}
    assert "S_m2" in blocked, p["reason"]       # …but it is NAMED
    assert "held FIXED" in blocked["S_m2"]["why"]
    assert "Unpin it" in blocked["S_m2"]["why"]
    assert "answered at stage 1" not in p["reason"]
    lines = [t for t, _ in relax.verdict({"plan": p, "record": rd})]
    assert any("Unpin it" in t for t in lines), lines


def test_the_stage_1_sentence_survives_where_it_belongs():
    """A gate that names NO row is what that sentence is for. Without a
    guard, the fix would have taken it away from its only legitimate use."""
    from gui.v3 import relax

    ctx = _heavy()
    rd = _all_refused(ctx, n=6)
    p = relax.plan(ctx.S, rd)
    # this box HAS a row to move, so the sentence must not appear here
    assert "answered at stage 1" not in p["reason"]
    assert [m["label"] for m in p["moves"]] == ["S_m2"]


# ----------------------------------------------------------------- D16
def test_every_seed_of_a_repeat_queue_lands_and_the_best_is_published():
    """``mgr.current`` returns the running job first and otherwise the LAST
    terminal one, and the worker flips job k done and k+1 running within a
    few bytecodes — so a 0.5 s heartbeat effectively never saw an
    intermediate job terminal and stage 4 was handed an arbitrary member of
    the queue. Scored 1.111 / 2.222 / 0.333, the shell published 0.333."""
    from gui import nice_app as v1
    from gui.v3 import config

    ctx = _shell()
    cfg0 = config.build_cfg(ctx.S, seed=0)
    dim = len(api.PROBLEM_SPECS[cfg0.problem_name].param_labels)
    jobs = []
    for k, score in enumerate((1.111, 2.222, 0.333)):
        cfg = config.build_cfg(ctx.S, seed=k)
        job = v1.RunJob(cfg=cfg, label=f"air · seed {k}", budget=4)
        job.result = api.partial_result(
            cfg, [{"x": [0.5] * dim, "f": score, "g": [0.1],
                   "feasible": True}])
        job.status = "done"
        jobs.append(job)
    ctx.manager.jobs = jobs
    ctx.manager.version += 1
    assert ctx.manager.current is jobs[-1], (
        "the cursor is the last terminal job — the state this is about")

    logged: list = []
    ctx.log = lambda text, level="info": logged.append(text)
    for poll in ctx.polls:
        poll()

    rec = ctx.S["run"]["record"]
    assert rec is not None and float(rec["best_score"]) == 2.222
    assert int(rec["config"]["seed"]) == 1
    # …and every seed's own outcome is on the log, not just the published one
    for score in ("1.111", "2.222", "0.333"):
        assert any(score in t for t in logged), logged
    assert any("best of 3" in t for t in logged), logged
    # a second heartbeat must not re-log or re-publish the same queue
    logged.clear()
    ctx.manager.version += 1
    for poll in ctx.polls:
        poll()
    assert logged == []


def test_a_queue_in_which_nothing_flew_still_publishes_a_record():
    """The whole no-solution card is read off a record, so a queue with no
    feasible incumbent anywhere must still publish one — the old code
    published whatever ``current`` pointed at, and the new rule must not
    become "publish nothing" when there is no best."""
    from gui import nice_app as v1
    from gui.v3 import config

    ctx = _heavy()
    cfg = config.build_cfg(ctx.S, seed=0)
    dim = len(api.PROBLEM_SPECS[cfg.problem_name].param_labels)
    jobs = []
    for k in range(2):
        job = v1.RunJob(cfg=config.build_cfg(ctx.S, seed=k),
                        label=f"air · seed {k}", budget=4)
        job.result = api.partial_result(
            cfg, [{"x": [0.5] * dim, "f": -100.0, "g": [-1.0],
                   "feasible": False}])
        job.status = "done"
        jobs.append(job)
    ctx.manager.jobs = jobs
    ctx.manager.version += 1
    for poll in ctx.polls:
        poll()
    rec = ctx.S["run"]["record"]
    assert rec is not None and rec.get("best_score") is None


# ----------------------------------------------------------------- D23
def test_a_box_that_cannot_be_built_is_a_conflict_the_card_states():
    """``api._size_band_kwargs`` raises at BUILD time on a ``ws_pa`` band
    above the mission's own ceiling — before any gate runs, so swallowing it
    drew nothing at all: not this conflict, not the aspect-ratio one, not the
    area one. The wing stage was silent until Run."""
    from gui.v3 import config, session

    ctx = _shell()
    session.set_planform(ctx.S, "wing_loading_free")
    cap = float(config.build_cfg(ctx.S).flags["wing_loading_limit_pa"])
    row = ctx.S["wing"]["bounds"][session.WS_ROW]
    ctx.S["wing"]["bounds"][session.WS_ROW] = [float(row[0]), cap + 50.0]

    # the state this is about: the build itself refuses the box
    try:
        api.size_box_conflicts(config.build_cfg(ctx.S))
    except ValueError as exc:
        message = str(exc)
    else:                                             # pragma: no cover
        raise AssertionError("the box was expected to refuse to build")

    for view in ("box", "solver"):
        ctx.render("wing", view)
        texts = _texts(ctx.views[("wing", view)])
        assert any(message in t for t in texts), (view, texts[-4:])
        # the api's own words name the row and both ways out; the card must
        # not add a button whose destination it had to guess
        assert not any(t.startswith("set ws_pa to") for t in texts)

    # ...and it goes away when the row is brought back under the ceiling
    ctx.S["wing"]["bounds"][session.WS_ROW] = [float(row[0]), cap]
    ctx.render("wing", "solver")
    assert not any("cannot both be stated" in t
                   for t in _texts(ctx.views[("wing", "solver")]))
