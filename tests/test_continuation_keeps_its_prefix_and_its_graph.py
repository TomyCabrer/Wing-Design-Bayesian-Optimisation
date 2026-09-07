"""A continuation continues: it inherits what was paid for, and buys the rest.

Reported three times, the last one flatly: *"I am doing 51 then want to
continue 8 (total 59). It restarts from 0 and does 51. It should start from
51 and do 59."* Sessions 63 and 65 answered it as a DISPLAY problem and then
as a Sobol-split problem; both were real, and neither was the report. The run
genuinely re-flew its 51 evaluations, because "one longer run of the same
search" is what continuing meant here.

It no longer does. A BO loop is a function of its observations, so the
previous run's evaluations are handed to the new one as its TRAINING SET
(``api.resume_payload`` -> ``api.run(resume=…)``): no initial design is
drawn, nothing is re-flown, the evaluation counter opens at 51, and the
record that comes back holds all 59. The re-flight below survives as the
FALLBACK — a record too old to carry its own ``eval_x``/``eval_y``, or an
optimiser whose state is not its observations (:data:`api.RESUMABLE_OPTIMISERS`
is BO alone) — and everything asserted about it still holds there.

Reported in session 65: *"Keep going, I believe it still starts from the
beginning instead of continuing from the last evaluation"*, and then *"the
continuation should also work for the sections. The graph shouldn't reset"*.

Both were real, and they were the same defect twice.

**THE SPLIT.** ``api._bo_split`` caps BO's Sobol block at ``budget - 1``, so a
run whose budget was small drew a SMALLER initial design than its problem asks
for. A longer run left to itself asks for the bigger one and therefore
diverges at its first evaluation — measured on 'tandem' at dim 10, budget
12 -> 20: the split moved (11, 1) -> (16, 4) and the histories parted at
evaluation 12. The continuation now PINS ``bo_n_init`` to the block that was
flown, so the prefix comes back bit-for-bit (and off the on-disk memo, so it
is not bought twice).

**THE SECTION.** ``session.continue_section`` passed only the dimension to
``api.continue_run_config`` and then threw the config it got back away — so
even once the api pinned the split, stage 2's run never heard about it. The
pin now travels in the continuation snapshot the worker flies
(``cont["search"]["n_init"]`` -> ``api.optimize_airfoil(n_init=…)``).

**THE GRAPH.** A continuation owns no records until its first evaluation
lands, so both convergence plots cleared to "no evaluations yet" and redrew
from evaluation 1 — the same thing the counter does, and the thing a user
reads as a restart. The curve of the run being continued now travels with the
job (``RunJob.prior_history``) and with the section's own state
(``A["opt"]["prior"]``), and is drawn as its own muted series.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ------------------------------------------------- 1. the resume itself
def test_a_continuation_buys_only_the_evaluations_that_were_added():
    """THE REPORT, as arithmetic: 12 evaluations continued by 4 must FLY 4.

    Asserted on what reached the physics — the progress callback fires once
    per evaluation this run performs — and not on a note claiming it.
    """
    from aerobo import api

    first = api.run(api.RunConfig(problem_name="trim wing", optimiser="bo",
                                  budget=12, seed=3))
    cfg, note = api.continue_run_config(first.to_dict(), 4)
    assert note["resume"] is not None and note["resumed"] == 12
    assert cfg.budget == 16

    flown: list = []
    longer = api.run(cfg, progress_cb=lambda i, b, **kw: flown.append(int(i)),
                     resume=note["resume"])
    # the four it was asked for, counted from where the first run stopped
    assert flown == [13, 14, 15, 16]
    # ...and the record is one run of 16, not a run of 4
    assert longer.n_evals == 16 and longer.resumed == 12
    assert longer.bo_split == [12, 4]
    a = np.asarray(first.eval_x, dtype=float)
    b = np.asarray(longer.eval_x, dtype=float)
    assert float(np.max(np.abs(a - b[:len(a)]))) == 0.0
    assert list(first.history) == list(longer.history[:len(first.history)])
    # a search given more evaluations cannot come back worse than the one it
    # continues: the incumbent is inherited with the observations
    assert (longer.best_score or -1e9) >= (first.best_score or -1e9)


def test_a_resumed_run_that_is_stopped_keeps_what_it_inherited():
    """Every V3 wing run stores as PARTIAL (the stop rule fires), so the
    partial path is the common one — and a continuation that dropped its
    inherited evaluations on a stop would delete the run it continues."""
    from aerobo import api

    first = api.run(api.RunConfig(problem_name="trim wing", optimiser="bo",
                                  budget=12, seed=3))
    cfg, note = api.continue_run_config(first.to_dict(), 6)
    stopped = api.run(cfg, resume=note["resume"],
                      # ...after two NEW evaluations, i.e. at 14 of 18
                      stop_rule=lambda i, best: int(i) >= 14)
    assert stopped.partial is True
    assert stopped.n_evals == 14 and stopped.resumed == 12
    a = np.asarray(first.eval_x, dtype=float)
    b = np.asarray(stopped.eval_x, dtype=float)
    assert float(np.max(np.abs(a - b[:len(a)]))) == 0.0
    # ...and that record can itself be continued, from 14
    _cfg2, note2 = api.continue_run_config(stopped.to_dict(), 3)
    assert note2["resumed"] == 14 and note2["budget"] == 17


def test_only_the_bo_arms_are_resumed_and_the_rest_still_re_fly():
    """A GP is its observations; a Sobol index and a trust region are not.
    An optimiser that cannot inherit says so instead of pretending."""
    from aerobo import api

    assert api.RESUMABLE_OPTIMISERS == frozenset({"bo", "bo_slsqp"})
    rec = {"config": {"problem_name": "trim wing", "optimiser": "random",
                      "budget": 12, "seed": 0},
           "n_evals": 12, "eval_x": [[0.5, 0.5]] * 12, "eval_y": [1.0] * 12}
    assert api.can_resume(rec) is False
    _cfg, note = api.continue_run_config(rec, 8)
    assert note["resume"] is None and note["budget"] == 20


def test_the_handoff_arm_resumes_and_buys_only_the_new_designs():
    """``bo_slsqp`` is what the V3 shell's recommended policy actually flies
    on a constrained wing, so a continuation that re-flew its prefix was the
    restart every user of that shell saw. Counted at the PHYSICS: the number
    of progress callbacks this run makes is the number of designs it bought.
    """
    from aerobo import api

    cfg = api.RunConfig(problem_name="tail", optimiser="bo_slsqp",
                        budget=12, seed=0)
    flown_a: list = []
    first = api.run(cfg, progress_cb=lambda i, b, **k: flown_a.append(int(i)))
    assert len(flown_a) == 12 and first.resumed is None
    assert api.can_resume(first.to_dict()) is True

    cont, note = api.continue_run_config(first.to_dict(), 4)
    assert note["resumed"] == 12 and note["budget"] == 16
    flown_b: list = []
    longer = api.run(cont, resume=note["resume"],
                     progress_cb=lambda i, b, **k: flown_b.append(int(i)))

    # FOUR designs reached the physics, and the counter opened at 13
    assert len(flown_b) == 4
    assert flown_b == [13, 14, 15, 16]
    assert longer.n_evals == 16 and longer.resumed == 12
    # ...and the record is one run of 16: the inherited rows, unchanged
    a = np.asarray(first.eval_x, dtype=float)
    b = np.asarray(longer.eval_x, dtype=float)
    assert len(b) == 16
    assert float(np.max(np.abs(a - b[:len(a)]))) == 0.0
    assert longer.eval_y[:12] == first.eval_y
    # a continuation cannot come back worse than the run it continues
    assert longer.best_score >= first.best_score - 1e-12
    # the phases report the NEW split and say what was inherited; there is
    # no Sobol block on a resume, so none is reported
    assert longer.handoff["n_prior"] == 12
    assert longer.handoff["n_a"] + longer.handoff["n_b"] == 4
    assert longer.handoff["bo_split"] is None


def test_a_resumed_handoff_refuses_an_initial_design_as_well():
    """The prior IS the BO phase's opening. Accepting an ``x_init`` beside it
    would silently drop one of the two — the flag-nobody-reads failure."""
    import numpy as _np
    import pytest as _pytest

    from aerobo.optimize.handoff import run_bo_slsqp_constrained

    bounds = _np.array([[0.0, 1.0], [0.0, 1.0]])

    def f_and_g(x):
        x = _np.atleast_2d(_np.asarray(x, dtype=float))
        return -_np.sum(x ** 2, axis=1), _np.ones(x.shape[0])

    prior = (_np.array([[0.5, 0.5], [0.2, 0.2]]),
             _np.array([-0.5, -0.08]), _np.ones((2, 1)))
    with _pytest.raises(ValueError, match="initial design"):
        run_bo_slsqp_constrained(f_and_g, bounds, 8, seed=0,
                                 bo_split=lambda n: (n - 1, 1),
                                 x_init=_np.array([[0.1, 0.1]]), prior=prior)
    # ...and a budget with nothing left to fly is refused, not run empty
    with _pytest.raises(ValueError, match="at least one new evaluation"):
        run_bo_slsqp_constrained(f_and_g, bounds, 2, seed=0,
                                 bo_split=lambda n: (n - 1, 1), prior=prior)


def test_a_single_constraint_log_is_still_a_resume_payload():
    """THE BUG THAT DISABLED EVERY RESUME ON THE COMMON FAMILIES.

    A family with ONE constraint stores ``eval_g`` as a flat list of n
    scalars. Read with ``np.atleast_2d`` that is ``(1, n)`` — one point with
    n margins — the row-count check fails, ``resume_payload`` returns None,
    and the continuation silently falls back to re-flying its whole prefix.
    Measured on a stored 53-evaluation wing run: every value finite, and not
    resumable for this reason alone.
    """
    from aerobo import api

    rec = {"config": {"problem_name": "tail", "optimiser": "bo",
                      "budget": 4, "seed": 0},
           "is_constrained": True, "n_evals": 4,
           "eval_x": [[0.1, 0.2]] * 4, "eval_y": [1.0, 2.0, 3.0, 4.0],
           # FLAT — one scalar margin per evaluation
           "eval_g": [0.5, -0.25, 0.75, 1.0]}
    pay = api.resume_payload(rec)
    assert pay is not None and pay["n"] == 4
    assert np.asarray(pay["g"]).shape == (4, 1)
    assert api.can_resume(rec) is True

    # the (n, m) form is untouched, and a log that matches neither is refused
    rec_rows = {**rec, "eval_g": [[0.5], [-0.25], [0.75], [1.0]]}
    assert np.asarray(api.resume_payload(rec_rows)["g"]).shape == (4, 1)
    assert api.resume_payload({**rec, "eval_g": [[0.5], [0.25]] * 3}) is None


def test_a_record_without_its_evaluations_falls_back_to_the_re_flight():
    """The old mechanism, still exact where it is the only one available: the
    Sobol block a small budget CLAMPED is pinned back so the prefix repeats
    bit-for-bit. Asserted on the evaluations, not on the note."""
    from aerobo import api

    cfg = api.RunConfig(problem_name="tandem", optimiser="bo", budget=12,
                        seed=0)
    first = api.run(cfg, eval_cache=True)
    assert first.bo_split[0] < api._bo_split(20, first.dim)[0]   # clamped

    thin = {k: v for k, v in first.to_dict().items()
            if k not in ("eval_x", "eval_y", "eval_g")}
    cont, note = api.continue_run_config(thin, 8)
    assert note["exact"] is True and note["resume"] is None
    assert cont.flags[api.BO_N_INIT_FLAG] == first.bo_split[0]

    longer = api.run(cont, eval_cache=True)
    a = np.asarray(first.eval_x, dtype=float)
    b = np.asarray(longer.eval_x, dtype=float)
    assert len(b) == 20 and len(a) == 12
    assert float(np.max(np.abs(a - b[:len(a)]))) == 0.0
    # ...and it came out of the memo rather than being bought twice —
    # asked only where the memo is actually live, because it can be switched
    # off for a whole process (``eval_cache.ENV_SWITCH``) and the containment
    # above is true either way
    if (first.eval_cache or {}).get("dir"):
        assert longer.eval_cache["hits"] >= len(a)


def test_a_resume_from_another_box_is_refused_not_clipped():
    """The payload comes off a record whose config the run re-uses. If the
    two boxes differ, the evaluations are not this search's — and a resume
    that silently moved them would report a history it never flew."""
    from aerobo import api

    first = api.run(api.RunConfig(problem_name="trim wing", optimiser="bo",
                                  budget=10, seed=1))
    cfg, note = api.continue_run_config(first.to_dict(), 4)
    narrowed = api.RunConfig(**{**cfg.__dict__,
                                "bounds_overrides": {"taper": [0.85, 0.95]}})
    try:
        api.run(narrowed, resume=note["resume"])
    except ValueError as exc:
        assert "box" in str(exc)
    else:                                   # pragma: no cover - the bug
        raise AssertionError("a resume from a different box was accepted")


def test_an_unclamped_continuation_is_untouched():
    """The pin fires only where the split moved: a budget big enough to have
    drawn its full initial design sends the legacy flag set, bit-for-bit."""
    from aerobo import api

    rec = {"config": {"problem_name": "trim wing", "optimiser": "bo",
                      "budget": 40, "seed": 4},
           "n_evals": 40, "dim": 2, "bo_split": [4, 36]}
    cfg, note = api.continue_run_config(rec, 16)
    assert note["exact"] is True
    assert api.BO_N_INIT_FLAG not in (cfg.flags or {})


def test_an_optimiser_that_cannot_be_continued_still_says_so():
    """The pin is not a way of claiming containment the api cannot promise."""
    from aerobo import api

    rec = {"config": {"problem_name": "trim wing", "optimiser": "ga",
                      "budget": 12, "seed": 0},
           "n_evals": 12, "dim": 2, "bo_split": None}
    _cfg, note = api.continue_run_config(rec, 8)
    assert note["exact"] is False
    assert "ga" in note["why"]


# ---------------------------------------------------------- 2. the section
def test_the_section_continuation_carries_the_pinned_split():
    from aerobo import api
    from gui.v3 import session

    record = {"config": {"problem_name": "airfoil (section)", "optimiser": "bo",
                         "budget": 10, "seed": 0, "flags": {}},
              "result": {"n_evals": 10, "dim": 8, "searched_dim": 8,
                         "bo_split": [9, 1]}}
    launch = {"shape": {"re": 5e5, "mach": 0.0, "cl_design": 0.4,
                        "objective": "cd"},
              "seed": 0,
              "search": {"optimiser": "bo", "budget": 10, "n_init": None},
              "n_restarts": 1}
    out = session.continue_section(record, launch, extra=8)

    assert out["error"] is None
    assert out["note"]["exact"] is True
    # the number the stage hands `api.optimize_airfoil(n_init=…)`
    assert out["cont"]["search"]["n_init"] == 9
    assert out["cont"]["search"]["budget"] == 18
    # ...and the api agrees this is the flag that pins it
    cfg, _note = api.continue_run_config(
        {"config": record["config"], "dim": 8, "bo_split": [9, 1]}, 8)
    assert cfg.flags[api.BO_N_INIT_FLAG] == 9


def test_the_section_continuation_resumes_from_its_own_evaluations():
    """Stage 2's button, on a record that carries its log: the snapshot the
    worker flies asks for the NEW evaluations only, and hands the previous
    ones over as the training set."""
    from gui.v3 import session

    record = {"config": {"problem_name": "airfoil (section)", "optimiser": "bo",
                         "budget": 10, "seed": 0, "flags": {}},
              "result": {"n_evals": 10, "dim": 8, "searched_dim": 8,
                         "bo_split": [9, 1], "is_constrained": True,
                         "eval_x": [[0.1 * (i + j) for j in range(8)]
                                    for i in range(10)],
                         "eval_y": [1.0 + 0.1 * i for i in range(10)],
                         "eval_g": [[0.5] for _ in range(10)]}}
    launch = {"shape": {"re": 5e5, "mach": 0.0, "cl_design": 0.4,
                        "objective": "cd"},
              "seed": 0,
              "search": {"optimiser": "bo", "budget": 10, "n_init": None},
              "n_restarts": 1}
    out = session.continue_section(record, launch, extra=8)

    assert out["error"] is None
    assert out["note"]["exact"] is True
    assert out["note"]["resumed"] == 10
    assert out["cont"]["search"]["budget"] == 18
    assert (out["cont"]["search"]["resume"] or {})["n"] == 10
    # nothing is re-flown, so the stage has no prefix to announce
    assert out["cont"]["was"] == 0
    assert out["cont"]["resumed"] == 10


def test_a_section_run_that_only_finishes_its_budget_pins_nothing():
    """Its budget does not move, so there is no split to restore and the
    continuation is the same configuration exactly."""
    from gui.v3 import session

    record = {"config": {"problem_name": "airfoil (section)",
                         "optimiser": "bo", "budget": 20, "seed": 0,
                         "flags": {}},
              "result": {"n_evals": 12, "dim": 8, "bo_split": [16, 4]}}
    launch = {"shape": {"re": 5e5, "mach": 0.0, "cl_design": 0.4},
              "seed": 0,
              "search": {"optimiser": "bo", "budget": 20, "n_init": None},
              "n_restarts": 1}
    out = session.continue_section(record, launch, extra=4)
    assert out["note"]["exact"] is True
    assert out["cont"]["search"]["budget"] == 20
    assert out["cont"]["search"].get("n_init") is None


# ------------------------------------------------------------ 3. the graph
def test_the_wing_continuation_carries_the_curve_it_continues():
    from gui.v3.app import assemble

    ctx = assemble("air")
    ctx.act("accept_mission")
    curve = [20.0 + 0.1 * i for i in range(30)]
    ctx.S["run"]["record"] = {
        "config": {"problem_name": "trim wing", "optimiser": "bo",
                   "budget": 40, "seed": 4, "mission_kwargs": {}, "flags": {}},
        "n_evals": 30, "dim": 2, "best_x": [0.5, 0.5], "history": curve}
    caught: list = []
    ctx.manager.start = lambda jobs: caught.extend(jobs)

    ctx.act("continue_run", None, 16)
    job = caught[0]
    assert job.replay == 30
    assert list(job.prior_history) == curve


def test_the_wing_keep_going_button_resumes_instead_of_re_flying():
    """The button the report is about. 12 flown, "keep going" 4: the job goes
    out carrying the 12 as its training set, with NOTHING marked as a
    re-flight, and its budget is 16."""
    from aerobo import api
    from gui.v3.app import assemble

    first = api.run(api.RunConfig(problem_name="trim wing", optimiser="bo",
                                  budget=12, seed=3))
    ctx = assemble("air")
    ctx.act("accept_mission")
    ctx.S["run"]["record"] = first.to_dict()
    caught: list = []
    ctx.manager.start = lambda jobs: caught.extend(jobs)

    ctx.act("continue_run", None, 4)
    job = caught[0]
    assert job.cfg.budget == 16
    assert (job.resume or {})["n"] == 12
    assert job.replay == 0          # nothing is repeated, so nothing to warn
    assert list(job.prior_history) == list(first.history)


def test_a_stopped_wing_continuation_rebuilds_the_whole_run():
    """The manager's cancel path, which is where a V3 run usually ends. It
    rebuilds the result from the progress log, and on a resumed run that log
    has to open with the evaluations the job inherited."""
    from aerobo import api
    from gui import nice_app as v1

    first = api.run(api.RunConfig(problem_name="trim wing", optimiser="bo",
                                  budget=12, seed=3))
    _cfg, note = api.continue_run_config(first.to_dict(), 4)
    rows = v1._resume_rows(note["resume"])
    assert len(rows) == 12
    assert [r["n"] for r in rows] == list(range(1, 13))
    assert rows[0]["x"] == [float(v) for v in first.eval_x[0]]


def test_the_prior_curve_survives_a_run_whose_first_design_was_refused():
    """A history stores ``None`` where the best-so-far is -inf.

    That is not an edge case, it is what every run looks like until its first
    FEASIBLE design lands — and on the car families, whose boxes contain real
    refusals, it is the common opening. The prior curve was drawn with a bare
    ``float(v)`` over that list, so pressing "Keep going" on such a run put

        TypeError: float() argument must be a string or a real number,
                   not 'NoneType'

    on screen in place of the convergence plot, the moment the continuation
    logged its first record (which is when the second trace is added).

    ``nice_app._nan_history`` exists for exactly this and the main trace
    already went through it; this asserts the SECOND one does too. Gated on
    what reaches the screen — traces present, no exception text — because the
    shell catches a render error and draws it as a card, so the failure is
    visible to a user and invisible to a test that only calls the function.
    """
    from gui import nice_app as v1
    from gui.v3.app import assemble

    ctx = assemble("track")
    ctx.select("wing", "run")

    job = v1.RunJob(cfg=None, label="cont", budget=12)
    #                  vvvv the first design was refused
    job.prior_history = [None, 25.99, 28.69, 28.69]
    job.records = [{"n": 1, "f": 30.1, "best": 30.1, "feasible": True}]
    job.status, job.replay = "running", 0
    ctx.manager.jobs = [job]

    ctx.render("wing", "run")
    view = ctx.views[("wing", "run")]

    blob = " ".join(str(getattr(e, "text", "") or "")
                    for e in view.descendants())
    for word in ("TypeError", "NoneType", "float()"):
        assert word not in blob, blob[:400]

    names = _named_traces(view)
    assert "the run this continues" in names
    assert "best feasible so far" in names

    # ...and the refused evaluation is a GAP in that curve, not a zero and
    # not a dropped point: the x axis still counts it
    prior = next(t for fig in _figures(view) for t in (fig.get("data") or [])
                 if t.get("name") == "the run this continues")
    ys = _trace_y(prior)
    assert ys.size == len(job.prior_history)
    assert np.isnan(ys[0]) and np.isfinite(ys[1:]).all()
    assert list(prior["x"]) == [1, 2, 3, 4]


def test_a_fresh_wing_run_carries_no_prior_curve():
    """The series is a continuation's, and a first run must not wear it."""
    from gui.v3.app import assemble

    ctx = assemble("air")
    ctx.act("accept_mission")
    caught: list = []
    ctx.manager.start = lambda jobs: caught.extend(jobs)
    ctx.act("launch")
    assert caught, "the launch handler queued nothing"
    assert not (getattr(caught[0], "prior_history", None) or [])


def _figures(view) -> list[dict]:
    """The plotly payloads a view rendered."""
    return [e._props["options"] for e in view.descendants()
            if type(e).__name__ == "Plotly"]


def _trace_y(trace) -> np.ndarray:
    """A trace's y as floats, however plotly chose to encode it.

    plotly serialises a numpy array as ``{"dtype": ..., "bdata": <base64>}``
    and a plain list as a list. Both are legitimate and which one appears
    depends on what the caller handed in, so a test that reads one of them
    is asserting a serialisation choice rather than the curve.
    """
    y = trace["y"]
    if isinstance(y, dict) and "bdata" in y:
        import base64
        return np.frombuffer(base64.b64decode(y["bdata"]),
                             dtype=np.dtype(y["dtype"]))
    return np.asarray(list(y), dtype=float)


def _named_traces(view) -> set:
    return {t.get("name") for fig in _figures(view)
            for t in (fig.get("data") or [])}


def test_the_section_plot_is_not_empty_while_the_prefix_re_flies():
    """The state the stage is in one tick after "keep going": running, with
    no records of its own yet. Asserted on the TRACES, because the emptiness
    is a property of the figure and not of any label beside it."""
    from gui.v3 import session
    from gui.v3.app import assemble

    ctx = assemble("air")
    ctx.act("accept_mission")
    A = session.airfoil_state(ctx.S, "main")

    A["opt"].update(running=True, records=[], prior=[], replay=8)
    ctx.render("airfoil", "optimise")
    assert "the run this continues" not in _named_traces(
        ctx.views[("airfoil", "optimise")])

    A["opt"].update(prior=[10.0 + 0.5 * i for i in range(8)])
    ctx.render("airfoil", "optimise")
    figs = _figures(ctx.views[("airfoil", "optimise")])
    drawn = [t for fig in figs for t in (fig.get("data") or [])
             if t.get("name") == "the run this continues"]
    assert len(drawn) == 1
    assert len(drawn[0]["x"]) == 8          # the curve, not a placeholder
