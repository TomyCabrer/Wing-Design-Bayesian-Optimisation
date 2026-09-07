"""A continuation LOOKS like a continuation.

User report, session 63: *"the keep going for wing and section starts from 0
instead of continuing from the last iteration."*

On the BO ARMS it no longer does: ``api.RESUMABLE_OPTIMISERS`` (``bo`` and
``bo_slsqp``) hand the record's evaluations to the new run as its TRAINING
SET, so the counter opens at what was inherited and only the new designs are
flown. Everything else — and any record too old to carry ``eval_x``/``eval_y``
— falls back to ``api.continue_run_config`` re-launching the record's own
configuration on the SAME seed at a bigger budget, where the prefix IS flown
bit-for-bit and the longer run genuinely CONTAINS the shorter one. The other
alternative — seeding the optimiser with the previous run's best — is the arm
this repo measured and lost with (``RESULTS_HANDOFF.md``: BO is the optimiser
it is worst to hand off TO), and a resume is not that: a GP opened on 53
observations is in the same state as one that flew them.

What the shell did not do was SAY so while the run was on screen. The prefix
is not re-computed: on the wing it comes out of the on-disk evaluation memo
(``aerobo.eval_cache`` — measured on ``trim wing``, bo, 10 -> 14: 10 hits, 4
misses, 0.6 s against 1.4 s for the first run), and on the section out of the
XFOIL polar cache. So the counter races from 1 to where the last run stopped
and then crawls, which read without a label is indistinguishable from starting
over.

Three statements are asserted here.

**The count travels.** A continuation carries how many of its evaluations are
a re-flight — ``RunJob.replay`` on the wing, ``cont["was"]`` on the section —
and it is ZERO where ``api.continue_run_config`` judged the longer run
uncontained, because those evaluations really are a different set of designs.

**The views say it.** The wing's run view reads "replaying" while it is inside
the prefix and quotes how much of it is done; the section's RUNNING chip says
"re-flying N/M already paid for". Both stop saying it once the new evaluations
begin.

**Nothing is warm-started.** The continuation's config is still the record's,
on the record's seed, and no previous evaluation is handed to the optimiser as
a STARTING POINT — the resume hands over the whole observation set, which is
a different thing and is asserted separately.

**And every counter adds back what was inherited**, including the shell-wide
status bar, which is the widget visible from every stage and the last place
this was still counting from 1.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _texts(view) -> list[str]:
    return [getattr(e, "text", None) or "" for e in view.descendants()]


def _open():
    from gui.v3.app import assemble

    ctx = assemble("air")
    ctx.act("accept_mission")
    return ctx


def _launch(optimiser: str = "bo", budget: int = 14) -> dict:
    """A section run's LAUNCH SNAPSHOT, as the worker keeps it."""
    return {"shape": {"re": 1e6, "mach": 0.0, "cl_design": 0.5,
                      "objective": "cd"},
            "seed": 3,
            "search": {"optimiser": optimiser, "budget": budget},
            "n_restarts": 1}


def _record(optimiser: str = "bo", budget: int = 14, spent: int = 10) -> dict:
    return {"config": {"problem_name": "section", "optimiser": optimiser,
                       "budget": budget, "seed": 3},
            "result": {"n_evals": spent, "dim": 8}}


# ------------------------------------------------------------ the section
def test_the_section_continuation_carries_its_replayed_prefix():
    from gui.v3 import session

    out = session.continue_section(_record(spent=30, budget=40),
                                   _launch(budget=40), extra=16)
    assert out["error"] is None
    # 30 spent + 16 more = 46, past the 40 it was given, so this is the
    # LONGER-RUN branch and the api has to judge containment. (The budget is
    # 40 and not 14 on purpose: BO's initial design is clamped by a budget
    # that small, and `continue_run_config` refuses containment for that
    # reason alone — a different failure from the one under test.)
    assert out["note"]["exact"] is True
    assert out["note"]["was"] == 30
    assert out["cont"]["was"] == 30
    assert out["cont"]["exact"] is True
    # ...and the search it hands back is still the record's own
    assert out["cont"]["seed"] == 3
    assert out["cont"]["search"]["budget"] == 46


def test_an_uncontained_continuation_claims_no_replay():
    """``ga`` re-shapes its population with the budget, so its prefix is NOT
    the same designs — and a shell that shaded them as "already paid for"
    would be telling the user something the api explicitly refused to."""
    from aerobo import api
    from gui.v3 import session

    assert "ga" not in api.CONTINUABLE_OPTIMISERS
    out = session.continue_section(_record("ga", spent=30, budget=40),
                                   _launch("ga", budget=40), extra=16)
    assert out["error"] is None
    assert out["note"]["exact"] is False
    assert out["note"]["was"] == 30          # what was SPENT is still true…
    assert out["cont"]["was"] == 0           # …but nothing is called a replay
    assert out["cont"]["exact"] is False
    # and the same arguments under a CONTINUABLE optimiser do claim it, so
    # the zero above is the optimiser's verdict and not a dead branch
    same = session.continue_section(_record("bo", spent=30, budget=40),
                                    _launch("bo", budget=40), extra=16)
    assert same["cont"]["was"] == 30


def test_a_run_that_only_finishes_its_budget_replays_all_of_it():
    """Spent 10 of 14 and asked for 4 more: the budget does not move, so the
    whole of what was spent is re-flown and containment does not arise."""
    from gui.v3 import session

    out = session.continue_section(_record(spent=10, budget=14),
                                   _launch(budget=14), extra=4)
    assert out["note"]["exact"] is True
    assert out["cont"]["was"] == 10
    assert out["cont"]["search"]["budget"] == 14


def test_the_running_chip_says_it_is_re_flying_and_then_stops_saying_it():
    from gui.v3 import session

    ctx = _open()
    A = session.airfoil_state(ctx.S, "main")
    A["opt"].update(running=True, replay=10, progress=4, run_budget=14,
                    records=[])
    ctx.render("airfoil", "optimise")
    said = _texts(ctx.views[("airfoil", "optimise")])
    assert any("re-flying 4/10" in t for t in said)
    assert any("CONTINUATION" in t and "first 10 evaluations" in t
               for t in said)

    # past the prefix, the chip is the ordinary counter again
    A["opt"]["progress"] = 12
    ctx.render("airfoil", "optimise")
    said = _texts(ctx.views[("airfoil", "optimise")])
    assert any("RUNNING · 12/14" in t for t in said)
    # the CHIP stops saying it. The note under it still explains the shading
    # on the convergence plot, which is why this looks at the chip alone.
    assert not any(t.startswith("RUNNING") and "re-flying" in t
                   for t in said)


def test_an_ordinary_section_run_says_nothing_about_replaying():
    """The label is a continuation's, and a fresh search must not wear it."""
    from gui.v3 import session

    ctx = _open()
    A = session.airfoil_state(ctx.S, "main")
    A["opt"].update(running=True, replay=0, progress=4, run_budget=14,
                    records=[])
    ctx.render("airfoil", "optimise")
    said = _texts(ctx.views[("airfoil", "optimise")])
    assert any("RUNNING · 4/14" in t for t in said)
    assert not any("re-fl" in t or "CONTINUATION" in t for t in said)


# --------------------------------------------------------------- the wing
def _job(replay: int, n: int, budget: int = 14):
    from aerobo import api
    from gui import nice_app as v1

    job = v1.RunJob(cfg=api.RunConfig(problem_name="trim wing",
                                      optimiser="bo", budget=budget, seed=0),
                    label="trim wing · seed 0", budget=budget, replay=replay)
    job.status = "running"
    job.records = [{"n": i, "best": -0.1 * i, "f": -0.1 * i,
                    "feasible": True, "x": None} for i in range(1, n + 1)]
    return job


def test_a_plain_run_job_claims_no_replay():
    from gui import nice_app as v1
    from aerobo import api

    job = v1.RunJob(cfg=api.RunConfig(problem_name="trim wing"),
                    label="x", budget=8)
    assert job.replay == 0


def test_the_wing_run_view_reads_replaying_inside_the_prefix():
    ctx = _open()
    ctx.manager.jobs = [_job(replay=10, n=5)]
    ctx.render("wing", "run")
    said = _texts(ctx.views[("wing", "run")])
    assert "replaying" in said
    assert "re-flown" in said
    assert "5/10" in said                      # how much of the prefix is done
    assert "5/14" in said                      # …and the whole run's counter
    assert any("CONTINUATION" in t for t in said)


def test_the_wing_run_view_stops_saying_it_past_the_prefix():
    ctx = _open()
    ctx.manager.jobs = [_job(replay=10, n=12)]
    ctx.render("wing", "run")
    said = _texts(ctx.views[("wing", "run")])
    assert "replaying" not in said
    assert "running" in said
    # the count is still quoted — the shading on the convergence plot needs a
    # caption once the run has moved past it
    assert "10/10" in said


def test_a_wing_run_that_is_not_a_continuation_says_nothing():
    ctx = _open()
    ctx.manager.jobs = [_job(replay=0, n=5)]
    ctx.render("wing", "run")
    said = _texts(ctx.views[("wing", "run")])
    assert "re-flown" not in said
    assert not any("CONTINUATION" in t for t in said)


def _band_annotations(view) -> list[str]:
    """Annotation texts of every plotly figure the view rendered."""
    out = []
    for e in view.descendants():
        if type(e).__name__ != "Plotly":
            continue
        opts = e._props["options"]
        out += [a.get("text", "") for a in opts["layout"].get("annotations",
                                                              [])]
    return out


def test_the_convergence_plot_marks_where_the_old_run_ended():
    """The question a continuation is asked to answer is "did it buy
    anything?", and that cannot be read off a curve with no mark on it."""
    ctx = _open()
    ctx.manager.jobs = [_job(replay=10, n=12)]
    ctx.render("wing", "run")
    assert any("re-flown (10)" in t for t in _band_annotations(
        ctx.views[("wing", "run")]))

    ctx.manager.jobs = [_job(replay=0, n=12)]
    ctx.render("wing", "run")
    assert not any("re-flown" in t for t in _band_annotations(
        ctx.views[("wing", "run")]))


def test_nothing_is_warm_started():
    """The whole point of re-flying the prefix. If a continuation ever starts
    handing the optimiser a previous best, this test is where it is caught."""
    from aerobo import api

    rec = {"config": {"problem_name": "trim wing", "optimiser": "bo",
                      "budget": 10, "seed": 4},
           "best_x": [0.5, 0.5], "n_evals": 10, "dim": 2}
    cfg, note = api.continue_run_config(rec, 4)
    assert cfg.seed == 4
    assert cfg.budget == 14
    assert cfg.optimiser == "bo"
    assert note["exact"] is True
    # no field of the re-launched configuration carries the old run's answer
    got = cfg.to_dict()
    for key, value in got.items():
        assert rec["best_x"] != value, f"{key} carries the previous best"


def test_the_wing_keep_going_button_marks_the_prefix_it_re_flies():
    """Driven through the REAL handler, with the manager stubbed so nothing
    is flown: writing the RunJob by hand here would only restate the code."""
    from gui.v3 import session

    ctx = _open()
    ctx.S["run"]["record"] = {
        "config": {"problem_name": "trim wing", "optimiser": "bo",
                   "budget": 40, "seed": 4,
                   "mission_kwargs": {}, "flags": {}},
        "n_evals": 30, "dim": 2, "best_x": [0.5, 0.5]}
    caught: list = []
    ctx.manager.start = lambda jobs: caught.extend(jobs)

    ctx.act("continue_run", None, 16)
    assert len(caught) == 1
    job = caught[0]
    assert job.budget == 46                 # 30 spent + 16 more
    assert job.cfg.seed == 4                # the record's seed, not a new one
    assert job.replay == 30                 # …and the prefix is named
    assert job.eval_cache == session.EVAL_CACHE


def test_an_uncontained_wing_continuation_names_no_prefix():
    ctx = _open()
    ctx.S["run"]["record"] = {
        "config": {"problem_name": "trim wing", "optimiser": "ga",
                   "budget": 40, "seed": 4,
                   "mission_kwargs": {}, "flags": {}},
        "n_evals": 30, "dim": 2, "best_x": [0.5, 0.5]}
    caught: list = []
    ctx.manager.start = lambda jobs: caught.extend(jobs)

    ctx.act("continue_run", None, 16)
    assert len(caught) == 1
    assert caught[0].budget == 46
    assert caught[0].replay == 0


# --------------------------------------------- the counter that is always on
def _resumed_job(resumed: int, n: int, budget: int = 57):
    """A RESUMED wing job: it inherited `resumed` and has flown `n`."""
    from aerobo import api
    from gui import nice_app as v1

    job = v1.RunJob(cfg=api.RunConfig(problem_name="trim wing",
                                      optimiser="bo_slsqp", budget=budget,
                                      seed=0),
                    label="trim wing · seed 0", budget=budget,
                    replay=0,
                    resume={"n": resumed, "x": [], "y": [], "g": None})
    job.status = "running"
    job.records = [{"n": resumed + i, "best": -0.1 * i, "f": -0.1 * i,
                    "feasible": True, "x": None} for i in range(1, n + 1)]
    return job


def _beat(ctx) -> None:
    """One turn of the shell's 0.5 s heartbeat."""
    ctx.manager.version += 1
    for fn in list(ctx.polls):
        fn()


def test_the_status_bar_counts_the_evaluations_the_run_inherited():
    """THE VISIBLE HALF OF THE REPORT.

    The wing run view's own read-out already added the inherited count, but
    the shell-wide status bar — the one strip visible from every stage — read
    ``len(job.records)/budget``. A continuation of 53 by 4 therefore announced
    "1/57" and then "4/57" across the top of the screen, which is exactly what
    starting over looks like on the widget a user is most likely to be
    watching.
    """
    ctx = _open()
    ctx.manager.jobs = [_resumed_job(resumed=53, n=1)]
    _beat(ctx)
    assert "54/57" in ctx.statusbar.msg.text
    assert 0.9 < float(ctx.statusbar.bar.value) <= 1.0


def test_a_run_that_inherited_nothing_is_counted_exactly_as_before():
    """The addition must be the resume's, not a constant offset."""
    ctx = _open()
    ctx.manager.jobs = [_job(replay=0, n=3, budget=14)]
    _beat(ctx)
    assert "3/14" in ctx.statusbar.msg.text


def test_the_run_view_and_the_status_bar_agree():
    """Two counters for one run is one counter too many: they must not be
    able to drift, so the same number is asserted in both places."""
    ctx = _open()
    ctx.manager.jobs = [_resumed_job(resumed=53, n=2)]
    _beat(ctx)
    ctx.render("wing", "run")
    said = _texts(ctx.views[("wing", "run")])
    assert "55/57" in said
    assert "55/57" in ctx.statusbar.msg.text
