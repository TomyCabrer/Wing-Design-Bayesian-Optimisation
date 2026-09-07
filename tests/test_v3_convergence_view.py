"""Watching a search must be possible: the page must hold still, and the
axis must be about the designs that flew.

Three defects, one view.

* THE PAGE SCROLLED ITSELF TO THE TOP. Both run views were rebuilt from
  scratch on every evaluation — ``box.clear()`` and a fresh set of plotly
  panes twice a second. The work area is one scroll pane, so clearing it
  collapses its content height and the browser answers by clamping the
  scroll position to zero: a user reading the live metrics of a running
  search was carried back to the top of the page for the whole run, and an
  expansion could never be left open long enough to tick anything in it.
* THE MENU WAS A WALL. Every numeric key of the breakdown was offered as a
  checkbox — dozens, most of them the same number in a second spelling.
* THE REFUSAL OWNED THE AXIS. A design the physics cannot score returns the
  -100 sentinel. Plotted beside an L/D of 40 or a 0-100 composite, that one
  point sets the whole scale and the convergence the plot exists to show is
  a flat line at the top of a cliff.

The assertions here are about the OUTCOME in each case — the same element
objects survive a tick, the offered checkboxes are few, the drawn range
excludes the sentinel — not about which function was called.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))


# ------------------------------------------------------------- the axis
def _records(fs, bests):
    return [{"n": i + 1, "f": f, "best": b, "feasible": f > -50.0}
            for i, (f, b) in enumerate(zip(fs, bests))]


def test_a_refusal_does_not_set_the_scale():
    """Five refusals and a climb from 20 to 30: the axis must be about the
    climb, and the sentinel must be off it."""
    from gui import nice_app as v1

    fs = [-100.0, -100.0, 20.0, 22.0, -100.0, 25.0, 26.0, 27.0, 28.0, 29.0,
          30.0]
    bests = [None, None, 20.0, 22.0, 22.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0]
    fig = v1.fig_convergence(_records(fs, bests))

    lo, hi = fig.layout.yaxis.range
    assert lo > -100.0, (
        f"the -100 refusal sentinel is still setting the axis ({lo} .. {hi})")
    assert lo <= 20.0 and hi >= 30.0, (
        f"the axis must contain every design that flew ({lo} .. {hi})")
    # ...and it must not be so generous that the climb is a flat line: the
    # 10 units the run actually covered are most of what is drawn
    assert (hi - lo) < 2.0 * (30.0 - 20.0), (
        f"the axis is {hi - lo:.3g} tall for a 10-unit climb")


def test_the_refusals_are_still_counted_below_the_axis():
    """Cropped is not deleted. A reader has to be able to see WHERE the run
    was refused, and how much of it is missing from the picture."""
    from gui import nice_app as v1

    fs = [-100.0, -100.0, 20.0, 22.0, -100.0, 25.0, 26.0, 27.0, 28.0, 29.0,
          30.0]
    bests = [None, None, 20.0, 22.0, 22.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0]
    fig = v1.fig_convergence(_records(fs, bests))

    marks = [t for t in fig.data if "refused" in (t.name or "")]
    assert len(marks) == 1, "the refused evaluations are not drawn at all"
    assert "3" in marks[0].name, (
        f"three evaluations were refused; the trace says {marks[0].name!r}")
    lo, hi = fig.layout.yaxis.range
    assert all(lo <= y <= hi for y in marks[0].y), (
        "the refusal marks are drawn outside the axis they were moved onto")
    # their TRUE value travels with them, or the hover lies about the run
    assert list(marks[0].customdata) == [-100.0, -100.0, -100.0]


def test_a_clean_run_is_left_to_plotly():
    """Nothing off scale, nothing changed: a run without refusals is drawn
    exactly as it always was, so "the axis was widened for a refusal" and
    "there were no refusals" cannot look the same."""
    from gui import nice_app as v1

    fs = [20.0, 22.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0]
    fig = v1.fig_convergence(_records(fs, fs))

    assert fig.layout.yaxis.range is None
    assert not [t for t in fig.data if "refused" in (t.name or "")]


def test_the_best_so_far_curve_is_never_cropped():
    """The percentile floor exists to survive a constrained family's own
    large penalties — but it must never crop the curve the user is reading:
    a best-so-far that started low is the whole story of the run."""
    from gui import metrics

    scored = [-100.0, 1.0] + [40.0 + 0.1 * i for i in range(30)]
    best = [None, 1.0] + [40.0 + 0.1 * i for i in range(30)]
    lo, hi = metrics.convergence_yrange(scored, best)

    assert lo < 1.0, (
        f"the best-so-far starts at 1.0 and the axis starts at {lo:.3g} — "
        f"the opening of the run is cropped out of its own convergence plot")


def test_an_all_refused_run_is_not_given_a_fake_scale():
    """Nothing flew: there is no scale to be aware of, and inventing one
    would draw an empty pane over a run that has a perfectly honest (flat,
    at the sentinel) story to tell."""
    from gui import metrics

    assert metrics.convergence_yrange([-100.0] * 6, []) is None


def test_the_sentinel_is_read_off_the_solver():
    """A clipping rule pinned to a hard-coded -100 would silently stop
    clipping the day the penalty moved."""
    from aerobo.objective import PENALTY
    from gui import metrics

    assert metrics.refusal_sentinel() == float(PENALTY)


# ------------------------------------------------ the view holds still
def _fake_sampler(keys_and_values):
    """An IncumbentSampler already holding two samples, so a panel can be
    rendered without a search having run."""
    from gui.v3 import sampler as sampler_mod

    class _Loaded(sampler_mod.IncumbentSampler):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.samples = [{"n": 1, "metrics": dict(keys_and_values)},
                            {"n": 2, "metrics": {k: v * 1.1 for k, v
                                                 in keys_and_values.items()}}]
            self.version = 2

    return _Loaded


_LIVE_KEYS = {
    "LoD": 32.0, "cd_counts": 88.0, "cl_max": 1.4, "CD_counts": 210.0,
    "CDi_counts": 120.0, "CDp_counts": 90.0, "e": 0.93, "cm": -0.05,
    "tc": 0.12, "r_le": 0.014, "alpha_deg": 3.2, "twist_env_deg": 4.0,
    "chord_dev": 0.2, "ldmax": 40.0, "ldcr": 33.0, "astall": 12.0,
    "some_unranked_number": 7.0, "another_one": 8.0,
}


def _shell(sampler_cls=None):
    from gui.v3 import sampler as sampler_mod
    from gui.v3.app import assemble

    if sampler_cls is None:
        return assemble("air")
    real = sampler_mod.IncumbentSampler
    sampler_mod.IncumbentSampler = sampler_cls
    try:
        return assemble("air")
    finally:
        sampler_mod.IncumbentSampler = real


def _job(ctx, n: int = 4, status: str = "running"):
    """A run on the manager the wing stage reads, without a thread."""
    from gui import nice_app as v1

    job = v1.RunJob(cfg=None, label="test", budget=20)
    job.status = status
    job.records = [{"n": i + 1, "f": 30.0 + i, "best": 30.0 + i,
                    "feasible": True} for i in range(n)]
    ctx.manager.jobs = [job]
    ctx.manager.version += 1
    return job


def _run_view(ctx):
    ctx.S["ui"]["selected"] = "wing"
    ctx.S["ui"]["tab"]["wing"] = "run"
    # the stage measures a recommended box in the background on arrival and
    # repaints when it lands. That is a real repaint of a real card and it
    # is not what these tests are about — switch it off so what remains is
    # the heartbeat, which is the thing that used to scroll the page.
    ctx.S["wing"].setdefault("auto_rec", {})["off"] = True
    ctx.render("wing", "run")
    return ctx.views[("wing", "run")]


def _plotlys(view):
    from nicegui.elements.plotly import Plotly

    return [e for e in view.descendants() if isinstance(e, Plotly)]


def _tick(ctx):
    for fn in ctx.polls:
        fn()


def test_an_evaluation_does_not_rebuild_the_run_view():
    """THE SCROLL BUG, as an object identity. A new evaluation must reach
    the panes that are already on screen; a fresh set of them is a page
    whose height collapsed, which is what threw the user back to the top."""
    ctx = _shell(_fake_sampler(_LIVE_KEYS))
    job = _job(ctx)
    view = _run_view(ctx)
    before = _plotlys(view)
    assert before, "the run view draws no figures at all"

    job.records.append({"n": 5, "f": 35.0, "best": 35.0, "feasible": True})
    ctx.manager.version += 1
    _tick(ctx)

    after = _plotlys(ctx.views[("wing", "run")])
    assert [id(e) for e in after] == [id(e) for e in before], (
        "the run view was rebuilt by an evaluation — the panes on screen "
        "are different objects, so the page collapsed and re-grew")


def test_an_evaluation_still_redraws_the_trace():
    """Holding still must not mean going stale: the same pane, new data."""
    ctx = _shell(_fake_sampler(_LIVE_KEYS))
    job = _job(ctx)
    view = _run_view(ctx)
    conv = _plotlys(view)[0]
    before = str(conv.figure)

    job.records.append({"n": 5, "f": 99.0, "best": 99.0, "feasible": True})
    ctx.manager.version += 1
    _tick(ctx)

    assert str(conv.figure) != before, (
        "the convergence pane held still AND held stale — the new "
        "evaluation never reached it")


def test_a_finished_run_does_rebuild_the_view():
    """The tick is not a refusal to repaint. When the STRUCTURE moves — a
    run finishes, and the score block and the "did it converge?" card
    arrive — the view is built again."""
    ctx = _shell(_fake_sampler(_LIVE_KEYS))
    job = _job(ctx)
    view = _run_view(ctx)
    before = _plotlys(view)

    job.status = "cancelled"
    ctx.manager.version += 1
    _tick(ctx)

    after = _plotlys(ctx.views[("wing", "run")])
    assert [id(e) for e in after] != [id(e) for e in before], (
        "a finished run left the running run's view on screen")


# -------------------------------------------------- the menu is a few
def _checkboxes(view):
    from nicegui.elements.checkbox import Checkbox

    return [e for e in view.descendants() if isinstance(e, Checkbox)]


def _inside_expansion(view):
    from nicegui.elements.checkbox import Checkbox
    from nicegui.elements.expansion import Expansion

    out = []
    for e in view.descendants():
        if isinstance(e, Expansion):
            out += [d for d in e.descendants() if isinstance(d, Checkbox)]
    return out


def test_the_live_menu_leads_with_a_few():
    """Eighteen reported numbers must not be eighteen checkboxes above the
    plot. The lead is short; everything else is behind one press."""
    ctx = _shell(_fake_sampler(_LIVE_KEYS))
    _job(ctx)
    view = _run_view(ctx)

    every = _checkboxes(view)
    hidden = _inside_expansion(view)
    lead = [c for c in every if c not in hidden]

    assert lead, "the live panel offers nothing at all"
    assert len(lead) <= 5, (
        f"{len(lead)} checkboxes are offered on sight out of "
        f"{len(_LIVE_KEYS)} reported numbers")
    assert hidden, (
        "the rest of the breakdown is not reachable — it was dropped, not "
        "collapsed")


def test_nothing_the_run_reported_is_dropped():
    """Collapsed, never gone: every ranked number the sample carried is
    still tickable somewhere on the panel."""
    from gui import metrics

    ctx = _shell(_fake_sampler(_LIVE_KEYS))
    _job(ctx)
    view = _run_view(ctx)

    primary, more = metrics.live_metric_panel(list(_LIVE_KEYS),
                                              metrics.live_metric_defaults(
                                                  list(_LIVE_KEYS)))
    assert len(_checkboxes(view)) == len(primary) + len(more) >= 10


def test_the_panel_opens_on_the_objective():
    """No stored list: the panel ticks what the catalogue says this run's
    objective is, so a family that spells it differently still opens on a
    plot with something in it."""
    ctx = _shell(_fake_sampler(_LIVE_KEYS))
    _job(ctx)
    view = _run_view(ctx)

    ticked = [c for c in _checkboxes(view) if c.value]
    assert ticked, "the live plot opens empty"
    assert len(ticked) <= 3, (
        f"{len(ticked)} series are ticked by default — more than three on "
        f"two axes is a spaghetti plot")


# ------------------------------- the SECTION stage keeps the same contract
def _optimise_view(ctx):
    """The section stage mid-run, without an XFOIL search behind it."""
    from gui.v3 import session

    A = session.airfoil_state(ctx.S, "main")
    A["opt"].update(running=True, progress=3, run_budget=24,
                    records=[{"n": i + 1, "best": 50.0 + i}
                             for i in range(4)])
    ctx.S["ui"]["selected"] = "airfoil"
    ctx.S["ui"]["tab"]["airfoil"] = "optimise"
    ctx.render("airfoil", "optimise")
    return A, ctx.views[("airfoil", "optimise")]


def test_a_section_evaluation_does_not_rebuild_its_view():
    """An XFOIL search runs for MINUTES at two repaints a second. The same
    contract as the wing's, on the view where it costs the most."""
    ctx = _shell(_fake_sampler(_LIVE_KEYS))
    A, view = _optimise_view(ctx)
    before = _plotlys(view)
    assert before, "the optimise view draws no figures at all"

    A["opt"]["records"].append({"n": 5, "best": 60.0})
    A["opt"]["progress"] = 5
    _tick(ctx)

    after = _plotlys(ctx.views[("airfoil", "optimise")])
    assert [id(e) for e in after] == [id(e) for e in before], (
        "the section's optimise view was rebuilt by an evaluation")


def test_a_section_evaluation_still_redraws_its_trace():
    ctx = _shell(_fake_sampler(_LIVE_KEYS))
    A, view = _optimise_view(ctx)
    conv = _plotlys(view)[0]
    before = str(conv.figure)

    A["opt"]["records"].append({"n": 5, "best": 99.0})
    _tick(ctx)

    assert str(conv.figure) != before


def test_the_section_opening_refusals_do_not_own_its_axis():
    """A CST search opens on shapes XFOIL cannot fly, so the best-so-far is
    the sentinel until one converges. That opening cliff is 100 units tall
    and the climb the user is watching is three."""
    ctx = _shell(_fake_sampler(_LIVE_KEYS))
    A, view = _optimise_view(ctx)
    A["opt"]["records"] = (
        [{"n": i + 1, "best": -100.0} for i in range(3)]
        + [{"n": i + 4, "best": 55.0 + 0.3 * i} for i in range(9)])
    _tick(ctx)

    fig = _plotlys(ctx.views[("airfoil", "optimise")])[0].figure
    rng = (fig.get("layout") or {}).get("yaxis", {}).get("range")
    assert rng is not None and rng[0] > -100.0, (
        f"the section convergence axis is still owned by the refusal "
        f"({rng})")
    assert rng[0] <= 55.0 and rng[1] >= 57.4
