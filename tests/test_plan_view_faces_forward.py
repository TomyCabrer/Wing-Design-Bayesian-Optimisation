"""A plan view must draw the aircraft NOSE-UP.

The bug this pins, reported from the shell: "wing and tail pointing backwards
(trailing edge is where leading edge should be)". Nothing was wrong with the
geometry. ``fig_planform`` puts the streamwise coordinate on plotly's y and
never reversed it, so the leading edge (-0.25 c) landed at the BOTTOM of the
pane and the trailing edge (+0.75 c) at the top, with an aft tail drawn far
ABOVE the wing. The pane mapped (span, starboard +) x (streamwise, aft +),
whose cross product is -z: a view from underneath. On a symmetric aircraft the
left/right mirror is invisible, so the only thing a reader can see is that the
nose points the wrong way.

WHAT IS ASSERTED, AND WHY IT IS THE LAYOUT AND NOT THE DATA. The fix reverses
the AXIS and leaves the numbers alone, because the numbers are the package's
own frame and ``test_geometry_views.py`` pins them there. So the naive
assertion — "the leading edge's plot-y is above the trailing edge's" — is
exactly backwards: with the data untouched the LE is and must remain the more
NEGATIVE value. Asserting it that way would fail on the very fix it is meant to
protect. The honest invariant is the pair: the data stays in package axes AND
the axis carries ``autorange='reversed'``. Both halves are checked here, so
neither can be satisfied by breaking the other.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from aerobo import api                      # noqa: E402


@pytest.fixture(scope="module")
def small_aircraft():
    """The reported case: a designed tail with a tip device, at 500 g."""
    v1 = pytest.importorskip("gui.nice_app")
    cfg = api.RunConfig(problem_name="tail [designed tail + tip device]",
                        mission_kwargs={"W_N": 0.5 * 9.81, "V": 12.0})
    built = api.PROBLEM_SPECS[cfg.problem_name].build(
        cfg.mission_kwargs, cfg.flags, None)
    x = np.array([(lo + hi) / 2 for lo, hi in built.bounds])
    geom = api.design_report(cfg, x)["geometry"]
    return v1, geom, x, built.param_labels


def _filled(fig):
    return [t for t in fig.data if getattr(t, "fill", None) == "toself"]


def test_the_plan_view_axis_is_reversed_so_the_nose_points_up(small_aircraft):
    v1, geom, x, labels = small_aircraft
    fig = v1.fig_planform(geom, x, labels)
    ya = fig.layout.yaxis
    assert "downstream" in (ya.title.text or ""), (
        "this test only means anything while the streamwise coordinate is the "
        f"one on plotly's y; it is now titled {ya.title.text!r}")
    assert ya.autorange == "reversed", (
        "the plan view draws the leading edge at -0.25 c, which is the SMALLER "
        "number; without a reversed axis plotly puts it at the bottom of the "
        "pane and the aircraft reads nose-down — the reported bug")


def test_the_plan_view_data_is_still_in_the_package_axes(small_aircraft):
    """The fix must not have been achieved by negating the geometry."""
    v1, geom, x, labels = small_aircraft
    fig = v1.fig_planform(geom, x, labels)
    traces = _filled(fig)
    assert traces, "no filled planform outline to check"
    wing = traces[0]
    yy = np.asarray(wing.y, float)
    c_root = float(np.max(np.asarray(geom["chord"], float)))
    assert yy.min() < 0.0 < yy.max(), (
        "the wing outline must straddle the quarter-chord station at 0")
    #: LE at -0.25 c and TE at +0.75 c means the outline reaches three times
    #: as far aft as it does forward. A negated plan view fails this.
    assert yy.max() / abs(yy.min()) == pytest.approx(3.0, rel=0.05), (
        f"outline spans [{yy.min():.4f}, {yy.max():.4f}] on a root chord of "
        f"{c_root:.4f}: that is not -0.25 c / +0.75 c, so the data was moved "
        f"instead of the axis")


def test_an_aft_tail_is_drawn_aft_in_the_data(small_aircraft):
    v1, geom, x, labels = small_aircraft
    fig = v1.fig_planform(geom, x, labels)
    traces = _filled(fig)
    named = {t.name: np.asarray(t.y, float) for t in traces if t.name}
    if "tail" not in named:
        pytest.skip("this build drew no separate tail outline")
    assert named["tail"].min() > named["planform"].max(), (
        "the tail's stations must all be aft of the wing's in the data; the "
        "axis reversal is what puts them at the top of the screen")


def test_the_v3_restyler_does_not_undo_the_reversal(small_aircraft):
    """V3 is the shell that matters, and it re-styles every figure it shows.

    ``gui/v3/figstyle.py`` calls ``update_yaxes`` with a block of cosmetic
    keys. If a future edit adds a range or autorange key to that block it
    would silently re-flip every plan view in the shell, and the pane would
    regress with all the tests above still green — they read the figure
    before it reaches V3.
    """
    v1, geom, x, labels = small_aircraft
    figstyle = pytest.importorskip("gui.v3.figstyle")
    fig = v1.fig_planform(geom, x, labels)
    styled = figstyle.style(fig) if hasattr(figstyle, "style") else None
    if styled is None:
        for name in ("apply", "restyle", "plot"):
            fn = getattr(figstyle, name, None)
            if callable(fn):
                styled = fn(fig)
                break
    if styled is None:
        pytest.skip("no restyle entry point found in gui.v3.figstyle")
    assert (styled if hasattr(styled, "layout") else fig).layout.yaxis.autorange \
        == "reversed", "the V3 restyler flipped the plan view back nose-down"
