"""The chord TREND is drawn, not just declared.

``chord_trend`` has been a real constraint since the chord limits went in: a
planform whose chord grows outboard is refused by ``ChordLimits`` and scores
the penalty. But it was asked in the wing size-limits card, in a list of
lengths and angles, while the picture of what the design box draws — the band
in the chord-law panel — went on including every law the box contains,
outboard bulges and all. So the drawing and the run disagreed: the band
showed planforms that could never fly.

Now :func:`aerobo.geometry.chord_reach` takes the trend, the band is the
band of laws the trend ALLOWS, and the trend is asked in the chord-law panel
beside the drawing it clips. The rules:

* ``trend="free"`` is every published call, and returns exactly what it did;
* a trend never widens anything: the band is a subset and the flyable share
  can only fall;
* what the band claims is flyable, the physics accepts — same test, same
  tolerance (:meth:`ChordLimits.violation`);
* the trend is ONE statement over the design, so one panel asks it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aerobo import geometry           # noqa: E402

#: the band is drawn on a grid, so a law BETWEEN grid points may sit a hair
#: outside it. Free, the extremes are box CORNERS and the grid holds them
#: exactly (test_chord_reach.py allows 2e-3); under a trend the extremes sit
#: on the trend boundary, which no grid lands on, and the measured residual
#: is ~4e-3 of the chord (chord_reach's docstring quotes it) — a line's width
BAND_TOL = 5e-3


def _build(S):
    """The problem the run would build from this session — the box the
    OPTIMISER searches, so a claim about what flies is a claim about it."""
    from aerobo import api

    from gui.v3 import config

    cfg = config.build_cfg(S)
    return api.PROBLEM_SPECS[cfg.problem_name].build(
        cfg.mission_kwargs, cfg.flags, cfg.bounds_overrides)


def _chord(reach, lam: float, edge: str = "hi") -> np.ndarray:
    """The band edge as a CHORD distribution (the band is a ratio to the
    straight-taper chord, and the trend is about the chord)."""
    base = 1.0 - (1.0 - lam) * reach.eta
    return getattr(reach, edge) * base


# ------------------------------------------------------------- the band itself
def test_free_is_the_published_answer_bit_for_bit():
    box = geometry.chord_bounds(3)
    for lam in (1.0, 0.55, 0.2):
        a = geometry.chord_reach(box, lam)
        b = geometry.chord_reach(box, lam, trend="free")
        assert a.trend == "free"
        assert np.array_equal(a.lo, b.lo) and np.array_equal(a.hi, b.hi)
        assert a.flyable_frac == b.flyable_frac
        assert a.dev_max == b.dev_max


def test_root_largest_clips_the_band_to_planforms_that_never_grow():
    box = geometry.chord_bounds(3)
    lam = 1.0                     # rectangular baseline: the hardest case
    free = geometry.chord_reach(box, lam)
    held = geometry.chord_reach(box, lam, trend="root_largest")

    # the free band DOES grow outboard somewhere — that is the bug this fixes
    assert np.max(np.diff(_chord(free, lam, "hi"))) > 1e-6
    # ...and the held one cannot, on either edge
    tol = geometry.CHORD_TREND_TOL * float(np.mean(_chord(held, lam, "hi")))
    assert np.max(np.diff(_chord(held, lam, "hi"))) <= tol
    assert np.max(np.diff(_chord(held, lam, "lo"))) <= tol


def test_root_smallest_is_the_mirror_of_it():
    box = geometry.chord_bounds(3)
    held = geometry.chord_reach(box, 1.0, trend="root_smallest")
    tol = geometry.CHORD_TREND_TOL * float(np.mean(_chord(held, 1.0, "hi")))
    assert np.min(np.diff(_chord(held, 1.0, "hi"))) >= -tol
    assert np.min(np.diff(_chord(held, 1.0, "lo"))) >= -tol


def test_a_trend_can_only_narrow_what_the_box_reaches():
    box = geometry.chord_bounds(3)
    for lam in (1.0, 0.6):
        free = geometry.chord_reach(box, lam)
        for trend in ("root_largest", "root_smallest"):
            held = geometry.chord_reach(box, lam, trend=trend)
            assert held.trend == trend
            assert held.flyable_frac <= free.flyable_frac
            if held.empty:
                continue
            assert np.all(held.lo >= free.lo - 1e-12)
            assert np.all(held.hi <= free.hi + 1e-12)


def test_the_share_it_quotes_is_the_share_the_physics_would_fly():
    """The band's own claim, checked against the constructor that actually
    refuses candidates — one sampled law at a time, no shared code."""
    lam, box = 0.6, geometry.chord_bounds(3, 0.4)
    trend = "root_largest"
    lim = geometry.ChordLimits(trend=trend)
    reach = geometry.chord_reach(box, lam, trend=trend)
    rng = np.random.default_rng(11)
    flown = 0
    n = 200
    for _ in range(n):
        k = rng.uniform(box[:, 0], box[:, 1])
        try:
            geometry.Wing(b=10.0, S=10.0, taper=lam, chord_coeffs=tuple(k),
                          chord_limits=lim)
        except ValueError:
            continue
        flown += 1
    # the grid's fraction and a random sample of the same box agree to
    # sampling noise (the grid is 17^3 over the same box)
    assert reach.flyable_frac == pytest.approx(flown / n, abs=0.08)


def test_the_band_and_the_physics_agree_law_by_law():
    """Not a fraction this time: the SAME law, asked of both.

    The two count stations differently — the band draws on
    CHORD_REACH_STATIONS (65) and ChordLimits checks on ``n_check`` (129) —
    so a monotonicity failure narrower than a station gap could in principle
    be admitted by one and refused by the other. Over 2k laws per taper it
    never is, which is what makes the picture safe to read as the run's own
    answer.
    """
    rng = np.random.default_rng(3)
    laws = rng.uniform(-0.5, 0.5, size=(2000, 3))
    checked = 0
    for lam in (1.0, 0.7, 0.45):
        lim = geometry.ChordLimits(trend="root_largest")
        for k in laws:
            if geometry.chord_multiplier_extrema(k)[0] <= \
                    geometry.CHORD_MULT_FLOOR:
                continue            # not a wing at all; both refuse it
            checked += 1
            # a box of ONE law: its flyable share is 1 if the band admits it
            one = np.column_stack([k, k])
            drawn = geometry.chord_reach(one, lam, n_grid=1,
                                         trend="root_largest")
            wing = geometry.Wing(b=10.0, S=10.0, taper=lam,
                                 chord_coeffs=tuple(k))
            y = np.linspace(0.0, wing.b / 2.0, lim.n_check)
            flies = lim.violation(y, wing.chord(y)) is None
            assert (not drawn.empty) == flies, (lam, k)
    assert checked > 3000


def test_every_law_the_band_admits_builds_a_wing_the_limits_accept():
    """No law inside the drawn band may be one the run would refuse."""
    lam, box = 0.5, geometry.chord_bounds(3)
    trend = "root_largest"
    lim = geometry.ChordLimits(trend=trend)
    reach = geometry.chord_reach(box, lam, trend=trend)
    assert not reach.empty
    # the band is the envelope of the ADMITTED laws, so a law the limits
    # accept must lie inside it — that is the containment the drawing claims
    rng = np.random.default_rng(5)
    checked = 0
    for _ in range(300):
        k = rng.uniform(box[:, 0], box[:, 1])
        try:
            wing = geometry.Wing(b=10.0, S=10.0, taper=lam,
                                 chord_coeffs=tuple(k), chord_limits=lim)
        except ValueError:
            continue
        checked += 1
        y = 0.5 * wing.b * reach.eta
        ratio = wing.chord(y) / (wing.c_root * (1.0 - (1.0 - lam) * reach.eta))
        assert np.all(ratio >= reach.lo - BAND_TOL), k
        assert np.all(ratio <= reach.hi + BAND_TOL), k
    assert checked > 20


def test_a_straight_taper_is_judged_by_the_same_rule():
    """No chord law at all is still a chord distribution: a taper that
    narrows outboard is root-largest and cannot be root-smallest."""
    assert geometry.chord_reach([], 0.5, trend="root_largest").flyable_frac == 1.0
    assert geometry.chord_reach([], 0.5, trend="root_smallest").empty
    # ...and a rectangle is both
    assert geometry.chord_reach([], 1.0, trend="root_smallest").flyable_frac == 1.0
    assert geometry.chord_reach([], 1.0, trend="root_largest").flyable_frac == 1.0


def test_an_impossible_trend_says_so_rather_than_drawing_nothing():
    """A box in which no law keeps the trend is reported as empty, the same
    way a box that collapses every chord is."""
    box = np.array([[0.3, 0.5], [0.3, 0.5], [0.3, 0.5]])   # every law grows
    assert geometry.chord_reach(box, 1.0, trend="root_largest").empty
    assert not geometry.chord_reach(box, 1.0).empty


def test_an_unknown_trend_is_refused():
    with pytest.raises(ValueError, match="unknown chord trend"):
        geometry.chord_reach(geometry.chord_bounds(3), 1.0, trend="tip_largest")
    with pytest.raises(ValueError, match="unknown chord trend"):
        geometry.chord_reach([], 1.0, trend="whatever")


# ------------------------------------------------------------------- the shell
def _open():
    from gui.v3.app import assemble

    ctx = assemble("air")
    ctx.act("accept_mission")
    ctx.render("wing", "box")
    return ctx


def _texts(view):
    return [getattr(e, "text", None) or "" for e in view.descendants()]


def test_the_trend_is_asked_beside_the_drawing_it_clips(capsys):
    ctx = _open()
    texts = _texts(ctx.views[("wing", "box")])
    assert "Chord law · wing" in texts
    # one place, and it is the chord law's panel — not the limits card below
    assert texts.count("trend") == 1
    assert texts.index("Chord law · wing") < texts.index("trend")
    assert texts.index("trend") < texts.index("Wing size limits")
    assert "Traceback" not in capsys.readouterr().err


def test_choosing_it_redraws_every_band_and_reaches_the_run(capsys):
    from gui.v3 import config

    ctx = _open()
    ctx.act("set_choice", "tail", True)
    ctx.act("set_choice", "tail_design", "planform")
    ctx.render("wing", "box")
    before = _texts(ctx.views[("wing", "box")])
    # two surfaces, two laws — and still ONE trend control between them
    assert before.count("Chord law · wing") == 1
    assert before.count("Chord law · second surface") == 1
    assert before.count("trend") == 1

    ctx.act("set_chord_trend", "root_largest")
    assert config.cfg_dict(ctx.S)["flags"]["chord_trend"] == "root_largest"
    after = _texts(ctx.views[("wing", "box")])
    assert any("the largest chord" in t for t in after)
    # the surface that does not own the control still says what holds for it
    assert any("one statement for the whole design" in t for t in after)

    # ...and leaving it again really does UNCONSTRAIN the run. Asserted as
    # the value the solver receives, not as the absence of a key: since the
    # shell OPENS on `root_largest` (config.CHORD_TREND), "free" has to be
    # stated to be reachable, and a test that demanded the key be popped
    # would be asserting the old mechanism rather than the outcome.
    ctx.act("set_chord_trend", "free")
    assert config.cfg_dict(ctx.S)["flags"]["chord_trend"] == "free"
    built = _build(ctx.S)
    labs = list(built.param_labels)
    x = np.asarray(built.bounds, dtype=float).mean(axis=1)
    x[labs.index("chord_k1")] = 0.5          # a chord that GROWS outboard
    assert built.evaluate(x.tolist()).get("feasible") is True
    assert "Traceback" not in capsys.readouterr().err


def _band_figures(view) -> list[dict]:
    """The plotly payloads the box view rendered."""
    return [e._props["options"] for e in view.descendants()
            if type(e).__name__ == "Plotly"]


def _trace_y(trace) -> np.ndarray:
    """A trace's y values — plotly ships arrays base64-packed."""
    y = trace["y"]
    if isinstance(y, dict):
        import base64

        return np.frombuffer(base64.b64decode(y["bdata"]), dtype=y["dtype"])
    return np.asarray(y, dtype=float)


def test_the_figure_says_which_trend_it_was_drawn_under():
    """The band alone would leave a reader to notice that both its edges now
    fall from the root; the drawing states it."""
    from gui.v3.stages.wing import CHORD_TREND_LABELS

    ctx = _open()
    # the shell OPENS on the constraint, so the annotation is there from the
    # first render — and it disappears only when the trend is lifted
    figs = _band_figures(ctx.views[("wing", "box")])
    notes = [a["text"] for f in figs for a in f["layout"].get("annotations", [])]
    assert CHORD_TREND_LABELS["root_largest"] in notes
    ctx.act("set_chord_trend", "free")
    assert not any(f["layout"].get("annotations")
                   for f in _band_figures(ctx.views[("wing", "box")]))

    ctx.act("set_chord_trend", "root_largest")
    figs = _band_figures(ctx.views[("wing", "box")])
    assert figs
    notes = [a["text"] for f in figs for a in f["layout"].get("annotations", [])]
    assert CHORD_TREND_LABELS["root_largest"] in notes

    # ...and the band it drew really is clipped: neither edge grows outboard
    band = [t for t in figs[0]["data"] if t.get("name") in ("widest",
                                                            "narrowest")]
    assert len(band) == 2
    for trace in band:
        y = _trace_y(trace)
        assert np.max(np.diff(y)) <= 1e-9 * float(np.mean(y))


def test_a_straight_taper_problem_still_asks_it_with_the_limits(capsys):
    """No chord law, no chord-law panel — and the trend is a real constraint
    on a trapezoid too (it decides whether the taper may exceed 1), so it
    appears in the limits card instead. One question, one place, either way.
    """
    ctx = _open()
    ctx.act("set_choice", "chord", "fixed")
    ctx.render("wing", "box")
    texts = _texts(ctx.views[("wing", "box")])
    assert not any(t.startswith("Chord law · ") for t in texts)
    assert texts.count("trend") == 1
    assert texts.index("Wing size limits") < texts.index("trend")
    assert "Traceback" not in capsys.readouterr().err
