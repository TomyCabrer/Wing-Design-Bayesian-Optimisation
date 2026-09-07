"""The chord limits are DRAWN, not just declared.

``chord_min_m``, ``chord_max_m`` and ``chord_rate_max_deg`` have been real
constraints since the chord limits went in: a planform that breaks one is
refused by ``ChordLimits`` where it is built and scores the penalty. But the
band in the chord-law panel — the picture of what the design box draws — went
on containing every law the box contains, sub-minimum chords and all, with the
limits shown as two dotted lines the band cheerfully crossed. On the shell's
own opening wing a live ``chord_min_m = 0.5 m`` left a band reaching 0.049 m
and a "flyable share of the box" of 97%.

That is the same disagreement :mod:`test_chord_trend_band` fixed for the
trend, and it is fixed the same way: :func:`aerobo.geometry.chord_reach` takes
the whole :class:`ChordLimits`, the band is the band of laws the limits ALLOW,
and the flyable share counts the ones they refuse. The rules:

* no ``limits`` is every published call, and returns exactly what it did;
* a limit never widens anything: the band is a subset and the flyable share
  can only fall;
* what the band claims is flyable, the physics accepts — same test, same
  object (:meth:`ChordLimits.violation`, through ``geometry.Wing``);
* the band REACHES the limit rather than stopping a grid step short of it:
  unlike a trend, a limit in metres can be projected onto along the ray;
* a limit in metres without the length that measures it raises, so a caller
  that cannot draw it finds out instead of drawing a band that ignores it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aerobo import api, geometry           # noqa: E402

#: the band is drawn on rays through a grid, so a law BETWEEN them may sit a
#: hair outside it. Free, the extremes are box corners the grid holds exactly
#: (test_chord_reach.py allows 2e-3); under a metre limit the extremes sit on
#: the limit boundary, which IS projected onto, and the measured residual is
#: ~1e-3 of the chord (chord_reach's docstring quotes it)
BAND_TOL = 2e-3

#: ...except where a TREND is live too. That one is not projected onto (there
#: is no closed form for it — chord_reach's docstring), so its own ~4e-3
#: stands whatever else is asked beside it: test_chord_trend_band's number.
TREND_TOL = 5e-3

#: a wing this shell actually opens on: b = 10 m, S = 10 m^2, so the
#: straight-taper root chord at lambda = 0.6 is 1.25 m and the tip 0.75 m
B_M, S_M2 = 10.0, 10.0


def _wing_lengths(lam: float, b: float = B_M, S: float = S_M2):
    """``(c_root, semi)`` of the trapezoid a band is drawn on."""
    return 2.0 * S / (b * (1.0 + lam)), 0.5 * b


def _chord(reach, c_root: float) -> tuple[np.ndarray, np.ndarray]:
    """The band's two edges as CHORDS in metres (it is a ratio to the
    straight-taper chord, and a limit in metres is about the chord)."""
    base = c_root * (1.0 - (1.0 - reach.taper) * reach.eta)
    return reach.lo * base, reach.hi * base


def _admitted(lam: float, lim, n: int = 3000, seed: int = 0):
    """Random laws the PHYSICS accepts under ``lim`` — the wing is built, so
    this is the same acceptance test a candidate meets in a run."""
    rng = np.random.default_rng(seed)
    box = geometry.chord_bounds(3)
    for k in rng.uniform(box[:, 0], box[:, 1], size=(n, 3)):
        try:
            yield geometry.Wing(b=B_M, S=S_M2, taper=lam,
                                chord_coeffs=tuple(k), chord_limits=lim)
        except ValueError:
            continue


# ------------------------------------------------------------- the band itself
def test_no_limits_is_the_published_answer_bit_for_bit():
    box = geometry.chord_bounds(3)
    for lam in (1.0, 0.6, 0.2):
        a = geometry.chord_reach(box, lam)
        b = geometry.chord_reach(box, lam, limits=None)
        assert a.limits is None and b.limits is None
        assert np.array_equal(a.lo, b.lo) and np.array_equal(a.hi, b.hi)
        assert a.flyable_frac == b.flyable_frac


def test_a_trend_only_limits_object_is_the_trend_argument():
    """The two ways of saying the same thing say it identically — otherwise
    the drawing would depend on which argument the caller reached for."""
    box = geometry.chord_bounds(3)
    for trend in ("root_largest", "root_smallest"):
        arg = geometry.chord_reach(box, 0.6, trend=trend)
        obj = geometry.chord_reach(box, 0.6,
                                   limits=geometry.ChordLimits(trend=trend))
        assert np.array_equal(arg.lo, obj.lo) and np.array_equal(arg.hi,
                                                                 obj.hi)
        assert arg.flyable_frac == obj.flyable_frac
        assert obj.limits.trend == trend and obj.trend == trend


def test_the_band_holds_the_minimum_chord_and_reaches_it():
    """The bug, in one assertion: the published box's band fell to 0.049 m
    under a live 0.5 m minimum. It now stops ON it."""
    lam = 0.6
    c_root, semi = _wing_lengths(lam)
    box = geometry.chord_bounds(3)
    free = geometry.chord_reach(box, lam)
    assert _chord(free, c_root)[0].min() < 0.1          # the old picture

    lim = geometry.ChordLimits(c_min_m=0.5)
    held = geometry.chord_reach(box, lam, limits=lim, c_root=c_root,
                                semi=semi)
    lo, _ = _chord(held, c_root)
    assert lo.min() >= 0.5 - 1e-9
    assert lo.min() == pytest.approx(0.5, abs=BAND_TOL)  # and reaches it
    assert held.limits is lim or held.limits == lim


def test_the_band_holds_the_maximum_chord_and_the_rate():
    lam = 0.6
    c_root, semi = _wing_lengths(lam)
    box = geometry.chord_bounds(3)
    free = geometry.chord_reach(box, lam)
    # both halves are measured against the FREE band, so a limit that was
    # quietly dropped cannot pass by drawing the picture it always drew
    assert _chord(free, c_root)[1].max() > 1.4        # the ceiling has to bite
    held = geometry.chord_reach(box, lam, c_root=c_root, semi=semi,
                                limits=geometry.ChordLimits(c_max_m=1.4))
    assert _chord(held, c_root)[1].max() <= 1.4 + 1e-9

    lim = geometry.ChordLimits(rate_max_deg=8.0)
    held = geometry.chord_reach(box, lam, limits=lim, c_root=c_root,
                                semi=semi)
    # the EDGE is an envelope of laws, not a law, so it is allowed to be
    # steeper than any of them — which is why the rate cannot be read off the
    # edge. The old form did read it off the edge and then asserted only that
    # the angle was FINITE, which arctan of a finite gradient always is: with
    # the rate dropped from chord_reach altogether this stayed green while the
    # panel drew laws ChordLimits.violation refuses. What CAN be asked is the
    # pair below — the limit BITES (dropped from both the ray projection and
    # the grid mask, the band is the free one), and every law it ADMITS is
    # still inside the band (dropped from the projection alone, the band is
    # too narrow to hold them).
    assert not held.empty
    assert held.flyable_frac < free.flyable_frac
    assert held.dev_max < free.dev_max
    lo, hi = _chord(held, c_root)
    tol = c_root * BAND_TOL
    seen = 0
    for wing in _admitted(lam, lim, n=1500):
        seen += 1
        c = wing.chord(held.eta * semi)
        assert np.all(c <= hi + tol) and np.all(c >= lo - tol), (lim, seen)
    assert seen > 0, "the 8 deg/m limit admitted no law to measure the band on"


def test_a_limit_never_widens_the_band_or_the_flyable_share():
    box = geometry.chord_bounds(3)
    for lam in (1.0, 0.6, 0.3):
        c_root, semi = _wing_lengths(lam)
        free = geometry.chord_reach(box, lam)
        for lim in (geometry.ChordLimits(c_min_m=0.55),
                    geometry.ChordLimits(c_max_m=1.45),
                    geometry.ChordLimits(rate_max_deg=10.0),
                    geometry.ChordLimits(c_min_m=0.5, c_max_m=1.5,
                                         rate_max_deg=14.0)):
            held = geometry.chord_reach(box, lam, limits=lim, c_root=c_root,
                                        semi=semi)
            assert held.flyable_frac <= free.flyable_frac + 1e-12
            if held.empty:
                continue
            assert np.all(held.lo >= free.lo - 1e-9)
            assert np.all(held.hi <= free.hi + 1e-9)
            assert held.dev_max <= free.dev_max + 1e-9


def test_what_the_band_claims_is_flyable_the_physics_accepts():
    """The contract. Every law geometry.Wing builds under the limits draws a
    chord inside the band — measured against the wing itself, not against a
    re-implementation of it."""
    box = geometry.chord_bounds(3)
    for lam in (1.0, 0.6, 0.35):
        c_root, semi = _wing_lengths(lam)
        for lim in (geometry.ChordLimits(c_min_m=0.55),
                    geometry.ChordLimits(c_max_m=1.35),
                    geometry.ChordLimits(rate_max_deg=9.0),
                    geometry.ChordLimits(c_min_m=0.5, c_max_m=1.4,
                                         rate_max_deg=12.0),
                    geometry.ChordLimits(c_min_m=0.5,
                                         trend="root_largest")):
            reach = geometry.chord_reach(box, lam, limits=lim, c_root=c_root,
                                         semi=semi)
            lo, hi = _chord(reach, c_root)
            tol = c_root * (TREND_TOL if lim.trend != "free" else BAND_TOL)
            n = 0
            for wing in _admitted(lam, lim, n=1500):
                n += 1
                c = wing.chord(reach.eta * semi)
                assert np.all(c <= hi + tol), (lam, lim)
                assert np.all(c >= lo - tol), (lam, lim)
            assert n > 0, (lam, lim)     # the limit sets have to admit some


def test_the_flyable_share_is_the_share_the_physics_admits():
    lam = 0.6
    c_root, semi = _wing_lengths(lam)
    for lim in (geometry.ChordLimits(c_min_m=0.5),
                geometry.ChordLimits(c_min_m=0.6, c_max_m=1.5),
                geometry.ChordLimits(rate_max_deg=7.0)):
        reach = geometry.chord_reach(geometry.chord_bounds(3), lam,
                                     limits=lim, c_root=c_root, semi=semi)
        built = sum(1 for _ in _admitted(lam, lim, n=2000, seed=3))
        assert reach.flyable_frac == pytest.approx(built / 2000.0, abs=0.03)


def test_an_impossible_limit_empties_the_band_rather_than_faking_one():
    lam = 0.6
    c_root, semi = _wing_lengths(lam)
    # the trapezoid's own tip chord is 0.75 m, and no area-preserving law in
    # the published box lifts the whole span above 1.3 m
    reach = geometry.chord_reach(geometry.chord_bounds(3), lam, c_root=c_root,
                                 semi=semi,
                                 limits=geometry.ChordLimits(c_min_m=1.3))
    assert reach.empty and reach.flyable_frac == 0.0
    assert np.all(np.isnan(reach.lo)) and np.all(np.isnan(reach.hi))


def test_a_law_can_rescue_a_baseline_the_limit_refuses():
    """The band is NOT "the baseline plus a wobble": the straight taper this
    one is drawn on breaks the minimum chord, and the band is the laws that
    lift its tip back over it."""
    lam = 0.6
    c_root, semi = _wing_lengths(lam)
    lim = geometry.ChordLimits(c_min_m=0.9)
    assert c_root * lam < 0.9                     # the baseline is refused
    reach = geometry.chord_reach(geometry.chord_bounds(3), lam, limits=lim,
                                 c_root=c_root, semi=semi)
    assert not reach.empty and reach.flyable_frac > 0.0
    assert _chord(reach, c_root)[0].min() >= 0.9 - 1e-9


def test_straight_taper_is_judged_by_the_limits_too():
    """order 0 — no chord law at all — is one planform, and it is refused or
    admitted by exactly the same rule."""
    lam = 0.6
    c_root, semi = _wing_lengths(lam)
    ok = geometry.chord_reach([], lam, c_root=c_root, semi=semi,
                              limits=geometry.ChordLimits(c_min_m=0.5))
    assert not ok.empty and ok.flyable_frac == 1.0
    bad = geometry.chord_reach([], lam, c_root=c_root, semi=semi,
                               limits=geometry.ChordLimits(c_min_m=0.9))
    assert bad.empty and bad.flyable_frac == 0.0


# ------------------------------------------------------- refusing to guess
def test_a_metre_without_a_length_to_measure_it_raises():
    box = geometry.chord_bounds(3)
    with pytest.raises(ValueError, match="needs the root chord"):
        geometry.chord_reach(box, 0.6,
                             limits=geometry.ChordLimits(c_min_m=0.5))
    with pytest.raises(ValueError, match="limit per metre of SPAN"):
        geometry.chord_reach(box, 0.6, c_root=1.25,
                             limits=geometry.ChordLimits(rate_max_deg=8.0))
    # ...but a trend needs no length, so it is drawn without one
    trend_only = geometry.chord_reach(
        box, 0.6, limits=geometry.ChordLimits(trend="root_largest"))
    assert trend_only.trend == "root_largest"


def test_the_trend_may_not_be_asked_twice_and_differently():
    box = geometry.chord_bounds(3)
    with pytest.raises(ValueError, match="asked twice"):
        geometry.chord_reach(box, 0.6, trend="root_largest",
                             limits=geometry.ChordLimits(
                                 trend="root_smallest"))
    # the same answer twice is not a disagreement
    both = geometry.chord_reach(box, 0.6, trend="root_largest",
                                limits=geometry.ChordLimits(
                                    trend="root_largest"))
    assert both.trend == "root_largest"


def test_the_flags_become_the_limits_in_one_place():
    """The shell draws the band from the same object the run is built with,
    so a flag cannot mean one thing to the picture and another to the run."""
    assert api.chord_limits_of({}) is None
    assert api.chord_limits_of(None) is None
    assert api.chord_limits_of({"mach": 0.0}) is None
    lim = api.chord_limits_of({"chord_min_m": 0.4, "chord_rate_max_deg": 20.0,
                               "chord_trend": "root_largest"})
    assert (lim.c_min_m, lim.c_max_m) == (0.4, None)
    assert lim.rate_max_deg == 20.0 and lim.trend == "root_largest"
    built = api.PROBLEM_SPECS["trim wing"].build(
        {}, {"chord_min_m": 0.4, "chord_rate_max_deg": 20.0,
             "chord_trend": "root_largest"}, None)
    assert built.problem.chord_limits == lim


# --------------------------------------------------------------- the shell
def test_the_panel_draws_the_band_under_the_live_limits(capsys):
    from gui.v3.app import assemble

    ctx = assemble("air")
    S = ctx.S
    ctx.render("wing", "box")

    def texts():
        return [getattr(e, "text", "") or ""
                for e in ctx.views[("wing", "box")].descendants()]

    assert not any("band holds" in t for t in texts())
    ctx.act("set_chord_limit_on", "chord_min_m", True)
    ctx.act("set_chord_limit", "chord_min_m", 0.5)
    assert S["wing"]["flags"]["chord_min_m"] == 0.5
    assert any("c ≥ 0.5 m" in t for t in texts()), texts()

    # ...and the number typed reaches the band without a re-render of the box
    ctx.act("set_chord_limit", "chord_min_m", 0.7)
    assert any("c ≥ 0.7 m" in t for t in texts()), texts()

    # a limit the baseline breaks says so rather than drawing a dashed line
    # through a planform the run refuses
    ctx.act("set_chord_limit", "chord_min_m", 0.9)
    assert any("dashed straight taper is itself refused" in t
               for t in texts()), texts()

    ctx.act("set_chord_limit_on", "chord_min_m", False)
    assert not any("band holds" in t for t in texts())
    err = capsys.readouterr().err
    assert "Traceback" not in err, err


def test_a_surface_whose_size_is_searched_says_it_cannot_draw_them(capsys):
    """A tandem's wings are sized inside the solver, so nothing in the shell
    can put a metre on their chord. The panel says that instead of drawing a
    band that quietly ignores a live limit."""
    from gui.v3.app import assemble

    ctx = assemble()
    ctx.act("accept_mission")
    ctx.act("set_lifting_system", "tandem")
    ctx.render("wing", "box")
    ctx.act("set_chord_limit_on", "chord_min_m", True)

    texts = [getattr(e, "text", "") or ""
             for e in ctx.views[("wing", "box")].descendants()]
    assert any("NOT holding the minimum chord" in t for t in texts), texts
    assert not any("band holds" in t for t in texts), texts
    err = capsys.readouterr().err
    assert "Traceback" not in err, err


def test_every_designed_surfaces_band_follows_the_one_limit(capsys):
    """One ChordLimits reaches every wing of a problem, so typing a limit has
    to move every band — not only the wing's, which is the one the handler
    used to redraw."""
    from gui.v3.app import assemble

    ctx = assemble()
    ctx.act("accept_mission")
    ctx.act("set_second_surface", True)
    ctx.render("wing", "box")
    ctx.act("set_chord_limit_on", "chord_min_m", True)
    ctx.act("set_chord_limit", "chord_min_m", 0.35)

    texts = [getattr(e, "text", "") or ""
             for e in ctx.views[("wing", "box")].descendants()]
    # the wing's panel and the tail's, both drawn under the same number
    assert sum("c ≥ 0.35 m" in t for t in texts) == 2, texts
    err = capsys.readouterr().err
    assert "Traceback" not in err, err
