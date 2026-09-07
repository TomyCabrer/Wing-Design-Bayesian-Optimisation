"""The chord distribution, stated in metres and degrees.

The design box's chord rows are COEFFICIENTS of the area-preserving law
(``chord_k1..k3``). They are the right variables for a solver and the wrong
ones for a person: nobody knows what ``chord_k2 = -0.31`` draws, and the four
things an engineer actually has to hold —

    a minimum chord, a maximum chord, a maximum local taper ANGLE, and which
    end of the wing is the wide one

— are all NONLINEAR in those coefficients once the area rescale is applied,
so no box on ``chord_k*`` can state any of them.

They are therefore VALUES (``api.CHORD_LIMIT_KEYS`` →
``geometry.ChordLimits``), checked on the chord distribution each candidate
actually draws, at the one place every family builds its planform. The
contract:

* absent — the default — nothing moves: every published run is bit-for-bit;
* a candidate that breaks a live limit is an IN-CONTRACT failure (the penalty
  contract), never quietly reshaped into something the user did not ask for;
* they apply with or without a chord law: "no chord below 0.4 m" constrains a
  trapezoid's taper exactly as meaningfully as it constrains a free law;
* they are declared on every problem that HAS a planform, read off the
  declared design vector, and on no problem that does not.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aerobo import api, geometry     # noqa: E402

X_TRIM = np.array([0.5, 1.0, -2.0])


def _built(name: str, flags: dict | None = None):
    return api.PROBLEM_SPECS[name].build({}, flags or {}, None)


# ------------------------------------------------------- the limits themselves
def test_a_limit_is_measured_on_the_chord_the_candidate_draws():
    w = geometry.Wing(b=10.0, S=10.0, taper=0.5)
    y = np.linspace(0.0, 5.0, 129)
    c = w.chord(y)
    assert c.max() == pytest.approx(4.0 / 3.0)
    assert c.min() == pytest.approx(2.0 / 3.0)

    assert geometry.ChordLimits().violation(y, c) is None
    assert geometry.ChordLimits(c_min_m=0.5).violation(y, c) is None
    assert "below the 0.9" in geometry.ChordLimits(
        c_min_m=0.9).violation(y, c)
    assert "above the 1.2" in geometry.ChordLimits(
        c_max_m=1.2).violation(y, c)


def test_the_rate_limit_is_the_local_taper_angle():
    """atan(|dc/dy|): a straight taper has one value of it, and it is the one
    the trapezoid's own geometry gives."""
    w = geometry.Wing(b=10.0, S=10.0, taper=0.5)
    y = np.linspace(0.0, 5.0, 129)
    dcdy = (w.chord(np.array([5.0]))[0] - w.chord(np.array([0.0]))[0]) / 5.0
    angle = np.rad2deg(np.arctan(abs(dcdy)))
    assert angle == pytest.approx(7.5946, abs=1e-3)

    lim = geometry.ChordLimits(rate_max_deg=angle + 0.5)
    assert lim.violation(y, w.chord(y)) is None
    tight = geometry.ChordLimits(rate_max_deg=angle - 0.5)
    assert "above the" in tight.violation(y, w.chord(y))


def test_the_trend_says_which_end_is_the_wide_one():
    tapered = geometry.Wing(b=10.0, S=10.0, taper=0.5)
    y = np.linspace(0.0, 5.0, 129)
    c = tapered.chord(y)
    assert geometry.ChordLimits(trend="root_largest").violation(y, c) is None
    assert "root to be the smallest" in geometry.ChordLimits(
        trend="root_smallest").violation(y, c)

    # ...and a rectangle satisfies BOTH: the trend test has to tolerate the
    # area rescale's last-bit wobble, not just exact equality
    rect = geometry.Wing(b=10.0, S=10.0, taper=1.0)
    for trend in ("root_largest", "root_smallest"):
        assert geometry.ChordLimits(trend=trend).violation(
            y, rect.chord(y)) is None


def test_an_impossible_limit_set_is_refused_at_construction():
    with pytest.raises(ValueError, match="unknown chord trend"):
        geometry.ChordLimits(trend="tapered")
    with pytest.raises(ValueError, match="must be > 0"):
        geometry.ChordLimits(c_min_m=0.0)
    with pytest.raises(ValueError, match="exceeds c_max_m"):
        geometry.ChordLimits(c_min_m=2.0, c_max_m=1.0)
    with pytest.raises(ValueError, match="rate_max_deg must be in"):
        geometry.ChordLimits(rate_max_deg=95.0)


def test_the_wing_refuses_a_planform_that_breaks_them():
    lim = geometry.ChordLimits(c_min_m=0.9)
    with pytest.raises(ValueError, match="chord limits"):
        geometry.Wing(b=10.0, S=10.0, taper=0.5, chord_limits=lim)
    # ...and accepts the one that does not
    assert geometry.Wing(b=10.0, S=10.0, taper=0.9,
                         chord_limits=lim).c_root > 0.9


# ------------------------------------------------------------- through the api
def test_absent_the_flags_nothing_moves():
    plain = _built("trim wing")
    assert plain.problem.chord_limits is None
    assert plain.evaluate(X_TRIM)["LoD"] == pytest.approx(
        _built("trim wing", {"mach": 0.0}).evaluate(X_TRIM)["LoD"],
        rel=0, abs=0)


def test_a_violating_candidate_scores_the_penalty_not_an_exception():
    built = _built("trim wing", {"chord_min_m": 1.2})
    out = built.evaluate(X_TRIM)
    assert out["feasible"] is False
    assert "chord limits" in out["reason"]
    assert built.callable(X_TRIM) == api.PENALTY


def test_a_satisfied_limit_changes_nothing_about_the_score():
    """A limit is a REFUSAL, not a modelling change: inside it, the same
    candidate scores exactly what it scored before."""
    free = _built("trim wing").evaluate(X_TRIM)
    held = _built("trim wing", {"chord_min_m": 0.5, "chord_max_m": 2.0,
                                "chord_rate_max_deg": 20.0,
                                "chord_trend": "root_largest"}
                  ).evaluate(X_TRIM)
    assert held["LoD"] == pytest.approx(free["LoD"], rel=0, abs=0)


def test_the_limits_reach_the_chord_law_families_too():
    name = "wing (free chord law)"
    # a rectangular baseline with a law that GROWS outboard
    x = np.array([1.0, 0.0, 0.0, 0.5, 0.0, 0.0])
    loose = _built(name).evaluate(x)
    assert loose["feasible"]
    # that law bulges the chord outboard, so "the root is the largest chord"
    # is exactly what refuses it
    held = _built(name, {"chord_trend": "root_largest"}).evaluate(x)
    assert held["feasible"] is False
    assert "root to be the largest" in held["reason"]


def test_every_family_with_a_planform_declares_them_and_honours_them():
    declared = {n for n, sp in api.PROBLEM_SPECS.items()
                if set(api.CHORD_LIMIT_KEYS) <= set(sp.flags)}
    # read off the DECLARED design vector, never off a family list
    expect = {n for n, sp in api.PROBLEM_SPECS.items()
              if any(str(lbl) == "taper" or str(lbl).startswith("taper_")
                     for lbl in sp.param_labels)}
    assert declared == expect
    # the one problem left out is the one that should be
    assert "airfoil (section)" in set(api.PROBLEM_SPECS) - declared

    def carries(obj, depth=0):
        if depth > 3 or obj is None:
            return False
        if getattr(obj, "chord_limits", None) is not None:
            return True
        return any(carries(getattr(obj, a, None), depth + 1)
                   for a in ("foil", "wing_tail", "_wing_prob", "problem"))

    # a declared flag that never reaches a planform would be a menu that lies
    for name in ("hydrofoil", "hydrofoil + elevator", "tandem",
                 "car rear wing", "tail", "winglet_capped",
                 "wing + airfoil (XFOIL)", "free planform (aircraft)",
                 "tandem (nonplanar) + winglets"):
        built = api.PROBLEM_SPECS[name].build({}, {"chord_min_m": 0.01}, None)
        assert carries(built.problem), name


def test_a_water_family_refuses_a_foil_chord_it_cannot_build():
    x = np.array([0.6, 1.0, -1.0, 0.12, 0.4, 12.0])
    free = _built("hydrofoil").evaluate(x)
    assert free["feasible"]
    held = _built("hydrofoil", {"chord_min_m": 0.5}).evaluate(x)
    assert held["feasible"] is False
    assert "chord limits" in held["reason"]


# ------------------------------------------------------------------- the shell
def test_the_stage_offers_the_limits_and_the_run_carries_them(capsys):
    from gui.v3 import config
    from gui.v3.app import assemble

    ctx = assemble("air")
    S = ctx.S
    ctx.render("wing", "box")
    texts = [getattr(e, "text", "") or ""
             for e in ctx.views[("wing", "box")].descendants()]
    assert any("minimum chord" in t for t in texts), texts
    assert any("max chord rate" in t for t in texts), texts

    ctx.act("set_chord_limit_on", "chord_min_m", True)
    assert S["wing"]["flags"]["chord_min_m"] > 0.0
    ctx.act("set_chord_limit", "chord_min_m", 0.55)
    assert config.cfg_dict(S)["flags"]["chord_min_m"] == 0.55

    ctx.act("set_chord_trend", "root_largest")
    assert config.cfg_dict(S)["flags"]["chord_trend"] == "root_largest"
    # ...and "free" is STATED, not expressed by the key's absence. The shell
    # opens on `root_largest` (config.CHORD_TREND), so a popped key would
    # read straight back as the default that was just left and "free" would
    # be unselectable. What matters is the value the SOLVER receives.
    ctx.act("set_chord_trend", "free")
    assert config.cfg_dict(S)["flags"]["chord_trend"] == "free"

    ctx.act("set_chord_limit_on", "chord_min_m", False)
    assert "chord_min_m" not in config.cfg_dict(S)["flags"]

    err = capsys.readouterr().err
    assert "Traceback" not in err, err


def test_switching_a_limit_on_cannot_refuse_the_wing_it_opens_on(capsys):
    """The default value is read off the planform actually flown, so turning
    a limit on is never a refusal of the design already on screen."""
    from gui.v3 import config
    from gui.v3.app import assemble

    ctx = assemble("air")
    S = ctx.S
    for key in ("chord_min_m", "chord_max_m", "chord_rate_max_deg"):
        ctx.act("set_chord_limit_on", key, True)
    cfg = config.build_cfg(S)
    built = api.PROBLEM_SPECS[cfg.problem_name].build(
        cfg.mission_kwargs, cfg.flags, cfg.bounds_overrides)
    mid = built.bounds.mean(axis=1)
    out = built.evaluate(mid)
    assert out["feasible"], out["reason"]

    err = capsys.readouterr().err
    assert "Traceback" not in err, err
