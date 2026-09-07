"""How big the second surface is, and who says so.

Three things, and the first is the bug the other two answer.

``tail.S_T_BOUNDS`` and ``tail.L_T_BOUNDS`` were absolute — 0.5-3.0 m^2 of
tail on an arm of 3-8 m — because the family that needed them had one size.
Nothing in the model knows how big an aeroplane is except the wing it is
handed, so a 0.5 kg model (1.2 m span, 0.18 m^2) was searched over the same
box a 10 m one was: the FLOOR of the area band was 2.8x its whole wing, the
tail span exceeded the wing span at every point of it, and the arm put the
tail metres behind a 1.2 m aeroplane. Every one of those is "feasible" — the
static-margin gate has a floor and no ceiling — so the search drove to the
bottom of a box that was still absurd and reported it as an answer.

The bands are now fractions of the wing (``tail.area_band`` /
``tail.arm_band``), which are the published metres exactly at the reference
aeroplane, and the user can state the surface in the units they measure it
in: a span and a chord (``tail.TailLimits``), which convert to an area and
an aspect-ratio band with no approximation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

MODEL = {"b_m": 1.2, "S_m2": 0.18}
MODEL_MISSION = {"W_N": 4.905, "V": 12.0}

#: every air family whose design vector carries a tail area
TAIL_FAMILIES = ("tail", "tail (fixed arm)", "tail [free height]",
                 "tail [designed tail]")


def _row(built, label: str):
    return built.bounds[list(built.param_labels).index(label)]


def test_the_published_aeroplane_keeps_its_published_box():
    """The fractions ARE the published metres at the reference aeroplane, so
    no result taken over that box moves."""
    from aerobo import tail

    assert tail.area_band(tail.S_REF_M2) == (0.5, 3.0)
    assert tail.arm_band(tail.B_REF_M) == (3.0, 8.0)
    assert tail.default_arm(tail.B_REF_M) == 5.5
    assert tail.default_x_cg("conventional") == 0.05
    assert tail.default_x_cg("canard") == -0.25
    from aerobo import api
    for name in TAIL_FAMILIES:
        built = api.PROBLEM_SPECS[name].build({}, {}, None)
        assert tuple(_row(built, "S_t_m2")) == (0.5, 3.0), name
        if "l_t_m" in built.param_labels:
            assert tuple(_row(built, "l_t_m")) == (3.0, 8.0), name
        assert built.problem.x_cg == 0.05, name


def test_a_model_aeroplane_gets_a_model_aeroplane_s_tail():
    """The failure in the report, as a measurement: the tail must not come
    out bigger than the wing it trims."""
    from aerobo import api

    S, b = MODEL["S_m2"], MODEL["b_m"]
    for name in TAIL_FAMILIES:
        built = api.PROBLEM_SPECS[name].build(MODEL_MISSION, MODEL, None)
        s_lo, s_hi = _row(built, "S_t_m2")
        # the whole band is a stabiliser, not a second wing
        assert 0.0 < s_lo < s_hi < 0.5 * S, (name, s_lo, s_hi)
        # ...and at NO point of it is the tail wider than the wing
        assert float(np.sqrt(4.0 * s_hi)) < b, (name, s_hi)
        if "l_t_m" in built.param_labels:
            l_lo, l_hi = _row(built, "l_t_m")
            assert 0.0 < l_lo < l_hi <= b, (name, l_lo, l_hi)
        else:
            assert 0.0 < built.problem.l_t_fixed <= b, name
        # the CG is a station on the chord, not an absolute length
        assert abs(built.problem.x_cg) < 0.5 * (S / b), name


def test_the_size_the_box_shows_is_the_size_it_searches():
    """The shell reads the tail rows off the BUILT problem, so a design box
    on a model aeroplane cannot paint the 10 m aeroplane's band 'default'."""
    from gui.v3 import config
    from gui.v3.app import assemble

    ctx = assemble()
    ctx.act("accept_mission")
    ctx.act("set_choice", "tail", True)
    S = ctx.S
    assert config.effective_bounds(S)["S_t_m2"][0] == [0.5, 3.0]
    S["wing"]["choices"]["span_m"] = MODEL["b_m"]
    S["wing"]["choices"]["area_m2"] = MODEL["S_m2"]
    shown = config.effective_bounds(S)
    assert shown["S_t_m2"][0][1] < 0.1, shown["S_t_m2"]
    assert shown["l_t_m"][0][1] <= MODEL["b_m"], shown["l_t_m"]
    # ...and it is still the SOLVER's own box, not an override the user
    # never typed
    assert shown["S_t_m2"][1] == "default"


def test_a_widened_tail_row_reaches_the_problem_and_not_just_the_sampler():
    """The band the shell shows has to BE the band the problem checks.

    Without the carrier the row moved in the design box alone: every draw
    outside the family's own band came back "bounds violation" and the run
    was a landscape of penalties reported as a search.
    """
    from aerobo import api

    wide = {"S_t_m2": [0.5, 3.0], "l_t_m": [3.0, 8.0]}
    built = api.PROBLEM_SPECS["tail"].build(MODEL_MISSION, MODEL, wide)
    assert tuple(_row(built, "S_t_m2")) == (0.5, 3.0)
    x = np.array([0.6, 0.0, -2.0, 1.75, 5.5])
    assert built.evaluate(x).get("reason") != "bounds violation"
    assert built.evaluate(x)["score"] > 0.0


def test_a_span_limit_narrows_the_box_it_is_equivalent_to():
    """b = sqrt(AR S) both ways: a span cap on a fixed-AR tail IS an area
    cap, exactly, and the box says so instead of refusing draws it could
    have avoided making."""
    from aerobo import api, tail

    built = api.PROBLEM_SPECS["tail"].build(
        MODEL_MISSION, dict(MODEL, tail_span_max_m=0.45), None)
    s_lo, s_hi = _row(built, "S_t_m2")
    assert s_hi == pytest.approx(0.45 ** 2 / tail.TAIL_AR)
    # every point of the narrowed box really does meet the limit
    for s_t in np.linspace(s_lo, s_hi, 9):
        assert float(np.sqrt(tail.TAIL_AR * s_t)) <= 0.45 + 1e-12


def test_a_chord_limit_is_the_reynolds_number_it_stands_for():
    from aerobo import api, tail

    built = api.PROBLEM_SPECS["tail"].build(
        MODEL_MISSION, dict(MODEL, tail_chord_min_m=0.08), None)
    s_lo, _ = _row(built, "S_t_m2")
    assert s_lo == pytest.approx(tail.TAIL_AR * 0.08 ** 2)


def test_a_limit_a_narrowed_box_cannot_express_is_refused_per_candidate():
    """With the aspect ratio SEARCHED the box is the smallest one containing
    what the limits allow, so the corners it adds have to be refused where
    they are drawn — the rule geometry.ChordLimits already follows."""
    from aerobo import api

    built = api.PROBLEM_SPECS["tail [designed tail]"].build(
        MODEL_MISSION, dict(MODEL, tail_span_max_m=0.45), None)
    lab = list(built.param_labels)
    x = np.array([0.5 * (lo + hi) for lo, hi in built.bounds])
    x[lab.index("S_t_m2")] = _row(built, "S_t_m2")[1]
    x[lab.index("AR_t")] = _row(built, "AR_t")[1]
    span = float(np.sqrt(x[lab.index("AR_t")] * x[lab.index("S_t_m2")]))
    assert span > 0.45, span             # the corner the box could not cut
    res = built.evaluate(x)
    assert res["score"] == -100.0
    assert "tail span" in str(res["reason"]), res["reason"]


def test_an_impossible_pair_of_limits_says_which_two():
    from aerobo import api

    with pytest.raises(ValueError, match="tail span limits"):
        api.PROBLEM_SPECS["tail"].build(
            MODEL_MISSION,
            dict(MODEL, tail_span_max_m=0.20, tail_chord_min_m=0.30), None)


def test_the_tail_is_not_refused_by_the_wing_s_chord_limits():
    """One question, one surface. A 0.5 m minimum chord meant for a 10 m
    wing used to be handed to the tail as well, where it refused every
    stabiliser correctly smaller than it."""
    from aerobo import api

    flags = {"chord_min_m": 0.5, "chord_max_m": 2.5}
    built = api.PROBLEM_SPECS["tail [designed tail]"].build({}, flags, None)
    lab = list(built.param_labels)
    x = np.array([0.5 * (lo + hi) for lo, hi in built.bounds])
    x[lab.index("S_t_m2")] = 0.8
    x[lab.index("AR_t")] = 6.0           # tail chord 0.365 m — under the WING's
    res = built.evaluate(x)
    assert res["score"] > 0.0, res.get("reason")


def test_untouched_limits_are_the_published_problem():
    from aerobo import api

    for name in TAIL_FAMILIES:
        plain = api.PROBLEM_SPECS[name].build({}, {}, None)
        assert plain.problem.tail_limits is None, name
        assert plain.problem.s_t_bounds_m2 is None, name
