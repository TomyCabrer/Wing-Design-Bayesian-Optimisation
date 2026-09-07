"""The Results stage says what the aeroplane WEIGHS — or why it cannot.

Asked for as "in summary we should see the total weight". The answer is not
one readout, because only the SIZE modifier weighs anything: 656 of the 2359
registered problems compute ``W_total_N``, and on the other 1703 the span and
area are fixed, ``CL_target`` is stated, and the wing's mass never enters the
objective at all.

So a panel that drew a dash on those would be saying "this number is missing"
about a question the run did not ask. It answers two ways instead, and this
file pins both — including that the "no weight here" branch names a family
that DOES weigh, rather than leaving a dead end.
"""
from __future__ import annotations

import numpy as np
import pytest

from aerobo import api

SIZED = "trim wing + free planform"
FIXED = "tail + winglet"


def _breakdown(name: str) -> dict:
    cfg = api.RunConfig(problem_name=name, budget=4, seed=0)
    built = api.PROBLEM_SPECS[name].build({}, {}, None)
    return api.design_report(cfg, built.bounds.mean(axis=1))["breakdown"]


# ------------------------------------------------- there IS a weight to show

def test_a_sized_family_reports_all_three_weights():
    bd = _breakdown(SIZED)
    for key in ("W_total_N", "W_wing_N", "W_fixed_N"):
        assert isinstance(bd.get(key), (int, float)), f"{key} missing"
        assert bd[key] > 0.0


def test_the_total_is_the_sum_of_its_two_parts():
    """Not a restatement: the panel shows three numbers and a percentage, and
    a total that is not its parts' sum makes every one of them a lie."""
    bd = _breakdown(SIZED)
    assert bd["W_total_N"] == pytest.approx(
        bd["W_wing_N"] + bd["W_fixed_N"], rel=1e-9)


def test_the_wing_is_a_real_share_of_the_total():
    """The percentage the panel prints has to mean something — a wing at
    0.01 % or 99 % of the total would be a sizing that failed to converge."""
    bd = _breakdown(SIZED)
    frac = bd["W_wing_N"] / bd["W_total_N"]
    assert 0.05 < frac < 0.95


def test_the_area_the_wing_loading_is_taken_over_is_reported():
    """The sized families call it ``S_m2`` — it is a design VARIABLE, not the
    reference constant the fixed families carry as ``S``. Reading the wrong
    one silently drops the row."""
    bd = _breakdown(SIZED)
    S = bd.get("S_m2")
    assert isinstance(S, (int, float)) and S > 0.0
    ws = bd["W_total_N"] / S
    assert 50.0 < ws < 20000.0, f"wing loading {ws:.1f} Pa is not an aeroplane"


# ------------------------------------------- ...and where there is not one

def test_a_fixed_size_family_reports_no_weight_at_all():
    """The fact the panel's second branch exists for. Not a gap to be filled
    with a zero — a question this family does not ask."""
    bd = _breakdown(FIXED)
    for key in ("W_total_N", "W_wing_N", "W_fixed_N"):
        assert bd.get(key) is None


def test_every_family_without_a_weight_has_a_sized_twin_to_name():
    """The address on the refusal. ``api.with_modifiers`` is asked rather
    than a name being spelled out, so the panel can only ever offer a problem
    that is registered."""
    for name in (FIXED, "tail", "tandem", "winglet"):
        twin = api.with_modifiers(
            name, set(api.modifiers_of(name)) | {"size"})
        assert twin, f"{name} has no sized twin to point a user at"
        assert twin != name
        assert twin in api.PROBLEM_SPECS
        assert "size" in api.modifiers_of(twin)


def test_the_named_twin_actually_weighs_something():
    """A twin that is named and then reports no weight is a worse dead end
    than saying nothing."""
    twin = api.with_modifiers(FIXED, set(api.modifiers_of(FIXED)) | {"size"})
    built = api.PROBLEM_SPECS[twin].build({}, {}, None)
    raw = built.evaluate(np.asarray(built.bounds.mean(axis=1), float))
    assert isinstance(raw.get("W_total_N"), (int, float))
    assert raw["W_total_N"] > 0.0


# ------------------------------------------------------- ON THE PAGE

def _summary_text(problem: str) -> str:
    """Every label and hint in a rendered Results summary, joined.

    Driven through the real view, because "we should see the total weight" is
    a claim about the SCREEN — the engine has carried these numbers all along
    and printed them nowhere, which is the whole defect.
    """
    import numpy as np

    from gui.v3.app import assemble

    ctx = assemble()
    built = api.PROBLEM_SPECS[problem].build({}, {}, None)
    x = np.asarray(built.bounds.mean(axis=1), float)
    cfg = api.RunConfig(problem_name=problem, budget=4, seed=0)
    rep = api.design_report(cfg, x)
    ctx.S["run"]["record"] = {
        "breakdown": rep["breakdown"], "best_x": list(x),
        "param_labels": list(built.param_labels),
        "bounds": [list(r) for r in built.bounds],
        "n_evals": 1, "feasible": True, "wall_time_s": 0.0,
        "best_score": rep["breakdown"].get("score", 0.0),
        "config": {"problem_name": problem, "optimiser": "bo",
                   "budget": 4, "seed": 0, "flags": {}}}
    ctx.S["run"]["report"] = rep
    ctx.render("results", "summary")

    def _walk(el):
        for d in el.descendants():
            for attr in ("text", "_text"):
                v = getattr(d, attr, None)
                if isinstance(v, str) and v:
                    yield v
            props = getattr(d, "_props", {}) or {}
            for key in ("label", "caption"):
                v = props.get(key)
                if isinstance(v, str) and v:
                    yield v

    return " ".join(_walk(ctx.views[("results", "summary")]))


def test_a_sized_run_puts_the_TOTAL_WEIGHT_on_the_page():
    text = _summary_text(SIZED)
    assert "Weight" in text
    assert "total" in text
    bd = _breakdown(SIZED)
    # the number itself, in the units the panel prints it in
    assert f"{bd['W_total_N']:,.1f}" in text, text[:400]
    assert "kg" in text
    assert "wing loading" in text
    # ...over the area the SIZING SOLVED (``S_m2``), not the one the rebuilt
    # planform integrates to. They differ by 0.04 % here, which is 0.1 Pa —
    # enough to print differently, and the sized one is the number
    # ``sizing.sized_state`` actually closed W = (W/S) S against.
    ws = bd["W_total_N"] / bd["S_m2"]
    assert f"{ws:,.1f} Pa" in text, (
        f"expected {ws:,.1f} Pa (over S_m2); the panel is taking its area "
        f"from somewhere else")


def test_a_fixed_size_run_says_WHY_there_is_no_weight_and_where_to_get_one():
    text = _summary_text(FIXED)
    assert "Weight" in text
    assert "FIXED size" in text or "fixed" in text.lower()
    twin = api.with_modifiers(FIXED, set(api.modifiers_of(FIXED)) | {"size"})
    assert twin in text, "the panel did not name a family that does weigh"


def test_the_split_between_weighing_and_not_is_the_size_modifier():
    """One rule, not a list of family names kept in step by hand."""
    for name, weighs in ((SIZED, True), (FIXED, False)):
        assert ("size" in api.modifiers_of(name)) is weighs
