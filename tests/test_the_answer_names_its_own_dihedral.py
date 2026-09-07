"""Meeting an anhedral wing should not mean reading a signed number.

A run that searched the wing's cant came back with ``wing_dihedral_deg``
sitting in the design-vector table — a signed number in a list of signed
numbers. Nothing anywhere in stage 4 said which way that points, that it is
the only wing-side ``Cl_beta`` there is, or whether the aeroplane's spiral
converges. The whole lateral half of the answer had a read-out in exactly
one place: stage 3's advice card, which is about a DESIGN BOX and not about
this answer.

So the user met an anhedral wing by inference. This file pins that they
cannot any more:

* the verdict itself (``api.lateral_verdict``) over its five states, read
  off a breakdown and re-deriving nothing;
* that it appears in the Results summary for a REAL run;
* that a family with no cant gets no card rather than a row of dashes;
* and that a divergent spiral is coloured as bad, not printed as a number
  the reader has to interpret.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aerobo import api                                          # noqa: E402


# ------------------------------------------------------- the verdict itself

def test_a_family_with_no_cant_gets_no_verdict_at_all():
    v = api.lateral_verdict({"LoD": 20.0, "CL": 0.5})
    assert v["status"] == "not_applicable"
    assert v["says"] == ""


def test_an_anhedral_wing_is_named_as_anhedral_not_shown_as_a_sign():
    v = api.lateral_verdict({"wing_dihedral_deg": -10.0, "sweep_deg": 0.0})
    assert v["status"] == "not_measured"
    assert v["level"] == "warn"
    assert "ANHEDRAL" in v["says"]
    assert "10.00 deg BELOW" in v["says"], v["says"]
    # ...and it says the consequence, which is the thing a sign does not
    assert "rolls INTO a sideslip" in v["says"]


def test_a_run_that_priced_nothing_lateral_says_so_rather_than_nothing():
    """``not_measured`` is an answer. Silence would read as "converges"."""
    v = api.lateral_verdict({"wing_dihedral_deg": 7.0})
    assert v["status"] == "not_measured"
    assert "NOT known from this run" in v["says"]
    assert "spiral" in v["says"]


def test_a_divergent_spiral_is_called_divergent_and_ranks_worst():
    v = api.lateral_verdict({"wing_dihedral_deg": 2.0, "Cn_beta": 0.11,
                             "spiral_margin": -0.0126,
                             "spiral_theta0_deg": 4.38})
    assert v["status"] == "diverges"
    assert v["level"] == "bad"
    assert "DIVERGES" in v["says"]
    # the attitude the criterion was measured at travels with it: the
    # classical form drops that term and runs 2.13 deg of dihedral
    # optimistic, so a margin without it cannot be reproduced
    assert "+4.38 deg" in v["says"], v["says"]


def test_a_negative_yaw_stiffness_is_refused_in_the_criterion_s_own_words():
    """ONE author for that sentence — the guard the composite applies."""
    from aerobo import wing_score

    bd = {"wing_dihedral_deg": 0.0, "Cn_beta": -0.19599,
          "spiral_margin": +0.02016}
    v = api.lateral_verdict(bd)
    assert v["status"] == "refused"
    assert wing_score.spiral_refusal(bd) in v["says"]
    # a POSITIVE margin, and it is still not called convergent
    assert "converges" not in v["says"]


def test_anhedral_is_still_named_when_the_spiral_converges():
    """A convergent spiral bought elsewhere does not make tips-down
    unremarkable — it is still the destabilising sign of the wing's own
    contribution, and still a row nothing priced."""
    v = api.lateral_verdict({"wing_dihedral_deg": -4.0, "Cn_beta": 0.11,
                             "spiral_margin": +0.004})
    assert v["status"] == "converges"
    assert v["level"] == "warn"
    assert "ANHEDRAL" in v["says"]


def test_a_nan_is_not_a_number():
    v = api.lateral_verdict({"wing_dihedral_deg": float("nan")})
    assert v["status"] == "not_applicable"


# ------------------------------------------------- and it reaches the page

def _summary(problem: str, record_patch: dict | None = None) -> dict:
    """Render stage 4's summary and collect what it drew.

    Returns ``{"kv": [(key, value)], "hints": [(text, level)]}``.
    """
    import gui.v3.widgets as widgets
    from gui.v3.app import assemble

    kv, hints = [], []
    real_kv, real_hint = widgets.kv, widgets.hint

    def kv_spy(key, value, **kw):
        kv.append((str(key), str(value)))
        return real_kv(key, value, **kw)

    def hint_spy(text, level="", *a, **kw):
        hints.append((str(text), str(level)))
        return real_hint(text, level, *a, **kw)

    widgets.kv, widgets.hint = kv_spy, hint_spy
    try:
        ctx = assemble()
        ctx.act("accept_mission")
        r = api.run(api.RunConfig(problem_name=problem, optimiser="sobol",
                                  budget=8, seed=0))
        rec = r.to_dict()
        if record_patch:
            rec.setdefault("breakdown", {}).update(record_patch)
        ctx.S["wing"]["problem"] = problem
        ctx.S["run"]["record"] = rec
        ctx.render("results", "summary")
    finally:
        widgets.kv, widgets.hint = real_kv, real_hint
    return {"kv": kv, "hints": hints}


def test_a_real_searched_cant_run_names_its_dihedral_on_the_summary():
    out = _summary("tail [free cant]")
    keys = [k for k, _ in out["kv"]]
    assert "wing dihedral" in keys, keys
    assert "wing sweep" in keys, keys
    said = " ".join(t for t, _ in out["hints"])
    assert ("ANHEDRAL" in said) or ("deg UP" in said) or ("planar" in said), \
        said[-500:]


def test_the_lateral_card_is_absent_where_there_is_no_cant():
    """A family that states none gets no card — not a row of dashes."""
    out = _summary("trim wing")
    keys = [k for k, _ in out["kv"]]
    assert "wing dihedral" not in keys, keys


def test_a_measured_spiral_reaches_the_summary_from_the_run_s_own_deck():
    """STATED IS NOT FLOWN: the margin shown comes from an evaluation that
    BUILT the lateral deck, not from a number typed into the record."""
    name = "tail [free cant]"
    built = api.PROBLEM_SPECS[name].build({}, {}, None)
    built.problem.lateral = True
    bnds = built.problem.bounds
    raw = built.evaluate(0.5 * (bnds[:, 0] + bnds[:, 1]))
    assert raw.get("feasible"), raw.get("reason")
    assert "spiral_margin" in raw, sorted(raw)
    lateral = {k: float(raw[k]) for k in api.LATERAL_KEYS if k in raw}

    out = _summary(name, record_patch=lateral)
    kv = dict(out["kv"])
    assert "spiral margin" in kv, sorted(kv)
    # the SAME number, to the six significant figures the read-out prints
    assert float(kv["spiral margin"]) == pytest.approx(
        raw["spiral_margin"], rel=1e-6), (kv["spiral margin"],
                                          raw["spiral_margin"])
    assert "yaw stiffness Cn_β" in kv
    said = " ".join(t for t, _ in out["hints"])
    assert ("spiral converges" in said) or ("SPIRAL DIVERGES" in said) \
        or ("refused" in said), said[-500:]


def test_a_sweep_alone_is_not_a_lateral_answer():
    """Every wing-like family records a ``sweep_deg`` — a plain wing and a
    winglet included — and none of those has a fin, a spiral, or anything
    that could price a dihedral. A card there would end in "weight `spiral`
    on stage 3's objective card", which is a recommendation that does not
    clear the gate it cites.
    """
    assert api.lateral_verdict({"sweep_deg": 0.0, "LoD": 20.0})["status"] \
        == "not_applicable"
    # ...and a design that states a PLANAR cant is still answered: zero is
    # an answer to the question, and "no dihedral effect at all" is the
    # thing a reader most needs told.
    v = api.lateral_verdict({"wing_dihedral_deg": 0.0, "sweep_deg": 5.0})
    assert v["status"] == "not_measured"
    assert "planar" in v["says"]


def test_a_plain_wing_run_draws_no_lateral_card():
    """The same thing through the page, on a family that really does
    record a sweep and nothing else."""
    out = _summary("winglet")
    assert "wing sweep" not in [k for k, _ in out["kv"]], out["kv"]


def test_the_offer_lands_on_the_card_that_holds_the_control():
    """A button whose label promises a control has to arrive at it.

    The objective card is on stage 3's SOLVER view; the cant card's own
    "weight it in the objective" button goes there, and this one must too —
    a recommendation that lands on another page is the same failure as one
    citing a gate it does not clear.
    """
    from gui.v3.app import assemble

    ctx = assemble()
    ctx.act("accept_mission")
    problem = "tail [free cant]"
    r = api.run(api.RunConfig(problem_name=problem, optimiser="sobol",
                              budget=8, seed=0))
    ctx.S["wing"]["problem"] = problem
    ctx.S["run"]["record"] = r.to_dict()
    ctx.render("results", "summary")
    view = ctx.views[("results", "summary")]
    btn = next((e for e in view.descendants()
                if type(e).__name__ == "Button"
                and "weight the spiral" in (getattr(e, "text", "") or "")),
               None)
    assert btn is not None, "the unpriced verdict offers no way to price it"
    for listener in (getattr(btn, "_event_listeners", None) or {}).values():
        if listener.type == "click":
            listener.handler(None)
            break
    else:
        raise AssertionError("the button has no click handler")
    assert ctx.S["ui"]["selected"] == "wing"
    assert ctx.S["ui"]["tab"]["wing"] == "solver", ctx.S["ui"]["tab"]
