"""Maximise downforce PLUS drag — and state the limit where the law is chosen.

Two asks, one card.

1. ``downforce_plus_drag`` is a seventh car objective and the only one that
   ADDS the two forces instead of trading them: ``F_z + D``, in newtons. It is
   the total aerodynamic load the wing puts into the car — what the mount
   carries, and what a braking-limited case wants, since the drag retards the
   car directly and the downforce retards it through the tyres.

2. Every car objective is one half of a trade whose other half is a LIMIT, and
   the two were asked a whole TAB apart in V3: the Maximise select on the Wing
   type card, the drag ceiling and the downforce floor under the design box. So
   picking "efficiency" and looking for where to say how much downforce is
   worth having found a SENTENCE pointing elsewhere. A pointer is not a way to
   state a number. The rows have moved to the select; these tests are that they
   moved rather than multiplied.

The first ask comes with a warning that has to be measured rather than felt:
``F_z + D`` RIDES ITS BOUNDS. More area at the same CZ makes more of both
terms, so the answer spends the whole area row. That is not the ``cz`` defect
(which moves OPPOSITE to the physics, being maximised by shrinking the area it
is referenced to) — here the preference is the physics, and the area band is a
regulation. So it is shipped, and the card says so before it is picked.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aerobo import api                                          # noqa: E402
from aerobo import carwing, carwing_multi, endplate             # noqa: E402

LAW = "downforce_plus_drag"

#: every registered family that scores a car objective, addressed the way the
#: registry declares them rather than listed here (api.car_pareto_families is
#: the same derivation, and exists for the same reason).
CAR_FAMILIES = ("car rear wing", "car rear wing + endplates",
                "car rear wing (two-element)")


def _built(name: str, **flags):
    return api.PROBLEM_SPECS[name].build({}, dict(flags), None)


def _centre(built):
    B = np.asarray(built.bounds)
    return 0.5 * (B[:, 0] + B[:, 1])


# ==================================================================== the law


def test_the_score_is_the_two_forces_added():
    """The whole content of the objective, on every family that offers it —
    and asserted against the REPORT's own two forces, so a scorer that
    quietly changed which drag it adds fails here."""
    for name in CAR_FAMILIES:
        b = _built(name, car_objective=LAW)
        out = b.evaluate(_centre(b))
        assert out["feasible"], (name, out["reason"])
        assert out["objective"] == LAW
        assert out["score"] == pytest.approx(
            float(out["downforce_N"]) + float(out["drag_N"])), name
        # ...and it is a DIFFERENT question from the six already there: the
        # sum ranks designs no other scalar here ranks the same way
        assert out["score"] > float(out["downforce_N"]), name
        assert out["score"] > -float(out["drag_N"]), name


def test_it_asks_something_none_of_the_others_ask():
    """Not a relabelling. Two designs the sum ORDERS ONE WAY and efficiency
    orders the other — which is what says the seventh entry is a seventh
    question rather than a sixth one renamed."""
    b = _built("car rear wing")
    B = np.asarray(b.bounds)
    lab = list(b.param_labels)
    i_a, i_s = lab.index("alpha_deg"), lab.index("S_m2")
    x = _centre(b)
    small = np.array(x, dtype=float)
    big = np.array(x, dtype=float)
    big[i_a] = B[i_a, 1]              # more incidence: more of both, worse CZ/CD
    big[i_s] = B[i_s, 1]              # ...and more area
    a, c = b.evaluate(small), b.evaluate(big)
    assert a["feasible"] and c["feasible"]
    sum_a = a["downforce_N"] + a["drag_N"]
    sum_c = c["downforce_N"] + c["drag_N"]
    assert sum_c > sum_a                     # the sum prefers the big one...
    assert c["efficiency"] < a["efficiency"]  # ...and efficiency the small one


def test_every_car_family_declares_it_and_can_score_it():
    """The three objective tables are three different SETS on purpose, and a
    name in a table with no scorer behind it is the KeyError-from-inside-an-
    optimiser defect those tables were written to close."""
    assert LAW in carwing.CAR_OBJECTIVES
    assert LAW in endplate.ENDPLATE_OBJECTIVES     # derived from the above
    assert LAW in carwing_multi.CAR_MULTI_OBJECTIVES
    for name in CAR_FAMILIES:
        b = _built(name, car_objective=LAW)
        assert b.evaluate(_centre(b))["objective_label"]


def test_it_is_not_refused_against_a_free_area():
    """The distinction from ``cz`` and ``cd``, which ARE refused there. Both
    terms are FORCES, so nothing about this score is referenced to the area
    being searched — there is no artefact to refuse."""
    for name in CAR_FAMILIES:
        b = _built(name, car_objective=LAW)      # every car family's area is free
        out = b.evaluate(_centre(b))
        assert out["feasible"] and out["area_free"] is True, name
    with pytest.raises(ValueError, match="referenced to the very area"):
        _built("car rear wing", car_objective="cz")


# ------------------------------------------------- the warning, measured


def test_it_rides_the_area_row_and_the_alpha_sweep():
    """THE TRIPWIRE FOR THE CARD'S OWN CLAIM. The note under the select says
    the answer takes 100 % of the area row and 100 % of the alpha sweep, and
    that sentence is the reason a user would state a drag ceiling. Re-derived
    from the model here rather than pinned: the score is strictly increasing
    in BOTH rows along the whole band, which is the mechanism the claim names.

    A monotonicity check and not a search, because it is the stronger
    statement: an optimiser landing on a bound could be a local answer, but a
    score rising all the way to the bound cannot have an interior optimum in
    that row at all.
    """
    for name in ("car rear wing", "car rear wing + endplates"):
        b = _built(name, car_objective=LAW)
        B = np.asarray(b.bounds)
        lab = list(b.param_labels)
        x0 = _centre(b)
        for row in ("S_m2", "alpha_deg"):
            i = lab.index(row)
            scores = []
            for f in (0.0, 0.25, 0.5, 0.75, 1.0):
                x = np.array(x0, dtype=float)
                x[i] = B[i, 0] + f * (B[i, 1] - B[i, 0])
                out = b.evaluate(x)
                assert out["feasible"], (name, row, f, out["reason"])
                scores.append(out["score"])
            assert scores == sorted(scores), (name, row, scores)
            assert scores[-1] > scores[0] * 1.05, (name, row, scores)


def _best(cap, seed=3, name="car rear wing"):
    """The best design this box holds under ``cap``, and where it sits.

    Nelder-Mead from the best of 128 Sobol draws. A search and not a sample,
    because the claim is about what the ANSWER does and the ceiling only
    binds near the answer: over 2048 Sobol draws the best design drags 46 N,
    a long way inside the allowance, so a sampled arm would assert nothing.
    Refusals and violated margins are scored out rather than penalised, so
    the arms differ only by the limit.
    """
    from scipy.optimize import minimize
    from scipy.stats import qmc

    flags = {"car_objective": LAW}
    if cap is not None:
        flags["drag_budget_n"] = float(cap)
    b = _built(name, **flags)
    B = np.asarray(b.bounds)

    def neg(u):
        out = b.evaluate(B[:, 0] + np.clip(u, 0.0, 1.0) * (B[:, 1] - B[:, 0]))
        if not out.get("feasible") or any(g < 0.0 for g in out["g"]):
            return 1e3
        return -float(out["score"])

    X = qmc.Sobol(B.shape[0], scramble=True, seed=seed).random(128)
    r = minimize(neg, min(X, key=neg), method="Nelder-Mead",
                 options=dict(maxiter=2000, xatol=1e-3, fatol=1e-3))
    u = np.clip(r.x, 0.0, 1.0)
    return b, u, b.evaluate(B[:, 0] + u * (B[:, 1] - B[:, 0]))


def test_a_drag_ceiling_is_what_gives_it_an_interior_answer():
    """...and the other half of the same claim: the ceiling is the lever.

    The card says a drag ceiling turns the SPAN from a bound into an answer.
    Asserted as what the search LANDS ON, in two arms that differ only by the
    limit: without one the answer drags more than the published allowance,
    with one the drag sits exactly ON the allowance and the wing is WIDER —
    span buys the downforce back at less drag, which is the whole mechanism
    the note describes.

    Nothing is pinned. The mechanism is the claim; the newton values move
    with the restart and are quoted in ``carwing.CAR_OBJECTIVES``, not here.
    """
    cap = carwing.published_drag_budget_n()
    assert cap == pytest.approx(81.5, abs=0.1)

    b, u_free, free = _best(None)
    _, u_capped, capped = _best(cap)
    i_b = list(b.param_labels).index("b_m")

    assert free["feasible"] and capped["feasible"]
    assert free["drag_N"] > cap                    # the ratchet spends drag...
    assert capped["drag_N"] == pytest.approx(cap, rel=1e-3)   # ...sits ON it
    assert u_capped[i_b] > u_free[i_b]             # ...and it buys span back
    # the capped answer is worse on the score, and only a little: the ceiling
    # costs the wing far less than it looks, which is why it is worth stating
    assert capped["score"] < free["score"]
    assert capped["score"] > 0.99 * free["score"]
    # ...and the limit is DECLARED, so a violation is a refusal rather than a
    # number in the report
    on = _built("car rear wing", car_objective=LAW, drag_budget_n=cap)
    B = np.asarray(b.bounds)
    x_free = B[:, 0] + u_free * (B[:, 1] - B[:, 0])
    out = on.evaluate(x_free)
    i = list(out["constraint_labels"]).index("drag force margin")
    assert out["g"][i] < 0.0
    assert "drag force margin" not in b.evaluate(x_free)["constraint_labels"]


# ============================================ the law and its limit, together


def _text(view) -> str:
    return " ".join((getattr(e, "text", "") or "")
                    for e in view.descendants())


def _car_ctx(objective: str = "efficiency"):
    from gui.v3.app import assemble

    ctx = assemble("track")
    ctx.act("accept_mission")
    ctx.act("set_choice", "car_objective", objective)
    return ctx


#: which limit each law is the other half of. Stated here as the CLAIM the
#: card must honour — the shell has its own copy and a test that imported it
#: would assert the code against itself.
WANTS = {"efficiency": "Downforce, at least",
         "drag": "Downforce, at least",
         "downforce": "Drag, no more than",
         LAW: "Drag, no more than"}


@pytest.mark.parametrize("objective,limit", sorted(WANTS.items()))
def test_the_limit_a_law_needs_is_typable_where_the_law_is_chosen(
        objective, limit):
    """THE ASK. Choosing a law and looking for the number that is its other
    half must find a FIELD, not a sentence about another tab."""
    ctx = _car_ctx(objective)
    ctx.render("wing", "type")
    view = ctx.views[("wing", "type")]
    text = _text(view)
    assert "Maximise" in text
    assert limit in text, (objective, text[-600:])
    # a LABEL is not a field: there must be a number input in this view
    assert [e for e in view.descendants() if type(e).__name__ == "Number"]
    # ...and the one this law needs is the one marked
    els = [(getattr(e, "text", "") or "") for e in view.descendants()]
    k = next(i for i, t in enumerate(els) if "this objective needs it" in t)
    assert limit in " ".join(els[max(0, k - 8):k]), (objective, els[k - 8:k])


def test_the_design_box_does_not_ask_it_a_second_time():
    """MOVED, never copied. Two homes for one number is how the box on
    screen and the box searched come to disagree, and this shell has paid for
    that once already."""
    ctx = _car_ctx()
    ctx.render("wing", "box")
    text = _text(ctx.views[("wing", "box")])
    assert "Downforce, at least" not in text
    assert "Drag, no more than" not in text


def test_typing_a_limit_does_not_replace_the_field_being_typed_into():
    """The field moved views, so the view that must survive a keystroke moved
    with it. Asserted on the widgets: a rebuilt input loses the focus and
    swallows the rest of the number."""
    from gui.v3.stages import wing as v3wing

    ctx = _car_ctx()
    ctx.render("wing", "type")
    view = ctx.views[("wing", "type")]
    before = [id(e) for e in view.descendants() if type(e).__name__ == "Number"]
    assert before
    for key, (first, whole) in {"car_downforce_min_n": (4.0, 425.0),
                                "car_drag_budget_n": (9.0, 95.5)}.items():
        ctx.act("set_choice", key, first)
        ctx.act("set_choice", key, whole)
        assert ctx.S["wing"]["choices"][key] == whole, key
    assert [id(e) for e in view.descendants()
            if type(e).__name__ == "Number"] == before
    # ...and the box IS free to repaint now, which is why they left that list
    assert "car_downforce_min_n" not in v3wing.BOX_VALUE_KEYS
    assert "car_drag_budget_n" not in v3wing.BOX_VALUE_KEYS


# ------------------------------------------------------- the menu and the run


def test_the_new_law_is_offered_and_reaches_the_run():
    """Through the shell's own channel — the menu, ``car_flags``,
    ``check_flags``, a build and an evaluation — on every family the card can
    derive. A menu that can send a refused combination is a menu that lies."""
    import gui.nice_app as v1

    for plates in (False, True):
        ch = v1.choices_from_problem("car rear wing")
        ch["car_endplates"] = plates
        v1.normalise_choices(ch)
        assert LAW in v1.car_objective_options(ch), plates
        ch["car_objective"] = LAW
        name = v1.derive_problem(ch)[0]
        flags = v1.car_flags(ch)
        assert flags["car_objective"] == LAW
        api.check_flags(name, flags)
        b = api.PROBLEM_SPECS[name].build({}, flags, None)
        out = b.evaluate(_centre(b))
        assert out["feasible"] and out["objective"] == LAW, (name, out["reason"])


def test_the_card_states_the_ratchet_before_the_law_is_picked():
    """A bound-riding objective that says so only in its ANSWER has told the
    user after they paid for the search."""
    ctx = _car_ctx(LAW)
    ctx.render("wing", "type")
    text = _text(ctx.views[("wing", "type")])
    assert "RIDES ITS BOUNDS" in text
    assert "100 % of the area row" in text
    assert "58 %" in text                    # what the ceiling buys, measured
