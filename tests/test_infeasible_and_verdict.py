"""Three complaints about the shape-optimisation card, and their fixes.

* AN IMPROVEMENT DID NOT LOOK LIKE ONE. The seed-vs-optimised tables printed
  a signed number in a column of signed numbers, so "did this get better?"
  was a sum the reader had to do per row. Every row already carried a verdict
  (``dir``); nothing painted it. Green/red/grey now does, in both tables on
  the card — and the verdict gained a fourth value (``same``) so "did not
  move" stops being spelt the same way as "cannot be judged".
* "▼ BELOW THE SEED" WAS SAID ABOUT CRITERIA NOBODY WEIGHTED. It reads as a
  charge against the search, and the search never defended a zero-weight
  criterion: it is not in J, ``goal_shortfalls`` prices its shortfall at
  exactly zero and ``asf_terms`` drops it outright. The mark is now for a
  weighted criterion only, and the unweighted one gets a hollow ▽ that says
  why.
* A RUN THAT FOUND NOTHING EXPLAINED NOTHING. "0 of 64 evaluations were
  feasible … widen the design box" names no box row and no direction. The run
  already logged every margin at every point it tried, so which constraint
  was never met, whether any single design could have met them all, and which
  bound the best attempt was pressed against are all measurable
  (:mod:`gui.diagnose`) — and they are measured off the record rather than
  guessed at.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gui import diagnose                            # noqa: E402
from gui import nice_app as v1                      # noqa: E402
from gui.v3.stages import airfoil as stage          # noqa: E402


# ===================================================== the verdict is shown
def test_an_unchanged_metric_is_not_spelt_like_an_unjudgeable_one():
    """``same`` and ``""`` are different sentences: one says the search did
    not move this, the other says nobody can say which way is better."""
    same = v1._cmp_row("wing L/D", 40.0, 40.0, lower_better=False)
    unjudged = v1._cmp_row("tip twist [deg]", 1.0, 2.0)
    assert same["dir"] == "same"
    assert unjudged["dir"] == ""


def test_the_verdict_still_follows_the_preferred_direction():
    """The fourth value must not have eaten the other three — and a
    lower-is-better metric that FELL is an improvement, which is the whole
    reason ``dir`` exists rather than the sign of the change."""
    assert v1._cmp_row("cd", 0.010, 0.009, lower_better=True)["dir"] == "better"
    assert v1._cmp_row("cd", 0.009, 0.010, lower_better=True)["dir"] == "worse"
    assert v1._cmp_row("LD", 40.0, 44.0, lower_better=False)["dir"] == "better"


@pytest.mark.parametrize("delta,expect", [(2.5, "better"), (-2.5, "worse"),
                                          (0.0, "same"), (None, ""),
                                          (float("nan"), "")])
def test_the_score_verdict_answers_on_an_up_is_better_band(delta, expect):
    assert stage._verdict(delta) == expect


def test_both_tables_on_the_card_are_painted_by_that_verdict():
    """One colour rule, applied through one helper: the two tables carry the
    same question and this card has already had a field-name drift between
    them once."""
    slots = {}

    class _FakeTable:
        def add_slot(self, name, template):
            slots[name] = template

    stage.verdict_slots(_FakeTable())
    assert set(slots) == {"body-cell-new", "body-cell-change"}
    tpl = slots["body-cell-change"]
    from gui.v3 import theme
    assert theme.GOOD in tpl and theme.BAD in tpl
    assert "__GOOD__" not in tpl and "__BAD__" not in tpl and "__FAINT__" \
        not in tpl
    # colour is not the only carrier — a red-green reader gets the glyph
    for glyph in ("▲", "▼", "="):
        assert glyph in tpl
    for verdict in ("better", "worse", "same"):
        assert f"'{verdict}'" in tpl


def test_every_row_the_tables_are_given_carries_a_verdict_to_paint():
    """A template reading ``props.row.dir`` on rows without one paints
    everything grey and the fix looks applied while doing nothing."""
    rep = {"section": {"baseline": {"tc": 0.12,
                                    "polar": {"cd_at_cl_design": 0.0080}},
                       "design": {"tc": 0.13,
                                  "polar": {"cd_at_cl_design": 0.0071}}}}
    for row in v1.airfoil_compare_rows(rep):
        assert "dir" in row, row
    for row in stage.score_rows(_SCORE):
        assert "dir" in row, row


# ============================================ the mark is for a weighted one
#: a scoring payload where the SAME sub-score fall happens to a criterion the
#: user weighted and to one they weighted at zero
_SCORE = {
    "weights": {"ldcr": 0.6, "clmax": 0.4, "cm": 0.0, "ldmax": 0.0,
                "thick": 0.0, "astall": 0.0},
    "seed": {"composite": 60.0,
             "scores": {"ldcr": 70.0, "clmax": 50.0, "cm": 40.0},
             "metrics": {"ldcr": 83.9, "clmax": 1.4, "cm_at": 0.05}},
    "optimised": {"composite": 64.0,
                  "scores": {"ldcr": 62.0, "clmax": 66.0, "cm": 28.0},
                  "metrics": {"ldcr": 73.6, "clmax": 1.6, "cm_at": 0.07}},
    "delta": {"composite": 4.0,
              "scores": {"ldcr": -8.0, "clmax": 16.0, "cm": -12.0}},
}


def _by_criterion(rows):
    """Rows keyed by the criterion's own label — the cell is
    ``"<label> (<raw> → <raw>) · weight <w>[ mark]"``, so the label is
    everything before the raw-value parenthesis the row appends."""
    out = {}
    for r in rows:
        head = r["metric"].split(" · weight")[0]
        # …and the label itself may contain a parenthesis ("|Cm| (lower
        # better)"), so it is the LAST one that gets dropped, not the first
        out[head.rsplit(" (", 1)[0] if head.endswith(")") else head] = r
    return out


def test_a_weighted_criterion_that_fell_is_still_marked():
    """The fix must not be "stop saying it": where the objective WAS
    defending the criterion, the complaint is earned."""
    row = _by_criterion(stage.score_rows(_SCORE))["L/D at design Cl"]
    assert "▼ below the seed" in row["metric"]
    assert row["dir"] == "worse"


def test_an_unweighted_criterion_that_fell_is_not_accused():
    """|Cm| fell by 12 points and carries weight 0.00 — it is not in J, so
    the search never sold it and the card must not say it did."""
    row = _by_criterion(stage.score_rows(_SCORE))["|Cm| at design Cl"]
    assert "▼ below the seed" not in row["metric"]
    assert "▽ below the seed (not weighted)" in row["metric"]
    # …and the colour follows the same rule, or the red says what the words
    # were just stopped from saying
    assert row["dir"] == ""


def test_the_criterion_is_still_shown_and_its_change_still_printed():
    """"Not accused" is not "hidden": the row, the sub-scores and the fall
    are all still on the card."""
    row = _by_criterion(stage.score_rows(_SCORE))["|Cm| at design Cl"]
    assert row["original"] == "40.0" and row["new"] == "28.0"
    assert row["change"] == "-12.0"
    assert "weight 0.00" in row["metric"]


def test_a_criterion_that_improved_is_marked_neither_way():
    row = _by_criterion(stage.score_rows(_SCORE))["Cl max"]
    assert "below the seed" not in row["metric"]
    assert row["dir"] == "better"


def test_the_objective_menu_no_longer_promises_more_than_it_defends():
    """``composite_goal`` prices a shortfall at ``penalty x weight x
    shortfall``, which is zero at weight 0 — so "no criterion below the seed"
    was a promise the objective does not make."""
    label = stage.OBJECTIVE_CHOICES(False)["composite_goal"]
    assert "WEIGHTED" in label


# =========================================== why a run came back with nothing
def _run(eval_x, eval_g, eval_y=None, bounds=None, labels=None, pinned=None):
    """A minimal ``RunResult.to_dict`` payload: the fields the diagnosis
    reads, and nothing it does not."""
    X = np.asarray(eval_x, dtype=float)
    G = np.asarray(eval_g, dtype=float)
    n_feas = int(np.sum(np.all(G >= 0.0, axis=1)))
    return {
        "best_x": None if n_feas == 0 else list(X[0]),
        "feasible": n_feas > 0,
        "n_feasible": n_feas,
        "n_evals": int(X.shape[0]),
        "eval_x": X.tolist(),
        "eval_g": G.tolist(),
        "eval_y": (list(eval_y) if eval_y is not None
                   else [-1.0] * int(X.shape[0])),
        "bounds": (bounds if bounds is not None
                   else [[0.0, 1.0]] * int(X.shape[1])),
        "param_labels": (labels if labels is not None
                         else [f"x{k}" for k in range(X.shape[1])]),
        "pinned": pinned,
    }


def _sweep(n=25, k=0, sign=1.0, offset=-0.30, span=0.25, d=2):
    """``n`` points across the box, with margin 0 rising (or falling) linearly
    along variable ``k`` and never quite reaching zero."""
    t = np.linspace(0.0, 1.0, n)
    X = np.full((n, d), 0.5)
    X[:, k] = t
    g0 = offset + span * (t if sign > 0 else (1.0 - t))
    g1 = np.full(n, 0.4)
    return X, np.column_stack([g0, g1])


def test_a_run_that_found_a_design_has_nothing_to_explain():
    X, G = _sweep()
    G[:, 0] += 1.0                                   # everything feasible
    assert diagnose.infeasibility_report(_run(X, G)) is None


def test_an_unconstrained_record_is_not_diagnosed():
    assert diagnose.infeasibility_report(
        {"best_x": None, "feasible": False, "eval_g": None}) is None


def test_it_names_the_constraint_that_was_never_met():
    X, G = _sweep()
    rep = diagnose.infeasibility_report(
        _run(X, G), constraint_labels=("t/c margin", "|Cm| margin"))
    assert rep["binding"]["label"] == "t/c margin"
    assert rep["binding"]["ever_met"] is False
    assert rep["binding"]["best"] == pytest.approx(-0.05)
    # the one that was always satisfied is reported, and not as the culprit
    other = rep["constraints"][1]
    assert other["label"] == "|Cm| margin" and other["n_violated"] == 0


def test_it_names_the_box_row_and_the_direction_to_move_it():
    """The margin rises with x0 and the best point sits on x0's CAP, so the
    cap is what stopped the run."""
    X, G = _sweep(k=0, sign=1.0)
    rep = diagnose.infeasibility_report(_run(X, G),
                                        constraint_labels=("t/c margin",))
    assert rep["box_is_binding"] is True
    move = rep["moves"][0]
    assert move["variable"] == "x0"
    assert move["at"] == "upper" and "RAISE" in move["direction"]
    assert move["corr"] > 0.9


def test_the_direction_follows_the_physics_and_not_the_column_order():
    """Same box, margin falling with x0 instead: the recommendation must flip
    to the FLOOR. A rule that always said "raise the cap" would pass the test
    above and send every user the wrong way half the time."""
    X, G = _sweep(k=0, sign=-1.0)
    rep = diagnose.infeasibility_report(_run(X, G))
    move = rep["moves"][0]
    assert move["at"] == "lower" and "LOWER" in move["direction"]
    assert move["corr"] < -0.9


def test_riding_a_bound_is_not_enough_the_sign_has_to_agree():
    """Both halves of the rule, separately. A variable can sit on its CAP at
    the best point while the margin FALLS as it rises (another variable is
    carrying the margin, or the relation is not monotone) — raising that cap
    makes things worse, and "the optimum is at the bound" alone would send
    the user there. The mirror case is a variable on its FLOOR that the
    margin rises with.
    """
    n = 25
    t = np.linspace(0.0, 1.0, n)
    g = -0.30 + 0.25 * t                       # the margin, carried by x0
    rising_at_cap = t.copy()                   # r > 0, best point at the cap
    falling_at_cap = 1.0 - t
    falling_at_cap[-1] = 1.0                   # r < 0, best point at the cap
    rising_at_floor = t.copy()
    rising_at_floor[-1] = 0.0                  # r > 0, best point on the floor
    X = np.column_stack([rising_at_cap, falling_at_cap, rising_at_floor])
    G = np.column_stack([g])
    rep = diagnose.infeasibility_report(
        _run(X, G, bounds=[[0.0, 1.0]] * 3,
             labels=["carries_it", "at_cap_wrong_way", "at_floor_wrong_way"]))
    assert [m["variable"] for m in rep["moves"]] == ["carries_it"]
    assert rep["moves"][0]["at"] == "upper"


def test_a_variable_the_search_never_pushed_against_is_not_recommended():
    """x1 is held at the middle of its box: it correlates with nothing and it
    is riding no bound, so widening it cannot be the fix."""
    X, G = _sweep(k=0, sign=1.0)
    rep = diagnose.infeasibility_report(_run(X, G))
    assert [m["variable"] for m in rep["moves"]] == ["x0"]


def test_a_bound_rider_with_no_measurable_correlation_is_not_recommended():
    """The other half of the same rule: this variable IS on its cap at the
    best point, and the margin does not follow it at all (r ~ 0). Sending a
    user to widen it costs them another full search for nothing, so the
    correlation floor is what makes the recommendation worth acting on."""
    n = 25
    t = np.linspace(0.0, 1.0, n)
    g = -0.30 + 0.25 * t
    noise = np.array([1.0 if i % 2 == 0 else 0.0 for i in range(n)])
    assert noise[-1] == 1.0                    # riding the cap at the best
    assert abs(np.corrcoef(noise, g)[0, 1]) < diagnose.MIN_CORR
    X = np.column_stack([t, noise])
    rep = diagnose.infeasibility_report(
        _run(X, np.column_stack([g]), bounds=[[0.0, 1.0]] * 2,
             labels=["carries_it", "unrelated"]))
    assert [m["variable"] for m in rep["moves"]] == ["carries_it"]


def test_a_correlated_variable_that_is_NOT_at_its_bound_is_not_recommended():
    """The margin still rises with x0, but the best point stops halfway up
    the box — so the box is not what is in the way and the honest answer is
    that no row is indicated."""
    t = np.linspace(0.0, 0.5, 25)                    # never reaches the cap
    X = np.column_stack([t, np.full(25, 0.5)])
    G = np.column_stack([-0.30 + 0.25 * t, np.full(25, 0.4)])
    rep = diagnose.infeasibility_report(_run(X, G))
    assert rep["moves"] == [] and rep["box_is_binding"] is False
    text = " ".join(t for t, _ in diagnose.infeasibility_lines(rep))
    assert "Relax the limit itself" in text


def test_a_failed_evaluation_cannot_vote_on_the_direction():
    """A solver failure returns ``(PENALTY, [G_FAIL, G_FAIL])`` — margins that
    were never measured. Left in, enough of them at the top of the box invert
    the correlation and the card sends the user to widen the FLOOR of the row
    whose CAP is the problem."""
    X, G = _sweep(k=0, sign=1.0, n=25)
    dead_x = np.column_stack([np.linspace(0.9, 1.0, 12), np.full(12, 0.5)])
    dead_g = np.full((12, 2), diagnose.G_FAIL)
    Xa = np.vstack([X, dead_x])
    Ga = np.vstack([G, dead_g])
    Y = [-0.5] * len(X) + [diagnose.PENALTY] * len(dead_x)
    rep = diagnose.infeasibility_report(_run(Xa, Ga, eval_y=Y))
    assert rep["n_unflyable"] == 12 and rep["n_measured"] == len(X)
    assert rep["moves"][0]["at"] == "upper"          # the real direction
    assert rep["binding"]["best"] == pytest.approx(-0.05)   # not -1.0


def test_a_run_that_mostly_could_not_be_flown_says_so_first():
    X, G = _sweep(k=0, sign=1.0, n=10)
    dead_x = np.column_stack([np.linspace(0.0, 1.0, 30), np.full(30, 0.5)])
    dead_g = np.full((30, 2), diagnose.G_FAIL)
    Y = [-0.5] * 10 + [diagnose.PENALTY] * 30
    rep = diagnose.infeasibility_report(
        _run(np.vstack([X, dead_x]), np.vstack([G, dead_g]), eval_y=Y))
    first_warn = next(t for t, lvl in diagnose.infeasibility_lines(rep)
                      if lvl == "warn")
    assert "could not be FLOWN at all" in first_warn


def test_two_constraints_that_are_each_reachable_but_never_together():
    """A different failure with a different fix: widening the box is not
    indicated, because the box already contains points that satisfy each of
    them — just never the same point."""
    t = np.linspace(0.0, 1.0, 25)
    X = np.column_stack([t, np.full(25, 0.5)])
    G = np.column_stack([-0.5 + t, 0.5 - t])         # they cross, both < 0
    G[G == 0.0] = -1e-6
    rep = diagnose.infeasibility_report(
        _run(X, G), constraint_labels=("t/c margin", "|Cm| margin"))
    assert rep["conflict"] is True
    text = " ".join(s for s, _ in diagnose.infeasibility_lines(rep))
    assert "no point that holds them all at once" in text
    # …and the box move it does offer is not sold as a free fix: raising x0's
    # cap lifts the binding margin and spends the other one, which is exactly
    # what a trade-off looks like from inside one constraint
    move = rep["moves"][0]
    assert [t["label"] for t in move["trades_against"]] == ["|Cm| margin"]
    assert "this is a trade, not a free fix" in text


def test_one_constraint_is_never_called_a_trade_off():
    """With a single margin, "met somewhere" and "no feasible design" cannot
    both be true. A record that says both is INCONSISTENT — a partial run, a
    point the objective could not be read at — and the answer to an
    inconsistent record is not "your two constraints conflict", because there
    is only one of them.
    """
    t = np.linspace(0.0, 1.0, 25)
    g = -0.30 + 0.60 * t                         # crosses zero: met somewhere
    rd = _run(np.column_stack([t, np.full(25, 0.5)]), np.column_stack([g]))
    rd.update(best_x=None, feasible=False, n_feasible=0)
    rep = diagnose.infeasibility_report(rd, constraint_labels=("t/c margin",))
    assert rep["constraints"][0]["ever_met"] is True
    assert rep["conflict"] is False
    text = " ".join(s for s, _ in diagnose.infeasibility_lines(rep))
    assert "holds them all at once" not in text


def test_a_pinned_variable_is_reported_as_the_thing_to_unpin():
    """A pin collapses the row to zero width, so it correlates with nothing
    and rides no bound — the two tests above would drop it silently, and
    "you pinned the only variable that could have fixed this" is the most
    useful sentence on the page."""
    t = np.linspace(0.0, 1.0, 25)
    X = np.column_stack([t, np.full(25, 0.3)])
    G = np.column_stack([np.full(25, -0.2)])
    rep = diagnose.infeasibility_report(
        _run(X, G, bounds=[[0.0, 1.0], [0.3, 0.3]],
             labels=["sweep", "t/c"], pinned={"t/c": 0.3}))
    moves = {m["variable"]: m for m in rep["moves"]}
    assert moves["t/c"]["pinned"] is True
    assert "unpin" in moves["t/c"]["direction"].lower()


def test_pins_do_not_evict_the_row_that_is_actually_riding_the_bound():
    """A pinned entry is emitted on the strength of the pin alone — no
    correlation, no bound-riding test — so it must not spend the budget the
    MEASURED rows are competing for.

    Three pins is ordinary (the per-row FIX switch plus the automatic
    ``taper`` pin), and with one shared cap of three they sorted ahead of the
    one row that correlates with the binding margin AND sits on its cap, and
    dropped it: the card then printed three "unpin it" bullets and no
    direction at all, and ``gui.v3.relax.plan`` — which skips pinned entries —
    called the same record "no-evidence".
    """
    t = np.linspace(0.0, 1.0, 25)
    X = np.column_stack([t, np.full(25, 3.0), np.full(25, 4.0),
                         np.full(25, 5.0)])
    G = np.column_stack([-0.30 + 0.25 * t])       # rises with the free row
    rep = diagnose.infeasibility_report(
        _run(X, G, bounds=[[0.0, 1.0], [3.0, 3.0], [4.0, 4.0], [5.0, 5.0]],
             labels=["sweep", "pin_a", "pin_b", "pin_c"],
             pinned={"pin_a": 3.0, "pin_b": 4.0, "pin_c": 5.0}))
    moves = {m["variable"]: m for m in rep["moves"]}
    assert set(moves) == {"sweep", "pin_a", "pin_b", "pin_c"}
    assert moves["sweep"]["pinned"] is False and moves["sweep"]["at"] == "upper"
    text = " ".join(s for s, _ in diagnose.infeasibility_lines(rep))
    assert "sweep — RAISE its upper bound" in text


def test_the_lines_lead_with_the_count_and_end_with_the_edit():
    X, G = _sweep(k=0, sign=1.0)
    lines = diagnose.infeasibility_lines(
        diagnose.infeasibility_report(_run(X, G),
                                      constraint_labels=("t/c margin",)))
    assert lines[0][1] == "bad" and "0 of 25" in lines[0][0]
    assert any("RAISE its upper bound" in t for t, _ in lines)
    assert all(isinstance(t, str) and t for t, _ in lines)


def test_nothing_to_explain_produces_no_sentences():
    assert diagnose.infeasibility_lines(None) == []


# ================================================ …and on the real V3 cards
def _infeasible_record(problem="wing+winglet (VLM)"):
    """A finished, constrained run whose every point missed one gate — the
    payload the shells actually hold (``S["run"]["record"]``)."""
    t = np.linspace(0.0, 1.0, 20)
    X = np.column_stack([t, np.full(20, 0.5)])
    G = np.column_stack([-0.30 + 0.25 * t, np.full(20, 0.4)])
    rd = _run(X, G, labels=["w_upper_0", "w_lower_0"])
    rd.update(problem_name=problem, is_constrained=True,
              config={"problem_name": problem, "mission_kwargs": {},
                      "flags": {}, "optimiser": "sobol", "budget": 20,
                      "seed": 0, "bounds_overrides": None},
              best_score=None, breakdown=None)
    return rd


def _texts(box):
    return [(getattr(e, "text", "") or "") for e in box.descendants()]


def _stage2(report):
    from gui.v3 import session
    from gui.v3.app import assemble

    ctx = assemble("air")
    ctx.act("accept_mission")
    session.airfoil_state(ctx.S, "main")["opt"]["report"] = report
    ctx.render("airfoil", "optimise")
    return ctx.views[("airfoil", "optimise")]


def test_the_shape_card_paints_its_real_compare_table():
    """The pure helper is wired to the table the user sees — a colour rule
    nothing calls is a colour rule that does not exist."""
    from gui.v3 import theme

    box = _stage2({
        "result": {"problem_name": "airfoil (section)", "best_x": [0.1] * 8,
                   "feasible": True, "n_feasible": 5, "n_evals": 20,
                   "best_score": -0.005},
        "conditions": {"tc_min": 0.10, "cm_max": 0.08, "objective": "cd",
                       "wing_mode": False},
        "baseline": {"breakdown": {"LD": 40.0, "CD": 0.0125}},
        "design": {"breakdown": {"LD": 44.0, "CD": 0.0114}},
        "section": {"baseline": {"tc": 0.12,
                                 "polar": {"cd_at_cl_design": 0.008}},
                    "design": {"tc": 0.13,
                               "polar": {"cd_at_cl_design": 0.0071}}}})
    painted = [e for e in box.descendants()
               if type(e).__name__ == "Table"
               and "body-cell-change" in getattr(e, "slots", {})]
    assert len(painted) == 1, "the seed-vs-optimised table is not painted"
    tpl = painted[0].slots["body-cell-change"].template
    assert theme.GOOD in tpl and theme.BAD in tpl


def test_the_shape_card_explains_a_run_that_found_no_section():
    """Instead of "best objective —" over two dead buttons: which gate, how
    close it came, and the gate's own units."""
    rd = _infeasible_record("airfoil (section)")
    box = _stage2({"result": rd,
                   "conditions": {"tc_min": 0.10, "cm_max": 0.08,
                                  "objective": "cd", "wing_mode": False}})
    text = " ".join(_texts(box))
    assert "No feasible section" in text
    assert "t/c margin is the binding one" in text
    assert "RAISE its upper bound" in text
    # the gate's OWN units: g0 = (t/c - tc_min)/tc_min, so the best t/c is
    # tc_min (1 + g0) = 0.10 x 0.95
    assert "t/c 0.0950" in text and "your floor asks 0.1000" in text
    # …and the buttons that would have done nothing are gone
    assert "Adopt this section" not in text


def test_the_result_page_names_the_row_instead_of_saying_widen_the_box():
    """Stage 4's constraints block used to end on "Widen the design box, or
    move the mission, and run it again" — advice with no object."""
    from gui.v3.app import assemble

    ctx = assemble("air")
    ctx.act("accept_mission")
    ctx.act("set_result", _infeasible_record())
    ctx.render("results", "summary")
    text = " ".join(_texts(ctx.views[("results", "summary")]))
    assert "Widen the design box, or move the mission" not in text
    assert "0 of 20 evaluations were feasible" in text
    assert "RAISE its upper bound" in text


def test_the_wing_run_tab_stops_saying_the_criteria_appear_once_it_finishes():
    """A finished run with nothing in it is not a run that has not finished
    yet, and that was the only sentence this block had for it."""
    from gui import nice_app as _v1
    from gui.v3.app import assemble

    ctx = assemble("air")
    ctx.act("accept_mission")
    rd = _infeasible_record()
    ctx.act("set_result", rd)
    # the run tab draws nothing at all without a job on the manager, so the
    # finished one is put there — status "done", no result to score
    job = _v1.RunJob(cfg=None, label="seed 0", budget=20)
    job.status = "done"
    ctx.manager.jobs = [job]
    ctx.render("wing", "run")
    text = " ".join(_texts(ctx.views[("wing", "run")]))
    assert "0 of 20 evaluations were feasible" in text
    assert "t/c margin" not in text or "binding one" in text
    assert "The criteria of the design appear here once a run finishes" \
        not in text


# ============================================ …and the same rule on stage 3
_WING_SCORE = {
    "weights": {"lod": 0.7, "astall": 0.0},
    "baseline": {"composite": 50.0, "scores": {"lod": 50.0, "astall": 60.0},
                 "metrics": {"lod": 18.0, "astall": 12.0}},
    "optimised": {"composite": 58.0, "scores": {"lod": 66.0, "astall": 44.0},
                  "metrics": {"lod": 22.0, "astall": 10.0}},
    "delta": {"composite": 8.0, "scores": {"lod": 16.0, "astall": -16.0}},
}


def test_the_wing_criteria_table_carries_the_same_verdict():
    """One colour rule across the shell: stage 3's seed-vs-optimised table
    asks the identical question stage 2's does, and a criterion weighted 0 is
    left unjudged on both."""
    from gui.v3.stages import wing as wing_stage

    rows = {r["metric"].split(" ·")[0]: r
            for r in wing_stage.wing_score_rows(_WING_SCORE)}
    assert rows["composite score J"]["dir"] == "better"
    weighted = next(r for r in rows.values() if "weight 0.70" in r["metric"])
    assert weighted["dir"] == "better"
    unweighted = next(r for r in rows.values()
                      if "weight 0.00" in r["metric"])
    assert unweighted["change"].startswith("-")     # it DID fall…
    assert unweighted["dir"] == ""                  # …and is not judged for it


def test_the_shared_slot_helper_is_the_one_both_stages_use():
    """Two copies of a colour rule is how the two tables on the shape card
    came to declare different field names in the first place."""
    from gui.v3 import widgets as v3w
    from gui.v3.stages import airfoil as af

    assert af.verdict_slots is v3w.verdict_slots
    assert af._verdict is __import__("gui.metrics",
                                     fromlist=["verdict"]).verdict
