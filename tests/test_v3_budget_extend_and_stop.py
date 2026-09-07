"""A finished run can be given more evaluations, and a stopped one keeps its best.

Three separate statements, and the first one is a live defect the other two
are built on top of.

**The adaptive stop was firing on every run.** ``ConvergenceStop.should_stop``
returns True at ``n >= max_evals``, and the shell was constructing it with
``max_evals`` = the run's own budget — which the optimiser stops at anyway. So
the rule fired on the LAST evaluation of every recommended wing run: a search
that spent exactly its budget came back ``partial=True`` with the reason "the
stop rule fired", was written to ``<ts>_partial.json``, lost ``n_screened`` /
``n_rescue`` / ``failures`` (``partial_result`` never sets them), and paid one
extra objective evaluation to rebuild its breakdown. The Results page told the
user "stopped early — 29 of 29 evaluations", which is the opposite of what
happened, and is exactly the reading that makes a user think their search never
converged.

**"Keep going" is one longer run, not a handoff.** Same config, same seed,
bigger budget. For every optimiser in ``api.CONTINUABLE_OPTIMISERS`` the
evaluations already paid for come back bit-for-bit, so the longer run genuinely
CONTAINS the shorter one. That matters because this repo has measured the
alternative — seeding BO with a previous run's best — and BO is the optimiser it
is worst to hand off TO (``RESULTS_HANDOFF.md``).

**A cancelled queue publishes the seed that ran**, not the one that never
started.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aerobo import api                                        # noqa: E402
from aerobo.optimize.budget import ConvergenceStop, converged_report  # noqa: E402


def _session():
    from gui.v3 import session

    return session.make_session("air")


# ------------------------------------------- 1. the stop that fired every time

def test_the_adaptive_stop_does_not_fire_because_the_budget_ran_out():
    """The budget is the OPTIMISER's ceiling; the rule reports the plateau.

    Asserted on the RESULT, not on the rule: what went wrong was not the
    arithmetic of ``should_stop`` (it is doing what it says) but a complete
    run being labelled and stored as a partial one.
    """
    from gui.v3 import config, session

    S = _session()
    eff = session.effective_wing_search(S)
    assert eff["plan"].patience is not None      # the rule exists at all
    cfg = config.build_cfg(S)

    got = api.run(cfg, stop_rule=session.stop_rule_factory(S, eff)())
    assert got.partial is False
    assert getattr(got, "stop_reason", None) is None
    assert got.n_evals == int(eff["budget"])
    # the fields `partial_result` cannot rebuild are present on a full result
    assert got.failures is not None

    # ...and it is the SAME search as one with no rule at all
    plain = api.run(cfg)
    assert got.best_score == plain.best_score
    assert np.array_equal(np.asarray(got.eval_x, dtype=float),
                          np.asarray(plain.eval_x, dtype=float))


def test_the_rule_still_stops_a_run_that_has_flattened_out():
    """Removing the redundant backstop must not remove the rule.

    Exercised at a budget the rule can actually reach: it compares against
    the incumbent ``patience`` evaluations ago, so it needs more than
    ``patience`` evaluations to have an opinion at all.
    """
    from gui.v3 import session

    S = _session()
    eff = dict(session.effective_wing_search(S))
    patience = int(eff["plan"].patience)
    eff["budget"] = patience * 4
    rule = session.stop_rule_factory(S, eff)()

    fired = next((i for i in range(1, eff["budget"] + 1)
                  if rule(i, 10.0 + min(i, 5) * 1.0)), None)
    assert fired is not None
    assert fired < eff["budget"]          # the PLATEAU, not the ceiling


def test_the_card_says_when_the_switch_cannot_fire_at_all():
    """The study publishes patience 40, and the plateau test needs that many
    evaluations behind it before it has an opinion. A switch that cannot fire
    must not promise that it will — and one that CAN must not deny it.

    Both cases are shipped now. The budget follows the design vector
    (3.1 per variable + 10), so a small vector is under the patience and a
    large one is over it: the opening aeroplane designs its own tail (15-D,
    budget 56) and is the second case, where it used to be the first.
    """
    from gui.v3 import session

    S = _session()
    reach = session.convergence_rule_reach(S)
    assert reach is not None
    assert reach["can_fire"] is (reach["budget"] > reach["patience"] + 1)

    small = dict(session.effective_wing_search(S))
    small["budget"] = reach["patience"]
    assert session.convergence_rule_reach(S, small)["can_fire"] is False

    big = dict(session.effective_wing_search(S))
    big["budget"] = reach["patience"] * 3
    assert session.convergence_rule_reach(S, big)["can_fire"] is True


# --------------------------------------------------- 2. the converged verdict

def test_the_verdict_distinguishes_flat_from_still_climbing_from_unanswerable():
    assert converged_report([1, 2, 3, 4, 5, 6], 3, 0.01)["verdict"] == "climbing"
    flat = converged_report([1.0, 2.0, 3.0, 3.0, 3.0, 3.0], 3, 0.01)
    assert flat["verdict"] == "converged"
    # a run no longer than the patience says nothing either way, which is the
    # state EVERY recommended wing run is in (patience 40, budgets 17-53)
    short = converged_report(list(range(30)), 40, 0.002)
    assert short["verdict"] == "too_short"
    assert "40" in short["text"] and "30" in short["text"]
    assert converged_report([], 3, 0.01)["verdict"] == "empty"
    assert converged_report([1, 2], None, None)["verdict"] == "unmeasured"


def test_the_verdict_uses_the_run_s_own_rule():
    """The readout and the rule that would have stopped the run are the same
    arithmetic, so they cannot drift apart."""
    hist = [1.0, 2.0, 3.0] + [3.0] * 10
    rep = converged_report(hist, 4, 0.01)
    rule = ConvergenceStop(patience=4, tol=0.01, max_evals=10 ** 9)
    stopped = any(rule.update(v) for v in hist)
    assert (rep["verdict"] == "converged") is stopped


# ------------------------------------------------------ 3. keep going, longer

def test_a_continuation_re_flies_what_was_already_paid_for():
    from gui.v3 import config

    S = _session()
    first = api.run(config.build_cfg(S))
    cfg, note = api.continue_run_config(first.to_dict(), 16)

    assert note["exact"] is True
    assert cfg.budget == first.budget + 16
    assert cfg.seed == first.seed
    assert cfg.flags == first.to_dict()["config"]["flags"]

    longer = api.run(cfg)
    a = np.asarray(first.eval_x, dtype=float)
    b = np.asarray(longer.eval_x, dtype=float)
    assert len(b) > len(a)
    assert float(np.max(np.abs(a - b[:len(a)]))) == 0.0
    # ...and a longer search cannot be worse: the incumbent is a max
    assert longer.best_score >= first.best_score


def test_an_optimiser_that_cannot_continue_says_so_instead():
    """``ga`` sizes its population from the budget, so a longer run is a
    different search from its first evaluation. Measured, not assumed."""
    import dataclasses

    from gui.v3 import config

    cfg0 = dataclasses.replace(config.build_cfg(_session()),
                               optimiser="ga", budget=12)
    first = api.run(cfg0)
    cfg, note = api.continue_run_config(first.to_dict(), 8)
    assert note["exact"] is False
    assert "ga" in note["why"]

    longer = api.run(cfg)
    a = np.asarray(first.eval_x, dtype=float)
    b = np.asarray(longer.eval_x, dtype=float)
    assert float(np.max(np.abs(a - b[:len(a)]))) > 0.0   # the note is true


def test_a_clamped_initial_design_is_pinned_ON_THE_FALLBACK_PATH():
    """``_bo_split`` caps the Sobol block at ``budget - 1``, so a short run
    drew a SMALLER initial design than its problem asks for, and a longer run
    left to itself asks for the bigger one and diverges at evaluation 1.

    Reported: *"Keep going still starts from the beginning instead of
    continuing from the last evaluation"* — and it did. Measured on 'tandem'
    at dim 10, budget 12 -> 20: the split moved (11, 1) -> (16, 4) and the
    two histories parted at evaluation 12.

    THE PIN IS NOW THE FALLBACK, not the answer. A record that carries its
    own evaluations is RESUMED and re-flies nothing at all; the pin applies
    where there is no log to resume from. Both halves are asserted here,
    because this test used to pass for the wrong reason: this family stores
    its single constraint's margins FLAT, ``resume_payload`` mis-read that
    shape as one point with n margins, and the record fell to the pin branch
    while looking like it belonged there (``api._margin_rows``).

    Asserted on the EVALUATIONS, not on the note: the whole claim is that the
    paid-for designs come back, and a note saying so is not evidence."""
    from gui.v3 import config          # noqa: F401 — parity with the module

    cfg0 = api.RunConfig(problem_name="free planform (aircraft)",
                         optimiser="bo", budget=12, seed=0)
    first = api.run(cfg0)
    assert first.bo_split[0] < api._bo_split(64, first.dim)[0]

    # WITH its log, this record is resumed and nothing is re-flown
    assert api.can_resume(first.to_dict()) is True
    r_cfg, r_note = api.continue_run_config(first.to_dict(), 8)
    assert r_note["resumed"] == 12 and r_cfg.budget == 20
    assert api.BO_N_INIT_FLAG not in (r_cfg.flags or {})

    # WITHOUT it — an older record — the pin is what keeps the prefix exact
    thin = {k: v for k, v in first.to_dict().items()
            if k not in ("eval_x", "eval_y", "eval_g")}
    cfg, note = api.continue_run_config(thin, 8)
    assert note["exact"] is True and note["resume"] is None
    assert cfg.flags[api.BO_N_INIT_FLAG] == first.bo_split[0]

    longer = api.run(cfg)
    a = np.asarray(first.eval_x, dtype=float)
    b = np.asarray(longer.eval_x, dtype=float)
    assert len(b) > len(a)
    assert float(np.max(np.abs(a - b[:len(a)]))) == 0.0

    # ...and the pin is what does it: the same continuation WITHOUT it is the
    # divergence the user reported, so this is not a test of BO being
    # deterministic
    import dataclasses
    bare = api.run(dataclasses.replace(cfg, flags={
        k: v for k, v in (cfg.flags or {}).items()
        if k != api.BO_N_INIT_FLAG}))
    c = np.asarray(bare.eval_x, dtype=float)
    assert float(np.max(np.abs(a - c[:len(a)]))) > 0.0


def test_the_shell_arms_the_continuation_off_the_record_not_the_form():
    """The box may have moved since; a longer run of a DIFFERENT search is not
    what the button says it is, so the drift is named."""
    from gui.v3 import config, session

    S = _session()
    rd = api.run(config.build_cfg(S)).to_dict()

    out = session.continue_run(S, rd)
    assert out["error"] is None and out["drift"] == []
    assert out["note"]["exact"] is True

    S["wing"]["bounds"]["taper"] = [0.5, 0.9]
    out = session.continue_run(S, rd)
    assert "bounds_overrides" in out["drift"]
    # ...and it still offers the RECORD's own configuration
    assert out["cfg"].bounds_overrides == rd["config"]["bounds_overrides"]


def test_a_stopped_run_is_offered_the_rest_of_the_budget_it_was_given():
    S = _session()
    rd = {"config": {"problem_name": "trim wing", "budget": 40,
                     "optimiser": "bo", "seed": 0},
          "n_evals": 12, "partial": True, "dim": 3}
    from gui.v3 import session

    assert session.continue_extra_default(rd) == 28
    rd_full = dict(rd, partial=False, n_evals=40)
    assert session.continue_extra_default(rd_full) == 20


def test_finishing_a_stopped_run_is_the_budget_it_was_given_not_more():
    """A run stopped at 12 of 40 has 28 evaluations left in it, and finishing
    it is a TOTAL OF 40.

    The offer used to be built by adding those 28 to the BUDGET rather than to
    what was SPENT, so the button said "28 more" and armed a search of 68 —
    a bigger search than the one that was interrupted, under a label promising
    the rest of the one that was. Stage 2 does this arithmetic correctly
    (``session.continue_section``); this is the wing saying the same thing.

    Asserted on the CONFIG the button would launch, not on the note, because
    the note is a sentence and the budget is what runs.
    """
    from gui.v3 import session

    S = _session()
    rd = {"config": {"problem_name": "trim wing", "budget": 40,
                     "optimiser": "bo", "seed": 0},
          "n_evals": 12, "partial": True, "dim": 3}

    out = session.continue_run(S, rd)

    assert out["error"] is None
    assert out["cfg"].budget == 40
    assert (out["note"]["was"], out["note"]["added"]) == (12, 28)
    assert out["note"]["budget"] == 40
    # re-flying a configuration NOTHING about which moved cannot break the
    # containment for any optimiser — not even the ones a bigger budget
    # re-shapes — so this is the one continuation that is exact by construction
    assert out["note"]["exact"] is True


def test_a_run_that_spent_its_whole_budget_is_offered_a_bigger_one():
    """The other half of the same rule: nothing is left of a finished run's
    budget, so more evaluations means a LONGER search — 40 spent + 20 = 60."""
    from gui.v3 import session

    S = _session()
    rd = {"config": {"problem_name": "trim wing", "budget": 40,
                     "optimiser": "bo", "seed": 0},
          "n_evals": 40, "partial": False, "dim": 3}

    out = session.continue_run(S, rd)

    assert out["cfg"].budget == 60
    assert (out["note"]["was"], out["note"]["added"]) == (40, 20)


def test_a_typed_number_past_the_rest_of_the_budget_lengthens_the_run():
    """28 finishes it; 40 finishes it and adds 12 — a total of 52, counted
    from what was SPENT, never from the budget it never reached."""
    from gui.v3 import session

    S = _session()
    rd = {"config": {"problem_name": "trim wing", "budget": 40,
                     "optimiser": "bo", "seed": 0},
          "n_evals": 12, "partial": True, "dim": 3}

    out = session.continue_run(S, rd, 40)

    assert out["cfg"].budget == 52
    assert (out["note"]["was"], out["note"]["added"]) == (12, 40)


def test_the_wing_and_the_section_answer_the_same_question_the_same_way():
    """One question, one arithmetic. The two stages had different answers for
    "what does +28 mean on a run stopped at 12 of 40" — 40 on stage 2 and 68
    on stage 3 — which is a difference a user can only find by reading both
    buttons on the same day."""
    from gui.v3 import session

    S = _session()
    cfgd = {"problem_name": "trim wing", "budget": 40, "optimiser": "bo",
            "seed": 0}
    wing = session.continue_run(
        S, {"config": cfgd, "n_evals": 12, "partial": True, "dim": 3})
    sec = session.continue_section(
        {"config": cfgd, "result": {"n_evals": 12, "partial": True, "dim": 3}},
        {"shape": {"re": 1e6}, "seed": 0,
         "search": {"optimiser": "bo", "budget": 40}, "n_restarts": 1})

    assert wing["note"]["budget"] == sec["note"]["budget"] == 40
    assert wing["note"]["added"] == sec["note"]["added"] == 28


# ----------------------------------------------- 4. a cancelled queue's best

def test_a_cancelled_queue_publishes_the_seed_that_actually_ran():
    """``current`` returns the LAST terminal job. A job the cancel took out
    before it started holds no result, so calling it "cancelled" hid the
    partial result of the seed that had really been running."""
    from gui import nice_app as v1

    mgr = v1.RunManager()
    ran = v1.RunJob(cfg=None, label="seed 0", budget=10)
    ran.status, ran.result = "cancelled", {"best_score": 1.0}
    never = v1.RunJob(cfg=None, label="seed 1", budget=10)
    never.status = "skipped"
    mgr.jobs = [ran, never]

    assert mgr.current is ran
    assert mgr.current.result is not None
