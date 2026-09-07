"""A refusal that says HOW FAR out it is, and the run that uses it.

``rescue`` gave a run with nothing feasible a feasibility phase: climb the
min-margin surrogate instead of the objective. On a box that is mostly
refusals that surface is CONSTANT — every refused design reports the same
flat ``-1`` however far outside the gate it sits — so the phase declares
itself degenerate and falls back to uniform draws. ``guide``
(:class:`aerobo.optimize.feasible.SizeGateMargin`) grades it from the design
vector, where the two gates that do the refusing are inequalities in numbers
the vector already carries.

The claims tested here are the ones a study cannot make for me: that the
grade is monotone in the overshoot, that it never outranks a design that
actually solved, that it reaches the RECORD and not only the runner, and that
every other mode still records exactly what it always did.
"""
import numpy as np
import pytest

from aerobo import api, sizing
from aerobo.optimize import feasible as F

CASE = "trim wing + free planform"
HEAVY = {"W_N": 7000.0, "V": 300.0}
CAP_PA = 75.202848
LABELS = ("taper", "twist_root_deg", "twist_tip_deg", "b_m", "S_m2")


def _grader():
    return F.SizeGateMargin(LABELS, ar_limits=sizing.AR_LIMITS,
                            cap_pa=CAP_PA, weight_n=HEAVY["W_N"])


def _x(b, S):
    x = np.zeros(len(LABELS))
    x[LABELS.index("b_m")] = b
    x[LABELS.index("S_m2")] = S
    return x


def test_a_design_inside_both_gates_is_not_graded():
    """The grade must be silent wherever the physics is not refusing on size:
    a wrong sign here would rewrite the margin of designs that FLEW."""
    g = _grader()
    S = HEAVY["W_N"] / CAP_PA * 1.5             # comfortably under the ceiling
    b = float(np.sqrt(10.0 * S))                # aspect ratio 10, mid-band
    assert g.excess(_x(b, S)) == 0.0
    assert g.graded(_x(b, S), [-0.4]) == [-0.4]


def test_the_grade_is_monotone_in_the_overshoot():
    """Further outside must read worse — that is the whole point, and it is
    what a flat sentinel cannot express."""
    g = _grader()
    S = 22.0                                    # the reported area row's top
    b = float(np.sqrt(10.0 * S))
    worse = [float(g.graded(_x(b, S / k), [-1.0])[0]) for k in (1, 2, 4)]
    assert worse == sorted(worse, reverse=True), worse
    assert all(v < -1.0 for v in worse)


def test_a_refusal_never_outranks_a_design_that_solved():
    """A design that ran and missed a limit by 0.046 is closer to an answer
    than one the physics would not run at all, and the ranking the feasibility
    phase and ``min_violation_index`` share has to keep it that way."""
    g = _grader()
    graded = float(g.graded(_x(20.0, 8.0), [-1.0])[0])
    assert graded <= -1.0 < -0.046


def test_the_binding_gate_is_the_one_that_speaks():
    """Two gates, one number: the larger overshoot decides, so a card built on
    this cannot name the gate that was nearly satisfied."""
    g = _grader()
    # an aspect ratio 5x over its band, on an area big enough that the LOADING
    # gate is satisfied — so only one of the two can be speaking
    S_ar = 200.0
    assert HEAVY["W_N"] / S_ar < CAP_PA
    tall = g.excess(_x(float(np.sqrt(200.0 * S_ar)), S_ar))
    assert tall == pytest.approx((200.0 - 40.0) / 40.0, rel=1e-9)
    # ...and the mirror: aspect ratio mid-band, loading 1.5x over the ceiling
    S_ws = HEAVY["W_N"] / (CAP_PA * 1.5)
    loaded = g.excess(_x(float(np.sqrt(10.0 * S_ws)), S_ws))
    assert loaded == pytest.approx(0.5, rel=1e-6)
    # and where BOTH are outside, the larger one is the number reported
    both = g.excess(_x(float(np.sqrt(200.0 * 8.0)), 8.0))
    assert both == pytest.approx((HEAVY["W_N"] / 8.0 - CAP_PA) / CAP_PA,
                                rel=1e-9)


def test_a_problem_with_nothing_to_measure_stays_flat():
    """No ceiling and no size rows -> no grader, so no run can be changed by
    a mode it cannot use."""
    assert not F.SizeGateMargin(("taper",), ar_limits=sizing.AR_LIMITS).live
    assert F.SizeGateMargin(LABELS, ar_limits=sizing.AR_LIMITS).live


# --------------------------------------------------- through a whole run
def _run(mode):
    flags = {"wing_loading_limit_pa": CAP_PA}
    if mode != "off":
        flags["bo_feasibility"] = mode
    return api.run(api.RunConfig(
        problem_name=CASE, mission_kwargs=dict(HEAVY), flags=flags,
        optimiser="bo", budget=6, seed=0))


def _margins(res):
    rows = [np.atleast_1d(np.asarray(g, dtype=float)).min()
            for g in (res.eval_g or [])]
    return [float(v) for v in rows]


def test_an_optimiser_that_does_not_read_the_mode_refuses_it():
    """The grading rides the wrapper EVERY optimiser's evaluations pass
    through, so a GA — whose feasibility-first tournament ranks infeasible
    members by violation — would have had its ranking quietly changed by a
    flag named for the BO loops. Refused, not ignored: a flag that is accepted
    and does nothing is the hole this api closes everywhere else."""
    cfg = api.RunConfig(
        problem_name=CASE, mission_kwargs=dict(HEAVY),
        flags={"wing_loading_limit_pa": CAP_PA, "bo_feasibility": "guide"},
        optimiser="ga", budget=8, seed=0)
    with pytest.raises(ValueError, match="does not read bo_feasibility"):
        api.run(cfg)


def test_a_graded_refusal_is_still_recognised_as_a_refusal():
    """THE DEFECT GRADING INTRODUCED, and the reason to look for it. Every
    diagnosis downstream splits the log into designs that FLEW and designs the
    solver never ran, and it did so by testing the margins for exactly -1. A
    graded refusal is not -1, so every one of them would have been read as a
    real design: the "nearest miss" would name a wing nobody flew, and the box
    moves would be correlated against margins no solver produced.

    The contract's SHAPE is what identifies it — one value, repeated across
    every margin, at or below the sentinel — and that is what is tested here,
    in both directions.
    """
    from gui.diagnose import _unflyable

    G = np.array([[-1.0, -1.0],      # a legacy refusal
                  [-1.7, -1.7],      # a graded one
                  [-0.5, -0.5],      # a design that flew and missed both
                  [-1.3, -0.2]])     # a design that flew, margins differ
    Y = np.array([api.PENALTY_SCORE if hasattr(api, "PENALTY_SCORE")
                  else -100.0, -100.0, 12.0, 9.0])
    assert list(_unflyable(G, Y)) == [True, True, False, False]
    # ...and the objective is still required: a graded-looking margin pair on
    # a design that SOLVED is a design, not a refusal
    assert not _unflyable(np.array([[-1.7, -1.7]]), np.array([12.0]))[0]


def test_the_feasibility_phase_stops_being_a_uniform_search():
    """THE MECHANISM, measured. ``rescue`` gives a run that has found nothing a
    feasibility phase; on a box of refusals that phase has a CONSTANT surface
    to climb, so every iteration of it declares itself degenerate and falls
    back to a uniform draw — a phase in name only, and indistinguishable from
    a working one in every output the run reports. ``n_rescue_blind`` is the
    number that tells them apart, and grading the margin is what moves it.

    The box here is the one the mission empties, so NEITHER arm can find a
    design (it is provably empty — tests/test_size_box_conflicts.py). That is
    deliberate: it isolates whether the phase had a direction from whether
    there was anywhere to go.
    """
    flags = {"wing_loading_limit_pa": CAP_PA, "bo_n_init": 6}
    out = {}
    for mode in ("rescue", "guide"):
        res = api.run(api.RunConfig(
            problem_name=CASE, mission_kwargs=dict(HEAVY),
            flags={**flags, "bo_feasibility": mode},
            optimiser="bo", budget=30, seed=0))
        assert res.n_rescue and res.n_rescue > 10, mode
        out[mode] = (res.n_rescue, res.n_rescue_blind)
    (n_r, blind_r), (n_g, blind_g) = out["rescue"], out["guide"]
    assert blind_r == n_r, ("a flat margin cannot be climbed; if this ever "
                            "passes with blind < n the surface stopped being "
                            "flat and this test is measuring something else",
                            out)
    assert blind_g <= 0.5 * n_g, out


def test_the_grade_reaches_the_record_and_only_under_guide():
    """It has to be in the LOG, not only in the runner: ``min_violation_index``
    and every diagnosis downstream rank on what was recorded, and a graded
    margin the record did not keep would leave them reading the flat sentinel
    the run itself had stopped using.

    The control is the same run under ``rescue``: identical mechanism, flat
    margins. That is what makes this a test of the grading rather than of the
    fact that a refused design has a negative margin.
    """
    guided, control = _run("guide"), _run("rescue")
    g_margins, c_margins = _margins(guided), _margins(control)
    assert g_margins and c_margins
    # the box is empty at this weight (tests/test_size_box_conflicts.py), so
    # every evaluation is a refusal and the control can only record -1
    assert set(np.round(c_margins, 12)) == {-1.0}
    assert min(g_margins) < -1.0
    assert len(set(np.round(g_margins, 9))) > 1, (
        "graded margins that are all equal are a flat surface by another name")
