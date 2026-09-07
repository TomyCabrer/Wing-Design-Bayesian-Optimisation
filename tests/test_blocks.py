"""Block-coordinate ("portfolio") optimiser — optimize/blocks.py.

Cheap analytic tests only; the real 13-D winglet+section problem it serves
is XFOIL-slow and covered by the slow-marked registry smoke in test_api.
"""

import numpy as np
import pytest

from aerobo.optimize.blocks import run_blocks_constrained


def _toy_fg(x):
    """Maximise -||x - 0.3||^2 s.t. x0 + x1 <= 1.2 (optimum at x = 0.3)."""
    f = -float(np.sum((x - 0.3) ** 2))
    g = np.array([1.2 - x[0] - x[1]])
    return f, g


BOUNDS4 = np.array([[-1.0, 1.0]] * 4)
BLOCKS4 = ({"name": "a", "indices": (0, 1), "optimiser": "bo"},
           {"name": "b", "indices": (2, 3), "optimiser": "sobol"})


def test_blocks_converges_and_logs_unique_evals():
    res = run_blocks_constrained(_toy_fg, BOUNDS4, n_evals=40, seed=0,
                                 blocks=BLOCKS4, x0=np.zeros(4), cycles=2)
    assert res.best_x is not None
    assert res.best_y > -0.15                  # approaches the optimum (0)
    assert len(res.y) <= 40                    # unique-eval budget honoured
    assert res.X.shape == (len(res.y), 4)
    assert res.best_so_far.shape == (len(res.y),)
    # history rows are unique points (memo prevents duplicate physics)
    assert len({tuple(np.round(r, 9)) for r in res.X}) == len(res.y)
    # meta records the block structure for the report
    assert [b["name"] for b in res.meta["blocks"]] == ["a", "b"]


def test_blocks_incumbent_improves_monotonically():
    res = run_blocks_constrained(_toy_fg, BOUNDS4, n_evals=30, seed=1,
                                 blocks=BLOCKS4, x0=np.zeros(4), cycles=2)
    bsf = res.best_so_far[np.isfinite(res.best_so_far)]
    assert np.all(np.diff(bsf) >= -1e-12)


def test_blocks_partition_validation():
    with pytest.raises(ValueError, match="partition"):
        run_blocks_constrained(_toy_fg, BOUNDS4, n_evals=10, seed=0,
                               blocks=({"indices": (0, 1),
                                        "optimiser": "sobol"},))
    with pytest.raises(ValueError, match="blocks"):
        run_blocks_constrained(_toy_fg, BOUNDS4, n_evals=10, seed=0,
                               blocks=())


def test_blocks_x0_seeds_the_incumbent():
    """The first logged evaluation IS x0 (warm start recorded)."""
    x0 = np.array([0.1, 0.2, -0.3, 0.4])
    res = run_blocks_constrained(_toy_fg, BOUNDS4, n_evals=12, seed=0,
                                 blocks=BLOCKS4, x0=x0, cycles=1)
    assert np.allclose(res.X[0], x0)


def test_winglet_section_problem_declares_blocks():
    """The 13-D problem's block metadata partitions the vector wing|section."""
    from aerobo.winglet_section import WingletSectionProblem
    prob = WingletSectionProblem()
    blocks = prob.blocks
    assert [b["name"] for b in blocks] == ["wing/winglet", "CST section"]
    idx = sorted(i for b in blocks for i in b["indices"])
    assert idx == list(range(prob.dim)) and prob.dim == 13
    assert prob.x0.shape == (13,)
    # x0 sits inside the box (warm start must be evaluable)
    b = prob.bounds
    assert np.all(prob.x0 >= b[:, 0]) and np.all(prob.x0 <= b[:, 1])


# ------------------------------------------------- budget & search-quality fixes

def test_blocks_spends_the_whole_budget():
    """The runner must deliver the budget it was asked for.

    The first implementation planned cycles*blocks sub-runs and never
    re-spent the remainder, delivering 13 unique evaluations for a requested
    18 (28 % short) — which makes any blocks-vs-BO comparison at equal
    nominal budget unfair, the one thing this project's crossover claims
    rest on.
    """
    for budget in (18, 24, 40):
        res = run_blocks_constrained(_toy_fg, BOUNDS4, n_evals=budget, seed=0,
                                     blocks=BLOCKS4, x0=np.zeros(4), cycles=2)
        # never over budget, and within one sub-run floor of hitting it
        assert len(res.y) <= budget
        assert len(res.y) >= budget - 3, (budget, len(res.y))


def test_blocks_records_actual_per_block_spend():
    res = run_blocks_constrained(_toy_fg, BOUNDS4, n_evals=30, seed=0,
                                 blocks=BLOCKS4, x0=np.zeros(4), cycles=2)
    spend = [b["evals"] for b in res.meta["blocks"]]
    assert sum(spend) <= len(res.y)
    assert all(s > 0 for s in spend)          # no silently starved block
    assert [b["name"] for b in res.meta["blocks"]] == ["a", "b"]


def test_blocks_validates_the_declaration_before_spending_anything():
    """A typo must surface immediately, not after minutes of physics."""
    calls = []

    def counting(x):
        calls.append(x)
        return _toy_fg(x)

    with pytest.raises(ValueError, match="not in the constrained registry"):
        run_blocks_constrained(counting, BOUNDS4, 20, blocks=(
            {"name": "a", "indices": (0, 1), "optimiser": "bo"},
            {"name": "b", "indices": (2, 3), "optimiser": "typo"}))
    with pytest.raises(ValueError, match="share"):
        run_blocks_constrained(counting, BOUNDS4, 20, blocks=(
            {"name": "a", "indices": (0, 1, 2, 3), "optimiser": "bo",
             "share": 0.0},))
    assert calls == []


def test_blocks_repairs_an_infeasible_incumbent():
    """With nothing feasible known, the runner must still move toward
    feasibility — otherwise an infeasible start freezes the incumbent and
    the whole budget is spent going nowhere."""
    def hard(x):
        # feasible only in a far corner; the start is deeply infeasible
        f = -float(np.sum((x - 0.9) ** 2))
        return f, np.array([float(np.sum(x)) - 2.0])

    blocks = ({"name": "a", "indices": (0, 1), "optimiser": "bo",
               "owns_constraints": True},
              {"name": "b", "indices": (2, 3), "optimiser": "bo"})
    res = run_blocks_constrained(hard, BOUNDS4, n_evals=40, seed=0,
                                 blocks=blocks, x0=-np.ones(4), cycles=2)
    g = np.asarray(res.g).reshape(len(res.y), -1)
    viol = np.max(np.maximum(0.0, -g), axis=1)
    assert viol[0] > 0.0                       # the start really is infeasible
    assert viol.min() < viol[0]                # and the search moved toward it


def test_blocks_constraint_owner_runs_first_when_infeasible():
    """Only a constraint-owning block can repair a violation, so it must be
    scheduled first while the incumbent is infeasible."""
    order = []

    def probe(x):
        # constraint depends ONLY on the second block's variables
        f = -float(np.sum(x ** 2))
        g = np.array([float(x[2] + x[3]) + 1.0])
        order.append(tuple(np.round(x, 6)))
        return f, g

    blocks = ({"name": "cheap", "indices": (0, 1), "optimiser": "sobol"},
              {"name": "owner", "indices": (2, 3), "optimiser": "sobol",
               "owns_constraints": True})
    res = run_blocks_constrained(probe, BOUNDS4, n_evals=20, seed=0,
                                 blocks=blocks, x0=np.array([0., 0., -1., -1.]),
                                 cycles=1)
    # the seed point is infeasible, so the FIRST sub-run must have moved the
    # owner block (indices 2,3), not the cheap one
    first_moves = [p for p in order[1:4]]
    assert any(p[2] != -1.0 or p[3] != -1.0 for p in first_moves)
    assert res.best_x is not None


def test_blocks_warm_start_reuses_only_same_subproblem_rows(monkeypatch):
    """Cycle-to-cycle warm start is only valid while the FROZEN coordinates
    are unchanged; once another block moves, the old rows describe a
    different function and must be dropped."""
    from aerobo.optimize import blocks as _blocks

    # The old "every logged point is unique" proxy is a property of the MEMO,
    # not of the frozen-coordinate filter: dropping that filter (block_rows
    # selecting every logged row) seeds each GP with rows measured under a
    # DIFFERENT frozen configuration and every logged point is still unique.
    # So watch what is actually handed to the sub-optimiser as seed data.
    evaluated: list[np.ndarray] = []
    seeded: list[tuple[int, np.ndarray]] = []
    real_bo = _blocks.run_bo_constrained

    def spy_bo(f, sub_bounds, *a, x_init=None, **kw):
        seeded.append((len(evaluated),
                       np.atleast_2d(np.asarray(x_init, dtype=float))))
        return real_bo(f, sub_bounds, *a, x_init=x_init, **kw)

    def logged(x):
        evaluated.append(np.asarray(x, dtype=float).copy())
        return _toy_fg(x)

    monkeypatch.setattr(_blocks, "run_bo_constrained", spy_bo)
    # block 'b' starts in the far corner, so its first sub-run is certain to
    # move it — without that the stale-row case this test exists for is never
    # constructed and nothing here could distinguish the two behaviours
    x0 = np.array([0.0, 0.0, -1.0, -1.0])
    res = run_blocks_constrained(logged, BOUNDS4, n_evals=30, seed=0,
                                 blocks=BLOCKS4, x0=x0, cycles=2,
                                 warm_start=True)
    # every logged point is unique, so no warm-start row was charged as physics
    assert len({tuple(np.round(r, 9)) for r in res.X}) == len(res.y)

    # ...and every seed of block 'a' (the only BO block) is the block slice of
    # a point already measured AT THE SAME frozen coordinates
    frozen, checked = [], 0
    for n_before, x_init in seeded:
        if n_before >= len(evaluated):
            continue                   # a sub-run that bought nothing fresh
        fz = evaluated[n_before][2:]   # what this sub-run held frozen
        prior = [r for r in evaluated[:n_before] if np.allclose(r[2:], fz)]
        for row in x_init:
            assert any(np.allclose(r[:2], row) for r in prior), (
                f"warm-start seed {row} was never measured with the frozen "
                f"coordinates {fz} this sub-problem holds")
        frozen.append(fz)
        checked += 1
    assert checked >= 2, "fewer than two BO sub-runs: nothing was checked"
    assert any(not np.allclose(frozen[0], fz) for fz in frozen[1:]), (
        "block 'b' never moved, so no warm-start row could go stale — the "
        "case this test exists for was not constructed")

    seeded.clear()
    evaluated.clear()
    off = run_blocks_constrained(logged, BOUNDS4, n_evals=30, seed=0,
                                 blocks=BLOCKS4, x0=x0, cycles=2,
                                 warm_start=False)
    assert len(off.y) > 0
    assert seeded and all(xi.shape[0] == 1 for _n, xi in seeded), (
        "warm_start=False must seed the incumbent and nothing else")


def test_winglet_section_blocks_declare_measured_choices():
    """The per-block optimiser and share are evidence-backed, and the
    section block is marked as the constraint owner (both margins are
    functions of the CST weights alone)."""
    from aerobo.winglet_section import WingletSectionProblem
    blocks = {b["name"]: b for b in WingletSectionProblem().blocks}
    assert blocks["CST section"]["optimiser"] == "bo"
    assert blocks["wing/winglet"]["optimiser"] == "bo"
    assert blocks["CST section"]["owns_constraints"] is True
    assert blocks["wing/winglet"]["owns_constraints"] is False
    # the cheap block gets the larger share of the evaluation budget
    assert blocks["wing/winglet"]["share"] > blocks["CST section"]["share"]
