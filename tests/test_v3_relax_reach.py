"""Reaching for the box nearest this one, when the run came back with nothing.

The reported failure is one sentence — "no solution was found" — and this repo
has already answered it six ways (``RESULTS_NO_SOLUTION.md``), every one of
which ends by handing the user a control and asking them to move it. What was
missing is the shell doing it: work out which rows are provably what stopped
the run, move only those, by the smallest amount the evidence requires, search
that box ONCE, and report the closest feasible design to the box the user drew.

Three claims are load-bearing and each has its own test below:

* the reach is **derived from this run's own evidence** — a proof where the
  two closed-form gates have one, a fitted slope of the binding margin where
  they do not, and NOTHING where neither speaks;
* the user's design box is **not written** by anything except the adopt
  button, and the adopt button moves only the rows the ANSWER is outside;
* every band is **built before it is offered**, so a row the family will not
  honour is named on the card instead of eating the budget.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from aerobo import api
from gui import diagnose
from gui.v3 import relax


def _texts(view):
    return [getattr(e, "text", "") or "" for e in view.descendants()]


# =========================================================== the pure layer

def _record(X, Y, G, bounds, labels, pinned=None) -> dict:
    """A run record with exactly the fields the reach reads off one."""
    return {"eval_x": [list(map(float, r)) for r in X],
            "eval_y": [float(v) for v in Y],
            "eval_g": [list(map(float, r)) for r in G],
            "bounds": [list(map(float, r)) for r in bounds],
            "param_labels": list(labels), "pinned": dict(pinned or {}),
            "best_x": None, "n_evals": len(X), "n_feasible": 0}


def test_move_magnitude_is_the_deficit_over_the_fitted_slope():
    """Exactly solvable so the number is checkable: g = 0.5x - 0.55 on [0, 1]
    is short by 0.05 at its best point, and 0.05/0.5 = 0.1 of the row."""
    x = np.linspace(0.0, 1.0, 21)
    rec = _record(x[:, None], np.full(x.size, 12.0), (0.5 * x - 0.55)[:, None],
                  [[0.0, 1.0]], ["a"])
    mag = diagnose.move_magnitude(rec, 0, {"index": 0, "at": "upper"})
    assert mag["basis"] == "fit"
    assert mag["deficit"] == pytest.approx(0.05)
    assert mag["delta"] == pytest.approx(0.1 * (1.0 + diagnose.HEADROOM))
    assert mag["capped"] is False


def test_a_noisy_row_gets_a_step_and_says_so():
    """The DIRECTION is worth acting on and the DISTANCE is not, so the move
    ships as a fixed step rather than as an extrapolation nobody can check."""
    rng = np.random.default_rng(3)
    x = rng.uniform(size=200)
    g = 0.05 * x - 0.30 + 0.5 * rng.normal(size=x.size)
    rec = _record(x[:, None], np.full(x.size, 12.0), g[:, None],
                  [[0.0, 2.0]], ["a"])
    # the PREMISE, asserted, or this passes for the wrong reason
    r = float(np.corrcoef(x, g)[0, 1])
    assert r * r < diagnose.MIN_R2
    mag = diagnose.move_magnitude(rec, 0, {"index": 0, "at": "upper"})
    assert mag["basis"] == "step"
    assert mag["delta"] == pytest.approx(diagnose.RELAX_STEP * 2.0)


def test_the_extrapolation_is_capped_at_one_row_width_and_only_then():
    """Both halves. A function that always capped would pass the first."""
    x = np.linspace(0.0, 1.0, 21)
    flat = _record(x[:, None], np.full(x.size, 12.0),
                   (1e-4 * x - 0.5)[:, None], [[0.0, 1.0]], ["a"])
    mag = diagnose.move_magnitude(flat, 0, {"index": 0, "at": "upper"})
    assert mag["capped"] is True and mag["delta"] == pytest.approx(1.0)
    # ...and a move well inside one width comes back uncapped, at its own size
    near = _record(x[:, None], np.full(x.size, 12.0),
                   (1.0 * x - 1.32)[:, None], [[0.0, 1.0]], ["a"])
    mag2 = diagnose.move_magnitude(near, 0, {"index": 0, "at": "upper"})
    assert mag2["capped"] is False
    assert mag2["delta"] == pytest.approx(0.32 * (1.0 + diagnose.HEADROOM))


def test_outside_box_is_measured_in_each_rows_own_width():
    B, labels = [[0.0, 2.5], [0.0, 10.0]], ["narrow", "wide"]
    assert diagnose.outside_box(B, labels, [1.0, 5.0])["rows"] == []
    assert diagnose.outside_box(B, labels, [2.5, 10.0])["dinf"] == 0.0
    d = diagnose.outside_box(B, labels, [2.78, 5.0])
    assert [r["label"] for r in d["rows"]] == ["narrow"]
    assert d["dinf"] == pytest.approx(0.28 / 2.5)
    assert d["rows"][0]["end"] == "top"
    # MUTATION: the SAME physical overshoot on the wider row is a smaller
    # distance. An absolute metric passes every assertion above and fails here.
    wide = diagnose.outside_box(B, labels, [1.0, 10.28])
    assert wide["dinf"] == pytest.approx(0.28 / 10.0)
    assert wide["dinf"] < d["dinf"]


def test_closest_feasible_returns_the_closest_not_the_best_scoring():
    B, labels = [[0.0, 1.0]], ["a"]
    rec = _record([[1.30], [1.05], [0.50]], [40.0, 12.0, -100.0],
                  [[0.2], [0.1], [-1.0]], B, labels)
    got = diagnose.closest_feasible(rec, B, labels)
    assert got["index"] == 1 and got["y"] == pytest.approx(12.0)
    assert got["dinf"] == pytest.approx(0.05)
    # ...and the REFUSED row at distance 0 is never the answer, however near
    assert got["x"] != [0.5]


def test_closest_feasible_is_none_when_nothing_flew_and_met_every_limit():
    B, labels = [[0.0, 1.0]], ["a"]
    rec = _record([[0.5], [0.6]], [-100.0, 11.0], [[-1.0], [-0.2]], B, labels)
    assert diagnose.closest_feasible(rec, B, labels) is None


def test_refused_fraction_reads_the_shape_of_the_contract():
    B, labels = [[0.0, 1.0]], ["a"]
    allref = _record([[0.5]] * 4, [-100.0] * 4, [[-1.0]] * 4, B, labels)
    assert diagnose.refused_fraction(allref) == 1.0
    # a GRADED refusal is one value repeated BELOW the sentinel, not equal
    graded = _record([[0.5]] * 3, [-100.0] * 3, [[-1.7]] * 3, B, labels)
    assert diagnose.refused_fraction(graded) == 1.0
    # ...and a design that FLEW to a margin of -1.7 is not a refusal
    flew = _record([[0.5]] * 3, [12.0] * 3, [[-1.7]] * 3, B, labels)
    assert diagnose.refused_fraction(flew) == 0.0
    # the legacy half: no margins is not "nothing was refused"
    assert diagnose.refused_fraction(None) is None
    assert diagnose.refused_fraction({"eval_x": [[0.5]]}) is None


def test_minimal_box_moves_only_what_is_needed_and_leaves_no_bound_ridden():
    B = [[0.5, 3.0], [0.0, 1.0]]
    labels = ["moved", "held"]
    rows = relax.minimal_box(B, labels, [3.28, 0.4])
    assert set(rows) == {"moved"}          # a row that contained it is ABSENT
    assert rows["moved"][0] == pytest.approx(0.5)
    assert rows["moved"][1] == pytest.approx(0.5 + 2.78 / (1.0 - relax.MARGIN))
    # ...and this is the assertion that matters: the adopted box must not have
    # the design riding its own new bound, or ``box_moves`` recommends
    # widening the row the card just built
    pos = diagnose._where_in_box(3.28, *rows["moved"])
    assert pos == pytest.approx(1.0 - diagnose.AT_BOUND_FRAC)


# ================================================= the plan, real registry

def _heavy(medium: str = "air"):
    """The reported session: a weight raised and the wing rows left alone."""
    from gui.v3 import session
    from gui.v3.app import assemble

    ctx = assemble(medium)
    ctx.act("accept_mission")
    ctx.S["mission"]["W_N"] = 7000.0
    session.sync_wing_from_mission(ctx.S)
    assert session.set_planform(ctx.S, "free") == []
    return ctx


def _all_refused(S, n: int = 8) -> dict:
    """A record in which every draw was refused before its solver — the state
    a box the mission empties actually produces."""
    from gui.v3 import config

    cfg = config.build_cfg(S, seed=0)
    dim = len(api.PROBLEM_SPECS[cfg.problem_name].param_labels)
    rows = [{"x": [0.5] * dim, "f": -100.0, "g": [-1.0], "feasible": False}
            for _ in range(n)]
    out = api.partial_result(cfg, rows).to_dict()
    assert out["best_x"] is None
    return out


def _riding(S, label: str, at: str = "upper", slope: float = 0.02,
            deficit: float = 0.05, n: int = 24) -> dict:
    """A record whose binding margin rises with ``label`` and whose best point
    is sitting on that row's bound — the state ``box_moves`` fires on."""
    from gui.v3 import config

    cfg = config.build_cfg(S, seed=0)
    spec = api.PROBLEM_SPECS[cfg.problem_name]
    built = spec.build(cfg.mission_kwargs, cfg.flags, cfg.bounds_overrides)
    labels, B = list(built.param_labels), np.asarray(built.bounds, float)
    k = labels.index(label)
    edge = B[k, 1] if at == "upper" else B[k, 0]
    sign = 1.0 if at == "upper" else -1.0
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        x = B[:, 0] + rng.uniform(size=len(labels)) * (B[:, 1] - B[:, 0])
        if i == 0:
            x[k] = edge
        g = sign * slope * (x[k] - edge) - deficit
        rows.append({"x": [float(v) for v in x], "f": 12.0 + 0.1 * i,
                     "g": [float(g)], "feasible": False})
    out = api.partial_result(cfg, rows).to_dict()
    assert out["best_x"] is None, (
        "a record with an incumbent is not the state this card is for")
    return out


def _deficit_for(S, label: str, slope: float, widths: float) -> float:
    """The ``deficit`` whose fitted fix is ``widths`` of ``label``'s own row.

    ``relax`` measures a reach in ROW WIDTHS and refuses past one of them, so
    a fixture that states its deficit in metres is really stating a number of
    widths — and silently states a different one the moment the box changes.
    It did: the design box is now derived from the mission rather than from
    the family's published row (``session.clip_size_box``), and a 20 m fix
    that was 0.6 of a 34 m span row became 1.9 of a 11 m one, so three tests
    whose comment says "sized well inside one row-width" stopped being.
    """
    from gui.v3 import config

    lo, hi = config.effective_bounds(S)[label][0]
    return abs(slope) * widths * (float(hi) - float(lo))


def test_the_emptied_mission_is_reached_by_the_row_the_gate_proves():
    """The headline case, end to end and entirely in closed form. Nothing is
    pinned as a constant — the band is re-derived from the api that offers it,
    so a change in the ceiling moves the test with the code."""
    from gui.v3 import config

    ctx = _heavy()
    cfg = config.build_cfg(ctx.S)
    want = api.size_box_conflicts(cfg)[0]["suggest"]
    p = relax.plan(ctx.S, _all_refused(ctx.S))
    assert p["verdict"] == "ready", p.get("reason")
    assert [m["label"] for m in p["moves"]] == ["S_m2"]
    move = p["moves"][0]
    assert move["source"] == "gate"
    assert (move["new_lo"], move["new_hi"]) == \
        (pytest.approx(float(want[0])), pytest.approx(float(want[1])))
    # ...and the reach RESOLVES the thing it claimed to
    reached = api.RunConfig(**dict(config.cfg_dict(ctx.S),
                                   bounds_overrides=p["overrides"]))
    assert [f for f in api.size_box_conflicts(reached) if f["empty"]] == []


def test_the_gates_band_is_taken_whole_because_a_narrower_one_is_empty():
    """Measured, and the reason the ceiling is not shrunk to the user's own
    width: the gate's FLOOR is ``W/cap``, which cannot see the wing's own
    weight, and the sizing loop adds it. Keeping the floor and cutting the
    ceiling back to the row the user drew gives a box every draw is refused
    from — and ``size_box_conflicts`` cannot see that, because it proves
    emptiness and never feasibility."""
    from gui.v3 import config

    ctx = _heavy()
    d = config.cfg_dict(ctx.S)
    gate = api.size_box_conflicts(config.build_cfg(ctx.S))[0]
    floor, ceiling = gate["suggest"]
    was = config.effective_bounds(ctx.S)["S_m2"][0]
    narrow = [float(floor), float(floor) + (float(was[1]) - float(was[0]))]
    assert narrow[1] < ceiling, "the premise: the gate offers a wider band"
    thin = api.box_refusal_probe(
        api.RunConfig(**dict(d, bounds_overrides={"S_m2": narrow})), n=64)
    whole = api.box_refusal_probe(
        api.RunConfig(**dict(d, bounds_overrides={
            "S_m2": [float(floor), float(ceiling)]})), n=64)
    assert thin["n_feasible"] == 0
    assert whole["n_feasible"] > 0
    # the closed-form gate calls neither of them empty, which is the point
    assert [f for f in api.size_box_conflicts(
        api.RunConfig(**dict(d, bounds_overrides={"S_m2": narrow})))
        if f["empty"]] == []


def test_a_measured_row_is_moved_by_the_fitted_distance():
    ctx = _heavy()
    ctx.S["mission"]["W_N"] = 700.0        # a mission the box is not empty for
    from gui.v3 import config, session

    session.sync_wing_from_mission(ctx.S)
    assert api.size_box_conflicts(config.build_cfg(ctx.S)) == []
    rd = _riding(ctx.S, "b_m", "upper", slope=0.02, deficit=0.05)
    p = relax.plan(ctx.S, rd)
    assert p["verdict"] == "ready", p.get("reason")
    move = next(m for m in p["moves"] if m["label"] == "b_m")
    assert move["source"] == "measured" and move["basis"] == "fit"
    assert move["new_lo"] == pytest.approx(move["lo"])       # only the end
    assert move["new_hi"] > move["hi"]                        # the run rode
    assert move["new_hi"] - move["hi"] == pytest.approx(
        (0.05 / 0.02) * (1.0 + diagnose.HEADROOM), rel=0.25)


def test_a_row_the_physics_does_not_honour_is_named_not_widened():
    """Both halves in one test: ``taper`` is written into the box the sampler
    draws from and re-checked against the family's own band, so widening it
    buys refusals; ``b_m`` travels, so it is offered."""
    from gui.v3 import config, session

    ctx = _heavy()
    ctx.S["mission"]["W_N"] = 700.0
    session.sync_wing_from_mission(ctx.S)
    assert api.size_box_conflicts(config.build_cfg(ctx.S)) == []

    blocked = relax.plan(ctx.S, _riding(ctx.S, "taper", "upper"))
    assert "taper" not in [m["label"] for m in blocked["moves"]]
    assert "taper" in [b["label"] for b in blocked["blocked"]]
    assert "refused" in blocked["blocked"][0]["why"]

    ok = relax.plan(ctx.S, _riding(ctx.S, "b_m", "upper"))
    assert "b_m" in [m["label"] for m in ok["moves"]]


def test_the_reached_box_never_adds_a_row_the_family_refuses():
    """The one-line form of the honesty invariant, and the check that survives
    drift in which rows travel."""
    from gui.v3 import config, session

    ctx = _heavy()
    ctx.S["mission"]["W_N"] = 700.0
    session.sync_wing_from_mission(ctx.S)
    cfg = config.build_cfg(ctx.S)
    spec = api.PROBLEM_SPECS[cfg.problem_name]
    p = relax.plan(ctx.S, _riding(ctx.S, "b_m", "upper"))
    assert p["verdict"] == "ready", p.get("reason")
    mine = spec.build(cfg.mission_kwargs, cfg.flags, cfg.bounds_overrides)
    reach = spec.build(cfg.mission_kwargs, cfg.flags, p["overrides"])
    assert api.rows_outside_validity(reach) == api.rows_outside_validity(mine)


def test_clip_to_validity_puts_a_row_back_and_lets_the_reaching_one_through():
    """Against the real registry, re-derived rather than pinned — and the new
    band is read off ``problem.bounds``, so the test proves the widening
    reached the PHYSICS and not only the sampler."""
    spec = api.PROBLEM_SPECS["tail"]
    built = spec.build({}, {}, None)
    labels, B = list(built.param_labels), np.asarray(built.bounds, float)
    # widened the way ``plan`` widens: a strictly-positive row keeps its sign,
    # or the family raises at BUILD time and never reaches the validity check
    wide = {name: [float(max(B[k, 0] - 0.45 * (B[k, 1] - B[k, 0]),
                             relax.MIN_FLOOR_FRAC * B[k, 0])),
                   float(B[k, 1] + 0.45 * (B[k, 1] - B[k, 0]))]
            for k, name in enumerate(labels) if B[k, 0] > 0}
    # MUTATION: unclipped, the family refuses some of what it was handed
    assert api.rows_outside_validity(spec.build({}, {}, dict(wide)))
    band, capped, err = clip = relax.clip_to_validity(spec, {}, {}, dict(wide))
    assert err is None and clip
    assert api.rows_outside_validity(spec.build({}, {}, band)) == []
    names = {c["label"] for c in capped}
    assert names and "l_t_m" not in names, names
    grew = spec.build({}, {}, band).problem.bounds
    k = labels.index("l_t_m")
    assert float(np.asarray(grew, float)[k, 1]) > float(B[k, 1])


def test_a_positive_row_never_crosses_zero_on_the_way_out():
    """A wing with a negative span is not a wider search, it is a build that
    raises before any evaluation."""
    from gui.v3 import config, session

    ctx = _heavy()
    ctx.S["mission"]["W_N"] = 700.0
    session.sync_wing_from_mission(ctx.S)
    # sized so the fitted fix is well inside one row-width (or the reach is
    # refused for reaching too far, and this would test nothing) and still
    # further than the row's own floor (or there is no zero to cross) — as a
    # FRACTION of the row, because a deficit in metres states a number of
    # widths without saying so (:func:`_deficit_for`)
    rd = _riding(ctx.S, "b_m", "lower", slope=0.9,
                 deficit=_deficit_for(ctx.S, "b_m", 0.9, 0.75))
    p = relax.plan(ctx.S, rd)
    assert p["verdict"] == "ready", p.get("reason")
    band = p["overrides"]["b_m"]
    was = config.effective_bounds(ctx.S)["b_m"][0]
    move = next(m for m in p["moves"] if m["label"] == "b_m")
    # the PREMISE, off the move's own numbers: unclamped, this fix walks the
    # row's floor through zero
    raw = (move["deficit"] / abs(move["slope"])) * (1.0 + diagnose.HEADROOM)
    assert float(was[0]) - raw < 0.0, (float(was[0]), raw)
    assert band[0] > 0.0
    assert band[0] == pytest.approx(relax.MIN_FLOOR_FRAC * float(was[0]))
    cfg = config.build_cfg(ctx.S)
    api.PROBLEM_SPECS[cfg.problem_name].build(
        cfg.mission_kwargs, cfg.flags, p["overrides"])       # must not raise
    # MUTATION: without the clamp that band is a raise, not a wider box
    with pytest.raises(ValueError):
        api.PROBLEM_SPECS[cfg.problem_name].build(
            cfg.mission_kwargs, cfg.flags,
            dict(p["overrides"], b_m=[-1.0, band[1]]))


def _ws_shell():
    """A session whose wing loading is a SEARCHED row, with the mission's own
    ceiling in force."""
    from gui.v3 import session
    from gui.v3.app import assemble

    ctx = assemble()
    ctx.act("accept_mission")
    assert session.set_planform(ctx.S, "wing_loading_free") == []
    return ctx


def test_a_row_already_at_its_ceiling_is_named_and_nothing_is_launched():
    """The failure this check exists for. The mission's ceiling travels as a
    flag and IS the top of the ``ws_pa`` row by construction, so an upward
    move on it is clipped back to the band it started from. Left in, the card
    printed "37.6 – 75.2 → 37.6 – 75.2. MEASURED on this run's own points",
    counted it as a row that moved, and sold a bit-identical, deterministic
    re-run of the search that had just failed — whose "nothing out there
    either" reads as a fact about the physics."""
    from gui.v3 import config

    ctx = _ws_shell()
    cfg = config.build_cfg(ctx.S)
    cap = cfg.flags.get("wing_loading_limit_pa")
    assert cap, "the premise: this mission carries a ceiling"
    built = api.PROBLEM_SPECS[cfg.problem_name].build(
        cfg.mission_kwargs, cfg.flags, cfg.bounds_overrides)
    k = list(built.param_labels).index("ws_pa")
    assert float(np.asarray(built.bounds, float)[k, 1]) == pytest.approx(cap), \
        "the premise: the row's top already IS the ceiling"

    p = relax.plan(ctx.S, _riding(ctx.S, "ws_pa", "upper",
                                  slope=0.02, deficit=0.5))
    assert "ws_pa" not in [m["label"] for m in p["moves"]]
    blocked = {b["label"]: b for b in p["blocked"]}
    assert "ws_pa" in blocked
    assert "stage 1" in blocked["ws_pa"]["why"]
    # ...and nothing is launchable whose box is the one that just failed
    assert p["verdict"] != "ready" or p["overrides"] != (
        config.cfg_dict(ctx.S)["bounds_overrides"])


def test_the_wing_loading_row_is_clipped_at_the_missions_own_ceiling():
    """Pre-empted, never discovered: the api refuses a ``ws_pa`` band above the
    ceiling loudly, at build time, on purpose. Asserted on a row the user has
    typed BELOW the ceiling, which is the only state in which the clip has
    something to do."""
    from gui.v3 import config

    ctx = _ws_shell()
    cap = float(config.build_cfg(ctx.S).flags["wing_loading_limit_pa"])
    # a row wide enough that the fitted fix stays inside one row-width (or the
    # reach is refused for reaching too far and the clip is never exercised)
    # and whose top is below the ceiling (or there is nothing to clip)
    ctx.S["wing"]["bounds"]["ws_pa"] = [0.4 * cap, 0.95 * cap]
    rd = _riding(ctx.S, "ws_pa", "upper", slope=0.02, deficit=0.1)
    p = relax.plan(ctx.S, rd)
    assert p["verdict"] == "ready", p.get("reason")
    move = next(m for m in p["moves"] if m["label"] == "ws_pa")
    # the PREMISE: unclipped, this move would land above the ceiling
    raw = (move["deficit"] / abs(move["slope"])) * (1.0 + diagnose.HEADROOM)
    assert move["hi"] + raw > cap, (move["hi"], raw, cap)
    assert move["new_hi"] == pytest.approx(cap)
    assert move["ceiling"] == pytest.approx(cap)
    cfg = config.build_cfg(ctx.S)
    api.PROBLEM_SPECS[cfg.problem_name].build(
        cfg.mission_kwargs, cfg.flags, p["overrides"])      # must not raise
    # MUTATION: one step above the ceiling is a refusal, not a wider box
    with pytest.raises(ValueError):
        api.PROBLEM_SPECS[cfg.problem_name].build(
            cfg.mission_kwargs, cfg.flags,
            dict(p["overrides"], ws_pa=[move["new_lo"], cap * 1.5]))


def test_a_released_row_is_refused_before_a_single_evaluation_is_spent():
    """Switching a row OFF says nothing this session constrains it, and
    ``config.bounds_overrides`` drops it — so a band written for a released row
    never reaches the run. Planned, priced and searched, it would then fail at
    ADOPT, with a message about a seed."""
    from gui.v3 import config, session

    ctx = _heavy()
    ctx.S["mission"]["W_N"] = 700.0
    session.sync_wing_from_mission(ctx.S)
    rd = _riding(ctx.S, "b_m", "upper")
    assert "b_m" in [m["label"] for m in relax.plan(ctx.S, rd)["moves"]]

    ctx.S["wing"]["bounds_off"] = ["b_m"]
    assert "b_m" in config.released_rows(ctx.S)
    # ...and the RECORD re-taken over the released box. Releasing a row that
    # is in ``bounds_overrides`` changes them, and a plan whose record was
    # measured over the old ones is honestly STALE — the question here is
    # what the reach does with a released row, not what it does with a
    # record from before the release. (This used to pass by coincidence: the
    # shell's band for this mission happened to equal the family's published
    # row, so it was not an override at all and releasing it changed nothing.)
    rd = _riding(ctx.S, "b_m", "upper")
    p = relax.plan(ctx.S, rd)
    assert "b_m" not in [m["label"] for m in p["moves"]]
    assert "b_m" not in (p.get("overrides") or {})
    why = next(b["why"] for b in p["blocked"] if b["label"] == "b_m")
    assert "switched OFF" in why


def test_the_reached_run_is_the_records_own_search_in_a_different_box():
    """"One press is one search, at the budget you already paid" has to be true
    of the SEARCH, not only of the sentence. ``is_stale`` deliberately does not
    compare budget/seed/optimiser — they do not change the box — so reading
    them off the live session let an un-relaunched edit re-price the reach."""
    from gui.v3 import config, session

    ctx = _heavy()
    rd = _all_refused(ctx.S)
    live = config.cfg_dict(ctx.S)
    # the record ran a DIFFERENT budget and seed from the ones the session now
    # holds — which ``is_stale`` deliberately allows, because neither changes
    # the box. The card's price and the search must both follow the record.
    rd["config"] = dict(rd["config"], budget=int(live["budget"]) * 11,
                        seed=int(live["seed"]) + 7)
    assert relax.is_stale(ctx.S, rd) is None, "the premise: still not stale"
    was = dict(rd["config"])

    p = relax.plan(ctx.S, rd)
    assert p["verdict"] == "ready", p.get("reason")
    assert p["budget"] == was["budget"] != int(live["budget"])
    assert p["seed"] == was["seed"] != int(live["seed"])
    assert p["cfg_dict"]["budget"] == was["budget"]
    assert p["cfg_dict"]["seed"] == was["seed"]
    assert p["cfg_dict"]["optimiser"] == was["optimiser"]
    # ...and the BOX is the only thing that differs from the record's config
    differs = {k for k in set(p["cfg_dict"]) | set(was)
               if p["cfg_dict"].get(k) != was.get(k)}
    assert differs <= {"bounds_overrides", "x_seed"}, differs
    del session


def test_a_start_design_never_travels_into_the_reached_box():
    """The card that offers the reach is the card that offers "start again
    from the closest design". A seed armed there is a statement about the box
    it was armed in; the reach searches a different one, and ``api.run`` would
    refuse it AFTER the press."""
    from gui.v3 import config, session

    ctx = _heavy()
    rd = _all_refused(ctx.S)
    cfg = config.build_cfg(ctx.S)
    built = api.PROBLEM_SPECS[cfg.problem_name].build(
        cfg.mission_kwargs, cfg.flags, cfg.bounds_overrides)
    B = np.asarray(built.bounds, float)
    assert session.start_from_design(ctx.S, list(0.5 * (B[:, 0] + B[:, 1]))) \
        is None
    assert config.cfg_dict(ctx.S).get("x_seed")
    p = relax.plan(ctx.S, rd)
    assert p["verdict"] == "ready", p.get("reason")
    assert "x_seed" not in p["cfg_dict"]
    api.check_x_seed(api.RunConfig(**p["cfg_dict"]))       # must not raise


def test_a_pinned_row_is_never_widened():
    from gui.v3 import config, session

    ctx = _heavy()
    ctx.S["mission"]["W_N"] = 700.0
    session.sync_wing_from_mission(ctx.S)
    rd = _riding(ctx.S, "b_m", "upper")
    assert "b_m" in [m["label"] for m in relax.plan(ctx.S, rd)["moves"]]
    rd["pinned"] = {"b_m": 12.0}
    p = relax.plan(ctx.S, rd)
    assert "b_m" not in [m["label"] for m in p["moves"]]
    assert "b_m" not in (p.get("overrides") or {})
    del config


def test_a_move_further_than_one_row_width_is_refused_and_a_shorter_one_is_not():
    from gui.v3 import session

    ctx = _heavy()
    ctx.S["mission"]["W_N"] = 700.0
    session.sync_wing_from_mission(ctx.S)
    # a nearly flat margin puts the fitted fix far outside the row
    far = relax.plan(ctx.S, _riding(ctx.S, "b_m", "upper",
                                    slope=1e-4, deficit=5.0))
    assert far["verdict"] == "blocked"
    assert "own width" in far["reason"]
    near = relax.plan(ctx.S, _riding(ctx.S, "b_m", "upper",
                                     slope=0.02, deficit=0.05))
    assert near["verdict"] == "ready", near.get("reason")


def test_a_box_with_no_evidence_refuses_to_guess():
    """Every draw refused, and the two cheap gates do not rule the box out.
    There is nothing here that says which row to move, and a reach would be a
    guess that silently widened the user's search."""
    from gui.v3 import config, session

    ctx = _heavy()
    ctx.S["mission"]["W_N"] = 700.0
    session.sync_wing_from_mission(ctx.S)
    assert api.size_box_conflicts(config.build_cfg(ctx.S)) == []
    p = relax.plan(ctx.S, _all_refused(ctx.S))
    assert p["verdict"] == "no-evidence"
    assert p["moves"] == []
    assert "Measure the design box" in p["reason"]
    # COMPANION: the same all-refused log on the emptied mission is READY —
    # the refusal to guess is about evidence, not about refusals
    ctx2 = _heavy()
    assert relax.plan(ctx2.S, _all_refused(ctx2.S))["verdict"] == "ready"


def test_a_record_from_another_session_is_stale_and_nothing_is_planned():
    from gui.v3 import session

    ctx = _heavy()
    rd = _all_refused(ctx.S)
    assert relax.plan(ctx.S, rd)["verdict"] == "ready"
    ctx.S["mission"]["W_N"] = 5000.0
    session.sync_wing_from_mission(ctx.S)
    p = relax.plan(ctx.S, rd)
    assert p["verdict"] == "stale" and p["overrides"] is None


def test_the_search_radio_does_not_stale_a_reach_but_a_physics_flag_does():
    """The staleness test is about the BOX, and V3.5 put the search policy in
    the same flags dict as the physics. Flipping stage 1's radio to "own"
    removes ``acqf``/``bo_n_init``/``bo_refusal`` while the design box, the
    mission, the pins and every physics flag stay bit-identical — and that
    used to throw away a finished, paid-for reach and demand a relaunch. Both
    halves are asserted here: a function that never staled would pass the
    first."""
    from gui.v3 import config, session

    ctx = _heavy()
    rd = _all_refused(ctx.S)
    p = relax.plan(ctx.S, rd)
    assert p["verdict"] == "ready", p.get("reason")
    was = dict(rd["config"])

    assert session.search_is_recommended(ctx.S)
    assert session.set_search_mode(ctx.S, "own") is True
    now = config.cfg_dict(ctx.S)
    # the PREMISE: the SEARCH POLICY moved and nothing about the problem did
    assert now["flags"] != was["flags"]
    assert set(was["flags"]) - set(now["flags"]) <= set(config.SEARCH_FLAG_KEYS)
    assert config.physics_flags(now) == config.physics_flags(was)
    assert now["bounds_overrides"] == was["bounds_overrides"]

    assert relax.is_stale(ctx.S, rd) is None
    after = relax.plan(ctx.S, rd)
    assert after["verdict"] == "ready", after.get("reason")
    assert [m["label"] for m in after["moves"]] == \
           [m["label"] for m in p["moves"]]
    assert after["overrides"] == p["overrides"]
    # ...and the reached run still flies the RECORD's own search flags, which
    # is why the live ones were never a reason to refuse the plan
    assert after["cfg_dict"]["flags"] == was["flags"] != now["flags"]

    # the other half: a flag about the PHYSICS still stales the record
    ctx.S["wing"]["flags"] = dict(ctx.S["wing"]["flags"],
                                  chord_trend="tip_largest")
    assert config.physics_flags(config.cfg_dict(ctx.S)) \
        != config.physics_flags(was)
    assert "flags has changed" in (relax.is_stale(ctx.S, rd) or "")
    assert relax.plan(ctx.S, rd)["verdict"] == "stale"


def test_the_verdict_prose_says_what_the_numbers_mean():
    ctx = _heavy()
    p = relax.plan(ctx.S, _all_refused(ctx.S))
    lines = [t for t, _ in relax.verdict({"plan": p})]
    assert any("PROVED" in t for t in lines)

    found = relax.verdict({"plan": p, "answer": {
        "kind": "found", "dinf": 0.12, "y": 34.76, "n_evals": 64,
        "rows": [{"label": "b_m", "value": 18.42, "lo": 6.0, "hi": 17.0,
                  "over": 1.42, "d": 0.12, "end": "top"}]}})
    text = " ".join(t for t, _ in found)
    assert "b_m" in text and "own width" in text
    assert "not the closest that exists" in text

    inside = relax.verdict({"plan": p, "answer": {
        "kind": "found", "dinf": 0.0, "rows": [], "y": 30.0, "n_evals": 64}})
    assert any("INSIDE the box you already drew" in t for t, _ in inside)

    stuck = relax.verdict({"plan": p, "answer": {
        "kind": "refused", "refused_fraction": 0.95,
        "own_refused_fraction": 0.95, "n_evals": 64, "nearest": None}})
    assert any("Widening is not the lever" in t for t, _ in stuck)

    limit = relax.verdict({"plan": p, "answer": {
        "kind": "refused", "refused_fraction": 0.06,
        "own_refused_fraction": 0.95, "n_evals": 64, "nearest": None}})
    assert any("a wider box cannot fix a limit" in t for t, _ in limit)


# ============================================================== the shell

def _finished(ctx, record):
    """The state a FINISHED run leaves the shell in — a record on the session
    AND a job on the manager, because the card lives on a view that draws
    nothing at all without one."""
    from gui import nice_app as v1
    from gui.v3 import config

    ctx.S["run"]["record"] = record
    job = v1.RunJob(cfg=config.build_cfg(ctx.S, seed=0),
                    label="probe · seed 0", budget=40)
    job.status = "done"
    ctx.manager.jobs = [job]
    ctx.manager.version += 1
    return ctx


def test_the_button_is_offered_and_priced_off_the_records_own_numbers(capsys):
    ctx = _heavy()
    rd = _all_refused(ctx.S)
    _finished(ctx, rd)
    ctx.render("wing", "run")
    texts = _texts(ctx.views[("wing", "run")])
    assert any("Reach for the nearest box that holds a design" in t
               for t in texts), texts[-10:]
    # the price is DERIVED from the record, never written as a literal
    p = relax.plan(ctx.S, rd)
    assert any(f"{p['budget']} evaluations and seed {p['seed']}" in t
               for t in texts), texts[-6:]
    err = capsys.readouterr().err
    assert "failed to render" not in err and "Traceback" not in err


def test_a_run_that_found_a_design_is_offered_no_reach(capsys):
    """The negative half: this card is an answer to a failure, and a card on
    every run would be read as decoration."""
    from gui.v3 import config

    ctx = _heavy()
    cfg = config.build_cfg(ctx.S, seed=0)
    dim = len(api.PROBLEM_SPECS[cfg.problem_name].param_labels)
    rows = [{"x": [0.5] * dim, "f": 12.0, "g": [0.4], "feasible": True}]
    _finished(ctx, api.partial_result(cfg, rows).to_dict())
    ctx.render("wing", "run")
    assert not any("Reach for the nearest box" in t
                   for t in _texts(ctx.views[("wing", "run")]))
    capsys.readouterr()


def test_the_card_is_reachable_when_there_are_no_margins_to_explain(capsys):
    """The state the reach exists for. ``infeasibility_lines`` is empty for a
    record with no ``eval_g``, and the offers below it are not conditional on
    the evidence above it."""
    ctx = _heavy()
    rd = _all_refused(ctx.S)
    rd.pop("eval_g", None)
    assert diagnose.infeasibility_lines(
        diagnose.infeasibility_report(rd)) == []
    _finished(ctx, rd)
    ctx.render("wing", "run")
    texts = _texts(ctx.views[("wing", "run")])
    assert any("Reach for the nearest box" in t for t in texts), texts[-8:]
    assert any("Measure the design box" in t for t in texts)
    capsys.readouterr()


def test_reaching_changes_nothing_about_the_users_own_box(capsys):
    """The whole honesty claim, and it needs no physics: a finished reach is
    planted and the three things a run may not touch are compared."""
    from gui.v3 import config

    ctx = _heavy()
    rd = _all_refused(ctx.S)
    _finished(ctx, rd)
    p = relax.plan(ctx.S, rd)
    before_bounds = dict(ctx.S["wing"]["bounds"])
    before_cfg = config.cfg_dict(ctx.S)
    before_jobs = ctx.manager.jobs
    ctx.S["run"]["relax"] = {
        "plan": p, "busy": False, "stamp": 1.0, "step": 0,
        "answer": {"kind": "found", "x": [0.5] * len(p["labels"]),
                   "y": 30.0, "dinf": 0.0, "d1": 0.0, "rows": [],
                   "n_evals": 29}}
    ctx.render("wing", "run")
    assert dict(ctx.S["wing"]["bounds"]) == before_bounds
    assert config.cfg_dict(ctx.S) == before_cfg
    assert ctx.S["run"]["record"] is rd
    assert ctx.manager.jobs is before_jobs and len(ctx.manager.jobs) == 1
    capsys.readouterr()


def test_adopting_writes_the_users_own_rows_and_launches_nothing(capsys):
    from gui.v3 import config, session

    ctx = _heavy()
    rd = _all_refused(ctx.S)
    _finished(ctx, rd)
    p = relax.plan(ctx.S, rd)
    labels, B = p["labels"], np.asarray(p["bounds"], float)
    x = 0.5 * (B[:, 0] + B[:, 1])
    k = labels.index("S_m2")
    x[k] = B[k, 1] + 0.10 * (B[k, 1] - B[k, 0])       # outside exactly one row
    ctx.S["run"]["relax"] = {
        "plan": p, "busy": False, "stamp": 1.0, "step": 0,
        "answer": dict(diagnose.outside_box(p["bounds"], labels, x),
                       kind="found", x=[float(v) for v in x], y=30.0,
                       n_evals=29)}
    started = []
    ctx.manager.start = lambda jobs: started.append(jobs)   # must NOT be hit

    was_shell = set(session.bounds_source(ctx.S))
    ctx.act("relax_adopt")
    # ONLY the row it needed becomes the USER's. The others in W["bounds"] are
    # the SIZE rows the shell derives from the mission — they were there before
    # the reach ran and still say so, which is the point of the distinction:
    # a key set cannot tell "the shell wrote this" from "the user typed it".
    assert set(session.bounds_source(ctx.S)) == was_shell - {"S_m2"}
    assert config.effective_bounds(ctx.S)["S_m2"][1] == "user"
    for row in was_shell - {"S_m2"}:
        assert config.effective_bounds(ctx.S)[row][1] == "mission"
    lo, hi = ctx.S["wing"]["bounds"]["S_m2"]
    assert lo < x[k] < hi
    assert session.start_design(ctx.S) is not None
    api.check_x_seed(config.build_cfg(ctx.S))     # the api accepts the pair
    assert started == [], "adopting must not launch"
    capsys.readouterr()


def test_a_record_from_another_box_says_so_on_the_run_page(capsys):
    """Derived by comparison, so it is also true when a user simply edits a
    row and does not re-launch."""
    ctx = _heavy()
    rd = _all_refused(ctx.S)
    _finished(ctx, rd)
    assert relax.banner(ctx.S, rd) is None            # the matching half
    ctx.S["wing"]["bounds"]["S_m2"] = [93.0, 200.0]
    said = relax.banner(ctx.S, rd)
    assert said and "S_m2" in said and "93 – 200" in said
    # the band the record ACTUALLY searched, not the words "its own band":
    # a user cannot compare a number against a phrase
    assert "8 – 22" in said, said
    ctx.render("wing", "run")
    assert any("searched a different box" in t
               for t in _texts(ctx.views[("wing", "run")]))
    capsys.readouterr()


def test_launch_is_refused_while_a_reach_holds_the_solver(capsys):
    ctx = _heavy()
    _finished(ctx, _all_refused(ctx.S))
    started = []
    ctx.manager.start = lambda jobs: started.append(jobs)
    ctx.S["run"]["relax"] = {"busy": True, "stamp": 1.0}
    ctx.act("launch")
    assert started == []
    ctx.S["run"]["relax"] = {"busy": False, "stamp": 1.0}
    ctx.act("launch")
    assert started, "the mirror: with no reach running, launch launches"
    capsys.readouterr()


@pytest.mark.parametrize("state", ["idle", "busy", "found", "inside",
                                   "refused", "error"])
def test_every_state_of_the_card_renders(state, capsys):
    ctx = _heavy()
    rd = _all_refused(ctx.S)
    _finished(ctx, rd)
    p = relax.plan(ctx.S, rd)
    n = len(p["labels"])
    payloads = {
        "idle": None,
        "busy": {"plan": p, "busy": True, "n": 7, "budget": 29, "stamp": 1.0},
        "found": {"plan": p, "busy": False, "stamp": 1.0, "answer": {
            "kind": "found", "x": [0.5] * n, "y": 22.0, "dinf": 0.3,
            "d1": 0.3, "n_evals": 29,
            "rows": [{"label": "S_m2", "value": 26.2, "lo": 8.0, "hi": 22.0,
                      "over": 4.2, "d": 0.3, "end": "top"}]}},
        "inside": {"plan": p, "busy": False, "stamp": 1.0, "answer": {
            "kind": "found", "x": [0.5] * n, "y": 22.0, "dinf": 0.0,
            "d1": 0.0, "rows": [], "n_evals": 29}},
        "refused": {"plan": p, "busy": False, "stamp": 1.0, "answer": {
            "kind": "refused", "refused_fraction": 0.9,
            "own_refused_fraction": 1.0, "n_evals": 29, "nearest": None}},
        "error": {"plan": p, "busy": False, "stamp": 1.0,
                  "error": "ValueError: nope"},
    }
    if payloads[state] is not None:
        ctx.S["run"]["relax"] = payloads[state]
    ctx.render("wing", "run")
    err = capsys.readouterr().err
    assert "failed to render" not in err and "Traceback" not in err
    # ...and the state SAID something. A view is allowed to be non-fatal, which
    # is exactly why silence has to be asserted against: mutation-checked —
    # an early `return False` at the top of _render_relax deletes the error
    # state from the card entirely and a no-traceback test stays green.
    says = {
        "idle": "Reach for the nearest box that holds a design",
        "busy": "searching the reached box",
        "found": "Move that row and start from this design",
        "inside": "INSIDE the box you already drew",
        # 0.90 against the user's own 1.00: the refusals FELL, so it is the
        # limit and not the box (relax.verdict's other branch is asserted in
        # test_the_verdict_prose_says_what_the_numbers_mean)
        "refused": "a wider box cannot fix a limit",
        "error": "ValueError: nope",
    }[state]
    texts = _texts(ctx.views[("wing", "run")])
    assert any(says in t for t in texts), (state, says, texts[-12:])


def test_the_session_declares_both_background_keys():
    """A key that only ever existed because a worker had run once would
    survive File ▸ New session carrying the previous session's answer."""
    from gui.v3 import session

    S = session.make_session("air")
    assert "relax" in S["run"] and "box_probe" in S["run"]
    assert S["run"]["relax"] is None and S["run"]["box_probe"] is None


def test_a_new_session_clears_a_finished_reach(capsys):
    from gui.v3 import app

    ctx = _heavy()
    ctx.S["run"]["relax"] = {"answer": {"kind": "found"}, "stamp": 1.0}
    app._new_session(ctx)
    assert not (ctx.S["run"].get("relax") or {})
    capsys.readouterr()


def test_the_three_actions_are_registered(capsys):
    """``ctx.act`` on an unregistered name silently returns None, so without
    this a typo in a handler name is a green test and a dead button."""
    ctx = _heavy()
    assert {"relax_reach", "relax_stop", "relax_adopt"} <= set(ctx.actions)
    capsys.readouterr()


# ============================================ the press, and what it touches

def _reached_rows(plan, kinds):
    """Designs inside the REACHED box, one per (feasible, distance) request.

    ``kinds`` is a list of ``(feasible, frac)`` — ``frac`` places the design
    across the reached row of the FIRST moved label, so a caller can put a
    good design far out and a worse one near in.
    """
    B = np.asarray(plan["bounds"], float)
    labels = plan["labels"]
    k = labels.index(plan["moves"][0]["label"])
    new_lo, new_hi = plan["overrides"][labels[k]]
    rows = []
    for i, (feasible, frac) in enumerate(kinds):
        x = 0.5 * (B[:, 0] + B[:, 1])
        x[k] = new_lo + frac * (new_hi - new_lo)
        rows.append({"x": [float(v) for v in x],
                     "f": (-100.0 if not feasible else 40.0 - 10.0 * i),
                     "g": [(-1.0 if not feasible else 0.2)],
                     "feasible": bool(feasible)})
    return rows


def _drive_reach(ctx, fake_run, step=0, timeout=30.0):
    """Press the button with ``api.run`` replaced, and wait for the worker."""
    import time as _t

    from aerobo import api as _api

    real = _api.run
    _api.run = fake_run
    try:
        ctx.act("relax_reach", step)
        st = ctx.S["run"]["relax"]
        t0 = _t.time()
        while st.get("busy") and _t.time() - t0 < timeout:
            _t.sleep(0.02)
        assert not st.get("busy"), "the worker never finished"
        return st
    finally:
        _api.run = real


def test_pressing_the_button_runs_the_planned_config_and_touches_nothing_else(
        capsys):
    """The feature's central architectural claim, and the one no planted
    payload can test: one press is one search, of the box the plan named, on a
    COPY of the problem. ``RunManager.start`` REPLACES its queue, so routing
    this through the manager would delete the run the user is looking at and
    repoint the readouts and the convergence trace at a box nobody asked for.
    """
    from gui.v3 import config

    ctx = _heavy()
    rd = _all_refused(ctx.S)
    _finished(ctx, rd)
    plan = relax.plan(ctx.S, rd)
    assert plan["verdict"] == "ready", plan.get("reason")

    before_bounds = dict(ctx.S["wing"]["bounds"])
    before_cfg = config.cfg_dict(ctx.S)
    before_jobs = ctx.manager.jobs
    started, seen = [], {}
    ctx.manager.start = lambda jobs: started.append(jobs)

    def fake_run(cfg, progress_cb=None, stop_rule=None, results_dir=None,
                 **kw):
        from aerobo import api as _api

        seen["cfg"], seen["results_dir"] = cfg, results_dir
        seen["stop_rule"] = stop_rule
        # a GOOD design far out, and a WORSE one near the user's own box
        rows = _reached_rows(plan, [(False, 0.5), (True, 0.95), (True, 0.02)])
        for i, r in enumerate(rows):
            if progress_cb is not None:
                progress_cb(i + 1, r["f"], **r)
        return _api.partial_result(cfg, rows)

    st = _drive_reach(ctx, fake_run)

    # ...the run it made is the run the plan priced
    cfg = seen["cfg"]
    assert {k: getattr(cfg, k) for k in plan["cfg_dict"]} == \
        dict(plan["cfg_dict"])
    assert cfg.x_seed is None, "a seed armed in the old box does not travel"
    assert seen["results_dir"], "the reach persists like any other run"
    # ...the manager and the user's own record are untouched
    assert started == [], "the reach must never go through the manager"
    assert ctx.manager.jobs is before_jobs and len(ctx.manager.jobs) == 1
    assert ctx.S["run"]["record"] is rd
    # ...and so is the box
    assert dict(ctx.S["wing"]["bounds"]) == before_bounds
    assert config.cfg_dict(ctx.S) == before_cfg
    # ...and the answer is the CLOSEST feasible design, not the best-scoring
    ans = st["answer"]
    assert ans["kind"] == "found"
    assert ans["y"] == pytest.approx(20.0), "40.0 is the far one"
    assert st["error"] is None
    capsys.readouterr()


def test_a_reach_that_raises_lands_on_the_card_and_not_in_the_shell(capsys):
    ctx = _heavy()
    _finished(ctx, _all_refused(ctx.S))

    def boom(cfg, **kw):
        raise ValueError("the solver fell over")

    st = _drive_reach(ctx, boom)
    assert not st["busy"] and st["error"] and "fell over" in st["error"]
    assert st["answer"] is None
    ctx.render("wing", "run")
    assert any("fell over" in t for t in _texts(ctx.views[("wing", "run")]))
    err = capsys.readouterr().err
    assert "failed to render" not in err and "Traceback" not in err


def test_stopping_a_reach_is_a_stop_rule_and_never_a_raise(capsys):
    """Raising out of a progress callback throws away every evaluation already
    paid for, which on an XFOIL family is the whole run."""
    from gui.v3 import config

    ctx = _heavy()
    rd = _all_refused(ctx.S)
    _finished(ctx, rd)
    plan = relax.plan(ctx.S, rd)
    saw = {}

    def fake_run(cfg, progress_cb=None, stop_rule=None, results_dir=None,
                 **kw):
        from aerobo import api as _api

        rows = _reached_rows(plan, [(True, 0.5), (True, 0.6)])
        # the STOP arrives mid-run, the way the button delivers it
        ctx.S["run"]["relax"]["stop"] = True
        saw["progress_raised"] = False
        try:
            progress_cb(1, rows[0]["f"], **rows[0])
        except BaseException:                     # noqa: BLE001
            saw["progress_raised"] = True
            raise
        saw["stop_rule_said"] = bool(stop_rule(1, rows[0]["f"]))
        return _api.partial_result(cfg, rows[:1])

    st = _drive_reach(ctx, fake_run)
    assert saw["progress_raised"] is False, (
        "the progress callback must never raise — a stop is a stop rule")
    assert saw["stop_rule_said"] is True
    assert st["error"] is None and not st["busy"]
    assert st["stop"] is False, "the flag is cleared for the next press"
    capsys.readouterr()
    del config


def test_the_worker_stamps_before_its_first_evaluation(capsys):
    """``api.run``'s first progress callback does not arrive until the first
    evaluation has been PAID FOR, and ``poll`` repaints on a CHANGED stamp —
    so a worker that stamped only at the end left its own spinner and its Stop
    button unpainted for the whole search."""
    ctx = _heavy()
    _finished(ctx, _all_refused(ctx.S))
    seen = {}

    def fake_run(cfg, progress_cb=None, **kw):
        from aerobo import api as _api

        st = ctx.S["run"]["relax"]
        seen["busy"], seen["stamp"] = st.get("busy"), st.get("stamp")
        return _api.partial_result(cfg, [])

    _drive_reach(ctx, fake_run)
    assert seen["busy"] is True
    assert seen["stamp"] is not None, "no stamp before the first evaluation"
    capsys.readouterr()


def test_a_failed_adopt_puts_every_bound_back(capsys):
    """The rollback, which is where a silent widening would hide: adopt writes
    the rows and only then asks the api to accept the seed."""
    from gui.v3 import config

    ctx = _heavy()
    rd = _all_refused(ctx.S)
    _finished(ctx, rd)
    p = relax.plan(ctx.S, rd)
    labels, B = p["labels"], np.asarray(p["bounds"], float)
    ctx.S["wing"]["bounds"]["taper"] = [0.3, 0.9]          # a row of the user's
    before = dict(ctx.S["wing"]["bounds"])
    x = 0.5 * (B[:, 0] + B[:, 1])
    k = labels.index("S_m2")
    x[k] = B[k, 1] * 2.0
    ctx.S["run"]["relax"] = {
        "plan": p, "busy": False, "stamp": 1.0, "step": 0,
        "answer": dict(diagnose.outside_box(p["bounds"], labels, x),
                       kind="found", x=[float(v) for v in x], y=30.0,
                       n_evals=29)}
    # make the api refuse the seed, whatever else is true of it
    x_bad = list(x)
    x_bad.append(0.0)                       # one value too many
    ctx.S["run"]["relax"]["answer"]["x"] = x_bad

    ctx.act("relax_adopt")
    assert dict(ctx.S["wing"]["bounds"]) == before, "a refused adopt must " \
        "leave the box exactly as it found it"
    assert config.cfg_dict(ctx.S).get("x_seed") is None
    capsys.readouterr()


def test_a_reach_in_flight_holds_the_session_and_the_stop_button(capsys):
    from gui.v3 import app

    ctx = _heavy()
    _finished(ctx, _all_refused(ctx.S))
    ctx.S["run"]["relax"] = {"busy": True, "stamp": 1.0}
    keep = dict(ctx.S["run"]["relax"])
    app._new_session(ctx)
    assert ctx.S["run"]["relax"] == keep, "a new session must not run under " \
        "a worker that is still writing into it"
    app._stop_all(ctx) if hasattr(app, "_stop_all") else ctx.act("relax_stop")
    assert ctx.S["run"]["relax"]["stop"] is True
    capsys.readouterr()


def test_poll_logs_the_outcome_once_and_repaints_the_run_view(capsys):
    ctx = _heavy()
    rd = _all_refused(ctx.S)
    _finished(ctx, rd)
    p = relax.plan(ctx.S, rd)
    said = []
    ctx.output = type("L", (), {"write": lambda _s, t, lvl="info":
                                said.append((t, lvl))})()
    ctx.S["ui"]["selected"], ctx.S["ui"]["tab"]["wing"] = "wing", "run"
    ctx.S["run"]["relax"] = {
        "plan": p, "busy": False, "stamp": 2.0, "step": 0,
        "answer": {"kind": "found", "x": [0.5] * len(p["labels"]), "y": 30.0,
                   "dinf": 0.4, "d1": 0.4, "n_evals": 29,
                   "rows": [{"label": "S_m2", "value": 27.0, "lo": 8.0,
                             "hi": 22.0, "over": 5.0, "d": 0.36,
                             "end": "top"}]}}
    for poll in ctx.polls:
        poll()
    assert any("the reach found a design" in t for t, _ in said), said
    assert any("Your design box is unchanged" in t for t, _ in said)
    n = len(said)
    for poll in ctx.polls:      # the stamp has not moved: it must not re-log
        poll()
    assert len(said) == n
    capsys.readouterr()


def test_a_second_press_reaches_further_on_a_measured_move(capsys):
    """STEPS = (1.0, 2.5): the second press is allowed to reach 2.5 widths,
    because that is what pressing it means. Comparing a scaled move against
    the unscaled limit made the second press able only to refuse."""
    from gui.v3 import session

    ctx = _heavy()
    ctx.S["mission"]["W_N"] = 700.0
    session.sync_wing_from_mission(ctx.S)
    rd = _riding(ctx.S, "b_m", "upper", slope=0.01,
                 deficit=_deficit_for(ctx.S, "b_m", 0.01, 0.6))
    p0, p1 = relax.plan(ctx.S, rd, step=0), relax.plan(ctx.S, rd, step=1)
    assert p0["verdict"] == "ready" and p1["verdict"] == "ready", (
        p0.get("reason"), p1.get("reason"))
    m0 = next(m for m in p0["moves"] if m["label"] == "b_m")
    m1 = next(m for m in p1["moves"] if m["label"] == "b_m")
    assert (m1["new_hi"] - m1["hi"]) == pytest.approx(
        relax.STEPS[1] * (m0["new_hi"] - m0["hi"]))
    assert p1["cfg_dict"] != p0["cfg_dict"]
    capsys.readouterr()


def test_reach_further_is_not_offered_when_it_would_repeat_the_search(capsys):
    """A gate move is a PROOF and is taken whole, so the scale does not touch
    it — and on the emptied-mission case, the one this feature was built for,
    a second press would spend the whole budget on a bit-identical answer."""
    ctx = _heavy()
    rd = _all_refused(ctx.S)
    _finished(ctx, rd)
    p0, p1 = relax.plan(ctx.S, rd, step=0), relax.plan(ctx.S, rd, step=1)
    assert p0["cfg_dict"] == p1["cfg_dict"], "the premise: the gate ignores it"
    ctx.S["run"]["relax"] = {
        "plan": p0, "busy": False, "stamp": 1.0, "step": 0,
        "answer": {"kind": "found", "x": [0.5] * len(p0["labels"]), "y": 30.0,
                   "dinf": 0.0, "d1": 0.0, "rows": [], "n_evals": 29}}
    ctx.render("wing", "run")
    assert not any("Reach further" in t
                   for t in _texts(ctx.views[("wing", "run")]))
    capsys.readouterr()


def test_a_blocked_reason_quotes_a_distance_past_the_limit_it_cites(capsys):
    """A refusal whose own number sits below the threshold it names is not a
    reason. The printed distance is the one THIS press would have reached."""
    from gui.v3 import session

    ctx = _heavy()
    ctx.S["mission"]["W_N"] = 700.0
    session.sync_wing_from_mission(ctx.S)
    p = relax.plan(ctx.S, _riding(ctx.S, "b_m", "upper",
                                  slope=1e-4, deficit=5.0))
    assert p["verdict"] == "blocked"
    got = float(re.search(r"move ([0-9.e+-]+) of its own width",
                          p["reason"]).group(1))
    cited = float(re.search(r"Past ([0-9.]+) width", p["reason"]).group(1))
    assert got > cited, p["reason"]
    capsys.readouterr()


def test_an_unconstrained_record_does_not_get_a_sentence_about_limits(capsys):
    """A record with no margins at all has no binding constraint, so there is
    nothing for the reach to fit against and nothing honest to say."""
    ctx = _heavy()
    rd = _all_refused(ctx.S)
    rd["eval_g"] = None
    rd["is_constrained"] = False
    p = relax.plan(ctx.S, rd)
    # the gate still proves this box empty, so the plan stands on the proof
    assert p["verdict"] == "ready"
    assert all(m["source"] == "gate" for m in p["moves"])
    assert diagnose.refused_fraction(rd) is None
    capsys.readouterr()


def test_a_run_that_finishes_with_nothing_paints_its_card_on_that_tick(capsys):
    """Found by running the shell, not by a test: ``poll`` renders the run view
    and THEN publishes the record, and the whole no-design card is read off
    that record — so a run that came back with nothing showed "the criteria
    appear here once a run finishes" until the user happened to switch tabs.
    """
    from gui import nice_app as v1
    from gui.v3 import config

    ctx = _heavy()
    ctx.S["ui"]["selected"], ctx.S["ui"]["tab"]["wing"] = "wing", "run"
    cfg = config.build_cfg(ctx.S, seed=0)
    dim = len(api.PROBLEM_SPECS[cfg.problem_name].param_labels)
    result = api.partial_result(cfg, [
        {"x": [0.5] * dim, "f": -100.0, "g": [-1.0], "feasible": False}
        for _ in range(6)])
    job = v1.RunJob(cfg=cfg, label="probe · seed 0", budget=6)
    job.status, job.result = "done", result
    ctx.manager.jobs = [job]
    ctx.manager.version += 1
    assert ctx.S["run"]["record"] is None, "the premise: nothing published yet"

    for poll in ctx.polls:
        poll()
    assert isinstance(ctx.S["run"]["record"], dict), "poll must publish it"
    texts = _texts(ctx.views[("wing", "run")])
    assert not any("appear here once a run finishes" in t for t in texts), \
        "the one sentence a FINISHED run that found nothing must never get"
    assert any("Reach for the nearest box that holds a design" in t
               for t in texts), texts[-10:]
    capsys.readouterr()


def test_a_far_answer_is_quoted_in_row_widths_not_in_percent(capsys):
    """13.9 row-widths reads as "1387 %" under a percent format — right, and
    unreadable beside the headline three lines above it."""
    ctx = _heavy()
    rd = _all_refused(ctx.S)
    p = relax.plan(ctx.S, rd)
    far = " ".join(t for t, _ in relax.verdict({"plan": p, "answer": {
        "kind": "found", "dinf": 13.87, "y": 25.09, "n_evals": 35,
        "rows": [{"label": "S_m2", "value": 216.2, "lo": 8.0, "hi": 22.0,
                  "over": 194.2, "d": 13.87, "end": "top"}]}}))
    assert "1387%" not in far and "1387 %" not in far, far
    assert "13.9 times that row's own width" in far, far
    # ...and a near one is still a percentage, which is what reads there
    near = " ".join(t for t, _ in relax.verdict({"plan": p, "answer": {
        "kind": "found", "dinf": 0.12, "y": 25.09, "n_evals": 35,
        "rows": [{"label": "b_m", "value": 18.4, "lo": 6.0, "hi": 17.0,
                  "over": 1.42, "d": 0.129, "end": "top"}]}}))
    assert "13% of that row's own width" in near, near
    capsys.readouterr()


def test_the_reach_moves_the_span_row_when_no_area_band_can_do_it(capsys):
    """A box whose SPAN row is what stops it must be reached for, not refused.

    Reported: the recommendations "only for area", and none of them solved
    the problem. With the span row capped at 12 m the two cheap gates want
    the area moved in opposite directions, so the reach planned a single
    area move, ``_empty_under`` correctly found the reached box still empty,
    and the whole card came back ``blocked`` — the honest answer to a plan
    that could not work, and no answer at all to the user.

    The claim is about the OUTCOME: the plan is ready, and the box it reaches
    is one these gates no longer rule out. Both survive a formula that is
    merely self-consistent, which an assertion on the band would not.
    """
    from gui.v3 import config

    ctx = _heavy()
    ctx.S["wing"]["bounds"]["b_m"] = [6.0, 12.0]
    p = relax.plan(ctx.S, _all_refused(ctx.S))
    assert p["verdict"] == "ready", p["reason"]
    moved = {m["label"] for m in p["moves"]}
    assert "b_m" in moved, moved
    # ...and the reached box is really not empty — the same question
    # ``_empty_under`` asks, asked again of the overrides the plan publishes
    reached = api.RunConfig(**dict(config.cfg_dict(ctx.S),
                                   bounds_overrides=p["overrides"]))
    assert not [f for f in api.size_box_conflicts(reached) if f["empty"]]
    capsys.readouterr()
