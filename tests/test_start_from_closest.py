"""A run that found nothing hands back the design that came CLOSEST.

The user's own proposal, and the only continuation that is monotone: the
incumbent of a constrained run is a max over evaluated points with every
margin >= 0, and a seed is evaluated first and never screened away
(``api.RunConfig.x_seed``), so a run TOLD a design cannot do worse for having
been told. Told a design that nearly flies, its surrogate starts on the
constraint boundary instead of on a flat prior.

Two halves are tested: the api's validation, which refuses rather than
repairs (a clipped seed is a different design from the one that was measured),
and the shell's, where the button lives.
"""
import numpy as np
import pytest

from aerobo import api

CASE = "trim wing + free planform"


def _cfg(**kw):
    return api.RunConfig(problem_name=CASE, optimiser="bo", budget=8, seed=0,
                         **kw)


def _box():
    built = api.PROBLEM_SPECS[CASE].build({}, {}, None)
    return np.asarray(built.bounds, dtype=float), list(built.param_labels)


def test_a_seed_inside_the_box_is_accepted_and_is_evaluated_first():
    """Accepted, and then USED: the run's first evaluation must be the design
    it was given, or "cannot do worse" is a claim about nothing."""
    box, _labels = _box()
    mid = 0.5 * (box[:, 0] + box[:, 1])
    cfg = _cfg(x_seed=[float(v) for v in mid])
    api.check_x_seed(cfg)                     # does not raise
    res = api.run(cfg)
    first = np.asarray((res.eval_x or [[]])[0], dtype=float)
    assert first.size == mid.size
    assert np.allclose(first, mid, rtol=0.0, atol=1e-12)


def test_a_seed_outside_the_box_is_refused_by_name_not_clipped():
    """The refusal has to name the row and both bands: silently clipping would
    start from a design nobody evaluated while calling it the closest one."""
    box, _labels = _box()
    bad = list(0.5 * (box[:, 0] + box[:, 1]))
    bad[0] = float(box[0, 1]) * 10.0
    with pytest.raises(ValueError, match="outside the box"):
        api.check_x_seed(_cfg(x_seed=bad))


def test_a_seed_that_disagrees_with_a_pin_is_refused():
    """Two designs got mixed up, and starting from either would be a lie about
    which one the run is continuing."""
    box, labels = _box()
    mid = list(0.5 * (box[:, 0] + box[:, 1]))
    i = labels.index("taper")
    with pytest.raises(ValueError, match="different design"):
        api.check_x_seed(_cfg(x_seed=mid,
                              pinned={"taper": float(mid[i]) * 0.5 + 0.1}))


def test_the_wrong_length_is_refused():
    box, _labels = _box()
    mid = list(0.5 * (box[:, 0] + box[:, 1]))
    with pytest.raises(ValueError, match="dimensions|design"):
        api.check_x_seed(_cfg(x_seed=mid[:-1]))


# ------------------------------------------------------------- the shell
def _shell():
    from gui.v3 import session
    from gui.v3.app import assemble

    ctx = assemble()
    ctx.act("accept_mission")
    assert session.set_planform(ctx.S, "free") == []
    return ctx


def test_v3_arming_a_start_design_reaches_the_config_and_can_be_taken_back():
    from gui.v3 import config, session

    ctx = _shell()
    cfg = config.build_cfg(ctx.S)
    built = api.PROBLEM_SPECS[cfg.problem_name].build(
        cfg.mission_kwargs, cfg.flags, cfg.bounds_overrides)
    box = np.asarray(built.bounds, dtype=float)
    mid = [float(v) for v in 0.5 * (box[:, 0] + box[:, 1])]

    assert "x_seed" not in config.cfg_dict(ctx.S), (
        "an untouched session must send no seed — the V2 contract is "
        "field-for-field")
    assert session.start_from_design(ctx.S, mid) is None
    assert config.cfg_dict(ctx.S)["x_seed"] == mid
    session.clear_start_design(ctx.S)
    assert "x_seed" not in config.cfg_dict(ctx.S)


def test_v3_a_design_the_run_could_not_return_is_refused_and_rolled_back():
    """The shell must not be left holding a seed the next launch will raise on
    — the state after a refusal is the state before it."""
    from gui.v3 import config, session

    ctx = _shell()
    cfg = config.build_cfg(ctx.S)
    built = api.PROBLEM_SPECS[cfg.problem_name].build(
        cfg.mission_kwargs, cfg.flags, cfg.bounds_overrides)
    box = np.asarray(built.bounds, dtype=float)
    good = [float(v) for v in 0.5 * (box[:, 0] + box[:, 1])]
    assert session.start_from_design(ctx.S, good) is None

    bad = list(good)
    bad[0] = float(box[0, 1]) * 10.0
    err = session.start_from_design(ctx.S, bad)
    assert err and "outside the box" in err
    assert session.start_design(ctx.S) == good, "the refusal kept the bad seed"
    api.check_x_seed(config.build_cfg(ctx.S))     # still launchable
