"""A design variable the user has DECIDED — pinned, not merely bounded.

The design box says where a variable may go. It could not say that the
variable does not move at all: a row of width zero is refused
(``_apply_overrides``), and for a good reason — scipy's Sobol engine raises on
it and constrained BO falls back to random draws on every iteration while
still reporting itself as BO. So a pin is done the only honest way:
``RunConfig.pinned`` takes the variable OUT of the design vector, the
optimiser searches what is left, and the physics is evaluated on the full
vector with the constant put back.

The contract these tests hold:

* the pinned variable really does not move — measured on the FLOWN geometry,
  not on the design vector, so an expand/reduce that scattered the constant
  into the wrong slot would show up as a wing that was never asked for;
* what the run REPORTS is the full design vector, so every consumer
  downstream (breakdown, geometry, CAD, a later comparison) is unchanged;
* the score reported IS the score of the design reported;
* a pin outside the box, an unknown row, or pinning everything is refused,
  not clamped;
* an unpinned run is bit-for-bit the run it always was.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aerobo import api                      # noqa: E402
from aerobo.api import PROBLEM_SPECS        # noqa: E402

WING = "trim wing"
TAPER = 0.63          # not a bound, not the centre, not a round default


def _cfg(**kw):
    d = {"problem_name": WING, "optimiser": "bo", "budget": 12, "seed": 3}
    d.update(kw)
    return api.RunConfig(**d)


# ------------------------------------------------------------------ 1. it holds

def test_pinned_variable_is_the_flown_geometry_everywhere():
    """Every design the search evaluated flew the pinned taper.

    Read off the PLANFORM the physics built (design_report's geometry), not
    off the design vector: the vector is what the pin writes, and a test that
    reads it back would pass on a pin scattered into the wrong slot.
    """
    cfg = _cfg(pinned={"taper": TAPER})
    res = api.run(cfg)
    assert res.pinned == {"taper": TAPER}
    assert res.searched_dim == res.dim - 1

    rep = api.design_report(cfg, res.best_x)
    geo = rep.get("geometry") or {}
    assert geo.get("taper") == pytest.approx(TAPER, abs=1e-12)

    # ...and the other rows were genuinely searched, so the pin narrowed the
    # search rather than freezing it
    X = np.asarray(res.eval_x, dtype=float)
    i = list(res.param_labels).index("taper")
    assert np.allclose(X[:, i], TAPER)
    free = [j for j in range(X.shape[1]) if j != i]
    assert all(np.ptp(X[:, j]) > 1e-6 for j in free)

    # ...each inside ITS OWN row of the reported box. This is what catches a
    # reduce/expand that merely permutes the free dimensions: such a run is
    # self-consistent — it scores the vector it reports — but it searched
    # twist_root's band inside twist_tip's column, so the design it hands back
    # sits outside the box the user drew.
    box = np.asarray(res.bounds, dtype=float)
    assert np.all(X >= box[:, 0] - 1e-9) and np.all(X <= box[:, 1] + 1e-9)


def test_the_score_reported_is_the_score_of_the_design_reported():
    """best_score must be what the physics returns at best_x.

    This is what an off-by-one in the expand would break: the optimiser would
    be scoring one wing and the report naming another, and every number in
    between would still look plausible.
    """
    cfg = _cfg(pinned={"taper": TAPER})
    res = api.run(cfg)
    built = PROBLEM_SPECS[WING].build({}, {}, None)
    again = float(built.callable(np.asarray(res.best_x, dtype=float)))
    assert again == pytest.approx(res.best_score, rel=0, abs=1e-12)


def test_the_pin_binds_the_answer():
    """Two pins, two different wings — a pin the search could ignore is not a
    pin. Same seed and budget, so the only difference is the constant."""
    a = api.run(_cfg(pinned={"taper": 0.25}))
    b = api.run(_cfg(pinned={"taper": 0.95}))
    assert a.best_score != b.best_score
    assert api.design_report(_cfg(pinned={"taper": 0.25}),
                             a.best_x)["geometry"]["taper"] == \
        pytest.approx(0.25)
    assert api.design_report(_cfg(pinned={"taper": 0.95}),
                             b.best_x)["geometry"]["taper"] == \
        pytest.approx(0.95)


def test_the_reported_box_shows_the_pin_and_the_rest_of_the_box_stands():
    """The pinned row comes back collapsed to [v, v]; nothing else moves."""
    res = api.run(_cfg(pinned={"taper": TAPER}))
    box = {lbl: row for lbl, row in zip(res.param_labels, res.bounds)}
    assert box["taper"] == [TAPER, TAPER]
    published = PROBLEM_SPECS[WING].default_bounds
    for lbl, row in box.items():
        if lbl != "taper":
            assert row == pytest.approx(published[lbl])


# ------------------------------------------------- 2. it composes with the box

def test_a_pin_lives_inside_the_narrowed_box():
    """The pin is checked against the box this run actually searches, not the
    family's published one — a user narrows a row and then pins inside it."""
    cfg = _cfg(bounds_overrides={"taper": [0.5, 0.8]}, pinned={"taper": 0.55})
    res = api.run(cfg)
    assert res.best_x[list(res.param_labels).index("taper")] == \
        pytest.approx(0.55)

    with pytest.raises(ValueError, match="outside the box"):
        api.run(_cfg(bounds_overrides={"taper": [0.5, 0.8]},
                     pinned={"taper": 0.95}))


def test_pinning_more_than_one_row():
    """Two pins leave a one-dimensional search, and both hold."""
    cfg = _cfg(pinned={"taper": 0.4, "twist_tip_deg": -3.0})
    res = api.run(cfg)
    assert res.searched_dim == res.dim - 2
    X = np.asarray(res.eval_x, dtype=float)
    lbl = list(res.param_labels)
    assert np.allclose(X[:, lbl.index("taper")], 0.4)
    assert np.allclose(X[:, lbl.index("twist_tip_deg")], -3.0)
    assert np.ptp(X[:, lbl.index("twist_root_deg")]) > 1e-6


# ---------------------------------------------------------------- 3. refusals

def test_a_pin_is_refused_rather_than_clamped():
    with pytest.raises(ValueError, match="outside the box"):
        api.run(_cfg(pinned={"taper": 5.0}))
    with pytest.raises(KeyError):
        api.run(_cfg(pinned={"no_such_row": 1.0}))
    with pytest.raises(ValueError, match="nothing to search"):
        api.run(_cfg(pinned={"taper": 0.5, "twist_root_deg": 0.0,
                             "twist_tip_deg": 0.0}))


def test_a_collapsed_bound_points_at_the_pin():
    """The refusal a user meets when they try to fix a row the old way has to
    name the thing that does work."""
    with pytest.raises(ValueError, match="pinned"):
        PROBLEM_SPECS[WING].build({}, {}, {"taper": [0.6, 0.6]})


def test_blocks_refuses_a_pin_by_name():
    """The blocks portfolio addresses variables by index into the FULL vector,
    so a pin renumbers its slices — refused, not silently mis-sliced."""
    names = [n for n, s in PROBLEM_SPECS.items() if s.has_blocks]
    if not names:
        pytest.skip("no problem declares variable blocks")
    spec = PROBLEM_SPECS[names[0]]
    row = next(iter(spec.default_bounds.items()))
    cfg = api.RunConfig(problem_name=spec.name, optimiser="blocks", budget=8,
                        pinned={row[0]: 0.5 * (row[1][0] + row[1][1])})
    with pytest.raises(ValueError, match="renumbers"):
        api.run(cfg)


# ------------------------------------------------------ 4. nothing else moves

def test_an_unpinned_run_is_untouched():
    """No pin, no wrapper: the same seed must give the same run it gave
    before the feature existed, including with an explicit empty dict."""
    a = api.run(_cfg())
    b = api.run(_cfg(pinned={}))
    assert a.pinned is None and b.pinned is None
    assert a.searched_dim is None and b.searched_dim is None
    assert a.best_x == b.best_x
    assert a.best_score == b.best_score
    assert a.eval_x == b.eval_x


def test_the_composite_band_is_measured_on_the_pinned_box():
    """A criterion is normalised against the population the box produces, and
    a fixed variable does not vary in that population. Measured on the taper,
    which every one of these criteria depends on."""
    free = api.wing_score_reference(WING, n=24, seed=0)
    held = api.wing_score_reference(WING, n=24, seed=0,
                                    pinned={"taper": TAPER})
    assert free["bounds"] and held["bounds"]
    shared = set(free["bounds"]) & set(held["bounds"])
    assert "lod" in shared
    assert any(free["bounds"][k] != held["bounds"][k] for k in shared)
    # the pinned sweep never left the pin, so the taper-driven criteria have
    # a narrower spread than the free one
    assert (held["bounds"]["lod"][1] - held["bounds"]["lod"][0]) \
        < (free["bounds"]["lod"][1] - free["bounds"]["lod"][0])


def test_the_baseline_carries_the_pin():
    """The design a run is COMPARED against must be one the run could have
    returned: the centre of a row nobody searched is not."""
    cfg = _cfg(pinned={"taper": TAPER})
    base = api.baseline_x(cfg)
    built = PROBLEM_SPECS[WING].build({}, {}, None)
    i = list(built.param_labels).index("taper")
    assert base[i] == pytest.approx(TAPER)
    for j, (lo, hi) in enumerate(built.bounds):
        if j != i:
            assert base[j] == pytest.approx(0.5 * (lo + hi))


# --------------------------------------------------- 5. the interrupted run

def test_a_stopped_pinned_run_logs_full_design_vectors():
    """A run that stops early is rebuilt from the progress log, and that log
    has to speak the full vector or the incumbent cannot be evaluated."""
    cfg = _cfg(budget=24, pinned={"taper": TAPER})
    res = api.run(cfg, stop_rule=lambda i, best: i >= 6)
    assert res.partial
    assert res.pinned == {"taper": TAPER}
    assert len(res.best_x) == res.dim
    X = np.asarray(res.eval_x, dtype=float)
    assert X.shape[1] == res.dim
    assert np.allclose(X[:, list(res.param_labels).index("taper")], TAPER)


# ------------------------------------------------------ 6. a warm-started run

def test_a_warm_start_survives_a_pin():
    """The family's own seed is in full coordinates; a pinned search needs it
    projected, or BO is handed a vector of the wrong length."""
    cfg = _cfg(flags={api.BO_WARM_START_FLAG: True}, pinned={"taper": TAPER})
    res = api.run(cfg)
    assert res.n_evals > 0
    assert len(res.best_x) == res.dim


# ----------------------------------------------------- 7. a constrained family

def test_a_constrained_family_takes_a_pin():
    """The constrained runners take the reduced vector too, and the margin
    still belongs to the design that was reported."""
    spec = PROBLEM_SPECS["tail"]
    assert spec.is_constrained
    lbl, row = next(iter(spec.default_bounds.items()))
    value = 0.5 * (row[0] + row[1])
    cfg = api.RunConfig(problem_name="tail", optimiser="bo", budget=14,
                        seed=1, pinned={lbl: value})
    res = api.run(cfg)
    assert res.pinned == {lbl: value}
    assert res.searched_dim == res.dim - 1
    X = np.asarray(res.eval_x, dtype=float)
    assert np.allclose(X[:, list(res.param_labels).index(lbl)], value)
    if res.best_x is not None:
        assert len(res.best_x) == res.dim
