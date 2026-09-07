"""A symmetric section search comes back with a SECTION.

The fin's search is the wing's machinery on half the box: ``w_lower =
-w_upper``, so ``AirfoilProblem`` carries the upper surface only and the
design vector is ``n_cst`` long instead of ``2 n_cst``. Three readers still
split the vector by hand at ``n_cst`` and every one of them failed on the
short vector:

* ``airfoil_select`` built the coordinates from an EMPTY lower surface —
  ``ValueError: need at least one Bernstein weight (got n_w=0)``, which is
  what the shell reported when the vertical tail's "Optimise the section"
  button was pressed;
* the built problem named ``2 n_cst`` parameters for an ``n_cst``-long
  vector, so ``api._section_weights`` refused the size mismatch and the run
  returned NO section at all — nothing to draw, adopt, export or loft — and
  the design-box table zipped four numbers against eight names;
* ``section_report`` split the ANCHOR the same way, so the baseline overlay
  died on the same empty half.

Asserted end to end, because each of the three is invisible to the other two.
"""
from __future__ import annotations

import numpy as np
import pytest

from aerobo import airfoil, api

RE = 6.96e5          # a fin chord's Reynolds number, not the library point
BUDGET = 6
N_INIT = 4


def _cfg():
    return api.airfoil_run_config(symmetric=True, cl_design=0.0, re=RE,
                                  tc_min=0.10, budget=BUDGET, seed=0)


# ------------------------------------------------------- the problem itself

def test_the_symmetric_problem_is_half_the_vector():
    prob = airfoil.AirfoilProblem(symmetric=True, cl_design=0.0)
    assert prob.dim == prob.n_cst
    assert len(prob.param_labels) == prob.dim
    w_u, w_l = prob.split_weights(prob.w0)
    assert np.allclose(w_l, -w_u)


def test_the_built_problem_names_exactly_its_own_variables():
    """``ProblemSpec.param_labels`` may not lie about the vector's length:
    every consumer that zips the two — the design box, the section reader —
    silently mispairs or refuses when it does."""
    cfg = _cfg()
    built = api.PROBLEM_SPECS[cfg.problem_name].build(
        cfg.mission_kwargs or {}, cfg.flags, cfg.bounds_overrides)
    assert built.dim == len(built.param_labels) == built.bounds.shape[0]
    assert all(str(l).startswith("w_upper_") for l in built.param_labels)


def test_the_section_is_read_back_off_the_short_vector():
    cfg = _cfg()
    built = api.PROBLEM_SPECS[cfg.problem_name].build(
        cfg.mission_kwargs or {}, cfg.flags, cfg.bounds_overrides)
    x = built.bounds.mean(axis=1)
    coords = api.section_coords(cfg, x)
    assert coords, "a symmetric design vector read back as 'no section'"
    assert len(coords) > 32


# --------------------------------------------------------------- the search

@pytest.fixture(scope="module")
def report():
    return api.optimize_airfoil(symmetric=True, cl_design=0.0, re=RE,
                                tc_min=0.10, budget=BUDGET, seed=0,
                                optimiser="bo", n_init=N_INIT,
                                with_section=True)


def test_the_search_runs_at_all(report):
    res = report.get("result") or {}
    assert res.get("best_x") is not None, report.get("error")
    assert len(res["best_x"]) == len(res["param_labels"])


def test_the_run_returns_the_shape_it_found(report):
    design = (report.get("section") or {}).get("design")
    assert design, "the fin's search finished with no section to adopt"
    w_u = np.asarray(design["w_upper"], dtype=float)
    w_l = np.asarray(design["w_lower"], dtype=float)
    assert np.allclose(w_l, -w_u), "a fin's section is not symmetric"
    assert len(design["coords"]) > 32
    assert design["tc"] > 0.0


def test_the_baseline_says_its_camber_was_removed(report):
    """The anchor is symmetrised into the box, so calling it "NACA 2412"
    would name a shape that is not the one drawn."""
    base = (report.get("section") or {}).get("baseline")
    if not base:
        pytest.skip("the search never left the anchor")
    assert "symmetrised" in base["name"]
    assert np.allclose(np.asarray(base["w_lower"], dtype=float),
                       -np.asarray(base["w_upper"], dtype=float))
