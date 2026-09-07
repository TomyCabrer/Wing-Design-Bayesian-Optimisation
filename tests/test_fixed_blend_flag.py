"""A blended tip that costs no design variable.

Three problems in the registry DESIGN the winglet's blend fraction — it is
x[5] of ``objective.BLENDED_MODES``. That is the right question for a study
of the blend itself, and the wrong one for a user who has already decided to
BUILD the tip blended: it spends a dimension of the search on a number the
shop drawing fixes, on top of the two the tip device actually has (its height
and its cant).

So the blend is now also a VALUE: ``objective.Problem.blend_frac_fixed``,
reached through the ``api`` flag ``winglet_blend_frac``. The contract:

* the design vector does NOT change — a blended winglet problem still carries
  ``winglet_h_frac`` and ``winglet_cant_deg`` and nothing else;
* at the same blend fraction the flag reproduces the problem that designs it
  BIT-FOR-BIT, including the span cap and the junction drag;
* one number, one owner: passing the flag to a mode that reads the blend out
  of its vector is refused, not silently ignored;
* absent, every published winglet run is untouched.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aerobo import api, geometry, objective     # noqa: E402

X5 = np.array([0.5, 1.0, -2.0, 0.10, 75.0])
BLEND = 0.5


def _built(name: str, flags: dict | None = None):
    return api.PROBLEM_SPECS[name].build({}, flags or {}, None)


def test_the_flag_costs_no_design_variable():
    built = _built("winglet_capped", {api.WINGLET_BLEND_KEY: BLEND})
    assert built.param_labels == ("taper", "twist_root_deg", "twist_tip_deg",
                                  "winglet_h_frac", "winglet_cant_deg")
    assert built.dim == 5
    assert built.bounds.shape == (5, 2)


def test_the_flag_reproduces_the_problem_that_designs_the_blend():
    """Same geometry, same cap, same junction drag — to the last bit."""
    flagged = _built("winglet_capped", {api.WINGLET_BLEND_KEY: BLEND})
    designed = _built("winglet, blended (span-capped)")

    a = flagged.evaluate(X5)
    b = designed.evaluate(np.append(X5, BLEND))
    assert a["feasible"] and b["feasible"]
    for key in ("LoD", "CD", "CDi", "CDp", "CD_junction", "e", "CL"):
        assert a[key] == pytest.approx(b[key], rel=0, abs=0), key


def test_the_blend_is_capped_on_the_developed_line_not_the_cosine():
    """The raked-tip trap, restated: a blended device leaves the wing plane
    tangentially and so reaches FURTHER outboard than h·cos(cant). Capping it
    on the cosine would hand the blend free projected span."""
    h_frac, cant = float(X5[3]), float(X5[4])
    cosine = 10.0 * (1.0 - h_frac * np.cos(np.deg2rad(cant)))

    sharp = geometry.wing_from_x(X5, mode="winglet_capped")
    blended = geometry.wing_from_x(X5, mode="winglet_capped",
                                   blend_frac=BLEND)
    assert sharp.b == pytest.approx(cosine, rel=0, abs=0)   # bit-for-bit
    assert blended.b < sharp.b


def test_absent_the_flag_nothing_moves():
    """Empty in, empty out — the rule every value flag in this package obeys."""
    plain = _built("winglet_capped")
    assert plain.problem.blend_frac_fixed == 0.0
    assert plain.problem.junction_drag is False
    # the span cap is the published closed form, evaluated exactly
    assert geometry.wing_from_x(X5, mode="winglet_capped").b == pytest.approx(
        10.0 * (1.0 - float(X5[3]) * np.cos(np.deg2rad(float(X5[4])))),
        rel=0, abs=0)


def test_a_blend_charges_the_corner_by_default():
    """Junction drag IS the reason to blend (junction.py): without it a blend
    is only a differently drawn wake. It stays overridable."""
    on = _built("winglet_capped", {api.WINGLET_BLEND_KEY: BLEND})
    assert on.problem.junction_drag is True
    off = _built("winglet_capped", {api.WINGLET_BLEND_KEY: BLEND,
                                    "junction_drag": False})
    assert off.problem.junction_drag is False
    assert off.evaluate(X5)["CD_junction"] == 0.0


def test_one_number_one_owner():
    with pytest.raises(ValueError, match="DESIGNS the winglet"):
        objective.Problem(mode="winglet_capped_blended", blend_frac_fixed=0.4)
    with pytest.raises(ValueError, match="carries none"):
        objective.Problem(mode="trim", blend_frac_fixed=0.4)
    with pytest.raises(ValueError, match="outside the design band"):
        objective.Problem(mode="winglet_capped", blend_frac_fixed=1.4)
    with pytest.raises(ValueError, match="needs a winglet"):
        _built("trim wing", {api.WINGLET_BLEND_KEY: BLEND})


def test_the_flag_is_declared_where_it_works_and_nowhere_else():
    """Declared by asking the BUILDER which objective mode it flies — never
    by matching a problem name, because the same four modes appear under a
    dozen names once the section, chord-law and size twins are generated."""
    declared = [n for n, sp in api.PROBLEM_SPECS.items()
                if api.WINGLET_BLEND_KEY in sp.flags]
    assert "winglet_capped" in declared
    assert "winglet" in declared
    assert "winglet + t/c" in declared
    assert "winglet_capped + free planform + free chord law" in declared
    # the modes that DESIGN the blend must not offer it as a flag as well
    assert "winglet, blended (span-capped)" not in declared
    assert "winglet, blended into the wing (span-capped)" not in declared
    # ...and neither may a problem with no winglet at all
    assert "trim wing" not in declared
    for name in declared:
        assert "winglet_blend_frac" not in api.PROBLEM_SPECS[name].param_labels


def test_the_turn_law_travels_with_it():
    """``blend_shape`` chooses HOW the turn is distributed; it is a value on
    the same footing, so the two compose."""
    arc = _built("winglet_capped", {api.WINGLET_BLEND_KEY: BLEND})
    spiral = _built("winglet_capped", {api.WINGLET_BLEND_KEY: BLEND,
                                       "blend_shape": "spiral"})
    assert spiral.problem.blend_shape == "spiral"
    assert arc.evaluate(X5)["LoD"] != spiral.evaluate(X5)["LoD"]


def test_a_run_can_be_assembled_and_scored_with_the_flag():
    cfg = api.RunConfig(problem_name="winglet_capped",
                        flags={api.WINGLET_BLEND_KEY: BLEND},
                        optimiser="random", budget=6, seed=0)
    res = api.run(cfg)
    assert np.isfinite(res.best_score)
    assert len(res.best_x) == 5
