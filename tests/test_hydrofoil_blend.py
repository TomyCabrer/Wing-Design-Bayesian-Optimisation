"""The foil's tip device can be blended.

The water tip device has run the nonplanar imaged VLM since Tier C — the same
solver the air winglet blends in — but it never passed the blend through, so
the one thing a foil/device corner is most exposed to (a sharp corner in a
fluid 800x denser than air) could not even be drawn.

The contract is the air family's, restated in the water problem's own field
names (``blend_frac`` / ``blend_shape`` / ``wing_blend_frac`` /
``junction_drag``, reached through the SAME ``api`` flags):

* it is a VALUE — the design vector still carries the device's height and its
  cant and nothing else;
* absent, every published water run is bit-for-bit unchanged;
* the corner's interference drag comes on with the blend, because without it
  a blend is only a differently drawn wake;
* the cant stays SIGNED here (+ up, − down), and the blend has to work at
  both signs: a device canted up sits in less static head and cavitates
  first, which is the trade this family exists to price;
* cavitation is still judged at EVERY panel's own submergence, blended
  panels included.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aerobo import api, geometry, hydrofoil     # noqa: E402

#: [taper, twist_root, twist_tip, tc, depth, V, h_frac, cant]
X = np.array([0.6, 1.0, -1.0, 0.12, 0.4, 12.0, 0.10, 60.0])
BLEND = 0.5


def _built(flags: dict | None = None, name: str = "hydrofoil + winglet"):
    return api.PROBLEM_SPECS[name].build({}, flags or {}, None)


def test_the_blend_costs_no_design_variable():
    built = _built({api.WINGLET_BLEND_KEY: BLEND})
    assert built.param_labels == ("taper", "twist_root_deg", "twist_tip_deg",
                                  "tc", "depth_m", "V_ms",
                                  "winglet_h_frac", "winglet_cant_deg")
    assert built.dim == 8


def test_absent_the_flag_the_published_foil_is_untouched():
    plain = _built()
    assert plain.problem.blend_frac == 0.0
    assert plain.problem.junction_drag is False
    out = plain.evaluate(X)
    assert out["CD_junction"] == 0.0
    assert out["winglet"]["blend_frac"] == 0.0
    assert "junction" not in out["winglet"]
    # ...and the numbers are the sharp-corner ones the solver always gave
    direct = hydrofoil.evaluate_hydrofoil_winglet(
        X, hydrofoil.HydrofoilWingletProblem())
    assert out["LoD"] == pytest.approx(direct["LoD"], rel=0, abs=0)


def test_the_blend_changes_the_geometry_and_charges_the_corner():
    sharp = _built().evaluate(X)
    blended = _built({api.WINGLET_BLEND_KEY: BLEND}).evaluate(X)
    assert blended["feasible"]
    assert blended["CD_junction"] > 0.0
    # the wake trace itself moved: the device leaves the foil plane
    # tangentially instead of at a corner
    assert blended["CDi"] != sharp["CDi"]
    jr = blended["winglet"]["junction"]
    assert 0.0 < jr["CD_junction"] < jr["CD_junction_sharp"]
    assert jr["blend_radius_m"] > 0.0


def test_the_junction_charge_is_the_reason_to_blend_and_stays_overridable():
    on = _built({api.WINGLET_BLEND_KEY: BLEND})
    assert on.problem.junction_drag is True
    off = _built({api.WINGLET_BLEND_KEY: BLEND, "junction_drag": False})
    assert off.problem.junction_drag is False
    assert off.evaluate(X)["CD_junction"] == 0.0


def test_the_blend_works_at_both_signs_of_cant():
    """Signed cant is the water family's whole point: up buys nonplanar span
    for less cavitation margin, down the reverse."""
    built = _built({api.WINGLET_BLEND_KEY: BLEND})
    up, down = X.copy(), X.copy()
    up[7], down[7] = +60.0, -60.0
    a, b = built.evaluate(up), built.evaluate(down)
    assert a["feasible"] and b["feasible"]
    # canted UP sits in less static head, so it is the tighter margin
    assert a["g"] < b["g"]
    # ``tip_z_m`` is the highest panel, which is the foil itself when the
    # device points DOWN — so the down-canted case is read off the solve
    assert a["winglet"]["tip_z_m"] > 0.0
    assert float(b["vlm"].z.min()) < 0.0


def test_cavitation_is_still_judged_at_every_panel():
    built = _built({api.WINGLET_BLEND_KEY: BLEND})
    out = built.evaluate(X)
    res = out["vlm"]
    # the blended panels are IN the solve, and the margin is the worst of all
    assert res.is_winglet.sum() > 0
    cav = hydrofoil.cavitation_margin_panels(
        np.rad2deg(res.alpha_eff), res.z, float(X[4]),
        built.problem.polar_family.at(float(X[3])), V=float(X[5]))
    assert out["g"] == pytest.approx(cav["g"], rel=0, abs=0)
    assert out["g"] == pytest.approx(float(np.min(cav["g_y"])), rel=1e-12)


def test_the_turn_law_and_the_foil_side_arc_travel_too():
    arc = _built({api.WINGLET_BLEND_KEY: BLEND})
    spiral = _built({api.WINGLET_BLEND_KEY: BLEND, "blend_shape": "spiral"})
    assert spiral.problem.blend_shape == "spiral"
    assert arc.evaluate(X)["LoD"] != spiral.evaluate(X)["LoD"]

    wing_side = _built({api.WINGLET_BLEND_KEY: BLEND,
                        "wing_blend_frac": 0.05})
    assert wing_side.problem.wing_blend_frac == 0.05
    jr = wing_side.evaluate(X)["winglet"]["junction"]
    # the turn that starts on the FOIL has more arc, so a bigger radius
    assert jr["blend_radius_m"] > arc.evaluate(X)["winglet"]["junction"][
        "blend_radius_m"]


def test_an_impossible_blend_is_refused_at_construction():
    with pytest.raises(ValueError, match="outside the design band"):
        hydrofoil.HydrofoilWingletProblem(blend_frac=1.5)
    with pytest.raises(ValueError, match="unknown blend_shape"):
        hydrofoil.HydrofoilWingletProblem(blend_shape="ogive")
    with pytest.raises(ValueError, match="outside the design band"):
        hydrofoil.HydrofoilWingletProblem(wing_blend_frac=0.9)
    assert geometry.WINGLET_BLEND_BOUNDS == (0.0, 1.0)


def test_the_flag_is_declared_on_every_twin_of_the_water_family():
    declared = {n for n, sp in api.PROBLEM_SPECS.items()
                if api.WINGLET_BLEND_KEY in sp.flags and sp.medium == "water"}
    assert "hydrofoil + winglet" in declared
    assert "hydrofoil + winglet + free chord law" in declared
    assert "hydrofoil + winglet [chosen section]" in declared
    assert "hydrofoil + winglet + CST section (XFOIL)" in declared
    # the PLANAR foil has no tip device, so it has no corner to blend
    assert "hydrofoil" not in declared


def test_v3_offers_the_blended_shape_under_water():
    from gui import nice_app as v1

    ch = v1.start_choices(medium="water")
    assert "blended" in v1.winglet_shapes(ch)
    v1.set_winglet_shape(ch, "blended")
    v1.normalise_choices(ch, keep="winglets")
    name, _notes = v1.derive_problem(ch)
    assert name.startswith("hydrofoil + winglet")
    assert v1.winglet_flags(ch)[api.WINGLET_BLEND_KEY] == v1.BLEND_FRAC_DEFAULT
