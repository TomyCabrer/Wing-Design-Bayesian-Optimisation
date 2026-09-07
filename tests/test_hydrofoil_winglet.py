"""Hydrofoil + tip device: the nonplanar (imaged-VLM) water problem.

Two claims are worth pinning. First, the new solver reproduces the planar
one when the device is deleted — otherwise "adding a winglet" would be
confounded with "changing solver". Second, cavitation is judged at each
panel's OWN submergence, which is the physical reason the cant sign matters
in water and does not in air.
"""

import numpy as np
import pytest

from aerobo import api
from aerobo.hydrofoil import (
    HydrofoilProblem,
    HydrofoilWingletProblem,
    PENALTY,
    G_FAIL,
    cavitation_margin_panels,
    evaluate_hydrofoil,
    evaluate_hydrofoil_winglet,
    fg_hydrofoil_winglet,
    sigma_cav,
)

PROB = HydrofoilWingletProblem()
MID = PROB.bounds.mean(axis=1)


def _x(depth=0.5, V=12.0, h_frac=0.10, cant=0.0, taper=0.6,
       tw_root=0.0, tw_tip=-2.0, tc=0.12):
    return np.array([taper, tw_root, tw_tip, tc, depth, V, h_frac, cant])


# ---------------- contract ----------------

def test_dimension_bounds_and_signed_cant():
    assert PROB.dim == 8 and PROB.bounds.shape == (8, 2)
    assert PROB.bounds[-1].tolist() == [-90.0, 90.0], "cant must be signed"
    assert PROB.bounds[-2].tolist() == [0.0, 0.15]


def test_evaluates_and_reports_the_device():
    out = evaluate_hydrofoil_winglet(_x(cant=60.0), PROB)
    assert out["feasible"] and out["LoD"] > 0.0
    wl = out["winglet"]
    assert wl["cant_deg"] == 60.0
    assert wl["tip_z_m"] > 0.0                    # canted up
    assert wl["tip_depth_m"] == pytest.approx(0.5 - wl["tip_z_m"])


def test_out_of_bounds_and_shape_are_in_contract_failures():
    assert evaluate_hydrofoil_winglet(np.zeros(6), PROB)["feasible"] is False
    bad = _x(depth=99.0)
    assert evaluate_hydrofoil_winglet(bad, PROB)["feasible"] is False
    assert fg_hydrofoil_winglet(bad, PROB) == (PENALTY, G_FAIL)


def test_fg_reports_true_value_with_a_signed_margin():
    f, g = fg_hydrofoil_winglet(MID, PROB)
    out = evaluate_hydrofoil_winglet(MID, PROB)
    assert f == pytest.approx(out["LoD"]) and g == pytest.approx(out["g"])


# ---------------- agreement with the planar solver ----------------

def test_zero_device_reproduces_the_planar_hydrofoil():
    """No winglet: imaged VLM vs imaged lifting line on the same design.
    They are different discretisations of the same physics, so they agree to
    a few per cent rather than exactly — the point is that the device's
    effect below is far larger than this gap."""
    x6 = np.array([0.6, 0.0, -2.0, 0.12, 0.5, 12.0])
    planar = evaluate_hydrofoil(x6, HydrofoilProblem())
    nonplanar = evaluate_hydrofoil_winglet(np.concatenate([x6, [0.0, 0.0]]),
                                           PROB)
    assert nonplanar["LoD"] == pytest.approx(planar["LoD"], rel=0.05)
    assert nonplanar["CL"] == pytest.approx(planar["CL"], rel=1e-6)
    assert nonplanar["g"] == pytest.approx(planar["g"], rel=0.1)


# ---------------- the physics the variant exists for ----------------

def test_cavitation_is_judged_at_each_panels_own_depth():
    """A panel lifted towards the surface loses static head, so the same
    aerodynamics cavitates sooner. The planar constraint cannot express
    this: it evaluates ONE sigma for the whole foil."""

    class _Pol:
        def cp_min(self, a):
            return np.full(np.shape(a), -2.0)

    z = np.array([0.0, 0.2, -0.2])
    out = cavitation_margin_panels(np.zeros(3), z, depth=0.5, polar=_Pol(),
                                   V=12.0)
    # worst panel is the SHALLOWEST one
    assert out["depth_worst"] == pytest.approx(0.3)
    assert out["g"] == pytest.approx(sigma_cav(0.3, 12.0) - 2.0)
    assert out["sigma_cav_root"] == pytest.approx(sigma_cav(0.5, 12.0))
    assert out["g"] < out["sigma_cav_root"] - 2.0


def test_a_panel_above_the_surface_is_refused():
    class _Pol:
        def cp_min(self, a):
            return np.zeros(np.shape(a))

    with pytest.raises(ValueError, match="above the free surface"):
        cavitation_margin_panels(np.zeros(2), np.array([0.0, 0.6]),
                                 depth=0.5, polar=_Pol(), V=12.0)


def test_canting_down_buys_cavitation_margin_over_canting_up():
    """Same device, same arc length, opposite sign: down sits deeper, so it
    keeps more static head AND (being further from the same-sign image)
    pays less induced drag. This trade does not exist in air."""
    up = evaluate_hydrofoil_winglet(_x(cant=90.0), PROB)
    down = evaluate_hydrofoil_winglet(_x(cant=-90.0), PROB)
    assert up["feasible"] and down["feasible"]
    assert down["g"] > up["g"]
    assert down["CDi_total"] < up["CDi_total"]
    assert down["LoD"] > up["LoD"]


def test_a_device_beats_no_device_at_the_same_operating_point():
    bare = evaluate_hydrofoil_winglet(_x(h_frac=0.0), PROB)
    tipped = evaluate_hydrofoil_winglet(_x(h_frac=0.15, cant=-60.0), PROB)
    assert tipped["CDi_total"] < bare["CDi_total"]
    assert tipped["LoD"] > bare["LoD"]


def test_shallow_water_still_costs_more_than_deep():
    """The free-surface image is the same physics as before: shallower is
    worse, and the tip device does not reverse that."""
    shallow = evaluate_hydrofoil_winglet(_x(depth=0.2, cant=-45.0), PROB)
    deep = evaluate_hydrofoil_winglet(_x(depth=0.9, cant=-45.0), PROB)
    assert deep["CDi_total"] < shallow["CDi_total"]
    assert deep["g"] > shallow["g"]          # more static head, more margin


# ---------------- API + GUI wiring ----------------

def test_registered_as_a_constrained_water_problem():
    spec = api.PROBLEM_SPECS["hydrofoil + winglet"]
    assert spec.medium == "water" and spec.is_constrained
    built = spec.build({}, {}, None)
    assert built.dim == 8 == len(built.param_labels)
    assert built.param_labels[-2:] == ("winglet_h_frac", "winglet_cant_deg")
    # Empty flags in, published problem out — so the registry's box IS this
    # module's, and the registry's callable must return this module's numbers.
    np.testing.assert_allclose(built.bounds, PROB.bounds)
    f, g = built.callable(MID)
    # `np.isfinite(f) and np.isfinite(g)` could not fail: the package's
    # TOTAL-failure pair is finite BY CONTRACT (PENALTY = -100, G_FAIL = -1).
    # A builder that wired the rows in the wrong order or dropped a kwarg
    # scored -100 at every design — a family that finds nothing from the
    # shell — and this was the only assertion on the REGISTRY-built callable.
    assert f > PENALTY and g > G_FAIL, (f, g)
    direct = evaluate_hydrofoil_winglet(MID, PROB)
    assert f == pytest.approx(direct["LoD"]) and g == pytest.approx(direct["g"])
    assert f > 1.0, "a mid-box water foil has a real lift-to-drag ratio"


def test_water_plus_winglets_now_maps_to_the_nonplanar_problem():
    """The builder used to answer "hydrofoil" (and warn that winglets were
    ignored) for every water configuration."""
    import gui.nice_app as v1

    ch = dict(v1.BUILDER_DEFAULTS, medium="water", winglets="free")
    name, notes = v1.derive_problem(ch)
    assert name == "hydrofoil + winglet"
    assert not any("ignored" in n for n in notes)
    assert v1.derive_problem(dict(v1.BUILDER_DEFAULTS,
                                  medium="water"))[0] == "hydrofoil"


def test_water_now_takes_an_elevator_and_still_refuses_the_rest():
    """The stabiliser joined the water family (hydrotail.py); the features
    that still have no water solver keep saying so."""
    import gui.nice_app as v1

    name, notes = v1.derive_problem(
        dict(v1.BUILDER_DEFAULTS, medium="water", tail=True))
    assert name == "hydrofoil + elevator"
    assert any("imaged solve" in n for n in notes)

    name, notes = v1.derive_problem(
        dict(v1.BUILDER_DEFAULTS, medium="water", planform="free"))
    assert name == "hydrofoil"
    # the size modifier is refused here BY NAME: the water solvers carry
    # calibrated geometry (cavitation margins are quoted on it)
    assert any("free span + area ignored" in n for n in notes)

    # a DESIGNED section is a water problem now (hydrofoil_section.py):
    # the sweep carries Cp_min, so the cavitation margin is the candidate's
    name, notes = v1.derive_problem(
        dict(v1.BUILDER_DEFAULTS, medium="water", airfoil="section_only"))
    assert name == "hydrofoil + CST section (XFOIL)"
    assert any("Cp_min" in n for n in notes)
    # ...the pre-optimised section LIBRARY still has none
    name, notes = v1.derive_problem(
        dict(v1.BUILDER_DEFAULTS, medium="water", airfoil="coupled"))
    assert name == "hydrofoil"
    assert any("air solvers only" in n for n in notes)
