"""A blended tip device is a SHAPE, available on every surface that has a tip.

It used to be a property of two solvers: the single-wing air winglet family
(four ``objective`` modes) and the imaged water foil. Add an empennage, a
second wing or a stabiliser and the blend disappeared — the shell's tip-device
menu offered three shapes on a wing and two on the same wing with a tail —
because ``wingtail.py``, ``tandemvlm.py`` and ``hydrotail.py`` drew the
surface/device corner as a corner and had nowhere to put a blend fraction.

What this file holds:

* the FLAG is declared wherever a tip device is carried, and nowhere else;
* the blend reaches the PANELISATION (the wake it draws changes), the
  JUNCTION charge (so it is not merely a differently drawn wake) and the
  breakdown;
* an untouched build is bit-for-bit the sharp corner every published run
  flew;
* each SURFACE answers for itself — a blend on the second surface does not
  reshape the wing's device.
"""

from __future__ import annotations

import numpy as np
import pytest

from aerobo import api, geometry


def _mid(bounds):
    return np.array([0.5 * (lo + hi) for lo, hi in bounds], dtype=float)


def _build(name: str, flags: dict | None = None):
    spec = api.PROBLEM_SPECS[name]
    return spec.build({} if spec.uses_mission else None, flags or {}, None)


#: one representative of every family that grew the blend, plus the two that
#: already had it (so the rule is stated once for all of them)
FAMILIES = [
    "winglet",                                   # the single wing (had it)
    "hydrofoil + winglet",                       # the imaged foil (had it)
    "tail + winglet",                            # wing + tail, wing's device
    "tail [designed tail + tip device]",         # ...the TAIL's device
    "tail + winglet [designed tail + tip device]",          # both
    "tandem (nonplanar) + winglets",             # the pair
    "hydrofoil + elevator + winglet",            # foil + elevator
    "hydrofoil + elevator [designed elevator + tip device]",
]

#: ...and the families that design the SECTION beside the device. They are
#: the same rule one level up, and they were the last hole in it: shaping an
#: aerofoil is orthogonal to what shape the tip device's root is, but the
#: live-XFOIL twins drew the corner as a corner, so a blend chosen on the
#: shell's tip-device menu quietly did not happen the moment "reshape the
#: section" was switched on beside it. Held separately because evaluating one
#: is a live XFOIL sweep (``spec.slow``): the FLAG rule is checked on all of
#: them, the numeric rules on the fast families above.
SECTION_FAMILIES = [
    "winglet + airfoil (XFOIL)",                 # free-span device
    "winglet_capped + airfoil (XFOIL)",          # ...and span-capped
    "hydrofoil + winglet + CST section (XFOIL)",
    "hydrofoil + elevator + winglet + CST section (XFOIL)",
    "hydrofoil + elevator [designed elevator + tip device]"
    " + CST section (XFOIL)",
]


@pytest.mark.parametrize("name", FAMILIES)
def test_the_blend_flags_are_declared_where_there_is_a_device(name):
    flags = api.PROBLEM_SPECS[name].flags
    for key in api._FIXED_BLEND_FLAGS:
        assert key in flags, (name, key)


@pytest.mark.parametrize("name", SECTION_FAMILIES)
def test_designing_the_section_does_not_take_the_blend_away(name):
    """A designed section and a blended device are ORTHOGONAL choices.

    The shell asks them in two different places — the tip device's SHAPE, and
    whether the wing search may reshape the section — so a family that takes
    one and drops the other makes the second menu silently undo the first.
    """
    spec = api.PROBLEM_SPECS[name]
    for key in api._FIXED_BLEND_FLAGS:
        assert key in spec.flags, (name, key)
    # ...and the value REACHES the solver, without costing a design variable
    plain = _build(name)
    blended = _build(name, {api.WINGLET_BLEND_KEY: 0.4,
                            "blend_shape": "spiral"})
    assert blended.dim == plain.dim, name
    inner = getattr(blended.problem, "foil", blended.problem)
    assert float(getattr(inner, "blend_frac_fixed",
                         getattr(inner, "blend_frac", 0.0))) == \
        pytest.approx(0.4), name
    assert inner.junction_drag is True           # the blend turns it on
    inner0 = getattr(plain.problem, "foil", plain.problem)
    assert float(getattr(inner0, "blend_frac_fixed",
                         getattr(inner0, "blend_frac", 0.0))) == 0.0, name


def test_the_blend_bites_where_the_section_is_designed_too():
    """The air pair, evaluated: the blend has to MOVE the answer.

    ``winglet_capped`` is the one that shows both halves at once — the
    corner's interference charge appears, and the span cap tightens, because
    a blended device reaches further outboard than h·cos(cant) and is capped
    on the whole developed line rather than on the cosine.
    """
    name = "winglet_capped + airfoil (XFOIL)"
    plain, blended = _build(name), _build(
        name, {api.WINGLET_BLEND_KEY: 0.5, "blend_shape": "spiral"})
    x = _mid(plain.bounds)
    a, b = plain.evaluate(x), blended.evaluate(x)
    assert a["feasible"] and b["feasible"], (a.get("reason"), b.get("reason"))
    assert float(a["CD_junction"]) == 0.0        # sharp corner, no charge
    assert float(b["CD_junction"]) > 0.0
    assert b["wing"].b < a["wing"].b             # capped on the developed line
    assert a["score"] != pytest.approx(b["score"])


@pytest.mark.parametrize("name", [
    "trim wing", "tail", "tail [designed tail]", "tandem",
    "hydrofoil", "hydrofoil + elevator", "airfoil (section)",
])
def test_a_family_with_no_tip_device_declares_no_blend(name):
    """The other half of the rule: a menu that offered "blended" where no
    device exists would be offering a shape of nothing."""
    assert api.WINGLET_BLEND_KEY not in api.PROBLEM_SPECS[name].flags, name


@pytest.mark.parametrize("name", FAMILIES)
def test_an_untouched_build_is_the_sharp_corner_it_always_was(name):
    built = _build(name)
    prob = built.problem
    frac = getattr(prob, "blend_frac_fixed", getattr(prob, "blend_frac", 0.0))
    assert float(frac) == 0.0, name
    assert float(getattr(prob, "wing_blend_frac", 0.0)) == 0.0
    assert getattr(prob, "junction_drag", False) is False
    out = built.evaluate(_mid(built.bounds))
    assert out["feasible"], (name, out.get("reason"))
    assert float(out.get("CD_junction", 0.0)) == 0.0


@pytest.mark.parametrize("name", FAMILIES)
def test_a_stated_blend_changes_the_answer_and_charges_the_corner(name):
    """Two things at once, because either alone would be a lie: the blend
    moves the SOLVE (a different wake) and it is PAID for (junction.py). A
    blend that changed only the drawing would be free lift."""
    plain = _build(name)
    blended = _build(name, {api.WINGLET_BLEND_KEY: 0.5})
    assert blended.param_labels == plain.param_labels     # no new variable

    x = _mid(plain.bounds)
    a, b = plain.evaluate(x), blended.evaluate(x)
    assert a["feasible"] and b["feasible"], (name, a.get("reason"),
                                             b.get("reason"))
    assert a["score"] != pytest.approx(b["score"]), name
    # the junction charge defaults ON with the blend — that add-on IS the
    # reason to blend
    assert blended.problem.junction_drag is True
    assert float(b["CD_junction"]) > 0.0, name


@pytest.mark.parametrize("name", FAMILIES)
def test_the_blend_is_reported_where_the_device_is(name):
    out = _build(name, {api.WINGLET_BLEND_KEY: 0.4}).evaluate(
        _mid(_build(name).bounds))
    # the two shapes a device is reported in: a dict per device (the wing's,
    # and the stabiliser's in water) or the wing+tail family's flat
    # ``tail_winglet_*`` keys. Either way the number has to be THERE — a
    # blend that changed the solve and appeared in no read-out is a shape
    # nobody can check.
    reported = [v.get("blend_frac") for k, v in out.items()
                if k in ("winglet", "tail_winglet") and isinstance(v, dict)]
    if out.get("tail_winglet_blend_frac") is not None:
        reported.append(out["tail_winglet_blend_frac"])
    assert reported, name
    assert any(v == pytest.approx(0.4) for v in reported if v is not None), \
        (name, reported)


def test_the_shape_and_the_wing_side_arc_travel_too():
    """The three VALUES that draw a given blend: which turn law, how much of
    the turn the surface does, and whether the corner is charged."""
    built = _build("tail + winglet",
                   {api.WINGLET_BLEND_KEY: 0.5, "blend_shape": "spiral",
                    "wing_blend_frac": 0.1, "junction_drag": False})
    prob = built.problem
    assert prob.blend_shape == "spiral"
    assert prob.wing_blend_frac == pytest.approx(0.1)
    assert prob.junction_drag is False           # explicitly asked for
    out = built.evaluate(_mid(built.bounds))
    assert out["feasible"]
    assert out["winglet"]["blend_shape"] == "spiral"
    assert out["winglet"]["wing_blend_frac"] == pytest.approx(0.1)
    assert float(out["CD_junction"]) == 0.0      # ...so it is not charged


def test_a_wing_side_arc_moves_the_projection_it_is_supposed_to():
    """The wing-side blend exists because a turn confined to the device has
    at most ``blend_frac * h`` of arc (geometry.span_path). It has to reach
    the SPAN accounting or it is decoration."""
    a = _build("tail + winglet, blended (span-capped)".replace(
        ", blended (span-capped)", ""),
        {api.WINGLET_BLEND_KEY: 0.5})
    b = _build("tail + winglet", {api.WINGLET_BLEND_KEY: 0.5,
                                  "wing_blend_frac": 0.12})
    x = _mid(a.bounds)
    pa, pb = a.evaluate(x)["winglet"], b.evaluate(x)["winglet"]
    assert pb["projection_m"] != pytest.approx(pa["projection_m"])
    assert pb["tip_height_m"] != pytest.approx(pa["tip_height_m"])


# ------------------------------------------- each surface answers for itself
def test_the_second_surface_carries_its_own_blend():
    name = "tail + winglet [designed tail + tip device]"
    both = _build(name, {api.WINGLET_BLEND_KEY: 0.5})
    assert both.problem.tail_blend_frac == pytest.approx(0.5)   # follows it

    split = _build(name, {api.WINGLET_BLEND_KEY: 0.5,
                          api.TAIL_WINGLET_BLEND_KEY: 0.0})
    assert split.problem.blend_frac_fixed == pytest.approx(0.5)
    assert split.problem.tail_blend_frac == pytest.approx(0.0)
    x = _mid(both.bounds)
    assert both.evaluate(x)["score"] != pytest.approx(
        split.evaluate(x)["score"])


def test_a_blend_on_the_second_surface_alone_still_pays_for_its_corner():
    """The failure mode this guards: shaping only the aft device, and the
    fillet arriving free because the junction charge read the WING's flag."""
    name = "tail [designed tail + tip device]"        # no device on the wing
    built = _build(name, {api.TAIL_WINGLET_BLEND_KEY: 0.5})
    assert built.problem.junction_drag is True
    out = built.evaluate(_mid(built.bounds))
    assert out["feasible"]
    assert float(out["CD_junction_tail"]) > 0.0


def test_the_tail_blend_is_refused_where_there_is_no_tail_device():
    from aerobo import wingtail

    with pytest.raises(ValueError, match="tip device ON THE TAIL"):
        wingtail.WingTailProblem(winglet=True, tail_blend_frac_fixed=0.5)


def test_a_blend_beside_a_DESIGNED_blend_is_one_number_with_two_owners():
    from aerobo import wingtail

    with pytest.raises(ValueError, match="same number twice"):
        wingtail.WingTailProblem(winglet=True, blended=True,
                                 blend_frac_fixed=0.5)


def test_a_blend_with_nothing_to_blend_is_refused():
    from aerobo import tandemvlm, wingtail

    with pytest.raises(ValueError, match="carries none"):
        wingtail.WingTailProblem(blend_frac_fixed=0.5)
    with pytest.raises(ValueError, match="carries none"):
        tandemvlm.TandemVLMProblem(blend_frac_fixed=0.5)


@pytest.mark.parametrize("value", [-0.1, 1.4])
def test_the_band_is_the_design_band(value):
    from aerobo import wingtail

    with pytest.raises(ValueError, match="outside the design band"):
        wingtail.WingTailProblem(winglet=True, blend_frac_fixed=value)


# ------------------------------------------------------------- the pair
def test_a_pair_charges_a_corner_per_wing():
    """Two wings, two tip devices, two corners — each on ITS OWN tip chord,
    because a pair may have two spans."""
    built = _build("tandem (nonplanar) + winglets",
                   {api.WINGLET_BLEND_KEY: 0.5})
    out = built.evaluate(_mid(built.bounds))
    assert out["feasible"], out.get("reason")
    wl = out["winglet"]
    assert "junction_front" in wl and "junction_rear" in wl
    assert float(out["CD_junction"]) == pytest.approx(
        wl["junction_front"]["CD_junction"]
        + wl["junction_rear"]["CD_junction"])


def test_the_pairs_clearance_gate_reads_the_height_a_blend_actually_reaches():
    """A blend trades tip height for projected span, so the gate that keeps
    the two devices apart has to measure the blended height. Reading the
    sharp-corner one refused pairs whose devices are in fact clear."""
    from aerobo import tandemvlm as tvm

    # a 10 m pair, 1 m apart: the clearance floor is 0.02 x semi = 0.10 m,
    # and a 0.95 m sharp device leaves 0.05 m — refused. The SAME device
    # blended reaches 0.833 m and leaves 0.167 m, which is clear.
    kw = dict(b=10.0, dz=1.0, h_frac_front=0.19, cant_front=90.0,
              h_frac_rear=0.19, cant_rear=90.0)
    assert geometry.winglet_tip_height(0.95, 90.0, 1.0, "arc", 0.75) \
        < geometry.winglet_tip_height(0.95, 90.0)
    assert tvm._device_clearance(**kw) is not None            # sharp: refused
    assert tvm._device_clearance(**kw, blend_frac=1.0,
                                 wing_blend_frac=0.15) is None
