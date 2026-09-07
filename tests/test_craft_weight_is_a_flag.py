"""The craft's WEIGHT is a stated value, and every water family honours it.

WHY THIS FILE EXISTS
--------------------
``L_design`` is the lift the foil is trimmed to carry, and until now it was a
dataclass default nobody could reach: 6000 N on ``HydrofoilProblem``,
``HydrofoilWingletProblem`` and ``HydrofoilTailProblem`` alike, with no flag
wired to it and the water builders raising on ``mission_kwargs``. That made
the ``b_m`` / ``S_m2`` resize flags actively misleading rather than merely
incomplete: a user who types a real windfoil planform (b = 0.90 m,
S = 0.090 m2) got it flown at a ~600 kg dinghy's weight, i.e. W/S = 66.7 kPa
and CL_target(12 m/s) = 0.903 — a craft 5.8x overloaded. The size was asked
and the load it carries was not.

``api.WEIGHT_KEY`` ("weight_n") closes that, and this file pins the four
things that can go wrong with a value flag:

* it changes NOTHING when it is absent (the frozen anchors below);
* it changes the trim target by exactly the ratio of the weights, at every
  speed in the box (the physics it is supposed to reach);
* it is DECLARED on every family whose builder honours it, and refused on
  every family that does not — the defect ``api.unhonoured_flags`` exists to
  expose is a flag accepted and silently dropped;
* nonsense is refused with the key named, and nothing else is.

THE TWO FROZEN ANCHORS, and why there are two
---------------------------------------------
This project has twice found a null at a box centre that reversed at a
corner, so a single centre-point anchor is not evidence that a change is
inert. Both were established by MEASUREMENT on the working tree before the
flag existed, by building ``hydrofoil + elevator`` through ``api`` and
evaluating it at a stated point of its own design box:

* CENTRE, x = (lo + hi)/2 = (taper 0.6, twist_root 0 deg, twist_tip -2 deg,
  t/c 0.12, depth 0.575 m, V 12 m/s, S_t 0.04 m2, l_t 1.0 m)
  -> L/D = 24.67276683709319, with SM = -0.2276, x_np = 0.1221,
  mac = 0.1225, l_t = 1.0, x_cg = 0.15, i_t = -0.095 deg.

* OFF-CENTRE, x = lo + 0.75*(hi - lo) = (taper 0.8, twist_root 2 deg,
  twist_tip 0 deg, t/c 0.14, depth 0.7875 m, V 14 m/s, S_t 0.05 m2,
  l_t 1.25 m)
  -> L/D = 22.568044635929535, with SM = +0.0858, x_np = 0.1978,
  x_cg = 0.1875, i_t = +0.801 deg.

The off-centre point is the more discriminating of the two on purpose: it
sits on the far side of the static-margin boundary from the centre
(g_sm = +0.0058 there against -0.3076 at the centre) and the stabiliser is
LIFTING rather than being trimmed down, so a change that quietly moved the
balance would show here even if the centre happened to be blind to it.
"""

import dataclasses

import numpy as np
import pytest

from aerobo import api
from aerobo import hydrotail
from aerobo.hydrofoil import HydrofoilProblem


#: the problem both anchors are measured on. Its box carries the speed and
#: the depth as design variables, which is what lets one build answer the
#: "at three speeds" question without three problems.
ANCHOR = "hydrofoil + elevator"

#: L/D at the CENTRE of ANCHOR's box, and at lo + 0.75*(hi - lo). Measured,
#: not derived — see the module docstring for the design points and the
#: trim state each of them reports.
ANCHOR_CENTRE_LOD = 24.67276683709319
ANCHOR_OFF_CENTRE_LOD = 22.568044635929535

#: a family that has no ``L_design`` at all: its weight comes from a mission
#: card, so ``weight_n`` there would be a second answer to a question already
#: asked. Named rather than derived because the POINT is that this specific
#: family refuses.
UNDECLARED = "trim wing"


def _water_families() -> list:
    """Every registered problem whose medium is water, off the registry.

    Derived, never listed: the water registry is generated (a tip device x an
    arm x a stabiliser depth x how much of the elevator is designed, then
    every chosen-section and chord-law twin of all of those), so a written
    list would be stale the day another combination is registered — and the
    silent-drop defect this file is about is exactly what a stale list would
    stop catching.
    """
    return [n for n, s in api.PROBLEM_SPECS.items() if s.medium == "water"]


def _carrier(prob):
    """The object that owns ``L_design`` for a built water problem.

    The CST-section composites wrap the foil problem
    (``HydrofoilSectionProblem.foil``) rather than subclassing it, so the
    weight lives one level down there and at the top level everywhere else.
    """
    return getattr(prob, "foil", prob)


#: THE PHYSICS THE TWO FROZEN ANCHORS WERE MEASURED ON, stated as flags
#: rather than left to the defaults. Session 68 changed two of the elevator
#: family's: the craft became FLAT (``hydrotail.Z_T_FRAC_DEFAULT`` = 0.01 b,
#: was ``tail.DZ_FRAC`` = 0.05 b) and its strut became a placed, loaded
#: surface instead of a rectangle of wetted area (``strut_model``). Both move
#: every L/D in the family.
#:
#: Re-pinned here rather than re-measured, because what the anchors are FOR
#: is unchanged: this file asks whether an ABSENT ``weight_n`` reproduces the
#: answer the engine gave before the flag existed, and the weight flag is not
#: what moved. The same two numbers are pinned the same way in
#: ``test_rig_couple_is_a_moment_not_a_cg_move.py``.
PUBLISHED_PHYSICS = {"strut_model": False,
                     "z_t_m": -hydrotail.DZ_FRAC * 1.2}


def _box(name: str, flags: dict | None = None):
    built = api.PROBLEM_SPECS[name].build(
        {}, dict(PUBLISHED_PHYSICS if flags is None else flags), None)
    return built, built.bounds[:, 0], built.bounds[:, 1]


# ---------------------------------------------------------------- (a)

def test_absent_flag_keeps_the_published_weight_everywhere():
    """No ``weight_n`` -> every water family is built with its published
    ``L_design``, which is one number across the whole registry."""
    published = dataclasses.fields(HydrofoilProblem)
    published = [f for f in published if f.name == "L_design"][0].default
    assert published == 6000.0        # the ~600 kg dinghy the tier was sized on

    families = _water_families()
    assert families, "no water families registered — the registry moved"
    for name in families:
        prob = api.PROBLEM_SPECS[name].build({}, {}, None).problem
        assert _carrier(prob).L_design == published, name


def test_absent_flag_reproduces_the_centre_anchor_bit_for_bit():
    """The frozen centre anchor, to the last bit."""
    built, lo, hi = _box(ANCHOR)
    x = 0.5 * (lo + hi)
    lod, g = built.callable(x)
    assert lod == ANCHOR_CENTRE_LOD
    assert built.problem.L_design == 6000.0

    # ...and the trim state the anchor was recorded with, so a change that
    # kept L/D while moving the balance cannot pass
    out = built.evaluate(x)
    assert out["SM"] == pytest.approx(-0.2276, abs=5e-5)
    assert out["x_np"] == pytest.approx(0.1221, abs=5e-5)
    assert out["mac"] == pytest.approx(0.1225, abs=5e-5)
    assert out["l_t"] == 1.0
    assert out["x_cg"] == 0.15
    assert out["i_t_deg"] == pytest.approx(-0.095, abs=5e-4)
    assert g[1] == pytest.approx(-0.3076, abs=5e-5)


# ---------------------------------------------------------------- (b)

@pytest.mark.parametrize("V", [9.0, 12.0, 15.0])
def test_the_trim_target_scales_with_the_stated_weight(V):
    """CL_target is the weight over the dynamic pressure and the area.

    The expected value is written out here from first principles —
    ``W / (1/2 rho V^2 S)`` — with W the weight that was ASKED for, rho and S
    the built problem's own water and reference area. Nothing is imported
    from the solver's own expression, so a solver that stopped dividing by
    the area (or started using the stabiliser's) fails this and not just its
    own restatement.
    """
    weight = 1030.0                     # a wingfoil rider + board + wing
    built, lo, hi = _box(ANCHOR)
    i_V = list(built.param_labels).index("V_ms")

    x = 0.5 * (lo + hi)
    x[i_V] = V

    heavy = api.PROBLEM_SPECS[ANCHOR].build({}, {}, None)
    light = api.PROBLEM_SPECS[ANCHOR].build({}, {api.WEIGHT_KEY: weight}, None)

    rho = light.problem.rho
    S = light.problem.S
    expected = weight / (0.5 * rho * V * V * S)

    got = light.evaluate(x)["CL_target"]
    assert got == expected

    # ...and the RATIO, which is the claim the flag makes: the same craft,
    # the same speed, a different weight, and nothing else moves. Exact
    # because both runs divide by an identical denominator.
    assert got / heavy.evaluate(x)["CL_target"] == weight / 6000.0


# ---------------------------------------------------------------- (c)

def test_absent_flag_reproduces_the_off_centre_anchor_bit_for_bit():
    """The SECOND frozen anchor, at lo + 0.75*(hi - lo).

    Established by measurement on the tree before ``weight_n`` existed (see
    the module docstring for the design point and the trim state). It is here
    because a box centre is not a box: this project has twice recorded a null
    at a centre that reversed at a corner, and the centre of THIS box happens
    to trim the stabiliser down through an unstable static margin while this
    point trims it up through a stable one — two different balances, one
    number each.
    """
    built, lo, hi = _box(ANCHOR)
    x = lo + 0.75 * (hi - lo)
    lod, g = built.callable(x)
    assert lod == ANCHOR_OFF_CENTRE_LOD

    out = built.evaluate(x)
    assert out["SM"] == pytest.approx(0.0858, abs=5e-5)
    assert out["x_np"] == pytest.approx(0.1978, abs=5e-5)
    assert out["x_cg"] == 0.1875
    assert out["i_t_deg"] == pytest.approx(0.8010, abs=5e-4)
    assert g[1] == pytest.approx(0.005815, abs=5e-6)
    # the stabiliser LIFTS here and is trimmed DOWN at the centre — the two
    # anchors are on opposite sides of the balance, which is the point
    assert out["i_t_deg"] > 0.0


# ---------------------------------------------------------------- (d)

def test_every_water_family_declares_and_honours_the_weight():
    """Declared AND honoured, family by family.

    Two assertions per family and not one, because they fail apart: a spec
    that lists the key while its builder ignores it accepts the flag and
    drops it (the defect ``unhonoured_flags`` exists to name), and a builder
    that reads a key its spec does not list raises at the Run button on a
    number it would have honoured.
    """
    weight = 1030.0
    families = _water_families()
    assert families

    for name in families:
        assert api.unhonoured_flags(name, {api.WEIGHT_KEY: weight}) == [], name
        api.check_flags(name, {api.WEIGHT_KEY: weight})
        prob = api.PROBLEM_SPECS[name].build(
            {}, {api.WEIGHT_KEY: weight}, None).problem
        assert _carrier(prob).L_design == weight, name


def test_the_weight_is_a_value_and_not_a_design_variable():
    """Stating the weight must not move the search: same rows, same box."""
    plain = api.PROBLEM_SPECS[ANCHOR].build({}, {}, None)
    stated = api.PROBLEM_SPECS[ANCHOR].build(
        {}, {api.WEIGHT_KEY: 1030.0}, None)
    assert stated.param_labels == plain.param_labels
    assert stated.dim == plain.dim
    assert np.array_equal(stated.bounds, plain.bounds)


def test_a_family_without_a_design_lift_still_refuses_the_key():
    """An air family whose weight comes from its mission card must RAISE."""
    assert api.PROBLEM_SPECS[UNDECLARED].medium != "water"
    assert api.unhonoured_flags(UNDECLARED, {api.WEIGHT_KEY: 1030.0}) == [
        api.WEIGHT_KEY]
    with pytest.raises(KeyError) as err:
        api.check_flags(UNDECLARED, {api.WEIGHT_KEY: 1030.0})
    assert api.WEIGHT_KEY in str(err.value)


# ---------------------------------------------------------------- (e)

@pytest.mark.parametrize("bad", [0.0, -5.0])
def test_a_non_positive_weight_is_refused_by_name(bad):
    """Zero and negative are nonsense, and the message says which key."""
    with pytest.raises(ValueError) as err:
        api.PROBLEM_SPECS[ANCHOR].build({}, {api.WEIGHT_KEY: bad}, None)
    msg = str(err.value)
    assert api.WEIGHT_KEY in msg
    assert repr(bad) in msg or f"{bad:g}" in msg


def test_a_heavy_craft_is_not_refused():
    """A calibration is a default, not a ban: there is no upper bound.

    Ten times the published weight is a hard design, not an illegal one — the
    cavitation margin is what refuses it, at the panel where it fails.
    """
    built = api.PROBLEM_SPECS[ANCHOR].build(
        {}, {api.WEIGHT_KEY: 60000.0}, None)
    assert built.problem.L_design == 60000.0


# ---------------------------------------------------------------- (f)

def test_halving_the_weight_halves_the_required_lift_coefficient():
    """The RATIO, at fixed geometry and speed — no second hand-run physics.

    The value is (b)'s business; this is the proportionality itself, and it
    is asserted as a ratio precisely so that it survives any future change to
    the reference area, the water or the speed the anchor is taken at.
    """
    built, lo, hi = _box(ANCHOR)
    x = 0.5 * (lo + hi)

    full = api.PROBLEM_SPECS[ANCHOR].build(
        {}, {api.WEIGHT_KEY: 2060.0}, None).evaluate(x)["CL_target"]
    half = api.PROBLEM_SPECS[ANCHOR].build(
        {}, {api.WEIGHT_KEY: 1030.0}, None).evaluate(x)["CL_target"]

    assert full > 0.0
    assert half / full == 0.5


# ---------------------------------------------------------------- (g)

#: the two PLANAR water kinds — the foil alone and the foil with its tip
#: device. Named rather than derived because the point of this block is that
#: the OUTCOME is checked somewhere other than the elevator family: the
#: registry-wide test above asserts the ATTRIBUTE ``L_design`` on all 156
#: water specs and never once flies them, and an attribute that reaches the
#: constructor but not the trim equation would pass it unchanged.
PLANAR_KINDS = ("hydrofoil", "hydrofoil + winglet")


@pytest.mark.parametrize("name", PLANAR_KINDS)
def test_the_planar_kinds_fly_the_stated_weight_and_not_only_store_it(name):
    """The trim target of a PLANAR water family scales by the weight ratio.

    ``hydrofoil`` and ``hydrofoil + winglet`` reach ``CL_target`` through
    ``hydrofoil.evaluate_*`` and not through ``hydrotail``'s balance, so the
    elevator family's outcome test says nothing about them: they are a
    different evaluator reading the same field.

    The expected value is written out here from first principles —
    ``W / (1/2 rho V^2 S)`` — from the built problem's OWN water and
    reference area, so a family that quietly kept the published 6000 N in
    its lift equation while storing 1030 N on the dataclass fails this and
    not merely its own restatement. The ratio is asserted as well, exactly,
    because a change to rho or S would move both sides of the value check
    together and leave the proportionality the only thing still pinned.
    """
    weight = 1030.0
    plain = api.PROBLEM_SPECS[name].build({}, {}, None)
    stated = api.PROBLEM_SPECS[name].build(
        {}, {api.WEIGHT_KEY: weight}, None)

    lo, hi = stated.bounds[:, 0], stated.bounds[:, 1]
    x = 0.5 * (lo + hi)
    i_V = list(stated.param_labels).index("V_ms")
    V = float(x[i_V])

    expected = weight / (0.5 * float(stated.problem.rho) * V * V
                         * float(stated.problem.S))
    got = stated.evaluate(x)["CL_target"]
    assert got == expected, name

    # ...and the RATIO, which is the whole claim: same craft, same speed,
    # a different weight and nothing else. Exact — one denominator.
    assert got / plain.evaluate(x)["CL_target"] == weight / 6000.0, name


# ---------------------------------------------------------------- (h)

@pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan")])
def test_a_non_finite_weight_is_refused_by_name(bad):
    """Non-finite is refused BEFORE the positivity test, and says "finite".

    ``inf > 0.0`` is True, so the "> 0 N" gate waves an infinite weight
    straight through and builds a problem whose ``CL_target`` is infinite at
    every speed. NaN reaches the same gate from the other side — every
    comparison against it is False, so it was refused, but with a message
    that called it non-positive, which it is not.

    Both of ``_weight_kwargs``'s siblings already refuse non-finite by name
    (``rig.RigLoads.__post_init__``, ``hydrofoil._operating_row``); this is
    the third statement of the same rule. It is a refusal of a number the
    solve cannot mean and NOT a cap — ``test_a_heavy_craft_is_not_refused``
    above is the other half of that sentence.
    """
    with pytest.raises(ValueError) as err:
        api.PROBLEM_SPECS[ANCHOR].build({}, {api.WEIGHT_KEY: bad}, None)
    msg = str(err.value)
    assert api.WEIGHT_KEY in msg
    assert repr(bad) in msg
    assert "finite" in msg
    # the message is the NEW one, not the positivity one it used to borrow
    assert "must be" not in msg and "> 0 N" not in msg


def test_the_infinite_weight_is_what_the_positivity_gate_lets_past():
    """WHY the finite check is a separate refusal, in one arithmetic line.

    The harm is derived here rather than trusted: ``inf > 0.0`` passes, and
    ``W / (1/2 rho V^2 S)`` at that weight — the problem's own water and
    area, the formula (b) already uses — is not a number the trim solve can
    converge to. Without this the test above could be satisfied by a
    refusal that fires for some unrelated reason.
    """
    assert float("inf") > 0.0            # the gate that does NOT catch it
    assert not (float("nan") > 0.0)      # ...and the one that did, mislabelled

    built, _, _ = _box(ANCHOR)
    q_S = 0.5 * float(built.problem.rho) * 12.0 ** 2 * float(built.problem.S)
    assert q_S > 0.0
    assert not np.isfinite(float("inf") / q_S)


# ---------------------------------------------------------------- (i)

def test_no_water_family_swallows_a_mission_card():
    """A stated mission is REFUSED by every water family, not discarded.

    The premise of this whole file (see the module docstring) was that "the
    water builders raise on ``mission_kwargs``". Two of them did not: the
    plain ``hydrofoil`` and ``hydrofoil + winglet`` builders took
    ``mission_kwargs`` and never referenced it, so
    ``build({"W_N": 1030.0}, {}, None)`` returned a problem still trimmed to
    6000 N — the exact silent-drop defect ``unhonoured_flags`` exists to
    prevent, one argument to the left of the flags it polices.

    It became MORE reachable the day ``weight_n`` shipped, not less: a user
    who has just learned that the plain hydrofoil honours a stated weight
    will reasonably reach for the name the air families use for the same
    quantity, and ``W_N`` is that name.

    Asserted over the whole generated registry rather than the two names,
    because a builder added tomorrow inherits the same argument.
    """
    families = _water_families()
    assert len(families) > 100, len(families)
    for name in families:
        assert api.PROBLEM_SPECS[name].mission_fields == (), name
        with pytest.raises(ValueError, match="design variables") as err:
            api.PROBLEM_SPECS[name].build({"W_N": 1030.0}, {}, None)
        assert "no mission spec" in str(err.value), name


@pytest.mark.parametrize("name", PLANAR_KINDS)
def test_the_refused_mission_is_the_only_thing_refused(name):
    """The empty mission still builds, and builds the published problem.

    The other half of the refusal: an empty ``mission_kwargs`` is what every
    caller in the tree actually passes (``api.trim_surface_cl`` sends
    ``mission_kwargs or {}``), and it must stay a build rather than become a
    raise — an over-eager guard here would take the two families off the
    registry entirely.
    """
    built = api.PROBLEM_SPECS[name].build({}, {}, None)
    assert built.problem.L_design == 6000.0
    assert api.PROBLEM_SPECS[name].build(None, {}, None) is not None


# ---------------------------------------------------------------- (j)

#: the WRAPPED kinds, and the third shape of the same defect.
#:
#: ``test_every_water_family_declares_and_honours_the_weight`` checks the
#: ATTRIBUTE on all 156 water specs and flies none of them; (b) and (g) fly
#: three families whose problem object IS the foil. A CST-section composite
#: is neither: ``HydrofoilSectionProblem`` OWNS a foil problem
#: (``.foil``) rather than subclassing one, so ``weight_n`` has to travel one
#: level down through a different builder before anything reads it, and
#: ``_carrier`` above exists precisely because of that indirection. A builder
#: that set ``L_design`` on the WRAPPER would satisfy every attribute check
#: in this file — ``_carrier`` would find it — and fly 6000 N.
#:
#: Two names and not one, because the two wrap DIFFERENT balances: the
#: planar evaluator (``hydrofoil.evaluate_hydrofoil``) and ``hydrotail``'s
#: three-way pitch trim. They are the cheap end of the section registry —
#: the weight never reaches XFOIL, so no section is re-run for this.
COMPOSITE_KINDS = ("hydrofoil + CST section (XFOIL)",
                   "hydrofoil + elevator + CST section (XFOIL)")


@pytest.mark.parametrize("name", COMPOSITE_KINDS)
def test_a_wrapped_composite_flies_the_weight_and_not_only_stores_it(name):
    """The trim target of a SECTION composite scales by the weight ratio.

    The premise is asserted first and is the reason this test is not a
    duplicate of (g): the composite's own problem object has no
    ``L_design`` at all, so the number under test is one an indirection had
    to carry — ``_carrier`` is not a convenience here, it is the mechanism.

    The expected value is written out from first principles again —
    ``W / (1/2 rho V^2 S)`` — from the CARRIER's own water and reference
    area, so a composite that stored 1030 N on its foil and kept flying the
    published 6000 N in the trim equation fails this and not merely its own
    restatement. The ratio is asserted exactly beside it, because a change
    to rho or S moves both sides of the value check together.
    """
    weight = 1030.0
    plain = api.PROBLEM_SPECS[name].build({}, {}, None)
    stated = api.PROBLEM_SPECS[name].build(
        {}, {api.WEIGHT_KEY: weight}, None)

    # the indirection is real: the wrapper does not own the field
    assert not hasattr(stated.problem, "L_design"), name
    carrier = _carrier(stated.problem)
    assert carrier is not stated.problem, name
    assert carrier.L_design == weight, name

    b = np.asarray(stated.bounds, dtype=float)
    x = 0.5 * (b[:, 0] + b[:, 1])
    V = float(x[list(stated.param_labels).index("V_ms")])

    expected = weight / (0.5 * float(carrier.rho) * V * V * float(carrier.S))
    got = stated.evaluate(x)
    assert got["reason"] == "", (name, got["reason"])
    assert got["CL_target"] == expected, name

    # ...and the RATIO: same craft, same speed, a different weight, exact
    # because both runs divide by an identical denominator
    assert got["CL_target"] / plain.evaluate(x)["CL_target"] \
        == weight / 6000.0, name


def test_a_weight_that_is_not_a_number_is_refused_by_name():
    """The likeliest of the three bad weights, and the one that used to get
    the worst message.

    Before this refusal existed the bare ``float(raw)`` raised
    ``could not convert string to float: 'heavy'`` — a sentence that names
    neither the flag nor what a weight is, from inside a helper whose
    docstring spends a paragraph promising that a bad weight is refused with
    the key named. The two crafted refusals beside it (non-finite, and
    non-positive) were unreachable for this input.
    """
    for bad in ("heavy", "1030 N", None.__class__, object()):
        with pytest.raises(ValueError) as exc:
            api.PROBLEM_SPECS[ANCHOR].build({}, {api.WEIGHT_KEY: bad}, None)
        msg = str(exc.value)
        assert api.WEIGHT_KEY in msg, msg
        assert "NEWTONS" in msg or "newton" in msg.lower(), msg
        assert "could not convert" not in msg, \
            "the raw float() error escaped the crafted refusal"
