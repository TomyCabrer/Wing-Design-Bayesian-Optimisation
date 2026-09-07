"""The SECOND SURFACE's tip device is asked what shape it is.

The wing's tip device has been a four-answer question for a long time — none,
a near-vertical fence, a canted winglet, or the same device with its root
corner replaced by a real transition. The second surface's was a yes/no
switch: "its own tip device (height and cant)". Same kind of fitting, smaller
question, and the card above it promised "the same three freedoms the wing
has".

The three answers it now maps onto, each in the layer that owns it:

* WHETHER there is one selects the FAMILY (``tail_design``);
* WHAT SHAPE narrows the cant band (``api.TAIL_WINGLET_TYPE_KEY`` ->
  ``wingtail.tail_cant_bounds_for``): the type's own magnitude, on whichever
  side the direction chose;
* whether it is BLENDED is that surface's OWN value
  (``api.TAIL_WINGLET_BLEND_KEY``), so shaping one device does not reshape
  the other.
"""

from __future__ import annotations

import numpy as np
import pytest

from aerobo import api, wingtail

from gui import nice_app as v1


def _mid(bounds):
    return np.array([0.5 * (lo + hi) for lo, hi in bounds], dtype=float)


# --------------------------------------------------------- the cant band
def test_a_type_narrows_the_band_and_keeps_the_direction_it_was_given():
    up = wingtail.tail_cant_bounds_for("up", "vertical")
    down = wingtail.tail_cant_bounds_for("down", "vertical")
    follow = wingtail.tail_cant_bounds_for("follow", "vertical")
    assert up == (84.0, 90.0)
    assert down == (-90.0, -84.0)       # the exact mirror
    assert follow == (84.0, 90.0)       # a magnitude; the side is derived


def test_canted_is_the_legacy_band_so_asking_for_it_narrows_nothing():
    assert wingtail.tail_cant_bounds_for(None, "canted") is None
    assert wingtail.tail_cant_bounds_for(None, None) is None
    # ...and asked WITH a direction it is that direction's own band
    assert wingtail.tail_cant_bounds_for("up", "canted") == \
        wingtail.TAIL_WINGLET_DIRECTIONS["up"]


def test_a_narrow_type_on_a_signed_band_is_refused_by_name():
    """"Near-vertical, on whichever side" is two intervals with a hole in the
    middle, and one bounds row cannot say that. Refused, pointing at the
    answer that IS one statement: follow the load."""
    with pytest.raises(ValueError, match="follow"):
        wingtail.tail_cant_bounds_for("either", "vertical")


def test_an_unknown_type_says_what_the_choices_are():
    with pytest.raises(ValueError, match="unknown winglet_type"):
        wingtail.tail_cant_bounds_for("up", "spiral_staircase")


# ------------------------------------------------------- through the flags
AIR = "tail [designed tail + tip device]"
WATER = "hydrofoil + elevator [designed elevator + tip device]"


@pytest.mark.parametrize("name", [AIR, WATER])
def test_the_type_and_the_blend_are_declared(name):
    flags = api.PROBLEM_SPECS[name].flags
    for key in api.TAIL_TIP_KEYS:
        assert key in flags, (name, key)


@pytest.mark.parametrize("name", [AIR, WATER])
def test_an_untouched_build_sends_nothing_and_flies_the_published_band(name):
    spec = api.PROBLEM_SPECS[name]
    prob = spec.build({} if spec.uses_mission else None, {}, None).problem
    assert prob.tail_cant_bounds is None        # the family's own
    assert prob.tail_blend_frac_fixed is None
    assert prob.tail_winglet_follow is False


@pytest.mark.parametrize("name", [AIR, WATER])
def test_each_shape_reaches_the_solver(name):
    spec = api.PROBLEM_SPECS[name]
    scores = {}
    for shape, flags in (
            ("canted", {api.TAIL_WINGLET_DIR_KEY: "follow"}),
            ("vertical", {api.TAIL_WINGLET_DIR_KEY: "follow",
                          api.TAIL_WINGLET_TYPE_KEY: "vertical"}),
            ("blended", {api.TAIL_WINGLET_DIR_KEY: "follow",
                         api.TAIL_WINGLET_BLEND_KEY: 0.5})):
        built = spec.build({} if spec.uses_mission else None, flags, None)
        out = built.evaluate(_mid(built.bounds))
        assert out["feasible"], (name, shape, out.get("reason"))
        scores[shape] = out["score"]
        if shape == "vertical":
            assert built.problem.tail_cant_bounds == (84.0, 90.0)
        if shape == "blended":
            assert built.problem.tail_blend_frac == pytest.approx(0.5)
    # three shapes, three different aeroplanes
    assert len(set(round(v, 9) for v in scores.values())) == 3, scores


# ------------------------------------------------------------- the menu
def _choices(medium: str) -> dict:
    ch = dict(v1.BUILDER_START, medium=medium, tail=True,
              tail_design="planform")
    v1.normalise_choices(ch)
    return ch


@pytest.mark.parametrize("medium", ["air", "water"])
def test_the_menu_offers_the_wings_own_four_answers(medium):
    ch = _choices(medium)
    assert list(v1.tail_tip_shapes(ch)) == ["none", "vertical", "canted",
                                            "blended"]
    # ...and never fewer than the wing's, which is the promise: the second
    # surface's device is asked the same question, not a smaller one.
    assert set(v1.tail_tip_shapes(ch)) >= set(v1.winglet_shapes(ch))
    # The WING's menu used to be the shorter one in water, and the asymmetry
    # was the registry's rather than this menu's: the hydrofoil families
    # declared no ``winglet_type``, so a 90 deg fence asked for on the WING
    # was dropped by api.sanitise_flags and the family's own cant band
    # (-90..90) searched instead — report 17.12 recorded that wing-side gap
    # as open. It is closed: the foil's device has a settable cant band
    # (hydrofoil/hydrotail ``winglet_cant_bounds``), the families declare the
    # key, and under water the fence is the DOWN half of the signed band
    # (api.WATER_TIP_DIRECTION — the side with the cavitation margin), so
    # both menus now offer the same four answers in both media.
    from aerobo import api

    pair = ("free", "vertical")
    trial = dict(ch, winglets=pair[0], winglet_type=pair[1])
    name, _ = v1.derive_problem(trial)
    honours = "winglet_type" in api.accepted_flags(name)
    assert honours, name
    assert ("vertical" in v1.winglet_shapes(ch)) is honours
    # ...and the band it narrows to has the medium's own SIGN, which is what
    # made this a solver question and not a menu one
    built = api.PROBLEM_SPECS[name].build(
        {} if not api.PROBLEM_SPECS[name].uses_mission else {},
        {"winglet_type": "vertical"}, None)
    row = built.bounds[list(built.param_labels).index("winglet_cant_deg")]
    assert tuple(float(v) for v in row) == ((84.0, 90.0) if medium == "air"
                                            else (-90.0, -84.0)), name


@pytest.mark.parametrize("medium", ["air", "water"])
@pytest.mark.parametrize("shape", ["none", "vertical", "canted", "blended"])
def test_a_shape_round_trips_through_the_choices(medium, shape):
    ch = _choices(medium)
    v1.set_tail_tip_shape(ch, shape)
    assert v1.tail_tip_shape_key(ch) == shape
    assert (ch["tail_design"] == "planform+tip") == (shape != "none")


def test_the_flags_carry_only_what_was_answered():
    ch = _choices("air")
    v1.set_tail_tip_shape(ch, "canted")
    ch["tail_winglet_dir"] = "follow"
    flags = v1.tail_flags(ch)
    # "canted" IS the published band, so its type does not travel
    assert api.TAIL_WINGLET_TYPE_KEY not in flags
    assert api.TAIL_WINGLET_BLEND_KEY not in flags
    assert flags[api.TAIL_WINGLET_DIR_KEY] == "follow"

    v1.set_tail_tip_shape(ch, "vertical")
    assert v1.tail_flags(ch)[api.TAIL_WINGLET_TYPE_KEY] == "vertical"

    v1.set_tail_tip_shape(ch, "blended")
    assert v1.tail_flags(ch)[api.TAIL_WINGLET_BLEND_KEY] > 0.0

    # ...and taking the device off takes all three with it
    v1.set_tail_tip_shape(ch, "none")
    ch["tail_winglet_dir"] = None
    assert not set(v1.tail_flags(ch)) & set(api.TAIL_TIP_KEYS)


def test_an_unknown_shape_says_what_the_choices_are():
    with pytest.raises(ValueError, match="unknown tip-device shape"):
        v1.set_tail_tip_shape(_choices("air"), "swept_forward")


# ---------------------------------------------------------- through V3
def test_the_stage_drives_the_whole_thing_from_one_control():
    from gui.v3 import config
    from gui.v3.app import assemble

    ctx = assemble("air")
    ctx.act("accept_mission")
    ctx.act("set_second_surface", True)
    S = ctx.S

    ctx.act("set_tail_tip", "blended")
    ch = S["wing"]["choices"]
    assert ch["tail_design"] == "planform+tip"
    flags = config.flags(S)
    assert flags[api.TAIL_WINGLET_BLEND_KEY] > 0.0
    # the DIRECTION is stated by the shell, not by this control
    # (config.TAIL_TIP_DIRECTION) — it points the way the surface is mounted
    assert flags[api.TAIL_WINGLET_DIR_KEY] == "down"
    built = api.PROBLEM_SPECS[S["wing"]["problem"]].build({}, flags, None)
    assert built.problem.tail_blend_frac > 0.0

    # ...and the card offers the shape rather than a yes/no
    ctx.render("wing", "type")
    texts = [(getattr(e, "text", "") or "")
             for e in ctx.views[("wing", "type")].descendants()]
    assert not any("its own tip device (height and " in t and "cant)" in t
                   and "switch" in t for t in texts)
    # the SHAPE's own note is under the control, so the card is describing a
    # blended device rather than a yes/no
    assert any("root corner replaced by a real transition" in t
               for t in texts)
    assert any("this surface's own" in t for t in texts)

    ctx.act("set_tail_tip", "none")
    assert S["wing"]["choices"]["tail_design"] == "planform"
    assert not set(config.flags(S)) & set(api.TAIL_TIP_KEYS)


def test_shaping_one_surfaces_device_does_not_reshape_the_other():
    """The reason the second surface has a blend of its OWN: with one shared
    value, choosing "blended" for the tail silently blended the wing's."""
    name = "tail + winglet [designed tail + tip device]"
    spec = api.PROBLEM_SPECS[name]
    aft_only = spec.build({}, {api.TAIL_WINGLET_BLEND_KEY: 0.5}, None).problem
    assert aft_only.blend_frac_fixed == 0.0          # the WING is untouched
    assert aft_only.tail_blend_frac == pytest.approx(0.5)


# ------------------------------- the box on screen is the box that is searched
@pytest.mark.parametrize("surface,shape,row", [
    ("wing", "canted", "winglet_cant_deg"),
    ("wing", "vertical", "winglet_cant_deg"),
    ("wing", "blended", "winglet_cant_deg"),
    ("aft", "canted", "winglet_cant_t_deg"),
    ("aft", "vertical", "winglet_cant_t_deg"),
    ("aft", "blended", "winglet_cant_t_deg"),
])
def test_the_design_box_quotes_the_cant_band_the_run_searches(surface, shape,
                                                              row):
    """A tip-device SHAPE narrows the cant band by FLAG, and the design box
    is built from the spec's static ``default_bounds``. With a vertical fence
    chosen, the table said 60–90° while the run searched 84–90° — a row
    painted "default" and another one flown."""
    from gui.v3 import config
    from gui.v3.app import assemble

    ctx = assemble("air")
    ctx.act("accept_mission")
    if surface == "aft":
        ctx.act("set_second_surface", True)
        ctx.act("set_tail_tip", shape)
    else:
        ctx.act("set_winglet", shape)
    S = ctx.S

    built = api.PROBLEM_SPECS[S["wing"]["problem"]].build(
        {}, config.flags(S), None)
    labels = list(built.param_labels)
    assert row in labels, (surface, shape, labels)
    lo, hi = built.bounds[labels.index(row)]
    assert config.default_bounds(S)[row] == pytest.approx([float(lo),
                                                           float(hi)])


def test_a_build_failure_leaves_the_box_rather_than_emptying_it():
    """The read-out is guarded: a design box that vanishes because a chosen
    section could not be loaded tells the user less than one row quoting the
    family's published band."""
    from gui.v3 import config
    from gui.v3.app import assemble

    ctx = assemble("air")
    ctx.act("accept_mission")
    ctx.act("set_winglet", "vertical")
    S = ctx.S
    # a flag no builder can honour: the guard has to swallow it. (Not
    # ``winglet_type`` — ``winglet_flags`` rewrites that from the choices,
    # so an injected value never reaches the builder.)
    S["wing"]["flags"]["blend_shape"] = "a_turn_law_that_does_not_exist"
    box = config.default_bounds(S)
    assert box[
        "winglet_cant_deg"] == pytest.approx(
        list(api.PROBLEM_SPECS[S["wing"]["problem"]]
             .default_bounds["winglet_cant_deg"]))
    assert len(box) > 3


def test_the_chord_trend_reaches_the_air_stabiliser_too():
    """The shell says the trend "holds on every designed surface". It did not.

    ``tail.py`` passes the stabiliser its own chord limits because the WING's
    metres are meaningless on a surface an order of magnitude smaller — a
    0.5 m minimum chord meant for a 10 m wing refused every correctly-smaller
    tail. That fix was written as ``chord_limits=None``, which threw out the
    two limits that are NOT lengths: the TREND and ``rate_max_deg`` (a local
    taper ANGLE) are invariant under a uniform rescale and mean exactly the
    same thing on a 0.6 m stabiliser.

    Measured before the fix: ``chord_k1_t = +0.5`` FLEW under
    ``chord_trend="root_largest"`` while the wing's own ``chord_k1 = +0.5``
    was refused — the same shape, allowed on one surface and not the other.
    The water twin (``hydrotail``) and both tandems always enforced it.
    """
    import numpy as np

    from aerobo import api

    name = "tail [designed tail] + free chord law"
    assert name in api.PROBLEM_SPECS

    def _flies(trend, row):
        built = api.PROBLEM_SPECS[name].build({}, {"chord_trend": trend}, None)
        labs = list(built.param_labels)
        x = np.asarray(built.bounds, dtype=float).mean(axis=1)
        x[labs.index(row)] = 0.5           # bends the chord UP outboard
        return bool(built.evaluate(x.tolist()).get("feasible"))

    assert _flies("free", "chord_k1") is True
    assert _flies("free", "chord_k1_t") is True
    assert _flies("root_largest", "chord_k1") is False
    assert _flies("root_largest", "chord_k1_t") is False


def test_the_chord_trend_reaches_the_LATTICE_stabiliser_too():
    """The same statement, on the family a designed tail actually runs in.

    ``tail.py`` (lifting line) was fixed; ``wingtail.py`` — the nonplanar
    wing+tail VLM behind every designed tail with a dihedral, a sweep, a tip
    device or a searched section, 2032 problem names — still built its
    stabiliser with ``chord_limits=None``. So the shell offered the trend,
    the flag was declared and accepted, and the tail flew unconstrained:
    measured before the fix, ``chord_k1_t = +0.5`` FLEW under
    ``chord_trend="root_largest"`` while ``chord_k1 = +0.5`` was refused.
    """
    import numpy as np

    from aerobo import api

    name = "tail [designed tail, free dihedral] + free chord law"
    assert name in api.PROBLEM_SPECS
    built0 = api.PROBLEM_SPECS[name].build({}, {}, None)
    assert type(built0.problem).__module__ == "aerobo.wingtail"

    def _flies(trend, row):
        built = api.PROBLEM_SPECS[name].build({}, {"chord_trend": trend}, None)
        labs = list(built.param_labels)
        x = np.asarray(built.bounds, dtype=float).mean(axis=1)
        x[labs.index(row)] = 0.5           # bends the chord UP outboard
        return bool(built.evaluate(x.tolist()).get("feasible"))

    assert _flies("free", "chord_k1") is True
    assert _flies("free", "chord_k1_t") is True
    assert _flies("root_largest", "chord_k1") is False
    assert _flies("root_largest", "chord_k1_t") is False


def test_the_LATTICE_stabiliser_does_not_inherit_the_wing_s_metres():
    """...and keeping the trend must not undo the defect the ``None`` fixed:
    a minimum chord in metres set for a 10 m wing still may not refuse a
    correctly-smaller stabiliser."""
    import numpy as np

    from aerobo import api

    name = "tail [designed tail, free dihedral] + free chord law"
    built = api.PROBLEM_SPECS[name].build(
        {}, {"chord_min_m": 0.5, "chord_trend": "root_largest"}, None)
    x = np.asarray(built.bounds, dtype=float).mean(axis=1)
    got = built.evaluate(x.tolist())
    assert "tail planform" not in str(got.get("reason") or "")


def test_the_stabiliser_still_does_not_inherit_the_wing_s_metres():
    """The fix `chord_limits=None` was written for a real defect, and keeping
    the trend must not undo it: a minimum chord in metres set for the wing
    still may not refuse a correctly-smaller tail."""
    import numpy as np

    from aerobo import api

    name = "tail [designed tail] + free chord law"
    built = api.PROBLEM_SPECS[name].build(
        {}, {"chord_min_m": 0.5, "chord_trend": "root_largest"}, None)
    x = np.asarray(built.bounds, dtype=float).mean(axis=1)
    got = built.evaluate(x.tolist())
    assert "chord limits" not in str(got.get("reason") or "")
