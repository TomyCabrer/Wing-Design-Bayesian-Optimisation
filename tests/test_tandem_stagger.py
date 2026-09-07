"""The tandem pair's STAGGER belongs to the user, and its two winglets are two.

Two defects, one family, and the second is a consequence of the first being
loose:

1. THE STAGGER WAS DERIVED, NOT STATED. dx and dz were rewritten as b/2 and
   0.1 b wherever a span appeared — including from a span the OPTIMISER chose
   under the size modifier. So the one geometric quantity these families exist
   to study (how far apart the two surfaces are, which is what sets the mutual
   induction) was a quantity the run picked for itself, and no one could type
   the layout they were actually building.

2. THE TWO TIP DEVICES WERE DRAWN AS ONE. The nonplanar pair returns ONE panel
   array — front wing, its device, rear wing, its device — and the geometry
   export shipped it whole. Read as a single spanwise polyline the front
   wing's STARBOARD device and the rear wing's PORT device are adjacent and
   both flagged as device panels, so every view and the CAD loft drew a single
   winglet running across the aircraft, and the rear wing appeared as extra
   span on the front one.

Plus the physics that (1) makes checkable: with the stagger stated, a pair
whose devices actually meet is a JOINED wing, and the solver says so instead
of answering for a different aircraft.
"""

import numpy as np
import pytest

from aerobo import api, cad, geometry
from aerobo import tandem as td
from aerobo import tandemvlm as tvm
from aerobo.vlm import MIN_PLANE_CLEARANCE_FRAC

PAIR = "tandem (nonplanar) + winglets"

X_VLM = np.array([0.6, 0.6, 0.0, -2.0, 0.0, -2.0, 0.5, 0.0,
                  0.12, 90.0, 0.12, 90.0])


def _built(name, **flags):
    return api.PROBLEM_SPECS[name].build({}, dict(flags), None)


# --------------------------------------------- 1. the stagger is the user's

@pytest.mark.parametrize("name", ["tandem", PAIR])
def test_the_stagger_is_configuration_and_travels_verbatim(name):
    spec = api.PROBLEM_SPECS[name]
    assert set(api.TANDEM_STAGGER_KEYS) <= set(spec.flags)
    # ...and it is NOT a design variable: no vector row is named for it
    assert not any(str(lbl).startswith("d") and lbl.endswith("_m")
                   for lbl in spec.param_labels)
    prob = _built(name, dx_m=6.5, dz_m=1.75).problem
    assert (prob.dx, prob.dz) == (6.5, 1.75)


@pytest.mark.parametrize("name", ["tandem", PAIR])
def test_either_end_of_the_stagger_alone_is_a_complete_sentence(name):
    """"The wings are 1.2 m apart vertically" says nothing about dx, and the
    other end falls back to the published fraction of the span in play."""
    fx, fz = api._TANDEM_STAGGER_FRACS
    prob = _built(name, dz_m=1.2).problem
    assert (prob.dx, prob.dz) == (fx * 10.0, 1.2)
    prob = _built(name, b_m=14.0, S_m2=28.0, dx_m=4.0).problem
    assert (prob.dx, prob.dz) == (4.0, fz * 14.0)


@pytest.mark.parametrize("name", ["tandem", PAIR])
def test_an_untouched_pair_still_flies_the_published_layout(name):
    prob = _built(name).problem
    assert (prob.dx, prob.dz) == (5.0, 1.0)
    # a span the USER types still carries the layout with it — that derivation
    # is a build-time convenience, and it is the only one left
    scaled = _built(name, b_m=20.0, S_m2=40.0).problem
    assert (scaled.dx, scaled.dz) == (10.0, 2.0)


def test_a_pair_with_no_stagger_at_all_is_refused_at_build_time():
    with pytest.raises(ValueError, match="needs a stagger"):
        _built("tandem", dx_m=0.0, dz_m=0.0)


@pytest.mark.parametrize("family,evaluate", [
    ("tandem + free planform", td.evaluate_tandem),
    ("tandem (nonplanar) + winglets + free planform",
     tvm.evaluate_tandem_vlm),
])
def test_the_size_modifier_does_not_move_the_stated_stagger(family, evaluate):
    """The size modifier puts the SPAN in the design vector. That is a wing
    question; where the two wings sit relative to each other is a layout the
    user stated, and no candidate span may rewrite it."""
    if family not in api.PROBLEM_SPECS:
        pytest.skip(f"{family} is not registered")
    built = _built(family, dx_m=6.0, dz_m=2.0)
    lab = list(built.param_labels)
    assert "b_m" in lab
    x = 0.5 * (built.bounds[:, 0] + built.bounds[:, 1])
    seen = []
    for span in (10.0, 16.0):          # both inside the AR band at this area
        xx = x.copy()
        xx[lab.index("b_m")] = span
        out = evaluate(xx, built.problem)
        assert out["feasible"], out["reason"]
        # the layout the candidate FLEW — a span-derived stagger would read
        # 5.0/1.0 at one span and 8.0/1.6 at the other
        assert (out["dx"], out["dz"]) == (6.0, 2.0)
        assert out["b"] == pytest.approx(span)
        seen.append(span)
    assert len(seen) == 2
    # ...and the problem the run was built from still says the same thing
    assert (built.problem.dx, built.problem.dz) == (6.0, 2.0)


def test_the_gui_states_the_stagger_and_sends_it_only_where_it_means_something():
    from gui import nice_app as v1

    air = {"system": "tandem", "medium": "air"}
    assert v1.tandem_flags(air) == {}                  # untouched sends nothing
    assert v1.tandem_stagger(air) == (5.0, 1.0)        # ...and says what it is
    stated = {**air, "tandem_dx_m": 6.0, "tandem_dz_m": 2.0}
    assert v1.tandem_flags(stated) == {"dx_m": 6.0, "dz_m": 2.0}
    assert v1.tandem_stagger(stated) == (6.0, 2.0)
    assert v1.tandem_stagger({**air, "span_m": 20.0}) == (10.0, 2.0)
    # a single wing, or a medium whose family has no second surface to
    # stagger, sends nothing at all
    assert v1.tandem_flags({**stated, "system": "single"}) == {}
    assert v1.tandem_flags({**stated, "medium": "track"}) == {}


# ------------------------------------------- 2. two devices, or say otherwise

def test_devices_that_meet_are_a_joined_wing_and_the_solver_says_so():
    prob = tvm.TandemVLMProblem(winglets=True, dz=0.4)   # h = 0.6 m each
    out = tvm.evaluate_tandem_vlm(X_VLM, prob)
    assert out["feasible"] is False
    assert "joined" in out["reason"].lower()
    assert out["score"] == tvm.PENALTY


def test_the_published_box_is_clear_of_that_gate_at_every_corner():
    """The gate must not bite anywhere in the offered design box, or it would
    be a bound in disguise. Height and cant are the two rows that reach."""
    prob = tvm.TandemVLMProblem(winglets=True)
    h_hi = tvm.WINGLET_H_FRAC_BOUNDS[1]
    for cant in tvm.WINGLET_CANT_BOUNDS_DEG:
        assert tvm._device_clearance(prob.b, prob.dz, h_hi, cant, h_hi,
                                     cant) is None
    tip = geometry.winglet_tip_height(h_hi * prob.b / 2.0, 90.0)
    assert prob.dz - tip >= MIN_PLANE_CLEARANCE_FRAC * prob.b / 2.0


def test_a_planar_pair_is_never_checked_for_a_clash_it_cannot_have():
    """Two wings with no tip device are separated by the STAGGER, which is the
    layout this family studies — including staggers whose vertical gap is
    small on purpose (the mutual-induction limit)."""
    prob = tvm.TandemVLMProblem(winglets=False, dz=0.05)
    x = np.array([0.6, 0.6, 0.0, -2.0, 0.0, -2.0, 0.5, 0.0])
    assert tvm.evaluate_tandem_vlm(x, prob)["feasible"]
    assert tvm._device_clearance(10.0, 0.05, 0.0, 90.0, 0.0, 90.0) is None


def test_a_dropped_device_cannot_clash_with_anything():
    """Below vlm.MIN_WINGLET_FRAC the panel builder drops the device, so the
    clearance rule has to read the same rule rather than a second one."""
    assert tvm._device_clearance(10.0, 0.2, 0.005, 90.0, 0.005, 90.0) is None


# ------------------------------- 2b. the export, the views and the CAD loft

def _report(name=PAIR, x=X_VLM, **flags):
    cfg = api.RunConfig(problem_name=name, optimiser="random", budget=2,
                        seed=0, flags=flags or None)
    return api.design_report(cfg, x)


def test_the_rear_wing_leaves_as_its_own_surface():
    rep = _report()
    geom = rep["geometry"]
    front_n = len(geom["y"])
    rear = geom["second_surface"]
    assert rear["name"] == "rear"
    # the front wing's arrays are the FRONT wing's — not both wings' concatenated
    assert front_n == len(rear["y"])
    for key in ("chord", "z", "Cl_y", "alpha_eff_deg", "is_winglet"):
        assert len(geom[key]) == front_n
        assert len(rear[key]) == front_n
    # each surface carries its OWN device, at its own height
    assert sum(geom["is_winglet"]) == sum(rear["is_winglet"]) > 0
    assert max(geom["z"]) < min(rear["z"])
    assert rear["x_offset"] == pytest.approx(5.0)
    assert rear["z_offset"] == pytest.approx(1.0)


def test_the_two_devices_are_two_traces_in_every_view():
    from gui import nice_app as v1

    rep = _report()
    geom, lab = rep["geometry"], rep["param_labels"]

    front = v1.fig_frontview(geom)
    names = [str(getattr(t, "name", "")) for t in front.data]
    assert any("rear" in n for n in names), names
    # THE BUG: one device trace spanning tip to tip. Every device trace must
    # stay on ONE side of the aircraft — a winglet is a tip device.
    for tr in front.data:
        if "tip device" not in str(getattr(tr, "name", "")) \
                and "winglet" not in str(getattr(tr, "name", "")):
            continue
        y = np.asarray(tr.x, dtype=float)
        assert np.all(y >= -1e-9) or np.all(y <= 1e-9), \
            f"{tr.name} runs from {y.min():.2f} to {y.max():.2f} m"

    plan = v1.fig_planform(geom, X_VLM, lab)
    assert any("rear" in str(getattr(t, "name", "")) for t in plan.data)
    span = v1.fig_spanwise(geom)
    assert any("rear" in str(getattr(t, "name", "")) for t in span.data)
    three_d = v1.fig_wing3d(geom, X_VLM, lab)
    assert three_d is not None


def test_the_cad_export_builds_two_wings_at_the_stagger():
    rep = _report()
    surfs = cad.surfaces(rep["geometry"], X_VLM, rep["param_labels"])
    by_name = {s.name: s for s in surfs}
    # THREE surfaces, because a tandem has a fin. It always did — the
    # report has stated one since V5 and the flight rebuild flew it — but
    # nothing charged it and nothing exported it, so a pair came out of the
    # CAD as two wings and no vertical. It is a lofted surface here now, at
    # the stagger the fin is sized against.
    assert set(by_name) == {"wing", "rear", "fin"}
    fin = by_name["fin"].points()
    assert fin[..., 0].min() > by_name["wing"].points()[..., 0].max(), \
        "the fin is sized on the stagger, so it stands aft of the front wing"
    front, rear = by_name["wing"].points(), by_name["rear"].points()
    # two surfaces, one behind the other and one above the other — not one
    # surface with 112 stations running through both
    assert rear[..., 0].min() > front[..., 0].max()
    assert rear[..., 2].min() > front[..., 2].max()
    assert by_name["wing"].X.shape == by_name["rear"].X.shape


def test_a_stated_stagger_is_what_the_picture_shows():
    rep = _report(dx_m=7.0, dz_m=2.5)
    rear = rep["geometry"]["second_surface"]
    assert rear["x_offset"] == pytest.approx(7.0)
    assert rear["z_offset"] == pytest.approx(2.5)


# --------------------------------- 3. V3 asks for it where the pair is stated

def _config_labels(view):
    """The Configuration group box's label texts, in the order drawn.

    ``widgets.group_box`` is a div carrying the title label plus a body
    column, so the box is the outermost element whose own descendants start
    with that title."""
    boxes = [e for e in view.descendants()
             if "grouping" in getattr(e, "classes", [])
             and any(getattr(d, "text", "") == "Configuration"
                     for d in e.descendants())]
    box = boxes[0] if boxes else view
    return [t for t in (getattr(e, "text", "") or ""
                        for e in box.descendants()) if t]


def test_v3_asks_for_the_separation_under_the_row_that_declares_the_pair():
    """Stage 3's Configuration list is where the aircraft is configured, and
    the separation IS the pair's configuration — so it is asked directly under
    "lifting system", not at the foot of the list below the tail and the chord
    law. Same rule the tail's own geometry follows: the question sits with the
    thing it is about."""
    from gui.v3.app import assemble

    ctx = assemble()
    ctx.act("accept_mission")
    ctx.act("set_choice", "system", "tandem")
    ctx.render("wing", "type")

    texts = _config_labels(ctx.views[("wing", "type")])
    for label in ("lifting system", "tandem pair (front + rear)",
                  "Rear wing, aft by", "Rear wing, above by"):
        assert label in texts, label
    # under the pair's own row, and above everything that is not the pair
    i_sys = texts.index("tandem pair (front + rear)")
    i_dx = texts.index("Rear wing, aft by")
    i_dz = texts.index("Rear wing, above by")
    assert i_sys < i_dx < i_dz
    for later in ("airfoil", "tip device", "planform", "chord law"):
        assert i_dz < texts.index(later), later


def test_a_typed_number_does_not_rebuild_the_field_it_was_typed_into():
    """THE BUG the user hit: the two stagger fields could not be typed into.

    Every keystroke posted a choice, and writing a choice rebuilt the whole
    configuration list — including the input the digit had just gone into, so
    the field lost the focus and the rest of the number went nowhere. The rule
    is the one the tail's numbers already followed: a choice that carries a
    NUMBER (no menu depends on its value) leaves its own view standing.
    """
    from gui.v3.app import assemble

    ctx = assemble()
    ctx.act("accept_mission")
    ctx.act("set_choice", "system", "tandem")
    ctx.render("wing", "type")

    view = ctx.views[("wing", "type")]
    fields = [id(e) for e in view.descendants()
              if type(e).__name__ == "Number"]
    assert fields, "no number field in the configuration list"

    # "7", then "7.5" — the way a person types it
    ctx.act("set_choice", "tandem_dx_m", 7.0)
    ctx.act("set_choice", "tandem_dx_m", 7.5)
    ctx.act("set_choice", "tandem_dz_m", 2.25)
    assert [id(e) for e in view.descendants()
            if type(e).__name__ == "Number"] == fields
    ch = ctx.S["wing"]["choices"]
    assert (ch["tandem_dx_m"], ch["tandem_dz_m"]) == (7.5, 2.25)

    # ...while a choice that DOES select a family still rebuilds it, or the
    # menus would go stale — which is the reason the rebuild exists at all
    ctx.act("set_choice", "chord", "fixed")
    assert [id(e) for e in view.descendants()
            if type(e).__name__ == "Number"] != fields


def test_every_typed_number_is_declared_as_one():
    """What makes skipping that rebuild SAFE is that no menu reads the value:
    a number key must never select a problem. Checked against the registry
    rather than asserted in a comment."""
    import gui.nice_app as v1
    from gui.v3.stages import wing as wing_stage

    # ...and the two shells agree about WHICH keys are numbers — the V3 list
    # is the one that grew late (the car's force budget and downforce floor
    # were added to V1's and not to it, and every field of the track card's
    # size block then rebuilt itself under the cursor).
    #
    # Not an equality, and the difference is NAMED rather than tolerated: a
    # key V3 draws no field for cannot have its field destroyed, so it has
    # nothing to be on that list for. `car_track_points` is the one such key
    # — the circuit left V3's card in 75ac6dc and took the sampling count
    # with it. That it really is unasked there is gated by
    # test_car_lap_control.py::test_the_sampling_count_is_no_longer_a_card_question,
    # which drives the rendered card; this list only has to stay a list of
    # ONE, so a key added to V1 tomorrow fails here instead of hiding.
    asked_only_by_v1 = {"car_track_points"}
    missing = set(v1.NUMBER_CHOICE_KEYS) - set(wing_stage.VALUE_KEYS)
    assert missing == asked_only_by_v1, missing
    for ch in ({"system": "tandem", "medium": "air"},
               {"system": "single", "medium": "track"}):
        base = dict(v1.BUILDER_DEFAULTS, **ch)
        v1.normalise_choices(base)
        before = v1.derive_problem(base)[0]
        for key in v1.NUMBER_CHOICE_KEYS:
            assert v1.derive_problem(dict(base, **{key: 1.7}))[0] == before, \
                f"{key} moved the problem"


def test_v3_does_not_ask_a_single_wing_where_its_second_wing_is():
    from gui.v3.app import assemble

    ctx = assemble()
    ctx.act("accept_mission")
    ctx.render("wing", "type")
    texts = _config_labels(ctx.views[("wing", "type")])
    assert "one lifting surface" in texts
    assert not [t for t in texts if t.startswith("Rear wing,")]


def test_v3_sends_the_typed_separation_to_the_run():
    """Typing in that card must reach the problem the stage builds — the
    whole point of asking is that the pair flies the layout stated."""
    from gui.v3 import config
    from gui.v3.app import assemble

    ctx = assemble()
    ctx.act("accept_mission")
    ctx.act("set_choice", "system", "tandem")
    ctx.act("set_choice", "tandem_dx_m", 7.5)
    ctx.act("set_choice", "tandem_dz_m", 2.25)

    flags = config.flags(ctx.S)
    assert flags.get("dx_m") == pytest.approx(7.5)
    assert flags.get("dz_m") == pytest.approx(2.25)

    prob = _built(ctx.S["wing"]["problem"],
                  **{k: v for k, v in flags.items()
                     if k in api.TANDEM_STAGGER_KEYS}).problem
    assert (prob.dx, prob.dz) == (7.5, 2.25)
