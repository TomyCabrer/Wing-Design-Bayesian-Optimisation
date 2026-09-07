"""V5's shell asks: the fin is a MISSION question, and cant is a choice.

Three of the four changes asked for:

1. *"In mission select if the user wants vertical tail independently"* — it
   was not askable at all. The fin could be shaped (volume coefficient,
   aspect ratio, thickness) and charged, but nothing could say the aircraft
   has none.
2. *"Then another airfoil section appears where the user can optimise it
   (should be simpler as this is only for symmetrical airfoil)"* — stage 2.7,
   its own module rather than a third build of stage 2, because a fin has no
   design lift coefficient, nothing to rank a library against, and no
   interior optimum in thickness.
3. *"In Wing Type, dihedral and sweep should be buttons"* — they were typed
   boxes.

The fourth (design-box min/max/fixed selects) needs the cant to become a
DESIGN VARIABLE, which this package deliberately cannot do with a flag:
``ProblemSpec.param_labels`` is static and a flag that changed the vector's
dimension would make it lie. It needs a registered family twin, and that is
a registry decision rather than a shell one.
"""
from __future__ import annotations

import re
import time

import pytest

from gui.v3 import session
from gui.v3.app import assemble


def _texts(el) -> str:
    out = []
    for d in el.descendants():
        for attr in ("text", "_text"):
            v = getattr(d, attr, None)
            if isinstance(v, str) and v:
                out.append(v)
        props = getattr(d, "_props", {}) or {}
        for key in ("label", "text"):
            v = props.get(key)
            if isinstance(v, str) and v:
                out.append(v)
        for opt in (props.get("options") or []):
            if isinstance(opt, dict) and isinstance(opt.get("label"), str):
                out.append(opt["label"])
    return " ".join(out)


def _lattice_shell():
    """A shell on a family whose objective can score a cant."""
    from aerobo import api

    ctx = assemble()
    ctx.act("set_choice", "tail", True)
    ctx.act("set_winglet", "canted")
    name = ctx.S["wing"]["problem"]
    assert set(api.WING_CANT_KEYS) <= set(api.PROBLEM_SPECS[name].flags), name
    return ctx


# ------------------------------------------------- 1. the fin, in the mission

def test_the_mission_asks_whether_there_is_a_vertical_tail():
    ctx = assemble()
    ctx.render("mission", "operating")
    assert "add a vertical stabiliser" in _texts(
        ctx.views[("mission", "operating")])


def test_turning_the_fin_off_hides_its_stage_and_says_what_it_costs():
    ctx = assemble()
    ctx.act("set_fin", False)
    ctx.render("mission", "operating")
    text = _texts(ctx.views[("mission", "operating")])
    # the CONSEQUENCE, not just the state: with no fin nothing makes yaw
    # stiffness, and Cn_beta is exactly zero rather than small
    assert "Cn_beta" in text
    assert not session.stage_visible(ctx.S, "airfoil_fin")
    state, reason = session.stage_states(ctx.S)["airfoil_fin"]
    assert state == "locked"
    assert "no vertical tail" in reason


def test_a_V_TAIL_answers_the_question_by_construction():
    """The layout's own rule reaches the shell from ``fin.size_fin``, which
    returns None for one — so the switch cannot claim a surface the design
    does not have."""
    ctx = assemble()
    ctx.S["wing"]["choices"]["tail_type"] = "v_tail"
    assert not session.stage_visible(ctx.S, "airfoil_fin")
    _state, reason = session.stage_states(ctx.S)["airfoil_fin"]
    assert "V-tail" in reason
    ctx.render("mission", "operating")
    assert "by construction" in _texts(ctx.views[("mission", "operating")])


def test_the_fin_is_ON_by_default_because_every_published_run_flew_one():
    """The SWITCH defaults on — but the stage still needs a family with a
    fin in it.

    This used to assert stage 2.7 was visible on a FRESH session, and it
    was: the default family is a plain trimmed wing, which declares none of
    the fin keys, so the shell offered to choose a section for a surface no
    run of that family would ever build, charge, weigh or fly. Adding a tail
    is what puts the aircraft on a family that has one — the same rule
    stage 3's Wing type card already applied to its own fin block.
    """
    ctx = assemble()
    assert session.FIN_DEFAULT is True
    assert not session.stage_visible(ctx.S, "airfoil_fin"), \
        "a plain wing was offered a fin's section stage"
    ctx.act("set_choice", "tail", True)
    assert session.stage_visible(ctx.S, "airfoil_fin")
    assert ctx.S["wing"]["choices"].get("fin", session.FIN_DEFAULT) is True


# --------------------------------------------- 2. the fin's own section stage

def test_the_stabiliser_stage_IS_stage_two_asked_a_third_time():
    """One question asked three times must not become three forms.

    It was briefly its own cut-down module, on the argument that a fin has
    no design lift coefficient and no shape to search. The first half is
    true; the second was wrong — a fin's section is the same CST-through-
    XFOIL problem with ``w_lower = -w_upper`` and ``cl_design = 0``. So the
    stage shares the module and the view tuple, exactly as the aft surface
    does, and the SAME four views appear.
    """
    from gui.v3.app import STAGE_MODULES

    assert "airfoil_fin" in session.STAGES
    assert STAGE_MODULES["airfoil_fin"] == STAGE_MODULES["airfoil"] == "airfoil"
    assert session.VIEWS["airfoil_fin"] is session.VIEWS["airfoil"]
    assert [v[0] for v in session.VIEWS["airfoil_fin"]] == \
        ["screen", "ranking", "section", "optimise"]


def test_all_four_of_its_views_build():
    ctx = assemble()
    for view in ("screen", "ranking", "section", "optimise"):
        ctx.render("airfoil_fin", view)
        assert ctx.views[("airfoil_fin", view)] is not None


def test_it_is_screened_at_ZERO_LIFT_and_its_OWN_reynolds_number():
    """The two things that make it a fin's section rather than a wing's."""
    ctx = assemble()
    geo = session.surface_geometry(ctx.S, "fin")
    assert geo is not None
    assert geo["cl"] == 0.0, "a fin's design point is no side force at all"
    assert geo["lifting"] is False
    # its OWN chord: 0.696 m against a wing mac of about 1.02
    assert 0.5 < geo["mac"] < 0.9

    ctx.render("airfoil_fin", "screen")
    text = _texts(ctx.views[("airfoil_fin", "screen")])
    assert "vertical stabiliser" in text
    assert "0.0000" in text                      # screened at Cl 0
    assert "volume coefficient" in text          # sized by the one law


def test_its_size_comes_from_the_ONE_law_that_sizes_a_fin():
    """``fin.size_fin`` returns None for a V-tail, so the stage cannot
    describe a surface the design does not have."""
    ctx = assemble()
    assert session.surface_geometry(ctx.S, "fin") is not None
    ctx.S["wing"]["choices"]["tail_type"] = "v_tail"
    assert session.surface_geometry(ctx.S, "fin") is None
    assert not session.stage_visible(ctx.S, "airfoil_fin")


def test_the_symmetric_section_problem_is_HALF_the_wings_vector():
    """"Simpler because it is only for a symmetrical aerofoil", made real:
    ``w_lower = -w_upper``, so four variables instead of eight."""
    import numpy as np

    from aerobo.airfoil import AirfoilProblem

    wing = AirfoilProblem()
    fin = AirfoilProblem(symmetric=True, cl_design=0.0)
    assert wing.dim == 8 and fin.dim == 4
    assert fin.param_labels == tuple(f"w_upper_{i}" for i in range(4))
    assert fin.bounds.shape[0] == 4
    w_u, w_l = fin.split_weights(fin.w0)
    assert np.allclose(w_l, -w_u), "the lower surface must be a MIRROR"
    # ...and the wing's problem is untouched
    assert np.array_equal(wing.w0, AirfoilProblem().w0)
    assert wing.param_labels[4:] == tuple(f"w_lower_{i}" for i in range(4))


def test_the_symmetric_anchor_is_the_anchors_own_THICKNESS():
    """Mirroring a CAMBERED upper surface is not symmetrising a section.

    The default anchor is a NACA 2412 — 12 % thick. Its symmetric twin is a
    0012, and getting there means halving the THICKNESS distribution
    (``(w_u - w_l)/2``), not reflecting the upper surface. Reflect it and the
    box centre comes out at t/c 0.158: a 16 %-thick section wearing a 12 %
    section's name, with a design box built around it.
    """
    from aerobo.airfoil import AirfoilProblem, cst_coords

    fin = AirfoilProblem(symmetric=True, cl_design=0.0)
    w_u, w_l = fin.split_weights(fin.w0)
    xy = cst_coords(w_u, w_l)
    tc = float(xy[:, 1].max() - xy[:, 1].min())
    assert tc == pytest.approx(0.12, abs=0.005), (
        f"the symmetric anchor is {tc:.4f} thick; a 2412's symmetric twin is "
        f"a 0012 and reflecting the upper surface instead gives 0.158")
    # ...and the box is built around THAT shape, not the cambered one
    assert fin.bounds[:, 1].max() < 0.33


def test_a_symmetric_candidate_really_is_symmetric_through_XFOIL():
    """Not just in the vector: the section it builds has to have no camber,
    and XFOIL has to agree."""
    from aerobo.airfoil import AirfoilProblem, evaluate_airfoil

    fin = AirfoilProblem(symmetric=True, cl_design=0.0, re=7e5)
    out = evaluate_airfoil(fin.w0, fin)
    assert out.get("feasible")
    # a symmetric section's quarter-chord pitching moment is zero identically
    assert abs(float(out["cm"])) < 1e-6, out["cm"]


def test_the_stabilisers_screening_preset_drops_what_it_cannot_rank():
    """|Cm| is identically zero on a symmetric section and cruise L/D needs
    a design lift it does not have. Ranking on either would be ranking
    nothing, so both are zero and the weight goes where it means something."""
    weights, why = session.recommended_weights(assemble().S, "fin")
    assert weights["cm"] == 0.0
    assert weights["ldcr"] == 0.0
    assert weights["ldmax"] > 0.0 and weights["thick"] > 0.0
    assert weights["astall"] > 0.0, \
        "a rudder earns its section at DEFLECTION, so stall angle counts"
    assert sum(weights.values()) == pytest.approx(1.0)
    assert "symmetric" in why.lower()


def test_the_stabiliser_has_its_OWN_job_not_the_trim_surfaces():
    """A trimming surface's section is chosen for the download it makes; a
    fin's for making none. Different job, different preset."""
    S = assemble().S
    assert session.surface_job(S, "fin") == "fin"
    assert session.surface_job(S, "main") != "fin"
    assert session.recommended_weights(S, "fin")[0] != \
        session.JOB_WEIGHTS["trim"][0]


def test_its_chosen_section_has_a_home_of_its_own():
    """And a full clear takes it with everything else — a section screened
    for the old vehicle must not survive a medium change."""
    assert session.SECTION_KEYS["fin"] == "section_fin"
    S = assemble().S
    session.set_section(S, {"name": "NACA 0010"}, None, surface="fin")
    assert S["airfoil"]["section_fin"]["name"] == "NACA 0010"
    assert session.section_is_own(S, "fin")
    session.set_section(S, None, None)
    assert S["airfoil"]["section_fin"] is None


# ------------------ a cambered section must not reach the stabiliser, anywhere

def test_a_CAMBERED_library_section_cannot_win_the_stabilisers_screen():
    """GOE741 won a fin screen. It is 4.8 % cambered.

    The optimiser was made symmetric and the LIBRARY SCREEN was not, so the
    stage ranked all 2174 members for a surface that must make no side force
    and a cambered one won on merit. Symmetry is measured from the
    COORDINATES, never from the name: most library members' names say
    nothing about camber, and ``goe741`` says nothing either.
    """
    from aerobo import api

    names = api.symmetric_section_names()
    assert names, "the coordinate sidecar is needed for this test"
    assert "goe741" not in names
    coords = api._coords_sidecar()
    assert api.section_max_camber(coords["goe741"]) > 0.04
    # ...and a genuinely symmetric one survives the filter
    assert any(api.section_max_camber(coords[n]) <= api.SYMMETRIC_CAMBER_TOL
               for n in names)
    # the restriction is a real cut, not a no-op
    assert 0 < len(names) < len(coords) // 2

    # ...and the STAGE applies it. Asserted through the same module-level
    # helper the screen calls, so the rule cannot live in a handler branch
    # no test can reach — which is how it came to be missing.
    from gui.v3.stages.airfoil import screen_names_for

    assert screen_names_for("main") is None       # the wing ranks everything
    assert screen_names_for("aft") is None
    fin_only = screen_names_for("fin")
    assert fin_only == names
    assert "goe741" not in fin_only


def test_the_restriction_reaches_the_SCREEN_CALL_not_only_the_helper():
    """The rule has to arrive at ``api.screen_airfoils``, not merely exist.

    The first version of this fix computed the restriction and the mutant
    that dropped it into the void stayed green, because every test asked the
    helper instead of the call.
    """
    from aerobo import api

    seen = {}

    def _fake(**kw):
        seen.update(kw)
        raise RuntimeError("stop here — the arguments are the assertion")

    # BOTH entry points, because which one runs depends on whether a library
    # cache is warm — and the restriction has to survive either route. The
    # first version of this test patched only ``screen_airfoils`` and the
    # stage took the shortlist path, so it asserted nothing at all.
    ctx = assemble()
    real_a, real_b = api.screen_airfoils, api.screen_at_point
    try:
        api.screen_airfoils = _fake
        api.screen_at_point = _fake
        ctx.act("run_airfoil_fin")
        for _ in range(200):
            if seen or ctx.S["airfoil_fin"]["screen"].get("error"):
                break
            time.sleep(0.05)
    finally:
        api.screen_airfoils, api.screen_at_point = real_a, real_b

    assert seen, "the fin's screen called neither screening entry point"
    assert "names" in seen, "the fin screened the WHOLE library"
    assert "goe741" not in seen["names"]
    assert set(seen["names"]) == set(api.symmetric_section_names())


def test_each_surface_has_its_OWN_action_namespace():
    """Three surfaces, three Run buttons. The suffix was
    ``"" if main else "_aft"``, so the stabiliser registered its Run and Stop
    under the AFT surface's names and whichever stage built last owned
    them — one surface's button driving the other's search."""
    ctx = assemble()
    for name in ("run_airfoil", "run_airfoil_aft", "run_airfoil_fin",
                 "stop_airfoil", "stop_airfoil_aft", "stop_airfoil_fin"):
        assert name in ctx.actions, name


def test_the_stabiliser_does_NOT_inherit_the_wings_cambered_section():
    """The deeper half of the same bug, and the one that was on by default.

    Every other secondary surface flies the wing's section until one is
    chosen for it — that is the default the solvers carry. A fin must not:
    the wing's is cambered, so the DEFAULT state handed a stabiliser a
    permanent side load with nothing to trim it against.
    """
    S = assemble().S
    S["airfoil"]["section"] = {"name": "goe741"}
    assert session.section_of(S, "main")["name"] == "goe741"
    assert session.section_of(S, "aft")["name"] == "goe741"   # correct
    fin = session.section_of(S, "fin")
    assert fin["name"] != "goe741"
    assert fin["symmetric"] is True
    assert fin["name"] == "NACA 0010"
    # ...and the default FOLLOWS the thickness, so the name cannot describe
    # a section nobody is flying
    S["wing"].setdefault("flags", {})["fin_tc"] = 0.14
    assert session.section_of(S, "fin")["name"] == "NACA 0014"


def test_the_stabilisers_SEARCH_cannot_leave_the_symmetric_family():
    """The third path. It is set in ``shape_kwargs`` — the dict that says
    what a run IS — so a "Continue" cannot resume a fin's search over
    cambered shapes."""
    from gui.v3.stages.airfoil import shape_kwargs

    S = assemble().S
    for surface, want in (("main", False), ("aft", False), ("fin", True)):
        A = session.airfoil_state(S, surface)
        kw = shape_kwargs(S, surface, A["opt"], A["weights"], None)
        assert kw["symmetric"] is want, surface
    fin_kw = shape_kwargs(S, "fin", session.airfoil_state(S, "fin")["opt"],
                          session.airfoil_state(S, "fin")["weights"], None)
    assert fin_kw["cl_design"] == 0.0


def test_the_symmetric_flag_reaches_the_PROBLEM_and_halves_it():
    from aerobo import api
    from aerobo.airfoil import AirfoilProblem

    cfg = api.airfoil_run_config(symmetric=True, cl_design=0.0, re=7e5)
    assert cfg.flags["airfoil_symmetric"] is True
    prob = AirfoilProblem(**api._airfoil_kwargs(cfg.flags))
    assert prob.symmetric and prob.dim == 4 and prob.cl_design == 0.0
    # ...and an unstated one leaves every frozen run's config hash alone
    assert "airfoil_symmetric" not in api.airfoil_run_config().flags


# ------------------------------------------------- 3. cant and sweep as buttons

def test_the_cant_is_offered_as_BUTTONS_not_a_typed_box_alone():
    ctx = _lattice_shell()
    ctx.render("wing", "type")
    box = ctx.views[("wing", "type")]
    toggles = [d for d in box.descendants()
               if type(d).__name__ == "Toggle"
               and any(str(o.get("label")) in ("5.7 deg", "30 deg")
                       for o in (d._props.get("options") or []))]
    assert len(toggles) == 2, "expected a button row for dihedral AND sweep"
    labels = [[o["label"] for o in t._props["options"]] for t in toggles]
    assert labels[0] == ["none", "2 deg", "5.7 deg", "6 deg"]
    assert labels[1] == ["none", "10 deg", "20 deg", "30 deg"]
    # ...AND THE RULE BEHIND THOSE STRINGS, so a qualitative word cannot come
    # back in by a different route. A step is "none" — which is not a value,
    # it is the branch that stores no flag at all — or it states its own
    # angle. "a little", "enough" and "a lot" were claims about an aeroplane
    # the step had never been measured on.
    for row in labels:
        for text in row:
            assert text == "none" or re.fullmatch(r"-?\d+(\.\d+)? deg",
                                                  text), \
                f"a cant step must state its own value, not {text!r}"


def test_a_button_stores_the_flag_and_NONE_stores_nothing():
    """Bit-for-bit matters here: "none" must send no flag at all, or every
    published run acquires a `wing_dihedral_deg=0.0` it never had."""
    from gui.v3 import config

    ctx = _lattice_shell()
    ctx.render("wing", "type")

    def _toggle():
        return [d for d in ctx.views[("wing", "type")].descendants()
                if type(d).__name__ == "Toggle"
                and any(str(o.get("label")) == "5.7 deg"
                        for o in (d._props.get("options") or []))][0]

    _toggle().set_value("5.7")
    assert ctx.S["wing"]["flags"]["wing_dihedral_deg"] == pytest.approx(5.7)
    assert config.flags(ctx.S)["wing_dihedral_deg"] == pytest.approx(5.7)

    _toggle().set_value("0.0")
    assert ctx.S["wing"]["flags"].get("wing_dihedral_deg") is None
    assert "wing_dihedral_deg" not in config.flags(ctx.S)


def test_the_typed_box_survives_as_the_escape_hatch():
    """A button set is a recommendation, never a limit — the same contract
    every other recommended value in this shell carries."""
    ctx = _lattice_shell()
    ctx.render("wing", "type")
    assert "state it" in _texts(ctx.views[("wing", "type")])
    # driven through the FIELD the user types into, not a handler — the
    # point of the escape hatch is that the widget exists and works
    nums = [d for d in ctx.views[("wing", "type")].descendants()
            if type(d).__name__ == "Number"]
    assert nums, "no typed field beside the buttons"
    held = ctx.S["wing"]["flags"].get("wing_dihedral_deg")
    for n in nums:
        n.set_value(3.7)
        if ctx.S["wing"]["flags"].get("wing_dihedral_deg") != held:
            break
    assert ctx.S["wing"]["flags"]["wing_dihedral_deg"] == pytest.approx(3.7)
    ctx.render("wing", "type")
    assert "your own number" in _texts(ctx.views[("wing", "type")])


def test_the_button_values_are_the_MEASURED_ones():
    """5.7 deg is where this family's spiral MODE crosses — not a round
    number chosen for the label.

    It read 3.56 until 2026-08-31, which is where the closed-form criterion
    crossed while it was still dropping the trim attitude: 2.13 deg
    optimistic, and the first step that actually converges the mode is 6.0.
    The value is RE-DERIVED against a measurement in
    ``tests/test_the_spiral_criterion_tells_the_truth.py``; this only pins
    that the card carries the measured one and not the old one.
    """
    from gui.v3.stages import wing as wing_stage

    src = open(wing_stage.__file__).read()
    ladder = src.split("CANT_STEPS")[1]
    assert '("5.7 deg", 5.7' in ladder, \
        "the measured crossing should be one of the steps"
    assert ", 3.56," not in ladder, \
        "the optimistic crossing is being offered again"
