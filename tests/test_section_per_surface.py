"""A SECTION PER SURFACE — and the pipeline that lets a user choose one.

An aircraft with a tail has two lifting surfaces, a foiling craft with an
elevator has two, and a tandem pair has two. Every one of those solvers
carried a field for the second surface's own aerofoil, or could — but the
design tool asked the question once and sent one answer, so both surfaces
flew the wing's section whatever the user did. This file gates the whole
chain, from the solvers up:

1. THE SOLVERS take a section per surface (``polar_tail`` / ``polar_rear``),
   it changes no design variable, and leaving it out reproduces the
   published run exactly — the second surface flies the first's.
2. THE REGISTRY declares :data:`api.SECTION_AFT_KEY` on every family that
   has a second surface, in air AND water, so a shell can find them without
   a list of its own.
3. THE PIPELINE (gui/v3) screens each surface at ITS OWN design point: a
   tail's chord is not the wing's and its design lift is not the mission's
   (it trims), and a tandem pair's two wings split one reference area, so
   they fly different chords at the same lift coefficient. That last one is
   the tandem fix: both wings used to be screened at the whole pair's chord.
4. THE MISSION owns whether there IS a second surface, because that decides
   how many sections stage 2 has to choose.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aerobo import api                                        # noqa: E402
from aerobo.polar import default_polar_family                 # noqa: E402


def _mid(bounds) -> np.ndarray:
    b = np.asarray(bounds, dtype=float)
    return 0.5 * (b[:, 0] + b[:, 1])


#: one family per medium that carries a second surface, and what that
#: surface is called in its own report
AFT_FAMILIES = [
    ("tail", "polar_tail"),
    ("tail + winglet", "polar_tail"),
    ("tandem", "polar_rear"),
    ("tandem (nonplanar) + winglets", "polar_rear"),
    ("hydrofoil + elevator", "polar_tail"),
]

#: ...and which of them can also STATE the section its MAIN surface flies, as
#: a flag. The water family cannot: its foil picks a table by THICKNESS off a
#: polar family, and api._wet_section_kwargs refuses SECTION_KEY on it
#: outright, so there is no name to hand the second surface back.
STATES_ITS_WING_SECTION = {"tail", "tail + winglet", "tandem",
                           "tandem (nonplanar) + winglets"}

#: the section those four are handed to test the FOLLOW contract with. NACA
#: 2412 at Re 1e6 is what polar.default_polar already returns, so stating it
#: cannot turn a feasible midpoint design infeasible — and it is deliberately
#: NOT "naca0012", the section the aft surface is given below, because an aft
#: default that substituted naca0012 is exactly what has to turn this red.
WING_SECTION = "naca2412"


# ------------------------------------------------------- 1. the solvers

@pytest.mark.parametrize("name,key", AFT_FAMILIES)
def test_the_second_surface_flies_its_own_section(name, key):
    """A section for the aft surface changes the RUN, not the vector — and
    the two surfaces report different polars."""
    spec = api.PROBLEM_SPECS[name]
    plain = spec.build(None if not spec.uses_mission else {}, {}, None)
    both = spec.build(None if not spec.uses_mission else {},
                      {api.SECTION_AFT_KEY: "naca0012"}, None)
    assert both.param_labels == plain.param_labels      # no new variable
    x = _mid(plain.bounds)
    a, b = plain.evaluate(x), both.evaluate(x)
    assert a["feasible"] and b["feasible"], (name, a.get("reason"),
                                             b.get("reason"))
    # untouched: the aft surface flies the main one's SECTION — the same
    # aerofoil, mounted the way its load asks. A surface that pushes down
    # flies it inverted (tail.tail_polar), and the reported name says so;
    # what matters here is that no OTHER section got in.
    assert a[key].replace(" (inverted)", "") == a["polar"]
    # ...and given one of its own, it flies that instead
    assert b[key] != b["polar"]
    assert b["polar"] == a["polar"]                     # the wing is untouched
    assert b["score"] != pytest.approx(a["score"], rel=1e-9)

    # THE FOLLOW CONTRACT AS A NUMBER — claim 1 of this module's docstring,
    # "leaving it out reproduces the published run exactly". Stating the very
    # section the aft surface was already following has to give back the same
    # run to the last digit. The line this replaces read
    # ``a["score"] == approx(a["score"])``: one dict value against itself,
    # which is red only for NaN, so an aft default that quietly flew some
    # OTHER table (or the same one mounted differently) moved the published
    # score with nothing here able to see it — the names above still matched.
    assert (api.SECTION_KEY in spec.flags) == \
        (name in STATES_ITS_WING_SECTION), name
    if name not in STATES_ITS_WING_SECTION:
        return                          # the water foil has no name to state
    mission = None if not spec.uses_mission else {}
    follows = spec.build(mission, {api.SECTION_KEY: WING_SECTION}, None)
    restated = spec.build(mission, {api.SECTION_KEY: WING_SECTION,
                                    api.SECTION_AFT_KEY: WING_SECTION}, None)
    assert restated.param_labels == follows.param_labels
    xs = _mid(follows.bounds)
    c, d = follows.evaluate(xs), restated.evaluate(xs)
    assert c["feasible"] and d["feasible"], (name, c.get("reason"),
                                             d.get("reason"))
    assert c[key].replace(" (inverted)", "") == c["polar"] == \
        d[key].replace(" (inverted)", "")
    assert c["score"] == pytest.approx(d["score"], rel=1e-12)


def test_a_tandem_pair_can_fly_two_different_aerofoils():
    """The pair's own case: front and rear are different surfaces carrying
    different shares of the weight, so "one aerofoil for both" is a choice."""
    from aerobo import tandemvlm as tvm

    fam = default_polar_family()
    shared = tvm.TandemVLMProblem(winglets=True)
    split = tvm.TandemVLMProblem(winglets=True, polar_rear=fam.at(0.18))
    assert shared.param_labels == split.param_labels
    x = _mid(shared.bounds)
    a = tvm.evaluate_tandem_vlm(x, shared)
    b = tvm.evaluate_tandem_vlm(x, split)
    assert a["feasible"] and b["feasible"]
    assert a["polar_rear"] == a["polar"]
    assert b["polar_rear"] != b["polar"] and b["polar"] == a["polar"]
    assert b["LoD"] != pytest.approx(a["LoD"], rel=1e-9)
    # each wing's own chord is reported, which is what makes the two
    # Reynolds numbers (and so the two sections) different questions
    assert a["Re_mac"] > 0.0 and a["Re_mac_rear"] > 0.0


# ------------------------------------------------------ 2. the registry

def test_every_family_with_a_second_surface_declares_its_section():
    """Read off the builder, so the modifier twins inherit it."""
    for name, spec in api.PROBLEM_SPECS.items():
        if not getattr(spec.build, "takes_section_aft", False):
            continue
        assert api.SECTION_AFT_KEY in spec.flags, name
    # the families that genuinely have one, in every medium
    for name, _key in AFT_FAMILIES:
        assert api.SECTION_AFT_KEY in api.PROBLEM_SPECS[name].flags, name
    # ...and one that does not
    assert api.SECTION_AFT_KEY not in api.PROBLEM_SPECS["winglet"].flags


def test_a_water_elevators_section_is_swept_with_cp_min():
    """The stabiliser sits deeper than the foil and can cavitate first, so
    its own Cp_min table has to come with its section."""
    spec = api.PROBLEM_SPECS["hydrofoil + elevator"]
    built = spec.build(None, {api.SECTION_AFT_KEY: "naca0012"}, None)
    pol = built.problem.polar_tail
    # the water family SELECTS the foil's section by thickness off the polar
    # family, so there is no foil polar to compare against — what matters is
    # that the elevator's own table can answer a cavitation question
    assert pol is not None and "naca0012" in pol.name
    assert pol.cp_min(np.array([0.0, 2.0])).shape == (2,)


# ------------------------------------------------------- 3. the pipeline

def _session_with(medium: str, **choices):
    import gui.nice_app as v1
    from gui.v3 import session as ses

    S = ses.make_session(medium)
    S["wing"]["choices"].update(choices)
    v1.normalise_choices(S["wing"]["choices"])
    ses.apply_choices(S)
    return S


def test_a_trimming_surface_is_screened_at_its_own_chord_and_trim_lift():
    from gui.v3 import session as ses

    S = _session_with("air", tail=True)
    assert ses.aft_surface(S) == "tail"
    wing = ses.surface_design_point(S, "main")
    tail = ses.surface_design_point(S, "aft")
    # the tail is a small surface: its chord, and so its Re, is its own
    assert tail["mac"] < wing["mac"]
    assert tail["re_mac"] < wing["re_mac"]
    # ...and it TRIMS, so its design lift is not the mission's — but it is
    # NOT zero either: it is what the family's own trim balance says
    trim = ses.trim_lift(S)
    assert trim is not None and abs(trim["cl"]) > 0.01
    # ...screened the way up the section is MOUNTED: this surface pushes
    # down, so it flies its section inverted and an upright catalogue read
    # at |cl| IS that mirrored section read at cl (tail.tail_polar)
    screen_cl = -trim["cl"] if trim["inverted"] else trim["cl"]
    assert tail["cl_design"] == screen_cl != wing["cl_design"]
    assert wing["cl_design"] > 0.0
    assert not ses.surface_geometry(S, "aft")["lifting"]
    # the STAGE on screen is the surface: no target setting to disagree with
    S["ui"]["selected"] = ses.SURFACE_STAGES["aft"]
    assert ses.target_surface(S) == "aft"
    assert ses.section_conditions(S)["cl_design"] == screen_cl
    S["ui"]["selected"] = ses.SURFACE_STAGES["main"]
    assert ses.section_conditions(S)["cl_design"] > 0.0


def test_the_water_elevator_is_a_surface_of_its_own_too():
    from gui.v3 import session as ses

    S = _session_with("water", tail=True)
    assert ses.aft_surface(S) == "elevator"
    assert ses.surface_design_point(S, "aft")["mac"] < \
        ses.surface_design_point(S, "main")["mac"]
    assert not ses.wing_objective(S)          # water: 2-D either way
    S["ui"]["selected"] = ses.SURFACE_STAGES["aft"]
    trim = ses.trim_lift(S)
    assert trim is not None and trim["cl"] > 0.01
    # The stage screens the section the way it is MOUNTED. This elevator is
    # mounted inverted and trims to an UP-load, so the upright catalogue is
    # read at -cl: `InvertedPolar.cl(a) = -base.cl(-a)`, so a surface carrying
    # aircraft-frame C flies its base section at -C. The unsigned form this
    # line used to carry was right only because `abs()` and `-cl` coincide
    # when cl < 0, which is every OTHER family with a second surface (1920 of
    # 2064) and not this one. Same identity as
    # tests/test_trim_surface_lift.py, and gated on the drag and the suction
    # peak in tests/test_the_screened_lift_is_the_flown_one.py.
    assert ses.section_conditions(S)["cl_design"] == (
        -trim["cl"] if trim["inverted"] else trim["cl"])


def test_a_tandem_pairs_wings_are_screened_at_their_own_chords():
    """THE TANDEM FIX. Both wings used to be handed the design point of the
    whole pair — a chord twice either wing's, since the pair splits one
    reference area over two surfaces of the SAME span."""
    from gui.v3 import session as ses

    S = _session_with("air", system="tandem")
    assert ses.aft_surface(S) == "rear wing"
    front = ses.surface_geometry(S, "main")
    rear = ses.surface_geometry(S, "aft")
    assert front is not None and rear is not None
    assert front["lifting"] and rear["lifting"]        # both carry weight
    area = float(S["mission"]["s_ref_m2"])
    span = (ses.nominal_aspect_ratio(S) * area) ** 0.5
    assert front["area"] + rear["area"] == pytest.approx(area)
    assert front["span"] == rear["span"] == pytest.approx(span)
    # each wing's chord is its OWN area over that span, not the pair's
    whole = ses.design_point(S)
    assert front["mac"] == pytest.approx(front["area"] / span)
    assert front["mac"] < whole["mac"]
    # both are lifting surfaces, so both keep the mission's design Cl
    for surface in ("main", "aft"):
        dp = ses.surface_design_point(S, surface)
        assert dp["cl_design"] == pytest.approx(whole["cl_design"])
        assert dp["re_mac"] < whole["re_mac"]
    assert ses.wing_objective(S)              # a lifting surface: wing L/D


def test_a_tandem_pairs_chords_come_from_the_estimate():
    """THE SCREENING CHORD IS STAGE 2'S, ON EVERY FAMILY. A section is
    screened at the chord the ESTIMATE implies; tandem was the one family
    that inverted it, taking its per-surface span off a flown aspect ratio
    instead, so the point moved under a stage that had not been asked.

    There is no second, wing-owned aspect ratio to invert any more (stage 3
    constrains the SPAN, and the aspect ratio it flies is b²/S) — so with the
    planform FIXED, which is where a section is screened, the estimate is the
    only thing this chord can come from, and moving it is what moves the
    screening point. (The pair CAN search its spans — two rows, one per wing,
    tests/test_tandem_two_spans.py — and that is a different state: the
    planform menu has to be taken there, and the size card stops being a pair
    of typed numbers.)"""
    from gui.v3 import session as ses

    S = _session_with("air", system="tandem")
    assert not ses.span_is_searched(S)         # the planform is FIXED here
    assert ses.set_section_aspect_ratio(S, 20.0)      # stage 2's estimate

    area = float(S["mission"]["s_ref_m2"])
    span = (ses.section_aspect_ratio(S) * area) ** 0.5
    front = ses.surface_geometry(S, "main")
    assert front["span"] == pytest.approx(span)
    assert front["mac"] == pytest.approx(front["area"] / span)

    # ...and it is REAL: re-pointing the estimate moves the Reynolds number
    # the surface is screened at, rather than only silencing a note
    S["airfoil"]["re_source"] = "mission"
    before = ses.section_conditions(S, "main")["re"]
    own = ses.surface_design_point(S, "main")
    assert before == pytest.approx(own["re_mac"])
    assert ses.set_section_aspect_ratio(S, 8.0)
    assert ses.section_conditions(S, "main")["re"] != pytest.approx(before)


def test_both_sections_travel_into_the_run():
    from gui.v3 import config, session as ses

    S = _session_with("air", tail=True)
    ses.set_section(S, {"name": "hg40", "source": "library"}, "library")
    assert config.flags(S)[api.SECTION_KEY] == "hg40"
    assert api.SECTION_AFT_KEY not in config.flags(S)   # follows the wing
    ses.set_section(S, {"name": "naca0012", "source": "library"}, None,
                    surface="aft")
    flags = config.flags(S)
    assert flags[api.SECTION_KEY] == "hg40"
    assert flags[api.SECTION_AFT_KEY] == "naca0012"
    # ...and the run built from them is the one the registry accepts
    built = api.PROBLEM_SPECS[S["wing"]["problem"]].build(
        config.mission_kwargs(S), flags, None)
    out = built.evaluate(_mid(built.bounds))
    assert out["feasible"]
    assert out["polar"] != out["polar_tail"]


def test_choosing_for_the_aft_surface_never_gates_the_pipeline():
    """Neither section gates stage 3 — the MISSION does. The second surface's
    is an addition to the wing's, and clearing it goes back to following the
    wing."""
    from gui.v3 import session as ses

    S = _session_with("air", tail=True)
    assert ses.stage_states(S)["wing"][0] == "locked"     # the mission is
    S["mission"]["accepted"] = True
    ses.set_section(S, {"name": "naca0012", "source": "library"}, None,
                    surface="aft")
    assert ses.stage_states(S)["wing"][0] == "ready"      # no wing section yet
    assert ses.section_is_own(S, "aft")
    ses.set_section(S, {"name": "hg40", "source": "library"}, "library")
    assert ses.stage_states(S)["wing"][0] == "ready"
    assert ses.section_of(S, "aft")["name"] == "naca0012"
    ses.set_section(S, None, None, surface="aft")
    assert not ses.section_is_own(S, "aft")
    assert ses.section_of(S, "aft")["name"] == "hg40"     # follows the wing


# -------------------------------------------------- 4. the mission owns it

def test_the_mission_stage_adds_and_removes_the_second_surface(capsys):
    from gui.v3.app import assemble

    ctx = assemble("air")
    assert "set_second_surface" in ctx.actions
    S = ctx.S
    assert not S["wing"]["choices"]["tail"]
    assert __import__("gui.v3.session", fromlist=["x"]).aft_surface(S) is None

    ctx.act("set_second_surface", True)
    assert S["wing"]["choices"]["tail"] is True
    assert "tail" in S["wing"]["problem"]
    assert api.SECTION_AFT_KEY in api.PROBLEM_SPECS[S["wing"]["problem"]].flags

    ctx.act("set_second_surface", False)
    assert S["wing"]["choices"]["tail"] is False
    assert api.SECTION_AFT_KEY not in \
        api.PROBLEM_SPECS[S["wing"]["problem"]].flags
    err = capsys.readouterr().err
    assert "Traceback" not in err, err


def test_the_media_that_cannot_carry_one_say_so(capsys):
    """A car's rear wing has nothing behind it, and a tandem pair's rear
    wing IS the second surface — neither can be given a tail."""
    import gui.nice_app as v1
    from gui.v3.app import assemble

    ctx = assemble("track")
    assert not v1.option_available(ctx.S["wing"]["choices"], "tail", True)
    ctx.act("set_choice", "medium", "air")
    ctx.act("set_choice", "system", "tandem")
    assert not v1.option_available(ctx.S["wing"]["choices"], "tail", True)
    ctx.act("set_choice", "system", "single")
    assert v1.option_available(ctx.S["wing"]["choices"], "tail", True)
    capsys.readouterr()


def test_the_elevator_gets_the_wings_freedoms_through_the_stage(capsys):
    """Stage 3's second-surface card, driven through the real handler: the
    planform, the tip device and the chord law each select a solver that
    actually carries them."""
    from gui.v3.app import assemble

    ctx = assemble("water")
    ctx.act("set_second_surface", True)
    S = ctx.S
    assert S["wing"]["problem"].startswith("hydrofoil + elevator")

    ctx.act("set_choice", "tail_design", "planform")
    assert "designed elevator" in S["wing"]["problem"]
    labels = api.PROBLEM_SPECS[S["wing"]["problem"]].param_labels
    assert {"taper_t", "AR_t", "washout_t_deg"} <= set(labels)

    ctx.act("set_choice", "tail_design", "planform+tip")
    labels = api.PROBLEM_SPECS[S["wing"]["problem"]].param_labels
    assert {"winglet_h_frac_t", "winglet_cant_t_deg"} <= set(labels)

    ctx.act("set_choice", "chord", "free")
    labels = api.PROBLEM_SPECS[S["wing"]["problem"]].param_labels
    assert labels[-6:] == ("chord_k1", "chord_k2", "chord_k3",
                           "chord_k1_t", "chord_k2_t", "chord_k3_t")
    err = capsys.readouterr().err
    assert "Traceback" not in err, err


def test_the_air_tail_gets_the_same_three_freedoms(capsys):
    from gui.v3.app import assemble

    ctx = assemble("air")
    ctx.act("set_second_surface", True)
    ctx.act("set_choice", "tail_design", "planform+tip")
    S = ctx.S
    labels = api.PROBLEM_SPECS[S["wing"]["problem"]].param_labels
    assert {"taper_t", "AR_t", "washout_t_deg", "winglet_h_frac_t",
            "winglet_cant_t_deg"} <= set(labels)
    ctx.act("set_choice", "chord", "free")
    labels = api.PROBLEM_SPECS[S["wing"]["problem"]].param_labels
    assert labels[-3:] == ("chord_k1_t", "chord_k2_t", "chord_k3_t")
    err = capsys.readouterr().err
    assert "Traceback" not in err, err
