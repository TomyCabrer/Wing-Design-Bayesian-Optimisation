"""Stage 2.5 — the second surface's own airfoil stage, and the criterion
weights each surface OPENS on.

Two rules are gated here:

* deciding in stage 1 that the vehicle has a tail (or a tandem rear wing)
  makes a NEW stage appear in the tree — a copy of stage 2, asking the same
  question for that surface, with its own workspace. It never appears where
  there is no such surface, and it never gates anything: a second surface
  with no section of its own flies the wing's, exactly as the solvers do.
* the weights each form opens on are the ones RECOMMENDED for that surface's
  job, and they come from ``airfoil_select.PRESETS`` — GDP's own front-wing
  and rear-wing presets for a tandem pair, the bulk-sweep preset for a wing,
  and for a trimming surface the symmetric preset with the one criterion
  that cannot rank anything at zero lift moved aside. An EDITED set is the
  user's and is never moved again.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _session(medium="air", **choices):
    import gui.nice_app as v1
    from gui.v3 import session as ses

    S = ses.make_session(medium)
    for key, value in choices.items():
        S["wing"]["choices"][key] = value
        v1.normalise_choices(S["wing"]["choices"], keep=key)
    ses.apply_choices(S)
    S["mission"]["accepted"] = True
    return S


# ----------------------------------------------------- the stage exists
def test_the_second_surfaces_stage_appears_only_when_it_has_a_surface():
    from gui.v3 import session as ses

    plain = _session("air")
    assert ses.aft_surface(plain) is None
    assert not ses.stage_visible(plain, "airfoil_aft")
    assert ses.stage_states(plain)["airfoil_aft"][0] == "locked"
    assert "stage 1" in ses.stage_states(plain)["airfoil_aft"][1]

    tailed = _session("air", tail=True)
    assert ses.aft_surface(tailed) == "tail"
    assert ses.stage_visible(tailed, "airfoil_aft")
    assert ses.stage_states(tailed)["airfoil_aft"][0] == "ready"


def test_the_stage_is_named_after_the_surface_it_designs():
    from gui.v3 import session as ses

    plain = _session("air")
    assert ses.stage_label(plain, "airfoil") == "2  Airfoil"

    for medium, kwargs, name in (("air", {"tail": True}, "tail"),
                                 ("water", {"tail": True}, "elevator"),
                                 ("air", {"system": "tandem"}, "rear wing")):
        S = _session(medium, **kwargs)
        assert ses.aft_surface(S) == name
        assert ses.stage_label(S, "airfoil_aft") == f"2.5  Airfoil · {name}"
        # ...and stage 2 says which surface IT is, once there are two
        assert ses.stage_label(S, "airfoil") == "2  Airfoil · wing"


def test_the_surface_stages_ask_the_same_question_from_the_same_views():
    """THREE surfaces now, and the point is unchanged: one question asked
    three times must not become three forms.

    The vertical stabiliser joined in V5 (stage 2.7). It shares the module
    and the view tuple for the same reason the aft surface does — what
    differs is the design point it is asked at (Cl 0, its own chord) and the
    family it may search (symmetric), not the form.
    """
    from gui.v3 import session as ses

    assert ses.VIEWS["airfoil_aft"] is ses.VIEWS["airfoil"]
    assert ses.VIEWS["airfoil_fin"] is ses.VIEWS["airfoil"]
    # ...and the CAR ENDPLATE's, which joined as stage 2.8 for the same
    # reason the fin did: it is a surface with its own chord, its own
    # Reynolds number and a section of its own to choose.
    assert ses.VIEWS["airfoil_plate"] is ses.VIEWS["airfoil"]
    assert ses.SURFACE_STAGES == {"main": "airfoil", "aft": "airfoil_aft",
                                  "fin": "airfoil_fin",
                                  "plate": "airfoil_plate"}
    assert ses.STAGE_SURFACE["airfoil_aft"] == "aft"
    assert ses.STAGE_SURFACE["airfoil_fin"] == "fin"
    assert ses.STAGE_SURFACE["airfoil_plate"] == "plate"
    # every surface but the wing keeps its chosen section under its own key,
    # and the wing's is absent because it is the one the others inherit
    assert set(ses.SECTION_KEYS) == set(ses.SURFACE_STAGES) - {"main"}


def test_each_surface_has_its_own_workspace():
    """One ranking per surface. Sharing one was what left a table screened
    for the wing on screen while the elevator's stage was being read."""
    from gui.v3 import session as ses

    S = _session("air", tail=True)
    main, aft = ses.airfoil_state(S, "main"), ses.airfoil_state(S, "aft")
    assert main is S["airfoil"] and aft is S["airfoil_aft"]
    assert main is not aft

    main["screen"]["report"] = {"conditions": {"re": 1e6, "cl_design": 0.5}}
    assert aft["screen"]["report"] is None
    aft["top_n"] = 3
    assert main["top_n"] != 3


def test_the_stage_on_screen_is_the_surface_being_designed():
    """ONE source. There is no target setting that can point at a surface
    the user is not looking at."""
    from gui.v3 import session as ses

    S = _session("air", tail=True)
    S["ui"]["selected"] = "airfoil"
    assert ses.target_surface(S) == "main"
    assert ses.section_conditions(S)["cl_design"] > 0.0

    S["ui"]["selected"] = "airfoil_aft"
    assert ses.target_surface(S) == "aft"
    # screened the way up the section is MOUNTED: this surface pushes down,
    # so it flies its section inverted and an upright catalogue read at |cl|
    # IS that mirrored section read at cl (tail.tail_polar)
    trim = ses.trim_lift(S)
    assert ses.section_conditions(S)["cl_design"] == (
        -trim["cl"] if trim["inverted"] else trim["cl"])

    # a configuration with no second surface cannot be pointed at one
    plain = _session("air")
    plain["ui"]["selected"] = "airfoil_aft"
    assert ses.target_surface(plain) == "main"


def test_the_second_surfaces_stage_gates_nothing():
    from gui.v3 import session as ses

    S = _session("air", tail=True)
    ses.set_section(S, {"name": "hg40", "source": "library"}, "library")
    assert ses.stage_states(S)["wing"][0] != "locked"
    assert ses.stage_states(S)["airfoil_aft"][0] == "ready"
    assert ses.section_of(S, "aft")["name"] == "hg40"    # flies the wing's

    ses.set_section(S, {"name": "naca0012", "source": "library"}, None,
                    surface="aft")
    assert ses.stage_states(S)["airfoil_aft"][0] == "done"
    assert ses.stage_states(S)["wing"][0] != "locked"
    assert ses.section_summary(S, "aft") == "naca0012"


def test_the_wings_section_hands_over_to_the_second_surfaces_stage(capsys):
    """"Done here" follows the PIPELINE, not a fixed destination. With two
    sections to choose, finishing the wing's must offer the second surface's
    stage — sending the user to stage 3 there skips a question they never
    answered, and the surface then flies the wing's aerofoil because nothing
    asked."""
    from gui.v3 import session as ses
    from gui.v3.app import assemble

    ctx = assemble()
    ctx.act("accept_mission")
    ctx.S["wing"]["choices"]["tail"] = True
    v1_normalise(ctx)

    ses.set_section(ctx.S, {"name": "hg40", "source": "library"}, "library")
    ctx.render("airfoil", "section")
    texts = [getattr(e, "text", "") or ""
             for e in ctx.views[("airfoil", "section")].descendants()]
    assert any("Choose the tail's section" in t for t in texts)
    assert not any("Go to the wing stage" in t for t in texts)

    # one surface: stage 3 is next. Since V5 the FIN is a surface with a
    # section of its own too, so "one surface" has to say so — with a
    # vertical stabiliser on the vehicle the wing's section hands over to
    # stage 2.7, which is the pipeline working, not this test failing.
    plain = assemble()
    plain.act("accept_mission")
    plain.act("set_fin", False)
    ses.set_section(plain.S, {"name": "hg40", "source": "library"}, "library")
    plain.render("airfoil", "section")
    texts = [getattr(e, "text", "") or ""
             for e in plain.views[("airfoil", "section")].descendants()]
    assert any("Go to the wing stage" in t for t in texts)
    capsys.readouterr()


def v1_normalise(ctx):
    """Apply a choice through the same path the mission stage's switch uses."""
    import gui.nice_app as v1
    from gui.v3 import session as ses

    v1.normalise_choices(ctx.S["wing"]["choices"], keep="tail")
    ses.apply_choices(ctx.S)
    ctx.render("airfoil")
    ctx.refresh()


# --------------------------------------------- the recommended weights
def test_the_recommended_weights_are_the_gdp_presets_not_our_own_numbers():
    """Provenance. Three of the four sets ARE ``airfoil_select.PRESETS``;
    if a preset moves, this fails rather than the shell drifting away from
    the methodology it was ported from."""
    from aerobo.airfoil_select import PRESETS
    from gui.v3 import session as ses

    def as_dict(preset):
        return {k: getattr(preset, k) for k in
                ("thick", "clmax", "ldmax", "ldcr", "cm", "astall")}

    assert ses.DEFAULT_WEIGHTS == as_dict(PRESETS["gdp-sweep"])
    assert ses.TANDEM_FRONT_WEIGHTS == as_dict(PRESETS["gdp-fwd"])
    assert ses.TANDEM_REAR_WEIGHTS == as_dict(PRESETS["gdp-rear"])


def test_the_trim_weights_are_the_symmetric_preset_verbatim():
    """A trimming surface is screened at the lift its family's own trim
    balance says it carries — not at zero — so "L/D at the design Cl" ranks
    something again and GDP's symmetric preset can be used as published.

    The zeroed-ldcr variant this replaced was a workaround for the zero: it
    handed that weight to |Cm|, i.e. to camber, which made a symmetric
    section the answer by construction."""
    from aerobo.airfoil_select import PRESETS
    from gui.v3 import session as ses

    sym = PRESETS["gdp-sym"]
    assert ses.TRIM_WEIGHTS == {k: getattr(sym, k) for k in
                                ("thick", "clmax", "ldmax", "ldcr", "cm",
                                 "astall")}
    assert ses.TRIM_WEIGHTS["ldcr"] > 0.0


@pytest.mark.parametrize("medium,kwargs,job_main,job_aft", [
    ("air", {}, "wing", None),
    # a wing WITH a trimming surface is a different job from a wing without
    # one: the tail carries its pitching moment, so |Cm| stops being ranked
    # and reflexed sections stop being eligible (session.nose_down_required)
    ("air", {"tail": True}, "wing-trimmed", "trim"),
    ("water", {"tail": True}, "wing-trimmed", "trim"),
    ("air", {"system": "tandem"}, "tandem-front", "tandem-rear"),
])
def test_every_surface_opens_on_the_weights_recommended_for_its_job(
        medium, kwargs, job_main, job_aft):
    from gui.v3 import session as ses

    S = _session(medium, **kwargs)
    assert ses.surface_job(S, "main") == job_main
    assert ses.airfoil_state(S, "main")["weights"] == \
        ses.JOB_WEIGHTS[job_main][0]
    assert ses.weights_are_recommended(S, "main")
    if job_aft is None:
        return
    assert ses.surface_job(S, "aft") == job_aft
    assert ses.airfoil_state(S, "aft")["weights"] == ses.JOB_WEIGHTS[job_aft][0]
    assert ses.weights_are_recommended(S, "aft")
    # every recommendation carries the reason it is one
    assert len(ses.recommended_weights(S, "aft")[1]) > 30


def test_the_weights_follow_the_configuration_until_the_user_moves_them():
    from gui.v3 import session as ses

    S = _session("air")
    assert ses.airfoil_state(S, "main")["weights"] == ses.DEFAULT_WEIGHTS

    # a tandem re-opens BOTH forms on the pair's own presets...
    S["wing"]["choices"]["system"] = "tandem"
    import gui.nice_app as v1
    v1.normalise_choices(S["wing"]["choices"], keep="system")
    notes = ses.apply_choices(S)
    assert any("front" in n.lower() for n in notes)
    assert ses.airfoil_state(S, "main")["weights"] == ses.TANDEM_FRONT_WEIGHTS
    assert ses.airfoil_state(S, "aft")["weights"] == ses.TANDEM_REAR_WEIGHTS

    # ...but an EDITED set is an answer, and nothing here resets an answer
    ses.set_weights(S, "aft", {"ldcr": 1.0, "clmax": 0.0, "cm": 0.0,
                               "ldmax": 0.0, "thick": 0.0, "astall": 0.0},
                    "user")
    S["wing"]["choices"]["system"] = "single"
    v1.normalise_choices(S["wing"]["choices"], keep="system")
    S["wing"]["choices"]["tail"] = True
    v1.normalise_choices(S["wing"]["choices"], keep="tail")
    ses.apply_choices(S)
    assert ses.surface_job(S, "aft") == "trim"
    assert ses.airfoil_state(S, "aft")["weights"]["ldcr"] == 1.0
    assert not ses.weights_are_recommended(S, "aft")
    # the wing's, untouched, did follow
    assert ses.weights_are_recommended(S, "main")


def test_the_second_surfaces_section_travels_on_its_own_account():
    """An unanswered stage 2 is a statement about the WING — it flies the
    family's own published section. A section chosen on stage 2.5 is a
    separate flag, and dropping it because the wing has none would be a
    silent loss."""
    from aerobo import api
    from gui.v3 import config, session as ses

    S = _session("air", tail=True)
    assert api.SECTION_AFT_KEY in api.PROBLEM_SPECS[S["wing"]["problem"]].flags
    ses.set_section(S, {"name": "naca0012", "source": "library"}, None,
                    surface="aft")
    flags = config.flags(S)
    assert flags[api.SECTION_AFT_KEY] == "naca0012"
    assert api.SECTION_KEY not in flags                   # the wing's is not
    # and with no section of its own, nothing is sent for it at all
    ses.set_section(S, None, None, surface="aft")
    assert api.SECTION_AFT_KEY not in config.flags(S)


def test_a_ranking_from_another_design_lift_is_reported_as_stale():
    """On the CACHED library point both surfaces share a Reynolds number, so
    the Re check cannot see a surface that changed job. The design lift did
    move, and that reorders the whole table."""
    from gui.v3 import session as ses

    S = _session("air", tail=True)
    aft = ses.airfoil_state(S, "aft")
    aft["screen"]["report"] = {"conditions": {
        "re": ses.section_conditions(S, "aft")["re"], "cl_design": 0.5}}
    assert ses.section_point_is_stale(S, "aft") is None
    trim = ses.trim_lift(S)
    screen_cl = -trim["cl"] if trim["inverted"] else trim["cl"]
    was, now = ses.section_lift_is_stale(S, "aft")
    assert (was, now) == (0.5, screen_cl)

    # ...and a table screened AT the current lift is not stale. This used to
    # write the trim_lift DICT here, which `float()` refuses: the checker
    # swallowed the TypeError and returned None before comparing anything, so
    # the "nothing moved" branch was never executed and deleting the freshness
    # comparison outright (every ranking nagged as stale, forever) stayed green.
    aft["screen"]["report"]["conditions"]["cl_design"] = screen_cl
    assert ses.section_lift_is_stale(S, "aft") is None
    # a re-derivation of the SAME lift differs in the last bits and must not
    # nag either; a lift that genuinely moved still must.
    aft["screen"]["report"]["conditions"]["cl_design"] = screen_cl + 1e-12
    assert ses.section_lift_is_stale(S, "aft") is None
    aft["screen"]["report"]["conditions"]["cl_design"] = screen_cl + 1e-6
    assert ses.section_lift_is_stale(S, "aft") == (screen_cl + 1e-6, screen_cl)


def test_a_second_lifting_surface_is_optimised_on_its_own_wing():
    """The wing-L/D objective flies a wing. A tandem pair's rear wing is not
    the pair — it carries its share of the area on the same span."""
    from gui.v3 import session as ses

    S = _session("air", system="tandem")
    assert ses.wing_objective(S, "aft")            # it lifts: wing mode
    pair = float(S["mission"]["s_ref_m2"])
    rear = ses.wing_guess(S, "aft")
    assert rear["s_ref_m2"] < pair
    front = ses.wing_guess(S, "main")
    assert front["s_ref_m2"] + rear["s_ref_m2"] == pytest.approx(pair)

    tailed = _session("air", tail=True)
    assert not ses.wing_objective(tailed, "aft")   # it trims: 2-D


# ------------------------------------------------------- through the shell
def test_the_shell_grows_the_stage_when_stage_one_adds_the_surface(capsys):
    """Driven through the REAL handlers: the mission stage's switch, and the
    tree the shell paints from it."""
    from gui.v3 import app, session
    from gui.v3.app import assemble

    ctx = assemble()
    ctx.act("accept_mission")
    keys = [n["key"] for n in app._nodes(ctx)]
    assert "airfoil_aft" not in keys

    ctx.act("set_second_surface", True)
    assert ctx.S["wing"]["choices"]["tail"]
    keys = [n["key"] for n in app._nodes(ctx)]
    assert "airfoil_aft" in keys
    node = next(n for n in app._nodes(ctx) if n["key"] == "airfoil_aft")
    assert node["label"] == session.stage_label(ctx.S, "airfoil_aft")
    assert node["badge"] == "as the wing"          # until one is chosen

    # every view of the new stage renders, and it can be navigated to
    for key, _lbl, _icon in session.VIEWS["airfoil_aft"]:
        ctx.render("airfoil_aft", key)
    ctx.select("airfoil_aft", "screen")
    assert ctx.S["ui"]["selected"] == "airfoil_aft"
    assert session.target_surface(ctx.S) == "aft"

    # ...and taking the surface away takes the stage with it
    ctx.act("set_second_surface", False)
    assert "airfoil_aft" not in [n["key"] for n in app._nodes(ctx)]
    err = capsys.readouterr().err
    assert "Traceback" not in err, err


def test_the_two_stages_screen_and_stop_independently(capsys):
    from gui.v3.app import assemble

    ctx = assemble()
    ctx.act("accept_mission")
    ctx.act("set_second_surface", True)
    for action in ("run_airfoil", "stop_airfoil", "run_airfoil_aft",
                   "stop_airfoil_aft"):
        assert action in ctx.actions, action
    # THE WING'S SKIP IS GONE. It declared what an unanswered stage 2 already
    # means, so stage 3 is reachable without it; the second surface's own skip
    # ("fly the wing's section") stays, because that one states an alternative
    # nothing else can express.
    assert "use_solver_section" not in ctx.actions
    from gui.v3 import session as ses

    assert ctx.S["airfoil"]["decision"] is None
    assert ses.stage_states(ctx.S)["wing"][0] != "locked"
    capsys.readouterr()


def test_stage_three_sends_the_user_to_the_second_surfaces_stage(capsys):
    from gui.v3 import session
    from gui.v3.app import assemble

    ctx = assemble()
    ctx.act("accept_mission")
    ctx.act("set_second_surface", True)
    ctx.act("choose_section_for", "aft")
    assert ctx.S["ui"]["selected"] == session.SURFACE_STAGES["aft"]
    capsys.readouterr()


def test_the_second_surfaces_form_says_whose_surface_it_is(capsys):
    from gui.v3.app import assemble

    ctx = assemble()
    ctx.act("accept_mission")
    ctx.act("set_second_surface", True)
    ctx.render("airfoil_aft", "screen")
    ctx.render("airfoil", "screen")

    def texts(view):
        return [getattr(e, "text", "") or "" for e in view.descendants()]

    aft = texts(ctx.views[("airfoil_aft", "screen")])
    assert any("TAIL" == t for t in aft)
    assert any("flies the wing's section" in t for t in aft)
    # the recommendation, with its reason, on the form it applies to
    assert any("Recommended for this surface (trim)" in t for t in aft)
    # ...and no aspect-ratio field: that estimate is the WING's
    assert not any("aspect ratio estimate" in t for t in aft)

    main = texts(ctx.views[("airfoil", "screen")])
    assert any("WING" == t for t in main)
    assert any("aspect ratio estimate" in t for t in main)
    # "wing-trimmed", not "wing": this configuration HAS a tail, so the
    # wing's own |Cm| is the tail's problem and stops being a criterion
    assert any("Recommended for this surface (wing-trimmed)" in t
               for t in main)
    capsys.readouterr()


def test_a_new_session_keeps_the_dicts_the_stages_are_writing_into(capsys):
    """Every stage closes over its own sub-dict when it is built, so "New
    session" refills them rather than replacing them — swapping the objects
    left each stage writing where nothing else read."""
    from gui.v3 import app
    from gui.v3.app import assemble

    ctx = assemble()
    ctx.act("accept_mission")
    before = {k: id(v) for k, v in ctx.S.items() if isinstance(v, dict)}
    app._new_session(ctx)
    after = {k: id(v) for k, v in ctx.S.items() if isinstance(v, dict)}
    assert before == after
    assert not ctx.S["mission"]["accepted"]         # it really did reset
    capsys.readouterr()
