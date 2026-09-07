"""V3: the section a surface carries is flown at the point it was chosen at,
and the shell says which point that is.

The stage-2 toggle offered "the mission's own Re", and choosing it changed
nothing twice over: the screen returned the cached Re-1e6 records (the
checkpoint had no operating point in its key) and the run flew the cached
Re-1e6 polar (the flag carried only a NAME, and no builder passed a Reynolds
number). Both ends are gated here. Of the two places the shell used to speak, only
stage 2 is left: the wing stage's "Section carried from stage 2" card was
DELETED at the user's request (it re-asked stage 2's question in stage 3's
words), and with it went the flown-point readout and the "screen this
surface at its own Re" button. The decision it made survives as an action —
``fix_section_point`` is still registered, still tested below, and is now
the only way to reach it — so what these tests can still gate is the state
change, not a sentence on screen.
"""

from __future__ import annotations

import sys
import time
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


def _texts(box):
    return [getattr(e, "text", "") or "" for e in box.descendants()]


def _library_re():
    from gui.v3 import session as ses

    pt = ses.library_point()
    if pt is None:
        pytest.skip("no screening cache on this machine")
    return float(pt["re"])


# ================================================ what travels to the run
def test_a_pick_at_the_library_point_still_travels_as_a_bare_name():
    """A pick made AT the cache is byte-for-byte what it was: the sidecar
    already holds that sweep, so there is nothing to say."""
    from gui.v3 import session as ses

    S = _session("air")
    lib = _library_re()
    ses.set_section(S, {"name": "hg40", "source": "library",
                        "conditions": {"re": lib, "mach": 0.0}}, "library")
    assert ses.section_flag_value(S, "main") == "hg40"
    assert ses.section_point(ses.section_of(S)) is None
    assert ses.section_flown_re(S) == pytest.approx(lib)


def test_the_shell_opens_on_the_surfaces_own_reynolds_number():
    """THE DEFAULT IS THE POINT THE SURFACE FLIES.

    The cached point does not merely rank elsewhere — a library polar is
    measured data at ONE Reynolds number, so it is FLOWN there too, and the
    shipped default used to hand every wing a section chosen and flown at
    Re 1e6 whatever its chord was. Both surfaces open on their own point, and
    an absent key means the same thing as the shipped one.
    """
    from gui.v3 import session as ses

    assert ses.RE_SOURCE_DEFAULT == "mission"
    S = _session("air", tail=True)
    for surface in ("main", "aft"):
        state = ses.airfoil_state(S, surface)
        assert state["re_source"] == "mission"
        own = ses.surface_design_point(S, surface)["re_mac"]
        assert ses.section_conditions(S, surface)["re"] == pytest.approx(own)
        del state["re_source"]           # a workspace that never stated one
        assert ses.section_conditions(S, surface)["re"] == pytest.approx(own)

    # ...and the cached point is still reachable, still pinned to ITS Re
    lib = _library_re()
    ses.airfoil_state(S, "main")["re_source"] = "library"
    assert ses.section_conditions(S, "main")["re"] == pytest.approx(lib)


def test_a_pick_at_another_point_carries_that_point():
    """...and one screened anywhere else must say WHERE, or the run rebuilds
    its polar at the cached Reynolds number and flies a section that does not
    exist at the one the surface actually flies."""
    from gui.v3 import session as ses

    S = _session("air")
    _library_re()
    ses.set_section(S, {"name": "hg40", "source": "library",
                        "conditions": {"re": 3.0e5, "mach": 0.0}}, "library")
    value = ses.section_flag_value(S, "main")
    assert isinstance(value, dict)
    assert value["name"] == "hg40" and value["re"] == pytest.approx(3.0e5)
    assert ses.section_flown_re(S) == pytest.approx(3.0e5)


def test_the_flag_reaches_the_run_config_with_its_point():
    """The whole chain, not just the helper: what config.flags() hands
    api.RunConfig is what the solver builds the polar from."""
    from aerobo import api
    from gui.v3 import config
    from gui.v3 import session as ses

    S = _session("air")
    _library_re()
    ses.set_section(S, {"name": "hg40", "source": "library",
                        "conditions": {"re": 2.5e5, "mach": 0.0}}, "library")
    if api.SECTION_KEY not in api.PROBLEM_SPECS[S["wing"]["problem"]].flags:
        pytest.skip("this family carries no chosen-section flag")
    flag = config.flags(S)[api.SECTION_KEY]
    assert flag["re"] == pytest.approx(2.5e5)
    assert config.flies_chosen_section(S, "main")


def test_adopting_a_row_records_the_point_the_row_was_screened_at(capsys):
    """Through the REAL handler, not a hand-written section dict: the row the
    user clicks becomes a section flown at the point its ranking was produced
    at, and that is the only thing tying the two ends of the fix together."""
    from gui.v3 import session as ses
    from gui.v3.app import assemble

    ctx = assemble()
    ctx.act("accept_mission")
    S = ctx.S
    S["airfoil"]["screen"]["report"] = {
        "conditions": {"re": 2.2e5, "mach": 0.0, "cl_design": 0.5},
        "n_eligible": 2, "n_screened": 2, "wall_time_s": 9.0,
        "ranked": [{"name": "hg40", "tc": 0.15, "ldcr": 44.0,
                    "clmax": 1.4, "cm_at": -0.02},
                   {"name": "hg41", "tc": 0.15, "ldcr": 43.0,
                    "clmax": 1.4, "cm_at": -0.02}]}
    ctx.act("adopt_section_row", 0)
    err = capsys.readouterr().err
    assert "Traceback" not in err, err

    sec = ses.section_of(S, "main")
    assert sec["name"] == "hg40"
    assert sec["conditions"]["re"] == pytest.approx(2.2e5)
    assert ses.section_flown_re(S) == pytest.approx(2.2e5)
    assert ses.section_flag_value(S, "main")["re"] == pytest.approx(2.2e5)


def test_a_designed_section_keeps_carrying_its_own_point():
    """The CST path already travelled with a point; adding the library one
    must not disturb it."""
    from gui.v3 import session as ses

    S = _session("air")
    ses.set_section(S, {"name": "CST section (optimised)", "source": "cst",
                        "w_upper": [0.2] * 4, "w_lower": [-0.1] * 4,
                        "conditions": {"re": 4.0e5, "mach": 0.0}}, "cst")
    value = ses.section_flag_value(S, "main")
    assert value["re"] == pytest.approx(4.0e5)
    assert value["w_upper"] and "name" in value
    assert ses.section_flown_re(S) == pytest.approx(4.0e5)


# ============================================== the disagreement, detected
def test_a_section_flown_off_its_surfaces_point_is_reported():
    from gui.v3 import session as ses

    S = _session("air")
    lib = _library_re()
    ses.set_section(S, {"name": "hg40", "source": "library",
                        "conditions": {"re": lib, "mach": 0.0}}, "library")
    own = ses.surface_design_point(S, "main")["re_mac"]
    off = ses.section_flies_off_point(S, "main")
    if abs(own - lib) <= ses.POINT_MOVE_TOL * lib:
        assert off is None                     # they agree: nothing to say
    else:
        flown, surface_re = off
        assert flown == pytest.approx(lib)
        assert surface_re == pytest.approx(own)

    # a section screened AT the surface's own point never disagrees
    ses.set_section(S, {"name": "hg40", "source": "library",
                        "conditions": {"re": own, "mach": 0.0}}, "library")
    assert ses.section_flies_off_point(S, "main") is None


# ==================================================== what the shell says
def test_the_flown_point_is_still_readable_after_the_card_went(capsys):
    """THE READOUT IS GONE; THE NUMBER BEHIND IT MUST NOT BE.

    Stage 3 used to print "wing flies it at Re 3.000e+05" in the "Section
    carried from stage 2" card. The card was deleted, so nothing on screen
    says it any more — that is the accepted cost of the deletion. What may
    NOT rot is ``session.section_flown_re``: the run still flies the polar
    at the point the section travels with, ``fix_section_point`` still
    re-points a surface at its own, and both read this. A silent wrong
    answer here is a section flown a decade away from where it was screened
    with nothing anywhere to catch it.
    """
    from aerobo import api
    from gui.v3 import session as ses
    from gui.v3.app import assemble

    ctx = assemble()
    ctx.act("accept_mission")
    S = ctx.S
    if api.SECTION_KEY not in api.PROBLEM_SPECS[S["wing"]["problem"]].flags:
        pytest.skip("this family carries no chosen-section flag")
    _library_re()
    ses.set_section(S, {"name": "hg40", "source": "library",
                        "conditions": {"re": 3.0e5, "mach": 0.0}}, "library")
    assert ses.section_flown_re(S, "main") == pytest.approx(3.0e5)
    ctx.render("wing", "type")
    err = capsys.readouterr().err
    assert "Traceback" not in err, err
    # ...and the deleted card really is deleted, in both its sentences
    texts = _texts(ctx.views[("wing", "type")])
    assert not any("flies it at" in t for t in texts), texts
    assert not any("Section carried" in t for t in texts), texts


def test_the_card_offers_to_make_the_two_points_the_same(capsys):
    """The user's fix: point stage 2 at this surface's own Reynolds number
    and go there. It must NOT start the sweep by itself — that costs real
    XFOIL time and belongs to the stage that asks for it.

    Driven through the action registry, which is now the ONLY caller: the
    button that used to invoke it lived in the deleted "Section carried from
    stage 2" card. The registration is therefore what keeps the fix
    reachable at all, and this test is what keeps the registration.
    """
    from aerobo import api
    from gui.v3 import session as ses
    from gui.v3.app import assemble

    ctx = assemble()
    ctx.act("accept_mission")
    S = ctx.S
    if api.SECTION_KEY not in api.PROBLEM_SPECS[S["wing"]["problem"]].flags:
        pytest.skip("this family carries no chosen-section flag")
    lib = _library_re()
    own = ses.surface_design_point(S, "main")["re_mac"]
    if abs(own - lib) <= ses.POINT_MOVE_TOL * lib:
        S["mission"]["V"] = float(S["mission"]["V"]) * 4.0   # move the point
        own = ses.surface_design_point(S, "main")["re_mac"]
    ses.set_section(S, {"name": "hg40", "source": "library",
                        "conditions": {"re": lib, "mach": 0.0}}, "library")
    # the user who has something to FIX is the one who screened at the cache
    S["airfoil"]["re_source"] = "library"

    # the disagreement is real — it is simply no longer said on screen
    assert ses.section_flies_off_point(S, "main") is not None
    ctx.render("wing", "type")
    texts = _texts(ctx.views[("wing", "type")])
    assert not any("screen the wing at Re" in t for t in texts), texts

    # the handler, through the shell's action registry
    assert S["airfoil"]["re_source"] == "library"
    ctx.act("fix_section_point", "main")
    err = capsys.readouterr().err
    assert "Traceback" not in err, err
    assert S["airfoil"]["re_source"] == "mission"
    assert not S["airfoil"]["override_point"]
    assert S["ui"]["selected"] == "airfoil"
    # nothing was screened by the button itself
    assert S["airfoil"]["screen"]["report"] is None
    assert not S["airfoil"]["screen"]["running"]


def test_the_fix_arms_a_priced_rescreen_on_the_stage_that_can_price_it(capsys):
    """The other half of the loop: the decision was taken on stage 3, so the
    user should not arrive on stage 2 holding an instruction.

    The request travels with the re-point and lands as a primed button that
    states what the sweep costs. It still does not start XFOIL — the price is
    shown where it is known, and the click that pays it is the user's."""
    from aerobo import api
    from gui.v3 import session as ses
    from gui.v3.app import assemble

    ctx = assemble()
    ctx.act("accept_mission")
    S = ctx.S
    if api.SECTION_KEY not in api.PROBLEM_SPECS[S["wing"]["problem"]].flags:
        pytest.skip("this family carries no chosen-section flag")
    lib = _library_re()
    own = ses.surface_design_point(S, "main")["re_mac"]
    if abs(own - lib) <= ses.POINT_MOVE_TOL * lib:
        S["mission"]["V"] = float(S["mission"]["V"]) * 4.0
    ses.set_section(S, {"name": "hg40", "source": "library",
                        "conditions": {"re": lib, "mach": 0.0}}, "library")

    assert ses.pending_rescreen(S, "main") is None
    ctx.act("fix_section_point", "main")
    req = ses.pending_rescreen(S, "main")
    assert req and "Reynolds number" in req["reason"]

    ctx.render("airfoil", "screen")
    err = capsys.readouterr().err
    assert "Traceback" not in err, err
    texts = _texts(ctx.views[("airfoil", "screen")])
    assert any("re-screen is waiting" in t.lower() for t in texts), texts
    # the PRICE is stated, not just the offer
    assert any("swept for real" in t or "already cached" in t
               for t in texts), texts
    # and still nothing ran
    assert S["airfoil"]["screen"]["report"] is None
    assert not S["airfoil"]["screen"]["running"]


def test_the_offer_carries_a_diagnosis_not_just_a_detection(capsys):
    """"These two Reynolds numbers disagree" makes the user price the
    disagreement themselves, in the one quantity they came here for the tool
    to know. The banner states what the section they already picked is worth
    where the surface actually flies."""
    from aerobo import api
    from gui.v3 import session as ses
    from gui.v3.app import assemble

    ctx = assemble()
    ctx.act("accept_mission")
    S = ctx.S
    if api.SECTION_KEY not in api.PROBLEM_SPECS[S["wing"]["problem"]].flags:
        pytest.skip("this family carries no chosen-section flag")
    lib = _library_re()
    if abs(ses.surface_design_point(S, "main")["re_mac"] - lib) \
            <= ses.POINT_MOVE_TOL * lib:
        S["mission"]["V"] = float(S["mission"]["V"]) * 4.0
    ses.set_section(S, {"name": "hg40", "source": "library",
                        "conditions": {"re": lib, "mach": 0.0}}, "library")

    pen = ses.section_point_penalty(S, "main")
    if pen is None:
        pytest.skip("the surface's own polar is not in the cache")
    assert pen["name"] == "hg40"
    assert pen["ld_screened"] > 0 and pen["ld_own"] > 0
    # the two points really are different, so the penalty is not zero
    assert abs(pen["loss_pct"]) > 1.0
    assert pen["re_screened"] != pen["re_own"]

    ctx.act("fix_section_point", "main")
    ctx.render("airfoil", "screen")
    assert "Traceback" not in capsys.readouterr().err
    texts = _texts(ctx.views[("airfoil", "screen")])
    assert any("where it was ranked" in t for t in texts), texts


def test_the_diagnosis_never_starts_xfoil_and_never_raises():
    """It runs inside a render. A view that can block for minutes is worse
    than a view that says nothing, so it is cache-only and returns None on
    anything it cannot answer."""
    from gui.v3 import session as ses
    from gui.v3.app import assemble

    ctx = assemble()
    ctx.act("accept_mission")
    S = ctx.S
    # no chosen section at all: nothing to diagnose, and no exception
    assert ses.section_point_penalty(S, "main") is None
    assert ses.section_point_penalty(S, "aft") is None


def test_a_pending_rescreen_that_no_longer_fits_the_point_is_dropped():
    """A request naming a Reynolds number the surface has since left would
    offer to re-screen at the wrong point — the exact failure it exists to
    fix — so it reads as absent."""
    from gui.v3 import session as ses
    from gui.v3.app import assemble

    ctx = assemble()
    ctx.act("accept_mission")
    S = ctx.S
    now = float(ses.section_conditions(S, "main")["re"])
    ses.arm_rescreen(S, "main", reason="test", re=now)
    assert ses.pending_rescreen(S, "main") is not None

    ses.arm_rescreen(S, "main", reason="test", re=now * 3.0)
    assert ses.pending_rescreen(S, "main") is None
    assert S["airfoil"]["rescreen"] is None      # and it cleared itself

    # a request with no point attached is not point-dependent and survives
    ses.arm_rescreen(S, "main", reason="test", re=None)
    assert ses.pending_rescreen(S, "main") is not None
    ses.disarm_rescreen(S, "main")
    assert ses.pending_rescreen(S, "main") is None


def test_stage_two_offers_the_shortlist_only_where_it_costs_xfoil(capsys):
    """The shortlist control is the honest option's price tag, so it appears
    exactly when the screen would sweep: asking for the cached point by
    either route is still instant and must not show it."""
    from gui.v3 import session as ses
    from gui.v3.app import assemble

    ctx = assemble()
    ctx.act("accept_mission")
    S = ctx.S
    _library_re()
    S["airfoil"]["re_source"] = "library"        # the instant route
    ctx.render("airfoil", "screen")
    assert not any("XFOIL marches per section" in t for t in
                   _texts(ctx.views[("airfoil", "screen")]))

    S["airfoil"]["re_source"] = "mission"
    S["mission"]["V"] = float(S["mission"]["V"]) * 4.0    # off the cache
    ctx.render("airfoil", "screen")
    err = capsys.readouterr().err
    assert "Traceback" not in err, err
    texts = _texts(ctx.views[("airfoil", "screen")])
    assert any("XFOIL marches per section" in t for t in texts)
    assert any("cannot appear in the ranking" in t for t in texts)
    assert S["airfoil"]["shortlist"] == ses.SHORTLIST_DEFAULT


def test_stage_two_is_not_a_gate_and_stage_three_says_what_flies(capsys):
    """THE SKIP BUTTON IS GONE, SO THE STATEMENT IT MADE HAS TO BE ELSEWHERE.

    "Use the wing family's own section" declared what an unanswered stage 2
    already means, gated stage 3 behind a click, and (now that the default
    screen sweeps live XFOIL) would have priced the way through at minutes of
    sweeps nobody asked for. Removing it is only safe if nothing goes silent.

    It used to be the section card that spoke, naming the four different
    polar sources by family. That card is deleted, so what is left is the
    ``airfoil`` row of the configuration list: it reads back
    ``session.section_summary``, which says "the family's own section" while
    stage 2 is unanswered and the section's name once it is. Shorter than
    the sentence it replaces — it does not say WHICH polar — and that is the
    accepted cost. What must not happen is stage 3 showing a section that
    was never chosen, or gating on one.
    """
    from gui.v3 import session as ses
    from gui.v3.app import assemble

    ctx = assemble()
    ctx.act("accept_mission")
    S = ctx.S
    assert S["airfoil"]["decision"] is None
    assert "use_solver_section" not in ctx.actions
    assert ses.stage_states(S)["wing"][0] != "locked"

    assert ses.section_summary(S) == "the family's own section"
    ctx.render("wing", "type")
    texts = _texts(ctx.views[("wing", "type")])
    assert any("the family's own section" in t for t in texts), texts
    # ...and it is a statement, not a chooser: no section menu came back
    assert not any("No section was chosen" in t for t in texts), texts

    # the same row on a family that selects its section by thickness
    water = assemble("water")
    water.act("accept_mission")
    water.render("wing", "type")
    texts = _texts(water.views[("wing", "type")])
    assert any("the family's own section" in t for t in texts), texts

    # the statement becomes the NAME as soon as there IS a section
    ses.set_section(S, {"name": "hg40", "source": "library", "tc": 0.15},
                    "library")
    ctx.render("wing", "type")
    texts = _texts(ctx.views[("wing", "type")])
    assert not any("the family's own section" in t for t in texts), texts
    assert any("hg40" in t for t in texts), texts
    err = capsys.readouterr().err
    assert "Traceback" not in err, err


def test_the_sweep_reports_its_progression_section_by_section(capsys):
    """A SWEEP THAT SAYS NOTHING FOR MINUTES LOOKS LIKE A HANG.

    The shortlist re-screen is a live viscous XFOIL sweep per section, and
    what it reports has to distinguish the two passes: the library pass is a
    cache read with no count to give, and the sweep counts sections. Each
    finished section lands with its own numbers — or with the reason it will
    not appear in the ranking, which is progress too.
    """
    from gui.v3.app import assemble
    from gui.v3.stages import airfoil as airfoil_stage

    ctx = assemble()
    ctx.act("accept_mission")
    S = ctx.S
    sc = S["airfoil"]["screen"]

    # pass 1: running, nothing finished yet
    sc.update(running=True, phase="library", progress=None, swept=[])
    ctx.render("airfoil", "screen")
    texts = _texts(ctx.views[("airfoil", "screen")])
    assert any("no XFOIL yet" in t for t in texts)
    assert not any("sweeping the shortlist" in t for t in texts)

    # pass 2: the sweep, one row per finished section
    sc.update(phase="sweep", progress=(3, 24), swept=[
        {"i": 1, "name": "hg40", "status": "ok", "eligible": True,
         "ldcr": 44.25, "tc": 0.1503},
        {"i": 2, "name": "thin1", "status": "gate_tc", "eligible": False,
         "ldcr": None, "tc": 0.06},
        {"i": 3, "name": "bad2", "status": "unconverged", "eligible": False,
         "ldcr": None, "tc": None}])
    ctx.render("airfoil", "screen")
    err = capsys.readouterr().err
    assert "Traceback" not in err, err
    texts = _texts(ctx.views[("airfoil", "screen")])
    assert any("3/24" in t for t in texts)
    assert any("hg40" in t and "44.25" in t for t in texts)   # its own numbers
    assert any("gate_tc" in t for t in texts)                 # and the refusals
    assert any("unconverged" in t for t in texts)

    # the log is bounded on screen, and says how much it is not showing
    sc["swept"] = [{"i": k + 1, "name": f"s{k}", "status": "ok",
                    "eligible": True, "ldcr": 40.0 + k, "tc": 0.15}
                   for k in range(airfoil_stage.SWEEP_LOG_SHOWN + 5)]
    sc["progress"] = (len(sc["swept"]), 40)
    ctx.render("airfoil", "screen")
    texts = _texts(ctx.views[("airfoil", "screen")])
    assert any("5 earlier" in t for t in texts)
    assert not any(t.strip().startswith("1  s0") for t in texts)

    # ...and nothing of it survives the run that produced it
    sc.update(running=False, phase=None, progress=None, swept=[])
    ctx.render("airfoil", "screen")
    texts = _texts(ctx.views[("airfoil", "screen")])
    assert not any("sweeping the shortlist" in t for t in texts)
    capsys.readouterr()


def test_the_screen_worker_records_what_the_sweep_reports(capsys):
    """The plumbing, not the picture: api hands the callback
    ``(i, n, record)`` per finished section, and the FIRST call is also what
    says the library pass is over. Driven through the real worker with a
    stubbed api, so no XFOIL runs."""
    from aerobo import api
    from gui.v3.app import assemble

    ctx = assemble()
    ctx.act("accept_mission")
    S = ctx.S
    S["airfoil"]["re_source"] = "mission"
    S["mission"]["V"] = float(S["mission"]["V"]) * 4.0       # off the cache

    seen = {}

    def fake_screen_at_point(**kw):
        seen["shortlist"] = kw.get("shortlist")
        cb = kw["progress_cb"]
        cb(1, 2, {"name": "hg40", "status": "ok", "eligible": True,
                  "ldcr": 44.0, "tc": 0.15})
        cb(2, 2, {"name": "thin1", "status": "gate_tc", "eligible": False})
        return {"conditions": {"re": kw["re"], "cl_design": kw["cl_design"]},
                "ranked": [], "n_screened": 2, "n_eligible": 1,
                "wall_time_s": 1.0}

    old = api.screen_at_point
    api.screen_at_point = fake_screen_at_point
    try:
        ctx.act("run_airfoil")
        for _ in range(200):                     # the worker is a thread
            if not S["airfoil"]["screen"]["running"]:
                break
            time.sleep(0.02)
    finally:
        api.screen_at_point = old

    sc = S["airfoil"]["screen"]
    assert not sc["running"] and sc["error"] is None, sc["error"]
    assert seen["shortlist"] == S["airfoil"]["shortlist"]
    assert sc["phase"] == "sweep"                # the callback moved it on
    assert sc["progress"] == (2, 2)
    assert [r["name"] for r in sc["swept"]] == ["hg40", "thin1"]
    assert sc["swept"][0]["ldcr"] == 44.0 and sc["swept"][0]["eligible"]
    assert sc["swept"][1]["status"] == "gate_tc"
    capsys.readouterr()


def test_the_ranking_says_what_it_was_drawn_from(capsys):
    """A shortlist that reads as a whole-library ranking is a silent
    truncation — the table has to name its own population."""
    from gui.v3.app import assemble

    ctx = assemble()
    ctx.act("accept_mission")
    S = ctx.S
    S["airfoil"]["screen"]["report"] = {
        "conditions": {"re": 3.0e5, "cl_design": 0.5},
        "n_eligible": 3, "n_screened": 3, "wall_time_s": 12.0,
        "point": {"matched": False, "honoured": True, "excluded": 0},
        "shortlist": {"source": "library", "n": 3, "names": ["a", "b", "c"],
                      "note": "the 3 best of 161 eligible sections at the "
                              "cached library point were re-screened at "
                              "Re 3e+05"},
        "ranked": [{"name": "a", "tc": 0.12, "rank_library": 2},
                   {"name": "b", "tc": 0.13, "rank_library": 1},
                   {"name": "c", "tc": 0.14, "rank_library": 3}]}
    ctx.render("airfoil", "ranking")
    err = capsys.readouterr().err
    assert "Traceback" not in err, err
    texts = _texts(ctx.views[("airfoil", "ranking")])
    assert any("re-screened at" in t for t in texts)
    # the reordering column: a table's headings are Quasar props, not text
    tables = [e for e in ctx.views[("airfoil", "ranking")].descendants()
              if "columns" in getattr(e, "_props", {})]
    assert tables
    labels = [c["label"] for c in tables[0]._props["columns"]]
    assert "# @cache" in labels
    ranks = [r["rank_library"] for r in tables[0]._props["rows"]]
    assert ranks == ["2", "1", "3"]
