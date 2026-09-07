"""The load the mission states is the load the run flies.

The user's report, in full: *"RECOMENDED DOES NOT GIVE A SOLUTION in 10H fix
it (this is for 5N and 0.8 span mission)"* — a 5 N craft on a 0.8 m foil at
10 m/s, whose design box came back with no admissible design at all.

Nothing was wrong with the box. The mission never left stage 1. Every water
family carries its design lift as a stated VALUE (``api.WEIGHT_KEY`` ->
``hydrofoil.L_design``) and declares no ``W_N`` mission field, and
``config.flags`` sent neither — so the shell took "5 N", stage 2 designed a
section for CL = W/(qS) off it, and stage 3 flew the family's published
6000 N. Six kilonewtons on a foil sized for five cavitates everywhere:

    before   0 of 128 draws admissible   (recommend_box -> empty)
    after   46 of 128 draws admissible

which the box card could only report as "no solution". The medium's own
blurb had been promising the opposite all along — *"the mission below sets
the section design point and the craft weight"* (``session.MEDIA``).
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

#: the mission that was reported, exactly
REPORTED = {"W_N": 5.0, "V": 10.0, "span_m": 0.8, "ar": 8.0}


def _shell(medium="water", *, W_N=None, V=None, span=None, ar=8.0):
    from gui.v3 import session
    from gui.v3.app import assemble

    ctx = assemble(medium)
    S = ctx.S
    if W_N is not None:
        S["mission"]["W_N"] = float(W_N)
    if V is not None:
        S["mission"]["V"] = float(V)
    if span is not None:
        session.set_size_statement(S, session.SIZE_AS_SPAN_AR)
        assert session.set_size_from_span_ar(S, span=span, ar=ar)
    ctx.act("accept_mission")
    return ctx


def _reported_shell():
    ctx = _shell(W_N=REPORTED["W_N"], V=REPORTED["V"],
                 span=REPORTED["span_m"], ar=REPORTED["ar"])
    ctx.act("set_planform", "free")
    ctx.render("wing", "box")
    return ctx


# ------------------------------------------------------- 1. it travels
def test_the_water_family_flies_the_load_the_mission_states():
    """The stated load reaches the config as ``weight_n``. Asserted on the
    flags the RUN is built from, not on the mission dict: the number was
    always in the mission dict, and that was the whole defect."""
    from aerobo import api
    from gui.v3 import config

    ctx = _reported_shell()
    cfg = config.build_cfg(ctx.S)
    assert cfg.flags.get(api.WEIGHT_KEY) == REPORTED["W_N"]
    # ...and the family honours it: the built problem trims to 5 N, not 6000
    built, _stripped = api._recommendable_problem(cfg)
    assert built.problem.L_design == REPORTED["W_N"]


# ------------------------------------------------- 2. the report itself
def test_the_reported_mission_has_a_solution(capsys):
    """The measurement finds admissible designs, and it finds them BECAUSE
    of the load — the same config with the flag taken back out is the empty
    box the user was looking at."""
    import copy

    from aerobo import api
    from gui.v3 import config

    ctx = _reported_shell()
    d = config.cfg_dict(ctx.S)
    got = api.recommend_box(api.RunConfig(**d), n=64)

    without = copy.deepcopy(d)
    without["flags"] = {k: v for k, v in (without["flags"] or {}).items()
                        if k != api.WEIGHT_KEY}
    was = api.recommend_box(api.RunConfig(**without), n=64)

    with capsys.disabled():
        print(f"\n  5 N stated  : empty={got['empty']} "
              f"admissible={got['n_admissible']}/{got['n']}")
        print(f"  6 kN flown  : empty={was['empty']} "
              f"admissible={was['n_admissible']}/{was['n']}")
    assert was["empty"], "the defect no longer reproduces — check the harness"
    assert not got["empty"]
    assert got["n_admissible"] > 0
    assert got["rows"]


# --------------------------------------------- 3. the published run moves
def test_an_untouched_water_session_sends_no_weight():
    """The mission form OPENS on the family's own design lift
    (``session.mission_defaults``), so an untouched session must send
    nothing and reproduce the published run bit-for-bit."""
    from aerobo import api
    from gui.v3 import config

    ctx = _shell()
    assert api.WEIGHT_KEY not in config.flags(ctx.S)
    # the same session with one newton typed into it does send one
    ctx.S["mission"]["W_N"] = float(ctx.S["mission"]["W_N"]) + 1.0
    assert config.flags(ctx.S)[api.WEIGHT_KEY] == ctx.S["mission"]["W_N"]


# ------------------------------------------------ 4. the silent fallback
def test_a_load_that_cannot_be_trimmed_to_is_not_sent_silently():
    """``api._weight_kwargs`` refuses a non-positive load by RAISING out of
    the build, and ``config.flags`` is on every repaint's path — so the flag
    is withheld (the family's published lift is flown) and the card is told
    to say so. A fallback nobody is told about is the failure this repo
    keeps re-finding."""
    from aerobo import api
    from gui.v3 import config

    ctx = _shell()
    for bad in (0.0, -5.0, float("nan"), float("inf")):
        ctx.S["mission"]["W_N"] = bad
        assert api.WEIGHT_KEY not in config.flags(ctx.S), bad
        # ...and the build the Run button makes still works
        config.build_cfg(ctx.S)
        note = config.stated_load_note(ctx.S)
        assert note and "6000 N" in note, (bad, note)
    ctx.S["mission"]["W_N"] = 5.0
    assert config.stated_load_note(ctx.S) is None


# ------------------------------------------------- 5. and where it is not
def test_the_car_wing_says_it_does_not_fly_the_load():
    """The car maximises downforce under a drag budget: it has no lift
    target at all, and the mission's load reaches its section stage and
    nothing further."""
    from gui.v3 import config

    ctx = _shell("track")
    note = config.stated_load_note(ctx.S)
    assert note and "no lift target" in note
    assert "stage 2" in note


def test_an_air_wing_carries_the_load_as_a_mission_field():
    """Air was never broken — its families declare ``W_N`` as a mission
    field — and this fix must not give it a second channel."""
    from aerobo import api
    from gui.v3 import config

    ctx = _shell("air", W_N=400.0)
    assert config.stated_load_note(ctx.S) is None
    assert api.WEIGHT_KEY not in config.flags(ctx.S)
    assert config.mission_kwargs(ctx.S)["W_N"] == 400.0


# ------------------------------------------------------- 6. on the card
def test_the_track_states_the_reason_and_asks_for_no_load_at_all():
    """The warning belongs under the field it is about — and on the TRACK
    there is no longer such a field.

    This used to assert "no lift target" on the track's own card, and it was
    right when the track card asked an aircraft's six mission numbers. It
    does not any more: a rear wing has no weight and no altitude to back a
    lift coefficient out of, so stage 1 asks it two numbers (the speed and
    the design CZ) and the design-load field is gone with the other four.
    A note has to sit under the field it contradicts, and there is nothing
    left here to contradict.

    So the claim moves to where it still holds: ``stated_load_note`` names
    the reason, and the card draws no load line because it asks no load.
    """
    from gui.v3 import config

    ctx = _shell("track")
    note = config.stated_load_note(ctx.S)
    assert note and "no lift target" in note, note

    ctx.render("mission", "operating")
    view = ctx.views[("mission", "operating")]
    text = " ".join(getattr(e, "text", None) or ""
                    for e in view.descendants())
    assert "failed to render" not in text
    # the field is gone, so the note under it is too — and neither is a
    # silent drop: test_car_operating_point_and_mount owns the assertion
    # that this card asks the two numbers a car actually reads
    labels = [str(getattr(e, "text", "") or "").strip().lower()
              for e in view.descendants()]
    assert not [t for t in labels if "design weight" in t or "design load" in t]
    assert "no lift target" not in text


def test_a_load_that_cannot_be_trimmed_to_is_named():
    """The other half of the note, on a family that DOES type a load.

    A water family carries its design lift as a stated value, and a
    non-positive one cannot be trimmed to — ``api._weight_kwargs`` refuses it
    out of the build — so the run flies the family's published lift and the
    user has to be told which.

    NOT asserted on the card, and that is a finding rather than an omission:
    rendering the operating card with a non-positive load raises
    ``KeyError: 'v_ms'`` out of ``session.ws_diagram`` (through
    ``mission_ws_ceiling``), which the shell turns into "this view failed to
    render" — the whole card, not one line. That predates this work (it
    reproduces at f542158) and belongs to the constraint diagram, so it is
    recorded here rather than worked around.
    """
    from gui.v3 import config
    from gui.v3.app import assemble

    ctx = assemble("water")
    ctx.S["mission"]["W_N"] = 0.0
    ctx.act("accept_mission")
    note = config.stated_load_note(ctx.S)
    assert note and "positive number of newtons" in note, note
    assert "flies this family's" in note
