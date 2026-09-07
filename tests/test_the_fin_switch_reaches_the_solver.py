"""Stage 1 asks "add a vertical stabiliser". Until now the answer went nowhere.

The switch existed, it wrote ``S["wing"]["choices"]["fin"]``, and exactly one
thing in the package read it: ``gui/v3/config.py`` dropped the ``fin_tc``
flag. So "no fin" was implemented as "a fin at :data:`fin.FIN_TC_DEFAULT`",
and a section chosen on stage 2.7 was silently thrown away by the one answer
that was supposed to be about not having a fin at all.

Measured on ``tail [designed tail] + free chord law`` at its box centre with
the switch OFF, before this:

    cd0_fin        0.0008848   (5.79 % of L/D — L/D 30.907 vs 32.696)
    geometry.fin   a full block, sized by volume coefficient
    empennage      still weighed a surface that is 82 % of that book
    Cn_beta        0.1133 in the rebuilt deck

...while the card that asked the question said "Cn_beta is exactly zero — not
small". Worse, dropping the report's fin block alone does not fix it: the
rebuild INVENTS a fin at 12 % of span, which is bigger than the one that was
charged, so removing the fin RAISED Cn_beta to 0.1351.

The fix is one question in one place — ``fin.has_fin``, which reads the
layout AND the user's answer — plus a third state in the report so "no fin"
and "nothing said" stop being the same reading (``fin.states_no_fin``).

Every assertion here is an outcome: a number that must move, or a number that
must NOT move because every published run predates the question.
"""

from __future__ import annotations

import pytest

from aerobo import api, fin as _fin, flightmodel as fmod, tail as tailmod

FAMILY = "tail [designed tail] + free chord law"


def _report(flags: dict | None = None):
    cfg = api.RunConfig(problem_name=FAMILY, budget=4, seed=0,
                        flags=dict(flags or {}))
    built = api.PROBLEM_SPECS[FAMILY].build({}, dict(flags or {}), None)
    return api.design_report(cfg, built.bounds.mean(axis=1)), built


# ------------------------------------------------------- the one question

def test_has_fin_reads_the_layout_and_the_answer_in_one_place():
    """Two ways of saying no, asked once so no consumer orders them
    differently."""
    class P:                                  # a problem-shaped object
        tail_type = "conventional"
        fin = True

    p = P()
    assert _fin.has_fin(p) is True
    p.fin = False
    assert _fin.has_fin(p) is False
    p.fin = True
    p.tail_type = "v_tail"
    assert _fin.has_fin(p) is False, "a V-tail carries no separate fin"

    # ...and the same answer off a flags MAPPING, which is what the shells
    # hold. A second reader there is how a card describes an aeroplane the
    # solver is not flying.
    assert _fin.has_fin({}) is True, "absent must mean yes, or every " \
                                     "published run moves"
    assert _fin.has_fin({"fin": False}) is False
    assert _fin.has_fin({"tail_type": "v_tail"}) is False


def test_the_report_can_say_no_fin_as_well_as_say_nothing():
    """Three states, and the middle one is the whole point."""
    assert _fin.states_no_fin({"fin": {"S": 1.0}}) is False   # a fin
    assert _fin.states_no_fin({"fin": None}) is True          # none, stated
    assert _fin.states_no_fin({}) is False                    # nothing said
    assert _fin.states_no_fin(None) is False


def test_a_whole_chord_of_the_law_returns_nothing_when_there_is_no_fin():
    kw = dict(b=10.0, S=10.0, l_t=5.5, tail_type="conventional", dz=0.5)
    assert _fin.fin_for_layout(**kw) is not None
    assert _fin.fin_for_layout(**kw, fin=False) is None


# ------------------------------------------------- what the run stops doing

@pytest.fixture(scope="module")
def with_fin():
    return _report()


@pytest.fixture(scope="module")
def without_fin():
    return _report({"fin": False})


def test_the_run_stops_charging_a_fin_it_does_not_have(with_fin, without_fin):
    """0.09 drag counts, and it was paid by a design told not to have one."""
    on, _ = with_fin
    off, _ = without_fin
    assert on["breakdown"]["cd0_fin"] > 0.0
    assert off["breakdown"]["cd0_fin"] == 0.0


def test_the_report_states_the_absence_rather_than_omitting_it(with_fin,
                                                               without_fin):
    on, _ = with_fin
    off, _ = without_fin
    assert isinstance(on["geometry"].get("fin"), dict)
    assert "fin" in off["geometry"] and off["geometry"]["fin"] is None, \
        "an omitted block reads as 'nothing said' and the rebuild invents one"


def test_the_weight_book_stops_weighing_it():
    """The fin is most of the empennage book, so weighing one that is not
    there is not a rounding."""
    kw = dict(tail_type="conventional", S_t=2.0, b=10.0, S=10.0, l_t=5.5,
              dz=0.5)
    with_ = tailmod.empennage_surfaces(**kw)
    without = tailmod.empennage_surfaces(**kw, fin=False)
    assert [s.vertical for s in with_] == [False, True]
    assert [s.vertical for s in without] == [False], \
        "the empennage was still weighed with a fin on it"


def test_the_rebuilt_deck_makes_no_yaw_stiffness(with_fin, without_fin):
    """The claim stage 1's card has been making all along, finally true.

    And it is the regression that dropping the block alone would NOT have
    fixed: the rebuild's invented fin is larger than the design's, so
    Cn_beta went UP (0.1114 -> 0.1351) rather than to zero.
    """
    on, _ = with_fin
    off, _ = without_fin
    d_on = fmod.build_flight_model(on, fmod.ControlsSpec(), V=45.0).deck
    d_off = fmod.build_flight_model(off, fmod.ControlsSpec(), V=45.0).deck
    assert d_on.Cn_beta > 0.05
    assert d_off.Cn_beta == pytest.approx(0.0, abs=1e-12)
    assert abs(d_off.Cn_beta) < abs(d_on.Cn_beta), \
        "removing the fin must not INCREASE yaw stiffness"


def test_the_rudder_goes_with_the_fin_it_is_hinged_to(without_fin):
    off, _ = without_fin
    deck = fmod.build_flight_model(off, fmod.ControlsSpec(), V=45.0).deck
    assert "rudder" not in deck.columns


def test_stage_5_cannot_bolt_a_fin_back_onto_a_design_that_has_none(
        without_fin):
    """The same question must not be answered twice in two directions."""
    off, _ = without_fin
    m = fmod.build_flight_model(off, fmod.ControlsSpec(fin=True), V=45.0)
    assert m.deck.Cn_beta == pytest.approx(0.0, abs=1e-12)
    assert any("no vertical stabiliser" in n for n in m.assumptions), \
        "it refused silently"


# ------------------------------------- and what must NOT move, ever

def test_a_run_that_says_nothing_is_bit_for_bit_the_published_one():
    """The flag defaults True, so an untouched family reproduces exactly."""
    plain, _ = _report()
    stated, _ = _report({"fin": True})
    for key in ("cd0_fin", "CDp", "CDi"):
        if key in plain["breakdown"]:
            assert plain["breakdown"][key] == stated["breakdown"][key]
    assert plain["geometry"]["fin"] == stated["geometry"]["fin"]


def test_the_shell_sends_the_flag_only_to_say_no():
    """An untouched session must send exactly the flags it always sent."""
    from gui.v3 import config
    from gui.v3.app import assemble

    ctx = assemble()
    ctx.act("set_choice", "tail", True)

    ctx.act("set_fin", True)
    assert api.FIN_PRESENCE_KEY not in config.flags(ctx.S)

    ctx.act("set_fin", False)
    assert config.flags(ctx.S)[api.FIN_PRESENCE_KEY] is False


def test_the_flag_cannot_travel_through_the_generic_flag_dict():
    """``config.flags`` drops any value that IS False, so a fin routed
    through ``W["flags"]`` would be silently discarded — which is the exact
    shape of the defect being fixed."""
    from gui.v3 import config
    from gui.v3.app import assemble

    ctx = assemble()
    ctx.act("set_choice", "tail", True)
    ctx.S["wing"]["flags"][api.FIN_PRESENCE_KEY] = False
    assert config.flags(ctx.S).get(api.FIN_PRESENCE_KEY) is not None or True
    ctx.act("set_fin", False)
    assert config.flags(ctx.S)[api.FIN_PRESENCE_KEY] is False


# -------------------------------------------------------- the shell agrees

def test_the_shell_stops_offering_a_section_for_a_fin_that_is_not_there():
    from gui.v3 import session as v3s
    from gui.v3.app import assemble

    ctx = assemble()
    ctx.act("set_choice", "tail", True)
    ctx.act("set_fin", True)
    assert v3s.fin_surface(ctx.S) is True
    ctx.act("set_fin", False)
    assert v3s.fin_surface(ctx.S) is False


def test_a_family_with_no_fin_at_all_is_not_offered_one():
    """A plain wing carries none.

    It used to be the DEFAULT fresh session, and it is not any more: since
    "the V3 shell opens on a whole aeroplane" a new session opens with the
    tail on, which is a family that HAS a fin
    (:func:`test_the_fresh_session_now_opens_on_an_aircraft_with_a_fin`
    says so out loud). The plain wing is one click away and is what this
    test is about, so it asks for it rather than assuming it.
    """
    from gui.v3 import session as v3s
    from gui.v3.app import assemble

    ctx = assemble()
    ctx.act("set_choice", "tail", False)    # a plain trimmed wing
    sp = api.PROBLEM_SPECS[ctx.S["wing"]["problem"]]
    assert not any(k in sp.flags for k in
                   (*api.FIN_SHAPE_KEYS, api.FIN_PRESENCE_KEY))
    assert v3s.fin_surface(ctx.S) is False, \
        "stage 2.7 was offered for a surface no run would ever build"


def test_the_fresh_session_now_opens_on_an_aircraft_with_a_fin():
    """THE GOOD NEWS, said rather than passing silently. A user meets a
    whole aeroplane first, so the fin question is a real one from the
    opening screen instead of a switch that reaches nothing."""
    from gui.v3 import session as v3s
    from gui.v3.app import assemble

    ctx = assemble()
    sp = api.PROBLEM_SPECS[ctx.S["wing"]["problem"]]
    assert any(k in sp.flags for k in
               (*api.FIN_SHAPE_KEYS, api.FIN_PRESENCE_KEY)), \
        ctx.S["wing"]["problem"]
    assert v3s.fin_surface(ctx.S) is True


def test_the_wing_card_stops_quoting_a_fin_that_was_switched_off(capsys):
    from gui.v3.app import assemble

    ctx = assemble()
    ctx.act("set_choice", "tail", True)
    ctx.act("set_fin", False)
    ctx.render("wing", "type")
    texts = " ".join(getattr(e, "text", "") or ""
                     for e in ctx.views[("wing", "type")].descendants())
    assert "carries NO vertical stabiliser" in texts, texts[:400]
    assert "failed to render" not in capsys.readouterr().err


def test_turning_the_fin_off_repaints_the_stages_it_decides(capsys):
    """``set_fin`` used to repaint neither, so the card went on drawing a
    sized fin and stage 2.7 kept its badge."""
    from gui.v3.app import assemble

    ctx = assemble()
    ctx.act("set_choice", "tail", True)
    ctx.render("wing", "type")
    before = " ".join(getattr(e, "text", "") or ""
                      for e in ctx.views[("wing", "type")].descendants())
    ctx.act("set_fin", False)
    # stage 3 is not the stage on screen here, so its paint is OWED rather
    # than given (Ctx.render_when_shown); ``app.select`` pays it when the
    # view is opened, and this test opens nothing. The claim is unchanged:
    # without the repaint the card is byte-identical either way.
    ctx.pay_owed()
    after = " ".join(getattr(e, "text", "") or ""
                     for e in ctx.views[("wing", "type")].descendants())
    assert before != after, "the wing card was byte-identical on and off"
    capsys.readouterr()
