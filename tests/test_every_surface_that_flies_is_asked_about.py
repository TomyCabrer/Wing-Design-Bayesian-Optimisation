"""Three families flew a surface nobody could ask about, and one could not fly.

Reported as: "Tandem wing no vertical fin airfoil can be selected but appears,
no flight section can be selected. Hydrofoil no fin can be selected."

Measured, all three are the same defect in different places — a surface that
is drawn, exported and flown while the shell has no question for it:

1. THE TANDEM'S FIN. ``api.design_report`` has sized one against the pair's
   stagger since V5 (1.6 m2 at V_v 0.04) and ``flightmodel`` flew it — the
   whole of the pair's ``Cn_beta`` +0.1174. But no tandem spec declared the
   fin keys, so ``session.fin_surface`` said no, stage 2.7 was hidden behind
   "turn one on in stage 1", the stage-1 switch wrote a choice that no flag
   carried, and ``tandem.evaluate_tandem`` charged nothing for it.

2. THE TANDEM COULD NOT BE FLOWN. ``flightmodel`` built its elevator column
   only from a ``tail`` block, so a pair's deck came back as
   ``[alpha, beta, p, q, r, aileron, rudder]``: ``sixdof.trim_level`` had no
   pitch control to solve for and stage 6 refused to arm with "it carries one
   lifting surface" — of a two-surface aircraft, telling the user to add a
   tail it cannot have.

3. THE HYDROFOIL'S STRUT. ``hydrofoil._mast_cd0`` has charged a mast on every
   water run ever published (two sides of depth x mast chord). It was in no
   report, so the rebuild invented a THIRD vertical surface — 0.144 m at 12 %
   of span against a real 0.575 m mast — and hung every water yaw number on
   it, while stage 1 hid the question ("a water craft's vertical is the
   strut") and stage 2.7 refused it.

Every test here asserts an OUTCOME: a number the solver returns, a column the
deck carries, an aircraft that arms. They were all run against the previous
commit as well, where each fails.
"""

import numpy as np
import pytest

from aerobo import api, dynamics as dyn, fin as _fin
from aerobo.flightmodel import ControlsSpec, build_flight_model

TANDEM = "tandem"
PAIR_VLM = "tandem (nonplanar) + winglets"
FOIL = "hydrofoil"
FOIL_TAIL = "hydrofoil + elevator"


def _built(name, **flags):
    return api.PROBLEM_SPECS[name].build({}, dict(flags), None)


def _centre(name, **flags):
    built = _built(name, **flags)
    return built, np.asarray(built.bounds, dtype=float).mean(axis=1)


def _report(name, **flags):
    built, x = _centre(name, **flags)
    cfg = api.RunConfig(problem_name=name, budget=4, seed=0, flags=dict(flags))
    return api.design_report(cfg, x)


def _shell(medium="air", **choices):
    """A V4 shell whose family is the one asked for."""
    from gui.v3 import session as v3s
    from gui.v4.app import assemble

    ctx = assemble(medium)
    ctx.S["wing"]["choices"].update(choices)
    v3s.apply_choices(ctx.S)
    return ctx


# ----------------------------------------- 1. the tandem's fin is a surface

@pytest.mark.parametrize("name", [TANDEM, PAIR_VLM])
def test_a_tandem_pays_for_the_fin_it_reports(name):
    """Charged, reported and flown are the same surface or they are three."""
    built, x = _centre(name)
    bd = built.evaluate(x)
    assert bd["cd0_fin"] > 0.0, "the pair flies a fin it does not pay for"

    blk = (_report(name).get("geometry") or {}).get("fin")
    assert isinstance(blk, dict), "the pair reports no fin at all"
    # the DRAG was computed on the reported surface: same area, same chord,
    # at the station the fin actually stands at — BETWEEN the two wings
    arm, z_root = _fin.tandem_fin_station(bd["dx"], bd.get("dz", 0.0))
    charged = _fin.size_fin(b=bd["b"], S=bd["Sref"], l_t=arm,
                            tail_type="tandem")
    assert blk["S"] == pytest.approx(charged.S)
    assert blk["chord_m"] == pytest.approx(charged.chord)


@pytest.mark.parametrize("name", [TANDEM, PAIR_VLM])
def test_switching_the_pairs_fin_off_removes_all_three(name):
    """One answer, and it must reach the drag, the report and the deck."""
    built, x = _centre(name)
    on = built.evaluate(x)
    off = _built(name, fin=False).evaluate(x)

    assert off["cd0_fin"] == 0.0
    assert off["LoD"] > on["LoD"], "no fin, and yet the same drag"

    geom = (_report(name, fin=False).get("geometry") or {})
    assert "fin" in geom and geom["fin"] is None, \
        "a design that was asked and said no must SAY no — a missing key " \
        "means 'this family does not report one' and gets an invented fin"

    deck = build_flight_model(_report(name, fin=False), ControlsSpec(),
                              V=45.0, rho=1.225).deck
    assert deck.Cn_beta == pytest.approx(0.0, abs=1e-9)


def test_a_stated_fin_shape_moves_the_pairs_answer():
    """The three shape keys are the pair's to state, like every other
    family's — and a key that reaches nothing is the defect this closes."""
    built, x = _centre(TANDEM)
    base = built.evaluate(x)["cd0_fin"]
    bigger = _built(TANDEM, fin_volume_coeff=0.08).evaluate(x)["cd0_fin"]
    thicker = _built(TANDEM, fin_tc=0.14).evaluate(x)["cd0_fin"]
    assert bigger > base * 1.5      # twice the volume coefficient, twice the area
    assert thicker > base           # a thicker section has more form drag


def test_the_shell_offers_the_pairs_fin_a_section_stage():
    """Stage 2.7 exists for a tandem, and the thickness chosen there travels."""
    from gui.v3 import config as v3c, session as v3s

    ctx = _shell(system="tandem")
    assert "tandem" in ctx.S["wing"]["problem"]
    assert v3s.fin_surface(ctx.S) is True
    assert v3s.stage_visible(ctx.S, "airfoil_fin") is True
    assert "fin_tc" in v3c.flags(ctx.S), \
        "the section chosen on stage 2.7 never left the shell"


def test_the_card_quotes_the_fin_the_pair_is_actually_charged_for():
    """A pair's fin is sized on the pair: its TOTAL area, on the stagger.

    Read through the wing+tail's own names it came out at 0.727 m2 on a
    5.5 m arm — less than half the 1.6 m2 the run is charged for.
    """
    from gui.v3 import session as v3s

    ctx = _shell(system="tandem")
    card = v3s.surface_geometry(ctx.S, "fin")
    blk = (_report(TANDEM).get("geometry") or {})["fin"]
    assert card is not None
    assert card["area"] == pytest.approx(blk["S"], rel=1e-9)
    assert card["span"] == pytest.approx(abs(blk["height_m"]), rel=1e-9)


# ------------------------------------------- 2. the tandem can be flown now

def test_a_tandem_deck_has_a_pitch_control():
    deck = build_flight_model(_report(TANDEM), ControlsSpec(),
                              V=45.0, rho=1.225).deck
    assert "elevator" in deck.columns, \
        "a two-surface aircraft with no pitch column cannot be trimmed"
    assert abs(deck.columns["elevator"]["Cm"]) > 1.0


def test_the_pairs_roll_and_pitch_hinges_are_different_trailing_edges():
    """Roll on the FRONT wing, pitch on the REAR one.

    The aileron mask was ``~(is_tail | is_vertical)``, and a tandem's rear
    wing is neither — so the aileron band ran across both surfaces, over the
    same trailing edge the elevon is cut from.
    """
    m = build_flight_model(_report(TANDEM), ControlsSpec(),
                           V=45.0, rho=1.225).model
    rear = np.asarray(m.is_second, dtype=bool)
    assert rear.any(), "this rebuild has no second wing to hinge"
    ail = dyn.aileron(m).gain
    ele = dyn.elevator(m).gain
    assert not np.any(ail[rear]), "the aileron reaches the rear wing"
    assert np.any(ele[rear]), "the elevon does not reach the rear wing"
    assert not np.any(ele[~rear]), "the elevon reaches the front wing"


def test_stage_6_arms_a_tandem():
    """The user's report, as an outcome: the Flight stage can be flown."""
    ctx = _shell(system="tandem")
    ctx.S["run"]["record"] = {"pretend": "a completed run"}
    ctx.S["run"]["report"] = _report(TANDEM)
    ctx.render("controls", "derivatives")
    assert ctx.S["controls"].get("deck") is not None
    assert ctx.act("flight_arm") is True, ctx.S["flight"].get("error")


def test_a_genuinely_single_surface_still_refuses_and_is_the_only_one_that_does():
    """The refusal is not deleted, it is made true: a wing alone has nothing
    to trim it, and that is a different sentence from a tandem's."""
    ctx = _shell(tail=False, system="single")
    assert "wing" in ctx.S["wing"]["problem"]
    ctx.S["run"]["record"] = {"pretend": "a completed run"}
    ctx.S["run"]["report"] = _report("winglet")
    ctx.render("controls", "derivatives")
    assert ctx.act("flight_arm") is False
    assert "one lifting surface" in (ctx.S["flight"].get("error") or "")


# --------------------------------------------- 3. the water craft's strut

@pytest.mark.parametrize("name", [FOIL, FOIL_TAIL])
def test_a_water_craft_reports_the_mast_it_is_charged_for(name):
    built, x = _centre(name)
    bd = built.evaluate(x)
    blk = (_report(name).get("geometry") or {}).get("fin")
    assert isinstance(blk, dict), "the craft reports no vertical surface"
    assert blk["kind"] == "mast"
    # the surface REPORTED is the surface CHARGED: _mast_cd0's own wetted area
    assert blk["height_m"] == pytest.approx(bd["depth"])
    assert blk["chord_m"] == pytest.approx(built.problem.c_mast)
    assert blk["S"] == pytest.approx(bd["depth"] * built.problem.c_mast)


def test_the_rebuild_flies_that_mast_instead_of_inventing_a_fin():
    rep = _report(FOIL_TAIL)
    fm = build_flight_model(rep, ControlsSpec(), V=12.0, rho=1025.0)
    v = fm.model.vertical
    blk = rep["geometry"]["fin"]
    assert v is not None
    assert abs(float(v.height)) == pytest.approx(blk["height_m"], rel=1e-9)
    assert float(v.chord) == pytest.approx(blk["chord_m"], rel=1e-9)
    # ...and it is not the 12 %-of-span fin the rebuild used to assume
    assert abs(float(v.height)) > 0.12 * float(rep["geometry"]["b"]) * 1.5
    assert any("MAST" in a for a in fm.assumptions)


def test_the_strut_can_be_switched_off_and_the_run_stops_paying_for_it():
    built, x = _centre(FOIL)
    on = built.evaluate(x)
    off = _built(FOIL, fin=False).evaluate(x)
    assert on["cd0_mast"] > 0.0 and off["cd0_mast"] == 0.0
    assert off["LoD"] > on["LoD"]
    geom = _report(FOIL, fin=False).get("geometry") or {}
    assert "fin" in geom and geom["fin"] is None


def test_the_shell_asks_the_water_craft_about_its_strut():
    from gui.v3 import config as v3c, session as v3s

    ctx = _shell("water")
    assert v3s.fin_surface(ctx.S) is True
    assert v3s.stage_visible(ctx.S, "airfoil_fin") is True
    assert "fin_tc" in v3c.flags(ctx.S)
    card = v3s.surface_geometry(ctx.S, "fin")
    assert card is not None and card["mac"] == pytest.approx(0.08)


def test_a_family_with_no_vertical_surface_is_still_offered_none():
    """The fix is not "everything gets a fin": a plain wing has none, and the
    switch that reaches nothing must not be on screen."""
    from gui.v3 import config as v3c, session as v3s

    ctx = _shell(tail=False, system="single")
    spec = api.PROBLEM_SPECS[ctx.S["wing"]["problem"]]
    assert not any(k in spec.flags
                   for k in (*api.FIN_SHAPE_KEYS, api.FIN_PRESENCE_KEY))
    assert v3s.fin_surface(ctx.S) is False
    assert v3s.stage_visible(ctx.S, "airfoil_fin") is False
    assert v3s.surface_geometry(ctx.S, "fin") is None, \
        "a fin was quoted for a family that will never build one"
    assert not any("fin" in k for k in v3c.flags(ctx.S))
