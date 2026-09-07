"""HOW MUCH dihedral — measured on the configuration on screen, not quoted.

The package could already say a divergent spiral is a WING answer and not a
fin one: stage 5 rebuilds ten fin heights over 1-30 % of the span and every
one of them leaves the margin negative. What it could not say was HOW MUCH
wing. Both shells quoted ONE constant, 5.7 deg, labelled "enough" — and that
is the box centre of ``tail [free cant]``, the family the study was run on.

Measured with :func:`aerobo.flightmodel.dihedral_for_spiral` over the
families this shell actually derives (RESULTS_SESSION69_SPIRAL_RECOMMENDATION
.md), the crossing is not one number:

    family                   recommendation   6-DOF spiral root at 5.7 deg
    tail + winglet                3.07 deg          -0.053   converges
    tail [free cant]              5.43 deg          -0.006   converges
    tail [free height]            5.57 deg          -0.004   converges
    tail                          6.13 deg          +0.0026  DIVERGES

The last row is the shell's own default wing+tail family (``tail + free
chord law`` derives the same aeroplane): press "enough" there and the
aeroplane still rolls off with the stick free. A step that does not clear
the gate it cites is the defect these tests close, and the first test below
is the one that fails if the shells go back to quoting a constant.
"""
from __future__ import annotations

import sys

import numpy as np
import pytest

from aerobo import api
from aerobo.flightmodel import (SPIRAL_DIHEDRAL_MAX_DEG, build_flight_model,
                                dihedral_for_spiral, _report_at_dihedral)

sys.path.insert(0, "gui")

#: the shell's DEFAULT wing+tail family (nice_app.BUILDER_START with the tail
#: switched on derives ``tail + free chord law``, which flies this same
#: aeroplane), and the constant both shells used to offer at it
DEFAULT_FAMILY = "tail"
SHIPPED_CONSTANT_DEG = 5.7


def _report(name: str, flags: dict | None = None) -> dict:
    built = api.PROBLEM_SPECS[name].build({}, flags or {}, None)
    cfg = api.RunConfig(problem_name=name, budget=4, seed=0, flags=flags or {})
    return api.design_report(
        cfg, np.asarray(built.bounds, dtype=float).mean(axis=1))


def _texts(view) -> list:
    """Every string a rendered view shows.

    ``getattr(e, "text", "")`` is not enough on its own: a few nicegui
    element CLASSES carry a ``text`` descriptor, so the attribute comes back
    as a type on those and every ``in`` test against it raises.
    """
    out = []
    for e in view.descendants():
        t = getattr(e, "text", "")
        if isinstance(t, str) and t:
            out.append(t)
    return out


def _spiral_root(report: dict, gamma_deg: float) -> float:
    """The 6-DOF spiral eigenvalue of the rebuild at ``gamma_deg``.

    The criterion is a proxy; this is the mode itself — a full nonlinear
    Jacobian about a trimmed state, classified by eigenvector participation.
    Positive is an aeroplane that rolls off.
    """
    from aerobo import sixdof as sd
    from v4 import modes as md

    fm = build_flight_model(_report_at_dihedral(report, gamma_deg))
    st, _ = sd.trim_level(fm.aircraft, V=float(fm.deck.V), altitude_m=0.0)
    return float(md.classify(sd.linearise(fm.aircraft, st),
                             float(st.V))["spiral"].real)


# ---------------------------------------------------------- the instrument

@pytest.mark.parametrize("name", ["tail", "tail + winglet",
                                  "tail [free height]"])
def test_the_recommendation_clears_the_gate_it_cites(name):
    """At the angle it offers, the MODE is stable — and half a degree less
    is not, so it is the smallest answer and not a padded one."""
    rep = _report(name)
    fix = dihedral_for_spiral(rep)
    assert fix.status == "found", fix
    assert fix.margin_at is not None and fix.margin_at >= 0.0
    assert _spiral_root(rep, fix.gamma_deg) < 0.0, (
        f"{name}: the criterion says {fix.gamma_deg:.3f} deg converges the "
        f"spiral and the eigenvalue disagrees — the proxy has drifted off "
        f"the mode it is a proxy for")
    assert _spiral_root(rep, fix.gamma_deg - 0.5) > 0.0, (
        f"{name}: half a degree less also converges, so the answer is not "
        f"the smallest one")


def test_the_shipped_CONSTANT_leaves_the_default_family_rolling_off():
    """THE DEFECT, as a tripwire.

    5.7 deg is ``tail [free cant]``'s crossing and it was offered at every
    configuration. On the family the shell derives first it is not enough:
    the spiral root is still positive there, so the button labelled "enough"
    handed back an aeroplane that rolls off. If this ever passes because the
    two numbers have met, the recommendation may go back to being a
    constant — until then it may not.
    """
    rep = _report(DEFAULT_FAMILY)
    assert _spiral_root(rep, SHIPPED_CONSTANT_DEG) > 0.0, (
        f"{SHIPPED_CONSTANT_DEG} deg now converges {DEFAULT_FAMILY!r}")
    fix = dihedral_for_spiral(rep)
    assert fix.gamma_deg > SHIPPED_CONSTANT_DEG
    assert _spiral_root(rep, fix.gamma_deg) < 0.0


def test_the_answer_MOVES_with_the_configuration():
    """A tip device carries most of Cl_beta where there is one, so the same
    mission needs half the dihedral with one fitted. That spread is why a
    constant cannot be right."""
    bare = dihedral_for_spiral(_report("tail")).gamma_deg
    device = dihedral_for_spiral(_report("tail + winglet")).gamma_deg
    assert device < bare - 2.0, (
        f"the tip-device family wants {device:.2f} deg and the bare one "
        f"{bare:.2f} — a single constant would now fit both")


def test_a_design_that_does_not_weathercock_is_REFUSED_not_answered():
    """Same guard, same reason, as ``wing_score.spiral_refusal``: the
    criterion is a difference of two products, so a negative Cn_beta flips
    the second one and "converges" the spiral on paper."""
    fix = dihedral_for_spiral(
        _report("tail [free cant]", {"fin_volume_coeff": 0.002}))
    assert fix.status == "refused"
    assert fix.Cn_beta < 0.0
    assert fix.gamma_deg is None


def test_a_design_that_ALREADY_converges_is_told_so():
    fix = dihedral_for_spiral(
        _report("tail + winglet", {"wing_dihedral_deg": 8.0}))
    assert fix.status == "converges"
    assert fix.flown_deg == pytest.approx(8.0)
    assert fix.margin_now > 0.0
    assert fix.extra_deg == pytest.approx(0.0)


def test_out_of_reach_is_SAID_rather_than_clamped_to_the_band():
    """A recommendation is not a limit: a band too narrow to hold the answer
    must not hand back its top edge as though it were one."""
    fix = dihedral_for_spiral(_report("tail"), band=(0.0, 1.0))
    assert fix.status == "out_of_reach"
    assert fix.gamma_deg is None
    assert fix.band == (0.0, 1.0)
    assert SPIRAL_DIHEDRAL_MAX_DEG > 1.0


def test_the_report_it_is_handed_is_not_MUTATED():
    """It measures by rebuilding at other cants; a caller's report must come
    back the way it went in, or stage 5's deck would quietly become the
    deck of the last angle tried."""
    rep = _report("tail")
    before = dict(rep.get("breakdown") or {})
    dihedral_for_spiral(rep)
    assert dict(rep.get("breakdown") or {}) == before


# ------------------------------------------------------- and in the shells

def test_stage_5_answers_the_wing_half_in_the_same_press(capsys):
    """The fin scan's verdict is "the spiral is a WING answer here". Until
    this existed the panel stopped one number short of the fix — it named
    the lever and not the setting."""
    from tests.test_v4_stages import _armed

    ctx = _armed()
    ctx.render("controls", "vertical")
    ctx.act("controls_scan_spiral")
    scan = ctx.S["controls"]["spiral_scan"]
    assert scan["best"][2] <= 0.0, "no fin fixes it — the premise"
    fix = ctx.S["controls"].get("dihedral_scan")
    assert fix and fix["status"] == "found", (
        "the fin scan said no fin fixes it and left the WING answer "
        "unmeasured: " + repr(fix))
    assert fix["gamma_deg"] > SHIPPED_CONSTANT_DEG      # this is `tail`
    ctx.render("controls", "vertical")
    texts = _texts(ctx.views[("controls", "vertical")])
    assert any(f"{fix['gamma_deg']:.2f} deg of wing dihedral" in t
               for t in texts), \
        "the panel measured the angle and did not print it"
    capsys.readouterr()


def test_stage_5s_wing_answer_is_thrown_away_with_its_design(capsys):
    """A stale recommendation is a verdict about an aeroplane that is no
    longer on screen — the rule the fin scan already follows."""
    from tests.test_v4_stages import _armed

    ctx = _armed()
    ctx.render("controls", "vertical")
    ctx.act("controls_scan_dihedral")
    assert ctx.S["controls"].get("dihedral_scan") is not None
    ctx.render("controls", "vertical")           # a render must NOT clear it
    assert ctx.S["controls"].get("dihedral_scan") is not None
    ctx.S["controls"]["size_Vv"] = 0.02
    ctx.act("controls_size_vv")                  # ...an edit must
    assert ctx.S["controls"].get("dihedral_scan") is None
    capsys.readouterr()


def test_stage_3_measures_the_configuration_it_is_looking_at(capsys):
    """The card's own read-out, driven through the shell: the number is the
    one this family needs, and the constant beside it NAMES the family it
    was measured on rather than reading as a universal answer."""
    from gui.v3 import session as v3session
    from gui.v3.app import assemble

    ctx = assemble()
    ctx.act("accept_mission")
    # the shell OPENS with a dihedral now (session.V3_START_FLAGS), so switch
    # it off to be the user this card is for: one who has not answered yet
    ctx.act("set_wing_cant", False)
    ctx.render("wing", "type")
    info = v3session.spiral_dihedral(ctx.S)
    assert info["status"] == "found", info
    assert info["gamma_deg"] > 3.0, info
    texts = _texts(ctx.views[("wing", "type")])
    assert any(f"{info['gamma_deg']:.2f} deg of dihedral is" in t
               for t in texts), (
        "the card did not print what it measured — and with the switch OFF "
        "is exactly when the user needs it")

    # ...and with the switch ON, the ladder's own constant NAMES the family
    # it came from rather than reading as a universal answer
    ctx.act("set_wing_cant", True)
    ctx.render("wing", "type")
    assert any("tail [free cant]" in t
               for t in _texts(ctx.views[("wing", "type")])), (
        "the 5.7 deg constant is quoted without naming the family it was "
        "measured on")
    capsys.readouterr()


def test_the_card_prices_the_cant_only_where_the_family_can(capsys):
    """A lifting-line family cannot score a dihedral at all, so quoting one
    family's price beside another family's angle would be the constant's
    mistake in the other column."""
    from gui.v3 import session as v3session
    from gui.v3.app import assemble

    ctx = assemble()
    ctx.act("accept_mission")
    # the LIFTING-LINE twin: the shell opens on the lattice one now, so this
    # is reached by putting the tail's height back to a stated value
    ctx.act("set_choice", "tail_height", "fixed")
    ctx.act("set_wing_cant", False)
    line = v3session.spiral_dihedral(ctx.S)
    assert api.WING_CANT_KEYS[0] not in api.PROBLEM_SPECS[
        ctx.S["wing"]["problem"]].flags
    assert line["lod_cost_pct"] is None

    ctx.act("set_choice", "tail_height", "free")          # the lattice twin
    lattice = v3session.spiral_dihedral(ctx.S)
    assert api.WING_CANT_KEYS[0] in api.PROBLEM_SPECS[
        ctx.S["wing"]["problem"]].flags
    assert lattice["status"] == "found"
    assert lattice["lod_cost_pct"] is not None
    assert 0.0 < lattice["lod_cost_pct"] < 5.0, lattice
    capsys.readouterr()


def test_stage_6_stops_pointing_a_divergent_spiral_at_the_FIN(capsys):
    """The stability strip beside the simulation said "Stage 5 sizes the fin
    against it", and stage 5's own measurement is that no fin size does. A
    pointer at the lever that cannot move is how a user ends up sizing fins
    for an aeroplane whose answer is in the wing."""
    from tests.test_v4_stages import _armed

    ctx = _armed()
    ctx.render("controls", "derivatives")
    ctx.render("flight", "fly")
    texts = _texts(ctx.views[("flight", "fly")])
    spiral = [t for t in texts if "The first three are SIGNS" in t]
    assert spiral, "the stability strip did not render its own note"
    assert any("no fin size does" in t for t in spiral), (
        "a divergent spiral is still pointed at the fin alone")
    capsys.readouterr()


def test_the_design_shell_asks_the_API_and_not_the_rebuild():
    """The wing's cant is stage 3's question and the measurement is a
    rebuild, which belongs to the stages that fly. ``api`` is where the two
    meet — and it hands back a JSON-safe dict, so a shell can cache the
    answer and compare it with the one it showed last time.

    The negative half of this claim is
    ``test_v4_stages.test_no_v3_source_file_MENTIONS_the_v4_machinery``,
    which fails if any V3 file so much as names the rebuild.
    """
    import json

    rep = _report("tail")
    got = api.dihedral_for_spiral(rep)
    assert json.loads(json.dumps(got)) == got, "not JSON-safe"
    fix = dihedral_for_spiral(rep)
    assert got["status"] == fix.status
    assert got["gamma_deg"] == pytest.approx(fix.gamma_deg)
    assert got["fin_stated"] is True                 # `tail` states a fin
    assert api.dihedral_for_spiral(
        _report("wing (free chord law)"))["fin_stated"] is False, (
        "a report with no vertical surface has one invented for it, and the "
        "answer has to say so")


#: the range the card's own step quotes for the wing+tail families with no
#: tip device, and the one it quotes for the family with one. A published
#: number needs a tripwire: these are re-derived below, not trusted.
QUOTED_BARE = (5.43, 6.13)
QUOTED_DEVICE = 3.07


def test_the_range_the_CARD_quotes_is_the_range_that_is_measured():
    """The step's own note used to say "5.69-5.74 by family", which no
    family measured — it was the study's one number given a spread. The
    card now quotes 5.43-6.13 with no tip device and 3.07 with one, and
    this is what fails when the aeroplane moves under those words."""
    from gui.v3.stages import wing as wing_stage

    bare = {n: dihedral_for_spiral(_report(n)).gamma_deg
            for n in ("tail", "tail [designed tail]", "tail [free height]",
                      "tail [free cant]")}
    lo, hi = min(bare.values()), max(bare.values())
    assert lo == pytest.approx(QUOTED_BARE[0], abs=0.05), bare
    assert hi == pytest.approx(QUOTED_BARE[1], abs=0.05), bare
    device = dihedral_for_spiral(_report("tail + winglet")).gamma_deg
    assert device == pytest.approx(QUOTED_DEVICE, abs=0.05)

    src = open(wing_stage.__file__).read()
    assert f"{QUOTED_BARE[0]:.2f}-{QUOTED_BARE[1]:.2f}" in src, (
        "the card no longer quotes the measured range")
    assert f"{QUOTED_DEVICE:.2f} with one" in src
    assert "5.69-5.74" not in src, (
        "the spread that no family measured is back on the card")


def test_pressing_the_button_STATES_the_angle_it_measured(capsys):
    """Driven, not grepped: the press writes the wing's cant flag, the
    rebuild flies it, and the card then reports a convergent spiral.

    It is the whole loop the complaint was about — a number on a card that
    nothing could act on is where "no spiral stability" lived.
    """
    from gui.v3 import session as v3session
    from gui.v3.app import assemble

    ctx = assemble()
    ctx.act("accept_mission")
    ctx.act("set_wing_cant", False)       # the aeroplane before it is answered
    ctx.render("wing", "type")
    info = v3session.spiral_dihedral(ctx.S)
    assert info["status"] == "found", info
    want = info["gamma_deg"]

    got = ctx.act("take_spiral_dihedral")
    assert got == pytest.approx(round(want, 2))
    assert ctx.S["wing"]["flags"][api.WING_CANT_KEYS[0]] == pytest.approx(
        round(want, 2)), "the press did not reach the flag the solver reads"

    after = v3session.spiral_dihedral(ctx.S)
    assert after["status"] == "converges", after
    assert after["flown_deg"] == pytest.approx(round(want, 2), abs=0.01)
    assert after["margin_now"] > 0.0
    capsys.readouterr()


@pytest.mark.parametrize("choices,want", [
    ({"medium": "track"}, "no_fin"),
    ({"medium": "water"}, "not_applicable"),
    ({"tail": False}, "no_fin"),          # a wing with no vertical surface
])
def test_a_design_that_is_not_a_FLYING_AEROPLANE_gets_no_angle(choices, want,
                                                               capsys):
    """A rebuild will fly anything. Asked about a car's rear wing it invents
    a fin, trims the assembly as an aircraft and reports that 6.92 deg of
    dihedral would turn "its" spiral — a number about an aeroplane nobody
    designed. The card asks only where the design IS one: air, and a
    vertical surface the DESIGN carries."""
    from gui.v3 import session as v3session
    from gui.v3.app import assemble

    ctx = assemble()
    ctx.act("accept_mission")
    for key, value in choices.items():
        ctx.act("set_choice", key, value)
    info = v3session.spiral_dihedral(ctx.S)
    assert info["status"] == want, info
    assert "gamma_deg" not in info
    ctx.render("wing", "type")
    texts = _texts(ctx.views[("wing", "type")])
    assert not any("of dihedral is the least" in t for t in texts), (
        "an angle was quoted for a design with no spiral to turn")
    capsys.readouterr()


@pytest.mark.parametrize("choices", [
    {"tail_height": "free"},          # the cant is STATED as a flag
    {"wing_cant": "free"},            # ...and the twin that SEARCHES it
    # ...and the two HALF states, where the card draws a searched band AND
    # the other row's typed buttons in the same group box
    {"wing_cant": "dihedral"},
    {"wing_cant": "sweep"},
])
def test_the_measurement_is_printed_ONCE_per_card(choices, capsys):
    """The searched branch draws its own note under the same group box, and
    the card renders both halves — so the number appeared twice on exactly
    the configuration a user asking for a searched cant is on."""
    from gui.v3.app import assemble

    ctx = assemble()
    ctx.act("accept_mission")
    ctx.act("set_choice", "tail", True)
    for key, value in choices.items():
        ctx.act("set_choice", key, value)
    ctx.render("wing", "type")
    # whatever the verdict is on this configuration, the card states it ONCE
    openers = ("MEASURED ON THIS CONFIGURATION", "ALREADY converges",
               "NOT ANSWERED", "no dihedral up to")
    seen = [t for t in _texts(ctx.views[("wing", "type")])
            if any(o in t for o in openers)]
    assert len(seen) == 1, f"the read-out is drawn {len(seen)} times"
    capsys.readouterr()


# ------------------------------------------- and WHOSE vertical surface it is

def _tail_report(**flags):
    built = api.PROBLEM_SPECS["tail"].build({}, flags, None)
    return api.design_report(
        api.RunConfig(problem_name="tail", budget=4, seed=0, flags=flags),
        np.asarray(built.bounds, dtype=float).mean(axis=1))


def test_a_V_TAIL_is_ANSWERED_because_its_panels_are_the_vertical_surface():
    """The gate asked "is there a fin block", and a V-tail has none.

    Its ``geometry["fin"]`` is an explicit None — "no SEPARATE fin" — and its
    yaw comes from the cant of its own two panels, which is stated geometry.
    It is the ONE layout in this package whose vertical surface is stated,
    charged and flown with nothing invented anywhere: ``cd0_fin`` is 0.0
    because there is no fin to charge, and Cn_beta moves with the cant the
    design states. Read as "no fin block, therefore no vertical surface", it
    was refused the answer it can give BEST — and told it had no yaw
    stiffness while carrying +0.067.
    """
    from gui.v3 import session as v3session

    rep = _tail_report(tail_type="v_tail", dihedral_deg=35.0)
    assert rep["geometry"]["fin"] is None, "a V-tail grew a separate fin"
    assert rep["breakdown"]["cd0_fin"] == 0.0, "charged for a fin it has not"
    assert v3session._states_its_own_vertical(rep) is True

    fm = build_flight_model(rep)
    assert float(fm.deck.Cn_beta) > 0.05, float(fm.deck.Cn_beta)
    fix = api.dihedral_for_spiral(rep)
    assert fix["status"] == "found", fix
    # ...and it is the family the criterion is most trustworthy on: at the
    # angle it offers, the MODE is stable, and half a degree less it is not
    assert _spiral_root(rep, fix["gamma_deg"]) < 0.0
    assert _spiral_root(rep, fix["gamma_deg"] - 0.5) > 0.0


def test_a_fin_switched_OFF_is_still_refused():
    """The case that looks identical at the ``fin`` key and must not be
    answered: nothing carries yaw, which is what the user asked for."""
    rep = _tail_report(tail_type="conventional", fin=False)
    assert rep["geometry"]["fin"] is None
    fm = build_flight_model(rep)
    assert float(fm.deck.Cn_beta) == pytest.approx(0.0, abs=1e-9)
    assert api.dihedral_for_spiral(rep)["status"] == "refused"


def test_a_CANARD_is_refused_because_the_rebuild_would_INVENT_its_fin():
    """Its ``fin`` key is ABSENT rather than None — the aft fin's arm is not
    the upstream surface station the block carries — so the design is charged
    ``cd0_fin`` for a surface it never describes and the rebuild makes one
    up. A dihedral measured on that is about an aeroplane nobody designed."""
    from gui.v3 import session as v3session

    rep = _tail_report(tail_type="canard")
    assert "fin" not in rep["geometry"], "the canard now states its fin"
    assert rep["breakdown"]["cd0_fin"] > 0.0, "charged for nothing at all"
    assert v3session._states_its_own_vertical(rep) is False

    fm = build_flight_model(rep)
    assert float(fm.deck.Cn_beta) > 0.1, "the invented fin is not stiff here"
    assert any("not in this report, so its size is assumed" in n
               for n in fm.assumptions), fm.assumptions


def test_the_water_refusal_says_something_TRUE():
    """It said "a water design has no spiral mode to turn". It has one — a
    hydrofoil+elevator rebuild trims and classifies a spiral, and it is
    divergent. The reason it is not answered is that nothing about it would
    be trustworthy: the only vertical surface is the mast."""
    from gui.v3 import session as v3session
    from gui.v3.app import assemble

    ctx = assemble()
    ctx.act("accept_mission")
    ctx.act("set_choice", "medium", "water")
    info = v3session.spiral_dihedral(ctx.S)
    assert info["status"] == "not_applicable"
    assert "no spiral mode" not in info["why"], info["why"]
    assert "mast" in info["why"]

    # ...and the claim that reason rests on: a water design does not state a
    # vertical surface of its own, so the number a forced rebuild would give
    # is about a fin that rebuild invented. Asserted through the ASSUMPTION
    # the rebuild itself records, which is true whether the report carries
    # the strut (a mast block) or nothing at all.
    built = api.PROBLEM_SPECS["hydrofoil + elevator"].build({}, {}, None)
    rep = api.design_report(
        api.RunConfig(problem_name="hydrofoil + elevator", budget=4, seed=0),
        np.asarray(built.bounds, dtype=float).mean(axis=1))
    fin = (rep.get("geometry") or {}).get("fin")
    fm = build_flight_model(rep)
    if fin:
        # the design states its strut: it stands AT the foil, so it has no
        # yaw arm to speak of and moves with the CG rather than against it
        assert fin.get("kind") == "mast", fin
        assert float(fin.get("V_v") or 0.0) == 0.0, fin
    else:
        # ...or it states nothing, and the rebuild says so in as many words
        assert any("not in this report, so its size is assumed" in n
                   for n in fm.assumptions), fm.assumptions
