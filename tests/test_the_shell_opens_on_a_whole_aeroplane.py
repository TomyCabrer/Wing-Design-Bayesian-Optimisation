"""V3 opens on an aeroplane, and its spiral converges.

Before this, the shell opened on ``nice_app.BUILDER_START`` — a WING ALONE.
A wing alone has no vertical surface, so it has no yaw stiffness, no spiral
mode and nothing for stages 5 and 6 to fly; switch a tail on and the answer
was "no spiral stability", because the family that switch derives solves its
wing on a LIFTING LINE and a lifting line cannot carry a dihedral.

So the opening configuration is a whole aeroplane (``session.V3_START_CHOICES``
/ ``V3_START_FLAGS``), and three things about it are measured rather than
asserted from memory:

* its spiral CONVERGES — criterion and 6-DOF eigenvalue, and not by a hair:
  the mode halves in about 21 s, where the criterion's own crossing (5.57
  deg) would leave it halving in 548 s, which is a neutral aeroplane;
* it carries a BODY, and the body is the shape that leaves the searched arm
  an interior optimum. With none the best arm is the top of the band, and
  with a fat one it is the bottom;
* and every one of the two switches this stage grew — the body, and the
  wing's own cant — writes what the shell can defend when it is turned on
  and removes the flag outright when it is turned off.
"""
from __future__ import annotations

import sys

import numpy as np
import pytest

from aerobo import api, drag
from aerobo.flightmodel import build_flight_model

sys.path.insert(0, "gui")


def _built_at(S, **flag_overrides):
    """The opening session's own problem, built the way its run would be."""
    from gui.v3 import config

    name = S["wing"]["problem"]
    flags = dict(config.flags(S), **flag_overrides)
    for k, v in list(flags.items()):
        if v is None:
            flags.pop(k)
    return api.PROBLEM_SPECS[name].build(
        config.mission_kwargs(S), flags, config.bounds_overrides(S)), flags


def _modes_of(S, **flag_overrides):
    from aerobo import sixdof as sd
    from v4 import modes as md

    built, flags = _built_at(S, **flag_overrides)
    x = np.asarray(built.bounds, dtype=float).mean(axis=1)
    rep = api.design_report(
        api.RunConfig(problem_name=S["wing"]["problem"], budget=4, seed=0,
                      flags=flags), x)
    fm = build_flight_model(rep)
    st, _ = sd.trim_level(fm.aircraft, V=float(fm.deck.V), altitude_m=0.0)
    return md.classify(sd.linearise(fm.aircraft, st), float(st.V))


# ------------------------------------------------- the aeroplane it opens on

def test_a_fresh_session_is_a_whole_aeroplane():
    from gui.v3 import session

    S = session.make_session()
    ch = S["wing"]["choices"]
    assert ch["tail"] is True, "the shell still opens on a wing alone"
    spec = api.PROBLEM_SPECS[S["wing"]["problem"]]
    assert api.WING_CANT_KEYS[0] in tuple(spec.flags), (
        f"{S['wing']['problem']!r} cannot be given a wing dihedral, so its "
        f"spiral cannot be turned by anything")
    flags = S["wing"]["flags"]
    assert flags.get("wing_dihedral_deg"), flags
    assert flags.get("fuselage_diameter_m"), flags


def test_the_opening_aeroplanes_SPIRAL_CONVERGES():
    """The ask, in one assertion — both instruments, on the session the
    shell actually opens."""
    from gui.v3 import session

    S = session.make_session()
    info = session.spiral_dihedral(S)
    assert info["status"] == "converges", info
    assert info["margin_now"] > 0.0

    spiral = _modes_of(S)["spiral"]
    assert spiral.stable, f"the criterion says converges and the MODE is "
    assert float(spiral.real) < 0.0


def test_it_converges_with_MARGIN_and_not_by_a_hair():
    """The crossing is not the answer: at 5.57 deg this design's spiral
    halves in 548 s, which is neutral. The opening angle is 7.0."""
    from gui.v3 import session

    S = session.make_session()
    gam = float(S["wing"]["flags"]["wing_dihedral_deg"])
    at = _modes_of(S)["spiral"]
    crossing = _modes_of(S, wing_dihedral_deg=5.57)["spiral"]
    assert gam > 5.6, gam
    assert at.double_or_half_s < 60.0, (
        f"the opening spiral halves in {at.double_or_half_s:.0f} s")
    assert crossing.double_or_half_s > 200.0, (
        "the criterion's crossing is no longer the near-neutral aeroplane "
        "this default exists to avoid — re-derive the opening angle")


def test_every_other_mode_is_stable_too():
    """A stable spiral bought by making something else diverge would not be
    an aeroplane. All five, on the opening design."""
    from gui.v3 import session

    modes = _modes_of(session.make_session())
    for name in ("spiral", "dutch roll", "roll subsidence", "short period",
                 "phugoid"):
        mode = modes.get(name)
        assert mode is not None, f"no {name} in the deck"
        assert mode.stable, f"{name} is unstable on the opening design"


# ------------------------------------------------------------------ the body

def test_the_opening_body_leaves_the_ARM_an_interior_optimum():
    """Why the body is there at all, and why it is THAT slender.

    With no body a longer arm is free — its drag unmodelled, its static
    margin and tail volume fully counted — so the best arm is the top of the
    band. Too fat and it is the bottom. Both failures are measured here, on
    the family the shell opens on.
    """
    from gui.v3 import session

    S = session.make_session()
    built, flags = _built_at(S)
    labels = list(built.param_labels)
    i = labels.index("l_t_m")
    lo, hi = np.asarray(built.bounds, dtype=float)[i]
    arms = np.linspace(float(lo), float(hi), 21)

    def best_arm(diameter):
        b, _ = _built_at(S, fuselage_diameter_m=diameter)
        x0 = np.asarray(b.bounds, dtype=float).mean(axis=1)
        lods = []
        for a in arms:
            x = x0.copy()
            x[i] = a
            lods.append(float(b.evaluate(x)["LoD"]))
        return float(arms[int(np.argmax(lods))])

    assert best_arm(None) == pytest.approx(float(hi)), (
        "with no body the arm no longer rides the top bound — the reason "
        "the shell opens with one has changed")
    fat = drag.diameter_for_fineness(0.5 * (lo + hi), drag.SLENDER_FINENESS)
    assert best_arm(fat) == pytest.approx(float(lo)), (
        f"a body at the form factor's own minimum ({fat:.3f} m) no longer "
        f"over-penalises the arm")
    own = float(flags["fuselage_diameter_m"])
    interior = best_arm(own)
    assert float(lo) < interior < float(hi), (
        f"the opening body puts the best arm at {interior:.2f} m, on a "
        f"bound of the {lo:g}-{hi:g} m band")


def test_the_opening_diameter_is_the_stated_slenderness():
    """A number nothing re-derives goes stale. This is the rule, not the
    value: whatever the arm band is, the body is that fineness ratio."""
    from gui.v3 import session

    S = session.make_session()
    got = float(S["wing"]["flags"]["fuselage_diameter_m"])
    row = api.PROBLEM_SPECS[S["wing"]["problem"]].default_bounds["l_t_m"]
    arm = 0.5 * (float(row[0]) + float(row[1]))
    want = drag.diameter_for_fineness(arm, session.V3_FUSELAGE_FINENESS)
    assert got == pytest.approx(want, abs=5e-4)
    assert session.fuselage_diameter_default(S) == pytest.approx(got, abs=5e-4)


def test_the_lattice_family_can_actually_TAKE_the_body_it_declares():
    """It declared ``fuselage_diameter_m`` and could not be built with one:
    ``check_flags`` accepted the flag and the builder raised TypeError. The
    one configuration that both searches an arm AND can carry a wing
    dihedral was the one that could not charge a body for that arm."""
    name = "tail [free height]"
    api.check_flags(name, {"fuselage_diameter_m": 0.22})
    spec = api.PROBLEM_SPECS[name]
    built = spec.build({}, {"fuselage_diameter_m": 0.22}, None)
    x = np.asarray(built.bounds, dtype=float).mean(axis=1)
    out = built.evaluate(x)
    assert out["cd0_fuselage"] > 0.0
    assert out["fuselage_fineness"] == pytest.approx(
        drag.body_length_for_arm(x[list(built.param_labels).index("l_t_m")])
        / 0.22, rel=1e-9)
    bare = spec.build({}, {}, None).evaluate(x)
    assert bare["cd0_fuselage"] == 0.0
    assert bare["LoD"] > out["LoD"], "the body is charged nothing"


# --------------------------------------------------------------- the switches

def test_the_body_switch_writes_what_it_can_defend_and_removes_it(capsys):
    from gui.v3 import session as v3session
    from gui.v3.app import assemble

    ctx = assemble()
    ctx.act("accept_mission")
    key = api.FUSELAGE_KEYS[0]
    assert ctx.S["wing"]["flags"].get(key)

    ctx.act("set_fuselage", False)
    assert key not in ctx.S["wing"]["flags"], (
        "off must REMOVE the flag — a zero diameter is refused by the api")

    ctx.act("set_fuselage", True)
    assert ctx.S["wing"]["flags"][key] == pytest.approx(
        v3session.fuselage_diameter_default(ctx.S)), (
        "on left the field blank, so the switch charged nothing")
    capsys.readouterr()


def test_the_cant_switch_offers_the_MEASURED_angle_when_it_is_turned_on(capsys):
    from gui.v3 import session as v3session
    from gui.v3.app import assemble

    ctx = assemble()
    ctx.act("accept_mission")
    ctx.act("set_wing_cant", False)
    assert not any(k in ctx.S["wing"]["flags"] for k in api.WING_CANT_KEYS)
    want = v3session.spiral_dihedral(ctx.S)
    assert want["status"] == "found", want

    ctx.act("set_wing_cant", True)
    got = ctx.S["wing"]["flags"][api.WING_CANT_KEYS[0]]
    assert got == pytest.approx(round(float(want["gamma_deg"]), 2))
    assert v3session.spiral_dihedral(ctx.S)["status"] == "converges"
    capsys.readouterr()


def test_switching_the_cant_off_leaves_no_row_the_optimiser_still_rides(capsys):
    """OFF on a family that SEARCHES the cant has to take the family off its
    free-cant twin: clearing a flag would leave the switch reading "off"
    over two rows the optimiser is still riding."""
    from gui.v3.app import assemble

    ctx = assemble()
    ctx.act("accept_mission")
    ctx.act("set_choice", "wing_cant", "free")
    assert api.cant_is_searched(ctx.S["wing"]["problem"])

    ctx.act("set_wing_cant", False)
    assert not api.cant_is_searched(ctx.S["wing"]["problem"]), \
        ctx.S["wing"]["problem"]
    assert not any(k in ctx.S["wing"]["flags"] for k in api.WING_CANT_KEYS)
    capsys.readouterr()


def test_both_switches_are_on_the_CONFIGURATION_card(capsys):
    """Where the reader is already answering what the aeroplane is. The body
    used to be asked three cards down inside the trim layout, and only when
    the arm happened to be searched; the cant was in the right-hand column
    beside the derived-solver read-out."""
    from gui.v3.app import assemble

    ctx = assemble()
    ctx.act("accept_mission")
    ctx.render("wing", "type")
    texts = []
    for e in ctx.views[("wing", "type")].descendants():
        t = getattr(e, "text", "")
        if isinstance(t, str) and t:
            texts.append(t)
    assert any("this aeroplane has a fuselage" in t for t in texts), texts[:8]
    assert any("this wing has dihedral" in t for t in texts)
    capsys.readouterr()


# ----------------------------------------------------------- the empennage

def test_the_empennage_carries_TWO_sections_into_the_run(capsys):
    """A horizontal tail and a vertical tail are different surfaces with
    different jobs, and each has its own stage (2.5 and 2.7). What this
    asserts is that the two answers travel SEPARATELY: the tailplane as a
    named polar, the fin as the thickness its own symmetric section has —
    which is the whole of what the lattice, the drag book and the CAD loft
    take from it (``fin.fin_shape_kwargs``).
    """
    from gui.v3 import config, session as v3session
    from gui.v3.app import assemble

    ctx = assemble()
    ctx.act("accept_mission")
    S = ctx.S
    assert v3session.surface_job(S, "aft") == "trim"
    assert v3session.surface_job(S, "fin") == "fin"

    v3session.set_section(S, {"name": "naca4412", "tc": 0.12,
                              "source": "library"}, "chosen", surface="aft")
    v3session.set_section(S, {"name": "naca0008", "tc": 0.08,
                              "source": "library", "symmetric": True},
                          "chosen", surface="fin")
    flags = config.flags(S)
    assert flags[api.SECTION_AFT_KEY] == "naca4412"
    assert flags["fin_tc"] == pytest.approx(0.08)
    assert v3session.fin_thickness(S) == pytest.approx(0.08)
    capsys.readouterr()


def test_the_fins_badge_reports_the_FINS_section(capsys):
    """It fell through to the wing's branch, so the vertical tail's own
    badge showed the WING's section — and, unanswered, said "the family's
    own section" about a surface that flies a symmetric one at its own
    thickness."""
    from gui.v3 import session as v3session
    from gui.v3.app import assemble

    ctx = assemble()
    ctx.act("accept_mission")
    S = ctx.S
    assert v3session.section_summary(S, "fin") == \
        f"{v3session.fin_default_section(S)['name']} (its own default)"

    v3session.set_section(S, {"name": "naca0006", "tc": 0.06,
                              "source": "library"}, "chosen", surface="main")
    assert v3session.section_summary(S, "fin") != \
        v3session.section_summary(S, "main"), \
        "the fin's badge is reporting the wing's section"

    v3session.set_section(S, {"name": "naca0009", "tc": 0.09,
                              "source": "library"}, "chosen", surface="fin")
    assert v3session.section_summary(S, "fin") == "naca0009"
    capsys.readouterr()


def test_the_fin_is_a_surface_the_shell_MAINTAINS():
    """``SURFACES`` is what the shell loops over to keep each surface's
    recommended weights following the configuration. The fin had a stage, a
    workspace and its own recommended weights, and was not in it — so its
    weights were the only ones that never followed."""
    from gui.v3 import session

    assert "fin" in session.SURFACES
    assert set(session.SURFACE_STAGES) >= set(session.SURFACES)
    S = session.make_session()
    want, _why = session.recommended_weights(S, "fin")
    assert want and session.airfoil_state(S, "fin")["weights_source"] == \
        "recommended"
