"""A flap is a CONFIGURATION, and every consumer downstream treated it as a
column.

The complaint was "no flaps for control or flight sim". The flap was not
missing — it was inert, in two independent ways.

**It bought nothing.** ``dynamics.flap`` builds an honest per-radian lift
column, and that was the whole of it: ``Stall.CL_max`` was a frozen scalar
taken from the CLEAN section, so the p-norm clip ate the entire increment and
all a flap did was move the stall to a LOWER incidence. Measured on the
reference tail before this file existed, the slowest speed it could hold
level flight at was 22.20 m/s clean and 22.15 m/s at 40 deg of flap. Nor did
it cost anything: profile drag came from the report's undeflected ``CDp``, so
a wing that had just added a third of its lift added none of the drag.

**It could not be flown.** The deflection lived in ``S["flight"]["stick"]``,
the dict ``arm()`` zeroes wholesale, while the slider that set it was written
only at creation — so a pilot who selected 30 deg and pressed "Re-trim &
reset" watched a slider reading 30 drive a model flying 0. There was no key,
no pad button, nothing on the glass, and nothing in the "Trimmed" box saying
what configuration the trim was for.

Both halves are asserted here. The numbers below are this package's own
physics; the two constants they ride
(:data:`sixdof.FLAP_CLMAX_RATIO`, :data:`sixdof.FLAP_CD_K`) and the
effectiveness table (:data:`sixdof.FLAP_ETA_DEG`) are stated ASSUMPTIONS and
say so in the flight model's own ``assumptions`` list — which the last test
here pins, because an invented number that stops saying it is invented is the
defect this repo keeps finding.
"""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace as NS

import numpy as np
import pytest

from aerobo import api, sixdof as sd
from aerobo.flightmodel import ControlsSpec, build_flight_model
from gui.v4 import app as v4app, hud, pad as pd, stick as stk


# --------------------------------------------------------------- the physics

@pytest.fixture(scope="module")
def flapped():
    cfg = api.RunConfig(problem_name="tail", budget=4, seed=0)
    built = api.PROBLEM_SPECS["tail"].build({}, {}, None)
    rep = api.design_report(
        cfg, np.asarray(built.bounds, dtype=float).mean(axis=1))
    return build_flight_model(rep, ControlsSpec(flap=True))


def _at(ac, deg):
    return replace(ac, controls={**ac.controls, "flap": np.deg2rad(deg)})


def _vmin(ac, deg):
    """The slowest level flight this aeroplane can hold, to 0.1 m/s."""
    slowest = None
    for V in np.arange(32.0, 12.0, -0.1):
        try:
            sd.trim_level(_at(ac, deg), V=float(V), altitude_m=300.0)
        except Exception:                            # noqa: BLE001
            break
        slowest = float(V)
    return slowest


def test_the_ceiling_moves_with_the_column(flapped):
    """One lever, not two. The ceiling is DERIVED from the deck's own flap
    column, so a design whose flap is worth less lift gets a ceiling that is
    worth less too — they cannot contradict each other."""
    st = flapped.aircraft.stall
    dCL = -float(flapped.deck.columns["flap"]["CZ"])
    assert dCL > 0.0
    assert st.dCL_max_dflap == pytest.approx(sd.FLAP_CLMAX_RATIO * dCL)
    assert st.ceiling(np.deg2rad(40.0)) > st.ceiling(0.0)
    # ...and the stall ANGLE falls, which is what a flapped wing does
    assert st.stall_alpha(np.deg2rad(40.0)) < st.stall_alpha(0.0)


def test_it_lowers_the_stall_speed(flapped):
    """The user's sentence, as a number. 22.20 -> 22.15 m/s was the
    measurement that made this a defect."""
    clean, down = _vmin(flapped.aircraft, 0.0), _vmin(flapped.aircraft, 40.0)
    assert clean is not None and down is not None
    assert down < clean - 1.0, (clean, down)


def test_it_costs_profile_drag(flapped):
    """...and it is NOT the deck's CX column, which is identically
    ``-CZ sin(alpha_ref)`` — the base-incidence tilt of the extra lift, and
    no drag at all. Restoring that would hand the aeroplane a forward force
    for putting the flaps down."""
    ac = flapped.aircraft
    assert ac.flap_drag_k > 0.0
    assert ac.flap_drag(0.0) == 0.0
    assert ac.flap_drag(np.deg2rad(40.0)) > ac.flap_drag(np.deg2rad(20.0)) > 0.0
    st, tr = sd.level_at(_at(ac, 0.0), V=30.0, alpha=np.deg2rad(2.0),
                         altitude_m=300.0)
    clean = tr.coefficients(st)["CD"]
    st, tr = sd.level_at(_at(ac, 40.0), V=30.0, alpha=np.deg2rad(2.0),
                         altitude_m=300.0)
    assert tr.coefficients(st)["CD"] > clean


def test_a_big_deflection_is_worth_less_than_a_linear_one(flapped):
    """Thin-aerofoil tau is the limit as the deflection goes to zero and the
    column is linear in the angle for ever, so 40 deg came back at roughly
    twice what a plain flap of that size is worth. The effectiveness ratio is
    one, exactly, at small angles."""
    assert sd.flap_effectiveness_ratio(0.0) == pytest.approx(1.0)
    assert sd.flap_effectiveness_ratio(np.deg2rad(5.0)) == pytest.approx(1.0)
    assert sd.flap_effectiveness_ratio(np.deg2rad(40.0)) < 0.8
    ac = flapped.aircraft
    st = sd.State(pos=np.zeros(3), quat=np.array([1.0, 0, 0, 0]),
                  vel=np.array([30.0, 0, 0]), rates=np.zeros(3))
    lin = -_at(replace(ac, stall=sd.Stall(enabled=False)),
               10.0).coefficients(st)["CZ"]
    big = -_at(replace(ac, stall=sd.Stall(enabled=False)),
               40.0).coefficients(st)["CZ"]
    clean = -replace(ac, stall=sd.Stall(enabled=False)).coefficients(st)["CZ"]
    assert (big - clean) < 4.0 * (lin - clean), \
        "40 deg is still worth four times what 10 deg is"


def test_an_aeroplane_with_no_flap_is_untouched(flapped):
    """The whole mechanism is zero where no flap is fitted, so nothing that
    does not have one can be changed by any of it."""
    cfg = api.RunConfig(problem_name="tail", budget=4, seed=0)
    built = api.PROBLEM_SPECS["tail"].build({}, {}, None)
    rep = api.design_report(
        cfg, np.asarray(built.bounds, dtype=float).mean(axis=1))
    fm = build_flight_model(rep, ControlsSpec(flap=False))
    assert "flap" not in fm.deck.columns
    assert fm.aircraft.flap_drag_k == 0.0
    assert fm.aircraft.stall.dCL_max_dflap == 0.0
    assert fm.aircraft.stall.dalpha_stall_dflap == 0.0


def test_the_assumptions_say_they_are_assumptions(flapped):
    """Two invented constants and a table, all three named on screen."""
    said = " ".join(flapped.assumptions)
    assert "ASSUMPTION" in said
    assert "FLAP_CD_K" in said and "FLAP_ETA_DEG" in said
    assert "FLAP_CLMAX_RATIO" in said


# ------------------------------------------------------- the trim it is at

def test_the_trim_bracket_is_the_aeroplanes_own(flapped):
    """A hard-coded +16 deg was a cap on every design in the package, and it
    bound: the reference tail reaches its ceiling AT the stop, so "cannot fly
    level" was a statement about the bracket. Never narrower than the
    constant it replaces."""
    lo, hi = sd.trim_alpha_bracket(flapped.aircraft)
    assert hi >= sd.TRIM_ALPHA_BRACKET[1]
    assert lo <= sd.TRIM_ALPHA_BRACKET[0]
    assert hi > flapped.aircraft.stall.alpha_stall_rad


def test_a_speed_it_cannot_fly_is_still_refused(flapped):
    """The refusal is preserved. Past the ceiling more incidence buys LESS
    lift, so alpha still pins at the stop with the residual left over."""
    with pytest.raises(ValueError):
        sd.trim_level(flapped.aircraft, V=8.0, altitude_m=300.0)


# ------------------------------------------------------------- the controls

def _armed(flap=True):
    ctx = v4app.assemble()
    cfg = api.RunConfig(problem_name="tail", budget=4, seed=0)
    built = api.PROBLEM_SPECS["tail"].build({}, {}, None)
    ctx.S["run"]["record"] = {"pretend": "a completed run"}
    ctx.S["run"]["report"] = api.design_report(
        cfg, np.asarray(built.bounds, dtype=float).mean(axis=1))
    ctx.S["controls"]["flap"]["on"] = bool(flap)
    ctx.act("controls_rebuild")
    ctx.render("flight", "fly")
    return ctx


class _Key:
    def __init__(self, name, down=True):
        self.key = NS(name=name)
        self.action = NS(keydown=down, keyup=not down)


def test_it_is_not_a_sprung_stick_axis():
    """``stick`` is the SPRUNG demand and ``step`` runs every axis of
    ``LIMITS`` back to centre. A flap in there was zeroed by ``arm``."""
    ctx = _armed()
    assert "flap" not in ctx.S["flight"]["stick"]
    assert "flap" not in stk.LIMITS
    assert "flap" not in stk.KEYMAP
    assert "flap_deg" in ctx.S["flight"]["live"]


def test_the_keyboard_steps_it_and_a_re_trim_keeps_it():
    """The defect, end to end: select 20, press Re-trim, still 20 — on the
    lever AND on the aeroplane."""
    ctx = _armed()
    F = ctx.S["flight"]
    ctx.act("flight_run", True)
    ctx.act("flight_key", _Key("f"))
    ctx.act("flight_key", _Key("f"))
    assert F["live"]["flap_deg"] == pytest.approx(20.0)
    assert np.rad2deg(F["ac"].controls["flap"]) == pytest.approx(20.0)
    ctx.act("flight_arm")
    assert F["live"]["flap_deg"] == pytest.approx(20.0)
    assert np.rad2deg(F["ac"].controls["flap"]) == pytest.approx(20.0)
    ctx.act("flight_key", _Key("v"))
    assert F["live"]["flap_deg"] == pytest.approx(10.0)


def test_the_pad_steps_it_too():
    ctx = _armed()
    ctx.act("flight_run", True)
    ctx.act("flight_pad", NS(args=[{"on": True, "a": ["flap_down"], "h": {}}]))
    assert ctx.S["flight"]["live"]["flap_deg"] == pytest.approx(10.0)
    ctx.act("flight_pad", NS(args=[{"on": True, "a": ["flap_up"], "h": {}}]))
    assert ctx.S["flight"]["live"]["flap_deg"] == pytest.approx(0.0)
    assert "flap_down" in pd.BUTTON_ACTIONS.values()
    assert "flap_down" in pd.js_config()["buttons"].values()


def test_the_detents_come_from_stage_5s_own_travel():
    """The travel was a literal 40 inside a slider tuple. It is an airframe
    property and it is answered where the surface is cut out."""
    ctx = _armed()
    ctx.S["controls"]["flap"]["max_deg"] = 25.0
    assert stk.detents(25.0) == (0.0, 10.0, 20.0, 25.0)
    ctx.render("flight", "fly")
    ctx.act("flight_run", True)
    for _ in range(6):
        ctx.act("flight_key", _Key("f"))
    assert ctx.S["flight"]["live"]["flap_deg"] == pytest.approx(25.0), \
        "the lever went past the airframe's own stop"


def test_the_trim_is_solved_AT_the_selected_flap():
    """"Re-trim with the flaps down" has to mean something: the alpha, the
    elevator and the thrust of THAT configuration."""
    ctx = _armed()
    ctx.act("flight_arm")
    clean = dict(ctx.S["flight"]["trim"])
    ctx.act("flight_run", True)
    for _ in range(4):
        ctx.act("flight_key", _Key("f"))
    ctx.act("flight_arm")
    down = dict(ctx.S["flight"]["trim"])
    assert down["flap_deg"] == pytest.approx(40.0)
    assert down["alpha_deg"] != pytest.approx(clean["alpha_deg"])


def test_the_glass_shows_it_and_says_nothing_when_none_is_fitted():
    """A flap position you cannot see is a configuration you forget you left
    out — and a read-out on an aeroplane with no flap is a lie in the place a
    real one would be."""
    assert 'id="hud-flap"' in hud.skeleton()
    kw = dict(speed=30.0, altitude=300.0, heading_deg=0.0, pitch_deg=0.0,
              roll_deg=0.0, alpha_deg=2.0, g=1.0, throttle=20.0,
              throttle_max=50.0)
    assert hud.state(**kw, flap_deg=None)["flap"] == ""
    assert hud.state(**kw, flap_deg=0.0)["flap"] == "FLAP UP"
    out = hud.state(**kw, flap_deg=20.0)
    assert out["flap"] == "FLAP 20"
    assert out["flap_col"] != hud.state(**kw, flap_deg=0.0)["flap_col"]


def test_the_flaps_own_numbers_are_not_hidden_behind_a_sibling():
    """The Control power box was drawn only when an aileron or a rudder
    existed, so a flaps-only configuration — exactly what somebody exploring
    flaps builds — hid the flap's own two derivatives."""
    ctx = _armed()
    ctx.S["controls"]["aileron"]["on"] = False
    ctx.S["controls"]["vertical"]["on"] = False
    ctx.act("controls_rebuild")
    ctx.render("controls", "derivatives")
    view = ctx.views[("controls", "derivatives")]
    # a few nicegui element CLASSES carry a `text` descriptor, so the
    # attribute comes back as a type on those and `in` would raise
    texts = " ".join(t for e in view.descendants()
                     for t in [getattr(e, "text", "")] if isinstance(t, str))
    assert "CZ_df" in texts, "the flap's own numbers are still hidden"
