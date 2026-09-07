"""The trim layout: the CG is stated, the separation is stated OR searched,
and the card says where the NEUTRAL POINT is.

Three things this file exists to hold.

**The stability criterion is x_cg < x_np, not "the CG is ahead of the wing
AC".** A surface behind the wing drags the whole aircraft's neutral point aft
of the wing's own aerodynamic centre (x_np = (x_w a_w + x_t a_t)/(a_w + a_t)
on the COUPLED lift-curve slopes, tail.py), which is exactly why a
conventional aft-tail aircraft is loaded with its CG BEHIND the wing AC and is
still stable. Where the surface is in front — the canard — the neutral point
moves ahead of the wing AC and the family's calibrated CG goes with it. So the
honest readout is the MARGIN, and the shell quotes it: a CG the user types aft
of x_np is reported unstable on the card, with the same sign the run's own
static-margin constraint gives it.

**The CG and the horizontal separation are two questions, and the card asks
both.** The CG is always determined: a field in every state, in metres aft of
the wing AC (negative = forward), opening on the value the run flies when
nobody states one — which is the LAYOUT's in air (``tail.X_CG_BY_TYPE``, +0.25
m on a conventional tail, −0.40 m on a canard) and a FRACTION OF THE ARM in
water (``hydrotail.X_CG_FRAC``). Beside it the separation has the same two
answers the vertical one has (``session.set_tail_arm``): state it, and
``l_t_m`` travels as a flag on the family's fixed-arm twin; optimise it, and
``l_t_m`` stays a design variable whose min and max are its design-box row.
Nothing clears the CG any more — ``x_cg_m`` travels on both twins, and the
card that used to promise the calibrated CG applied is gone with the either/or
that made the promise.

**A stated arm has to be one its own family can fly.** The solvers refuse an
arm outside their calibrated box (``tail.L_T_BOUNDS`` = 3–8 m,
``hydrotail.L_T_BOUNDS`` = 0.5–1.5 m), and the shell's held value is an AIR
length: choosing "you state it" under water sent 5.5 m and every build after
it raised, with the margin read-out silently gone. Switching to a stated arm
snaps a value the new family cannot take onto the middle of the box it can.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aerobo import api, tail            # noqa: E402


def _shell(medium: str = "air"):
    from gui.v3.app import assemble

    return assemble(medium)


def _asked(view) -> list[str]:
    """The labels of the CONTROLS in a view — the class the card puts on a
    field's or a row's label, which is what tells a question apart from the
    read-out that quotes the same word."""
    return [(getattr(e, "text", "") or "").strip() for e in view.descendants()
            if "field-label" in (getattr(e, "_classes", None) or [])]


def _with_tail(medium: str = "air"):
    ctx = _shell(medium)
    ctx.act("set_choice", "tail", True)
    assert api.TAIL_CG_KEY in api.PROBLEM_SPECS[ctx.S["wing"]["problem"]].flags
    return ctx


# ------------------------------------------------------- the physics itself
def test_an_aft_tail_puts_the_neutral_point_behind_the_wing_ac():
    """The reason a CG aft of the wing AC is not an instability: the tail
    moves the station the CG has to stay in front of."""
    prob = api.PROBLEM_SPECS["tail"].build({}, {}, None)
    out = prob.evaluate(np.asarray(prob.bounds.mean(axis=1)))

    assert out["x_cg"] > 0.0                      # aft of the wing AC...
    assert out["x_np"] > out["x_cg"]              # ...and still ahead of x_np
    assert out["SM"] == pytest.approx(
        (out["x_np"] - out["x_cg"]) / out["mac"])
    assert out["SM"] > 0.0                        # stable


def test_a_canard_is_calibrated_the_other_way_round():
    """Where the surface is in FRONT the neutral point moves ahead of the
    wing AC, so the calibrated CG is ahead of it too — the rule "CG in front
    of the wing AC" is this layout's, not every layout's."""
    assert tail.X_CG_BY_TYPE["canard"] < 0.0
    assert tail.X_CG_BY_TYPE["conventional"] > 0.0


def test_a_cg_behind_the_neutral_point_is_refused_by_the_run():
    """The card's warning and the optimiser's constraint are the same sign,
    so a design the shell calls unstable is one the run rejects."""
    prob = api.PROBLEM_SPECS["tail"].build({}, {api.TAIL_CG_KEY: 0.9}, None)
    x = np.asarray(prob.bounds.mean(axis=1))
    out = prob.evaluate(x)
    _f, g = prob.callable(x)

    assert out["x_cg"] > out["x_np"]              # behind the neutral point
    assert out["SM"] < 0.0                        # ...i.e. unstable
    assert g == pytest.approx(out["SM"] - out["SM_min"])
    assert g < 0.0                                # and refused


# ------------------------------------------------------------ the read-out
def test_the_shell_reports_the_margin_the_run_would_fly():
    from gui.v3 import config, session

    ctx = _with_tail()
    S = ctx.S
    st = session.pitch_stability(S)
    assert st is not None

    built = api.PROBLEM_SPECS[S["wing"]["problem"]].build(
        config.mission_kwargs(S), config.flags(S), config.bounds_overrides(S))
    out = built.evaluate(np.asarray(built.bounds.mean(axis=1)))
    assert st["x_np"] == pytest.approx(out["x_np"])
    assert st["SM"] == pytest.approx(out["SM"])
    assert st["stable"] and st["accepted"]


def test_a_family_with_no_pitch_balance_reports_no_margin():
    """The wing-alone and tandem problems carry no CG, no Cm and no
    static-margin constraint. The shell says nothing rather than inventing
    one."""
    from gui.v3 import session

    ctx = _shell()
    assert session.pitch_stability(ctx.S) is None      # wing alone
    ctx.act("set_choice", "system", "tandem")
    assert session.pitch_stability(ctx.S) is None


def test_a_typed_cg_moves_the_margin_and_can_go_forward():
    from gui.v3 import session

    ctx = _with_tail()
    S = ctx.S
    base = session.pitch_stability(S)

    # 0.7 m and not the 0.9 this used to type: the second surface is always
    # mounted to push DOWN now (gui.v3.config.TAIL_MOUNT), and past about
    # 0.8 m of aft CG the up-load that layout asks of an inverted section
    # runs the trim incidence off the end of the polar — the mid-box design
    # stops evaluating and there is no margin to read. 0.7 m is the same
    # statement (aft, unstable, SM -0.224) at a design that exists.
    ctx.act("set_choice", "tail_cg_m", 0.7)
    unstable = session.pitch_stability(S)
    assert unstable["x_cg"] == pytest.approx(0.7)
    assert unstable["SM"] < 0.0 and not unstable["stable"]

    # NEGATIVE is forward of the wing AC, and it is allowed: that is the
    # tailless/canard loading, and here it simply buys margin
    ctx.act("set_choice", "tail_cg_m", -0.4)
    forward = session.pitch_stability(S)
    assert forward["x_cg"] == pytest.approx(-0.4)
    assert forward["SM"] > base["SM"]


# ------------------------------------ the separation: stated, or searched
def test_stating_the_separation_selects_the_fixed_arm_twin():
    from gui.v3 import config, session

    ctx = _with_tail()
    S = ctx.S
    assert "set_tail_arm" in ctx.actions      # not a silently-ignored name
    assert "l_t_m" in config.effective_bounds(S)       # searched to begin with

    ctx.act("set_tail_arm", "fixed")
    assert "(fixed arm)" in S["wing"]["problem"]
    assert config.flags(S)["l_t_m"] == pytest.approx(5.5)
    assert "l_t_m" not in config.effective_bounds(S)   # stated, not searched

    ctx.act("set_choice", "tail_arm_m", 7.0)
    moved = session.pitch_stability(S)
    assert config.flags(S)["l_t_m"] == pytest.approx(7.0)
    # a longer arm moves the neutral point aft — the separation IS a
    # stability control, which is why the card reports the margin under it
    assert moved["x_np"] > session.pitch_stability(
        _with_tail().S)["x_np"]


def test_stating_the_separation_keeps_the_stated_cg():
    """The two are independent questions: where the mass is, and how far back
    the surface sits. ``x_cg_m`` travels on both twins, so the CG has no
    reason to be thrown away when the arm stops being searched — it used to
    be, because the card then claimed the calibrated CG applied."""
    from gui.v3 import config

    ctx = _with_tail()
    S = ctx.S
    ctx.act("set_choice", "tail_cg_m", 0.9)
    assert config.flags(S)[api.TAIL_CG_KEY] == pytest.approx(0.9)

    ctx.act("set_tail_arm", "fixed")
    assert S["wing"]["choices"]["tail_cg_m"] == pytest.approx(0.9)
    assert config.flags(S)[api.TAIL_CG_KEY] == pytest.approx(0.9)
    assert config.flags(S)["l_t_m"] == pytest.approx(5.5)   # both ends, sent

    # ...and going back gives the arm to the optimiser again, CG intact
    ctx.act("set_tail_arm", "free")
    assert "l_t_m" not in config.flags(S)
    assert "l_t_m" in config.effective_bounds(S)
    assert config.flags(S)[api.TAIL_CG_KEY] == pytest.approx(0.9)


def test_both_ends_of_the_lever_reach_the_run_together():
    """Not just the flags dict: the BUILT problem carries the stated CG and
    the stated arm at once, on a design vector the arm has left."""
    from gui.v3 import config

    ctx = _with_tail()
    S = ctx.S
    ctx.act("set_tail_arm", "fixed")
    ctx.act("set_choice", "tail_arm_m", 7.0)
    ctx.act("set_choice", "tail_cg_m", 0.42)

    built = api.PROBLEM_SPECS[S["wing"]["problem"]].build(
        config.mission_kwargs(S), config.flags(S), config.bounds_overrides(S))
    assert built.problem.l_t_fixed == pytest.approx(7.0)
    assert built.problem.x_cg == pytest.approx(0.42)
    assert "l_t_m" not in built.param_labels          # stated, not searched


def test_the_searched_arm_is_the_design_boxs_own_row():
    """"If free, the design box determines min and max" — the box row is not
    a note about the search, it IS the interval the run is built with."""
    from gui.v3 import config

    ctx = _with_tail()
    S = ctx.S
    assert S["wing"]["choices"]["tail_arm"] == "free"      # the shell's own
    assert config.effective_bounds(S)["l_t_m"][0] == [3.0, 8.0]

    ctx.act("set_bound", "l_t_m", 0, 4.0)
    ctx.act("set_bound", "l_t_m", 1, 6.0)
    assert config.bounds_overrides(S)["l_t_m"] == [4.0, 6.0]

    built = api.PROBLEM_SPECS[S["wing"]["problem"]].build(
        config.mission_kwargs(S), config.flags(S), config.bounds_overrides(S))
    row = built.bounds[list(built.param_labels).index("l_t_m")]
    assert list(row) == pytest.approx([4.0, 6.0])


def test_the_arm_is_asked_in_exactly_one_place():
    """It used to be two: a free/fixed toggle in the tail's configuration
    block and the CG two cards down, so the margin that came out belonged to
    neither control. The arm is a question again — asked ONCE, here, beside
    the CG and the margin the pair moves."""
    from gui.v3 import session

    ctx = _with_tail()
    ctx.render("wing", "type")
    labels = [(getattr(e, "text", "") or "").strip()
              for e in ctx.views[("wing", "type")].descendants()]
    # the OLD control's own option labels, matched exactly: the phrase still
    # appears inside the hint that says the separation is being searched,
    # and a sentence is not a second control
    assert "arm is a design variable" not in labels
    assert "fixed arm" not in labels
    assert "stated as" not in labels           # the either/or is gone
    # ...counted as CONTROLS (class field-label), not as text: "CG" is also
    # the caption of the read-out underneath, and a read-out is not a place
    # a question is asked
    asked = _asked(ctx.views[("wing", "type")])
    assert asked.count("separation") == 1
    assert asked.count("CG") == 1
    # the margin is quoted beside it, which is the number neither end of the
    # lever states on its own
    assert any(t.strip() == "neutral point" for t in labels)
    assert any(t.strip() == "static margin" for t in labels)
    assert session.pitch_stability(ctx.S) is not None


def test_a_typed_cg_moves_the_readout_without_rebuilding_the_field():
    """The three numbers that place the surface are VALUE_KEYS: typing one
    may not rebuild the view its own field is in (the focus trap), which is
    why the margin read-out is redrawn on its own. It went stale once — the
    CG moved the run while the neutral point beside it quoted the old one."""
    ctx = _with_tail()
    ctx.render("wing", "type")

    def _texts():
        return [(getattr(e, "text", "") or "").strip()
                for e in ctx.views[("wing", "type")].descendants()]

    def _fields():
        return [e for e in ctx.views[("wing", "type")].descendants()
                if type(e).__name__ == "Number"]

    # quoted from the run, never from a constant restated here: the family's
    # calibration is the solver's to move
    from gui.v3 import session

    was = session.pitch_stability(ctx.S)
    before, fields = _texts(), _fields()
    assert f"{was['x_cg']:.3f}" in before and f"{was['SM']:+.3f}" in before

    ctx.act("set_trim_number", "tail_cg_m", 0.7)     # aft, and evaluable
    moved = session.pitch_stability(ctx.S)
    after = _texts()
    assert "0.700" in after and f"{moved['SM']:+.3f}" in after
    assert moved["SM"] < was["SM"] and not moved["stable"]
    assert any("UNSTABLE" in t for t in after)
    # ...and the field the number was typed into is the same object
    assert [id(f) for f in _fields()] == [id(f) for f in fields]


# ------------------------------------------- the OTHER half of where it sits
def test_the_vertical_separation_can_be_a_design_variable():
    """Both answers are real, and both are reachable from the one card: a
    stated height sends ``z_t_m`` as a flag, an optimised one selects the
    free twin and puts ``z_t_m`` in the design vector with its own box."""
    from gui.v3 import config, session

    ctx = _with_tail()
    S = ctx.S
    assert "z_t_m" not in config.effective_bounds(S)

    ctx.act("set_choice", "tail_height", "free")
    assert "[free height]" in S["wing"]["problem"]
    assert "z_t_m" in api.PROBLEM_SPECS[S["wing"]["problem"]].param_labels
    row = config.effective_bounds(S)["z_t_m"]
    assert row[0][0] < row[0][1]                   # a real band, searched
    assert api.TAIL_HEIGHT_KEY not in config.flags(S)   # ...not also stated
    # it is not cosmetic: out of the trailing sheet the surface sees less
    # downwash, so the margin the card above reports moves with it
    assert session.pitch_stability(S) is not None

    ctx.act("set_choice", "tail_height", "fixed")
    assert "[free height]" not in S["wing"]["problem"]
    assert "z_t_m" not in config.effective_bounds(S)


def test_a_stated_height_reaches_the_run_and_moves_the_margin():
    """It is a real physical choice, not a label: dz enters the coupled
    solve's horseshoe kernel, so a surface further out of the wing's trailing
    sheet sees less downwash and the neutral point moves aft with it."""
    from gui.v3 import config, session

    ctx = _with_tail()
    S = ctx.S
    assert api.TAIL_HEIGHT_KEY in api.PROBLEM_SPECS[S["wing"]["problem"]].flags
    base = session.pitch_stability(S)

    ctx.act("set_trim_number", "tail_height_m", 1.5)
    assert config.flags(S)[api.TAIL_HEIGHT_KEY] == pytest.approx(1.5)
    raised = session.pitch_stability(S)
    assert raised["x_np"] > base["x_np"]
    assert raised["SM"] > base["SM"]


def test_the_height_field_pre_fills_with_what_the_run_flies():
    """The layout's OWN height, read off a built problem — so stating one
    starts from the truth instead of from an empty field."""
    ctx = _with_tail()
    ctx.render("wing", "type")
    values = [e.value for e in ctx.views[("wing", "type")].descendants()
              if type(e).__name__ == "Number"]
    from aerobo import tail as tailmod

    assert tailmod.DZ_FRAC * 10.0 in values           # 0.05 b on the 10 m wing


def test_the_vertical_separation_is_asked_beside_the_horizontal_one():
    """One card holds where the surface SITS — both components — and the
    static margin the pair implies. It used to be split across two."""
    ctx = _with_tail()
    ctx.render("wing", "type")
    asked = _asked(ctx.views[("wing", "type")])
    assert asked.count("CG") == 1                     # where the mass is
    assert asked.count("separation") == 1             # the horizontal one
    assert asked.count("height") == 1                 # the vertical one
    # ...and BOTH separations offer the design variable, in the same words:
    # one pattern asked twice, not two controls that happen to be adjacent
    options = [[o["label"] for o in (e._props.get("options") or [])]
               for e in ctx.views[("wing", "type")].descendants()
               if type(e).__name__ == "Toggle"]
    assert options.count(["you state it", "optimise it"]) == 2


def test_a_t_tail_is_still_asked_and_says_why_it_cannot_answer():
    """Its height IS its fin span. Dropping the row entirely would say "there
    is no height question here", which is a different — and false —
    statement."""
    ctx = _with_tail()
    ctx.act("set_choice", "tail_type", "t_tail")
    ctx.render("wing", "type")
    texts = [(getattr(e, "text", "") or "") for e in
             ctx.views[("wing", "type")].descendants()]
    assert any(t.strip() == "height" for t in texts)
    assert any("fin span" in t for t in texts)
    assert ctx.S["wing"]["choices"]["tail_height"] != "free"


def test_the_water_elevator_carries_the_same_two_questions():
    from gui.v3 import config, session

    ctx = _with_tail("water")
    S = ctx.S
    assert session.pitch_stability(S) is not None
    ctx.act("set_tail_arm", "fixed")
    assert "fixed arm" in S["wing"]["problem"]
    assert "l_t_m" in config.flags(S)
    ctx.act("set_choice", "tail_cg_m", 0.05)
    assert config.flags(S)[api.TAIL_CG_KEY] == pytest.approx(0.05)


def test_a_stated_arm_lands_inside_the_box_its_own_family_flies():
    """The held arm is an AIR length (5.5 m) and the water family is
    calibrated over 0.5–1.5 m, so "you state it" under water used to send an
    arm every build then refused — the margin read-out simply disappeared and
    the only way out was the toggle the user had just clicked."""
    from gui.v3 import config, session

    ctx = _with_tail("water")
    S = ctx.S
    assert session.published_arm_box(S) == (0.5, 1.5)

    ctx.act("set_tail_arm", "fixed")
    assert config.flags(S)["l_t_m"] == pytest.approx(1.0)     # snapped
    built = api.PROBLEM_SPECS[S["wing"]["problem"]].build(
        config.mission_kwargs(S), config.flags(S), config.bounds_overrides(S))
    assert built.problem.l_t_fixed == pytest.approx(1.0)
    assert session.pitch_stability(S) is not None             # and it reports

    # an arm the family CAN take is left exactly where the user put it
    ctx.act("set_choice", "tail_arm_m", 1.4)
    ctx.act("set_tail_arm", "free")
    ctx.act("set_tail_arm", "fixed")
    assert config.flags(S)["l_t_m"] == pytest.approx(1.4)


def test_clearing_the_stated_arm_keeps_the_one_the_family_can_fly():
    """The one number on this card that may not be BLANK. The others mean
    "leave it to the family" empty and the flags honour that; a stated arm is
    sent whenever it is stated, so an emptied field reached ``float(None)``,
    took ``config.flags`` down with it, and the render that followed dropped
    the CG, the toggle and the field itself off the page — with no widget
    left to type the replacement into."""
    from gui.v3 import config

    ctx = _with_tail()
    S = ctx.S
    ctx.act("set_tail_arm", "fixed")
    ctx.render("wing", "type")

    ctx.act("set_trim_number", "tail_arm_m", None)      # field backspaced out
    assert S["wing"]["choices"]["tail_arm_m"] == pytest.approx(5.5)
    assert config.flags(S)["l_t_m"] == pytest.approx(5.5)
    ctx.render("wing", "type")
    assert _asked(ctx.views[("wing", "type")]).count("CG") == 1

    # ...and a real number still lands
    ctx.act("set_trim_number", "tail_arm_m", 6.5)
    assert config.flags(S)["l_t_m"] == pytest.approx(6.5)


def test_a_long_arm_flies_and_says_it_is_extrapolating():
    """The separation is COMPLETELY FREE — any positive length the geometry
    can hold — and the note beside it says where the family's measurements
    stop rather than refusing to leave them.

    The note is a READ-OUT of the field above it, so it follows a typed
    number: ``tail_arm_m`` is a VALUE_KEY and may not rebuild the type view,
    and a note that only comes back with the whole view never appears at the
    moment it is needed."""
    from gui.v3 import session

    ctx = _with_tail()
    ctx.act("set_tail_arm", "fixed")
    ctx.render("wing", "type")

    def _texts():
        return [(getattr(e, "text", "") or "")
                for e in ctx.views[("wing", "type")].descendants()]

    assert not any("calibrated over" in t for t in _texts())
    ctx.act("set_trim_number", "tail_arm_m", 12.0)      # past the 3–8 m box
    # it FLIES: the geometry is built and solved at the stated arm
    st = session.pitch_stability(ctx.S)
    assert st is not None and st["SM"] > 0.0
    assert any("12 m is outside the 3 – 8 m band" in t for t in _texts())
    assert any("It will fly" in t for t in _texts())

    ctx.act("set_trim_number", "tail_arm_m", 6.0)       # ...and it goes away
    assert not any("calibrated over" in t for t in _texts())


def test_empty_means_the_familys_value_even_under_a_typed_cg():
    """The sentence beside the field says what EMPTY would fly. Quoting the
    run's own CG made it quote the number the user had just typed, so the
    card read "empty = 0.900 m" beside a stated 0.900."""
    from gui.v3 import session

    ctx = _with_tail()
    S = ctx.S
    own = session.family_cg(S)

    ctx.act("set_choice", "tail_cg_m", 0.7)
    assert session.pitch_stability(S)["x_cg"] == pytest.approx(0.7)
    assert session.family_cg(S) == pytest.approx(own)     # unchanged by it
    assert S["wing"]["choices"]["tail_cg_m"] == pytest.approx(0.7)  # restored

    ctx.render("wing", "type")
    texts = [(getattr(e, "text", "") or "")
             for e in ctx.views[("wing", "type")].descendants()]
    empty = [t for t in texts if "Empty = this family's" in t]
    assert empty and f"{own:.3f}" in empty[0] and "0.700" not in empty[0]


def test_the_water_cg_field_follows_the_arm_it_is_a_fraction_of():
    """In water the CG opens on ``X_CG_FRAC · l_t``, so the arm's own field
    moves it. It is not the field being typed into, so it is redrawn — and it
    went stale otherwise, quoting the CG of the arm before last."""
    from gui.v3 import session

    def _fields(ctx):
        return [e.value for e in ctx.views[("wing", "type")].descendants()
                if type(e).__name__ == "Number"]

    ctx = _with_tail("water")
    ctx.act("set_tail_arm", "fixed")
    ctx.render("wing", "type")
    assert session.family_cg(ctx.S) in _fields(ctx)

    ctx.act("set_trim_number", "tail_arm_m", 1.4)
    moved = session.family_cg(ctx.S)
    assert moved == pytest.approx(0.15 / 1.0 * 1.4, rel=1e-6)
    assert moved in _fields(ctx)


def test_the_cg_field_opens_on_the_cg_the_run_flies():
    """Not on a constant, and not on a problem built with no flags: in air the
    CG follows the LAYOUT (a canard is loaded −0.40 m, FORWARD) and in water
    it is a fraction of the arm. The field is what the user 'determines', so
    it has to open on the number the margin beside it is quoting."""
    from gui.v3 import session

    def _fields(ctx):
        return [e.value for e in ctx.views[("wing", "type")].descendants()
                if type(e).__name__ == "Number"]

    from aerobo import hydrotail

    ctx = _with_tail()
    ctx.act("set_choice", "tail_type", "canard")
    ctx.render("wing", "type")
    canard = tail.X_CG_BY_TYPE["canard"]
    assert canard < 0.0 < tail.X_CG_BY_TYPE["conventional"]   # loaded FORWARD
    assert session.pitch_stability(ctx.S)["x_cg"] == pytest.approx(canard)
    assert canard in _fields(ctx)
    assert tail.X_CG_BY_TYPE["conventional"] not in _fields(ctx)

    wet = _with_tail("water")
    wet.render("wing", "type")
    lo, hi = session.published_arm_box(wet.S)
    flown = session.pitch_stability(wet.S)["x_cg"]
    # a FRACTION of the arm, and the arm to quote is the middle of the box
    # the run searches — half of it was the old stand-in
    assert flown == pytest.approx(hydrotail.X_CG_FRAC * 0.5 * (lo + hi))
    assert flown in _fields(wet)


def test_an_untouched_card_still_states_no_cg_of_its_own():
    """"Always determined" is the field always showing what the run flies —
    not the shell writing that number into the session. A written one would
    make an untouched run stop being the published one, and in water it would
    sever the CG from the arm it is a fraction of."""
    from gui.v3 import config, session, widgets

    ctx = _with_tail()
    S = ctx.S
    assert S["wing"]["choices"]["tail_cg_m"] is None
    assert api.TAIL_CG_KEY not in config.flags(S)
    ctx.render("wing", "type")

    # the browser echoing the DRAWN value back is not a statement either
    drawn = widgets.shown(session.pitch_stability(S)["x_cg"])
    ctx.act("set_trim_number", "tail_cg_m", drawn)
    assert S["wing"]["choices"]["tail_cg_m"] is None
    assert api.TAIL_CG_KEY not in config.flags(S)
