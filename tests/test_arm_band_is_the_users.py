"""The SEARCHED horizontal separation is boxed by the user, not by the
calibration.

``tail.L_T_BOUNDS`` (3-8 m in air, 0.5-1.5 m in water) is the interval every
published tail result was MEASURED over. A stated arm has always been free
above the pitch solve's own floor (``tail.L_T_MIN_M`` = 0.10 m) for the
reason written beside that floor: how long the aeroplane is, is a design
decision, not an out-of-contract input.

A SEARCHED arm was not. The design box's ``l_t_m`` row went to
``api._apply_overrides``, which moves the box the SAMPLER draws from — while
``prob.bounds`` stayed at the calibrated band, so ``fg_tail``'s own bounds
check returned "bounds violation" for every draw outside it. Widening the row
therefore bought a page of refused candidates rather than a longer aeroplane,
and it did so silently: a NARROWER row behaved correctly, which is why the
one-directional ban went unseen.

So the row travels INTO the problem (``l_t_bounds_m`` -> ``tail.arm_row``),
exactly as the span's band does, and what is checked is only what the trim
solve genuinely needs: a real moment arm, low end first.

The physics says the same thing — a stated arm solves from 0.2 m to 40 m,
with the static margin growing monotonically and nothing degenerating — so
there was never a solver behind the ban to protect.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aerobo import api, hydrotail, tail, wingtail          # noqa: E402


def _row(built, label: str):
    return built.bounds[list(built.param_labels).index(label)]


def _at(built, label: str, value: float):
    """Evaluate the mid-box design with one row moved to ``value``."""
    x = built.bounds.mean(axis=1)
    x[list(built.param_labels).index(label)] = float(value)
    return built.evaluate(np.asarray(x))


# --------------------------------------------- 1. there is no solver behind it
@pytest.mark.parametrize("arm", [0.2, 1.0, 3.0, 8.0, 12.0, 20.0, 40.0])
def test_a_stated_arm_already_flies_far_outside_the_calibrated_band(arm):
    """The band was never a solvability limit: every one of these trims."""
    built = api.PROBLEM_SPECS["tail (fixed arm)"].build({}, {"l_t_m": arm},
                                                       None)
    out = built.evaluate(np.asarray(built.bounds.mean(axis=1)))

    assert out["reason"] == ""
    assert out["LoD"] > 0.0
    assert np.isfinite(out["SM"])


def test_a_longer_arm_moves_the_neutral_point_aft_monotonically():
    """...and it moves the one thing an arm is for, in the right direction."""
    sms = []
    for arm in (1.0, 3.0, 8.0, 20.0):
        built = api.PROBLEM_SPECS["tail (fixed arm)"].build(
            {}, {"l_t_m": arm}, None)
        sms.append(built.evaluate(
            np.asarray(built.bounds.mean(axis=1)))["SM"])

    assert all(b > a for a, b in zip(sms, sms[1:]))


# ------------------------------------------------ 2. the row IS the searched box
def test_an_untouched_run_is_the_published_box_bit_for_bit():
    built = api.PROBLEM_SPECS["tail"].build({}, {}, None)

    assert tuple(_row(built, "l_t_m")) == tail.L_T_BOUNDS
    assert built.problem.l_t_bounds_m is None


@pytest.mark.parametrize("name,band,probe", [
    ("tail", (1.0, 20.0), 17.0),                 # coupled lifting lines
    ("tail + winglet", (2.0, 15.0), 14.0),       # the nonplanar VLM
    ("hydrofoil + elevator", (0.3, 4.0), 3.5),   # water
])
def test_a_widened_row_is_searched_and_flown(name, band, probe):
    """The row the shell sends reaches the PROBLEM's own bounds, so a design
    outside the calibrated band is scored rather than refused."""
    built = api.PROBLEM_SPECS[name].build(
        {} if api.PROBLEM_SPECS[name].uses_mission else None, {},
        {"l_t_m": [band[0], band[1]]})

    assert tuple(_row(built, "l_t_m")) == band
    assert tuple(built.problem.bounds[
        list(built.problem.param_labels).index("l_t_m")]) == band

    out = _at(built, "l_t_m", probe)
    assert out["reason"] == ""              # ...not "bounds violation"
    assert np.isfinite(out["LoD"]) and out["LoD"] > 0.0


def test_a_narrowed_row_still_narrows():
    """The direction that always worked keeps working — the row is the box,
    both ways."""
    built = api.PROBLEM_SPECS["tail"].build({}, {}, {"l_t_m": [4.0, 6.0]})

    assert tuple(_row(built, "l_t_m")) == (4.0, 6.0)
    assert _at(built, "l_t_m", 5.0)["reason"] == ""
    # ...and the sampler cannot leave it, which is what a narrowed row means
    assert _at(built, "l_t_m", 7.0)["reason"] == "bounds violation"


# ------------------------------------------------------- 3. what IS still refused
def test_a_band_below_the_pitch_floor_is_refused_with_the_reason():
    with pytest.raises(ValueError, match="floor"):
        api.PROBLEM_SPECS["tail"].build({}, {}, {"l_t_m": [0.0, 8.0]})


def test_a_zero_width_band_is_refused_and_says_to_state_the_arm_instead():
    with pytest.raises(ValueError, match="zero-width"):
        tail.arm_row((5.0, 5.0))


def test_the_arm_cannot_be_both_stated_and_boxed():
    for cls, band in ((tail.TailProblem, (2.0, 9.0)),
                      (wingtail.WingTailProblem, (2.0, 9.0)),
                      (hydrotail.HydrofoilTailProblem, (0.4, 2.0))):
        with pytest.raises(ValueError, match="state it, or box it"):
            cls(l_t_fixed=5.0 if cls is not hydrotail.HydrofoilTailProblem
                else 1.0, l_t_bounds_m=band)


def test_the_default_is_the_family_s_own_band():
    assert tail.arm_row(None) == tail.L_T_BOUNDS
    assert tail.arm_row(None, hydrotail.L_T_BOUNDS) == hydrotail.L_T_BOUNDS


# --------------------------------------------------------------- 4. the shell
def test_the_v3_design_box_row_reaches_the_run():
    """Typing the band in stage 3's design box is the whole interaction: the
    row is tagged as the user's, and the run searches exactly it."""
    from gui.v3 import config
    from gui.v3.app import assemble

    ctx = assemble("air")
    ctx.act("set_choice", "tail", True)
    S = ctx.S
    assert "l_t_m" in config.effective_bounds(S)      # searched to begin with

    ctx.act("set_bound", "l_t_m", 0, 2.0)
    ctx.act("set_bound", "l_t_m", 1, 18.0)
    row, source = config.effective_bounds(S)["l_t_m"]
    assert row == [2.0, 18.0] and source == "user"

    built = api.PROBLEM_SPECS[S["wing"]["problem"]].build(
        config.mission_kwargs(S), config.flags(S), config.bounds_overrides(S))
    assert tuple(_row(built, "l_t_m")) == (2.0, 18.0)
    assert _at(built, "l_t_m", 17.0)["reason"] == ""


def test_the_card_says_where_the_calibration_ends():
    """A band outside the measured interval is EXTRAPOLATION, and the shell
    says so beside it — the same sentence a stated arm outside it gets."""
    from gui.v3 import config
    from gui.v3.app import assemble

    ctx = assemble("air")
    ctx.act("set_choice", "tail", True)
    S = ctx.S

    def _hints():
        return " ".join(
            getattr(e, "text", "") or ""
            for e in ctx.views[("wing", "type")].descendants())

    assert "CALIBRATED" not in _hints()
    ctx.act("set_bound", "l_t_m", 1, 18.0)
    assert config.effective_bounds(S)["l_t_m"][0][1] == 18.0
    assert "CALIBRATED" in _hints()


# ------------------------------- 5. the STATED arm the user typed is not reset
def test_a_typed_arm_outside_the_band_survives_the_toggle():
    """The last place the calibration was still a ban: ``set_tail_arm``
    re-defaulted any held value outside the published box, so a 1.5 m arm —
    a small aeroplane, exactly what the model is for — was 5.5 m again the
    moment the toggle was touched."""
    from gui.v3 import config, session
    from gui.v3.app import assemble

    ctx = assemble("air")
    ctx.act("set_choice", "tail", True)
    S = ctx.S
    session.set_tail_arm(S, "fixed")
    ctx.act("set_trim_number", "tail_arm_m", 1.5)

    session.set_tail_arm(S, "free")
    notes = session.set_tail_arm(S, "fixed")

    assert S["wing"]["choices"]["tail_arm_m"] == 1.5
    assert config.flags(S)["l_t_m"] == pytest.approx(1.5)
    assert any("kept at 1.5" in n and "extrapolation" in n for n in notes)


def test_an_untouched_default_is_still_re_defaulted_across_media():
    """What the snap was FOR survives: 5.5 m is the air shell's own default,
    not anybody's answer, and on the water family it is a hull four times the
    craft."""
    from gui.v3 import session
    from gui.v3.app import assemble

    ctx = assemble("water")
    ctx.act("set_choice", "tail", True)
    S = ctx.S
    notes = session.set_tail_arm(S, "fixed")

    assert S["wing"]["choices"]["tail_arm_m"] == 1.0     # mid of 0.5-1.5
    assert any("starting point, not a limit" in n for n in notes)


def test_a_typed_arm_crosses_media_untouched():
    from gui.v3 import session
    from gui.v3.app import assemble

    ctx = assemble("water")
    ctx.act("set_choice", "tail", True)
    S = ctx.S
    S["wing"]["choices"]["tail_arm_m"] = 3.0
    session.set_tail_arm(S, "fixed")

    assert S["wing"]["choices"]["tail_arm_m"] == 3.0


def test_an_arm_below_the_pitch_floor_is_still_re_defaulted():
    """The one refusal that is physics: at no arm the trim solve is
    singular."""
    from gui.v3 import session
    from gui.v3.app import assemble

    ctx = assemble("air")
    ctx.act("set_choice", "tail", True)
    S = ctx.S
    S["wing"]["choices"]["tail_arm_m"] = 0.5 * tail.L_T_MIN_M
    session.set_tail_arm(S, "fixed")

    assert S["wing"]["choices"]["tail_arm_m"] == 5.5


def test_the_stated_arm_field_carries_no_cap():
    """The field itself must not refuse what the solver takes — the V1/V2
    card capped it at min=3/max=8 while the solver flew 0.2-40 m."""
    from gui.v3.app import assemble

    ctx = assemble("air")
    ctx.act("set_choice", "tail", True)
    ctx.act("set_tail_arm", "fixed")          # the toggle, so the view redraws
    ctx.act("set_trim_number", "tail_arm_m", 1.2)

    # the value the RUN gets — the field itself is deliberately not rebuilt
    # under the cursor (VALUE_KEYS), so the browser holds the typed digits
    assert ctx.S["wing"]["choices"]["tail_arm_m"] == 1.2

    fields = [e for e in ctx.views[("wing", "type")].descendants()
              if type(e).__name__ == "Number"]
    assert fields
    for f in fields:
        props = getattr(f, "_props", {})
        assert props.get("min") is None and props.get("max") is None


# ------------------------------------------- 6. the height row is what it says
def test_the_height_row_shown_is_the_height_row_searched():
    """``z_t_m``'s box is a FRACTION OF THE SPAN, so on a resized wing the
    static band (the 10 m trim wing's 0.5-3 m) is not the one the run uses.
    A design box quoting it let a user type a bound every draw then failed
    on."""
    from gui.v3 import config, session
    from gui.v3.app import assemble

    ctx = assemble("air")
    ctx.act("set_choice", "tail", True)
    ctx.act("set_choice", "tail_height", "free")
    S = ctx.S
    session.set_wing_loading(S, 45.0)
    assert session.set_span_m(S, 23.6)

    shown = config.effective_bounds(S)["z_t_m"][0]
    built = api.PROBLEM_SPECS[S["wing"]["problem"]].build(
        config.mission_kwargs(S), config.flags(S), config.bounds_overrides(S))
    searched = _row(built, "z_t_m")

    assert shown == pytest.approx(list(searched))
    # ...which is the measured clearance on THIS span, not on the trim wing's
    assert shown[0] == pytest.approx(tail.DZ_FRAC * 23.6)
