"""The measured design box is STABLE, and it contains the design on screen.

Three defects, all reported by the user in the same breath — *"sometimes the
design box is 'mission' and sometimes 'recommended'… I don't want it to
change"*, and *"when the user finishes the airfoil the design box should
already be generated (the mission span should be in the range of
recommended)"*. Reproduced on the shipped air mission, before any of this:

    on arrival               b_m  8.124 – 18.634  (mission)
    after the measurement    b_m 11.509 – 13.925  (recommended)  <- 10 m OUT
    after a tip-device menu  b_m  8.124 – 18.634  (mission)
    re-measured              b_m  9.944 – 15.989  (recommended)

1. the measured box REFUSED the mission's own wing (10 m, and the row opened
   at 11.509) — see ``session.hold_the_stated_design``;
2. every builder menu threw the measurement away and took a second to make
   another, so the row flicked full-width and back twice per press — see
   ``session.relax_measured_bands`` and ``wing._unnarrowed_cfg``;
3. none of it happened until stage 3 was OPENED, so the box narrowed under a
   user who had just walked into it — see ``wing._auto_ready``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _shell(**choices):
    from gui.v3.app import assemble

    ctx = assemble("air")
    ctx.act("accept_mission")
    for key, value in choices.items():
        ctx.act("set_choice", key, value)
    ctx.act("set_planform", "free")
    ctx.render("wing", "box")
    return ctx


def _band(ctx, row="b_m"):
    from gui.v3 import config

    got = config.effective_bounds(ctx.S)[row]
    return [float(got[0][0]), float(got[0][1])], got[1]


# ------------------------------------------------ 1. the wing on screen
def test_the_measured_box_contains_the_span_the_mission_states(capsys):
    """The measurement is the min/max of the draws that FLEW — it owes the
    stated wing nothing, and on this mission it came back 11.509 – 13.925 m
    over a mission stating 10 m. A box that refuses its own aeroplane while
    the row reads "recommended" is advice nobody can act on."""
    from gui.v3 import session

    ctx = _shell(tail=True)
    ctx.act("auto_recommend")
    band, source = _band(ctx)
    assert source == "recommended"
    span = session.nominal_span(ctx.S)
    assert band[0] <= span <= band[1], (band, span)
    # ...and it still NARROWED: holding the stated wing is not giving up
    assert band[1] < session.size_band_defaults(ctx.S)["b_m"][1]
    assert "Traceback" not in capsys.readouterr().err


def test_a_typed_span_is_inside_the_measured_box_too(capsys):
    """The same rule, against a number the user actually typed."""
    from gui.v3 import session

    ctx = _shell(tail=True)
    assert session.set_size_from_span_ar(ctx.S, 7.0, 9.0)
    session.sync_wing_from_mission(ctx.S)
    ctx.act("auto_recommend")
    band, _src = _band(ctx)
    assert band[0] <= session.nominal_span(ctx.S) <= band[1], band
    assert "Traceback" not in capsys.readouterr().err


def test_holding_the_design_never_widens_past_the_shell_s_own_band(capsys):
    """It holds the mission's own numbers, every one of them already inside
    the row the shell published — so this can never turn into a search
    WIDER than the box the stage opened on."""
    from gui.v3 import session

    ctx = _shell(tail=True)
    opened = session.size_band_defaults(ctx.S)
    ctx.act("auto_recommend")
    for row, band in opened.items():
        now, _src = _band(ctx, row)
        assert now[0] >= band[0] - 1e-9, (row, now, band)
        assert now[1] <= band[1] + 1e-9, (row, now, band)
    assert "Traceback" not in capsys.readouterr().err


# ------------------------------------------------ 2. it does not flicker
def test_a_builder_menu_relaxes_the_measurement_instead_of_dropping_it(
        capsys):
    """The user's own specification, in numbers:

        after the measurement    b_m 10.0  – 13.925
        after a tip-device menu  b_m 8.124 – 13.925

    the bottom back to the mission's band, the top kept.
    """
    from gui.v3 import session

    ctx = _shell(tail=True)
    ctx.act("auto_recommend")
    measured, source = _band(ctx)
    assert source == "recommended"
    mission = session.size_band_defaults(ctx.S)["b_m"]

    ctx.act("set_winglet", "canted")
    after, source_after = _band(ctx)
    # the row does not flick back to full width, and does not change its name
    assert source_after == "recommended"
    assert after[1] == measured[1]
    assert after[0] == session.size_band_defaults(ctx.S)["b_m"][0]
    assert after[0] < measured[0]
    assert after[1] < mission[1]
    assert "Traceback" not in capsys.readouterr().err


def test_carrying_a_band_through_a_menu_is_not_a_ratchet(capsys):
    """Press three menus and the box may not creep. The next measurement is
    taken over ``size_band_defaults``, never over the carried band."""
    ctx = _shell(tail=True)
    ctx.act("auto_recommend")
    first, _ = _band(ctx)
    for shape in ("canted", "vertical", "none"):
        ctx.act("set_winglet", shape)
    ctx.act("set_winglet", "none")
    ctx.act("auto_recommend")
    again, _ = _band(ctx)
    # measured over the same family and the same mission as the first pass
    assert again == first
    assert "Traceback" not in capsys.readouterr().err


def test_a_mission_change_still_drops_the_measurement_whole(capsys):
    """A builder menu is not a mission. Where the mission's OWN band has
    moved, ``[mission_lo, measured_hi]`` is two aeroplanes spliced together,
    so the measured rows go entirely (``drop_recommended_bounds``)."""
    from gui.v3 import session

    ctx = _shell(tail=True)
    ctx.act("auto_recommend")
    assert _band(ctx)[1] == "recommended"
    assert session.set_size_from_span_ar(ctx.S, 1.0, 8.0)
    session.sync_wing_from_mission(ctx.S)
    band, source = _band(ctx)
    assert source == "mission", (band, source)
    assert band == list(session.size_band_defaults(ctx.S)["b_m"])
    assert band[0] <= 1.0 <= band[1], band
    assert "Traceback" not in capsys.readouterr().err


def test_a_medium_change_carries_nothing(capsys):
    """A 10 m air wing is not an opinion about a 1.2 m hydrofoil — the same
    rule the chosen section and the stated span already go by."""
    from gui.v3 import session

    ctx = _shell(tail=True)
    ctx.act("auto_recommend")
    assert _band(ctx)[1] == "recommended"
    ctx.act("set_choice", "medium", "water")
    assert "recommended" not in session.bounds_source(ctx.S).values()
    assert "Traceback" not in capsys.readouterr().err


def test_a_row_the_user_typed_is_not_restored_by_the_carry(capsys):
    """The carry writes ``"recommended"`` rows over rows the SHELL owns. A
    row the user has answered since the measurement is their answer and
    outranks it — a band arriving from a menu press would be the shell
    overwriting a typed number, which is the one thing it may never do.

    (A PROBLEM change is a separate rule and already drops the typed row
    whole: a band typed for the old family's vector is not an opinion about
    the new one's. This is about the carry, so it drives it directly.)
    """
    from gui.v3 import session

    ctx = _shell(tail=True)
    ctx.render("wing", "size")
    ctx.act("auto_recommend")
    carried = session.measured_bands(ctx.S)
    assert carried.get("b_m")

    ctx.act("set_bound", "b_m", 0, 9.0)
    typed, source = _band(ctx)
    assert source == "user", (typed, source)

    written = session.relax_measured_bands(ctx.S, carried)
    assert "b_m" not in written, written
    assert _band(ctx) == (typed, "user")
    assert "Traceback" not in capsys.readouterr().err


def test_the_measurement_is_taken_over_the_mission_s_band_not_the_carried_one(
        capsys):
    """``_unnarrowed_cfg``: the pass measures over a box with no measured row
    in it, WITHOUT taking the one on screen away while it runs. Same answer
    as the old drop-then-measure, which is the whole point."""
    from aerobo import api, recommend
    from gui.v3 import config, session

    ctx = _shell(tail=True)
    ctx.act("auto_recommend")
    ctx.act("set_winglet", "canted")          # leaves a carried band on screen
    assert _band(ctx)[1] == "recommended"
    ctx.act("auto_recommend")
    got, _src = _band(ctx)

    # the same measurement, taken by hand over the mission's own rows
    session.drop_recommended_bounds(ctx.S)
    cfg = config.build_cfg(ctx.S)
    built = api.PROBLEM_SPECS[cfg.problem_name].build(
        cfg.mission_kwargs, cfg.flags, cfg.bounds_overrides)
    rows = recommend.recommend_box(built)["rows"]
    want = session.hold_the_stated_design(ctx.S, rows)["b_m"]
    assert [round(v, 9) for v in got] == [round(float(v), 9) for v in want]
    assert "Traceback" not in capsys.readouterr().err


# ------------------------------------------------ 3. before stage 3 opens
def test_the_box_is_measured_before_the_wing_stage_is_opened(capsys):
    """The user's third sentence. A fast family is measured wherever the
    session is standing, so stage 3 opens on the answer instead of narrowing
    under whoever just walked in — and a LOCKED stage 3 is measured nowhere,
    because everything the measurement is about comes from the mission."""
    from gui.v3.app import assemble

    ctx = _shell(tail=True)
    for stage in ("mission", "airfoil", "wing"):
        ctx.S["ui"]["selected"] = stage
        assert ctx.act("auto_ready") is True, stage

    fresh = assemble("air")               # mission not accepted -> locked
    assert fresh.act("auto_ready") is False
    assert "Traceback" not in capsys.readouterr().err


def test_a_slow_family_is_still_measured_on_arrival_only(capsys):
    """48 draws each holding an XFOIL sweep is minutes of CPU, and stage 2 is
    itself running XFOIL on those families. Starting both is a session
    competing with itself for cores to fill in a box nobody may walk to."""
    ctx = _shell(tail=True)
    ctx.S["airfoil"]["decision"] = {"kind": "designed",
                                    "name": "CST (designed)",
                                    "weights": [0.2] * 8}
    if not ctx.act("auto_ready"):        # this configuration reads slow
        ctx.S["ui"]["selected"] = "wing"
        assert ctx.act("auto_ready") is True
    assert "Traceback" not in capsys.readouterr().err
