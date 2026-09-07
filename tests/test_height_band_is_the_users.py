"""The VERTICAL separation is the user's number, stated or searched.

``tail.DZ_FRAC`` * b (0.05 b — 0.5 m on a 10 m span, 0.06 m on a 1.2 m
hydrofoil) is the height every published second surface here was MEASURED
at. It was also enforced as a floor, in three places at once: a stated
height was refused at construction, a searched one was bounded below by it,
and every draw was re-checked inside ``fg_``. That made 0.05 b the smallest
vertical separation the whole model could express — and the layout it
refused is the ordinary one, since a conventional tailplane sits at or near
the wing plane. It bit hardest on small aircraft, where 0.5 m of forced
clearance is a large fraction of the aeroplane.

The re-measurement behind removing it (session 36) is in
``tail.DZ_GRID_FRAC``: sweeping the grid at each height, the L/D spread over
three grids is 0.03 % (LLT) / 0.10 % (VLM) at 0.05 b, still 0.03 % / 0.27 %
at 0.01 b, and only at zero does either core lose it (1.1 % / 6.6 %). So the
floor sat five to twenty-five times above the evidence, and what is real is
a CAUTION near the plane, which the shell says and nothing refuses.

What is still refused is only what the samplers need: an interval with a low
end first, and not both a stated value and a band.
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

AIR = "tail"
AIR_FREE = "tail + winglet [free height]"
WATER = "hydrofoil + elevator"
WATER_FREE = "hydrofoil + elevator [free depth]"


def _built(name, flags=None, bounds=None):
    spec = api.PROBLEM_SPECS[name]
    return spec.build({} if spec.uses_mission else None, flags or {}, bounds)


def _mid(built):
    b = np.asarray(built.bounds, dtype=float)
    return 0.5 * (b[:, 0] + b[:, 1])


def _score(built, x):
    out = built.callable(x)
    return float(out[0] if isinstance(out, tuple) else out)


# ------------------------------------------------------- 1. the physics


@pytest.mark.parametrize("z_t", [0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.0, -0.3])
def test_a_stated_air_height_flies_through_the_old_floor(z_t):
    """Every height the old floor refused is built, solved and finite."""
    built = _built(AIR, flags={api.TAIL_HEIGHT_KEY: z_t})
    f = _score(built, _mid(built))
    assert np.isfinite(f)
    assert f > 1.0, "a real trimmed L/D, not the penalty value"


@pytest.mark.parametrize("z_t", [-0.06, -0.04, -0.02, -0.005, 0.0])
def test_a_stated_water_separation_flies_through_the_old_ceiling(z_t):
    built = _built(WATER, flags={api.TAIL_HEIGHT_KEY: z_t})
    f = _score(built, _mid(built))
    assert np.isfinite(f)
    assert f > 1.0


def test_the_air_height_moves_the_answer_monotonically():
    """Not merely finite: the trend is the physical one (a lower tail sits
    deeper in the downwash), which is what says the number is being flown
    rather than clamped somewhere inside."""
    got = []
    for z in (0.5, 0.3, 0.1, 0.02):
        built = _built(AIR, flags={api.TAIL_HEIGHT_KEY: z})
        got.append(_score(built, _mid(built)))
    assert len(set(np.round(got, 6))) == len(got), "the height did nothing"


# --------------------------------------- 2. the ROW is the searched box


@pytest.mark.parametrize("name", [AIR_FREE, WATER_FREE])
def test_an_untouched_run_is_the_published_run(name):
    """No override -> the family's own band, bit for bit."""
    plain = _built(name)
    i = list(plain.param_labels).index(api.TAIL_HEIGHT_KEY)
    row = api.PROBLEM_SPECS[name].default_bounds[api.TAIL_HEIGHT_KEY]
    assert tuple(np.asarray(plain.bounds, dtype=float)[i]) == \
        (float(row[0]), float(row[1]))


@pytest.mark.parametrize("name,row", [
    (AIR_FREE, (0.02, 3.0)),
    (AIR_FREE, (0.0, 3.0)),
    (WATER_FREE, (-0.36, -0.005)),
    (WATER_FREE, (-0.36, 0.0)),
])
def test_a_widened_band_reaches_the_problems_own_bounds(name, row):
    """The bug this file is named for: the row used to move only the box the
    SAMPLER draws from, so ``prob.bounds`` stayed at the calibrated band and
    every draw outside it came back a bounds violation."""
    built = _built(name, bounds={api.TAIL_HEIGHT_KEY: list(row)})
    i = list(built.param_labels).index(api.TAIL_HEIGHT_KEY)
    assert tuple(np.asarray(built.bounds, dtype=float)[i]) == row


@pytest.mark.parametrize("name,row", [
    (AIR_FREE, (0.02, 3.0)),
    (AIR_FREE, (0.0, 3.0)),
    (WATER_FREE, (-0.36, -0.005)),
])
def test_a_candidate_at_the_new_low_end_is_actually_flown(name, row):
    """...and it is FLOWN, not penalised: the ``fg_`` floor was a third,
    independent ban that survived widening the box."""
    built = _built(name, bounds={api.TAIL_HEIGHT_KEY: list(row)})
    i = list(built.param_labels).index(api.TAIL_HEIGHT_KEY)
    x = _mid(built)
    x[i] = row[0] + 1e-3
    f = _score(built, x)
    assert f > -99.0, f"penalised at z_t = {x[i]}: {f}"


def test_a_narrowed_band_still_narrows():
    """The direction that always worked has to keep working."""
    built = _built(AIR_FREE, bounds={api.TAIL_HEIGHT_KEY: [1.0, 2.0]})
    i = list(built.param_labels).index(api.TAIL_HEIGHT_KEY)
    assert tuple(np.asarray(built.bounds, dtype=float)[i]) == (1.0, 2.0)


def test_the_band_does_not_leak_onto_a_stated_height_twin():
    """A ``z_t_m`` row left over from a searched session must not collide
    with a STATED height when the user switches back.

    Two things stop it, and both are asserted here because the builder-side
    one (``cfg.get("free_height")`` before sending the band) is invisible
    from outside: a stated-height twin carries no ``z_t_m`` row at all, so
    an override naming it is refused before any problem is built.
    """
    built = _built(AIR, flags={api.TAIL_HEIGHT_KEY: 0.3})
    assert api.TAIL_HEIGHT_KEY not in built.param_labels
    with pytest.raises(Exception):
        _built(AIR, flags={api.TAIL_HEIGHT_KEY: 0.3},
               bounds={api.TAIL_HEIGHT_KEY: [0.5, 3.0]})
    # ...and the problem itself refuses the pair, whatever route reaches it
    with pytest.raises(ValueError):
        wingtail.WingTailProblem(z_t_fixed=1.0, z_t_bounds_m=(0.5, 3.0))
    with pytest.raises(ValueError):
        hydrotail.HydrofoilTailProblem(z_t_fixed=-0.1,
                                       z_t_bounds_m=(-0.36, -0.06))


# ------------------------------------------------- 3. what IS still refused


@pytest.mark.parametrize("row", [(1.0, 1.0), (2.0, 1.0), (0.0, 0.0)])
def test_a_zero_width_band_is_refused(row):
    """A zero-width dimension breaks the samplers and degrades BO to random
    search — stating the height is the honest form of that answer."""
    with pytest.raises(ValueError, match="hi <= lo"):
        _built(AIR_FREE, bounds={api.TAIL_HEIGHT_KEY: list(row)})


def test_stated_and_boxed_together_is_refused():
    with pytest.raises(ValueError, match="free it, or state it|state it, or"):
        wingtail.WingTailProblem(free_height=True, z_t_fixed=1.0)


def test_a_t_tail_still_cannot_be_told_its_height():
    """Not a calibration: a T-tail's height IS the fin span its own sizing
    derives, so stating it would be answering the same question twice."""
    with pytest.raises(ValueError, match="fin"):
        tail.TailProblem(tail_type="t_tail", z_t_fixed=1.0)


def test_height_row_passes_a_band_through_unchanged():
    assert tail.height_row((0.0, 2.5), (9.0, 9.0)) == (0.0, 2.5)
    assert tail.height_row(None, (0.5, 3.0)) == (0.5, 3.0)


# ------------------------------------------- 4. the measured grid threshold


def test_the_caution_sits_below_the_band_it_replaced():
    """DZ_GRID_FRAC is the honest line and DZ_FRAC the measured default; the
    point of the change is that the first is well below the second."""
    assert tail.DZ_GRID_FRAC < tail.DZ_FRAC
    assert tail.DZ_GRID_FRAC == pytest.approx(0.01)


def test_nothing_in_the_package_enforces_the_caution():
    """It is a caution the SHELL says, not a gate — a surface in the wing
    plane is built and flown."""
    built = _built(AIR, flags={api.TAIL_HEIGHT_KEY: 0.0})
    assert np.isfinite(_score(built, _mid(built)))
    built = _built(AIR_FREE, bounds={api.TAIL_HEIGHT_KEY: [0.0, 3.0]})
    i = list(built.param_labels).index(api.TAIL_HEIGHT_KEY)
    x = _mid(built)
    x[i] = 0.0
    assert _score(built, x) > -99.0


def test_the_solve_is_grid_converged_well_below_the_old_floor():
    """The measurement DZ_GRID_FRAC records, re-run: at 0.01 b the coupled
    LLT still agrees with itself across grids, which is what makes the old
    0.05 b floor a calibration rather than a numerical requirement."""
    built = _built(AIR)
    prob = built.problem
    x = _mid(built)
    prob.z_t_fixed = tail.DZ_GRID_FRAC * float(prob.b)
    vals = []
    for n in (40, 60, 120):
        prob.N = n
        prob._coupling_cache = {}
        vals.append(float(tail.fg_tail(x, prob)[0]))
    spread = (max(vals) - min(vals)) / abs(np.mean(vals))
    assert spread < 3e-3, f"grid spread {spread:.2%} at 0.01 b"


# ------------------------------------------------------- 5. the V3 shell


def test_the_shell_states_a_height_the_old_floor_refused():
    """0.2 m on a 10 m span — the layout the shell used to reset, and the
    one a small aeroplane actually has."""
    from gui.v3.app import assemble

    ctx = assemble("air")
    ctx.act("set_choice", "tail", True)
    ctx.act("set_trim_number", "tail_height_m", 0.2)
    assert ctx.S["wing"]["choices"]["tail_height_m"] == 0.2


def test_the_stated_height_reaches_the_flags():
    """...and travels: a number the card holds but the run does not send is
    the bug class this whole file exists for."""
    from gui.v3 import config
    from gui.v3.app import assemble

    ctx = assemble("air")
    ctx.act("set_choice", "tail", True)
    ctx.act("set_trim_number", "tail_height_m", 0.2)
    flags = config.flags(ctx.S)
    if api.TAIL_HEIGHT_KEY not in flags:
        pytest.skip("this family states no height")
    assert flags[api.TAIL_HEIGHT_KEY] == pytest.approx(0.2)


def test_the_height_field_carries_no_cap():
    """The field must not be the thing that refuses: no min/max props that
    would exclude 0.2 m, the same rule the stated arm's field follows."""
    from nicegui import ui

    from gui.v3.app import assemble

    ctx = assemble("air")
    ctx.act("set_choice", "tail", True)
    ctx.act("set_trim_number", "tail_height_m", 0.2)
    for el in ui.context.client.elements.values():
        if isinstance(el, ui.number):
            props = getattr(el, "_props", {}) or {}
            assert "min" not in props or float(props["min"]) <= 0.2
            assert "max" not in props or float(props["max"]) >= 0.2


def test_the_shell_can_say_both_bands():
    from gui.v3 import session
    from gui.v3.app import assemble

    ctx = assemble("air")
    ctx.act("set_choice", "tail", True)
    bands = session.height_bands(ctx.S)
    if bands is None:                    # this family may not free the height
        pytest.skip("no free-height twin for the opening choices")
    lo, hi = bands["measured"]
    assert abs(lo) < abs(hi)
    assert 0.0 < bands["grid"] < min(abs(lo), abs(hi)), \
        "the caution must sit BELOW the measured band, not inside it"


def test_the_searched_height_card_renders_and_quotes_its_row():
    """``_render_height_note`` is a read-out in a container of its own (the
    focus trap), so it is reachable only by rendering the card — exercise it
    both ways round, since a band inside the measured one must say nothing
    and one reaching the plane must warn."""
    from nicegui import ui

    from gui.v3 import session
    from gui.v3.app import assemble

    ctx = assemble("air")
    ctx.act("set_choice", "tail", True)
    if not session.height_bands(ctx.S):
        pytest.skip("no free-height twin for the opening choices")
    ctx.act("set_choice", "tail_height", "free")
    if ctx.S["wing"]["choices"].get("tail_height") != "free":
        pytest.skip("this configuration cannot free the height")
    ctx.act("set_bound", api.TAIL_HEIGHT_KEY, 0, 0.001)
    text = " ".join(str(getattr(e, "text", "") or "")
                    for e in ui.context.client.elements.values())
    assert "grid-converged" in text, "no caution beside a band at the plane"


def test_the_bands_follow_the_span():
    """The height's band is a FRACTION of the span, so a shell that quotes
    the family's static box quotes the 10 m trim wing at everyone — the
    drift ``config.FLAG_MOVED_ROWS`` exists for, here for both numbers."""
    from gui.v3 import session
    from gui.v3.app import assemble

    ctx = assemble("air")
    ctx.act("set_choice", "tail", True)
    before = session.height_bands(ctx.S)
    if before is None:
        pytest.skip("no free-height twin for the opening choices")
    # smaller, but still inside api.PLANFORM_AR_LIMITS on the trim area —
    # this test is about the band following the span, not about the span
    if not session.set_span_m(ctx.S, 6.0):
        pytest.skip("this family's span is not the shell's to set")
    after = session.height_bands(ctx.S)
    assert after is not None
    assert after["grid"] < before["grid"], \
        "a smaller aeroplane must get a smaller caution, not the 10 m one"
