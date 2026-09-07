"""The aspect ratio is a LIMIT the user can state, and it is enforced twice.

Asked for in session 65: *"for wing limit should also let aspect ratio
limit"* — beside the chord limits, which are the same kind of statement. The
design box speaks in a span row and an area row, and b²/S is a DIAGONAL
across the two: every span x area rectangle covers a range of ratios (the
shipped 3.6-24 m span against a 0.8-2.2x area band reaches AR 1.3-160), so
the ratio cannot be a bound and has to be a limit.

Four statements are asserted here.

**It is declared where it is READ.** ``api.AR_LIMIT_KEYS`` goes on the
families whose evaluate calls ``sizing.check_ar_limit`` (``build.gates_ar``)
and on no others, so a family cannot accept the flag and then silently ignore
it — the failure ``api.check_flags`` exists to refuse.

**It refuses designs.** Measured on draws, not asserted from the code: the
same box under the same seed refuses more of its own draws with the limit set
than without it, and the refusal names the limit.

**It can only narrow.** ``sizing.AR_LIMITS`` is where these solvers describe
a wing at all; a user band is intersected with it, and the two refusals are
named apart.

**The box is clipped to it.** ``session.ar_band`` feeds
``session.clip_size_box``, so the span row's ceiling follows the limit —
except that the wing the mission states is still inside its own box, which
the card says out loud rather than quietly re-opening the search.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _texts(view) -> list[str]:
    return [getattr(e, "text", None) or "" for e in view.descendants()]


# --------------------------------------------------- 1. declared where read
def test_the_flag_is_declared_exactly_where_it_is_gated():
    from aerobo import api

    declared = {k for k, v in api.PROBLEM_SPECS.items()
                if set(api.AR_LIMIT_KEYS) <= set(v.flags)}
    gated = {k for k, v in api.PROBLEM_SPECS.items()
             if getattr(v.build, "gates_ar", False)}
    assert declared == gated
    assert len(declared) > 1000            # the wing, the pair, the aeroplane
    # ...and the families with no such gate do NOT accept it, so the limit
    # cannot be typed somewhere it would be dropped
    assert "ar_min" not in api.PROBLEM_SPECS["hydrofoil"].flags
    try:
        api.check_flags("hydrofoil", {"ar_min": 8.0})
    except Exception as exc:               # noqa: BLE001 — the point is refusal
        assert "ar_min" in str(exc)
    else:
        raise AssertionError("a family with no AR gate accepted the limit")


def test_the_wing_and_the_pair_both_declare_it():
    from aerobo import api

    for name in ("wing (free chord law)", "trim wing + free span (W/S)",
                 "tandem", "tail"):
        assert "ar_max" in api.PROBLEM_SPECS[name].flags, name


# ------------------------------------------------------- 2. it refuses
def _refused(name: str, flags: dict, n: int = 24) -> int:
    from aerobo import api
    from aerobo.optimize.feasible import sobol_pool

    cfg = api.RunConfig(problem_name=name, optimiser="bo", budget=6, seed=0,
                        flags=flags)
    built, _stripped = api._recommendable_problem(cfg)
    box = np.asarray(built.bounds, dtype=float)
    return sum(0 if bool(built.evaluate(np.asarray(x, float)).get("feasible"))
               else 1 for x in sobol_pool(box, n, 0))


def test_a_stated_band_refuses_draws_the_same_box_flew_without_it():
    """The measurement, on the family the shell opens on when the span is
    searched. Without the limit the box flies; with it, part of the same box
    is refused — and it is the LIMIT that did it, not a narrower box."""
    name = "trim wing + free span (W/S)"
    before = _refused(name, {})
    after = _refused(name, {"ar_min": 12.0, "ar_max": 18.0})
    assert after > before
    # ...and taking the limit back off restores the original count exactly
    assert _refused(name, {}) == before


def test_the_refusal_names_the_users_limit_and_not_the_solvers_band():
    from aerobo.sizing import check_ar, check_ar_limit

    # inside the solvers' band, outside the user's
    assert check_ar(10.0, 10.0) is None                     # AR 10
    said = check_ar(10.0, 10.0, (12.0, 20.0))
    assert said is not None and "you set" in said
    # outside the solvers' band: that is what is named, whatever the user set
    said = check_ar(40.0, 10.0, (12.0, 200.0))              # AR 160
    assert said is not None and "solvers are valid over" in said
    # ...and no limit is no gate at all, not a gate that always passes
    assert check_ar_limit(10.0, 10.0, None) is None
    assert check_ar_limit(10.0, 10.0, (None, None)) is None


def test_the_limit_can_only_narrow_the_solvers_band():
    """A user band wider than ``AR_LIMITS`` buys nothing: the solvers' band
    still refuses. Asserted through the gate, and through the shell's own
    reader, because both are asked this question."""
    from aerobo.sizing import AR_LIMITS, check_ar

    assert check_ar(40.0, 10.0, (1.0, 1000.0)) is not None  # AR 160

    from gui.v3.app import assemble
    from gui.v3 import session

    ctx = assemble("air")
    ctx.act("accept_mission")
    ctx.S["wing"]["flags"].update(ar_min=1.0, ar_max=1000.0)
    assert session.ar_band(ctx.S) == (float(AR_LIMITS[0]), float(AR_LIMITS[1]))


# ------------------------------------------------------- 3. it reaches the run
def test_the_typed_limit_reaches_the_built_problem():
    from aerobo import api
    from gui.v3 import config
    from gui.v3.app import assemble

    ctx = assemble("air")
    ctx.act("accept_mission")
    ctx.act("set_planform", "free")
    ctx.render("wing", "box")
    ctx.act("set_ar_limit_on", "ar_max", True)
    ctx.act("set_ar_limit", "ar_max", 9.0)

    cfg = config.build_cfg(ctx.S)
    assert cfg.flags["ar_max"] == 9.0
    built, _stripped = api._recommendable_problem(cfg)
    assert built.problem.ar_limits == (None, 9.0)


def test_a_maximum_below_the_minimum_is_refused_and_rolled_back():
    from gui.v3.app import assemble

    ctx = assemble("air")
    ctx.act("accept_mission")
    ctx.act("set_planform", "free")
    ctx.render("wing", "box")
    ctx.act("set_ar_limit_on", "ar_max", True)
    ctx.act("set_ar_limit", "ar_max", 9.0)
    ctx.act("set_ar_limit_on", "ar_min", True)
    ctx.act("set_ar_limit", "ar_min", 20.0)          # above the maximum

    assert ctx.S["wing"]["flags"]["ar_max"] == 9.0
    assert ctx.S["wing"]["flags"]["ar_min"] < 9.0    # the old value stands


# ------------------------------------------------------- 4. the box follows
def test_the_span_row_is_clipped_to_the_limit():
    from gui.v3 import config, session
    from gui.v3.app import assemble

    ctx = assemble("air")
    ctx.act("accept_mission")
    ctx.act("set_planform", "free")                  # a span row AND an area
    ctx.render("wing", "box")
    before = config.effective_bounds(ctx.S)["b_m"][0]

    ctx.act("set_ar_limit_on", "ar_max", True)
    ctx.act("set_ar_limit", "ar_max", 9.0)
    after = config.effective_bounds(ctx.S)["b_m"][0]
    area = config.effective_bounds(ctx.S)[session.AREA_ROW][0]

    assert after[1] < before[1]
    # the ceiling is the corner arithmetic, unless the stated wing is wider
    # (which the card says out loud — the rule below)
    corner = (9.0 * float(area[0])) ** 0.5
    assert after[1] <= max(corner, session.nominal_span(ctx.S)) + 1e-9


def test_the_stated_wing_stays_inside_its_own_box_and_the_card_says_so():
    """A limit may not push the design on screen out of the box that is
    about it — and a box holding a corner the limit refuses is exactly the
    thing the card has to name, because the refusal otherwise arrives inside
    the run with nothing on screen explaining it."""
    from gui.v3 import config, session
    from gui.v3.app import assemble

    ctx = assemble("air")
    ctx.act("accept_mission")
    ctx.act("set_planform", "free")
    ctx.render("wing", "box")
    span = session.nominal_span(ctx.S)
    area = float(ctx.S["mission"]["s_ref_m2"])
    stated = span * span / area

    ctx.act("set_ar_limit_on", "ar_max", True)
    ctx.act("set_ar_limit", "ar_max", round(0.5 * stated, 2))
    ctx.render("wing", "box")

    row = config.effective_bounds(ctx.S)["b_m"][0]
    assert row[0] <= span <= row[1]
    said = _texts(ctx.views[("wing", "box")])
    assert any("outside" in t and "stated" in t for t in said)
