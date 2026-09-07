"""A box the MISSION empties must be named before the run, not after it.

The reported failure: three searches launched from the V3 wing stage came
back "no solution was found" in 12-21 s each. The mission's weight had been
raised to 7000 N and its stall/landing requirement had not, so the ceiling
off its own constraint diagram (75.2 Pa) needed at least 93 m^2 of wing while
the area row still stopped at 22 m^2 — every draw refused before its solver,
on every seed, at every budget, for ever. No search can answer that, and
:func:`api.size_box_conflicts` is the check that says so in closed form.

WHAT EACH TEST IS FOR. The arithmetic is the easy part and restating it here
would test nothing (a test that restates the code is not a test), so every
claim is tied to the PHYSICS instead: the box the function calls empty is
evaluated at its own best point and must be refused, and the band it offers
is probed and must contain designs that fly. Both survive mutation of the
formula; an equality against a hard-coded 93.08 would not.
"""
import numpy as np
import pytest

from aerobo import api, sizing

#: the cheapest family carrying both size rows and the mission's ceiling
CASE = "trim wing + free planform"

#: the reported mission: a weight raised without its landing requirement
HEAVY = {"W_N": 7000.0, "V": 300.0}

#: the ceiling that mission's own constraint diagram put on the run
CAP_PA = 75.202848


def _cfg(mission=None, overrides=None, cap=CAP_PA):
    return api.RunConfig(
        problem_name=CASE, mission_kwargs=dict(mission or {}),
        flags={"wing_loading_limit_pa": cap} if cap else {},
        bounds_overrides=overrides, optimiser="bo", budget=8, seed=0)


def _row(cfg, name):
    built = api.PROBLEM_SPECS[cfg.problem_name].build(
        cfg.mission_kwargs, cfg.flags, cfg.bounds_overrides)
    labels = list(built.param_labels)
    lo, hi = np.asarray(built.bounds, dtype=float)[labels.index(name)]
    return float(lo), float(hi)


def _best_case_design(cfg):
    """The mid-box design at the LARGEST area the box allows, with a span
    chosen to sit in the middle of the aspect-ratio band — i.e. the friendliest
    point the two cheap gates have. If this is refused, so is everything."""
    built = api.PROBLEM_SPECS[cfg.problem_name].build(
        cfg.mission_kwargs, cfg.flags, cfg.bounds_overrides)
    labels = list(built.param_labels)
    box = np.asarray(built.bounds, dtype=float)
    x = 0.5 * (box[:, 0] + box[:, 1])
    s_hi = box[labels.index("S_m2")][1]
    x[labels.index("S_m2")] = s_hi
    b_lo, b_hi = box[labels.index("b_m")]
    ar_mid = float(np.sqrt(sizing.AR_LIMITS[0] * sizing.AR_LIMITS[1]))
    x[labels.index("b_m")] = float(np.clip(np.sqrt(ar_mid * s_hi), b_lo, b_hi))
    return built, x


def test_the_emptied_box_is_named_and_the_physics_agrees():
    """The finding, and the evaluation that makes it more than arithmetic."""
    cfg = _cfg(HEAVY)
    found = api.size_box_conflicts(cfg)
    assert found, "a box nothing can fly in reported no conflict"
    first = found[0]
    assert first["empty"] is True
    assert first["row"] == "S_m2"
    # the number in the sentence is DERIVED here, never pinned: a constant
    # would go stale the first time the mission or the ceiling moved
    s_need = HEAVY["W_N"] / CAP_PA
    assert f"{s_need:.4g}" in first["text"]
    assert f"{_row(cfg, 'S_m2')[1]:.4g}" in first["text"]

    # ...and the claim itself, against the solver: the friendliest point of
    # the box must come back refused, with the gate the finding names
    built, x = _best_case_design(cfg)
    out = built.evaluate(x)
    assert not out["feasible"]
    assert "wing loading" in str(out["reason"])


def test_the_band_it_offers_contains_designs_that_fly():
    """The mutation-killer. A wrong minimum area still produces a fluent
    sentence; it does not produce a band the physics can fly in. The offer is
    taken as the shell's button takes it, and the same box is then probed."""
    cfg = _cfg(HEAVY)
    band = api.size_box_conflicts(cfg)[0]["suggest"]
    assert band is not None
    widened = _cfg(HEAVY, overrides={"S_m2": list(band)})
    assert api.size_box_conflicts(widened) == [] or all(
        not f["empty"] for f in api.size_box_conflicts(widened))
    probe = api.box_refusal_probe(widened, n=64)
    assert probe["n_feasible"] > 0, (
        "the band the card offers still cannot fly", probe["reasons"])


def test_a_box_the_gates_do_not_rule_out_reports_nothing():
    """The published mission on the same family: the check must not invent a
    conflict, or it would cry wolf on every run in the repo."""
    assert api.size_box_conflicts(_cfg()) == []


def test_the_aspect_ratio_share_is_exact_not_sampled():
    """``frac`` is the share of the (span, area) rectangle inside the
    aspect-ratio band, and that gate is a function of those two numbers and
    nothing else — so the closed form must agree with a direct count, and must
    not be the degenerate 0 or 1 that any broken integral would return."""
    b_lo, b_hi = 6.0, 40.0
    s_lo, s_hi = 93.0, 533.0
    cfg = _cfg(HEAVY, overrides={"S_m2": [s_lo, s_hi], "b_m": [b_lo, b_hi]})
    found = [f for f in api.size_box_conflicts(cfg) if f["kind"] == "aspect ratio"]
    assert found, "a rectangle two thirds outside the band said nothing"
    frac = found[0]["frac"]
    grid_b = np.linspace(b_lo, b_hi, 1201)
    grid_s = np.linspace(s_lo, s_hi, 1201)
    bb, ss = np.meshgrid(grid_b, grid_s, indexing="ij")
    ar = bb * bb / ss
    counted = float(np.mean((ar >= sizing.AR_LIMITS[0])
                            & (ar <= sizing.AR_LIMITS[1])))
    assert 0.0 < frac < 1.0
    assert frac == pytest.approx(counted, abs=2e-3)


def test_the_minimum_area_is_necessary_and_is_not_claimed_to_be_sufficient():
    """The bound uses the PAYLOAD weight, and the gate sees payload + wing, so
    the reported minimum is necessary and not sufficient. The test holds both
    halves: at the stated minimum the payload alone exactly meets the ceiling
    (so nothing smaller can), and the design there is still refused (so the
    function must not be read as a feasibility proof)."""
    s_need = HEAVY["W_N"] / CAP_PA
    assert api.size_box_conflicts(_cfg(HEAVY))[0]["suggest"][0] == \
        pytest.approx(s_need, rel=1e-12)
    at_min = _cfg(HEAVY, overrides={"S_m2": [s_need * 0.999, s_need]})
    built, x = _best_case_design(at_min)
    out = built.evaluate(x)
    assert not out["feasible"], "the wing's own weight was not charged"
    assert "wing loading" in str(out["reason"])


def test_a_pinned_row_is_judged_at_its_pin_and_not_divided_by():
    """A FIXED row has no width, and the box's (span, area) rectangle then has
    no area to take a share of — the first version divided by it. Pinning is
    also the state this check matters most in: a pinned area is a decision,
    and the user is entitled to hear that the mission forbids it."""
    cfg = _cfg(HEAVY, overrides=None)
    pinned = api.RunConfig(
        problem_name=CASE, mission_kwargs=dict(HEAVY),
        flags={"wing_loading_limit_pa": CAP_PA}, pinned={"S_m2": 20.0},
        optimiser="bo", budget=8, seed=0)
    found = api.size_box_conflicts(pinned)
    assert found and found[0]["empty"] is True
    assert "20 m^2" in found[0]["text"], found[0]["text"]
    # ...and the same pin inside a row the mission CAN meet says nothing
    ok = api.RunConfig(
        problem_name=CASE, mission_kwargs=dict(HEAVY),
        flags={"wing_loading_limit_pa": CAP_PA},
        bounds_overrides={"S_m2": [93.0, 533.0]}, pinned={"S_m2": 120.0},
        optimiser="bo", budget=8, seed=0)
    assert api.size_box_conflicts(ok) == []
    # ...and a pinned SPAN is judged against it: 8 m of span on 120 m^2 is an
    # aspect ratio of 0.53, which no solver here is valid over
    both = api.RunConfig(
        problem_name=CASE, mission_kwargs=dict(HEAVY),
        flags={"wing_loading_limit_pa": CAP_PA},
        bounds_overrides={"S_m2": [93.0, 533.0]},
        pinned={"S_m2": 120.0, "b_m": 8.0}, optimiser="bo", budget=8, seed=0)
    assert [f["kind"] for f in api.size_box_conflicts(both)] == ["aspect ratio"]
    assert api.size_box_conflicts(cfg)          # the case above still fires


def test_a_family_with_no_ceiling_is_left_alone():
    """No mission requirement bounds W/S -> no conflict, whatever the rows
    say. The ceiling is the only thing that makes an area row too small."""
    assert api.size_box_conflicts(_cfg(HEAVY, cap=None)) == []


# ------------------------------- the offer has to clear the gate it cites
#
# The reported failure, in the user's own words: "it gives a recommendation
# but it doesn't work. Gives multiple recommendations but only for area. And
# doesn't solve the problem." Both halves were real and both were here.
#
# Every finding that carried a pressable row named ``S_m2``, so a box whose
# SPAN row is what stops it got area advice it could not use — and worse, the
# two gates then wanted the area moved in opposite directions: the loading
# gate pushed it up to ``W/cap``, the aspect-ratio gate pushed it back down to
# ``b_hi^2/ar_lo``, and pressing them alternately never emptied the box less.
#
# These tests are about the OUTCOME of pressing, not about the arithmetic of
# the band: press what the card offers, as often as it is offered, and the box
# must become one these gates do not rule out. A ping-pong fails it, an offer
# that leaves its own gate unsatisfied fails it, and neither survives a
# formula that is merely self-consistent.

def _press_until_clean(cfg_kwargs, overrides, limit: int = 6):
    """Press the first offer the card would draw, over and over.

    Returns ``(presses, overrides, findings)`` — where ``findings`` is what is
    left. Raises on a repeat, because a card that offers a press it has
    already been given is the loop the user reported.
    """
    seen, over = set(), dict(overrides)
    for n in range(limit):
        found = api.size_box_conflicts(_cfg(overrides=over or None,
                                            **cfg_kwargs))
        empty = [f for f in found if f["empty"]]
        if not empty:
            return n, over, found
        offers = [f for f in empty if f.get("row") and f.get("suggest")]
        assert offers, ("a box these gates call empty offered no way out: "
                        + "; ".join(f["text"] for f in empty))
        f = offers[0]
        band = [float(f["suggest"][0]), float(f["suggest"][1])]
        key = (str(f["row"]), round(band[0], 6), round(band[1], 6))
        assert key not in seen, (
            f"the card offered {key} a second time — pressing its own "
            f"recommendations does not converge")
        seen.add(key)
        over[str(f["row"])] = band
    raise AssertionError(f"still not clean after {limit} presses: {over}")


@pytest.mark.parametrize("span", [(6.0, 12.0), (6.0, 40.0), (10.0, 14.0),
                                  (20.0, 24.0)])
def test_pressing_the_offers_reaches_a_box_the_gates_do_not_rule_out(span):
    """Whatever the span row is, the offers converge — and they converge
    because each one clears the gate that produced it, not because the loop
    happens to stop.

    ``(6, 12)`` is the case that motivated it: at 7000 N against 75.2 Pa the
    wing must be 93.09 m^2, and a 12 m span cannot fly 93 m^2 at aspect ratio
    3 or above. NO area band can fix that, and the card used to offer four of
    them in turn.
    """
    n, over, left = _press_until_clean({"mission": HEAVY},
                                       {"b_m": [span[0], span[1]]})
    assert n >= 1, "this box was supposed to start out empty"
    assert all(not f["empty"] for f in left)
    # ...and the box it reached is one the physics agrees with: its own
    # friendliest point must no longer be refused by the gate that emptied it
    built, x = _best_case_design(_cfg(mission=HEAVY, overrides=over))
    out = built.evaluate(x)
    assert "wing loading" not in str(out.get("reason") or ""), out.get("reason")
    assert "aspect" not in str(out.get("reason") or "").lower(), out.get("reason")


def test_the_span_row_is_named_when_no_area_band_can_do_it():
    """The recommendation names the row that CAN fix it.

    Proved rather than asserted: with a 12 m span cap every area band leaves
    the box empty, so an area recommendation here is not a weaker answer, it
    is a wrong one.
    """
    narrow = {"b_m": [6.0, 12.0]}
    s_need = HEAVY["W_N"] / CAP_PA
    for s_lo, s_hi in [(1.0, 48.0), (s_need, 2 * s_need), (0.5, 1000.0),
                       (48.0, 96.0)]:
        found = api.size_box_conflicts(
            _cfg(HEAVY, dict(narrow, S_m2=[s_lo, s_hi])))
        assert any(f["empty"] for f in found), (
            f"S_m2 {s_lo}–{s_hi} was supposed to leave the box empty")

    rows = {f["row"] for f in api.size_box_conflicts(_cfg(HEAVY, narrow))
            if f["empty"] and f.get("suggest")}
    assert "b_m" in rows, rows
    band = next(f["suggest"] for f in api.size_box_conflicts(_cfg(HEAVY, narrow))
                if f["empty"] and f["row"] == "b_m")
    # the span it asks for is the one the aspect-ratio band needs at the area
    # the loading ceiling needs — DERIVED, so a changed AR_LIMITS moves it
    assert band[1] >= np.sqrt(sizing.AR_LIMITS[0] * s_need)


def test_a_wide_span_still_gets_the_area_row_and_nothing_else():
    """The published box's own failure is still an AREA failure. A fix that
    started naming the span everywhere would have replaced one wrong row with
    another."""
    found = api.size_box_conflicts(_cfg(HEAVY))
    rows = {f["row"] for f in found if f["empty"] and f.get("suggest")}
    assert rows == {"S_m2"}, rows
