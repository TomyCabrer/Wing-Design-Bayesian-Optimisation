"""The wing–tail separation stops being free.

Asked as *"is there a constraint in the wing and tail separation?"*. There was
a band and a floor and no ceiling — and, more to the point, no COST. The
objective rewarded a longer arm without limit:

    arm      L/D       SM    fin S      (b = 10 m, fin drag charged)
    1.0  25.4068  +0.0031   4.0000
    5.5  30.9074  +0.4129   0.7273
   40.0  32.3947  +3.4501   0.1000
  100.0  32.5695  +8.7053   0.0400

Monotone to 100 m, no interior optimum — a RATCHET, the same shape as the
known W/S one. The fin's area goes as ``V_v b S / l_t``, so a longer arm
needs less fin and less fin drag, and nothing pushed back because the
FUSELAGE was free: ``cd0_extra`` defaults to 0.0 and ``aircraft.py`` records
the omission in as many words.

The fix is not a limit — a typed ceiling would leave every searched arm
riding its bound, the user having merely chosen the bound. It is the missing
COST: ``drag.fuselage_cd0``, a Raymer body that has sat unused in this package
since the beginning, wired to the arm through one shared length constant.
"""
from __future__ import annotations

import numpy as np
import pytest

from aerobo import drag, tail
from aerobo.flightmodel import build_flight_model

D_M = 0.35          # a fuselage diameter, stated as the user must state one


def _at(arm: float, diameter=None, **kw) -> dict:
    # the fin's drag needs no asking for now — it is always charged
    prob = tail.TailProblem(l_t_fixed=arm,
                            fuselage_diameter_m=diameter, **kw)
    x = np.asarray(prob.bounds, float).mean(axis=1)
    return tail.evaluate_tail(x, prob)


# ------------------------------------------------------- the ratchet is real

def test_without_a_fuselage_a_longer_arm_is_free_and_the_arm_ratchets():
    """The defect, pinned, so a future session cannot 'fix' the cure away."""
    lods = [_at(a)["LoD"] for a in (1.0, 3.0, 5.5, 12.0, 40.0, 100.0)]
    assert lods == sorted(lods), f"expected a monotone ratchet, got {lods}"
    assert lods[-1] > lods[0] + 5.0
    # ...and the static margin runs away with it
    assert _at(100.0)["SM"] > 8.0


# ------------------------------------------------- charging it cures the ratchet

def test_charging_the_fuselage_gives_the_arm_an_INTERIOR_optimum():
    """The whole point. Not a bound — a turning point."""
    arms = [1.0, 2.0, 3.0, 4.0, 5.5, 8.0, 12.0, 40.0]
    lods = [_at(a, D_M)["LoD"] for a in arms]
    best = int(np.argmax(lods))
    assert 0 < best < len(arms) - 1, (
        f"the optimum is at an END of the range, so it is still a ratchet: "
        f"{list(zip(arms, lods))}")
    # it lands inside the family's own calibrated band, which is independent
    # corroboration that 0.30-0.80 x b was measured on real aeroplanes
    lo, hi = tail.arm_band(10.0)
    assert lo <= arms[best] <= hi


def test_the_fuselage_drag_grows_with_the_arm_and_overtakes_what_the_fin_saves():
    short, long_ = _at(5.5, D_M), _at(40.0, D_M)
    assert long_["cd0_fus"] > short["cd0_fus"]
    fin_saved = short["cd0_fin"] - long_["cd0_fin"]
    fus_cost = long_["cd0_fus"] - short["cd0_fus"]
    assert fus_cost > 10.0 * fin_saved, (
        f"the fuselage costs {fus_cost:.6f} where the fin saves "
        f"{fin_saved:.6f} — not enough to turn the trade")


def test_an_unstated_diameter_charges_nothing_and_is_bit_for_bit():
    """`None` is off, and off is the published aeroplane exactly."""
    for arm in (3.0, 5.5, 12.0):
        off, on = _at(arm), _at(arm, D_M)
        assert off["cd0_fus"] == 0.0
        assert off["fuselage_length_m"] == 0.0
        assert off["fuselage_diameter_m"] is None
        assert on["LoD"] < off["LoD"]          # charging it costs something


# ---------------------------------------------------- what it reports

def test_the_fineness_ratio_is_reported_because_it_says_more_than_a_drag_count():
    """A 40 m arm asks for a 183:1 body. That is a clearer 'not an aeroplane'
    than any number of drag counts."""
    r = _at(40.0, D_M)
    assert r["fuselage_fineness"] == pytest.approx(
        r["fuselage_length_m"] / D_M, rel=1e-12)
    assert r["fuselage_fineness"] > 100.0
    assert _at(3.0, D_M)["fuselage_fineness"] < 20.0


def test_a_body_no_longer_than_it_is_wide_is_refused_not_returned():
    """``fuselage_cd0``'s form factor is ``1 + 60/fr^3``, which diverges as
    the fineness ratio falls — so a bad input has to be named as one instead
    of coming back as a colossal drag."""
    with pytest.raises(ValueError, match="not a fuselage"):
        _at(0.2, 2.0)
    with pytest.raises(ValueError, match="must be > 0"):
        _at(5.5, 0.0)


# ------------------------------------------------- ONE length, not two

def test_the_body_length_has_a_single_author():
    """It was a bare ``1.6`` inside ``flightmodel``, used for an inertia
    length. The drag term needs the same number, and two copies of a constant
    that decides both a moment of inertia and a drag count is exactly how the
    FIN came to have two sizes."""
    assert drag.body_length_for_arm(5.5) == pytest.approx(
        drag.BODY_LENGTH_FRAC * 5.5, rel=1e-12)
    # a CANARD's arm is negative and a length is a length
    assert drag.body_length_for_arm(-5.5) == drag.body_length_for_arm(5.5)
    # ...and the drag term is built on that very function
    r = _at(5.5, D_M)
    assert r["fuselage_length_m"] == pytest.approx(
        drag.body_length_for_arm(5.5), rel=1e-12)


def test_the_flight_model_uses_the_same_length_for_its_inertia():
    """The other consumer. If these two ever fork, a design's pitch inertia
    and its drag start describing different aeroplanes."""
    from aerobo import api

    cfg = api.RunConfig(problem_name="tail", budget=4, seed=0)
    built = api.PROBLEM_SPECS["tail"].build({}, {}, None)
    rep = api.design_report(cfg, built.bounds.mean(axis=1))
    arm = float(rep["geometry"]["tail"]["dist_m"])
    fm = build_flight_model(rep, V=45.0)
    want = drag.body_length_for_arm(arm)
    # the Inertia states the length it was built on, in its own basis string
    basis = fm.aircraft.inertia.basis
    assert f"L={want:.3g} m" in basis, (
        f"the inertia was built on a different body length than the drag "
        f"was: expected L={want:.3g} m, basis says {basis!r}")

    # ...and it must CALL the shared rule, not happen to agree with it. A
    # literal 1.6 here is numerically identical today and forks the moment
    # anyone tunes the constant — which is precisely how the fin came to have
    # two sizes. Swap the rule and the inertia has to follow.
    import aerobo.flightmodel as fmod

    real = drag.body_length_for_arm
    try:
        drag.body_length_for_arm = lambda l_t, **kw: 3.0 * abs(float(l_t))
        moved = build_flight_model(rep, V=45.0)
    finally:
        drag.body_length_for_arm = real
    assert fmod._drag is drag                      # the same module object
    assert f"L={3.0 * arm:.3g} m" in moved.aircraft.inertia.basis, (
        "flightmodel did not go through drag.body_length_for_arm — it is "
        "carrying its own copy of the constant")
