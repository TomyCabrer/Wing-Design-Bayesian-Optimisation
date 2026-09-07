"""A number the optimiser maximised must say WHAT it is.

Free the span or the area and ``score`` stops being an L/D: every sized
family overwrites it with ``sizing.SizedState.payload_lod`` = W_fixed/D
(``sizing.py``), while ``LoD`` goes on holding the aero number. Nothing
renamed the axis, so a user comparing a fixed-span run against a free-span
one was differencing two different quantities. Measured across the 376
registered sized variants that evaluate at their own box centre, the gap is
17.0 to 34.7 L/D points — larger than most of the effects the search is
being asked to resolve. On the wing+tail case that started this it was
17.269831 of an apparent 11.816584 "loss", i.e. 146 % of the whole thing.

Three statements are pinned here:

* the sized funnel declares its units, so a family cannot add the size block
  and forget to rename its axis (one line, ``SizedState.report``, covering
  seven of the ten overwrite sites);
* ``api.score_units`` resolves a LEGACY record — one stored before the field
  existed — by an OUTCOME (``score == LoD`` to the bit), never by guessing
  from the problem's name, because the name is exactly where the ``size``
  modifier IS spelled and the number is where it is not;
* two runs in different units cannot share one axis, and a run that declines
  to say what its score is cannot share one either.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aerobo import api, sizing                                   # noqa: E402

_PLAIN = "trim wing"
_SIZED = "trim wing + free planform"


def _at_box_centre(name: str) -> dict:
    spec = api.PROBLEM_SPECS[name]
    prob = spec.build({}, {}, None)
    lo, hi = np.asarray(prob.bounds).T
    out = prob.evaluate((lo + hi) / 2.0)
    assert out["feasible"], out.get("reason")
    return out


# ------------------------------------------------------- the engine says it

def test_a_sized_score_is_a_payload_lod_and_says_so():
    """The closed form, and the name, from one evaluation."""
    out = _at_box_centre(_SIZED)
    assert out["score_units"] == sizing.PAYLOAD_LOD_UNITS != "L/D"
    # the identity that makes the two numbers different: score = LoD x
    # W_fixed/W_total, with W_wing > 0 always, so score < LoD everywhere
    assert out["score"] == pytest.approx(
        out["LoD"] * out["W_fixed_N"] / out["W_total_N"], rel=1e-12)
    assert out["W_wing_N"] > 0.0
    assert out["score"] < out["LoD"]


def test_an_unsized_score_is_the_aero_lod():
    """The control. Without the size block the two numbers ARE one number,
    so the patch must not invent a difference where there is none."""
    out = _at_box_centre(_PLAIN)
    assert out["score"] == out["LoD"]
    assert api.score_units(out) == api.LOD_UNITS


#: registered variants that carry the size block, spanning four different
#: engine files (aircraft.py's inline W/D, objective.py's winglet path, and
#: the plain trim wing) — the funnel is only a funnel if unrelated families
#: come out of it with the same name.
_SIZED_VARIANTS = [
    "free planform (aircraft)",
    "trim wing + free planform",
    "trim wing + free planform + free flight state",
    "trim wing + free span (W/S)",
    "trim wing + free span (W/S) + free flight state",
]


@pytest.mark.parametrize("name", _SIZED_VARIANTS)
def test_every_sized_family_inherits_the_name_from_one_place(name):
    """The tripwire: a family that calls ``SizedState.report`` gets the
    units for free, and one that stops must fail HERE rather than silently
    mislabel its axis. Asserted by evaluating each family, not by reading
    the funnel's source."""
    out = _at_box_centre(name)
    assert out["score_units"] == sizing.PAYLOAD_LOD_UNITS
    assert out["score_units"] != api.LOD_UNITS
    # ...and it is genuinely a different number, so this is not five
    # families agreeing about a label they never use
    assert out["score"] < out["LoD"]
    assert api.score_units(out) == sizing.PAYLOAD_LOD_UNITS


def test_the_two_restatements_are_one_string():
    """``api`` restates the constant rather than importing physics. The two
    may not drift."""
    assert api.PAYLOAD_LOD_UNITS == sizing.PAYLOAD_LOD_UNITS


# --------------------------------------------------- the resolver recovers

def test_a_legacy_record_is_recovered_by_outcome_not_by_name():
    """A run stored before the field existed still resolves — and it does so
    from ``score == LoD``, so a family whose NAME says "free planform" but
    whose number is an L/D is read correctly, and vice versa."""
    legacy = {"problem_name": "trim wing + free planform",
              "breakdown": {"score": 30.0, "LoD": 30.0}}
    assert api.record_score_units(legacy) == api.LOD_UNITS
    silent = {"problem_name": "trim wing", "breakdown": {"score": 30.0}}
    assert api.record_score_units(silent) is None


def test_a_declaration_beats_the_outcome():
    """When the family said, the family is believed — the outcome test is
    the fallback for records that predate the field, not a second opinion."""
    rec = {"breakdown": {"score": 30.0, "LoD": 30.0,
                         "score_units": "composite J (0-100)"}}
    assert api.record_score_units(rec) == "composite J (0-100)"


# --------------------------------------------------- two units, two axes

def _rec(name, score, lod=None, units=None):
    bd = {"score": score}
    if lod is not None:
        bd["LoD"] = lod
    if units is not None:
        bd["score_units"] = units
    return {"problem_name": name, "breakdown": bd}


def test_one_run_is_always_comparable_with_itself():
    """The rule is about CO-plotting. One history on one axis states nothing
    it cannot support, so it must never raise."""
    assert api.check_comparable([_rec("a", 30.0, 30.0)]) == api.LOD_UNITS
    assert api.check_comparable([]) is None


def test_matching_units_pass_and_return_the_unit():
    got = api.check_comparable([_rec("a", 30.0, 30.0),
                                _rec("b", 31.0, 31.0)])
    assert got == api.LOD_UNITS


def test_two_units_cannot_share_one_axis():
    with pytest.raises(ValueError, match="different quantities"):
        api.check_comparable([
            _rec("wing", 30.0, 30.0),
            _rec("wing + free planform", 12.0, 40.0,
                 sizing.PAYLOAD_LOD_UNITS)])


def test_a_run_that_will_not_say_shares_with_nothing():
    """``None`` does not agree with anything, including another ``None``:
    two runs that both decline to say are exactly the pair that could be a
    lap time and an L/D."""
    with pytest.raises(ValueError, match="do not say what their score"):
        api.check_comparable([_rec("a", 30.0, 30.0), _rec("mystery", 1.0)])
    with pytest.raises(ValueError, match="do not say what their score"):
        api.check_comparable([_rec("m1", 1.0), _rec("m2", 2.0)])


# ------------------------------------------------------------- f_lod

def test_f_lod_is_an_lod_even_on_a_sized_family():
    """``f_lod`` exists so a shell can draw the family's own objective under
    its own name. It read ``score`` first, so on a sized family the one key
    promising an L/D handed back a payload L/D."""
    from aerobo import wing_score
    out = _at_box_centre(_SIZED)
    prob = api.PROBLEM_SPECS[_SIZED].build({}, {}, None)
    ref = wing_score.sample_reference(prob.evaluate, prob.bounds,
                                      problem=_SIZED, n=32, seed=0)
    # (32, not 8: only ~5 in 8 blind draws of a sized box fly,
    #  which is the same low feasible fraction the search meets)
    ev = wing_score.composite_evaluation(
        out, ref, wing_score.PRESETS["cruise"])
    # the aero L/D, not the payload one the sized family put in "score"
    assert ev["f_lod"] == pytest.approx(out["LoD"], rel=1e-12)
    assert ev["f_lod"] != pytest.approx(out["score"], rel=1e-6)
    # ...and the composite renamed the axis on its way past
    assert ev["score_units"] == wing_score.COMPOSITE_UNITS
