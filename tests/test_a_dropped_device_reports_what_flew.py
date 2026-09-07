"""A device the lattice DROPPED must not be reported as a device.

``vlm.MIN_WINGLET_FRAC`` is a resolution floor: a tip device shorter than 1 %
of the semi-span never gets a panel, so the wing that flies is planar. The
height every report published, though, was recomputed from the design
variable — ``h_frac * b / 2`` — which is the height that was ASKED for, not
the one that flew. Two consequences, both measured:

* an optimiser that parks on the plateau below the floor (two independent
  runs of the designed-tail family stopped at ``winglet_h_frac_t`` =
  0.009999796627 and 0.009999977685, walking to the cliff edge because the
  gradient is zero on one side and negative on the other) publishes a small
  device. A bound-riding census then reads that as "interior, u = 0.05" when
  the design is really ON THE FLOOR with no device at all;
* the stabiliser's device had no report block anywhere, so the shell's
  "requested but NOT flown" sentence (``gui/v3/stages/results.py``) keyed on
  the ``winglet`` dict could never fire for it.

And one that has nothing to do with the floor: the wing's ``S_planform``
summed ``res.is_winglet`` over the WHOLE lattice, so the wing's tip device
grew whenever the TAIL's did.

Every assertion here is on the report a solver returned, never on a
restatement of the rule.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aerobo import api, geometry, vlm as vlm_mod                 # noqa: E402
from aerobo.vlm import VLM, device_is_flown                      # noqa: E402

_FAMILY = "tail + winglet (span-capped) [designed tail + tip device]"


def _mid_design():
    """The registered family at its own box centre, with the two device
    heights left free to be set by name (never by a hand-counted index —
    this vector has gained rows before)."""
    spec = api.PROBLEM_SPECS[_FAMILY]
    prob = spec.build({}, {}, None)
    labels = list(spec.param_labels)
    lo, hi = np.asarray(prob.bounds).T
    return prob, labels, (lo + hi) / 2.0


def _fly(h_frac: float, h_frac_t: float) -> dict:
    prob, labels, x = _mid_design()
    x = x.copy()
    x[labels.index("winglet_h_frac")] = h_frac
    x[labels.index("winglet_h_frac_t")] = h_frac_t
    out = prob.evaluate(x)
    assert out["feasible"], out.get("reason")
    return out


# --------------------------------------------------------------- one rule

@pytest.mark.parametrize("h_frac", [0.0, 0.005, 0.0099, 0.00999999,
                                    0.01, 0.02, 0.15])
def test_the_drop_rule_has_one_definition(h_frac):
    """``device_is_flown`` and the panel builder must never disagree.

    The helper exists so a reader holding a fraction and no lattice — a
    breakdown, a CAD loft, the census — asks the SAME question the builder
    asks in metres. Two definitions is how the two drift apart.
    """
    wing = geometry.Wing(b=10.0, S=10.0, taper=0.5,
                         twist_root_deg=0.0, twist_tip_deg=-2.0)
    model = VLM(wing, N=40, winglet_h_frac=h_frac,
                winglet_cant_deg=90.0, n_winglet=8)
    assert model.has_winglet is device_is_flown(h_frac)
    # ...and the lattice itself is the third witness: panels, or none
    assert bool(model.is_winglet.any()) is device_is_flown(h_frac)


def test_a_device_with_no_panels_has_no_height():
    """Below the floor the FLOWN height is zero — and the requested one is
    still there beside it, so the pair reads asked / flown."""
    wing = geometry.Wing(b=10.0, S=10.0, taper=0.5,
                         twist_root_deg=0.0, twist_tip_deg=-2.0)
    for h_frac in (0.0, 0.005, 0.0099):
        model = VLM(wing, N=40, winglet_h_frac=h_frac,
                    winglet_cant_deg=90.0, n_winglet=8)
        assert model.winglet_h_m == 0.0
        assert model.winglet_h_frac_flown == 0.0
    # just over the cliff the height is the real one, in metres
    model = VLM(wing, N=40, winglet_h_frac=0.01,
                winglet_cant_deg=90.0, n_winglet=8)
    assert model.winglet_h_m == pytest.approx(0.01 * 10.0 / 2.0, rel=1e-12)


# ------------------------------------------------ what the report publishes

@pytest.mark.parametrize("h_frac", [0.0, 0.005, 0.0099, 0.009999,
                                    0.0101, 0.05, 0.15])
def test_the_report_publishes_the_height_that_flew(h_frac):
    """The published height is zero exactly when the lattice has no device.

    Asserted against the LATTICE, not against the floor: this family is
    span-capped, so the flown fraction is ``h_frac * prob.b / wing.b`` and
    the cliff does not sit at the number the user typed. h = 0.009999 is the
    case that proves it — the wing's device flies (wing.b has shrunk to
    9.974121 m, so the floor moved under it) while the tail's, whose span is
    not capped, does not. A test that pinned the typed number would have
    called that a bug.
    """
    out = _fly(h_frac, h_frac)
    res = out["vlm"]
    tail = res.is_tail
    wl = out["winglet"]
    assert wl["h_frac"] == pytest.approx(h_frac)          # what was ASKED
    assert (wl["h_m"] > 0.0) is bool((res.is_winglet & ~tail).any())
    assert (out["tail_winglet_h_m"] > 0.0) is bool((res.is_winglet
                                                    & tail).any())
    if wl["h_m"] == 0.0:
        # ...and every length derived from it follows the flown height
        assert wl["S_planform"] == 0.0
        assert wl["tip_height_m"] == 0.0
        assert wl["projection_m"] == 0.0


def test_the_stabilisers_device_is_reported_at_all():
    """The tail's device had no block, so nothing downstream could say it
    was dropped. It has one now, and it is gated on what flew."""
    dropped = _fly(0.05, 0.0099)
    assert dropped["tail_winglet_h_m"] == 0.0
    assert dropped["tail_winglet_side"] is None
    assert dropped["tail_winglet_h_frac"] == pytest.approx(0.0099)
    # the WING's device is flying in the same run, so this is the tail's own
    # answer and not one dropped device standing in for two
    assert dropped["winglet"]["h_m"] > 0.0

    flown = _fly(0.05, 0.05)
    assert flown["tail_winglet_h_m"] > 0.0
    assert flown["tail_winglet_side"] in ("up", "down")
    assert flown["tail_winglet_S_planform"] > 0.0


def test_the_wings_device_area_is_the_wings_own():
    """``S_planform`` summed the whole lattice, so growing the TAIL's device
    grew the WING's reported planform area without the wing changing."""
    small_t = _fly(0.05, 0.02)
    big_t = _fly(0.05, 0.15)
    assert small_t["winglet"]["S_planform"] == pytest.approx(
        big_t["winglet"]["S_planform"], rel=1e-12), (
            "the wing's tip-device area moved when only the TAIL's device did")
    # ...and the tail's own area DID move, so this is not two zeros agreeing
    assert big_t["tail_winglet_S_planform"] > \
        small_t["tail_winglet_S_planform"] > 0.0


def test_the_floor_is_a_resolution_statement_not_a_ban():
    """A calibration is a default, not a ban: the family still ACCEPTS a
    sub-floor height and still returns a feasible answer — it just reports
    the planar wing it actually flew."""
    out = _fly(0.005, 0.005)
    assert out["feasible"]
    assert np.isfinite(out["LoD"])
    assert out["winglet"]["h_frac"] == pytest.approx(0.005)


def test_the_floor_is_the_published_constant():
    """A tripwire, not a restatement: if the floor is ever retuned this test
    fails and the reason above must be re-read, rather than the number being
    quietly re-pinned in three report builders."""
    assert vlm_mod.MIN_WINGLET_FRAC == 0.01
    assert device_is_flown(vlm_mod.MIN_WINGLET_FRAC)
    assert not device_is_flown(np.nextafter(vlm_mod.MIN_WINGLET_FRAC, 0.0))
    # n_winglet = 0 is the other way to have no device
    assert not device_is_flown(0.15, n_winglet=0)
