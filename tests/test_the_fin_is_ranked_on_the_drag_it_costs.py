"""A vertical stabiliser is chosen on the drag it costs, not on an L/D it
never reaches.

The composite ranks a section on six criteria and two of them are lift-to-drag
ratios: ``ldcr`` (|cl_design|/cd at the design lift) and ``ldmax`` (max cl/cd
over the linear band). A fin makes NO side force at zero sideslip, so:

* ``ldcr`` is 0 for every candidate — it cannot rank anything, and the fin's
  preset has always said so;
* ``ldmax`` is the same objection one step out. It is an L/D read at whatever
  lift maximises it, which is a lift this surface never carries. The preset
  spent 0.35 on it anyway, on the argument that it "stands in for" the drag at
  zero lift. It does not, and the first test below is the measurement.

So that weight moved to ``cdcr`` — the drag at the design lift, which at
cl = 0 IS the zero-lift drag. Nothing new is measured: it is the number the
ranking already showed in its unweighted "cd @Cl" column.

The library screen is instant here because every polar is cached (the branch
sidecar re-derives the cl-dependent metrics at any design Cl); the tests skip
rather than run XFOIL for hours if that cache is absent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

#: the shipped fin preset, before this change — kept as a literal so the two
#: rankings can be compared inside one test rather than across a git history
OLD_FIN_WEIGHTS = {"ldcr": 0.0, "clmax": 0.25, "cm": 0.0, "ldmax": 0.35,
                   "thick": 0.20, "astall": 0.20}


@pytest.fixture(scope="module")
def screened():
    """(rows under the shipped preset, rows under the new one) over the whole
    SYMMETRIC library at the fin's own condition — zero lift."""
    from aerobo import api
    from gui.v3 import session as ses

    names = api.symmetric_section_names()
    point = ses.library_point()
    if not names or not point:
        pytest.skip("the screening cache is not built on this machine")

    def screen(weights):
        rep = api.screen_airfoils(
            cl_design=0.0, re=float(point["re"]),
            mach=float(point.get("mach", 0.0)), names=names, weights=weights,
            tc_min=0.0, cm_max=1e9, top_n=10_000)
        return rep["ranked"]

    return screen(OLD_FIN_WEIGHTS), screen(dict(ses.FIN_WEIGHTS))


# ------------------------------------------- (L/D)max is not a drag criterion


def test_ldmax_does_not_stand_in_for_the_drag_at_zero_lift(screened):
    """THE MEASUREMENT THE OLD PRESET RESTED ON, and it does not hold.

    If (L/D)max were a proxy for zero-lift drag, ranking on it would rank on
    -cd. Over the eligible symmetric library the rank correlation is ~0.08:
    it ranks these sections essentially independently of the drag they cost.
    """
    rows = screened[0]
    assert len(rows) > 150, "too few sections to say anything"
    ldmax = np.array([r["ldmax"] for r in rows], dtype=float)
    cd0 = np.array([r["cd_at"] for r in rows], dtype=float)
    ok = np.isfinite(ldmax) & np.isfinite(cd0)

    def rank(v):
        return np.argsort(np.argsort(v)).astype(float)

    spearman = float(np.corrcoef(rank(ldmax[ok]), rank(-cd0[ok]))[0, 1])
    assert abs(spearman) < 0.30, (
        f"(L/D)max ranks the symmetric library like -cd (rho {spearman:+.3f}) "
        "— if that were true the old preset was sound and this change is not")


def test_the_two_lift_ratios_cannot_rank_a_fin_at_all(screened):
    """``ldcr`` is identically 0 here — the reason a weight on it is dead."""
    rows = screened[0]
    assert all(float(r["ldcr"]) == 0.0 for r in rows)


# ------------------------------------------------------- what the preset buys


def test_the_fin_preset_spends_nothing_on_either_lift_ratio():
    from gui.v3 import session as ses

    w = ses.FIN_WEIGHTS
    assert w["ldcr"] == 0.0 and w["ldmax"] == 0.0
    assert w["cm"] == 0.0                      # symmetric: |Cm| is 0
    assert w["cdcr"] > 0.0, "the fin has no drag criterion at all"
    assert sum(w.values()) == pytest.approx(1.0)
    # and it is the SAME weight the (L/D)max criterion used to hold: the
    # criterion changed, the emphasis did not
    assert w["cdcr"] == OLD_FIN_WEIGHTS["ldmax"]


def test_the_new_preset_elects_a_lower_drag_section(screened):
    """End to end on the library: the ranking a user sees moves, and it moves
    towards the number a fin is actually paid for."""
    old, new = screened
    assert old[0]["name"] != new[0]["name"], "the ranking did not move at all"
    assert new[0]["cd_at"] < old[0]["cd_at"], (
        f"{new[0]['name']} ({new[0]['cd_at']:.5f}) is not cheaper than "
        f"{old[0]['name']} ({old[0]['cd_at']:.5f})")
    # the old winner was not merely a little draggier: it sat well above the
    # cheapest section it was ranked against
    cheapest = min(float(r["cd_at"]) for r in old)
    assert float(old[0]["cd_at"]) / cheapest > 1.3


def test_the_drag_criterion_is_scored_and_ranks_the_population(screened):
    """A weight that reaches the score. The sub-score has to be finite, has to
    differ between candidates, and has to be the LOWER-better direction."""
    rows = screened[1]
    s = np.array([r["scores"]["cdcr"] for r in rows], dtype=float)
    cd = np.array([r["cd_at"] for r in rows], dtype=float)
    assert np.all(np.isfinite(s))
    assert s.max() - s.min() > 5.0, "the criterion separates nothing"
    assert float(np.corrcoef(s, cd)[0, 1]) < -0.99   # lower cd, higher score


# --------------------------------------------------------------- the band


def test_the_criterion_carries_a_band_the_shipped_references_predate():
    """The five shipped reference payloads were measured before this criterion
    existed. They must go on loading — and scoring it — rather than being
    rebuilt, because two of them are XFOIL samples of a design box that cannot
    be re-measured without re-running the study they were published from.
    """
    from aerobo import airfoil_select as A

    assert set(A.LATE_CRITERION_BANDS) == {"cdcr"}
    for name in ("screen_reference.json", "screen_reference_re3e5.json",
                 "screen_reference_re3e6.json"):
        ref = A.load_screen_reference(A.SCREEN_REFERENCE_PATH.parent / name)
        assert "cdcr" not in ref.bounds
        lo, hi = ref.band("cdcr")
        assert (lo, hi) == A.LATE_CRITERION_BANDS["cdcr"] and hi > lo
    for path in (A.BOX_BAND_PATH, A.BOX_BAND_2415_PATH):
        assert "cdcr" not in A.load_box_band(path)["bounds"]


def test_a_zero_weight_on_it_leaves_every_published_score_alone():
    """It arrived at weight 0 in every GDP preset, and a zero-weight criterion
    contributes nothing to J. Asserted on the arithmetic rather than trusted:
    the normalisation divides by a sum this term does not move."""
    from aerobo import airfoil_select as A

    for name, preset in A.PRESETS.items():
        assert preset.cdcr == 0.0, name
        assert sum(preset.normalised().values()) == pytest.approx(1.0)
    plain = A.PRESETS["gdp-sweep"].normalised()
    assert plain["cdcr"] == 0.0
    assert sum(v for k, v in plain.items() if k != "cdcr") == pytest.approx(1.0)


# --------------------------------------------------------------- the shell


def test_the_form_keeps_the_dead_rows_and_says_why():
    """The rows stay on the fin's form. "Nobody can weight this" and "you set
    this to zero" are different sentences and a slider at 0.00 makes neither,
    so the criteria that cannot rank this surface are named in one place with
    the reason — including the one that LOOKS like it works."""
    from gui.v3.app import assemble
    from gui.v3.stages import airfoil as afs

    ctx = assemble()
    ctx.act("accept_mission")
    ctx.render("airfoil_fin", "screen")
    texts = [getattr(e, "text", "") or ""
             for e in ctx.views[("airfoil_fin", "screen")].descendants()]

    assert any("cannot rank anything" in t for t in texts), texts
    assert any("(L/D) max" in t for t in texts)
    # ...and the fin's zero lift is stated as an ANSWER, not as a family that
    # would not say what it trims at
    assert any("makes no side force at zero sideslip" in t for t in texts)
    assert not any("will not say what its" in t for t in texts)
    # every criterion the table calls dead is a real criterion, and the drag
    # one is NOT among them
    assert set(afs.DEAD_CRITERIA["fin"]) <= set(afs.WEIGHT_META)
    assert "cdcr" not in afs.DEAD_CRITERIA["fin"]


def test_the_drag_column_is_now_a_weighted_one():
    """The ranking's "cd @Cl" column used to be the one number no criterion
    priced, which is what left a zero-lift surface with nothing to be ranked
    on but its thickness and its stall angle."""
    from gui.v3.stages import airfoil as afs

    assert afs.CRITERION_OF_COLUMN["cd_at"] == "cdcr"
    assert "cdcr" in afs.WEIGHT_META
