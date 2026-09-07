"""A blunt trailing edge is drawn blunt.

`MS3-15Retro` ends open: (1, +0.00555) and (1, -0.00235), a 0.00790c
trailing-edge gap that is part of the aerofoil rather than a defect in the
file. The whole MS3-xx Retro family carries one (0.0058c to 0.0079c), and the
UIUC database holds 193 sections with a gap above 0.005c.

The shell drew none of them. A ranked section was drawn from the CST refit the
screen takes for stage 3 (`airfoil.cst_anchor_from_coords`), and that refit
DROPS the fitted `dz_te` on purpose — the design box fixes the gap at zero, and
every frozen Tier B number in this repo is on that. So the Section card closed
the trailing edge to a point and showed the user an aerofoil nobody has.

What flies is unchanged: the sharp-TE projection, 0.14824 t/c against the
database's 0.15008. What changed is the picture — the section's own
coordinates now travel with the candidate (`api.screen_seed_candidates`),
survive the join onto the ranked row (`merge_candidate`), and are what
`section_outline` hands the plot. The refit stays the fallback for a section
that has no stored outline, which is every optimised shape.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aerobo import api                                        # noqa: E402
from aerobo.airfoil import cst_anchor_from_coords, cst_thickness  # noqa: E402
from aerobo.airfoil_select import load_airfoil_dat            # noqa: E402

_DB = _REPO_ROOT / "data" / "airfoils" / "uiuc"
_BLUNT = "MS3-15Retro"


def _coords(name: str) -> np.ndarray:
    path = _DB / f"{name}.dat"
    if not path.exists():                       # the library is not in git
        pytest.skip(f"{name}.dat is not in this checkout")
    return np.asarray(load_airfoil_dat(path), dtype=float)


def _gap(loop) -> float:
    a = np.asarray(loop, dtype=float)
    return float(abs(a[0, 1] - a[-1, 1]))


# ------------------------------------------------------------ the aerofoil

def test_the_blunt_section_is_measured_as_blunt():
    """The gap, straight off the file."""
    assert api._te_gap(_coords(_BLUNT)) == pytest.approx(0.00790, abs=5e-5)


def test_a_sharp_section_measures_zero():
    c = np.asarray([[1.0, 0.0], [0.5, 0.06], [0.0, 0.0], [0.5, -0.06],
                    [1.0, 0.0]], dtype=float)
    assert api._te_gap(c) == 0.0


def test_the_refit_still_closes_it_and_that_is_what_flies():
    """UNCHANGED, and pinned here on purpose.

    If a later session carries `dz_te` into the design box, this is the test
    that should fail and be rewritten — rather than every frozen Tier B number
    moving under a change nobody announced.
    """
    w_u, w_l = cst_anchor_from_coords(_coords(_BLUNT), 4)
    assert cst_thickness(w_u, w_l) == pytest.approx(0.14824, abs=5e-5)
    assert cst_thickness(w_u, w_l) < 0.15008, "the projection is not thinner"


# -------------------------------------------------------------- the picture

def test_the_outline_drawn_is_the_section_and_it_is_open():
    """THE DEFECT. Drawn from the refit, the loop closes to a point.

    Both shapes are built here and compared, because "the outline is open" is
    only meaningful against the thing that was being drawn instead.
    """
    from gui.v3.stages.airfoil import section_outline

    raw = _coords(_BLUNT)
    w_u, w_l = cst_anchor_from_coords(raw, 4)
    sec = {"name": _BLUNT, "coords": raw.tolist(),
           "w_upper": [float(v) for v in w_u],
           "w_lower": [float(v) for v in w_l]}

    drawn = section_outline(sec)
    assert _gap(drawn) == pytest.approx(0.00790, abs=5e-5)

    refit_only = section_outline({k: sec[k] for k in ("w_upper", "w_lower")})
    assert _gap(refit_only) == pytest.approx(0.0, abs=1e-9), \
        "the refit is the shape that used to be drawn — closed"


def test_a_section_with_no_outline_still_draws_from_its_weights():
    """Every optimised shape is in this case: it has weights and no file."""
    from gui.v3.stages.airfoil import section_outline

    w_u, w_l = cst_anchor_from_coords(_coords(_BLUNT), 4)
    got = section_outline({"w_upper": [float(v) for v in w_u],
                           "w_lower": [float(v) for v in w_l]})
    assert got is not None and got.shape[1] == 2 and got.shape[0] > 20


def test_a_section_with_neither_draws_nothing():
    from gui.v3.stages.airfoil import section_outline

    for sec in ({}, None, {"coords": []}, {"coords": [[0.0, 0.0]]},
                {"w_upper": None, "w_lower": None}):
        assert section_outline(sec) is None


# ------------------------------------- the coordinates reach the card at all

def test_the_outline_travels_with_the_seed_shortlist():
    """`screen_seed_candidates` takes the refit, so it is the one pass that
    still has the section's own loop in hand."""
    report = {"ranked": [{"name": _BLUNT, "tc": 0.15008, "composite": 1.0,
                          "ldcr": 50.0}]}
    got = api.screen_seed_candidates(report, n=1)
    if not got:
        pytest.skip("the coords sidecar has no entry for this section")
    row = got[0]
    assert row["te_gap"] == pytest.approx(0.00790, abs=5e-5)
    assert _gap(row["coords"]) == pytest.approx(row["te_gap"], abs=1e-9)


def test_the_outline_survives_the_row_the_user_clicks():
    """The candidate holds the loop; the RANKED ROW is what gets adopted.

    `merge_candidate` is that join, and it used to carry the CST weights
    alone — which is exactly how the card ended up with a projection and no
    way to draw anything else. Dropping `coords` from the key list is what
    this test is here to catch.
    """
    from gui.v3.stages.airfoil import merge_candidate, section_outline

    raw = _coords(_BLUNT)
    row = merge_candidate(
        {"name": _BLUNT, "tc": 0.15008, "clmax": 1.7},
        [{"name": "something else", "coords": [[0.0, 0.0]], "te_gap": 0.02},
         {"name": _BLUNT, "w_upper": [0.2] * 4, "w_lower": [-0.1] * 4,
          "coords": raw.tolist(), "te_gap": 0.00790}])

    assert row["tc"] == pytest.approx(0.15008), "the row's own metrics stay"
    assert _gap(section_outline(row)) == pytest.approx(0.00790, abs=5e-5)


def test_a_row_with_no_refit_is_left_alone():
    """A section whose loop could not be refitted is skipped by
    `screen_seed_candidates`, so its row gains nothing here — and must not
    gain an invented outline on the way through."""
    from gui.v3.stages.airfoil import merge_candidate, section_outline

    row = merge_candidate({"name": "unfittable", "tc": 0.12}, [])
    assert row == {"name": "unfittable", "tc": 0.12}
    assert section_outline(row) is None
