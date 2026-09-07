"""The library screen shortlists on one map and scores on another.

``screen_at_point`` is two passes. Pass 1 ranks the whole database at the
cached library point and picks who is worth a real XFOIL sweep; pass 2 sweeps
those at the point that was asked for. Both are supposed to be comparable, and
pass 2 is scored on a FROZEN band measured over pass 1's population — which is
right, because a live min-max would make a 24-section table and a 2000-section
table put the same section in different places.

But pass 1 itself could not use that band: the band does not exist until pass 1
has been run. So pass 1 ranked on a live min-max over its own population, and
the sections allowed to compete were chosen by one map and then judged by
another. Only the band's WIDTH enters the composite, so the two maps re-weight
the criteria against each other — the declared (.10/.20/.15/.35/.20) becomes an
effective (.105/.228/.116/.325/.226) — and the orderings genuinely disagree.

Re-ranking the SAME cached records on the frozen band costs no XFOIL at all,
and the shortlist becomes the union of both maps' leaders. Measured against a
whole-database screen at the same Reynolds number ON THE SAME BAND:

    gate   Re     true winner   shortlisted on pass 1 alone   union
    0.10   3e5    oa213         oa213                          oa213
    0.10   3e6    fx05h126      ah80136   (J 110.774 < 111.871) fx05h126
    0.15   3e5    e1213         hg41      (J  45.353 <  45.385) e1213
    0.15   3e6    hg40          hg40                            hg40

Two of four winners recovered, for 3-10 extra sections swept.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aerobo import api                                   # noqa: E402
from aerobo.airfoil_select import ScoreReference         # noqa: E402


def _library_point():
    lib = api.screen_library_point()
    if not lib:
        pytest.skip("no cached whole-library screen on this machine")
    return lib


def _band(gate: float):
    """The frozen band, measured exactly the way ``screen_at_point`` does."""
    lib = _library_point()
    first = api.screen_airfoils(
        re=float(lib["re"]), mach=float(lib.get("mach", 0.0)),
        tc_min=gate, top_n=64, with_shape=False, measure_reference=True,
        workers=6)
    payload = first.get("reference")
    if not payload:
        pytest.skip("the library population is degenerate at this gate")
    ref = ScoreReference(
        bounds={k: tuple(v) for k, v in payload["bounds"].items()},
        sha=payload["sha"], n_records=int(payload["n_records"]))
    return first, ref


def _cached(re: float, gate: float, ref):
    """A whole-database screen at ``re``, or skip: this needs a checkpoint."""
    import time

    t0 = time.time()
    got = api.screen_airfoils(re=re, tc_min=gate, top_n=10, with_shape=False,
                              reference=ref, workers=6)
    if time.time() - t0 > 60.0:            # pragma: no cover - safety net
        pytest.skip("no checkpoint at this point; a live screen is hours")
    return got


@pytest.mark.slow          # a whole-library XFOIL screen per point: hours on a cold cache
@pytest.mark.parametrize("re,gate", [(3e5, 0.10), (3e6, 0.10),
                                     (3e5, 0.15), (3e6, 0.15)])
def test_the_two_pass_screen_returns_the_whole_database_s_own_winner(re, gate):
    """The shortlist exists to save XFOIL time, not to change the answer."""
    _first, ref = _band(gate)
    truth = _cached(re, gate, ref)
    if not truth.get("ranked"):
        pytest.skip("nothing eligible at this gate")

    got = api.screen_at_point(re=re, tc_min=gate, top_n=3, with_shape=False,
                              workers=6)
    assert got["ranked"][0]["name"] == truth["ranked"][0]["name"]
    assert got["ranked"][0]["composite"] == pytest.approx(
        truth["ranked"][0]["composite"], rel=1e-9)


@pytest.mark.slow
@pytest.mark.parametrize("re,gate", [(3e6, 0.10), (3e5, 0.15)])
def test_pass_one_s_own_ranking_would_have_missed_it(re, gate):
    """The two cells the union recovers — asserted so the union cannot be
    removed as "no measured benefit"."""
    first, ref = _band(gate)
    truth = _cached(re, gate, ref)
    winner = truth["ranked"][0]["name"]

    pass_one_only = [r["name"] for r in first["ranked"]][:api.SHORTLIST_N]
    assert winner not in pass_one_only

    got = api.screen_at_point(re=re, tc_min=gate, top_n=3, with_shape=False,
                              workers=6)
    assert winner in got["shortlist"]["names"]
    assert got["ranked"][0]["name"] == winner


@pytest.mark.slow
def test_the_union_costs_only_the_sections_the_two_maps_disagree_on():
    lib = _library_point()
    got = api.screen_at_point(re=3e5, tc_min=0.15, top_n=3, with_shape=False,
                              workers=6)
    names = got["shortlist"]["names"]
    assert len(names) == len(set(names))            # a union, not a repeat
    assert api.SHORTLIST_N <= len(names) <= 2 * api.SHORTLIST_N
    assert got["shortlist"]["library_point"]["re"] == lib["re"]
    assert "union of the leaders under BOTH maps" in got["shortlist"]["note"]


def test_asking_for_the_cached_point_still_ranks_the_whole_library():
    """No sweep, no shortlist — and no union either: there is nothing to
    choose between when every section has already been screened here."""
    lib = _library_point()
    got = api.screen_at_point(re=float(lib["re"]),
                              mach=float(lib.get("mach", 0.0)),
                              tc_min=0.15, top_n=3, with_shape=False,
                              workers=6)
    assert got["shortlist"]["source"] == "cache"
    assert got["shortlist"]["names"] == []
