"""A chosen section is a section AT A REYNOLDS NUMBER — screened there, and
flown there.

Both halves of that used to be false, and they were false independently:

* the SCREEN's checkpoint was keyed by section name and .dat fingerprint, with
  no operating point in it, so a request at any Reynolds number returned the
  cached 1e6 records instantly — zero XFOIL runs — while the report printed
  the requested Re as its own conditions. Worse, the mismatch also disabled
  the design-Cl re-derivation, so the "honest" option silently ranked on the
  checkpoint's own lift coefficient;
* the FLOWN polar came from a flag carrying only the section's NAME, and no
  builder passed a Reynolds number, so ``library_section_polar`` fell back to
  the cached point for every family, both surfaces, both fluids. hg40 at
  cl 0.5: L/D 80.6 at Re 1e6, 42.2 at 3e5, 19.5 at 1.5e5.

XFOIL is monkeypatched wherever a real sweep is not the thing under test.
"""

from __future__ import annotations

import numpy as np
import pytest

from aerobo import airfoil_select as sel
from aerobo import api, xfoil_run
from aerobo.airfoil import AirfoilProblem, naca4_coords
from aerobo.airfoil_select import ScoreWeights, geometric_tc, screen_database
from aerobo.xfoil_run import XfoilPolarResult


# --------------------------------------------------------------- fixtures
def _write_selig(tmp_path, coords, fname):
    lines = [fname] + [f" {x:.7f} {y:.7f}" for x, y in coords]
    p = tmp_path / fname
    p.write_text("\n".join(lines) + "\n")
    return p


def _patch_xfoil(monkeypatch, calls=None):
    """Fake XFOIL whose drag depends on the REYNOLDS NUMBER, as the real one
    does: cd ~ Re^-0.2 (flat-plate turbulent scaling). Without that a
    point-blind cache and a point-aware one would return the same numbers and
    nothing here would be testable."""
    def fake(coords, re, mach, alphas, **kw):
        if calls is not None:
            calls.append(float(re))
        tc = geometric_tc(np.asarray(coords))
        a = np.asarray(alphas, dtype=float)
        scale = (1.0e6 / float(re)) ** 0.2
        cl = 0.20 + tc + 0.11 * a
        cd = (0.005 + 0.02 * tc) * scale + 1e-4 * (a - 1.0) ** 2
        return XfoilPolarResult(alpha_deg=a, cl=cl, cd=cd,
                                cm=np.full_like(a, -0.04), n_requested=a.size)
    monkeypatch.setattr(xfoil_run, "run_xfoil_polar", fake)


@pytest.fixture
def db(tmp_path):
    """Three sections, thick enough to clear the default t/c gate."""
    return [_write_selig(tmp_path, naca4_coords(code, 161), f"naca{code}.dat")
            for code in ("2415", "2418", "2416")]


# ================================================== the screen's own cache
def test_a_cached_record_is_a_record_at_one_point(db, tmp_path, monkeypatch):
    """The bug, stated as a test: screening at a second Reynolds number off a
    warm checkpoint used to run NOTHING and return the first point's numbers.
    """
    calls: list[float] = []
    _patch_xfoil(monkeypatch, calls)
    ck = tmp_path / "ck.json"

    first = screen_database(db, AirfoilProblem(re=1e6), ScoreWeights(),
                            workers=1, checkpoint=ck, verbose=False)
    assert calls and all(re == 1e6 for re in calls)
    ld_1e6 = {r["name"]: r["ldcr"] for r in first["ranked"]}

    calls.clear()
    second = screen_database(db, AirfoilProblem(re=2e5), ScoreWeights(),
                             workers=1, checkpoint=ck, verbose=False,
                             trust_cache=True)
    # real work, at the point that was asked for — not a silent cache hit
    assert calls, "a new Reynolds number must re-run XFOIL"
    assert set(calls) == {2e5}
    ld_2e5 = {r["name"]: r["ldcr"] for r in second["ranked"]}
    assert set(ld_2e5) == set(ld_1e6)
    for name, ld in ld_2e5.items():
        assert ld < ld_1e6[name]          # more drag at the lower Re

    # ...and the second visit to the SAME point is free again
    calls.clear()
    screen_database(db, AirfoilProblem(re=2e5), ScoreWeights(), workers=1,
                    checkpoint=ck, verbose=False, trust_cache=True)
    assert calls == []


def test_records_carry_their_point_and_unstamped_ones_read_as_legacy(
        db, tmp_path, monkeypatch):
    """New records say where they were measured; the shipped checkpoint, whose
    records predate the stamp, still counts as the point it really covers —
    or a warm library cache would be thrown away on first use."""
    _patch_xfoil(monkeypatch)
    ck = tmp_path / "ck.json"
    out = screen_database(db, AirfoilProblem(re=4e5), ScoreWeights(),
                          workers=1, checkpoint=ck, verbose=False)
    rec = out["records"]["naca2415"]
    assert rec["point"]["re"] == 4e5
    assert sel.record_point(rec) == sel.screen_point(AirfoilProblem(re=4e5))

    legacy = {k: v for k, v in rec.items() if k != "point"}
    assert sel.record_point(legacy) == sel.LEGACY_SCREEN_POINT
    assert sel.LEGACY_SCREEN_POINT == sel.screen_point(AirfoilProblem())


def test_a_cancelled_screen_never_mixes_two_points(db, tmp_path, monkeypatch):
    """Stopping half way leaves the rest of the database at its OLD point in
    the checkpoint. Those records must not be ranked beside the fresh ones —
    that table would compare two Reynolds numbers as if they were one."""
    _patch_xfoil(monkeypatch)
    ck = tmp_path / "ck.json"
    screen_database(db, AirfoilProblem(re=1e6), ScoreWeights(), workers=1,
                    checkpoint=ck, verbose=False)

    stop = {"n": 0}

    def cancel():
        stop["n"] += 1
        return stop["n"] >= 1              # stop after the first completion

    out = screen_database(db, AirfoilProblem(re=2e5), ScoreWeights(),
                          workers=1, checkpoint=ck, verbose=False,
                          trust_cache=True, cancel=cancel)
    assert 0 < len(out["records"]) < len(db)
    for rec in out["records"].values():
        assert rec["point"]["re"] == 2e5


def test_one_checkpoint_file_per_point():
    """Two points under one filename would evict each other section by
    section; the shipped point keeps the shipped path so a warm library cache
    stays warm."""
    assert api.screen_checkpoint(AirfoilProblem()) == api.SCREEN_CHECKPOINT
    other = api.screen_checkpoint(AirfoilProblem(re=3e5))
    assert other != api.SCREEN_CHECKPOINT
    assert other.parent == api.SCREEN_CHECKPOINT.parent
    # the same point always names the same file, a different one never does
    assert other == api.screen_checkpoint(AirfoilProblem(re=3e5, cl_design=1.0))
    assert other != api.screen_checkpoint(AirfoilProblem(re=3.1e5))


def test_the_design_lift_is_honoured_away_from_the_library_point(
        db, tmp_path, monkeypatch):
    """The branch sidecar covers ONE point, so a screen anywhere else used to
    have no way to re-derive the cl-dependent metrics — it returned whatever
    lift the checkpoint had been written at. The record now carries its own
    pre-stall branch, which is what makes the design Cl a live knob at every
    point rather than only at the cached one."""
    calls: list[float] = []
    _patch_xfoil(monkeypatch, calls)
    ck = tmp_path / "ck.json"
    common = dict(re=2e5, tc_min=0.14, cm_max=0.08, db_dir=tmp_path,
                  checkpoint=ck, workers=1, with_shape=False, top_n=5)
    lo = api.screen_airfoils(cl_design=0.4, **common)
    assert calls and set(calls) == {2e5}
    assert all("branch" in r for r in
               sel._load_json_checkpoint(ck, "screen").values())

    calls.clear()
    hi = api.screen_airfoils(cl_design=0.8, **common)
    assert calls == []                 # same point: no new XFOIL work...
    assert hi["point"]["from_record"] == hi["point"]["rederived"] > 0
    # ...and yet the lift moved the metrics, which is the knob that was dead
    a = {r["name"]: r["cd_at"] for r in lo["ranked"]}
    b = {r["name"]: r["cd_at"] for r in hi["ranked"]}
    assert set(a) == set(b)
    assert all(b[n] > a[n] for n in a), (a, b)


# ============================================ the polar the RUN actually flies
def test_a_named_section_can_travel_with_its_point(monkeypatch):
    """``section_polar_for`` accepts a name, a name AT A POINT, or CST
    weights. Without the middle shape a library pick could only ever say
    WHICH section, never WHERE, and every family flew the cached sweep."""
    seen: list[float] = []
    _patch_xfoil(monkeypatch, seen)
    coords = naca4_coords("2415", 161)
    monkeypatch.setattr(api, "_library_coords", lambda name: coords)
    monkeypatch.setattr(api, "_branch_sidecar", dict)   # force a live sweep

    pol = api.section_polar_for({"name": "naca2415", "re": 3.0e5})
    assert pol.Re == pytest.approx(3.0e5)
    assert 3.0e5 in seen
    # an explicit keyword still wins over the point the value carries
    seen.clear()
    pol2 = api.section_polar_for({"name": "naca2415", "re": 3.0e5}, re=7.0e5)
    assert pol2.Re == pytest.approx(7.0e5)


def test_a_bare_name_still_means_the_library_point(monkeypatch):
    """The default path is unchanged: a section chosen at the cached point
    travels as a bare name and is served from the sidecar, so an untouched run
    is the run it always was."""
    if api.screen_library_point() is None:
        pytest.skip("no screening cache on this machine")
    pt = api.screen_library_point()
    pol = api.section_polar_for("hg40")
    assert pol.Re == pytest.approx(float(pt["re"]))


@pytest.mark.slow
def test_the_flown_polar_changes_with_the_point_it_is_flown_at():
    """The magnitude, on real XFOIL: this is why the point has to travel."""
    if api.screen_library_point() is None:
        pytest.skip("no screening cache on this machine")

    def ld(polar):
        o = np.argsort(polar.CL)
        return 0.5 / float(np.interp(0.5, polar.CL[o], polar.CD[o]))

    at_cache = ld(api.section_polar_for("hg40"))
    at_small = ld(api.section_polar_for({"name": "hg40", "re": 3.0e5}))
    assert at_cache > 1.5 * at_small


# ================================================= the two-pass shortlist
def test_screen_at_point_sweeps_a_shortlist_of_the_cached_ranking(
        monkeypatch, tmp_path):
    """Pass 1 ranks the whole library where it is cached (instant, exact in
    Cl) and only chooses WHO is worth a real sweep; pass 2 sweeps those at the
    point that was asked for. The report has to say so — a shortlist that
    reads as a whole-library ranking is a silent truncation."""
    _patch_xfoil(monkeypatch)
    if api.screen_library_point() is None:
        pytest.skip("no screening cache on this machine")

    lib = api.screen_library_point()
    rep = api.screen_at_point(re=float(lib["re"]) * 0.4, cl_design=0.5,
                              tc_min=0.15, cm_max=0.08, shortlist=3,
                              top_n=3, with_shape=False, workers=1)
    assert rep["conditions"]["re"] == pytest.approx(float(lib["re"]) * 0.4)
    assert rep["shortlist"]["source"] == "library"
    # the shortlist is the UNION of the leaders under the two maps pass 1 can
    # be ranked on — a live min-max over its own population (which is all it
    # has before the band exists) and the FROZEN band pass 2 is scored on. So
    # it is `shortlist` per map, deduplicated: at least N, never more than 2N.
    # Chosen on one map and judged on the other, the screen returned the wrong
    # winner in 2 of 4 measured (Re, gate) cells.
    names = rep["shortlist"]["names"]
    assert 3 <= len(names) <= 6
    assert len(names) == len(set(names))
    assert rep["n_screened"] == len(names)
    assert "re-screened" in rep["shortlist"]["note"]
    assert "union of the leaders under BOTH maps" in rep["shortlist"]["note"]
    # the point was honoured, and every ranked row knows where it stood before
    assert rep["point"]["honoured"]
    assert rep["point"]["from_record"] == rep["point"]["rederived"]
    assert all(r["rank_library"] is not None for r in rep["ranked"])
    assert rep["library_pass"]["conditions"]["re"] == pytest.approx(
        float(lib["re"]))


def test_asking_for_the_cached_point_is_never_shortlisted():
    """A shortlist is the price of a sweep. Asked for the point the cache
    already holds, the whole library is ranked instantly — truncating it to N
    there would throw the ranking away for nothing."""
    if api.screen_library_point() is None:
        pytest.skip("no screening cache on this machine")
    lib = api.screen_library_point()
    rep = api.screen_at_point(re=float(lib["re"]),
                              mach=float(lib.get("mach", 0.0)),
                              cl_design=0.5, tc_min=0.15, cm_max=0.08,
                              shortlist=3, top_n=5, with_shape=False)
    assert rep["shortlist"]["source"] == "cache"
    assert rep["n_screened"] > 100
    assert "library_pass" not in rep
    assert rep["wall_time_s"] < 5.0


def test_screen_airfoils_reports_whether_the_point_was_honoured():
    """``point.honoured`` is the guarantee the shells read: the ranking's
    metrics belong to the Re and the Cl the report claims as its conditions.
    """
    if api.screen_library_point() is None:
        pytest.skip("no screening cache on this machine")
    rep = api.screen_airfoils(re=1e6, cl_design=0.5, tc_min=0.15,
                              cm_max=0.08, with_shape=False, top_n=3)
    assert rep["point"]["matched"] and rep["point"]["honoured"]
    # The tracked sidecar (records/) was built over 2,169 sections; the
    # library on disk has since grown, and a section the sidecar has never
    # seen is read as the conservative default rather than refused. So the
    # sidecar covers SOME of the re-derived sections, never more than them.
    assert 0 < rep["point"]["from_sidecar"] <= rep["point"]["rederived"]
