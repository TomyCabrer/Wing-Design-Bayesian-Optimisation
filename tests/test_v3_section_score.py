"""The shape optimiser's OTHER answer: what the run did to the six criteria
the weights actually asked for.

Stage 2 (and 2.5) asks the user for six criterion weights, ranks the library
under them — and then hands the winner to a search that maximises exactly ONE
number (section c_d at the design lift, or wing L/D). Everything the weights
said about c_l max, stall angle, thickness and pitching moment stopped being
reported the moment the optimiser started. These tests pin the fix:

* ``api.score_sections`` scores NAMED shapes on the screen's own metric
  extraction and composite, against a FROZEN band — never against each other,
  which would hand out 100 and 0 by construction;
* gates are reported, not applied, so a design that fails one is still
  comparable and its failure is visible rather than a missing row;
* ``api.score_optimised_section`` turns a run report into seed / optimised /
  delta, and ``stages.airfoil.score_rows`` turns that into rows whose fields
  are the ones the card's columns ask for (the field-name contract that has
  already broken this card once).
"""

from __future__ import annotations

import numpy as np
import pytest

from aerobo import api, xfoil_run
from aerobo.airfoil import naca4_coords
from aerobo.airfoil_select import CRITERIA
from aerobo.xfoil_run import XfoilPolarResult
from gui.v3 import session
from gui.v3.stages import airfoil as stage


# ---------------------------------------------------------------- fixtures


def _fake_polar(alphas, cl_alpha=0.11, cl0=0.25, cd0=0.006, cm=-0.05):
    """Synthetic linear polar (tests/test_airfoil_select.py's helper)."""
    a = np.asarray(alphas, dtype=float)
    return XfoilPolarResult(alpha_deg=a, cl=cl0 + cl_alpha * a,
                            cd=cd0 + 1e-4 * (a - 1.0) ** 2,
                            cm=np.full_like(a, cm), n_requested=a.size)


def _patch_xfoil(monkeypatch, cm_by_tc=None):
    """Fake XFOIL whose polar depends on the geometry, so two shapes differ."""
    from aerobo.airfoil_select import geometric_tc

    def fake(coords, re, mach, alphas, **kw):
        tc = geometric_tc(np.asarray(coords, dtype=float))
        cm = (cm_by_tc or {}).get(round(tc, 3), -0.05)
        return _fake_polar(alphas, cl0=0.20 + tc, cd0=0.005 + 0.02 * tc,
                           cm=cm)

    monkeypatch.setattr(xfoil_run, "run_xfoil_polar", fake)


def _two_sections():
    return [{"name": "seed", "coords": naca4_coords("2412", 161)},
            {"name": "optimised", "coords": naca4_coords("2415", 161)}]


# ------------------------------------------------- the band is frozen


def test_two_sections_are_not_normalised_against_each_other(monkeypatch):
    """The bug this design exists to avoid: min-max normalising a population
    of two awards 100 and 0 on every criterion, whatever the two sections
    are. A frozen band makes the numbers mean something."""
    _patch_xfoil(monkeypatch)
    out = api.score_sections(_two_sections(), "gdp-sweep")

    assert out["reference"]["source"] == "shipped"
    assert out["reference"]["sha"]
    scored = {s["name"]: s for s in out["sections"]}
    assert set(scored) == {"seed", "optimised"}
    for row in scored.values():
        assert row["status"] == "ok"
        assert row["composite"] is not None
        assert set(row["scores"]) <= set(CRITERIA)
        # the population-relative map would put every sub-score at exactly
        # 0 or 100 — the frozen band does not
        assert any(0.0 < v < 100.0 for v in row["scores"].values()), \
            row["scores"]


def test_no_band_means_no_score_rather_than_a_made_up_one(monkeypatch):
    """A reference that cannot be loaded leaves the composite MISSING and
    says why. Falling back to the live min-max would silently restore the
    100-and-0 artefact under the same field name."""
    _patch_xfoil(monkeypatch)
    from aerobo import airfoil_select

    def broken(*a, **kw):
        raise ValueError("no band on this machine")

    monkeypatch.setattr(airfoil_select, "load_screen_reference", broken)
    out = api.score_sections(_two_sections(), "gdp-sweep")

    assert "no band on this machine" in out["reference"]["reason"]
    for row in out["sections"]:
        assert row["composite"] is None
        assert "normalisation band" in row["reason"]


def test_a_screens_own_band_is_honoured(monkeypatch):
    """A screen that measured its own band (screen_at_point's library pass)
    can hand it here, so the run's two sections land on the SAME map as the
    ranking the user is looking at."""
    _patch_xfoil(monkeypatch)
    band = {"bounds": {k: (0.0, 2.0) if k == "cm" else (0.0, 200.0)
                       for k in CRITERIA},
            "sha": "deadbeef", "n_records": 42}
    out = api.score_sections(_two_sections(), "gdp-sweep", reference=band)

    assert out["reference"] == {"source": "caller", "sha": "deadbeef",
                                "n_records": 42}
    assert all(s["composite"] is not None for s in out["sections"])


# ------------------------------------------------- gates are reported


def test_a_failed_gate_is_reported_not_filtered(monkeypatch):
    """The screen DROPS a section that fails t/c or |Cm|. A comparison of two
    named designs must not: the user is asking what the search did, and 'the
    row vanished' is the one answer that cannot be read."""
    # NACA 2415's |cm| is driven past the 0.08 cap; 2412 stays inside it
    _patch_xfoil(monkeypatch, cm_by_tc={0.15: -0.14})
    out = api.score_sections(_two_sections(), "gdp-sweep", cm_max=0.08)

    rows = {s["name"]: s for s in out["sections"]}
    assert rows["seed"]["gates"] == {"tc": True, "cm": True}
    assert rows["optimised"]["gates"]["cm"] is False
    # …and it is still scored, so the two composites remain comparable
    assert rows["optimised"]["composite"] is not None


def test_a_shape_with_no_cached_polar_is_named_not_faked(monkeypatch):
    """``cache_only`` is the setting a VIEW uses: it must never spend a
    minute of XFOIL on a comparison nobody asked for, and a miss is a stated
    status rather than a zero."""
    monkeypatch.setattr(api, "_cached_dat_polar", lambda *a, **kw: None)
    out = api.score_sections(_two_sections(), "gdp-sweep", cache_only=True)

    for row in out["sections"]:
        assert row["status"] == "no_polar"
        assert row["composite"] is None
    assert out["best"] is None


# ------------------------------------------------- seed vs optimised


def _run_report() -> dict:
    """The shape of the report ``api.optimize_airfoil`` returns, trimmed to
    what the scoring reads."""
    return {
        "conditions": {"re": 1e6, "mach": 0.0, "cl_design": 0.5,
                       "tc_min": 0.10, "cm_max": 0.08, "wing_mode": False},
        "result": {"best_score": -0.0055},
        "section": {
            "design": {"coords": naca4_coords("2415", 161).tolist()},
            "baseline": {"name": "hg40 CST refit",
                         "coords": naca4_coords("2412", 161).tolist()},
        },
    }


def test_seed_and_optimised_come_back_with_their_delta(monkeypatch):
    _patch_xfoil(monkeypatch)
    out = api.score_optimised_section(_run_report(), "gdp-sweep")

    assert out["seed"]["label"] == "hg40 CST refit"
    assert out["optimised"]["composite"] is not None
    assert out["delta"]["composite"] == pytest.approx(
        out["optimised"]["composite"] - out["seed"]["composite"])
    for key, dv in out["delta"]["scores"].items():
        assert dv == pytest.approx(out["optimised"]["scores"][key]
                                   - out["seed"]["scores"][key])


def test_a_run_that_never_left_its_anchor_has_no_seed_row(monkeypatch):
    """``section_report`` omits the baseline shape when the design IS the
    anchor. The delta is then EMPTY, not zero: there is no second section."""
    _patch_xfoil(monkeypatch)
    rep = _run_report()
    rep["section"].pop("baseline")
    out = api.score_optimised_section(rep, "gdp-sweep")

    assert out["seed"] is None
    assert out["optimised"]["composite"] is not None
    assert out["delta"]["composite"] is None
    assert out["delta"]["scores"] == {}


def test_no_section_shape_means_no_score_block(monkeypatch):
    _patch_xfoil(monkeypatch)
    assert api.score_optimised_section({"conditions": {}}, "gdp-sweep") is None


# ------------------------------------------------- the card's own contract


def test_score_rows_carry_the_fields_the_card_declares():
    """Same rule the metric table already learned the hard way: every column
    the card declares has to name a key the rows actually have."""
    import re
    from pathlib import Path

    src = Path(stage.__file__).read_text()
    block = src.split("rows=rows, row_key=\"metric\")", 1)[0]
    block = block.split("_render_score_block", 1)[1]
    fields = set(re.findall(r'"field":\s*"([a-z_]+)"', block))
    assert fields, "the score table declares no fields at all"

    rows = stage.score_rows({
        "weights": {k: 1.0 / len(CRITERIA) for k in CRITERIA},
        "seed": {"composite": 68.8,
                 "scores": {k: 50.0 for k in CRITERIA}},
        "optimised": {"composite": 71.2,
                      "scores": {k: 55.0 for k in CRITERIA}},
        "delta": {"composite": 2.4, "scores": {k: 5.0 for k in CRITERIA}},
    })
    assert rows and fields <= set(rows[0])
    assert rows[0]["metric"] == "composite score"
    assert rows[0]["change"] == "+2.40"
    # every criterion the weights name gets a row, with its weight on it
    assert len(rows) == 1 + len(api.SCREEN_METRICS)
    assert "weight 0.17" in rows[1]["metric"]


def test_an_unmeasured_criterion_is_a_dash_not_a_zero():
    rows = stage.score_rows({
        "weights": {k: 0.0 for k in CRITERIA},
        "seed": {"composite": None, "scores": {}},
        "optimised": {"composite": 40.0, "scores": {"ldcr": 61.0}},
        "delta": {"composite": None, "scores": {}},
    })
    assert rows[0]["original"] == "—" and rows[0]["change"] == "—"
    ldcr = [r for r in rows if r["metric"].startswith("L/D at design Cl")][0]
    assert ldcr["original"] == "—" and ldcr["new"] == "61.0"


def test_score_rows_of_nothing_is_nothing():
    assert stage.score_rows(None) == []
    assert stage.score_rows({}) == []


def test_the_workspace_carries_the_score_slots():
    """The stage's worker writes into its OWN surface's workspace — one score
    per surface, because stage 2 and stage 2.5 are two different questions."""
    S = session.make_session("air")
    for surface in ("main", "aft"):
        opt = session.airfoil_state(S, surface)["opt"]
        assert opt["score"] is None
        assert opt["scoring"] is False
        assert opt["score_error"] is None
