"""V5 item 8 — the CAD export carries the vertical surface.

The user's ask was "the 3D cad has wing + horizontal and vertical tail". Two
of the three were already exported; the third did not exist as geometry
anywhere until item 4 put it in the report, and ``cad.surfaces`` drew wing,
tail and — for a tandem — front and rear, and nothing standing up.

A fin is this package's own wing loft turned ninety degrees about x: the same
cosine spanwise clustering and the same triangulation. Its SECTION is not the
wing's — a fin is symmetric, and lofting it from a cambered wing section draws
a surface flying a permanent side load. It is drawn at the thickness its own
drag was charged at, with no camber. In OpenVSP it is a ``WING`` geom at 90
degrees of dihedral, which is what a fin is there too.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from aerobo import api, cad

V_TAIL = {"tail_type": "v_tail", "dihedral_deg": 35.0}


def _geom(problem: str = "tail", flags: dict | None = None) -> dict:
    flags = flags or {}
    cfg = api.RunConfig(problem_name=problem, budget=4, seed=0, flags=flags)
    built = api.PROBLEM_SPECS[problem].build({}, flags, None)
    return api.design_report(cfg, built.bounds.mean(axis=1))["geometry"]


def _named(geom: dict) -> dict:
    return {s.name: s for s in cad.surfaces(geom)}


def _non_manifold(tri) -> int:
    edges: Counter = Counter()
    for t in tri:
        k = [tuple(np.round(p, 9)) for p in t]
        for a, b in ((0, 1), (1, 2), (2, 0)):
            edges[frozenset((k[a], k[b]))] += 1
    return sum(1 for c in edges.values() if c != 2)


# ------------------------------------------------------------ it is exported

def test_a_conventional_design_exports_wing_tail_AND_fin():
    assert set(_named(_geom())) == {"wing", "tail", "fin"}


def test_a_tandem_exports_both_wings_and_its_fin():
    assert set(_named(_geom("tandem"))) == {"front", "rear", "fin"}


def test_a_v_tail_exports_no_fin_because_it_has_none():
    """The layout's rule reaches the exporter without being restated:
    the report carries no fin block, so nothing draws one."""
    assert "fin" not in _named(_geom("tail", V_TAIL))
    assert set(_named(_geom("tail", V_TAIL))) == {"wing", "tail"}


# --------------------------------------------------------- it is the RIGHT fin

def test_the_exported_fin_is_the_fin_the_report_states():
    geom = _geom()
    blk = geom["fin"]
    fin = _named(geom)["fin"]
    assert np.ptp(fin.Z) == pytest.approx(abs(blk["height_m"]), rel=1e-9)
    assert fin.Z.min() == pytest.approx(blk["z_root_m"], rel=1e-9)
    # the chord runs in x, the LEADING EDGE at x_le
    assert np.ptp(fin.X) == pytest.approx(blk["chord_m"], rel=1e-9)
    assert fin.X.min() == pytest.approx(blk["x_le_m"], rel=1e-9)


def test_the_fin_stands_up_rather_than_lying_flat():
    """Its span is z and its THICKNESS is y — the ninety-degree turn.

    A fin exported like a wing would have a metre of y extent and almost no
    z; this asserts the opposite, which no amount of scaling can fake.
    """
    geom = _geom()
    fin = _named(geom)["fin"]
    assert np.ptp(fin.Z) > 10.0 * np.ptp(fin.Y)
    assert np.ptp(fin.Y) > 0.0                 # it has a section, not a plate
    # ...and it is SYMMETRIC about the centreline. A fin lofted from the
    # wing's cambered section would sit off-centre in y, which is a surface
    # flying a permanent side load — the thing vlm.VerticalSurface's
    # alpha_L0 = 0 says a fin does not do.
    assert abs(fin.Y.mean()) < 1e-9
    assert fin.Y.min() == pytest.approx(-fin.Y.max(), rel=1e-9)


def test_the_fin_mesh_is_watertight():
    tri = cad.triangles(_named(_geom())["fin"])
    assert len(tri) > 100
    assert _non_manifold(tri) == 0


def test_adding_the_fin_did_not_break_the_other_surfaces():
    for name in ("wing", "tail"):
        assert _non_manifold(cad.triangles(_named(_geom())[name])) == 0


# --------------------------------------------------------------- and in VSP

def test_the_vsp_script_builds_the_fin_as_a_ninety_degree_wing():
    geom = _geom()
    src = cad.vsp_script(geom)
    compile(src, "vsp", "exec")           # it must at least be python
    assert "FIN_SECTIONS = [" in src and "FIN_SECTIONS = []" not in src
    assert "dihedral_deg=90.0" in src
    assert f"FIN_X = {geom['fin']['x_le_m']!r}" in src
    assert 'build_wing(FIN_SECTIONS' in src


def test_the_vsp_script_of_a_v_tail_builds_no_fin():
    src = cad.vsp_script(_geom("tail", V_TAIL))
    compile(src, "vsp", "exec")
    assert "FIN_SECTIONS = []" in src


def test_a_ventral_fin_hangs_down_in_both_exporters():
    """Negative height is how a ventral fin — and a hydrofoil strut — is
    asked for everywhere else, so it must mean the same here."""
    geom = _geom()
    geom["fin"] = dict(geom["fin"])
    geom["fin"]["height_m"] = -abs(geom["fin"]["height_m"])
    fin = _named(geom)["fin"]
    assert fin.Z.max() == pytest.approx(geom["fin"]["z_root_m"], rel=1e-9)
    assert fin.Z.min() < geom["fin"]["z_root_m"]
    assert "dihedral_deg=-90.0" in cad.vsp_script(geom)
