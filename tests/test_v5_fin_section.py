"""V5 item 7 — the fin's section is symmetric, at the fin's own Re.

The ask that opened the session was "optimisation on the rudder airfoil
(simpler symmetrical airfoil only, changed based on Re and chord)". It was
ranked LAST because the fin's SIZE was not a design variable at all, so its
section was the third-order term of an assumed object. Items 4 and 8 made the
fin a real surface with a real chord; this is what follows from that.

What is closed here:

* the fin's section is SYMMETRIC everywhere it is drawn or written — the
  lattice already pinned ``alpha_L0 = 0``, the CAD loft and the OpenVSP
  script did not, and both were building a cambered fin from the wing's file;
* it is stated at the fin's OWN Reynolds number. Measured on the shipped tail
  design the wing mac is 1.47x the fin chord, so a section chosen at the
  wing's Re is chosen 47 % too high;
* it leaves the bundle as its own ``_fin.dat``, and the OpenVSP script reads
  that file rather than the wing's.

What is NOT closed, and is the open remainder: the fin's section is the
symmetric NACA stand-in at its charged thickness, not a SEARCHED shape.
Making it searched means the fin becoming surface #3 in stage 2, which is a
shell change on top of this engine one.
"""
from __future__ import annotations

import pathlib
import tempfile

import numpy as np
import pytest

from aerobo import api, cad, fin as finmod
from aerobo.objective import MU_SL, RHO_SL

V_TAIL = {"tail_type": "v_tail", "dihedral_deg": 35.0}


def _report(problem: str = "tail", flags: dict | None = None) -> dict:
    flags = flags or {}
    cfg = api.RunConfig(problem_name=problem, budget=4, seed=0, flags=flags)
    built = api.PROBLEM_SPECS[problem].build({}, flags, None)
    return api.design_report(cfg, built.bounds.mean(axis=1))


# -------------------------------------------------- at its OWN Reynolds number

def test_the_reynolds_number_is_built_on_the_fin_chord():
    g = finmod.size_fin(b=10.0, S=10.0, l_t=5.5)
    assert g.reynolds(30.0, RHO_SL, MU_SL) == pytest.approx(
        RHO_SL * 30.0 * g.chord / MU_SL, rel=1e-12)


def test_the_reported_fin_re_is_not_the_wings():
    """The whole point of asking the question per surface.

    On the shipped tail design the wing mac is 1.47x the fin chord, so
    reusing the wing's Reynolds number picks a section 47 % too high.
    """
    rep = _report()
    blk = rep["geometry"]["fin"]
    mac = float(rep["geometry"]["tail"]["mac"])
    assert blk["Re"] > 0.0
    ratio = mac / blk["chord_m"]
    assert ratio == pytest.approx(1.47, rel=0.05)
    # what the WING's Reynolds number would have been, from the same speed
    V = float(blk["Re"]) * MU_SL / (RHO_SL * blk["chord_m"])
    re_wing = RHO_SL * V * mac / MU_SL
    assert re_wing / blk["Re"] == pytest.approx(ratio, rel=1e-9)
    assert re_wing - blk["Re"] > 2e5, "the two Re are not far enough apart"


def test_the_drag_book_charges_the_fin_at_that_same_reynolds_number():
    """One author, extended to the section: the Re the block states must be
    the Re ``vtail_cd0`` builds its skin friction on."""
    from aerobo.drag import skin_friction_cf, wing_form_factor
    from aerobo.tail import vtail_cd0
    b, S, l_t, V = 10.0, 10.0, 5.5, 14.6
    g = finmod.size_fin(b=b, S=S, l_t=l_t)
    Re = g.reynolds(V, RHO_SL, MU_SL)
    want = (skin_friction_cf(Re, lref=g.chord) * wing_form_factor(g.tc)
            * 1.05 * 2.0 * g.S / S)
    assert vtail_cd0(S, l_t, b=b, V=V) == want


# ------------------------------------------------------------- and SYMMETRIC

def test_the_lofted_fin_section_has_no_camber():
    """A cambered fin flies a permanent side load with nothing to trim it.

    The lattice already says so (``VerticalSurface.alpha_L0`` is pinned to
    0); this is the geometry agreeing. Measured as the mean line: for every
    chordwise station the two surfaces must be equal and opposite.
    """
    geom = _report()["geometry"]
    fin = {s.name: s for s in cad.surfaces(geom)}["fin"]
    root = fin.Y[0]                     # one chordwise loop
    assert abs(root.mean()) < 1e-12
    assert root.max() == pytest.approx(-root.min(), rel=1e-9)


def test_the_wing_section_is_NOT_what_the_fin_is_drawn_from():
    """The published wing section is cambered (NACA 24XX), so a fin lofted
    from it would fail the test above — which is why that one is the check
    and this one names the cause."""
    geom = _report()["geometry"]
    xc, zc = cad.section_path(geom)
    assert abs(np.asarray(zc, float).mean()) > 1e-6, (
        "the wing section is not cambered, so this file guards nothing")


def test_the_fin_is_drawn_at_the_thickness_its_drag_was_charged_at():
    geom = _report()["geometry"]
    tc = float(geom["fin"]["tc"])
    fin = {s.name: s for s in cad.surfaces(geom)}["fin"]
    got = np.ptp(fin.Y[0]) / np.ptp(fin.X[0])
    assert got == pytest.approx(tc, rel=0.03)


# ------------------------------------------------------- and in the bundle

def test_the_fin_leaves_the_bundle_as_its_own_dat():
    geom = _report()["geometry"]
    with tempfile.TemporaryDirectory() as d:
        written = cad.export(geom, d, stem="t")
        assert "dat_fin" in written
        path = pathlib.Path(written["dat_fin"])
        assert path.name == cad.export_name("t", "dat_fin") == "t_fin.dat"
        head, *rows = path.read_text().splitlines()
        assert "symmetric" in head
        pts = np.array([[float(v) for v in r.split()] for r in rows if r])
        assert abs(pts[:, 1].mean()) < 1e-12          # no camber, in the FILE


def test_a_v_tail_bundle_promises_no_fin_file():
    geom = _report("tail", V_TAIL)["geometry"]
    with tempfile.TemporaryDirectory() as d:
        assert "dat_fin" not in cad.export(geom, d, stem="t")


def test_the_openvsp_script_reads_the_fin_file_not_the_wings():
    src = cad.vsp_script(_report()["geometry"])
    assert 'AIRFOIL_FIN = os.path.join(HERE, STEM + "_fin.dat")' in src
    assert "fin_foil = AIRFOIL_FIN if os.path.exists(AIRFOIL_FIN)" in src
    assert "build_wing(FIN_SECTIONS, 0.0, \"fin\", fin_foil" in src


def test_the_stem_contract_covers_the_new_file():
    """A naming table that grows in two places is how the first one drifted."""
    assert cad.FILE_SUFFIXES["dat_fin"] == "_fin.dat"
    assert len(set(cad.FILE_SUFFIXES.values())) == len(cad.FILE_SUFFIXES)
