"""CAD export (cad.py) — what has to be true for a file to leave the tool.

An exported STL is a claim about a shape, and a claim in a file outlives the
session that made it. So the tests are about the ways that claim could be
false:

  1. it must be the SAME surface the 3-D view draws — the GUI's loft helpers
     are this module (delegation, not a copy), or the picture and the file
     would drift apart silently;
  2. the solid must be CLOSED and CORRECTLY WOUND — every edge shared by
     exactly two facets, outward normals, and an enclosed volume that matches
     the strip integral of the section area. A mesher that has to guess is a
     mesher that will guess wrong on the winglet;
  3. it must carry what was FLOWN — the winglet at its height and cant, the
     tail at its arm and height, a canard ahead of the wing, the elevator at
     its hinge station, the optimised section rather than a stand-in;
  4. the OpenVSP script must be VALID PYTHON with dependency-free data, and
     its section chain must reproduce the span, the chord and the cant;
  5. it must fail as a message, not a traceback, on geometry it cannot draw.

Cheap by construction: everything below builds geometry dicts of the shape
``api.design_report`` returns, and only the two integration tests pay for a
physics evaluation.
"""

import struct
from collections import Counter

import numpy as np
import pytest

from aerobo import cad


# ------------------------------------------------------------------ helpers

def _geom(**over) -> dict:
    """A plain planar wing report."""
    g = {"b": 10.0, "S": 10.0, "taper": 0.5, "twist_root_deg": 2.0,
         "twist_tip_deg": -1.0, "tc": 0.12}
    g.update(over)
    return g


def _winglet_geom(h=1.0, cant=75.0, n=41) -> dict:
    """A nonplanar report: one continuous path, winglets marked, port to
    starboard, exactly as the VLM exports its panel stations."""
    from aerobo import geometry

    semis, b = 5.0, 10.0
    s = np.linspace(0.0, semis + h, n)
    y, z = geometry.span_path(s, semis, h, cant)
    mask = s > semis
    c = 2.0 * (1.0 - 0.5 * np.clip(s / semis, 0.0, 1.0))
    ys = np.concatenate([-y[::-1], y[1:]])
    zs = np.concatenate([z[::-1], z[1:]])
    cs = np.concatenate([c[::-1], c[1:]])
    ms = np.concatenate([mask[::-1], mask[1:]])
    return {"b": b, "S": 10.0, "y": ys.tolist(), "z": zs.tolist(),
            "chord": cs.tolist(), "is_winglet": [bool(v) for v in ms],
            "tc": 0.12, "twist_root_deg": 0.0, "twist_tip_deg": -2.0,
            "winglet": {"h_m": h, "cant_deg": cant, "blend_frac": 0.0}}


def _tail_block(x0=6.0, z0=0.5, b_t=3.0, c_t=0.8, n=21) -> dict:
    y = np.linspace(-b_t / 2, b_t / 2, n)
    return {"name": "tail", "y": y.tolist(),
            "chord": np.full(n, c_t).tolist(),
            "x_offset": x0, "z_offset": z0}


def edge_histogram(T, tol=9) -> Counter:
    """How many facets share each edge — 2 everywhere iff the mesh is closed."""
    def key(p):
        return (round(float(p[0]), tol), round(float(p[1]), tol),
                round(float(p[2]), tol))

    edges = Counter()
    for tri in T:
        v = [key(p) for p in tri]
        for a, b in ((0, 1), (1, 2), (2, 0)):
            edges[tuple(sorted((v[a], v[b])))] += 1
    return Counter(edges.values())


def signed_volume(T) -> float:
    return float(np.einsum("ij,ij->i", T[:, 0],
                           np.cross(T[:, 1], T[:, 2])).sum() / 6.0)


def parse_binary_stl(data: bytes):
    n = struct.unpack("<I", data[80:84])[0]
    tris = np.zeros((n, 3, 3))
    normals = np.zeros((n, 3))
    for i in range(n):
        off = 84 + i * 50
        vals = struct.unpack("<12f", data[off:off + 48])
        normals[i] = vals[:3]
        tris[i] = np.asarray(vals[3:]).reshape(3, 3)
        assert data[off + 48:off + 50] == b"\x00\x00"
    return normals, tris


# ------------------------------------------- 1. one loft, shared with the GUI

def test_gui_delegates_its_loft_to_this_module():
    """The 3-D view and the export must draw the same surface. Not "produce
    the same numbers" — BE the same code, so they cannot drift."""
    from gui import nice_app as v1

    g = _geom()
    y, c = cad.planform_arrays(g)
    xc, zc = cad.naca4_section(0.12)
    tw = cad.wing_twist(g, y)
    a = v1._loft_surface(y, c, tw, xc, zc)
    b = cad.loft_surface(y, c, tw, xc, zc)
    for u, v in zip(a, b):
        assert np.array_equal(u, v)
    assert np.array_equal(v1._naca4_section(0.12)[0], cad.naca4_section(0.12)[0])
    assert v1._tc_for_view(g) == cad.tc_for_view(g)
    assert np.array_equal(v1._planform_arrays(g)[0], y)


def test_the_nonplanar_loft_reduces_to_the_planar_one():
    """A path with z = 0 and no winglet is a flat wing: the two lofts must
    agree to machine zero, or the winglet modes are drawn on a different
    surface from every other mode."""
    g = _geom()
    y, c = cad.planform_arrays(g)
    xc, zc = cad.naca4_section(0.12)
    tw = cad.wing_twist(g, y)
    flat = cad.loft_surface(y, c, tw, xc, zc)
    path = cad.loft_path(y, np.zeros_like(y), c, tw, xc, zc,
                         np.zeros(y.size, dtype=bool))
    for u, v in zip(flat, path):
        assert np.allclose(u, v, atol=1e-12)


def test_section_sits_where_the_breakdown_says_it_does():
    """LE a quarter chord ahead of the origin, TE three quarters behind, and
    nose-up twist raises the leading edge."""
    y = np.array([0.0])
    c = np.array([2.0])
    xc, zc = cad.naca4_section(0.12)
    X, _Y, Z = cad.loft_surface(y, c, np.zeros(1), xc, zc)
    assert X.min() == pytest.approx(-0.5, abs=1e-12)     # -0.25 * 2
    assert X.max() == pytest.approx(1.5, abs=1e-12)      # +0.75 * 2
    up = cad.loft_surface(y, c, np.deg2rad([5.0]), xc, zc)
    le = int(np.argmin(X[0]))
    assert up[2][0, le] > Z[0, le]                       # LE lifted


# --------------------------------------------- 2. the solid, and its winding

@pytest.mark.parametrize("geom, part", [
    (_geom(), "wing"),
    (_winglet_geom(), "wing"),
])
def test_every_closed_solid_is_watertight(geom, part):
    surf = next(s for s in cad.surfaces(geom) if s.name == part)
    T = cad.triangles(surf)
    hist = edge_histogram(T)
    assert set(hist) == {2}, f"open mesh: edge multiplicities {dict(hist)}"


def test_normals_point_out_and_match_the_winding():
    surf = cad.surfaces(_geom())[0]
    T = cad.triangles(surf)
    assert signed_volume(T) > 0.0
    normals, tris = parse_binary_stl(cad.stl_bytes([surf]))
    n_from_winding = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n_from_winding /= np.linalg.norm(n_from_winding, axis=1, keepdims=True)
    # STL stores float32: the tolerance is the format's, not the geometry's
    assert np.allclose(normals, n_from_winding, atol=1e-4)


def test_enclosed_volume_is_the_strip_integral():
    """The mesh must enclose the wing it says it does: the section's area times
    c^2, integrated over the span, to within the facet resolution."""
    g = _geom(twist_root_deg=0.0, twist_tip_deg=0.0)
    surf = cad.surfaces(g, section_n=101)[0]
    V = signed_volume(cad.triangles(surf))
    xc, zc = cad.naca4_section(0.12, n=101)
    area = 0.5 * abs(np.sum(xc * np.roll(zc, -1) - np.roll(xc, -1) * zc))
    y, c = cad.planform_arrays(g)
    V_ref = float(np.trapezoid(area * c ** 2, y))
    assert V == pytest.approx(V_ref, rel=2e-3)


def test_an_open_trailing_edge_is_stitched_not_left_gaping():
    """A CST section with a finite TE thickness has two distinct TE points;
    the loop still has to close, or the solid leaks along the whole span."""
    from aerobo import airfoil

    n = 4
    coords = airfoil.cst_coords(np.full(n, 0.15), np.full(n, -0.15),
                                n_pts=120, dz_te=0.01)
    surf = cad.surfaces(_geom(), section=coords)[0]
    assert set(edge_histogram(cad.triangles(surf))) == {2}


def test_open_sheets_are_left_open_and_unflipped():
    """The elevator plate encloses nothing; capping it would invent a solid the
    physics never modelled."""
    tail = {"type": "conventional", "control": "elevator",
            "elevator_chord_frac": 0.3, "delta_e_deg": -5.0, "i_t_deg": 0.0}
    surfs = cad.tail_surfaces(tail, _tail_block(), *cad.naca4_section(0.10))
    plate = next(s for s in surfs if s.name == "elevator")
    assert plate.closed is False
    T = cad.triangles(plate)
    assert set(edge_histogram(T)) != {2}          # a membrane, not a solid
    assert T.shape[0] == 2 * (plate.shape[0] - 1) * (plate.shape[1] - 1)


def test_binary_and_ascii_carry_the_same_facets():
    surfs = cad.surfaces(_geom())
    _n, tris = parse_binary_stl(cad.stl_bytes(surfs, binary=True))
    text = cad.stl_bytes(surfs, binary=False)
    assert isinstance(text, str)
    assert text.count("facet normal") == tris.shape[0]
    assert text.startswith("solid wing")
    assert text.rstrip().endswith("endsolid wing")


def test_ascii_names_every_part_separately():
    """Binary STL has no parts; ASCII does, and "which solid is the elevator"
    is exactly what a reviewer asks first."""
    g = _winglet_geom()
    g["tail"] = {"type": "conventional", "control": "elevator",
                 "elevator_chord_frac": 0.25, "delta_e_deg": 2.0}
    g["tail_surface"] = _tail_block()
    text = cad.stl_bytes(cad.surfaces(g), binary=False)
    assert "solid wing" in text and "solid tail" in text
    assert "solid elevator" in text


def test_degenerate_facets_are_dropped():
    """A repeated station or a sharp TE makes zero-area facets — legal STL,
    rejected by every mesher downstream."""
    y = np.array([0.0, 0.0, 1.0])          # a repeated station
    c = np.array([1.0, 1.0, 1.0])
    xc, zc = cad.naca4_section(0.12)
    X, Y, Z = cad.loft_surface(y, c, np.zeros(3), xc, zc)
    T = cad.triangles(cad.Surface("s", X, Y, Z))
    areas = 0.5 * np.linalg.norm(np.cross(T[:, 1] - T[:, 0],
                                          T[:, 2] - T[:, 0]), axis=1)
    assert (areas > cad.MIN_FACET_AREA).all()


# --------------------------------------------------- 3. it carries the flight

def test_the_winglet_is_exported_at_its_height_and_cant():
    h, cant = 1.2, 60.0
    surf = cad.surfaces(_winglet_geom(h=h, cant=cant))[0]
    from aerobo import geometry

    tip_z = geometry.winglet_tip_height(h, cant)
    # the tip section straddles the path point, so the surface reaches above it
    assert surf.Z.max() > 0.9 * tip_z
    assert surf.Z.max() < tip_z + 0.5      # not a flat span extension either
    assert surf.Y.max() < 5.0 + h          # cant eats projected span


def test_a_canard_exports_ahead_and_an_aft_tail_behind():
    for x0, ahead in ((-4.0, True), (6.0, False)):
        g = _geom()
        g["tail"] = {"type": "canard" if ahead else "conventional",
                     "control": "stabilator", "i_t_deg": 0.0}
        g["tail_surface"] = _tail_block(x0=x0)
        tail = next(s for s in cad.surfaces(g) if s.name == "tail")
        wing = next(s for s in cad.surfaces(g) if s.name == "wing")
        assert bool(tail.X.mean() < wing.X.min()) is ahead


def test_the_elevator_plate_hinges_where_the_solver_hinged_it():
    frac, c_t, x0 = 0.3, 0.8, 6.0
    tail = {"type": "conventional", "control": "elevator",
            "elevator_chord_frac": frac, "delta_e_deg": 0.0, "i_t_deg": 0.0}
    surfs = cad.tail_surfaces(tail, _tail_block(x0=x0, c_t=c_t),
                              *cad.naca4_section(0.10))
    plate = next(s for s in surfs if s.name == "elevator")
    hinge = x0 + (0.75 - frac) * c_t
    assert plate.X.min() == pytest.approx(hinge, abs=1e-12)
    assert plate.X.max() == pytest.approx(x0 + 0.75 * c_t, abs=1e-12)


def test_a_deflected_elevator_moves_the_right_way():
    """delta_e > 0 is trailing edge DOWN — the sign the trim solve reports."""
    base = {"type": "conventional", "control": "elevator",
            "elevator_chord_frac": 0.3, "i_t_deg": 0.0}
    down = cad.tail_surfaces({**base, "delta_e_deg": 10.0}, _tail_block(),
                             *cad.naca4_section(0.10))[-1]
    up = cad.tail_surfaces({**base, "delta_e_deg": -10.0}, _tail_block(),
                           *cad.naca4_section(0.10))[-1]
    assert down.Z.min() < up.Z.min()


def test_a_v_tail_exports_two_dihedralled_panels():
    tail = {"type": "v_tail", "control": "stabilator", "dihedral_deg": 35.0,
            "b_t": 3.0, "S_t": 2.0, "i_t_deg": 0.0}
    g = _geom()
    g["tail"] = tail
    g["tail_surface"] = _tail_block(z0=0.4)
    surf = next(s for s in cad.surfaces(g) if s.name == "tail")
    rise = 1.5 * np.tan(np.deg2rad(35.0))
    assert surf.Z.max() > 0.4 + 0.8 * rise
    assert set(edge_histogram(cad.triangles(surf))) == {2}


def test_a_tail_with_its_own_tip_device_is_lofted_not_flattened():
    """The designed tail can carry a winglet of its own; averaging it into the
    panel height would export a surface the solver did not fly."""
    g = _geom()
    g["tail"] = {"type": "conventional", "control": "stabilator",
                 "i_t_deg": 0.0}
    blk = _tail_block(z0=0.5, b_t=3.0)
    y = np.asarray(blk["y"])
    z = np.full(y.size, 0.5)
    z[np.abs(y) > 1.3] = 0.5 + 0.4                     # a crude tip device
    blk["z"] = z.tolist()
    blk["is_winglet"] = [bool(v) for v in np.abs(y) > 1.3]
    g["tail_surface"] = blk
    surf = next(s for s in cad.surfaces(g) if s.name == "tail")
    assert surf.Z.max() > 0.85


def test_the_optimised_section_is_the_one_exported():
    """A design that flew a CST section must not export a NACA stand-in."""
    from aerobo import airfoil

    coords = np.asarray(airfoil.cst_coords(np.array([0.30, 0.24, 0.22, 0.30]),
                                           np.array([-0.10, -0.05, -0.08,
                                                     -0.10]), n_pts=140))
    xc, zc = cad.section_path(_geom(), section=coords)
    assert xc.size == coords.shape[0]
    assert np.array_equal(zc, coords[:, 1])
    # the same section, handed over as a whole section REPORT
    xr, _zr = cad.section_path(_geom(),
                               section={"design": {"coords": coords.tolist()}})
    assert np.array_equal(xr, coords[:, 0])


def test_the_section_dat_round_trips():
    xc, zc = cad.naca4_section(0.12)
    text = cad.airfoil_dat(xc, zc, "check")
    rows = [ln.split() for ln in text.strip().split("\n")[1:]]
    pts = np.asarray(rows, dtype=float)
    assert pts.shape == (xc.size, 2)
    assert np.allclose(pts[:, 0], xc, atol=1e-6)
    assert pts[0, 0] == pytest.approx(1.0)         # Selig order: TE first
    assert pts[np.argmin(pts[:, 0]), 0] == pytest.approx(0.0, abs=1e-9)


# ------------------------------------------------- 4. the OpenVSP rebuild

def test_the_vsp_script_is_valid_dependency_free_python():
    src = cad.vsp_script(_winglet_geom(), stem="check")
    compile(src, "check_vsp.py", "exec")            # syntax
    data = "\n".join(ln for ln in src.split("\n")
                     if ln.startswith(("WING_SECTIONS", "TAIL_SECTIONS",
                                       "    dict(", "]", "WING_ROOT",
                                       "TAIL_ROOT", "TAIL_X", "TAIL_Z",
                                       "ELEVATOR", "SWEEP_DEG")))
    ns: dict = {}
    exec(data, ns)                                  # noqa: S102 — the point
    assert ns["WING_SECTIONS"]                      # no numpy scalars needed
    assert all(isinstance(v, float)
               for s in ns["WING_SECTIONS"] for v in
               (s["span"], s["root_chord"], s["dihedral_deg"]))


def test_the_vsp_sections_reproduce_span_chord_and_cant():
    h, cant = 1.0, 75.0
    g = _winglet_geom(h=h, cant=cant)
    secs = cad.wing_sections(g)
    # the chain is the panel POLYLINE, so it is the developed span less the
    # corner the stations cut across — within one panel, and never more
    total = sum(s["span"] for s in secs)
    panel = (5.0 + h) / 40.0
    assert total <= 5.0 + h + 1e-9
    assert total > 5.0 + h - 2.0 * panel
    assert secs[0]["root_chord"] == pytest.approx(2.0, rel=1e-6)
    wl = [s for s in secs if s["is_winglet"]]
    assert wl, "the tip device must survive the reduction"
    assert wl[-1]["dihedral_deg"] == pytest.approx(cant, abs=1e-6)
    assert max(s["dihedral_deg"] for s in secs if not s["is_winglet"]) \
        == pytest.approx(0.0, abs=1e-9)


def test_the_reduction_reports_what_it_costs():
    """A polynomial chord law cannot survive a linear loft between a dozen
    sections, so the script has to SAY how much it missed by."""
    g = _geom()
    y = np.linspace(-5.0, 5.0, 61)
    eta = np.abs(y / 5.0)
    g["y"] = y.tolist()
    g["chord"] = (2.0 * (1.0 - 0.6 * eta ** 3)).tolist()   # strongly curved
    few = cad.wing_sections(g, max_sections=3)
    many = cad.wing_sections(g, max_sections=24)
    assert few[0]["chord_error"] > many[0]["chord_error"]
    assert f"{many[0]['chord_error']:.2%}" in cad.vsp_script(g,
                                                             max_sections=24)


def test_the_junction_lands_on_a_section_boundary():
    """The corner is a panel EDGE: a section running from mid-wing across it
    would have a linear loft cut the tip device's corner off."""
    secs = cad.wing_sections(_winglet_geom(h=1.0, cant=75.0))
    straddling = [s for s in secs
                  if 5.0 < s["dihedral_deg"] < 70.0]
    assert len(straddling) <= 1
    if straddling:
        assert straddling[0]["span"] < 0.2      # half a panel either side


def test_a_tail_reaches_the_script_with_its_arm_and_hinge():
    g = _geom()
    g["tail"] = {"type": "conventional", "control": "elevator",
                 "elevator_chord_frac": 0.3, "delta_e_deg": -4.0,
                 "i_t_deg": 0.0}
    g["tail_surface"] = _tail_block(x0=6.0, z0=0.5)
    src = cad.vsp_script(g)
    assert "TAIL_X = 6.0" in src
    assert "TAIL_Z = 0.5" in src
    assert "ELEVATOR_CHORD_FRAC = 0.3" in src
    assert "-4.0" in src                       # the deflection is recorded
    assert "add_elevator" in src


# ----------------------------------------------- 5. failures are messages

def test_geometry_it_cannot_draw_is_a_message_not_a_traceback(tmp_path):
    assert cad.surfaces({}) == []
    assert cad.wing_sections({}) == []
    with pytest.raises(ValueError, match="no geometry"):
        cad.export({}, tmp_path)


def test_empty_surfaces_still_write_a_valid_stl():
    data = cad.stl_bytes([])
    assert struct.unpack("<I", data[80:84])[0] == 0
    assert len(data) == 84


# ------------------------------------------------------- the whole bundle

def test_export_writes_the_bundle_and_it_all_parses(tmp_path):
    g = _winglet_geom()
    g["tail"] = {"type": "conventional", "control": "elevator",
                 "elevator_chord_frac": 0.25, "delta_e_deg": 3.0,
                 "i_t_deg": 0.0}
    g["tail_surface"] = _tail_block()
    out = cad.export(g, tmp_path, stem="demo")
    for key in ("stl", "stl_ascii", "stl_wing", "stl_tail", "stl_elevator",
                "dat", "vsp", "readme"):
        assert key in out, key
        assert (tmp_path / f"{out[key].split('/')[-1]}").exists()
    _n, tris = parse_binary_stl((tmp_path / "demo.stl").read_bytes())
    per_part = sum(parse_binary_stl(
        (tmp_path / f"demo_{p}.stl").read_bytes())[1].shape[0]
        for p in ("wing", "tail", "elevator"))
    assert tris.shape[0] == per_part
    compile((tmp_path / "demo_vsp.py").read_text(), "demo_vsp.py", "exec")
    readme = (tmp_path / "demo_README.txt").read_text()
    assert "Stereolith" in readme and "x downstream" in readme
    assert "wing" in readme and "elevator" in readme


# ------------------------------------------------ integration: real solvers

def test_a_real_wing_tail_design_exports_every_surface_it_flew():
    """One physics evaluation, through the same path the result page uses."""
    from aerobo import api

    name = "tail + winglet"
    flags = {"control": "elevator", "elevator_chord_frac": 0.3}
    built = api.PROBLEM_SPECS[name].build({}, flags, None)
    x = np.array([0.5 * (lo + hi) for lo, hi in built.bounds])
    rep = api.design_report(api.RunConfig(problem_name=name, flags=flags), x)
    surfs = cad.surfaces(rep["geometry"], x, rep["param_labels"])
    names = [s.name for s in surfs]
    # the FIN joined this list in V5: it is now a surface the design carries
    # (``geometry["fin"]``, the same one whose parasite drag the drag book
    # sizes), so "every surface it flew" includes it.
    assert names == ["wing", "tail", "elevator", "fin"]
    for s in surfs:
        if s.closed:
            assert set(edge_histogram(cad.triangles(s))) == {2}, s.name
    wing, tail = surfs[0], surfs[1]
    assert tail.X.min() > wing.X.max()               # the arm is to scale
    assert wing.Z.max() > 0.2                        # the winglet is up there


def test_the_hydrofoil_elevator_is_exported_too():
    """It reports no ``tail_type`` — the second surface must still leave with
    its arm, or the water aeroplane exports as a lone foil.

    THREE surfaces now, not two: the STRUT is a reported surface since the
    mast stopped being charged in silence, and a craft that exports without
    the thing holding it out of the water exports a foil nobody can build.
    """
    from aerobo import api

    name = "hydrofoil + elevator"
    built = api.PROBLEM_SPECS[name].build({}, {}, None)
    x = np.array([0.5 * (lo + hi) for lo, hi in built.bounds])
    rep = api.design_report(api.RunConfig(problem_name=name), x)
    assert rep["geometry"].get("tail"), "the elevator lost its scalars"
    surfs = cad.surfaces(rep["geometry"], x, rep["param_labels"])
    assert [s.name for s in surfs] == ["wing", "tail", "fin"]
    assert surfs[1].X.min() > surfs[0].X.max()
    # the strut is the MAST, and it is lofted at the wetted area the run was
    # charged for: it runs from the foil UP through the submergence depth
    blk = rep["geometry"]["fin"]
    assert blk["kind"] == "mast"
    assert surfs[2].Z.max() == pytest.approx(blk["height_m"], rel=1e-9)


# ------------------------- 6. panel CENTRES are not the surface's EDGE
#
# Everything below was found by building the model in OpenVSP 3.51.2 and
# measuring it: the export is right when VSP's own TotalSpan / TotalArea come
# back as the design's numbers, and it was not, by 3.7 % of span and 6.7 % of
# tail area, until the ends were treated properly.

def test_the_chain_starts_on_the_plane_of_symmetry():
    """VSP mirrors the half-span chain: starting it at the innermost STATION
    (half a panel out) builds a wing that much narrower."""
    g = _winglet_geom()
    secs = cad.wing_sections(g)
    h = float(g["winglet"]["h_m"])
    assert sum(s["span"] for s in secs) == pytest.approx(5.0 + h, rel=1e-9)


def test_the_tail_chain_carries_the_area_the_drag_was_charged_on():
    g = _geom()
    g["tail"] = {"type": "conventional", "control": "stabilator",
                 "i_t_deg": 0.0, "b_t": 3.0, "S_t": 2.4}
    # stations are centres, and inboard of both edges
    y = np.linspace(-1.4, 1.4, 15)
    g["tail_surface"] = {"name": "tail", "y": y.tolist(),
                         "chord": np.full(y.size, 0.8).tolist(),
                         "x_offset": 6.0, "z_offset": 0.5}
    sec = cad._tail_sections(g)[0]
    assert sec["span"] == pytest.approx(1.5, rel=1e-9)          # b_t / 2
    area = 2 * sec["span"] * 0.5 * (sec["root_chord"] + sec["tip_chord"])
    assert area == pytest.approx(2.4, rel=1e-9)                 # S_t


def test_extending_to_the_edge_never_invents_geometry():
    """It fires on a half-panel deficit and refuses anything else — a
    ``developed_span`` that does not belong to these arrays must not silently
    stretch them."""
    y = np.linspace(-4.9, 4.9, 21)
    z = np.zeros_like(y)
    c = np.full(y.size, 1.0)
    m = np.zeros(y.size, dtype=bool)
    ext = cad.extend_to_edges(y, z, c, m, 10.0)
    assert ext[0][0] == pytest.approx(-5.0)
    assert ext[0][-1] == pytest.approx(5.0)
    assert float(np.sum(np.hypot(np.diff(ext[0]), np.diff(ext[1])))) \
        == pytest.approx(10.0)
    for bad in (9.0, 25.0):        # already past it / not this surface
        same = cad.extend_to_edges(y, z, c, m, bad)
        assert np.array_equal(same[0], y)
    assert np.array_equal(cad.extend_to_edges(y, z, c, m, float("nan"))[0], y)


def test_the_chord_is_extrapolated_at_the_rate_the_stations_set():
    y = np.linspace(-4.9, 4.9, 21)
    c = 2.0 - 0.2 * np.abs(y)                       # straight taper
    ext = cad.extend_to_edges(y, np.zeros_like(y), c,
                              np.zeros(y.size, dtype=bool), 10.0)
    assert ext[2][-1] == pytest.approx(2.0 - 0.2 * 5.0, rel=1e-9)


def test_the_build_script_spreads_its_panels_and_resolves_the_device():
    """A chain of sections with VSP's defaults clusters panels at every joint
    and gives a short tip device two of them; both were measured to cost span
    efficiency in VSPAERO."""
    src = cad.vsp_script(_winglet_geom())
    assert "SectTess_U" in src and "InCluster" in src and "OutCluster" in src
    # both knobs are declared in the script, so a reader can change them
    assert "DEVICE_PANELS = " in src and "SPAN_PANELS = " in src
    assert "is_winglet" in src                 # the floor is applied by it
    ns: dict = {}
    exec("\n".join(ln for ln in src.split("\n")
                   if ln.startswith(("SPAN_PANELS", "DEVICE_PANELS"))), ns)
    assert ns["DEVICE_PANELS"] >= 4 and ns["SPAN_PANELS"] >= 10
