"""What the geometry views DRAW must be what the solver flew.

Three defects motivated these, all of them "the picture disagrees with the
numbers" rather than a crash:

  1. FRAME. The lofts were drawn with +X FORWARD but the axis was labelled
     "x", which is downstream-positive everywhere else in the package: an
     aft tail at l_t = +5 m appeared at x = -5 and every aerofoil looked as
     if it were flying backwards. The views are now in the package's axes.
  2. THE V-TAIL was drawn as the flat equivalent surface the solver flies —
     no V anywhere — so the one thing the layout choice is about was
     invisible. It is now drawn as its two dihedralled panels, built from
     the same numbers the physics uses (S_t, its projection, Gamma).
  3. THE BLENDED WINGLET was lofted with a crease at the junction, because
     the loft differenced the wing and winglet segments separately to
     preserve a KINK — exactly the kink a blend exists not to have.

Plus: a tail solved in the same nonplanar system (wingtail.py) must be
exported and drawn as its own surface, not appended to the wing's spanwise
polyline.
"""

import numpy as np
import pytest

from aerobo import api, cad
from gui import nice_app as v1

X_TAIL = np.array([0.6, 1.0, -2.0, 1.5, 5.0])


def _report(name, x, **flags):
    cfg = api.RunConfig(problem_name=name, optimiser="random", budget=2,
                        seed=0, flags=flags)
    return api.design_report(cfg, x)


def _section_at(trace, y_target=0.0):
    """One chordwise loop: ``(x, thickness-axis)`` for whichever axis it is.

    A horizontal surface carries its section thickness in z and its span in
    y; a FIN is that same loft turned ninety degrees, so its thickness is in
    Y and its span in Z. Reading z for both reports a fin as infinitely thin
    and its nose as nowhere. The thickness axis is the one that VARIES along
    a chordwise row, which is a property of the trace rather than a list of
    surface names to keep in step.
    """
    X = np.asarray(trace.x)
    Y = np.asarray(trace.y)
    Z = np.asarray(trace.z)
    vertical = np.ptp(Y[0]) > np.ptp(Z[0])          # a fin: y is the thickness
    span = Z[:, 0] if vertical else Y[:, 0]
    j = int(np.argmin(np.abs(span - (span.mean() if vertical else y_target))))
    return X[j], (Y[j] if vertical else Z[j])


# ------------------------------------------------------------ 1. the frame

def test_sections_face_upstream_in_the_packages_axes():
    """LE upstream of the TE, for the wing AND the tail, in one frame."""
    rep = _report("tail", X_TAIL, control="elevator")
    fig = v1.fig_wing3d(rep["geometry"], X_TAIL,
                        api.PROBLEM_SPECS["tail"].param_labels)
    for tr in fig.data:
        if getattr(tr, "name", None) == "elevator":
            continue
        xs, zs = _section_at(tr)
        lo, hi = xs.min(), xs.max()
        thick_front = np.ptp(zs[xs < lo + 0.2 * (hi - lo)])
        thick_back = np.ptp(zs[xs > hi - 0.2 * (hi - lo)])
        assert thick_front > thick_back, "the round nose must sit UPSTREAM"


def test_the_tail_is_drawn_aft_and_a_canard_ahead():
    aft = _report("tail", X_TAIL)
    fig = v1.fig_wing3d(aft["geometry"], X_TAIL,
                        api.PROBLEM_SPECS["tail"].param_labels)
    tail_x = max(float(np.asarray(tr.x).mean()) for tr in fig.data)
    assert tail_x > 4.0                      # l_t = +5 m, drawn at +5 m
    canard = _report("tail", X_TAIL, tail_type="canard")
    fig = v1.fig_wing3d(canard["geometry"], X_TAIL,
                        api.PROBLEM_SPECS["tail"].param_labels)
    canard_x = min(float(np.asarray(tr.x).mean()) for tr in fig.data)
    assert canard_x < -4.0                   # upstream, as l_t = -5 m says


def test_the_plan_view_uses_the_same_frame_as_the_numbers():
    rep = _report("tail", X_TAIL)
    fig = v1.fig_planform(rep["geometry"], X_TAIL,
                          api.PROBLEM_SPECS["tail"].param_labels)
    tail_block = rep["geometry"]["tail"]
    named = {tr.name: tr for tr in fig.data if tr.name}
    assert float(np.asarray(named["tail"].y).mean()) > 4.0
    for nm, key in (("CG", "x_cg"), ("neutral point", "x_np")):
        assert float(np.asarray(named[nm].y)[0]) == \
            pytest.approx(tail_block[key], rel=1e-9)


def test_the_elevator_plate_hangs_off_the_trailing_edge():
    rep = _report("tail", X_TAIL, control="elevator")
    fig = v1.fig_wing3d(rep["geometry"], X_TAIL,
                        api.PROBLEM_SPECS["tail"].param_labels)
    plate = next(tr for tr in fig.data if getattr(tr, "name", "") == "elevator")
    tail = next(tr for tr in fig.data
                if getattr(tr, "name", "") in ("tail", "V-tail panels"))
    assert np.asarray(plate.x).max() == pytest.approx(
        np.asarray(tail.x).max(), rel=0.02)      # both end at the TE
    assert np.asarray(plate.x).min() > np.asarray(tail.x).min()


# ---------------------------------------------------------- 2. the V-tail

def test_the_v_tail_is_drawn_as_two_dihedralled_panels():
    rep = _report("tail", X_TAIL, tail_type="v_tail", dihedral_deg=35.0)
    geom = rep["geometry"]
    fig = v1.fig_wing3d(geom, X_TAIL, api.PROBLEM_SPECS["tail"].param_labels)
    vee = next(tr for tr in fig.data
               if getattr(tr, "name", "") == "V-tail panels")
    Y, Z = np.asarray(vee.y), np.asarray(vee.z)
    gam = np.deg2rad(35.0)
    tail = geom["tail"]
    # the tips rise by |y| tan(Gamma) above the tail's own height
    z0 = tail["dz_m"]
    tip = np.abs(Y).max()
    assert Z.max() == pytest.approx(z0 + tip * np.tan(gam), rel=0.05)
    assert Z.min() == pytest.approx(z0, abs=0.05)
    # ...and the panels' HORIZONTAL projection is the flat equivalent's span.
    # Measured off the LOFTED SKIN, so it carries the section's own thickness,
    # which a panel canted at 35 deg projects sideways: half a 12 % section on
    # a 0.66 m chord is ~0.04 m, and sin(35 deg) of that is ~0.023 m per side
    # on a 2.0 m span — about 2.5 %. Which side that thickness sits on now
    # depends on which way up the section is mounted (tail.tail_polar — a
    # downloading surface flies it inverted), so the tolerance has to admit
    # the skin rather than assume a zero-thickness panel.
    assert 2.0 * tip == pytest.approx(tail["b_t"], rel=0.05)


def test_the_v_panels_carry_the_panel_area_the_drag_is_charged_on():
    tail = {"type": "v_tail", "dihedral_deg": 35.0, "b_t": 2.0065,
            "S_t": 1.5}
    y, z, c, mask = v1._v_tail_path(tail, 0.5)
    length = float(np.sum(np.hypot(np.diff(y), np.diff(z))))
    assert length * c[0] == pytest.approx(tail["S_t"], rel=1e-3)
    assert v1._v_tail_path(dict(tail, type="conventional"), 0.5) is None
    assert v1._v_tail_path(dict(tail, dihedral_deg=0.0), 0.5) is None


# ------------------------------------------------------ 3. the blend loft

def test_a_blended_junction_is_lofted_without_a_crease():
    """The section normals must rotate CONTINUOUSLY through a blended
    junction; a sharp one keeps its crease."""
    y = np.linspace(-1.0, 1.0, 41)
    z = np.where(np.abs(y) > 0.5, (np.abs(y) - 0.5), 0.0)   # kinked path
    mask = np.abs(y) > 0.5
    _, n_sharp = v1._path_normals(y, z, mask, smooth=False)
    _, n_smooth = v1._path_normals(y, z, mask, smooth=True)
    jump_sharp = np.max(np.abs(np.diff(n_sharp[:, 1])))
    jump_smooth = np.max(np.abs(np.diff(n_smooth[:, 1])))
    assert jump_smooth < jump_sharp


def test_the_blended_winglet_view_uses_the_smooth_loft():
    name = "winglet, blended (span-capped)"
    x = np.array([0.6, 1.0, -2.0, 0.15, 70.0, 1.0])
    rep = _report(name, x, blend_shape="smooth", junction_drag=True)
    fig = v1.fig_wing3d(rep["geometry"], x,
                        api.PROBLEM_SPECS[name].param_labels)
    surf = fig.data[0]
    Y, Z = np.asarray(surf.y), np.asarray(surf.z)
    # walk the leading-edge line outboard: no step in the section's tilt
    le_y, le_z = Y[:, Y.shape[1] // 2], Z[:, Z.shape[1] // 2]
    d = np.hypot(np.diff(le_y), np.diff(le_z))
    assert np.max(d) < 5.0 * np.median(d)      # no jump at the junction
    assert "blended" in fig.layout.title.text


# ------------------------------------- 4. wing + winglet + tail in one view

def test_a_tail_in_the_nonplanar_solve_is_exported_as_its_own_surface():
    name = "tail + winglet, blended (span-capped) [free height]"
    x = np.array([0.6, 1.0, -2.0, 1.5, 5.0, 1.5, 0.13, 70.0, 0.9])
    geom = _report(name, x, junction_drag=True, blend_shape="smooth")["geometry"]
    assert "tail_surface" in geom
    ts = geom["tail_surface"]
    assert ts["x_offset"] == pytest.approx(5.0)
    assert ts["z_offset"] == pytest.approx(1.5)
    # the wing arrays must NOT contain the tail panels any more: every
    # station of the wing path is inside the wing + winglet span
    y = np.asarray(geom["y"], float)
    assert len(y) == len(geom["chord"]) == len(geom["is_winglet"])
    assert np.abs(y).max() > 4.0                 # wing tips, not tail tips
    assert np.abs(np.asarray(ts["y"], float)).max() < 2.0


def test_the_wing_winglet_and_tail_are_all_drawn_together():
    name = "tail + winglet [free height]"
    x = np.array([0.6, 1.0, -2.0, 1.5, 5.0, 1.5, 0.13, 70.0])
    rep = _report(name, x)
    fig = v1.fig_wing3d(rep["geometry"], x, api.PROBLEM_SPECS[name].param_labels)
    assert fig is not None
    xs = np.concatenate([np.asarray(tr.x).ravel() for tr in fig.data])
    zs = np.concatenate([np.asarray(tr.z).ravel() for tr in fig.data])
    assert xs.max() > 4.0                       # the tail is in the picture
    assert zs.max() > 1.0                       # ...and so is its height
    assert "tail" in fig.layout.title.text
    # the plan view shows it too, at its arm
    plan = v1.fig_planform(rep["geometry"], x,
                           api.PROBLEM_SPECS[name].param_labels)
    assert max(float(np.asarray(tr.y).max()) for tr in plan.data) > 4.0


# ------------------------------------- 4. a PLANAR main surface, second one

@pytest.mark.parametrize("name", [
    "hydrofoil + elevator",
    "hydrofoil + elevator [designed elevator + tip device]",
])
def test_the_elevator_is_drawn_when_the_main_surface_is_planar(name):
    """The three views each had a branch for a NONPLANAR main surface (which
    draws the second one) and a branch for a planar one (which did not).

    A foiling craft with an elevator but no tip device on the foil takes the
    planar branch, so every view drew the foil alone — the surface whose arm
    and depth are the design variables was invisible in the only pictures
    that show them.
    """
    spec = api.PROBLEM_SPECS[name]
    built = spec.build(None, {}, None)
    x = 0.5 * (built.bounds[:, 0] + built.bounds[:, 1])
    geom = _report(name, x)["geometry"]
    assert "tail_surface" in geom and geom.get("tail")
    arm = float(geom["tail"]["l_t"])
    assert arm > 0.0

    # 3-D: the elevator's own surface, at its arm and below the foil
    fig = v1.fig_wing3d(geom, x, spec.param_labels)
    tail = next(tr for tr in fig.data if getattr(tr, "name", "") == "tail")
    assert np.asarray(tail.x).max() == pytest.approx(arm, rel=0.5)
    assert np.asarray(tail.z).mean() < 0.0          # it hangs BELOW the foil
    assert "elevator" in fig.layout.title.text

    # plan view: a second outline, at that arm
    plan = v1.fig_planform(geom, x, spec.param_labels)
    assert any(getattr(tr, "name", "") == "tail" for tr in plan.data)
    assert max(float(np.asarray(tr.y).max()) for tr in plan.data) > 0.5 * arm

    # front view: the DEPTH the elevator sits at is what this view is for
    front = v1.fig_frontview(geom)
    assert front is not None
    zt = next(tr for tr in front.data if getattr(tr, "name", "") == "tail")
    assert float(np.asarray(zt.y).mean()) < 0.0
    # ...and it says which side of the wing plane that is, from the geometry
    # rather than from a hard-coded word
    assert "below the wing plane" in front.layout.title.text


def _spanwise_cl(fig, label: str):
    """The drawn local-Cl curve of surface ``label``, or None."""
    for tr in fig.data:
        nm = str(getattr(tr, "name", ""))
        if nm.startswith(label) and "local Cl" in nm:
            return np.asarray(tr.y, float)
    return None


@pytest.mark.parametrize("name,x", [
    # a PLANAR main surface: the foil carries no tip device, so the whole
    # design reaches the loading view through geom["y"]/geom["Cl_y"]
    ("hydrofoil + elevator", None),
    # ...and a NONPLANAR one: the wing's winglet series was drawn and the
    # tail's, sitting in the very same report, was not
    ("tail + winglet [free height]",
     np.array([0.6, 1.0, -2.0, 1.5, 5.0, 1.5, 0.13, 70.0])),
])
def test_the_loading_view_draws_every_surface_that_was_flown(name, x):
    """The spanwise view had the same two-branch shape the plan, front and
    3-D views were fixed for: the published lifting-line pair (which exports
    a two-entry ``surfaces`` list) drew both surfaces, and every family that
    solves its second surface in the SAME panel array — exporting it as
    ``tail_surface`` — drew the wing alone.

    So a foiling craft's elevator and an air tail flown beside a winglet
    were both dropped from the one view their loading lives in, with no note
    saying so, while their Cl_y sat in the same report colouring the 3-D
    picture two views away. The caption calls this the loading "across the
    span of the design that was scored", and that design has two surfaces.
    """
    spec = api.PROBLEM_SPECS[name]
    if x is None:
        box = spec.default_bounds
        x = np.array([0.5 * (box[l][0] + box[l][1])
                      for l in spec.param_labels])
    geom = _report(name, x)["geometry"]
    ts = geom["tail_surface"]
    assert ts.get("Cl_y"), "the second surface reported no loading"
    assert not geom.get("surfaces"), "this must be the tail_surface branch"

    fig = v1.fig_spanwise(geom)
    assert fig is not None
    drawn = [tr for tr in fig.data
             if "local Cl" in str(getattr(tr, "name", ""))]
    assert len(drawn) >= 2, [str(tr.name) for tr in fig.data]

    # and the drawn numbers ARE the solver's own, station for station —
    # not a re-derivation of them
    tail_cl = _spanwise_cl(fig, ts.get("name", "tail"))
    assert tail_cl is not None, [str(tr.name) for tr in fig.data]
    wl = np.asarray(ts.get("is_winglet") or [], bool)
    flat = ~wl if wl.size == len(ts["y"]) else np.ones(len(ts["y"]), bool)
    assert tail_cl == pytest.approx(
        np.asarray(ts["Cl_y"], float)[flat], abs=1e-12)


def test_the_second_surfaces_tip_device_is_not_a_spanwise_station():
    """A tip device's panels all sit at (nearly) the SAME y, so charting
    them against y glues a spike to the tip. The wing's winglet has been
    split out into its own series against tip + arc height since the
    nonplanar modes landed; a designed elevator carries a tip device of its
    own and must be split the same way, off its OWN plane — the elevator
    hangs below the foil, so its panels' z is around -0.06 m and an arc
    height measured from z = 0 would place the device inboard of the tip.
    """
    name = "hydrofoil + elevator [designed elevator + tip device]"
    spec = api.PROBLEM_SPECS[name]
    box = spec.default_bounds
    x = [0.5 * (box[l][0] + box[l][1]) for l in spec.param_labels]
    x[spec.param_labels.index("winglet_h_frac_t")] = 0.12
    x[spec.param_labels.index("winglet_cant_t_deg")] = 70.0
    geom = _report(name, x)["geometry"]
    ts = geom["tail_surface"]
    y = np.asarray(ts["y"], float)
    wl = np.asarray(ts["is_winglet"], bool)
    assert wl.any(), "the tip device was not flown"

    fig = v1.fig_spanwise(geom)
    tail_cl = _spanwise_cl(fig, ts.get("name", "tail"))
    # the curve is the FLAT panels only...
    assert tail_cl.size == int((~wl).sum())
    # ...and the device is its own series, outboard of the tail tip
    dev = [tr for tr in fig.data
           if "tip device" in str(getattr(tr, "name", ""))
           and "tail" in str(getattr(tr, "name", ""))]
    assert dev, [str(tr.name) for tr in fig.data]
    xs = np.concatenate([np.asarray(tr.x, float) for tr in dev])
    assert np.abs(xs).min() >= np.abs(y[~wl]).max() - 1e-9


@pytest.mark.parametrize("cant", [70.0, -70.0])
def test_the_second_surfaces_tip_device_is_drawn_where_its_cant_puts_it(cant):
    """The front view is the (y, z) plane — the ONE view a cant angle lives
    in — and it drew the second surface as a flat segment at its offset, so
    a designed elevator's tip device was invisible in the only picture that
    could show it. Signed cant: + is up, - is down (geometry.winglet_path).
    """
    name = "hydrofoil + elevator [designed elevator + tip device]"
    spec = api.PROBLEM_SPECS[name]
    box = spec.default_bounds
    x = [0.5 * (box[l][0] + box[l][1]) for l in spec.param_labels]
    x[spec.param_labels.index("winglet_h_frac_t")] = 0.12
    x[spec.param_labels.index("winglet_cant_t_deg")] = cant
    geom = _report(name, x)["geometry"]

    ts = geom["tail_surface"]
    z = np.asarray(ts["z"], float)
    wl = np.asarray(ts["is_winglet"], bool)
    flat = float(z[~wl].mean())
    assert wl.any(), "the tip device was not flown"

    front = v1.fig_frontview(geom)
    dev = [tr for tr in front.data
           if getattr(tr, "name", "") == "tail tip device"]
    assert dev, "the tip device is not drawn"
    tips = np.concatenate([np.asarray(tr.y, float) for tr in dev])
    # the drawn device leaves the tail plane on the side its cant says
    if cant > 0.0:
        assert tips.max() > flat + 1e-4
        assert z[wl].max() > flat
    else:
        assert tips.min() < flat - 1e-4
        assert z[wl].min() < flat
    # the drawing IS the solver's own path, not a re-derivation of it: the
    # extreme station on the cant's side is the solver's own extreme (the
    # other end of the trace is the junction it shares with the tail plane)
    assert (tips.max() if cant > 0.0 else tips.min()) == pytest.approx(
        z[wl].max() if cant > 0.0 else z[wl].min(), abs=1e-9)


def test_the_elevators_height_reaches_the_captions_that_quote_it():
    """``dz_m`` is the height every layout caption is drawn from. The air
    tail reports it as ``dz_tail``; the hydrofoil's elevator reports the same
    quantity as ``z_t``, and without the fallback every view said "h 0.00 m"
    about a surface hanging below the foil."""
    name = "hydrofoil + elevator"
    spec = api.PROBLEM_SPECS[name]
    box = spec.default_bounds
    x = [0.5 * (box[l][0] + box[l][1]) for l in spec.param_labels]
    geom = _report(name, x)["geometry"]
    dz = geom["tail"]["dz_m"]
    assert dz < 0.0                                   # below the foil
    assert dz == pytest.approx(geom["tail_surface"]["z_offset"], abs=1e-9)


# --------------------------------- 5. a V-tail that ALSO carries tip devices

#: the one configuration that states its layout twice — a butterfly tail with
#: a device on each panel tip. Both shapes leave the tail's own plane, and the
#: views had one branch each: ``cad.tail_surfaces`` took the surface's OWN
#: path (device, no V) and ``fig_frontview`` took ``v_tail_path`` (V, no
#: device). So the 3-D view drew a FLAT tail with two little devices on it —
#: the dihedral the whole layout is about was nowhere in the picture — and the
#: front view drew the V with the devices deleted. Neither was the design.
V_TAIL_TIP = "tail [designed tail + tip device] + free chord law"
V_FLAGS = {"tail_type": "v_tail", "dihedral_deg": 35.0}


def _v_tail_tip_geom(name=V_TAIL_TIP, **flags):
    spec = api.PROBLEM_SPECS[name]
    box = spec.default_bounds
    x = [0.5 * (box[l][0] + box[l][1]) for l in spec.param_labels]
    x[spec.param_labels.index("winglet_h_frac_t")] = 0.12
    return _report(name, x, **flags)["geometry"], x, spec.param_labels


def test_a_v_tail_that_carries_a_tip_device_is_still_drawn_as_a_V():
    geom, x, labels = _v_tail_tip_geom(**V_FLAGS)
    tail, ts = geom["tail"], geom["tail_surface"]
    wl = np.asarray(ts["is_winglet"], bool)
    assert wl.any(), "the tip device was not flown — nothing to collide with"
    gam = np.deg2rad(V_FLAGS["dihedral_deg"])
    z0 = float(ts["z_offset"])
    tip = z0 + 0.5 * float(tail["b_t"]) * np.tan(gam)

    y, z, _c, drawn_wl, _seg = cad.tail_path(tail, ts)
    # the PANEL reaches the height Gamma puts it at (the stations are panel
    # centres, so the last one falls half a panel short of the tip)
    assert float(z[~drawn_wl].max()) == pytest.approx(tip, rel=0.02)
    # ...and the device is STILL there, above the panel it grows from
    assert drawn_wl.any() and float(z.max()) > float(z[~drawn_wl].max())
    # the panels' horizontal projection is the flat equivalent's span, which
    # is the number the pitch effectiveness came from
    assert 2.0 * float(np.abs(y[~drawn_wl]).max()) == pytest.approx(
        float(tail["b_t"]), rel=0.02)

    # and that is the shape the 3-D view is lofted on
    surf = next(s for s in cad.tail_surfaces(tail, ts, *cad.naca4_section(0.1))
                if s.name != "elevator")
    flat = _report(V_TAIL_TIP, x, tail_type="conventional")["geometry"]
    flat_surf = next(
        s for s in cad.tail_surfaces(flat["tail"], flat["tail_surface"],
                                     *cad.naca4_section(0.1))
        if s.name != "elevator")
    assert float(surf.Z.max()) - z0 > 2.0 * (float(flat_surf.Z.max())
                                             - float(flat["tail_surface"]
                                                     ["z_offset"]))


def test_the_folded_v_carries_the_panel_area_and_the_device_it_was_flown_with():
    """The fold is a re-statement of what was flown, not a redesign of it.

    The panels total S_t — the area the profile drag was charged on, which is
    :func:`cad.v_tail_path`'s contract — and the device keeps the arc length
    and the cant RELATIVE to its panel that the solver gave it.
    """
    geom, _x, _l = _v_tail_tip_geom(**V_FLAGS)
    tail, ts = geom["tail"], geom["tail_surface"]
    gam = np.deg2rad(V_FLAGS["dihedral_deg"])
    yf, zf, cf, mf = cad.nonplanar_arrays(ts)
    yv, zv, cv, mv, _seg = cad.tail_path(tail, ts)

    def area(y, z, c, sel):
        ds = np.hypot(np.diff(y[sel]), np.diff(z[sel]))
        return float(np.sum(0.5 * (c[sel][:-1] + c[sel][1:]) * ds))

    # the flown flat surface is S_t cos^2 Gamma; the panels drawn from it are
    # S_t. Both are read off panel CENTRES, so both are a couple of per cent
    # short of the closed form — the point is which number they are short OF.
    assert area(yf, zf, cf, ~mf) == pytest.approx(
        float(tail["S_t"]) * np.cos(gam) ** 2, rel=0.03)
    assert area(yv, zv, cv, ~mv) == pytest.approx(float(tail["S_t"]), rel=0.03)

    dev = np.flatnonzero(mv)
    star = dev[yf[dev] > 0.0]
    assert star.size > 1, "no starboard tip device was flown"
    arc = lambda y, z, s: float(np.sum(np.hypot(np.diff(y[s]), np.diff(z[s]))))
    assert arc(yv, zv, star) == pytest.approx(arc(yf, zf, star), rel=1e-9)
    ang = lambda y, z, s: np.rad2deg(np.arctan2(z[s[-1]] - z[s[0]],
                                                y[s[-1]] - y[s[0]]))
    assert ang(yv, zv, star) - ang(yf, zf, star) == pytest.approx(
        V_FLAGS["dihedral_deg"], abs=1e-6)


#: the front view has TWO entry points and each had the bug the other way
#: round, so both are driven here: a planar wing falls to the wing+tail view,
#: a wing with a device of its own to the nonplanar one.
@pytest.mark.parametrize("name", [
    V_TAIL_TIP, "tail + winglet [designed tail + tip device]"])
def test_the_front_view_draws_the_V_and_its_devices_in_one_path(name):
    """The front view is the (y, z) plane, so BOTH angles live in it."""
    geom, _x, _l = _v_tail_tip_geom(name, **V_FLAGS)
    ts = geom["tail_surface"]
    z0 = float(ts["z_offset"])
    fig = v1.fig_frontview(geom)
    panel = [tr for tr in fig.data if getattr(tr, "name", "") == "tail"]
    dev = [tr for tr in fig.data
           if getattr(tr, "name", "") == "tail tip device"]
    assert panel and dev, [str(tr.name) for tr in fig.data]
    zs = np.concatenate([np.asarray(tr.y, float) for tr in panel])
    assert zs.max() > z0 + 0.3, "the panels are drawn flat"
    # the device is drawn ABOVE the panel it grows from, not off a flat tail
    assert np.concatenate(
        [np.asarray(tr.y, float) for tr in dev]).max() > zs.max()
    # ...and the panels are not ALSO claimed by the dotted annotation, which
    # exists for the tail that reports no path of its own
    assert not [tr for tr in fig.data if "V panels" in str(tr.name)]


def test_the_v_panels_are_not_labelled_a_tip_device():
    """``v_tail_path``'s fourth array is the ROOT crease, not an is-winglet
    flag. Handed to the front view as one, it drew the starboard panel in the
    device's colour and called it "tail tip device"."""
    geom, _x, _l = _v_tail_tip_geom(**V_FLAGS)
    tail = geom["tail"]
    y, _z, _c, wl, seg = cad.tail_path(tail, {"y": [-1.0, 1.0],
                                              "z_offset": 0.4})
    assert not wl.any(), "a V panel is not a tip device"
    assert len(set(seg.tolist())) == 2, "the root vertex is a crease"
    assert set(np.sign(y[seg == seg[0]]).tolist()) <= {-1.0, 0.0}


def test_a_conventional_tail_with_a_device_is_untouched_by_the_fold():
    geom, _x, _l = _v_tail_tip_geom(tail_type="conventional")
    ts = geom["tail_surface"]
    y, z, c, wl = cad.nonplanar_arrays(ts)
    yv, zv, cv, wv, seg = cad.tail_path(geom["tail"], ts)
    assert np.allclose(yv, y) and np.allclose(zv, z) and np.allclose(cv, c)
    assert np.array_equal(wv, wl) and np.array_equal(seg, wl.astype(int))
