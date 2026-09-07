""""solver gets broken when dihedral is selected to optimise — the wing
connects to the vertical tail geometry."

The solver was not broken. The DRAWERS were, and both halves of the cant were
missing from every one of them.

``cad.nonplanar_arrays`` asks one question — "did a TIP DEVICE fly" — because
a device was the only way out of the wing plane when it was written. A wing
DIHEDRAL is the other one, and it has been a design variable since the cant
became two design-box rows. A design that carries one WITHOUT a device
answered that question False, so ``cad.wing_path`` set ``z = zeros`` and the
loft drew a flat wing: measured on `tail [free cant]` at 15 deg, the report
states z rising to 1.2931 m and the loft returned z in [-0.0523, +0.0976] —
bit-identical to the planar design. That is the STL, the OpenVSP script, the
3-D view and stage 6's flight scene all showing an aeroplane the solver did
not fly. Only the FRONT view escaped, and by accident: it synthesises its
wing straight from the report's own arrays rather than asking the gate.

SWEEP was worse: no drawer had it at all. ``loft_surface``/``loft_path`` took
a scalar ``x0`` (a tandem's stagger) and nothing put the quarter-chord line
where the lattice puts it (``vlm.VLM``: ``xe = |ye| tan Lambda``), so the
lofted wing spanned x [-0.3076, +0.9228] at 0, 10 AND 20 deg of sweep.

This file pins:

* a dihedral the solver flew is a dihedral the loft draws, read off the
  REPORT's own z array rather than re-derived from the angle;
* a sweep the solver flew is a sweep the loft draws, at the lattice's rule;
* a planar, unswept design takes the SAME branch it always did, so no
  published export moved, and a REDUCED-ORDER sweep is still not drawn --
  a lifting line's quarter-chord line really is straight;
* the second-surface blocks are NOT caught by the new gate — their z array is
  absolute AND applied again as z0, so widening the shared gate would draw
  them at twice their height;
* a per-station x0 shifts stations along the SPAN, never across the chord;
* the OpenVSP rebuild carries the same sweep the STL beside it does;
* the front view, which was already right, stays right.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aerobo import api, cad                                      # noqa: E402

#: a free-cant family with NO tip device — the configuration whose dihedral
#: no drawer could see
PLAIN = "tail [free cant]"
#: ...and one WITH a device, which could, so the fix has to leave it alone
DEVICE = "tail + winglet [free cant]"
GAMMA, SWEEP = api.WING_CANT_KEYS


def _geometry(name: str, gamma: float = 0.0, sweep: float = 0.0) -> dict:
    """The design report's geometry at the box centre, with the cant stated."""
    flags = {"fin": True}
    built = api.PROBLEM_SPECS[name].build({}, flags, None)
    labels = list(built.param_labels)
    x = np.asarray(built.bounds, dtype=float).mean(axis=1)
    x[labels.index(GAMMA)] = float(gamma)
    x[labels.index(SWEEP)] = float(sweep)
    cfg = api.RunConfig(problem_name=name, budget=4, seed=0, flags=flags)
    return api.design_report(cfg, x)["geometry"]


def _wing(geom: dict):
    surf = [s for s in cad.surfaces(geom) if s.name == "wing"]
    assert surf, "the export carries no wing"
    return surf[0]


# ------------------------------------------------------ the dihedral is drawn

def test_a_dihedral_without_a_tip_device_reaches_the_loft():
    """The defect, in one assertion: the flat wing was the WHOLE fix."""
    flat = _wing(_geometry(PLAIN, gamma=0.0))
    canted = _wing(_geometry(PLAIN, gamma=15.0))
    assert canted.Z.max() > flat.Z.max() + 1.0, (
        f"the loft is still flat: {flat.Z.max():.4f} -> {canted.Z.max():.4f}")


def test_the_drawn_height_is_the_reported_one_not_a_re_derivation():
    """Read off the report's own path, so the picture cannot drift from it."""
    geom = _geometry(PLAIN, gamma=15.0)
    reported = float(np.asarray(geom["z"], dtype=float).max())
    assert reported > 1.0, reported          # the solver flew it
    drawn = float(_wing(geom).Z.max())
    # the loft adds the section's own half-thickness on top of the path
    assert reported <= drawn <= reported + 0.15, (reported, drawn)


def test_the_device_family_is_untouched():
    """It already drew its cant; the fix must not move it."""
    geom = _geometry(DEVICE, gamma=15.0)
    assert cad.nonplanar_arrays(geom) is not None
    reported = float(np.asarray(geom["z"], dtype=float).max())
    assert float(_wing(geom).Z.max()) >= reported


# --------------------------------------------------------- the sweep is drawn

def test_a_swept_wing_is_lofted_swept():
    unswept = _wing(_geometry(DEVICE, sweep=0.0))
    swept = _wing(_geometry(DEVICE, sweep=20.0))
    assert swept.X.max() > unswept.X.max() + 1.0, (
        f"the loft is still unswept: {unswept.X.max():.4f} -> "
        f"{swept.X.max():.4f}")


def test_the_sweep_offset_is_the_lattices_own_rule():
    """``xe = |ye| tan Lambda`` — vlm.VLM, on the planform-view y."""
    y = np.array([-4.0, -1.0, 0.0, 2.5])
    got = cad.sweep_offset({"sweep_deg": 20.0, "z": np.zeros_like(y)}, y)
    assert np.allclose(got, np.abs(y) * np.tan(np.deg2rad(20.0)))


def test_a_reduced_order_sweep_is_not_drawn():
    """A LIFTING LINE's quarter-chord line really is straight.

    Sweep reaches it as simple-sweep theory on the section slope and a drag
    build-up, never as geometry: `wing t/c + sweep` states 25 deg and reports
    no panel path at all. Drawing that swept would export a wing that was
    never flown — the warning ``cad.vsp_script`` has been carrying. The
    discriminator is the path: ``geometry["z"]`` exists exactly where a
    lattice bent the line.
    """
    name = "wing t/c + sweep"
    built = api.PROBLEM_SPECS[name].build({}, {}, None)
    labels = list(built.param_labels)
    x = np.asarray(built.bounds, dtype=float).mean(axis=1)
    x[labels.index("sweep_deg")] = 25.0
    geom = api.design_report(
        api.RunConfig(problem_name=name, budget=4, seed=0), x)["geometry"]
    assert float(geom["sweep_deg"]) == 25.0
    assert "z" not in geom
    assert cad.sweep_offset(geom, np.asarray(geom["y"], dtype=float)) == 0.0


# ------------------------------------------ and nothing published moved

def test_a_planar_unswept_design_takes_the_branch_it_always_took():
    geom = _geometry(PLAIN, gamma=0.0, sweep=0.0)
    assert cad.canted_arrays(geom) is None, (
        "a planar wing must not be re-routed through the path loft")
    y = np.asarray(geom["y"], dtype=float)
    assert cad.sweep_offset(geom, y) == 0.0


def test_the_second_surface_blocks_are_not_caught_by_the_new_gate():
    """Their z is ABSOLUTE and their height is applied again as z0.

    ``geometry["tail_surface"]["z"]`` is 0.5 everywhere on a tail whose
    ``z_offset`` is 0.5, so a gate that said "any non-zero z is a path" would
    draw the surface at 1.0 m. The wing's gate is the wing's.
    """
    geom = _geometry(PLAIN, gamma=15.0)
    tail_sf = geom.get("tail_surface") or {}
    z = np.asarray(tail_sf.get("z"), dtype=float)
    z0 = float(tail_sf.get("z_offset", 0.0))
    assert z.size and np.allclose(z, z0) and z0 > 0.0, (z0, z[:3])
    tail = [s for s in cad.surfaces(geom) if s.name == "tail"]
    assert tail, "the export carries no tail"
    assert abs(float(np.median(tail[0].Z)) - z0) < 0.2, (
        "the tail is drawn at twice its height")


# ------------------------------------------------- the per-station offset

def test_a_per_station_x0_shifts_the_span_not_the_chord():
    y = np.linspace(-3.0, 3.0, 7)
    c = np.full(y.size, 0.8)
    tw = np.zeros(y.size)
    xc = np.linspace(0.0, 1.0, 7)             # SAME count as the span, on
    zc = np.zeros_like(xc)                    # purpose: a bare `+ x0` would
    base = cad.loft_surface(y, c, tw, xc, zc)[0]        # broadcast across it
    dx = np.arange(y.size, dtype=float)
    got = cad.loft_surface(y, c, tw, xc, zc, x0=dx)[0]
    assert np.allclose(got - base, dx[:, None])


# ------------------------------------------------------------ the front view

def test_the_front_view_already_drew_the_cant_and_still_does():
    """The one view that was NOT broken, pinned so the fix did not break it.

    ``fig_frontview`` sends a device-less design to ``_fig_frontview_tail``,
    which synthesises its wing straight from ``geometry["y"]``/``["z"]`` — so
    it has always drawn the dihedral. That fallback is also where a V-tail's
    panels are folded and drawn dotted, which the path branch does not do, so
    routing this view through the path would have cost more than it bought.
    """
    from gui import nice_app as v1

    geom = _geometry(PLAIN, gamma=15.0)
    fig = v1.fig_frontview(geom)
    assert fig is not None
    wing = [t for t in fig.data if t.name in ("wing", "winglet")]
    assert wing, [t.name for t in fig.data]
    z = np.concatenate([np.asarray(t.y, dtype=float).ravel() for t in wing])
    assert z.max() > 1.0, (
        f"the front view draws a flat wing (z max {z.max():.4f})")


# --------------------------------------------------------- the OpenVSP script

def test_the_vsp_script_writes_the_sweep_the_lattice_flew():
    """It wrote 0 and said sweep "is never the lifting-surface geometry".

    True of a lifting line and false of the lattice, which has bent the
    quarter-chord line since ``vlm.VLM`` gained ``xe = |ye| tanL`` — so the
    native rebuild disagreed with the STL beside it, on the same claim the
    loft was getting wrong.
    """
    geom = _geometry(PLAIN, gamma=15.0, sweep=20.0)
    script = cad.vsp_script(geom)
    assert "SWEEP_DEG = 20.0" in script, [
        ln for ln in script.splitlines() if "SWEEP_DEG" in ln]


def test_the_vsp_script_still_refuses_a_reduced_order_sweep():
    name = "wing t/c + sweep"
    built = api.PROBLEM_SPECS[name].build({}, {}, None)
    labels = list(built.param_labels)
    x = np.asarray(built.bounds, dtype=float).mean(axis=1)
    x[labels.index("sweep_deg")] = 25.0
    geom = api.design_report(
        api.RunConfig(problem_name=name, budget=4, seed=0), x)["geometry"]
    script = cad.vsp_script(geom)
    assert "SWEEP_DEG = 0.0" in script
    assert "not in its geometry" in script
