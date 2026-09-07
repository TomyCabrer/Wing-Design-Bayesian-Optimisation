""""3D geometry still broken when L3 is chosen + dihedral/sweep opt."

L3 is MIL-F-8785C Level 3 — the handling gate in stage 3 — and choosing it
arms the LATERAL DECK (``api._arm_lateral``), which is what finally puts a
``vlm.VerticalSurface`` in the lattice. That is the missing ingredient in the
report: the wing, the tail, the rear wing and the FIN all live in ONE panel
array, and ``api._fill_llt_geometry`` split off the tail (``is_tail``) and the
tandem's rear wing (``is_second``) and nothing else.

So the fin's panels stayed in the WING's arrays. They sit at y = 0 with z
running from the fin root to the fin tip, i.e. on the end of the wing's own
spanwise polyline — so the drawn wing reached its tip, turned ninety degrees
and climbed the fin, into a surface ``cad.fin_surface`` was ALSO drawing from
``geometry["fin"]``. Measured on `tail [free cant]` at the box centre with 12
deg of dihedral:

    handling_level unset : 40 stations, wing z max 1.039, loft z max 1.101
    handling_level = 3   : 52 stations, wing z max 1.540, loft z max 1.540
    the fin block        : z_root 0.500 + height 1.044 = 1.544

``_split_surface_panels``' own docstring describes this exactly — "exporting
it whole draws the second surface as a continuation of the first's spanwise
polyline, a surface that does not exist" — for the two surfaces it does split.

TWO things were missing, and the first is why the obvious fix does nothing:
``vlm.VLMResult`` carried ``is_tail`` and ``is_second`` but NOT
``is_vertical``. The mask existed on the model — both solvers take their aero
deck off it (``wingtail``'s ``solid``, ``tandemvlm``'s) — and stopped there,
so nothing downstream of a solve could tell a fin from a wing.

AND THE SAME HOLE EXISTED A SECOND TIME. ``wing_score._split`` — the
COMPOSITE OBJECTIVE's own splitter — knew ``is_tail`` and ``is_second`` and
not ``is_vertical`` either, so the fin was scored as wing. That one is not a
picture: measured at the same design, arming the deck moved the wing block's
narrowest chord from 0.7504 to 0.6963 m (the FIN's chord) and with it
``vol`` 0.914746 -> 0.898506 and ``bend`` 5.85118 -> 5.86476. It does not
cancel against the frozen normalisation band either, because the fin is sized
from b, S and the ARM, and the arm is a design variable of this family.

This file pins the outcome: with the deck armed the wing is the wing, the fin
is drawn once, every per-panel array is the same length, and every criterion
the search ranks is the one it ranked before.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aerobo import api, cad, vlm                                 # noqa: E402
from aerobo import wing_score as wsc                             # noqa: E402

PLAIN = "tail [free cant]"
DEVICE = "tail + winglet [free cant]"
GAMMA, SWEEP = api.WING_CANT_KEYS

#: every per-panel array the geometry block carries: they describe the SAME
#: stations, so a split that misses one leaves the report self-inconsistent
PER_PANEL = ("y", "z", "chord", "Cl_y", "alpha_eff_deg", "alpha_i_deg",
             "is_winglet")


def _geometry(name: str, *, level=None, gamma: float = 0.0,
              sweep: float = 0.0) -> dict:
    flags = {"fin": True}
    if level is not None:
        flags[api.HANDLING_LEVEL_KEY] = int(level)
    built = api.PROBLEM_SPECS[name].build({}, flags, None)
    labels = list(built.param_labels)
    x = np.asarray(built.bounds, dtype=float).mean(axis=1)
    x[labels.index(GAMMA)] = float(gamma)
    x[labels.index(SWEEP)] = float(sweep)
    cfg = api.RunConfig(problem_name=name, budget=4, seed=0, flags=flags)
    return api.design_report(cfg, x)["geometry"]


# ------------------------------------------------- the mask leaves the solver

def test_the_result_carries_the_vertical_mask():
    """It carried ``is_tail`` and ``is_second`` and stopped.

    Without this the split below is unreachable: ``_drop_vertical_panels``
    reads the mask off the RESULT, and ``getattr(..., None)`` on a result that
    does not have it is a silent no-op.
    """
    assert "is_vertical" in vlm.VLMResult.__dataclass_fields__


def test_the_flown_lattice_reports_which_panels_are_the_fin():
    from aerobo.geometry import Wing

    wing = Wing(b=8.0, S=8.0, taper=0.6)
    model = vlm.VLM(wing, N=20, V=25.0,
                    vertical=vlm.VerticalSurface(height=1.2, chord=0.7,
                                                 x=4.0, z_root=0.2, N=12))
    res = model.solve(np.deg2rad(3.0))
    assert res.is_vertical is not None
    assert res.is_vertical.shape == res.y.shape
    assert int(res.is_vertical.sum()) == 12


# -------------------------------------------- the wing is only the wing again

def test_arming_the_deck_does_not_add_stations_to_the_wing():
    """The defect, in one assertion: 40 -> 52 was the whole thing."""
    off = _geometry(PLAIN, gamma=12.0)
    on = _geometry(PLAIN, level=3, gamma=12.0)
    assert len(on["y"]) == len(off["y"]), (
        f"the fin's panels are still in the wing: {len(off['y'])} -> "
        f"{len(on['y'])} stations")


def test_the_armed_wing_is_the_same_wing_geometrically():
    """The lateral deck is free of charge to the shape, and now says so.

    ``wingtail`` states it of the aero ("CDi comes back bit-identical"); the
    geometry has to be exact, because a fin cannot move a wing station.
    """
    off = _geometry(PLAIN, gamma=12.0)
    on = _geometry(PLAIN, level=3, gamma=12.0)
    for key in ("y", "z", "chord"):
        assert np.array_equal(np.asarray(off[key], dtype=float),
                              np.asarray(on[key], dtype=float)), key


def test_no_station_sits_on_the_plane_of_symmetry():
    """The fin's signature: twelve stations at y = 0, climbing in z.

    A wing's panels are cosine-clustered about the centreline and land at
    panel CENTRES, so exactly zero of them sit at y = 0.
    """
    geom = _geometry(PLAIN, level=3, gamma=12.0)
    y = np.asarray(geom["y"], dtype=float)
    assert int(np.sum(np.abs(y) < 1e-9)) == 0, np.round(y, 4).tolist()


def test_every_per_panel_array_is_the_same_length():
    geom = _geometry(PLAIN, level=3, gamma=12.0)
    present = {k: len(geom[k]) for k in PER_PANEL if k in geom}
    assert len(set(present.values())) == 1, present
    assert present["y"] == len(geom["chord"])


# ---------------------------------------------------- and the picture is right

def test_the_drawn_wing_stops_at_its_own_tip():
    """Measured against the UNARMED report, not against the armed one.

    Comparing the loft to ``geometry["z"].max()`` of the same report cannot
    catch this: with the fin's stations in the array both numbers are the
    fin's, and the check passes while the picture is wrong. The wing's own
    tip height is a fact about the WING, so it is read off the run that has
    no fin in its lattice at all.
    """
    own = float(np.asarray(_geometry(PLAIN, gamma=12.0)["z"],
                           dtype=float).max())
    geom = _geometry(PLAIN, level=3, gamma=12.0)
    wing = [s for s in cad.surfaces(geom) if s.name == "wing"][0]
    fin_top = float(geom["fin"]["z_root_m"]) + float(geom["fin"]["height_m"])
    assert own < fin_top                         # the fin IS higher...
    assert float(wing.Z.max()) <= own + 0.15, (  # ...and the wing stops short
        f"the wing is lofted to {wing.Z.max():.4f}, its own tip is "
        f"{own:.4f}, the fin tops out at {fin_top:.4f}")


def test_the_fin_is_drawn_exactly_once():
    """One surface, one author: ``geometry["fin"]`` via ``cad.fin_surface``.

    The panel leak made it two — the block's loft AND the tail of the wing's
    own polyline — so the check is on the SHAPE, not on the surface names,
    which never changed.
    """
    geom = _geometry(PLAIN, level=3, gamma=12.0)
    surfs = cad.surfaces(geom)
    names = [s.name for s in surfs]
    assert names.count("fin") == 1, names
    assert set(names) == {"wing", "tail", "fin"}, names
    fin = [s for s in surfs if s.name == "fin"][0]
    wing = [s for s in surfs if s.name == "wing"][0]
    # nothing of the wing may stand inside the fin's own span
    inside = (wing.Z > fin.Z.min()) & (np.abs(wing.Y) < 0.2)
    assert not inside.any(), (
        f"{int(inside.sum())} wing points stand on the fin's line")


def test_the_armed_loft_is_the_unarmed_loft():
    off = [s for s in cad.surfaces(_geometry(PLAIN, gamma=12.0))
           if s.name == "wing"][0]
    on = [s for s in cad.surfaces(_geometry(PLAIN, level=3, gamma=12.0))
          if s.name == "wing"][0]
    assert np.allclose(off.Z, on.Z) and np.allclose(off.Y, on.Y)


def test_it_holds_with_a_tip_device_and_a_sweep_too():
    """The device family reaches the path loft either way, so it was broken
    even before a dihedral was searched — its ``is_winglet`` mask is what let
    the fin's panels through as a continuation of the winglet arc."""
    off = _geometry(DEVICE, gamma=12.0, sweep=15.0)
    on = _geometry(DEVICE, level=3, gamma=12.0, sweep=15.0)
    assert len(on["y"]) == len(off["y"])
    assert len(on["is_winglet"]) == len(on["y"])
    wing = [s for s in cad.surfaces(on) if s.name == "wing"][0]
    assert float(wing.Z.max()) < float(on["fin"]["z_root_m"]) \
        + float(on["fin"]["height_m"])


# ------------------------------------------------ and nothing else moved

def test_a_design_with_no_vertical_is_untouched():
    """The mask is absent or all-False and the split must be a no-op."""
    geom = _geometry(PLAIN, gamma=12.0)
    assert geom.get("fin")                      # a fin is REPORTED...
    keep = np.ones(len(geom["y"]), dtype=bool)   # ...but not in the lattice

    class _NoMask:
        is_vertical = None

    assert np.array_equal(api._drop_vertical_panels(_NoMask(), keep), keep)

    class _AllFalse:
        is_vertical = np.zeros(len(geom["y"]), dtype=bool)

    assert np.array_equal(api._drop_vertical_panels(_AllFalse(), keep), keep)


def test_the_masked_view_covers_the_surface_masks_it_publishes():
    """``is_second``/``is_vertical`` were missing from ``_PER_PANEL``, so a
    reader that took one off a masked result got a FULL-LENGTH boolean."""
    per = api._MaskedPanels._PER_PANEL
    for key in ("is_tail", "is_second", "is_vertical", "is_winglet"):
        assert key in per, (key, per)


def test_the_water_families_never_had_the_defect():
    """A hydrofoil's MAST is a vertical surface and is NOT optional — but it
    is charged as drag and never panelled, so its reports were always clean.
    Pinned so a future mast in the lattice does not land silently."""
    for name in ("hydrofoil", "hydrofoil + elevator"):
        built = api.PROBLEM_SPECS[name].build({}, {}, None)
        x = np.asarray(built.bounds, dtype=float).mean(axis=1)
        geom = api.design_report(
            api.RunConfig(problem_name=name, budget=4, seed=0), x)["geometry"]
        y = np.asarray(geom["y"], dtype=float)
        assert int(np.sum(np.abs(y) < 1e-9)) == 0, (name, y[:6])
        assert len(geom["Cl_y"]) == y.size, name


# ------------------------------------------ and what the SEARCH is ranked on

def _raw(name: str, *, level=None, gamma: float = 0.0):
    flags = {"fin": True}
    if level is not None:
        flags[api.HANDLING_LEVEL_KEY] = int(level)
    built = api.PROBLEM_SPECS[name].build({}, flags, None)
    labels = list(built.param_labels)
    x = np.asarray(built.bounds, dtype=float).mean(axis=1)
    x[labels.index(GAMMA)] = float(gamma)
    return built.evaluate(x)


def test_the_composite_scores_the_same_design_the_same_way():
    """Arming the lateral deck may add ``spiral`` and must move NOTHING else.

    ``vol`` read 0.914746 unarmed and 0.898506 armed, and ``bend`` 5.85118 vs
    5.86476, purely because a vertical plank at y = 0 was in the block called
    "wing".
    """
    off = wsc.design_metrics(_raw(PLAIN, gamma=12.0))
    on = wsc.design_metrics(_raw(PLAIN, level=3, gamma=12.0))
    # to a RELATIVE 1e-12, not exactly: ``wingtail`` states the price of the
    # deck as "CDi comes back bit-identical and CL, e, alpha and SM move by at
    # most one ulp", and ``espan`` is that e — it lands 2 ulp apart
    # (0.8020430587193247 vs 0.802043058719325). The defect was 1.78 % of
    # ``vol``, so this tolerance is four orders clear of what it has to catch.
    moved = {}
    for k in set(off) | set(on):
        if k == "spiral":
            continue
        a, b = off.get(k), on.get(k)
        if a is None or b is None:
            if a is not b:
                moved[k] = (a, b)
        elif abs(float(a) - float(b)) > 1e-12 * max(1.0, abs(float(a))):
            moved[k] = (a, b)
    assert not moved, moved
    # ...and the deck DID arm, or this test proves nothing
    assert off.get("spiral") is None and on.get("spiral") is not None


def test_the_scored_blocks_are_the_lifting_surfaces():
    """No fin block, no fin stations — and the TAIL still splits off.

    That last clause is not padding: the first fix trimmed the surface before
    reading ``is_tail``, ``_mask`` refused the now-wrong-length array, and the
    stabiliser's 24 panels went back into the wing (40 -> 64, vol 0.648209).
    """
    blocks = wsc.surfaces_of(_raw(PLAIN, level=3, gamma=12.0))
    names = [b["name"] for b in blocks]
    assert names == ["wing", "tail"], names
    for b in blocks:
        assert int(np.sum(np.abs(b["y"]) < 1e-9)) == 0, b["name"]
    off = wsc.surfaces_of(_raw(PLAIN, gamma=12.0))
    assert [b["y"].size for b in off] == [b["y"].size for b in blocks]


def test_the_reported_alpha_range_is_the_range_the_gate_tested():
    """``wingtail`` quoted the RAW array while its own gate was masked and
    ``tandemvlm`` reported the masked one. It moves nothing measurable (102
    of 102 Sobol draws agree) and it is now one number with one author."""
    off = _raw(PLAIN, gamma=12.0)["alpha_eff_range_deg"]
    on = _raw(PLAIN, level=3, gamma=12.0)["alpha_eff_range_deg"]
    assert tuple(off) == tuple(on), (off, on)


# ------------------------------------------------------------- and the tandem

def test_the_pair_keeps_its_two_wings_and_neither_grows_a_fin():
    """The audit called this the same leak twice as large, and nothing pinned
    it: a pair's fin panels land in the FRONT wing."""
    name = "tandem (nonplanar) [free cant]"
    for level in (None, 3):
        flags = {} if level is None else {api.HANDLING_LEVEL_KEY: int(level)}
        built = api.PROBLEM_SPECS[name].build({}, flags, None)
        x = np.asarray(built.bounds, dtype=float).mean(axis=1)
        geom = api.design_report(
            api.RunConfig(problem_name=name, budget=4, seed=0, flags=flags),
            x)["geometry"]
        front = np.asarray(geom["y"], dtype=float)
        rear = np.asarray((geom.get("second_surface") or {}).get("y"),
                          dtype=float)
        assert int(np.sum(np.abs(front) < 1e-9)) == 0, (level, front[:6])
        assert rear.size and int(np.sum(np.abs(rear) < 1e-9)) == 0, level
        assert front.size == rear.size, (level, front.size, rear.size)
        blocks = [b["name"] for b in
                  wsc.surfaces_of(built.evaluate(x))]
        assert blocks == ["wing", "rear"], (level, blocks)


# ------------------------------------------------- the OTHER way in

def test_the_spiral_weight_arms_the_same_deck_and_is_covered_too():
    """There are TWO routes to a fin in the lattice, not one.

    ``api._needs_lateral`` fires on ``handling_level`` OR on a non-zero
    ``spiral`` weight in the composite. The second one matters because it
    reaches families the handling card never appears on: `tail + CST section
    (XFOIL)` does not declare ``handling_level`` at all (``sanitise_flags``
    strips it) but declares the composite keys and wraps a ``WingTailProblem``
    with a ``lateral`` attribute — so a weight arms it while no card is on
    screen. Pinning the route, not just the level.
    """
    weights = {"lod": 0.7, "spiral": 0.3}
    base = {"fin": True, "wing_objective": "composite",
            "wing_score_weights": weights}
    assert api.wants_spiral(base) and not api.wants_spiral({"fin": True})
    armed = dict(base, wing_score_reference=api.wing_score_reference(
        PLAIN, flags=base, n=16))

    built = api.PROBLEM_SPECS[PLAIN].build({}, armed, None)
    labels = list(built.param_labels)
    x = np.asarray(built.bounds, dtype=float).mean(axis=1)
    x[labels.index(GAMMA)] = 12.0
    geom = api.design_report(
        api.RunConfig(problem_name=PLAIN, budget=4, seed=0, flags=armed),
        x)["geometry"]
    y = np.asarray(geom["y"], dtype=float)
    assert int(np.sum(np.abs(y) < 1e-9)) == 0
    assert y.size == len(_geometry(PLAIN, gamma=12.0)["y"])
    assert [b["name"] for b in wsc.surfaces_of(built.evaluate(x))] == \
        ["wing", "tail"]


def test_a_family_with_no_handling_card_can_still_be_armed():
    """The 768: armable by weight, and the level is not even a flag on them.

    Structural, so it costs no XFOIL sweep — the point is that the section
    family takes the SAME solver, and therefore the same two splitters, as the
    family the rest of this file measures.
    """
    name = "tail + CST section (XFOIL)"
    spec = api.PROBLEM_SPECS[name]
    assert api.HANDLING_LEVEL_KEY not in spec.flags
    assert all(k in spec.flags for k in ("wing_objective",
                                         "wing_score_weights",
                                         "wing_score_reference"))
    inner = getattr(spec.build({}, {}, None).problem, "wing_tail", None)
    assert inner is not None and hasattr(inner, "lateral")
