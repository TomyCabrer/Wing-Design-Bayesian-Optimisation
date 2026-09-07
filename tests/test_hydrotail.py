"""Hydrofoil + ELEVATOR: the stabiliser is really in the solve.

The claim this file gates is that hydrotail.py composes three things that
already existed — the free-surface image, a second surface in the same
influence matrix, and the per-panel cavitation margin — into ONE solve, and
that each of them is actually doing work:

1. The stabiliser trims the craft in PITCH: Cm about the CG is zero at the
   reported (alpha, i_t), and the static margin comes off the same solve.
2. The stabiliser is a lifting surface in the same system, not a bookkeeping
   term: it carries lift, it appears in the drag, and removing its area
   changes the trim.
3. Cavitation is judged at EVERY panel's own submergence, so the stabiliser —
   which sits deeper — has more static head than the main foil.
4. The design box is honest: the SM = SM_min boundary is interior (both
   feasible and infeasible designs exist), which is the calibration claim in
   the module docstring.
5. Penalty contract, registry wiring and the GUI mapping.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from aerobo import api, geometry, hydrofoil, hydrotail
from aerobo.hydrotail import HydrofoilTailProblem as HTP
from aerobo.hydrotail import evaluate_hydrofoil_tail as ev
from aerobo.hydrotail import fg_hydrofoil_tail as fg


def _mid(prob) -> np.ndarray:
    b = prob.bounds
    return 0.5 * (b[:, 0] + b[:, 1])


# ----------------------------------------------------------- 1. the vector

def test_the_blocks_switch_on_independently():
    base = HTP()
    assert base.param_labels == ("taper", "twist_root_deg", "twist_tip_deg",
                                 "tc", "depth_m", "V_ms", "S_t_m2", "l_t_m")
    assert HTP(l_t_fixed=1.0).param_labels[-1] == "S_t_m2"
    assert HTP(free_height=True).param_labels[-1] == "z_t_m"
    assert HTP(winglet=True).param_labels[-2:] == ("winglet_h_frac",
                                                   "winglet_cant_deg")
    both = HTP(winglet=True, free_height=True, chord_order=3)
    assert both.param_labels == (
        "taper", "twist_root_deg", "twist_tip_deg", "tc", "depth_m", "V_ms",
        "S_t_m2", "l_t_m", "z_t_m", "winglet_h_frac", "winglet_cant_deg",
        "chord_k1", "chord_k2", "chord_k3")
    assert both.dim == both.bounds.shape[0] == len(both.param_labels)


def test_the_stabiliser_sits_below_the_foil_and_the_cant_is_signed():
    p = HTP(winglet=True, free_height=True)
    lo, hi = p.z_t_bounds
    # strictly below, always — and the shallow end is the COPLANAR craft
    # (Z_T_FRAC_DEFAULT = tail.DZ_GRID_FRAC), not the air families' 0.05 b
    # measurement height: one fuselage, two wings bolted to it
    assert lo < hi <= -hydrotail.Z_T_FRAC_DEFAULT * p.b
    i = list(p.param_labels).index("winglet_cant_deg")
    assert tuple(p.bounds[i]) == hydrotail.WINGLET_CANT_BOUNDS_DEG
    assert p.bounds[i][0] < 0.0 < p.bounds[i][1]    # up AND down are offered


# ------------------------------------------------------------- 2. the trim

def test_the_craft_is_trimmed_in_lift_and_pitch():
    p = HTP()
    out = ev(_mid(p), p)
    assert out["feasible"], out["reason"]
    assert out["CL"] == pytest.approx(out["CL_target"], rel=1e-10)
    assert out["Cm_cg"] == pytest.approx(0.0, abs=1e-10)
    # ...and the static margin is (x_np - x_cg)/mac off the same solve
    assert out["SM"] == pytest.approx(
        (out["x_np"] - out["x_cg"]) / out["mac"], rel=1e-12)
    assert out["g_sm"] == pytest.approx(out["SM"] - p.SM_min, rel=1e-12)


def test_the_stabiliser_carries_lift_and_moving_it_changes_the_trim():
    p = HTP()
    x = _mid(p)
    out = ev(x, p)
    assert abs(out["CL_stab"]) > 1e-3
    assert out["CDp_stab"] > 0.0
    # a bigger stabiliser at the same arm moves the neutral point aft
    i_S = list(p.param_labels).index("S_t_m2")
    big = x.copy(); big[i_S] = p.bounds[i_S][1]
    out_big = ev(big, p)
    assert out_big["feasible"]
    assert out_big["x_np"] > out["x_np"]
    # ...and so does a longer arm
    i_l = list(p.param_labels).index("l_t_m")
    far = x.copy(); far[i_l] = p.bounds[i_l][1]
    assert ev(far, p)["x_np"] > out["x_np"]


def test_an_untrimmable_design_is_the_penalty_contract():
    """|i_t| beyond the limit is a failed design, not a scored one."""
    p = HTP(i_t_max_deg=0.001)          # nothing can trim inside this
    out = ev(_mid(p), p)
    assert not out["feasible"]
    assert "untrimmable" in out["reason"]
    f, g = fg(_mid(p), p)
    assert f == -100.0 and np.all(g == hydrotail.G_FAIL)


# -------------------------------------------------------- 3. the free surface

def test_the_stabiliser_is_deeper_and_the_margin_knows_it():
    p = HTP(free_height=True)
    x = _mid(p)
    out = ev(x, p)
    assert out["feasible"], out["reason"]
    assert out["z_t"] < 0.0
    assert out["stab_depth_m"] == pytest.approx(out["depth"] - out["z_t"])
    assert out["stab_depth_m"] > out["depth"]     # more static head than foil
    # deeper overall => more head => a larger cavitation margin
    i_d = list(p.param_labels).index("depth_m")
    deep = x.copy(); deep[i_d] = p.bounds[i_d][1]
    shallow = x.copy(); shallow[i_d] = p.bounds[i_d][0]
    assert ev(deep, p)["g_cav"] > ev(shallow, p)["g_cav"]


def test_speed_pulls_the_cavitation_margin_both_ways():
    """The two-way coupling hydrofoil.py's docstring names, now with a
    stabiliser in it: speed SHRINKS the cavitation number (sigma ~ 1/V^2 at
    fixed head) but also LIGHTENS the loading the craft's weight demands
    (CL = W/(q S)), which raises Cp_min. Neither term is the whole story, so
    the margin is not monotone in speed — at this design the loading term
    wins and the fast end is the safer one (MEASURED: g = -0.442 at 8 m/s
    against -0.046 at 16 m/s)."""
    p = HTP()
    x = _mid(p)
    i_V = list(p.param_labels).index("V_ms")
    slow = x.copy(); slow[i_V] = p.bounds[i_V][0]
    fast = x.copy(); fast[i_V] = p.bounds[i_V][1]
    out_slow, out_fast = ev(slow, p), ev(fast, p)
    assert out_slow["feasible"] and out_fast["feasible"]
    assert out_fast["sigma_cav_root"] < out_slow["sigma_cav_root"]
    assert out_fast["CL_target"] < out_slow["CL_target"]
    assert out_fast["g_cav"] > out_slow["g_cav"]
    # ...and at FIXED loading the pure sigma effect is the expected one
    assert hydrofoil.sigma_cav(0.5, 16.0) < hydrofoil.sigma_cav(0.5, 8.0)


def test_the_image_plane_is_actually_there():
    """Same design at two depths must differ in INDUCED drag, not just in
    the cavitation number: the free-surface image is a same-sign mirror, so
    the foil flies in its own downwash and that weakens with depth."""
    p = HTP()
    x = _mid(p)
    i_d = list(p.param_labels).index("depth_m")
    near = x.copy(); near[i_d] = p.bounds[i_d][0]
    far = x.copy(); far[i_d] = p.bounds[i_d][1]
    assert ev(near, p)["CDi"] > ev(far, p)["CDi"]


# ------------------------------------------------------- 4. the design box

def test_the_static_margin_boundary_is_inside_the_box():
    """The calibration claim: X_CG_FRAC leaves both stable and unstable
    designs in the box, so the constraint is asked rather than assumed."""
    p = HTP()
    grids = [np.linspace(lo, hi, 3) for lo, hi in p.bounds]
    margins = [ev(np.array(pt), p) for pt in itertools.product(*grids)]
    sm = np.array([o["g_sm"] for o in margins if o["feasible"]])
    assert sm.size > 100
    assert sm.min() < 0.0 < sm.max()


def test_a_forward_cg_would_switch_the_constraint_off(monkeypatch):
    """The other half of the calibration: at 2 % of the arm every solvable
    design is stable, i.e. the stability question stops being asked."""
    monkeypatch.setattr(hydrotail, "X_CG_FRAC", 0.02)
    p = HTP()
    grids = [np.linspace(lo, hi, 3) for lo, hi in p.bounds]
    sm = [ev(np.array(pt), p) for pt in itertools.product(*grids)]
    sm = np.array([o["g_sm"] for o in sm if o["feasible"]])
    assert sm.min() > 0.0


# --------------------------------------------------------- 5. compositions

def test_the_tip_device_and_the_chord_law_compose_with_the_elevator():
    plain = HTP()
    full = HTP(winglet=True, free_height=True, chord_order=3)
    x = _mid(plain)
    # the same design with the extra blocks NEUTRAL (no device, flat law,
    # stabiliser at its default depth) reproduces the plain problem
    z0 = -hydrotail.Z_T_FRAC_DEFAULT * plain.b
    xf = np.concatenate([x, [z0, 0.0, 90.0], np.zeros(3)])
    out_plain, out_full = ev(x, plain), ev(xf, full)
    assert out_plain["feasible"] and out_full["feasible"]
    assert out_full["LoD"] == pytest.approx(out_plain["LoD"], rel=1e-12)
    # ...and a real device changes it
    xf[list(full.param_labels).index("winglet_h_frac")] = 0.12
    assert ev(xf, full)["LoD"] != pytest.approx(out_plain["LoD"], rel=1e-9)


def test_the_elevator_is_a_surface_with_the_foils_own_freedoms():
    """Planform, tip device and chord law — the three the main foil has.

    Each is a block of its own in the vector, in the documented order, and
    switching them on with NEUTRAL values reproduces the published rectangle
    bit-for-bit (a designed elevator at AR 4, unity taper, no washout and no
    device IS the published surface).
    """
    designed = HTP(tail_free=True)
    assert designed.param_labels[-3:] == ("taper_t", "AR_t", "washout_t_deg")
    tipped = HTP(tail_free=True, tail_winglet=True)
    assert tipped.param_labels[-2:] == ("winglet_h_frac_t",
                                        "winglet_cant_t_deg")
    lawed = HTP(tail_free=True, chord_order=3)
    assert lawed.param_labels[-6:] == ("chord_k1", "chord_k2", "chord_k3",
                                       "chord_k1_t", "chord_k2_t",
                                       "chord_k3_t")
    assert lawed.n_chord_rows == 6 and HTP(chord_order=3).n_chord_rows == 3

    plain = HTP()
    x = _mid(plain)
    i_ar = list(designed.param_labels).index("taper_t")
    xd = np.insert(x, i_ar, [1.0, 4.0, 0.0])          # AR 4 rectangle
    a, b = ev(x, plain), ev(xd, designed)
    assert a["feasible"] and b["feasible"]
    assert b["LoD"] == pytest.approx(a["LoD"], rel=2e-3)
    # ...and a real planform moves it
    xd[i_ar + 1] = 7.0
    assert ev(xd, designed)["LoD"] != pytest.approx(a["LoD"], rel=1e-6)


def test_a_tip_device_on_the_elevator_needs_a_designed_elevator():
    """A rectangle drawn from its area alone has no tip to carry one — so
    the combination is refused rather than silently dropping two variables."""
    with pytest.raises(ValueError, match="designed surface"):
        HTP(tail_winglet=True)


def test_the_elevator_can_fly_its_own_section():
    """``polar_tail`` replaces the stabiliser's table only — the foil keeps
    its own, both alpha-validity ranges are honoured, and the cavitation
    margin is read off EACH surface's own Cp_min."""
    from aerobo.polar import default_polar_family

    fam = default_polar_family()
    shared = HTP(tail_free=True)
    own = HTP(tail_free=True, polar_tail=fam.at(0.18))
    assert shared.param_labels == own.param_labels     # no new variable
    x = _mid(shared)
    a, b = ev(x, shared), ev(x, own)
    assert a["feasible"] and b["feasible"]
    assert a["polar"] == b["polar"]                    # the foil is untouched
    assert a["polar_tail"] == a["polar"]               # default: it shares
    assert b["polar_tail"] != b["polar"]
    assert b["LoD"] != pytest.approx(a["LoD"], rel=1e-9)
    # the margin says which surface it came off, and the two are separate
    assert b["cav_surface_worst"] in ("foil", "stabiliser")
    assert b["g_cav"] == pytest.approx(min(b["g_cav_foil"], b["g_cav_stab"]))
    assert "cav_surface_worst" not in a       # one section, no question asked


def test_the_registry_offers_the_elevator_designs_and_its_section():
    """Every elevator design is a registered problem, and every plain
    elevator family declares the second surface's section flag."""
    designs = {v["design"] for v in api.HYDRO_TAIL_VARIANTS.values()}
    assert designs == {"fixed", "planform", "planform+tip"}
    for name, v in api.HYDRO_TAIL_VARIANTS.items():
        assert name in api.PROBLEM_SPECS
        assert api.SECTION_AFT_KEY in api.PROBLEM_SPECS[name].flags
        assert api.hydro_tail_problem(winglet=v["winglets"], arm=v["arm"],
                                      height=v["height"],
                                      design=v["design"]) == name
    with pytest.raises(KeyError):
        api.hydro_tail_problem(design="nonsense")


def test_a_chosen_elevator_section_travels_as_a_flag():
    """The flag path, end to end: a library name reaches the stabiliser as a
    polar WITH a Cp_min table, and changes no design variable."""
    name = "hydrofoil + elevator [designed elevator + tip device]"
    spec = api.PROBLEM_SPECS[name]
    plain = spec.build(None, {}, None)
    with_sec = spec.build(None, {api.SECTION_AFT_KEY: "naca0012"}, None)
    assert with_sec.param_labels == plain.param_labels
    x = _mid_arr(plain.bounds)
    a, b = plain.evaluate(x), with_sec.evaluate(x)
    assert a["feasible"] and b["feasible"]
    assert b["polar_tail"].startswith("naca0012")
    assert b["polar"] == a["polar"]


def test_a_collapsed_chord_law_is_refused_here_too():
    p = HTP(chord_order=3)
    x = _mid(p)
    x[-3:] = -0.5
    out = ev(x, p)
    assert not out["feasible"] and out["reason"].startswith("planform:")


# ------------------------------------------------------------ 6. registry

@pytest.mark.parametrize("name", sorted(api.HYDRO_TAIL_VARIANTS))
def test_every_variant_builds_evaluates_and_reports_two_margins(name):
    spec = api.PROBLEM_SPECS[name]
    assert spec.medium == "water" and spec.is_constrained
    assert spec.n_constraints == 2
    assert spec.constraint_labels == ("cavitation margin",
                                      "static margin - SM_min")
    built = spec.build({}, {}, None)
    assert built.dim == len(built.param_labels) == len(spec.param_labels)
    f, g = built.callable(_mid_arr(built.bounds))
    assert np.isfinite(f) and np.asarray(g).shape == (2,)


def _mid_arr(bounds) -> np.ndarray:
    b = np.asarray(bounds, dtype=float)
    return 0.5 * (b[:, 0] + b[:, 1])


def test_the_family_refuses_a_mission_card():
    """Speed and depth are design variables here, so a mission spec would be
    two answers to one question — it is refused, not ignored."""
    spec = api.PROBLEM_SPECS["hydrofoil + elevator"]
    assert spec.mission_fields == ()
    with pytest.raises(ValueError, match="design variables"):
        spec.build({"W_N": 6000.0}, {}, None)


def test_the_chord_law_composes_across_the_whole_family():
    for name, v in api.HYDRO_TAIL_VARIANTS.items():
        twin = api.CHORD_TWINS[name]
        assert twin in api.PROBLEM_SPECS
        base_dim = len(api.PROBLEM_SPECS[name].param_labels)
        # ONE LAW PER DESIGNED SURFACE: the foil always, and the elevator too
        # wherever the elevator is a designed surface rather than the
        # published rectangle (wingtail.py's convention, in water)
        n_surfaces = 1 if v["design"] == "fixed" else 2
        labels = api.PROBLEM_SPECS[twin].param_labels
        assert len(labels) == base_dim + 3 * n_surfaces
        assert labels[-3:] == (("chord_k1_t", "chord_k2_t", "chord_k3_t")
                               if n_surfaces == 2 else
                               ("chord_k1", "chord_k2", "chord_k3"))
        # ...and the flight modifier is correctly NOT offered
        assert api.add_modifier(name, "flight") is None


def test_the_gui_maps_water_plus_an_elevator_onto_the_family():
    import gui.nice_app as v1

    for name, v in api.HYDRO_TAIL_VARIANTS.items():
        ch = v1.choices_from_problem(name)
        assert ch["medium"] == "water" and ch["tail"] is True
        assert v1.derive_problem(ch)[0] == name, name
        assert v1.derive_problem(dict(ch, chord="free"))[0] == \
            api.CHORD_TWINS[name]
        assert (ch["winglets"] != "none") == bool(v["winglets"])


def test_the_air_only_tail_controls_say_so_under_water():
    import gui.nice_app as v1

    ch = dict(v1.BUILDER_DEFAULTS, medium="water", tail=True,
              tail_type="t_tail", tail_fin_drag=True)
    name, notes = v1.derive_problem(ch)
    assert name in api.HYDRO_TAIL_VARIANTS
    assert any("layout" in n.lower() for n in notes)
    assert any("no fin to charge" in n for n in notes)


def test_the_solver_reuses_the_published_cavitation_and_image_code():
    """No second copy of the water physics: the margin function and the image
    plane are hydrofoil.py's own."""
    assert hydrotail.cavitation_margin_panels is \
        hydrofoil.cavitation_margin_panels
    assert hydrotail.RHO_WATER == hydrofoil.RHO_WATER
    assert geometry.chord_labels(3) == ("chord_k1", "chord_k2", "chord_k3")
