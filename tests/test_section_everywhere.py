"""A DESIGNED aerofoil, in every family that can fly one.

Before this the live-XFOIL CST section existed only beside a wing and its tip
device. It now composes with the tail (wingtail_section.py), with the water
families INCLUDING the elevator (hydrofoil_section.py) and with the tandem
pair (tandemvlm.py through the same wrapper). Three things had to be true,
and are gated here:

1. THE T/C VARIABLE GOES. The CST weights ARE the thickness, so a family that
   selected a section by t/c drops that row when the section is designed —
   offering both would let the polar family and the live section disagree
   about one number.
2. THE MARGINS COMPOSE. The family's own margins come first, then the
   section's two (t/c and |Cm|), and the size modifier's stress margin lands
   between them rather than at the end.
3. THE WATER FAMILIES GET Cp_min. A cavitation constraint needs the minimum
   surface pressure, which XFOIL only reports through CPMN — so the sweep has
   a with_cpmin mode, and a section whose Cp_min cannot be paired is a design
   FAILURE rather than a foil flown without a cavitation check.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aerobo import api, hydrofoil, hydrotail, tandemvlm, wingtail, xfoil_run
from aerobo.airfoil import AirfoilProblem, cst_coords
from aerobo.hydrofoil_section import HydrofoilSectionProblem
from aerobo.hydrofoil_section import evaluate_hydrofoil_section as ev_water
from aerobo.wingtail_section import WingTailSectionProblem
from aerobo.wingtail_section import evaluate_wing_tail_section as ev_air

needs_xfoil = pytest.mark.skipif(
    not Path(xfoil_run.DEFAULT_XFOIL_BIN).exists(),
    reason="local XFOIL binary not installed")


# ------------------------------------------------------ 1. the vector

def test_the_designed_section_takes_the_thickness_row_with_it():
    # air: the wing+tail family's t/c variant is refused outright
    with pytest.raises(ValueError, match="tc_free"):
        WingTailSectionProblem(
            wing_tail=wingtail.WingTailProblem(tc_free=True))
    # water: the t/c row leaves the vector when a section is supplied
    plain = hydrofoil.HydrofoilWingletProblem()
    assert "tc" in plain.param_labels
    designed = HydrofoilSectionProblem(foil=plain)
    assert "tc" not in designed.param_labels
    assert designed.dim == plain.dim - 1 + 2 * designed.section.n_cst
    # ...and the section block is the trailing one, so the family's own
    # blocks keep their positions
    assert designed.param_labels[-8:] == tuple(
        [f"w_upper_{i}" for i in range(4)]
        + [f"w_lower_{i}" for i in range(4)])


def test_the_blocks_split_cheap_planform_from_expensive_section():
    for prob in (WingTailSectionProblem(),
                 HydrofoilSectionProblem(),
                 WingTailSectionProblem(
                     wing_tail=tandemvlm.TandemVLMProblem(winglets=True))):
        wing_block, section_block = prob.blocks
        assert not wing_block["owns_constraints"]
        assert section_block["owns_constraints"]
        assert (len(wing_block["indices"]) + len(section_block["indices"])
                == prob.dim)
        assert len(section_block["indices"]) == 2 * prob.section.n_cst


# ---------------------------------------------------- 2. margins compose

@needs_xfoil
def test_the_family_margins_come_first_then_the_section_s_two():
    p = WingTailSectionProblem(
        wing_tail=wingtail.WingTailProblem(winglet=True, capped=True))
    out = ev_air(p.x0, p)
    assert out["feasible"], out["reason"]
    g = np.asarray(out["g"], dtype=float)
    assert g.size == p.n_constraints == 3
    assert g[0] == pytest.approx(out["SM"] - p.wing_tail.SM_min)
    assert g[1] == pytest.approx(out["g_tc"])
    assert g[2] == pytest.approx(out["g_cm"])


@needs_xfoil
def test_the_size_margin_lands_between_the_family_and_the_section():
    name = "tail + winglet (span-capped) + CST section (XFOIL) + free planform"
    spec = api.PROBLEM_SPECS[name]
    assert spec.constraint_labels == ("static margin - SM_min",
                                      "root-bending stress margin",
                                      "t/c margin", "|Cm| margin")
    built = spec.build({}, {}, None)
    f, g = built.callable(0.5 * (built.bounds[:, 0] + built.bounds[:, 1]))
    assert np.asarray(g).size == spec.n_constraints == 4


@needs_xfoil
def test_the_tandem_pair_designs_a_section_too():
    p = WingTailSectionProblem(
        wing_tail=tandemvlm.TandemVLMProblem(winglets=True))
    out = ev_air(p.x0, p)
    assert out["feasible"], out["reason"]
    # the pair owns no margin of its own, so the vector IS the section's two
    assert np.asarray(out["g"]).size == p.n_constraints == 2
    assert out["CL_front"] != out["CL_rear"]      # a real two-surface solve


# ------------------------------------------------------- 3. Cp_min in water

@needs_xfoil
def test_the_sweep_can_report_the_minimum_surface_pressure():
    sec = AirfoilProblem()
    coords = cst_coords(sec.w0[:4], sec.w0[4:], dz_te=sec.dz_te)
    plain = xfoil_run.run_xfoil_polar(coords, sec.re, sec.mach, sec.alphas,
                                      timeout_s=60)
    withcp = xfoil_run.run_xfoil_polar(coords, sec.re, sec.mach, sec.alphas,
                                       timeout_s=60, with_cpmin=True)
    assert not plain.has_cp_min and withcp.has_cp_min
    assert withcp.cp_min.size == withcp.alpha_deg.size
    assert np.all(withcp.cp_min < 0.0)             # a suction peak, always
    # more incidence, deeper suction peak (the cavitation mechanism)
    order = np.argsort(withcp.alpha_deg)
    cp = withcp.cp_min[order]
    assert cp[-1] < cp[len(cp) // 2]
    # the two sweeps are separate cache entries: same geometry, different ask
    assert xfoil_run.cache_key(coords, sec.re, sec.mach, sec.alphas) != \
        xfoil_run.cache_key(coords, sec.re, sec.mach, sec.alphas,
                            with_cpmin=True)


@needs_xfoil
@pytest.mark.parametrize("foil, n_own", [
    (hydrofoil.HydrofoilProblem(), 1),
    (hydrofoil.HydrofoilWingletProblem(), 1),
    (hydrotail.HydrofoilTailProblem(winglet=True, free_height=True), 2),
], ids=["hydrofoil", "hydrofoil + winglet", "hydrofoil + elevator"])
def test_every_water_family_flies_a_designed_section(foil, n_own):
    p = HydrofoilSectionProblem(foil=foil)
    out = ev_water(p.x0, p)
    assert out["feasible"], out["reason"]
    g = np.asarray(out["g"], dtype=float)
    assert g.size == p.n_constraints == n_own + 2
    # the CAVITATION margin is the family's first, and it is read off the
    # designed section's own Cp_min (not a NACA family member's)
    # the family's first margin IS the cavitation one, and it is built from
    # the designed section's own Cp_min (not a NACA family member's)
    assert g[0] == pytest.approx(out["sigma_cav"] + out["cp_min_worst"])
    assert out["cp_min_worst"] < 0.0
    assert out["polar"] == "cst-live"
    # t/c is REPORTED off the designed shape rather than searched as a row,
    # so it has to equal the thickness of the COORDINATES XFOIL was handed.
    # The old line compared out["tc"] with itself, which is true of any
    # number: reporting the placeholder's constant 0.12 (the anchor's
    # thickness, _PLACEHOLDER_POLAR.tc) for every candidate stayed green.
    n = p.section.n_cst
    w_u, w_l = p.x0[p.n_wing:p.n_wing + n], p.x0[p.n_wing + n:]
    loop = cst_coords(w_u, w_l, dz_te=p.section.dz_te)
    i_le = int(np.argmin(loop[:, 0]))
    upper, lower = loop[:i_le + 1][::-1], loop[i_le:]
    psi = np.linspace(0.0, 1.0, 20001)
    tc_flown = float(np.max(np.interp(psi, upper[:, 0], upper[:, 1])
                            - np.interp(psi, lower[:, 0], lower[:, 1])))
    # rel 3e-4 sits ~4x above the flown 160-point loop's grid-max error and
    # ~3x below the gap between the seed's t/c (0.1197) and the 0.12 anchor
    # constant it has to reject
    assert out["tc"] == pytest.approx(tc_flown, rel=3e-4), (out["tc"], tc_flown)
    # ...and the margin actually ENFORCED is built from that same thickness
    assert out["g_tc"] == pytest.approx(
        (out["tc"] - p.section.tc_min) / p.section.tc_min, rel=1e-9)


def test_a_section_without_cp_min_is_a_failure_not_a_flown_foil():
    """The load-bearing honesty gate: no cavitation number, no result."""
    from aerobo.polar import TablePolar
    pol = TablePolar(alpha_deg=np.array([-2.0, 0.0, 2.0]),
                     CL=np.array([-0.1, 0.1, 0.3]),
                     CD=np.array([0.01, 0.009, 0.011]),
                     CM=np.array([-0.05, -0.05, -0.05]), name="no-cpmin")
    assert not pol.has_cp_min
    with pytest.raises(ValueError, match="Cp_min"):
        pol.cp_min(np.array([0.0]))


# ------------------------------------------------------------ 4. registry

def test_the_registry_offers_a_designed_section_in_every_family():
    # 4 tip devices x arm x height x how much of the TAIL is designed x the
    # four states of the WING's own cant. (This count read 48 until the cant
    # axis reached the section family and nobody moved it, so it was stale
    # by a factor of two before the axis grew to four states.)
    assert len(api.WING_TAIL_SECTION_VARIANTS) == 4 * 2 * 2 * 3 \
        * len(api.WING_CANTS)
    # the planar foil, the foil + tip device, and EVERY elevator variant —
    # which now includes how much of the elevator itself is designed, so the
    # count is read off that table rather than restated here
    assert len(api.HYDRO_SECTION_VARIANTS) == 2 + len(api.HYDRO_TAIL_VARIANTS)
    assert len(api.HYDRO_TAIL_VARIANTS) == 2 * 2 * 2 * 3
    for name in (set(api.WING_TAIL_SECTION_VARIANTS)
                 | set(api.HYDRO_SECTION_VARIANTS)):
        spec = api.PROBLEM_SPECS[name]
        assert spec.slow, name                 # live XFOIL: minutes per run
        assert spec.has_blocks, name           # portfolio split declared
        assert spec.is_constrained, name
        assert spec.constraint_labels[-2:] == ("t/c margin", "|Cm| margin")
        assert "w_upper_0" in spec.param_labels and \
            "tc" not in spec.param_labels, name


def test_the_gui_reaches_a_designed_section_in_every_family():
    import gui.nice_app as v1

    cases = {
        ("air", "tail"): dict(v1.BUILDER_DEFAULTS, tail=True,
                              winglets="capped", airfoil="section_only"),
        ("water", "elevator"): dict(v1.BUILDER_DEFAULTS, medium="water",
                                    tail=True, winglets="free",
                                    airfoil="section_only"),
        ("tandem",): dict(v1.BUILDER_DEFAULTS, system="tandem",
                          winglets="free", airfoil="section_only"),
    }
    for key, ch in cases.items():
        name, notes = v1.derive_problem(ch)
        assert name in api.PROBLEM_SPECS, (key, name)
        assert "CST section (XFOIL)" in name, (key, name)
        assert not any("ignored" in n for n in notes), key
        # ...and every one of them round-trips
        assert v1.derive_problem(v1.choices_from_problem(name))[0] == name


# ------------------------------- 5. flying the section the USER chose

def test_a_water_family_can_fly_a_chosen_section_with_its_cp_min():
    """A cavitation constraint needs Cp_min, so a chosen section is swept
    WITH it; the thickness row leaves the vector because the shape has one."""
    plain = api.PROBLEM_SPECS["hydrofoil"]
    twin = api.PROBLEM_SPECS[api.chosen_section_problem("hydrofoil")]
    assert "tc" in plain.param_labels
    assert "tc" not in twin.param_labels
    assert api.SECTION_KEY in twin.flags
    assert api.needs_chosen_section(twin.name)

    built = twin.build({}, {api.SECTION_KEY: "hg40"}, None)
    pol = built.problem.section_polar
    assert "hg40" in pol.name
    assert pol.cp_min(2.0) < 0.0                  # a real Cp_min table
    out = built.evaluate(0.5 * (built.bounds[:, 0] + built.bounds[:, 1]))
    assert out["feasible"], out["reason"]
    # The cavitation margin is a NUMBER read off the section the USER picked:
    # g = sigma_cav + the worst strip's Cp_min, taken from hg40's own table at
    # the effective angles this design actually flies. The old line joined the
    # output dict's KEY NAMES, so "g"/"sigma_cav" being in the schema made it
    # true whatever the margin was built from (the family's NACA anchor, or a
    # sentinel).
    alpha_eff_deg = np.rad2deg(out["hydrofoil"].foil.alpha_eff_y)
    cp_worst = float(np.min(pol.cp_min(alpha_eff_deg)))
    assert cp_worst < 0.0                         # a suction peak, always
    assert out["cp_min_worst"] == pytest.approx(cp_worst, rel=1e-9)
    assert out["sigma_cav"] > 0.0
    assert float(out["g"]) == pytest.approx(out["sigma_cav"] + cp_worst,
                                            rel=1e-9)


def test_a_water_family_run_without_its_section_says_so_loudly():
    """The failure contract must NOT swallow this: a run of the wrong
    problem is a configuration mistake, not an infeasible design."""
    twin = api.chosen_section_problem("hydrofoil")
    with pytest.raises(api.MissingSectionError):
        api.run(api.RunConfig(problem_name=twin, optimiser="random",
                              budget=2, seed=0))
    # ...but the design BOX is still describable, which is what a shell
    # needs to draw the problem before a section has been picked
    assert "taper" in api.PROBLEM_SPECS[twin].default_bounds


def test_the_track_family_flies_a_chosen_section_without_changing_its_vector():
    """The car's thickness was never a design variable, so there is no twin:
    the chosen section is a plain flag."""
    assert api.chosen_section_problem("car rear wing") is None
    spec = api.PROBLEM_SPECS["car rear wing"]
    plain = spec.build({}, {}, None)
    chosen = spec.build({}, {api.SECTION_KEY: "hg40"}, None)
    assert plain.param_labels == chosen.param_labels
    assert "hg40" in chosen.problem.section_polar.name
    a = plain.evaluate(0.5 * (plain.bounds[:, 0] + plain.bounds[:, 1]))
    b = chosen.evaluate(0.5 * (chosen.bounds[:, 0] + chosen.bounds[:, 1]))
    assert a["feasible"] and b["feasible"]
    assert a["score"] != pytest.approx(b["score"])     # a different aerofoil
