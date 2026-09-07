"""The free chord law composes with EVERY solver family.

The chord law changes only ``Wing.chord(y)``, which every solver in the
package samples the same way, so it is a MODIFIER rather than a solver
family: each problem gains a twin carrying three extra coefficients
(api.CHORD_TWINS) instead of the registry needing one hand-written entry per
combination.

Three things have to hold for that to be honest, and are checked here:

1. ORDER 0 IS THE PUBLISHED PROBLEM. Not "close to" — the same bounds, the
   same dimension, the same number.
2. ALL-ZERO COEFFICIENTS ARE A STRAIGHT TAPER. A twin evaluated with its
   chord coefficients at zero must return exactly its base problem's score,
   in every family. That is what makes the twin an extension of the base
   rather than a different problem wearing its name.
3. A COLLAPSED LAW IS THE PENALTY CONTRACT, everywhere — never a
   plausible-looking number from a planform that cannot be flown.
"""

from __future__ import annotations

import numpy as np
import pytest

from aerobo import (aircraft, api, carwing, endplate, geometry, hydrofoil,
                    objective, tail, tandem, weights, wing_airfoil)

ORDER = 3


def _mid(prob) -> np.ndarray:
    b = prob.bounds
    return 0.5 * (b[:, 0] + b[:, 1])


#: (name, problem factory taking chord_order, evaluate, laws-per-problem).
#: The tandem pair carries TWO independent chord laws, one per wing.
FAMILIES = [
    ("trim wing",
     lambda **kw: objective.Problem(mode="trim", **kw),
     objective.evaluate, 1),
    ("winglet",
     lambda **kw: objective.Problem(mode="winglet", **kw),
     objective.evaluate, 1),
    ("winglet_capped_blended",
     lambda **kw: objective.Problem(mode="winglet_capped_blended", **kw),
     objective.evaluate, 1),
    ("hydrofoil", hydrofoil.HydrofoilProblem, hydrofoil.evaluate_hydrofoil,
     1),
    ("hydrofoil + winglet", hydrofoil.HydrofoilWingletProblem,
     hydrofoil.evaluate_hydrofoil_winglet, 1),
    ("car rear wing", carwing.CarWingProblem, carwing.evaluate_car_wing, 1),
    ("car rear wing + endplates", endplate.CarWingEndplateProblem,
     endplate.evaluate_car_wing_endplate, 1),
    ("tail", tail.TailProblem, tail.evaluate_tail, 1),
    ("tail (fixed arm)",
     lambda **kw: tail.TailProblem(l_t_fixed=5.5, **kw),
     tail.evaluate_tail, 1),
    ("free planform", aircraft.AircraftProblem, aircraft.evaluate_aircraft,
     1),
    ("wing+airfoil", wing_airfoil.WingAirfoilProblem,
     wing_airfoil.evaluate_wing_airfoil, 1),
    ("tandem", tandem.TandemProblem, tandem.evaluate_tandem, 2),
]

IDS = [f[0] for f in FAMILIES]


@pytest.mark.parametrize("name, factory, evaluate, laws", FAMILIES,
                         ids=IDS)
def test_chord_order_zero_is_the_published_problem(name, factory, evaluate,
                                                  laws):
    base, twin0 = factory(), factory(chord_order=0)
    assert twin0.dim == base.dim
    assert np.array_equal(twin0.bounds, base.bounds)
    x = _mid(base)
    assert evaluate(x, twin0)["score"] == evaluate(x, base)["score"]


@pytest.mark.parametrize("name, factory, evaluate, laws", FAMILIES,
                         ids=IDS)
def test_the_chord_rows_are_appended_last_and_leave_the_others_alone(
        name, factory, evaluate, laws):
    base, twin = factory(), factory(chord_order=ORDER)
    assert twin.dim == base.dim + laws * ORDER
    assert np.array_equal(twin.bounds[:base.dim], base.bounds)
    assert np.array_equal(twin.bounds[base.dim:],
                          np.vstack([geometry.chord_bounds(ORDER)] * laws))


@pytest.mark.parametrize("name, factory, evaluate, laws", FAMILIES,
                         ids=IDS)
def test_zero_coefficients_reproduce_the_base_problem_exactly(
        name, factory, evaluate, laws):
    """The load-bearing guarantee: a twin with a flat chord law IS its base."""
    base, twin = factory(), factory(chord_order=ORDER)
    x = _mid(base)
    out_base = evaluate(x, base)
    out_twin = evaluate(np.concatenate([x, np.zeros(laws * ORDER)]), twin)
    assert out_base["feasible"] and out_twin["feasible"]
    assert out_twin["score"] == out_base["score"]


@pytest.mark.parametrize("name, factory, evaluate, laws", FAMILIES,
                         ids=IDS)
def test_a_collapsed_chord_law_is_refused_everywhere(name, factory,
                                                    evaluate, laws):
    twin = factory(chord_order=ORDER)
    x = _mid(twin)
    x[-ORDER:] = -0.5                       # 1 - 0.5 - 0.5 - 0.5 < 0 at eta=1
    out = evaluate(x, twin)
    assert not out["feasible"]
    assert out["reason"].startswith("planform:")
    assert out["score"] == objective.PENALTY


@pytest.mark.parametrize("name, factory, evaluate, laws", FAMILIES,
                         ids=IDS)
def test_a_real_chord_law_actually_reshapes_the_planform(name, factory,
                                                         evaluate, laws):
    """Not just accepted — used. A nonzero law must move the flown chord and
    the score, or the coefficients are decorative."""
    twin = factory(chord_order=ORDER)
    x = _mid(twin)
    n = laws * ORDER
    flat = evaluate(np.concatenate([x[:-n], np.zeros(n)]), twin)
    shaped = evaluate(np.concatenate([x[:-n], [0.3, -0.2, 0.1] * laws]), twin)
    assert shaped["feasible"]
    assert shaped["score"] != flat["score"]
    wing = shaped.get("wing")
    if wing is not None:                    # families that return the Wing
        assert wing.chord_dev > 0.05
        # area is held EXACTLY by the closed-form rescale
        y = np.linspace(-wing.b / 2, wing.b / 2, 20001)
        assert np.trapezoid(wing.chord(y), y) == pytest.approx(wing.S,
                                                               rel=1e-8)


# ------------------------------------------------------------------ registry

def test_every_twin_is_derived_from_its_base_spec():
    """Twins are GENERATED, so nothing about them can drift from the problem
    they extend except the builder and the chord coefficients themselves.

    The coefficients are NOT always the trailing entries: they belong to the
    wing BLOCK, which is the last block for most families but not for the
    13-D CST problems (the section weights follow) — so the check is that
    removing them leaves exactly the base problem's vector."""
    for base_name, twin_name in api.CHORD_TWINS.items():
        base = api.PROBLEM_SPECS[base_name]
        twin = api.PROBLEM_SPECS[twin_name]
        assert twin.medium == base.medium
        assert twin.is_constrained == base.is_constrained
        assert twin.n_constraints == base.n_constraints
        assert twin.constraint_labels == base.constraint_labels
        assert twin.uses_mission == base.uses_mission
        assert twin.mission_fields == base.mission_fields
        assert twin.slow == base.slow

        extra = [str(l) for l in twin.param_labels
                 if l not in base.param_labels]
        assert all(l.startswith("chord_") for l in extra), twin_name
        assert len(extra) % ORDER == 0 and extra, twin_name
        kept = tuple(l for l in twin.param_labels if l not in extra)
        assert kept == tuple(base.param_labels), twin_name

        built = twin.build({}, {}, None)
        assert built.dim == len(twin.param_labels)
        assert built.dim == len(base.build({}, {}, None).bounds) + len(extra)


def test_chord_max_frac_flag_widens_every_twin_s_coefficient_box():
    for twin_name, twin in api.PROBLEM_SPECS.items():
        if twin_name not in set(api.CHORD_TWINS.values()):
            continue
        built = twin.build({}, {"chord_max_frac": 1.0}, None)
        rows = [list(b) for lbl, b in zip(built.param_labels, built.bounds)
                if str(lbl).startswith("chord_")]
        assert rows and all(r == [-1.0, 1.0] for r in rows), twin_name


def test_the_gui_treats_the_chord_law_as_a_modifier_not_a_family():
    """Picking a chord law must not reset any other choice, and every
    family that has a twin must resolve to it."""
    import gui.nice_app as v1

    assert "chord" not in v1.SPECIAL_KEYS
    for base_name, twin_name in api.CHORD_TWINS.items():
        ch = dict(v1.choices_from_problem(base_name), chord="free")
        got, notes = v1.derive_problem(ch)
        assert got == twin_name, base_name
        base_family = api.base_of(base_name)
        if base_family in api.WING_TAIL_VARIANTS:
            # the nonplanar wing+tail family always names the solver it is
            # (it is NOT the published lifting-line tail); every other
            # family resolves with nothing to report
            assert len(notes) == 1 and "nonplanar" in notes[0], base_name
        elif base_family in api.WING_TAIL_SECTION_VARIANTS:
            # the wing+tail+CST family always names the solver it is (live
            # XFOIL in the loop, and the t/c control switched off)
            assert len(notes) == 1 and "CST section" in notes[0], base_name
        elif base_family in api.TANDEM_VLM_VARIANTS:
            # the nonplanar pair always names the solver it is (a different
            # aero core AND a different design space from the planar one)
            assert any("nonplanar solve" in n for n in notes), base_name
        elif base_family in api.HYDRO_SECTION_VARIANTS:
            # the designed-section water family always says what it is (and
            # the elevator variants name their solver as well)
            assert any("DESIGNED" in n for n in notes), base_name
        elif base_family in api.HYDRO_TAIL_VARIANTS:
            # ...and so does the imaged hydrofoil + elevator family, plus one
            # more line wherever the ELEVATOR itself is a designed surface
            # (its planform, its own tip device and its own chord law)
            assert any("imaged solve" in n for n in notes), base_name
            designed = api.HYDRO_TAIL_VARIANTS[base_family]["design"] != "fixed"
            assert len(notes) == (2 if designed else 1), base_name
            if designed:
                assert any("elevator is designed too" in n
                           for n in notes), base_name
        elif base_family == "free planform (aircraft)":
            # the published sizing problem says why it is not the modifier
            assert all("its own family" in n for n in notes), base_name
        elif base_family in api.CHOSEN_SECTION_TWINS.values():
            # a chosen-section twin says what it flies and what that costs
            # (the elevator ones also name their solver, as they do without it)
            assert any("chosen upstream" in n for n in notes), base_name
        elif base_family in ("tail [designed tail]",
                             "tail (fixed arm) [designed tail]"):
            # the lifting-line designed-tail twin states the freedom it adds
            # (and the washout convention that freedom follows)
            assert len(notes) == 1 and "designed too" in notes[0], base_name
        else:
            assert notes == [], base_name
    # the ONE family with no twin says so instead of silently dropping it:
    # a 2-D section has no wing planform to reshape
    name, notes = v1.derive_problem(
        dict(v1.choices_from_problem("airfoil (section)"), chord="free"))
    assert name == "airfoil (section)"
    assert notes and "no wing planform to reshape" in notes[0]
    # ...and it is the only one
    missing = [n for n in api.PROBLEM_SPECS
               if n not in api.CHORD_TWINS
               and n not in set(api.CHORD_TWINS.values())]
    assert missing == ["airfoil (section)"], missing


# -------------------------------------------------- structure sees the change

def test_root_bending_stress_can_be_sized_on_the_flown_root_chord():
    """A chord law moves the spar box, so the structural constraint has to be
    told about it — otherwise a design could widen its root for free."""
    args = (11.0, 16.0, 0.45, 0.12, 9500.0)
    trapezoid = 2.0 * 16.0 / (11.0 * (1.0 + 0.45))
    assert weights.root_bending_stress(*args) == pytest.approx(
        weights.root_bending_stress(*args, c_root=trapezoid))
    # sigma ~ 1/c_root^3 (cap area ~ c^2, box depth ~ c)
    wide = weights.root_bending_stress(*args, c_root=2 * trapezoid)
    assert wide == pytest.approx(
        weights.root_bending_stress(*args) / 8.0, rel=1e-12)
    with pytest.raises(ValueError, match="c_root"):
        weights.root_bending_stress(*args, c_root=0.0)


def test_a_fatter_root_relaxes_the_aircraft_s_stress_constraint():
    prob = aircraft.AircraftProblem(chord_order=ORDER)
    x = _mid(prob)
    flat = aircraft.evaluate_aircraft(
        np.concatenate([x[:-ORDER], np.zeros(ORDER)]), prob)
    # a law that thins the tip and fattens the root, at the same area
    fat = aircraft.evaluate_aircraft(
        np.concatenate([x[:-ORDER], [-0.4, 0.0, 0.0]]), prob)
    assert fat["feasible"]
    assert fat["wing"].chord(np.array([0.0]))[0] > \
        flat["wing"].chord(np.array([0.0]))[0]
    assert fat["sigma_root_Pa"] < flat["sigma_root_Pa"]
    assert fat["g_sigma"] > flat["g_sigma"]


def test_shaping_the_section_and_the_chord_together_is_a_real_problem():
    """The user-facing ask: shape the aerofoil AND free the chord law. Both
    the with- and without-winglet CST problems carry it, and the chord
    coefficients sit in the WING block (before the section weights), because
    that is the block whose geometry they change."""
    from aerobo import winglet_section as ws

    plain = ws.WingletSectionProblem(winglet=False)
    tipped = ws.WingletSectionProblem(winglet=True)
    # removing the tip device REMOVES its two variables, it does not zero them
    assert tipped.n_wing == plain.n_wing + 2
    assert plain.dim == plain.n_wing + plain.section.dim

    withchord = ws.WingletSectionProblem(winglet=False, chord_order=ORDER)
    assert withchord.dim == plain.dim + ORDER
    assert np.array_equal(withchord.bounds[:3], plain.bounds[:3])
    assert np.array_equal(withchord.bounds[3:3 + ORDER],
                          geometry.chord_bounds(ORDER))
    assert np.array_equal(withchord.bounds[3 + ORDER:], plain.bounds[3:])

    for name, dim, has_winglet in (
            ("wing + airfoil (XFOIL)", 11, False),
            ("wing + airfoil (XFOIL) + free chord law", 14, False),
            ("winglet + airfoil (XFOIL)", 13, True),
            ("winglet + airfoil (XFOIL) + free chord law", 16, True)):
        spec = api.PROBLEM_SPECS[name]
        assert spec.slow and spec.is_constrained and spec.has_blocks
        labels = [str(x) for x in spec.param_labels]
        assert len(labels) == dim
        assert ("winglet_h_frac" in labels) is has_winglet
        chord_at = [i for i, x in enumerate(labels) if x.startswith("chord_")]
        cst_at = [i for i, x in enumerate(labels) if x.startswith("w_")]
        if chord_at:                      # chord rows precede the CST block
            assert max(chord_at) < min(cst_at), name
        built = spec.build({}, {}, None)
        assert built.dim == dim
        assert list(built.param_labels) == labels


def test_the_gui_offers_the_section_plus_wing_co_design():
    import gui.nice_app as v1

    ch = dict(v1.BUILDER_DEFAULTS, airfoil="section_wing", chord="free")
    assert v1.derive_problem(ch) == ("wing + airfoil (XFOIL) + free chord law",
                                     [])
    # with a tip device it is the winglet variant, not a different family
    ch = dict(ch, winglets="free")
    assert v1.derive_problem(ch) == (
        "winglet + airfoil (XFOIL) + free chord law", [])
    # the pure 2-D section is still its own problem and still refuses a
    # chord law it has no planform for
    ch = dict(v1.BUILDER_DEFAULTS, airfoil="section_only")
    assert v1.derive_problem(ch)[0] == "airfoil (section)"
