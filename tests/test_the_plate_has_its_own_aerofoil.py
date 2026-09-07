"""The car's endplate is a surface, so it has a SECTION and not just a family.

``endplate.SECTIONS`` names a CONSTRUCTION — flat, rounded, shaped — which
sets a drag law and a stiffness law and knows nothing about a shape. That is
enough to choose between a bent sheet and a moulded strut and not enough to
design anything: under it every aerofoil of a given thickness scores
identically, so "optimise the endplate aerofoil" had no meaning at all.

``endplate.py``'s own module docstring records the missing piece — "that needs
a section polar for the plate, which would mean running the polar family at
the plate's own Reynolds number, and is left as an extension point rather than
invented". These tests are that extension point, asserted as what a user can
see: naming a section changes the plate's drag AND its stiffness, the section
is flown at the PLATE's Reynolds number rather than the wing's, a cambered one
is refused, and naming none leaves every published run exactly where it was.
"""

import numpy as np
import pytest

from aerobo import api
from aerobo import endplate as ep

NAME = "car rear wing + endplates"

#: two symmetric library members. Symmetric because a vertical panel built at
#: theta = twist - alpha_L0 carries a side force at zero toe otherwise, which
#: is the trap endplate.py's "section trap" section is about.
FOIL_A, FOIL_B = "e168", "e169"


def _built(flags=None):
    return api.PROBLEM_SPECS[NAME].build({}, dict(flags or {}), None)


def _centre(built):
    return built.bounds.mean(axis=1)


@pytest.fixture(scope="module")
def baseline():
    b = _built()
    return b, b.evaluate(_centre(b))


def test_naming_a_section_changes_what_the_plate_costs(baseline):
    """The whole point: with an aerofoil on it the plate's profile drag is
    that section's own, so two sections are two different wings. Under the
    build-up they were the same wing twice."""
    base, r0 = baseline
    x = _centre(base)
    re_plate = float(r0["endplate_Re"])

    a = _built({api.SECTION_PLATE_KEY: {"name": FOIL_A, "re": re_plate}})
    b = _built({api.SECTION_PLATE_KEY: {"name": FOIL_B, "re": re_plate}})
    ra, rb = a.evaluate(x), b.evaluate(x)

    assert ra["feasible"] and rb["feasible"]
    assert ra["endplate_section_designed"] and rb["endplate_section_designed"]
    # the section is what moved, and it moved the PLATE's drag line
    assert ra["CD_endplate"] != rb["CD_endplate"]
    assert ra["CD"] != rb["CD"]
    # ...through its own two-dimensional drag coefficient, which is reported
    assert ra["endplate_cd_section"] > 0.0
    assert rb["endplate_cd_section"] > 0.0
    assert ra["endplate_cd_section"] != rb["endplate_cd_section"]
    # and the plate's INDUCED drag is still the lattice's, charged once
    assert ra["CDi"] == pytest.approx(r0["CDi"])


def test_the_section_is_flown_at_the_plates_own_reynolds_number(baseline):
    """A plate chord is not a wing chord. The wing's mean chord and the
    plate's differ by a factor the design box moves over, so a section chosen
    at the wing's Reynolds number is measured somewhere the plate never
    flies."""
    base, r0 = baseline
    assert r0["endplate_Re"] > 0.0
    assert r0["endplate_Re"] != pytest.approx(r0["Re_mac"], rel=0.05)

    r = _built({api.SECTION_PLATE_KEY: {"name": FOIL_A,
                                        "re": float(r0["endplate_Re"])}}
               ).evaluate(_centre(base))
    # the polar the plate flew names the point it was measured at
    assert f"{r0['endplate_Re']:.3g}" in str(r["endplate_polar"])


def test_the_plates_stiffness_is_its_own_shapes(baseline):
    """A section is chosen for two things, and drag is only one of them. With
    coordinates the plate's k_I is the integral of its own thickness
    distribution; the fallback exists but has to SAY it is a fallback."""
    base, r0 = baseline
    x = _centre(base)
    re_plate = float(r0["endplate_Re"])

    ra = _built({api.SECTION_PLATE_KEY: {"name": FOIL_A, "re": re_plate}}
                ).evaluate(x)
    rb = _built({api.SECTION_PLATE_KEY: {"name": FOIL_B, "re": re_plate}}
                ).evaluate(x)

    assert ra["endplate_I_m4"] != rb["endplate_I_m4"]
    assert ra["endplate_inertia_source"] == "designed section (its own " \
                                            "coordinates)"
    # ...and it is not the family constant it would have borrowed
    assert ra["endplate_I_m4"] != pytest.approx(r0["endplate_I_m4"])
    # the fallback names itself rather than passing as a measurement
    assert r0["endplate_inertia_source"] == "shaped"


def test_a_designed_sections_inertia_is_the_integral_of_its_thickness():
    """k_I is not quoted anywhere: a rectangle integrates to exactly 1/12, a
    NACA symmetric section to the number the family table already carries, and
    both come out of the same function the coordinates go through."""
    xi = np.linspace(0.0, 1.0, 501)
    assert ep.inertia_factor_of(xi, np.ones_like(xi)) == pytest.approx(1 / 12)
    # the shipped family constant IS this integral, not a literal beside it
    assert ep.SECTIONS["shaped"]["inertia_factor"] == \
        pytest.approx(ep._naca_inertia_factor())

    # ...and an outline gives the same answer as the law that drew it
    from aerobo.endplate import _naca_thickness_shape
    t = _naca_thickness_shape(xi)
    loop = np.vstack([np.column_stack([xi[::-1], +0.5 * t[::-1]]),
                      np.column_stack([xi[1:], -0.5 * t[1:]])])
    assert ep.section_inertia_factor(loop) == \
        pytest.approx(ep._naca_inertia_factor(), rel=1e-3)


def test_a_cambered_section_on_a_vertical_plate_is_refused():
    """The trap this module exists to close, now that a plate can be given a
    real section: alpha_L0 tilts a vertical panel exactly as it tilts a
    horizontal one, so a cambered plate at zero toe carries a side force
    nobody asked for — and the two plates' loads cancel in CY, so it would
    never have shown in the totals."""
    class _Cambered:
        alpha_L0 = -2.3
        tc = 0.12
        alpha_valid = (-8.0, 8.0)

        def cd(self, a):
            return np.full_like(np.asarray(a, dtype=float), 0.008)

    with pytest.raises(ValueError, match="must be SYMMETRIC"):
        ep.CarWingEndplateProblem(endplate_polar=_Cambered())

    # ...and a symmetric one is accepted, so the refusal is about camber and
    # not about giving the plate a section at all
    class _Symmetric(_Cambered):
        alpha_L0 = 0.0

    assert ep.CarWingEndplateProblem(endplate_polar=_Symmetric()) is not None


def test_a_fitted_zero_lift_angle_of_nearly_zero_is_still_symmetric():
    """THE GATE MUST NOT REFUSE ITS OWN LIBRARY. A measured polar's alpha_L0
    is a fit through XFOIL points, so a section that is symmetric BY
    CONSTRUCTION comes back at -0.009 deg rather than at 0 — and a gate tight
    enough to call that camber refuses sections the shell has just offered.

    The tolerance is derived from the family's own toe box (one per cent of
    its half-width), not chosen: the question is whether a zero-lift angle
    could command a side force the toe ROW cannot trivially cancel."""
    tol = ep.CarWingEndplateProblem.ALPHA_L0_TOL_DEG
    assert tol == pytest.approx(
        0.01 * max(abs(b) for b in
                   ep.CarWingEndplateProblem.ENDPLATE_TOE_BOUNDS_DEG))

    class _Fit:
        alpha_L0 = -0.009359          # the number a real run was refused on
        tc = 0.12
        alpha_valid = (-8.0, 8.0)

        def cd(self, a):
            return np.full_like(np.asarray(a, dtype=float), 0.008)

    assert ep.CarWingEndplateProblem(endplate_polar=_Fit()) is not None
    # ...and a real camber still does not get through on the same path
    _Fit.alpha_L0 = -2.3
    with pytest.raises(ValueError, match="must be SYMMETRIC"):
        ep.CarWingEndplateProblem(endplate_polar=_Fit())


def test_symmetry_is_measured_the_way_the_library_screen_measures_it():
    """ONE QUESTION, ONE ANSWER. Where the coordinates are there, the plate's
    symmetry test IS `api.symmetric_section_names`'s — a second, stricter one
    here would refuse sections the shell had just put in the menu."""
    name = api.symmetric_section_names()[0]
    coords = api.chosen_section_coords(name)
    assert coords is not None
    assert api.section_max_camber(coords) <= api.SYMMETRIC_CAMBER_TOL

    class _AnyPolar:
        alpha_L0 = -0.5               # would fail the polar fallback...
        tc = 0.12
        alpha_valid = (-8.0, 8.0)

        def cd(self, a):
            return np.full_like(np.asarray(a, dtype=float), 0.008)

    # ...and does not, because the COORDINATES are the measurement when they
    # are there, and the library screen already passed this section
    assert ep.CarWingEndplateProblem(endplate_polar=_AnyPolar(),
                                     endplate_coords=coords) is not None

    # a genuinely cambered outline is refused on that same measurement
    cambered = api.chosen_section_coords("goe741")
    if cambered is not None:
        with pytest.raises(ValueError, match="of chord of camber"):
            ep.CarWingEndplateProblem(endplate_polar=_AnyPolar(),
                                      endplate_coords=cambered)


def test_naming_no_section_leaves_the_published_plate_exactly(baseline):
    """Empty in, empty out. The whole point of a default is that a family
    gaining a freedom cannot move its own baseline."""
    base, r0 = baseline
    assert r0["endplate_section_designed"] is False
    assert r0["endplate_cd_section"] == 0.0
    assert r0["endplate_polar"] is None
    # the build-up is still what charges it: friction, a form factor and the
    # family's edge lump
    assert r0["CD_endplate"] == pytest.approx(r0["CD_endplate_friction"]
                                              + r0["CD_endplate_edges"])
    assert r0["CD_endplate_friction"] > 0.0


def test_the_flag_is_declared_only_where_the_plate_is_a_designed_part():
    """A flag is declared where it is READ. The plain car wing's plate is a
    fence of free height carrying the wing's own chord — there is no part
    there to give a section to — so offering the key on it would be a control
    that changes nothing."""
    assert api.SECTION_PLATE_KEY in api.PROBLEM_SPECS[NAME].flags
    assert api.SECTION_PLATE_KEY not in \
        api.PROBLEM_SPECS["car rear wing"].flags
    assert api.SECTION_PLATE_KEY not in \
        api.PROBLEM_SPECS["car rear wing (two-element)"].flags
    with pytest.raises(KeyError):
        api.check_flags("car rear wing", {api.SECTION_PLATE_KEY: FOIL_A})
    # ...and it survives onto the twins, because they share the builder
    for twin in (f"{NAME} + free chord law",
                 f"{NAME} [free cant]",
                 f"{NAME} [free cant, free blend] + free chord law"):
        assert api.SECTION_PLATE_KEY in api.PROBLEM_SPECS[twin].flags


def test_the_shell_offers_only_the_symmetric_members():
    """"Cannot restrict" and "no restriction" are opposite answers, and the
    plate's menu must never give the second when it means the first."""
    import gui.nice_app as v1

    opts = v1._plate_airfoil_options()
    assert opts[None].startswith("the section family")
    offered = [k for k in opts if k is not None]
    assert offered, "the coordinate sidecar is present, so a menu is possible"
    assert set(offered) == set(api.symmetric_section_names())


def test_the_mount_card_does_not_ask_for_the_plates_aerofoil():
    """ONE QUESTION, ONE PLACE — and here the second place would have lost
    silently: `config.flags` merges `car_flags` first and the per-surface
    section loop second, so a select on the Mount card would be overwritten
    by stage 2.8's answer without a word. So the card points at the stage
    instead of duplicating it."""
    import gui.nice_app as v1

    ch = v1.choices_from_problem(NAME)
    ch["car_endplate_airfoil"] = FOIL_A          # a stale/hand-set key
    assert v1.car_flags(ch, 55.0).get(api.SECTION_PLATE_KEY) is None
    assert "car_endplate_airfoil" not in v1.BUILDER_DEFAULTS
