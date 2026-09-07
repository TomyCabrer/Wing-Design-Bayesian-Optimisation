"""Wing SIZE as configuration: span and area chosen by the user.

The design vector RESHAPES a planform (taper, twist, chord law); it never
RESIZES one. So span and area are values — api.PLANFORM_KEYS — on every
fixed-planform problem, exactly like the winglet cant band or the tail
layout, and the honest contract is:

  1. absent flags reproduce the published problem BIT-FOR-BIT,
  2. a chosen area reaches the MISSION as well as the geometry (the trim
     target is CL = W/(qS), so the two cannot be allowed to disagree),
  3. an aspect ratio outside the band the solvers are valid over is refused
     up front, not scored,
  4. problems whose size is ALREADY a design variable — the free-planform
     aircraft (b and S both) and the car wing (its span, against a fixed
     reference area) — do NOT accept the flags, and the GUI never sends them
     there: a typed span beside a searched one is two answers to one
     question,
  5. the tandem pair's stagger, DEFINED as a fraction of the span, follows a
     chosen span instead of staying at its 10 m metres.

And the completeness rule that falls out of (4): a wing's span is TYPED or
SEARCHED, never both and never NEITHER. The third state — fixed and implied,
no field to type it into and no band to search it over — is what
:func:`test_every_family_with_a_wing_lets_someone_state_its_span` exists to
keep at zero.
"""

import numpy as np
import pytest

from aerobo import api
from aerobo.api import PROBLEM_SPECS


def _mid(bounds):
    return np.array([0.5 * (lo + hi) for lo, hi in bounds])


# ------------------------------------------------------------- the contract

def test_no_size_flags_is_bit_for_bit_the_published_problem():
    for name in ("trim wing", "winglet", "tail", "tandem",
                 "wing+airfoil (coupled)"):
        spec = PROBLEM_SPECS[name]
        base = spec.build({}, {}, None)
        again = spec.build({}, {"mach": None}, None)     # nothing size-ish
        x = _mid(base.bounds)
        assert base.evaluate(x)["score"] == again.evaluate(x)["score"], name


@pytest.mark.parametrize("name", ["trim wing", "winglet", "wing t/c + sweep",
                                  "mission wing", "tail", "tail (fixed arm)",
                                  "wing+airfoil (coupled)",
                                  "winglet, blended (span-capped)"])
def test_a_chosen_size_is_what_the_solver_flies(name):
    built = PROBLEM_SPECS[name].build({}, {"b_m": 12.0, "S_m2": 9.0}, None)
    assert built.problem.b == pytest.approx(12.0)
    assert built.problem.S == pytest.approx(9.0)
    out = built.evaluate(_mid(built.bounds))
    assert out["feasible"], out.get("reason")


def test_the_design_box_is_untouched_by_a_chosen_size():
    """Resizing must not silently move the search box: the vector is the
    same shape, over the same bounds — only the wing it reshapes changed."""
    spec = PROBLEM_SPECS["winglet"]
    a = spec.build({}, {}, None)
    b = spec.build({}, {"b_m": 12.0, "S_m2": 9.0}, None)
    assert a.param_labels == b.param_labels
    assert np.array_equal(a.bounds, b.bounds)


def test_aspect_ratio_is_what_changes_the_answer():
    """A pure scale-up at constant AR moves L/D only through Reynolds and
    the fixed-Re polar cannot see that, so it must land on the same answer;
    raising AR at constant area must raise L/D."""
    spec = PROBLEM_SPECS["trim wing"]
    base = spec.build({}, {}, None)
    x = _mid(base.bounds)
    scaled = spec.build({}, {"b_m": 20.0, "S_m2": 40.0}, None)   # AR 10 again
    assert scaled.evaluate(x)["LoD"] == pytest.approx(
        base.evaluate(x)["LoD"], rel=1e-12)
    slender = spec.build({}, {"b_m": 14.0, "S_m2": 10.0}, None)  # AR 19.6
    assert slender.evaluate(x)["LoD"] > base.evaluate(x)["LoD"]


def test_the_trim_target_follows_the_chosen_area():
    """CL_target = W/(qS). With a weight given, a bigger wing flies at a
    LOWER lift coefficient — the alternative (holding CL) would silently
    change the aeroplane's weight instead."""
    spec = PROBLEM_SPECS["trim wing"]
    W = api.default_mission_values("trim wing")["W_N"]
    small = spec.build({"W_N": W}, {}, None).problem
    big = spec.build({"W_N": W}, {"S_m2": 20.0}, None).problem
    assert small.CL_target == pytest.approx(0.5, rel=1e-12)
    assert big.CL_target == pytest.approx(0.25, rel=1e-9)
    # and the default WEIGHT offered for a resized wing is the one that
    # trims it to the legacy CL, so an untouched mission card is consistent
    W_big = api.default_mission_values("trim wing", {"S_m2": 20.0})["W_N"]
    assert W_big == pytest.approx(2.0 * W, rel=1e-12)
    assert api.mission_sref("trim wing", {"S_m2": 20.0}) == pytest.approx(20.0)


def test_an_unflyable_aspect_ratio_is_refused_not_scored():
    for flags in ({"b_m": 5.0, "S_m2": 20.0},        # AR 1.25
                  {"b_m": 40.0, "S_m2": 10.0},       # AR 160
                  {"b_m": -1.0}, {"S_m2": 0.0}):
        with pytest.raises(ValueError):
            PROBLEM_SPECS["trim wing"].build({}, flags, None)


def test_only_fixed_planform_problems_declare_the_flags():
    for name in ("trim wing", "winglet", "tail", "tandem",
                 "winglet + airfoil (XFOIL)", "wing (free chord law)",
                 "tail + free chord law",
                 # ...and the WATER families, since 2026-08-01: a foiling
                 # craft's span and area are the first two numbers its
                 # designer chooses, and the cavitation margin is recomputed
                 # at every panel's own submergence for whatever geometry the
                 # candidate draws — the calibration justifies the DEFAULT
                 # (1.2 m / 0.144 m2), not a ban
                 "hydrofoil", "hydrofoil + winglet",
                 "hydrofoil + elevator",
                 "hydrofoil + elevator [designed elevator + tip device]",
                 # ...and the water families as V3 actually flies them: the
                 # CHOSEN-SECTION twin (every V3 water pipeline flies the
                 # section stage 2 picked, so this is the ordinary case, not
                 # an exotic one) and the twin that DESIGNS its section
                 "hydrofoil [chosen section]",
                 "hydrofoil + elevator [free depth] [chosen section]",
                 "hydrofoil + CST section (XFOIL)",
                 "hydrofoil + elevator + winglet + CST section (XFOIL)",
                 # ...and the two air specs that shipped under their own
                 # names after this list was written: designing the
                 # STABILISER, or starting the tip device's transition
                 # inboard of the tip, says nothing about whether the WING's
                 # span is the user's
                 "tail [designed tail]", "tail (fixed arm) [designed tail]",
                 "winglet, blended into the wing (span-capped)"):
        assert api.resizable(name), name
    for name in ("free planform (aircraft)",       # b and S are variables
                 "car rear wing",                  # ...and so is the span
                 "car rear wing + endplates",
                 "airfoil (section)"):             # no wing at all
        assert not api.resizable(name), name


def test_the_water_default_is_the_published_foil():
    """Resizable does not mean resized: an untouched water build is the
    1.2 m / 0.144 m2 foil every published result was measured on."""
    for name in ("hydrofoil", "hydrofoil + winglet", "hydrofoil + elevator"):
        prob = PROBLEM_SPECS[name].build(None, {}, None).problem
        assert (prob.b, prob.S) == (1.2, 0.144), name
        assert api.planform_size(name) == (1.2, 0.144), name


def test_every_chord_twin_inherits_its_base_problem_s_size_flags():
    for base, twin in api.CHORD_TWINS.items():
        assert api.resizable(twin) == api.resizable(base), twin


def test_every_chosen_section_twin_inherits_its_base_problem_s_size_flags():
    """Same rule, the other axis — and the one that had actually broken.

    A twin copies its base's flags when it is REGISTERED, which happens
    before ``api._RESIZABLE_PROBLEMS`` is applied, so the water twins were
    born without the size their bases were about to be given. That is not a
    corner: V3 flies the section stage 2 chose, so under water the twin IS
    the pipeline, and the span silently stopped being a question exactly
    when the design was most complete.
    """
    for base, twin in api.CHOSEN_SECTION_TWINS.items():
        assert api.resizable(twin) == api.resizable(base), twin


# ------------------------------ a span is TYPED or SEARCHED, never neither

#: the only registered problem with no wing to size: the 2-D section search.
#: A span there would be a length with nothing to measure.
_NO_WING = "airfoil (section)"


def test_every_family_with_a_wing_lets_someone_state_its_span():
    """No problem may leave its span fixed AND implied.

    Two honest states: the span is TYPED (the family declares
    api.PLANFORM_KEYS, and the size card asks for it in metres) or SEARCHED
    (``b_m`` is a row of the design box, with a band the user states). A
    problem in neither state flies a span nobody chose and nobody can see —
    which is what 117 of them did: every chosen-section water twin, every
    water section co-design, both designed-tail specs and the wing-side
    blended winglet, plus their generated chord-law twins.
    """
    implied = sorted(
        name for name, spec in PROBLEM_SPECS.items()
        if name != _NO_WING
        and not api.resizable(name) and "b_m" not in spec.param_labels)
    assert implied == []


@pytest.mark.parametrize("name", [
    "winglet, blended into the wing (span-capped)",
    "tail [designed tail]", "tail (fixed arm) [designed tail]"])
def test_designing_a_surface_does_not_take_the_wing_s_span_away(name):
    """These three shipped as spec literals after the resizable list was
    written, so they honoured a size their spec did not advertise — and the
    GUI, which asks the spec, hid the field."""
    spec = PROBLEM_SPECS[name]
    published = spec.build({}, {}, None)
    built = spec.build({}, {"b_m": 14.0, "S_m2": 12.0}, None)
    assert built.problem.b == pytest.approx(14.0)
    assert built.problem.S == pytest.approx(12.0)
    # ...and it is FLOWN, not merely stored: the same vector on a wing of a
    # different aspect ratio has to land on a different answer
    x = _mid(built.bounds)
    assert built.evaluate(x)["feasible"]
    assert built.evaluate(x)["LoD"] != pytest.approx(
        published.evaluate(x)["LoD"], rel=1e-6)


@pytest.mark.parametrize("name,flags", [
    ("hydrofoil [chosen section]", {api.SECTION_KEY: "naca2412"}),
    ("hydrofoil + elevator [designed elevator + tip device] [chosen section]",
     {api.SECTION_KEY: "naca2412"}),
    ("hydrofoil + CST section (XFOIL)", {}),
    ("hydrofoil + winglet + CST section (XFOIL)", {}),
    ("hydrofoil + elevator [fixed arm] + CST section (XFOIL)", {})])
def test_a_water_family_is_built_at_the_size_it_is_given(name, flags):
    """Build only — the XFOIL twins cost seconds per new section, and what
    is under test is where the two numbers LAND, not what they score. The
    section problems wrap the foil, so the size is read on the surface that
    has one."""
    spec = PROBLEM_SPECS[name]
    built = spec.build({} if spec.uses_mission else None,
                       {**flags, "b_m": 2.4, "S_m2": 0.5}, None)
    foil = getattr(built.problem, "foil", built.problem)
    assert (foil.b, foil.S) == (pytest.approx(2.4), pytest.approx(0.5))
    # ...and untouched it is still the 1.2 m / 0.144 m2 calibration every
    # published water result was measured on
    plain = spec.build({} if spec.uses_mission else None, dict(flags), None)
    bare = getattr(plain.problem, "foil", plain.problem)
    assert (bare.b, bare.S) == (1.2, 0.144)


def test_the_water_chosen_section_twin_flies_the_span_it_is_given():
    """One evaluate, on the cheap twin: a foil twice as wide at three and a
    half times the area is a different foil, and the score has to say so."""
    spec = PROBLEM_SPECS["hydrofoil [chosen section]"]
    sec = {api.SECTION_KEY: "naca2412"}
    published = spec.build({}, dict(sec), None)
    resized = spec.build({}, {**sec, "b_m": 2.4, "S_m2": 0.5}, None)
    x = _mid(published.bounds)
    assert resized.evaluate(x)["LoD"] != pytest.approx(
        published.evaluate(x)["LoD"], rel=1e-6)


def test_planform_size_reports_what_will_be_flown():
    assert api.planform_size("trim wing") == (10.0, 10.0)
    assert api.planform_size("trim wing", {"b_m": 12.0}) == (12.0, 10.0)
    assert api.planform_size("tandem") == (10.0, 20.0)      # TOTAL area
    assert api.planform_size("hydrofoil") == (1.2, 0.144)
    # no fixed planform to report: b and S are design variables / no wing
    assert api.planform_size("free planform (aircraft)") is None
    assert api.planform_size("airfoil (section)") is None


def test_the_tandem_stagger_follows_the_span_it_is_defined_from():
    built = PROBLEM_SPECS["tandem"].build({}, {"b_m": 12.0, "S_m2": 24.0},
                                          None)
    prob = built.problem
    assert prob.b == pytest.approx(12.0)
    assert prob.S_total == pytest.approx(24.0)
    assert prob.dx == pytest.approx(6.0)      # b/2, as the default states
    assert prob.dz == pytest.approx(1.2)      # 0.1 b
    assert built.evaluate(_mid(built.bounds))["feasible"]


def test_the_tandem_size_is_validated_per_wing():
    """Total area would report half the aspect ratio each surface flies at,
    so a pair that is fine on paper but AR 2 per wing must still be refused."""
    with pytest.raises(ValueError, match="aspect ratio"):
        PROBLEM_SPECS["tandem"].build({}, {"b_m": 5.0, "S_m2": 24.0},
                                      None)   # AR 2.08 per wing


def test_a_run_config_round_trips_the_size():
    cfg = api.RunConfig(problem_name="trim wing", optimiser="random",
                        budget=4, seed=0, flags={"b_m": 12.0, "S_m2": 9.0})
    res = api.run(cfg)
    assert res.config["flags"]["b_m"] == 12.0
    # the breakdown AR is the SOLVER's (numerical planform area from the
    # station grid), so it matches the geometric 12^2/9 to quadrature
    assert res.breakdown["AR"] == pytest.approx(16.0, rel=1e-3)


# ----------------------------------------------------------------- the GUI

def test_the_builder_only_sends_a_size_where_it_is_honoured():
    from gui import nice_app as v1

    ch = dict(v1.BUILDER_DEFAULTS)
    assert v1.planform_flags(ch) == {}                    # untouched
    ch["span_m"], ch["area_m2"] = 12.0, 9.0
    assert v1.planform_flags(ch) == {"b_m": 12.0, "S_m2": 9.0}
    # ...the water families take one too (the hydrofoil's size is a design
    # question, not a calibration to protect)
    water = dict(ch, medium="water")
    assert v1.planform_flags(water) == {"b_m": 12.0, "S_m2": 9.0}
    # ...but never to a solver that would IGNORE it: with the size freed it
    # is a design variable, so a typed one would be two answers to one
    # question
    free = dict(ch, planform="free")
    assert v1.planform_flags(free) == {}
    # ...nor to the car, whose SPAN is a design variable with a band of its
    # own (carwing.span_row)
    track = dict(ch, medium="track")
    assert v1.planform_flags(track) == {}


def test_the_geometry_summary_reports_the_chosen_wing():
    from gui import nice_app as v1

    rows = dict(v1.geometry_summary("trim wing", {"taper": [0.2, 1.0]},
                                    {"b_m": 12.0, "S_m2": 9.0}))
    assert rows["span b"].startswith("12")
    assert rows["area S"].startswith("9")
    assert rows["aspect ratio"].startswith("16")
    assert "chosen" in rows["span b"]
    # untouched still reads as the solver's own
    plain = dict(v1.geometry_summary("trim wing", {"taper": [0.2, 1.0]}))
    assert plain["span b"] == "10 m · fixed"


def test_the_size_card_is_offered_exactly_where_the_flags_are():
    from gui import nice_app as v1

    assert v1.planform_resizable("trim wing")
    assert v1.planform_resizable("hydrofoil")        # since 2026-08-01
    assert not v1.planform_resizable("car rear wing")
    assert v1.planform_size_defaults("trim wing") == (10.0, 10.0)
    assert v1.planform_size_defaults("airfoil (section)") is None
    note = v1.planform_size_note("tail", dict(v1.BUILDER_DEFAULTS))
    assert "static-margin" in note and "bit-for-bit" in note
