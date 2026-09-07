"""The SLOTTED rear wing, as a registered problem and as a shell control.

``carwing_multi.py`` is tested as physics by its own file. This one tests the
five things that make it *reachable*, each of which this repository has been
bitten by before:

* the family is REGISTERED with the vector it really searches and the margins
  it really returns (a static ``ProblemSpec`` that disagrees with its builder
  is a menu that lies);
* every flag key is declared exactly where it is READ — the mount continuum
  and the lap belong to the single-element family, the panel count to this
  one, and ``check_flags`` refuses each of them everywhere else. A flag that
  is silently dropped is worse than one that is refused;
* the CHORD-LAW twin builds and evaluates, because the slot rows sit ahead of
  the size block precisely so that the chord coefficients stay the tail of the
  vector;
* the shell ROUND-TRIPS: choices -> problem -> choices, in both shells;
* the box the CARD QUOTES is the box the RUN SEARCHES.

...and the four decisions this surface has since been rebuilt around, each
asserted as an OUTCOME so that reverting one fails the test:

* THE DESIGN BOX OWNS THE SIZE BANDS. ``b_m`` and ``S_m2`` are rows of the
  design box and nothing else; there is no ``span_min_m`` / ``area_min_m2``
  flag left to answer the same question a second time, and a widened row
  reaches the PROBLEM's own bounds rather than only the sampler's;
* THE REFERENCE AREA IS ALWAYS A DESIGN VARIABLE. The "(free area)" names are
  gone and the BASE names are the free-area problems — six car entries, not
  twelve — while the SOLVER stays general;
* BOTH MOUNT LAYOUTS TAKE THE LOAD OUT THROUGH THE PLATES. ``tips`` and
  ``inboard``, no pylon anywhere, differing in the beam and in what the extra
  sheets cost;
* NOTHING IS BUDGETED BY DEFAULT. One margin with nothing stated; a drag
  ceiling and a downforce floor are limits the user may state, in newtons.

The car's SPEED still has to reach the solver, and the size card still has to
price the wing in the car's own currency.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aerobo import api                                        # noqa: E402
from gui import nice_app as v1                                # noqa: E402

MULTI = "car rear wing (two-element)"
SINGLE = "car rear wing"
PLATES = "car rear wing + endplates"

#: every car family the registry offers, base names only. The area is a
#: design variable on all three, so there is no fixed-area/free-area pair.
CAR_BASES = (SINGLE, MULTI, PLATES)

#: the four rows the slot adds, in the order carwing_multi lays them out
SLOT_ROWS = ("flap_chord_frac", "flap_deflection_deg",
             "slot_gap_frac", "slot_overlap_frac")

#: the two SIZE rows every car family designs, in vector order. The area goes
#: ahead of the span (api._CAR_WING_AREA_LABELS says why).
SIZE_ROWS = ("S_m2", "b_m")

#: the two plate-borne mount layouts. There is no pylon layout.
MOUNT_NAMES = ("tips", "inboard")

#: the size bands as FLAGS. Every one of these is refused now: how wide and
#: how big the wing may be is asked once, in the design box.
DEAD_SIZE_FLAGS = ("span_min_m", "span_max_m", "area_min_m2", "area_max_m2")

def _mid(built) -> np.ndarray:
    return 0.5 * (built.bounds[:, 0] + built.bounds[:, 1])


# ------------------------------------------------- 1. the family is registered

def test_the_family_is_registered_with_the_vector_it_searches():
    """The static spec and the built problem must agree — on the LABELS, not
    only on the count, and the slot rows must sit AHEAD of the size block."""
    from aerobo import carwing_multi

    spec = api.PROBLEM_SPECS[MULTI]
    built = spec.build({}, {}, None)
    assert built.param_labels == spec.param_labels
    assert built.dim == len(spec.param_labels) == built.bounds.shape[0]
    assert built.param_labels == tuple(built.problem.param_labels)
    # the slot block, then the size block (area then span), then nothing else
    assert built.param_labels[-len(SIZE_ROWS):] == SIZE_ROWS
    i = built.param_labels.index(SLOT_ROWS[0])
    assert built.param_labels[i:i + 4] == SLOT_ROWS
    assert i + 4 == len(built.param_labels) - len(SIZE_ROWS)
    # ...and the bands are the module's own, not restated here
    assert tuple(built.bounds[i]) == carwing_multi.FLAP_CHORD_FRAC_BOUNDS
    assert tuple(built.bounds[i + 1]) == \
        carwing_multi.FLAP_DEFLECTION_BOUNDS_DEG
    assert tuple(built.bounds[i + 2]) == carwing_multi.SLOT_GAP_FRAC_BOUNDS
    assert tuple(built.bounds[i + 3]) == carwing_multi.SLOT_OVERLAP_FRAC_BOUNDS


def test_the_display_string_carries_a_dimension_token():
    """The variant machinery RETARGETS the dimension with a regex, so a
    display string without an ``N-D`` token silently keeps the base family's
    dimension on every generated twin."""
    for name in CAR_BASES:
        spec = api.PROBLEM_SPECS[name]
        assert api._DIM_TOKEN.search(spec.display), spec.display
        assert f"{len(spec.param_labels)}-D" in spec.display, spec.display
    # ...and the twin really is retargeted rather than repeating the base
    base_dim = len(api.PROBLEM_SPECS[MULTI].param_labels)
    twin = api.PROBLEM_SPECS[f"{MULTI} + free chord law"]
    assert len(twin.param_labels) == base_dim + 3
    assert f"{base_dim + 3}-D" in twin.display, twin.display


def test_the_area_is_a_design_row_and_the_fixed_area_names_are_gone():
    """THE REFERENCE AREA IS ALWAYS A DESIGN VARIABLE.

    "How big is this wing" used to be answered by picking a problem NAME —
    each car family had a fixed-area entry and a "(free area)" twin, and the
    two disagreed about the design vector. It is a band on a design variable
    like every other, so the pair collapsed and the BASE name is the
    free-area problem.

    Asserted as what a user can see: the old names cannot be selected at all,
    the registry offers six car entries rather than twelve, and every base
    family SEARCHES its area — the row is in the vector, it is drawn from,
    and moving it moves the answer.
    """
    for name in CAR_BASES:
        assert f"{name.rstrip(')')}, free area)" not in api.PROBLEM_SPECS
    assert "car rear wing (free area)" not in api.PROBLEM_SPECS
    assert "car rear wing (two-element, free area)" not in api.PROBLEM_SPECS
    assert "car rear wing + endplates (free area)" not in api.PROBLEM_SPECS
    cars = [n for n in api.PROBLEM_SPECS if n.startswith("car rear wing")]
    # three bases, the designed-plate family's three PLATE-freedom twins (its
    # cant and its root blend, searched instead of stated), and a
    # free-chord-law twin of every one of the six
    bases = set(CAR_BASES) | {
        "car rear wing + endplates [free cant]",
        "car rear wing + endplates [free blend]",
        "car rear wing + endplates [free cant, free blend]"}
    assert len(cars) == 12, cars
    assert set(cars) == bases | {f"{n} + free chord law" for n in bases}

    for name in CAR_BASES:
        built = api.PROBLEM_SPECS[name].build({}, {}, None)
        labels = list(built.param_labels)
        assert labels[-2:] == list(SIZE_ROWS), name
        # the row is really flown: two draws that differ ONLY in area report
        # two different reference areas and two different answers
        j = labels.index("S_m2")
        small, big = _mid(built), _mid(built)
        small[j], big[j] = built.bounds[j]
        a, b = built.evaluate(small), built.evaluate(big)
        assert a["feasible"] and b["feasible"], (name, a.get("reason"))
        assert a["S_m2"] < b["S_m2"], name
        assert a["score"] != b["score"], name


def test_the_solver_stays_general_even_though_the_registry_does_not():
    """The control on the decision above: only the REGISTRY always frees the
    area. ``carwing.CarWingProblem`` handed no band still fixes it and still
    defaults to the downforce COEFFICIENT — which is what ``carsection.py``'s
    fixed reference planform and ``compare_split``'s matched-area comparison
    are built on. Deleting the fixed-area branch instead of un-registering it
    would take those with it.
    """
    from aerobo import carwing

    fixed = carwing.CarWingProblem(area_bounds_m2=None)
    assert fixed.area_bounds_m2 is None
    assert fixed.objective == "cz"          # well posed against a fixed area
    free = carwing.CarWingProblem(area_bounds_m2=carwing.AREA_BOUNDS_M2,
                                  objective="efficiency")
    # the band is one more design row, and it is the LAST but one (the span
    # keeps its slot either way)
    assert free.dim == fixed.dim + 1
    assert list(api.PROBLEM_SPECS[SINGLE].param_labels)[-2:] == list(SIZE_ROWS)
    # ...and the registry never reaches the fixed-area branch, because it
    # always hands a band: every registered car family is one row wider
    assert api.PROBLEM_SPECS[SINGLE].build({}, {}, None).dim == free.dim


def test_nothing_is_budgeted_by_default_and_a_limit_is_a_force():
    """NOTHING IS BUDGETED BY DEFAULT.

    A drag ALLOWANCE handed to a maximiser is a number the answer rides, so
    the registered families carry no drag margin at all: what survives is the
    deflection margin the problem declares unconditionally. A ceiling is a
    limit the user may state, in newtons, and so is a downforce floor.

    The report reads the PROBLEM's labels while the menu quotes the SPEC's,
    so the two must agree at the shipped configuration and move together away
    from it.
    """
    for name in CAR_BASES:
        spec = api.PROBLEM_SPECS[name]
        built = spec.build({}, {}, None)
        assert built.problem.constraint_labels == spec.constraint_labels, name
        assert built.problem.n_constraints == spec.n_constraints, name
        assert "drag" not in " ".join(spec.constraint_labels), name
        assert built.problem.CD_budget is None, name
        assert built.problem.drag_budget_n is None, name
        assert built.problem.downforce_min_n is None, name
        score, g = built.callable(_mid(built))
        assert len(g) == built.problem.n_constraints, name
        assert np.isfinite(score), name

    plain = api.PROBLEM_SPECS[MULTI].build({}, {}, None)
    assert plain.problem.constraint_labels == ("deflection margin",)
    assert len(plain.callable(_mid(plain))[1]) == 1

    # a FLOOR is one more margin, and it is the one that was asked for
    floored = api.PROBLEM_SPECS[MULTI].build(
        {}, {"downforce_min_n": 200.0}, None)
    assert floored.problem.constraint_labels == ("deflection margin",
                                                 "downforce floor margin")
    assert len(floored.callable(_mid(floored))[1]) == 2

    # ...and a CEILING is stated in NEWTONS, which is the unit that still
    # means something once the reference area is itself designed
    capped = api.PROBLEM_SPECS[MULTI].build(
        {}, {"CD_budget": "off", "drag_budget_n": 90.0}, None)
    assert capped.problem.constraint_labels == ("drag force margin",
                                                "deflection margin")
    assert capped.problem.drag_budget_n == 90.0

    # the published allowance still EXISTS — it is simply not a default any
    # more, which is the whole decision
    from aerobo import carwing
    assert carwing.published_drag_budget_n() > 0.0


def test_the_family_cannot_maximise_a_coefficient():
    """CZ is referenced to the very area being searched, so maximising it
    shrinks the wing — and here the shrink would be credited to the slot.

    So the registered family opens on ``api.CAR_DEFAULT_OBJECTIVE`` instead,
    and asking for the coefficient is refused by name rather than answered
    with a wing that got smaller.
    """
    built = api.PROBLEM_SPECS[MULTI].build({}, {}, None)
    assert built.problem.objective == api.CAR_DEFAULT_OBJECTIVE == "efficiency"
    with pytest.raises(ValueError, match="meaningless with a free"):
        api.PROBLEM_SPECS[MULTI].build({}, {"car_objective": "cz"}, None)
    # ...on every car family, for the same reason
    for name in CAR_BASES:
        with pytest.raises(ValueError, match="meaningless with a free"):
            api.PROBLEM_SPECS[name].build({}, {"car_objective": "cz"}, None)


def test_the_family_refuses_a_mission_card_by_name():
    """Its operating point is the single speed V; silently dropping a weight
    and an altitude would let the Mission card look like it did something."""
    with pytest.raises(ValueError, match="no mission spec"):
        api.PROBLEM_SPECS[MULTI].build({"W_N": 100.0}, {}, None)
    assert api.PROBLEM_SPECS[MULTI].uses_mission is False


def test_the_family_takes_the_chord_modifier_and_nothing_else():
    """Omitting ``_VARIANT_SUPPORT`` grants all five modifiers and registers
    bogus size and flight twins."""
    for name in (MULTI, SINGLE):
        assert api._VARIANT_SUPPORT[name] == ("chord",)
        assert api.add_modifier(name, "flight") is None
        assert api.with_modifiers(name, {"size_ws"}) is None
        assert not api.resizable(name)          # the SIZE rows ARE the design
        assert name not in api._WING_LOADING_FAMILIES


# ------------------------------------------ 2. every flag is read where it is

def test_the_panel_count_is_read_only_by_the_slotted_family():
    built = api.PROBLEM_SPECS[MULTI].build({}, {"n_section_nodes": 160}, None)
    assert built.problem.n_section_nodes == 160
    assert "n_section_nodes" in api.CAR_WING_MULTI_KEYS
    for other in (SINGLE, PLATES):
        with pytest.raises(KeyError, match="n_section_nodes"):
            api.check_flags(other, {"n_section_nodes": 160})


def test_the_mount_continuum_is_refused_on_the_slotted_family_and_the_lap_is_not():
    """``CarWingMultiProblem`` has no mount_spec, no deck, no dCZ/dh ceiling
    and no flown-Reynolds switch — declared there they would be accepted and
    read by nothing. It DOES have ``track_spec``/``car_spec``, so the two lap
    keys are declared and reach real fields.

    Catches: declaring the whole single-element tuple on the slotted family
    (the mount keys go through check_flags and are then dropped on the floor),
    and the reverse — dropping the lap keys after the fields were added, which
    makes ``car_track`` unreachable through the registry while
    ``CarWingMultiProblem`` can plainly answer it.
    """
    for key in api.CAR_SINGLE_ONLY_KEYS:
        assert key in api.accepted_flags(SINGLE), key
        assert key not in api.accepted_flags(MULTI), key
        with pytest.raises(KeyError, match=key):
            api.check_flags(MULTI, {key: 0.25})
    for key in api.CAR_LAP_KEYS:
        for name in (SINGLE, MULTI):
            assert key in api.accepted_flags(name), (name, key)
    # ...and it is not merely accepted: it reaches the field and is flown
    built = api.PROBLEM_SPECS[MULTI].build(
        {}, {"car_track": "synthetic", "car_objective": "laptime",
             "track_points": 3}, None)
    assert built.problem.track_spec is not None
    assert built.problem.track_points == 3
    out = built.evaluate(_mid(built))
    assert out["feasible"], out["reason"]
    assert out["lap_time_s"] > 0.0
    assert out["score"] == pytest.approx(-out["lap_time_s"])
    assert len(out["track_points"]) <= 3
    # a sampling count without a lap to sample is refused here too
    with pytest.raises(ValueError, match="track_points"):
        api.PROBLEM_SPECS[MULTI].build({}, {"track_points": 3}, None)


def test_the_two_car_families_time_the_same_lap():
    """A lap on the slotted family and a lap on its single-element twin are
    the same lap MODEL — same circuit, same car, same reference speed set —
    so a difference between the two is the wing.

    This is the contract the transfer measurement rests on. Catches: giving
    the slotted family its own track default, its own car, or its own
    ``track_points``, any of which turns "the slot bought 0.4 s" into a
    comparison of two circuits.
    """
    from aerobo import carwing, carwing_multi

    single = api.PROBLEM_SPECS[SINGLE].build(
        {}, {"car_track": "synthetic", "car_objective": "laptime"}, None)
    multi = api.PROBLEM_SPECS[MULTI].build(
        {}, {"car_track": "synthetic", "car_objective": "laptime"}, None)
    assert single.problem.track_spec == multi.problem.track_spec
    assert single.problem.car_used == multi.problem.car_used
    assert single.problem.track_points == multi.problem.track_points
    # the same objective label, off the same shared dictionary
    assert (carwing.CAR_OBJECTIVES["laptime"]
            == carwing_multi.CAR_MULTI_OBJECTIVES["laptime"])
    # ...and both report the lap under the SAME keys, as a POSITIVE time
    # scored as its negative, on the same named circuit
    for built in (single, multi):
        out = built.evaluate(_mid(built))
        assert out["feasible"], out["reason"]
        assert out["lap_time_s"] > 0.0
        assert out["score"] == pytest.approx(-out["lap_time_s"])
        assert out["lap"]["track"] == "synthetic reference lap (not a real circuit)"
        assert out["lap"]["length_m"] == pytest.approx(3280.0)
        assert {r["V"] for r in out["track_points"]}


def test_the_mount_refinements_are_refused_without_the_continuum():
    """They describe a carmount.MountSpec; the published two-valued lookup
    has no attachment chord, no GJ and no suction loss to set."""
    from aerobo import carmount

    for key in api._CAR_MOUNT_REFINE_KEYS:
        with pytest.raises(ValueError, match="MountSpec"):
            api.PROBLEM_SPECS[SINGLE].build({}, {key: 0.3}, None)
    built = api.PROBLEM_SPECS[SINGLE].build(
        {}, {"car_mount_model": "continuum", "x_attach_frac": 0.35,
             "gj_nm2": 1.0e5, "mount": "inboard"}, None)
    spec = built.problem.mount_spec
    assert spec is not None and spec.kind == "endplate"
    assert (spec.x_attach_frac, spec.gj_nm2) == (0.35, 1.0e5)
    # the LAYOUT the user picked became the spec — the station is the inboard
    # grip's, not the tip's, and no pylon came with it
    assert spec.station_frac == carmount.INBOARD_STATION_FRAC
    assert spec.n_pylons == 0
    # stated ONCE: the spec IS the mount, so the plain field is left at the
    # family's own default rather than sent alongside (CarWingProblem refuses
    # the pair rather than resolving it with a precedence rule)
    assert built.problem.mount == "tips"


def test_a_lap_needs_a_circuit_and_a_circuit_needs_a_lap():
    with pytest.raises(ValueError, match="track_points"):
        api.PROBLEM_SPECS[SINGLE].build({}, {"track_points": 3}, None)
    built = api.PROBLEM_SPECS[SINGLE].build(
        {}, {"car_track": "synthetic", "car_objective": "laptime"}, None)
    out = built.evaluate(_mid(built))
    assert out["feasible"] and out["lap_time_s"] > 0.0
    assert out["score"] == pytest.approx(-out["lap_time_s"])
    # ...and the slotted family, which has no circuit, refuses the objective
    # rather than raising KeyError at the first evaluation
    with pytest.raises(ValueError, match="needs a circuit"):
        api.PROBLEM_SPECS[MULTI].build({}, {"car_objective": "laptime"}, None)


def test_the_shared_car_knobs_still_reach_the_slotted_family():
    """The VALUE flags reach the fields — and the SIZE band is not among them
    any more. It is a row of the design box, so it arrives as a bounds
    override and the flag spelling is refused outright."""
    built = api.PROBLEM_SPECS[MULTI].build(
        {}, {"mount": "inboard", "V": 40.0, "deflection_limit_m": 0.02},
        {"b_m": (1.3, 1.5)})
    p = built.problem
    assert (p.mount, p.V, p.deflection_limit_m) == ("inboard", 40.0, 0.02)
    assert tuple(built.bounds[built.param_labels.index("b_m")]) == (1.3, 1.5)
    # ...and the PROBLEM's own band moved with it, which is the half that
    # used to be missing (see the size-row tests below)
    assert p.span_bounds_m == (1.3, 1.5)
    for key in DEAD_SIZE_FLAGS:
        with pytest.raises(KeyError, match=key):
            api.check_flags(MULTI, {key: 1.3})


def test_both_mount_layouts_take_the_load_out_through_the_plates():
    """BOTH MOUNT LAYOUTS TAKE THE LOAD OUT THROUGH THE PLATES.

    The pylon layout is gone from every registered car family, so the two
    remaining layouts are the same load path gripping at two different
    STATIONS. Asserted as the trade a user reads off the breakdown rather
    than as the dict: gripping inboard is stiffer, and it is not free — it
    buys the stiffness with two extra sheets of wetted area and the two
    corners where they meet the wing, so it costs drag and it costs score.

    Catches: reinstating a mount that carries the load anywhere but the
    plates, and swapping the two layouts' beams (an interior support that
    made the wing softer would still "differ", and the old test would have
    passed).
    """
    from aerobo import carwing

    assert set(carwing.MOUNTS) == set(MOUNT_NAMES)
    assert "centre" not in carwing.MOUNTS and "ends" not in carwing.MOUNTS
    with pytest.raises(ValueError, match="unknown mount 'centre'"):
        api.PROBLEM_SPECS[MULTI].build({}, {"mount": "centre"}, None)

    out = {}
    for name in MOUNT_NAMES:
        built = api.PROBLEM_SPECS[MULTI].build({}, {"mount": name}, None)
        assert built.problem.mount == name
        o = built.evaluate(_mid(built))
        assert o["feasible"], (name, o["reason"])
        out[name] = o
    tips, inboard = out["tips"], out["inboard"]

    # the BEAM: an interior support is stiffer, by a lot
    assert inboard["deflection_m"] < tips["deflection_m"]
    assert inboard["deflection_m"] < 0.5 * tips["deflection_m"]
    assert inboard["g_deflection"] > tips["g_deflection"]
    # ...and the tip-borne wing pays nothing for it, while the inboard grip
    # pays in SHEETS and in the corners they make
    assert tips["cd0_struts"] == 0.0 and tips["CD_junction"] == 0.0
    assert inboard["cd0_struts"] > 0.0 and inboard["CD_junction"] > 0.0
    assert inboard["CD"] > tips["CD"]
    # the same downforce for more drag, so the efficiency score falls
    assert inboard["CZ"] == pytest.approx(tips["CZ"])
    assert inboard["score"] < tips["score"]


def test_no_registered_car_family_has_a_pylon():
    """The other half of the same decision, and the one a dict rename could
    hide: whichever layout is chosen and whichever mount MODEL prices it,
    nothing reaching up through the wake is built or charged."""
    from aerobo import carmount

    assert set(carmount.PUBLISHED_LAYOUTS) == set(MOUNT_NAMES)
    for name in MOUNT_NAMES:
        assert carmount.PUBLISHED_LAYOUTS[name].kind == "endplate"
        assert carmount.PUBLISHED_LAYOUTS[name].n_pylons == 0
        built = api.PROBLEM_SPECS[SINGLE].build(
            {}, {"car_mount_model": "continuum", "mount": name}, None)
        o = built.evaluate(_mid(built))
        assert o["feasible"], (name, o["reason"])
        assert o["n_pylons"] == 0, name
        assert o["mount_kind"] == "endplate", name
    # the two layouts still grip at different stations under the continuum...
    assert (carmount.PUBLISHED_LAYOUTS["tips"].station_frac
            > carmount.PUBLISHED_LAYOUTS["inboard"].station_frac)
    # ...and the continuum PRICES that difference exactly as the published
    # lookup does. This is the assertion that matters: "no pylon" must not be
    # read as "no charge". A MountSpec used to price pylons only, and with
    # neither layout carrying one the continuum path handed the inboard grip
    # its stiffness for free — same CD, same score as the tip-borne wing,
    # 4.5x stiffer — which any optimiser would take.
    same = {}
    for model in ("published", "continuum"):
        for name in MOUNT_NAMES:
            fl = {"mount": name}
            if model == "continuum":
                fl["car_mount_model"] = model
            b = api.PROBLEM_SPECS[SINGLE].build({}, fl, None)
            o = b.evaluate(_mid(b))
            same[(model, name)] = (o["CD"], o["cd0_struts"],
                                   o["CD_junction"], o["score"])
    for name in MOUNT_NAMES:
        assert same[("continuum", name)] == pytest.approx(
            same[("published", name)], rel=1e-12), name
    tips, inboard = same[("continuum", "tips")], same[("continuum", "inboard")]
    assert inboard[1] > 0.0 and tips[1] == 0.0     # sheet charge
    assert inboard[2] > 0.0 and tips[2] == 0.0     # its two corners
    assert inboard[0] > tips[0] and inboard[3] < tips[3]


# ------------------------------------------------ 3. the chord-law twin flies

def test_the_chord_law_twin_builds_and_evaluates():
    """The slot rows sit AHEAD of the size block so that the chord
    coefficients stay the TAIL of the vector — assert the consequence, which
    is that a chord law actually reshapes the planform of a flyable design."""
    twin = f"{MULTI} + free chord law"
    assert api.base_of(twin) == MULTI
    built = api.PROBLEM_SPECS[twin].build({}, {}, None)
    assert built.param_labels[-3:] == ("chord_k1", "chord_k2", "chord_k3")
    # ...and the SIZE block is still where it was, immediately ahead of them:
    # the area row joining the vector must not have pushed the span out of
    # the slot carwing.span_from_x counts back to
    assert built.param_labels[-5:-3] == SIZE_ROWS
    assert built.dim == api.PROBLEM_SPECS[MULTI].build({}, {}, None).dim + 3

    x = _mid(built)
    flat = built.evaluate(x)
    assert flat["feasible"], flat["reason"]
    assert flat["n_elements"] == 2
    # off the flown planform, not off a breakdown key: NEITHER car family
    # reports ``chord_dev`` (see the session note), and asserting on a key
    # that does not exist would pass for the wrong reason if it appeared
    assert flat["wing"].chord_dev == pytest.approx(0.0, abs=1e-9)

    x[-3] = 0.3                              # bend the chord law
    bent = built.evaluate(x)
    assert bent["feasible"], bent["reason"]
    assert bent["wing"].chord_dev > 0.05
    # the area is HELD by the law, so the reference is untouched and only the
    # aerodynamics moved
    assert bent["S_m2"] == pytest.approx(flat["S_m2"])
    assert bent["CZ"] != flat["CZ"]


def test_the_slot_rows_are_the_freedom_and_they_move_the_answer():
    """A registered row nothing reads is the failure this asserts against."""
    built = api.PROBLEM_SPECS[MULTI].build({}, {}, None)
    x = _mid(built)
    base = built.evaluate(x)
    assert base["feasible"], base["reason"]
    i = built.param_labels.index("flap_deflection_deg")
    x2 = x.copy()
    x2[i] = x[i] + 8.0
    more = built.evaluate(x2)
    assert more["feasible"], more["reason"]
    # more flap is more downforce, and the slot the flow sees OPENS as it
    # deflects (carwing_multi measures this; it is the opposite of the
    # intuition, which is why it is asserted rather than assumed)
    assert more["CZ"] > base["CZ"]
    assert more["slot_width_frac"] > base["slot_width_frac"]
    assert more["slot_width_frac"] < x[built.param_labels.index(
        "slot_gap_frac")]


# --------------------------------------------------- 4. the shell round-trips

def test_the_shell_round_trips_the_family():
    """choices -> problem -> choices, for the BASE names. There is no
    free-area switch to carry round any more: the area row is in every one of
    these vectors, so the round trip has one fewer thing to lose."""
    for name in (SINGLE, MULTI):
        ch = v1.choices_from_problem(name)
        assert ch["medium"] == "track"
        assert ch["car_two_element"] is (name == MULTI)
        # the MOUNT is the family key now, and neither of these two is the
        # designed-plate family, so both round-trip through the pylon answer
        assert ch["car_endplates"] is False
        assert "car_free_area" not in ch
        assert v1.derive_problem(ch)[0] == name
        # ...and the flags those choices send are honoured by that family
        flags = v1.car_flags(ch)
        api.check_flags(name, flags)
        built = api.PROBLEM_SPECS[name].build({}, flags, None)
        assert built.dim == len(api.PROBLEM_SPECS[name].param_labels)
        assert "S_m2" in built.param_labels


def test_the_tandem_switch_on_the_track_asks_for_the_slot_and_says_so():
    """A car's rear wing has nothing behind it, so 'two surfaces' on the track
    can only mean a slot — and it has a solver now."""
    ch = dict(v1.start_choices(medium="track"), chord="fixed",
              system="tandem")
    name, notes = v1.derive_problem(ch)
    assert name == MULTI, name
    assert any("nothing behind it" in n and "SLOTTED" in n for n in notes), \
        notes


def test_designed_endplates_do_not_travel_to_the_slotted_family():
    """One plate spans both elements, so that family has no designed-plate
    twin — and its flags must not be sent to it, or check_flags refuses the
    run at the Run button."""
    ch = dict(v1.start_choices(medium="track"), chord="fixed",
              car_endplates=True, car_two_element=True,
              car_endplate_chord_min_m=0.1)
    name, notes = v1.derive_problem(ch)
    assert name == MULTI, name
    assert any("no two-element solver" in n for n in notes), notes
    flags = v1.car_flags(ch)
    assert "section" not in flags and "endplate_chord_min_m" not in flags
    api.check_flags(name, flags)


def test_an_untouched_slotted_card_sends_the_published_flag_set():
    """One flag, and it is the family's own default — so an untouched card is
    the published run bit for bit. There is no size band here and no budget
    menu: both were card questions once, and both moved."""
    ch = dict(v1.start_choices(medium="track"), car_two_element=True)
    # car_endplates opens True (the plates carry the car) and the two-element
    # gate stops every plate flag travelling anyway — one flag reaches the
    # slotted family, and it is the family's own default
    assert v1.car_flags(ch) == {"mount": "tips"}
    assert v1.derive_problem(ch)[0] == f"{MULTI} + free chord law"
    # the OUTCOME of "bit for bit": the card's default mount is not merely
    # spelled the same as the family's, it flies the same design
    carded = api.PROBLEM_SPECS[MULTI].build({}, v1.car_flags(ch), None)
    bare = api.PROBLEM_SPECS[MULTI].build({}, {}, None)
    assert carded.evaluate(_mid(carded))["score"] == \
        bare.evaluate(_mid(bare))["score"]


def test_the_v3_shell_reaches_the_family_and_builds_it():
    from gui.v3 import config, session

    S = session.make_session("track")
    S["wing"]["choices"]["car_two_element"] = True
    session.apply_choices(S)
    assert S["wing"]["problem"] == f"{MULTI} + free chord law"
    flags = config.flags(S)
    api.check_flags(S["wing"]["problem"], flags)
    built = config.spec(S).build({}, flags, None)
    assert "flap_chord_frac" in built.param_labels
    out = built.evaluate(_mid(built))
    assert out["feasible"], out["reason"]


def test_the_card_does_not_offer_the_switch_and_the_family_still_flies():
    """THE ELEMENTS SWITCH LEFT THE CARD, and the family did not leave the
    registry.

    A slotted wing is a different solver family and four more design rows,
    and it is still registered, still derivable and still flown by anything
    that sets the choice (``_car_is_two_element`` reads it, and the tandem
    switch on the track still means "a slot"). What went is the QUESTION: the
    track card asks the mount, the size, the chord law and what to maximise,
    and nothing else.
    """
    from gui.v3.app import assemble

    ctx = assemble("track")
    ctx.render("wing", "type")
    texts = " ".join(_texts(ctx.views[("wing", "type")]))
    assert "two elements (slotted flap)" not in texts
    assert "Elements" not in texts

    # ...and the family is reachable and flyable exactly as before
    ch = dict(v1.start_choices(medium="track"), car_two_element=True)
    v1.normalise_choices(ch)
    name = v1.derive_problem(ch)[0]
    assert name.startswith(MULTI)
    built = api.PROBLEM_SPECS[MULTI].build({}, v1.car_flags(ch, None, MULTI),
                                           None)
    assert "flap_chord_frac" in built.param_labels
    assert np.isfinite(built.evaluate(_mid(built))["score"])



def _texts(view) -> list[str]:
    out = []
    for e in view.descendants():
        t = getattr(e, "text", "") or ""
        if t:
            out.append(t)
        opts = getattr(e, "options", None)
        if isinstance(opts, dict):
            out += [str(v) for v in opts.values()]
        elif isinstance(opts, (list, tuple)):
            out += [str(v) for v in opts]
    return out


def test_the_slot_is_presentable_on_the_results_page():
    """A breakdown key nothing can render is a run whose answer is invisible.

    The family is detected off the BREAKDOWN, not the problem name, and the
    slot's own share key must not collide with the tandem pair's — otherwise
    a two-element car wing would be reported as a tandem.
    """
    from gui import metrics

    built = api.PROBLEM_SPECS[MULTI].build({}, {}, None)
    out = built.evaluate(_mid(built))
    assert out["feasible"], out["reason"]
    assert metrics.family(out) == "carwing"
    assert "lift_share_front" not in out          # the tandem marker
    shown = set()
    for group in metrics.groups(out):
        shown |= {m["key"] for m in group[1]}
    for key in ("n_elements", "flap_chord_frac", "flap_deflection_deg",
                "slot_gap_frac", "slot_overlap_frac", "slot_width_frac",
                # the wind tunnel's gap, beside the other two lengths: all
                # three differ and a bound taken from a paper is stated
                # against THIS one
                "slot_gap_te_frac",
                "lift_share_main", "stall_margin_main", "stall_margin_flap",
                "cp_min_main",
                # ONE ceiling per element, because the flap can now be a
                # different shape; plus where the pair came from, because
                # the NACA-24XX substitution is this model's largest single
                # assumption and a reader of a result must be able to see
                # which of the two paths produced it
                "cp_ceiling_main", "cp_ceiling_flap",
                "tc_flap", "Re_main", "Re_flap"):
        assert key in shown, key
    # ...and the two STRINGS that say which stall data produced those
    # ceilings are in the breakdown rather than on the metrics page, because
    # that catalogue is numeric by construction. They still have to be
    # somewhere a reader can find them: the substitution is the model's
    # largest single assumption.
    assert out["ceiling_source"] == "naca24xx-substitution"
    assert out["ceiling_section"] == "NACA2412 (wide-alpha XFOIL, Re 1e6)"


def test_the_lap_and_the_windup_are_presentable_too():
    from gui import metrics

    # dCZ/dh is a DIAGNOSTIC and is off by default (it costs two more solves
    # in every optimiser's inner loop), so this asks for it. Asking must not
    # change the search: test_asking_for_the_sensitivity_is_not_a_constraint
    # below is the gate on that, and it is the reason the report flag exists
    # separately from the ceiling.
    built = api.PROBLEM_SPECS[SINGLE].build(
        {}, {"car_track": "synthetic", "car_objective": "laptime",
             "car_mount_model": "continuum", "x_attach_frac": 0.35,
             "dczdh_report": True}, None)
    out = built.evaluate(_mid(built))
    assert out["feasible"], out["reason"]
    shown = set()
    for group in metrics.groups(out):
        shown |= {m["key"] for m in group[1]}
    for key in ("lap_time_s", "dCZ_dh_per_m", "windup_root_deg",
                "pylon_length_m"):
        assert key in shown, key


# ------------------------------ 5. the box quoted IS the box searched (A)

def _track_session(bounds=None, **choices):
    from gui.v3 import session

    S = session.make_session("track")
    S["wing"]["choices"].update(choices)
    session.apply_choices(S)
    S["wing"]["bounds"].update(bounds or {})
    return S


def test_the_slotted_box_the_card_quotes_is_the_box_the_run_searches():
    from gui.v3 import config

    S = _track_session(bounds={"b_m": [1.3, 1.5], "S_m2": [0.2, 0.3]},
                       car_two_element=True)
    shown = {k: v[0] for k, v in config.effective_bounds(S).items()}
    built = config.spec(S).build({}, config.flags(S),
                                 config.bounds_overrides(S))
    labels = list(built.param_labels)
    assert set(shown) == set(labels)
    for key in labels:
        lo, hi = built.bounds[labels.index(key)]
        assert shown[key] == [pytest.approx(lo), pytest.approx(hi)], key
    assert shown["b_m"] == [1.3, 1.5]
    assert shown["S_m2"] == [0.2, 0.3]


def test_the_size_rows_are_the_only_place_the_size_is_stated():
    """THE DESIGN BOX OWNS THE SIZE BANDS.

    Both halves of the defect this closed, asserted as outcomes:

    * the box row now travels to the PROBLEM, not only to the sampler.
      ``api.rows_outside_validity`` is the measurement — a widened row that
      does not travel is drawn from and then refused with "bounds violation",
      score -100, so a user who widened ``b_m`` got a run made entirely of
      refusals. The ``taper`` row is the control: it is still one of the rows
      that does NOT travel, so an empty list here cannot be the helper
      failing to see anything.
    * the same band can no longer be stated a second time as a FLAG, which
      is what let the run search one band while the card showed another.
    """
    wide = {"b_m": (1.2, 3.0), "S_m2": (0.05, 0.60)}
    for name in CAR_BASES:
        built = api.PROBLEM_SPECS[name].build({}, {}, wide)
        assert api.rows_outside_validity(built) == [], name
        assert built.problem.span_bounds_m == wide["b_m"], name
        assert built.problem.area_bounds_m2 == wide["S_m2"], name
        control = api.PROBLEM_SPECS[name].build({}, {}, {"taper": (0.05, 1.0)})
        assert [r["label"] for r in api.rows_outside_validity(control)] \
            == ["taper"], name
        for key in DEAD_SIZE_FLAGS:
            with pytest.raises(KeyError, match=key):
                api.check_flags(name, {key: 1.3})

    # ...and the design a widened row reaches is a real one, not a refusal:
    # a span above the family's published ceiling is FLOWN and reported
    from aerobo import carwing
    built = api.PROBLEM_SPECS[SINGLE].build({}, {}, {"b_m": (1.2, 3.0)})
    labels = list(built.param_labels)
    x = _mid(built)
    x[labels.index("b_m")] = 2.6
    assert 2.6 > carwing.SPAN_BOUNDS_M[1]
    out = built.evaluate(x)
    assert out["feasible"], out["reason"]
    assert out["b_m"] == pytest.approx(2.6)
    assert out["score"] > -100.0


def test_the_size_band_is_idempotent_under_rebuild():
    """A CONTRACT, not an accident. ``gui.v3.relax`` clips a widened row back
    to the family's validated band, rebuilds ONCE and asserts the clip
    converged — so a helper that re-derived or re-clamped the band it was
    handed would leave the row outside validity on the second pass too, and
    the whole no-solution "reach" card would abort for the car with "this
    family re-derives its own bounds from the ones it is handed".
    """
    from gui.v3 import relax

    wide = {"b_m": (1.2, 3.0), "S_m2": (0.05, 0.60)}
    for name in CAR_BASES:
        spec = api.PROBLEM_SPECS[name]
        # WIDER than the family's published bands, so a helper that clamped
        # or re-derived what it was handed would show up here
        first = spec.build({}, {}, dict(wide))
        labels = list(first.param_labels)
        assert first.problem.span_bounds_m == wide["b_m"], name
        assert first.problem.area_bounds_m2 == wide["S_m2"], name
        # feed the built box straight back in: nothing may move
        again = spec.build({}, {}, {k: (float(first.bounds[i, 0]),
                                        float(first.bounds[i, 1]))
                                    for i, k in enumerate(labels)})
        assert np.allclose(again.bounds, first.bounds), name
        assert again.problem.span_bounds_m == first.problem.span_bounds_m
        assert again.problem.area_bounds_m2 == first.problem.area_bounds_m2

        # ...and the shell's own clip converges in ONE pass on a box that
        # widens a size row (which travels) beside a row that does NOT
        band = {k: [float(first.bounds[i, 0]), float(first.bounds[i, 1])]
                for i, k in enumerate(labels)}
        band["taper"] = [0.05, 1.0]
        clipped, capped, err = relax.clip_to_validity(spec, {}, {}, dict(band))
        assert err is None, (name, err)
        # the row that cannot travel was put back; the size rows the user
        # widened survived the clip, so the reach card can still offer them
        assert [c["label"] for c in capped] == ["taper"], name
        assert clipped["taper"] != band["taper"], name
        assert clipped["b_m"] == list(wide["b_m"]), name
        assert clipped["S_m2"] == list(wide["S_m2"]), name


def test_the_car_limits_live_under_the_design_box_and_are_optional():
    """NOTHING IS BUDGETED BY DEFAULT, in the shell as well as in the api.

    The three-way drag MENU is gone — a card that always sends a budget is a
    card that always picks the answer — and what is left is two blank fields
    under the design box, in newtons. Blank means no constraint at all, not a
    constraint set to zero.
    """
    from gui.v3 import config
    from gui.v3.stages import wing as wing_stage

    S = _track_session(car_two_element=True)
    assert [r[0] for r in wing_stage.CAR_LIMIT_ROWS] == ["drag_budget_n",
                                                         "downforce_min_n"]
    for key, *_ in wing_stage.CAR_LIMIT_ROWS:
        assert key in config.spec(S).flags, key
    # nothing typed: no budget flag leaves the shell, and the run carries one
    # margin — the deflection limit the problem declares unconditionally
    flags = config.flags(S)
    assert "drag_budget_n" not in flags and "CD_budget" not in flags
    plain = config.spec(S).build({}, flags, None)
    assert plain.problem.constraint_labels == ("deflection margin",)

    # ...and a typed floor is a real, extra margin the answer has to satisfy
    S["wing"]["flags"]["downforce_min_n"] = 200.0
    floored = config.spec(S).build({}, config.flags(S), None)
    assert floored.problem.downforce_min_n == 200.0
    assert floored.problem.constraint_labels == ("deflection margin",
                                                 "downforce floor margin")
    x = _mid(plain)
    assert len(floored.callable(x)[1]) == len(plain.callable(x)[1]) + 1


# ------------------------------------------- the car's SPEED reaches it (B)

def test_the_mission_speed_reaches_the_car_solver():
    """``car_flags`` never emitted ``V``, so every car run flew
    ``carwing.CarWingProblem.V`` = 55 whatever the mission card said — while
    the family DECLARES a V flag and refuses a mission spec outright, so
    there was no other way in.
    """
    default = v1.car_default_v_ms()
    assert default == 55.0                      # the field, not a constant
    ch = v1.start_choices(medium="track")
    assert "V" not in v1.car_flags(ch)                     # untouched card
    assert "V" not in v1.car_flags(ch, default)            # ...and at 55 m/s
    assert v1.car_flags(ch, 40.0)["V"] == 40.0
    # never to a family that does not declare it
    assert v1.car_flags(dict(ch, medium="air"), 40.0) == {}
    for name in CAR_BASES:
        assert "V" in api.PROBLEM_SPECS[name].flags, name


def test_the_mission_card_does_not_call_the_car_s_speed_unhonoured():
    """The same defect one card along: the disabled V field said "not
    honoured by this solver", which is false — it is honoured, as a flag."""
    for name in (SINGLE, MULTI):
        note = v1.mission_field_note(name, "V")
        assert "not honoured" not in note, (name, note)
        assert "flag" in note, (name, note)


def test_a_stated_speed_changes_what_the_car_flies():
    """The OUTCOME, not the flag: a slower car makes less downforce."""
    from gui.v3 import config

    S = _track_session()
    fast = config.spec(S).build({}, config.flags(S), None)
    assert fast.problem.V == v1.car_default_v_ms()
    S["mission"]["V"] = 40.0
    flags = config.flags(S)
    assert flags["V"] == 40.0
    slow = config.spec(S).build({}, flags, None)
    assert slow.problem.V == 40.0
    x = _mid(fast)
    a, b = fast.evaluate(x), slow.evaluate(x)
    assert a["feasible"] and b["feasible"]
    assert b["downforce_N"] < a["downforce_N"]
    assert b["downforce_N"] / a["downforce_N"] == pytest.approx(
        (40.0 / 55.0) ** 2, rel=2e-2)


# ------------------------------------- what a car's size is spent on (C)

def test_the_car_size_card_does_not_price_the_wing_in_payload():
    """The size hint ended "the score is payload L/D and a root-bending
    stress margin is a constraint" — true of the free-planform AIRCRAFT whose
    branch it shares, false of a car, which is scored on downforce (or CZ/CD)
    and constrained by the deflection limit and whatever drag ceiling the
    user chose to state.

    It also has to say that BOTH dimensions are designed now, not just the
    span: the area stopped being a problem-name choice and became a row.

    The "no traceback" half of this test lives in
    ``test_the_car_card_renders_without_a_traceback`` below, because the card
    currently raises after this hint is drawn.
    """
    from gui.v3.app import assemble

    ctx = assemble("track")
    ctx.render("wing", "type")
    texts = " ".join(_texts(ctx.views[("wing", "type")]))
    # the COUNT is the tandem's, not this wing's: a single-span car has two
    # dimensions and says "both". Asserted on the claim rather than on the
    # quantifier, so the grammar can follow the family it is drawn for.
    assert "the span and the area are both design variables" in texts, texts
    assert "rows of the design box" in texts, texts
    assert "payload L/D" not in texts, texts
    assert "root-bending stress" not in texts, texts
    assert "DRAG budget and the DEFLECTION limit" in texts, texts


def test_the_car_card_renders_without_a_traceback(capsys):
    """A view that dies half way through is a card whose bottom half is not
    on screen — here, everything from the "Maximise" row down: the objective,
    the slotted-flap switch and the designed-endplate switch."""
    from gui.v3.app import assemble

    ctx = assemble("track")
    ctx.render("wing", "type")
    assert "Traceback" not in capsys.readouterr().err


def test_the_aircraft_size_card_still_prices_it_in_payload(capsys):
    """The control: the sentence belongs to the family it was written for."""
    from gui.v3.app import assemble

    ctx = assemble("air")
    ctx.S["wing"]["choices"]["planform"] = "aircraft"
    from gui.v3 import session
    session.apply_choices(ctx.S)
    if "free planform" not in ctx.S["wing"]["problem"]:
        pytest.skip("V3 does not offer the free-planform family here")
    ctx.render("wing", "type")
    texts = " ".join(_texts(ctx.views[("wing", "type")]))
    assert "payload L/D" in texts, texts
    assert "Traceback" not in capsys.readouterr().err


def test_asking_for_the_sensitivity_is_not_a_constraint():
    """A diagnostic must not move the feasible set.

    dCZ/dh can be asked for two ways and they are deliberately different
    questions: ``dczdh_report`` REPORTS the number, ``dczdh_max_per_m``
    CONSTRAINS it. Only the second may add a margin. Without the split the
    only way to see the number would be to impose a limit on it, which is
    this repo's rule ("answering one more question must never widen a
    search") exactly inverted.

    Mutation this is designed to catch: routing ``dczdh_report`` to
    ``dczdh_max_per_m``, or making the report imply the ceiling.
    """
    plain = api.PROBLEM_SPECS[SINGLE].build({}, {}, None)
    asked = api.PROBLEM_SPECS[SINGLE].build({}, {"dczdh_report": True}, None)
    capped = api.PROBLEM_SPECS[SINGLE].build(
        {}, {"dczdh_max_per_m": 0.5}, None)
    x = _mid(plain)
    a, b, c = plain.evaluate(x), asked.evaluate(x), capped.evaluate(x)

    # the number appears only when it is asked for, and is ABSENT rather than
    # None when it is not -- a reader cannot mistake a silence for a zero
    assert "dCZ_dh_per_m" not in a
    assert isinstance(b["dCZ_dh_per_m"], float)

    # ...and asking changed NOTHING else: same score, same margins, same width
    assert b["score"] == a["score"]
    assert list(b["g"]) == list(a["g"])
    assert plain.problem.n_constraints == asked.problem.n_constraints

    # the CEILING, by contrast, is a constraint and says so
    assert capped.problem.n_constraints == plain.problem.n_constraints + 1
    assert len(c["g"]) == len(a["g"]) + 1
