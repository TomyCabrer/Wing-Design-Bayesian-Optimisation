"""The car's OPERATING POINT, and its MOUNT — the two things a rear wing has
that this shell was asking as though it were an aeroplane.

Two gaps, both measurable from the code's own side before a line of UI was
touched:

1. **Stage 1 asked a car four numbers the car has no answer to.**
   ``api._make_car_wing_builder`` refuses a mission spec outright ("weight and
   altitude do not enter"), ``carwing.CarWingProblem.rho`` is the module
   constant ``RHO_AIR``, and both of the wing's dimensions are ``b_m`` /
   ``S_m2`` rows of the design box. So of the six fields the aircraft mission
   card put on screen — a design weight, a speed, an altitude, a wing loading,
   a reference area and a constraint diagram — the track problem reads exactly
   ONE, the speed. The other five existed so that stage 2 could back a lift
   coefficient out of ``CL = W/(qS)``.

   The coefficient is asked directly now (``session.track_design_cz``), and
   the chord it is screened at comes from the design box's own two rows. The
   stored ``W_N`` / ``s_ref_m2`` / ``altitude_m`` are a MIRROR of those
   answers (``session.track_point_sync``) so nothing downstream has to learn a
   second vocabulary.

   The tests below assert the OUTCOME, not the wiring: a stated CZ IS the
   coefficient stage 2 screens at; it does NOT move when the speed moves; an
   altitude cannot move the density the section is designed against, because
   the solver's is fixed; and narrowing the design box's area row DOES move
   the Reynolds number, which is the mutation that proves the chord really
   comes from the box.

2. **Every mount reachable from any shell was one of ``carwing.MOUNTS``' two.**
   ``carmount.py`` exists to take the mount apart into a station, a kind, a
   side and a chordwise attachment — its own module docstring names the case
   ("a mount 35% out along the semi-span ... is not an exotic layout") — but
   ``api._CAR_MOUNT_REFINE_KEYS`` declared four scalars and no station, so the
   continuum's endpoints were the only points on it. The station, the kind,
   the side and the pylon count are declared now, and the card asks them.

   The load-bearing assertion is the one about the FREE LUNCH: moving the
   station must re-count the sheets and the pylons that mount costs, never
   inherit them. ``carmount.MountSpec.n_sheets`` exists because the continuum
   path once handed the inboard grip its stiffness for nothing, and a station
   flag would re-open that hole one step further out if the count were not
   derived.

   The CARD, though, asks two things and not eleven. Declaring the continuum
   in ``api`` and exposing it field-by-field are different decisions: the
   first makes a mount statable, the second made the reader design one before
   they could choose one. So the shell asks WHERE THE LOAD LEAVES THE WING —
   through the plates at the tips, or down swan-neck struts near the middle —
   and derives every ``MountSpec`` field each answer implies
   (``v1.car_mount_layout_flags``). The api tests above are unchanged and
   still cover the whole continuum, because none of it was withdrawn.

   ...and that one answer decides the rest of the card. A plate that carries
   the car is a designed structure — chord, thickness, toe, and a constraint
   that it reach the deck — so ``tips`` IS the designed-endplate family and
   asks the two things such a plate is built from (its section, its root
   blend). A plate on a pylon-borne wing is a fence, which is exactly a tip
   DEVICE, so ``pylons`` asks its shape in the aircraft menu's own four keys.
   The mount is therefore stored in one key and one key only
   (``car_endplates``), it is the first control on the card, and neither the
   old "Endplates — design them" switch nor a second layout key survives it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aerobo import api, carmount, carwing                     # noqa: E402
from gui import nice_app as v1                                # noqa: E402
from gui.v3 import config, session                            # noqa: E402

SINGLE = "car rear wing"


def _mid(built) -> np.ndarray:
    return 0.5 * (built.bounds[:, 0] + built.bounds[:, 1])


def _evaluate(flags: dict, name: str = SINGLE) -> dict:
    built = api.PROBLEM_SPECS[name].build({}, flags, None)
    return built.evaluate(_mid(built))


def _track_session() -> dict:
    S = session.make_session()
    S["medium"] = "track"
    S["wing"]["choices"]["medium"] = "track"
    session.apply_choices(S)
    return S


def _field_labels(view) -> list[str]:
    """The labels of the CONTROLS on a view — ``widgets.number_field`` and
    ``select_field`` both stamp ``field-label`` on theirs.

    Asserted on instead of on every string in the view, because the track
    card explains at length that a car HAS no wing loading, and a plain text
    search cannot tell a field from the sentence saying there is no field.
    """
    out = []
    for e in view.descendants():
        if "field-label" in (getattr(e, "_classes", None) or []):
            t = getattr(e, "text", "") or ""
            if t:
                out.append(str(t).strip().lower())
    return out


def _texts(view) -> list[str]:
    out = []
    for e in view.descendants():
        t = getattr(e, "text", "") or ""
        if t:
            out.append(str(t))
        label = (getattr(e, "_props", {}) or {}).get("label")
        if label:
            out.append(str(label))
        opts = getattr(e, "options", None)
        if isinstance(opts, dict):
            out += [str(x) for x in opts.values()]
        elif isinstance(opts, (list, tuple)):
            out += [str(x) for x in opts]
    return out


# =========================================== 1. what the car actually reads

def test_the_car_family_reads_one_number_off_the_operating_point():
    """The premise of the whole card, taken from the registry and the solver.

    Catches someone re-adding a weight or an altitude field on the strength of
    "the mission asks it everywhere else": if this ever stops holding, the
    track card should grow those fields back, and the test says so.
    """
    for name in (SINGLE, f"{SINGLE} + free chord law"):
        spec = api.PROBLEM_SPECS[name]
        assert spec.mission_fields == (), name
        assert not spec.uses_mission, name
        with pytest.raises(ValueError, match="no mission spec"):
            spec.build({"W_N": 500.0}, {}, None)
    # ...and the speed is a FLAG on every registered track family
    assert "V" in api._CAR_CORE_KEYS
    # the density is the module constant, not an altitude
    assert carwing.CarWingProblem.rho == carwing.RHO_AIR


def test_the_stated_cz_is_the_coefficient_stage_2_screens_at():
    """Typed CZ -> ``design_point['cl_design']``, exactly and by identity.

    The number stage 2 designs its aerofoil at is the number on the card, not
    a coefficient reconstructed from an invented weight and an invented area.
    """
    S = _track_session()
    assert session.design_point(S)["cl_design"] == pytest.approx(
        session.REFERENCE_CL, rel=1e-12)
    for cz in (0.6, 1.0, 1.45, 2.2):
        assert session.set_track_design_cz(S, cz)
        assert session.design_point(S)["cl_design"] == pytest.approx(
            cz, rel=1e-12)


def test_the_speed_moves_the_reynolds_number_and_not_the_coefficient():
    """A car holds no weight, so its lift coefficient does not follow its
    speed. The aircraft card's CL = W/(qS) does exactly the opposite, and
    that difference is the reason the track card is its own view.

    Mutation: without the mirror being rewritten on a speed edit the stored
    weight would stay put and CZ would fall as V², which is the aeroplane's
    answer to a question the car did not ask.
    """
    S = _track_session()
    session.set_track_design_cz(S, 1.3)
    before = session.design_point(S)
    S["mission"]["V"] = 2.0 * float(S["mission"]["V"])
    session.sync_wing_from_mission(S)
    after = session.design_point(S)
    assert after["cl_design"] == pytest.approx(1.3, rel=1e-12)
    assert after["re_mac"] == pytest.approx(2.0 * before["re_mac"], rel=1e-9)
    assert after["q"] == pytest.approx(4.0 * before["q"], rel=1e-9)


def test_an_altitude_cannot_move_the_density_the_section_is_designed_against():
    """The defect this removes, stated as a measurement.

    ``carwing`` flies RHO_AIR whatever a card says. With an altitude field on
    the track mission a user could design a section against 1.112 kg/m³ (1000
    m) while stage 3 flew 1.225 — a 9 % error in q and 5 % in the section
    Reynolds number, with nothing on screen saying so.
    """
    S = _track_session()
    S["mission"]["altitude_m"] = 1000.0
    dp = session.design_point(S)
    assert dp["rho"] == pytest.approx(carwing.RHO_AIR, rel=1e-5)
    assert float(S["mission"]["altitude_m"]) == 0.0


def test_the_section_chord_follows_the_design_box_and_not_a_guess():
    """THE MUTATION. Narrow the ``S_m2`` row and the Reynolds number the
    section is screened at has to follow it, because a rear wing's chord is
    its own design box's and not an aspect-ratio estimate.

    Before this, stage 1 turned a wing loading into an area and stage 2
    guessed an aspect ratio, so a user who bounded the wing to a quarter of
    its default area still screened the section at the old chord.
    """
    S = _track_session()
    wide = session.design_point(S)
    box = config.effective_bounds(S)
    assert "S_m2" in box and "b_m" in box
    S["wing"]["bounds"]["S_m2"] = [0.10, 0.14]      # as the box card writes it
    narrow = session.design_point(S)
    assert narrow["s_ref_m2"] == pytest.approx(0.12, rel=1e-12)
    # the chord shrinks with the area at fixed span, so the Reynolds number
    # falls — and by the ratio of the mean chords, exactly
    assert narrow["re_mac"] < wide["re_mac"]
    assert narrow["re_mac"] / wide["re_mac"] == pytest.approx(
        narrow["mac"] / wide["mac"], rel=1e-9)
    assert session.section_aspect_ratio(S) == pytest.approx(
        narrow["b"] ** 2 / narrow["s_ref_m2"], rel=1e-9)


# ============================================== 2. what the track card asks

def test_the_track_operating_card_asks_the_two_numbers_and_no_others():
    """The card itself: speed and CZ in, weight / altitude / wing loading /
    constraint diagram out.

    Asserted on the RENDERED view rather than on the source, so a field
    re-added through any branch is caught.
    """
    from gui.v3.app import assemble

    ctx = assemble("track")
    ctx.select("mission", "operating")
    fields = _field_labels(ctx.views[("mission", "operating")])
    assert "speed" in fields
    assert "reference cz" in fields
    for gone in ("design weight", "design downforce", "design lift",
                 "wing loading w/s", "altitude", "depth",
                 "reference area s", "span (sketch)",
                 "aspect ratio (estimate)", "stall / approach speed",
                 "t/w available (closes the diagram)", "cd0", "climb rate"):
        assert gone not in fields, gone
    # ...and TWO typed numbers on the whole card, not six
    from nicegui import ui as _ui
    numbers = [e for e in ctx.views[("mission", "operating")].descendants()
               if isinstance(e, _ui.number)]
    assert len(numbers) == 2, [(e._props or {}) for e in numbers]


def test_the_reset_returns_the_car_to_its_own_point_and_says_so():
    """“Published defaults” on the track resets the two numbers the card
    states — and takes its own branch, because the aircraft one quotes a
    ``w_source`` about a design load this card does not show and an
    aspect-ratio estimate nobody owns here.
    """
    from gui.v3.app import assemble

    ctx = assemble("track")
    ctx.act("set_track_cz", 1.7)
    assert session.design_point(ctx.S)["cl_design"] == pytest.approx(1.7)
    ctx.act("reset_mission")
    assert session.track_design_cz(ctx.S) == pytest.approx(
        session.REFERENCE_CL)
    assert float(ctx.S["mission"]["altitude_m"]) == 0.0
    # ...and the air branch is untouched by the split
    air = assemble("air")
    air.act("reset_mission")
    assert air.S["mission"]["w_source"]


def test_the_air_session_still_asks_all_of_them():
    """...and the removals are the TRACK's, not the mission stage's.

    Without this the previous test passes just as well on a card that asks
    nothing at all.
    """
    from gui.v3.app import assemble

    ctx = assemble("air")
    ctx.select("mission", "operating")
    fields = _field_labels(ctx.views[("mission", "operating")])
    for kept in ("design weight", "speed", "altitude", "wing loading w/s"):
        assert kept in fields, (kept, fields)


def test_the_track_wing_card_states_the_size_once_and_offers_no_dead_menus():
    """Stage 3's Wing type card, for a car.

    Three things it must not do, each of which it did:

    * offer a PLANFORM menu whose one entry reads "fixed span + area (you
      choose the size)" on a family that searches both — a label that
      contradicted the hint one line below it;
    * offer a TIP DEVICE menu holding "none" on a wing whose endplate height
      is a design variable and whose plates are the load path;
    * state "both dimensions are designed, their bands are design-box rows"
      twice on one screen (once under the planform row, once under the mount).
    """
    from gui.v3.app import assemble

    ctx = assemble("track")
    ctx.select("wing", "type")
    texts = _texts(ctx.views[("wing", "type")])
    blob = " ".join(texts).lower()
    assert "fixed span + area" not in blob
    assert "span and area are searched" in blob
    # the card OPENS on the endplate mount, where the plates ARE the load
    # path — so there is no tip-device row at all, and the two things a
    # load-bearing plate is built from are asked under the Mount instead
    assert "endplates — always fitted" not in blob
    assert "tip device" not in blob
    assert v1.CAR_ENDPLATE_SECTION_LABELS["shaped"].lower() in blob
    assert "root blend" in blob
    # the size paragraph appears once, not once per shell convention
    assert sum("their bands are rows of the design box" in t.lower()
               for t in texts) <= 1
    assert "bands are rows of the design box (s_m2 and b_m)" not in blob
    # ...and no row that sends the user to a stage-1 switch that is disabled
    # when they get there
    assert "lifting system" not in blob
    # the derived panel must not claim a planform this run does not fly
    assert "planform searched" in blob
    assert "planform flown" not in blob


def test_the_air_wing_card_keeps_its_menus():
    """The counterpart: the removals above are the car's alone."""
    from gui.v3.app import assemble

    ctx = assemble("air")
    ctx.select("wing", "type")
    blob = " ".join(_texts(ctx.views[("wing", "type")])).lower()
    assert "tip device" in blob
    assert "planform" in blob
    assert "lifting system" in blob


# ==================================================== 3. more mounts, honestly

def test_the_registry_declares_the_station_the_kind_and_the_side():
    """The gap, from the registry's side: before this, ``flags`` could refine
    four scalars of a MountSpec and could not say WHERE the mount was."""
    for key in ("mount_station_frac", "mount_station_m", "mount_kind",
                "mount_side", "mount_n_pylons"):
        assert key in api._CAR_MOUNT_REFINE_KEYS, key
        assert key in api.accepted_flags(SINGLE), key
    # ...and only on the family that HAS a mount_spec field
    for name in ("car rear wing (two-element)", "car rear wing + endplates"):
        assert "mount_station_frac" not in api.accepted_flags(name), name


def test_the_continuum_reproduces_both_published_layouts_exactly():
    """Adoption must move nothing. Bit-for-bit, not "close"."""
    for layout in carwing.MOUNTS:
        published = _evaluate({"mount": layout})
        continuum = _evaluate({"mount": layout,
                               "car_mount_model": "continuum"})
        assert continuum["score"] == published["score"], layout
        assert continuum["CD"] == published["CD"], layout
        assert continuum["deflection_m"] == published["deflection_m"], layout


def test_a_station_stated_as_a_number_reproduces_the_named_layout():
    """0.35 IS "inboard" — so the continuum contains the lookup rather than
    running beside it, sheets and corners included."""
    named = _evaluate({"mount": "inboard"})
    stated = _evaluate({"car_mount_model": "continuum",
                        "mount_station_frac": carwing.INBOARD_STATION_FRAC})
    assert stated["score"] == named["score"]
    assert stated["CD"] == named["CD"]


def test_moving_the_station_re_counts_what_the_mount_costs():
    """THE FREE LUNCH, closed.

    An endplate grip inboard of the tip is a SECOND PAIR of plates. Inherit
    the tip layout's ``n_sheets = 0`` and the optimiser is handed a 25x
    stiffer wing for no drag at all — which is exactly the defect
    ``MountSpec.n_sheets`` was added for, and a station flag re-opens it
    unless the count is derived from the station and the kind.
    """
    def spec(**flags) -> carmount.MountSpec:
        return api._car_mount_kwargs(
            {"car_mount_model": "continuum", **flags})["mount_spec"]

    assert spec().n_sheets == 0                       # at the tips: free
    assert spec(mount_station_frac=1.0).n_sheets == 0
    for frac in (0.0, 0.2, 0.35, 0.5, 0.9):
        assert spec(mount_station_frac=frac).n_sheets == 2, frac
    # a LENGTH is never "at the tip" — the span is a design variable, so no
    # length is the tip for every candidate
    assert spec(mount_station_m=0.8).n_sheets == 2
    assert spec(mount_station_m=0.8).station_frac is None

    tips = _evaluate({"car_mount_model": "continuum",
                      "mount_station_frac": 1.0})
    inboard = _evaluate({"car_mount_model": "continuum",
                         "mount_station_frac": 0.5})
    assert inboard["deflection_m"] < 0.1 * tips["deflection_m"]
    assert inboard["CD"] > tips["CD"]     # the stiffness is PAID for


def test_a_pylon_mount_carries_pylons_and_no_sheets():
    """The other kind, and the count that goes with it."""
    def spec(**flags) -> carmount.MountSpec:
        return api._car_mount_kwargs(
            {"car_mount_model": "continuum", **flags})["mount_spec"]

    got = spec(mount_kind="pylon", mount_station_frac=0.0)
    assert got.kind == "pylon"
    assert got.n_pylons == 2 and got.n_sheets == 0
    assert spec(mount_kind="pylon", mount_station_frac=0.0,
                mount_n_pylons=1).n_pylons == 1
    # a contradiction meets MountSpec's own refusal rather than a silent fix
    with pytest.raises(ValueError, match="no pylons to charge"):
        spec(mount_kind="endplate", mount_n_pylons=2)


def test_the_swan_neck_is_a_geometry_statement_and_the_loss_is_a_calibration():
    """``MOUNT_SIDES`` decides WHETHER a suction loss applies; its magnitude
    is the user's. So a swan neck needs no number to be better than an
    underslung pylon that has been given one, and an underslung pylon with
    the loss left at zero must be identical to it."""
    base = {"car_mount_model": "continuum", "mount_kind": "pylon",
            "mount_station_frac": 0.3}
    swan = _evaluate({**base, "mount_side": "pressure", "suction_loss": 0.08})
    under_off = _evaluate({**base, "mount_side": "suction"})
    under_on = _evaluate({**base, "mount_side": "suction",
                          "suction_loss": 0.08})
    assert swan["score"] == under_off["score"]     # the loss is OFF by default
    assert under_on["score"] < swan["score"]       # ...and it costs when on


def test_the_station_is_stated_exactly_once():
    """Two spellings of one question, refused rather than resolved."""
    with pytest.raises(ValueError, match="exactly once"):
        api._car_mount_kwargs({"car_mount_model": "continuum",
                               "mount_station_frac": 0.4,
                               "mount_station_m": 0.3})


def test_the_refinements_are_refused_without_a_spec_to_read_them():
    """Under ``published`` nothing reads a station, so asking for one is an
    error and not a silently dropped flag."""
    for key, value in (("mount_station_frac", 0.4), ("mount_kind", "pylon"),
                       ("mount_side", "pressure"), ("mount_n_pylons", 1)):
        with pytest.raises(ValueError, match="read by nothing"):
            api._car_mount_kwargs({key: value})


# ================================================ 4. the card asks two things

def test_the_mount_is_one_question_with_two_answers():
    """The card's whole mount block: one select, two entries.

    It used to be eleven controls — a mode, a load path, a spelling for the
    station, the station, a side, a pylon count, a pylon drag model, a deck
    height, a suction loss, a chordwise attachment and a torsional stiffness.
    Every one was read by ``carmount``; together they asked the reader to
    DESIGN a mount before they could CHOOSE one.
    """
    from nicegui import ui

    from gui.v3.app import assemble

    ctx = assemble("track")
    ctx.select("wing", "type")
    view = ctx.views[("wing", "type")]

    # the two answers reach the screen, in the card's own words
    assert set(v1.CAR_MOUNT_LAYOUT_LABELS) == {"tips", "pylons"}
    blob = " ".join(_texts(view)).lower()
    for label in v1.CAR_MOUNT_LAYOUT_LABELS.values():
        assert label.lower() in blob, label

    # ...and the block that draws them is ONE control. Counted off a
    # standalone render rather than searched for in the page text, because a
    # text search cannot tell a control from the sentence describing one —
    # and the removed fields were plain ui rows, which carry no field-label.
    def drawn(**ch_extra):
        ch = v1.start_choices(medium="track")
        ch.update(ch_extra)
        v1.normalise_choices(ch)
        with view:
            box = ui.column()
        with box:
            v1._car_mount_controls(ch, lambda key, value: None)
        got = list(box.descendants())
        return (sum(isinstance(e, ui.select) for e in got),
                sum(isinstance(e, ui.number) for e in got))

    # PYLONS: the mount is the whole block — one select, no numbers. The
    # plate is a tip device there and its shape is asked elsewhere.
    assert drawn(car_endplates=False) == (1, 0)
    # TIPS: the mount, plus exactly the three things a LOAD-BEARING plate is
    # built from — its section, its root blend and its CANT. Not a switch to
    # reveal them, and nothing else: the eleven MountSpec fields are still
    # gone. The cant is the third because a plate that carries the car can
    # lean and this family has always flown one; it used to be reachable only
    # through the tip-device menu, which correctly does not exist here, so the
    # question was withheld rather than answered elsewhere.
    assert drawn(car_endplates=True) == (2, 2)

    # the vocabularies those fields were asked in are gone with them; what
    # they described is still reachable through api, which is the point
    for gone in ("CAR_MOUNT_MODEL_LABELS", "CAR_MOUNT_STATION_LABELS",
                 "CAR_PYLON_DRAG_LABELS", "car_mount_kind_labels",
                 "car_mount_side_labels"):
        assert not hasattr(v1, gone), gone
    # ...and the answer is stored in ONE key. A second key holding the same
    # fact is how two controls come to disagree on screen, so the mount IS
    # `car_endplates` — the boolean that also picks the family, because a
    # plate that carries the car is a designed plate.
    assert "car_mount_layout" not in v1.BUILDER_DEFAULTS
    assert v1.car_mount_layout_of({"car_endplates": True}) == "tips"
    assert v1.car_mount_layout_of({"car_endplates": False}) == "pylons"
    assert set(api._CAR_MOUNT_REFINE_KEYS) >= {"mount_kind", "mount_side",
                                               "mount_station_frac"}


def test_the_tip_mount_designs_the_plate_and_sends_no_strut():
    """The layout the card opens on, and what it means.

    ``tips`` is not "the pylon answer minus its flags": it says the PLATES
    carry the car, and a plate carrying a car is a designed structure — its
    chord, its thickness and its toe are design rows. So it picks the
    designed-endplate family, and it sends no mount refinement at all,
    because there is no strut in it to describe.
    """
    ch = v1.start_choices(medium="track")
    v1.normalise_choices(ch)
    assert v1.car_mount_layout_of(ch) == "tips"
    assert v1.derive_problem(ch)[0].startswith("car rear wing + endplates")

    name = v1.derive_problem(ch)[0]
    flags = v1.car_flags(ch, v1.car_default_v_ms(), name)
    assert "car_mount_model" not in flags
    for key in api._CAR_MOUNT_REFINE_KEYS:
        assert key not in flags, key
    assert flags["mount"] == carwing.CarWingProblem.mount
    assert api.check_flags(name, flags) is None
    # ...and the two things a load-bearing plate is built from travel with it
    assert flags["section"] == "shaped"


def test_the_pylon_answer_is_a_swan_neck_near_the_middle():
    """What the second entry IS — derived from carmount's own numbers, not
    from four more fields on the card.

    The SIDE is the assertion that matters. A card with no loss knob can only
    honestly offer the strut whose zero loss is a GEOMETRY statement: an
    underslung pylon sits in the suction peak and nothing in this package
    prices that, so shipping it from a two-entry menu would quietly hand the
    run an unpriced advantage.
    """
    ch = v1.start_choices(medium="track")
    ch["car_endplates"] = False          # = the pylon mount
    v1.normalise_choices(ch)
    name = v1.derive_problem(ch)[0]
    flags = v1.car_flags(ch, 55.0, name)

    assert api.check_flags(name, flags) is None
    spec = api._car_mount_kwargs(flags)["mount_spec"]
    assert spec.kind == "pylon"
    assert spec.side == "pressure"
    assert spec.station_frac == pytest.approx(carmount.INBOARD_STATION_FRAC)
    # the load path is re-counted, never inherited: struts carry it, so the
    # plates are not charged a second pair of sheets for stiffness
    assert spec.n_pylons > 0 and spec.n_sheets == 0
    # and nothing else was invented — every remaining field is the module's
    # own default, read off the dataclass rather than pasted here
    for field in ("x_attach_frac", "gj_nm2", "deck_height_m", "suction_loss",
                  "pylon_drag_model"):
        assert getattr(spec, field) == getattr(carmount.MountSpec, field), \
            field


def test_the_two_answers_differ_in_the_load_path_and_it_costs():
    """Both layouts fly, and the pylon one buys stiffness with drag.

    Asserted as a DIRECTION rather than a magnitude: the struts are wetted
    area and two wing/strut corners, and holding the wing near the middle
    turns one long sagging span into two short ones.
    """
    ch = v1.start_choices(medium="track")
    v1.normalise_choices(ch)
    name = v1.derive_problem(ch)[0]

    # BOTH ANSWERS FLY, and they are two different problems — the mount picks
    # the family as well as the load path, because a plate that carries the
    # car is a designed structure.
    names = {}
    for layout in ("tips", "pylons"):
        ch["car_endplates"] = layout == "tips"
        v1.normalise_choices(ch)
        names[layout] = v1.derive_problem(ch)[0]
        flags = v1.car_flags(ch, v1.car_default_v_ms(), names[layout])
        assert api.check_flags(names[layout], flags) is None
        built = api.PROBLEM_SPECS[names[layout]].build({}, flags, None)
        assert np.isfinite(built.evaluate(_mid(built))["score"]), layout
    assert names["tips"] != names["pylons"]
    assert "endplates" in names["tips"]

    # ...and WHAT THE STRUTS COST is measured inside ONE family, against the
    # same wing with the same plate. Comparing the two cards' answers would
    # compare two problems with different design vectors and different wetted
    # plate — a difference in family reported as a difference in mount.
    ch["car_endplates"] = False
    v1.normalise_choices(ch)
    plain = v1.derive_problem(ch)[0]
    struts = api.PROBLEM_SPECS[plain].build(
        {}, v1.car_flags(ch, v1.car_default_v_ms(), plain), None)
    grip = api.PROBLEM_SPECS[plain].build({}, {}, None)
    a, b = struts.evaluate(_mid(struts)), grip.evaluate(_mid(grip))
    assert a["deflection_m"] < 0.5 * b["deflection_m"]
    assert a["CD"] > b["CD"]
    # the swan neck loses no lift, and says so rather than being silent
    assert a["mount_lift_knockdown"] == 0.0


def test_the_pylon_answer_is_offered_only_where_it_can_be_flown():
    """A layout a family cannot take must not be drawable OR sendable — the
    designed-endplate and two-element families carry no ``mount_spec``, and
    their card falls back to the two-valued lookup, both of whose entries are
    plate-borne."""
    for extra, expected in (({"car_endplates": False}, True),
                            ({"car_endplates": False,
                              "car_two_element": True}, False),
                            ({"car_endplates": True}, False)):
        ch = v1.start_choices(medium="track")
        ch.update(extra)
        v1.normalise_choices(ch)
        name = v1.derive_problem(ch)[0]
        assert v1.car_mount_continuum_available(ch, name) is expected, extra
        flags = v1.car_flags(ch, 55.0, name)
        assert ("car_mount_model" in flags) is expected, extra
        assert api.check_flags(name, flags) is None
        built = api.PROBLEM_SPECS[name].build({}, flags, None)
        assert np.isfinite(built.evaluate(_mid(built))["score"])


# ========================================== 5. the plate is a tip device now

def test_the_car_tip_device_is_a_menu_in_the_aircraft_s_own_keys():
    """Stage 3's tip-device row, for a car.

    It was a sentence — "endplates — always fitted" — which was true and
    useless: the plate has a SHAPE, and three of the four shapes the aircraft
    menu offers are things a rear wing's plates really do. Same four keys, so
    the two menus read alike.

    WHERE it is asked is the mount's answer, and it is the whole gate: under
    the PYLON mount the plate is a fence and has a shape; under the ENDPLATE
    mount the plate is the load path, its section and root blend are asked
    under the Mount itself, and a "shape" row would be the same question in a
    second vocabulary. So the row appears on one answer and not the other.
    """
    from nicegui import ui

    from gui.v3.app import assemble

    assert set(v1.CAR_TIP_SHAPE_LABELS) == set(v1.WINGLET_SHAPES)

    ctx = assemble("track")
    ctx.select("wing", "type")
    view = ctx.views[("wing", "type")]
    # OPENS on the endplate mount: no tip-device row anywhere
    assert "tip device" not in _field_labels(view)

    sel = [e for e in view.descendants()
           if isinstance(e, ui.select) and "tips" in (e.options or {})]
    assert len(sel) == 1
    sel[0].value = "pylons"              # fires the card's own handler
    view = ctx.views[("wing", "type")]
    assert "tip device" in _field_labels(view)
    blob = " ".join(_texts(view)).lower()
    assert "endplates — always fitted" not in blob
    # ...and it offers all four, because the family charges every one of them
    for label in v1.CAR_TIP_SHAPE_LABELS.values():
        assert label.lower() in blob, label
    ch = dict(ctx.S["wing"]["choices"])
    assert set(v1.car_tip_shapes(ch, ctx.S["wing"]["problem"])) \
        == set(v1.CAR_TIP_SHAPE_LABELS)


def test_the_vertical_plate_is_the_published_run_bit_for_bit():
    """The shape the card opens on states nothing and sends nothing."""
    ch = v1.start_choices(medium="track")
    v1.normalise_choices(ch)
    assert v1.car_tip_shape_key(ch) == "vertical"
    flags = v1.car_flags(ch, v1.car_default_v_ms())
    assert "endplate_cant_deg" not in flags
    assert "endplate_chord_follows" not in flags


def test_a_canted_plate_costs_span_instead_of_buying_it():
    """THE FREE LUNCH THIS FEATURE ALMOST SHIPPED.

    ``endplate_h_m`` is an arc length along the plate, so leaning the plate
    outboard projects it past the wing's tips. On ``carwing``/``carwing_multi``
    ``b_m`` is the WING's span, so nothing counts that projection: measured
    before the gate went in, cant 60 scored 39.79 against vertical's 38.28 and
    bought it by flying 1.725 m of wing inside a 1.6 m band — which on a car
    is a regulation or a piece of bodywork.

    ``endplate.CarWingEndplateProblem`` reports ``b_m`` as the OVERALL width
    and pays for the projection out of the wing's own span. So the cant is
    declared THERE and refused everywhere else, and the direction of the
    effect flips: leaning the plate now costs.
    """
    name = "car rear wing + endplates"
    out = {}
    for cant in (90.0, 70.0, 45.0):
        built = api.PROBLEM_SPECS[name].build(
            {}, {} if cant == 90.0 else {"endplate_cant_deg": cant}, None)
        out[cant] = built.evaluate(_mid(built))

    # the width the run is allowed does not move; the WING gives way
    widths = {c: r["overall_width_m"] for c, r in out.items()}
    assert len(set(round(w, 9) for w in widths.values())) == 1, widths
    assert out[45.0]["b_m"] < out[70.0]["b_m"] < out[90.0]["b_m"]
    assert out[45.0]["endplate_projection_m"] > \
        out[70.0]["endplate_projection_m"] > 0.0
    # ...so leaning the plate is a cost, not a discovery
    assert out[45.0]["score"] < out[70.0]["score"] < out[90.0]["score"]
    # and the vertical plate is still the published run, to the last bit
    plain = api.PROBLEM_SPECS[name].build({}, {}, None)
    assert out[90.0]["score"] == plain.evaluate(_mid(plain))["score"]


def test_every_car_family_charges_the_plate_s_outboard_reach():
    """THE FREE LUNCH, CLOSED WHERE IT WAS ACTUALLY OPEN.

    The cant used to be withheld from ``carwing``/``carwing_multi`` because
    their ``b_m`` was the WING's span, so a leaning plate flew past the band
    for nothing. Withholding it never closed the hole: ``blend_frac`` was a
    declared, settable flag on both families the whole time and went through
    the same gap — measured before this, blend 1.0 bought +12.3 % CZ (0.7116
    -> 0.7993) by flying 1.92 m of hardware inside a stated 1.6 m band.

    All three families read the span row as the OVERALL width now and take
    the plate's projection out of the wing's own span. So the cant is
    declared everywhere, and every shape COSTS what it reaches.
    """
    from aerobo import carwing as cw

    for name in ("car rear wing", "car rear wing (two-element)",
                 "car rear wing + endplates"):
        assert api.check_flags(name, {"endplate_cant_deg": 60.0}) is None, name

    x = [0.7, 0.0, -2.0, 6.0, 0.25, 0.30, 1.6]
    plain = cw.evaluate_car_wing(x, cw.CarWingProblem())
    for kw in ({"endplate_cant_deg": 45.0}, {"blend_frac": 1.0}):
        r = cw.evaluate_car_wing(x, cw.CarWingProblem(**kw))
        # the width the row states does not move; the WING gives way
        assert r["overall_width_m"] == plain["overall_width_m"] == 1.6, kw
        assert r["endplate_projection_m"] > 0.0, kw
        assert r["b_m"] < plain["b_m"], kw

    # ...and a vertical, unblended plate never enters that arithmetic at all
    assert plain["endplate_projection_m"] == 0.0
    assert plain["b_m"] == 1.6

    # the card can express the cant now, on BOTH mounts — asked as the canted
    # tip DEVICE's own number under the pylons...
    ch = v1.start_choices(medium="track")
    ch["car_endplates"] = False          # = the pylon mount
    ch["car_tip_shape"] = "canted"
    ch["car_endplate_cant_deg"] = 60.0
    v1.normalise_choices(ch)
    name = v1.derive_problem(ch)[0]
    assert v1.car_flags(ch, 55.0, name)["endplate_cant_deg"] == 60.0
    assert api.check_flags(name, v1.car_flags(ch, 55.0, name)) is None
    # ...and as a typed FIELD under the plates, where the plate is the mount.
    # The SHAPE menu is still absent there and that is still right — a mount
    # does not have a shape — but a cant is a dimension of a designed part,
    # not a shape, and withholding it withheld a real rear wing. One key
    # carries the answer through the switch, because it is one question.
    ch["car_endplates"] = True
    v1.normalise_choices(ch)
    name = v1.derive_problem(ch)[0]
    assert v1.car_tip_shapes(ch, name) == {}
    assert v1.car_flags(ch, 55.0, name)["endplate_cant_deg"] == 60.0
    assert api.check_flags(name, v1.car_flags(ch, 55.0, name)) is None
    # ...and an untouched card still sends none of it, so the published
    # upright plate is bit-for-bit what it always was
    clean = v1.start_choices(medium="track")
    v1.normalise_choices(clean)
    assert "endplate_cant_deg" not in v1.car_flags(
        clean, 55.0, v1.derive_problem(clean)[0])


def test_a_leaning_plate_is_measured_leaning():
    """A plate that CARRIES the car was credited with the reach of a vertical
    one however far it leaned.

    ``endplate.py`` read the cant correctly everywhere it costs — the wing
    gives up span for the plate's outboard projection, and the lattice flies
    the leaning surface — and then computed the one constraint that says the
    wing is still ATTACHED to the car (``g_reach``) from a hard-coded 90. So
    the plate leaned in the picture, in the width and in the aerodynamics, and
    stood up straight in its own reach. Two more places read it as vertical:
    the cantilever arm (the gap it closes, not the length it spans) and the
    load that bends it (the y component of a force whose direction is the
    plate's own normal). All three errors point the same way — the design
    looks buildable when it is not.

    Asserted as OUTCOMES: what the numbers do as the plate leans, and the
    corner where a design that passed now fails.
    """
    name = "car rear wing + endplates"
    spec = api.PROBLEM_SPECS[name]
    labels = list(spec.build({}, {}, None).param_labels)

    def fly(x, cant):
        flags = {"endplate_cant_deg": cant}
        assert api.check_flags(name, flags) is None
        return spec.build({}, flags, None).evaluate(np.asarray(x, dtype=float))

    x = _mid(spec.build({}, {}, None))
    out = {c: fly(x, c) for c in (90.0, 75.0, 60.0)}

    # 1. THE REACH FALLS WITH THE LEAN. It used to be the same number at every
    #    cant, which is the whole defect.
    tip = [out[c]["endplate_tip_z_m"] for c in (90.0, 75.0, 60.0)]
    assert tip[0] > tip[1] > tip[2]
    #    ...and it falls by exactly the geometry: h sin(cant)
    h_ep = float(x[labels.index("endplate_h_m")])
    for c in (75.0, 60.0):
        assert out[c]["endplate_tip_z_m"] == pytest.approx(
            h_ep * np.sin(np.deg2rad(c)), rel=1e-9)
    #    ...so the margin that says the wing reaches the car falls with it
    assert (out[90.0]["g_reach"] > out[75.0]["g_reach"]
            > out[60.0]["g_reach"])

    # 2. THE MEMBER IS LONGER THAN THE GAP IT CLOSES, and the load that bends
    #    it is bigger than the side force it makes.
    for c in (75.0, 60.0):
        assert out[c]["endplate_arm_m"] == pytest.approx(
            out[90.0]["reach_m"] / np.sin(np.deg2rad(c)), rel=1e-6)
        assert (out[c]["endplate_normal_load_toe_N"]
                > abs(out[c]["endplate_side_load_toe_N"]))

    # 3. A VERTICAL PLATE IS UNTOUCHED, to the bit. Every published run on
    #    this family flies one, so none of the above may move any of them.
    plain = spec.build({}, {}, None).evaluate(np.asarray(x, dtype=float))
    assert out[90.0]["score"] == plain["score"]
    assert out[90.0]["endplate_arm_m"] == plain["reach_m"]
    assert (out[90.0]["endplate_normal_load_toe_N"]
            == abs(plain["endplate_side_load_toe_N"]))

    # 4. THE CORNER THAT USED TO PASS AND DOES NOT. A 0.46 m plate at the top
    #    of the ride band clears the deck upright by 1 cm; leaned 15 deg it
    #    does not clear it at all, and the run now says so.
    tall = list(x)
    tall[labels.index("endplate_h_m")] = 0.46
    tall[labels.index("ride_height_m")] = 0.70
    assert fly(tall, 90.0)["g_reach"] > 0.0
    assert fly(tall, 75.0)["g_reach"] < 0.0
    assert fly(tall, 60.0)["g_reach"] < fly(tall, 75.0)["g_reach"]


def test_no_plate_is_a_pin_and_needs_a_load_path_that_is_not_the_plate():
    """"none" is not a flag — there is no plate of no height, only a height of
    zero — so it is a pin on the design row, and it is offered only where the
    PYLONS carry the load: with the plates gone a tip-borne mount has nothing
    to grip."""
    from gui.v3 import config, session
    from gui.v3.app import assemble

    ch = v1.start_choices(medium="track")
    v1.normalise_choices(ch)
    assert "none" not in v1.car_tip_shapes(ch, v1.derive_problem(ch)[0])
    ch["car_endplates"] = False          # = the pylon mount
    assert "none" in v1.car_tip_shapes(ch, v1.derive_problem(ch)[0])

    ctx = assemble("track")
    # the MOUNT first, and applied — it changes the FAMILY, and a pin belongs
    # to the problem it was set on, so apply_choices drops the fixed rows on
    # a family change. That is the real order too: the shape menu only exists
    # once the mount has already put the plate on a pylon-borne wing.
    ctx.S["wing"]["choices"]["car_endplates"] = False    # = the pylon mount
    session.apply_choices(ctx.S)
    ctx.S["wing"]["choices"]["car_tip_shape"] = "none"
    ctx.S["wing"].setdefault("fixed", {})["endplate_h_m"] = 0.0
    session.apply_choices(ctx.S)
    # the row leaves the design vector rather than becoming a zero-width band
    assert config.fixed_rows(ctx.S)["endplate_h_m"] == 0.0
    assert config.cfg_dict(ctx.S)["pinned"] == {"endplate_h_m": 0.0}
    assert "endplate_h_m" not in (config.bounds_overrides(ctx.S) or {})

    # ...and a plateless wing really flies: it loses the fence and the
    # nonplanar benefit, which is a cost the answer can see
    built = api.PROBLEM_SPECS["car rear wing"].build({}, {}, None)
    x = _mid(built)
    x[list(built.param_labels).index("endplate_h_m")] = 0.0
    bare = built.evaluate(x)
    assert bare["feasible"] and np.isfinite(bare["score"])
    assert bare["score"] < built.evaluate(_mid(built))["score"]


def test_the_plate_can_continue_the_wing_s_chord_law():
    """The car's half of the aircraft's "its chord" switch.

    Declared on the two families whose plate has no chord row of its own, and
    refused on the designed-endplate family, where continuing the wing's law
    onto the plate would be a second answer to a question the search is
    already answering.
    """
    assert "endplate_chord_follows" in api.accepted_flags("car rear wing")
    assert "endplate_chord_follows" in api.accepted_flags(
        "car rear wing (two-element)")
    with pytest.raises(KeyError):
        api.check_flags("car rear wing + endplates",
                        {"endplate_chord_follows": True})

    # asked under the PYLON mount, where the plate is a tip DEVICE and has no
    # chord row of its own
    ch = v1.start_choices(medium="track")
    ch["car_endplates"] = False
    ch["car_endplate_chord_follows"] = True
    v1.normalise_choices(ch)
    name = v1.derive_problem(ch)[0]
    assert v1.car_flags(ch, 55.0, name)["endplate_chord_follows"] is True
    # ...and the designed-endplate family does not send it even when asked
    ch["car_endplates"] = True
    v1.normalise_choices(ch)
    name = v1.derive_problem(ch)[0]
    assert "endplate_chord_follows" not in v1.car_flags(ch, 55.0, name)
    assert api.check_flags(name, v1.car_flags(ch, 55.0, name)) is None

    rect = api.PROBLEM_SPECS["car rear wing"].build({}, {}, None)
    follow = api.PROBLEM_SPECS["car rear wing"].build(
        {}, {"endplate_chord_follows": True}, None)
    a, b = rect.evaluate(_mid(rect)), follow.evaluate(_mid(follow))
    # a tapered wing's plate stops being a rectangle, so the shape changes
    assert a["score"] != b["score"]
    assert np.isfinite(b["score"])


# =========================================== 6. the mount is in the picture

def test_the_pylons_are_drawn_and_the_plate_borne_mount_is_not():
    """``carmount.py`` has priced a strut-borne car wing since it was written
    and no figure ever drew one: the picture showed a wing floating over the
    track with the single body that distinguishes a swan-neck layout from a
    plate-borne one invisible.

    An ENDPLATE mount still draws nothing, and that is not an omission: its
    load path IS the plate, which is already in the picture as the wing's own
    outboard panels. A body for it would be a second plate no solver flew.
    """
    from aerobo import cad, carmount

    ch = v1.start_choices(medium="track")
    v1.normalise_choices(ch)

    drawn = {}
    names = {}
    for layout in ("tips", "pylons"):
        # the mount picks the family, so each answer is drawn from the
        # problem it actually derives to
        ch["car_endplates"] = layout == "tips"
        v1.normalise_choices(ch)
        name = names[layout] = v1.derive_problem(ch)[0]
        flags = v1.car_flags(ch, v1.car_default_v_ms(), name)
        built = api.PROBLEM_SPECS[name].build({}, flags, None)
        x = _mid(built)
        report = api.design_report(
            api.RunConfig(problem_name=name, flags=flags), x)
        geom = report["geometry"]
        bodies = cad.mount_surfaces(geom, x, report["param_labels"])
        fig = v1.fig_wing3d(geom, x, report["param_labels"])
        drawn[layout] = (geom, bodies, len(fig.data), fig.layout.title.text)

    assert drawn["tips"][1] == []
    assert "mount" not in drawn["tips"][3].lower()

    # ...and the case that actually tests the RULE rather than the absence of
    # a spec: the published tip mount carries no MountSpec at all, so it would
    # draw nothing whatever the rule said. An endplate-borne mount stated as a
    # continuum HAS a spec, a station and a length — and must still draw
    # nothing, because its load path is already in the picture as the wing's
    # own outboard panels.
    plate_flags = {"car_mount_model": "continuum", "mount_kind": "endplate",
                   "mount_station_frac": 0.35}
    name = names["pylons"]              # the family that HAS a mount spec
    plate_built = api.PROBLEM_SPECS[name].build({}, plate_flags, None)
    plate_x = _mid(plate_built)
    plate_rep = api.design_report(
        api.RunConfig(problem_name=name, flags=plate_flags), plate_x)
    assert plate_rep["geometry"]["mount"]["kind"] == "endplate"
    assert plate_rep["geometry"]["mount"]["stations_m"]
    assert plate_rep["geometry"]["mount"]["length_m"] > 0.0
    assert cad.mount_surfaces(plate_rep["geometry"], plate_x,
                              plate_rep["param_labels"]) == []

    geom, bodies, n_traces, title = drawn["pylons"]
    assert len(bodies) == 2                      # the beam's two supports
    assert n_traces == drawn["tips"][2] + 2      # ...both in the figure
    assert "swan neck" in title

    # WHERE they are: the stations the BEAM is supported at, off the spec —
    # never footprint_stations_m, which is the suction-side loss footprint and
    # is deliberately empty on a swan neck (the layout most worth seeing)
    spec = carmount.published_layout("tips")
    del spec
    # the midpoint of each body's y extent — the loop repeats its first
    # corner, so a plain mean is biased by half a strut thickness
    ys = sorted(0.5 * float(b.Y.min() + b.Y.max()) for b in bodies)
    want = sorted(geom["mount"]["stations_m"])
    assert ys == pytest.approx(want, abs=1e-6)
    assert abs(ys[0]) < 0.5 * float(geom["b"])   # inboard of the tip

    # ...and how far they REACH: to the car's deck, not to the track. A pylon
    # that ran the whole ride height would be six times too long at the
    # published point, which is the error carmount.pylon_length_m documents.
    for body in bodies:
        assert float(body.Z.max() - body.Z.min()) == pytest.approx(
            geom["mount"]["length_m"], abs=1e-9)
    assert geom["mount"]["length_m"] < geom["mount"]["ride_height_m"]


def test_the_drawn_mount_holds_the_wing_where_the_run_said():
    """The chordwise attachment is in the picture too — the number that
    decides whether the wing winds up towards more downforce with speed or
    sheds incidence. Drawn at the quarter chord when the run flew it aft, it
    would hide exactly the geometry that argument is about."""
    from aerobo import cad, carmount

    name = "car rear wing"
    seen = {}
    for frac in (carmount.X_AC_FRAC, 0.45):
        flags = {"car_mount_model": "continuum", "mount_kind": "pylon",
                 "mount_side": "pressure", "mount_station_frac": 0.35,
                 "x_attach_frac": frac}
        built = api.PROBLEM_SPECS[name].build({}, flags, None)
        x = _mid(built)
        report = api.design_report(
            api.RunConfig(problem_name=name, flags=flags), x)
        bodies = cad.mount_surfaces(report["geometry"], x,
                                    report["param_labels"])
        assert bodies
        seen[frac] = float(np.mean([b.X.mean() for b in bodies]))
    # aft of the aerodynamic centre is drawn aft of it
    assert seen[0.45] > seen[carmount.X_AC_FRAC]
