"""The LAP — now an ENGINE capability rather than a card question.

READ THIS FIRST. The circuit control has been REMOVED from every shell. The
whole of what follows was written when the card asked for a circuit, and most
of it still holds because almost none of it was ever about the widget: the
lap reaches ``track_spec``, is flown, is scored as minus the time, is
reported segment by segment, and is refused in contract where it cannot be
honoured. All of that is unchanged and still statable from ``car_flags``, a
script or a saved session.

What changed, and where it shows up below:

* the card asks no circuit, so ``car_track_of`` reads "off" for every
  card-built choice dict and ``car_objective_options`` drops ``laptime`` on
  its own. The tests that asserted the CONTROL now assert its ABSENCE
  (``test_the_sampling_count_is_no_longer_a_card_question``,
  ``test_the_pair_is_refused_by_neither_and_no_card_asks_it``);
* the MOUNT picks the family, and the lap lives on the families the PYLON
  mount derives to — the designed-endplate family carries no ``track_spec``
  at all. So ``_track`` builds its choices on that mount, and the V3 sessions
  below set ``car_endplates=False`` for the same reason. This is not a
  workaround: it is the same fact
  ``test_the_designed_endplate_family_is_offered_no_circuit_at_all``
  has always asserted, now reached through the mount instead of a switch.


Session 63 built ``cartrack.py`` and gave ``carwing`` a circuit; session 64
gave the same circuit to ``carwing_multi``. Neither wired a control, and the
gap was measurable in one line — ``grep -rn car_track gui/`` returned nothing
while ``api.accepted_flags`` declared ``car_track`` and ``track_points`` on
FOUR registered families. The objective the whole of session 64 was designed
around (``carwing.CAR_OBJECTIVES['laptime']``) could not be selected from any
shell, on any family.

This file gates the wiring, and every assertion is an OUTCOME rather than a
restatement of the code: a flag that reaches the FIELD and is FLOWN, a menu
whose every entry builds, a state that falls back when its menu does.

The five things it holds:

1. **The circuit is reachable and it is FLOWN**, on all four lap-capable
   families and from both shells (V1's card and V3's, which draws V1's).
2. **``laptime`` is offered only where a circuit is stated.** Both car problem
   classes refuse the objective by name without one ("objective 'laptime'
   needs a circuit"), so a Maximise menu that could send it without one would
   be a menu promising a run the registry declines at the Run button. The
   direction is deliberate and is asserted as such: the CIRCUIT unlocks the
   OBJECTIVE, never the other way round (``v1.car_objective_options``' own
   docstring argues it, and the test below holds the consequence — choosing
   ``laptime`` installs no circuit).
3. **Every combination the card can produce passes ``api.check_flags`` and
   builds** — the whole product of circuit x sampling count x objective x
   elements x plates x chord law, on every family it derives to. This is the
   `test_every_combination_builds` contract narrowed onto the new control.
4. **``track_points`` cannot be sent without a circuit.**
   ``api._car_track_kwargs`` refuses that pair by name; the card must not be
   able to express it, and it must not merely be normalised out of the state
   — ``car_flags`` refuses to send it even from a hand-built choices dict.
5. **The lap is PRESENTABLE.** ``lap_time_s`` had a Spec and nothing else did:
   the per-segment breakdown, the representative speeds the wing was re-flown
   at, the drag-limited top speed and the aero-balance window were computed on
   every lap run and printed nowhere.

...and the one DESIGN DECISION this session took on the user's behalf, which
is asserted rather than written down only in prose: **a drag ceiling and a lap
are both honoured, neither is refused, and the pair is stated.** It used to be
stated on the card where the circuit was chosen as well; that half went with
the control, and what remains is the field where the ceiling is typed and the
no-solution path when the ceiling is the binding limit. The
lap prices drag physically (every newton is paid for on the straights, at the
circuit's own exchange rate) so a ceiling beside it can only delete designs
the lap has already judged worth their drag —
``RESULTS_SESSION64_CARSECTION`` section 7 measured the extreme case: under
the family's own retired coefficient allowance the 2-D lap optimum has NO
feasible design in 24 searches of 4000 evaluations each.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aerobo import api, carmount                              # noqa: E402
from gui import diagnose, metrics                             # noqa: E402
from gui import nice_app as v1                                # noqa: E402

SINGLE = "car rear wing"
MULTI = "car rear wing (two-element)"
PLATES = "car rear wing + endplates"

#: the four families that declare the two lap keys. Named here so the test
#: fails if the registry stops declaring one of them, rather than quietly
#: testing three.
LAP_FAMILIES = (SINGLE, f"{SINGLE} + free chord law",
                MULTI, f"{MULTI} + free chord law")

#: ...and the two that cannot. ``CarWingEndplateProblem`` has no
#: ``track_spec``, ``car_spec`` or ``track_points`` field at all.
NO_LAP_FAMILIES = (PLATES, f"{PLATES} + free chord law")


def _mid(built) -> np.ndarray:
    return 0.5 * (built.bounds[:, 0] + built.bounds[:, 1])


def _track(**choices) -> dict:
    """A track configuration, already normalised, as a card would hold it.

    ON THE PYLON MOUNT unless a caller says otherwise, because that is the
    layout the lap exists on. The mount picks the family (``car_endplates``
    is its key), and a plate-borne wing is the designed-endplate family,
    which carries no ``track_spec`` — ``api.CAR_ENDPLATE_KEYS`` declares
    neither lap key, and ``test_the_designed_endplate_family_is_offered_no_
    circuit_at_all`` is the assertion about that. Every test below is about
    the lap, so the default here is the mount that has one.
    """
    ch = v1.start_choices(medium="track")
    ch.setdefault("car_endplates", False)
    ch["car_endplates"] = choices.pop("car_endplates", False)
    ch.update(choices)
    v1.normalise_choices(ch)
    return ch


def _texts(view) -> list[str]:
    out = []
    for e in view.descendants():
        t = getattr(e, "text", "") or ""
        if t:
            out.append(t)
        opts = getattr(e, "options", None)
        if isinstance(opts, dict):
            out += [str(x) for x in opts.values()]
        elif isinstance(opts, (list, tuple)):
            out += [str(x) for x in opts]
    return out


def _selects(view) -> list[dict]:
    return [e.options for e in view.descendants()
            if isinstance(getattr(e, "options", None), dict)]


# ============================================ 1. the circuit reaches the field

def test_the_registry_declares_the_lap_on_four_families_and_not_the_other_two():
    """The gap this file closes, measured from the registry's own side.

    Catches: adding the control against three families and calling it done,
    and the reverse — declaring the lap keys on the designed-endplate family,
    where they would be accepted by ``check_flags`` and read by nothing.
    """
    for name in LAP_FAMILIES:
        flags = api.accepted_flags(name)
        for key in api.CAR_LAP_KEYS:
            assert key in flags, (name, key)
    for name in NO_LAP_FAMILIES:
        flags = api.accepted_flags(name)
        for key in api.CAR_LAP_KEYS:
            assert key not in flags, (name, key)


@pytest.mark.parametrize("two_element", (False, True))
@pytest.mark.parametrize("chord", ("fixed", "free"))
def test_the_card_sends_a_circuit_that_is_flown(two_element, chord):
    """choices -> flags -> the PROBLEM'S OWN FIELD -> a lap that was timed.

    The whole point of the wiring, asserted end to end on all four
    lap-capable families: not that the flag is accepted, but that it reaches
    ``track_spec``, that the sampling count reaches ``track_points``, and that
    the evaluation comes back with a positive lap time scored as its negative.

    Catches: dropping the ``car_track`` branch out of ``car_flags`` (no flag
    is sent, the problem has no ``track_spec``, and the objective is refused
    at build); and sending the flag under a name the builder ignores, which
    passes ``check_flags`` and leaves ``track_spec`` None.
    """
    ch = _track(car_two_element=two_element, chord=chord,
                car_track="synthetic", car_objective="laptime",
                car_track_points=3)
    name = v1.derive_problem(ch)[0]
    assert name in LAP_FAMILIES, name
    flags = v1.car_flags(ch)
    assert flags["car_track"] == "synthetic"
    assert flags["track_points"] == 3
    api.check_flags(name, flags)

    built = api.PROBLEM_SPECS[name].build({}, flags, None)
    assert built.problem.track_spec is not None
    assert built.problem.track_points == 3
    out = built.evaluate(_mid(built))
    assert out["feasible"], out["reason"]
    assert out["lap_time_s"] > 0.0
    assert out["score"] == pytest.approx(-out["lap_time_s"])
    # ...and the count was honoured, not merely stored
    assert 0 < len(out["track_points"]) <= 3


def test_the_v3_shell_reaches_the_lap_on_both_car_families():
    """The same journey through V3, which draws V1's card but builds its own
    flag set (``gui.v3.config.flags``).

    Catches: a lap key that ``config.flags`` strips on its way past — the
    function pops several key groups that do not belong to the derived family
    — and a V3 session whose choices never reach ``car_flags`` at all.
    """
    from gui.v3 import config, session

    for two_element in (False, True):
        S = session.make_session("track")
        S["wing"]["choices"].update(car_endplates=False,   # the pylon mount
                                    car_two_element=two_element,
                                    car_track="synthetic",
                                    car_objective="laptime",
                                    car_track_points=2)
        session.apply_choices(S)
        name = S["wing"]["problem"]
        assert name in LAP_FAMILIES, name
        flags = config.flags(S)
        assert flags["car_track"] == "synthetic"
        assert flags["track_points"] == 2
        api.check_flags(name, flags)
        built = config.spec(S).build({}, flags, config.bounds_overrides(S))
        assert built.problem.track_spec is not None
        out = built.evaluate(_mid(built))
        assert out["feasible"], out["reason"]
        assert out["score"] == pytest.approx(-out["lap_time_s"])


def test_the_two_shells_send_the_same_lap():
    """One control, one home. V3 draws ``v1._car_controls``, so the two shells
    must not be able to disagree about which circuit was chosen.

    Catches: V3 growing its own circuit key or its own default sampling count
    — which would make "the slot bought 0.4 s" a comparison of two shells.
    """
    from gui.v3 import config, session

    ch = _track(car_track="synthetic", car_objective="laptime")
    S = session.make_session("track")
    S["wing"]["choices"].update(car_endplates=False,       # the pylon mount
                                car_track="synthetic",
                                car_objective="laptime")
    session.apply_choices(S)
    v3_flags = config.flags(S)
    v1_flags = v1.car_flags(ch, S["mission"].get("V"))
    for key in api.CAR_LAP_KEYS:
        assert v3_flags.get(key) == v1_flags.get(key), key


# ================================= 2. laptime is offered only with a circuit

def test_laptime_is_in_the_menu_only_once_a_circuit_is_chosen():
    """The rule, at the menu.

    Catches: putting ``laptime`` in ``CAR_OBJECTIVE_LABELS`` and drawing that
    table directly (the shipped state before this session had the table
    without the entry; the failure mode being gated is the fix done by half).
    """
    off = _track()
    on = _track(car_track="synthetic")
    assert "laptime" not in v1.car_objective_options(off)
    assert "laptime" in v1.car_objective_options(on)
    # ...and the entry is not silently lost: the card says why it is missing
    note = v1.missing_options_note(v1.car_objective_options(off),
                                   v1.CAR_OBJECTIVE_LABELS,
                                   v1.CAR_OBJECTIVE_WHY)
    assert "circuit" in note.lower(), note
    assert not v1.missing_options_note(v1.car_objective_options(on),
                                       v1.CAR_OBJECTIVE_LABELS,
                                       v1.CAR_OBJECTIVE_WHY)


def test_the_circuit_unlocks_the_objective_and_not_the_reverse():
    """THE DIRECTION, held as an outcome.

    Choosing ``laptime`` must not INSTALL a circuit. Every number a lap
    produces is a property of ``cartrack.synthetic_lap``, which is synthetic
    and says so — RESULTS_SESSION64_CARSECTION section 8 opens on exactly
    that, and measures a layout on which this repo's own winner is 0.79 s
    SLOWER than the section it beat. A mission that appears as a side effect
    of a scoring menu is what that finding forbids.

    Catches: "helpfully" defaulting ``car_track`` to the one shipped circuit
    whenever the objective is a lap — which would make the card's Circuit row
    a display of a decision taken elsewhere, and would leave de-selection
    ambiguous.
    """
    ch = _track()
    ch["car_objective"] = "laptime"
    assert v1.car_track_of(ch) is None
    assert "car_track" not in v1.car_flags(ch)
    # ...and the state does not hold an objective its own menu cannot offer:
    # the normaliser puts it back to "the family's own default" (None), never
    # to a spelling of that default
    dropped = v1.normalise_choices(ch)
    assert "car_objective" in dropped
    assert ch["car_objective"] is None


def test_switching_the_circuit_off_takes_the_objective_and_the_count_with_it():
    """The stale-menu rule, one control down — the same one
    ``_normalise_winglet_blend`` obeys.

    Catches: normalising only the family-SELECTING keys. ``car_track``
    selects no family, so ``choices_consistent`` is True the whole way
    through and a circuit switched off would leave ``laptime`` in the state,
    a Maximise select showing a value its own menu no longer offers, and a
    run refused at the Run button.
    """
    ch = _track(car_track="synthetic", car_objective="laptime",
                car_track_points=6)
    ch["car_track"] = "off"
    dropped = v1.normalise_choices(ch)
    assert set(dropped) >= {"car_objective", "car_track_points"}
    assert ch["car_objective"] is None
    assert ch["car_track_points"] is None
    # what is left is the MOUNT and nothing else. On the pylon layout that is
    # four statements rather than none — the struts are a real load path, not
    # an absence — and every one of them is derived from the one answer
    # (v1.car_mount_layout_flags), never asked.
    assert v1.car_flags(ch) == {
        "mount": "tips", "car_mount_model": "continuum", "mount_kind": "pylon",
        "mount_station_frac": pytest.approx(carmount.INBOARD_STATION_FRAC),
        "mount_side": "pressure"}


def test_leaving_the_track_takes_the_whole_lap_with_it():
    """Catches: a circuit surviving a medium change, then being sent to an
    air family that has never heard of it."""
    ch = _track(car_track="synthetic", car_objective="laptime")
    ch["medium"] = "air"
    dropped = v1.normalise_choices(ch)
    assert "car_track" in dropped and ch["car_track"] == "off"
    assert v1.car_flags(ch) == {}


def test_the_designed_endplate_family_is_offered_no_circuit_at_all():
    """That family has no ``track_spec`` field, so the row is not drawn, the
    flag is not sent, and a circuit already in the state falls back.

    Catches: gating the Circuit row on the medium alone. The endplate family
    IS a track family; what it lacks is the field, and only the registry
    knows that.
    """
    ch = _track(car_track="synthetic", car_objective="laptime",
                car_track_points=4)
    ch["car_endplates"] = True
    assert v1.derive_problem(ch)[0] in NO_LAP_FAMILIES
    assert v1.car_track_options(ch) == {}
    assert "laptime" not in v1.car_objective_options(ch)
    v1.normalise_choices(ch)
    assert ch["car_track"] == "off" and ch["car_track_points"] is None
    flags = v1.car_flags(ch)
    for key in api.CAR_LAP_KEYS:
        assert key not in flags
    api.check_flags(v1.derive_problem(ch)[0], flags)


def test_the_endplate_family_refuses_the_objective_in_contract():
    """...and if a caller reaches past the card, it is REFUSED rather than
    accepted and then raising.

    Before this session ``CarWingEndplateProblem`` validated its objective
    against the whole of ``carwing.CAR_OBJECTIVES``, which contains
    ``laptime``: the problem CONSTRUCTED and then raised ``KeyError('laptime')``
    out of the scorer's own lookup at the first evaluation. An in-contract
    question answered with a programming error, and invisible until something
    ran.

    Catches: removing ``ENDPLATE_OBJECTIVES`` and validating against
    ``CAR_OBJECTIVES`` again.
    """
    from aerobo import carwing, endplate

    assert set(endplate.ENDPLATE_OBJECTIVES) == \
        set(carwing.CAR_OBJECTIVES) - {"laptime"}
    for name in NO_LAP_FAMILIES:
        with pytest.raises(ValueError, match="laptime"):
            api.PROBLEM_SPECS[name].build({}, {"car_objective": "laptime"},
                                          None)
    # ...and every objective it DOES declare still scores, at the box centre
    for objective in endplate.ENDPLATE_OBJECTIVES:
        flags = {"car_objective": objective}
        if objective in ("cz", "cd"):
            # a COEFFICIENT against a designed area is refused by the same
            # class for its own separate reason — assert THAT, not silence.
            # Both halves of the pair: CZ is maximised by shrinking the area
            # it is referenced to and CD is minimised by growing it, so this
            # registered family (whose area IS a design row) can score
            # neither. A third coefficient objective would land here and fail
            # loudly rather than be skipped.
            with pytest.raises(ValueError, match="free reference area"):
                api.PROBLEM_SPECS[PLATES].build({}, flags, None)
            continue
        built = api.PROBLEM_SPECS[PLATES].build({}, flags, None)
        out = built.evaluate(_mid(built))
        assert out["feasible"], (objective, out["reason"])
        assert out["objective"] == objective


# ============================== 3. every combination the card can produce

def _card_configurations():
    """The product of every control this card can be walked into.

    Not a sample: the card has exactly these switches, and the combination
    matrix is small enough to take whole.
    """
    for two_element, plates, chord, track, points in itertools.product(
            (False, True), (False, True), ("fixed", "free"),
            ("off", "synthetic"), (None, 2)):
        ch = _track(car_two_element=two_element, car_endplates=plates,
                    chord=chord, car_track=track, car_track_points=points)
        # the card cannot hold a count without a circuit; normalise_choices
        # is what makes that true, and it has already run
        for objective in v1.car_objective_options(ch):
            yield dict(ch, car_objective=objective)


def test_every_combination_the_card_can_produce_builds_and_evaluates():
    """The gate: a menu that can send a refused combination is a menu that
    lies.

    Every offered objective, in every circuit state, on every family this
    card derives to — through the shell's own ``car_flags``, through
    ``api.check_flags``, into a build and an evaluation.

    Catches: offering ``laptime`` on the designed-endplate family (refused at
    build), sending ``car_track`` there (refused by ``check_flags``), and
    sending ``track_points`` with no circuit (refused by name in
    ``api._car_track_kwargs``).
    """
    seen_names, seen_objectives, n = set(), set(), 0
    for ch in _card_configurations():
        name = v1.derive_problem(ch)[0]
        flags = v1.car_flags(ch)
        api.check_flags(name, flags)
        built = api.PROBLEM_SPECS[name].build({}, flags, None)
        out = built.evaluate(_mid(built))
        assert out["feasible"], (name, flags, out["reason"])
        assert out["objective"] == ch["car_objective"]
        if v1.car_track_of(ch) is not None:
            assert built.problem.track_spec is not None, (name, flags)
            assert out["lap_time_s"] > 0.0
        else:
            assert flags.get("car_track") is None
            assert "track_points" not in flags
        seen_names.add(name)
        seen_objectives.add(ch["car_objective"])
        n += 1
    # the sweep really did reach every family and every objective, so a gate
    # that silently stopped generating cases fails here rather than passing
    assert seen_names == set(LAP_FAMILIES) | set(NO_LAP_FAMILIES), seen_names
    # DERIVED from the menu, not pinned: the claim is "the sweep reached
    # every entry the card can offer", and a literal list here would silently
    # stop being that claim the next time the menu grows an objective.
    assert seen_objectives == set(v1.CAR_OBJECTIVE_LABELS)
    assert n >= 32, n


def test_the_menu_can_never_state_a_count_without_a_circuit():
    """``api._car_track_kwargs`` refuses that pair BY NAME, and the card must
    not be able to express it — including from a hand-built choices dict that
    never went through the normaliser.

    Catches: sending ``track_points`` outside the ``car_track`` branch, which
    normalisation would hide on the happy path and a preset or a saved record
    would not.
    """
    raw = dict(v1.BUILDER_DEFAULTS, medium="track",
               car_track="off", car_track_points=6)
    assert "track_points" not in v1.car_flags(raw)
    # ...and the api-side refusal it would have hit is still there
    with pytest.raises(ValueError, match="track_points"):
        api.PROBLEM_SPECS[SINGLE].build({}, {"track_points": 6}, None)


def test_the_label_table_names_exactly_the_circuits_the_registry_has():
    """A menu entry with no circuit behind it, or a circuit with no entry.

    Catches: registering a second lap in ``api.CAR_TRACKS`` and leaving the
    card offering one — the capability would exist and be unclickable, which
    is the bug ``test_menus_offer_what_exists`` exists for one menu across.
    """
    assert v1._CAR_TRACK_OFF_VALUE in api.CAR_TRACK_OFF
    named = set(v1.CAR_TRACK_LABELS) - {v1._CAR_TRACK_OFF_VALUE}
    assert named == set(api.CAR_TRACKS), named
    offered = v1.car_track_options(_track())
    assert set(offered) == {v1._CAR_TRACK_OFF_VALUE} | set(api.CAR_TRACKS)


def test_an_untouched_track_card_states_its_mount_and_no_lap():
    """What an untouched card asks for, now that the circuit has left it.

    The card states exactly ONE thing: the mount. Everything else is the
    family's own default, so the only difference between an untouched card's
    problem and the family's bare one is the load path it chose — never a
    circuit, never a budget, never a size.

    Catches: defaulting ``car_track`` to the shipped circuit, or sending the
    "off" mode string as a flag (both car builders accept it, so neither
    would raise — the run would simply stop being the published one).
    """
    for two_element in (False, True):
        ch = _track(car_two_element=two_element)
        assert ch["car_track"] == "off"
        flags = v1.car_flags(ch)
        # NOTHING about the lap travels from a card that has none
        assert not (set(flags) & set(api.CAR_LAP_KEYS))
        # ...and what does travel is the mount, and only the mount
        assert set(flags) - {"mount"} <= set(api._CAR_MOUNT_REFINE_KEYS) | {
            "car_mount_model"}
        name = v1.derive_problem(ch)[0]
        built = api.PROBLEM_SPECS[name].build({}, flags, None)
        bare = api.PROBLEM_SPECS[name].build({}, {}, None)
        assert built.problem.track_spec is None
        assert built.problem.objective == bare.problem.objective
        got = built.evaluate(_mid(built))
        was = bare.evaluate(_mid(bare))
        if set(flags) == {"mount"}:
            # no refinement to send (the slotted family takes no MountSpec),
            # so the card's problem IS the family's own, bit for bit
            assert got["score"] == was["score"]
        else:
            # the pylon mount is a real load path and it is priced: two
            # wetted struts and two corners, holding the wing near the middle
            assert got["CD"] > was["CD"]
            assert got["deflection_m"] < was["deflection_m"]
            assert got["mount_lift_knockdown"] == 0.0


# ================================================== 4. the choices ROUND-TRIP

@pytest.mark.parametrize("name", LAP_FAMILIES)
def test_the_choices_round_trip_with_a_circuit(name):
    """choices -> problem -> choices, with the lap on.

    The circuit is a FLAG and not a family, so the round trip has two halves
    and both are asserted: the problem NAME is unchanged by choosing a
    circuit (a lap must not silently select a different solver), and the
    flags those choices send are honoured by that family down to the field.

    Catches: making the lap a family variant, which would put a second axis
    in the registry for a question that is configuration; and an inverse that
    forgets the circuit, so re-deriving from a preset drops it.
    """
    ch = v1.choices_from_problem(name)
    assert ch["medium"] == "track"
    before = v1.derive_problem(ch)[0]
    assert before == name

    ch["car_track"] = "synthetic"
    ch["car_objective"] = "laptime"
    ch["car_track_points"] = 5
    assert v1.normalise_choices(ch) == []      # nothing had to fall back
    assert v1.derive_problem(ch)[0] == name    # ...and the family is the same

    flags = v1.car_flags(ch, problem_name=name)
    api.check_flags(name, flags)
    built = api.PROBLEM_SPECS[name].build({}, flags, None)
    assert built.problem.track_spec is not None
    assert built.problem.track_points == 5
    assert built.problem.objective == "laptime"


def test_the_v3_state_round_trips_the_circuit_through_a_rebuild():
    """V3 re-derives the problem on every choice; the circuit must survive it.

    Catches: ``apply_choices`` rebuilding ``W["flags"]`` and losing a control
    that is a CHOICE — the class of defect
    `a-flag-the-new-family-declares-is-kept` records, here for a key that
    would look like it had never been set.
    """
    from gui.v3 import config, session

    S = session.make_session("track")
    S["wing"]["choices"].update(car_endplates=False,       # the pylon mount
                                car_track="synthetic", car_objective="laptime",
                                car_track_points=3)
    session.apply_choices(S)
    first = config.flags(S)
    # a second, unrelated edit and a second apply — the circuit must not move
    S["wing"]["choices"]["car_mount"] = "inboard"
    session.apply_choices(S)
    second = config.flags(S)
    for key in api.CAR_LAP_KEYS:
        assert second.get(key) == first.get(key) is not None, key
    assert S["wing"]["choices"]["car_objective"] == "laptime"


# =================================================== 5. the lap is REPORTABLE

def _lap_breakdown(points: int = 4, **flags):
    built = api.PROBLEM_SPECS[SINGLE].build(
        {}, dict({"car_track": "synthetic", "car_objective": "laptime",
                  "track_points": points}, **flags), None)
    out = built.evaluate(_mid(built))
    assert out["feasible"], out["reason"]
    return out


def test_the_lap_report_says_where_the_time_went():
    """The per-segment breakdown, as an outcome rather than a shape: the
    segment times SUM to the lap time, and the shares sum to one.

    Catches: reading ``V_mean`` where ``t_s`` is meant, dropping a segment
    from the table, and computing the share against the wrong total — all of
    which produce a table that looks right and does not add up.
    """
    out = _lap_breakdown()
    rep = metrics.lap_report(out)
    assert rep is not None
    segs = rep["segments"]
    assert len(segs) == len(out["lap"]["segments"]) > 0
    assert sum(s["t_s"] for s in segs) == pytest.approx(out["lap_time_s"])
    assert sum(s["share"] for s in segs) == pytest.approx(1.0)
    assert rep["summary"]["lap_time_s"] == out["lap_time_s"]
    # the circuit's own NAME travels with its numbers, and it says what it is
    assert "not a real circuit" in rep["summary"]["track"]
    # a corner is grip-limited on this layout and a straight is not
    kinds = {s["kind"] for s in segs}
    assert kinds == {"corner", "straight"}
    assert any(s["kind"] == "corner" and s["limited_by"] == "grip"
               for s in segs)


def test_the_long_straight_is_reported_as_drag_limited():
    """The one number that says whether this lap CHARGES the wing's drag.

    ``cartrack.synthetic_lap``'s docstring picks its 1400 m straight so that
    the end of it is drag-limited rather than power-limited; the report has
    to carry that, because a straight that ran out first would be pricing the
    wing's drag at nothing.

    Catches: dropping ``drag_frac_at_peak`` / ``V_peak_over_top`` from the
    segment rows, which leaves the table saying only how long each piece took.
    """
    rep = metrics.lap_report(_lap_breakdown())
    straights = [s for s in rep["segments"] if s["kind"] == "straight"]
    assert straights
    longest = max(straights, key=lambda s: s["length_m"])
    assert longest["length_m"] == 1400.0
    # DERIVED from the run, never pinned: the longest straight must be the one
    # that gets closest to the drag-limited top speed, and must spend most of
    # the engine on drag when it does
    assert longest["v_peak_over_top"] == max(s["v_peak_over_top"]
                                             for s in straights)
    assert longest["v_peak_over_top"] > 0.9
    assert longest["drag_frac_at_peak"] > 0.8
    corners = [s for s in rep["segments"] if s["kind"] == "corner"]
    assert all(s["drag_frac_at_peak"] is None for s in corners)


def test_the_per_speed_rows_are_reported_and_the_count_is_the_one_asked_for():
    """What ``track_points`` bought, on the page.

    Each point is a full re-solve of the wing, so a user who pays four times
    per candidate has to be able to see the four rows. This is the defect
    `tests/test_a_registered_endpoint_is_reported.py` exists for, one layer
    down: a number computed on every run and never printed.

    Catches: reporting the lap total and dropping the rows; and reporting a
    fixed number of rows regardless of what was asked for.
    """
    for asked in (1, 3):
        rep = metrics.lap_report(_lap_breakdown(points=asked))
        assert 0 < len(rep["points"]) <= asked
        for row in rep["points"]:
            assert row["v"] > 0.0
            assert row["cz"] is not None and row["cd"] is not None
        # the weights are a partition of the lap's own work
        assert all(0.0 <= r["weight"] <= 1.0 for r in rep["points"])
    # ...and more points really is a different sample, not a repeated one
    few = metrics.lap_report(_lap_breakdown(points=2))["points"]
    many = metrics.lap_report(_lap_breakdown(points=4))["points"]
    assert len(many) > len(few)
    assert {round(r["v"], 6) for r in many} != {round(r["v"], 6) for r in few}


def test_the_three_lap_speeds_are_named_metrics():
    """``V_top``, the fastest point and the slowest corner reach the scalar
    grid; ``V_mean`` deliberately does not.

    ``V_mean`` is ``length_m / lap_time_s`` EXACTLY — it is the objective
    restated in another unit, and this catalogue's rule is one entry per
    question.

    Catches: leaving the three inside the nested ``lap`` dict, where nothing
    renders them; and adding ``V_mean`` beside the lap time.
    """
    out = _lap_breakdown()
    shown = {}
    for _title, rows in metrics.groups(out):
        shown.update({r["key"]: r for r in rows})
    for key in ("lap_time_s", "lap_V_top_ms", "lap_V_max_ms", "lap_V_min_ms",
                "aero_balance"):
        assert key in shown, (key, sorted(shown))
    assert shown["lap_V_top_ms"]["value"] == pytest.approx(out["lap"]["V_top"])
    # the redundancy that is deliberately absent, and the identity that makes
    # it one
    assert out["lap"]["V_mean"] == pytest.approx(
        out["lap"]["length_m"] / out["lap_time_s"])
    assert not any(k.endswith("V_mean_ms") for k in shown)
    # a run with no circuit gains none of them
    plain = api.PROBLEM_SPECS[SINGLE].build({}, {}, None)
    bare = plain.evaluate(_mid(plain))
    keys = {r["key"] for _t, rows in metrics.groups(bare) for r in rows}
    assert not any(k.startswith("lap_") for k in keys), keys
    assert metrics.lap_report(bare) is None


def test_the_slotted_family_s_lap_is_presentable_too():
    """The two-element family times the same lap and must report it the same
    way — the report reads the BREAKDOWN, so a family that spells a lap key
    differently would be silently unreportable.

    Catches: keying the reader off the problem name, or off a single-element
    breakdown field, either of which leaves the slotted family's lap invisible
    on the page it is measured for.
    """
    built = api.PROBLEM_SPECS[MULTI].build(
        {}, {"car_track": "synthetic", "car_objective": "laptime",
             "track_points": 2}, None)
    out = built.evaluate(_mid(built))
    assert out["feasible"], out["reason"]
    rep = metrics.lap_report(out)
    assert rep is not None
    assert sum(s["t_s"] for s in rep["segments"]) == pytest.approx(
        out["lap_time_s"])
    assert 0 < len(rep["points"]) <= 2
    shown = {r["key"] for _t, rows in metrics.groups(out) for r in rows}
    for key in ("lap_time_s", "lap_V_top_ms"):
        assert key in shown, key
    # ...and it is the SAME circuit as the single-element family's, which is
    # what makes a lap timed on one comparable with a lap timed on the other
    single = api.PROBLEM_SPECS[SINGLE].build(
        {}, {"car_track": "synthetic", "car_objective": "laptime"}, None)
    assert rep["summary"]["track"] == metrics.lap_report(
        single.evaluate(_mid(single)))["summary"]["track"]


def test_the_balance_is_reported_against_the_window_that_judges_it():
    """0.49 is unreadable on its own: the window's CENTRE is derived from the
    car's static weight split, so the number only means something beside it.

    Catches: reporting ``aero_balance`` and dropping the window, which is the
    state this page was in — the Spec existed and the window did not.
    """
    out = _lap_breakdown()
    bal = metrics.lap_report(out)["balance"]
    assert bal is not None
    assert bal["value"] == pytest.approx(out["aero_balance"])
    lo, hi = bal["window"]
    assert lo < hi
    # DERIVED, not pinned: the window is centred on the static front weight
    # fraction (cartrack equation 7) with the module's own half-width
    from aerobo import cartrack

    assert 0.5 * (lo + hi) == pytest.approx(bal["static_front_frac"])
    assert hi - lo == pytest.approx(2.0 * cartrack.BALANCE_HALF_WIDTH)


def test_the_results_page_draws_the_lap():
    """...and the panel really renders, with the circuit's name and a
    per-speed row on it.

    Catches: adding ``metrics.lap_report`` and never calling it — the exact
    shape of the gap this session closed one layer up.
    """
    from gui.v3 import session
    from gui.v3.app import assemble

    out = _lap_breakdown(points=3)
    ctx = assemble("track")
    S = ctx.S
    S["wing"]["choices"].update(car_track="synthetic", car_objective="laptime")
    session.apply_choices(S)
    S["run"]["record"] = {
        "breakdown": out, "best_x": [1.6], "param_labels": ["b_m"],
        "bounds": [[1.2, 2.0]], "n_evals": 1, "feasible": True,
        "best_score": out["score"], "wall_time_s": 0.0,
        "config": {"problem_name": SINGLE, "optimiser": "bo", "budget": 1,
                   "seed": 0, "flags": {"car_track": "synthetic",
                                        "car_objective": "laptime"}}}
    ctx.render("results", "summary")
    text = " ".join(_texts(ctx.views[("results", "summary")]))
    assert "not a real circuit" in text, text
    assert "limited by" in text, text
    assert "weight in the lap" in text, text
    # the balance is on the page beside its window
    assert "aero balance" in text, text


def test_the_answer_leads_with_the_number_the_run_maximised():
    """A headline that does not lead with what was SCORED is a headline about
    a different question.

    On a lap run the answer IS the lap time. Until this session the headline
    led with C_Z — the first of the car family's specialities in catalogue
    order — the lap time sat six rows down, and the shells' hero readout fell
    back to the raw score, which on a lap is MINUS the lap time: the biggest
    number on the page read "-61.429" beside a "lap time 61.429 s" lower in
    the same panel. One quantity, two spellings, one of them negative.

    Catches: dropping ``OBJECTIVE_LEAD``; hoisting the row AFTER the headline
    is truncated to three specialities (the row that is the answer would be
    the one the truncation dropped); and leaving the V3 hero on the raw score.
    """
    from gui.v3 import session
    from gui.v3.app import assemble

    out = _lap_breakdown()
    rows = metrics.headline(out)
    assert rows[0]["key"] == "lap_time_s", [r["key"] for r in rows]
    assert rows[0]["unit"] == "s"
    assert rows[0]["value"] == pytest.approx(out["lap_time_s"])
    # ...and it is said ONCE: the negated score is not on the page beside it
    assert sum(1 for r in rows if r["key"] == "lap_time_s") == 1

    # the control: a run that maximised something else leads with that
    plain = api.PROBLEM_SPECS[SINGLE].build({}, {}, None)
    bare = plain.evaluate(_mid(plain))
    lead = metrics.objective_row(bare)
    assert lead is not None and lead["key"] == "efficiency"
    assert metrics.headline(bare)[0]["key"] == "efficiency"

    # ...and the V3 hero readout is that row rather than the raw score
    ctx = assemble("track")
    S = ctx.S
    S["wing"]["choices"].update(car_track="synthetic",
                                car_objective="laptime")
    session.apply_choices(S)
    S["run"]["record"] = {
        "breakdown": out, "best_x": [1.6], "param_labels": ["b_m"],
        "bounds": [[1.2, 2.0]], "n_evals": 1, "feasible": True,
        "best_score": out["score"], "wall_time_s": 0.0,
        "config": {"problem_name": SINGLE, "flags": {}}}
    ctx.render("results", "summary")
    texts = _texts(ctx.views[("results", "summary")])
    assert "lap time" in texts, texts
    assert "objective" not in texts, texts
    # the negated score must not be on the page at all
    assert not any(str(t).startswith("-61.4") for t in texts), texts


# ============================ 6. the DRAG BUDGET decision, as an outcome

def test_the_pair_is_refused_by_neither_and_no_card_asks_it():
    """THE DECISION. A drag ceiling and a lap are both honoured; the pair is
    refused by neither and not silently resolved.

    It is now an ENGINE claim and not a card one, because the circuit left
    every card. What used to be here — the card's own note that a chosen lap
    already prices drag physically, so a typed ceiling is a SECOND,
    independent limit on the same drag — went with the question that could
    raise it. The pairing itself is unchanged and still statable from a
    script or a saved session, which is what this asserts.

    Catches: refusing the pair (which would refuse a real engineering
    constraint — a drag budget decided elsewhere does not stop being one
    because the wing is scored on a lap:
    `a-calibration-is-a-default-not-a-ban`).
    """
    from gui.v3 import session
    from gui.v3.app import assemble

    ch = _track(car_track="synthetic", car_objective="laptime",
                car_drag_budget_n=60.0)
    name = v1.derive_problem(ch)[0]
    flags = v1.car_flags(ch, problem_name=name)
    assert flags["car_track"] == "synthetic"
    assert flags["drag_budget_n"] == 60.0
    assert api.check_flags(name, flags) is None
    built = api.PROBLEM_SPECS[name].build({}, flags, None)
    assert built.problem.track_spec is not None
    assert built.problem.drag_budget_n == 60.0
    assert built.problem.objective == "laptime"
    assert "drag force margin" in built.problem.constraint_labels

    # ...and NO CARD asks the circuit any more, in either view
    ctx = assemble("track")
    ctx.S["wing"]["choices"]["car_endplates"] = False
    session.apply_choices(ctx.S)
    ctx.render("wing", "type")
    ctx.render("wing", "box")
    seen = " ".join(_texts(ctx.views[("wing", "type")])
                    + _texts(ctx.views[("wing", "box")]))
    for gone in ("Circuit", "synthetic", "Speeds sampled", "lap time",
                 "two different limits on the same drag"):
        assert gone not in seen, gone
    assert "laptime" not in v1.car_objective_options(
        ctx.S["wing"]["choices"])


def test_the_card_says_nothing_about_a_pair_that_is_not_stated():
    """The control. A note that appears whether or not it applies is noise.

    Catches: drawing the pairing note off the circuit alone, so every lap run
    is lectured about a ceiling nobody typed.
    """
    from gui.v3 import session
    from gui.v3.app import assemble

    for choices in ({"car_track": "synthetic", "car_objective": "laptime"},
                    {"car_drag_budget_n": 60.0}):
        ctx = assemble("track")
        ctx.S["wing"]["choices"].update(choices)
        session.apply_choices(ctx.S)
        ctx.render("wing", "type")
        card = " ".join(_texts(ctx.views[("wing", "type")]))
        assert "two different limits on the same drag" not in card, choices


def _refused_record(objective: str, label: str = "drag force margin") -> dict:
    """A run that found nothing, with ``label`` the constraint furthest from
    ever being satisfied."""
    return {"config": {"flags": {"car_objective": objective,
                                 "car_track": "synthetic",
                                 "drag_budget_n": 5.0}},
            "best_x": None, "feasible": False, "n_evals": 4, "n_feasible": 0,
            "eval_g": [[-0.9, 0.5], [-0.8, 0.4], [-0.95, 0.6], [-0.7, 0.3]],
            "eval_y": [1.0, 1.1, 0.9, 1.2],
            "_labels": [label, "deflection margin"]}


def test_the_no_solution_path_names_the_ceiling_when_it_is_the_binding_one():
    """The other half of the decision: a user handed "no design satisfied
    every constraint" over a limit their own objective was already charging
    for will widen a design-box row, pay for another search and get the same
    refusal.

    Catches: leaving the no-solution card generic, which is the state that
    lets the calibration's refusal read as the wing's — the failure
    RESULTS_SESSION64_CARSECTION section 7 measured (no feasible design in 24
    searches of 4000 evaluations, because of the allowance and not the wing).
    """
    rec = _refused_record("laptime")
    rep = diagnose.infeasibility_report(rec,
                                        constraint_labels=rec["_labels"])
    assert rep["binding"]["label"] == "drag force margin"
    assert rep["already_priced"]
    text = " ".join(t for t, _lvl in diagnose.infeasibility_lines(rep))
    assert "drag ceiling you stated" in text, text
    assert "already prices drag" in text, text
    # the coefficient allowance gets its own sentence, naming what makes it
    # worse on this family
    rec2 = _refused_record("laptime", label="drag budget margin")
    rep2 = diagnose.infeasibility_report(rec2,
                                         constraint_labels=rec2["_labels"])
    text2 = " ".join(t for t, _lvl in diagnose.infeasibility_lines(rep2))
    assert "COEFFICIENT drag allowance" in text2, text2


def test_the_no_solution_path_says_nothing_extra_without_a_lap():
    """The control, twice over: a non-lap objective, and a lap whose binding
    limit is not the drag ceiling.

    Catches: attaching the sentence to the drag margin unconditionally, so
    every constrained car run that fails is told its objective prices drag —
    which is true only when the objective is a lap.
    """
    rec = _refused_record("efficiency")
    rep = diagnose.infeasibility_report(rec,
                                        constraint_labels=rec["_labels"])
    assert rep["already_priced"] is None
    text = " ".join(t for t, _lvl in diagnose.infeasibility_lines(rep))
    assert "already prices drag" not in text
    # a lap whose binding constraint is the DEFLECTION margin says nothing
    # either: the sentence is about the constraint that stopped the run
    rec2 = _refused_record("laptime")
    rec2["eval_g"] = [[0.5, -0.9], [0.4, -0.8], [0.6, -0.95], [0.3, -0.7]]
    rep2 = diagnose.infeasibility_report(rec2,
                                         constraint_labels=rec2["_labels"])
    assert rep2["binding"]["label"] == "deflection margin"
    assert rep2["already_priced"] is None


# ================================================ 7. what the lap COSTS

def test_a_stale_objective_does_not_take_the_whole_card_down():
    """A STATE HOLDING ``laptime`` MUST STILL RENDER.

    ``ui.select`` raises ValueError on a value outside its options, and in
    this shell that exception is caught one level up and replaces the ENTIRE
    view with "this view failed to render" — so one stale key costs the
    reader the mount, the size, the chord law and the objective, not just the
    row that was wrong.

    It is reachable: the circuit left the card, so ``car_objective_options``
    no longer offers ``laptime`` anywhere, while a saved session, a restored
    state or a shell whose ``apply_choices`` does not run
    ``_normalise_car_lap`` can still be holding it. Measured before the
    clamp: the card died exactly here.

    Catches: dropping the clamp in ``_car_controls`` and trusting the
    normaliser to have run.
    """
    from gui.v3 import session
    from gui.v3.app import assemble

    for extra in ({"car_objective": "laptime"},
                  {"car_objective": "laptime", "car_track": "synthetic"}):
        ctx = assemble("track")
        ctx.S["wing"]["choices"].update(extra)
        session.apply_choices(ctx.S)
        ctx.render("wing", "type")
        card = " ".join(_texts(ctx.views[("wing", "type")]))
        assert "failed to render" not in card, (extra, card[:200])
        # ...and the row shows the family's own default instead of the value
        # it cannot offer
        assert "Maximise" in card
        assert api.CAR_DEFAULT_OBJECTIVE in v1.car_objective_options(
            ctx.S["wing"]["choices"])


def test_the_sampling_count_is_no_longer_a_card_question():
    """Each representative speed is a full re-solve of the wing, so the count
    multiplies what every candidate costs — which is why it was never allowed
    to be a silent knob.

    It is not a card question at all now: the circuit left, and a sampling
    count with no lap to sample is refused by name
    (``api._car_track_kwargs``), so a field for it would be asking about
    something the card can no longer produce. The COUNT is still the
    dataclass's own default and still statable through api, which the two
    tests either side of this one assert.
    """
    from gui.v3 import session
    from gui.v3.app import assemble

    ctx = assemble("track")
    ctx.S["wing"]["choices"]["car_endplates"] = False
    session.apply_choices(ctx.S)
    ctx.render("wing", "type")
    card = " ".join(_texts(ctx.views[("wing", "type")]))
    assert "Speeds sampled" not in card
    assert "car_track_points" not in card
    # ...and the key is off the list that protects a FIELD from its own
    # handler, because there is no field left to protect
    from gui.v3.stages import wing as v3wing
    assert "car_track_points" not in v3wing.VALUE_KEYS
    assert "car_track_points" not in v3wing.BOX_VALUE_KEYS
    # the default is still read off the dataclass, never pasted
    from aerobo.carwing import CarWingProblem
    assert v1.car_default_track_points() == int(CarWingProblem.track_points)


def test_a_sampling_count_below_one_is_refused_by_name():
    """The field is bounded at 1 and says so; a count that reaches the family
    anyway is refused IN CONTRACT rather than clamped or crashed.

    ``lo=`` on a ``_num_row`` is an HTML ``min`` — a browser hint, not a
    clamp — so a preset, a saved record or a pasted value can still carry a
    zero. The right answer there is the family's own named refusal, because a
    silently-clamped number is its own surprise (``_num_row``'s docstring
    says so for the blend fraction, which was fixed the same way).

    Catches: clamping the count inside ``car_flags`` (the user's number would
    become a different number with nothing on screen saying so), and dropping
    it silently (the run would sample a different lap from the card).
    """
    ch = _track(car_track="synthetic")
    ch["car_track_points"] = 0
    flags = v1.car_flags(ch)
    assert flags["track_points"] == 0          # not clamped, not dropped
    name = v1.derive_problem(ch)[0]
    api.check_flags(name, flags)               # it is a declared flag
    with pytest.raises(ValueError, match="track_points must be >= 1"):
        api.PROBLEM_SPECS[name].build({}, flags, None)


def test_the_default_sampling_count_is_read_off_the_dataclass():
    """Never pasted: a family that re-chooses its sampling must not leave a
    constant in the shell saying otherwise.

    Catches: writing 4 into the card.
    """
    from aerobo.carwing import CarWingProblem

    assert v1.car_default_track_points() == CarWingProblem.track_points
    # ...and the card's blank field really means that number
    ch = _track(car_track="synthetic")
    assert ch["car_track_points"] is None
    built = api.PROBLEM_SPECS[SINGLE].build({}, v1.car_flags(ch), None)
    assert built.problem.track_points == v1.car_default_track_points()
