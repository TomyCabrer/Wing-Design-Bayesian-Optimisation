"""What V3 ASKS, and what it opens on.

Two rules, one per half of this file:

* **one question, one place.** The operating point is stage 1's — speed and
  altitude are typed there and every Reynolds number, design point and trim
  target downstream is quoted at them — so stage 3 has no flight-state
  control. The MODIFIER is untouched: the registry still carries every
  ``+ free flight state`` twin and V1/V2 still offer it. V3 does not ask.
* **a surface opens designed.** Adding a second surface (a tail, an
  elevator) adds a SURFACE, not a fitting, so its planform — taper, aspect
  ratio, washout — is designed wherever a solver can design it, and the
  published rectangle at AR 4 is the fallback for a family with no
  designed-tail variant.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


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


# ------------------------------------------------- the flight state is gone
def test_a_fresh_v3_session_pins_the_flight_state():
    from gui.v3 import session

    for medium in ("air", "water", "track"):
        S = session.make_session(medium)
        assert S["wing"]["choices"]["flight"] == "fixed", medium
        assert "flight state" not in S["wing"]["problem"], S["wing"]["problem"]
    assert session.V3_PINNED_CHOICES["flight"] == "fixed"


def test_stage_three_offers_no_flight_state_control(capsys):
    from gui.v3.app import assemble

    ctx = assemble("air")
    ctx.render("wing", "type")
    texts = _texts(ctx.views[("wing", "type")])
    assert not any("speed + altitude free" in t for t in texts), texts
    assert not any(t.strip() == "flight state" for t in texts), texts
    # the controls that DO belong there are still there
    assert any("polynomial chord law" in t for t in texts), texts
    err = capsys.readouterr().err
    assert "Traceback" not in err, err


def test_the_modifier_itself_is_untouched():
    """V3 not asking is a MENU decision — the freedom still exists."""
    from aerobo import api
    from gui import nice_app as v1

    assert "flight" in api.MODIFIERS
    assert api.add_modifier("trim wing", "flight") == "mission wing"
    ch = dict(v1.BUILDER_DEFAULTS, flight="free")
    assert v1.derive_problem(ch)[0] == "mission wing"


# --------------------------------------------- a second surface is designed
def test_a_newly_added_surface_opens_designed():
    from gui import nice_app as v1

    for medium in ("air", "water"):
        ch = v1.start_choices(medium=medium)
        assert v1.tail_design_start(ch) == "planform", medium
        ch["tail_design"] = v1.tail_design_start(ch)
        ch["tail"] = True
        v1.normalise_choices(ch, keep="tail")
        name, _notes = v1.derive_problem(ch)
        assert "designed" in name, (medium, name)


def test_a_family_with_no_designed_tail_opens_on_the_rectangle():
    """Asked of the registry, never assumed: the car's rear wing has nothing
    behind it, so the default has to fall back rather than hold a value its
    own control cannot offer."""
    from gui import nice_app as v1

    ch = v1.start_choices(medium="track")
    assert v1.tail_design_start(ch) == v1.BUILDER_DEFAULTS["tail_design"]


def test_the_neutral_reset_is_still_the_rectangle():
    """BUILDER_DEFAULTS is the value every control resets to and the one
    ``choices_consistent`` trusts without asking the registry, so the
    designed default must NOT live there (the same split the chord law
    keeps between BUILDER_DEFAULTS and BUILDER_START)."""
    from gui import nice_app as v1

    assert v1.BUILDER_DEFAULTS["tail_design"] == "fixed"
    assert v1.BUILDER_START["tail_design"] == "fixed"
    assert v1.TAIL_DESIGN_START == "planform"


def test_switching_the_surface_on_in_stage_one_designs_it(capsys):
    from gui.v3.app import assemble

    for medium, word in (("air", "designed tail"),
                         ("water", "designed elevator")):
        ctx = assemble(medium)
        # AIR OPENS WITH THE SURFACE ON now (session.V3_START_CHOICES), so
        # the switch is exercised from off — which is the transition this
        # test is about, whichever side the shell happens to open on
        ctx.act("set_second_surface", False)
        ctx.act("set_second_surface", True)
        name = ctx.S["wing"]["problem"]
        assert word in name, (medium, name)
        assert ctx.S["wing"]["choices"]["tail_design"] == "planform"
        # ...and the whole shell still renders on it
        ctx.render("wing", "type")
        ctx.render("wing", "box")
        err = capsys.readouterr().err
        assert "Traceback" not in err, err


def test_the_published_wing_only_run_is_ONE_SWITCH_away_and_untouched():
    """V3 opens on a whole aeroplane now — a tail, and the lattice family
    that can carry a wing dihedral, because a wing alone has no spiral to be
    stable (``session.V3_START_CHOICES``). The published wing-only run did
    not go anywhere: it is what the second-surface switch turns back to, and
    nothing this shell does to it may perturb it.
    """
    from gui.v3 import config, session
    from gui.v3.app import assemble

    S = session.make_session("air")
    # ...and the second surface opens DESIGNED, the same rule a surface the
    # user switches on follows (nice_app.tail_design_start): the opening
    # aeroplane used to skip it, which left its tail a published rectangle
    # with no planform rows in the box and its tip-device menu greyed out
    assert S["wing"]["problem"] == \
        "tail [free height, designed tail] + free chord law"

    ctx = assemble("air")
    ctx.act("set_second_surface", False)
    d = config.cfg_dict(ctx.S)
    assert d["problem_name"] == "wing (free chord law)"
    # ...up to the physics V3 ANSWERS instead of asking, which it declares in
    # one place and this reads rather than restating
    assert config.physics_flags(d) == config.stated_physics(d["problem_name"])
    assert d["bounds_overrides"] is None
    # and NOTHING of the opening aeroplane's own is left on it: the body and
    # the dihedral are flags of a family this one is not
    assert not [k for k in d["flags"] if k in ("wing_dihedral_deg",
                                               "fuselage_diameter_m")], d


# ------------------------------------------------- the fin IS somebody's now
def test_stage_three_states_the_fin_charge_and_offers_no_switch(capsys):
    """Two replacements deep, and both previous names were about a control.

    First ``test_stage_three_offers_no_fin_drag_switch`` — reason: "the fin
    is not a surface this package models". Every clause of that became false
    (``fin.py`` sizes it, ``vlm.VerticalSurface`` panels it,
    ``cad.fin_surface`` lofts it, ``flightmodel`` flies it and takes all its
    yaw stiffness from it). Then ``..._offers_the_fin_drag_switch``, on the
    argument that the charge was an allowance the user should decide.

    The user decided otherwise: "I don't want it to be an option, makes it
    more complicated for the user unnecessarily... do as the horizontal
    tail." A tailplane's profile drag is not a question, and neither is
    this. So the card STATES the charge and carries no control.
    """
    from gui.v3.app import assemble

    ctx = assemble("air")
    ctx.act("set_second_surface", True)
    ctx.render("wing", "type")
    texts = _texts(ctx.views[("wing", "type")])
    assert not any("charge the fin" in t for t in texts), texts
    assert any("is CHARGED" in t for t in texts), texts
    # the second surface's OTHER controls are untouched
    assert any("conventional (aft, on the fuselage)" in t for t in texts), \
        texts
    assert any("arm is a design variable" in t for t in texts), texts
    err = capsys.readouterr().err
    assert "Traceback" not in err, err


def test_the_v1_fin_CHOICE_is_pinned_and_dead():
    """``tail_fin_drag`` is the V1/V2 builder CHOICE, still pinned False in
    every V3 session — and now dead everywhere, not merely unasked: nothing
    maps it to a flag, because there is no flag. The fin's parasite drag is
    charged unconditionally, so no shell can decide it and no config can
    carry a decision about it."""
    from aerobo import api
    from gui import nice_app as v1
    from gui.v3 import config, session
    from gui.v3.app import assemble

    assert session.V3_PINNED_CHOICES["tail_fin_drag"] is False
    for medium in ("air", "water", "track"):
        S = session.make_session(medium)
        assert S["wing"]["choices"]["tail_fin_drag"] is False, medium

    # the V1 builder no longer emits a flag for it, even when the choice is on
    ch = dict(v1.BUILDER_DEFAULTS, tail=True, tail_fin_drag=True)
    assert "charge_fin_drag" not in v1.tail_flags(ch)

    ctx = assemble("air")
    ctx.act("set_second_surface", True)
    name = ctx.S["wing"]["problem"]
    assert "charge_fin_drag" not in api.PROBLEM_SPECS[name].flags
    ctx.act("set_choice", "tail_arm", "fixed")
    flags = config.cfg_dict(ctx.S)["flags"]
    assert "l_t_m" in flags, flags
    assert "charge_fin_drag" not in flags, flags


def test_the_v_tail_note_reports_a_verdict_and_not_an_allowance(capsys):
    """The note SAID the comparison was rigged — "the fin a V-tail exists to
    delete is not a modelled surface here and costs nothing, so this layout
    can only show the COST of dihedral and never the benefit". It is not
    rigged any more: every other layout pays for its fin, so the note has to
    stop apologising and report what the numbers say."""
    from gui.v3.app import assemble
    from gui.v3.stages import wing as wing_stage

    note = wing_stage.V_TAIL_NOTE_V3
    assert "costs nothing" not in note
    assert "no fin to drag" in note

    ctx = assemble("air")
    ctx.act("set_second_surface", True)
    ctx.S["wing"]["choices"]["tail_type"] = "v_tail"
    ctx.render("wing", "type")
    assert any("no fin to drag" in t
               for t in _texts(ctx.views[("wing", "type")]))
    err = capsys.readouterr().err
    assert "Traceback" not in err, err


def test_the_fin_drag_model_itself_is_untouched():
    """Deleting the SWITCH must not delete the MODEL: the same Raymer
    build-up sizes and charges the surface, it simply always runs now."""
    import numpy as np
    from aerobo import api, tail

    assert "charge_fin_drag" not in api.TAIL_CONFIG_KEYS
    assert tail.vtail_cd0(10.0, 5.5) > 0.0
    x = np.array([0.6, 1.0, -2.0, 1.5, 5.0])
    assert tail.evaluate_tail(x, tail.TailProblem())["cd0_fin"] > 0.0


# --------------------------------------- the surface is all-moving, always
def test_stage_three_offers_no_control_type_toggle(capsys):
    """A hinged elevator trims to the SAME state as an all-moving surface —
    the flap enters the boundary condition as the same uniform RHS shift, so
    the re-parameterisation is exact. Its only real effect is a deflection
    gate that nothing in the box can trip, so the toggle asked for a decision
    that could not change an answer."""
    from gui.v3.app import assemble

    ctx = assemble("air")
    ctx.act("set_second_surface", True)
    ctx.render("wing", "type")
    texts = _texts(ctx.views[("wing", "type")])
    assert not any("all-moving stabilator" in t for t in texts), texts
    assert not any("fixed stabiliser + elevator" in t for t in texts), texts
    assert not any("elevator chord" in t for t in texts), texts
    err = capsys.readouterr().err
    assert "Traceback" not in err, err


def test_no_v3_session_flies_a_hinged_elevator():
    from gui.v3 import config, session
    from gui.v3.app import assemble

    assert session.V3_PINNED_CHOICES["tail_control"] == "stabilator"
    for medium in ("air", "water", "track"):
        S = session.make_session(medium)
        assert S["wing"]["choices"]["tail_control"] == "stabilator", medium

    ctx = assemble("air")
    ctx.act("set_second_surface", True)
    ctx.act("set_choice", "tail_arm", "fixed")
    flags = config.cfg_dict(ctx.S)["flags"]
    assert "control" not in flags, flags
    assert "elevator_chord_frac" not in flags, flags
    # the field that fed it is gone from the focus-guard list too
    from gui.v3.stages import wing as wing_stage
    assert "tail_elevator_chord" not in wing_stage.VALUE_KEYS


def test_the_elevator_changes_nothing_the_optimiser_can_see():
    """The reason the toggle went. Not an opinion — the equality is exact,
    and the gate that COULD have differed needs an elevator under 5% chord
    before it fails a design the tail box can build."""
    import numpy as np

    from aerobo import tail

    x = np.array([0.6, 1.0, -2.0, 1.5, 5.0])
    stab = tail.evaluate_tail(x, tail.TailProblem())
    elev = tail.evaluate_tail(x, tail.TailProblem(control="elevator"))
    assert elev["LoD"] == stab["LoD"]          # exactly
    assert elev["i_t_deg"] == stab["i_t_deg"]

    # the gate's reach, measured over the box rather than assumed
    box = tail.TailProblem().bounds
    rng = np.random.default_rng(0)
    X = rng.uniform(box[:, 0], box[:, 1], size=(120, box.shape[0]))
    worst = max(abs(tail.evaluate_tail(xi, tail.TailProblem())["i_t_deg"])
                for xi in X)
    assert worst < 15.0                        # i_t_max_deg, never approached
    # |delta_e| = |i_t| / tau, so the 25 deg limit is reached only if
    # tau < worst/25. Even a 10% elevator clears that with room to spare:
    assert tail.flap_effectiveness(0.10) > worst / 25.0
    # ...and at the card's default the deflection gate is unreachable in
    # principle, not merely in this box: i_t_max_deg fails first for any
    # c_e/c_t whose tau exceeds 15/25
    assert tail.flap_effectiveness(0.30) > 15.0 / 25.0


# ------------------- the track card owns the QUESTION, the box owns the SIZE
#
# Three things moved here, and they are one decision seen from three sides.
# The car's reference AREA used to be configuration (a switch that picked a
# different problem NAME) and its SPAN band was four typed fields on the card,
# while the design box carried b_m and S_m2 rows of its own — so the same
# question had two answers, and they disagreed in both directions. And the
# objective was fixed at the downforce COEFFICIENT under a drag ALLOWANCE the
# answer simply spent.
#
# Now: both dimensions are always designed, their bands are rows of the design
# box and nowhere else, the score is CZ/CD, and a drag ceiling is an optional
# limit under the box.

def test_stage_three_asks_the_car_what_to_maximise_and_not_how_to_budget(
        capsys):
    from gui.v3.app import assemble

    ctx = assemble("track")
    ctx.render("wing", "type")
    texts = _texts(ctx.views[("wing", "type")])
    joined = " ".join(texts).replace("  ", " ")
    # what it DOES ask
    assert "efficiency CZ/CD" in joined, texts
    assert "downforce, in newtons" in joined, texts
    # ...and what it must NOT: a coefficient against a designed area, a drag
    # MENU, a size question the design box already owns
    assert "downforce coefficient CZ" not in joined, texts
    assert "drag COEFFICIENT allowance" not in joined, texts
    assert "design the reference area too" not in joined, texts
    for gone in ("Span, no less than", "Span, no more than",
                 "Area, no less than", "Area, no more than"):
        assert gone not in joined, (gone, texts)
    err = capsys.readouterr().err
    assert "Traceback" not in err, err


def test_the_car_limits_are_beside_the_law_and_the_size_rows_are_in_the_box(
        capsys):
    """The two halves of "one question, one place", on one card each.

    THE LIMITS MOVED. They used to be drawn under the design box with the
    chord and tail limits — a defensible home for a bound and the wrong one
    for these two, because they are not independent bounds: a car objective
    is one half of a trade and the limit is the other half, and the two were
    a whole TAB apart. Choosing "efficiency" and looking for where to say how
    much downforce is worth having found a sentence pointing elsewhere.

    So the claim is now that they are beside the Maximise select AND NOT in
    the box: a test that only checked the new home would pass just as well
    if they were drawn in both, which is the defect the move exists to avoid.
    The size rows are unmoved and still checked here — they are the other
    half of the sentence.
    """
    from gui.v3.app import assemble
    from gui.v3 import config

    ctx = assemble("track")
    ctx.render("wing", "type")
    ctx.render("wing", "box")
    beside = " ".join(_texts(ctx.views[("wing", "type")]))
    box = " ".join(_texts(ctx.views[("wing", "box")]))
    for field in ("Drag, no more than", "Downforce, at least"):
        assert field in beside, (field, beside[-400:])
        assert field not in box, (field, box[:400])
    assert "Maximise" in beside
    # ...and the SIZE is a pair of ordinary rows of the design box, which is
    # where a BAND belongs and where these two never were
    rows = config.effective_bounds(ctx.S)
    assert "b_m" in rows and "S_m2" in rows, sorted(rows)
    assert "Traceback" not in capsys.readouterr().err


def test_a_typed_size_row_is_the_band_the_run_searches():
    """The defect this collapse fixed, end to end through the shell.

    Widening b_m past the family's own band used to move the SAMPLER and
    leave the problem's box where it was, so every draw above 2.0 m came back
    ``feasible=False, reason='bounds violation'``, score -100 — a run made
    entirely of refusals. And a band typed on the card moved the search while
    the box on screen still showed the family default.
    """
    import pytest

    from aerobo import api
    from gui.v3.app import assemble
    from gui.v3 import config

    ctx = assemble("track")
    S = ctx.S
    S["wing"]["bounds"]["b_m"] = [1.2, 3.0]
    S["wing"]["bounds"]["S_m2"] = [0.20, 0.30]
    built = api.PROBLEM_SPECS[S["wing"]["problem"]].build(
        None, config.flags(S), config.bounds_overrides(S))
    lbl = list(built.param_labels)
    for row in ("b_m", "S_m2"):
        shown = list(config.effective_bounds(S)[row][0])
        assert list(built.bounds[lbl.index(row)]) == shown, row
        assert list(built.problem.bounds[lbl.index(row)]) == shown, row
    # ...and a design out at 2.6 m is a real design, not a refusal
    x = 0.5 * (built.bounds[:, 0] + built.bounds[:, 1])
    x[lbl.index("b_m")] = 2.6
    out = built.evaluate(x)
    assert out["feasible"], out.get("reason")
    assert out["b_m"] == pytest.approx(2.6)


def test_the_track_card_reaches_the_family_it_names_and_its_flags():
    """The card's choices have to BUILD — a control that resolves to a family
    the flags are then refused by is worse than no control."""
    from aerobo import api
    from gui import nice_app as v1

    # the shell OPENS on the polynomial chord law, so the family it really
    # builds is the chord twin and the vector carries its three coefficients.
    # Pinned to "fixed" here so the dims below say one thing at a time; the
    # twin is checked on its own in the last case.
    # ...and to the PYLON mount, so the base cases below are the plain fence
    # family. ``car_endplates`` is the mount key: True (what the card opens
    # on) says the plates carry the car and picks the designed-plate family,
    # which is the case listed on its own below.
    ch = dict(v1.start_choices(medium="track"), chord="fixed",
              car_endplates=False)
    # EVERY PYLON CASE CARRIES TWO MOUNT MARGINS, and that is the mount showing
    # up in the answer rather than in a menu. The strut has to get from the
    # wing down to the deck ("pylon reach margin", the same signed constraint
    # the designed PLATE answers as "endplate reach margin" in its own case
    # below) — and the PLATE beside it has to stop short of that same deck
    # ("mount clearance margin"), because past the deck there is bodywork
    # rather than air. Raise the wing and both of them grow.
    #
    # The clearance one is LAST in every tuple below, which is the contract it
    # was added under: a margin that is newer than another may not push it to
    # a different index, because gui/diagnose.py names margins BY INDEX.
    cases = {
        # (choices) -> (problem, dim, objective, margins)
        (): ("car rear wing", 8, "efficiency",
             ("deflection margin", "pylon reach margin",
              "mount clearance margin")),
        (("car_objective", "downforce"),): (
            "car rear wing", 8, "downforce",
            ("deflection margin", "pylon reach margin",
             "mount clearance margin")),
        # a stated ceiling is ONE more margin, not two: typing a force must
        # not revive the coefficient allowance beside it
        (("car_drag_budget_n", 90.0),): (
            "car rear wing", 8, "efficiency",
            ("drag force margin", "deflection margin",
             "pylon reach margin", "mount clearance margin")),
        (("car_downforce_min_n", 400.0),): (
            "car rear wing", 8, "efficiency",
            ("deflection margin", "downforce floor margin",
             "pylon reach margin", "mount clearance margin")),
        (("car_endplates", True),): (
            "car rear wing + endplates", 11, "efficiency",
            ("wing deflection margin", "endplate deflection margin",
             "endplate reach margin")),
        # ...and it all composes with the chord law the shell opens on
        (("chord", "free"),): (
            "car rear wing + free chord law", 11, "efficiency",
            ("deflection margin", "pylon reach margin",
             "mount clearance margin")),
    }
    for kv, (want, dim, obj, margins) in cases.items():
        c = dict(ch, **dict(kv))
        name, _notes = v1.derive_problem(c)
        flags = v1.car_flags(c, None, name)
        assert name == want, (kv, name)
        api.check_flags(name, flags)          # raises if a flag is not read
        built = api.PROBLEM_SPECS[name].build({}, flags, None)
        assert built.dim == dim == len(built.param_labels), kv
        assert built.problem.objective == obj, kv
        assert built.problem.constraint_labels == margins, kv


def test_an_untouched_track_card_sends_the_mount_and_the_plate_it_implies():
    """Every control on this card defaults to what the family already does, so
    an untouched card is the family's own configuration exactly — and the
    size, which the card no longer asks, is not smuggled out as a flag.

    The card OPENS on the endplate mount, which is what a rear wing is bolted
    to the car by. That is not a flag either: it picks the designed-plate
    FAMILY, whose own default section travels with it. Nothing else is sent —
    no chord band, no blend, no cant, no size.
    """
    from gui import nice_app as v1

    ch = v1.start_choices(medium="track")
    assert v1.car_flags(ch) == {"mount": "tips", "section": "shaped"}
    from aerobo.endplate import CarWingEndplateProblem
    assert CarWingEndplateProblem.section == "shaped"      # not a paste
    # the chord law is the shell's own opening default (session 28), so the
    # untouched family is the twin — and the point stands: none of these
    # controls has added a flag or moved the family
    assert v1.derive_problem(ch)[0] == \
        "car rear wing + endplates + free chord law"
    assert v1.derive_problem(dict(ch, chord="fixed"))[0] == \
        "car rear wing + endplates"
    # ...and the PYLON answer is the plain fence family, sending the struts
    # and nothing about a plate
    pyl = dict(ch, car_endplates=False, chord="fixed")
    assert v1.derive_problem(pyl)[0] == "car rear wing"
    flags = v1.car_flags(pyl, None, "car rear wing")
    assert "section" not in flags and flags["mount_kind"] == "pylon"


def test_nothing_is_budgeted_until_a_limit_is_typed():
    """The change of substance: an allowance handed to a maximiser is the
    number that picks the answer, not one that bounds it.

    The family used to open on CD 0.11 (restated as 81.5 N once the area
    moved) and the answer spent it — on both certified optima the drag margin
    sits near-binding while binding almost nowhere else in the box. So the
    default carries NO drag margin, and a typed ceiling adds exactly one.
    """
    import pytest

    from aerobo import api
    from gui import nice_app as v1
    from aerobo.carwing import published_drag_budget_n

    ch = v1.start_choices(medium="track")
    flags = v1.car_flags(ch)
    assert "drag_budget_n" not in flags and "CD_budget" not in flags
    built = api.PROBLEM_SPECS["car rear wing"].build({}, flags, None)
    assert built.problem.CD_budget is None
    assert built.problem.drag_budget_n is None
    assert not [m for m in built.problem.constraint_labels if "drag" in m]

    # ...and the allowance the family used to carry is still derivable, so
    # the number is retired rather than lost
    typed = v1.car_flags(dict(ch, car_drag_budget_n=published_drag_budget_n()))
    assert typed["drag_budget_n"] == pytest.approx(81.5, rel=1e-3)
    capped = api.PROBLEM_SPECS["car rear wing"].build({}, typed, None)
    assert capped.problem.CD_budget is None        # exactly one live budget
    assert [m for m in capped.problem.constraint_labels
            if "drag" in m] == ["drag force margin"]


def test_the_size_never_travels_as_a_flag_again():
    """The four old spellings are refused, not silently ignored.

    Accepted-and-ignored is the failure `check_flags` exists to close: a band
    sent to a family that does not read it looks, from every downstream
    surface, exactly like a band that was honoured.
    """
    import pytest as _pytest

    from aerobo import api

    for name in ("car rear wing", "car rear wing (two-element)",
                 "car rear wing + endplates"):
        for key in ("span_min_m", "span_max_m", "area_min_m2", "area_max_m2"):
            with _pytest.raises(KeyError, match="does not honour"):
                api.check_flags(name, {key: 0.3})
