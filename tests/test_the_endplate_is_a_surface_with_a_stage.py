"""Stage 2.8: the car's endplate gets a section stage of its own.

Choosing a section is a stage per surface in this shell — one question, one
place — and the endplate was the surface that had no stage. Its section flag
named a CONSTRUCTION (flat, rounded, shaped), which is a drag law and a
stiffness law and no shape at all, so "optimise the endplate aerofoil" had
nowhere to be asked.

The plate asks the FIN's questions on a different vehicle: it is a vertical
panel at nominally zero incidence whose section buys drag and stiffness, so it
is screened over the symmetric library at cl = 0 and searched as the same
four-variable symmetric CST problem. What it does NOT share is its size — the
plate's chord is a design row of its own (a ratio of the wing's TIP chord), so
its Reynolds number is its own and screening it at the wing's was the reason
it could not be designed.
"""

import pytest

from aerobo import api

import gui.v3.session as session
from gui.v3 import app as v3app
from gui.v3 import config as v3config
from gui.v3.stages import airfoil as af

PLATE_FAMILY = "car rear wing + endplates"


def _car(problem: str = PLATE_FAMILY) -> dict:
    S = session.make_session()
    S["wing"]["choices"].update(medium="track", car_endplates=True)
    session.apply_choices(S)
    return S


def test_the_plate_is_a_surface_the_shell_knows_about():
    """A surface exists when every table names it — and the tables are what a
    fourth surface arriving half-registered breaks."""
    assert "plate" in session.SURFACES
    assert session.SURFACE_STAGES["plate"] == "airfoil_plate"
    assert session.STAGE_SURFACE["airfoil_plate"] == "plate"
    assert "airfoil_plate" in session.STAGES
    assert "plate" in session.SECTION_KEYS
    # it shares the wing's views by IDENTITY, so one question asked four times
    # cannot drift into four forms
    assert session.VIEWS["airfoil_plate"] is session.VIEWS["airfoil"]
    assert v3app.STAGE_MODULES["airfoil_plate"] == "airfoil"
    # ...and a workspace, or airfoil_state would KeyError on arrival
    assert "airfoil_plate" in session.make_session()


def test_the_stage_appears_only_where_the_plate_is_a_designed_part():
    """A stage for a surface that does not exist is worse than no stage. The
    plain car wing's plate is a FENCE carrying the wing's own chord and the
    wing's own section — there is nothing there to give an aerofoil to."""
    S = _car()
    assert session.plate_surface(S)
    assert session.stage_visible(S, "airfoil_plate")

    S["wing"]["choices"]["car_endplates"] = False
    session.apply_choices(S)
    assert S["wing"]["problem"].startswith("car rear wing")
    assert not session.plate_surface(S)
    assert not session.stage_visible(S, "airfoil_plate")
    state, why = session.stage_states(S)["airfoil_plate"]
    assert state == "locked"
    assert "FENCE" in why and "endplate mount" in why

    # ...and an aircraft has no plate at all
    air = session.make_session()
    assert not session.plate_surface(air)


def test_the_plate_is_screened_at_its_own_reynolds_number():
    """The plate's chord is a RATIO of the wing's tip chord and moves with
    three design rows. Screening it at the wing's Reynolds number designs a
    section for a chord the plate never has."""
    S = _car()
    geo = session.surface_geometry(S, "plate")
    assert geo is not None
    assert geo["lifting"] is False
    assert geo["cl"] == 0.0
    assert "mid design box" in geo["source"]

    cond = session.section_conditions(S, "plate")
    wing = session.section_conditions(S, "main")
    assert cond["cl_design"] == 0.0
    assert cond["re"] > 0.0
    assert cond["re"] != pytest.approx(wing["re"], rel=0.05)

    # ...and it is the chord the LATTICE actually builds, not an estimate of
    # one: the closed form and the flown plate agree at the published cant
    built = api.PROBLEM_SPECS[api.base_of(S["wing"]["problem"])].build(
        {}, {}, None)
    flown = built.evaluate(built.bounds.mean(axis=1))
    assert geo["mac"] == pytest.approx(flown["endplate_chord_m"], rel=1e-6)
    assert cond["re"] == pytest.approx(flown["endplate_Re"], rel=1e-3)


def test_the_plate_may_only_be_offered_a_symmetric_section():
    """endplate.py's whole "section trap": a vertical panel is built at
    theta = twist - alpha_L0, so a cambered plate at zero toe carries a side
    force — and the two plates' loads cancel in CY, which is what would have
    made it invisible. The problem refuses one; this is the half that stops
    it being offered."""
    names = af.screen_names_for("plate")
    assert names is not None, "the plate's screen must be restricted"
    assert set(names) == set(api.symmetric_section_names())
    # the wing's is not restricted, so this is about the surface and not a
    # blanket rule
    assert af.screen_names_for("main") is None


def test_the_plates_shape_search_is_the_symmetric_one():
    """A symmetric CST section is four variables, not eight — and `symmetric`
    is set inside shape_kwargs rather than at the call site so a CONTINUATION
    re-flies the same problem."""
    S = _car()
    A = session.airfoil_state(S, "plate")
    kw = af.shape_kwargs(S, "plate", A["opt"], A["weights"], None)
    assert kw["symmetric"] is True
    assert kw["cl_design"] == 0.0
    assert kw["re"] == pytest.approx(
        session.section_conditions(S, "plate")["re"])


def test_the_plate_is_ranked_on_what_a_plate_is_chosen_for():
    """Three criteria are identically dead on a surface that makes no lift,
    and a weight that ranks nothing is worse than an absent one — it looks
    like an answer. Shared with the fin's entry rather than copied, because
    two copies of one argument drift."""
    assert session.surface_job(_car(), "plate") == "fin"
    weights, why = session.recommended_weights(_car(), "plate")
    assert weights["cdcr"] > 0.0
    for dead in ("ldcr", "ldmax", "cm"):
        assert weights[dead] == 0.0
        assert dead in af.DEAD_CRITERIA["plate"]
    assert af.DEAD_CRITERIA["plate"] is af.DEAD_CRITERIA["fin"]
    assert "no side force" in why or "does not carry" in why


def test_an_unanswered_stage_sends_no_section_and_flies_the_build_up():
    """The plate's default is a stand-in NAMED for the card, not a library
    entry — sending it would make the solver look "NACA 0010" up in the UIUC
    directory. Until the stage is answered the plate flies its construction
    family, which is every published run."""
    S = _car()
    assert session.section_summary(S, "plate").endswith("(its own default)")
    assert session.section_flag_value(S, "plate") is None
    assert not v3config.flies_chosen_section(S, "plate")
    assert api.SECTION_PLATE_KEY not in v3config.flags(S)

    # ...and the stand-in's thickness follows the design box rather than a
    # constant this shell would have to keep in step with endplate.TC_BOUNDS
    own = session.plate_default_section(S)
    assert own["symmetric"] is True
    tc = session._mid(session._searched_box(S), "endplate_tc")
    assert own["tc"] == pytest.approx(tc)


def test_answering_the_stage_sends_the_plates_own_section():
    """...and it travels on its OWN key, beside the wing's, because a car can
    carry a chosen wing section and a chosen plate section at once."""
    S = _car()
    S["airfoil"]["section_plate"] = {"name": "e168", "tc": 0.10,
                                     "symmetric": True, "source": "library"}
    assert session.section_is_own(S, "plate")
    assert v3config.flies_chosen_section(S, "plate")
    flags = v3config.flags(S)
    assert flags[api.SECTION_PLATE_KEY]
    assert session.section_summary(S, "plate") == "e168"
    # the whole config still builds and is accepted by the family it names
    api.check_flags(S["wing"]["problem"], flags)


def test_the_stage_is_named_after_the_plate_and_not_the_second_surface():
    """The fin's stage was once labelled with the tailplane's name. The same
    rule, and the same fix, for a surface that is not the second one."""
    S = _car()
    assert session.stage_label(S, "airfoil_plate") == "2.8  Endplate"
    assert session.surface_name(S, "plate") == "endplate"


def test_the_run_button_knows_what_the_plate_stage_means_by_run():
    """The toolbar's map is a literal dict indexed with [stage]; a stage
    missing from it raises KeyError on the Run button rather than doing
    nothing."""
    import inspect
    src = inspect.getsource(v3app._run_stage)
    assert '"airfoil_plate": "run_airfoil_plate"' in src


def test_the_pipeline_chains_through_the_plate_without_a_literal():
    """`next_section_stage` walks STAGES filtered by visibility — 'a third
    surface must not need a third literal', and nor does a fourth."""
    S = _car()
    seen, stage = [], "airfoil"
    while stage is not None and len(seen) < 8:
        nxt = session.next_section_stage(S, stage)
        if nxt is None:
            break
        seen.append(nxt)
        stage = nxt
    assert "airfoil_plate" in seen
