"""The bridge — flightmodel.build_flight_model.

STATED IS NOT FLOWN. The solver discards its lattice, so anything that flies a
stored design has to rebuild one, and a rebuild is exactly where a shell
starts flying a different aeroplane from the one it scored. The contract this
module offers is not "the rebuild is correct" — it is "the rebuild MEASURES
its own disagreement and hands it over".
"""

import numpy as np
import pytest

from aerobo import api, flightmodel as fm, sixdof as sd

PROBLEM = "tail"


@pytest.fixture(scope="module")
def report():
    cfg = api.RunConfig(problem_name=PROBLEM, budget=4, seed=0)
    built = api.PROBLEM_SPECS[PROBLEM].build({}, {}, None)
    return api.design_report(cfg, built.bounds.mean(axis=1))


@pytest.fixture(scope="module")
def model(report):
    return fm.build_flight_model(report, V=45.0)


# ------------------------------------------------------- it checks itself

def test_the_bridge_measures_its_own_disagreement(model):
    """Every row it can compare, it compares — and it keeps the number."""
    names = {r[0] for r in model.fidelity.rows}
    assert {"CL_alpha", "x_np", "SM"} <= names
    comparable = [r for r in model.fidelity.rows if r[3] is not None]
    assert comparable, "nothing was actually checked against the report"
    assert model.fidelity.worst > 0.0


def test_the_fidelity_reads_as_a_sentence(model):
    text = model.fidelity.as_text()
    assert "flown" in text and "rebuilt" in text and "%" in text


def test_a_disagreeing_rebuild_is_reported_not_swallowed(model):
    """On this problem the design is scored through the LIFTING-LINE path and
    reflown here through the lattice, so the two legitimately differ by a few
    per cent. What must never happen is the gap being hidden: `ok` has to go
    False and the text has to carry the numbers.
    """
    assert model.fidelity.ok == (model.fidelity.worst < 0.02)
    if not model.fidelity.ok:
        assert "%" in model.fidelity.as_text()


def test_every_invented_number_is_recorded_as_an_assumption(model):
    """Saying which numbers were invented is the difference between an
    estimate and a fabrication.

    The FIN used to be the headline example — "it does not exist in any
    report, so its size is invented". V5 gave it a report block
    (``fin.size_fin``, ``geometry.fin``), so the honest assertion flipped:
    the fin must now be READ, and the note must say whose it is. What is
    still invented is asserted alongside it, so this test keeps testing the
    rule rather than one example of it."""
    joined = " ".join(model.assumptions).lower()
    assert "fin" in joined
    # READ, not invented — and the note says so
    assert "the fin is the design's" in joined
    assert "not in the report" not in joined
    # ...and the numbers that genuinely are invented still say they are
    assert "the report states no" in joined
    assert "assumed" in joined


# ------------------------------------------------------- it produces a model

def test_the_bridge_returns_something_that_actually_flies(model):
    st, trimmed = sd.trim_level(model.aircraft, V=45.0, altitude_m=300.0)
    hist = sd.integrate(trimmed, st, dt=0.02, n=750)      # 15 s
    assert abs(hist[-1].altitude_m - 300.0) < 5.0
    assert abs(hist[-1].V - 45.0) < 1.0


def test_the_rebuilt_deck_has_the_controls_that_were_asked_for(model):
    assert {"aileron", "elevator", "rudder"} <= set(model.deck.columns)


def test_a_fin_is_fitted_by_default_so_yaw_stiffness_is_not_zero(model):
    assert model.deck.Cn_beta > 0.0
    assert "Cn_beta" not in model.deck.zeros


def test_turning_the_fin_off_returns_the_hole(report):
    """Without a vertical surface the design measures EXACTLY zero yaw
    stiffness, and the deck says why. This is the before-picture that
    justifies stage 5 existing."""
    bare = fm.build_flight_model(report, fm.ControlsSpec(fin=False), V=45.0)
    assert bare.deck.Cn_beta == pytest.approx(0.0, abs=1e-12)
    assert "Cn_beta" in bare.deck.zeros


def test_a_bigger_fin_buys_more_yaw_stiffness(report):
    small = fm.build_flight_model(
        report, fm.ControlsSpec(fin_height_m=0.6), V=45.0)
    big = fm.build_flight_model(
        report, fm.ControlsSpec(fin_height_m=1.8), V=45.0)
    assert 0.0 < small.deck.Cn_beta < big.deck.Cn_beta


def test_a_ventral_fin_flips_the_roll_contribution(report):
    up = fm.build_flight_model(report, fm.ControlsSpec(), V=45.0)
    down = fm.build_flight_model(
        report, fm.ControlsSpec(fin_ventral=True), V=45.0)
    assert up.deck.Cn_beta > 0.0 and down.deck.Cn_beta > 0.0
    assert down.deck.Cl_beta > up.deck.Cl_beta


def test_moving_the_cg_moves_the_static_margin(report):
    """The CG is a moment arm, not a label: it has to move the physics."""
    a = fm.build_flight_model(report, V=45.0, x_cg_m=0.0)
    b = fm.build_flight_model(report, V=45.0, x_cg_m=0.4)
    assert b.deck.static_margin < a.deck.static_margin


def test_the_planform_comes_off_the_chord_array_the_solver_produced(report):
    """Taper is MEASURED from the spanwise chord the report carries, not read
    from a design variable that many problems do not have."""
    m = fm.build_flight_model(report, V=45.0)
    assert 0.0 < m.model.wing.taper <= 1.0
    assert m.deck.b == pytest.approx(report["geometry"]["b"], rel=1e-9)


def test_a_report_with_no_planform_is_refused_with_a_reason():
    with pytest.raises(ValueError, match="nothing to fly"):
        fm.build_flight_model({"geometry": {}, "breakdown": {}})


def test_inertia_is_produced_and_traceable(model):
    i = model.aircraft.inertia
    assert i.mass_kg > 0 and i.Ixx > 0 and i.Iyy > 0 and i.Izz > 0
    assert i.basis and i.basis != "given"


def test_the_controls_spec_defaults_match_the_shell_defaults():
    """One question, one place: the engine's default and the stage's opening
    answer must not drift apart."""
    from gui.v4 import session

    spec = fm.ControlsSpec()
    d = session.CONTROLS_DEFAULTS
    assert spec.aileron is d["aileron"]["on"]
    # the shell asks for no flap at all, so the engine must not fit one
    # unasked: its own default is the answer the stage no longer gives
    assert "flap" not in d
    assert spec.flap is False
    assert spec.fin is d["vertical"]["on"]
    assert spec.aileron_chord_frac == d["aileron"]["chord_frac"]
    assert spec.aileron_span == (d["aileron"]["span_from"],
                                 d["aileron"]["span_to"])
    assert spec.rudder_chord_frac == d["vertical"]["rudder_chord_frac"]
    assert spec.fin_ventral is d["vertical"]["ventral"]


def test_the_deck_is_affine_and_the_sim_reads_the_same_columns(model):
    """Every column the aircraft evaluates has to exist in the deck, or a
    control silently does nothing."""
    ac = model.aircraft
    st = sd.State(vel=np.array([45.0, 0.0, 0.0]))
    C = ac.coefficients(st)
    assert {"CL", "CD", "CX", "CY", "CZ", "Cl", "Cm", "Cn"} <= set(C)
    for name in ("alpha", "beta", "p", "q", "r"):
        assert name in model.deck.columns


def test_the_bridge_linearises_at_the_designs_own_LIFT(model, report):
    """Not at zero — or the flight model silently has no Cl_r and no Cn_p.

    Both are proportional to the base lift, so a deck built at alpha = 0
    reports them as exactly zero.

    This asserted the report's ANGLE, and the bridge deliberately stopped
    using it: the two cores do not share a section convention, so taking the
    angle put the rebuilt aeroplane at the wrong lift — CL 0.5857 where the
    design was scored at 0.5000 on `tail + winglet`, which is every
    lift-proportional derivative ~17 % out. It trims to the LIFT instead,
    which makes those derivatives right by construction and turns the
    disagreement into a stated angle difference. So the contract asserted
    here is the lift, and the angle difference is required to be REPORTED
    rather than absorbed.
    """
    bd = report["breakdown"]
    CL_scored = float(bd.get("CL_total") or bd["CL_target"])
    assert model.model.solve(model.alpha_trim,
                             i_t=model.i_t_trim).CL == pytest.approx(
        CL_scored, rel=1e-6)
    # the angle the two cores disagree by is on the record, with both numbers
    joined = " ".join(model.assumptions)
    assert "linearised at the LIFT" in joined
    assert f"{np.rad2deg(float(bd['alpha_rad'])):+.2f} deg" in joined
    assert model.alpha_trim > 0.01
    assert model.deck.Cl_r > 0.01
    assert model.deck.Cn_p < -0.01
    assert "Cl_r" not in model.deck.zeros
    assert "Cn_p" not in model.deck.zeros
