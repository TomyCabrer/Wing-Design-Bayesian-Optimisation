"""The six-DOF simulator's own claims, each against a closed form.

Every test here pins a defect that the 605-test V4 suite was green over,
because each of them asserted the SHAPE of the code rather than an outcome:
a thrust moment whose sign was asserted from the implementation, a lift
offset nothing summed, a stall nothing measured past, a sideslip nothing
priced. So each test below states a number that can be computed by hand,
and each has been shown to fail against a mutant.
"""

from dataclasses import replace

import numpy as np
import pytest

from aerobo import dynamics as dyn, flightmodel as fmod, sixdof as sd
from aerobo.geometry import Wing, cosine_stations
from aerobo.vlm import VLM, TailSurface, VerticalSurface


def _lattice():
    return VLM(Wing(b=10.0, S=10.0, taper=0.5), N=40, V=30.0,
               tail=TailSurface(S=2.0, x=4.0, z=0.5, AR=4.0, N=16),
               vertical=VerticalSurface(height=1.2, chord=0.65, x=4.0,
                                        z_root=0.5, N=12))


def _deck_at(m, alpha_deg, x_cg=0.20):
    ctl = [dyn.aileron(m), dyn.elevator(m), dyn.rudder(m)]
    return dyn.deck(m, x_cg=x_cg, controls=ctl,
                    alpha=np.deg2rad(alpha_deg))


def _ac(deck, **kw):
    kw.setdefault("rho", 1.225)
    kw.setdefault("CD0", 0.03)
    kw.setdefault("oswald_e", 0.85)
    return sd.Aircraft(deck=deck,
                       inertia=sd.Inertia.from_layout(120.0, 10.0, 6.0), **kw)


# ------------------------------------------- the deck's own expansion point

def test_the_base_incidence_is_not_counted_twice():
    """A deck's ``const`` is the coefficient set AT its base state.

    ``deck(alpha=a)`` solves the lattice at ``a``, so ``const["CZ"]`` is
    exactly ``-CL_alpha * a`` on an uncambered wing — measured to six
    figures. A consumer that then adds ``CL_alpha * alpha`` on top of it is
    flying an aeroplane with two lots of the base lift, and the visible
    symptom is a trim incidence that slides 1:1 with a number that describes
    the LINEARISATION and not the aeroplane.

    The closed form and the outcome, both:
    """
    m = _lattice()
    base = _deck_at(m, 0.0)
    for a_deg in (3.0, 6.0):
        D = _deck_at(m, a_deg)
        # the closed form: const IS the base lift, and the deck says so
        assert D.alpha_ref == pytest.approx(np.deg2rad(a_deg))
        assert -D.const["CZ"] == pytest.approx(
            base.CL_alpha * np.deg2rad(a_deg), rel=2e-3)
        # ...and the disturbance is what the columns are evaluated at
        assert D.disturbance(np.deg2rad(a_deg)) == pytest.approx(0.0)

    # THE OUTCOME: the same aeroplane, trimmed at the same speed, must reach
    # the same body incidence whichever state its deck was expanded about.
    trims = []
    for a_deg in (0.0, 3.0, 6.0):
        st, _ = sd.trim_level(_ac(_deck_at(m, a_deg)), V=30.0)
        trims.append(st.alpha)
    assert max(trims) - min(trims) < np.deg2rad(0.02), (
        f"trim incidence slid with the linearisation angle: "
        f"{[float(np.rad2deg(t)) for t in trims]} deg")


def test_the_lift_at_the_base_state_is_the_base_lift_and_not_twice_it():
    """Evaluated AT its own base, the deck must reproduce the lattice."""
    m = _lattice()
    for a_deg in (0.0, 4.0):
        D = _deck_at(m, a_deg)
        ac = _ac(D, stall=sd.Stall(enabled=False))
        a = np.deg2rad(a_deg)
        st = sd.State(vel=30.0 * np.array([np.cos(a), 0.0, np.sin(a)]))
        # the affine model at its own expansion point is the constant
        assert ac.coefficients(st)["CL"] == pytest.approx(-D.const["CZ"],
                                                          abs=1e-9)


# ------------------------------------------------------------ the propulsion

def test_the_thrust_moment_is_r_cross_F():
    """No convention to have an opinion about: ``M = r x F``."""
    for z in (-0.7, 0.0, 0.35, 1.2):
        p = sd.Propulsion(thrust_n=800.0, z_offset_m=z)
        F, M = p.force_moment()
        assert M == pytest.approx(np.cross(np.array([0.0, 0.0, z]), F))
    # ...and a thrust line BELOW the CG (z down, so +z) is nose UP
    assert sd.Propulsion(thrust_n=800.0, z_offset_m=1.0).force_moment()[1][1] \
        > 0.0


def test_the_thrust_line_is_answered_once_in_the_controls_spec():
    """``through_cg`` is a real answer, not a zero somebody forgot."""
    s = fmod.ControlsSpec()
    assert s.thrust_through_cg is True
    assert s.thrust_z_offset_m == 0.0
    # ...and the value is IGNORED while the switch is on, so the two halves
    # cannot disagree
    assert fmod.ControlsSpec(thrust_z_below_cg_m=0.9).thrust_z_offset_m == 0.0
    assert fmod.ControlsSpec(thrust_through_cg=False,
                             thrust_z_below_cg_m=0.9).thrust_z_offset_m \
        == pytest.approx(0.9)


# -------------------------------------------------------------- the sideslip

def test_a_sideslip_costs_drag():
    """The fin makes a side force, and a lifting surface pays for a force.

    Before this the drag was built from ``CL`` alone: measured ``CD =
    0.03169`` at beta = 0, 15 AND 30 degrees, so full rudder into a 30-degree
    skid neither slowed the aeroplane nor cost it anything.
    """
    m = _lattice()
    D = _deck_at(m, 3.0)
    AR_v = 1.2 ** 2 / (1.2 * 0.65)
    ac = _ac(D, AR_vertical=AR_v, stall=sd.Stall(enabled=False))
    a = D.alpha_ref
    got = []
    for b_deg in (0.0, 10.0, 20.0, 30.0):
        b = np.deg2rad(b_deg)
        st = sd.State(vel=30.0 * np.array(
            [np.cos(a) * np.cos(b), np.sin(b), np.sin(a) * np.cos(b)]))
        C = ac.coefficients(st)
        got.append(C["CD"])
        # THE CLOSED FORM, the same one the lift uses, on the FIN's own AR.
        # The deck's own body-axis CY is what is squared: the reported CY
        # has the drag's own lean into body y taken off it afterwards.
        CY_deck = C["CY"] + C["CD"] * np.sin(b)
        assert C["CDv"] == pytest.approx(
            CY_deck ** 2 / (np.pi * AR_v * ac.oswald_e), rel=1e-9)
    assert got[0] < got[1] < got[2] < got[3], f"sideslip was free: {got}"
    # ...and it is a real number, not a rounding: measured +10.2 % of the
    # whole drag at 30 deg of skid, on a CD0 of 0.03 that dominates it. This
    # is the FIN's induced drag alone — the fuselage crossflow is not
    # modelled, so the term understates rather than invents.
    assert got[3] - got[0] > 0.003

    # with NO fin measured the term is ABSENT rather than guessed
    bare = _ac(D, AR_vertical=None, stall=sd.Stall(enabled=False))
    b = np.deg2rad(30.0)
    st = sd.State(vel=30.0 * np.array(
        [np.cos(a) * np.cos(b), np.sin(b), np.sin(a) * np.cos(b)]))
    assert bare.coefficients(st)["CDv"] == 0.0


def test_the_drag_leans_with_the_flight_path_in_sideslip():
    """The drag acts along -V, and V leaves the plane of symmetry by beta."""
    m = _lattice()
    ac = _ac(_deck_at(m, 3.0), stall=sd.Stall(enabled=False))
    a = ac.deck.alpha_ref
    b = np.deg2rad(20.0)
    st = sd.State(vel=30.0 * np.array(
        [np.cos(a) * np.cos(b), np.sin(b), np.sin(a) * np.cos(b)]))
    C = ac.coefficients(st)
    # the body-x and body-z components carry cos(beta); the body-y component
    # of the drag is -CD sin(beta), and all three are just -CD * V_hat
    assert C["CX"] == pytest.approx(
        -C["CD"] * np.cos(a) * np.cos(b) + C["CL"] * np.sin(a))
    assert C["CZ"] == pytest.approx(
        -C["CD"] * np.sin(a) * np.cos(b) - C["CL"] * np.cos(a))


# ----------------------------------------------------------------- the stall

def test_the_stall_reaches_the_controls_and_the_roll_damping():
    """A wing that has stopped making lift cannot still roll at full rate.

    Measured before this existed: 20 degrees of aileron gave ``dCl =
    -0.10041`` at alpha = 2, 14, 25 AND 40 degrees, and the roll damping at
    p = 0.5 rad/s was ``-0.04463`` at every one of them.
    """
    m = _lattice()
    D = _deck_at(m, 0.0)
    ac = _ac(D)
    roll, damp = [], []
    for a_deg in (2.0, 14.0, 25.0, 40.0):
        a = np.deg2rad(a_deg)
        st = sd.State(vel=30.0 * np.array([np.cos(a), 0.0, np.sin(a)]))
        base = ac.coefficients(st)["Cl"]
        roll.append(replace(ac, controls={"aileron": np.deg2rad(20.0)}
                            ).coefficients(st)["Cl"] - base)
        spun = sd.State(vel=st.vel, rates=np.array([0.5, 0.0, 0.0]))
        damp.append(ac.coefficients(spun)["Cl"]
                    - ac.coefficients(st)["Cl"])
    for name, seq in (("aileron", roll), ("roll damping", damp)):
        mags = [abs(v) for v in seq]
        assert mags[0] > mags[1] > mags[2] > mags[3], \
            f"{name} did not fade with the stall: {seq}"
        assert mags[3] < 0.25 * mags[0], \
            f"{name} kept {mags[3] / mags[0]:.0%} of its authority at 40 deg"

    # ...and the TAIL is not stalled when the wing is: its channels are left
    # alone, because they fly at their own local incidence
    assert "elevator" not in ac.STALL_SCALED
    assert "rudder" not in ac.STALL_SCALED
    assert "beta" not in ac.STALL_SCALED


def test_the_slope_factor_is_the_derivative_of_the_clip():
    """``slope_factor`` must BE ``d(lift)/d(CL_lin)``, not resemble it."""
    s = sd.Stall()
    for x in (0.2, 0.9, 1.4, 2.0, 3.5):
        h = 1e-6
        fd = (s.lift(x + h, 0.0) - s.lift(x - h, 0.0)) / (2 * h)
        assert s.slope_factor(x, 0.0) == pytest.approx(fd, rel=1e-5)
    # the property the clip was chosen for: normal flight is untouched
    assert s.slope_factor(0.5) > 0.999
    # ...and deep in the stall there is nothing left
    assert s.slope_factor(2.0 * s.CL_max) < 0.01
    # disabled means disabled
    assert replace(s, enabled=False).slope_factor(9.0) == 1.0


def test_the_stall_breaks_the_pitching_moment():
    """The lift the stall removes had a moment arm, and it is x_cg exactly.

    The lattice puts the wing's quarter-chord line at x = 0, so lift removed
    from the wing changes the moment about the CG by ``dCL * x_cg / mac``.
    """
    m = _lattice()
    D = _deck_at(m, 0.0, x_cg=0.6)
    ac = _ac(D)
    lin = _ac(D, stall=sd.Stall(enabled=False))
    a = np.deg2rad(30.0)
    st = sd.State(vel=30.0 * np.array([np.cos(a), 0.0, np.sin(a)]))
    stalled, clean = ac.coefficients(st), lin.coefficients(st)
    dCL = stalled["CL"] - clean["CL"]
    assert dCL < 0.0, "the stall did not remove any lift at 30 deg"
    # the controls are scaled too, so compare the Cm the deck itself gives
    assert stalled["Cm"] - clean["Cm"] == pytest.approx(
        dCL * D.x_cg / D.mac, rel=1e-9)


def test_the_stall_is_the_designs_own_section_where_the_report_states_it():
    """``CL_max = 1.4`` is a default, not a measurement.

    The whole package measures a section's cl_max; a simulation that ignores
    it flies a 1.1 section to 1.4 and clips an 1.8 one.
    """
    m = _lattice()
    D = _deck_at(m, 0.0)
    hi, why_hi = fmod._stall_from_report({"cl_max": 1.85}, D)
    lo, why_lo = fmod._stall_from_report({"cl_max": 1.05}, D)
    assert hi.CL_max > lo.CL_max
    # the 3-D correction is the lift-slope ratio, and it is stated
    ratio = D.CL_alpha / (2.0 * np.pi)
    assert hi.CL_max == pytest.approx(1.85 * ratio)
    assert hi.alpha_stall_rad == pytest.approx(hi.CL_max / D.CL_alpha)
    assert "1.850" in why_hi and "cl_max" in why_hi
    # ...and a report that states nothing says SO rather than pretending
    d, why = fmod._stall_from_report({}, D)
    assert d.CL_max == sd.Stall().CL_max
    assert "no section cl_max" in why and "default" in why
    assert "measured" in why_lo


# -------------------------------------------------------------- the planform

def _geom_of(**kw):
    w = Wing(b=10.0, S=10.0, **kw)
    _, y = cosine_stations(40, 10.0)
    return w, {"b": 10.0, "S": 10.0,
               "surfaces": [{"name": "wing", "y": list(y),
                             "chord": list(w.chord(y))}]}


@pytest.mark.parametrize("kw", [
    dict(taper=0.5),
    dict(taper=1.6),                                # INVERSE taper
    dict(taper=0.6, chord_coeffs=(0.4, -0.3)),
    dict(taper=0.8, chord_coeffs=(-0.35, 0.2, 0.1)),
])
def test_the_rebuilt_planform_is_the_reported_one(kw):
    """The report writes the wing out as its own chord array, and that array
    IS the planform. Reading two end chords and clipping the ratio to 1
    turned an inverse taper into a constant chord and a chord law into a
    trapezoid — and then flew it.

    The claim is on the DISTRIBUTION, not on the parameters: taper and the
    law trade off against each other, so the same c(y) has many
    parameterisations and only the curve is the aeroplane.
    """
    w, geom = _geom_of(**kw)
    b, S, taper, coeffs, rms = fmod._planform_from_report(geom)
    _, y = cosine_stations(40, 10.0)
    got = Wing(b=b, S=S, taper=taper, chord_coeffs=coeffs)
    err = (np.sqrt(np.mean((got.chord(y) - w.chord(y)) ** 2))
           / np.mean(w.chord(y)))
    assert err < 1e-3, f"rebuilt chord distribution is {err:.2e} off"
    assert rms == pytest.approx(err, abs=1e-6), \
        "the reported residual is not the residual"
    if kw["taper"] > 1.0:
        assert taper > 1.0, "an inverse taper was clipped to a straight wing"


def test_a_tip_device_is_not_the_wings_tip_chord():
    """The widest station of a winglet wing belongs to the WINGLET.

    ``chord[argmax(|y|)]`` read that strip's chord and called it the wing's
    tip chord, which on a device of a third the local chord turns a taper-0.5
    wing into a taper-0.17 one and then flies it.
    """
    w, geom = _geom_of(taper=0.5)
    y = np.asarray(geom["surfaces"][0]["y"], dtype=float)
    c = np.asarray(geom["surfaces"][0]["chord"], dtype=float)
    tip_c = float(c[np.argmax(np.abs(y))])
    # ...four extra strips per side, outboard of the tip, at a third the
    # chord: a canted device, which is exactly what the lattice draws
    dev_y = np.array([5.05, 5.15, 5.25, 5.35])
    dev_y = np.concatenate([-dev_y[::-1], dev_y])
    dev_c = np.full(dev_y.shape, tip_c / 3.0)
    geom["surfaces"][0]["y"] = list(np.concatenate([y, dev_y]))
    geom["surfaces"][0]["chord"] = list(np.concatenate([c, dev_c]))

    b, S, taper, coeffs, rms = fmod._planform_from_report(geom)
    assert b == pytest.approx(10.0), "the device widened the span"
    assert taper == pytest.approx(0.5, rel=0.05), (
        f"the device's chord was read as the wing's tip chord: taper "
        f"{taper:.3f}")
    got = Wing(b=b, S=S, taper=taper, chord_coeffs=coeffs)
    on_wing = np.abs(y) <= 5.0 + 1e-9
    err = (np.sqrt(np.mean((got.chord(y[on_wing]) - c[on_wing]) ** 2))
           / np.mean(c[on_wing]))
    assert err < 1e-3


def test_the_planform_residual_is_reported_as_a_fidelity_row():
    """A rebuild this module cannot do must be VISIBLE, not silent."""
    from aerobo import api
    cfg = api.RunConfig(problem_name="tail", budget=4, seed=0)
    built = api.PROBLEM_SPECS["tail"].build({}, {}, None)
    rep = api.design_report(cfg, built.bounds.mean(axis=1))
    fm = fmod.build_flight_model(rep, V=45.0)
    names = [r[0] for r in fm.fidelity.rows]
    assert "planform c(y)" in names
    row = next(r for r in fm.fidelity.rows if r[0] == "planform c(y)")
    assert row[2] < 0.01, f"the shipped tail planform rebuilds to {row[2]:.2e}"


def test_the_fin_gives_the_aeroplane_its_vertical_aspect_ratio():
    """The sideslip drag needs the FIN's aspect ratio, and the bridge is the
    only thing that knows it."""
    from aerobo import api
    cfg = api.RunConfig(problem_name="tail", budget=4, seed=0)
    built = api.PROBLEM_SPECS["tail"].build({}, {}, None)
    rep = api.design_report(cfg, built.bounds.mean(axis=1))
    with_fin = fmod.build_flight_model(rep, V=45.0)
    assert with_fin.aircraft.AR_vertical is not None
    assert with_fin.aircraft.AR_vertical > 0.0
    none = fmod.build_flight_model(
        rep, fmod.ControlsSpec(fin=False), V=45.0)
    assert none.aircraft.AR_vertical is None


def test_a_cg_taken_at_the_quarter_chord_says_so():
    """Every other invented number is recorded; this one was not."""
    from aerobo import api
    cfg = api.RunConfig(problem_name="tail", budget=4, seed=0)
    built = api.PROBLEM_SPECS["tail"].build({}, {}, None)
    rep = api.design_report(cfg, built.bounds.mean(axis=1))
    stripped = dict(rep)
    stripped["geometry"] = {**rep["geometry"],
                            "tail": {k: v for k, v
                                     in (rep["geometry"]["tail"] or {}).items()
                                     if k != "x_cg"}}
    stripped["breakdown"] = {k: v for k, v in rep["breakdown"].items()
                             if k != "x_cg"}
    fm = fmod.build_flight_model(stripped, V=45.0)
    joined = " ".join(fm.assumptions).lower()
    assert "no cg" in joined and "quarter-chord" in joined


# -------------------------------------------------------------- the trimming

def test_trim_drives_the_equations_of_motion_to_zero():
    """Not a rewrite of them — the residual ``derivative`` actually leaves.

    The trim used to solve a hand-resolved ``CL + CD tan(alpha)`` form, which
    stopped being the equations the moment ``coefficients`` grew a term the
    rewrite did not know about.
    """
    m = _lattice()
    for a_deg in (0.0, 4.0):
        ac = _ac(_deck_at(m, a_deg), AR_vertical=1.85)
        st, trimmed = sd.trim_level(ac, V=30.0, altitude_m=100.0)
        d = sd.derivative(trimmed, st)
        assert np.linalg.norm(d[7:10]) < 1e-9, "forces do not balance"
        assert np.linalg.norm(d[10:13]) < 1e-9, "moments do not balance"


def test_a_speed_the_wing_cannot_lift_is_refused_by_name():
    """And refused as 'cannot fly level', not as a Newton diagnostic."""
    m = _lattice()
    ac = _ac(_deck_at(m, 0.0))
    with pytest.raises(ValueError, match="cannot fly level"):
        sd.trim_level(ac, V=6.0)
