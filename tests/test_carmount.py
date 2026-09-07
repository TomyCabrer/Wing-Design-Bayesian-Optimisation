"""The mount as a continuum: does it contain the two published ends, and does
it say anything true in between?

carwing.py answers "where is this wing bolted on" with a two-valued flag and
two hard-coded beam formulas. carmount.py answers it with a support STATION,
so the first thing these tests establish is that the generalisation is not a
rewrite: at station 0 and at station b/2 it must return carwing's own floats,
bit for bit, on the same load array. Everything after that is the interior —
gated against closed forms derived in the module docstring, never against the
numerics restating themselves.

The gates that would catch a mutation, listed so they can be checked:

* the two published ends, ``==`` against carwing on a synthetic load AND on a
  solved one (a formula that drifted would move both);
* the overhanging beam's moment and deflection against their closed forms at
  seven interior stations (a sign error in the reaction term survives the two
  ends and dies here);
* the junction charge against (c(y_s)/c_root)^2, exactly (a helper that kept
  reading the root chord passes on an untapered wing and fails on a tapered
  one — both are tested);
* the torsion against its own closed form, and the divergence speed against a
  hand-written copy of equation (12);
* the OFF invariants with ``==``: zero windup at the quarter chord, exactly
  1.0 modifiers at zero loss, and the flat-plate pylon charge against
  carwing._strut_cd0's own float.

Three of these gates exist because the mutation they name once SURVIVED the
whole file, and each says so in its own docstring:

* ``footprint_is_resolved`` tested a proximity COUNT while claiming to test
  resolution, and passed grids recovering -98% to +28% of the footprint's
  integral. It now measures that integral, and the tolerance it measures
  against is bisected here from both sides;
* deleting the mirrored footprint (``[-y_s, y_s]`` -> ``[y_s]``) exactly
  halved the loss of every off-centreline mount and no test moved, because
  they all sat at station 0 or compared ratios;
* deleting the aeroelastic feedback (``phi_rigid * amp`` -> ``phi_rigid``)
  changed nothing any test read, because ``twist_amplification`` was gated
  only as a free function.
"""

import numpy as np
import pytest

from aerobo import carmount as cm
from aerobo import carwing, geometry, junction, slipstream
from aerobo.carwing import CarWingProblem, _strut_cd0, evaluate_car_wing
from aerobo.drag import skin_friction_cf, wing_form_factor

B = 1.6
L = 0.5 * B
W0 = 100.0
RHO, V, MU, S_REF = 1.225, 55.0, 1.789e-5, 0.4


def _uniform(n=2001, b=B, w=W0):
    """A symmetric station grid whose halves are EXACT negations.

    np.linspace(-L, L, n) is not: its samples are start + i*step, so y[i] and
    y[-1-i] differ in the last bit and the half-span machinery would compare
    two slightly different grids. Built from one half and mirrored instead, so
    a bit-for-bit gate is testing the formulas and not the grid.
    """
    yh = np.linspace(0.0, 0.5 * b, n)
    y = np.concatenate([-yh[::-1][:-1], yh])
    return y, np.full(y.size, float(w))


@pytest.fixture(scope="module")
def solved():
    """One real solved car wing: stations, load, chord, panel widths, wing."""
    prob = CarWingProblem()
    x = np.array([0.8, 0.0, -2.0, 8.0, 0.12, 0.30, B])
    out = evaluate_car_wing(x, prob)
    assert out["feasible"], out["reason"]
    res = out["vlm"]
    main = ~res.is_winglet
    return {
        "y": out["y"], "load": out["load_Npm"],
        "c": res.c[main], "width": res.width[main],
        "b": out["b_m"], "S": out["S_m2"], "q": out["q_Pa"],
        "CD": out["CD"],
        "wing": out["wing"], "prob": prob,
    }


# =====================================================================
# 1. the continuum contains the two published ends
# =====================================================================

def test_station_zero_is_the_published_cantilever_bit_for_bit():
    """A mount at the centreline IS carwing's 'centre'.

    Not "agrees with to four figures": the reaction arm max(0 - y, 0) is an
    exact 0.0 at every station on the half span, so the second term of
    equation (1) vanishes identically and the remaining integral is character
    for character carwing's. If that ever becomes only approximately true,
    the generalisation has started rewriting the published answer.
    """
    y, load = _uniform()
    mine = np.abs(cm.beam_moment(y, load, 0.0))
    theirs = carwing.bending_moment(y, load, B, "centre")
    assert np.array_equal(mine, theirs)

    idx = cm.beam_deflection_index(y, cm.beam_moment(y, load, 0.0), 0.0)
    assert idx == carwing.deflection_index(y, theirs, B, "centre")


def test_station_at_the_tip_is_the_published_simply_supported_beam():
    """A mount at the tips IS carwing's 'ends' — and is its exact NEGATION.

    carwing returns a magnitude, so the sign it threw away is recoverable
    only here: between two supports the wing SAGS (tension on the load side,
    which on a car is the suction surface), where a cantilever HOGS. Both
    facts are asserted, because the magnitude alone would pass with the sign
    convention reversed.
    """
    y, load = _uniform()
    signed = cm.beam_moment(y, load, L)
    theirs = carwing.bending_moment(y, load, B, "ends")
    assert np.array_equal(np.abs(signed), theirs)
    assert np.array_equal(-signed, theirs)          # sagging, not hogging
    assert cm.bending_sense(signed) == "sagging"
    assert cm.bending_sense(cm.beam_moment(y, load, 0.0)) == "hogging"

    idx = cm.beam_deflection_index(y, signed, L)
    assert idx == carwing.deflection_index(y, theirs, B, "ends")


def test_the_published_ends_are_reproduced_on_a_solved_load(solved):
    """The same identity on the load a VLM actually produced.

    A synthetic uniform load is symmetric by construction; a solved one is
    symmetric only because the solver made it so, on cosine stations that are
    not evenly spaced. Running the gate on both is what says the half-span
    machinery is right rather than the test grid being kind.
    """
    y, load, b = solved["y"], solved["load"], solved["b"]
    for station, name in ((0.0, "centre"), (0.5 * b, "ends")):
        mine = np.abs(cm.beam_moment(y, load, station))
        theirs = carwing.bending_moment(y, load, b, name)
        assert np.array_equal(mine, theirs), name
        assert (cm.beam_deflection_index(y, cm.beam_moment(y, load, station),
                                         station)
                == carwing.deflection_index(y, theirs, b, name)), name


def test_the_reaction_is_half_the_load_whatever_the_station(solved):
    """Moving the mount changes the DISTRIBUTION, never the reaction.

    The layout and the load are both symmetric, so each support carries half
    the total wherever it stands. That is why a wide-set mount buys nothing in
    pylon sizing and everything in bending — and it is the invariant that
    catches a reaction accidentally scaled by the station.
    """
    y, load = solved["y"], solved["load"]
    r = cm.support_reaction(y, load)
    half = float(np.trapezoid(load[y >= 0.0][np.argsort(y[y >= 0.0])],
                              np.sort(y[y >= 0.0])))
    assert r == pytest.approx(half, rel=1e-12)
    for station in (0.0, 0.2, 0.5, 0.8):
        rep = cm.beam_report(y, load, station)
        assert rep["reaction_N"] == pytest.approx(r, rel=1e-12)


# =====================================================================
# 2. the interior, against the overhanging beam's closed form
# =====================================================================

STATIONS = (0.0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.7, L)


@pytest.mark.parametrize("a", STATIONS)
def test_moment_matches_the_overhanging_beam_closed_form(a):
    """M(y) against equations (3) and (4), derived in the module docstring.

    Independent of the quadrature: the closed form is written from the
    statics, the numerics integrate the load array. A reaction applied with
    the wrong sign, or over the wrong side of the support, reproduces both
    published ends and fails every station in between.
    """
    y, load = _uniform()
    assert cm.beam_moment(y, load, a) == pytest.approx(
        cm.uniform_moment(y, W0, B, a), abs=1e-10, rel=1e-10)


@pytest.mark.parametrize("a", STATIONS)
def test_deflection_matches_the_overhanging_beam_closed_form(a):
    """Tip and centre movement relative to the mount, against the closed form.

    The closed form's own two ends are the published pair — w0 b^4/128 for the
    cantilever and 5 w0 b^4/384 for the simply supported beam — so this test
    also re-derives the two numbers carwing's own tests pin.
    """
    y, load = _uniform()
    w = cm.beam_deflection(y, cm.beam_moment(y, load, a), a)
    cf = cm.uniform_deflection(W0, B, a)
    tip = float(w[np.argmax(y)])
    centre = float(w[np.argmin(np.abs(y))])
    assert tip == pytest.approx(cf["tip"], abs=1e-9, rel=1e-5)
    assert centre == pytest.approx(cf["centre"], abs=1e-9, rel=1e-5)


def test_the_deflection_converges_on_the_closed_form_at_second_order():
    """The closed form is the LIMIT, not merely a nearby number.

    The double trapezoid is O(h^2), so halving the station spacing must
    quarter the error. That is the assertion which says the two are the same
    quantity — a formula that was merely close would not converge on it.
    """
    a = 0.4
    err = []
    for n in (251, 501, 1001):
        y, load = _uniform(n=n)
        w = cm.beam_deflection(y, cm.beam_moment(y, load, a), a)
        err.append(abs(float(w[np.argmax(y)])
                       - cm.uniform_deflection(W0, B, a)["tip"]))
    assert err[0] / err[1] == pytest.approx(4.0, rel=0.05)
    assert err[1] / err[2] == pytest.approx(4.0, rel=0.05)


def test_the_closed_form_reproduces_the_two_published_deflections():
    """The published pair, straight out of the module's own formula."""
    assert cm.uniform_deflection(W0, B, 0.0)["tip"] == pytest.approx(
        W0 * B**4 / 128.0, rel=1e-12)
    assert cm.uniform_deflection(W0, B, L)["centre"] == pytest.approx(
        5.0 * W0 * B**4 / 384.0, rel=1e-12)


def test_the_balanced_station_equalises_hogging_and_sagging():
    """At a = (b/2)(2 - sqrt 2) the two senses make the same demand.

    Equation (5). It is a root of the statics, so it is checked by evaluating
    the two peaks and comparing them — not by re-evaluating the formula.
    """
    a = cm.balanced_moment_station(B)
    y, load = _uniform()
    rep = cm.beam_report(y, load, a)
    assert rep["M_hogging_max_Nm"] == pytest.approx(
        rep["M_sagging_max_Nm"], rel=2e-3)
    assert rep["sense"] == "mixed"
    # and it lies strictly between the two layouts carwing can express
    assert 0.0 < a < L


def test_both_published_stations_are_the_worst_two_for_peak_moment():
    """The published family offers only the two ends of the demand curve.

    Measured on a uniform load: the peak |M| is w0 b^2/8 = 32 N m at BOTH
    published stations and 5.49 N m at the balanced station — a factor of 5.8.
    Asserted as the factor, computed here, so the claim cannot rot into a
    pasted constant.
    """
    y, load = _uniform()
    peak = {a: float(np.max(np.abs(cm.beam_moment(y, load, a))))
            for a in (0.0, cm.balanced_moment_station(B), L)}
    ends = [peak[0.0], peak[L]]
    best = peak[cm.balanced_moment_station(B)]
    assert ends[0] == pytest.approx(W0 * B * B / 8.0, rel=1e-4)
    assert ends[1] == pytest.approx(W0 * B * B / 8.0, rel=1e-4)
    assert min(ends) / best > 5.0


def test_an_interior_mount_is_far_stiffer_than_either_published_end():
    """The deflection index collapses in the interior.

    Swept over the semi-span on a uniform load: the best station is at
    a/L ~ 0.555 and its index is 28x below the centre mount's and 46x below
    the end mount's. The assertion is the ORDER OF MAGNITUDE and the location,
    both computed here — a mount question whose two available answers are the
    two worst is the finding this module exists to make sayable.
    """
    y, load = _uniform(n=1001)
    grid = np.linspace(0.0, L, 201)
    idx = np.array([cm.beam_deflection_index(y, cm.beam_moment(y, load, a), a)
                    for a in grid])
    best = grid[int(np.argmin(idx))]
    assert 0.5 * L < best < 0.65 * L
    assert idx[0] / idx.min() > 20.0            # vs the centre mount
    assert idx[-1] / idx.min() > 40.0           # vs the end mount


def test_the_deflection_is_zero_at_the_mount_and_the_sense_is_mixed():
    """The index measures movement RELATIVE TO THE MOUNT, so the mount is 0.

    And an interior mount puts the wing in both senses at once: hogging over
    the overhang, sagging between the supports. carwing cannot report that,
    because each of its two layouts has one sign throughout — which is also
    why it can get away with passing a magnitude into its deflection integral
    and this module cannot.
    """
    y, load = _uniform()
    a = 0.5
    m = cm.beam_moment(y, load, a)
    w = cm.beam_deflection(y, m, a)
    at_mount = float(np.interp(a, np.sort(y[y >= 0.0]),
                               w[y >= 0.0][np.argsort(y[y >= 0.0])]))
    assert at_mount == pytest.approx(0.0, abs=1e-12)
    assert cm.bending_sense(m) == "mixed"


def test_an_outboard_mount_levers_the_tip_away_from_the_load():
    """A physical prediction the two published layouts cannot make.

    With the supports outboard of about 0.55 semi-span, the sagging span
    between them rotates the wing over the support and the TIP rises AGAINST
    the load — away from the track, on a car. Measured at a = 0.6 m on a
    1.6 m span: the tip moves -1.18 (against the load) while the centre moves
    +2.34 (with it), both matching the closed form. Neither published station
    shows it: one has no interior span and the other has no overhang.
    """
    y, load = _uniform()
    a = 0.6
    w = cm.beam_deflection(y, cm.beam_moment(y, load, a), a)
    cf = cm.uniform_deflection(W0, B, a)
    tip, centre = float(w[np.argmax(y)]), float(w[np.argmin(np.abs(y))])
    assert tip < 0.0 < centre
    assert tip == pytest.approx(cf["tip"], rel=1e-5)
    assert centre == pytest.approx(cf["centre"], rel=1e-5)


def test_passing_a_magnitude_moment_gets_an_interior_mount_wrong():
    """The reason the SIGNED moment is the contract, stated as a measurement.

    carwing.deflection_index is handed |M|. That is harmless for its own two
    layouts, each of which has one sign of moment throughout, and wrong for
    every station in between, where the moment changes sign at the support.
    Fed the magnitude, the deflected shape at a = 0.6 m comes back with the
    tip moving the WRONG WAY — +1.23 instead of -1.18 — and the whole curve
    reflected. Asserted as a reversed direction, not as a tolerance, so
    nobody 'simplifies' the signed array away.
    """
    y, load = _uniform()
    a = 0.6
    m = cm.beam_moment(y, load, a)
    signed = cm.beam_deflection(y, m, a)
    magnitude = cm.beam_deflection(y, np.abs(m), a)
    tip = int(np.argmax(y))
    assert signed[tip] < 0.0 < magnitude[tip]
    assert np.max(np.abs(signed - magnitude)) > np.max(np.abs(signed))
    # ...and at both published stations the two agree exactly, bit for bit,
    # which is why the published family never had to notice
    for end in (0.0, L):
        me = cm.beam_moment(y, load, end)
        assert (cm.beam_deflection_index(y, me, end)
                == cm.beam_deflection_index(y, np.abs(me), end))


# =====================================================================
# 3. the pylon has a length
# =====================================================================

def test_the_pylon_reaches_the_deck_not_the_track():
    ride, deck = 0.30, 0.25
    assert cm.pylon_length_m(ride, deck) == pytest.approx(0.05, rel=1e-12)
    # ...and it is the same distance endplate.py makes the plate span
    from aerobo.endplate import CarWingEndplateProblem
    ep = CarWingEndplateProblem(deck_height_m=deck)
    assert cm.pylon_length_m(ride, deck) == pytest.approx(ep.reach_m(ride),
                                                          rel=1e-12)


def test_the_published_pylon_is_charged_over_six_times_its_own_length(solved):
    """carwing charges the pylon from the wing to the TRACK.

    At the published point (ride 0.30 m, deck 0.25 m) the pylon is 0.05 m
    long and 0.30 m is charged. The strut model is linear in length, so the
    charge is out by exactly that factor — asserted as the ratio, and as the
    drag-count difference, both recomputed here rather than pinned.

    The SCALE of that error used to be quoted against the family's coefficient
    drag budget. Nothing is budgeted by default any more, so it is quoted here
    against a number the family always has: the drag a solved car wing
    actually flies.
    """
    ride, deck, chord = 0.30, 0.25, 0.12
    reach = cm.pylon_length_m(ride, deck)
    assert ride / reach == pytest.approx(6.0, rel=1e-9)

    as_charged = _strut_cd0(2, ride, chord, S_REF, 0.005)
    correct = _strut_cd0(2, reach, chord, S_REF, 0.005)
    assert as_charged / correct == pytest.approx(ride / reach, rel=1e-9)
    counts = (as_charged - correct) * 1e4
    assert counts == pytest.approx(15.0, rel=1e-6)
    # for scale: those 15 counts are spent on a length of pylon that is not
    # there, and they are several per cent of the whole drag of a solved wing
    # in a family whose objective is fought over hundredths
    assert 0.03 < counts * 1e-4 / solved["CD"] < 0.10


def test_a_wing_at_its_own_deck_has_no_pylon_and_says_so_in_contract():
    """An impossible pylon is a REASON, not an exception.

    A ride height at or below the attachment deck is a design the optimiser
    can propose from inside its own box (carwing's ride band opens at 0.15 m
    and endplate.py's deck sits at 0.25 m), so it has to arrive as the
    package's ``_fail``-shaped reason string.
    """
    spec = cm.MountSpec(kind="pylon", station_frac=0.0, deck_height_m=0.25)
    assert spec.violation(B, 0.30) is None
    why = spec.violation(B, 0.20)
    assert isinstance(why, str) and "deck" in why
    # NEITHER published layout has a pylon to be impossible: both take the
    # load out through the plates, so a ride height under the deck is a design
    # they can still fly. (It used to be one of the two.)
    for name in cm.PUBLISHED_LAYOUTS:
        assert cm.published_layout(name).violation(B, 0.20) is None, name


def test_a_station_outboard_of_the_tip_is_a_reason_not_an_exception():
    spec = cm.MountSpec(kind="pylon", y_station_m=1.2)
    assert spec.violation(B, 0.30) is not None
    assert spec.violation(3.0, 0.30) is None      # a wider wing reaches it


# =====================================================================
# 4. pylon parasite drag, properly
# =====================================================================

def test_the_flat_plate_branch_is_bit_for_bit_the_published_strut_model():
    """Adoption must not move a float, so it is written to move no bits."""
    for n, length, chord in ((2, 0.30, 0.12), (2, 0.05, 0.12),
                             (1, 0.42, 0.09), (4, 0.15, 0.20)):
        got = cm.pylon_parasite_cd(n, length, chord, 0.12, S_REF,
                                   RHO, V, MU, model="flat_plate")["CD"]
        assert got == _strut_cd0(n, length, chord, S_REF, 0.005)


def test_the_buildup_charges_the_pylon_at_its_own_reynolds_number():
    """Cf on the PYLON's chord, and a form factor the crude model has not got.

    The ratio between the two models is the product of two things that can be
    written down independently — Cf(Re_pylon)/0.005 and FF(t/c) — so it is
    asserted against that product, not against a number read off the output.
    """
    chord, tc, length = 0.12, 0.12, 0.05
    bu = cm.pylon_parasite_cd(2, length, chord, tc, S_REF, RHO, V, MU,
                              model="buildup")
    fp = cm.pylon_parasite_cd(2, length, chord, tc, S_REF, RHO, V, MU,
                              model="flat_plate")
    re = RHO * V * chord / MU
    assert bu["Re"] == pytest.approx(re, rel=1e-12)
    assert bu["Cf"] == pytest.approx(skin_friction_cf(re, lref=chord),
                                     rel=1e-12)
    assert bu["FF"] == pytest.approx(wing_form_factor(tc), rel=1e-12)
    expect = (bu["Cf"] / 0.005) * bu["FF"]
    assert bu["CD"] / fp["CD"] == pytest.approx(expect, rel=1e-9)
    # measured: the crude model is OPTIMISTIC by ~31% at the same length
    assert 1.30 < bu["CD"] / fp["CD"] < 1.32


def test_the_pylons_reynolds_number_follows_its_chord_not_its_length():
    """A taller pylon is not a longer plate: the length scale is the chord."""
    a = cm.pylon_parasite_cd(2, 0.05, 0.12, 0.12, S_REF, RHO, V, MU,
                             model="buildup")
    taller = cm.pylon_parasite_cd(2, 0.50, 0.12, 0.12, S_REF, RHO, V, MU,
                                  model="buildup")
    wider = cm.pylon_parasite_cd(2, 0.05, 0.24, 0.12, S_REF, RHO, V, MU,
                                 model="buildup")
    assert taller["Re"] == a["Re"] and taller["Cf"] == a["Cf"]
    assert taller["CD"] == pytest.approx(10.0 * a["CD"], rel=1e-12)
    assert wider["Re"] == pytest.approx(2.0 * a["Re"], rel=1e-12)
    assert wider["Cf"] < a["Cf"]                  # bigger Re, thinner boundary


def test_the_two_errors_in_the_published_charge_pull_opposite_ways():
    """Length too long (conservative) times model too crude (optimistic).

    The published charge is 18.0 counts. The right length with the right
    build-up is 3.94. The net is a 14-count over-charge — so a pylon's drag is
    not being modelled to within a factor, in a family whose objective is
    fought over hundredths. (What that is a percentage OF is measured on a
    solved wing above; it is no longer a budget, because nothing is budgeted
    by default.)
    """
    ride, deck, chord, tc = 0.30, 0.25, 0.12, 0.12
    published = _strut_cd0(2, ride, chord, S_REF, 0.005)
    proper = cm.pylon_parasite_cd(2, cm.pylon_length_m(ride, deck), chord, tc,
                                  S_REF, RHO, V, MU, model="buildup")["CD"]
    assert published > proper
    assert (published - proper) * 1e4 == pytest.approx(14.06, rel=1e-3)
    assert published / proper == pytest.approx(4.57, rel=1e-2)


def test_no_pylons_are_charged_nothing():
    for model in ("buildup", "flat_plate"):
        out = cm.pylon_parasite_cd(0, 0.05, 0.12, 0.12, S_REF, RHO, V, MU,
                                   model=model)
        assert out["CD"] == 0.0 and out["Swet_m2"] == 0.0
    # ...and neither is a pylon of no length
    assert cm.pylon_parasite_cd(2, 0.0, 0.12, 0.12, S_REF, RHO, V, MU,
                                model="buildup")["CD"] == 0.0


# =====================================================================
# 5. the junction, at its own station
# =====================================================================

def _wing(taper=0.8):
    return geometry.Wing(b=B, S=S_REF, taper=taper, tc=0.12)


def test_a_centreline_mount_reproduces_the_published_junction_charge():
    """Bit for bit what carwing computes today, at station 0."""
    w = _wing()
    published = junction.junction_cd(0.12, float(w.chord(np.array([0.0]))[0]),
                                     S_REF, radius=0.0, n_junctions=2)
    assert cm.mount_junction_cd(w, 0.0, S_REF, 2) == published


@pytest.mark.parametrize("frac", (0.0, 0.35, 0.7, 1.0))
def test_the_junction_charge_scales_as_the_square_of_the_local_chord(frac):
    """Hoerner's drag AREA is built on t = (t/c) c, so it goes as c^2.

    Exactly — the correlation and the fillet credit are both homogeneous at
    fixed thickness ratio and zero radius, so this is an identity and not a
    trend. It is also the assertion that dies if the helper goes back to
    reading the root chord: on a tapered wing the ratio would be 1 everywhere.
    """
    w = _wing(taper=0.8)
    y_s = frac * 0.5 * B
    base = cm.mount_junction_cd(w, 0.0, S_REF, 2)
    got = cm.mount_junction_cd(w, y_s, S_REF, 2)
    ratio = (float(w.chord(np.array([y_s]))[0])
             / float(w.chord(np.array([0.0]))[0])) ** 2
    assert got / base == pytest.approx(ratio, rel=1e-12)


def test_moving_the_mount_outboard_cuts_the_corner_only_on_a_tapered_wing():
    """Two wings, one helper: the taper is what the station can act through.

    The untapered case is the control. A helper that ignored the station would
    pass it and fail the tapered case; a helper that used the wrong station
    would fail both.
    """
    tapered, straight = _wing(taper=0.8), _wing(taper=1.0)
    fr = np.linspace(0.0, 1.0, 6)
    ct = np.array([cm.mount_junction_cd(tapered, f * 0.5 * B, S_REF, 2)
                   for f in fr])
    cs = np.array([cm.mount_junction_cd(straight, f * 0.5 * B, S_REF, 2)
                   for f in fr])
    assert np.all(np.diff(ct) < 0.0)                     # strictly falling
    assert np.allclose(cs, cs[0], rtol=1e-12)            # nothing to act on
    assert ct[-1] / ct[0] == pytest.approx(0.8**2, rel=1e-9)   # (c_tip/c_root)^2


def test_a_corner_is_charged_per_piece_of_structure_whatever_carries_it():
    """One corner per pylon OR per extra sheet — the count follows the LOAD
    PATH, not the word describing it.

    It used to follow the pylons alone, which was right while the only
    plate-borne layout gripped at the tip with the plates the wing already
    had. It stopped being right the moment a layout gripped INBOARD: that is
    a second pair of sheets, each of which meets the wing in a corner exactly
    as a pylon does. Charged as zero, the inboard grip bought its stiffness
    for nothing — measured through the public flag, ``mount='inboard'`` with
    ``car_mount_model='continuum'`` reported the tip-borne wing's CD and the
    tip-borne wing's score while keeping 4.5x its stiffness.

    So: the tips pay no corner (their plates are there anyway), the inboard
    grip pays two, and a hand-built swan neck at the same station still pays
    two for its pylons — the term is live on both load paths, dormant only
    where there is genuinely no new structure.
    """
    w = _wing()
    assert cm.mount_junction_cd(w, 0.8, S_REF, 0) == 0.0
    tips = cm.PUBLISHED_LAYOUTS["tips"]
    assert tips.n_junctions == 0
    assert tips.junction_cd(w, S_REF) == 0.0
    inboard = cm.PUBLISHED_LAYOUTS["inboard"]
    assert inboard.n_junctions == inboard.n_sheets == 2
    assert inboard.junction_cd(w, S_REF) > 0.0
    # ...and the count agrees with what carwing's own lookup declares, or the
    # two ways of describing one mount would fly two different wings
    for name, spec in cm.PUBLISHED_LAYOUTS.items():
        assert spec.n_junctions == carwing.MOUNTS[name]["n_junctions"], name
        assert spec.n_sheets == carwing.MOUNTS[name]["n_sheets"], name
    swan = cm.MountSpec(kind="pylon", station_frac=cm.INBOARD_STATION_FRAC)
    assert swan.n_junctions == 2
    assert swan.junction_cd(w, S_REF) > 0.0


def test_an_endplate_mount_is_charged_for_the_sheets_it_ADDS():
    """The parasite charge is wetted area the wing was not already paying.

    The tip-borne layout uses the endplates the wing carries for the
    nonplanar benefit, so its load path costs exactly nothing. The inboard
    grip adds two sheets, and they are charged over the height they stand and
    the chord where they grip — both of which the caller passes, because both
    are design variables. Asserted as a RATIO so it cannot be satisfied by a
    charge that ignores its own dimensions.
    """
    tips = cm.PUBLISHED_LAYOUTS["tips"]
    inboard = cm.PUBLISHED_LAYOUTS["inboard"]
    kw = dict(s_ref=S_REF, rho=1.225, V=55.0, mu=1.81e-5)
    assert tips.pylon_cd(0.30, sheet_height_m=0.12, sheet_chord_m=0.25,
                         **kw)["CD"] == 0.0
    base = inboard.pylon_cd(0.30, sheet_height_m=0.12, sheet_chord_m=0.25,
                            **kw)["CD"]
    assert base > 0.0
    taller = inboard.pylon_cd(0.30, sheet_height_m=0.24, sheet_chord_m=0.25,
                              **kw)["CD"]
    longer = inboard.pylon_cd(0.30, sheet_height_m=0.12, sheet_chord_m=0.50,
                              **kw)["CD"]
    assert taller / base == pytest.approx(2.0, rel=1e-9)
    assert longer / base == pytest.approx(2.0, rel=1e-9)
    # ...and NOT on the ride height: there is no pylon reaching the deck
    assert inboard.pylon_cd(0.55, sheet_height_m=0.12, sheet_chord_m=0.25,
                            **kw)["CD"] == base


def test_a_mount_cannot_be_charged_for_two_load_paths_at_once():
    """``n_sheets`` describes the plates an ENDPLATE mount adds, so a pylon
    mount carrying them as well would price one load path twice."""
    with pytest.raises(ValueError, match="one load path twice|n_sheets"):
        cm.MountSpec(kind="pylon", station_frac=0.0, n_pylons=2, n_sheets=2)


# =====================================================================
# 6. torsion, and a chordwise attachment point
# =====================================================================

def test_a_mount_at_the_aerodynamic_centre_winds_the_wing_up_by_exactly_nothing(
        solved):
    """The OFF invariant of the whole torsion model, asserted with ``==``.

    x_attach - x_ac is a difference of two equal floats, so the sectional
    torque is an exact array of zeros, the twist is exact zeros, there is no
    divergence at any speed, and the amplification is exactly 1.0. A module
    that changed a published answer by existing would fail here first.
    """
    y, load, c, b = solved["y"], solved["load"], solved["c"], solved["b"]
    spec = cm.MountSpec(kind="pylon", station_frac=0.0)
    assert spec.x_attach_frac == cm.X_AC_FRAC and spec.e_frac == 0.0
    m = cm.torsion_moment(y, load, c, spec.x_attach_frac)
    assert np.all(m == 0.0)
    t = spec.torsion(y, load, c, b, solved["S"], solved["wing"].mac,
                     solved["q"])
    assert np.all(t["phi_rad"] == 0.0)
    assert t["q_div_Pa"] == float("inf")
    assert t["amplification"] == 1.0
    assert t["phi_max_deg"] == 0.0


def test_the_windup_reverses_with_the_sign_of_the_offset(solved):
    """Aft of the quarter chord the wing winds ON, ahead of it it washes OFF.

    Equal and opposite offsets give equal and opposite twist distributions,
    which is the statement that the moment is linear in the offset and that
    the sign convention is carried consistently through the integration.
    """
    y, load, c, b = solved["y"], solved["load"], solved["c"], solved["b"]
    aft = cm.torsion_twist(y, cm.torsion_moment(y, load, c, 0.35), 0.0, 4.0e4)
    fwd = cm.torsion_twist(y, cm.torsion_moment(y, load, c, 0.15), 0.0, 4.0e4)
    assert np.max(aft) > 0.0 and np.min(fwd) < 0.0
    assert aft == pytest.approx(-fwd, rel=1e-9, abs=1e-18)


@pytest.mark.parametrize("a", STATIONS)
def test_twist_matches_the_uniform_torque_closed_form(a):
    """phi(0) = m a^2/(2 GJ) and phi(L) = m (L-a)^2/(2 GJ).

    The step in internal torque at the support is integrated about the
    station rather than smeared over a panel, so this holds at interior
    stations to the quadrature's own accuracy and not to one panel width.
    """
    gj, m0 = 4.0e4, 5.0
    y, _ = _uniform()
    m = np.full(y.size, m0)
    phi = cm.torsion_twist(y, m, a, gj)
    cf = cm.uniform_twist(m0, B, a, gj)
    assert float(phi[np.argmax(y)]) == pytest.approx(cf["tip"], abs=1e-14,
                                                     rel=1e-9)
    assert float(phi[np.argmin(np.abs(y))]) == pytest.approx(
        cf["centre"], abs=1e-14, rel=1e-9)


def test_both_published_layouts_twist_by_the_same_peak():
    """A torsion cantilever of length L and a fixed-fixed shaft of length 2L
    under the same distributed torque twist by exactly m L^2/(2 GJ).

    A real identity, not a coincidence of the discretisation — which is why
    the single-degree-of-freedom reduction gives the two published layouts the
    SAME divergence speed even though their distributions differ.
    """
    gj, m0 = 4.0e4, 5.0
    y, _ = _uniform()
    m = np.full(y.size, m0)
    peak_c = float(np.max(np.abs(cm.torsion_twist(y, m, 0.0, gj))))
    peak_e = float(np.max(np.abs(cm.torsion_twist(y, m, L, gj))))
    expect = m0 * L * L / (2.0 * gj)
    assert peak_c == pytest.approx(expect, rel=1e-9)
    assert peak_e == pytest.approx(expect, rel=1e-9)
    assert peak_c == pytest.approx(peak_e, rel=1e-9)


def test_an_interior_mount_is_four_times_stiffer_in_torsion():
    """max(a, L-a)^2 is minimised at mid-semispan, and the factor is 4.

    The same statement twice — once through the effective length (10) and once
    through the twist the distributed model actually produces — so a change to
    one that did not reach the other is caught.
    """
    gj, m0 = 4.0e4, 5.0
    y, _ = _uniform()
    m = np.full(y.size, m0)
    assert (cm.torsion_effective_length(B, 0.0)
            / cm.torsion_effective_length(B, 0.5 * L)) == pytest.approx(
        4.0, rel=1e-12)
    peak_end = float(np.max(np.abs(cm.torsion_twist(y, m, 0.0, gj))))
    peak_mid = float(np.max(np.abs(cm.torsion_twist(y, m, 0.5 * L, gj))))
    assert peak_end / peak_mid == pytest.approx(4.0, rel=1e-6)
    # the effective torsion length at either published station is b/4: a
    # distributed torque acts at half the physical length
    assert cm.torsion_effective_length(B, 0.0) == pytest.approx(B / 4.0,
                                                                rel=1e-12)
    assert cm.torsion_effective_length(B, L) == pytest.approx(B / 4.0,
                                                              rel=1e-12)


def test_the_divergence_speed_is_the_hand_derivation(solved):
    """q_div against equation (12), written out again here from scratch.

    q_div = GJ_eff/(a S MAC ehat) with GJ_eff = 2 GJ / l_eff. At the published
    stations l_eff = b/4, so this reduces to 8 GJ/(a S MAC ehat b) — spelled
    out independently, because a formula tested against itself is not tested.
    """
    b, s_ref, mac = solved["b"], solved["S"], solved["wing"].mac
    gj, a_lin, e = 4.0e4, 2.0 * np.pi, 0.10
    got = cm.divergence_q(gj, b, s_ref, mac, a_lin, e, 0.0)
    assert got == pytest.approx(8.0 * gj / (a_lin * s_ref * mac * e * b),
                                rel=1e-12)
    assert cm.divergence_q(gj, b, s_ref, mac, a_lin, e, 0.5 * b) == \
        pytest.approx(got, rel=1e-12)
    # and moving the mount inboard from either end raises it by 4x
    assert cm.divergence_q(gj, b, s_ref, mac, a_lin, e,
                           0.25 * b) == pytest.approx(4.0 * got, rel=1e-12)


def test_divergence_is_impossible_at_or_ahead_of_the_aerodynamic_centre(
        solved):
    """Negative feedback has no divergence speed, at any stiffness.

    A mount ahead of the aerodynamic centre sheds incidence as the wing loads
    up. That is a physical statement and it is returned as an infinite q_div,
    an infinite margin and an amplification of exactly 1.0 — not as a guard
    against dividing by zero.
    """
    b, s_ref, mac = solved["b"], solved["S"], solved["wing"].mac
    for e in (-0.20, -0.01, 0.0):
        qd = cm.divergence_q(4.0e4, b, s_ref, mac, 2 * np.pi, e, 0.0)
        assert qd == float("inf")
        assert cm.divergence_margin(1000.0, qd) == float("inf")
        assert cm.twist_amplification(1000.0, qd) == 1.0


def test_the_twist_grows_without_bound_towards_divergence():
    """1/(1 - q/q_div): monotone, exactly 1 at rest, unbounded at the limit.

    The trend is what is asserted, never the singularity — the model has
    nothing to say at q_div and says so by returning inf rather than a
    negative twist, which is what the formula would give if read past it.
    """
    q_div = 5.0e3
    qs = np.array([0.0, 1.0e3, 2.0e3, 4.0e3, 4.9e3, 4.99e3])
    amp = np.array([cm.twist_amplification(q, q_div) for q in qs])
    assert amp[0] == 1.0
    assert np.all(np.diff(amp) > 0.0)
    assert amp[-1] > 100.0
    assert cm.twist_amplification(q_div, q_div) == float("inf")
    assert cm.twist_amplification(2.0 * q_div, q_div) == float("inf")


def test_a_stiff_wing_recovers_the_rigid_answer(solved):
    """GJ -> infinity: no twist, and no amplification of it either.

    Both halves matter. A model that drove the twist to zero but left the
    amplification finite-but-not-one would still be multiplying a zero by
    something, and would come apart the moment the twist was fed back into a
    load.
    """
    y, load, c, b = solved["y"], solved["load"], solved["c"], solved["b"]
    prev = None
    for gj in (4.0e4, 4.0e6, 4.0e8, 4.0e30):
        spec = cm.MountSpec(kind="pylon", station_frac=0.0,
                            x_attach_frac=0.45, gj_nm2=gj)
        t = spec.torsion(y, load, c, b, solved["S"], solved["wing"].mac,
                         solved["q"])
        if prev is not None:
            assert t["phi_max_deg"] < prev
        prev = t["phi_max_deg"]
    assert prev == pytest.approx(0.0, abs=1e-20)
    assert t["amplification"] == 1.0


def test_the_published_wing_is_torsionally_rigid_and_says_so(solved):
    """Measured, so the module cannot be read as claiming a live constraint.

    At the published operating point with the mount a tenth of a chord aft of
    the aerodynamic centre, q_div is about 3.1e6 Pa against q = 1853 Pa: a
    margin near 1700, and a peak windup of 0.004 deg. Divergence is not a
    binding constraint for this family at this spar stiffness. The stiffness
    at which it WOULD bind is derived here rather than asserted from memory.
    """
    y, load, c, b = solved["y"], solved["load"], solved["c"], solved["b"]
    spec = cm.MountSpec(kind="pylon", station_frac=0.0, x_attach_frac=0.35)
    t = spec.torsion(y, load, c, b, solved["S"], solved["wing"].mac,
                     solved["q"])
    assert t["g_divergence"] > 1000.0
    assert t["phi_max_deg"] < 0.01
    # q_div is linear in GJ, so the critical stiffness follows exactly
    gj_crit = spec.gj_nm2 * solved["q"] / t["q_div_Pa"]
    assert 20.0 < gj_crit < 30.0
    critical = cm.MountSpec(kind="pylon", station_frac=0.0,
                            x_attach_frac=0.35, gj_nm2=gj_crit)
    assert critical.torsion(y, load, c, b, solved["S"],
                            solved["wing"].mac, solved["q"]
                            )["g_divergence"] == pytest.approx(0.0, abs=1e-9)


def test_the_aeroelastic_feedback_reaches_the_twist_it_is_applied_to(solved):
    """MUTATION THIS TEST IS BUILT TO CATCH: ``phi = phi_rigid * amp`` becoming
    ``phi = phi_rigid * 1.0`` in :meth:`MountSpec.torsion` — deleting the whole
    single-mode feedback the method exists to apply, and collapsing the
    ``phi_rigid_rad`` / ``phi_rad`` distinction the returned dict carries.

    ``twist_amplification`` is gated above as a free function, and nothing
    asserted that its answer ever arrived anywhere. The stiffness is chosen
    here to make the factor exactly 2: q_div is linear in GJ, so setting GJ to
    twice the critical stiffness puts q_div at 2q and 1/(1 - q/q_div) at 2.
    Then the array, its peak and the reported degrees are each asserted
    against that factor, and a second stiffness checks it is the LAW that is
    applied and not just some constant greater than one.
    """
    y, load, c, b = solved["y"], solved["load"], solved["c"], solved["b"]
    q, s_ref, mac = solved["q"], solved["S"], solved["wing"].mac

    def at(gj):
        return cm.MountSpec(kind="pylon", station_frac=0.0, x_attach_frac=0.35,
                            gj_nm2=gj).torsion(y, load, c, b, s_ref, mac, q)

    ref = cm.MountSpec(kind="pylon", station_frac=0.0, x_attach_frac=0.35)
    gj_crit = ref.gj_nm2 * q / ref.torsion(y, load, c, b, s_ref, mac,
                                           q)["q_div_Pa"]
    t = at(2.0 * gj_crit)
    assert t["q_div_Pa"] == pytest.approx(2.0 * q, rel=1e-9)
    assert t["amplification"] == pytest.approx(2.0, rel=1e-9)
    # a hand-written 1/(1 - q/q_div), so the dict is not checked against itself
    assert t["amplification"] == pytest.approx(1.0 / (1.0 - q / t["q_div_Pa"]),
                                               rel=1e-12)
    # the rigid twist is a real number, so doubling it is a real difference
    assert t["phi_rigid_max_deg"] > 0.5
    assert not np.array_equal(t["phi_rad"], t["phi_rigid_rad"])
    assert np.array_equal(t["phi_rad"],
                          t["phi_rigid_rad"] * t["amplification"])
    assert float(np.max(np.abs(t["phi_rad"]))) == pytest.approx(
        2.0 * float(np.max(np.abs(t["phi_rigid_rad"]))), rel=1e-12)
    assert t["phi_max_deg"] == pytest.approx(2.0 * t["phi_rigid_max_deg"],
                                             rel=1e-12)
    # ...and it is the LAW, not a constant greater than one: four times the
    # critical stiffness puts q_div at 4q and the amplification at 4/3, while
    # the RIGID twist it multiplies has itself halved (phi_rigid goes as 1/GJ),
    # so the flexible peaks must stand in the ratio 2 / (0.5 * 4/3) = 3.
    stiff = at(4.0 * gj_crit)
    assert stiff["amplification"] == pytest.approx(4.0 / 3.0, rel=1e-9)
    assert stiff["phi_rigid_max_deg"] == pytest.approx(
        0.5 * t["phi_rigid_max_deg"], rel=1e-12)
    assert t["phi_max_deg"] / stiff["phi_max_deg"] == pytest.approx(3.0,
                                                                   rel=1e-9)


# =====================================================================
# 7. the suction-side loss
# =====================================================================

def test_the_loss_is_off_by_default_bit_for_bit(solved):
    """The default modifiers are EXACTLY 1.0 — asserted with ``==``.

    The whole point of a calibration whose default is zero is that a run which
    does not set it is the run that was published, so this is an equality and
    not a tolerance.
    """
    y, c, b, width = (solved["y"], solved["c"], solved["b"], solved["width"])
    spec = cm.MountSpec(kind="pylon", station_frac=0.0)
    assert spec.suction_loss == 0.0
    for kw in ({}, {"width": width}):
        mods = spec.modifiers(y, c, b, **kw)
        assert np.all(mods == 1.0)
        assert mods.dtype == np.dtype(float)


def test_a_pressure_side_mount_never_loses_lift(solved):
    """The swan neck's whole argument, and why the knob has a geometry OFF.

    A pylon that attaches on the pressure side leaves the suction surface —
    the one facing the track on an inverted wing, and the one the downforce
    comes from — undisturbed. So the loss does not apply at all, at any
    magnitude, and the modifiers are exactly 1.0 rather than nearly so.
    """
    y, c, b, width = (solved["y"], solved["c"], solved["b"], solved["width"])
    swan = cm.MountSpec(kind="pylon", y_station_m=0.28, side="pressure",
                        suction_loss=0.9)
    assert np.all(swan.modifiers(y, c, b, width=width) == 1.0)
    under = cm.MountSpec(kind="pylon", y_station_m=0.28, side="suction",
                         suction_loss=0.9)
    assert np.min(under.modifiers(y, c, b, width=width)) < 1.0


def test_an_endplate_mount_has_no_footprint(solved):
    y, c, b, width = (solved["y"], solved["c"], solved["b"], solved["width"])
    plates = cm.MountSpec(kind="endplate", station_frac=1.0, n_pylons=0,
                          suction_loss=0.5)
    assert plates.footprint_stations_m(b).size == 0
    assert np.all(plates.modifiers(y, c, b, width=width) == 1.0)


def test_the_footprint_is_derived_from_the_pylon_it_comes_from():
    """half-width = spread * t_pylon / 2, and the flat core is the pylon.

    Every length here comes out of the pylon's own geometry and one stated
    spread; nothing is a second free number. At the published pylon
    (chord 0.12 m, t/c 0.12) that is a 0.043 m band — 2.7% of a 1.6 m span.
    """
    chord, tc, spread = 0.12, 0.12, cm.DEFAULT_LOSS_SPREAD
    hw = cm.footprint_half_width_m(chord, tc, spread)
    assert hw == pytest.approx(0.5 * spread * tc * chord, rel=1e-12)
    assert 2.0 * hw / B == pytest.approx(0.027, abs=5e-4)
    # the loss is full over the pylon's own thickness and gone at the edge
    y = np.array([0.0, 0.5 * tc * chord, hw, 2.0 * hw])
    k = cm.suction_loss_modifiers(y, [0.0], hw, 0.4,
                                  core_frac=1.0 / spread)
    assert k[0] == pytest.approx(0.6, rel=1e-12)
    assert k[1] == pytest.approx(0.6, rel=1e-12)     # still inside the core
    assert k[2] == 1.0                               # exactly zero at the edge
    assert k[3] == 1.0
    assert k[0] < k[2]
    with pytest.raises(ValueError):
        cm.footprint_half_width_m(chord, tc, 0.5)    # narrower than the pylon


def test_a_footprint_narrower_than_a_panel_is_refused_not_silently_lost(
        solved):
    """The bug this refusal exists to make impossible.

    On the published 40-panel cosine grid the centre panels are 0.063 m wide
    and a centre-mounted pylon's footprint is 0.043 m, so it falls BETWEEN two
    stations: point-sampling a live 20% knockdown returns exactly 1.0 at every
    station, i.e. a calibration the user switched on reading as its own OFF
    default. Passing the panel widths integrates the footprint instead and
    recovers a real 0.44% loss on the same grid.
    """
    y, c, b, width = (solved["y"], solved["c"], solved["b"], solved["width"])
    load = solved["load"]
    hw = cm.footprint_half_width_m(0.12, 0.12)
    assert not cm.footprint_is_resolved(y, [0.0], hw)
    # the silent null, demonstrated on the raw helper
    naive = cm.suction_loss_modifiers(y, [0.0], hw, 0.2,
                                      core_frac=1.0 / cm.DEFAULT_LOSS_SPREAD)
    assert np.all(naive == 1.0)
    # ...which the spec refuses to hand back
    spec = cm.MountSpec(kind="pylon", station_frac=0.0, suction_loss=0.2)
    with pytest.raises(ValueError, match="does not resolve"):
        spec.modifiers(y, c, b)
    # ...and the panel average removes a real, and much smaller, amount
    mods = spec.modifiers(y, c, b, width=width)
    assert np.min(mods) < 1.0
    kd = cm.lift_knockdown(load, width, mods)
    assert kd == pytest.approx(0.0044, rel=0.05)


def _phase_grid(r, phase, hw, n_half=8):
    """A uniform station grid of panel width ``r`` footprint half-widths.

    ``phase`` slides the whole grid between two stations. It is the thing a
    RESOLUTION test must not be sensitive to and the proximity count was: at
    fixed panel width, the phase alone moved the recovered integral of the
    published pylon's footprint from -98% to +28%.
    """
    h = r * float(hw)
    n = int(np.ceil(n_half / r)) + n_half
    return (np.arange(-n, n + 1) + phase) * h


def _old_proximity_count(y, stations, hw):
    """The criterion ``footprint_is_resolved`` USED to apply, written out here.

    ``count_nonzero(|y - y_p| < half_width) >= 2``. Kept in the test file so
    that "put the count back" is a mutation these tests can actually run,
    rather than a claim about a version of the module nobody can execute.
    """
    inside = np.zeros(np.asarray(y, dtype=float).shape, dtype=bool)
    for p in np.atleast_1d(np.asarray(stations, dtype=float)):
        inside |= np.abs(y - float(p)) < float(hw)
    return bool(np.count_nonzero(inside) >= 2)


def test_the_footprint_guard_measures_the_integral_not_a_proximity_count():
    """MUTATION THIS TEST IS BUILT TO CATCH: reverting ``footprint_is_resolved``
    to the proximity COUNT it used to be, ``count_nonzero(|y - y_p| <
    half_width) >= 2``.

    The count is not a resolution test, because the raised cosine of (13)
    tapers to ZERO at the edge of the footprint: two stations can sit inside
    the band and sample almost none of it. Both directions are demonstrated
    here on grids the count passes — one that recovers 0.7% of the loss and
    one that removes 25% more lift than the footprint contains — and the
    guard has to refuse both.
    """
    hw = cm.footprint_half_width_m(0.12, 0.12)
    core = 1.0 / cm.DEFAULT_LOSS_SPREAD

    # (i) two stations inside the band, both on the taper's outer skirt
    y = _phase_grid(1.96, 0.5, hw)
    inside = np.abs(y) < hw
    assert int(np.count_nonzero(inside)) == 2
    assert np.allclose(np.abs(y[inside]) / hw, 0.98)
    assert _old_proximity_count(y, [0.0], hw)          # the count passed it
    err = cm.footprint_integral_error(y, [0.0], hw, core_frac=core)
    assert err == pytest.approx(-0.9935, abs=2e-3)     # 0.7% of the loss
    assert not cm.footprint_is_resolved(y, [0.0], hw, core_frac=core)

    # (ii) the same count also passes grids that remove MORE than there is
    over = _phase_grid(0.9, 0.5, hw)
    assert _old_proximity_count(over, [0.0], hw)
    assert cm.footprint_integral_error(
        over, [0.0], hw, core_frac=core) == pytest.approx(0.2505, abs=3e-3)
    assert not cm.footprint_is_resolved(over, [0.0], hw, core_frac=core)

    # (iii) and the spec refuses instead of handing the knockdown back
    spec = cm.MountSpec(kind="pylon", station_frac=0.0, suction_loss=0.2)
    for bad in (y, over):
        with pytest.raises(ValueError, match="does not resolve"):
            spec.modifiers(bad, np.full(bad.size, 0.25), B)

    # (iv) the census the guard's own docstring quotes, re-derived rather than
    # pasted: over 2400 uniform grids (panel width 0.05-3 half-widths, 40
    # phases each) the count passed 1080, and among those the recovered
    # integral ran from -98% to +28% with 28% of them out by more than 10%
    grids = [_phase_grid(r, d, hw, n_half=6)
             for r in np.linspace(0.05, 3.0, 60)
             for d in np.linspace(0.0, 0.999, 40)]
    counted = [cm.footprint_integral_error(g, [0.0], hw, core_frac=core)
               for g in grids if _old_proximity_count(g, [0.0], hw)]
    assert len(grids) == 2400
    assert len(counted) == 1080
    assert min(counted) == pytest.approx(-0.98, abs=0.01)
    assert max(counted) == pytest.approx(+0.28, abs=0.01)
    assert np.mean([abs(e) > 0.10 for e in counted]) == pytest.approx(
        0.28, abs=0.01)
    # ...and the guard's verdict is now the recovery itself, so it is a
    # property of the GRID where the count was a property of the phase: every
    # grid with panels under half a footprint half-width passes at EVERY phase
    assert all(cm.footprint_is_resolved(g, [0.0], hw, core_frac=core)
               for g in grids
               if float(np.max(np.diff(np.sort(g)))) <= 0.5 * hw)


def test_the_footprint_guard_is_checked_at_its_own_tolerance_from_both_sides():
    """MUTATION THIS TEST IS BUILT TO CATCH: moving FOOTPRINT_INTEGRAL_TOL,
    dropping the ``abs()`` around the measured error, or ignoring the ``tol``
    argument and hard-coding the module default.

    The threshold was untested. Here it is bisected: at a fixed phase the
    recovered integral crosses 5% at a panel width of 0.7012 half-widths —
    2.85 stations across the footprint's full width — and the guard's verdict
    has to flip there and only there.
    """
    hw = cm.footprint_half_width_m(0.12, 0.12)
    core = 1.0 / cm.DEFAULT_LOSS_SPREAD
    tol = cm.FOOTPRINT_INTEGRAL_TOL

    def err(r):
        return cm.footprint_integral_error(_phase_grid(r, 0.5, hw), [0.0], hw,
                                           core_frac=core)

    lo, hi = 0.4, 1.0
    assert abs(err(lo)) < tol < abs(err(hi))
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if abs(err(mid)) <= tol:
            lo = mid
        else:
            hi = mid
    assert lo == pytest.approx(0.7012, abs=2e-3)
    assert 2.0 / lo == pytest.approx(2.85, abs=0.02)   # stations per footprint
    # both sides of the crossing, on the guard itself
    assert abs(err(lo)) <= tol < abs(err(hi))
    assert cm.footprint_is_resolved(_phase_grid(lo, 0.5, hw), [0.0], hw,
                                    core_frac=core)
    assert not cm.footprint_is_resolved(_phase_grid(hi, 0.5, hw), [0.0], hw,
                                        core_frac=core)
    # the tolerance is CONSULTED, not baked in
    over = _phase_grid(0.9, 0.5, hw)
    assert not cm.footprint_is_resolved(over, [0.0], hw, core_frac=core)
    assert cm.footprint_is_resolved(over, [0.0], hw, core_frac=core, tol=0.30)
    # ...and an UNDER-recovering grid cannot buy a pass with its sign: this one
    # is 99% short, so it must be refused at every tolerance below 0.99
    starved = _phase_grid(1.96, 0.5, hw)
    assert not cm.footprint_is_resolved(starved, [0.0], hw, core_frac=core,
                                        tol=0.90)


def test_the_footprint_reference_integral_is_the_analytic_one():
    """The guard is only as good as what it calls exact, so that is gated too.

    MUTATIONS THIS TEST IS BUILT TO CATCH: summing the footprints' reference
    integrals instead of merging their UNION (which double-counts an overlap
    the shape takes a max over), and dropping the clip of the reference to the
    span the panels actually cover (which would turn a tip-mounted footprint,
    half of which hangs off the wing, into a permanent resolution failure).

    The isolated footprint's integral is closed form::

        int_-w^w f(|y|/w) dy = 2w int_0^1 f(s) ds = w (1 + core_frac)

    since the raised cosine averages exactly 1/2 over its taper. On a grid
    fine enough that its own sum IS that number, whatever error the helper
    reports is the reference's own — measured below 2e-8 at three core
    fractions, six orders under the 5% it is compared against.
    """
    hw = cm.footprint_half_width_m(0.12, 0.12)
    fine = np.linspace(-0.3, 0.3, 60001)
    w = cm.station_widths(fine)
    for core in (0.0, 1.0 / cm.DEFAULT_LOSS_SPREAD, 0.6):
        closed = hw * (1.0 + core)
        got = float(np.sum(slipstream.axial_shape(np.abs(fine) / hw, core) * w))
        assert got == pytest.approx(closed, rel=1e-9)      # the closed form
        assert abs(cm.footprint_integral_error(
            fine, [0.0], hw, core_frac=core)) < 2e-8       # the reference
    core = 1.0 / cm.DEFAULT_LOSS_SPREAD
    # OVERLAPPING footprints: the max rule means the union is integrated once,
    # so a fine grid still reports no error. Summed instead, it would report
    # about -0.4 on this pair.
    for stations in ([-0.4 * hw, 0.4 * hw], [-0.02, 0.0, 0.02]):
        assert abs(cm.footprint_integral_error(
            fine, stations, hw, core_frac=core)) < 1e-5
    # COVERAGE IS NOT RESOLUTION: half of this footprint is off the panels
    edge = np.linspace(-0.3, 0.0, 30001)
    covered = float(np.sum(
        slipstream.axial_shape(np.abs(edge) / hw, core)
        * cm.station_widths(edge)))
    assert covered / (hw * (1.0 + core)) == pytest.approx(0.5, abs=1e-3)
    assert abs(cm.footprint_integral_error(
        edge, [0.0], hw, core_frac=core)) < 1e-6          # still resolved


def test_the_footprint_tolerance_sits_on_a_plateau_that_is_measured_here():
    """The measurement behind FOOTPRINT_INTEGRAL_TOL, re-derived not pinned.

    A docstring number the code no longer produces is a defect, and that
    constant's docstring quotes an error envelope and three cut points. Both
    are recomputed here from the shipped helper. The point of the envelope is
    that the cut is insensitive to the tolerance: 2%, 5% and 10% land within
    a quarter of a panel width of each other, which is why choosing 5% decides
    almost nothing while choosing a proximity count decided everything.
    """
    hw = cm.footprint_half_width_m(0.12, 0.12)
    core = 1.0 / cm.DEFAULT_LOSS_SPREAD
    phases = np.linspace(0.0, 1.0, 120, endpoint=False)

    def envelope(r):
        return max(abs(cm.footprint_integral_error(
            _phase_grid(r, d, hw), [0.0], hw, core_frac=core)) for d in phases)

    for r, quoted in ((0.2, 0.00105), (0.5, 0.0185), (0.6, 0.0510),
                      (0.8, 0.1706), (1.0, 0.2803)):
        assert envelope(r) == pytest.approx(quoted, rel=0.02), r
    # the cut, per tolerance: the coarsest grid that passes at EVERY phase
    rs = np.round(np.arange(0.30, 1.001, 0.01), 4)
    err = np.array([[abs(cm.footprint_integral_error(
        _phase_grid(r, d, hw), [0.0], hw, core_frac=core)) for d in phases]
        for r in rs])
    cut = {}
    for tol in (0.02, 0.05, 0.10):
        bad = np.where(~(err <= tol).all(axis=1))[0]
        cut[tol] = float(rs[bad[0] - 1])
    assert cut == {0.02: pytest.approx(0.50), 0.05: pytest.approx(0.57),
                   0.10: pytest.approx(0.73)}
    assert cut[0.10] - cut[0.02] < 0.25            # the plateau, in r
    # stated the way the constant's docstring states it: stations across the
    # footprint's full width
    assert 2.0 / cut[0.02] == pytest.approx(4.0, abs=0.05)
    assert 2.0 / cut[0.10] == pytest.approx(2.7, abs=0.05)


def test_the_panel_average_agrees_with_the_point_sample_where_both_are_valid():
    """On a grid that resolves the footprint the two paths must not differ.

    Otherwise the averaging is not a quadrature of the same law, it is a
    second model — and the refusal above would be hiding a discrepancy rather
    than a resolution failure.
    """
    hw = cm.footprint_half_width_m(0.12, 0.12)
    y = np.linspace(-0.2, 0.2, 401)
    width = np.full(y.size, float(y[1] - y[0]))
    assert cm.footprint_is_resolved(y, [0.05], hw)
    point = cm.suction_loss_modifiers(y, [0.05], hw, 0.3, core_frac=1 / 3)
    avg = cm.suction_loss_modifiers(y, [0.05], hw, 0.3, core_frac=1 / 3,
                                    width=width)
    assert avg == pytest.approx(point, abs=2e-3)


def test_the_knockdown_is_linear_in_the_magnitude_the_user_owns(solved):
    """It is a calibration, so it must scale exactly with what is calibrated.

    A knockdown that was not proportional to its own magnitude would be
    hiding a second, unstated number inside the shape.
    """
    y, c, b = solved["y"], solved["c"], solved["b"]
    load, width = solved["load"], solved["width"]
    kd = []
    for k0 in (0.1, 0.2, 0.4):
        spec = cm.MountSpec(kind="pylon", y_station_m=0.28, suction_loss=k0)
        kd.append(cm.lift_knockdown(load, width,
                                    spec.modifiers(y, c, b, width=width)))
    assert kd[1] == pytest.approx(2.0 * kd[0], rel=1e-9)
    assert kd[2] == pytest.approx(4.0 * kd[0], rel=1e-9)
    assert 0.0 < kd[0] < 0.01


def test_the_knockdown_follows_the_load_under_the_pylon(solved):
    """Where the pylon stands decides what it costs.

    Same pylon, same magnitude, three stations: the loss tracks the sectional
    load it stands on, so a tip-mounted footprint costs far less than one
    inboard. Measured rather than assumed — it is the only thing in this
    module that makes the footprint's POSITION an aerodynamic question and not
    just a structural one.
    """
    y, c, b = solved["y"], solved["c"], solved["b"]
    load, width = solved["load"], solved["width"]
    out = {}
    for st in (0.0, 0.28, 0.5 * b):
        spec = cm.MountSpec(kind="pylon", y_station_m=st, suction_loss=0.2)
        out[st] = cm.lift_knockdown(load, width,
                                    spec.modifiers(y, c, b, width=width))
    assert out[0.28] > out[0.5 * b]
    assert all(v > 0.0 for v in out.values())


def test_an_off_centreline_mount_loses_lift_on_both_sides_of_the_wing(solved):
    """MUTATION THIS TEST IS BUILT TO CATCH: ``MountSpec.footprint_stations_m``
    returning ``np.array([y_s])`` instead of ``np.array([-y_s, y_s])`` —
    deleting the mirrored footprint and exactly HALVING the lift loss of every
    mount that is not on the centreline.

    Nothing else in this file could see it. The tests at station 0 have one
    footprint anyway (the mirror of 0 is the same band), and the tests that
    compare RATIOS — the knockdown against its own magnitude, one station
    against another — halve on both sides together. So the two things asserted
    here are the ones a ratio cannot carry: the ABSOLUTE knockdown, against a
    number built from the solved load and the footprint's closed-form integral
    w (1 + core_frac), and the SYMMETRY of the modifier array in y.
    """
    y, load, c = solved["y"], solved["load"], solved["c"]
    b, width = solved["b"], solved["width"]
    hw = cm.footprint_half_width_m(0.12, 0.12)
    core = 1.0 / cm.DEFAULT_LOSS_SPREAD
    mag = 0.2
    total = float(np.sum(load * width))
    assert np.allclose(y, -y[::-1], atol=1e-14)        # the grid is a mirror

    # a mount 0.28 m out: TWO footprints, so twice one footprint's deficit
    spec = cm.MountSpec(kind="pylon", y_station_m=0.28, suction_loss=mag)
    mods = spec.modifiers(y, c, b, width=width)
    kd = cm.lift_knockdown(load, width, mods)
    expect = mag * (1.0 + core) * hw * 2.0 * float(np.interp(0.28, y, load))
    assert kd == pytest.approx(expect / total, rel=0.01)   # absolute, not a ratio
    assert np.max(np.abs(mods - mods[::-1])) < 1e-12       # symmetric in y
    assert float(np.min(mods[y < 0.0])) < 1.0          # and the port side pays

    # the same footprint on ONE side only, built by hand: exactly half
    one = cm.suction_loss_modifiers(y, [0.28], hw, mag, core_frac=core,
                                    width=width)
    assert kd / cm.lift_knockdown(load, width, one) == pytest.approx(2.0,
                                                                    rel=1e-9)
    assert np.max(np.abs(one - one[::-1])) > 0.05      # one side is NOT a mount

    # a centreline mount is the case where one footprint is the whole story,
    # which is exactly why it hid this: its knockdown is the SINGLE band's
    centre = cm.MountSpec(kind="pylon", station_frac=0.0, suction_loss=mag)
    kd0 = cm.lift_knockdown(load, width,
                            centre.modifiers(y, c, b, width=width))
    expect0 = mag * (1.0 + core) * hw * float(np.interp(0.0, y, load))
    assert kd0 == pytest.approx(expect0 / total, rel=0.01)


def test_the_loss_never_superposes_where_footprints_overlap():
    """Two overlapping bands take the max, not the sum (slipstream's rule).

    A summed loss could drive the multiplier negative — a strip producing
    negative lift because two pylons stood near each other — which is not a
    knockdown, it is a modelling failure.
    """
    hw = 0.05
    y = np.linspace(-0.1, 0.1, 401)
    one = cm.suction_loss_modifiers(y, [0.0], hw, 0.6)
    two = cm.suction_loss_modifiers(y, [-0.01, 0.01], hw, 0.6)
    assert np.min(two) >= np.min(one) - 1e-12
    assert np.min(two) >= 0.4 - 1e-12
    assert np.all(two > 0.0)


# =====================================================================
# 8. the spec, and the two published layouts
# =====================================================================

def test_the_tip_layout_is_carwings_beam_and_the_inboard_one_is_stiffer(solved):
    """PUBLISHED_LAYOUTS is the adoption path, so it is gated as one.

    Only ONE of the two entries is a beam carwing's closed form can express
    now: "tips" is its simply-supported pair and must land on its floats bit
    for bit. "inboard" is an INTERIOR support, which is the whole reason the
    published family kept a second layout at all, so what is asserted there is
    the outcome a user buys — the same load, the same reaction, a fraction of
    the movement, and a wing that moves BOTH WAYS about its mount, which
    neither of carwing's two closed forms can report.
    """
    y, load, b = solved["y"], solved["load"], solved["b"]
    tips = cm.PUBLISHED_LAYOUTS["tips"].beam(y, load, b, ei_nm2=4.0e4)
    supports = carwing.MOUNTS["tips"]["supports"]
    theirs = carwing.bending_moment(y, load, b, supports)
    assert np.array_equal(tips["moment_abs_Nm"], theirs)
    assert tips["deflection_index"] == carwing.deflection_index(
        y, theirs, b, supports)
    assert tips["deflection_m"] == tips["deflection_index"] / 4.0e4
    assert tips["sense"] == "sagging" and tips["w_tip_index"] == 0.0

    inboard = cm.PUBLISHED_LAYOUTS["inboard"].beam(y, load, b, ei_nm2=4.0e4)
    assert inboard["reaction_N"] == pytest.approx(tips["reaction_N"],
                                                  rel=1e-12)
    assert tips["deflection_index"] / inboard["deflection_index"] > 4.0
    assert inboard["M_max_Nm"] < 0.5 * tips["M_max_Nm"]
    # both ways about the mount: the overhang hogs the tip up while the middle
    # sags down. A layout read off the "ends" label instead of the station
    # would report the tip pinned at exactly zero, as "tips" does above.
    assert inboard["w_tip_index"] * inboard["w_centre_index"] < 0.0
    assert inboard["w_tip_index"] != 0.0


def test_the_solved_wing_takes_its_beam_from_this_module_at_both_stations():
    """MUTATION THIS TEST IS BUILT TO CATCH: carwing reading the "ends" LABEL
    both layouts now carry instead of the STATION they differ in — i.e. going
    back to its own two-end closed form for the inboard layout.

    Both registered layouts declare ``supports="ends"``, so the label no
    longer separates them and only the station does. The gate is the number a
    user reads off a solved run: its deflection, its bending sense and its
    moment array must be this module's, computed at ``station_frac * b/2``.
    A solver that fell back on the label would hand the inboard layout the
    tip-mounted answer — over four times the movement, and a wing pinned at
    its tips rather than 35% out.
    """
    from dataclasses import replace
    x = np.array([0.8, 0.0, -2.0, 8.0, 0.12, 0.30, B])
    got = {}
    for name in carwing.MOUNTS:
        prob = replace(CarWingProblem(), mount=name)
        out = evaluate_car_wing(x, prob)
        assert out["feasible"], out["reason"]
        rep = cm.published_layout(name).beam(out["y"], out["load_Npm"],
                                             out["b_m"], ei_nm2=prob.ei_nm2)
        assert out["deflection_m"] == rep["deflection_m"], name
        assert out["bending_sense"] == rep["sense"], name
        assert np.array_equal(out["moment_Nm"], rep["moment_abs_Nm"]), name
        got[name] = out
    # ...and the trade the second layout exists to offer, in both directions
    assert got["tips"]["deflection_m"] / got["inboard"]["deflection_m"] > 4.0
    assert got["inboard"]["CD"] > got["tips"]["CD"]
    assert got["tips"]["cd0_struts"] == 0.0
    assert got["tips"]["CD_junction"] == 0.0
    assert got["inboard"]["cd0_struts"] > 0.0      # two more sheets
    assert got["inboard"]["CD_junction"] > 0.0     # and the corners they make


def test_the_published_layouts_declare_the_published_stations():
    """The station is the only thing the two layouts differ in.

    carwing.MOUNTS and PUBLISHED_LAYOUTS have to name the same two layouts —
    a third name in one of them is a layout nobody can build through the
    other — and neither of them may carry a pylon: both take the load out
    through the plates, so ``n_struts`` is zero on both sides.
    """
    from dataclasses import replace
    assert (set(cm.PUBLISHED_LAYOUTS) == set(carwing.MOUNTS)
            == {"tips", "inboard"})
    for name, spec in cm.PUBLISHED_LAYOUTS.items():
        published = carwing.MOUNTS[name]
        assert not spec.carries_pylons, name
        assert spec.n_pylons == published["n_struts"] == 0, name
        assert published["supports"] == "ends", name
        assert spec.station_frac == published["station_frac"], name
    assert cm.PUBLISHED_LAYOUTS["tips"].station_frac == 1.0
    assert (cm.PUBLISHED_LAYOUTS["inboard"].station_frac
            == cm.INBOARD_STATION_FRAC < 1.0)
    # ...and the station is not the ONLY difference, which is the whole point
    # of the inboard layout not being free: moving its station to the tip
    # still leaves the two extra sheets it is charged for, so the specs must
    # NOT be equal. (They were, and that equality was the bug: the continuum
    # path priced the inboard grip as if it were the tip-borne one.)
    moved = replace(cm.PUBLISHED_LAYOUTS["inboard"], station_frac=1.0)
    assert moved != cm.PUBLISHED_LAYOUTS["tips"]
    assert moved.n_sheets == 2 and cm.PUBLISHED_LAYOUTS["tips"].n_sheets == 0
    assert (replace(moved, n_sheets=0) == cm.PUBLISHED_LAYOUTS["tips"])


def test_the_published_layouts_charge_nothing_for_the_mount():
    """Every default that CAN reproduce carwing does — and the pylon terms
    have nothing left to act on.

    The correction this module exists to make (a pylon reaches the DECK, not
    the track) used to be visible on the published centre mount. There is no
    published pylon any more, so the charge is exactly 0.0 at every ride
    height at both stations, including ride heights BELOW the deck, where the
    length is negative. The correction itself is asserted to be dormant rather
    than deleted, on a hand-built swan neck.
    """
    for name, spec in cm.PUBLISHED_LAYOUTS.items():
        assert spec.x_attach_frac == cm.X_AC_FRAC, name    # no windup
        assert spec.suction_loss == 0.0, name              # no loss
        assert spec.footprint_half_width_m() == 0.0, name  # no footprint
        for ride in (0.15, 0.30, 0.60):
            charged = spec.pylon_cd(ride, S_REF, RHO, V, MU)
            assert charged["CD"] == 0.0, (name, ride)
            assert charged["Swet_m2"] == 0.0, (name, ride)

    swan = cm.MountSpec(kind="pylon", station_frac=0.0)
    assert swan.pylon_drag_model == "flat_plate"       # carwing's own model
    ride = 0.30
    charged = swan.pylon_cd(ride, S_REF, RHO, V, MU)["CD"]
    assert charged == _strut_cd0(2, swan.pylon_length_m(ride), 0.12,
                                 S_REF, 0.005)
    assert charged != _strut_cd0(2, ride, 0.12, S_REF, 0.005)


def test_published_layout_hands_back_a_fresh_spec():
    """A caller that edits what it was given must not move the published one."""
    a = cm.published_layout("tips")
    a.x_attach_frac = 0.9
    assert cm.PUBLISHED_LAYOUTS["tips"].x_attach_frac == cm.X_AC_FRAC
    assert cm.published_layout("tips").x_attach_frac == cm.X_AC_FRAC
    for gone in ("nose", "centre", "ends"):
        with pytest.raises(ValueError):
            cm.published_layout(gone)


def test_the_station_is_stated_exactly_once():
    """A length OR a fraction, never both and never neither.

    Both exist because they are different statements — a chassis hardpoint is
    a length, 'at the tips' is a fraction that has to track a span which is
    itself a design variable — and a precedence rule would let one of them be
    silently ignored.
    """
    with pytest.raises(ValueError, match="exactly once"):
        cm.MountSpec(kind="pylon")
    with pytest.raises(ValueError, match="exactly once"):
        cm.MountSpec(kind="pylon", y_station_m=0.2, station_frac=0.5)
    by_len = cm.MountSpec(kind="pylon", y_station_m=0.4)
    by_frac = cm.MountSpec(kind="pylon", station_frac=0.5)
    assert by_len.station_m(B) == 0.4
    assert by_frac.station_m(B) == 0.4
    # ...and they part company the moment the span moves, which is the point
    assert by_len.station_m(3.2) == 0.4
    assert by_frac.station_m(3.2) == 0.8


@pytest.mark.parametrize("kwargs", [
    {"kind": "swan"},
    {"kind": "pylon", "side": "top"},
    {"kind": "pylon", "station_frac": 1.5},
    {"kind": "pylon", "y_station_m": -0.1},
    {"kind": "pylon", "x_attach_frac": 1.4},
    {"kind": "pylon", "n_pylons": 0},
    {"kind": "endplate", "n_pylons": 2},
    {"kind": "pylon", "pylon_tc": 0.0},
    {"kind": "pylon", "gj_nm2": 0.0},
    {"kind": "pylon", "deck_height_m": 0.0},
    {"kind": "pylon", "suction_loss": 1.0},
    {"kind": "pylon", "suction_loss": -0.1},
    {"kind": "pylon", "loss_spread": 0.5},
    {"kind": "pylon", "pylon_drag_model": "guess"},
])
def test_a_mount_that_cannot_exist_is_refused_at_construction(kwargs):
    """Configuration errors raise; per-candidate ones return a reason.

    The split is the package's: a spec is written by a person and its
    contradictions are programming errors, while a station outboard of a span
    the optimiser just shrank is an in-contract failure (``violation``).
    """
    kwargs = dict(kwargs)
    if "station_frac" not in kwargs and "y_station_m" not in kwargs:
        kwargs["station_frac"] = 0.0
    with pytest.raises(ValueError):
        cm.MountSpec(**kwargs)


def test_the_kind_and_the_station_are_independent_questions():
    """The generalisation, stated as a fact about the vocabulary.

    carwing.MOUNTS now answers only "where does it leave", and always through
    the plates: every registered layout is an endplate mount. The pylon KIND
    is still describable here — a swan neck can be priced, at any station —
    which is what says the family dropped a layout rather than this module
    dropping a concept.
    """
    assert set(cm.MOUNT_KINDS) == {"pylon", "endplate"}
    assert not any(cm.published_layout(n).carries_pylons
                   for n in cm.PUBLISHED_LAYOUTS)
    centre_pylon = cm.MountSpec(kind="pylon", station_frac=0.0)
    outboard_pylon = cm.MountSpec(kind="pylon", station_frac=0.35)
    assert centre_pylon.carries_pylons and outboard_pylon.carries_pylons
    assert outboard_pylon.n_junctions == 2
    assert cm.published_layout("tips").n_junctions == 0
    # the two the published family does NOT offer: a pylon at either station
    assert centre_pylon.station_m(B) == 0.0
    assert outboard_pylon.station_m(B) == pytest.approx(0.28, rel=1e-12)


def test_carmount_does_not_import_carwing():
    """The dependency runs one way: a family imports its mount, not the reverse.

    Checked by importing the module in a clean interpreter, because the whole
    test session has carwing loaded already and an in-process check would pass
    for the wrong reason.
    """
    import subprocess
    import sys
    code = ("import sys, aerobo.carmount; "
            "print('aerobo.carwing' in sys.modules); "
            "print('aerobo.endplate' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, check=True)
    assert out.stdout.split() == ["False", "False"], out.stdout
