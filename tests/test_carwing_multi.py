"""carwing_multi: the two-element car rear wing, against what it claims.

The gates here are in four groups, and the order is deliberate.

1. THE VECTOR. Four rows joined the family block, ahead of the size block,
   and the whole point of putting them there is that ``carwing``'s own
   readers — which count BACKWARDS from the chord coefficients — keep
   reading the same slots. That is tested on a vector whose every entry is
   different, because an index read from the wrong end passes any test built
   on a symmetric vector, and this repository has shipped that bug before.

2. THE SECTION PHYSICS, against things it cannot fake: an independent
   ``panel2d.solve`` for the two-basis superposition; the ISOLATED-SECTION
   LIMIT for the drag bridge (take the flap away and the MAIN ELEMENT must be
   charged the isolated section's own cd exactly — and the ASSEMBLY must not
   be, for two reasons this file measures rather than assumes); an
   independent re-derivation of the suction-peak ceiling written out here
   rather than by calling the module's own helper.

3. THE REFUSALS, each one shown to fire on the geometry it is about and to
   come back as a REASON rather than an exception — and each shown to be
   protecting something real: the geometry just under the slot floor has a
   discretisation error three times larger AND of the other sign, and the
   ones the intersection test catches swing the answer by several per cent
   non-monotonically.

4. THE CLAIM, in DIRECTION only. "Does splitting the wing buy downforce" has
   a measured answer with a sign that changes, and the sign and the crossing
   are what a test can hold; the percentages belong in the report.

A NOTE ON MUTATION. Several tests here carry a "MUTATION THIS TEST IS BUILT
TO CATCH" paragraph naming the exact edit that must turn them red. They are
there because the four they name all survived this file as it first shipped:
a gate replaced by ``if False:``, a load-bearing call deleted, an identity
that holds for any value of the quantity it was meant to check, and a
population test whose only threshold DELETING A REFUSAL CAN ONLY HELP. Each
was applied, run, confirmed red, and restored. When one of these tests is
changed, apply its named mutation again.
"""

import numpy as np
import pytest

from aerobo import airfoil, carwing, panel2d
from aerobo import carwing_multi as cm


CENTRE = None            # filled by _centre()


def _centre(prob=None):
    prob = prob or cm.CarWingMultiProblem()
    b = prob.bounds
    return 0.5 * (b[:, 0] + b[:, 1])


def _naca4_open_te(code: str, n_pts: int = 120) -> np.ndarray:
    """The TEXTBOOK NACA 4-digit: a section with a REAL trailing edge.

    Written out here rather than taken from ``airfoil.naca4_coords``, because
    that generator deliberately uses the CLOSED-TE variant of the thickness
    polynomial (last coefficient -0.1036 instead of the published -0.1015, as
    its own docstring says) and so cannot produce a blunt section. Everything
    else — the two-parabola camber line, thickness applied perpendicular to
    it, the package's TE -> upper -> LE -> lower loop order — is the same
    construction, so the ONLY difference from the module's own section is the
    trailing-edge base, which is what the test using this needs.
    """
    m, p, t = int(code[0]) / 100.0, int(code[1]) / 10.0, int(code[2:]) / 100.0
    x = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, n_pts // 2 + 1)))
    yt = 5.0 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2
                    + 0.2843 * x ** 3 - 0.1015 * x ** 4)   # OPEN trailing edge
    if m == 0.0 or p == 0.0:
        yc = dyc = np.zeros_like(x)
    else:
        fwd = x < p
        yc = np.where(fwd, m / p ** 2 * (2.0 * p * x - x ** 2),
                      m / (1.0 - p) ** 2 * ((1.0 - 2.0 * p) + 2.0 * p * x
                                            - x ** 2))
        dyc = np.where(fwd, 2.0 * m / p ** 2 * (p - x),
                       2.0 * m / (1.0 - p) ** 2 * (p - x))
    th = np.arctan(dyc)
    xu, yu = x - yt * np.sin(th), yc + yt * np.cos(th)
    xl, yl = x + yt * np.sin(th), yc - yt * np.cos(th)
    return np.column_stack([np.concatenate([xu[::-1], xl[1:]]),
                            np.concatenate([yu[::-1], yl[1:]])])


# ------------------------------------------------------ 1. the design vector


def test_the_size_block_is_still_read_from_the_end():
    """carwing's own readers must be untouched by the four new rows.

    The vector is built so that EVERY entry is a different number and none of
    them is a plausible neighbour of any other: an off-by-one that reads the
    span out of the overlap row, or the area out of the span row, has to show
    up as a wrong VALUE, not as a wrong-looking one.
    """
    prob = cm.CarWingMultiProblem(area_bounds_m2=(0.10, 0.48),
                                  objective="downforce", chord_order=3)
    labels = prob.param_labels
    assert labels == (
        "taper", "twist_root_deg", "twist_tip_deg", "alpha_deg",
        "endplate_h_m", "ride_height_m", "flap_chord_frac",
        "flap_deflection_deg", "slot_gap_frac", "slot_overlap_frac",
        "S_m2", "b_m", "chord_k1", "chord_k2", "chord_k3")
    x = np.array([0.61, 1.1, -1.2, 5.3, 0.14, 0.42,      # family
                  0.31, 21.7, 0.037, 0.041,              # slot
                  0.271, 1.83,                           # size: S then b
                  0.011, 0.012, 0.013], dtype=float)     # chord
    assert x.size == prob.dim

    # carwing's readers, counting back from the chord coefficients
    assert carwing.span_from_x(x, prob.chord_order) == 1.83
    assert carwing.s_from_x(x, prob.chord_order) == 0.271
    # this module's reader, counting forward from a fixed offset
    assert cm.slot_from_x(x) == (0.31, 21.7, 0.037, 0.041)

    # and with the area FIXED the span row moves back one slot, while the
    # slot rows do not move at all — which is the reason they sit where they
    # sit rather than next to the span
    prob0 = cm.CarWingMultiProblem(chord_order=3)
    x0 = np.delete(x, 10)
    assert carwing.span_from_x(x0, 3) == 1.83
    assert cm.slot_from_x(x0) == (0.31, 21.7, 0.037, 0.041)
    assert prob0.dim == x0.size


def test_the_box_rows_are_the_bands_the_module_declares():
    prob = cm.CarWingMultiProblem()
    b = prob.bounds
    assert b.shape == (11, 2)
    assert tuple(b[cm.SLOT_ROW_OFFSET]) == cm.FLAP_CHORD_FRAC_BOUNDS
    assert tuple(b[cm.SLOT_ROW_OFFSET + 1]) == cm.FLAP_DEFLECTION_BOUNDS_DEG
    assert tuple(b[cm.SLOT_ROW_OFFSET + 2]) == cm.SLOT_GAP_FRAC_BOUNDS
    assert tuple(b[cm.SLOT_ROW_OFFSET + 3]) == cm.SLOT_OVERLAP_FRAC_BOUNDS
    # the size block is carwing's, unchanged
    assert tuple(b[-1]) == carwing.SPAN_BOUNDS_M
    np.testing.assert_array_equal(b[:6], carwing.CarWingProblem().bounds[:6])


def test_slot_from_x_refuses_a_vector_with_no_room():
    with pytest.raises(ValueError, match="no room for four slot rows"):
        cm.slot_from_x(np.zeros(8))


# --------------------------------------------------- 2. the section physics


def _bodies(ff=0.30, defl=20.0, gap=0.025, ovl=0.02, n=120, tc=0.12):
    (xm, ym), (xf, yf) = cm.cascade_coords(tc, ff, defl, gap, ovl, n)
    return (panel2d.Body(xm, ym, name="main"),
            panel2d.Body(xf, yf, name="flap"))


def test_two_basis_solves_reproduce_a_direct_solve():
    """The exactness the whole cost model rests on.

    ``panel2d.solve`` is run independently at five incidences and compared
    with the reconstruction from the two basis solves. This is not a
    tolerance to be relaxed: the reconstruction is algebraically exact and
    anything above round-off means the superposition is wrong.
    """
    bodies = _bodies()
    s0, s90 = cm._basis(bodies)
    alphas = np.array([-4.0, 0.0, 4.0, 8.0, 12.0])
    dm, df = cm._sweep(s0, s90, alphas)
    for k, a in enumerate(alphas):
        ref = panel2d.solve(bodies, np.deg2rad(a), V=1.0, c_ref=1.0,
                            ref_point=(0.25, 0.0))
        assert abs(dm["cl"][k] + df["cl"][k] - ref.cl) < 1e-13
        assert abs(dm["cm"][k] + df["cm"][k] - ref.cm) < 1e-13
        assert abs(dm["cl"][k] - ref.body("main").cl) < 1e-13
        assert abs(df["cl"][k] - ref.body("flap").cl) < 1e-13
        assert abs(dm["cp_min"][k] - ref.body("main").cp_min) < 1e-12
        assert abs(df["cp_min"][k] - ref.body("flap").cp_min) < 1e-12


def test_the_elements_coefficients_add_to_the_assemblys():
    """The reference-chord convention, stated as an identity.

    Every element coefficient is on the STOWED chord, so the two add to the
    section's exactly; the OWN-chord values do not, and the share key is what
    reconciles them.
    """
    pol, why = cm.cascade_polar(0.12, 0.30, 20.0, 0.025, 0.02, 9.5e5)
    assert why is None
    np.testing.assert_allclose(pol.cl_main + pol.cl_flap, pol.CL, rtol=0,
                               atol=1e-13)
    np.testing.assert_allclose(pol.cl_own_main * pol.c_main, pol.cl_main,
                               rtol=0, atol=1e-13)
    np.testing.assert_allclose(pol.cl_own_flap * pol.c_flap, pol.cl_flap,
                               rtol=0, atol=1e-13)
    assert abs(pol.c_main + pol.c_flap - 1.0) < 1e-15
    # the share reconciles them wherever the section is making lift
    live = np.abs(pol.CL) > 0.1
    np.testing.assert_allclose(pol.share_main[live] * pol.CL[live],
                               pol.cl_main[live], rtol=0, atol=1e-13)


def test_a_lonely_element_is_priced_exactly_like_the_single_element_family():
    """The drag bridge's exactness in the isolated limit — its strongest claim.

    Push the flap 500 chords away and the main element is an isolated
    section. The suction-peak map must then be the IDENTITY (its equivalent
    incidence is its own incidence) and the drag the MAIN ELEMENT is charged
    must be the single-element table's, at the element's own Reynolds number
    and on its own chord. If the map is not the identity here, it is not a
    map from a section to itself and nothing else it says can be trusted.
    """
    ff, re_ref = 0.30, 1.0e6
    pol, why = cm.cascade_polar(0.12, ff, 0.0, 500.0, 0.0, re_ref)
    assert why is None
    iso, re_used, clamped = cm._wide_polar(0.12, re_ref * (1.0 - ff))
    assert not clamped
    for a in (-4.0, 0.0, 4.0, 8.0, 12.0):
        i = int(np.argmin(np.abs(pol.alpha_deg - a)))
        assert abs(pol.alpha_eq_main[i] - a) < 2e-3, (
            f"the suction-peak map is not the identity on an isolated "
            f"element at alpha {a}")
        got = pol.cd_main[i] / pol.c_main
        assert abs(got - float(iso.cd(a))) < 1e-5

    # ...and it is NOT the identity once the flap is really there, which is
    # the other half: a test that passes both ways is not a test.
    near, why = cm.cascade_polar(0.12, ff, 20.0, 0.025, 0.02, re_ref)
    assert why is None
    j = int(np.argmin(np.abs(near.alpha_deg - 4.0)))
    assert abs(near.alpha_eq_main[j] - 4.0) > 1.0


def test_the_lonely_assembly_is_NOT_carwings_cd_and_the_gap_is_the_two_named():
    """The other half of the isolated limit, because it was once claimed wrong.

    The module docstring used to say that in this limit "this family's cd is
    bit-for-bit ``carwing.py``'s cd". It is not, and it cannot be. This test
    holds the corrected claim so it cannot rot back: the ASSEMBLY's cd is
    13-28 % above ``carwing``'s at every incidence, and the gap decomposes
    EXACTLY into the two reasons the docstring names — the flap is still
    charged, and the main element is a shorter chord at a lower Reynolds
    number. Both halves are re-derived here from ``carwing``'s own table
    rather than read off anything this module stores.
    """
    from aerobo.polar import default_polar_family

    tc, ff, re_ref = 0.12, 0.30, 1.0e6
    pol, why = cm.cascade_polar(tc, ff, 0.0, 500.0, 0.0, re_ref)
    assert why is None
    carwings = default_polar_family().at(tc)          # what carwing.py reads

    errs = []
    for a in (-4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0):
        i = int(np.argmin(np.abs(pol.alpha_deg - a)))
        cw = float(carwings.cd(a))
        errs.append((float(pol.CD[i]) - cw) / cw)
        # the two elements' charges are the whole of the assembly's
        assert pol.CD[i] == pytest.approx(pol.cd_main[i] + pol.cd_flap[i],
                                          rel=0, abs=1e-15)
    assert min(errs) > 0.10 and max(errs) < 0.35, (
        f"the docstring says 13-28 % high at every incidence; got "
        f"{min(errs):+.1%} to {max(errs):+.1%}")

    # the decomposition, at alpha 4, against the docstring's two bullets
    i = int(np.argmin(np.abs(pol.alpha_deg - 4.0)))
    cw = float(carwings.cd(4.0))
    assert pol.cd_flap[i] / cw == pytest.approx(0.428, abs=0.01)
    assert pol.cd_main[i] / cw - 1.0 == pytest.approx(-0.222, abs=0.01)

    # and the reason that is NOT one: the two banks agree bit-for-bit in cd
    # at a matched Reynolds number, so the bank contributes nothing here
    wide_1e6, _re, _clip = cm._wide_polar(tc, 1.0e6)
    for a in np.arange(-6.0, 14.1, 2.0):
        assert float(wide_1e6.cd(a)) == float(carwings.cd(a))


def test_the_suction_peak_ceiling_is_the_measured_stall_not_a_constant():
    """Re-derived here from the shipped tables, without the module's helper.

    If the ceiling were ever replaced by a pasted number this fails, because
    the derivation is written out independently: take the wide-alpha family,
    find the RESOLVED stall incidence, solve the isolated section there.
    """
    from aerobo.polar import stall_point, stall_polar_family

    fam = stall_polar_family()
    for tc in (0.09, 0.12, 0.15, 0.18):
        sp = stall_point(fam.members[tc], tc)
        assert not sp.censored
        c = airfoil.naca4_coords(f"24{int(round(tc * 100)):02d}",
                                 cm.N_SECTION_NODES)
        x, y = panel2d.close_trailing_edge(c[:, 0], c[:, 1])
        ref = panel2d.solve([panel2d.Body(x, y)],
                            np.deg2rad(sp.alpha_stall_deg), c_ref=1.0,
                            ref_point=(0.25, 0.0))
        got, a_got = cm.suction_peak_ceiling(tc)
        assert a_got == sp.alpha_stall_deg
        assert abs(got - ref.cp_min) < 1e-3 * abs(ref.cp_min)

    # it MOVES with thickness, and in the direction a thicker nose implies:
    # a blunter leading edge reaches its stall with a shallower peak
    ceils = [cm.suction_peak_ceiling(t)[0] for t in (0.09, 0.12, 0.15, 0.18)]
    assert ceils[0] < ceils[1] < ceils[2] < ceils[3] < 0.0


def test_a_thickness_with_no_resolved_stall_gets_no_ceiling():
    """t/c 0.06 is CENSORED in the shipped family: a lower bound, not a peak."""
    with pytest.raises(ValueError, match="outside the RESOLVED stall range"):
        cm.suction_peak_ceiling(0.06)
    # and the family turns that into a REASON, not an exception
    prob = cm.CarWingMultiProblem(tc=0.06)
    out = cm.evaluate_car_wing_multi(_centre(prob), prob)
    assert out["feasible"] is False
    assert "RESOLVED stall range" in out["reason"]


def test_the_ceiling_is_rebuilt_at_the_panel_count_it_will_be_judged_on():
    """A criterion measured on one grid and applied on another is not one."""
    a, b = cm.suction_peak_ceiling(0.12, 120), cm.suction_peak_ceiling(0.12, 320)
    assert a[1] == b[1]                       # same stall incidence
    assert a[0] != b[0]                       # different grid, different peak
    assert abs(a[0] - b[0]) < 0.02 * abs(b[0])   # ...but only just


def test_the_panel_count_table_is_the_one_the_module_quotes():
    """The 120-node default's convergence numbers, re-derived not trusted.

    Three numbers are quoted for the default, in two places (the module
    docstring's panel-count table and :data:`carwing_multi.N_SECTION_NODES`),
    all against 240 nodes at alpha 4 on the box centre's own slot: 0.25 % in
    the assembly cl, 1.25 % SHALLOW in the main element's Cp_min, and 0.81 %
    shallow in the suction-peak ceiling — and the 0.48 % OPTIMISM in the
    stall margin the last two make together.

    They disagreed until this test existed: ``N_SECTION_NODES`` claimed
    "0.0 % in the main element's Cp_min against 240 nodes" while the module
    docstring measured 1.25 % on the same case and built a residual-optimism
    argument on it, and the argument itself compared a peak against 240 nodes
    with a ceiling against 160. Two references make a ratio that is not one.
    """
    tc, ff, defl, gap, ovl = 0.12, 0.275, 17.5, 0.031, 0.02   # the box centre
    assert (ff, defl, gap, ovl) == tuple(
        round(float(v), 12) for v in cm.slot_from_x(_centre())), (
        "this test is stated at the box centre's own slot")

    cl, cpm, ceil = {}, {}, {}
    for n in (100, 120, 160, 240):
        (xm, ym), (xf, yf) = cm.cascade_coords(tc, ff, defl, gap, ovl, n)
        s0, s90 = cm._basis([panel2d.Body(xm, ym), panel2d.Body(xf, yf)])
        dm, df = cm._sweep(s0, s90, np.array([4.0]))
        cl[n] = float(dm["cl"][0] + df["cl"][0])
        cpm[n] = float(dm["cp_min"][0])
        ceil[n] = float(cm.suction_peak_ceiling(tc, n)[0])

    # the table the docstring prints
    for n, want in ((100, 1.86546), (120, 1.86822), (160, 1.87158),
                    (240, 1.87282)):
        assert cl[n] == pytest.approx(want, abs=5e-5)
    for n, want in ((100, -7.215), (120, -7.172), (160, -7.264),
                    (240, -7.263)):
        assert cpm[n] == pytest.approx(want, abs=5e-3)
    for n, want in ((100, -13.647), (120, -13.881), (160, -13.926),
                    (240, -13.994)):
        assert ceil[n] == pytest.approx(want, abs=5e-3)

    def pct(n):
        return 100.0 * (n[120] - n[240]) / abs(n[240])

    assert pct(cl) == pytest.approx(-0.25, abs=0.02)
    assert pct(cpm) == pytest.approx(+1.25, abs=0.02)      # SHALLOW: |Cp| low
    assert pct(ceil) == pytest.approx(+0.81, abs=0.02)     # shallow the same way

    # ...and the margin the last two make is the more converged of the three,
    # in the OPTIMISTIC direction
    m120 = 1.0 - cpm[120] / ceil[120]
    m240 = 1.0 - cpm[240] / ceil[240]
    assert m120 == pytest.approx(0.4833, abs=5e-4)
    assert m240 == pytest.approx(0.4810, abs=5e-4)
    assert 100.0 * (m120 - m240) / m240 == pytest.approx(0.48, abs=0.02)
    assert m120 > m240, "the residual is supposed to be optimistic"


def test_the_stall_margin_is_zero_exactly_at_the_ceiling():
    pol, why = cm.cascade_polar(0.12, 0.30, 20.0, 0.025, 0.02, 9.5e5)
    assert why is None
    lo, hi = pol.alpha_valid
    i_hi = int(np.argmin(np.abs(pol.alpha_deg - hi)))
    m_main, m_flap = pol.stall_margin(pol.alpha_deg[i_hi])
    assert min(float(m_main), float(m_flap)) >= 0.0
    # one step beyond the window, an element is past its ceiling
    j = i_hi + 1
    if j < pol.alpha_deg.size:
        m2 = pol.stall_margin(pol.alpha_deg[j])
        assert min(float(m2[0]), float(m2[1])) < min(float(m_main),
                                                     float(m_flap))


def test_no_flyable_incidence_reads_a_clipped_drag():
    """The drag tables are clipped only where the window already refuses."""
    pol, why = cm.cascade_polar(0.12, 0.30, 20.0, 0.025, 0.02, 9.5e5)
    assert why is None
    lo, hi = pol.alpha_valid
    inside = (pol.alpha_deg >= lo) & (pol.alpha_deg <= hi)
    p_main, _, _ = cm._wide_polar(pol.tc, pol.re_main)
    p_flap, _, _ = cm._wide_polar(pol.tc, pol.re_flap)
    for aeq, p in ((pol.alpha_eq_main[inside], p_main),
                   (pol.alpha_eq_flap[inside], p_flap)):
        a, b = p.alpha_valid
        assert aeq.min() >= a - 1e-9 and aeq.max() <= b + 1e-9


def test_the_flap_deflection_moves_the_section_the_way_a_flap_must():
    """Sign gate. More flap = more camber = more lift and a lower alpha_L0."""
    prev_cl0, prev_aL0 = None, None
    for defl in (0.0, 10.0, 20.0, 30.0):
        pol, why = cm.cascade_polar(0.12, 0.30, defl, 0.025, 0.02, 9.5e5)
        assert why is None, why
        cl0 = float(pol.cl(0.0))
        aL0 = float(np.rad2deg(pol.alpha_L0))
        if prev_cl0 is not None:
            assert cl0 > prev_cl0, "deflecting the flap must ADD lift"
            assert aL0 < prev_aL0, "deflecting the flap must ADD camber"
        prev_cl0, prev_aL0 = cl0, aL0


# --------------------------------------------------------- 3. the refusals


def test_an_intersecting_flap_is_a_reason_not_an_exception():
    """panel2d checks a body against itself; nothing checks it against another.

    The geometries this catches are the ones that produce the wild numbers:
    at the three placement gaps just below the crossing threshold the
    160-node assembly cl runs -4.7 %, -1.9 %, +5.0 % against the 960-node
    answer, non-monotone, which is exactly what an optimiser reports as a
    discovery.
    """
    pol, why = cm.cascade_polar(0.12, 0.30, 25.0, 0.0045, 0.02, 9.5e5,
                                n_nodes=160)
    assert pol is None
    assert "intersects the main element" in why
    # ...and it is not refusing everything: the same slot one thousandth of a
    # chord wider is a solvable cascade
    ok, why2 = cm.cascade_polar(0.12, 0.30, 25.0, 0.0080, 0.02, 9.5e5,
                                n_nodes=160)
    assert why2 is None and ok is not None


def test_an_unresolvable_slot_is_refused_and_the_floor_is_protecting_something():
    """The floor is not taste: below it the error stops shrinking.

    The refusal is shown to fire, and then the geometry it refuses is solved
    anyway, at 160 nodes against 960, beside a geometry the floor accepts.
    The refused one's discretisation error must be materially larger — if it
    is not, the floor is refusing designs for nothing.
    """
    ff, defl, ovl = 0.30, 25.0, 0.02
    bad_gap, good_gap = 0.0058, 0.0080         # neither one intersects

    pol, why = cm.cascade_polar(0.12, ff, defl, bad_gap, ovl, 9.5e5,
                                n_nodes=160)
    assert pol is None
    assert "resolution floor" in why and "main-element" in why
    ok, why2 = cm.cascade_polar(0.12, ff, defl, good_gap, ovl, 9.5e5,
                                n_nodes=160)
    assert why2 is None, why2

    def err(gap):
        cl = []
        for n in (160, 960):
            (xm, ym), (xf, yf) = cm.cascade_coords(0.12, ff, defl, gap, ovl, n)
            m, f = panel2d.Body(xm, ym), panel2d.Body(xf, yf)
            assert not cm._loops_cross(m.nodes, f.nodes)
            cl.append(panel2d.solve([m, f], np.deg2rad(4.0), c_ref=1.0,
                                    ref_point=(0.25, 0.0)).cl)
        return (cl[0] - cl[1]) / cl[1]

    e_bad, e_good = err(bad_gap), err(good_gap)
    assert abs(e_bad) > 2.5 * abs(e_good), (
        f"the floor is supposed to separate a converged slot from an "
        f"unconverged one; got {e_bad:+.4%} refused against {e_good:+.4%} "
        f"accepted")
    assert np.sign(e_bad) != np.sign(e_good), (
        "below the floor the error is not the same error being refined away")


def test_the_wing_must_fly_inside_the_sections_own_window():
    """The one place the stall criterion reaches the three-dimensional wing.

    MUTATION THIS TEST IS BUILT TO CATCH: replacing the window gate in
    ``evaluate_car_wing_multi`` —

        if a_wing.min() < lo or a_wing.max() > hi:      ->      if False:

    Everything else about the suction-peak criterion lives in the SECTION:
    ``cascade_polar`` uses it to build ``alpha_valid``, and that is where the
    testable arithmetic is. This gate is the only thing that carries it onto
    the lattice, and it produces the family's largest refusal class (53 of
    400 box draws at seed 0, and the largest class at every seed tried).
    Deleted, the family hands an optimiser stalled designs marked feasible.

    Three assertions, and the last two are what make the first mean
    something. The gate FIRES; the design it refuses is genuinely past the
    measured ceiling (the main element's stall margin where the wing really
    flies is NEGATIVE, so this is not a band with margin to spare); and the
    same design two degrees lower is FEASIBLE, so the gate is not refusing
    the neighbourhood.
    """
    prob = cm.CarWingMultiProblem()
    x = _centre(prob)
    x[cm.SLOT_ROW_OFFSET + 1] = 30.0            # flap deflection, in band

    ok = x.copy()
    ok[3] = 6.0
    good = cm.evaluate_car_wing_multi(ok, prob)
    assert good["feasible"], good.get("reason")
    assert good["stall_margin_main"] > 0.0

    bad = x.copy()
    bad[3] = 8.0
    out = cm.evaluate_car_wing_multi(bad, prob)
    assert out["feasible"] is False
    assert "outside the section's flyable window" in out["reason"]

    # the wing really does leave the window, at the top end
    lo, hi = out["alpha_valid"]
    flown_lo, flown_hi = out["alpha_eff_range"]
    assert flown_hi > hi and flown_lo >= lo

    # ...and out there an element is past its measured ceiling: the refusal
    # reports the (main, flap) margins at BOTH ends of what the wing flies,
    # and the main element's at the top end is negative
    assert out["stall_margin_main"][1] < 0.0, (
        "the window gate is supposed to be the stall criterion reaching the "
        "wing; if the design it refuses still has margin, it is a band")

    # the section agrees when asked directly, without going through the gate
    pol = good["section"]
    assert float(pol.stall_margin(flown_hi)[0]) < 0.0


def test_a_blunt_section_is_closed_before_it_is_flown():
    """MUTATION: dropping ``panel2d.close_trailing_edge`` from ``_base_coords``.

    That call is load-bearing and looks decorative, because every section the
    rest of this suite feeds is already closed: ``airfoil.naca4_coords`` uses
    the CLOSED-TE thickness polynomial (-0.1036) and the closure returns it
    unchanged bit-for-bit, which is what
    ``test_a_closed_naca_is_returned_unchanged_by_the_closure`` says.

    A section from almost any other source is not closed. The TEXTBOOK NACA
    4-digit — the open-TE polynomial with -0.1015, which is what published
    NACA tables and most coordinate files carry, and which
    ``airfoil.naca4_coords`` names in its own docstring as the variant it
    does not use — lands its two trailing nodes 0.00252 chords apart, 2.5x
    ``panel2d.TE_GAP_MAX_FRAC``. So it is fed here, raw, and the family has
    to fly it.
    """
    c = _naca4_open_te("2412", cm.N_SECTION_NODES)
    chord = float(np.ptp(c[:, 0]))
    base = float(np.hypot(c[0, 0] - c[-1, 0], c[0, 1] - c[-1, 1])) / chord
    assert base > panel2d.TE_GAP_MAX_FRAC, "this section is not actually blunt"
    assert base == pytest.approx(0.00252, abs=5e-5)

    # panel2d refuses it as it stands — which is the whole reason the closure
    # is in _base_coords and not left to the caller
    why_raw = panel2d.check_body(c[:, 0], c[:, 1])
    assert why_raw is not None and "blunt" in why_raw

    # _base_coords hands back a CLOSED loop, and it is not the input
    xb, yb = cm._base_coords(0.12, cm.N_SECTION_NODES, c)
    assert panel2d.check_body(xb, yb) is None
    assert np.hypot(xb[0] - xb[-1], yb[0] - yb[-1]) < 1e-9 * chord
    assert not np.array_equal(yb, c[:, 1]), (
        "a blunt section that comes back unchanged has not been closed")

    # ...so the section builds, and a whole design flies on it
    pol, why = cm.cascade_polar(0.12, 0.30, 18.0, 0.026, 0.021, 9.5e5,
                                coords=c)
    assert why is None, why
    prob = cm.CarWingMultiProblem(section_coords=c)
    out = cm.evaluate_car_wing_multi(_centre(prob), prob)
    assert out["feasible"], out.get("reason")


def test_a_slot_that_stalls_at_every_incidence_is_refused_with_that_reason():
    pol, why = cm.cascade_polar(0.12, 0.40, 60.0, 0.030, 0.02, 9.5e5)
    assert pol is None
    assert "no incidence" in why and "suction-peak ceiling" in why


def test_the_measured_slot_width_is_not_the_placement_gap():
    """Rotating the flap about its leading edge closes the slot it was given.

    Reported rather than assumed, because the refusals are stated against the
    width the flow sees and a reader who thinks the gap row IS the slot will
    misread every one of them.
    """
    widths = []
    for defl in (0.0, 15.0, 30.0):
        pol, why = cm.cascade_polar(0.12, 0.30, defl, 0.030, 0.02, 9.5e5)
        assert why is None
        widths.append(pol.slot_width_frac)
        # the flap's leading edge is ROUND: the shortest distance from the
        # main element's trailing edge to it is always less than the
        # centre-to-centre placement the gap row states
        assert pol.slot_width_frac < 0.030
    # and deflecting swings the flap's upper surface away from the main
    # element's trailing edge, so the slot OPENS with deflection — the
    # opposite of what "more flap = tighter slot" would suggest, which is
    # why the width is measured rather than assumed
    assert widths[0] < widths[1] < widths[2]


#: every refusal the default box actually produces, with a design INSIDE the
#: box chosen to trigger it. Each entry is
#: ``(name, {row: value overriding the box centre}, reason fragment)``.
#: Rows are named rather than indexed so that a vector-layout change shows up
#: as a failure here too.
_REFUSAL_TRIGGERS = [
    ("wing AoA leaves the section's window",
     {"alpha_deg": 8.0, "flap_deflection_deg": 30.0},
     "outside the section's flyable window"),
    ("the slot stalls at every incidence",
     {"flap_chord_frac": 0.40, "flap_deflection_deg": 35.0,
      "slot_gap_frac": 0.012, "slot_overlap_frac": 0.06},
     "every incidence puts an element past its ceiling"),
    ("the flap intersects the main element",
     {"flap_chord_frac": 0.40, "flap_deflection_deg": 0.0,
      "slot_gap_frac": 0.012, "slot_overlap_frac": 0.06},
     "intersects the main element"),
    ("the slot is below the resolution floor",
     {"flap_chord_frac": 0.30, "flap_deflection_deg": 0.0,
      "slot_gap_frac": 0.019, "slot_overlap_frac": 0.04},
     "resolution floor"),
    ("the endplate reaches the track",
     {"endplate_h_m": 0.25, "ride_height_m": 0.15},
     "ground plane"),
    ("the endplate's own incidence leaves its own polar",
     {"taper": 0.4, "alpha_deg": 12.0, "endplate_h_m": 0.25,
      "ride_height_m": 0.60},
     "endplate effective AoA"),
    ("the flyable window is too narrow to state a slope on",
     {"flap_chord_frac": 0.35, "flap_deflection_deg": 32.0,
      "slot_gap_frac": 0.03, "slot_overlap_frac": 0.04},
     "too narrow to state a lift-curve slope on"),
]


def _with(prob, **rows):
    """The box centre with named rows overridden, checked to stay in the box."""
    x = _centre(prob)
    labels = prob.param_labels
    for name, value in rows.items():
        x[labels.index(name)] = value
    b = prob.bounds
    assert np.all(x >= b[:, 0] - 1e-12) and np.all(x <= b[:, 1] + 1e-12), (
        "a refusal trigger must be INSIDE the box, or it is testing the "
        "bounds check and not the physics")
    return x


@pytest.mark.parametrize("name, rows, fragment", _REFUSAL_TRIGGERS,
                         ids=[t[0] for t in _REFUSAL_TRIGGERS])
def test_each_refusal_fires_on_geometry_chosen_to_trigger_it(name, rows,
                                                             fragment):
    """MUTATION: deleting any one refusal from ``evaluate_car_wing_multi``.

    The population test below used to be the only gate on the refusals, and
    its only threshold was ``feasible >= 50 %`` — which DELETING a refusal
    can only help. So every refusal the default box actually produces is
    named here with a design inside the box that fires it, and the reason
    that comes back is checked to be that refusal's own and not another's.

    The list IS the module docstring's census, at its stated seed: over 400
    draws from ``default_rng(0)``, 271 are feasible and the other 129 fall
    into exactly these seven classes. One of the seven — the endplate
    reaching the track — is raised inside ``vlm.py`` rather than here, and is
    listed because this family has to return it as a REASON; the rest are
    this module's own.
    """
    prob = cm.CarWingMultiProblem()
    out = cm.evaluate_car_wing_multi(_with(prob, **rows), prob)
    assert out["feasible"] is False, f"{name}: this design was not refused"
    assert fragment in out["reason"], (
        f"{name}: expected {fragment!r}, got {out['reason']!r}")
    assert out["score"] == carwing.PENALTY


def test_every_refusal_in_the_family_comes_back_as_a_reason():
    """No in-contract failure may raise (the ``objective._fail`` contract).

    The population half of the refusal gates. It carries one assertion that
    a deleted refusal cannot satisfy: every design this family calls
    FEASIBLE must report non-negative suction-peak margins on both elements.
    A feasible design with a stalled element is what deleting the window gate
    in ``evaluate_car_wing_multi`` produces, and the 50 % threshold below
    would welcome it.
    """
    prob = cm.CarWingMultiProblem()
    b = prob.bounds
    rng = np.random.default_rng(11)
    X = b[:, 0] + rng.random((120, b.shape[0])) * (b[:, 1] - b[:, 0])
    feasible = 0
    for x in X:
        out = cm.evaluate_car_wing_multi(x, prob)     # must never raise
        assert set(("feasible", "reason", "score")) <= set(out)
        if out["feasible"]:
            feasible += 1
            assert len(out["g"]) == prob.n_constraints
            assert out["stall_margin_main"] >= 0.0, (
                "a design this family calls feasible has its main element "
                "past the measured suction-peak ceiling")
            assert out["stall_margin_flap"] >= 0.0
        else:
            assert out["score"] == carwing.PENALTY
            assert out["reason"]
    # the box is not mostly dead: a family whose box refuses nearly
    # everything is a box problem, not a physics result
    assert feasible >= 0.5 * X.shape[0], (
        f"only {feasible}/{X.shape[0]} of the default box is feasible")


def test_out_of_box_and_wrong_shape_are_reasons():
    prob = cm.CarWingMultiProblem()
    x = _centre(prob)
    assert cm.evaluate_car_wing_multi(x[:-1], prob)["reason"].startswith(
        "design vector shape")
    bad = x.copy()
    bad[cm.SLOT_ROW_OFFSET] = 0.99
    assert cm.evaluate_car_wing_multi(bad, prob)["reason"] == "bounds violation"


# ------------------------------------------------ 4. the family's contracts


def test_the_failure_vector_is_as_wide_as_the_problem_declares():
    for kw, n in ((dict(), 2),
                  (dict(CD_budget=None), 1),
                  (dict(drag_budget_n=90.0), 3),
                  (dict(CD_budget=None, drag_budget_n=90.0,
                        downforce_min_n=300.0), 3)):
        prob = cm.CarWingMultiProblem(**kw)
        assert prob.n_constraints == n == len(prob.constraint_labels)
        x = _centre(prob)
        bad = x.copy()
        bad[cm.SLOT_ROW_OFFSET + 1] = cm.FLAP_DEFLECTION_BOUNDS_DEG[1]
        bad[cm.SLOT_ROW_OFFSET + 2] = cm.SLOT_GAP_FRAC_BOUNDS[0]
        score, g = cm.fg_car_wing_multi(bad, prob)
        assert len(g) == n
        score, g = cm.fg_car_wing_multi(x, prob)
        assert len(g) == n


def test_the_constraint_labels_track_the_budgets_that_are_on():
    assert cm.CarWingMultiProblem().constraint_labels == (
        "drag budget margin", "deflection margin")
    assert cm.CarWingMultiProblem(
        CD_budget=None, drag_budget_n=81.5,
        downforce_min_n=300.0).constraint_labels == (
        "drag force margin", "deflection margin", "downforce floor margin")
    # ...and they are the SAME labels the single-element family declares:
    # the slot adds refusals, never a margin
    for kw in (dict(), dict(CD_budget=None, drag_budget_n=81.5)):
        assert (cm.CarWingMultiProblem(**kw).constraint_labels
                == carwing.CarWingProblem(**kw).constraint_labels)


def test_a_free_area_refuses_the_coefficient_objective():
    with pytest.raises(ValueError, match="meaningless with a free reference"):
        cm.CarWingMultiProblem(area_bounds_m2=(0.1, 0.48))
    prob = cm.CarWingMultiProblem(area_bounds_m2=(0.1, 0.48),
                                  objective="downforce")
    assert prob.dim == 12
    out = cm.evaluate_car_wing_multi(_centre(prob), prob)
    assert out["feasible"], out.get("reason")
    assert out["score"] == out["downforce_N"]
    assert out["area_free"] is True


def test_a_bad_switch_is_a_call_site_error_not_a_reason():
    for kw, msg in ((dict(mount="wheelbarrow"), "unknown mount"),
                    (dict(objective="loudness"), "unknown objective"),
                    (dict(CD_budget=-1.0), "CD_budget must be"),
                    (dict(drag_budget_n=0.0), "drag_budget_n must be"),
                    (dict(n_section_nodes=8), "too coarse")):
        with pytest.raises(ValueError, match=msg):
            cm.CarWingMultiProblem(**kw)


# ------------------------------------------- 5. the endplate is not the wing


def test_the_endplate_does_not_fly_the_cascade():
    """A plate is a plate — and giving it the flap's camber refuses outright.

    Not a preference. The cascade's zero-lift angle at the box centre is far
    negative; hand it to a flat vertical sheet and the sheet is being told it
    has the camber of a deployed flap. The measured consequence is that the
    box centre stops being flyable at all, which is what this asserts, so a
    revert to "one polar for both surfaces" cannot pass quietly.
    """
    prob = cm.CarWingMultiProblem()
    x = _centre(prob)
    ok = cm.evaluate_car_wing_multi(x, prob)
    assert ok["feasible"], ok.get("reason")

    ff, defl, gap, ovl = cm.slot_from_x(x)
    pol, why = cm.cascade_polar(prob.tc, ff, defl, gap, ovl,
                                ok["Re_mac"], n_nodes=prob.n_section_nodes)
    assert why is None
    plate = prob.plate_section()
    assert np.rad2deg(pol.alpha_L0) < -8.0
    assert -4.0 < np.rad2deg(plate.alpha_L0) < 0.0

    wrong = cm.CarWingMultiProblem(plate_polar=pol)
    out = cm.evaluate_car_wing_multi(x, wrong)
    assert out["feasible"] is False
    assert "endplate" in out["reason"]


def test_the_plate_is_charged_exactly_what_the_single_element_family_charges_it():
    prob = cm.CarWingMultiProblem()
    from aerobo.polar import default_polar_family
    ref = default_polar_family().at(prob.tc)
    plate = prob.plate_section()
    assert plate is ref or (plate.a_lin == ref.a_lin
                            and plate.alpha_L0 == ref.alpha_L0)


def test_each_profile_drag_split_is_the_element_it_names():
    """MUTATION: feeding ``CDp_main`` the FLAP's drag table.

        np.interp(a_wing, pol.alpha_deg, pol.cd_main)
                                      -> pol.cd_flap

    This test used to assert only that the three splits add back up to
    ``CDp``. That identity is a TAUTOLOGY: ``CDp_flap`` is DEFINED as
    ``cdp_w / S - CDp_main``, so it closes for any value of ``CDp_main``
    whatsoever. The mutation above swaps the two reported numbers — they
    differ by 3.49x at the box centre — and the old assertion stayed green.

    So each split is re-derived INDEPENDENTLY, from the drag bridge the
    module docstring states rather than from the line under test: the
    element's own equivalent incidence, read on the wide-alpha table at the
    ELEMENT's own Reynolds number, weighted by its OWN chord fraction. The
    two derivations differ only in interpolation order (this one interpolates
    the incidence and then reads the table; the module tabulates the drag and
    interpolates that), which is 2e-4 relative — three orders below the
    factor the mutation moves either number by.
    """
    prob = cm.CarWingMultiProblem()
    out = cm.evaluate_car_wing_multi(_centre(prob), prob)
    assert out["feasible"], out.get("reason")
    pol, res = out["section"], out["vlm"]
    on_wing = ~res.is_winglet
    a = np.rad2deg(res.alpha_eff)[on_wing]
    w = res.c[on_wing] * res.width[on_wing] / res.S

    def charged(alpha_eq_table, c_frac, re):
        """(c_i / c_ref) * cd_i(alpha_eq_i) integrated over the strips."""
        tab, _re_used, _clamped = cm._wide_polar(pol.tc, re)
        lo, hi = tab.alpha_valid
        aeq = np.interp(a, pol.alpha_deg, alpha_eq_table)
        cd = np.asarray(tab.cd(np.clip(aeq, lo, hi)), dtype=float)
        return float(np.sum(c_frac * cd * w))

    exp_main = charged(pol.alpha_eq_main, pol.c_main, pol.re_main)
    exp_flap = charged(pol.alpha_eq_flap, pol.c_flap, pol.re_flap)
    assert out["CDp_main"] == pytest.approx(exp_main, rel=1e-3)
    assert out["CDp_flap"] == pytest.approx(exp_flap, rel=1e-3)

    # the two are not interchangeable — which is what the mutation trades on
    assert out["CDp_main"] / out["CDp_flap"] == pytest.approx(3.49, abs=0.15)
    assert abs(out["CDp_main"] - exp_flap) > 0.5 * out["CDp_main"]

    # ...and, still, everything adds up
    total = out["CDp_main"] + out["CDp_flap"] + out["CDp_endplate"]
    assert abs(total - out["CDp"]) < 1e-12
    assert out["CD"] == pytest.approx(
        out["CDi"] + out["CDp"] + out["cd0_struts"] + out["CD_junction"])


# ------------------------------------------------ 6. per-element reporting


def test_each_element_is_reported_on_its_own_reference_and_the_shares_close():
    prob = cm.CarWingMultiProblem()
    out = cm.evaluate_car_wing_multi(_centre(prob), prob)
    assert out["feasible"], out.get("reason")
    assert out["n_elements"] == 2
    assert abs(out["lift_share_main"] + out["lift_share_flap"] - 1.0) < 1e-12
    assert 0.0 < out["lift_share_flap"] < out["lift_share_main"]

    # each element's Reynolds number is its OWN chord's
    assert out["Re_main"] / out["Re_flap"] == pytest.approx(
        out["c_main_frac"] / out["c_flap_frac"], rel=1e-9)
    assert out["Re_main"] < out["Re_mac"] and out["Re_flap"] < out["Re_main"]

    # each element's own cl is bigger than the section's, because the section
    # is referenced to the STOWED chord and each element to a shorter one
    assert out["cl_own_main"] > out["CZ"]
    # ...and each carries its own suction peak and its own margin
    assert out["cp_min_main"] < 0.0 and out["cp_min_flap"] < 0.0
    assert out["cp_min_main"] >= out["cp_ceiling"]
    assert out["cp_min_flap"] >= out["cp_ceiling"]
    assert out["stall_margin_main"] >= 0.0 and out["stall_margin_flap"] >= 0.0
    assert out["stall_margin_main"] == pytest.approx(
        1.0 - out["cp_min_main"] / out["cp_ceiling"])


def test_the_main_element_carries_most_of_the_load_and_stalls_first():
    """Direction gate on the per-element split, over the deflection band."""
    prob = cm.CarWingMultiProblem()
    x = _centre(prob)
    for defl in (5.0, 15.0, 25.0):
        xx = x.copy()
        xx[cm.SLOT_ROW_OFFSET + 1] = defl
        out = cm.evaluate_car_wing_multi(xx, prob)
        if not out["feasible"]:
            continue
        assert out["lift_share_main"] > 0.5
        assert out["stall_margin_main"] < out["stall_margin_flap"], (
            "the main element is the one a slotted section stalls on")


# ------------------------------------------------------------- 7. the cache


def test_the_cache_returns_the_same_section_and_a_moved_slot_a_different_one():
    args = (0.12, 0.30, 18.0, 0.026, 0.021, 9.5e5)
    a, _ = cm.cascade_polar(*args)
    b, _ = cm.cascade_polar(*args)
    # bit-for-bit: a cache that returns "close enough" is a second solver
    np.testing.assert_array_equal(a.CL, b.CL)
    np.testing.assert_array_equal(a.CD, b.CD)
    assert a.a_lin == b.a_lin and a.alpha_L0 == b.alpha_L0
    moved = list(args)
    moved[2] = 18.5
    c, _ = cm.cascade_polar(*moved)
    assert not np.array_equal(a.CL, c.CL)


def test_the_cache_key_is_the_geometry_and_not_the_reynolds_number():
    """Two Reynolds numbers, one panel solve — and different drag tables."""
    a, _ = cm.cascade_polar(0.12, 0.30, 18.0, 0.026, 0.021, 9.5e5)
    b, _ = cm.cascade_polar(0.12, 0.30, 18.0, 0.026, 0.021, 3.0e5)
    np.testing.assert_array_equal(a.CL, b.CL)          # inviscid: identical
    assert a.re_main != b.re_main
    assert not np.array_equal(a.CD, b.CD)              # viscous: not


def test_a_caller_supplied_section_is_keyed_by_its_contents():
    """Two equal arrays must give one answer; two different ones, two."""
    c1 = airfoil.naca4_coords("2412", cm.N_SECTION_NODES)
    c2 = airfoil.naca4_coords("2412", cm.N_SECTION_NODES)
    c3 = airfoil.naca4_coords("2415", cm.N_SECTION_NODES)
    a, _ = cm.cascade_polar(0.12, 0.30, 18.0, 0.026, 0.021, 9.5e5, coords=c1)
    b, _ = cm.cascade_polar(0.12, 0.30, 18.0, 0.026, 0.021, 9.5e5, coords=c2)
    d, _ = cm.cascade_polar(0.12, 0.30, 18.0, 0.026, 0.021, 9.5e5, coords=c3)
    np.testing.assert_array_equal(a.CL, b.CL)
    assert not np.array_equal(a.CL, d.CL)
    assert abs(d.tc - 0.15) < 5e-3          # thickness MEASURED, not asked


def test_a_closed_naca_is_returned_unchanged_by_the_closure():
    """The closure is applied always and is an exact no-op where it should be."""
    c = airfoil.naca4_coords("2412", cm.N_SECTION_NODES)
    x, y = panel2d.close_trailing_edge(c[:, 0], c[:, 1])
    np.testing.assert_array_equal(x, c[:, 0])
    np.testing.assert_array_equal(y, c[:, 1])


# --------------------------------------------------- 8. the matched control


def test_the_single_element_twin_is_carwing_untouched():
    """The comparison arm must BE carwing, not a lookalike."""
    prob = cm.CarWingMultiProblem()
    x = _centre(prob)
    p1 = cm.single_element_problem(prob)
    x1 = cm.single_element_x(x, prob)
    assert x1.size == p1.dim == 7
    np.testing.assert_array_equal(x1, np.delete(x, [6, 7, 8, 9]))

    hand = carwing.CarWingProblem(
        b=prob.b, S=prob.S, tc=prob.tc, mount=prob.mount, V=prob.V)
    a = carwing.evaluate_car_wing(x1, p1)
    b = carwing.evaluate_car_wing(x1, hand)
    assert a["feasible"] and b["feasible"]
    assert a["CZ"] == b["CZ"] and a["CD"] == b["CD"]     # bit-for-bit


def test_the_wide_alpha_control_reaches_further_than_the_cruise_one():
    """The control exists because the two families read different tables."""
    prob = cm.CarWingMultiProblem()
    p_cruise = cm.single_element_problem(prob)
    p_wide = cm.single_element_problem(prob, wide_alpha=True)
    assert p_cruise.section_polar is None
    assert p_wide.section_polar is not None
    assert p_wide.section_polar.alpha_valid[1] > 14.0
    assert p_cruise.polar_family.at(prob.tc).alpha_valid[1] == 14.0


# ---------------------------------------------------------- 9. the claim


@pytest.mark.parametrize("cd_budget, expect_gain", [(0.015, False),
                                                    (0.11, True)])
def test_whether_the_slot_pays_depends_on_the_drag_budget(cd_budget,
                                                          expect_gain):
    """The measured answer, held by its SIGN.

    Against the wide-alpha control (both families on the same drag table),
    the slot LOSES at a very tight budget — where the wing is barely loaded
    and the extra element is wetted area buying lift nobody asked for — and
    WINS once there is drag to spend. The magnitudes are in the module's
    report; what a test can hold is that the sign changes, and that it
    changes in this direction and not the other.
    """
    from dataclasses import replace

    prob = replace(cm.CarWingMultiProblem(), CD_budget=cd_budget)
    d = cm.compare_split(_centre(prob), prob, alpha_walk_deg=(0.0, 24.0),
                         n_alpha=25, n_deflection=15)
    assert d["CZ_single_wide"] is not None
    assert d["CZ_multi"] is not None
    if expect_gain:
        assert d["delta_frac_wide"] > 0.01
    else:
        assert d["delta_frac_wide"] < 0.0


def test_the_comparison_reports_which_limit_each_answer_rides():
    """A comparison whose arms sit on different limits is not a comparison."""
    prob = cm.CarWingMultiProblem()
    d = cm.compare_split(_centre(prob), prob, alpha_walk_deg=(0.0, 24.0),
                         n_alpha=25, n_deflection=11)
    assert "drag budget" in d["limit_single_wide"]
    assert "drag budget" in d["limit_multi"]
    # and with the walk left at the family's own row the single-element arm
    # is INCIDENCE limited instead — the artefact the default walk exists to
    # remove, asserted so that removing the widening cannot pass
    d0 = cm.compare_split(_centre(prob), prob, alpha_walk_deg=None,
                          n_alpha=13, n_deflection=7)
    assert "incidence ceiling" in d0["limit_single"]
    assert d0["delta_frac_wide"] > d["delta_frac_wide"]


def test_the_limit_reported_does_not_move_with_the_walk_resolution():
    """MUTATION: deciding the limit on PROXIMITY instead of on mechanism.

    Restore ``compare_split``'s old rule —

        out["CD"] >= prob.CD_budget * (1.0 - 1.0 / (n_alpha - 1))

    — and this test goes red at 49 stations. That tolerance is the walk's own
    step: refining the measurement changes the mechanism it reports about an
    answer that has not moved. MEASURED at the box centre with the published
    0.11 budget, the wide-alpha single-element arm's winner is CD 0.10730 at
    13, 25, 49 and 97 stations, and the old rule called it "interior (section
    stall)" / "drag budget" / "interior (section stall)" / "interior (section
    stall)". The middle one is the only one the shipped default happened to
    produce, and the label was false in every case: that arm does not stall
    anywhere in its walk — the station above its winner is feasible at
    CZ 1.7332 and CD 0.12022, which is over budget, and the budget is
    therefore what stopped it.
    """
    prob = cm.CarWingMultiProblem()
    x = _centre(prob)
    seen = {}
    for n_alpha in (13, 25, 49):
        d = cm.compare_split(x, prob, alpha_walk_deg=(0.0, 24.0),
                             n_alpha=n_alpha, n_deflection=5)
        seen[n_alpha] = (d["limit_single_wide"], d["limit_multi"])
    assert set(seen.values()) == {("drag budget", "drag budget")}, seen

    # and the mechanism is really there: a feasible design above the winner,
    # thrown out by the budget and nothing else
    d = cm.compare_split(x, prob, alpha_walk_deg=(0.0, 24.0), n_alpha=25,
                         n_deflection=5)
    win = d["single_wide"]["CZ"]
    blocked = [o for _a, o in d["walk_single_wide"]
               if o["feasible"] and o["CZ"] > win]
    assert blocked and all(o["CD"] > prob.CD_budget for o in blocked)


def test_at_a_matched_downforce_the_slot_costs_less_drag():
    """The other way round: the same CZ, and which family pays less for it."""
    from dataclasses import replace

    prob = cm.CarWingMultiProblem()
    p1 = cm.single_element_problem(prob, wide_alpha=True)
    p1.ALPHA_BOUNDS_DEG = (0.0, 24.0)
    p2 = replace(prob)
    p2.ALPHA_BOUNDS_DEG = (0.0, 24.0)
    x = _centre(prob)
    x1 = cm.single_element_x(x, prob)

    target = 1.4
    best1 = min((carwing.evaluate_car_wing(np.r_[x1[:3], a, x1[4:]], p1)
                 for a in np.linspace(0, 24, 25)),
                key=lambda o: o["CD"] if (o["feasible"] and o["CZ"] >= target)
                else np.inf)
    cands = []
    for a in np.linspace(0, 24, 25):
        for dfl in np.linspace(0, 35, 15):
            xx = x.copy()
            xx[3] = a
            xx[cm.SLOT_ROW_OFFSET + 1] = dfl
            o = cm.evaluate_car_wing_multi(xx, p2)
            if o["feasible"] and o["CZ"] >= target:
                cands.append(o)
    assert cands and best1["feasible"] and best1["CZ"] >= target
    best2 = min(cands, key=lambda o: o["CD"])
    assert best2["CD"] < best1["CD"], (
        f"at CZ >= {target} the two-element wing should need less drag; got "
        f"{best2['CD']:.5f} against {best1['CD']:.5f}")
    assert best2["efficiency"] > best1["efficiency"]


# ------------------------------------------------------ 10. carwing's beam


def test_the_grip_station_changes_the_structure_and_the_parasite_drag():
    """carwing's own mount question, still answered through this family — and
    it is no longer "structure only".

    Both layouts take the load out through the plates, so neither changes the
    CIRCULATION: the same wing at the same incidence makes the same CZ and the
    same induced drag whichever station grips it. What the inboard grip
    changes is the BEAM (a support inboard of the tip cuts the movement
    sharply) and the PARASITE drag, because two extra sheets are wetted area
    and two more corners. That second half is the part that must not go
    missing: priced at zero, the inboard grip buys its stiffness for free and
    an optimiser takes it every time.
    """
    from dataclasses import replace

    prob = cm.CarWingMultiProblem()
    x = _centre(prob)
    tips = cm.evaluate_car_wing_multi(x, replace(prob, mount="tips"))
    inb = cm.evaluate_car_wing_multi(x, replace(prob, mount="inboard"))
    assert tips["feasible"] and inb["feasible"]
    assert tips["CZ"] == inb["CZ"]                  # same lift, same load
    assert tips["CDi"] == inb["CDi"]
    # the tip-borne wing sags at the centre; gripping inboard cuts that
    assert inb["deflection_m"] < tips["deflection_m"]
    # ...and the grip is NOT free
    assert tips["cd0_struts"] == 0.0 and inb["cd0_struts"] > 0.0
    assert tips["CD_junction"] == 0.0 and inb["CD_junction"] > 0.0
    assert inb["CD"] > tips["CD"]
    # the whole CD difference IS those two charges: nothing else moved
    assert (inb["CD"] - tips["CD"]) == pytest.approx(
        inb["cd0_struts"] + inb["CD_junction"], rel=1e-9)


def test_the_frame_is_the_mirrored_one_and_says_so():
    prob = cm.CarWingMultiProblem()
    out = cm.evaluate_car_wing_multi(_centre(prob), prob)
    assert out["CZ"] > 0.0
    assert out["CZ"] == out["CL_model"]
    assert "mirrored" in out["frame"]
    assert out["downforce_N"] == pytest.approx(
        out["q_Pa"] * out["S_m2"] * out["CZ"])
