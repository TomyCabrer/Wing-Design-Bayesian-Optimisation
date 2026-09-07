"""section_stall: the stall of the shape being flown, not of a stand-in.

The module exists to remove two substitutions from the two-element section's
stall criterion — the wrong SHAPE and the wrong REYNOLDS NUMBER — so the gates
are about exactly those two, plus the censoring rule that decides when there
is a stall to report at all.

The strongest gate here is an AGREEMENT gate, and it is the one to read first:
where the measured path and the substituted path are asking the same question,
they must give the same answer to the bit. A NACA 2412 at Re 1e6 on
``carwing_multi``'s own 120-node loop is exactly that case, and if the two
disagree there then one of them is not doing what its docstring says.

Everything that drives XFOIL is marked ``slow``.
"""

from __future__ import annotations

import numpy as np
import pytest

from aerobo import carwing_multi as cm, panel2d, section_stall as ss


def _flown(tc: float, n_nodes: int = cm.N_SECTION_NODES) -> np.ndarray:
    """The loop the cascade will actually panel, as an (n, 2) array."""
    x, y = cm.element_base_coords(tc, n_nodes)
    return np.column_stack([x, y])


def test_the_sweep_is_the_shipped_banks_own():
    """Same range, same step, same censoring tolerance as the bank.

    A measured ceiling and a substituted one differ in the SHAPE and the
    REYNOLDS NUMBER. If they also differed in the sweep or in what counts as
    a resolved peak, no comparison between them would mean anything.

    MUTATION: change ``STALL_SWEEP_DEG`` to ``(-6.0, 20.0, 0.5)``. The
    agreement test below moves the measured stall angle off 16.0 deg for
    thick sections and its bit-for-bit assertion fails.
    """
    from aerobo.polar import stall_point

    assert ss.STALL_SWEEP_DEG == (-6.0, 22.0, 0.5)
    a = ss.stall_sweep_alphas()
    assert a[0] == -6.0 and a[-1] == 22.0
    assert np.allclose(np.diff(a), 0.5)
    assert 0.0 in a                      # the split-at-zero march needs it
    # the censoring tolerance is polar.stall_point's own default
    assert ss.STALL_DROP_TOL == stall_point.__defaults__[-1]


def test_a_section_that_is_not_a_section_is_a_reason_not_an_exception():
    """In-contract failures RETURN; only infrastructure raises."""
    for bad, key in ((np.zeros((3, 2)), "at least"),
                     (np.full((40, 2), np.nan), "finite")):
        got = ss.measured_stall(bad, 1.0e6)
        assert not got["feasible"] and key in got["reason"]
    got = ss.measured_stall(_flown(0.12), -1.0)
    assert not got["feasible"] and "Reynolds" in got["reason"]


@pytest.mark.slow
def test_the_measured_ceiling_reproduces_the_substituted_one_exactly():
    """Where the two paths ask the same question they give the same answer.

    A NACA 2412 at Re 1e6 on the cascade's own 120-node loop is the ONE case
    in which the substitution is not a substitution: the shape is a NACA 24XX
    and the Reynolds number is the bank's. So the XFOIL-measured stall angle
    must be the bank's stall angle, and the panel solve at it must give the
    bank-derived ceiling to the bit — not to a tolerance, because both sides
    end in the same ``panel2d`` call on the same array.

    MUTATION: in ``measured_ceiling``, solve at ``alpha_stall + 1.0``. The
    equality fails by 1.3 in Cp_min.
    """
    sub_cp, sub_a = cm.suction_peak_ceiling(0.12, cm.N_SECTION_NODES)
    got = ss.measured_ceiling(_flown(0.12), 1.0e6)
    assert got["feasible"], got["reason"]
    assert got["alpha_stall_deg"] == sub_a
    assert got["cp_ceiling"] == sub_cp
    assert got["ceiling"] == (sub_cp, sub_a)
    assert got["n_panels"] == cm.N_SECTION_NODES - 1


@pytest.mark.slow
def test_the_ceiling_is_a_property_of_the_reynolds_number_it_is_measured_at():
    """Lower Reynolds number, earlier stall, shallower ceiling — measured.

    This is the direction ``carwing_multi``'s docstring stated and declined
    to quantify: its ceiling is a Re-1e6 number applied at whatever Reynolds
    number the wing flies, and a flap on a small car wing reaches ~2e5. The
    test asserts the ORDERING (which is the physics) and a floor on the SIZE
    (which is what makes the substitution worth removing rather than
    labelling).

    MUTATION: in ``measured_ceiling``, pass ``1.0e6`` to ``measured_stall``
    instead of ``re``. Every Reynolds number then returns one ceiling and
    both the ordering and the size assertions fail.
    """
    c = _flown(0.12)
    rows = [ss.measured_ceiling(c, re) for re in (1.0e6, 3.0e5, 1.5e5)]
    for r in rows:
        assert r["feasible"], r["reason"]
    a = [r["alpha_stall_deg"] for r in rows]
    cp = [r["cp_ceiling"] for r in rows]
    assert a[0] > a[1] > a[2], a          # stalls earlier as Re falls
    assert cp[0] < cp[1] < cp[2], cp      # shallower ceiling (all negative)
    # and it is not a rounding-level effect: at the Reynolds number a flap
    # on a small car wing reaches, the Re-1e6 ceiling is far too deep
    assert cp[2] / cp[0] < 0.80, cp


@pytest.mark.slow
def test_a_censored_table_is_refused_rather_than_used():
    """A march that ran out is a LOWER BOUND on the stall, not a stall.

    Forced by truncating the sweep below the peak, which is exactly what a
    right-censored table is. The refusal has to name the censoring, because
    the alternative — quietly taking the last converged row as the stall — is
    a ceiling with nothing behind it, and it is the error
    `censored-clmax-is-a-lower-bound` records.

    MUTATION: in ``measured_stall``, drop the ``past.size and ...`` test and
    always accept ``argmax``. The refusal assertion fails.
    """
    got = ss.measured_stall(_flown(0.12), 1.0e6, sweep=(-6.0, 8.0, 0.5))
    assert not got["feasible"]
    assert got["censored"] is True
    assert "censored" in got["reason"].lower()
    assert got["cl_max"] > 0.0            # there IS a maximum; it is not a stall


@pytest.mark.slow
def test_the_measured_pair_is_what_cascade_polar_takes():
    """End to end: measure both elements, hand the pair back, fly it.

    The two paths must agree where they should (a NACA 24XX pair at the
    element Reynolds numbers the cascade itself computes) and the polar must
    say which one produced it.

    MUTATION: in ``cascade_polar``, ignore ``ceilings`` and always take the
    substitution. The ``ceiling_source`` assertion fails.
    """
    tc, ff = 0.12, 0.275
    re_ref = 1.0e6
    c = _flown(tc)
    pair, why = ss.measured_ceilings(c, c, re_ref * (1.0 - ff), re_ref * ff)
    assert why is None, why

    meas, why = cm.cascade_polar(tc, ff, 17.5, 0.031, 0.02, re_ref,
                                 ceilings=pair)
    assert why is None, why
    assert meas.ceiling_source == "measured"
    assert "measured on the section flown" in cm._ceiling_section_label(meas)
    # the main element's Reynolds number is 7.25e5, the flap's 2.75e5: the
    # substitution reads BOTH off a Re-1e6 bank, so the measured ceilings are
    # shallower and the flap's much more so
    sub, _ = cm.cascade_polar(tc, ff, 17.5, 0.031, 0.02, re_ref)
    # WHAT THE SUBSTITUTION COSTS, AT THIS OPERATING POINT. The main element
    # flies at Re 7.25e5 and the flap at 2.75e5; MEASURED, a NACA 2412 stalls
    # at 16.0 deg from Re 1e6 all the way down to 5e5 and only then starts to
    # move (15.5 at 2.75e5, 14.5 at 2e5, 13.5 at 1.5e5, 12.0 at 1e5). So the
    # main element's measured ceiling is the substituted one EXACTLY, and the
    # flap's is 5.7 % shallower — the substitution is free on the element
    # that does not need it and not free on the element that does.
    assert meas.cp_ceiling_main == sub.cp_ceiling_main
    assert meas.cp_ceiling_flap > sub.cp_ceiling_flap
    assert 0.90 < meas.cp_ceiling_flap / sub.cp_ceiling_flap < 0.98
    # a shallower ceiling is a STRICTER criterion: the flyable window cannot
    # widen
    assert (meas.alpha_valid[1] - meas.alpha_valid[0]
            <= sub.alpha_valid[1] - sub.alpha_valid[0])


@pytest.mark.slow
def test_the_ceiling_is_read_on_the_grid_the_cascade_flies():
    """A ceiling read on a different sampling of the same shape is wrong.

    Cp_min is a panel-count quantity, so a ceiling on one grid and a peak on
    another do not cancel their shared discretisation error. MEASURED: the
    same NACA 2412 at its own measured stall gives -13.8808 on the 120-node
    loop and -13.9441 on the 200-node one — 0.46 %, which lands straight in
    the margin. The module's contract is that the CALLER passes the flown
    loop; this asserts the two really are different numbers, so that contract
    is load-bearing rather than decorative.
    """
    a = ss.measured_ceiling(_flown(0.12, 120), 1.0e6)
    b = ss.measured_ceiling(_flown(0.12, 200), 1.0e6)
    assert a["feasible"] and b["feasible"]
    assert a["alpha_stall_deg"] == b["alpha_stall_deg"]     # same shape
    assert a["cp_ceiling"] != b["cp_ceiling"]               # different grid
    assert abs(b["cp_ceiling"] / a["cp_ceiling"] - 1.0) > 2e-3
    assert a["cp_ceiling"] == cm.suction_peak_ceiling(0.12, 120)[0]


@pytest.mark.slow
def test_a_designed_section_gets_a_ceiling_of_its_own():
    """The case the substitution cannot serve: a shape that is not a NACA.

    A cambered CST section of the same THICKNESS as a NACA 2412 gets a
    different measured stall and therefore a different ceiling — which is the
    whole content of "the substitution is doing more work now".

    MUTATION: none — this asserts the module does the thing it exists for.
    If a designed section ever comes back with the NACA's ceiling, the
    measurement is not being made.
    """
    from aerobo import airfoil, carsection

    wu, wl = airfoil.cst_anchor("6412", 4)
    u, l = carsection.scaled_weights(wu, wl, 0.12)
    coords = airfoil.cst_coords(u, l, cm.N_SECTION_NODES)
    assert airfoil.cst_thickness(u, l) == pytest.approx(0.12, abs=1e-9)
    assert panel2d.check_body(coords[:, 0], coords[:, 1]) is None

    got = ss.measured_ceiling(coords, 1.0e6)
    assert got["feasible"], got["reason"]
    naca_cp, naca_a = cm.suction_peak_ceiling(0.12, cm.N_SECTION_NODES)
    assert got["alpha_stall_deg"] != naca_a
    assert abs(got["cp_ceiling"] / naca_cp - 1.0) > 0.02
