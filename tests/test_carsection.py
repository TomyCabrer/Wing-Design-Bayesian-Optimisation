"""The two-element SECTION as a design problem, against what it claims.

Five groups, and the order is the order in which a reader should stop
believing the study if one of them fails.

1. THE GEOMETRY THE FLAP GOT. ``carwing_multi`` could not give the flap a
   shape of its own; now it can, and the gates are that the OLD path is
   unchanged to the bit, that the NEW path actually changes the answer, and
   that every per-element quantity the criterion reads — thickness, Reynolds
   number, isolated map, drag table, suction-peak ceiling — is the element's
   own rather than the main element's borrowed.

2. THE THREE SLOT LENGTHS. A placement row, a resolution width and a
   trailing-edge gap. They are three numbers and the study's one imported
   bound is stated against the third; a test that let two of them be the same
   quantity would make that bound mean something else.

3. THE BRIDGE. A 2-D objective that scores a wing needs a reduction, and a
   reduction is worth exactly what it reproduces on data it was not fitted
   to. Gated on HELD-OUT sections, and then on the thing that actually
   matters — that the reduction's own error is small against the spread of
   lap times it is being asked to rank.

4. THE OBJECTIVE. The lap and the weighted control are not the same question,
   and the control's own defect (it cannot state an incidence) is asserted
   rather than described.

5. THE REFUSAL SENTINEL, on a car slow enough to make the obvious sentinel
   wrong. This repository has shipped a scale-blind sentinel before.

A NOTE ON MUTATION. Every test below carries a "MUTATION" line naming the
exact edit that must turn it red. Each was applied to the source, run,
confirmed red, and restored (``git diff`` clean afterwards). A test whose
mutation has not been run is a test that has not been trusted.
"""

from __future__ import annotations

import numpy as np
import pytest

from aerobo import airfoil, carsection as cs, carwing_multi as cm, panel2d

REF_SLOT = (0.275, 17.5, 0.031, 0.020)
RE_REF = 1.0e6


# ------------------------------------------- 1. the flap got its own section


def test_a_flap_with_no_section_of_its_own_is_the_old_answer_bit_for_bit():
    """The single-shape path did not move, and the new one is not a no-op.

    Two statements in one test because either alone is satisfiable by a
    mistake: "unchanged" passes if the flap freedom is wired to nothing, and
    "changed" passes if the default silently changed too.

    MUTATION: in ``cascade_polar``, set ``tcf_flap = tcf`` unconditionally
    (i.e. ignore ``flap_tc``). The first three assertions still pass; the
    thin-flap ones fail, because a 0.09 flap then flies as a 0.12 one.
    """
    base, why = cm.cascade_polar(0.12, *REF_SLOT, RE_REF)
    assert why is None
    same, why = cm.cascade_polar(0.12, *REF_SLOT, RE_REF, flap_tc=0.12)
    assert why is None
    # bit-for-bit, not approx: a flap "the same shape" must be the same solve
    np.testing.assert_array_equal(base.CL, same.CL)
    np.testing.assert_array_equal(base.CD, same.CD)
    assert (base.cp_ceiling_main, base.cp_ceiling_flap) == \
           (same.cp_ceiling_main, same.cp_ceiling_flap)
    assert base.tc_flap == same.tc_flap == base.tc

    thin, why = cm.cascade_polar(0.12, *REF_SLOT, RE_REF, flap_tc=0.09)
    assert why is None
    assert thin.tc_flap == pytest.approx(0.09)
    assert thin.tc == base.tc                      # the MAIN element did not move
    assert not np.array_equal(thin.CL, base.CL)
    assert thin.cp_ceiling_flap != thin.cp_ceiling_main


def test_every_per_element_quantity_is_that_elements_own():
    """Thickness, Reynolds number, ceiling and stall margin, all per element.

    The failure this catches is not "the flap is wrong" but "the flap is
    judged as though it were the main element" — which admits or refuses a
    designed flap for a shape it does not have, silently, with every number
    in the report looking reasonable.

    MUTATION: in ``CascadePolar.stall_margin``, divide the flap's peak by
    ``self.cp_ceiling_main`` (what it did before the flap had a section).
    The margin assertions below fail.
    """
    pol, why = cm.cascade_polar(0.12, *REF_SLOT, RE_REF, flap_tc=0.09)
    assert why is None
    # a THINNER section has a sharper nose: it stalls EARLIER and at a DEEPER
    # suction peak. MEASURED on the shipped bank at 120 nodes: t/c 0.09 peaks
    # at Cp_min -16.778 at 14.0 deg, t/c 0.12 at -13.881 at 16.0 deg
    assert pol.cp_ceiling_flap < pol.cp_ceiling_main       # both negative
    assert pol.alpha_stall_iso_flap_deg < pol.alpha_stall_iso_deg
    # each element's Reynolds number is its own chord's
    assert pol.re_flap == pytest.approx(RE_REF * pol.c_flap, rel=1e-9)
    assert pol.re_main == pytest.approx(RE_REF * pol.c_main, rel=1e-9)
    # ...and the margin is the peak against the element's OWN ceiling
    a = 0.0
    g_main, g_flap = pol.stall_margin(a)
    assert g_main == pytest.approx(
        1.0 - float(np.interp(a, pol.alpha_deg, pol.cpmin_main))
        / pol.cp_ceiling_main)
    assert g_flap == pytest.approx(
        1.0 - float(np.interp(a, pol.alpha_deg, pol.cpmin_flap))
        / pol.cp_ceiling_flap)
    # the two ceilings differ, so borrowing one for the other is measurable
    borrowed = 1.0 - float(np.interp(a, pol.alpha_deg, pol.cpmin_flap)) \
        / pol.cp_ceiling_main
    assert abs(borrowed - g_flap) > 0.01


def test_a_designed_flap_changes_the_section_and_the_label_says_so():
    """A CST flap flies, and the report marks the ceiling as a substitution.

    The label is the family's flag on its largest single assumption, so it is
    gated like a number: a designed element must be NAMED as one, or a reader
    of a result cannot tell which of the two ceiling paths produced it.

    MUTATION: delete the ``" SUBSTITUTED for a designed section"`` branch in
    ``_ceiling_section_label``. The label assertions fail.
    """
    wu, wl = airfoil.cst_anchor("6412", 4)
    coords = airfoil.cst_coords(wu, wl, 200)
    plain, _ = cm.cascade_polar(0.12, *REF_SLOT, RE_REF)
    des, why = cm.cascade_polar(0.12, *REF_SLOT, RE_REF, flap_coords=coords)
    assert why is None
    assert des.flap_is_own_section and des.flap_is_designed
    assert not des.main_is_designed
    assert not np.array_equal(des.CL, plain.CL)
    label = cm._ceiling_section_label(des)
    assert "SUBSTITUTED for a designed section" in label
    assert label.startswith("main NACA2412 / flap ")
    assert des.ceiling_source == "naca24xx-substitution"
    # ...and the single-shape label is exactly what it always was
    assert cm._ceiling_section_label(plain) == \
        "NACA2412 (wide-alpha XFOIL, Re 1e6)"


def test_flap_tc_and_flap_coords_together_are_a_call_site_error():
    """Two statements of one shape is a programming error, not a design."""
    with pytest.raises(ValueError, match="read by nothing"):
        cm.CarWingMultiProblem(flap_tc=0.10,
                               flap_coords=airfoil.naca4_coords("2412", 120))


# ------------------------------------------------ 2. the three slot lengths


def test_the_three_slot_lengths_are_three_quantities():
    """Placement gap, minimum separation and trailing-edge gap all differ,
    and two of them are ordered by construction.

    ``slot_width_frac <= slot_gap_te_frac`` identically: a minimum over the
    whole surface cannot exceed the distance from one point of it. That
    identity is the cheap check that these are not one quantity under two
    names — which is exactly the confusion an imported gap bound would land
    in (LITERATURE_REVIEW_S63_CARWING.md section 1.4).

    MEASURED, and the structure is asserted rather than a bound on it: on a
    0.275 c flap the two are EQUAL up to about 10 deg of deflection, because
    the closest point of the flap to the main element really is the main
    element's trailing edge there, and they SEPARATE from 15 deg, because the
    turning flap brings its upper surface up under the main element's lower
    surface aft of the trailing edge. So the deflection is swept: a test at
    one rigging would assert the wrong half of that.

        deflection     0      5     10     15     20     25     30   deg
        te gap      .01689 .01917 .02131 .02325 .02506 .02665 .02810
        width       .01689 .01917 .02131 .02321 .02460 .02560 .02637
        (placement gap 0.031 c throughout)

    MUTATION: make ``slot_gap_te`` return ``_min_gap(main, flap)``. Every
    ``strict`` case becomes an equality and the count assertion fails.
    """
    seen = strict = equal = 0
    for defl in (0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0):
        for gap in (0.020, 0.031, 0.045):
            pol, why = cm.cascade_polar(0.12, 0.275, defl, gap, 0.02, RE_REF)
            if why is not None:
                continue
            seen += 1
            assert pol.slot_width_frac <= pol.slot_gap_te_frac + 1e-12
            assert pol.slot_gap_te_frac != gap          # never the placement row
            if defl >= 15.0:
                assert pol.slot_width_frac < pol.slot_gap_te_frac
                strict += 1
            else:
                assert pol.slot_width_frac == pytest.approx(
                    pol.slot_gap_te_frac, rel=1e-12)
                equal += 1
    assert seen >= 18, "the sweep must actually fly something"
    assert strict >= 9 and equal >= 8


def test_the_non_merging_floor_refuses_below_itself_and_is_off_by_default():
    """The floor bites where it should, and nowhere the family has published.

    Two halves. (a) A rigging whose trailing-edge gap is under the floor is
    REFUSED with a reason naming the floor, and the same rigging flies with
    the floor off — so the refusal is the floor's and not the geometry's.
    (b) The default is off, so every published number of this family is
    unmoved.

    MUTATION: delete the ``gap_te_min_frac`` block in ``cascade_polar``. Half
    (a)'s refusal assertion fails.
    """
    tight = (0.12, 0.30, 5.0, 0.014, 0.02, RE_REF)
    free, why = cm.cascade_polar(*tight)
    assert why is None, "the control rigging must fly with the floor off"
    assert free.slot_gap_te_frac < cm.SLOT_GAP_TE_NON_MERGING_FRAC

    bound, why = cm.cascade_polar(
        *tight, gap_te_min_frac=cm.SLOT_GAP_TE_NON_MERGING_FRAC)
    assert bound is None
    assert "trailing-edge gap" in why and "non-merging floor" in why

    # ...and a rigging above the floor is untouched by it, to the bit
    wide = (0.12, *REF_SLOT, RE_REF)
    a, _ = cm.cascade_polar(*wide)
    b, _ = cm.cascade_polar(*wide,
                            gap_te_min_frac=cm.SLOT_GAP_TE_NON_MERGING_FRAC)
    assert b is not None and a.slot_gap_te_frac > \
        cm.SLOT_GAP_TE_NON_MERGING_FRAC
    np.testing.assert_array_equal(a.CL, b.CL)

    assert cm.CarWingMultiProblem().slot_gap_te_min_frac is None
    assert cs.SectionProblem().slot_gap_te_min_frac == \
        cm.SLOT_GAP_TE_NON_MERGING_FRAC


def test_the_criterion_scale_is_the_sensitivity_knob_and_one_changes_nothing():
    """``ceiling_scale`` widens and narrows the flyable window, and 1.0 is
    the shipped answer to the bit.

    Every result this family produces has to carry its own
    criterion-sensitivity sweep, and a knob that quietly moved the default
    would make the sweep a comparison against a moved baseline.

    MUTATION: apply ``scale`` to only ONE of the two ceilings in
    ``cascade_polar``. The monotone-window assertion still passes; the
    equal-ceilings-scale-equally assertion below fails.
    """
    base, _ = cm.cascade_polar(0.12, *REF_SLOT, RE_REF)
    one, _ = cm.cascade_polar(0.12, *REF_SLOT, RE_REF, ceiling_scale=1.0)
    np.testing.assert_array_equal(base.CL, one.CL)
    assert base.alpha_valid == one.alpha_valid

    wide, _ = cm.cascade_polar(0.12, *REF_SLOT, RE_REF, ceiling_scale=1.15)
    tight, _ = cm.cascade_polar(0.12, *REF_SLOT, RE_REF, ceiling_scale=0.85)
    assert wide.alpha_valid[1] > base.alpha_valid[1] > tight.alpha_valid[1]
    for pol, s in ((wide, 1.15), (tight, 0.85)):
        assert pol.cp_ceiling_main == pytest.approx(base.cp_ceiling_main * s)
        assert pol.cp_ceiling_flap == pytest.approx(base.cp_ceiling_flap * s)

    with pytest.raises(ValueError, match="ceiling_scale"):
        cm.CarWingMultiProblem(ceiling_scale=0.0)


def test_the_element_ceiling_lets_a_section_set_its_own_and_the_default_does_not():
    """The criterion's own worst failure mode, asserted rather than described.

    With ``ceiling_shape="element"`` the ceiling is the ELEMENT's own Cp_min
    at a fixed incidence, so the test "your peak must be no deeper than your
    ceiling" degenerates into a bound on the ANGLE — and a section can raise
    its own ceiling by having a deeper peak at that angle, which rewards a
    sharp nose. An optimiser finds that immediately: both of this study's
    first-attempt 2-D winners were infeasible the moment their stall was
    measured on them.

    The gate is the mechanism, not the anecdote: a DEEPER-nosed section must
    get a deeper ceiling under ``"element"`` and the SAME ceiling under
    ``"naca"``, at one thickness.

    MUTATION: change ``CarWingMultiProblem.ceiling_shape``'s default to
    ``"element"``. The default-is-naca assertions fail.
    """
    tc = 0.12
    shapes = []
    for code in ("2412", "6412", "0012"):
        wu, wl = airfoil.cst_anchor(code, 4)
        u, l = cs.scaled_weights(wu, wl, tc)
        shapes.append((code, airfoil.cst_coords(u, l, cm.N_SECTION_NODES)))

    naca = {c: cm.suction_peak_ceiling(tc, cm.N_SECTION_NODES, co, "naca")
            for c, co in shapes}
    elem = {c: cm.suction_peak_ceiling(tc, cm.N_SECTION_NODES, co, "element")
            for c, co in shapes}
    # "naca" is a function of the thickness alone: one number for all three
    assert len(set(naca.values())) == 1
    # "element" is a function of the shape, and the three differ
    assert len({round(v[0], 9) for v in elem.values()}) == 3
    # ...and the stall ANGLE is the same in both, which is the point: only the
    # ceiling moved, so what "element" varies is how permissive it is
    assert {v[1] for v in naca.values()} == {v[1] for v in elem.values()}

    # the defaults
    assert cm.CarWingMultiProblem().ceiling_shape == "naca"
    assert cs.SectionProblem().ceiling_shape == "naca"
    assert cm.CEILING_SHAPES == ("naca", "element")
    with pytest.raises(ValueError, match="unknown ceiling_shape"):
        cm.CarWingMultiProblem(ceiling_shape="whatever")

    # a NACA element gets the same answer either way, to the bit
    plain = airfoil.naca4_coords("2412", cm.N_SECTION_NODES)
    assert cm.suction_peak_ceiling(0.12, cm.N_SECTION_NODES, plain, "naca") \
        == cm.suction_peak_ceiling(0.12, cm.N_SECTION_NODES, plain, "element")
    assert cm.suction_peak_ceiling(0.12, cm.N_SECTION_NODES, plain, "naca") \
        == cm.suction_peak_ceiling(0.12, cm.N_SECTION_NODES)


def test_the_naca_ceiling_is_not_cached_under_the_coordinates_it_ignores():
    """One ceiling per thickness, not one per section of that thickness.

    Under ``"naca"`` the caller's coordinates are read for the THICKNESS and
    for nothing else, so keying the cache on them would store the same number
    once per distinct section and evict the useful entries with copies of one
    answer.

    MUTATION: in ``suction_peak_ceiling``, key on ``_coords_key(coords)``
    unconditionally. The cache-size assertion fails.
    """
    cm._CEIL_CACHE.clear()
    for code in ("2412", "6412", "0012", "4415"):
        wu, wl = airfoil.cst_anchor(code, 4)
        u, l = cs.scaled_weights(wu, wl, 0.12)
        c = airfoil.cst_coords(u, l, cm.N_SECTION_NODES)
        cm.suction_peak_ceiling(0.12, cm.N_SECTION_NODES, c, "naca")
    assert len(cm._CEIL_CACHE) == 1


def test_the_section_cache_evicts_instead_of_freezing():
    """A FULL cache still stores the section a candidate is using now.

    The defect this gates is not a wrong answer, it is a silent 2.5x: under
    the old "stop taking entries when full" policy, a section first seen
    after the cache filled was never stored at all, so a lap that flies one
    section at four speeds re-solved its two panel systems every time.
    MEASURED before the fix, over 700 random draws of the 2-D box: 20.4 ms a
    candidate over the first hundred, 51.2 ms over the last four hundred.

    Identity (``is``) is the assertion, because that is the difference
    between a cache hit and a recomputation that happens to agree. The
    section is NEW — asked for the first time only after the cache is
    already full — which is precisely the case the old policy dropped.

    MUTATION: in ``_cache_put``, replace the body with
    ``if len(cache) < cap: cache[key] = val`` and ``return val``. The final
    identity assertion fails.
    """
    cm._CASCADE_CACHE.clear()
    for i in range(cm._CACHE_MAX + 20):          # fill it, and then some
        cm._cascade_inviscid(0.12, 0.20 + 1e-4 * i, 12.0, 0.030, 0.02,
                             120, None, None, None)
    assert len(cm._CASCADE_CACHE) == cm._CACHE_MAX

    fresh = (0.12, 0.29, 16.0, 0.030, 0.019, 120, None, None, None)
    first, why = cm._cascade_inviscid(*fresh)    # never seen before now
    assert why is None
    again, why = cm._cascade_inviscid(*fresh)
    assert why is None
    assert again is first, (
        "a section first met AFTER the cache filled was not stored, so every "
        "speed of a lap re-solves it")
    assert len(cm._CASCADE_CACHE) == cm._CACHE_MAX


def test_a_rewritten_cache_entry_is_not_the_next_thing_evicted():
    """A key written again is FRESH, not merely still present.

    Python dicts order by FIRST insertion, so ``cache[key] = val`` on an
    existing key leaves it at the oldest end. In a least-recently-used store
    that makes a re-written entry the next eviction — and for
    ``_COORDS_BY_KEY`` that is not a stale-cache nuisance but a crash, because
    the store is written by ``_coords_key`` and read a few statements later by
    ``_cascade_inviscid`` and ``_isolated_maps``.

    MEASURED: with a GA re-evaluating its elite, ``cascade_polar`` re-writes
    the main element's key, then writes the flap's, which tips the store over
    its cap and evicts the main element's — and the next line looks it up.
    Both 23-row GA arms died at every seed; the random and Sobol arms, whose
    sections never repeat, never saw it.

    MUTATION: drop the ``cache.pop(key, None)`` from ``_cache_put``. The
    re-written key is evicted and the search below raises KeyError.
    """
    cap = cm._COORDS_MAX
    store = {}
    for i in range(cap):
        cm._cache_put(store, i, i, cap=cap)
    cm._cache_put(store, 0, 0, cap=cap)          # re-write the OLDEST entry
    cm._cache_put(store, "new", "new", cap=cap)  # ...now push one over the cap
    assert 0 in store, "a re-written entry was evicted before an older one"
    assert 1 not in store, "the actually-oldest entry should have gone"

    # ...and the same thing end to end: one design, evaluated repeatedly, with
    # enough distinct sections in between to fill the store
    prob = cs.SectionProblem()
    x = prob.reference_x().copy()
    x[prob.param_labels.index("tc_main")] = 0.155
    first = cs.evaluate_section(x, prob)
    assert first["feasible"], first["reason"]
    rng = np.random.default_rng(2)
    lo, hi = prob.element_box
    for _ in range(cm._COORDS_MAX):
        y = x.copy()
        y[:prob.n_element_rows] = rng.uniform(lo, hi)
        cs.evaluate_section(y, prob)
        again = cs.evaluate_section(x, prob)     # the ELITE, re-evaluated
        assert again["feasible"], again["reason"]
        assert again["score"] == first["score"]


def test_every_exported_name_exists():
    """``from <module> import *`` must work for every module in the package.

    Nothing in ``aerobo`` does a star import, so an ``__all__`` naming a
    symbol that does not exist is invisible to the whole suite — and two of
    them were (``carwing_multi``'s ``SHARE_MIN_CL`` and
    ``check_section_coords``, both left behind by an earlier draft of the
    section model). The sweep is over every module rather than the one that
    was broken, because the class of defect is not specific to it.

    MUTATION: add ``"nonexistent_name"`` to any module's ``__all__``.
    """
    import importlib
    import pkgutil

    import aerobo

    bad = {}
    n = 0
    for m in pkgutil.iter_modules(aerobo.__path__):
        if m.ispkg:
            continue
        n += 1
        mod = importlib.import_module(f"aerobo.{m.name}")
        missing = [k for k in getattr(mod, "__all__", []) if not hasattr(mod, k)]
        if missing:
            bad[m.name] = missing
    assert n > 40, "the sweep must actually see the package"
    assert bad == {}


# --------------------------------------------------------- 3. the geometry


def test_the_thickness_row_hits_its_target_and_leaves_the_camber_alone():
    """``scaled_weights`` is exact, and it scales THICKNESS, not the section.

    A uniform y-scale would also hit the thickness target, and would move the
    camber line with it — so the section at t/c 0.09 would not be the section
    at t/c 0.18 made thinner, it would be a different shape. The design row
    is meant to be thickness alone.

    MUTATION: return ``(k * wu, k * wl)`` from ``scaled_weights``. The
    thickness assertion still passes; the camber one fails.
    """
    wu, wl = airfoil.cst_anchor("2412", 4)
    cam0 = 0.5 * (wu + wl)
    for tc in (0.09, 0.12, 0.15, 0.18):
        u, l = cs.scaled_weights(wu, wl, tc)
        assert airfoil.cst_thickness(u, l) == pytest.approx(tc, abs=1e-9)
        np.testing.assert_allclose(0.5 * (u + l), cam0, atol=1e-12)
    with pytest.raises(ValueError, match="no thickness to scale"):
        cs.scaled_weights(wu, wu, 0.12)


def test_a_rescaled_section_never_self_intersects_over_the_box():
    """The anchor box's non-crossing proof survives the thickness rescaling.

    ``y_u' - y_l' = k (y_u - y_l)``, so a positive scale cannot make two
    surfaces cross that did not. Asserted on the geometry rather than on the
    algebra, over the box the search actually draws from.
    """
    prob = cs.SectionProblem()
    lo, hi = prob.element_box
    rng = np.random.default_rng(0)
    for _ in range(200):
        v = rng.uniform(lo, hi)
        c = cs.section_coords_of(v[:4], v[4:8], float(v[8]), 160)
        assert panel2d.check_body(c[:, 0], c[:, 1]) is None
        i_le = int(np.argmin(c[:, 0]))
        xu, yu = c[:i_le + 1, 0][::-1], c[:i_le + 1, 1][::-1]
        xl, yl = c[i_le:, 0], c[i_le:, 1]
        g = np.linspace(0.02, 0.98, 200)
        assert np.all(np.interp(g, xu, yu) >= np.interp(g, xl, yl) - 1e-12)


def test_the_thickness_band_is_the_resolved_one():
    """The box's thickness band IS the range the shipped bank resolves.

    A bank that grew or lost a resolved member would make the box wrong in
    one direction (a band reaching where there is no measured stall, so every
    candidate there is refused) or the other (a band that cannot reach a
    thickness the data supports).

    MUTATION: change either end of ``carsection.TC_BOUNDS``.
    """
    assert cs.TC_BOUNDS == cs.resolved_tc_bounds()
    with pytest.raises(ValueError, match="RESOLVES a stall"):
        cs.SectionProblem(tc_bounds=(0.06, 0.18))


# ----------------------------------------------------------- 4. the bridge


def test_the_bridge_reproduces_the_lattice_on_sections_it_never_saw():
    """Held-out validation, not fit quality.

    A surrogate reported only on its own fit is not reported. The held-out
    set shares no row value with the calibration set and varies the overlap,
    which the calibration set deliberately holds fixed.

    MUTATION: pass ``dCL_coef=(0.0, 0.0, 0.0)`` from ``default_bridge`` (i.e.
    drop the non-affine ``alpha_L0`` term). The held-out mean error goes from
    0.26 % to about 5 %, so the 1 % assertion fails.
    """
    br = cs.default_bridge()
    rep = br.fit_report
    assert rep["n_sections"] >= 8 and rep["n_points"] >= 90
    assert rep["fit"]["rel_mean"] < 0.005
    held = rep["holdout"]
    assert held is not None and held["n_sections"] >= 5
    assert held["rel_mean"] < 0.01, held
    assert held["rel_max"] < 0.02, held
    # the held-out set must actually be held out
    assert not (set(cs.BRIDGE_HOLDOUT_SLOTS) & set(cs.BRIDGE_CALIBRATION_SLOTS))


def test_the_bridge_error_is_small_against_the_spread_it_ranks():
    """The reduction cannot reorder anything but near-ties.

    This is the assertion the whole 2-D study rests on and it is the one most
    easily left out: a surrogate accurate to 0.3 % is useless if the thing it
    ranks varies by 0.3 %. Converted into the objective's own unit — seconds
    — the bridge's WORST CL error must move the lap by a small fraction of
    the spread of laps the design box produces.

    MUTATION: pass ``e_lift=1.0, dCL_coef=(0, 0, 0)`` from ``default_bridge``.
    The worst error becomes ~24 % of CL, worth ~0.22 s against a 0.87 s
    spread, and the 10 % assertion fails.
    """
    from aerobo import cartrack

    br = cs.default_bridge()
    car, track = cartrack.CarSpec(), cartrack.synthetic_lap()
    cz, cd, S = 1.667, 0.1013, br.S_m2

    def lap(scale):
        return cartrack.lap_time(cz * scale * S, cd * S, car,
                                 track)["lap_time_s"]

    worst = br.fit_report["holdout"]["rel_max"]
    dt = abs(lap(1.0 + worst) - lap(1.0))

    prob = cs.SectionProblem()
    rng = np.random.default_rng(0)
    B = prob.bounds
    laps = [o["lap_time_s"] for o in
            (cs.evaluate_section(x, prob)
             for x in rng.uniform(B[:, 0], B[:, 1], size=(300, prob.dim)))
            if o["feasible"]]
    assert len(laps) > 50, "the census must find designs to spread"
    spread = max(laps) - min(laps)
    assert spread > 0.3, spread
    assert dt < 0.10 * spread, (dt, spread)


def test_the_bridge_reproduces_the_three_d_family_at_the_reference_design():
    """The 2-D reference section and the 3-D box centre time the same lap.

    Not a tautology: the 2-D path goes section -> bridge -> cartrack and the
    3-D path goes section -> lattice -> cartrack, sharing only the circuit
    and the car. They agree because the bridge is a good reduction, and if it
    stops being one this is where it shows.

    MUTATION: change ``SectionBridge.e_drag`` to 1.0 in ``calibrate_bridge``.
    The CD and lap-time assertions fail.
    """
    from aerobo import cartrack

    prob2 = cs.SectionProblem()
    got2 = cs.evaluate_section(prob2.reference_x(), prob2)
    assert got2["feasible"], got2["reason"]

    prob3 = cm.CarWingMultiProblem(track_spec=cartrack.synthetic_lap(),
                                   objective="laptime")
    got3 = cm.evaluate_car_wing_multi(prob3.bounds.mean(axis=1), prob3)
    assert got3["feasible"], got3["reason"]

    assert got2["CZ"] == pytest.approx(got3["CZ"], rel=0.01)
    assert got2["CD"] == pytest.approx(got3["CD"], rel=0.05)
    assert got2["lap_time_s"] == pytest.approx(got3["lap_time_s"], abs=0.02)


def test_the_lap_reflies_the_section_at_each_speed():
    """Each track point is a re-solve, not one point referred by a q ratio.

    The tell is the DRAG: CZ is Reynolds-independent in this model (the
    lattice reads only the section's linear pair) while CD is not, because
    each element's viscous table is read at its own Reynolds number. A
    referral would give the same CD at every speed.

    MUTATION: in ``evaluate_section``, replace the per-speed
    ``_section_at(V)`` with the base-point polar. Every row's CD becomes
    equal and the strict-monotone assertion fails.
    """
    prob = cs.SectionProblem()
    got = cs.evaluate_section(prob.reference_x(), prob)
    assert got["feasible"], got["reason"]
    rows = got["track_points"]
    assert len(rows) >= 3
    V = np.array([r["V"] for r in rows])
    CD = np.array([r["CD"] for r in rows])
    Re = np.array([r["Re_ref"] for r in rows])
    assert np.all(np.diff(V) > 0)
    assert np.allclose(Re / V, Re[0] / V[0])          # Re tracks the speed
    # drag falls as Reynolds number rises, strictly, at every step
    assert np.all(np.diff(CD) < 0), CD
    assert len(set(np.round(CD, 9))) == len(CD)


# --------------------------------------------------------- 5. the objective


def test_the_lap_and_the_weighted_control_are_not_the_same_question():
    """Two designs, and the two objectives disagree about which is better.

    The control exists to be beaten or to win; it can do neither if it is
    the lap under another name. This asserts they are genuinely different
    orderings, which is the premise of the pre-registered comparison.

    MUTATION: make ``evaluate_section`` score ``weighted`` as
    ``-lap["lap_time_s"]``. The disagreement assertion fails.
    """
    plap = cs.SectionProblem(objective="laptime")
    pw = cs.SectionProblem(objective="weighted")
    rng = np.random.default_rng(3)
    B = plap.bounds
    pairs = []
    for x in rng.uniform(B[:, 0], B[:, 1], size=(200, plap.dim)):
        a = cs.evaluate_section(x, plap)
        if not a["feasible"]:
            continue
        b = cs.evaluate_section(x, pw)
        pairs.append((a["score"], b["score"]))
        if len(pairs) >= 40:
            break
    assert len(pairs) >= 20
    lap = np.array([p[0] for p in pairs])
    wgt = np.array([p[1] for p in pairs])
    # not perfectly rank-correlated: at least one inversion exists
    inversions = sum(1 for i in range(len(lap)) for j in range(i + 1, len(lap))
                     if (lap[i] - lap[j]) * (wgt[i] - wgt[j]) < 0)
    assert inversions > 0


def test_the_weighted_control_cannot_state_an_incidence():
    """The control's score does not depend on the rigging angle. The lap's
    does.

    Recorded as a test rather than as a remark because it is a property of
    the composite objective and not of this implementation: ``cl_max`` and
    ``(l/d)_max`` are maxima OVER incidence, so a weighted sum of them is
    silent about which incidence to fly. The lap is not, because a lap is
    flown at one rigging.

    MUTATION: none needed — this asserts a designed property. If the control
    ever gains an incidence dependence, this test is the notice.
    """
    plap = cs.SectionProblem(objective="laptime")
    pw = cs.SectionProblem(objective="weighted")
    x = plap.reference_x()
    i_alpha = plap.param_labels.index("alpha_deg")
    scores_w, scores_l = [], []
    for a in (2.0, 4.0, 6.0, 8.0):
        xx = x.copy()
        xx[i_alpha] = a
        gl = cs.evaluate_section(xx, plap)
        if not gl["feasible"]:
            continue                    # outside the section's own window
        gw = cs.evaluate_section(xx, pw)
        assert gw["feasible"]
        scores_w.append(gw["score"])
        scores_l.append(gl["score"])
    assert len(scores_l) >= 3, "the rigging sweep must fly at least 3 angles"
    assert len(set(np.round(scores_w, 12))) == 1, scores_w
    assert len(set(np.round(scores_l, 9))) == len(scores_l), scores_l


def test_a_refused_candidate_scores_below_every_feasible_one_even_on_a_slow_lap():
    """The sentinel is scale-aware, and the test is run where it matters.

    ``objective.PENALTY`` is -100, which on THIS circuit sits below every
    feasible lap only because this circuit is fast. Give the reference car a
    fifth of its power and the laps run past 100 s, at which point a -100
    sentinel starts BEATING real designs — and an optimiser handed that will
    walk towards infeasibility.

    MUTATION: set ``REFUSAL_SCORE["laptime"] = -100.0``. This test fails on
    the slow car and passes on the fast one, which is the point.
    """
    from aerobo import cartrack

    slow = cartrack.CarSpec(power_w=30.0e3)
    prob = cs.SectionProblem(car_spec=slow)
    got = cs.evaluate_section(prob.reference_x(), prob)
    assert got["feasible"], got["reason"]
    assert got["lap_time_s"] > 100.0, got["lap_time_s"]
    assert cs.fg_section(prob.reference_x(), prob)[0] == got["score"]
    # a refusal must still lose to it
    bad = prob.reference_x().copy()
    bad[prob.param_labels.index("slot_gap_frac")] = 0.012
    bad[prob.param_labels.index("slot_overlap_frac")] = 0.06
    refused = cs.evaluate_section(bad, prob)
    assert not refused["feasible"], "the probe design must actually refuse"
    assert cs.f_section(bad, prob) < got["score"]
    assert cs.f_section(bad, prob) == cs.REFUSAL_SCORE["laptime"]


def test_the_problem_declares_what_it_can_score_and_refuses_the_rest():
    with pytest.raises(ValueError, match="unknown objective"):
        cs.SectionProblem(objective="cz")
    p = cs.SectionProblem()
    assert p.n_constraints == 0 and p.constraint_labels == ()
    assert p.dim == 2 * (2 * p.n_cst + 1) + 5
    assert p.param_labels[-1] == "alpha_deg"
    assert cs.ALPHA_BOUNDS_DEG == cm.CarWingMultiProblem.ALPHA_BOUNDS_DEG


def test_out_of_box_and_wrong_shape_are_reasons_not_exceptions():
    p = cs.SectionProblem()
    assert not cs.evaluate_section(np.zeros(3), p)["feasible"]
    x = p.reference_x().copy()
    x[p.param_labels.index("tc_main")] = 0.5
    out = cs.evaluate_section(x, p)
    assert not out["feasible"] and "bounds violation" in out["reason"]
