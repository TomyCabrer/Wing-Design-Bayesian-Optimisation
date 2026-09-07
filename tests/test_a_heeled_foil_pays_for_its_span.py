"""Nothing paid for span, so a searched span was a ratchet. Heel pays for it.

Reported as: "Why not free span for hydrofoil? Not stable for hydrofoil. For
hydrofoil it should be done for wind foiling." — and, in the same breath, "no
free span when also vertical fin selected."

Three findings behind these tests, each asserted as an outcome and not as a
declaration.

1. THE SPAN ROW WAS REFUSED ON THE WRONG FAMILIES, FOR A ROW THAT IS IN
   METRES. ``api.WET_SIZE_KEYS`` said the elevator family states its
   stabiliser's height AND ARM as fractions of the span. The arm is
   ``hydrotail.L_T_BOUNDS = (0.5, 1.5)`` metres and ``tail.arm_row``
   validates it in metres; only the DEPTH band is span-scaled, and only when
   the depth is searched. Half the elevator families fix it — and those are
   exactly the families a wind foiler flies, because ``rig.RigLoads`` needs a
   second surface to trim its couple against. So the one cell that mattered,
   "a rig AND a searched span", was empty for a reason that did not apply to
   it.

2. NOTHING IN THE WATER MODEL READ THE SPAN. No structural weight, and
   ``_mast_cd0`` charges ``2·depth·c_mast`` referenced to the AREA. A
   searched span therefore ran to the top of whatever band it was given and
   stopped on ``sizing.AR_LIMITS``. HEEL is the term that reads it: rolled by
   phi, the windward tip climbs ``(b/2)·sin(phi)`` towards the free surface,
   loses static head in proportion to the span, and eventually leaves the
   water altogether.

3. A SEARCHED SPAN AND A VERTICAL SURFACE ARE NOT OFFERED TOGETHER. The
   water vertical is the MAST, whose size and drag read the depth and the
   area and never the span, so a search given both would spend a freedom
   against a surface blind to it.
"""

import numpy as np
import pytest

from aerobo import api
from aerobo import fin as _fin, hydrofoil, hydrotail, rig

FOIL = "hydrofoil"
FOIL_WL = "hydrofoil + winglet"
FOIL_TAIL = "hydrofoil + elevator"
FOIL_TAIL_FREE = "hydrofoil + elevator [free depth]"


def _built(name, **flags):
    return api.PROBLEM_SPECS[name].build({}, dict(flags), None)


def _centre(name, **flags):
    built = _built(name, **flags)
    return built, np.asarray(built.bounds, dtype=float).mean(axis=1)


# ------------------------------------------- 1. off, nothing moved anywhere

@pytest.mark.parametrize("name,expect", [(FOIL, 27.9242903),
                                         (FOIL_WL, None),
                                         (FOIL_TAIL, None)])
def test_a_water_run_with_no_heel_is_the_published_one(name, expect):
    """Zero heel is not a special case in the code — ``sin 0 = 0`` and
    ``cos 0 = 1`` — and this is the assertion that keeps it that way."""
    built, x = _centre(name)
    out = built.evaluate(x)
    assert out["feasible"], out.get("reason")
    assert out["heel_deg"] == 0.0
    # the margin vector is exactly as wide as it was: no immersion margin is
    # reported as a constraint unless a clearance is stated
    assert built.problem.n_constraints == np.atleast_1d(out["g"]).size
    assert out["g_immersion"] is None
    if expect is not None:
        assert out["LoD"] == pytest.approx(expect, rel=1e-7)


# --------------------------------------------- 2. what heel does, in metres

def test_the_rising_tip_climbs_with_the_span():
    """The whole of the geometry, and the whole of why span has a price."""
    for phi in (5.0, 20.0, 45.0):
        for b in (1.2, 2.4):
            z = hydrofoil.heeled_z(np.array([-b / 2.0, b / 2.0]), 0.0, phi)
            assert z[1] == pytest.approx(b / 2.0 * np.sin(np.deg2rad(phi)))
            assert z[0] == pytest.approx(-z[1])       # one tip up, one down
    # ...and at zero it is the flat foil, exactly
    assert hydrofoil.heeled_z(np.array([-0.6, 0.6]), 0.0, 0.0).tolist() \
        == [0.0, 0.0]


def test_a_heeled_foil_has_to_make_more_lift():
    """L cos(phi) carries the weight, so the trim target goes as 1/cos."""
    flat = hydrofoil.HydrofoilProblem()
    over = hydrofoil.HydrofoilProblem(heel_deg=30.0)
    assert over.CL_target(12.0) == pytest.approx(
        flat.CL_target(12.0) / np.cos(np.deg2rad(30.0)))


def test_the_static_head_becomes_a_function_of_the_span_station():
    """Before this the planar family's cavitation margin could not see how
    wide the foil was: one depth, one sigma, and the only spanwise variation
    was ``Cp_min(alpha_eff(y))``, which is scale-free."""
    narrow = _built(FOIL, free_span=True, fin=False, heel_deg=25.0)
    lab = list(narrow.param_labels)
    x = np.asarray(narrow.bounds, dtype=float).mean(axis=1)
    x[lab.index("b_m")] = 1.2
    tight = narrow.evaluate(x)
    x[lab.index("b_m")] = 2.2
    wide = narrow.evaluate(x)
    assert tight["feasible"] and wide["feasible"]
    # the wider foil reaches nearer the surface and pays for it in margin
    assert wide["immersion_m"] < tight["immersion_m"]
    assert wide["g_cav"] < tight["g_cav"]
    # ...and FLAT the same pair of spans has the same margin, to the bit:
    # this is the term that reads the span, and the only one
    flat = _built(FOIL, free_span=True, fin=False)
    x = np.asarray(flat.bounds, dtype=float).mean(axis=1)
    x[lab.index("b_m")] = 1.2
    a = flat.evaluate(x)["immersion_m"]
    x[lab.index("b_m")] = 2.2
    assert flat.evaluate(x)["immersion_m"] == a


def test_a_span_that_lifts_its_tip_out_of_the_water_is_refused():
    """Emergence is a REFUSAL and not a bad score: a partly emerged foil is
    a different problem. ``(b/2) sin(phi) >= depth`` is the whole rule, so
    at 30 deg over 0.30 m of water the boundary is exactly b = 1.2 m."""
    built = _built(FOIL, free_span=True, fin=False, heel_deg=30.0)
    lab = list(built.param_labels)
    x = np.asarray(built.bounds, dtype=float).mean(axis=1)
    x[lab.index("depth_m")] = 0.30
    x[lab.index("b_m")] = 1.0                      # rise 0.25 m < 0.30 m
    assert built.evaluate(x)["feasible"]
    x[lab.index("b_m")] = 1.4                      # rise 0.35 m > 0.30 m
    out = built.evaluate(x)
    assert not out["feasible"]
    assert "emerged" in out["reason"] and "30 deg of heel" in out["reason"]
    # ...and flat, the same 1.4 m span is fine: the refusal is the HEEL's
    flat = _built(FOIL, free_span=True, fin=False)
    assert flat.evaluate(x)["feasible"]


def test_the_span_the_search_can_reach_is_set_by_the_water_not_by_the_band():
    """The finding, stated as a number. Flat, the best draw is the top of
    whatever band was typed; heeled, it is a span — and moving the band no
    longer moves the answer."""
    def best(band, **flags):
        built = _built(FOIL, free_span=band, fin=False, **flags)
        lab = list(built.param_labels)
        x = np.asarray(built.bounds, dtype=float).mean(axis=1)
        x[lab.index("depth_m")] = 0.45
        top = None
        for b in np.linspace(band[0], band[1], 45):
            x[lab.index("b_m")] = b
            out = built.evaluate(x)
            if out["feasible"] and float(np.atleast_1d(out["g"])[0]) >= 0.0:
                top = b
        return top

    # FLAT: the answer is wherever the band stops. Widen it and the answer
    # widens with it, which is the statement "the band IS the design
    # decision" that this family has carried since the row existed.
    assert best((0.8, 2.0)) == pytest.approx(2.0, abs=0.05)
    assert best((0.8, 2.4)) == pytest.approx(2.4, abs=0.05)

    # HEELED: it stops at a span, and widening the band does not move it.
    over = best((0.8, 2.4), heel_deg=25.0)
    assert over < 2.3
    assert over == pytest.approx(best((0.8, 3.0), heel_deg=25.0), abs=0.06)


# ------------------------------------- 3. the immersion margin is a MARGIN

def test_a_stated_clearance_adds_a_margin_of_the_right_width():
    """The width of the margin vector is a harness contract: a scalar on the
    success path against a wider failure path either raises inside the
    optimiser or silently drops a margin (the defect recorded on the draught
    cap, asserted here for its twin)."""
    for name in (FOIL, FOIL_WL, FOIL_TAIL):
        built, x = _centre(name, tip_clearance_m=0.05)
        prob = built.problem
        assert prob.immersion_capped is True
        assert prob.constraint_labels[-1] == \
            hydrofoil.IMMERSION_CONSTRAINT_LABEL
        out = built.evaluate(x)
        assert np.atleast_1d(out["g"]).size == prob.n_constraints
        assert out["g_immersion"] == pytest.approx(
            out["immersion_m"] - 0.05)
        # the FAILURE path is the same width
        bad = np.asarray(built.bounds, dtype=float)[:, 0] - 1.0
        _, g = built.callable(bad)
        assert np.atleast_1d(g).size == prob.n_constraints


def test_a_clearance_the_craft_cannot_keep_is_a_negative_margin_not_a_crash():
    built, x = _centre(FOIL, tip_clearance_m=5.0)
    out = built.evaluate(x)
    assert out["feasible"] and out["g_immersion"] < 0.0


# ---------------------------------- 4. the span row, and where it is offered

@pytest.mark.parametrize("name", [FOIL, FOIL_WL, FOIL_TAIL])
def test_a_searched_span_is_offered_beside_a_vertical_surface(name):
    """THE DEFAULT CONFIGURATION MAY ASK HOW WIDE ITS FOIL SHOULD BE.

    This used to raise. Every water family ships with the mast on and the
    craft flown flat, so the one combination that was refused was the one
    every hydrofoil starts in — and "search the span" was reachable only by
    turning the strut off (a lie about a craft that hangs off it) or by
    typing a heel angle nobody had asked for. The measurement behind the
    refusal is real and is kept; what is gone is the ban.
    """
    built = _built(name, free_span=True)             # fin defaults ON, flat
    assert "b_m" in built.param_labels
    assert _fin.has_fin(built.problem)
    # ...and the AREA row has no such history: the mast's drag is referenced
    # TO the area, so it does move with it
    assert "S_m2" in _built(name, free_area=True).param_labels
    # ...and BOTH rows together, which is the pair a foil designer starts from
    assert {"b_m", "S_m2"} <= set(
        _built(name, free_span=True, free_area=True).param_labels)


@pytest.mark.parametrize("name", [FOIL, FOIL_WL, FOIL_TAIL])
def test_an_unpriced_span_is_flown_and_said_rather_than_refused(name):
    """...and what the user is told instead of "no".

    The note is not decoration: it is the difference between an answer and a
    bound. It fires exactly where the old refusal did — a vertical surface
    on flat water — and goes silent the moment something reads the span.
    """
    flat = _built(name, free_span=True)                   # fin ON, flat
    assert not hydrofoil.span_is_priced(flat.problem)
    assert hydrofoil.span_pricing_note(flat.problem) is not None

    heeled = _built(name, free_span=True, heel_deg=12.0)  # fin still ON
    assert hydrofoil.span_is_priced(heeled.problem)
    assert _fin.has_fin(heeled.problem)
    assert hydrofoil.span_pricing_note(heeled.problem) is None

    # ...a tip clearance is still NOT a price: it tightens the same margin
    # rather than creating it, and at zero heel every station is at one depth
    clr = _built(name, free_span=True, tip_clearance_m=0.05)
    assert hydrofoil.span_pricing_note(clr.problem) is not None

    # ...and a FIXED span is never lectured about a row it does not have
    assert hydrofoil.span_pricing_note(_built(name).problem) is None


def test_every_number_the_caution_quotes_is_still_true():
    """A published number needs a tripwire, and this sentence had none.

    The caution is shown to a user to be BELIEVED — it tells them what their
    answer is worth and what to type to change it — and it existed in two
    hand-written copies that had already drifted apart on both numbers that
    matter: where the climb stops (2.2 m against the true 2.4 m) and when
    heel starts to bind (40 deg against the true ~29 deg). Re-derived here
    rather than pinned, so the sentence cannot outlive the measurement.
    """
    import math

    sentence = hydrofoil.SPAN_IS_UNPRICED_AT_ZERO_HEEL

    def climb(fin: bool):
        flags = {"free_span": (0.6, 4.8)}
        if not fin:
            flags["fin"] = False
        built = _built(FOIL, **flags)
        lab = list(built.param_labels)
        i = lab.index("b_m")
        x = np.asarray(built.bounds, dtype=float).mean(axis=1)
        rows = []
        for b_try in np.arange(0.9, 2.65, 0.1):
            x[i] = b_try
            out = built.evaluate(x)
            rows.append((round(float(b_try), 2), out["feasible"],
                         float(out["score"]), str(out.get("reason", ""))))
        return rows

    on, off = climb(True), climb(False)
    feas_on = [r for r in on if r[1]]
    feas_off = [r for r in off if r[1]]

    # ...the climb is monotone and ends at 2.4 m, not before it
    assert all(a[2] < b[2] for a, b in zip(feas_on, feas_on[1:]))
    assert feas_on[-1][0] == 2.4 and feas_off[-1][0] == 2.4
    assert f"{feas_on[0][2]:.2f} -> {feas_on[-1][2]:.2f}" == "20.02 -> 46.33"
    assert f"20.02 -> 46.33" in sentence
    # ...and taking the mast off makes it STEEPER, not flatter
    assert feas_off[-1][2] > feas_on[-1][2]
    assert f"{feas_off[0][2]:.2f} -> {feas_off[-1][2]:.2f}" == "22.01 -> 58.63"
    assert "22.01 -> 58.63" in sentence

    # ...the next station is refused, and by the ASPECT RATIO
    nxt = [r for r in on if not r[1]][0]
    assert nxt[0] == 2.5 and "aspect ratio 43.4" in nxt[3]
    assert "(2.5 m) is refused at aspect ratio 43.4" in sentence

    # ...and the heel at which the water takes over from the solver's own
    # validity limit, which is where the band top runs out of immersion
    built = _built(FOIL, free_span=(0.6, 4.8))
    depth = float(np.mean(built.problem.DEPTH_BOUNDS))
    crossover = math.degrees(math.asin(2.0 * depth / 2.4))
    assert crossover == pytest.approx(28.63, abs=0.01)
    assert "28.63 deg" in sentence and "about 29 deg" in sentence

    # ...and the shell says all of it by SPLICING this sentence in, not by
    # keeping a second copy of it
    from gui.v3 import session as v3s
    assert sentence in v3s.unpriced_span_why("span")
    assert sentence in v3s.unpriced_span_why("both")


@pytest.mark.parametrize("name", [FOIL, FOIL_WL, FOIL_TAIL])
def test_the_unpriced_span_answer_is_a_bound_and_the_note_says_so(name):
    """THE OUTCOME the note promises, measured rather than asserted.

    An un-priced span has no interior optimum: nothing on the craft grows
    with it, so the score is monotone in b right up to the point something
    REFUSES the candidate. That is the whole content of the caution, and a
    caution nobody checks is a comment.
    """
    built = _built(name, free_span=True)             # fin ON, flat
    lab = list(built.param_labels)
    i = lab.index("b_m")
    x = np.asarray(built.bounds, dtype=float).mean(axis=1)
    lo, hi = built.bounds[i]

    scored = []
    for b_try in np.linspace(lo, hi, 7):
        x[i] = b_try
        out = built.evaluate(x)
        if out["feasible"]:
            scored.append((float(b_try), float(out["score"])))
    assert len(scored) >= 4
    # monotone increasing over every feasible station: no interior optimum
    assert all(b < a for (_, b), (_, a) in zip(scored, scored[1:])), scored
    # ...so the best span in the band is the widest one that is not refused
    assert scored[-1][0] == pytest.approx(max(b for b, _ in scored))

    # ...and what stops it is a LIMIT. The default band's top is exactly the
    # aspect-ratio ceiling these solvers are honest to, which is why the
    # answer lands there rather than inside.
    from aerobo import sizing
    assert hi ** 2 / built.problem.S == pytest.approx(
        sizing.AR_LIMITS[1] if hasattr(sizing, "AR_LIMITS")
        else built.problem.ar_limits[1])


def test_the_depth_row_changes_units_where_the_span_is_searched():
    """This pair used to be REFUSED, and the reason named the depth row's
    units rather than the pair: its band is a fraction of the span, so in
    metres the same number means two layouts at the two ends of a searched
    span. Asked as a fraction it means one, so the pair is allowed and the
    ROW SWAPS — which is the part worth asserting, because a box that kept
    the metre row beside the fraction one would be searching neither."""
    free = hydrotail.HydrofoilTailProblem(free_height=True, fin=False,
                                          span_bounds_m=(0.8, 2.0))
    assert "z_t_frac" in free.param_labels
    assert "z_t_m" not in free.param_labels
    lo, hi = free.bounds[list(free.param_labels).index("z_t_frac")]
    assert (lo, hi) == pytest.approx(hydrotail.Z_T_FRAC_BOUNDS)
    # ...and it is the CANDIDATE's span the fraction is of, not the nominal
    x = np.asarray(free.bounds, dtype=float).mean(axis=1)
    lab = list(free.param_labels)
    x[lab.index("z_t_frac")] = 0.20
    for b_try in (0.8, 2.0):
        x[lab.index("b_m")] = b_try
        assert free.unpack(x)["z_t"] == pytest.approx(-0.20 * b_try)
    # with the depth FIXED the row is still in metres, unchanged
    ok = hydrotail.HydrofoilTailProblem(fin=False, span_bounds_m=(0.8, 2.0))
    assert "b_m" in ok.param_labels and "z_t_frac" not in ok.param_labels
    # ...and both families declare the row now
    for name in (FOIL_TAIL_FREE, FOIL_TAIL):
        assert "free_span" in api.PROBLEM_SPECS[name].flags, name


def test_a_metre_depth_band_is_refused_beside_a_searched_span():
    """The one thing the swap cannot take: a band in metres for a row that
    is now a fraction. Refused by name rather than silently ignored."""
    with pytest.raises(ValueError, match="z_t_frac"):
        hydrotail.HydrofoilTailProblem(free_height=True, fin=False,
                                       span_bounds_m=(0.8, 2.0),
                                       z_t_bounds_m=(-0.4, -0.1))


def test_a_wind_foiling_family_can_now_search_its_span():
    """The cell that was empty. A rig needs a second surface to trim its
    couple against, so every rig family is an elevator family — and every
    elevator family used to refuse the span row."""
    rigged = [n for n, s in api.PROBLEM_SPECS.items()
              if set(api.RIG_KEYS) & set(s.flags)]
    assert rigged
    both = [n for n in rigged if "free_span" in api.PROBLEM_SPECS[n].flags]
    assert len(both) == len(rigged)               # every one of them, now
    built = _built(FOIL_TAIL, free_span=True, fin=False,
                   rig_ce_height_m=2.0, rig_side_n=274.0, heel_deg=15.0)
    assert "b_m" in built.param_labels
    assert built.problem.rig is not None


def test_a_searched_span_is_the_span_the_elevator_family_flies():
    """The four hazards session 71 closed on the planar families, closed on
    this one: the trim, the moment reference, the mast's reference area and
    the reported planform all have to be the candidate's."""
    built = _built(FOIL_TAIL, free_span=True, fin=False)
    lab = list(built.param_labels)
    x = np.asarray(built.bounds, dtype=float).mean(axis=1)
    x[lab.index("b_m")] = 2.0
    out = built.evaluate(x)
    assert out["feasible"], out.get("reason")
    assert out["b"] == pytest.approx(2.0)
    assert out["AR"] == pytest.approx(4.0 / built.problem.S)
    # the stabiliser's default separation is a fraction of the SPAN, so it
    # has to have moved with it (Z_T_FRAC_DEFAULT: the craft is FLAT, and
    # the fraction is the grid line rather than the air families' 0.05 b)
    assert out["z_t"] == pytest.approx(-hydrotail.Z_T_FRAC_DEFAULT * 2.0)
    # ...and a wider foil at the same area is less induced drag
    narrow = x.copy(); narrow[lab.index("b_m")] = 1.2
    assert out["LoD"] > built.evaluate(narrow)["LoD"]


def test_the_draught_floor_reads_the_band_and_not_the_nominal_span():
    """``min_draught_m`` is what a draught cap is refused against, and with
    the span searched the shallowest design in the box is the NARROWEST one."""
    prob = hydrotail.HydrofoilTailProblem(fin=False, span_bounds_m=(0.6, 2.4))
    assert prob.min_draught_m == pytest.approx(
        prob.DEPTH_BOUNDS[0] + hydrotail.Z_T_FRAC_DEFAULT * 0.6)


# ------------------------------------------- 5. the rig answers for its own

def test_the_rig_says_what_would_have_to_hold_the_craft_up():
    """``side_n`` used to be recorded and flown by nothing. It is still flown
    by nothing — the solvers are longitudinal — but it is now PRICED."""
    r = rig.RigLoads(z_ce_m=2.0, side_n=274.0)
    assert r.roll_moment_nm(0.5) == pytest.approx(274.0 * 2.5)
    assert r.righting_lever_required_m(weight_n=1030.0, depth_m=0.5) == \
        pytest.approx(274.0 * 2.5 / 1030.0)
    # no side force, no roll couple, and nothing to hold up
    assert rig.RigLoads(z_ce_m=2.0).roll_moment_nm(0.5) == 0.0


# ------------------------------------------------- 6. it reaches the shell

def test_the_shell_asks_the_heel_and_the_answer_reaches_the_solver():
    from gui.v3 import config as v3c, session as v3s
    from gui.v4.app import assemble

    ctx = assemble("water")
    S = ctx.S
    assert v3s.wet_heel_deg(S) == 0.0
    assert "heel_deg" not in v3c.flags(S)            # zero sends nothing

    assert v3s.set_wet_heel_deg(S, 20.0) is True
    v3s.apply_choices(S)
    assert v3c.flags(S)["heel_deg"] == 20.0
    built = api.PROBLEM_SPECS[S["wing"]["problem"]].build(
        {}, v3c.flags(S), None)
    assert built.problem.heel_deg == 20.0
    # ...and a heel this model cannot describe is refused, not stored
    assert v3s.set_wet_heel_deg(S, 90.0) is False
    assert v3s.wet_heel_deg(S) == 20.0
    for view in ("type", "box", "solver"):
        ctx.render("wing", view)


def test_a_designed_section_water_family_hears_the_strut_switch():
    """Found by the span/fin rule needing to read it: the section builder
    declared ``api.STRUT_KEYS`` and never passed them on, so on those
    families the vertical stabiliser switch changed nothing at all."""
    name = "hydrofoil + CST section (XFOIL)"
    assert api.FIN_PRESENCE_KEY in api.PROBLEM_SPECS[name].flags
    assert _built(name).problem.foil.fin is True
    assert _built(name, fin=False).problem.foil.fin is False
