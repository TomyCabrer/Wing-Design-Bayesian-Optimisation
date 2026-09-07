"""Two more surfaces nobody could see, and one dimension nobody could search.

Reported as: "Vertical didn't appear in hydrofoil 3D diagram. No free span
optimisation for hydrofoil." and "vertical stabiliser should appear between
front and rear wing."

Measured, three things:

1. THE HYDROFOIL'S STRUT WAS NOT DRAWN. ``fig_wing3d`` has three branches —
   nonplanar, a listed ``surfaces`` pair, and a plain planar wing — and only
   the first two called ``_fin_traces``. A plain foil and a foil + elevator
   both land in the third, so the same craft had its mast in the winglet
   family's picture and not in its own. The FRONT view was worse: no branch
   drew a vertical at all, in any medium, which is the one projection a
   vertical surface's height actually lives in.

2. THE TANDEM'S FIN STOOD ON THE REAR WING. Sized against the whole stagger
   (``l_t = dx``) its quarter chord landed at x = dx — inside the rear
   wing's own chord footprint — rather than between the two wings.
   ``fin.tandem_fin_station`` is now the one author of that station and all
   four consumers (both engines' drag, the report, the loft) read it.

3. A WATER CRAFT COULD NOT SEARCH ITS OWN SIZE. ``b`` and ``S`` were
   constructor values only: typeable, never searchable. The V3 planform menu
   offered a water family exactly one entry, "fixed span + area (you choose
   the size)" — a select with nothing to select — while span and area are
   the first two questions a foil design asks.

Every test asserts an OUTCOME: a trace in a figure, a station in metres, a
row in the design vector, a solver number that moves.
"""

import numpy as np
import pytest

from aerobo import api, cad, fin as _fin, hydrotail
from aerobo.flightmodel import ControlsSpec, build_flight_model
from gui import nice_app as v1

FOIL = "hydrofoil"
FOIL_TAIL = "hydrofoil + elevator"
FOIL_WL = "hydrofoil + winglet"
TANDEM = "tandem"
PAIR_VLM = "tandem (nonplanar) + winglets"


def _built(name, **flags):
    return api.PROBLEM_SPECS[name].build({}, dict(flags), None)


def _centre(name, **flags):
    built = _built(name, **flags)
    return built, np.asarray(built.bounds, dtype=float).mean(axis=1)


def _report(name, **flags):
    built, x = _centre(name, **flags)
    cfg = api.RunConfig(problem_name=name, budget=4, seed=0, flags=dict(flags))
    return api.design_report(cfg, x), built, x


def _names(fig):
    return [getattr(t, "name", None) or t.type for t in (fig.data if fig
                                                         else [])]


# ------------------------------------------ 1. the vertical is on the screen

@pytest.mark.parametrize("name", [FOIL, FOIL_TAIL, FOIL_WL, TANDEM, "tail"])
def test_every_reported_vertical_reaches_the_three_dimensional_view(name):
    """Drawn, exported and flown are the same surface, or the picture lies."""
    rep, built, x = _report(name)
    geom = rep["geometry"]
    assert geom.get("fin"), "this family reports no vertical to draw"

    fig = v1.fig_wing3d(geom, x, built.param_labels)
    assert "fin" in _names(fig), "the vertical is reported but not drawn"
    # ...the SAME loft the STL and the VSP script use, not a second one
    lofted = [s for s in cad.surfaces(geom, x, built.param_labels)
              if s.name == "fin"]
    assert lofted, "the export lost the surface the view draws"
    drawn = [t for t in fig.data if getattr(t, "name", None) == "fin"][0]
    assert float(np.max(drawn.z)) == pytest.approx(
        float(np.max(lofted[0].Z)), rel=1e-9)


@pytest.mark.parametrize("name", [FOIL, FOIL_TAIL, TANDEM, "tail"])
def test_the_front_view_is_where_a_height_shows_and_it_shows_it(name):
    """The projection a vertical surface exists in had no line for one."""
    rep, _built_, _x = _report(name)
    geom = rep["geometry"]
    fig = v1.fig_frontview(geom)
    assert fig is not None, "no front view at all"
    blk = geom["fin"]
    want = "strut" if blk.get("kind") == "mast" else "fin"
    assert want in _names(fig)
    line = [t for t in fig.data if getattr(t, "name", None) == want][0]
    # it spans exactly the surface's own height, from its own root
    assert float(line.y[0]) == pytest.approx(blk["z_root_m"])
    assert float(line.y[1]) == pytest.approx(blk["z_root_m"]
                                             + blk["height_m"])


@pytest.mark.parametrize("name", [FOIL, FOIL_WL, TANDEM, PAIR_VLM, "tail"])
def test_the_plan_view_shows_where_the_vertical_STANDS(name):
    """The view a STATION is read in, and it had no line for the one surface
    whose station was in question."""
    rep, built, x = _report(name)
    geom = rep["geometry"]
    fig = v1.fig_planform(geom, x, built.param_labels)
    edge = [t for t in fig.data if "edge-on" in str(getattr(t, "name", ""))]
    assert len(edge) == 1, [getattr(t, "name", None) for t in fig.data]
    blk = geom["fin"]
    # drawn edge-on at its own chord, from its own leading edge
    assert float(min(edge[0].y)) == pytest.approx(blk["x_le_m"])
    assert float(max(edge[0].y) - min(edge[0].y)) == pytest.approx(
        blk["chord_m"])


def test_the_pairs_fin_reads_as_standing_on_the_rear_wing_in_plan():
    """The placement, as an outcome. A pair's fin is MOUNTED, and the only
    structure a pair has at that station is the rear wing — so the picture
    must show it on that surface rather than floating ahead of it."""
    rep, built, x = _report(TANDEM)
    fig = v1.fig_planform(rep["geometry"], x, built.param_labels)
    by = {getattr(t, "name", None): t for t in fig.data}
    fin = by["fin (edge-on)"]
    front, rear = by["front"], by["rear"]
    assert float(min(fin.y)) > float(np.max(front.y)), \
        "the fin overlaps the front wing"
    # ON the rear wing: the fin's edge-on trace lies within that surface's
    # own chord footprint, which is what being bolted to it looks like in
    # plan. Clear of it would mean carried by a boom nothing draws.
    assert float(np.min(rear.y)) <= float(np.max(fin.y)) \
        <= float(np.max(rear.y)), \
        "the fin does not stand on the rear wing"


def test_a_boats_vertical_is_never_captioned_as_a_fin():
    """The report calls it a mast; the picture must not call it a tail fin."""
    rep, built, x = _report(FOIL)
    title = v1.fig_wing3d(rep["geometry"], x, built.param_labels).layout.title
    assert "strut" in str(title.text)
    assert "fin" not in str(title.text)


def test_a_design_with_no_vertical_draws_none():
    """The switch that deletes the surface deletes it from the view too."""
    rep, built, x = _report(FOIL, fin=False)
    assert rep["geometry"]["fin"] is None
    fig = v1.fig_wing3d(rep["geometry"], x, built.param_labels)
    assert "fin" not in _names(fig)
    assert "strut" not in str(fig.layout.title.text)


# ------------------------------- 2. the pair's fin stands between its wings

@pytest.mark.parametrize("name", [TANDEM, PAIR_VLM])
def test_the_pairs_fin_stands_on_the_rear_wing(name):
    """ON the rear wing, which is the only structure a pair has to mount a
    vertical on. At the midpoint the surface is carried by a boom the design
    neither draws, weighs nor charges for."""
    rep, built, x = _report(name)
    geom = rep["geometry"]
    blk = geom["fin"]
    ev = built.evaluate(x)
    dx = float(ev["dx"])
    assert blk["x_qc_m"] == pytest.approx(dx), \
        "the fin is not at the rear wing's station"
    assert blk["x_qc_m"] == pytest.approx(_fin.tandem_fin_station(dx)[0])
    # ...and in z as well: a fin mounted on a wing is at that wing's HEIGHT,
    # not merely at its x with the foot floating.
    assert blk["z_root_m"] == pytest.approx(
        _fin.tandem_fin_station(dx, float(ev["dz"]))[1])
    # and the LOFT is there too, INSIDE the rear wing's chord footprint
    lofted = {s.name: s for s in cad.surfaces(geom, x, built.param_labels)}
    rear = lofted.get("rear")
    assert rear is not None
    assert float(np.min(rear.X)) <= float(np.max(lofted["fin"].X)) \
        <= float(np.max(rear.X)), "the fin does not sit on the rear wing"


def test_the_pair_is_charged_for_the_fin_at_the_arm_it_stands_at():
    """One station, four consumers. A fin drawn at one arm and charged at
    another is two surfaces, and the volume-coefficient law makes them
    differ by the ratio of the arms."""
    built, x = _centre(TANDEM)
    bd = built.evaluate(x)
    blk = (_report(TANDEM)[0]["geometry"])["fin"]
    arm, z_root = _fin.tandem_fin_station(bd["dx"], bd["dz"])
    assert blk["l_t_m"] == pytest.approx(arm)
    assert blk["z_root_m"] == pytest.approx(z_root)
    # S_vt = V_v*b*S/l_t: the reported area IS the one the arm implies
    assert blk["S"] == pytest.approx(
        blk["V_v"] * bd["b"] * bd["Sref"] / arm)
    # ...and the charge is THAT surface's. Standing the fin at the midpoint
    # instead halves the arm, which DOUBLES the area the volume coefficient
    # asks for and close to doubles the parasite drag with it — measured
    # 0.0009034 on the rear wing against 0.0016952 at 0.5.
    mid = _fin_cd0_at(0.5)
    assert mid > 1.5 * bd["cd0_fin"], \
        "a fin at half the arm has twice the area and cannot cost the same"


def _fin_cd0_at(frac: float) -> float:
    """``cd0_fin`` with the fin standing at ``frac`` of the stagger."""
    before = _fin.TANDEM_FIN_X_FRAC
    try:
        _fin.TANDEM_FIN_X_FRAC = frac
        built, x = _centre(TANDEM)
        return float(built.evaluate(x)["cd0_fin"])
    finally:
        _fin.TANDEM_FIN_X_FRAC = before


def test_the_shell_card_quotes_the_fin_at_the_same_station():
    """Stage 3's card reads the pair's own arm, not the whole stagger."""
    from gui.v3 import session as v3s
    from gui.v4.app import assemble

    ctx = assemble("air")
    ctx.S["wing"]["choices"].update(system="tandem")
    v3s.apply_choices(ctx.S)
    card = v3s.surface_geometry(ctx.S, "fin")
    blk = (_report(TANDEM)[0]["geometry"])["fin"]
    assert card is not None
    assert card["area"] == pytest.approx(blk["S"], rel=1e-9)


# ---------------------------------------- 3. a water craft searches its size

def test_an_untouched_water_run_has_the_published_vector_and_score():
    """The size is a ROW THE USER OPENS. Absent, nothing moved."""
    built, x = _centre(FOIL)
    assert built.param_labels == ("taper", "twist_root_deg", "twist_tip_deg",
                                  "tc", "depth_m", "V_ms")
    assert built.evaluate(x)["LoD"] == pytest.approx(27.9242903, rel=1e-7)


@pytest.mark.parametrize("name", [FOIL, FOIL_WL])
@pytest.mark.parametrize("flag,label", [("free_span", "b_m"),
                                        ("free_area", "S_m2")])
def test_asking_for_a_size_row_puts_it_in_the_design_vector(name, flag, label):
    # ...with the strut at its DEFAULT (on), which is where this used to have
    # to switch it off to dodge a refusal that no longer exists. The whole
    # point of the row is that a foiling craft — which hangs off its mast —
    # can ask how wide its foil should be.
    built = _built(name, **{flag: True})
    assert label in built.param_labels
    assert built.dim == _built(name).dim + 1
    # the row is a band around the family's own size, not around a wing's
    lo, hi = built.bounds[list(built.param_labels).index(label)]
    nominal = {"b_m": built.problem.b, "S_m2": built.problem.S}[label]
    assert lo < nominal < hi


def test_a_searched_span_is_the_span_that_is_flown():
    """Not merely accepted: the whole solve has to move with it — the image
    system, the trim CL, the report, the loft and the rebuild."""
    flags = {"free_span": True, "fin": False}
    built = _built(FOIL, **flags)
    lab = list(built.param_labels)
    x = np.asarray(built.bounds, dtype=float).mean(axis=1)
    x[lab.index("b_m")] = 2.0
    out = built.evaluate(x)
    assert out["feasible"], out.get("reason")
    assert out["b"] == pytest.approx(2.0)

    cfg = api.RunConfig(problem_name=FOIL, budget=4, seed=0, flags=flags)
    geom = api.design_report(cfg, x)["geometry"]
    assert geom["b"] == pytest.approx(2.0)
    assert 2.0 * max(geom["y"]) == pytest.approx(2.0, rel=1e-3)
    wing = [s for s in cad.surfaces(geom, x, built.param_labels)
            if s.name == "wing"][0]
    assert 2.0 * float(np.max(wing.Y)) == pytest.approx(2.0, rel=1e-3)
    # a wider foil at the same area is less induced drag — the answer moves
    narrow = x.copy(); narrow[lab.index("b_m")] = 1.2
    assert out["LoD"] > built.evaluate(narrow)["LoD"]


def test_the_image_operator_cache_is_keyed_on_the_span_too():
    """It was keyed on depth alone, which is correct exactly while the span
    is a constant. The second candidate of a searched span would otherwise
    have flown the first one's image system."""
    from aerobo.hydrofoil import HydrofoilProblem

    prob = HydrofoilProblem(span_bounds_m=(0.8, 2.0), fin=False)
    a = prob.ops(0.5, 0.8)
    b = prob.ops(0.5, 2.0)
    assert a is not b
    assert prob.ops(0.5, 0.8) is a


def test_a_searched_area_is_the_loading_and_moves_the_trim():
    from aerobo.hydrofoil import HydrofoilProblem

    prob = HydrofoilProblem()
    assert prob.CL_target(12.0, 0.288) == pytest.approx(
        0.5 * prob.CL_target(12.0, 0.144))


def test_a_size_the_solvers_cannot_describe_is_refused_per_design():
    """Refused with its reason, not banned at the band — and only where the
    size is SEARCHED, so a published fixed planform cannot start refusing
    itself."""
    built = _built(FOIL, free_span=True, fin=False)
    lab = list(built.param_labels)
    x = np.asarray(built.bounds, dtype=float).mean(axis=1)
    x[lab.index("b_m")] = 0.61              # AR 2.6 on the published area
    out = built.evaluate(x)
    assert not out["feasible"]
    assert "aspect ratio" in out["reason"]
    # the fixed-planform family has no such gate and is unchanged
    fixed, xf = _centre(FOIL)
    assert fixed.evaluate(xf)["feasible"]


def test_the_water_size_is_offered_where_the_rows_exist():
    """Declared exactly where the problem carries the row, and NOT by family.

    The rule this replaces refused BOTH size rows on every elevator family,
    for a reason ("its stabiliser's height and ARM are stated against the
    span") that named a row which is in metres. Its successor kept half of
    that — no searched span beside a searched stabiliser DEPTH, whose band
    is a fraction of the span — and that half is gone too, because what it
    named was the depth row's UNITS and not the pair: with both free the
    depth is asked as a fraction of the candidate's own span. So:

    * BOTH rows are on EVERY water family;
    * and on a family that searches its depth, a searched span REPLACES the
      ``z_t_m`` row with ``z_t_frac`` rather than sitting beside it.

    Asserted as an OUTCOME: the declared row has to reach the built
    problem's design vector, and the swapped one has to swap.
    """
    for name, spec in api.PROBLEM_SPECS.items():
        if spec.medium != "water":
            assert not any(k in spec.flags for k in api.WET_SIZE_KEYS), name
            continue
        for key in api.WET_SIZE_KEYS:
            assert key in spec.flags, (name, key)
        free_depth = "z_t_m" in spec.param_labels
        wanted = {k: True for k in api.WET_SIZE_KEYS}
        # the strut stays ON: the span row is no longer refused beside it
        built = spec.build({}, wanted, None)
        assert "b_m" in built.param_labels, name
        assert "S_m2" in built.param_labels, name
        # the depth row changed units, and did not merely gain a twin
        assert ("z_t_frac" in built.param_labels) is free_depth, name
        assert "z_t_m" not in built.param_labels, name


def test_the_shell_asks_a_water_craft_who_decides_its_size():
    """The menu had one entry and therefore asked nothing.

    It has all four now, on the DEFAULT water session — mast on, flown flat
    — which is the state every hydrofoil starts in. The two span entries
    used to be removed there, so "how wide should this foil be?" was a
    question the shell could not be asked without first turning the strut
    off or typing a heel. The measurement that motivated the removal is
    kept, and is said BESIDE the entry the user takes.
    """
    from gui.v3 import config as v3c, session as v3s
    from gui.v4.app import assemble

    ctx = assemble("water")
    S = ctx.S
    assert v3s.wet_size_available(S) is True
    assert v3s.wet_size_mode(S) == "fixed"
    assert not any(k in v3c.flags(S) for k in api.WET_SIZE_KEYS)

    # the strut is ON by default and the craft is flat, and all four modes
    # are offered anyway
    assert S["wing"]["choices"].get("fin", v3s.FIN_DEFAULT) is not False
    assert set(v3s.wet_size_modes(S)) == set(v3s.WET_SIZE_MODES)
    # ...with nothing to say while the span is not being searched
    assert v3s.wet_size_why(S) is None

    # ...and "both" is honoured rather than clamped to the half of it that
    # used to be offered
    assert v3s.set_wet_size_mode(S, "both") is True
    v3s.apply_choices(S)
    assert v3s.wet_size_mode(S) == "both"
    assert v3c.flags(S)["free_span"] is True
    assert v3c.flags(S)["free_area"] is True
    labels = v3s.searched_labels(S)
    assert "b_m" in labels and "S_m2" in labels
    # ...and NOW there is something to say: the answer will be a bound
    assert "TOP OF YOUR BAND" in v3s.wet_size_why(S)

    # ...a stored "span" alone survives too, and keeps the caution
    assert v3s.set_wet_size_mode(S, "span") is True
    v3s.apply_choices(S)
    assert v3s.wet_size_mode(S) == "span"
    assert v3c.flags(S)["free_span"] is True
    assert "free_area" not in v3c.flags(S)
    assert "b_m" in v3s.searched_labels(S)
    assert "TOP OF YOUR BAND" in v3s.wet_size_why(S)

    # ...and with the vertical taken off, the same rows — AND THE SAME
    # CAUTION, because the mast was never what made the span unpriced. Only
    # heel reads the span, and with the mast gone the ratchet is STEEPER
    # (L/D to the top of the band 58.63 against 46.33). The shell used to AND
    # the strut into this test, which withheld the warning from the one
    # configuration it applies hardest to.
    assert v3s.set_wet_size_mode(S, "both") is True
    S["wing"]["choices"]["fin"] = False
    v3s.apply_choices(S)
    assert set(v3s.wet_size_modes(S)) == set(v3s.WET_SIZE_MODES)
    assert v3s.wet_size_mode(S) == "both"
    assert v3c.flags(S)["free_span"] is True
    assert v3c.flags(S)["free_area"] is True
    labels = v3s.searched_labels(S)
    assert "b_m" in labels and "S_m2" in labels
    assert "TOP OF YOUR BAND" in v3s.wet_size_why(S)
    for view in ("type", "box", "solver"):
        ctx.render("wing", view)


def test_a_band_the_user_drags_survives_the_rebuild():
    """The design box's own row is the band, and it has to come back out of
    ``prob.bounds`` unchanged or the no-solution reach never converges."""
    flags = {"free_span": True, "fin": False}
    built = api.PROBLEM_SPECS[FOIL].build({}, flags, {"b_m": (0.9, 1.6)})
    row = built.bounds[list(built.param_labels).index("b_m")]
    assert tuple(row) == (0.9, 1.6)
    assert tuple(built.problem.span_bounds_m) == (0.9, 1.6)
    again = api.PROBLEM_SPECS[FOIL].build({}, flags, {"b_m": (0.9, 1.6)})
    assert tuple(again.bounds[list(again.param_labels).index("b_m")]) == \
        (0.9, 1.6)


def test_a_collapsed_size_row_is_refused_as_a_pin():
    with pytest.raises(ValueError, match="pin"):
        api.PROBLEM_SPECS[FOIL].build({}, {"free_span": True, "fin": False},
                                      {"b_m": (1.2, 1.2)})


def test_a_typed_heel_changes_what_the_span_answer_is_worth():
    """The heel no longer changes WHICH ENTRIES EXIST — it changes the note.

    A foiling craft hangs off its mast, so "turn the vertical stabiliser
    off" is not an answer a user can honestly give to buy a design freedom;
    and neither is "type a heel you are not sailing at". Both used to be the
    price of reaching the span row. The row is always there now, and heel
    decides whether the answer is an optimum or a bound — asserted through
    the SHELL, because that is where the user meets it, and in both
    directions, because a caution that never clears is noise.
    """
    from gui.v3 import config as v3c, session as v3s
    from gui.v4.app import assemble

    ctx = assemble("water")
    S = ctx.S
    assert S["wing"]["choices"].get("fin", v3s.FIN_DEFAULT) is not False
    assert set(v3s.wet_size_modes(S)) == set(v3s.WET_SIZE_MODES)

    assert v3s.set_wet_size_mode(S, "both") is True
    v3s.apply_choices(S)
    assert "HEEL ANGLE" in v3s.wet_size_why(S)      # flat: the caution

    assert v3s.set_wet_heel_deg(S, 12.0) is True
    v3s.apply_choices(S)
    assert set(v3s.wet_size_modes(S)) == set(v3s.WET_SIZE_MODES)
    assert v3s.wet_size_mode(S) == "both"
    assert "TOP OF YOUR BAND" not in (v3s.wet_size_why(S) or "")
    fl = v3c.flags(S)
    assert fl["free_span"] is True and fl["free_area"] is True
    assert fl["heel_deg"] == 12.0
    # ...and it BUILDS with the strut on, which is the whole point
    built = api.PROBLEM_SPECS[S["wing"]["problem"]].build(None, fl, None)
    assert "b_m" in built.param_labels and "S_m2" in built.param_labels

    # ...back to flat: the rows STAY and the caution comes back
    assert v3s.set_wet_heel_deg(S, 0.0) is True
    v3s.apply_choices(S)
    assert set(v3s.wet_size_modes(S)) == set(v3s.WET_SIZE_MODES)
    assert v3s.wet_size_mode(S) == "both"
    fl = v3c.flags(S)
    assert fl["free_span"] is True and fl["free_area"] is True
    assert "TOP OF YOUR BAND" in v3s.wet_size_why(S)


def test_the_objective_cannot_hide_a_size_row_the_run_still_searches():
    """The box SHOWN must be the box SEARCHED — including under an objective
    that cannot build yet.

    ``gui.v3.config.default_bounds`` builds the problem to read the rows a
    flag ADDS (``FLAG_ADDED_ROWS``), and wrapped that build in a blanket
    ``except`` that fell back to the static box. The composite objective
    refuses to build until its normalisation band is measured
    (``api.wing_score_reference``), so one click on the objective made b_m
    and S_m2 VANISH from the design box — while ``config.flags(S)`` went on
    sending ``free_span``/``free_area`` and the run went on searching both.
    A row the optimiser is riding and the box does not draw is the exact
    defect this file exists to catch, one level further in.

    The fix reads the box off a build with the OBJECTIVE stripped, which is
    sound because an objective is not a dimension: it says what is
    maximised, never which rows exist.
    """
    from gui.v3 import config as v3c, session as v3s
    from gui.v4.app import assemble

    ctx = assemble("water")
    S = ctx.S
    assert v3s.set_wet_size_mode(S, "both") is True
    v3s.apply_choices(S)
    fl = v3c.flags(S)
    assert fl["free_span"] is True and fl["free_area"] is True

    seen = {}
    for objective in ("lod", "composite"):
        v3s.set_wing_objective(S, objective)
        v3s.apply_choices(S)
        # the flags are unchanged: the run searches both rows either way
        fl = v3c.flags(S)
        assert fl["free_span"] is True, objective
        assert fl["free_area"] is True, objective
        # ...so the box has to draw both, and with real bands
        box = v3c.default_bounds(S)
        for row in ("b_m", "S_m2"):
            assert row in box, (objective, row)
            lo, hi = box[row]
            assert hi > lo, (objective, row)
        # ...and the box the shell shows agrees with it
        labels = v3s.searched_labels(S)
        assert "b_m" in labels and "S_m2" in labels, objective
        seen[objective] = (box, labels)

    # ...AND THE TWO BOXES ARE THE SAME BOX. This is the soundness claim the
    # fix rests on — an objective says what is maximised, never which rows
    # exist or how wide they are — so asserting only that the rows are
    # PRESENT would pass even if the stripped build read a different box,
    # which is the one way this fix could be wrong.
    assert seen["lod"][0] == seen["composite"][0]
    assert seen["lod"][1] == seen["composite"][1]


def test_no_water_size_mode_is_told_to_pick_an_air_menu_entry():
    """The size note must never quote a menu entry this card does not have.

    The note branched on whether the AREA happened to be a design-box row,
    so the two water modes that do not free the area — "fixed" and "free
    SPAN" — fell through to the AIR branch, whose closing sentence reads
    "Choose 'area from the mission's W/S, span optimised' above to search
    the span instead". The water menu has never had an entry by that name,
    and a water craft has no wing loading to derive an area from, so the
    sentence was both unfollowable and about machinery this family does not
    carry — printed on the very card a foil designer hunts for a free span.
    Asserted on all four modes, through the rendered stage.
    """
    from nicegui import ui
    from gui.v3 import session as v3s
    from gui.v4.app import assemble

    ctx = assemble("water")
    S = ctx.S

    def rendered() -> str:
        ctx.render("wing", "type")
        got = ctx.views.get(("wing", "type"))
        assert got is not None
        return " ".join((getattr(e, "text", "") or "")
                        for e in got.descendants() if isinstance(e, ui.label))

    for mode in v3s.WET_SIZE_MODES:
        assert v3s.set_wet_size_mode(S, mode) is True
        v3s.apply_choices(S)
        assert v3s.wet_size_mode(S) == mode
        text = rendered()
        assert text, mode
        assert "span optimised" not in text, mode
        assert "W/S" not in text, mode
        # ...and the note still SAYS which rows are searched, per mode
        if mode == "fixed":
            assert "Neither size row is searched" in text
        else:
            assert "design variable" in text, mode

    # ...and the AIR card is untouched: it keeps the sentence that is true
    # there, because an air wing's area really does follow the mission's W/S
    air = assemble("air")
    air.render("wing", "type")
    got = air.views.get(("wing", "type"))
    air_text = " ".join((getattr(e, "text", "") or "")
                        for e in got.descendants() if isinstance(e, ui.label))
    assert "W/S" in air_text


def test_the_water_size_rows_are_drawn_in_the_design_box():
    """The box SHOWN must be the box SEARCHED — on the rows a flag ADDS.

    ``gui.v3.config.FLAG_MOVED_ROWS`` only ever looked for rows the static
    box already had, and a water family's span and area are not in any
    static box: they are stated numbers until a flag makes them variables.
    So "free AREA" chose a search over a seventh design variable the design
    box never drew — no band to read, none to type and none to pin — while
    the card above it still said the size was the user's.
    """
    from gui.v3 import config as v3c, session as v3s
    from gui.v4.app import assemble

    ctx = assemble("water")
    S = ctx.S
    assert "S_m2" not in v3c.effective_bounds(S)
    assert "b_m" not in v3c.effective_bounds(S)

    v3s.set_wet_heel_deg(S, 12.0)
    v3s.set_wet_size_mode(S, "both")
    v3s.apply_choices(S)
    box = v3c.effective_bounds(S)
    built = api.PROBLEM_SPECS[S["wing"]["problem"]].build(
        None, v3c.flags(S), None)
    labels = list(built.param_labels)
    for row in ("b_m", "S_m2"):
        assert row in box, row
        lo, hi = built.bounds[labels.index(row)]
        assert box[row][0] == pytest.approx([lo, hi]), row
    # every row the run searches is drawn, and nothing else is
    assert set(box) == set(labels)


def test_the_depth_row_swaps_units_in_the_design_box_too():
    """...and the swap has to reach the box, not just the solver.

    On a family that searches its stabiliser's depth, freeing the span
    turns ``z_t_m`` into ``z_t_frac``. A box that kept the metre row would
    be drawing a band nothing searches AND hiding the one that is searched
    — both halves of "the box shown is the box searched", in one row.
    """
    from gui.v3 import config as v3c, session as v3s
    from gui.nice_app import derive_problem

    S = v3s.make_session()
    S["wing"]["choices"]["medium"] = "water"
    name, _ = derive_problem(S["wing"]["choices"])
    S["wing"]["problem"] = name
    assert "z_t_m" in api.PROBLEM_SPECS[name].param_labels   # searches it
    assert "z_t_m" in v3c.effective_bounds(S)

    v3s.set_wet_heel_deg(S, 12.0)
    assert v3s.set_wet_size_mode(S, "both") is True
    v3s.apply_choices(S)
    box = v3c.effective_bounds(S)
    assert "z_t_m" not in box
    assert box["z_t_frac"][0] == pytest.approx(list(hydrotail.Z_T_FRAC_BOUNDS))
    built = api.PROBLEM_SPECS[name].build(None, v3c.flags(S), None)
    assert set(box) == set(built.param_labels)
    # ...and the shell says why the row it knew changed name
    assert "z_t_frac" in v3s.wet_size_why(S)


def _in_view(ctx, stage, view, kind):
    """Elements of one TYPE inside ONE view.

    Scoped to the view rather than to the client, because these tests share
    a client and a global scan is order-dependent (v3-ui-tests-share-one-client).
    """
    from nicegui import ui

    got = ctx.views.get((stage, view))
    return [] if got is None else [e for e in got.descendants()
                                   if isinstance(e, kind)]


def _field(ctx, stage, view, label):
    """The ui.number sitting in the row whose label starts with ``label``."""
    from nicegui import ui

    for row in _in_view(ctx, stage, view, ui.row):
        kids = list(row.descendants())
        if not any(isinstance(k, ui.label)
                   and (getattr(k, "text", "") or "").strip().startswith(label)
                   for k in kids):
            continue
        for k in kids:
            if isinstance(k, ui.number):
                return k
    return None


def _select(ctx, stage, view):
    """The water SIZE select — the one whose options are WET_SIZE_MODES."""
    from nicegui import ui

    from gui.v3 import session as v3s

    for sel in _in_view(ctx, stage, view, ui.select):
        if set(sel.options or {}) <= set(v3s.WET_SIZE_MODES) and sel.options:
            return sel
    return None


def test_typing_a_heel_neither_destroys_its_own_field_nor_a_stated_size():
    """The heel field re-options the size menu; it must do NOTHING else.

    ``ui.select.set_options`` assigns ``value`` before anything else, so a
    display write fires the select's own ``on_change``. Left unguarded that
    handler re-renders the type card — destroying the field the heel is
    being typed into, so "12" becomes "1" — and STORES the clamped mode,
    turning ``wet_size_mode``'s deliberately read-time clamp into a
    destructive edit: a stated "free SPAN" would not come back when the heel
    did. Both halves asserted here, through the rendered stage.

    Answering the SELECT may rebuild the card — that is a click, not a typed
    number — so the handles are re-resolved after it and not before.
    """
    from gui.v3 import session as v3s
    from gui.v4.app import assemble

    ctx = assemble("water")
    S = ctx.S
    ctx.render("wing", "type")

    heel = _field(ctx, "wing", "type", "heel")
    assert heel is not None
    heel.set_value(5.0)
    assert _field(ctx, "wing", "type", "heel") is heel      # not rebuilt
    size = _select(ctx, "wing", "type")
    assert size is not None
    assert set(size.options) == set(v3s.WET_SIZE_MODES)

    size.set_value("span")                                  # a click may rebuild
    assert v3s.wet_size_mode(S) == "span"
    ctx.render("wing", "type")
    heel = _field(ctx, "wing", "type", "heel")
    size = _select(ctx, "wing", "type")
    assert heel is not None and size is not None

    # THE CAUTION CLEARS WHEN THE USER DOES WHAT IT SAYS, asserted on the
    # RENDERED label and not on session.wet_size_why: the sentence's whole
    # remedy is "TYPE A HEEL ANGLE IN THE FIELD BELOW", and it was drawn once
    # by _render_type while _set_wet_heel deliberately never re-renders — so
    # it stayed on screen verbatim after the heel was typed. A caution that
    # never clears is noise, which is what this test's name has always
    # claimed to check.
    def shown() -> str:
        from nicegui import ui
        got = ctx.views.get(("wing", "type"))
        return " ".join((getattr(e, "text", "") or "")
                        for e in got.descendants()
                        if isinstance(e, ui.label)
                        and getattr(e, "visible", True))

    heel.set_value(0.0)
    assert "TOP OF YOUR BAND" in shown()
    heel.set_value(12.0)
    assert "TOP OF YOUR BAND" not in shown()
    assert _field(ctx, "wing", "type", "heel") is heel   # still not rebuilt

    # ...back to flat. Neither the entries NOR the answer go — the menu no
    # longer shrinks at all — and the field the number was typed into is the
    # same object it was. What flat changes is the note, not the offer.
    heel.set_value(0.0)
    assert "TOP OF YOUR BAND" in shown()
    assert _field(ctx, "wing", "type", "heel") is heel
    assert set(size.options) == set(v3s.WET_SIZE_MODES)
    assert S["wing"]["choices"]["wet_size"] == "span"       # not overwritten
    assert v3s.wet_size_mode(S) == "span"                   # and not clamped
    assert _v3config().flags(S)["free_span"] is True
    assert "TOP OF YOUR BAND" in v3s.wet_size_why(S)

    heel.set_value(5.0)                                     # priced again
    assert _field(ctx, "wing", "type", "heel") is heel
    assert set(size.options) == set(v3s.WET_SIZE_MODES)
    assert v3s.wet_size_mode(S) == "span"
    assert _v3config().flags(S)["free_span"] is True
    assert "TOP OF YOUR BAND" not in (v3s.wet_size_why(S) or "")


def _v3config():
    from gui.v3 import config as v3c
    return v3c
