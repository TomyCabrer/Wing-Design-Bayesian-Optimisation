"""The STRUT — placed between the wings, charged as a surface, and loaded.

WHY THIS FILE EXISTS
--------------------
A windfoil is mostly mast. Until session 68 this engine's strut was a drag
term with no position and no lift:

* ``hydrofoil._mast_cd0`` charged two sides of ``depth x c_mast`` at a
  CONSTANT flat-plate Cf — no Reynolds number, no form factor, no junction;
* ``fin.mast`` reported its quarter chord AT THE FOIL (x = 0), so the only
  vertical surface a foiling craft has stood at the reference point every
  arm is measured from, and the flight rebuild's yaw stiffness was the
  stiffness of a fin with no lever;
* ``fin_tc`` — a strut SECTION the shell asks for on stage 2.7 — reached no
  number anywhere in the package;
* and ``rig.RigLoads.side_n``, 274 N measured on a windfoil and 615 N on a
  kitefoil, was a stated load that NOTHING in the craft resisted. FoilingBO
  put that on its own card as the headline gap and measured the resulting
  under-charge against published strut data at 1.85-2.35x.

``hydrofoil.strut_report`` and ``hydrotail``'s station row are those four,
and this file is their contract. It is arranged so that each test names the
thing that used to be silently absent, because "absent" is what every one of
these defects looked like from the outside: a flag accepted, a surface
drawn, a number reported, and no arithmetic behind any of it.

THE PUBLISHED CHARGE IS STILL REACHABLE. ``strut_model=False`` is the switch
back, and it is exercised here rather than merely offered: every recorded
water number was scored on it, and a study that reproduces one has to be
able to ask for it.
"""

import numpy as np
import pytest

from aerobo import api, flightmodel as fm, hydrofoil, hydrotail
from aerobo.rig import RigLoads

FAMILY = "hydrofoil + elevator"

#: the measured windfoil rig: drake_blog's 145 N drive, zhang2025's 2.0 m
#: centre of effort, and the side force the mast has to carry.
WINDFOIL = {"z_ce_m": 2.0, "thrust_n": 145.0, "side_n": 274.0}


def _x(prob=None):
    prob = prob or hydrotail.HydrofoilTailProblem()
    return np.asarray(prob.bounds, dtype=float).mean(axis=1)


def _ev(prob, x=None):
    return hydrotail.evaluate_hydrofoil_tail(_x() if x is None else x, prob)


def _report(flags: dict, x=None):
    built = api.PROBLEM_SPECS[FAMILY].build({}, dict(flags), None)
    xx = np.asarray(built.bounds, dtype=float).mean(axis=1) if x is None else x
    return api.design_report(
        api.RunConfig(problem_name=FAMILY, flags=dict(flags)), xx)


# ------------------------------------------------- (a) it has a station now

def test_the_strut_stands_between_the_two_wings():
    """Not at the front wing, which is where it used to be reported.

    ``X_MAST_FRAC`` is the one measured layout this repo carries — FoilingBO's
    windfoil, whose front wing hangs 0.06 m forward of the mast foot on a
    0.95 m fuselage — so the assertion is that the station is strictly
    between the two surfaces and strictly forward of the stabiliser, not that
    it equals any particular decimal.
    """
    out = _ev(hydrotail.HydrofoilTailProblem())
    assert 0.0 < out["x_mast"] < out["l_t"], \
        "the mast is bolted to the fuselage, between the two wings"
    assert out["x_mast_frac"] == pytest.approx(
        out["x_mast"] / out["l_t"], rel=1e-12)


def test_the_station_follows_the_arm_because_it_is_a_fraction_of_it():
    """A searched arm moves the fuselage; the mast is bolted to it.

    Asked in metres the station would mean a different layout at each end of
    the arm's band — the same reason ``Z_T_FRAC_LABEL`` exists.
    """
    prob = hydrotail.HydrofoilTailProblem()
    lab = list(prob.param_labels)
    short, long = _x(prob).copy(), _x(prob).copy()
    short[lab.index("l_t_m")] = 0.6
    long[lab.index("l_t_m")] = 1.4
    a, b = _ev(prob, short), _ev(prob, long)
    assert a["x_mast"] < b["x_mast"]
    assert a["x_mast_frac"] == pytest.approx(b["x_mast_frac"], rel=1e-12)


def test_the_station_can_be_searched_and_only_then_is_it_a_row():
    """A freedom that is not asked for must not lengthen the design vector."""
    fixed = hydrotail.HydrofoilTailProblem()
    free = hydrotail.HydrofoilTailProblem(free_mast_station=True)
    assert hydrotail.X_MAST_FRAC_LABEL not in fixed.param_labels
    assert hydrotail.X_MAST_FRAC_LABEL in free.param_labels
    assert free.dim == fixed.dim + 1
    row = free.bounds[list(free.param_labels).index(
        hydrotail.X_MAST_FRAC_LABEL)]
    assert tuple(row) == pytest.approx(hydrotail.X_MAST_FRAC_BOUNDS)
    # ...and the row is the same question, so a design drawn at the default
    # fraction reproduces the stated problem exactly
    x_free = np.concatenate([_x(fixed), [fixed.unpack(_x(fixed))["x_mast_frac"]]])
    assert _ev(free, x_free)["LoD"] == pytest.approx(
        _ev(fixed)["LoD"], rel=1e-12)


# ------------------------------------------------- (b) the station IS the arm

def test_the_station_reaches_the_fin_block_and_becomes_the_yaw_arm():
    """The one number the flight rebuild reads about a vertical surface.

    ``fin.mast`` used to report ``x_qc = 0`` for every craft, so the rebuild
    flew a strut standing exactly on the station every arm is measured from.
    """
    rep = _report({})
    assert rep["geometry"]["fin"]["kind"] == "mast"
    assert rep["geometry"]["fin"]["x_qc_m"] == pytest.approx(
        rep["breakdown"]["x_mast"], rel=1e-12)
    # ...and a mast is still not a tail: no volume coefficient, and no arm it
    # was SIZED against. The station is not written into either field.
    assert rep["geometry"]["fin"]["V_v"] == 0.0
    assert rep["geometry"]["fin"]["l_t_m"] == 0.0


def test_moving_the_strut_aft_of_the_cg_is_what_turns_yaw_stiffness_positive():
    """THE POINT OF PLACING IT. Cn_beta crosses zero AT the CG, which is what
    a lever does and what a surface standing on the reference point cannot.

    The default windfoil station is FORWARD of the CG, so the strut's own
    yaw moment is destabilising — that is the craft, not a modelling choice:
    the rider stands over the mast and steers it. What the station row buys
    is that the sign is now a design decision instead of a fixed −0.195.
    """
    x_cg = hydrotail.X_CG_FRAC          # the CG, as a fraction of the arm
    seen = {}
    for frac in (hydrotail.X_MAST_FRAC, x_cg, 0.5):
        rep = _report({"x_mast_frac": frac})
        seen[frac] = fm.build_flight_model(rep, V=12.0).deck.Cn_beta
    assert seen[hydrotail.X_MAST_FRAC] < 0.0        # ahead of the CG
    assert seen[0.5] > 0.0                          # aft of it
    assert abs(seen[x_cg]) < abs(seen[hydrotail.X_MAST_FRAC])
    assert abs(seen[x_cg]) < abs(seen[0.5])
    # monotone in the lever, which is the only shape a moment arm can have
    assert seen[hydrotail.X_MAST_FRAC] < seen[x_cg] < seen[0.5]


def test_the_station_is_a_stability_lever_and_not_a_drag_one():
    """Moving the mast along the fuselage changes no drag: its wetted area,
    its Reynolds number and the side force it carries are all unchanged. A
    coupling here would mean the station had leaked into the drag book."""
    a = _report({"x_mast_frac": 0.10, **_rig_flags()})["breakdown"]
    b = _report({"x_mast_frac": 0.60, **_rig_flags()})["breakdown"]
    assert a["LoD"] == pytest.approx(b["LoD"], rel=1e-12)
    assert a["strut"]["CD"] == pytest.approx(b["strut"]["CD"], rel=1e-12)
    assert a["strut"]["arm_m"] != b["strut"]["arm_m"]


def _rig_flags() -> dict:
    return {api.RIG_CE_HEIGHT_KEY: WINDFOIL["z_ce_m"],
            "rig_thrust_n": WINDFOIL["thrust_n"],
            "rig_side_n": WINDFOIL["side_n"]}


# ------------------------------------------- (c) the section reaches a number

def test_the_strut_section_finally_moves_a_number():
    """``fin_tc`` is asked on stage 2.7 and used to reach NOTHING.

    A wetted-area charge has no form factor, so the thickness a user chose
    for their mast changed no published number in the whole package — the
    dataclass said so in a comment. It is a form factor now, so a thicker
    strut costs more drag, monotonically.
    """
    thin, thick = 0.06, 0.15
    a = _ev(hydrotail.HydrofoilTailProblem(fin_tc=thin))
    b = _ev(hydrotail.HydrofoilTailProblem(fin_tc=thick))
    assert b["strut"]["FF"] > a["strut"]["FF"] > 1.0
    assert b["strut"]["cd0"] > a["strut"]["cd0"]
    assert b["LoD"] < a["LoD"]
    # ...and on the PUBLISHED charge it still reaches nothing, which is what
    # makes that switch a faithful reproduction rather than an approximation
    c = _ev(hydrotail.HydrofoilTailProblem(fin_tc=thin, strut_model=False))
    d = _ev(hydrotail.HydrofoilTailProblem(fin_tc=thick, strut_model=False))
    assert c["LoD"] == d["LoD"]


def test_the_strut_chord_is_the_one_the_drawing_uses():
    """``c_mast`` is a flag now, on every water family.

    FoilingBO drew a 0.115 m windfoil mast while the engine charged the
    family default of 0.08 m: a 44 % error straight into the wetted area,
    and — with the model on — into the Reynolds number and the aspect ratio
    as well.
    """
    for name, spec in api.PROBLEM_SPECS.items():
        if spec.medium == "water":
            assert "c_mast" in (spec.flags or ()), name
    wide = _ev(hydrotail.HydrofoilTailProblem(c_mast=0.115))
    narrow = _ev(hydrotail.HydrofoilTailProblem(c_mast=0.08))
    assert wide["strut"]["chord_m"] == 0.115
    assert wide["strut"]["Re"] > narrow["strut"]["Re"]      # its OWN chord
    assert wide["strut"]["AR"] < narrow["strut"]["AR"]      # depth / chord
    assert wide["strut"]["cd0"] > narrow["strut"]["cd0"]    # more wetted area


def test_the_strut_reynolds_number_is_its_own_and_not_the_foils():
    """A 0.08 m mast beside a 0.12 m foil chord is a different Reynolds
    number, and a constant Cf could not express either."""
    out = _ev(hydrotail.HydrofoilTailProblem())
    prob = hydrotail.HydrofoilTailProblem()
    expect = prob.rho * out["V"] * out["strut"]["chord_m"] / prob.mu
    assert out["strut"]["Re"] == pytest.approx(expect, rel=1e-12)
    assert out["strut"]["Re"] != pytest.approx(out["Re_mac"], rel=1e-3)
    assert 0.002 < out["strut"]["Cf"] < 0.008


# --------------------------------------------- (d) the side force is FLOWN

def test_the_mast_carries_the_rigs_side_load():
    """It carried nothing before: the rig stated the load, the report printed
    it, and no surface in the craft resisted it."""
    quiet = _ev(hydrotail.HydrofoilTailProblem(
        rig=RigLoads(z_ce_m=2.0, thrust_n=145.0)))
    loud = _ev(hydrotail.HydrofoilTailProblem(rig=RigLoads(**WINDFOIL)))
    assert quiet["strut"]["Cy"] == 0.0
    assert quiet["strut"]["CDi_side"] == 0.0
    assert loud["strut"]["Cy"] > 0.0
    assert 0.0 < loud["strut"]["leeway_deg"] < 5.0
    assert loud["strut"]["CDi_side"] > 0.0
    assert loud["LoD"] < quiet["LoD"]


def test_the_leeway_drag_is_a_low_speed_term():
    """``Cy`` goes as 1/V^2 at a fixed side force and the induced drag as its
    square, so the term is a rounding error at top speed and the strut's
    largest at take-off. A model that charged wetted area alone had the
    opposite shape — its only speed dependence was Cf."""
    prob = hydrotail.HydrofoilTailProblem(rig=RigLoads(**WINDFOIL))
    lab = list(prob.param_labels)
    got = {}
    for V in (9.0, 12.0, 15.0):
        x = _x(prob).copy()
        x[lab.index("V_ms")] = V
        got[V] = _ev(prob, x)["strut"]
    assert got[9.0]["CDi_side"] > got[12.0]["CDi_side"] > got[15.0]["CDi_side"]
    assert got[9.0]["leeway_deg"] > got[15.0]["leeway_deg"]
    # the fourth-power law, checked against itself rather than assumed
    ratio = got[9.0]["CDi_side"] / got[15.0]["CDi_side"]
    assert ratio == pytest.approx((15.0 / 9.0) ** 4, rel=1e-9)


def test_a_strut_that_cannot_hold_its_side_load_is_refused_not_priced():
    """The straight line the side-force slope is read off has an end.

    Asked for enough side force the linear model will return a thirty-degree
    leeway and the induced drag of a lift the section cannot make — the
    "optimiser hunts solver bugs" failure this repo has already been bitten
    by. A craft that cannot hold its rig's side load is not a craft with
    extra drag; it is sliding sideways, and nothing here models that.
    """
    ok = _ev(hydrotail.HydrofoilTailProblem(rig=RigLoads(**WINDFOIL)))
    assert ok["feasible"] and ok["strut"]["linear"] is True

    absurd = _ev(hydrotail.HydrofoilTailProblem(
        rig=RigLoads(z_ce_m=2.0, thrust_n=145.0, side_n=6000.0)))
    assert not absurd["feasible"]
    assert "leeway" in absurd["reason"]
    assert absurd["strut_leeway_deg"] > hydrofoil.STRUT_LINEAR_DEG
    # a REFUSAL, in the penalty contract — not an exception, and not a clamp
    # that would have priced the design as if it flew
    assert "CD_strut" not in absurd


def test_a_surface_piercing_strut_gets_no_reflection_credit():
    """``AR = depth / chord``, NOT twice it.

    A wing against a rigid wall flies at twice its geometric aspect ratio.
    The top of a mast is the FREE SURFACE, where the boundary condition is
    p = 0 rather than no-flow, so the image carries the opposite sign and
    the reflection buys nothing.
    """
    assert hydrofoil.strut_aspect_ratio(0.6, 0.1) == pytest.approx(6.0)
    # and the side-force slope is the finite-surface one, well below 2 pi
    a = hydrofoil.strut_cl_alpha(hydrofoil.strut_aspect_ratio(0.6, 0.1))
    assert 3.5 < a < 5.5
    assert a < 2.0 * np.pi
    # degenerate geometry answers zero rather than dividing by it
    assert hydrofoil.strut_aspect_ratio(0.0, 0.1) == 0.0
    assert hydrofoil.strut_cl_alpha(0.0) == 0.0


def test_the_whole_strut_charge_is_the_sum_of_its_three_named_terms():
    """A breakdown that did not add up would let one term hide inside
    another — the reason ``cd0_mast`` is still reported beside ``CD_strut``
    rather than replaced by it."""
    out = _ev(hydrotail.HydrofoilTailProblem(rig=RigLoads(**WINDFOIL)))
    s = out["strut"]
    assert s["CD"] == pytest.approx(
        s["cd0"] + s["CD_junction"] + s["CDi_side"], rel=1e-12)
    assert out["cd0_mast"] == s["cd0"]
    assert out["CD_strut"] == s["CD"]
    assert out["CD"] == pytest.approx(
        out["CDi"] + out["CDp"] + out["CD_strut"] + out["CD_junction"],
        rel=1e-12)


def test_the_flat_keys_a_card_can_show_are_the_blocks_own_numbers():
    """``gui/metrics.py`` resolves breakdown keys by NAME and cannot walk
    into a block, so a shell that could only show ``cd0_mast`` would print a
    mast charge that is no longer the mast's total. The flat keys are copied
    from the block, never recomputed."""
    out = _ev(hydrotail.HydrofoilTailProblem(rig=RigLoads(**WINDFOIL)))
    assert out["CDi_strut_side"] == out["strut"]["CDi_side"]
    assert out["strut_leeway_deg"] == out["strut"]["leeway_deg"]
    assert out["CD_strut"] != out["cd0_mast"]      # the point of showing both

    from gui import metrics
    shown = {k for spec in metrics.SPECIALITY["hydrofoil"]
             for k in spec.keys}
    assert {"CD_strut", "CDi_strut_side", "strut_leeway_deg"} <= shown


def test_no_strut_pays_nothing_and_reports_nothing():
    """The three-state contract every vertical surface in this repo has: a
    design that says it has no strut must not be charged for one, and must
    not have a block describing it either."""
    out = _ev(hydrotail.HydrofoilTailProblem(fin=False))
    assert out["cd0_mast"] == 0.0 and out["CD_strut"] == 0.0
    assert "strut" not in out and "x_mast" not in out
    assert _report({"fin": False})["geometry"]["fin"] is None


# ------------------------------------------ (e) the published charge survives

def test_the_published_charge_is_reproducible_bit_for_bit():
    """``strut_model=False`` is the switch back, and it must be EXACT: the
    flat-plate constant, no form factor, no junction, no side force."""
    prob = hydrotail.HydrofoilTailProblem(strut_model=False,
                                          rig=RigLoads(**WINDFOIL))
    out = _ev(prob)
    expect = hydrofoil._mast_cd0(out["depth"], out["S"], prob.c_mast,
                                 prob.cf_mast)
    assert out["cd0_mast"] == expect
    assert out["CD_strut"] == expect
    assert "strut" not in out


def test_the_published_charge_reports_the_published_geometry_too():
    """A run reproducing a recorded number must also reproduce the fin block
    that number was reported with — which stood at x = 0. A station emitted
    anyway would move the rebuilt craft's yaw stiffness while claiming to
    reproduce the run."""
    rep = _report({"strut_model": False})
    assert "x_mast" not in rep["breakdown"]
    assert rep["geometry"]["fin"]["x_qc_m"] == 0.0


def test_the_constant_cf_is_read_on_the_published_path_and_only_there():
    """``cf_mast`` is the published flat-plate constant. With the model on
    the strut reads ``drag.skin_friction_cf`` at its own Reynolds number, so
    the field must move nothing — stated here rather than implied, because a
    field that is silently ignored is this repo's named defect class."""
    kw = dict(rig=RigLoads(**WINDFOIL))
    on = [_ev(hydrotail.HydrofoilTailProblem(cf_mast=c, **kw))["LoD"]
          for c in (0.004, 0.008)]
    assert on[0] == on[1]
    off = [_ev(hydrotail.HydrofoilTailProblem(cf_mast=c, strut_model=False,
                                              **kw))["LoD"]
           for c in (0.004, 0.008)]
    assert off[0] > off[1]


# --------------------------------------------------- (f) the craft is FLAT

def test_an_unstated_craft_is_flat():
    """Wing, strut and stabiliser on one fuselage, which is how a windfoil,
    a wingfoil and a Moth are built. The default separation is the SOLVER's
    line (``tail.DZ_GRID_FRAC``), not the air families' 0.05 b measurement
    height, and it is 12 mm under the shipped foil rather than 60."""
    out = _ev(hydrotail.HydrofoilTailProblem())
    assert out["z_t"] == pytest.approx(
        -hydrotail.Z_T_FRAC_DEFAULT * out["b"], rel=1e-12)
    assert abs(out["z_t"]) < 0.02
    # ...and the searched band opens there too, so the ordinary craft is
    # inside the box rather than below its floor
    free = hydrotail.HydrofoilTailProblem(free_height=True)
    assert max(free.z_t_bounds) == pytest.approx(
        -hydrotail.Z_T_FRAC_DEFAULT * free.b, rel=1e-12)


def test_a_flat_stabiliser_is_still_typeable_all_the_way_to_zero():
    """The default is a DEFAULT, not a floor: an exact zero is reachable and
    is not refused. What it costs is grid convergence, which is why it is not
    the default."""
    flat = _ev(hydrotail.HydrofoilTailProblem(z_t_fixed=0.0))
    assert flat["z_t"] == 0.0
    assert flat["feasible"]


def test_what_the_two_new_defaults_moved():
    """THE TRIPWIRE. Both defaults change every L/D this family reports, so
    the sizes are pinned here — separately, because they are two decisions
    and a single combined number could not say which one drifted."""
    published = _ev(hydrotail.HydrofoilTailProblem(
        strut_model=False, z_t_fixed=-hydrotail.DZ_FRAC * 1.2))["LoD"]
    coplanar = _ev(hydrotail.HydrofoilTailProblem(strut_model=False))["LoD"]
    strut = _ev(hydrotail.HydrofoilTailProblem(
        z_t_fixed=-hydrotail.DZ_FRAC * 1.2))["LoD"]
    both = _ev(hydrotail.HydrofoilTailProblem())["LoD"]
    assert published == pytest.approx(24.6728, abs=5e-4)
    # the flat craft is FASTER (the stabiliser leaves the foil's downwash
    # gradient), and it is worth well under a per cent
    assert 0.004 < coplanar / published - 1.0 < 0.008
    # the strut costs several times more than that, and it is a COST
    assert -0.050 < strut / published - 1.0 < -0.040
    assert both == pytest.approx(23.7027, abs=5e-4)


# ------------------------------------------------------- (g) the refusals

@pytest.mark.parametrize("kw, match", [
    (dict(x_mast_frac=0.3, free_mast_station=True), "state it, or search it"),
    (dict(x_mast_frac=1.4), "outside the fuselage"),
    (dict(x_mast_frac=-0.1), "outside the fuselage"),
    (dict(x_mast_frac_bounds=(0.2, 0.6)), "a row this problem does not"),
    (dict(free_mast_station=True, x_mast_frac_bounds=(0.6, 0.2)),
     "0 <= lo < hi <= 1"),
    (dict(free_mast_station=True, strut_model=False), "nothing reads the"),
])
def test_the_station_refuses_what_it_cannot_honour(kw, match):
    with pytest.raises(ValueError, match=match):
        hydrotail.HydrofoilTailProblem(**kw)


def test_the_station_keys_are_declared_only_where_there_is_a_station():
    """A strut only HAS a station between two surfaces. The single-foil
    families have no ``x_mast_frac`` field, so a flag declared there would be
    accepted by the shell and rejected by the constructor."""
    for name, spec in api.PROBLEM_SPECS.items():
        if spec.medium != "water":
            continue
        prob = spec.build({}, {}, None).problem
        prob = getattr(prob, "foil", prob)
        has_field = hasattr(prob, "x_mast_frac")
        declared = set(api.STRUT_STATION_KEYS) <= set(spec.flags or ())
        assert declared == has_field, name


def test_a_single_foil_keeps_the_published_charge_and_the_foil_station():
    """The family the user did not ask to change. It has no fuselage to
    stand a mast on and no elevator to stand it between."""
    from aerobo.hydrofoil import HydrofoilProblem, evaluate_hydrofoil
    prob = HydrofoilProblem()
    out = evaluate_hydrofoil(np.asarray(prob.bounds).mean(axis=1), prob)
    assert out["cd0_mast"] == hydrofoil._mast_cd0(
        out["depth"], out["S"], prob.c_mast, prob.cf_mast)
    assert "strut" not in out


# ------------------------------------------ (h) the shell asks the question

def _v3(medium: str, tail: bool = True):
    """A V3 session on ``medium``, with the second surface switched on."""
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from gui.v3.app import assemble

    ctx = assemble(medium)
    if tail:
        ctx.act("set_choice", "tail", True)
    return ctx


def _texts(view) -> list[str]:
    out = []
    for e in view.descendants():
        t = getattr(e, "text", "") or ""
        if t:
            out.append(t)
    return out


def test_stage_three_asks_a_water_craft_where_its_strut_stands():
    ctx = _v3("water")
    ctx.render("wing", "type")
    t = " ".join(_texts(ctx.views[("wing", "type")]))
    assert "strut station" in t
    assert "quarter chord" in t and "stabiliser" in t
    # ...and the read-out says which side of the CG that lands on, because
    # that is the sign of the only yaw moment this craft has
    assert "FORWARD of the CG" in t or "AFT of the CG" in t


def test_the_station_is_not_asked_where_there_is_nowhere_to_stand():
    """An aeroplane's vertical is sized by a volume coefficient and stands at
    the tail; a single foil has no fuselage. Neither is asked."""
    air = _v3("air")
    air.render("wing", "type")
    assert "strut station" not in " ".join(
        _texts(air.views[("wing", "type")]))
    solo = _v3("water", tail=False)
    solo.render("wing", "type")
    assert "strut station" not in " ".join(
        _texts(solo.views[("wing", "type")]))


def test_the_shells_answer_reaches_the_problem():
    """Stated and searched both, and NEITHER on an untouched session — which
    is what keeps every stored water run bit-for-bit."""
    from gui.v3 import config, session as v3session
    from gui.nice_app import derive_problem

    S = v3session.make_session("water")
    S["wing"]["choices"]["tail"] = True
    name, _ = derive_problem(S["wing"]["choices"])
    S["wing"]["problem"] = name
    assert not {"x_mast_frac", "free_mast_station"} & set(config.flags(S))

    S["wing"]["choices"]["mast_station_frac"] = 0.45
    assert config.flags(S)["x_mast_frac"] == 0.45
    built = api.PROBLEM_SPECS[name].build({}, config.flags(S), None)
    assert built.problem.x_mast_frac == 0.45

    S["wing"]["choices"]["mast_station"] = "free"
    flags = config.flags(S)
    assert flags["free_mast_station"] is True
    # never BOTH: the problem refuses that pair, and a shell must not be
    # able to build a refusal
    assert "x_mast_frac" not in flags
    built = api.PROBLEM_SPECS[name].build({}, flags, None)
    assert hydrotail.X_MAST_FRAC_LABEL in built.param_labels
