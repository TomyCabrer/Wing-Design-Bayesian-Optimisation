"""The car wing after it adopted its mount (carmount.py) and its mission
(cartrack.py): what the opt-ins change, and what they are forbidden to.

tests/test_carwing.py is the COMPATIBILITY CONTRACT and is not touched. This
file holds the other half of it — that the published family is bit-for-bit
untouched by machinery it never asked for — and then gates the new behaviour
against closed forms and independent hand computations rather than against the
implementation restating itself.

The gates that would catch a mutation, listed so they can be checked:

* the published breakdown is INDEPENDENT of every new field's own knob (the
  sensitivity step, the windup tolerance, the track-point count), asserted
  with ``==`` on the whole shared breakdown;
* the two published mount layouts both take the load out through the PLATES,
  so the only question left between them is WHERE ALONG THE SPAN they grip:
  the tip layout charges exactly nothing and is bit-for-bit what its MountSpec
  reproduces, and the inboard one buys a beam several times stiffer for two
  sheets of wetted area and the two corners they make — every one of those
  differences asserted with its direction, and the sheet charge re-derived
  from ``_strut_cd0`` at the LOCAL chord where the sheets grip;
* the deck, and the pylon that has to reach it, survive only as a hand-built
  :class:`carmount.MountSpec` — no registered layout has a pylon — so the
  reach margin is asserted where it still exists and its absence is asserted
  where the plates carry;
* the windup fixed point against the geometric sum of its own feedback series,
  computed by hand off the RIGID solve through the public carmount API (a
  fixed point that converged to the wrong place passes every "it converged"
  test and dies here);
* dCZ/dh against its own order of accuracy (halving the step must divide the
  change by four) and against the trend it is a statement about — and against
  its COST, by counting the VLM builds an evaluation performs: one on the
  published path, three when the diagnostic is asked for;
* the multi-point lap against the single-point one where the coefficients do
  not move with speed (they agree to round-off, 9.2e-14 s in 61.3 s) and a
  MEASURED difference eleven orders of magnitude larger where they do;
* the OFF invariants with ``==``: zero windup at the quarter chord, an exactly
  1.0 knockdown at zero suction loss, an exactly unchanged ride height under a
  rigid car.
"""

import numpy as np
import pytest

from aerobo import carmount, cartrack, geometry
from aerobo.carwing import (
    CAR_OBJECTIVES,
    G_FAIL,
    PENALTY,
    CarWingProblem,
    _strut_cd0,
    _twist_projection,
    bending_moment,
    compare_mounts,
    deflection_index,
    evaluate_car_wing,
    fg_car_wing,
)

B = 1.6
RIDE = 0.30
DECK = 0.25


def _x(taper=0.8, tw_root=0.0, tw_tip=-2.0, alpha=8.0, ep=0.12, ride=RIDE,
       b=B):
    return np.array([taper, tw_root, tw_tip, alpha, ep, ride, b])


#: the published keys a compatibility comparison must agree on. Everything the
#: family reported before the opt-ins existed; the new keys are additive.
PUBLISHED_KEYS = (
    "score", "CZ", "objective", "objective_label", "g", "g_drag",
    "g_deflection", "g_drag_force", "g_downforce", "constraint_labels",
    "CL_model", "CD", "CDi", "CDp", "cd0_struts", "CD_junction", "efficiency",
    "e", "AR", "alpha_deg", "ride_height_m", "ride_height_over_b", "b_m",
    "S_m2", "area_free", "endplate_h_m", "endplate_h_frac", "mount",
    "mount_label", "supports", "frame", "M_max_Nm", "M_root_Nm",
    "deflection_m", "downforce_N", "drag_N", "q_Pa", "Re_mac", "polar",
    "y", "load_Npm", "moment_Nm",
)


#: the same list without the mount's own NAME. A layout stated as a MountSpec
#: reports the carmount vocabulary ("endplate") where the flag reports its own
#: key ("tips"), which is the description changing and not the answer.
UNNAMED_KEYS = tuple(k for k in PUBLISHED_KEYS
                     if k not in ("mount", "mount_label"))


def _same(a: dict, bb: dict, keys=PUBLISHED_KEYS):
    """Every listed key equal, with ``==`` and never a tolerance."""
    for k in keys:
        va, vb = a[k], bb[k]
        if isinstance(va, np.ndarray):
            assert np.array_equal(va, vb), k
        elif isinstance(va, (list, tuple)):
            assert list(va) == list(vb), k
        else:
            assert va == vb, (k, va, vb)


@pytest.fixture(scope="module")
def base():
    out = evaluate_car_wing(_x(), CarWingProblem())
    assert out["feasible"], out["reason"]
    return out


# =====================================================================
# 1. the compatibility contract: the published family did not move
# =====================================================================

def test_the_published_defaults_are_all_off():
    p = CarWingProblem()
    assert p.mount_spec is None and p.deck_height_m is None
    assert p.car_spec is None and p.track_spec is None
    assert p.flown_reynolds is False
    assert p.downforce_min_v_ms is None and p.drag_budget_v_ms is None
    assert p.dczdh_max_per_m is None
    assert p.dczdh_report is False and p.dczdh_required is False
    assert p.constraint_labels == ("drag budget margin", "deflection margin")
    assert p.n_constraints == 2 and p.dim == 7
    # the SOLVER is still the general one: it fixes its area and scores the
    # coefficient unless a caller asks otherwise. Only the registry always
    # frees the area, and this class is what carsection and compare_split fly
    assert p.area_free is False and p.objective == "cz"
    assert p.CD_budget is not None
    # the published mount lookup still answers, under its own name — and both
    # of its answers are now plate-borne, so there is no pylon in this family
    # and nothing that could have a deck to reach
    assert p.mount == "tips"
    assert p.mount_layout["supports"] == "ends"
    assert p.n_pylons == 0 and CarWingProblem(mount="inboard").n_pylons == 0
    assert p.deck_m is None and p.reach_required is False
    assert CarWingProblem(deck_height_m=DECK).reach_required is False


@pytest.mark.parametrize("kw", [
    {"dczdh_report": True, "dczdh_step_m": 0.02},
    {"dczdh_report": True, "dczdh_step_m": 0.001},
    {"windup_tol_deg": 1.0, "windup_max_iter": 2},
    {"track_points": 9},
])
def test_a_knob_of_an_opt_in_never_moves_the_published_answer(kw):
    """The new fields' own settings must not reach the published breakdown.

    This is the compatibility contract stated as something a mutation can
    fail: if the sensitivity difference ever leaked into the solve it is
    reported off (a shared array, a mutated ride height, a cached polar keyed
    wrongly), changing its step would move CZ.

    The two sensitivity rows turn the diagnostic ON as well as changing its
    step. Without that they would be vacuous now that it is opt-in — the step
    would be read by nothing and the parametrisation would gate air. What is
    asserted is the real invariant: measuring dCZ/dh, at whatever step, leaves
    the published breakdown bit-for-bit alone.
    """
    ref = CarWingProblem()
    for x in (_x(), _x(alpha=2.0, ep=0.0, ride=0.55, b=1.25, taper=0.45),
              _x(alpha=11.5, ep=0.24, ride=0.45, b=1.95, taper=1.0)):
        _same(evaluate_car_wing(x, CarWingProblem(**kw)),
              evaluate_car_wing(x, ref))


def test_the_published_mount_is_free_and_a_deck_says_nothing_to_it(base):
    """The default layout charges NOTHING, and the deck has nothing to measure.

    The plates are on the wing for the nonplanar benefit already, so bolting
    the car to them adds no wetted area and no new corner: both mount drag
    terms are EXACTLY zero and the whole of CD is the wing's own two terms.
    The attachment deck — the number that used to set a pylon's length, and
    the only reason this family ever had one — now moves nothing: stating it
    leaves the published breakdown bit-for-bit alone, which is the honest way
    to say that the pylon is gone rather than merely hidden.
    """
    assert base["cd0_struts"] == 0.0 and base["CD_junction"] == 0.0
    assert base["CD"] == base["CDi"] + base["CDp"]
    assert base["deck_height_m"] is None and base["pylon_length_m"] is None
    assert base["g_pylon_reach"] is None

    decked = evaluate_car_wing(_x(), CarWingProblem(deck_height_m=DECK))
    _same(decked, base)
    assert decked["deck_height_m"] == DECK        # reported, and inert


def test_the_published_beam_is_a_wing_simply_supported_at_its_tips(base):
    """The published beam, off carmount's integrator AND off the closed form.

    The plates grip at +/- b/2, so the published wing is the classical simply
    supported beam: it SAGS, its largest moment is at the centreline, and
    carmount's general signed integrator evaluated at the tip station has to
    reproduce ``bending_moment(..., "ends")`` and ``deflection_index`` to the
    bit. That identity is what says generalising the beam did not move the
    family it generalised — and it is why the two closed forms are still
    exported rather than deleted with the flag that used to pick between them.
    """
    p = CarWingProblem()
    signed = carmount.beam_moment(base["y"], base["load_Npm"], 0.5 * B)
    assert np.array_equal(base["moment_Nm"], np.abs(signed))
    assert np.array_equal(
        base["moment_Nm"],
        bending_moment(base["y"], base["load_Npm"], B, "ends"))
    assert base["deflection_m"] == (
        carmount.beam_deflection_index(base["y"], signed, 0.5 * B) / p.ei_nm2)
    assert base["deflection_m"] == (
        deflection_index(base["y"], base["moment_Nm"], B, "ends") / p.ei_nm2)
    assert np.all(signed <= 0.0) and base["bending_sense"] == "sagging"
    assert base["M_root_Nm"] == base["M_max_Nm"]   # the centreline carries it
    assert base["supports"] == "ends"
    assert base["CZ"] == base["vlm"].CL      # no knockdown on the published path


def test_the_published_score_and_labels_are_untouched(base):
    assert base["score"] == base["CZ"]
    assert base["constraint_labels"] == ["drag budget margin",
                                         "deflection margin"]
    assert len(base["g"]) == 2
    assert fg_car_wing(_x(), CarWingProblem())[1] == pytest.approx(base["g"])


# =====================================================================
# 2. the mount as a continuum
# =====================================================================

def _spec(**kw):
    kw.setdefault("station_frac", 0.0)
    return carmount.MountSpec(**kw)


def test_a_published_layout_adopted_as_a_mountspec_changes_nothing(base):
    """Adopting carmount on the published layout now moves NOTHING at all.

    The one thing the two descriptions used to disagree about was the PYLON
    LENGTH — carmount charges it over ride - deck and carwing charged it over
    the whole ride height — and with no pylon on either published layout that
    exception has nothing left to apply to. So the assertion is the strongest
    one available: every published key equal with ``==``, the two mount drag
    terms included, and the grip at the tip where the flag put it.

    Only the mount's NAME moves, because the two descriptions have different
    vocabularies for the same layout; that is asserted rather than skipped.
    """
    out = evaluate_car_wing(_x(), CarWingProblem(
        mount_spec=carmount.published_layout("tips")))
    assert out["feasible"], out["reason"]
    _same(out, base, UNNAMED_KEYS)
    assert out["cd0_struts"] == base["cd0_struts"] == 0.0
    assert out["n_pylons"] == 0 and out["pylon"]["CD"] == 0.0
    assert out["y_station_m"] == pytest.approx(0.5 * B)
    assert base["mount"] == "tips" and out["mount"] == "endplate"


def test_the_two_layouts_buy_stiffness_with_sheets_and_with_corners(base):
    """WHERE the plates grip is the whole of the mount question now.

    Neither published layout builds a pylon, so the mount is priced and never
    flown: the wing's own aerodynamics are bit-for-bit the same on both. What
    the inboard grip BUYS is a beam several times stiffer, and what it PAYS is
    two more sheets of wetted area and the two corners they make with the
    wing. Every one of those is asserted with its direction — a layout that
    came out stiffer AND cheaper would mean the sheets were never charged —
    and the sheet charge itself is re-derived from ``_strut_cd0`` at the LOCAL
    chord where the grip lands, which is the pricing rule and not the float.
    """
    both = compare_mounts(_x())
    assert sorted(both) == ["inboard", "tips"]
    tips, inboard = both["tips"], both["inboard"]
    _same(tips, base)
    # the same wing, flown once: the mount is a charge, not a flow condition
    _same(inboard, tips, ("CZ", "CL_model", "CDi", "CDp", "e", "AR", "S_m2",
                          "b_m", "y", "load_Npm"))

    # bought: an inboard support cuts the movement several times over
    assert inboard["deflection_m"] < 0.5 * tips["deflection_m"]
    assert inboard["M_max_Nm"] < tips["M_max_Nm"]
    assert inboard["g_deflection"] > tips["g_deflection"]
    # ...and the two grips bend the spar in opposite senses, which is the
    # mechanism rather than a relabelling
    assert tips["bending_sense"] == "sagging"
    assert inboard["bending_sense"] == "hogging"

    # paid for: two sheets and two corners, at the chord where they grip
    assert tips["cd0_struts"] == 0.0 and tips["CD_junction"] == 0.0
    p = CarWingProblem()
    c_grip = float(tips["wing"].chord(
        np.array([carmount.INBOARD_STATION_FRAC * 0.5 * B]))[0])
    assert inboard["cd0_struts"] == _strut_cd0(
        2, 0.12, c_grip, p.S, p.strut_cf)          # ep_h = 0.12 from _x()
    # ...and the LOCAL chord is load-bearing: a sheet is as long as the wing
    # is where it grips it, so on a tapered planform the root-chord charge is
    # a different — and larger — number
    c_root = float(tips["wing"].chord(np.array([0.0]))[0])
    assert c_grip < c_root
    assert inboard["cd0_struts"] < _strut_cd0(2, 0.12, c_root, p.S, p.strut_cf)
    assert inboard["CD_junction"] > 0.0
    assert inboard["CD"] > tips["CD"]
    assert inboard["efficiency"] < tips["efficiency"]


def test_an_interior_station_is_stiffer_than_either_end():
    """The reason the continuum exists: the flag offers two of these stations.

    carmount derives the balanced-moment station; what the WING cares about is
    the deflection, and a mount between the ends beats both of them on it by a
    wide margin. A helper that ignored the station and kept the cantilever
    formula would return the centre answer here and fail.
    """
    got = {}
    for name, spec in (("centre", _spec(station_frac=0.0)),
                       ("mid", _spec(station_frac=0.5)),
                       ("ends", carmount.published_layout("tips"))):
        out = evaluate_car_wing(_x(), CarWingProblem(mount_spec=spec))
        assert out["feasible"], out["reason"]
        got[name] = out
    assert got["mid"]["deflection_m"] < got["centre"]["deflection_m"]
    assert got["mid"]["deflection_m"] < got["ends"]["deflection_m"]
    # ...and it is the SIGNED moment that was integrated. At an interior mount
    # the moment changes sign at the support, so a magnitude integrates to a
    # different — and wrong — deflection; carmount says so and this is where
    # carwing has to prove it took the advice.
    mid = got["mid"]
    signed = carmount.beam_moment(mid["y"], mid["load_Npm"],
                                  mid["y_station_m"])
    assert mid["deflection_m"] == (
        carmount.beam_deflection_index(mid["y"], signed, mid["y_station_m"])
        / CarWingProblem().ei_nm2)
    assert mid["deflection_m"] != pytest.approx(
        carmount.beam_deflection_index(mid["y"], np.abs(signed),
                                       mid["y_station_m"])
        / CarWingProblem().ei_nm2, rel=1e-3)
    assert np.array_equal(mid["moment_Nm"], np.abs(signed))
    assert got["mid"]["supports"] == "station"
    assert got["mid"]["y_station_m"] == pytest.approx(0.25 * B)
    # ...and the interior mount is in BOTH bending senses at once, which is
    # the statement neither published end can make
    assert got["mid"]["bending_sense"] == "mixed"
    assert got["centre"]["bending_sense"] == "hogging"
    assert got["ends"]["bending_sense"] == "sagging"


def test_the_junction_is_priced_where_the_pylon_meets_the_wing():
    """On a TAPERED wing, moving the mount outboard must cut the corner.

    The charge is a drag area built on the local thickness, so it scales as
    the square of the local chord. Asserted against the chord ratio the
    planform itself reports, which is what makes this a test of the pricing
    and not of the number.
    """
    x = _x(taper=0.5)
    root = evaluate_car_wing(x, CarWingProblem(mount_spec=_spec(station_frac=0.0)))
    out = evaluate_car_wing(x, CarWingProblem(mount_spec=_spec(station_frac=0.8)))
    wing = root["wing"]
    ratio = (float(wing.chord(np.array([0.8 * 0.5 * B]))[0])
             / float(wing.chord(np.array([0.0]))[0]))
    assert out["CD_junction"] / root["CD_junction"] == pytest.approx(
        ratio ** 2, rel=1e-9)
    assert out["CD_junction"] < root["CD_junction"]


def test_the_pylon_build_up_charges_more_than_the_flat_plate_at_the_same_length():
    """The two pylon drag models, on ONE length, so only the model differs.

    carmount derives why: no form factor at all (1 against 1.26 at t/c 0.12)
    and a Cf frozen at 0.005 against the pylon's own Reynolds number. The
    crude model is optimistic, and the RATIO is Cf(Re) FF / 0.005 — recomputed
    here from the drag module rather than pasted.
    """
    from aerobo.drag import skin_friction_cf, wing_form_factor

    p = CarWingProblem()
    plate = evaluate_car_wing(_x(), CarWingProblem(
        mount_spec=_spec(pylon_drag_model="flat_plate")))
    build = evaluate_car_wing(_x(), CarWingProblem(
        mount_spec=_spec(pylon_drag_model="buildup")))
    chord = 0.12
    cf = skin_friction_cf(p.rho * p.V * chord / p.mu, lref=chord)
    expect = cf * wing_form_factor(0.12) / 0.005
    assert build["cd0_struts"] / plate["cd0_struts"] == pytest.approx(
        expect, rel=1e-12)
    assert build["cd0_struts"] > plate["cd0_struts"]


# ---------------- the windup fixed point ----------------

def test_a_mount_at_the_aerodynamic_centre_winds_the_wing_up_by_nothing(base):
    """The OFF invariant of the whole torsion model, with ``==``.

    The torque is a difference of two fractions, so at the quarter chord it is
    an exact array of zeros and the fixed point settles on the first pass.
    Anything less than exact here would mean the published answer moves the
    day someone adopts a MountSpec, which is precisely what carmount's
    defaults were chosen to prevent.
    """
    out = evaluate_car_wing(_x(), CarWingProblem(mount_spec=_spec()))
    assert out["windup_root_deg"] == 0.0 and out["windup_tip_deg"] == 0.0
    assert out["windup_max_deg"] == 0.0
    assert out["windup_iters"] == 0
    assert out["e_frac"] == 0.0
    assert out["CZ"] == base["CZ"]           # the whole point: nothing moved
    assert out["g_divergence"] == float("inf")


@pytest.mark.parametrize("x_attach,sign", [(0.35, +1), (0.15, -1)])
def test_the_windup_follows_the_sign_of_the_offset(base, x_attach, sign):
    """Aft of the aerodynamic centre the wing winds UP (more downforce, the
    divergence-prone sense); ahead of it, it washes OUT. Both signs, because a
    model that only ever added lift would pass the first half."""
    out = evaluate_car_wing(_x(alpha=4.0), CarWingProblem(
        mount_spec=_spec(x_attach_frac=x_attach, gj_nm2=100.0)))
    rigid = evaluate_car_wing(_x(alpha=4.0), CarWingProblem())
    assert out["feasible"], out["reason"]
    assert np.sign(out["windup_tip_deg"]) == sign
    assert np.sign(out["CZ"] - rigid["CZ"]) == sign
    assert np.sign(out["e_frac"]) == sign


def test_the_converged_windup_is_the_sum_of_its_own_feedback_series():
    """The fixed point, against a hand computation it does not share code with.

    The map is affine to first order: each round adds the twist the PREVIOUS
    round's extra load produces, so the limit is the geometric sum

        phi_inf = phi_rigid / (1 - gain)

    with phi_rigid the twist of the RIGID solve — computed here straight off
    the published run through carmount's public API — and ``gain`` the
    contraction the solver measured. A fixed point that converged to the wrong
    place, or that applied the twist twice, satisfies "it converged" and dies
    here. The residual is the non-affine part (the load redistributes as the
    wing twists), so it is checked as a SMALL percentage rather than zero.
    """
    x = _x(alpha=4.0)
    rigid = evaluate_car_wing(x, CarWingProblem())
    res = rigid["vlm"]
    main = ~res.is_winglet
    for gj, tol_pct in ((400.0, 0.2), (100.0, 0.5), (40.0, 1.0)):
        m = carmount.torsion_moment(rigid["y"], rigid["load_Npm"],
                                    res.c[main], 0.35)
        phi = carmount.torsion_twist(rigid["y"], m, 0.0, gj)
        fit = _twist_projection(rigid["y"], phi, res.c[main] * res.width[main],
                                B)
        out = evaluate_car_wing(x, CarWingProblem(
            mount_spec=_spec(x_attach_frac=0.35, gj_nm2=gj)))
        assert out["feasible"], out["reason"]
        gain = out["windup_contraction"]
        assert 0.0 < gain < 1.0
        predicted = fit[1] / (1.0 - gain)
        assert predicted == pytest.approx(out["windup_tip_deg"],
                                          rel=tol_pct / 100.0)
        # ...and the rigid twist itself is linear in 1/GJ, so the series is
        # being summed on a torsion model that is doing what it says
        assert fit[1] * gj == pytest.approx(
            _twist_projection(
                rigid["y"],
                carmount.torsion_twist(rigid["y"], m, 0.0, 1.0),
                res.c[main] * res.width[main], B)[1], rel=1e-9)


def test_the_measured_feedback_gain_is_below_the_single_mode_prediction():
    """carmount's q/q_div is CONSERVATIVE, and this says by how much.

    carmount.divergence_q derives its single-DOF estimate from the 2-D lift
    slope and says so ("puts q_div LOW: CONSERVATIVE, deliberately"). The
    fixed point measures the real thing. The two must agree in ORDER and the
    measured one must be the smaller — a solver that fed back the strip-theory
    response instead of the solved one would push the ratio to 1.
    """
    x = _x(alpha=1.0)
    seen = []
    for gj in (400.0, 100.0, 40.0):
        out = evaluate_car_wing(x, CarWingProblem(
            mount_spec=_spec(x_attach_frac=0.35, gj_nm2=gj)))
        assert out["feasible"], out["reason"]
        ratio = out["windup_contraction"] / (out["q_Pa"] / out["q_div_Pa"])
        seen.append(ratio)
        assert 0.4 < ratio < 0.7, (gj, ratio)
    # the ratio barely moves with stiffness: it is a property of the wing's
    # response, not of how hard it is being pushed
    assert max(seen) - min(seen) < 0.01


def test_divergence_is_refused_in_contract_and_names_itself():
    """A wing that will not stop winding up is a reason, never an exception.

    Two distinct refusals, and the difference matters to whoever reads it: a
    map that has STOPPED CONTRACTING is divergence and no amount of extra
    iteration helps, while one that is merely slow is a wing being flown close
    to the limit and the cap is the thing to raise. Both carry the margin.
    """
    x = _x(alpha=1.0)
    diverged = evaluate_car_wing(x, CarWingProblem(
        mount_spec=_spec(x_attach_frac=0.35, gj_nm2=10.0)))
    assert diverged["feasible"] is False
    assert "divergence" in diverged["reason"]
    assert diverged["diverged"] is True
    assert diverged["windup_contraction"] >= 1.0
    assert diverged["g_divergence"] < 0.0        # the single mode agrees here
    assert diverged["q_div_Pa"] > 0.0

    slow = evaluate_car_wing(x, CarWingProblem(
        mount_spec=_spec(x_attach_frac=0.35, gj_nm2=40.0), windup_max_iter=3))
    assert slow["feasible"] is False
    assert slow["diverged"] is False
    assert "had not settled" in slow["reason"]
    assert slow["windup_contraction"] < 1.0
    # ...and raising the cap is exactly what the reason says to do
    assert evaluate_car_wing(x, CarWingProblem(
        mount_spec=_spec(x_attach_frac=0.35, gj_nm2=40.0)))["feasible"]


def test_a_refused_windup_returns_the_declared_failure_width():
    prob = CarWingProblem(mount_spec=_spec(x_attach_frac=0.35, gj_nm2=10.0),
                          downforce_min_n=200.0)
    f, g = fg_car_wing(_x(alpha=1.0), prob)
    assert f == PENALTY
    assert g == [G_FAIL] * prob.n_constraints
    # 5: the published four PLUS the mount clearance margin, which every
    # pylon-borne mount declares (the plate may not hang past the deck)
    assert len(g) == len(prob.constraint_labels) == 5


# ---------------- the suction-side loss ----------------

def test_the_suction_loss_is_off_by_default_bit_for_bit(base):
    out = evaluate_car_wing(_x(), CarWingProblem(mount_spec=_spec()))
    assert out["mount_lift_knockdown"] == 0.0
    assert out["CZ"] == base["CZ"]
    assert np.array_equal(out["load_Npm"], base["load_Npm"])


def test_a_live_suction_loss_removes_lift_in_proportion_to_itself():
    """The calibration the user owns has to actually DO something.

    Linear in the magnitude, because the knockdown is a per-strip multiplier
    applied to a fixed load — and a pressure-side (swan neck) mount takes it
    all back, which is the geometry statement carmount built the switch for.
    """
    rigid = evaluate_car_wing(_x(), CarWingProblem(mount_spec=_spec()))
    got = []
    for mag in (0.1, 0.2, 0.4):
        out = evaluate_car_wing(_x(), CarWingProblem(
            mount_spec=_spec(suction_loss=mag)))
        assert out["feasible"], out["reason"]
        got.append(out)
        assert out["CZ"] < rigid["CZ"]
        # ONCE, not twice: the span-integrated knockdown is the whole of what
        # the footprint takes off the coefficient
        assert out["CZ"] == pytest.approx(
            rigid["CZ"] * (1.0 - out["mount_lift_knockdown"]), rel=1e-12)
    ratios = [o["mount_lift_knockdown"] / m
              for o, m in zip(got, (0.1, 0.2, 0.4))]
    assert ratios[0] == pytest.approx(ratios[1], rel=1e-12)
    assert ratios[0] == pytest.approx(ratios[2], rel=1e-12)
    swan = evaluate_car_wing(_x(), CarWingProblem(
        mount_spec=_spec(suction_loss=0.4, side="pressure")))
    assert swan["mount_lift_knockdown"] == 0.0
    assert swan["CZ"] == rigid["CZ"]


def test_the_loss_lands_on_the_strips_under_the_pylon_and_nowhere_else():
    """A per-strip loss has to be per-strip, not a scale factor on CZ.

    The span-integrated identity (the load integral falls by exactly the
    reported knockdown) plus the LOCATION (the strips beyond the footprint are
    untouched, to the bit) is what separates a footprint from a coefficient
    fudge — and it is what a "knock CZ down and leave the loads alone" mutation
    fails, because the beam is fed the same array the coefficient was.
    """
    spec = carmount.MountSpec(station_frac=0.5)
    rigid = evaluate_car_wing(_x(), CarWingProblem(mount_spec=spec))
    lost = evaluate_car_wing(_x(), CarWingProblem(
        mount_spec=carmount.MountSpec(station_frac=0.5, suction_loss=0.4)))
    width = rigid["vlm"].width[~rigid["vlm"].is_winglet]
    y = rigid["y"]
    total_r = float(np.sum(rigid["load_Npm"] * width))
    total_l = float(np.sum(lost["load_Npm"] * width))
    assert total_l / total_r == pytest.approx(
        1.0 - lost["mount_lift_knockdown"], rel=1e-12)
    assert np.all(lost["load_Npm"] <= rigid["load_Npm"])
    assert np.any(lost["load_Npm"] < rigid["load_Npm"])
    far = np.abs(np.abs(y) - rigid["y_station_m"]) > (
        spec.footprint_half_width_m() + width.max())
    assert far.any()
    assert np.array_equal(lost["load_Npm"][far], rigid["load_Npm"][far])


def test_compare_mounts_refuses_a_problem_whose_mount_is_a_spec():
    with pytest.raises(ValueError, match="two-valued mount question"):
        compare_mounts(_x(), CarWingProblem(mount_spec=_spec()))


def test_the_mount_is_stated_exactly_once():
    with pytest.raises(ValueError, match="stated once"):
        CarWingProblem(mount="inboard", mount_spec=_spec())
    with pytest.raises(ValueError, match="stated once"):
        CarWingProblem(deck_height_m=0.25, mount_spec=_spec())
    with pytest.raises(ValueError, match="MountSpec"):
        CarWingProblem(mount_spec="centre")


def test_a_mount_that_cannot_be_built_is_a_reason_not_an_exception():
    """Both of carmount's in-contract violations, reached from the design box.

    A station outboard of the tip and a wing at or below its own deck are
    designs an optimiser can propose without leaving its own bounds.
    """
    out = evaluate_car_wing(_x(b=1.2), CarWingProblem(
        mount_spec=carmount.MountSpec(y_station_m=0.9)))
    assert out["feasible"] is False and "outboard of the tip" in out["reason"]
    low = evaluate_car_wing(_x(ride=0.20), CarWingProblem(
        mount_spec=_spec(deck_height_m=0.25)))
    assert low["feasible"] is False and "pylon" in low["reason"]


# =====================================================================
# 3. the deck, and the pylon that has to reach it
# =====================================================================

def _pylon(**kw):
    """A hand-built pylon mount — the only way this family still gets one."""
    kw.setdefault("station_frac", 0.0)
    kw.setdefault("deck_height_m", DECK)
    return carmount.MountSpec(**kw)


def test_a_pylon_is_charged_over_the_gap_between_the_wing_and_its_deck(base):
    """The deck is what a PYLON is measured against, and only a pylon has one.

    No registered layout builds one any more — both grip through the plates —
    but a caller pricing a swan neck by hand still states a deck, and the
    charge has to be over ride - deck rather than down to the track. Re-derived
    through ``_strut_cd0`` at the corrected length and cross-checked for
    LINEARITY in that length, which is the whole content of the crude model,
    and then against the plate-borne family, which charges nothing at all.
    """
    p = CarWingProblem(mount_spec=_pylon())
    out = evaluate_car_wing(_x(), p)
    assert out["feasible"], out["reason"]
    assert out["cd0_struts"] == _strut_cd0(
        p.n_pylons, RIDE - DECK, p.mount_spec.pylon_chord_m, p.S,
        p.mount_spec.pylon_cf)
    assert out["pylon_length_m"] == pytest.approx(RIDE - DECK)
    # linear in the length: a deck twice as far below the wing costs twice
    lower = evaluate_car_wing(_x(), CarWingProblem(
        mount_spec=_pylon(deck_height_m=RIDE - 2.0 * (RIDE - DECK))))
    assert lower["cd0_struts"] / out["cd0_struts"] == pytest.approx(
        2.0, rel=1e-12)
    # ...and it is a real cost against the layout that needs no pylon
    assert out["cd0_struts"] > 0.0 == base["cd0_struts"]
    assert out["CD"] > base["CD"] and out["efficiency"] < base["efficiency"]


def test_the_reach_margin_exists_iff_a_pylon_and_a_deck_are_both_stated():
    """A margin exists iff its budget does — the family's own rule.

    Deleting the pylon deleted the reach question from every REGISTERED
    layout: the plates already span the gap, so a deck stated on the published
    family declares nothing and the problem keeps the two margins it always
    had. Both halves of the rule are still asserted, because both are still
    reachable — a hand-built pylon mount puts the third margin back, and a
    plate mount that carries a deck (every MountSpec does, by default) still
    does not.
    """
    assert CarWingProblem().constraint_labels[-1] == "deflection margin"
    decked = CarWingProblem(deck_height_m=DECK)
    assert decked.deck_m == DECK and decked.reach_required is False
    assert decked.constraint_labels[-1] == "deflection margin"
    assert decked.n_constraints == 2
    assert CarWingProblem(mount="inboard",
                          deck_height_m=DECK).n_constraints == 2

    plates = CarWingProblem(mount_spec=carmount.published_layout("inboard"))
    assert plates.deck_m is not None          # a deck, and nothing to reach it
    assert plates.reach_required is False and plates.n_constraints == 2

    with_pylon = CarWingProblem(mount_spec=_pylon())
    assert with_pylon.reach_required is True
    # a pylon declares TWO margins: it has to reach the deck, and the plate
    # beside it has to stop short of the same deck. The clearance one is the
    # newer and therefore the LAST, so "pylon reach margin" keeps its index.
    assert with_pylon.constraint_labels[-2] == "pylon reach margin"
    assert with_pylon.constraint_labels[-1] == "mount clearance margin"
    assert with_pylon.n_constraints == 4
    out = evaluate_car_wing(_x(), with_pylon)
    assert len(out["g"]) == 4 == with_pylon.n_constraints
    assert out["g"][-2] == out["g_pylon_reach"]
    assert out["g"][-1] == out["g_mount_clearance"]
    # ...and switching it off removes the margin rather than satisfying it
    off = CarWingProblem(mount_spec=_pylon(), plate_deck_clearance=False)
    assert off.constraint_labels[-1] == "pylon reach margin"
    assert off.n_constraints == 3


def test_a_wing_below_its_own_deck_is_refused_instead_of_scored():
    """Where the reach margin used to change sign there is now a REFUSAL.

    The published family used to build a wing under its own deck, charge it no
    pylon drag and hand back a negative margin. carmount refuses it instead:
    a pylon of negative length is not a thing to have an opinion about, and
    the check runs per point, before the wing is ever solved. Swept over the
    whole ride-height row, every buildable design has a strictly positive
    reach and every refused one names the pylon in its reason — so the margin
    is a report and the GEOMETRY is the gate. (Which also means the margin can
    no longer bite; see the report.)
    """
    prob = CarWingProblem(mount_spec=_pylon())
    lo, hi = CarWingProblem.RIDE_HEIGHT_BOUNDS_M
    built = refused = 0
    for h in np.linspace(lo, hi, 13):
        out = evaluate_car_wing(_x(ride=h), prob)
        if out["feasible"]:
            built += 1
            assert h > DECK
            assert out["g_pylon_reach"] == pytest.approx(h / DECK - 1.0)
            assert out["g_pylon_reach"] > 0.0
        else:
            refused += 1
            assert h <= DECK
            assert "pylon" in out["reason"]
            assert out["score"] == PENALTY
    assert built and refused


# =====================================================================
# 4. the mission
# =====================================================================

@pytest.fixture(scope="module")
def track():
    return cartrack.synthetic_lap()


def test_laptime_is_refused_by_name_without_a_circuit():
    with pytest.raises(ValueError, match="track_spec"):
        CarWingProblem(objective="laptime")
    assert "laptime" in CAR_OBJECTIVES
    CarWingProblem(objective="laptime", track_spec=cartrack.synthetic_lap())


def test_the_lap_is_the_score_and_the_wing_is_worth_time(track):
    """A wing has to EARN its lap time, and the score has to be a lap time.

    The transform is stated as the negation, so the score is minus the
    seconds; and the wing has to beat the bare car, which is the only reason
    to fit one. Both are outcomes: a score that ranked by CZ would pass
    neither.
    """
    prob = CarWingProblem(track_spec=track, objective="laptime",
                          CD_budget=None)
    out = evaluate_car_wing(_x(), prob)
    assert out["feasible"], out["reason"]
    assert out["score"] == -out["lap_time_s"]
    assert out["objective_label"] == CAR_OBJECTIVES["laptime"]
    bare = cartrack.lap_time(0.0, 0.0, prob.car_used, track)
    assert out["lap_time_s"] < bare["lap_time_s"]
    # ...and a wing with more downforce for the same drag is never slower
    more = evaluate_car_wing(_x(alpha=10.0), prob)
    assert more["CZ"] > out["CZ"]
    assert more["lap_time_s"] < out["lap_time_s"]


def test_the_multi_point_lap_is_the_single_point_lap_where_nothing_moves(track):
    """E5's own claim, and then the same claim broken on purpose.

    This family's coefficients are speed-INDEPENDENT with the shipped polars
    (the VLM is incompressible and Reynolds-free, the tables are read at one
    Reynolds number, the strut Cf is a constant) — to ROUND-OFF rather than to
    the bit, because the circulation is solved at the speed and the
    coefficient divides it back out, which costs one ulp. So the four
    evaluated points must agree to 1e-15 and the lap must land on the
    single-point one to the same order (measured: 9.2e-14 s in 61.3 s). Turn
    the flown Reynolds number on and it must NOT — measured 7.1e-3 s, eleven
    orders of magnitude away — otherwise the multi-point machinery is
    decoration.
    """
    prob = CarWingProblem(track_spec=track, objective="laptime",
                          CD_budget=None)
    out = evaluate_car_wing(_x(), prob)
    rows = out["track_points"]
    assert len(rows) == 4
    assert all(r["CZ"] == pytest.approx(out["CZ"], rel=1e-15)
               and r["CD"] == pytest.approx(out["CD"], rel=1e-15)
               for r in rows)
    single = cartrack.lap_time(out["CZ"] * prob.S, out["CD"] * prob.S,
                               prob.car_used, track)
    assert out["lap_time_s"] == pytest.approx(single["lap_time_s"], rel=1e-13)

    flown = evaluate_car_wing(_x(), CarWingProblem(
        track_spec=track, objective="laptime", CD_budget=None,
        flown_reynolds=True))
    czs = [r["CZ"] for r in flown["track_points"]]
    assert czs[0] > czs[-1]                 # low speed = low Re = fatter polar
    single_flown = cartrack.lap_time(flown["CZ"] * prob.S,
                                     flown["CD"] * prob.S, prob.car_used,
                                     track)
    # SIGNED, because the sign is the physics: the straights are where the
    # lap spends its time, they are flown at the highest speed and therefore
    # at the highest Reynolds number, where the section is at its least
    # draggy. Evaluating there instead of at the mean speed must make the lap
    # FASTER, and a law that paired the speeds up the wrong way round would
    # land on the other side of the single-point answer.
    gap = single_flown["lap_time_s"] - flown["lap_time_s"]
    assert 1e-3 < gap < 0.05
    assert flown["track_points"][0]["cd_a_m2"] > flown["track_points"][-1]["cd_a_m2"]
    assert abs(gap) > 1e6 * abs(out["lap_time_s"] - single["lap_time_s"])
    # and the same sign argument corner by corner: the SLOW corners are flown
    # at a low Reynolds number where this section makes more downforce, so
    # they must come out faster than the single-point lap says, while the
    # fastest corner — above the 55 m/s reference — must not
    delta = [a["V_in"] - b["V_in"]
             for a, b in zip(flown["lap"]["segments"], single_flown["segments"])
             if a["kind"] == "corner"]
    assert len(delta) == 3
    assert delta[0] > 0.0 and delta[1] > 0.0 and delta[2] < 0.0


def test_each_representative_point_is_flown_at_its_own_reynolds_number(track):
    prob = CarWingProblem(track_spec=track, objective="laptime",
                          CD_budget=None, flown_reynolds=True)
    out = evaluate_car_wing(_x(), prob)
    for r in out["track_points"]:
        expect = prob.rho * r["V"] * out["wing"].mac / prob.mu
        assert r["Re_flown"] == pytest.approx(expect, rel=1e-12)
    res = [r["Re_flown"] for r in out["track_points"]]
    assert res == sorted(res)


def test_a_heave_law_sinks_the_wing_with_speed_and_a_rigid_car_does_not(track):
    """The ride height is a design variable and the car does not hold it still.

    With a rigid car every point flies the STATED height, bit for bit — the
    law short-circuits — and with a live one the wing sinks, makes more
    downforce and less drag, and the lap gets faster. The balance moves too,
    which is the mechanism cartrack names for a balance that is otherwise
    speed-independent.
    """
    rigid = evaluate_car_wing(_x(), CarWingProblem(
        track_spec=track, objective="laptime", CD_budget=None))
    assert all(r["ride_height_m"] == RIDE for r in rigid["track_points"])

    car = cartrack.CarSpec(heave=cartrack.HeaveLaw(k_m_per_pa=2.0e-5))
    soft = evaluate_car_wing(_x(), CarWingProblem(
        track_spec=track, car_spec=car, objective="laptime", CD_budget=None))
    hs = [r["ride_height_m"] for r in soft["track_points"]]
    assert hs == sorted(hs, reverse=True)         # faster => lower
    assert max(hs) < RIDE
    czs = [r["CZ"] for r in soft["track_points"]]
    assert czs == sorted(czs)                     # lower => more downforce
    assert soft["lap_time_s"] < rigid["lap_time_s"]
    assert soft["aero_balance"] != rigid["aero_balance"]


def test_a_force_requirement_stated_at_a_speed_is_evaluated_there():
    """E5's pair. The force is q(V) S C(V), so BOTH factors move.

    With speed-independent coefficients that reduces EXACTLY to the q ratio,
    which is what makes this checkable by hand; with the flown Reynolds number
    on it does not, and the difference is the reason the coefficient is
    re-solved instead of referred.
    """
    p0 = CarWingProblem(downforce_min_n=300.0, drag_budget_n=30.0)
    at55 = evaluate_car_wing(_x(), p0)
    p1 = CarWingProblem(downforce_min_n=300.0, downforce_min_v_ms=30.0,
                        drag_budget_n=30.0, drag_budget_v_ms=70.0)
    out = evaluate_car_wing(_x(), p1)
    assert out["downforce_req_V_ms"] == 30.0 and out["drag_req_V_ms"] == 70.0
    assert out["downforce_req_N"] == pytest.approx(
        at55["downforce_N"] * (30.0 / 55.0) ** 2, rel=1e-12)
    assert out["drag_req_N"] == pytest.approx(
        at55["drag_N"] * (70.0 / 55.0) ** 2, rel=1e-12)
    # the reported operating point is still the family's own speed
    assert out["downforce_N"] == at55["downforce_N"]
    assert out["drag_N"] == at55["drag_N"]
    # the margins follow the requirement's speed, not the family's
    assert at55["g_downforce"] > 0.0 > out["g_downforce"]
    assert at55["g_drag_force"] > 0.0 > out["g_drag_force"]

    flown = evaluate_car_wing(_x(), CarWingProblem(
        downforce_min_n=300.0, downforce_min_v_ms=30.0, flown_reynolds=True))
    ref = evaluate_car_wing(_x(), CarWingProblem(flown_reynolds=True))
    q_ratio = ref["downforce_N"] * (30.0 / 55.0) ** 2
    assert flown["downforce_req_N"] > q_ratio * 1.02     # measured +6.9 %


def test_a_requirement_at_a_speed_is_asked_at_that_speed_s_ride_height():
    """The heave law reaches the E5 pair too, or the pair is asking about a
    wing the car never presents: at 70 m/s a soft car has already pulled the
    wing down, and that is where the downforce floor has to be met."""
    car = cartrack.CarSpec(heave=cartrack.HeaveLaw(k_m_per_pa=2.0e-5))
    rigid = evaluate_car_wing(_x(), CarWingProblem(
        downforce_min_n=300.0, downforce_min_v_ms=70.0))
    soft = evaluate_car_wing(_x(), CarWingProblem(
        downforce_min_n=300.0, downforce_min_v_ms=70.0, car_spec=car))
    assert car.ride_height_at(RIDE, 70.0) < RIDE
    assert soft["downforce_req_N"] > rigid["downforce_req_N"] * 1.01
    # ...and the number is the wing flown AT that height, not a heave-free one
    at_h = evaluate_car_wing(_x(ride=car.ride_height_at(RIDE, 70.0)),
                             CarWingProblem(V=70.0))
    assert soft["downforce_req_N"] == pytest.approx(at_h["downforce_N"],
                                                    rel=1e-12)


def test_a_speed_without_a_force_is_not_a_requirement():
    with pytest.raises(ValueError, match="downforce_min_n"):
        CarWingProblem(downforce_min_v_ms=30.0)
    with pytest.raises(ValueError, match="drag_budget_n"):
        CarWingProblem(drag_budget_v_ms=70.0)


def test_the_balance_is_reported_on_a_frame_and_constrained_on_a_window():
    """The decision this family makes about E8's balance, as an outcome.

    cartrack's reference car with this family's published wing sits BELOW its
    own derived window — "a rear wing alone drives the balance rearward" is
    the finding, not a bug — so declaring a margin on the FRAME alone would
    make every lap run start infeasible. The balance is therefore reported
    whenever a car is stated and constrained only when that car states a
    window.
    """
    framed = CarWingProblem(car_spec=cartrack.CarSpec())
    assert framed.balance_required is False
    out = evaluate_car_wing(_x(), framed)
    assert out["aero_balance"] is not None and out["g_balance"] is None
    lo, hi = framed.car_used.balance_window_used()
    assert out["aero_balance"] < lo                 # the finding, in the open

    windowed = CarWingProblem(
        car_spec=cartrack.CarSpec(balance_window=(0.30, 0.50)))
    assert windowed.balance_required is True
    assert windowed.constraint_labels[-1] == "aero balance margin"
    got = evaluate_car_wing(_x(), windowed)
    assert got["g"][-1] == got["g_balance"] > 0.0
    tight = evaluate_car_wing(_x(), CarWingProblem(
        car_spec=cartrack.CarSpec(balance_window=(0.45, 0.55))))
    assert tight["g_balance"] < 0.0


# =====================================================================
# 5. ride-height sensitivity
# =====================================================================

#: every breakdown key that only exists because dCZ/dh was measured
SENSITIVITY_KEYS = ("dCZ_dh_per_m", "dCZ_dh_scheme", "dCZ_dh_step_m",
                    "dCZ_dh_reason")


def _solve_counter(monkeypatch):
    """Count the VLM builds ``evaluate_car_wing`` performs, in place.

    ``carwing`` holds its own name for the class (``from .vlm import VLM``),
    so the counter has to replace the name IN carwing or it counts nothing.
    """
    from aerobo import carwing as cw

    real, n = cw.VLM, {"n": 0}

    def counted(*a, **kw):
        n["n"] += 1
        return real(*a, **kw)

    monkeypatch.setattr(cw, "VLM", counted)
    return n


def test_the_published_path_is_one_solve_and_the_diagnostic_is_two_more(
        monkeypatch):
    """The COST contract, as something a mutation fails.

    dCZ/dh was unconditional until session 62, which quietly made the
    published path three VLM builds and solves instead of one — measured at
    x2.97 on the whole evaluation, paid inside every optimiser's inner loop
    and again inside carwing_multi's. Reverting the ``if prob.dczdh_required``
    guard (or defaulting ``dczdh_report`` to True) puts the published count
    back at 3 and fails here. Counting BUILDS is what catches it: a mutation
    that computed the difference from a cached solve would still be honest,
    and one that re-solves is exactly the regression.
    """
    n = _solve_counter(monkeypatch)
    out = evaluate_car_wing(_x(), CarWingProblem())
    assert out["feasible"], out["reason"]
    assert n["n"] == 1, "the published path is ONE solve"

    n["n"] = 0
    got = evaluate_car_wing(_x(), CarWingProblem(dczdh_report=True))
    assert got["feasible"], got["reason"]
    assert n["n"] == 3, "the central difference is the base solve plus two"

    # ...and a declared ceiling buys the same two, because it cannot be
    # checked without them
    n["n"] = 0
    evaluate_car_wing(_x(), CarWingProblem(dczdh_max_per_m=2.0))
    assert n["n"] == 3


def test_an_unmeasured_sensitivity_is_absent_from_the_breakdown_not_None(base):
    """A key that is missing is honest; a key that is silently None is not.

    ``dCZ_dh_scheme == "unavailable"`` already MEANS "measured, and neither
    side could be flown". If the published breakdown carried the same four
    keys with None in them, a reader could not tell that from "nobody
    measured it", and gui/metrics.py — which drops a None row — would show
    the same nothing for both. So they are absent, and asking raises where
    the asking happened.
    """
    for k in SENSITIVITY_KEYS:
        assert k not in base, k
        with pytest.raises(KeyError):
            base[k]
    # the MARGIN key is the documented exception: every g_* is present and
    # None when its budget is not declared
    assert "g_dczdh" in base and base["g_dczdh"] is None


def test_the_opt_in_reports_the_sensitivity_and_points_the_right_way():
    got = evaluate_car_wing(_x(), CarWingProblem(dczdh_report=True))
    assert all(k in got for k in SENSITIVITY_KEYS)
    assert got["dCZ_dh_per_m"] < 0.0         # closer to the track = more CZ
    assert got["dCZ_dh_scheme"] == "central"
    assert got["dCZ_dh_step_m"] == 0.005
    assert got["g_dczdh"] is None            # reported, not constrained
    # ...and it is ADDITIVE: everything the published path reports is bit-for-
    # bit what it reported without the diagnostic
    _same(got, evaluate_car_wing(_x(), CarWingProblem()))


def test_a_low_wing_is_a_knife_edge_platform():
    """What dCZ/dh MEANS, as a trend rather than a number.

    Ground effect is strongly nonlinear in the gap, so the sensitivity grows
    as the wing comes down — and that is the whole engineering content of the
    quantity: the same millimetre of heave costs several times more downforce
    on a low wing than on a high one.
    """
    prob = CarWingProblem(dczdh_report=True)
    got = [evaluate_car_wing(_x(ride=h), prob)["dCZ_dh_per_m"]
           for h in (0.16, 0.20, 0.30, 0.45, 0.59)]
    assert all(g < 0.0 for g in got)
    assert got == sorted(got)                # increasingly negative downwards
    assert abs(got[0]) > 5.0 * abs(got[2])   # measured 2.416 vs 0.451 1/m


def test_the_central_difference_converges_at_second_order():
    """Halving the step must divide the change by four, or it is not a central
    difference. The VLM is deterministic, so there is no noise floor to hide
    a first-order scheme behind."""
    vals = [evaluate_car_wing(_x(), CarWingProblem(
        dczdh_report=True, dczdh_step_m=s))["dCZ_dh_per_m"]
        for s in (0.04, 0.02, 0.01, 0.005, 0.0025)]
    diffs = np.diff(vals)
    ratios = diffs[:-1] / diffs[1:]
    assert np.all(np.abs(ratios - 4.0) < 0.2), ratios


def test_a_declared_ceiling_switches_the_measurement_on_by_itself():
    """The ceiling is the second door to the same diagnostic.

    A ceiling that did NOT imply the measurement would refuse every design in
    contract ("could not be measured at either side"), which is a family that
    cannot be given this constraint at all. ``dczdh_required`` is the one
    place that answers, so the margin and the breakdown cannot disagree.
    """
    prob = CarWingProblem(dczdh_max_per_m=2.0)
    assert prob.dczdh_report is False        # nobody asked for the report...
    assert prob.dczdh_required is True       # ...the ceiling did
    got = evaluate_car_wing(_x(), prob)
    assert got["feasible"], got["reason"]
    assert all(k in got for k in SENSITIVITY_KEYS)
    assert got["g_dczdh"] == pytest.approx(
        1.0 - abs(got["dCZ_dh_per_m"]) / 2.0)


def test_the_sensitivity_can_be_constrained_and_the_margin_declares_itself():
    ride = 0.20
    free = evaluate_car_wing(_x(ride=ride), CarWingProblem(dczdh_report=True))
    limit = abs(free["dCZ_dh_per_m"])
    loose = CarWingProblem(dczdh_max_per_m=limit * 1.1)
    tight = CarWingProblem(dczdh_max_per_m=limit * 0.9)
    assert loose.constraint_labels[-1] == "ride-height sensitivity margin"
    assert loose.n_constraints == 3
    a = evaluate_car_wing(_x(ride=ride), loose)
    b = evaluate_car_wing(_x(ride=ride), tight)
    assert a["g_dczdh"] > 0.0 > b["g_dczdh"]
    assert a["g"][-1] == a["g_dczdh"] and len(a["g"]) == 3
    assert a["dCZ_dh_per_m"] == free["dCZ_dh_per_m"]


def test_a_step_the_wing_cannot_take_falls_back_and_says_which_side():
    """The difference has to survive a step the geometry refuses.

    A tall plate two centimetres above the track cannot come down another
    two, and a step past the ride height would put the wing under the road —
    where the image plane, being side-agnostic, solves happily and returns a
    quotient between two unrelated flows. Both come back as a ONE-SIDED
    quotient with the side that failed named, never as a silent central
    difference.
    """
    plate = evaluate_car_wing(_x(ep=0.25, ride=0.28),
                              CarWingProblem(dczdh_report=True,
                                             dczdh_step_m=0.02))
    assert plate["feasible"] is True
    assert plate["dCZ_dh_scheme"] == "forward"
    assert plate["dCZ_dh_per_m"] < 0.0
    assert "h - 0.02 m" in plate["dCZ_dh_reason"]

    under = evaluate_car_wing(_x(), CarWingProblem(dczdh_report=True,
                                                  dczdh_step_m=0.31))
    assert under["dCZ_dh_scheme"] == "forward"
    assert "below the track" in under["dCZ_dh_reason"]
    # ...and a declared ceiling is still answerable off the one-sided value
    got = evaluate_car_wing(_x(ep=0.25, ride=0.28),
                            CarWingProblem(dczdh_step_m=0.02,
                                           dczdh_max_per_m=1.0))
    assert got["feasible"] is True
    assert got["g_dczdh"] == pytest.approx(
        1.0 - abs(plate["dCZ_dh_per_m"]) / 1.0)


# =====================================================================
# 6. the Reynolds number the wing actually flies at
# =====================================================================

def test_the_flown_reynolds_number_is_off_by_default_and_says_so(base):
    assert base["flown_reynolds"] is False and base["Re_flown"] is None
    assert base["polar"].endswith("(XFOIL)")
    assert base["Re_mac"] < 1.0e6            # the mismatch the docstring flags


def test_reading_the_polar_at_the_flown_reynolds_number_changes_the_answer(base):
    """The opt-in has to move something, in a direction that follows from the
    polars themselves: below Re 1e6 the section is draggier, so CD rises."""
    out = evaluate_car_wing(_x(), CarWingProblem(flown_reynolds=True))
    assert out["feasible"], out["reason"]
    assert out["Re_flown"] == pytest.approx(out["Re_mac"], rel=1e-12)
    assert "Re" in out["polar"] and out["polar"] != base["polar"]
    assert out["CD"] > base["CD"]
    assert out["CDp"] > base["CDp"]
    assert out["efficiency"] < base["efficiency"]
    assert abs(out["CD"] / base["CD"] - 1.0) < 0.05


def test_a_reynolds_number_outside_the_bank_is_refused_in_contract():
    """No extrapolated viscous polar, ever — and a refusal, not a raise: the
    speed is configuration but the MAC is a design variable, so the bank's
    edge is reachable from inside the box."""
    out = evaluate_car_wing(_x(), CarWingProblem(V=2.0, flown_reynolds=True))
    assert out["feasible"] is False
    assert "outside the bank" in out["reason"]


def test_the_flown_reynolds_number_needs_a_bank_to_read():
    from aerobo.polar import default_polar_family

    with pytest.raises(ValueError, match="flown_reynolds"):
        CarWingProblem(flown_reynolds=True,
                       section_polar=default_polar_family().at(0.15))


# =====================================================================
# 7. the declaration stays a property
# =====================================================================

@pytest.mark.parametrize("kw", [
    {},
    {"CD_budget": None},
    {"deck_height_m": DECK},
    {"dczdh_max_per_m": 2.0},
    {"deck_height_m": DECK, "dczdh_max_per_m": 2.0,
     "drag_budget_n": 90.0, "downforce_min_n": 200.0, "CD_budget": None},
    {"car_spec": cartrack.CarSpec(balance_window=(0.30, 0.50))},
    {"mount_spec": carmount.published_layout("tips")},
    {"mount_spec": carmount.published_layout("inboard")},
    {"mount_spec": carmount.MountSpec(station_frac=0.0, deck_height_m=DECK)},
])
def test_the_margin_vector_is_exactly_what_the_problem_declares(kw):
    prob = CarWingProblem(**kw)
    out = evaluate_car_wing(_x(), prob)
    assert out["feasible"], out["reason"]
    assert len(out["g"]) == prob.n_constraints == len(prob.constraint_labels)
    assert out["constraint_labels"] == list(prob.constraint_labels)
    # the failure vector follows the same declaration, or the optimiser gets a
    # shape error at the first refused design instead of a bad score
    f, g = fg_car_wing(np.zeros(prob.dim) - 99.0, prob)
    assert f == PENALTY and len(g) == prob.n_constraints
    assert fg_car_wing(_x(), prob)[1] == pytest.approx(out["g"])


def test_the_published_margins_keep_their_index_when_a_new_one_is_added():
    """A label that moved would relabel every margin a report already shows."""
    prob = CarWingProblem(drag_budget_n=90.0, downforce_min_n=200.0,
                          mount_spec=_pylon(), dczdh_max_per_m=2.0,
                          car_spec=cartrack.CarSpec(balance_window=(0.3, 0.5)))
    assert prob.constraint_labels == (
        "drag budget margin", "drag force margin", "deflection margin",
        "downforce floor margin", "pylon reach margin",
        "ride-height sensitivity margin", "aero balance margin",
        # ...and the newest one appended, which is the whole point of this
        # test: the seven above are at the indices they always were
        "mount clearance margin")
    out = evaluate_car_wing(_x(), prob)
    assert out["g"][:4] == pytest.approx(
        [out["g_drag"], out["g_drag_force"], out["g_deflection"],
         out["g_downforce"]])
    assert out["g"][4:] == pytest.approx(
        [out["g_pylon_reach"], out["g_dczdh"], out["g_balance"],
         out["g_mount_clearance"]])


def test_the_new_fields_never_touch_the_design_vector():
    ref = CarWingProblem()
    for kw in ({"mount_spec": carmount.published_layout("inboard")},
               {"mount": "inboard"},
               {"deck_height_m": DECK},
               {"track_spec": cartrack.synthetic_lap()},
               {"dczdh_max_per_m": 2.0},
               {"flown_reynolds": True},
               {"downforce_min_n": 200.0, "downforce_min_v_ms": 30.0}):
        prob = CarWingProblem(**kw)
        assert prob.dim == ref.dim
        assert np.array_equal(prob.bounds, ref.bounds)


def test_the_projection_preserves_the_added_lift_and_is_exact_on_a_line():
    """The two properties :func:`_twist_projection` is chosen FOR.

    A chord-weighted least-squares fit reproduces the chord-weighted mean
    exactly (the residual is orthogonal to the constant), which is the added
    lift under strip theory; and it reproduces a linear phi exactly, so it is
    invisible where it has nothing to do. A projection that used the tip value
    or an unweighted fit fails the first.
    """
    y = np.linspace(-0.8, 0.8, 41)
    c = 0.3 - 0.1 * np.abs(y)                  # tapered
    w = c * (y[1] - y[0])
    eta = np.abs(y) / 0.8
    line = np.deg2rad(0.4 + 1.3 * eta)
    fit = _twist_projection(y, line, w, B)
    assert fit[0] == pytest.approx(0.4, rel=1e-9)
    assert fit[1] == pytest.approx(1.7, rel=1e-9)

    curved = np.deg2rad(2.0 * eta - 0.9 * eta ** 2)
    p0, p1 = _twist_projection(y, curved, w, B)
    fitted = np.deg2rad(p0 + (p1 - p0) * eta)
    assert float(np.sum(w * (curved - fitted))) == pytest.approx(0.0, abs=1e-15)
    assert not np.allclose(curved, fitted)     # it is a fit, not an identity


def test_the_wing_the_windup_flies_is_the_wing_the_breakdown_reports():
    """The re-solve must hand back the TWISTED planform, not the rigid one.

    A fixed point that iterated on a local wing and then reported the original
    would be invisible to every scalar test — the numbers would all be the
    wound-up ones — until someone drew the geometry.
    """
    out = evaluate_car_wing(_x(alpha=4.0), CarWingProblem(
        mount_spec=_spec(x_attach_frac=0.35, gj_nm2=100.0)))
    assert out["feasible"], out["reason"]
    wing = out["wing"]
    assert isinstance(wing, geometry.Wing)
    assert wing.twist_root_deg == pytest.approx(0.0 + out["windup_root_deg"])
    assert wing.twist_tip_deg == pytest.approx(-2.0 + out["windup_tip_deg"])
    assert out["windup_tip_deg"] > 0.05           # something to get wrong
    # the same wing flown RIGIDLY, by handing its twists to the design vector,
    # must produce the same aerodynamics — which is what says the reported
    # numbers are the twisted wing's and not the vector's
    rigid = evaluate_car_wing(
        _x(alpha=4.0, tw_root=wing.twist_root_deg,
           tw_tip=wing.twist_tip_deg), CarWingProblem())
    assert rigid["CZ"] == pytest.approx(out["CZ"], rel=1e-12)
    assert rigid["CDi"] == pytest.approx(out["CDi"], rel=1e-12)
