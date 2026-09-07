"""A blend BLENDS: the wing and the endplate have to meet.

The car endplate is the one tip device in the package that is a genuinely
DIFFERENT surface from the wing it hangs on — its own chord (up to 3x the
wing's tip chord), its own symmetric section, its own toe. All three used to
change in ONE STEP at the junction panel while the quarter-chord line turned
smoothly through the blend, so a "blended" corner joined a 0.21 m chord to a
0.62 m one: the leading edge jumped 0.10 m forward and the trailing edge
0.31 m aft, across a corner the junction model was simultaneously giving a
fillet credit for.

What this file holds:

  1. THE RAMP. vlm.transition_ramp is the turn angle normalised by the cant,
     so the surface finishes becoming the device exactly where it finishes
     turning into it — and a sharp corner (no turning arc) is the step, which
     is why blend_frac = 0 is bit-for-bit the published problem.
  2. THE CHORD MEETS. With a blend the device's first panel carries the wing's
     tip chord, not the plate's.
  3. THE AEROFOIL MEETS TOO. The section (a, alpha_L0) and the toe ramp on the
     same law — the plate's symmetric section used to appear in one step
     against the wing's cambered one.
  4. WHAT IS FLOWN IS WHAT IS CHARGED. Every reduced-order model that asks for
     "the plate's chord" asks the flown geometry at its own station: the drag
     build-up the mean, the junction charge the corner, the beam its built-in
     end. And t/c is what is held through the ramp, not thickness — a blend
     lofts the section with the chord.
  5. THE SPAN BAND BOUNDS THE CAR. A blended plate reaches OUTBOARD, so the
     wing pays for its projection out of its own span: a regulation measures
     the car, not the wing.
"""

from dataclasses import replace

import numpy as np
import pytest

from aerobo import endplate as ep
from aerobo import geometry, junction
from aerobo.geometry import Wing
from aerobo.vlm import VLM, transition_ramp

#: a mid-box design with the WIDEST plate the family offers (chord ratio 3),
#: which is where the step was worst
X = np.array([0.7, 0.0, -2.0, 6.0, 0.45, 0.55, 3.0, 0.12, 0.0, 1.6])


def _out(blend, shape="spiral", mount="tips", **kw):
    prob = replace(ep.CarWingEndplateProblem(), blend_frac=blend,
                   blend_shape=shape, mount=mount, **kw)
    out = ep.evaluate_car_wing_endplate(X, prob)
    assert out["feasible"], out["reason"]
    return out


def _model(blend, shape="spiral", scale=3.0, toe=4.0, cant=90.0,
           wing_blend=0.0):
    wing = Wing(b=1.6, S=0.4, taper=0.7, twist_root_deg=0.0,
                twist_tip_deg=-2.0)
    return VLM(wing, N=40, winglet_h_frac=0.5625, winglet_cant_deg=cant,
               n_winglet=10, winglet_blend_frac=blend,
               winglet_blend_shape=shape,
               winglet_wing_blend_frac=wing_blend,
               winglet_chord_scale=scale, winglet_toe_deg=toe,
               a=2 * np.pi, alpha_L0=-0.05,
               winglet_a=2 * np.pi, winglet_alpha_L0=0.0)


def _sides(m):
    """(wing panels, device panels) of the starboard side, root to tip."""
    star = np.where(m.y > 0.0)[0]
    return ([i for i in star if not m.is_winglet[i]],
            [i for i in star if m.is_winglet[i]])


def _tip_chord(m):
    """The chord AT the tip — the last wing PANEL sits inboard of it."""
    return float(m.wing.chord(np.array([m.wing.b / 2.0]))[0])


# --------------------------------------------------------------- 1. the ramp

def test_a_sharp_corner_is_a_step_and_a_blend_is_a_ramp():
    s = np.array([-0.3, -0.05, 0.01, 0.1, 0.2, 0.4])
    device = s > 0.0

    sharp = transition_ramp(s, device, h=0.45, cant_deg=90.0, blend_frac=0.0,
                            blend_shape="spiral", wing_arc=0.0)
    assert np.array_equal(sharp, device.astype(float))     # exactly the step

    ramp = transition_ramp(s, device, h=0.45, cant_deg=90.0, blend_frac=0.8,
                           blend_shape="spiral", wing_arc=0.0)
    assert np.all(ramp[~device] == 0.0)          # nothing turns before the turn
    assert np.all(np.diff(ramp) >= 0.0)          # monotone through the turn
    assert 0.0 < ramp[2] < ramp[3] < ramp[4] < 1.0
    assert ramp[-1] == pytest.approx(1.0)        # complete once the turn is


def test_the_ramp_is_the_turn_angle_normalised_by_the_cant():
    """Not a shape of its own — the transition's own law, so the surface stops
    being the wing exactly where it stops pointing like it."""
    s = np.linspace(-0.2, 0.45, 41)
    for shape in geometry.BLEND_SHAPES:
        w = transition_ramp(s, s > 0, h=0.45, cant_deg=90.0, blend_frac=0.8,
                            blend_shape=shape, wing_arc=0.0)
        psi = geometry.winglet_turn_angle(s, 0.45, 90.0, 0.8, shape, 0.0)
        expect = np.where(s > 0.0, psi / np.deg2rad(90.0), 0.0)
        assert w == pytest.approx(expect)


def test_a_wing_side_blend_starts_the_ramp_inboard_of_the_tip():
    """The turn starts on the WING there, so that is where the surface starts
    becoming the device — and panels further inboard are still pure wing."""
    s = np.array([-0.30, -0.10, -0.02, 0.05, 0.30])
    w = transition_ramp(s, s > 0.0, h=0.45, cant_deg=90.0, blend_frac=0.5,
                        blend_shape="spiral", wing_arc=0.15)
    assert w[0] == 0.0                    # outside the transition entirely
    assert 0.0 < w[2] < 1.0               # on the wing, already turning
    assert w[-1] == pytest.approx(1.0)


def test_a_coplanar_device_has_no_turn_to_ramp_on():
    s = np.array([-0.1, 0.1, 0.3])
    w = transition_ramp(s, s > 0.0, h=0.45, cant_deg=0.0, blend_frac=0.7,
                        blend_shape="arc", wing_arc=0.0)
    assert np.array_equal(w, (s > 0.0).astype(float))


# ------------------------------------------------- 2/3. the surfaces now meet

def test_without_a_blend_the_step_is_exactly_the_published_one():
    m = _model(0.0)
    _, dev_i = _sides(m)
    c_tip = _tip_chord(m)
    assert m.c[dev_i[0]] == pytest.approx(3.0 * c_tip)      # the step, intact
    assert np.allclose(m.c[dev_i], 3.0 * c_tip)             # a rectangle
    assert np.allclose(m.blend_ramp[dev_i], 1.0)


@pytest.mark.parametrize("blend", [0.3, 0.5, 1.0])
def test_a_blended_plate_leaves_the_wing_at_the_wings_own_chord(blend):
    m = _model(blend)
    _, dev_i = _sides(m)
    c_tip = _tip_chord(m)
    # the first device panel sits just outboard of the junction, where the
    # ramp has barely started: it carries the WING's chord, not the plate's
    assert m.c[dev_i[0]] == pytest.approx(c_tip, rel=0.05)
    assert m.c[dev_i[-1]] == pytest.approx(3.0 * c_tip, rel=1e-3)
    assert np.all(np.diff(m.c[dev_i]) >= -1e-12)            # monotone growth


@pytest.mark.parametrize("blend", [0.3, 1.0])
def test_the_aerofoil_and_the_toe_blend_with_the_chord(blend):
    """The plate's SECTION is the other half of the discontinuity: a symmetric
    alpha_L0 = 0 arriving in one step against the wing's cambered one, with
    the toe on top of it."""
    sharp = _model(0.0)
    m = _model(blend)
    _, dev_i = _sides(m)
    _, dev_sharp = _sides(sharp)

    # sharp: the device is fully itself from the first panel
    assert np.allclose(sharp.alpha_L0_panel[dev_sharp], 0.0)
    assert np.allclose(sharp.twist[dev_sharp],
                       sharp.twist[dev_sharp][0])

    # blended: it arrives at the wing's section and toe, and reaches its own
    assert m.alpha_L0_panel[dev_i[0]] == pytest.approx(-0.05, rel=0.1)
    assert m.alpha_L0_panel[dev_i[-1]] == pytest.approx(0.0, abs=1e-4)
    assert np.all(np.diff(m.alpha_L0_panel[dev_i]) >= -1e-12)
    toe = np.deg2rad(4.0)
    assert m.twist[dev_i[-1]] - m.twist[dev_i[0]] == pytest.approx(
        toe, rel=0.1)


def test_the_wing_itself_is_untouched_by_the_ramp():
    """Nothing inboard of the junction moves — the device's chord scale,
    section and toe are the device's."""
    sharp, blended = _model(0.0), _model(1.0)
    wing_sharp, _ = _sides(sharp)
    wing_blend, _ = _sides(blended)
    assert np.allclose(sharp.c[wing_sharp], blended.c[wing_blend])
    assert np.allclose(sharp.alpha_L0_panel[wing_sharp], blended.alpha_L0_panel[wing_blend])
    assert np.allclose(sharp.twist[wing_sharp], blended.twist[wing_blend])


# ------------------------------------- 4. what is flown is what is charged

def test_with_no_blend_every_reported_chord_is_the_plates_chord():
    out = _out(0.0)
    c_ep = out["endplate_chord_m"]
    for k in ("endplate_chord_mean_m", "endplate_chord_corner_m",
              "endplate_chord_root_m"):
        assert out[k] == pytest.approx(c_ep)
    assert out["endplate_chord_step"] == pytest.approx(3.0)
    assert out["endplate_projection_m"] == pytest.approx(0.0, abs=1e-12)
    assert out["overall_width_m"] == pytest.approx(out["b_m"])


@pytest.mark.parametrize("blend", [0.3, 1.0])
def test_a_blend_reports_the_chords_it_actually_flew(blend):
    out = _out(blend)
    c_ep = out["endplate_chord_m"]
    assert out["endplate_chord_step"] == pytest.approx(1.0)   # they meet
    assert out["endplate_chord_corner_m"] < out["endplate_chord_mean_m"] < c_ep
    # THE PLATE IS ALWAYS BUILT IN AT THE DECK, whichever layout carries: it
    # is bolted to the car in both, so its beam root is the far end, whose
    # chord the ramp never touched. There used to be a hung-off-the-wing
    # branch, rooted at the junction chord the ramp DID touch — it belonged to
    # the pylon layout, where the plate carried nothing and was free to be a
    # short fence. With no pylon layout there is no such plate, and leaving
    # that branch reachable would understate the bending a carrying plate does.
    for mount in ("tips", "inboard"):
        got = _out(blend, mount=mount)
        assert got["endplate_chord_root_m"] == pytest.approx(
            got["endplate_chord_m"], rel=1e-3), mount
        assert (got["endplate_chord_root_m"]
                > got["endplate_chord_corner_m"]), mount


@pytest.mark.parametrize("blend", [0.0, 0.3, 1.0])
def test_the_ramp_holds_the_thickness_RATIO_not_the_thickness(blend):
    """A blend lofts the section with the chord. Holding the thickness in
    metres instead would leave the transition a wedge — at full blend a 0.31 m
    chord carrying the plate's 0.11 m thickness is t/c = 0.36, outside the
    junction correlation's own validity band, and the corner would be charged
    as a strut rather than credited as a fillet."""
    out = _out(blend)
    tc = out["endplate_tc"]
    assert out["endplate_t_corner_m"] == pytest.approx(
        tc * out["endplate_chord_corner_m"])
    assert out["endplate_t_root_m"] == pytest.approx(
        tc * out["endplate_chord_root_m"])
    assert junction.TC_VALID[0] <= tc <= junction.TC_VALID[1]
    assert out["junction"]["tc_in_correlation_band"]


def test_the_junction_charge_is_taken_at_the_corners_own_chord():
    out = _out(0.6)
    j = out["junction"]
    expect = junction.report(out["endplate_tc"],
                             out["endplate_chord_corner_m"], out["S_m2"],
                             out["endplate_h_m"], 90.0, 0.6, n_junctions=2,
                             blend_shape=out["endplate_blend_shape"])
    assert j["CD_junction"] == pytest.approx(expect["CD_junction"])


# ---------------------------------------------- 5. the band bounds the car

@pytest.mark.parametrize("blend", [0.0, 0.2, 0.5, 1.0])
def test_the_span_band_bounds_the_whole_car_not_just_the_wing(blend):
    """A plate at cant 90 projects nothing, but a BLENDED one leaves the wing
    plane tangentially and reaches outboard — 0.27 m a side at full blend,
    which would put a 1.6 m wing 2.15 m wide inside a 2.0 m box."""
    out = _out(blend)
    assert out["overall_width_m"] == pytest.approx(X[9])       # the band's
    proj = out["endplate_projection_m"]
    assert proj == pytest.approx(
        geometry.winglet_projection(out["endplate_h_m"], 90.0, blend,
                                    out["endplate_blend_shape"]))
    assert out["b_m"] == pytest.approx(X[9] - 2.0 * proj)
    if blend > 0.0:
        assert out["b_m"] < X[9]                # the wing paid for its plates


def test_blending_is_no_longer_a_free_lunch():
    """Before the projection was counted and the chord ramped, CZ/CD climbed
    monotonically to +43% at full blend against a junction credit that had
    saturated at blend ~0.2 — nothing in the model opposed curling the plate
    into a quarter-round winglet. It has a cost now: the wing shortens to pay
    for the projection, so the efficiency turns over inside the range."""
    eff = [_out(b)["efficiency"] for b in (0.0, 0.2, 0.5, 1.0)]
    assert eff[1] > eff[0]                     # a real corner is worth softening
    assert eff[-1] < max(eff)                  # ...and full blend is not free
    assert _out(1.0)["CDi"] > 1.5 * _out(0.0)["CDi"]


def test_a_plate_that_eats_the_whole_width_is_a_stated_refusal():
    """Not an exception at the optimiser and not a silent clamp: the penalty
    contract, with a reason a reader can act on."""
    x = X.copy()
    x[4], x[9] = 0.60, 0.50              # tallest plate, narrowest car
    prob = replace(ep.CarWingEndplateProblem(), blend_frac=1.0,
                   blend_shape="spiral", span_bounds_m=(0.2, 2.0))
    out = ep.evaluate_car_wing_endplate(x, prob)
    assert not out["feasible"]
    assert "projection" in out["reason"]
    assert out["g"] == [ep.G_FAIL] * 4


# ----------------------------------------------------------------- the card

def test_the_card_can_only_ask_the_blend_a_question_the_model_can_answer():
    from gui.nice_app import _opt_float_in

    assert _opt_float_in(None, 0.0, 1.0) is None      # blank stays "default"
    assert _opt_float_in("", 0.0, 1.0) is None
    assert _opt_float_in(0.4, 0.0, 1.0) == pytest.approx(0.4)
    assert _opt_float_in(1.5, 0.0, 1.0) == pytest.approx(1.0)
    assert _opt_float_in(-0.2, 0.0, 1.0) == pytest.approx(0.0)
    # unbounded rows are untouched — most cards ask lengths, not fractions
    assert _opt_float_in(7.5) == pytest.approx(7.5)


def test_the_blend_row_states_its_range_and_carries_the_bounds():
    """The range is in the label as well as on the field: a silently clamped
    number is its own surprise.

    Asserted on what the card DRAWS, not on its source text. Reading the range
    out of ``inspect.getsource`` passed for a label that had been commented
    out, and it fails for a label that is spelt with an escape — neither of
    which is the question. What matters is that a user looking at the field
    sees the range, and that a number outside it never reaches the solver.
    """
    from gui.v3.app import assemble

    ctx = assemble("track")
    ctx.S["wing"]["choices"]["car_endplates"] = True
    ctx.render("wing", "type")
    view = ctx.views[("wing", "type")]
    texts = []
    for e in view.descendants():
        t = getattr(e, "text", "") or ""
        if t:
            texts.append(t)
    labels = [t for t in texts if "Root blend" in t]
    assert labels, texts[:40]
    assert "0" in labels[0] and "1" in labels[0], labels

    # ...and the field REFUSES what it says it refuses rather than clamping:
    # 1.5 used to reach geometry.span_path and come back as a whole run of
    # "solver failure: blend_frac must be in [0, 1]" penalties
    # a number outside the range never reaches the solver: the field clamps
    # it into the band it advertises, so geometry.span_path is never asked a
    # question it can only answer with a ValueError
    from gui.nice_app import _opt_float_in

    assert _opt_float_in(1.5, 0.0, 1.0) == pytest.approx(1.0)
    assert _opt_float_in(-0.2, 0.0, 1.0) == pytest.approx(0.0)
    assert _opt_float_in(0.4, 0.0, 1.0) == pytest.approx(0.4)


def test_at_full_blend_the_plate_only_becomes_the_plate_at_its_tip():
    """blend = 1 spends the WHOLE height turning, so the ramp completes only at
    the extreme tip: the surface is a quarter round that is still part wing
    most of the way up, and its nominal chord and section are reached nowhere
    but the last station. Worth stating, because "1" reads like "fully
    blended" and what it means is "never finished arriving"."""
    m = _model(1.0)
    _, dev_i = _sides(m)
    c_tip = _tip_chord(m)
    half = m.blend_ramp[dev_i][len(dev_i) // 2]
    assert 0.3 < half < 0.7                        # halfway up, halfway there
    assert m.c[dev_i[-1]] < 3.0 * c_tip            # short of the plate's chord
    assert m.c[dev_i[-1]] == pytest.approx(3.0 * c_tip, rel=1e-3)
    # ...and the height it buys is not the height it spends: a quarter round
    # reaches 61% as far down as the same arc spent going straight
    reach = geometry.winglet_tip_height(0.45, 90.0, 1.0, "spiral")
    assert reach == pytest.approx(0.61 * 0.45, rel=0.02)
