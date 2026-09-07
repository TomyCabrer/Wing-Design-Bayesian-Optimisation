"""Wing/winglet transition that starts INBOARD of the tip, and the clothoid.

Why these exist
---------------
The blended winglet shipped with its turn confined to the DEVICE: the turning
arc is ``blend_frac * h`` with ``h <= 0.15 * b/2``, so the blend radius can
never exceed ~0.9 tip chords, and at the blend fraction the span-capped
optimiser actually settles on (~0.15) it is ~0.13 tip chords. That is a
corner, and no turn law can draw it as anything else — which is why picking
the "smooth" shape did not visibly smooth the junction.

Two things follow, and both are pinned here:

  1. ``geometry.span_path`` — the transition may start on the WING. The
     spanwise line is then one curve parameterised by DEVELOPED ARC LENGTH,
     so the invariant of the whole family is arc length: a wing-side blend
     trades PROJECTED span for height and never deletes wing, area or chord.
  2. ``"spiral"`` — a clothoid (Euler-spiral) turn law. Crease-free like the
     smoothstep, but its peak curvature is 4/3 of the mean rather than 3/2,
     so its tightest point is a third gentler at the same arc and cant.

And, as always: everything with the wing-side blend switched off is
bit-for-bit the published geometry.
"""

import numpy as np
import pytest

from aerobo import api, geometry, junction
from aerobo.geometry import (
    Wing,
    developed_semispan,
    span_path,
    span_projection,
    wing_from_x,
    winglet_curvature,
    winglet_path,
    winglet_turn_angle,
)
from aerobo.objective import Problem, evaluate
from aerobo.vlm import VLM

B = 10.0
SEMI = B / 2.0
H = 0.75            # winglet arc length [m] — the top of the h_frac band
CANT = 75.0
BETA = 0.6
W = 0.10            # wing-side blend: 10% of the semi-span


# ------------------------------------------------------------ the clothoid

@pytest.mark.parametrize("shape", geometry.BLEND_SHAPES)
def test_every_shape_reaches_exactly_the_cant_and_holds_it(shape):
    s_out = BETA * H
    psi = winglet_turn_angle(np.array([s_out, H]), H, CANT, BETA, shape)
    assert psi[0] == pytest.approx(np.deg2rad(CANT), rel=1e-14)
    assert psi[1] == pytest.approx(psi[0], rel=1e-14)


def test_the_clothoid_is_crease_free_but_gentler_than_the_smoothstep():
    """Both G2 laws start and end at zero curvature; the clothoid's PEAK is
    4/3 of the mean against the smoothstep's 3/2 — that ratio IS the price a
    crease-free junction pays in local tightness, and it is what makes the
    clothoid the shape to draw a short blend with."""
    s_out = BETA * H
    t = np.linspace(0.0, s_out, 20_001)
    k_arc = winglet_curvature(np.array([0.5 * s_out]), H, CANT, BETA, "arc")[0]
    for shape, ratio in (("smooth", 1.5), ("spiral", 4.0 / 3.0)):
        k = winglet_curvature(t, H, CANT, BETA, shape)
        assert k[0] == pytest.approx(0.0, abs=1e-15)
        assert k[-1] == pytest.approx(0.0, abs=1e-15)
        assert k.max() == pytest.approx(ratio * k_arc, rel=1e-4)
    assert junction.blend_radius_min(H, CANT, BETA, "spiral") > \
        junction.blend_radius_min(H, CANT, BETA, "smooth")


def test_the_clothoid_curvature_is_continuous_along_the_whole_device():
    t = np.linspace(0.0, H, 4001)
    k = winglet_curvature(t, H, CANT, BETA, "spiral")
    assert np.max(np.abs(np.diff(k))) < 0.01 * k.max()


def test_the_clothoid_ramp_fraction_spans_the_two_published_laws():
    """r -> 0 is the circular fillet's peak (creases back), r = 1/3 is the
    smoothstep's — so the constant is a position on a known scale, not a
    magic number."""
    assert 0.0 < geometry.SPIRAL_RAMP_FRAC < 0.5
    u = np.linspace(0.0, 1.0, 5001)
    peak = geometry._turn_law_deriv(u, "spiral").max()
    assert peak == pytest.approx(1.0 / (1.0 - geometry.SPIRAL_RAMP_FRAC),
                                 rel=1e-9)
    assert 1.0 < peak < 1.5


# ---------------------------------------------- arc length is the invariant

@pytest.mark.parametrize("shape", geometry.BLEND_SHAPES)
@pytest.mark.parametrize("w", [0.0, 0.05, 0.15])
def test_the_developed_span_is_conserved_exactly(shape, w):
    """The whole point: a blend moves projected span into height, it never
    deletes wing. Measured on the drawn curve, wing AND winglet."""
    s = np.linspace(0.0, SEMI + H, 400_001)
    y, z = span_path(s, SEMI, H, CANT, BETA, shape, w)
    length = float(np.sum(np.hypot(np.diff(y), np.diff(z))))
    assert length == pytest.approx(SEMI + H, rel=1e-9)


@pytest.mark.parametrize("shape", geometry.BLEND_SHAPES)
def test_a_wing_side_blend_costs_projected_span_and_buys_height(shape):
    flat = span_projection(SEMI, H, CANT, BETA, shape, 0.0)
    bent = span_projection(SEMI, H, CANT, BETA, shape, W)
    assert bent < flat                       # the outer wing turns inboard-up
    z_flat = span_path(np.array([SEMI + H]), SEMI, H, CANT, BETA, shape,
                       0.0)[1][0]
    z_bent = span_path(np.array([SEMI + H]), SEMI, H, CANT, BETA, shape,
                       W)[1][0]
    assert z_bent > z_flat                   # ...and reaches higher for it


def test_the_path_does_not_depend_on_how_it_is_sampled():
    coarse = np.linspace(0.0, SEMI + H, 9)
    fine = np.linspace(0.0, SEMI + H, 801)
    for shape in geometry.BLEND_SHAPES:
        y_c, z_c = span_path(coarse, SEMI, H, CANT, BETA, shape, W)
        y_f, z_f = span_path(fine, SEMI, H, CANT, BETA, shape, W)
        assert np.allclose(y_c, y_f[::100], rtol=0, atol=1e-14)
        assert np.allclose(z_c, z_f[::100], rtol=0, atol=1e-14)


def test_the_wing_is_flat_until_the_turn_starts_and_the_turn_is_tangential():
    s0 = SEMI - W * SEMI
    s = np.array([0.0, 0.5 * s0, s0])
    _, z = span_path(s, SEMI, H, CANT, BETA, "spiral", W)
    assert np.allclose(z, 0.0, atol=1e-15)
    # leaving tangentially with zero curvature: z ~ s^3 just past the start
    d = np.array([1e-4, 2e-4])
    _, z2 = span_path(s0 + d, SEMI, H, CANT, BETA, "spiral", W)
    assert z2[1] / z2[0] == pytest.approx(8.0, rel=1e-3)


# --------------------------------------------------------- the radius grows

def test_the_wing_side_blend_is_what_makes_the_radius_a_real_one():
    """The complaint this whole feature answers: confined to the winglet the
    radius is a fraction of a tip chord; let the wing turn and it reaches the
    order of a chord, which is what a blended winglet actually looks like."""
    c_tip = float(Wing(b=B, S=10.0, taper=0.5).chord(np.array([SEMI]))[0])
    r_device = junction.blend_radius(H, CANT, 0.15)          # the BO optimum
    r_wing = junction.blend_radius(H, CANT, 0.15, wing_arc=0.15 * SEMI)
    assert r_device / c_tip < 0.2                            # a corner
    assert r_wing / c_tip > 0.9                              # a blend
    # ...and the CORRELATION cannot tell them apart: the fillet credit is
    # already at its floor by R/c = FILLET_FULL_R_OVER_C, which the
    # device-only radius passes. That is exactly why the optimiser settles on
    # a barely-blended corner in the 6-D family — past saturation more blend
    # is pure projected-span cost. What a wing-side blend buys is therefore
    # NOT junction drag but WAKE GEOMETRY, and only the solve can see it
    # (test_the_wing_side_blend_changes_the_solve_not_just_the_picture).
    assert junction.fillet_credit(r_device, c_tip) == \
        pytest.approx(junction.FILLET_CREDIT_FLOOR)
    assert junction.fillet_credit(r_wing, c_tip) == \
        pytest.approx(junction.FILLET_CREDIT_FLOOR)


def test_the_wing_side_blend_changes_the_solve_not_just_the_picture():
    """Bending the outer wing up is a nonplanar change the Trefftz plane sees
    on its own account, so it moves L/D even with the junction correlation
    pinned at its floor."""
    prob = dict(mode="winglet_capped_blended", junction_drag=True,
                blend_shape="spiral")
    x = np.array([0.5, 2.0, -2.0, 0.15, CANT, 0.15])
    flat = evaluate(x, Problem(**prob))
    bent = evaluate(x, Problem(**prob, wing_blend_frac=0.15))
    assert flat["feasible"] and bent["feasible"]
    assert bent["LoD"] > flat["LoD"]
    assert bent["winglet"]["projected_semispan_m"] == \
        pytest.approx(flat["winglet"]["projected_semispan_m"], rel=1e-9)


# ------------------------------------------------------------- the span cap

@pytest.mark.parametrize("shape", geometry.BLEND_SHAPES)
@pytest.mark.parametrize("w", [0.0, 0.05, 0.15])
def test_the_cap_holds_the_projected_span_exactly(shape, w):
    """developed_semispan is a closed-form solve, not an approximation: the
    flat wing inboard of the turn projects 1:1, so the relation is affine."""
    arc = w * SEMI
    dev = developed_semispan(SEMI, H, CANT, BETA, shape, wing_arc=arc)
    got = span_projection(dev, H, CANT, BETA, shape,
                          arc / dev if dev > 0 else 0.0)
    assert got == pytest.approx(SEMI, rel=1e-12)


def test_the_capped_mode_shrinks_the_wing_for_the_wing_side_blend_too():
    x = np.array([0.5, 2.0, -2.0, 0.15, CANT, BETA, 0.15])
    wing = wing_from_x(x, b=B, S=10.0, mode="winglet_capped_blended_wing",
                       blend_shape="spiral")
    assert wing.b < B
    assert wing.S == 10.0                    # area is held, span is charged


# ---------------------------------------------------------- nothing regressed

@pytest.mark.parametrize("shape", ["arc", "smooth"])
@pytest.mark.parametrize("beta", [0.0, 0.3, 1.0])
def test_zero_wing_blend_is_the_published_geometry(shape, beta):
    """The wing is EXACTLY the flat line it always was — that identity is
    what keeps every published panel array bit-for-bit, since the VLM builds
    its winglet leg from winglet_path exactly as it always did. The leg
    itself agrees with winglet_path to machine precision (it is the same
    quadrature, evaluated at an arc offset by the semi-span, so only the
    rounding of that offset can differ)."""
    s = np.linspace(0.0, SEMI + H, 257)
    y, z = span_path(s, SEMI, H, CANT, beta, shape, 0.0)
    wing = s <= SEMI
    assert np.array_equal(y[wing], s[wing])
    assert np.array_equal(z[wing], np.zeros(int(wing.sum())))
    t = s[~wing] - SEMI
    dy, dz = winglet_path(t, H, CANT, beta, shape)
    assert np.allclose(y[~wing], SEMI + dy, rtol=0, atol=1e-14)
    assert np.allclose(z[~wing], dz, rtol=0, atol=1e-14)


def test_the_vlm_panels_are_unchanged_without_a_wing_side_blend():
    wing = Wing(b=B, S=10.0, taper=0.5, twist_root_deg=0.0, twist_tip_deg=-2.0)
    kw = dict(N=24, winglet_h_frac=0.15, winglet_cant_deg=CANT, n_winglet=8,
              winglet_blend_frac=BETA, winglet_blend_shape="arc")
    a = VLM(wing, **kw)
    b = VLM(wing, **kw, winglet_wing_blend_frac=0.0)
    for name in ("y", "z", "c", "width"):
        assert np.array_equal(getattr(a, name), getattr(b, name))


def test_a_dropped_winglet_drops_its_wing_side_transition_too():
    """No device to finish the turn on -> the wing must stay planar, or a
    sub-MIN_WINGLET_FRAC request would bend the tip up for nothing."""
    wing = Wing(b=B, S=10.0, taper=0.5)
    m = VLM(wing, N=24, winglet_h_frac=0.0, winglet_cant_deg=CANT,
            n_winglet=8, winglet_blend_frac=BETA,
            winglet_blend_shape="spiral", winglet_wing_blend_frac=0.15)
    assert np.allclose(m.z, 0.0)
    assert not m.is_winglet.any()


# ------------------------------------------------------------- the plumbing

def test_the_objective_flies_the_wing_side_blend_and_reports_it():
    prob = Problem(mode="winglet_capped_blended_wing", junction_drag=True,
                   blend_shape="spiral")
    x = np.array([0.5, 2.0, -2.0, 0.15, CANT, 0.3, 0.12])
    out = evaluate(x, prob)
    assert out["feasible"], out["reason"]
    wl = out["winglet"]
    assert wl["wing_blend_frac"] > 0.0
    assert wl["wing_blend_arc_m"] == pytest.approx(
        wl["wing_blend_frac"] * out["vlm"].S / out["vlm"].S * wl["h_m"]
        / max(wl["h_frac"], 1e-30), rel=1e-9)     # both are fractions of b/2
    # the cap is honoured on the WHOLE developed line
    assert wl["projected_semispan_m"] == pytest.approx(SEMI, rel=1e-9)
    assert wl["junction"]["blend_wing_arc_m"] > 0.0
    assert wl["junction"]["blend_radius_m"] > junction.blend_radius(
        wl["h_m"], CANT, 0.3)


def test_the_wing_side_blend_is_a_flag_on_the_six_d_family():
    """Same knob, other tier: a VALUE on the published blended problem (its
    vector is untouched) and a DESIGN VARIABLE on the seven-D one."""
    six = Problem(mode="winglet_capped_blended", junction_drag=True,
                  blend_shape="spiral", wing_blend_frac=0.12)
    assert six.dim == 6
    seven = Problem(mode="winglet_capped_blended_wing", junction_drag=True,
                    blend_shape="spiral")
    assert seven.dim == 7
    x6 = np.array([0.5, 2.0, -2.0, 0.15, CANT, 0.3])
    flat = evaluate(x6, Problem(mode="winglet_capped_blended",
                                junction_drag=True, blend_shape="spiral"))
    bent = evaluate(x6, six)
    assert flat["feasible"] and bent["feasible"]
    assert bent["LoD"] != flat["LoD"]
    # ...and the seven-D family IGNORES the field, because its vector says it
    ignored = Problem(mode="winglet_capped_blended_wing", junction_drag=True,
                      blend_shape="spiral", wing_blend_frac=0.12)
    x7 = np.array([*x6, 0.0])
    assert evaluate(x7, ignored)["LoD"] == pytest.approx(flat["LoD"],
                                                         rel=1e-12)


def test_the_registry_offers_it_and_the_gui_maps_it_both_ways():
    import gui.nice_app as v1

    name = "winglet, blended into the wing (span-capped)"
    spec = api.PROBLEM_SPECS[name]
    assert spec.param_labels[-1] == "wing_blend_frac"
    built = spec.build({}, {}, None)
    assert built.dim == 7
    assert v1.derive_problem(v1.choices_from_problem(name))[0] == name
    pretty, tip = v1.param_help("wing_blend_frac")
    assert pretty and tip
    # and it composes with the modifiers like every other family
    assert api.add_modifier(name, "chord") in api.PROBLEM_SPECS


def test_the_shape_flag_reaches_the_problem_and_bad_names_are_refused():
    built = api.PROBLEM_SPECS["winglet, blended (span-capped)"].build(
        {}, {"blend_shape": "spiral", "wing_blend_frac": 0.1}, None)
    assert built.problem.blend_shape == "spiral"
    assert built.problem.wing_blend_frac == pytest.approx(0.1)
    with pytest.raises(ValueError):
        Problem(mode="winglet_capped_blended", blend_shape="clothoid")
    with pytest.raises(ValueError):
        Problem(mode="winglet_capped_blended", wing_blend_frac=0.9)
