"""Curvature-continuous wing/winglet transition (geometry.BLEND_SHAPES).

The circular fillet the blended winglet shipped with is C1 but not C2: its
curvature JUMPS from 0 to 1/R where the wing ends and back to 0 where the
winglet straightens, so both ends of the "blend" are still creases. The
``smooth`` shape turns through the same cant over the same arc length with a
smoothstep turn law, whose curvature rises from zero and returns to zero —
the genuinely smooth horizontal-to-vertical transition.

What is pinned here:
  1. the TURN LAW (both shapes reach the same cant, hold it after, and the
     smooth one starts and ends with zero curvature),
  2. arc length still conserved EXACTLY (the path is unit-speed by
     construction, so a smooth and a sharp device of equal h have equal
     wetted area — the whole comparison rests on it),
  3. sampling independence (the quadrature grid is internal),
  4. the legacy path: blend_frac = 0, or blend_shape = "arc", is bit-for-bit
     what shipped,
  5. the pricing: junction.py's mean radius is shape-INDEPENDENT (the
     correlation cannot see the difference), while the reported minimum
     radius is 2/3 of it for the smooth law,
  6. the plumbing: VLM panels, the span cap, the objective breakdown, the
     api flag and the GUI builder all carry the shape.
"""

import numpy as np
import pytest

from aerobo import api, geometry, junction
from aerobo.geometry import (
    Wing,
    wing_from_x,
    winglet_curvature,
    winglet_path,
    winglet_projection,
    winglet_tip_height,
    winglet_turn_angle,
)
from aerobo.objective import Problem, evaluate
from aerobo.vlm import VLM

H = 0.75            # winglet arc length [m] on the b = 10 m baseline
CANT = 75.0
BETA = 0.6


# ------------------------------------------------------------ the turn law

@pytest.mark.parametrize("shape", ["arc", "smooth"])
def test_the_turn_reaches_exactly_the_cant_and_then_holds(shape):
    s_arc = BETA * H
    psi_end = winglet_turn_angle(np.array([s_arc]), H, CANT, BETA, shape)[0]
    assert psi_end == pytest.approx(np.deg2rad(CANT), rel=1e-14)
    after = winglet_turn_angle(np.array([s_arc, H]), H, CANT, BETA, shape)
    assert after[1] == pytest.approx(after[0], rel=1e-14)


def test_smooth_curvature_starts_and_ends_at_zero_while_the_arc_jumps():
    s_arc = BETA * H
    ends = np.array([0.0, s_arc])
    k_smooth = winglet_curvature(ends, H, CANT, BETA, "smooth")
    k_arc = winglet_curvature(ends, H, CANT, BETA, "arc")
    assert np.allclose(k_smooth, 0.0, atol=1e-15)      # no crease at either end
    assert np.all(k_arc > 0.0)                          # the fillet's step
    # ...and the smooth law pays for that with a tighter middle: peak
    # curvature is exactly 3/2 of the constant fillet's (max of 6u(1-u) = 3/2)
    t = np.linspace(0.0, s_arc, 20_001)
    assert winglet_curvature(t, H, CANT, BETA, "smooth").max() == \
        pytest.approx(1.5 * k_arc[0], rel=1e-6)


def test_smooth_curvature_is_continuous_along_the_whole_device():
    """No jump anywhere: |dk| between neighbouring stations stays O(dt)."""
    t = np.linspace(0.0, H, 4001)
    k = winglet_curvature(t, H, CANT, BETA, "smooth")
    k_arc = winglet_curvature(t, H, CANT, BETA, "arc")
    assert np.max(np.abs(np.diff(k))) < 0.01 * k.max()          # continuous
    assert np.max(np.abs(np.diff(k_arc))) == pytest.approx(
        k_arc.max(), rel=1e-12)          # the fillet jumps its whole value


# ------------------------------------------------------- geometry contract

@pytest.mark.parametrize("beta", [0.0, 0.25, 0.6, 1.0])
def test_smooth_conserves_arc_length_exactly(beta):
    t = np.linspace(0.0, H, 200_001)
    y, z = winglet_path(t, H, CANT, beta, "smooth")
    s = float(np.sum(np.hypot(np.diff(y), np.diff(z))))
    assert s == pytest.approx(H, rel=1e-7)


def test_smooth_leaves_the_wing_plane_tangentially_and_flatter_than_the_arc():
    """Tangency is shared; the smooth law additionally leaves with zero
    curvature, so its initial rise is a cubic, not a parabola: doubling t
    multiplies z by 8, not by 4."""
    t = np.array([1e-4, 2e-4])
    _, z = winglet_path(t, H, CANT, BETA, "smooth")
    assert z[1] / z[0] == pytest.approx(8.0, rel=1e-3)


def test_the_two_shapes_draw_genuinely_different_lines():
    t = np.linspace(0.0, H, 101)
    y_a, z_a = winglet_path(t, H, CANT, BETA, "arc")
    y_s, z_s = winglet_path(t, H, CANT, BETA, "smooth")
    assert not np.allclose(z_a, z_s)
    # the smooth law turns LATER, so it stays nearer the wing plane through
    # the transition and its tip ends up lower and less far outboard
    assert winglet_tip_height(H, CANT, BETA, "smooth") < \
        winglet_tip_height(H, CANT, BETA, "arc")
    assert winglet_projection(H, CANT, BETA, "smooth") < \
        winglet_projection(H, CANT, BETA, "arc")


def test_the_path_does_not_depend_on_how_it_is_sampled():
    """Same t, same point — whether 9 or 900 stations were asked for. The
    quadrature grid is internal for exactly this reason (the chord-law area
    factor is closed-form for the same reason)."""
    coarse = np.linspace(0.0, H, 9)
    fine = np.linspace(0.0, H, 801)      # 100 fine steps per coarse step
    y_c, z_c = winglet_path(coarse, H, CANT, BETA, "smooth")
    y_f, z_f = winglet_path(fine, H, CANT, BETA, "smooth")
    idx = np.searchsorted(fine, coarse)
    assert np.allclose(y_c, y_f[idx], atol=1e-14)
    assert np.allclose(z_c, z_f[idx], atol=1e-14)


@pytest.mark.parametrize("cant", [-60.0, 60.0, 90.0])
def test_signed_cant_turns_the_same_way_for_both_shapes(cant):
    y, z = winglet_path(np.array([H]), H, cant, 1.0, "smooth")
    assert np.sign(z) == np.sign(cant)
    assert y > 0.0


def test_zero_blend_is_the_legacy_straight_ray_for_every_shape():
    t = np.linspace(0.0, H, 17)
    for shape in geometry.BLEND_SHAPES:
        y, z = winglet_path(t, H, CANT, 0.0, shape)
        assert np.array_equal(y, t * np.cos(np.deg2rad(CANT)))
        assert np.array_equal(z, t * np.sin(np.deg2rad(CANT)))


def test_an_unknown_shape_is_refused():
    with pytest.raises(ValueError, match="blend_shape"):
        winglet_path(np.array([H]), H, CANT, BETA, "spline")


# --------------------------------------------------------------- pricing

def test_the_mean_blend_radius_is_shape_independent():
    """The fillet credit is a calibrated shape on the MEAN radius (total
    turn / turning length), which every law reaching the same cant over the
    same arc shares. Letting it separate the two would invent a distinction
    the correlation cannot make."""
    r_arc = junction.blend_radius(H, CANT, BETA)
    assert r_arc == pytest.approx(BETA * H / np.deg2rad(CANT), rel=1e-12)
    for shape in geometry.BLEND_SHAPES:
        rep = junction.report(0.12, 0.9, 10.0, H, CANT, BETA,
                              blend_shape=shape)
        assert rep["blend_radius_m"] == pytest.approx(r_arc, rel=1e-12)
        assert rep["blend_shape"] == shape


def test_the_reported_minimum_radius_separates_them():
    r_mean = junction.blend_radius(H, CANT, BETA)
    assert junction.blend_radius_min(H, CANT, BETA, "arc") == \
        pytest.approx(r_mean, rel=1e-3)
    assert junction.blend_radius_min(H, CANT, BETA, "smooth") == \
        pytest.approx(r_mean / 1.5, rel=1e-3)
    assert junction.blend_radius_min(H, CANT, 0.0, "smooth") == 0.0


# --------------------------------------------------------------- plumbing

def test_the_vlm_panels_follow_the_smooth_line():
    wing = Wing(b=10.0, S=10.0, taper=0.6)
    model = VLM(wing, N=20, winglet_h_frac=0.15, winglet_cant_deg=CANT,
                n_winglet=8, winglet_blend_frac=1.0,
                winglet_blend_shape="smooth")
    wl = model.is_winglet
    y, z = model.y[wl], model.z[wl]
    star = y > 0.0
    ys, zs = y[star] - wing.b / 2.0, z[star]     # relative to the wing tip
    h = 0.15 * wing.b / 2.0
    # the outermost station sits where the smooth path says it does
    y_end, z_end = winglet_path(np.array([h]), h, CANT, 1.0, "smooth")
    assert ys.max() < y_end[0] + 1e-9
    assert zs.max() < z_end[0] + 1e-9
    # and lower than the constant-radius fillet would have put it
    _, z_arc = winglet_path(np.array([h]), h, CANT, 1.0, "arc")
    assert zs.max() < z_arc[0]


def test_the_span_cap_charges_the_smooth_projection():
    """Capping on the arc's projection would hand the smooth blend free
    projected span (the raked-tip trap again, one shape further down)."""
    x = np.array([0.6, 1.0, -2.0, 0.15, CANT, 1.0])
    w_arc = wing_from_x(x, mode="winglet_capped_blended", blend_shape="arc")
    w_smooth = wing_from_x(x, mode="winglet_capped_blended",
                           blend_shape="smooth")
    h = 0.15 * 10.0 / 2.0
    assert w_smooth.b == pytest.approx(
        10.0 - 2.0 * winglet_projection(h, CANT, 1.0, "smooth"), rel=1e-12)
    assert w_smooth.b > w_arc.b        # it reaches less far outboard


def test_the_objective_carries_the_shape_and_stays_bit_for_bit_at_zero_blend():
    x = np.array([0.6, 1.0, -2.0, 0.12, CANT, 0.0])
    arc = evaluate(x, Problem(mode="winglet_capped_blended"))
    smooth = evaluate(x, Problem(mode="winglet_capped_blended",
                                 blend_shape="smooth"))
    assert smooth["LoD"] == pytest.approx(arc["LoD"], rel=1e-15)

    # free span here, so BOTH shapes fly the same wing (the span cap would
    # hand them different tip chords, and the junction charge is a function
    # of the local chord as well as the radius)
    x_blend = np.array([0.6, 1.0, -2.0, 0.12, CANT, 0.8])
    arc_b = evaluate(x_blend, Problem(mode="winglet_blended",
                                      junction_drag=True))
    smooth_b = evaluate(x_blend, Problem(mode="winglet_blended",
                                         junction_drag=True,
                                         blend_shape="smooth"))
    assert arc_b["feasible"] and smooth_b["feasible"]
    assert smooth_b["winglet"]["blend_shape"] == "smooth"
    assert smooth_b["LoD"] != arc_b["LoD"]
    # the junction charge is the same (same mean radius) — the difference is
    # the wake the two shapes draw, which is the only thing the VLM can see
    assert smooth_b["CD_junction"] == pytest.approx(arc_b["CD_junction"],
                                                    rel=1e-12)


def test_an_unknown_shape_is_refused_by_the_problem():
    with pytest.raises(ValueError, match="blend_shape"):
        Problem(mode="winglet_capped_blended", blend_shape="clothoid")


def test_the_api_flag_reaches_the_problem():
    spec = api.PROBLEM_SPECS["winglet, blended (span-capped)"]
    assert "blend_shape" in spec.flags
    built = spec.build({}, {"blend_shape": "smooth"}, None)
    assert built.problem.blend_shape == "smooth"
    assert spec.build({}, {}, None).problem.blend_shape == "arc"   # legacy


def test_the_builder_sends_the_shape_for_the_blended_menu_entry():
    from gui import nice_app as v1

    ch = dict(v1.BUILDER_DEFAULTS, winglets="capped", winglet_type="blended")
    assert v1.derive_problem(ch)[0] == "winglet, blended (span-capped)"
    # the GUI defaults the blended winglet to a CREASE-FREE transition (that
    # is what a blend is drawn for) and says so explicitly, rather than
    # relying on the library default, which stays "arc" for bit-for-bit
    # reasons. Of the two G2 laws it picks the CLOTHOID: the smoothstep buys
    # its crease-free ends with a 3/2 peak-to-mean curvature, so a short
    # blend drawn that way has a TIGHTER elbow than the circular fillet it
    # replaces — which is exactly how "smooth" came to look aggressive.
    assert v1.winglet_flags(ch) == {"blend_shape": "spiral"}
    assert v1.winglet_flags(dict(ch, blend_shape="smooth")) == \
        {"blend_shape": "smooth"}
    assert v1.winglet_flags(dict(ch, blend_shape="arc")) == \
        {"blend_shape": "arc"}
    # a non-blended winglet sends no shape at all
    assert v1.winglet_flags(dict(v1.BUILDER_DEFAULTS, winglets="free")) == {}
