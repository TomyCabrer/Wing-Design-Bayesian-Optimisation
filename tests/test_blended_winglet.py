"""Blended wing/winglet transition: the arc geometry, the junction-drag
add-on that makes it worth anything, and the span-accounting trap.

A lifting-surface method values the transition ONLY through the wake trace,
so on its own "blending" just redraws the line — and at free span it redraws
it OUTBOARD, which is the raked-wingtip trap in disguise. Both halves are
pinned here: the geometry (arc length conserved, projection grows, height
shrinks) and the pricing (junction.py charged, span capped on the TRUE
projection).
"""

import numpy as np
import pytest

from aerobo import api, junction
from aerobo.geometry import (
    Wing,
    bounds,
    wing_from_x,
    winglet_path,
    winglet_projection,
    winglet_tip_height,
)
from aerobo.objective import Problem, evaluate
from aerobo.vlm import VLM

H = 0.75            # winglet arc length [m] on the b = 10 m baseline
CANT = 75.0


# ---------------- the arc itself ----------------

def test_zero_blend_is_the_legacy_straight_ray():
    t = np.linspace(0.0, H, 17)
    y, z = winglet_path(t, H, CANT, 0.0)
    assert np.array_equal(y, t * np.cos(np.deg2rad(CANT)))
    assert np.array_equal(z, t * np.sin(np.deg2rad(CANT)))


@pytest.mark.parametrize("beta", [0.0, 0.25, 0.5, 1.0])
def test_arc_length_is_conserved(beta):
    """The blend must be a SHAPE change: same quarter-chord length, so a
    blended and a sharp device of equal h have equal wetted area."""
    t = np.linspace(0.0, H, 200_001)
    y, z = winglet_path(t, H, CANT, beta)
    s = float(np.sum(np.hypot(np.diff(y), np.diff(z))))
    assert s == pytest.approx(H, rel=1e-7)


def test_the_line_leaves_the_wing_plane_tangentially():
    """The whole point: no kink. The first step of a blended path is along
    the wing plane, while a sharp one starts at the full cant angle."""
    t = np.array([1e-4, 2e-4])
    _, z_blend = winglet_path(t, H, CANT, 0.5)
    _, z_sharp = winglet_path(t, H, CANT, 0.0)
    # tangential departure means z ~ t^2/(2R): doubling t QUADRUPLES the rise
    assert z_blend[1] / z_blend[0] == pytest.approx(4.0, rel=1e-3)
    # the kinked one just goes straight up at the cant angle: z ~ t
    assert z_sharp[1] / z_sharp[0] == pytest.approx(2.0, rel=1e-12)
    assert z_blend[1] < 1e-3 * z_sharp[1]


@pytest.mark.parametrize("cant", [75.0, -75.0])
def test_blending_trades_tip_height_for_projected_span(cant):
    sharp_y, sharp_z = winglet_projection(H, cant, 0.0), \
        winglet_tip_height(H, cant, 0.0)
    blend_y, blend_z = winglet_projection(H, cant, 1.0), \
        winglet_tip_height(H, cant, 1.0)
    assert blend_y > sharp_y                  # reaches further outboard
    assert abs(blend_z) < abs(sharp_z)        # and not as far up/down
    assert np.sign(blend_z) == np.sign(sharp_z)


def test_blend_frac_outside_the_unit_interval_is_refused():
    with pytest.raises(ValueError, match="blend_frac"):
        winglet_path(np.array([0.0]), H, CANT, 1.5)


def test_the_vlm_panels_follow_the_blended_line():
    m = VLM(Wing(), N=40, winglet_h_frac=0.15, winglet_cant_deg=CANT,
            winglet_blend_frac=0.8)
    sharp = VLM(Wing(), N=40, winglet_h_frac=0.15, winglet_cant_deg=CANT)
    assert m.z.max() < sharp.z.max()
    assert m.y.max() > sharp.y.max()


# ---------------- junction drag (the add-on) ----------------

def test_hoerner_correlation_magnitude_and_floor():
    """t/c = 0.12 on a 1 m chord: ~28 drag counts on a 1 m^2 reference, i.e.
    ~2.8 counts on the 10 m^2 baseline wing — the right order for an
    unfilleted junction. Below the correlation's zero-crossing it floors at
    0 rather than crediting negative drag."""
    assert junction.hoerner_drag_area(0.12, 1.0) == pytest.approx(0.0028, abs=2e-4)
    assert junction.hoerner_drag_area(0.05, 1.0) == 0.0


def test_fillet_credit_is_bounded_and_monotone():
    c = 1.0
    credits = [junction.fillet_credit(r, c) for r in (0.0, 0.02, 0.05, 0.2)]
    assert credits[0] == 1.0
    assert credits == sorted(credits, reverse=True)
    assert credits[-1] == pytest.approx(junction.FILLET_CREDIT_FLOOR)


def test_blend_radius_matches_the_arc_that_drew_it():
    """R = (beta h) / |cant| — the radius junction.py prices is the radius
    geometry.winglet_path actually drew."""
    beta = 0.4
    r = junction.blend_radius(H, CANT, beta)
    assert r == pytest.approx(beta * H / abs(np.deg2rad(CANT)))
    # the drawn arc has that radius: check the sagitta of the first arc chord
    t = np.array([0.0, beta * H])
    y, z = winglet_path(t, H, CANT, beta)
    chord_len = float(np.hypot(y[1] - y[0], z[1] - z[0]))
    theta = abs(np.deg2rad(CANT))
    assert chord_len == pytest.approx(2.0 * r * np.sin(theta / 2.0), rel=1e-9)
    assert junction.blend_radius(H, CANT, 0.0) == 0.0


def test_junction_cd_scales_with_count_and_reference_area():
    one = junction.junction_cd(0.12, 1.0, 10.0, radius=0.0, n_junctions=1)
    two = junction.junction_cd(0.12, 1.0, 10.0, radius=0.0, n_junctions=2)
    assert two == pytest.approx(2.0 * one)
    assert junction.junction_cd(0.12, 1.0, 20.0, n_junctions=2) == \
        pytest.approx(two / 2.0)


# ---------------- the objective ----------------

def test_the_blended_mode_appends_one_variable_after_the_winglet_pair():
    box = bounds("winglet_blended")
    assert box.shape == (6, 2)
    assert np.array_equal(box[:5], bounds("winglet"))
    assert box[5].tolist() == [0.0, 1.0]


def test_span_cap_uses_the_true_projection_not_the_cosine():
    """Capping a blended device on h·cos(cant) would hand it free projected
    span — exactly the fabricated benefit the raked-tip rule exists for."""
    x = np.array([0.5, 0.0, -2.0, 0.15, 75.0, 1.0])
    w = wing_from_x(x, mode="winglet_capped_blended")
    proj = winglet_projection(0.15 * 10.0 / 2.0, 75.0, 1.0)
    assert w.b == pytest.approx(10.0 - 2.0 * proj)
    cosine_only = 10.0 * (1.0 - 0.15 * np.cos(np.deg2rad(75.0)))
    assert w.b < cosine_only            # the blend really does reach further


def test_junction_drag_is_off_by_default_and_priced_when_on():
    x = np.array([0.5, 1.0, -2.0, 0.12, 75.0, 0.0])
    off = evaluate(x, Problem(mode="winglet_blended"))
    on = evaluate(x, Problem(mode="winglet_blended", junction_drag=True))
    assert off["CD_junction"] == 0.0
    assert on["CD_junction"] > 0.0
    assert on["LoD"] < off["LoD"]


def test_a_legacy_winglet_run_is_untouched_by_the_feature():
    """Same 5-D vector, same answer as before the blend existed: the sharp
    mode has no blend variable and no junction charge."""
    x = np.array([0.5, 1.0, -2.0, 0.12, 75.0])
    plain = evaluate(x, Problem(mode="winglet"))
    blended_at_zero = evaluate(np.append(x, 0.0),
                               Problem(mode="winglet_blended"))
    assert blended_at_zero["LoD"] == pytest.approx(plain["LoD"], rel=1e-12)


def test_blending_pays_for_itself_only_against_a_charged_junction():
    """Span-capped, the blend costs tip height and buys a smooth corner. It
    wins when the corner is priced and loses when it is not — which is why
    the problem charges it."""
    def ld(beta, junction_drag):
        x = np.array([0.5, 1.0, -2.0, 0.12, 75.0, beta])
        return evaluate(x, Problem(mode="winglet_capped_blended",
                                   junction_drag=junction_drag))["LoD"]

    assert ld(0.25, True) > ld(0.0, True)      # charged: a modest blend wins
    assert ld(0.25, False) < ld(0.0, False)    # uncharged: pure height loss


def test_free_span_blending_is_the_raked_tip_trap():
    """Documented, not offered: with span free, blending monotonically
    'improves' L/D by rebuilding the wing wider. The API exposes only the
    span-capped problem for exactly this reason."""
    def ld(beta):
        x = np.array([0.5, 1.0, -2.0, 0.12, 75.0, beta])
        return evaluate(x, Problem(mode="winglet_blended",
                                   junction_drag=True))["LoD"]

    assert ld(1.0) > ld(0.5) > ld(0.0)
    # ...so the registry must not offer it. Asked on the MODE each builder
    # declares (``_build_objective`` records it as ``build.objective_mode``),
    # never on the builder's ``__name__``: every builder in api.py is an inner
    # closure literally called "build", so the old form searched the set
    # {"build"} and a registered free-span family would have left it green.
    modes = {getattr(spec.build, "objective_mode", None)
             for spec in api.PROBLEM_SPECS.values()}
    assert "winglet_capped_blended" in modes    # this set CAN see a blend mode
    assert "winglet_blended" not in modes


# ---------------- API + GUI wiring ----------------

def test_spec_is_registered_span_capped_and_charged():
    spec = api.PROBLEM_SPECS["winglet, blended (span-capped)"]
    built = spec.build({}, {}, None)
    assert built.param_labels[-1] == "winglet_blend_frac"
    assert built.problem.mode == "winglet_capped_blended"
    assert built.problem.junction_drag is True
    off = spec.build({}, {"junction_drag": False}, None)
    assert off.problem.junction_drag is False


def test_builder_maps_the_blended_menu_entry_without_a_cant_band_flag():
    import gui.nice_app as v1

    ch = dict(v1.BUILDER_DEFAULTS, winglets="capped", winglet_type="blended")
    assert v1.derive_problem(ch)[0] == "winglet, blended (span-capped)"
    # "blended" is a transition shape, not a cant band — it must not be sent
    # as a winglet_type flag (the API would reject it). What it DOES send is
    # the transition's shape (tests/test_smooth_blend.py owns that contract).
    assert "winglet_type" not in v1.winglet_flags(ch)
    assert set(v1.winglet_flags(ch)) == {"blend_shape"}
    assert v1.winglet_option_key(ch) == "blended"
