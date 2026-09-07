"""A CAR's geometry must be drawn the way it sits on the car.

The ask, from the shell: "for car wing I want the geometry to be displayed
upside down. The wing on top of the endplates."

Nothing was wrong with the geometry. A car family solves the vehicle MIRRORED
about the horizontal plane (``geometry.MIRRORED_FRAME``) so that the model's
lift IS the downforce and the track is an image plane above it; an endplate
that reaches down for the deck therefore has a POSITIVE height in the model.
Drawn in that frame the endplates stand above the wing and the car is on its
roof — which the solvers have said in ``breakdown["frame"]`` since the family
shipped, and which no drawer ever read: ``api.design_report`` did not carry
the key onto the ``geometry`` block the drawers are handed.

WHAT IS ASSERTED, AND WHY IT IS THE LAYOUT AND NOT THE DATA. The fix reverses
the AXIS and leaves the numbers alone, exactly as
``test_plan_view_faces_forward.py`` pins for the streamwise one: every height,
clearance and margin quoted beside these figures is in the model frame, and a
hover that disagreed with the breakdown would be a second frame nobody asked
for. So the honest invariant is the pair — the plate's z stays POSITIVE in the
data AND the vertical axis is drawn downward. Both halves are checked, so
neither can be satisfied by breaking the other.

The 3-D half is the axis and not the camera's ``up`` vector, and that is not a
style choice: plotly's default 3-D drag mode is the turntable, which holds z
vertical on screen and discards an inverted up silently. This was measured in
a browser, not reasoned about.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from aerobo import api, geometry              # noqa: E402

CAR = "car rear wing + endplates"
#: the control: an aeroplane is solved in the package's own frame and every
#: one of these views must be untouched by the car's flip
AIR = "winglet"


def _report(name: str):
    cfg = api.RunConfig(problem_name=name)
    built = api.PROBLEM_SPECS[name].build({}, {}, None)
    x = np.asarray(built.bounds.mean(axis=1), dtype=float)
    return api.design_report(cfg, x), x, list(built.param_labels)


@pytest.fixture(scope="module")
def car():
    v1 = pytest.importorskip("gui.nice_app")
    rep, x, labels = _report(CAR)
    return v1, rep["geometry"], x, labels, rep


@pytest.fixture(scope="module")
def aircraft():
    v1 = pytest.importorskip("gui.nice_app")
    rep, x, labels = _report(AIR)
    return v1, rep["geometry"], x, labels, rep


# --------------------------------------------------------------- the frame

def test_the_solver_declares_the_frame_and_the_report_carries_it(car):
    """The drawers hold the ``geometry`` block; the frame has to reach it."""
    _v1, geom, _x, _labels, rep = car
    assert rep["breakdown"]["frame"] == geometry.MIRRORED_FRAME
    assert geom.get("frame") == geometry.MIRRORED_FRAME, (
        "api.design_report must carry the solver's own frame onto the "
        "geometry block — a drawer that has to guess it from a problem NAME "
        "is a second author of which way up a car is")
    assert geometry.is_mirrored(geom)


def test_an_aeroplane_declares_no_mirrored_frame(aircraft):
    _v1, geom, _x, _labels, _rep = aircraft
    assert not geometry.is_mirrored(geom)


# ----------------------------------------------------- the data is untouched

def test_the_plate_still_reaches_positive_z_in_the_data(car):
    """The flip must not have been achieved by negating the geometry."""
    _v1, geom, _x, _labels, rep = car
    z = np.asarray(geom.get("z") or [], dtype=float)
    wl = np.asarray(geom.get("is_winglet") or [], dtype=bool)
    assert z.size and wl.size == z.size and wl.any(), (
        "this test needs a design whose endplate was actually flown")
    h = float(rep["breakdown"]["endplate_h_m"])
    assert z[wl].max() > 0.0, (
        "the endplate reaches the car's DOWNWARD direction, which is +z in "
        "the model — negating it here would put the numbers in a frame no "
        "margin in the breakdown is quoted in")
    assert z[wl].max() == pytest.approx(h, rel=0.2)


# ------------------------------------------------------- the axis is flipped

def test_the_front_view_draws_the_wing_above_its_endplates(car):
    v1, geom, _x, _labels, _rep = car
    fig = v1.fig_frontview(geom)
    assert fig is not None
    assert fig.layout.yaxis.autorange == "reversed", (
        "the plate's z is the LARGER number; without a reversed axis plotly "
        "puts it at the top of the pane and the car reads upside down")
    assert "car" in (fig.layout.yaxis.title.text or "").lower(), (
        "an axis drawn downward has to say so — the V3 shell strips figure "
        "titles, so the axis is where a reader is told")


def test_the_sections_are_drawn_the_way_the_car_meets_the_air(car):
    v1, geom, x, labels, rep = car
    fig = v1.fig_sections_as_flown(geom, x, labels,
                                   breakdown=rep["breakdown"])
    assert fig is not None
    assert fig.layout.yaxis.autorange == "reversed", (
        "a section that makes downforce arches DOWN on the car and up in the "
        "model; drawn in the model frame it reads as an aeroplane's")


def test_the_three_d_view_draws_its_z_axis_downward(car):
    v1, geom, x, labels, _rep = car
    fig = v1.fig_wing3d(geom, x, labels)
    assert fig is not None
    assert fig.layout.scene.zaxis.autorange == "reversed"
    assert fig.layout.scene.camera.up.z in (None, 1), (
        "the camera's up vector is NOT the mechanism: plotly's default "
        "turntable drag mode holds z vertical and discards an inverted up, "
        "so a fix written there is a fix that does nothing")


def test_the_aeroplane_views_are_untouched(aircraft):
    """Nothing here may reach a design solved in the package's own frame."""
    v1, geom, x, labels, rep = aircraft
    front = v1.fig_frontview(geom)
    if front is not None:
        assert front.layout.yaxis.autorange is None
    sec = v1.fig_sections_as_flown(geom, x, labels,
                                   breakdown=rep["breakdown"])
    if sec is not None:
        assert sec.layout.yaxis.autorange is None
    three_d = v1.fig_wing3d(geom, x, labels)
    assert three_d is not None
    assert three_d.layout.scene.zaxis.autorange is None
    assert "up +" in (three_d.layout.scene.zaxis.title.text or "")
