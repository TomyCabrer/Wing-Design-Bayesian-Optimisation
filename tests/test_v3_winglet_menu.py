"""One question about the tip device: what SHAPE is it?

The seven-entry menu (``nice_app.WINGLET_OPTIONS``) is the STUDY menu — it
puts the span accounting (free vs capped) and the two blend-DESIGNING
families on screen as separate answers, because comparing them is the winglet
study. A design session is asking something smaller and the answer set is
three: a vertical fence, a canted winglet, a blended one.

So V3 offers ``nice_app.WINGLET_SHAPES``, and two things follow:

* the SPAN ACCOUNTING is chosen by the honest rule, not by the user — a fence
  has no projection to cap, a canted or blended device is capped wherever a
  capped variant exists, and the water families (scored span-free on purpose)
  fall through to their free-span pair;
* the BLEND is a value (``api.WINGLET_BLEND_KEY``), so a blended tip costs the
  same two design variables a canted one does: its height and its cant.

V1 and V2 keep the full menu: the studies that produced the published numbers
are selected through it.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def test_the_shape_menu_is_three_answers_plus_none():
    from gui import nice_app as v1

    assert list(v1.WINGLET_SHAPE_LABELS) == ["none", "vertical", "canted",
                                             "blended"]
    ch = v1.start_choices(medium="air")
    assert list(v1.winglet_shapes(ch)) == ["none", "vertical", "canted",
                                           "blended"]


def test_the_span_accounting_is_decided_not_asked():
    from gui import nice_app as v1

    air = v1.start_choices(medium="air")
    # a fence has ~zero horizontal projection: free span is the honest
    # accounting. A canted device projects, so it is capped.
    assert v1.winglet_shape_pair(air, "vertical") == ("free", "vertical")
    assert v1.winglet_shape_pair(air, "canted") == ("capped", "canted")
    assert v1.winglet_shape_pair(air, "blended") == ("capped", "canted")
    # ...and the water family is scored span-free, so it falls through
    water = v1.start_choices(medium="water")
    assert v1.winglet_shape_pair(water, "canted") == ("free", "canted")
    assert v1.winglet_shape_pair(water, "blended") == ("free", "canted")


def test_a_blended_tip_costs_two_design_variables():
    from aerobo import api
    from gui import nice_app as v1

    for medium in ("air", "water"):
        ch = v1.start_choices(medium=medium)
        v1.set_winglet_shape(ch, "blended")
        v1.normalise_choices(ch, keep="winglets")
        name, _ = v1.derive_problem(ch)
        labels = api.PROBLEM_SPECS[name].param_labels
        assert "winglet_h_frac" in labels and "winglet_cant_deg" in labels
        assert "winglet_blend_frac" not in labels, (medium, name)
        assert v1.winglet_flags(ch)[api.WINGLET_BLEND_KEY] > 0.0


def test_the_shape_is_read_back_off_the_state():
    from gui import nice_app as v1

    ch = v1.start_choices(medium="air")
    assert v1.winglet_shape_key(ch) == "none"
    for shape in ("vertical", "canted", "blended"):
        v1.set_winglet_shape(ch, shape)
        assert v1.winglet_shape_key(ch) == shape
    # the blend-DESIGNING families (V1's own menu) map onto the same word:
    # what is on screen has to be what will be flown
    v1.set_winglet_shape(ch, "canted")
    ch["winglet_type"] = "blended"
    assert v1.winglet_shape_key(ch) == "blended"


def test_a_blend_the_new_family_cannot_draw_is_dropped_and_reported():
    """The stale-menu bug class, one level down: the blend is a VALUE, so the
    normalisation loop over the speciality keys would never see it, and a
    family whose solver draws the corner as a corner would leave the select
    holding a shape its own menu does not offer."""
    from gui import nice_app as v1

    ch = v1.start_choices(medium="air")
    v1.set_winglet_shape(ch, "blended")
    assert ch["winglet_blend_frac"] > 0.0

    ch["medium"] = "track"                    # the endplates ARE the device
    dropped = v1.normalise_choices(ch, keep="medium")
    assert ch["winglet_blend_frac"] is None
    assert "winglet_blend_frac" in dropped
    assert v1._SPECIAL_LABEL["winglet_blend_frac"] == "blended tip device"


def test_an_unknown_shape_is_refused_rather_than_guessed():
    import pytest

    from gui import nice_app as v1

    ch = v1.start_choices(medium="air")
    with pytest.raises(ValueError, match="unknown tip-device shape"):
        v1.set_winglet_shape(ch, "raked")     # the STUDY menu's vocabulary


def test_v1_and_v2_keep_the_study_menu():
    from gui import nice_app as v1

    assert len(v1.WINGLET_OPTIONS) == 7
    ch = v1.start_choices(medium="air")
    assert set(v1.winglet_options(ch)) == set(v1.WINGLET_OPTIONS)


def test_the_stage_renders_the_three_shapes(capsys):
    from gui.v3.app import assemble

    ctx = assemble("air")
    ctx.render("wing", "type")
    labels = []
    for e in ctx.views[("wing", "type")].descendants():
        opts = getattr(e, "options", None)
        if isinstance(opts, dict) and "none" in opts:
            labels.append(opts)
    assert any(set(o) == {"none", "vertical", "canted", "blended"}
               for o in labels), labels
    err = capsys.readouterr().err
    assert "Traceback" not in err, err


def test_selecting_a_shape_through_the_stage_flies_it(capsys):
    from aerobo import api
    from gui.v3 import config
    from gui.v3.app import assemble

    ctx = assemble("air")
    # a WING's tip device is what this file is about, so the tail the shell
    # now opens with (session.V3_START_CHOICES) is switched off first — with
    # it on the same two presses derive the wing+tail twins, which is a
    # different family's menu
    ctx.act("set_second_surface", False)
    ctx.act("set_winglet", "blended")
    S = ctx.S
    assert S["wing"]["problem"] == "winglet_capped + free chord law"
    flags = config.cfg_dict(S)["flags"]
    assert flags[api.WINGLET_BLEND_KEY] == 0.5
    assert flags["blend_shape"] == "spiral"

    ctx.act("set_winglet", "none")
    # the chord TREND is V3's declared opening answer and travels on every
    # family that declares it (tests/test_v3_pipeline.py: V3_ONLY_PHYSICS_FLAGS)
    assert config.physics_flags(config.cfg_dict(S)) == {
        api.CHORD_TREND_KEY: config.CHORD_TREND}
    assert S["wing"]["problem"] == "wing (free chord law)"

    err = capsys.readouterr().err
    assert "Traceback" not in err, err
