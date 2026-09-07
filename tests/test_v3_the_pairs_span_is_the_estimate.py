"""V3 stage 2: a tandem's two wings are sized BY the aspect-ratio estimate.

``surface_geometry`` answers for both surfaces of a tandem pair — one
reference area split between two wings on ONE span — and that span is
``sqrt(AR · S)``, the stage-2 estimate itself. The note under the estimate
field was written for a surface sized out of the family's own design box (a
tail: ``S_t_m2`` at ``AR_t``, which the estimate really does not move), and it
fired on the tandem front wing too, directly beneath the only control that
moves that chord: "Stage 2's aspect-ratio estimate is the WING's, and does not
move it", while typing 16 instead of 5 moves the quoted chord 1.0000 →
0.5590 m and the quoted Re 9.996e+05 → 5.588e+05.

The engine side is right and is tested at
tests/test_section_per_surface.py:264-299 — what is gated here is the
sentence, against the numbers on the same card.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _texts(view):
    return [getattr(e, "text", "") or "" for e in view.descendants()]


def _note(ctx, stage):
    """The design-point card's own sentences, re-rendered."""
    ctx.render(stage, "screen")
    return [t for t in _texts(ctx.views[(stage, "screen")]) if "mean chord" in t]


def _quoted(ctx, surface):
    """What the shell WOULD have to quote: this surface's chord and Re."""
    from gui.v3 import session as ses

    geo = ses.surface_geometry(ctx.S, surface)
    dp = ses.surface_design_point(ctx.S, surface)
    return float(geo["mac"]), float(dp["re_mac"])


def _pair():
    from gui.v3.app import assemble

    ctx = assemble()
    ctx.act("accept_mission")
    ctx.act("set_choice", "system", "tandem")
    assert ctx.S["wing"]["choices"]["system"] == "tandem"
    return ctx


def test_neither_wing_of_a_pair_is_told_the_estimate_is_inert(capsys):
    """Both stages, because both wings sit on the estimate's span."""
    from gui.v3 import session as ses

    ctx = _pair()
    for stage, surface in (("airfoil", "main"), ("airfoil_aft", "aft")):
        note = _note(ctx, stage)
        assert note, stage
        line = note[-1]
        assert "does not move it" not in line, (stage, line)
        # it names the split and the shared span it is a split ON
        assert ses.surface_geometry(ctx.S, surface)["source"] in line
        assert "share of the pair" in line
        mac, re = _quoted(ctx, surface)
        assert f"{mac:.4f} m" in line and f"{re:.3e}" in line, line
    capsys.readouterr()


def test_the_estimate_moves_the_chord_the_note_quotes(capsys):
    """The defect was a claim about a control, so the test moves the control.

    Nothing here asserts a call: the aspect ratio goes in, and the chord and
    Reynolds number the card quotes have to be the ones the surface now flies
    at.
    """
    from gui.v3 import session as ses

    ctx = _pair()
    mac0, re0 = _quoted(ctx, "main")
    before = _note(ctx, "airfoil")[-1]

    assert ses.set_section_aspect_ratio(ctx.S, 16.0)
    mac1, re1 = _quoted(ctx, "main")
    after = _note(ctx, "airfoil")[-1]

    # the estimate is the pair's span, so it IS the chord and the Reynolds
    # number of both wings
    assert mac1 < 0.9 * mac0 and re1 < 0.9 * re0, (mac0, mac1, re0, re1)
    assert f"{mac0:.4f} m" in before and f"{mac1:.4f} m" in after
    assert f"{re0:.3e}" in before and f"{re1:.3e}" in after
    assert "DOES move this chord" in after
    capsys.readouterr()


def test_a_surface_sized_by_its_own_box_keeps_the_sentence(capsys):
    """The sentence is TRUE for a tail — its area and aspect ratio are the
    run's own variables — so the fix must not take it away from the one
    surface it was written for."""
    from gui.v3.app import assemble

    ctx = assemble()
    ctx.act("accept_mission")
    ctx.act("set_second_surface", True)
    # THE WHOLE CARD, lead and popup together. A sentence longer than
    # widgets.HINT_WORDS_ON_SCREEN keeps its opening on screen and puts
    # the rest behind the "?" beside it, so "the card says X" is a claim
    # about the card, not about one label of it.
    ctx.render("airfoil_aft", "screen")
    aft = " ".join(_texts(ctx.views[("airfoil_aft", "screen")]))
    assert "S_t_m2" in aft and "does not move it" in aft
    assert "share of the pair" not in aft

    # ...and the wing of that same aircraft has no surface geometry of its
    # own, so it keeps the mission-area sentence
    main = _texts(ctx.views[("airfoil", "screen")])
    ctx.render("airfoil", "screen")
    main = _texts(ctx.views[("airfoil", "screen")])
    assert any("on the mission's" in t for t in main)
    assert not any("share of the pair" in t for t in main)
    capsys.readouterr()
