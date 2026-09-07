"""Three surfaces, one chain: wing -> horizontal tail -> vertical tail.

Stage 2 is asked once per surface that can be handed a section of its own.
"Done here" on each of those stages named ``airfoil_aft`` LITERALLY and sent
every other surface to stage 3, which was right while there were two surfaces
and became wrong the moment V5 added the fin's stage:

* on the HORIZONTAL tail's stage, "done here" walked past stage 2.7 to the
  wing stage — the reported defect. The fin then flew whatever section it was
  left with, because nothing asked;
* on a vehicle whose only second surface is the FIN, the wing's stage offered
  nothing at all and the surface row naming it never drew.

The order is :data:`session.STAGES` filtered by :func:`session.stage_visible`
(``session.next_section_stage``), so a fourth surface joins the chain by being
in the stage list rather than by a fourth literal.
"""
from __future__ import annotations

import pytest

from gui.v3 import session
from gui.v3.app import assemble


def _texts(view) -> list[str]:
    return [getattr(e, "text", "") or "" for e in view.descendants()]


def _with_tail_and_fin():
    import gui.nice_app as v1

    ctx = assemble()
    ctx.act("accept_mission")
    ctx.S["wing"]["choices"]["tail"] = True
    v1.normalise_choices(ctx.S["wing"]["choices"], keep="tail")
    session.apply_choices(ctx.S)
    ctx.refresh()
    assert session.aft_surface(ctx.S) and session.fin_surface(ctx.S)
    return ctx


# ----------------------------------------------------------- the order itself

def test_the_chain_is_wing_then_horizontal_then_vertical():
    S = _with_tail_and_fin().S
    assert session.next_section_stage(S, "airfoil") == "airfoil_aft"
    assert session.next_section_stage(S, "airfoil_aft") == "airfoil_fin"
    assert session.next_section_stage(S, "airfoil_fin") is None


def test_a_surface_the_vehicle_lacks_is_not_in_the_chain():
    ctx = _with_tail_and_fin()
    ctx.act("set_fin", False)
    assert session.next_section_stage(ctx.S, "airfoil") == "airfoil_aft"
    assert session.next_section_stage(ctx.S, "airfoil_aft") is None


def test_a_vehicle_whose_only_second_surface_is_the_fin_still_chains():
    """The wing's section must hand over to stage 2.7 directly. This was the
    half of the defect nothing else could reach: with no aft surface the
    handover branch returned early and the fin's stage was never offered."""
    ctx = assemble()
    ctx.act("accept_mission")
    ctx.S["wing"]["choices"]["tail"] = False
    session.apply_choices(ctx.S)
    if session.aft_surface(ctx.S) is not None:
        pytest.skip("this family always carries a second lifting surface")
    assert session.fin_surface(ctx.S)
    assert session.next_section_stage(ctx.S, "airfoil") == "airfoil_fin"


# ------------------------------------------------- ...as the SHELL offers it

def test_the_horizontal_tails_stage_offers_the_vertical_one(capsys):
    """The reported defect, at the control the user presses."""
    ctx = _with_tail_and_fin()
    session.set_section(ctx.S, {"name": "naca0012", "source": "library"},
                        None, surface="aft")
    ctx.render("airfoil_aft", "section")
    texts = _texts(ctx.views[("airfoil_aft", "section")])
    assert any("vertical stabiliser" in t for t in texts), texts
    assert not any("Go to the wing stage" in t for t in texts)
    capsys.readouterr()


def test_the_vertical_tails_stage_is_the_last_one_before_the_wing(capsys):
    ctx = _with_tail_and_fin()
    session.set_section(ctx.S, {"name": "naca0012", "source": "library"},
                        None, surface="fin")
    ctx.render("airfoil_fin", "section")
    texts = _texts(ctx.views[("airfoil_fin", "section")])
    assert any("Go to the wing stage" in t for t in texts), texts
    capsys.readouterr()


def test_the_stage_says_which_surface_it_is_designing(capsys):
    """The "designing: …" row drew only where there was an AFT surface, so a
    fin-only vehicle got no heading and no way back to the wing's section."""
    ctx = _with_tail_and_fin()
    ctx.render("airfoil_fin", "screen")
    texts = " ".join(_texts(ctx.views[("airfoil_fin", "screen")]))
    assert "VERTICAL STABILISER" in texts
    assert "the wing's section" in texts
    capsys.readouterr()


def test_the_surface_is_named_in_one_place():
    S = _with_tail_and_fin().S
    assert session.surface_name(S, "main") == "wing"
    assert session.surface_name(S, "aft") == session.aft_surface(S)
    assert session.surface_name(S, "fin") == "vertical stabiliser"
