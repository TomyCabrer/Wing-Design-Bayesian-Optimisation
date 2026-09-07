"""The vertical tail's thickness is its SECTION's, and is asked once.

The Wing type card asked for ``fin_tc`` as a typed number while stage 2.7
chose the fin's aerofoil — and an aerofoil has a thickness. Two controls, one
property: the sizing law, the drag book's form factor and the CAD loft took
the typed number, so a fin could be charged and lofted at 10 % while flying a
14 % section chosen one stage earlier.

The number now comes from the section (``session.fin_thickness``), the card
reports it instead of asking, and the run is sent the same value.
"""
from __future__ import annotations

import pytest

from gui.v3 import config, session
from gui.v3.app import assemble

THICK = 0.14


def _shell():
    """A vehicle with a TAIL: only those families declare the fin's shape
    flags, and a skip here would have hidden the one assertion that proves
    the thickness reaches the solver."""
    import gui.nice_app as v1

    ctx = assemble()
    ctx.act("accept_mission")
    ctx.S["wing"]["choices"]["tail"] = True
    v1.normalise_choices(ctx.S["wing"]["choices"], keep="tail")
    session.apply_choices(ctx.S)
    assert session.fin_surface(ctx.S)
    assert "fin_tc" in config.spec(ctx.S).flags, ctx.S["wing"]["problem"]
    return ctx


def test_with_no_section_it_is_the_stand_ins_own_thickness():
    S = _shell().S
    from aerobo import fin as finmod

    assert session.fin_thickness(S) == pytest.approx(finmod.FIN_TC_DEFAULT)
    # the stand-in section and the charged thickness are one answer
    assert session.fin_default_section(S)["tc"] == pytest.approx(
        session.fin_thickness(S))


def test_a_chosen_section_sets_it():
    ctx = _shell()
    session.set_section(ctx.S, {"name": "naca0014", "tc": THICK,
                                "source": "library"}, None, surface="fin")
    assert session.fin_thickness(ctx.S) == pytest.approx(THICK)


def test_the_run_is_sent_the_sections_thickness():
    """The flag that reaches the solver, not merely the read-out."""
    ctx = _shell()
    session.set_section(ctx.S, {"name": "naca0014", "tc": THICK,
                                "source": "library"}, None, surface="fin")
    assert config.flags(ctx.S)["fin_tc"] == pytest.approx(THICK)


def test_the_sized_fin_follows_it():
    """The geometry the card draws and the export lofts."""
    ctx = _shell()
    before = session.surface_geometry(ctx.S, "fin")
    session.set_section(ctx.S, {"name": "naca0014", "tc": THICK,
                                "source": "library"}, None, surface="fin")
    after = session.surface_geometry(ctx.S, "fin")
    assert before and after
    # the sizing law takes t/c, so the fin it returns must not be the same
    # object it returned for a 10 % section
    assert session.fin_thickness(ctx.S) == pytest.approx(THICK)
    assert after["area"] > 0.0


def test_the_card_does_not_ask_for_it_twice():
    """Asked of the RENDERED card, not of a literal in the source.

    It used to read ``FIN_SHAPE_ROWS`` off the module text, which was a
    restatement of the code and stopped meaning anything the moment the
    table went: the card now asks for no fin dimension at all (the volume
    coefficient and the aspect ratio were withdrawn with it — see
    ``tests/test_the_fin_is_asked_like_the_tailplane.py``), so the honest
    question is what a field on it is LABELLED.
    """
    ctx = _shell()
    ctx.render("wing", "type")
    labels = [(getattr(e, "text", "") or "").strip().lower()
              for e in ctx.views[("wing", "type")].descendants()
              if "field-label" in (getattr(e, "_classes", None) or [])]
    assert not any("thickness" in lbl or lbl.startswith("t/c")
                   for lbl in labels), labels
    # ...and it is REPORTED instead, at the value the SECTION states — the
    # number the run is sent (test_the_run_is_sent_the_sections_thickness),
    # so a gap where the field was would be caught here too
    session.set_section(ctx.S, {"name": "naca0014", "tc": THICK,
                                "source": "library"}, None, surface="fin")
    ctx.render("wing", "type")
    shown, pending = [], None
    for e in ctx.views[("wing", "type")].descendants():
        classes = getattr(e, "_classes", None) or []
        if "readout-label" in classes:
            pending = (getattr(e, "text", "") or "").strip()
        elif "readout-big" in classes and pending == "t/c":
            shown.append((getattr(e, "text", "") or "").strip())
            pending = None
    assert shown == [f"{THICK:.3g}"], shown
