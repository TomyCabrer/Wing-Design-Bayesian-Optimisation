"""The mission's own measured box reaches the row the WING'S CANT is searched
in — or says out loud why it does not.

The complaint was "fix the recommendation for the design box when dihedral is
a new design variable".

``aerobo.recommend`` is label-generic and always produced a band for
``wing_dihedral_deg``; the SHELL threw it away. ``config.effective_bounds``
grew a seventh provenance value, ``"unpriced"`` — the floor this shell puts
on the searched cant while nothing in the objective prices its sign
(``session.unpriced_cant_floor``) — and the one consumer that decides which
rows a measurement may write still enumerated only three of the shell-derived
values. So the row read as "an answer the user gave", the measured band was
``continue``d, and because the "why not" record beside it was gated on span
rows, nothing was drawn either. On the shipped free-cant family the
measurement had 0.752 – 14.625 deg to offer and the row stayed at its full
floored 0 – 15.

Two more places assumed that anything which is not a size row falls back to
the family's PUBLISHED band, and both re-opened the row to the anhedral half
the floor exists to keep out: ``_unnarrowed_cfg`` popped the row out of an
already-flattened override dict that the floor had been merged into, and
``session.relax_measured_bands`` took its base from ``default_bounds`` and
then stamped the result ``"recommended"``, which outranks the floor. All
three have to hold together or the row oscillates.
"""
from __future__ import annotations

import sys

import pytest

from aerobo import api

sys.path.insert(0, "gui")

ROW = "wing_dihedral_deg"

#: WHICH ANSWERS TO THE CANT QUESTION SEARCH THE DIHEDRAL. Asked of the api
#: rather than listed here, because the freedom was split in two ("optimise
#: the dihedral" and "optimise the sweep") after the floor was written, and a
#: hard-coded pair would silently stop covering the half that matters.
CANTS = tuple(c for c in getattr(api, "WING_CANTS", ("fixed", "free"))
              if c != "fixed")


def _searches_the_dihedral(name: str) -> bool:
    """``api.dihedral_is_searched`` where it exists, ``cant_is_searched``
    where the freedom has not been split yet — the same question either
    way, and this test is about the ROW, not about the menu."""
    ask = getattr(api, "dihedral_is_searched", None) or api.cant_is_searched
    return bool(ask(name))


def _measured(cant="free"):
    from gui.v3.app import assemble

    ctx = assemble()
    ctx.act("accept_mission")
    ctx.act("set_choice", "wing_cant", cant)
    if not _searches_the_dihedral(ctx.S["wing"]["problem"]):
        pytest.skip(f"'{cant}' does not search the dihedral in this build")
    ctx.act("auto_recommend", True)
    return ctx


def _held(ctx):
    return (ctx.S["wing"].get("auto_rec") or {}).get("held") or {}


@pytest.mark.parametrize("cant", CANTS)
def test_the_row_is_written_or_QUOTED_but_never_silently_dropped(cant):
    """The headline. Whether this mission's draws have anything narrower to
    say about the row is a measurement; saying nothing at all is a bug."""
    from gui.v3 import config

    ctx = _measured(cant)
    band, source = config.effective_bounds(ctx.S)[ROW]
    assert source == "recommended" or ROW in _held(ctx), (
        f"{cant}: the measured band for {ROW} was neither written nor "
        f"quoted — the silent drop is back")


def test_the_written_band_stays_inside_the_shells_own_floor():
    """A recommendation may narrow a search and may never widen one, and the
    floor is part of what it must not widen past."""
    from gui.v3 import config, session

    ctx = _measured("free")
    band, source = config.effective_bounds(ctx.S)[ROW]
    if source != "recommended":
        pytest.skip("this mission had nothing narrower to say about the row")
    assert band[0] >= session.CANT_FLOOR_DEG - 1e-9, band
    assert band[1] <= 15.0 + 1e-9


def test_a_second_pass_does_not_re_open_the_floor():
    """``_unnarrowed_cfg`` takes the measured bands back before re-measuring,
    and it was taking the FLOOR back with them — the next pass then measured
    over the published -10..15 and came back anhedral."""
    from gui.v3 import config, session

    ctx = _measured("free")
    ctx.act("auto_recommend", True)
    band, _src = config.effective_bounds(ctx.S)[ROW]
    assert band[0] >= session.CANT_FLOOR_DEG - 1e-9, band


def test_a_builder_MENU_does_not_re_open_it_either():
    """``relax_measured_bands`` carries a measurement through a builder
    change by opening its bottom back to a base — and the base for this row
    is the floor, not the family's published band."""
    from gui.v3 import config, session

    ctx = _measured("free")
    ctx.act("set_choice", "tail_design", "designed")
    band, _src = config.effective_bounds(ctx.S)[ROW]
    assert band[0] >= session.CANT_FLOOR_DEG - 1e-9, band


def test_an_L_D_RANKING_MAY_NOT_BOUND_THIS_ROW():
    """Weighting the spiral lifts the floor — the objective prices the sign
    now — but with no frozen normalisation band the measurement ranks the
    draws on the family's own L/D instead of on the composite. L/D is
    precisely the objective that does NOT price this row's sign (2.0-2.5 %
    end to end, inside that ranking's own noise), and the band it drew came
    back -8.08 to 14.38 deg on a session that had just asked for spiral
    stability. Quoted, never written."""
    from gui.v3 import config, session

    ctx = _measured("free")
    weights = dict(session.wing_score_state(ctx.S)["weights"])
    weights["spiral"] = 0.3
    ctx.act("set_wing_objective", "composite")
    for key, value in weights.items():
        ctx.act("set_wing_weight", key, value)
    if not api.wants_spiral(config.flags(ctx.S)):
        pytest.skip("this build does not price the spiral")
    ctx.act("auto_recommend", True)
    a = ctx.S["wing"].get("auto_rec") or {}
    if not a.get("stripped"):
        pytest.skip("the composite had a normalisation band after all")
    _band, source = config.effective_bounds(ctx.S)[ROW]
    assert source != "recommended", "an L/D ranking bounded the cant"
    assert ROW in _held(ctx), "...and it was not even quoted"


def test_the_quote_carries_the_ROWS_OWN_unit():
    """This hard-coded "m" while it only ever drew spans. The first non-span
    row to reach it would have been quoted in metres."""
    from gui.v3.stages import wing as wing_stage

    assert wing_stage._row_unit("b_m") == "m"
    assert wing_stage._row_unit(ROW) == "deg"
    assert wing_stage._row_unit("S_m2") == "m²"
    assert wing_stage._row_unit("taper") == ""


def test_the_floored_row_is_not_painted_as_an_untouched_one():
    """A chip the same colour as "default" is how a floored row went a whole
    session without being noticed."""
    from gui.v3.stages import wing as wing_stage

    assert "unpriced" in wing_stage._SOURCE_COLOR
    assert wing_stage._SOURCE_COLOR["unpriced"] \
        != wing_stage._SOURCE_COLOR["default"]


def test_the_shell_owned_sources_are_named_ONCE():
    """The defect was a second copy of this list that did not grow the
    seventh value."""
    from gui.v3.stages import wing as wing_stage

    for name in ("default", "mission", "recommended", "unpriced"):
        assert name in wing_stage.SHELL_OWNED_SOURCES
    assert "user" not in wing_stage.SHELL_OWNED_SOURCES
