"""A tandem pair's vertical tail: on a card, at a station, and the one flown.

The report, verbatim: *"for tandem wing there is no way to state or constrain
position of vertical tail"*.

It was right, and the mechanism was one line of nesting. ``_fin_controls`` —
the whole "Vertical tail" card, the sizing law's read-out, the station, the
section it flies — lived INSIDE ``_tail_controls``, which stage 3 draws only
under ``ch["tail"]``. A tandem's second surface is its REAR WING, so that
switch is off and disabled for the family, and the card was drawn for no
configuration of it. Stage 1 meanwhile offered "add a vertical stabiliser
(fin and rudder)" with the sentence *"Its section is stage 2.7; stage 3 sizes
it"*, and stage 3 sized nothing.

The one field that did exist — the BOOM — sat on the pair's layout block
between "Rear wing, aft by" and "Rear wing, above by", under a paragraph
about the rear wing's stagger and above one about the rear wing's span, with
no sentence of its own. Three "…, aft by" fields, one of which was the fin's.

And ``_fin_geometry`` could not have answered for a pair anyway: every input
it reads is a WING+TAIL input (``tail_flags``' ``l_t_m``, then
``session.published_arm_box``), and a tandem has neither — so both branches
fell through to ``(None, None)`` and the card would have said "state the size
and the separation and this fills in" for ever.

What this file holds is the OUTCOME, not the wiring: the card exists, the
station is typed in exactly one place, and the fin the card draws is the fin
``api.design_report`` states — the same ``fin.tandem_fin_station`` /
``fin.size_fin`` pair the two tandem engines charge the drag on.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aerobo import api, fin as finmod            # noqa: E402

BOOM_LABEL = "Fin, aft of the rear wing by"


def _shell(system: str = "tandem"):
    from gui.v3.app import assemble

    ctx = assemble("air")
    if system != "single":
        ctx.act("set_choice", "system", system)
    return ctx


def _texts(view) -> list[str]:
    return [(getattr(e, "text", "") or "").strip()
            for e in view.descendants()]


def _readouts(view) -> dict:
    """``label -> value`` of every ``widgets.readout`` in a view.

    Paired by DOM ORDER, which is what ``readout`` builds: the caption
    (``readout-label``) immediately precedes the number (``readout-big``) in
    its own column.
    """
    out, last = {}, None
    for e in view.descendants():
        cls = getattr(e, "_classes", None) or []
        txt = (getattr(e, "text", "") or "").strip()
        if "readout-label" in cls:
            last = txt
        elif "readout-big" in cls and last is not None:
            out[last] = txt
            last = None
    return out


def _card_fin(ctx):
    """The fin the pair's card is quoting, read off the shell's own state."""
    import gui.nice_app as v1
    from gui.v3 import session as v3session

    S = ctx.S
    b, area = v3session.flown_size(S)
    dx, dz = v1.tandem_stagger(S["wing"]["choices"])
    boom = S["wing"]["choices"].get("tandem_fin_boom_m")
    x_qc, z_root = finmod.tandem_fin_station(
        float(dx), float(dz), None if boom is None else float(boom))
    return finmod.fin_for_layout(
        b=float(b), S=float(area), l_t=float(x_qc), tail_type="tandem",
        dz=float(z_root), **finmod.fin_law_kwargs(S["wing"]["flags"]))


def _run_fin(ctx):
    """The fin the RUN reports for the same configuration."""
    import gui.nice_app as v1

    S = ctx.S
    name = S["wing"]["problem"]
    flags = dict(v1.tandem_flags(S["wing"]["choices"], name))
    built = api.PROBLEM_SPECS[name].build({}, flags, None)
    rep = api.design_report(api.RunConfig(problem_name=name, flags=flags),
                            np.asarray(built.bounds.mean(axis=1)))
    return (rep.get("geometry") or {}).get("fin")


# ------------------------------------------------------- the card exists
def test_a_pair_has_a_vertical_tail_card_with_the_fin_on_it():
    """The defect itself: the family drew no card at all, so none of the
    four numbers that describe its fin was anywhere on screen."""
    ctx = _shell()
    view = ctx.views[("wing", "type")]
    assert "Vertical tail" in _texts(view)
    got = _readouts(view)
    for key in ("area", "height", "chord", "t/c", "quarter chord", "foot"):
        assert key in got, f"the fin's {key} is not on the card: {got}"
    # ...and they are NUMBERS, not the "one of those is not settled yet"
    # placeholder ``_fin_geometry`` returned for every pair
    assert float(got["area"]) > 0.0
    assert float(got["quarter chord"]) > 0.0


def test_the_station_is_asked_in_exactly_one_place():
    """One question, one place. The boom is the FIN's station, so it is
    asked on the fin's card — and taken off the pair's layout block, where
    it read as a third rear-wing length."""
    ctx = _shell()
    n = sum(t == BOOM_LABEL
            for key in (("wing", "type"), ("wing", "box"), ("wing", "solver"))
            for t in _texts(ctx.views[key]))
    assert n == 1, f"the boom is asked {n} times"


def test_v1_keeps_the_field_on_its_layout_block():
    """``with_fin`` is a statement by the SHELL that it asks the question
    elsewhere — V1 and V2 have no vertical-tail card and must keep it."""
    import gui.nice_app as v1
    from nicegui import ui

    def _draw(**kw):
        with ui.column() as col:
            v1._tandem_controls(dict(v1.BUILDER_START), lambda *a: None, **kw)
        return [(getattr(e, "text", "") or "").strip()
                for e in col.descendants()]

    assert BOOM_LABEL in _draw()
    assert BOOM_LABEL not in _draw(with_fin=False)


# -------------------------------------------- the card draws what is flown
@pytest.mark.parametrize("boom", [None, 0.0, 2.0, 12.0])
def test_the_fin_on_the_card_is_the_fin_the_run_reports(boom):
    """THE VALUE TEST. A card that quotes a fin the solver is not flying is
    the defect ``fin.py`` exists to have fixed, and a station has one author
    (``fin.tandem_fin_station``) precisely so a second reader cannot drift.

    ``boom = 0`` is the on-the-wing station this family shipped with and is
    still reachable; ``None`` is the published default.
    """
    ctx = _shell()
    if boom is not None:
        ctx.act("set_choice", "tandem_fin_boom_m", boom)
    card, run = _card_fin(ctx), _run_fin(ctx)
    assert run is not None
    assert card.S == pytest.approx(run["S"], rel=0, abs=1e-12)
    assert card.height == pytest.approx(run["height_m"], rel=0, abs=1e-12)
    assert card.chord == pytest.approx(run["chord_m"], rel=0, abs=1e-12)
    assert card.x_qc == pytest.approx(run["x_qc_m"], rel=0, abs=1e-12)
    assert card.z_root == pytest.approx(run["z_root_m"], rel=0, abs=1e-12)


def test_typing_a_boom_moves_the_numbers_on_the_card():
    """The read-out follows the field. It is redrawn from its own container
    because the boom is a ``VALUE_KEYS`` number — ``set_choice`` deliberately
    does NOT rebuild the type card from a field inside it — and without that
    hand-off the card quoted the fin of the PREVIOUS boom."""
    ctx = _shell()
    before = _readouts(ctx.views[("wing", "type")])
    ctx.act("set_choice", "tandem_fin_boom_m", 0.0)
    after = _readouts(ctx.views[("wing", "type")])
    # boom 0 stands the fin on the rear wing's root: half the arm, so
    # (volume coefficient) twice the area
    assert float(after["quarter chord"]) < float(before["quarter chord"])
    assert float(after["area"]) > float(before["area"])
    assert float(after["quarter chord"]) == pytest.approx(
        _card_fin(ctx).x_qc, rel=1e-3)


def test_the_stagger_moves_the_fin_too():
    """Both ends of the pair's stagger place the fin — the arm is
    ``dx + boom`` and the foot is ``dz`` — so a typed stagger has to reach
    the same read-out. It is the same skip that hid the boom's own edit."""
    ctx = _shell()
    before = _readouts(ctx.views[("wing", "type")])
    ctx.act("set_choice", "tandem_dx_m", 9.0)
    after = _readouts(ctx.views[("wing", "type")])
    assert float(after["quarter chord"]) > float(before["quarter chord"])
    assert float(after["quarter chord"]) == pytest.approx(
        _card_fin(ctx).x_qc, rel=1e-3)


# --------------------------------------------------------- the other families
def test_a_plain_wing_still_draws_no_vertical_tail_card():
    """The card is self-gating on the family's own fin flags, which is what
    lets it be called for anything in air without a tail card: hundreds of
    registered problems declare none of them and must not be offered one.

    A bare wing is the case the new call site could have broken — it is
    reached by the same ``elif`` a pair is — and it is the case stage 1's own
    switch already knows about ("this family carries no vertical surface").
    """
    ctx = _shell("single")
    ctx.act("set_choice", "tail", False)
    assert not any(k in api.PROBLEM_SPECS[ctx.S["wing"]["problem"]].flags
                   for k in api.FIN_SHAPE_KEYS)
    assert "Vertical tail" not in _texts(ctx.views[("wing", "type")])


def test_a_wing_and_tail_still_has_its_card_under_the_tail():
    """Unchanged: the wing+tail's fin card is still drawn from inside
    ``_tail_controls``, and it still has no boom field — that station is the
    tailplane's arm, which is asked once, below."""
    ctx = _shell("single")
    ctx.act("set_choice", "tail", True)
    texts = _texts(ctx.views[("wing", "type")])
    assert "Vertical tail" in texts
    assert BOOM_LABEL not in texts
    assert "area" in _readouts(ctx.views[("wing", "type")])


# ------------------------------------- the empennage is not a pair's question
def test_a_stale_v_tail_does_not_take_the_pairs_fin_away():
    """An empennage is how a SECOND SURFACE is arranged, and a pair's second
    surface is its rear wing — so ``tail_type`` has no meaning here. It
    SURVIVES a configuration change, though, and a session that chose a
    V-tail and then moved to a tandem read ``v_tail`` in two places: the card
    printed "a V-tail has NO separate fin" and ``session.fin_surface``
    answered False, which also took stage 2.7 away — while the run went on
    building, charging, lofting and flying one (``api.design_report`` sizes a
    pair's fin at ``tail_type="tandem"``, and the flag is not even in the
    family's spec, so it never travels).
    """
    from gui.v3 import session as v3session

    ctx = _shell("single")
    ctx.act("set_choice", "tail_type", "v_tail")
    ctx.act("set_choice", "system", "tandem")
    assert v3session.fin_surface(ctx.S) is True
    texts = _texts(ctx.views[("wing", "type")])
    assert BOOM_LABEL in texts
    assert not any("NO SEPARATE FIN" in t or "NO vertical stabiliser" in t
                   for t in texts)
    # ...and the fin on the card is still the fin the run reports
    card, run = _card_fin(ctx), _run_fin(ctx)
    assert card.S == pytest.approx(run["S"], rel=0, abs=1e-12)


def test_a_v_tail_that_really_has_no_fin_still_says_so_and_says_why():
    """The other direction, and the sentence that was unreachable.
    ``fin_surface`` answers the LAYOUT before the switch, so the V-tail
    paragraph below that branch could never be drawn and every V-tail read
    "you said so on stage 1" about a decision the layout had made."""
    ctx = _shell("single")
    ctx.act("set_choice", "tail_type", "v_tail")
    texts = _texts(ctx.views[("wing", "type")])
    assert "Vertical tail" in texts
    assert any("V-TAIL HAS NO SEPARATE FIN" in t for t in texts)
    assert not any("you said so on stage 1" in t for t in texts)


def test_the_switch_off_still_says_the_user_said_so():
    ctx = _shell("single")
    ctx.act("set_choice", "fin", False)
    texts = _texts(ctx.views[("wing", "type")])
    assert any("you said so on stage 1" in t for t in texts)
