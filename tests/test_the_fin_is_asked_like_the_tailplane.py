"""The Wing type card, reviewed: the vertical surface is asked what the
horizontal one is asked, and nothing else.

Four asks, and what each of them turned out to be sitting on:

1. **"Should be the same as horizontal."** The horizontal surface's size is
   the solver's; the vertical one's was two typed fields.
2. **"Shouldn't ask for AR, volume coefficient."** Both fields are gone. The
   flags stay — a stored record may carry them — so the READ-OUT has to
   follow them, and it did not: ``_fin_geometry`` called the sizing law with
   no shape at all.
3. **"The horizontal and vertical separation give option to optimise or
   select."** Both toggles existed. What did not hold was the layout that
   cannot answer: a T-tail's toggle was disabled with the tooltip *"a
   T-tail's height IS its fin span"* and a typed height field was drawn under
   it anyway, writing ``z_t_m`` — which ``TailProblem.dz_for`` honours over
   the layout's own, so the tailplane flew at the typed height while the fin
   was still sized to ``sqrt(AR_vt·S_vt)`` and the T came apart.
4. **"Fuselage diameter is required only when asked for optimise horizontal
   separation."** It was drawn in every state, under the fin card, two
   blocks above the toggle that decides whether it means anything.

And underneath all four, three defects nothing on screen could show:

* ``fin_shape_kwargs``'s two spellings were SWAPPED in ``wingtail.py``, so a
  stated ``fin_ar`` — a number the card offered a field for — raised
  ``TypeError`` on every lattice-backed wing+tail family in the registry;
* ``_fin_geometry`` read the arm from ``api.TAIL_ARM_KEY``, which does not
  exist, behind a ``hasattr`` that turned the miss into ``None``: every fin
  on the card was drawn at the middle of the SEARCH BAND, stated arm or not;
* the fin read-out was not redrawn when the arm or the height was typed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aerobo import api, fin as finmod, tail as tailmod        # noqa: E402


#: a family whose objective is the LATTICE (``wingtail.py``) rather than the
#: lifting line — the one the swapped spellings lived in
LATTICE = "tail + winglet"


class _Stated:
    """The three fields ``fin_shape_kwargs`` reads off a problem."""

    def __init__(self, V_v=None, AR=None, tc=None):
        self.fin_volume_coeff = V_v
        self.fin_ar = AR
        self.fin_tc = tc


def _x(name: str):
    return api.PROBLEM_SPECS[name].build({}, {}, None).bounds.mean(axis=1)


def _shell(**choices):
    from gui.v3.app import assemble

    ctx = assemble("air")
    ctx.act("set_choice", "tail", True)
    for key, value in choices.items():
        ctx.act("set_choice", key, value)
    ctx.render("wing", "type")
    return ctx


def _shell_stating_nothing(**choices):
    """A shell with the OPENING ANSWERS cleared, for the tests about what
    an UNSTATED value costs.

    Since "the V3 shell opens on a whole aeroplane" a fresh session already
    states a body diameter (0.22 m) and opens with the tail's HEIGHT free.
    Both are good, and both make "what does this card say when nothing is
    stated" a question about a session that no longer exists — so the tests
    that are about the unstated case ask for it, rather than assuming the
    default is empty.
    """
    ctx = _shell(**choices)
    ctx.S["wing"]["flags"].pop("fuselage_diameter_m", None)
    ctx.render("wing", "type")
    return ctx


def _text(view) -> str:
    return " ".join((getattr(e, "text", "") or "")
                    for e in view.descendants())


def _labels(view) -> list:
    return [(getattr(e, "text", "") or "").strip()
            for e in view.descendants()
            if "field-label" in (getattr(e, "_classes", None) or [])]


def _readouts(view) -> dict:
    """``{caption: value}`` for every ``widgets.readout`` on the card."""
    out, pending = {}, None
    for e in view.descendants():
        classes = getattr(e, "_classes", None) or []
        text = getattr(e, "text", "") or ""
        if "readout-label" in classes:
            pending = text
        elif "readout-big" in classes and pending is not None:
            out.setdefault(pending, []).append(text)
            pending = None
    return out


# ------------------------------------- the two spellings, and their consumers

def test_each_wrapper_is_ACCEPTED_by_the_call_it_is_named_for():
    """The contract the ``ar_key`` argument existed to keep, and did not.

    Asserted by CALLING, not by reading the string: a signature is what
    decides this, and both consumers refuse an unexpected keyword — which is
    exactly how the swap showed up, as a ``TypeError`` rather than a silently
    dropped argument.
    """
    stated = _Stated(V_v=0.05, AR=3.0, tc=0.12)
    g = finmod.fin_for_layout(b=10.0, S=10.0, l_t=5.5,
                              tail_type="conventional", dz=0.5,
                              **finmod.fin_law_kwargs(stated))
    cd0 = tailmod.vtail_cd0(10.0, 5.5, b=10.0, S=10.0,
                            **finmod.fin_drag_kwargs(stated))
    # ...and each stated number BIT, or "accepted" would only mean "ignored"
    plain = finmod.fin_for_layout(b=10.0, S=10.0, l_t=5.5,
                                  tail_type="conventional", dz=0.5)
    assert g.V_v == pytest.approx(0.05) and g.AR == pytest.approx(3.0)
    assert g.S > plain.S and g.height > plain.height
    assert g.tc == pytest.approx(0.12)
    assert cd0 != pytest.approx(
        tailmod.vtail_cd0(10.0, 5.5, b=10.0, S=10.0))


def test_the_wrappers_disagree_about_the_spelling_and_that_is_the_point():
    """One name each, so no call site picks a string."""
    stated = _Stated(AR=2.0)
    assert finmod.fin_law_kwargs(stated) == {"AR": 2.0}
    assert finmod.fin_drag_kwargs(stated) == {"AR_vt": 2.0}


def test_a_MAPPING_reads_the_same_as_a_problem():
    """The shell holds the same three numbers in a flags dict, and a second
    reader there is how a card ends up describing a fin nobody is flying."""
    assert finmod.fin_law_kwargs({"fin_ar": 2.0, "fin_volume_coeff": None}) \
        == finmod.fin_law_kwargs(_Stated(AR=2.0))
    assert finmod.fin_law_kwargs({}) == {}


# --------------------------------------- the crash the shell offered a field for

def test_a_stated_fin_ASPECT_RATIO_does_not_take_the_lattice_family_down():
    """``TypeError: vtail_cd0() got an unexpected keyword argument 'AR'``.

    Unconditional — the drag build-up is on every evaluation of every
    ``wingtail.py`` family, and the card offered a field for the number that
    triggered it.
    """
    built = api.PROBLEM_SPECS[LATTICE].build({}, {"fin_ar": 1.8}, None)
    row = built.evaluate(_x(LATTICE))
    base = api.PROBLEM_SPECS[LATTICE].build({}, {}, None).evaluate(_x(LATTICE))
    assert row["cd0_fin"] != pytest.approx(base["cd0_fin"], rel=1e-12), \
        "the stated aspect ratio reached the drag book but changed nothing"
    assert row["score"] != pytest.approx(base["score"], rel=1e-12)


def test_it_reaches_the_LATERAL_DECK_too_which_is_the_other_spelling():
    """``size_fin() got an unexpected keyword argument 'AR_vt'`` — the same
    swap, the other way round, in the panelled fin the yaw stiffness comes
    from."""
    def deck(flags):
        built = api.PROBLEM_SPECS[LATTICE].build({}, flags, None)
        built.problem.lateral = True
        return built.evaluate(_x(LATTICE))

    base, tall = deck({}), deck({"fin_ar": 1.8})
    assert tall["Cn_beta"] > base["Cn_beta"], \
        "a taller fin at the same area must make more yaw stiffness"


def test_an_unstated_shape_is_still_the_identity():
    """The rename must not have moved a published run.

    Asserted on ``cd0_fin`` BIT-FOR-BIT, because that term is the whole of
    the fin's contribution to this family's score, and on the score itself
    only to 1e-12. The lattice family's score is not bit-reproducible across
    INTERLEAVED configurations — measured: twelve identical calls in a row
    agree exactly, but a ``none`` evaluation between two fin-flagged ones
    lands 1 ulp apart (31.942900275633107 against ...114, 2.2e-16 relative).
    That predates this work and is not what this test is about; demanding
    `==` here made the file order-dependent, which is a worse test than a
    tolerance thirty ulp wider than any effect it could hide.
    """
    a = api.PROBLEM_SPECS[LATTICE].build({}, {}, None).evaluate(_x(LATTICE))
    b = api.PROBLEM_SPECS[LATTICE].build(
        {}, {"fin_volume_coeff": finmod.V_V_DEFAULT,
             "fin_ar": finmod.AR_VT_DEFAULT}, None).evaluate(_x(LATTICE))
    assert a["cd0_fin"] == b["cd0_fin"], "the fin's own term moved"
    assert a["score"] == pytest.approx(b["score"], rel=1e-12)

    # ...and on the LIFTING-LINE family, where bit-for-bit IS a property the
    # suite already relies on, the whole score is unmoved
    c = api.PROBLEM_SPECS["tail"].build({}, {}, None).evaluate(_x("tail"))
    d = api.PROBLEM_SPECS["tail"].build(
        {}, {"fin_volume_coeff": finmod.V_V_DEFAULT,
             "fin_ar": finmod.AR_VT_DEFAULT}, None).evaluate(_x("tail"))
    assert c["score"] == d["score"]
    assert c["cd0_fin"] == d["cd0_fin"]


# ------------------------------------------------ the card asks what it should

def test_the_vertical_tail_card_asks_for_NOTHING():
    """Every number on it is a read-out, the way the horizontal surface's
    area is. The two fields it used to carry were the only ones."""
    view = _shell().views[("wing", "type")]
    labels = _labels(view)
    assert "volume coefficient V_v" not in labels
    assert "Vertical tail" in _text(view)
    got = _readouts(view)
    for caption in ("area", "height", "chord", "t/c", "quarter chord", "foot"):
        assert caption in got, (caption, sorted(got))


def test_the_fin_is_drawn_at_the_STATED_arm_not_the_middle_of_the_band():
    """The ``api.TAIL_ARM_KEY`` that does not exist.

    ``hasattr(api, "TAIL_ARM_KEY")`` was False, so the lookup beside it was
    dead and every fin was sized at ``0.5*(lo+hi)`` of the search band. On
    the shipped 3–8 m band that is 5.5 m, which is also the default stated
    arm — so the defect was invisible until the arm was typed.
    """
    assert not hasattr(api, "TAIL_ARM_KEY"), \
        "if this key now exists, _fin_geometry should read IT, not tail_flags"
    ctx = _shell(tail_arm="fixed", tail_arm_m=3.0)
    got = _readouts(ctx.views[("wing", "type")])["area"]
    # S_vt = V_v b S / l_t = 0.04*10*10/3.0
    assert "1.33" in got, got
    assert "0.727" not in got, "still the 5.5 m fin"


def test_typing_the_arm_moves_the_fin_WITHOUT_a_redraw():
    """The read-out is in its own container precisely so a typed number can
    move it; it was not in the handler's list, so it went stale instead."""
    ctx = _shell(tail_arm="fixed")
    view = ctx.views[("wing", "type")]
    assert "0.727" in _readouts(view)["area"]
    field = [e for e in view.descendants()
             if type(e).__name__ == "Number"
             and str((e._props or {}).get("model-value")) == "5.5"]
    assert field, "no stated-arm field on the card"
    field[0].set_value(3.0)
    assert "1.33" in _readouts(view)["area"], "the fin did not follow the arm"


def test_a_stated_HEIGHT_moves_the_fins_foot():
    """The vertical separation is the other half of where the surface sits,
    and the card was standing the fin on the layout's own height whatever
    the user had said — ``TailProblem.dz_for`` prefers the stated one."""
    # ...and it has to BE stated: the shell now opens with the height FREE,
    # where the number is a design-box row and a typed one moves nothing.
    low = _readouts(_shell(tail_height="fixed", tail_height_m=0.5)
                    .views[("wing", "type")])["foot"]
    high = _readouts(_shell(tail_height="fixed", tail_height_m=2.0)
                     .views[("wing", "type")])["foot"]
    assert low != high, (low, high)
    assert "2" in high[0]


# ------------------------------------------------- the T-tail cannot answer it

def test_a_T_TAIL_is_not_offered_a_height_it_cannot_have():
    """The toggle said the layout owns it; the field under it said otherwise.

    Both cannot be true: a T-tail's tailplane sits on the fin's tip, so its
    height IS the fin's span, and a typed ``z_t_m`` moved one without the
    other.
    """
    view = _shell(tail_type="t_tail").views[("wing", "type")]
    assert "above the wing plane" not in _labels(view), _labels(view)
    # ...and it is REPORTED instead, off the same law that sizes the fin
    size = __import__("gui.v3.session", fromlist=["session"])
    b, area = size.flown_size(_shell(tail_type="t_tail").S)[:2]
    g = finmod.fin_for_layout(
        b=float(b), S=float(area), l_t=5.5, tail_type="t_tail",
        dz=tailmod.tail_height("t_tail", 5.5, float(b), float(area)))
    assert f"{abs(g.height):.3g}" in _text(view)


def test_a_CONVENTIONAL_tail_still_is_offered_one():
    """The point is the layout that cannot answer, not the question.

    Asked of the STATED height, which is what the T-tail is refused: with
    the height free the number is a design-box row on every layout and
    there is no field on this card to be offered or refused.
    """
    view = _shell(tail_height="fixed").views[("wing", "type")]
    assert "above the wing plane" in _labels(view)


# ------------------------------------ ask 3: the two toggles read the same way

def _toggles(view) -> list:
    return [(e._props or {}).get("options") or []
            for e in view.descendants() if type(e).__name__ == "Toggle"]


def test_BOTH_separations_offer_the_same_two_answers():
    """The literal ask. The horizontal one always read `you state it` /
    `optimise it`; the vertical one read `the layout's own` whenever the
    height was currently OPTIMISED, because it asked whether the family the
    user is ON declares ``z_t_m`` — and a free-height family by construction
    does not, the height being in its design vector instead.
    """
    for choices in ({}, {"tail_height": "free"}, {"tail_type": "v_tail"},
                    {"tail_type": "v_tail", "tail_height": "free"}):
        view = _shell(**choices).views[("wing", "type")]
        pairs = [[o.get("label") if isinstance(o, dict) else o for o in opts]
                 for opts in _toggles(view)]
        same = [p for p in pairs if p == ["you state it", "optimise it"]]
        assert len(same) == 2, (choices, pairs)


def test_the_T_TAIL_is_the_documented_exception_and_says_so():
    """The one layout where "you state it" would be a lie, so it does not say
    it: the height is the fin's span and the toggle is disabled."""
    view = _shell(tail_type="t_tail").views[("wing", "type")]
    pairs = [[o.get("label") if isinstance(o, dict) else o for o in opts]
             for opts in _toggles(view)]
    assert ["the layout's own", "optimise it"] in pairs, pairs
    assert ["you state it", "optimise it"] in pairs, \
        "the HORIZONTAL separation still asks both ways on a T-tail"


def test_and_the_new_label_is_TRUE_not_merely_uniform():
    """Picking `you state it` from a free-height session really does give a
    field — asserted through the family it lands on, not through the word."""
    from gui.nice_app import derive_problem

    ctx = _shell(tail_height="free")
    on_now = api.PROBLEM_SPECS[ctx.S["wing"]["problem"]]
    assert api.TAIL_HEIGHT_KEY not in on_now.flags, \
        "a free-height family should carry the height as a VARIABLE"
    name, _ = derive_problem(dict(ctx.S["wing"]["choices"],
                                  tail_height="fixed"))
    assert api.TAIL_HEIGHT_KEY in api.PROBLEM_SPECS[name].flags
    # ...and the field really is drawn there
    assert "above the wing plane" in _labels(
        _shell(tail_height="fixed").views[("wing", "type")])


# ------------------------------------------- the body, where it decides something

def test_the_body_is_ASKED_either_way_and_the_CONSEQUENCE_is_what_changes():
    """The claim moved, deliberately, and this is the claim now.

    It used to be "the body is asked ONLY where the separation is
    optimised" — the field appeared with a searched arm and vanished with a
    stated one. Hidden means unanswered, so the card grew a SWITCH ("this
    aeroplane has a fuselage"), which is drawn in both states; what depends
    on the arm is what the card SAYS the answer costs. Both halves here, so
    a card that stopped distinguishing them fails.
    """
    searched = _shell_stating_nothing().views[("wing", "type")]
    assert api.arm_is_searched(_shell().S["wing"]["problem"])
    assert "this aeroplane has a fuselage" in _text(searched)
    assert "top of the arm band" in _text(searched)

    stated = _shell_stating_nothing(tail_arm="fixed").views[("wing", "type")]
    assert "this aeroplane has a fuselage" in _text(stated)
    assert "top of the arm band" not in _text(stated)
    assert "cannot move anything" in _text(stated)


def test_it_is_still_SHOWN_where_a_value_is_already_there():
    """Hidden means unanswered, never silently answered: a stored record or a
    session that switched the toggle back must be able to see and clear it."""
    from gui.v3.app import assemble

    ctx = assemble("air")
    ctx.act("set_choice", "tail", True)
    ctx.act("set_choice", "tail_arm", "fixed")
    ctx.S["wing"]["flags"]["fuselage_diameter_m"] = 0.35
    ctx.render("wing", "type")
    assert any(lbl.startswith("body diameter")
               for lbl in _labels(ctx.views[("wing", "type")]))


def test_the_required_case_says_what_an_unstated_body_costs():
    """The measurement, not an adjective: with the arm searched and no body
    the answer is the top of the arm band whatever else is true.

    The two halves are the two sides of the fuselage SWITCH now, and they
    cannot be in one render: switched off there is no diameter to label,
    switched on there is nothing unstated to cost. Both are asserted.
    """
    off = _text(_shell_stating_nothing().views[("wing", "type")])
    assert "top of the arm band" in off
    assert "body diameter" not in off, "a body nobody has still has a field"

    on = _text(_shell().views[("wing", "type")])
    assert "body diameter (required)" in on, \
        "a searched arm does not say the body is required"
    assert "top of the arm band" not in on
