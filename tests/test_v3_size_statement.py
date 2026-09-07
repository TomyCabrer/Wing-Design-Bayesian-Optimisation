"""The size can be stated in whichever unit the user actually has.

Stage 1 asked the size in exactly one unit — the wing loading — and derived
the reference area from it. That is what a specification carries, and it is
not what everybody has: a rule, a mould or a class limit gives you an AREA,
and a wing is first sketched as "about this wide, about this slender".

So the same one stored number is now asked in three faces. What these tests
are about is that they stay ONE number:

* whichever face is typed, the other two move to agree with it;
* switching between them stores nothing at all;
* the span in the sketch face writes NO span — the span is stage 3's
  constraint, stated there once — and the aspect ratio it writes is stage
  2's existing estimate, not a second one.

The last two are the ones worth having. This shell spent sessions 31 and 33
taking a span and an aspect ratio OUT of the form because two places to
state one thing is two answers that can disagree; a size card that quietly
put them back would undo that, and would do it in a way every screen still
looked right about.
"""
import pytest


def _S(medium: str = "air"):
    from gui.v3 import session

    return session.make_session(medium)


# ----------------------------------------------------- three faces, one wing
def test_stating_the_area_gives_the_matching_loading():
    from gui.v3 import session

    S = _S()
    S["mission"]["W_N"] = 900.0
    assert session.set_reference_area(S, 4.0)

    assert session.reference_area(S) == pytest.approx(4.0)
    assert session.wing_loading(S) == pytest.approx(225.0)


def test_stating_the_loading_gives_the_matching_area():
    from gui.v3 import session

    S = _S()
    S["mission"]["W_N"] = 900.0
    assert session.set_wing_loading(S, 225.0)

    assert session.reference_area(S) == pytest.approx(4.0)


def test_stating_a_span_and_an_aspect_ratio_gives_the_matching_area():
    from gui.v3 import session

    S = _S()
    S["mission"]["W_N"] = 900.0
    assert session.set_size_from_span_ar(S, span=6.0, ar=9.0)

    assert session.reference_area(S) == pytest.approx(4.0)     # b²/AR
    assert session.wing_loading(S) == pytest.approx(225.0)
    assert session.mission_span(S) == pytest.approx(6.0)


def test_the_three_faces_are_the_same_number_however_it_was_reached():
    """Reaching one wing three different ways must land on one area."""
    from gui.v3 import session

    areas = []
    for setter in (lambda S: session.set_wing_loading(S, 225.0),
                   lambda S: session.set_reference_area(S, 4.0),
                   lambda S: session.set_size_from_span_ar(S, span=6.0,
                                                           ar=9.0)):
        S = _S()
        S["mission"]["W_N"] = 900.0
        assert setter(S)
        areas.append(session.reference_area(S))
    assert areas[0] == pytest.approx(areas[1])
    assert areas[1] == pytest.approx(areas[2])


# ------------------------------------------------- switching stores no size
def test_switching_the_face_moves_nothing():
    from gui.v3 import session

    S = _S()
    before = dict(S["mission"])
    for mode in (session.SIZE_AS_AREA, session.SIZE_AS_SPAN_AR,
                 session.SIZE_AS_LOADING):
        assert session.set_size_statement(S, mode)
        assert session.size_statement(S) == mode

    after = dict(S["mission"])
    after.pop("size_stated_as", None)
    before.pop("size_stated_as", None)
    assert after == before, ("switching which unit the size is TYPED in "
                             "changed the session's stored numbers")


def test_the_default_face_is_the_wing_loading():
    """An untouched session must open exactly where it always did."""
    from gui.v3 import session

    assert session.size_statement(_S()) == session.SIZE_AS_LOADING


def test_an_unknown_face_is_refused_and_leaves_the_old_one():
    from gui.v3 import session

    S = _S()
    assert session.set_size_statement(S, session.SIZE_AS_AREA)
    assert not session.set_size_statement(S, "chords")
    assert session.size_statement(S) == session.SIZE_AS_AREA


# --------------------------------------------- which of the pair is held
def test_typing_a_span_holds_the_aspect_ratio():
    from gui.v3 import session

    S = _S()
    assert session.set_size_from_span_ar(S, span=6.0, ar=9.0)
    ar = session.section_aspect_ratio(S)

    assert session.set_size_from_span_ar(S, span=8.0)
    assert session.section_aspect_ratio(S) == pytest.approx(ar)
    assert session.reference_area(S) == pytest.approx(64.0 / 9.0)
    assert session.mission_span(S) == pytest.approx(8.0)


def test_typing_an_aspect_ratio_holds_the_span():
    """The span on screen must not jump when the slenderness is retyped —
    it is the number the user just committed to."""
    from gui.v3 import session

    S = _S()
    assert session.set_size_from_span_ar(S, span=6.0, ar=9.0)

    assert session.set_size_from_span_ar(S, ar=12.0)
    assert session.mission_span(S) == pytest.approx(6.0)
    assert session.reference_area(S) == pytest.approx(36.0 / 12.0)


# ---------------------------------------------- no second span, no second AR
def test_the_sketch_writes_no_span():
    """Stage 3 states the span. Nothing on stage 1 may store one."""
    from gui.v3 import session

    S = _S()
    assert session.chosen_span(S) is None
    assert session.set_size_from_span_ar(S, span=6.0, ar=9.0)
    assert session.chosen_span(S) is None, (
        "stage 1 stored a span; the span is asked in exactly one place")
    assert S["wing"]["choices"].get("span_m_choice") in (None, )


def test_the_sketch_aspect_ratio_is_stage_twos_own_estimate():
    """One aspect ratio, one place — the one the section's chord and
    Reynolds number are already derived from."""
    from gui.v3 import session

    S = _S()
    assert session.set_size_from_span_ar(S, span=6.0, ar=11.0)

    assert session.section_aspect_ratio(S) == pytest.approx(11.0)
    assert session.nominal_aspect_ratio(S) == pytest.approx(11.0)
    assert float(S["airfoil"]["ar"]) == pytest.approx(11.0)


def test_the_sketch_span_is_the_span_an_unconstrained_wing_flies():
    """√(AR·S) is what ``nominal_span`` already derives, so the sketch and
    the run must not be two different widths where nobody chose a span."""
    from gui.v3 import session

    S = _S()
    assert session.set_size_from_span_ar(S, span=6.0, ar=9.0)
    assert session.nominal_span(S) == pytest.approx(session.mission_span(S))


# ------------------------------------------------------ the disagreement
def test_a_span_chosen_on_stage_three_is_reported_not_overwritten():
    from gui.v3 import session

    S = _S()
    assert session.set_size_from_span_ar(S, span=6.0, ar=9.0)
    assert session.mission_span_disagrees(S) is None

    assert session.set_span_m(S, 4.5)
    gap = session.mission_span_disagrees(S)
    assert gap is not None, ("stage 3 flies 4.5 m and the sketch says 6 m; "
                             "the card must be able to say so")
    assert gap[0] == pytest.approx(6.0)
    assert gap[1] == pytest.approx(4.5)
    # and the mission's own area is untouched by stage 3's choice
    assert session.reference_area(S) == pytest.approx(4.0)


def test_agreeing_spans_are_not_reported_as_a_disagreement():
    from gui.v3 import session

    S = _S()
    assert session.set_size_from_span_ar(S, span=6.0, ar=9.0)
    assert session.set_span_m(S, 6.0)
    assert session.mission_span_disagrees(S) is None


# ------------------------------------------------------------- refusals
@pytest.mark.parametrize("bad", [0.0, -1.0, "", None, "wide"])
def test_an_unusable_area_is_refused_and_stores_nothing(bad):
    from gui.v3 import session

    S = _S()
    was = session.reference_area(S)
    assert not session.set_reference_area(S, bad)
    assert session.reference_area(S) == pytest.approx(was)


@pytest.mark.parametrize("kw", [{"span": 0.0}, {"span": -3.0},
                                {"ar": 0.0}, {"ar": -8.0},
                                {"span": "wide"}, {"ar": "slender"}])
def test_an_unusable_sketch_is_refused_and_stores_nothing(kw):
    from gui.v3 import session

    S = _S()
    was_area = session.reference_area(S)
    was_ar = session.section_aspect_ratio(S)
    assert not session.set_size_from_span_ar(S, **kw)
    assert session.reference_area(S) == pytest.approx(was_area)
    assert session.section_aspect_ratio(S) == pytest.approx(was_ar)


def test_naming_neither_is_refused():
    from gui.v3 import session

    assert not session.set_size_from_span_ar(_S())


# ------------------------------------------------------------ on the card
def _texts(view):
    return [getattr(e, "text", "") or "" for e in view.descendants()]


def _labels(view):
    """Field labels and toggle option labels — a toggle carries its options
    as a prop, so ``_texts`` alone cannot see the switch."""
    out = list(_texts(view))
    for e in view.descendants():
        for opt in getattr(e, "_props", {}).get("options") or []:
            if isinstance(opt, dict) and "label" in opt:
                out.append(str(opt["label"]))
    return out


def _card(mode=None):
    from gui.v3 import session
    from gui.v3.app import assemble

    ctx = assemble("air")
    if mode is not None:
        assert session.set_size_statement(ctx.S, mode)
        ctx.render("mission", "operating")
    return ctx.views[("mission", "operating")]


def test_the_switch_is_on_the_card():
    from gui.v3 import session

    labels = _labels(_card())
    for mode in session.SIZE_STATEMENTS:
        assert session.SIZE_STATEMENT_LABELS[mode] in labels, (
            f"no way to reach the {mode} face of the size")


@pytest.mark.parametrize("mode,wanted,unwanted", [
    ("loading", "wing loading W/S", "reference area S"),
    ("area", "reference area S", "wing loading W/S"),
    ("span_ar", "span (sketch)", "wing loading W/S"),
])
def test_exactly_one_face_has_a_field(mode, wanted, unwanted):
    """Two fields for one number is the pair that can disagree."""
    labels = _labels(_card(mode))
    assert wanted in labels
    assert unwanted not in labels


def test_the_sketch_field_says_the_span_is_not_the_wings():
    from gui.v3 import session

    labels = _labels(_card(session.SIZE_AS_SPAN_AR))
    assert "aspect ratio (estimate)" in labels
    blob = " ".join(labels).lower()
    assert "stage 3" in blob, ("the card must name who owns the span it is "
                              "showing a sketch of")
