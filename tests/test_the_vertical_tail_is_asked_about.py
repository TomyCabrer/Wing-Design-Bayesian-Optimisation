"""The fin is a surface now, so the card that chooses the layout says so.

The Wing type card carried this, as a comment, for the whole of V3:

    # NO fin-drag switch and NO control-type toggle. V1/V2 offer both;
    # this shell offers neither, for the two reasons in
    # session.V3_PINNED_CHOICES: the fin is not a surface this package
    # models ...

and `session.V3_PINNED_CHOICES` spelled it out: "no panels in any solver ...
nothing in geometry.py or cad.py ... it survives as two Raymer scalars only".

Every clause of that was falsified by V5 and none of them was updated:
`fin.py` is one sizing law, `vlm.VerticalSurface` panels it, `cad.fin_surface`
lofts it into the STL and the OpenVSP script, the 3-D view draws it, and
`flightmodel` flies it — where it produces all of the yaw stiffness.

The ONE clause still true is the one worth a control: the scored objective
does not charge the fin's parasite drag unless asked, so the design flies a
surface it did not pay for. That is an allowance the answer rides on, and a
shell that hides it is deciding it for the user.

What this file does NOT claim: the fin's SIZE is askable. It briefly was —
the card grew a volume-coefficient field and an aspect-ratio field once the
registry declared the flags — and the user withdrew both under the same rule
that deleted the drag switch: *"do as the horizontal tail"*. The horizontal
surface is asked for neither, so the vertical one is asked for neither, and
this file pins the reversal along with the read-out that has to follow the
flags a stored record can still carry.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aerobo import api, fin as finmod                           # noqa: E402


def _shell():
    from gui.v3.app import assemble
    return assemble("air")


def _with_tail(tail_type: str = "conventional"):
    ctx = _shell()
    ctx.act("set_choice", "tail", True)
    if tail_type != "conventional":
        ctx.act("set_choice", "tail_type", tail_type)
    ctx.render("wing", "type")
    return ctx


def _text(view) -> str:
    return " ".join((getattr(e, "text", "") or "")
                    for e in view.descendants())


def _switch(view, label: str):
    for e in view.descendants():
        if type(e).__name__ == "Switch" and label in (
                (getattr(e, "text", "") or "")):
            return e
    return None


# ------------------------------------------------------------- it is asked

def test_the_card_that_chooses_the_layout_describes_the_fin():
    ctx = _with_tail()
    text = _text(ctx.views[("wing", "type")])
    assert "Vertical tail" in text
    assert "VOLUME COEFFICIENT" in text, \
        "the card must say how the surface is sized, not just that it exists"


def test_the_drag_is_charged_and_is_no_longer_a_question():
    """It WAS a switch, and off — an allowance worth +5.5 % L/D that the
    user had to find and turn on. The user's call: "I don't want it to be
    an option... do as the horizontal tail." So the card states it, offers
    no control, and the shell can no longer write the flag at all."""
    ctx = _with_tail()
    view = ctx.views[("wing", "type")]
    assert _switch(view, "charge the fin") is None, \
        "the fin's drag is a switch again"
    assert "is CHARGED" in _text(view)
    name = ctx.S["wing"]["problem"]
    assert "charge_fin_drag" not in api.PROBLEM_SPECS[name].flags


def test_the_charge_reaches_the_run_without_being_asked_for():
    """The flag is gone, so the only evidence that survives is the number:
    an untouched session's design pays cd0_fin, and a V-tail does not."""
    from aerobo import tail as tailmod

    x = api.PROBLEM_SPECS["tail"].build({}, {}, None).bounds.mean(axis=1)
    plain = api.PROBLEM_SPECS["tail"].build({}, {}, None).evaluate(x)
    assert plain["cd0_fin"] > 0.0
    v = tailmod.evaluate_tail(x, tailmod.TailProblem(tail_type="v_tail",
                                                     dihedral_deg=35.0))
    assert v["cd0_fin"] == 0.0


def test_a_v_tail_is_told_it_has_no_fin():
    """From the sizing law, which is the same place the exporter and the
    flight model learn it — not from a sentence written here."""
    ctx = _with_tail("v_tail")
    text = _text(ctx.views[("wing", "type")])
    assert "no separate fin" in text.lower() or "NO separate fin" in text
    assert finmod.fin_for_layout(b=10.0, S=10.0, l_t=5.5,
                                 tail_type="v_tail", dz=0.5) is None


def test_a_t_tail_is_told_what_makes_it_one():
    ctx = _with_tail("t_tail")
    text = _text(ctx.views[("wing", "type")])
    assert "tip" in text.lower() and "T-TAIL" in text


def test_the_numbers_shown_are_the_sizing_laws_own():
    """The readout must not be a second derivation of the fin.

    Whatever area the card prints has to be the one ``fin_for_layout``
    returns for the same configuration, or the shell is describing a
    surface the design does not have.
    """
    ctx = _with_tail()
    from gui.v3 import session as v3session
    size = v3session.flown_size(ctx.S)
    if not size:
        pytest.skip("this configuration has no shell-decided size")
    box = v3session.published_arm_box(ctx.S)
    arm = 0.5 * (box[0] + box[1])
    from aerobo import tail as tailmod
    dz = tailmod.tail_height("conventional", arm, size[0], size[1])
    g = finmod.fin_for_layout(b=size[0], S=size[1], l_t=arm,
                              tail_type="conventional", dz=dz)
    assert f"{g.S:.3g}" in _text(ctx.views[("wing", "type")])


def test_the_size_is_NOT_asked_because_the_horizontal_ones_is_not():
    """Replaces `test_the_size_IS_askable_now` — the user's call, reversed.

    The rule for this surface is the one already recorded against its drag
    switch: *"do as the horizontal tail"*. The horizontal surface is never
    asked for a volume coefficient or an aspect ratio — its area is the
    solver's — so neither is the vertical one. The three flags stay in the
    registry for a library caller and a stored record; what goes is the
    pair of FIELDS on the Wing type card.
    """
    fin_flags = {f for s_ in api.PROBLEM_SPECS.values()
                 for f in s_.flags if "fin" in f}
    assert set(api.FIN_SHAPE_KEYS) <= fin_flags, \
        "the flags are the engine's; only the shell's fields were withdrawn"
    ctx = _with_tail()
    labels = [(getattr(e, "text", "") or "").strip()
              for e in ctx.views[("wing", "type")].descendants()
              if "field-label" in (getattr(e, "_classes", None) or [])]
    assert "volume coefficient V_v" not in labels, labels
    assert not any(lbl.startswith("aspect ratio") for lbl in labels), labels
    # ...and the card still SAYS how it is sized, or withdrawing the fields
    # would have withdrawn the answer with them
    assert "VOLUME COEFFICIENT" in _text(ctx.views[("wing", "type")])


def test_a_stated_shape_still_reaches_the_READ_OUT():
    """The fields are gone; the flags are not, and a card that ignored them
    would describe a fin the design is not flying.

    It did: ``_fin_geometry`` called the sizing law with no shape at all, so
    with ``fin_volume_coeff`` 0.08 stated the card printed the 0.04 fin —
    0.727 m2 where the law says 1.455.
    """
    ctx = _with_tail()
    before = _text(ctx.views[("wing", "type")])
    assert "0.727" in before, before[:400]
    ctx.S["wing"]["flags"]["fin_volume_coeff"] = 0.08
    ctx.render("wing", "type")
    after = _text(ctx.views[("wing", "type")])
    assert "1.45" in after and "0.727" not in after


def test_a_stated_shape_reaches_the_DRAG_and_not_only_the_drawing():
    """The property the whole fin surface exists to have.

    A control that moved the picture while the drag book kept sizing its
    own fin would be the original defect wearing a UI. Both the charged
    parasite drag and the reported geometry must follow the same numbers.
    """
    x = api.PROBLEM_SPECS["tail"].build({}, {}, None).bounds.mean(axis=1)

    def run(flags):
        built = api.PROBLEM_SPECS["tail"].build({}, flags, None)
        cfg = api.RunConfig(problem_name="tail", budget=4, seed=0,
                            flags=flags)
        return (built.evaluate(x), api.design_report(cfg, x)["geometry"]["fin"])

    base_raw, base_fin = run({})
    big_raw, big_fin = run({"fin_volume_coeff": 0.06})
    assert big_fin["S"] > 1.4 * base_fin["S"], "the drawing did not follow"
    assert big_raw["cd0_fin"] > 1.3 * base_raw["cd0_fin"], \
        "the DRAG did not follow — the fin is sized twice again"
    assert big_raw["score"] < base_raw["score"]

    # the aspect ratio reshapes without resizing, and the drag still moves
    ar_raw, ar_fin = run({"fin_ar": 2.0})
    assert ar_fin["S"] == pytest.approx(base_fin["S"], rel=1e-12)
    assert ar_fin["height_m"] > base_fin["height_m"]
    assert ar_raw["cd0_fin"] != base_raw["cd0_fin"]


def test_an_unstated_shape_is_bit_for_bit_the_published_law():
    """Three flags absent must be the identity, or every stored run moved."""
    from aerobo import fin as f
    x = api.PROBLEM_SPECS["tail"].build({}, {}, None).bounds.mean(axis=1)
    a = api.PROBLEM_SPECS["tail"].build({}, {}, None).evaluate(x)
    b = api.PROBLEM_SPECS["tail"].build(
        {}, {"fin_volume_coeff": f.V_V_DEFAULT}, None).evaluate(x)
    assert a["score"] == b["score"]

    class _Bare:
        fin_volume_coeff = fin_ar = fin_tc = None
    assert f.fin_shape_kwargs(_Bare()) == {}


# ----------------------------------------------------- and the body it needs

def test_the_body_is_asked_for_where_the_arm_is_searched():
    """A searched arm with no fuselage is an ill-posed search, and the shell
    is where that is said.

    The arm raises the static margin and the tail volume at almost no cost
    while the body's drag is unmodelled, so the answer is always the longest
    aeroplane in the box. The api accepts the absence — stored records have
    to reproduce — so the warning has to live here.
    """
    ctx = _with_tail()
    name = ctx.S["wing"]["problem"]
    assert api.arm_is_searched(name), "this configuration searches its arm"
    assert "fuselage_diameter_m" in api.PROBLEM_SPECS[name].flags
    text = _text(ctx.views[("wing", "type")])
    assert "Fuselage" in text
    assert "top of the arm band" in text, \
        "the shell does not say what an unstated body costs"


def test_stating_a_body_removes_the_warning_and_changes_the_answer():
    ctx = _with_tail()
    ctx.S["wing"]["flags"]["fuselage_diameter_m"] = 0.35
    ctx.render("wing", "type")
    assert "top of the arm band" not in _text(ctx.views[("wing", "type")])

    # ...and it is not decoration: it moves the score
    x = api.PROBLEM_SPECS["tail"].build({}, {}, None).bounds.mean(axis=1)
    bare = api.PROBLEM_SPECS["tail"].build({}, {}, None).evaluate(x)
    body = api.PROBLEM_SPECS["tail"].build(
        {}, {"fuselage_diameter_m": 0.35}, None).evaluate(x)
    assert body["score"] < bare["score"]
