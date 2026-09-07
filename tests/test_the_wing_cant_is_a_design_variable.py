""""fix no dihedral or sweep. There is no roll stability."

Both halves of that were true and they were one defect. The wing's dihedral
and quarter-chord sweep existed as STATED values (``api.WING_CANT_KEYS``) on
every lattice-backed family, with a card to type them into — and nothing an
optimiser could ever choose. So every design this package searched came back
planar and unswept, whatever the objective was asked for, and a planar wing
has exactly one source of ``Cl_beta``: the fin (plus a tip device, where the
configuration has one). That is a spiral mode no fin SIZE converges — a
bigger fin raises the yaw stiffness and the dihedral effect together — which
is the "roll keeps increasing" this shell has been reporting since session 65
and the "Still open" item every session since has carried forward.

This file pins the freedom and the price of it:

* the free-cant twin of every wing+tail configuration carries TWO design-box
  rows, and a zero cant on them reproduces the stated twin bit-for-bit, so
  no published run moved;
* a dihedral is the only wing-side roll stiffness there is, and it converges
  the spiral where growing the fin does not;
* an L/D objective will not buy one — the composite's ``spiral`` criterion is
  what makes the row worth searching, and the shell says so where the row is
  offered;
* the searched rows are FLOWN: stages 5 and 6 rebuild from the report, so the
  test asserts the rebuilt lattice's own ``Cl_beta`` moves with the row
  rather than that the report echoes it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aerobo import api, wing_score as wsc                        # noqa: E402
from aerobo.dynamics import spiral_margin                        # noqa: E402
from aerobo.flightmodel import build_flight_model                # noqa: E402

FIXED = "tail + winglet"
FREE = "tail + winglet [free cant]"
GAMMA, SWEEP = api.WING_CANT_KEYS


def _built(name: str, flags: dict | None = None, lateral: bool = False):
    built = api.PROBLEM_SPECS[name].build({}, flags or {}, None)
    if lateral:
        built.problem.lateral = True
    return built


def _at(built, **rows):
    """Box centre with named rows moved to stated values."""
    labels = list(built.param_labels)
    x = np.asarray(built.bounds, dtype=float).mean(axis=1)
    for key, value in rows.items():
        x[labels.index(key)] = float(value)
    return x


def _report(name: str, **rows) -> dict:
    built = _built(name)
    cfg = api.RunConfig(problem_name=name, budget=4, seed=0)
    x = _at(built, **rows)
    rep = api.design_report(cfg, x)
    rep["_x"] = x
    return rep


# --------------------------------------------------------- the rows exist

def test_the_free_twin_searches_the_pair_and_the_stated_twin_does_not():
    free = list(api.PROBLEM_SPECS[FREE].param_labels)
    fixed = list(api.PROBLEM_SPECS[FIXED].param_labels)
    assert free[-2:] == [GAMMA, SWEEP], free
    assert [lbl for lbl in fixed if lbl in (GAMMA, SWEEP)] == []
    # ...and the rest of the design is untouched by the freedom
    assert free[:-2] == fixed


def test_the_question_is_asked_once_per_family():
    """A stated value beside a searched row is one question with two
    answers, and the box would silently overwrite the flag."""
    assert set(api.WING_CANT_KEYS) <= set(api.PROBLEM_SPECS[FIXED].flags)
    assert not set(api.WING_CANT_KEYS) & set(api.PROBLEM_SPECS[FREE].flags)
    with pytest.raises(KeyError, match="does not honour"):
        api.check_flags(FREE, {GAMMA: 3.0})
    api.check_flags(FIXED, {GAMMA: 3.0})          # the stated twin takes it


def test_the_problem_itself_refuses_being_told_twice():
    from aerobo.wingtail import WingTailProblem

    with pytest.raises(ValueError, match="stated twice"):
        WingTailProblem(cant_free=True, wing_dihedral_deg=3.0)


def test_cant_is_searched_answers_for_every_registered_problem():
    """The shell's discriminators, checked against the design vectors rather
    than against the list they were derived from.

    ANY, not both: since the freedom split into four states a family can
    search the dihedral and state the sweep, so ``cant_is_searched`` cannot
    be "the dihedral label is present" any more — and the three callers that
    do mean the dihedral ask ``dihedral_is_searched`` instead.
    """
    for name, spec in api.PROBLEM_SPECS.items():
        labels = tuple(spec.param_labels)
        assert api.dihedral_is_searched(name) == (GAMMA in labels), name
        assert api.sweep_is_searched(name) == (SWEEP in labels), name
        assert api.cant_is_searched(name) == (
            GAMMA in labels or SWEEP in labels), name
        # ...and the state NAMES the pair: what it says is searched is
        # exactly what the vector carries, in vector order
        assert api.searched_cant_keys(api.searched_cant(name)) == tuple(
            k for k in api.WING_CANT_KEYS if k in labels), name


def test_a_half_variant_searches_one_row_and_states_the_other():
    """The whole point of splitting the freedom: a design can ask for a
    neutral point without also handing the optimiser a roll lever, and the
    reverse. The row it does NOT search stays an ordinary stated flag."""
    from aerobo.wingtail import WingTailProblem

    for state, searched, stated in (("dihedral", GAMMA, SWEEP),
                                    ("sweep", SWEEP, GAMMA)):
        name = api.wing_tail_problem(winglets="free", tc=False, arm="free",
                                     height="fixed", cant=state)
        spec = api.PROBLEM_SPECS[name]
        assert searched in spec.param_labels, name
        assert stated not in spec.param_labels, name
        assert stated in spec.flags and searched not in spec.flags, name
        # ...and check_flags enforces exactly that split
        api.check_flags(name, {stated: 5.0})
        with pytest.raises(KeyError):
            api.check_flags(name, {searched: 5.0})
    # the dataclass belt under it: the guard is PER ROW
    WingTailProblem(dihedral_free=True, wing_sweep_deg=15.0)
    WingTailProblem(sweep_free=True, wing_dihedral_deg=3.0)
    with pytest.raises(ValueError, match="stated twice"):
        WingTailProblem(dihedral_free=True, wing_dihedral_deg=3.0)
    with pytest.raises(ValueError, match="stated twice"):
        WingTailProblem(sweep_free=True, wing_sweep_deg=15.0)


def test_cant_free_still_means_both_and_survives_a_replace():
    """``cant_free`` is what every published free-cant run was built with.
    It is an alias now, expanded in ``__post_init__`` — which
    ``dataclasses.replace`` re-runs, so the expansion has to be idempotent
    rather than a one-shot."""
    import dataclasses

    from aerobo.wingtail import WingTailProblem

    both = WingTailProblem(dihedral_free=True, sweep_free=True)
    alias = WingTailProblem(cant_free=True)
    assert alias.param_labels == both.param_labels
    again = dataclasses.replace(alias, b=alias.b)
    assert again.dihedral_free and again.sweep_free
    assert again.param_labels == both.param_labels


def test_the_both_state_keeps_the_name_every_published_run_used():
    """1984 registered names, every stored run config and five RESULTS_*
    documents spell it "free cant". The halves get fragments of their own;
    the pair keeps its identity."""
    for name in ("tail [free cant]",
                 "tail + winglet [free cant]",
                 "tail [fixed arm, free height, designed tail, free cant]",
                 "tail [free cant] + CST section (XFOIL)",
                 "tandem (nonplanar) + winglets [free cant]"):
        assert name in api.PROBLEM_SPECS, name
    assert "tail [free dihedral]" in api.PROBLEM_SPECS
    assert "tail [free sweep]" in api.PROBLEM_SPECS


def test_the_two_rows_are_not_transposed():
    """Labels and bands come out of ONE table, so the only way they can
    disagree is if that table is wrong — and then they disagree together,
    which no label-only assertion would see."""
    from aerobo import geometry

    for state in api.WING_CANTS:
        # a tip device, so the "fixed" state is a registered family of its
        # own rather than the lifting-line problem's job
        name = api.wing_tail_problem(winglets="free", tc=False, arm="free",
                                     height="fixed", cant=state)
        spec = api.PROBLEM_SPECS[name]
        box = spec.default_bounds or {}
        rows = api.searched_cant_keys(state)
        assert tuple(l for l in spec.param_labels
                     if l in api.WING_CANT_KEYS) == rows, name
        for row, band in ((GAMMA, geometry.DIHEDRAL_BOUNDS_DEG),
                          (SWEEP, geometry.SWEEP_BOUNDS_DEG)):
            if row in rows:
                assert tuple(box[row]) == tuple(band), (name, row)


def test_a_zero_cant_reproduces_the_stated_family_bit_for_bit():
    """Every published run flew the planar unswept wing. The freedom may
    not move one of them.

    BIT-FOR-BIT WHERE THE PACKAGE IS BIT-REPRODUCIBLE, AND AT THE MEASURED
    FLOOR WHERE IT IS NOT. The GEOMETRY is exact arithmetic and the claim is
    exact there. The FORCES go through ``lu_factor`` and therefore through
    BLAS, which may pick a different reduction order run to run — this tree
    has measured 1.4e-14 of spread in L/D (4.2e-16 relative) on repeated
    identical calls, and this pair lands 1-2 ulp apart in whichever order
    they are called. Asserting ``==`` on those is measuring that floor, not
    this change (it is also how a green test in this file went red an hour
    later for no reason in the session that wrote it). So the forces are
    compared at 1e-14 relative — two orders of magnitude tighter than any
    physical difference a cant row could make, and a hundred times looser
    than the noise.
    """
    fixed = _built(FIXED)
    free = _built(FREE)
    x_fixed = np.asarray(fixed.bounds, dtype=float).mean(axis=1)
    a = fixed.evaluate(x_fixed)
    b = free.evaluate(_at(free, **{GAMMA: 0.0, SWEEP: 0.0}))
    assert b["b_wing"] == a["b_wing"] and b["mac"] == a["mac"]
    assert b["taper"] == a["taper"] and b["b_t"] == a["b_t"]
    assert b["LoD"] == pytest.approx(a["LoD"], rel=1e-14)
    assert b["CDi"] == pytest.approx(a["CDi"], rel=1e-14)
    assert b["CD"] == pytest.approx(a["CD"], rel=1e-14)
    assert b["SM"] == pytest.approx(a["SM"], rel=1e-14)


def test_the_band_reaches_anhedral_and_far_enough_to_converge_the_spiral():
    """A band is a calibration, not a ban — it opens on both signs, because
    anhedral is a real answer — and its top end has to be past the crossing
    or the row cannot buy what it exists to buy."""
    free = _built(FREE, lateral=True)
    lo, hi = np.asarray(free.bounds, dtype=float)[
        list(free.param_labels).index(GAMMA)]
    assert lo < 0.0 < hi
    top = free.evaluate(_at(free, **{GAMMA: hi, SWEEP: 0.0}))
    assert top["spiral_margin"] > 0.0, \
        "the top of the band does not reach a convergent spiral"


# ------------------------------------------- what the dihedral actually buys

@pytest.mark.parametrize("key", [GAMMA])
def test_the_dihedral_is_the_only_wing_side_roll_stiffness(key):
    """Monotone in the row, on the lattice that SCORED the design."""
    free = _built(FREE, lateral=True)
    out = [free.evaluate(_at(free, **{GAMMA: g, SWEEP: 0.0}))
           for g in (-6.0, 0.0, 3.0, 6.0, 12.0)]
    clb = [o["Cl_beta"] for o in out]
    assert all(b < a for a, b in zip(clb, clb[1:])), clb
    margins = [o["spiral_margin"] for o in out]
    assert margins[0] < margins[1] < margins[-1]
    assert margins[1] < 0.0 < margins[-1], \
        "the planar design is not the divergent one this row exists for"


def test_a_bigger_fin_does_not_converge_the_spiral_and_a_dihedral_does():
    """The claim the whole freedom rests on, measured both ways.

    ``Cn_beta`` and ``Cl_beta`` both come off the same surface when the wing
    is planar, and the spiral criterion is their DIFFERENCE of products — so
    growing the fin moves the margin the wrong way or barely at all, while
    the wing's own dihedral moves it monotonically to a crossing.
    """
    fins = []
    for vv in (0.02, 0.04, 0.08, 0.16):
        built = _built(FREE, flags={"fin_volume_coeff": vv}, lateral=True)
        out = built.evaluate(_at(built, **{GAMMA: 0.0, SWEEP: 0.0}))
        fins.append((vv, out["Cn_beta"], out["spiral_margin"]))
    assert all(b[1] < c[1] for b, c in zip(fins, fins[1:])), fins
    assert all(m < 0.0 for _v, _c, m in fins), \
        f"a planar wing found a convergent spiral from fin size alone: {fins}"

    free = _built(FREE, lateral=True)
    crossed = [g for g in (0.0, 3.0, 6.0, 9.0, 12.0)
               if free.evaluate(_at(free, **{GAMMA: g,
                                             SWEEP: 0.0}))["spiral_margin"] > 0]
    assert crossed, "no dihedral in the band converges the spiral"


def test_sweep_moves_the_neutral_point_aft_and_charges_for_it():
    """Not a roll lever, a MARGIN lever — and the read on it is the price."""
    free = _built(FREE)
    a = free.evaluate(_at(free, **{GAMMA: 0.0, SWEEP: 0.0}))
    b = free.evaluate(_at(free, **{GAMMA: 0.0, SWEEP: 20.0}))
    assert b["x_np"] > a["x_np"]
    assert b["SM"] > a["SM"]
    assert b["LoD"] < a["LoD"]


# ------------------------------------------ what has to price it (the gate)

def _composite(weights: dict):
    ref = api.wing_score_reference(
        FREE, n=32, seed=0,
        flags={"wing_objective": "composite", "wing_score_weights": weights})
    flags = {"wing_objective": "composite", "wing_score_weights": weights,
             "wing_score_reference": ref}
    return _built(FREE, flags=flags)


def test_an_l_over_d_objective_will_not_buy_a_dihedral_and_the_spiral_one_does():
    """The gate, asserted as the SIGN OF THE GRADIENT at the planar wing.

    A search is not needed to show which way each objective pulls, and a
    slope is what a search would be riding: with L/D alone the composite is
    flat-to-falling in the dihedral at zero (a cant costs projected span and
    buys nothing the score reads), and with the spiral criterion weighted it
    RISES — so the row has something to buy and the run goes and buys it.
    """
    plain = _composite({"lod": 1.0})
    lateral = _composite({"lod": 0.7, "spiral": 0.3})
    assert plain.problem.lateral is False
    assert lateral.problem.lateral is True, \
        "the weight did not arm the deck the criterion is measured on"

    def slope(built, h=3.0):
        up = built.evaluate(_at(built, **{GAMMA: +h, SWEEP: 0.0}))
        dn = built.evaluate(_at(built, **{GAMMA: 0.0, SWEEP: 0.0}))
        return float(up["score"] - dn["score"])

    assert slope(plain) < 0.0, "L/D alone appears to pay for a dihedral"
    assert slope(lateral) > 0.0, \
        "the spiral criterion does not pay for the dihedral it needs"


def test_the_criterion_stops_paying_once_the_spiral_converges():
    """min(margin, 0) — otherwise the row is a ratchet into its own bound."""
    lateral = _composite({"lod": 0.7, "spiral": 0.3})
    out = [lateral.evaluate(_at(lateral, **{GAMMA: g, SWEEP: 0.0}))
           for g in (9.0, 12.0, 15.0)]
    assert all(o["spiral_margin"] > 0 for o in out), \
        "this test needs three CONVERGENT designs to compare"
    m = [wsc.design_metrics(o, lateral.problem, {})["spiral"] for o in out]
    assert m == [0.0, 0.0, 0.0]
    # ...so past the crossing the score follows L/D alone, and L/D falls
    assert out[0]["LoD"] > out[-1]["LoD"]
    assert out[0]["score"] > out[-1]["score"]


# -------------------------------------------------- stated is not flown

def test_the_searched_cant_is_FLOWN_by_stages_5_and_6():
    """The rebuild reads the report, so the row has to survive the round
    trip into a lattice nobody handed the design vector to."""
    decks = {}
    for g in (0.0, 6.0, 12.0):
        rep = _report(FREE, **{GAMMA: g, SWEEP: 0.0})
        assert rep["breakdown"][GAMMA] == pytest.approx(g)
        d = build_flight_model(rep).deck
        decks[g] = (d.Cl_beta, spiral_margin(Cl_beta=d.Cl_beta, Cn_r=d.Cn_r,
                                             Cn_beta=d.Cn_beta, Cl_r=d.Cl_r))
    clb = [decks[g][0] for g in (0.0, 6.0, 12.0)]
    assert all(b < a for a, b in zip(clb, clb[1:])), clb
    assert decks[0.0][1] < 0.0 < decks[12.0][1], decks


def test_the_sweep_reaches_the_flown_lattice_too():
    a = build_flight_model(_report(FREE, **{GAMMA: 0.0, SWEEP: 0.0}))
    b = build_flight_model(_report(FREE, **{GAMMA: 0.0, SWEEP: 20.0}))
    assert b.deck.Cm_alpha < a.deck.Cm_alpha, \
        "20 deg of sweep did not stiffen the rebuilt pitch response"


# ------------------------------------------------------------ the shell

def _shell_with_tail():
    """A wing+tail configuration whose family REFUSES the cant.

    The shell used to open on one. Since "the V3 shell opens on a whole
    aeroplane" it opens with the tail's HEIGHT free, which already makes
    the family lattice-backed — so the routes below are not drawn on the
    default any more and there is nothing to press. A FIXED tail height is
    an ordinary answer and lands back on the refusal, which is the state
    these tests are about.
    """
    from gui.v3.app import assemble

    ctx = assemble("air")
    ctx.act("set_choice", "tail", True)
    ctx.act("set_choice", "tail_height", "fixed")
    assert not api.cant_is_searched(ctx.S["wing"]["problem"])
    return ctx


def _text(view) -> str:
    return " ".join((getattr(e, "text", "") or "")
                    for e in view.descendants())


def _press(view, label: str) -> None:
    btn = next((e for e in view.descendants()
                if type(e).__name__ == "Button"
                and (getattr(e, "text", "") or "").strip() == label), None)
    assert btn is not None, f"no button {label!r}"
    for listener in (getattr(btn, "_event_listeners", None) or {}).values():
        if listener.type == "click":
            try:
                listener.handler(None)
            except TypeError:
                listener.handler()
            return
    raise AssertionError(f"{label!r} has no click handler")


def test_the_search_is_reachable_from_the_configuration_the_shell_opens_on():
    """The answer a user asking for roll stability actually wants, taken
    through the card's own either/or.

    It used to be a BUTTON on the refusal card of a family that cannot
    score a cant. That card is gone — the user asked for it after meeting
    it on a hydrofoil — so what is pinned here is the path that remains
    and the one that matters: the configuration V3 OPENS ON carries the
    rows, its card asks state-or-optimise, and taking "optimise them"
    produces the searched family. A fixed tail height reaches the same
    place by first giving the height back, which is an ordinary answer on
    the same card.
    """
    from gui.v3.app import assemble

    ctx = assemble("air")
    ctx.act("set_choice", "tail", True)
    ctx.render("wing", "type")
    assert "Wing cant and sweep" in _text(ctx.views[("wing", "type")])
    ctx.act("set_choice", "wing_cant", "free")
    name = ctx.S["wing"]["problem"]
    assert api.cant_is_searched(name), name
    assert GAMMA in tuple(api.PROBLEM_SPECS[name].param_labels)


def test_taking_the_search_keeps_every_other_answer():
    from gui.v3.app import assemble

    ctx = assemble("air")
    ctx.act("set_choice", "tail", True)
    ctx.act("set_choice", "tail_design", "planform")
    ctx.act("set_choice", "wing_cant", "free")
    assert ctx.S["wing"]["choices"]["tail_design"] == "planform"
    assert api.cant_is_searched(ctx.S["wing"]["problem"])


def _toggle_labels(view) -> list[list[str]]:
    return [[o["label"] for o in (t._props.get("options") or [])]
            for t in view.descendants() if type(t).__name__ == "Toggle"]


def test_the_card_asks_state_or_optimise_like_the_separations_do():
    ctx = _shell_with_tail()
    ctx.act("set_choice", "wing_cant", "free")
    ctx.render("wing", "type")
    view = ctx.views[("wing", "type")]
    assert ["you state them", "optimise the dihedral", "optimise the sweep",
            "optimise both"] in _toggle_labels(view), \
        "the cant is not asked as the either/or the separations are"
    assert "the solver searches" in _text(view)


@pytest.mark.parametrize("state,searched,stated",
                         [("dihedral", "dihedral", "quarter-chord sweep"),
                          ("sweep", "quarter-chord sweep", "dihedral")])
def test_a_half_variant_draws_a_BAND_and_a_TYPED_field_on_one_card(
        state, searched, stated):
    """The card has to say both halves at once now: the searched row's band,
    and the buttons that state the other. Drawing only one of them is how a
    user concludes the freedom they did not ask for went missing."""
    ctx = _shell_with_tail()
    ctx.act("set_choice", "wing_cant", state)
    ctx.render("wing", "type")
    text = _text(ctx.views[("wing", "type")])
    assert f"{searched}: the solver searches" in text
    assert "…or state it" in text or "or state it" in text
    # ...and the sweep-only card must NOT warn about a dihedral SIGN nothing
    # is searching
    if state == "sweep":
        assert "NOTHING IN THIS OBJECTIVE PRICES" not in text


def test_the_card_says_when_nothing_in_the_objective_prices_it():
    """A row with nothing to buy is worse than no row: the run comes back
    planar and the user reads that as the feature not working."""
    ctx = _shell_with_tail()
    ctx.act("set_choice", "wing_cant", "free")
    ctx.render("wing", "type")
    # anchored on the shortest fragment carrying the CLAIM, not on the
    # whole sentence: the wording moved to "...PRICES THE SIGN OF THIS ROW"
    # when the row was floored, and an anchor on prose goes quietly missing
    assert "NOTHING IN THIS OBJECTIVE PRICES" in _text(
        ctx.views[("wing", "type")])


def test_the_searched_rows_are_on_the_design_box_with_their_own_help():
    ctx = _shell_with_tail()
    ctx.act("set_choice", "wing_cant", "free")
    ctx.render("wing", "box")
    text = _text(ctx.views[("wing", "box")])
    assert GAMMA in text and SWEEP in text
    assert "anhedral" in text        # param_help, not the raw label alone


def test_the_family_inverts_back_to_the_choice_that_made_it():
    """The round trip the menus are built on.

    ``option_available`` applies a choice, derives the family and reads the
    choices back OFF that family — so a free-cant twin that inverts to
    "fixed" makes the shell grey out the very option it just took, and a
    preset saved on one reopens as the other. Neither is visible from the
    card, which is why it is asserted here.
    """
    from gui import nice_app as v1

    assert v1.choices_from_problem(FREE)["wing_cant"] == "free"
    assert v1.choices_from_problem(FIXED)["wing_cant"] == "fixed"
    ch = dict(v1.BUILDER_DEFAULTS)
    ch["tail"] = True
    assert v1.option_available(ch, "wing_cant", "free")
    assert v1.option_available(dict(ch, wing_cant="free"), "wing_cant",
                               "fixed")


def test_taking_the_search_leaves_the_shell_on_a_stable_configuration():
    """choices -> family -> choices, on the state the toggle wrote: an
    answer that does not survive its own inverse re-derives to a different
    problem the next time anything touches the menus."""
    from gui import nice_app as v1
    from gui.v3.app import assemble

    ctx = assemble("air")
    ctx.act("set_choice", "tail", True)
    ctx.act("set_choice", "wing_cant", "free")
    ch = dict(ctx.S["wing"]["choices"])
    name = ctx.S["wing"]["problem"]
    assert ch["wing_cant"] == "free"
    assert v1.derive_problem(ch)[0] == name
    assert v1.choices_from_problem(name)["wing_cant"] == "free"


def test_a_builder_switch_drops_a_cant_the_new_family_cannot_search():
    """Rule 6, the stale-menu rule: state may never hold a value its own
    menu no longer offers. The cant SELECTS a family, so it is normalised
    like every other family-selecting control — otherwise switching to the
    water craft leaves "optimise them" in the session with nothing behind
    it, and the next thing to read the choices derives a different problem.
    """
    from gui import nice_app as v1

    ch = dict(v1.BUILDER_DEFAULTS)
    ch["tail"], ch["wing_cant"] = True, "free"
    assert v1.choices_consistent(ch)
    ch["medium"] = "water"
    assert not v1.choices_consistent(ch)
    assert "wing_cant" in v1.normalise_choices(ch)
    assert ch["wing_cant"] == "fixed"


def test_the_shell_config_BUILDS_a_problem_that_searches_the_cant():
    """The whole path, once: the card's choice -> the session -> the dict
    ``api.run`` is handed -> a built problem whose box has the two rows.

    A stage that renders is not a stage that runs. The chord modifier's
    coefficients sit AFTER the cant rows, so this also pins the stacking:
    the family block first, the modifier blocks after it.
    """
    from gui.v3 import config
    from gui.v3.app import assemble

    ctx = assemble("air")
    ctx.act("set_choice", "tail", True)
    ctx.act("set_choice", "wing_cant", "free")
    d = config.cfg_dict(ctx.S)
    api.check_flags(d["problem_name"], d["flags"])        # raises if not
    assert not set(api.WING_CANT_KEYS) & set(d["flags"])
    built = api.PROBLEM_SPECS[d["problem_name"]].build(
        d.get("mission_kwargs") or {}, d["flags"],
        d.get("bounds_overrides"))
    labels = list(built.param_labels)
    assert GAMMA in labels and SWEEP in labels, labels
    assert labels.index(SWEEP) == labels.index(GAMMA) + 1
    assert labels[-1].startswith("chord_"), \
        "the modifier block no longer trails the family's own rows"
    box = np.asarray(built.bounds, dtype=float)
    lo, hi = box[labels.index(GAMMA)]
    # THE BOX THE RUN GETS IS FLOORED AT 0 while nothing prices the sign
    # (``gui.v3.session.CANT_FLOOR_DEG``, and
    # ``tests/test_the_searched_cant_has_a_floor.py`` for why) — so this
    # asserts what the run is actually handed, and separately that the
    # FAMILY's published band still spans both signs, which is what the
    # floor has to have something to floor.
    from gui.v3 import session as v3s

    assert lo == pytest.approx(v3s.CANT_FLOOR_DEG)
    assert hi > 0.0
    published = api.PROBLEM_SPECS[d["problem_name"]].default_bounds[GAMMA]
    assert float(published[0]) < 0.0 < float(published[1]), published


def test_a_family_that_cannot_search_it_says_so_rather_than_dropping_it():
    from gui import nice_app as v1

    ch = dict(v1.BUILDER_DEFAULTS)
    ch["wing_cant"] = "free"
    name, notes = v1.derive_problem(ch)
    assert not api.cant_is_searched(name)
    assert any("dihedral" in n for n in notes), notes


def test_the_3d_view_DRAWS_the_dihedral_the_search_chose():
    """A change the user cannot SEE is not delivered — the lesson V5 paid
    for, when a fin reached the STL and the exporter and never reached the
    picture. The wing's own lofted surface has to climb with the row while
    the fin, which the row does not touch, stays exactly where it was.
    """
    from gui import nice_app as v1

    def drawn(gamma: float) -> dict:
        rep = _report(FREE, **{GAMMA: gamma, SWEEP: 0.0})
        fig = v1.fig_wing3d(rep["geometry"], rep["_x"], rep["param_labels"])
        out = {}
        for trace in fig.data:
            z = getattr(trace, "z", None)
            if z is None:
                continue
            name = getattr(trace, "name", None) or "wing"
            z = np.asarray(z, dtype=float)
            out.setdefault(name, []).append((float(z.min()), float(z.max())))
        return {k: (min(a for a, _b in v), max(b for _a, b in v))
                for k, v in out.items()}

    flat, canted = drawn(0.0), drawn(12.0)
    b = _report(FREE, **{GAMMA: 0.0, SWEEP: 0.0})["geometry"]["b"]
    rise = canted["wing"][1] - flat["wing"][1]
    assert rise > 0.5 * (b / 2.0) * np.tan(np.deg2rad(12.0)), \
        f"the wing was drawn flat: {flat['wing']} -> {canted['wing']}"
    assert canted["fin"] == flat["fin"], "the fin moved with the wing's cant"


# ------------------------------------- what stage 5 says about the spiral now
#
# The scan is run on the family with NO TIP DEVICE, and that is the whole
# point of the choice: a canted winglet carries 88 % of Cl_beta on the
# `tail + winglet` design, so a fin CAN converge its spiral and the advice
# below never fires there. Strip the device and the wing has nothing
# out-of-plane left — which is the aeroplane the user was flying when they
# reported that the roll just keeps increasing.

NO_DEVICE = "tail [free cant]"


def _v4_with(report: dict):
    """A V4 shell holding a finished design, stage 5 ready to render."""
    from gui.v4 import app as v4app

    ctx = v4app.assemble()
    ctx.S["run"]["record"] = {"pretend": "a completed run"}
    ctx.S["run"]["report"] = report
    return ctx


def _scan(report: dict):
    ctx = _v4_with(report)
    ctx.render("controls", "vertical")
    _press(ctx.views[("controls", "vertical")],
           "does any fin size fix the spiral?")
    ctx.render("controls", "vertical")
    return ctx, _text(ctx.views[("controls", "vertical")])


def test_the_spiral_advice_reads_the_dihedral_the_design_IS_FLYING():
    """The advice was a constant sentence — "the lattice wing carries no
    dihedral" — written when no wing could have one. It is opposite advice
    on two designs that differ only by a searched row, so it has to be read
    off the built wing."""
    _ctx, planar = _scan(_report(NO_DEVICE, **{GAMMA: 0.0, SWEEP: 0.0}))
    assert "No." in planar, planar[:400]
    assert "carries NO dihedral" in planar
    assert "Wing cant and sweep" in planar, \
        "the advice does not say where the dihedral is asked for"

    # ...and the OTHER half of the sentence, on a wing carrying a small
    # dihedral that is not yet enough (0.25 deg: the same "No" branch, so
    # the two texts differ only in what the wing is flying)
    _ctx, canted = _scan(_report(NO_DEVICE, **{GAMMA: 0.25, SWEEP: 0.0}))
    assert "No." in canted, canted[:400]
    assert "carries NO dihedral" not in canted
    assert "+0.25 deg of dihedral already" in canted
    assert "the answer is MORE of it" in canted

    # ...and where the dihedral IS enough, the panel stops advising at all
    _ctx, enough = _scan(_report(NO_DEVICE, **{GAMMA: 9.0, SWEEP: 0.0}))
    assert "carries NO dihedral" not in enough


def _put_on(S: dict, problem: str) -> None:
    """Put a session on ``problem`` the way the shell does — CHOICES too.

    ``session.cant_is_answerable`` asks the registry what this
    CONFIGURATION can be switched to, not only what its current family
    declares, so a problem name injected over another family's choices is a
    question about neither. Every one of these round-trips
    (``derive_problem(choices_from_problem(n)) == n``).
    """
    from gui.nice_app import choices_from_problem

    S["wing"]["problem"] = problem
    S["wing"]["choices"].update(choices_from_problem(problem))


def test_the_spiral_advice_never_names_a_card_that_is_not_drawn():
    """Stage 5 may only send a user to stage 3's cant card where it exists.

    The card is drawn where the cant can be ANSWERED — as the two stated
    flags, as the two searched rows, or by switching to a twin that carries
    them — and a configuration that can do none of the three is asked
    nothing about a cant, no heading and no button (the user's ask, after
    meeting the old refusal card on a hydrofoil). The advice here is the
    one place that names that card by its heading, so it has to branch on
    the same predicate (``session.cant_is_answerable``); otherwise it sends
    exactly the users who most need the dihedral to a card that is not
    there.

    THE PAIR IS THE THIRD ARM and it is why the predicate reads the
    choices. A lifting line has no out-of-plane geometry, so `tandem`
    declares neither key and searches neither row — and it was refused the
    card on that reading, while its nonplanar twin carries both rows and
    this card is the only control that selects one. Stage 5 then told the
    one family whose spiral no fin size can turn that there was no field
    for the answer.

    Both halves for each, and the stage-3 view is rendered every time so
    this cannot pass against a shell that has stopped drawing the card at
    all.
    """
    from gui.v3.app import assemble

    report = _report(NO_DEVICE, **{GAMMA: 0.0, SWEEP: 0.0})
    for problem, drawn in ((NO_DEVICE, True), ("tandem", True),
                           ("trim wing", False)):
        v3 = assemble("air")
        _put_on(v3.S, problem)
        v3.render("wing", "type")
        assert ("Wing cant and sweep" in _text(v3.views[("wing", "type")])) \
            is drawn, problem

        ctx = _v4_with(report)
        _put_on(ctx.S, problem)
        ctx.render("controls", "vertical")
        _press(ctx.views[("controls", "vertical")],
               "does any fin size fix the spiral?")
        ctx.render("controls", "vertical")
        text = _text(ctx.views[("controls", "vertical")])
        assert "carries NO dihedral" in text, text[:400]
        assert ("Wing cant and sweep" in text) is drawn, text[-600:]
        if not drawn:
            assert "no field to give it one" in text, text[-600:]


def test_the_pair_searches_its_dihedral_beside_a_free_span():
    """The user's report, driven through the controls that answer it.

    "Doesn't let the tandem wing optimise dihedral and free span at the same
    time" was ONE control, not a missing solver: every combination of the
    pair's cant with every sizing mode has been a registered problem for
    some time (``api.tandem_vlm_problem`` crossed with the size modifiers),
    and the shell could reach none of them. Stage 3 draws exactly one
    control that selects a free-cant twin, and it drew it only where the
    family already declared the two stated flags — which a lifting line
    never can, having no out-of-plane geometry for them to act on. So the
    pair had no field to type a dihedral into (correct: its solver cannot
    fly one) and no toggle to search one with (not correct: its nonplanar
    twin exists for that).

    Driven end to end because both halves of the failure are in the shell:
    the card has to be DRAWN, its switch has to land on a twin that
    actually searches the row, and the sizing menu has to still compose
    with it afterwards.
    """
    from gui.v3.app import assemble

    v3 = assemble("air")
    v3.act("set_choice", "system", "tandem")
    lifting_line = v3.S["wing"]["problem"]
    assert lifting_line.startswith("tandem"), lifting_line
    # the state the card used to refuse: neither stated nor searched
    assert not api.cant_is_searched(lifting_line)
    assert not any(k in api.PROBLEM_SPECS[lifting_line].flags
                   for k in api.WING_CANT_KEYS)

    v3.render("wing", "type")
    assert "Wing cant and sweep" in _text(v3.views[("wing", "type")]), \
        "the pair is asked nothing about a dihedral it can be given"

    # ...and the switch answers by SELECTING the twin, because there is no
    # flag on this family for it to write
    v3.act("set_wing_cant", True)
    assert api.dihedral_is_searched(v3.S["wing"]["problem"]), \
        v3.S["wing"]["problem"]

    # ...and the size question still composes with it, which is the half of
    # the report that named two freedoms
    v3.act("set_planform", "wing_loading")
    both = v3.S["wing"]["problem"]
    labels = api.PROBLEM_SPECS[both].param_labels
    assert api.WING_CANT_KEYS[0] in labels, labels
    assert {"b_m", "b_rear_m"} <= set(labels), labels


def test_no_fin_size_converges_the_planar_wing_and_a_dihedral_does():
    """The user's report, measured through the panel that answers it."""
    ctx, planar = _scan(_report(NO_DEVICE, **{GAMMA: 0.0, SWEEP: 0.0}))
    assert ctx.S["controls"]["spiral_scan"]["best"] is not None
    assert ctx.S["controls"]["spiral_scan"]["best"][2] < 0.0
    ctx, _canted = _scan(_report(NO_DEVICE, **{GAMMA: 9.0, SWEEP: 0.0}))
    assert ctx.S["controls"]["spiral_scan"]["best"][2] > 0.0


def test_the_scan_never_recommends_a_fin_that_yaws_the_wrong_way():
    """A spiral margin bought by taking Cn_beta NEGATIVE is arithmetic, not
    stability — the criterion is a difference of two products and the sign
    flips when the yaw stiffness does. The panel used to offer exactly that:
    on the box-centre `tail + winglet` its best of ten was a fin at 1 % of
    span, margin +0.000566, Cn_beta -0.00707.
    """
    ctx, text = _scan(_report("tail + winglet [free cant]",
                              **{GAMMA: 0.0, SWEEP: 0.0}))
    scan = ctx.S["controls"]["spiral_scan"]
    assert scan["unstable_wins"], \
        "this design no longer has the case the filter exists for"
    assert scan["best"][3] > 0.0, "an unstable fin was recommended"
    assert "yaws AWAY from the airflow" in text, \
        "the rejected candidates are dropped silently"
