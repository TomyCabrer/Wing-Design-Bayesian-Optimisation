"""A refusal that names the wrong solver is a refusal about another design.

Stage 3's "Wing cant and sweep" card refuses the question on every family
that cannot score a dihedral, and it gave all of them ONE reason:

    "This configuration solves its wing on a LIFTING LINE, which has no
     out-of-plane geometry — a dihedral would have nothing to act on."

That is true of the tail and tandem twins the air shell lands on. It is
FALSE of most of the rest, and it is false in the one place the user asked
about. Measured at each family's own box centre by whether its evaluation
returns a lattice at all (``"vlm" in out``):

    hydrofoil                        lattice False   the reason holds
    tail (fixed arm) / tandem        lattice False   the reason holds
    hydrofoil + elevator             lattice TRUE
    hydrofoil + winglet              lattice TRUE
    car rear wing (+ endplates)      lattice TRUE

156 water families and every car rear wing were being told about a solver
they are not running — and then told "A wing with a tail is the
configuration that does" while carrying an elevator.

THE REASON THAT IS TRUE THERE was already measured, one card away, by
``session.spiral_dihedral``: a foiling craft's only vertical surface is the
MAST that carries the foil, it stands at the foil with no yaw arm, and with
an elevator the CG moves AFT of it — so ``Cn_beta`` comes out NEGATIVE and
there is no yaw stiffness for a dihedral to race against. The last test
here pins that fact on the engine, because it is also the reason the water
families must not be given a searched cant row: ``wing_score.spiral_refusal``
refuses the only criterion that could price one.
"""
from __future__ import annotations

import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aerobo import api                                          # noqa: E402


def _text(view) -> str:
    return " ".join((getattr(e, "text", "") or "")
                    for e in view.descendants())


def _card(medium: str, **choices) -> str:
    """The stage-3 geometry card's text for a configuration."""
    from gui.nice_app import derive_problem
    from gui.v3.app import assemble

    ctx = assemble(medium)
    if choices:
        ctx.S["wing"]["choices"].update(choices)
        ctx.S["wing"]["problem"] = derive_problem(ctx.S["wing"]["choices"])[0]
    ctx.render("wing", "type")
    return _text(ctx.views[("wing", "type")])


def _has_lattice(name: str) -> bool:
    """Does this family's own box-centre evaluation return a lattice?

    The question the card was answering by assumption, asked of the solver
    instead. A refusal is not consulted: an infeasible centre would answer
    "no lattice" for a family that has one, so this asserts feasibility
    first and every caller below is on a feasible centre.
    """
    spec = api.PROBLEM_SPECS[name]
    built = spec.build({}, {}, None)
    bnds = built.problem.bounds
    out = built.evaluate(0.5 * (bnds[:, 0] + bnds[:, 1]))
    assert out.get("feasible"), f"{name}: {out.get('reason')}"
    return "vlm" in out


# --------------------------------------------------------------- the water

def test_a_hydrofoil_with_an_elevator_runs_a_lattice():
    """The premise. Without this the next test passes for the wrong reason."""
    assert _has_lattice("hydrofoil + elevator")


def test_it_is_not_told_that_it_solves_its_wing_on_a_lifting_line():
    text = _card("water", tail=True)
    assert "Wing cant and sweep" in text
    assert "LIFTING LINE" not in text, \
        "a lattice-backed family is told it has no out-of-plane geometry"


def test_the_water_cant_card_names_the_mast_as_the_reason():
    """...and names it as the REASON, not as a footnote elsewhere.

    The card used to REFUSE the question outright and name the mast as why.
    It now offers the STATED pair — the imaged lattice can score a cant, and
    measured at the box centre a sweep buys cavitation margin and moves the
    neutral point — so the mast is no longer why there is no question. It is
    why the question is not the one an air user is used to: not a roll or
    spiral lever, because the one vertical surface this craft has stands
    ahead of the CG and its yaw stiffness is negative. The card must still
    say that, in the same place, or a user reads the air card's meaning onto
    a water design.
    """
    text = _card("water", tail=True)
    assert "Wing cant and sweep" in text
    assert "mast" in text.lower(), text[:600]
    assert "AHEAD of the CG" in text, text[:600]
    # and it must NOT sell the row as the air card does
    assert "spiral MODE into a convergent one" not in text


def test_a_craft_that_already_has_a_tail_is_not_told_to_add_one():
    text = _card("water", tail=True)
    assert "A wing with a tail is the configuration that does" not in text


def test_the_bare_hydrofoil_is_asked_nothing_about_a_cant():
    """THE CARD IS GONE where the family cannot score a cant.

    It used to name the measured reason (the mast, not the solver) and then
    draw a button that changed the family to one that could answer. Asked
    for by the user, on this family: a heading that appears only to say
    "not here" and a button whose only action is to become a different
    aeroplane are not the question the heading asks.

    Both halves, so this cannot pass on a shell that draws nothing anywhere:
    the registry must agree the family carries no cant, and the card must be
    absent — while `hydrofoil + winglet`'s card, asserted above, is still
    drawn.
    """
    from gui.v3.app import assemble

    ctx = assemble("water")
    name = ctx.S["wing"]["problem"]
    assert not set(api.WING_CANT_KEYS) <= set(api.PROBLEM_SPECS[name].flags)
    assert not api.cant_is_searched(name)
    text = _card("water")
    assert "Wing cant and sweep" not in text, text[:600]
    assert "LIFTING LINE" not in text
    assert "add a tip device" not in text


# ---------------------------------------------------------------- the track

def test_a_car_rear_wing_runs_a_lattice_and_is_told_nothing_at_all():
    """It has the lattice and NOT the rows, so it gets no card either.

    The lattice half stays asserted because it is what made the old
    sentence wrong; what changed is that a family which cannot state the
    cant is now silent about it rather than explaining itself.
    """
    assert _has_lattice("car rear wing + endplates")
    text = _card("track")
    assert "LIFTING LINE" not in text
    assert "Wing cant and sweep" not in text, text[:600]


# ------------------------------------------------------------------ the air

def test_the_air_twin_that_really_is_a_lifting_line_still_says_no_reason():
    """The SENTENCE is deleted, in air as in water — the heading is not.

    The card was taken away "wherever the cant cannot be scored", and this
    family reads as if it qualifies: its own solver is a lifting line and
    it declares no cant rows. It does not qualify. One toggle away is
    ``tail [designed tail, free cant]``, the same aeroplane with the cant
    searched, so the question IS scoreable here and the card asks it. What
    the user objected to was the refusal prose and the route buttons, and
    both are still gone.
    """
    from gui.v3.app import assemble

    ctx = assemble("air")
    ctx.act("set_choice", "tail", True)
    ctx.act("set_choice", "tail_height", "fixed")
    name = ctx.S["wing"]["problem"]
    assert not set(api.WING_CANT_KEYS) <= set(api.PROBLEM_SPECS[name].flags)
    assert not _has_lattice(name.split(" + free chord law")[0]), name
    ctx.render("wing", "type")
    text = _text(ctx.views[("wing", "type")])
    assert "Wing cant and sweep" in text, text[:600]
    assert "LIFTING LINE" not in text


# ------------------------------------------- and why there is no cant row

def test_a_hydrofoils_only_vertical_surface_stands_AHEAD_of_its_cg():
    """The engine fact that keeps the water families out of the cant rows.

    ``fin.mast`` puts the strut's quarter chord at the FOIL (x = 0) and the
    elevator puts the CG aft of it, so the one vertical surface a foiling
    craft has is DEstabilising. Asserted on the shipped bridge, on the
    shipped design report — not on a hand-built lattice — because it is the
    number stage 5 and stage 6 print.
    """
    from aerobo import flightmodel

    name = "hydrofoil + elevator"
    spec = api.PROBLEM_SPECS[name]
    built = spec.build({}, {}, None)
    bnds = built.problem.bounds
    x = 0.5 * (bnds[:, 0] + bnds[:, 1])
    rep = api.design_report(api.RunConfig(problem_name=name), x)
    deck = flightmodel.build_flight_model(rep).deck
    assert deck.Cn_beta < 0.0, deck.Cn_beta
    # ...and the arm is the reason: the strut is ahead of the CG.
    out = built.evaluate(x)
    from aerobo import fin as _fin
    mast = _fin.mast(depth=out["depth"], chord=built.problem.c_mast)
    assert mast is not None and mast.x_qc < out["x_cg"], (mast, out["x_cg"])


def test_the_spiral_criterion_refuses_that_design_rather_than_scoring_it():
    """So a searched cant row on a water family would buy NOTHING.

    ``min(margin, 0)`` on a negative ``Cn_beta`` is arithmetic — the guard
    ``wing_score.spiral_refusal`` exists for exactly this — so the row would
    ride a bound the way a cant with nothing to price it always has.
    """
    from aerobo import dynamics as dyn, flightmodel, wing_score

    name = "hydrofoil + elevator"
    spec = api.PROBLEM_SPECS[name]
    built = spec.build({}, {}, None)
    bnds = built.problem.bounds
    x = 0.5 * (bnds[:, 0] + bnds[:, 1])
    rep = api.design_report(api.RunConfig(problem_name=name), x)
    deck = flightmodel.build_flight_model(rep).deck
    raw = {"Cn_beta": float(deck.Cn_beta),
           "spiral_margin": float(dyn.spiral_margin_of(deck))}
    # the margin is POSITIVE and the criterion still refuses it — which is
    # the whole point: a positive number here is the wrong sign twice over
    assert raw["spiral_margin"] > 0.0, raw
    assert wing_score.spiral_refusal(raw), raw
    assert wing_score._spiral(raw) is None


def test_no_water_family_SEARCHES_a_cant_and_only_the_lattice_ones_state_it():
    """The guard, narrowed to what the two tests above actually establish.

    THEY ARE ABOUT THE SPIRAL, and only about the spiral: the mast stands
    ahead of the CG, ``Cn_beta`` is negative, and the criterion that would
    price a SEARCHED cant refuses the design rather than scoring it. That
    argument still holds exactly as written, so a searched cant ROW is still
    refused on every water family and this test still says so.

    It does NOT establish that a foil's cant does nothing — and it does not,
    measured at the box centre of the two imaged-lattice families:

        hydrofoil + winglet   sweep 0 -> 20    cavitation margin
                              0.34955 -> 0.39747 for 1.9 % of L/D
                              dihedral -5      L/D 29.7673 -> 29.8286,
                              draught 0.575 -> 0.631 m
        hydrofoil + elevator  sweep 0 -> 20    L/D 24.6728 -> 25.4343 AND
                              the static margin -0.30759 -> +0.47401,
                              i.e. an infeasible box centre made feasible

    So the STATED pair is declared on the families that can score it, which
    is the order this package uses everywhere: state it, measure what it
    buys, and only then decide whether it is worth a design row. The planar
    families still refuse both — a monoplane Fourier solve has a straight
    quarter-chord line and a dihedral there would have nothing to act on.
    """
    water = [n for n, s in api.PROBLEM_SPECS.items() if s.medium == "water"]
    assert len(water) >= 156, len(water)
    lattice = 0
    for name in water:
        spec = api.PROBLEM_SPECS[name]
        # THE ROW IS STILL REFUSED, everywhere, for the spiral reason above
        assert not api.cant_is_searched(name), name
        stated = set(api.WING_CANT_KEYS) <= set(spec.flags)
        # ...and the STATED pair is declared exactly where a lattice can
        # score it, never on a lifting line
        if stated:
            lattice += 1
        else:
            assert not (set(api.WING_CANT_KEYS) & set(spec.flags)), (
                f"{name} declares HALF the cant pair")
    assert lattice == 150, lattice


# ------------------------------------------------------------- the tandem

def test_the_tandem_really_is_two_lifting_lines_and_is_asked_anyway():
    """The published pair really is two lifting lines — asserted on the
    solver, which is why it has no cant rows of its own.

    THE CARD IS STILL DRAWN, and this is the correction the user's report
    forced: "a pair carries no cant rows" was read as "a pair cannot be
    asked", and the card that would have selected the pair's NONPLANAR twin
    is the only control in the shell that writes ``wing_cant``. So the pair
    had no field to type a dihedral into (right — a lifting line cannot fly
    one) and no toggle to search one with (wrong — 240 registered
    ``tandem (nonplanar)`` specs do exactly that).
    """
    assert not _has_lattice("tandem")
    text = _card("air", system="tandem")
    assert "Wing cant and sweep" in text, text[:600]
    assert "LIFTING LINE" not in text, text[:600]


def test_the_tip_device_control_lands_the_tandem_on_a_cant_capable_family():
    """ITEM 2, VERIFIED THROUGH THE SHELL rather than a hand-built dict.

    A hand-assembled choices dict said no configuration reached a
    cant-capable tandem. It was wrong, and wrong in the way a hand-built
    dict is: the tip-device control does not write ``winglets="canted"`` —
    that is the SHAPE vocabulary — it writes ``winglets="free"`` plus a
    ``winglet_type``, which is what selects ``tandemvlm``.

    The card used to OFFER that as a button. The button is gone (the user
    asked for it), so what is driven here is the tip-device control itself:
    the ordinary answer a user gives on this same card. The claim under
    test is unchanged and is the one that matters — a tandem reaches
    ``tandem (nonplanar) + winglets``, and the cant card appears there
    without anything else being answered again.
    """
    from gui.v3.app import assemble

    ctx = assemble("air")
    ctx.act("set_choice", "system", "tandem")
    assert not api.cant_is_searched(ctx.S["wing"]["problem"])
    ctx.render("wing", "type")
    # BEFORE: the pair scores no cant in its own solver. The card is drawn
    # regardless now (its switch reaches the nonplanar twin), so the claim
    # pinned here is the one this control is about — which FAMILY it lands
    # on — and not whether a heading is on screen.
    assert not set(api.WING_CANT_KEYS) <= set(api.PROBLEM_SPECS[
        ctx.S["wing"]["problem"]].flags)

    ctx.act("set_winglet", "canted")
    name = ctx.S["wing"]["problem"]
    assert name.startswith("tandem (nonplanar)"), name
    assert set(api.WING_CANT_KEYS) <= set(api.PROBLEM_SPECS[name].flags), name
    ctx.render("wing", "type")
    assert "Wing cant and sweep" in _text(ctx.views[("wing", "type")])


def test_every_nonplanar_tandem_answers_the_cant_and_no_planar_one_does():
    """The census the route rides on, so it cannot rot silently.

    EITHER WAY OF ANSWERING COUNTS, which is the shell's own rule
    (``gui.v3.stages.wing._family_answers_cant``): the question a route has
    to make answerable is "does this pair have a dihedral", and a family
    answers it with the STATED flags or with the two SEARCHED rows. The
    free-cant twins carry the rows and NOT the flags — stating a flag the
    vector already carries is the same question answered twice, which
    ``api.check_flags`` refuses — so an assertion that demanded the flags
    everywhere would fail precisely where the pair got MORE freedom.

    The counts are derived, not pinned: the nonplanar family is a product
    over tip devices, designed section and the pair's own cant now.
    """
    # NOT ``split(" + ")[0]``: a bracketed qualifier is part of the family
    # core, so that idiom silently drops every ``... [free cant]`` name and
    # would have censused 5 of the 7 nonplanar bases while reporting success.
    planar = [n for n in api.PROBLEM_SPECS
              if n.split(" + ")[0] == "tandem"]
    nonplanar = [n for n in api.PROBLEM_SPECS
                 if n.startswith("tandem (nonplanar)")]
    assert len(planar) == 32, len(planar)
    assert len(nonplanar) == 16 * len(api.TANDEM_VLM_VARIANTS), \
        (len(nonplanar), len(api.TANDEM_VLM_VARIANTS))
    assert not any(set(api.WING_CANT_KEYS) <= set(api.PROBLEM_SPECS[n].flags)
                   or api.cant_is_searched(n) for n in planar)
    for n in nonplanar:
        stated = set(api.WING_CANT_KEYS) <= set(api.PROBLEM_SPECS[n].flags)
        searched = api.cant_is_searched(n)
        assert stated != searched, (n, stated, searched)


# ------------------------------------- the round trip on the SECTION twins

def test_every_free_cant_family_reads_its_cant_back_as_FREE():
    """choices -> family -> choices has to close, on every family.

    ``nice_app.wing_tail_choices`` carried the wing's cant into the inverse
    and ``wing_tail_section_choices`` did not, so all 48 free-cant SECTION
    families inverted to ``wing_cant="fixed"``: the shell greys out the
    option the user just took, a saved preset reopens as the other family,
    and 192 of the 2142 chord twins mis-derived — which is how it surfaced.

    Asserted over the whole registry rather than a sample, because the
    defect was exactly one branch of one inverse being missed.
    """
    from gui import nice_app as v1

    free = [n for n in api.PROBLEM_SPECS if api.cant_is_searched(n)]
    assert len(free) >= 48, len(free)
    # ...and it reads back the STATE, not merely "something is searched":
    # the axis has four values now, so an inverse that answered "free" for a
    # dihedral-only family would round-trip to a family with an extra row.
    wrong = [n for n in free
             if (v1.choices_from_problem(n) or {}).get("wing_cant")
             != api.searched_cant(n)]
    assert not wrong, wrong[:8]


def test_a_fixed_cant_family_reads_back_as_FIXED():
    """The control — an inverse that answered "free" for everybody would
    pass the test above and be just as broken."""
    from gui import nice_app as v1

    fixed = [n for n in api.PROBLEM_SPECS
             if not api.cant_is_searched(n)
             and (v1.choices_from_problem(n) or {}).get("wing_cant")
             is not None]
    assert len(fixed) >= 48, len(fixed)
    wrong = [n for n in fixed
             if v1.choices_from_problem(n)["wing_cant"] != "fixed"]
    assert not wrong, wrong[:8]


def test_the_section_twins_are_the_ones_that_were_broken():
    """Named, so a future reader knows which branch this is about."""
    from gui import nice_app as v1

    section = [n for n in api.PROBLEM_SPECS
               if api.cant_is_searched(n) and "CST section (XFOIL)" in n]
    # 48 registered VARIANTS per cant state, times the chord-law and
    # modifier twins that carry them — the count is asserted as a floor, not
    # a total, so adding a modifier does not fail this
    assert len(section) >= 48, len(section)
    for name in section:
        assert v1.choices_from_problem(name)["wing_cant"] == \
            api.searched_cant(name), name
