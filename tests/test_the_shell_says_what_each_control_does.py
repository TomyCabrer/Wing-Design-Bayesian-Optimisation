"""V4, finalised: what every control SAYS, and the defects saying it found.

Three claims, and they pull in the same direction.

**A control explains itself where it is.** The shell had two channels and
neither did the job: ``tip=`` is a Quasar tooltip on a field's label, so
nothing on screen says it is there, and :func:`gui.v3.widgets.hint` is a
line of prose under the control, which is right for eight words and wrong
for four sentences — and stage 5 and stage 6 between them carried
thirty-one hints longer than twelve words, one of them a hundred and
sixty-three. So there is now a VISIBLE ``?``
(:func:`gui.v3.widgets.help_dot`) holding the paragraph, and the line
beside the control is short. The tests below hold both halves: nothing
long left on screen, and nothing that got shorter by being deleted.

**A question is asked once.** A tab that reprints another tab's numbers, a
box that restates the paragraph above it, a second button that re-runs the
first one's work, and six session keys nothing reads — each of those is a
reader's time spent to learn nothing.

**A stage that cannot answer is not offered.** A foiling craft has no
control surface to cut and no free-flight aeroplane to fly, exactly as a
car rear wing has not, and the reason is a measurement.

Every test here is the outcome, not the code: a rendered view's own text, a
deck's own numbers, a closed form the shell has to agree with.
"""

from __future__ import annotations

import base64
import re
from pathlib import Path

import numpy as np
import pytest

from gui.v3 import widgets
from gui.v4 import app as v4app, session

REPO = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------- fixtures

def _armed(medium: str = "air", problem: str = "tail"):
    """A V4 shell with a real design behind it and stage 5's deck built."""
    from aerobo import api

    ctx = v4app.assemble(medium)
    built = api.PROBLEM_SPECS[problem].build({}, {}, None)
    cfg = api.RunConfig(problem_name=problem, budget=4, seed=0)
    ctx.S["run"]["record"] = {"pretend": "a completed run"}
    ctx.S["run"]["report"] = api.design_report(cfg, built.bounds.mean(axis=1))
    return ctx


#: ONE shell for the sweeps below. ``_armed`` builds a V4 window and scores
#: a design; over 60 parametrised cases that is minutes of the same work.
#: Views are cleared and rebuilt on every ``render``, so re-using the shell
#: is re-using the assembly, not the picture.
_SWEEP: list = []


def _swept():
    if not _SWEEP:
        _SWEEP.append(_armed())
    return _SWEEP[0]


def _labels(el):
    """Every label's text under ``el``, with the ones inside a ``?`` popup
    told apart from the ones on screen."""
    on_screen, in_popup, _hover = _channels(el)
    return on_screen, in_popup


def _channels(el):
    """The three channels a word can be in: on screen, behind a ``?``, and
    HOVER-ONLY — a Quasar tooltip, which nothing on screen announces. The
    third one is why this file exists: it is where the shell used to keep
    its explanations."""
    on_screen, in_popup, hover = [], [], []
    def walk(node, inside, tipped):
        for child in node.default_slot.children:
            cls = list(getattr(child, "_classes", []) or [])
            popped = inside or "help-pop" in cls
            hovered = tipped or type(child).__name__ == "Tooltip"
            text = getattr(child, "text", None)
            if isinstance(text, str) and text.strip():
                (in_popup if popped else
                 hover if hovered else on_screen).append(text.strip())
            walk(child, popped, hovered)
    walk(el, False, False)
    return on_screen, in_popup, hover


def _dots(el) -> int:
    n = 0
    def walk(node):
        nonlocal n
        for child in node.default_slot.children:
            if "help-dot" in (getattr(child, "_classes", []) or []):
                n += 1
            walk(child)
    walk(el)
    return n


# ---------------------------------------------- the widget the ask needed

def test_a_help_dot_is_a_visible_mark_that_carries_its_paragraph():
    """``tip=`` is hover-only and invisible; this is neither."""
    from nicegui import ui

    with ui.element("div") as root:
        widgets.help_dot("First sentence.\n\nSecond one.", title="Why?")
    on_screen, in_popup = _labels(root)
    assert on_screen == ["?"], "the mark itself has to be on screen"
    assert _dots(root) == 1
    # ...and the paragraph is BEHIND it, split where the blank lines were
    assert in_popup == ["Why?", "First sentence.", "Second one."]


def test_a_field_row_carries_the_short_text_and_the_long_one_separately():
    from nicegui import ui

    with ui.element("div") as root:
        widgets.number_field("fin height", 1.2, lambda e: None, unit="m",
                             note="how tall the fin is",
                             help="A tall fin buys Cn_beta and costs drag.")
    on_screen, in_popup = _labels(root)
    assert "how tall the fin is" in on_screen
    assert "A tall fin buys Cn_beta and costs drag." in in_popup
    assert "A tall fin buys Cn_beta and costs drag." not in on_screen
    assert _dots(root) == 1


def test_hint_help_is_one_short_line_with_the_rest_behind_the_mark():
    from nicegui import ui

    with ui.element("div") as root:
        widgets.hint_help("Signs, not a certificate.",
                          "A design that passes every one of them can still "
                          "be unpleasant to fly.", title="What this means")
    on_screen, in_popup = _labels(root)
    assert "Signs, not a certificate." in on_screen
    assert any("unpleasant" in t for t in in_popup)
    assert not any("unpleasant" in t for t in on_screen)


# --------------------------------- ...and it is USED, on every V4 control

#: the longest a line of on-screen prose may be. Not a style rule: the ask
#: was "keep only small text next to what each thing does", and a paragraph
#: on screen is what the ``?`` exists to replace. Generous, because a
#: measured sentence ("6.13 deg of dihedral takes the margin to +0.000007")
#: is a number and not prose.
MAX_WORDS_ON_SCREEN = 30


# --------------------------------- the rule that made it reach every stage

def test_a_long_hint_keeps_its_first_sentence_and_hands_over_the_rest():
    """ONE PLACE, not two hundred. The shell writes its explanations as
    paragraphs — which is right, they are what a reader needs when they need
    anything — so :func:`gui.v3.widgets.hint` decides how much of each is on
    screen. Nothing is deleted: the words move."""
    long = ("This mission caps W/S at 75 N/m², and that cap reaches the "
            "solver: a sized search refuses every design above it. Switch to "
            "the other answer if you own the number.")
    lead, rest = widgets.split_hint(long)
    assert len(lead.split()) <= widgets.HINT_WORDS_ON_SCREEN
    assert rest and rest not in lead
    assert lead + " " + rest == long          # every word, in order
    assert widgets.split_hint("eight words is not a paragraph at all") == \
        ("eight words is not a paragraph at all", "")


@pytest.mark.parametrize("text", [
    "clearly adequate for the job — the pilot flies the mission and does "
    "not have to fight the aeroplane to do it  [measured -3.1 to -2.7 %]",
    "b_t²/S_t — how much span the tail's area is spread over (b_t = "
    "sqrt(AR_t S_t)). The published surface is fixed at 4, and the "
    "conventional H-tail range is 3 to 5.",
    "the straight taper the law multiplies; it is a design variable, so "
    "the band is drawn on the middle of the box the run searches, and a "
    "narrower taper row draws a narrower band.",
])
def test_the_split_never_swallows_a_character(text):
    """THE ONE THING IT MAY NOT DO. The clause split used to consume its own
    em-dash — the two halves rejoined to a sentence one character short of
    the one that was written, which is a deletion however small."""
    lead, rest = widgets.split_hint(text)
    assert rest, "these are all long enough to split"
    assert (lead + " " + rest).split() == text.split()


def test_a_sentence_end_has_to_open_a_SENTENCE():
    """A full stop is not a boundary unless what follows starts one. The
    guard is what keeps the split off an abbreviation — without it this line
    breaks after "e.g." and puts "…charged at its own Reynolds number, e.g."
    on screen as if that were the point."""
    text = ("Every station is charged at its own Reynolds number, e.g. a "
            "60 mm chord at 12 m/s, which the section tables do not reach "
            "without extrapolation and so has to be flown instead.")
    lead, rest = widgets.split_hint(text)
    assert not lead.rstrip().endswith("e.g.")
    assert (lead + " " + rest).split() == text.split() or rest == text

    # ...and the one it was written for: a decimal is not a full stop, and
    # the sentence that follows a real one still is a boundary
    lead, _rest = widgets.split_hint(
        "Stable: the CG is 0.488 m in front of the neutral point, SM +0.478 "
        "mac against the SM_min 0.08 the run requires. Note the criterion is "
        "the NEUTRAL POINT.")
    assert lead.endswith("requires.")


def test_a_sentence_with_no_boundary_keeps_all_of_itself_in_the_popup():
    """The fallback may not cut a sentence in half and lose the end."""
    run_on = " ".join(f"word{i}" for i in range(60))
    lead, rest = widgets.split_hint(run_on)
    assert rest == run_on          # the whole thing is still readable
    assert lead.endswith("…")


def test_a_long_hint_does_not_call_itself_forever():
    """The fallback lead is limit+1 words (it ends in an ellipsis), so a
    hint that split on it and split again would recurse until the render
    died — which is exactly what it did."""
    from nicegui import ui

    with ui.column() as root:
        widgets.hint(" ".join(f"w{i}" for i in range(80)))
    on_screen, popup, _hover = _channels(root)
    assert _dots(root) == 1
    words = [w for w in " ".join(on_screen).split() if w != "?"]
    assert len(words) <= widgets.HINT_WORDS_ON_SCREEN + 1   # + the ellipsis
    assert len(" ".join(popup).split()) >= 80


def test_a_long_tooltip_becomes_a_mark_and_a_short_one_stays_a_tooltip():
    """A tip is a unit or a state; past that it is an argument."""
    from nicegui import ui

    with ui.column() as short:
        widgets.number_field("min t/c", 0.1, lambda _e: None,
                             tip="structural depth")
    with ui.column() as long:
        widgets.number_field(
            "min t/c", 0.1, lambda _e: None,
            tip="sections thinner than this are refused outright, because "
                "the spar has to fit inside the section it is drawn in")
    assert _dots(short) == 0 and _dots(long) == 1
    assert _channels(short)[2] and not _channels(long)[2]


def test_an_author_who_wrote_both_keeps_both():
    """``help`` beside a short ``tip`` is two different sentences by
    choice, and the promotion must not eat one. Beside a LONG tip they both
    go behind the mark — leaving one on hover would leave a paragraph in
    the channel this rule exists to empty."""
    from nicegui import ui

    with ui.column() as short:
        widgets.number_field("span", 10.0, lambda _e: None,
                             tip="metres", help="The one hard constraint.")
    _on, popup, hover = _channels(short)
    assert "metres" in hover and any("hard constraint" in t for t in popup)

    with ui.column() as both:
        widgets.number_field(
            "span", 10.0, lambda _e: None,
            tip="the spar, the hangar, the trailer and the roll rate all "
                "ask for a number here",
            help="The one hard constraint.")
    _on2, popup2, hover2 = _channels(both)
    blob = " ".join(popup2)
    assert "hard constraint" in blob and "the trailer" in blob
    assert not [t for t in hover2
                if len(t.split()) > widgets.TIP_WORDS_ON_HOVER]


def test_a_read_out_and_a_group_box_can_both_carry_one():
    """The two chrome pieces that had no way to explain themselves: a
    figure's title bar, and a big number with a three-word caption."""
    from nicegui import ui

    with ui.column() as root:
        widgets.readout("Re", "1.0e+06", help="What the polars were read at.")
        with widgets.group_box("Planform", help="Seen from above."):
            ui.label("(a figure)")
    assert _dots(root) == 2
    blob = " ".join(_channels(root)[1])
    assert "polars were read at" in blob and "Seen from above" in blob


def test_stage_4_says_what_each_picture_is():
    """Five figures, one of them explained, before this: a reader who
    wanted to know which way up a section is drawn had nowhere to look."""
    from gui.v3.stages import results as results_stage

    assert set(results_stage.FIG_HELP) == {
        "planform", "front_view", "sections_as_flown", "wing3d", "spanwise"}
    assert "upside down" in results_stage.FIG_HELP["sections_as_flown"]
    assert "true-scale" in results_stage.FIG_HELP["wing3d"]
    ctx = _swept()
    ctx.render("results", "geometry")
    _on, popup, _hover = _channels(ctx.views[("results", "geometry")])
    assert any("upside down" in t for t in popup)


def test_the_box_row_paragraph_is_not_also_on_hover():
    """It is behind the ? beside the row's name; a tooltip carrying the
    same words was the longest text on the view (158 words on z_t_m)."""
    ctx = _swept()
    ctx.render("wing", "box")
    _on, popup, hover = _channels(ctx.views[("wing", "box")])
    assert any("Vertical distance of the second surface" in t for t in popup)
    assert not any("Vertical distance of the second surface" in t
                   for t in hover)


#: every view of the pipeline, V3's four stages and V4's two. The ask was
#: "the question mark should be in all sections, not only the mission", and
#: the way to hold that is to name every view once and let the same two
#: tests run over all of them.
V3_VIEWS = [("mission", "operating"), ("mission", "point"),
            ("mission", "search"),
            ("airfoil", "screen"), ("airfoil", "ranking"),
            ("airfoil", "section"), ("airfoil", "optimise"),
            ("wing", "type"), ("wing", "box"), ("wing", "solver"),
            ("wing", "run"),
            ("results", "summary"), ("results", "geometry"),
            ("results", "loading"), ("results", "log")]

#: ...except the two that are an EMPTY STATE without work behind them: the
#: ranking has nothing until a screen is run, and the run view nothing until
#: a search is. "Nothing screened yet" needs no ``?``, and demanding one
#: would be demanding an explanation of an absence.
V3_VIEWS_WITH_CONTENT = [v for v in V3_VIEWS
                         if v not in (("airfoil", "ranking"), ("wing", "run"))]

V4_VIEWS = [("controls", "surfaces"), ("controls", "propulsion"),
            ("controls", "vertical"), ("controls", "derivatives"),
            ("flight", "setup"), ("flight", "modes")]


@pytest.mark.parametrize("stage,view", V3_VIEWS + V4_VIEWS)
def test_nothing_is_explained_on_hover_alone(stage, view):
    """THE CHANNEL THE ASK WAS ABOUT. A Quasar tooltip is invisible until
    the pointer is already on the control, so an explanation kept there is
    read only by someone who knew it was there. Short ones are fine — a unit,
    a state — but a paragraph belongs behind a mark the eye can find."""
    ctx = _swept()
    ctx.render(stage, view)
    _on, _popup, hover = _channels(ctx.views[(stage, view)])
    hidden = [t for t in hover
              if len(t.split()) > widgets.TIP_WORDS_ON_HOVER]
    assert not hidden, ("hover-only paragraph: "
                        + " || ".join(t[:90] for t in hidden))


@pytest.mark.parametrize("stage,view", V3_VIEWS + V4_VIEWS)
def test_no_wall_of_text_survives_on_any_view(stage, view):
    """The V3 half of the same rule, over every stage of the pipeline."""
    ctx = _swept()
    ctx.render(stage, view)
    on_screen, _popup, _hover = _channels(ctx.views[(stage, view)])
    long = [t for t in on_screen if len(t.split()) > MAX_WORDS_ON_SCREEN]
    assert not long, ("still a paragraph on screen: "
                      + " || ".join(t[:90] for t in long))


@pytest.mark.parametrize("stage,view", V3_VIEWS_WITH_CONTENT)
def test_every_v3_view_offers_a_mark_of_its_own(stage, view):
    ctx = _swept()
    ctx.render(stage, view)
    box = ctx.views[(stage, view)]
    assert _dots(box) >= 1, f"no ? anywhere on {stage}/{view}"


@pytest.mark.parametrize("stage,view", [
    ("controls", "surfaces"), ("controls", "propulsion"),
    ("controls", "vertical"), ("controls", "derivatives"),
    ("flight", "setup"), ("flight", "modes"),
])
def test_no_wall_of_text_survives_on_any_v4_view(stage, view):
    ctx = _armed()
    ctx.render(stage, view)
    on_screen, _popup = _labels(ctx.views[(stage, view)])
    long = [t for t in on_screen if len(t.split()) > MAX_WORDS_ON_SCREEN]
    assert not long, ("still a paragraph on screen: "
                      + " || ".join(t[:90] for t in long))


@pytest.mark.parametrize("stage,view", [
    ("controls", "surfaces"), ("controls", "propulsion"),
    ("controls", "vertical"), ("controls", "derivatives"),
    ("flight", "setup"),
])
def test_every_v4_view_offers_the_mark_it_hid_its_prose_behind(stage, view):
    """The other half: prose that got SHORTER by being deleted would pass
    the test above and lose the explanation."""
    ctx = _armed()
    ctx.render(stage, view)
    box = ctx.views[(stage, view)]
    assert _dots(box) >= 1, "no ? anywhere on this view"
    _on, popup = _labels(box)
    assert sum(len(t.split()) for t in popup) >= 40, \
        "the marks are there and carry nothing"


def test_the_words_that_moved_are_still_readable_not_deleted():
    """Three specific explanations the shell is not allowed to lose."""
    ctx = _armed()
    ctx.render("controls", "vertical")
    _on, popup = _labels(ctx.views[("controls", "vertical")])
    blob = " ".join(popup)
    assert "80.6" in blob and "42.2" in blob, "the Reynolds measurement"
    assert "denominator" in blob.lower(), "why a bigger fin is not better"
    assert "OUT OF REACH" in blob, "what an unreachable target does"


# ------------------------------------------ a foiler has neither stage

@pytest.mark.parametrize("medium", ["water", "track"])
def test_a_craft_that_does_not_fly_free_is_offered_neither_stage(medium):
    S = session.make_session(medium)
    assert session.free_flight(S) is False
    for stage in ("controls", "flight"):
        assert session.stage_visible(S, stage) is False
        state, why = session.stage_states(S)[stage]
        assert state == "locked" and why


def test_the_air_session_is_the_positive_control():
    S = session.make_session("air")
    assert session.free_flight(S) is True
    assert all(session.stage_visible(S, s) for s in ("controls", "flight"))


def test_the_foiler_lock_quotes_the_number_that_justifies_it():
    S = session.make_session("water")
    why = session.stage_states(S)["controls"][1]
    assert "mast" in why.lower() and "-0.196" in why


def test_the_v4_banner_does_not_promise_a_stage_the_tree_will_not_show():
    """The startup log line branched on the CAR alone, so a foiling session
    was told stage 5 cuts control surfaces and stage 6 flies it."""
    ctx = v4app.assemble("water")
    said = " ".join(t for row in ctx.output.lines
                    for t in _labels(row)[0])
    assert "stage 5 cuts" not in said
    assert "ends at stage 4" in said


# ------------------------------------------------- one question, one place

def test_stage_5_has_four_tabs_and_the_stability_check_is_not_one_of_them():
    """It printed four of the Derivatives table's own rows again, off the
    same deck object, one tab-click away."""
    keys = [k for k, _l, _i in session.VIEWS["controls"]]
    assert keys == ["surfaces", "propulsion", "vertical", "derivatives"]


def test_a_row_of_the_WRONG_sign_says_so_where_the_number_is():
    """The colour and the verdict are what the deleted Stability check tab
    uniquely had, so they have to survive its deletion — a table that prints
    Cn_beta +0.00 to five places and nothing else is the table that was
    there before either view existed."""
    ctx = _armed()
    ctx.S["controls"]["vertical"]["x_le_m"] = -1.0     # fin AHEAD of the CG
    ctx.render("controls", "derivatives")
    assert float(ctx.S["controls"]["deck"].Cn_beta) < 0.0
    on_screen, _p = _labels(ctx.views[("controls", "derivatives")])
    blob = " ".join(on_screen)
    assert "Wrong sign" in blob and "Cn_b" in blob, blob[-400:]
    assert "Every sign test passes" not in blob


def test_the_derivatives_table_now_carries_the_sign_test_and_the_margin():
    ctx = _armed()
    ctx.render("controls", "derivatives")
    on_screen, _p = _labels(ctx.views[("controls", "derivatives")])
    blob = " ".join(on_screen)
    assert "static margin" in blob, "the one row the deleted tab owned"
    assert "Cn_b" in blob and "Cl_p" in blob
    # ...and the verdict is a line, not a tab
    assert "sign test" in blob.lower() or "wrong sign" in blob.lower()


def test_no_flight_default_is_a_key_nothing_reads():
    """Six of them were: an inertia mode, two mass fractions and Ixx/Iyy/Izz,
    written into every session and read by nothing."""
    #: where the dict itself lives — a key that appears ONLY here is state
    #: declared and never read
    home = (REPO / "gui/v4/session.py").resolve()
    src = "\n".join(p.read_text() for p in (REPO / "gui").rglob("*.py")
                    if p.resolve() != home)
    for key in session.FLIGHT_DEFAULTS:
        assert re.search(rf'"{re.escape(key)}"', src), \
            f'FLIGHT_DEFAULTS["{key}"] is declared and never read'


def test_the_same_holds_for_stage_5s_answers():
    home = (REPO / "gui/v4/session.py").resolve()
    src = "\n".join(p.read_text() for p in (REPO / "gui").rglob("*.py")
                    if p.resolve() != home)
    for key in session.CONTROLS_DEFAULTS:
        assert re.search(rf'"{re.escape(key)}"', src), \
            f'CONTROLS_DEFAULTS["{key}"] is declared and never read'


def test_the_two_stages_do_not_each_own_a_copy_of_the_matrix():
    from gui.v4 import world as wld
    from gui.v4.stages import flight as flt

    assert np.allclose(np.asarray(flt.M_EARTH_TO_SCENE),
                       np.asarray(wld.M_EARTH_TO_SCENE))
    # ...and the ground posts are placed by the tested helper, not by a
    # second copy of its loop
    src = (REPO / "gui/v4/stages/flight.py").read_text()
    assert "wld.lattice_offsets(" in src


# ------------------------------------------------------- the tree is honest

def test_the_two_v4_stages_do_not_wear_the_wing_searchs_objective():
    ctx = _armed()
    shell = v4app._load_shell()
    ctx.S["run"]["record"]["best_score"] = -1.234
    results = shell._badge(ctx, "results")
    assert results == "f -1.234"
    for stage in ("controls", "flight"):
        assert shell._badge(ctx, stage) != results


def test_stage_5s_badge_is_the_number_building_it_changed():
    ctx = _armed()
    ctx.render("controls", "derivatives")
    shell = v4app._load_shell()
    badge = shell._badge(ctx, "controls")
    assert badge.startswith("Cn_b")
    assert f"{float(ctx.S['controls']['deck'].Cn_beta):+.4f}" in badge


def test_a_deck_built_for_the_PREVIOUS_design_is_not_called_done():
    """``C["deck"]`` outlives the run that produced it, so a re-run left the
    tree's two most confident words about an aeroplane that was gone."""
    ctx = _armed()
    ctx.render("controls", "derivatives")
    assert session.stage_states(ctx.S)["controls"][0] == "done"
    assert session.stage_states(ctx.S)["flight"][0] != "locked"

    ctx.S["run"]["report"] = dict(ctx.S["run"]["report"])   # a NEW design
    state, why = session.stage_states(ctx.S)["controls"]
    assert state == "ready" and "changed" in why
    assert session.stage_states(ctx.S)["flight"][0] == "locked"


def test_the_toolbar_can_run_and_stop_on_every_stage_the_shell_mounts():
    """``_run_stage`` subscripted a dict literal that named V3's stages, so
    the Run button raised KeyError on stages 5 and 6."""
    shell = v4app._load_shell()
    for stage in session.STAGES:
        assert stage in shell.RUN_ACTIONS, stage


def test_the_stop_button_can_see_the_flight_loop():
    S = session.make_session("air")
    assert not any(r for r, _a, _kw in session.extra_running(S))
    S["flight"]["running"] = True
    running = [(a, kw) for r, a, kw in session.extra_running(S) if r]
    assert running == [("flight_run", (False,))]


# ------------------------------------------------- the numbers themselves

def test_the_load_factor_is_the_specific_force_and_not_a_derivative_row():
    """``derivative()[9]`` is ``F/m + g_b - w x v``. The HUD added the
    gravity term back and dropped the rate one, so the number was right only
    while the aeroplane was not manoeuvring."""
    from aerobo import sixdof as sd

    ctx = _armed()
    ctx.render("controls", "derivatives")
    assert ctx.act("flight_arm") is True
    ac, st = ctx.S["flight"]["ac"], ctx.S["flight"]["state"]

    # a real body rate, which is where the two expressions part company
    st.rates[1] = np.deg2rad(30.0)
    Fb, _M = ac.forces_moments(st)
    truth = float(-Fb[2] / (ac.inertia.mass_kg * sd.G))
    d = sd.derivative(ac, st)
    roll, pitch, _yaw = st.euler
    old = float(-d[9] / sd.G + np.cos(pitch) * np.cos(roll))
    assert abs(truth - old) > 0.05, "no rate term to tell apart"

    ctx.act("flight_readout")
    shown = ctx.S["flight"]["hud"] and ctx.act("flight_payload")["hud"]["g"]
    assert shown == pytest.approx(truth, abs=5e-3)


def test_the_cg_lever_can_reach_the_neutral_point():
    """The stage's headline lesson is dragging the CG aft until it
    diverges. A band of +-0.35 mac about the DESIGN CG cannot get there on
    any design whose static margin is bigger than that."""
    ctx = _armed()
    ctx.render("controls", "derivatives")
    assert ctx.act("flight_arm") is True
    ctx.render("flight", "fly")
    lo, hi = ctx.act("flight_probe")["cg_band"]
    x_np = ctx.act("flight_probe")["x_np"]
    assert x_np is not None
    assert lo < x_np < hi, (lo, x_np, hi)


def test_moving_the_altitude_lever_does_not_throw_the_trace_away():
    """``trim_level`` uses the altitude for the state's POSITION and nothing
    else — the density is the mission's — so re-arming for it discarded a
    flight to arrive at the identical equilibrium."""
    ctx = _armed()
    ctx.render("controls", "derivatives")
    assert ctx.act("flight_arm") is True
    ctx.render("flight", "fly")
    F = ctx.S["flight"]
    F["history"] = list(F["history"]) * 12
    F["history_t"] = [0.1 * i for i in range(len(F["history"]))]
    before = len(F["history"])
    alpha = float(F["trim"]["alpha_deg"])

    ctx.act("flight_set_condition", "altitude_m", 250.0)
    assert len(F["history"]) == before, "the trace was thrown away"
    assert float(F["trim"]["alpha_deg"]) == pytest.approx(alpha)
    # ...and the aeroplane is where it was asked to be
    assert float(-F["state"].pos[2]) == pytest.approx(250.0, abs=1e-6)


def test_the_speed_lever_still_DOES_re_trim():
    """The positive control for the one above."""
    ctx = _armed()
    ctx.render("controls", "derivatives")
    assert ctx.act("flight_arm") is True
    F = ctx.S["flight"]
    alpha = float(F["trim"]["alpha_deg"])
    ctx.act("flight_set_condition", "V_trim", float(F["trim"]["V"]) * 1.4)
    assert float(F["trim"]["alpha_deg"]) != pytest.approx(alpha)
    assert len(F["history"]) == 1, "a re-trim restarts the trace"


def test_the_traces_redraw_while_the_aeroplane_is_still_flying():
    """Leaving the Fly tab was never meant to pause the simulation, so the
    six plots froze at the instant the tab was opened while the history went
    on growing — and the only way to see the rest of a manoeuvre was to
    click away and back, which rebuilds the Fly view and ends the flight."""
    ctx = _armed()
    ctx.render("controls", "derivatives")
    assert ctx.act("flight_arm") is True
    ctx.render("flight", "fly")
    ctx.S["flight"]["running"] = True
    for _ in range(4):
        ctx.act("flight_advance", 0.05)
    ctx.S["ui"]["tab"]["flight"] = "traces"
    ctx.render("flight", "traces")

    panes = ctx.act("flight_probe")["traces"]
    assert panes and len(panes) == 6, panes
    first = {name: el for name, el in panes.items()}

    for _ in range(64):                       # past TRACE_EVERY, twice
        ctx.act("flight_advance", 0.05)
        ctx.act("flight_frame_traces")
    after = ctx.act("flight_probe")["traces"]
    # ...IN PLACE. A rebuilt pane is a new DOM node, and a view that changes
    # height twice a second throws the page back to the top.
    assert all(after[k] is first[k] for k in first)
    # ...and it is showing the WHOLE history, not the four samples that
    # existed when the tab was built. Plotly ships the axis as base64
    # float64, so the point count is the payload's own length.
    n = len(ctx.S["flight"]["history"])
    x = after["Speed [m/s]"]._props["options"]["data"][0]["x"]
    drawn = len(base64.b64decode(x["bdata"])) // 8
    assert drawn == n > 4, (drawn, n)


def test_the_aircraft_mesh_is_addressed_by_its_CONTENT():
    """The route was ``/_flight/<whole seconds>.stl``, so two lofts built in
    one second claimed one path and the browser was served the aeroplane
    that is no longer flying."""
    src = (REPO / "gui/v4/stages/flight.py").read_text()
    assert "int(time.time())" not in src
    assert "hashlib.sha1(blob)" in src


# ------------------------------------------- stage 5's own arithmetic

def test_a_directionally_unstable_design_is_not_told_what_sideslip_it_holds():
    """``Cn_dr d_max / |Cn_beta|`` is a number for an equilibrium the
    aeroplane runs AWAY from, and the panel then offered a smaller rudder."""
    from aerobo.dynamics import weathercocks

    ctx = _armed()
    ctx.S["controls"]["vertical"]["x_le_m"] = -1.0    # fin AHEAD of the CG
    ctx.render("controls", "vertical")
    D = ctx.S["controls"]["deck"]
    assert not weathercocks(float(D.Cn_beta)), "not the case under test"

    on_screen, _p = _labels(ctx.views[("controls", "vertical")])
    assert not any("holds sideslip" in t for t in on_screen)
    assert not any("recommended hinge chord" in t for t in on_screen)


def test_a_sizing_target_out_of_reach_SAYS_so(monkeypatch):
    """``_solve_height`` returns None outside the band and the button then
    did nothing at all — no edit, no message — while the panel's own ``?``
    promised it would be reported."""
    ctx = _armed()
    ctx.render("controls", "vertical")
    before = ctx.S["controls"]["vertical"].get("height_m")

    ctx.S["controls"]["size_Cnb"] = 9.0                # unreachable
    ctx.act("controls_size_cnb")
    assert ctx.S["controls"]["vertical"].get("height_m") == before
    note = ctx.S["controls"]["size_note"]
    assert note is not None and note[0] == "Cnb"
    ctx.render("controls", "vertical")
    on_screen, _p = _labels(ctx.views[("controls", "vertical")])
    assert any("reaches Cn_beta" in t for t in on_screen), on_screen


def test_a_fin_volume_target_never_stores_a_negative_height():
    """A fin ahead of the CG makes the arm negative, and the division then
    wrote a NEGATIVE height — which the model reads as a ventral fin, so the
    switch said dorsal while the surface hung below."""
    ctx = _armed()
    ctx.S["controls"]["vertical"]["x_le_m"] = -1.0
    ctx.render("controls", "vertical")
    ctx.S["controls"]["size_Vv"] = 0.03
    ctx.act("controls_size_vv")
    h = ctx.S["controls"]["vertical"].get("height_m")
    assert h is None or float(h) > 0.0
    # ...and the reason names the ARM, not the band it also happens to fall
    # outside of. "13.2 m is outside 1-35 % of the span" sends the reader to
    # change a chord; the fin being ahead of the CG is what has to move.
    key, why = ctx.S["controls"]["size_note"]
    assert key == "Vv" and "AHEAD of the CG" in why, why


def test_a_scan_belongs_to_the_design_it_was_measured_on():
    """The session's own comment says every recommendation is cleared when
    the design changes; only an edit did it, so a new RUN left the verdict
    of the previous aeroplane on screen."""
    ctx = _armed()
    ctx.render("controls", "vertical")
    ctx.act("controls_scan_spiral")
    assert ctx.S["controls"]["spiral_scan"] is not None

    ctx.S["run"]["report"] = dict(ctx.S["run"]["report"])   # a NEW design
    ctx.render("controls", "vertical")
    assert ctx.S["controls"]["spiral_scan"] is None


def test_a_render_alone_does_not_throw_a_scan_away():
    """The other side of it: repainting the view must not discard the answer
    the user just asked for."""
    ctx = _armed()
    ctx.render("controls", "vertical")
    ctx.act("controls_scan_spiral")
    scan = ctx.S["controls"]["spiral_scan"]
    ctx.render("controls", "vertical")
    ctx.render("controls", "vertical")
    assert ctx.S["controls"]["spiral_scan"] is scan


def test_the_deck_is_rebuilt_when_the_MISSION_moves_under_it():
    """The deck is built from the spec AND from the mission's air. Keyed on
    the report alone, a stage-1 speed edit left the flown deck at the old
    point while every comparison lattice on the panel quoted the new one."""
    ctx = _armed()
    ctx.render("controls", "derivatives")
    first = ctx.S["controls"]["deck"]
    V0 = ctx.S["controls"]["deck_point"][0]

    ctx.S["mission"]["V"] = float(ctx.S["mission"]["V"]) * 2.0
    ctx.render("controls", "derivatives")
    assert ctx.S["controls"]["deck"] is not first
    assert ctx.S["controls"]["deck_point"][0] != V0


def test_a_switched_off_surface_does_not_take_answers_nothing_reads():
    ctx = _armed()
    ctx.S["controls"]["aileron"]["on"] = False
    ctx.render("controls", "surfaces")
    box = ctx.views[("controls", "surfaces")]
    disabled = [el for el in box.descendants()
                if getattr(el, "tag", "") == "q-input"
                and el._props.get("disable")]
    assert disabled, "the aileron band is live under a switch that is off"


def test_a_zero_fin_dimension_is_refused_where_it_is_typed():
    """``fields.is_pinned`` counts 0.0 as an answer and
    ``build_flight_model`` tests truthiness, so a typed zero pinned the
    field and was then dropped by the model."""
    ctx = _armed()
    ctx.render("controls", "vertical")
    flown = float(ctx.act("controls_fin")["h"])

    ctx.act("controls_edit_fin", "height_m", 0.0)
    assert ctx.S["controls"]["vertical"]["height_m"] is None
    assert float(ctx.act("controls_fin")["h"]) == pytest.approx(flown)
    assert ctx.S["controls"]["size_note"][0] == "fin"
