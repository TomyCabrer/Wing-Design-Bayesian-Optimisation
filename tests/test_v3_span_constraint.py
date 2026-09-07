"""The SPAN is stage 3's question, and the PLANFORM menu decides how it is
asked.

A wing is specified by its span before anything else — a hangar, a trailer, a
class rule, a spar — so the span is never a number derived from an aspect
ratio somebody typed. The ASPECT RATIO is what it actually is: b²/S, a
consequence, guessed by stage 2 so the section has a chord to be designed for
and reported by stage 3 as what the run implies.

WHO decides the span is one control, and the answer changes what is asked
under it:

* ``fixed`` — the size card asks for the span, in metres, and there is NO
  ``b_m`` row in the design box: a row there is a BAND on a searched
  variable, and a low/high pair on a span nobody searches is a question
  pretending to be one. The area is not asked at all (stage 1 states a wing
  loading, S = W/(W/S));
* ``wing_loading`` — the span joins the design vector and the AREA leaves it
  (it follows the mission's W/S through the weight loop), which is the only
  pairing in which a span BAND is an honest question. The band is then the
  ``b_m`` row, opened around the span the card held, and it cannot be
  released from inside the table — giving the span up is the menu's question.

Three more rules this file holds:

* the aircraft-sizing FAMILY is not offered in V3, and the menu says why: it
  flies its own section, so a pipeline that designed one would not fly it;
* the box the view SHOWS is the box the run SEARCHES. The span box of a sized
  family is a fraction of the nominal span it is given, so a view reading the
  registered spec's static ``default_bounds`` painted a row "default" at
  6–40 m while the solver searched 1.44–9.6 m (api.span_box);
* what the band bounds is the PROJECTED span — the tip device included —
  because the width the aircraft occupies is what a hangar cares about.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aerobo import api          # noqa: E402
from aerobo import sizing       # noqa: E402


# --------------------------------------------------------------- the api
def test_the_span_box_is_read_off_the_built_problem():
    """The drift this function exists to close: the same problem, resized,
    searches a different span band — and the registered spec cannot say so."""
    name = "trim wing + free span (W/S) + free chord law"
    static = api.PROBLEM_SPECS[name].default_bounds["b_m"]
    assert tuple(static) == api.span_box(name)          # nothing sent: agree

    resized = api.span_box(name, {"b_m": 2.4, "S_m2": 0.5})
    assert resized == pytest.approx(
        (sizing.B_FRAC_BOUNDS[0] * 2.4, sizing.B_FRAC_BOUNDS[1] * 2.4))
    assert resized != tuple(static)                     # ...the drift itself

    # a stated band wins over both, and is what the run gets
    assert api.span_box(name, {"span_min_m": 2.0, "span_max_m": 2.4}) == (
        2.0, 2.4)


def test_a_problem_with_no_span_variable_has_no_span_box():
    assert api.span_box("trim wing") is None
    assert api.span_box("airfoil (section)") is None


# ------------------------------------------------------------- the shell
def _shell(medium: str = "air"):
    from gui.v3.app import assemble

    return assemble(medium)


def test_the_fixed_planform_asks_for_a_span_and_has_no_band():
    """One number, in metres — and no ``b_m`` row anywhere, because nothing
    is searching a span to bound."""
    from gui.v3 import config, session

    ctx = _shell()
    S = ctx.S
    assert S["wing"]["choices"]["planform"] == "fixed"
    assert not session.span_is_searched(S)
    assert session.span_box(S) is None                  # nothing searched
    assert "b_m" not in config.effective_bounds(S)      # ...so no band

    # untouched: the family's own planform, from stage 2's estimate
    assert session.chosen_span(S) is None
    b, area = session.flown_size(S)
    assert b == pytest.approx((session.nominal_aspect_ratio(S) * area) ** 0.5)
    assert b == pytest.approx(session.nominal_span(S))


def test_a_chosen_span_is_the_span_the_run_flies():
    """The card writes the SIZE flags, so the wing built is the wing on
    screen — and the area stays the mission's."""
    from gui.v3 import config, session

    ctx = _shell()
    S = ctx.S
    ctx.act("set_span", 12.0)

    assert session.chosen_span(S) == pytest.approx(12.0)
    area = float(S["mission"]["s_ref_m2"])
    assert session.flown_size(S) == pytest.approx((12.0, area))
    flags = config.flags(S)
    assert flags["b_m"] == pytest.approx(12.0)
    assert flags["S_m2"] == pytest.approx(area)
    # the aspect ratio is a CONSEQUENCE of it, never an input
    assert session.flown_aspect_ratio(S) == pytest.approx(144.0 / area)
    assert "aspect_ratio" not in S["wing"]

    # ...and it is still a fixed planform: the vector reshapes it
    assert not session.span_is_searched(S)
    assert "b_m" not in config.effective_bounds(S)

    # clearing gives the size back to stage 2's estimate — which is what
    # keeps an untouched session bit-for-bit the published run
    ctx.act("set_span", None)
    assert session.chosen_span(S) is None
    assert "b_m" not in config.flags(S)


def test_a_span_this_area_cannot_carry_is_refused_where_it_is_typed():
    """The span and the area are ONE statement, and the pair is what the
    solvers are valid over (``api.PLANFORM_AR_LIMITS``, 3–40).

    Typed 25 m on the default 10 m² wing, the shell stored AR 62.5 and the
    ``ValueError`` came out of ``api._planform_size`` INSIDE the field's own
    event handler: no notify, no log line, no card repainted, and
    ``build_cfg`` still passed — so the job was queued and died in the
    worker. The setter answers False instead, which is what the field's
    existing refusal path is for."""
    from gui.v3 import config, session

    ctx = _shell()
    S = ctx.S
    area = float(S["mission"]["s_ref_m2"])
    assert session.set_span_m(S, 25.0) is False
    assert S["wing"]["span_m"] is None                 # nothing stored
    assert session.chosen_span(S) is None
    # ...and the handler behind the field does not raise either
    ctx.act("set_span", 25.0)
    assert session.chosen_span(S) is None
    assert config.build_cfg(S) is not None             # still a runnable run

    # the reason names the band and the range that would satisfy it
    why = session.planform_size_refusal(S, span=25.0)
    assert "62.5" in why and "5.477" in why and "20" in why

    # ...and a span the area CAN carry is untouched by the guard
    lo, hi = api.PLANFORM_AR_LIMITS
    assert session.set_span_m(S, (hi * area) ** 0.5 - 0.01) is True


def test_an_area_the_chosen_span_cannot_carry_is_refused_too():
    """The same pair from the MISSION's side, where there was not even a
    read-out to warn: plain span mode draws no aspect ratio at all, so
    3 m² under a typed 12 m span (AR 48) was stored in silence."""
    from gui.v3 import config, session

    ctx = _shell()
    S = ctx.S
    ctx.act("set_span", 12.0)
    before = float(S["mission"]["s_ref_m2"])

    assert session.set_reference_area(S, 3.0) is False
    assert float(S["mission"]["s_ref_m2"]) == pytest.approx(before)
    assert config.build_cfg(S) is not None
    # an area the span CAN carry still goes in
    assert session.set_reference_area(S, 6.0) is True
    assert float(S["mission"]["s_ref_m2"]) == pytest.approx(6.0)
    # ...and with NO span chosen the area can never refuse itself: the
    # nominal span is √(AR·S) and follows it
    ctx.act("set_span", None)
    assert session.set_reference_area(S, 0.5) is True


def test_the_menu_offers_the_wing_loading_mode_and_says_what_it_hides():
    """A mode reachable only through a switch inside another card is a mode
    nobody chooses; a menu that simply loses an entry is a menu that lies."""
    from gui import nice_app as v1
    from gui.v3.stages import wing as wing_stage

    offered = {k: lab for k, lab in v1.planform_options(
        _shell().S["wing"]["choices"]).items() if k != "aircraft"}
    assert set(offered) == {"fixed", "wing_loading", "wing_loading_free",
                            "free"}

    why = dict(v1.PLANFORM_OPTION_WHY,
               aircraft=wing_stage.PLANFORM_AIRCRAFT_WHY)
    note = v1.missing_options_note(offered, v1.PLANFORM_CHOICE_LABELS, why)
    assert v1.PLANFORM_CHOICE_LABELS["aircraft"] in note
    assert "no tip-device, tail or designed-section solver" in note


def test_choosing_the_wing_loading_mode_puts_the_span_in_the_vector():
    from gui.v3 import config, session

    ctx = _shell()
    S = ctx.S
    before = S["wing"]["problem"]
    ctx.act("set_span", 12.0)
    ctx.act("set_planform", "wing_loading")

    assert session.span_is_searched(S)
    assert "b_m" in api.PROBLEM_SPECS[S["wing"]["problem"]].param_labels
    # ...and the AREA left it: one size question, answered by the mission
    assert "S_m2" not in api.PROBLEM_SPECS[S["wing"]["problem"]].param_labels
    assert config.flags(S)["wing_loading_pa"] == pytest.approx(
        session.wing_loading(S))
    # the band is WRITTEN, not left implicit: an implicit band is a fraction
    # of a nominal span, so it would move under the user whenever that did.
    # What is written is that band CLIPPED — `clip_size_box` removes the
    # corners the aspect-ratio gate would refuse, and the box shown has to be
    # the box searched. Measured here: the fraction of the held 12 m span is
    # (7.2, 48.0) and the box holds (7.2, 20.0).
    # tests/test_tandem_two_spans.py pins the same clip on a pair.
    box = session.clip_size_box(S, {"b_m": list(session.span_band_default(S))})
    assert S["wing"]["bounds"]["b_m"] == list(box["b_m"])
    # ...and it opens around the span the card held, not around a guess the
    # user has already overruled. Checked on the band that is actually
    # searched, which is where this rule can be broken.
    assert S["wing"]["bounds"]["b_m"][0] < 12.0 < S["wing"]["bounds"]["b_m"][1]
    # the UNCLIPPED band is still exactly the fraction of the held span, so
    # the clip is the only thing standing between the two. Asserted
    # separately so that moving B_FRAC_BOUNDS still fails here instead of
    # being quietly absorbed by the clip.
    assert list(session.span_band_default(S)) == [
        pytest.approx(round(sizing.B_FRAC_BOUNDS[0] * 12.0, 3)),
        pytest.approx(round(sizing.B_FRAC_BOUNDS[1] * 12.0, 3))]

    ctx.act("set_planform", "fixed")
    assert S["wing"]["problem"] == before
    assert not session.span_is_searched(S)
    assert "b_m" not in S["wing"]["bounds"]
    # the chosen span is the user's answer to a question they may switch back
    # on — kept, exactly like a typed design-box bound
    assert session.chosen_span(S) == pytest.approx(12.0)


def test_the_span_row_cannot_be_released_from_inside_the_table():
    """Releasing it would search the family's published band (6–40 m on the
    10 m trim wing), which is nobody's opinion about the wing on screen.
    Giving the span up is the planform menu's question."""
    from gui.v3 import config, session

    ctx = _shell()
    S = ctx.S
    ctx.act("set_planform", "wing_loading")
    band = list(S["wing"]["bounds"]["b_m"])

    ctx.act("set_row_on", "b_m", False)
    assert session.span_is_searched(S)
    assert "b_m" not in (S["wing"]["bounds_off"] or [])
    assert S["wing"]["bounds"]["b_m"] == band
    # "mission", not "user": the band the mode wrote is the one the MISSION
    # implies, and it keeps following the mission until somebody types in the
    # row. Reporting it as "user" was the reason a mission stated after the
    # mode was chosen left a stale band nothing was allowed to refresh.
    assert config.effective_bounds(S)["b_m"][1] == "mission"
    ctx.act("set_bound", "b_m", 0, 3.25)
    assert config.effective_bounds(S)["b_m"][1] == "user"


def test_a_typed_span_limit_is_what_the_optimiser_searches():
    from gui.v3 import config, session

    ctx = _shell()
    S = ctx.S
    ctx.act("set_planform", "wing_loading")
    ctx.act("set_bound", "b_m", 0, 2.0)
    ctx.act("set_bound", "b_m", 1, 2.4)

    assert session.span_box(S) == (2.0, 2.4)
    cfg = config.build_cfg(S)
    built = api.PROBLEM_SPECS[cfg.problem_name].build(
        cfg.mission_kwargs, cfg.flags, cfg.bounds_overrides)
    i = list(built.param_labels).index("b_m")
    assert list(built.bounds[i]) == [2.0, 2.4]


def test_the_box_on_screen_is_the_box_the_run_searches():
    """The view reads the same function the run is built through, so a
    resized wing cannot leave the design box quoting the family's own band."""
    from gui.v3 import config, session

    ctx = _shell()
    S = ctx.S
    ctx.act("set_planform", "wing_loading")
    ctx.act("reset_box")                       # no user row: the default one
    shown = config.effective_bounds(S)["b_m"][0]
    cfg = config.build_cfg(S)
    built = api.PROBLEM_SPECS[cfg.problem_name].build(
        cfg.mission_kwargs, cfg.flags, cfg.bounds_overrides)
    i = list(built.param_labels).index("b_m")
    assert list(built.bounds[i]) == pytest.approx([float(shown[0]),
                                                   float(shown[1])])
    assert session.span_box(S) == pytest.approx(tuple(built.bounds[i]))


def test_resetting_the_box_does_not_walk_the_span_band_outwards():
    """The reset band is taken around the NOMINAL span, never around the
    middle of the band being reset — the fractional band is not centred on
    its own midpoint, so that would grow it 2.3x per click."""
    from gui.v3 import session

    ctx = _shell()
    S = ctx.S
    ctx.act("set_planform", "wing_loading")
    first = session.span_box(S)
    for _ in range(3):
        ctx.act("reset_box")
    assert session.span_box(S) == pytest.approx(first)


def test_nobody_types_an_aspect_ratio_any_more():
    """It is b²/S. Stage 2 guesses one to have a chord; stage 3 reports the
    one its span band implies, and the estimate follows the wing (never the
    reverse) through the one button that resolves the disagreement."""
    from gui.v3 import session

    ctx = _shell()
    S = ctx.S
    assert "set_wing_aspect_ratio" not in ctx.actions
    # not merely unused: the wing carries no aspect ratio AT ALL, because
    # nothing on screen asks for one (state may not outlive its control)
    assert "aspect_ratio" not in S["wing"]
    assert not hasattr(session, "set_wing_aspect_ratio")

    ctx.act("set_planform", "wing_loading")
    ctx.act("set_bound", "b_m", 0, 8.0)
    ctx.act("set_bound", "b_m", 1, 12.0)
    assert "aspect_ratio" not in S["wing"]              # still nobody's input

    _, area = session.flown_size(S)
    band = session.flown_ar_band(S)
    assert band == pytest.approx((64.0 / area, 144.0 / area))
    assert session.flown_aspect_ratio(S) == pytest.approx(100.0 / area)


def test_the_untouched_session_still_sends_nothing():
    """A row nobody switched on is not an override, and the published run
    stays bit-for-bit."""
    from gui.v3 import config

    ctx = _shell()
    over = config.bounds_overrides(ctx.S)
    assert over is None or "b_m" not in over


# ---------------------------------------------- the span is the PROJECTED one
def test_a_tip_device_is_paid_for_inside_the_span_band():
    """A span limit limits the width the aircraft OCCUPIES. On the capped
    families the wing panel shrinks to pay for the device's projection, so
    the flown projected span is the band's; on a free-span device it is not,
    and the shell says so rather than quoting a width the run exceeds."""
    b_cap = 12.0
    flown = {}
    for name in ("winglet_capped + free span (W/S)",
                 "winglet + free span (W/S)"):
        built = api.PROBLEM_SPECS[name].build(
            {}, {"span_min_m": 8.0, "span_max_m": b_cap}, None)
        x = built.bounds.mean(axis=1)
        x[list(built.param_labels).index("b_m")] = b_cap
        out = built.evaluate(np.asarray(x))
        # the flown PANEL span, from the planform the run reports
        b_panel = (out["AR"] * out["S_m2"]) ** 0.5
        flown[name] = b_panel + 2.0 * out["winglet"]["projection_m"]

    assert flown["winglet_capped + free span (W/S)"] == pytest.approx(
        b_cap, rel=1e-9)
    assert flown["winglet + free span (W/S)"] > b_cap


def test_the_shell_picks_the_capped_device_so_the_band_is_the_width():
    from gui.v3 import session

    ctx = _shell()
    S = ctx.S
    ctx.act("set_planform", "wing_loading")
    ctx.act("set_winglet", "canted")
    assert S["wing"]["choices"]["winglets"] == "capped"
    assert not session.device_reaches_past_the_span(S)
    assert session.span_is_searched(S), S["wing"]["problem"]
