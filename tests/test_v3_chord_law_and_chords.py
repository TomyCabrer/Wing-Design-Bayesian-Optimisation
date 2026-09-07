"""Stage 3 gained two questions it could not ask.

WHICH LAW. The chord-law panel could say how FAR the law may bend the chord
and never what it bends it INTO. A cubic that fills the tip in also lifts the
root chord; an interior-only law cannot touch either end; a cranked law draws
straight panels. Those are different design decisions, not different amounts
of one — so the law is a control (``api.CHORD_LAW_KEY``), and the design box
shows the rows THAT law searches, which for the elliptic blend is one row
where the cubic has three.

THE TWO END CHORDS. A trapezoid has three numbers and this shell stated two:
the area (stage 1, through the wing loading) and the span. The taper was the
search's, and with it the root and tip chords. That is the wrong way round for
a wing whose CHORDS are the stated thing — a spar depth, a hinge line, a
mould, a rib kit. So the size card can be asked the other way: give the two
chords, and the taper is pinned at their ratio while the span follows from the
area.

The two compose, and that is the point: under the interior-only law the two
chords typed into the size card are EXACTLY the chords that fly, however hard
the optimiser works on the planform between them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aerobo import api                      # noqa: E402
from aerobo import geometry as g            # noqa: E402


def _shell(medium: str = "air"):
    from gui.v3.app import assemble

    return assemble(medium)


def _run(S, budget=8):
    from gui.v3 import config

    cfg = api.RunConfig(**{**config.cfg_dict(S), "optimiser": "sobol",
                           "budget": budget, "seed": 0})
    return cfg, api.run(cfg)


# ------------------------------------------------------------- 1. which law

def test_the_shell_opens_on_the_published_law():
    from gui.v3 import config

    S = _shell().S
    assert S["wing"]["flags"].get(api.CHORD_LAW_KEY) is None
    assert config.flags(S).get(api.CHORD_LAW_KEY) is None


def test_choosing_a_law_changes_the_planform_the_run_draws():
    """Same design vector, different shape — measured on the flown chord.

    Run with the chord TREND lifted, because this is a question about the
    LAWS and the shell's opening trend is not part of it: `root_largest`
    leaves the `ends` and `kinked` laws 6 and 4 of 128 draws (measured), so
    with the default on, two of the four searches return no design at all and
    this comparison would be about the trend instead. What that costs is
    asserted directly in `test_what_the_opening_trend_costs_each_law`.
    """
    from gui.v3 import config

    drawn = {}
    for law in g.CHORD_LAWS:
        ctx = _shell()
        S = ctx.S
        ctx.act("set_chord_trend", "free")
        ctx.act("set_chord_law", law)
        cfg, res = _run(S)
        assert (config.flags(S).get(api.CHORD_LAW_KEY) or "poly") == law
        rep = api.design_report(cfg, res.best_x)
        drawn[law] = np.asarray(rep["geometry"]["chord"], dtype=float)
        assert (rep.get("breakdown") or {}).get("chord_law", "poly") == law
    # four laws, four different planforms
    for a in g.CHORD_LAWS:
        for b in g.CHORD_LAWS:
            if a < b:
                assert not np.allclose(drawn[a], drawn[b])


def test_the_box_shows_the_rows_that_law_searches():
    """The elliptic blend is ONE number. A design box still offering three
    chord rows would be showing a box the run does not search — the drift
    that the flag-moved rows exist to close."""
    from gui.v3 import config

    ctx = _shell()
    S = ctx.S
    assert [k for k in config.effective_bounds(S) if k.startswith("chord_k")] \
        == ["chord_k1", "chord_k2", "chord_k3"]

    ctx.act("set_chord_law", "elliptic")
    rows = [k for k in config.effective_bounds(S) if k.startswith("chord_k")]
    assert rows == ["chord_k1"]
    # ...and it is the law's own kind of box: a fraction, never negative
    assert config.effective_bounds(S)["chord_k1"][0][0] == 0.0

    cfg, res = _run(S)
    assert [lbl for lbl in res.param_labels if lbl.startswith("chord_k")] \
        == ["chord_k1"]


def test_changing_the_law_drops_the_old_law_s_own_numbers():
    """A band typed for the cubic is not an opinion about the ellipse, and a
    row the new law does not carry would refuse to build at all."""
    from gui.v3 import config

    ctx = _shell()
    S = ctx.S
    ctx.act("set_bound", "chord_k2", 0, -0.2)
    ctx.act("set_row_fixed", "chord_k3", True)
    assert "chord_k2" in S["wing"]["bounds"]
    assert "chord_k3" in config.fixed_rows(S)

    ctx.act("set_chord_law", "elliptic")
    assert not [k for k in S["wing"]["bounds"] if k.startswith("chord_k")]
    assert not [k for k in config.fixed_rows(S) if k.startswith("chord_k")]
    _run(S)                         # ...and it builds


def test_the_band_is_drawn_under_the_chosen_law():
    """The picture the panel draws has to be the law's own, or it shows
    planforms the run cannot reach."""
    from gui.v3 import config
    from gui.v3.stages import wing as stage

    ctx = _shell()
    S = ctx.S
    eff = config.effective_bounds(S)
    rows = [[*eff[k][0]] for k in ("chord_k1", "chord_k2", "chord_k3")]
    poly = g.chord_reach(rows, 0.6, law="poly")
    ends = g.chord_reach(rows, 0.6, law="ends")
    assert not np.allclose(poly.hi, ends.hi)
    # the ends law cannot move either end of the planform, and the band says so
    assert ends.root == pytest.approx((1.0, 1.0), abs=1e-9)
    assert ends.tip == pytest.approx((1.0, 1.0), abs=1e-9)
    assert poly.root[1] > 1.0 + 1e-6
    assert stage.CHORD_GROUPS[("", "")][0] == "wing"


# ------------------------------------------------------- 2. the two chords

def test_the_two_chords_state_the_taper_and_the_span():
    from gui.v3 import config, session

    ctx = _shell()
    S = ctx.S
    area = float(S["mission"]["s_ref_m2"])
    ctx.act("set_size_mode", "chords")
    ctx.act("set_size_chord", "root", 1.30)
    ctx.act("set_size_chord", "tip", 0.70)

    assert session.chord_taper(S) == pytest.approx(0.70 / 1.30)
    assert session.chord_span(S) == pytest.approx(2.0 * area / 2.0)
    assert session.chosen_span(S) == pytest.approx(session.chord_span(S))
    assert config.fixed_rows(S)["taper"] == pytest.approx(0.70 / 1.30)


def test_half_an_answer_is_no_answer():
    """One chord says nothing the span does not already say, and a taper
    nobody chose is exactly what this control exists to stop."""
    from gui.v3 import config, session

    ctx = _shell()
    S = ctx.S
    ctx.act("set_size_mode", "chords")
    ctx.act("set_size_chord", "root", 1.30)
    assert session.chord_taper(S) is None
    assert "taper" not in config.fixed_rows(S)
    ctx.act("set_size_chord", "tip", 0.70)
    assert "taper" in config.fixed_rows(S)
    ctx.act("set_size_chord", "tip", None)
    assert "taper" not in config.fixed_rows(S)


def test_a_chord_pair_that_implies_an_impossible_span_is_refused():
    """The chords state the span (b = 2S/(c_r+c_t)), so they reach the same
    aspect-ratio gate a typed span does — and reached it as a ``ValueError``
    out of the field's own handler, with the chord already stored. A 1.2 m
    root and a 0.1 m tip on the default 10 m² wing is a 15.4 m span at
    AR 23.6 — fine; drop the tip to 0.05 m and it is 16 m at AR 25.6, also
    fine; ask for 0.2 and 0.1 and it is 66.7 m at AR 444."""
    from gui.v3 import config, session

    ctx = _shell()
    S = ctx.S
    ctx.act("set_size_mode", "chords")
    ctx.act("set_size_chord", "root", 0.20)
    assert session.set_size_chord(S, "tip", 0.10) is False
    assert S["wing"]["chord_tip_m"] is None            # not stored either
    assert session.chord_span(S) is None
    assert config.build_cfg(S) is not None

    # ...and the pair that DOES describe a wing goes in unchanged
    assert session.set_size_chord(S, "root", 1.30) is True
    assert session.set_size_chord(S, "tip", 0.70) is True
    assert session.chord_span(S) == pytest.approx(10.0)


def test_the_interior_law_flies_exactly_the_chords_that_were_typed():
    """The whole point of the pair of features: state both end chords, pick
    the law that holds them, and the search cannot move either."""
    ctx = _shell()
    S = ctx.S
    ctx.act("set_size_mode", "chords")
    ctx.act("set_size_chord", "root", 1.30)
    ctx.act("set_size_chord", "tip", 0.70)
    ctx.act("set_chord_law", "ends")

    cfg, res = _run(S, budget=10)
    assert res.pinned == {"taper": pytest.approx(0.70 / 1.30)}
    rep = api.design_report(cfg, res.best_x)

    # rebuilt on the size the run was HANDED (span from the chords, area from
    # the mission), not on the reported one — that is the LLT quadrature of
    # the sampled chord and is a few parts in 10^4 short of S by construction
    from gui.v3 import session

    b, area = session.flown_size(S)
    w = g.Wing(b=b, S=area, taper=float(rep["geometry"]["taper"]),
               chord_coeffs=g.ChordCoeffs(rep["breakdown"]["chord_coeffs"],
                                          "ends"))
    assert float(w.chord(0.0)) == pytest.approx(1.30, rel=1e-9)
    assert float(w.chord(w.b / 2.0)) == pytest.approx(0.70, rel=1e-9)
    # ...and the law really was searched: a flat one would prove nothing
    assert any(abs(k) > 1e-6 for k in rep["breakdown"]["chord_coeffs"])


def test_the_published_law_does_not_hold_them_and_the_card_must_not_claim_it():
    """The contrast the size card warns about: under the cubic the typed
    chords are the straight-taper BASELINE, not what flies."""
    ctx = _shell()
    S = ctx.S
    ctx.act("set_size_mode", "chords")
    ctx.act("set_size_chord", "root", 1.30)
    ctx.act("set_size_chord", "tip", 0.70)

    cfg, res = _run(S, budget=10)
    rep = api.design_report(cfg, res.best_x)
    geo = rep["geometry"]
    w = g.Wing(b=float(geo["b"]), S=float(geo["S"]), taper=float(geo["taper"]),
               chord_coeffs=tuple(rep["breakdown"]["chord_coeffs"]))
    assert abs(float(w.chord(0.0)) - 1.30) > 1e-3


def test_going_back_to_the_span_gives_the_taper_back():
    from gui.v3 import config, session

    ctx = _shell()
    S = ctx.S
    ctx.act("set_size_mode", "chords")
    ctx.act("set_size_chord", "root", 1.30)
    ctx.act("set_size_chord", "tip", 0.70)
    assert "taper" in config.fixed_rows(S)

    ctx.act("set_size_mode", "span")
    assert session.chord_taper(S) is None
    assert "taper" not in config.fixed_rows(S)
    assert config.cfg_dict(S).get("pinned") is None


def test_a_span_searching_family_is_not_offered_the_chords():
    """Its size IS the design vector; there is no area to derive a span from
    and no single taper the two chords could state."""
    from gui.v3 import session

    ctx = _shell()
    S = ctx.S
    ctx.act("set_planform", "wing_loading")
    assert session.span_is_searched(S)
    assert not session.chord_span_available(S)


# --------------------------------------- 5b. and the chords do not outlive it

def test_the_planform_menu_takes_the_chords_with_the_question():
    """The chords were cleared NOWHERE, and the card that offers them is
    gated on ``chord_span_available``. Switching to the wing-loading mode
    therefore left a pair nothing on screen could reach, still setting the
    span (16.67 m against a mission stating 10 m²) and still pinning
    ``taper`` — a pin the design box's own fixed/released switches cannot
    remove, because they read ``W["fixed"]``, which never held it."""
    from gui.v3 import config, session

    ctx = _shell()
    S = ctx.S
    ctx.act("set_size_mode", "chords")
    ctx.act("set_size_chord", "root", 0.80)
    ctx.act("set_size_chord", "tip", 0.40)
    assert config.fixed_rows(S)["taper"] == pytest.approx(0.5)
    assert session.nominal_span(S) == pytest.approx(
        2.0 * float(S["mission"]["s_ref_m2"]) / 1.20)

    ctx.act("set_planform", "wing_loading")
    assert session.size_mode(S) == "span"
    assert (S["wing"]["chord_root_m"], S["wing"]["chord_tip_m"]) == (None,
                                                                    None)
    assert session.chord_span(S) is None and session.chord_taper(S) is None
    assert "taper" not in config.fixed_rows(S)
    assert (config.cfg_dict(S).get("pinned") or {}).get("taper") is None
    # ...and the span the box is taken around is the mission's own again
    assert session.nominal_span(S) == pytest.approx(session.mission_span(S))


def test_the_medium_takes_them_too():
    """A 1.3 m root chord is a statement about a 10 m² aircraft wing, not
    about a 0.144 m² hydrofoil — every air/water carry-over is refused
    outright, the admissible chord-sum intervals being disjoint."""
    from gui.v3 import config, session

    ctx = _shell()
    S = ctx.S
    ctx.act("set_size_mode", "chords")
    ctx.act("set_size_chord", "root", 1.30)
    ctx.act("set_size_chord", "tip", 0.70)
    assert "taper" in config.fixed_rows(S)

    S["wing"]["choices"]["medium"] = "water"
    session.apply_choices(S)
    assert S["medium"] == "water"
    assert session.size_mode(S) == "span"
    assert session.chord_span(S) is None
    assert (config.cfg_dict(S).get("pinned") or {}).get("taper") is None
    # the size the run flies is the hydrofoil's own, not the aircraft's
    assert session.nominal_span(S) == pytest.approx(
        api.planform_size(S["wing"]["problem"])[0])


def test_a_stale_pair_cannot_pin_a_family_that_cannot_be_asked_in_chords():
    """The gate itself, independent of the clearing above: whatever route
    leaves two chords in the state, they only mean something on a family
    whose size card can ASK for them. Until this gate the car families —
    ``api.resizable`` is False, so the whole size block is not even drawn —
    carried ``pinned={'taper': …}`` into every run."""
    from gui.v3 import config, session

    ctx = _shell()
    S = ctx.S
    ctx.act("set_planform", "wing_loading")
    assert not session.chord_span_available(S)
    S["wing"]["size_mode"] = "chords"          # as any stale route would
    S["wing"]["chord_root_m"], S["wing"]["chord_tip_m"] = 1.30, 0.70

    assert session.chord_taper(S) is None
    assert session.chord_span(S) is None
    assert "taper" not in config.fixed_rows(S)
    assert (config.cfg_dict(S).get("pinned") or {}).get("taper") is None


# ---------------------------------------------- 6. what the OPENING trend costs

def test_the_shell_opens_on_never_grows_outboard():
    """The wing V3 hands you does not get wider towards the tip.

    A default, not a ban: the library still means ``free``
    (``geometry.ChordLimits``), so every published run and every scripted
    study is untouched, and one click on the chord-law card gives the
    unconstrained case back. It is stated in ``config.flags`` rather than on
    the card so the band DRAWN and the box SEARCHED read the same answer.
    """
    from gui.v3 import config

    ctx = _shell()
    assert config.flags(ctx.S)[api.CHORD_TREND_KEY] == "root_largest"

    cfg = config.build_cfg(ctx.S)
    built = api.PROBLEM_SPECS[cfg.problem_name].build(
        cfg.mission_kwargs, cfg.flags, cfg.bounds_overrides)
    labs = list(built.param_labels)
    x = np.asarray(built.bounds, dtype=float).mean(axis=1)
    x[labs.index("chord_k1")] = 0.5             # bends the chord UP outboard
    assert built.evaluate(x.tolist()).get("feasible") is False

    ctx.act("set_chord_trend", "free")
    cfg = config.build_cfg(ctx.S)
    built = api.PROBLEM_SPECS[cfg.problem_name].build(
        cfg.mission_kwargs, cfg.flags, cfg.bounds_overrides)
    assert built.evaluate(x.tolist()).get("feasible") is True


def test_what_the_opening_trend_costs_each_law_is_on_the_card():
    """Two of the four laws are all but emptied by it, and the card says so.

    MEASURED here rather than asserted from memory, over the same 128 Sobol
    draws the design box's own probe uses: `poly` (the law the shell opens
    on) keeps most of its box, while `ends` and `kinked` bend the chord about
    the MIDDLE of the span and turn it up again outboard almost everywhere in
    their published coefficient box. A search over a box that thin reports
    "no solution" at any budget, so the number and both ways out have to be
    on the card that caused it — not discovered after a run.
    """
    from scipy.stats import qmc

    from gui.v3 import config

    kept = {}
    for law in g.CHORD_LAWS:
        ctx = _shell()
        ctx.act("set_chord_law", law)
        cfg = config.build_cfg(ctx.S)
        built = api.PROBLEM_SPECS[cfg.problem_name].build(
            cfg.mission_kwargs, cfg.flags, cfg.bounds_overrides)
        box = np.asarray(built.bounds, dtype=float)
        draws = qmc.scale(
            qmc.Sobol(box.shape[0], scramble=True, seed=0).random(128),
            box[:, 0], box[:, 1])
        kept[law] = sum(1 for x in draws
                        if (built.evaluate(x.tolist()) or {}).get("feasible"))

    assert kept["poly"] >= 64, kept        # the law the shell opens on
    assert kept["elliptic"] >= 64, kept    # a fraction in [0, f]: never turns up
    assert kept["ends"] < 32, kept         # measured 6/128
    assert kept["kinked"] < 32, kept       # measured 4/128

    # ...and the card for a law the trend empties names the share AND both
    # ways out, so the thin box is never a surprise
    ctx = _shell()
    ctx.act("set_chord_law", "ends")
    ctx.render("wing", "box")
    texts = [getattr(e, "text", None) or ""
             for e in ctx.views[("wing", "box")].descendants()]
    warn = [t for t in texts if "refused by the chord TREND" in t]
    assert warn, texts
    # the share opens the line and both ways out follow it behind the
    # "?" (widgets.split_hint), so the claim is read off the whole card
    said = " ".join(texts)
    assert "% away from straight taper" in said
    assert "either end may be the wide one" in said
