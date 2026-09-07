"""Stage 3's design box asks the chord law as a PLANFORM.

Every V3 shell opens on the polynomial chord law, so ``chord_k1``,
``chord_k2`` and ``chord_k3`` were the first rows a user met in the design
box — three coefficients of ``1 + Σ kⱼ ηʲ`` with a ±0.5 box each, which is
the solver's variable and nobody's design intent. Nothing on screen said what
that box draws, and it draws a lot: on the published wing the chord may reach
nearly twice the trapezoid's at the root and a twentieth of it at the tip.

So the rows leave the table and the law gets a panel of its own, per DESIGNED
SURFACE, asking the one question a person has: how far may the law bend the
chord away from the straight taper it starts from. The rules:

* the answer is stored in the coefficients the solver really searches
  (:func:`aerobo.geometry.chord_bound_for_dev`), so nothing downstream learns
  a new variable;
* an untouched panel WRITES NOTHING — a session that only looked at it still
  builds the family's published run bit-for-bit;
* the coefficients stay one disclosure away, and a row typed by hand still
  wins (the band above it follows);
* the numbers quoted are the box the RUN will search, never the family's
  published box after the session has narrowed it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _open(medium: str = "air"):
    """A shell standing on stage 3 with the box view drawn."""
    from gui.v3.app import assemble

    ctx = assemble(medium)
    ctx.act("accept_mission")
    ctx.render("wing", "box")
    return ctx


def _texts(view):
    return [getattr(e, "text", None) or "" for e in view.descendants()]


def _fields_after(view, label: str):
    """The ``ui.number`` inputs that follow ``label`` — the widgets a user
    actually types into, with their real handlers attached."""
    seen, out = False, []
    for e in view.descendants():
        if getattr(e, "text", None) == label:
            seen = True
        elif seen and type(e).__name__ == "Number":
            out.append(e)
    return out


def _switch_after(view, label: str):
    from nicegui import ui

    seen = False
    for e in view.descendants():
        if getattr(e, "text", None) == label:
            seen = True
        elif seen and isinstance(e, ui.switch):
            return e
    raise AssertionError(f"no switch after {label!r}")


def test_the_coefficients_leave_the_table_for_the_planform_they_draw():
    ctx = _open()
    assert "free chord law" in ctx.S["wing"]["problem"]
    texts = _texts(ctx.views[("wing", "box")])

    assert "Chord law · wing" in texts
    # each coefficient is asked in exactly ONE place, and that place is
    # inside the law's own panel — not in the box's table above it
    for row in ("chord_k1", "chord_k2", "chord_k3"):
        assert texts.count(row) == 1, row
        assert texts.index(row) > texts.index("Chord law · wing"), row
    assert texts.index("taper") < texts.index("Chord law · wing")

    # and what the box draws is stated in the units of a wing
    assert "root chord" in texts and "tip chord" in texts
    assert any(t.endswith("m") and "–" in t for t in texts)
    assert "flyable share of the box" in texts


def test_looking_at_the_panel_writes_nothing():
    """The published run has to stay bit-for-bit after a session that only
    read the box."""
    from gui.v3 import config

    ctx = _open()
    assert config.bounds_overrides(ctx.S) is None
    assert config.cfg_dict(ctx.S)["bounds_overrides"] is None
    ctx.render("wing", "box")
    assert config.bounds_overrides(ctx.S) is None


def test_a_deviation_is_stored_in_the_coefficients_the_solver_searches():
    from aerobo import geometry

    from gui.v3 import config

    ctx = _open()
    box = ctx.views[("wing", "box")]
    field = _fields_after(box, "may bend the chord")[0]

    # it opens showing what the PUBLISHED box already allows, which is a lot
    assert field.value > 50.0

    field.set_value(30.0)
    rows = config.bounds_overrides(ctx.S)
    assert sorted(rows) == ["chord_k1", "chord_k2", "chord_k3"]
    assert all(lo == -hi and hi > 0.0 for lo, hi in rows.values())

    eff = config.effective_bounds(ctx.S)
    taper = 0.5 * sum(eff["taper"][0])
    reach = geometry.chord_reach([rows[k] for k in sorted(rows)], taper)
    assert reach.dev_max == pytest.approx(0.30, abs=0.005)
    assert eff["chord_k1"][1] == "user"

    # the readout that follows it agrees, without the field being redrawn
    texts = _texts(box)
    assert texts.index("root chord") > texts.index("may bend the chord")
    assert _fields_after(box, "may bend the chord")[0].value == 30.0


def test_the_field_recognises_its_own_rounded_value_coming_back():
    """The same guard every V3 field carries: the browser posts the rendered
    number back, and taking it would turn an untouched box into an edit."""
    from gui.v3 import config

    ctx = _open()
    field = _fields_after(ctx.views[("wing", "box")], "may bend the chord")[0]
    field.set_value(field.value)
    assert config.bounds_overrides(ctx.S) is None


def test_a_coefficient_typed_by_hand_still_wins(capsys):
    from gui.v3 import config

    ctx = _open()
    box = ctx.views[("wing", "box")]
    lo, hi = _fields_after(box, "chord_k1")[:2]
    assert [lo.value, hi.value] == [-0.5, 0.5]
    hi.set_value(0.1)
    assert config.bounds_overrides(ctx.S) == {"chord_k1": [-0.5, 0.1]}

    # the band above it followed the row, in place
    ctx.render("wing", "box")
    assert _fields_after(ctx.views[("wing", "box")], "chord_k1")[1].value == 0.1
    assert "Traceback" not in capsys.readouterr().err


def test_releasing_the_law_says_the_published_box_is_searched():
    from gui.v3 import config

    ctx = _open()
    for row in ("chord_k1", "chord_k2", "chord_k3"):
        _switch_after(ctx.views[("wing", "box")], row).set_value(False)
    assert config.released_rows(ctx.S) >= {"chord_k1", "chord_k2", "chord_k3"}
    assert config.bounds_overrides(ctx.S) is None
    assert any("released" in t.lower()
               for t in _texts(ctx.views[("wing", "box")]))

    # asking for a deviation is asking for the rows to be constrained again
    _fields_after(ctx.views[("wing", "box")],
                  "may bend the chord")[0].set_value(20.0)
    assert not config.released_rows(ctx.S) & {"chord_k1", "chord_k2",
                                              "chord_k3"}
    assert sorted(config.bounds_overrides(ctx.S)) == ["chord_k1", "chord_k2",
                                                      "chord_k3"]


def test_every_designed_surface_is_asked_about_its_own_law(capsys):
    """The law is a property of a SURFACE: a designed tail carries one of its
    own and a tandem one per wing, so one panel over all of them would ask
    about a planform that does not exist."""
    ctx = _open()
    ctx.act("set_choice", "tail", True)
    ctx.act("set_choice", "tail_design", "planform")
    ctx.render("wing", "box")
    texts = _texts(ctx.views[("wing", "box")])
    assert "Chord law · wing" in texts
    assert "Chord law · second surface" in texts
    assert texts.count("chord_k1") == 1 and texts.count("chord_k1_t") == 1

    pair = _open()
    pair.act("set_choice", "system", "tandem")
    pair.render("wing", "box")
    texts = _texts(pair.views[("wing", "box")])
    assert pair.S["wing"]["problem"] == "tandem + free chord law"
    assert "Chord law · front wing" in texts
    assert "Chord law · rear wing" in texts
    # the pair's per-wing size is not this shell's to state, so the band is
    # quoted in chords rather than inventing metres
    assert any("× root chord (straight taper)" in t for t in texts)
    assert "Traceback" not in capsys.readouterr().err


def test_the_second_surface_deviation_moves_only_its_own_rows():
    from gui.v3 import config

    ctx = _open()
    ctx.act("set_choice", "tail", True)
    ctx.act("set_choice", "tail_design", "planform")
    ctx.render("wing", "box")
    fields = _fields_after(ctx.views[("wing", "box")], "may bend the chord")
    # the wing's panel comes first; the second surface's field follows it
    tail_field = _fields_after(ctx.views[("wing", "box")],
                               "Chord law · second surface")[0]
    assert tail_field in fields
    tail_field.set_value(25.0)
    rows = config.bounds_overrides(ctx.S)
    assert sorted(rows) == ["chord_k1_t", "chord_k2_t", "chord_k3_t"]


def test_the_derived_geometry_follows_the_box_it_is_derived_from():
    """The same box read twice — the panel that says what the law may draw,
    and the summary row underneath that names it — so the two cannot
    disagree. Typing must not rebuild the view (the focus trap), so the
    derived rows are redrawn by hand; they were not, and the row went on
    quoting the published ±0.5 beside a box the session had just narrowed.
    """
    ctx = _open()
    box = ctx.views[("wing", "box")]
    assert any("|k| ≤ 0.5" in t for t in _texts(box))

    # EVERY panel, not the first one: the opening aeroplane designs its tail,
    # so it carries a law per surface (chord_k*_t beside chord_k*) and the
    # summary row quotes the widest of them. Narrowing one and reading the
    # row would then be a test of which surface happens to be drawn first.
    fields = _fields_after(box, "may bend the chord")
    assert len(fields) >= 1
    for f in fields:
        f.set_value(25.0)
    texts = _texts(box)
    assert any("±25%" in t for t in texts), [t for t in texts if "±" in t]
    assert not any("|k| ≤ 0.5" in t for t in texts)


def test_every_chord_row_the_registry_declares_has_a_panel_to_live_in():
    """The rows leave the table, so a row the panel does not recognise would
    leave the design box altogether. Asked of the REGISTRY, not of a list:
    that is the rule that stops the stale-menu bug class."""
    from aerobo import api

    from gui.v3.stages.wing import CHORD_GROUPS, CHORD_ROW

    seen = set()
    for spec in api.PROBLEM_SPECS.values():
        for row in spec.default_bounds:
            if not row.startswith("chord_"):
                continue
            m = CHORD_ROW.fullmatch(row)
            assert m, row
            seen.add((m.group(1) or "", m.group(3) or ""))
    assert seen and seen <= set(CHORD_GROUPS), seen - set(CHORD_GROUPS)


def test_a_row_no_panel_claims_stays_in_the_table(monkeypatch):
    """...and if one ever does appear, it falls back to the table it came
    from rather than vanishing between the two."""
    from aerobo import api

    ctx = _open()
    spec = api.PROBLEM_SPECS[ctx.S["wing"]["problem"]]
    monkeypatch.setitem(spec.default_bounds, "chord_wingtip", [-0.5, 0.5])
    ctx.render("wing", "box")

    texts = _texts(ctx.views[("wing", "box")])
    assert texts.count("chord_wingtip") == 1
    assert texts.index("chord_wingtip") < texts.index("Chord law · wing")


def test_the_law_the_run_gets_is_the_law_the_panel_showed():
    """One question, one place: what the panel writes is what
    ``api.run`` is handed."""
    from gui.v3 import config

    ctx = _open()
    _fields_after(ctx.views[("wing", "box")],
                  "may bend the chord")[0].set_value(15.0)
    cfg = config.build_cfg(ctx.S, seed=0)
    assert cfg.bounds_overrides == config.bounds_overrides(ctx.S)
    assert set(cfg.bounds_overrides) == {"chord_k1", "chord_k2", "chord_k3"}
