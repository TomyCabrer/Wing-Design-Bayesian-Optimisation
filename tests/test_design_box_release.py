"""A design-box row can be switched OFF.

The box view answers one question per row: what interval should the optimiser
search? For most rows the honest answer is "whatever the solver publishes" —
the user has no opinion, and typing the published numbers back in turns a
default into an override that differs from it in the seventh decimal.

So each row carries a "constrain" switch. OFF means UNCONSTRAINED, not "not
searched":

* the variable is still designed — the design vector does not change;
* the row is dropped from ``bounds_overrides`` entirely, which IS the
  family's own published box;
* a section PIN on that row is released with it (a row cannot be off and
  pinned at the same time);
* the numbers the user typed are kept, and come back with the switch.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _session():
    from gui.v3 import session
    return session.make_session("air")


def test_a_fresh_session_constrains_nothing_and_releases_nothing():
    from gui.v3 import config

    S = _session()
    assert S["wing"]["bounds_off"] == []
    assert config.released_rows(S) == set()
    assert config.bounds_overrides(S) is None
    assert all(src == "default"
               for _row, src in config.effective_bounds(S).values())


def test_releasing_a_row_drops_the_override_the_user_typed():
    from gui.v3 import config

    S = _session()
    S["wing"]["bounds"]["taper"] = [0.4, 0.6]
    assert config.bounds_overrides(S) == {"taper": [0.4, 0.6]}

    S["wing"]["bounds_off"] = ["taper"]
    assert config.bounds_overrides(S) is None          # nothing left to send
    row, source = config.effective_bounds(S)["taper"]
    assert source == "released"
    # ...and the row shown is the SOLVER's, not the user's
    default = config.spec(S).default_bounds["taper"]
    assert row == [float(default[0]), float(default[1])]


def test_the_typed_numbers_survive_the_switch():
    from gui.v3 import config

    S = _session()
    S["wing"]["bounds"]["taper"] = [0.4, 0.6]
    S["wing"]["bounds_off"] = ["taper"]
    assert config.bounds_overrides(S) is None
    S["wing"]["bounds_off"] = []
    assert config.bounds_overrides(S) == {"taper": [0.4, 0.6]}


def test_releasing_a_row_releases_the_section_pin_on_it():
    """A row cannot be 'not constrained' and 'pinned to the section chosen in
    stage 2' at the same time — that is two answers to one question."""
    from gui.v3 import config, session

    S = session.make_session("air")
    S["wing"]["choices"]["airfoil"] = "section_wing"
    from gui import nice_app as v1
    v1.normalise_choices(S["wing"]["choices"], keep="airfoil")
    session.apply_choices(S)

    S["airfoil"]["decision"] = "library"
    S["airfoil"]["section"] = {
        "name": "naca2412", "tc": 0.12,
        "w_upper": [0.17, 0.18, 0.17, 0.15],
        "w_lower": [-0.12, -0.10, -0.05, 0.02]}
    linked, _notes = config.section_link_rows(S)
    assert linked, "the family designs its section, so the pin has rows"

    row = sorted(linked)[0]
    assert config.effective_bounds(S)[row][1] == "section"
    S["wing"]["bounds_off"] = [row]
    assert config.effective_bounds(S)[row][1] == "released"
    assert row not in (config.bounds_overrides(S) or {})


def test_the_stage_switch_writes_the_state_and_the_run_follows(capsys):
    from gui.v3 import config
    from gui.v3.app import assemble

    ctx = assemble("air")
    S = ctx.S
    ctx.render("wing", "box")
    view = ctx.views[("wing", "box")]
    switches = [e for e in view.descendants()
                if type(e).__name__ == "Switch"]
    assert switches, "every row carries a constrain switch"

    S["wing"]["bounds"]["taper"] = [0.4, 0.6]
    ctx.act("set_row_on", "taper", False)
    assert S["wing"]["bounds_off"] == ["taper"]
    assert config.build_cfg(S).bounds_overrides is None

    ctx.act("set_row_on", "taper", True)
    assert S["wing"]["bounds_off"] == []
    assert config.build_cfg(S).bounds_overrides == {"taper": [0.4, 0.6]}

    err = capsys.readouterr().err
    assert "Traceback" not in err, err


def test_reset_puts_every_row_back_on(capsys):
    from gui.v3.app import assemble

    ctx = assemble("air")
    S = ctx.S
    S["wing"]["bounds"]["taper"] = [0.4, 0.6]
    ctx.act("set_row_on", "taper", False)
    ctx.act("reset_box")
    assert S["wing"]["bounds"] == {}
    assert S["wing"]["bounds_off"] == []
    err = capsys.readouterr().err
    assert "Traceback" not in err, err


def test_a_released_row_does_not_outlive_its_problem(capsys):
    """The design VECTOR changes with the family, so an answer about one of
    its rows cannot travel to the next one."""
    from gui.v3.app import assemble

    ctx = assemble("air")
    S = ctx.S
    S["wing"]["bounds"]["taper"] = [0.4, 0.6]
    ctx.act("set_row_on", "taper", False)
    assert S["wing"]["bounds_off"] == ["taper"]

    ctx.act("set_choice", "medium", "water")
    assert S["wing"]["bounds_off"] == []
    assert S["wing"]["bounds"] == {}
    err = capsys.readouterr().err
    assert "Traceback" not in err, err
