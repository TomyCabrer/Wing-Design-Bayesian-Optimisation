"""The box the design table calls "default" is the box THIS mission searches.

``config.default_bounds`` is the band every un-narrowed row is painted at, and
the rows a FLAG moves (``config.FLAG_MOVED_ROWS``) are read off a freshly
built problem rather than off the registry's static table — that table exists
because a static band is the 10 m trim wing's band and the run may be a model
aeroplane's.

It was built with an EMPTY mission while ``config.cfg_dict`` builds with
``config.mission_kwargs(S)``, so the "default" was the FAMILY'S PUBLISHED
mission's band and not this session's. ``ws_pa`` is derived as a fraction band
around W/S (``objective.Problem``), so every weight edit moved the searched
centre and left the shown one where it was — the exact failure FLAG_MOVED_ROWS
was written to prevent, one level in.

The rule this file holds the shell to is the repo's: **the box shown is the
box searched**.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

#: a family whose W/S is a design-box row AND whose mission moves it
_WS_FAMILY = "trim wing + free span + free W/S"


def _session(w_n: float | None = None) -> dict:
    from gui.v3 import session

    S = session.make_session("air")
    S["wing"]["problem"] = _WS_FAMILY
    if w_n is not None:
        S["mission"]["W_N"] = float(w_n)
    session.sync_wing_from_mission(S)
    return S


def _searched(S: dict, row: str):
    """The band the RUN gets for ``row`` — built the way cfg_dict builds."""
    from aerobo import api
    from gui.v3 import config

    spec = api.PROBLEM_SPECS[S["wing"]["problem"]]
    built = spec.build(config.mission_kwargs(S) if spec.uses_mission else None,
                       config.flags(S), None)
    labels = list(built.param_labels)
    lo, hi = built.bounds[labels.index(row)]
    return [float(lo), float(hi)]


def test_an_edited_weight_moves_the_shown_wing_loading_band():
    """The measured case: at W_N = 700 N the table said 32.640125-75.202848
    and the search flew 35.0-75.202848, so a pin at 33 Pa — a value inside the
    band on screen — was refused by the run."""
    from gui.v3 import config

    S = _session(700.0)
    assert config.mission_kwargs(S) == {"W_N": 700.0}, \
        "the premise: this weight really is an edit the problem honours"

    shown = config.default_bounds(S)["ws_pa"]
    searched = _searched(S, "ws_pa")
    assert shown == pytest.approx(searched, rel=0, abs=0), (shown, searched)

    # ...and the divergence this replaces was a REAL one, not a rounding
    # difference: the published band's floor is not the flown one here
    from aerobo import api

    static = api.PROBLEM_SPECS[_WS_FAMILY].default_bounds["ws_pa"]
    assert searched[0] > float(static[0]) + 1.0, (static, searched)


def test_the_shown_floor_is_a_value_the_run_will_actually_take():
    """The consequence a user meets: fixing a row at a number the box shows.

    ``api.size_box_conflicts`` judges a pin against the box the run searches,
    so a floor read off the design table has to be inside it.
    """
    from aerobo import api
    from gui.v3 import config

    S = _session(700.0)
    shown = config.default_bounds(S)["ws_pa"]
    # what the FIX switch on the row writes (wing.py's _set_fixed)
    S["wing"]["fixed"] = {"ws_pa": float(shown[0])}
    cfg = config.build_cfg(S)
    assert cfg.pinned["ws_pa"] == pytest.approx(float(shown[0]))
    assert api.size_box_conflicts(cfg) == []
    # and it really runs: the pin is inside the box the problem is built with
    api.PROBLEM_SPECS[_WS_FAMILY].build(config.mission_kwargs(S),
                                        config.flags(S),
                                        cfg.bounds_overrides)


def test_an_untouched_mission_is_unchanged_to_the_last_bit():
    """A session that has edited nothing sends ``{}``, so this fix moves no
    published band — which is why the original report's untouched-session
    divergence could not be reproduced (AUDIT 4.3)."""
    from aerobo import api
    from gui.v3 import config

    S = _session()
    assert config.mission_kwargs(S) == {}
    shown = config.default_bounds(S)["ws_pa"]
    static = api.PROBLEM_SPECS[_WS_FAMILY].default_bounds["ws_pa"]
    assert shown == [float(static[0]), float(static[1])] or \
        shown == _searched(S, "ws_pa")
    assert shown == _searched(S, "ws_pa")


def test_a_mission_the_family_cannot_fly_leaves_the_static_box():
    """The guard is the point of the try/except: at W_N = 7000 N the build
    RAISES (no loading the mission allows), and a design box that vanished
    would tell the user less than one quoting the published band."""
    from aerobo import api
    from gui.v3 import config

    S = _session(7000.0)
    spec = api.PROBLEM_SPECS[_WS_FAMILY]
    with pytest.raises(ValueError):
        spec.build(config.mission_kwargs(S), config.flags(S), None)

    shown = config.default_bounds(S)["ws_pa"]
    static = api.PROBLEM_SPECS[_WS_FAMILY].default_bounds["ws_pa"]
    assert shown == [float(static[0]), float(static[1])], shown
