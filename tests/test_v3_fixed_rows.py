"""A design-box row the user has DECIDED — stage 3's third state.

The box could say two things about a variable: search it over this band, or
search it over the solver's own (released). It could not say the one thing an
engineer says most often — "this one is 0.7, do not touch it" — because the
only way the table could express that was a band of width zero, which the api
refuses outright: scipy's Sobol engine raises on it and constrained BO falls
back to random draws on every iteration while still calling itself BO.

So the row gained a FIX. The variable leaves the design vector
(``api.RunConfig.pinned``), the search is that many dimensions smaller, and
everything the run reports is still the whole wing.

What this file holds:

* the shell's fix reaches the run — the variable is not searched, and the
  design flown carries the typed value;
* fixed and released are exclusive, and a fix survives nothing it should not
  (a family change clears it, a reset clears it, a chord-law band claims its
  own rows back);
* a value outside the row's published band WIDENS it rather than being
  clamped into it — a band is a statement about a search, a fixed row is not
  searched, and a calibrated band is a default rather than a ban;
* every MID-BOX readout in the shell follows the fix, because the middle of a
  band nobody searches is not a number about this design;
* an untouched session sends nothing at all: ``pinned`` is None.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aerobo import api                              # noqa: E402


def _shell(medium: str = "air"):
    from gui.v3.app import assemble

    return assemble(medium)


# ------------------------------------------------------- 1. nothing by default

def test_an_untouched_session_fixes_nothing():
    from gui.v3 import config

    S = _shell().S
    assert config.fixed_rows(S) == {}
    assert config.cfg_dict(S).get("pinned") is None
    assert config.build_cfg(S).pinned is None


# --------------------------------------------------------- 2. the fix travels

def test_a_fixed_row_leaves_the_design_vector():
    """End to end: fix taper in the shell, run, and the search never moves
    it — measured on the flown planform, not on the state dict."""
    from gui.v3 import config

    ctx = _shell()
    S = ctx.S
    S["wing"]["problem"] = "trim wing"
    ctx.act("set_row_fixed", "taper", True)
    ctx.act("set_fixed_value", "taper", 0.62)

    assert config.fixed_rows(S) == {"taper": pytest.approx(0.62)}
    cfg = config.build_cfg(S)
    assert cfg.pinned == {"taper": pytest.approx(0.62)}

    cfg.optimiser, cfg.budget, cfg.seed = "bo", 12, 0
    res = api.run(cfg)
    assert res.searched_dim == res.dim - 1
    X = np.asarray(res.eval_x, dtype=float)
    assert np.allclose(X[:, list(res.param_labels).index("taper")], 0.62)
    geo = api.design_report(cfg, res.best_x)["geometry"]
    assert geo["taper"] == pytest.approx(0.62)


def test_the_fix_opens_at_the_middle_of_the_band_it_replaces():
    """It has to open at a value this session has already said something
    about — the centre of the row's own band — or the switch would silently
    move the design the moment it is flipped."""
    from gui.v3 import config

    ctx = _shell()
    S = ctx.S
    S["wing"]["problem"] = "trim wing"
    band = config.effective_bounds(S)["taper"][0]
    ctx.act("set_row_fixed", "taper", True)
    assert config.fixed_rows(S)["taper"] == \
        pytest.approx(0.5 * (band[0] + band[1]))


def test_a_fix_outside_the_published_band_widens_it_rather_than_refusing():
    """A published band is a default, not a ban. A fixed row is not searched,
    so clamping the value would refuse a design nobody's physics refuses —
    the band travels widened instead, and the two statements agree."""
    from gui.v3 import config

    ctx = _shell()
    S = ctx.S
    S["wing"]["problem"] = "trim wing"
    lo, hi = config.effective_bounds(S)["taper"][0]
    ctx.act("set_row_fixed", "taper", True)
    ctx.act("set_fixed_value", "taper", hi + 0.35)      # an inverse taper

    assert config.fixed_rows(S)["taper"] == pytest.approx(hi + 0.35)
    sent = config.bounds_overrides(S)
    assert sent["taper"][1] >= hi + 0.35
    assert sent["taper"][0] == pytest.approx(lo)

    # ...and what it sends is a config the api will build and run
    cfg = api.RunConfig(**{**config.cfg_dict(S), "optimiser": "sobol",
                           "budget": 4})
    res = api.run(cfg)
    i = list(res.param_labels).index("taper")
    assert np.allclose(np.asarray(res.eval_x)[:, i], hi + 0.35)


def test_fixing_a_row_takes_it_off_the_released_list():
    """"nothing constrains it" and "it is exactly this" cannot both hold."""
    from gui.v3 import config

    ctx = _shell()
    S = ctx.S
    S["wing"]["problem"] = "trim wing"
    ctx.act("set_row_on", "taper", False)
    assert "taper" in config.released_rows(S)
    ctx.act("set_row_fixed", "taper", True)
    assert "taper" not in config.released_rows(S)
    assert "taper" in config.fixed_rows(S)


def test_unfixing_gives_the_variable_back():
    from gui.v3 import config

    ctx = _shell()
    S = ctx.S
    S["wing"]["problem"] = "trim wing"
    ctx.act("set_row_fixed", "taper", True)
    ctx.act("set_row_fixed", "taper", False)
    assert config.fixed_rows(S) == {}
    assert config.cfg_dict(S).get("pinned") is None


# ------------------------------------------------------- 3. it never outlives

def test_a_family_change_clears_the_fix():
    """A value is a stronger statement than a band, and the band is already
    cleared: a row of the same name on another family is another variable."""
    from gui.v3 import config, session

    ctx = _shell()
    S = ctx.S
    was = S["wing"]["problem"]
    ctx.act("set_row_fixed", "taper", True)
    assert config.fixed_rows(S)

    S["wing"]["choices"]["chord"] = "fixed"     # the chord-law twin -> its base
    session.apply_choices(S)
    assert S["wing"]["problem"] != was
    assert (S["wing"].get("fixed") or {}) == {}


def test_resetting_the_box_clears_the_fix():
    from gui.v3 import config

    ctx = _shell()
    S = ctx.S
    ctx.act("set_row_fixed", "taper", True)
    ctx.act("reset_box")
    assert config.fixed_rows(S) == {}


def test_asking_the_chord_law_for_a_band_unfixes_its_coefficients():
    """The deviation field WRITES the coefficient rows; a coefficient left
    pinned would sit outside the band drawn under the field that wrote it."""
    from gui.v3 import config

    ctx = _shell()
    S = ctx.S
    # every shell opens ON the chord law (nice_app.BUILDER_START)
    assert S["wing"]["choices"]["chord"] == "free"
    rows = [k for k in config.effective_bounds(S) if k.startswith("chord_k")]
    assert rows
    ctx.act("set_row_fixed", rows[0], True)
    assert rows[0] in config.fixed_rows(S)

    ctx.act("set_chord_dev", ("", ""), 40.0)
    assert rows[0] not in config.fixed_rows(S)


# --------------------------------------------------- 4. the readouts follow it

def test_every_mid_box_readout_is_the_fixed_value():
    """The shell quotes mid-box numbers everywhere (the screening chord, the
    taper it reports, the geometry it draws). Fixed means the middle of that
    row IS the value."""
    from gui.v3 import session

    ctx = _shell()
    S = ctx.S
    S["wing"]["problem"] = "trim wing"
    ctx.act("set_row_fixed", "taper", True)
    ctx.act("set_fixed_value", "taper", 0.35)
    assert session._searched_box(S)["taper"] == [pytest.approx(0.35)] * 2
    # ...and the band a composite would be normalised against is measured on
    # the same box, so a fix after measuring shows up as stale
    assert session.wing_band_box(S)["taper"] == [pytest.approx(0.35)] * 2


def test_the_budget_recommendation_is_made_at_the_searched_dimension():
    """The budget law is a law in the dimension actually searched."""
    plan_full = api.recommended_search("trim wing", measure_cost=False)
    plan_pinned = api.recommended_search("trim wing", measure_cost=False,
                                         pinned={"taper": 0.5})
    assert plan_pinned.dim == plan_full.dim - 1
