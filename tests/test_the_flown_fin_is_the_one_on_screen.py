"""Two authors for one fin, and both of them ended at stage 6.

Stage 5 asks about a vertical surface and rebuilds a lattice around the
answer. Two things downstream did not hear it.

**The deck.** ``flight.py::_live`` rebuilds the derivative deck when the CG
lever moves, and cached the result against the CG NUMBER ALONE. Nothing
invalidated that cache — not a new report, not ``_teardown``, not stage 5 —
so once the CG had been touched, changing the fin re-armed onto the PREVIOUS
fin's deck while ``AR_vertical`` and the induced-drag term updated to the new
one. Measured before the fix, on the ``tail`` design at its box centre with
the CG moved 0.3 mac aft and the fin taken to 2.5 m:

    flown  Cn_beta  0.106967      <- the 1.04 m fin, at the new CG
    honest Cn_beta  0.367735      <- what the aeroplane actually had

An aeroplane whose yaw stiffness is wrong by 3.4x, with the right number
printed on the stage beside it.

**The picture.** ``cad.fin_surface`` lofts ``geometry["fin"]`` — the design
REPORT's block — while the lattice flies ``fm.model.vertical``, built from
stage 5's :class:`flightmodel.ControlsSpec`. They agree only until somebody
answers stage 5. So every fin lever moved the physics and left the drawing
bit-identical, and switching the fin OFF still drew one.

Both are asserted as OUTCOMES: a number that must move with the answer, and a
surface that must leave the scene when the surface leaves the aeroplane.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from aerobo import api, cad, dynamics as dyn, sixdof as sd
from gui.v4 import app as v4app

#: 0.3 mac aft. Enough to fill the deck cache and to move the margin, and
#: well inside the band the lever offers.
CG_SHIFT_MAC = 0.3

#: the fin the test resizes to. More than twice the fitted 1.04447 m, so a
#: stale deck cannot be mistaken for a fresh one at the noise level.
BIG_FIN_M = 2.5


def _armed():
    """A V4 shell with a real design behind it and stage 5's deck built."""
    ctx = v4app.assemble()
    cfg = api.RunConfig(problem_name="tail", budget=4, seed=0)
    built = api.PROBLEM_SPECS["tail"].build({}, {}, None)
    ctx.S["run"]["record"] = {"pretend": "a completed run"}
    ctx.S["run"]["report"] = api.design_report(cfg, built.bounds.mean(axis=1))
    ctx.render("controls", "derivatives")
    return ctx


def _resize_fin(ctx, height_m):
    """What stage 5's ``edit`` does, without a browser."""
    ctx.S["controls"]["vertical"]["height_m"] = height_m
    ctx.act("controls_rebuild")
    return ctx.S["controls"]["fm"]


# ------------------------------------------------------------ THE DECK

def test_a_fin_change_after_the_cg_lever_reaches_the_aeroplane(capsys):
    """The defect, driven through the shell's own actions.

    The order matters and is the whole test: move the CG FIRST (which is
    what fills the cache), then change the fin. Either one alone was always
    fine.
    """
    ctx = _armed()
    ctx.act("flight_arm")
    ac0 = ctx.S["flight"]["ac"]
    cg0, mac = float(ac0.deck.x_cg), float(ac0.deck.mac)
    cn0 = float(ac0.deck.Cn_beta)

    cg1 = cg0 + CG_SHIFT_MAC * mac
    ctx.act("flight_set_live", "x_cg_m", cg1)
    stale = float(ctx.S["flight"]["ac"].deck.Cn_beta)

    fm = _resize_fin(ctx, BIG_FIN_M)
    ctx.act("flight_arm")
    flown = ctx.S["flight"]["ac"]

    honest = dyn.deck(fm.model, x_cg=cg1, mac=fm.deck.mac, b=fm.deck.b,
                      controls=fm.controls, alpha=fm.alpha_trim,
                      i_t=fm.i_t_trim)

    assert flown.deck.x_cg == pytest.approx(cg1), \
        "the CG lever stopped reaching the deck"
    assert float(flown.deck.Cn_beta) == pytest.approx(
        float(honest.Cn_beta), rel=1e-12), \
        "stage 6 flew a deck built for a different fin"
    # ...and it is not merely equal because nothing moved: the new fin is a
    # different aeroplane from both the armed one and the stale cache.
    assert float(flown.deck.Cn_beta) > 2.0 * max(cn0, stale), \
        "a 2.5 m fin must be worth far more yaw stiffness than a 1.04 m one"
    capsys.readouterr()


def test_the_stale_deck_was_worth_degrees_of_roll(capsys):
    """Not a derivative-table discrepancy — a different aeroplane to fly.

    Same sequence, but the assertion is on the trajectory: hold the rudder
    and see where the wings end up.
    """
    ctx = _armed()
    ctx.act("flight_arm")
    ac0 = ctx.S["flight"]["ac"]
    cg1 = float(ac0.deck.x_cg) + CG_SHIFT_MAC * float(ac0.deck.mac)
    ctx.act("flight_set_live", "x_cg_m", cg1)
    stale_deck = ctx.S["flight"]["ac"].deck

    fm = _resize_fin(ctx, BIG_FIN_M)
    ctx.act("flight_arm")
    flown = ctx.S["flight"]["ac"]

    V = float(ctx.S["flight"].get("V_trim") or 45.0)

    def roll_after_rudder(ac, drud=np.deg2rad(10.0), T=4.0, dt=0.005):
        st = sd.trim_level(ac, V)
        st = st[0] if isinstance(st, tuple) else st
        ac = replace(ac, controls=dict(ac.controls, rudder=drud))
        for _ in range(int(T / dt)):
            st = sd.step(ac, st, dt)
        return float(np.rad2deg(st.euler[0]))

    now = roll_after_rudder(flown)
    then = roll_after_rudder(replace(flown, deck=stale_deck))
    assert abs(now - then) > 1.0, (
        "the stale deck flew indistinguishably, so this test no longer "
        f"exercises the defect (now {now:.3f} deg, stale {then:.3f} deg)")
    capsys.readouterr()


def test_moving_only_the_cg_still_costs_nothing_on_a_repeat(capsys):
    """The cache is still a cache. Keying it on the model as well must not
    turn every repaint into a lattice solve."""
    ctx = _armed()
    ctx.act("flight_arm")
    ac0 = ctx.S["flight"]["ac"]
    cg1 = float(ac0.deck.x_cg) + CG_SHIFT_MAC * float(ac0.deck.mac)
    ctx.act("flight_set_live", "x_cg_m", cg1)
    first = ctx.S["flight"]["ac"].deck
    ctx.act("flight_set_live", "x_cg_m", cg1)
    assert ctx.S["flight"]["ac"].deck is first, \
        "the same CG on the same model rebuilt the deck"
    capsys.readouterr()


# --------------------------------------------------------- THE PICTURE

def _fin_top(ctx):
    """The highest point of the drawn fin, or None when none is drawn."""
    _rep, geom, key = ctx.act("flight_drawn_geometry")
    surfs = cad.surfaces(geom)
    tops = [float(np.max(s.Z)) for s in surfs if s.name == "fin"]
    return (max(tops) if tops else None), key, len(surfs)


def test_the_drawing_follows_the_fin_that_is_flown(capsys):
    ctx = _armed()
    ctx.act("flight_arm")
    before, _k, n_before = _fin_top(ctx)
    assert before is not None, "the tail design draws a fin to begin with"

    fm = _resize_fin(ctx, BIG_FIN_M)
    after, _k, n_after = _fin_top(ctx)

    assert n_after == n_before, "resizing the fin changed the surface COUNT"
    assert after == pytest.approx(
        float(fm.model.vertical.z_root) + float(fm.model.vertical.height)), \
        "the drawn fin is not the lattice's"
    assert after > before + 1.0, "the picture did not follow the answer"
    capsys.readouterr()


def test_a_fin_switched_off_is_not_drawn(capsys):
    """The deck says so — no ``rudder`` column, ``Cn_beta`` an honest zero —
    and now the aeroplane on screen says so too."""
    ctx = _armed()
    ctx.act("flight_arm")
    before, _k, n_before = _fin_top(ctx)
    assert before is not None

    ctx.S["controls"]["vertical"]["on"] = False
    ctx.act("controls_rebuild")
    fm = ctx.S["controls"]["fm"]
    assert getattr(fm.model, "vertical", None) is None, \
        "the switch stopped reaching the lattice"

    after, key, n_after = _fin_top(ctx)
    assert after is None, "a fin was drawn on an aeroplane that has none"
    assert n_after == n_before - 1
    assert key == ("none",)
    capsys.readouterr()


def test_the_served_mesh_is_rewritten_when_the_fin_changes(capsys):
    """The URL cache is the other half of it.

    ``view`` outlives every rebuild of the Fly view, so a URL keyed on
    "have we ever written one" served the first design's mesh for the rest
    of the session — which would have made the fix above invisible.
    """
    ctx = _armed()
    ctx.act("flight_arm")
    _t0, k0, _n = _fin_top(ctx)
    _resize_fin(ctx, BIG_FIN_M)
    _t1, k1, _n = _fin_top(ctx)
    assert k0 != k1, \
        "the STL cache key does not move with the fin, so the mesh is stale"
    capsys.readouterr()
