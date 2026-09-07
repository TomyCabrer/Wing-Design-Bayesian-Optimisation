"""A canard was charged for a fin, flew one, weighed one — and reported none.

Stage 1 asks "add a vertical stabiliser (fin and rudder)". On a canard the
answer reached NOTHING: ``api.design_report`` excluded the layout from its
fin block, so the report stated no vertical surface whichever way the
switch was thrown, ``session.spiral_dihedral`` answered ``no_fin`` and gave
the card no spiral number at all, and ``flightmodel`` invented a fin of its
own to rebuild with.

The exclusion's stated reason was that "a canard has an aft fin whose arm
is NOT the (negative, upstream) surface station this block carries". The
station the block carries is ``dist_m`` — the UNSIGNED separation — which
is exactly the arm the solver's own fin is sized and placed at
(``wingtail._lateral_fin`` passes ``v["dist"]``) and exactly the arm the
drag book charges. The two layouts' ``cd0_fin`` agree to every digit, which
is the measurement that makes the inclusion right rather than convenient,
and the first test here pins it.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aerobo import api, fin as _fin                             # noqa: E402

NAME = "tail [free height, designed tail] + free chord law"


def _report(tail_type: str, fin: bool) -> dict:
    flags = {"tail_type": tail_type, "fin": fin,
             "fuselage_diameter_m": 0.22}
    spec = api.PROBLEM_SPECS[NAME]
    built = spec.build({}, flags, None)
    bnds = built.problem.bounds
    x = 0.5 * (bnds[:, 0] + bnds[:, 1])
    return api.design_report(api.RunConfig(problem_name=NAME, flags=flags), x)


def test_a_canard_is_charged_the_same_fin_drag_as_its_aft_tailed_twin():
    """THE PREMISE, and the reason the layout belongs in the fin block.

    If these ever stop agreeing, the canard's fin is a DIFFERENT surface
    from the aft tail's and the report must not size it the same way.
    """
    can = (_report("canard", True)["breakdown"] or {})["cd0_fin"]
    con = (_report("conventional", True)["breakdown"] or {})["cd0_fin"]
    assert can == con, (can, con)
    assert can > 0.0


def test_the_canard_now_states_the_vertical_surface_it_flies():
    geom = _report("canard", True)["geometry"]
    assert "fin" in geom and geom["fin"], geom.get("fin")
    blk = geom["fin"]
    # AFT, at the separation — not at the (negative) surface station
    assert blk["x_qc_m"] > 0.0, blk
    bd = _report("canard", True)["breakdown"]
    assert blk["x_qc_m"] == abs(float(bd["l_t"])) == float(
        bd["dist_from_wing_m"])


def test_it_is_the_SOLVER_s_own_fin_and_not_a_second_one():
    """ONE surface: the block the report states must be the geometry
    ``wingtail`` puts in the lattice, asked of ``fin`` itself."""
    rep = _report("canard", True)
    bd, blk = rep["breakdown"], rep["geometry"]["fin"]
    spec = api.PROBLEM_SPECS[NAME]
    prob = spec.build({}, {"tail_type": "canard", "fin": True,
                           "fuselage_diameter_m": 0.22}, None).problem
    flown = _fin.fin_for_layout(
        b=prob.b, S=prob.S, l_t=float(bd["dist_from_wing_m"]),
        tail_type="canard", dz=float(bd["dz_tail"]),
        fin=_fin.has_fin(prob), **_fin.fin_law_kwargs(prob))
    assert flown is not None
    assert blk["S"] == flown.S
    assert blk["x_qc_m"] == flown.x_qc
    assert blk["z_root_m"] == flown.z_root
    assert blk["height_m"] == flown.height


def test_switching_the_fin_off_states_none_rather_than_saying_nothing():
    geom = _report("canard", False)["geometry"]
    assert "fin" in geom and geom["fin"] is None
    assert _fin.states_no_fin(geom)


def test_the_aft_tailed_twin_did_not_move():
    """The change is an inclusion, not a re-sizing: every layout that was
    already in the block reports exactly what it reported before."""
    blk = _report("conventional", True)["geometry"]["fin"]
    assert blk["x_qc_m"] == 5.5
    assert blk["S"] == 0.7272727272727273
    assert blk["height_m"] == 1.0444659357341872
    assert blk["chord_m"] == 0.6963106238227914
    assert blk["z_root_m"] == 1.75


# ------------------------------------------- and the switch reaches it now

def test_the_stage_one_fin_switch_moves_a_canard_s_lateral_numbers():
    """The reported symptom, through the shell's own instrument.

    Driven with ``ctx.act("set_fin", ...)`` — the registered action the
    switch fires — so a switch wired to nothing fails this.
    """
    from gui.v3 import session as sess
    from gui.v3.app import assemble

    ctx = assemble("air")
    ctx.act("set_choice", "tail", True)
    ctx.act("set_choice", "tail_type", "canard")

    ctx.act("set_fin", True)
    on = sess.spiral_dihedral(ctx.S)
    ctx.act("set_fin", False)
    off = sess.spiral_dihedral(ctx.S)

    assert on["status"] != "no_fin", on
    assert on["Cn_beta"] is not None and on["Cn_beta"] > 0.0, on
    assert off["status"] == "no_fin", off
    assert off.get("Cn_beta") is None
    # ...and the design STATES the surface, rather than the rebuild
    # assuming one: that flag is what the card reads to know whether the
    # yaw stiffness it quotes belongs to this aeroplane
    assert on["fin_stated"] is True, on


# ------------------------------------------------------------- the rule itself

def test_the_arm_a_fin_is_placed_at_is_the_separation_not_the_station():
    """``fin.fin_arm``: one author for "the fin is aft on every layout".

    Exercised on a record that carries ONLY the signed station, which is
    the case the sign rule exists for — with ``dist_m`` present the two
    agree on every aft layout and the rule is invisible.
    """
    assert _fin.fin_arm(dist_m=5.5, l_t=5.5) == 5.5
    assert _fin.fin_arm(dist_m=5.5, l_t=-5.5) == 5.5
    assert _fin.fin_arm(dist_m=None, l_t=-5.5) == 5.5       # canard, no dist
    assert _fin.fin_arm(dist_m=None, l_t=None) is None
    assert _fin.fin_arm(dist_m=float("nan"), l_t=-4.0) == 4.0
    assert _fin.fin_arm(dist_m=True, l_t=-3.0) == 3.0       # a bool is not an arm


def test_a_fin_placed_at_the_signed_station_would_sit_upstream():
    """WHY the rule is not cosmetic: the same law at the signed station
    puts the surface ahead of the wing, where it destabilises."""
    aft = _fin.size_fin(b=10.0, S=10.0, l_t=5.5, tail_type="canard",
                        z_root=1.75)
    fwd = _fin.size_fin(b=10.0, S=10.0, l_t=-5.5, tail_type="canard",
                        z_root=1.75)
    assert aft.S == fwd.S                    # the AREA is the same law
    assert aft.x_qc == +5.5 and fwd.x_qc == -5.5
