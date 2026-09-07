"""WHICH WAY UP the second surface flies its section — asked, not derived.

Three things, and the first is a bug the other two are the fix for.

``tail.tail_polar`` mounts a stabiliser upside down exactly when the trim
balance asks it to push down, and the rule that reads that load excludes the
surface's OWN camber couple (``tail.stabiliser_load``): mirroring a section
mirrors its couple, so a reading that counted it would be letting the choice
vote on itself. ``wingtail.py`` — the nonplanar wing+tail core — counted it.
On a normal aeroplane the stabiliser's couple is a few per cent of the
wing's and nothing shows; on a stabiliser large against its wing it is
dominant, and the two cores then disagree about which way the surface is
mounted AND about the sign of what it carries. That is reachable from the
shell today, because ``tail.S_T_BOUNDS`` is in absolute m^2: a 0.5 kg model
aeroplane is searched over the same 0.5-3.0 m^2 tail box a 10 m one is.

Then the mounting itself. ``api.TAIL_MOUNT_KEY`` takes all three answers —
follow the load, upright, upside down — and the ENGINE tests below are about
that flag reaching the section actually flown.

The SHELL no longer asks. V3 states one answer for every family that has a
second surface: it is built to push DOWN (``config.TAIL_MOUNT``), because
that is what mounting a cambered aerofoil upside down is for. The last test
here is that contract — no control, one flag, on every consumer that reads
it — and it is deliberately paired with the engine tests, so a shell that
stopped sending it could not pass by deleting the question.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

#: a 0.5 kg model aeroplane: 1.2 m span, 0.18 m^2 wing. Flown through the
#: family's own (absolute) tail box, so the stabiliser comes out several
#: times the wing's area and its couple dominates the balance.
MODEL = {"b_m": 1.2, "S_m2": 0.18}
MODEL_MISSION = {"W_N": 4.905, "V": 12.0}


def _at(built, *values):
    return built.evaluate(np.asarray(values, dtype=float))


def test_the_two_cores_agree_which_way_a_big_stabiliser_is_mounted():
    """The lifting-line core and the nonplanar one, at MATCHED designs.

    Not "wingtail decides on the wing's couple" — that restates the code.
    What is asserted is the outcome that was wrong: the two solvers put the
    same SIGN on the same surface at the same design, on an aeroplane whose
    stabiliser is big enough for the excluded term to matter.

    Measured before the fix, at S_t = 1.75 m^2 on this aircraft: the LLT
    carried CL_t -0.0064 (a download, section upright) and the VLM +0.0066
    (up-load, section inverted) — the tail in the geometry view mounted
    upside down and then trimmed to lift.
    """
    from aerobo import api, tail

    # the family's own tail box is a FRACTION of this wing, so reaching a
    # stabiliser several times the wing's area now takes a band the user
    # states — which is exactly how the aeroplane in the bug report got one
    wide = {"S_t_m2": [0.5, 3.0], "l_t_m": [3.0, 8.0]}
    llt = api.PROBLEM_SPECS["tail"].build(MODEL_MISSION, MODEL, wide)
    vlm = api.PROBLEM_SPECS["tail [free height]"].build(
        MODEL_MISSION, MODEL, wide)
    z_t = 0.05 * MODEL["b_m"]
    for s_t, l_t in ((0.5, 3.0), (1.75, 5.5), (3.0, 8.0)):
        a = _at(llt, 0.6, 0.0, -2.0, s_t, l_t)
        b = _at(vlm, 0.6, 0.0, -2.0, s_t, l_t, z_t)
        where = f"S_t={s_t} l_t={l_t}"
        # the stabiliser's own couple really is the dominant term here —
        # otherwise this aircraft would not exercise the rule at all. Its
        # couple is cm_ac * S_t * mac_t against the wing's cm_ac * S * mac,
        # so the ratio is a pure geometry statement about how oversized the
        # surface is: 7x at the small end of this band and 96x at the large.
        mac_t = float(np.sqrt(s_t / tail.TAIL_AR))
        wing_ref = MODEL["S_m2"] * (MODEL["S_m2"] / MODEL["b_m"])
        assert (s_t * mac_t) / wing_ref > 5.0, (where, s_t * mac_t / wing_ref)
        # Scalar-ness is ASSERTED, not assumed. What stood here compared
        # b["CL_t"] against np.float64(b["CL_t"]) — a value against an exact
        # widening of ITSELF — which a one-element ARRAY also passes, and so
        # do the sign, flag and tolerance comparisons (`array([True])` is
        # truthy). A core handing back array([-0.0064]) in place of a number
        # kept every line of this test green.
        for core, r in (("llt", a), ("vlm", b)):
            assert isinstance(r["CL_t"], (float, np.floating)) \
                and np.ndim(r["CL_t"]) == 0, \
                (where, core, type(r["CL_t"]), r["CL_t"])
            assert isinstance(r["tail_section_inverted"], (bool, np.bool_)), \
                (where, core, type(r["tail_section_inverted"]))
        # ...and the two cores are then compared as NUMBERS and as BOOLS, so
        # nothing below can be satisfied by one truthy element in a container
        cl_a, cl_b = float(a["CL_t"]), float(b["CL_t"])
        assert abs(cl_a) > 1e-6, \
            (where, "a stabiliser carrying nothing cannot exercise the rule",
             cl_a)
        assert (cl_a < 0.0) == (cl_b < 0.0), (where, cl_a, cl_b)
        assert bool(a["tail_section_inverted"]) \
            is bool(b["tail_section_inverted"]), \
            (where, a["tail_section_inverted"], b["tail_section_inverted"])
        assert abs(cl_a - cl_b) < 0.05 * abs(cl_a) + 1e-4, \
            (where, cl_a, cl_b)


def test_the_published_aeroplane_is_untouched_by_the_rule():
    """The two readings choose the same orientation everywhere on the
    published 10 m aeroplane's own box, so no published run moves."""
    from aerobo import geometry, tail
    from aerobo.polar import default_polar, section_cm_ac

    prob = tail.TailProblem()
    pol = (default_polar(prob.polar) if isinstance(prob.polar, str)
           else prob.polar)
    wing = geometry.Wing(b=prob.b, S=prob.S, taper=0.6,
                         twist_root_deg=0.0, twist_tip_deg=-2.0)
    m_w = section_cm_ac(pol) * wing.S * wing.mac
    for s_t in np.linspace(*tail.area_band(prob.S), 11):
        m_t = section_cm_ac(pol) * s_t * float(np.sqrt(s_t / tail.TAIL_AR))
        for l_t in np.linspace(*tail.arm_band(prob.b), 11):
            wing_only = tail.trim_lift_coefficient(
                prob.CL_target, prob.S, prob.x_cg, s_t, l_t, M_ac=m_w)
            with_own = tail.trim_lift_coefficient(
                prob.CL_target, prob.S, prob.x_cg, s_t, l_t, M_ac=m_w + m_t)
            assert (wing_only < 0.0) == (with_own < 0.0), (s_t, l_t)


def test_a_stated_mounting_is_the_section_actually_flown():
    """``tail_mount`` is a value the user states, on every family that has a
    surface to mount — and it reaches the SECTION, not just the report."""
    from aerobo import api

    for name in ("tail", "tail (fixed arm)", "tail [free height]",
                 "tail [designed tail]"):
        assert api.TAIL_MOUNT_KEY in api.PROBLEM_SPECS[name].flags, name
        seen = {}
        for mount in ("auto", "upright", "inverted"):
            flags = dict(MODEL)
            flags[api.TAIL_MOUNT_KEY] = mount
            api.check_flags(name, flags, "test")
            built = api.PROBLEM_SPECS[name].build(MODEL_MISSION, flags, None)
            res = built.evaluate(np.array([0.5 * (lo + hi)
                                           for lo, hi in built.bounds]))
            seen[mount] = res
            assert res["tail_section_follows_load"] is (mount == "auto"), \
                (name, mount)
        assert seen["upright"]["tail_section_inverted"] is False, name
        assert seen["inverted"]["tail_section_inverted"] is True, name
        # ...and the mirrored section is a different aeroplane, not a label:
        # its own couple flips with it, so the incidence that trims it moves
        assert abs(seen["inverted"]["i_t_deg"]
                   - seen["upright"]["i_t_deg"]) > 0.5, name


def test_an_unstated_mounting_is_the_published_run_bit_for_bit():
    from aerobo import api

    x = np.array([0.6, 0.0, -2.0, 1.75, 5.5])
    plain = api.PROBLEM_SPECS["tail"].build({}, {}, None)
    auto = api.PROBLEM_SPECS["tail"].build({}, {"tail_mount": "auto"}, None)
    a, b = _at(plain, *x), _at(auto, *x)
    assert a["LoD"] == b["LoD"]
    assert a["CL_t"] == b["CL_t"]
    assert a["tail_section_inverted"] == b["tail_section_inverted"]


def test_a_mounting_that_is_not_one_is_refused_by_name():
    import pytest

    from aerobo import api

    with pytest.raises(ValueError, match="not a mounting"):
        api.PROBLEM_SPECS["tail"].build(
            MODEL_MISSION, dict(MODEL, tail_mount="sideways"), None)


def test_stage_three_states_the_mounting_and_it_reaches_the_run(capsys):
    """The shell asks nothing and sends "upside down" anyway.

    Driving ``config.flags`` alone would pass while the card went on
    describing a choice that no longer exists, so the read-out is asserted
    beside the flag — and so is the absence of the control that used to
    write it.
    """
    from nicegui import ui

    from aerobo import api
    from gui.v3 import config, session
    from gui.v3.app import assemble

    ctx = assemble()
    ctx.act("accept_mission")
    ctx.act("set_choice", "tail", True)
    S = ctx.S
    ctx.render("wing", "type")
    capsys.readouterr()

    view = ctx.views[("wing", "type")]
    assert not [e for e in view.descendants()
                if isinstance(e, ui.toggle) and "inverted" in (e.options or {})]

    # ...and the answer travels anyway, to the run and to the two views that
    # draw the section the way it is mounted
    assert config.flags(S)[api.TAIL_MOUNT_KEY] == "inverted"
    assert session.trim_lift(S)["inverted"] is True

    texts = [getattr(e, "text", "") or "" for e in view.descendants()]
    assert any("UPSIDE DOWN" in t for t in texts), texts

    # a family with no second surface has nothing to mount, and a flag it
    # does not declare would be refused by name
    ctx.act("set_choice", "tail", False)
    assert config.flags(S).get(api.TAIL_MOUNT_KEY) is None
    err = capsys.readouterr().err
    assert "Traceback" not in err, err


def test_the_mounting_the_shell_sends_is_one_the_engine_takes():
    """One string, two owners: a shell constant that drifted off
    ``api.TAIL_MOUNTS`` would be refused at build time on every run."""
    from aerobo import api
    from gui.v3 import config

    assert config.TAIL_MOUNT in api.TAIL_MOUNTS
    assert api.TAIL_MOUNTS[config.TAIL_MOUNT] is True     # pushes DOWN


def test_a_cg_the_down_mounting_cannot_trim_says_so_out_loud():
    """The price of the rule, and the card paying it.

    Mounted upside down, a surface asked to LIFT is trimmed against its own
    camber, and far enough aft the incidence leaves the polar: past about
    0.8 m of CG on the published tail the mid-box design stops evaluating.
    The margin read-out is three numbers off ``session.pitch_stability``, so
    it simply vanished there — a card that goes quiet at exactly the number
    that silenced it. The sentence that replaces it quotes the SOLVER's own
    reason, and the numbers come back when the CG does.
    """
    from gui.v3 import session
    from gui.v3.app import assemble

    ctx = assemble()
    ctx.act("accept_mission")
    ctx.act("set_choice", "tail", True)
    S = ctx.S
    assert session.pitch_stability(S) is not None
    assert session.stability_unevaluable(S) is None

    ctx.act("set_choice", "tail_cg_m", 0.9)
    assert session.pitch_stability(S) is None
    why = session.stability_unevaluable(S)
    assert why and "polar" in why

    ctx.render("wing", "type")
    texts = [getattr(e, "text", "") or ""
             for e in ctx.views[("wing", "type")].descendants()]
    assert any("No margin to report" in t and why in t for t in texts)

    ctx.act("set_choice", "tail_cg_m", 0.7)         # forward again
    assert session.pitch_stability(S) is not None
    assert session.stability_unevaluable(S) is None


def test_a_family_with_no_cg_still_says_nothing():
    """The other None: the wing-alone and tandem problems carry no pitch
    balance, so there is no margin AND no failure — the card must not grow a
    warning where it used to be silent."""
    from gui.v3 import session
    from gui.v3.app import assemble

    ctx = assemble()
    ctx.act("accept_mission")
    assert session.pitch_stability(ctx.S) is None
    assert session.stability_unevaluable(ctx.S) is None
