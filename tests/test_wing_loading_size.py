"""Area from the mission's W/S, span from the optimiser.

The size modifier that shipped frees the span AND the area, which quietly
hands the search the mission's own question: W/S sets the trim lift
coefficient (CL = W/(qS)), so an optimiser that moves the area is choosing an
operating point, not just a wing.

This is the other mode (``api`` modifier ``size_ws``, ``sizing.SIZE_MODE_WS``):

* the AREA follows the chosen wing loading through the same weight loop —
  ``S = W_total/(W/S)``, closed as a fixed point because the wing's own
  statistical weight depends on the area being solved for;
* the design vector therefore carries the SPAN alone, inside a band the user
  states (``span_min_m`` / ``span_max_m``);
* everything the free-planform mode costs is paid here too: the wing weighs
  what its size implies, the score is payload L/D = W_fixed/D, and the
  root-bending stress margin is a CONSTRAINT;
* and the trim lift coefficient is fixed by construction at ``(W/S)/q``,
  whatever span comes out — which is the whole point.

The two modes are mutually exclusive: a wing is sized one way.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aerobo import api, objective, sizing     # noqa: E402

WS_NAME = "trim wing + free span (W/S)"


def _built(name: str = WS_NAME, flags: dict | None = None):
    return api.PROBLEM_SPECS[name].build({}, flags or {}, None)


# ------------------------------------------------------------ the mode itself
def test_the_mode_is_a_value_not_a_bool():
    assert sizing.size_mode(False) == "off"
    assert sizing.size_mode(True) == sizing.SIZE_MODE_FREE
    assert sizing.size_mode("wing_loading") == sizing.SIZE_MODE_WS
    with pytest.raises(ValueError, match="unknown size mode"):
        sizing.size_mode("enormous")
    # every `if prob.size_free:` in the package still means "this is sized"
    assert bool(sizing.SIZE_MODE_WS)


def test_the_size_block_is_one_row_and_it_is_the_span():
    assert sizing.size_labels(True) == ("b_m", "S_m2")
    assert sizing.size_labels(sizing.SIZE_MODE_WS) == ("b_m",)
    assert sizing.n_size_rows(sizing.SIZE_MODE_WS) == 1

    box = sizing.with_size_bounds(np.zeros((3, 2)), sizing.SIZE_MODE_WS,
                                  b0=10.0, S0=10.0, span_bounds_m=(8.0, 15.0))
    assert box.shape == (4, 2)
    assert list(box[-1]) == [8.0, 15.0]


def test_the_area_closes_the_weight_loop():
    """S = W_total/(W/S) with W_total = W_fixed + W_wing(b, S): a fixed point,
    because a bigger wing weighs more and therefore needs more area."""
    q = 0.5 * 1.225 * 14.6 ** 2
    W_fixed = 0.5 * q * 10.0                    # the published CL x q x S
    ws = W_fixed / 10.0
    S = sizing.area_for_wing_loading(W_fixed_N=W_fixed, b=12.0,
                                     wing_loading_Pa=ws, taper=0.6, tc=0.12,
                                     q_Pa=q)
    st = sizing.sized_state(W_fixed_N=W_fixed, b=12.0, S=S, taper=0.6,
                            tc=0.12, q_Pa=q)
    assert st.W_total_N / S == pytest.approx(ws, rel=1e-9)
    # the wing's own weight is what makes S bigger than W_fixed/(W/S)
    assert S > W_fixed / ws
    assert st.W_wing_N > 0.0


def test_a_loading_that_cannot_be_closed_is_refused_not_returned():
    with pytest.raises(ValueError, match="wing loading must be > 0"):
        sizing.area_for_wing_loading(W_fixed_N=1000.0, b=10.0,
                                     wing_loading_Pa=0.0, taper=0.6,
                                     tc=0.12, q_Pa=100.0)


# --------------------------------------------------------------- the problem
def test_the_vector_carries_the_span_alone():
    built = _built()
    assert built.param_labels == ("taper", "twist_root_deg", "twist_tip_deg",
                                  "b_m")
    assert built.is_constrained            # the stress margin comes with it
    assert api.PROBLEM_SPECS[WS_NAME].constraint_labels == (
        "root-bending stress margin",)


def test_the_trim_lift_coefficient_is_the_loading_and_nothing_else():
    """Whatever span the optimiser picks, CL = (W/S)/q — that is what makes
    the wing loading the MISSION's answer."""
    built = _built()
    base = objective.Problem(mode="trim")
    q = 0.5 * base.rho * base.V ** 2
    for b_m in (8.0, 12.0, 20.0):
        out = built.evaluate(np.array([0.6, 0.0, -2.0, b_m]))
        assert out["feasible"], out["reason"]
        assert out["b_m"] == pytest.approx(b_m)
        assert out["CL"] == pytest.approx(base.CL_target, rel=1e-6)
        assert out["W_total_N"] / out["S_m2"] == pytest.approx(
            base.CL_target * q, rel=1e-6)


def test_a_chosen_loading_moves_the_area_and_the_lift_coefficient():
    built = _built(flags={"wing_loading_pa": 120.0})
    out = built.evaluate(np.array([0.6, 0.0, -2.0, 12.0]))
    q = 0.5 * objective.Problem().rho * objective.Problem().V ** 2
    assert out["W_total_N"] / out["S_m2"] == pytest.approx(120.0, rel=1e-6)
    assert out["CL"] == pytest.approx(120.0 / q, rel=1e-6)


def test_the_span_band_is_the_users():
    band = _built(flags={"span_min_m": 8.0, "span_max_m": 14.0})
    assert list(band.bounds[-1]) == [8.0, 14.0]
    # one end alone is a complete sentence
    top = _built(flags={"span_max_m": 14.0})
    assert top.bounds[-1][1] == 14.0
    assert top.bounds[-1][0] == pytest.approx(sizing.B_FRAC_BOUNDS[0] * 10.0)
    with pytest.raises(ValueError, match="span bounds"):
        _built(flags={"span_min_m": 15.0, "span_max_m": 9.0}).bounds


def test_the_score_is_payload_lod_and_the_margin_is_the_spar():
    built = _built()
    x = np.array([0.6, 0.0, -2.0, 12.0])
    f, g = built.callable(x)
    out = built.evaluate(x)
    assert f == pytest.approx(out["score"])
    assert g == pytest.approx(out["g_sigma"])
    # payload L/D, not L/D: the fixed weight over the drag
    q = 0.5 * objective.Problem().rho * objective.Problem().V ** 2
    D = q * out["S_m2"] * out["CD"]
    assert f == pytest.approx(out["W_fixed_N"] / D, rel=1e-9)


# -------------------------------------------------------------- the registry
def test_the_two_size_modes_are_alternatives():
    """Three of them now (the searched loading joined), and still
    alternatives: a wing is sized one way."""
    assert api.with_modifiers("trim wing", {"size", "size_ws"}) is None
    assert api.add_modifier("trim wing + free planform", "size_ws") is None
    assert api.add_modifier("trim wing + free span (W/S)", "size") is None
    assert api.add_modifier("trim wing + free span (W/S)",
                            "size_ws_free") is None
    assert set(api.SIZE_MODIFIERS) == {"size", "size_ws", "size_ws_free"}


def test_it_composes_with_the_other_modifiers_and_with_a_tip_device():
    for name in ("trim wing + free span (W/S) + free chord law",
                 "trim wing + free span (W/S) + free flight state",
                 "winglet_capped + free span (W/S)",
                 "winglet, blended (span-capped) + free span (W/S) "
                 "+ free chord law"):
        built = api.PROBLEM_SPECS[name].build({}, {}, None)
        assert "b_m" in built.param_labels
        assert "S_m2" not in built.param_labels
        out = built.evaluate(built.bounds.mean(axis=1))
        assert np.isfinite(out["score"]), (name, out["reason"])


def test_the_aircraft_with_a_tail_sizes_its_WING_from_the_loading():
    """The tail's own area stays a design variable: sizing the aircraft is
    not sizing its stabiliser."""
    built = api.PROBLEM_SPECS["tail + free span (W/S)"].build({}, {}, None)
    assert built.param_labels == ("taper", "twist_root_deg", "twist_tip_deg",
                                  "S_t_m2", "l_t_m", "b_m")
    out = built.evaluate(built.bounds.mean(axis=1))
    assert out["feasible"], out["reason"]
    assert out["W_total_N"] / out["S_m2"] == pytest.approx(
        objective.Problem().CL_target * 0.5 * objective.Problem().rho
        * objective.Problem().V ** 2, rel=1e-6)


def test_the_size_block_is_read_back_past_BOTH_chord_blocks():
    """A designed tail carries one chord law PER SURFACE, so the trailing
    block is 2 x chord_order — and the size rows sit ahead of ALL of it.

    Counting back by ``chord_order`` alone landed on the wing's own
    coefficients: the free-planform designed-tail-with-a-chord-law problems
    read (b, S) = (chord_k2, chord_k3), which the aspect-ratio band then
    refused, so every candidate of those eight problems scored the penalty.
    """
    name = "tail [designed tail] + free planform + free chord law"
    built = api.PROBLEM_SPECS[name].build({}, {}, None)
    labels = list(built.param_labels)
    x = built.bounds.mean(axis=1).copy()
    x[labels.index("b_m")], x[labels.index("S_m2")] = 11.0, 12.0
    out = built.evaluate(x)
    assert out["feasible"], out["reason"]
    assert out["b_m"] == pytest.approx(11.0)
    assert out["S_m2"] == pytest.approx(12.0)


def test_the_nonplanar_wing_and_tail_sizes_the_same_way():
    """The whole wing+tail family (tip devices, designed tail, its own tip
    device — every switch it carries) sizes from the loading too."""
    wt = [n for n in api.problem_names()
          if "free span (W/S)" in n and api.base_of(n) in api.WING_TAIL_VARIANTS]
    assert len(wt) > 300
    for name in wt[:12]:
        built = api.PROBLEM_SPECS[name].build({}, {}, None)
        assert "b_m" in built.param_labels
        assert "S_m2" not in built.param_labels
        out = built.evaluate(built.bounds.mean(axis=1))
        assert np.isfinite(out["score"]), (name, out["reason"])


def test_a_family_without_the_mode_says_so_rather_than_offering_it():
    """Declared per FAMILY (api._WING_LOADING_FAMILIES): a family whose
    evaluate() has not been wired for a size block must not appear with a
    variant whose vector and whose box would disagree.

    The TANDEM is deliberately not in this list any more: its block is TWO
    span rows, one per wing (:mod:`tests.test_tandem_two_spans`).
    """
    for name in ("hydrofoil", "car rear wing",
                 "free planform (aircraft)", "wing+airfoil (coupled)"):
        assert api.with_modifiers(name, {"size_ws"}) is None
    ws_names = [n for n in api.problem_names() if "free span (W/S)" in n]
    assert len(ws_names) == 444       # +4 with the tandem's t/c twin,
    #                                   +8 with the two coupled-library
    #                                   winglet families (4 names each: the
    #                                   W/S mode x flight x chord)
    for n in ws_names:
        assert api.base_of(n) in api._WING_LOADING_FAMILIES


def test_the_size_flags_are_replaced_by_the_loading_flags():
    """Two answers to one question: a family whose size is in the vector may
    not also offer 'choose a span and an area'."""
    spec = api.PROBLEM_SPECS[WS_NAME]
    assert not set(api.PLANFORM_KEYS) & set(spec.flags)
    assert set(api.WING_LOADING_KEYS) <= set(spec.flags)


def test_nothing_about_the_published_problems_moved():
    plain = api.PROBLEM_SPECS["trim wing"].build({}, {}, None)
    assert plain.problem.size_free is False
    assert plain.problem.wing_loading_Pa is None
    assert plain.param_labels == ("taper", "twist_root_deg", "twist_tip_deg")
    free = api.PROBLEM_SPECS["trim wing + free planform"].build({}, {}, None)
    assert free.param_labels == ("taper", "twist_root_deg", "twist_tip_deg",
                                 "b_m", "S_m2")


# ------------------------------------------------------------------- the shell
def test_the_builder_offers_it_and_derives_it():
    from gui import nice_app as v1

    ch = v1.start_choices(medium="air")
    assert "wing_loading" in v1.PLANFORM_CHOICE_LABELS
    assert v1.option_available(ch, "planform", "wing_loading")
    ch["planform"] = "wing_loading"
    name, notes = v1.derive_problem(ch)
    assert name == "trim wing + free span (W/S) + free chord law"
    assert notes == []
    assert v1.active_modifiers(ch) == ["size_ws", "chord"]


def test_the_mission_s_loading_is_what_the_run_flies(capsys):
    from gui.v3 import config, session
    from gui.v3.app import assemble

    ctx = assemble("air")
    S = ctx.S
    ctx.act("set_choice", "planform", "wing_loading")
    flags = config.cfg_dict(S)["flags"]
    assert flags["wing_loading_pa"] == pytest.approx(session.wing_loading(S))

    # the span limit is typed in the DESIGN BOX, as the b_m row (V3 asks it
    # nowhere else — gui/v3/stages/wing.py), and it reaches the run as an
    # ordinary bound override
    ctx.act("set_bound", "b_m", 1, 14.0)
    cfg = config.build_cfg(S)
    built = api.PROBLEM_SPECS[cfg.problem_name].build(
        cfg.mission_kwargs, cfg.flags, cfg.bounds_overrides)
    assert built.bounds[3][1] == 14.0
    out = built.evaluate(built.bounds.mean(axis=1))
    assert out["W_total_N"] / out["S_m2"] == pytest.approx(
        session.wing_loading(S), rel=1e-6)

    err = capsys.readouterr().err
    assert "Traceback" not in err, err
