"""A T-tail's fin has to REACH the tailplane it carries.

Session 66 fixed the live half of the reported bug — "when T tail is
selected it creates a conventional one but higher" — in
``api.design_report``: the fin's ROOT was taken as the tailplane's height for
every layout, so a T-tail's fin started at its own tailplane and stood
entirely above it. ``tests/test_v5_fin.py`` pins that.

This file pins the other end of the same surface, which was closed by
coincidence rather than by construction. Two formulas set it:

* ``tail.tail_height('t_tail', ...)`` places the TAILPLANE at
  ``max(DZ_FRAC * b, sqrt(AR_VT * S_vt))`` — the second term is the fin's
  span, the first is the measured kernel-regularisation clearance;
* ``fin.size_fin`` builds the FIN at ``sqrt(AR_VT * S_vt)`` and never hears
  about that floor.

They agree exactly while the floor is slack, and only while. Where it binds
the tailplane is placed above a fin too short to touch it — measured below at
b = 7 m, S = 1.25 m2, l_t = 8 m: a 0.256 m fin under a tailplane at 0.350 m,
a gap of 37 % of the fin's own span.

**Honest reachability label.** That corner is NOT reachable through any
shipped family at its default box: every problem spec carrying ``tail_type``
resolves to a single (b, S, l_t) box, and no point in it that also satisfies
``api.PLANFORM_AR_LIMITS`` binds the floor (asserted below, so the label
cannot rot). This is therefore a LATENT defect closed by construction, not a
user-visible fix — the two heights now agree because one is derived from the
other, instead of agreeing because the numbers happened to.
"""
from __future__ import annotations

import numpy as np
import pytest

from aerobo import api, fin as finmod, tail as tailmod

#: the measured binding corner: admissible aspect ratio, floor wins
BINDING = dict(b=7.0, S=1.25, l_t=8.0)
#: the shipped ``tail`` family's own numbers, where the floor is slack
SLACK = dict(b=10.0, S=10.0, l_t=5.5)


def _fin(where: dict, *, reach: bool):
    dz = tailmod.tail_height("t_tail", where["l_t"], where["b"], where["S"])
    return dz, finmod.size_fin(
        b=where["b"], S=where["S"], l_t=where["l_t"], tail_type="t_tail",
        z_root=0.0, min_height=(dz if reach else None))


def test_a_slack_floor_is_the_identity_map_field_for_field():
    """Every design that flies today must be built by the same numbers.

    Not ``approx``: the branch is a strict ``>``, so a fin already tall
    enough takes the untouched path and the object is the same object.
    """
    dz, reached = _fin(SLACK, reach=True)
    _, plain = _fin(SLACK, reach=False)
    assert reached == plain, "a slack floor rebuilt a fin it should not touch"
    assert plain.AR == finmod.AR_VT_DEFAULT
    assert plain.height == pytest.approx(dz, rel=1e-12)


def test_without_the_layouts_height_the_fin_stops_short_of_the_tailplane():
    """The defect, stated as the measurement that finds it."""
    dz, short = _fin(BINDING, reach=False)
    assert short.height < dz, "this corner no longer binds — reselect it"
    assert dz - short.height == pytest.approx(0.09382623085101005, rel=1e-9)
    assert (dz - short.height) / short.height > 0.36


def test_the_fin_spans_body_to_tailplane_exactly():
    dz, reached = _fin(BINDING, reach=True)
    assert reached.z_root == 0.0
    assert reached.z_root + reached.height == pytest.approx(dz, rel=1e-12)


def test_the_volume_coefficient_still_owns_the_AREA():
    """The layout sets the SPAN; the drag book keeps the area, so the yaw
    stiffness and the parasite drag are the ones the design was priced at.
    The aspect ratio is then REALISED from the two, not assumed."""
    dz, reached = _fin(BINDING, reach=True)
    _, plain = _fin(BINDING, reach=False)
    assert reached.S == plain.S
    assert reached.S == pytest.approx(
        finmod.V_V_DEFAULT * BINDING["b"] * BINDING["S"] / BINDING["l_t"],
        rel=1e-12)
    assert reached.height * reached.chord == pytest.approx(reached.S,
                                                           rel=1e-12)
    assert reached.AR == pytest.approx(dz * dz / reached.S, rel=1e-12)
    assert reached.AR > finmod.AR_VT_DEFAULT


def test_a_ventral_fin_keeps_the_sign_and_the_area():
    """``min_height`` is a SPAN, and the sign lives on the height."""
    dz = tailmod.tail_height("t_tail", BINDING["l_t"], BINDING["b"],
                             BINDING["S"])
    g = finmod.size_fin(b=BINDING["b"], S=BINDING["S"], l_t=BINDING["l_t"],
                        tail_type="t_tail", ventral=True, min_height=dz)
    assert g.height == pytest.approx(-dz, rel=1e-12)
    assert g.S > 0.0


def test_the_api_hands_the_layouts_height_to_the_law():
    """The wiring, not the law: a T-tail report's fin must span its own dz.

    This holds at the default box (where the floor is slack) too, because
    the report is built through the same argument — which is the point:
    there is no longer a corner where it stops holding.
    """
    cfg = api.RunConfig(problem_name="tail", budget=4, seed=0,
                        flags={"tail_type": "t_tail"})
    built = api.PROBLEM_SPECS["tail"].build({}, {"tail_type": "t_tail"}, None)
    rep = api.design_report(cfg, built.bounds.mean(axis=1))
    f, t = rep["geometry"]["fin"], rep["geometry"]["tail"]
    assert f["z_root_m"] == pytest.approx(0.0, abs=1e-12)
    assert f["z_root_m"] + f["height_m"] == pytest.approx(float(t["dz_m"]),
                                                          rel=1e-12)
    assert f["height_m"] * f["chord_m"] == pytest.approx(f["S"], rel=1e-9)


def test_the_binding_corner_is_out_of_reach_of_every_shipped_box():
    """The honesty label, asserted so it cannot go stale.

    If a family ever opens a box that reaches this corner, this test fails
    and the docstring above ("latent, not user-visible") has to be rewritten
    — which is the only way that claim stays true.
    """
    lo_ar, hi_ar = api.PLANFORM_AR_LIMITS
    boxes, reachable = set(), []
    for name, sp in api.PROBLEM_SPECS.items():
        labels = list(sp.param_labels)
        if "tail_type" not in sp.flags:
            continue
        if not {"b_m", "S_m2", "l_t_m"} <= set(labels):
            continue
        try:
            built = sp.build({}, {"tail_type": "t_tail"}, None)
        except Exception:
            continue
        lo, hi = np.array(built.bounds).T
        idx = [labels.index(k) for k in ("b_m", "S_m2", "l_t_m")]
        box = (tuple(lo[idx]), tuple(hi[idx]))
        if box in boxes:
            continue
        boxes.add(box)
        (b_lo, s_lo, t_lo), (b_hi, s_hi, t_hi) = box
        for b in np.linspace(b_lo, b_hi, 21):
            for s in np.linspace(s_lo, s_hi, 21):
                if not (lo_ar <= b * b / s <= hi_ar):
                    continue
                for l_t in np.linspace(t_lo, t_hi, 9):
                    if tailmod.DZ_FRAC * b > np.sqrt(
                            finmod.AR_VT_DEFAULT * finmod.V_V_DEFAULT
                            * b * s / l_t):
                        reachable.append((name, b, s, l_t))
    assert boxes, "no family carries a tail_type any more — read this file"
    assert not reachable, (
        f"a shipped box now reaches the binding corner ({reachable[:3]}): "
        f"the T-tail floor is a LIVE defect, not a latent one, and this "
        f"file's docstring says otherwise")
