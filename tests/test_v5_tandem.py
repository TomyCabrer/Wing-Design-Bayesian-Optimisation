"""V5 item 5 — a tandem reaches V4 at all.

Before this, ``build_flight_model`` raised

    ValueError: this report carries no planform (no span/area and no
    spanwise chord array), so there is nothing to fly

on every tandem, because the family's report carries its two wings under
``geometry["surfaces"]`` as ``front``/``rear`` and no ``tail`` block, while
the rebuild looked for a surface named ``wing`` and fell through to the
top-level scalars. So stages 5 and 6 had never flown the family.

What it flies now, measured at the box centre (V = 45 m/s):

    112 panels (60 front + 40 rear + 12 fin)   S_ref 20.0 m2 (the PAIR's)
    CL_alpha +4.4348   x_np 2.1429   with the rear wing at dx = 5.0 m
"""
from __future__ import annotations

import numpy as np
import pytest

from aerobo import api
from aerobo.flightmodel import build_flight_model


def _report(problem: str = "tandem", flags: dict | None = None,
            x=None) -> dict:
    flags = flags or {}
    cfg = api.RunConfig(problem_name=problem, budget=4, seed=0, flags=flags)
    built = api.PROBLEM_SPECS[problem].build({}, flags, None)
    return api.design_report(
        cfg, built.bounds.mean(axis=1) if x is None else x)


def _asymmetric() -> dict:
    """A tandem whose two wings are NOT the same wing.

    At the box centre both tapers are 0.6 and the area split is 0.5, so the
    pair is symmetric and a rebuild that gave the rear the front's planform
    would be indistinguishable from a correct one. Everything that has to
    tell the two surfaces apart is measured here instead.
    """
    built = api.PROBLEM_SPECS["tandem"].build({}, {}, None)
    labels = list(built.param_labels)
    x = built.bounds.mean(axis=1).copy()
    x[labels.index("taper_front")] = 0.25
    x[labels.index("taper_rear")] = 0.95
    x[labels.index("area_split_front")] = 0.65
    return _report(x=x)


def test_a_tandem_can_be_flown_at_all():
    fm = build_flight_model(_report(), V=45.0)
    assert fm.deck.CL_alpha > 0.0


def test_the_rear_wing_is_IN_the_lattice_not_merely_claimed():
    """Panel count, not a note. A second surface that is only described is
    a second surface that contributes nothing."""
    rep = _report()
    fm = build_flight_model(rep, V=45.0)
    m = fm.model
    assert m.second is not None
    assert int(m.is_second.sum()) == 40
    # ...and it is AT THE STAGGER: the rear panels' station is dx
    dx = float(rep["breakdown"]["dx"])
    assert float(m.x[m.is_second][0]) == pytest.approx(dx, rel=1e-12)
    assert float(m.z[m.is_second].mean()) == pytest.approx(
        float(rep["breakdown"]["dz"]), rel=1e-9)


def test_the_neutral_point_lands_between_the_two_wings():
    """A pair's aerodynamic centre cannot be ahead of the front wing or
    behind the rear one; that it does says both surfaces carry load."""
    rep = _report()
    dx = float(rep["breakdown"]["dx"])
    x_np = build_flight_model(rep, V=45.0).model.neutral_point()
    assert 0.0 < x_np < dx


def test_the_pair_is_referenced_to_its_TOTAL_area():
    """``tandem.py`` quotes its coefficients on ``Sref``.

    Referencing the rebuild to the front wing alone would read CL_alpha
    about twice high, which is the kind of error that looks like a plausible
    aeroplane.
    """
    rep = _asymmetric()
    fm = build_flight_model(rep, V=45.0)
    assert fm.model.S == pytest.approx(float(rep["breakdown"]["Sref"]),
                                       rel=1e-12)
    assert fm.model.S > 1.4 * float(rep["breakdown"]["S_front"])


def test_a_rear_wing_of_a_DIFFERENT_SPAN_is_rebuilt_at_that_span():
    """``b_rear_m`` makes the pair's two spans genuinely different.

    With both wings at 10 m the front's span can be read off the report's
    top-level ``b`` and the rear's off its own surface block, so nothing
    distinguishes a correct rebuild from one that guesses. Here they cannot
    both be right by accident.
    """
    rep = _report(flags={"b_rear_m": 7.0})
    bd = rep["breakdown"]
    assert bd["b_rear"] == pytest.approx(7.0)
    assert bd["b_front"] != pytest.approx(7.0)
    m = build_flight_model(rep, V=45.0).model
    rear, front = m.is_second, ~(m.is_second | m.is_vertical)
    assert np.ptp(m.y[rear]) == pytest.approx(7.0, rel=5e-3)
    assert np.ptp(m.y[front]) == pytest.approx(float(bd["b_front"]),
                                              rel=5e-3)


def test_the_stated_span_beats_anything_the_report_says_elsewhere():
    """The hint is the contract, not a fallback.

    A tandem report writes the rear wing's span on its own surface block and
    the FRONT's under the top-level ``b`` — an arrangement that happens to
    make both readable today. It is not guaranteed: the moment a top-level
    ``b`` means the pair's widest span, a rebuild that trusts it gives the
    narrower wing the wider one's span. The breakdown states each surface's
    span outright, so that is what wins, and this proves it does by making
    every other source wrong.
    """
    rep = _report(flags={"b_rear_m": 7.0})
    rep["geometry"]["b"] = 99.0                       # a poisoned pair span
    for surf in rep["geometry"]["surfaces"]:
        surf.pop("b", None)                           # ...and no own scalars
    m = build_flight_model(rep, V=45.0).model
    assert np.ptp(m.y[m.is_second]) == pytest.approx(7.0, rel=5e-3)
    assert np.ptp(m.y[~(m.is_second | m.is_vertical)]) == pytest.approx(
        float(rep["breakdown"]["b_front"]), rel=5e-3)


def test_the_rear_wing_carries_its_own_area_not_the_front_ones():
    """The hint has to reach the rebuild, on a pair that is not symmetric.

    The AREA is what a hint carries and the span is what it decides, so this
    reads the rebuilt rear surface's own reference area back out of its
    chord distribution.
    """
    rep = _asymmetric()
    bd = rep["breakdown"]
    fm = build_flight_model(rep, V=45.0)
    m = fm.model
    rear = m.is_second
    S_rear = float(np.trapezoid(m.c[rear], m.y[rear]))
    assert S_rear == pytest.approx(float(bd["S_rear"]), rel=2e-3)
    assert S_rear != pytest.approx(float(bd["S_front"]), rel=0.05)


def test_each_wing_is_rebuilt_from_its_OWN_reported_chord():
    """The rear must not inherit the front's planform.

    Checked through the rebuilt lattice's chord at the two stations: the
    reported ``chord`` arrays are what the solver flew, and the rebuild has
    to reproduce each surface's own to the RMS the fidelity row quotes.
    """
    rep = _asymmetric()
    assert rep["breakdown"]["S_front"] != rep["breakdown"]["S_rear"]
    fm = build_flight_model(rep, V=45.0)
    m = fm.model
    surf = {s["name"]: s for s in rep["geometry"]["surfaces"]}
    for name, mask in (("front", ~(m.is_second | m.is_vertical)),
                       ("rear", m.is_second)):
        y_rep = np.asarray(surf[name]["y"], float)
        c_rep = np.asarray(surf[name]["chord"], float)
        # sample the rebuilt planform at the reported stations
        c_got = np.interp(y_rep, m.y[mask], m.c[mask])
        rms = float(np.sqrt(np.mean((c_got - c_rep) ** 2)) / c_rep.mean())
        assert rms < 0.02, f"{name} planform off by {100 * rms:.2f} % RMS"
    assert fm.fidelity.rows[0][0] == "planform c(y)"
    assert fm.fidelity.rows[0][2] < 0.02


def test_a_tandem_gets_the_same_single_author_fin_everyone_else_gets():
    """Its arm is the STAGGER — the only separation a pair has."""
    rep = _report()
    blk = rep["geometry"]["fin"]
    assert blk["l_t_m"] == pytest.approx(float(rep["breakdown"]["dx"]))
    assert blk["S"] == pytest.approx(
        0.04 * float(rep["breakdown"]["b"])
        * float(rep["breakdown"]["Sref"]) / float(rep["breakdown"]["dx"]),
        rel=1e-12)
    fm = build_flight_model(rep, V=45.0)
    assert fm.model.is_vertical.any()
    assert float(fm.model.c[fm.model.is_vertical][0]) == pytest.approx(
        blk["chord_m"], rel=1e-12)


def test_the_missing_decalage_is_declared_rather_than_silently_zero():
    """No tandem report writes its twist laws out, and the pair's decalage
    lives in them. That is a real hole and the model must say so."""
    fm = build_flight_model(_report(), V=45.0)
    assert any("TANDEM" in a and "DECALAGE" in a for a in fm.assumptions)


def test_the_conventional_family_still_reads_the_wing_surface():
    """The named-surface change must not redirect anyone else's planform."""
    rep = _report("tail")
    fm = build_flight_model(rep, V=45.0)
    assert fm.model.second is None
    assert fm.fidelity.rows[0][2] < 1e-6      # the tail family fits exactly
