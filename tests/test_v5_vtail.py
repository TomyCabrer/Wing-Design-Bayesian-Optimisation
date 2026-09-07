"""V5 item 2 — a V-tail is flown as a V-tail.

Before this, ``build_flight_model`` drew a V-tail as a FLAT surface of its
panel area and then bolted on a fin the design does not have. Measured on the
`tail` family at its box centre with a 35 deg cant:

    Cn_beta  +0.13930 -> +0.06530   (-53 %: a fin's worth of yaw stiffness
                                     the aeroplane never had)
    CY_beta  -0.25491 -> -0.12234
    SM vs the SCORED design: +17.30 % -> +10.23 %
    panels                       92 -> 80

The residual +10.23 % is not a bug: it is the reduced-order tail model
(``tail.lifting_area``'s ``S cos^2 G``) disagreeing with the lattice, and the
lattice says a canted surface is MORE effective than the closed form. That
disagreement is asserted here rather than tuned away.
"""
from __future__ import annotations

import numpy as np
import pytest

from aerobo import api, dynamics as dyn
from aerobo.flightmodel import build_flight_model
from aerobo.geometry import Wing
from aerobo.tail import lifting_area
from aerobo.vlm import VLM, TailSurface

WING = dict(b=10.0, S=10.0, taper=0.6)
V_TAIL_FLAGS = {"tail_type": "v_tail", "dihedral_deg": 35.0}


def _model(gamma_deg: float, S: float = 1.75) -> VLM:
    return VLM(Wing(**WING), N=40, V=30.0,
               tail=TailSurface(S=S, x=5.5, z=0.5, AR=4.0, N=20,
                                dihedral_deg=gamma_deg))


def _deck(gamma_deg: float):
    m = _model(gamma_deg)
    return dyn.deck(m, x_cg=0.05, mac=1.02, b=WING["b"],
                    alpha=np.deg2rad(4.0))


def _report(flags: dict) -> dict:
    cfg = api.RunConfig(problem_name="tail", budget=4, seed=0, flags=flags)
    built = api.PROBLEM_SPECS["tail"].build({}, flags, None)
    return api.design_report(cfg, built.bounds.mean(axis=1))


# ------------------------------------------------------- the lattice surface

def test_a_flat_tail_has_no_yaw_stiffness_and_a_canted_one_does():
    """With no fin anywhere, the cant is the ONLY source of ``Cn_beta``."""
    flat = _deck(0.0)
    assert abs(flat.Cn_beta) < 1e-12
    assert abs(flat.CY_beta) < 1e-12

    cant = _deck(35.0)
    assert cant.Cn_beta > 0.0          # directionally STABLE
    assert cant.CY_beta < 0.0          # and the side force opposes the slip


def test_yaw_stiffness_goes_as_sin_squared_of_the_cant():
    """The closed form a V-tail's fin-equivalent area is derived from.

    A canted panel presents ``S sin(G)`` of vertical projection and returns
    ``sin(G)`` of its normal force sideways, so its fin-equivalent area — and
    with it ``Cn_beta`` at a fixed arm — goes as ``sin^2 G``. Asserted as a
    RATIO held constant across the range, which no single-angle test can do.
    """
    ks = [_deck(g).Cn_beta / np.sin(np.deg2rad(g)) ** 2
          for g in (15.0, 30.0, 45.0)]
    assert min(ks) > 0.0
    assert max(ks) / min(ks) < 1.05, f"sin^2 scaling broken: {ks}"


def test_zero_cant_is_the_flat_tail_bit_for_bit():
    a = VLM(Wing(**WING), N=40, V=30.0,
            tail=TailSurface(S=1.75, x=5.5, z=0.5, AR=4.0, N=20))
    b = _model(0.0)
    assert np.array_equal(a.A3, b.A3) and np.array_equal(a.st3, b.st3)
    # the ARRAYS are exact; the force is float-identical (BLAS picks its
    # own reduction order — see test_v5_sweep.py for the measurement)
    assert a.solve(0.07, i_t=0.01).CL == pytest.approx(
        b.solve(0.07, i_t=0.01).CL, rel=1e-14)


def test_the_cant_cannot_be_stated_twice():
    """One question, one place — enforced at construction, not documented.

    A designed tail is a Wing, and a Wing has a dihedral of its own, so the
    cant has two possible homes and applying both would fly the surface at
    2 Gamma.
    """
    with pytest.raises(ValueError, match="stated twice"):
        TailSurface(S=1.75, x=5.5, z=0.5, wing=Wing(b=3.0, S=1.75,
                                                    dihedral_deg=20.0),
                    dihedral_deg=35.0)
    # ...and either one ALONE is fine
    TailSurface(S=1.75, x=5.5, z=0.5, dihedral_deg=35.0)
    TailSurface(S=1.75, x=5.5, z=0.5,
                wing=Wing(b=3.0, S=1.75, dihedral_deg=20.0))


def test_a_cant_stated_on_a_DESIGNED_tail_is_applied_exactly_once():
    """The other home the cant has, and the reason for the refusal above.

    A designed tail is a Wing, so its cant can live there — and the sub-VLM
    that panelises it applies it. The parent must NOT rotate again. Measured
    against the surface built the other way round: the two must agree, which
    they cannot if either path double-applies.
    """
    tw = Wing(b=2.65, S=1.75, taper=1.0, dihedral_deg=30.0)
    on_wing = VLM(Wing(**WING), N=40, V=30.0,
                  tail=TailSurface(S=1.75, x=5.5, z=0.5, N=20, wing=tw))
    on_surf = VLM(Wing(**WING), N=40, V=30.0,
                  tail=TailSurface(S=1.75, x=5.5, z=0.5, N=20,
                                   wing=Wing(b=2.65, S=1.75, taper=1.0),
                                   dihedral_deg=30.0))
    zt_a = on_wing.z[on_wing.is_tail]
    zt_b = on_surf.z[on_surf.is_tail]
    assert np.ptp(zt_a) == pytest.approx(np.ptp(zt_b), rel=1e-9)

    # ...and it is ONE cant, not two. The z extent goes as sin(Gamma) at a
    # fixed panel span, so the 15/30 pair fixes the angle without needing to
    # know where the cosine stations fall: applying the cant twice would make
    # this ratio sin(60)/sin(30) = 1.732 instead of sin(30)/sin(15) = 1.932.
    half = VLM(Wing(**WING), N=40, V=30.0,
               tail=TailSurface(S=1.75, x=5.5, z=0.5, N=20,
                                wing=Wing(b=2.65, S=1.75, taper=1.0,
                                          dihedral_deg=15.0)))
    ratio = np.ptp(zt_a) / np.ptp(half.z[half.is_tail])
    want = np.sin(np.deg2rad(30.0)) / np.sin(np.deg2rad(15.0))
    assert ratio == pytest.approx(want, rel=1e-3), (
        f"z extent grew by {ratio:.4f}x from 15 to 30 deg; once is "
        f"{want:.4f}x and twice would be "
        f"{np.sin(np.deg2rad(60.0)) / np.sin(np.deg2rad(30.0)):.4f}x")


def test_the_lattice_beats_the_closed_form_and_the_gap_grows_with_cant():
    """``S cos^2 G`` is a strip argument; the canted surface is NONPLANAR.

    Its own wake is out of plane, so it recovers some of what the projection
    takes away — the same reason a winglet works. The closed form is
    therefore a LOWER bound on pitch effectiveness, and the excess grows.
    This is the estimate-versus-flown disagreement the V-tail's fidelity row
    reports; pinning it stops a future session "fixing" the lattice to agree
    with the reduced-order model.
    """
    def tail_alpha_slope(g: float) -> float:
        m, e = _model(g), 1e-5
        def CLt(al):
            G = m._gamma(al, 0.0)
            return float(np.sum(2.0 * G * m.lvec[:, 1] * m.is_tail)
                         / (m.V * m.S))
        return (CLt(e) - CLt(-e)) / (2.0 * e)

    base = tail_alpha_slope(0.0)
    excess = []
    for g in (15.0, 30.0, 45.0):
        got = tail_alpha_slope(g) / base
        closed = lifting_area("v_tail", g, 1.0)
        assert got > closed, f"at {g} deg the lattice fell below cos^2"
        excess.append(got / closed - 1.0)
    assert excess == sorted(excess), f"excess is not monotone: {excess}"
    assert 0.15 < excess[-1] < 0.30      # +21.9 % at 45 deg, as measured


# -------------------------------------------------------- through the bridge

def test_a_v_tail_design_is_flown_with_no_fin_and_no_rudder():
    fm = build_flight_model(_report(V_TAIL_FLAGS), V=45.0)
    assert not fm.model.is_vertical.any(), "a V-tail was given a fin"
    assert [c.name for c in fm.controls] == ["aileron", "elevator"]
    assert any("V-tail" in a and "no separate fin" in a
               for a in fm.assumptions)


def test_the_v_tail_still_has_yaw_stiffness_without_that_fin():
    """Deleting the invented fin must not leave the aeroplane directionless."""
    d = build_flight_model(_report(V_TAIL_FLAGS), V=45.0).deck
    assert d.Cn_beta > 0.02
    assert d.CY_beta < 0.0
    # ...and it is roughly HALF the conventional design's, which is the whole
    # measured finding: V4 was crediting a V-tail with a fin's worth of yaw.
    conv = build_flight_model(_report({}), V=45.0).deck
    assert 0.35 < d.Cn_beta / conv.Cn_beta < 0.65


def test_the_panel_span_reaches_the_lattice_not_the_projected_span():
    """``b_t`` in the report is already ``sqrt(AR * S cos^2 G)``.

    Handing that to a surface that then applies the cant itself shrinks the
    tail twice. Measured as the PROJECTION coming back out equal to what the
    report states.
    """
    rep = _report(V_TAIL_FLAGS)
    t = rep["geometry"]["tail"]
    fm = build_flight_model(rep, V=45.0)
    y_tail = fm.model.y[fm.model.is_tail]
    assert y_tail.max() - y_tail.min() == pytest.approx(
        float(t["b_t"]), rel=0.02)


def test_a_conventional_design_still_gets_its_fin():
    """The V-tail branch must not delete everyone else's vertical surface."""
    fm = build_flight_model(_report({}), V=45.0)
    assert fm.model.is_vertical.any()
    assert "rudder" in [c.name for c in fm.controls]
