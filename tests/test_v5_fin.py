"""V5 item 4 — the fin has ONE author.

Two different fins used to exist for the same aeroplane:

* ``tail.vtail_cd0`` sized one by volume coefficient to charge its drag —
  and that charge was a switch that defaulted OFF, so the shipped design
  paid ``cd0_fin = 0.0`` for a fin it flew. It is unconditional now;
* ``flightmodel.build_flight_model`` sized another (12 % of span by 0.65 mac)
  and that one produced 100 % of the yaw stiffness in flight.

Measured on the `tail` family at its box centre, moving the flight rebuild
onto the drag book's fin:

    Cn_beta  +0.136961 -> +0.113230   (-17.3 %)
    Cl_beta  -0.020102 -> -0.014716
    CY_beta  -0.250531 -> -0.207062

So V4 was flying a fin 17 % stronger in yaw than the one the design was
priced against. Both numbers were defensible; having both was not.
"""
from __future__ import annotations

import pytest

from aerobo import api, fin as finmod
from aerobo.flightmodel import build_flight_model
from aerobo.tail import AR_VT_DEFAULT, V_V_DEFAULT, vtail_cd0


def _report(flags: dict) -> dict:
    cfg = api.RunConfig(problem_name="tail", budget=4, seed=0, flags=flags)
    built = api.PROBLEM_SPECS["tail"].build({}, flags, None)
    return api.design_report(cfg, built.bounds.mean(axis=1))


# ------------------------------------------------------------- the one law

def test_the_volume_coefficient_law_is_what_it_says_it_is():
    """``S_vt = V_v b S / l_t``, written out independently."""
    g = finmod.size_fin(b=12.0, S=14.0, l_t=6.0, V_v=0.05, AR=1.8)
    assert g.S == pytest.approx(0.05 * 12.0 * 14.0 / 6.0, rel=1e-12)
    assert g.AR == 1.8
    assert g.height * g.chord == pytest.approx(g.S, rel=1e-12)
    assert g.height / g.chord == pytest.approx(1.8, rel=1e-12)
    assert g.x_qc == 6.0
    assert g.x_le == pytest.approx(6.0 - 0.25 * g.chord, rel=1e-12)


def test_the_fin_area_falls_as_the_arm_grows():
    """A longer arm needs less fin — the whole point of a volume coefficient."""
    areas = [finmod.size_fin(b=10.0, S=10.0, l_t=lt).S
             for lt in (3.0, 5.5, 9.0)]
    assert areas == sorted(areas, reverse=True)
    assert areas[0] / areas[2] == pytest.approx(9.0 / 3.0, rel=1e-12)


def test_a_v_tail_gets_no_fin_from_the_law_itself():
    """The rule lives in one place, not re-stated at every call site."""
    assert finmod.size_fin(b=10.0, S=10.0, l_t=5.5,
                           tail_type="v_tail") is None
    assert finmod.size_fin(b=10.0, S=10.0, l_t=5.5) is not None


def test_an_arm_no_fin_can_be_sized_against_is_refused_not_drawn():
    with pytest.raises(ValueError, match="not an? |ARM|arm"):
        finmod.size_fin(b=10.0, S=10.0, l_t=0.0)


def test_ventral_is_a_sign_on_the_height_and_never_on_the_area():
    v = finmod.size_fin(b=10.0, S=10.0, l_t=5.5, ventral=True)
    d = finmod.size_fin(b=10.0, S=10.0, l_t=5.5)
    assert v.height < 0.0 < d.height
    assert v.S == d.S == pytest.approx(abs(v.height) * v.chord, rel=1e-12)


def test_the_drag_book_and_the_law_size_the_SAME_fin():
    """``vtail_cd0`` must be a consumer of ``size_fin``, not a second author.

    Asserted through the drag it returns: rebuilding the same chord by the
    law and pushing it through the same Raymer build-up has to reproduce
    ``vtail_cd0`` exactly, which it cannot do if the two disagree about the
    chord by so much as an ulp (the Reynolds number rides on it).
    """
    from aerobo.drag import skin_friction_cf, wing_form_factor
    from aerobo.objective import MU_SL, RHO_SL

    b, S, l_t, V = 10.0, 10.0, 5.5, 14.6
    g = finmod.size_fin(b=b, S=S, l_t=l_t)
    Re = RHO_SL * V * g.chord / MU_SL
    want = (skin_friction_cf(Re, lref=g.chord) * wing_form_factor(g.tc)
            * 1.05 * 2.0 * g.S / S)
    assert vtail_cd0(S, l_t, b=b, V=V) == want


def test_the_fin_is_built_at_the_aspect_ratio_it_was_asked_for():
    """EXACTLY — which decides which of the two dimensions is derived.

    Sizing the height as ``sqrt(AR S)`` and dividing for the chord leaves
    ``height / chord`` a rounding away from ``AR`` (measured: at five of the
    six arms below). Sizing the chord and multiplying does not. The fin is a
    low-aspect-ratio surface whose AR is quoted in reports and drawn in CAD,
    so it should be the number that was asked for.
    """
    for l_t in (1.0, 2.5, 3.7, 5.5, 8.0, 12.3):
        g = finmod.size_fin(b=10.0, S=10.0, l_t=l_t)
        assert g.height == g.AR * g.chord, f"arm {l_t} m is off its AR"


def test_the_charged_fin_drag_is_bit_for_bit_what_it_was():
    """Refactoring the sizing must not move a single published score.

    ``cd0_fin`` enters the tail family's L/D on every layout that has a fin.
    These are the values before ``fin.py`` existed. Re-derive with
    ``.venv/bin/python -c "from aerobo.tail import vtail_cd0; ..."`` if they
    ever need updating deliberately; a surprise change here is a defect.
    """
    # RE-DERIVED once, deliberately: ``objective.MU_SL`` stopped being the
    # rounded literal 1.789e-5 and became the ISA model's own sea-level
    # viscosity, so every Reynolds in the book moved by 1.7e-4 relative and
    # these by 3.3e-5 (Cf ~ Re^-0.2). That was not a free choice — a family
    # defaulting to the literal while its MissionSpec derived ISA properties
    # made the flight modifier fail to reproduce its own published point.
    frozen = {1.0: 0.00415332902648218,
              2.5: 0.0018068739177588711,
              3.7: 0.0012666039415983757,
              5.5: 0.0008848460629241606,
              8.0: 0.0006307449654529177,
              12.3: 0.00042791152145944654}
    for l_t, want in frozen.items():
        assert vtail_cd0(10.0, l_t, b=10.0) == want, f"arm {l_t} m moved"


def test_the_published_constants_have_one_definition():
    assert V_V_DEFAULT is finmod.V_V_DEFAULT
    assert AR_VT_DEFAULT is finmod.AR_VT_DEFAULT


# ---------------------------------------------------- through the report

def test_the_report_carries_the_fin_and_a_v_tail_report_does_not():
    assert _report({})["geometry"]["fin"]["S"] > 0.0
    assert _report({"tail_type": "v_tail",
                    "dihedral_deg": 35.0})["geometry"].get("fin") is None


def test_the_fin_flown_is_the_fin_the_report_states():
    """STATED IS NOT FLOWN, closed: the lattice surface must BE the block."""
    rep = _report({})
    blk = rep["geometry"]["fin"]
    fm = build_flight_model(rep)
    m = fm.model
    v = m.is_vertical
    assert v.any()
    assert float(m.c[v][0]) == pytest.approx(blk["chord_m"], rel=1e-12)
    z = m.z[v]
    assert z.max() - z.min() == pytest.approx(abs(blk["height_m"]),
                                              rel=2e-2)
    assert float(m.x[v][0]) == pytest.approx(blk["x_qc_m"], rel=1e-12)


def test_moving_onto_the_designs_fin_weakened_the_yaw_by_the_measured_amount():
    """The size of the defect, pinned.

    The invented fin (0.12 b by 0.65 mac) was 17 % stronger in yaw than the
    one the drag book prices. If a future session re-invents a fin, this
    ratio moves and says so.
    """
    rep = _report({})
    blk = rep["geometry"]["fin"]
    mac = float(rep["geometry"]["tail"]["mac"])
    b = float(rep["geometry"]["b"])
    invented_area = (0.12 * b) * (0.65 * mac)
    assert invented_area / blk["S"] == pytest.approx(1.096, rel=0.05)

    from aerobo.flightmodel import ControlsSpec
    stated = build_flight_model(rep).deck
    forced = build_flight_model(
        rep, ControlsSpec(fin_height_m=0.12 * b,
                          fin_chord_m=0.65 * mac)).deck
    assert forced.Cn_beta / stated.Cn_beta == pytest.approx(1.21, rel=0.05)


def test_a_T_TAIL_carries_its_tailplane_ON_the_fin_tip():
    """The arrangement that makes a T-tail a T-tail, and the reported bug.

    Every other layout hangs the tailplane off the body and stands the fin
    beside it — root at the tail's height. A T-tail is the opposite: the
    tailplane sits ON TOP of the fin. Taking ``dz`` as the root for both put
    the fin's root AT the tailplane and its entire span above it, which is
    a conventional tail raised with a fin bolted over the top of it.

    The two heights need no reconciling: ``tail.tail_height`` gives a T-tail
    ``dz = sqrt(AR_VT * S_vt)``, which is exactly the height a
    volume-coefficient fin at that aspect ratio has. So the fin spans body to
    tailplane EXACTLY, and that closure is what this asserts.
    """
    t_rep = _report({"tail_type": "t_tail"})
    fin, tail = t_rep["geometry"]["fin"], t_rep["geometry"]["tail"]
    assert fin["z_root_m"] == pytest.approx(0.0, abs=1e-12), \
        "a T-tail's fin must start at the BODY"
    tip = fin["z_root_m"] + fin["height_m"]
    assert tip == pytest.approx(float(tail["dz_m"]), rel=1e-9), \
        f"the fin tip is at {tip:.4f} m and the tailplane at {tail['dz_m']:.4f}"

    # ...and a conventional tail is NOT rearranged: its fin still starts at
    # the tail's own height, and its tailplane is nowhere near the fin tip
    c_rep = _report({})
    c_fin, c_tail = c_rep["geometry"]["fin"], c_rep["geometry"]["tail"]
    assert c_fin["z_root_m"] == pytest.approx(float(c_tail["dz_m"]))
    assert (c_fin["z_root_m"] + c_fin["height_m"]) > float(c_tail["dz_m"]) + 0.5


def test_the_t_tail_fin_is_the_same_size_as_everyone_elses():
    """Moving the root must not have resized the surface: the volume
    coefficient still sets the area, and only WHERE it sits changed."""
    t = _report({"tail_type": "t_tail"})["geometry"]["fin"]
    c = _report({})["geometry"]["fin"]
    assert t["AR"] == c["AR"] and t["V_v"] == c["V_v"]
    # the arm differs (a T-tail's tailplane sits further out), so compare the
    # LAW rather than the value
    assert t["S"] == pytest.approx(t["V_v"] * 10.0 * 10.0 / t["l_t_m"],
                                   rel=1e-12)


def test_there_is_no_flown_but_uncharged_fin_left_to_warn_about():
    """This used to assert the WARNING — "the design flies a fin it did not
    pay for" — because the charge was a switch and it was off. There is no
    such state now, so the correct assertion is that the state is gone: the
    default design pays, and the flight rebuild has nothing to apologise
    for."""
    rep = _report({})
    assert rep["breakdown"]["cd0_fin"] > 0.0
    fm = build_flight_model(rep)
    assert not any("did not pay for" in a for a in fm.assumptions)


def test_a_report_with_no_fin_block_still_flies_and_says_it_assumed_one():
    """Reports written before this block exists must not stop working."""
    rep = _report({})
    rep["geometry"].pop("fin")
    fm = build_flight_model(rep)
    assert fm.model.is_vertical.any()
    assert any("not in this report" in a for a in fm.assumptions)


def test_overriding_the_chord_keeps_the_quarter_chord_ON_THE_ARM():
    """The invariant a fin volume coefficient rides on.

    ``V_v = S_v * arm / (S_w b_w)``, and the ARM is the quarter-chord
    station. Defaulting a blank station to the fin block's LEADING EDGE
    would pin a fixed number, so changing only the chord would slide the
    quarter chord aft and move ``V_v`` — measured in the V4 shell as
    0.03964 -> 0.04019 on taking the Reynolds recommendation, which is a
    panel silently changing the thing the user sized the fin by.

    So a blank station means "quarter chord on the design's arm", and the
    other two dimensions are still the design's.
    """
    from aerobo.flightmodel import ControlsSpec
    rep = _report({})
    blk = rep["geometry"]["fin"]
    for chord in (0.4, 1.0, 1.9):
        fm = build_flight_model(rep, ControlsSpec(fin_chord_m=chord))
        v = fm.model.is_vertical
        assert float(fm.model.c[v][0]) == pytest.approx(chord)
        assert float(fm.model.x[v][0]) == pytest.approx(blk["x_qc_m"],
                                                        rel=1e-12)
        zs = fm.model.z[v]
        assert zs.max() - zs.min() == pytest.approx(abs(blk["height_m"]),
                                                    rel=2e-2)
