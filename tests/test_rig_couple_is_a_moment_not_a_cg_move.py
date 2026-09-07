"""The RIG's couple is a MOMENT. Implementing it as a CG move passes half of
this file and fails the other half, which is the whole reason the file is
shaped in pairs.

WHY THIS FILE EXISTS
--------------------
A windfoil, wingfoil or kitefoil is driven by a rig whose centre of effort
sits one to three metres ABOVE the water, while the resistance it balances
acts a few tenths of a metre below it. That is a couple — ~399 N.m for a
windfoil, re-derived three independent ways in ``LITERATURE_REVIEW_FOILINGBO``
and recorded in ``aerobo/rig.py`` — and it is the largest longitudinal moment
on the craft. A kite additionally carries 21-46 % of the rider's weight
straight up (mean 35 %, five measured cases, vlugt2009 Table 6), so a foil
sized to the full weight is sized to a load it does not carry. Neither was in
this engine.

THE TEMPTATION, and why the pairs exist. The couple can be applied by
SHIFTING the CG by ``Delta = M / L``. It is one line, it needs no new
argument anywhere, and it reproduces the trim EXACTLY — the algebra is in
``rig.py`` and this file measures it at four offsets: the same alpha, the
same i_t and the same circulation on every panel, to 2.5e-16. It is also
wrong, and silently so. ``SM = (x_np - x_cg)/mac`` reads the shifted station,
so the stability constraint becomes a margin about a CG the craft does not
have. Measured here: the driven case moves the static margin by +3.30 while
the honest route moves it by EXACTLY zero. A constrained run under the CG
route would optimise against one real constraint and advertise two.

So test ``(b)`` uses the CG shift as an ORACLE — the one thing it is good
for — and test ``(c)`` is the assertion the oracle cannot satisfy. An
implementation that moves the CG passes ``(b)`` and fails ``(c)``; that pair
is the point, and mutation (ii) below runs it.

THE SIGN CONVENTION, and what it physically says
-------------------------------------------------
x AFT positive from the main foil's quarter chord, z UP positive with the
free surface at z = +depth, pitching moment NOSE-UP positive
(``vlm.VLM._cm``). A rig driving the craft FORWARD from a centre of effort
ABOVE the water therefore makes a NEGATIVE (nose-down) moment of magnitude
``T (depth + z_ce)``: sheet in hard and the bow goes down. The stabiliser
holds the nose up, so it must push DOWN harder — i_t falls and the
stabiliser's load falls with it. Test ``(g)`` asserts exactly that, on the
literature's own 145 N windfoil drive.

THE TWO FROZEN ANCHORS
----------------------
Both re-established by measurement on this working tree before any edit, by
building ``hydrofoil + elevator`` through ``api`` and evaluating it at a
stated point of its own design box:

* CENTRE, x = (lo + hi)/2 = (taper 0.6, twist_root 0 deg, twist_tip -2 deg,
  t/c 0.12, depth 0.575 m, V 12 m/s, S_t 0.04 m2, l_t 1.0 m)
  -> L/D = 24.67276683709319, SM = -0.2276, x_np = 0.1221, mac = 0.1225,
  l_t = 1.0, x_cg = 0.15, i_t = -0.095 deg.

* OFF-CENTRE, x = lo + 0.75*(hi - lo) = (taper 0.8, twist_root 2 deg,
  twist_tip 0 deg, t/c 0.14, depth 0.7875 m, V 14 m/s, S_t 0.05 m2,
  l_t 1.25 m)
  -> L/D = 22.568044635929535, SM = +0.0858, x_np = 0.1978, x_cg = 0.1875,
  i_t = +0.801 deg.

They are the same two ``tests/test_craft_weight_is_a_flag.py`` and
``tests/test_the_speed_row_travels_to_the_problem.py`` froze, kept
deliberately: the off-centre point sits on the OPPOSITE side of the
static-margin boundary from the centre (g_sm +0.0058 against -0.3076), so a
change that quietly moved the balance shows up even where the centre is
blind. It also matters here for a second reason — the large negative offset
is untrimmable at the centre and flies at the off-centre point, so the oracle
has somewhere to run.

WHAT WAS MEASURED WHILE WRITING THIS (quoted, not assumed)
-----------------------------------------------------------
* the four offsets, at the two design points:
  tow +0.0968 / +0.0813 m, kite -0.0541 / -0.0643 m,
  windfoil -0.0622 / -0.0674 m, driven -0.3679 / -0.3982 m. The windfoil
  couple lands at 373-404 N.m against the literature's ~399 N.m, which is a
  cross-check rather than a coincidence: it is the same 145 N at the same
  2.0 m, with this engine's own flying depth supplying the rest of the lever.
* oracle agreement, cm_ac route vs CG-shift route, over those offsets:
  d(alpha) <= 3.5e-17 rad, d(i_t) <= 9.8e-17 rad, max panel
  |d(Gamma)| <= 2.5e-16. The driven case is refused at the box CENTRE — the
  foil runs past its polar's validity — and by BOTH arms with the same
  reason, which is itself part of the agreement.
* the honest route's effect on x_np and SM: EXACTLY zero, not merely small.
  It is exact by construction — ``cm_ac`` never enters the influence matrix,
  so the geometry, ``_Gam1`` and ``neutral_point()`` are bit-identical — and
  the assertions say so rather than hiding behind a tolerance.
* what the CG route would do to the constraint, over 200 seeded draws with
  the driven rig: the honest route satisfies the static margin on 40 of 127
  flown designs (31.5 %, worst -0.97), the CG route on 134 of 134 (100 %,
  worst +1.87). Vacuous is measured, not asserted.
* pre-trim readout vs solved stabiliser load, 329 evaluations over the two
  rigs (800 over four while this was being built): they agree to 4.4e-16 in
  CL and NEVER disagree in sign.
* which way up the section is mounted vs the sign actually flown: 6
  disagreements in 199 flown draws for the kite rig and 0 in 130 for the
  driven one, the worst at |CL_stab| = 0.0196 — inside the crossing band
  ``tail.stabiliser_load`` documents, and nothing outside it.
"""

import numpy as np
import pytest

from aerobo import api, hydrotail
from aerobo.objective import PENALTY
from aerobo.rig import CRAFT_LOD_DEFAULT, NO_RIG, RigLoads

#: the problem both anchors are measured on. Speed and depth are design
#: variables in its box, which is what lets one build answer questions at
#: several operating points without several problems.
ANCHOR = "hydrofoil + elevator"

#: L/D at the CENTRE of ANCHOR's box and at lo + 0.75*(hi - lo). Measured on
#: this tree before the rig existed; the module docstring records the design
#: points and the trim state each was taken with.
ANCHOR_CENTRE_LOD = 24.67276683709319
ANCHOR_OFF_CENTRE_LOD = 22.568044635929535

#: the published craft weight of every water family (hydrofoil.py's
#: ``L_design``), quoted here only so the arithmetic in the tests below can be
#: written out in full. It is ASSERTED against the dataclass in (a), so a
#: change to the default breaks this file loudly rather than silently.
WEIGHT_N = 6000.0

#: the crossing band ``tail.stabiliser_load`` documents: near zero load the
#: mirrored section can land marginally the other side of the crossing from
#: the reading that mirrored it, because mirroring a section mirrors its
#: couple. Its width is not a constant — it is twice the stabiliser's own
#: couple over ``S_t l_t``, so it grows where the stabiliser is small and the
#: arm short. Measured over the box here: 6 draws in 199 for the kite rig
#: (worst |CL_stab| 0.0196) and 3 in 176 for a lighter drive (worst 0.0035).
#: 0.05 is that with room; the test asserts BOTH that nothing outside the
#: band disagrees AND that the band is not doing the work, by capping how
#: many draws may fall in it.
CROSSING_BAND_CL = 0.05


# ------------------------------------------------------------------ helpers

#: THE PHYSICS THE ANCHORS WERE FROZEN ON, stated as flags rather than left
#: to the defaults. Session 68 changed two of this family's defaults — the
#: craft became FLAT (``Z_T_FRAC_DEFAULT`` = 0.01 b, was 0.05 b) and its
#: strut became a placed, loaded surface instead of a rectangle of wetted
#: area (``strut_model``) — and both move every L/D in the family.
#:
#: The anchors are re-pinned here rather than re-measured, because what they
#: are FOR is unchanged: they ask whether ``rig=None`` reproduces the answer
#: the engine gave before the rig existed, and the rig code is not what
#: moved. Pinning them to the published switches keeps that question exact
#: AND makes this file the tripwire that the published charge is still
#: reachable bit-for-bit — which is the only reason ``strut_model`` survives
#: as a flag instead of being deleted.
PUBLISHED_PHYSICS = {"strut_model": False,
                     "z_t_m": -hydrotail.DZ_FRAC * 1.2}


#: ...and the same two switches as CONSTRUCTOR kwargs, for the tests that
#: build the problem directly instead of through the registry.
PUBLISHED_KWARGS = {"strut_model": False,
                    "z_t_fixed": -hydrotail.DZ_FRAC * 1.2}


def _published_htp(**kw):
    """A ``HydrofoilTailProblem`` on the physics the frozen numbers here were
    measured on. Only the tests that pin an ABSOLUTE number need it; the
    pairwise comparisons are about the rig and hold on either physics."""
    return hydrotail.HydrofoilTailProblem(**{**PUBLISHED_KWARGS, **kw})


def _box(name: str = ANCHOR, flags: dict | None = None):
    built = api.PROBLEM_SPECS[name].build(
        {}, dict(PUBLISHED_PHYSICS if flags is None else flags), None)
    return built, built.bounds[:, 0], built.bounds[:, 1]


def _points():
    """The two frozen design points, as (name, x, depth_m, l_t_m)."""
    _, lo, hi = _box()
    for tag, x in (("centre", 0.5 * (lo + hi)),
                   ("off-centre", lo + 0.75 * (hi - lo))):
        labels = api.PROBLEM_SPECS[ANCHOR].param_labels
        yield tag, x, float(x[labels.index("depth_m")]), \
            float(x[labels.index("l_t_m")])


#: The four rigs the oracle runs, each a NAMED craft rather than a set of
#: numbers, and each chosen for the offset it produces (module docstring):
#:
#: ``tow``      a foil towed from a low, forward bridle that also lifts —
#:              the only POSITIVE offset here, because the vertical share
#:              acting well ahead of the CG out-pitches the drive.
#: ``kite``     the measured kitefoil: 35 % of the weight lifted vertically
#:              [vlugt2009 Table 6, mean of five cases], a real side force
#:              that the longitudinal solver correctly ignores, a centre of
#:              effort 0.30 m ahead of the CG, and a drive closed from the
#:              craft L/D. Both moment terms, opposite signs.
#: ``windfoil`` drake_blog's MEASURED 145 N drive at zhang2025's MEASURED
#:              2.0 m rig height — the literature's own ~399 N.m couple,
#:              reproduced here at 373-404 N.m depending on how deep the foil
#:              is flying (the lever includes the depth).
#: ``driven``   the same craft with its drive CLOSED from the default craft
#:              L/D of 7.0 instead of stated: 857 N, a -0.40 m offset, and
#:              the large negative case the brief asks for. It is the one
#:              that takes the static margin vacuous (+3.30) under the CG
#:              route.
RIGS = {
    "tow": dict(z_ce_m=0.0, thrust_n=500.0, vertical_n=600.0, x_ce_m=-1.2),
    "kite": dict(z_ce_m=1.5, vertical_n=2100.0, side_n=900.0, x_ce_m=-0.30),
    "windfoil": dict(z_ce_m=2.0, thrust_n=145.0),
    "driven": dict(z_ce_m=2.0),
}


def _expected_offset(kw: dict, *, weight_n: float, depth_m: float,
                     x_cg_m: float) -> tuple:
    """``(M_rig [N.m], L_required [N], Delta [m])``, written out here.

    Deliberately NOT imported from ``rig.py``: this is the balance itself,
    from the module docstring's frame, so an engine that changed the lever,
    dropped the vertical term or flipped a sign fails the oracle rather than
    agreeing with its own mistake.

        L_required = W - Z_rig
        T          = thrust_n, or L_required / craft_lod when none is stated
        M_rig      = -T (depth + z_ce)  +  (x_cg - x_ce) Z_rig
        Delta      = M_rig / L_required
    """
    z_rig = float(kw.get("vertical_n", 0.0))
    lift = float(weight_n) - z_rig
    thrust = kw.get("thrust_n")
    if thrust is None:
        thrust = lift / float(kw.get("craft_lod", CRAFT_LOD_DEFAULT))
    moment = (-float(thrust) * (float(depth_m) + float(kw["z_ce_m"]))
              + (float(x_cg_m) - float(kw.get("x_ce_m", 0.0))) * z_rig)
    return float(moment), float(lift), float(moment / lift)


def _arms(kw: dict, *, depth_m: float, l_t_m: float, x_cg_m: float):
    """The three problems a single oracle case needs.

    ``rig``    the honest route: the couple through ``cm_ac``.
    ``shift``  the oracle: NO rig, the CG moved by ``Delta`` and the weight
               reduced to the lift the rig leaves — zero engine code, which
               is what makes it an independent check rather than a mirror.
    ``base``   neither: the same craft with the same reduced weight and the
               unmoved CG, so ``(c)`` has a no-couple reference to compare
               both routes against.

    ``tail_inverted`` is PINNED to the same value in all three. That is not
    hiding anything — it is what makes the comparison about the trim algebra:
    the mounting rule bakes ``alpha_L0`` into the panel normals, so a surface
    that flipped in one arm and not another would be a different aeroplane
    with a legitimately different ``x_np``. Which way up the section actually
    goes is test (d)'s question, and it is asked there with the rule live.
    """
    moment, lift, delta = _expected_offset(
        kw, weight_n=WEIGHT_N, depth_m=depth_m, x_cg_m=x_cg_m)
    common = dict(tail_inverted=False)
    rig = hydrotail.HydrofoilTailProblem(
        rig=RigLoads(**kw), x_cg=x_cg_m, **common)
    shift = hydrotail.HydrofoilTailProblem(
        x_cg=x_cg_m + delta, L_design=lift, **common)
    base = hydrotail.HydrofoilTailProblem(
        x_cg=x_cg_m, L_design=lift, **common)
    return rig, shift, base, moment, lift, delta


# ---------------------------------------------------------------- (a)

def test_no_rig_reproduces_the_centre_anchor_bit_for_bit():
    """``rig=None`` -> the frozen centre anchor, to the last bit."""
    built, lo, hi = _box()
    x = 0.5 * (lo + hi)
    lod, g = built.callable(x)
    assert lod == ANCHOR_CENTRE_LOD
    assert built.problem.rig is NO_RIG is None

    out = built.evaluate(x)
    assert out["SM"] == pytest.approx(-0.2276, abs=5e-5)
    assert out["x_np"] == pytest.approx(0.1221, abs=5e-5)
    assert out["mac"] == pytest.approx(0.1225, abs=5e-5)
    assert out["l_t"] == 1.0
    assert out["x_cg"] == 0.15
    assert out["i_t_deg"] == pytest.approx(-0.095, abs=5e-4)
    assert g[1] == pytest.approx(-0.3076, abs=5e-5)
    # ...and the feature is ABSENT rather than present-and-zero: no report
    # block, and the lift the foil owes is the whole craft weight
    assert "rig" not in out
    assert built.problem.L_required == built.problem.L_design == WEIGHT_N


def test_no_rig_reproduces_the_off_centre_anchor_bit_for_bit():
    """The SECOND frozen anchor, at lo + 0.75*(hi - lo).

    A box centre is not a box: this project has twice recorded a null at a
    centre that reversed at a corner. The centre of THIS box trims the
    stabiliser down through an unstable static margin and this point trims it
    up through a stable one — two different balances, one number each.
    """
    built, lo, hi = _box()
    x = lo + 0.75 * (hi - lo)
    lod, g = built.callable(x)
    assert lod == ANCHOR_OFF_CENTRE_LOD

    out = built.evaluate(x)
    assert out["SM"] == pytest.approx(0.0858, abs=5e-5)
    assert out["x_np"] == pytest.approx(0.1978, abs=5e-5)
    assert out["x_cg"] == 0.1875
    assert out["i_t_deg"] == pytest.approx(0.801, abs=5e-4)
    assert g[1] == pytest.approx(0.0058, abs=5e-5)
    assert "rig" not in out


def test_the_published_weight_is_still_the_one_the_arithmetic_assumes():
    """A tripwire, not a restatement: every offset below is computed from
    :data:`WEIGHT_N`, so a change to the family's published ``L_design``
    must break here and not in an unexplained numeric drift."""
    assert hydrotail.HydrofoilTailProblem().L_design == WEIGHT_N
    assert CRAFT_LOD_DEFAULT == 7.0


# ---------------------------------------------------------------- (b)

@pytest.mark.parametrize("rig_name", sorted(RIGS))
def test_the_couple_agrees_with_a_cg_shift_of_M_over_L(rig_name):
    """THE ORACLE: cm_ac route == CG-shift route, in alpha, i_t and every
    circulation.

    The CG-shift arm needs no engine code at all — it is a second
    ``HydrofoilTailProblem`` with ``x_cg`` moved by ``Delta = M/L`` and
    ``L_design`` reduced by the rig's vertical share — so agreement is
    evidence about the moment, not a mirror of the implementation. ``Delta``
    itself is computed in :func:`_expected_offset` from the balance written
    out in this file.

    Both design points are run, and the offsets span +0.10 m to -0.40 m. The
    large negative one is untrimmable at the box CENTRE (the foil runs past
    its polar's validity long before the stabiliser runs out of incidence) —
    which is itself a consistency check, since BOTH arms refuse it with the
    same reason — so those cases are skipped there and flown off-centre.
    """
    kw = RIGS[rig_name]
    seen = 0
    for tag, x, depth, l_t in _points():
        x_cg = 0.15 * l_t          # this file's own stated CG, pinned below
        rig, shift, base, moment, lift, delta = _arms(
            kw, depth_m=depth, l_t_m=l_t, x_cg_m=x_cg)
        a = hydrotail.evaluate_hydrofoil_tail(x, rig)
        b = hydrotail.evaluate_hydrofoil_tail(x, shift)
        if not a["feasible"] or not b["feasible"]:
            # the two arms must at least AGREE about refusing it
            assert a["feasible"] == b["feasible"], (tag, rig_name)
            assert a["reason"] == b["reason"], (tag, rig_name)
            continue
        seen += 1
        assert a["alpha_rad"] == pytest.approx(b["alpha_rad"], abs=1e-14)
        assert a["i_t_rad"] == pytest.approx(b["i_t_rad"], abs=1e-14)
        ga, gb = a["vlm"].Gamma, b["vlm"].Gamma
        assert ga.shape == gb.shape
        assert float(np.max(np.abs(ga - gb))) < 1e-14
        # ...and the offset really is the size claimed, so a case that
        # agreed because both arms were near zero cannot pass quietly
        assert abs(delta) > 1e-3, (tag, rig_name, delta)
    assert seen, f"{rig_name} flew at neither design point"


@pytest.mark.parametrize("rig_name", sorted(RIGS))
def test_the_reported_couple_is_the_couple_the_balance_says(rig_name):
    """The report block against the balance written out in this file.

    Kept OUT of the oracle above on purpose. The oracle's claim is that two
    routes fly the same aeroplane; this one's is that the number on the card
    is the moment the physics implies. Mixing them would mean an
    implementation could fail the oracle for a reporting reason, and the
    whole value of (b)/(c) as a pair is that each fails for exactly one
    reason.

    ``seen`` is not decoration. The loop skips a design point the rig cannot
    trim (the driven case is untrimmable at the box CENTRE and flies only
    off-centre — measured, 1 of 2 points), so without the count a future
    bound change or physics fix that made BOTH points infeasible would leave
    this test green having executed no assertion at all. Its two siblings in
    (b) and (c) already carry exactly this guard.
    """
    kw = RIGS[rig_name]
    seen = 0
    for tag, x, depth, l_t in _points():
        x_cg = 0.15 * l_t
        rig, _, _, moment, lift, delta = _arms(
            kw, depth_m=depth, l_t_m=l_t, x_cg_m=x_cg)
        out = hydrotail.evaluate_hydrofoil_tail(x, rig)
        if not out["feasible"]:
            continue
        seen += 1
        assert out["rig"]["M_rig_nm"] == pytest.approx(moment, rel=1e-12)
        assert out["rig"]["lift_required_n"] == pytest.approx(lift, rel=1e-12)
        assert out["rig"]["couple_as_cg_shift_m"] == pytest.approx(delta,
                                                                   rel=1e-12)
        # the non-dimensional form the solver actually carries: M / (q S mac)
        q = 0.5 * rig.rho * out["V"] ** 2
        assert out["rig"]["Cm_rig"] == pytest.approx(
            moment / (q * rig.S * out["mac"]), rel=1e-12)
    assert seen, f"{rig_name} flew at neither design point"


def test_the_large_negative_offset_is_flown_and_is_large():
    """The offsets are not all small, and the biggest is negative.

    Without this the parametrised oracle above could be satisfied by four
    nearly-identical tiny couples, which is the failure mode "we tested three
    offsets" is supposed to exclude.
    """
    _, x, depth, l_t = list(_points())[1]          # off-centre
    x_cg = 0.15 * l_t
    offsets = {name: _expected_offset(kw, weight_n=WEIGHT_N, depth_m=depth,
                                      x_cg_m=x_cg)[2]
               for name, kw in RIGS.items()}
    assert min(offsets.values()) < -0.35, offsets
    assert max(offsets.values()) > +0.05, offsets
    # ...and the large one is worth several mean chords of trim station: the
    # foil's mac here is 0.1205 m, so -0.40 m is 3.3 MAC
    rig, _, _, _, _, delta = _arms(RIGS["driven"], depth_m=depth, l_t_m=l_t,
                                   x_cg_m=x_cg)
    out = hydrotail.evaluate_hydrofoil_tail(x, rig)
    assert out["feasible"], out["reason"]
    assert abs(delta) / out["mac"] > 3.0


# ---------------------------------------------------------------- (c)

@pytest.mark.parametrize("rig_name", sorted(RIGS))
def test_the_couple_leaves_the_neutral_point_and_the_margin_alone(rig_name):
    """AND THEY DISAGREE WHERE IT MATTERS.

    The honest route leaves ``x_np`` and ``SM`` EXACTLY where the no-couple
    baseline had them — exactly, not approximately, because ``cm_ac`` never
    enters the influence matrix, so the panel normals, ``_Gam1`` and
    ``neutral_point()`` are bit-identical. The CG-shift route moves the
    margin by ``-Delta/mac`` about a station the craft does not have.

    This is the assertion an implementation that "just moves the CG" cannot
    satisfy while satisfying (b), and mutation (ii) runs exactly that.
    """
    kw = RIGS[rig_name]
    seen = 0
    for tag, x, depth, l_t in _points():
        x_cg = 0.15 * l_t
        rig, shift, base, moment, lift, delta = _arms(
            kw, depth_m=depth, l_t_m=l_t, x_cg_m=x_cg)
        a = hydrotail.evaluate_hydrofoil_tail(x, rig)
        c = hydrotail.evaluate_hydrofoil_tail(x, base)
        b = hydrotail.evaluate_hydrofoil_tail(x, shift)
        if not (a["feasible"] and b["feasible"] and c["feasible"]):
            continue
        seen += 1
        # the honest route: untouched, and the brief's 1e-12 contract as well
        assert a["x_np"] == c["x_np"], tag
        assert a["SM"] == c["SM"], tag
        assert abs(a["x_np"] - c["x_np"]) <= 1e-12
        assert abs(a["SM"] - c["SM"]) <= 1e-12
        # ...and it did something: the trim moved even though the margin did
        # not, which is what "a couple, not a CG move" means
        assert abs(a["i_t_rad"] - c["i_t_rad"]) > 1e-6, tag

        # the CG route: the margin moves by exactly -Delta/mac
        assert b["SM"] - c["SM"] == pytest.approx(-delta / c["mac"],
                                                  rel=1e-10)
        # x_np is a property of the GEOMETRY, so even the CG route leaves it
        # alone — the damage is entirely in what SM is measured FROM, which
        # is why a test that only watched x_np would have seen nothing
        assert b["x_np"] == c["x_np"], tag
    assert seen, f"{rig_name} flew at neither design point"


def test_the_cg_route_would_make_the_stability_constraint_vacuous():
    """Why (c) is worth a test rather than a comment, priced over the BOX.

    A single design point can only show that the two routes report different
    margins. What makes the CG route a defect rather than a difference is
    what it does to the CONSTRAINT: measured here over 200 seeded draws with
    the driven rig, the honest route satisfies the static-margin constraint
    on 40 of 127 flown designs (31.5 %, worst margin -0.97), while the CG
    route satisfies it on 134 of 134 (100 %, worst margin +1.87). A
    constrained run under the CG route would be advertising two constraints
    and optimising against one.

    The CG route is given the box-CENTRE equivalent shift, because that is
    the best a CG shift can do: the couple's lever includes ``depth``, which
    is a design VARIABLE, so no single station can represent it across the
    box at all — a second, independent reason the substitution is wrong that
    the single-point oracle in (b) cannot show.
    """
    _, lo, hi = _box()
    labels = api.PROBLEM_SPECS[ANCHOR].param_labels
    depth_c = 0.5 * (lo[labels.index("depth_m")] + hi[labels.index("depth_m")])
    l_t_c = 0.5 * (lo[labels.index("l_t_m")] + hi[labels.index("l_t_m")])
    x_cg = 0.15 * l_t_c
    _, _, delta = _expected_offset(RIGS["driven"], weight_n=WEIGHT_N,
                                   depth_m=depth_c, x_cg_m=x_cg)

    honest = hydrotail.HydrofoilTailProblem(
        rig=RigLoads(**RIGS["driven"]), x_cg=x_cg)
    cg_route = hydrotail.HydrofoilTailProblem(x_cg=x_cg + delta)
    assert honest.SM_min == cg_route.SM_min

    rng = np.random.default_rng(20260810)
    draws = lo + rng.random((200, lo.size)) * (hi - lo)
    counts = {}
    for tag, prob in (("honest", honest), ("cg", cg_route)):
        flown = [hydrotail.evaluate_hydrofoil_tail(x, prob) for x in draws]
        margins = [o["g"][1] for o in flown if o["feasible"]]
        counts[tag] = (len(margins), sum(1 for m in margins if m >= 0.0),
                       min(margins))

    n_h, ok_h, worst_h = counts["honest"]
    n_c, ok_c, worst_c = counts["cg"]
    assert n_h > 100 and n_c > 100, counts
    # the honest route asks a real question: some designs are stable, most
    # are not, and the boundary is interior to the box
    assert 0 < ok_h < n_h, counts
    assert worst_h < 0.0, counts
    # the CG route answers it for everybody, everywhere, with slack
    assert ok_c == n_c, counts
    assert worst_c > 1.0, counts


# ---------------------------------------------------------------- (d)

@pytest.mark.parametrize("rig_name", ["kite", "driven"])
def test_the_pretrim_readout_and_the_solve_agree_on_the_load(rig_name):
    """THE TWO-AUTHORS TEST, over 200 seeded draws from the box.

    The stabiliser's load has two authors here: ``tail.trim_lift_coefficient``
    closes it in closed form BEFORE the run (it is what a card quotes, and
    what decides which way up the section is mounted), and the 3-way solve
    flies it. This repo has a recorded bug of exactly the shape where those
    two saw different physics — "the tail's download has two authors" — so
    the rig couple is routed through ``cm_ac_model``, which both of them
    read, and this test is the tripwire on that routing.

    Two rigs, because one sign regime is not a test: the kite case flies both
    a lifting and a pushing stabiliser across the box, the driven case pushes
    on nearly every draw.

    Asserted, in increasing strength:

    * they never disagree on the SIGN (0 disagreements in 800 evaluations
      while this was written);
    * they agree NUMERICALLY to 1e-9 — measured at 5.6e-16, i.e. the closed
      form and the lattice are the same balance and not two approximations
      of it;
    * ``tail_section_inverted`` — the field the geometry views and the CAD
      export read — matches the sign actually flown, except inside the
      few-thousandths crossing band ``tail.stabiliser_load`` documents, and
      the test pins that the band is where the exceptions are rather than
      merely large enough to hide them.
    """
    _, lo, hi = _box()
    rng = np.random.default_rng(20260810)
    draws = lo + rng.random((200, lo.size)) * (hi - lo)
    prob = hydrotail.HydrofoilTailProblem(rig=RigLoads(**RIGS[rig_name]))

    flown, mismatched = 0, []
    up = down = 0
    for x in draws:
        out = hydrotail.evaluate_hydrofoil_tail(x, prob)
        if not out["feasible"]:
            continue
        flown += 1
        pre, act = out["CL_stab_pretrim"], out["CL_stab"]
        assert np.sign(pre) == np.sign(act), (pre, act)
        assert pre == pytest.approx(act, abs=1e-9), (pre, act)
        up, down = (up + 1, down) if act >= 0.0 else (up, down + 1)
        if out["tail_section_inverted"] != (act < 0.0):
            mismatched.append(abs(act))

    assert flown > 100, flown
    assert down > 10, "the rig never put a download on the stabiliser"
    if mismatched:
        assert max(mismatched) < CROSSING_BAND_CL, sorted(mismatched)
        # ...and the band is not swallowing the test: the overwhelming
        # majority of draws sit well outside it
        assert len(mismatched) < 0.05 * flown, (len(mismatched), flown)


#: :data:`RIGS` in the FLAG spelling a shell uses. Derived rather than
#: restated, so the two names for one craft cannot drift apart — and the
#: test below asserts the round trip (``built.problem.rig ==
#: RigLoads(**RIGS[name])``), which is what makes the mapping evidence
#: rather than a convention.
_RIG_FLAG_OF = {
    "z_ce_m": api.RIG_CE_HEIGHT_KEY,
    "thrust_n": api.RIG_THRUST_KEY,
    "side_n": api.RIG_SIDE_KEY,
    "vertical_n": api.RIG_VERTICAL_KEY,
    "x_ce_m": api.RIG_CE_X_KEY,
    "craft_lod": api.RIG_CRAFT_LOD_KEY,
}


def _rig_flags(rig_name: str) -> dict:
    return {_RIG_FLAG_OF[k]: v for k, v in RIGS[rig_name].items()}


#: THE DESIGN BOX A CALLER ACTUALLY CREATES. ``depth_m`` is a design row a
#: shell may narrow (``api._water_band_kwargs`` carries it into
#: ``DEPTH_BOUNDS`` on the built problem), and a craft on a 0.30 m mast is
#: the case ``trim_surface_cl``'s ``bounds_overrides`` was added for: with
#: the box withheld the couple's lever closes on the PUBLISHED band's
#: 0.575 m mid-depth, i.e. nearly twice the lever the run flies.
SHALLOW_MAST_BOX = {"depth_m": (0.20, 0.40)}

#: What the read-out must SAY, per rig, measured on this tree.
#:
#: ``inverted`` is the discriminating column and the reason this test is
#: parametrised at all: only the DRIVEN case turns the stabiliser over. "A
#: rig flips the tail" is false, so an implementation cannot satisfy this
#: table by keying the orientation on the presence of a rig — it has to
#: close the balance.
#:
#: ``couple_ratio_min`` is a floor on ``|m_rig| / |m_ac of the plain
#: read-out|``, the "the rig's share is the dominant one" claim. Measured
#: (published box / shallow mast): driven 25.5/22.8, tow 6.0/7.6, windfoil
#: 4.3/3.9, kite 2.4/0.67. The kite's 0.67 is not a slack floor — it is the
#: fact that the rig couple is NOT always the larger term, which is the
#: other half of why one rig shape is not a test of three.
READOUT_EXPECT = {
    "driven": dict(inverted=True, couple_ratio_min=20.0),
    "tow": dict(inverted=False, couple_ratio_min=5.0),
    "windfoil": dict(inverted=False, couple_ratio_min=3.5),
    "kite": dict(inverted=False, couple_ratio_min=0.5),
}


@pytest.mark.parametrize("box", [None, SHALLOW_MAST_BOX],
                         ids=["published-box", "shallow-mast"])
@pytest.mark.parametrize("rig_name", sorted(RIGS))
def test_the_pre_run_readout_agrees_with_the_run_it_predicts(rig_name, box):
    """THE THIRD AUTHOR: ``api.trim_surface_cl``.

    ``hydrotail`` has two authors of the stabiliser's load and they share
    ``cm_ac_model`` (the test above). ``api.trim_surface_cl`` is a third — it
    is what a shell quotes BEFORE any run, and what a section database is
    screened at — and it closes the same balance from the box's own
    mid-points with its own copy of the terms. Left alone, it would have gone
    on quoting the sections' couple as the whole story: at the box centre it
    said the elevator LIFTS at CL +0.276 while the run flies a DOWNLOAD at
    -0.461, i.e. a database screened on the wrong side of the aerofoil.

    Both numbers are asserted, and their agreement is exact rather than
    close, because the closed form is the same closed form.

    WHY FOUR RIGS AND TWO BOXES, and not the one shape this used to run.
    The couple has THREE terms — the drive times ``(depth + z_ce)``, the
    vertical share times ``(x_cg - x_ce)``, and the relief that vertical
    share takes out of ``L_required`` and so out of ``CL_target``. The
    driven case ``{RIG_CE_HEIGHT_KEY: 2.0}`` sets TWO OF THEM TO ZERO: with
    ``vertical_n = 0`` the ``(x_cg - x_ce) Z`` term vanishes whatever
    ``x_ce_m`` says, and ``L_required`` stays at the whole weight. A
    read-out that dropped either — or closed the moment about the wrong
    station — agreed with the run on that rig and on that rig alone. The
    kite and the tow carry both terms with a non-zero ``rig_ce_x_m``, and
    :func:`_expected_offset` (this file's own algebra, not the engine's) is
    what ``m_rig_m3`` is checked against, so the check is of the balance and
    not a mirror of it.

    And ``bounds_overrides`` is the RUN'S OWN BOX. A caller whose craft
    rides on a 0.30 m mast gets a lever nearly half the published band's,
    so the ``shallow-mast`` arm asserts not only that the read-out still
    agrees with the run but that a read-out given NO box would have
    disagreed with it — the defect the parameter exists to close, priced.
    """
    flags = _rig_flags(rig_name)
    expect = READOUT_EXPECT[rig_name]

    plain = api.trim_surface_cl(ANCHOR, bounds_overrides=box)
    rigged = api.trim_surface_cl(ANCHOR, flags=flags, bounds_overrides=box)
    assert plain is not None and rigged is not None

    built = api.PROBLEM_SPECS[ANCHOR].build({}, flags, box)
    b = np.asarray(built.bounds, dtype=float)
    flown = built.evaluate(0.5 * (b[:, 0] + b[:, 1]))   # the box CENTRE, which
    assert flown["feasible"], flown["reason"]           # the read-out reads

    # the FLAG spelling and the value object are the same craft
    assert built.problem.rig == RigLoads(**RIGS[rig_name])

    # the read-out is the run, to the last bit the two arithmetics share
    assert rigged["cl"] == pytest.approx(flown["CL_stab"], abs=1e-12)
    assert rigged["inverted"] is flown["tail_section_inverted"]
    assert rigged["inverted"] is expect["inverted"]     # ...and only DRIVEN
    assert rigged["cl_section"] == pytest.approx(abs(flown["CL_stab"]),
                                                 abs=1e-12)
    assert plain["cl"] > 0.0 and plain["inverted"] is False
    if expect["inverted"]:
        # the rig turns the read-out over, exactly as it turns the run over
        assert plain["cl"] > 0.0 > rigged["cl"]

    # THE COUPLE ITSELF, against the balance written out in this file rather
    # than read back off the engine: all three terms, with the depth and the
    # speed taken from the row the RUN searches (the built problem's own
    # ``DEPTH_BOUNDS`` / ``V_BOUNDS``), which is the whole of the
    # ``bounds_overrides`` fix one function up.
    prob = built.problem
    depth = 0.5 * (prob.DEPTH_BOUNDS[0] + prob.DEPTH_BOUNDS[1])
    v_design = 0.5 * (prob.V_BOUNDS[0] + prob.V_BOUNDS[1])
    arm = 0.5 * sum(api.PROBLEM_SPECS[ANCHOR].default_bounds["l_t_m"])
    x_cg = float(prob.x_cg_for(arm))                    # derived, not quoted
    assert rigged["x_cg_m"] == pytest.approx(x_cg, rel=1e-15)

    moment, lift, _delta = _expected_offset(
        RIGS[rig_name], weight_n=WEIGHT_N, depth_m=depth, x_cg_m=x_cg)
    q = 0.5 * float(prob.rho) * v_design ** 2
    assert rigged["m_rig_m3"] == pytest.approx(moment / q, rel=1e-12)
    #  ...the vertical share's OTHER effect, which the driven rig also
    #  cannot see: less lift to make, so a smaller trim target
    assert prob.L_required == lift
    assert rigged["cl_target"] == pytest.approx(prob.CL_target(v_design),
                                                rel=1e-15)

    # the rig's share is reported separately, and how big it is against the
    # sections' own couple is a property of the CRAFT (READOUT_EXPECT)
    assert plain["m_rig_m3"] == 0.0
    assert abs(rigged["m_rig_m3"]) \
        > expect["couple_ratio_min"] * abs(plain["m_ac_m3"])
    assert "rig couple" in rigged["source"]
    assert "rig" not in plain["source"]

    # ...and the total is NOT the naive sum of the two readings. What is
    # left in ``m_ac`` after the rig's share is the SECTIONS' couple — bit
    # for bit the plain read-out's whole ``m_ac`` while the stabiliser stays
    # upright, and a DIFFERENT number exactly where the rig mirrors it,
    # because mirroring a section mirrors that surface's own couple.
    residual = abs(rigged["m_ac_m3"] - rigged["m_rig_m3"])
    if expect["inverted"]:
        assert residual == pytest.approx(0.6303 * abs(plain["m_ac_m3"]),
                                         rel=1e-3)
    else:
        assert residual == pytest.approx(abs(plain["m_ac_m3"]), rel=1e-12)

    if box is not None:
        # THE DEFECT ``bounds_overrides`` EXISTS TO CLOSE, priced. Withhold
        # the box and the lever closes on the published band's 0.575 m mid
        # depth whatever the run was told to fly, so the number quoted stops
        # being the number flown. Measured gaps in CL on this craft:
        # driven 0.080, kite 0.052, tow 0.047, windfoil 0.014.
        withheld = api.trim_surface_cl(ANCHOR, flags=flags)
        assert abs(withheld["cl"] - flown["CL_stab"]) > 1e-2, rig_name


def test_the_kite_case_flies_the_stabiliser_both_ways_up():
    """The mixed-sign regime (d) leans on actually exists.

    Without it, "the flag matches the sign flown" could be satisfied by a
    field that is constant and a load that never changes sign.
    """
    _, lo, hi = _box()
    rng = np.random.default_rng(20260810)
    draws = lo + rng.random((200, lo.size)) * (hi - lo)
    prob = hydrotail.HydrofoilTailProblem(rig=RigLoads(**RIGS["kite"]))
    seen = set()
    for x in draws:
        out = hydrotail.evaluate_hydrofoil_tail(x, prob)
        if out["feasible"]:
            seen.add(bool(out["tail_section_inverted"]))
    assert seen == {False, True}, seen


# ---------------------------------------------------------------- (e)

def test_the_vertical_relief_removes_exactly_its_share_of_the_lift():
    """35 % of the weight carried by the rig -> 35 % less lift to make.

    The expected value is written from first principles here —
    ``(W - Z) / (1/2 rho V^2 S)`` — with W, Z, rho and S stated or read off
    the built problem, never taken from the solver's expression. The RATIO is
    exact rather than approximate because both runs divide by an identical
    denominator.

    35 % is the MEAN of vlugt2009's five measured kitefoil cases (21-46 %),
    which is why that number and not a round one.
    """
    relief = 0.35 * WEIGHT_N                       # 2100.0 N, exactly
    heavy = hydrotail.HydrofoilTailProblem()
    light = hydrotail.HydrofoilTailProblem(
        rig=RigLoads(z_ce_m=2.0, vertical_n=relief))

    assert heavy.L_required == WEIGHT_N
    assert light.L_required == WEIGHT_N - relief == 3900.0

    for V in (9.0, 12.0, 15.0):
        # EXACT: the target is the independently written closed form, bit for
        # bit, at three speeds spanning the box
        expected = (WEIGHT_N - relief) / (0.5 * light.rho * V * V * light.S)
        assert light.CL_target(V) == expected
        # the RATIO is 0.65 to the last representable place. It is not
        # asserted with `==` because (a/D)/(b/D) is not a/b in binary
        # floating point even when both divisions are correctly rounded —
        # the exact claim is the line above, this is the claim about the size
        # of the relief
        assert light.CL_target(V) / heavy.CL_target(V) \
            == pytest.approx(0.65, rel=1e-15)

    # ...and it reaches the flown design, not just the problem object
    _, lo, hi = _box()
    x = 0.5 * (lo + hi)
    a = hydrotail.evaluate_hydrofoil_tail(x, heavy)
    b = hydrotail.evaluate_hydrofoil_tail(x, light)
    assert a["feasible"] and b["feasible"]
    assert b["CL_target"] / a["CL_target"] == pytest.approx(0.65, rel=1e-15)
    assert b["rig"]["lift_required_n"] == 3900.0
    assert b["rig"]["lift_relief_frac"] == pytest.approx(0.35)


def test_the_relief_travels_through_the_api_flag():
    """The same 35 %, stated the way a shell states it."""
    flags = {api.RIG_CE_HEIGHT_KEY: 2.0,
             api.RIG_VERTICAL_KEY: 0.35 * WEIGHT_N}
    built = api.PROBLEM_SPECS[ANCHOR].build({}, flags, None)
    plain, lo, hi = _box()
    assert built.problem.L_required == 3900.0
    x = 0.5 * (lo + hi)
    assert built.evaluate(x)["CL_target"] \
        / plain.evaluate(x)["CL_target"] == pytest.approx(0.65, rel=1e-15)


def test_a_rig_that_carries_the_whole_craft_is_refused_by_name():
    """Nonsense refused, and nothing else.

    A vertical share AT the weight leaves the foil zero lift to make, which
    is a different craft rather than a hard design. Anything strictly below
    is accepted, priced and flown — the repo's rule that a calibration is a
    default and never a ban.
    """
    with pytest.raises(ValueError, match="vertical_n"):
        hydrotail.HydrofoilTailProblem(
            rig=RigLoads(z_ce_m=2.0, vertical_n=WEIGHT_N))
    with pytest.raises(ValueError, match="vertical_n"):
        hydrotail.HydrofoilTailProblem(
            rig=RigLoads(z_ce_m=2.0, vertical_n=WEIGHT_N * 1.5))
    # 95 % of the weight is a hard design, not a refused one
    hot = hydrotail.HydrofoilTailProblem(
        rig=RigLoads(z_ce_m=2.0, vertical_n=0.95 * WEIGHT_N))
    assert hot.L_required == pytest.approx(300.0)
    # ...and a heavier craft with the same rig is fine too: the refusal is
    # about the pair, not about a ceiling on either number
    hydrotail.HydrofoilTailProblem(
        L_design=20000.0, rig=RigLoads(z_ce_m=2.0, vertical_n=WEIGHT_N))


# ---------------------------------------------------------------- (f)

@pytest.mark.parametrize("cap", [None, 1.2])
def test_an_untrimmable_design_fails_in_contract_with_the_right_width(cap):
    """The failure path: a NAMED reason, no exception, and the SAME margin
    vector — by width AND by value.

    This repo has a recorded bug where a margin was a scalar on success and a
    vector on failure: every width test passed and a constraint was silently
    dropped. So the value contract is asserted too — ``PENALTY`` for the
    score and ``G_FAIL`` in every slot, not merely "an array of the right
    length".

    The design is the box centre with the stabiliser at its smallest, flown
    with the driven rig: the couple then asks for more incidence than the
    stabiliser is allowed, which is a REAL design outcome and exactly what
    the gate is for.
    """
    _, lo, hi = _box()
    labels = api.PROBLEM_SPECS[ANCHOR].param_labels
    x = 0.5 * (lo + hi)
    x[labels.index("S_t_m2")] = lo[labels.index("S_t_m2")]
    prob = hydrotail.HydrofoilTailProblem(rig=RigLoads(**RIGS["driven"]),
                                          draught_max_m=cap)
    width = prob.n_constraints
    assert width == (2 if cap is None else 3)

    out = hydrotail.evaluate_hydrofoil_tail(x, prob)       # must not raise
    assert out["feasible"] is False
    assert out["reason"] == "untrimmable: |i_t| exceeds limit"
    assert abs(out["i_t_deg"]) > prob.i_t_max_deg

    score, g = hydrotail.fg_hydrofoil_tail(x, prob)
    assert score == PENALTY
    assert g.shape == (width,)
    assert np.all(g == hydrotail.G_FAIL)

    # ...and the SUCCESS path returns the same width, on the same problem
    ok_score, ok_g = hydrotail.fg_hydrofoil_tail(0.5 * (lo + hi), prob)
    assert ok_g.shape == g.shape
    assert ok_score != PENALTY
    assert np.all(np.isfinite(ok_g))


def test_a_rig_never_raises_out_of_an_evaluation():
    """Whatever the rig, the failure contract holds: an in-contract failure
    carries a non-empty reason and the penalty pair, and nothing escapes.

    BOTH regimes are counted at the end. The interesting half of this test
    is the ``else`` branch — a failure that returns instead of raising — and
    it is entered only by draws the solver actually refuses (measured over
    this seed: 218 flown and 22 refused across the four rigs, but 0 refused
    for the kite and the windfoil on their own). A future widening that made
    every draw trimmable would leave the failure contract unasserted while
    the test stayed green, which is the same defect as a ``continue`` with
    no guard, written as an ``if``.
    """
    _, lo, hi = _box()
    rng = np.random.default_rng(4242)
    draws = lo + rng.random((60, lo.size)) * (hi - lo)
    flown = refused = 0
    for name, kw in RIGS.items():
        prob = hydrotail.HydrofoilTailProblem(rig=RigLoads(**kw))
        for x in draws:
            out = hydrotail.evaluate_hydrofoil_tail(x, prob)
            score, g = hydrotail.fg_hydrofoil_tail(x, prob)
            assert g.shape == (2,)
            if out["feasible"]:
                flown += 1
                assert score == float(out["LoD"])
            else:
                refused += 1
                assert out["reason"], name
                assert score == PENALTY
                assert np.all(g == hydrotail.G_FAIL)
    assert flown, "no draw flew — the success contract was never asserted"
    assert refused, "no draw was refused — the FAILURE contract, which is "\
                    "what this test is named for, was never asserted"


# ---------------------------------------------------------------- (g)

def test_a_rig_driving_forward_from_above_pitches_the_craft_nose_down():
    """THE SIGN CONVENTION, as physics rather than as bookkeeping.

    A rig pushing the craft FORWARD at a centre of effort ABOVE the water
    makes a NOSE-DOWN couple (negative in the nose-up-positive convention the
    solvers use). The stabiliser sits aft of the CG, so holding the nose up
    means pushing DOWN harder: the trim incidence falls and the stabiliser's
    load falls with it. Sheet in hard, the bow goes down, the tail carries
    more download.

    Run on drake_blog's MEASURED 145 N windfoil drive at zhang2025's MEASURED
    2.0 m rig height, and the magnitude is checked against the lever written
    out here — depth plus rig height, because the couple's other end is the
    resistance at the foil.
    """
    _, lo, hi = _box()
    x = 0.5 * (lo + hi)
    depth = float(x[list(api.PROBLEM_SPECS[ANCHOR].param_labels)
                    .index("depth_m")])

    plain = hydrotail.evaluate_hydrofoil_tail(
        x, hydrotail.HydrofoilTailProblem())
    windy = hydrotail.evaluate_hydrofoil_tail(
        x, hydrotail.HydrofoilTailProblem(rig=RigLoads(**RIGS["windfoil"])))
    assert plain["feasible"] and windy["feasible"]

    # the couple: nose-DOWN, and 145 N times the lever from the rig's centre
    # of effort down to the foil
    assert windy["rig"]["M_rig_nm"] < 0.0
    assert windy["rig"]["lever_m"] == depth + 2.0
    assert windy["rig"]["M_rig_nm"] == pytest.approx(-145.0 * (depth + 2.0),
                                                     rel=1e-12)
    # ...and it is the size the literature quotes for a windfoil, ~399 N.m
    assert 350.0 < abs(windy["rig"]["M_rig_nm"]) < 450.0

    # what the craft does about it: less incidence, less stabiliser lift
    assert windy["i_t_deg"] < plain["i_t_deg"]
    assert windy["CL_stab"] < plain["CL_stab"]
    assert windy["CL_stab_pretrim"] < plain["CL_stab_pretrim"]

    # ...and a bigger drive from the same height goes further, all the way
    # through zero into a download and an inverted section
    hard = hydrotail.evaluate_hydrofoil_tail(
        x, hydrotail.HydrofoilTailProblem(rig=RigLoads(**RIGS["driven"])))
    assert hard["feasible"]
    assert hard["rig"]["M_rig_nm"] < windy["rig"]["M_rig_nm"] < 0.0
    assert hard["i_t_deg"] < windy["i_t_deg"]
    assert plain["CL_stab"] > 0.0 > hard["CL_stab"]
    assert hard["tail_section_inverted"] is True
    assert plain["tail_section_inverted"] is False


def test_a_rig_pulling_from_below_the_waterline_is_refused():
    """``z_ce_m`` is a HEIGHT ABOVE THE WATERLINE, and the refusal says so.

    A frame statement, not a ceiling: 0.0 (a tow rope at the surface) is
    accepted and still makes a couple, because the foil is below the water.
    """
    with pytest.raises(ValueError, match="z_ce_m"):
        RigLoads(z_ce_m=-0.1)
    surface = RigLoads(z_ce_m=0.0, thrust_n=400.0)
    assert surface.lever_m(0.575) == 0.575
    assert surface.pitching_moment_nm(weight_n=WEIGHT_N, depth_m=0.575,
                                      x_cg_m=0.15) == -400.0 * 0.575


# ---------------------------------------------------------------- (h)

def test_the_engines_own_appendage_lod_is_reported_beside_the_stated_one():
    """The loop this feature refuses to close, made visible.

    The drive force is divided by a STATED craft L/D (default 7.0, inside the
    6.1-7.9 towing-tank band [martinez2026] and just above the 6.5 CFD figure
    [backas2016]). This engine's own appendage L/D is 19-25 on the same
    design, because it models a wing and a mast where the measurements are of
    a whole appendage. Closing the loop would understate the dominant
    longitudinal moment by that factor, so the two numbers are printed side
    by side instead.
    """
    _, lo, hi = _box()
    x = 0.5 * (lo + hi)
    out = hydrotail.evaluate_hydrofoil_tail(
        x, hydrotail.HydrofoilTailProblem(rig=RigLoads(z_ce_m=2.0)))
    assert out["feasible"]
    assert out["rig"]["craft_LoD_stated"] == CRAFT_LOD_DEFAULT == 7.0
    assert out["rig"]["appendage_LoD"] == out["LoD"]
    # the gap is real and it is the reason the loop stays open
    assert out["rig"]["appendage_LoD"] > 2.5 * out["rig"]["craft_LoD_stated"]
    # the drive follows the STATED ratio, not the engine's
    assert out["rig"]["thrust_n"] == pytest.approx(WEIGHT_N / 7.0, rel=1e-12)
    assert out["rig"]["thrust_stated"] is False
    # ...and a stated drive wins over the closure
    stated = hydrotail.evaluate_hydrofoil_tail(
        x, hydrotail.HydrofoilTailProblem(
            rig=RigLoads(z_ce_m=2.0, thrust_n=145.0, craft_lod=3.0)))
    assert stated["rig"]["thrust_n"] == 145.0
    assert stated["rig"]["thrust_stated"] is True


def test_the_side_force_is_flown_by_the_strut_and_not_by_the_pitch_plane():
    """A stated side force is CARRIED BY THE MAST and changes nothing in pitch.

    This test used to be called ``..._carried_and_not_flown``, and the second
    half of that name was the finding rather than the design: the rig stated
    a side load, the report printed it, and no surface in the craft resisted
    it. A windfoil's mast IS its daggerboard, so ``strut_model`` gives it the
    leeway angle and the induced drag that carrying the load costs.

    What has NOT changed, and is asserted first because it is the invariant
    the whole file is about: every solver downstream is a symmetric mirrored
    half-model with no sideslip unknown, so a side force still has no
    business in the pitch balance. Same trim, same neutral point, same
    stabiliser load, to the last bit.
    """
    _, lo, hi = _box()
    x = 0.5 * (lo + hi)
    quiet = RigLoads(z_ce_m=2.0, thrust_n=145.0)
    loud = RigLoads(z_ce_m=2.0, thrust_n=145.0, side_n=900.0)
    a = hydrotail.evaluate_hydrofoil_tail(
        x, hydrotail.HydrofoilTailProblem(rig=quiet))
    b = hydrotail.evaluate_hydrofoil_tail(
        x, hydrotail.HydrofoilTailProblem(rig=loud))

    # ---- the pitch plane does not hear it, bit for bit
    for k in ("alpha_rad", "i_t_rad", "x_np", "SM", "CL_stab", "CDi",
              "CDp", "cm_residual"):
        assert a[k] == b[k], k
    assert a["rig"]["M_rig_nm"] == b["rig"]["M_rig_nm"]
    assert b["rig"]["side_n"] == 900.0 and a["rig"]["side_n"] == 0.0

    # ---- ...and the STRUT does, in the only three places it can
    assert a["strut"]["Cy"] == 0.0 and a["strut"]["leeway_deg"] == 0.0
    assert a["strut"]["CDi_side"] == 0.0
    assert b["strut"]["Cy"] > 0.0
    assert 0.0 < b["strut"]["leeway_deg"] < 10.0     # a leeway, not a stall
    assert b["strut"]["CDi_side"] > 0.0
    # the whole difference in drag IS that term: nothing else moved
    assert b["CD"] - a["CD"] == pytest.approx(b["strut"]["CDi_side"],
                                              rel=1e-12)
    assert b["LoD"] < a["LoD"]

    # ---- and on the PUBLISHED charge the old contract still holds exactly:
    # wetted area cannot carry a side load, and does not pretend to
    c = hydrotail.evaluate_hydrofoil_tail(x, _published_htp(rig=quiet))
    d = hydrotail.evaluate_hydrofoil_tail(x, _published_htp(rig=loud))
    assert c["LoD"] == d["LoD"]
    assert "strut" not in c and "strut" not in d


# ---------------------------------------------------------------- (i)

def _tail_water_families() -> list:
    """Every registered water family whose problem IS (or wraps) a
    ``HydrofoilTailProblem``, off the registry.

    Derived, never listed. The water registry is generated — a tip device x
    an arm x a stabiliser depth x how much of the elevator is designed, then
    every chosen-section and free-chord-law twin of all of those — so a
    written list would be stale the day another combination is registered,
    and a stale list is exactly what would stop this test catching the
    silent-drop defect it is about.
    """
    out = []
    for name, spec in api.PROBLEM_SPECS.items():
        if spec.medium != "water":
            continue
        prob = spec.build({}, {}, None).problem
        prob = getattr(prob, "foil", prob)     # the CST composites wrap it
        if isinstance(prob, hydrotail.HydrofoilTailProblem):
            out.append(name)
    return out


def test_every_family_with_a_pitch_balance_declares_and_honours_the_rig():
    """Declared where it is honoured, refused where it is not.

    ``api.check_flags`` exists because a flag accepted and silently dropped is
    worse than one refused. The rig has a home only where there is a second
    surface to trim, so the elevator families take it and the planar and
    winglet ones must not — and both halves are measured off the registry
    rather than asserted from a list.
    """
    tails = _tail_water_families()
    assert len(tails) > 100, len(tails)          # the generated family is big
    probe = {k: 1.0 for k in api.RIG_KEYS}
    probe[api.RIG_CE_HEIGHT_KEY] = 2.0

    for name in tails:
        assert api.unhonoured_flags(name, probe) == [], name
        prob = api.PROBLEM_SPECS[name].build({}, probe, None).problem
        prob = getattr(prob, "foil", prob)
        assert isinstance(prob.rig, RigLoads), name
        assert prob.rig.z_ce_m == 2.0

    others = [n for n, s in api.PROBLEM_SPECS.items()
              if s.medium == "water" and n not in tails]
    assert others, "no water family without a stabiliser — the registry moved"
    for name in others:
        assert api.unhonoured_flags(name, probe) == sorted(api.RIG_KEYS), name
        with pytest.raises(KeyError):
            api.check_flags(name, probe)


def test_no_rig_flags_means_no_rig_everywhere():
    """The absent case, measured across the whole generated family.

    ``rig=None`` is the entire off switch, so this is the assertion that
    every frozen water study still reproduces: not "the couple is zero" but
    "there is no rig".
    """
    for name in _tail_water_families():
        prob = api.PROBLEM_SPECS[name].build({}, {}, None).problem
        prob = getattr(prob, "foil", prob)
        assert prob.rig is None, name
        assert prob.L_required == prob.L_design, name


def test_a_rig_force_without_a_lever_is_refused_with_the_reason_named():
    """The one incompleteness this wiring refuses.

    The couple is the drive times ``(depth + z_ce)``. A thrust with no centre
    of effort height therefore has no moment at all, and published heights
    run from 0 m to ~2.75 m — a range no default can stand in for. It is
    refused where the incompleteness is (the flag layer), not in the value
    object, so the message names the flag rather than a Python argument.
    """
    for key in (api.RIG_THRUST_KEY, api.RIG_VERTICAL_KEY, api.RIG_SIDE_KEY,
                api.RIG_CE_X_KEY, api.RIG_CRAFT_LOD_KEY):
        with pytest.raises(ValueError, match=api.RIG_CE_HEIGHT_KEY):
            api.PROBLEM_SPECS[ANCHOR].build({}, {key: 1.0}, None)
    # ...and the height alone is a complete statement: the drive closes from
    # the craft L/D
    built = api.PROBLEM_SPECS[ANCHOR].build(
        {}, {api.RIG_CE_HEIGHT_KEY: 2.0}, None)
    assert built.problem.rig == RigLoads(z_ce_m=2.0)
    # empty in, empty out
    assert api.PROBLEM_SPECS[ANCHOR].build({}, {}, None).problem.rig is None
    assert api.PROBLEM_SPECS[ANCHOR].build(
        {}, {api.RIG_CE_HEIGHT_KEY: None}, None).problem.rig is None


def test_the_value_object_refuses_nonsense_and_caps_nothing():
    """Refusals name the field and the value; nothing has a ceiling."""
    with pytest.raises(ValueError, match="craft_lod"):
        RigLoads(z_ce_m=2.0, craft_lod=0.0)
    with pytest.raises(ValueError, match="craft_lod"):
        RigLoads(z_ce_m=2.0, craft_lod=-7.0)
    with pytest.raises(ValueError, match="thrust_n"):
        RigLoads(z_ce_m=2.0, thrust_n=float("nan"))
    with pytest.raises(ValueError, match="z_ce_m"):
        RigLoads(z_ce_m=float("inf"))
    with pytest.raises(TypeError):
        hydrotail.HydrofoilTailProblem(rig={"z_ce_m": 2.0})
    # no ceiling anywhere: an implausible rig is a hard design, not a refused
    # input, and the constraints are what price it
    big = RigLoads(z_ce_m=25.0, thrust_n=5000.0, craft_lod=100.0,
                   vertical_n=-4000.0, x_ce_m=12.0)
    assert big.lever_m(0.5) == 25.5
    assert big.lift_required_n(WEIGHT_N) == 10000.0


# ---------------------------------------------------------------- (j)
#
# THE CAVITATION SWEEP READS THE TABLE THAT IS FLOWN, NOT THE ONE THAT WAS
# STATED.
#
# ``evaluate_hydrofoil_tail`` used to choose its cavitation sweep on
# ``prob.polar_tail is None`` — "did the user give the stabiliser a section
# of its own?". That is the wrong question, and the rig is what makes it
# wrong in practice. Between that test and the sweep sits
# ``tail.orient_section``, which MIRRORS the shared section when the load is
# a download; a mirrored section is a second table (``polar.InvertedPolar``:
# ``cp_min(a) = cp_min0(-a)``, and its own docstring says the mirror "is
# what keeps the cavitation constraint honest on an inverted hydrofoil
# surface"). So with one section stated and a rig couple on, the stabiliser
# flew INVERTED and had its margin read off the UPRIGHT table.
#
# The right question is ``pol_t is pol`` — are these two surfaces flying the
# same table — and the defect's own shape is the proof that it is: the
# PROFILE DRAG four lines earlier already used ``pol_t``. One reading off
# the mirrored table, one off the upright one, in the same evaluation.
#
# WHAT MOVES, measured before and after on this tree. No objective anywhere:
# L/D is identical on all 527 flown draws of the three sweeps below, because
# the branch is downstream of the solve and of every drag term. No anchor:
# the stabiliser is upright at both frozen points, so the fix provably
# cannot reach them (asserted below). Only the cavitation CONSTRAINT, and
# only on designs whose stabiliser is mirrored — 34 of 130 flown draws with
# the driven rig, 23 of 199 with the kite rig, and 0 of 198 with no rig at
# all. It is NOT a one-way correction: mirroring moves Cp_min in whichever
# direction the local incidence dictates, so it tightens some designs and
# loosens others (worst tightening -1.19 in margin, largest loosening +0.89
# over the driven sweep), and 4 of those 130 cross the feasibility boundary.

def _cav_margins(out, prob, x) -> tuple:
    """``(g_upright_everywhere, g_as_flown, the_stabiliser_binds)``.

    The cavitation margin closed in full from this file, off the solved
    geometry the run reports, so neither number comes from the branch under
    test. From ``hydrofoil.sigma_cav``'s own derivation:

        sigma(h) = (p_atm + rho g h - p_vap) / (1/2 rho V^2),   h = depth - z
        g_panel  = sigma(h_panel) + Cp_min(alpha_eff_panel)
        g        = min over panels

    with h the panel's OWN submergence (z is up, the free surface at z =
    depth). The two answers differ in one place only: which table the
    STABILISER's panels read ``Cp_min`` from. ``g_upright_everywhere`` is
    what the old branch computed whenever the user stated one section;
    ``g_as_flown`` mirrors the table on the stabiliser exactly when the run
    reports that it flew mirrored.
    """
    from aerobo import polar as _pol
    from aerobo.hydrofoil import G_GRAV, P_ATM

    res = out["vlm"]
    tail = np.asarray(res.is_tail, dtype=bool)
    depth, V = float(out["depth"]), float(out["V"])
    pol = prob.polar_family.at(float(x[list(_labels()).index("tc")]))
    alpha = np.rad2deg(np.asarray(res.alpha_eff, dtype=float))
    sigma = ((P_ATM + prob.rho * G_GRAV * (depth - np.asarray(res.z, float))
              - prob.p_vap) / (0.5 * prob.rho * V * V))

    upright = sigma + pol.cp_min(alpha)
    flown = upright.copy()
    if out["tail_section_inverted"]:
        flown[tail] = sigma[tail] + _pol.inverted(pol).cp_min(alpha[tail])
    return float(upright.min()), float(flown.min()), \
        bool(tail[int(np.argmin(flown))])


def _labels():
    return api.PROBLEM_SPECS[ANCHOR].param_labels


#: A design that the defect gets the VERDICT wrong on, stated in round
#: numbers on ``ANCHOR``'s own box: a sharply tapered 10 % foil at 0.80 m
#: and the top of the published speed band, with a big stabiliser on a 1 m
#: arm. Flown with the driven rig the stabiliser carries a download
#: (CL_stab -0.20), so it mounts inverted; the upright table then says
#: +0.0183 (cavitation-free) and its own says -0.0701 (cavitating).
FLIPPED = np.array([0.25, -2.0, -3.0, 0.10, 0.80, 16.0, 0.06, 1.0])


def test_the_cavitation_sweep_reads_the_stabilisers_own_mirrored_table():
    """THE DEFECT, on a design whose FEASIBILITY it decides.

    One section stated (``polar_tail is None``), two tables flown. The
    margin asserted here is the one closed from the MIRRORED table in
    :func:`_cav_margins`, and the upright answer is asserted to be a
    different number on the other side of zero — a test that only checked
    "the margin is some number" or that the two agreed would prove nothing.
    """
    prob = hydrotail.HydrofoilTailProblem(rig=RigLoads(**RIGS["driven"]))
    out = hydrotail.evaluate_hydrofoil_tail(FLIPPED, prob)
    assert out["feasible"], out["reason"]

    # the premise: ONE section stated, and the stabiliser flying mirrored
    assert prob.polar_tail is None
    assert out["tail_section_inverted"] is True
    assert out["CL_stab"] < 0.0                    # ...because it pushes down

    upright, flown, stab_binds = _cav_margins(out, prob, FLIPPED)

    # the two readings are genuinely different, and not by a rounding
    assert upright != flown
    assert abs(upright - flown) > 0.08
    # ...and they disagree about whether this foil cavitates at all
    assert upright > 0.0 > flown

    # the engine reports the one it FLEW
    assert out["g_cav"] == pytest.approx(flown, abs=1e-12)
    assert out["g_cav"] != pytest.approx(upright, abs=1e-3)
    # and the constraint vector carries it, so a run refuses this design
    assert out["g"][0] == out["g_cav"]
    assert stab_binds and out["cav_surface_worst"] == "stabiliser"

    # the report names both surfaces now that there are two tables, and the
    # FOIL's own margin is exactly what the old branch used to publish as
    # the whole assembly's — i.e. the bug was reporting the non-binding
    # surface's number
    assert out["g_cav_stab"] == out["g_cav"]
    assert out["g_cav_foil"] == pytest.approx(upright, abs=1e-12)


def test_the_drag_and_the_cavitation_read_the_same_table():
    """The defect's shape, as one assertion: two readings, one section.

    ``CDp_stab`` has always been summed off ``pol_t`` — the mirrored table —
    so on this design the profile drag knew the stabiliser was inverted
    while the cavitation margin did not. Asserted as an OUTCOME: the
    stabiliser's profile drag matches the mirrored table's ``cd`` and not
    the upright one's, on the same panels whose ``cp_min`` decided the
    margin above.
    """
    from aerobo import polar as _pol

    prob = hydrotail.HydrofoilTailProblem(rig=RigLoads(**RIGS["driven"]))
    out = hydrotail.evaluate_hydrofoil_tail(FLIPPED, prob)
    assert out["feasible"] and out["tail_section_inverted"] is True

    res = out["vlm"]
    tail = np.asarray(res.is_tail, dtype=bool)
    alpha = np.rad2deg(np.asarray(res.alpha_eff, dtype=float))
    pol = prob.polar_family.at(float(FLIPPED[list(_labels()).index("tc")]))
    strip = np.asarray(res.c, float) * np.asarray(res.width, float)

    up = float(np.sum((pol.cd(alpha) * strip)[tail]) / res.S)
    inv = float(np.sum((_pol.inverted(pol).cd(alpha) * strip)[tail]) / res.S)
    assert up != inv                               # the tables really differ
    assert out["CDp_stab"] == pytest.approx(inv, rel=1e-12)
    assert out["CDp_stab"] != pytest.approx(up, rel=1e-6)


@pytest.mark.parametrize("rig_name", ["driven", "kite", None])
def test_the_margin_follows_the_mirror_over_the_whole_box(rig_name):
    """200 seeded draws per rig, and the contract has three halves.

    * the engine's margin equals the AS-FLOWN closed form on every flown
      draw — so the branch is right everywhere, not at one lucky point;
    * it differs from the upright-everywhere closed form on a substantial
      number of them — so the fix is doing work rather than agreeing with
      what it replaced;
    * with NO rig the stabiliser never mirrors and the two closed forms are
      identical on every draw — which is the assertion that the change
      cannot have touched any published run.
    """
    _, lo, hi = _box()
    rng = np.random.default_rng(20260810)
    draws = lo + rng.random((200, lo.size)) * (hi - lo)
    prob = hydrotail.HydrofoilTailProblem(
        rig=None if rig_name is None else RigLoads(**RIGS[rig_name]))

    flown = differ = mirrored = 0
    for x in draws:
        out = hydrotail.evaluate_hydrofoil_tail(x, prob)
        if not out["feasible"]:
            continue
        flown += 1
        mirrored += bool(out["tail_section_inverted"])
        upright, as_flown, _ = _cav_margins(out, prob, x)
        assert out["g_cav"] == pytest.approx(as_flown, abs=1e-12)
        if upright != as_flown:
            differ += 1
            # a reading can only move where the table moved
            assert out["tail_section_inverted"], x

    assert flown > 100, flown
    if rig_name is None:
        assert mirrored == 0 and differ == 0, (mirrored, differ)
    else:
        assert mirrored > 20, mirrored
        assert differ > 10, differ


def test_the_frozen_anchors_are_out_of_this_fixs_reach_by_construction():
    """WHY neither anchor moved, rather than that it did not.

    ``test_no_rig_reproduces_the_centre_anchor_bit_for_bit`` already pins
    the L/D. This pins the MECHANISM: at both frozen points the stabiliser
    lifts, so nothing is mirrored, so ``pol_t is pol`` and the old branch
    and the new one are the same line of code. The published cavitation
    margins are asserted here for the first time, as the frozen numbers they
    are.
    """
    built, lo, hi = _box()
    for tag, x, g_cav in (("centre", 0.5 * (lo + hi), 0.47548809330520014),
                          ("off-centre", lo + 0.75 * (hi - lo),
                           0.22448980568883625)):
        out = built.evaluate(x)
        assert out["tail_section_inverted"] is False, tag
        assert out["CL_stab"] > 0.0, tag
        assert out["g_cav"] == g_cav, tag
        assert "cav_surface_worst" not in out, tag   # one table, no question
        upright, as_flown, _ = _cav_margins(out, built.problem, x)
        assert upright == as_flown == pytest.approx(g_cav, abs=1e-12), tag


# ---------------------------------------------------------------- (k)
#
# "UNTOUCHED BY CONSTRUCTION" IS TRUE AT FIXED GEOMETRY AND FALSE END TO END.
#
# ``rig.py`` and ``hydrotail.py`` both used to say flatly that the couple
# leaves the Jacobian, ``neutral_point()`` and the static margin untouched by
# construction. Of ``cm_ac`` that is exactly right and test (c) above asserts
# it with ``==``. End to end it is measurably wrong, because the couple flips
# the SIGN of the stabiliser's load and two rules in this engine follow that
# sign and are GEOMETRY: the section's mounting, and — where the family asks
# for it — which side the stabiliser's tip device goes.
#
# The docstrings now say so and quote these numbers. This is the tripwire on
# them (a published number in this repo has gone stale inside the session
# that measured it), and it is also the assertion that the narrowed claim is
# true rather than merely more cautious: pin the mounting and the movement is
# EXACTLY zero, leave it free and it is not.

#: the two paths, as measured on this tree with the driven rig, in metres of
#: neutral-point travel at the box centre. Quoted verbatim in
#: ``hydrotail``'s and ``rig``'s module docstrings.
X_NP_MOUNTING_ONLY = (0.1221206762291545, 0.12225796485175514)
X_NP_WITH_FOLLOWING_DEVICE = (0.14219956586471855, 0.1432741908779214)

#: the family whose stabiliser tip device follows its load, and the flags
#: that turn the following on. ``follow`` is a DIRECTION, not a type: the
#: band is the magnitude and the side is derived per candidate.
FOLLOW_FAMILY = "hydrofoil + elevator [designed elevator + tip device]"


def test_the_couple_moves_x_np_only_by_changing_the_shape():
    """The mounting path, both frozen design points, and the control.

    Free mounting: the driven rig turns the stabiliser over, the mirrored
    ``alpha_L0`` goes into the panel normals and the neutral point moves.
    Pinned mounting: the SAME rig, the same couple, and ``x_np`` is
    bit-identical — which is what makes the first half a statement about
    geometry rather than about the moment.

    BOTH halves are counted. ``moved`` was already here; ``controlled`` was
    not, and the control is the half that carries the claim: it runs under
    an ``if both arms fly`` that is true at exactly ONE of the two points
    today (the box centre is untrimmable for the driven rig once the
    stabiliser is held upright against it, measured). A change that made it
    untrimmable at the off-centre point too would have left this test green
    with the control never executed — the same shape as
    ``test_the_reported_couple_is_the_couple_the_balance_says``'s missing
    guard, one indentation level in.
    """
    _, lo, hi = _box()
    driven = RigLoads(**RIGS["driven"])
    moved = controlled = 0
    for tag, x in (("centre", 0.5 * (lo + hi)),
                   ("off-centre", lo + 0.75 * (hi - lo))):
        plain = hydrotail.evaluate_hydrofoil_tail(
            x, hydrotail.HydrofoilTailProblem())
        rigged = hydrotail.evaluate_hydrofoil_tail(
            x, hydrotail.HydrofoilTailProblem(rig=driven))
        assert plain["feasible"] and rigged["feasible"], tag

        # the shape changed: upright without the rig, mirrored with it
        assert plain["tail_section_inverted"] is False, tag
        assert rigged["tail_section_inverted"] is True, tag
        # ...and so did the neutral point, by ~0.1 % of the mean chord
        assert rigged["x_np"] != plain["x_np"], tag
        travel = abs(rigged["x_np"] - plain["x_np"]) / plain["mac"]
        assert 0.0009 < travel < 0.0013, (tag, travel)
        moved += 1

        # THE CONTROL: pin the mounting and the same couple moves nothing.
        # (Only where both arms fly — the box centre is untrimmable for the
        # driven rig once the stabiliser is held upright against it.)
        a = hydrotail.evaluate_hydrofoil_tail(
            x, hydrotail.HydrofoilTailProblem(tail_inverted=False))
        b = hydrotail.evaluate_hydrofoil_tail(
            x, hydrotail.HydrofoilTailProblem(rig=driven, tail_inverted=False))
        if a["feasible"] and b["feasible"]:
            controlled += 1
            assert b["x_np"] == a["x_np"], tag         # EXACTLY, not approx
            assert b["SM"] == a["SM"], tag
            assert abs(b["i_t_rad"] - a["i_t_rad"]) > 1e-6, tag
    assert moved == 2
    assert controlled, "the pinned-mounting CONTROL ran at neither point"

    # the docstrings' own two numbers, at the centre
    x = 0.5 * (lo + hi)
    was, now = X_NP_MOUNTING_ONLY
    assert hydrotail.evaluate_hydrofoil_tail(
        x, _published_htp())["x_np"] == pytest.approx(was, abs=1e-12)
    assert hydrotail.evaluate_hydrofoil_tail(
        x, _published_htp(rig=driven))["x_np"] \
        == pytest.approx(now, abs=1e-12)


def test_a_following_tip_device_moves_x_np_an_order_of_magnitude_further():
    """The second, larger path — the one the docstrings had to name.

    On a family whose stabiliser carries a tip device that FOLLOWS its load,
    the rig does not merely mirror a section: the device swaps sides and the
    whole surface is re-flown before ``neutral_point()`` is read. Measured
    here against the mounting-only travel above, which is the comparison the
    docstrings make.
    """
    from aerobo import wingtail

    base = {api.TAIL_WINGLET_TYPE_KEY: "canted",
            api.TAIL_WINGLET_DIR_KEY: wingtail.FOLLOW_DIRECTION,
            **PUBLISHED_PHYSICS}
    free = api.PROBLEM_SPECS[FOLLOW_FAMILY].build({}, dict(base), None)
    rigged = api.PROBLEM_SPECS[FOLLOW_FAMILY].build(
        {}, {**base, api.RIG_CE_HEIGHT_KEY: 2.0}, None)
    assert free.problem.tail_winglet_follow is True

    b = np.asarray(free.bounds, dtype=float)
    x = 0.5 * (b[:, 0] + b[:, 1])
    a, c = free.evaluate(x), rigged.evaluate(x)
    assert a["feasible"] and c["feasible"]

    # the device really did swap sides, which is the geometry change
    assert a["tail_winglet"]["side"] == "up"
    assert c["tail_winglet"]["side"] == "down"
    assert a["CL_stab"] > 0.0 > c["CL_stab"]

    was, now = X_NP_WITH_FOLLOWING_DEVICE
    assert a["x_np"] == pytest.approx(was, abs=1e-12)
    assert c["x_np"] == pytest.approx(now, abs=1e-12)
    travel = abs(c["x_np"] - a["x_np"]) / a["mac"]
    assert 0.008 < travel < 0.010                      # ~0.88 % MAC

    # ...and it is the ORDER OF MAGNITUDE claim the docstrings make: this
    # path is worth several times the mounting alone
    _, lo, hi = _box()
    xm = 0.5 * (lo + hi)
    mount_travel = abs(
        hydrotail.evaluate_hydrofoil_tail(
            xm, hydrotail.HydrofoilTailProblem(
                rig=RigLoads(**RIGS["driven"])))["x_np"]
        - hydrotail.evaluate_hydrofoil_tail(
            xm, hydrotail.HydrofoilTailProblem())["x_np"]) / 0.1225
    assert travel > 5.0 * mount_travel, (travel, mount_travel)
