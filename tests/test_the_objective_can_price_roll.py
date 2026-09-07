"""L/D and stability, weighted — and the lateral half measured where it is true.

The ask was an objective the user can weight between L/D and stability.
Stability is two different questions and this file pins both halves:

* PITCH — ``stab``, the static margin. Free: every wing+tail family already
  computes it, because it is the constraint the run is gated on. Weighting
  it buys margin ABOVE that floor (tests/test_wing_composite.py).
* ROLL — ``spiral``, the spiral criterion ``Cl_beta*Cn_r - Cn_beta*Cl_r``.
  Not free, and not previously computable inside an evaluation at all.

Two things had to be true before roll could be an objective term, and they
are what this file asserts.

**It must be measured on the aeroplane that was SCORED.** The obvious source
was ``flightmodel.build_flight_model``, which stages 5 and 6 already use —
and at the time this file was written it DROPPED THE TIP DEVICE: the rebuild
passed no winglet argument to its ``VLM``, so ``Cl_beta`` came back
bit-identical for every winglet height from 0.00 to 0.15 while L/D moved 5 %.
Scoring roll off that would have been scoring an aeroplane the design is not,
which is why the criterion is computed inside the evaluation instead.

That defect has SINCE BEEN FIXED (the rebuild now reads
``geometry["winglet"]``), so stages 5 and 6 are trustworthy about roll again
— see ``test_the_flight_rebuild_SEES_the_device_it_was_scored_with`` below,
which replaced the test that pinned it. The criterion still belongs in the
evaluation: measuring it there is what makes it the SCORED aeroplane by
construction rather than by a rebuild agreeing.

**It must cost nothing it is not asked for.** The fin has to be in the
lattice for a sideslip to push on, and the lattice charges profile drag by
panel area — so simply adding it moved CDp +5.48 % and L/D -2.7 %. With the
vertical masked out of the horizontal book-keeping the objective is
untouched BIT-FOR-BIT, and the fin's drag keeps its single author, the
opt-in Raymer ``cd0_fin``.

**What the measurement then said, and it is not what V5 concluded.**
``PLAN_SESSION66_V5.md`` reports "three degrees of dihedral buys a
convergent spiral for 0.22 % of L/D", measured through the rebuild. On the
design as scored, with its tip device in the lattice, that device carries
88 % of ``Cl_beta`` and the spiral is already convergent past
``winglet_h_frac`` 0.095 — which GAINS L/D rather than costing it. The
dihedral is not wrong, it is just not the cheapest lever, and the reason
V5 could not see that is the rebuild defect above. (V5's plan and handover
now carry the correction, and on the fixed rebuild the crossing measures at
h_frac 0.0755 for +3.2 % of L/D.)
"""
from __future__ import annotations

import numpy as np
import pytest

from aerobo import api, wing_score as wsc
from aerobo.dynamics import spiral_margin, spiral_margin_of
from aerobo.flightmodel import build_flight_model

NAME = "tail + winglet"


def _built(lateral: bool):
    b = api.PROBLEM_SPECS[NAME].build({}, {}, None)
    b.problem.lateral = lateral
    return b


# --------------------------------------------- it costs the objective nothing

#: The solver's own floating-point noise, in ulps of the quantity compared.
#:
#: MEASURED, and it is NOT this module's doing: the same problem object at
#: the same point, evaluated 60 times with the deck OFF, returns 3 distinct
#: scores spanning 2 ulp under threaded BLAS — and exactly 1 value, 60 times
#: out of 60, with OMP_NUM_THREADS=1. The lattice solve is a dense linear
#: system, and a threaded BLAS does not fix the reduction order, so the last
#: bit of the answer depends on how the work happened to be split.
#:
#: Consequence well beyond this file: every exact-float pin in this suite is
#: a coin flip at the last digit unless the runner is single-threaded. A
#: claim of "bit-for-bit" measured once is a claim about one draw from this
#: distribution.
SOLVER_ULP_NOISE = 4


def _ulps(a: float, b: float) -> float:
    import math
    return abs(b - a) / math.ulp(abs(a)) if a else 0.0


def test_the_lateral_deck_does_not_move_the_score_at_all():
    """The objective is untouched to within the solver's own noise floor.

    An earlier version of this asserted exact equality and said "not
    ``approx``". That was wrong, and measuring it is what showed why: the
    baseline jitters by up to 2 ulp on its own (:data:`SOLVER_ULP_NOISE`),
    so exact equality here was testing the BLAS thread schedule, not this
    module. What is true and what matters is that the deck moves the score
    by no more than that floor — which is what lets the criterion be
    offered without re-publishing every stored L/D.
    """
    # ONE problem, toggled — not two builds. The claim is that turning the
    # deck on does not change the score, and comparing two separately built
    # problems would fold in any build-to-build variation as well, which is
    # a different (and much weaker) statement.
    built = _built(False)
    x = built.bounds.mean(axis=1)
    a = built.evaluate(x)
    built.problem.lateral = True
    b = built.evaluate(x)
    assert "spiral_margin" in b and "spiral_margin" not in a
    for key in ("score", "LoD", "CD", "CDi", "CDp", "CDp_wing", "CDp_tail",
                "alpha_deg"):
        moved = _ulps(a[key], b[key])
        assert moved <= SOLVER_ULP_NOISE, (
            f"{key}: {a[key]!r} -> {b[key]!r} ({moved:.0f} ulp, floor "
            f"{SOLVER_ULP_NOISE})")


def test_the_fin_is_not_charged_twice():
    """The vertical's panels must stay out of the strip profile-drag sum.

    Before the mask, asking for the deck moved CDp by +5.48 % — a design
    silently paying for a fin through the lattice while ``cd0_fin``, the
    one author of that charge, still read 0.0.
    """
    on = _built(True)
    out = on.evaluate(on.bounds.mean(axis=1))
    # the fin is charged ONCE, through cd0_fin — the lattice panels that
    # carry it must stay out of the profile sum, or asking for the deck
    # would charge the same surface a second time
    assert out["cd0_fin"] > 0.0
    assert out["CDp"] == pytest.approx(out["CDp_wing"] + out["CDp_tail"],
                                       rel=1e-12)


def test_the_deck_only_runs_when_it_is_weighted():
    """The cost is opted into by asking for the criterion, and by nothing
    else — there is no second switch to keep in step with the weight."""
    out = _built(False).evaluate(_built(False).bounds.mean(axis=1))
    assert "spiral_margin" not in out
    on = _built(True)
    assert "spiral_margin" in on.evaluate(on.bounds.mean(axis=1))


def test_the_weight_is_what_arms_it_through_the_api():
    plain = api.PROBLEM_SPECS[NAME].build({}, {}, None)
    assert plain.problem.lateral is False
    payload = api.wing_score_reference(
        NAME, n=32, seed=0,
        flags={"wing_objective": "composite",
               "wing_score_weights": {"lod": 0.7, "spiral": 0.3}})
    assert "spiral" in payload["bounds"], \
        "the BAND must be measured on the same aeroplane the run scores"
    armed = api.PROBLEM_SPECS[NAME].build(
        {}, {"wing_objective": "composite",
             "wing_score_weights": {"lod": 0.7, "spiral": 0.3},
             "wing_score_reference": payload}, None)
    assert armed.problem.lateral is True
    unarmed = api.PROBLEM_SPECS[NAME].build(
        {}, {"wing_objective": "composite",
             "wing_score_weights": {"lod": 1.0},
             "wing_score_reference": payload}, None)
    assert unarmed.problem.lateral is False


# ------------------------------------------------- the criterion is a criterion

def test_the_criterion_saturates_at_the_physical_boundary():
    """min(margin, 0): divergence is scored, convergence ties.

    Unsaturated this term is a RATCHET — its gradient never turns over, so
    any appreciable weight drives the design to whichever bound buys the
    most margin. Clipped, it buys a convergent spiral and stops paying.
    """
    on = _built(True)
    x = on.bounds.mean(axis=1)
    raw = on.evaluate(x)
    m = wsc.design_metrics(raw, on.problem, {})
    assert m["spiral"] == pytest.approx(min(raw["spiral_margin"], 0.0))

    fake = dict(raw)
    fake["spiral_margin"] = +0.05
    assert wsc.design_metrics(fake, on.problem, {})["spiral"] == 0.0
    fake["spiral_margin"] = +999.0
    assert wsc.design_metrics(fake, on.problem, {})["spiral"] == 0.0, \
        "an arbitrarily convergent design must not outscore a convergent one"


def test_a_family_that_took_no_deck_is_refused_not_zeroed():
    off = _built(False)
    raw = off.evaluate(off.bounds.mean(axis=1))
    assert wsc.design_metrics(raw, off.problem, {})["spiral"] is None


def test_the_margin_is_the_engines_formula_not_the_shells():
    """``spiral_margin`` moved from gui/v4/modes.py into dynamics.py when an
    objective started scoring designs on it. One definition, two callers.

    And the formula is the one WITH the trim attitude in it: the classical
    ``theta0 = 0`` form is available from the same function and is a
    different number on this design, which is asserted here so that scoring
    the optimistic one again is a test failure and not a silent regression.
    """
    from gui.v4 import modes
    assert modes.spiral_margin is spiral_margin
    on = _built(True)
    raw = on.evaluate(on.bounds.mean(axis=1))
    t0 = np.deg2rad(raw["spiral_theta0_deg"])
    assert raw["spiral_margin"] == spiral_margin(
        Cl_beta=raw["Cl_beta"], Cn_r=raw["Cn_r"],
        Cn_beta=raw["Cn_beta"], Cl_r=raw["Cl_r"],
        Cn_p=raw["Cn_p"], Cl_p=raw["Cl_p"], theta0_rad=t0)
    # the attitude is the design's own trim, not a constant
    assert raw["spiral_theta0_deg"] == pytest.approx(raw["alpha_deg"])
    classical = spiral_margin(
        Cl_beta=raw["Cl_beta"], Cn_r=raw["Cn_r"],
        Cn_beta=raw["Cn_beta"], Cl_r=raw["Cl_r"])
    assert classical > raw["spiral_margin"], (
        "the classical form must be the OPTIMISTIC one — it drops "
        "tan(theta0)(Cl_beta.Cn_p - Cn_beta.Cl_p), and theta0 is nose-up")
    assert abs(classical - raw["spiral_margin"]) > 0.002, (
        f"the trim term is not being carried: {classical:+.6f} vs "
        f"{raw['spiral_margin']:+.6f}")


# ------------------------------------------------- what the measurement said

def test_the_tip_device_is_where_the_roll_stiffness_comes_from():
    """And therefore why the rebuild cannot be the source.

    MEASURED: Cl_beta -0.014869 with no device, -0.123190 at h_frac 0.15 —
    the device carries 88 % of it. The spiral crosses to convergent near
    h_frac 0.095, GAINING L/D on the way (32.8085 -> 34.0122 by 0.09).
    """
    on = _built(True)
    labels = list(api.PROBLEM_SPECS[NAME].param_labels)
    i = labels.index("winglet_h_frac")
    x = on.bounds.mean(axis=1)

    def at(h):
        y = x.copy(); y[i] = h
        return on.evaluate(y)

    bare, full = at(0.0), at(0.15)
    assert bare["spiral_margin"] < 0.0, "the bare wing must diverge"
    assert full["spiral_margin"] > 0.0, "the device must fix it"
    share = 1.0 - abs(bare["Cl_beta"]) / abs(full["Cl_beta"])
    assert share > 0.8, f"the device carries only {share:.0%} of Cl_beta"
    # ...and it is not bought with L/D: it is GAINED
    assert full["LoD"] > bare["LoD"]

    hs = np.linspace(0.0, 0.15, 11)
    margins = [at(h)["spiral_margin"] for h in hs]
    assert all(b > a for a, b in zip(margins, margins[1:])), \
        "monotone in the device height, or the crossing is not a crossing"


def test_the_flight_rebuild_SEES_the_device_it_was_scored_with():
    """Replaces `test_the_flight_rebuild_cannot_see_the_device_it_was_scored_with`.

    That test pinned a known, unfixed limitation — the rebuild's ``Cl_beta``
    was bit-identical across the whole tip-device range — and said in as many
    words: *"If this test ever fails because the values started moving, the
    rebuild has been taught about the device and the roll numbers in stages 5
    and 6 became trustworthy — delete it and say so."*

    Saying so. The cause was in ``flightmodel.build_flight_model``, which
    passed no winglet arguments to the VLM at all, so every rebuilt lattice
    had zero device panels whatever the report stated. It now reads
    ``geometry["winglet"]``. Measured after, at 45 m/s:

        h_frac 0.00  Cl_beta -0.0147156   0 device panels
        h_frac 0.05  Cl_beta -0.0404960  16
        h_frac 0.10  Cl_beta -0.0795236  16
        h_frac 0.15  Cl_beta -0.1221607  16

    So the roll numbers in stages 5 and 6 are now about the aeroplane that
    was scored, and the finding this file exists for — that the device is the
    cheaper roll lever — is visible from them rather than only from the
    objective.
    """
    labels = list(api.PROBLEM_SPECS[NAME].param_labels)
    i = labels.index("winglet_h_frac")
    cfg = api.RunConfig(problem_name=NAME, budget=4, seed=0, flags={})
    plain = api.PROBLEM_SPECS[NAME].build({}, {}, None)
    x = plain.bounds.mean(axis=1)

    seen, lods, panels = [], [], []
    for h in (0.0, 0.05, 0.10, 0.15):
        y = x.copy(); y[i] = h
        rep = api.design_report(cfg, y)
        fm = build_flight_model(rep)
        seen.append(float(fm.deck.Cl_beta))
        panels.append(int(fm.model.is_winglet.sum()))
        lods.append(plain.evaluate(y)["score"])

    # the device is IN the flown lattice, and only where the design has one
    assert panels[0] == 0 and all(p > 0 for p in panels[1:]), panels
    # ...and the dihedral effect moves with it, strongly and monotonically
    spread = (max(seen) - min(seen)) / abs(np.mean(seen))
    assert spread > 1.0, (
        f"the rebuild still barely sees the device (spread {spread:.2e})")
    assert seen == sorted(seen, reverse=True), (
        f"a taller device must make Cl_beta MORE negative: {seen}")
    # ...while the scored L/D moves too — both halves of the same design
    assert (max(lods) - min(lods)) / min(lods) > 0.04


def test_the_two_instruments_now_agree_on_the_spiral_margin():
    """Replaces `test_the_spiral_margin_is_not_resolvable_between_the_two_instruments`.

    That test asserted the two paths could NOT be reconciled, and said in its
    own failure message: *"Cl_r now agrees — the two instruments have
    converged and this test should become an equality."* They have, so it is.

    What was wrong was never lateral. ``build_flight_model`` was reading the
    wing's twist from the BREAKDOWN, where it is not — it lives in
    ``geometry`` — so every rebuild flew an UNTWISTED wing against a design
    carrying 2 deg of washout. That is a pure zero-lift offset, which is why
    the three shape derivatives agreed all along and only ``Cl_r``, the one
    proportional to base loading, did not.

        rebuilt zero-lift alpha   -1.6789 deg -> -0.8095 deg
        scored                    -0.8091 deg
        Cl_r ratio                 1.2411    ->  0.9986

    Measured after, at the box centre:

        Cl_beta 1.0048   Cn_beta 0.9903   Cn_r 0.9910   Cl_r 0.9986
        spiral margin  scored -0.00219705, rebuilt -0.00211837  (3.6 %)

    The bounds stay loose on purpose. These are two independent codes — a
    coupled lifting-line/lattice objective and a rebuilt lattice — and the
    margin is a DIFFERENCE of two products, so a few per cent on any factor
    is amplified near the zero. Agreement to a few per cent is the right
    claim; equality is not, and a crossing point still should not be quoted
    to three figures by either instrument.
    """
    on = _built(True)
    x = on.bounds.mean(axis=1)
    scored = on.evaluate(x)
    cfg = api.RunConfig(problem_name=NAME, budget=4, seed=0, flags={})
    deck = build_flight_model(api.design_report(cfg, x)).deck

    for key in ("Cl_beta", "Cn_beta", "Cn_r", "Cl_r"):
        ratio = float(getattr(deck, key)) / scored[key]
        assert 0.95 < ratio < 1.05, f"{key} ratio {ratio:.4f}"

    rebuilt_margin = spiral_margin_of(deck)
    scored_margin = scored["spiral_margin"]
    assert np.sign(rebuilt_margin) == np.sign(scored_margin), (
        "the two instruments disagree about whether the spiral converges")
    assert abs(rebuilt_margin - scored_margin) < 0.15 * abs(scored_margin), (
        f"margins {scored_margin:+.8f} vs {rebuilt_margin:+.8f}")


def test_the_solver_noise_floor_is_real_and_is_the_BLAS():
    """Pins the diagnosis, because it licenses every tolerance above.

    If this ever fails with ONE distinct value under threads, the solve has
    become deterministic and the tolerances in this file can be tightened
    back to equality. If it fails with several under a single thread, the
    non-determinism is NOT the BLAS and the diagnosis needs redoing.
    """
    import os
    built = api.PROBLEM_SPECS[NAME].build({}, {}, None)
    x = built.bounds.mean(axis=1)
    vals = {built.evaluate(x)["score"] for _ in range(40)}
    if len(vals) == 1:
        pytest.skip("this runner is single-threaded (%s); nothing to measure"
                    % os.environ.get("OMP_NUM_THREADS", "unset"))
    spread = _ulps(min(vals), max(vals))
    assert spread <= SOLVER_ULP_NOISE, (
        f"the solver's noise floor is now {spread:.0f} ulp, above the "
        f"{SOLVER_ULP_NOISE} this file's tolerances assume")
