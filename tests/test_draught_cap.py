"""The draught cap: the water answer to the air families' span cap.

Session 48 (registry track), `PREREG_SESSION48_REGISTRY.md` S48-1, report
§17.10. Session 47 measured that a *lateral* span cap is EVEN in the tip
device's cant while the water trade is ODD in it, and argued the analogue is a
cap on DRAUGHT — vertical reach below the free surface.

What these tests hold:

* **nothing published moved** — uncapped is bit-for-bit what it was, and the
  margin vector stays width 1 (winglet) / 2 (tail);
* the cap measures the **deepest panel of the solved geometry**, not
  `h·sin(cant)`, so a blended device is priced correctly;
* it is **odd in cant** — canting up adds no draught at all, canting down
  adds it — which is the property a span cap does not have;
* it is **not a bound on `depth_m`** (the pre-registered kill criterion);
* an unsatisfiable cap is **refused at config time**, not searched for.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aerobo import api, hydrofoil, hydrofoil_section, hydrotail  # noqa: E402

#: The mid-box design of the uncapped winglet family, and its frozen score.
#: NOTE this is a BOX-CENTRE anchor and the centre cant is 0.0, so it is blind
#: to anything that only affects canted designs — fine as a reproducibility
#: anchor, useless as a discriminating probe. The off-centre anchor below is
#: the one that would catch a change to the cant path.
FROZEN_WINGLET_MIDBOX_LOD = 29.76726343609261
#: (cant −30 deg, h_frac 0.15, depth 0.25) — off centre in all three.
FROZEN_WINGLET_CANTED_LOD = 31.070885529670562
FROZEN_WINGLET_CANTED_DRAUGHT_M = 0.2945676688090727


def _x(prob, **over):
    labels = list(prob.param_labels)
    x = np.array([0.5 * (lo + hi) for lo, hi in prob.bounds])
    for k, v in over.items():
        x[labels.index(k)] = v
    return x


# ------------------------------------------------ 1. nothing published moved

def test_uncapped_is_bit_for_bit_and_still_a_scalar_margin():
    prob = hydrofoil.HydrofoilWingletProblem()
    assert prob.draught_max_m is None
    assert prob.n_constraints == 1
    assert prob.constraint_labels == ("cavitation margin",)
    f, g = hydrofoil.fg_hydrofoil_winglet(_x(prob), prob)
    assert np.isscalar(g) or np.ndim(g) == 0
    assert f == FROZEN_WINGLET_MIDBOX_LOD

    # ...and the anchor that is NOT at the box centre, so a change confined to
    # the canted path cannot slip past the one above (the centre cant is 0.0).
    out = hydrofoil.evaluate_hydrofoil_winglet(
        _x(prob, winglet_cant_deg=-30.0, winglet_h_frac=0.15, depth_m=0.25),
        prob)
    assert out["LoD"] == FROZEN_WINGLET_CANTED_LOD
    assert out["draught_m"] == FROZEN_WINGLET_CANTED_DRAUGHT_M


def test_the_cap_does_not_move_the_objective():
    """It is a CONSTRAINT. The same design must score the same either way."""
    a = hydrofoil.HydrofoilWingletProblem()
    b = hydrofoil.HydrofoilWingletProblem(draught_max_m=0.30)
    x = _x(a, winglet_cant_deg=-30.0)
    fa, _ = hydrofoil.fg_hydrofoil_winglet(x, a)
    fb, _ = hydrofoil.fg_hydrofoil_winglet(x, b)
    assert fa == fb


def test_the_tail_family_keeps_its_two_margins_uncapped():
    prob = hydrotail.HydrofoilTailProblem()
    assert prob.n_constraints == 2
    assert prob.constraint_labels == ("cavitation margin",
                                      "static margin - SM_min")
    _, g = hydrotail.fg_hydrofoil_tail(_x(prob), prob)
    assert np.asarray(g).shape == (2,)


# --------------------------------------------- 2. what draught actually is

def test_draught_is_the_deepest_panel_not_a_closed_form():
    """Measured off the solved geometry, so a blend is priced correctly.

    Asserted against `depth - z.min()` recomputed from the VLM result, and
    asserted to DIFFER from the naive `depth + h*|sin(cant)|`, which is what
    a closed form would have given.
    """
    prob = hydrofoil.HydrofoilWingletProblem()
    x = _x(prob, winglet_cant_deg=-90.0, winglet_h_frac=0.15, depth_m=0.15)
    out = hydrofoil.evaluate_hydrofoil_winglet(x, prob)
    res = out["vlm"]
    assert out["draught_m"] == pytest.approx(
        float(out["depth"] - res.z.min()), rel=0, abs=1e-12)
    naive = 0.15 + 0.15 * (prob.b / 2.0)          # h*|sin(90 deg)| = h
    assert out["draught_m"] < naive               # panel centres, not the arc end
    assert out["draught_m"] == pytest.approx(0.2391, abs=5e-4)


def test_the_foil_plane_is_z_zero_which_is_why_the_sign_never_bites():
    """A premise, stated because it makes a wrong formula look right.

    ``depth - z.min()`` and ``depth + abs(z.min())`` are the SAME number for
    every design this family can fly, because the main foil sits at z = 0 and
    is always present, so ``z.min() <= 0`` always. A mutation swapping one for
    the other therefore survives — it is an equivalent mutant *on this
    geometry*, not a gap in the tests.

    It stops being equivalent the moment the foil itself acquires dihedral or
    is displaced in z: then ``z.min()`` can be positive and only the
    subtraction is right. This test is the tripwire for that day.
    """
    prob = hydrofoil.HydrofoilWingletProblem()
    for cant in (-90.0, -30.0, 0.0, 30.0, 90.0):
        out = hydrofoil.evaluate_hydrofoil_winglet(
            _x(prob, winglet_cant_deg=cant, winglet_h_frac=0.15,
               depth_m=0.30), prob)
        z = out["vlm"].z
        assert float(z.min()) <= 0.0, cant
        assert float(z[~out["vlm"].is_winglet].min()) == pytest.approx(0.0,
                                                                      abs=1e-12)


@pytest.mark.parametrize("cant", [5.0, 15.0, 45.0, 90.0])
def test_canting_UP_adds_no_draught_at_all(cant):
    """The device rises; the foil stays the deepest thing. Odd, not even."""
    prob = hydrofoil.HydrofoilWingletProblem()
    depth = 0.30
    out = hydrofoil.evaluate_hydrofoil_winglet(
        _x(prob, winglet_cant_deg=+cant, winglet_h_frac=0.15,
           depth_m=depth), prob)
    assert out["draught_m"] == pytest.approx(depth, abs=1e-9)


@pytest.mark.parametrize("cant", [5.0, 15.0, 45.0, 90.0])
def test_canting_DOWN_adds_draught_and_that_is_the_asymmetry(cant):
    prob = hydrofoil.HydrofoilWingletProblem()
    depth = 0.30
    up = hydrofoil.evaluate_hydrofoil_winglet(
        _x(prob, winglet_cant_deg=+cant, winglet_h_frac=0.15,
           depth_m=depth), prob)["draught_m"]
    dn = hydrofoil.evaluate_hydrofoil_winglet(
        _x(prob, winglet_cant_deg=-cant, winglet_h_frac=0.15,
           depth_m=depth), prob)["draught_m"]
    assert dn > up
    assert up == pytest.approx(depth, abs=1e-9)


def test_a_zero_height_device_draws_exactly_the_foil_depth():
    prob = hydrofoil.HydrofoilWingletProblem()
    for cant in (-90.0, 0.0, 90.0):
        out = hydrofoil.evaluate_hydrofoil_winglet(
            _x(prob, winglet_cant_deg=cant, winglet_h_frac=0.0,
               depth_m=0.42), prob)
        assert out["draught_m"] == pytest.approx(0.42, abs=1e-9)


# ------------------------- 2b. the TAIL family, by VALUE not just by width
#
# Adversarial review of this change found that the tail side was exercised
# only for vector width and label strings: mutations that ignored the
# stabiliser, flipped the margin's sign, replaced the draught with the plain
# depth, or made the margin a constant ALL survived the suite. The tail is 96
# of the 100 registry names that take the cap, so the gap was on the wrong
# side. These assert the number.

def _tail_x(prob, **over):
    labels = list(prob.param_labels)
    x = np.array([0.5 * (lo + hi) for lo, hi in prob.bounds])
    for k, v in over.items():
        x[labels.index(k)] = v
    return x


def test_the_tail_draws_MORE_than_its_foil_depth():
    """The stabiliser sits below the foil, so draught is strictly deeper.

    Kills both "ignore the stabiliser" and "draught := depth".
    """
    prob = hydrotail.HydrofoilTailProblem()
    depth = 0.30
    out = hydrotail.evaluate_hydrofoil_tail(_tail_x(prob, depth_m=depth), prob)
    assert out["draught_m"] > depth + 1e-6
    # ...and by exactly the stabiliser's own separation, which is what
    # `stab_depth_m` reports
    assert out["draught_m"] == pytest.approx(out["stab_depth_m"], abs=1e-9)


def test_the_tail_margin_is_cap_minus_draught_with_the_right_sign():
    """Kills the sign flip, the constant, and `cap - depth`."""
    cap = 0.40
    prob = hydrotail.HydrofoilTailProblem(draught_max_m=cap)
    # comfortably inside the cap
    shallow = hydrotail.evaluate_hydrofoil_tail(
        _tail_x(prob, depth_m=0.20), prob)
    assert shallow["g_draught"] == pytest.approx(
        cap - shallow["draught_m"], abs=1e-12)
    assert shallow["g_draught"] > 0.0
    # and outside it
    deep = hydrotail.evaluate_hydrofoil_tail(_tail_x(prob, depth_m=0.90), prob)
    assert deep["g_draught"] == pytest.approx(
        cap - deep["draught_m"], abs=1e-12)
    assert deep["g_draught"] < 0.0
    # the margin must MOVE with the draught, so a constant cannot pass
    assert deep["g_draught"] < shallow["g_draught"] - 0.1
    # and it is not `cap - depth`
    assert deep["g_draught"] != pytest.approx(cap - deep["depth"], abs=1e-9)
    # the vector carries it in the last slot
    _, g = hydrotail.fg_hydrofoil_tail(_tail_x(prob, depth_m=0.90), prob)
    assert g[-1] == pytest.approx(deep["g_draught"])


def test_the_tail_min_draught_is_the_draught_it_actually_reaches():
    """The config-time floor must equal the measured shallowest draught."""
    prob = hydrotail.HydrofoilTailProblem()
    out = hydrotail.evaluate_hydrofoil_tail(
        _tail_x(prob, depth_m=prob.DEPTH_BOUNDS[0]), prob)
    assert prob.min_draught_m == pytest.approx(out["draught_m"], abs=1e-9)
    assert prob.min_draught_m > prob.DEPTH_BOUNDS[0]     # NOT the depth floor


def test_a_cap_between_the_depth_floor_and_the_tail_floor_is_refused():
    """The band adversarial review found accepted-then-infeasible.

    DEPTH_BOUNDS[0] is 0.15 m but the elevator family cannot draw less than
    0.21 m, so every cap in [0.15, 0.21) is arithmetically impossible.
    """
    prob = hydrotail.HydrofoilTailProblem()
    assert prob.DEPTH_BOUNDS[0] < prob.min_draught_m
    mid = 0.5 * (prob.DEPTH_BOUNDS[0] + prob.min_draught_m)
    with pytest.raises(ValueError, match="shallowest draught"):
        hydrotail.HydrofoilTailProblem(draught_max_m=mid)
    # the winglet family, which has no second surface, ACCEPTS the same number
    hydrofoil.HydrofoilWingletProblem(draught_max_m=mid)


# --------------------------------------------------- 3. the margin it makes

def test_the_margin_is_cap_minus_draught_and_goes_negative():
    prob = hydrofoil.HydrofoilWingletProblem(draught_max_m=0.20)
    out = hydrofoil.evaluate_hydrofoil_winglet(
        _x(prob, winglet_cant_deg=-90.0, winglet_h_frac=0.15,
           depth_m=0.15), prob)
    assert out["g_draught"] == pytest.approx(0.20 - out["draught_m"])
    assert out["g_draught"] < 0.0                 # 0.239 draws more than 0.20
    f, g = hydrofoil.fg_hydrofoil_winglet(
        _x(prob, winglet_cant_deg=-90.0, winglet_h_frac=0.15,
           depth_m=0.15), prob)
    assert np.asarray(g).shape == (2,)
    assert g[1] == pytest.approx(out["g_draught"])


def test_the_failure_path_returns_the_matching_width():
    """The optimiser sizes its constraint block up front off one call."""
    prob = hydrofoil.HydrofoilWingletProblem(draught_max_m=0.30)
    f, g = hydrofoil.fg_hydrofoil_winglet(np.zeros(prob.dim), prob)
    assert f == hydrofoil.PENALTY
    assert np.asarray(g).shape == (2,)
    tail = hydrotail.HydrofoilTailProblem(draught_max_m=0.70)
    ft, gt = hydrotail.fg_hydrofoil_tail(np.zeros(tail.dim), tail)
    assert ft == hydrotail.PENALTY
    assert np.asarray(gt).shape == (3,)


def test_the_capped_labels_name_the_extra_margin():
    prob = hydrofoil.HydrofoilWingletProblem(draught_max_m=0.30)
    assert prob.n_constraints == 2
    assert prob.constraint_labels == ("cavitation margin",
                                      hydrofoil.DRAUGHT_CONSTRAINT_LABEL)
    tail = hydrotail.HydrofoilTailProblem(draught_max_m=0.70)
    assert tail.n_constraints == 3
    assert tail.constraint_labels[-1] == hydrofoil.DRAUGHT_CONSTRAINT_LABEL


# ------------------------------------- 4. it is NOT a bound on depth_m

def test_a_depth_bound_would_overshoot_the_same_number_in_draught():
    """The pre-registered kill criterion, as an assertion.

    Bound depth alone at 0.30 and the best design still reaches deeper than
    0.30, because its downward cant is unpriced. That is what makes the cap a
    different constraint rather than a renamed bound.
    """
    prob = hydrofoil.HydrofoilWingletProblem()
    best_lod, best = -np.inf, None
    for cant in np.linspace(-90.0, 90.0, 37):
        out = hydrofoil.evaluate_hydrofoil_winglet(
            _x(prob, winglet_cant_deg=cant, depth_m=0.30), prob)
        if out["feasible"] and out["g"] >= 0.0 and out["LoD"] > best_lod:
            best_lod, best = out["LoD"], out
    assert best is not None
    assert best["winglet"]["cant_deg"] < 0.0       # it cants DOWN, unpriced
    assert best["draught_m"] > 0.30                # OVERSHOOT — the point
    assert best["draught_m"] == pytest.approx(0.3077, abs=5e-4)


# ------------------------------------------- 5. an impossible cap is refused

@pytest.mark.parametrize("bad", [0.0, -0.5])
def test_a_non_positive_cap_is_refused(bad):
    with pytest.raises(ValueError, match="positive"):
        hydrofoil.HydrofoilWingletProblem(draught_max_m=bad)


def test_a_cap_under_the_shallowest_depth_is_refused_at_config_time():
    lo = hydrofoil.HydrofoilWingletProblem.DEPTH_BOUNDS[0]
    with pytest.raises(ValueError) as ei:
        hydrofoil.HydrofoilWingletProblem(draught_max_m=lo * 0.5)
    msg = str(ei.value)
    assert "below the shallowest draught" in msg
    assert "Raise the cap" in msg
    with pytest.raises(ValueError):
        hydrotail.HydrofoilTailProblem(draught_max_m=lo * 0.5)


def test_a_cap_exactly_at_the_shallowest_depth_is_allowed():
    """It is satisfiable — by a flat or upward device at minimum depth."""
    lo = hydrofoil.HydrofoilWingletProblem.DEPTH_BOUNDS[0]
    prob = hydrofoil.HydrofoilWingletProblem(draught_max_m=lo)
    out = hydrofoil.evaluate_hydrofoil_winglet(
        _x(prob, winglet_cant_deg=+90.0, depth_m=lo), prob)
    assert out["g_draught"] == pytest.approx(0.0, abs=1e-9)


# ----------------------------------------------------- 6. the api surface

def test_the_flag_reaches_both_families_and_widens_the_vector():
    for name in ("hydrofoil + winglet", "hydrofoil + elevator"):
        spec = api.PROBLEM_SPECS[name]
        assert api.DRAUGHT_MAX_KEY in (spec.flags or ()), name
        base = spec.build({}, {}, None)
        capped = spec.build({}, {api.DRAUGHT_MAX_KEY: 0.30}, None)
        n0 = len(base.problem.constraint_labels)
        n1 = len(capped.problem.constraint_labels)
        assert n1 == n0 + 1, name
        x = np.array([0.5 * (lo + hi) for lo, hi in base.bounds])
        f0, g0 = base.callable(x)
        f1, g1 = capped.callable(x)
        assert f0 == f1, name                      # objective untouched
        assert np.asarray(g1).shape == (n1,), name


def test_a_flag_of_none_is_the_uncapped_family():
    spec = api.PROBLEM_SPECS["hydrofoil + winglet"]
    b = spec.build({}, {api.DRAUGHT_MAX_KEY: None}, None)
    assert b.problem.draught_max_m is None
    assert b.problem.constraint_labels == ("cavitation margin",)


def test_which_water_families_take_the_cap_and_which_deliberately_do_not():
    """The declaration table, pinned — and the gap is now exactly one rule.

    Session 48 shipped the cap on two builders (the imaged winglet family and
    the hydrofoil+elevator family) and recorded an honest wiring gap: the
    CST-section co-design water families build their foil through
    ``_make_hydrofoil_section_builder``, which did not thread the flag, and
    they DO have geometry below the foil plane. **Session 49 closed that
    gap.** What is left is one rule, applied twice:

        the cap is declared wherever something reaches below the foil plane.

    So the only water families without it are the PLANAR ones — no tip
    device, no second surface — where draught IS ``depth_m`` and a cap would
    be exactly the bound the design box already offers. That includes the
    planar co-design twin: designing its section does not give it anything to
    hang below the foil.

    Two earlier versions of this reasoning are recorded as wrong, because
    both are easy to reach for again:

    * "the flag is withheld because a silently dropped flag is worse than an
      absent one" — that did not hold while ``api`` validated nothing;
      passing the key to a family that ignored it was accepted in silence.
      (:mod:`tests.test_flag_validation` is the fix, also session 49.)
    * "has a tip-device height row" as the rule — wrong in both directions: a
      plain elevator has no ``winglet_h_frac`` row and still reaches below the
      foil through its stabiliser, and the co-design families had the row
      while having no wiring.
    """
    water = {n: s for n, s in api.PROBLEM_SPECS.items() if s.medium == "water"}
    declaring = {n for n, s in water.items()
                 if api.DRAUGHT_MAX_KEY in (s.flags or ())}
    assert declaring, "nothing declares the draught cap"

    # every declarer really honours it...
    for n in sorted(declaring):
        built = api.PROBLEM_SPECS[n].build({}, {api.DRAUGHT_MAX_KEY: 0.65},
                                           None)
        prob = built.problem
        foil = getattr(prob, "foil", prob)      # the co-design wrapper
        assert getattr(foil, "draught_max_m", None) == 0.65, n

    # ...and the gap is now the PLANAR families and nothing else: no tip
    # device and no second surface, so nothing reaches below the foil plane
    # and draught IS depth. A cap there would be exactly a bound on depth_m,
    # which the design box already provides — correctly absent rather than
    # forgotten.
    gap = sorted(set(water) - declaring)
    assert set(gap) == {"hydrofoil", "hydrofoil + free chord law",
                        "hydrofoil [chosen section]",
                        "hydrofoil [chosen section] + free chord law",
                        "hydrofoil + CST section (XFOIL)",
                        "hydrofoil + CST section (XFOIL) + free chord law"}
    for n in gap:
        prob = api.PROBLEM_SPECS[n].build({}, {}, None).problem
        foil = getattr(prob, "foil", prob)
        assert not hasattr(foil, "draught_max_m") or \
            foil.draught_max_m is None, n


# ------------------------------------- 6b. the CST-section co-design wiring

#: the two co-design kinds that reach below the foil plane, one of each
_CODESIGN = ("hydrofoil + winglet + CST section (XFOIL)",
             "hydrofoil + elevator + CST section (XFOIL)")


@pytest.mark.parametrize("name", _CODESIGN)
def test_the_codesign_families_carry_the_cap_onto_their_foil(name):
    spec = api.PROBLEM_SPECS[name]
    assert api.DRAUGHT_MAX_KEY in (spec.flags or ()), name
    built = spec.build({}, {api.DRAUGHT_MAX_KEY: 0.65}, None)
    assert built.problem.foil.draught_max_m == 0.65
    assert built.problem.foil.section_polar is None   # still DESIGNED


@pytest.mark.parametrize("name", _CODESIGN)
def test_the_codesign_width_follows_the_foil_not_a_table(name):
    """ASSERTED, not assumed.

    ``HydrofoilSectionProblem.n_constraints`` asks the FOIL rather than
    reading a per-type table, which is what makes the cap's extra margin
    appear here at all. The session-48 review found the same wrapper sizing
    success from ``out["g"]`` and failure from a static count; this is the
    value-level check that the two agree once the flag is live.
    """
    spec = api.PROBLEM_SPECS[name]
    base = spec.build({}, {}, None)
    capped = spec.build({}, {api.DRAUGHT_MAX_KEY: 0.65}, None)
    n0, n1 = base.problem.n_constraints, capped.problem.n_constraints
    assert n1 == n0 + 1, name
    assert len(capped.problem.constraint_labels) == n1
    # the section's own two stay LAST, and the draught margin lands with the
    # family's own margins — the order api._SIZE_MARGIN_INSERT relies on
    labels = capped.problem.constraint_labels
    assert labels[-2:] == hydrofoil_section.SECTION_CONSTRAINT_LABELS
    assert labels[-3] == hydrofoil.DRAUGHT_CONSTRAINT_LABEL
    # success AND failure return that width
    x = capped.problem.x0.copy()
    _f, g = capped.callable(x)
    assert np.asarray(g).shape == (n1,)
    bad = capped.bounds[:, 0].copy()          # a section that cannot fly
    # …and it really is the FAILURE path. Without this the shape assertion
    # below would silently start testing the success path if this corner ever
    # became flyable, which is the vacuous-test class this file exists in.
    assert not capped.evaluate(bad)["feasible"]
    _fb, gb = capped.callable(bad)
    assert np.asarray(gb).shape == (n1,)


@pytest.mark.parametrize("name", _CODESIGN)
def test_the_codesign_margin_is_cap_minus_draught(name):
    """The VALUE, not the width — the session-48 lesson, applied here.

    A width-and-label test would pass with the margin wired to a constant, to
    ``depth - cap``, or to the wrong sign.
    """
    spec = api.PROBLEM_SPECS[name]
    built = spec.build({}, {api.DRAUGHT_MAX_KEY: 0.65}, None)
    out = built.evaluate(built.problem.x0.copy())
    assert out["feasible"], out.get("reason")
    assert out["g_draught"] == pytest.approx(0.65 - out["draught_m"], abs=1e-12)
    # and it is the margin actually handed to the optimiser
    _f, g = built.callable(built.problem.x0.copy())
    i = list(built.problem.constraint_labels).index(
        hydrofoil.DRAUGHT_CONSTRAINT_LABEL)
    assert float(np.asarray(g)[i]) == pytest.approx(out["g_draught"],
                                                    abs=1e-12)


@pytest.mark.parametrize("name", _CODESIGN)
def test_the_codesign_cap_does_not_move_the_objective(name):
    """Uncapped is bit-for-bit; capping only adds a margin."""
    spec = api.PROBLEM_SPECS[name]
    base = spec.build({}, {}, None)
    x = base.problem.x0.copy()
    # both arms must be on the SUCCESS path, or `f0 == f1` compares two
    # PENALTY sentinels and the margin slices compare two G_FAIL vectors —
    # a test that passes having compared nothing
    assert base.evaluate(x)["feasible"]
    f0, g0 = base.callable(x)
    f1, g1 = spec.build({}, {api.DRAUGHT_MAX_KEY: 0.65}, None).callable(x)
    assert f0 == f1
    # every margin the uncapped family reported is unchanged, in order
    g0, g1 = np.asarray(g0), np.asarray(g1)
    assert np.array_equal(g0[:-2], g1[:-3])          # the family's own
    assert np.array_equal(g0[-2:], g1[-2:])          # the section's own


def test_the_codesign_cap_binds_through_api_run_on_every_optimiser():
    """Through the PUBLIC path, on each optimiser, with the NUMBER checked.

    Session 48 shipped a margin that was scalar on success and width-2 on
    failure; every width and label test passed while two optimisers crashed
    and the third silently dropped the constraint. The only thing that would
    have caught it is this: run it, and read the margin back.
    """
    name = "hydrofoil + winglet + CST section (XFOIL)"
    cfg_rep = api.RunConfig(problem_name=name,
                            flags={api.DRAUGHT_MAX_KEY: 0.65})
    for opt in ("random", "ga", "bo"):
        seen: dict = {}
        res = api.run(api.RunConfig(
            problem_name=name, optimiser=opt, budget=4, seed=0,
            flags={api.DRAUGHT_MAX_KEY: 0.65}))
        assert res.eval_g, opt
        widths = {len(g) for g in res.eval_g}
        assert widths == {4}, (opt, widths)

        # …and the VALUE, read back through the public report. This part was
        # dead in the first version of this test: at budget 4 nothing on this
        # problem is feasible, so ``res.best_x`` is None, ``design_report``
        # returns a shape failure, and the value assertions sat behind an
        # ``if feasible`` that never fired — leaving exactly the width-only
        # test the docstring above says is not enough. Found by the
        # pre-commit review. The design point is now the problem's OWN x0,
        # which is feasible, so the numbers are always checked; ``best_x`` is
        # additionally checked wherever the search did find something.
        # …at TWO design points with DIFFERENT draughts, not one. A margin
        # wired to a constant equal to the x0 value survives a single-point
        # check — verified by mutation, which is why the deeper point is
        # here: the two must disagree, and each must equal cap − its own
        # draught.
        built = api.PROBLEM_SPECS[name].build(
            {}, {api.DRAUGHT_MAX_KEY: 0.65}, None)
        deeper = built.problem.x0.copy()
        lbl = list(built.problem.param_labels)
        deeper[lbl.index("depth_m")] = 0.75
        for tag, x in (("x0", built.problem.x0),
                       ("deeper", deeper),
                       ("best_x", res.best_x)):
            if x is None:
                continue
            rep = api.design_report(cfg_rep, x)
            bd = rep["breakdown"]
            assert hydrofoil.DRAUGHT_CONSTRAINT_LABEL in \
                rep["constraint_labels"], (opt, tag)
            if not bd.get("feasible"):
                continue
            assert bd["g_draught"] == pytest.approx(0.65 - bd["draught_m"],
                                                    abs=1e-12), (opt, tag)
            i = rep["constraint_labels"].index(
                hydrofoil.DRAUGHT_CONSTRAINT_LABEL)
            assert float(np.asarray(bd["g"])[i]) == pytest.approx(
                bd["g_draught"], abs=1e-12), (opt, tag)
            seen[tag] = float(bd["g_draught"])
        # the two probe points must NOT agree — otherwise the check above is
        # satisfiable by a constant
        assert seen["x0"] != seen["deeper"], opt


def test_the_flag_is_declared_only_where_a_builder_honours_it():
    """A silently dropped flag is worse than a refused one.

    Every spec that declares the key must produce a problem that carries it.
    On the co-design families the problem is a WRAPPER around the foil, and
    the cap is a property of the foil — the wrapper adds a section, not a
    hull — so the check follows ``.foil`` where there is one. It does not
    accept the wrapper answering ``None``: that would pass the very families
    session 49 wired.
    """
    for name, spec in api.PROBLEM_SPECS.items():
        if api.DRAUGHT_MAX_KEY not in (spec.flags or ()):
            continue
        built = spec.build({}, {api.DRAUGHT_MAX_KEY: 0.65}, None)
        prob = built.problem
        carrier = getattr(prob, "foil", prob)
        assert getattr(carrier, "draught_max_m", None) == 0.65, name
        # ...and it really reached the constraint vector, not just a field
        assert hydrofoil.DRAUGHT_CONSTRAINT_LABEL in prob.constraint_labels, \
            name
