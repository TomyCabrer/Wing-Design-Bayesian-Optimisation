"""The tandem pair may search its section thickness, like every other family.

The 2026-08-02 combination audit left five refusals standing. This was the
cheapest of them: both tandem solvers already fly a NACA 24XX member correctly
when handed one (L/D 28.5 -> 24.9 across t/c 0.06 -> 0.18); what was missing
was the design VARIABLE. 57 configurations were being refused for the absence
of two rows.

Two rows, not one, and that is the design decision worth pinning: this family
already gives each surface its own span, its own chord law and its own
section, so a pair forced to share a thickness would be carrying a constraint
nobody asked for.

What these tests hold:

* the twin at t/c = 0.12 IS the published pair, bit-for-bit — the family's
  0.12 member is `naca2412_re1e6.pol`, which is the plain pair's own polar;
* the thickness actually reaches the answer, on BOTH wings independently;
* the rows sit AHEAD of the size / flight / chord blocks, because those three
  are read back by counting from the END of the design vector and a family
  block that grew at the tail would silently re-index all of them;
* a searched thickness and a stated section are refused together rather than
  one of them silently losing.
"""
from __future__ import annotations

import numpy as np
import pytest

from aerobo import api, tandem

TWIN = "tandem + t/c"


def _built(name):
    return api.PROBLEM_SPECS[name].build({}, {}, None)


def test_the_twin_is_registered_with_two_thickness_rows():
    spec = api.PROBLEM_SPECS[TWIN]
    assert spec.param_labels[-2:] == ("tc_front", "tc_rear")
    built = _built(TWIN)
    assert built.dim == _built("tandem").dim + 2
    assert built.param_labels[-2:] == ("tc_front", "tc_rear")


def test_at_the_family_member_it_is_the_published_pair_bit_for_bit():
    """t/c = 0.12 selects `naca2412_re1e6.pol`, which is `default_polar` —
    so this is an identity, not an approximation, and `==` is the right
    assertion."""
    base, twin = _built("tandem"), _built(TWIN)
    x = base.bounds.mean(axis=1)
    a = base.evaluate(x)
    b = twin.evaluate(np.concatenate([x, [0.12, 0.12]]))
    assert a["feasible"] and b["feasible"]
    assert b["LoD"] == a["LoD"]
    assert b["CDp"] == a["CDp"]
    assert b["CL_total"] == a["CL_total"]


def test_the_thickness_reaches_the_answer_and_is_monotone_in_drag():
    """Thicker sections drag more at this design lift, so L/D falls."""
    twin = _built(TWIN)
    x = twin.bounds.mean(axis=1)
    lods = {}
    for tc in (0.09, 0.12, 0.15):
        xx = x.copy()
        xx[-2] = xx[-1] = tc
        out = twin.evaluate(xx)
        assert out["feasible"], (tc, out.get("reason"))
        lods[tc] = out["LoD"]
        assert out["tc_front"] == pytest.approx(tc)
        assert out["tc_rear"] == pytest.approx(tc)
    assert lods[0.09] > lods[0.12] > lods[0.15], lods


def test_each_wing_carries_its_own_thickness():
    """The two rows are independent variables, and each is felt mostly by its
    own wing.

    NOT "only by its own wing", and the difference is the physics rather than
    a leak. A family member brings its own lift slope and zero-lift angle as
    well as its own cd, so thickening the REAR surface changes the lift it
    makes, the pair re-trims as a system to the same CL_total, and the front
    wing's effective incidence — and therefore its profile drag — moves with
    it. The named polars are the exact check that the rows are not crossed;
    the magnitudes are the check that each row lands on its own wing.
    """
    twin = _built(TWIN)
    x = twin.bounds.mean(axis=1)
    x[-2] = x[-1] = 0.12
    base = twin.evaluate(x)

    x_rear = x.copy()
    x_rear[-1] = 0.15
    rear = twin.evaluate(x_rear)
    assert rear["feasible"]
    assert rear["polar"] == base["polar"]            # front table untouched
    assert rear["polar_rear"] != base["polar_rear"]  # rear table moved
    d_front = abs(rear["CDp_front"] - base["CDp_front"])
    d_rear = abs(rear["CDp_rear"] - base["CDp_rear"])
    assert rear["CDp_rear"] > base["CDp_rear"]
    assert d_rear > 10 * d_front, (d_front, d_rear)

    x_front = x.copy()
    x_front[-2] = 0.15
    front = twin.evaluate(x_front)
    assert front["feasible"]
    assert front["polar_rear"] == base["polar_rear"]
    assert front["polar"] != base["polar"]
    assert front["CDp_front"] > base["CDp_front"]
    assert (abs(front["CDp_front"] - base["CDp_front"])
            > 10 * abs(front["CDp_rear"] - base["CDp_rear"]))


def test_the_breakdown_says_whether_the_thickness_was_searched():
    assert _built("tandem").evaluate(
        _built("tandem").bounds.mean(axis=1))["tc_front"] is None
    twin = _built(TWIN)
    out = twin.evaluate(twin.bounds.mean(axis=1))
    assert out["tc_front"] is not None and out["tc_rear"] is not None


def test_the_rows_sit_ahead_of_every_trailing_block():
    """size / flight / chord are all read by counting from the END, so the
    thickness pair must not be last. Checked structurally — where the rows
    are is the whole reason this composes."""
    for name in (TWIN + " + free chord law",
                 TWIN + " + free flight state",
                 TWIN + " + free flight state + free chord law"):
        labels = _built(name).param_labels
        i_tc = labels.index("tc_rear")
        assert i_tc < len(labels) - 1, name
        for trailing in ("V_ms", "altitude_m", "chord_k1", "chord_k1_t"):
            if trailing in labels:
                assert labels.index(trailing) > i_tc, (name, trailing)


def test_every_modifier_combination_builds_and_evaluates():
    """The refusal being closed was a COMBINATION refusal, so the closure is
    only real if every combination flies."""
    names = sorted(n for n in api.PROBLEM_SPECS if n.startswith(TWIN))
    assert len(names) == 16, names
    for name in names:
        built = _built(name)
        out = built.evaluate(built.bounds.mean(axis=1))
        # the box centre is not required to be FEASIBLE (the plain pair's
        # free-planform variant is untrimmable there too, for the same
        # reason) — it is required to be SCORED rather than to raise
        assert "score" in out and np.isfinite(out["score"]), (name, out)


def test_the_plain_pair_still_has_exactly_its_own_variants():
    """A new base family must not steal or duplicate the old one's names."""
    plain = sorted(n for n in api.PROBLEM_SPECS
                   if n == "tandem" or (n.startswith("tandem + ")
                                        and "t/c" not in n))
    assert len(plain) == 16, plain


def test_a_searched_thickness_and_a_stated_section_are_refused_together():
    """One of the two would otherwise lose silently, and which one would
    depend on the order of two lines."""
    with pytest.raises(ValueError, match="tc_free"):
        tandem.TandemProblem(tc_free=True, polar_rear=object())


def test_the_twin_declares_no_chosen_section_flag():
    """A pair that PICKS its tables off the thickness rows has no fixed
    section a chosen one could replace — the rule SECTION_KEY already
    states for every other by-thickness family."""
    spec = api.PROBLEM_SPECS[TWIN]
    assert api.SECTION_KEY not in spec.flags
    assert api.SECTION_AFT_KEY not in spec.flags
    assert api.SECTION_KEY in api.PROBLEM_SPECS["tandem"].flags


# --------------------------------------------------------------------------
# The design vector is verified BY CONSTRUCTION, per row, on all 16 variants.
#
# Session 41's adversarial review never reached the correctness dimension, and
# what it would have found first is above: the thickness pair is read by
# POSITIVE index (`x[4 + 2k]`, `x[5 + 2k]`) while the size, flight and chord
# blocks are all read by counting from the END (`resolve_spans`,
# `flight_from_x`, the `x[-2m:-m]` slices). The placement is what keeps those
# negative indices valid, and `test_every_modifier_combination_builds_and_
# evaluates` above only asserts the score is FINITE — which a vector read
# entirely off by two would also satisfy.
#
# So: for every label, assert the value the solver READ equals the value at
# that label's own index; and perturb one row at a time and assert nothing
# owned by another block moves.

#: label -> the breakdown field that must equal ``x[i]`` exactly
_READBACK = {
    "tc_front": "tc_front", "tc_rear": "tc_rear",
    "b_m": "b_front", "b_rear_m": "b_rear",
    "V_ms": "V", "altitude_m": "altitude_m", "S_m2": "Sref",
}

#: label -> the field that row MUST move, and the fields it must NOT.
#: Only cross-BLOCK pairs are listed, because a within-block swap (front vs
#: rear) is what the ``polar`` / ``polar_rear`` split already catches.
_ISOLATION = {
    "tc_front": ("polar", ("polar_rear", "b_front", "b_rear", "V")),
    "tc_rear": ("polar_rear", ("polar", "b_front", "b_rear", "V")),
    "b_m": ("b_front", ("tc_front", "tc_rear", "b_rear", "V")),
    "b_rear_m": ("b_rear", ("tc_front", "tc_rear", "b_front", "V")),
    "V_ms": ("V", ("tc_front", "tc_rear", "b_front", "b_rear")),
    "altitude_m": (None, ("tc_front", "tc_rear", "b_front", "b_rear", "V")),
    "S_m2": ("Sref", ("tc_front", "tc_rear", "b_front", "b_rear", "V")),
    "ws_pa": ("Sref", ("tc_front", "tc_rear", "b_front", "b_rear", "V")),
    "chord_front_k1": (None, ("tc_front", "tc_rear", "b_front", "b_rear", "V")),
    "chord_front_k3": (None, ("tc_front", "tc_rear", "b_front", "b_rear", "V")),
    "chord_rear_k1": (None, ("tc_front", "tc_rear", "b_front", "b_rear", "V")),
    "chord_rear_k3": (None, ("tc_front", "tc_rear", "b_front", "b_rear", "V")),
}


def _all_tc_variants():
    names = sorted(n for n in api.PROBLEM_SPECS if n.startswith(TWIN))
    assert len(names) == 16, names
    return names


def _feasible_base(built):
    """A FLYABLE point in the box, and the breakdown there.

    The box centre of a sized variant is AR 73 and refuses, so the sized
    families are searched rather than assumed. Seeded, so the point is the
    same on every run.
    """
    bnds = np.asarray(built.bounds, dtype=float)
    x0 = bnds.mean(axis=1)
    out = built.evaluate(x0)
    if out.get("feasible"):
        return x0, out
    rng = np.random.default_rng(0)
    for _ in range(4000):
        x = bnds[:, 0] + rng.random(bnds.shape[0]) * (bnds[:, 1] - bnds[:, 0])
        out = built.evaluate(x)
        if out.get("feasible"):
            return x, out
    raise AssertionError("no feasible point found in the box")


def _perturb(built, x0, i):
    """Move row ``i`` alone, as far as stays feasible. Returns the breakdown
    and the moved value, or (None, None) if the row cannot move at all."""
    lo, hi = np.asarray(built.bounds, dtype=float)[i]
    for frac in (0.30, 0.15, 0.07, 0.03, 0.01, 0.003):
        for sgn in (+1.0, -1.0):
            v = float(np.clip(x0[i] + sgn * frac * (hi - lo), lo, hi))
            if abs(v - x0[i]) < 1e-12:
                continue
            x = np.asarray(x0, dtype=float).copy()
            x[i] = v
            out = built.evaluate(x)
            if out.get("feasible"):
                return out, v
    return None, None


def test_every_label_reads_back_the_value_at_its_own_index():
    """The value the solver USED for each label is the value the vector holds
    at that label's index — on all 16 variants, exactly.

    This is the assertion the whole ahead-of-the-trailing-blocks placement
    exists to make true, and it was never made.
    """
    checked = 0
    for name in _all_tc_variants():
        built = _built(name)
        labels = list(built.param_labels)
        assert len(labels) == np.asarray(built.bounds).shape[0], name
        x0, base = _feasible_base(built)
        for i, lbl in enumerate(labels):
            field = _READBACK.get(lbl)
            if field is None:
                continue
            assert base.get(field) == pytest.approx(float(x0[i]), abs=1e-12), (
                name, lbl, field, float(x0[i]), base.get(field))
            checked += 1
    # a guard on the guard: if the label vocabulary ever changes under this
    # test, it must not silently check nothing
    assert checked >= 70, checked


def test_no_row_reaches_across_a_block_boundary():
    """Perturb ONE row; the fields owned by the other blocks do not move.

    A vector read off by k shows up here and in no other test in the file:
    the row that should have moved does not, and a row in a different block
    does. The `polar` / `polar_rear` pair also pins that the two thickness
    rows are not swapped with each other.
    """
    seen = set()
    for name in _all_tc_variants():
        built = _built(name)
        labels = list(built.param_labels)
        x0, base = _feasible_base(built)
        for i, lbl in enumerate(labels):
            rule = _ISOLATION.get(lbl)
            if rule is None:
                continue
            owned, foreign = rule
            out, moved_to = _perturb(built, x0, i)
            assert out is not None, (name, lbl, "no feasible perturbation")
            if owned is not None:
                assert out[owned] != base[owned], (
                    name, lbl, f"moving it left {owned} alone", moved_to)
            for field in foreign:
                if base.get(field) is None:
                    continue
                assert out[field] == base[field], (
                    name, lbl, f"moving it also moved {field}",
                    base[field], out[field])
            seen.add(lbl)
    # every label in the table was actually exercised somewhere
    assert seen == set(_ISOLATION), sorted(set(_ISOLATION) - seen)
