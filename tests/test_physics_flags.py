"""Phase 5 gate: the three physics flags threaded through objective.Problem /
evaluate — compressibility (mach), ground effect (ground_h_m), and the prop
slipstream (SlipstreamSpec).

Every flag is ADDITIVE with an OFF default, so the overriding gate is the same
one that guards mission mode: with all flags OFF the objective must be
BIT-FOR-BIT the frozen Tier A value. The rest of the gates check that each flag,
once ON, moves the physics in the CORRECT direction (this project's bug history
is sign errors), that the transonic / winglet cases fail into the -100 penalty
contract instead of raising, and that the flags compose and never leak into a
later default Problem (prob is copied, never mutated).
"""

import numpy as np
import pytest

from aerobo.objective import PENALTY, Problem, evaluate, objective
from aerobo.slipstream import SlipstreamSpec

# same frozen literal as test_mission.py / the Tier A regression
FROZEN_LOD = 34.761763013754106
PROBES = [
    np.array([0.4, 1.0, -2.0]),
    np.array([0.5, 0.0, 0.0]),
    np.array([0.7, 1.0, -3.0]),
]

# float fields that must match bit-for-bit between two OFF evaluations
_NUM_FIELDS = [
    "score", "LoD", "CL", "CL_target", "CDi", "CDp", "cd0_extra",
    "CD", "e", "AR", "alpha_deg", "Re_mac",
]


def _off_spec():
    """SlipstreamSpec at the OFF default (mu_inf = 1, no swirl)."""
    return SlipstreamSpec(D_p=1.0, y_centres=[2.0, -2.0])


# --------------------------------------------------- 1. bit-for-bit OFF

def test_all_flags_off_is_frozen_value_exact():
    assert objective(PROBES[0], Problem()) == FROZEN_LOD          # ==, no tol


@pytest.mark.parametrize("x", PROBES)
def test_evaluate_dict_unchanged_when_flags_explicitly_off(x):
    """Explicitly OFF flags (mach = 0, ground = None, slipstream = None AND an
    OFF SlipstreamSpec) reproduce the default-Problem breakdown field-by-field
    with == on floats — the flag plumbing adds nothing when idle."""
    ref = evaluate(x, Problem())
    for prob in (
        Problem(mach=0.0, ground_h_m=None, slipstream=None),
        Problem(slipstream=_off_spec()),         # OFF spec == no spec
    ):
        out = evaluate(x, prob)
        assert out["feasible"]
        for f in _NUM_FIELDS:
            assert out[f] == ref[f], f            # bit-for-bit


# --------------------------------------------------- 2. compressibility (mach)

def test_mach_subsonic_is_feasible_and_shifts_trim():
    """M = 0.3 steepens the section slope (PG), so the wing trims to CL_target
    at a LOWER alpha and the L/D changes — feasible, and provably different
    from the incompressible result."""
    x = PROBES[0]
    off = evaluate(x, Problem())
    on = evaluate(x, Problem(mach=0.3))
    assert on["feasible"], on["reason"]
    assert on["alpha_deg"] < off["alpha_deg"]     # PG raises CL_alpha -> less a
    assert on["LoD"] != off["LoD"]
    assert on["CL"] == pytest.approx(on["CL_target"], abs=1e-6)   # still trimmed


def test_mach_transonic_is_penalty_not_exception():
    """M >= PG_MACH_MAX (0.7) is out of the linearised regime: the pg_beta
    ValueError is mapped to PENALTY (a finite score), NEVER raised."""
    x = PROBES[0]
    val = objective(x, Problem(mach=0.75))        # must not raise
    assert val == PENALTY
    out = evaluate(x, Problem(mach=0.75))
    assert not out["feasible"]
    assert "compressibility" in out["reason"]


def test_mach_negative_is_penalty():
    assert objective(PROBES[0], Problem(mach=-0.1)) == PENALTY


def test_mach_monotone_slope_effect_on_trim_alpha():
    """Rising Mach monotonically lowers the trim alpha (1/beta_n monotone)."""
    x = PROBES[0]
    alphas = [evaluate(x, Problem(mach=m))["alpha_deg"] for m in (0.0, 0.2, 0.4, 0.6)]
    assert all(np.diff(alphas) < 0)


# --------------------------------------------------- 3. ground effect

def test_ground_effect_improves_LoD():
    """h/b = 0.25 (h = 2.5 m, b = 10 m): the rigid-ground image upwash cuts
    induced drag, so trimmed L/D IMPROVES vs free air at the same CL_target."""
    x = PROBES[0]
    off = evaluate(x, Problem())
    on = evaluate(x, Problem(ground_h_m=2.5))
    assert on["feasible"], on["reason"]
    assert on["LoD"] > off["LoD"]
    assert on["CDi"] < off["CDi"]                 # induced drag drops
    assert on["CL"] == pytest.approx(on["CL_target"], abs=1e-6)


def test_ground_closer_is_stronger():
    """Lower height -> larger induced-drag reduction -> higher L/D."""
    x = PROBES[0]
    hi = evaluate(x, Problem(ground_h_m=5.0))["LoD"]
    lo = evaluate(x, Problem(ground_h_m=1.5))["LoD"]
    off = evaluate(x, Problem())["LoD"]
    assert lo > hi > off


@pytest.mark.parametrize("mode", ["winglet", "winglet_capped"])
def test_ground_with_winglet_mode_is_penalty(mode):
    """Winglet modes run on the nonplanar VLM, which has no ground image: an
    in-contract failure (PENALTY + reason), not a silent wrong answer."""
    x = np.array([0.5, 0.0, 0.0, 0.05, 80.0])     # valid winglet vector
    val = objective(x, Problem(mode=mode, ground_h_m=2.5))
    assert val == PENALTY
    out = evaluate(x, Problem(mode=mode, ground_h_m=2.5))
    assert not out["feasible"]
    assert "ground effect not implemented" in out["reason"]


def test_winglet_mode_without_ground_still_works():
    """The ground gate must not break an ordinary winglet evaluation."""
    x = np.array([0.5, 0.0, 0.0, 0.05, 80.0])
    out = evaluate(x, Problem(mode="winglet"))
    assert out["feasible"], out["reason"]


# --------------------------------------------------- 4. slipstream

def test_slipstream_off_spec_is_bit_for_bit():
    """A mu_inf = 1 SlipstreamSpec (OFF) reproduces the legacy score EXACTLY."""
    x = PROBES[0]
    assert objective(x, Problem(slipstream=_off_spec())) == FROZEN_LOD


def test_slipstream_axial_increases_profile_drag():
    """A pure-axial mu = 1.3 slipstream (no swirl) raises q_ratio > 1 over the
    wake footprint, so CDp increases; with no swirl the trim alpha is unchanged
    (the swirl term is what would move it)."""
    x = PROBES[0]
    off = evaluate(x, Problem())
    spec = SlipstreamSpec(D_p=2.0, y_centres=[2.5, -2.5], mu_inf=1.3, dalpha_ref_deg=0.0)
    on = evaluate(x, Problem(slipstream=spec))
    assert on["feasible"], on["reason"]
    assert on["CDp"] > off["CDp"]                 # q_ratio scales the CDp integrand
    assert on["alpha_deg"] == pytest.approx(off["alpha_deg"], abs=1e-12)  # no swirl


def test_slipstream_swirl_redistributes_loading_not_lift():
    """The swirl dalpha is added to the twist BEFORE the solve, but for a
    symmetric prop pair (same spin) it is ANTISYMMETRIC about the span centre,
    so it excites only the even (roll) Fourier modes: CL — and hence the trim
    alpha — is INVARIANT, while the loading is redistributed (CDi rises, span
    efficiency e drops, the effective-AoA spread widens). This is the correct
    'prop swirl makes roll, not lift' result and it exercises the twist-like
    compose step independently of the axial q (mu_inf = 1 here isolates it)."""
    x = PROBES[0]
    off = evaluate(x, Problem())
    swirl = SlipstreamSpec(D_p=2.0, y_centres=[2.5, -2.5], mu_inf=1.0, dalpha_ref_deg=4.0)
    on = evaluate(x, Problem(slipstream=swirl))
    assert on["feasible"], on["reason"]
    # trim alpha & CL unchanged (antisymmetric twist is CL-neutral)
    assert on["alpha_deg"] == pytest.approx(off["alpha_deg"], abs=1e-12)
    assert on["CL"] == pytest.approx(on["CL_target"], abs=1e-6)
    # loading redistributed by the added twist
    assert on["CDi"] > off["CDi"]
    assert on["e"] < off["e"]
    lo_on, hi_on = on["alpha_eff_range_deg"]
    lo_off, hi_off = off["alpha_eff_range_deg"]
    assert (hi_on - lo_on) > (hi_off - lo_off)    # wider effective-AoA spread


# --------------------------------------------------- 5. no leak

def test_flags_do_not_leak_into_a_later_default_problem():
    """prob is copied, never mutated: after evaluating a fully-flagged Problem,
    a fresh Problem() still returns the frozen value exactly."""
    x = PROBES[0]
    flagged = Problem(
        mach=0.3, ground_h_m=2.5,
        slipstream=SlipstreamSpec(D_p=2.0, y_centres=[2.5, -2.5], mu_inf=1.3,
                                  dalpha_ref_deg=4.0),
    )
    _ = objective(x, flagged)                     # exercise every flag
    assert objective(x, Problem()) == FROZEN_LOD  # unchanged, exact
    # and the flagged Problem's own fields are intact (not consumed)
    assert flagged.mach == 0.3 and flagged.ground_h_m == 2.5


# --------------------------------------------------- 6. composition

def test_mach_and_ground_compose():
    """mach + ground_h_m together: feasible, and each pulls its own way — the
    combined L/D beats mach-only (ground adds induced-drag relief on top)."""
    x = PROBES[0]
    mach_only = evaluate(x, Problem(mach=0.3))
    both = evaluate(x, Problem(mach=0.3, ground_h_m=2.5))
    assert both["feasible"], both["reason"]
    assert both["LoD"] > mach_only["LoD"]         # ground relief on top of PG
    assert both["CL"] == pytest.approx(both["CL_target"], abs=1e-6)


def test_ground_and_slipstream_compose():
    """Ground image + slipstream: the twist-shift (swirl) and CDp scaling apply
    inside the ground solve; feasible and trimmed."""
    x = PROBES[0]
    spec = SlipstreamSpec(D_p=2.0, y_centres=[2.5, -2.5], mu_inf=1.2, dalpha_ref_deg=3.0)
    out = evaluate(x, Problem(ground_h_m=2.5, slipstream=spec))
    assert out["feasible"], out["reason"]
    assert out["CL"] == pytest.approx(out["CL_target"], abs=1e-6)
    # ...and each half of the slipstream demonstrably FIRED. Feasible + trimmed
    # is already true of a ground-only run (test_ground_effect_improves_LoD),
    # so the two assertions above could not tell "composed" from "slipstream
    # silently dropped whenever ground is on". Each half is isolated against
    # its own single-flag reference, as test_mach_and_ground_compose does:
    gnd = evaluate(x, Problem(ground_h_m=2.5))
    swirl = SlipstreamSpec(D_p=2.0, y_centres=[2.5, -2.5], mu_inf=1.0,
                           dalpha_ref_deg=3.0)
    gnd_swirl = evaluate(x, Problem(ground_h_m=2.5, slipstream=swirl))
    # swirl: the antisymmetric dalpha reaches the GROUND solve's twist, so the
    # loading is redistributed at the same trimmed CL (CDi up, e down)
    assert gnd_swirl["CDi"] > gnd["CDi"]
    assert gnd_swirl["e"] < gnd["e"]
    # axial: q_ratio scales the ground solve's OWN profile-drag integrand.
    # ISOLATED, it is lift-neutral to the last bit — asserted here against a
    # swirl-free axial reference, which is the form that actually holds:
    axial = SlipstreamSpec(D_p=2.0, y_centres=[2.5, -2.5], mu_inf=1.2,
                           dalpha_ref_deg=0.0)
    gnd_axial = evaluate(x, Problem(ground_h_m=2.5, slipstream=axial))
    assert gnd_axial["e"] == pytest.approx(gnd["e"], rel=1e-12)
    assert gnd_axial["CDi"] == pytest.approx(gnd["CDi"], rel=1e-12)
    assert gnd_axial["CDp"] > gnd["CDp"]
    # OPEN, measured and deliberately NOT asserted either way: the two halves
    # do not superpose. Adding the axial term on top of swirl moves CDi
    # (8.295e-3 -> 7.613e-3) and e (0.9589 -> 1.0448), which a scaling of the
    # PROFILE-drag integrand alone cannot do — yet with swirl off the same
    # term leaves both untouched to 1e-12 (asserted above). Either q_ratio
    # reaches the induced side only through the swirl branch, or the trimmed
    # alpha shifts and redistributes the load; this file cannot tell which and
    # must not pin a number it has not explained. See AUDIT_SESSION57.md.
    assert out["CDp"] > gnd_swirl["CDp"]
