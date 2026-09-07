"""``polar_at_re`` states its linear-region pair instead of re-fitting it.

The defect (found in session 42, reproduced and fixed here). The Reynolds
blend is carried on the INTERSECTION of the two members' alpha grids, so
``TablePolar.__post_init__`` re-fitted ``a_lin`` / ``alpha_L0`` in a window
clipped by *the other* member's convergence range. The re-fit then left the
convex hull of the two slopes it sits between — which no interpolation of two
numbers can do — and jumped across the bank nodes it is supposed to be
continuous through.

It contradicted the module's own stated convention: ``BlendedPolar`` blends
the pair, and ``BilinearPolar2D``'s docstring says it in as many words
("blended linearly, not re-extracted from the blended table").
``polar_at_re`` was the only blend in the file that re-extracted.

Why the DEFAULT moved rather than the fix shipping off: nothing in the
package reads ``a_lin`` off a blended polar today (``_cd_per_strip`` takes cd
only, by session 41's deliberate design), so no published number depends on
which path runs — and ``blend_slope=False`` reproduces the old one
bit-for-bit for anyone re-deriving an old figure. Both statements are
asserted below.
"""

import numpy as np
import pytest

from aerobo import polar

# frozen 2026-08-07 at t/c 0.06, Re 2.9e5, between members at 1e5 and 3e5
FROZEN_LEGACY_A_LIN = 7.196603229193632
FROZEN_MEMBER_A_LIN = (6.985581853364138, 6.877752086337346)


@pytest.fixture(scope="module")
def bank():
    b = polar.polar_re_bank()
    if not b:
        pytest.skip("no NACA 24XX Reynolds bank on disk")
    return b


def _segments(bank):
    for tc in sorted(bank):
        res = sorted(bank[tc])
        for lo, hi in zip(res[:-1], res[1:]):
            yield tc, lo, hi


def test_the_legacy_path_reproduces_the_frozen_number(bank):
    """``blend_slope=False`` is the old behaviour, to the last bit — the
    parameter is a way back, not an approximation of one."""
    got = polar.polar_at_re(0.06, 2.9e5, bank=bank, blend_slope=False)
    assert got.a_lin == FROZEN_LEGACY_A_LIN
    m = bank[0.06]
    assert (m[1e5].a_lin, m[3e5].a_lin) == FROZEN_MEMBER_A_LIN
    # ...and that number is ABOVE BOTH members, which is the defect itself
    assert got.a_lin > max(FROZEN_MEMBER_A_LIN)


def test_the_blended_slope_stays_in_the_hull_everywhere(bank):
    """An interpolation of two numbers lies between them. Over the whole
    bank the re-fit broke that on 61 of 375 log-spaced interior samples;
    the blend breaks it on none."""
    bad_new = bad_old = total = 0
    for tc, r_lo, r_hi in _segments(bank):
        lo_a, hi_a = sorted((bank[tc][r_lo].a_lin, bank[tc][r_hi].a_lin))
        for r in np.exp(np.linspace(np.log(r_lo), np.log(r_hi), 27))[1:-1]:
            new = polar.polar_at_re(tc, r, bank=bank).a_lin
            old = polar.polar_at_re(tc, r, bank=bank,
                                    blend_slope=False).a_lin
            bad_new += not (lo_a - 1e-12 <= new <= hi_a + 1e-12)
            bad_old += not (lo_a - 1e-12 <= old <= hi_a + 1e-12)
            total += 1
    assert total == 375
    assert bad_new == 0
    assert bad_old == 61           # the defect, measured, so the fix is not
    #                                a fix of nothing


def test_alpha_L0_is_blended_too_and_it_is_the_larger_effect(bank):
    """The OTHER half of the pair, and the half nothing asserted anything
    about until an adversarial review mutated it away and the whole suite
    stayed green.

    It is not the cosmetic half. In RELATIVE terms the zero-lift angle is
    worse than the slope: the re-fit leaves its own hull on 12 of 375
    samples and jumps 8.52 % across the worst bank node, against 4.49 % for
    ``a_lin``. The report leads with the slope because it is the quantity a
    consumer would read; the bigger discontinuity is here.
    """
    bad_new = bad_old = total = 0
    worst_new = worst_old = 0.0
    for tc, r_lo, r_hi in _segments(bank):
        lo_a, hi_a = sorted((bank[tc][r_lo].alpha_L0,
                             bank[tc][r_hi].alpha_L0))
        for r in np.exp(np.linspace(np.log(r_lo), np.log(r_hi), 27))[1:-1]:
            new = polar.polar_at_re(tc, r, bank=bank).alpha_L0
            old = polar.polar_at_re(tc, r, bank=bank,
                                    blend_slope=False).alpha_L0
            bad_new += not (lo_a - 1e-12 <= new <= hi_a + 1e-12)
            bad_old += not (lo_a - 1e-12 <= old <= hi_a + 1e-12)
            total += 1
    assert total == 375
    assert bad_new == 0
    assert bad_old == 12
    for tc in sorted(bank):
        res = sorted(bank[tc])
        for r in res[1:-1]:
            for kw, which in (({}, "new"), ({"blend_slope": False}, "old")):
                a = polar.polar_at_re(tc, r * (1 - 1e-6), bank=bank,
                                      **kw).alpha_L0
                b = polar.polar_at_re(tc, r * (1 + 1e-6), bank=bank,
                                      **kw).alpha_L0
                jump = abs(b - a) / abs(a)
                if which == "new":
                    worst_new = max(worst_new, jump)
                else:
                    worst_old = max(worst_old, jump)
    assert worst_new < 1e-5
    assert worst_old == pytest.approx(0.0852, abs=5e-4)
    # ...and it really is BLENDED, not merely "one of the two members":
    # taking p0's value would also be in-hull and continuous nowhere
    lo_r, hi_r = 1e5, 3e5
    mid = polar.polar_at_re(0.06, 1.7320508075688772e5, bank=bank)   # t=0.5
    m = bank[0.06]
    assert mid.alpha_L0 == pytest.approx(
        0.5 * (m[lo_r].alpha_L0 + m[hi_r].alpha_L0))
    assert mid.alpha_L0 != m[lo_r].alpha_L0


def test_the_slope_is_continuous_through_every_bank_node(bank):
    """A blend must agree with the member it is approaching. The re-fit
    jumped 4.49 % across the worst node."""
    worst_new = worst_old = 0.0
    for tc in sorted(bank):
        res = sorted(bank[tc])
        for r in res[1:-1]:
            for kw, name in (({}, "new"), ({"blend_slope": False}, "old")):
                lo = polar.polar_at_re(tc, r * (1 - 1e-6), bank=bank,
                                       **kw).a_lin
                hi = polar.polar_at_re(tc, r * (1 + 1e-6), bank=bank,
                                       **kw).a_lin
                jump = abs(hi - lo) / abs(lo)
                if name == "new":
                    worst_new = max(worst_new, jump)
                else:
                    worst_old = max(worst_old, jump)
    assert worst_new < 1e-5           # continuous to the probe's own step
    assert worst_old == pytest.approx(0.0449, abs=5e-4)


def test_a_bank_node_is_still_the_member_table_itself(bank):
    """The exactness promise the docstring makes, unchanged by any of this:
    asking at a bank Reynolds number returns the shipped polar object, so
    nothing that reaches this function by asking for the default point can
    drift."""
    for tc in sorted(bank):
        for r in sorted(bank[tc]):
            assert polar.polar_at_re(tc, r, bank=bank) is bank[tc][r]


def test_only_the_pair_moved(bank):
    """cl, cd and cm are untouched by the fix — it is a statement about two
    scalars, not a change to the tables."""
    new = polar.polar_at_re(0.09, 5.5e5, bank=bank)
    old = polar.polar_at_re(0.09, 5.5e5, bank=bank, blend_slope=False)
    assert np.array_equal(new.alpha_deg, old.alpha_deg)
    assert np.array_equal(new.CL, old.CL)
    assert np.array_equal(new.CD, old.CD)
    assert np.array_equal(new.CM, old.CM)
    assert new.a_lin != old.a_lin


# ---------------------------------------------- the override, on its own

def test_a_stated_pair_does_not_stop_a_degenerate_table_raising():
    """The override replaces the extracted value; it does not skip the
    extraction. A table that cannot define a linear region still fails at
    construction, where it always has — otherwise stating a slope would be a
    way to make a broken polar look valid."""
    with pytest.raises((ValueError, TypeError, np.linalg.LinAlgError,
                        IndexError)):
        polar.TablePolar(alpha_deg=np.array([0.0]), CL=np.array([0.5]),
                         CD=np.array([0.01]), CM=np.array([0.0]),
                         a_lin_override=6.28, alpha_L0_override=-0.03)


def test_no_override_is_the_published_behaviour(bank):
    """Every FILE-backed polar extracts, exactly as it always did."""
    m = bank[0.12][1e6]
    again = polar.TablePolar(alpha_deg=m.alpha_deg, CL=m.CL, CD=m.CD,
                             CM=m.CM, name=m.name, Re=m.Re)
    assert again.a_lin == m.a_lin
    assert again.alpha_L0 == m.alpha_L0


# ------------------------------------------- the Cp_min table travels too

def _synthetic(re, cp_offset):
    a = np.arange(-4.0, 10.5, 1.0)
    return polar.TablePolar(
        alpha_deg=a, CL=0.1 * a + 0.2, CD=0.01 + 0.0002 * a**2,
        CM=-0.05 + 0.0 * a, name=f"synthetic Re{re:.0e}", Re=re,
        alpha_cpmin=a, CPMIN=-1.0 - cp_offset - 0.05 * a)


def test_cp_min_survives_the_blend_when_both_members_have_one():
    """The secondary half of the same defect: the blended polar dropped the
    Cp_min table, so ``polar.cp_min()`` raised at Re 999999 and worked at
    Re 1e6 — and hydrofoil.py calls it unconditionally.

    Gated on SYNTHETIC members because the shipped bank has ``.cpmin`` files
    at Re 1e6 only, so no real pair can exercise it: this is the path that
    becomes live the moment the Cp_min bank grows, and an untested path is
    how the drop happened in the first place.
    """
    bank = {0.12: {1e5: _synthetic(1e5, 0.0), 1e6: _synthetic(1e6, 0.4)}}
    p0, p1 = bank[0.12][1e5], bank[0.12][1e6]
    # DELIBERATELY OFF THE MIDPOINT. The first version of this test probed at
    # t = 0.5, where (1-t)*p0 + t*p1 and t*p0 + (1-t)*p1 are the same number
    # — so it could not tell the two weightings apart, and a mutation that
    # flipped them survived the whole suite. An asymmetric t is the only
    # probe that identifies the weight.
    re_q = 1e5 * (10.0 ** 0.25)                       # t = 0.25 in log Re
    mid = polar.polar_at_re(0.12, re_q, bank=bank)
    assert mid.has_cp_min
    t = 0.25
    assert mid.cp_min(2.0) == pytest.approx(
        (1.0 - t) * p0.cp_min(2.0) + t * p1.cp_min(2.0))
    # ...and it is NOT the flipped weight, stated so the guard is explicit
    assert mid.cp_min(2.0) != pytest.approx(
        t * p0.cp_min(2.0) + (1.0 - t) * p1.cp_min(2.0))
    # the same weight cl gets, which is the claim the docstring makes
    assert mid.cl(2.0) == pytest.approx(
        (1.0 - t) * p0.cl(2.0) + t * p1.cl(2.0))


def test_cp_min_is_absent_when_either_member_lacks_one():
    """Never fabricated. One member without a table means there is nothing
    to interpolate, and the honest answer is the raise the caller already
    handles."""
    bank = {0.12: {1e5: _synthetic(1e5, 0.0), 1e6: _synthetic(1e6, 0.4)}}
    stripped = polar.TablePolar(
        alpha_deg=bank[0.12][1e6].alpha_deg, CL=bank[0.12][1e6].CL,
        CD=bank[0.12][1e6].CD, CM=bank[0.12][1e6].CM, Re=1e6)
    bank[0.12][1e6] = stripped
    mid = polar.polar_at_re(0.12, 3.0e5, bank=bank)
    assert not mid.has_cp_min
    with pytest.raises(ValueError, match="no Cp_min table"):
        mid.cp_min(2.0)
