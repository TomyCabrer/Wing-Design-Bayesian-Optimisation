"""Prandtl-Glauert compressibility on the SECTION lift-curve slope (llt.py).

Gates for the additive compressibility term:
  - pg_beta = sqrt(1 - M^2): exact at M = 0 and M = 0.5, ValueError in the
    transonic / negative range (caller maps to PENALTY).
  - swept_section_slope reduces BIT-FOR-BIT to the legacy a0 cos Λ at mach = 0
    (the PHASE INVARIANT: default reproduces current results by == equality).
  - sign sanity: 1/beta is monotone in M, so at fixed sweep the slope RISES
    with Mach (this project's bug history is sign errors — every term is
    checked against a known limit).
  - the compressibility uses the NORMAL Mach M cos Λ (simple-sweep theory),
    not the free-stream M: at Λ = 60 deg, M = 0.6 the factor sees 0.3.
"""

import numpy as np
import pytest

from aerobo.llt import PG_MACH_MAX, pg_beta, swept_section_slope


# ---------------- pg_beta ----------------


def test_pg_beta_incompressible_is_exactly_one():
    # M = 0 must give beta = 1.0 EXACTLY (a -> a/beta is the identity there)
    assert pg_beta(0.0) == 1.0


def test_pg_beta_half_mach_exact_expression():
    # beta(0.5) = sqrt(1 - 0.25) = sqrt(0.75), bit-for-bit
    assert pg_beta(0.5) == float(np.sqrt(0.75))


def test_pg_beta_rises_the_slope_and_is_monotone():
    # a -> a / beta, beta < 1 for 0 < M < 1, so the slope increases with M
    a = 2 * np.pi
    betas = [pg_beta(m) for m in (0.0, 0.2, 0.4, 0.6)]
    assert betas[0] == 1.0
    assert np.all(np.diff(betas) < 0)          # beta falls with M
    slopes = [a / b for b in betas]
    assert np.all(np.diff(slopes) > 0)         # a/beta rises with M


@pytest.mark.parametrize("mach", [0.7, 0.9, 1.0, 2.0])
def test_pg_beta_transonic_raises(mach):
    # M >= 0.7 is outside the linearised subsonic regime
    with pytest.raises(ValueError):
        pg_beta(mach)


@pytest.mark.parametrize("mach", [-1e-9, -0.1, -1.0])
def test_pg_beta_negative_raises(mach):
    with pytest.raises(ValueError):
        pg_beta(mach)


def test_pg_mach_max_is_the_ceiling():
    # exactly at the ceiling raises (>=), just below it is fine
    with pytest.raises(ValueError):
        pg_beta(PG_MACH_MAX)
    assert pg_beta(PG_MACH_MAX - 1e-6) > 0.0


# ---------------- swept_section_slope: mach = 0 bit-for-bit ----------------


@pytest.mark.parametrize("sweep_deg", [0.0, 10.0, 25.0, 45.0, 60.0])
def test_swept_slope_mach0_is_legacy_bit_for_bit(sweep_deg):
    a0 = 6.283185307179586
    legacy = float(swept_section_slope(a0, sweep_deg))
    explicit = float(swept_section_slope(a0, sweep_deg, mach=0.0))
    assert explicit == legacy                          # == float equality
    # and it is exactly the closed-form legacy expression
    assert legacy == float(a0 * np.cos(np.deg2rad(sweep_deg)))


# ---------------- swept_section_slope: compressibility sign ----------------


@pytest.mark.parametrize("sweep_deg", [0.0, 20.0, 40.0])
def test_swept_slope_increases_with_mach(sweep_deg):
    a0 = 6.0
    machs = [0.0, 0.15, 0.3, 0.45, 0.6]
    vals = [float(swept_section_slope(a0, sweep_deg, mach=m)) for m in machs]
    # strictly rising with Mach at fixed sweep (1/beta_n monotone)
    assert np.all(np.diff(vals) > 0)
    # every M > 0 value exceeds the incompressible baseline
    assert all(v > vals[0] for v in vals[1:])


def test_swept_slope_zero_sweep_is_pure_prandtl_glauert():
    # at Λ = 0 the normal Mach is the free-stream Mach: a_eff = a0 / beta(M)
    a0 = 5.7
    for m in (0.1, 0.3, 0.55):
        got = float(swept_section_slope(a0, 0.0, mach=m))
        assert got == pytest.approx(a0 / pg_beta(m))


def test_swept_slope_uses_normal_mach_not_freestream():
    """L = 60 deg, M = 0.6: the PG factor must see M cos L = 0.3, not 0.6."""
    a0 = 6.1
    L = 60.0
    M = 0.6
    cosL = np.cos(np.deg2rad(L))                        # = 0.5
    Mn = M * cosL                                       # = 0.3
    assert Mn == pytest.approx(0.3)

    got = float(swept_section_slope(a0, L, mach=M))
    expected_normal = a0 * cosL / np.sqrt(1.0 - Mn**2)      # sqrt(0.91)
    wrong_freestream = a0 * cosL / np.sqrt(1.0 - M**2)      # sqrt(0.64)

    assert got == pytest.approx(expected_normal)
    # the two forms are numerically distinct — guards against the M-vs-Mn bug
    assert not got == pytest.approx(wrong_freestream)
    assert got < wrong_freestream                      # sweep relieves PG
