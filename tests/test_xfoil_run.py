"""Gates for aerobo.xfoil_run — the one-shot XFOIL polar wrapper.

Fast MOCKED tests (monkeypatched subprocess.run, no XFOIL process):
  1. Parse a canned v6.99 PACC polar (real header + rows copied from
     data/airfoils/naca2412_re1e6.pol, incl. the extra Top_Itr/Bot_Itr
     columns) -> alpha/cl/cd/cm arrays correct, cm = column 5.
  2. Timeout -> graceful partial (rows written before the kill) or empty
     result; never an exception.
  3. Garbage polar output / no polar file at all -> empty result, no raise.
  4. Cache: second identical call is served from JSON (from_cache=True,
     byte-identical arrays, no second subprocess); key is invariant to
     1e-9 coordinate perturbations (1e-6 rounding) but not to 1e-3 ones.
  5. Failure contract: missing binary raises XfoilError (infrastructure);
     bad geometry NEVER raises (empty result -> caller applies -100).

REAL-XFOIL tests (skipped when /opt/homebrew/bin/xfoil is absent):
  6. NACA 2412 (inline 4-digit generator, closed TE) at Re 1e6,
     alpha 0/2/4 -> >= 2 converged rows, cl increasing, cd in (0.004, 0.02)
     (the .pol regression anchor has cd(0..4 deg) ~ 0.0056-0.0067).
     Measured 0.31 s -> NOT marked slow (< 5 s rule).
  7. Self-intersecting garbage loop -> empty result, no crash. XFOIL hangs
     on a prompt for this geometry and is killed by the 10 s timeout ->
     measured ~10 s -> marked slow.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from aerobo import xfoil_run
from aerobo.xfoil_run import (
    XfoilError,
    cache_key,
    run_xfoil_polar,
)

XFOIL_BIN = "/opt/homebrew/bin/xfoil"
needs_xfoil = pytest.mark.skipif(
    not Path(XFOIL_BIN).exists(), reason="local XFOIL binary not installed"
)

# Canned v6.99 PACC polar: header + first rows copied VERBATIM from the repo's
# real data/airfoils/naca2412_re1e6.pol (incl. Top_Itr/Bot_Itr columns).
CANNED_POL = """\

       XFOIL         Version 6.99

 Calculated polar for: NACA 2412

 1 1 Reynolds number fixed          Mach number fixed

 xtrf =   1.000 (top)        1.000 (bottom)
 Mach =   0.000     Re =     1.000 e 6     Ncrit =   9.000  9.000

   alpha    CL        CD       CDp       CM     Top_Xtr  Bot_Xtr  Top_Itr  Bot_Itr
  ------ -------- --------- --------- -------- -------- -------- -------- --------
  -2.000   0.0220   0.00659   0.00062  -0.0540   0.7828   0.2959  15.5904 117.0705
   0.000   0.2371   0.00564   0.00049  -0.0520   0.6516   0.6796  23.5819 139.8470
   2.000   0.4496   0.00578   0.00079  -0.0481   0.5256   0.9675  31.2665 157.0441
   4.000   0.6635   0.00683   0.00121  -0.0459   0.4368   0.9924  35.9088 159.6096
"""
CANNED_ROWS = {  # alpha: (cl, cd, cm)
    -2.0: (0.0220, 0.00659, -0.0540),
    0.0: (0.2371, 0.00564, -0.0520),
    2.0: (0.4496, 0.00578, -0.0481),
    4.0: (0.6635, 0.00683, -0.0459),
}


def _naca4_coords(code: str = "2412", n: int = 160) -> np.ndarray:
    """Inline NACA 4-digit generator (Abbott & von Doenhoff 1959, Sec. 6.4),
    closed-TE thickness polynomial (last coeff -0.1036), cosine x-spacing,
    XFOIL loop order TE-upper -> LE -> TE-lower. Kept local so this test does
    not depend on agent A's aerobo.airfoil landing."""
    m, p, t = int(code[0]) / 100.0, int(code[1]) / 10.0, int(code[2:]) / 100.0
    x = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, n)))
    yt = 5.0 * t * (
        0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1036 * x**4
    )
    yc = np.where(
        x < p, m / p**2 * (2 * p * x - x**2), m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x - x**2)
    )
    dyc = np.where(x < p, 2 * m / p**2 * (p - x), 2 * m / (1 - p) ** 2 * (p - x))
    th = np.arctan(dyc)
    up = np.column_stack([x - yt * np.sin(th), yc + yt * np.cos(th)])
    lo = np.column_stack([x + yt * np.sin(th), yc - yt * np.cos(th)])
    return np.vstack([up[::-1], lo[1:]])  # TE-upper -> LE -> TE-lower


class FakeXfoil:
    """subprocess.run stand-in: writes ``pol_text`` (if any) into the temp
    cwd exactly like PACC would, then returns / raises like subprocess."""

    def __init__(self, pol_text: str | None = None, raise_timeout: bool = False):
        self.pol_text = pol_text
        self.raise_timeout = raise_timeout
        self.calls = 0

    def __call__(self, cmd, input=None, timeout=None, cwd=None, **kwargs):
        self.calls += 1
        assert cwd is not None, "XFOIL must run with an explicit temp cwd"
        if self.pol_text is not None:
            (Path(cwd) / xfoil_run._POLAR_NAME).write_text(self.pol_text)
        if self.raise_timeout:
            raise subprocess.TimeoutExpired(cmd, timeout)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")


@pytest.fixture()
def coords() -> np.ndarray:
    # rounded to the cache-key resolution so the 1e-9 perturbation test is
    # deterministic (no coordinate sits on a 0.5e-6 rounding boundary)
    return _naca4_coords("2412").round(6)


# ---------------------------------------------------------------------------
# 1. mocked parse correctness
# ---------------------------------------------------------------------------


def test_parses_canned_v699_polar(monkeypatch, tmp_path, coords):
    fake = FakeXfoil(pol_text=CANNED_POL)
    monkeypatch.setattr(xfoil_run.subprocess, "run", fake)
    res = run_xfoil_polar(coords, 1e6, 0.0, [-2, 0, 2, 4], cache_dir=tmp_path)
    assert fake.calls == 1
    assert not res.from_cache
    assert res.n_requested == 4
    assert res.n_converged == 4
    np.testing.assert_allclose(res.alpha_deg, [-2.0, 0.0, 2.0, 4.0])
    for i, a in enumerate(res.alpha_deg):
        cl, cd, cm = CANNED_ROWS[float(a)]
        assert res.cl[i] == pytest.approx(cl)
        assert res.cd[i] == pytest.approx(cd)
        assert res.cm[i] == pytest.approx(cm)  # CM = PACC col 5, not CDp


def test_rows_sorted_even_if_file_unordered(monkeypatch, tmp_path, coords):
    # split sweep writes 0..hi then lo..-step: file order is NOT alpha order
    lines = CANNED_POL.splitlines()
    shuffled = "\n".join(lines[:12] + [lines[14], lines[15], lines[12], lines[13]])
    fake = FakeXfoil(pol_text=shuffled)
    monkeypatch.setattr(xfoil_run.subprocess, "run", fake)
    res = run_xfoil_polar(coords, 1e6, 0.0, [-2, 0, 2, 4], cache_dir=tmp_path)
    assert np.all(np.diff(res.alpha_deg) > 0)
    assert res.cl[list(res.alpha_deg).index(0.0)] == pytest.approx(0.2371)


# ---------------------------------------------------------------------------
# 2. timeout -> graceful partial / empty
# ---------------------------------------------------------------------------


def test_timeout_returns_partial_polar(monkeypatch, tmp_path, coords):
    partial = "\n".join(CANNED_POL.splitlines()[:14])  # header + 2 data rows
    fake = FakeXfoil(pol_text=partial, raise_timeout=True)
    monkeypatch.setattr(xfoil_run.subprocess, "run", fake)
    res = run_xfoil_polar(coords, 1e6, 0.0, [-2, 0, 2, 4], cache_dir=tmp_path)
    assert res.n_converged == 2
    np.testing.assert_allclose(res.alpha_deg, [-2.0, 0.0])
    # a timed-out (partial) result must NOT be cached
    assert not list(Path(tmp_path).glob("*.json"))


def test_timeout_with_no_polar_is_empty_not_raise(monkeypatch, tmp_path, coords):
    fake = FakeXfoil(pol_text=None, raise_timeout=True)
    monkeypatch.setattr(xfoil_run.subprocess, "run", fake)
    res = run_xfoil_polar(coords, 1e6, 0.0, [0, 2], cache_dir=tmp_path)
    assert res.n_converged == 0
    assert res.n_requested == 2
    # a timeout that produced NOTHING is the deterministic hung-prompt
    # garbage mode: it IS cached (tagged) so revisits don't re-burn 20 s
    assert list(Path(tmp_path).glob("*.json"))
    res2 = run_xfoil_polar(coords, 1e6, 0.0, [0, 2], cache_dir=tmp_path)
    assert res2.from_cache and res2.n_converged == 0
    assert fake.calls == 1  # no second subprocess


# ---------------------------------------------------------------------------
# 3. garbage output -> empty result, no exception
# ---------------------------------------------------------------------------


def test_garbage_polar_text_gives_empty_result(monkeypatch, tmp_path, coords):
    fake = FakeXfoil(pol_text="PLOP\nnot a polar\n1 2\nword 3 4 5 6 7\n")
    monkeypatch.setattr(xfoil_run.subprocess, "run", fake)
    res = run_xfoil_polar(coords, 1e6, 0.0, [0, 2, 4], cache_dir=tmp_path)
    assert res.n_converged == 0
    assert res.alpha_deg.size == res.cl.size == res.cd.size == res.cm.size == 0


def test_no_polar_file_gives_empty_result(monkeypatch, tmp_path, coords):
    fake = FakeXfoil(pol_text=None)  # XFOIL died before PACC wrote anything
    monkeypatch.setattr(xfoil_run.subprocess, "run", fake)
    res = run_xfoil_polar(coords, 1e6, 0.0, [0, 2, 4], cache_dir=tmp_path)
    assert res.n_converged == 0


def test_nonfinite_coords_fast_fail_without_subprocess(monkeypatch, tmp_path, coords):
    fake = FakeXfoil(pol_text=CANNED_POL)
    monkeypatch.setattr(xfoil_run.subprocess, "run", fake)
    bad = coords.copy()
    bad[7, 1] = np.nan
    res = run_xfoil_polar(bad, 1e6, 0.0, [0, 2], cache_dir=tmp_path)
    assert res.n_converged == 0
    assert fake.calls == 0  # NaN geometry never reaches XFOIL


def test_missing_binary_is_infrastructure_error(tmp_path, coords):
    with pytest.raises(XfoilError):
        run_xfoil_polar(
            coords, 1e6, 0.0, [0.0],
            xfoil_bin=str(tmp_path / "no_such_xfoil"), cache_dir=tmp_path,
        )


# ---------------------------------------------------------------------------
# 4. cache behaviour
# ---------------------------------------------------------------------------


def test_cache_second_call_no_subprocess(monkeypatch, tmp_path, coords):
    fake = FakeXfoil(pol_text=CANNED_POL)
    monkeypatch.setattr(xfoil_run.subprocess, "run", fake)
    first = run_xfoil_polar(coords, 1e6, 0.0, [-2, 0, 2, 4], cache_dir=tmp_path)
    second = run_xfoil_polar(coords, 1e6, 0.0, [-2, 0, 2, 4], cache_dir=tmp_path)
    assert fake.calls == 1  # second call served from JSON
    assert not first.from_cache
    assert second.from_cache
    for name in ("alpha_deg", "cl", "cd", "cm"):
        a, b = getattr(first, name), getattr(second, name)
        assert a.tobytes() == b.tobytes()  # byte-identical through the JSON trip


def test_cache_key_invariant_to_1e9_perturbation(coords):
    base = cache_key(coords, 1e6, 0.0, [0, 2, 4])
    assert cache_key(coords + 1e-9, 1e6, 0.0, [0, 2, 4]) == base  # < 1e-6 rounding
    assert cache_key(coords + 1e-3, 1e6, 0.0, [0, 2, 4]) != base  # real design move
    assert cache_key(coords, 2e6, 0.0, [0, 2, 4]) != base
    assert cache_key(coords, 1e6, 0.0, [0, 2, 6]) != base
    assert cache_key(coords, 1e6, 0.0, [0, 2, 4], n_panel=240) != base


def test_perturbed_coords_hit_same_cache_entry(monkeypatch, tmp_path, coords):
    fake = FakeXfoil(pol_text=CANNED_POL)
    monkeypatch.setattr(xfoil_run.subprocess, "run", fake)
    run_xfoil_polar(coords, 1e6, 0.0, [0, 2], cache_dir=tmp_path)
    res = run_xfoil_polar(coords + 1e-9, 1e6, 0.0, [0, 2], cache_dir=tmp_path)
    assert fake.calls == 1
    assert res.from_cache


def test_empty_but_completed_run_is_cached(monkeypatch, tmp_path, coords):
    fake = FakeXfoil(pol_text=None)  # deterministic failure (no timeout)
    monkeypatch.setattr(xfoil_run.subprocess, "run", fake)
    run_xfoil_polar(coords, 1e6, 0.0, [0, 2], cache_dir=tmp_path)
    res = run_xfoil_polar(coords, 1e6, 0.0, [0, 2], cache_dir=tmp_path)
    assert fake.calls == 1  # failure memoised: no 20 s re-fail per BO revisit
    assert res.from_cache and res.n_converged == 0


# ---------------------------------------------------------------------------
# 5. real XFOIL (skipped if binary absent)
# ---------------------------------------------------------------------------


@needs_xfoil  # 0.31 s measured: under the 5 s threshold, so not marked slow
def test_real_xfoil_naca2412(tmp_path, coords):
    res = run_xfoil_polar(
        coords, 1e6, 0.0, [0.0, 2.0, 4.0],
        xfoil_bin=XFOIL_BIN, cache_dir=tmp_path,
    )
    assert not res.from_cache
    assert res.n_requested == 3
    assert res.n_converged >= 2
    assert np.all(np.diff(res.cl) > 0)  # linear region: cl increasing
    assert np.all((res.cd > 0.004) & (res.cd < 0.02))
    # sanity vs the repo's regression-anchor polar (NACA-command geometry,
    # 160->200 panels differ slightly from our coordinate-file route)
    i0 = int(np.argmin(np.abs(res.alpha_deg)))
    assert res.cl[i0] == pytest.approx(0.2371, abs=0.05)
    assert res.cm[i0] == pytest.approx(-0.052, abs=0.01)


@needs_xfoil
@pytest.mark.slow
def test_real_xfoil_garbage_airfoil_no_crash(tmp_path):
    # self-intersecting figure-eight "airfoil": XFOIL must not crash the
    # wrapper; whatever happens inside (paneling failure, VISCAL divergence,
    # hung prompt killed by the timeout) maps to an empty result (-100 path).
    th = np.linspace(0.0, 2.0 * np.pi, 81)
    garbage = np.column_stack([0.5 + 0.5 * np.cos(th), 0.08 * np.sin(2.0 * th)])
    res = run_xfoil_polar(
        garbage, 1e6, 0.0, [0.0, 2.0],
        xfoil_bin=XFOIL_BIN, timeout_s=10.0, cache_dir=tmp_path,
    )
    assert res.n_converged == 0
    assert res.alpha_deg.size == 0
