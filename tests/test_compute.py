"""Machine detection, compute budget and runtime estimation (compute.py).

The contract that matters here is a NEGATIVE one: nothing in this module may
change a result. It only decides how many XFOIL sweeps run at once and what
wall-clock number the shell prints.
"""

import pytest

from aerobo import compute


def test_detect_never_raises_and_always_answers():
    mi = compute.detect(refresh=True)
    assert mi.logical >= 1
    assert mi.perf >= 1
    assert mi.eff >= 0
    assert isinstance(mi.arch, str) and mi.arch
    assert isinstance(mi.label, str) and mi.label
    if mi.ram_gb is not None:
        assert mi.ram_gb > 0.0


def test_detect_survives_a_broken_sysctl(monkeypatch):
    """Probes run at UI-render time: a sandbox, a frozen bundle or a
    non-macOS host must degrade to a fallback, never raise into a slot."""
    def boom(*_a, **_k):
        raise OSError("no sysctl here")

    monkeypatch.setattr(compute.subprocess, "run", boom)
    mi = compute.detect(refresh=True)
    assert mi.detected is False
    assert mi.logical >= 1 and mi.perf >= 1
    compute.detect(refresh=True)          # restore the cache for other tests


def test_presets_and_auto_stay_within_the_machine():
    mi = compute.detect(refresh=True)
    for name in ("quiet", "balanced", "full", "auto"):
        b = compute.resolve(name, mi)
        assert 1 <= b.xfoil_workers <= max(1, mi.logical)
        assert b.seed_workers >= 1
        # torch threading is NOT enabled by default: its bit-stability is
        # empirical only, and a "free speedup" that moved a headline number
        # would be the worst possible trade
        assert b.torch_threads is None


def test_auto_reproduces_the_frozen_pipeline_configuration():
    """auto() must not silently differ from the configuration the frozen
    §15 results were produced at."""
    mi = compute.detect(refresh=True)
    b = compute.auto(mi)
    assert b.xfoil_workers <= compute.FROZEN_XFOIL_WORKERS
    if mi.perf >= compute.FROZEN_XFOIL_WORKERS + 2:
        assert b.xfoil_workers == compute.FROZEN_XFOIL_WORKERS


def test_unknown_preset_falls_back_rather_than_raising():
    b = compute.resolve("nonsense-preset")
    assert b.xfoil_workers >= 1


def test_bo_iter_seconds_interpolates_and_clamps():
    table = compute.REFERENCE["s_bo_iter"]
    lo_d, hi_d = min(table), max(table)
    assert compute.bo_iter_seconds(lo_d) == table[lo_d]
    assert compute.bo_iter_seconds(hi_d) == table[hi_d]
    assert compute.bo_iter_seconds(1) == table[lo_d]      # clamped
    assert compute.bo_iter_seconds(99) == table[hi_d]     # clamped
    mid = compute.bo_iter_seconds(4)
    assert min(table[3], table[5]) <= mid <= max(table[3], table[5])


def test_estimate_is_a_range_and_separates_physics_from_optimiser():
    slow = compute.estimate_seconds(
        40, 13, "blocks", compute.REFERENCE["s_xfoil_sweep_fresh"])
    assert slow["low"] < slow["seconds"] < slow["high"]
    assert slow["physics_s"] > slow["optimiser_s"]        # XFOIL dominates

    # ...but for a millisecond solver the OPTIMISER is the wall clock, which
    # is the whole reason "more cores" cannot be promised for these problems
    fast = compute.estimate_seconds(40, 5, "bo",
                                    compute.REFERENCE["s_eval_vlm"])
    assert fast["optimiser_s"] > 100 * fast["physics_s"]

    # a non-BO runner carries no acquisition overhead
    cheap = compute.estimate_seconds(40, 5, "sobol",
                                     compute.REFERENCE["s_eval_vlm"])
    assert cheap["optimiser_s"] == 0.0
    assert compute.estimate_seconds(0, 5, "bo", 1.0)["seconds"] == 0.0


def test_cached_sweeps_are_charged_at_the_cache_rate():
    """fresh_fraction is the honest knob: a re-run over cached sections is
    nearly free, and the estimate has to say so."""
    n, d = 40, 13
    s = compute.REFERENCE["s_xfoil_sweep_fresh"]
    all_fresh = compute.estimate_seconds(n, d, "blocks", s, fresh_fraction=1.0)
    all_cached = compute.estimate_seconds(n, d, "blocks", s,
                                          fresh_fraction=0.0)
    assert all_cached["physics_s"] < 0.01 * all_fresh["physics_s"]


@pytest.mark.parametrize("seconds,expect", [
    (5.0, "5 s"), (45.0, "45 s"), (600.0, "10 min"), (7200.0, "2.0 h"),
])
def test_human_time(seconds, expect):
    assert compute.human_time(seconds) == expect


def test_reference_constants_are_documented_measurements():
    """These drive a number the user reads; they must stay tagged with what
    machine and when, or the estimate becomes folklore."""
    ref = compute.REFERENCE
    assert ref["machine"] and ref["date"]
    assert ref["s_xfoil_sweep_fresh"] > ref["s_xfoil_sweep_cached"]
    assert ref["s_eval_vlm"] < ref["s_xfoil_sweep_fresh"]
    assert all(v > 0 for v in ref["s_bo_iter"].values())
    # the finding that motivates the whole honest framing
    assert min(ref["s_bo_iter"].values()) > 10 * ref["s_eval_vlm"]
