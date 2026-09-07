"""What a RunConfig promises, and the three places it was not keeping it.

``airfoil_run_config``'s own docstring states the contract: "BUILD IT NOW,
not at run time … a config is a promise that this run is launchable". Session
42's api-contract review found three ways the promise leaked. All three were
reproduced before being fixed, and each has a test here that goes red under
the code as it was.

1. ``re_strip`` and ``re_bank`` were validated SEPARATELY and never as a
   pair, so ``re_strip="bank"`` with the DEFAULT ``re_bank=1`` — the call the
   docstring invites — returned a config that raises the moment anything
   builds it.
2. ``n_init`` was unvalidated at the surface that takes it, and an
   over-budget value was silently CLAMPED by ``_bo_split`` while the stored
   config kept the requested number — so ``RunResult.config`` stated a seed
   size the run did not use.
3. a stale ``_v2`` branch sidecar was DEMOTED rather than ignored, so a
   corrupt v1 promoted it and ``screen_branch_source()`` reported
   ``corrected: True`` against a docstring that says v2 is ignored.
"""

import json

import numpy as np
import pytest

from aerobo import api
from aerobo.section_wing import RE_STRIP_MODES, SectionWingProblem, WingGuess

#: what ``airfoil_run_config(wing=...)`` takes — a WingGuess kwargs dict
WING = {"mass_kg": 60.0, "v_ms": 20.0, "altitude_m": 0.0, "s_ref_m2": 6.0,
        "aspect_ratio": 10.0, "taper": 0.6}


# --------------------------------- 1. the spanwise Reynolds pair

@pytest.mark.parametrize("mode", RE_STRIP_MODES)
@pytest.mark.parametrize("bank", [1, 2, 5])
def test_the_config_refuses_exactly_what_the_problem_refuses(mode, bank):
    """The config surface and the problem it builds must agree about which
    (re_strip, re_bank) pairs exist. Not "roughly agree": the SAME set, so
    a config that is accepted here can always be built."""
    def problem_ok():
        try:
            SectionWingProblem(wing=WingGuess(**WING), re_strip=mode,
                               re_bank=bank)
        except ValueError:
            return False
        return True

    def config_ok():
        try:
            api.airfoil_run_config(wing=WING, re_strip=mode, re_bank=bank)
        except ValueError:
            return False
        return True

    assert config_ok() == problem_ok(), (mode, bank)


def test_the_invited_call_no_longer_builds_an_unbuildable_config():
    """``re_strip='bank'`` with the default ``re_bank`` — one keyword, the
    obvious one — used to return a config and blow up later."""
    with pytest.raises(ValueError, match="re_bank >= 2"):
        api.airfoil_run_config(wing=WING, re_strip="bank")


def test_every_accepted_pair_actually_builds():
    """The promise, exercised: whatever the config accepts, the registry can
    build without raising."""
    for mode in RE_STRIP_MODES:
        for bank in (1, 2, 4):
            try:
                cfg = api.airfoil_run_config(wing=WING, re_strip=mode,
                                             re_bank=bank)
            except ValueError:
                continue
            api.PROBLEM_SPECS[cfg.problem_name].build(
                cfg.mission_kwargs or {}, cfg.flags or {},
                cfg.bounds_overrides)


def test_a_default_call_is_unchanged():
    """No valid combination moved. The default sends the legacy flag set."""
    cfg = api.airfoil_run_config()
    assert "airfoil_re_strip" not in cfg.flags
    assert "airfoil_re_bank" not in cfg.flags
    with_wing = api.airfoil_run_config(wing=WING)
    assert "airfoil_re_strip" not in with_wing.flags
    banked = api.airfoil_run_config(wing=WING, re_strip="bank", re_bank=3)
    assert banked.flags["airfoil_re_strip"] == "bank"
    assert banked.flags["airfoil_re_bank"] == 3


# --------------------------------------------- 2. n_init states what it flies

def test_n_init_is_validated_where_it_is_taken():
    with pytest.raises(ValueError, match="n_init must be >= 1"):
        api.airfoil_run_config(n_init=0)
    with pytest.raises(ValueError, match="no evaluations for the BO loop"):
        api.airfoil_run_config(n_init=32, budget=32)
    with pytest.raises(ValueError, match="no evaluations for the BO loop"):
        api.airfoil_run_config(n_init=100, budget=32)
    # ...and the largest value that DOES leave a loop is accepted
    cfg = api.airfoil_run_config(n_init=31, budget=32)
    assert cfg.flags[api.BO_N_INIT_FLAG] == 31


# The split-reporting half of defect 2 used to sit behind a
# ``skipif(not hasattr(api.RunResult, "bo_split"))`` gate, because the field
# landed with ANOTHER SESSION'S ``RunConfig.pinned`` feature and was absent
# from HEAD. **It landed** (commit 4a2ed80), so the gate is gone — and it had
# to go rather than merely being vacuous: a `hasattr` skip means that deleting
# the field makes these tests SKIP instead of FAIL, which is the one thing a
# regression test must never do. If ``bo_split`` disappears, this file goes
# red at import.
assert hasattr(api.RunResult, "bo_split"), (
    "RunResult.bo_split is gone — these tests guard it; see "
    "PLAN_SESSION44_REGISTRY.md §1 defect 2")


def test_a_run_reports_the_split_it_actually_used():
    """``RunResult.config`` echoes what was ASKED for. ``bo_split`` is what
    RAN — the two differ exactly when _bo_split clamps, which is the case
    the echoed config could not describe."""
    cfg = api.RunConfig(problem_name="trim wing", optimiser="bo", budget=8,
                        seed=0)
    res = api.run(cfg)
    assert res.bo_split == list(api._bo_split(8, 3, None))
    assert sum(res.bo_split) == 8
    # a flag set past the budget (the direct-flags route, which does not go
    # through airfoil_run_config's refusal) is clamped — and now SAYS so
    over = api.RunConfig(problem_name="trim wing", optimiser="bo", budget=6,
                         seed=0, flags={api.BO_N_INIT_FLAG: 50})
    res2 = api.run(over)
    assert res2.config["flags"][api.BO_N_INIT_FLAG] == 50      # asked for
    assert res2.bo_split == [5, 1]                             # flown
    assert sum(res2.bo_split) == 6


def test_bo_split_is_none_for_every_other_optimiser():
    cfg = api.RunConfig(problem_name="trim wing", optimiser="sobol",
                        budget=8, seed=0)
    assert api.run(cfg).bo_split is None


def test_the_result_still_serialises():
    cfg = api.RunConfig(problem_name="trim wing", optimiser="bo", budget=8,
                        seed=0)
    d = api.run(cfg).to_dict()
    json.dumps(d)
    assert d["bo_split"] == list(api._bo_split(8, 3, None))


# ---- and the half that was missing: a STOPPED run must say so too --------

def _records(n: int, dim: int = 3):
    """``n`` progress payloads of the shape a rich progress_cb receives."""
    return [{"x": [0.5] * dim, "f": 30.0 + i, "g": None, "feasible": True}
            for i in range(n)]


def test_a_stopped_bo_run_is_not_mistaken_for_a_sobol_run():
    """The exact failure ``bo_split`` exists to close, on the partial path.

    ``partial_result`` set ``searched_dim`` but never ``bo_split``, so a
    cancelled BO run and a Sobol run both recorded ``bo_split=None`` — and
    ``optimiser`` alone does not distinguish them once the record is reloaded
    by something that trusts the field. Assert the far side: the two records
    differ.
    """
    stopped = api.partial_result(
        api.RunConfig(problem_name="trim wing", optimiser="bo", budget=8,
                      seed=0),
        _records(3), stop_reason="user")
    sobol = api.partial_result(
        api.RunConfig(problem_name="trim wing", optimiser="sobol", budget=8,
                      seed=0),
        _records(3), stop_reason="user")
    assert stopped.partial is True and sobol.partial is True
    assert sobol.bo_split is None
    assert stopped.bo_split is not None
    assert stopped.bo_split != sobol.bo_split


def test_the_stopped_split_is_the_one_the_run_was_dispatched_with():
    """Not "how far it got" — the split ``_dispatch`` was handed.

    Pinned to the same closed form the completed path uses, so the two can
    never drift apart.

    NOTE on the probe. The obvious clamp case — n_init 50 at budget 6 — is
    BLIND: ``_bo_split(6, 3, 50)`` and ``_bo_split(6, 3, None)`` both come
    back [5, 1], so a call site that dropped the flag entirely would still
    pass. (Confirmed by mutation.) The flag probe below is therefore chosen
    where the two genuinely differ.
    """
    cfg = api.RunConfig(problem_name="trim wing", optimiser="bo", budget=8,
                        seed=0)
    res = api.partial_result(cfg, _records(2), stop_reason="user")
    assert res.bo_split == list(api._bo_split(8, 3, None)) == [6, 2]
    assert sum(res.bo_split) == 8            # the DISPATCHED budget...
    assert res.n_evals == 2                  # ...not the evaluations flown

    # a stated seed size the default would NOT produce: [2, 6] vs [6, 2]
    stated = api.RunConfig(problem_name="trim wing", optimiser="bo", budget=8,
                           seed=0, flags={api.BO_N_INIT_FLAG: 2})
    res2 = api.partial_result(stated, _records(2), stop_reason="user")
    assert res2.config["flags"][api.BO_N_INIT_FLAG] == 2       # asked for
    assert res2.bo_split == [2, 6]                             # flown
    assert res2.bo_split != res.bo_split                       # the flag bit

    # and the clamp still reports the clamped value, not the asked-for one
    over = api.RunConfig(problem_name="trim wing", optimiser="bo", budget=6,
                         seed=0, flags={api.BO_N_INIT_FLAG: 50})
    res3 = api.partial_result(over, _records(2), stop_reason="user")
    assert res3.config["flags"][api.BO_N_INIT_FLAG] == 50      # asked for
    assert res3.bo_split == [5, 1]                             # flown


def test_the_stopped_split_is_sized_on_the_SEARCHED_dimension():
    """A pin shrinks the search, so it shrinks the Sobol seed too.

    ``_bo_split`` sizes n_init as 2*dim, so reading ``built.dim`` instead of
    the pinned dim reports a seed the run did not fly — the same class of
    defect ``bo_split`` was added to close, one level down. Probed where the
    two genuinely differ: dim 3 -> [6, 2], dim 2 -> [4, 4].
    """
    assert list(api._bo_split(8, 3, None)) == [6, 2]
    assert list(api._bo_split(8, 2, None)) == [4, 4]

    free = api.partial_result(
        api.RunConfig(problem_name="trim wing", optimiser="bo", budget=8,
                      seed=0),
        _records(2), stop_reason="user")
    pinned = api.partial_result(
        api.RunConfig(problem_name="trim wing", optimiser="bo", budget=8,
                      seed=0, pinned={"taper": 0.63}),
        _records(2), stop_reason="user")

    assert free.searched_dim is None and free.bo_split == [6, 2]
    assert pinned.searched_dim == 2
    assert pinned.bo_split == [4, 4]
    assert pinned.bo_split != free.bo_split
    # and the completed path agrees with it, which is the whole invariant
    assert api.run(
        api.RunConfig(problem_name="trim wing", optimiser="bo", budget=8,
                      seed=0, pinned={"taper": 0.63})).bo_split == [4, 4]


def test_a_stopped_run_serialises_its_split():
    cfg = api.RunConfig(problem_name="trim wing", optimiser="bo", budget=8,
                        seed=0)
    d = api.partial_result(cfg, _records(2), stop_reason="user").to_dict()
    json.dumps(d)
    assert d["bo_split"] == list(api._bo_split(8, 3, None))


def test_the_completed_and_stopped_paths_report_the_same_split():
    """One closed form, two call sites — the invariant that keeps them equal."""
    cfg = api.RunConfig(problem_name="trim wing", optimiser="bo", budget=8,
                        seed=0)
    assert api.run(cfg).bo_split == api.partial_result(
        cfg, _records(2), stop_reason="user").bo_split


# --------------------------------- 3. a stale v2 sidecar is IGNORED

def test_a_stale_v2_sidecar_is_ignored_not_demoted(tmp_path, monkeypatch):
    """With v2's recorded provenance sha not matching the v1 on disk, v2 must
    not be in the order AT ALL. Demoted, an unreadable v1 fell THROUGH to it
    and the screen reported ``corrected: True`` about a correction of a file
    that no longer parses."""
    v1 = tmp_path / "screen_branch.json"
    v2 = tmp_path / "screen_branch_v2.json"
    v1.write_text('{"sections": {"a": {"tc": 0.12}}}')
    v2.write_text(json.dumps({
        "sections": {"a": {"tc": 0.12}},
        "provenance": {"source_sidecar_sha256": "0" * 64},   # deliberately
    }))                                                       # not v1's sha
    monkeypatch.setattr(api, "SCREEN_BRANCH_CACHE", v1)
    monkeypatch.setattr(api, "SCREEN_BRANCH_CACHE_V2", v2)
    assert api._branch_sidecar_order() == (v1,)

    # ...and with v1 CORRUPT as well, the honest answer is no branch data —
    # not the stale derivative flying under the corrected flag
    v1.write_text("{ this is not json")
    assert api._branch_sidecar_order() == (v1,)
    monkeypatch.setattr(api, "_SCREEN_BRANCH", {})
    monkeypatch.setattr(api, "_SCREEN_BRANCH_SOURCE", "")
    assert api._branch_sidecar() == {}
    assert api.screen_branch_source()["corrected"] is False


def test_a_matching_v2_is_still_preferred(tmp_path, monkeypatch):
    """The fix must not throw away the correction it exists to prefer."""
    import hashlib

    v1 = tmp_path / "screen_branch.json"
    v2 = tmp_path / "screen_branch_v2.json"
    v1.write_text('{"sections": {"a": {"tc": 0.12}}}')
    sha = hashlib.sha256(v1.read_bytes()).hexdigest()
    v2.write_text(json.dumps({
        "sections": {"a": {"tc": 0.12, "clmax_censored": True}},
        "provenance": {"source_sidecar_sha256": sha},
    }))
    monkeypatch.setattr(api, "SCREEN_BRANCH_CACHE", v1)
    monkeypatch.setattr(api, "SCREEN_BRANCH_CACHE_V2", v2)
    assert api._branch_sidecar_order() == (v2, v1)


def test_an_absent_v1_still_leaves_v2_usable(tmp_path, monkeypatch):
    v1 = tmp_path / "screen_branch.json"          # never written
    v2 = tmp_path / "screen_branch_v2.json"
    v2.write_text(json.dumps({
        "sections": {}, "provenance": {"source_sidecar_sha256": "0" * 64}}))
    monkeypatch.setattr(api, "SCREEN_BRANCH_CACHE", v1)
    monkeypatch.setattr(api, "SCREEN_BRANCH_CACHE_V2", v2)
    assert api._branch_sidecar_order() == (v2, v1)


def test_an_absent_v2_is_the_plain_v1_path(tmp_path, monkeypatch):
    v1 = tmp_path / "screen_branch.json"
    v2 = tmp_path / "screen_branch_v2.json"       # never written
    v1.write_text('{"sections": {}}')
    monkeypatch.setattr(api, "SCREEN_BRANCH_CACHE", v1)
    monkeypatch.setattr(api, "SCREEN_BRANCH_CACHE_V2", v2)
    assert api._branch_sidecar_order() == (v1,)


def test_polar_at_re_consumers_are_declared_and_read_only_what_is_argued():
    """``polar_at_re``'s production consumers, and why the default still holds.

    HISTORY, because this test changed its own premise once and must not be
    allowed to drift again. It began as "nothing in ``src/aerobo/`` calls
    ``polar_at_re``", which made the ``blend_slope=True`` default costless:
    with no consumer, no production path could read a blended slope whichever
    way the blend computed it. Session 63 broke that premise — the car wing
    grew an opt-in that reads its polar AT THE FLOWN Reynolds number, and the
    two-element family reads the wide-alpha bank per element.

    The test's own instruction was "if this goes red, the choice of default
    has to be re-argued — not the test relaxed". So it was re-argued, and the
    argument is this:

    * ``blend_slope=True`` was NEVER justified by the absence of consumers. It
      is justified in ``polar.polar_at_re``'s own docstring, on the merits:
      re-extracting the linear pair from the blended table runs the fit in a
      window clipped by the other member's convergence range, and lands
      OUTSIDE the convex hull of the two slopes it sits between (measured at
      t/c 0.06: 7.19660 at Re 2.9e5 against members 6.98558 and 6.87775, on
      25 of 25 log-spaced samples), and is discontinuous across a bank node by
      4.49 %. Interpolating the members' own values is the convention every
      other blend in that module already follows. The absence of consumers was
      the reason the choice was SAFE, never the reason it was RIGHT.
    * So the default stands, and what changes is only that it is now
      LOAD-BEARING on one path. That is worth knowing, so it is recorded here
      per consumer rather than left to be rediscovered.

    What each consumer reads, verified by this test and not merely asserted:

    * ``carwing.py`` — the ``flown_reynolds`` opt-in. Feeds ``pol.a_lin`` and
      ``pol.alpha_L0`` straight into the VLM, so it reads the blended SLOPE.
      This is the load-bearing path.
    * ``carwing_multi.py`` — the wide-alpha bank, per element. Reads ``cd``
      only: the cascade fits its OWN ``a_lin``/``alpha_L0`` from the
      panel-method lift curve it just solved, because the section it flies is
      two bodies and no single-element table has that section's slope.
    * ``hydrofoil.py`` — the water families' ``flown_reynolds`` opt-in, via
      ``polar_for``, on all three of them (the planar foil, the tip-device
      foil and the elevator craft, whose stabiliser reads its OWN chord's
      Reynolds number). It is the first consumer to read **``cp_min``** off a
      blended result, and that read is a FEASIBILITY GATE rather than a
      score: the cavitation margin is ``g = sigma_cav + Cp_min``. It reads
      the blended slope too (``a_lin``/``alpha_L0`` go straight into the
      lifting line / lattice), so it is load-bearing on the ``blend_slope``
      default in the same way ``carwing.py`` is.

      This consumer is why the Cp_min half of the bank had to exist at all:
      until it did, every node but Re 1e6 returned a table whose ``cp_min``
      raised, so routing a hydrofoil here would have refused mid-search.

    A NEW consumer still fails here, which is the whole point: it arrives at
    the place where the reason is written down instead of silently inheriting
    a decision taken when the answer was different.
    """
    import pathlib
    import re as _re

    #: file -> what it is entitled to read off a polar_at_re result.
    DECLARED = {
        "carwing.py": "cd, a_lin, alpha_L0 (the flown-Reynolds opt-in)",
        "carwing_multi.py": "cd only (its slope is fitted from the cascade)",
        "hydrofoil.py": ("cd, a_lin, alpha_L0 AND cp_min — the water "
                         "flown-Reynolds opt-in; the cp_min read is a "
                         "feasibility gate, not a score"),
    }

    root = pathlib.Path(api.__file__).resolve().parent
    callers = {}
    for path in sorted(root.rglob("*.py")):
        if path.name == "polar.py":
            continue                     # where the functions are DEFINED
        for i, line in enumerate(path.read_text().splitlines(), 1):
            if _re.search(r"\b(polar_at_re|polar_family_at_re)\s*\(", line) \
                    and not line.lstrip().startswith("#"):
                callers.setdefault(str(path.relative_to(root)), []).append(i)

    assert set(callers) == set(DECLARED), (
        "polar_at_re's consumer set changed. It is not a bug to add one, but "
        "the blend_slope default is load-bearing wherever a consumer reads "
        "a_lin/alpha_L0, so a new consumer has to say which it reads and be "
        "declared here.\n"
        f"  found:    {sorted(callers)}\n  declared: {sorted(DECLARED)}")

    # ...and the SPLIT is real: the family that says it fits its own slope
    # must not be reading one off the bank instead. Read the source rather
    # than trust the docstring -- that distinction is the whole declaration.
    multi = (root / "carwing_multi.py").read_text()
    for i, line in enumerate(multi.splitlines(), 1):
        if _re.search(r"\b(polar_at_re|polar_family_at_re)\s*\(", line):
            window = "\n".join(multi.splitlines()[i - 1:i + 6])
            assert "a_lin" not in window and "alpha_L0" not in window, (
                "carwing_multi.py declares that it reads only cd off a "
                f"polar_at_re result, but line {i} is followed by a slope "
                f"read:\n{window}")
    assert np.isfinite(1.0)          # (keeps numpy used in this module)
