"""The program must open, and must still search, on a machine with no PyTorch.

This is not hypothetical. PyTorch has published no macOS x86_64 wheel since
2.2.2, and its Apple-silicon wheel is tagged ``macosx_14_0`` — so an Intel Mac
and a Mac on Ventura or older cannot install a current one, and both are
machines the downloadable app supports. Before ``optimize/torch_optional.py``,
``optimize.feasible`` imported torch at the top and ``api.run`` reaches that
module on EVERY run through ``_gate_grader``; ``optimize.constrained`` did the
same while holding the GA, SLSQP, penalty and DOE runners for constrained
problems. The measured result was that such a machine could not run a single
search of any kind, and the shell would not open.

Everything here runs in a SUBPROCESS with an import blocker in front of torch,
because this test session has torch imported already and ``sys.modules`` would
hand it back. The blocker is the honest simulation: the wheel is absent.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: Pretend the wheel was never installed. A meta-path finder is used rather
#: than deleting from sys.modules because sub-imports (botorch -> torch) must
#: fail too, at whatever depth they are attempted.
BLOCKER = """
import sys

BLOCKED = ("torch", "botorch", "gpytorch", "linear_operator", "pyro")

class _NoTorch:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ModuleNotFoundError("No module named " + repr(name), name=name)
        return None

sys.meta_path.insert(0, _NoTorch())
sys.path[:0] = [REPO_PATH, SRC_PATH]
"""


def _run_without_torch(body: str) -> str:
    header = (f"REPO_PATH = {str(REPO)!r}\nSRC_PATH = {str(REPO / 'src')!r}\n")
    code = header + BLOCKER + textwrap.dedent(body)
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True, timeout=900)
    assert proc.returncode == 0, (
        f"the torch-free subprocess failed:\n{proc.stdout}\n{proc.stderr}")
    return proc.stdout


def test_the_api_imports_with_no_torch():
    """``import aerobo.api`` is the shell's front door: if it raises, the app
    does not open at all, whatever the user meant to search."""
    out = _run_without_torch("""
        import importlib.util
        assert importlib.util.find_spec is not None
        try:
            import torch
        except ModuleNotFoundError:
            pass
        else:
            raise AssertionError("the blocker did not block torch")
        import aerobo.api as api
        print("IMPORTED", len(api.OPTIMISER_SPECS))
    """)
    assert out.startswith("IMPORTED")


def test_the_strategy_list_drops_what_cannot_run_and_says_so():
    """Both halves matter. Dropping alone leaves a list that silently got
    shorter; the note alone leaves a name that fails when chosen."""
    out = _run_without_torch("""
        import aerobo.api as api
        for prob in ("trim wing", "free planform (aircraft)"):
            names = api.compatible_optimisers(prob)
            assert not (set(names) & set(api.TORCH_ONLY_OPTIMISERS)), (prob, names)
            assert names, prob
        note = api.torch_optimiser_note()
        assert "PyTorch" in note and note, note
        print("FILTERED")
    """)
    assert "FILTERED" in out


def test_a_search_still_runs_unconstrained_and_constrained():
    """The point of the whole exercise: a machine with no GP library is not a
    machine with no optimiser. One run each side of the constrained split."""
    out = _run_without_torch("""
        from aerobo.api import RunConfig, run, compatible_optimisers
        for prob, opt in (("trim wing", "ga"),
                          ("free planform (aircraft)", "slsqp")):
            assert opt in compatible_optimisers(prob)
            res = run(RunConfig(problem_name=prob, optimiser=opt, budget=8, seed=0))
            assert res is not None
            print("RAN", prob, opt)
    """)
    assert out.count("RAN") == 2


def test_choosing_a_gp_optimiser_anyway_names_the_reason():
    """A caller reaching past the filter — a saved record, a script, an older
    session file — must get a sentence naming PyTorch, not an AttributeError
    on a None several frames down."""
    out = _run_without_torch("""
        from aerobo.api import RunConfig, run
        try:
            run(RunConfig(problem_name="free planform (aircraft)",
                          optimiser="bo", budget=8, seed=0))
        except ModuleNotFoundError as exc:
            assert "PyTorch" in str(exc), str(exc)
            print("NAMED")
        else:
            raise AssertionError("constrained BO ran without torch")
    """)
    assert "NAMED" in out
