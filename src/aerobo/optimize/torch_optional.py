"""Is PyTorch on this machine, and what to say to the user when it is not.

Most of this package's optimisers are pymoo and scipy: the GA, SLSQP, the
penalty method, the DOE runners and every grid search need no PyTorch at all.
Only the ones that fit a Gaussian process (``bo``, ``bo_slsqp``) or take an
exact gradient through the twin (``adjoint``) do.

That distinction is not academic, because PyTorch is the one dependency a
supported machine can be unable to install:

* there has been **no macOS x86_64 wheel since torch 2.2.2**, so an Intel Mac
  cannot have a current one at all; and
* the Apple-silicon wheel is tagged ``macosx_14_0``, so a Mac on Ventura or
  older cannot either.

Before this module existed, ``optimize.constrained`` and ``optimize.feasible``
imported torch at the top. ``feasible`` is reached by ``api.run`` on EVERY run
through ``_gate_grader``, and ``constrained`` holds the GA/SLSQP/penalty
runners for constrained problems as well as BO-LogCEI — so on those machines
the import took down not just Bayesian optimisation but every search in the
program, and the shell with it.

The import is still made at module import time where torch exists, so nothing
about a working machine changes — not the import order, not a monkeypatch
point, not a number. What changes is that its failure is caught and named.
"""

from __future__ import annotations

try:                                   # the normal case
    import torch
except ImportError as exc:             # a machine with no wheel for it
    torch = None                       # type: ignore[assignment]
    IMPORT_ERROR: ImportError | None = exc
else:
    IMPORT_ERROR = None

#: True where a GP-based or gradient-based optimiser can actually run.
HAVE_TORCH = torch is not None

#: Why it is missing, in the user's terms rather than pip's.
REASON = (
    "PyTorch is not installed in this environment. On macOS that is usually "
    "not something you did: PyTorch has published no wheel for Intel Macs "
    "since 2.2.2, and its Apple-silicon wheel needs macOS 14 or newer. "
    "Searches that do not fit a Gaussian process — GA, SLSQP, penalty, "
    "Sobol/DOE and grid — run normally without it."
)


def require(what: str) -> None:
    """Raise a legible error if ``what`` cannot run on this machine.

    Called at the top of the functions that genuinely need torch, so the
    failure names the optimiser the user chose instead of surfacing as an
    ``AttributeError`` on a ``None`` several frames deeper.
    """
    if torch is None:
        raise ModuleNotFoundError(f"{what} needs PyTorch. {REASON}")
