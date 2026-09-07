"""Machine capability, compute budget and honest wall-clock estimation.

What "how powerful is this laptop" can and cannot buy
----------------------------------------------------
MEASURED on the development machine (Apple arm64, 8 performance + 4
efficiency cores, 16 GB, macOS 15.5) on 2026-07-23 — see REFERENCE below:

* a constrained-BO acquisition step costs 0.19-0.34 s and is SERIAL by
  construction (fit the GPs, optimise the acquisition, pick ONE point);
* the physics under it costs 1.2 ms for a VLM/LLT evaluation and 1.37 s
  for a fresh viscous XFOIL sweep (0.2 ms when the sweep is cached).

So for every problem except the two live-XFOIL ones, the optimiser — not
the aerodynamics — is the wall clock, and extra cores cannot shorten a
sequential decision chain. Cores buy exactly two things here: parallel
XFOIL sweeps (the §15 database screen, airfoil_select.screen_database)
and independent seed batches. A control that claims more is a lie, and a
"performance" preset that quietly coarsens N/N_vlm/n_panel to look fast
would be trading answers for a saving invisible against BO overhead.

Hence the split this module enforces: :class:`ComputeBudget` changes ONLY
speed, never a number. Fidelity knobs live with the problem, are labelled
as answer-changing, and are not touched here.
"""

from __future__ import annotations

import os
import platform
import subprocess
from dataclasses import dataclass, replace

#: Measured cost constants. These are DATED, MACHINE-TAGGED measurements,
#: not guesses; an estimate built on them is a projection from this machine
#: to the user's, and is reported as a range for that reason.
REFERENCE = {
    "machine": "Apple arm64, 8 P-cores + 4 E-cores, 16 GB, macOS 15.5",
    "date": "2026-07-23",
    "s_xfoil_sweep_fresh": 1.37,      # median of 3 novel CST sections
    "s_xfoil_sweep_cached": 0.0002,   # disk-cache hit
    "s_eval_vlm": 0.0012,             # full 13-D eval, section cached
    #: constrained-BO seconds per acquisition step, by dimension
    "s_bo_iter": {3: 0.191, 5: 0.213, 8: 0.344, 13: 0.311},
}

#: workers the frozen §15 pipeline results were produced at
FROZEN_XFOIL_WORKERS = 6


@dataclass(frozen=True)
class MachineInfo:
    """What could be detected about this machine. Never raises."""

    logical: int = 1
    perf: int = 1                 # performance cores (== logical off Apple)
    eff: int = 0                  # efficiency cores
    ram_gb: float | None = None
    arch: str = "unknown"
    detected: bool = False

    @property
    def label(self) -> str:
        ram = f", {self.ram_gb:.0f} GB" if self.ram_gb else ""
        cores = (f"{self.perf}P + {self.eff}E" if self.eff
                 else f"{self.logical} cores")
        return f"{self.arch} · {cores}{ram}"


def _sysctl_int(key: str) -> int | None:
    """One sysctl probe, individually guarded.

    Each probe is isolated because this runs at UI-render time: a sandbox,
    a frozen app bundle or a non-macOS host can make `sysctl` missing,
    non-executable or slow, and none of those may raise into a UI slot.
    (`os.sched_getaffinity` is deliberately not used — it does not exist
    on macOS.)
    """
    try:
        out = subprocess.run(["sysctl", "-n", key], capture_output=True,
                             text=True, timeout=2.0)
        return int(out.stdout.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


_CACHED: MachineInfo | None = None


def detect(refresh: bool = False) -> MachineInfo:
    """Best-effort machine description; module-cached, dependency-free."""
    global _CACHED
    if _CACHED is not None and not refresh:
        return _CACHED
    logical = os.cpu_count() or 1
    perf = _sysctl_int("hw.perflevel0.logicalcpu")
    eff = _sysctl_int("hw.perflevel1.logicalcpu")
    mem = _sysctl_int("hw.memsize")
    try:
        arch = platform.machine() or "unknown"
    except Exception:                                    # noqa: BLE001
        arch = "unknown"
    _CACHED = MachineInfo(
        logical=logical,
        perf=perf if perf else logical,
        eff=eff or 0,
        ram_gb=(mem / 2 ** 30) if mem else None,
        arch=arch,
        detected=any(v is not None for v in (perf, eff, mem)),
    )
    return _CACHED


@dataclass(frozen=True)
class ComputeBudget:
    """Speed-only settings. Changing these must never change a result."""

    xfoil_workers: int = FROZEN_XFOIL_WORKERS
    seed_workers: int = 1          # 1 = today's serial run queue
    torch_threads: int | None = None   # None = leave torch alone (legacy)
    label: str = "balanced"

    def clamped(self, mi: MachineInfo | None = None) -> ComputeBudget:
        mi = mi or detect()
        return replace(
            self,
            xfoil_workers=int(max(1, min(self.xfoil_workers,
                                         max(1, mi.logical)))),
            seed_workers=int(max(1, self.seed_workers)),
        )


PRESETS: dict[str, ComputeBudget] = {
    # keep the laptop responsive while something else is running
    "quiet": ComputeBudget(xfoil_workers=2, seed_workers=1, label="quiet"),
    # the frozen-results configuration
    "balanced": ComputeBudget(xfoil_workers=FROZEN_XFOIL_WORKERS,
                              seed_workers=1, label="balanced"),
    # use the performance cores, leaving two for the UI and the OS
    "full": ComputeBudget(xfoil_workers=0, seed_workers=1, label="full"),
}


def auto(mi: MachineInfo | None = None) -> ComputeBudget:
    """Compute budget for THIS machine.

    Capped at :data:`FROZEN_XFOIL_WORKERS` so that on the reference machine
    auto-detection reproduces the configuration every frozen §15 result was
    produced at — a default that silently differs from the published runs
    would be the worst kind of convenience. Two cores are left for the UI
    and the OS, and efficiency cores are never counted: scheduling XFOIL
    onto them lengthens the wall-clock timeout window that
    ``run_xfoil_polar`` measures against.
    """
    mi = mi or detect()
    workers = max(1, min(FROZEN_XFOIL_WORKERS, mi.perf - 2))
    return ComputeBudget(xfoil_workers=workers, seed_workers=1,
                         label="auto")


def resolve(name: str, mi: MachineInfo | None = None) -> ComputeBudget:
    """Preset name -> ComputeBudget ("auto"/"full" resolve against the host)."""
    mi = mi or detect()
    if name == "auto":
        return auto(mi)
    budget = PRESETS.get(name, PRESETS["balanced"])
    if budget.xfoil_workers == 0:                        # "full"
        budget = replace(budget, xfoil_workers=max(1, mi.perf - 2))
    return budget.clamped(mi)


def bo_iter_seconds(dim: int) -> float:
    """Acquisition-step cost at dimension ``dim`` (interpolated REFERENCE)."""
    table = sorted(REFERENCE["s_bo_iter"].items())
    if dim <= table[0][0]:
        return float(table[0][1])
    if dim >= table[-1][0]:
        return float(table[-1][1])
    for (d0, t0), (d1, t1) in zip(table, table[1:]):
        if d0 <= dim <= d1:
            w = (dim - d0) / (d1 - d0)
            return float(t0 + w * (t1 - t0))
    return float(table[-1][1])


def estimate_seconds(n_evals: int, dim: int, optimiser: str,
                     s_per_eval: float, fresh_fraction: float = 1.0,
                     s_per_eval_cached: float | None = None) -> dict:
    """Wall-clock estimate for a run, as a RANGE.

    ``s_per_eval`` is the cost of an evaluation that does real work;
    ``fresh_fraction`` is how many of them are expected to (a cached XFOIL
    sweep costs ``s_per_eval_cached`` instead). Optimiser overhead is added
    only for BO — every other runner in the registry is negligible against
    the physics, which is itself part of the answer to "will more cores
    help".

    The +-40 % band is not decoration: the measured fresh-sweep spread on
    the reference machine alone was 0.91-2.01 s, and the projection to a
    different machine is wider still.
    """
    n_evals = max(0, int(n_evals))
    cached = (s_per_eval_cached if s_per_eval_cached is not None
              else REFERENCE["s_xfoil_sweep_cached"])
    frac = min(1.0, max(0.0, float(fresh_fraction)))
    physics = n_evals * (frac * float(s_per_eval) + (1.0 - frac) * cached)
    overhead = 0.0
    if optimiser in ("bo", "blocks"):
        # blocks runs BO inside each sub-run, on sub-problems of lower
        # dimension; charge the same per-decision cost
        overhead = n_evals * bo_iter_seconds(dim)
    mid = physics + overhead
    return {"seconds": mid, "low": 0.6 * mid, "high": 1.4 * mid,
            "physics_s": physics, "optimiser_s": overhead}


def human_time(seconds: float) -> str:
    s = float(max(0.0, seconds))
    if s < 90:
        return f"{s:.0f} s"
    if s < 5400:
        return f"{s / 60:.0f} min"
    return f"{s / 3600:.1f} h"
