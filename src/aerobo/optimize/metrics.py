"""Convergence metrics + plotting for optimiser benchmarking.

All optimisers (bo.run_bo, baselines.*) return objects exposing
`y` (eval-order objective history) and `best_so_far`; these helpers
aggregate across seeds and render the BO-vs-baselines comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class RunHistory:
    """Common result container for non-BO optimisers."""

    X: np.ndarray
    y: np.ndarray
    best_x: np.ndarray
    best_y: float
    best_so_far: np.ndarray
    name: str = ""
    seed: int = 0
    meta: dict = field(default_factory=dict)


def make_history(X, y, name: str, seed: int, maximize: bool = True,
                 n_evals: int | None = None, **meta) -> RunHistory:
    """Build a RunHistory, truncating to the eval budget for fair comparison."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if n_evals is not None:
        X, y = X[:n_evals], y[:n_evals]
    running = np.maximum.accumulate(y) if maximize else np.minimum.accumulate(y)
    i_best = int(np.argmax(y)) if maximize else int(np.argmin(y))
    return RunHistory(X=X, y=y, best_x=X[i_best], best_y=float(y[i_best]),
                      best_so_far=running, name=name, seed=seed, meta=dict(meta))


def evals_to_target(y: np.ndarray, target: float, maximize: bool = True) -> int | None:
    """First evaluation count (1-based) reaching the target, or None."""
    y = np.asarray(y, dtype=float)
    hit = y >= target if maximize else y <= target
    idx = np.flatnonzero(hit)
    return int(idx[0]) + 1 if idx.size else None


def aggregate_best_so_far(curves: list[np.ndarray]) -> dict:
    """Median + interquartile band of best-so-far curves (truncated to min len)."""
    n = min(len(c) for c in curves)
    arr = np.vstack([np.asarray(c[:n], dtype=float) for c in curves])
    return {
        "n_evals": np.arange(1, n + 1),
        "median": np.median(arr, axis=0),
        "q25": np.quantile(arr, 0.25, axis=0),
        "q75": np.quantile(arr, 0.75, axis=0),
        "n_seeds": arr.shape[0],
    }


def convergence_plot(results: dict[str, list], path: str | Path,
                     title: str = "", ylabel: str = "best objective",
                     target: float | None = None) -> Path:
    """Median + IQR best-so-far per optimiser. `results` maps
    optimiser name -> list of run objects (with .best_so_far)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, runs in results.items():
        agg = aggregate_best_so_far([r.best_so_far for r in runs])
        (line,) = ax.plot(agg["n_evals"], agg["median"], label=f"{name} (n={agg['n_seeds']})")
        ax.fill_between(agg["n_evals"], agg["q25"], agg["q75"],
                        alpha=0.2, color=line.get_color())
    if target is not None:
        ax.axhline(target, ls="--", c="k", lw=0.8, label="target")
    ax.set_xlabel("evaluations")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
