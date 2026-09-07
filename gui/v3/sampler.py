"""Live named-metric sampling of a running search's incumbent.

A run's progress log carries the objective, the constraint margins and the
design vector — not L/D, Cl max, span efficiency or a static margin. Those
live in the breakdown, and asking for a breakdown on EVERY evaluation would
double the cost of the search (an XFOIL family would pay a second viscous
sweep per candidate).

So this samples the INCUMBENT instead: whenever the best-so-far design
changes, one ``api.design_report`` is run on it in a background thread and
every named metric it reports is appended to a series. On the XFOIL
families that call is usually near-free — the incumbent was just evaluated,
so its polar is already in the sweep cache — and on the reduced-order
families it is milliseconds. Nothing is ever inferred: a metric appears
only when the breakdown that sample came from actually reported it.

The sampler owns a thread and a list; the UI polls. It never touches a
nicegui element.
"""

from __future__ import annotations

import threading
import time


class IncumbentSampler:
    """Samples ``api.design_report`` at the incumbent while a search runs.

    ``cfg_fn()`` returns the ``api.RunConfig`` the run was launched with.
    ``incumbent_fn()`` returns ``(n_evals, best_x)`` — or ``(n, None)``
    while there is no incumbent yet. Sampling stops when ``running_fn()``
    goes false and the last incumbent has been sampled.
    """

    def __init__(self, min_interval_s: float = 1.0):
        self.min_interval_s = float(min_interval_s)
        self.samples: list[dict] = []      # [{"n": int, "metrics": {k: v}}]
        self.error: str | None = None
        self.version = 0                   # bumped per sample (UI dirty flag)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    # ------------------------------------------------------------- control
    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def reset(self):
        self.samples = []
        self.error = None
        self.version += 1

    def start(self, cfg_fn, incumbent_fn, running_fn):
        if self.running:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._work, args=(cfg_fn, incumbent_fn, running_fn),
            daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    # -------------------------------------------------------------- worker
    def _work(self, cfg_fn, incumbent_fn, running_fn):
        last_x = None
        while not self._stop.is_set():
            alive = bool(running_fn())
            try:
                n, x = incumbent_fn()
            except Exception as exc:                       # noqa: BLE001
                self.error = f"{type(exc).__name__}: {exc}"
                return
            if x is not None and list(x) != last_x:
                last_x = list(x)
                self._sample(cfg_fn, n, x)
            if not alive:
                return
            time.sleep(self.min_interval_s)

    def _sample(self, cfg_fn, n: int, x):
        from aerobo import api

        from gui import metrics as metric_catalogue

        try:
            report = api.design_report(cfg_fn(), x)
        except Exception as exc:      # noqa: BLE001 — a view, never fatal
            self.error = f"{type(exc).__name__}: {exc}"
            return
        bd = metric_catalogue.enrich(report.get("breakdown") or {},
                                     report.get("geometry") or {})
        values = {k: float(v) for k, v in bd.items()
                  if isinstance(v, (int, float)) and not isinstance(v, bool)}
        if not values:
            return
        self.samples.append({"n": int(n), "metrics": values})
        self.error = None
        self.version += 1

    # --------------------------------------------------------------- views
    def keys(self) -> list[str]:
        """Metric keys seen so far, in the order they first appeared."""
        out: list[str] = []
        for s in self.samples:
            for k in s["metrics"]:
                if k not in out:
                    out.append(k)
        return out

    def series(self, key: str) -> tuple[list, list]:
        """``(evaluations, values)`` for one metric, gaps dropped."""
        xs, ys = [], []
        for s in self.samples:
            if key in s["metrics"]:
                xs.append(s["n"])
                ys.append(s["metrics"][key])
        return xs, ys

    def latest(self) -> dict:
        return dict(self.samples[-1]["metrics"]) if self.samples else {}
