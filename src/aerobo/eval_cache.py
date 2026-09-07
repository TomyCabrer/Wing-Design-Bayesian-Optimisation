"""On-disk memo of objective evaluations — what makes a CONTINUATION cheap.

WHAT PROBLEM THIS SOLVES. "Give this run more evaluations"
(:func:`api.continue_run_config`) is one longer run of the SAME search: same
config, same seed, a bigger budget, so the evaluations already paid for come
back bit-for-bit and the new ones follow them. That is the arm this repo
measured and won with — handing a previous run's best to BO as ``x_seed`` is
the arm it lost with (``RESULTS_HANDOFF.md``). The price of it is that the
prefix is genuinely RE-FLOWN: a continuation of a 40-evaluation run pays for
40 evaluations of physics before it reaches evaluation 41, and a user watching
the progress bar go back to 1 is watching exactly that. Measured on
``trim wing`` (bo, 10 -> 14): the prefix was bit-identical and cost 0.7 s of
solver time it had already bought once.

This module hands that prefix back off disk instead. The objective is a pure
function of the design vector — ``results/diverge.log`` looked for a single
evaluation where the same design returned a different ``f`` and found none —
so memoising it changes NOTHING about what the search flies. The optimiser
still calls the objective once per evaluation, the progress counter still
counts it, the history and the per-evaluation log are byte-identical; only the
seconds change.

THE KEY IS THE PHYSICS, NOT THE RUN. A cached value must never outlive the
code that produced it, so every key carries a fingerprint of the source tree
(:func:`source_fingerprint`) alongside the design vector and the fields of the
configuration that can change what ``f(x)`` means (:func:`problem_fingerprint`).
Edit ``src/aerobo``, and every key moves: the old entries are unreachable
rather than wrong. What the fingerprint does NOT cover is the coordinate
library under ``data/airfoils`` (10 MB of read-only ``.dat`` files that a run
reaches through its design vector, and that nothing in this repo edits) — if
you do edit one, call :func:`clear`.

NOT ON BY DEFAULT. ``api.run`` takes the cache as an argument and defaults to
None, because a wall clock that reads 0.7 s instead of 1.9 s is not a faster
search — it is a search that was not re-flown, and this repo has a rule about
reading a wall clock as a search outcome. Every study script therefore keeps
the un-memoised path exactly. The V3 shell turns it on, and the run it
produces says so: ``RunResult.eval_cache`` carries ``{hits, misses, dir}``,
so any wall time read next to it can be read for what it is.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np

#: repo root — this file is ``<root>/src/aerobo/eval_cache.py``
REPO_ROOT = Path(__file__).resolve().parents[2]

#: where memoised evaluations live. Beside the XFOIL polar cache and under the
#: same gitignored ``results/`` tree, for the same reason: it is a derived
#: artefact of this machine, never a record.
DEFAULT_CACHE_DIR = REPO_ROOT / "results" / "eval_cache"

#: set ``AEROBO_EVAL_CACHE=0`` to force the memo off for a whole process, even
#: where a caller asked for it — the switch a timing study wants.
ENV_SWITCH = "AEROBO_EVAL_CACHE"

#: configuration fields that can change what ``f(x)`` IS. ``budget``, ``seed``
#: and ``x_seed`` are deliberately absent: they change which points a search
#: visits, never the value at a point — which is the whole reason a longer run
#: can re-use a shorter one's evaluations.
PROBLEM_KEYS = ("problem_name", "mission_kwargs", "flags", "bounds_overrides",
                "pinned")


def resolve_dir(spec) -> Path | None:
    """``None`` (the memo is off) or the directory it lives in.

    ``None``/``False`` off, ``True`` the default directory, anything else a
    path. ``AEROBO_EVAL_CACHE=0`` in the environment overrides all of them,
    so one export switches a whole study back to un-memoised physics.
    """
    if str(os.environ.get(ENV_SWITCH, "")).strip() in ("0", "off", "false"):
        return None
    if spec is None or spec is False:
        return None
    if spec is True:
        return Path(DEFAULT_CACHE_DIR)
    return Path(spec)


def _hash_files(paths) -> str:
    """sha256 over (relative path, file bytes) for a sorted list of files."""
    h = hashlib.sha256()
    for p in sorted(paths):
        try:
            data = Path(p).read_bytes()
        except OSError:                     # a file that vanished under us
            continue                        # cannot be part of the fingerprint
        h.update(str(Path(p).relative_to(REPO_ROOT)).encode())
        h.update(hashlib.sha256(data).digest())
    return h.hexdigest()


@lru_cache(maxsize=1)
def source_fingerprint() -> str:
    """What the physics IS, right now — hashed once per process.

    ``src/aerobo/**/*.py`` plus the published tables at the top of ``data/``
    (boxes, references, the budget law). About 12 MB and ~50 ms, paid once,
    and it is what makes a stale entry impossible rather than unlikely: an
    edited solver moves every key in the cache.
    """
    files = list((REPO_ROOT / "src" / "aerobo").rglob("*.py"))
    files += list((REPO_ROOT / "data").glob("*.json"))
    return _hash_files(files)[:32]


def problem_fingerprint(config: dict | None) -> str:
    """The part of a run configuration that changes ``f(x)`` — as one key."""
    cfgd = dict(config or {})
    payload = {k: cfgd.get(k) for k in PROBLEM_KEYS}
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:32]


def clear(cache_dir=None) -> int:
    """Delete every memoised evaluation. Returns how many files went.

    For the one case the fingerprint cannot see: a hand-edited coordinate file
    under ``data/airfoils``.
    """
    root = Path(cache_dir if cache_dir is not None else DEFAULT_CACHE_DIR)
    n = 0
    for p in root.rglob("*.json"):
        try:
            p.unlink()
            n += 1
        except OSError:
            pass
    return n


class EvalMemo:
    """``fn(x)`` with its answers kept on disk, keyed by (physics, config, x).

    Wraps the callable the optimiser evaluates — the SEARCHED one, so ``x`` is
    the free dimensions only and the pin is part of the configuration key.
    Returns exactly what the inner callable returns: a float for an
    unconstrained problem, ``(f, g)`` for a constrained one, with ``g`` handed
    back as an array so a cached evaluation and a flown one are the same object
    shape.

    Every failure path is a MISS, never an error: a corrupt entry, an
    unwritable directory or a full disk costs the run its speed-up and nothing
    else.
    """

    def __init__(self, fn, is_constrained: bool, cache_dir,
                 config: dict | None = None):
        self.fn = fn
        self.is_constrained = bool(is_constrained)
        self.dir = Path(cache_dir)
        self.problem = problem_fingerprint(config)
        self.source = source_fingerprint()
        self.hits = 0
        self.misses = 0
        self.stores = 0

    # ------------------------------------------------------------ keys
    def key(self, x) -> str:
        """sha256 of (source fingerprint, problem fingerprint, x as raw bits).

        The design vector is hashed at FULL precision, never rounded: two
        designs a rounding would merge are two evaluations the optimiser asked
        for separately, and answering the second with the first's value would
        change the search. (The XFOIL polar cache rounds at 1e-6 because it
        keys on GEOMETRY, where a sub-panel perturbation is not a different
        aerofoil; this keys on a search coordinate, where it is.)
        """
        xb = np.ascontiguousarray(
            np.asarray(x, dtype=np.float64).ravel()).tobytes()
        h = hashlib.sha256()
        h.update(self.source.encode())
        h.update(b"|")
        h.update(self.problem.encode())
        h.update(b"|")
        h.update(xb)
        return h.hexdigest()

    def path_for(self, key: str) -> Path:
        """One file per evaluation, two hex characters of shard.

        Flat would work and is what ``xfoil_run`` does; a wing session makes
        tens of thousands of these, which is where a flat directory starts
        costing more to list than the entries save.
        """
        return self.dir / key[:2] / f"{key}.json"

    # ------------------------------------------------------------ store
    def _load(self, path: Path):
        try:
            rec = json.loads(path.read_text())
        except (OSError, ValueError):
            return None
        if "f" not in rec:
            return None
        f = float(rec["f"])
        if not self.is_constrained:
            return f
        g = rec.get("g")
        if g is None:
            return None            # a constrained key written unconstrained
        return f, np.asarray(g, dtype=float)

    def _store(self, path: Path, x, r) -> None:
        if self.is_constrained:
            f, g = r
            rec = {"f": float(f),
                   "g": [float(v) for v in np.atleast_1d(
                       np.asarray(g, dtype=float))]}
        else:
            rec = {"f": float(r), "g": None}
        rec["x"] = [float(v) for v in
                    np.asarray(x, dtype=np.float64).ravel()]
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            # atomic, because two shells share this tree: a reader must see a
            # whole entry or no entry, never half of one
            fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
            with os.fdopen(fd, "w") as fh:
                json.dump(rec, fh)
            os.replace(tmp, path)
            self.stores += 1
        except (OSError, ValueError, TypeError):
            pass

    # ------------------------------------------------------------ call
    def __call__(self, x):
        key = self.key(x)
        path = self.path_for(key)
        if path.exists():
            hit = self._load(path)
            if hit is not None:
                self.hits += 1
                return hit
        r = self.fn(x)
        self.misses += 1
        self._store(path, x, r)
        return r

    def stats(self) -> dict:
        """``{hits, misses, dir}`` — what the record carries, so a wall time
        read beside it can be read for what it is."""
        return {"hits": int(self.hits), "misses": int(self.misses),
                "dir": str(self.dir)}
