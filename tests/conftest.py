"""Make the app importable from a plain checkout.

The shell (``launch.py``, ``gui/v4/app.py``) puts the repository root and
``src/`` on ``sys.path`` itself, so a downloaded copy runs without being
installed. The tests get the same treatment here, so ``pytest`` works from a
fresh clone with nothing more than the runtime environment.

``PYTHONPATH`` is set as well as ``sys.path``: several tests start a second
interpreter (``subprocess.run([sys.executable, ...])``) to check what a module
imports, and that child sees only the environment, not this process's path.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_PATHS = [str(ROOT), str(ROOT / "src")]

for _p in _PATHS:
    if _p not in sys.path:
        sys.path.insert(0, _p)

_existing = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
os.environ["PYTHONPATH"] = os.pathsep.join(
    _PATHS + [p for p in _existing if p not in _PATHS])
