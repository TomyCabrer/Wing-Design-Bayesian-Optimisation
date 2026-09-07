"""AeroBO — the launcher a downloaded copy of the app is started from.

``gui/v4/app.py`` is the DEVELOPER's entry point: it
assumes a ``.venv`` that somebody already built and a repository somebody
already knows their way around. This one is the entry point for a person who
downloaded a zip, and so it has to answer, before opening a window, the four
questions that a downloaded copy can actually get wrong:

* **Is this interpreter the one the launcher script built?** Running
  ``python launch.py`` with a system Python that has no NiceGUI in it is the
  single most likely way to arrive here broken, so a missing import is
  reported as one sentence naming the launcher, not as a traceback.
* **Is there a Bayesian-optimisation stack?** On an Intel Mac, and on any
  Mac older than Sonoma, there is no PyTorch wheel to install (see
  ``installer/requirements-core.in``). That is not fatal — ``api.py`` imports
  ``optimize.bo`` inside the function that runs it — so the shell opens and
  every other optimiser works. It is worth SAYING, though, because the
  alternative is a user meeting it as a failed search an hour later.
* **Is there an XFOIL?** Designing a section drives the real binary — stage 2,
  and the 1,781 of 8,317 families whose names say ``(XFOIL)`` or
  ``(coupled)``, which solve the section inside the wing search. The other
  6,536 read the pre-computed polar tables in ``data/airfoils`` and do not
  care. So the launcher names what is lost rather than letting it surface as
  a raise, minutes in.
* **Is the port free?** A second copy started while the first is still up
  must not die on ``[Errno 48] Address already in use``; it takes the next
  free port and says so.

Run::

    python launch.py                 # native window if it can, browser if not
    python launch.py --browser       # always a browser tab
    python launch.py --port 9000
    python launch.py --v3            # the four-stage V3 shell, no flight
    python launch.py --check         # preflight only, print the report, exit
"""

from __future__ import annotations

import os
import platform
import shutil
import socket
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent
DEFAULT_PORT = 8769

for _p in (str(APP_ROOT), str(APP_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# --------------------------------------------------------------------------
# preflight
# --------------------------------------------------------------------------

#: The imports without which there is no shell at all. Every one of these is
#: in installer/requirements-core.txt, which the launcher scripts install
#: FIRST and treat as must-succeed.
REQUIRED = ("nicegui", "plotly", "numpy", "scipy")


def _missing_required() -> list[str]:
    import importlib.util

    return [m for m in REQUIRED if importlib.util.find_spec(m) is None]


def find_xfoil() -> str | None:
    """Where this machine's XFOIL is, or None.

    Same order the solver itself uses (``aerobo.xfoil_run``): an explicit
    ``AEROBO_XFOIL_BIN`` wins, then ``PATH``. The extra place checked here is
    ``bin/xfoil`` inside the downloaded copy, so a user who cannot install
    system-wide can drop the binary next to the app and be found.
    """
    env = os.environ.get("AEROBO_XFOIL_BIN")
    if env and Path(env).exists():
        return env
    local = APP_ROOT / "bin" / ("xfoil.exe" if os.name == "nt" else "xfoil")
    if local.exists():
        os.environ["AEROBO_XFOIL_BIN"] = str(local)   # so the solver sees it
        return str(local)
    return shutil.which("xfoil")


def _xfoil_hint() -> str:
    if sys.platform == "darwin":
        return "brew install xfoil"
    if sys.platform.startswith("linux"):
        return "sudo apt install xfoil   (or build from web.mit.edu/drela/Public/web/xfoil/)"
    return ("download xfoil.exe from web.mit.edu/drela/Public/web/xfoil/ and put it in "
            + str(APP_ROOT / "bin"))


def _have(mod: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):             # a half-installed package
        return False


def preflight(verbose: bool = True) -> dict:
    """Report what this machine can and cannot do, without importing the app.

    Deliberately uses ``find_spec`` rather than importing: importing torch
    costs seconds and we only want to know whether it is THERE.
    """
    report = {
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.machine()}",
        "missing_required": _missing_required(),
        "bo": _have("torch") and _have("botorch"),
        "native": _have("webview"),
        "xfoil": find_xfoil(),
    }
    if not verbose:
        return report

    print(f"AeroBO  ·  Python {report['python']}  ·  {report['platform']}")
    if report["missing_required"]:
        print("  ✗ missing: " + ", ".join(report["missing_required"]))
    else:
        print("  ✓ shell        NiceGUI + Plotly + NumPy + SciPy")
    if report["bo"]:
        print("  ✓ optimiser    Bayesian optimisation (BoTorch) available")
    else:
        print("  ! optimiser    no PyTorch on this machine — Bayesian optimisation is off.")
        print("                 GA, SLSQP, DOE and penalty searches all still run.")
        if sys.platform == "darwin":
            print("                 (PyTorch ships no macOS wheel for Intel, nor for macOS 13 or older.)")
    if report["xfoil"]:
        print(f"  ✓ sections     XFOIL at {report['xfoil']}")
    else:
        print("  ! sections     no XFOIL found — designing a SECTION cannot run: that is")
        print("                 stage 2, and any family whose name says (XFOIL) or (coupled).")
        print("                 Families that read the polar tables in data/airfoils are")
        print("                 unaffected, and so are the results, controls and flight stages.")
        print(f"                 To fix: {_xfoil_hint()}")
    return report


# --------------------------------------------------------------------------
# port
# --------------------------------------------------------------------------

def free_port(start: int, tries: int = 20) -> int:
    """``start`` if nothing holds it, else the next free port above it.

    The probe binds ``0.0.0.0`` WITHOUT ``SO_REUSEADDR``, because that is what
    the server itself does and nothing weaker answers the question. Bound to
    ``127.0.0.1`` with ``SO_REUSEADDR`` set, the probe succeeds while another
    process is already listening on ``0.0.0.0`` at the same port — BSD lets a
    specific address and the wildcard coexist under that flag — so a second
    copy of the app reported the busy port as free, printed a URL for it, and
    died on ``[Errno 48]`` a second later. Measured, not reasoned about.
    """
    for port in range(start, start + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", port))
                return port
            except OSError:
                continue
    return start


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------

def _arg(flag: str, default):
    if flag in sys.argv:
        i = sys.argv.index(flag)
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return default


def main() -> int:
    report = preflight()
    if report["missing_required"]:
        print("\nThis Python has no AeroBO environment in it. Start the app with the "
              "launcher instead:")
        print("  macOS    double-click AeroBO.command")
        print("  Windows  double-click AeroBO.bat")
        print("  Linux    ./AeroBO.sh")
        return 1
    if "--check" in sys.argv:
        return 0

    # Native needs pywebview, and pywebview on Linux needs GTK/WebKit system
    # packages we cannot install for the user — so Linux is browser-only
    # unless it asks for a window and has one.
    want_native = "--browser" not in sys.argv
    native = want_native and report["native"] and not sys.platform.startswith("linux")
    if want_native and not native:
        print("  · opening in your browser (no native window backend here)")

    port = free_port(int(_arg("--port", DEFAULT_PORT)))
    if port != int(_arg("--port", DEFAULT_PORT)):
        print(f"  · port {_arg('--port', DEFAULT_PORT)} is busy — using {port}")
    print(f"\n  AeroBO is at http://127.0.0.1:{port}   (Ctrl-C here to stop it)\n")

    if "--v3" in sys.argv:
        from gui.v3.app import run
    else:
        from gui.v4.app import run
    run(native=native, port=port)
    return 0


if __name__ in {"__main__", "__mp_main__"}:
    # NOT `raise SystemExit(main())`. NiceGUI serves a page whose function was
    # not registered by re-executing sys.argv[0] with runpy — that is this
    # file — so main() runs a second time INSIDE the request handler, where
    # ui.run() returns at once because a server is already up. A SystemExit
    # raised there is an exception in the middle of answering a request, and
    # the browser gets 500 instead of the shell. Measured on `--v3`: every
    # request failed that way while the identical V4 path was fine, because
    # V4's root is a registered page and V3's is not.
    _code = main()
    if _code:
        raise SystemExit(_code)
