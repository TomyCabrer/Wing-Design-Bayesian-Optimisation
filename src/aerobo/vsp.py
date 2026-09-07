"""Take a flown design INTO OpenVSP — build the model, and open the app on it.

``cad.py`` writes the files; this module drives the two things a user means
by "open it in OpenVSP":

* **build** the ``.vsp3`` — run the generated API script (``cad.vsp_script``)
  in OpenVSP's own interpreter, so what lands is a NATIVE wing chain that
  VSPAERO can fly, not an imported mesh;
* **load the physics with it** — write the design's own reference quantities
  and attitude (Sref, bref, cref, x_cg, Re, Mach, the alpha sweep around the
  trim point) into the model's ``VSPAEROSettings`` container and save the
  file, so the GUI's VSPAERO panel opens on the numbers the run was scored
  at instead of on VSPAERO's defaults;
* **open** the OpenVSP application on that file.

The isolation story is ``scripts/vsp_crosscheck.py``'s, unchanged and for
the same reason: the OpenVSP Python packages pin their own numpy and ship
their own binaries, so they live in their own environment and talk to this
one through two JSON files (``scripts/vsp_runner.py``). Nothing here imports
``openvsp``.

Everything is discovered rather than assumed, and every discovery is
overridable by environment variable, because "OpenVSP is not installed" and
"OpenVSP is installed somewhere else" have to be different messages:

===========================  ==================================  ============
what                         environment variable                default
===========================  ==================================  ============
the OpenVSP interpreter      ``AEROBO_VSP_PYTHON``               ``~/opt/vsp-venv/bin/python``
the OpenVSP application      ``AEROBO_VSP_APP``                  the ``vsp`` binary in any ``~/opt/OpenVSP-*`` (or on PATH, or in /Applications)
===========================  ==================================  ============
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

#: the OpenVSP interpreter, when nobody says otherwise
DEFAULT_VSP_PYTHON = Path.home() / "opt" / "vsp-venv" / "bin" / "python"

#: where an OpenVSP install is looked for, in order. Globs, so a version
#: bump does not need a code change.
APP_CANDIDATES = (
    "~/opt/OpenVSP-*/vsp",
    "~/opt/OpenVSP*/OpenVSP.app/Contents/MacOS/vsp",
    "/Applications/OpenVSP*/vsp",
    "/Applications/OpenVSP*.app/Contents/MacOS/vsp",
    "/usr/local/bin/vsp",
    "/opt/homebrew/bin/vsp",
)


def runner_path() -> Path:
    """``scripts/vsp_runner.py`` — the OpenVSP half of the bridge.

    Deliberately NOT importable from here: it runs in the other interpreter.
    Located relative to the installed package (``src/aerobo/`` ->
    ``<repo>/scripts/``), which is where an editable install puts it.
    """
    return Path(__file__).resolve().parents[2] / "scripts" / "vsp_runner.py"


def vsp_python() -> Path | None:
    """The interpreter that can ``import openvsp``, or ``None``."""
    p = Path(os.environ.get("AEROBO_VSP_PYTHON") or DEFAULT_VSP_PYTHON)
    return p if p.exists() else None


def vsp_app() -> Path | None:
    """The OpenVSP APPLICATION binary, or ``None`` if none was found."""
    env = os.environ.get("AEROBO_VSP_APP")
    if env:
        p = Path(env).expanduser()
        return p if p.exists() else None
    for pattern in APP_CANDIDATES:
        pat = os.path.expanduser(pattern)
        root = Path(pat).anchor or "/"
        rel = pat[len(root):] if pat.startswith(root) else pat
        try:
            hits = sorted(Path(root).glob(rel))
        except (OSError, ValueError, IndexError):
            continue
        for hit in hits:
            if hit.is_file() and os.access(hit, os.X_OK):
                return hit
    found = shutil.which("vsp")
    return Path(found) if found else None


def availability() -> dict:
    """What is installed, as three answers a shell can put on screen.

    ``{"python": path|None, "app": path|None, "runner": path|None}`` — never
    an exception, because "is OpenVSP here?" is a question a view asks while
    drawing itself.
    """
    runner = runner_path()
    return {"python": vsp_python(), "app": vsp_app(),
            "runner": runner if runner.exists() else None}


def aero_setup(geom: dict, breakdown: dict | None = None,
               alpha_span_deg: float = 1.5, alpha_npts: int = 3,
               wake_iter: int = 5) -> dict:
    """The PHYSICS block: this design's own references and attitude.

    Read off the same design report the geometry came from — never restated
    — so a model opened in OpenVSP is set up for the aeroplane the run
    scored, at the attitude it was trimmed to. The one convention worth
    stating: ``symmetry`` is FALSE because the exported model is full span
    (``cad.vsp_script`` mirrors the wing), and VSPAERO's symmetry flag would
    add an image of it.
    """
    bd = breakdown or {}
    tail = geom.get("tail") or {}
    v = bd.get("V") or geom.get("V")
    out = {
        "sref": float(geom.get("S") or bd.get("S") or bd.get("S_ref") or 1.0),
        "bref": float(geom.get("b") or bd.get("b") or 1.0),
        "cref": float(bd.get("mac") or bd.get("mac_true")
                      or tail.get("mac") or 1.0),
        "x_cg": float(bd.get("x_cg") if bd.get("x_cg") is not None
                      else tail.get("x_cg") or 0.0),
        "alpha_deg": float(bd.get("alpha_deg") or 0.0),
        "mach": float(bd.get("Mach") or bd.get("mach") or 0.0),
        "recref": float(bd.get("Re_mac") or 1e6),
        "alpha_span_deg": float(alpha_span_deg),
        "alpha_npts": int(alpha_npts),
        "wake_iter": int(wake_iter),
        "symmetry": False,
    }
    if v is not None:
        out["v_inf"] = float(v)
    rho = bd.get("rho") or geom.get("rho")
    if rho is not None:
        out["rho"] = float(rho)
    return out


def run_job(job: dict, outdir) -> dict:
    """Run one job through ``scripts/vsp_runner.py`` and return its answer.

    Failures come back as ``{"error": …}`` rather than as exceptions: the
    caller is usually a view, and "OpenVSP is not installed" is a result.
    """
    d = Path(outdir)
    d.mkdir(parents=True, exist_ok=True)
    py, runner = vsp_python(), runner_path()
    if py is None:
        return {"error": f"no OpenVSP interpreter at "
                         f"{os.environ.get('AEROBO_VSP_PYTHON') or DEFAULT_VSP_PYTHON}"
                         f" — set AEROBO_VSP_PYTHON, or make one (see "
                         f"scripts/vsp_crosscheck.py's docstring)"}
    if not runner.exists():
        return {"error": f"the OpenVSP half of the bridge is missing "
                         f"({runner})"}
    # the two sidecars carry the JOB, not the design, and they are read
    # back by name in the same call — but they are written into the
    # user's chosen folder, so they take the design's name too rather
    # than colliding between two exports that share a folder
    tag = str(job.get("vsp3") or "vsp").rsplit(".", 1)[0] or "vsp"
    job_path, out_path = d / f"{tag}_vsp_job.json", d / f"{tag}_vsp_out.json"
    job_path.write_text(json.dumps(job, indent=1))
    try:
        proc = subprocess.run([str(py), str(runner), str(job_path),
                               str(out_path)], capture_output=True, text=True,
                              cwd=str(d), timeout=900)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    if not out_path.exists():
        detail = (proc.stderr or proc.stdout or "").strip().split("\n")[-1]
        return {"error": f"the OpenVSP side wrote nothing: {detail}"}
    try:
        return json.loads(out_path.read_text())
    except ValueError as exc:
        return {"error": f"unreadable answer from the OpenVSP side: {exc}"}


def build_model(geom: dict, outdir, stem: str = "aerobo", x_best=None,
                labels=None, section=None, section_aft=None,
                breakdown: dict | None = None, setup: bool = True) -> dict:
    """Export the design and build its ``.vsp3``. Returns a report dict.

    ``{"files": {kind: path}, "vsp3": path|None, "error": str|None,
    "setup": {...}, "vsp_version": str}``. The ``.vsp3`` is None when the
    OpenVSP side could not run — the exported files are still on disk and
    still openable by hand, which is the whole point of writing them first.
    """
    from . import cad

    d = Path(outdir)
    files = cad.export(geom, d, stem=stem, x_best=x_best, labels=labels,
                       section=section, section_aft=section_aft)
    out: dict = {"files": files, "vsp3": None, "error": None,
                 "dir": str(d.resolve())}
    # both sides of the bridge spell the .vsp3 the same way, off cad.py's
    # own table: this name is handed to ANOTHER interpreter as a bare
    # relative path and only lands here because the subprocess runs with
    # cwd=d, so a second spelling is a file written where nobody looks
    vsp3_name = cad.export_name(stem, "vsp3")
    job = {"build": Path(files["vsp"]).name, "vsp3": vsp3_name,
           "geometry": True}
    if setup:
        job["setup"] = aero_setup(geom, breakdown)
    res = run_job(job, d)
    out["vsp_version"] = res.get("vsp_version")
    if res.get("error"):
        out["error"] = res["error"]
        return out
    built = d / vsp3_name
    if not built.exists():
        out["error"] = "the build script ran but wrote no .vsp3"
        return out
    out["vsp3"] = str(built.resolve())
    out["setup"] = res.get("setup")
    out["geometry"] = res.get("geometry")
    return out


def open_in_app(path) -> dict:
    """Launch the OpenVSP application on ``path``. ``{"error": …}`` or
    ``{"app": …}``.

    Detached on purpose (``Popen``, no wait): the caller is a web shell
    serving a request, and OpenVSP is a GUI the user is about to work in.
    """
    app = vsp_app()
    if app is None:
        return {"error": "no OpenVSP application found — set AEROBO_VSP_APP "
                         "to its `vsp` binary"}
    try:
        subprocess.Popen([str(app), str(path)],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         cwd=str(Path(path).parent))
    except (OSError, subprocess.SubprocessError) as exc:
        return {"error": f"could not launch {app}: {exc}"}
    return {"app": str(app)}
