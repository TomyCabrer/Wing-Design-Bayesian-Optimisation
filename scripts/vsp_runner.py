#!/usr/bin/env python3
"""The OpenVSP half of the bridge — runs in the VSP interpreter, alone.

    <vsp-python> scripts/vsp_runner.py job.json out.json

Deliberately the only file in this repo that imports ``openvsp``, and it
imports NOTHING of aerobo: the OpenVSP Python packages pin their own numpy and
ship their own binaries, so they live in their own environment
(``AEROBO_VSP_PYTHON``, default ``~/opt/vsp-venv/bin/python``) and talk to the
project through two JSON files. That is the whole isolation story — no shared
site-packages, no import of one from the other.

The job it accepts:

    {"build": "<stem>_vsp.py",     # optional: run the generated build script
     "vsp3":  "<stem>.vsp3",       # the model to read (written by the above)
     "geometry": true,             # read back span / area / placement
     "compgeom": true,             # wetted area + volume of the built model
     "setup": {…same fields as "aero"…},   # WRITE them into the model and
     #                                        save it, so opening the .vsp3
     #                                        in the GUI shows the analysis
     #                                        already configured
     "aero": {"sref":…, "bref":…, "cref":…, "x_cg":…,
              "alpha_deg":…, "mach":…, "recref":…,
              "wake_iter":…, "symmetry": true}}

and the answer is JSON of the same shape: what VSP measured, plus whatever
VSPAERO converged to. Failures are reported as ``{"error": …}`` rather than a
traceback, because the caller is a comparison table and a missing number is a
result too.
"""

import json
import os
import subprocess
import sys


def _geometry(vsp) -> list:
    """Span, area and placement of every geom, straight off its own parms."""
    out = []
    for gid in vsp.FindGeoms():
        rec = {"name": vsp.GetGeomName(gid),
               "type": vsp.GetGeomTypeName(gid)}
        for parm, group in (("TotalSpan", "WingGeom"),
                            ("TotalProjectedSpan", "WingGeom"),
                            ("TotalArea", "WingGeom"),
                            ("TotalAR", "WingGeom"),
                            ("TotalChord", "WingGeom"),
                            ("X_Rel_Location", "XForm"),
                            ("Z_Rel_Location", "XForm")):
            pid = vsp.GetParm(gid, parm, group)
            if pid:
                rec[parm] = float(vsp.GetParmVal(pid))
        subs = []
        for sid in vsp.GetSubSurfIDVec(gid):
            sub = {"id": sid, "name": vsp.GetSubSurfName(gid, sid)}
            # by parm ID, not by (name, group): a sub-surface's container is
            # not the geom's, so GetParm(sid, ...) does not resolve after a
            # file reload even though the parms are there
            wanted = ("Length_C_Start", "Length_C_End", "Length_Start",
                      "UStart", "UEnd", "Surf_Type")
            for pid in vsp.GetSubSurfParmIDs(sid):
                nm = vsp.GetParmName(pid)
                if nm in wanted:
                    sub[nm] = float(vsp.GetParmVal(pid))
            subs.append(sub)
        if subs:
            rec["subsurfaces"] = subs
        # the section chain, as VSP ended up holding it
        surf = vsp.GetXSecSurf(gid, 0)
        n = vsp.GetNumXSec(surf)
        secs = []
        for i in range(1, n):
            grp = "XSec_%d" % i
            sec = {}
            for parm in ("Span", "Root_Chord", "Tip_Chord", "Twist",
                         "Dihedral", "Sweep"):
                pid = vsp.GetParm(gid, parm, grp)
                if pid:
                    sec[parm] = float(vsp.GetParmVal(pid))
            secs.append(sec)
        if secs:
            rec["sections"] = secs
        out.append(rec)
    return out


def _compgeom(vsp) -> dict:
    """Wetted area and volume of the model VSP actually built."""
    vsp.SetAnalysisInputDefaults("CompGeom")
    rid = vsp.ExecAnalysis("CompGeom")
    out = {}
    for key in ("Wet_Area", "Wet_Vol", "Total_Area", "Total_Vol", "Comp_Name"):
        try:
            vals = vsp.GetDoubleResults(rid, key)
        except Exception:
            vals = []
        if vals:
            out[key] = [float(v) for v in vals]
    try:
        out["Comp_Name"] = list(vsp.GetStringResults(rid, "Comp_Name"))
    except Exception:
        pass
    return out


#: VSPAERO settings that are PARMS of the model rather than inputs of one
#: analysis run: written into the VSPAEROSettings container, they are saved
#: with the .vsp3 and are what the GUI's Analysis > VSPAERO panel opens on.
#: (SetDoubleAnalysisInput below configures ONE execution; it does not
#: persist, so a file built that way opens with VSPAERO's own defaults and
#: the user has to retype the reference quantities the design was scored on.)
_SETUP_DOUBLE = (("Sref", "sref"), ("bref", "bref"), ("cref", "cref"),
                 ("Xcg", "x_cg"), ("ReCref", "recref"), ("Machref", "mach"),
                 ("MachStart", "mach"), ("MachEnd", "mach"),
                 ("Vinf", "v_inf"), ("Rho", "rho"))


def _setup(vsp, cfg: dict) -> dict:
    """Write the design's own reference quantities and attitude into the
    model's VSPAERO settings, and report what landed."""
    cid = vsp.FindContainer("VSPAEROSettings", 0)
    if not cid:
        return {"error": "no VSPAEROSettings container in this model"}
    out = {}

    def _set(name, value, integer=False):
        pid = vsp.FindParm(cid, name, "VSPAERO")
        if not pid:
            return
        vsp.SetParmVal(pid, float(value))
        out[name] = (int(vsp.GetParmVal(pid)) if integer
                     else float(vsp.GetParmVal(pid)))

    _set("RefFlag", 0, integer=True)        # 0 = the manual references below
    for parm, key in _SETUP_DOUBLE:
        if cfg.get(key) is not None:
            _set(parm, cfg[key])
    a = float(cfg.get("alpha_deg", 0.0))
    span = float(cfg.get("alpha_span_deg", 1.0))
    npts = int(cfg.get("alpha_npts", 3))
    _set("AlphaStart", a - span)
    _set("AlphaEnd", a + span)
    _set("AlphaNpts", npts, integer=True)
    _set("MachNpts", 1, integer=True)
    _set("WakeNumIter", int(cfg.get("wake_iter", 5)), integer=True)
    # the model is FULL span (the wing geom mirrors itself); VSPAERO's
    # symmetry flag would add an image of it and fly the aeroplane twice
    _set("Symmetry", 1 if cfg.get("symmetry", False) else 0, integer=True)
    _set("NCPU", int(cfg.get("ncpu", 4)), integer=True)
    vsp.Update()
    return out


def _aero(vsp, cfg: dict) -> dict:
    """One VSPAERO vortex-lattice point at the design attitude."""
    here = os.path.dirname(os.path.abspath(vsp.__file__))
    if not vsp.CheckForVSPAERO(vsp.GetVSPAEROPath()):
        vsp.SetVSPAEROPath(here)
    if not vsp.CheckForVSPAERO(vsp.GetVSPAEROPath()):
        return {"error": "vspaero binary not found beside the API (%s)" % here}

    # defaults only: overriding this step's own Symmetry made it write no
    # VSPGEOM at all, and the solver then reported "could not load ... file"
    # rather than a wrong answer. Half-model symmetry belongs on the SWEEP.
    vsp.SetAnalysisInputDefaults("VSPAEROComputeGeometry")
    vsp.ExecAnalysis("VSPAEROComputeGeometry")

    name = "VSPAEROSweep"
    vsp.SetAnalysisInputDefaults(name)
    vsp.SetIntAnalysisInput(name, "RefFlag", [0])       # 0 = manual reference
    for key, val in (("Sref", cfg["sref"]), ("bref", cfg["bref"]),
                     ("cref", cfg["cref"])):
        vsp.SetDoubleAnalysisInput(name, key, [float(val)])
    a = float(cfg["alpha_deg"])
    span = float(cfg.get("alpha_span_deg", 1.0))       # a small sweep AROUND
    npts = int(cfg.get("alpha_npts", 3))               # the design attitude
    for key, val in (("AlphaStart", a - span), ("AlphaEnd", a + span),
                     ("MachStart", float(cfg.get("mach", 0.0))),
                     ("MachEnd", float(cfg.get("mach", 0.0))),
                     ("ReCref", float(cfg.get("recref", 1e6))),
                     ("Xcg", float(cfg.get("x_cg", 0.0)))):
        vsp.SetDoubleAnalysisInput(name, key, [val])
    # Symmetry OFF: the model is FULL span (the wing geom mirrors itself), and
    # VSPAERO's symmetry flag adds an image of it. Measured on a plain tapered
    # wing: with the flag on, CL 0.76 / e 0.50; off, CL 0.46 / e 0.84 — the
    # flag was flying the aeroplane twice.
    for key, val in (("AlphaNpts", npts), ("MachNpts", 1),
                     ("WakeNumIter", int(cfg.get("wake_iter", 5))),
                     ("Symmetry", 1 if cfg.get("symmetry", False) else 0),
                     ("NCPU", int(cfg.get("ncpu", 4)))):
        vsp.SetIntAnalysisInput(name, key, [int(val)])
    vsp.Update()
    rid = vsp.ExecAnalysis(name)

    # the sweep hands back a WRAPPER: the numbers are in the results it
    # points at (VSPAERO_Polar / VSPAERO_History), under VSPAERO's own names.
    # Whole ARRAYS leave here — the caller wants the polar, not one point, so
    # it can compare at matched LIFT rather than only at matched attitude.
    wanted = {"Alpha": "alpha_deg", "CLtot": "CL", "CDi": "CDi",
              "CDtot": "CDtot", "CDo": "CDo", "CMytot": "CMy", "E": "e"}
    out: dict = {"reference": {"Sref": float(cfg["sref"]),
                               "bref": float(cfg["bref"]),
                               "cref": float(cfg["cref"])}}
    ids = [rid]
    if "ResultsVec" in vsp.GetAllDataNames(rid):
        ids += [r for r in vsp.GetStringResults(rid, "ResultsVec") if r]
    for res in ids:
        try:
            names = set(vsp.GetAllDataNames(res))
        except Exception:
            continue
        if vsp.GetResultsName(res) != "VSPAERO_Polar":
            continue
        for key, label in wanted.items():
            if key not in names or label in out:
                continue
            try:
                vals = vsp.GetDoubleResults(res, key)
            except Exception:
                vals = []
            if vals:
                out[label] = [float(v) for v in vals]
    return out


def main(argv) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    job = json.loads(open(argv[1]).read())
    out: dict = {}
    try:
        import openvsp as vsp
    except ImportError as exc:
        json.dump({"error": "openvsp not importable: %s" % exc},
                  open(argv[2], "w"))
        return 1
    out["vsp_version"] = vsp.GetVSPVersion()

    build = job.get("build")
    if build:
        # the generated build script is a standalone program: run it as one,
        # in this very interpreter, so what is measured below is what a user
        # running it by hand would get
        proc = subprocess.run([sys.executable, build], capture_output=True,
                              text=True)
        out["build_stdout"] = proc.stdout.strip().split("\n")
        if proc.returncode != 0:
            out["error"] = "build script failed: %s" % proc.stderr.strip()
            json.dump(out, open(argv[2], "w"), indent=1)
            return 1

    vsp.ClearVSPModel()
    vsp.ReadVSPFile(job["vsp3"])

    if job.get("setup"):
        try:
            out["setup"] = _setup(vsp, job["setup"])
            vsp.WriteVSPFile(job["vsp3"], vsp.SET_ALL)
            out["setup_written"] = job["vsp3"]
        except Exception as exc:      # a convenience, never fatal
            out["setup"] = {"error": "%s: %s" % (type(exc).__name__, exc)}
    if job.get("geometry", True):
        out["geometry"] = _geometry(vsp)
    if job.get("compgeom"):
        try:
            out["compgeom"] = _compgeom(vsp)
        except Exception as exc:      # a measurement, never fatal
            out["compgeom"] = {"error": "%s: %s" % (type(exc).__name__, exc)}
    if job.get("aero"):
        try:
            # from a CLEAN model: CompGeom above leaves its MeshGeom in the
            # model, and VSPAERO then finds no degenerate geometry to fly and
            # reports "could not load ... VSPGEOM file" instead of an answer
            vsp.ClearVSPModel()
            vsp.ReadVSPFile(job["vsp3"])
            out["aero"] = _aero(vsp, job["aero"])
        except Exception as exc:
            out["aero"] = {"error": "%s: %s" % (type(exc).__name__, exc)}

    json.dump(out, open(argv[2], "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
