"""Generate per-alpha minimum surface-Cp tables for the NACA 24XX family.

REAL XFOIL data (v6.99, local binary), for the Tier C cavitation constraint:
cavitation inception when -Cp_min exceeds the cavitation number sigma_cav.

For each family member this drives XFOIL through the SAME conditions as
scripts/gen_polar_family.py (Re = 1e6, M = 0, alpha -6..14 step 0.5, 2406
refined paneling + sharp TE, sweep split at 0 so each branch marches away
from the benign start) and issues ``CPMN`` after every converged ``ALFA``.
XFOIL prints

    Minimum Inviscid Cp = -1.3832   at x =  0.0171
    Minimum Viscous  Cp = -1.3374   at x =  0.0171

per point; the VISCOUS minimum is recorded (BL displacement effects
included — the physically relevant value for cavitation inception at the
polar's Re). Points whose viscous solution did not converge (XFOIL prints
"VISCAL:  Convergence failed") are dropped, mirroring how unconverged
alphas are silently absent from PACC polars.

Output: data/airfoils/naca24XX_re<TAG>.cpmin — two columns "alpha cp_min",
loaded by aerobo.polar as the companion of the .pol table of the same name.

THE REYNOLDS NUMBER IS AN ARGUMENT, and it did not used to be. ``RE`` was a
module constant at 1e6 and the output name was spelt "re1e6" in a literal, so
this script could only ever regenerate the node it had already produced —
while ``scripts/gen_polar_family_re.py`` had long since shipped .pol tables
at 1e5 / 3e5 / 3e6 as well. The consequence was structural rather than
cosmetic: ``polar.polar_at_re`` could answer for cd/cl/cm anywhere in the
bank and could NOT answer for Cp_min anywhere but 1e6, so the water families
— whose cavitation margin is ``g = sigma_cav + Cp_min`` — could not be routed
to their own flown Reynolds number at all. Passing the Re here is what closes
that, and the tag must match the .pol file's exactly (polar.RE_BANK_TAGS),
because the two are loaded as a pair by name.

Usage:  python scripts/gen_cpmin_family.py [xfoil_binary] [re_tag ...]
        re_tag is one of polar.RE_BANK_TAGS ("1e5", "3e5", "1e6", "3e6");
        with none given it regenerates the shipped 1e6 node, bit for bit.

XFOIL traps handled via the shared batch-driving helpers in aerobo.xfoil_run
(see gen_polar_family.py / PROGRESS.md): headless PLOP G, cwd = data/airfoils
(long absolute paths get truncated), NACA 2406 needs 280 panels + sharp TE.

"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # allow running without installed aerobo

from aerobo.xfoil_run import (  # noqa: E402  (path bootstrap above)
    HEADLESS_PREAMBLE,
    VISCAL_FAIL_RE as FAIL_RE,
    oper_visc_commands,
    run_xfoil_script,
    split_alpha_list,
    thin_te_refine_commands,
)

OUT_DIR = ROOT / "data" / "airfoils"

#: the Reynolds nodes this script can be asked for, tag -> value. THE SAME
#: TABLE ``polar.RE_BANK_TAGS`` carries, imported rather than restated: a
#: .cpmin whose tag does not match a .pol on disk is a companion of nothing.
from aerobo.polar import RE_BANK_TAGS          # noqa: E402
RE_BY_TAG = {tag: re for re, tag in RE_BANK_TAGS}

DEFAULT_TAG = "1e6"
ALPHA_LO, ALPHA_HI, ALPHA_STEP = -6.0, 14.0, 0.5
CODES = ["2406", "2409", "2412", "2415", "2418"]
REFINE = {"2406"}  # thin section: finer panels + sharp TE (gen_polar_family)

CPMN_RE = re.compile(r"Minimum Viscous\s+Cp\s*=\s*(-?\d+\.\d+)")


class _SweepDied(RuntimeError):
    """XFOIL stopped mid-sweep, so the CPMN outputs cannot be paired to the
    alphas that asked for them. Recoverable by running the points one at a
    time; see :func:`_run_alphas`."""
# FAIL_RE (aerobo.xfoil_run.VISCAL_FAIL_RE): ONLY the top-level viscous-solve
# failure counts. "MRCHDU: Convergence failed" lines are transient BL-march
# sub-iteration noise that can appear on points whose VISCAL solve then
# converges fine (verified by hand).


def _alpha_sweep() -> list[float]:
    """0 .. 14 up, then -0.5 .. -6 down (each branch marches from alpha=0)."""
    return split_alpha_list(ALPHA_LO, ALPHA_HI, ALPHA_STEP)


def xfoil_script(code: str, alphas: list[float], init_before: set[int],
                 re: float) -> str:
    refine = thin_te_refine_commands() if code in REFINE else []
    lines = [
        *HEADLESS_PREAMBLE,
        f"NACA {code}",
        *refine,
        *oper_visc_commands(re),
    ]
    for i, a in enumerate(alphas):
        if i in init_before:
            lines.append("INIT")           # restart the BL from scratch
        lines += [f"ALFA {a:.2f}", "CPMN"]
    lines += ["", "QUIT", ""]
    return "\n".join(lines)


def _run_alphas(code: str, xfoil: str, alphas: list[float],
                init_before: set[int],
                re: float) -> tuple[dict[float, float], list[float]]:
    """Run one XFOIL process over ``alphas``; return (converged, dropped)."""
    proc = run_xfoil_script(
        xfoil_script(code, alphas, init_before, re),
        cwd=OUT_DIR,
        xfoil_bin=xfoil,
        timeout_s=600,
    )
    # Segment stdout by ALFA blocks: each CPMN emits exactly one
    # "Minimum Viscous Cp" line, in command order. A block containing
    # "Convergence failed" is dropped.
    blocks = proc.stdout.split("Minimum Viscous")
    # blocks[i] = text BEFORE the i-th CPMN's viscous line's value; the value
    # itself starts blocks[i+1]. Re-scan pairwise:
    cpmns = CPMN_RE.findall(proc.stdout)
    if len(cpmns) != len(alphas):
        # XFOIL DIED PART WAY THROUGH THE SWEEP. The block-pairing below is
        # positional — block i is the text before CPMN i — so a short list
        # cannot be attributed to alphas and nothing here can be salvaged.
        #
        # It is a SWEEP failure and not a POINT failure, which is why it is
        # its own exception rather than the RuntimeError it used to raise.
        # The retry path in :func:`run_member` already isolates points in
        # their own processes and already survives a member that crashes; it
        # simply never ran, because this raise killed the whole generation
        # first. Measured: at Re 1e5, NACA 2409 emits 38 CPMN lines for 41
        # alphas and took the entire family's Re-1e5 node down with it.
        raise _SweepDied(
            f"NACA {code} at Re {re:.0e}: expected {len(alphas)} CPMN "
            f"outputs, got {len(cpmns)}")
    out: dict[float, float] = {}
    dropped: list[float] = []
    for i, (a, cp) in enumerate(zip(alphas, cpmns)):
        # convergence failure for alpha i appears between CPMN i-1 and CPMN i
        if FAIL_RE.search(blocks[i]):
            dropped.append(a)
            continue
        out[a] = float(cp)
    for junk in OUT_DIR.glob("*.bl"):
        junk.unlink()
    return out, dropped


def run_member(code: str, xfoil: str, re: float) -> dict[float, float]:
    alphas = _alpha_sweep()
    n_up = sum(1 for a in alphas if a >= 0.0)
    try:
        out, dropped = _run_alphas(code, xfoil, alphas, init_before={n_up},
                                   re=re)
    except _SweepDied as exc:
        # ...then every point is a retry. Slower (one process per alpha) and
        # it is what makes the low-Re nodes reachable at all: at Re 1e5 the
        # thin members march into separation part way up the sweep and take
        # the process with them.
        print(f"  {exc} -- falling back to one process per alpha")
        out, dropped = {}, list(alphas)
    # in-sweep BL warm-start failures often converge from a cold start
    # (verified by hand: NACA 2412 alpha 2.5 fails mid-sweep, converges cold
    # with CL matching the .pol row) -- retry each alpha in its OWN xfoil
    # process (fresh process = cold BL; NO explicit INIT, which with no
    # viscous solution yet triggers an MRCHDU NaN cascade). Per-alpha
    # isolation also contains the occasional hard crash where CPMN emits
    # nothing at all.
    for a in dropped:
        try:
            got, still = _run_alphas(code, xfoil, [a], init_before=set(),
                                     re=re)
        except RuntimeError:
            still = [a]
            got = {}
        out.update(got)
        if still:
            print(f"  NACA {code} alpha {a:+.1f}: unconverged (cold retry), dropped")
    return out


def main() -> None:
    xfoil = sys.argv[1] if len(sys.argv) > 1 else "xfoil"
    tags = sys.argv[2:] or [DEFAULT_TAG]
    bad = [t for t in tags if t not in RE_BY_TAG]
    if bad:
        raise SystemExit(
            f"unknown Reynolds tag(s) {bad}; this script writes the COMPANION "
            f"of a .pol table on disk, so the tag has to be one the bank "
            f"already carries: {sorted(RE_BY_TAG)} (polar.RE_BANK_TAGS).")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for tag in tags:
        re_num = RE_BY_TAG[tag]
        for code in CODES:
            pol = OUT_DIR / f"naca{code}_re{tag}.pol"
            if not pol.exists():
                raise SystemExit(
                    f"{pol.name} is not on disk: a .cpmin is loaded as that "
                    f"file's companion, so generating one without it would "
                    f"write a table nothing can read. Run "
                    f"scripts/gen_polar_family_re.py first.")
            print(f"[gen_cpmin_family] NACA {code} at Re {tag}")
            table = run_member(code, xfoil, re_num)
            if len(table) < 20:
                raise RuntimeError(
                    f"NACA {code} at Re {tag}: only {len(table)} converged "
                    f"points")
            path = OUT_DIR / f"naca{code}_re{tag}.cpmin"
            rows = sorted(table.items())
            path.write_text(
                f"# NACA {code} minimum viscous surface Cp per alpha\n"
                f"# XFOIL 6.99, Re = {re_num:.0e}, M = 0, Ncrit = 9 "
                f"(companion of naca{code}_re{tag}.pol)\n"
                "# alpha_deg   cp_min\n"
                + "\n".join(f"{a:8.2f}   {cp:9.4f}" for a, cp in rows)
                + "\n"
            )
            cps = np.array([cp for _, cp in rows])
            print(
                f"  {len(rows)} pts, cp_min range "
                f"[{cps.min():.3f}, {cps.max():.3f}] -> {path.name}"
            )


if __name__ == "__main__":
    main()
