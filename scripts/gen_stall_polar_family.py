"""Generate the WIDE-ALPHA NACA 24XX XFOIL polar family (REAL polars).

Drives a local XFOIL binary (v6.99) in batch mode to produce
data/airfoils/naca24XX_re1e6_stall.pol for t/c = 0.06, 0.09, 0.12, 0.15, 0.18
at Re = 1e6, M = 0, alpha -6..22 deg step 0.5.

WHY a second family. The cruise family (scripts/gen_polar_family.py,
data/airfoils/naca24XX_re1e6.pol) stops at alpha = +14 deg, which is below
the viscous cl peak of every NACA 24XX section at Re = 1e6: the tables are
right-CENSORED, so cl_max is unobservable and no wing CL_max can be built
from them. Everything else -- Re, Mach, Ncrit, the paneling recipe, the
split-at-zero sweep -- is IDENTICAL to the cruise family, so the two are
commensurable: on the overlapping alpha range the two files hold the same
XFOIL solve of the same geometry.

The cruise family and its NACA 2412 member are the frozen regression anchors
of the 3-D objective and are NEVER regenerated or touched here. These outputs
are NEW files with the "_stall" suffix, which the frozen glob in
aerobo.polar.default_polar_family ("naca24??_re1e6.pol", exact-suffix match)
does not select; aerobo.polar.stall_polar_family loads them instead. Unlike
the cruise family this one DOES include 2412: aerobo.polar.default_polar()
uses naca2412_re1e6.pol as the Tier A default section, so omitting it would
leave exactly that section censored.

XFOIL gotchas handled here (see gen_polar_family.py / PROGRESS.md), via the
shared batch-driving helpers in aerobo.xfoil_run:
  - "PLOP\nG\n\n" first: disables the X11 plot window (headless run).
  - v6.99 PACC .pol format has extra Top_Itr/Bot_Itr columns; the parser in
    aerobo.polar is tolerant of them.
  - Unconverged alphas are simply not written to the polar by XFOIL. Past
    stall this is the NORM, not a fault: the separated-BL Newton solve stops
    converging somewhere above the cl peak and the table just ends there. A
    member is only useful if it ends AFTER its peak (see the ``censored``
    flag in aerobo.polar.stall_points).
  - The sweep is split at alpha = 0 (up 0..22, INIT, down -0.5..-6) so each
    branch marches away from the benign starting point; a single -6..22 sweep
    stalls on the thin NACA 2406 (BL solution never converges at -6). The
    parser sorts rows by alpha, so the on-file ordering is irrelevant.
  - NACA 2406 additionally needs refined paneling (280 nodes) and a sharp
    trailing edge (GDES/TGAP 0, blend 0.2c): with the default 160-panel blunt
    TE geometry the side-2 BL march diverges near the TE at every alpha. Even
    so the thin section is expected to stay censored -- its BL diverges well
    before the peak (t/c = 0.06 usable-floor note in PROGRESS.md).
  - XFOIL holds filenames in a fixed-length Fortran string and silently
    truncates long absolute paths (the polar then never appears). The process
    is therefore run with cwd = data/airfoils and a RELATIVE polar filename.
  - XFOIL leaves ":00.bl" BL-state droppings in cwd; they are removed.

Usage:  python scripts/gen_stall_polar_family.py [xfoil_binary]
        python scripts/gen_stall_polar_family.py --re-grid [xfoil_binary]
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # allow running without installed aerobo

from aerobo.xfoil_run import (  # noqa: E402  (path bootstrap above)
    HEADLESS_PREAMBLE,
    oper_visc_commands,
    run_xfoil_script,
    split_sweep_commands,
    thin_te_refine_commands,
)

OUT_DIR = ROOT / "data" / "airfoils"

RE = 1e6
ALPHA_LO, ALPHA_HI, ALPHA_STEP = -6.0, 22.0, 0.5
# Every member, 2412 INCLUDED (the cruise family excludes it as the frozen
# regression anchor; here it is a NEW file and the Tier A default section
# would otherwise stay censored).
CODES = ["2406", "2409", "2412", "2415", "2418"]
REFINE = {"2406"}  # thin section: needs finer panels + sharp TE (see docstring)

#: Session 42, item 3a. The wide-alpha family had ONE Reynolds number, exactly
#: as the cruise family did before `gen_polar_family_re.py` — so `stall.py`
#: reads every strip's cl_max ceiling off a Re-1e6 table however fast that
#: strip is actually flying. `--re-grid` extends the family along Re under the
#: IDENTICAL recipe (same alpha sweep, same Ncrit, same paneling rules), into
#:
#:     data/airfoils/naca{code}_re{tag}_stall.pol
#:
#: The shipped `_re1e6_stall` members are never regenerated and never touched:
#: `polar.stall_polar_family`'s glob matches the exact `_re1e6_stall.pol`
#: suffix, so nothing this grid writes can be selected by it, and the default
#: (no-argument) run of this script writes exactly the files it always did.
RE_GRID: tuple[tuple[float, str], ...] = (
    (1.0e5, "1e5"),
    (3.0e5, "3e5"),
    (3.0e6, "3e6"),
)

#: below this Reynolds number every member gets the refined-panel / sharp-TE
#: treatment, not just the thin one — the same threshold and the same reason as
#: `gen_polar_family_re.REFINE_BELOW`, and it matters more here because the
#: sweep runs 8 deg further past the peak.
REFINE_BELOW = 5.0e5


def polar_path(code: str, tag: str) -> Path:
    return OUT_DIR / f"naca{code}_re{tag}_stall.pol"


def xfoil_script(code: str, pol_name: str, re: float = RE) -> str:
    refine = (thin_te_refine_commands()
              if (code in REFINE or re < REFINE_BELOW) else [])
    return "\n".join(
        [
            *HEADLESS_PREAMBLE,
            f"NACA {code}",
            *refine,
            *oper_visc_commands(re),
            "PACC",
            pol_name,  # RELATIVE name; xfoil truncates long absolute paths
            "",  # no dump file
            *split_sweep_commands(ALPHA_LO, ALPHA_HI, ALPHA_STEP),
            "PACC",
            "",
            "QUIT",
            "",
        ]
    )


def generate_grid(xfoil: str) -> None:
    """The `--re-grid` arm: the same wide-alpha family at the other Reynolds
    numbers of the bank. Members whose march fails are REPORTED AND SKIPPED,
    never faked — `polar.stall_re_bank` treats a missing (code, Re) cell as
    absent, which is what makes ragged coverage safe."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    failed: list[tuple[str, str]] = []
    for re, tag in RE_GRID:
        for code in CODES:
            pol = polar_path(code, tag)
            if pol.exists():
                print(f"[gen_stall_polar_family] NACA {code} Re{tag} "
                      f"-> exists, skipped")
                continue
            print(f"[gen_stall_polar_family] NACA {code} Re{tag} -> {pol.name}")
            proc = run_xfoil_script(xfoil_script(code, pol.name, re),
                                    cwd=OUT_DIR, xfoil_bin=xfoil,
                                    timeout_s=900)
            for junk in OUT_DIR.glob("*.bl"):
                junk.unlink()
            if not pol.exists():
                print(f"    FAILED (no polar written); last XFOIL output:\n"
                      f"{proc.stdout[-600:]}")
                failed.append((code, tag))
    if failed:
        print(f"\n{len(failed)} cells failed and were skipped: {failed}")
    else:
        print("\nall cells generated")


def main() -> None:
    args = [a for a in sys.argv[1:]]
    if "--re-grid" in args:
        args.remove("--re-grid")
        generate_grid(args[0] if args else "xfoil")
        return
    xfoil = args[0] if args else "xfoil"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for code in CODES:
        pol = OUT_DIR / f"naca{code}_re1e6_stall.pol"
        if pol.exists():
            pol.unlink()  # XFOIL appends to existing polar files
        print(f"[gen_stall_polar_family] NACA {code} -> {pol}")
        proc = run_xfoil_script(
            xfoil_script(code, pol.name), cwd=OUT_DIR, xfoil_bin=xfoil, timeout_s=600
        )
        if not pol.exists():
            print(proc.stdout[-2000:])
            print(proc.stderr[-2000:])
            raise RuntimeError(f"XFOIL produced no stall polar for NACA {code}")
        for junk in OUT_DIR.glob("*.bl"):  # XFOIL BL-state droppings (":00.bl")
            junk.unlink()

    # report stall coverage using the repo loader/extractor
    from aerobo.polar import stall_points, stall_polar_family, usable_tc_floor

    fam = stall_polar_family(OUT_DIR)
    pts = stall_points(fam)
    print(f"{'t/c':>6} {'cl_max':>8} {'a_stall':>8} {'n_pts':>6} {'a_hi':>6}  censored")
    for tc, sp in sorted(pts.items()):
        print(
            f"{tc:6.2f} {sp.cl_max:8.4f} {sp.alpha_stall_deg:8.2f} "
            f"{sp.n_points:6d} {sp.alpha_max_deg:6.1f}  "
            f"{'YES' if sp.censored else 'no'}"
        )
    print(f"usable t/c floor (first uncensored member): {usable_tc_floor(fam)}")


if __name__ == "__main__":
    main()
