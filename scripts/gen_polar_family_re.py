"""Generate the NACA 24XX polar family at SEVERAL Reynolds numbers.

The repo's default section family is `data/airfoils/naca24??_re1e6.pol` — one
Reynolds number, 1e6, for every design the tool flies unless the user picked a
library section by name. That is a real modelling hole: a hydrofoil elevator
at 1.5e5 and a 12 m wing at 3e6 are looked up in the same table, and nothing
downstream reads `TablePolar.Re`, so no correction is applied anywhere either.

This script extends the family along Re, using exactly the recipe of
`gen_polar_family.py` (real XFOIL polars, never fabricated) so the new tables
are the same object as the existing ones at a different operating point:

    data/airfoils/naca{code}_re{tag}.pol       tag in RE_TAGS below

The existing `_re1e6` members are NOT regenerated and NOT touched — they are
the regression anchor for the whole 3-D objective, and the `naca24??_re1e6.pol`
glob that `polar.default_polar_family` uses deliberately does not match any
file this script writes.

XFOIL at low Re: the boundary-layer march is much harder to converge than at
1e6, so below `REFINE_BELOW` every member gets the thin-section treatment the
original script reserved for NACA 2406 (280 panels + sharp TE). Members that
still fail are reported and skipped rather than faked; the loader treats a
missing (code, Re) cell as absent, which is why partial coverage is safe.

Usage:  python scripts/gen_polar_family_re.py [xfoil_binary]
        python scripts/gen_polar_family_re.py --report        (no XFOIL; audit)
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

ALPHA_LO, ALPHA_HI, ALPHA_STEP = -6.0, 14.0, 0.5

#: the family's t/c members, spelled as NACA 24XX codes (as in gen_polar_family)
CODES = ["2406", "2409", "2412", "2415", "2418"]

#: (Re, filename tag). 1e6 is the SHIPPED table and is deliberately absent —
#: regenerating it would move the frozen anchor. The grid is log-spaced at
#: roughly half a decade, which is the resolution a log-Re interpolation needs
#: to stay inside a drag count over this range (see results/polar_re_bank.json).
RE_GRID: tuple[tuple[float, str], ...] = (
    (1.0e5, "1e5"),
    (3.0e5, "3e5"),
    (3.0e6, "3e6"),
    # ...and the TOP of the bank, added when the water families were routed
    # onto their own Reynolds number and immediately ran off the end of it:
    # with the span and area rows open a hydrofoil box reaches Re_mac 7.3e6
    # (a 0.24 m chord at 16 m/s in sea water), which 3e6 cannot answer for.
    # A design there was refused rather than extrapolated, which is correct
    # and is not the same as being able to price it.
    (1.0e7, "1e7"),
)

#: below this Reynolds number every member gets the refined-panel / sharp-TE
#: treatment, not just the thin ones: the laminar separation bubble that makes
#: the march fail moves forward and thickens as Re falls.
REFINE_BELOW = 5.0e5


def polar_path(code: str, tag: str) -> Path:
    return OUT_DIR / f"naca{code}_re{tag}.pol"


def xfoil_script(code: str, re: float, pol_name: str) -> str:
    refine = (thin_te_refine_commands()
              if (re < REFINE_BELOW or code == "2406") else [])
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


def report() -> int:
    """Audit what is on disk. No XFOIL, safe to run any time."""
    import numpy as np

    from aerobo.polar import load_xfoil_polar

    missing = 0
    print(f"{'code':>6} {'Re':>8} {'pts':>5} {'alpha range':>16} "
          f"{'a_lin':>7} {'aL0deg':>7} {'cd(2deg)':>9}")
    for code in CODES:
        for re, tag in ((1.0e6, "1e6"),) + RE_GRID:
            pol = polar_path(code, tag)
            if not pol.exists():
                print(f"{code:>6} {tag:>8}   -- MISSING --")
                missing += 1
                continue
            p = load_xfoil_polar(pol)
            print(f"{code:>6} {tag:>8} {p.alpha_deg.size:5d} "
                  f"[{p.alpha_deg[0]:+6.1f},{p.alpha_deg[-1]:+6.1f}] "
                  f"{p.a_lin:7.3f} {np.rad2deg(p.alpha_L0):+7.2f} "
                  f"{p.cd(2.0):9.5f}")
    if missing:
        print(f"\n{missing} cells missing — run without --report to generate")
    return missing


def main() -> None:
    args = [a for a in sys.argv[1:]]
    if "--report" in args:
        report()
        return
    xfoil = args[0] if args else "xfoil"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    failed: list[tuple[str, str]] = []
    for re, tag in RE_GRID:
        for code in CODES:
            pol = polar_path(code, tag)
            if pol.exists():
                print(f"[re-bank] NACA {code} @ Re {tag}: present, skipping")
                continue
            print(f"[re-bank] NACA {code} @ Re {re:.3g} -> {pol.name}")
            proc = run_xfoil_script(xfoil_script(code, re, pol.name),
                                    cwd=OUT_DIR, xfoil_bin=xfoil,
                                    timeout_s=600)
            for junk in OUT_DIR.glob("*.bl"):  # XFOIL BL-state droppings
                junk.unlink()
            if not pol.exists():
                print(proc.stdout[-800:])
                failed.append((code, tag))
                continue
            # a polar with too few rows is worse than no polar: it would be
            # loaded and silently interpolated over a hole
            from aerobo.polar import load_xfoil_polar
            try:
                p = load_xfoil_polar(pol)
            except Exception as exc:          # unparseable => treat as failed
                print(f"  unparseable ({exc}); removing")
                pol.unlink()
                failed.append((code, tag))
                continue
            if p.alpha_deg.size < 12:
                print(f"  only {p.alpha_deg.size} converged alphas; removing")
                pol.unlink()
                failed.append((code, tag))
    print()
    report()
    if failed:
        print("\nfailed cells (recorded, not faked): "
              + ", ".join(f"{c}@{t}" for c, t in failed))


if __name__ == "__main__":
    main()
