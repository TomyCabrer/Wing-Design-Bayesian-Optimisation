# XFOIL for Apple-silicon Macs

`xfoil` here is XFOIL 6.99 (Mark Drela and Harold Youngren, GPL — source at
<https://web.mit.edu/drela/Public/web/xfoil/>) as it was built for this
project in May 2026: gfortran 14.2 (Homebrew `gcc 14.2.0_1`), double
precision, the stock `Xfoil/bin/Makefile` flags. It is the binary every
published result in this repository was produced with.

There is nothing to install instead: Homebrew has no `xfoil` formula and
conda-forge has no build. So the binary is shipped, with the six libraries it
links copied into `xfoil-lib/`:

| library | from | licence |
|---|---|---|
| `libgfortran.5`, `libquadmath.0`, `libgcc_s.1.1` | GCC 14.2 runtime | GPL with the GCC Runtime Library Exception |
| `libX11.6`, `libxcb.1`, `libXau.6` | XQuartz (arm64 slice only) | MIT/X11 |

Each load path was rewritten to `@loader_path/…` with `install_name_tool`,
the build machine's Homebrew rpaths deleted, and every file re-signed ad hoc
(`codesign -f -s -`). Nothing loads from `/opt/homebrew` or `/opt/X11`
(`DYLD_PRINT_LIBRARIES=1` lists only this folder and the system). X11 is
linked but never opened: the app switches plotting off (`PLOP G`) before
anything else, so XQuartz does not need to be installed or running.

**Checked against the original:** 15 polars (NACA 0012, 2412, 4415, 6409 and
0021, each at Re 2e5, 1e6 and 4e6, −4° to 16°, with Cp_min) came out
bit-identical, 291 of 291 converged rows, zero difference in every column.

**Why not rebuild it:** the same source, compiler and Makefile, built again
in September 2026, is NOT bit-identical — and on NACA 0012 at Re 2e5 it hangs
in the boundary-layer march where this one converges in about 3 s. So the
solver (`aerobo.xfoil_run`) prefers this copy over any `xfoil` on `PATH`;
set `AEROBO_XFOIL_BIN` to use another one.

**Downloaded as a zip:** macOS marks every file quarantined, and a quarantined
unsigned binary does not fail — it waits on a Gatekeeper prompt that never
appears. The solver clears the flag on this folder before first use.
