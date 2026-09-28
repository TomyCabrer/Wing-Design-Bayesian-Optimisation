# XFOIL for Intel Macs

`xfoil` here is XFOIL 6.99 (Mark Drela and Harold Youngren, GPL — source at
<https://web.mit.edu/drela/Public/web/xfoil/>), built for x86_64 from the
same source as the Apple-silicon copy in `../macos-arm64`. It runs on macOS
11 (Big Sur) and later.

It is one self-contained file: the gfortran runtime is linked in statically
(`-static-libgfortran -static-libquadmath -static-libgcc`) and it depends on
nothing but the system library. It has no X11 either: Drela's X11 window code
(`plotlib/Xwin2.c`) was replaced by empty functions, because the app switches
plotting off (`PLOP G`) before anything is drawn.

**How it was built:** cross-compiled on an Apple-silicon Mac with conda-forge's
`gfortran_osx-64` 16.2.0, double precision (`-fdefault-real-8`), the stock
`Xfoil/bin/Makefile` optimisation (`-O`), `MACOSX_DEPLOYMENT_TARGET=11.0`,
ad-hoc signed.

**What was checked, and what was not:** it could not be run on the build
machine. The arm64 build made with the same compiler, flags and window stub
was run against the binary behind the published results on 15 polars (NACA
0012, 2412, 4415, 6409, 0021 at Re 2e5, 1e6, 4e6, −4° to 16°): on the 283
points both converge, cl agrees to a median 0.009 % (95th percentile 0.09 %),
cd and cm to a median of zero. A few points near stall converge to a different
solution, and which angles converge differs by a row or two on 9 of the 15
polars. So results from this copy are close to the published ones but not
identical. On NACA 0012 at Re 2e5 it needs longer than the app's 20 s per
polar, so that polar comes back with 15 of its 19 rows.

**Downloaded as a zip:** macOS marks every file quarantined, and a quarantined
unsigned binary does not fail — it waits on a Gatekeeper prompt that never
appears. The solver clears the flag on this folder before first use.
