"""One-shot XFOIL polar evaluation for arbitrary airfoil coordinates.

Subprocess wrapper around a local XFOIL binary (Drela & Youngren, XFOIL 6.99;
Drela 1989, "XFOIL: An Analysis and Design System for Low Reynolds Number
Airfoils", Conf. on Low Reynolds Number Airfoil Aerodynamics, Notre Dame) for
the CST-parameterised 2-D section evaluations of the BO loop: given a closed
coordinate loop (TE-upper -> LE -> TE-lower, the XFOIL native order produced
by ``aerobo.airfoil.cst_coords``), a Reynolds number, Mach number and a set of
angles of attack, run a viscous PACC polar sweep and return the converged
(alpha, cl, cd, cm) rows.

XFOIL batch-driving gotchas (solved in scripts/gen_polar_family.py and
scripts/gen_cpmin_family.py, lifted here as the shared helpers those scripts
now import; see PROGRESS.md):

  - ``PLOP / G`` first: disables the X11 plot window (headless run).
  - XFOIL holds filenames in fixed-length Fortran strings and silently
    truncates long absolute paths (the polar then never appears). Every run
    therefore uses a UNIQUE temp working directory with RELATIVE file names.
  - Alpha sweeps are split at alpha = 0 (march up from 0, ``INIT``, march
    down from -step) so each Newton/BL continuation branch marches away from
    the benign zero-lift-ish starting point; a single lo..hi sweep stalls on
    thin sections.
  - Only "VISCAL:  Convergence failed" is a real per-point failure;
    "MRCHDU:" lines are transient BL-march sub-iteration noise.
  - Unconverged alphas are silently ABSENT from the PACC polar file; the
    caller sees only converged rows (possibly none).
  - v6.99 PACC tables carry extra Top_Itr/Bot_Itr columns; the private
    parser here (like aerobo.polar.load_xfoil_polar) keys on "any line with
    >= 5 leading numeric columns" and is layout-tolerant. A private parser is
    used instead of extending ``aerobo.polar.TablePolar`` so that module's
    behaviour for its existing callers is untouched.

Failure contract (frozen inter-agent API): ``run_xfoil_polar`` raises
``XfoilError`` ONLY for infrastructure faults (XFOIL binary missing /
unexecutable). If XFOIL ran but nothing converged -- garbage geometry,
timeout with no rows, all points unconverged -- it returns a result with
EMPTY arrays and the caller applies the -100.0 penalty contract. No exception
ever propagates to the optimiser from a bad *design*.

Caching: results are memoised as JSON under ``cache_dir`` keyed by
``cache_key`` = sha256 of the coordinates rounded to 1e-6 plus
(re, mach, alphas rounded to 1e-4, n_panel). Geometry perturbations below
1e-6 chord (far under XFOIL's own panel discretisation error) hit the same
entry. Timed-out runs are NOT cached (their truncation point is machine-load
dependent, i.e. non-deterministic); empty-but-completed runs ARE cached
(deterministic failures should not cost 20 s per BO revisit).
"""

from __future__ import annotations

import hashlib
import json
import math as _math
import os
import re as _re
import shutil
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

#: Homebrew's Apple-silicon prefix -- where the binary lives on the machine
#: every published run was produced on, and so the answer this resolves to
#: there. It is only the LAST resort: PATH is consulted first, so a checkout
#: on Linux (or an Intel Mac, /usr/local/bin) finds its own xfoil instead of
#: raising XfoilError for a path that was never going to exist. The cache key
#: does not include the binary, so which one answers cannot change a result.
_HOMEBREW_XFOIL = "/opt/homebrew/bin/xfoil"
DEFAULT_XFOIL_BIN = (os.environ.get("AEROBO_XFOIL_BIN")
                     or shutil.which("xfoil") or _HOMEBREW_XFOIL)
DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / "results" / "xfoil_cache"

# Relative working-file names (XFOIL truncates long absolute paths).
_COORDS_NAME = "foil.dat"
_POLAR_NAME = "xfoil.pol"

# ---------------------------------------------------------------------------
# Shared batch-driving vocabulary (imported by scripts/gen_polar_family.py and
# scripts/gen_cpmin_family.py -- keep the emitted strings STABLE, the family
# generators' output is byte-compared against the pre-refactor scripts).
# ---------------------------------------------------------------------------

#: Headless preamble: disable the X11 plot window before anything else.
HEADLESS_PREAMBLE: tuple[str, ...] = ("PLOP", "G", "")

#: Only the top-level viscous-solve failure counts as unconverged.
#: "MRCHDU:  Convergence failed" lines are BL-march sub-iteration noise that
#: can appear on points whose VISCAL solve then converges fine.
VISCAL_FAIL_RE = _re.compile(r"VISCAL:\s+Convergence failed")


def oper_visc_commands(re: float, mach: float = 0.0, n_iter: int = 200) -> list[str]:
    """OPER-menu preamble: viscous mode at ``re``, Mach, Newton iteration cap."""
    return ["OPER", f"VISC {re:.0f}", f"MACH {mach:g}", f"ITER {n_iter}"]


def thin_te_refine_commands(n_panel: int = 280, blend: float = 0.2) -> list[str]:
    """Refined paneling + sharp trailing edge (GDES/TGAP 0, blend ``blend``*c).

    Needed by thin sections (NACA 2406): with the default 160-panel blunt-TE
    geometry the side-2 BL march diverges near the TE at every alpha.
    """
    return [
        "PPAR",
        f"N {n_panel}",
        "",
        "",
        "GDES",
        "TGAP",
        "0",
        f"{blend}",
        "",
        "PANE",
    ]


def split_sweep_commands(alpha_lo: float, alpha_hi: float, alpha_step: float) -> list[str]:
    """ASEQ sweep split at alpha = 0: up 0..hi, INIT, down -step..lo.

    Each branch marches away from the benign alpha = 0 start; a single lo..hi
    ASEQ stalls on thin sections (BL never converges starting at alpha_lo).
    Requires alpha_lo < 0 < alpha_hi with 0 on the step grid.
    """
    return [
        f"ASEQ 0 {alpha_hi} {alpha_step}",
        "INIT",
        f"ASEQ -{alpha_step} {alpha_lo} -{alpha_step}",
    ]


def split_alpha_list(alpha_lo: float, alpha_hi: float, alpha_step: float) -> list[float]:
    """Per-ALFA visiting order for the split sweep: 0..hi up, then -step..lo down."""
    up = list(np.arange(0.0, alpha_hi + 1e-9, alpha_step))
    down = list(np.arange(-alpha_step, alpha_lo - 1e-9, -alpha_step))
    return up + down


def run_xfoil_script(
    script: str,
    *,
    cwd: str | Path,
    xfoil_bin: str = DEFAULT_XFOIL_BIN,
    timeout_s: float = 300.0,
) -> subprocess.CompletedProcess:
    """Feed a command ``script`` to one XFOIL process; return the completed run.

    Raises XfoilError if the binary is missing/unexecutable (infrastructure
    fault). subprocess.TimeoutExpired propagates to the caller, which decides
    whether a partial polar is salvageable (run_xfoil_polar does).
    """
    try:
        return subprocess.run(
            [xfoil_bin],
            input=script,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            cwd=str(cwd),
        )
    except FileNotFoundError as exc:
        raise XfoilError(f"XFOIL binary not found: {xfoil_bin}") from exc
    except PermissionError as exc:
        raise XfoilError(f"XFOIL binary not executable: {xfoil_bin}") from exc


# ---------------------------------------------------------------------------
# One-shot polar evaluation (frozen inter-agent API)
# ---------------------------------------------------------------------------


class XfoilError(RuntimeError):
    """Infrastructure fault (XFOIL binary missing/unexecutable) -- NOT a
    design failure. Design failures return an empty XfoilPolarResult."""


@dataclass
class XfoilPolarResult:
    """Converged rows of one XFOIL PACC polar sweep.

    Arrays hold CONVERGED points only, sorted by alpha; unconverged alphas
    are absent (XFOIL never writes them). All four arrays share one length,
    which may be ZERO -- the caller maps an empty result to the -100.0
    penalty. cm is the quarter-chord pitching-moment coefficient (PACC col 5).
    """

    alpha_deg: np.ndarray
    cl: np.ndarray
    cd: np.ndarray
    cm: np.ndarray
    n_requested: int
    from_cache: bool = False
    #: minimum viscous surface Cp per CONVERGED alpha, when the caller asked
    #: for it (``with_cpmin=True``). Empty otherwise. This is what a
    #: cavitation constraint needs — sigma_cav + Cp_min >= 0 — and XFOIL only
    #: reports it through the CPMN command, one alpha at a time, so asking
    #: for it forces the per-ALFA driving path (see _build_script).
    cp_min: np.ndarray = field(default_factory=lambda: np.empty(0))
    #: PACC pressure-drag column (CDp) per converged alpha. Kept because its
    #: SIGN is a convergence diagnostic: a viscous solution whose surface-
    #: pressure integration disagrees with its momentum deficit reports
    #: CDp < 0, i.e. a total drag below its own friction component. Empty for
    #: a result built without it (synthetic polars in tests, pre-2026-08-04
    #: cache entries), which ``drop_unphysical`` reads as "unknown".
    cdp: np.ndarray = field(default_factory=lambda: np.empty(0))
    #: PACC Top_Xtr / Bot_Xtr columns: where XFOIL says each surface went
    #: turbulent. Kept because they price the row's own friction floor --
    #: see :func:`mixed_bl_cd`. Empty when the result was built without them.
    xtr_top: np.ndarray = field(default_factory=lambda: np.empty(0))
    xtr_bot: np.ndarray = field(default_factory=lambda: np.empty(0))

    @property
    def has_cp_min(self) -> bool:
        return bool(self.cp_min.size) and self.cp_min.size == self.alpha_deg.size

    @property
    def has_cdp(self) -> bool:
        return bool(self.cdp.size) and self.cdp.size == self.alpha_deg.size

    @property
    def has_xtr(self) -> bool:
        n = self.alpha_deg.size
        return (bool(self.xtr_top.size) and self.xtr_top.size == n
                and self.xtr_bot.size == n)

    @property
    def n_converged(self) -> int:
        return int(self.alpha_deg.size)


# ------------------------------------------------- physically impossible rows

#: Blasius flat-plate skin friction coefficient, one side, fully laminar.
BLASIUS_C = 1.328

#: How negative CDp has to be, as a fraction of CD, before the row is read as
#: a broken solution rather than as roundoff in the surface-pressure
#: integration. XFOIL prints CDp to five decimals, so a value of -0.00002 on a
#: CD of 0.00533 is two units in the last place — not a claim that the drag is
#: below its own friction.
#:
#: WHERE THIS NUMBER CAME FROM, and what is now known about it.
#:
#: It was set on 2026-08-04 from "every negative-CDp row in the polar cache" —
#: which at that moment, minutes after the CDp column was first parsed, was 29
#: rows. Those 29 did separate into two clean populations with an empty
#: interval between 0.0441 and 0.1117, and 0.05 was placed in it.
#:
#: Re-measured 2026-08-05 over the cache the section studies actually rest on
#: (481k rows, 11190 of them with CDp < 0), that interval is not empty and the
#: separation is not clean:
#:
#:   |CDp|/CD        rows   below flat-plate floor   below 0.75x own friction
#:   <=0.0038         619            0                         1
#:   0.0038-0.0441   5435            0                        33
#:   0.0441-0.05      450            0                        11
#:   0.05-0.1117     3053            0                        67
#:   0.1117-0.256    1374           47                       237
#:   >0.256           259           37                       145
#:
#: So the ratio is a WEAK discriminator, not a bimodal one: rows that actually
#: contradict the physics concentrate above 0.11 and overwhelmingly above 0.26,
#: while the cut at 0.05 also removes ~3500 rows that violate no bound. What
#: keeps the cut earning its place is a different measurement — dropped rows
#: sit ~22x further from their own polar's local trend than kept ones (leave-
#: one-out median 8.9 % against 0.41 %), i.e. they are locally inconsistent
#: even when they are not globally impossible.
#:
#: This is therefore a calibrated cut on a continuum, and is documented as one.
#: It is legitimate because it refuses a SOLVER's row rather than a user's
#: design — the repo's rule about calibrations becoming bans is about the
#: latter. It is NOT retuned here: every section number in the frozen study was
#: measured through this exact value, so moving it is a re-measurement, not an
#: edit. The candidate replacement, if that re-run is ever paid for, is to make
#: the CDp test a conjunction with the friction bound and keep the standalone
#: form only for polars with no transition columns.
CDP_REL_TOL = 0.05

#: Process-wide declaration that this interpreter is one of several running
#: XFOIL at once, so ``timeout_s`` — a WALL clock — is being measured under
#: contention. ORed with :func:`run_xfoil_polar`'s ``concurrent`` argument.
#: A sweep's worker sets it once (budget_study._worker_init) rather than every
#: call site having to know it is inside a sweep; the alternative, which is
#: what happens today, is that a load-induced timeout is cached as a
#: fabricated "hopeless geometry" and served to every later run.
ASSUME_CONCURRENT = False


#: Flat-plate turbulent drag coefficient constant, ``0.074 / Re^0.2``.
PRANDTL_C = 0.074

#: How far below the friction implied by a row's OWN transition locations the
#: reported total drag is allowed to fall before the row is refused. Not 1.0,
#: because the mixed flat-plate estimate is an engineering bound and not a
#: theorem: an adverse pressure gradient can push local skin friction below
#: the flat-plate value as a surface approaches separation. (It buys that
#: with pressure drag, which the row's TOTAL includes, so the bound holds
#: comfortably in practice.) Measured separation is wide enough that the
#: exact value does not matter: the pathological rows land near 0.55 and
#: ordinary sections near 1.5.
FRICTION_MARGIN = 0.75


def mixed_bl_cd(xtr_top, xtr_bot, re: float):
    """Lower bound on friction drag from XFOIL's OWN transition locations.

    Each surface is treated as a flat plate that runs laminar to ``x_tr`` and
    turbulent thereafter, referenced to chord and with the wetted length taken
    as exactly the chord (a real aerofoil's is longer, so this is
    conservative):

        laminar   [0, x]  ->  1.328 * sqrt(x) / sqrt(Re)
        turbulent [x, 1]  ->  0.074 * (1 - x**0.8) / Re**0.2

    Summed over the two surfaces. This is the test that catches the second
    failure mode: a row whose CDp is perfectly positive, but whose total drag
    is less than the friction its own reported transition implies. XFOIL says
    the upper surface went turbulent at 26 % chord and simultaneously reports
    a drag 43 % below what that costs — two of its own outputs contradicting
    each other, which is exactly the class of thing an optimiser converts
    into an optimum.
    """
    re = float(re)
    if not np.isfinite(re) or re <= 0.0:
        return None
    x = np.clip(np.asarray([xtr_top, xtr_bot], dtype=float), 0.0, 1.0)
    lam = BLASIUS_C * np.sqrt(x) / _math.sqrt(re)
    turb = PRANDTL_C * (1.0 - x ** 0.8) / re ** 0.2
    return (lam + turb).sum(axis=0)


def laminar_cd_floor(re: float) -> float:
    """Strict lower bound on a section's drag coefficient at ``re``.

    A flat plate of ZERO thickness, wetted length exactly the chord on each
    side, laminar all the way to the trailing edge: ``2 * 1.328 / sqrt(Re)``.
    Every real aerofoil is worse on all three counts — its wetted length
    exceeds the chord, its boundary layer transitions somewhere, and it
    carries pressure drag the plate does not — so no converged viscous
    solution can sit below this and mean anything.

    Deliberately the LOOSEST defensible bound rather than a fitted one. This
    is not a calibration band that encodes what designs the study happened to
    see (those belong in the score reference, and turning one into a gate is
    a mistake this repo has made before): it is arithmetic, and a design is
    refused by it only if it broke physics.
    """
    re = float(re)
    if not np.isfinite(re) or re <= 0.0:
        return 0.0
    return 2.0 * BLASIUS_C / _math.sqrt(re)


def drop_unphysical(res: "XfoilPolarResult", re: float) -> "XfoilPolarResult":
    """Drop converged rows that contradict themselves.

    Two tests, both arithmetic — neither is a calibration band, and a design
    is refused by them only if its polar row is not a solution at all:

    * ``CDp < -CDP_REL_TOL * CD``. XFOIL reports total drag from the wake
      momentum deficit and pressure drag from integrating surface pressure. A
      substantially negative CDp means those two disagree in the one direction
      that is impossible: the total drag comes out BELOW its own friction
      component (``CD < CD-CDp``). This is the dominant mode in practice. The
      tolerance is there because the sign alone also fires on print roundoff
      (see :data:`CDP_REL_TOL`), and punching holes in an otherwise good polar
      around ``cl_design`` is its own failure.
    * ``cd < laminar_cd_floor(re)``. Backstop for a row whose CDp is missing
      or benign but whose total drag is under the flat-plate bound.
    * a non-finite ``cd``, which no comparison would otherwise reject.

    Why this matters more than its 0.05 % rate suggests: such a row is the
    GLOBAL OPTIMUM of every objective built on the polar — minimum cd
    outright, and L/D through ``cl/cd`` — so a search does not merely trip
    over it, it hunts it. Measured on the 2026-08-04 section sweeps, these
    rows supplied the best-ever design of the frozen ``sec8_cd`` study
    (cd 0.002637, below the flat-plate floor of 0.002656) and an entire
    second mode of the composite: nothing scored between 75 and 113, and six
    evaluations of 3233 scored 113-156, every one of them on a CDp < 0 row.

    Dropping the row rather than failing the polar is the existing contract —
    "unconverged alphas are absent" — this is one more alpha that did not
    converge, XFOIL's optimism notwithstanding. A polar that loses too many
    then fails ``MIN_CONVERGED`` in the normal way.

    Applied on the way OUT of :func:`run_xfoil_polar`, so the cache stays a
    faithful record of what the solver said and the filter is a property of
    the reader.
    """
    if res.alpha_deg.size == 0:
        return res
    n = int(res.alpha_deg.size)
    keep = np.isfinite(res.cd)
    if res.has_cdp:
        keep &= np.isfinite(res.cdp)
        keep &= ~(res.cdp < -CDP_REL_TOL * np.abs(res.cd))
    if res.has_xtr:
        bound = mixed_bl_cd(res.xtr_top, res.xtr_bot, re)
        if bound is not None:
            known = np.isfinite(bound)
            keep &= ~(known & (res.cd < FRICTION_MARGIN * bound))
    floor = laminar_cd_floor(re)
    if floor > 0.0:
        keep &= ~(res.cd < floor)
    if bool(keep.all()):
        return res

    def _side(arr, present: bool):
        """Filter a side array, or pass it through when it is not one.

        ``has_cdp`` / ``has_xtr`` / ``has_cp_min`` are LENGTH tests. A side
        array of the wrong length therefore reads as absent and was passed
        through unfiltered — after which it could match the SHORTENED alpha
        array by coincidence, flipping the property from False to True and
        pairing values with the wrong angles of attack. Anything not usable
        going in is emptied rather than smuggled through at a new length.
        """
        if not present:
            return np.array([], dtype=float) if (
                arr is not None and np.size(arr) not in (0, n)) else arr
        return arr[keep]

    return XfoilPolarResult(
        alpha_deg=res.alpha_deg[keep], cl=res.cl[keep], cd=res.cd[keep],
        cm=res.cm[keep], n_requested=res.n_requested,
        from_cache=res.from_cache,
        cp_min=_side(res.cp_min, res.has_cp_min),
        cdp=_side(res.cdp, res.has_cdp),
        xtr_top=_side(res.xtr_top, res.has_xtr),
        xtr_bot=_side(res.xtr_bot, res.has_xtr),
    )


#: XFOIL's CPMN output line ("Minimum Viscous Cp = -1.234")
CPMN_RE = _re.compile(r"Minimum Viscous\s+Cp\s*=\s*(-?\d+\.\d+)")


def cache_key(coords, re, mach, alphas, n_panel: int = 200,
              with_cpmin: bool = False) -> str:
    """sha256 key over coords rounded to 1e-6 + (re, mach, alphas@1e-4, n_panel).

    The 1e-6 coordinate rounding means geometry perturbations far below the
    panel discretisation error share a cache entry; 1e-3-level changes (real
    design moves) do not. ``+ 0.0`` after rounding normalises IEEE negative
    zeros (round(-1e-9, 6) = -0.0 which is == 0.0 but BYTE-different, and the
    key hashes bytes).
    """
    c = np.ascontiguousarray(np.asarray(coords, dtype=np.float64).round(6) + 0.0)
    meta = repr(
        (float(re), float(mach), tuple(round(float(a), 4) for a in alphas),
         int(n_panel))
        + ((" cpmin",) if with_cpmin else ())
    )
    return hashlib.sha256(c.tobytes() + meta.encode()).hexdigest()


def _alpha_commands(alphas: np.ndarray) -> list[str]:
    """OPER alpha-driving commands for the requested angles.

    Two cases (the two that can be made robust -- see module docstring):

    1. UNIFORM grid: sorted alphas form an arithmetic sequence. Driven with
       ASEQ marching away from alpha = 0: if the grid crosses zero AND
       contains 0, split there (up branch, INIT, down branch) exactly like
       scripts/gen_polar_family.py; a single-signed grid is one ASEQ starting
       at the endpoint nearest zero.
    2. Anything else: individual ALFA commands, non-negative alphas ascending
       first (marching up from the benign near-zero start), then one INIT,
       then negative alphas descending from zero -- the per-ALFA analogue of
       the split sweep (scripts/gen_cpmin_family.py order). INIT is only
       issued at the sign change, i.e. after converged solutions exist; a
       cold INIT with no viscous solution triggers an MRCHDU NaN cascade
       (PROGRESS.md).
    """
    a = np.sort(np.asarray(alphas, dtype=float))
    # -- case 1: uniform grid ------------------------------------------------
    if a.size >= 2:
        d = np.diff(a)
        h = float(d[0])
        if h > 1e-9 and np.allclose(d, h, rtol=0.0, atol=1e-6):
            has_zero = bool(np.any(np.abs(a) < 1e-9))
            if a[0] < -1e-9 and a[-1] > 1e-9 and has_zero:
                return split_sweep_commands(float(a[0]), float(a[-1]), h)
            if a[0] >= -1e-9:  # all non-negative: march up from the low end
                return [f"ASEQ {a[0]:.4f} {a[-1]:.4f} {h:.4f}"]
            if a[-1] <= 1e-9:  # all non-positive: march down from the high end
                return [f"ASEQ {a[-1]:.4f} {a[0]:.4f} {-h:.4f}"]
    # -- case 2: individual ALFA, split-at-zero visiting order ---------------
    up = [x for x in a if x >= 0.0]  # ascending (a is sorted)
    down = [x for x in a[::-1] if x < 0.0]  # descending towards alpha_lo
    cmds = [f"ALFA {x:.4f}" for x in up]
    if up and down:
        cmds.append("INIT")
    cmds += [f"ALFA {x:.4f}" for x in down]
    return cmds


def _cpmin_alpha_commands(alphas: np.ndarray) -> tuple[list[str], list[float]]:
    """Per-ALFA driving with a CPMN after each point, and the visiting order.

    CPMN reports one number per COMMAND, so the compact ASEQ marching the
    plain polar uses cannot be interleaved with it — asking for Cp_min
    therefore forces the individual-ALFA path. The visiting order is the same
    split-at-zero one _alpha_commands uses in its general case (march up from
    the benign near-zero start, INIT, then down), and it is returned so the
    parser can pair CPMN outputs with the alphas that produced them.
    """
    a = np.sort(np.asarray(alphas, dtype=float))
    up = [float(x) for x in a if x >= 0.0]
    down = [float(x) for x in a[::-1] if x < 0.0]
    cmds: list[str] = []
    for x in up:
        cmds += [f"ALFA {x:.4f}", "CPMN"]
    if up and down:
        cmds.append("INIT")
    for x in down:
        cmds += [f"ALFA {x:.4f}", "CPMN"]
    return cmds, up + down


def _build_script(re: float, mach: float, alphas: np.ndarray, n_panel: int,
                  with_cpmin: bool = False) -> str:
    """Full XFOIL stdin script: headless, LOAD, re-panel, viscous PACC sweep."""
    alpha_cmds = (_cpmin_alpha_commands(alphas)[0] if with_cpmin
                  else _alpha_commands(alphas))
    return "\n".join(
        [
            *HEADLESS_PREAMBLE,
            f"LOAD {_COORDS_NAME}",
            "PPAR",
            f"N {n_panel}",
            "",
            "",
            "PANE",
            *oper_visc_commands(re, mach),
            "PACC",
            _POLAR_NAME,  # relative name; XFOIL truncates long absolute paths
            "",  # no dump file
            *alpha_cmds,
            "PACC",
            "",
            "QUIT",
            "",
        ]
    )


def _parse_cpmin(stdout: str, order: list[float],
                 alpha_deg: np.ndarray) -> np.ndarray:
    """Pair CPMN outputs with the CONVERGED polar rows.

    XFOIL prints one "Minimum Viscous Cp" line per CPMN command, in command
    order, whether or not that point converged — so the values are matched to
    the alphas that were VISITED (``order``) and then selected for the alphas
    the PACC table actually kept. A count mismatch (a truncated run) returns
    an empty array rather than a guess: a cavitation margin read off the
    wrong angle would be worse than no margin at all.
    """
    vals = CPMN_RE.findall(stdout)
    if len(vals) != len(order) or alpha_deg.size == 0:
        return np.empty(0, dtype=float)
    by_alpha = {round(a, 4): float(v) for a, v in zip(order, vals)}
    out = []
    for a in alpha_deg:
        v = by_alpha.get(round(float(a), 4))
        if v is None:
            return np.empty(0, dtype=float)
        out.append(v)
    return np.asarray(out, dtype=float)


def _parse_pacc_table(text: str):
    """Parse a PACC polar table into (alpha, CL, CD, CM, CDp).

    Rows with >= 5 numeric columns are (alpha, CL, CD, CDp, CM, ...);
    tolerant of the extra v6.99 Top_Itr/Bot_Itr columns, header/banner lines
    are skipped because they fail the float parse or have < 5 columns (same
    tolerance rule as aerobo.polar.load_xfoil_polar, plus the CM column the
    2-D loop needs). Duplicate alphas (never produced by our sweeps, but
    possible from caller-supplied duplicates) keep the FIRST occurrence. Rows
    sorted by alpha.

    CDp (column 3) is returned as well as CM: its sign is what
    :func:`drop_unphysical` reads to spot a row whose total drag came out
    below its own friction component.
    """
    rows: dict[float, tuple] = {}
    for line in text.splitlines():
        parts = line.split()
        try:
            vals = [float(p) for p in parts]
        except ValueError:
            continue
        if len(vals) >= 5:
            key = round(vals[0], 4)
            if key not in rows:
                # Top_Xtr / Bot_Xtr are columns 5 and 6 and are absent from a
                # 5-column table; NaN then reads as "unknown" downstream.
                rows[key] = (vals[0], vals[1], vals[2], vals[4], vals[3],
                             vals[5] if len(vals) > 5 else float("nan"),
                             vals[6] if len(vals) > 6 else float("nan"))
    if not rows:
        empty = np.empty(0, dtype=float)
        return tuple(empty.copy() for _ in range(7))
    arr = np.array(sorted(rows.values()), dtype=float)
    return (arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4],
            arr[:, 5], arr[:, 6])


def _empty_result(n_requested: int, from_cache: bool = False) -> XfoilPolarResult:
    e = np.empty(0, dtype=float)
    return XfoilPolarResult(
        alpha_deg=e, cl=e.copy(), cd=e.copy(), cm=e.copy(),
        n_requested=n_requested, from_cache=from_cache,
    )


def _cache_load(path: Path, n_requested: int) -> XfoilPolarResult | None:
    try:
        blob = json.loads(path.read_text())
        return XfoilPolarResult(
            alpha_deg=np.asarray(blob["alpha_deg"], dtype=float),
            cl=np.asarray(blob["cl"], dtype=float),
            cd=np.asarray(blob["cd"], dtype=float),
            cm=np.asarray(blob["cm"], dtype=float),
            n_requested=n_requested,
            from_cache=True,
            cp_min=np.asarray(blob.get("cp_min", []), dtype=float),
            # KeyError on a pre-2026-08-04 entry, deliberately: those were
            # written before CDp was parsed, so their rows cannot be checked
            # for the self-contradiction drop_unphysical looks for. Treating
            # them as a MISS re-solves them on demand and overwrites the
            # entry — the cache heals itself as it is used, and nothing
            # serves an unverifiable polar in the meantime.
            cdp=np.asarray(blob["cdp"], dtype=float),
            xtr_top=np.asarray(blob["xtr_top"], dtype=float),
            xtr_bot=np.asarray(blob["xtr_bot"], dtype=float),
        )
    except (OSError, ValueError, KeyError, TypeError):
        return None  # corrupt/unreadable entry: fall through to a fresh run


def _cache_store(path: Path, result: XfoilPolarResult, meta: dict) -> None:
    """Atomic-ish JSON write (temp file + os.replace) so parallel BO workers
    computing the same key never see a torn file. Cache failures are
    swallowed: a broken cache must not fail a valid evaluation.

    The temp name carries a UUID as well as the PID: PID alone is unique
    across PROCESSES but not across THREADS, and the screening path runs a
    ThreadPoolExecutor (airfoil_select.screen_database), where two threads
    landing on the same cache key would otherwise interleave writes into one
    temp file and os.replace a torn document into the cache."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".tmp.{os.getpid()}.{uuid.uuid4().hex}")
        tmp.write_text(
            json.dumps(
                {
                    "alpha_deg": result.alpha_deg.tolist(),
                    "cl": result.cl.tolist(),
                    "cd": result.cd.tolist(),
                    "cm": result.cm.tolist(),
                    "cp_min": result.cp_min.tolist(),
                    "cdp": result.cdp.tolist(),
                    "xtr_top": result.xtr_top.tolist(),
                    "xtr_bot": result.xtr_bot.tolist(),
                    "meta": meta,
                }
            )
        )
        os.replace(tmp, path)
    except OSError:
        pass


def run_xfoil_polar(
    coords: np.ndarray,
    re: float,
    mach: float,
    alphas,
    xfoil_bin: str = DEFAULT_XFOIL_BIN,
    timeout_s: float = 20.0,
    cache_dir: str | Path | None = None,
    n_panel: int = 200,
    concurrent: bool = False,
    with_cpmin: bool = False,
) -> XfoilPolarResult:
    """Viscous XFOIL polar for a coordinate loop at (re, mach) over ``alphas``.

    coords: (n, 2) closed loop in XFOIL order (TE-upper -> LE -> TE-lower,
    e.g. from aerobo.airfoil.cst_coords). Written to a labelled coordinate
    file in a UNIQUE temp directory (concurrency-safe: parallel calls never
    share working files) and re-paneled to ``n_panel`` nodes with PANE.

    Returns converged rows only; EMPTY arrays if XFOIL ran but nothing
    converged (bad geometry, universal VISCAL failure, timeout before the
    first converged point) -- the caller applies the -100.0 penalty contract.
    On timeout the partial polar written so far is parsed and returned (PACC
    appends row-by-row), but NOT cached. Raises XfoilError only for
    infrastructure faults (binary missing/unexecutable).

    ``concurrent=True`` (set by callers running sweeps in parallel) declares
    that ``timeout_s`` is being measured under contention, and suppresses
    caching of the timed-out-EMPTY result so a load-induced failure can
    never be written into the shared cache as if it were a property of the
    geometry. Default False is the legacy path, bit-for-bit.
    """
    coords = np.asarray(coords, dtype=float)
    alphas = [float(a) for a in alphas]
    n_req = len(alphas)
    if coords.ndim != 2 or coords.shape[1] != 2 or coords.shape[0] < 10:
        return _empty_result(n_req)  # not a plausible loop: fast design-failure
    if not np.all(np.isfinite(coords)) or n_req == 0:
        return _empty_result(n_req)  # NaN geometry / nothing requested

    key = cache_key(coords, re, mach, alphas, n_panel, with_cpmin)
    cache_path = Path(cache_dir if cache_dir is not None else DEFAULT_CACHE_DIR) / f"{key}.json"
    if cache_path.exists():
        hit = _cache_load(cache_path, n_req)
        if hit is not None:
            # filtered on READ, so caches written before the floor existed
            # serve the same polar as a fresh solve
            return drop_unphysical(hit, re)

    workdir = Path(tempfile.mkdtemp(prefix="aerobo_xfoil_"))
    timed_out = False
    try:
        # Labelled coordinate file (name line first): an unlabelled file makes
        # XFOIL prompt for a name and desynchronises the command stream.
        # Coordinates are written at the same 1e-6 rounding the cache key
        # uses, so key-equivalent geometry produces byte-identical input.
        c = coords.round(6)
        (workdir / _COORDS_NAME).write_text(
            "AEROBO-CST\n" + "\n".join(f" {x: .6f}  {y: .6f}" for x, y in c) + "\n"
        )
        script = _build_script(float(re), float(mach), np.asarray(alphas),
                               int(n_panel), with_cpmin)
        stdout = ""
        try:
            proc = run_xfoil_script(
                script, cwd=workdir, xfoil_bin=xfoil_bin, timeout_s=timeout_s
            )
            stdout = getattr(proc, "stdout", "") or ""
        except subprocess.TimeoutExpired:
            timed_out = True  # child killed by subprocess.run; salvage partial polar
        pol = workdir / _POLAR_NAME
        text = pol.read_text() if pol.exists() else ""
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    alpha_deg, cl, cd, cm, cdp, xtr_t, xtr_b = _parse_pacc_table(text)
    cp_min = np.empty(0, dtype=float)
    if with_cpmin and not timed_out:
        order = _cpmin_alpha_commands(np.asarray(alphas))[1]
        cp_min = _parse_cpmin(stdout, order, alpha_deg)
    result = XfoilPolarResult(
        alpha_deg=alpha_deg, cl=cl, cd=cd, cm=cm,
        n_requested=n_req, from_cache=False, cp_min=cp_min, cdp=cdp,
        xtr_top=xtr_t, xtr_bot=xtr_b,
    )
    # Timeout truncation is load-dependent: never cache a PARTIAL polar.
    # A timeout that produced NOTHING, however, is the dominant garbage-
    # geometry mode (XFOIL hangs at a prompt) and is deterministic for the
    # same geometry — cache it (tagged) so a BO revisit of a hopeless
    # design does not re-burn the full timeout budget every time.
    #
    # ...but only when this call had the machine to itself. `timeout_s` is
    # WALL-CLOCK, so under concurrency a perfectly good section can time out
    # purely from load; caching that as an empty polar would permanently
    # serve a fabricated failure to every FUTURE run, serial ones included.
    # Callers that run sweeps in parallel pass concurrent=True and forfeit
    # the hopeless-geometry shortcut rather than risk poisoning the cache.
    if not timed_out or (alpha_deg.size == 0
                         and not (concurrent or ASSUME_CONCURRENT)):
        meta = {
            "re": float(re), "mach": float(mach), "alphas": alphas,
            "n_panel": int(n_panel), "created": time.time(),
        }
        if timed_out:
            meta.update(timed_out_empty=True, timeout_s=float(timeout_s))
        # cached BEFORE filtering: the cache records what XFOIL said, the
        # floor is applied by the reader
        _cache_store(cache_path, result, meta=meta)
    return drop_unphysical(result, re)
