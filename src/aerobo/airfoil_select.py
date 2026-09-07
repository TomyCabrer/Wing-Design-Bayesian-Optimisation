"""Stages 1-2 of the §15 airfoil design pipeline: weighted multi-criterion
screening of a coordinate database -> ranked candidates -> base section.

Methodology is a Python port of the GDP tandem-wing airfoil ranking
(``~/Desktop/GDP/tandem_wing/legacy/scripts/rank_xfoil_sweep.m`` +
``compare_airfoils.m``), the same three-step design flow: fix the weights
first, screen the database second, and only then commit to a base section.

Scoring (GDP formula, verbatim semantics)
-----------------------------------------
Each metric is min-max normalised to [0, 100] ACROSS THE ELIGIBLE SET:

    higher-better:  s(v) = 100 (v - min v) / max(eps, max v - min v)
    lower-better:   s(v) = 100 (max v - v) / max(eps, max v - min v)

and the composite is sum_i w_i s_i with the weights normalised to sum 1.
Higher-better: t/c, cl_max, (L/D)_max, (L/D)_cruise, alpha_stall;
lower-better: |cm| at the design lift. The worst candidate scores 0 and
the best 100 on every metric, so a rank is only meaningful WITHIN one
screen run (the normalisation is population-relative — GDP property,
kept deliberately).

Frozen normalisation (ScoreReference)
-------------------------------------
The population-relative map is correct for ranking a FIXED catalogue and
useless as an OPTIMISATION target: the (min, max) pair moves whenever the
candidate set moves, so the composite is non-stationary, best-so-far is
not monotone in the design, and median-best across seeds compares scores
computed under different maps. ``ScoreReference`` replaces the live
(min, max) of each criterion with a frozen (lo, hi) pair read from disk,
which makes the composite a fixed function of the scored metrics alone.

The frozen pair is the p2/p98 percentile of the criterion over the
screened eligible population, NOT its (min, max): one 46%-thick section
(fx79w470a) stretches the raw t/c span to 2.211x the p2-p98 span, which
divides every t/c sub-score by 2.211 and so silently down-weights the
declared 0.10 thickness weight by the same factor. Scores are NOT clipped
into [0, 100] — a candidate outside the reference band must be allowed to
score past the ends, or the objective saturates exactly where the
optimiser is winning.


Weight presets are the GDP numbers with the two tunnel-trust terms
dropped (this screen is XFOIL-only, exactly like ``rank_xfoil_sweep.m``
whose header says "same as compare_airfoils.m 'sym' preset, no trust").
The GDP Prandtl-Glauert Cl correction (factor 1.095 at M 0.407) is ALSO
dropped: the UROP mission is incompressible (M = 0 polars), so the
correction is identically 1.

Operating point / gates
-----------------------
Metrics are extracted from a real viscous XFOIL polar at the
AirfoilProblem conditions (Re 1e6, M 0, alphas -4..10 x 0.5, n_panel 200,
SHA-cached) with the SAME pre-stall-branch cl-interpolation as
``evaluate_airfoil`` — screening metrics and stage-3 objective values are
commensurable by construction. Hard eligibility gates mirror the stage-3
constraints (t/c >= tc_min, |cm| <= cm_max) plus convergence sanity
(>= MIN_CONVERGED points, cl_design bracketed by the monotone branch).
Ineligible candidates keep their metrics in the record (status/reason
fields) but take no part in the min-max normalisation.

XfoilError (missing/unexecutable binary) PROPAGATES — infrastructure
fault, never a scored failure (airfoil.py contract).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import hashlib
import json
import os
import threading

import numpy as np

from .airfoil import (
    G_FAIL,
    PENALTY,
    AirfoilProblem,
    _longest_increasing_run,
    anchor_box,
    cst_anchor_from_coords,
    cst_coords,
    cst_thickness,
    evaluate_airfoil,
)

# GDP linear-band restriction for (L/D)_max: max(cl/cd) over |alpha| <= 8 deg
# only (compare_airfoils.m) — keeps the metric off noisy near-stall points.
LD_MAX_ALPHA_BAND = 8.0

# Screening needs a trustworthy cl_max on top of the cd interpolation, so the
# convergence bar is stricter than evaluate_airfoil's >= 3 (GDP's run_sweep
# required >= 5 converged points).
MIN_CONVERGED = 5

# Second, wider sweep for the STALL metrics only (clmax / astall): the cruise
# sweep stops at 10 deg (stage-3 commensurability), which CENSORS clmax for
# most sections — GDP swept to +20 deg for exactly this reason. Marches away
# from alpha = 0 (xfoil_run trap). cd_at / cm_at / ldcr stay on the cruise
# sweep so screening and stage-3 objective values remain bit-comparable.
STALL_ALPHAS = np.arange(0.0, 20.5, 0.5)

HIGHER_BETTER = ("thick", "clmax", "ldmax", "ldcr", "astall")
# ``cdcr`` — the DRAG at the design lift, the criterion a surface that flies at
# ZERO lift is actually chosen on.
#
# Both efficiency criteria are ratios with the design lift on top: ``ldcr`` is
# |cl_design|/cd_at, and ``ldmax`` is max(cl/cd) over the linear band, i.e. the
# same kind of number read at whatever lift happens to maximise it. On a
# VERTICAL STABILISER, which must make no side force at zero sideslip, the
# first is identically 0 for every candidate and the second is measured at a
# lift the surface never carries. Measured over the 226 eligible symmetric
# library sections at zero lift, (L/D)max ranks them almost independently of
# the drag they actually cost: Spearman +0.077 against -cd_at (Pearson +0.359),
# and the section the shipped fin preset elected carries 61.9 % more zero-lift
# drag than the lowest-drag section it was ranked against. So a fin had no
# criterion for its own drag at all, which is what this one is.
#
# LOWER-better, mirrored on the frozen band exactly as |cm| is. The value is
# ``cd_at`` — already computed by :func:`polar_metrics`, already shown as the
# ranking's unweighted "cd @Cl" column — so nothing new is measured and no
# XFOIL sweep is added.
LOWER_BETTER = ("cm", "cdcr")
# Canonical criterion order — fixes the weight-vector layout that the frozen
# reference SHA is taken over. ``cdcr`` is APPENDED, never inserted: the five
# shipped reference payloads hash their bands in this order (:func:`reference_sha`)
# and an insertion would silently re-order every one of them.
CRITERIA = HIGHER_BETTER + LOWER_BETTER

# Extra minimum floors a caller may impose on top of the two stage-3 gates
# (t/c >= tc_min, |cm| <= cm_max, which travel on the AirfoilProblem). Each
# maps a floor key to the record field it gates, higher-is-better: a section
# is dropped when the metric is below the floor. All optional — an empty
# ``floors`` dict reproduces the legacy two-gate screen bit-for-bit.
FLOOR_FIELDS = {"clmax": "clmax", "ldcr": "ldcr",
                "ldmax": "ldmax", "astall": "astall"}

#: A floor on how NOSE-DOWN the section's moment must be: the gated quantity
#: is -cm_at, so ``nose_down = 0.0`` admits only cm <= 0 and rejects every
#: REFLEXED section. Separate from FLOOR_FIELDS because it gates the negative
#: of a record field rather than the field.
#:
#: Why this exists: |cm| is a lower-better CRITERION, so a screen that ranks
#: on it rewards reflex — and reflex is what an aerofoil does INSTEAD of
#: having a tail. Measured on the published wing screen (gdp-sweep, cm 0.20),
#: five of the top six were reflexed or near-zero, four of them Horten (hg*)
#: and Hepperle (mh*) FLYING-WING sections. Flown on a tailed aircraft they
#: invert the stabiliser's job: the wing holds its own pitching moment, so
#: the trim balance asks the tail for UP-load and a "stabiliser" lifts.
#: 25.2 % of the 2120 library sections do it at the published CG.
NOSE_DOWN_FLOOR = "nose_down"


# ---------------------------------------------------------------- weights


@dataclass(frozen=True)
class ScoreWeights:
    """Criterion weights (auto-normalised to sum 1 in ``normalised``).

    Defaults = the GDP bulk-sweep preset (rank_xfoil_sweep.m, verbatim):
    thick 0.10, Clmax 0.20, LDmax 0.15, LDcr 0.35, Cm 0.20 — "same as
    compare_airfoils.m 'sym' preset, no trust (XFOIL only)". astall
    (stall-margin criterion of compare_airfoils.m) defaults to 0 here
    because the bulk-sweep preset did not score it.
    """

    thick: float = 0.10
    clmax: float = 0.20
    ldmax: float = 0.15
    ldcr: float = 0.35
    cm: float = 0.20
    astall: float = 0.0
    #: drag at the design lift, lower-better. DEFAULTS TO ZERO, so every
    #: preset, every published composite and every stored run is unchanged to
    #: the bit: ``normalised`` divides by a sum this term does not move, and a
    #: zero-weight criterion contributes nothing to J, nothing to the goal
    #: composite's penalty and is DROPPED from the ASF's aggregates
    #: (:func:`asf_terms`). It is weighted only where the surface has no other
    #: drag criterion — the vertical stabiliser (gui.v3.session.FIN_WEIGHTS).
    cdcr: float = 0.0

    def normalised(self) -> dict[str, float]:
        d = asdict(self)
        if any(v < 0.0 for v in d.values()):
            raise ValueError(f"negative weight in {d}")
        tot = sum(d.values())
        if tot <= 0.0:
            raise ValueError("all weights zero")
        return {k: v / tot for k, v in d.items()}


# GDP presets, trust terms dropped (XFOIL-only screen); 'gdp-sweep' is the
# default and equals the rank_xfoil_sweep.m numbers.
PRESETS: dict[str, ScoreWeights] = {
    "gdp-sweep": ScoreWeights(),
    "gdp-fwd": ScoreWeights(thick=0.10, clmax=0.25, ldmax=0.10,
                            ldcr=0.40, cm=0.10, astall=0.05),
    "gdp-rear": ScoreWeights(thick=0.10, clmax=0.10, ldmax=0.05,
                             ldcr=0.40, cm=0.10, astall=0.20),
    "gdp-sym": ScoreWeights(thick=0.10, clmax=0.20, ldmax=0.15,
                            ldcr=0.25, cm=0.10, astall=0.10),
}


# ------------------------------------------------------ frozen normalisation

# Default reference band. p2/p98 rather than (min, max) because the GDP
# min-max map is hostage to single outliers — see module docstring and
# ``build_screen_reference``.
REF_P_LO, REF_P_HI = 2.0, 98.0

# Reference payload format; bumped whenever the stored fields change meaning,
# so an old file fails loudly instead of being read under new semantics.
# v2: the (lo, hi) bands themselves entered reference_sha's hashed blob — a
# v1 payload's stored sha cannot validate under v2 and must be rebuilt.
REFERENCE_VERSION = 2

#: THE BAND FOR A CRITERION THE SHIPPED PAYLOADS PREDATE.
#:
#: ``cdcr`` (drag at the design lift) was added after five reference payloads
#: were already frozen and published — ``data/screen_reference.json``, its two
#: per-Reynolds siblings, and the two design-box bands — and two of those
#: cannot be re-measured without re-running screens the published numbers were
#: taken on. Bumping :data:`REFERENCE_VERSION` would have invalidated all five;
#: measuring the new band ONCE, over the same population and stating it here,
#: invalidates nothing: every existing payload keeps its numbers and its SHA
#: (see :func:`reference_sha`), and a payload built from now on carries its own
#: ``cdcr`` band, which WINS over this fallback (:meth:`ScoreReference.band`).
#:
#: Measured exactly as the others were: p2/p98 of ``cd_at`` over the SAME 631
#: eligible sections of the frozen §15 UIUC screen, at the SAME point
#: (:data:`LEGACY_SCREEN_POINT`, Re 1e6 / M 0) and the same design lift
#: (cl 0.5, ``AirfoilProblem``'s default) and the same gates (tc_min 0.10,
#: cm_max 0.08). It is RE-DERIVED from that screen, not pinned:
#: ``tests/test_airfoil_select.py::
#: test_shipped_reference_is_the_p2p98_of_the_frozen_screen`` rebuilds the
#: payload off the frozen checkpoint and asserts this pair, so a hard-coded
#: number cannot drift away from the measurement it claims to be.
#:
#: It is a band and not a limit: sub-scores are not clipped
#: (:func:`score_candidates`), so a fin section whose zero-lift drag sits well
#: under the whole library's cruise drag scores past 100 and is still ranked
#: against its rivals on the same slope.
LATE_CRITERION_BANDS: dict[str, tuple[float, float]] = {
    "cdcr": (0.0054042307343555645, 0.013155037031561819),
}

# Shipped reference: p2/p98 over the 631 eligible sections of the frozen §15
# UIUC screen (gdp-sweep preset, tc_min 0.10, cm_max 0.08).
SCREEN_REFERENCE_PATH = (Path(__file__).resolve().parents[2]
                         / "data" / "screen_reference.json")

#: The measured alternative: p2/p98 of what the 8-D CST DESIGN BOX reaches,
#: rather than of what a 631-section catalogue contains
#: (``scripts/box_band_probe.py``). ``BOX_BAND_PATH`` is the 2412 box — the one
#: the shipped search searches; ``BOX_BAND_2415_PATH`` is the 2415 box, which
#: is the band that carries the HELD-OUT claim because no search here uses the
#: box it was measured over.
#:
#: These are OPTIONS, not defaults, and the distinction is the whole decision
#: of `RESULTS_SESSION47_VERDICTS.md` §E: measured on 42 paired seeds the
#: 2415 band buys `cd` -0.00056 at the design lift (sign 0.0009 / Wilcoxon
#: 0.0001 / paired t 0.0001, n80 = 18) and cruise L/D +8.07, at a plain-J
#: difference that is null on all three tests — but the shipped band stays the
#: default, because every published number in report §15 and §16 is measured on
#: it and re-basing those is a separate decision this study did not price.
#:
#: They are tracked here rather than left in `results/` because `results/` is
#: gitignored: an option a user cannot load without re-running a four-minute
#: XFOIL probe is prose, not an option.
BOX_BAND_PATH = (Path(__file__).resolve().parents[2] / "data" / "box_band.json")
BOX_BAND_2415_PATH = (Path(__file__).resolve().parents[2]
                      / "data" / "box_band_2415.json")


def load_box_band(path: str | Path = BOX_BAND_PATH) -> dict:
    """The measured box band as a payload ``score_reference=`` takes verbatim.

    Returns the stored dict rather than a :class:`ScoreReference`, because that
    is the form every entry point already accepts (``api.optimize_airfoil``,
    ``api.airfoil_run_config``, ``api.pareto_airfoil``) and converting here
    would add a second way to say the same thing.

    It is deliberately NOT loadable through :func:`load_screen_reference`, and
    its ``sha`` says why: this band was not measured over the library, so it
    must not be able to masquerade as a screen reference. That loader would
    reject it, and should.
    """
    path = Path(path)
    payload = json.loads(path.read_text())
    # ...and the same one documented exemption the screen loader makes: a band
    # measured before a criterion existed omits it and falls back to
    # :data:`LATE_CRITERION_BANDS`. Both box bands are XFOIL SAMPLES of a
    # design box, not a re-derivable screen, so re-measuring one is a study and
    # not a rebuild.
    missing = ({k for k in CRITERIA if k not in LATE_CRITERION_BANDS}
               - set(payload.get("bounds") or {}))
    if missing:
        raise ValueError(f"{path.name} is not a band payload: no bounds for "
                         f"{sorted(missing)}")
    for k, (lo, hi) in payload["bounds"].items():
        if not (float(hi) > float(lo)):
            raise ValueError(f"{path.name}: degenerate band for {k!r} "
                             f"({lo}..{hi}) — a zero-width band divides by "
                             "zero in every exchange rate")
    return payload


def reference_sha(names, gates: dict, weights: dict, bounds: dict) -> str:
    """Fingerprint of the reference: the POPULATION it was measured over
    (sorted section names + gate values + the weight vector in CRITERIA
    order) AND the (lo, hi) bands themselves.

    ``bounds`` is in the blob because it is the only field ``score_candidates``
    ever reads: hashing the provenance while leaving the numbers unhashed
    would let an edited band pass the integrity check silently, which is
    precisely the failure this SHA exists to stop. Any input changing changes
    every number the composite reports, so the SHA travels with the reference
    and ``load_screen_reference`` refuses a payload that no longer hashes to
    it."""
    # Indexed over the keys the payload ACTUALLY carries, in CRITERIA order,
    # rather than over CRITERIA itself. The two are the same thing for a
    # payload measured under the criterion set that hashed it — which is what
    # keeps the five shipped references validating BIT-FOR-BIT after a
    # criterion was appended (:data:`LATE_CRITERION_BANDS`). Hashing CRITERIA
    # directly would have raised KeyError on every one of them, and padding the
    # missing entry with a zero would have changed every stored SHA.
    blob = json.dumps(
        {"names": sorted(str(n) for n in names),
         "gates": {k: float(gates[k]) for k in sorted(gates)},
         "weights": [float(weights[k]) for k in CRITERIA if k in weights],
         "bounds": [[float(bounds[k][0]), float(bounds[k][1])]
                    for k in CRITERIA if k in bounds]},
        sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class ScoreReference:
    """Frozen per-criterion (lo, hi) normalisation band.

    Substituted for the live (min, max) inside ``score_candidates``, which
    turns the GDP composite from a population-relative rank into a fixed
    function of the scored metrics — the property an optimisation target needs
    (stationary objective, monotone best-so-far, cross-seed comparability).

    ``bounds`` must carry every CRITERIA key with hi > lo, EXCEPT two
    documented cases: the ones in :data:`LATE_CRITERION_BANDS` — criteria
    added after the shipped payloads were measured, which fall back to a band
    measured over the same population and stated in this module (a payload
    that DOES carry one of them wins over the fallback) — and the ones the
    population itself could not separate, which the payload has to DECLARE in
    ``unbanded`` ({criterion: why}). ``sha`` identifies the population it was
    measured over (``reference_sha``).

    THE SECOND CASE IS A REAL SURFACE, not a corner. A vertical stabiliser is
    screened at zero design lift, and ``ldcr`` is |cl_design|/cd_at — so it is
    identically 0 for every candidate the fin could possibly be given, the
    measured band is (0.0, 0.0), and there is no map from that criterion's
    values onto 0-100 points. Refusing the whole band for it (which is what a
    degenerate entry did) refused every composite run on every fin, including
    the shipped preset that weights the criterion at zero. So the band DROPS
    it and says why, and the composite refuses a WEIGHT on it instead
    (:func:`score_candidates`) — the same shape ``wing_score`` already ships
    for a family that does not report a criterion.
    """

    bounds: dict[str, tuple[float, float]]
    sha: str = ""
    n_records: int = 0
    preset: str = ""
    #: {criterion: why it has no band} — the criteria this population could
    #: not separate, stated rather than silently absent. None means "none".
    unbanded: dict = None

    def __post_init__(self) -> None:
        dead = dict(self.unbanded or {})
        stated = set(dead) - set(CRITERIA)
        if stated:
            raise ValueError(
                f"reference declares unknown criteria unbanded "
                f"{sorted(stated)} (known: {list(CRITERIA)})")
        both = set(dead) & set(self.bounds)
        if both:
            raise ValueError(
                f"reference both bands and declares unbanded {sorted(both)} "
                "— a criterion has a band or it does not")
        required = tuple(k for k in CRITERIA
                         if k not in LATE_CRITERION_BANDS and k not in dead)
        missing = set(required) - set(self.bounds)
        extra = set(self.bounds) - set(CRITERIA)
        if missing or extra:
            raise ValueError(
                f"reference bounds must be exactly {list(required)} "
                f"(plus, optionally, {sorted(LATE_CRITERION_BANDS)}) "
                f"(missing {sorted(missing)}, extra {sorted(extra)})")
        if not self.bounds:
            raise ValueError(
                "no criterion varied over the screened population, so there "
                "is nothing to normalise against")
        for key, (lo, hi) in self.bounds.items():
            if not (np.isfinite(lo) and np.isfinite(hi)):
                raise ValueError(f"non-finite reference band for {key}")
            if hi <= lo:
                raise ValueError(
                    f"degenerate reference band for {key}: "
                    f"hi {hi} <= lo {lo}")

    def covers(self, key: str) -> bool:
        """Can this band score ``key`` at all?

        False only for a criterion the population could not separate. The
        LATE_CRITERION_BANDS fallback counts as covered — that criterion has a
        band, it just is not stored in the payload.
        """
        return key not in dict(self.unbanded or {})

    def why_unbanded(self, key: str) -> str:
        """Why ``key`` has no band, in the measurement's own words."""
        return str(dict(self.unbanded or {}).get(key, ""))

    def band(self, key: str) -> tuple[float, float]:
        if not self.covers(key):
            # never a fabricated band: every caller that scores a criterion
            # has to decide what to do about one that cannot be scored, and
            # only the ones holding the WEIGHTS can (a zero weight is fine,
            # a positive one is a refusal)
            raise ValueError(
                f"no normalisation band for {key}: {self.why_unbanded(key)}")
        pair = self.bounds.get(key)
        if pair is None:
            pair = LATE_CRITERION_BANDS[key]      # KeyError = a real mistake
        lo, hi = pair
        return float(lo), float(hi)


def load_screen_reference(path: str | Path = SCREEN_REFERENCE_PATH,
                          expect_sha: str | None = None) -> ScoreReference:
    """Load a frozen reference, RAISING on any inconsistency.

    Deliberately the opposite of ``_load_json_checkpoint``'s tolerance: a
    checkpoint that fails to load costs XFOIL time, whereas a reference that
    silently fails to load (or loads stale) changes EVERY composite score in
    the report and every objective value the optimiser saw, with no visible
    symptom. So: malformed JSON, a missing field, a wrong format version, a
    self-inconsistent SHA, or a mismatch against ``expect_sha`` (the SHA of
    the population being scored NOW) all raise ValueError.
    """
    path = Path(path)
    try:
        raw = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"screen reference {path} unreadable: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"screen reference {path}: not a JSON object")

    for field in ("version", "bounds", "records", "gates", "weights", "sha"):
        if field not in raw:
            raise ValueError(f"screen reference {path}: missing '{field}'")
    if int(raw["version"]) != REFERENCE_VERSION:
        raise ValueError(
            f"screen reference {path}: format version {raw['version']} "
            f"!= {REFERENCE_VERSION}")

    # Structural validation BEFORE hashing: reference_sha indexes weights by
    # criterion and bounds by pair, so a truncated payload would otherwise
    # surface as KeyError/IndexError/TypeError from inside the hash — the
    # loader's contract is that every malformed reference raises ValueError.
    if not isinstance(raw["weights"], dict):
        raise ValueError(f"screen reference {path}: 'weights' is not an object")
    if not isinstance(raw["bounds"], dict):
        raise ValueError(f"screen reference {path}: 'bounds' is not an object")
    # A payload measured before a criterion existed may omit it, and only
    # those: :data:`LATE_CRITERION_BANDS` names exactly which, and
    # ``ScoreReference.band`` says what is used instead. Every other criterion
    # is still required, and a payload that carries a late one is validated
    # like any other — an omission has to be the ONE documented case, never a
    # truncated file.
    unbanded = raw.get("unbanded") or {}
    if not isinstance(unbanded, dict):
        raise ValueError(f"screen reference {path}: 'unbanded' is not an "
                         f"object")
    for key in CRITERIA:
        if key not in raw["weights"]:
            if key in LATE_CRITERION_BANDS:
                continue
            raise ValueError(
                f"screen reference {path}: weights missing criterion '{key}'")
        pair = raw["bounds"].get(key)
        # …the second documented omission: a criterion the measured population
        # could not separate, which the payload has to NAME (see
        # ``build_screen_reference``). Undeclared, it is still a truncated file
        if pair is None and key in unbanded:
            continue
        if pair is None and key in LATE_CRITERION_BANDS:
            continue
        if (not isinstance(pair, (list, tuple)) or len(pair) != 2
                or not all(isinstance(v, (int, float)) for v in pair)):
            raise ValueError(
                f"screen reference {path}: bounds['{key}'] must be a "
                f"2-element numeric [lo, hi], got {pair!r}")
    if not isinstance(raw["records"], list):
        raise ValueError(f"screen reference {path}: 'records' is not a list")
    n_declared = int(raw.get("n_records", len(raw["records"])))
    if n_declared != len(raw["records"]):
        raise ValueError(
            f"screen reference {path}: n_records {n_declared} != "
            f"{len(raw['records'])} listed record names")

    got = reference_sha(raw["records"], raw["gates"], raw["weights"],
                        raw["bounds"])
    if got != raw["sha"]:
        raise ValueError(
            f"screen reference {path}: SHA mismatch — stored {raw['sha']}, "
            f"recomputed {got} from {len(raw['records'])} records / gates "
            f"{raw['gates']} / weights {raw['weights']}. The reference has "
            "been edited; rebuild it, do not score against it.")
    if expect_sha is not None and got != expect_sha:
        raise ValueError(
            f"screen reference {path}: population SHA {got} != expected "
            f"{expect_sha} — this reference was measured over a different "
            "population/gate/weight set.")

    return ScoreReference(
        bounds={k: (float(v[0]), float(v[1]))
                for k, v in raw["bounds"].items()},
        sha=str(raw["sha"]), n_records=int(raw.get("n_records", 0)),
        preset=str(raw.get("preset", "")),
        unbanded={str(k): str(v) for k, v in unbanded.items()})


# ---------------------------------------------------------------- io helpers


def _atomic_write_json(path: Path, obj) -> None:
    """Crash-safe checkpoint write (xfoil_run._cache_store pattern): a kill
    mid-write leaves the OLD file intact, never torn JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)


def _load_json_checkpoint(path: Path, label: str) -> dict:
    """Tolerant checkpoint load: corrupt/torn JSON degrades to a fresh run
    (with a loud warning), never a hard crash on resume."""
    try:
        obj = json.loads(path.read_text())
        return obj if isinstance(obj, dict) else {}
    except (json.JSONDecodeError, OSError) as exc:
        print(f"{label} checkpoint unreadable ({exc}) — starting fresh",
              flush=True)
        return {}


def file_sha(path: str | Path) -> str:
    """Content fingerprint stored per screen record: a checkpoint hit is
    only trusted while the .dat bytes are unchanged."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


# ---------------------------------------------------------------- parsing


def load_airfoil_dat(path: str | Path) -> np.ndarray:
    """Robust UIUC .dat reader -> (n, 2) closed loop, unit chord, XFOIL order.

    Handles Selig files (name line, then TE-upper -> LE -> TE-lower pairs)
    and Lednicer files (name line, "Nu. Nl." count line, upper LE -> TE
    block, lower LE -> TE block). Non-numeric lines are skipped, blank
    lines separate Lednicer blocks, consecutive duplicate points are
    dropped, and coordinates are normalised to x in [0, 1] (shift + single
    isotropic chord scale). Raises ValueError on anything unusable.
    """
    path = Path(path)
    blocks: list[list[tuple[float, float]]] = [[]]
    for raw in path.read_text(errors="replace").splitlines():
        s = raw.strip()
        if not s:
            if blocks[-1]:
                blocks.append([])
            continue
        parts = s.replace(",", " ").split()
        try:
            vals = [float(p) for p in parts]
        except ValueError:
            continue                      # header / annotation line
        if len(vals) != 2:
            # Coordinate pairs are exactly 2 columns in this database; a
            # >2-float line is a header (e.g. the MSES grid-domain line of
            # the tasopt-* files: "-2.0 3.0 -2.646 3.454"), not a point.
            continue
        blocks[-1].append((vals[0], vals[1]))
    blocks = [b for b in blocks if b]
    if not blocks:
        raise ValueError(f"{path.name}: no coordinate pairs found")

    loop = None
    if (len(blocks) >= 2
            and blocks[0] and blocks[0][0][0] > 1.5
            and float(blocks[0][0][0]).is_integer()
            and float(blocks[0][0][1]).is_integer()):
        # Lednicer candidate: first numeric line is the "Nu. Nl." count
        # pair, surfaces run LE -> TE. Guarded: if the would-be upper
        # surface is not x-ascending this is really a (rescaled) Selig
        # file whose blank lines split it — fall through to the join.
        pts = blocks[0][1:] if len(blocks[0]) > 1 else []
        if pts:                            # counts shared the first block
            upper, lower = pts, [p for b in blocks[1:] for p in b]
        elif len(blocks) >= 3:
            upper = blocks[1]
            lower = [p for b in blocks[2:] for p in b]
        else:
            upper = lower = []
        if (len(upper) >= 2 and len(lower) >= 2
                and upper[0][0] < upper[-1][0]
                and lower[0][0] < lower[-1][0]):
            loop = list(reversed(upper)) + lower[1:]
    if loop is None:
        loop = [p for b in blocks for p in b]

    c = np.asarray(loop, dtype=float)
    keep = np.ones(len(c), dtype=bool)
    keep[1:] = np.any(np.abs(np.diff(c, axis=0)) > 1e-12, axis=1)
    c = c[keep]
    if c.shape[0] < 20:
        raise ValueError(f"{path.name}: only {c.shape[0]} distinct points")

    x_min, x_max = float(c[:, 0].min()), float(c[:, 0].max())
    chord = x_max - x_min
    if chord <= 1e-9:
        raise ValueError(f"{path.name}: zero chord")
    c = (c - [x_min, 0.0]) / chord         # unit chord, LE at x = 0

    i_le = int(np.argmin(c[:, 0]))
    if i_le < 3 or i_le > c.shape[0] - 4:
        raise ValueError(f"{path.name}: LE at loop end — not a closed loop")
    if np.max(np.abs(c[:, 1])) > 0.5:
        raise ValueError(f"{path.name}: |y|/c > 0.5 — not an airfoil")
    return c


def split_surfaces(
    coords: np.ndarray, n_grid: int = 201
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Closed loop -> (x_grid, y_upper, y_lower) on a common cosine grid
    (the cst_anchor recipe: split at min-x, sort each branch by x)."""
    x, y = coords[:, 0], coords[:, 1]
    i_le = int(np.argmin(x))
    xu, yu = x[: i_le + 1][::-1], y[: i_le + 1][::-1]
    xl, yl = x[i_le:], y[i_le:]
    su, sl = np.argsort(xu), np.argsort(xl)
    xg = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, n_grid)))
    return xg, np.interp(xg, xu[su], yu[su]), np.interp(xg, xl[sl], yl[sl])


def geometric_tc(coords: np.ndarray) -> float:
    """Max thickness-to-chord measured from the coordinates (GDP measured
    t/c from coords for non-NACA sections)."""
    _, yu, yl = split_surfaces(coords, n_grid=1001)
    return float(np.max(yu - yl))


# ---------------------------------------------------------------- metrics


def polar_metrics(pol, coords: np.ndarray, prob: AirfoilProblem,
                  pol_stall=None) -> dict:
    """Screening metrics from the cruise-sweep XfoilPolarResult (plus an
    optional wide stall sweep), evaluate_airfoil-consistent
    cl-interpolation. Returns a record with ``status`` != "ok" (and NaN
    metrics) when the polar cannot support the metric set.

    clmax / astall come from the argmax over the cruise + stall sweeps
    combined; ``clmax_censored`` flags a max sitting at the LAST CONVERGED
    alpha (still rising where the data stops — the true clmax is ABOVE the
    reported value). Without a stall sweep the cruise cap of 10 deg censors
    almost every section, which is why screen_one runs the second sweep.
    """
    rec = {
        "status": "ok", "reason": "",
        "tc": geometric_tc(coords),
        "n_converged": int(pol.n_converged),
        "clmax": float("nan"), "astall": float("nan"),
        "clmax_censored": True,
        "ldmax": float("nan"),
        "cd_at": float("nan"), "cm_at": float("nan"),
        "alpha_at": float("nan"), "ldcr": float("nan"),
    }
    if pol.n_converged < MIN_CONVERGED:
        rec.update(status="unconverged",
                   reason=f"{pol.n_converged} < {MIN_CONVERGED} converged "
                          "XFOIL points")
        return rec

    pols = [pol] + ([pol_stall] if pol_stall is not None
                    and pol_stall.n_converged > 0 else [])
    a_all = np.concatenate([p.alpha_deg for p in pols])
    cl_all = np.concatenate([p.cl for p in pols])
    i_max = int(np.argmax(cl_all))
    rec["clmax"] = float(cl_all[i_max])
    rec["astall"] = float(a_all[i_max])
    # Censoring is a property of the CONVERGED data, not of the REQUESTED
    # sweep: XFOIL silently omits every alpha whose boundary-layer march
    # diverged (xfoil_run module contract), so a section whose march dies at
    # 6.5 deg leaves its cl still climbing at the last row it produced —
    # the true clmax is above the reported value exactly as for a section
    # swept to 20 deg and still rising. Comparing a_all[i_max] against the
    # last CONVERGED alpha therefore catches BOTH cases; the old comparison
    # against the top of the swept range (20 deg with a stall sweep, else
    # max(prob.alphas)) reported those early-death marches as uncensored.
    # The comparison is EXACT, with no slack: a_all.max() is itself a
    # converged data point, not a requested bound, so a peak one sweep step
    # below it means the polar actually MEASURED a decline past the peak —
    # that is an observed stall, not a truncated one. (The old rule needed
    # 0.51 deg of slack only because it compared against a requested alpha
    # that no converged row need coincide with.)
    rec["clmax_censored"] = bool(a_all[i_max] >= float(a_all.max()))

    band = np.abs(pol.alpha_deg) <= LD_MAX_ALPHA_BAND
    band &= pol.cd > 0.0
    if band.any():
        # max cl/cd over the band: the section's best L/D on its LIFTING
        # side. Deliberately left signed and point-INDEPENDENT, unlike
        # ``ldcr`` below. Two reasons, and the second is the binding one:
        #   * it is a property of the section, not of the operating point;
        #   * it is SHA-cached with the rest of the point-independent
        #     metrics (screen_one), so a value that depended on cl_design
        #     would be served from a warm checkpoint answered at another
        #     one — the exact silent-staleness this cache is documented to
        #     avoid.
        # The limit that leaves is real and is stated rather than papered
        # over: for a surface that flies a NEGATIVE design lift (a
        # conventional stabiliser — tail.trim_lift_coefficient) this
        # criterion measures the side of the polar that surface never
        # works. Only ``ldcr``, which IS evaluated at the design point,
        # speaks for it there.
        rec["ldmax"] = float(np.max(pol.cl[band] / pol.cd[band]))

    sl = _longest_increasing_run(pol.cl)
    cl_b, cd_b = pol.cl[sl], pol.cd[sl]
    cm_b, al_b = pol.cm[sl], pol.alpha_deg[sl]
    if (cl_b.size < 2 or cl_b[-1] < prob.cl_design
            or cl_b[0] > prob.cl_design):
        rec.update(status="no_cl_bracket",
                   reason="cl_design not bracketed by the pre-stall "
                          "monotone branch")
        return rec
    rec["cd_at"] = float(np.interp(prob.cl_design, cl_b, cd_b))
    rec["cm_at"] = float(np.interp(prob.cl_design, cl_b, cm_b))
    rec["alpha_at"] = float(np.interp(prob.cl_design, cl_b, al_b))
    if rec["cd_at"] <= 0.0 or not np.isfinite(rec["cd_at"]):
        rec.update(status="bad_cd", reason=f"cd_at = {rec['cd_at']}")
        return rec
    # |cl_design|/cd, higher-better. The magnitude is the point: a
    # stabiliser's design lift is NEGATIVE on any conventional layout
    # (tail.trim_lift_coefficient), and cl/cd would then be negative for
    # every candidate, so the min-max normalisation (module docstring)
    # would award 100 to the LEAST negative — i.e. elect the DRAGGIEST
    # section in the database. Identical for every cl_design > 0 caller.
    rec["ldcr"] = abs(prob.cl_design) / rec["cd_at"]
    return rec


def _floor_reason(rec: dict, floors: dict | None) -> str | None:
    """First unmet minimum floor for ``rec`` (or None if all pass).

    Floors are a RANKING-time filter, kept out of the SHA-cached record so
    one screen checkpoint serves any floor setting: the cached XFOIL metrics
    never change, only which of them clear the user's minimums."""
    if not floors:
        return None
    for key, field in FLOOR_FIELDS.items():
        lim = floors.get(key)
        if lim is None:
            continue
        val = rec.get(field, float("nan"))
        if not (val >= float(lim)):
            return f"{field} {val:.4g} < floor {float(lim):.4g}"
    lim = floors.get(NOSE_DOWN_FLOOR)
    if lim is not None:
        cm = rec.get("cm_at", float("nan"))
        if not (-cm >= float(lim)):
            return (f"cm {cm:+.4g} is not nose-down enough "
                    f"(needs cm <= {-float(lim):+.4g})")
    return None


def _alphas_sig(alphas) -> list:
    """Alpha sweep signature, rounded exactly as the XFOIL cache key rounds."""
    return [round(float(a), 4) for a in alphas]


def screen_point(prob: AirfoilProblem) -> dict:
    """The cl-INDEPENDENT part of a screening operating point.

    Two screens sharing this dict share every XFOIL polar, so one can be
    re-derived from the other by interpolation alone; differing on it means
    different polars and therefore real XFOIL work. This is the identity a
    screened RECORD is valid at — a checkpoint hit is only a hit at the same
    point (:func:`screen_database`).
    """
    return {"re": float(prob.re), "mach": float(prob.mach),
            "n_panel": int(prob.n_panel), "alphas": _alphas_sig(prob.alphas)}


#: the point a checkpoint record with NO stamp was screened at.
#:
#: Records written before screening became point-aware carry no ``point``
#: field, and the checkpoint has no header to read one from. They were all
#: produced by the default :class:`AirfoilProblem` — Re 1e6, M 0, 200 panels,
#: alpha -4..10 x 0.5 — which is what ``results/airfoil_screen.json``
#: (``conditions``: re 1e6, mach 0) and the branch sidecar's own ``point``
#: both record for that screen. Stamping them as such keeps a warm checkpoint
#: warm at the point it really covers, instead of either re-running the whole
#: database or silently reusing 1e6 polars at another Reynolds number.
LEGACY_SCREEN_POINT = screen_point(AirfoilProblem())


def record_point(rec: dict) -> dict:
    """The operating point a screened record's metrics belong to."""
    return dict(rec.get("point") or LEGACY_SCREEN_POINT)


def screen_one(name: str, path: str | Path, prob: AirfoilProblem,
               concurrent: bool = False) -> dict:
    """Parse + XFOIL (cruise sweep + wide stall sweep) + metrics +
    eligibility gates for one database file.

    ``concurrent`` is forwarded to the sweeps: when this screen runs inside
    the thread pool, a wall-clock timeout may be pure contention, and the
    empty result must not be cached as a property of the section.

    The record carries the POINT it was screened at (:func:`screen_point`):
    every metric on it is a property of the section AND of that point, and a
    checkpoint keyed by name alone cannot tell the two apart.

    Never raises for in-contract failures (parse errors, unconvergeable
    sections); XfoilError propagates (infrastructure)."""
    from .xfoil_run import run_xfoil_polar   # late import: monkeypatchable

    rec = {"name": name, "path": str(path), "sha": file_sha(path),
           "point": screen_point(prob), "eligible": False}
    try:
        coords = load_airfoil_dat(path)
    except ValueError as exc:
        rec.update(status="parse_error", reason=str(exc))
        return rec

    pol = run_xfoil_polar(
        coords, prob.re, prob.mach, prob.alphas,
        timeout_s=prob.timeout_s, cache_dir=prob.cache_dir,
        n_panel=prob.n_panel, concurrent=concurrent,
    )
    pol_stall = None
    if pol.n_converged >= MIN_CONVERGED:
        # stall sweep only when the cruise sweep converged — an
        # unconvergeable section is already out, no point burning 20 s
        pol_stall = run_xfoil_polar(
            coords, prob.re, prob.mach, STALL_ALPHAS,
            timeout_s=prob.timeout_s, cache_dir=prob.cache_dir,
            n_panel=prob.n_panel, concurrent=concurrent,
        )
    rec.update(polar_metrics(pol, coords, prob, pol_stall=pol_stall))
    # The PRE-STALL MONOTONE BRANCH travels with the record, JSON-safe.
    #
    # Everything above depends on the design lift (cd_at, cm_at, alpha_at,
    # ldcr and the no_cl_bracket status), and a checkpoint keyed by name and
    # point cannot see a changed cl_design — so a warm checkpoint answered at
    # whatever Cl it happened to be screened at. Re-deriving them needs only
    # this branch, and it is the same slice polar_metrics interpolates on, so
    # the recomputation is exact rather than an approximation. (At the shipped
    # library point the branch sidecar does this for the whole database; this
    # is that mechanism made a property of the record, so it works at every
    # other point too.)
    if rec["status"] != "unconverged":
        sl = _longest_increasing_run(pol.cl)
        rec["branch"] = {
            "alpha": [float(v) for v in np.asarray(pol.alpha_deg, float)[sl]],
            "cl": [float(v) for v in np.asarray(pol.cl, float)[sl]],
            "cd": [float(v) for v in np.asarray(pol.cd, float)[sl]],
            "cm": [float(v) for v in np.asarray(pol.cm, float)[sl]],
        }
    if rec["status"] != "ok":
        return rec

    # Hard eligibility gates = the stage-3 constraints on the RAW section.
    if rec["tc"] < prob.tc_min:
        rec.update(status="gate_tc",
                   reason=f"t/c {rec['tc']:.3f} < {prob.tc_min}")
    elif abs(rec["cm_at"]) > prob.cm_max:
        rec.update(status="gate_cm",
                   reason=f"|cm| {abs(rec['cm_at']):.3f} > {prob.cm_max}")
    else:
        rec["eligible"] = True
    return rec


# ---------------------------------------------------------------- scoring


def _metric_valid(rec: dict) -> bool:
    """True iff a record carries a full, finite metric set (converged +
    cl_design bracketed). Equivalent to screen_one's status 'ok', but read
    off the RAW metric fields so eligibility can be re-derived at any gate
    from a checkpoint that was screened under different gates."""
    return all(np.isfinite(rec.get(k, float("nan")))
               for k in ("tc", "cd_at", "cm_at", "ldcr", "clmax"))


def _passes_gates(rec: dict, tc_min: float, cm_max: float,
                  floors: dict | None) -> bool:
    """Re-derive eligibility from raw metrics at the given gates + floors."""
    return (_metric_valid(rec)
            and rec["tc"] >= tc_min
            and abs(rec["cm_at"]) <= cm_max
            and _floor_reason(rec, floors) is None)


def _criterion_values(elig: list[dict]) -> dict[str, np.ndarray]:
    """The scored criteria as arrays, in CRITERIA order. |cm| is the scored
    value for the ``cm`` criterion (lower-better), and ``cd_at`` — the drag at
    the design lift — is the value for ``cdcr`` (lower-better)."""
    return {
        "thick": np.array([r["tc"] for r in elig]),
        "clmax": np.array([r["clmax"] for r in elig]),
        "ldmax": np.array([r["ldmax"] for r in elig]),
        "ldcr": np.array([r["ldcr"] for r in elig]),
        "astall": np.array([r["astall"] for r in elig]),
        "cm": np.array([abs(r["cm_at"]) for r in elig]),
        # the same cd the ranking shows as "cd @Cl", now scoreable: on a
        # surface that flies at zero lift it is the ONLY criterion that
        # measures the drag the surface costs (see LOWER_BETTER)
        "cdcr": np.array([r["cd_at"] for r in elig]),
    }


def score_candidates(
    records: list[dict], weights: ScoreWeights, floors: dict | None = None,
    tc_min: float | None = None, cm_max: float | None = None,
    reference: ScoreReference | None = None,
) -> list[dict]:
    """GDP composite scoring over the ELIGIBLE records.

    Adds ``score_<metric>`` (0..100 sub-scores) and ``composite`` to each
    eligible record and returns the eligible records sorted by composite,
    best first (ties broken by name for determinism). |cm| is the scored
    value for the ``cm`` criterion (lower-better).

    ``reference``: a frozen (lo, hi) band per criterion, used INSTEAD of the
    live (min, max) of the eligible set. This is what makes the composite an
    optimisation target rather than a within-run rank — see the module
    docstring. Sub-scores are NOT clipped to [0, 100]: a candidate outside
    the band scores past the end, which is the whole point (clipping would
    flatten the objective exactly where a candidate beats the library, e.g.
    it drops the best-L/D_cruise section s4158 from rank 2 to rank 46). Its
    ``sha`` is stamped on every scored record as ``score_reference``.
    ``reference=None`` is the default and the legacy GDP behaviour:
    normalise across the live eligible set, bit-for-bit as before.

    Eligibility source:
      * ``tc_min``/``cm_max`` given -> eligibility is RE-DERIVED from the raw
        metric fields at those gates (``_passes_gates``). This is what lets a
        single warm checkpoint (screened once at, say, tc_min 0.10) be
        re-ranked at any tighter/looser t/c or |Cm| gate without re-running
        XFOIL — the cached ``eligible`` flag, baked at the ORIGINAL gates,
        is deliberately ignored.
      * both None -> the legacy path: trust the record's cached ``eligible``.

    ``floors`` (optional FLOOR_FIELDS minimums) is applied as an extra filter
    in BOTH paths, BEFORE the min-max normalisation, so the sub-scores span
    only the surviving population. Records are never mutated (the SHA cache
    stays gate-agnostic); ``score_candidates(recs, w)`` reproduces the legacy
    two-gate ranking bit-for-bit.
    """
    w = weights.normalised()
    if tc_min is not None or cm_max is not None:
        lo_tc = 0.0 if tc_min is None else float(tc_min)
        hi_cm = float("inf") if cm_max is None else float(cm_max)
        elig = [r for r in records if _passes_gates(r, lo_tc, hi_cm, floors)]
    else:
        elig = [r for r in records
                if r.get("eligible") and _floor_reason(r, floors) is None]
    if not elig:
        return []

    values = _criterion_values(elig)
    # A CRITERION THE BAND DOES NOT COVER IS NOT SCORED — and a WEIGHT on one
    # is refused, not quietly ignored. The band drops what its population
    # could not separate (``build_screen_reference``), which on a fin is both
    # L/D criteria; the shipped fin preset weights them at zero, so J is
    # unchanged to the bit and the refusal only reaches a user who asked to
    # be ranked on a number that ranks nothing.
    if reference is not None:
        dead = [k for k in CRITERIA if not reference.covers(k)]
        refused = [k for k in dead if w[k] > 0.0]
        if refused:
            raise ValueError(
                "the frozen band cannot score "
                + ", ".join(f"{k} ({reference.why_unbanded(k)})"
                            for k in refused)
                + " — so a weight on it buys nothing. Set those weights to "
                  "zero, or measure a band over a population that separates "
                  "them")
    else:
        dead = []
    scores: dict[str, np.ndarray] = {}
    for key, v in values.items():
        if key in dead:
            continue
        lo, hi = (reference.band(key) if reference is not None
                  else (float(v.min()), float(v.max())))
        span = max(np.finfo(float).eps, hi - lo)
        if key in LOWER_BETTER:
            scores[key] = 100.0 * (hi - v) / span
        else:
            scores[key] = 100.0 * (v - lo) / span

    for i, r in enumerate(elig):
        comp = 0.0
        for key in dead:
            # these dicts are mutated in place and re-scored, so a sub-score
            # left over from a band that DID cover the criterion would read as
            # this band's answer for it
            r.pop(f"score_{key}", None)
        for key in scores:
            r[f"score_{key}"] = float(scores[key][i])
            comp += w[key] * scores[key][i]
        r["composite"] = float(comp)
        # Provenance stamp, cleared on the live-min-max path: score_candidates
        # mutates records in place, so re-scoring the SAME dicts without a
        # reference must not leave a frozen-reference sha attached to
        # population-relative numbers.
        if reference is not None:
            r["score_reference"] = reference.sha
        else:
            r.pop("score_reference", None)
    return sorted(elig, key=lambda r: (-r["composite"], r["name"]))


def build_screen_reference(
    records: list[dict], weights: ScoreWeights, preset: str,
    tc_min: float, cm_max: float, floors: dict | None = None,
    p_lo: float = REF_P_LO, p_hi: float = REF_P_HI,
    source: str = "",
) -> dict:
    """Measure a frozen normalisation band over a screened population.

    Eligibility is RE-DERIVED from the raw metrics at (``tc_min``,
    ``cm_max``, ``floors``) — ``_passes_gates``, not the cached flag — so the
    reference states the gates it belongs to and can be rebuilt from any
    checkpoint. Each criterion's band is the (p_lo, p_hi) percentile pair
    (numpy 'linear' interpolation) over that population.

    Percentiles, not (min, max), because the raw span is a single-outlier
    quantity and the composite divides by it: over the frozen §15 UIUC screen
    (631 eligible) the raw/p2-p98 span ratios are t/c 2.211, cl_max 1.660,
    (L/D)_max 1.719, (L/D)_cr 1.612, alpha_stall 1.754, |cm| 1.020. A ratio
    of 2.211 on t/c means the DECLARED 0.10 thickness weight acts like 0.045
    against the other criteria — the 46%-thick fx79w470a section alone pushes
    86.1% of the library below 20/100 on t/c (57.5% under p2/p98).

    Returns the JSON payload (write it with ``json.dumps``); it is
    ``load_screen_reference``-readable and carries the population SHA.
    """
    elig = [r for r in records if _passes_gates(r, tc_min, cm_max, floors)]
    if not elig:
        raise ValueError("no eligible records to measure a reference over")
    values = _criterion_values(elig)
    if not all(np.all(np.isfinite(v)) for v in values.values()):
        raise ValueError("non-finite criterion value in the eligible set")

    w = weights.normalised()
    gates = {"tc_min": float(tc_min), "cm_max": float(cm_max)}
    names = sorted(str(r["name"]) for r in elig)
    measured = {k: (float(np.percentile(values[k], p_lo)),
                    float(np.percentile(values[k], p_hi)))
                for k in CRITERIA}
    # A CRITERION THIS POPULATION CANNOT SEPARATE HAS NO BAND, and the payload
    # says so instead of carrying (v, v). The case that forced it: a vertical
    # stabiliser is screened at zero design lift, where ldcr = |cl|/cd is 0
    # for every candidate — the band came out (0.0, 0.0), no ScoreReference
    # would load it, and every fin composite run was refused with a message
    # about a missing FROZEN band. ``wing_score.measure_reference`` has always
    # dropped a criterion that was constant across its sample; this is the
    # same rule on the screen, and :func:`score_candidates` refuses a WEIGHT
    # on the dropped criterion rather than scoring it zero.
    bounds = {k: [lo, hi] for k, (lo, hi) in measured.items() if hi > lo}
    unbanded = {
        k: (f"every one of the {len(elig)} eligible sections measured "
            f"{lo:.6g} for it, so there is no band to normalise against")
        for k, (lo, hi) in measured.items() if not hi > lo}
    if not bounds:
        raise ValueError(
            "no criterion varied across the eligible sections, so there is "
            "nothing to normalise against")
    return {
        "version": REFERENCE_VERSION,
        "source": source,
        "preset": preset,
        "weights": w,
        "gates": gates,
        "floors": {k: float(v) for k, v in (floors or {}).items()},
        "percentiles": {"lo": float(p_lo), "hi": float(p_hi),
                        "method": "linear"},
        "n_records": len(elig),
        "directions": {k: ("lower" if k in LOWER_BETTER else "higher")
                       for k in CRITERIA},
        "bounds": bounds,
        # …and, where there is no band, WHY. Read by ``ScoreReference`` (which
        # will not accept an omitted criterion the payload does not account
        # for) and shown by the shells beside the weight nobody can spend.
        "unbanded": unbanded,
        # the raw span stays stated for EVERY criterion, banded or not: it is
        # the measurement, and "all 226 sections measured 0.0" is the evidence
        # for the omission above
        "raw_bounds": {k: [float(values[k].min()), float(values[k].max())]
                       for k in CRITERIA},
        # only where there is a percentile band to inflate against
        "span_inflation": {
            k: float((values[k].max() - values[k].min())
                     / max(np.finfo(float).eps,
                           np.percentile(values[k], p_hi)
                           - np.percentile(values[k], p_lo)))
            for k in bounds},
        "records": names,
        "sha": reference_sha(names, gates, w, bounds),
    }


# ---------------------------------------------------------------- screening


def screen_database(
    files: list[Path],
    prob: AirfoilProblem,
    weights: ScoreWeights,
    workers: int = 6,
    checkpoint: Path | None = None,
    verbose: bool = True,
    floors: dict | None = None,
    progress_cb=None,
    cancel=None,
    trust_cache: bool = False,
    reference: ScoreReference | None = None,
) -> dict:
    """Screen ``files`` (parallel XFOIL, SHA-cached), score, rank.

    Returns {"records": {name: record for the REQUESTED files only},
    "ranked": [scored eligible records, best first], "weights":
    normalised weights}. The checkpoint is a CACHE, not the population:
    it may hold records for files outside ``files`` (a smoke run resumed
    from a full-screen checkpoint, or vice versa), but ranking and the
    min-max normalisation only ever see the requested set, and a cache
    hit is only trusted while the .dat content fingerprint matches.
    Checkpointed (atomically) every 25 completions — a crash costs at
    most 25 XFOIL sweeps, and those land in the SHA cache anyway.

    ``trust_cache``: skip the per-file SHA re-hash for any section already in
    the checkpoint (only truly-new files are screened). The SHA guard exists
    to catch an edited .dat, so this trades that safety for NOT having to read
    every coordinate file — the difference between instant and unusable when
    the database lives on a cloud-evicted volume (each cold read blocks on a
    download). Intended for an IMMUTABLE curated database (the fetched UIUC
    set); default False keeps the fingerprint-checked behaviour bit-for-bit.

    THE OPERATING POINT IS PART OF THE KEY, under both settings. A screened
    record's metrics belong to the section AND to the (Re, Mach, panelling,
    alpha sweep) they were measured at, and neither the SHA nor the name can
    see that: a checkpoint warm at Re 1e6 answered a request at any other
    Reynolds number instantly, with 1e6 numbers, and the caller was told it
    had screened at the point it asked for. A record at a different point is
    therefore a MISS and is re-run. Records with no stamp are read as
    :data:`LEGACY_SCREEN_POINT` (see there).

    One checkpoint file holds ONE point: a re-run replaces the record under
    its name, so pointing two points at one file would have them evict each
    other. Callers pick the path from the point (``api.screen_checkpoint``).

    ``reference``: frozen normalisation band forwarded to ``score_candidates``
    (None = the legacy live min-max, bit-for-bit).
    """
    cache: dict[str, dict] = {}
    if checkpoint and checkpoint.exists():
        cache = _load_json_checkpoint(checkpoint, "screen")
        if verbose and cache:
            print(f"screen checkpoint: {len(cache)} airfoils cached",
                  flush=True)

    want_point = screen_point(prob)

    def at_point(stem: str) -> bool:
        rec = cache.get(stem)
        return rec is not None and record_point(rec) == want_point

    if trust_cache:
        # trust any cached record as-is; only screen files never seen before
        # AT THIS POINT.
        todo = [p for p in files if not at_point(p.stem)]
    else:
        shas = {p.stem: file_sha(p) for p in files}
        todo = [p for p in files
                if not at_point(p.stem)
                or cache.get(p.stem, {}).get("sha") != shas[p.stem]]
    done_since_write = 0
    if todo:
        if verbose and len(todo) < len(files):
            print(f"screen cache: {len(files) - len(todo)} hits, "
                  f"{len(todo)} to run", flush=True)
        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            futs = {ex.submit(screen_one, p.stem, p, prob,
                              max(1, workers) > 1): p for p in todo}
            for i, fut in enumerate(as_completed(futs)):
                rec = fut.result()          # XfoilError propagates
                cache[rec["name"]] = rec
                done_since_write += 1
                if progress_cb is not None:
                    progress_cb(i + 1, len(todo), rec)
                if verbose and (i + 1) % 50 == 0:
                    print(f"screened {i + 1}/{len(todo)}", flush=True)
                if checkpoint and done_since_write >= 25:
                    _atomic_write_json(checkpoint, cache)
                    done_since_write = 0
                if cancel is not None and cancel():
                    # cooperative stop: persist what is screened and rank it
                    if verbose:
                        print(f"screen cancelled at {i + 1}/{len(todo)}",
                              flush=True)
                    break
    if checkpoint and done_since_write:
        _atomic_write_json(checkpoint, cache)

    # ONLY records that belong to this point. A cancelled screen leaves the
    # rest of the database sitting in the checkpoint at its old point, and
    # ranking those beside freshly-screened ones would compare metrics from
    # two different Reynolds numbers in one table.
    records = {p.stem: cache[p.stem] for p in files if at_point(p.stem)}
    # When trusting the checkpoint, its cached ``eligible`` flags were baked at
    # whatever gates that screen ran — re-derive eligibility from the raw
    # metrics at THIS prob's tc_min/cm_max so a warm checkpoint re-ranks
    # correctly under a new gate. A fresh (fingerprint-checked) screen keeps
    # the legacy cached-eligible path bit-for-bit.
    regate = (dict(tc_min=prob.tc_min, cm_max=prob.cm_max)
              if trust_cache else {})
    ranked = score_candidates(list(records.values()), weights,
                              floors=floors, reference=reference, **regate)
    return {"records": records, "ranked": ranked,
            "weights": weights.normalised()}


# ---------------------------------------------------------------- base pick


def pick_base(
    ranked: list[dict], prob_template: AirfoilProblem, verbose: bool = True
) -> dict | None:
    """Walk the ranking, best first, to the first candidate whose CST
    refit yields a usable stage-3 problem.

    A usable base: (a) cst_anchor_from_coords fit succeeds, (b) the
    anchor_box is non-degenerate (AirfoilProblem(anchor=...) constructs),
    and (c) evaluate_airfoil at the (clipped) anchor point is
    solver-feasible with BOTH constraint margins >= 0 — the seed handed to
    stage-3 BO must be a feasible incumbent, or "seeded" would mean
    nothing. Returns {"name", "w_upper", "w_lower", "refit", "rank",
    "raw"} or None if no candidate survives; the refit entry records the
    CST-projection cd so the raw-vs-refit delta is explicit in the report.
    """
    for rank, cand in enumerate(ranked):
        try:
            coords = load_airfoil_dat(cand["path"])
            w_u, w_l = cst_anchor_from_coords(coords, prob_template.n_cst)
            prob = AirfoilProblem(
                re=prob_template.re, mach=prob_template.mach,
                cl_design=prob_template.cl_design,
                tc_min=prob_template.tc_min, cm_max=prob_template.cm_max,
                n_cst=prob_template.n_cst, dz_te=prob_template.dz_te,
                alphas=prob_template.alphas,
                cache_dir=prob_template.cache_dir,
                timeout_s=prob_template.timeout_s,
                n_panel=prob_template.n_panel,
                anchor=(w_u, w_l),
            )
        except ValueError as exc:
            if verbose:
                print(f"pick_base: {cand['name']} rejected ({exc})",
                      flush=True)
            continue
        out = evaluate_airfoil(prob.w0, prob)
        ok = bool(out["feasible"]) and bool(np.all(out.get("g", [-1]) >= 0.0))
        if verbose:
            why = out.get("reason") or "constraint margin < 0"
            tag = "BASE" if ok else f"rejected ({why})"
            print(f"pick_base: #{rank + 1} {cand['name']} "
                  f"composite {cand['composite']:.1f} -> {tag}", flush=True)
        if ok:
            return {
                "name": cand["name"], "rank": rank + 1, "raw": cand,
                "w_upper": [float(v) for v in w_u],
                "w_lower": [float(v) for v in w_l],
                "w0_clipped": bool(
                    not np.allclose(prob.w0, np.concatenate([w_u, w_l]))),
                "refit": {
                    "cd": out["cd"], "cm": out["cm"], "tc": out["tc"],
                    "g": [float(v) for v in out["g"]],
                    "tc_cst": cst_thickness(w_u, w_l),
                    "cd_delta_vs_raw": out["cd"] - cand["cd_at"],
                },
            }
    return None


# ------------------------------------------------- the composite as a TARGET
#
# The screen ranks on six criteria and the shape optimiser then maximises one
# number, so five of the six are discarded the moment the search starts (report
# §15.4). Everything below closes that gap: the SAME weighted composite J that
# ranked the database, evaluated against the FROZEN reference band (not the
# live population), as a constrained-harness objective — which is what makes
# best-so-far monotone and cross-seed comparable (module docstring,
# ScoreReference).
#
# Lives here rather than in experiments/ because the design tool runs it too:
# it is the objective behind ``api.airfoil_run_config(objective="composite")``,
# and ``experiments.airfoil_pipeline`` imports these names so the frozen §16.3
# study and the GUI evaluate ONE function.


def _n_margins(prob: AirfoilProblem) -> int:
    """Number of signed constraint margins ``fg_airfoil``/``evaluate_airfoil``
    return for ``prob`` — the composite objective mirrors this exact channel.

    Two by default (g_tc, g_cm). If the ``r_le_min`` leading-edge-radius gate
    is set on the problem, ``evaluate_airfoil`` appends a third margin and this
    returns 3 so the failure/censored returns keep the right width and the
    pymoo GA is sized correctly."""
    return 2 + (1 if getattr(prob, "r_le_min", None) is not None else 0)


#: Start ``composite_evaluation``'s stall sweep concurrently with the cruise
#: sweep, instead of after it. Both are subprocesses, so the search spends
#: max(cruise, stall) per candidate rather than their sum: 1.41x over three
#: seeds x 48 evaluations of sec8_composite, with the visited points and every
#: objective value bit-identical to the serial path.
#:
#: False restores the strictly serial order. It is here for a caller that is
#: ALREADY saturating the machine — ``screen_database`` runs its own thread
#: pool over candidates, and one more thread per candidate on top of that buys
#: nothing (the cores are busy) while making a wall-clock XFOIL timeout more
#: likely to be contention rather than a property of the section.
PREFETCH_STALL_SWEEP = True

#: How many speculative stall sweeps may be in flight at once, across every
#: thread in this interpreter. A rejected candidate does not wait for the sweep
#: it started, so without a cap the number of live XFOIL subprocesses is set by
#: how fast candidates are refused, not by anything anyone chose. On refusal the
#: sweep is simply not started and the caller runs it in line — the serial path,
#: which is always correct.
#:
#: 4 because the measured throughput ceiling for concurrent XFOIL on this class
#: of machine is ~4x (12 cores, 8 workers = 4.12x, 12 workers = 3.99x): past
#: that a further subprocess buys wall clock from no one and only makes a
#: wall-clock timeout likelier to be contention.
MAX_INFLIGHT_STALL_SWEEPS = 4

_STALL_SLOTS = threading.BoundedSemaphore(MAX_INFLIGHT_STALL_SWEEPS)


def _start_stall_sweep(x: np.ndarray, prob: AirfoilProblem):
    """Fire the wide stall sweep in a side thread; return (coords, joiner).

    ``joiner()`` blocks until the sweep is done and returns its
    ``XfoilPolarResult``, or None if it raised — the caller then runs the
    sweep itself, so a failed prefetch costs time and never correctness.
    The sweep is not started at all when the switch is off, when the
    coordinates cannot be built, when :func:`airfoil.xfoil_precheck` says
    ``evaluate_airfoil`` is going to refuse this design before it ever reaches
    the solver, or when the t/c margin already refuses it: g0 is a property of
    the CST geometry alone, so a section too thin to be feasible is known to
    be unscoreable without asking XFOIL anything. Speculation is only ever
    spent where the answer genuinely needs a polar.

    ``concurrent=True`` is what the sweep is entitled to claim: its
    ``timeout_s`` is being measured against a cruise sweep running beside it,
    so a load-induced empty polar must never be written into the shared cache
    as a property of the geometry (``xfoil_run.run_xfoil_polar``).

    At most :data:`MAX_INFLIGHT_STALL_SWEEPS` sweeps are speculated at once;
    past that the caller gets the serial path.
    """
    from .airfoil import xfoil_precheck
    from .xfoil_run import run_xfoil_polar   # late import: monkeypatchable

    if not PREFETCH_STALL_SWEEP or xfoil_precheck(x, prob) is not None:
        return None, None
    try:
        # THE PROBLEM SPLITS ITS OWN VECTOR (AirfoilProblem.split_weights):
        # a symmetric problem — the fin's — carries the upper surface only and
        # mirrors it, so this idiom read the empty tail of a 4-long vector as
        # a lower surface and cst_coords refused with n_w=0
        w_u, w_l = prob.split_weights(x)
        coords = cst_coords(w_u, w_l, dz_te=prob.dz_te)
        thin = cst_thickness(w_u, w_l, dz_te=prob.dz_te) < prob.tc_min
    except Exception:
        return None, None
    if thin:                         # g0 < 0 on geometry alone: never scoreable
        return coords, None
    if not _STALL_SLOTS.acquire(blocking=False):
        return coords, None          # already saturated: run it in line

    box: dict = {}
    done = threading.Event()

    def work():
        try:
            box["pol"] = run_xfoil_polar(
                coords, prob.re, prob.mach, STALL_ALPHAS,
                timeout_s=prob.timeout_s, cache_dir=prob.cache_dir,
                n_panel=prob.n_panel, concurrent=True)
        except Exception:
            pass                     # a failed prefetch is a cache miss, not a
        finally:                     # failure: the caller re-runs the sweep
            _STALL_SLOTS.release()
            done.set()

    try:
        threading.Thread(target=work, daemon=True).start()
    except Exception:                # cannot start a thread: give the slot back
        _STALL_SLOTS.release()       # and let the caller run the sweep in line
        return coords, None

    def joiner():
        done.wait()
        return box.get("pol")

    return coords, joiner


#: What a RIGHT-CENSORED ``cl_max`` is to the composite. See
#: :func:`composite_evaluation`.
CENSORED_MODES: tuple[str, ...] = ("refuse", "lower_bound")
DEFAULT_CENSORED = "refuse"


def check_censored(mode: str | None) -> str:
    """Validate a censoring mode; ``None`` means the default."""
    m = DEFAULT_CENSORED if mode is None else str(mode)
    if m not in CENSORED_MODES:
        raise ValueError(
            f"unknown censored mode {mode!r}; choices: {CENSORED_MODES}")
    return m


#: what each scalarisation this module maximises is CALLED. Three
#: different numbers have shared the name "score"; the ASF string is
#: the module's own warning, moved onto the record so it does not
#: depend on one page having been read.
COMPOSITE_UNITS = "composite J (0-100)"
COMPOSITE_GOAL_UNITS = "composite J - goal penalty"
COMPOSITE_ASF_UNITS = "ASF scalarisation from the seed (NOT a composite J)"


def composite_evaluation(
    x: np.ndarray, prob: AirfoilProblem, reference: ScoreReference,
    weights: ScoreWeights, censored: str | None = DEFAULT_CENSORED,
) -> dict:
    """One composite evaluation, with its breakdown — ``composite_objective``'s
    working, kept so a shell can report WHY J moved.

    Right-censored ``cl_max``: ``refuse`` or ``lower_bound``
    -------------------------------------------------------
    The stall sweep stops at the last alpha XFOIL converged. When ``cl`` is
    still rising there, ``cl_max`` is not measured — only bounded below. This
    is textbook **right censoring**, and the handling is a choice:

    ``refuse`` (default, legacy, bit-for-bit)
        Treat it as a solver failure: the candidate is not scored. Chosen
        originally because scoring a censored value at face value would pay
        the optimiser to make XFOIL march further up the post-stall branch
        (§15.4) rather than to design a better section.
    ``lower_bound``
        Keep the observation and score it at the bound. **This is sound, not
        a guess**: J is a weighted sum of criteria in which BOTH censored
        quantities — ``clmax`` and ``astall`` — are higher-better
        (:data:`HIGHER_BETTER`), and every other criterion comes from the
        *cruise* sweep, which converged. So the J computed at the bound is a
        true LOWER BOUND on the J the section would score if XFOIL had
        marched further, for ANY non-negative weight vector. The exploit the
        default guards against is also bounded: marching further can only
        raise the score toward its true value, never past it.

    Hutter, Hoos & Leyton-Brown (*BO With Censored Response Data*) measure the
    alternatives directly: imputing censored responses from the model's
    predictive distribution truncated at the bound (Schmee & Hahn) beats both
    **dropping** the censored points and **treating them as uncensored**, with
    dropping the worst of the three. ``refuse`` is the dropping arm.

    The dict carries ``clmax_lower_bound`` so a caller can tell a bound from a
    measurement; in ``refuse`` mode it is only ever False, because a censored
    candidate never reaches the scoring step.

    Returns the ``evaluate_airfoil`` dict (cd/cm/tc/g/... — the -cd problem's
    own fields) extended with ``composite`` (the J that
    ``composite_objective`` returns), ``scores`` (the 0-100 sub-scores, one per
    CRITERIA key), the raw screening metrics (``clmax`` / ``astall`` /
    ``ldmax`` / ``ldcr`` / ``cd_at``)
    and ``clmax_censored``. On any failure the dict carries ``composite``
    None and a ``reason`` — never a partial number under a finished-looking
    name.

    ``score`` / ``f`` ARE THE OBJECTIVE THE OPTIMISER SAW, i.e. J when it
    resolved and ``PENALTY`` when it did not — never the -cd value the
    underlying evaluation also computes (that stays available as ``f_cd``).
    A shell reads ``score`` to draw "the best objective"; handing it -0.0055
    under a run labelled "composite" is precisely the drift this contract
    exists to stop.

    Two XFOIL sweeps, SPECULATIVE
    -----------------------------
    clmax / astall need the wider stall sweep (STALL_ALPHAS = 0..20 deg), which
    is ~+154 % on top of the cruise sweep. The two are INDEPENDENT
    ``run_xfoil_polar`` calls on the same coordinates (the cache key hashes the
    alpha tuple, so merging them would invalidate the shared cruise cache,
    results/xfoil_cache) — so the stall sweep is started in a side thread the
    moment the geometry is known, and joined at the one place its result is
    read. The cruise sweep inside ``evaluate_airfoil`` runs against it.

    It used to be gate-first: fired only after the tc/cm gates passed, so an
    infeasible candidate never paid for it. That saving turned out to be worth
    almost nothing — over three 48-evaluation ``sec8_composite`` runs, 42-48 of
    every 48 candidates reached the stall sweep anyway — while the serialisation
    cost 1.4x on the whole search. What the gate still buys is kept: a rejected
    candidate returns WITHOUT waiting for the speculative sweep, which finishes
    into the cache on a spare core.

    Nothing about the result moves. Same two calls, same arguments, same cache
    keys, so the same J: measured bit-identical (visited points AND every
    objective value) against the serial path over three seeds x 48 evaluations,
    at 1.41x the speed. Set :data:`PREFETCH_STALL_SWEEP` False for the strictly
    serial path.
    """
    from .xfoil_run import run_xfoil_polar   # late import: monkeypatchable

    censored = check_censored(censored)
    x = np.asarray(x, dtype=float)
    coords, stall = _start_stall_sweep(x, prob)
    out = dict(evaluate_airfoil(x, prob))
    # the name of the axis, set on EVERY exit — the returns below
    # include early ones, and a PENALTY sentinel is no more a
    # -c_d than a 0-100 index is
    out["score_units"] = COMPOSITE_UNITS
    out["composite"], out["scores"] = None, {}
    out["clmax_lower_bound"] = False
    out["f_cd"] = out.get("f")            # what the -cd problem would score
    out["score"] = out["f"] = PENALTY     # until J resolves, below
    if not out["feasible"]:
        return out                        # never waits on the speculative sweep
    g = np.asarray(out["g"], dtype=float)
    if not np.all(g >= 0.0):
        # INFEASIBLE: J cannot be scored. The margins are still true, so BO
        # keeps learning the feasible boundary from them.
        out["reason"] = out["reason"] or "infeasible: not scored"
        return out                        # never waits on the speculative sweep

    if coords is None:                    # prefetch disabled / geometry refused
        w_u, w_l = prob.split_weights(x)      # symmetric mirrors; see above
        coords = cst_coords(w_u, w_l, dz_te=prob.dz_te)
    pol = run_xfoil_polar(
        coords, prob.re, prob.mach, prob.alphas,
        timeout_s=prob.timeout_s, cache_dir=prob.cache_dir,
        n_panel=prob.n_panel)                        # cache hit
    pol_stall = stall() if stall is not None else None
    if pol_stall is None:                            # not started, or it raised
        pol_stall = run_xfoil_polar(
            coords, prob.re, prob.mach, STALL_ALPHAS,
            timeout_s=prob.timeout_s, cache_dir=prob.cache_dir,
            n_panel=prob.n_panel)                    # the ~+154% stall sweep
    rec = polar_metrics(pol, coords, prob, pol_stall=pol_stall)
    # ``cd_at`` travels beside the rest because it is now a scored criterion
    # (``cdcr``), and it is the SCREEN's number rather than the -cd problem's
    # ``cd``: the two are the same interpolation but reach the caller by two
    # paths, and a reference point has to be stated in the units the composite
    # actually scored (the same reason ``thick`` is measured twice — see
    # :class:`ScoreGoals`).
    out.update({k: rec.get(k) for k in
                ("clmax", "astall", "ldmax", "ldcr", "cd_at",
                 "clmax_censored")})
    if rec["status"] != "ok":
        out.update(feasible=False, reason=rec["reason"] or rec["status"])
        return out
    if rec["clmax_censored"] and censored == "refuse":
        # A CENSORED clmax is a SOLVER FAILURE for scoring, never a low J:
        # scoring it would reward the optimiser for making XFOIL march the
        # post-stall branch further (§15.4), not for real stall performance.
        out.update(feasible=False,
                   reason="cl_max censored (still rising at the last "
                          "converged alpha) = not scoreable")
        return out
    out["clmax_lower_bound"] = bool(rec["clmax_censored"])

    rec["name"], rec["eligible"], rec["path"] = "candidate", True, ""
    scored = score_candidates([rec], weights, reference=reference)
    if not scored or not np.isfinite(scored[0]["composite"]):
        out.update(feasible=False, reason="non-finite criterion")
        return out
    out["composite"] = float(scored[0]["composite"])
    # only the criteria the band can score: one it does not cover has no
    # sub-score, and a reference point built from this dict must not carry a
    # goal nothing can be measured against (:func:`seed_goals`)
    out["scores"] = {k: float(scored[0][f"score_{k}"]) for k in CRITERIA
                     if f"score_{k}" in scored[0]}
    out["score"] = out["f"] = out["composite"]
    return out


def composite_objective(
    x: np.ndarray, prob: AirfoilProblem, reference: ScoreReference,
    weights: ScoreWeights, censored: str | None = DEFAULT_CENSORED,
) -> tuple[float, np.ndarray]:
    """GDP composite J as a constrained-harness objective: ``(J, margins)``.

    ``censored`` is forwarded to :func:`composite_evaluation`: ``"refuse"``
    (default, legacy) makes a right-censored ``cl_max`` a solver failure;
    ``"lower_bound"`` scores it at its bound, which is a true lower bound on J.

    ``J(x) = sum_i w_i s_i(x)`` with the CRITERIA s_i normalised
    against the FROZEN (lo_i, hi_i) reference band (higher-better; mirrored
    100 (hi - |cm|) / (hi - lo) for the lower-better |cm| term) and the
    caller's weights w_i. Using the frozen band rather than a live min-max is
    the whole point — a per-evaluation min-max would be non-stationary
    (module docstring), so J would not be a fixed function of x.

    Commensurability with the screen: the v_i are read from the SAME XFOIL
    data and the SAME cl-interpolation the screen uses, by reusing
    ``polar_metrics`` on the candidate's coordinates and scoring through
    ``score_candidates`` with the frozen reference — screening values and
    objective values are byte-for-byte the same map.

    Constraint channel (IDENTICAL to the -cd arms)
    ----------------------------------------------
    ``margins`` is exactly ``evaluate_airfoil``'s g — g0 = (t/c - tc_min)/tc_min
    on the CST thickness, g1 = (cm_max - |cm|)/cm_max, plus g2 if r_le_min is
    set. So t/c and |cm| are BOTH scored (in J, via geometric t/c and |cm|) and
    gated (in g, via cst t/c and |cm|) — exactly as GDP did it — while clmax and
    astall are only scored. Reusing ``evaluate_airfoil`` for g guarantees the
    composite arms and the -cd controls share one feasible region.

    Return contract (objective.py / airfoil.py PENALTY convention)
    -------------------------------------------------------------
    * solver/geometry failure (bad geometry, < 3 converged, cl_design not
      bracketed) -> ``(PENALTY, [G_FAIL, ...])`` — same as ``fg_airfoil``.
    * solvable but INFEASIBLE (a gate margin < 0) -> ``(PENALTY, g)`` with the
      TRUE margins: the stall sweep is skipped so J cannot be scored, but BO
      still learns the feasible boundary from real g. Such a point is
      infeasible and can never be the reported best-feasible, so a penalised
      objective there costs nothing.
    * CENSORED clmax after the stall sweep -> ``(PENALTY, [G_FAIL, ...])``
      (``composite_evaluation``, and the reason it is a failure).
    * feasible and fully resolved -> ``(J, g)``.
    """
    n_g = _n_margins(prob)
    out = composite_evaluation(x, prob, reference, weights, censored=censored)
    if out["composite"] is None:
        g = out.get("g")
        if (out.get("feasible") and g is not None
                and not np.all(np.asarray(g, dtype=float) >= 0.0)):
            return PENALTY, np.asarray(g, dtype=float)   # true margins
        return PENALTY, np.full(n_g, G_FAIL)
    return float(out["composite"]), np.asarray(out["g"], dtype=float)


def make_composite_fg(prob: AirfoilProblem, reference: ScoreReference,
                      weights: ScoreWeights,
                      censored: str | None = DEFAULT_CENSORED):
    """Bind ``composite_objective`` to (prob, frozen reference, weights) as a
    single-argument constrained-harness callable ``fg(x) -> (J, g)``.

    ``censored`` selects the right-censored ``cl_max`` policy; the default is
    the legacy refusal, so an unstated caller is unchanged."""
    censored = check_censored(censored)

    def fg(x):
        return composite_objective(x, prob, reference, weights,
                                   censored=censored)

    return fg


# ============================================================ goal composite
#
# The composite is a WEIGHTED SUM, and a weighted sum buys and sells. Measured
# on the frozen sec8_composite case (NACA 2412 anchor, budget 48, seed 0), the
# winner scores J 70.59 against the seed's 59.32 — and gets there by selling
# the criterion the wing actually flies:
#
#     criterion   seed      winner    sub-score change   w    contribution
#     |cm|        0.0464    0.0041    +54.2              0.20  +10.84
#     (L/D)max    102.9     121.9     +21.9              0.15   +3.29
#     t/c         0.120     0.150     +18.0              0.10   +1.80
#     cl_max      1.444     1.528      +9.8              0.20   +1.96
#     L/D cruise   83.9      73.6     -18.9              0.35   -6.62
#     alpha_stall  15.5      14.5      -8.8              0.00   -0.00
#
# i.e. cruise L/D fell 12 % and cd at the design lift ROSE 14 % (0.005957 ->
# 0.006791) while J rose 11 points, because 0.042 of pitching moment is worth
# more to the sum than 10 counts of L/D. That is not a bug in the arithmetic:
# it is what a linear scalarisation IS. The per-unit exchange rate is
# dJ/dv = 100 w / (hi - lo), and over the shipped band that is
#
#     |cm|  256 J per unit      t/c   60.7      cl_max 23.4
#     ldcr    0.642 J per unit  ldmax  0.173    astall  0 (weight 0)
#
# so the composite prices 0.01 of |cm| at four points of cruise L/D, and prices
# the stall angle at nothing at all. A user who typed "0.35 on L/D cruise, the
# biggest weight" did not ask for that and cannot see it in the weights.
#
# This is the standard weighted-sum pathology and it has a standard fix, which
# is what the objective below implements: keep the same J, and add a ONE-SIDED
# penalty on falling short of a reference point (Wierzbicki's reference-point /
# achievement scalarisation; equivalently one-sided goal programming, Charnes &
# Cooper). The reference is normally the SEED — the section the user already
# has — so the search is told "improve the composite, but do not hand me back
# a section that is worse than the one I started from".
#
# Nothing here touches ``composite_objective``: every frozen study, every
# published number and the report's §16.3 stay bit-for-bit what they were.


#: What a shortfall costs, as a multiple of the composite's own exchange rate.
#:
#: **3, and it is MEASURED rather than argued.** It shipped at 10 on the
#: reasoning that the pathological trade had to be refused with margin (giving
#: up 18.9 sub-score points of ldcr at w 0.35 costs 10 x 0.35 x 18.9 = 66 J
#: against the +10.8 J the |cm| gain pays). Two studies then priced it:
#:
#: * the mu sweep (`RESULTS_GOAL_PENALTY_SWEEP.md`, 12 paired seeds,
#:   mu in {0,1,3,10,30,100}) found criteria-sold SATURATING by mu = 3 — every
#:   step from 1 -> 3 onward moves the paired median by exactly 0.00 — while J
#:   kept falling all the way to mu = 100;
#: * the powered head-to-head (`RESULTS_MU_ARM_STUDY.md`, pre-registered in
#:   `PREREG_SESSION46.md` §C, 42 paired seeds) then found mu = 3 **better on
#:   BOTH endpoints**: criteria sold -0.357 (sign 0.0227 / Wilcoxon 0.0106 /
#:   paired t 0.0095) and plain composite J **+1.14 median** (sign 0.0436 /
#:   Wilcoxon 0.0074 / t 0.0272). Runs selling nothing go 19/42 -> 28/42 and
#:   cd at the design lift falls 0.005708 -> 0.005565.
#:
#: The pre-registration predicted a NULL on the primary and a gain on J; the
#: primary came out better than predicted, which is recorded rather than
#: re-described. 10 was not wrong — it is above the knee and conservative — it
#: was simply paying for protection it had already bought at 3.
#:
#: It remains a DEFAULT, not a ban: ``penalty=0`` recovers
#: ``composite_objective`` exactly, and a caller who wants the
#: hard-constrained problem raises it (the exact-penalty limit; the floor is
#: unbreached on every seed only at mu = 100, at -5.50 J).
#:
#: NOTE for anyone reading sessions 44/45: their studies ran the goal arm at
#: **mu = 10**, which was the default then. Their numbers are unaffected and
#: still describe mu = 10; they simply no longer describe the default.
GOAL_PENALTY = 3.0

#: Free slack on every goal, in sub-score points (0-100 scale). 0 = "no
#: criterion may end below where it started"; 2.0 = "a couple of points of
#: noise is not a regression".
GOAL_TOL = 0.0


#: A natural step per criterion — the amount a reader thinks in (a point of
#: t/c, a tenth of cl_max, ten counts of L/D, a degree of stall angle, a
#: hundredth of |Cm|). Only used to state the exchange rate in units someone
#: can compare; nothing scores through it.
CRITERION_STEP = {"thick": 0.01, "clmax": 0.1, "ldmax": 10.0, "ldcr": 10.0,
                  "astall": 1.0, "cm": 0.01,
                  # ten drag counts, which is the unit a section's drag is
                  # read in everywhere else in this package
                  "cdcr": 0.001}


def exchange_rates(weights: ScoreWeights, reference: ScoreReference) -> dict:
    """What each criterion is WORTH to the composite, per natural step.

    ``dJ/dv_k = 100 w_k / (hi_k - lo_k)`` — the weight divided by the width of
    its band. This is the number that decides what the search trades, and it is
    NOT the weight: over the shipped library band the gdp-sweep preset prices

        0.01 of |Cm|            2.56 J        10 counts of L/D at the design Cl
        0.01 of t/c             0.61 J          6.42 J
        0.1 of cl_max           2.34 J        10 counts of (L/D)max  1.73 J
        1 deg of stall angle    0.00 J  (weight 0)

    so a user who put 0.35 — the largest weight — on cruise L/D has in fact
    told the search that four counts of it are worth one hundredth of pitching
    moment. Returns ``{criterion: {"per_unit", "step", "per_step", "weight",
    "span"}}`` in CRITERIA order.
    """
    w = weights.normalised()
    out: dict[str, dict] = {}
    for key in CRITERIA:
        step = float(CRITERION_STEP[key])
        if not reference.covers(key):
            # NAMED, not dropped: a rate table missing a row the weights panel
            # shows is the silence this table exists to break. There is no
            # rate because there is no band — ``per_unit``/``per_step``/
            # ``span`` are None and ``unbanded`` says why.
            out[key] = {"per_unit": None, "step": step, "per_step": None,
                        "weight": float(w[key]), "span": None,
                        "unbanded": reference.why_unbanded(key)}
            continue
        lo, hi = reference.band(key)
        span = max(np.finfo(float).eps, hi - lo)
        per_unit = 100.0 * w[key] / span
        out[key] = {"per_unit": float(per_unit), "step": step,
                    "per_step": float(per_unit * step),
                    "weight": float(w[key]), "span": float(span),
                    "unbanded": ""}
    return out


def step_normalised_bands(reference: ScoreReference, weights: ScoreWeights,
                          anchor: str = "ldcr") -> dict:
    """A band in which the WEIGHTS ARE THE EXCHANGE RATES, exactly.

    The composite trades at ``dJ/dv_k = 100 w_k / (hi_k - lo_k)``, so per
    NATURAL step (:data:`CRITERION_STEP`) it trades at
    ``100 w_k step_k / (hi_k - lo_k)``. Setting every width proportional to its
    own natural step — ``hi_k - lo_k = N step_k`` — collapses that to
    ``100 w_k / N``: **proportional to the weight and to nothing else**. A user
    who writes 0.35 on cruise L/D and 0.20 on |Cm| then gets exactly what they
    wrote, one natural step against one natural step.

    ``N`` is fixed by holding ``anchor``'s width at its current value, so the
    anchored criterion's rate is unchanged and every other criterion is
    rescaled relative to it — the transformation has one degree of freedom and
    this spends it on "keep the biggest weight where it was". Midpoints are
    preserved, so a typical section still scores near the middle of each band.

    This is a band, not a gate: sub-scores remain unclipped, exactly as the
    shipped band leaves them, so a section outside the band still scores past
    the end rather than being refused.

    Returns a plain ``{criterion: [lo, hi]}`` payload, which is what
    ``api.optimize_airfoil(score_reference=...)`` accepts.
    """
    if anchor not in CRITERIA:
        raise ValueError(f"unknown anchor {anchor!r}; "
                         f"choose from {list(CRITERIA)}")
    lo_a, hi_a = reference.band(anchor)
    n = (hi_a - lo_a) / float(CRITERION_STEP[anchor])
    out: dict[str, list[float]] = {}
    for key in CRITERIA:
        lo, hi = reference.band(key)
        mid = 0.5 * (lo + hi)
        half = 0.5 * n * float(CRITERION_STEP[key])
        out[key] = [float(mid - half), float(mid + half)]
    return out


def band_payload(bounds: dict, sha: str, n_records: int = 0,
                 preset: str = "gdp-sweep", **extra) -> dict:
    """Wrap a ``{criterion: [lo, hi]}`` map as a score-reference payload.

    ``sha`` is a LABEL here, not a screen-reference hash: a band that was not
    measured over the library must not be able to masquerade as one, and
    :func:`load_screen_reference` would reject it outright. Naming it after how
    it was built is what keeps a stored run readable.
    """
    ref = ScoreReference(bounds={k: (float(v[0]), float(v[1]))
                                 for k, v in dict(bounds).items()},
                         sha=str(sha), n_records=int(n_records),
                         preset=str(preset))          # validates, may raise
    return {"bounds": {k: [float(v[0]), float(v[1])]
                       for k, v in ref.bounds.items()},
            "sha": ref.sha, "n_records": int(ref.n_records),
            "preset": ref.preset, **extra}


@dataclass(frozen=True)
class ScoreGoals:
    """A per-criterion floor the search should not sell, in RAW metric units.

    ``goals`` maps a subset of :data:`CRITERIA` to the raw value that criterion
    already reaches — t/c, cl_max, (L/D)max, (L/D) at the design lift, the
    stall angle, and |cm| (the MAGNITUDE, as ``_criterion_values`` scores it).
    Raw rather than sub-scores because that is the form a user can state and
    the form the screening floors are already in; the band converts them.

    ``penalty`` prices a shortfall as a multiple of the composite's own rate,
    ``tol`` allows that many sub-score points of slack before a shortfall
    counts, and ``source`` records where the goals came from ("seed", "floors",
    a section name) so a stored run says what it was measured against.

    ``scores`` is the SAME reference point already in sub-score points, and it
    BINDS where it is present — the raw value is then the human-readable label
    beside it. It exists because two of the criteria are measured twice in
    this codebase: ``evaluate_airfoil`` reports the CST thickness (what the t/c
    GATE tests) while the composite scores the GEOMETRIC thickness off the
    coordinates, and the two differ in the fourth decimal. A reference point
    measured off a real evaluation therefore carries the numbers the composite
    itself produced, so a design that merely reproduces the seed scores a
    shortfall of exactly zero rather than of one ten-thousandth.
    """

    goals: dict[str, float]
    penalty: float = GOAL_PENALTY
    tol: float = GOAL_TOL
    source: str = ""
    scores: dict[str, float] | None = None

    def __post_init__(self) -> None:
        extra = set(self.goals) - set(CRITERIA)
        if extra:
            raise ValueError(f"unknown goal criteria {sorted(extra)}; "
                             f"choose from {list(CRITERIA)}")
        for key, val in self.goals.items():
            if not np.isfinite(float(val)):
                raise ValueError(f"non-finite goal for {key}: {val!r}")
        loose = set(self.scores or {}) - set(self.goals)
        if loose:
            raise ValueError(
                f"goal sub-scores {sorted(loose)} name criteria the goals "
                "themselves do not: a reference point states the criterion "
                "once, in raw units, and may restate it in score points")
        for key, val in (self.scores or {}).items():
            if not np.isfinite(float(val)):
                raise ValueError(f"non-finite goal sub-score for {key}: "
                                 f"{val!r}")
        if not np.isfinite(self.penalty) or self.penalty < 0.0:
            raise ValueError(f"penalty must be finite and >= 0, "
                             f"got {self.penalty!r}")
        if not np.isfinite(self.tol) or self.tol < 0.0:
            raise ValueError(f"tol must be finite and >= 0, got {self.tol!r}")

    def score_of(self, key: str, reference: ScoreReference) -> float:
        """The goal for ``key`` in sub-score points: the measured one where the
        reference point carries it, else the raw goal through the band."""
        got = (self.scores or {}).get(key)
        if got is not None:
            return float(got)
        return sub_score(key, self.goals[key], reference)

    def to_dict(self) -> dict:
        out = {"goals": {k: float(v) for k, v in self.goals.items()},
               "penalty": float(self.penalty), "tol": float(self.tol),
               "source": str(self.source)}
        if self.scores:
            out["scores"] = {k: float(v) for k, v in self.scores.items()}
        return out

    @classmethod
    def from_dict(cls, payload: dict) -> "ScoreGoals":
        d = dict(payload or {})
        goals = d.get("goals", d if "penalty" not in d else {})
        scores = d.get("scores")
        return cls(goals={str(k): float(v) for k, v in dict(goals).items()},
                   penalty=float(d.get("penalty", GOAL_PENALTY)),
                   tol=float(d.get("tol", GOAL_TOL)),
                   source=str(d.get("source", "")),
                   scores=(None if scores is None else
                           {str(k): float(v) for k, v in dict(scores).items()}))


def sub_score(key: str, value: float, reference: ScoreReference) -> float:
    """One criterion's 0-100 sub-score under a frozen band.

    The same map ``score_candidates`` applies (including the unclipped ends and
    the mirrored lower-better branch), for ONE value instead of a population —
    so a goal stated in raw units and a candidate's own metric land on one
    scale. ``value`` for ``cm`` is |cm|, as :func:`_criterion_values` scores it.
    """
    lo, hi = reference.band(key)
    span = max(np.finfo(float).eps, hi - lo)
    v = float(value)
    if key in LOWER_BETTER:
        return 100.0 * (hi - v) / span
    return 100.0 * (v - lo) / span


def goal_shortfalls(scores: dict, goals: ScoreGoals, reference: ScoreReference,
                    weights: ScoreWeights) -> dict:
    """Per-criterion shortfall of ``scores`` against ``goals``, and its price.

    Returns ``{"rows": {criterion: {...}}, "penalty": float,
    "worst": criterion | None}``. Each row carries the goal in raw units and in
    sub-score points, the candidate's own sub-score, the ``shortfall`` (0 when
    the criterion is at or above its goal, after ``tol``) and what that
    shortfall costs J.

    The price is ``penalty * w_k * shortfall_k``, summed: the SAME weight the
    composite pays a gain at, times the multiplier, so ``penalty=1`` makes a
    point lost exactly as expensive as a point won and the default makes it ten
    times as expensive. A criterion with weight 0 is priced at 0 — it is not
    in J either, and protecting what the user weighted at nothing would be a
    preference nobody stated.
    """
    w = weights.normalised()
    rows: dict[str, dict] = {}
    total = 0.0
    worst, worst_cost = None, 0.0
    # every WEIGHTED criterion the floor cannot defend, and why — see
    # :func:`asf_terms`. The floor is built from the reference design's own
    # measured criteria, so one the seed never produced has no goal to be
    # held to and drops out in silence.
    dropped: dict[str, str] = {}
    for key in CRITERIA:
        if float(w[key]) <= 0.0:
            continue
        if key not in goals.goals:
            dropped[key] = "the reference design has no value for it"
        else:
            got = scores.get(key)
            if got is None or not np.isfinite(float(got)):
                dropped[key] = "this design produced no value for it"
    for key, raw in goals.goals.items():
        got = scores.get(key)
        if got is None or not np.isfinite(float(got)):
            continue
        goal_s = goals.score_of(key, reference)
        short = max(0.0, (goal_s - goals.tol) - float(got))
        cost = float(goals.penalty) * w[key] * short
        rows[key] = {"goal_raw": float(raw), "goal_score": float(goal_s),
                     "score": float(got), "shortfall": float(short),
                     "weight": float(w[key]), "cost": cost}
        total += cost
        if cost > worst_cost:
            worst, worst_cost = key, cost
    return {"rows": rows, "penalty": float(total), "worst": worst,
            "dropped": dropped}


def seed_goals(prob: AirfoilProblem, reference: ScoreReference,
               weights: ScoreWeights, censored: str | None = DEFAULT_CENSORED,
               penalty: float = GOAL_PENALTY, tol: float = GOAL_TOL,
               source: str = "seed") -> ScoreGoals:
    """The goals the SEED already meets — one composite evaluation of ``w0``.

    The design box is centred on a section the user chose (the screen winner,
    or the NACA anchor), so its own metrics are the honest floor: "what I
    already have". Costs one cruise + one stall XFOIL sweep, both of which the
    surrounding run pays for anyway (``api.optimize_airfoil`` evaluates the
    seed to build its comparison table).

    A seed that cannot be scored — infeasible at its own gates, unconverged, a
    censored cl_max under ``refuse`` — yields EMPTY goals rather than a guess:
    the objective is then the plain composite, and ``source`` says so.

    The reference point keeps the seed's own SUB-SCORES beside its raw metrics
    (see :class:`ScoreGoals`), so the seed scores a shortfall of exactly zero
    and ``J_goal(seed) == J(seed)`` to the bit.
    """
    out = composite_evaluation(np.asarray(prob.w0, dtype=float), prob,
                               reference, weights, censored=censored)
    if out.get("composite") is None:
        return ScoreGoals(goals={}, penalty=penalty, tol=tol,
                          source=f"{source} (not scoreable: "
                                 f"{out.get('reason') or 'unstated'})")
    raw = {"thick": out.get("tc"), "clmax": out.get("clmax"),
           "ldmax": out.get("ldmax"), "ldcr": out.get("ldcr"),
           "astall": out.get("astall"), "cdcr": out.get("cd_at"),
           "cm": None if out.get("cm") is None else abs(float(out["cm"]))}
    # A GOAL NEEDS A BAND: the reference point is stated in raw units and read
    # in sub-score points (``ScoreGoals.score_of``), so a criterion the band
    # cannot score has no floor to be held to — it is not in J either, because
    # a weight on it is refused outright (:func:`score_candidates`).
    goals = {k: float(v) for k, v in raw.items()
             if v is not None and np.isfinite(float(v))
             and reference.covers(k)}
    scores = {k: float(v) for k, v in (out.get("scores") or {}).items()
              if k in goals and v is not None and np.isfinite(float(v))}
    return ScoreGoals(goals=goals, penalty=penalty, tol=tol, source=source,
                      scores=scores)


def goal_evaluation(
    x: np.ndarray, prob: AirfoilProblem, reference: ScoreReference,
    weights: ScoreWeights, goals: ScoreGoals,
    censored: str | None = DEFAULT_CENSORED,
) -> dict:
    """One GOAL-composite evaluation, with its working.

    ``J_goal = J - penalty * sum_k w_k * max(0, s_k(goal) - tol - s_k(x))``

    The first term is :func:`composite_evaluation`'s J, unchanged and still
    reported as ``composite``. The second is a one-sided (hinge) penalty on
    falling short of the reference point — zero for every criterion at or above
    its goal, so a design that dominates the seed scores EXACTLY the composite
    and the two objectives agree wherever the complaint does not arise.

    Why one-sided and not a two-sided distance: a symmetric distance-to-goal
    would also punish a candidate for being BETTER than the seed, which is the
    opposite of the point. Why a penalty and not a hard constraint: an exact
    penalty with a large enough multiplier has the same solution as the
    constrained problem (Bertsekas) while never making the feasible set empty —
    a seed that is already excellent would otherwise refuse every candidate and
    the run would return nothing at all.

    Adds to the ``composite_evaluation`` dict: ``composite_goal`` (the scalar
    maximised), ``goal_penalty`` (what the shortfalls cost), ``goal`` (the
    per-criterion working from :func:`goal_shortfalls`) and ``goals`` (what it
    was measured against). ``score`` / ``f`` ARE ``composite_goal`` — the same
    contract ``composite_evaluation`` documents, for the same reason.
    """
    out = composite_evaluation(x, prob, reference, weights, censored=censored)
    # the name of the axis, set on EVERY exit — the returns below
    # include early ones, and a PENALTY sentinel is no more a
    # -c_d than a 0-100 index is
    out["score_units"] = COMPOSITE_GOAL_UNITS
    out["goals"] = goals.to_dict()
    out["goal"] = {"rows": {}, "penalty": 0.0, "worst": None}
    out["goal_penalty"], out["composite_goal"] = None, None
    if out.get("composite") is None:
        return out                      # refused; score/f are already PENALTY
    work = goal_shortfalls(out.get("scores") or {}, goals, reference, weights)
    out["goal"] = work
    out["goal_penalty"] = float(work["penalty"])
    out["composite_goal"] = float(out["composite"]) - float(work["penalty"])
    out["score"] = out["f"] = out["composite_goal"]
    return out


def goal_objective(
    x: np.ndarray, prob: AirfoilProblem, reference: ScoreReference,
    weights: ScoreWeights, goals: ScoreGoals,
    censored: str | None = DEFAULT_CENSORED,
) -> tuple[float, np.ndarray]:
    """The goal composite as a constrained-harness objective: ``(J_goal, g)``.

    Same constraint channel, same PENALTY convention and the same censoring
    policy as :func:`composite_objective` — the ONLY difference is the scalar,
    so a goal run and a composite run share one feasible region and can be
    compared candidate for candidate.
    """
    n_g = _n_margins(prob)
    out = goal_evaluation(x, prob, reference, weights, goals,
                          censored=censored)
    if out.get("composite_goal") is None:
        g = out.get("g")
        if (out.get("feasible") and g is not None
                and not np.all(np.asarray(g, dtype=float) >= 0.0)):
            return PENALTY, np.asarray(g, dtype=float)   # true margins
        return PENALTY, np.full(n_g, G_FAIL)
    return float(out["composite_goal"]), np.asarray(out["g"], dtype=float)


def make_goal_fg(prob: AirfoilProblem, reference: ScoreReference,
                 weights: ScoreWeights, goals: ScoreGoals,
                 censored: str | None = DEFAULT_CENSORED):
    """Bind :func:`goal_objective` to (prob, band, weights, goals) as a
    single-argument constrained-harness callable ``fg(x) -> (J_goal, g)``."""
    censored = check_censored(censored)

    def fg(x):
        return goal_objective(x, prob, reference, weights, goals,
                              censored=censored)

    return fg


# ============================================================= ASF composite
#
# The goal composite above fixes SUBSTITUTION: it prices the sale. It does not
# fix REACHABILITY, and that is a theorem, not an oversight — a weighted sum
# can only ever return points on the CONVEX HULL of the achievable set, at
# every weight vector, so every non-convex part of the six-criterion front is
# invisible to J and to J_goal alike (LITERATURE_REVIEW_S44.md §1).
#
# The standard fix is a different scalarising function, not a different
# penalty: the augmented Tchebycheff / achievement scalarising function
# (Wierzbicki; Steuer & Choo), which can produce ANY Pareto-optimal point and
# which the direct BO study (Chugh et al., arXiv:1904.05760) reports beating
# the weighted sum on tight budgets, with the reference point critical.
#
#     J_asf(x) = min_k [ w_k (s_k(x) - r_k) ] + rho * sum_k w_k (s_k(x) - r_k)
#
# in MAXIMISING form, because everything in this module maximises and every
# s_k is higher-better on the frozen band. The leading term is the pure
# Tchebycheff achievement — the search is rewarded for lifting its WORST
# criterion relative to the reference point, so it cannot buy |Cm| with cruise
# L/D at all, whatever the exchange rate. The `rho` term is the AUGMENTATION,
# and it is not decoration: without it the max-min is flat along any direction
# that improves a non-binding criterion, so weakly-Pareto points (better on
# one criterion, no worse on none) score identically to the points that
# dominate them. `rho = 0` is reachable and stated (RHO_ASF is the default,
# not the only value) precisely so that degenerate case can be MEASURED
# rather than asserted.
#
# What it shares with the goal composite, deliberately: the same frozen band,
# the same weights, the same `ScoreGoals` reference point resolved once per
# built problem, the same constraint channel, the same censoring policy, the
# same two refusals. The ONLY difference is the scalar.
#
# What is NOT shared: `penalty`. There is no multiplier in an ASF — the min
# does the work a penalty was standing in for — so an ASF run's stored
# reference point carries penalty 0.0, and stating `airfoil_goal_penalty` on
# an ASF run RAISES rather than being quietly ignored.
#
# SCALE. J_asf is NOT J and is not comparable to it: it is a weighted
# sub-score DIFFERENCE against the reference point, so it is ~0 at the seed,
# positive for a design that dominates it, and negative for one that does not.
# Any comparison of an ASF run against a composite run must re-score both
# winners under `composite_objective` — the common currency — which is what
# the arm study does. Nothing here reports J_asf as a composite.
#
# WHAT `w` DOES HERE IS NOT WHAT IT DOES IN J, and the direction is worth
# stating because it is counter-intuitive and it is MEASURED. In the region
# where a design is above the reference point on every criterion — which is
# where an ASF winner normally sits — every term w_k a_k is positive, so the
# min is attained at the SMALLEST product, and a criterion with a large weight
# needs a smaller achievement to clear the same bar. Concretely, on the
# gdp-sweep preset a design +10 sub-score points on all five weighted criteria
# binds at `thick` (w 0.10, product 1.0) while `ldcr` (w 0.35, product 3.5)
# would have to fall below +2.86 points before it could bind at all. Measured
# on the arm study: the ASF's binding criterion is `thick` in most runs, where
# the plain composite's most-sold criterion is `ldcr`. The weighted sum
# overspends the criterion it weights HIGHEST; this function attends to the
# one it weights LOWEST. Neither is what the weights say.
#
# That is not a bug in this implementation — it is the specified form,
# `min_k w_k (s_k - r_k)`, and it is the maximising image of Wierzbicki's
# `max_k lambda_k (f_k - z_k)`. It is a consequence of what `lambda` IS in
# that literature: a SCALING coefficient (normally 1/range), not a preference
# weight. Our sub-scores are already range-normalised by the frozen band, so
# the honest ASF would take lambda = 1 and put the user's preference in the
# REFERENCE POINT, which is where reference-point methods put it. Passing the
# screening weights as lambda is what was asked for and what is measured; the
# lambda = 1 arm is a stated, un-run alternative, not a silent one.


#: The AUGMENTATION weight in the augmented Tchebycheff function. Small
#: enough that the min term decides the ordering wherever two designs differ
#: on their binding criterion, large enough to break the ties the pure
#: Tchebycheff cannot see. 0.05 is the value the ParEGO line of work uses
#: (Knowles 2006 uses 0.05); it is a DEFAULT, not a ban — ``rho=0`` is the
#: pure Tchebycheff and is reachable through ``airfoil_asf_rho``.
RHO_ASF = 0.05

#: What plays the role of Wierzbicki's ``lambda`` — the ASF's SCALING
#: coefficient — in ``min_k lambda_k (s_k - r_k)``.
#:
#: ``"weights"`` (the default, and the configuration the n=42 study measured)
#: passes the screening weights. ``"unit"`` passes 1.
#:
#: The difference is not cosmetic and the study measured its consequence. With
#: ``lambda = w``, above the reference point every term is positive so the min
#: sits at the SMALLEST product, and a heavily weighted criterion needs a
#: smaller achievement to clear the same bar — measured, the ASF was held to
#: `thick` (w 0.10) in 24 of 42 runs and to cruise L/D (w 0.35) in 2, and it
#: returned the seed unchanged in 6. With ``lambda = 1`` every criterion is
#: held to the same number of sub-score points, which is the textbook reading:
#: in that literature ``lambda`` is a scaling coefficient (normally 1/range),
#: and OUR SUB-SCORES ARE ALREADY RANGE-NORMALISED by the frozen band, so the
#: textbook value here is 1 and preference belongs in the REFERENCE POINT.
#:
#: The weights still decide MEMBERSHIP under either setting — a criterion
#: weighted 0 is dropped, not scaled — because "not in J" and "not defended"
#: are one rule, and only the scaling is at issue here.
ASF_LAMBDA_MODES = ("weights", "unit")
DEFAULT_ASF_LAMBDA = "weights"


def check_asf_lambda(mode: str | None) -> str:
    if mode is None:
        return DEFAULT_ASF_LAMBDA
    m = str(mode)
    if m not in ASF_LAMBDA_MODES:
        raise ValueError(f"unknown asf lambda mode {m!r}; "
                         f"choose from {list(ASF_LAMBDA_MODES)}")
    return m


#: What happens to a criterion the WEIGHTS name and the REFERENCE POINT does
#: not carry.
#:
#: The ASF's iteration domain is the reference point, not the weights: every
#: term is ``lambda_k (s_k(x) - r_k)`` and a criterion with no ``r_k`` has no
#: term. So a criterion the seed never produced — an unscoreable metric, or a
#: reference point STATED over a subset (``airfoil_score_goals``) — leaves the
#: objective entirely, while the weights panel goes on showing the number the
#: user set. Unlike the plain and goal composites, where the criterion is still
#: in ``J`` and the weight still buys something, here the weight buys nothing
#: at all: ``J_asf`` is not ``J``.
#:
#: ``"drop"`` (the default, legacy, bit-for-bit, and what every published ASF
#:     run was measured under)
#:     Leave it out, and RECORD it (``dropped``) so a shell can say so.
#: ``"band_floor"``
#:     Re-enter it at the bottom of the frozen band — the value that scores 0
#:     sub-score points, ``lo`` for a higher-better criterion and ``hi`` for
#:     ``|cm|``. The term is then ``lambda_k (s_k(x) + tol)``, so the weight
#:     binds: raising that criterion raises ``J_asf``, monotonically, through
#:     the augmentation ``rho * sum_k`` and through the min wherever the
#:     candidate scores below the band floor.
#:
#: WHAT THE FLOOR IS AND IS NOT. It is the weakest TRUE statement available:
#: "no reference design was measured here, so this criterion is held only to
#: the worst of the library's own spread". It is NOT a substitute reference
#: point — at ordinary sub-scores a floored term sits ~``100 w_k`` above the
#: seed-relative terms, so it will essentially never be the criterion the
#: max-min is held to, and this mode therefore CANNOT make a missing criterion
#: the binding one. What it buys is the gradient: at ``rho`` per unit weight,
#: ``dJ_asf/ds_k = rho * lambda_k``, where "drop" gives exactly zero. A caller
#: who needs a missing criterion to bind must state a reference value for it
#: (``airfoil_score_goals``) — inventing one here would be a preference nobody
#: expressed, which is the same rule the frozen band is frozen for.
#:
#: THE FLOOR NEVER PRE-EMPTS THE FALLBACK. A reference point with NO usable
#: criterion at all already falls back to the plain composite, where every
#: weight binds at gradient ``w_k`` — twenty times the ``rho * w_k`` a floored
#: term buys at the shipped rho. So ``"band_floor"`` applies only where at
#: least one criterion DOES carry a reference value; on a wholly unscoreable
#: seed it changes nothing, and ``asf_terms`` says exactly that in ``dropped``.
ASF_MISSING_MODES = ("drop", "band_floor")
DEFAULT_ASF_MISSING = "drop"


def check_asf_missing(mode: str | None) -> str:
    if mode is None:
        return DEFAULT_ASF_MISSING
    m = str(mode)
    if m not in ASF_MISSING_MODES:
        raise ValueError(f"unknown asf missing-criterion mode {m!r}; "
                         f"choose from {list(ASF_MISSING_MODES)}")
    return m


def band_floor_value(key: str, reference: ScoreReference) -> float:
    """The RAW value that scores exactly 0 sub-score points for ``key``.

    ``lo`` for a higher-better criterion, ``hi`` for a lower-better one — the
    inverse of :func:`sub_score` at 0, so ``sub_score(key,
    band_floor_value(key, ref), ref) == 0.0`` by construction.
    """
    lo, hi = reference.band(key)
    return float(hi if key in LOWER_BETTER else lo)


def check_asf_rho(rho) -> float:
    """The augmentation weight, validated where the caller stated it.

    Negative would invert the augmentation — it would reward being worse on
    the criteria that are not binding — so it is refused rather than clipped.
    """
    r = float(rho)
    if not np.isfinite(r) or r < 0.0:
        raise ValueError(f"asf rho must be finite and >= 0, got {rho!r}")
    return r


def asf_terms(scores: dict, goals: ScoreGoals, reference: ScoreReference,
              weights: ScoreWeights,
              lam: str = DEFAULT_ASF_LAMBDA,
              missing: str = DEFAULT_ASF_MISSING) -> dict:
    """Per-criterion achievement ``lambda_k (s_k(x) - r_k)``, and its aggregates.

    Returns ``{"rows": {criterion: {...}}, "min": float | None,
    "sum": float, "worst": criterion | None}``. ``min`` is the Tchebycheff
    term, ``sum`` the augmentation's, and ``worst`` the criterion the min is
    attained at — the one the search is actually being held to.

    ``tol`` shifts the reference point DOWN by that many sub-score points, the
    same meaning it has on the goal composite: that much slack is not a
    shortfall. A criterion with weight 0 is DROPPED from both aggregates
    rather than entering at ``lambda_k = 0``: a zero-weight term is identically
    0, so it would pin the min at 0 for every candidate and silently turn the
    Tchebycheff function off. That is the same rule the goal composite holds —
    a criterion nobody weighted is not in J and is not defended here either —
    but here it would have broken the objective rather than merely not helped.
    **The weights decide MEMBERSHIP under either ``lam``**; only the scaling
    is at issue (:data:`ASF_LAMBDA_MODES`).

    ``missing`` decides the OTHER way a weighted criterion can fall out — the
    reference point having no value for it (:data:`ASF_MISSING_MODES`).
    ``"drop"`` is the default and the legacy arithmetic. ``"band_floor"``
    re-enters it at the value that scores 0 on the frozen band, so its weight
    binds instead of buying nothing; such a row carries ``floored`` True, and
    the criterion moves from ``dropped`` to ``floored`` so a shell reports what
    happened rather than what would have.
    """
    lam = check_asf_lambda(lam)
    missing = check_asf_missing(missing)
    w = weights.normalised()
    rows: dict[str, dict] = {}
    for key, raw in goals.goals.items():
        got = scores.get(key)
        if got is None or not np.isfinite(float(got)):
            continue
        if float(w[key]) <= 0.0:
            continue
        lam_k = float(w[key]) if lam == "weights" else 1.0
        ref_s = goals.score_of(key, reference) - float(goals.tol)
        term = lam_k * (float(got) - ref_s)
        rows[key] = {"goal_raw": float(raw), "goal_score": float(ref_s),
                     "score": float(got), "weight": float(w[key]),
                     "lambda": lam_k, "term": float(term), "floored": False}

    # ...and a RECORD of every criterion that carries a weight and is NOT in
    # the aggregates, with the reason. Under ``missing="drop"`` the arithmetic
    # above is the whole function — this only writes down what it already did
    # silently — and it matters because a criterion missing from the SEED
    # never enters the loop at all (the iteration domain is the reference
    # point), so a weight the user set stopped binding with nothing on any
    # card saying so.
    dropped: dict[str, str] = {}
    #: weighted, absent from the reference point, and re-entered at the band
    #: floor rather than dropped (``missing="band_floor"`` only)
    floored: dict[str, str] = {}
    # THE FLOOR NEVER PRE-EMPTS THE FALLBACK. With no reference row at all the
    # evaluation falls back to the PLAIN COMPOSITE, where every weight binds at
    # gradient ``w_k`` — twenty times the ``rho * w_k`` a floored term buys at
    # the shipped rho. Flooring an empty reference point would therefore
    # DEFEND THE WEIGHTS LESS than doing nothing, which is the opposite of what
    # this mode is for. So it applies only where there is an achievement to be
    # measured against in the first place, and where there is not, the missing
    # criteria are reported as dropped with the reason.
    can_floor = missing == "band_floor" and bool(rows)
    for key in CRITERIA:
        if float(w[key]) <= 0.0:
            continue
        got = scores.get(key)
        has_value = got is not None and np.isfinite(float(got))
        if key in goals.goals:
            if not has_value:
                dropped[key] = "this design produced no value for it"
            continue
        if can_floor and has_value:
            floored[key] = ("the reference design has no value for it, so it "
                            "is held to the bottom of the frozen band")
            lam_k = float(w[key]) if lam == "weights" else 1.0
            raw = band_floor_value(key, reference)
            ref_s = 0.0 - float(goals.tol)      # sub_score(raw) is 0 by defn
            # appended in CRITERIA order, so two runs that reach the same set
            # reach it in the same order — ``worst`` breaks ties by iteration
            # order, and a reproducible run may not depend on insertion luck
            rows[key] = {"goal_raw": float(raw), "goal_score": float(ref_s),
                         "score": float(got), "weight": float(w[key]),
                         "lambda": lam_k,
                         "term": float(lam_k * (float(got) - ref_s)),
                         "floored": True}
        elif missing == "band_floor" and not rows and has_value:
            dropped[key] = ("the reference design has no value for it, and "
                            "neither does any other criterion — the run falls "
                            "back to the plain composite, where its weight "
                            "binds harder than the band floor would")
        else:
            dropped[key] = "the reference design has no value for it"
    if not rows:
        return {"rows": {}, "min": None, "sum": 0.0, "worst": None,
                "dropped": dropped, "floored": floored}
    worst = min(rows, key=lambda k: rows[k]["term"])
    return {"rows": rows,
            "min": float(rows[worst]["term"]),
            "sum": float(sum(r["term"] for r in rows.values())),
            "worst": worst,
            "dropped": dropped,
            "floored": floored}


def asf_evaluation(
    x: np.ndarray, prob: AirfoilProblem, reference: ScoreReference,
    weights: ScoreWeights, goals: ScoreGoals, rho: float = RHO_ASF,
    censored: str | None = DEFAULT_CENSORED,
    lam: str = DEFAULT_ASF_LAMBDA,
    missing: str = DEFAULT_ASF_MISSING,
) -> dict:
    """One ASF evaluation, with its working.

    ``J_asf = min_k w_k (s_k - r_k) + rho * sum_k w_k (s_k - r_k)``

    The underlying :func:`composite_evaluation` is unchanged and its J is still
    reported as ``composite``, so an ASF run's winner can be re-scored in the
    composite's own currency without a second evaluation.

    Adds ``composite_asf`` (the scalar maximised), ``asf`` (the per-criterion
    working from :func:`asf_terms`), ``asf_rho`` and ``goals``. ``score`` / ``f``
    ARE ``composite_asf`` — the same contract ``composite_evaluation``
    documents, for the same reason.

    A reference point with no scoreable weighted criterion (an unscoreable
    seed) leaves the ASF undefined; the evaluation then falls back to the plain
    composite and SAYS SO in ``asf["fallback"]``, exactly as the goal composite
    falls back to a zero penalty. The fallback is a property of the built
    problem, not of the candidate — the goals are resolved once — so a run
    never changes scale halfway through.

    ``missing`` is forwarded to :func:`asf_terms`: under ``"band_floor"`` a
    weighted criterion the reference point does not carry is held to the bottom
    of the frozen band instead of leaving the objective. It does NOT change the
    fallback above — an empty reference point still scores the plain composite,
    which defends those same weights harder than a floor would
    (:data:`ASF_MISSING_MODES`).
    """
    rho = check_asf_rho(rho)
    lam = check_asf_lambda(lam)
    missing = check_asf_missing(missing)
    out = composite_evaluation(x, prob, reference, weights, censored=censored)
    # the name of the axis, set on EVERY exit — the returns below
    # include early ones, and a PENALTY sentinel is no more a
    # -c_d than a 0-100 index is
    out["score_units"] = COMPOSITE_ASF_UNITS
    out["goals"] = goals.to_dict()
    out["asf_rho"] = float(rho)
    out["asf_lambda"] = str(lam)
    out["asf_missing"] = str(missing)
    out["asf"] = {"rows": {}, "min": None, "sum": 0.0, "worst": None}
    out["composite_asf"] = None
    if out.get("composite") is None:
        return out                      # refused; score/f are already PENALTY
    work = asf_terms(out.get("scores") or {}, goals, reference, weights, lam,
                     missing=missing)
    out["asf"] = work
    if work["min"] is None:
        work["fallback"] = ("no weighted criterion in the reference point: "
                            "scored as the plain composite")
        out["composite_asf"] = float(out["composite"])
    else:
        out["composite_asf"] = (float(work["min"])
                                + float(rho) * float(work["sum"]))
    out["score"] = out["f"] = out["composite_asf"]
    return out


def asf_objective(
    x: np.ndarray, prob: AirfoilProblem, reference: ScoreReference,
    weights: ScoreWeights, goals: ScoreGoals, rho: float = RHO_ASF,
    censored: str | None = DEFAULT_CENSORED,
    lam: str = DEFAULT_ASF_LAMBDA,
    missing: str = DEFAULT_ASF_MISSING,
) -> tuple[float, np.ndarray]:
    """The ASF composite as a constrained-harness objective: ``(J_asf, g)``.

    Same constraint channel, same PENALTY convention and the same censoring
    policy as :func:`composite_objective` and :func:`goal_objective` — the ONLY
    difference is the scalar, so all three share one feasible region and their
    winners can be compared candidate for candidate under the plain composite.
    """
    n_g = _n_margins(prob)
    out = asf_evaluation(x, prob, reference, weights, goals, rho,
                         censored=censored, lam=lam, missing=missing)
    if out.get("composite_asf") is None:
        g = out.get("g")
        if (out.get("feasible") and g is not None
                and not np.all(np.asarray(g, dtype=float) >= 0.0)):
            return PENALTY, np.asarray(g, dtype=float)   # true margins
        return PENALTY, np.full(n_g, G_FAIL)
    return float(out["composite_asf"]), np.asarray(out["g"], dtype=float)


def make_asf_fg(prob: AirfoilProblem, reference: ScoreReference,
                weights: ScoreWeights, goals: ScoreGoals,
                rho: float = RHO_ASF,
                censored: str | None = DEFAULT_CENSORED,
                lam: str = DEFAULT_ASF_LAMBDA,
                missing: str = DEFAULT_ASF_MISSING):
    """Bind :func:`asf_objective` to (prob, band, weights, goals, rho) as a
    single-argument constrained-harness callable ``fg(x) -> (J_asf, g)``."""
    censored = check_censored(censored)
    rho = check_asf_rho(rho)
    lam = check_asf_lambda(lam)
    missing = check_asf_missing(missing)

    def fg(x):
        return asf_objective(x, prob, reference, weights, goals, rho,
                             censored=censored, lam=lam, missing=missing)

    return fg


# ============================================================ the front
#
# Every objective above collapses six criteria to one number before the search
# starts. The scalarisations differ in HOW they collapse — and the ASF above
# fixes the reachability limitation the weighted sum has — but all of them
# answer "which single section?" when the honest question is "which sections
# can I have, and what does each cost?". That is what a Pareto front is, and
# it is what the airfoil MOO literature reports (NSGA-II + XFOIL + CST is a
# whole genre) and what a reader of the report expects to see.
#
# THREE criteria, not six:
#
#   * cruise L/D at the design lift  (ldcr)   -- what the wing actually flies
#   * cl_max                          (clmax) -- the landing end
#   * |Cm|                            (cm)    -- the trim cost
#
# `t/c` stays a CONSTRAINT: it is already gate g0, and the front's constraint
# channel is deliberately the composite's own so that the two share one
# feasible region and their winners compare candidate for candidate.
#
# `alpha_stall` is in NEITHER -- it is not an objective here and it is not a
# gate anywhere. That is the same position it holds under the composite (the
# gdp-sweep preset weights it 0), stated rather than quietly fixed: adding a
# gate for it would move the feasible region and break the comparison this
# front exists to make. Give it a weight, or a gate, as one deliberate change.
#
# `(L/D)max` is left out, and NOT for the reason the plan assumed. The claim
# was that it is nearly collinear with cruise L/D at fixed cl; measured over
# the 631 scoreable sections of the shipped library screen it is
#
#     ldcr ~ ldmax   Pearson +0.506   Spearman +0.458      (r^2 = 0.26)
#
# i.e. moderately correlated, and cruise L/D explains a quarter of its
# variance. It is dropped because a front is read in three dimensions and
# hypervolume degrades badly in more, not because it carries no information.
# (For context from the same screen: ldcr~clmax is +0.017 over the whole
# library but -0.69 over the top 30 by composite score -- a selection effect,
# and a warning against measuring these correlations on a ranked shortlist.)
#
# The objective vector is in SUB-SCORE points on the FROZEN band, not in raw
# units. Three reasons, all load-bearing: the band puts three quantities with
# different units and one inverted sign on one 0-100 scale; a hypervolume is a
# product of ranges, so raw units would make the number mean whatever the
# units happened to be; and a frozen band is the only thing that makes run A's
# hypervolume and run B's the same number at all.

#: the three criteria the front is built over, in the order the objective
#: vector carries them. All HIGHER-BETTER after :func:`sub_score`, which
#: mirrors the lower-better ``cm``.
PARETO_CRITERIA = ("ldcr", "clmax", "cm")

#: …and the 4-objective set, which adds `(L/D)max`. It exists because the
#: reason for leaving `ldmax` out is DIMENSIONALITY, not redundancy: measured
#: over the 631 scoreable library sections `ldcr`~`ldmax` is Pearson +0.506
#: (r^2 = 0.26), so it carries three quarters of its variance independently.
#: Hypervolume degrades badly in more dimensions and a front is read in three,
#: which is why three is the default — but "we checked" and "we assumed" are
#: different claims, and this is what makes the first one available.
PARETO_CRITERIA_4 = ("ldcr", "clmax", "cm", "ldmax")

PARETO_CRITERIA_SETS = {3: PARETO_CRITERIA, 4: PARETO_CRITERIA_4}


def check_pareto_criteria(criteria=None) -> tuple:
    """The objective set for a front, validated where the caller stated it."""
    if criteria is None:
        return PARETO_CRITERIA
    if isinstance(criteria, int):
        if criteria not in PARETO_CRITERIA_SETS:
            raise ValueError(f"no {criteria}-objective front set; "
                             f"choose from {sorted(PARETO_CRITERIA_SETS)}")
        return PARETO_CRITERIA_SETS[criteria]
    got = tuple(str(k) for k in criteria)
    unknown = [k for k in got if k not in CRITERIA]
    if unknown:
        raise ValueError(f"unknown front criteria {unknown}; "
                         f"choose from {list(CRITERIA)}")
    if len(set(got)) != len(got):
        raise ValueError(f"repeated front criterion in {got}")
    if len(got) < 2:
        raise ValueError("a front needs at least two objectives")
    return got


def pareto_evaluation(
    x: np.ndarray, prob: AirfoilProblem, reference: ScoreReference,
    weights: ScoreWeights, censored: str | None = DEFAULT_CENSORED,
    criteria=None,
) -> dict:
    """One MULTI-OBJECTIVE evaluation: the three sub-scores, plus the working.

    Returns the :func:`composite_evaluation` dict extended with ``pareto`` —
    the length-3 objective VECTOR in :data:`PARETO_CRITERIA` order, on the
    frozen band — and ``pareto_criteria``. ``pareto`` is None on any refusal,
    exactly as ``composite`` is.

    ``score`` / ``f`` ARE NOT THE OBJECTIVE HERE, and this is the one place in
    this module where that is true. A multi-objective run has no scalar
    objective; ``score`` stays the plain composite J because that is what a
    front is RANKED by once it exists (see ``optimize.mobo.rank_front``), and
    ranking a front is the job the composite should always have had. Nothing
    in the multi-objective search path reads ``score``; the search reads
    :func:`pareto_objective`, which returns the vector.
    """
    criteria = check_pareto_criteria(criteria)
    out = composite_evaluation(x, prob, reference, weights, censored=censored)
    out["pareto_criteria"] = list(criteria)
    out["pareto"] = None
    if out.get("composite") is None:
        return out
    s = out.get("scores") or {}
    vec = [s.get(k) for k in criteria]
    if any(v is None or not np.isfinite(float(v)) for v in vec):
        return out
    out["pareto"] = [float(v) for v in vec]
    return out


def pareto_objective(
    x: np.ndarray, prob: AirfoilProblem, reference: ScoreReference,
    weights: ScoreWeights, censored: str | None = DEFAULT_CENSORED,
    criteria=None,
) -> tuple[np.ndarray, np.ndarray]:
    """The front's objective as a constrained MULTI-objective callable.

    ``(y, g)`` with ``y`` the three sub-scores (higher-better) and ``g`` the
    composite's own margins — IDENTICAL constraint channel, so the front and
    every scalarised arm share one feasible region.

    The refusal convention is the scalar path's, one component at a time:
    a design the solver could not fly returns ``PENALTY`` in EVERY component,
    which is what ``optimize.mobo`` tests for and what the refusal encoding
    then replaces.
    """
    n_g = _n_margins(prob)
    criteria = check_pareto_criteria(criteria)
    k = len(criteria)
    out = pareto_evaluation(x, prob, reference, weights, censored=censored,
                            criteria=criteria)
    if out.get("pareto") is None:
        g = out.get("g")
        if (out.get("feasible") and g is not None
                and not np.all(np.asarray(g, dtype=float) >= 0.0)):
            return np.full(k, PENALTY), np.asarray(g, dtype=float)
        return np.full(k, PENALTY), np.full(n_g, G_FAIL)
    return (np.asarray(out["pareto"], dtype=float),
            np.asarray(out["g"], dtype=float))


def make_pareto_fg(prob: AirfoilProblem, reference: ScoreReference,
                   weights: ScoreWeights,
                   censored: str | None = DEFAULT_CENSORED,
                   criteria=None):
    """Bind :func:`pareto_objective` as ``fg(x) -> (y_vector, g)``."""
    censored = check_censored(censored)
    criteria = check_pareto_criteria(criteria)

    def fg(x):
        return pareto_objective(x, prob, reference, weights,
                                censored=censored, criteria=criteria)

    return fg


def seed_pareto_point(prob: AirfoilProblem, reference: ScoreReference,
                      weights: ScoreWeights,
                      censored: str | None = DEFAULT_CENSORED,
                      criteria=None):
    """The SEED's own objective vector — the hypervolume reference point.

    The seed exactly, not a nudged-outward nadir: a design contributes volume
    only if it beats the section the user already has on **all three**
    criteria at once, and a run that sweeps zero volume has said something
    true and readable rather than failed. It is the same reference point the
    goal composite floors at and the ASF measures from, for the same reason —
    one project, one answer to "compared with what?".

    ``None`` when the seed cannot be scored, which the caller must handle
    rather than substitute a guess for.
    """
    out = pareto_evaluation(np.asarray(prob.w0, dtype=float), prob, reference,
                            weights, censored=censored, criteria=criteria)
    if out.get("pareto") is None:
        return None
    return np.asarray(out["pareto"], dtype=float)


def pareto_weights(weights: ScoreWeights, criteria=None) -> np.ndarray:
    """The user's weights restricted to the three front criteria and
    RENORMALISED — the vector ``rank_front`` ranks by, so the ranking a user
    sees over a front is their own preference and not a fourth one."""
    criteria = check_pareto_criteria(criteria)
    w = weights.normalised()
    v = np.array([float(w[k]) for k in criteria], dtype=float)
    total = v.sum()
    return v / total if total > 0 else np.full(v.size, 1.0 / v.size)


__all__ = [
    "CENSORED_MODES", "CRITERIA", "CRITERION_STEP", "DEFAULT_CENSORED",
    "FLOOR_FIELDS",
    "GOAL_PENALTY", "GOAL_TOL", "PARETO_CRITERIA", "RHO_ASF",
    "LATE_CRITERION_BANDS", "LD_MAX_ALPHA_BAND", "LEGACY_SCREEN_POINT",
    "MIN_CONVERGED", "PRESETS", "REFERENCE_VERSION", "REF_P_HI", "REF_P_LO",
    "BOX_BAND_PATH", "BOX_BAND_2415_PATH", "load_box_band",
    "SCREEN_REFERENCE_PATH", "STALL_ALPHAS", "ScoreGoals", "ScoreReference",
    "ScoreWeights",
    "anchor_box", "asf_evaluation", "asf_objective", "asf_terms",
    "ASF_LAMBDA_MODES", "DEFAULT_ASF_LAMBDA", "check_asf_lambda",
    "ASF_MISSING_MODES", "DEFAULT_ASF_MISSING", "check_asf_missing",
    "band_floor_value",
    "PARETO_CRITERIA_4", "check_pareto_criteria",
    "band_payload", "step_normalised_bands",
    "build_screen_reference", "check_asf_rho", "check_censored",
    "composite_evaluation",
    "composite_objective", "exchange_rates", "file_sha", "geometric_tc",
    "goal_evaluation", "goal_objective", "goal_shortfalls",
    "load_airfoil_dat", "load_screen_reference", "make_asf_fg",
    "make_composite_fg",
    "make_goal_fg", "make_pareto_fg", "pareto_evaluation", "pareto_objective",
    "pareto_weights", "pick_base", "polar_metrics", "seed_pareto_point",
    "record_point", "reference_sha", "score_candidates", "screen_database",
    "screen_one", "screen_point", "seed_goals", "split_surfaces",
    "sub_score",
]
