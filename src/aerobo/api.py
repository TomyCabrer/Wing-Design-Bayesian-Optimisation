"""GUI-agnostic run layer — the single thin API that BOTH the Streamlit shell
and the pure-API twin test drive.

Architecture rule (plan Phase 6): the GUI is a THIN SHELL with ZERO physics
and ZERO optimiser logic. It builds a RunConfig (a problem choice + a mission
+ flags + an optimiser + a budget/seed) and calls :func:`run` here — exactly
what tests/test_api.py does. Everything below composes the SAME public APIs
the experiment harnesses use (objective/aircraft/hydrofoil/tail/airfoil/
wing_airfoil problems + optimize.bo / optimize.baselines / optimize.constrained
registries); nothing new is reimplemented. No streamlit import lives here — the
module is pure python so the twin test is a faithful stand-in for the Run page.

Two registries drive the shell:

* :data:`PROBLEM_SPECS` — every problem the GUI offers, each with a display
  name, medium (air/water), the design-vector param labels, default bounds, a
  ``slow`` flag, and a ``build(mission_kwargs, flags, bounds_overrides)``
  callable returning a :class:`_BuiltProblem` (problem object + is_constrained
  + the objective/fg callable + the full evaluate() breakdown fn + the
  effective bounds).
* :data:`OPTIMISER_SPECS` — every optimiser, spanning BOTH the unconstrained
  ``optimize.baselines.REGISTRY`` (+ ``optimize.bo.run_bo``) and the
  ``optimize.constrained.REGISTRY_CONSTRAINED``. Each declares whether it
  supports unconstrained and/or constrained problems; :func:`run` picks the
  right registry from the problem's ``is_constrained``.

Heavy imports (torch/botorch/pymoo via the optimisers, and the physics
modules) are done LAZILY inside the builders and inside :func:`run`, so a bare
``import aerobo.api`` (what the GUI's listing pages need) stays cheap.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import re
import time
import uuid
import warnings
from dataclasses import dataclass, field, fields as dc_fields, replace
from functools import lru_cache, wraps
from pathlib import Path
from typing import Any, Callable

import numpy as np

# imported under a second name because ``run`` takes an ``eval_cache``
# ARGUMENT, and a parameter that shadows its own module is how a caller's
# `True` ends up being asked for `.resolve_dir`
from . import eval_cache as eval_cache_mod
from . import fin as _fin
from .objective import MU_SL, RHO_SL

PENALTY = -100.0


# =====================================================================
# JSON sanitisation
# =====================================================================

_DROP = object()   # sentinel: a value that cannot be represented in JSON


def _json_safe(v: Any) -> Any:
    """Recursively convert ``v`` to JSON-serialisable python types.

    numpy scalars/arrays -> python scalars/lists; non-finite floats -> None
    (so ``json.dumps`` produces strict, round-trippable JSON — no
    ``Infinity``/``NaN`` tokens); unknown objects (dataclass instances such as
    Wing/LLTResult carried in an evaluate() breakdown) are DROPPED from the
    enclosing dict rather than raising.
    """
    if v is None or isinstance(v, str):
        return v
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        f = float(v)
        return f if math.isfinite(f) else None
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.ndarray):
        return [_json_safe(e) for e in v.tolist()]
    if isinstance(v, (list, tuple)):
        return [_json_safe(e) for e in v]
    if isinstance(v, dict):
        out = {}
        for k, val in v.items():
            s = _json_safe(val)
            if s is _DROP:
                continue
            out[str(k)] = s
        return out
    return _DROP


def _finite_or_none(v: Any) -> float | None:
    """Float, or None if non-finite / not a number."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


# =====================================================================
# Problem registry
# =====================================================================

# Design-vector labels per objective.Problem mode (geometry.bounds order).
_TRIM_LABELS = ("taper", "twist_root_deg", "twist_tip_deg")
_WINGLET_LABELS = _TRIM_LABELS + ("winglet_h_frac", "winglet_cant_deg")
_MISSION_LABELS = _TRIM_LABELS + ("V_ms", "altitude_m")

# Design-vector labels of the non-objective problem families. Named here (not
# inline in the ProblemSpec) because each one is now shared by the base
# problem and its free-chord-law twin.
_HYDROFOIL_LABELS = _TRIM_LABELS + ("tc", "depth_m", "V_ms")
_HYDROFOIL_WINGLET_LABELS = _HYDROFOIL_LABELS + ("winglet_h_frac",
                                                 "winglet_cant_deg")
#: the car families' vectors. Their tip device's height is a LENGTH and their
#: SPAN is a design variable (carwing.CarWingProblem says why); the span is
#: their size block, so it sits after the family's own rows and ahead of the
#: chord coefficients, exactly where every other size block sits.
_CAR_WING_LABELS = _TRIM_LABELS + ("alpha_deg", "endplate_h_m",
                                   "ride_height_m", "b_m")
#: ...and with the reference AREA freed too. The area row goes AHEAD of the
#: span row (carwing.s_from_x says why: it leaves the span in the same slot
#: whether or not the area is free, so every existing span reader is
#: unchanged).
_CAR_WING_AREA_LABELS = _TRIM_LABELS + ("alpha_deg", "endplate_h_m",
                                        "ride_height_m", "S_m2", "b_m")
#: the TWO-ELEMENT (slotted) rear wing's four extra rows — the flap's size,
#: its deflection and where its leading edge is placed relative to the main
#: element's trailing edge (carwing_multi.slot_rows, in that module's own
#: order).
_CAR_SLOT_LABELS = ("flap_chord_frac", "flap_deflection_deg",
                    "slot_gap_frac", "slot_overlap_frac")
#: ...and the two vectors they build. The slot rows are the LAST of the
#: family block and sit AHEAD of the size block, which is not a style choice:
#: ``carwing.span_from_x`` / ``s_from_x`` read the size rows BACKWARDS from
#: the chord coefficients, so a new row anywhere behind them would silently
#: re-address the span (carwing_multi.SLOT_ROW_OFFSET says the same thing from
#: the other end). The area row still goes ahead of the span row, for the
#: reason ``_CAR_WING_AREA_LABELS`` gives.
_CAR_WING_MULTI_LABELS = (_TRIM_LABELS
                          + ("alpha_deg", "endplate_h_m", "ride_height_m")
                          + _CAR_SLOT_LABELS + ("b_m",))
_CAR_WING_MULTI_AREA_LABELS = (_TRIM_LABELS
                               + ("alpha_deg", "endplate_h_m", "ride_height_m")
                               + _CAR_SLOT_LABELS + ("S_m2", "b_m"))
_CAR_ENDPLATE_LABELS = (_TRIM_LABELS
                        + ("alpha_deg", "endplate_h_m", "ride_height_m",
                           "endplate_chord_ratio", "endplate_tc",
                           "endplate_toe_deg", "b_m"))
#: ...and the same family with its reference AREA freed. Same insertion point
#: and same reason as ``_CAR_WING_AREA_LABELS``: the area row goes ahead of
#: the span row, so ``carwing.span_from_x`` keeps its slot either way. Note
#: that ``b_m`` here is the OVERALL WIDTH (endplate.py's class docstring).
_CAR_ENDPLATE_AREA_LABELS = (_TRIM_LABELS
                             + ("alpha_deg", "endplate_h_m", "ride_height_m",
                                "endplate_chord_ratio", "endplate_tc",
                                "endplate_toe_deg", "S_m2", "b_m"))
_AIRCRAFT_LABELS = _TRIM_LABELS + ("b_m", "S_m2", "tc")
_WING_AIRFOIL_LABELS = _TRIM_LABELS + ("tc_sec", "cl_sec")
_TAIL_LABELS = _TRIM_LABELS + ("S_t_m2",)
#: the block a DESIGNED tail adds (tail.TailProblem.tail_free) — its own
#: planform, in the same order the problem itself lays it out
_DESIGNED_TAIL_LABELS = ("taper_t", "AR_t", "washout_t_deg")

#: chord-law variants. The order is FIXED per spec (cubic) because it changes
#: the design dimension, and a dimension-changing flag would make the static
#: ProblemSpec.param_labels lie — the same rule that gives "tail (fixed arm)"
#: its own spec instead of an l_t flag.
_CHORD_ORDER = 3
_CHORD_LABELS = tuple(f"chord_k{j}" for j in range(1, _CHORD_ORDER + 1))

#: flight-state modifier labels (geometry.FLIGHT_LABELS, restated here so
#: importing this module still costs no physics import). The block sits
#: BETWEEN a family's own variables and the chord coefficients, which is the
#: stacking rule geometry.with_flight_bounds / flight_from_x implement.
_FLIGHT_LABELS = ("V_ms", "altitude_m")


@dataclass
class _BuiltProblem:
    """A concrete, ready-to-optimise problem (the builder's return value).

    Superset of the plan's ``(problem_obj, is_constrained, callable)`` contract:
    also carries the evaluate() breakdown fn, the EFFECTIVE bounds (defaults
    with any overrides applied), the design dimension, the param labels and the
    medium — everything :func:`run` and the GUI need without re-deriving.
    """

    problem: Any
    callable: Callable          # unconstrained: x -> float; constrained: x -> (f, g)
    evaluate: Callable          # x -> full breakdown dict
    bounds: np.ndarray          # (d, 2) effective bounds
    is_constrained: bool
    dim: int
    param_labels: tuple
    medium: str


@dataclass
class ProblemSpec:
    """Static description of one GUI-offered problem + its builder.

    ``flags`` lists the physics flags this problem actually honours (only the
    objective.Problem wing modes read ``mach``/``ground_h_m``/``slipstream``;
    every other problem ignores them), so the shell can offer exactly the
    right toggles instead of guessing from the medium.

    ``default_bounds`` is a lazily-computed ``{param_label: [lo, hi]}`` mapping
    of the DEFAULT design box — the thing the GUI pre-fills its bounds editor
    with and narrows into ``RunConfig.bounds_overrides``. It is derived from
    the SAME build path the harnesses use (``build({}, {}, None).bounds``), so
    it never duplicates ``geometry.bounds``; being a property (not a field) it
    triggers no physics import at ``import aerobo.api`` time — only on first
    access, e.g. when the Mission page renders the editor. Result is cached.
    """

    name: str
    display: str
    medium: str                 # "air" | "water"
    is_constrained: bool
    param_labels: tuple
    slow: bool
    build: Callable             # (mission_kwargs, flags, bounds_overrides) -> _BuiltProblem
    n_constraints: int = 1
    description: str = ""
    flags: tuple = ()           # physics flags honoured (empty => none)
    constraint_labels: tuple = ()   # display names for g_i (empty => g[i])
    uses_mission: bool = False  # builder honours mission_kwargs
    mission_fields: tuple = ()  # subset of (W_N, V, altitude_m, depth_m) honoured
    has_blocks: bool = False    # built problem declares .blocks / .x0
    #                             (enables the "blocks" portfolio optimiser)
    #: flags the BUILDER honours that are deliberately not in ``flags``.
    #: ``flags`` drives the shell's physics toggles, so the airfoil-namespaced
    #: section contract (:data:`AIRFOIL_FLAG_KEYS` and friends) has always been
    #: kept out of it — but they ARE honoured, and :func:`check_flags` must
    #: accept them or it would refuse every study script in the repo. Read by
    #: :func:`accepted_flags`; never shown to the user.
    builder_flags: tuple = ()

    @property
    def default_bounds(self) -> dict:
        """Default box bounds as ``{param_label: [lo, hi]}`` (lazy + cached)."""
        cached = self.__dict__.get("_default_bounds_cache")
        if cached is None:
            built = self.build({}, {}, None)
            cached = {str(lbl): [float(lo), float(hi)]
                      for lbl, (lo, hi) in zip(built.param_labels, built.bounds)}
            self.__dict__["_default_bounds_cache"] = cached
        return cached


#: Rows where a typed low end of ZERO means "as small as this can get" and is
#: read as :data:`sizing.MIN_POSITIVE` instead of travelling as a literal 0.
#:
#: A WHITELIST, deliberately, and a short one. These are the quantities that
#: are a SCALE — the solve divides by them, or derives an aspect ratio or a
#: loading from them — so zero is degenerate for the arithmetic while being a
#: perfectly good question about the design ("how small can this get before it
#: stops working?"). Each was measured at 1e-9 and answers honestly:
#:
#:   ``S_t_m2``   ``untrimmable: |i_t| exceeds limit``
#:   ``b_m``      an aspect ratio outside its band, named
#:   ``S_m2``     the same, from the other side
#:   ``taper``    a bounds violation against the family's validated row
#:
#: Everything NOT here keeps today's behaviour, which is the safe direction
#: for a row nobody has measured. Two kinds of row must never be added:
#:
#: * one where zero is an ordinary interior value, not a limit at all —
#:   ``z_t_m`` is the example, a tail in the wing plane is a layout this repo
#:   deliberately allows, and an epsilon there would answer a question nobody
#:   asked;
#: * one where the solve does not FAIL at zero but lies. Measured at 1e-9,
#:   ``V_ms`` returns ``solver failure: untrimmable: alpha = 9122722`` (the
#:   trim solve blowing up, reported as a finding) and ``depth_m`` returns
#:   **feasible** — a foil flying happily at the free surface it would
#:   ventilate through. Both keep the refusal ``hydrofoil._operating_row``
#:   raises, which says why.
ZERO_MEANS_SMALLEST: frozenset = frozenset(
    {"b_m", "b_rear_m", "S_m2", "S_t_m2", "ws_pa", "taper", "taper_t"})


def _apply_overrides(bounds: np.ndarray, labels: tuple,
                     overrides: dict | None) -> np.ndarray:
    """Return a copy of ``bounds`` with per-parameter [lo, hi] overrides applied.

    Override keys may be integer indices or param-label strings. Intended to
    NARROW a bound (a subset of the default box); the problem's own bounds
    check then never fires spuriously.
    """
    b = np.array(bounds, dtype=float).copy()
    if not overrides:
        return b
    for key, (lo, hi) in overrides.items():
        if isinstance(key, str):
            if key not in labels:
                raise KeyError(f"unknown parameter {key!r}; labels={labels}")
            i = labels.index(key)
        else:
            i = int(key)
        lo_f, hi_f = float(lo), float(hi)
        # "AS SMALL AS THIS CAN GET" is a question the box should be able to
        # ask, and typing 0 into the low end of a SCALE row is how a designer
        # asks it. Read as the smallest positive number rather than passed to
        # a solve that divides by it — for the named rows only
        # (:data:`ZERO_MEANS_SMALLEST`), never inferred from the sign of what
        # was typed: twist_root_deg (-4, 4), twist_tip_deg (-6, 2) and
        # chord_k1..k3 (-0.5, 0.5) all legitimately start below zero, and a
        # rule keyed on "the value is <= 0" would silently rewrite a typed
        # washout of -6 degrees into +1e-9.
        #
        # Resolved to the LABEL, never the key as written: an override may be
        # given by integer index, and a whitelist checked against the key
        # would silently miss it.
        from .sizing import MIN_POSITIVE

        name_of = labels[i] if 0 <= i < len(labels) else None
        if (lo_f <= 0.0 and hi_f > MIN_POSITIVE
                and name_of in ZERO_MEANS_SMALLEST):
            # a NEGATIVE span, area, loading or taper is not a smaller one;
            # it reads as the same request, and answering it the same way
            # beats inventing a refusal these rows have never had
            lo_f = MIN_POSITIVE
        # A zero- (or negative-) width dimension is not a "fixed variable":
        # scipy's Sobol engine raises on it, and constrained BO completes
        # while falling back to random draws on EVERY iteration — i.e. it
        # still reports itself as BO. Fixing a variable must be done by
        # choosing a problem variant with a smaller design vector, never by
        # collapsing a bound, so this is refused up front.
        if not (hi_f > lo_f):
            name = key if isinstance(key, str) else f"index {i}"
            raise ValueError(
                f"bound override for {name} has hi <= lo ({lo_f}, {hi_f}); "
                f"a zero-width dimension breaks the samplers and silently "
                f"degrades BO to random search — state it as "
                f"RunConfig.pinned instead, which takes the variable OUT of "
                f"the design vector rather than collapsing its box")
        b[i, 0] = lo_f
        b[i, 1] = hi_f
    return b


# =====================================================================
# Pinned design variables
# =====================================================================
#
# A variable the user has DECIDED. It is not the same statement as a narrow
# box: a box of width zero is refused above, because scipy's Sobol engine
# raises on it and constrained BO quietly falls back to random draws on every
# iteration while still calling itself BO. So a pin is done the only way that
# is honest — the variable LEAVES the design vector. The optimiser searches
# the free dimensions and never sees the pinned one; the physics is evaluated
# on the full vector with the constant put back.
#
# Everything a run REPORTS stays in the full vector: param_labels, eval_x,
# best_x and the breakdown are the same shape they are without a pin, so every
# consumer downstream (the geometry views, the CAD export, design_report, the
# saved JSON, a later comparison) keeps working unchanged. What says a pin
# happened is ``RunResult.pinned`` and the bounds row, which comes back
# collapsed to [v, v] — the honest picture of the box that was searched.


@dataclass
class _Pin:
    """Which design variables a run holds fixed, and at what value."""

    labels: tuple               # the FULL param labels
    values: np.ndarray          # (d,) the pinned value, NaN where free
    free: tuple                 # indices the optimiser actually searches
    fixed: dict                 # {label: value}, in design-vector order
    bounds: np.ndarray          # the full box, pinned rows collapsed to [v, v]

    @property
    def dim(self) -> int:
        """How many dimensions are left to search."""
        return len(self.free)

    def expand(self, x) -> np.ndarray:
        """A searched vector (or a stack of them) back in FULL coordinates."""
        xa = np.asarray(x, dtype=float)
        if xa.ndim == 1:
            out = self.values.copy()
            out[list(self.free)] = xa
            return out
        out = np.tile(self.values, (xa.shape[0], 1))
        out[:, list(self.free)] = xa
        return out

    def reduce(self, x) -> np.ndarray:
        """A full vector (or a stack) projected onto the searched dimensions."""
        xa = np.asarray(x, dtype=float)
        return xa[..., list(self.free)]


def _pin_of(built: "_BuiltProblem", pinned: dict | None) -> "_Pin | None":
    """Validate ``{label: value}`` against a built problem's own box.

    ``None`` for nothing pinned, which is the bit-for-bit legacy path — no
    wrapper is built and the optimiser sees exactly the problem it always did.

    A pin OUTSIDE the effective box is refused rather than clamped. The box
    is the caller's own statement of where this design lives, and a value
    outside it is two contradictory statements about the same variable; the
    families also validate x against their bounds, so a silent clamp would
    only move the surprise one layer down.
    """
    if not pinned:
        return None
    labels = tuple(str(v) for v in built.param_labels)
    bounds = np.array(built.bounds, dtype=float).copy()
    values = np.full(len(labels), np.nan, dtype=float)
    fixed: dict = {}
    for key, value in pinned.items():
        if isinstance(key, str):
            if key not in labels:
                raise KeyError(f"cannot pin unknown parameter {key!r}; "
                               f"labels={labels}")
            i = labels.index(key)
        else:
            i = int(key)
            if not (0 <= i < len(labels)):
                raise KeyError(f"cannot pin index {i}: the design vector has "
                               f"{len(labels)} entries")
        v = float(value)
        if not math.isfinite(v):
            raise ValueError(f"pinned value for {labels[i]!r} is not finite")
        lo, hi = float(bounds[i, 0]), float(bounds[i, 1])
        if not (lo <= v <= hi):
            raise ValueError(
                f"pinned value {v:g} for {labels[i]!r} is outside the box "
                f"this run searches ({lo:g}–{hi:g}); widen the row or move "
                f"the pin")
        values[i] = v
        bounds[i, 0] = bounds[i, 1] = v
        fixed[labels[i]] = v
    free = tuple(i for i in range(len(labels)) if math.isnan(values[i]))
    if not free:
        raise ValueError(
            "every design variable is pinned, so there is nothing to search; "
            "evaluate the design directly (api.design_report) instead of "
            "running an optimiser over it")
    return _Pin(labels=labels, values=values, free=free,
                fixed={k: fixed[k] for k in labels if k in fixed},
                bounds=bounds)


class _PinnedView:
    """A problem as the REDUCED search sees it.

    Everything of the original except its seeds: ``x0``/``w0`` are the
    family's own warm start in full coordinates, and a search over the free
    dimensions needs them projected onto those (:func:`_bo_x_init` clips a
    seed to the box it is searching and would otherwise be handed a vector of
    the wrong length).
    """

    def __init__(self, inner, pin: "_Pin"):
        self._inner = inner
        self._pin = pin

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def _seed(self, name):
        v = getattr(self._inner, name, None)
        if v is None:
            return None
        return self._pin.reduce(np.asarray(v, dtype=float).ravel())

    @property
    def x0(self):
        return self._seed("x0")

    @property
    def w0(self):
        return self._seed("w0")


def _pinned_built(built: "_BuiltProblem", pin: "_Pin") -> "_BuiltProblem":
    """``built`` as the optimiser sees it: the free dimensions only."""
    return _BuiltProblem(
        problem=_PinnedView(built.problem, pin),
        callable=lambda x: built.callable(pin.expand(x)),
        evaluate=lambda x: built.evaluate(pin.expand(x)),
        bounds=np.array(built.bounds, dtype=float)[list(pin.free), :],
        is_constrained=built.is_constrained,
        dim=pin.dim,
        param_labels=tuple(pin.labels[i] for i in pin.free),
        medium=built.medium,
    )


def _search_problem(cfg: "RunConfig", built: "_BuiltProblem"
                    ) -> tuple["_BuiltProblem", "_Pin | None"]:
    """``(what the optimiser searches, the pin)`` for one run configuration."""
    pin = _pin_of(built, getattr(cfg, "pinned", None))
    if pin is None:
        return built, None
    if cfg.optimiser == "blocks":
        # the blocks portfolio addresses variables by INDEX into the full
        # design vector (problem.blocks), and a pin renumbers them. Refused
        # rather than silently optimising the wrong slice.
        raise ValueError(
            "the 'blocks' optimiser splits the design vector by index, and "
            "pinning a variable renumbers it; run the pinned design with "
            "another optimiser, or drop the pin")
    return _pinned_built(built, pin), pin


#: mission fields a caller may set. ``medium`` is deliberately NOT here: it
#: is a property of the problem, and letting it through would silently give
#: an air problem water density (rho ~ 1025) and a CL_target ~800x too small
#: with no error anywhere.
MISSION_KEYS = ("W_N", "V", "altitude_m", "depth_m")

#: documented floor for the ISA model. mission.isa_temperature has NO lower
#: bound — isa_density(-5000) happily returns 1.93 kg/m^3 — so the guard has
#: to live here.
ALTITUDE_MIN_M = -500.0


def _mission_for(mission_kwargs: dict | None, S: float):
    """Build a MissionSpec from the legacy default, overriding provided fields.

    Starting from :func:`mission.default_mission` means partial
    ``mission_kwargs`` (e.g. only ``W_N``) work: that default reproduces the
    legacy V and CL_target (exactly 0.5) and therefore the legacy SCORE,
    though note its rho/mu come from this package's own ISA evaluation
    (1.2249994633486807 / 1.7892976e-05) rather than the RHO_SL/MU_SL
    literals, so Re_mac shifts by ~1.7e-4 relative once a mission is built.

    Invalid input RAISES rather than returning a penalty: the penalty
    contract covers evaluation failures of a valid configuration, not an
    invalid configuration (same rule as the slipstream flag guard).
    """
    from .mission import H_ISA_MAX_M, default_mission
    spec = default_mission(S=S)
    if not mission_kwargs:
        return spec
    unknown = sorted(set(mission_kwargs) - set(MISSION_KEYS))
    if unknown:
        raise ValueError(f"unknown mission field(s) {unknown}; "
                         f"choices: {list(MISSION_KEYS)}")
    h = mission_kwargs.get("altitude_m")
    if h is not None and not (ALTITUDE_MIN_M <= float(h) <= H_ISA_MAX_M):
        raise ValueError(
            f"altitude {h} m is outside the ISA model's validity "
            f"[{ALTITUDE_MIN_M}, {H_ISA_MAX_M}] m")
    for key in ("W_N", "V"):
        val = mission_kwargs.get(key)
        if val is not None and not float(val) > 0.0:
            raise ValueError(f"mission {key} must be > 0 (got {val})")
    return replace(spec, **mission_kwargs)


def _problem_flags(flags: dict | None) -> dict:
    """Phase-5 physics flags that objective.Problem accepts (all OFF by default).

    Only ``mach`` / ``ground_h_m`` / ``slipstream`` are wing-LLT/VLM flags;
    other problems ignore them (documented — they have no such hook).

    ``slipstream`` accepts a :class:`aerobo.slipstream.SlipstreamSpec` OR a
    plain JSON-safe parameter dict (``D_p``, ``y_centres``, and ``CT`` or
    ``mu_inf`` etc.) which is built into a spec here — that is the form a
    RunConfig round-trips through JSON. A bare ``True`` is rejected with a
    pointer to the dict form: there is no default propeller geometry.
    """
    out: dict = {}
    if not flags:
        return out
    for k in ("mach", "ground_h_m", "slipstream"):
        if k in flags and flags[k] is not None:
            out[k] = flags[k]
    ss = out.get("slipstream")
    if isinstance(ss, bool):
        if ss:
            raise ValueError(
                "flags['slipstream'] needs propeller parameters — pass a dict "
                "like {'D_p': 2.0, 'y_centres': [-2.5, 2.5], 'CT': 0.6} (or a "
                "slipstream.SlipstreamSpec); a bare True has no geometry")
        out.pop("slipstream")
    elif isinstance(ss, dict):
        from .slipstream import SlipstreamSpec
        out["slipstream"] = SlipstreamSpec(**ss)
    return out


def _sref_of(prob) -> float:
    """Reference area the problem's CL_target is defined on.

    Tandem is the trap: it has NO ``S`` attribute at all, only
    ``S_total = 20``. A ``getattr(prob, "S", 10.0)`` would silently halve
    the area, double CL_target and produce a fabricated-but-plausible
    tandem L/D — so the total is checked FIRST and a missing area is an
    error, never a default.
    """
    for attr in ("S_total", "S"):
        val = getattr(prob, attr, None)
        if isinstance(val, (int, float)) and float(val) > 0.0:
            return float(val)
    raise ValueError(f"{type(prob).__name__} exposes no reference area "
                     f"(S_total / S) — cannot apply a design weight")


def _apply_mission(prob, mission_kwargs: dict | None):
    """Apply (W_N, V, altitude_m) to a problem carrying its flow state as
    plain fields (tail / tandem / coupled wing+airfoil / winglet+section)
    rather than as a MissionSpec.

    Empty ``mission_kwargs`` returns the problem UNCHANGED — identity, not
    "rebuild it from the default mission". That is the whole bit-for-bit
    guarantee for these four problems, since the default MissionSpec's rho
    and mu differ from their RHO_SL/MU_SL literals in the 7th digit.
    Whenever a mission IS supplied, all of V, rho, mu and CL_target come
    from that ONE spec, so the reported operating point cannot disagree
    with the loading it implies.
    """
    if not mission_kwargs:
        return prob
    S_ref = _sref_of(prob)
    spec = _mission_for(mission_kwargs, S=S_ref)
    fields = {f.name for f in dc_fields(prob)}
    kw = {"V": spec.V, "CL_target": spec.cl_target(S_ref)}
    if "rho" in fields:
        kw["rho"] = spec.rho
    if "mu" in fields:
        kw["mu"] = spec.mu
    if "mission" in fields:
        # families that carry the flight MODIFIER keep the spec itself, not
        # just the state it implies: the modifier flies THAT design weight at
        # each candidate's (V, altitude), so leaving the old spec in place
        # would fly a different aeroplane from the one the mission card set.
        # Unused (and therefore harmless) when the modifier is off.
        kw["mission"] = spec
    return replace(prob, **{k: v for k, v in kw.items() if k in fields})


#: objective.Problem modes that carry a winglet (and therefore honour a
#: winglet-TYPE cant band). The capped pair is the span-charged variant.
_WINGLET_MODES = ("winglet", "winglet_capped",
                  "winglet_tc", "winglet_capped_tc",
                  "winglet_coupled", "winglet_capped_coupled")


def _winglet_cant_bounds(mode: str, flags: dict | None):
    """Resolve the winglet cant band for a build from ``flags``.

    Accepts either an explicit ``winglet_cant_bounds`` [lo, hi] or a named
    ``winglet_type`` (objective.WINGLET_TYPES: canted / vertical / raked).
    Returns ``None`` (the legacy 60-90 band, bit-for-bit) when neither is
    given or the mode carries no winglet. A ``raked`` type on a FREE-span
    mode is refused: raking is only honest when the span is capped
    (objective.WINGLET_TYPES docstring)."""
    if mode not in _WINGLET_MODES or not flags:
        return None
    explicit = flags.get("winglet_cant_bounds")
    if explicit is not None:
        return (float(explicit[0]), float(explicit[1]))
    wtype = flags.get("winglet_type")
    if wtype is None or wtype == "canted":
        return None
    from . import objective
    if wtype not in objective.WINGLET_TYPES:
        raise ValueError(
            f"unknown winglet_type {wtype!r}; "
            f"choose from {sorted(objective.WINGLET_TYPES)}")
    spec = objective.WINGLET_TYPES[wtype]
    # asked of the MODE's capped-ness, never of a name list. As a literal
    # (`mode in ("winglet", "winglet_tc")`) this guard silently stopped
    # applying to every free-span winglet family added after it was written —
    # the coupled pair walked straight through it and would have reported the
    # fabricated L/D the message describes. Only modes reach here that carry
    # a winglet at all (the early return above), so "not capped" is the whole
    # test. The repo's own "helper matches a problem NAME exactly" bug class.
    if spec["force_capped"] and mode not in objective.CAPPED_MODES:
        raise ValueError(
            f"winglet_type {wtype!r} must be span-capped (raking a free-span "
            f"tip reports a fabricated L/D) — use the span-capped winglet "
            f"problem variant")
    return tuple(spec["cant_bounds"])


#: the winglet's own blend, as a VALUE: how much of the device's arc length
#: is spent turning out of the wing plane. Three registered problems DESIGN
#: this number (objective.BLENDED_MODES read it at x[5]); this flag applies it
#: to a PLAIN winglet problem instead, so "build the tip blended" costs no
#: design variable at all and the vector stays [taper, twist_root, twist_tip,
#: winglet_h_frac, winglet_cant_deg]. Absent = the sharp corner every
#: published winglet run flies, bit-for-bit.
WINGLET_BLEND_KEY = "winglet_blend_frac"

#: flags a fixed-blend winglet problem accepts: the blend itself, the turn law
#: that draws it, how much of the turn the WING does, and whether the corner's
#: interference drag is charged (defaulted ON by the blend — that add-on IS
#: the reason to blend).
_FIXED_BLEND_FLAGS = (WINGLET_BLEND_KEY, "blend_shape", "wing_blend_frac",
                      "junction_drag")

#: does the tip device's CHORD continue the wing's chord distribution?
#: Absent/False = the rectangular device every published run flew (the device
#: holds the wing's tip chord, vlm.py). True samples the wing's own chord law
#: at the device's developed arc instead, so the planform is ONE chord
#: distribution from root to device tip — the straight taper included, which
#: is why this is offered wherever there is a tip device and not only on the
#: chord-law twins.
#:
#: A VALUE, never a design variable: the vector stays [taper, twist_root,
#: twist_tip, winglet_h_frac, winglet_cant_deg] and the same optimiser runs.
WINGLET_CHORD_KEY = "winglet_chord_follows"


def _winglet_chord_kwargs(flags: dict | None) -> dict:
    """:data:`WINGLET_CHORD_KEY` -> Problem kwarg. Empty in, empty out, so a
    family that is never asked builds bit-for-bit as it always did."""
    if not flags or not flags.get(WINGLET_CHORD_KEY):
        return {}
    return {"winglet_chord_follows": True}


def _fixed_blend_kwargs(mode: str, flags: dict | None) -> dict:
    """``winglet_blend_frac`` flag -> Problem kwarg (a VALUE, not a dim).

    Refused on a mode that has no winglet (there is nothing to blend into the
    wing) and on the modes that DESIGN the blend, where the same number lives
    at x[5]: objective.Problem raises for the pair, and asking here as well
    means the registry says so before a run starts.
    """
    if not flags or flags.get(WINGLET_BLEND_KEY) in (None, ""):
        return {}
    value = float(flags[WINGLET_BLEND_KEY])
    if mode not in _WINGLET_MODES:
        raise ValueError(
            f"{WINGLET_BLEND_KEY} needs a winglet to blend into the wing; "
            f"this problem carries none")
    return {"blend_frac_fixed": value}


def _blend_frac_kwargs(flags: dict | None) -> dict:
    """``winglet_blend_frac`` flag -> ``blend_frac_fixed`` kwarg.

    The families whose Problem carries the field under that name and whose
    own ``__post_init__`` does the refusing (the nonplanar wing+tail, the
    tandem pair, the imaged hydrofoil+elevator). Empty in, empty out, so an
    untouched run keeps the sharp corner it always flew.
    """
    if not flags or flags.get(WINGLET_BLEND_KEY) in (None, ""):
        return {}
    return {"blend_frac_fixed": float(flags[WINGLET_BLEND_KEY])}


def _blend_default_on(flags: dict | None, designed: bool) -> bool:
    """Does the junction charge default ON? A blend is the reason to charge
    it, whether that blend is DESIGNED or STATED — and whichever SURFACE it
    was stated for: a blend on the second surface alone still buys a fillet
    that has to be paid for, or it is only a differently drawn wake."""
    if designed:
        return True
    for key in (WINGLET_BLEND_KEY, TAIL_WINGLET_BLEND_KEY):
        try:
            if float((flags or {}).get(key) or 0.0) > 0.0:
                return True
        except (TypeError, ValueError):
            continue
    return False


def _wet_blend_kwargs(flags: dict | None) -> dict:
    """The same fixed-blend flags, in the WATER family's own field names.

    ``hydrofoil.HydrofoilWingletProblem`` carries the blend as
    ``blend_frac`` / ``blend_shape`` / ``wing_blend_frac`` and charges the
    corner through ``junction_drag``. The names differ because the water
    problem has no ``objective.Problem`` to inherit them from; the CONTRACT
    is the same one the air winglet keeps — empty in, empty out, and a blend
    turns the junction charge on unless the caller says otherwise.
    """
    if not flags:
        return {}
    out: dict = {}
    blend = flags.get(WINGLET_BLEND_KEY)
    if blend not in (None, ""):
        out["blend_frac"] = float(blend)
    if flags.get("blend_shape") is not None:
        out["blend_shape"] = str(flags["blend_shape"])
    if flags.get("wing_blend_frac") is not None:
        out["wing_blend_frac"] = float(flags["wing_blend_frac"])
    if flags.get("junction_drag") is not None:
        out["junction_drag"] = bool(flags["junction_drag"])
    elif out.get("blend_frac", 0.0) > 0.0:
        out["junction_drag"] = True
    return out


def _chord_max_frac(flags: dict | None) -> dict:
    """``chord_max_frac`` flag -> Problem kwarg (a VALUE change, not a dim).

    Widening the coefficient box does not add design variables, so it is a
    flag rather than a problem variant — the same status as the winglet cant
    band. geometry.chord_bounds validates the range.
    """
    if not flags or flags.get("chord_max_frac") is None:
        return {}
    return {"chord_max_frac": float(flags["chord_max_frac"])}


#: which SHAPE the chord coefficients describe (geometry.CHORD_LAWS). A
#: physics flag like every other: it changes what the same design vector
#: draws, not how long the vector is, so the family, its labels and every
#: index in it are untouched.
CHORD_LAW_KEY = "chord_law"


def _chord_law_of(flags: dict | None) -> str:
    """The chord law these flags ask for. Unset = the published polynomial."""
    from .geometry import DEFAULT_CHORD_LAW, check_chord_law

    if not flags or not flags.get(CHORD_LAW_KEY):
        return DEFAULT_CHORD_LAW
    return check_chord_law(str(flags[CHORD_LAW_KEY]))


def chord_order_for(chord_order: int, flags: dict | None) -> int:
    """The chord ORDER a law can actually carry.

    A law's order IS the number of trailing design-vector entries it takes
    (geometry.chord_param_count), and the elliptic blend is one number — so
    asking a registered cubic twin for the elliptic law gives a 1-D chord
    block, not a cubic's worth of ellipse. Clamped HERE, once, because
    everything downstream (the box, the labels, the slice) reads the order
    off the built problem and would otherwise disagree about it.
    """
    from .geometry import CHORD_LAW_ORDERS

    n = int(chord_order)
    if n <= 0:
        return n
    return min(n, max(CHORD_LAW_ORDERS[_chord_law_of(flags)]))


def _chord_law_kwargs(flags: dict | None) -> dict:
    """``chord_law`` flag -> problem kwarg. Empty in, empty out."""
    from .geometry import DEFAULT_CHORD_LAW

    law = _chord_law_of(flags)
    return {} if law == DEFAULT_CHORD_LAW else {"chord_law": law}


#: wing SIZE flags. Span and area are configuration VALUES on every
#: fixed-planform problem (the design vector reshapes the planform; it never
#: resizes it), so they travel as flags — the same status as the winglet cant
#: band. Absent means the problem's own published size, bit-for-bit.
#:
#: Deliberately NOT offered on the hydrofoil or the car rear wing: their size
#: is part of a calibrated problem (cavitation margins, ride-height and deck
#: boxes, drag/deflection budgets all quoted on that geometry), so resizing
#: them silently would move calibrations rather than the design. The
#: free-planform aircraft problem does not take them either — there b and S
#: ARE design variables (edit their box bounds instead).
PLANFORM_KEYS = ("b_m", "S_m2")

#: aspect-ratio band the wing solvers are honest over. Below ~3 the
#: lifting-line reduction (llt.py) and the one-chordwise-panel Weissinger VLM
#: stop describing a wing; above ~40 the fixed single-Re section polar and the
#: rigid-planform structure assumptions are the binding lie, not the aero. A
#: size outside it is refused up front rather than scored.
PLANFORM_AR_LIMITS = (3.0, 40.0)


def _planform_size(flags: dict | None, b_default: float,
                   S_default: float) -> tuple[float, float]:
    """(b, S) for a build: the flags where given, the problem's own otherwise.

    Validated together, because it is the PAIR that has to make sense: a
    positive span and a positive area still describe nothing flyable if their
    aspect ratio leaves the band the solvers are valid over.
    """
    b = float(b_default if not flags or flags.get("b_m") is None
              else flags["b_m"])
    S = float(S_default if not flags or flags.get("S_m2") is None
              else flags["S_m2"])
    if not (b > 0.0 and S > 0.0):
        raise ValueError(
            f"wing span and area must both be > 0 (got b = {b}, S = {S})")
    ar = b * b / S
    lo, hi = PLANFORM_AR_LIMITS
    if not (lo <= ar <= hi):
        raise ValueError(
            f"span {b:g} m with area {S:g} m^2 is aspect ratio {ar:.3g}, "
            f"outside the band {PLANFORM_AR_LIMITS} these solvers are valid "
            f"over (lifting-line / one-panel Weissinger below, fixed-Re "
            f"section polar above)")
    return b, S


def _planform_kwargs(flags: dict | None, b_default: float = 10.0,
                     S_default: float = 10.0, area_field: str = "S") -> dict:
    """Problem kwargs for a chosen wing size; ``{}`` when nothing was chosen.

    Empty in, empty out: a problem built without these flags is constructed
    with exactly its published arguments, so no existing result moves.
    """
    if not flags or (flags.get("b_m") is None and flags.get("S_m2") is None):
        return {}
    b, S = _planform_size(flags, b_default, S_default)
    return {"b": b, area_field: S}


#: the WATER families' published size (hydrofoil.HydrofoilProblem's own
#: defaults), so :func:`_planform_kwargs` fills in whichever of the pair the
#: user did not state with the number that family actually flies rather than
#: with the 10 m aircraft wing's.
_WET_SIZE = (1.2, 0.144)


def _wing_span_of(flags: dict | None, b_default: float = 10.0,
                  S_default: float = 10.0) -> float:
    """The SPAN the wing will actually be built at, from the size flags.

    Read before the problem exists, because the two things that are stated
    as a fraction of it — the tail's arm band and the arm a fixed-arm layout
    flies when nobody states one (``tail.arm_band`` / ``tail.default_arm``)
    — are constructor arguments. Unstated size, published span.
    """
    if not flags or (flags.get("b_m") is None and flags.get("S_m2") is None):
        return float(b_default)
    return float(_planform_size(flags, b_default, S_default)[0])


def _blend_shape_flag(flags: dict | None) -> dict:
    """``blend_shape`` / ``wing_blend_frac`` flags -> Problem kwargs.

    Both are VALUE changes, never dimensions. ``blend_shape`` selects HOW a
    blended winglet turns out of the wing plane (geometry.BLEND_SHAPES: the
    constant-radius fillet, the curvature-continuous smoothstep, or the
    clothoid pair); ``wing_blend_frac`` selects how much of that turn the
    WING does, i.e. how far inboard of the tip the transition starts. Each
    changes the shape a given ``winglet_blend_frac`` draws, not the design
    vector, so both are flags — the same status as the winglet cant band.
    Empty in, empty out, so every published blended run stays bit-for-bit on
    the ``arc`` / 0.0 defaults.

    ``wing_blend_frac`` is ignored by the problem that designs it
    ("winglet, blended into the wing", whose vector carries it at x[6]).
    """
    out: dict = {}
    if not flags:
        return out
    if flags.get("blend_shape") is not None:
        out["blend_shape"] = str(flags["blend_shape"])
    if flags.get("wing_blend_frac") is not None:
        out["wing_blend_frac"] = float(flags["wing_blend_frac"])
    return out


def _junction_flag(flags: dict | None, default: bool) -> dict:
    """``junction_drag`` flag -> Problem kwarg (a VALUE change, not a dim).

    Blended-winglet problems default it ON (that add-on IS the reason to
    blend); passing ``junction_drag: false`` shows the pure wake-geometry
    effect, which is a legitimate study and not the default.
    """
    if flags and flags.get("junction_drag") is not None:
        return {"junction_drag": bool(flags["junction_drag"])}
    return {"junction_drag": True} if default else {}


#: the chord distribution's limits, IN THE UNITS AN ENGINEER STATES THEM IN.
#: Each is a VALUE (no design variable moves), each is independently optional,
#: and all of them absent — the default — is every published run.
#:
#: They exist because the design box speaks in chord-law COEFFICIENTS, and
#: nobody knows what ``chord_k2 = -0.31`` draws. The quantities that actually
#: have to be held are a minimum chord (structure, or the section's Reynolds
#: number), a maximum (the mould, the trailer), how fast the chord may change
#: (``chord_rate_max_deg``, the local taper ANGLE) and which end of the wing
#: is the wide one (``chord_trend``, geometry.CHORD_TRENDS).
CHORD_LIMIT_KEYS = ("chord_min_m", "chord_max_m", "chord_rate_max_deg",
                    "chord_trend")

#: the ASPECT-RATIO limit, as flags: the band the user will accept b^2/S in.
#: Each end is independently optional and both absent — the default — is every
#: published run.
#:
#: It is the same KIND of thing as the chord limits above and it is asked
#: beside them: the design box speaks in a span row and an area row, and the
#: quantity a builder, a class rule or a hangar door actually states is the
#: RATIO of the two. A box cannot state it — the aspect ratio of a box is a
#: diagonal across two rows, so the corners of any span x area rectangle span
#: a range of ratios (the shipped 3.6-24 m span against a 0.8-2.2x area band
#: reaches AR 1.3-160) — which is exactly why it has to be a limit and not a
#: bound.
#:
#: NEVER WIDER THAN THE SOLVERS. ``sizing.check_ar`` keeps its own validity
#: band (:data:`PLANFORM_AR_LIMITS`) whatever these say; a user band can only
#: narrow it, and the refusal names which of the two refused.
AR_LIMIT_KEYS = ("ar_min", "ar_max")


def ar_limits_of(flags: dict | None):
    """``(ar_min, ar_max)`` for these flags, or None where they set neither.

    The one place the ``ar_*`` flags become the pair the gate reads, so the
    problem builder below, the shell's box clip and any card drawing the
    limit all ask the same question of the same flags. Either end may be
    None on its own — "at least AR 8" is a limit, and so is "at most 25".
    """
    if not flags:
        return None
    lo = flags.get("ar_min")
    hi = flags.get("ar_max")
    lo = None if lo in (None, "") else float(lo)
    hi = None if hi in (None, "") else float(hi)
    if lo is None and hi is None:
        return None
    if lo is not None and hi is not None and not (hi > lo):
        raise ValueError(
            f"the aspect-ratio limit must have ar_max > ar_min "
            f"(got {lo:.4g} and {hi:.4g})")
    return (lo, hi)


def _ar_limit_kwargs(flags: dict | None) -> dict:
    """``ar_*`` limit flags -> the ``ar_limits`` problem kwarg.

    Empty in, empty out, for the reason :func:`_chord_limits_kwargs` gives:
    with no limit set the constructor call is bit-for-bit the published
    problem's, so a family that gains the limit cannot move its own baseline.
    """
    lim = ar_limits_of(flags)
    return {} if lim is None else {"ar_limits": lim}


#: the one of those four that is not a length — which END of the wing is the
#: wide one. Named because a shell that states a default for it needs to ask
#: whether the family declares it, and a spelling kept in two places is how
#: a flag comes to travel to a builder that would raise on it.
CHORD_TREND_KEY = "chord_trend"


def chord_limits_of(flags: dict | None):
    """The ``ChordLimits`` these flags ask for, or None where they ask none.

    The one place the ``chord_*`` limit flags become the object that judges a
    planform, so that everything reading them — the problem builder below, and
    a shell drawing the band of planforms a design box may propose — asks the
    same question of the same flags. A drawing built from its own reading of
    them is how the picture and the run come to disagree.
    """
    if not flags:
        return None
    sent = {k: flags.get(k) for k in CHORD_LIMIT_KEYS}
    if all(v in (None, "") for v in sent.values()):
        return None
    from .geometry import ChordLimits
    return ChordLimits(
        c_min_m=(None if sent["chord_min_m"] in (None, "")
                 else float(sent["chord_min_m"])),
        c_max_m=(None if sent["chord_max_m"] in (None, "")
                 else float(sent["chord_max_m"])),
        rate_max_deg=(None if sent["chord_rate_max_deg"] in (None, "")
                      else float(sent["chord_rate_max_deg"])),
        trend=str(sent["chord_trend"] or "free"))


def _chord_limits_kwargs(flags: dict | None) -> dict:
    """``chord_*`` limit flags -> the ``chord_limits`` problem kwarg.

    Empty in, empty out: with none of them set the constructor call is
    bit-for-bit the published problem's. A limit that cannot be satisfied is
    NOT caught here — it is caught per candidate, where the planform is
    built, and arrives as the penalty contract (geometry.Wing).
    """
    lim = chord_limits_of(flags)
    out = {} if lim is None else {"chord_limits": lim}
    # ...and the ASPECT-RATIO limit, which travels with them because it is
    # the same kind of statement about the same planform and reaches the same
    # families. Only the families that GATE it are allowed to be sent it
    # (``_declare_ar_limit_flags``), so this cannot arrive somewhere that
    # would accept the flag and then never read it.
    out.update(_ar_limit_kwargs(flags))
    return out


def _chord_kwargs(chord_order: int, flags: dict | None) -> dict:
    """``chord_order`` (+ its value flags) as problem kwargs.

    Empty in / empty out is the point: with no chord law and no limits the
    constructor call is bit-for-bit the published problem's, so a family
    gaining a chord-law twin cannot move its own baseline.

    The LIMITS ride here rather than on the chord-law twin alone, because
    they are limits on the chord DISTRIBUTION, and a straight taper has one
    of those too: "no chord below 0.4 m" is exactly as meaningful on a
    trapezoid (where it constrains the taper) as on a free law.
    """
    limits = _chord_limits_kwargs(flags)
    n = chord_order_for(chord_order, flags)
    if n <= 0:
        return limits
    return {"chord_order": n, **_chord_max_frac(flags),
            **_chord_law_kwargs(flags), **limits}


def _chord_label_block(chord_order: int, flags: dict | None = None) -> tuple:
    """The chord rows a build appends, as labels.

    Built from the ORDER the law can carry, so a build whose law is narrower
    than the registered twin (the elliptic blend on a cubic twin) reports the
    rows it really searches. With no flags this is the registered cubic's own
    block, which is what every static ProblemSpec carries.
    """
    from .geometry import chord_labels

    return chord_labels(chord_order_for(chord_order, flags),
                        _chord_law_of(flags))


def _with_chord_labels(labels: tuple, chord_order: int,
                       flags: dict | None = None) -> tuple:
    """Param labels of a family, extended for a chord-law variant."""
    return tuple(labels) + _chord_label_block(chord_order, flags)


def _flight_kwargs(flight_free: bool) -> dict:
    """``flight_free`` as a problem kwarg; empty when the modifier is off.

    Empty in / empty out, exactly like :func:`_chord_kwargs`: at ``False``
    the constructor call is bit-for-bit the published problem's, so a family
    gaining a flight-state variant cannot move its own baseline.
    """
    return {"flight_free": True} if flight_free else {}


#: the STATED-loading mode's own VALUE flags (MODIFIER_FLAGS["size_ws"]):
#: the loading the area follows, the span band the one row is searched in,
#: and the mission's ceiling. Stated here rather than read off
#: MODIFIER_FLAGS because that table is built further down the module.
WING_LOADING_KEYS = ("wing_loading_pa", "span_min_m", "span_max_m",
                     "wing_loading_limit_pa")

#: every VALUE flag any sized mode can carry — the union of the three
#: MODIFIER_FLAGS entries, which is what a cache key over "things that can
#: move a size box" has to cover.
SIZE_BAND_KEYS = WING_LOADING_KEYS + ("area_min_m2", "area_max_m2",
                                      "ws_min_pa", "ws_max_pa")


def _band(flags: dict, lo_key: str, hi_key: str):
    """``(min, max)`` from a pair of flags, or None if neither was given.

    Giving ONE end is legal — the other travels as None and sizing.py fills
    it from the family's own fractional default — because "no wider than
    15 m" is a complete sentence.
    """
    lo, hi = flags.get(lo_key), flags.get(hi_key)
    if lo is None and hi is None:
        return None
    return (None if lo is None else float(lo),
            None if hi is None else float(hi))


def _wing_loading_kwargs(size_free, flags: dict | None,
                         b_default: float = 10.0) -> dict:
    """The sized modes' VALUE flags -> Problem kwargs.

    Empty in, empty out, in every mode: with no loading given the W/S mode
    opens on the one the problem already flies (CL_target x q), with no band
    each row opens on its own fractional default, and a problem built with
    none of these flags is the published one, bit-for-bit.

    Which flag belongs to which mode is decided HERE and nowhere else:

    * ``span_min_m`` / ``span_max_m`` — every sized mode (all three put a
      span in the vector);
    * ``area_min_m2`` / ``area_max_m2`` — the free planform, whose second row
      is the area;
    * ``wing_loading_pa`` — the STATED-loading mode, whose area follows it;
    * ``ws_min_pa`` / ``ws_max_pa`` — the SEARCHED-loading mode's own row;
    * ``wing_loading_limit_pa`` — every sized mode: it is the mission's
      ceiling, not a band end (sizing.check_wing_loading).

    A band end left out travels as None and is filled where the DEFAULT
    lives — sizing.py, against the family's own reference — rather than here
    against a guessed one: ``b_default`` survives only for the callers that
    already pass it.
    """
    from .sizing import (SIZE_MODE_FREE, SIZE_MODE_WS, SIZE_MODE_WS_FREE,
                         size_mode)

    if not size_free or not flags:
        return {}
    mode = size_mode(size_free)
    out: dict = {}
    if mode == SIZE_MODE_WS and flags.get("wing_loading_pa") is not None:
        out["wing_loading_Pa"] = float(flags["wing_loading_pa"])
    band = _band(flags, "span_min_m", "span_max_m")
    if band is not None:
        out["span_bounds_m"] = band
    if mode == SIZE_MODE_FREE:
        band = _band(flags, "area_min_m2", "area_max_m2")
        if band is not None:
            out["area_bounds_m2"] = band
    if mode == SIZE_MODE_WS_FREE:
        band = _band(flags, "ws_min_pa", "ws_max_pa")
        if band is not None:
            out["ws_bounds_pa"] = band
    if flags.get("wing_loading_limit_pa") is not None:
        out["wing_loading_max_Pa"] = float(flags["wing_loading_limit_pa"])
    return out


def _size_kwargs(size_free, flags: dict | None = None) -> dict:
    """``size_free`` as a problem kwarg; empty when the modifier is off.

    The VALUE travels, not a bool: ``True`` is the two-variable mode and
    ``sizing.SIZE_MODE_WS`` the wing-loading one, and one place — sizing.py's
    ``size_mode`` — decides what each means.

    The MATERIAL rides here for one reason: this helper is called at every
    sized builder and nowhere else, so "what is it built of" reaches exactly
    the families that weigh a wing. A fixed-size family computes no weight
    and no stress at all, which is why the flag is not offered there and why
    :func:`check_flags` refuses it there (MODIFIER_FLAGS).
    """
    if not size_free:
        return {}
    return {"size_free": size_free, **_material_kwargs(flags)}


#: WHAT THE WING IS BUILT OF, as flags. ``material`` names a table entry
#: (``materials.MATERIALS``); the three numbers describe one the table does
#: not have. Given together the numbers WIN — typing a density is a more
#: specific statement than picking a preset, and a control that silently
#: preferred the preset would leave a typed number on screen doing nothing.
MATERIAL_KEY = "material"
MATERIAL_RHO_KEY = "material_rho_kgm3"
MATERIAL_SIGMA_KEY = "material_sigma_allow_mpa"
MATERIAL_SKIN_RHO_KEY = "material_skin_rho_kgm3"
MATERIAL_FLAG_KEYS = (MATERIAL_KEY, MATERIAL_RHO_KEY, MATERIAL_SIGMA_KEY,
                      MATERIAL_SKIN_RHO_KEY)


def _material_kwargs(flags: dict | None) -> dict:
    """Material flags -> the ``material`` problem kwarg. Empty in, empty out.

    Empty out means ``None``, and ``None`` means 2024-T3 aluminium — the
    material Raymer's correlation was regressed over — so a problem built
    with no material flag is the published one, bit for bit.

    A density WITHOUT an allowable (or the other way round) is refused rather
    than half-applied: they are two halves of one answer, and a material with
    a made-up second half is exactly the silent-and-wrong the flag checker
    exists to stop.
    """
    from . import materials

    if not flags:
        return {}
    rho = flags.get(MATERIAL_RHO_KEY)
    sig = flags.get(MATERIAL_SIGMA_KEY)
    if rho is not None or sig is not None:
        if rho is None or sig is None:
            raise ValueError(
                f"a custom material needs BOTH {MATERIAL_RHO_KEY} [kg/m^3] "
                f"and {MATERIAL_SIGMA_KEY} [MPa] — got "
                f"rho={rho!r}, sigma={sig!r}. One without the other would "
                f"be half a material.")
        skin = flags.get(MATERIAL_SKIN_RHO_KEY)
        return {"material": materials.custom(
            float(rho), float(sig) * 1e6,
            None if skin is None else float(skin))}
    key = flags.get(MATERIAL_KEY)
    if key is None:
        return {}
    return {"material": materials.get(str(key))}


def _variant_labels(labels: tuple, chord_order: int = 0,
                    flight_free: bool = False,
                    size_free: bool = False,
                    flags: dict | None = None) -> tuple:
    """Family labels + the modifier blocks, in the package's stacking order.

    [family][flight][chord] — the same order geometry.py appends the rows in,
    so a variant's static ProblemSpec.param_labels lines up with the box its
    builder returns. (Two families override this because their modifier rows
    are not trailing: the CST section problems keep the section weights last,
    and the tandem pair carries one chord law PER WING.)
    """
    from .sizing import size_labels
    return (tuple(labels)
            + tuple(size_labels(size_free))
            + (_FLIGHT_LABELS if flight_free else ())
            + _chord_label_block(chord_order, flags))


def _tandem_labels(chord_order: int = 0, flight_free: bool = False,
                   size_free: bool = False, tc_free: bool = False,
                   flags: dict | None = None) -> tuple:
    """Tandem labels: size (A SPAN PER WING + the pair's TOTAL area), the
    flight block, then BOTH wings' chord laws.

    Two span rows, because the pair has two wings and they need not be the
    same width (sizing.span_labels). Everything else about the block is the
    package's: spans first, area last, size ahead of flight ahead of chord.
    """
    from .sizing import size_labels
    return (_TANDEM_LABELS
            + (("tc_front", "tc_rear") if tc_free else ())
            + tuple(size_labels(size_free, n_spans=2))
            + (_FLIGHT_LABELS if flight_free else ())
            + tuple(lbl.replace("chord_", f"chord_{side}_", 1)
                    for side in ("front", "rear")
                    for lbl in _chord_label_block(chord_order, flags)))


#: fly a CHOSEN section instead of the family's default table polar.
#: The value is the section's name in the UIUC database ("hg40"), or that name
#: WITH THE POINT it was chosen at ({"name": "hg40", "re": 3e5}), or a designed
#: section's CST weights — see :func:`section_polar_for`. All three are
#: JSON-safe, which is the point: a polar OBJECT could not round-trip through a
#: RunConfig, and rebuilding it from the name is instant off the screening
#: sidecar (:func:`library_section_polar`) whenever the point is the cached one.
SECTION_KEY = "section_name"

#: the SECOND surface's own section, where a family has one (the tail /
#: elevator, or a tandem pair's rear wing). Same value shape as SECTION_KEY.
SECTION_AFT_KEY = "section_name_aft"

#: ...and the CAR ENDPLATE's own section. Same value shape again, and a third
#: slot rather than a reuse of the aft key because the plate is a third
#: surface, not the second one: the designed-endplate family's second surface
#: IS the plate, and its first is the wing, so a car carrying both a chosen
#: wing section and a chosen plate section needs two keys that cannot collide.
#:
#: WHY THE PLATE GETS ONE AT ALL. Its ``section`` flag (``endplate.SECTIONS``)
#: names a CONSTRUCTION FAMILY — flat, rounded, shaped — which sets a drag law
#: and a stiffness law and knows nothing about a shape. That is enough to
#: choose between a bent sheet and a moulded strut and not enough to design
#: anything: every aerofoil of a given thickness scores identically under it.
#: With this key the plate carries a real section, flown at the PLATE's own
#: Reynolds number, and the drag bucket that ``endplate.py``'s own docstring
#: records as the missing piece is no longer assumed.
#:
#: THE SECTION MUST BE SYMMETRIC — ``endplate.CarWingEndplateProblem`` refuses
#: a cambered one, because a vertical panel built at theta = twist - alpha_L0
#: carries a side force at zero toe. ``symmetric_section_names()`` is the
#: library subset a shell may offer, and a DESIGNED plate section is optimised
#: with ``optimize_airfoil(symmetric=True, ...)`` — the same pair of rules the
#: fin already obeys.
SECTION_PLATE_KEY = "section_name_plate"

#: objective.Problem modes that pick their section BY THICKNESS off the NACA
#: 24XX family, so they have no fixed table a chosen section could replace.
#: Mirrors ``objective.TC_MODES`` — kept here as a literal so importing api
#: stays free of the physics modules; ``test_api`` gates the two together.
TC_MODES = ("tier_a_plus", "winglet_tc", "winglet_capped_tc")

#: Mirrors ``objective.COUPLED_MODES`` (itself ``geometry.COUPLED_MODES``) —
#: the modes that index the pre-optimised (t/c x cl) CST section library
#: through the TWO-argument lookup. Kept here as a literal for the same
#: reason TC_MODES is, and gated against the source by the same test.
COUPLED_MODES = ("winglet_coupled", "winglet_capped_coupled")


def library_section_polar(name: str, *, re: float | None = None,
                          mach: float = 0.0, n_panel: int = 200,
                          alphas=None, with_cpmin: bool = False,
                          cache_dir=None, timeout_s: float = 20.0,
                          cache_only: bool = False):
    """A :class:`aerobo.polar.TablePolar` for a named library section.

    Two sources, in this order:

    * the SCREENING BRANCH SIDECAR (:data:`SCREEN_BRANCH_CACHE`) — the
      pre-stall (alpha, cl, cd, cm) branch of every screened section at the
      cached point. Instant, no XFOIL, and it is the same data the ranking
      the user picked from was scored on. Used when the requested point is
      that point and no Cp_min table is asked for;
    * a cached XFOIL sweep of the section's coordinates otherwise (another
      Reynolds number, or the ``with_cpmin`` sweep the cavitation constraint
      needs). Cached on disk like every other polar, so it is paid once.

    ``cache_only=True`` returns ``None`` instead of invoking XFOIL on a cache
    miss. It exists for CALLERS THAT RUN INSIDE A RENDER: the second branch
    above is a synchronous XFOIL sweep, seconds to minutes, and a view that
    can block for that long is worse than a view that says nothing. The first
    branch (the sidecar) is unaffected — it never touches XFOIL either way.

    The returned polar carries ``.tc`` (the section's thickness) because the
    problems that fly a chosen section read the thickness off the polar
    instead of a design variable.
    """
    from .airfoil import cst_thickness, fit_cst
    from .polar import TablePolar

    branch = _branch_sidecar()
    point = branch.get("point") or {}
    rec = (branch.get("sections") or {}).get(name)
    wants = (re is None or float(re) == float(point.get("re", 0.0))) \
        and float(mach) == float(point.get("mach", 0.0)) \
        and int(n_panel) == int(point.get("n_panel", 0)) \
        and alphas is None
    if rec and wants and not with_cpmin:
        pol = TablePolar(
            alpha_deg=np.asarray(rec["alpha"], float),
            CL=np.asarray(rec["cl"], float), CD=np.asarray(rec["cd"], float),
            CM=np.asarray(rec["cm"], float),
            name=f"{name} (library, screened)", Re=float(point["re"]))
        pol.tc = float(rec.get("tc") or 0.12)
        return pol

    coords = _library_coords(name)
    re_eff = float(point.get("re", 1e6) if re is None else re)
    sweep = (np.asarray(alphas, float) if alphas is not None
             else np.asarray(point.get("alphas")
                             or np.arange(-4.0, 10.5, 0.5), float))
    from .xfoil_run import DEFAULT_CACHE_DIR, _cache_load, cache_key
    from .xfoil_run import run_xfoil_polar

    if cache_only:
        # LOADABLE, not merely present — a pre-gate cache entry has no CDp
        # column and is a miss, so exists() would turn this into the very
        # synchronous XFOIL solve the flag exists to avoid (_cached_dat_polar
        # learned the same lesson).
        key = cache_key(coords, re_eff, float(mach), list(sweep), int(n_panel))
        root = Path(cache_dir if cache_dir is not None else DEFAULT_CACHE_DIR)
        if _cache_load(root / f"{key}.json", len(list(sweep))) is None:
            return None
    pol_raw = run_xfoil_polar(coords, re_eff, float(mach), sweep,
                              timeout_s=timeout_s, cache_dir=cache_dir,
                              n_panel=int(n_panel), with_cpmin=with_cpmin)
    if pol_raw.n_converged < 3:
        raise ValueError(
            f"section {name!r} converged at only {pol_raw.n_converged} "
            f"angles at Re {re_eff:.3g} — it cannot be flown there")
    pol = TablePolar(
        alpha_deg=pol_raw.alpha_deg, CL=pol_raw.cl, CD=pol_raw.cd,
        CM=pol_raw.cm, name=f"{name} (library, Re {re_eff:.3g})", Re=re_eff,
        alpha_cpmin=(pol_raw.alpha_deg if pol_raw.has_cp_min else None),
        CPMIN=(pol_raw.cp_min if pol_raw.has_cp_min else None))
    x = coords[:, 0]
    i_le = int(np.argmin(x))
    xu, yu = coords[: i_le + 1, 0][::-1], coords[: i_le + 1, 1][::-1]
    xl, yl = coords[i_le:, 0], coords[i_le:, 1]
    grid = np.linspace(0.0, 1.0, 121)
    w_u, w_l, dz = fit_cst(grid, np.interp(grid, xu, yu),
                           np.interp(grid, xl, yl), n_cst=4)
    pol.tc = float(cst_thickness(w_u, w_l, dz_te=dz))
    return pol


def _library_coords(name: str) -> np.ndarray:
    """Closed-loop coordinates of a named library section (sidecar first)."""
    hit = _coords_sidecar().get(name)
    if hit is not None:
        return np.asarray(hit, dtype=float)
    from .airfoil_select import load_airfoil_dat
    path = UIUC_DB_DIR / f"{name}.dat"
    if not path.exists():
        raise FileNotFoundError(
            f"no library section named {name!r} (looked in {UIUC_DB_DIR})")
    return load_airfoil_dat(path)


def section_polar_for(value, **kw):
    """Polar for a chosen section, from any shape the flag can carry.

    * a NAME (``"hg40"``) — a section of the screened library, flown at the
      cached library point;
    * a NAMED section AT A POINT, ``{"name": "hg40", "re": ..., "mach": ...}``
      — the same section, flown at the point it was CHOSEN at. This is what
      makes a mission-Reynolds-number choice mean anything downstream: a
      library polar is measured data at ONE Reynolds number, and the section
      that wins at Re 1e6 need not be the one that wins at the Re a small
      chord actually flies (hg40: L/D 80.6 at 1e6, 42.2 at 3e5). Without the
      point the flag could only ever say WHICH section, never WHERE, and the
      run flew 1e6 polars whatever the mission said;
    * a DESIGNED section, ``{"w_upper": [...], "w_lower": [...]}`` with an
      optional ``"re"``/``"mach"``/``"dz_te"`` — the CST weights the section
      stage optimised. Its XFOIL sweep is cached, and the section stage has
      just run that very sweep, so this is normally a cache hit.

    Explicit keyword arguments WIN over the point carried in the value, so a
    caller that knows the surface's own point can still impose it.

    All three shapes are JSON-safe, which is what lets a chosen section travel
    in a RunConfig instead of a live polar object.
    """
    if isinstance(value, str):
        return library_section_polar(value, **kw)
    if isinstance(value, dict) and value.get("w_upper") is not None:
        return designed_section_polar(value, **kw)
    if isinstance(value, dict) and value.get("name"):
        at = {k: value[k] for k in ("re", "mach", "n_panel")
              if value.get(k) is not None}
        at.update(kw)
        return library_section_polar(str(value["name"]), **at)
    raise TypeError(
        f"a chosen section must be a library name, a named section at a "
        f"point, or a CST weight dict, got {type(value).__name__}")


def water_family_polar(tc: float | None = None):
    """The section a WATER family flies at thickness ``tc`` — a real polar.

    ``hydrofoil.HydrofoilProblem`` reads its section from
    ``polar_family.at(tc)`` on every evaluation (the NACA 24XX bank, with the
    Cp_min table the cavitation constraint needs), and until this existed a
    shell that wanted to draw the same section's suction peak had to reach
    into ``polar`` itself and repeat the family choice. One place, so the
    picture and the solver cannot end up on different sections.

    ``tc`` is clamped to the bank's own range rather than refused: a design
    box centred outside it still has a nearest measured section, and a
    drawing that says "no data" where the solver would have used the end
    member is a refusal dressed as a fact. ``None`` takes the bank's middle.
    """
    from .polar import default_polar_family

    fam = default_polar_family()
    lo, hi = (float(v) for v in fam.tc_range)
    t = 0.5 * (lo + hi) if tc is None else float(tc)
    return fam.at(min(max(t, lo), hi))


def section_polar_point(value) -> dict:
    """The (re, mach) :func:`section_polar_for` WILL fly ``value`` at.

    The resolution rules live in one place — the two polar builders' own
    fallbacks — so a shell can state the flown point without restating them
    and drifting. That statement is the whole reason this exists: a chosen
    section's polar is measured data at one Reynolds number, and which one
    used to be invisible everywhere above the solver.
    """
    lib = screen_library_point() or {}
    if isinstance(value, str):
        return {"re": float(lib.get("re", 1e6)),
                "mach": float(lib.get("mach", 0.0))}
    if not isinstance(value, dict):
        raise TypeError(f"not a chosen section: {type(value).__name__}")
    if value.get("w_upper") is not None:
        # designed_section_polar's own fallbacks
        return {"re": float(value.get("re") or 1e6),
                "mach": float(value.get("mach") or 0.0)}
    return {"re": float(value.get("re") or lib.get("re", 1e6)),
            "mach": float(value.get("mach") or lib.get("mach", 0.0))}


def designed_section_polar(section: dict, *, re: float | None = None,
                           mach: float | None = None, n_panel: int = 200,
                           alphas=None, with_cpmin: bool = False,
                           cache_dir=None, timeout_s: float = 20.0):
    """TablePolar of a DESIGNED (CST) section — the shape stage 2 optimised.

    ``section`` carries ``w_upper`` / ``w_lower`` and may carry the point it
    was designed at (``re``, ``mach``, ``dz_te``); explicit arguments win.
    """
    from .airfoil import cst_coords, cst_thickness
    from .polar import TablePolar
    from .xfoil_run import run_xfoil_polar

    w_u = np.asarray(section["w_upper"], float)
    w_l = np.asarray(section["w_lower"], float)
    dz = float(section.get("dz_te") or 0.0)
    re_eff = float(re if re is not None else section.get("re") or 1e6)
    mach_eff = float(mach if mach is not None else section.get("mach") or 0.0)
    sweep = (np.asarray(alphas, float) if alphas is not None
             else np.arange(-4.0, 10.5, 0.5))
    coords = cst_coords(w_u, w_l, n_pts=160, dz_te=dz)
    raw = run_xfoil_polar(coords, re_eff, mach_eff, sweep,
                          timeout_s=timeout_s, cache_dir=cache_dir,
                          n_panel=int(n_panel), with_cpmin=with_cpmin)
    if raw.n_converged < 3:
        raise ValueError(
            f"the designed section converged at only {raw.n_converged} "
            f"angles at Re {re_eff:.3g} — it cannot be flown there")
    pol = TablePolar(
        alpha_deg=raw.alpha_deg, CL=raw.cl, CD=raw.cd, CM=raw.cm,
        name=f"{section.get('name') or 'designed CST'} (Re {re_eff:.3g})",
        Re=re_eff,
        alpha_cpmin=(raw.alpha_deg if raw.has_cp_min else None),
        CPMIN=(raw.cp_min if raw.has_cp_min else None))
    pol.tc = float(cst_thickness(w_u, w_l, dz_te=dz))
    return pol


def _section_polar_flags(flags: dict | None, key: str = SECTION_KEY,
                         **kw):
    """The chosen section's polar, or ``None`` when none was asked for. One
    place, so every family reads the flag the same way."""
    if not flags:
        return None
    value = flags.get(key)
    if not value:
        return None
    return section_polar_for(value, **kw)


def _section_kwargs(flags: dict | None, field: str = "polar", **kw) -> dict:
    """``{field: polar}`` for the chosen section, or ``{}``."""
    pol = _section_polar_flags(flags, **kw)
    return {} if pol is None else {field: pol}


class MissingSectionError(RuntimeError):
    """A family that flies a CHOSEN section was run without one.

    Deliberately NOT a ValueError: the solvers' failure contract catches
    ValueError and converts it into the -100 penalty, which would turn a
    configuration mistake into a run full of infeasible points and a silent
    ``best_score = None``. This one propagates.
    """


class _SectionRequired:
    """Describes the design vector of a family that flies a CHOSEN section,
    while refusing to be flown itself.

    Only ``tc`` is read to lay the vector out (dropping the thickness row),
    so that is all this carries. Anything that tries to fly it — a lift
    slope, a drag lookup, a Cp_min — raises with the flag to set.
    """

    tc = 0.12
    name = "chosen section (not given)"

    def __getattr__(self, item):
        raise MissingSectionError(
            f"this problem flies the section you chose, and none was given "
            f"(asked for {item!r}). Pass flags={{{SECTION_KEY!r}: "
            f"'<library section name>'}}, or the CST weights of a designed "
            f"one. To search the thickness instead, run the plain family.")


_SECTION_REQUIRED = _SectionRequired()


def _wet_section_kwargs(flags: dict | None, key: str = SECTION_KEY,
                        required: bool = False) -> dict:
    """``{"section_polar": polar}`` for a WATER family's chosen section.

    Two differences from the air path, both physics:

    * the sweep is run ``with_cpmin`` — a cavitation constraint needs the
      minimum surface pressure at every angle, and XFOIL only reports it
      through CPMN. Without that table the foil would be flown with no
      cavitation check at all, which is the one thing this family exists to
      do. It costs one cached XFOIL sweep the first time a section is
      chosen;
    * installing it DROPS the ``tc`` design variable: the chosen section has
      a thickness, so searching one as well would be two answers to one
      question (the same rule the designed-section water family follows).
    """
    out = _section_kwargs(flags, field="section_polar", key=key,
                          with_cpmin=True)
    if required and not out:
        # No section given. The DESIGN VECTOR of this family is still
        # perfectly well defined (taper, twists, depth, speed... none of
        # them is the section), and the shell asks for exactly that when it
        # draws a design box. So describe the vector with a placeholder that
        # cannot be flown: anything that tries raises, with the message
        # below, instead of quietly flying a section nobody chose.
        return {"section_polar": _SECTION_REQUIRED}
    if out and not required:
        raise ValueError(
            "flying a chosen section DROPS the t/c design variable, so it "
            "is its own problem rather than a flag on this one: use "
            "api.chosen_section_problem(name) to get the twin that flies it")
    return out


def _wet_aft_section_kwargs(flags: dict | None) -> dict:
    """``{"polar_tail": polar}`` for a WATER family's SECOND surface.

    The elevator has no thickness variable of its own — it flies a table —
    so handing it a section replaces that table and changes no dimension.
    That is why this is a plain flag where the main foil's is a whole
    problem (:func:`_wet_section_kwargs`). The sweep is still ``with_cpmin``:
    the stabiliser sits deeper than the foil and is perfectly capable of
    being the surface that cavitates first, so its margin has to be read off
    its own Cp_min.
    """
    return _section_kwargs(flags, field="polar_tail", key=SECTION_AFT_KEY,
                           with_cpmin=True)


def _build_objective(mode: str, labels: tuple, medium: str = "air",
                     chord_order: int = 0, junction_drag: bool = False,
                     flight_free: bool = False, size_free: bool = False):
    """Factory for objective.Problem-backed builders (trim/winglet/mission...).

    ``chord_order`` > 0 appends that many chord-law coefficients to the design
    vector (geometry.bounds); ``labels`` must already carry their names, since
    ProblemSpec.param_labels is static.
    """

    def build(mission_kwargs, flags, bounds_overrides):
        from . import objective
        size = _planform_kwargs(flags)
        # the trim target is CL = W/(q S), so a chosen area has to reach the
        # mission BEFORE it is built — otherwise a resized wing would be
        # flown at the 10 m^2 lift coefficient and the reported weight and
        # the reported CL would disagree
        S_ref = size.get("S", 10.0)
        mission = None
        if mode == "mission":
            mission = _mission_for(mission_kwargs, S=S_ref)   # W_N required
        elif mission_kwargs:
            mission = _mission_for(mission_kwargs, S=S_ref)   # custom op point
        # a FIXED blend turns the corner's interference drag on by default,
        # for the same reason the blended-winglet PROBLEMS do: without it a
        # blend is only a differently drawn wake (junction.py)
        blend = _fixed_blend_kwargs(mode, flags)
        prob = objective.Problem(
            mode=mode, mission=mission, **size,
            winglet_cant_bounds=_winglet_cant_bounds(mode, flags),
            chord_order=chord_order_for(chord_order, flags),
            **_chord_law_kwargs(flags), flight_free=flight_free,
            **_size_kwargs(size_free, flags),
            **_wing_loading_kwargs(size_free, flags, size.get("b", 10.0)),
            **_size_band_kwargs(size_free, bounds_overrides, flags),
            **blend,
            **_chord_max_frac(flags), **_chord_limits_kwargs(flags),
            **_winglet_chord_kwargs(flags),
            **_junction_flag(flags, junction_drag or bool(blend)),
            **_blend_shape_flag(flags), **_problem_flags(flags),
            **(_section_kwargs(flags) if takes_section else {}))
        # the chord block is TRAILING in this family, so a law that carries
        # fewer rows than the registered twin (the elliptic blend on a cubic
        # twin) replaces exactly that tail of the static label list
        built_labels = tuple(labels)
        if int(chord_order) > 0:
            built_labels = built_labels[:len(built_labels) - int(chord_order)] \
                + _chord_label_block(chord_order, flags)
        bounds = _apply_overrides(prob.bounds, built_labels, bounds_overrides)
        return _BuiltProblem(
            problem=prob,
            # the SIZE modifier adds the root-bending constraint, so the
            # harness contract changes with it (sizing.py)
            callable=((lambda x: objective.fg(x, prob)) if size_free
                      else (lambda x: objective.objective(x, prob))),
            evaluate=lambda x: objective.evaluate(x, prob),
            bounds=bounds, is_constrained=bool(size_free), dim=prob.dim,
            param_labels=built_labels, medium=medium,
        )

    # a mode that SELECTS its section reads a polar FAMILY per candidate, so
    # there is no fixed table to replace: handing it a chosen section would
    # silently disable the design variables it flies. Two lists, because
    # there are two families with two arities — the NACA 24XX bank indexed by
    # thickness (TC_MODES) and the pre-optimised CST library indexed by
    # (t/c, cl) (COUPLED_MODES).
    takes_section = mode not in TC_MODES and mode not in COUPLED_MODES
    build.takes_section = takes_section
    # the MODE this builder flies, so the registry can decide which specs
    # accept the fixed-blend flags by asking the builder rather than by
    # matching problem names
    build.objective_mode = mode
    # this family GATES the aspect ratio per candidate
    # (``sizing.check_ar_limit``), so the registry may declare the
    # limit flags on it: a flag is declared where it is READ
    build.gates_ar = True
    return build


def _make_aircraft_builder(chord_order: int = 0, flight_free: bool = False):
    def build(mission_kwargs, flags, bounds_overrides):
        from . import aircraft
        kw = _chord_kwargs(chord_order, flags)
        kw.update(_flight_kwargs(flight_free))
        prob = aircraft.AircraftProblem(**kw)
        if mission_kwargs:
            prob = aircraft.AircraftProblem(
                mission=replace(prob.mission, **mission_kwargs), **kw)
        labels = _variant_labels(_AIRCRAFT_LABELS, chord_order, flight_free,
                                 flags=flags)
        bounds = _apply_overrides(prob.bounds, labels, bounds_overrides)
        return _BuiltProblem(
            problem=prob,
            callable=lambda x: aircraft.fg_aircraft(x, prob),
            evaluate=lambda x: aircraft.evaluate_aircraft(x, prob),
            bounds=bounds, is_constrained=True, dim=prob.dim,
            param_labels=labels, medium="air",
        )
    return build


_build_aircraft = _make_aircraft_builder()


#: which water the foil flies in. Sea is the published default, so an
#: untouched run sends nothing and reproduces the frozen studies bit-for-bit;
#: "fresh" carries density, viscosity AND vapour pressure together
#: (hydrofoil.WATER_KINDS), because a lake's cavitation threshold is not sea
#: water's.
WATER_KEY = "water"


def _water_kwargs(flags: dict | None) -> dict:
    """``{rho, mu, p_vap}`` for the requested water, or ``{}`` for the
    published sea-water default."""
    kind = (flags or {}).get(WATER_KEY)
    if kind in (None, "", "sea"):
        return {}
    from .hydrofoil import water_properties
    return water_properties(str(kind))


#: how deep the whole assembly may reach below the free surface [m]. A
#: CONSTRAINT, not a bound: what it limits is ``depth_m`` PLUS the tip
#: device's downward reach, and ``depth_m`` is itself a design variable, so
#: no box on either row can express it. Absent/None = uncapped, which is
#: every published water run, bit-for-bit.
#:
#: This is the water answer to the air families' span cap, and it is
#: deliberately NOT the same quantity — see report §17.10: the lateral
#: projection is EVEN in the tip device's cant while the physics is ODD in
#: it, so a span cap here would hold fixed something blind to the trade.
DRAUGHT_MAX_KEY = "draught_max_m"


def _draught_kwargs(flags: dict | None) -> dict:
    """``{draught_max_m}`` for a capped run, or ``{}`` for the uncapped one."""
    v = (flags or {}).get(DRAUGHT_MAX_KEY)
    if v in (None, ""):
        return {}
    return {"draught_max_m": float(v)}


def water_kinds() -> dict:
    """``{key: label}`` of the waters a hydrofoil run may be flown in."""
    from .hydrofoil import WATER_KINDS
    return {k: v["label"] for k, v in WATER_KINDS.items()}


def _make_hydrofoil_builder(chord_order: int = 0,
                            chosen_section: bool = False):
    def build(mission_kwargs, flags, bounds_overrides):
        from . import hydrofoil
        if mission_kwargs:
            # the water families carry their own operating point IN the
            # design vector (speed and depth), so a mission card here would
            # be two answers to one question
            raise ValueError(
                "the hydrofoil problem has no mission spec — its speed and "
                "depth are design variables (edit their box bounds instead)")
        sec = _wet_section_kwargs(flags, required=chosen_section)
        prob = hydrofoil.HydrofoilProblem(
            **_chord_kwargs(chord_order, flags), **_water_kwargs(flags),
            **_strut_config(flags),
            **_planform_kwargs(flags, *_WET_SIZE), **_weight_kwargs(flags),
            **_water_band_kwargs(bounds_overrides),
            **_water_flight_kwargs(flags), **_water_re_kwargs(flags),
            **_wet_size_kwargs(flags, bounds_overrides), **sec)
        labels = prob.param_labels
        bounds = _apply_overrides(prob.bounds, labels, bounds_overrides)
        return _BuiltProblem(
            problem=prob,
            callable=lambda x: hydrofoil.fg_hydrofoil(x, prob),
            evaluate=lambda x: hydrofoil.evaluate_hydrofoil(x, prob),
            bounds=bounds, is_constrained=True, dim=prob.dim,
            param_labels=labels, medium="water",
        )
    return build


_build_hydrofoil = _make_hydrofoil_builder()


def _make_hydrofoil_winglet_builder(chord_order: int = 0,
                                    chosen_section: bool = False):
    """The imaged nonplanar water family: foil + tip device.

    Its builder is TAGGED ``takes_fixed_blend`` so the registry can declare
    the blend flags on every twin of it (chord law, chosen section) without a
    name list — the same rule the section flags are declared by.
    """

    def build(mission_kwargs, flags, bounds_overrides):
        from . import hydrofoil
        if mission_kwargs:
            # the water families carry their own operating point IN the
            # design vector (speed and depth), so a mission card here would
            # be two answers to one question
            raise ValueError(
                "the hydrofoil + winglet problem has no mission spec — its "
                "speed and depth are design variables (edit their box "
                "bounds instead)")
        sec = _wet_section_kwargs(flags, required=chosen_section)
        prob = hydrofoil.HydrofoilWingletProblem(
            **_chord_kwargs(chord_order, flags), **_water_kwargs(flags),
            **_strut_config(flags), **_foil_cant_kwargs(flags),
            # the FOIL's own cant and sweep (not the tip device's, which is
            # what _foil_cant_kwargs above narrows) — lattice geometry, so
            # declared on the imaged families and refused on the planar one
            **_wing_cant_kwargs(flags),
            **_wet_blend_kwargs(flags), **_winglet_chord_kwargs(flags),
            **_planform_kwargs(flags, *_WET_SIZE), **_draught_kwargs(flags),
            **_weight_kwargs(flags),
            **_water_band_kwargs(bounds_overrides),
            **_water_flight_kwargs(flags), **_water_re_kwargs(flags),
            **_wet_size_kwargs(flags, bounds_overrides), **sec)
        labels = prob.param_labels
        bounds = _apply_overrides(prob.bounds, labels, bounds_overrides)
        return _BuiltProblem(
            problem=prob,
            callable=lambda x: hydrofoil.fg_hydrofoil_winglet(x, prob),
            evaluate=lambda x: hydrofoil.evaluate_hydrofoil_winglet(x, prob),
            bounds=bounds, is_constrained=True, dim=prob.dim,
            param_labels=labels, medium="water",
        )
    build.takes_fixed_blend = True
    return build


_build_hydrofoil_winglet = _make_hydrofoil_winglet_builder()


#: the car-wing configuration keys EVERY car family reads. All VALUE
#: changes (mount layout, budgets, add-on switches) — none of them alters
#: the design vector, so ProblemSpec.param_labels stays honest.
_CAR_CORE_KEYS = ("mount", "CD_budget", "deflection_limit_m", "junction_drag",
                  "blend_frac", "blend_shape", "V", "ei_nm2",
                  # NOTE there is no span_min_m / area_min_m2 here. How wide
                  # and how big the wing may be are BANDS ON DESIGN
                  # VARIABLES, so they are the design box's own b_m and S_m2
                  # rows and reach the problem through
                  # _car_size_band_kwargs. Asking them twice is what broke
                  # them, in both directions (see _CAR_SIZE_ROWS), and
                  # check_flags now refuses the old flag rather than
                  # honouring a second answer to a question the box owns.
                  # WHICH SCALAR is maximised (carwing.CAR_OBJECTIVES) and the
                  # two optional budgets that go with it. All three are VALUE
                  # changes — none of them touches the design vector.
                  "car_objective", "drag_budget_n", "downforce_min_n",
                  # THE PLATE'S CHORD LAW. A VALUE change, hard-wired at the
                  # one VLM call site each family makes: the plate carried
                  # the wing's TIP chord, i.e. a rectangle bolted to a
                  # tapered wing. vlm.py has been able to continue the chord
                  # law onto a tip device since winglet_chord_follows was
                  # written; the car simply never asked. Off reproduces every
                  # published run bit-for-bit.
                  #
                  # ...and THE PLATE'S CANT, which these families withheld
                  # until they could charge it. It used to be undeclarable
                  # here because ``b_m`` was read as the WING's span, so a
                  # leaning plate projected PAST the band for nothing:
                  # measured, cant 60 scored 39.79 against 90's 38.28 by
                  # flying 1.725 m of wing inside a 1.6 m band, which on a
                  # car is a regulation or a piece of bodywork. Withholding
                  # it never closed the hole — ``blend_frac`` was declared
                  # the whole time and walked through the same gap, buying
                  # +12.3 % CZ at a 1.92 m width inside a stated 1.6 m.
                  #
                  # Both families now read the span row as the OVERALL width
                  # and take the plate's outboard reach out of the wing's own
                  # span (carwing.py / carwing_multi.py, the same developed
                  # line endplate.py uses), refusing a projection that eats
                  # the whole width. A vertical unblended plate is guarded
                  # out of that arithmetic entirely and is bit-for-bit the
                  # published run.
                  "endplate_cant_deg",
                  "endplate_chord_follows")

#: ...and the keys only the SINGLE-ELEMENT family reads: the mount as a
#: continuum (carmount.py), the lap (cartrack.py), the two speeds a force
#: requirement can be stated AT, the ride-height sensitivity (as a DIAGNOSTIC
#: and, separately, as a CEILING) and the flown-Reynolds switch.
#:
#: ``dczdh_report`` and ``dczdh_max_per_m`` are two keys for what looks like
#: one question, and the split is the point. A ceiling is a CONSTRAINT: it
#: adds a margin, so it changes the feasible set and the width of the vector
#: the optimiser sees. Asking what dCZ/dh IS must not do any of that — this
#: repo's rule is that answering one more question never widens or narrows a
#: search. With only the ceiling declared, the only way to see the number
#: would be to impose a limit on it, which is the rule inverted. Kept apart from :data:`_CAR_CORE_KEYS` rather than
#: folded into it because ``carwing_multi.CarWingMultiProblem`` has none of
#: these fields: declared on that family they would be accepted and read by
#: nothing, which is the exact hole :func:`check_flags` exists to close.
#: the two keys that state a LAP, named apart from the rest of
#: :data:`CAR_MOUNT_MISSION_KEYS` because they are the two that stopped being
#: single-element-only. Both car families now carry ``track_spec`` /
#: ``car_spec`` / ``track_points``; the rest of that tuple describes a
#: ``carmount.MountSpec`` or a ride-height sensitivity, and
#: ``CarWingMultiProblem`` still has no field for any of them.
CAR_LAP_KEYS = ("car_track", "track_points")

#: how the mount is described. ``published`` is carwing's two-valued
#: :data:`carwing.MOUNTS` lookup, which is what every published run flew;
#: ``continuum`` is a :class:`carmount.MountSpec` built from the same layout,
#: which changes five things at once (carwing's module docstring lists them)
#: and is therefore a MODE rather than a silent upgrade. Spelled as a mode
#: string for the reason :data:`CAR_BUDGET_OFF` is.
CAR_MOUNT_MODELS = ("published", "continuum")

#: WHERE the load leaves the wing, as the two statements
#: :class:`carmount.MountSpec` makes and refuses to conflate — a FRACTION of
#: the semi-span (what "at the tips" means, and the only spelling that tracks
#: a span which is itself a design variable) or a LENGTH from the centreline
#: (what a chassis hardpoint is: a distance between two pieces of car, which
#: does not move when the wing's span does). Exactly one of them, refused
#: rather than resolved by precedence, for the reason the dataclass gives.
CAR_MOUNT_STATION_KEYS = ("mount_station_frac", "mount_station_m")

#: ...and the two ``MountSpec`` fields they map onto, in the same order.
CAR_MOUNT_STATION_MOUNTSPEC_FIELDS = ("station_frac", "y_station_m")

#: the MountSpec fields a caller may refine once the continuum is selected.
#: Refusing them under ``published`` is the point: they describe a MountSpec
#: and nothing reads them without one.
#:
#: THE STATION, THE KIND AND THE SIDE ARE HERE because without them the
#: continuum was a mode with nothing to say. ``carmount``'s own module
#: docstring names the case it exists for — "a mount 35% out along the
#: semi-span ... is not an exotic layout, it is what a wide wing with a
#: narrow chassis does" — yet every mount reachable through ``flags`` was
#: still one of ``carwing.MOUNTS``' two, because the only refinements
#: declared were the four scalars below and the layout itself came from
#: :func:`carmount.published_layout`. So the two published ends were the
#: two ends, and the continuum between them was unaskable.
_CAR_MOUNT_REFINE_KEYS = ("x_attach_frac", "gj_nm2", "pylon_drag_model",
                          "suction_loss",
                          *CAR_MOUNT_STATION_KEYS,
                          "mount_kind", "mount_side", "mount_n_pylons")

#: flag key -> ``carmount.MountSpec`` field, where the two differ. The
#: ``mount_`` prefix is not decoration: ``kind``, ``side`` and ``station_frac``
#: are generic enough that a bare spelling in the one flat flag namespace this
#: package has would read as a property of the wing rather than of its mount.
_CAR_MOUNT_REFINE_FIELDS = {"mount_station_frac": "station_frac",
                            "mount_station_m": "y_station_m",
                            "mount_kind": "kind",
                            "mount_side": "side",
                            "mount_n_pylons": "n_pylons"}

#: which refinements are STRINGS (the rest are numbers).
_CAR_MOUNT_STR_REFINE_KEYS = ("pylon_drag_model", "mount_kind", "mount_side")

CAR_MOUNT_MISSION_KEYS = ("car_mount_model", "deck_height_m",
                          # the MOUNT AS A CONTINUUM: the four scalars, plus
                          # the station, the kind and the side that make it
                          # more than a two-valued lookup. Declared through
                          # _CAR_MOUNT_REFINE_KEYS rather than re-listed, so
                          # a refinement added there is askable without a
                          # second edit here (which is where they would drift)
                          *_CAR_MOUNT_REFINE_KEYS,
                          *CAR_LAP_KEYS,
                          "downforce_min_v_ms", "drag_budget_v_ms",
                          "dczdh_report", "dczdh_max_per_m",
                          "flown_reynolds")

#: the mount/sensitivity half of the above: exactly the keys that are still
#: single-element-only. A family declaring one of these without a MountSpec,
#: a deck or a dCZ/dh switch would accept a flag nothing reads.
CAR_SINGLE_ONLY_KEYS = tuple(k for k in CAR_MOUNT_MISSION_KEYS
                             if k not in CAR_LAP_KEYS)

CAR_WING_KEYS = _CAR_CORE_KEYS + CAR_MOUNT_MISSION_KEYS

#: ...and the TWO-ELEMENT family's own. The four slot rows are DESIGN
#: VARIABLES (narrow them with bounds_overrides like any other row), so what
#: is left to configure is how finely the 2-D cascade is panelled — a
#: cost/accuracy knob whose price carwing_multi.N_SECTION_NODES measures —
#: and the LAP.
#:
#: ``car_track`` / ``track_points`` are the same two keys the single-element
#: family takes and they mean the same two things. They are declared here
#: because ``carwing_multi.CarWingMultiProblem`` now HAS the fields they set,
#: which is the test :func:`check_flags` applies: the rest of
#: :data:`CAR_MOUNT_MISSION_KEYS` is still absent from this tuple because that
#: class still has none of those fields, and a flag declared on a family that
#: reads it nowhere is exactly the hole check_flags exists to close.
CAR_WING_MULTI_KEYS = _CAR_CORE_KEYS + ("n_section_nodes",
                                        "car_track", "track_points")


#: the circuits ``car_track`` names, and the off switch. One shipped lap
#: (cartrack.synthetic_lap) plus a mode string for "no mission", again
#: because a flag loop that skips ``None`` cannot carry "off" as a value.
CAR_TRACK_OFF = ("off", "none")
CAR_TRACKS = ("synthetic",)

#: keys the two helpers below own, so the plain value loop must skip them.
_CAR_MOUNT_MISSION_HELPER_KEYS = (("car_mount_model", "deck_height_m")
                                  + _CAR_MOUNT_REFINE_KEYS
                                  + ("car_track", "track_points"))

#: flag key -> ``CarWingProblem`` field, where the two differ. ``objective``
#: is namespaced on the way in because the shell and the section stage each
#: already have one and a bare ``objective`` would be three questions sharing
#: a key.
_CAR_KEY_ALIASES = {"car_objective": "objective"}

#: what ``CD_budget`` accepts besides a number, and what it means. A COEFFICIENT
#: budget is the published constraint, but "maximise it and do not budget it"
#: is a legitimate question — the car's drag may be paid for elsewhere — and
#: the family had no way to ask it. Spelled as a mode string for the same
#: reason ``bo_refusal`` is: a flag loop that skips ``None`` cannot carry
#: "switch this off" as a value.
CAR_BUDGET_OFF = ("off", "none")

#: the span BAND a car wing may be searched over, in metres. Named exactly as
#: the wing-loading mode's own band (WING_LOADING_KEYS) because it is the same
#: question — "how wide may this wing be" — and :func:`span_box` already
#: caches on those keys.
#: THE CAR'S TWO SIZE ROWS, and the one place they are stated. ``b_m`` and
#: ``S_m2`` are ordinary rows of the design box — a band on a design variable,
#: exactly like the twists and the ride height — so that is where a user says
#: how wide and how big the wing may be. There is no ``span_min_m`` /
#: ``area_min_m2`` FLAG any more, and this is not tidying: the two paths were
#: a live defect of :func:`_arm_band_kwargs`'s class, in both directions at
#: once.
#:
#: MEASURED before the change, on ``car rear wing``:
#:
#: * widen the box's ``b_m`` row to (1.2, 3.0) and the SAMPLER drew from it
#:   while ``prob.bounds`` stayed (1.2, 2.0) — every draw above 2.0 m came
#:   back ``feasible=False, reason='bounds violation'``, score -100. A user
#:   who widened the row got a run made entirely of refusals.
#: * type "span, no more than 1.5 m" on the card and the run SEARCHED
#:   (1.2, 1.5) while the box on screen read (1.2, 2.0) — the box shown was
#:   not the box searched, on the row the card had just been used to state.
#:
#: Both vanish when the row IS the band. The map runs row label -> the
#: problem field it sets and the ``carwing`` validator that owns its
#: refusals, named here beside the field for the reason
#: :data:`_WATER_OPERATING_ROWS` gives: the row reaches the validator WHOLE,
#: so a malformed band meets that module's message and not a TypeError.
_CAR_SIZE_ROWS = {"b_m": ("span_bounds_m", "span_row"),
                  "S_m2": ("area_bounds_m2", "area_row")}


def _car_size_band_kwargs(bounds_overrides: dict | None) -> dict:
    """The ``b_m`` and ``S_m2`` design-box rows -> the car problem's own bands.

    The car's instance of :func:`_arm_band_kwargs`'s bug class, and the one
    that bit in both directions (see :data:`_CAR_SIZE_ROWS` for the two
    measurements). Empty in, empty out: with no rows the family opens on
    ``carwing.SPAN_BOUNDS_M`` and ``carwing.AREA_BOUNDS_M2``, bit-for-bit.

    IDEMPOTENT UNDER REBUILD, which is a contract and not an accident:
    ``gui.v3.relax._cap_to_validity`` clips a widened row back to the
    problem's validated band, rebuilds, and asserts the clip converged in ONE
    pass — a helper that re-derived or clamped the band it was handed would
    leave the row outside validity on the second pass too, and the whole
    no-solution "reach" card would abort for the car with "this family
    re-derives its own bounds from the ones it is handed". So the band that
    goes in comes back out of ``prob.bounds`` unchanged, and a corner it
    reaches that the solvers cannot describe is refused PER CANDIDATE by
    ``sizing.check_ar`` — which is where that limit belongs, because it is a
    property of one chordwise panel and not of the band.
    """
    from . import carwing

    ov = bounds_overrides or {}
    out: dict = {}
    for label, (field, validator) in _CAR_SIZE_ROWS.items():
        row = ov.get(label)
        if row is None:
            continue
        band = (float(row[0]), float(row[1]))
        getattr(carwing, validator)(band)      # refuse a malformed band here
        out[field] = band
    return out


def _car_budget(value):
    """A car budget flag as a number, or None when it is switched off."""
    if isinstance(value, str):
        if value.strip().lower() not in CAR_BUDGET_OFF:
            raise ValueError(
                f"a car budget is a number or one of {list(CAR_BUDGET_OFF)} "
                f"(switch it off), got {value!r}")
        return None
    return float(value)


def _car_core_kwargs(flags: dict | None, keys: tuple) -> dict:
    """The car families' shared VALUE flags, as problem kwargs.

    ``keys`` is the family's own declared tuple, so a family that does not
    read a key never has it converted for it. Empty in / empty out: with no
    flags the constructor call is bit-for-bit the published problem's.
    """
    if not flags:
        return {}
    out: dict = {}
    for k in keys:
        v = flags.get(k)
        if v is None or k in _CAR_MOUNT_MISSION_HELPER_KEYS:
            continue
        field = _CAR_KEY_ALIASES.get(k, k)
        if k in ("mount", "blend_shape", "car_objective"):
            out[field] = str(v)
        elif k in ("junction_drag", "flown_reynolds", "dczdh_report",
                   "endplate_chord_follows"):
            out[field] = bool(v)
        elif k == "n_section_nodes":
            out[field] = int(v)
        elif k == "CD_budget":
            out[field] = _car_budget(v)
        else:
            out[field] = float(v)
    return out


def _car_mount_kwargs(flags: dict | None) -> dict:
    """The MOUNT, stated ONCE — carwing's lookup or a carmount.MountSpec.

    ``carwing.CarWingProblem`` refuses ``mount_spec`` beside a non-default
    ``mount`` (and beside its own ``deck_height_m``) rather than resolving the
    clash with a precedence rule, so the two statements are assembled here
    instead of both being sent: under ``car_mount_model='continuum'`` the
    layout the user picked becomes the SPEC (``carmount.published_layout``)
    and the plain ``mount`` kwarg is dropped by the caller.

    The refinement keys are REFUSED under ``published`` for the reason
    :func:`check_flags` exists: without a MountSpec nothing reads them, and a
    silently dropped flag is worse than a refused one.
    """
    f = flags or {}
    model = f.get("car_mount_model")
    stated = [k for k in _CAR_MOUNT_REFINE_KEYS if f.get(k) is not None]
    if model is None or str(model) == "published":
        if stated:
            raise ValueError(
                f"{stated} describe a carmount.MountSpec and are read by "
                f"nothing without one — set car_mount_model='continuum' to "
                f"ask for it. (carwing.py's published MOUNTS lookup has no "
                f"attachment chord, no GJ, no pylon build-up and no suction "
                f"loss to set.)")
        return ({"deck_height_m": float(f["deck_height_m"])}
                if f.get("deck_height_m") is not None else {})
    if str(model) not in CAR_MOUNT_MODELS:
        raise ValueError(
            f"car_mount_model is one of {list(CAR_MOUNT_MODELS)}, "
            f"got {model!r}")
    from dataclasses import replace

    from . import carmount, carwing
    # the default layout's NAME is read off the field, never pasted: it moved
    # once already (the pylon layout left this family) and a literal written
    # down in a second place is a literal that goes stale in one of them
    spec = carmount.published_layout(
        str(f.get("mount") or carwing.CarWingProblem.mount))
    changes: dict = {}
    if f.get("deck_height_m") is not None:
        changes["deck_height_m"] = float(f["deck_height_m"])
    for k in _CAR_MOUNT_REFINE_KEYS:
        if f.get(k) is None:
            continue
        field = _CAR_MOUNT_REFINE_FIELDS.get(k, k)
        if k in _CAR_MOUNT_STR_REFINE_KEYS:
            changes[field] = str(f[k])
        elif k == "mount_n_pylons":
            changes[field] = int(f[k])
        else:
            changes[field] = float(f[k])
    _car_mount_station(spec, changes, f)
    _car_mount_load_path(spec, changes)
    # replace() re-runs MountSpec.__post_init__, so an out-of-band refinement
    # is refused by the spec's own validation rather than here
    return {"mount_spec": replace(spec, **changes) if changes else spec}


def _car_mount_station(spec, changes: dict, flags: dict) -> None:
    """Resolve the station into ``changes`` — stated once, or refused.

    ``MountSpec`` refuses a spec carrying both spellings, and every entry of
    :data:`carmount.PUBLISHED_LAYOUTS` carries ``station_frac``. So a caller
    who states the station as a LENGTH would hand ``replace()`` a spec with
    both set and meet the dataclass's message about a station it never
    mentioned. The other spelling is cleared HERE, where the two flags are,
    and stating both is refused in the flags' own vocabulary.
    """
    both = [k for k in CAR_MOUNT_STATION_KEYS if flags.get(k) is not None]
    if len(both) > 1:
        raise ValueError(
            f"state the mount station exactly once: {both} are two answers "
            f"to one question. mount_station_frac is a fraction of the "
            f"semi-span (it TRACKS a span that is a design variable), "
            f"mount_station_m is a length from the centreline (what a "
            f"chassis hardpoint is, and it does not move when the span "
            f"does)")
    if "y_station_m" in changes:
        changes["station_frac"] = None
    elif "station_frac" in changes:
        changes["y_station_m"] = None


def _car_mount_load_path(spec, changes: dict) -> None:
    """COUNT the wetted members this mount adds. Never inherit them.

    ``n_sheets`` and ``n_pylons`` are what a mount COSTS, and they are a
    consequence of the kind and the station rather than a third statement:

    * a ``pylon`` mount carries its load on struts and adds no sheets;
    * an ``endplate`` mount AT THE TIP adds nothing at all — the plates are
      there for the nonplanar benefit and the load path is free — while a
      grip anywhere inboard of the tip is a SECOND PAIR of plates, which is
      wetted area and two more wing corners.

    Deriving it is not tidiness. ``carmount.MountSpec.n_sheets`` exists
    because the continuum path once handed the inboard grip its stiffness
    for free (that field's own comment carries the measurement: CD and score
    bit-identical to the tip-borne wing while keeping a quarter of its
    deflection). Moving the station without re-counting the load path would
    re-open exactly that free lunch, one flag further out.

    A station stated as a LENGTH is never "at the tip": the span is a design
    variable, so no length is the tip for every candidate. It therefore
    charges the second pair, which is the conservative reading and the one
    that matches what such a mount is.
    """
    if not ({"kind"} | set(CAR_MOUNT_STATION_MOUNTSPEC_FIELDS)) & set(changes):
        return                      # neither the kind nor the station moved
    kind = str(changes.get("kind", spec.kind))
    if kind == "pylon":
        # setdefault, so an explicitly stated count wins and a contradiction
        # (a pylon mount asked for zero pylons) meets MountSpec's own refusal
        changes.setdefault("n_pylons", spec.n_pylons or 2)
        changes["n_sheets"] = 0
        return
    changes.setdefault("n_pylons", 0)
    if "y_station_m" in changes and changes["y_station_m"] is not None:
        frac = None
    else:
        frac = changes.get("station_frac", spec.station_frac)
    changes["n_sheets"] = 0 if (frac is not None and frac >= 1.0) else 2



def _car_track_kwargs(flags: dict | None) -> dict:
    """``car_track`` / ``track_points`` -> the LAP (cartrack.TrackSpec).

    Off is a MODE STRING, not an absent key, for the reason
    :data:`CAR_BUDGET_OFF` gives; and ``track_points`` without a track is
    refused, because sampling a lap that does not exist is not a request.
    """
    f = flags or {}
    v = f.get("car_track")
    off = v is None or str(v).strip().lower() in CAR_TRACK_OFF
    if off:
        if f.get("track_points") is not None:
            raise ValueError(
                "track_points says how many representative speeds a LAP is "
                "sampled at, and no lap was asked for — set car_track "
                f"({list(CAR_TRACKS)}) too, or drop track_points")
        return {}
    if str(v) not in CAR_TRACKS:
        raise ValueError(
            f"car_track is one of {list(CAR_TRACKS)} or one of "
            f"{list(CAR_TRACK_OFF)}, got {v!r}")
    from .cartrack import synthetic_lap
    out: dict = {"track_spec": synthetic_lap()}
    if f.get("track_points") is not None:
        out["track_points"] = int(f["track_points"])
    return out


def _car_wing_kwargs(flags: dict | None) -> dict:
    """CarWingProblem kwargs from ``flags`` (empty in -> published defaults)."""
    out = _car_core_kwargs(flags, CAR_WING_KEYS)
    out.update(_car_mount_kwargs(flags))
    out.update(_car_track_kwargs(flags))
    if "mount_spec" in out:
        # stated once: the spec IS the mount, and CarWingProblem refuses the
        # pair rather than letting one of them be silently ignored
        out.pop("mount", None)
    return out


def _car_wing_multi_kwargs(flags: dict | None) -> dict:
    """CarWingMultiProblem kwargs from ``flags`` (empty in -> defaults).

    The LAP goes through the same helper the single-element family uses, so
    ``car_track='synthetic'`` names the same circuit on both and a lap timed
    on one is comparable with a lap timed on the other. That comparability is
    the point: this family exists to be measured against its sibling, and two
    families reading two different tracks off one flag name would make every
    such comparison a comparison of circuits.
    """
    out = _car_core_kwargs(flags, CAR_WING_MULTI_KEYS)
    out.update(_car_track_kwargs(flags))
    return out


#: what a registered car family maximises when the caller says nothing, and
#: what it budgets. Both departures from ``CarWingProblem``'s own published
#: defaults, and both forced rather than chosen — which is why they live here,
#: at the registry, and NOT on the dataclass: the solver stays the published
#: solver, and ``carsection.py``'s fixed reference planform and
#: ``carwing_multi.compare_split``'s matched-area comparison keep answering
#: the questions they were calibrated on.
#:
#: THE SCORE IS A RATIO. ``cz`` is refused outright against a free area
#: (``carwing.CAR_OBJECTIVES`` says why: a coefficient referenced to the very
#: area being searched is maximised by shrinking the wing), so the family
#: cannot inherit it — and ``downforce`` cannot be the default either, because
#: with no drag ceiling a FORCE is maximised by spending drag until the
#: section stalls. ``efficiency`` = CZ/CD = F_z/D is the one scalar that is
#: well-posed with the area free AND nothing budgeted: the area cancels, and
#: it has an interior optimum instead of running to a bound. MEASURED, at a
#: fixed shape over the default area band: CZ/CD peaks at 35.53 near
#: S = 0.14 m^2 and falls to 25.37 at 0.48, so the peak is interior and the
#: search is not a race to the smallest wing. What it does NOT do is make
#: much downforce (192.6 N at that peak against 486.1 N at 0.40 m^2), which
#: is what ``downforce_min_n`` is for and why the shell offers it beside the
#: drag ceiling.
#:
#: NOTHING IS BUDGETED. The published families carried a drag allowance
#: (``CD_budget`` 0.11, restated as ``drag_budget_n`` 81.5 N once the area
#: moved) and the answer SPENT it: an allowance handed to a maximiser is a
#: number the answer rides, not a limit it respects. On the certificates
#: ``g_star[0]`` is 0.0482 and 0.035 — the drag margin is near-binding at
#: both certified optima while binding almost nowhere in the box, which is
#: the signature of a ceiling picking the answer rather than bounding it. So
#: the default is NO drag constraint at all, and a ceiling is a limit the
#: user may state (``drag_budget_n``), in newtons.
#:
#: What survives with nothing stated is the DEFLECTION margin, which
#: ``CarWingProblem`` declares unconditionally for exactly this reason — the
#: family is never a constrained problem with nothing to constrain.
CAR_DEFAULT_OBJECTIVE = "efficiency"


def _car_registry_defaults(flags: dict | None) -> dict:
    """The registered car family's opening configuration.

    See :data:`CAR_DEFAULT_OBJECTIVE` for why each departure is forced. Each
    is only a DEFAULT: a caller who states ``car_objective``, ``CD_budget``,
    ``drag_budget_n`` or ``downforce_min_n`` keeps what they stated, and
    ``CD_budget`` is switched off here only when the caller has asked for no
    budget of either kind — "no budget by default" must not become "no
    budget, ever".
    """
    f = flags or {}
    out: dict = {}
    if f.get("car_objective") is None:
        out["objective"] = CAR_DEFAULT_OBJECTIVE
    if f.get("CD_budget") is None:
        # the dataclass ships a COEFFICIENT allowance; switch it off rather
        # than let it pick the answer. An explicit None, not an absent key,
        # because the field's own default is a number.
        #
        # Gated on CD_budget ALONE, and that is the fix for a real defect:
        # gated on "neither budget stated" as well, a user who typed a drag
        # CEILING in newtons got the coefficient allowance back with it, and
        # the run reported TWO drag margins ("drag budget margin" beside
        # "drag force margin") — one of them a limit nobody asked for, on the
        # reference area the search is moving. Stating a ceiling must never
        # revive a different ceiling.
        out["CD_budget"] = None
    return out


#: the AREA band a car wing opens on when the design box says nothing, in
#: m^2. DERIVED, not chosen (``carwing.AREA_BOUNDS_M2`` shows the arithmetic):
#: one chordwise panel is honest only over ``sizing.AR_LIMITS``, and this is
#: the area band that keeps every corner of the default SPAN band inside it.
#: A user who widens either row past that reaches corners the solvers cannot
#: describe, and those are refused per candidate by ``sizing.check_ar`` — the
#: limit belongs to the solver, the band is only a default.
def _car_area_defaults() -> dict:
    """The published AREA band, as the constructor kwarg.

    Always set, because the registered families always design their area: an
    absent band means "the default band", never "keep it fixed". The design
    box's ``S_m2`` row overrides it (:func:`_car_size_band_kwargs`).
    """
    from .carwing import AREA_BOUNDS_M2

    return {"area_bounds_m2": (float(AREA_BOUNDS_M2[0]),
                               float(AREA_BOUNDS_M2[1]))}


def _make_car_wing_builder(chord_order: int = 0, area_free: bool = False):
    def build(mission_kwargs, flags, bounds_overrides):
        from . import carwing
        if mission_kwargs:
            # the car problem has no MissionSpec: its operating point is the
            # single speed V (a flag), and silently dropping W_N/altitude here
            # would let the Mission card look like it did something
            raise ValueError(
                "the car rear wing has no mission spec — set its speed with "
                "flags={'V': ...}; weight and altitude do not enter")
        # the size bands, innermost-first: the family's published defaults,
        # then the DESIGN BOX's own b_m / S_m2 rows, which are the only place
        # a user states either (_CAR_SIZE_ROWS says what that fixed)
        area_kw = ({**_car_registry_defaults(flags), **_car_area_defaults()}
                   if area_free else {})
        area_kw.update(_car_size_band_kwargs(bounds_overrides))
        prob = carwing.CarWingProblem(
            **{**_car_wing_kwargs(flags), **area_kw},
            **_chord_kwargs(chord_order, flags),
            **_section_kwargs(flags, field="section_polar"))
        labels = _with_chord_labels(
            _CAR_WING_AREA_LABELS if area_free else _CAR_WING_LABELS,
            chord_order, flags)
        bounds = _apply_overrides(prob.bounds, labels, bounds_overrides)
        return _BuiltProblem(
            problem=prob,
            callable=lambda x: carwing.fg_car_wing(x, prob),
            evaluate=lambda x: carwing.evaluate_car_wing(x, prob),
            bounds=bounds, is_constrained=True, dim=prob.dim,
            param_labels=labels, medium="air",
        )
    # the car's thickness was never a design variable, so a chosen section
    # replaces the polar without touching the vector — a plain flag
    build.takes_section = True
    # this family GATES the aspect ratio per candidate
    # (``sizing.check_ar_limit``), so the registry may declare the
    # limit flags on it: a flag is declared where it is READ
    build.gates_ar = True
    return build


#: THE AREA IS ALWAYS A DESIGN VARIABLE. There is one single-element car
#: family, not a fixed-area/free-area pair: how big the wing is is a design
#: question exactly as how wide it is, and it is asked in the same place (the
#: ``S_m2`` row of the design box). The old fixed-area twin is gone, so the
#: name below IS the free-area problem — kept under the family's own name
#: rather than renamed to "(free area)", because a rename orphans every
#: stored case key, both certificates and every CAD export stem to say a
#: thing the design box already says.
#:
#: ``area_free`` survives as a builder parameter because the SOLVER stays
#: general (``carwing.CarWingProblem`` still fixes its area when handed no
#: band, which is what ``carsection.py``'s fixed reference planform and
#: ``compare_split``'s matched-area comparison need). What changed is which
#: problem the registry offers.
_build_car_wing = _make_car_wing_builder(area_free=True)


def _make_car_wing_multi_builder(chord_order: int = 0,
                                 area_free: bool = False):
    """The TWO-ELEMENT rear wing (carwing_multi.py), built like its parent.

    Everything outside the section is ``carwing.py``'s and so is everything
    here: the same refusal of a mission card, the same free-area defaults (a
    downforce COEFFICIENT against a free reference area is maximised by
    shrinking the wing, so that family cannot inherit ``cz``), the same span
    and area bands, the same chord machinery.

    One departure, because the SECTION is different: no ``section`` flag.
    This family's section is a slotted CASCADE built from a base shape and
    four design rows, not a polar looked up by name, so
    ``build.takes_section`` is deliberately not set — a section flag here
    would be accepted and read by nothing.

    ``car_objective='laptime'`` used to be refused HERE, unconditionally, and
    is not any more — because the reason it was refused has been fixed rather
    than restated. The refusal existed because ``carwing_multi`` validated the
    objective against ``carwing.CAR_OBJECTIVES``, which grew a ``laptime``
    entry, while having no ``track_spec`` and no ``laptime`` row in its
    scoring table: it CONSTRUCTED and raised ``KeyError`` on the first
    evaluation. That module now carries the circuit, the car and the scorer,
    so the name is answerable — and it is still refused, by that class's own
    constructor and in the same words ("needs a circuit"), when no
    ``car_track`` is stated. The refusal moved to where the fields are; it did
    not go away.
    """
    def build(mission_kwargs, flags, bounds_overrides):
        from . import carwing_multi
        if mission_kwargs:
            # same refusal, same reason, as the single-element family: this
            # problem has no MissionSpec, and dropping W_N/altitude silently
            # would let the Mission card look like it did something
            raise ValueError(
                "the two-element car rear wing has no mission spec — set its "
                "speed with flags={'V': ...}; weight and altitude do not "
                "enter")
        # the size bands, innermost-first: the family's published defaults,
        # then the DESIGN BOX's own b_m / S_m2 rows, which are the only place
        # a user states either (_CAR_SIZE_ROWS says what that fixed)
        area_kw = ({**_car_registry_defaults(flags), **_car_area_defaults()}
                   if area_free else {})
        area_kw.update(_car_size_band_kwargs(bounds_overrides))
        prob = carwing_multi.CarWingMultiProblem(
            **{**_car_wing_multi_kwargs(flags), **area_kw},
            **_chord_kwargs(chord_order, flags))
        labels = _with_chord_labels(
            _CAR_WING_MULTI_AREA_LABELS if area_free
            else _CAR_WING_MULTI_LABELS, chord_order, flags)
        bounds = _apply_overrides(prob.bounds, labels, bounds_overrides)
        return _BuiltProblem(
            problem=prob,
            callable=lambda x: carwing_multi.fg_car_wing_multi(x, prob),
            evaluate=lambda x: carwing_multi.evaluate_car_wing_multi(x, prob),
            bounds=bounds, is_constrained=True, dim=prob.dim,
            param_labels=labels, medium="air",
        )
    # this family GATES the aspect ratio per candidate
    # (``sizing.check_ar_limit``), so the registry may declare the
    # limit flags on it: a flag is declared where it is READ
    build.gates_ar = True
    return build


#: ...and the same for the slotted family (see ``_build_car_wing``).
_build_car_wing_multi = _make_car_wing_multi_builder(area_free=True)


#: car-wing-with-designed-endplates configuration keys accepted through
#: ``flags``. All VALUE changes again — the endplate's GEOMETRY (chord,
#: thickness, toe) is in the design vector, while its section FAMILY, the
#: attachment height, the material and the load case are configuration.
CAR_ENDPLATE_KEYS = ("mount", "section", "deck_height_m", "CD_budget",
                     "deflection_limit_m", "endplate_deflection_limit_m",
                     "E_endplate_pa", "yaw_deg", "junction_drag",
                     "blend_frac", "blend_shape", "V", "ei_nm2",
                     # no span/area band keys — see _CAR_CORE_KEYS
                     # the plate's chord in metres: the second of its two
                     # bands (the first is the design box's ratio row), and
                     # configuration for the same reason the wing's
                     # chord_limits are — a box the part must fit is a
                     # statement about the car, not a design variable
                     "endplate_chord_min_m", "endplate_chord_max_m",
                     # the same three the plain car wing takes, and they mean
                     # the same things here: WHICH SCALAR is maximised
                     # (carwing.CAR_OBJECTIVES) and the two optional budgets
                     # that go with it, none of which touches the vector
                     "car_objective", "drag_budget_n", "downforce_min_n",
                     # the plate's CANT, as the other two families take it.
                     # NOT endplate_chord_follows: on this family the plate's
                     # chord is a design row (endplate_chord_ratio), and a
                     # second answer to a question the search is answering is
                     # exactly what check_flags exists to refuse.
                     "endplate_cant_deg")

_CAR_ENDPLATE_STR_KEYS = ("mount", "section", "blend_shape", "car_objective")


def _car_endplate_kwargs(flags: dict | None) -> dict:
    """CarWingEndplateProblem kwargs from ``flags`` (empty in -> defaults)."""
    if not flags:
        return {}
    out: dict = {}
    for k in CAR_ENDPLATE_KEYS:
        v = flags.get(k)
        if v is None:
            continue
        field = _CAR_KEY_ALIASES.get(k, k)
        if k in _CAR_ENDPLATE_STR_KEYS:
            out[field] = str(v)
        elif k == "junction_drag":
            out[field] = bool(v)
        elif k == "CD_budget":
            out[field] = _car_budget(v)
        else:
            out[field] = float(v)
    return out


def _car_endplate_defaults(flags: dict | None) -> dict:
    """The designed-endplate family's opening configuration.

    :func:`_car_registry_defaults`'s twin, and identical to it today: the two
    families used to differ here only because they carried different published
    drag allowances (0.13 at 0.4 m^2 = 96.3 N against the plain wing's 0.11 =
    81.5 N), and neither carries one by default any more. Kept as a separate
    function rather than aliased so that a future departure has somewhere to
    live and is visible in the diff when it appears.
    """
    return _car_registry_defaults(flags)


def _plate_coords_kwargs(flags: dict | None) -> dict:
    """The plate section's COORDINATES, where the chosen section has them.

    Stiffness only: ``endplate.section_inertia_factor`` integrates k_I from
    the outline, so a designed plate's bending inertia is its own shape's
    rather than a construction family's constant. A polar is a table of
    forces and carries no shape, which is why this cannot be read off the one
    :func:`_section_kwargs` already built — and why the fallback is NAMED in
    the breakdown instead of being silent.

    Empty in, empty out: a plate with no chosen section keeps the published
    build-up exactly.
    """
    value = (flags or {}).get(SECTION_PLATE_KEY)
    if not value:
        return {}
    try:
        coords = chosen_section_coords(value)
    except Exception:                       # noqa: BLE001 — stiffness only
        return {}
    return {} if coords is None else {"endplate_coords": coords}


def _make_car_endplate_builder(chord_order: int = 0, area_free: bool = False,
                               cant_free: bool = False,
                               blend_free: bool = False):
    def build(mission_kwargs, flags, bounds_overrides):
        from . import endplate
        if mission_kwargs:
            raise ValueError(
                "the car rear wing has no mission spec — set its speed with "
                "flags={'V': ...}; weight and altitude do not enter")
        area_kw = ({**_car_endplate_defaults(flags),
                    **_car_area_defaults()} if area_free else {})
        area_kw.update(_car_size_band_kwargs(bounds_overrides))
        free_kw = {k: True for k, on in (("cant_free", cant_free),
                                         ("blend_free", blend_free)) if on}
        prob = endplate.CarWingEndplateProblem(
            **{**_car_endplate_kwargs(flags), **area_kw}, **free_kw,
            **_chord_kwargs(chord_order, flags),
            **_section_kwargs(flags, field="section_polar"),
            # ...and the PLATE's own aerofoil, in its own slot. Resolved by
            # the same one function every other chosen section goes through,
            # so a library name, a named section AT A POINT and a designed
            # CST pair all reach the plate the way they reach a wing.
            **_section_kwargs(flags, field="endplate_polar",
                              key=SECTION_PLATE_KEY),
            **_plate_coords_kwargs(flags))
        # ASK THE PROBLEM for its own row order. The static tuples below are
        # the FIXED variant's and are kept as the registry's declaration, but
        # a variant that adds rows in the MIDDLE of the family block cannot be
        # described by appending to them — and two sources for one order is
        # the transposition defect `_mission_mode_labels` records.
        labels = tuple(prob.param_labels)
        bounds = _apply_overrides(prob.bounds, labels, bounds_overrides)
        return _BuiltProblem(
            problem=prob,
            callable=lambda x: endplate.fg_car_wing_endplate(x, prob),
            evaluate=lambda x: endplate.evaluate_car_wing_endplate(x, prob),
            bounds=bounds, is_constrained=True, dim=prob.dim,
            param_labels=labels, medium="air",
        )
    build.takes_section = True
    # ...and the PLATE is a surface of its own, with its own section slot
    build.takes_section_plate = True
    # this family GATES the aspect ratio per candidate
    # (``sizing.check_ar_limit``), so the registry may declare the
    # limit flags on it: a flag is declared where it is READ
    build.gates_ar = True
    return build


#: ...and for the designed-endplate family. Note this one's area is the
#: variable that makes the plate's REACH bite hardest — a smaller wing is a
#: shorter chord and the plate still has to get down to the deck — so freeing
#: it unconditionally is what puts that constraint in play on every run.
_build_car_endplate = _make_car_endplate_builder(area_free=True)


#: tail/elevator configuration keys accepted through ``flags``. These are
#: VALUE changes only — they never alter the design vector's dimension,
#: which is what keeps ProblemSpec.param_labels (a static attribute) honest.
#: The fixed-arm layout DOES change the dimension and therefore gets its own
#: ProblemSpec instead of a flag.
#: THE FIN'S OWN SHAPE, as stated values. Absent on all three means the
#: published sizing law (fin.V_V_DEFAULT / AR_VT_DEFAULT / FIN_TC_DEFAULT),
#: so every stored run is bit-for-bit.
#:
#: Declared with the tail's other configuration keys because they answer the
#: same question — what the second surface IS — and because they must travel
#: to every consumer together: the drag book that charges the fin, the
#: lateral deck that flies it, the report that states it, the CAD that lofts
#: it.
#:
#: it. What they do NOT say is whether there is a fin at all — that is
#: :data:`FIN_PRESENCE_KEY`, beside them.
FIN_SHAPE_KEYS = ("fin_volume_coeff", "fin_ar", "fin_tc")

#: IS THERE A FIN AT ALL. ``True`` on every problem, so an untouched run is
#: bit-for-bit; the shells send it only to say ``False``.
#:
#: It used to be argued that this key should not exist: "there is no separate
#: 'is there one' key: the LAYOUT answers that (``fin.size_fin`` returns None
#: for a V-tail and every consumer follows), which keeps one question in one
#: place." That was true while a V-tail was the only way to answer no. Stage
#: 1 now asks the user directly ("add a vertical stabiliser (fin and
#: rudder)") — and with no key to travel on, that answer reached NOTHING.
#: Measured on `tail [designed tail] + free chord law` at its box centre with
#: the fin switched OFF: ``cd0_fin`` still 0.0008848 (5.79 % of L/D), a full
#: ``geometry["fin"]`` block still drawn, 82 % of the empennage weight book
#: still charged, and ``Cn_beta`` 0.1133 in the rebuilt deck while the card
#: that asked the question said it was "exactly zero — not small".
#:
#: One question in one place is still the rule. The place is
#: ``fin.has_fin``, which reads the layout AND this key, so no consumer
#: checks the two in a different order.
FIN_PRESENCE_KEY = "fin"

#: HOW FAR AFT OF THE REAR WING A PAIR'S FIN STANDS [m] — the BOOM.
#:
#: A TANDEM-ONLY key, and it is a key at all because a pair is the one
#: layout with nowhere obvious to put a vertical surface: a conventional
#: aeroplane has a body that ends somewhere, and a pair has two wings and a
#: gap. That station was a constant nobody chose for as long as this family
#: has had a fin — first the midpoint between the wings, then the rear
#: wing's root — and both readings are decisions about the AIRFRAME, which
#: is the builder's to make and not the package's.
#:
#: Absent is ``fin.TANDEM_FIN_BOOM_FRAC`` of the stagger (the measured table
#: is on that constant); ``0`` is the fin ON the rear wing's root, which is
#: the station this key was added over, kept reachable rather than deleted.
#: It is a VALUE flag — it moves no design row and changes no dimension.
TANDEM_FIN_BOOM_KEY = "fin_boom_m"

#: ``charge_fin_drag`` is NOT here any more: the fin's parasite drag is
#: charged the way the tailplane's is, always, so there is no flag to carry
#: and a stored config that names one is refused rather than silently
#: honoured as a switch that no longer exists.
TAIL_CONFIG_KEYS = ("tail_type", "dihedral_deg",
                    "control", "elevator_chord_frac",
                    "delta_e_max_deg") + FIN_SHAPE_KEYS + (FIN_PRESENCE_KEY,)

#: THE WING'S OWN CANT AND SWEEP [deg], as stated values.
#:
#: Not to be confused with ``dihedral_deg`` above, which is the TAIL's cant
#: (what makes a V-tail). These two are the wing's, and they are declared
#: ONLY on lattice-backed families (``wingtail.py``): a lifting line has no
#: out-of-plane geometry, so on the published `tail` family they would be
#: flags the objective cannot read — which :func:`check_flags` refuses, and
#: rightly.
#:
#: Both are geometry in the lattice (``geometry.dihedral_rotate`` and the
#: quarter-chord sweep offset in ``vlm.VLM``), so they are SCORED: measured
#: on the wing+tail box centre, 5 deg of dihedral costs 0.35 % of L/D and
#: 15 deg of sweep costs 12.7 % while moving x_np from 0.443 to 1.014 m.
WING_CANT_KEYS = ("wing_dihedral_deg", "wing_sweep_deg")

#: FUSELAGE DIAMETER [m] — the one number that makes a SEARCHED wing-to-tail
#: separation a well-posed question.
#:
#: Unstated, the fuselage is free and the arm is a RATCHET: measured on the
#: `tail` family at b = 10 m, L/D climbs monotonically 25.41 -> 32.57 from a
#: 1 m arm to a 100 m one, so the optimum is wherever the box edge is. Stated,
#: ``drag.fuselage_cd0`` charges the body the arm implies and the trade
#: acquires an interior optimum inside the family's own calibrated band.
#:
#: It is REQUIRED only where the arm is a design variable
#: (:func:`arm_is_searched`) — a fixed arm is a decision already taken, and a
#: constant added to CD cannot move a variable nobody is varying. Everywhere
#: else it is offered and never demanded. See PLAN_SESSION66_SEPARATION.md.
FUSELAGE_KEYS = ("fuselage_diameter_m",)


def arm_is_searched(name: str) -> bool:
    """Is the wing-to-tail separation a DESIGN VARIABLE in this family?

    The discriminator for whether a fuselage diameter has to be asked for. A
    property of the registered problem, read off its own parameter labels
    rather than inferred from its name — `tail` searches the arm and
    `tail (fixed arm)` states it, and a TANDEM never searches its stagger
    (``dx`` is a flag in every variant), so a pair is never asked.
    """
    spec = PROBLEM_SPECS.get(name)
    return bool(spec) and "l_t_m" in tuple(spec.param_labels)


def searched_cant(name: str) -> str:
    """WHICH of the wing's cant rows this family searches, as one of the
    :data:`WING_CANTS` keys.

    The discriminator a shell needs to decide, per row, whether to offer the
    stated flag (:data:`WING_CANT_KEYS`) or point at the design box. Read off
    the registered problem's own parameter labels, never off its NAME or a
    kept list: the free-cant twin of every wing+tail configuration is
    generated, the modifier variants of those twins carry the rows too, and
    a name is a label rather than a contract.
    """
    spec = PROBLEM_SPECS.get(name)
    if spec is None:
        return "fixed"
    labels = set(spec.param_labels)
    have = {row for row in _CANT_ROW_OF.values() if row in labels}
    for key in WING_CANTS:
        if set(searched_cant_keys(key)) == have:
            return key
    return "fixed"                              # unreachable: four subsets


def cant_is_searched(name: str) -> bool:
    """Is EITHER of the wing's cant rows a design variable in this family?

    Any, not both: since the axis split into four states a family can search
    the dihedral and state the sweep, or the reverse. Three callers mean the
    DIHEDRAL specifically — the unpriced floor on its sign, the "the run
    cannot reach it" band check and the anhedral note — and they ask
    :func:`dihedral_is_searched` instead.
    """
    return searched_cant(name) != "fixed"


def dihedral_is_searched(name: str) -> bool:
    """Is ``wing_dihedral_deg`` a design variable in this family?"""
    return WING_CANT_KEYS[0] in searched_cant_keys(searched_cant(name))


def sweep_is_searched(name: str) -> bool:
    """Is ``wing_sweep_deg`` a design variable in this family?"""
    return WING_CANT_KEYS[1] in searched_cant_keys(searched_cant(name))


def _fuselage_kwargs(flags: dict | None) -> dict:
    """``fuselage_diameter_m`` -> the problem's field. Empty in, empty out,
    so a family that never states one charges no fuselage at all."""
    if not flags:
        return {}
    v = flags.get(FUSELAGE_KEYS[0])
    return {} if v is None else {"fuselage_diameter_m": float(v)}


#: which way a tip device ON THE SECOND SURFACE may point
#: (``wingtail.TAIL_WINGLET_DIRECTIONS``: up / down / either). A VALUE flag
#: like the keys above — it moves one bounds ROW, never the dimension — but
#: it needs a translation (a named direction to a cant band), so it is not
#: one of them. Absent, and every family keeps its own published band: the
#: aircraft tail's is up, the hydrofoil elevator's is already signed.
TAIL_WINGLET_DIR_KEY = "tail_winglet_dir"

#: ...and WHAT SHAPE it is, in the same vocabulary the wing's tip device
#: uses (``objective.WINGLET_TYPES``: canted / vertical fence / raked). Also
#: a VALUE flag — it narrows the same one bounds row to that type's own cant
#: MAGNITUDE band, keeping whichever side the direction chose — so a tip
#: device on the second surface can be asked the same question the wing's is
#: asked instead of being an unshaped yes/no. Absent = ``canted``, the band
#: every published run flies.
TAIL_WINGLET_TYPE_KEY = "tail_winglet_type"

#: ...and its own BLEND, where the second surface answers the shape question
#: for itself. Absent = the design's blend (:data:`WINGLET_BLEND_KEY`), which
#: is what one control saying "build the tip devices blended" means; present,
#: the two surfaces carry different transitions, which is what a tip-device
#: menu PER SURFACE asks for.
TAIL_WINGLET_BLEND_KEY = "tail_winglet_blend_frac"

#: every VALUE that shapes a tip device on the SECOND SURFACE. Declared
#: together, on exactly the families whose second surface carries one.
TAIL_TIP_KEYS = (TAIL_WINGLET_DIR_KEY, TAIL_WINGLET_TYPE_KEY,
                 TAIL_WINGLET_BLEND_KEY)


def _tail_blend_kwargs(flags: dict | None) -> dict:
    """``tail_winglet_blend_frac`` flag -> ``tail_blend_frac_fixed`` kwarg."""
    if not flags or flags.get(TAIL_WINGLET_BLEND_KEY) in (None, ""):
        return {}
    return {"tail_blend_frac_fixed": float(flags[TAIL_WINGLET_BLEND_KEY])}


def _tail_cant_kwargs(flags: dict | None) -> dict:
    """``{"tail_cant_bounds": band}`` for a named direction and type, else
    ``{}``."""
    if not flags:
        return {}
    direction = flags.get(TAIL_WINGLET_DIR_KEY)
    wtype = flags.get(TAIL_WINGLET_TYPE_KEY)
    if direction is None and wtype in (None, "canted"):
        return {}
    from . import wingtail

    band = wingtail.tail_cant_bounds_for(
        None if direction is None else str(direction),
        None if wtype is None else str(wtype))
    out = {} if band is None else {"tail_cant_bounds": band}
    if direction is not None \
            and str(direction) == wingtail.FOLLOW_DIRECTION:
        # the band is the magnitude; the SIDE is derived per candidate
        out["tail_winglet_follow"] = True
    return out


#: which side a NARROWED tip device sits on, on the MAIN FOIL under water.
#:
#: In air the wing's cant band is unsigned (60-90) and a shape that narrows
#: it — a vertical fence, |cant| 84-90 — needs no side. Under water the
#: band is signed (-90…90: + points the device up towards the free surface,
#: − down away from it), so the same shape is two bands with a hole between
#: them and one bounds row has to pick. It picks DOWN, which is the side
#: with the cavitation margin: a device canted up sits in less static head
#: and cavitates first (hydrofoil.py's own note beside the free-surface
#: image). A user who wants the search to choose the side asks for the
#: CANTED shape, which keeps the whole signed band — nothing is taken away.
WATER_TIP_DIRECTION = "down"


def _foil_cant_kwargs(flags: dict | None) -> dict:
    """``{"winglet_cant_bounds": band}`` for a stated device TYPE in water.

    The water twin of :func:`_tail_cant_kwargs`, and it narrows nothing
    unless a type was actually asked for: ``None``/``canted`` IS the
    published band, so an untouched run is bit-for-bit unchanged.
    """
    wtype = (flags or {}).get("winglet_type")
    if wtype in (None, "canted"):
        return {}
    from . import wingtail

    band = wingtail.tail_cant_bounds_for(WATER_TIP_DIRECTION, str(wtype))
    return {} if band is None else {"winglet_cant_bounds": band}


#: THE LAYOUT the second surface is trimmed in, as stated values: where the
#: CG sits and how far the surface is from the wing. Both are values (one
#: number each, no design variable moves), and both decide the SIGN of what
#: the trim solve puts on that surface — a CG ahead of the wing's own centre
#: of lift needs a DOWNLOAD aft to balance it, a CG behind it needs lift.
#: Which is why they belong to the user: the device direction, the section's
#: camber and the elevator's deflection all follow from that sign, and every
#: family's own value is a CALIBRATION (tail.X_CG_BY_TYPE,
#: hydrotail.X_CG_FRAC) chosen so the static-margin boundary crosses its
#: design box — a sensible default, never a statement about the aircraft
#: being designed.
TAIL_CG_KEY = "x_cg_m"
TAIL_HEIGHT_KEY = "z_t_m"

#: ...and WHICH WAY UP the second surface is MOUNTED — the third statement
#: about the same balance, and the one the geometry view draws. A cambered
#: section mounted upside down works its camber on the side the surface is
#: actually loaded on; ``tail.tail_polar`` decides that from the load the
#: rest of the aircraft asks for, which is the right default and is not the
#: only defensible answer. A builder who has already cut the stabiliser, or
#: who wants the two orientations compared at one design, is stating a
#: value, not overriding physics — the trim solve is unchanged, it is the
#: SECTION that mirrors, and the resulting load is reported either way.
#:
#: ``auto`` (absent, and every published run bit-for-bit) follows the load;
#: ``upright`` and ``inverted`` state it. Where the stated mounting and the
#: trimmed load disagree, that is a real aeroplane — a surface flying
#: against its own camber — and the shells say so rather than refusing it.
#:
#: The values name the MOUNTING and not the load, because on this balance
#: they are not the same question: the surface's own camber couple is one of
#: the two authors of its load (tail.section_moment), so mirroring the
#: section can mirror the load it is then asked for. Small tail, small
#: effect; a stabiliser oversized against its wing, and the mounting decides
#: the sign. A control labelled "make it push down" would therefore be
#: lying at exactly the designs where the answer matters.
#: what the second surface's own SPAN and CHORD may measure, in metres
#: (``tail.TailLimits``). The design box states that surface as an AREA and
#: an aspect ratio — the solver's variables — and neither is the question a
#: builder has: a fuselage is only so wide, a mould only so long. Both
#: directions are exact (b = sqrt(AR S), c = sqrt(S/AR)), so a span band and
#: a chord band ARE an area band and an aspect-ratio band, and stating them
#: narrows the SEARCHED box rather than only refusing candidates out of it.
#:
#: All four optional and all four off by default, which is the published
#: behaviour exactly. They are limits, never a resize: a design that breaks
#: a live one is refused (it scores the penalty), never quietly reshaped —
#: the rule ``geometry.ChordLimits`` already follows on the wing.
TAIL_LIMIT_KEYS = ("tail_span_min_m", "tail_span_max_m",
                   "tail_chord_min_m", "tail_chord_max_m")


def tail_limits_of(flags: dict | None):
    """The ``TailLimits`` these flags ask for, or None where they ask none.

    The public twin of :func:`chord_limits_of`, and for the same reason: a
    shell that has just been typed into needs to ask whether the pair it now
    holds is a pair at all (``TailLimits`` refuses a maximum below its
    minimum), and it must ask it of the SAME constructor the problem builder
    uses. Reading the flags a second way is how a shell comes to accept a
    band the run then raises on.
    """
    if not flags:
        return None
    got = {k: flags[k] for k in TAIL_LIMIT_KEYS
           if flags.get(k) is not None}
    if not got:
        return None
    from . import tail as _tailmod
    return _tailmod.TailLimits(
        b_min_m=got.get("tail_span_min_m"),
        b_max_m=got.get("tail_span_max_m"),
        c_min_m=got.get("tail_chord_min_m"),
        c_max_m=got.get("tail_chord_max_m"))


def _tail_limits_kwargs(flags: dict | None) -> dict:
    """:data:`TAIL_LIMIT_KEYS` -> the ``tail_limits`` problem kwarg.

    Empty in, empty out: nothing stated builds no TailLimits at all, so the
    problem is constructed with exactly its published arguments.
    """
    lim = tail_limits_of(flags)
    return {} if lim is None else {"tail_limits": lim}


TAIL_MOUNT_KEY = "tail_mount"
TAIL_MOUNT_AUTO = "auto"
#: mounting -> ``tail_inverted`` on every family's problem (None = follow)
TAIL_MOUNTS = {TAIL_MOUNT_AUTO: None, "upright": False, "inverted": True}


def _tail_mount_kwargs(flags: dict | None) -> dict:
    """``tail_mount`` flag -> ``tail_inverted`` kwarg. Empty in, empty out."""
    if not flags:
        return {}
    want = flags.get(TAIL_MOUNT_KEY)
    if want in (None, "", TAIL_MOUNT_AUTO):
        return {}
    key = str(want)
    if key not in TAIL_MOUNTS:
        raise ValueError(
            f"{TAIL_MOUNT_KEY}={want!r} is not a mounting; choices are "
            f"{sorted(TAIL_MOUNTS)} — {TAIL_MOUNT_AUTO} lets the trim "
            f"balance decide, which is the published default")
    return {"tail_inverted": TAIL_MOUNTS[key]}


def _tail_layout_kwargs(flags: dict | None) -> dict:
    """CG / vertical separation from ``flags``, in each family's own frame.

    ``x_cg_m`` is metres aft of the WING's aerodynamic centre in air (the
    frame ``tail.X_CG_BY_TYPE`` is quoted in) and metres aft of the foil's in
    water; ``z_t_m`` is the second surface's height above the wing plane in
    air and BELOW the foil (negative) in water. Absent, each family keeps the
    layout it published.
    """
    out: dict = {}
    if not flags:
        return out
    if flags.get(TAIL_CG_KEY) is not None:
        out["x_cg"] = float(flags[TAIL_CG_KEY])
    if flags.get(TAIL_HEIGHT_KEY) is not None:
        out["z_t_fixed"] = float(flags[TAIL_HEIGHT_KEY])
    out.update(_tail_mount_kwargs(flags))
    out.update(_tail_limits_kwargs(flags))
    return out


#: THE CRAFT'S WEIGHT [N] — the lift the water families are trimmed to
#: carry. It belongs beside :data:`TAIL_CG_KEY` and not with the geometry
#: because the two are the same kind of statement: how much load there is,
#: and where it acts. Together they are the whole of what the balance is
#: written in, and neither is a shape.
#:
#: WHY IT IS A VALUE AND NOT A DESIGN VARIABLE. Every other number the water
#: box carries is something the optimiser may choose — the speed, the depth,
#: the taper, the stabiliser's area. The weight is not: it is the craft the
#: foil is being designed FOR, decided before the search starts by the rider,
#: the board and the rig. An optimiser handed the weight would simply set it
#: to zero, because ``CL_target = W / (q S)`` and no lift is the cheapest
#: lift there is. So it enters as a stated value, exactly as ``x_cg_m`` does.
#:
#: WHY IT MATTERS, measured. The published water families carry
#: ``L_design = 6000 N`` (hydrofoil.HydrofoilProblem), a ~600 kg foiling
#: dinghy, on a 1.2 m / 0.144 m2 foil. A windfoil, wingfoil or kitefoil flies
#: at 780-1230 N. Since ``b_m``/``S_m2`` became flags, a user could type
#: their own 0.90 m / 0.090 m2 board foil in — and get it flown at 6000 N,
#: i.e. W/S = 66.7 kPa and CL_target(12 m/s) = 0.903, a craft 5.8x
#: overloaded. The resize flags are therefore actively misleading without
#: this one: the size is asked and the load it carries is not.
#:
#: ABSENT: each family keeps its published ``L_design``, so every frozen
#: water study reproduces bit-for-bit.
WEIGHT_KEY = "weight_n"


def _weight_kwargs(flags: dict | None) -> dict:
    """The stated craft weight -> ``L_design``; ``{}`` when nothing was said.

    Empty in, empty out, the rule every value flag in this module follows: a
    build with no ``weight_n`` constructs the family's published problem with
    exactly its published arguments.

    A non-positive weight is REFUSED rather than clamped, and that is a
    refusal of nonsense, not a cap on the user's number. Zero weight makes
    ``CL_target = W/(q S)`` identically zero, so the trim solve is answering
    a different question from the one the objective scores, and a negative
    weight asks the foil to hold the craft down. There is deliberately NO
    upper bound: a heavy craft is a hard design (the cavitation margin is
    what refuses it, at the panel where it actually fails), not one this
    function is entitled to decline — the repo's own rule that a calibration
    is a default and never a ban.

    A NON-FINITE weight is refused first, by name, because ``inf > 0.0`` is
    True and the positivity test alone waves it straight through: an
    infinite ``L_design`` builds a problem whose ``CL_target`` is infinite at
    every speed, so the trim solve has nothing to converge to and the
    cavitation margin — the thing that is supposed to price a heavy craft —
    never gets a number to price. Both of this function's siblings already
    refuse non-finite by name (``rig.RigLoads.__post_init__`` and
    ``hydrofoil._operating_row``); this is the third statement of the same
    rule, and it is a refusal of a number the solve cannot mean, not a cap:
    every finite weight is still accepted, however large.
    """
    out: dict = {}
    if not flags:
        return out
    raw = flags.get(WEIGHT_KEY)
    if raw in (None, ""):
        return out
    try:
        w = float(raw)
    except (TypeError, ValueError):
        # ...before the two crafted refusals below, a bare float() raised
        # "could not convert string to float: 'heavy'" — a message that names
        # neither the flag nor what a weight is, for the most likely mistake
        # of the three.
        raise ValueError(
            f"flags[{WEIGHT_KEY!r}] = {raw!r} is not a number. It is the "
            f"craft's weight in NEWTONS — rider, board, rig and foil — so it "
            f"is one positive figure, not a mass, a name or a range.") \
            from None
    if not math.isfinite(w):
        raise ValueError(
            f"flags[{WEIGHT_KEY!r}] = {raw!r} is not a finite number. It is "
            f"the lift the foil is trimmed to carry, so an infinite or NaN "
            f"weight makes CL_target = W/(q S) infinite or NaN at every "
            f"speed and there is no trim to solve. This is a refusal of a "
            f"number the solve cannot mean, not a cap — every finite weight "
            f"is accepted, and the cavitation margin is what refuses a "
            f"heavy one.")
    if not (w > 0.0):
        raise ValueError(
            f"flags[{WEIGHT_KEY!r}] = {raw!r} is not a weight: it must be "
            f"> 0 N. It is the lift the foil is trimmed to carry, so zero "
            f"asks for a craft with no design lift and a negative value asks "
            f"the foil to hold the craft down. No upper bound is imposed — a "
            f"heavier craft is a harder design, and the cavitation margin is "
            f"what refuses it.")
    return {"L_design": w}


#: THE RIG's loads, as stated values — the set of flags that turns a bare
#: appendage into a driven craft (``rig.RigLoads``, which carries the physics
#: and the measurements; this is only the wiring).
#:
#: They sit here beside :data:`WEIGHT_KEY` because they answer the same kind
#: of question — how much load there is and where it acts — and because a
#: weight without a rig is exactly the craft this package could already
#: describe: one that carries its own weight and is pushed by nothing.
#:
#: ``rig_ce_height_m`` is the LEVER (centre-of-effort height above the
#: waterline) and it is REQUIRED as soon as any of the others is given. That
#: is not a cap on a user's number, it is a refusal to invent one: the couple
#: is thrust times ``(depth + z_ce)``, published centre-of-effort heights run
#: from 0 m (a tow rope at the surface) to ~2.75 m (a windsurf rig), and
#: defaulting it would silently set the size of the craft's single largest
#: pitching moment. The same rule the arm and the CG follow — a layout number
#: is asked, never assumed.
#:
#: ``rig_thrust_n`` is optional on purpose: absent, the drive is CLOSED from
#: the craft's stated lift-to-drag ratio (``craft_lod``, default
#: ``rig.CRAFT_LOD_DEFAULT`` = 7.0). Read ``rig.py`` before passing this
#: engine's own appendage L/D there — it is 3-4x the two measured
#: whole-appendage values and would understate the couple by that factor.
#:
#: ABSENT (none of them given): no rig at all, ``rig=None``, and every
#: published water study reproduces bit-for-bit.
RIG_CE_HEIGHT_KEY = "rig_ce_height_m"
RIG_THRUST_KEY = "rig_thrust_n"
RIG_SIDE_KEY = "rig_side_n"
RIG_VERTICAL_KEY = "rig_vertical_n"
RIG_CE_X_KEY = "rig_ce_x_m"
RIG_CRAFT_LOD_KEY = "craft_lod"

#: The whole set, in one tuple so a spec declares them the way it declares
#: ``TAIL_TIP_KEYS`` — as a group. A rig stated one flag at a time is still
#: one statement, and a family that honours any of them honours all of them.
RIG_KEYS = (RIG_CE_HEIGHT_KEY, RIG_THRUST_KEY, RIG_SIDE_KEY,
            RIG_VERTICAL_KEY, RIG_CE_X_KEY, RIG_CRAFT_LOD_KEY)

#: flag key -> the ``rig.RigLoads`` field it fills. Written as a table rather
#: than six ``if``s so that the two directions (what a shell may say, what the
#: value object accepts) cannot drift apart, and so a new field is one line.
_RIG_FIELDS = {
    RIG_CE_HEIGHT_KEY: "z_ce_m",
    RIG_THRUST_KEY: "thrust_n",
    RIG_SIDE_KEY: "side_n",
    RIG_VERTICAL_KEY: "vertical_n",
    RIG_CE_X_KEY: "x_ce_m",
    RIG_CRAFT_LOD_KEY: "craft_lod",
}


def _rig_kwargs(flags: dict | None) -> dict:
    """The rig flags -> ``{"rig": RigLoads(...)}``; ``{}`` when none is given.

    Empty in, empty out, the rule every value flag in this module follows —
    and here it is load-bearing rather than tidy: ``rig=None`` is the ONLY
    thing that keeps the couple, the vertical relief and the report block out
    of a published run, so a helper that returned a zero-valued ``RigLoads``
    would change every frozen water number by an amount that happens to be
    zero today and would not stay zero the first time a default moved.

    The one refusal: a rig force with no centre-of-effort height. It is
    raised here rather than in ``rig.RigLoads`` because this is where the
    incompleteness exists — the value object requires the height as a
    positional field and cannot be built without it, so without this message
    a shell that offered only a thrust box would get a ``TypeError`` naming a
    Python argument instead of a physical omission.

    Values are refused by ``rig.RigLoads.__post_init__`` (non-finite, a
    centre of effort under the water, a non-positive craft L/D) and by
    ``HydrofoilTailProblem.__post_init__`` (a vertical share at or above the
    craft's weight), each with the key and the value named. Nothing is
    clamped and nothing has an upper bound.
    """
    if not flags:
        return {}
    given = {k: flags[k] for k in RIG_KEYS
             if flags.get(k) not in (None, "")}
    if not given:
        return {}
    if RIG_CE_HEIGHT_KEY not in given:
        raise ValueError(
            f"flags {sorted(given)} state a rig but not "
            f"flags[{RIG_CE_HEIGHT_KEY!r}], the height of its centre of "
            f"effort above the waterline. That height IS the lever: the rig "
            f"couple is the drive force times (foil depth + this height), so "
            f"a rig stated without it has no moment and would be accepted "
            f"and silently ignored. Published values run from 0 m (a tow rope "
            f"at the surface) to about 2.75 m (a windsurf rig), which is why "
            f"it is asked rather than assumed.")
    from .rig import RigLoads

    return {"rig": RigLoads(**{_RIG_FIELDS[k]: float(v)
                               for k, v in given.items()})}


def tail_winglet_direction(problem_name: str) -> str | None:
    """Which way THIS family's tail tip device may point, as a named
    direction — or ``None`` where it carries no such device.

    Read off the family's OWN default box and matched against
    ``wingtail.TAIL_WINGLET_DIRECTIONS``, never from a table restated here:
    the aircraft tail opens on ``up`` and the hydrofoil's elevator on
    ``either``, and a shell that hard-coded that pair would be wrong the day
    a family changed its band.
    """
    from . import wingtail

    box = PROBLEM_SPECS[problem_name].default_bounds.get("winglet_cant_t_deg")
    if box is None:
        return None
    for name, band in wingtail.TAIL_WINGLET_DIRECTIONS.items():
        if (float(box[0]), float(box[1])) == (float(band[0]), float(band[1])):
            return name
    return None


def _wing_cant_kwargs(flags: dict | None) -> dict:
    """``wing_dihedral_deg`` / ``wing_sweep_deg`` -> the problem's fields.

    Empty in, empty out — so a family that never states them builds the
    planar unswept wing every published run flew, bit-for-bit.
    """
    if not flags:
        return {}
    return {k: float(flags[k]) for k in WING_CANT_KEYS
            if flags.get(k) is not None}


def _tail_config(flags: dict | None) -> dict:
    """Tail configuration subset of ``flags`` (unknown keys are ignored —
    the physics flags travel in the same dict)."""
    if not flags:
        return {}
    return {k: flags[k] for k in TAIL_CONFIG_KEYS
            if k in flags and flags[k] is not None}


#: THE WATER CRAFT'S VERTICAL SURFACE, declared where it is read. A boat's
#: is the MAST (``fin.mast``), not a volume-coefficient tail, so it takes
#: only two of the four keys: is there one (``fin.has_fin`` gates the mast
#: drag every water run already pays) and what SECTION it flies. There is no
#: volume coefficient and no aspect ratio to state — a strut's size is the
#: submergence depth and the mast chord, both of which the design already
#: carries.
#: ``c_mast`` IS ITS CHORD, and it is here because a strut that is DRAWN at
#: one chord and CHARGED at another is the defect ``fin.py`` exists to have
#: fixed, one medium over. FoilingBO places a 0.115 m windfoil mast and the
#: engine charged the family default of 0.08 m — a 44 % error straight into
#: the wetted area, and with ``strut_model`` on it is also the strut's
#: Reynolds number, its aspect ratio and the side force it can carry. Every
#: water problem carries the field, so unlike the station keys below this one
#: is declared on all of them.
STRUT_KEYS = ("fin_tc", "c_mast", FIN_PRESENCE_KEY)


#: ...AND WHERE THE STRUT STANDS, on the family that has somewhere to put
#: it. A station only means something between two surfaces, so these keys
#: are declared on the ELEVATOR families alone: the single-foil problems
#: have no fuselage, no ``x_mast_frac`` field, and a flag declared there
#: would reach a constructor that cannot take it.
#:
#: ``strut_model`` is the switch back to the published wetted-area charge
#: (``hydrofoil._mast_cd0``) — kept as a flag rather than deleted, because
#: every recorded hydrotail number was scored on it and a study that wants
#: to reproduce one has to be able to ask for it.
STRUT_STATION_KEYS = ("x_mast_frac", "free_mast_station",
                      "x_mast_frac_bounds", "strut_model")


def _strut_station_config(flags: dict | None) -> dict:
    """The strut's STATION keys for an elevator-family water problem.

    Separate from :func:`_strut_config` because the two groups have
    different homes: every water family answers "is there a strut and what
    section is it", and only a family with a second surface can answer
    "where does it stand".
    """
    if not flags:
        return {}
    return {k: flags[k] for k in STRUT_STATION_KEYS
            if k in flags and flags[k] is not None}


def _strut_config(flags: dict | None) -> dict:
    """The STRUT's subset of ``flags`` for a water problem.

    :func:`_fin_config` is the aeroplane's four keys; a boat states two of
    them (:data:`STRUT_KEYS`), because a mast has no volume coefficient and
    no aspect ratio to be given — its height is the submergence depth and
    its chord is the mast chord, both already on the problem.
    """
    if not flags:
        return {}
    return {k: flags[k] for k in STRUT_KEYS
            if k in flags and flags[k] is not None}


def _fin_config(flags: dict | None) -> dict:
    """The VERTICAL SURFACE's subset of ``flags`` — presence and shape.

    :func:`_tail_config` carries these to the families that also have a
    TAILPLANE to configure. A tandem has no tailplane and still has a fin:
    its vertical surface is sized against the pair's stagger, reported,
    lofted, charged and flown exactly as a wing+tail's is. So the four keys
    travel on their own here, rather than by declaring tail-type and
    elevator-chord flags on a family that has neither.

    ...AND THE BOOM WITH THEM (:data:`TANDEM_FIN_BOOM_KEY`), which is the
    one of the five that ONLY a pair has: where a wing+tail's fin stands is
    answered by the body it hangs off, and a pair owns no body. It travels
    here and not in :func:`_tail_config` for exactly that reason — the
    families that route through the other function have no boom to state.
    """
    if not flags:
        return {}
    return {k: flags[k] for k in FIN_SHAPE_KEYS + (FIN_PRESENCE_KEY,
                                                   TANDEM_FIN_BOOM_KEY)
            if k in flags and flags[k] is not None}


def _arm_band_kwargs(bounds_overrides: dict | None) -> dict:
    """The ``l_t_m`` design-box row -> ``l_t_bounds_m`` on the problem.

    A BOUND, not a flag, because the horizontal separation IS a design
    variable while it is searched, and where a shell asks for a band is that
    variable's design-box row. Everything else about a searched arm already
    travels that way; what did not was the band reaching the PROBLEM.

    Without this, :func:`_apply_overrides` moved the box the sampler draws
    from while ``prob.bounds`` stayed at the family's calibrated band, so
    every draw outside it came back "bounds violation" from ``fg_tail`` —
    the calibration acting as a ban, in the one direction (wider) where it
    is not one. A narrower row behaved correctly by accident, which is why
    it went unseen.

    Empty in, empty out: a run with no row typed builds the published
    problem, bit-for-bit. ``tail.arm_row`` does the validating, so a band
    below the pitch solve's floor is refused in one place with one message.
    """
    row = (bounds_overrides or {}).get("l_t_m")
    if row is None:
        return {}
    return {"l_t_bounds_m": (float(row[0]), float(row[1]))}


def _area_band_kwargs(bounds_overrides: dict | None) -> dict:
    """The ``S_t_m2`` design-box row -> ``s_t_bounds_m2`` on the problem.

    :func:`_arm_band_kwargs` for the other half of the same surface, and it
    became necessary the day the tail box stopped being a constant: the band
    the family opens on is a FRACTION of the wing (``tail.area_band``), so
    "the tail I want is bigger than the box" is now an ordinary thing for a
    user to say — and a row widened in the shell alone moves the SAMPLER's
    box while the problem keeps its own, which turns every draw outside it
    into a bounds violation and the whole search into penalties.

    Empty in, empty out. ``tail.area_row`` does the validating.
    """
    row = (bounds_overrides or {}).get("S_t_m2")
    if row is None:
        return {}
    return {"s_t_bounds_m2": (float(row[0]), float(row[1]))}


def _size_band_kwargs(size_free, bounds_overrides: dict | None,
                      flags: dict | None = None) -> dict:
    """The SIZE rows of the design box -> the problem's own size bands.

    The third instance of :func:`_arm_band_kwargs`'s bug class, and the one
    that bit hardest: ``b_m``, ``S_m2`` and the searched loading ``ws_pa``
    are design-box rows, so a shell can move them — but a widened row moved
    the SAMPLER's box only, while the problem's own box stayed the fractional
    band around its reference size, and every draw outside it came back
    "bounds violation" from the objective. A user could therefore widen the
    area band to reach their own mission's wing loading and get a run made
    entirely of penalties.

    Both span rows of a tandem pair feed ONE band, as their union: sizing.py
    gives every span row the same problem-level box on purpose, and the
    per-row overrides then narrow each back to what the user typed. The
    physics box has only to CONTAIN the sampler's.

    Empty in, empty out: no rows, published problem, bit-for-bit.
    """
    from .sizing import (EXTRA_SPAN_LABELS, SIZE_MODE_FREE,
                         SIZE_MODE_WS_FREE, WS_LABEL, size_mode)

    ov = bounds_overrides or {}
    if not size_free or not ov:
        return {}
    mode = size_mode(size_free)
    out: dict = {}
    spans = [ov[k] for k in ("b_m",) + EXTRA_SPAN_LABELS
             if ov.get(k) is not None]
    if spans:
        out["span_bounds_m"] = (min(float(r[0]) for r in spans),
                                max(float(r[1]) for r in spans))
    if mode == SIZE_MODE_FREE and ov.get("S_m2") is not None:
        out["area_bounds_m2"] = (float(ov["S_m2"][0]), float(ov["S_m2"][1]))
    if mode == SIZE_MODE_WS_FREE and ov.get(WS_LABEL) is not None:
        row = (float(ov[WS_LABEL][0]), float(ov[WS_LABEL][1]))
        cap = (flags or {}).get("wing_loading_limit_pa")
        if cap is not None and row[1] > float(cap):
            # loud, not clipped: a band the mission forbids is a disagreement
            # between two things the user stated, and quietly moving one of
            # them is how a limit stops being a limit
            raise ValueError(
                f"the wing-loading band {row[0]:g}-{row[1]:g} Pa reaches "
                f"above the mission's own ceiling of {float(cap):g} Pa "
                f"(constraint_diagram) — narrow the ws_pa row, or state a "
                f"mission that allows it")
        out["ws_bounds_pa"] = row
    return out


def _height_band_kwargs(bounds_overrides: dict | None) -> dict:
    """The ``z_t_m`` design-box row -> ``z_t_bounds_m`` on the problem.

    The VERTICAL separation's twin of :func:`_arm_band_kwargs`, needed for
    the same reason and only on the families that SEARCH the height
    (``free_height``): the row moved the sampler's box while the problem's
    own stayed at ``Z_T_FRAC_BOUNDS`` on the family's trim span, so a band
    the user widened downwards bought refused draws.

    It bit harder here than the arm did, because the height's published
    band is a fraction of a span the shell has usually already changed —
    the row a user reads is not even the row the run was using
    (``gui.v3.config.FLAG_MOVED_ROWS``).
    """
    row = (bounds_overrides or {}).get(TAIL_HEIGHT_KEY)
    if row is None:
        return {}
    return {"z_t_bounds_m": (float(row[0]), float(row[1]))}


#: the design-box row that carries each surface's tip-device HEIGHT, and the
#: ``Problem`` field of the same meaning. Read BY LABEL because the same
#: dozen families appear under 1878 registered names once the twins are
#: generated.
_WINGLET_H_ROWS = {
    "winglet_h_frac": "winglet_h_bounds",
    "winglet_h_frac_t": "winglet_h_bounds_t",
    "winglet_h_front": "winglet_h_bounds_front",
    "winglet_h_rear": "winglet_h_bounds_rear",
}


def _winglet_h_band_kwargs(bounds_overrides: dict | None,
                           rows: tuple = ()) -> dict:
    """The tip-device HEIGHT rows -> the problem's own box.

    The same bug class :func:`_arm_band_kwargs` names, on the last tip-device
    row that did not travel. Measured before this existed: widening
    ``winglet_h_frac`` to (0, 0.25) and PINNING it at 0.20 gave the sampler a
    box of [0, 0.25] while ``prob.bounds`` stayed at the calibrated
    [0, 0.15], so every evaluation came back ``"bounds violation"`` and the
    run reported ``best_score = -100.0`` — "no solution" where the answer was
    a calibration acting as a ban.

    It matters more here than on most rows because the height is the row a
    user is invited to STATE, and a stated number is precisely the kind that
    sits outside a calibrated band: at the published box design this model's
    own optimum is h_frac ~ 0.45, three times the 0.15 ceiling.

    ``rows`` restricts which labels are read, so a family is never handed a
    band for a device it does not carry. Empty in, empty out: bit-for-bit
    published.
    """
    ov = bounds_overrides or {}
    want = rows or tuple(_WINGLET_H_ROWS)
    out = {}
    for label in want:
        row = ov.get(label)
        if row is None:
            continue
        out[_WINGLET_H_ROWS[label]] = (float(row[0]), float(row[1]))
    return out


#: The water families' OPERATING-POINT rows: the design-box label -> the
#: field of the same meaning on ``HydrofoilProblem``,
#: ``HydrofoilWingletProblem`` and ``HydrofoilTailProblem``, and the
#: ``hydrofoil`` validator that owns that row's refusals. The field is named
#: for the box row rather than the other way round because the row is what a
#: user sees; ``hydrofoil.depth_row`` / ``hydrofoil.speed_row`` do the
#: validating, so a band is refused in one place with one message.
#:
#: The validator is named HERE, beside the field, so that the row reaches it
#: WHOLE — see :func:`_water_band_kwargs` for why unpacking it first silences
#: the refusal that was written for a malformed one.
_WATER_OPERATING_ROWS = {"depth_m": ("DEPTH_BOUNDS", "depth_row"),
                         "V_ms": ("V_BOUNDS", "speed_row")}


#: THE WATER FAMILIES' SIZE, WHERE IT IS SEARCHED. Both are flags and not
#: box rows, because the row is what they CREATE: absent, a water problem
#: flies its published planform (b = 1.2 m, S = 0.144 m2) and its published
#: design vector, bit-for-bit; present, ``b_m`` / ``S_m2`` join the vector as
#: ordinary design-box rows whose band can then be moved like any other.
#:
#: The value is ``True`` — "search it, you pick the band"
#: (``hydrofoil.SIZE_BAND_FRAC`` around the family's own size) — or the band
#: itself.
#:
#: WHERE EACH ROW IS OFFERED, and this is the corrected version of a rule
#: that used to be stated once, too widely, and wrongly. It read: "NOT
#: offered on the elevator family: its stabiliser height and ARM rows are
#: stated as fractions of the span, so a searched span would make two box
#: rows depend on a third." Two things were wrong with it.
#:
#: * The ARM is not a span fraction. ``hydrotail.L_T_BOUNDS`` is ``(0.5,
#:   1.5)`` METRES and ``tail.arm_row`` validates it in metres; nothing in
#:   that row reads ``b``. Only the stabiliser DEPTH band is span-scaled
#:   (``hydrotail.Z_T_FRAC_BOUNDS`` times ``b``).
#: * ...and that band is only IN the design box when the depth is searched.
#:   With it fixed, ``z_t = -DZ_FRAC*b`` is a derived VALUE — exactly like
#:   ``winglet_h_frac``'s semi-span on the families that have searched their
#:   span all along — and no row moves any other row's bounds. Half the
#:   elevator families are of that kind, and they were refused by a
#:   sentence written about the other half.
#: * ...and with the depth searched TOO, the row simply changes units: it is
#:   a FRACTION of the candidate's own span (``hydrotail.Z_T_FRAC_LABEL``),
#:   which is the form the band was written in to begin with. So the other
#:   half is not refused either. That half is what the shells open water on,
#:   so this was the difference between "search the span" being one click and
#:   being unreachable.
#:
#: So: ``free_area`` is offered on EVERY water family (nothing in any water
#: box is stated as a fraction of the area), and ``free_span`` is offered
#: on every water family. The span row's ONE refusal lives in
#: ``hydrofoil._size_check``, which is also the one place that knows about
#: the strut: a searched span is not offered beside a vertical surface AT
#: ZERO HEEL (``hydrofoil.SPAN_NEEDS_NO_FIN`` — the mast's size and drag
#: read the depth and the reference area and never the span, so on flat
#: water the search would be spending a freedom against a surface blind to
#: it). Heel is what prices it (``hydrofoil.span_is_priced``).
WET_SIZE_KEYS = ("free_span", "free_area")


def wet_size_keys(free_height: bool) -> tuple:
    """Which size rows a water family may be offered — ALL OF THEM, now.

    ONE reader, asked by the registry at declaration time and by the shells
    at menu time, so a control cannot offer a row the builder will refuse.

    It used to answer differently for a family whose stabiliser DEPTH is a
    design variable: that row's band is ``hydrotail.Z_T_FRAC_BOUNDS`` times
    the span, so a searched span was refused beside it. The argument is kept
    and the answer no longer depends on it, because the coupling was in the
    depth row's UNITS rather than in the pair — with both free the depth is
    searched as a FRACTION of the candidate's own span
    (``hydrotail.Z_T_FRAC_LABEL``). The families this freed are the ones the
    shells open water on, where "search the span" was unreachable without
    changing family first.
    """
    return WET_SIZE_KEYS


#: THE WATER CRAFT'S FLIGHT CONDITION beyond its speed and depth: how far
#: over it is flying, and how much water the shallowest part of it must keep.
#: Values and not design-box rows, because neither is a shape the optimiser
#: chooses — they are statements about the day and the rider
#: (``hydrofoil.heel_angle`` says why the heel is asked and not derived).
#:
#: ``heel_deg`` is the only quantity anywhere in these solvers that reads how
#: WIDE the foil is: at heel the rising tip climbs ``(b/2) sin(phi)`` towards
#: the surface, which is what finally gives a searched span something to
#: trade against (``hydrofoil.heeled_z``, ``hydrofoil.immersion_margin``).
#: Zero — every published run — leaves every water number bit-for-bit.
WATER_FLIGHT_KEYS = ("heel_deg", "tip_clearance_m")

#: READ THE SECTION AT THE FLOWN REYNOLDS NUMBER. The same key the car
#: families already take, spelt the same way on purpose: it is the same
#: question about the same bank, and a second name for it would be a second
#: thing to keep in step.
#:
#: The water families needed it more and could not have it. Their SPEED has
#: been a design-box row since the beginning and ``Re_mac`` has always been
#: in the breakdown — it was simply never used to pick the table. What
#: blocked the wiring was the CAVITATION half: ``g = sigma_cav + Cp_min``,
#: and only Re 1e6 shipped a ``.cpmin`` companion, so ``polar.polar_at_re``
#: answered for drag everywhere in the bank and for Cp_min nowhere else.
#: All twenty (t/c, Re) cells carry one now.
WATER_RE_KEY = "flown_reynolds"


def _water_re_kwargs(flags: dict | None) -> dict:
    """``{flown_reynolds: bool}`` where a design asked for it, else empty.

    Empty in, empty out, and False is dropped as well as None: the problem's
    own default IS False, so sending it would put a flag in every stored
    water config that no published run ever carried.
    """
    got = (flags or {}).get(WATER_RE_KEY)
    return {} if not got else {WATER_RE_KEY: True}


def _water_flight_kwargs(flags: dict | None) -> dict:
    """The heel and the tip clearance -> problem kwargs. Empty in, empty out.

    The PROBLEM validates them (``hydrofoil.heel_angle``, ``_clearance``), so
    a malformed value is refused once, with the sentence written for it,
    rather than by a TypeError raised here — the rule
    :func:`_water_band_kwargs` states in full about its own two rows.
    """
    if not flags:
        return {}
    return {k: flags[k] for k in WATER_FLIGHT_KEYS
            if k in flags and flags[k] is not None}

#: flag -> (problem field, the design-box row it creates, the validator that
#: owns its refusals). Same three-part shape as :data:`_CAR_SIZE_ROWS`, and
#: for the same reason: the band reaches the validator WHOLE, so a malformed
#: one meets ``hydrofoil``'s own message rather than a TypeError here.
_WET_SIZE_ROWS = {"free_span": ("span_bounds_m", "b_m", "span_row"),
                  "free_area": ("area_bounds_m2", "S_m2", "area_row")}


def _wet_size_kwargs(flags: dict | None,
                     bounds_overrides: dict | None) -> dict:
    """Water SIZE flags (+ the box rows they created) -> problem kwargs.

    TWO INPUTS, ONE ANSWER, and the order matters. The FLAG says whether the
    row exists at all; the design-box OVERRIDE says how wide it is once it
    does. A shell that turns the row on and then drags its band therefore
    sends both, and the override wins — which is what makes this idempotent
    under rebuild, the contract ``gui.v3.relax._cap_to_validity`` needs and
    the one :func:`_car_size_band_kwargs` states in full.

    Empty in, empty out: no flag, no row, no change to any published run.
    """
    from . import hydrofoil

    fl, ov = flags or {}, bounds_overrides or {}
    out: dict = {}
    for key, (field, label, validator) in _WET_SIZE_ROWS.items():
        got = fl.get(key)
        if got is None or got is False:
            continue
        row = ov.get(label)
        band = row if row is not None else got
        if band is True:
            out[field] = True              # the family's own default bracket
            continue
        band = (float(band[0]), float(band[1]))
        getattr(hydrofoil, validator)(band)
        out[field] = band
    return out


def _water_band_kwargs(bounds_overrides: dict | None) -> dict:
    """The ``depth_m`` and ``V_ms`` design-box rows -> the problem's own box.

    The seventh and eighth instances of :func:`_arm_band_kwargs`'s bug class,
    and the last two rows in the water box that did not travel. Until this
    existed, ``DEPTH_BOUNDS`` and ``V_BOUNDS`` were bare class attributes
    with no annotation — not dataclass fields, so no builder keyword could
    reach them — and :func:`_apply_overrides` moved the SAMPLER's box while
    the problem kept the published (0.15, 1.0) m and (8, 16) m/s. A widened
    speed row therefore bought a run made of ``"bounds violation"``, and
    ``RunConfig.pinned`` refused a pin at 5 m/s against a box the run was no
    longer searching. :func:`rows_outside_validity` reported both, which is
    how they were priced; this is the fix it was waiting for.

    It is the whole difference between a foiling-dinghy tool and a foiling
    tool: 8 m/s is above every take-off speed in the literature the water
    families are now aimed at, so the two rows that decide the operating
    point were the two a small-craft user most needed to move
    (``hydrofoil.speed_row`` carries the measured numbers).

    Empty in, empty out: a run with neither row typed builds the published
    problem with its published arguments, bit-for-bit. Rows are read BY
    LABEL, exactly as :func:`_arm_band_kwargs` reads its own — an override
    keyed by integer index still reaches the sampler through
    :func:`_apply_overrides` and still does not reach the problem, which is
    what :func:`rows_outside_validity` is there to say out loud.

    THE WHOLE ROW GOES TO THE VALIDATOR, and that is the load-bearing detail
    rather than a style choice. This function used to unpack the row itself
    (``float(row[0]), float(row[1])``) and hand the validator two numbers it
    could no longer fault, which made ``hydrofoil._operating_row``'s crafted
    refusal — "the depth_m design-box row must be a pair (lo, hi) in m" —
    unreachable: a shell that sent a bare 0.5 instead of (0.4, 0.6) got a
    ``TypeError`` about subscripting a float, from this line, with neither
    the row's name nor the shape it should have been. The sentence exists to
    be met, so the row travels intact and the refusal happens where it was
    written. Every well-formed row is unchanged — the validators return the
    same ``(float, float)`` pair this line built.
    """
    ov = bounds_overrides or {}
    out: dict = {}
    if not ov:
        return out
    from . import hydrofoil
    for label, (field_name, validator) in _WATER_OPERATING_ROWS.items():
        row = ov.get(label)
        if row is None:
            continue
        out[field_name] = getattr(hydrofoil, validator)(row)
    return out


def _make_tail_builder(fixed_arm: bool, chord_order: int = 0,
                       flight_free: bool = False, size_free: bool = False,
                       tail_free: bool = False):
    def build(mission_kwargs, flags, bounds_overrides):
        from . import tail
        kw = _tail_config(flags)
        # the fuselage the arm implies. This family is where the arm
        # ratchet was measured, so it is the first that has to be able
        # to charge it (tail.TailProblem.fuselage_diameter_m).
        kw.update(_fuselage_kwargs(flags))
        if tail_free:
            kw["tail_free"] = True
        kw.update(_chord_kwargs(chord_order, flags))
        kw.update(_flight_kwargs(flight_free))
        kw.update(_size_kwargs(size_free, flags))
        kw.update(_wing_loading_kwargs(size_free, flags, 10.0))
        kw.update(_size_band_kwargs(size_free, bounds_overrides, flags))
        kw.update(_planform_kwargs(flags))
        if fixed_arm:
            arm = (flags or {}).get("l_t_m")
            if arm is None:
                # the middle of the band THIS wing's arm is searched over —
                # the arm is a fraction of the span (tail.arm_band), so a
                # fixed-arm default read off the reference aeroplane put a
                # model aeroplane's tail metres behind it
                arm = tail.default_arm(kw.get("b", tail.B_REF_M))
            kw["l_t_fixed"] = float(arm)
        else:
            kw.update(_arm_band_kwargs(bounds_overrides))
        kw.update(_area_band_kwargs(bounds_overrides))
        # each surface flies the section chosen for it; an absent aft choice
        # keeps the documented "tail reuses the wing polar" default
        # the LAYOUT this family trims in: both ends of it are the user's
        # where they state one, and the family's own otherwise. The height
        # is a real physical choice in the coupled LLT — dz enters the
        # horseshoe kernel, so a surface further out of the wing's trailing
        # sheet sees less downwash and the neutral point moves aft with it —
        # and tail.TailProblem validates it (T-tail refused, magnitude
        # floored at the measured clearance).
        if (flags or {}).get(TAIL_CG_KEY) is not None:
            kw["x_cg"] = float(flags[TAIL_CG_KEY])
        if (flags or {}).get(TAIL_HEIGHT_KEY) is not None:
            kw["z_t_fixed"] = float(flags[TAIL_HEIGHT_KEY])
        kw.update(_tail_mount_kwargs(flags))
        kw.update(_tail_limits_kwargs(flags))
        kw.update(_section_kwargs(flags))
        kw.update(_section_kwargs(flags, field="polar_tail",
                                  key=SECTION_AFT_KEY))
        prob = _apply_mission(tail.TailProblem(**kw), mission_kwargs)
        labels = prob.param_labels
        bounds = _apply_overrides(prob.bounds, labels, bounds_overrides)
        return _BuiltProblem(
            problem=prob,
            callable=lambda x: tail.fg_tail(x, prob),
            evaluate=lambda x: tail.evaluate_tail(x, prob),
            bounds=bounds, is_constrained=True, dim=prob.dim,
            param_labels=labels, medium="air",
        )
    build.takes_section = True
    build.takes_section_aft = True      # the stabiliser is a real surface
    # this family GATES the aspect ratio per candidate
    # (``sizing.check_ar_limit``), so the registry may declare the
    # limit flags on it: a flag is declared where it is READ
    build.gates_ar = True
    return build


_build_tail = _make_tail_builder(fixed_arm=False)


#: keys the standalone ``airfoil (section)`` builder reads from ``flags`` —
#: the 2-D operating point, the two design gates, and an optional base-seed
#: anchor. All optional and airfoil-namespaced (so they can never collide with
#: the wing LLT/VLM ``mach`` flag in a shared RunConfig): an empty/absent set
#: reproduces ``airfoil.AirfoilProblem()`` bit-for-bit. These are a BUILDER
#: contract, deliberately NOT declared on the ProblemSpec.flags tuple (that
#: drives the GUI's wing physics toggles), so the Design-page section_only path
#: stays unchanged.
#: ``airfoil_n_cst`` is the design-space DIMENSION: CST weights per surface,
#: so d = 2 n_cst. 4 (d = 8) is the published study and the default; the flag
#: exists because "how many variables should the section have" is a question
#: the literature answers for REPRESENTATION (Masters et al. 2017: 20-25 DVs
#: for geometric coverage) and not for a 60-160 evaluation budget, which is
#: what scripts/cst_dimension_study.py measures.
#: the CRUISE polar sweep, in degrees (``airfoil.alpha_sweep``). Its TOP END
#: is a ceiling on the design lift the section problem can be asked for: a
#: candidate whose lift curve is still rising when the sweep ends is refused
#: for "cannot reach design lift", which reads as physics and is not. Measured
#: at re 8.5e5, ``cl_design`` 1.3 is 0/14 feasible for BO, GA and Sobol alike
#: on the shipped 10 deg ceiling. The car rear wing meets it first
#: (``session.REFERENCE_CL`` is 1.0 there against 0.5 everywhere else).
#:
#: Deliberately NOT named ``airfoil_alpha_max_deg``: that key already exists in
#: :data:`AIRFOIL_WING_FLAG_KEYS` and is the WING's incidence cap, a different
#: number that moves nothing here. Two names, because they are two questions.
#:
#: Unstated is bit-for-bit :data:`airfoil.ALPHA_SWEEP_DEG`, which is what keeps
#: every cached polar valid — ``xfoil_run.cache_key`` hashes the sweep.
AIRFOIL_SWEEP_FLAG_KEYS = ("airfoil_sweep_alpha_min_deg",
                           "airfoil_sweep_alpha_max_deg",
                           "airfoil_sweep_alpha_step_deg")

AIRFOIL_FLAG_KEYS = ("airfoil_re", "airfoil_mach", "airfoil_cl_design",
                     "airfoil_tc_min", "airfoil_cm_max", "airfoil_anchor",
                     "airfoil_n_cst",
                     # SYMMETRIC — the vertical stabiliser's mode. Halves the
                     # design vector (w_lower = -w_upper), so the search
                     # cannot leave the symmetric family. See
                     # airfoil.AirfoilProblem.symmetric.
                     "airfoil_symmetric") + AIRFOIL_SWEEP_FLAG_KEYS

#: WHICH SCALAR the section search maximises. ``cd`` (default, and the frozen
#: Tier-B study bit-for-bit) is f = -cd at the design lift; ``composite`` is
#: the GDP weighted composite J of the SAME six criteria the library screen
#: ranks on (``airfoil_select.composite_objective``), which is the only way
#: the five criteria the screen chose a section for survive into stage 3.
#:
#: ``airfoil_score_weights`` is the criterion weight dict (screen_weights
#: accepts a preset name or a partial dict); ``airfoil_score_reference`` is the
#: FROZEN (lo, hi) band payload the sub-scores are normalised against — absent
#: means the shipped library band. The band is part of the objective, not a
#: display choice: a live min-max would make J non-stationary and no GP could
#: fit it (airfoil_select module docstring).
#: ``airfoil_censored`` picks what a RIGHT-CENSORED ``cl_max`` means to the
#: composite: ``refuse`` (the default, and every published composite run
#: bit-for-bit) treats a stall sweep that ended with cl still rising as a
#: solver failure; ``lower_bound`` keeps the record and scores J at the largest
#: converged cl, which is a true LOWER BOUND on the J that section would score
#: for any non-negative weight vector — both censorable criteria (``clmax``,
#: ``astall``) are higher-better and the other four come from the cruise sweep,
#: which converged. Measured over 42 PAIRED seeds
#: (results/censored_arm_study.json): the refusal rate falls 11.9 % -> 9.0 %
#: and censored cl_max is 29.1 % of all refusals — but the SCORE effect is a
#: measured NULL, mean paired difference +0.050 J with 95 % CI
#: [-0.824, +0.923], sign p = 0.868 / Wilcoxon p = 0.462. (A 5-seed pilot had
#: reported +1.58 J; it was noise.) So the flag ships for CORRECTNESS — it
#: stops discarding observations that are lower bounds — and the default does
#: not move, because there is no score to buy (report §16.3a).
#:
#: ``composite_goal`` is the composite with a REFERENCE POINT: the same J,
#: minus a one-sided penalty for ending below a per-criterion goal
#: (``airfoil_score_goals``, ``airfoil_goal_penalty``, ``airfoil_goal_tol`` —
#: see :func:`airfoil_run_config` and ``airfoil_select.goal_evaluation``). It
#: exists because a weighted sum sells: on the frozen sec8_composite case the
#: composite winner gains 11 J while cruise L/D falls 12 % and cd at the design
#: lift rises 14 %, which is what the arithmetic asks for and not what the user
#: asked for. Goals default to the SEED's own six metrics, so the search is
#: told "improve J, but do not hand back a section worse than the one I have".
#:
#: ``composite_asf`` is the augmented Tchebycheff / achievement scalarising
#: function on the SAME band, weights and reference point:
#: ``min_k w_k (s_k - r_k) + rho sum_k w_k (s_k - r_k)`` (``airfoil_asf_rho``).
#: It exists because the goal composite fixes the SALE and not the
#: REACHABILITY: a weighted sum plus a hinge is still a weighted sum, so it
#: still only returns points on the convex hull of the achievable set. The min
#: term holds the search to its WORST criterion relative to the seed, which no
#: exchange rate can buy off. Its value is NOT a composite J and is never
#: reported as one — see ``airfoil_select.asf_evaluation``.
#:
#: ``pareto`` is the odd one out and is documented at
#: :data:`PARETO_OBJECTIVE_NAME`: it is NOT a scalar. It searches three of the
#: six criteria as separate objectives (constrained qNEHVI) and returns a
#: FRONT, which :class:`RunResult` carries on ``front`` while ``best_x`` is the
#: front point with the highest plain composite J. It is in this tuple only
#: because ``run`` can now deliver one — see ``RESULTS_FRONT_REACHABILITY.md``
#: for what it is FOR (41/42 seeds carry a design no scalarisation dominates)
#: and, just as importantly, what it is not (a worse single-answer search than
#: ``composite_goal`` at equal budget).
AIRFOIL_OBJECTIVE_FLAG_KEYS = ("airfoil_objective", "airfoil_score_weights",
                               "airfoil_score_reference", "airfoil_censored",
                               "airfoil_score_goals", "airfoil_goal_penalty",
                               "airfoil_goal_tol", "airfoil_asf_rho",
                               "airfoil_asf_lambda", "airfoil_asf_missing",
                               "airfoil_pareto_acqf",
                               "airfoil_pareto_q", "airfoil_pareto_criteria")

#: The name of the one objective that is not a scalar.
#:
#: For a long time this was deliberately absent from
#: :data:`AIRFOIL_OBJECTIVES`, on the ground that every name in that tuple is
#: something :func:`run` can maximise while :class:`RunResult` carries one
#: ``best_x`` — so naming a front before ``run`` could return one would have
#: promised a mode no evaluation delivers. ``run`` now returns one:
#: :attr:`RunResult.front` carries the whole front and ``best_x`` is its
#: highest-plain-J point, so the name is real and the tuple says so.
PARETO_OBJECTIVE_NAME = "pareto"

#: what a section run can be pointed at: four scalars, and one FRONT.
AIRFOIL_OBJECTIVES = ("cd", "composite", "composite_goal", "composite_asf",
                      PARETO_OBJECTIVE_NAME)

#: the objectives built out of the screening composite: same band, same
#: weights, same censoring policy, same extra stall sweep — they differ only in
#: what is done with the six sub-scores once measured. ``pareto`` belongs here:
#: its objectives ARE sub-scores on the frozen band, which is what makes a
#: front point and a scalarised winner comparable at all.
AIRFOIL_COMPOSITE_OBJECTIVES = ("composite", "composite_goal", "composite_asf",
                                PARETO_OBJECTIVE_NAME)

#: the objectives that carry a REFERENCE POINT (``ScoreGoals``): the goal
#: composite measures a shortfall against it, the ASF measures an achievement
#: from it. Both read ``airfoil_score_goals`` and ``airfoil_goal_tol``;
#: neither the plain composite nor ``-cd`` reads either.
AIRFOIL_REFERENCE_OBJECTIVES = ("composite_goal", "composite_asf")

_AIRFOIL_SCALAR_FLAGS = {
    "airfoil_re": "re", "airfoil_mach": "mach",
    "airfoil_cl_design": "cl_design", "airfoil_tc_min": "tc_min",
    "airfoil_cm_max": "cm_max",
}

#: ...and the BOOLEAN one. Kept apart from the scalars because ``float()``
#: on it would turn True into 1.0 and quietly build a cambered problem.
_AIRFOIL_BOOL_FLAGS = {"airfoil_symmetric": "symmetric"}


def _airfoil_kwargs(flags: dict | None) -> dict:
    """AirfoilProblem kwargs from the airfoil-namespaced ``flags`` subset.

    ``airfoil_anchor`` is either a NACA 4-digit code string (re-centres the
    design box on that section) or an explicit ``[w_upper, w_lower]`` CST fit —
    the §15 screen-then-optimise pipeline's BASE SEED, so the 8-D box is built
    around a chosen known aerofoil instead of the fixed NACA-2412 anchor.
    Everything else is a scalar operating point / design-gate override. Empty
    in, empty out (bit-for-bit AirfoilProblem())."""
    if not flags:
        return {}
    out: dict = {}
    for fk, pk in _AIRFOIL_SCALAR_FLAGS.items():
        v = flags.get(fk)
        if v is not None:
            out[pk] = float(v)
    # the DIMENSION of the section design space: weights per surface, so
    # d = 2 n_cst. 4 (d = 8) is the published study; it is stated here rather
    # than inferred from the anchor's length because an anchor of the wrong
    # length against a stated dimension is a caller error worth raising on,
    # and AirfoilProblem does raise on it.
    n_cst = flags.get("airfoil_n_cst")
    if n_cst is not None:
        out["n_cst"] = int(n_cst)
    for fk, pk in _AIRFOIL_BOOL_FLAGS.items():
        v = flags.get(fk)
        if v is not None:
            out[pk] = bool(v)
    # the CRUISE sweep (AIRFOIL_SWEEP_FLAG_KEYS). Built only when a caller
    # actually states an end, so an unstated sweep leaves ``alphas`` on its own
    # default factory and every cached polar keeps its key.
    sweep = {k: flags.get(f"airfoil_sweep_alpha_{k}_deg")
             for k in ("min", "max", "step")}
    if any(v is not None for v in sweep.values()):
        from .airfoil import alpha_sweep
        out["alphas"] = alpha_sweep(sweep["min"], sweep["max"], sweep["step"])
    anchor = flags.get("airfoil_anchor")
    if anchor is not None:
        if isinstance(anchor, str):
            out["anchor"] = anchor
        else:
            w_u, w_l = anchor      # (w_upper[], w_lower[]) CST fit of the base
            out["anchor"] = (np.asarray(w_u, dtype=float),
                             np.asarray(w_l, dtype=float))
    return out


#: keys that switch the ``airfoil (section)`` builder from the pure 2-D
#: section problem into SECTION + TWIST wing mode (section_wing.py). The
#: presence of ``airfoil_wing`` — a WingGuess kwargs dict — is the switch;
#: absent, the builder is bit-for-bit the 8-D section problem. In wing mode
#: ``airfoil_re`` / ``airfoil_cl_design`` are IGNORED: both are DERIVED from
#: the wing (Re at the MAC, CL = W/(qS) from the area guess), and honouring a
#: typed-in value alongside would let the two disagree silently.
#:
#: ``airfoil_re_strip`` / ``airfoil_re_bank`` are the SPANWISE Reynolds
#: treatment (section_wing.RE_STRIP_MODES). ``mac`` (the default and every
#: published run) reads all strips off one polar at the MAC Reynolds number;
#: ``bank`` runs ``airfoil_re_bank`` XFOIL sweeps log-spaced over
#: [Re(tip), Re(root)] and interpolates each strip's cd log-log; ``power``
#: applies Drela's flat-plate exponent at no extra XFOIL. The correction is
#: worth +0.12 counts of CDp on the taper-0.6 default rising to +2.1 counts
#: (-1.9 % of L/D) at taper 0.2, and the free ``power`` mode recovers between
#: 40 % and 313 % of it depending on the regime — see
#: results/spanwise_re_probe.json before choosing either.
AIRFOIL_WING_FLAG_KEYS = ("airfoil_wing", "airfoil_twist_order",
                          "airfoil_twist_max_deg", "airfoil_alpha_max_deg",
                          "airfoil_chord_order", "airfoil_chord_max_frac",
                          "airfoil_re_strip", "airfoil_re_bank")

#: WingGuess fields a caller may set through ``airfoil_wing``.
WING_GUESS_KEYS = ("mass_kg", "v_ms", "altitude_m", "s_ref_m2",
                   "aspect_ratio", "taper", "n_stations")


def _section_wing_kwargs(flags: dict | None) -> dict | None:
    """SectionWingProblem kwargs from the wing-namespaced ``flags`` subset.

    ``None`` when no ``airfoil_wing`` block is present (i.e. stay in the pure
    2-D section mode). Unknown wing keys are rejected rather than ignored: a
    typo'd ``area`` silently falling back to the default 6 m^2 would move the
    derived CL_design without saying so.
    """
    if not flags:
        return None
    wing_kw = flags.get("airfoil_wing")
    if not wing_kw:
        # ...and the SEVEN dependent keys are refused rather than dropped.
        # They were accepted by accepted_flags(), returned here as None, and
        # discarded: measured, `airfoil_alpha_max_deg` at 3.0 / 2.0 / 1.0 /
        # 0.0 all returned a bit-identical score of -0.006622871125611746 on
        # `airfoil (section)`, whose trim alpha is 3.4796 deg — a gate that
        # could not fire in either direction. This function's own docstring
        # three lines up states the opposite policy for the keys INSIDE the
        # block; the same argument applies to the keys that need it.
        orphan = [k for k in AIRFOIL_WING_FLAG_KEYS
                  if k != "airfoil_wing" and k in flags]
        if orphan:
            raise ValueError(
                f"{sorted(orphan)} is read by nothing without an "
                f"'airfoil_wing' block: these keys configure the 3-D wing "
                f"the section is judged on, and with no wing there is "
                f"nothing for them to configure. State 'airfoil_wing', or "
                f"drop them rather than have them silently do nothing.")
        return None
    bad = [k for k in dict(wing_kw) if k not in WING_GUESS_KEYS]
    if bad:
        raise ValueError(f"unknown airfoil_wing key(s) {sorted(bad)}; "
                         f"keys={list(WING_GUESS_KEYS)}")
    out: dict = {"wing_kwargs": {k: float(v) if k != "n_stations" else int(v)
                                 for k, v in dict(wing_kw).items()}}
    for fk, pk, cast in (("airfoil_twist_order", "twist_order", int),
                         ("airfoil_twist_max_deg", "twist_max_deg", float),
                         ("airfoil_alpha_max_deg", "alpha_max_deg", float),
                         ("airfoil_chord_order", "chord_order", int),
                         ("airfoil_chord_max_frac", "chord_max_frac", float),
                         ("airfoil_re_bank", "re_bank", int),
                         ("airfoil_re_strip", "re_strip", str)):
        v = flags.get(fk)
        if v is not None:
            out[pk] = cast(v)
    return out


def _airfoil_censored(flags: dict | None, objective: str) -> str:
    """The right-censored ``cl_max`` policy for a section build.

    Validated HERE, against ``airfoil_select.CENSORED_MODES``, so an unknown
    mode fails where the caller asked for it rather than on the first candidate
    whose stall sweep happens to be censored — which may be evaluation 40 of a
    60-evaluation run.

    Stated on a ``-cd`` run it RAISES rather than being ignored: nothing in the
    ``cd`` objective reads ``cl_max`` at all, so accepting the flag there would
    promise a policy no evaluation applies. (The repo's standing rule: a flag
    that is silently dropped is worse than a flag that is refused.)
    """
    from .airfoil_select import DEFAULT_CENSORED, check_censored
    raw = (flags or {}).get("airfoil_censored")
    if raw is None:
        return DEFAULT_CENSORED
    mode = check_censored(raw)
    if objective not in AIRFOIL_COMPOSITE_OBJECTIVES \
            and mode != DEFAULT_CENSORED:
        raise ValueError(
            f"airfoil_censored={mode!r} is a COMPOSITE policy: it decides what "
            "a right-censored cl_max scores, and the -cd objective never reads "
            "cl_max. Pass airfoil_objective='composite' as well, or drop the "
            "flag")
    return mode


def _airfoil_objective(flags: dict | None):
    """``(name, reference, weights, censored)`` for the section objective.

    ``("cd", None, None, "refuse")`` — the default and the frozen Tier-B study
    bit-for-bit — or ``("composite", ScoreReference, ScoreWeights, mode)`` with
    the band and weights the composite J is measured against, both resolved
    HERE so the built problem carries them and no evaluation can silently pick
    a different map. ``censored`` is resolved the same way and for the same
    reason (:func:`_airfoil_censored`).

    A composite run REFUSES wing mode rather than quietly ignoring half the
    design vector: J scores a 2-D section, so a twist law and a chord law would
    move the geometry the objective cannot see — freedoms searched against a
    number that does not depend on them. The two are different questions and
    the caller has to say which one it is asking.
    """
    if not flags:
        from .airfoil_select import DEFAULT_CENSORED
        return "cd", None, None, DEFAULT_CENSORED
    name = str(flags.get("airfoil_objective") or "cd")
    if name not in AIRFOIL_OBJECTIVES:
        raise ValueError(f"unknown airfoil objective {name!r}; "
                         f"choose from {list(AIRFOIL_OBJECTIVES)}")
    censored = _airfoil_censored(flags, name)
    if name == "cd":
        return name, None, None, censored
    if flags.get("airfoil_wing"):
        raise ValueError(
            "the composite objective scores a 2-D SECTION, so it cannot be "
            "run in wing mode: the twist and chord laws would be searched "
            "against a number that does not depend on them. Optimise the "
            "section on the composite, or the wing on its L/D — not both at "
            "once")
    ref, info = _score_reference_arg(flags.get("airfoil_score_reference"))
    if ref is None:
        raise ValueError(
            "the composite objective needs a FROZEN normalisation band and "
            f"none could be loaded ({info.get('reason', 'no band')}). A live "
            "min-max would make J non-stationary — a moving target no GP can "
            "fit — so the run is refused rather than answered on a scale that "
            "changes every evaluation")
    weights = screen_weights(flags.get("airfoil_score_weights"))
    # A WEIGHT THE BAND CANNOT SPEND IS REFUSED HERE, where a config is still
    # a promise the run is launchable, rather than on the first evaluation
    # after the seed's XFOIL sweeps have been paid for. The case: a vertical
    # stabiliser's band covers no L/D criterion (both are read at a lift it
    # never carries), so a user who left the wing preset's 0.35 on cruise L/D
    # would have been searching against a number identical for every section.
    from .airfoil_select import CRITERIA, check_pareto_criteria
    w = weights.normalised()
    dead = [k for k in CRITERIA if not ref.covers(k) and w[k] > 0.0]
    if name == PARETO_OBJECTIVE_NAME:
        # A FRONT'S AXIS IS NOT A WEIGHT. The front searches named criteria as
        # a vector, so one the band cannot score is not "worth nothing" — it
        # is an axis with no coordinate, and every evaluation would come back
        # unscoreable. That reads as "the search found nothing", which is the
        # silence this refusal exists to break.
        axes = check_pareto_criteria(
            flags.get("airfoil_pareto_criteria"))
        dead += [k for k in axes if not ref.covers(k) and k not in dead]
    if dead:
        asked = ", ".join(
            f"{k} (weight {w[k]:.3g})" if w[k] > 0.0 else f"{k} (a front axis)"
            for k in dead)
        raise ValueError(
            "the frozen band cannot score "
            + ", ".join(f"{k}: {ref.why_unbanded(k)}" for k in dead)
            + f" — and this run asks it to rank {asked}. Drop "
              "them from the objective: on a surface that flies at zero lift "
              "the drag at the design lift (cdcr) is the criterion that "
              "carries what the L/D ones cannot")
    return (name, ref, weights, censored)


#: the goal-composite's own flag keys, refused on any other objective
AIRFOIL_GOAL_FLAG_KEYS = ("airfoil_score_goals", "airfoil_goal_penalty",
                          "airfoil_goal_tol")

#: WHICH objectives read WHICH reference-point flag. Stated as one table
#: because it is the only thing that stops the bug class this repo keeps
#: meeting: a flag accepted by an objective that never reads it is a promise
#: no evaluation keeps. ``airfoil_goal_penalty`` is the goal composite's
#: multiplier and there is no multiplier in an ASF (the min does that work);
#: ``airfoil_asf_rho`` is the ASF's augmentation and the goal composite has no
#: augmentation to weight. Both are refused, in both directions.
#: The FRONT's own three keys join the same table for the same reason. A
#: multi-objective acquisition, a batch size and a criteria set are meaningless
#: on a scalar run — there is no acquisition over one objective to choose, and
#: naming three criteria when six are being summed states a search nobody
#: performs. They are owned by ``pareto`` alone, in both directions: the goal
#: penalty and the ASF's rho/lambda are equally refused on a front, which has
#: neither a hinge nor a min term.
AIRFOIL_REFERENCE_FLAG_OWNERS = {
    "airfoil_score_goals": AIRFOIL_REFERENCE_OBJECTIVES,
    "airfoil_goal_tol": AIRFOIL_REFERENCE_OBJECTIVES,
    "airfoil_goal_penalty": ("composite_goal",),
    "airfoil_asf_rho": ("composite_asf",),
    "airfoil_asf_lambda": ("composite_asf",),
    "airfoil_asf_missing": ("composite_asf",),
    "airfoil_pareto_acqf": (PARETO_OBJECTIVE_NAME,),
    "airfoil_pareto_q": (PARETO_OBJECTIVE_NAME,),
    "airfoil_pareto_criteria": (PARETO_OBJECTIVE_NAME,),
}


def _airfoil_goal_settings(flags: dict | None, objective: str):
    """``(goals | None, penalty, tol, rho)`` for a reference-point build.

    ``None`` goals means "measure them off the SEED" — the section the design
    box is centred on, which is the reference point both objectives exist to
    work from. A dict states them outright, in RAW criterion units (t/c,
    cl_max, (L/D)max, (L/D) at the design lift, the stall angle, |cm|), which
    is the form the screening minimums are already in.

    Every key RAISES on an objective that does not read it, per
    :data:`AIRFOIL_REFERENCE_FLAG_OWNERS` — including the two that separate the
    pair (``airfoil_goal_penalty`` on an ASF run, ``airfoil_asf_rho`` on a goal
    run). Accepting either would promise a term no evaluation applies, which is
    the same rule as ``airfoil_censored`` and the same bug class as a stale
    menu entry.
    """
    from .airfoil_select import (CRITERIA, DEFAULT_ASF_LAMBDA,
                                 DEFAULT_ASF_MISSING, GOAL_PENALTY,
                                 GOAL_TOL, RHO_ASF, check_asf_lambda,
                                 check_asf_missing, check_asf_rho)

    d = flags or {}
    wrong = {k: owners for k, owners in AIRFOIL_REFERENCE_FLAG_OWNERS.items()
             if d.get(k) is not None and objective not in owners}
    if wrong:
        detail = "; ".join(f"{k} belongs to "
                           f"{' / '.join(repr(o) for o in owners)}"
                           for k, owners in sorted(wrong.items()))
        names = ", ".join(sorted(wrong))
        raise ValueError(
            f"{names} {'names' if len(wrong) == 1 else 'name'} a "
            f"reference-point term that {objective!r} does not read — "
            f"{detail}. Pass the matching airfoil_objective as well, or drop "
            "the flag: a term nobody evaluates is worse stated than absent")
    penalty = float(d.get("airfoil_goal_penalty", GOAL_PENALTY))
    tol = float(d.get("airfoil_goal_tol", GOAL_TOL))
    rho = check_asf_rho(d.get("airfoil_asf_rho", RHO_ASF))
    lam = check_asf_lambda(d.get("airfoil_asf_lambda"))
    miss = check_asf_missing(d.get("airfoil_asf_missing"))
    if objective not in AIRFOIL_REFERENCE_OBJECTIVES:
        return (None, GOAL_PENALTY, GOAL_TOL, RHO_ASF, DEFAULT_ASF_LAMBDA,
                DEFAULT_ASF_MISSING)
    raw = d.get("airfoil_score_goals")
    if raw is None or (isinstance(raw, str) and str(raw) == "seed"):
        return None, penalty, tol, rho, lam, miss
    goals = {str(k): float(v) for k, v in dict(raw).items()
             if v is not None}
    unknown = set(goals) - set(CRITERIA)
    if unknown:
        raise ValueError(f"unknown goal criteria {sorted(unknown)}; "
                         f"choose from {list(CRITERIA)}")
    return goals, penalty, tol, rho, lam, miss


def _resolve_airfoil_goals(prob, flags, reference, weights, censored,
                           objective: str = "composite_goal"):
    """The :class:`airfoil_select.ScoreGoals` a reference-point run works from
    — stated outright, or measured off the seed (one XFOIL cruise + stall
    sweep, which the run pays for anyway).

    An ASF run's reference point carries ``penalty = 0``: there is no
    multiplier in an achievement function, and a stored run that reported 10
    could be read as having applied one.
    """
    from .airfoil_select import ScoreGoals, seed_goals

    spec, penalty, tol, _rho, _lam, _miss = _airfoil_goal_settings(
        flags, objective)
    if objective == "composite_asf":
        penalty = 0.0
    if spec is not None:
        return ScoreGoals(goals=spec, penalty=penalty, tol=tol,
                          source="stated")
    return seed_goals(prob, reference, weights, censored=censored,
                      penalty=penalty, tol=tol)


def _build_airfoil(mission_kwargs, flags, bounds_overrides):
    from . import airfoil
    prob = airfoil.AirfoilProblem(**_airfoil_kwargs(flags))
    objective, reference, weights, censored = _airfoil_objective(flags)
    # …and the goal keys, HERE as well as in airfoil_run_config: a RunConfig
    # can be assembled directly (every study script does), and a reference
    # point stated on an objective that has none must not be quietly dropped
    # just because it did not come through the config helper.
    _airfoil_goal_settings(flags, objective)
    wing_kw = _section_wing_kwargs(flags)
    if wing_kw is not None:
        from . import section_wing as sw
        swp = sw.SectionWingProblem(
            airfoil=prob, wing=sw.WingGuess(**wing_kw.pop("wing_kwargs")),
            **wing_kw)
        labels = swp.param_labels
        bounds = _apply_overrides(swp.bounds, labels, bounds_overrides)
        return _BuiltProblem(
            problem=swp,
            callable=lambda x: sw.fg_section_wing(x, swp),
            evaluate=lambda x: sw.evaluate_section_wing(x, swp),
            bounds=bounds, is_constrained=True, dim=swp.dim,
            param_labels=labels, medium="air",
        )
    # THE PROBLEM'S OWN LABELS, not a pair per n_cst: a SYMMETRIC section
    # (the fin's) searches the upper surface alone and mirrors it, so it has
    # n_cst variables, not 2 n_cst. Naming eight of them for a four-long
    # vector made ProblemSpec.param_labels lie — `_section_weights` refused
    # the size mismatch and the run came back with NO section at all, and the
    # design-box table zipped four numbers against eight names.
    labels = tuple(prob.param_labels)
    bounds = _apply_overrides(prob.bounds, labels, bounds_overrides)
    if objective == PARETO_OBJECTIVE_NAME:
        from .airfoil_select import composite_evaluation as _plain_composite

        # A front has no scalar to maximise, and `run` dispatches this
        # objective to the multi-objective path before it ever asks for a
        # callable. If something else asks, it RAISES: silently handing back
        # the composite here would let a caller believe it had searched a
        # front while it had in fact run a weighted sum — the same
        # promise-nobody-keeps class this module refuses everywhere else.
        def _no_scalar(x):
            raise TypeError(
                "the 'pareto' objective is a FRONT, not a scalar: it has no "
                "single value to maximise, so this problem cannot be handed "
                "to a scalar optimiser. api.run dispatches it to the "
                "multi-objective path and returns the front on "
                "RunResult.front; call api.pareto_airfoil for the report "
                "form")

        # `evaluate` is the PLAIN COMPOSITE, and that is deliberate: it is the
        # currency the front is re-scored in, the one `best_x`'s `best_score`
        # is reported in, and the one every scalarised arm is measured in. A
        # breakdown at a front point is a composite breakdown.
        return _BuiltProblem(
            problem=prob, callable=_no_scalar,
            evaluate=lambda x: _plain_composite(x, prob, reference, weights,
                                                censored=censored),
            bounds=bounds, is_constrained=True, dim=prob.dim,
            param_labels=labels, medium="air",
        )
    if objective == "composite":
        from .airfoil_select import composite_evaluation, make_composite_fg
        return _BuiltProblem(
            problem=prob,
            callable=make_composite_fg(prob, reference, weights,
                                       censored=censored),
            evaluate=lambda x: composite_evaluation(x, prob, reference,
                                                    weights,
                                                    censored=censored),
            bounds=bounds, is_constrained=True, dim=prob.dim,
            param_labels=labels, medium="air",
        )
    if objective in AIRFOIL_REFERENCE_OBJECTIVES:
        import threading

        from .airfoil_select import (asf_evaluation, asf_objective,
                                     goal_evaluation, goal_objective)
        # The reference point is resolved ONCE per built problem, on first use.
        # Once, because a moving reference point is a moving objective — the
        # same reason the band is frozen. Lazily, because building a problem
        # must stay cheap and pure: `run`, `design_report` and the budget
        # recommender all build one, and only the first two ever evaluate it.
        cell: dict = {}
        lock = threading.Lock()
        _s, _p, _t, rho, lam, miss = _airfoil_goal_settings(flags, objective)

        def goals():
            with lock:
                if "goals" not in cell:
                    cell["goals"] = _resolve_airfoil_goals(
                        prob, flags, reference, weights, censored, objective)
            return cell["goals"]

        if objective == "composite_asf":
            call = lambda x: asf_objective(x, prob, reference, weights,   # noqa: E731
                                           goals(), rho, censored=censored,
                                           lam=lam, missing=miss)
            ev = lambda x: asf_evaluation(x, prob, reference, weights,     # noqa: E731
                                          goals(), rho, censored=censored,
                                          lam=lam, missing=miss)
        else:
            call = lambda x: goal_objective(x, prob, reference, weights,  # noqa: E731
                                            goals(), censored=censored)
            ev = lambda x: goal_evaluation(x, prob, reference, weights,   # noqa: E731
                                           goals(), censored=censored)
        return _BuiltProblem(
            problem=prob, callable=call, evaluate=ev,
            bounds=bounds, is_constrained=True, dim=prob.dim,
            param_labels=labels, medium="air",
        )
    return _BuiltProblem(
        problem=prob,
        callable=lambda x: airfoil.fg_airfoil(x, prob),
        evaluate=lambda x: airfoil.evaluate_airfoil(x, prob),
        bounds=bounds, is_constrained=True, dim=prob.dim,
        param_labels=labels, medium="air",
    )


_TANDEM_LABELS = (
    "taper_front", "taper_rear",
    "twist_front_k1_deg", "twist_front_k2_deg", "twist_front_k3_deg",
    "twist_rear_k1_deg", "twist_rear_k2_deg", "twist_rear_k3_deg",
    "area_split_front", "decalage_deg",
)


#: the tandem pair carries TWO chord laws (one per wing), front then rear
_TANDEM_CHORD_LABELS = (
    tuple(f"chord_front_k{j}" for j in range(1, _CHORD_ORDER + 1))
    + tuple(f"chord_rear_k{j}" for j in range(1, _CHORD_ORDER + 1)))


#: the tandem pair's STAGGER, in metres: how far aft of the front wing's
#: quarter-chord the rear wing's sits, and how far above it. Configuration,
#: not design variables — where the two surfaces sit relative to each other
#: is a layout the user states (a fuselage length, a boom, a wing box), and
#: it is the one geometric quantity these families exist to study.
TANDEM_STAGGER_KEYS = ("dx_m", "dz_m")

#: the REAR WING'S SPAN, in metres. Same kind of question as the stagger and
#: the same contract: stated it travels verbatim and nothing in the run moves
#: it, left alone the rear wing is as wide as the front one — which is every
#: published pair, bit-for-bit. A tandem's two wings share a fuselage, not a
#: span, so "how wide is the other wing" is a question the pair has always
#: had and could not be asked.
#:
#: It is a VALUE, not a dimension: it changes no design vector. Where the
#: SIZE modifier is on the two spans are design variables instead (one row
#: each, sizing.span_labels), and this flag is stripped with the other
#: chosen-size ones — a wing is sized one way.
TANDEM_SPAN_KEYS = ("b_rear_m",)

#: the stagger a span implies when nobody states one: the published pair's own
#: dx = b/2, dz = 0.1 b. Used ONLY at build time, off a span the USER typed —
#: never off a span the optimiser chose (tandem.TandemProblem.dx says why).
_TANDEM_STAGGER_FRACS = (0.5, 0.1)


def _tandem_planform_kwargs(flags: dict | None, size_free=False,
                            span_bounds_m=None) -> dict:
    """Wing-size + STAGGER kwargs for the tandem pair.

    Three questions, and the last two are the user's outright:

    * ``b_m`` / ``S_m2`` — the FRONT wing's span and the pair's TOTAL area.
    * ``b_rear_m`` — the REAR wing's span, where it is not the front's
      (:data:`TANDEM_SPAN_KEYS`). Unstated the rear wing is as wide as the
      front one, which is the pair every published run flew.
    * ``dx_m`` / ``dz_m`` — the stagger, in metres. Stated, it travels
      verbatim. Not stated, it is derived from the span in play as the
      published fractions (b/2 and 0.1 b), which is what makes a user who
      only types a span get a pair that still looks like a tandem instead of
      a 20 m-span pair 5 m apart.

    Either end of the stagger may be given alone; the other falls back to its
    fraction of the span, since "the wings are 1.2 m apart vertically" is a
    complete sentence about a layout. The stagger fractions are read off the
    FRONT wing's span: it is the reference wing, and a rear wing that is
    narrower does not move the fuselage it is bolted to.

    ...AND WHICH SPAN, WHERE THE SPAN IS SEARCHED. The stagger must never
    ride a span the OPTIMISER chose (``tandem.TandemProblem.dx``: the layout
    is the user's, and a dx that moved per candidate would make the pair a
    different aeroplane at every draw). But the span the fractions were read
    off was the family's NOMINAL 10 m even when the design box was about to
    search 6-40 m — and a stagger sized for a 10 m pair puts the two tip
    devices inside a 1 m vertical gap that a 23 m pair's devices, whose
    height is a fraction of THEIR OWN semi-span, cannot fit in. The box
    centre of every size-free pair with tip devices was therefore an
    in-contract failure ("the two tip devices are not separated"), and so
    was everything above about 22 m: 5 of 18 spans across the row flew.

    So the reference is the MIDPOINT OF THE BAND when the size is a design
    variable. A band is a statement the user makes at build time, exactly
    like a typed span — it is not a draw — so this keeps the contract while
    making the box admissible. A stated ``dx_m``/``dz_m`` still travels
    verbatim, and a fixed-size pair is unchanged, bit-for-bit.
    """
    flags = flags or {}
    size = flags.get("b_m") is not None or flags.get("S_m2") is not None
    stagger = any(flags.get(k) is not None for k in TANDEM_STAGGER_KEYS)
    b_rear = flags.get(TANDEM_SPAN_KEYS[0])
    if not (size or stagger or b_rear is not None or size_free):
        return {}
    b = 10.0 if flags.get("b_m") is None else float(flags["b_m"])
    S_total = 20.0 if flags.get("S_m2") is None else float(flags["S_m2"])
    if size:
        # validated on ONE wing's aspect ratio (b^2 / (S_total/2)) — the pair's
        # total area would report half the AR each surface actually flies at
        _planform_size({"b_m": b, "S_m2": 0.5 * S_total}, 10.0, 10.0)
    out: dict = {"b": b, "S_total": S_total}
    if b_rear is not None:
        # the rear wing is a wing: checked on its own aspect ratio, against
        # the same half of the total area the front one is checked on, so a
        # 2 m rear wing on a 20 m² pair is refused HERE rather than becoming
        # a penalty every optimiser step spends its budget rediscovering
        _planform_size({"b_m": float(b_rear), "S_m2": 0.5 * S_total},
                       10.0, 10.0)
        out["b_rear"] = float(b_rear)
    b_ref = b
    if size_free:
        # the span the BOX will search, not the one the family is calibrated
        # on: sizing._ends resolves the user's own band where they gave one
        # and the fractional B_FRAC_BOUNDS where they did not, so a narrowed
        # row narrows the stagger with it and the two cannot disagree.
        from .sizing import B_FRAC_BOUNDS, _ends
        lo, hi = _ends(span_bounds_m, B_FRAC_BOUNDS, b)
        b_ref = 0.5 * (float(lo) + float(hi))
    for key, attr, frac in zip(TANDEM_STAGGER_KEYS, ("dx", "dz"),
                               _TANDEM_STAGGER_FRACS):
        out[attr] = (frac * b_ref if flags.get(key) is None
                     else float(flags[key]))
    if out["dx"] == 0.0 and out["dz"] == 0.0:
        raise ValueError(
            "a tandem pair needs a stagger: dx_m and dz_m are both zero, "
            "which puts the rear wing exactly on top of the front one")
    return out


def _make_tandem_builder(chord_order: int = 0, flight_free: bool = False,
                         size_free: bool = False, tc_free: bool = False):
    def build(mission_kwargs, flags, bounds_overrides):
        from . import tandem
        planform = _tandem_planform_kwargs(
            flags, size_free=size_free,
            span_bounds_m=_size_band_kwargs(
                size_free, bounds_overrides, flags).get("span_bounds_m"))
        # a t/c-searching pair PICKS its tables off the thickness rows, so it
        # has no fixed section a chosen one could replace — the same rule
        # SECTION_KEY already states for every other by-thickness family
        section_kw = ({} if tc_free else
                      {**_section_kwargs(flags),
                       **_section_kwargs(flags, field="polar_rear",
                                         key=SECTION_AFT_KEY)})
        prob = _apply_mission(
            tandem.TandemProblem(**_chord_kwargs(chord_order, flags),
                                 **_fin_config(flags),
                                 **_flight_kwargs(flight_free),
                                 **_size_kwargs(size_free, flags),
                                 **_wing_loading_kwargs(
                                     size_free, flags,
                                     planform.get("b", 10.0)),
                                 **_size_band_kwargs(size_free,
                                                     bounds_overrides,
                                                     flags),
                                 **planform,
                                 **({"tc_free": True} if tc_free else {}),
                                 **section_kw),
            mission_kwargs)
        labels = _tandem_labels(chord_order, flight_free, size_free, tc_free,
                                flags=flags)
        bounds = _apply_overrides(prob.bounds, labels, bounds_overrides)
        return _BuiltProblem(
            problem=prob,
            callable=((lambda x: tandem.fg_tandem(x, prob)) if size_free
                      else (lambda x: tandem.objective_tandem(x, prob))),
            evaluate=lambda x: tandem.evaluate_tandem(x, prob),
            bounds=bounds, is_constrained=bool(size_free), dim=prob.dim,
            param_labels=labels, medium="air",
        )
    build.takes_section = not tc_free
    build.takes_section_aft = not tc_free    # the pair's REAR wing
    # this family GATES the aspect ratio per candidate
    # (``sizing.check_ar_limit``), so the registry may declare the
    # limit flags on it: a flag is declared where it is READ
    build.gates_ar = True
    return build


_build_tandem = _make_tandem_builder()


_CST_LABELS = tuple([f"w_upper_{i}" for i in range(4)]
                    + [f"w_lower_{i}" for i in range(4)])


def _winglet_section_labels(chord_order: int = 0, winglet: bool = True,
                            flight_free: bool = False,
                            size_free: bool = False,
                            flags: dict | None = None) -> tuple:
    """Design-vector labels of the live-XFOIL CST + wing problem.

    NOTE the chord rows land in the MIDDLE, not at the end: they belong to
    the WING block (geometry.bounds appends them there), and the section
    weights follow. The generic ``_with_chord_labels`` would put them last
    and silently mislabel every row of the section block.
    """
    return ((_WINGLET_LABELS if winglet else _TRIM_LABELS)
            + (_SIZE_LABELS if size_free else ())
            + (_FLIGHT_LABELS if flight_free else ())
            + _chord_label_block(chord_order, flags)
            + _CST_LABELS)


_WINGLET_SECTION_LABELS = _winglet_section_labels()
_WING_SECTION_LABELS = _winglet_section_labels(winglet=False)


def _make_winglet_section_builder(capped: bool, chord_order: int = 0,
                                  winglet: bool = True,
                                  flight_free: bool = False,
                                  size_free: bool = False):
    # the wing state this family builds IS an objective.Problem in one of the
    # plain winglet modes, so a FIXED blend has exactly the same home here as
    # it does on the winglet family without a designed section
    mode = ("winglet_capped" if capped else "winglet") if winglet else "trim"

    def build(mission_kwargs, flags, bounds_overrides):
        from . import winglet_section as ws
        blend = _fixed_blend_kwargs(mode, flags)
        prob = _apply_mission(
            ws.WingletSectionProblem(capped=capped, winglet=winglet,
                                     **_chord_kwargs(chord_order, flags),
                                     **_winglet_chord_kwargs(flags),
                                     **blend, **_blend_shape_flag(flags),
                                     **_junction_flag(flags, bool(blend)),
                                     **_flight_kwargs(flight_free),
                                     **_size_kwargs(size_free, flags),
                                     **_size_band_kwargs(size_free,
                                                         bounds_overrides,
                                                         flags),
                                     **_planform_kwargs(flags)),
            mission_kwargs)
        labels = _winglet_section_labels(chord_order, winglet, flight_free,
                                         size_free, flags=flags)
        bounds = _apply_overrides(prob.bounds, labels, bounds_overrides)
        return _BuiltProblem(
            problem=prob,
            callable=lambda x: ws.fg_winglet_section(x, prob),
            evaluate=lambda x: ws.evaluate_winglet_section(x, prob),
            bounds=bounds, is_constrained=True, dim=prob.dim,
            param_labels=labels, medium="air",
        )
    # tagged so the registry declares the four fixed-blend flags on this
    # family and on every twin generated from it (chord law, flight state,
    # either size mode) without a name list — the rule the water family's
    # builder already follows. Only where there IS a device to blend: the
    # wingless variant's mode is "trim", and objective.Problem refuses a
    # blend on a wing that carries no tip device.
    build.takes_fixed_blend = winglet
    # this family GATES the aspect ratio per candidate
    # (``sizing.check_ar_limit``), so the registry may declare the
    # limit flags on it: a flag is declared where it is READ
    build.gates_ar = True
    return build


def _make_wing_airfoil_builder(chord_order: int = 0,
                               flight_free: bool = False,
                               size_free: bool = False):
    def build(mission_kwargs, flags, bounds_overrides):
        from . import wing_airfoil
        prob = _apply_mission(
            wing_airfoil.WingAirfoilProblem(**_chord_kwargs(chord_order,
                                                            flags),
                                            **_flight_kwargs(flight_free),
                                            **_size_kwargs(size_free, flags),
                                            **_size_band_kwargs(
                                                size_free, bounds_overrides,
                                                flags),
                                            **_planform_kwargs(flags)),
            mission_kwargs)
        labels = _variant_labels(_WING_AIRFOIL_LABELS, chord_order,
                                 flight_free, size_free, flags=flags)
        bounds = _apply_overrides(prob.bounds, labels, bounds_overrides)
        return _BuiltProblem(
            problem=prob,
            callable=((lambda x: wing_airfoil.fg_wing_airfoil(x, prob))
                      if size_free
                      else (lambda x: wing_airfoil.objective_wing_airfoil(
                          x, prob))),
            evaluate=lambda x: wing_airfoil.evaluate_wing_airfoil(x, prob),
            bounds=bounds, is_constrained=bool(size_free), dim=prob.dim,
            param_labels=labels, medium="air",
        )
    # this family GATES the aspect ratio per candidate
    # (``sizing.check_ar_limit``), so the registry may declare the
    # limit flags on it: a flag is declared where it is READ
    build.gates_ar = True
    return build


_build_wing_airfoil = _make_wing_airfoil_builder()


_TIER_A_PLUS_LABELS = _TRIM_LABELS + ("sweep_deg", "tc")
_WINGLET_TC_LABELS = _WINGLET_LABELS + ("tc",)
#: the winglet vector + the two SUMMARY VARIABLES of the coupled section
#: library. Same names as the wingless coupled family's (_WING_AIRFOIL_LABELS
#: ends on them), because they are the same two demands on the same library.
_WINGLET_COUPLED_LABELS = _WINGLET_LABELS + ("tc_sec", "cl_sec")

_TRIM_CHORD_LABELS = _TRIM_LABELS + _CHORD_LABELS
_WINGLET_CHORD_LABELS = _WINGLET_LABELS + _CHORD_LABELS
_WINGLET_BLEND_LABELS = _WINGLET_LABELS + ("winglet_blend_frac",)
_WINGLET_WINGBLEND_LABELS = _WINGLET_BLEND_LABELS + ("wing_blend_frac",)

PROBLEM_SPECS: dict[str, ProblemSpec] = {
    "trim wing": ProblemSpec(
        "trim wing", "Trim wing (Tier A, 3-D)", "air", False,
        _TRIM_LABELS, False, _build_objective("trim", _TRIM_LABELS),
        description="Single trapezoidal wing, taper + linear twist, trimmed "
                    "to CL_target; maximise L/D.",
        flags=("mach", "ground_h_m", "slipstream"),
        uses_mission=True,
        mission_fields=("W_N", "V", "altitude_m"),
    ),
    "wing (free chord law)": ProblemSpec(
        "wing (free chord law)", "Trim wing, free chord law (6-D)", "air",
        False, _TRIM_CHORD_LABELS, False,
        _build_objective("trim", _TRIM_CHORD_LABELS,
                         chord_order=_CHORD_ORDER),
        description="Trim wing whose chord distribution is a free cubic on "
                    "top of the straight-taper baseline, at UNCHANGED area — "
                    "the planform is no longer restricted to a trapezoid, so "
                    "the loading can approach elliptic (e ~ 0.999 vs 0.984 "
                    "for the best straight taper). A law that collapses the "
                    "chord anywhere is refused (penalty).",
        flags=("mach", "ground_h_m", "slipstream", "chord_max_frac",
               CHORD_LAW_KEY),
        uses_mission=True,
        mission_fields=("W_N", "V", "altitude_m"),
    ),
    "winglet + free chord law": ProblemSpec(
        "winglet + free chord law",
        "Winglet + free chord law (Tier A+, 8-D)", "air", False,
        _WINGLET_CHORD_LABELS, False,
        _build_objective("winglet", _WINGLET_CHORD_LABELS,
                         chord_order=_CHORD_ORDER),
        description="Nonplanar VLM wing + winglet with the chord distribution "
                    "free (cubic law, area held). The winglet chord follows "
                    "the FLOWN tip chord, so the planform and the tip device "
                    "are co-designed.",
        flags=("mach", "chord_max_frac", CHORD_LAW_KEY),
        uses_mission=True,
        mission_fields=("W_N", "V", "altitude_m"),
    ),
    "wing t/c + sweep": ProblemSpec(
        "wing t/c + sweep", "Wing + section t/c + sweep (Tier A+, 5-D)",
        "air", False,
        _TIER_A_PLUS_LABELS, False,
        _build_objective("tier_a_plus", _TIER_A_PLUS_LABELS),
        description="Trim wing with quarter-chord sweep and NACA 24XX section "
                    "thickness as design variables (real XFOIL polar family; "
                    "simple sweep theory on the section slope).",
        flags=("mach", "ground_h_m", "slipstream"),
        uses_mission=True,
        mission_fields=("W_N", "V", "altitude_m"),
    ),   # (winglet-family specs declare flags=("mach",): the VLM honours the
    #      PG section-slope correction; ground/slipstream are LLT-only and
    #      fail loudly in evaluate() if forced through the API.)
    "free planform (aircraft)": ProblemSpec(
        "free planform (aircraft)", "Free planform aircraft (Tier B, 6-D)",
        "air", True,
        _AIRCRAFT_LABELS,
        False, _build_aircraft,
        description="Weight-coupled wing sizing (span + area free); maximise "
                    "payload L/D s.t. root-bending stress margin >= 0.",
        constraint_labels=("root-bending stress margin",),
        uses_mission=True,
        mission_fields=("W_N", "V", "altitude_m"),
    ),
    "winglet": ProblemSpec(
        "winglet", "Winglet (Tier A+, 5-D)", "air", False,
        _WINGLET_LABELS, False, _build_objective("winglet", _WINGLET_LABELS),
        description="Nonplanar VLM wing + winglet (free span extension); "
                    "maximise trimmed L/D.",
        flags=("mach",),
        uses_mission=True,
        mission_fields=("W_N", "V", "altitude_m"),
    ),
    "winglet_capped": ProblemSpec(
        "winglet_capped", "Winglet, span-capped (Tier A+, 5-D)", "air", False,
        _WINGLET_LABELS, False,
        _build_objective("winglet_capped", _WINGLET_LABELS),
        description="As winglet, but total projected span held fixed — the "
                    "wing shrinks to pay for the winglet's projection.",
        flags=("mach",),
        uses_mission=True,
        mission_fields=("W_N", "V", "altitude_m"),
    ),
    "winglet, blended (span-capped)": ProblemSpec(
        "winglet, blended (span-capped)",
        "Blended winglet, span-capped (Tier A+, 6-D)", "air", False,
        _WINGLET_BLEND_LABELS, False,
        _build_objective("winglet_capped_blended", _WINGLET_BLEND_LABELS,
                         junction_drag=True),
        description="Winglet whose root TRANSITION is a design variable: the "
                    "device leaves the wing plane tangentially over a "
                    "constant-radius arc instead of a corner. Span-capped, "
                    "because a blend reaches further outboard than h·cos(cant) "
                    "and free span would pay it for the rake. The corner's "
                    "interference drag is charged (junction.py, a "
                    "reduced-order add-on) — without it a blend is only a "
                    "differently drawn wake and the trade is fake.",
        flags=("mach", "junction_drag", "blend_shape", "wing_blend_frac"),
        uses_mission=True,
        mission_fields=("W_N", "V", "altitude_m"),
    ),
    "winglet, blended into the wing (span-capped)": ProblemSpec(
        "winglet, blended into the wing (span-capped)",
        "Blended winglet, wing-side transition, span-capped (Tier A+, 7-D)",
        "air", False,
        _WINGLET_WINGBLEND_LABELS, False,
        _build_objective("winglet_capped_blended_wing",
                         _WINGLET_WINGBLEND_LABELS, junction_drag=True),
        description="As the blended winglet, but the transition may START "
                    "INBOARD OF THE TIP: x[6] is the fraction of the "
                    "semi-span over which the OUTER WING turns up into the "
                    "device. That is the freedom the 6-D blend does not have "
                    "— confined to the winglet the turning arc is at most "
                    "blend_frac·h, so the blend radius stays a fraction of a "
                    "tip chord and the fillet credit saturates almost at "
                    "once, which is why the span-capped optimiser settles on "
                    "a barely-blended corner. Developed span is the "
                    "invariant: a wing-side blend trades PROJECTED span for "
                    "height, it never deletes wing, and the cap is solved on "
                    "the whole developed line (geometry.developed_semispan).",
        flags=("mach", "junction_drag", "blend_shape"),
        uses_mission=True,
        mission_fields=("W_N", "V", "altitude_m"),
    ),
    "winglet + t/c": ProblemSpec(
        "winglet + t/c", "Winglet + section t/c (Tier A+, 6-D)", "air", False,
        _WINGLET_TC_LABELS, False,
        _build_objective("winglet_tc", _WINGLET_TC_LABELS),
        description="Nonplanar VLM wing + winglet with the section thickness "
                    "co-optimised (NACA 24XX polar family); maximise trimmed "
                    "L/D. Sweep is not modelled in the VLM.",
        flags=("mach",),
        uses_mission=True,
        mission_fields=("W_N", "V", "altitude_m"),
    ),
    "winglet_capped + t/c": ProblemSpec(
        "winglet_capped + t/c",
        "Winglet, span-capped + section t/c (Tier A+, 6-D)", "air", False,
        _WINGLET_TC_LABELS, False,
        _build_objective("winglet_capped_tc", _WINGLET_TC_LABELS),
        description="As winglet + t/c, but total projected span held fixed — "
                    "the wing shrinks to pay for the winglet's projection.",
        flags=("mach",),
        uses_mission=True,
        mission_fields=("W_N", "V", "altitude_m"),
    ),
    # ...and the same two families with the section taken from the
    # PRE-OPTIMISED CST library instead of the NACA 24XX bank: two summary
    # variables (structural depth, design lift) rather than one thickness,
    # and every member already drag-minimised at its own design cl. This is
    # the hierarchical decomposition the wingless "wing+airfoil (coupled)"
    # family flies (wing_airfoil.py docstring A), now with a tip device —
    # so the library's ~0.5 ms lookup buys a designed section at VLM speed,
    # where "winglet + airfoil (XFOIL)" pays seconds per new shape.
    "winglet + airfoil (coupled)": ProblemSpec(
        "winglet + airfoil (coupled)",
        "Winglet + coupled section library (Tier A+, 7-D)", "air", False,
        _WINGLET_COUPLED_LABELS, False,
        _build_objective("winglet_coupled", _WINGLET_COUPLED_LABELS),
        description="Nonplanar VLM wing + winglet whose SECTION is demanded "
                    "from the pre-optimised (t/c x cl) CST library by two "
                    "summary variables, in the hierarchical-decomposition "
                    "sense: the wing level commands structural depth and "
                    "design lift, a section-level BO already delivered the "
                    "shape. cl_sec selects WHICH pre-optimised member, it "
                    "does not force the wing to operate there — the wing "
                    "still trims to CL_target and pays the mismatch through "
                    "the polar. Intermediate members are bilinear "
                    "interpolations, never XFOIL runs.",
        flags=("mach",),
        uses_mission=True,
        mission_fields=("W_N", "V", "altitude_m"),
    ),
    "winglet_capped + airfoil (coupled)": ProblemSpec(
        "winglet_capped + airfoil (coupled)",
        "Winglet, span-capped + coupled section library (Tier A+, 7-D)",
        "air", False,
        _WINGLET_COUPLED_LABELS, False,
        _build_objective("winglet_capped_coupled", _WINGLET_COUPLED_LABELS),
        description="As winglet + airfoil (coupled), but total projected "
                    "span held fixed — the wing shrinks to pay for the "
                    "winglet's projection.",
        flags=("mach",),
        uses_mission=True,
        mission_fields=("W_N", "V", "altitude_m"),
    ),
    "wing + airfoil (XFOIL)": ProblemSpec(
        "wing + airfoil (XFOIL)",
        "Wing + free-form CST section, live XFOIL (11-D)", "air", True,
        _WING_SECTION_LABELS, True,
        _make_winglet_section_builder(capped=False, winglet=False),
        n_constraints=2,
        description="Shape the SECTION and the PLANFORM together, with no tip "
                    "device: taper + twist trimmed on the candidate CST "
                    "section's own polar, built by a live (cached) XFOIL "
                    "sweep each evaluation; maximise L/D s.t. t/c and |Cm| "
                    "margins >= 0 (SLOW — seconds per new section). This is "
                    "the winglet problem with the two tip-device variables "
                    "REMOVED, not set to zero: the optimiser never spends "
                    "budget on them. Add the free chord law for the full "
                    "section + planform co-design.",
        constraint_labels=("t/c margin", "|Cm| margin"),
        uses_mission=True,
        mission_fields=("W_N", "V", "altitude_m"),
        has_blocks=True,
    ),
    "winglet + airfoil (XFOIL)": ProblemSpec(
        "winglet + airfoil (XFOIL)",
        "Winglet + free-form CST section, live XFOIL (13-D)", "air", True,
        _WINGLET_SECTION_LABELS, True,
        _make_winglet_section_builder(capped=False), n_constraints=2,
        description="Full-fidelity co-design: wing+winglet VLM trim with the "
                    "candidate CST section's polar built by a live (cached) "
                    "XFOIL sweep each evaluation; maximise L/D s.t. t/c and "
                    "|Cm| margins >= 0 (SLOW — seconds per new section). "
                    "Declares wing vs section blocks for the portfolio "
                    "optimiser.",
        constraint_labels=("t/c margin", "|Cm| margin"),
        uses_mission=True,
        mission_fields=("W_N", "V", "altitude_m"),
        has_blocks=True,
    ),
    "winglet_capped + airfoil (XFOIL)": ProblemSpec(
        "winglet_capped + airfoil (XFOIL)",
        "Winglet, span-capped + CST section, live XFOIL (13-D)", "air", True,
        _WINGLET_SECTION_LABELS, True,
        _make_winglet_section_builder(capped=True), n_constraints=2,
        description="As winglet + airfoil (XFOIL), but total projected span "
                    "held fixed.",
        constraint_labels=("t/c margin", "|Cm| margin"),
        uses_mission=True,
        mission_fields=("W_N", "V", "altitude_m"),
        has_blocks=True,
    ),
    "tail": ProblemSpec(
        "tail", "Wing + H-tail (Tier / Phase 4, 5-D)", "air", True,
        _TAIL_LABELS + ("l_t_m",),
        False, _build_tail,
        description="Coupled wing+tail 2-D trim; maximise L/D s.t. static "
                    "margin >= SM_min. Tail layout (conventional / T-tail / "
                    "V-tail / canard) and the control type are configuration "
                    "flags; the arm l_t is a design variable here.",
        constraint_labels=("static margin - SM_min",),
        flags=TAIL_CONFIG_KEYS + FUSELAGE_KEYS + (TAIL_CG_KEY, TAIL_MOUNT_KEY,
                                  TAIL_HEIGHT_KEY) + TAIL_LIMIT_KEYS,
        uses_mission=True,
        mission_fields=("W_N", "V", "altitude_m"),
    ),
    "tail (fixed arm)": ProblemSpec(
        "tail (fixed arm)", "Wing + H-tail, arm CHOSEN (4-D)", "air", True,
        _TAIL_LABELS,
        False, _make_tail_builder(fixed_arm=True),
        description="As tail, but the distance from the wing is a chosen "
                    "configuration value (flag l_t_m) instead of a design "
                    "variable — the honest way to fix it, since collapsing "
                    "a bound to zero width breaks the samplers and silently "
                    "degrades BO to random search.",
        constraint_labels=("static margin - SM_min",),
        flags=TAIL_CONFIG_KEYS + FUSELAGE_KEYS + ("l_t_m", TAIL_CG_KEY,
                                  TAIL_MOUNT_KEY, TAIL_HEIGHT_KEY)
        + TAIL_LIMIT_KEYS,
        uses_mission=True,
        mission_fields=("W_N", "V", "altitude_m"),
    ),
    # ...and the same two with the TAIL as a designed surface: its taper,
    # aspect ratio and washout join the vector (and its own chord law rides
    # the chord modifier), so both surfaces are shaped by the search instead
    # of only the wing. A tip device on the tail is NOT here — these lifting
    # lines are planar; that lives in the nonplanar wing+tail family.
    "tail [designed tail]": ProblemSpec(
        "tail [designed tail]", "Wing + DESIGNED tail (8-D)", "air", True,
        _TAIL_LABELS + ("l_t_m",) + _DESIGNED_TAIL_LABELS,
        False, _make_tail_builder(fixed_arm=False, tail_free=True),
        description="As tail, with the stabiliser designed rather than "
                    "assumed: its taper, aspect ratio and washout are design "
                    "variables beside the wing's. Its INCIDENCE stays the "
                    "trim unknown, so the twist freedom is the washout — a "
                    "root twist would be the same number as i_t twice. At "
                    "taper 1, AR 4 and zero washout it reproduces the "
                    "published rectangular tail bit-for-bit.",
        constraint_labels=("static margin - SM_min",),
        flags=TAIL_CONFIG_KEYS + FUSELAGE_KEYS + (TAIL_CG_KEY, TAIL_MOUNT_KEY,
                                  TAIL_HEIGHT_KEY) + TAIL_LIMIT_KEYS,
        uses_mission=True,
        mission_fields=("W_N", "V", "altitude_m"),
    ),
    "tail (fixed arm) [designed tail]": ProblemSpec(
        "tail (fixed arm) [designed tail]",
        "Wing + DESIGNED tail, arm CHOSEN (7-D)", "air", True,
        _TAIL_LABELS + _DESIGNED_TAIL_LABELS,
        False, _make_tail_builder(fixed_arm=True, tail_free=True),
        description="The designed tail with the arm as a chosen value "
                    "(flag l_t_m) instead of a design variable.",
        constraint_labels=("static margin - SM_min",),
        flags=TAIL_CONFIG_KEYS + FUSELAGE_KEYS + ("l_t_m", TAIL_CG_KEY,
                                  TAIL_MOUNT_KEY, TAIL_HEIGHT_KEY)
        + TAIL_LIMIT_KEYS,
        uses_mission=True,
        mission_fields=("W_N", "V", "altitude_m"),
    ),
    "tandem": ProblemSpec(
        "tandem", "Tandem two-wing system (Tier B, 10-D)", "air", False,
        _TANDEM_LABELS, False, _build_tandem,
        description="Two coupled wings (front + rear): per-wing taper, 3-knot "
                    "twist laws, area split and decalage; system-trimmed L/D "
                    "(mutual-induction LLT, Munk-stagger-gated). The STAGGER "
                    "— how far aft and how far above the front wing the rear "
                    "one sits — is yours to state in metres (dx_m / dz_m), "
                    "not something the run chooses, and so is the rear "
                    "wing's own SPAN (b_rear_m) when it is not the front's.",
        uses_mission=True,
        mission_fields=("W_N", "V", "altitude_m"),
        # the pair's VERTICAL SURFACE, declared where it is read: the fin
        # is sized against the stagger, charged in ``tandem.evaluate_tandem``
        # and flown by the rebuild, so its four keys travel here exactly as
        # they travel on a wing+tail (``TAIL_CONFIG_KEYS``)
        flags=(TANDEM_STAGGER_KEYS + TANDEM_SPAN_KEYS
               + FIN_SHAPE_KEYS + (FIN_PRESENCE_KEY,
                                   TANDEM_FIN_BOOM_KEY)),
    ),
    "tandem + t/c": ProblemSpec(
        "tandem + t/c", "Tandem pair + per-wing section t/c (Tier B, 12-D)",
        "air", False,
        _TANDEM_LABELS + ("tc_front", "tc_rear"), False,
        _make_tandem_builder(tc_free=True),
        description="As tandem, with each wing's SECTION THICKNESS a design "
                    "variable off the NACA 24XX polar family — two rows, one "
                    "per wing, because this pair already gives each surface "
                    "its own span, chord law and section. A pair searching "
                    "its thickness picks its own tables, so it takes no "
                    "chosen section.",
        uses_mission=True,
        mission_fields=("W_N", "V", "altitude_m"),
        # the pair's VERTICAL SURFACE, declared where it is read: the fin
        # is sized against the stagger, charged in ``tandem.evaluate_tandem``
        # and flown by the rebuild, so its four keys travel here exactly as
        # they travel on a wing+tail (``TAIL_CONFIG_KEYS``)
        flags=(TANDEM_STAGGER_KEYS + TANDEM_SPAN_KEYS
               + FIN_SHAPE_KEYS + (FIN_PRESENCE_KEY,
                                   TANDEM_FIN_BOOM_KEY)),
    ),
    "hydrofoil": ProblemSpec(
        "hydrofoil", "Hydrofoil under a free surface (Tier C, 6-D)", "water",
        True,
        _HYDROFOIL_LABELS,
        False, _build_hydrofoil,
        description="Cavitation-constrained lifting surface; maximise L/D "
                    "s.t. cavitation margin >= 0. Speed and depth are design "
                    "variables (edit their box bounds, not a mission field).",
        constraint_labels=("cavitation margin",),
        flags=((WATER_KEY, WEIGHT_KEY) + STRUT_KEYS + WET_SIZE_KEYS
               + WATER_FLIGHT_KEYS + (WATER_RE_KEY,)),
    ),
    "hydrofoil + winglet": ProblemSpec(
        "hydrofoil + winglet",
        "Hydrofoil + tip device under a free surface (8-D)", "water", True,
        _HYDROFOIL_WINGLET_LABELS,
        False, _build_hydrofoil_winglet,
        description="As hydrofoil, but nonplanar: an imaged VLM carries a "
                    "tip device whose cant is SIGNED (+ up towards the "
                    "surface, − down away from it). Cavitation is checked "
                    "per panel at the LOCAL submergence, so canting up costs "
                    "static head and canting down buys it — the trade the "
                    "planar solver cannot see. Flag draught_max_m caps how "
                    "deep the assembly may reach and adds a SECOND margin; "
                    "it is the water answer to the air families' span cap, "
                    "and deliberately not the same quantity.",
        constraint_labels=("cavitation margin",),
        # ...and WHAT SHAPE that device is (``winglet_type``), which is what
        # narrows the signed band above: a vertical fence is |cant| 84-90,
        # and under water it is the DOWN half of that (WATER_TIP_DIRECTION).
        # Undeclared, the flag was dropped on the way to the run and the V3
        # shape menu refused the entry rather than print "90 deg" over a
        # -90…90 search — the one tip-device shape no foil could be given.
        flags=((WATER_KEY, WEIGHT_KEY, DRAUGHT_MAX_KEY)
               + STRUT_KEYS + WET_SIZE_KEYS + WATER_FLIGHT_KEYS
               # the FOIL's own cant and sweep, declared where they are READ
               # (this family is an imaged LATTICE; the planar `hydrofoil`
               # above is a monoplane Fourier solve and declares neither)
               + WING_CANT_KEYS
               + (WATER_RE_KEY, "winglet_type")),
    ),
    # THREE car families, not six. The reference AREA used to be the thing a
    # fixed-area/free-area PAIR disagreed about, which made "how big is this
    # wing" a question answered by picking a problem name. It is a design
    # variable on all three now, asked where every other band is asked — the
    # ``S_m2`` row of the design box — so the pair collapses and the family
    # keeps its own name. Two consequences are carried in the descriptions
    # below because they are forced rather than chosen: a downforce
    # COEFFICIENT is referenced to the very area being searched (so ``cz`` is
    # refused and the score is a force or a ratio), and a COEFFICIENT drag
    # allowance is not an allowance on anything the car can feel.
    "car rear wing": ProblemSpec(
        "car rear wing", "Car rear wing over the track (8-D)", "air", True,
        _CAR_WING_AREA_LABELS,
        False, _build_car_wing, n_constraints=1,
        description="Inverted wing near the ground with endplates. Both of "
                    "the wing's DIMENSIONS are designed — the reference AREA "
                    "and the SPAN, each a length band in the design box, "
                    "because a rear wing's size is set by a regulation or by "
                    "the bodywork and never by a fraction of some default — "
                    "and the endplate's height is a length for the same "
                    "reason. The score is CZ/CD, the downforce made per unit "
                    "drag: a downforce COEFFICIENT would be referenced to "
                    "the area being searched (so maximising it just shrinks "
                    "the wing), and a downforce FORCE with no ceiling is "
                    "maximised by spending drag until the section stalls. "
                    "Nothing is budgeted by default; a drag ceiling and a "
                    "downforce floor are limits you may state, in newtons. "
                    "The MOUNT layout decides where along the span the "
                    "plates take the load, which changes the beam and what "
                    "the extra sheets cost.",
        constraint_labels=("deflection margin",),
        flags=CAR_WING_KEYS,
    ),
    "car rear wing (two-element)": ProblemSpec(
        "car rear wing (two-element)",
        "Car rear wing, slotted two-element section (12-D)", "air", True,
        _CAR_WING_MULTI_AREA_LABELS,
        False, _build_car_wing_multi, n_constraints=1,
        description="As the car rear wing, but the wing flies a SLOTTED "
                    "two-element section instead of a single one. The flap "
                    "is in the SECTION, not in the lattice: a slot is a "
                    "chordwise interaction at one spanwise station, which one "
                    "chordwise panel per strip cannot represent, so the two "
                    "elements are solved in 2-D (panel2d.py, two bodies, one "
                    "dense system) and enter the VLM as a polar. Four rows "
                    "join the design vector — the flap's chord fraction and "
                    "deflection, and the gap and overlap that place its "
                    "leading edge — and ONE endplate carries the whole "
                    "assembly, because a second sheet at the same spanwise "
                    "station makes the influence matrix exactly singular. "
                    "MEASURED by carwing_multi.compare_split against the "
                    "single-element family at matched area and matched drag: "
                    "the slot LOSES below a drag budget of about 0.018, buys "
                    "+4.2 % at the published one, and saves 5-25 % of the "
                    "drag at a demanded downforce (the saving grows with the "
                    "downforce asked for) — what it really buys is a "
                    "ceiling, which is why it is worth most against a "
                    "stated drag limit rather than against none.",
        constraint_labels=("deflection margin",),
        flags=CAR_WING_MULTI_KEYS,
    ),
    "car rear wing + endplates": ProblemSpec(
        "car rear wing + endplates",
        "Car rear wing, endplates designed (11-D)", "air", True,
        _CAR_ENDPLATE_AREA_LABELS,
        False, _build_car_endplate, n_constraints=3,
        description="As the car rear wing, but the ENDPLATES are a designed "
                    "part rather than a fence of free height: their chord, "
                    "thickness and toe are design variables and their "
                    "section family is a flag. The plates are also the load "
                    "path — that is what a rear wing is bolted to the car "
                    "by — so the plate must REACH its attachment deck, "
                    "which is a signed constraint on a distance in metres, "
                    "and it then carries the lateral load (its own toe side "
                    "force plus a yaw case) as a cantilever whose arm the "
                    "mount decides. The reference AREA being designed is "
                    "what makes the reach bite hardest: a smaller wing is a "
                    "shorter chord, and the plate still has to get down to "
                    "the deck.",
        constraint_labels=("wing deflection margin",
                           "endplate deflection margin",
                           "endplate reach margin"),
        flags=CAR_ENDPLATE_KEYS,
    ),
    "airfoil (section)": ProblemSpec(
        "airfoil (section)", "Airfoil section, real XFOIL (Tier B, 8-D)",
        "air", True, (), True, _build_airfoil, n_constraints=2,
        description="CST section drag minimisation at fixed cl via XFOIL; "
                    "maximise -cd s.t. t/c and |Cm| margins >= 0 (SLOW).",
        constraint_labels=("t/c margin", "|Cm| margin"),
        # the 2-D contract, the objective it maximises and the optional
        # section+twist wing mode. Honoured by ``_build_airfoil`` (and, for
        # the objective keys, by ``run`` itself) but deliberately NOT in
        # ``flags``, which drives the shell's WING physics toggles — this
        # problem has no wing. Declared here so ``check_flags`` can accept
        # what the builder really reads; see :func:`_declare_builder_flags`.
        builder_flags=(AIRFOIL_FLAG_KEYS + AIRFOIL_OBJECTIVE_FLAG_KEYS
                       + AIRFOIL_WING_FLAG_KEYS),
    ),
    "wing+airfoil (coupled)": ProblemSpec(
        "wing+airfoil (coupled)", "Coupled wing + airfoil (summary vars, 5-D)",
        "air", False,
        _WING_AIRFOIL_LABELS,
        False, _build_wing_airfoil,
        description="Wing trimmed on a pre-optimised CST section library "
                    "indexed by (t/c, cl); maximise L/D.",
        uses_mission=True,
        mission_fields=("W_N", "V", "altitude_m"),
    ),
    "mission wing": ProblemSpec(
        "mission wing", "Mission wing (V + altitude free, 5-D)", "air", False,
        _MISSION_LABELS, False, _build_objective("mission", _MISSION_LABELS),
        description="Trim wing where speed and altitude are design variables; "
                    "CL_target recomputed per candidate from the design weight.",
        flags=("mach", "ground_h_m", "slipstream"),
        uses_mission=True,
        mission_fields=("W_N",),
    ),
}


# =====================================================================
# Wing + tip device + tail (nonplanar family)
# =====================================================================
# The published ``tail`` / ``tail (fixed arm)`` problems couple wing and tail
# through planar lifting lines, so a winglet cannot appear in them at all.
# wingtail.py puts the tail INTO the nonplanar VLM, which makes every wing
# freedom composable with the tail — and each combination changes the design
# DIMENSION, so each is its own ProblemSpec (the package rule that gave
# "tail (fixed arm)" its own entry).
#
# The specs are GENERATED from one table so labels, bounds, constraint names
# and flags cannot drift between 30 near-identical problems. The two
# combinations that would duplicate the published problems (no tip device, no
# free thickness, tail at its default height) are deliberately NOT generated:
# those keep their lifting-line solver and their published numbers.

#: tip-device options in the tail family: (name fragment, winglet, capped,
#: blended). "free" and "capped" are the span-accounting pair the winglet
#: problems define; "blended" additionally designs the root transition.
_TAIL_WINGLETS = {
    None: ("", dict(winglet=False, capped=False, blended=False)),
    "free": ("winglet", dict(winglet=True, capped=False, blended=False)),
    "capped": ("winglet (span-capped)",
               dict(winglet=True, capped=True, blended=False)),
    "blended": ("winglet, blended (span-capped)",
                dict(winglet=True, capped=True, blended=True)),
}

_WING_TAIL_NOTE = (
    "Solved by the NONPLANAR wing+tail VLM (wingtail.py): one influence "
    "matrix over wing, tip device and tail, trimmed in lift AND pitch, with "
    "the static margin read off the same coupled solve. It is a different "
    "aero core from the published lifting-line 'tail' problem — measured "
    "over the design box they agree to within 2.4 % in L/D (mean 0.8 %) and "
    "0.07 in static margin (mean 0.03), so compare within one of them, not "
    "across."
)


#: how much of the TAIL is designed. "fixed" is the published surface — a
#: rectangle at AR 4 whose only freedoms are its area and its arm. The other
#: two make it a surface in its own right (wingtail.WingTailProblem):
#: "planform" frees its taper, aspect ratio and washout (and gives it its own
#: chord law wherever the wing carries one), "planform+tip" adds a tip device
#: ON THE TAIL — height and cant, panelised by the same VLM code as the
#: wing's. Which is what "design the tail with the same freedoms as the wing"
#: means here.
_TAIL_DESIGNS: dict[str, tuple[str, dict]] = {
    "fixed": ("", {}),
    "planform": ("designed tail", dict(tail_free=True)),
    "planform+tip": ("designed tail + tip device",
                     dict(tail_free=True, tail_winglet=True)),
}

#: the option keys above, public so a shell can offer exactly these and no
#: more (the same contract PLANFORM_KEYS / WATER_KINDS have)
TAIL_DESIGNS: tuple = tuple(_TAIL_DESIGNS)

#: ON THE WING — its own cant and sweep, STATED or SEARCHED.
#:
#: The stated pair is :data:`WING_CANT_KEYS`, a flag on every lattice-backed
#: family. The searched pair is two rows on the design box
#: (``wingtail.WingTailProblem.cant_free``) and therefore a different
#: dimension, which is why it is a variant axis here and not a flag — the
#: same rule "tail (fixed arm)" exists for.
#:
#: WHY IT IS WORTH A DESIGN VARIABLE: the wing is the only surface in this
#: package that can carry a dihedral, and without one every bit of a
#: design's ``Cl_beta`` comes from the fin and the tip device — which is a
#: spiral mode no fin SIZE can converge, because a bigger fin raises the
#: yaw stiffness and the dihedral effect together (measured on the SCORED
#: lattice, `tail + winglet` box centre: ``Cl_beta*Cn_r - Cn_beta*Cl_r``
#: -0.002197 planar, +0.005139 at 3 deg of dihedral, for 0.21 % of L/D).
#:
#: And a row nothing PRICES is worse than no row: over 42 paired searches
#: an L/D objective spends this freedom on the anhedral bound and comes
#: back divergent (RESULTS_SESSION67_WING_CANT.md). The composite's
#: ``spiral`` criterion is what makes it worth searching.
#: FOUR STATES, because the dihedral and the sweep are two questions. They
#: were one axis with two states — stated, or both searched — and that made
#: "give me a static margin without giving the optimiser a roll lever"
#: unaskable. Each half is its own dimension, so each is its own registered
#: problem: the same rule `tail (fixed arm)` exists for, and the same shape
#: ``_PLATE_FREEDOMS`` already uses for the endplate's two freedoms.
#:
#: The ``free`` FRAGMENT is still exactly "free cant". Every published
#: free-cant name, every stored run config and RESULTS_SESSION67/69/72/76/77
#: name it, so the both-state keeps its identity and the halves get new
#: fragments of their own.
_WING_CANTS: dict[str, tuple[str, dict]] = {
    "fixed": ("", {}),
    "dihedral": ("free dihedral", dict(dihedral_free=True)),
    "sweep": ("free sweep", dict(sweep_free=True)),
    "free": ("free cant", dict(dihedral_free=True, sweep_free=True)),
}

#: the option keys above, public for the same reason TAIL_DESIGNS is
WING_CANTS: tuple = tuple(_WING_CANTS)

#: which design-vector ROW each freedom above puts in the box. One table, so
#: "what does this state search" and "what does it still let you state" are
#: two readings of the same fact rather than two lists that can drift.
_CANT_ROW_OF: dict[str, str] = {"dihedral_free": "wing_dihedral_deg",
                                "sweep_free": "wing_sweep_deg"}


#: how each state reads in a family's own description. One sentence per
#: state, keyed by what the vector actually holds, so the prose cannot claim
#: a row the box does not carry.
_CANT_DESCRIPTIONS: dict[tuple, str] = {
    (): "",
    ("wing_dihedral_deg",): "the WING's own dihedral",
    ("wing_sweep_deg",): "the WING's own quarter-chord sweep",
    ("wing_dihedral_deg", "wing_sweep_deg"):
        "the WING's own dihedral and quarter-chord sweep",
}


def _cant_description(cant: str) -> str:
    """The description fragment for a :data:`WING_CANTS` state."""
    return _CANT_DESCRIPTIONS[searched_cant_keys(cant)]


def searched_cant_keys(cant: str) -> tuple:
    """The design-vector rows a :data:`WING_CANTS` state searches, in vector
    order (dihedral first, always)."""
    kw = _WING_CANTS[cant][1]
    return tuple(row for attr, row in _CANT_ROW_OF.items() if kw.get(attr))


def stated_cant_keys(cant: str) -> tuple:
    """The :data:`WING_CANT_KEYS` a family in this state still DECLARES.

    The complement of :func:`searched_cant_keys`, and the reason the halves
    are worth having: a family that searches the dihedral still takes a
    STATED sweep as an ordinary flag, and the other way round. ``check_flags``
    is what refuses the half the vector already carries.
    """
    searched = set(searched_cant_keys(cant))
    return tuple(k for k in WING_CANT_KEYS if k not in searched)


def _wing_tail_name(wl: str | None, tc: bool, arm: str, height: str,
                    design: str = "fixed", cant: str = "fixed") -> str:
    parts = ["tail"]
    frag = _TAIL_WINGLETS[wl][0]
    if frag:
        parts.append(frag)
    if tc:
        parts.append("t/c")
    name = " + ".join(parts)
    qual = ([] if arm == "free" else ["fixed arm"]) + \
        ([] if height == "fixed" else ["free height"]) + \
        ([] if design == "fixed" else [_TAIL_DESIGNS[design][0]]) + \
        ([] if cant == "fixed" else [_WING_CANTS[cant][0]])
    return f"{name} [{', '.join(qual)}]" if qual else name


#: ASK THE AEROPLANE TO FLY TO A STANDARD — MIL-F-8785C Class I, Category B,
#: as five signed constraint rows (:mod:`aerobo.handling`). A VALUE flag, not
#: a toggle: the value IS the level.
HANDLING_LEVEL_KEY = "handling_level"

#: the rungs a shell may offer, LOOSEST FIRST. Read off
#: :data:`handling.LADDER` and not sorted here: rung 3 asks for convergence
#: rather than for one of the standard's workload grades, so strictness no
#: longer runs with the level number and a menu that sorted by the number
#: would be sorting by nothing.
def handling_levels() -> tuple[int, ...]:
    from . import handling as _hq

    got = tuple(_hq.LADDER)
    assert set(got) == set(_hq.LEVELS), "LADDER and LEVELS disagree"
    return got


def _handling_kwargs(flags: dict | None) -> dict:
    """``handling_level`` for the problem, or nothing.

    ``None`` and absent both mean "no handling rows", which is what every
    published run has and what keeps them bit-for-bit. Anything else is
    validated by :func:`handling.level_of` when the problem is constructed —
    HERE would be a second place to keep in step with it.
    """
    lvl = (flags or {}).get(HANDLING_LEVEL_KEY)
    return {} if lvl is None else {"handling_level": lvl}


def _make_wing_tail_builder(kw: dict, chord_order: int = 0,
                            flight_free: bool = False,
                            size_free: bool = False):
    def build(mission_kwargs, flags, bounds_overrides):
        from . import wingtail
        cfg = dict(kw)
        cfg.update(_flight_kwargs(flight_free))
        cfg.update(_size_kwargs(size_free, flags))
        # the sized wing+tail families declare the loading flags like every
        # other sized family; until this line they DECLARED them and dropped
        # them, so a stated W/S changed nothing on any nonplanar variant
        cfg.update(_wing_loading_kwargs(size_free, flags, 10.0))
        cfg.update(_size_band_kwargs(size_free, bounds_overrides, flags))
        cfg.update(_tail_config(flags))
        cfg.update(_wing_cant_kwargs(flags))
        cfg.update(_fuselage_kwargs(flags))
        if cfg.pop("_fixed_arm", False):
            arm = (flags or {}).get("l_t_m")
            from . import tail as tailmod
            cfg["l_t_fixed"] = float(
                tailmod.default_arm(_wing_span_of(flags))
                if arm is None else arm)
        else:
            cfg.update(_arm_band_kwargs(bounds_overrides))
        cfg.update(_area_band_kwargs(bounds_overrides))
        if cfg.get("free_height"):        # see _make_wing_tail_builder
            cfg.update(_height_band_kwargs(bounds_overrides))
        cfg.update(_chord_kwargs(chord_order, flags))
        cfg.update(_planform_kwargs(flags))
        cfg.update(_blend_shape_flag(flags))
        cfg.update(_blend_frac_kwargs(flags))
        cfg.update(_junction_flag(
            flags, _blend_default_on(flags, bool(kw.get("blended")))))
        cfg.update(_winglet_chord_kwargs(flags))
        cfg["winglet_cant_bounds"] = wingtail.winglet_cant_bounds_for(
            (flags or {}).get("winglet_type"), bool(kw.get("capped")))
        # ...and each device's HEIGHT band, gated on the device
        # the family actually carries — the Problem refuses a
        # band for a surface with no device.
        cfg.update(_winglet_h_band_kwargs(
            bounds_overrides,
            (("winglet_h_frac",) if cfg.get("winglet") else ())
            + (("winglet_h_frac_t",)
               if cfg.get("tail_winglet") else ())))
        if cfg.get("tail_winglet"):
            cfg.update(_tail_cant_kwargs(flags))
            cfg.update(_tail_blend_kwargs(flags))
        cfg.update(_tail_layout_kwargs(flags))
        cfg.update(_handling_kwargs(flags))
        if not cfg.get("tc_free"):
            # a t/c variant SELECTS its section as it searches; the others
            # fly one table per surface, so a chosen section replaces it
            cfg.update(_section_kwargs(flags))
            cfg.update(_section_kwargs(flags, field="polar_tail",
                                       key=SECTION_AFT_KEY))
        prob = _apply_mission(wingtail.WingTailProblem(**cfg), mission_kwargs)
        labels = prob.param_labels
        bounds = _apply_overrides(prob.bounds, labels, bounds_overrides)
        return _BuiltProblem(
            problem=prob,
            callable=lambda x: wingtail.fg_wing_tail(x, prob),
            evaluate=lambda x: wingtail.evaluate_wing_tail(x, prob),
            bounds=bounds, is_constrained=True, dim=prob.dim,
            param_labels=labels, medium="air",
        )
    build.takes_section = not kw.get("tc_free")
    build.takes_section_aft = build.takes_section
    # ...and the FIXED BLEND, wherever this configuration has a tip device to
    # blend and does not already DESIGN the blend (the same rule the air
    # winglet family's objective-mode pass applies, read off the builder so
    # every generated twin inherits it)
    build.takes_fixed_blend = bool(
        (kw.get("winglet") or kw.get("tail_winglet"))
        and not kw.get("blended"))
    # this family GATES the aspect ratio per candidate
    # (``sizing.check_ar_limit``), so the registry may declare the
    # limit flags on it: a flag is declared where it is READ
    build.gates_ar = True
    # ...and it is the ONLY family with a lateral deck to gate the modes on
    # (``wingtail.WingTailProblem.lateral`` is the one such attribute in the
    # package). Marked on the builder rather than matched by name, because
    # these thirty-odd specs are GENERATED and a name list would reach some
    # of them and not others — the failure ``_declare_wing_objective_flags``
    # already exists to avoid.
    build.has_lateral_deck = True
    return build


#: generated problem name -> the builder configuration behind it. Public so
#: the GUI can map its choices onto the family (and back) from one table
#: instead of a hand-kept list of 30 names.
WING_TAIL_VARIANTS: dict[str, dict] = {}

for _wl, (_frag, _wlkw) in _TAIL_WINGLETS.items():
    for _tc in (False, True):
        for _arm in ("free", "fixed"):
            for _height in ("fixed", "free"):
                for _design, (_dfrag, _dkw) in _TAIL_DESIGNS.items():
                    for _cant, (_cfrag, _ckw) in _WING_CANTS.items():
                        if _wl is None and not _tc and _height == "fixed" \
                                and _design != "planform+tip" \
                                and _cant == "fixed":
                            # that IS the published lifting-line tail (which
                            # carries its own designed-tail twin). A tip
                            # device ON THE TAIL has no lifting-line twin —
                            # planar surfaces cannot hold one — so it is
                            # generated. Nor has a searched CANT: a lifting
                            # line has no out-of-plane geometry at all, so
                            # the free-cant twin of that combination is
                            # generated here too.
                            continue
                        _nm = _wing_tail_name(_wl, _tc, _arm, _height,
                                              _design, _cant)
                        WING_TAIL_VARIANTS[_nm] = {
                            "winglets": _wl, "tc": _tc, "arm": _arm,
                            "height": _height, "design": _design,
                            "cant": _cant,
                            "kwargs": dict(_wlkw, **_dkw, **_ckw,
                                           tc_free=_tc,
                                           free_height=(_height == "free"),
                                           _fixed_arm=(_arm == "fixed")),
                        }
del _wl, _frag, _wlkw, _tc, _arm, _height, _design, _dfrag, _dkw, _nm
del _cant, _cfrag, _ckw


def wing_tail_problem(winglets: str | None = None, tc: bool = False,
                      arm: str = "free", height: str = "fixed",
                      design: str = "fixed",
                      cant: str = "fixed") -> str | None:
    """Name of the nonplanar wing+tail problem for a configuration.

    ``None`` when that combination is one of the PUBLISHED lifting-line tail
    problems (no tip device, fixed thickness, tail at its layout height —
    with or without a designed tail planform, which that family now carries
    too) — the caller should keep those rather than switch solver for no
    added freedom. A SEARCHED CANT is never such a combination: a lifting
    line has no out-of-plane geometry for a dihedral to act on, so asking
    for one is asking for the panel solver. Raises KeyError for an unknown
    option.
    """
    if winglets not in _TAIL_WINGLETS:
        raise KeyError(f"unknown tip-device option {winglets!r}")
    if design not in _TAIL_DESIGNS:
        raise KeyError(f"unknown tail-design option {design!r}")
    if cant not in _WING_CANTS:
        raise KeyError(f"unknown wing-cant option {cant!r}")
    if winglets is None and not tc and height == "fixed" \
            and design != "planform+tip" and cant == "fixed":
        return None
    return _wing_tail_name(winglets, bool(tc), arm, height, design, cant)


#: ONE section table for every LABEL PROBE in this module.
#:
#: A problem built only to read ``param_labels`` is thrown away immediately
#: and never evaluated, but its default polar is read and PARSED FROM DISK
#: on construction — measured at import: 11 780 ``polar.load_xfoil_polar``
#: calls, 2.7 s of a 4.0 s profile, for label lists that do not depend on a
#: section at all. Sharing one table across the probes is safe exactly
#: because they are probes: nothing solves on them, so nothing can alias
#: another problem's polar through them. The BUILDERS still take their own.
def _probe_polars() -> dict:
    """Cheap ``polar`` / ``polar_family`` kwargs for a label-only probe."""
    global _PROBE_POLAR, _PROBE_FAMILY
    if _PROBE_POLAR is None:
        from .polar import default_polar, default_polar_family
        _PROBE_POLAR = default_polar()
        _PROBE_FAMILY = default_polar_family()
    return {"polar": _PROBE_POLAR, "polar_family": _PROBE_FAMILY}


_PROBE_POLAR = None
_PROBE_FAMILY = None


def _wing_tail_spec(name: str, variant: dict, chord_order: int = 0,
                    flight_free: bool = False) -> ProblemSpec:
    from . import wingtail
    kw = dict(variant["kwargs"])
    probe_kw = {k: v for k, v in kw.items() if k != "_fixed_arm"}
    if kw.get("_fixed_arm"):
        from . import tail as tailmod
        probe_kw["l_t_fixed"] = tailmod.default_arm()
    labels = wingtail.WingTailProblem(chord_order=chord_order,
                                      flight_free=flight_free,
                                      **_probe_polars(),
                                      **probe_kw).param_labels
    bits = []
    if variant["winglets"]:
        bits.append("a tip device (height and cant"
                    + (", and the blended root transition"
                       if variant["winglets"] == "blended" else "")
                    + ")")
    if variant["tc"]:
        bits.append("the section thickness t/c")
    if variant["height"] == "free":
        bits.append("the tail's height above the wing plane")
    if variant.get("design", "fixed") != "fixed":
        bits.append("the TAIL's own planform (taper, aspect ratio, washout"
                    + (", chord law" if chord_order else "")
                    + (" and its own tip device"
                       if variant["design"] == "planform+tip" else "")
                    + ")")
    cant_bit = _cant_description(variant.get("cant", "fixed"))
    if cant_bit:
        bits.append(cant_bit)
    if chord_order:
        bits.append("the chord distribution (free cubic law, area held)")
    if flight_free:
        bits.append("the flight state (speed and altitude)")
    extra = ("Free alongside the wing planform and the tail: "
             + ", ".join(bits) + ". ") if bits else ""
    arm = ("The distance to the tail is a chosen value (flag l_t_m). "
           if variant["arm"] == "fixed" else "")
    # the wing's own cant and sweep: declared HERE and not on the
    # lifting-line `tail` family, because only this one can score them —
    # and NOT the ROW this variant searches, where the design box owns the
    # question and a flag beside it would be the same answer given twice.
    # Per row, so a family that searches the dihedral still takes a stated
    # sweep (``stated_cant_keys``).
    flags = TAIL_CONFIG_KEYS \
        + stated_cant_keys(variant.get("cant", "fixed")) \
        + FUSELAGE_KEYS + PLANFORM_KEYS
    if variant["arm"] == "fixed":
        flags = flags + ("l_t_m",)
    if variant.get("design") == "planform+tip":
        # only a family whose SECOND surface carries a tip device can be
        # asked which way it points
        flags = flags + TAIL_TIP_KEYS
    # ...and the LAYOUT it trims in: the CG always, the height only where it
    # is not already a design variable (two answers to one question)
    flags = flags + (TAIL_CG_KEY, TAIL_MOUNT_KEY) + TAIL_LIMIT_KEYS
    if variant["height"] != "free":
        flags = flags + (TAIL_HEIGHT_KEY,)
    if variant["winglets"]:
        flags = flags + ("winglet_type", "junction_drag")
    if variant["winglets"] == "blended":
        flags = flags + ("blend_shape",)
    if chord_order:
        flags = flags + ("chord_max_frac", CHORD_LAW_KEY)
    return ProblemSpec(
        name, f"Wing + tail, nonplanar ({len(labels)}-D)", "air", True,
        labels, False, _make_wing_tail_builder(kw, chord_order, flight_free),
        description=f"{extra}{arm}{_WING_TAIL_NOTE}",
        constraint_labels=("static margin - SM_min",),
        flags=tuple(dict.fromkeys(flags)),
        uses_mission=True,
        mission_fields=("W_N", "V", "altitude_m"),
    )


for _name, _variant in WING_TAIL_VARIANTS.items():
    PROBLEM_SPECS[_name] = _wing_tail_spec(_name, _variant)
del _name, _variant


# =====================================================================
# Wing + tail + a designed CST section (live XFOIL)
# =====================================================================
# The section co-design the winglet family has, now with the tail in the same
# solve (wingtail_section.py). Generated from the wing+tail table, minus the
# variants that free the section THICKNESS: the CST weights ARE the
# thickness, so the two would be describing the same number twice.

_WT_SECTION_SUFFIX = " + CST section (XFOIL)"

_WT_SECTION_NOTE = (
    "The section is DESIGNED, not selected: eight CST weights get a live "
    "(cached) viscous XFOIL sweep each evaluation and the resulting polar "
    "drives the coupled wing + tip device + tail trim. Wing, winglet and "
    "tail all fly that one section. SLOW — seconds per new section, "
    "milliseconds once it is cached, which is why the portfolio optimiser "
    "splits the cheap planform block from the expensive section block."
)


def _wing_tail_section_problem(variant: dict, chord_order: int = 0,
                               flight_free: bool = False,
                               size_free: bool = False):
    from . import wingtail
    from . import wingtail_section as wts
    kw = dict(variant["kwargs"])
    probe = {k: v for k, v in kw.items() if k != "_fixed_arm"}
    if kw.get("_fixed_arm"):
        from . import tail as tailmod
        probe["l_t_fixed"] = tailmod.default_arm()
    wt = wingtail.WingTailProblem(chord_order=chord_order,
                                  flight_free=flight_free,
                                  size_free=size_free, **probe)
    return wts.WingTailSectionProblem(wing_tail=wt)


def _make_wing_tail_section_builder(kw: dict, chord_order: int = 0,
                                    flight_free: bool = False,
                                    size_free: bool = False):
    def build(mission_kwargs, flags, bounds_overrides):
        from . import wingtail
        from . import wingtail_section as wts
        cfg = dict(kw)
        cfg.update(_flight_kwargs(flight_free))
        cfg.update(_size_kwargs(size_free, flags))
        # the sized wing+tail families declare the loading flags like every
        # other sized family; until this line they DECLARED them and dropped
        # them, so a stated W/S changed nothing on any nonplanar variant
        cfg.update(_wing_loading_kwargs(size_free, flags, 10.0))
        cfg.update(_size_band_kwargs(size_free, bounds_overrides, flags))
        cfg.update(_tail_config(flags))
        cfg.update(_wing_cant_kwargs(flags))
        cfg.update(_fuselage_kwargs(flags))
        if cfg.pop("_fixed_arm", False):
            arm = (flags or {}).get("l_t_m")
            from . import tail as tailmod
            cfg["l_t_fixed"] = float(
                tailmod.default_arm(_wing_span_of(flags))
                if arm is None else arm)
        else:
            cfg.update(_arm_band_kwargs(bounds_overrides))
        # ...and the VERTICAL band, only where the height is the thing being
        # searched: on a stated-height twin a stale z_t_m row would collide
        # with z_t_fixed rather than be ignored
        if cfg.get("free_height"):
            cfg.update(_height_band_kwargs(bounds_overrides))
        cfg.update(_chord_kwargs(chord_order, flags))
        cfg.update(_planform_kwargs(flags))
        cfg.update(_blend_shape_flag(flags))
        cfg.update(_blend_frac_kwargs(flags))
        cfg.update(_junction_flag(
            flags, _blend_default_on(flags, bool(kw.get("blended")))))
        cfg.update(_winglet_chord_kwargs(flags))
        cfg["winglet_cant_bounds"] = wingtail.winglet_cant_bounds_for(
            (flags or {}).get("winglet_type"), bool(kw.get("capped")))
        # ...and each device's HEIGHT band, gated on the device
        # the family actually carries — the Problem refuses a
        # band for a surface with no device.
        cfg.update(_winglet_h_band_kwargs(
            bounds_overrides,
            (("winglet_h_frac",) if cfg.get("winglet") else ())
            + (("winglet_h_frac_t",)
               if cfg.get("tail_winglet") else ())))
        if cfg.get("tail_winglet"):
            cfg.update(_tail_cant_kwargs(flags))
            cfg.update(_tail_blend_kwargs(flags))
        cfg.update(_tail_layout_kwargs(flags))
        # the WING's section is designed here; the TAIL may still fly one the
        # user chose (wingtail replaces ``polar`` per candidate and leaves
        # ``polar_tail`` alone), which is the only way to give the two
        # surfaces different aerofoils in this family
        cfg.update(_section_kwargs(flags, field="polar_tail",
                                   key=SECTION_AFT_KEY))
        wt = _apply_mission(wingtail.WingTailProblem(**cfg), mission_kwargs)
        prob = wts.WingTailSectionProblem(
            section=airfoil_problem_from_flags(flags), wing_tail=wt)
        labels = prob.param_labels
        bounds = _apply_overrides(prob.bounds, labels, bounds_overrides)
        return _BuiltProblem(
            problem=prob,
            callable=lambda x: wts.fg_wing_tail_section(x, prob),
            evaluate=lambda x: wts.evaluate_wing_tail_section(x, prob),
            bounds=bounds, is_constrained=True, dim=prob.dim,
            param_labels=labels, medium="air",
        )
    build.takes_section_aft = True      # the tail's own section
    build.takes_fixed_blend = bool(
        (kw.get("winglet") or kw.get("tail_winglet"))
        and not kw.get("blended"))
    build.designs_section = True
    # this family GATES the aspect ratio per candidate
    # (``sizing.check_ar_limit``), so the registry may declare the
    # limit flags on it: a flag is declared where it is READ
    build.gates_ar = True
    return build


def airfoil_problem_from_flags(flags: dict | None):
    """AirfoilProblem carrying any airfoil-namespaced flags (empty -> default)."""
    from . import airfoil
    return airfoil.AirfoilProblem(**_airfoil_kwargs(flags))


def wing_tail_section_problem(winglets: str | None = None,
                              arm: str = "free",
                              height: str = "fixed",
                              design: str = "fixed",
                              cant: str = "fixed") -> str:
    """Name of the wing+tail+CST-section problem for a configuration.

    Unlike :func:`wing_tail_problem` this never returns None: with the
    section DESIGNED there is no published lifting-line problem to fall back
    to, so every combination is its own registered problem — including the
    free-cant twin, for the same reason: a lifting line is what cannot carry
    a searched dihedral, and this family is lattice-backed throughout.
    """
    if cant not in _WING_CANTS:
        raise KeyError(f"unknown wing-cant option {cant!r}")
    return (_wing_tail_name(winglets, False, arm, height, design, cant)
            + _WT_SECTION_SUFFIX)


#: generated name -> the wing+tail configuration it designs a section for.
#: Built from the SAME table as the wing+tail family but over the full
#: product: the one combination that family leaves out (no tip device, fixed
#: thickness, tail at its layout height) is excluded there because the
#: published lifting-line problem covers it — with the section DESIGNED there
#: is no lifting-line problem to keep, so it is generated here.
WING_TAIL_SECTION_VARIANTS: dict[str, dict] = {}

for _wl, (_frag, _wlkw) in _TAIL_WINGLETS.items():
    for _arm in ("free", "fixed"):
        for _height in ("fixed", "free"):
            for _design, (_dfrag, _dkw) in _TAIL_DESIGNS.items():
                # ...and the WING CANT, stated or searched, on every one of
                # them. It was left off when the axis landed on the plain
                # wing+tail family, so a user who asked for a designed
                # section AND a searched dihedral got the section family
                # with the cant merely STATED. This family is lattice-backed
                # everywhere, so it can score the row it is given.
                for _cant, (_cfrag, _ckw) in _WING_CANTS.items():
                    _nm = wing_tail_section_problem(_wl, _arm, _height,
                                                    _design, _cant)
                    WING_TAIL_SECTION_VARIANTS[_nm] = {
                        "winglets": _wl, "tc": False, "arm": _arm,
                        "height": _height, "design": _design, "cant": _cant,
                        "kwargs": dict(_wlkw, **_dkw, **_ckw, tc_free=False,
                                       free_height=(_height == "free"),
                                       _fixed_arm=(_arm == "fixed")),
                    }
del _wl, _frag, _wlkw, _arm, _height, _nm, _design, _dfrag, _dkw
del _cant, _cfrag, _ckw


def _wing_tail_section_labels(variant: dict, chord_order: int = 0,
                              flight_free: bool = False,
                              size_free: bool = False) -> tuple:
    return _wing_tail_section_problem(variant, chord_order, flight_free,
                                      size_free).param_labels


def _wing_tail_section_spec(name: str, variant: dict, chord_order: int = 0,
                            flight_free: bool = False,
                            size_free: bool = False) -> ProblemSpec:
    prob = _wing_tail_section_problem(variant, chord_order, flight_free,
                                      size_free)
    labels = prob.param_labels
    # the wing's own cant and sweep: declared HERE and not on the
    # lifting-line `tail` family, because only this one can score them —
    # and not whichever of them this variant carries as a design ROW.
    # Stating a flag the vector already carries is the same question
    # answered twice, and ``WingTailProblem`` raises on it.
    flags = TAIL_CONFIG_KEYS \
        + stated_cant_keys(variant.get("cant", "fixed")) \
        + FUSELAGE_KEYS + PLANFORM_KEYS
    if variant["arm"] == "fixed":
        flags = flags + ("l_t_m",)
    if variant.get("design") == "planform+tip":
        # only a family whose SECOND surface carries a tip device can be
        # asked which way it points
        flags = flags + TAIL_TIP_KEYS
    # ...and the LAYOUT it trims in: the CG always, the height only where it
    # is not already a design variable (two answers to one question)
    flags = flags + (TAIL_CG_KEY, TAIL_MOUNT_KEY) + TAIL_LIMIT_KEYS
    if variant["height"] != "free":
        flags = flags + (TAIL_HEIGHT_KEY,)
    if variant["winglets"]:
        flags = flags + ("winglet_type", "junction_drag")
    if variant["winglets"] == "blended":
        flags = flags + ("blend_shape",)
    labels_g = ("static margin - SM_min", "t/c margin", "|Cm| margin")
    return ProblemSpec(
        name, f"Wing + tail + CST section, live XFOIL ({len(labels)}-D)",
        "air", True, labels, True,
        _make_wing_tail_section_builder(variant["kwargs"], chord_order,
                                        flight_free, size_free),
        n_constraints=3,
        description=f"{_WT_SECTION_NOTE} {_WING_TAIL_NOTE}",
        constraint_labels=labels_g,
        flags=tuple(dict.fromkeys(flags)),
        uses_mission=True,
        mission_fields=("W_N", "V", "altitude_m"),
        has_blocks=True,
    )


for _name, _variant in WING_TAIL_SECTION_VARIANTS.items():
    PROBLEM_SPECS[_name] = _wing_tail_section_spec(_name, _variant)
del _name, _variant


# =====================================================================
# Hydrofoil + elevator (the foiling craft)
# =====================================================================
# The water family's counterpart to the nonplanar wing+tail problems: main
# foil + optional tip device + STABILISER in one imaged VLM solve
# (hydrotail.py), trimmed in lift AND pitch, cavitation checked at every
# panel's own submergence. Each combination changes the design DIMENSION, so
# each is its own ProblemSpec — generated from one table, exactly as the
# wing+tail family is.
#
# No flight-state modifier here, and it is not an oversight: speed and depth
# have been design variables in this family since Tier C.

_HYDRO_TAIL_NOTE = (
    "Solved by the IMAGED wing+tail VLM (hydrotail.py): one influence matrix "
    "over the main foil, its tip device, the stabiliser and their free-"
    "surface images. The craft is trimmed in lift AND pitch, the static "
    "margin comes off the same basis solutions, and the cavitation margin is "
    "the worst panel of EITHER surface at its own submergence — the "
    "stabiliser sits deeper, so it carries more static head than the foil it "
    "trims."
)


#: how much of the ELEVATOR is designed — the same three answers the air
#: tail's :data:`TAIL_DESIGNS` gives, and the same builder kwargs, because it
#: is the same question about the same kind of surface. Only the NAME
#: fragment differs: under water the surface is called the elevator.
_HYDRO_TAIL_DESIGNS: dict[str, str] = {
    "fixed": "",
    "planform": "designed elevator",
    "planform+tip": "designed elevator + tip device",
}


def _hydro_tail_name(winglet: bool, arm: str, height: str,
                     design: str = "fixed") -> str:
    name = "hydrofoil + elevator" + (" + winglet" if winglet else "")
    qual = ([] if arm == "free" else ["fixed arm"]) + \
        ([] if height == "fixed" else ["free depth"]) + \
        ([] if design == "fixed" else [_HYDRO_TAIL_DESIGNS[design]])
    return f"{name} [{', '.join(qual)}]" if qual else name


def _make_hydro_tail_builder(kw: dict, chord_order: int = 0,
                             chosen_section: bool = False):
    def build(mission_kwargs, flags, bounds_overrides):
        from . import hydrotail
        if mission_kwargs:
            # the water families carry their own operating point IN the
            # design vector (speed and depth), so a mission card here would
            # be two answers to one question
            raise ValueError(
                "the hydrofoil + elevator problem has no mission spec — its "
                "speed and depth are design variables (edit their box "
                "bounds instead)")
        cfg = dict(kw)
        if cfg.pop("_fixed_arm", False):
            arm = (flags or {}).get("l_t_m")
            cfg["l_t_fixed"] = float(
                0.5 * (hydrotail.L_T_BOUNDS[0] + hydrotail.L_T_BOUNDS[1])
                if arm is None else arm)
        else:
            cfg.update(_arm_band_kwargs(bounds_overrides))
        cfg.update(_area_band_kwargs(bounds_overrides))
        if cfg.get("free_height"):        # see _make_wing_tail_builder
            cfg.update(_height_band_kwargs(bounds_overrides))
        cfg.update(_chord_kwargs(chord_order, flags))
        cfg.update(_planform_kwargs(flags, *_WET_SIZE))
        cfg.update(_water_kwargs(flags))
        # ...and the LOAD that size has to carry. Beside the size on purpose:
        # a span and an area with the wrong weight behind them is a stated
        # planform flown at somebody else's wing loading (WEIGHT_KEY).
        cfg.update(_weight_kwargs(flags))
        # ...and the OPERATING POINT the design vector searches over. A row
        # here is a bound, not a flag, for _arm_band_kwargs's reason: depth
        # and speed ARE design variables, so where a shell asks for a band
        # is that variable's design-box row.
        cfg.update(_water_band_kwargs(bounds_overrides))
        # ...and the FLIGHT CONDITION that operating point is flown at. The
        # heel is what makes this family's searched span a question with an
        # answer, so it travels beside the size rows and not somewhere else.
        cfg.update(_water_flight_kwargs(flags))
        cfg.update(_water_re_kwargs(flags))
        # ...and the MAIN FOIL's own cant and sweep. Lattice geometry, and
        # this family is imaged-lattice throughout, so it can score them.
        cfg.update(_wing_cant_kwargs(flags))
        cfg.update(_wet_size_kwargs(flags, bounds_overrides))
        # the fixed BLEND, in this family's own field names: it carries the
        # air twin's ``blend_frac_fixed`` (not hydrofoil.py's ``blend_frac``),
        # so the shape flag pass and the fraction are asked separately
        cfg.update(_blend_shape_flag(flags))
        cfg.update(_blend_frac_kwargs(flags))
        cfg.update(_junction_flag(flags, _blend_default_on(flags, False)))
        cfg.update(_winglet_chord_kwargs(flags))
        cfg.update(_wet_section_kwargs(flags, required=chosen_section))
        cfg.update(_wet_aft_section_kwargs(flags))
        if cfg.get("tail_winglet"):
            cfg.update(_tail_cant_kwargs(flags))
            cfg.update(_tail_blend_kwargs(flags))
        if cfg.get("winglet"):
            # ...and the MAIN FOIL's own device, asked the same way the
            # stabiliser's is (_foil_cant_kwargs)
            cfg.update(_foil_cant_kwargs(flags))
        cfg.update(_tail_layout_kwargs(flags))
        # ...and the RIG that drives all of it. Last of the load statements
        # on purpose: it is the only one that needs the weight to already be
        # settled (the drive force is closed from it), and the problem's own
        # constructor is what checks the two against each other.
        cfg.update(_rig_kwargs(flags))
        cfg.update(_draught_kwargs(flags))
        cfg.update(_strut_config(flags))
        # ...and WHERE the strut stands, which only this family can answer
        cfg.update(_strut_station_config(flags))
        prob = hydrotail.HydrofoilTailProblem(**cfg)
        labels = prob.param_labels
        bounds = _apply_overrides(prob.bounds, labels, bounds_overrides)
        return _BuiltProblem(
            problem=prob,
            callable=lambda x: hydrotail.fg_hydrofoil_tail(x, prob),
            evaluate=lambda x: hydrotail.evaluate_hydrofoil_tail(x, prob),
            bounds=bounds, is_constrained=True, dim=prob.dim,
            param_labels=labels, medium="water",
        )
    # the MAIN foil selects its section by thickness (only the chosen-section
    # twin flies one), but the ELEVATOR flies a table either way — so a
    # section for it replaces that table without touching the design vector
    build.takes_section = bool(chosen_section)
    build.takes_section_aft = True
    build.takes_fixed_blend = bool(kw.get("winglet") or kw.get("tail_winglet"))
    return build


#: generated problem name -> the builder configuration behind it (public, so
#: the GUI maps its choices onto the family from one table).
HYDRO_TAIL_VARIANTS: dict[str, dict] = {}

for _wl in (False, True):
    for _arm in ("free", "fixed"):
        for _height in ("fixed", "free"):
            for _design in _HYDRO_TAIL_DESIGNS:
                _nm = _hydro_tail_name(_wl, _arm, _height, _design)
                HYDRO_TAIL_VARIANTS[_nm] = {
                    "winglets": _wl, "arm": _arm, "height": _height,
                    "design": _design,
                    "kwargs": dict(
                        _TAIL_DESIGNS[_design][1],
                        winglet=_wl, free_height=(_height == "free"),
                        _fixed_arm=(_arm == "fixed")),
                }
del _wl, _arm, _height, _design, _nm


def hydro_tail_problem(winglet: bool = False, arm: str = "free",
                       height: str = "fixed", design: str = "fixed") -> str:
    """Name of the hydrofoil+elevator problem for a configuration."""
    if design not in _HYDRO_TAIL_DESIGNS:
        raise KeyError(f"unknown elevator-design option {design!r}")
    return _hydro_tail_name(bool(winglet), arm, height, design)


def _hydro_tail_labels(variant: dict, chord_order: int = 0) -> tuple:
    from . import hydrotail
    kw = dict(variant["kwargs"])
    probe = {k: v for k, v in kw.items() if k != "_fixed_arm"}
    if kw.get("_fixed_arm"):
        probe["l_t_fixed"] = 0.5 * (hydrotail.L_T_BOUNDS[0]
                                    + hydrotail.L_T_BOUNDS[1])
    return hydrotail.HydrofoilTailProblem(chord_order=chord_order,
                                          **probe).param_labels


def _hydro_tail_spec(name: str, variant: dict,
                     chord_order: int = 0) -> ProblemSpec:
    labels = _hydro_tail_labels(variant, chord_order)
    bits = []
    if variant["winglets"]:
        bits.append("a tip device on the main foil (height and SIGNED cant)")
    if variant["height"] == "free":
        bits.append("the stabiliser's depth below the foil")
    if variant.get("design", "fixed") != "fixed":
        bits.append("the ELEVATOR's own planform (taper, aspect ratio, "
                    "washout"
                    + (", chord law" if chord_order else "")
                    + (" and its own tip device"
                       if variant["design"] == "planform+tip" else "")
                    + ")")
    if chord_order:
        bits.append("the chord distribution (free cubic law, area held)")
    extra = ("Free alongside the foil planform, the speed, the depth and the "
             "stabiliser: " + ", ".join(bits) + ". ") if bits else ""
    arm = ("The distance to the stabiliser is a chosen value (flag l_t_m). "
           if variant["arm"] == "fixed" else "")
    flags = ("l_t_m",) if variant["arm"] == "fixed" else ()
    if chord_order:
        flags = flags + ("chord_max_frac", CHORD_LAW_KEY)
    if variant.get("design") == "planform+tip":
        flags = flags + TAIL_TIP_KEYS
    flags = flags + (TAIL_CG_KEY, TAIL_MOUNT_KEY) + TAIL_LIMIT_KEYS
    if variant["height"] != "free":
        flags = flags + (TAIL_HEIGHT_KEY,)
    flags = (flags + (WATER_KEY, WEIGHT_KEY, DRAUGHT_MAX_KEY)
             + RIG_KEYS + STRUT_KEYS + STRUT_STATION_KEYS
             + WATER_FLIGHT_KEYS + (WATER_RE_KEY,)
             # the MAIN FOIL's own cant and sweep: this family is an imaged
             # LATTICE throughout, so it can score them (hydrotail builds
             # the foil with them; the stabiliser keeps its own plane)
             + WING_CANT_KEYS
             # THE SIZE. Both rows on every variant — the span row's one
             # remaining refusal is the strut at zero heel, and that is a
             # stage-1 answer rather than a property of the family
             # (:func:`wet_size_keys`, ``hydrofoil.SPAN_NEEDS_NO_FIN``).
             + wet_size_keys(variant["height"] == "free"))
    if variant["winglets"]:
        # the MAIN FOIL's tip-device SHAPE, declared where the stabiliser's
        # already is (TAIL_TIP_KEYS above) — see WATER_TIP_DIRECTION
        flags = flags + ("winglet_type",)
    return ProblemSpec(
        name, f"Hydrofoil + elevator ({len(labels)}-D)", "water", True,
        labels, False, _make_hydro_tail_builder(variant["kwargs"],
                                                chord_order),
        n_constraints=2,
        description=f"{extra}{arm}{_HYDRO_TAIL_NOTE}",
        constraint_labels=("cavitation margin", "static margin - SM_min"),
        flags=flags,
    )


for _name, _variant in HYDRO_TAIL_VARIANTS.items():
    PROBLEM_SPECS[_name] = _hydro_tail_spec(_name, _variant)
del _name, _variant


# =====================================================================
# Water families flying a CHOSEN section
# =====================================================================
# A water family SELECTS its section by thickness off the NACA 24XX polar
# family, so handing it the section the user picked replaces that lookup —
# and with it the ``tc`` design variable, since the chosen shape already has
# a thickness. That changes the DIMENSION, which in this package makes it a
# problem of its own rather than a flag on an existing one (the same rule
# that gave "tail (fixed arm)" its own entry).
#
# The section travels as api.SECTION_KEY, exactly as in air; what is extra
# here is that its XFOIL sweep is run WITH Cp_min, because the cavitation
# constraint needs the minimum surface pressure at every angle. One cached
# sweep per (section, point) — paid the first time, free thereafter.

CHOSEN_SECTION_SUFFIX = " [chosen section]"

_CHOSEN_SECTION_NOTE = (
    "Flies the SECTION THE USER CHOSE (flag section_name: a library name, or "
    "the CST weights of a designed one) instead of selecting one by "
    "thickness off the NACA 24XX family. Its polar — and its Cp_min table, "
    "which the cavitation constraint needs — come from one cached XFOIL "
    "sweep of that shape, so t/c leaves the design vector: the section "
    "already has one."
)

#: plain name -> the twin of it that flies a chosen section
CHOSEN_SECTION_TWINS: dict[str, str] = {}


def needs_chosen_section(name: str) -> bool:
    """Does this problem REQUIRE a chosen section to be flown at all?

    True for the water twins: their polar IS the chosen section, so a run
    without one has no lift slope, no drag and no Cp_min. Everything else
    takes a chosen section as an optional flag (or not at all).
    """
    return base_of(name) in set(CHOSEN_SECTION_TWINS.values())


def chosen_section_problem(name: str) -> str | None:
    """The twin of ``name`` that FLIES a chosen section, or None.

    None where the family needs no twin — an air family whose section is a
    single table takes the section as a plain flag, because nothing about
    its design vector changes.
    """
    return CHOSEN_SECTION_TWINS.get(base_of(name))


def _chosen_section_spec(base: str, build) -> ProblemSpec:
    sp = PROBLEM_SPECS[base]
    labels = tuple(l for l in sp.param_labels if l != "tc")
    return ProblemSpec(
        base + CHOSEN_SECTION_SUFFIX,
        re.sub(r"\d+-D", f"{len(labels)}-D", sp.display, count=1)
        + " + chosen section",
        sp.medium, sp.is_constrained, labels, sp.slow, build,
        n_constraints=sp.n_constraints,
        description=sp.description + " " + _CHOSEN_SECTION_NOTE,
        constraint_labels=sp.constraint_labels,
        flags=tuple(dict.fromkeys(sp.flags + (SECTION_KEY,))),
        uses_mission=sp.uses_mission, mission_fields=sp.mission_fields,
        has_blocks=sp.has_blocks,
    )


for _base, _factory in (
        ("hydrofoil",
         lambda co: _make_hydrofoil_builder(co, chosen_section=True)),
        ("hydrofoil + winglet",
         lambda co: _make_hydrofoil_winglet_builder(co,
                                                    chosen_section=True))):
    _nm = _base + CHOSEN_SECTION_SUFFIX
    PROBLEM_SPECS[_nm] = _chosen_section_spec(_base, _factory(0))
    CHOSEN_SECTION_TWINS[_base] = _nm
del _base, _factory, _nm

for _name, _variant in HYDRO_TAIL_VARIANTS.items():
    _nm = _name + CHOSEN_SECTION_SUFFIX
    PROBLEM_SPECS[_nm] = _chosen_section_spec(
        _name, _make_hydro_tail_builder(_variant["kwargs"], 0,
                                        chosen_section=True))
    CHOSEN_SECTION_TWINS[_name] = _nm
del _name, _variant, _nm


#: the chosen-section twins take the chord modifier (and only that: speed and
#: depth are already design variables there, and the SIZE is configuration —
#: a chosen span and area, declared with the base family's below). So the
#: factories are registered beside the plain families'.
_CHOSEN_SECTION_FACTORIES = {
    "hydrofoil": lambda co: _make_hydrofoil_builder(co, chosen_section=True),
    "hydrofoil + winglet":
        lambda co: _make_hydrofoil_winglet_builder(co, chosen_section=True),
}
for _name, _variant in HYDRO_TAIL_VARIANTS.items():
    _CHOSEN_SECTION_FACTORIES[_name] = (
        lambda kw: (lambda co: _make_hydro_tail_builder(kw, co,
                                                        chosen_section=True))
    )(_variant["kwargs"])
del _name, _variant


# =====================================================================
# Tandem pair in the nonplanar solver (tip devices, and a designed section)
# =====================================================================
# tandem.py couples two PLANAR lifting lines and so can carry no tip device at
# all; tandemvlm.py puts the pair into the VLM (vlm.SecondWing), which is what
# makes winglets, an imaged section and every modifier available to it. The
# twist law differs (root/tip, the Wing law the chord law rides on, rather
# than the planar family's 3 knots), so the two are different design spaces
# and the published tandem numbers stay with the published problem.

_TANDEM_VLM_LABELS = (
    "taper_front", "taper_rear", "twist_root_front_deg",
    "twist_tip_front_deg", "twist_root_rear_deg", "twist_tip_rear_deg",
    "area_split_front", "decalage_deg")

_TANDEM_VLM_NOTE = (
    "Solved by the NONPLANAR pair (tandemvlm.py): front wing, rear wing and "
    "either tip device in ONE influence matrix, with the induced drag from a "
    "single Trefftz-plane sum over the combined wake. A different aero core "
    "from the published lifting-line tandem — and a different design space "
    "(root/tip twist rather than 3 knots per wing) — so compare within one "
    "of them, not across."
)


def _make_tandem_vlm_builder(winglets: bool, chord_order: int = 0,
                             flight_free: bool = False,
                             size_free: bool = False, section: bool = False,
                             cant: str = "fixed"):
    # the CST variant DESIGNS its section, so there is no table for a chosen
    # one to replace; the plain pair flies one and can take it
    def build(mission_kwargs, flags, bounds_overrides):
        from . import tandemvlm as tvm
        # the cant STATE, splatted from the one table — this was a boolean,
        # and a boolean cannot say "the dihedral and not the sweep"
        kw = dict(winglets=winglets, **_WING_CANTS[cant][1])
        kw.update(_chord_kwargs(chord_order, flags))
        kw.update(_flight_kwargs(flight_free))
        kw.update(_size_kwargs(size_free, flags))
        size_band = _size_band_kwargs(size_free, bounds_overrides, flags)
        planform = _tandem_planform_kwargs(
            flags, size_free=size_free,
            span_bounds_m=size_band.get("span_bounds_m"))
        kw.update(_wing_loading_kwargs(size_free, flags,
                                       planform.get("b", 10.0)))
        kw.update(size_band)
        kw.update(planform)
        kw.update(_blend_shape_flag(flags))
        kw.update(_blend_frac_kwargs(flags))
        kw.update(_junction_flag(flags, _blend_default_on(flags, False)))
        kw.update(_winglet_chord_kwargs(flags))
        # WHAT SHAPE THE PAIR'S TIP DEVICE IS, as the same cant band every
        # other lattice family narrows from the same flag (api.py's wing+tail
        # branch is the twin). Without it "vertical fence (90 deg)" was the
        # one tip-device shape no tandem could be asked for: the flag was
        # undeclared, ``sanitise_flags`` dropped it, and the pair searched
        # its full 60-90 band under a label that said 90 — so the V3 shape
        # menu refused the entry outright (nice_app.winglet_shape_pair).
        if winglets:
            from . import wingtail
            kw["winglet_cant_bounds"] = wingtail.winglet_cant_bounds_for(
                (flags or {}).get("winglet_type"), capped=False)
        kw.update(_wing_cant_kwargs(flags))
        kw.update(_fin_config(flags))
        kw.update(_handling_kwargs(flags))
        if not section:
            kw.update(_section_kwargs(flags))
        # the REAR wing's own section, either way: where the front's is
        # DESIGNED (the CST variants) the rear may still fly a chosen table,
        # which is the pair's honest case — the two wings carry different
        # shares of the weight at different chords
        kw.update(_section_kwargs(flags, field="polar_rear",
                                  key=SECTION_AFT_KEY))
        prob = _apply_mission(tvm.TandemVLMProblem(**kw), mission_kwargs)
        if section:
            from . import wingtail_section as wts
            prob = wts.WingTailSectionProblem(
                section=airfoil_problem_from_flags(flags), wing_tail=prob)
            labels = prob.param_labels
            bounds = _apply_overrides(prob.bounds, labels, bounds_overrides)
            return _BuiltProblem(
                problem=prob,
                callable=lambda x: wts.fg_wing_tail_section(x, prob),
                evaluate=lambda x: wts.evaluate_wing_tail_section(x, prob),
                bounds=bounds, is_constrained=True, dim=prob.dim,
                param_labels=labels, medium="air",
            )
        labels = prob.param_labels
        bounds = _apply_overrides(prob.bounds, labels, bounds_overrides)
        # A PAIR IS CONSTRAINED IF IT DECLARES A MARGIN — asked of the
        # problem's own ``constraint_labels`` rather than of ``size_free``,
        # because the handling gate is the second thing that can put rows on
        # this family and a boolean naming only the first would send an
        # unsized, gated pair through the UNconstrained callable, where the
        # five margins it just measured would be silently dropped.
        _n_g = int(getattr(prob, "n_constraints", 0) or 0)
        return _BuiltProblem(
            problem=prob,
            callable=((lambda x: tvm.fg_tandem_vlm(x, prob)) if _n_g
                      else (lambda x: tvm.objective_tandem_vlm(x, prob))),
            evaluate=lambda x: tvm.evaluate_tandem_vlm(x, prob),
            bounds=bounds, is_constrained=bool(_n_g), dim=prob.dim,
            param_labels=labels, medium="air",
        )
    build.takes_section = not section
    build.takes_section_aft = True      # the pair's REAR wing, always
    build.takes_fixed_blend = bool(winglets)
    build.designs_section = bool(section)
    # this family GATES the aspect ratio per candidate
    # (``sizing.check_ar_limit``), so the registry may declare the
    # limit flags on it: a flag is declared where it is READ
    build.gates_ar = True
    # ...AND IT FLIES A VERTICAL SURFACE, so the modes can be gated on it
    # (:func:`_declare_handling_flags`). The wing+tail branch was the only
    # family marked with this, and the sentence that said so — "the only
    # family with a lateral deck" — stopped being true the moment the pair
    # got one: ``tandemvlm.TandemVLMProblem.lateral`` puts the fin in the
    # SCORED lattice exactly as ``wingtail``'s does, which is the whole test
    # (a family that only CHARGES a fin has no Cn_beta to gate). Marked on
    # the builder rather than matched by name, because these hundred-odd
    # specs are generated.
    build.has_lateral_deck = True
    return build


def _tandem_vlm_labels(winglets: bool, chord_order: int = 0,
                       flight_free: bool = False, size_free: bool = False,
                       section: bool = False,
                       cant: str = "fixed") -> tuple:
    from . import tandemvlm as tvm
    prob = tvm.TandemVLMProblem(winglets=winglets, chord_order=chord_order,
                                flight_free=flight_free, size_free=size_free,
                                **_WING_CANTS[cant][1])
    if not section:
        return prob.param_labels
    from . import wingtail_section as wts
    return wts.WingTailSectionProblem(wing_tail=prob).param_labels


#: generated name -> (tip devices?, designed section?)
#: NOTE the plain (no tip device, no designed section) nonplanar pair is
#: deliberately NOT registered: it would be the published lifting-line
#: problem's question answered by a second solver, reachable from the builder
#: only by an extra control, and the pair's own cross-solver comparison lives
#: in tests/test_tandem_vlm.py where it belongs. The nonplanar family is
#: selected by asking for what the planar one CANNOT do — a tip device, or a
#: designed section.
#: ...AND THE PAIR'S OWN CANT, stated or SEARCHED, on every one of them.
#: The searched side is two design ROWS and therefore a different dimension,
#: which is why it is an axis here and not a flag (:data:`_WING_CANTS` says
#: why, in the family this axis was built for first). It belongs on THIS
#: family and not on the lifting-line ``tandem``: each surface's
#: quarter-chord line there is straight along y, so a dihedral would have
#: nothing to act on — and the free-cant pair with NO tip device and NO
#: designed section IS registered, because that combination is exactly what
#: a lifting line cannot answer.
def _tandem_vlm_name(winglets: bool, section: bool, cant: str) -> str:
    """The registered name of one nonplanar-pair configuration.

    The bracketed-qualifier convention :func:`_wing_tail_name` uses, and the
    rule that protects every pre-existing name: ``fixed`` contributes no
    bracket at all, so the three names this family shipped under regenerate
    byte-identically. The bracket sits on the family CORE, before the
    section suffix, exactly as ``tail + winglet [free cant] + CST section
    (XFOIL)`` does.
    """
    if cant not in _WING_CANTS:
        raise KeyError(f"unknown wing-cant option {cant!r}")
    out = "tandem (nonplanar)" + (" + winglets" if winglets else "")
    if cant != "fixed":
        out += f" [{_WING_CANTS[cant][0]}]"
    return out + (_WT_SECTION_SUFFIX if section else "")


def tandem_vlm_problem(winglets: bool = False, section: bool = False,
                       cant: str = "fixed") -> str | None:
    """Name of the nonplanar-pair problem for a configuration, or ``None``.

    ``None`` only for the one combination the published lifting-line pair
    already answers — no tip device, no designed section, a stated cant. A
    shell asking for anything else gets a registered name rather than a
    table it has to keep in step with this one.
    """
    if not (winglets or section or cant != "fixed"):
        return None
    return _tandem_vlm_name(winglets, section, cant)


TANDEM_VLM_VARIANTS: dict[str, dict] = {
    _tandem_vlm_name(_wl, _sec, _ct): {"winglets": _wl, "section": _sec,
                                       "cant": _ct}
    for _wl in (False, True)
    for _sec in (False, True)
    for _ct in _WING_CANTS
    if (_wl or _sec or _ct != "fixed")
}


def _tandem_vlm_spec(name: str, variant: dict, chord_order: int = 0,
                     flight_free: bool = False,
                     size_free: bool = False) -> ProblemSpec:
    wl, section = variant["winglets"], variant["section"]
    cant = variant.get("cant", "fixed")
    labels = _tandem_vlm_labels(wl, chord_order, flight_free, size_free,
                                section, cant)
    bits = []
    if wl:
        bits.append("a tip device on EACH wing (height and cant)")
    if section:
        bits.append("the aerofoil itself (8 CST weights, live XFOIL)")
    cant_rows = searched_cant_keys(cant)
    if cant_rows:
        bits.append(
            _CANT_DESCRIPTIONS[cant_rows].replace("the WING's", "the pair's")
            + (", which is the only wing-side source of Cl_beta a tandem "
               "has that holds at any lift (weight `spiral` or the row has "
               "nothing to buy)"
               if WING_CANT_KEYS[0] in cant_rows else
               ", which moves the neutral point and buys no roll: its own "
               "dihedral effect goes as CL and is gone at cruise"))
    extra = ("Free alongside both planforms, the area split and the "
             "decalage: " + ", ".join(bits) + ". ") if bits else ""
    labels_g = ("t/c margin", "|Cm| margin") if section else ()
    return ProblemSpec(
        name,
        (f"Tandem, nonplanar ({len(labels)}-D)" if not section
         else f"Tandem + CST section, live XFOIL ({len(labels)}-D)"),
        "air", bool(section), labels, bool(section),
        _make_tandem_vlm_builder(wl, chord_order, flight_free, size_free,
                                 section, cant),
        n_constraints=max(1, len(labels_g)),
        description=f"{extra}{_TANDEM_VLM_NOTE}"
        + (f" {_WT_SECTION_NOTE}" if section else ""),
        constraint_labels=labels_g,
        # the pair's cant and sweep: declared HERE and not on the
        # lifting-line `tandem`, for the same reason the wing+tail
        # pair declares them only on its nonplanar twin
        # the pair's cant and sweep: declared HERE and not on the
        # lifting-line `tandem`, and NOT on the free-cant twin, where they
        # are design ROWS. Stating a flag the vector already carries is the
        # same question answered twice, and TandemVLMProblem raises on it.
        flags=(PLANFORM_KEYS + TANDEM_STAGGER_KEYS + TANDEM_SPAN_KEYS
               + stated_cant_keys(cant)
               + FIN_SHAPE_KEYS + (FIN_PRESENCE_KEY,
                                   TANDEM_FIN_BOOM_KEY)
               # ...and WHAT SHAPE the pair's tip device is, declared where
               # every other lattice family declares it (the wing+tail
               # branch above). Undeclared, the flag was dropped on the way
               # to the run and the V3 menu refused the "vertical fence"
               # entry rather than offer a 90 deg label over a 60-90 search
               # — the only tip-device shape a pair could not be given.
               + (("winglet_type",) if wl else ())),
        uses_mission=True,
        mission_fields=("W_N", "V", "altitude_m"),
        has_blocks=bool(section),
    )


for _name, _variant in TANDEM_VLM_VARIANTS.items():
    PROBLEM_SPECS[_name] = _tandem_vlm_spec(_name, _variant)
del _name, _variant


# =====================================================================
# Water families + a designed CST section (live XFOIL, with Cp_min)
# =====================================================================
# Shaping a section for a FOIL needs the minimum surface pressure as well as
# cl/cd/cm — the cavitation constraint reads sigma_cav + Cp_min — which is
# why xfoil_run grew a with_cpmin sweep and why the CST section LIBRARY (no
# Cp_min table) could never serve here. hydrofoil_section.py composes the two;
# the t/c row leaves the design vector because the CST weights ARE the
# thickness.

_HF_SECTION_SUFFIX = " + CST section (XFOIL)"

_HF_SECTION_NOTE = (
    "The section is DESIGNED, not selected: eight CST weights get a live "
    "(cached) viscous XFOIL sweep WITH the minimum-surface-pressure output "
    "each evaluation, so the cavitation margin is read off the candidate's "
    "own Cp_min instead of a NACA 24XX family member's. The t/c variable is "
    "gone — the weights are the thickness — and its margin becomes a "
    "constraint. SLOW: seconds per new section."
)


def _make_hydrofoil_section_builder(kind: str, kw: dict,
                                    chord_order: int = 0):
    def build(mission_kwargs, flags, bounds_overrides):
        from . import hydrofoil, hydrofoil_section as hfs, hydrotail
        if mission_kwargs:
            raise ValueError(
                "the water families have no mission spec — speed and depth "
                "are design variables (edit their box bounds instead)")
        cfg = dict(kw)
        cfg.update(_chord_kwargs(chord_order, flags))
        cfg.update(_water_kwargs(flags))
        # the FOIL's size, exactly as the twin that selects its section by
        # thickness reads it (_make_hydro_tail_builder): designing the
        # section says nothing about how wide the foil is, and without this
        # line the whole water section co-design flew the 1.2 m calibration
        # whatever the size card had been told.
        cfg.update(_planform_kwargs(flags, *_WET_SIZE))
        # ...and the craft weight, for the same reason and on all three kinds:
        # ``L_design`` is a field of every foil problem this builder wraps
        # (planar, winglet, elevator), and a designed section changes what
        # shape carries the load, never how much of it there is.
        cfg.update(_weight_kwargs(flags))
        # ...and the operating point's own two rows, on all three kinds:
        # ``DEPTH_BOUNDS``/``V_BOUNDS`` are fields of every foil problem this
        # builder wraps, and ``HydrofoilSectionProblem.bounds`` is the foil's
        # box stacked under the section's, so the row reaches the sampler and
        # the validator together (_water_band_kwargs).
        cfg.update(_water_band_kwargs(bounds_overrides))
        # ...and the FLIGHT CONDITION, on all three kinds: ``heel_deg`` and
        # ``tip_clearance_m`` are fields of every foil problem this builder
        # wraps, and the heel is the only term any of them has that reads the
        # foil's SPAN.
        cfg.update(_water_flight_kwargs(flags))
        cfg.update(_water_re_kwargs(flags))
        # ...and the FOIL'S OWN CANT, on the two LATTICE kinds only. The
        # planar kind is a monoplane Fourier solve with a straight
        # quarter-chord line and has no field for it, which is the same
        # reason the registry declares the keys on those two kinds alone.
        if kind in ("winglet", "tail"):
            cfg.update(_wing_cant_kwargs(flags))
        # ...and THE STRUT, which this builder did not pass on. Found by the
        # span/fin rule below needing to read it: ``_hydro_section_spec``
        # declares STRUT_KEYS on every designed-section water family, and
        # nothing here carried them to the constructor — so on those families
        # "no vertical stabiliser" changed nothing (the mast drag was still
        # charged, +14.5 % L/D on the shipped foil) and a fin section chosen
        # on stage 2.7 never reached the problem either. The exact defect
        # class this file is arranged against: a flag declared, offered, and
        # swallowed on the way to the run.
        cfg.update(_strut_config(flags))
        if kind == "tail":
            # ...and its STATION, which needs a second surface to be between
            cfg.update(_strut_station_config(flags))
        # ...and the SIZE rows, on all three kinds now. Declared to match by
        # ``_hydro_section_spec``, and the pairing has to hold in both
        # directions: a flag declared there that the constructor swallowed
        # would be a control the shell offers and the run never hears. The
        # elevator kind takes both, its stabiliser depth searched as a
        # fraction of the span where the span is searched too
        # (``hydrotail.Z_T_FRAC_LABEL``); ``hydrofoil._size_check`` refuses
        # the one combination that is left, in one place.
        cfg.update(_wet_size_kwargs(flags, bounds_overrides))
        if kind == "planar":
            # deliberately NOT capped: nothing on a planar foil reaches below
            # the foil plane, so draught IS depth_m and a cap here would be
            # exactly the bound the design box already offers. Same rule that
            # keeps the plain "hydrofoil" family out of the declaration.
            foil = hydrofoil.HydrofoilProblem(**cfg)   # no tip device at all
        elif kind == "winglet":
            cfg.update(_wet_blend_kwargs(flags))
            cfg.update(_winglet_chord_kwargs(flags))
            cfg.update(_draught_kwargs(flags))
            cfg.update(_foil_cant_kwargs(flags))
            foil = hydrofoil.HydrofoilWingletProblem(**cfg)
        else:
            cfg.update(_winglet_chord_kwargs(flags))
            if cfg.pop("_fixed_arm", False):
                arm = (flags or {}).get("l_t_m")
                cfg["l_t_fixed"] = float(
                    0.5 * (hydrotail.L_T_BOUNDS[0] + hydrotail.L_T_BOUNDS[1])
                    if arm is None else arm)
            else:
                cfg.update(_arm_band_kwargs(bounds_overrides))
            if cfg.get("free_height"):    # see _make_wing_tail_builder
                cfg.update(_height_band_kwargs(bounds_overrides))
            # ...and the FIXED BLEND, in this family's own field names — the
            # same four flags the elevator family without a designed section
            # already takes (_make_hydro_tail_builder). Designing the foil's
            # section is orthogonal to what shape its tip device's root is,
            # so this twin used to be the one place a blend asked for on the
            # shell's tip-device menu quietly did not happen.
            cfg.update(_blend_shape_flag(flags))
            cfg.update(_blend_frac_kwargs(flags))
            cfg.update(_junction_flag(flags, _blend_default_on(flags, False)))
            # the FOIL's section is designed here; the elevator may still fly
            # a chosen one (its Cp_min table comes with it)
            cfg.update(_wet_aft_section_kwargs(flags))
            if cfg.get("tail_winglet"):
                cfg.update(_tail_cant_kwargs(flags))
                cfg.update(_tail_blend_kwargs(flags))
            if cfg.get("winglet"):
                cfg.update(_foil_cant_kwargs(flags))
            cfg.update(_tail_layout_kwargs(flags))
            # ...and the RIG, on THIS kind only. It is a field of
            # HydrofoilTailProblem and of nothing else: the planar and
            # winglet kinds have no second surface to trim, so their trim is
            # in LIFT alone and a pitching couple has nowhere to go. Declaring
            # it on them would be the silent-drop defect check_flags exists
            # to expose, which is why _hydro_section_spec branches on the
            # same `kind` a few lines below.
            cfg.update(_rig_kwargs(flags))
            cfg.update(_draught_kwargs(flags))
            foil = hydrotail.HydrofoilTailProblem(**cfg)
        prob = hfs.HydrofoilSectionProblem(
            section=airfoil_problem_from_flags(flags), foil=foil)
        labels = prob.param_labels
        bounds = _apply_overrides(prob.bounds, labels, bounds_overrides)
        return _BuiltProblem(
            problem=prob,
            callable=lambda x: hfs.fg_hydrofoil_section(x, prob),
            evaluate=lambda x: hfs.evaluate_hydrofoil_section(x, prob),
            bounds=bounds, is_constrained=True, dim=prob.dim,
            param_labels=labels, medium="water",
        )
    build.takes_section_aft = kind == "tail"
    # ...wherever there is a device to blend: the main foil's on the winglet
    # kind, and on the elevator kind either the foil's or the elevator's own
    # (the same test _make_hydro_tail_builder applies to the family this one
    # designs a section for)
    build.takes_fixed_blend = (kind == "winglet"
                               or (kind == "tail"
                                   and bool(kw.get("winglet")
                                            or kw.get("tail_winglet"))))
    build.designs_section = True
    return build


#: generated name -> (kind, foil kwargs, family constraint labels)
HYDRO_SECTION_VARIANTS: dict[str, dict] = {
    "hydrofoil" + _HF_SECTION_SUFFIX: {
        "kind": "planar", "kwargs": {},
        "labels_g": ("cavitation margin",)},
    "hydrofoil + winglet" + _HF_SECTION_SUFFIX: {
        "kind": "winglet", "kwargs": {},
        "labels_g": ("cavitation margin",)},
}
for _n, _v in HYDRO_TAIL_VARIANTS.items():
    HYDRO_SECTION_VARIANTS[_n + _HF_SECTION_SUFFIX] = {
        "kind": "tail", "kwargs": dict(_v["kwargs"]),
        "labels_g": ("cavitation margin", "static margin - SM_min")}
del _n, _v


def _hydro_section_problem(variant: dict, chord_order: int = 0):
    from . import hydrofoil, hydrofoil_section as hfs, hydrotail
    cfg = dict(variant["kwargs"])
    if chord_order:
        cfg["chord_order"] = chord_order
    kind = variant["kind"]
    if kind == "planar":
        foil = hydrofoil.HydrofoilProblem(**cfg)
    elif kind == "winglet":
        foil = hydrofoil.HydrofoilWingletProblem(**cfg)
    else:
        if cfg.pop("_fixed_arm", False):
            cfg["l_t_fixed"] = 0.5 * (hydrotail.L_T_BOUNDS[0]
                                      + hydrotail.L_T_BOUNDS[1])
        foil = hydrotail.HydrofoilTailProblem(**cfg)
    return hfs.HydrofoilSectionProblem(foil=foil)


def _hydro_section_spec(name: str, variant: dict,
                        chord_order: int = 0) -> ProblemSpec:
    prob = _hydro_section_problem(variant, chord_order)
    labels = prob.param_labels
    labels_g = tuple(variant["labels_g"]) + ("t/c margin", "|Cm| margin")
    flags = ("l_t_m",) if variant["kwargs"].get("_fixed_arm") else ()
    if chord_order:
        flags = flags + ("chord_max_frac", CHORD_LAW_KEY)
    if variant["kwargs"].get("tail_winglet"):
        flags = flags + TAIL_TIP_KEYS
    if variant["kind"] == "tail":
        # the CG, the separation and the RIG all belong to the same object:
        # HydrofoilTailProblem is the only foil problem this builder wraps
        # that has a second surface, hence the only one with a pitch balance
        # for a couple to enter (see the builder's `kind == "tail"` branch).
        flags = (flags + (TAIL_CG_KEY, TAIL_MOUNT_KEY)
                 + TAIL_LIMIT_KEYS + RIG_KEYS + STRUT_STATION_KEYS)
        if not variant["kwargs"].get("free_height"):
            flags = flags + (TAIL_HEIGHT_KEY,)
    flags = flags + (WATER_KEY, WEIGHT_KEY) + STRUT_KEYS
    if variant["kind"] in ("winglet", "tail"):
        # the FOIL's own cant and sweep, on the LATTICE kinds only — the
        # planar kind is a monoplane Fourier solve with a straight
        # quarter-chord line and has no field for them
        flags = flags + WING_CANT_KEYS
    if variant["kwargs"].get("winglet") or variant["kind"] == "winglet":
        # the MAIN FOIL's tip-device SHAPE, on the section twins too
        flags = flags + ("winglet_type",)
    # THE SIZE and the FLIGHT CONDITION. Every kind carries both rows
    # (:func:`wet_size_keys`); the span row's remaining refusal is the strut
    # at zero heel, which is a stage-1 answer and not a family property.
    flags = flags + WATER_FLIGHT_KEYS + (WATER_RE_KEY,) + wet_size_keys(
        variant["kind"] == "tail" and bool(variant["kwargs"].get("free_height")))
    # the DRAUGHT CAP, wherever this family has geometry below the foil plane
    # — the tip device on the winglet kind, the stabiliser (and its own tip
    # device) on the elevator kind. Not on the planar kind, where draught is
    # depth_m and the cap would duplicate a box row (see the builder).
    if variant["kind"] != "planar":
        flags = flags + (DRAUGHT_MAX_KEY,)
    return ProblemSpec(
        name, f"Hydrofoil + CST section, live XFOIL ({len(labels)}-D)",
        "water", True, labels, True,
        _make_hydrofoil_section_builder(variant["kind"], variant["kwargs"],
                                        chord_order),
        n_constraints=len(labels_g),
        description=f"{_HF_SECTION_NOTE}",
        constraint_labels=labels_g,
        flags=flags,
        has_blocks=True,
    )


for _name, _variant in HYDRO_SECTION_VARIANTS.items():
    PROBLEM_SPECS[_name] = _hydro_section_spec(_name, _variant)
del _name, _variant


# =====================================================================
# Wing size (span + area) as configuration
# =====================================================================
# Every problem below builds a wing whose planform the design vector
# RESHAPES but never RESIZES: b and S are constructor values. Declaring them
# as flags here — rather than writing PLANFORM_KEYS into seventeen spec
# literals — keeps the list of "problems you may resize" in ONE place, and
# the free-chord-law twins generated next inherit it automatically.
#
# Not in the list, on purpose: "free planform (aircraft)" (b and S are design
# variables there — narrow their box instead), the CAR families (their span is
# a design VARIABLE now, carrying its own band — carwing.span_row — and a
# typed span beside a searched one is two answers to one question) and the
# 2-D "airfoil (section)" (no wing).
#
# The WATER families ARE in the list, since 2026-08-01. They were held out
# because "their size carries calibrated cavitation numbers", and the first
# half of that is true: the published margins were MEASURED on a 1.2 m /
# 0.144 m2 foil. But the margin is not a stored number — it is recomputed at
# every panel's own submergence for whatever geometry the candidate draws
# (hydrofoil.cavitation_margin_panels), and so are the Reynolds number, the
# mast drag and the Froude depth. Holding the size fixed therefore removed a
# design question rather than protecting a calibration: a foiling craft's
# span and area are the first two numbers its designer chooses, and under
# water the whole planform menu was a disabled control saying "no". What the
# calibration justifies is the DEFAULT, not the ban.

_RESIZABLE_PROBLEMS = (
    "trim wing", "wing t/c + sweep", "mission wing",
    "winglet", "winglet_capped", "winglet, blended (span-capped)",
    "winglet + t/c", "winglet_capped + t/c",
    # ...and the coupled-library twins of that pair, for the reason the pair
    # is here: demanding a section from a library says nothing about whether
    # the SPAN is the user's, and leaving them out would make the size card
    # vanish the moment the two library rows appeared
    "winglet + airfoil (coupled)", "winglet_capped + airfoil (coupled)",
    "wing + airfoil (XFOIL)", "winglet + airfoil (XFOIL)",
    "winglet_capped + airfoil (XFOIL)",
    "wing+airfoil (coupled)", "tail", "tail (fixed arm)", "tandem",
    # ...and the pair with its thickness searched, beside the pair it is a
    # variant of: both go through _make_tandem_builder, which calls
    # _tandem_planform_kwargs the same way. Searching a section thickness
    # says nothing about whether the SPANS are the user's, and leaving this
    # out would make the size card vanish the moment the t/c row appeared —
    # the exact failure the designed-tail specs hit two entries below.
    "tandem + t/c",
    # ...and the wing-side blended device, beside the winglet-side one it is
    # the same wing as: both fly objective.Problem, which reads the chosen
    # size the same way whichever end of the corner the transition starts at.
    "winglet, blended into the wing (span-capped)",
    # ...and the two DESIGNED-TAIL specs, beside the two assumed-tail ones
    # they are a variant of. Same builder (_make_tail_builder), same
    # _planform_kwargs call: designing the stabiliser says nothing about
    # whether the WING's span is the user's, and leaving them out made the
    # size question disappear the moment the tail became a designed surface.
    "tail [designed tail]", "tail (fixed arm) [designed tail]",
    # the two chord-law twins that shipped under their own names (every
    # other twin is generated from a base spec below and inherits this)
    "wing (free chord law)", "winglet + free chord law",
    # ...and the water families: the foil alone, the foil with a tip device,
    # and every hydrofoil+elevator variant (added just below, from the
    # generated table, for the reason every generated family is)
    "hydrofoil", "hydrofoil + winglet",
)
_RESIZABLE_PROBLEMS = _RESIZABLE_PROBLEMS + tuple(HYDRO_TAIL_VARIANTS)

# ...and every water family that DESIGNS its section (HYDRO_SECTION_VARIANTS),
# beside the air one it mirrors ("wing + airfoil (XFOIL)", which has honoured
# a chosen size since the flags existed). Its builder reads them the same way
# now — a designed section and a chosen span are answers to two different
# questions, and holding the size fixed here made the size card vanish for
# exactly the pipeline that has designed the most.
_RESIZABLE_PROBLEMS = _RESIZABLE_PROBLEMS + tuple(HYDRO_SECTION_VARIANTS)

# ...and the CHOSEN-SECTION twin of every one of them. A twin copies its
# base's flags when it is REGISTERED (_chosen_section_spec), which happens
# above this block, so a size the base honours would otherwise be a size the
# twin silently dropped — and in water the twin is not an exotic case but the
# ordinary one: every V3 water pipeline flies the section stage 2 chose, so
# the span stopped being a question exactly when the pipeline was complete.
# Derived from CHOSEN_SECTION_TWINS rather than listed, so a family added to
# either table cannot fall out of the other.
_RESIZABLE_PROBLEMS = _RESIZABLE_PROBLEMS + tuple(
    CHOSEN_SECTION_TWINS[_n] for _n in _RESIZABLE_PROBLEMS
    if _n in CHOSEN_SECTION_TWINS)

for _name in _RESIZABLE_PROBLEMS:
    _spec = PROBLEM_SPECS[_name]
    _spec.flags = tuple(dict.fromkeys(tuple(_spec.flags) + PLANFORM_KEYS))
del _name, _spec


#: the plain winglet modes — the ones whose vector has no blend row, so a
#: FIXED blend (WINGLET_BLEND_KEY) has somewhere to live. The three modes
#: that DESIGN the blend are excluded by construction: objective.Problem
#: refuses the pair, and a flag offered on a problem that would raise is a
#: menu entry that lies.
_FIXED_BLEND_MODES = ("winglet", "winglet_capped", "winglet_tc",
                      "winglet_capped_tc",
                      "winglet_coupled", "winglet_capped_coupled")


# =====================================================================
# Modifier variants: the free chord law x the free flight state
# =====================================================================
# A MODIFIER is a design freedom that composes with a whole FAMILY instead of
# selecting one. Two of them exist:
#
#   chord   the chord law reshapes Wing.chord(y), which every solver in the
#           package samples the same way (the VLM at its panel stations, the
#           lifting line at its own, the beam models through the solved
#           loading), so it composes with every family that builds a Wing.
#   flight  speed and altitude are not geometry at all: they set the flow
#           state (ISA + Sutherland) and, through the design weight, the trim
#           target CL = W/(q S) that every family already flies at — so any
#           family that knows its weight can carry them (mission.flight_state).
#
# Neither is a flag, because each changes the design DIMENSION and
# ProblemSpec.param_labels is static. Each combination is therefore its own
# registered problem — and since the modifiers are orthogonal to each other
# as well as to the family, the registry offers the full PRODUCT: every
# family x every subset of the modifiers it supports. That is what makes
# "optimise the winglet AND the chord law AND where it flies, together" a
# problem you can select rather than a combination that resets itself.
#
# The variants are GENERATED from their base specs: labels, constraint names,
# mission contract, flags and slow/constrained status are all inherited, so a
# variant cannot drift away from the problem it extends. Only the builder
# (which passes chord_order / flight_free) and the extra label blocks differ.
#
# Who supports what, and why not (extension points, not oversights):
#   * hydrofoil / hydrofoil + winglet: speed AND depth are ALREADY design
#     variables there — the water family has carried its own flight state
#     since Tier C — so a flight modifier would offer the same freedom twice.
#   * car rear wing (+ endplates): the car's speed is a TRACK condition, not
#     a design variable; the drag budget and the deflection limit are quoted
#     at it. Freeing it would let the optimiser choose to go slower.
#   * airfoil (section): a 2-D section, no wing and no trim target.
#   * trim wing + flight IS "mission wing" (and its chord twin), which shipped
#     first and keeps its name; generating it again would register the same
#     problem under two names.

#: base problem -> how many of its constraint labels are the FAMILY's own
#: (the size margin is inserted after them). Default: all of them.
_SIZE_MARGIN_INSERT: dict[str, int] = {}


#: modifier order in the design vector — the stacking rule
#: ([family][size][flight][chord]) and the order the names are composed in.
#: The three SIZE modifiers occupy the same slot and are mutually exclusive
#: (SIZE_MODIFIERS): they are three answers to one question — how is this
#: wing sized — so no problem may carry two.
MODIFIERS = ("size", "size_ws", "size_ws_free", "flight", "chord")

#: the size modes, in MODIFIERS order. A variant may carry at most one.
SIZE_MODIFIERS = ("size", "size_ws", "size_ws_free")

MODIFIER_SUFFIX = {"size": "free planform", "size_ws": "free span (W/S)",
                   "size_ws_free": "free span + free W/S",
                   "flight": "free flight state", "chord": "free chord law"}
MODIFIER_DISPLAY = {"size": " + free planform",
                    "size_ws": " + free span (W/S fixed)",
                    "size_ws_free": " + free span + searched W/S",
                    "flight": " + flight state", "chord": " + chord law"}
#: extra ProblemSpec.flags a modifier brings (a VALUE knob on its own block).
#:
#: Every sized mode takes ``wing_loading_limit_pa`` — the MISSION's own
#: ceiling on W/S, off its constraint diagram — because every sized mode can
#: fly past it: the free planform reaches a loading through its area row, and
#: the two loading modes state or search one directly. It is a LIMIT, not a
#: band end: it clips the searched band and refuses any candidate above it
#: (sizing.check_wing_loading), which is what a stall speed does.
MODIFIER_FLAGS = {
    "size": ("span_min_m", "span_max_m", "area_min_m2", "area_max_m2",
             "wing_loading_limit_pa") + MATERIAL_FLAG_KEYS,
    "chord": ("chord_max_frac", CHORD_LAW_KEY), "flight": (),
    "size_ws": ("wing_loading_pa", "span_min_m", "span_max_m",
                "wing_loading_limit_pa") + MATERIAL_FLAG_KEYS,
    "size_ws_free": ("ws_min_pa", "ws_max_pa", "span_min_m", "span_max_m",
                     "wing_loading_limit_pa") + MATERIAL_FLAG_KEYS,
}

_CHORD_NOTE = (
    "CHORD LAW: the chord distribution is a free cubic on top of the "
    "straight-taper baseline at UNCHANGED area (3 coefficients appended to "
    "the design vector), so the planform is no longer restricted to a "
    "trapezoid. A law that collapses the chord anywhere is refused (penalty "
    "contract), and the area rescale is closed-form, so the trim target is "
    "unaffected."
)

_FLIGHT_NOTE = (
    "FLIGHT STATE: speed and altitude are design variables (2 rows before "
    "the chord block). Each candidate's density and viscosity come from ISA "
    "+ Sutherland at ITS altitude and the trim target is CL = W/(q S) for "
    "the mission's design WEIGHT, so flying faster genuinely re-trims the "
    "aircraft instead of relabelling it. At the family's own (V, altitude) "
    "the score is its published one. Documented limit: the section polars "
    "stay a single-Re set, so Re moves with the state while the section data "
    "do not."
)

_SIZE_NOTE = (
    "FREE PLANFORM: span and area are design variables (2 rows ahead of the "
    "flight block), and freeing them honestly costs three changes at once — "
    "the wing WEIGHS what its size implies (Raymer's statistical GA wing "
    "weight, closed as a fixed point in the gross weight), the trim target "
    "follows that weight (CL = W_total/(q S)), and the score becomes PAYLOAD "
    "L/D = W_fixed/D, because with W varying across candidates 'maximise "
    "L/D' would reward a heavier wing that lifts better. A root-bending "
    "stress margin joins as a constraint; without it the span would simply "
    "slam the box (sizing.py / aircraft.py). The default fixed weight is the "
    "one this family's own trim target implies and the wing's structure is "
    "ADDED to it, so a sized problem is heavier than the fixed-CL problem it "
    "extends."
)

_SIZE_WS_NOTE = (
    "FREE SPAN AT FIXED WING LOADING: the AREA is not a design variable — it "
    "follows the chosen W/S through the same weight loop (S = W_total/(W/S), "
    "closed as a fixed point), so the vector carries the SPAN alone, inside "
    "a band you state. Everything the free-planform modifier costs is paid "
    "here too: the wing weighs what its size implies, the score is payload "
    "L/D = W_fixed/D and the root-bending stress margin is a constraint. "
    "What is different is WHICH question the optimiser answers. W/S is the "
    "mission's — a constraint diagram answers it (constraint_diagram.py) and "
    "it fixes the trim lift coefficient outright, CL = (W/S)/q — while the "
    "span is the aerodynamicist's. Freeing both lets the search re-open the "
    "mission; freeing the span alone does not."
)

_SIZE_WS_FREE_NOTE = (
    "FREE SPAN AND SEARCHED WING LOADING: the same weight loop as the other "
    "two sized modes — the area follows the loading (S = W_total/(W/S), a "
    "fixed point), the score is payload L/D and the root-bending stress "
    "margin is a constraint — but the LOADING is a design variable, in a "
    "band you state in N/m². This is the mode for 'I do not know my W/S': it "
    "asks the question in the units a constraint diagram answers it in, "
    "rather than through an area row whose band is a fraction of some "
    "reference size and so cannot even reach the right loadings for a "
    "mission of a different weight. The mission's own ceiling "
    "(wing_loading_limit_pa — stall or landing field in air, fly-up or "
    "cavitation in water) clips that band, so the search may re-open the "
    "mission's question only between loadings the mission allows."
)

MODIFIER_NOTES = {"size": _SIZE_NOTE, "chord": _CHORD_NOTE,
                  "flight": _FLIGHT_NOTE, "size_ws": _SIZE_WS_NOTE,
                  "size_ws_free": _SIZE_WS_FREE_NOTE}

#: labels a modifier appends to the design vector (order as in MODIFIERS)
_SIZE_LABELS = ("b_m", "S_m2")

#: variants that shipped under their OWN names, before the modifiers were
#: generalised. Kept verbatim so saved runs, presets and the report's tables
#: still resolve — "mission wing" IS the trim wing with the flight modifier
#: on, and naming it twice would register one problem under two names.
_VARIANT_ALIASES = {
    ("trim wing", frozenset({"chord"})): "wing (free chord law)",
    ("winglet", frozenset({"chord"})): "winglet + free chord law",
    ("trim wing", frozenset({"flight"})): "mission wing",
    ("trim wing", frozenset({"flight", "chord"})):
        "mission wing + free chord law",
}


def _size_mode_of(mods) -> bool | str:
    """The ``size_free`` VALUE a modifier set implies (sizing.size_mode).

    ``False`` when no size modifier is on, ``True`` for the two-variable
    one, ``"wing_loading"`` for the one that derives the area from a STATED
    W/S and ``"wing_loading_free"`` for the one that searches the loading.
    They are mutually exclusive by construction — the generation loop never
    offers a set with two — and this is where that fact is turned into the
    value every builder passes down.
    """
    from .sizing import SIZE_MODE_WS, SIZE_MODE_WS_FREE

    if "size_ws" in mods:
        return SIZE_MODE_WS
    if "size_ws_free" in mods:
        return SIZE_MODE_WS_FREE
    return "size" in mods


def is_sized(mods) -> bool:
    """Does this modifier set size the wing (either way)?

    Both modes cost the same three things — the weight coupling, the payload
    L/D objective and the root-bending stress CONSTRAINT — so everything
    that keys on "is this problem sized" has to ask about the pair, never
    about ``"size"`` alone.
    """
    return bool(set(mods) & set(SIZE_MODIFIERS))


def variant_name(base: str, mods) -> str:
    """Registered name of ``base`` carrying the modifier set ``mods``."""
    mods = frozenset(mods)
    if not mods:
        return base
    alias = _VARIANT_ALIASES.get((base, mods))
    if alias is not None:
        return alias
    return base + "".join(f" + {MODIFIER_SUFFIX[m]}"
                          for m in MODIFIERS if m in mods)


def _variant_display(display: str, dim: int, mods) -> str:
    """Base display string, dimension token retargeted and modifiers named."""
    out = _DIM_TOKEN.sub(f"{dim}-D", display, count=1)
    return out + "".join(MODIFIER_DISPLAY[m] for m in MODIFIERS if m in mods)


_DIM_TOKEN = re.compile(r"\d+-D")


#: base problem -> (labels, chord_order, flight_free) -> builder. One entry
#: per FAMILY; the product over modifier subsets is generated below.
_VARIANT_FACTORIES: dict[str, Callable] = {
    # objective.Problem modes (geometry.bounds appends both modifier blocks)
    # the flight modifier on the trim wing IS objective.py's "mission" mode:
    # same two rows, same position, and it shipped first — so the factory
    # selects the mode rather than adding a second way to say it
    "trim wing":
        lambda lbl, co, ff, sf: _build_objective(
            "mission" if ff else "trim", lbl, chord_order=co, size_free=sf),
    "wing t/c + sweep":
        lambda lbl, co, ff, sf: _build_objective("tier_a_plus", lbl,
                                             chord_order=co, flight_free=ff,
                                             size_free=sf),
    "winglet":
        lambda lbl, co, ff, sf: _build_objective("winglet", lbl, chord_order=co,
                                             flight_free=ff, size_free=sf),
    "winglet_capped":
        lambda lbl, co, ff, sf: _build_objective("winglet_capped", lbl,
                                             chord_order=co, flight_free=ff,
                                             size_free=sf),
    "winglet, blended (span-capped)":
        lambda lbl, co, ff, sf: _build_objective("winglet_capped_blended", lbl,
                                             chord_order=co, flight_free=ff,
                                             size_free=sf,
                                             junction_drag=True),
    "winglet, blended into the wing (span-capped)":
        lambda lbl, co, ff, sf: _build_objective(
            "winglet_capped_blended_wing", lbl, chord_order=co,
            flight_free=ff, size_free=sf, junction_drag=True),
    "winglet + t/c":
        lambda lbl, co, ff, sf: _build_objective("winglet_tc", lbl,
                                             chord_order=co, flight_free=ff,
                                             size_free=sf),
    "winglet_capped + t/c":
        lambda lbl, co, ff, sf: _build_objective("winglet_capped_tc", lbl,
                                             chord_order=co, flight_free=ff,
                                             size_free=sf),
    "winglet + airfoil (coupled)":
        lambda lbl, co, ff, sf: _build_objective("winglet_coupled", lbl,
                                             chord_order=co, flight_free=ff,
                                             size_free=sf),
    "winglet_capped + airfoil (coupled)":
        lambda lbl, co, ff, sf: _build_objective("winglet_capped_coupled",
                                             lbl, chord_order=co,
                                             flight_free=ff, size_free=sf),
    # families with their own solver and their own kwargs
    "free planform (aircraft)":
        lambda lbl, co, ff, sf: _make_aircraft_builder(co, ff),
    "hydrofoil":
        lambda lbl, co, ff, sf: _make_hydrofoil_builder(co),
    "hydrofoil + winglet":
        lambda lbl, co, ff, sf: _make_hydrofoil_winglet_builder(co),
    # every car family designs its area (see the registry block above), so
    # the chord-law twins do too — ``area_free=True`` here is not a variant,
    # it is what the base family already is
    "car rear wing":
        lambda lbl, co, ff, sf: _make_car_wing_builder(co, area_free=True),
    "car rear wing (two-element)":
        lambda lbl, co, ff, sf: _make_car_wing_multi_builder(co,
                                                             area_free=True),
    "car rear wing + endplates":
        lambda lbl, co, ff, sf: _make_car_endplate_builder(co,
                                                           area_free=True),
    "tail":
        lambda lbl, co, ff, sf: _make_tail_builder(False, co, ff, sf),
    "tail (fixed arm)":
        lambda lbl, co, ff, sf: _make_tail_builder(True, co, ff, sf),
    # the designed-tail twins: the chord modifier gives BOTH surfaces a law
    # (wing block then tail block), exactly as it does for a tandem pair
    "tail [designed tail]":
        lambda lbl, co, ff, sf: _make_tail_builder(False, co, ff, sf,
                                                   tail_free=True),
    "tail (fixed arm) [designed tail]":
        lambda lbl, co, ff, sf: _make_tail_builder(True, co, ff, sf,
                                                   tail_free=True),
    "wing+airfoil (coupled)":
        lambda lbl, co, ff, sf: _make_wing_airfoil_builder(co, ff, sf),
    # shape the SECTION, the chord distribution and the flight state together
    # (live XFOIL): both modifier blocks join the WING block, so the
    # portfolio split between a cheap planform block and an expensive section
    # block still holds
    "winglet + airfoil (XFOIL)":
        lambda lbl, co, ff, sf: _make_winglet_section_builder(
            False, co, flight_free=ff, size_free=sf),
    "winglet_capped + airfoil (XFOIL)":
        lambda lbl, co, ff, sf: _make_winglet_section_builder(
            True, co, flight_free=ff, size_free=sf),
    "wing + airfoil (XFOIL)":
        lambda lbl, co, ff, sf: _make_winglet_section_builder(
            False, co, winglet=False, flight_free=ff, size_free=sf),
    # both tandem wings get their own chord law; the flight state is ONE
    # state for the pair (they fly together)
    "tandem":
        lambda lbl, co, ff, sf: _make_tandem_builder(co, ff, sf),
    # the t/c twin composes with everything the plain pair does: the two
    # thickness rows sit ahead of the size / flight / chord blocks, so those
    # three still read back by counting from the end (tandem.bounds)
    "tandem + t/c":
        lambda lbl, co, ff, sf: _make_tandem_builder(co, ff, sf,
                                                     tc_free=True),
}

#: base problem -> modifiers it supports (see the block comment for the
#: families that support only one, and why).
_VARIANT_SUPPORT: dict[str, tuple] = {
    # b and S are already DESIGN VARIABLES there (it is the calibrated
    # weight-coupled sizing problem the size modifier generalises), so
    # offering the modifier would free them twice
    "free planform (aircraft)": ("flight", "chord"),
    "hydrofoil": ("chord",),                 # speed + depth already free
    "hydrofoil + winglet": ("chord",),
    # ONE modifier each, and the same one. NOT "flight": the track's speed is
    # a condition, not a design variable. NOT "size" either — that modifier
    # is the aircraft's weight-coupled sizing (span AND area, closed against
    # a Raymer wing weight, plus a wing-loading ceiling off a mission these
    # families do not have), and BOTH of its dimensions are already design
    # variables here. The chord law composes exactly as it does everywhere:
    # on the slotted family the slot rows sit AHEAD of the size block, so the
    # chord coefficients are still the tail of the vector.
    "car rear wing": ("chord",),
    "car rear wing (two-element)": ("chord",),
    "car rear wing + endplates": ("chord",),
}

#: label builders for the families whose modifier rows are NOT trailing
def _mission_mode_labels(labels: tuple, co: int, ff: bool, sf) -> tuple:
    """Stacking for a family whose FLIGHT modifier is objective.py's
    ``"mission"`` MODE rather than a trailing block.

    ``geometry.bounds`` puts MISSION_V_BOUNDS / MISSION_ALT_BOUNDS_M inside
    the FAMILY block for ``mode == "mission"`` and vstacks the size rows
    after, so the real order is [family][V, alt][size][chord] — while the
    generic stacking emits size first. The labels were therefore transposed
    against their own boxes: measured on
    ``trim wing + free planform + free flight state``, ``b_m`` carried the
    (10, 25) V band and ``V_ms`` carried the (6, 40) span band, and setting
    ``pinned={"V_ms": 12.0}`` flew a SPAN of 12 m. A mechanical sweep of all
    772 buildable specs carrying V_ms + altitude_m + b_m found exactly these
    6 mismatched.

    Labels only: the physics layout is the truth here and is unchanged, so no
    stored design vector moves.
    """
    from .sizing import size_labels
    return (tuple(labels)
            + (_FLIGHT_LABELS if ff else ())
            + tuple(size_labels(sf))
            + _chord_label_block(co, None))


_VARIANT_LABEL_FNS: dict[str, Callable] = {
    "trim wing":
        lambda co, ff, sf: _mission_mode_labels(
            PROBLEM_SPECS["trim wing"].param_labels, co, ff, sf),
    "winglet + airfoil (XFOIL)":
        lambda co, ff, sf: _winglet_section_labels(co, True, ff, sf),
    "winglet_capped + airfoil (XFOIL)":
        lambda co, ff, sf: _winglet_section_labels(co, True, ff, sf),
    "wing + airfoil (XFOIL)":
        lambda co, ff, sf: _winglet_section_labels(co, False, ff, sf),
    "tandem": _tandem_labels,
    "tandem + t/c":
        lambda co, ff, sf: _tandem_labels(co, ff, sf, tc_free=True),
    # a designed tail carries a chord law PER SURFACE, so the trailing block
    # is not the generic single one — ask the problem itself
    "tail [designed tail]":
        lambda co, ff, sf: _designed_tail_labels(False, co, ff, sf),
    "tail (fixed arm) [designed tail]":
        lambda co, ff, sf: _designed_tail_labels(True, co, ff, sf),
}


def _designed_tail_labels(fixed_arm: bool, chord_order: int,
                          flight_free: bool, size_free: bool = False
                          ) -> tuple:
    from . import tail as tailmod

    kw = {}
    if fixed_arm:
        kw["l_t_fixed"] = tailmod.default_arm()
    return tailmod.TailProblem(tail_free=True, chord_order=chord_order,
                               flight_free=flight_free, size_free=size_free,
                               **kw).param_labels


def _wing_tail_labels(variant: dict, chord_order: int, flight_free: bool,
                      size_free: bool = False) -> tuple:
    """Labels of a nonplanar wing+tail variant — asked of the problem itself
    (its blocks are switched on individually, so a static list would rot)."""
    from . import wingtail
    kw = dict(variant["kwargs"])
    probe = {k: v for k, v in kw.items() if k != "_fixed_arm"}
    if kw.get("_fixed_arm"):
        from . import tail as tailmod
        probe["l_t_fixed"] = tailmod.default_arm()
    return wingtail.WingTailProblem(chord_order=chord_order,
                                    flight_free=flight_free,
                                    size_free=size_free,
                                    **_probe_polars(),
                                    **probe).param_labels


# ...and the nonplanar tandem family, which takes all three modifiers.
for _n, _v in TANDEM_VLM_VARIANTS.items():
    _VARIANT_FACTORIES[_n] = (
        lambda v: (lambda lbl, co, ff, sf:
                   _make_tandem_vlm_builder(v["winglets"], co, ff, sf,
                                            v["section"],
                                            v.get("cant", "fixed")))
    )(_v)
    _VARIANT_LABEL_FNS[_n] = (
        lambda v: (lambda co, ff, sf:
                   _tandem_vlm_labels(v["winglets"], co, ff, sf, v["section"],
                                      v.get("cant", "fixed")))
    )(_v)
    if _v["section"]:
        # [ (stress), t/c, |Cm| ]: the section's two stay last
        _SIZE_MARGIN_INSERT[_n] = 0
del _n, _v


# ...and every water + CST section variant (chord only: speed and depth are
# design variables there, and the size is calibrated geometry).
for _n, _v in HYDRO_SECTION_VARIANTS.items():
    _VARIANT_FACTORIES[_n] = (
        lambda v: (lambda lbl, co, ff, sf:
                   _make_hydrofoil_section_builder(v["kind"], v["kwargs"], co))
    )(_v)
    _VARIANT_LABEL_FNS[_n] = (
        lambda v: (lambda co, ff, sf:
                   _hydro_section_problem(v, co).param_labels)
    )(_v)
    _VARIANT_SUPPORT[_n] = ("chord",)
    # [cavitation, (static margin), t/c, |Cm|]: the section's two stay last
    _SIZE_MARGIN_INSERT[_n] = len(_v["labels_g"])
del _n, _v


# ...and every generated hydrofoil+elevator variant. Chord only: that family
# has flown speed and depth as design variables since Tier C.
for _n, _v in HYDRO_TAIL_VARIANTS.items():
    _VARIANT_FACTORIES[_n] = (
        lambda kw: (lambda lbl, co, ff, sf: _make_hydro_tail_builder(kw, co))
    )(_v["kwargs"])
    _VARIANT_LABEL_FNS[_n] = (
        lambda v: (lambda co, ff, sf: _hydro_tail_labels(v, co))
    )(_v)
    _VARIANT_SUPPORT[_n] = ("chord",)
del _n, _v


# ...and every CHOSEN-SECTION twin of a water family. Chord only, for the
# same reason: speed and depth are already design variables there.
for _base, _twin in CHOSEN_SECTION_TWINS.items():
    _VARIANT_FACTORIES[_twin] = (
        lambda mk: (lambda lbl, co, ff, sf: mk(co))
    )(_CHOSEN_SECTION_FACTORIES[_base])
    _VARIANT_LABEL_FNS[_twin] = (
        lambda mk: (lambda co, ff, sf:
                    mk(co)({}, {}, None).param_labels)
    )(_CHOSEN_SECTION_FACTORIES[_base])
    _VARIANT_SUPPORT[_twin] = ("chord",)
del _base, _twin


# ...and every wing+tail+SECTION variant, so the section co-design carries
# the modifiers too (the size and flight rows join the WING block, ahead of
# the section weights, exactly as they do in the winglet+section family).
for _n in WING_TAIL_SECTION_VARIANTS:
    # [static margin, (stress), t/c, |Cm|]: one family margin, then the
    # section's two
    _SIZE_MARGIN_INSERT[_n] = 1
del _n

for _n, _v in WING_TAIL_SECTION_VARIANTS.items():
    _VARIANT_FACTORIES[_n] = (
        lambda kw: (lambda lbl, co, ff, sf:
                    _make_wing_tail_section_builder(kw, co, ff, sf))
    )(_v["kwargs"])
    _VARIANT_LABEL_FNS[_n] = (
        lambda v: (lambda co, ff, sf: _wing_tail_section_labels(v, co, ff, sf))
    )(_v)
del _n, _v


# ...and every generated wing+tail variant, so both modifiers compose with
# the tail exactly as they do with everything else.
for _n, _v in WING_TAIL_VARIANTS.items():
    _VARIANT_FACTORIES[_n] = (
        lambda kw: (lambda lbl, co, ff, sf: _make_wing_tail_builder(kw, co, ff,
                                                                    sf))
    )(_v["kwargs"])
    _VARIANT_LABEL_FNS[_n] = (
        lambda v: (lambda co, ff, sf: _wing_tail_labels(v, co, ff, sf))
    )(_v)
del _n, _v


def _variant_labels_for(base: str, mods) -> tuple:
    """Design-vector labels of one variant (family override, else stacking)."""
    co = _CHORD_ORDER if "chord" in mods else 0
    ff = "flight" in mods
    sf = _size_mode_of(mods)
    fn = _VARIANT_LABEL_FNS.get(base)
    if fn is not None:
        return tuple(fn(co, ff, sf))
    return _variant_labels(PROBLEM_SPECS[base].param_labels, co, ff, sf)


def _make_variant_spec(base: ProblemSpec, name: str, mods, labels, build
                       ) -> ProblemSpec:
    """Spec of ``base`` carrying the modifier set ``mods`` (see above)."""
    notes = " ".join(MODIFIER_NOTES[m] for m in MODIFIERS if m in mods)
    extra_flags = tuple(f for m in MODIFIERS if m in mods
                        for f in MODIFIER_FLAGS[m])
    base_flags = tuple(base.flags)
    if is_sized(mods):
        # the SIZE is part of the design now — both modes put the span in
        # the vector, and the free-planform one the area as well — so the
        # "choose a size" flags must go: offering both would be two answers
        # to one question. (The wing-loading mode brings its OWN value
        # flags, MODIFIER_FLAGS["size_ws"]: the loading the area follows and
        # the span band the row is searched in.)
        #
        # The tandem pair's REAR span goes with them, for that reason and no
        # second one: sized, the pair carries one span row PER WING, so a
        # typed rear span would be a second answer to a question the design
        # box is already asking.
        _drop = set(PLANFORM_KEYS) | set(TANDEM_SPAN_KEYS)
        base_flags = tuple(f for f in base_flags if f not in _drop)
    # with the flight modifier on, speed and altitude are DESIGN VARIABLES,
    # so the mission card must stop offering them: a typed-in cruise speed
    # beside a speed the optimiser chooses is two answers to one question.
    # The design WEIGHT stays — that is what the modifier trims against.
    mission_fields = tuple(f for f in base.mission_fields
                           if "flight" not in mods
                           or f not in ("V", "altitude_m"))
    # the SIZE modifier turns an unconstrained family into a constrained one
    # and adds a margin to an already-constrained one (sizing.py)
    is_constrained = base.is_constrained or is_sized(mods)
    n_constraints = (base.n_constraints if base.is_constrained else 0)
    labels_g = tuple(base.constraint_labels) if base.is_constrained else ()
    if is_sized(mods):
        n_constraints += 1
        # the stress margin is appended to the FAMILY's own margins, which is
        # not always the end of the vector: the CST-section families keep the
        # section's two margins last (their evaluate builds
        # [family..., stress, t/c, |Cm|]), so the label has to land in the
        # same place the number does.
        at = _SIZE_MARGIN_INSERT.get(base.name, len(labels_g))
        labels_g = labels_g[:at] + ("root-bending stress margin",) \
            + labels_g[at:]
    return ProblemSpec(
        name, _variant_display(base.display, len(labels), mods),
        base.medium, is_constrained, tuple(labels), base.slow, build,
        n_constraints=max(1, n_constraints),
        description=f"{base.description} {notes}",
        # dict.fromkeys: keep order, no duplicates
        flags=tuple(dict.fromkeys(base_flags + extra_flags)),
        constraint_labels=labels_g,
        uses_mission=base.uses_mission,
        mission_fields=mission_fields,
        has_blocks=base.has_blocks,
    )


#: registered variant name -> (base problem, frozenset of modifiers)
# The FIXED BLEND, declared on every problem whose builder flies a plain
# winglet mode. Asked of the BUILDER, never of the problem's NAME: the same
# four modes appear under a dozen names once the section, chord-law and size
# twins are generated, and a name list would rot the moment another one is
# added.
#
# Placed here — after MODIFIER_FLAGS, before the variants are generated — so
# the order is the registry's own: a family's own flags first, the modifiers'
# VALUE knobs last. The handful of variants that shipped under their own
# names carry their modifier knob in a spec LITERAL, so appending blindly
# would leave those specs listing the blend after the chord-box width while
# every generated twin listed it before, and the two orders would disagree
# about the same problem.
_MODIFIER_VALUE_FLAGS = {f for fs in MODIFIER_FLAGS.values() for f in fs}

for _spec in PROBLEM_SPECS.values():
    if getattr(_spec.build, "objective_mode", None) not in _FIXED_BLEND_MODES:
        continue
    _own = [f for f in _spec.flags if f not in _MODIFIER_VALUE_FLAGS]
    _knobs = [f for f in _spec.flags if f in _MODIFIER_VALUE_FLAGS]
    _spec.flags = tuple(dict.fromkeys(_own + list(_FIXED_BLEND_FLAGS)
                                      + _knobs))
del _spec, _own, _knobs


#: families whose solver has been wired for the WING-LOADING size mode
#: (sizing.SIZE_MODE_WS): the area follows W/S, so the size block is ONE row
#: and the area's own fixed point has to be closed before the planform is
#: built. Every other family still offers the two-variable mode only —
#: listing them here before their evaluate() reads a one-row block would
#: register a problem whose vector and whose box disagree.
#:
#: Extension point, deliberately explicit: adding a family here is a promise
#: that its evaluate() calls sizing.resolve_spans (or resolve_size, its
#: one-span case) with a loading.
_WING_LOADING_FAMILIES = frozenset({
    "trim wing", "wing t/c + sweep", "winglet", "winglet_capped",
    "winglet, blended (span-capped)",
    "winglet, blended into the wing (span-capped)",
    "winglet + t/c", "winglet_capped + t/c",
    # ...and the coupled-library twins. The promise this list makes holds for
    # exactly the same reason it does on the "+ t/c" pair: they fly
    # objective.evaluate, whose size block is resolved before the planform is
    # built, and their tc_sec is in geometry.TC_INDEX so the weight model
    # reads the structural depth the candidate is actually demanding rather
    # than the 12 % default.
    "winglet + airfoil (coupled)", "winglet_capped + airfoil (coupled)",
    # ...and the aircraft WITH a tail: the wing's area follows the loading,
    # the tail's own area stays a design variable of its own (sizing the
    # aircraft is not sizing its stabiliser)
    "tail", "tail (fixed arm)",
    "tail [designed tail]", "tail (fixed arm) [designed tail]",
    # ...and the TANDEM PAIR, whose size block is TWO span rows: the pair's
    # TOTAL area follows the loading (split by the same area split the
    # candidate flies, each wing weighed on its own span — sizing.
    # resolve_spans), and the two spans are what the search is left to
    # decide. That is the pairing this family most needs: a tandem exists to
    # trade the rear wing's width against the front wing's wake, and against
    # a FREE area that trade is confounded with simply growing the pair.
    "tandem", "tandem (nonplanar) + winglets",
    "tandem (nonplanar) + CST section (XFOIL)",
    "tandem (nonplanar) + winglets + CST section (XFOIL)",
    # ...and the t/c twin of the plain pair: the thickness rows sit AHEAD of
    # the size block, so evaluate_tandem still reaches resolve_spans by the
    # same trailing count and the promise this list makes still holds
    "tandem + t/c",
})
#: ...and every NONPLANAR wing+tail variant (wingtail.py), added below once
#: WING_TAIL_VARIANTS is built — the same solver, with the tip devices and
#: the tail's own freedoms switched on individually.

#: every modifier subset the registry offers: the three size modes are
#: ALTERNATIVES (a problem is sized one way, never two), and each composes
#: with the flight state and the chord law independently.
MODIFIER_SUBSETS = tuple(
    frozenset(s for s in (_size, _fl, _ch) if s)
    for _size in (None, "size", "size_ws", "size_ws_free")
    for _fl in (None, "flight")
    for _ch in (None, "chord")
    if any((_size, _fl, _ch))
)


_WING_LOADING_FAMILIES = _WING_LOADING_FAMILIES | frozenset(
    WING_TAIL_VARIANTS) | frozenset(TANDEM_VLM_VARIANTS)


#: THE CAR ENDPLATE'S OWN TWO FREEDOMS — stated, or SEARCHED.
#:
#: Both are numbers the shell asks for today and nothing in this repository
#: records an answer to: no measurement here says what a rear wing's plates
#: lean by, or how far its corner is turned. That is the case for handing
#: them to the optimiser, and it is why they are offered as a TOGGLE rather
#: than replaced: a plate built to a drawing has a cant the builder knows,
#: and answering one more question must never take a freedom away.
#:
#: EACH IS A DIMENSION CHANGE, so each is a variant axis and not a flag —
#: the same rule ``_WING_CANTS`` exists for, and the same three consequences:
#: the row lands inside the family block
#: (``endplate.CarWingEndplateProblem.FREE_ROWS``), the STATED flag is
#: subtracted from the variant that searches it, and the discriminator a
#: shell branches on is the registered ``param_labels``
#: (:func:`plate_cant_is_searched`), never the name.
#:
#: WHY EACH IS WORTH A DESIGN VARIABLE — and the reason they are two axes and
#: not one four-way menu — is that they are priced against DIFFERENT things.
#: The cant trades the nonplanar benefit against two costs at once: the plate
#: projects OUTBOARD, which comes out of the span row because ``b_m`` here is
#: the OVERALL width, and it loses vertical reach (h·sin), which the reach
#: margin charges against the attachment deck. The blend trades the wing/plate
#: corner's interference charge against the same outboard projection AND a
#: plate that ends shorter. Two interior optima, two questions, two switches
#: — and a builder who knows one and not the other says so.
#:
#: ``blend_shape`` is deliberately NOT an axis: it names WHICH LAW the turn
#: follows (``geometry.BLEND_SHAPES``), and a categorical law is not a length
#: a continuous optimiser can search.
_PLATE_FREEDOMS: dict[str, tuple[str, str]] = {
    "cant": ("free cant", "endplate_cant_deg"),
    "blend": ("free blend", "blend_frac"),
}

#: the option keys above, public so a shell can offer exactly these and no
#: more (the contract :data:`WING_CANTS` and :data:`TAIL_DESIGNS` have)
PLATE_FREEDOMS: tuple = tuple(_PLATE_FREEDOMS)

#: the family whose plate is a DESIGNED part, and therefore the only one
#: these freedoms can be generated on: on ``car rear wing`` the plate is a
#: tip device of free height with no section, no toe and no reach to trade.
_PLATE_BASE = "car rear wing + endplates"


def _plate_variant_name(base: str, free) -> str:
    """``car rear wing + endplates [free cant, free blend]``.

    The bracketed-qualifier convention :func:`_wing_tail_name` uses, and the
    rule that protects every pre-existing name: the EMPTY set of freedoms
    contributes no bracket at all, so the two names this family shipped under
    regenerate byte-identically.
    """
    quals = [_PLATE_FREEDOMS[k][0] for k in _PLATE_FREEDOMS if k in free]
    return f"{base} [{', '.join(quals)}]" if quals else base


def _plate_variant_spec(base: ProblemSpec, free) -> ProblemSpec:
    """One free-plate variant of the designed-endplate family."""
    name = _plate_variant_name(base.name, free)
    build = _make_car_endplate_builder(area_free=True,
                                       cant_free=("cant" in free),
                                       blend_free=("blend" in free))
    labels = tuple(build({}, {}, None).param_labels)
    # THE STATED FLAG GOES. A value flag beside the row that already answers
    # it is one question answered twice, which is what `check_flags` exists to
    # refuse — and `CarWingEndplateProblem.__post_init__` raises on the pair
    # as well, so the two halves cannot drift.
    dropped = {_PLATE_FREEDOMS[k][1] for k in free}
    said = " and ".join(_PLATE_FREEDOMS[k][0].replace("free ", "")
                        for k in _PLATE_FREEDOMS if k in free)
    return ProblemSpec(
        name, _variant_display(base.display, len(labels), ()),
        base.medium, base.is_constrained, labels, base.slow, build,
        n_constraints=base.n_constraints,
        description=(f"{base.description} The plate's {said} is a DESIGN "
                     f"VARIABLE here rather than a number you state: it is "
                     f"priced at both ends — the plate's outboard projection "
                     f"comes out of the OVERALL width the span row bounds, "
                     f"and its vertical reach has to still get to the "
                     f"attachment deck — so there is an interior answer to "
                     f"find rather than a bound to ride."),
        flags=tuple(f for f in base.flags if f not in dropped),
        constraint_labels=base.constraint_labels,
    )


for _free in ({"cant"}, {"blend"}, {"cant", "blend"}):
    _pname = _plate_variant_name(_PLATE_BASE, _free)
    PROBLEM_SPECS[_pname] = _plate_variant_spec(
        PROBLEM_SPECS[_PLATE_BASE], _free)
    # ...and the chord law composes on every one of them, exactly as it does
    # on the base: the plate's rows sit inside the FAMILY block, so the chord
    # coefficients are still the tail of the vector. Registered HERE, above
    # the modifier product, because that loop is the ordering boundary — a
    # base name that arrives after it gets no twin, and one that arrives after
    # the declaration passes gets no section, chord-limit or AR flags either.
    _VARIANT_FACTORIES[_pname] = (
        lambda lbl, co, ff, sf, _f=frozenset(_free):
        _make_car_endplate_builder(co, area_free=True,
                                   cant_free=("cant" in _f),
                                   blend_free=("blend" in _f)))
    _VARIANT_SUPPORT[_pname] = ("chord",)
del _free, _pname


def plate_cant_is_searched(name: str) -> bool:
    """Is the ENDPLATE's cant a DESIGN VARIABLE in this family?

    The discriminator a shell needs to decide whether to offer the typed
    field or point at the design box — :func:`cant_is_searched`'s twin for
    the car's plate. Read off the registered problem's own ``param_labels``,
    never off its name: the chord-law twin of every free-plate variant is
    generated, and so is the pair that frees both.
    """
    spec = PROBLEM_SPECS.get(name)
    return bool(spec) and "endplate_cant_deg" in tuple(spec.param_labels)


def plate_blend_is_searched(name: str) -> bool:
    """...and the same question about the wing/plate CORNER."""
    spec = PROBLEM_SPECS.get(name)
    return bool(spec) and "endplate_blend_frac" in tuple(spec.param_labels)


def plate_freedom_name(base: str, free) -> str:
    """The registered family that frees ``free`` on ``base``, as a name.

    Public so a shell can turn a pair of toggles into a problem without
    restating the bracket convention. Returns ``base`` unchanged for the
    empty set, which is what "state them both" means.
    """
    return _plate_variant_name(base, set(free))


MODIFIER_VARIANTS: dict[str, tuple] = {}

for _base_name in list(_VARIANT_FACTORIES):
    _base_spec = PROBLEM_SPECS[_base_name]
    _supported = set(_VARIANT_SUPPORT.get(_base_name, MODIFIERS))
    if _base_name not in _WING_LOADING_FAMILIES:
        # both loading modes close the same fixed point, so a family is
        # wired for both or for neither — one list, not two
        _supported.discard("size_ws")
        _supported.discard("size_ws_free")
    for _mods in MODIFIER_SUBSETS:
        if not _mods <= set(_supported):
            continue
        _name = variant_name(_base_name, _mods)
        _labels = _variant_labels_for(_base_name, _mods)
        MODIFIER_VARIANTS[_name] = (_base_name, frozenset(_mods))
        if _name in PROBLEM_SPECS:
            continue        # shipped under its own name (the legacy twins)
        PROBLEM_SPECS[_name] = _make_variant_spec(
            _base_spec, _name, _mods, _labels,
            _VARIANT_FACTORIES[_base_name](
                _labels, _CHORD_ORDER if "chord" in _mods else 0,
                "flight" in _mods, _size_mode_of(_mods)))
del _base_name, _base_spec, _supported, _mods, _name, _labels


def _declare_section_flags():
    """Declare :data:`SECTION_KEY` on every family whose builder can fly a
    chosen section, :data:`SECTION_AFT_KEY` where it has a second surface
    with its own, and :data:`SECTION_PLATE_KEY` where its ENDPLATE is a
    designed part rather than a fence.

    Read off the BUILDER rather than restated per spec: the generated
    variants (chord / flight / size twins) get their builder from the same
    factory, so a family that can fly a chosen section keeps that property
    through every modifier, and one that selects its section by thickness
    never claims it. That is also what puts the plate's slot on the free-cant
    and free-blend twins without a line each: they are built by the same
    factory, so they declare the same slots.
    """
    for spec in PROBLEM_SPECS.values():
        extra = [k for k, attr in ((SECTION_KEY, "takes_section"),
                                   (SECTION_AFT_KEY, "takes_section_aft"),
                                   (SECTION_PLATE_KEY, "takes_section_plate"))
                 if getattr(spec.build, attr, False) and k not in spec.flags]
        if extra:
            spec.flags = tuple(spec.flags) + tuple(extra)


def _declare_builder_flags():
    """Declare the airfoil-namespaced SECTION contract on every family that
    designs a CST section from ``flags``.

    These keys (:data:`AIRFOIL_FLAG_KEYS` — the 2-D operating point, the two
    design gates, the CST dimension) are honoured by every co-design builder
    through :func:`airfoil_problem_from_flags`, and they have always been kept
    out of ``ProblemSpec.flags`` on purpose, because that tuple drives the
    shell's PHYSICS toggles. That was harmless while nothing validated flags.
    :func:`check_flags` does, so the contract has to be written down somewhere
    the validator can read it — hence ``builder_flags``, which is exactly the
    same declaration without the shell consequence.

    Read off the BUILDER (``designs_section``) rather than restated per spec,
    for the reason :func:`_declare_section_flags` gives: the generated chord /
    flight / size twins share their factory, so the property travels with them.
    """
    for spec in PROBLEM_SPECS.values():
        extra: tuple = ()
        if getattr(spec.build, "designs_section", False):
            extra += AIRFOIL_FLAG_KEYS
        # THE TIP DEVICE'S CANT BAND, on every plain winglet mode.
        # ``_winglet_cant_bounds`` reads both of these off ``flags`` and hands
        # the result to ``objective.Problem``, so both are honoured — and
        # NEITHER was declared anywhere on these families, which
        # :func:`check_flags` surfaced the moment it existed. See the session
        # 49 note in HANDOVER: ``winglet_type`` is a user-facing CHOICE that
        # the 1440 wing+tail families do put in ``flags`` (the shell offers a
        # menu), and the plain winglet family does not — that asymmetry is
        # real and is reported rather than fixed here, because moving a key
        # into ``flags`` puts a control on screen and that is a shell
        # decision, not a validation one. ``winglet_cant_bounds`` is the
        # explicit (lo, hi) override behind the menu and belongs here in
        # either case: it is a caller's escape hatch, not a toggle.
        if getattr(spec.build, "objective_mode", None) in _WINGLET_MODES:
            extra += ("winglet_type", "winglet_cant_bounds")
        if extra:
            spec.builder_flags = tuple(dict.fromkeys(
                tuple(spec.builder_flags) + extra))


def _declare_chord_limit_flags():
    """Declare :data:`CHORD_LIMIT_KEYS` on every problem that has a PLANFORM.

    "Has a planform" is read off the declared design vector — a taper row,
    per surface where there are two — rather than from a family list, for the
    usual reason: the same dozen families appear under 1444 names once the
    section, chord-law, flight and size twins are generated. The one problem
    excluded by that rule is the one that should be: the 2-D section problem
    has no wing, so a chord it does not draw cannot be limited.
    """
    for spec in PROBLEM_SPECS.values():
        if not any(str(lbl) == "taper" or str(lbl).startswith("taper_")
                   for lbl in spec.param_labels):
            continue
        spec.flags = tuple(dict.fromkeys(tuple(spec.flags)
                                         + CHORD_LIMIT_KEYS))


def _declare_ar_limit_flags():
    """Declare :data:`AR_LIMIT_KEYS` on every family that GATES the ratio.

    Read off the BUILDER (``gates_ar``) and NOT off the design vector, which
    is the difference between this pass and :func:`_declare_chord_limit_flags`
    above. Every family with a span and an area has an aspect ratio; only
    some of them ask ``sizing.check_ar_limit`` about it per candidate (the
    water families size their foil against a draught cap instead and have no
    such gate). Declaring the flag by the vector would therefore have put the
    limit on screen for a family that accepts it and never reads it — a
    silently dropped limit, which is the failure ``check_flags`` exists to
    refuse.
    """
    for spec in PROBLEM_SPECS.values():
        if not getattr(spec.build, "gates_ar", False):
            continue
        spec.flags = tuple(dict.fromkeys(tuple(spec.flags) + AR_LIMIT_KEYS))


def _declare_water_blend_flags():
    """Declare the FIXED-BLEND flags on the imaged water family that carries
    a tip device — read off the builder, exactly as above.

    The air winglet family got them in one pass over the objective modes; the
    water family has its own solver (an imaged nonplanar VLM) and its own
    field names, so it needs its own pass, but the flags — and their meaning
    — are the same four.
    """
    for spec in PROBLEM_SPECS.values():
        if not getattr(spec.build, "takes_fixed_blend", False):
            continue
        # ...in the registry's own order — a family's own flags first, the
        # modifiers' VALUE knobs last — so a generated twin lists them in the
        # same order as the base it extends (this pass runs after the
        # variants are generated, so appending blindly would put the blend
        # after the chord-box width on the twins and before it on the base)
        own = [f for f in spec.flags if f not in _MODIFIER_VALUE_FLAGS]
        knobs = [f for f in spec.flags if f in _MODIFIER_VALUE_FLAGS]
        spec.flags = tuple(dict.fromkeys(own + list(_FIXED_BLEND_FLAGS)
                                         + knobs))


def _declare_winglet_chord_flags():
    """Declare :data:`WINGLET_CHORD_KEY` on every problem that flies a WINGLET.

    "Flies a winglet" is read off the declared design vector — a device
    HEIGHT row, under whichever name the family gives it (``winglet_h_frac``,
    the tail's ``winglet_h_frac_t``, the tandem pair's ``winglet_h_front`` /
    ``winglet_h_rear``) — for the reason every declaration pass here reads
    the vector: the same dozen families appear under 1400+ names once the
    section, chord-law, flight and size twins are generated, and a name list
    would rot on the first new twin.

    The car families are the deliberate exclusion. Their device is an
    ENDPLATE (``endplate_h_m``), and its chord is already a design
    variable with its own reported thickness (``endplate_chord_ratio``,
    endplate.py) — one number, one owner. A plate is not a tip extension of
    the wing, which is the whole premise of continuing the chord law onto it.
    """
    for spec in PROBLEM_SPECS.values():
        if not any(str(lbl).startswith("winglet_h") for lbl in
                   spec.param_labels):
            continue
        # ...in the registry's own order — a family's own flags first, the
        # modifiers' VALUE knobs last — so a generated twin lists it in the
        # same place as the base it extends (the rule
        # _declare_water_blend_flags follows, and the one test_api's
        # flag-order test checks)
        own = [f for f in spec.flags if f not in _MODIFIER_VALUE_FLAGS]
        knobs = [f for f in spec.flags if f in _MODIFIER_VALUE_FLAGS]
        spec.flags = tuple(dict.fromkeys(own + [WINGLET_CHORD_KEY] + knobs))


#: WHICH SCALAR a wing-level search maximises. ``lod`` (the default, and every
#: stored run bit-for-bit) is the family's own objective — L/D at the trimmed
#: design lift, payload L/D under the size modifier, downforce/drag for a car
#: wing. ``composite`` is the weighted composite J of the six wing criteria
#: (:mod:`aerobo.wing_score`), which is the only way a user's stated priorities
#: — stall margin, spar load, a chord the shop can build — survive into the
#: search instead of being read off the answer afterwards.
#:
#: ``wing_score_weights`` is the criterion weight dict (a preset name or a
#: partial dict, merged over the ``efficiency`` preset);
#: ``wing_score_reference`` is the FROZEN (lo, hi) band payload the sub-scores
#: are normalised against — REQUIRED, because a wing has no library to
#: normalise against and a live min-max would make J non-stationary. It is
#: measured over the design box itself by :func:`wing_score_reference`.
WING_OBJECTIVE_FLAG_KEYS = ("wing_objective", "wing_score_weights",
                            "wing_score_reference")

#: the two scalars a wing-level run can maximise, and how each is spelled.
WING_OBJECTIVES = ("lod", "composite")


def wing_score_weights(weights: dict | str | None):
    """Coerce a preset name / partial dict to :class:`WingScoreWeights`.

    The wing-level twin of :func:`screen_weights`, and it merges the same
    way: a partial dict sits on top of the ``efficiency`` preset (L/D alone),
    so "bend 0.3" means what it looks like.
    """
    from .wing_score import weights_of
    return weights_of(weights)


def _wing_objective(flags: dict | None):
    """``(name, reference, weights)`` for a wing-level objective.

    ``("lod", None, None)`` — the default, and every stored run unchanged —
    or ``("composite", WingScoreReference, WingScoreWeights)`` with the band
    and the weights resolved HERE, so the built problem carries them and no
    evaluation can silently score against a different map.

    Three refusals, all of them at CONFIG time rather than mid-search:

    * an unknown objective name;
    * a composite with no frozen band. There is no shipped wing band and
      there cannot be one — an L/D of 30 is excellent for a car wing and poor
      for a sailplane — so the band is measured over the user's own design
      box (:func:`wing_score_reference`) and a run without one is refused
      rather than answered on a scale that moves every evaluation;
    * a weight on a criterion the band does not cover. That happens when the
      criterion did not vary across the box sample, or when this family never
      reports it; either way the run would be maximising a number that is
      partly undefined, and the caller is told which criterion it is.
    """
    if not flags:
        return "lod", None, None
    name = str(flags.get("wing_objective") or "lod")
    if name not in WING_OBJECTIVES:
        raise ValueError(f"unknown wing objective {name!r}; "
                         f"choose from {list(WING_OBJECTIVES)}")
    if name == "lod":
        return name, None, None
    from .wing_score import reference_from_payload
    payload = flags.get("wing_score_reference")
    if not payload:
        raise ValueError(
            "the composite objective needs a FROZEN normalisation band and "
            "none was given. A wing has no library to normalise against, so "
            "the band is measured over YOUR design box "
            "(api.wing_score_reference) before the search starts — a live "
            "min-max would make J non-stationary, a moving target no GP can "
            "fit")
    ref = reference_from_payload(payload)
    weights = wing_score_weights(flags.get("wing_score_weights"))
    missing = [k for k in weights.active() if k not in ref.bounds]
    if missing:
        raise ValueError(
            f"the normalisation band does not cover {missing} — measured "
            f"over {ref.n_feasible} box samples, that criterion either never "
            "varied or is not reported by this family. Re-measure the band "
            "on the box you are searching, or set those weights to zero")
    return name, ref, weights


#: The FOILING composite (:mod:`aerobo.foil_score`): a weight per FLIGHT
#: POINT rather than per property of one point, because a rider's priorities
#: are other operating conditions — light wind is the same craft at 7 m/s and
#: top speed is the same craft at 16.
#:
#: ``foil_score_weights`` is the criterion weight dict (a preset name or a
#: partial dict merged over ``efficiency``); ``foil_score_reference`` is the
#: FROZEN band, measured over the user's own box by :func:`foil_score_reference`;
#: ``foil_light_ms`` / ``foil_top_ms`` / ``foil_turn_n`` state WHERE the three
#: non-cruise points are. The points are flags rather than constants because a
#: windfoil's light wind is not a kitefoil's, and they are hashed into the
#: band's fingerprint so two runs measured at different points cannot be
#: compared by accident.
FOIL_OBJECTIVE_FLAG_KEYS = ("foil_objective", "foil_score_weights",
                            "foil_score_reference", "foil_light_ms",
                            "foil_top_ms", "foil_turn_n")

#: the two scalars a foiling run can maximise, and how each is spelled.
FOIL_OBJECTIVES = ("lod", "composite")


def foil_score_weights(weights: dict | str | None):
    """Coerce a preset name / partial dict to :class:`FoilScoreWeights`."""
    from .foil_score import weights_of
    return weights_of(weights)


def _foil_points(flags: dict | None):
    """The three non-cruise flight points, from the flags."""
    from .foil_score import (DEFAULT_LIGHT_MS, DEFAULT_TOP_MS, DEFAULT_TURN_N,
                             FlightPoints)
    f = flags or {}
    return FlightPoints(
        light_ms=float(f.get("foil_light_ms") or DEFAULT_LIGHT_MS),
        top_ms=float(f.get("foil_top_ms") or DEFAULT_TOP_MS),
        turn_n=float(f.get("foil_turn_n") or DEFAULT_TURN_N))


def _foil_objective(flags: dict | None):
    """``(name, reference, weights, points)`` for a foiling-level objective.

    ``("lod", None, None, points)`` — the default, and every stored run
    unchanged — or the composite with its band, weights and points resolved
    HERE, at config time, so no evaluation can score against a different map
    than the one the run declared.

    Refusals mirror :func:`_wing_objective`'s, plus one of its own: a run
    cannot ask for the wing composite AND the foiling composite, because both
    replace the same objective and the answer would depend on which
    declaration pass wrapped the builder last.
    """
    if not flags:
        return "lod", None, None, _foil_points(flags)
    name = str(flags.get("foil_objective") or "lod")
    if name not in FOIL_OBJECTIVES:
        raise ValueError(f"unknown foiling objective {name!r}; "
                         f"choose from {list(FOIL_OBJECTIVES)}")
    points = _foil_points(flags)
    if name == "lod":
        return name, None, None, points
    if str(flags.get("wing_objective") or "lod") == "composite":
        raise ValueError(
            "wing_objective and foil_objective are both 'composite' — they "
            "replace the same objective. Choose one: the wing composite "
            "scores ten properties of ONE flight point, the foiling one "
            "scores four flight points")
    from .foil_score import reference_from_payload
    payload = flags.get("foil_score_reference")
    if not payload:
        raise ValueError(
            "the foiling composite needs a FROZEN normalisation band and none "
            "was given. There is no library of foiling craft to normalise "
            "against, so the band is measured over YOUR design box "
            "(api.foil_score_reference) before the search starts — a live "
            "min-max would make J non-stationary, a moving target no GP can "
            "fit")
    ref = reference_from_payload(payload)
    weights = foil_score_weights(flags.get("foil_score_weights"))
    missing = [k for k in weights.active() if k not in ref.bounds]
    if missing:
        measured = dict(ref.n_measured or {})
        detail = ", ".join(f"{k} (measured on {measured.get(k, 0)} of "
                           f"{ref.n_samples} box samples)" for k in missing)
        raise ValueError(
            f"the normalisation band does not cover {missing} — {detail}. A "
            "criterion too few designs could reach is not a scale: move the "
            "flight point to somewhere this craft flies, re-measure the band, "
            "or set that weight to zero")
    return name, ref, weights, points


def _foil_metrics_fn(built, spec, mission_kwargs, flags, bounds_overrides,
                     weights, points, inner_build):
    """``x -> {criterion: value}`` bound to THIS problem and its points.

    Builds the TURN twin — the same craft at ``turn_n`` times its weight —
    once, here, and only when a turn weight asks for it: the load factor is a
    flag, not a design row, so it is the one point that cannot be reached by
    substituting a column.
    """
    from .foil_score import flight_metrics

    labels = tuple(built.param_labels)
    wanted = weights.active() if weights is not None else ()
    speed_wanted = tuple(k for k in wanted if k != "margin_turn")
    if "V_ms" not in labels:
        if speed_wanted:
            raise ValueError(
                f"{spec.name if hasattr(spec, 'name') else 'this problem'} has "
                f"no V_ms row, so its speed cannot be moved and {list(speed_wanted)} "
                f"cannot be measured. Weight margin_turn and ld_cruise only, "
                f"or choose a family whose speed is a design variable")
        i_v = 0
    else:
        i_v = labels.index("V_ms")

    band = None
    if "V_ms" in labels:
        i = labels.index("V_ms")
        band = (float(built.bounds[i][0]), float(built.bounds[i][1]))
    points.check(band, wanted)

    evaluate_turn = None
    if "margin_turn" in wanted:
        if "weight_n" not in tuple(spec.flags or ()):
            raise ValueError(
                "the turn point needs the craft's weight as a flag "
                "('weight_n'), which this problem does not declare, so a "
                "load factor cannot be applied. Set margin_turn to zero")
        weight = float(getattr(built.problem, "L_design", 0.0) or 0.0)
        if weight <= 0.0:
            raise ValueError(
                "the turn point needs a positive design lift and this problem "
                "reports none")
        twin_flags = dict(_strip_foil_objective(flags))
        twin_flags["weight_n"] = weight * float(points.turn_n)
        twin = inner_build(mission_kwargs or {}, twin_flags, bounds_overrides)
        evaluate_turn = twin.evaluate

    def metrics(x):
        return flight_metrics(x, evaluate=built.evaluate, i_v=i_v,
                              points=points, evaluate_turn=evaluate_turn,
                              wanted=wanted or CRITERIA_ALL)

    return metrics


#: every criterion, for the band sweep — which measures all four so a user can
#: change the weights afterwards without paying for the sweep again.
CRITERIA_ALL = ("ld_cruise", "ld_light", "cav_top", "margin_turn")


def _with_foil_objective(spec):
    """Wrap ``spec.build`` so a foiling composite run scores every caller the
    same way — the registry-level trick :func:`_with_wing_objective` documents.
    """
    inner = spec.build

    @wraps(inner)
    def build(mission_kwargs, flags, bounds_overrides):
        built = inner(mission_kwargs, flags, bounds_overrides)
        name, reference, weights, points = _foil_objective(flags)
        if name != "composite":
            return built
        from .foil_score import composite_evaluation, make_composite_fg
        metrics = _foil_metrics_fn(built, spec, mission_kwargs, flags,
                                   bounds_overrides, weights, points, inner)
        raw_evaluate = built.evaluate
        n_g = int(len(getattr(built.problem, "constraint_labels", ()) or ())
                  or getattr(built.problem, "n_constraints", None)
                  or spec.n_constraints or 1)
        return replace(
            built,
            callable=make_composite_fg(raw_evaluate, reference, weights,
                                       metrics, n_constraints=n_g,
                                       constrained=built.is_constrained),
            evaluate=lambda x: composite_evaluation(
                x, raw_evaluate(x), reference, weights, metrics))

    return build


def _declare_foil_objective_flags():
    """Offer the foiling composite on every problem whose SPEED is a variable.

    "Has a speed row" is the honest test: three of the four criteria are other
    speeds, and a family that flies at one fixed speed cannot be asked what it
    does in light wind. That selects the water families and excludes every air
    problem, none of which carries ``V_ms``.
    """
    for spec in PROBLEM_SPECS.values():
        if not any(str(lbl) == "V_ms" for lbl in spec.param_labels):
            continue
        spec.flags = tuple(dict.fromkeys(tuple(spec.flags)
                                         + FOIL_OBJECTIVE_FLAG_KEYS))
        spec.build = _with_foil_objective(spec)


def section_thickness(value) -> float | None:
    """Geometric t/c of a CHOSEN section flag value, or None.

    Accepts everything :func:`section_polar_for` does — a library name, a
    ``{"name", "re", "mach"}`` pick, or ``{"w_upper", "w_lower"}`` CST
    weights — and answers with the thickness of the SHAPE, measured the same
    way the screen measures it (``airfoil_select.geometric_tc``). Never
    raises: a section that cannot be resolved has no thickness, and the
    criteria that need one are simply not measured.

    Exists because a thickness is a property of the section a surface flies,
    and a fixed-table polar carries none — so a wing+tail run that pinned a
    section on stages 2 and 2.5 knows its thicknesses while the breakdown
    does not.
    """
    from .airfoil_select import geometric_tc

    try:
        if isinstance(value, str):
            coords = _library_coords(value)
        elif isinstance(value, dict) and value.get("w_upper") is not None:
            from .airfoil import cst_coords
            coords = cst_coords(np.asarray(value["w_upper"], dtype=float),
                                np.asarray(value["w_lower"], dtype=float))
        elif isinstance(value, dict) and value.get("name"):
            coords = _library_coords(str(value["name"]))
        else:
            return None
        tc = float(geometric_tc(np.asarray(coords, dtype=float)))
    except Exception:            # noqa: BLE001 — a hint, never fatal
        return None
    return tc if np.isfinite(tc) and tc > 0.0 else None


def wing_thickness_hints(flags: dict | None) -> dict:
    """``{surface: t/c}`` for the sections this run pins, by the names
    :func:`wing_score.surfaces_of` gives the surfaces.

    The wing's section (:data:`SECTION_KEY`) names ``wing``/``front``; the
    second surface's (:data:`SECTION_AFT_KEY`) names ``tail``/``rear``, and
    a second surface with none of its own falls back to the wing's inside
    ``wing_score`` — which is what the solvers do too.
    """
    out: dict = {}
    for key, names in ((SECTION_KEY, ("wing", "front")),
                       (SECTION_AFT_KEY, ("tail", "rear"))):
        tc = section_thickness((flags or {}).get(key))
        if tc is not None:
            out.update(dict.fromkeys(names, tc))
    return out


def _wing_metrics_fn(built, flags: dict | None):
    """``raw -> criteria`` bound to THIS problem and its pinned sections.

    One closure, built in one place and handed to the band sweep, the
    objective and the score block alike: a band measured over a different
    criterion set than the objective reads would normalise J against
    something else entirely.
    """
    from . import wing_score as wsc

    hints = wing_thickness_hints(flags)
    prob = built.problem
    return lambda raw: wsc.design_metrics(raw, prob, hints)


def check_wing_objective(flags: dict | None) -> str:
    """The wing objective ``flags`` ask for, RAISING if the run is not
    launchable (:func:`_wing_objective`'s three refusals).

    For a caller that wants the refusal HERE — a shell validating its form
    before it queues a background run, so the reason lands beside the button
    that caused it rather than in a job's error field.
    """
    return _wing_objective(flags)[0]


def _with_wing_objective(spec: ProblemSpec):
    """Wrap a family's builder so ``wing_objective="composite"`` replaces the
    scalar it maximises — and NOTHING else.

    Wrapped at the registry rather than inside each of the dozen builders for
    the reason every declaration pass here works this way: those dozen
    builders appear under 1800+ names once the section, chord-law, flight and
    size twins are generated, and a per-builder branch would reach some of
    them and not others. Wrapping ``spec.build`` puts it on the ONE path every
    caller already goes through — :func:`run`, :func:`design_report`, the live
    metric sampler and the GUI all call ``spec.build`` — so a composite run
    cannot be scored one way by the optimiser and another way by the readout.

    The constraint channel is untouched: the composite reports the family's
    OWN margins, so a composite arm and an L/D arm share one feasible region
    (wing_score.composite_objective).
    """
    inner = spec.build

    @wraps(inner)
    def build(mission_kwargs, flags, bounds_overrides):
        built = inner(mission_kwargs, flags, bounds_overrides)
        name, reference, weights = _wing_objective(flags)
        if name != "composite":
            return built
        # THE LATERAL DECK IS PAID FOR ONLY WHERE IT IS WEIGHTED. Measuring
        # the spiral criterion puts a fin into the scored lattice and runs
        # dynamics.deck on it: +44 % of an evaluation, for a number no other
        # criterion reads. So the switch is the WEIGHT, not a flag the user
        # has to find and keep in step with it — ask for the criterion and
        # you have asked for its cost, leave it at zero and nothing changes.
        #
        # A family with no such attribute (the lifting-line tail, the plain
        # wing) is left alone and simply reports no spiral margin, which
        # ``design_metrics`` returns as None and ``composite`` refuses to
        # weight — the same contract every other unmeasurable criterion has.
        _arm_lateral(built, flags)
        from .wing_score import composite_evaluation, make_composite_fg
        evaluate = built.evaluate
        metrics = _wing_metrics_fn(built, flags)
        # the FAILURE vector's width. A built problem may declare more
        # margins than its static spec knows about (design_report follows the
        # same rule for the labels), so its own declaration wins and the
        # spec's count is the fallback — a penalty returned at the wrong
        # width is a shape error inside the optimiser, not a bad score.
        n_g = int(len(getattr(built.problem, "constraint_labels", ()) or ())
                  or getattr(built.problem, "n_constraints", None)
                  or spec.n_constraints or 1)
        return replace(
            built,
            callable=make_composite_fg(evaluate, reference, weights,
                                       n_constraints=n_g,
                                       constrained=built.is_constrained,
                                       metrics=metrics),
            evaluate=lambda x: composite_evaluation(evaluate(x), reference,
                                                    weights, metrics=metrics))

    return build


def _declare_wing_objective_flags():
    """Offer the composite objective on every problem that flies a PLANFORM.

    "Has a planform" is read off the declared design vector — a taper row,
    per surface where there are two — the same rule
    :func:`_declare_chord_limit_flags` uses, and it excludes exactly the one
    problem it should: the 2-D section problem has no wing, and it already
    has its own composite (``airfoil_objective``) over the six SECTION
    criteria.
    """
    for spec in PROBLEM_SPECS.values():
        if not any(str(lbl) == "taper" or str(lbl).startswith("taper_")
                   for lbl in spec.param_labels):
            continue
        spec.flags = tuple(dict.fromkeys(tuple(spec.flags)
                                         + WING_OBJECTIVE_FLAG_KEYS))
        spec.build = _with_wing_objective(spec)


def _declare_handling_flags():
    """Offer the handling gate on every family that has a lateral deck.

    Read off ``build.has_lateral_deck``, which
    :func:`_make_wing_tail_builder` sets — the nonplanar wing+tail families
    and nothing else, because ``wingtail.WingTailProblem.lateral`` is the
    only such attribute in the package. The lifting-line ``tail`` family is
    deliberately NOT offered it: it charges a fin's drag and weighs it but
    never flies one, so a Dutch-roll requirement there would be a
    requirement about a mode its solver cannot produce.
    """
    for spec in PROBLEM_SPECS.values():
        if not getattr(spec.build, "has_lateral_deck", False):
            continue
        spec.flags = tuple(dict.fromkeys(tuple(spec.flags)
                                         + (HANDLING_LEVEL_KEY,)))


_declare_handling_flags()
_declare_section_flags()
_declare_water_blend_flags()
_declare_chord_limit_flags()
_declare_ar_limit_flags()
_declare_winglet_chord_flags()
_declare_builder_flags()
# LAST: these two wrap the builder every pass above reads attributes off
# (``takes_section``, ``objective_mode``…). functools.wraps carries those
# through, but running them last means no pass ever has to rely on that.
# The foiling one goes outermost so its turn twin is built from a builder
# that already carries every other declaration.
_declare_wing_objective_flags()
_declare_foil_objective_flags()


#: problem name -> the same problem with the chord law ADDED (the mapping the
#: GUI and the presets have always used). Derived over the whole registry, so
#: a flight variant has a chord twin exactly as its base does.
CHORD_TWINS: dict[str, str] = {}

#: problem name -> the same problem with the flight state ADDED.
FLIGHT_TWINS: dict[str, str] = {}


def modifiers_of(name: str) -> frozenset:
    """Modifier set a registered problem carries (empty for a base family)."""
    return MODIFIER_VARIANTS.get(name, (None, frozenset()))[1]


def base_of(name: str) -> str:
    """The family a registered problem belongs to (itself, if it is one)."""
    return MODIFIER_VARIANTS.get(name, (name, None))[0]


def with_modifiers(name: str, mods) -> str | None:
    """Registered problem for ``name``'s family with EXACTLY ``mods`` on.

    ``None`` when that family does not support the combination — the one
    place the GUI has to ask, instead of keeping its own table of what
    composes with what.
    """
    base = base_of(name)
    mods = frozenset(mods)
    if not mods:
        return base if base in PROBLEM_SPECS else None
    if base not in _VARIANT_FACTORIES:
        return None
    if len(mods & set(SIZE_MODIFIERS)) > 1:
        # two answers to one question: a wing is sized ONE way. Asked here
        # rather than left to the name lookup so a caller composing
        # modifiers gets None (the "no such problem" answer every other
        # unsupported combination gives) instead of a KeyError.
        return None
    supported = set(_VARIANT_SUPPORT.get(base, MODIFIERS))
    if base not in _WING_LOADING_FAMILIES:
        supported.discard("size_ws")
    if not mods <= supported:
        return None
    target = variant_name(base, mods)
    return target if target in PROBLEM_SPECS else None


def add_modifier(name: str, mod: str) -> str | None:
    """The registered problem that is ``name`` PLUS one more modifier.

    ``None`` if it already carries it or its family does not support it.
    This is the whole composition rule the GUI needs: pick a family, then
    switch modifiers on one at a time, and every combination that exists is
    reachable by name.
    """
    mods = modifiers_of(name)
    if mod in mods:
        return None
    return with_modifiers(name, mods | {mod})


for _n in list(PROBLEM_SPECS):
    for _m, _table in (("chord", CHORD_TWINS), ("flight", FLIGHT_TWINS)):
        _t = add_modifier(_n, _m)
        if _t is not None:
            _table[_n] = _t
del _n, _m, _table, _t


# =====================================================================
# Optimiser registry
# =====================================================================

@dataclass
class OptimiserSpec:
    """Static description of one optimiser, spanning both registries."""

    name: str
    display: str
    supports_unconstrained: bool
    supports_constrained: bool
    needs_torch_problem: bool = False   # adjoint: exact-gradient torch problem
    note: str = ""


OPTIMISER_SPECS: dict[str, OptimiserSpec] = {
    "bo": OptimiserSpec(
        "bo", "Bayesian optimisation (BoTorch)", True, True,
        note="LogEI unconstrained; LogCEI (feasibility-weighted) constrained."),
    "ga": OptimiserSpec(
        "ga", "Genetic algorithm (pymoo)", True, True,
        note="feasibility-first tournament when constrained."),
    "random": OptimiserSpec("random", "Random search", True, True),
    "sobol": OptimiserSpec("sobol", "Sobol DOE", True, True),
    "grid": OptimiserSpec(
        "grid", "Full-factorial grid", True, False,
        note="unconstrained only; points-per-dim collapse with dimension."),
    "gradient": OptimiserSpec(
        "gradient", "Gradient multistart (L-BFGS-B, FD)", True, False,
        note="unconstrained only; use slsqp/penalty for constrained."),
    "adjoint": OptimiserSpec(
        "adjoint", "Adjoint (exact-gradient L-BFGS-B)", True, False,
        needs_torch_problem=True,
        note="trim-wing only (reverse-mode LLT gradient)."),
    "slsqp": OptimiserSpec(
        "slsqp", "SLSQP (constrained gradient)", False, True),
    "bo_slsqp": OptimiserSpec(
        "bo_slsqp", "BO, then SLSQP finisher (handoff)", False, True,
        note="a quarter of the budget builds the surrogate, the rest polishes "
             "from its best feasible design; constrained only, because slsqp "
             "is. Measured: above pure BO on 42 of 54 independent runs."),
    "penalty": OptimiserSpec(
        "penalty", "Quadratic penalty + L-BFGS-B", False, True),
    "blocks": OptimiserSpec(
        "blocks", "Block-coordinate portfolio (per-block optimisers)",
        False, True,
        note="alternating Gauss–Seidel over the problem's declared variable "
             "blocks, each with its own sub-optimiser; only for problems "
             "that declare blocks (winglet + airfoil XFOIL)."),
}


#: Optimisers that run a BO loop directly and therefore emit per-iteration GP
#: diagnostics. ``blocks`` is absent although its sub-searches ARE BO runs:
#: it never forwarded an ``iter_cb`` and wiring one is a separate change.
BO_ITER_OPTIMISERS: frozenset = frozenset({"bo", "bo_slsqp"})


#: The optimisers that can be HANDED a starting design — the second half of
#: "run A, then continue with B". The BO loops evaluate the stated design
#: first; the multistart local runners use it for their FIRST start and keep
#: drawing random ones after that, so a handoff never silently collapses a
#: multistart into a single polish. ``ga``, ``sobol``, ``random`` and ``grid``
#: are absent because they generate their own points by construction, and a
#: seed handed to them would be read by nothing.
#:
#: ``bo_slsqp`` is here for exactly the reason ``bo`` is, and its absence was
#: a defect rather than a policy: the seed enters the BO PHASE's initial
#: design (``_bo_x_init`` -> ``x_init``), so it is evaluated first and never
#: screened away, and the finisher then starts from the best FEASIBLE point
#: that phase saw — which is the seed itself, or better. The guarantee the
#: seed exists to buy ("this run cannot come back worse than the design I
#: gave it") therefore holds on the handoff exactly as it does on plain BO.
#: While it was missing, the shell's answer to "it found nothing" — "start
#: again from the closest design" — was dead on the arm EVERY constrained
#: family opens on (``v3.session.default_optimiser``): the button armed the
#: seed, ``check_x_seed`` passed it, and the launch raised.
X_SEED_OPTIMISERS: frozenset = frozenset(
    {"bo", "bo_slsqp", "gradient", "slsqp", "penalty"})


def problem_names() -> list[str]:
    return list(PROBLEM_SPECS)


def optimiser_names() -> list[str]:
    return list(OPTIMISER_SPECS)


def default_mission_values(problem_name: str, flags: dict | None = None
                           ) -> dict:
    """Exact default operating-point values for the spec's ``mission_fields``.

    The GUI pre-fills its always-visible mission editor with these and sends
    back ONLY the fields the user actually changed — an untouched editor
    therefore reproduces the legacy defaults bit-for-bit (W_N in particular is
    an exact IEEE expression, NOT the rounded literal 652.8; see
    mission.default_mission).

    ``flags`` (optional) is the same dict the run will be built with. It
    matters when the run RESIZES the wing (PLANFORM_KEYS): the default weight
    is the one that trims that area to the legacy CL of 0.5, so a resized
    wing carries a proportionally resized default weight instead of flying
    the 10 m^2 weight at a lift coefficient nobody chose. Omitted (or empty)
    reproduces the published values exactly.
    """
    spec = PROBLEM_SPECS[problem_name]
    if not spec.mission_fields:
        return {}
    from .mission import default_mission
    # the FAMILY carries the operating point, so a MODIFIER twin has to be
    # asked as its base: matching the name exactly handed
    # "free planform (aircraft) + free chord law" a generic wing's 652.8 N at
    # 14.6 m/s instead of that problem's own 12 kN at 50 m/s
    if base_of(problem_name) == "free planform (aircraft)":
        from . import aircraft
        m = aircraft.AircraftProblem().mission
    else:
        # The design weight is per-problem because CL_target = W_N/(q*Sref)
        # and Sref is NOT 10 everywhere — tandem references its TOTAL area of
        # 20 m^2, so sharing one W_N across problems would silently halve or
        # double the trim lift coefficient.
        m = default_mission(S=mission_sref(problem_name, flags))
    vals = {"W_N": float(m.W_N), "V": float(m.V),
            "altitude_m": float(m.altitude_m)}
    return {k: vals[k] for k in spec.mission_fields}


def mission_sref(problem_name: str, flags: dict | None = None) -> float:
    """Reference area the problem's CL_target is defined on [m^2].

    Read off the built problem itself rather than hard-coded, so a change
    to a problem's area — including one the caller CHOSE through
    ``flags`` (PLANFORM_KEYS) — cannot silently desynchronise the mission
    card from what the solver flies.

    Through the WRAPPER on a section-designing family (:func:`_flown_family`),
    because that is where the builder applies the mission: the nonplanar
    tandem's CST twin converts a stated ``W_N`` against the pair's own 20 m²,
    so reading the wrapper's (absent) area and falling back to 10 m² made
    :func:`default_mission_values` publish half the design weight — a
    "default" that, sent back, would have halved CL_target from 0.5 to 0.25.
    """
    spec = PROBLEM_SPECS[problem_name]
    built = spec.build({}, flags or {}, None)
    prob = _flown_family(built.problem)
    for attr in ("S_total", "S"):
        val = getattr(prob, attr, None)
        if isinstance(val, (int, float)) and float(val) > 0.0:
            return float(val)
    mission = getattr(prob, "mission", None)
    if mission is not None and getattr(prob, "S", None):
        return float(prob.S)
    return 10.0


#: the attributes a SECTION-DESIGNING problem keeps its flown family on.
#: Those problems are WRAPPERS: ``HydrofoilSectionProblem`` carries the foil
#: on ``.foil`` and ``WingTailSectionProblem`` the layout (wing+tail, or the
#: nonplanar tandem) on ``.wing_tail``, and neither wrapper has a span, an
#: area or a design load of its own. So every question about what is FLOWN
#: has to be asked of the wrapped problem, or 457 of the registry's problems
#: answer "no planform, no design load" about families that publish both.
#: Same attribute list :func:`_trim_problem` searches, and for the same
#: reason: growing a section does not change what a family flies.
_SECTION_WRAPPER_ATTRS = ("wing_tail", "foil")


def _flown_family(prob):
    """The problem whose geometry and operating point are actually flown.

    Unwraps a section-designing wrapper (see
    :data:`_SECTION_WRAPPER_ATTRS`); returns ``prob`` itself otherwise, so
    every non-wrapper family is untouched.
    """
    for attr in _SECTION_WRAPPER_ATTRS:
        inner = getattr(prob, attr, None)
        if inner is not None:
            return inner
    return prob


def planform_size(problem_name: str, flags: dict | None = None):
    """(span [m], area [m^2]) the problem will actually fly, or ``None``.

    ``None`` means the problem has no fixed planform to report — either the
    size is in the design vector (the free-planform aircraft) or there is no
    wing at all (the 2-D section). Read off the BUILT problem, so it reflects
    a chosen size (PLANFORM_KEYS) and can never drift from what is flown.
    The tandem pair reports its per-wing span and its TOTAL area.

    A section-designing family is read through its wrapper
    (:func:`_flown_family`): ``tail + CST section (XFOIL)`` flies the very
    wing+tail ``tail`` flies, so it has to report the same planform.
    """
    built = PROBLEM_SPECS[problem_name].build({}, flags or {}, None)
    prob = _flown_family(built.problem)
    b = getattr(prob, "b", None)
    S = getattr(prob, "S_total", None)
    if not isinstance(S, (int, float)) or not float(S) > 0.0:
        S = getattr(prob, "S", None)
    if not isinstance(b, (int, float)) or not isinstance(S, (int, float)):
        return None
    if not (float(b) > 0.0 and float(S) > 0.0):
        return None
    return float(b), float(S)


def resizable(problem_name: str) -> bool:
    """Does this problem honour a chosen span/area (PLANFORM_KEYS)?"""
    return all(k in PROBLEM_SPECS[problem_name].flags for k in PLANFORM_KEYS)


#: flags that can MOVE a problem's span box, and nothing else: the nominal
#: span the fractional band is taken around (PLANFORM_KEYS) and the band a
#: caller states outright (WING_LOADING_KEYS). Restricting the cache key to
#: these is what makes :func:`span_box` cheap enough to call from a view.
_SPAN_BOX_KEYS = (*PLANFORM_KEYS, *SIZE_BAND_KEYS)


@lru_cache(maxsize=512)
def _span_box_cached(problem_name: str, items: tuple,
                     label: str = "b_m") -> tuple | None:
    spec = PROBLEM_SPECS[problem_name]
    built = spec.build({}, dict(items), None)
    labels = tuple(getattr(built, "param_labels", spec.param_labels))
    if label not in labels:
        return None
    row = np.asarray(built.bounds, dtype=float)[labels.index(label)]
    return float(row[0]), float(row[1])


def span_box(problem_name: str, flags: dict | None = None,
             label: str = "b_m") -> tuple | None:
    """The ``b_m`` interval a run will ACTUALLY search [m], or ``None``.

    ``label`` names WHICH span row: a tandem pair carries one per wing
    (``b_m``, then ``b_rear_m`` — sizing.span_labels), and each is boxed off
    the built problem for the reason below.

    ``None`` where the span is not a design variable at all — every problem
    without a size modifier, where the span is the constructor value
    :func:`planform_size` reports.

    Read off the BUILT problem, exactly as :func:`planform_size` is, and for
    the same reason: in the wing-loading mode the span box is a FRACTION of
    the nominal span the problem is GIVEN (sizing.span_bounds), so a caller
    quoting ``PROBLEM_SPECS[name].default_bounds["b_m"]`` is quoting the
    family's published band — 6–40 m on the 10 m trim wing — at a wing that
    has been resized to 2.4 m. That is a view disagreeing with its own run,
    which is why this is a function of the FLAGS rather than a lookup.
    """
    sent = tuple(sorted((k, float(v)) for k, v in (flags or {}).items()
                        if k in _SPAN_BOX_KEYS and v is not None))
    return _span_box_cached(problem_name, sent, label)


def constraint_labels_of(problem_name: str, flags: dict | None = None,
                         mission_kwargs: dict | None = None) -> tuple:
    """The margins a run will ACTUALLY report, in order — names, not a count.

    ``ProblemSpec.constraint_labels`` is a STATIC declaration, and on a family
    whose margin set depends on its configuration it is only the default one.
    The car families are now exactly that: nothing is budgeted unless the user
    states a limit, so a plain ``car rear wing`` reports one margin (the
    deflection) and the same family with a drag ceiling reports two.

    That matters because every consumer names a margin BY INDEX
    (``gui.diagnose.infeasibility_report``: ``labels[j] if j < len(labels)``).
    Against a static list one index too short the extra margin comes out as
    ``g[1]``; against one too long — which is what a static
    ``("drag budget margin", "deflection margin")`` would have been after the
    budget stopped being a default — a wing's DEFLECTION margin is printed
    under the drag budget's name, on every surface that reports margins, and
    the relax planner then proposes a widening for the wrong constraint.

    Read off the BUILT problem for the reason :func:`span_box` is: a view that
    reads the static table is a view disagreeing with its own run. Falls back
    to the static declaration when the build fails, because a report with
    approximate names still tells a reader more than one with none.
    """
    spc = PROBLEM_SPECS[problem_name]
    static = tuple(spc.constraint_labels or ())
    try:
        built = spc.build(mission_kwargs if spc.uses_mission else None,
                          flags or {}, None)
    except Exception:                     # noqa: BLE001 — a read-out
        return static
    got = getattr(built.problem, "constraint_labels", None)
    return tuple(got) if got else static


def _trim_problem(prob):
    """The trimming problem inside a built one, or None.

    A section-carrying family wraps the layout it flies (``.wing_tail`` on
    the CST wing+tail, ``.foil`` on the hydrofoil+elevator), so the search
    is by ATTRIBUTE, not by problem name: a family that grows a section
    keeps trimming exactly as it did.
    """
    for attr in ("wing_tail", "foil"):
        inner = getattr(prob, attr, None)
        if inner is not None and hasattr(inner, "CL_target"):
            prob = inner
            break
    return prob if hasattr(prob, "CL_target") and hasattr(prob, "S") else None


def trim_surface_cl(problem_name: str, *, s_t_m2: float | None = None,
                    arm_m: float | None = None,
                    mission_kwargs: dict | None = None,
                    flags: dict | None = None,
                    bounds_overrides: dict | None = None) -> dict | None:
    """What the SECOND (trimming) surface flies at, before any run.

    ``None`` when the family has no trimming surface. Otherwise a dict:
    ``cl`` (the lift coefficient the trim implies), ``cl_target``,
    ``sref_m2``, ``s_lift_m2``, ``arm_m``, ``x_cg_m`` and ``source`` — one
    line naming the balance and the numbers it was closed on.

    The value is EXACT, not a correlation: :func:`tail.trim_lift_coefficient`
    eliminates the wing's load between the two trim equations the solvers
    themselves impose (Cm about the CG = 0, total CL = the target), so it
    agrees with the solved ``CL_t`` / ``CL_stab`` to the numerical area
    error. That matters to the shells: a stabiliser screened at zero lift is
    screened at a lift it never flies, and "L/D at the design Cl" is then 0
    for every candidate in the database.

    ``s_t_m2`` / ``arm_m`` default to the family's own mid-box (the same
    thing the design box will open on); an arm the family fixes wins over
    both, since there is nothing to choose.

    ``bounds_overrides`` is THE RUN'S OWN DESIGN BOX, and passing it is what
    stops this read-out being a third author of the tail's download. The
    water families' OPERATING POINT — depth and speed — is a pair of design
    rows a caller may widen (``_water_band_kwargs``), and the rig couple that
    decides the SIGN of the stabiliser's load is thrust times
    (depth + z_ce). With the box withheld, the lever closed on
    ``spec.default_bounds['depth_m']``'s mid-point, 0.575 m, whatever depth
    the run was actually told to fly: a caller whose craft rides on a 0.30 m
    mast got a couple computed at nearly twice its lever, and the card could
    therefore quote a LIFTING stabiliser in front of a run that flies a
    download. That is the defect recorded in this repo as "the tail's
    download has two authors", one level up, and the fix is the same — the
    number quoted is read off the problem that was BUILT, never off the
    family's published band. ``None`` (nothing overridden) is bit-for-bit
    what this function has always returned: with no rows given the built
    problem's ``DEPTH_BOUNDS`` / ``V_BOUNDS`` ARE the published box rows,
    checked family by family in tests/test_api.py.
    """
    from . import tail as _tail

    spec = PROBLEM_SPECS[problem_name]
    box = spec.default_bounds
    if "S_t_m2" not in box:
        return None
    built = spec.build(mission_kwargs or {}, flags or {}, bounds_overrides)
    prob = _trim_problem(built.problem)
    if prob is None:
        return None

    _labels = tuple(getattr(built.problem, "param_labels", ()) or ())
    _searched = np.asarray(getattr(built, "bounds", ()), dtype=float)

    def _mid(label):
        """Mid-point of a design-box row — off the box the RUN SEARCHES.

        The published :attr:`ProblemSpec.default_bounds` is what a row
        DEFAULTS to, not what this run asks. Once a caller hands in
        ``bounds_overrides`` the two differ, and a readout that quotes the
        published row computes the stabiliser's load from an area and an arm
        the run never flies — which can put the SIGN of that load, and so
        which way up its section is mounted, opposite to what the run then
        reports. That is this repo's recorded "the tail's download has two
        authors", arriving through the readout instead of through a core.

        Falls back to the published row for anything the built problem does
        not carry as a SEARCHED row — a pinned one, a fixed arm, or a family
        that never had the row — which is every caller that passed no
        override.
        """
        if label in _labels and _searched.size:
            lo, hi = _searched[_labels.index(label)]
            return 0.5 * (float(lo) + float(hi))
        row = box.get(label)
        return None if row is None else 0.5 * (float(row[0]) + float(row[1]))

    def _operating_mid(field: str, label: str):
        """The mid-point of an operating row, off the BUILT problem.

        ``DEPTH_BOUNDS`` / ``V_BOUNDS`` are instance fields the design box
        travels into, so they are the rows the run searches; the published
        box row is only what they default to. Falls back to the published
        row for a family that carries no such field (every air one).
        """
        row = getattr(prob, field, None)
        if row is None:
            return _mid(label)
        return 0.5 * (float(row[0]) + float(row[1]))

    s_t = float(s_t_m2) if s_t_m2 is not None else _mid("S_t_m2")
    arm = arm_m if arm_m is not None else _mid("l_t_m")
    fixed = getattr(prob, "l_t_fixed", None)
    if fixed is not None:
        arm = float(fixed)
    if s_t is None or arm is None or not float(s_t) > 0.0:
        return None
    s_t = float(s_t)

    # the SIGNED station and the EQUIVALENT HORIZONTAL area, both from the
    # layout's own helpers: a canard trims from upstream, and a V-tail
    # carries its share on S cos^2 G
    t_type = getattr(prob, "tail_type", None)
    l_t = (_tail.surface_x(t_type, abs(arm)) if isinstance(t_type, str)
           else float(arm))
    s_lift = (_tail.lifting_area(t_type, float(getattr(prob, "dihedral_deg",
                                                       0.0)), s_t)
              if isinstance(t_type, str) else s_t)

    # CL_target is a number on the aircraft families and a function of the
    # speed on the hydrofoil (its weight is fixed, so the coefficient moves
    # with V); the design speed is the one the mission states — or, where
    # speed is a design row, the mid-point of the row the RUN searches
    cl_target = prob.CL_target
    v_design = None
    if callable(cl_target):
        v = getattr(prob, "V", None)
        if v is None:
            v = _operating_mid("V_BOUNDS", "V_ms")
        if v is None:
            return None
        v_design = float(v)
        cl_target = cl_target(float(v))
    x_cg = getattr(prob, "x_cg", None)
    if x_cg is None:
        x_cg_for = getattr(prob, "x_cg_for", None)
        if x_cg_for is None:
            return None
        x_cg = x_cg_for(l_t)
    sref = float(prob.S)
    # the SECTIONS' OWN COUPLE, on the same terms the solver carries it
    # (tail.section_moment): a cambered surface pitches nose-down at every
    # lift, so it is part of the balance that fixes this surface's load —
    # and on a conventional layout it is what makes that load a DOWNLOAD.
    # Read before the run, from the same box mid-point the area and the arm
    # come from, so the number quoted here is the number flown.
    m_ac, m_ac_t, m_ac_note = _section_couple(prob, box, s_lift, l_t)
    # ...and the RIG's, which on a driven craft is several hundred times the
    # sections' and is what actually decides the sign (_rig_couple). It joins
    # ``m_ac`` rather than sitting beside it because the balance takes ONE
    # moment term, and it deliberately does NOT join ``m_ac_t``: mirroring
    # the stabiliser's section mirrors that surface's own couple and nothing
    # else, so the rig stays in the reference reading the orientation is
    # decided on — the same rule the wing's couple follows.
    m_rig, m_rig_note = (0.0, "") if v_design is None else _rig_couple(
        prob, box, float(x_cg), v_design,
        depth_m=_operating_mid("DEPTH_BOUNDS", "depth_m"))
    m_ac = m_ac + m_rig
    m_ac_note = m_ac_note + m_rig_note
    cl = _tail.trim_lift_coefficient(float(cl_target), sref, float(x_cg),
                                     float(s_lift), float(l_t), M_ac=m_ac)
    if not math.isfinite(cl):
        return None
    # WHICH WAY UP the surface will fly its section, and what that does to
    # this very number. A surface that pushes down mounts its section
    # inverted (tail.tail_polar), and an inverted section's couple has the
    # OPPOSITE sign — so the orientation feeds back into the balance that
    # chose it. Decided once from the upright reading, exactly as the
    # solvers decide it (tail.stabiliser_load), then the load is re-read on
    # the section actually flown.
    stated = getattr(prob, "tail_inverted", None)
    if stated is None:
        # decided on the WING's couple alone (tail.stabiliser_load): the
        # surface's own couple mirrors with the choice, so letting it vote
        # would be self-reference
        cl_ref = _tail.trim_lift_coefficient(
            float(cl_target), sref, float(x_cg), float(s_lift), float(l_t),
            M_ac=m_ac - m_ac_t)
        inverted = bool(math.isfinite(cl_ref) and cl_ref < 0.0)
    else:
        inverted = bool(stated)
    if inverted and m_ac_t:
        m_ac = m_ac - 2.0 * m_ac_t          # mirror the TAIL's share alone
        cl = _tail.trim_lift_coefficient(float(cl_target), sref, float(x_cg),
                                         float(s_lift), float(l_t), M_ac=m_ac)
        if not math.isfinite(cl):
            return None
    return {"cl": float(cl), "cl_target": float(cl_target),
            "sref_m2": sref, "s_lift_m2": float(s_lift),
            "arm_m": float(l_t), "x_cg_m": float(x_cg),
            "m_ac_m3": float(m_ac),
            # ...and the rig's share of that one number, separately, so a
            # card can say which author is doing the work. 0.0 means "no
            # rig", which is every family that is not a driven craft.
            "m_rig_m3": float(m_rig),
            # THE SECTION IS SCREENED THE WAY UP IT IS MOUNTED, and the value
            # that does that is -cl, not |cl|.
            #
            # `cl` here is the AIRCRAFT-frame load (`trim_lift_coefficient` is
            # closed-form geometry and moments — no polar, no orientation), and
            # `cl_section` is contractually the lift at which the UPRIGHT
            # catalogue is read, in the base section's own frame. `polar.py`
            # fixes the map: `InvertedPolar.cl(a) = -base.cl(-a)`. So a surface
            # mounted inverted carrying aircraft-frame C flies its base section
            # at -C. The upright database must be read at **-cl**.
            #
            # This said `abs(cl)`, which is `-cl` generalised one step too far.
            # It is exactly right whenever the surface carries a DOWNLOAD —
            # 1920 of the 2064 families with a second surface, where -cl == |cl|
            # bit for bit — and that is every case the identity was written
            # against. It is wrong for a surface STATED inverted that trims to
            # an UP-load: the water elevator at its default CG carries +0.286
            # and gets there by cranking incidence against its own camber
            # (i_t -0.095 deg upright -> +4.514 deg inverted). `abs()` screened
            # it on the cambered side it never works.
            #
            # Measured on the cached NACA 2412 Re 1e6 table: the screened point
            # (+0.2864, alpha +0.469 deg) reads 55.3 counts and cp_min -0.611;
            # the flown point (-0.2865, alpha -4.828 deg) reads 83.3 counts and
            # cp_min -2.073. That is +50.7 % profile drag and a suction peak
            # 3.4x deeper — and on a hydrofoil cp_min feeds a cavitation gate
            # 1:1, so the screen could clear a section the surface cavitates on.
            # It also reordered the library: screened winner NACA 2406 is the
            # flown LAST of five. Stated with its denominator, because the two
            # readings differ and the first draft quoted the wrong one: taking
            # the screened winner costs 18.4 % of the section L/D actually
            # available (28.07 against 2412's 34.39), which is the same fact as
            # "the flown best is 22.5 % better than the screened winner"
            # (34.39/28.07). The cost is 18.4 %; 22.5 % is its reciprocal.
            #
            # No solver result moves — `tail.orient_section` always handled the
            # mounting correctly inside the cores, and `cl_section` is a
            # pre-run read-out. No published number moves either: `tail_mount`
            # appears in no script, study or report, only in gui/v3/config.py.
            "inverted": inverted,
            "cl_section": float(-cl) if inverted else float(cl),
            "source": f"trim balance: (CL_target {float(cl_target):.4g} × "
                      f"Sref {sref:.4g} m² × x_cg {float(x_cg):.4g} m "
                      f"{'+' if m_ac >= 0 else '−'} "
                      f"{'section + rig' if m_rig else 'section'} couple "
                      f"{abs(m_ac):.4g} m³) / "
                      f"(S {float(s_lift):.4g} m² × arm {float(l_t):.4g} m)"
                      + m_ac_note}


def _section_couple(prob, box: dict, s_lift: float,
                    l_t: float) -> tuple[float, float, str]:
    """(M_ac, the SURFACE's own share of it, a note), before any run.

    The share is returned separately because the orientation feeds back:
    a surface flown inverted carries the mirror of its own couple, and
    mirroring it means subtracting twice that share — never the wing's.

    The same quantity :func:`tail.section_moment` integrates on the solved
    planform, rebuilt from the published geometry and the box mid-point:
    M/q = cm_ac S mac per surface, MAC being (1/S)∫c² dy by definition. The
    wing is built at the box's mid taper (what the design box opens on); the
    trimming surface is its own rectangle at the family's aspect ratio.

    Returns (0.0, note) — the symmetric-section value, and a note saying so
    — for anything that cannot answer, so a family with no polar to read
    still gets a load rather than a None.
    """
    from . import geometry as _geom
    from . import tail as _tail
    from .polar import section_cm_ac

    def _pol(obj, tc=None):
        """A polar that can be READ, or None.

        Deliberately paranoid: a family that flies the section the user
        chose carries a _SectionRequired sentinel here, and touching any
        attribute on it RAISES (that is its whole job). A pre-run read-out
        must not become the thing that raises, so every probe is guarded
        and an unreadable polar simply means no couple to report.
        """
        for attr in ("polar", "section_polar"):
            try:
                pol = getattr(obj, attr, None)
            except Exception:                  # noqa: BLE001 — a read-out
                continue
            if pol is not None and _readable_cm(pol):
                return pol
        try:
            fam = getattr(obj, "polar_family", None)
            if fam is not None and tc is not None:
                pol = fam.at(float(tc))
                if _readable_cm(pol):
                    return pol
        except Exception:                      # noqa: BLE001 — a read-out
            pass
        return None

    def _readable_cm(pol) -> bool:
        try:
            return callable(getattr(pol, "cm", None))
        except Exception:                      # noqa: BLE001 — a sentinel
            return False

    def _mid(label):
        row = box.get(label)
        return None if row is None else 0.5 * (float(row[0]) + float(row[1]))

    tc = _mid("tc")
    pol_w = _pol(prob, tc)
    if pol_w is None:
        return 0.0, 0.0, "; no section moment (this family flies a section "\
                         "the run will choose, so its couple is not knowable "\
                         "here)"
    try:
        pol_t = getattr(prob, "polar_tail", None)
    except Exception:                          # noqa: BLE001 — a read-out
        pol_t = None
    if pol_t is None or not _readable_cm(pol_t):
        pol_t = pol_w
    taper = _mid("taper")
    try:
        wing = _geom.Wing(b=float(prob.b), S=float(prob.S),
                          taper=1.0 if taper is None else float(taper))
        mac_w = float(wing.mac)
    except (ValueError, TypeError):
        return 0.0, 0.0, "; no section moment (no wing planform to read)"
    ar_t = float(getattr(prob, "AR_t", None) or _tail.TAIL_AR)
    mac_t = float(np.sqrt(s_lift / ar_t)) if s_lift > 0.0 and ar_t > 0 else 0.0
    m_ac_t = section_cm_ac(pol_t) * float(s_lift) * mac_t
    m_ac = section_cm_ac(pol_w) * float(prob.S) * mac_w + m_ac_t
    if m_ac == 0.0:
        return 0.0, 0.0, "; symmetric sections carry no couple"
    return float(m_ac), float(m_ac_t), ""


def _rig_couple(prob, box: dict, x_cg: float, v: float,
                depth_m: float | None = None) -> tuple[float, str]:
    """``(M_rig/q [m^3], a note)`` — the RIG's couple, before any run.

    The THIRD author of the trimming surface's load, and the reason this
    function exists rather than the couple simply living in
    :func:`_section_couple`: on a driven craft it is the LARGEST of the three
    (``rig.py``: ~399 N.m for a windfoil, against ~1.2 N.m of section couple
    on this foil), it decides the sign of that load, and the sign decides
    which way up the section is screened and drawn.

    ``hydrotail.evaluate_hydrofoil_tail`` routes it through ``cm_ac_model``,
    which both of ITS authors read. This is the same statement one level up:
    without it, a card would quote a lifting stabiliser before the run and
    the run would fly a download — the defect "the tail's download has two
    authors" recorded in this repo, with a third author added.

    Returns ``(0.0, "")`` for every family that has no rig, which is all of
    them by default and all of the air ones by construction, so nothing that
    reads this changes until a rig is stated.

    The lever needs a DEPTH, and depth is a design variable here, so the box
    mid-point is used — the same thing the area and the arm above come from.
    It is therefore a pre-run READING of a quantity the run will vary, which
    is what a pre-run read-out is; the note says so.

    WHICH box's mid-point is the whole of finding 5, and the caller passes it
    in as ``depth_m`` rather than this function reading it: ``depth_m`` is a
    design row a shell may widen or narrow (``_water_band_kwargs`` carries it
    into ``DEPTH_BOUNDS`` on the built problem), and a lever closed on the
    PUBLISHED band's 0.575 m while the run flies a 0.30 m mast is off by
    nearly a factor of two — enough, on a driven craft where this couple is
    several hundred times the sections', to quote the opposite SIGN for the
    stabiliser's load. ``None`` falls back to ``box``, which is what every
    caller that has no design box to offer still gets.
    """
    rig = getattr(prob, "rig", None)
    if rig is None:
        return 0.0, ""
    row = box.get("depth_m")
    depth = getattr(prob, "depth", None)
    if depth is None and depth_m is not None:
        depth = float(depth_m)
    if depth is None and row is not None:
        depth = 0.5 * (float(row[0]) + float(row[1]))
    q = 0.5 * float(getattr(prob, "rho", 0.0)) * float(v) ** 2
    if depth is None or not (q > 0.0):
        return 0.0, ""
    moment = rig.pitching_moment_nm(weight_n=float(prob.L_design),
                                    depth_m=float(depth), x_cg_m=float(x_cg))
    return float(moment / q), (
        f"; + rig couple {moment:.4g} N·m (drive "
        f"{rig.drive_n(float(prob.L_design)):.4g} N x lever "
        f"{rig.lever_m(float(depth)):.4g} m at the box's mid depth)")


#: The optimisers that cannot run at all without PyTorch, measured rather
#: than assumed: ``bo`` fits a GP, ``bo_slsqp`` starts as ``bo``, ``blocks``
#: calls ``run_bo_constrained`` per block, and ``adjoint`` differentiates a
#: torch twin. Every other name here is pymoo or scipy and runs without it.
TORCH_ONLY_OPTIMISERS = ("bo", "bo_slsqp", "blocks", "adjoint")


def torch_optimiser_note() -> str:
    """Why the strategy list is short on this machine, or "" if it is not.

    :func:`compatible_optimisers` drops what cannot run, and this is the
    sentence that says so. The two go together on purpose: a list that
    silently got shorter is the kind of thing nobody notices, and a user who
    came for Bayesian optimisation is owed the reason it is not on offer
    rather than a traceback thirty seconds into a search.
    """
    from .optimize.torch_optional import HAVE_TORCH, REASON

    if HAVE_TORCH:
        return ""
    return ("Bayesian optimisation, the BO→SLSQP handoff, the per-block "
            "portfolio and the adjoint solver are missing from this list. "
            + REASON)


def compatible_optimisers(problem_name: str) -> list[str]:
    """Optimiser names valid for a problem given its constrained-ness.

    Also drops what this MACHINE cannot run — see
    :data:`TORCH_ONLY_OPTIMISERS` and :func:`torch_optimiser_note`.
    """
    spec = PROBLEM_SPECS[problem_name]
    key = "supports_constrained" if spec.is_constrained else "supports_unconstrained"
    names = [n for n, o in OPTIMISER_SPECS.items() if getattr(o, key)]
    if not spec.is_constrained and problem_name != "trim wing":
        names = [n for n in names if n != "adjoint"]   # torch trim-vector only
    if not spec.has_blocks:
        names = [n for n in names if n != "blocks"]    # needs declared blocks
    from .optimize.torch_optional import HAVE_TORCH
    if not HAVE_TORCH:                                # no wheel for this Mac
        names = [n for n in names if n not in TORCH_ONLY_OPTIMISERS]
    return names


# =====================================================================
# Run config / result
# =====================================================================

@dataclass
class RunConfig:
    """Everything needed to launch one optimisation — the GUI's Run form."""

    problem_name: str
    mission_kwargs: dict = field(default_factory=dict)
    flags: dict = field(default_factory=dict)
    optimiser: str = "bo"
    budget: int = 32
    seed: int = 0
    bounds_overrides: dict | None = None
    #: ``{param label: value}`` the optimiser may NOT change. The variable
    #: leaves the design vector entirely (see :class:`_Pin`) — it is not a
    #: box of width zero, which the samplers cannot draw from. None/{} is the
    #: bit-for-bit legacy path.
    pinned: dict | None = None
    #: A DESIGN, in FULL design-vector coordinates, evaluated FIRST — the one
    #: mechanism that makes "opening a variable cannot lose" exact rather than
    #: probable. The incumbent is a max over evaluated points with all margins
    #: >= 0, and a seed is never screened out, so a run told a design that
    #: FLIES cannot return "no solution", and cannot report a score below it.
    #: Everything else here (a screened initial design, a feasibility phase)
    #: raises the ODDS of finding something; only this settles it.
    #:
    #: The Sobol draw shrinks by one so the budget is unchanged. On a pinned
    #: run the seed is given in full coordinates and must AGREE with the pins —
    #: a seed that disagrees is a different design, and it is refused rather
    #: than projected. None (default) is the bit-for-bit legacy path.
    x_seed: list | None = None

    def to_dict(self) -> dict:
        return _json_safe({
            "problem_name": self.problem_name,
            "mission_kwargs": self.mission_kwargs,
            "flags": self.flags,
            "optimiser": self.optimiser,
            "budget": int(self.budget),
            "seed": int(self.seed),
            "bounds_overrides": self.bounds_overrides,
            "pinned": self.pinned,
            "x_seed": self.x_seed,
        })


@dataclass
class RunResult:
    """JSON-serialisable outcome of one :func:`run` — the shell's result page."""

    problem_name: str
    optimiser: str
    budget: int
    seed: int
    medium: str
    is_constrained: bool
    param_labels: list
    dim: int
    bounds: list                # (d, 2) effective bounds as nested lists
    best_x: list | None
    best_score: float | None
    feasible: bool
    n_feasible: int | None
    best_g: float | None
    history: list               # best-so-far per eval (None where -inf)
    n_evals: int
    eval_x: list                # (n_evals, d) evaluated points
    eval_y: list                # per-eval objective values
    eval_g: list | None         # per-eval constraint margins (constrained)
    breakdown: dict | None      # sanitised evaluate() at best_x
    wall_time_s: float
    timestamp: str
    config: dict                # echoed RunConfig
    acqf: str | None = None
    failures: list | None = None
    path: str | None = None
    bo_iters: list | None = None    # per-BO-iteration GP diagnostics (bo only)
    partial: bool = False           # True = run stopped early (cancelled);
    #                                 the log is what was paid for, not a
    #                                 completed search — never compare a
    #                                 partial run's best against a full one
    stop_reason: str | None = None  # why a partial run ended, when it was a
    #                                 RULE that ended it (``run(stop_rule=…)``)
    #                                 rather than the user
    pinned: dict | None = None      # {label: value} held fixed. Those rows
    #                                 come back collapsed to [v, v] in
    #                                 ``bounds``; ``param_labels``, ``dim``,
    #                                 ``best_x`` and ``eval_x`` stay the FULL
    #                                 design vector, so nothing downstream has
    #                                 to know a pin happened
    #: ``bo_slsqp`` ONLY, None everywhere else: what the two phases actually
    #: did — ``{phi, n_a, n_b, bo_split, handed_over, a_best}``. ``handed_over``
    #: is the load-bearing one. False means the BO phase found no feasible
    #: design, so the finisher started from a random multistart instead of from
    #: BO's incumbent: a BO head with a pure-SLSQP tail, which is NOT the arm
    #: ``RESULTS_HANDOFF.md`` measured. A caller comparing this run against that
    #: study has to be able to see that, and inferring it from a good-looking
    #: score is exactly what it could never do.
    handoff: dict | None = None
    searched_dim: int | None = None  # how many dimensions the optimiser
    #                                  actually searched (dim minus the pins).
    #                                  None = no pin, i.e. dim
    front: dict | None = None       # objective="pareto" ONLY. The whole
    #                                 `pareto_airfoil` report — conditions,
    #                                 seed, the ranked non-dominated rows and
    #                                 the hypervolume block — stored WHOLE, so
    #                                 that anything already able to read a
    #                                 front report (gui.v3.stages.airfoil
    #                                 .front_rows, every test in
    #                                 test_pareto_front.py) reads this field
    #                                 unchanged. On a front run `best_x` is the
    #                                 front point with the highest PLAIN
    #                                 composite J and `best_score` is that J:
    #                                 one answer for a caller that needs one,
    #                                 chosen in the common currency and not by
    #                                 a rule invented here. `eval_y` holds the
    #                                 per-evaluation objective VECTORS, not
    #                                 scalars, and `history` is the
    #                                 hypervolume trace — see `run`.
    #: how many draws the screened initial design DISCARDED as refusals, and
    #: how many iterations the feasibility phase spent, on a run that asked for
    #: them (``bo_feasibility``); None on every legacy run. Reported for the
    #: reason every other measured-but-invisible number here is: a run whose
    #: search spent its draws differently than the flat Sobol prefix must be
    #: able to say so, and "92 % of your design box is refused before it is
    #: solved" is the most useful sentence such a run can give a user.
    n_screened: int | None = None
    n_rescue: int | None = None
    #: ...and how many of those feasibility iterations the surrogate could NOT
    #: steer, because every observed margin was the same flat refusal sentinel
    #: and there was nothing to climb. A phase that is 100 % blind is a
    #: uniform search wearing the phase's name — which is what a flat refusal
    #: margin makes it, and what ``bo_feasibility="guide"`` exists to change.
    n_rescue_blind: int | None = None
    #: rows whose SEARCHED band reached outside the band the family validates
    #: over, so every draw in the excess came back refused
    #: (:func:`rows_outside_validity`). Empty on every run that widened
    #: nothing, which is every published study.
    box_outside_validity: list | None = None
    #: ``{hits, misses, dir}`` when the run was given an evaluation memo
    #: (:mod:`aerobo.eval_cache`), None on every run that was not — which is
    #: every study in this repo. It is here because it is the one thing that
    #: makes ``wall_time_s`` readable: 40 hits and 20 misses is a search that
    #: flew 20 designs and took 40 back off disk, and a wall clock read next
    #: to that number is not a measurement of the search.
    eval_cache: dict | None = None
    #: how many of this run's evaluations came from the run it CONTINUES
    #: (``run(resume=…)``) rather than from its own physics. None on every
    #: run that flew all of its own. It is here because ``n_evals`` counts the
    #: whole training set and ``wall_time_s`` only the new part, and a reader
    #: who cannot tell the two apart would price a continuation as a search
    #: that got 59 evaluations for the cost of 8.
    resumed: int | None = None
    bo_split: list | None = None    # [n_init, n_iter] the BO run ACTUALLY
    #                                 used, after _bo_split's clamps. None on
    #                                 every non-BO optimiser. It is here for
    #                                 the same reason searched_dim is: the
    #                                 echoed ``config`` states what was ASKED
    #                                 for, and an over-budget bo_n_init is
    #                                 clamped — so without this field the
    #                                 result reported a seed size the run did
    #                                 not fly. Re-derived from the same
    #                                 (budget, dim, flag) the dispatch used,
    #                                 so it cannot disagree with it.
    #: WHAT ``best_score`` IS, in the family's own words — ``"L/D"``,
    #: ``"payload L/D"``, ``"composite J (0-100)"``, or a car
    #: family's ``objective_label``. ``None`` means the family never
    #: said, and ``check_comparable`` refuses such a run from a
    #: shared axis rather than guessing. Defaulted so every stored
    #: result predating the field still loads;
    #: ``record_score_units`` recovers it from the breakdown.
    score_units: str | None = None

    def to_dict(self) -> dict:
        return _json_safe({
            "problem_name": self.problem_name,
            "optimiser": self.optimiser,
            "budget": self.budget,
            "seed": self.seed,
            "medium": self.medium,
            "is_constrained": self.is_constrained,
            "param_labels": list(self.param_labels),
            "dim": self.dim,
            "bounds": self.bounds,
            "best_x": self.best_x,
            "best_score": self.best_score,
            "feasible": self.feasible,
            "n_feasible": self.n_feasible,
            "best_g": self.best_g,
            "history": self.history,
            "n_evals": self.n_evals,
            "eval_x": self.eval_x,
            "eval_y": self.eval_y,
            "eval_g": self.eval_g,
            "breakdown": self.breakdown,
            "wall_time_s": self.wall_time_s,
            "timestamp": self.timestamp,
            "config": self.config,
            "acqf": self.acqf,
            "failures": self.failures,
            "path": self.path,
            "bo_iters": self.bo_iters,
            "partial": self.partial,
            "stop_reason": self.stop_reason,
            "pinned": self.pinned,
            "searched_dim": self.searched_dim,
            "n_screened": self.n_screened,
            "n_rescue": self.n_rescue,
            "n_rescue_blind": self.n_rescue_blind,
            "box_outside_validity": self.box_outside_validity,
            "eval_cache": self.eval_cache,
            "resumed": self.resumed,
            "bo_split": self.bo_split,
            "score_units": self.score_units,
            # None on every arm but `bo_slsqp`, so the stored shape of every
            # existing result is unchanged. It must survive the round trip for
            # the reason the field's own comment gives: `handed_over=False` is
            # the only witness that the finisher was never seeded, and a stored
            # run that dropped it could not be told apart afterwards from the
            # arm the study measured.
            "handoff": self.handoff,
            # None on every scalar run, so the stored shape of a -cd or a
            # composite result is unchanged. On a front run this is the whole
            # front, and it must survive the round trip: a result page that
            # dropped it would show a single answer for a search whose ANSWER
            # IS THE SET.
            "front": self.front,
        })


# =====================================================================
# Progress-counting callable wrapper
# =====================================================================

def _cb_accepts_kwargs(cb: Callable) -> bool:
    """True iff ``cb`` declares a ``**kwargs`` catch-all (rich-payload opt-in).

    Legacy callbacks keep the exact two-positional ``cb(i, best)`` call; a
    callback that declares ``**kwargs`` additionally receives the per-eval
    detail (``f=``, ``g=``, ``x=``, ``feasible=``) the live GUI plots.
    """
    try:
        sig = inspect.signature(cb)
    except (TypeError, ValueError):
        return False
    return any(p.kind is inspect.Parameter.VAR_KEYWORD
               for p in sig.parameters.values())


#: the failure contract's value, as :mod:`optimize.feasible` states it. Named
#: here so the progress counter can recognise a refused screening draw without
#: importing the physics layer's ``objective.PENALTY`` at module scope.
_FEASIBLE_PENALTY = -100.0

#: the feasibility mode that means "legacy", restated for the same reason
_F_OFF = "off"

#: THE NAMES OF THE MAXIMISED SCALAR, restated rather than imported: the
#: module docstring promises that ``import aerobo.api`` costs no physics
#: import, and ``sizing`` is physics. Kept equal to ``sizing`` by a test, the
#: same contract ``_FEASIBLE_PENALTY`` already lives under.
LOD_UNITS = "L/D"
PAYLOAD_LOD_UNITS = "payload L/D"


def score_units(breakdown: dict | None) -> str | None:
    """What a breakdown's maximised scalar IS, in the family's own words.

    ``None`` means the family never said, and no caller may assume. It is
    deliberately not a default of ``"L/D"``: the whole failure this closes is
    a score of W_fixed/D being read as an aero L/D, and a default that
    guessed the common case would reopen it for precisely the families that
    get it wrong.

    Precedence, most specific first:

    * ``score_units`` — the family's own declaration, written by the same
      statement that writes ``score`` (``sizing.SizedState.report``,
      ``wing_score.composite_evaluation``, ``airfoil.evaluate_airfoil``);
    * ``objective_label`` — the car families' existing idiom, already stored
      beside ``objective``;
    * ``score == LoD`` to the bit — a LEGACY record with neither, recovered
      by an OUTCOME and not by a guess. Nothing is inferred from the
      problem's NAME: the name is exactly what cannot see the ``size``
      modifier's effect on the units, because the modifier is spelled in the
      name and nowhere in the number.
    """
    bd = breakdown if isinstance(breakdown, dict) else {}
    declared = bd.get("score_units") or bd.get("objective_label")
    if declared:
        return str(declared)
    sc, lod = bd.get("score"), bd.get("LoD")
    if (isinstance(sc, (int, float)) and not isinstance(sc, bool)
            and isinstance(lod, (int, float)) and not isinstance(lod, bool)
            and math.isfinite(float(sc)) and float(sc) == float(lod)):
        return LOD_UNITS
    return None


def record_score_units(record) -> str | None:
    """``score_units`` of a STORED run.

    The field first; the breakdown second, so a result archived before the
    field existed still resolves and no reader branches on the file's age.
    """
    rd = record.to_dict() if hasattr(record, "to_dict") else (record or {})
    if not isinstance(rd, dict):
        return None
    return rd.get("score_units") or score_units(rd.get("breakdown"))


def check_comparable(records, where: str = "compare") -> str | None:
    """The ONE unit all these runs' scores are in — or raise ``ValueError``.

    A single run is always comparable with itself and never raises: the rule
    is about CO-plotting, and one history on one axis states nothing it
    cannot support.

    Two or more must agree, and ``None`` never agrees with anything —
    including another ``None``. Two runs that both decline to say what their
    score is are exactly the pair that could be a lap time and an L/D.
    """
    rds = [r for r in (records or [])]
    if len(rds) < 2:
        return record_score_units(rds[0]) if rds else None
    seen: dict = {}
    for rd in rds:
        d = rd.to_dict() if hasattr(rd, "to_dict") else (rd or {})
        who = (f"{(d or {}).get('problem_name', '?')}"
               f"@{(d or {}).get('timestamp', '?')}")
        seen.setdefault(record_score_units(rd), []).append(who)
    if len(seen) == 1 and None not in seen:
        return next(iter(seen))
    if None in seen:
        raise ValueError(
            f"{where}: {', '.join(seen[None])} do not say what their score "
            f"IS (no score_units on the record, and score != LoD in the "
            f"breakdown), so nothing may share an axis with them")
    raise ValueError(
        f"{where}: these runs maximise different quantities and must not "
        f"share one axis — "
        + "; ".join(f"{u} [{', '.join(v)}]" for u, v in sorted(seen.items())))


class _ProgressCounter:
    """Wrap the objective/fg callable to drive a live best-so-far callback.

    Does NOT touch the optimisers — it simply counts each (unique) evaluation
    that reaches the physics and reports the running best. The constrained
    runners memoise by x, so this fires once per solved point; the
    unconstrained ones call straight through, so it fires once per evaluation.

    Callback protocol: ``cb(i, best)`` exactly as before, UNLESS the callback
    declares ``**kwargs`` — then it also receives ``f`` (this eval's value),
    ``g`` (margin list, constrained only), ``feasible`` and ``x``.

    ``expand`` (a pinned run's :meth:`_Pin.expand`) is applied to ``x`` BEFORE
    it is reported, so every listener — the live design view, the
    interrupted-run log :func:`partial_result` rebuilds from — receives the
    full design vector whether or not a variable was held fixed.
    """

    def __init__(self, fn: Callable, is_constrained: bool,
                 progress_cb: Callable | None, expand: Callable | None = None,
                 grader=None, start: int = 0, best: float = -math.inf):
        self.fn = fn
        self.is_constrained = is_constrained
        self.cb = progress_cb
        self.rich = progress_cb is not None and _cb_accepts_kwargs(progress_cb)
        self.expand = expand
        #: :class:`optimize.feasible.SizeGateMargin` when the run asked for
        #: graded refusals (``bo_feasibility="guide"``), else None — the
        #: legacy flat sentinel, bit-for-bit
        self.grader = grader
        #: WHERE THE COUNT STARTS. A RESUMED run (``run(resume=…)``) is handed
        #: the previous run's evaluations as its training set and flies only
        #: the new ones, so its first evaluation is number ``start + 1`` and
        #: its best-so-far opens at the incumbent it inherited. Both are here
        #: rather than in the shell because every listener — the live plot,
        #: the stop rule, ``partial_result``'s log — reads this counter, and a
        #: continuation whose progress bar restarted at 1 is the report this
        #: exists to answer.
        self.n = int(start)
        self.best = float(best)
        #: while True, a REFUSED draw is neither counted nor reported. Set by
        #: :mod:`optimize.feasible` around a screened initial design: a design
        #: the physics refuses before its solver runs did not cost an
        #: evaluation, and counting one would run the progress bar 32x fast
        #: and hand the stop rule a budget nothing spent. A draw that SOLVES
        #: is kept and counted exactly once, so the accounting is unchanged.
        self.screening = False

    def __call__(self, x):
        r = self.fn(x)
        if self.is_constrained and self.grader is not None:
            # a REFUSED design reports how far outside the gate it is, so the
            # feasibility phase has a surface to climb instead of a constant.
            # Applied HERE because this is the one point every optimiser's
            # every evaluation passes through, and because it must reach the
            # RECORD as well as the runner: a graded margin the log did not
            # keep would leave `min_violation_index` ranking on the flat -1
            # the run itself no longer used
            f_seen, g_seen = r
            if float(f_seen) == _FEASIBLE_PENALTY:
                x_full = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
                if self.expand is not None:
                    x_full = np.atleast_1d(self.expand(x_full))
                r = (f_seen, self.grader.graded(x_full, g_seen))
        if self.screening:
            f_seen = float(r[0] if self.is_constrained else r)
            if f_seen == _FEASIBLE_PENALTY:
                return r
        self.n += 1
        g_list = None
        if self.is_constrained:
            f, g = r
            g_arr = np.atleast_1d(np.asarray(g, dtype=float))
            g_list = [float(v) for v in g_arr]
            feas = bool(np.all(g_arr >= 0.0))
            if feas and float(f) > self.best:
                self.best = float(f)
        else:
            f = float(r)
            feas = True
            if f > self.best:
                self.best = f
        if self.cb is not None:
            if self.rich:
                x_rep = np.atleast_1d(np.asarray(x, dtype=float))
                if self.expand is not None:
                    x_rep = np.atleast_1d(self.expand(x_rep.ravel()))
                self.cb(self.n, self.best, f=float(f), g=g_list,
                        feasible=feas,
                        x=[float(v) for v in x_rep])
            else:
                self.cb(self.n, self.best)
        return r


# =====================================================================
# Budget split + dispatch
# =====================================================================

def _bo_split(budget: int, dim: int, n_init: int | None = None
              ) -> tuple[int, int]:
    """(n_init, n_iter) for a BO budget: Sobol init ~2d capped at 16, >=1 iter.

    Mirrors the harness sizing intent (hydrofoil_bo / airfoil_bo) while staying
    valid for tiny GUI budgets.

    ``n_init`` (the ``bo_n_init`` flag, :data:`BO_N_INIT_FLAG`) states the
    Sobol seed size explicitly instead of taking the 2d-capped-at-16 default.
    It is what the budget study varies — the split is a SEARCH decision, not
    a property of the problem — and what a measured recommendation sends
    (:mod:`aerobo.optimize.budget`). The clamps are identical either way, so
    an unstated split is bit-for-bit the legacy path.
    """
    if n_init is None:
        n_init = min(max(4, 2 * dim), 16)
    n_init = max(1, int(n_init))
    n_init = max(1, min(n_init, budget - 1))
    n_iter = budget - n_init
    if n_iter < 1:                    # pathological tiny budget
        n_init, n_iter = max(1, budget - 1), 1
    return n_init, n_iter


BO_WARM_START_FLAG = "bo_warm_start"

#: explicit Sobol seed size for a BO run (see :func:`_bo_split`). A search
#: flag, not a physics one: no problem builder reads it.
BO_N_INIT_FLAG = "bo_n_init"


def _bo_n_init(cfg: RunConfig) -> int | None:
    """The stated Sobol seed size, or None for the 2d-capped-at-16 default."""
    raw = (cfg.flags or {}).get(BO_N_INIT_FLAG)
    if raw is None:
        return None
    n = int(raw)
    if n < 1:
        raise ValueError(f"flags[{BO_N_INIT_FLAG!r}] must be >= 1 (got {raw})")
    return n


#: how a REFUSED design (the -100 failure sentinel) is shown to the BO
#: surrogate — see :mod:`aerobo.optimize.refusal`. A search flag, not a
#: physics one: the objective still RETURNS -100, the recorded history still
#: CARRIES -100, and no problem builder reads this. Unstated (the default)
#: is bit-for-bit the legacy path.
BO_REFUSAL_FLAG = "bo_refusal"


def _bo_refusal(cfg: RunConfig) -> str:
    """The stated refusal-imputation mode, validated ('sentinel' by default)."""
    from .optimize.refusal import check_mode
    return check_mode((cfg.flags or {}).get(BO_REFUSAL_FLAG))


#: what to do about a run that has not found a FEASIBLE design — see
#: :mod:`aerobo.optimize.feasible`. A search flag like :data:`BO_REFUSAL_FLAG`
#: and for the same reason: the physics is untouched (the same gates refuse the
#: same designs, the history records what it always did), only WHERE the search
#: spends its draws changes. Unstated is bit-for-bit the legacy path; the V3
#: shell asks for ``"rescue"``.
#:
#: WHY IT IS NOT ON BY DEFAULT HERE. Every frozen study in this repo is a
#: stored comparison between search methods, and a mechanism that changed what
#: a run evaluates would move those numbers underneath their own result files.
#: So the default is legacy and the shell — which is where a user meets a run
#: that found nothing — opts in, exactly as it already opts into
#: ``bo_refusal="worst"`` over this module's ``"sentinel"``.
BO_FEASIBILITY_FLAG = "bo_feasibility"

#: the optimisers that READ a feasibility mode. Every other runner is a study
#: control and was deliberately left on the legacy path; naming them here is
#: what lets a run refuse the flag instead of accepting it and doing nothing.
#:
#: ``bo_slsqp`` is here because its FIRST PHASE is a real
#: ``run_bo_constrained`` call and reads the mode exactly as a plain BO run
#: does. Its SLSQP tail does not, and does not need to: the mode governs how
#: an initial design is spent hunting the feasible set, and the tail has no
#: initial design — it starts at the design the BO phase handed it.
_FEASIBILITY_OPTIMISERS = ("bo", "blocks", "bo_slsqp")


def _bo_feasibility(cfg: RunConfig) -> str:
    """The stated feasibility mode, validated ('off' by default)."""
    from .optimize.feasible import check_mode
    return check_mode((cfg.flags or {}).get(BO_FEASIBILITY_FLAG))


#: how many box draws decide whether a live size gate ever fires. Cheap —
#: ``excess`` is arithmetic on the design vector, no solver — so this is a
#: few hundred microseconds against a run that costs seconds per evaluation.
_GATE_PROBE_DRAWS = 256


def _gate_is_flat(grader, built) -> bool:
    """Does this grader fire NOWHERE in this problem's box?

    Structural liveness ("the rows are here") and useful liveness ("some
    design in the box trips the gate") are different questions, and a run
    that asked to be guided deserves to be told when the answer to the
    second is no.
    """
    try:
        box = np.asarray(built.bounds, dtype=float)
        rng = np.random.default_rng(0)
        u = rng.random((_GATE_PROBE_DRAWS, box.shape[0]))
        pts = box[:, 0] + u * (box[:, 1] - box[:, 0])
        return not any(grader.excess(x) > 0.0 for x in pts)
    except Exception:                         # pragma: no cover — a probe
        return False


def _design_ws_pa(prob):
    """The W/S a derived-area family flies — ``objective``'s own definition,
    not a second one, so the cheap gate cannot bound an area the solver did
    not fly."""
    try:
        from .objective import design_wing_loading_pa
    except ImportError:                       # pragma: no cover
        return getattr(prob, "wing_loading_Pa", None)
    return design_wing_loading_pa(prob)


def _gate_grader(cfg: RunConfig, built):
    """The graded-refusal margin for this run, or None for the flat sentinel.

    Built from what the problem ALREADY states — its design-box labels, the
    solvers' aspect-ratio band, and the mission ceiling it was given — so no
    family has to learn a new failure contract to be guided. Returns None
    unless the run asked for ``bo_feasibility="guide"`` AND the problem states
    something measurable, which keeps every other run bit-for-bit.
    """
    from .optimize.feasible import SizeGateMargin, grades

    if not built.is_constrained or not grades(_bo_feasibility(cfg)):
        return None
    if cfg.optimiser not in _FEASIBILITY_OPTIMISERS:
        # the flag is named for the runners that read it, and this is the one
        # part of it that would otherwise reach EVERY optimiser through the
        # shared wrapper: a GA's feasibility-first tournament ranks infeasible
        # members by violation, so grading would quietly change what
        # ``bo_feasibility`` means on a runner that never asked
        raise ValueError(
            f"optimiser {cfg.optimiser!r} does not read "
            f"{BO_FEASIBILITY_FLAG}={_bo_feasibility(cfg)!r}: the feasibility "
            f"modes are the BO loops' (and the per-block sub-searches'). Use "
            f"optimiser='bo', or drop the flag rather than have it silently "
            f"do nothing")
    from . import sizing
    prob = built.problem
    weight = getattr(prob, "W_fixed_N", None)
    if weight is None:
        try:
            from .mission import design_weight_n
            # STATED first: the gate must bound the aircraft the solver
            # flies, and at altitude the derived weight is not that one
            weight = design_weight_n(prob)
        except (AttributeError, TypeError, ValueError):
            weight = None
    grader = SizeGateMargin(
        list(built.param_labels), ar_limits=sizing.AR_LIMITS,
        cap_pa=getattr(prob, "wing_loading_max_Pa", None), weight_n=weight,
        # the DERIVED-area families have no S_m2 row; their loading is the
        # mission's, and with it the area — and so the aspect ratio — can be
        # bounded from the design vector alone
        ws_fixed_pa=_design_ws_pa(prob))
    # NOT SILENT, and MEASURED rather than assumed. "live" is a statement
    # about the rows a problem carries; whether the gate ever actually fires
    # is a statement about the BOX, and the two come apart. The derived-area
    # families are the case that proves it: with the mission's W/S the aspect
    # ratio is bounded by b^2 * (W/S) / W_fixed, which on a 6-40 m span band
    # cannot reach under the lower AR limit at all — live, sound, and inert.
    # A grader that is live but never fires is worse than a dead one, because
    # it silences the warning while grading nothing.
    if grader.live and _gate_is_flat(grader, built):
        warnings.warn(
            f"{BO_FEASIBILITY_FLAG}={_bo_feasibility(cfg)!r}: the size gate "
            f"is live on {cfg.problem_name!r} but does not fire anywhere in "
            f"this box, so every refusal will still be the flat sentinel. "
            f"The guiding is costing nothing and buying nothing.",
            RuntimeWarning, stacklevel=2)
    if not grader.live:
        # NOT SILENT. The run asked to be guided and cannot be: a flat
        # sentinel is what it will get, and the one thing worse than that is
        # not knowing. (Measured: 444 registered searched-W/S variants
        # graded nothing at all, and said nothing about it.)
        warnings.warn(
            f"{BO_FEASIBILITY_FLAG}={_bo_feasibility(cfg)!r} was asked for, "
            f"but {cfg.problem_name!r} states nothing this grader can "
            f"measure (rows "
            f"{[r for r in ('b_m', 'S_m2', 'ws_pa') if r in built.param_labels]}"
            f", W_fixed={weight}, W/S={getattr(prob, 'wing_loading_Pa', None)}"
            f", ceiling={getattr(prob, 'wing_loading_max_Pa', None)}). "
            f"Refusals will be the flat sentinel, as with feasibility='off'.",
            RuntimeWarning, stacklevel=2)
        return None
    return grader


#: the fraction of a ``bo_slsqp`` budget spent on the BO phase before the
#: SLSQP finisher takes over (see :mod:`aerobo.optimize.handoff`). A search
#: flag, not a physics one: no problem builder reads it. Unstated is
#: :data:`optimize.handoff.DEFAULT_PHI` = 0.25, the split with the best median
#: gap of the three ``RESULTS_HANDOFF.md`` measured.
HANDOFF_PHI_FLAG = "handoff_phi"


def _handoff_phi(cfg: RunConfig) -> float:
    """The stated BO share, or the measured default.

    Validated HERE, before a single evaluation is spent, for the reason
    :func:`optimize.bo.run_bo` gives about a bad acquisition name: a phi
    outside (0, 1) is a caller error that cannot become valid later, and a run
    that pays for its BO phase before refusing has charged for a mistake it
    could have caught for free.
    """
    from .optimize.handoff import DEFAULT_PHI
    raw = (cfg.flags or {}).get(HANDOFF_PHI_FLAG)
    if raw is None:
        return DEFAULT_PHI
    phi = float(raw)
    if not 0.0 < phi < 1.0:
        raise ValueError(
            f"flags[{HANDOFF_PHI_FLAG!r}] must be strictly between 0 and 1 "
            f"(got {raw!r}); phi=0 is pure 'slsqp' and phi=1 is pure 'bo', "
            f"and both are optimisers you can name directly")
    return phi


#: flags a run honours WHATEVER problem it names, because they are not
#: physics: they are the search policy V3.5 stage 1 owns. ``ProblemSpec.flags``
#: describes the physics a family honours and drives the shell's toggles, so
#: these are legitimately absent from every spec — and a validator that knew
#: only ``spec.flags`` would refuse the whole repo's own study scripts.
#:
#: ``block_optimisers`` / ``block_cycles`` are here for the same reason and
#: not a different one: they are options OF the ``blocks`` optimiser, read
#: straight off ``cfg.flags`` in ``_dispatch``, and no builder sees them.
RUN_POLICY_FLAG_KEYS = ("acqf", BO_WARM_START_FLAG, BO_N_INIT_FLAG,
                        BO_REFUSAL_FLAG, BO_FEASIBILITY_FLAG,
                        HANDOFF_PHI_FLAG,
                        "block_optimisers", "block_cycles")


#: The acquisition function is a CHOICE ONLY ON THE UNCONSTRAINED PATH.
#: ``optimize/constrained.py`` hardwires LogConstrainedExpectedImprovement and
#: takes no ``acqf`` argument at all, so on a constrained problem the key was
#: accepted by :func:`check_flags` and then read by nothing: ``logei``,
#: ``ucb``, ``qmes`` and the literal ``"nonsense"`` all produced a
#: bit-identical evaluation sequence. An arm labelled "bo-ucb (constrained)"
#: was therefore the LogCEI control wearing a false name — exactly the
#: silently-dropped-flag failure this module exists to refuse, sitting inside
#: the refusal itself.
ACQF_FLAG_KEY = "acqf"


def accepted_flags(problem_name: str) -> frozenset:
    """Every ``flags`` key this problem honours: physics + builder + policy.

    ``acqf`` is withheld from CONSTRAINED problems: see :data:`ACQF_FLAG_KEY`.
    Widening this to plumb an acquisition through the constrained path is a
    design decision, not a bug fix, and is deliberately not taken here.
    """
    spec = PROBLEM_SPECS[problem_name]
    policy = tuple(k for k in RUN_POLICY_FLAG_KEYS
                   if not (k == ACQF_FLAG_KEY and spec.is_constrained))
    return frozenset(tuple(spec.flags or ())
                     + tuple(spec.builder_flags or ())
                     + policy)


def check_flags(problem_name: str, flags: dict | None,
                where: str = "flags") -> None:
    """Refuse a flag this problem does not honour. Raises :class:`KeyError`.

    WHY THIS EXISTS. Until session 49 ``api`` accepted any key at all and the
    builders read the ones they knew about, so asking for — say — a 0.20 m
    ``draught_max_m`` on a family with no draught cap produced an UNCAPPED run
    and no signal whatever. That is the repo's own rule ("a flag that is
    silently dropped is worse than one that is refused") going unenforced at
    the very surface that states it, and it is general: the same hole swallows
    a mistyped ``junction_drag``, a ``mach`` on a family that has no
    compressibility term, and a section flag on a family whose section is
    selected by thickness.

    It is a KeyError and not a warning because the failure it prevents is
    silent-and-wrong rather than noisy-and-late: the run completes, reports a
    score, and stores a config that says the flag was set. Nothing downstream
    can tell that apart from a run that honoured it.

    The three sources of legitimate keys are kept apart on purpose:
    ``spec.flags`` (physics the shell offers), ``spec.builder_flags`` (a
    builder contract deliberately not shown as a toggle) and
    :data:`RUN_POLICY_FLAG_KEYS` (search policy, which is not a property of
    the problem at all). Widening the first changes the GUI; widening the
    second does not — so a flag that is honoured but should not appear as a
    toggle has a home that is not "accept everything".
    """
    # the NAME first, and before the empty-flags shortcut: a checker that
    # accepts an unknown problem when the flag dict happens to be empty and
    # refuses it when it is not would answer the same question two ways
    if problem_name not in PROBLEM_SPECS:
        raise KeyError(f"unknown problem {problem_name!r}")
    if not flags:
        return
    ok = accepted_flags(problem_name)
    unknown = sorted(k for k in flags if k not in ok)
    if not unknown:
        return
    raise KeyError(
        f"{where}: {problem_name!r} does not honour {unknown} — it would be "
        f"accepted and silently ignored. Honoured here: "
        f"{sorted(ok)}")


def unhonoured_flags(problem_name: str, flags: dict | None) -> list:
    """Keys this problem would ignore — the same test :func:`check_flags`
    raises on, as data."""
    if not flags:
        return []
    ok = accepted_flags(problem_name)
    return sorted(k for k in flags if k not in ok)


def sanitise_flags(problem_name: str, flags: dict | None) -> tuple[dict, list]:
    """``(flags this problem honours, the keys dropped)``.

    THE SPLIT, and why it is not a loophole in :func:`check_flags`.

    A **search** must not proceed on a specification it has misunderstood, so
    :func:`run` refuses. But a **report on a design that has already been
    searched** must not become unreadable because the config it was stored
    with carries a key the family never honoured: refusing there protects
    nobody (the run happened long ago) and costs the user their archive. So
    the read-only paths sanitise instead — and *say so*, by carrying the
    dropped keys out on ``unhonoured_flags`` where a card can render them.
    Dropped-and-reported is not the failure this module is about; the failure
    is dropped-and-silent.

    A SHELL should call this before building a config at all, so its Run
    button never offers a control the chosen family cannot honour. Both
    shells do (``gui/nice_app.cfg_dict_from_state``, ``gui/v3/config.flags``);
    without it, four of the configs stored under ``results/gui_runs/`` — a
    winglet type on a family that selects its own, a tail card left on while a
    hydrofoil was selected — would have started raising at the Run button.
    Found by the pre-commit review, not by a test.
    """
    dropped = unhonoured_flags(problem_name, flags)
    if not dropped:
        return (dict(flags or {}), [])
    return ({k: v for k, v in (flags or {}).items() if k not in dropped},
            dropped)


#: The one problem whose search class is the SECTION's: 8+ CST weights with
#: a live viscous XFOIL polar behind every candidate. Everything else in the
#: registry — including the families that carry a designed section — is a
#: planform search and is measured as one (:mod:`aerobo.optimize.budget`).
AIRFOIL_PROBLEM = "airfoil (section)"


def search_kind(problem_name: str) -> str:
    """``"airfoil"`` or ``"wing"`` — which measured class this problem is in."""
    return "airfoil" if problem_name == AIRFOIL_PROBLEM else "wing"


def measure_eval_cost(problem_name: str, *, flags: dict | None = None,
                      mission_kwargs: dict | None = None,
                      bounds_overrides: dict | None = None,
                      n: int = 3) -> float:
    """Seconds ONE evaluation of this problem takes, measured at its own box.

    The centre of the box and two Sobol points, so a family whose centre is
    unusually cheap (an infeasible geometry that returns the penalty without
    solving) cannot understate the cost. Only worth calling for a problem
    whose solve is milliseconds — the caller decides; a slow spec should use
    the study's measured median instead of paying seconds to learn it.
    """
    spec = PROBLEM_SPECS[problem_name]
    built = spec.build(mission_kwargs or {}, flags or {}, bounds_overrides)
    lo, hi = built.bounds[:, 0], built.bounds[:, 1]
    from scipy.stats import qmc
    pts = [0.5 * (lo + hi)]
    if n > 1:
        u = qmc.Sobol(built.dim, scramble=True, seed=0).random(n - 1)
        pts.extend(lo + u * (hi - lo))
    times = []
    for x in pts:
        t0 = time.time()
        try:
            built.callable(np.asarray(x, dtype=float))
        except Exception:               # a refused point still costs its time
            pass
        times.append(time.time() - t0)
    return float(np.median(times))


def recommended_search(problem_name: str, *, flags: dict | None = None,
                       mission_kwargs: dict | None = None,
                       bounds_overrides: dict | None = None,
                       effort: str = "balanced", objective: str | None = None,
                       measure_cost: bool = True,
                       pinned: dict | None = None):
    """The measured search recommendation for one problem — a ``SearchPlan``.

    Reads the dimension and the constrained-ness off the BUILT problem rather
    than off a name (the modifiers move both), and quotes a wall clock from a
    cost timed on this machine when the physics is cheap enough to time.
    Raises :class:`optimize.budget.MissingStudyError` if the frozen study is
    not present, so a shell can offer the user's own values and say why the
    recommendation is unavailable instead of inventing one.

    ``pinned`` is the run's own :attr:`RunConfig.pinned`: a fixed variable is
    not searched, and the budget law is a law in the SEARCHED dimension, so a
    recommendation made at the family's full dimension would over-budget
    every design box a user has decided part of.
    """
    from .optimize import budget as B

    spec = PROBLEM_SPECS[problem_name]
    built = spec.build(mission_kwargs or {}, flags or {}, bounds_overrides)
    pin = _pin_of(built, pinned)
    kind = search_kind(problem_name)
    per_eval = None
    if measure_cost and not spec.slow:
        per_eval = measure_eval_cost(problem_name, flags=flags,
                                     mission_kwargs=mission_kwargs,
                                     bounds_overrides=bounds_overrides)
    elif spec.slow:
        # a live-XFOIL family: seconds per candidate, and the study measured
        # exactly that on the section problem
        per_eval = (B.payload().get("kinds", {}).get("airfoil", {})
                    .get("cost", {}).get("per_eval_s"))
    if objective is None:
        # the flags already say it where the caller did not
        objective = ((flags or {}).get("wing_objective")
                     or (flags or {}).get("airfoil_objective"))
    return B.recommend(dim=int(built.dim if pin is None else pin.dim),
                       kind=kind,
                       constrained=bool(built.is_constrained), effort=effort,
                       objective=objective, per_eval_s=per_eval)


def _seed_rows(cfg: RunConfig, bounds: np.ndarray,
               pin: "_Pin | None") -> np.ndarray | None:
    """``cfg.x_seed`` as one validated row in the box the OPTIMISER searches.

    THE GUARANTEE THIS BUYS, and it is the only exact one available: the
    incumbent of a constrained run is a max over evaluated points with all
    margins >= 0, and a seed is evaluated first and never screened out. So a run
    told a design that FLIES returns a design that flies, and returns a score at
    least as good as the seed's. That is what makes "opening a variable cannot
    lose" a fact rather than a probability — every other mechanism here (a
    screened initial design, a feasibility phase) only shortens the odds, and no
    published constrained-BO method turns them into one.

    Validation, all of it refusals rather than repairs, for the reason
    :func:`_pin_of` gives: a seed is a stated design, and quietly moving it
    answers a different question.

    * the wrong LENGTH is refused (against the full vector, which is the frame
      the caller states a design in);
    * a seed that DISAGREES with a pin is refused, naming the row — the pinned
      run's own answer carries the pin's value there, so agreement is the normal
      case and a disagreement means two designs got mixed up;
    * a seed OUTSIDE the searched box is refused, naming the row and both
      bands. It is not clipped: a clipped seed is not the design that flew, so
      the guarantee above would be sold on a design nobody evaluated.
    """
    raw = getattr(cfg, "x_seed", None)
    if raw is None:
        return None
    x = np.asarray(raw, dtype=float).ravel()
    if pin is not None:
        if x.size != len(pin.labels):
            raise ValueError(
                f"x_seed has {x.size} values but this problem's FULL design "
                f"vector has {len(pin.labels)}; a seed is stated in full "
                f"coordinates, pinned rows included")
        for label, value in pin.fixed.items():
            i = list(pin.labels).index(label)
            if not math.isclose(float(x[i]), float(value),
                                rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError(
                    f"x_seed puts {label} at {x[i]:g} while the run pins it at "
                    f"{float(value):g}: that is a different design, not a seed "
                    f"for this one. Pass the design this run could return, or "
                    f"drop the pin")
        x = pin.reduce(x)
    if x.size != bounds.shape[0]:
        raise ValueError(f"x_seed has {x.size} values but the search has "
                         f"{bounds.shape[0]} dimensions")
    lo, hi = bounds[:, 0], bounds[:, 1]
    bad = np.flatnonzero((x < lo - 1e-12) | (x > hi + 1e-12))
    if bad.size:
        i = int(bad[0])
        raise ValueError(
            f"x_seed value {x[i]:g} for dimension {i} is outside the box "
            f"[{lo[i]:g}, {hi[i]:g}] this run searches. A seed is evaluated as "
            f"given and never clipped — a clipped seed is not the design that "
            f"flew. Widen the row, or seed a design inside it")
    return x.reshape(1, -1)


#: Optimisers whose evaluations are a PREFIX of the same search run longer —
#: so "give it N more evaluations" re-flies what was already paid for and then
#: carries on, rather than starting a different search. MEASURED, not assumed
#: (session 58): each was run at budget 12 and at budget 20 on the same
#: config, and ``eval_x[:12]`` compared element-wise.
#:
#:   bo, sobol, random, grid, gradient, slsqp, penalty   identical (max |d| 0)
#:   ga                                                  NOT — its population
#:                                                       is sized from the
#:                                                       budget, so a longer
#:                                                       run is a different
#:                                                       search from step one
#:
#: ``adjoint`` and ``blocks`` are absent because neither was reachable on a
#: family the measurement could use; treat them as unproven, not as refused.
CONTINUABLE_OPTIMISERS: frozenset = frozenset(
    {"bo", "sobol", "random", "grid", "gradient", "slsqp", "penalty"})

#: ...and the ones that can be RESUMED: handed the previous run's evaluations
#: as a training set so that nothing is re-flown at all. The BO arms, and for
#: a reason that is not an implementation detail — a GP is a function of its
#: observations and of nothing else, so a BO loop opened on 51 inherited
#: points is in the same state as one that flew them. A Sobol/grid/random
#: sequence has no state to inherit (its next point depends on its index, and
#: re-flying it off the memo costs nothing), and a local method's state is its
#: trust region, which the record does not carry.
#:
#: ``bo_slsqp`` is here because its state is its BO phase's: the finisher is
#: seeded from the best feasible point the phase knows about, and a phase
#: opened on the prior knows about the prior. A resumed handoff splits the
#: NEW evaluations (``optimize.handoff.run_bo_slsqp_constrained``), so it is
#: the same arm run longer and not a different one. This matters far more
#: than the count of names suggests: ``bo_slsqp`` is what the V3 shell's
#: recommended policy actually flies on a constrained wing
#: (``gui.v3.session.effective_wing_search``), so while it was absent here
#: EVERY "keep going" in that shell fell back to re-flying its whole prefix.
RESUMABLE_OPTIMISERS: frozenset = frozenset({"bo", "bo_slsqp"})


def run_config_of(record: dict) -> RunConfig:
    """The :class:`RunConfig` a stored record was run with, verbatim.

    Fields the record carries that this dataclass no longer has are dropped
    rather than raised on, so a result written by an older version of this
    module still re-runs. Nothing is defaulted in the other direction: a
    record with no ``problem_name`` is not a run, and saying so here is what
    keeps every caller from inventing one.
    """
    cfgd = dict((record or {}).get("config") or {})
    if not cfgd.get("problem_name"):
        raise ValueError("this record carries no configuration, so there is "
                         "nothing to run again")
    keep = {f.name for f in dc_fields(RunConfig)}
    return RunConfig(**{k: v for k, v in cfgd.items() if k in keep})


def _continue_dim(record: dict) -> int:
    """The dimension a re-run of ``record`` would split its BO budget over.

    ``searched_dim`` where the record carries one and ``dim`` otherwise: a run
    with a pinned row searches the FREE rows (``_pinned_built``), and that is
    the number ``run`` hands ``_bo_split``. Reading ``dim`` alone made the
    continuation's arithmetic disagree with the run's on exactly the configs
    stage 2 pins a section into.
    """
    for key in ("searched_dim", "dim"):
        val = (record or {}).get(key)
        if val:
            return int(val)
    return 0


def _bo_n_init_flown(record: dict, cfg: RunConfig) -> int | None:
    """How many Sobol points the run stored in ``record`` actually flew.

    Off ``RunResult.bo_split`` where there is one — that is the split the run
    REPORTED, not a re-derivation of it, and a re-derivation is what a
    continuation must not depend on. Falls back to the same arithmetic the run
    does, and returns None where the record says too little to know: a
    continuation that cannot tell what was flown may not claim to re-fly it.
    """
    split = (record or {}).get("bo_split")
    if split:
        try:
            return int(split[0])
        except (TypeError, ValueError, IndexError):
            pass
    dim = _continue_dim(record)
    if dim <= 0:
        return None
    old = int(((record or {}).get("config") or {}).get("budget") or 0)
    if old <= 0:
        return None
    return _bo_split(old, dim, _bo_n_init(cfg))[0]


def continue_run_config(record: dict, extra: int) -> tuple[RunConfig, dict]:
    """This record's search, given ``extra`` MORE evaluations.

    Two mechanisms, and the first one is the answer wherever it applies:

    RESUME (``bo``, :data:`RESUMABLE_OPTIMISERS`). The evaluations the record
    paid for are handed to the new run as its training set
    (:func:`resume_payload` -> ``run(resume=…)``). Nothing is re-flown: a
    51-evaluation run continued by 8 evaluates 8 designs, its counter opens at
    51, and its record holds all 59. A GP is a function of its observations,
    so a loop opened on those 51 points is in the state the first run ended
    in — this is a continuation in the literal sense, not a re-run that
    happens to arrive at the same place.

    RE-FLY (everything else). The config is re-launched verbatim at
    ``budget + extra`` on the SAME seed, so for every optimiser in
    :data:`CONTINUABLE_OPTIMISERS` the evaluations already paid for come back
    bit-for-bit and the new ones follow them. The prefix is genuinely
    re-flown; with the evaluation memo on (:mod:`aerobo.eval_cache`) it comes
    off disk, and without it, it is bought twice. This is what "Keep going"
    did for every optimiser until resuming existed, and it is still what a
    record too old to carry its own ``eval_x``/``eval_y`` gets.

    Neither is a HANDOFF: nothing is seeded with a previous run's best, which
    is the arm this repo measured and lost with (``RESULTS_HANDOFF.md``).

    The honest answer to "it had not converged — give it more evaluations".
    Nothing is warm-started and nothing is handed a seed: the config is
    re-launched verbatim, at ``budget + extra``, on the SAME seed, and for
    every optimiser in :data:`CONTINUABLE_OPTIMISERS` the evaluations already
    paid for come back bit-for-bit, so the longer run genuinely contains the
    shorter one. That matters because this repo has measured the alternative
    — handing BO a previous run's best as ``x_seed`` — and BO is the worst
    optimiser to hand off TO (``RESULTS_HANDOFF.md``); "one longer run" is the
    arm that wins.

    The one thing that could break the containment is the BO split.
    ``_bo_split`` clamps the Sobol block to ``budget - 1``, so growing a
    budget that was small enough to be clamped un-clamps it and re-draws the
    initial design from scratch (measured: 'free planform (aircraft)' at dim
    6 splits (11, 1) at budget 12 and (12, 8) at budget 20, and the two runs
    diverge at evaluation 11 — the "Keep going starts from the beginning"
    report). So the split that was FLOWN is pinned back on
    (:data:`BO_N_INIT_FLAG`, off ``RunResult.bo_split``): Sobol is a
    sequence, so the same seed's first ``n_init`` points are the same points
    whatever budget follows them, and the pin makes the longer run re-fly the
    prefix bit-for-bit instead of drawing a different starting set. What
    cannot be pinned back — an optimiser outside
    :data:`CONTINUABLE_OPTIMISERS`, or a record that does not say what it
    flew — is reported here rather than discovered afterwards.

    Returns ``(cfg, note)``. ``note`` carries:

    ``exact``     True when the new run re-flies the old one's evaluations
    ``budget``    the new budget
    ``added``     how many evaluations were added
    ``why``       one sentence for the user, always populated
    """
    cfgd = dict((record or {}).get("config") or {})
    extra = int(extra)
    if extra < 1:
        raise ValueError("a continuation must add at least one evaluation")
    old = int(cfgd.get("budget") or 0)
    cfg = replace(run_config_of(record), budget=old + extra)

    opt = str(cfg.optimiser)
    exact, why = True, ""

    resume = resume_payload(record) if opt in RESUMABLE_OPTIMISERS else None
    if resume is not None:
        # THE COUNT THAT MATTERS IS WHAT WAS FLOWN, not what was budgeted: a
        # run the stop rule ended at 34 of 51 is continued by ``extra`` new
        # evaluations from 34, because "8 more" is a statement about physics
        # to be bought and not about a number in a form.
        flown = int(resume["n"])
        cfg = replace(cfg, budget=flown + extra,
                      # the seed design was evaluated by the run being
                      # continued and is IN the training set below; leaving it
                      # on would spend one of the new evaluations re-flying it
                      x_seed=None)
        return cfg, {
            "exact": True, "budget": flown + extra, "added": extra,
            "was": flown, "resume": resume, "resumed": flown,
            "why": (f"a resume, not a re-run: the {flown} evaluations already "
                    f"paid for are handed to this run as its training set, so "
                    f"it starts at {flown} and buys {extra} NEW designs")}
    if opt not in CONTINUABLE_OPTIMISERS:
        exact = False
        why = (f"'{opt}' draws its points from the budget itself, so a longer "
               f"run is a DIFFERENT search from its first evaluation — the "
               f"{old} you already paid for are not re-flown")
    elif opt == "bo":
        was = _bo_n_init_flown(record, cfg)
        now = (None if was is None
               else _bo_split(old + extra, _continue_dim(record),
                              _bo_n_init(cfg))[0])
        if was is not None and now is not None and was != now:
            # THE OLD RUN'S SOBOL BLOCK WAS CLAMPED BY ITS OWN BUDGET
            # (``_bo_split`` caps n_init at budget - 1), so the longer run
            # would ask for a different initial design and diverge at the
            # first evaluation. Measured before this was pinned: 'tandem'
            # at dim 10 split (11, 1) at budget 12 and (16, 4) at budget 20,
            # and the two runs' histories differed from evaluation 12.
            #
            # So the split is PINNED to the one that was flown rather than
            # reported as lost. Sobol is a sequence: the first ``was`` points
            # of the same seed are the same points whatever comes after them,
            # so pinning re-flies the paid-for prefix bit-for-bit (and off
            # the eval memo, for free) and the added evaluations are BO
            # iterations that carry on from it. That is what the button says
            # it does, and what "one longer run" means in RESULTS_HANDOFF.md.
            cfg = replace(cfg, flags={**(cfg.flags or {}),
                                      BO_N_INIT_FLAG: int(was)})
            why = (f"the first run's initial design was cut to {was} points "
                   f"by its own {old}-evaluation budget, so this "
                   f"continuation PINS the Sobol block to those {was} "
                   f"(bo_n_init) instead of the {now} a fresh "
                   f"{old + extra}-evaluation run would draw: the {old} "
                   f"evaluations already paid for come back identically and "
                   f"{extra} more follow them")
    if not why:
        why = (f"the same search, run longer: the {old} evaluations already "
               f"paid for are re-flown identically (off the evaluation memo "
               f"where one is on) and {extra} more follow them")
    return cfg, {"exact": exact, "budget": old + extra, "added": extra,
                 "was": old, "resume": None, "resumed": 0, "why": why}


def _margin_rows(g, n: int) -> "np.ndarray | None":
    """A stored ``eval_g`` as an ``(n, m)`` array of margins, or None.

    THE SHAPE IS NOT REDUNDANT with ``np.atleast_2d``, and getting it wrong
    silently disabled every resume on this repo's most common problems. A
    family with ONE constraint stores its log as a flat list of ``n`` scalars
    (the runners' histories carry ``g`` as ``(n,)`` when ``m == 1`` — the
    split :func:`optimize.handoff._stack` exists to reconcile), and
    ``atleast_2d`` turns that into ``(1, n)``: one point with n margins. The
    row-count check then fails, :func:`resume_payload` returns None, and the
    continuation quietly falls back to re-flying the whole prefix. Measured
    on a stored ``tail [designed tail] + free chord law`` run: 53 evaluations,
    every value finite, and not resumable for this reason alone.

    ``(n,)`` is read as a COLUMN whenever ``n != 1``: a log has one row per
    evaluation, and a one-evaluation log is the only case where a flat list
    of m values is the other reading. Returns None where the rows cannot be
    made to match ``n`` rather than reshaping into a lie.
    """
    a = np.asarray(g, dtype=float)
    if a.ndim == 1:
        a = a[None, :] if n == 1 else a[:, None]
    elif a.ndim != 2:
        return None
    return a if a.shape[0] == n else None


def resume_payload(record: dict) -> dict | None:
    """The evaluations a stored run PAID FOR, ready to be resumed from.

    ``{"x": [[…]], "y": [...], "g": [[…]] | None, "n": int}`` in FULL design
    coordinates, or None where the record cannot be resumed from — an empty
    log, a log whose objective values are not all finite (a partial record
    written before anything solved), or a constrained run with no margins.

    None is not a failure: :func:`continue_run_config` falls back to re-flying
    the prefix, which is what this repo did before resuming existed.
    """
    rd = record or {}
    X = rd.get("eval_x") or []
    y = rd.get("eval_y") or []
    if not X or len(X) != len(y):
        return None
    try:
        Xa = np.atleast_2d(np.asarray(X, dtype=float))
        ya = np.asarray(y, dtype=float).ravel()
    except (TypeError, ValueError):
        return None
    if Xa.shape[0] != ya.size or not np.all(np.isfinite(Xa)):
        return None
    if not np.all(np.isfinite(ya)):
        # a value the GP cannot be trained on. Refusals are FINITE (the -100
        # sentinel); a None/inf here means the log is not a complete record
        # of what was evaluated, and half a training set is not a resume
        return None
    g = rd.get("eval_g")
    if rd.get("is_constrained"):
        if not g or len(g) != len(X):
            return None
        try:
            ga = _margin_rows(g, Xa.shape[0])
        except (TypeError, ValueError):
            return None
        if ga is None or not np.all(np.isfinite(ga)):
            return None
        g = _json_safe(ga)
    else:
        g = None
    return {"x": _json_safe(Xa), "y": _json_safe(ya), "g": g,
            "n": int(Xa.shape[0])}


def feasible_points_of(record: dict, quantile: float = 0.25) -> dict | None:
    """The FEASIBLE designs a stored run evaluated — ``{"x": [...], "y": [...]}``.

    A search is a far better instrument than a uniform sample on a family
    whose feasible set is a per-cent of its box: the run that has already
    been paid for knows where the designs are, and a recommendation drawn
    around its own answers is measured in the strongest sense this repo has.
    ``quantile`` keeps the best share by objective, as
    :func:`recommend.best_share` does for draws.

    None where the record carries no per-evaluation log, or nothing in it was
    feasible. Points come back in FULL design coordinates — the log's own —
    so a caller with a pin still has to reduce them.
    """
    rd = record or {}
    X = rd.get("eval_x") or []
    Y = rd.get("eval_y") or []
    G = rd.get("eval_g")
    if not X or len(X) != len(Y):
        return None
    keep_x, keep_y = [], []
    for i, x in enumerate(X):
        y = Y[i]
        if y is None or not np.isfinite(float(y)):
            continue
        if G is not None:
            row = G[i] if i < len(G) else None
            if row is None:
                continue
            # a log may store one margin per evaluation as a SCALAR or as a
            # row of them, depending on how many constraints the family has
            margins = ([float(row)] if isinstance(row, (int, float))
                       else [float(v) for v in row])
            if not margins or min(margins) < 0.0:
                continue
        keep_x.append([float(v) for v in x])
        keep_y.append(float(y))
    if not keep_x:
        return None
    if 0.0 < float(quantile) < 1.0 and len(keep_x) > 1:
        k = max(1, int(round(float(quantile) * len(keep_x))))
        cut = float(np.sort(np.asarray(keep_y))[::-1][k - 1])
        pairs = [(x, y) for x, y in zip(keep_x, keep_y) if y >= cut]
        keep_x = [x for x, _ in pairs]
        keep_y = [y for _, y in pairs]
    return {"x": keep_x, "y": keep_y, "n": len(keep_x)}


def can_resume(record: dict) -> bool:
    """True iff this record can be CONTINUED without re-flying anything.

    Its optimiser carries a training set (:data:`RESUMABLE_OPTIMISERS`) and
    its log is complete enough to be one (:func:`resume_payload`). A shell
    asks this to know which arithmetic its "keep going" button is doing:
    a resume adds evaluations to what was FLOWN, a re-fly adds them to the
    budget that was asked for.
    """
    opt = str(((record or {}).get("config") or {}).get("optimiser") or "")
    return opt in RESUMABLE_OPTIMISERS and resume_payload(record) is not None


def _resume_prior(cfg: RunConfig, resume: dict, search: "_BuiltProblem",
                  pin: "_Pin | None") -> tuple:
    """A resume payload as the OPTIMISER takes it: ``(X, y, g)`` over the
    dimensions it searches, validated against the box it will search.

    Raises rather than clipping. The payload comes off a record whose config
    this run re-uses, so a point outside the box means the two are not the
    same search — and a resume that silently moved the training set would be
    a run reporting a history it did not fly.
    """
    X = np.atleast_2d(np.asarray((resume or {}).get("x"), dtype=float))
    y = np.asarray((resume or {}).get("y"), dtype=float).ravel()
    if X.shape[0] != y.size or X.size == 0:
        raise ValueError("a resume needs one objective value per point")
    if pin is not None:
        if X.shape[1] != len(pin.labels):
            raise ValueError(
                f"the run to resume evaluated {X.shape[1]}-dimensional "
                f"designs and this problem states {len(pin.labels)}")
        X = np.atleast_2d(pin.reduce(X))
    if X.shape[1] != search.dim:
        raise ValueError(
            f"the run to resume evaluated {X.shape[1]}-dimensional designs "
            f"and this search has {search.dim} dimensions — the design box "
            f"moved, so this is a different search and not a continuation")
    box = np.asarray(search.bounds, dtype=float)
    tol = 1e-9 * np.maximum(1.0, np.abs(box).max(axis=1))
    if np.any(X < box[:, 0] - tol) or np.any(X > box[:, 1] + tol):
        raise ValueError(
            "the run to resume evaluated designs outside this run's box; the "
            "box moved between the two, so the evaluations are not this "
            "search's")
    g = (resume or {}).get("g")
    if search.is_constrained:
        if g is None:
            raise ValueError("a constrained run cannot be resumed without its "
                             "per-evaluation margins")
        # the same (n,)/(n, m) reading resume_payload does — a payload that
        # came from somewhere else must not be shaped by a second rule
        G = _margin_rows(g, X.shape[0])
        if G is None:
            raise ValueError("a resume needs one margin row per point")
        return (X, y, G)
    return (X, y)


def check_x_seed(cfg: RunConfig) -> None:
    """Raise if this config's ``x_seed`` is not a design this run could return.

    The same validation the run itself does (:func:`_seed_rows`), reachable
    BEFORE a run is launched so a shell can offer "start from this design" as
    a control that either arms or explains itself, rather than as a button that
    raises 20 seconds into a search. No physics is evaluated: the problem is
    built, the pin resolved, and the seed checked against the box.

    The OPTIMISER is checked first and for free, because it is the one thing
    here that can be wrong before anything is built. Without it this function
    passed a config the run then refused, which is precisely the button that
    raises 20 seconds into a search — the failure it exists to remove.
    """
    if (getattr(cfg, "x_seed", None) is not None
            and cfg.optimiser not in X_SEED_OPTIMISERS):
        raise ValueError(
            f"optimiser {cfg.optimiser!r} cannot take an x_seed; the ones that "
            f"can are {sorted(X_SEED_OPTIMISERS)}. Use one of those, or drop "
            f"x_seed")
    spec = PROBLEM_SPECS[cfg.problem_name]
    flags, _dropped = sanitise_flags(cfg.problem_name, cfg.flags)
    built = spec.build(cfg.mission_kwargs or {}, _strip_wing_objective(flags),
                       cfg.bounds_overrides)
    # ...against the box the OPTIMISER searches, which is what ``run`` hands
    # ``_seed_rows`` (``_search_problem`` -> ``_pinned_built.bounds``, the free
    # rows only). ``_Pin.bounds`` is the FULL box with the pinned rows
    # collapsed, and ``_seed_rows`` has already reduced the seed by the time it
    # compares the two — so passing it here made every pinned config raise
    # "x_seed has 7 values but the search has 8 dimensions", and the shell's
    # "start from this design" button was dead on any session with a fixed row
    # while the same config ran fine.
    search, pin = _search_problem(cfg, built)
    _seed_rows(cfg, np.asarray(search.bounds, dtype=float), pin)


def _bo_x_init(cfg: RunConfig, built: "_BuiltProblem",
               bounds: np.ndarray,
               pin: "_Pin | None" = None) -> np.ndarray | None:
    """Seed rows for the BO initial design, or ``None`` for the plain Sobol.

    Only assembled when the run explicitly asks via the ``bo_warm_start``
    flag, so every existing study keeps its untouched Sobol draw
    (:func:`optimize.constrained.run_bo_constrained` documents ``x_init=None``
    as the bit-for-bit legacy path).

    It matters whenever EXTRA design variables are opened that the legacy
    design pins at zero — a chord law, say. The wider box's Sobol draw no
    longer contains the un-reshaped design, so a search that provably cannot
    do worse AT CONVERGENCE (the smaller problem's feasible set is a subset)
    can still come back worse at a real budget. Seeding the problem's own
    ``x0`` puts that design back in the training set; the Sobol draw shrinks
    by the same count, so the total budget is unchanged.

    …and it matters for the same reason on ``composite_goal``, where the
    objective's REFERENCE POINT is the seed: a run told to defend a section it
    never evaluated can still hand back something worse than it. The seed
    vector has two spellings, ``x0`` on the section+twist wing and ``w0`` on
    the bare 2-D section (:func:`_seed_design_report` documents the same
    split), and reading only the first is why the warm start was a silent
    no-op on every section problem.
    """
    seeded = _seed_rows(cfg, bounds, pin)
    if not (cfg.flags or {}).get(BO_WARM_START_FLAG):
        # an explicit seed needs no flag: it is not a policy, it is a design
        return seeded
    x0 = getattr(built.problem, "x0", None)
    if x0 is None:
        x0 = getattr(built.problem, "w0", None)
    if x0 is None:
        return seeded
    x0 = np.clip(np.asarray(x0, dtype=float).ravel(),
                 bounds[:, 0], bounds[:, 1])
    rows = x0.reshape(1, -1)
    if seeded is None:
        return rows
    # both, the stated design FIRST, and never the same design twice (two
    # identical rows would cost two evaluations of one point)
    if np.allclose(seeded[0], rows[0], rtol=0.0, atol=1e-12):
        return seeded
    return np.vstack([seeded, rows])


def rows_outside_validity(built: "_BuiltProblem") -> list:
    """Design-box rows whose SEARCHED band reaches outside the band the family
    validates over — each ``{label, searched, validated}``.

    WHY THIS EXISTS. ``_apply_overrides`` writes the user's rows into a COPY of
    the problem's box, and that copy is what the sampler and the acquisition
    draw from; the problem object keeps its own box and every family re-checks
    the candidate against THAT, returning ``"bounds violation"`` for anything
    outside it. So a row the user WIDENS is drawn from and then refused, and the
    run silently reduces to the published band: measured on ``tail``, a
    ``taper`` row widened to 0.05-1.0 refuses every draw below 0.20, and the
    same for ``twist_root_deg``.

    Seven rows do not have this problem, because each was taught to travel to
    the problem as well as to the sampler (:func:`_arm_band_kwargs`,
    :func:`_size_band_kwargs`, :func:`_height_band_kwargs`, and the tail AREA
    through :func:`_area_band_kwargs` — the newest, and the one that stopped
    being optional the day the tail box became a fraction of the wing rather
    than a constant nobody could reach past). The rest have not been done, and
    until they
    are, the honest thing is to be able to SAY which rows are in that state —
    "a flag that is silently dropped is worse than one that is refused", and a
    widened row that quietly does nothing is the same failure wearing a band.

    NOT a refusal. Narrowing a row is always honoured, widening one is a
    legitimate thing to want (a calibration is a default, not a ban), and the
    part of the row inside the validated band still searches normally — so this
    reports, and the caller decides what to say. Empty for every run that did
    not widen anything, which is every published study.
    """
    own = getattr(getattr(built, "problem", None), "bounds", None)
    if own is None:
        return []
    own = np.asarray(own, dtype=float)
    box = np.asarray(built.bounds, dtype=float)
    if own.shape != box.shape:
        return []
    labels = list(built.param_labels)
    out = []
    for i in range(box.shape[0]):
        lo_out = box[i, 0] < own[i, 0] - 1e-12
        hi_out = box[i, 1] > own[i, 1] + 1e-12
        if not (lo_out or hi_out):
            continue
        out.append({
            "label": (labels[i] if i < len(labels) else f"x[{i}]"),
            "searched": [float(box[i, 0]), float(box[i, 1])],
            "validated": [float(own[i, 0]), float(own[i, 1])],
        })
    return out


def _extract(res: Any, is_constrained: bool) -> dict:
    """Common view over BOResult / RunHistory / ConstrainedRunHistory."""
    X = np.asarray(res.X, dtype=float)
    y = np.asarray(res.y, dtype=float)
    bsf = np.asarray(res.best_so_far, dtype=float)
    out = {
        "X": X, "y": y, "best_so_far": bsf,
        "best_x": getattr(res, "best_x", None),
        "best_y": getattr(res, "best_y", None),
        "acqf": getattr(res, "acqf", None),
        "g": getattr(res, "g", None),
        "best_g": getattr(res, "best_g", None),
        "n_feasible": getattr(res, "n_feasible", None),
    }
    meta = getattr(res, "meta", None) or {}
    out["failures"] = getattr(res, "failures", None)
    if out["failures"] is None:
        out["failures"] = meta.get("failures")
    # absent on every legacy run, which is what makes them None on the result
    out["n_screened"] = (getattr(res, "n_screened", None)
                         if getattr(res, "feasibility", _F_OFF) != _F_OFF
                         else meta.get("n_screened"))
    out["n_rescue"] = meta.get("n_rescue")
    out["n_rescue_blind"] = meta.get("n_rescue_blind")
    return out


def _handoff_report(cfg: RunConfig, res: Any, search: _BuiltProblem
                    ) -> dict | None:
    """What a ``bo_slsqp`` run's two phases did, or None for every other arm.

    Read off the history the runner returned rather than recomputed from the
    config, so it reports what HAPPENED. The one exception is ``bo_split``,
    which the runner does not store: it is re-derived from the same three
    inputs ``_dispatch`` passed it, exactly as the plain-BO field above is.
    """
    if cfg.optimiser != "bo_slsqp":
        return None
    meta = getattr(res, "meta", None) or {}
    n_a = meta.get("n_a")
    n_prior = int(meta.get("n_prior") or 0)
    return {
        "phi": meta.get("phi"),
        "n_a": n_a, "n_b": meta.get("n_b"),
        # how many of this run's evaluations were INHERITED, and therefore
        # which of the two numbers above the split was taken over: n_a/n_b
        # are the NEW evaluations on a resumed handoff
        "n_prior": n_prior,
        # ...and there is NO Sobol block on a resumed run, so re-deriving one
        # here would report an initial design that was never drawn
        "bo_split": (None if n_a is None or n_prior else
                     list(_bo_split(int(n_a), search.dim, _bo_n_init(cfg)))),
        "handed_over": meta.get("handed_over"),
        "a_best": meta.get("a_best"),
    }


def _dispatch(cfg: RunConfig, built: _BuiltProblem, wrapped: _ProgressCounter,
              iter_cb: Callable | None = None, pin: "_Pin | None" = None,
              prior: tuple | None = None):
    """Run the chosen optimiser at cfg.budget/seed; return its native result.

    ``iter_cb`` (optional) is forwarded to the BO loops only — one call per
    BO iteration with GP posterior diagnostics at the chosen candidate.
    ``None`` (default) is bit-for-bit the legacy path.

    ``prior`` (optional, BO only) are evaluations ALREADY PAID FOR by the run
    this one continues (:func:`_resume_prior`): the loop opens with them as
    its training set, draws no initial design, and spends the whole remaining
    budget on new points. Refused on every other optimiser rather than
    ignored — a GA resamples its population and a DOE its grid, so neither
    can be told where a previous run had been.
    """
    opt = OPTIMISER_SPECS.get(cfg.optimiser)
    if opt is None:
        raise ValueError(f"unknown optimiser {cfg.optimiser!r}; "
                         f"choices: {optimiser_names()}")
    if (getattr(cfg, "x_seed", None) is not None
            and cfg.optimiser not in X_SEED_OPTIMISERS):
        # REFUSED, not ignored. A GA samples its own population and a DOE its
        # own points, so neither can be handed a start; accepting a seed here
        # would let a caller believe the guarantee it exists for ("this run
        # cannot come back worse than the design I gave it") was armed while
        # nothing read the design — the silently-dropped-flag failure this
        # module refuses everywhere.
        raise ValueError(
            f"optimiser {cfg.optimiser!r} cannot take an x_seed; the ones that "
            f"can are {sorted(X_SEED_OPTIMISERS)}. Use one of those, or drop "
            f"x_seed")
    budget = int(cfg.budget)
    seed = int(cfg.seed)
    bounds = built.bounds
    d = built.dim
    if prior is not None and cfg.optimiser not in RESUMABLE_OPTIMISERS:
        raise ValueError(
            f"optimiser {cfg.optimiser!r} cannot be resumed from a previous "
            f"run's evaluations; the ones that carry a training set are "
            f"{sorted(RESUMABLE_OPTIMISERS)}. Continue it as one longer run "
            f"instead")
    if prior is not None and getattr(cfg, "x_seed", None) is not None:
        # REFUSED, not ignored, for the third time in this function. A resumed
        # run opens on its inherited training set and draws NO initial design,
        # which is the only place a seed is evaluated — so accepting both here
        # would drop the seed on the floor while the caller believed the
        # cannot-come-back-worse guarantee was armed. ``continue_run_config``
        # already clears the seed on the resume path (the design is IN the
        # training set, and re-flying it would spend one of the new
        # evaluations), so this catches the hand-built config only.
        raise ValueError(
            "a resumed run cannot also be given an x_seed: it opens on the "
            "evaluations it inherited and draws no initial design, so the "
            "seed would be read by nothing. Drop x_seed to continue this "
            "search, or drop the resume to start a fresh one from that design")
    n_prior = 0 if prior is None else int(np.asarray(prior[0]).shape[0])
    if prior is not None and budget - n_prior < 1:
        raise ValueError(
            f"a resumed run needs at least one new evaluation: budget "
            f"{budget} against {n_prior} already flown")

    if built.is_constrained:
        if not opt.supports_constrained:
            raise ValueError(
                f"optimiser {cfg.optimiser!r} does not support the CONSTRAINED "
                f"problem {cfg.problem_name!r}; use one of "
                f"{compatible_optimisers(cfg.problem_name)}")
        from .optimize import constrained as C
        if cfg.optimiser == "blocks":
            prob_obj = built.problem
            blocks = getattr(prob_obj, "blocks", None)
            if not blocks:
                raise ValueError(
                    f"optimiser 'blocks' needs a problem declaring variable "
                    f"blocks; {cfg.problem_name!r} does not")
            from .optimize.blocks import run_blocks_constrained
            x0 = getattr(prob_obj, "x0", None)
            if x0 is not None:      # keep the warm start inside a narrowed box
                x0 = np.clip(np.asarray(x0, dtype=float),
                             bounds[:, 0], bounds[:, 1])
            # per-block sub-optimiser overrides + runner options travel in
            # flags, so a shell can answer "which method for which block?"
            # without a new RunConfig field
            opts = cfg.flags or {}
            per_block = opts.get("block_optimisers")
            if per_block:
                blocks = tuple(
                    {**b, "optimiser": per_block.get(b.get("name"),
                                                     b.get("optimiser", "bo"))}
                    for b in blocks)
            return run_blocks_constrained(
                wrapped, bounds, budget, seed=seed, blocks=blocks, x0=x0,
                cycles=int(opts.get("block_cycles", 2)),
                # the per-block sub-searches ARE BO runs, so the feasibility
                # policy reaches them too — a flag this optimiser accepted and
                # then ignored would be the silently-dropped-flag hole this
                # module refuses everywhere else
                feasibility=_bo_feasibility(cfg))
        if cfg.optimiser == "bo":
            if prior is not None:
                # RESUMED: no initial design, and every evaluation left in the
                # budget is a new point
                return C.run_bo_constrained(
                    wrapped, bounds, n_init=0, n_iter=budget - n_prior,
                    seed=seed, prior=prior, iter_cb=iter_cb,
                    refusal=_bo_refusal(cfg),
                    feasibility=_bo_feasibility(cfg))
            n_init, n_iter = _bo_split(budget, d, _bo_n_init(cfg))
            return C.run_bo_constrained(wrapped, bounds, n_init=n_init,
                                        n_iter=n_iter, seed=seed,
                                        x_init=_bo_x_init(cfg, built, bounds,
                                                          pin),
                                        iter_cb=iter_cb,
                                        refusal=_bo_refusal(cfg),
                                        feasibility=_bo_feasibility(cfg),
                                        # cost only: the screen's
                                        # pool skips a draw the size
                                        # gate already refuses,
                                        # without paying the solver.
                                        # The kept set is unchanged.
                                        gate=_gate_grader(cfg, built))
        if cfg.optimiser == "bo_slsqp":
            # the BO phase sizes its Sobol block off its OWN share, not the
            # total: `_bo_split(budget, ...)` would spend the finisher's
            # evaluations on initial-design points it never gets to use.
            from .optimize.handoff import run_bo_slsqp_constrained
            stated_n_init = _bo_n_init(cfg)
            return run_bo_slsqp_constrained(
                wrapped, bounds, budget, seed=seed,
                phi=_handoff_phi(cfg),
                bo_split=lambda n: _bo_split(n, d, stated_n_init),
                # RESUMED: the prior IS the BO phase's opening, so there is
                # no initial design to seed and the runner refuses being
                # handed both. The split is over the NEW evaluations.
                x_init=(None if prior is not None
                        else _bo_x_init(cfg, built, bounds, pin)),
                prior=prior,
                iter_cb=iter_cb,
                refusal=_bo_refusal(cfg),
                feasibility=_bo_feasibility(cfg))
        if cfg.optimiser == "ga":
            return C.run_ga_constrained(
                wrapped, bounds, budget, seed=seed,
                pop_size=min(20, max(4, budget // 3)))
        return C.REGISTRY_CONSTRAINED[cfg.optimiser](
            wrapped, bounds, budget, seed=seed,
            **({"x_seed": cfg.x_seed}
               if getattr(cfg, "x_seed", None) is not None else {}))

    # ---- unconstrained ----
    if not opt.supports_unconstrained:
        raise ValueError(
            f"optimiser {cfg.optimiser!r} does not support the UNCONSTRAINED "
            f"problem {cfg.problem_name!r}; use one of "
            f"{compatible_optimisers(cfg.problem_name)}")
    if cfg.optimiser == "bo":
        from .optimize.bo import run_bo
        acqf = (cfg.flags or {}).get("acqf", "logei")
        if prior is not None:
            return run_bo(wrapped, bounds, n_init=0, n_iter=budget - n_prior,
                          acqf=acqf, seed=seed, maximize=True, iter_cb=iter_cb,
                          refusal=_bo_refusal(cfg),
                          feasibility=_bo_feasibility(cfg), prior=prior)
        n_init, n_iter = _bo_split(budget, d, _bo_n_init(cfg))
        return run_bo(wrapped, bounds, n_init=n_init, n_iter=n_iter,
                      acqf=acqf, seed=seed, maximize=True, iter_cb=iter_cb,
                      refusal=_bo_refusal(cfg),
                      feasibility=_bo_feasibility(cfg),
                      x_init=_bo_x_init(cfg, built, bounds, pin))
    if cfg.optimiser == "adjoint":
        if cfg.problem_name != "trim wing" or d != 3:
            raise ValueError(
                "adjoint is only wired for the 3-D trim wing (reverse-mode "
                "LLT gradient == Tier A trim vector)")
        from .adjoint import TorchLLTProblem
        from .optimize import baselines
        tp = TorchLLTProblem(n_knots=1, fix_root=False)
        res = baselines.run_adjoint(tp, bounds, budget, seed=seed)
        if wrapped.cb is not None:      # adjoint bypasses the callable wrapper
            for i, v in enumerate(res.best_so_far, 1):
                wrapped.cb(i, float(v))
        return res
    from .optimize import baselines
    if cfg.optimiser == "ga":
        return baselines.run_ga(wrapped, bounds, budget, seed=seed,
                                pop_size=min(20, max(4, budget // 3)))
    return baselines.REGISTRY[cfg.optimiser](
        wrapped, bounds, budget, seed=seed,
        **({"x_seed": cfg.x_seed}
           if getattr(cfg, "x_seed", None) is not None else {}))


# =====================================================================
# run / persist / load
# =====================================================================

class _StopSearch(Exception):
    """Raised inside the progress wrapper when ``run``'s stop rule fires."""


def run(cfg: RunConfig, progress_cb: Callable | None = None,
        results_dir: str | Path | None = None,
        iter_cb: Callable | None = None,
        stop_rule: Callable | None = None,
        stop_reason: str | None = None,
        eval_cache=None,
        resume: dict | None = None) -> RunResult:
    """Build the problem, dispatch the optimiser, return a JSON-safe RunResult.

    ``progress_cb(i, best_so_far)`` (optional) is called once per evaluation
    for the live GUI plot; declare ``**kwargs`` on the callback to receive the
    rich per-eval payload (``f``, ``g``, ``feasible``, ``x``). ``iter_cb(rec)``
    (optional) fires once per BO iteration with GP posterior diagnostics
    (``mu``/``sigma``/``acq``/…); the records are also collected onto
    ``RunResult.bo_iters``. ``results_dir`` (optional) persists the result to
    ``<results_dir>/<timestamp>.json`` and sets ``RunResult.path``; the default
    None writes nothing.

    ``cfg.pinned`` (optional) holds named design variables FIXED: the
    optimiser searches the remaining dimensions only, and everything reported
    here is still the full design vector (:class:`_Pin`), with the pinned
    rows' bounds collapsed to ``[v, v]`` and the pins echoed on
    ``RunResult.pinned``.

    ``stop_rule(i, best_so_far) -> bool`` (optional) ENDS the search early —
    the "run until it stops improving" of
    :class:`optimize.budget.ConvergenceStop`. The evaluations already made
    are not thrown away: the result is rebuilt from the per-evaluation log by
    :func:`partial_result` and comes back with ``partial=True`` and
    ``stop_reason`` set, exactly as a cancelled run does, because that is
    what it is — a search that stopped where something stopped it rather than
    where the budget ran out. Nothing about the optimisers changes; the rule
    is applied between evaluations, so a run WITHOUT one is bit-for-bit the
    legacy path.

    ``eval_cache`` (optional) memoises the objective on disk
    (:mod:`aerobo.eval_cache`): ``True`` for the default directory, a path for
    another, None (the default) for no memo at all. It changes NOTHING the run
    reports — the objective is a pure function of the design vector, so every
    evaluation, the per-evaluation log and the history are bit-identical — and
    it exists for one case: a CONTINUATION (:func:`continue_run_config`) re-flies
    the whole prefix of the run it lengthens, and this hands that prefix back
    off disk instead of buying it twice. What it costs is the meaning of
    ``wall_time_s``, which is why it is off by default and why a run that used
    it says so on ``RunResult.eval_cache`` (``{hits, misses, dir}``).

    ``resume`` (optional, BO only) is a previous run's evaluations
    (:func:`resume_payload`) handed to this one as its TRAINING SET. Nothing
    in it is flown again: the loop opens with a GP fitted to those points and
    spends ``budget - len(resume)`` evaluations on new ones, so "give this run
    8 more evaluations" costs 8. The evaluation counter, the live callback and
    the stop rule all start at ``len(resume)`` — a continuation counts from
    where the run it continues stopped, because that is where it is — and the
    returned history contains the inherited points followed by the new ones,
    so the record is one run of ``budget`` evaluations. ``RunResult.resumed``
    says how many of them were inherited.
    """
    t0 = time.time()
    if cfg.problem_name not in PROBLEM_SPECS:
        raise ValueError(f"unknown problem {cfg.problem_name!r}; "
                         f"choices: {problem_names()}")
    spec = PROBLEM_SPECS[cfg.problem_name]
    check_flags(cfg.problem_name, cfg.flags, "RunConfig.flags")
    built = spec.build(cfg.mission_kwargs or {}, cfg.flags or {},
                       cfg.bounds_overrides)
    # what the OPTIMISER sees. Identical object when nothing is pinned, so
    # every existing run is bit-for-bit its old self.
    if str((cfg.flags or {}).get("airfoil_objective") or "") == \
            PARETO_OBJECTIVE_NAME:
        # A front is not a scalar search, so it never reaches _dispatch. The
        # branch is HERE, above `_search_problem`, because everything below
        # assumes one objective value per evaluation.
        if stop_rule is not None:
            # REFUSED, not ignored. `optimize.mobo.run_mobo_constrained` has no
            # early-stop hook, so accepting the argument here would let a
            # caller — the V3 stage-2 runner passes one on every run — believe
            # a Stop button was armed while nothing read it. That is the
            # flag-nobody-reads failure this module refuses everywhere else,
            # and a silent no-op on a CANCEL is the worst place to have one.
            # Wiring a real early stop into the multi-objective loop, keeping
            # the evaluations already paid for, is stated remaining work.
            raise ValueError(
                "a front run cannot be stopped early yet: the multi-objective "
                "loop has no stop hook, so a stop_rule passed here would "
                "never be consulted. Run the front to its budget, or use a "
                "scalar objective if you need to be able to stop")
        if eval_cache is not None and eval_cache is not False:
            # REFUSED, not ignored — the same rule the stop_rule above obeys.
            # A front evaluates a VECTOR objective through a different loop,
            # and `EvalMemo` stores one scalar (plus margins) per design; a
            # memo accepted here would be an argument nothing reads on the one
            # run whose evaluations are the most expensive in the repo.
            raise ValueError(
                "a front run cannot use the evaluation memo: its objective "
                "returns a vector per evaluation and aerobo.eval_cache stores "
                "one scalar. Run the front without eval_cache")
        return _pareto_run_result(cfg, built, t0, progress_cb=progress_cb,
                                  results_dir=results_dir)

    search, pin = _search_problem(cfg, built)

    # WHAT THIS RUN INHERITS. Validated against the box it is about to
    # search, before an evaluation is spent: a payload from a different box
    # is a different search, and the honest place to say so is here.
    prior = None if resume is None else _resume_prior(cfg, resume, search, pin)
    n_prior = 0 if prior is None else int(np.asarray(prior[0]).shape[0])
    prior_rows: list = []
    prior_best = -math.inf
    if prior is not None:
        X_full = (np.asarray(prior[0], dtype=float) if pin is None
                  else np.atleast_2d(pin.expand(np.asarray(prior[0],
                                                           dtype=float))))
        G_prior = prior[2] if len(prior) > 2 else None
        for i in range(n_prior):
            g_row = (None if G_prior is None
                     else [float(v) for v in np.atleast_1d(G_prior[i])])
            feas_i = True if g_row is None else bool(min(g_row) >= 0.0)
            f_i = float(prior[1][i])
            if feas_i and f_i > prior_best:
                prior_best = f_i
            # the same shape a rich progress callback reports, so a resumed
            # run that is STOPPED rebuilds into a record holding everything
            # it inherited as well as everything it flew — the alternative is
            # a continuation that throws away the run it continues the moment
            # the user presses stop
            prior_rows.append({"n": i + 1, "x": [float(v) for v in X_full[i]],
                               "f": f_i, "g": g_row, "feasible": feas_i})

    # THE ONE THING BETWEEN THE OPTIMISER AND THE PHYSICS, when a caller asked
    # for it: evaluations already flown for this (source tree, configuration,
    # design vector) come back off disk instead of being bought again. It sits
    # BELOW `_ProgressCounter`, so a cached evaluation is still counted,
    # reported to the live callback and written to the per-evaluation log
    # exactly as a flown one — the run is the same run, only faster.
    memo = None
    _cache_dir = eval_cache_mod.resolve_dir(eval_cache)
    if _cache_dir is not None:
        memo = eval_cache_mod.EvalMemo(
            search.callable, is_constrained=search.is_constrained,
            cache_dir=_cache_dir, config=cfg.to_dict())

    bo_iters: list = []

    def _iter_rec(rec: dict):
        bo_iters.append(_json_safe(rec))
        if iter_cb is not None:
            iter_cb(rec)

    stop_log: list = []

    # the rule needs the SAME rich payload partial_result reads, so this
    # wrapper collects it rather than asking the caller's callback (which may
    # be the plain two-argument form) to hand it over
    def _watch(i, best, _inner=progress_cb, **kw):
        stop_log.append({"n": int(i), "x": kw.get("x"), "f": kw.get("f"),
                         "g": kw.get("g"), "feasible": kw.get("feasible")})
        if _inner is not None:
            if _cb_accepts_kwargs(_inner):
                _inner(i, best, **kw)
            else:
                _inner(i, best)
        if stop_rule(int(i), best):
            raise _StopSearch

    cb = progress_cb if stop_rule is None else _watch

    wrapped = _ProgressCounter(search.callable if memo is None else memo,
                               search.is_constrained, cb,
                               expand=None if pin is None else pin.expand,
                               grader=_gate_grader(cfg, built),
                               start=n_prior, best=prior_best)
    try:
        res = _dispatch(cfg, search, wrapped, pin=pin,
                        iter_cb=(_iter_rec if cfg.optimiser in BO_ITER_OPTIMISERS
                                 else None),
                        prior=prior)
    except _StopSearch:
        return partial_result(
            cfg, prior_rows + stop_log, wall_time_s=time.time() - t0,
            resumed=(n_prior or None),
            bo_iters=bo_iters or None, results_dir=results_dir,
            stop_reason=stop_reason or "the stop rule fired",
            # a STOPPED run is the one most likely to be continued, so its
            # record is exactly where the memo's accounting has to survive
            eval_cache=(memo.stats() if memo is not None else None))
    ex = _extract(res, built.is_constrained)

    # back into FULL design coordinates before anything is reported: the
    # optimiser searched the free dimensions, every consumer speaks the whole
    # vector (a no-op without a pin)
    if pin is not None:
        if ex["best_x"] is not None:
            ex["best_x"] = pin.expand(np.asarray(ex["best_x"], dtype=float))
        if len(ex["X"]):
            ex["X"] = pin.expand(ex["X"])

    # best point / score
    best_x = ex["best_x"]
    best_x_list = None if best_x is None else [float(v) for v in np.asarray(best_x)]
    best_score = _finite_or_none(ex["best_y"])

    # breakdown + feasibility at the incumbent
    breakdown = None
    feasible = False
    if best_x_list is not None:
        raw = built.evaluate(np.asarray(best_x_list, dtype=float))
        breakdown = _json_safe(raw)
        feasible = bool(raw.get("feasible", False))
        if built.is_constrained:
            # for constrained problems, "feasible" means a feasible incumbent
            # exists (constraints satisfied), not merely a solvable point
            feasible = best_x is not None

    # per-eval log + histories
    eval_x = _json_safe(ex["X"])
    eval_y = _json_safe(ex["y"])
    history = _json_safe(ex["best_so_far"])
    eval_g = _json_safe(ex["g"]) if built.is_constrained and ex["g"] is not None \
        else None
    best_g = _finite_or_none(ex["best_g"]) if built.is_constrained else None
    n_feasible = (int(ex["n_feasible"])
                  if built.is_constrained and ex["n_feasible"] is not None
                  else None)
    failures = _json_safe(ex["failures"]) if ex["failures"] is not None else None

    ts = time.strftime("%Y%m%dT%H%M%S") + f"_{uuid.uuid4().hex[:8]}"
    result = RunResult(
        problem_name=cfg.problem_name,
        optimiser=cfg.optimiser,
        budget=int(cfg.budget),
        seed=int(cfg.seed),
        medium=built.medium,
        is_constrained=built.is_constrained,
        param_labels=list(built.param_labels),
        dim=built.dim,
        bounds=[[float(lo), float(hi)]
                for lo, hi in (built.bounds if pin is None else pin.bounds)],
        best_x=best_x_list,
        best_score=best_score,
        feasible=feasible,
        n_feasible=n_feasible,
        best_g=best_g,
        history=history,
        n_evals=int(len(ex["y"])),
        eval_x=eval_x,
        eval_y=eval_y,
        eval_g=eval_g,
        breakdown=breakdown,
        # the family's own name for what it maximised, resolved at
        # the ONE place the breakdown and the record meet. Stored so
        # a reader never has to re-derive it, and so a run archived
        # today still says what its number is in a year.
        score_units=score_units(breakdown),
        wall_time_s=float(time.time() - t0),
        timestamp=ts,
        config=cfg.to_dict(),
        acqf=(ex["acqf"] if not built.is_constrained else None),
        failures=failures,
        bo_iters=bo_iters or None,
        pinned=(dict(pin.fixed) if pin is not None else None),
        searched_dim=(pin.dim if pin is not None else None),
        # the split that RAN. Re-derived from the same three inputs _dispatch
        # passed (cfg.budget, the SEARCHED problem's dim, the bo_n_init flag),
        # so it is the same call, not a second guess at it.
        # ...and on a RESUMED run there is no initial design at all: the
        # first number is what it inherited, the second what it flew.
        bo_split=([n_prior, int(cfg.budget) - n_prior] if prior is not None
                  else (list(_bo_split(int(cfg.budget), search.dim,
                                       _bo_n_init(cfg)))
                        if cfg.optimiser == "bo" else None)),
        resumed=(n_prior or None),
        # what the memo did, on the record, beside the wall clock it changes
        eval_cache=(memo.stats() if memo is not None else None),
        handoff=_handoff_report(cfg, res, search),
        n_screened=(None if ex.get("n_screened") is None
                    else int(ex["n_screened"])),
        n_rescue=(None if ex.get("n_rescue") is None
                  else int(ex["n_rescue"])),
        n_rescue_blind=(None if ex.get("n_rescue_blind") is None
                        else int(ex["n_rescue_blind"])),
        box_outside_validity=(rows_outside_validity(built) or None),
    )

    if results_dir is not None:
        d = Path(results_dir)
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{ts}.json"
        result.path = str(path)                       # set first so the file
        path.write_text(json.dumps(result.to_dict(), indent=2))  # records it
    return result


def _pareto_settings(flags: dict | None) -> dict:
    """``{acqf, q, criteria}`` for a front run, read off its flags.

    One reader, so the acquisition a front reports is the acquisition it used.
    The keys are refused on every other objective by
    :data:`AIRFOIL_REFERENCE_FLAG_OWNERS`, which is checked in
    :func:`_airfoil_goal_settings` on every build.
    """
    d = flags or {}
    return {
        "acqf": str(d.get("airfoil_pareto_acqf") or PARETO_DEFAULT_ACQF),
        "q": int(d.get("airfoil_pareto_q") or PARETO_DEFAULT_Q),
        "criteria": d.get("airfoil_pareto_criteria"),
        # `airfoil_run_config` sets this on every front, for the reason given
        # there: the seed is the hypervolume reference point. Read rather than
        # assumed, so a config assembled by hand says what it means.
        "warm_start": bool(d.get(BO_WARM_START_FLAG, False)),
    }


def _best_front_point(rows: list) -> tuple:
    """``(x, J)`` of the front point with the highest PLAIN composite J.

    A front has no best point — that is the whole content of
    ``RESULTS_FRONT_REACHABILITY.md``, where 41 of 42 seeds carry a design no
    scalarisation dominates. But a caller holding a :class:`RunResult` needs
    ONE ``best_x``, and the honest way to pick one is to say out loud which
    scalar picked it. This one is the plain composite J the rest of the project
    reports in, computed per row by ``pareto_evaluation`` — not the front's own
    rank order (which uses only the three searched criteria renormalised), and
    not a rule invented for this function. Ties keep the earlier row, which is
    the better-ranked one.

    ``(None, None)`` when the front is empty or nothing on it could be scored:
    a run that found no scoreable non-dominated design has no incumbent, and
    inventing one is the mistake the goal composite already refuses to make.
    """
    best_x = best_j = None
    for r in rows or []:
        j = r.get("composite")
        if j is None:
            continue
        j = float(j)
        if best_j is None or j > best_j:
            best_x, best_j = list(r.get("x") or []), j
    return (best_x or None), best_j


def _pareto_run_result(cfg: RunConfig, built, t0: float, *,
                       progress_cb: Callable | None = None,
                       results_dir: str | Path | None = None) -> RunResult:
    """Drive the front search and package it as a :class:`RunResult`.

    Three fields do NOT mean what they mean on a scalar run, and each is
    documented where it is set rather than left for a reader to discover:

    * ``eval_y`` holds the per-evaluation objective VECTORS (one row per
      evaluation, one column per searched criterion), because that is what was
      measured. A scalar there would have to be invented.
    * ``history`` is the running HYPERVOLUME against the seed — the genuine
      monotone best-so-far of a multi-objective search, and the number
      ``RESULTS_FRONT_REACHABILITY.md`` compares arms on.
    * ``best_score`` is the plain composite J at ``best_x``, so it is NOT
      ``max(eval_y)``; see :func:`_best_front_point`.
    """
    rep, hist = _pareto_from_config(cfg, progress_cb=progress_cb,
                                    **_pareto_settings(cfg.flags))
    best_x, best_j = _best_front_point(rep.get("front") or [])

    breakdown = None
    if best_x is not None:
        breakdown = _json_safe(built.evaluate(np.asarray(best_x, dtype=float)))

    Y = np.asarray(hist.Y, dtype=float)
    res = rep["result"]
    ts = time.strftime("%Y%m%dT%H%M%S") + f"_{uuid.uuid4().hex[:8]}"
    result = RunResult(
        problem_name=cfg.problem_name,
        optimiser=cfg.optimiser,
        budget=int(cfg.budget),
        seed=int(cfg.seed),
        medium=built.medium,
        is_constrained=True,
        param_labels=list(built.param_labels),
        dim=built.dim,
        bounds=[[float(lo), float(hi)] for lo, hi in built.bounds],
        best_x=best_x,
        best_score=best_j,
        # a front run is feasible iff it produced a non-dominated design that
        # could be scored — the same test every constrained run applies to its
        # incumbent
        feasible=best_x is not None,
        n_feasible=int(res["n_feasible"]),
        best_g=None,
        history=[float(v) for v in res["hypervolume_trace"]],
        n_evals=int(res["n_evals"]),
        eval_x=_json_safe(np.asarray(hist.X, dtype=float)),
        eval_y=_json_safe(Y),
        eval_g=None,
        breakdown=breakdown,
        # the family's own name for what it maximised, resolved at
        # the ONE place the breakdown and the record meet. Stored so
        # a reader never has to re-derive it, and so a run archived
        # today still says what its number is in a year.
        score_units=score_units(breakdown),
        wall_time_s=float(time.time() - t0),
        timestamp=ts,
        config=cfg.to_dict(),
        acqf=str(rep["conditions"]["acqf"]),
        failures=None,
        bo_iters=None,
        pinned=None,
        searched_dim=None,
        bo_split=[int(res["n_init"]),
                  int(cfg.budget) - int(res["n_init"])],
        front=rep,
    )
    if results_dir is not None:
        d = Path(results_dir)
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{ts}.json"
        result.path = str(path)
        path.write_text(json.dumps(result.to_dict(), indent=2))
    return result


def partial_result(cfg: RunConfig, records, wall_time_s: float = 0.0,
                   bo_iters=None,
                   results_dir: str | Path | None = None,
                   stop_reason: str | None = None,
                   eval_cache: dict | None = None,
                   resumed: int | None = None) -> RunResult:
    """Assemble a RunResult from the per-eval records of an INTERRUPTED run.

    A cancelled search has still paid for every evaluation it made — on the
    XFOIL problems that can be an hour of solver time — and the incumbent at
    the moment of cancellation is a real, evaluated design. Throwing it away
    because the optimiser never returned is a bookkeeping decision, not a
    physics one, so the shell reconstructs the same RunResult shape from the
    progress log and flags it ``partial=True``.

    ``records`` are the payloads a rich ``progress_cb`` receives (``x``,
    ``f``, ``g``, ``feasible``); rows without an ``x`` are ignored. On a
    RESUMED run (``run(resume=…)``) they open with the evaluations the run
    inherited, and ``resumed`` says how many — a stopped continuation must
    not hand back a record shorter than the run it continues. Costs
    one evaluation to rebuild the breakdown at the incumbent (a cache hit
    for the section problems).

    The result is NOT budget-comparable with a completed run: it stopped
    where the user stopped it. ``partial`` marks that in the record so no
    later comparison can silently treat it as a finished search.
    """
    t0 = time.time()
    if cfg.problem_name not in PROBLEM_SPECS:
        raise ValueError(f"unknown problem {cfg.problem_name!r}; "
                         f"choices: {problem_names()}")
    spec = PROBLEM_SPECS[cfg.problem_name]
    _flags, _dropped = sanitise_flags(cfg.problem_name, cfg.flags)
    built = spec.build(cfg.mission_kwargs or {}, _flags,
                       cfg.bounds_overrides)
    # the records already carry FULL design vectors even on a pinned run
    # (_ProgressCounter expands before reporting), so the pin is needed here
    # only to say what the box searched was
    pin = _pin_of(built, getattr(cfg, "pinned", None))

    rows = [r for r in (records or [])
            if isinstance(r, dict) and r.get("x") is not None]
    X = [[float(v) for v in r["x"]] for r in rows]
    Y = [(float(r["f"]) if r.get("f") is not None and
          math.isfinite(float(r["f"])) else -math.inf) for r in rows]
    G = ([[float(v) for v in (r.get("g") or [])] for r in rows]
         if built.is_constrained else None)

    best_i = None
    if X:
        if built.is_constrained:
            ok = [i for i, g in enumerate(G or []) if g and min(g) >= 0.0]
            pool = ok or []
        else:
            pool = [i for i, v in enumerate(Y) if math.isfinite(v)]
        if pool:
            best_i = max(pool, key=lambda i: Y[i])

    # best-so-far over the SAME feasibility rule the runners use
    history, run_best = [], -math.inf
    for i in range(len(X)):
        ok_i = (not built.is_constrained
                or (G is not None and G[i] and min(G[i]) >= 0.0))
        if ok_i and Y[i] > run_best:
            run_best = Y[i]
        history.append(run_best if math.isfinite(run_best) else None)

    best_x_list = X[best_i] if best_i is not None else None
    breakdown, feasible = None, False
    if best_x_list is not None:
        raw = built.evaluate(np.asarray(best_x_list, dtype=float))
        breakdown = _json_safe(raw)
        feasible = (True if built.is_constrained
                    else bool(raw.get("feasible", False)))

    n_feasible = (sum(1 for g in (G or []) if g and min(g) >= 0.0)
                  if built.is_constrained else None)
    best_g = (min(G[best_i]) if built.is_constrained and best_i is not None
              and G and G[best_i] else None)

    ts = time.strftime("%Y%m%dT%H%M%S") + f"_{uuid.uuid4().hex[:8]}"
    result = RunResult(
        problem_name=cfg.problem_name,
        optimiser=cfg.optimiser,
        budget=int(cfg.budget),
        seed=int(cfg.seed),
        medium=built.medium,
        is_constrained=built.is_constrained,
        param_labels=list(built.param_labels),
        dim=built.dim,
        bounds=[[float(lo), float(hi)]
                for lo, hi in (built.bounds if pin is None else pin.bounds)],
        best_x=best_x_list,
        best_score=_finite_or_none(Y[best_i]) if best_i is not None else None,
        feasible=feasible,
        n_feasible=n_feasible,
        best_g=_finite_or_none(best_g) if best_g is not None else None,
        history=_json_safe(history),
        n_evals=len(X),
        eval_x=_json_safe(X),
        eval_y=_json_safe([v if math.isfinite(v) else None for v in Y]),
        eval_g=_json_safe(G) if G else None,
        breakdown=breakdown,
        # the family's own name for what it maximised, resolved at
        # the ONE place the breakdown and the record meet. Stored so
        # a reader never has to re-derive it, and so a run archived
        # today still says what its number is in a year.
        score_units=score_units(breakdown),
        wall_time_s=float(wall_time_s or (time.time() - t0)),
        timestamp=ts,
        config=cfg.to_dict(),
        failures=None,
        bo_iters=_json_safe(list(bo_iters)) if bo_iters else None,
        partial=True,
        stop_reason=stop_reason,
        pinned=(dict(pin.fixed) if pin is not None else None),
        searched_dim=(pin.dim if pin is not None else None),
        # Same re-derivation as the completed path, and it belongs here for a
        # sharper reason: a STOPPED BO run that left this field None was
        # indistinguishable from a Sobol run in the record -- exactly the
        # confusion bo_split exists to close, and the partial record is where
        # it bites, because a run cancelled during its Sobol phase has flown
        # no GP iterations at all. The split reported is the split the run was
        # DISPATCHED with, not how much of it was reached; n_evals against
        # bo_split[0] is what says whether the GP phase was ever entered.
        bo_split=([int(resumed), int(cfg.budget) - int(resumed)]
                  if resumed else
                  (list(_bo_split(int(cfg.budget),
                                  pin.dim if pin is not None else built.dim,
                                  _bo_n_init(cfg)))
                   if cfg.optimiser == "bo" else None)),
        # how many of the rows above this run INHERITED rather than flew (a
        # stopped continuation still owns the evaluations it was handed)
        resumed=(int(resumed) or None) if resumed else None,
        # ...and the rows being DRAWN FROM and then refused, which is a
        # property of the built problem and not of how far the run got. A
        # stopped run is exactly when a user wants to know that the row they
        # widened is not the row the search used. ``n_screened`` cannot be
        # recovered here — the runner never returned its history — and stays
        # None rather than guessed at.
        box_outside_validity=(rows_outside_validity(built) or None),
        # ``{hits, misses, dir}`` of the run that was interrupted, when it had
        # a memo — the caller passes it because this function never sees the
        # search, only its log
        eval_cache=(dict(eval_cache) if eval_cache else None),
    )
    if results_dir is not None:
        d = Path(results_dir)
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{ts}_partial.json"
        result.path = str(path)
        path.write_text(json.dumps(result.to_dict(), indent=2))
    return result


def load_run(path: str | Path) -> dict:
    """Load a persisted RunResult back into a plain dict (the Compare page)."""
    return json.loads(Path(path).read_text())


def list_runs(results_dir: str | Path) -> list[dict]:
    """List persisted runs in ``results_dir`` as metadata dicts, oldest first.

    Each entry carries ``path`` plus the headline fields the Compare page
    tabulates; unparseable files are skipped.
    """
    d = Path(results_dir)
    if not d.exists():
        return []
    runs = []
    for p in sorted(d.glob("*.json")):
        try:
            data = json.loads(p.read_text())
        except (ValueError, OSError):
            continue
        runs.append({
            "path": str(p),
            "timestamp": data.get("timestamp", p.stem),
            "problem_name": data.get("problem_name"),
            "optimiser": data.get("optimiser"),
            "budget": data.get("budget"),
            "seed": data.get("seed"),
            "best_score": data.get("best_score"),
            "feasible": data.get("feasible"),
            "medium": data.get("medium"),
            "is_constrained": data.get("is_constrained"),
        })
    runs.sort(key=lambda r: r["timestamp"])
    return runs


# =====================================================================
# Post-run design views (presentation data for the GUI — no GUI imports)
# =====================================================================

def _nested_values(raw: dict):
    """Attribute values of the breakdown's own result objects (one level).

    A solver that composes physics on top of a lifting line stores that line
    as an attribute of its own result (``hydrofoil.HydrofoilResult.foil``)
    rather than in the breakdown, so the outer scan sees nothing spanwise.
    Named nothing solver-specific: any result-like value is unwrapped.
    """
    for v in raw.values():
        if v is None or isinstance(v, (str, bytes, bool, int, float, dict,
                                       list, tuple, np.ndarray)):
            continue
        nested = getattr(v, "__dict__", None)
        if isinstance(nested, dict):
            yield from nested.values()


class _MaskedPanels:
    """A panel-carrying result seen through a boolean mask (read-only view).

    Lets the exporter below stay one code path: it reads the same attribute
    names off this as off the solver's own result, and gets the masked
    arrays. Scalars (S, AR) pass straight through — they are references, not
    per-panel data.
    """

    # ...INCLUDING THE SURFACE MASKS THEMSELVES. ``is_second`` and
    # ``is_vertical`` were missing, so a reader that took one off a masked
    # result got a FULL-LENGTH boolean against short arrays — an index error
    # where the shapes are compared and a silent mis-selection where they are
    # not. They are listed for the same reason ``is_tail`` always was.
    # ...and the three ``VLMResult`` carries for a structural reader that does
    # not exist yet: ``vlm`` says of them "with Gamma this gives the PER-PANEL
    # side force rho V Gamma lz, which the total CY cancels by symmetry but a
    # structural model needs per side". Listed now, while the reason is
    # written down, rather than left to be found as a shape mismatch.
    _PER_PANEL = ("y", "z", "c", "cl", "Cl_y", "alpha_eff", "alpha_eff_y",
                  "alpha_i_y", "is_winglet", "is_tail", "is_second",
                  "is_vertical", "width", "x", "Gamma", "ly", "lz")

    def __init__(self, inner, mask: np.ndarray):
        self._inner = inner
        self._mask = np.asarray(mask, dtype=bool)

    def __getattr__(self, name):
        val = getattr(self._inner, name)
        if name in self._PER_PANEL and val is not None:
            arr = np.asarray(val)
            if arr.shape == self._mask.shape:
                return arr[self._mask]
        return val


def _split_surface_panels(v, geometry: dict, keep: np.ndarray, *,
                          mask_attr: str, key: str, name: str) -> np.ndarray:
    """Export a multi-surface VLM result's SECOND-surface panels as their own
    block, and return the mask of everything else.

    A nonplanar solver that carries more than one surface (wingtail.py's tail,
    tandemvlm.py's rear wing) returns ONE panel array covering all of them.
    Exporting it whole draws the second surface as a continuation of the
    first's spanwise polyline — a surface that does not exist. Two things go
    visibly wrong when it does, and the tandem pair shows both: its rear wing
    appears as extra span bolted to the front wing's tip, and the front wing's
    STARBOARD tip device and the rear wing's PORT one, adjacent in the
    concatenated array and both flagged as device panels, are drawn as ONE
    device running clean across the aircraft. Two winglets, joined by nothing
    but array order.

    So the second surface's panels leave under ``key`` — its own
    y/z/chord/Cl/device mask, plus the streamwise and vertical offsets that
    place it — and the caller keeps the rest for the first surface.
    """
    mask = getattr(v, mask_attr, None)
    if mask is None:
        return keep
    t_arr = np.asarray(mask, dtype=bool)
    if t_arr.shape != keep.shape or not t_arr.any():
        return keep
    t_arr = t_arr & keep          # never claim panels another block took
    y_arr = np.asarray(v.y, dtype=float)
    c_arr = np.asarray(v.c, dtype=float)
    x_arr = getattr(v, "x", None)
    z_arr = getattr(v, "z", None)
    # a DESIGNED tail can carry its own tip device, and those panels are as
    # nonplanar as the wing's. Two consequences: the tail's plane height is the
    # mean over its FLAT panels only (averaging the device in would float the
    # whole surface upwards), and the panel path itself leaves with the block,
    # so a view or a CAD export can loft the device instead of flattening it.
    wl_arr = getattr(v, "is_winglet", None)
    wl_t = None
    flat = t_arr
    if wl_arr is not None:
        wl_full = np.asarray(wl_arr, dtype=bool)
        if wl_full.shape == t_arr.shape:
            wl_t = wl_full[t_arr]
            own = t_arr & ~wl_full
            if wl_t.any() and own.any():
                flat = own
    block = {"name": name,
             "y": _json_safe(y_arr[t_arr]),
             "chord": _json_safe(c_arr[t_arr]),
             "x_offset": float(np.mean(np.asarray(x_arr, float)[flat]))
             if x_arr is not None else 0.0,
             "z_offset": float(np.mean(np.asarray(z_arr, float)[flat]))
             if z_arr is not None else 0.0}
    if z_arr is not None:
        block["z"] = _json_safe(np.asarray(z_arr, float)[t_arr])
    if wl_t is not None and wl_t.any():
        block["is_winglet"] = [bool(t) for t in wl_t]
    cl = getattr(v, "cl", None)
    if cl is not None:
        cl_arr = np.asarray(cl, dtype=float)
        if cl_arr.shape == t_arr.shape:
            block["Cl_y"] = _json_safe(cl_arr[t_arr])
    ae = getattr(v, "alpha_eff", None)
    if ae is not None:
        ae_arr = np.rad2deg(np.asarray(ae, dtype=float))
        if ae_arr.shape == t_arr.shape:
            block["alpha_eff_deg"] = _json_safe(ae_arr[t_arr])
    geometry[key] = block
    return keep & ~t_arr


def _split_tail_panels(v, geometry: dict, keep: np.ndarray) -> np.ndarray:
    """The TAIL panels of a wing+tail solve, as ``geometry["tail_surface"]``."""
    return _split_surface_panels(v, geometry, keep, mask_attr="is_tail",
                                 key="tail_surface", name="tail")


def _drop_vertical_panels(v, keep: np.ndarray) -> np.ndarray:
    """Take the FIN's (or the mast's) panels OUT of the wing's arrays.

    The third surface in the same panel array, and the one nothing split off.
    :func:`_split_surface_panels`' docstring describes exactly what that does
    — "exporting it whole draws the second surface as a continuation of the
    first's spanwise polyline, a surface that does not exist" — and a VERTICAL
    surface is the worst case of it: its stations sit at y = 0 with z running
    from the fin root to the fin tip, so the wing's polyline ends by turning
    ninety degrees and climbing the fin. Measured on `tail [free cant]` at the
    box centre with 12 deg of dihedral, once the lateral deck is armed
    (``handling_level=3``): 40 stations become 52, the wing's reported z max
    goes 1.039 -> 1.540 against a fin standing z_root 0.500 + height 1.044,
    and the lofted wing follows it — into a fin that ``cad.fin_surface`` is
    ALSO drawing from ``geometry["fin"]``, so the surface appears twice.

    DROPPED rather than split into a block of its own. Both solvers already
    treat these panels as not-the-aeroplane's-lifting-surfaces
    (``wingtail.py``'s ``solid`` mask and ``tandemvlm.py``'s: "the deck is
    free of charge to the objective"), and the fin already leaves the run as
    ``geometry["fin"]`` — the block the drag was charged on, the report
    prints, the CAD lofts and stage 5 rebuilds. A second, panel-level
    description of the same surface would be a second author of it, which is
    the failure ``cad.fin_surface`` exists to prevent.

    Runs FIRST, before the tail and rear-wing splits, so neither can be handed
    a vertical panel to claim — they are disjoint in the lattice today
    (``vlm.VLM`` writes zeros into ``tl_panel``/``sd_panel`` for the vertical
    block) and this does not rely on their staying that way.
    """
    mask = getattr(v, "is_vertical", None)
    if mask is None:
        return keep
    arr = np.asarray(mask, dtype=bool)
    if arr.shape != keep.shape or not arr.any():
        return keep
    return keep & ~arr


def _split_second_panels(v, geometry: dict, keep: np.ndarray) -> np.ndarray:
    """The REAR WING of a nonplanar tandem, as ``geometry["second_surface"]``.

    Its own key rather than ``tail_surface``: a tandem's rear surface is a
    WING — it carries the pair's area split and its own tip device, it is not
    trimmed by an incidence, and the caption machinery that reads a ``tail``
    block would name it an elevator.
    """
    return _split_surface_panels(v, geometry, keep, mask_attr="is_second",
                                 key="second_surface", name="rear")


def _fill_llt_geometry(candidates, geometry: dict) -> bool:
    """Export the first lifting-line-shaped candidate's arrays into geometry.

    Matches on 1-D ``y``/``c`` of equal shape; returns whether one was found.
    A result carrying further SURFACES in the same panel array — the nonplanar
    wing+tail solver's tail, the nonplanar tandem's rear wing — has those
    split off first, so what stays behind is the first surface alone (see
    :func:`_split_surface_panels`).
    """
    for v in candidates:
        y = getattr(v, "y", None)
        c = getattr(v, "c", None)
        if y is None or c is None:
            continue
        y_arr = np.asarray(y, dtype=float)
        c_arr = np.asarray(c, dtype=float)
        if y_arr.ndim != 1 or y_arr.shape != c_arr.shape:
            continue
        keep = _drop_vertical_panels(v, np.ones(y_arr.shape, dtype=bool))
        keep = _split_tail_panels(v, geometry, keep)
        keep = _split_second_panels(v, geometry, keep)
        if not keep.all():
            v = _MaskedPanels(v, keep)
            y_arr, c_arr = y_arr[keep], c_arr[keep]
        geometry["y"] = _json_safe(y_arr)
        geometry["chord"] = _json_safe(c_arr)
        # Nonplanar solvers (the VLM winglet modes) place panels OUT of the
        # wing plane and mark the winglet strips. Both must be exported or a
        # shell can only draw a flat span extension — which is not what was
        # flown: at cant < 90 deg the winglet stations also sit beyond b/2 in
        # y, so without z they masquerade as extra planar span.
        z = getattr(v, "z", None)
        if z is not None:
            z_arr = np.asarray(z, dtype=float)
            if z_arr.shape == y_arr.shape:
                geometry["z"] = _json_safe(z_arr)
        wl = getattr(v, "is_winglet", None)
        if wl is not None:
            wl_arr = np.asarray(wl, dtype=bool)
            if wl_arr.shape == y_arr.shape and wl_arr.any():
                geometry["is_winglet"] = [bool(t) for t in wl_arr]
        cl = getattr(v, "Cl_y", None)
        if cl is None:
            cl = getattr(v, "cl", None)          # VLMResult naming
        if cl is not None:
            geometry["Cl_y"] = _json_safe(np.asarray(cl, dtype=float))
        for out_key, attr in (("alpha_eff_deg", "alpha_eff_y"),
                              ("alpha_i_deg", "alpha_i_y")):
            a = getattr(v, attr, None)
            if a is not None:
                geometry[out_key] = _json_safe(
                    np.rad2deg(np.asarray(a, dtype=float)))
        if "alpha_eff_deg" not in geometry:
            a = getattr(v, "alpha_eff", None)    # VLMResult naming [rad]
            if a is not None:
                geometry["alpha_eff_deg"] = _json_safe(
                    np.rad2deg(np.asarray(a, dtype=float)))
        S = getattr(v, "S", None)
        AR = getattr(v, "AR", None)
        if S is not None:
            geometry.setdefault("S", float(S))
            if AR is not None:
                geometry.setdefault("b",
                                    float(math.sqrt(float(AR) * float(S))))
        return True
    return False


def design_report(cfg: RunConfig, x) -> dict:
    """Re-evaluate a design point and return breakdown + geometry arrays.

    The persisted RunResult carries only the JSON-safe scalar breakdown (the
    solver result objects are dropped by :func:`_json_safe`). This helper
    rebuilds the problem from ``cfg`` exactly as :func:`run` does, evaluates
    ``x`` once, and additionally extracts the spanwise arrays every LLT-backed
    breakdown carries (``y``/``chord``/``Cl_y``/``alpha_eff_deg``) plus the
    planform scalars (``b``/``S``/``taper``…) — everything a shell needs to
    draw planforms and distributions without touching physics itself.

    Costs one physics evaluation (an XFOIL call for the section problems).
    """
    spec = PROBLEM_SPECS[cfg.problem_name]
    _flags, _dropped = sanitise_flags(cfg.problem_name, cfg.flags)
    built = spec.build(cfg.mission_kwargs or {}, _flags,
                       cfg.bounds_overrides)
    raw = built.evaluate(np.asarray(x, dtype=float))
    if not isinstance(raw, dict):
        raw = {}

    geometry: dict = {}
    # spanwise arrays: first breakdown value that looks like an LLT result
    _fill_llt_geometry(raw.values(), geometry)
    # tandem: a TandemResult-like value carries per-surface LLT results —
    # export both as geometry["surfaces"] with the stagger offsets so a shell
    # can draw the two-wing system (front at the origin, rear at (dx, dz)).
    #
    # The TAIL problem shares that container but NOT the tandem problem's
    # dx/dz attributes: its rear surface sits at the SOLVED moment arm l_t
    # (signed — a canard is upstream, at negative x) and at the height
    # dz_for(arm) that layout implies. Reading dx off the problem therefore
    # placed every tail at x = 0, inside the wing, and rendered a canard
    # identically to an aft tail; the breakdown's own l_t / dz_tail are the
    # authority when they are present.
    if "y" not in geometry:
        for v in raw.values():
            fr, rr = getattr(v, "front", None), getattr(v, "rear", None)
            if fr is None or rr is None:
                continue
            rear_x = raw.get("l_t")
            rear_z = raw.get("dz_tail")
            if not isinstance(rear_x, (int, float, np.integer, np.floating)):
                rear_x = getattr(built.problem, "dx", 0.0)
            if not isinstance(rear_z, (int, float, np.integer, np.floating)):
                rear_z = getattr(built.problem, "dz", 0.0)
            is_tail = isinstance(raw.get("tail_type"), str)
            names = (("wing", "tail") if is_tail else ("front", "rear"))
            offsets = {names[0]: (0.0, 0.0),
                       names[1]: (float(rear_x), float(rear_z))}
            surfs = []
            for nm, o in ((names[0], fr), (names[1], rr)):
                y_ = getattr(o, "y", None)
                c_ = getattr(o, "c", None)
                if y_ is None or c_ is None:
                    continue
                d = {"name": nm,
                     "y": _json_safe(np.asarray(y_, dtype=float)),
                     "chord": _json_safe(np.asarray(c_, dtype=float)),
                     "x_offset": offsets[nm][0], "z_offset": offsets[nm][1]}
                cl = getattr(o, "Cl_y", None)
                if cl is not None:
                    d["Cl_y"] = _json_safe(np.asarray(cl, dtype=float))
                ae = getattr(o, "alpha_eff_y", None)
                if ae is not None:
                    d["alpha_eff_deg"] = _json_safe(
                        np.rad2deg(np.asarray(ae, dtype=float)))
                surfs.append(d)
            if len(surfs) == 2:
                geometry["surfaces"] = surfs
                b_attr = getattr(built.problem, "b", None)
                if isinstance(b_attr, (int, float)) and b_attr:
                    geometry.setdefault("b", float(b_attr))
            break
    # LAST resort: a solver that WRAPS its lifting line one level down
    # (hydrofoil.HydrofoilResult carries it under ``.foil``) exported no
    # spanwise arrays at all, so a shell could only redraw the planform from
    # the design labels — no loading, no real chord distribution. Unwrapped
    # only after the two branches above have failed, or a tandem/tail would
    # match ITS front surface here and lose the two-surface export.
    if "y" not in geometry and "surfaces" not in geometry:
        _fill_llt_geometry(_nested_values(raw), geometry)
    # winglet scalars (h_frac / cant / height / planform area) exactly as the
    # physics reported them — a shell needs cant to draw the surface, and the
    # dropped-winglet rule (vlm.MIN_WINGLET_FRAC) means the flown geometry can
    # differ from the requested one, so "is_winglet" above is the authority.
    wl_info = raw.get("winglet")
    if isinstance(wl_info, dict):
        geometry["winglet"] = {k: float(v) for k, v in wl_info.items()
                               if isinstance(v, (int, float, np.integer,
                                                 np.floating))
                               and not isinstance(v, bool)}
        # the transition SHAPE is a string, so it would fall out of the
        # numeric filter above — and it is exactly what a reader needs to
        # know when a blend fraction is non-zero (geometry.BLEND_SHAPES)
        if isinstance(wl_info.get("blend_shape"), str):
            geometry["winglet_blend_shape"] = wl_info["blend_shape"]

    # THE MOUNT, as the picture needs it. carmount.py priced this layout from
    # the day it was written and no figure has ever shown it: a swan-neck car
    # wing was drawn floating, with the one body that distinguishes it from
    # the plate-borne wing invisible. Everything here is a length or a count
    # the physics already solved — the station off the spec and the flown
    # span, the pylon's length off the flown ride height — so the drawing and
    # the beam cannot disagree about where the wing is held.
    # TYPE-CHECKED, not truth-checked. ``mount_spec`` is not one thing across
    # the car families: ``carwing`` carries a real ``carmount.MountSpec`` (or
    # None), while ``endplate.CarWingEndplateProblem`` spells the same
    # attribute as the two-valued MOUNTS LOOKUP — a plain dict with
    # "supports"/"n_struts"/"label" in it and no station, no side and no
    # pylon. A `is not None` gate therefore read a dict's ``.kind`` and threw
    # AttributeError on every report of a designed-endplate run. It only
    # surfaced when that family became what the track card opens on.
    from . import carmount as _carmount

    spec_mount = getattr(built.problem, "mount_spec", None)
    span_m = raw.get("b_m")
    if isinstance(spec_mount, _carmount.MountSpec) and isinstance(
            span_m, (int, float, np.integer, np.floating)):
        ride = raw.get("ride_height_m")
        mount: dict = {
            "kind": str(spec_mount.kind),
            "side": str(spec_mount.side),
            "n_pylons": int(spec_mount.n_pylons or 0),
            "n_sheets": int(spec_mount.n_sheets or 0),
            "x_attach_frac": float(spec_mount.x_attach_frac),
            "chord_m": float(spec_mount.pylon_chord_m),
            "tc": float(spec_mount.pylon_tc),
            "deck_height_m": float(spec_mount.deck_height_m),
            "stations_m": [float(v) for v in
                           spec_mount.support_stations_m(float(span_m))],
        }
        if isinstance(ride, (int, float, np.integer, np.floating)):
            # the length the STRUT is, at the ride height actually flown —
            # not the ride height, which is what carwing charges its drag
            # over (carmount.pylon_length_m says so in full)
            mount["length_m"] = float(
                spec_mount.pylon_length_m(float(ride)))
            mount["ride_height_m"] = float(ride)
        geometry["mount"] = mount

    # tail/elevator scalars: everything needed to DRAW the layout that was
    # flown — the signed arm, the height, the panel area/span, the control
    # type and, for an elevator, the hinge position and the deflection the
    # trim solve landed on. Without these a shell can only draw the wing,
    # which is exactly what it used to do.
    # A solver can fly a second surface WITHOUT naming a layout: the hydrofoil
    # elevator (hydrotail.py) reports its arm, area and incidence but no
    # ``tail_type``, and keying this block on that string alone left its
    # surface with panels but no incidence, no arm and no CG/NP stations — so
    # every view drew the foil alone. The panel block is the trigger too.
    _fin_args: dict | None = None
    if isinstance(raw.get("tail_type"), str) or "tail_surface" in geometry:
        tail: dict = {"type": raw.get("tail_type") or "elevator",
                      "control": raw.get("control", "stabilator")}
        for key, src in (("l_t", "l_t"), ("dist_m", "dist_from_wing_m"),
                         ("dz_m", "dz_tail"), ("S_t", "S_t"), ("b_t", "b_t"),
                         ("S_lift", "S_lift"), ("dihedral_deg",
                                                "dihedral_deg"),
                         ("i_t_deg", "i_t_deg"), ("delta_e_deg",
                                                  "delta_e_deg"),
                         ("tau_elevator", "tau_elevator"), ("mac", "mac"),
                         ("x_cg", "x_cg"), ("x_np", "x_np"), ("SM", "SM")):
            val = raw.get(src)
            if isinstance(val, (int, float, np.integer, np.floating)) \
                    and not isinstance(val, bool):
                tail[key] = float(val)
        # ...and the HEIGHT under its own name. The air tail reports it as
        # ``dz_tail``; the hydrofoil's elevator hangs BELOW the foil and
        # reports the same quantity as ``z_t`` (signed, negative down), which
        # left every caption saying "h 0.00 m" about a surface 60 mm under
        # the foil — the one number the arm-and-height layout is drawn from.
        if "dz_m" not in tail:
            zt = raw.get("z_t")
            if isinstance(zt, (int, float, np.integer, np.floating)) \
                    and not isinstance(zt, bool):
                tail["dz_m"] = float(zt)
        # the elevator chord fraction is a CONFIGURATION choice, not a solved
        # quantity, so it lives on the problem rather than in the breakdown
        ecf = getattr(built.problem, "elevator_chord_frac", None)
        if isinstance(ecf, (int, float)) and tail["control"] == "elevator":
            tail["elevator_chord_frac"] = float(ecf)
        # WHICH WAY UP the surface flies its section. A stabiliser that
        # pushes down mounts its camber the other way (tail.tail_polar), and
        # anything that draws or lofts it has to be told, or the picture
        # contradicts the polar the answer was solved on.
        inv = raw.get("tail_section_inverted")
        if isinstance(inv, (bool, np.bool_)):
            tail["section_inverted"] = bool(inv)
        geometry["tail"] = tail

        # ---- THE VERTICAL SURFACE, from the design rather than from the
        # rebuild. The drag book has been sizing a fin by volume coefficient
        # (fin.size_fin) while ``flightmodel`` sized a different one from the
        # span and the mac, so the surface that decided directional stability
        # was never the surface whose drag was charged, and neither was in
        # the report. It is now, for every layout that HAS one: a V-tail gets
        # no block, which is how a rebuild knows not to draw one.
        # b and S come from the PROBLEM, because that is what
        # ``tail.evaluate_tail`` hands ``vtail_cd0`` when it charges the fin's
        # drag. Reading them from anywhere else would make the fin reported
        # differ from the fin charged, which is the defect this block exists
        # to close — if the pair is ever wrong for a size-free family it is
        # wrong in both places, and there is one line to fix.
        # ...for the AIR LAYOUTS THAT HAVE ONE, and no others. A hydrofoil
        # reports its elevator under the same block with ``type="elevator"``
        # and has no fin at all — its vertical is the strut, which is a
        # ventral surface between the wing and the tail, not a Raymer
        # volume-coefficient tail. It is left to the mast block below.
        #
        # A CANARD IS IN, AND WAS NOT. It was excluded on the grounds that
        # "a canard has an aft fin whose arm is NOT the (negative,
        # upstream) surface station this block carries" — but the arm a fin
        # is placed at is ``fin.fin_arm``, the UNSIGNED separation, which is
        # exactly what ``wingtail._lateral_fin`` sizes and places the
        # solver's own vertical at (``v["dist"]``) and exactly what the drag
        # book charges: ``cd0_fin`` is 0.00088485 on a canard and on the
        # conventional twin, to every digit. So the surface was charged,
        # flown and weighed and then reported as ABSENT, which made stage
        # 1's fin switch a no-op on every canard: the report stated no
        # vertical surface either way, ``session.spiral_dihedral`` answered
        # ``no_fin``, and the rebuild invented one.
        if str(tail["type"]) in ("conventional", "t_tail", "v_tail",
                                 "canard"):
            _fin_args = dict(
                b=getattr(built.problem, "b", raw.get("b")),
                S=getattr(built.problem, "S", raw.get("Sref", raw.get("S"))),
                l_t=_fin.fin_arm(dist_m=tail.get("dist_m"),
                                 l_t=tail.get("l_t")),
                tail_type=str(tail["type"]),
                # WHERE THE FIN'S ROOT IS, and a T-TAIL IS THE EXCEPTION.
                # Every other layout hangs its tailplane off the body and
                # stands the fin beside it, so the fin's root is the tail's
                # own height. A T-tail is the opposite arrangement: the
                # tailplane sits ON TOP OF THE FIN, so the fin runs from the
                # BODY up to it. Taking dz for both put the fin's root at the
                # tailplane and its whole span above it — which is exactly
                # the reported "a T-tail is a conventional tail just raised".
                #
                # The two heights already agree: tail.tail_height gives a
                # T-tail dz = sqrt(AR_VT * S_vt), which IS the height a
                # volume-coefficient fin at that aspect ratio has. So the fin
                # spans body-to-tailplane exactly, and nothing needed
                # resizing — only the root was in the wrong place.
                z_root=(0.0 if str(tail["type"]) == "t_tail"
                        else float(tail.get("dz_m", 0.0) or 0.0)),
                # ...and, for the same layout, the height it has to REACH.
                # The root above put the fin's foot on the body; without
                # this its TIP can still stop short of the tailplane,
                # because tail.tail_height floors the surface at the
                # kernel's DZ_FRAC*b clearance and the volume coefficient
                # never hears about that floor (fin.size_fin's min_height).
                min_height=(float(tail.get("dz_m") or 0.0)
                            if str(tail["type"]) == "t_tail" else None))

    # ---- ...and a TANDEM's, whose arm is the STAGGER. The pair carries no
    # ``tail`` block at all, so without this the one family the user asked
    # for by name would be the one still flying an invented fin. Referenced
    # to the pair's TOTAL area, which is what its own solver uses.
    #
    # THE PAIR IS RECOGNISED THE WAY ``flightmodel`` RECOGNISES IT: a listed
    # front/rear pair (the lifting-line family) OR a ``second_surface``
    # block named "rear" (the NONPLANAR family, which lists no surfaces at
    # all). Asking only the first question is how the nonplanar pair came to
    # be the tandem still flying an invented fin while its lifting-line twin
    # flew the one the drag book charges — two families, one aeroplane, two
    # different vertical surfaces.
    elif "dx" in raw and (
            any(s.get("name") == "rear"
                for s in geometry.get("surfaces") or [])
            or str((geometry.get("second_surface") or {}).get("name") or "")
            == "rear"):
        #
        # ...AND IT STANDS ON A BOOM AFT OF THE REAR WING.
        # ``fin.tandem_fin_station`` is the one author of that station; both
        # tandem engines charge the drag on the arm it returns, so the fin
        # reported here is the fin paid for. The BOOM comes off the problem
        # and not off a default here: a run that stated one and a report
        # that assumed another would describe a different aeroplane from the
        # one flown, which is the whole reason that station has one author.
        _dx = raw.get("dx")
        _x_qc, _z_root = _fin.tandem_fin_station(
            float(_dx) if isinstance(_dx, (int, float, np.integer,
                                           np.floating))
            and not isinstance(_dx, bool) else 0.0,
            float(raw.get("dz", 0.0) or 0.0),
            getattr(built.problem, "fin_boom_m", None))
        _fin_args = dict(
            b=raw.get("b", getattr(built.problem, "b", None)),
            S=raw.get("Sref", getattr(built.problem, "S_total", None)),
            l_t=(_x_qc if _dx is not None else None), tail_type="tandem",
            z_root=float(_z_root))

    # ...AND SAY SO WHEN THERE IS NONE, which is not the same as saying
    # nothing. This block calls ``size_fin`` directly rather than
    # ``fin_for_layout``, so it is the one consumer the layout's own None
    # does not reach; and it is what ``flightmodel`` reads to decide whether
    # to draw a fin at all. A MISSING key means "this family does not report
    # a fin" and the rebuild assumes one — correct for a hydrofoil or a
    # canard, and catastrophic here: an aeroplane with the fin switched off
    # was rebuilt with an invented fin at 12 % of span, which is BIGGER than
    # the one it had been charged for, so switching the fin off raised
    # Cn_beta from 0.1114 to 0.1351. An explicit None is the third state
    # (``fin.states_no_fin``).
    if _fin_args is not None and not _fin.has_fin(built.problem):
        geometry["fin"] = None
    elif _fin_args is not None and all(
            isinstance(v, (int, float, np.integer, np.floating))
            and not isinstance(v, bool)
            for v in (_fin_args["b"], _fin_args["S"], _fin_args["l_t"])):
        try:
            _geo = _fin.size_fin(
                b=float(_fin_args["b"]), S=float(_fin_args["S"]),
                l_t=float(_fin_args["l_t"]),
                tail_type=_fin_args["tail_type"],
                z_root=_fin_args["z_root"],
                min_height=_fin_args.get("min_height"),
                # ...at the shape the DESIGN states, so the fin reported is
                # the fin charged and the fin flown. Read off the built
                # problem rather than the flags, for the reason the block
                # already reads b/S/l_t there: the problem is what the
                # solver actually used.
                **_fin.fin_law_kwargs(built.problem))
        except ValueError:
            _geo = None                # an arm no fin can be sized against
        if _geo is not None:
            blk = _geo.as_dict()
            # ...at the fin's OWN Reynolds number. It is a short-chord
            # surface — on the shipped tail design the wing mac is 1.47x the
            # fin chord — so choosing its section at the wing's Re chooses it
            # 47 % too high (fin.FinGeometry.reynolds).
            _V = getattr(built.problem, "V", None)
            if isinstance(_V, (int, float)) and not isinstance(_V, bool):
                blk["Re"] = _geo.reynolds(
                    float(_V), float(getattr(built.problem, "rho", RHO_SL)),
                    float(getattr(built.problem, "mu", MU_SL)))
            geometry["fin"] = blk

    # ---- ...AND A WATER CRAFT'S, WHICH IS THE MAST. Not a volume-
    # coefficient tail — ``fin.mast`` says why — and not a new surface
    # either: ``hydrofoil._mast_cd0`` has charged two sides of
    # ``depth x c_mast`` on every water run this package has ever published.
    # It simply was not in the report, so ``flightmodel`` invented a THIRD
    # vertical surface for the rebuild (0.144 m at 12 % of span, against a
    # real 0.3-0.6 m mast) and hung 100 % of a hydrofoil's Cn_beta on it.
    # One craft, one strut: charged here, stated here, flown from here.
    if "fin" not in geometry and "depth" in raw and hasattr(built.problem,
                                                            "c_mast"):
        if not _fin.has_fin(built.problem):
            geometry["fin"] = None      # asked, and the design says none
        else:
            _mast = _fin.mast(
                depth=float(raw["depth"]),
                chord=float(getattr(built.problem, "c_mast", 0.08)),
                # the SECTION stage 2.7 chose for it, where one was chosen
                tc=float(getattr(built.problem, "fin_tc", None)
                         or _fin.FIN_TC_DEFAULT),
                # WHERE IT STANDS. The elevator family places the strut on
                # the fuselage BETWEEN its two surfaces and reports the
                # station (``hydrotail.X_MAST_FRAC``); a single foil has no
                # fuselage to stand on, so 0 — at the foil — is still the
                # honest answer there. This is the yaw arm the flight
                # rebuild flies: before it was read from here, every foiling
                # craft was rebuilt with its only vertical surface AT the
                # CG's own reference point.
                x_qc=float(raw.get("x_mast", 0.0) or 0.0))
            if _mast is not None:
                blk = _mast.as_dict()
                _V = raw.get("V", getattr(built.problem, "V", None))
                if isinstance(_V, (int, float)) and not isinstance(_V, bool):
                    blk["Re"] = _mast.reynolds(
                        float(_V), float(getattr(built.problem, "rho",
                                                 RHO_SL)),
                        float(getattr(built.problem, "mu", MU_SL)))
                geometry["fin"] = blk

    # planform scalars: a Wing-like object under "wing", else the problem's own
    for src_obj in (raw.get("wing"), getattr(built.problem, "wing", None),
                    built.problem):
        if src_obj is None:
            continue
        for attr in ("b", "S", "taper", "twist_root_deg", "twist_tip_deg",
                     "tc", "sweep_deg"):
            val = getattr(src_obj, attr, None)
            if isinstance(val, (int, float, np.integer, np.floating)):
                geometry.setdefault(attr, float(val))

    # THE SECOND WING'S OWN SPAN, on the second wing's own block. A tandem
    # pair may fly a span per wing (tandem.TandemProblem.b_rear), and every
    # reader of ``geometry["b"]`` means the FRONT wing's — so a rear surface
    # drawn or lofted against that number would be stretched to the wrong
    # width and given the wrong twist at its tip (cad.second_wing_twist).
    b_rear = raw.get("b_rear")
    if isinstance(b_rear, (int, float, np.integer, np.floating)) \
            and not isinstance(b_rear, bool) and float(b_rear) > 0.0:
        blk = geometry.get("second_surface")
        if isinstance(blk, dict):
            blk.setdefault("b", float(b_rear))
        for sf in geometry.get("surfaces") or ():
            if isinstance(sf, dict) and sf.get("name") == "rear":
                sf.setdefault("b", float(b_rear))

    # WHICH WAY UP THIS GEOMETRY IS. A car family solves the vehicle
    # MIRRORED (geometry.MIRRORED_FRAME): model +z is the car's DOWN, so its
    # endplates run +z and a picture drawn in the model frame stands the car
    # on its roof. The solvers have declared that in the breakdown since the
    # family shipped and it stopped there — the ``geometry`` block is what
    # every drawer is handed, so the one dict that says where the surfaces
    # are said nothing about which way up they go. Carried out verbatim: a
    # frame is a property of the numbers, not a flag a shell may invent.
    if isinstance(raw.get("frame"), str):
        geometry["frame"] = raw["frame"]

    # a built problem may carry MORE constraints than the static spec knows
    # about (the section problem grows two twist margins in wing mode), so its
    # own labels win when it declares them.
    clabels = getattr(built.problem, "constraint_labels", None)
    return {"breakdown": _json_safe(raw), "geometry": geometry,
            "param_labels": list(built.param_labels),
            "constraint_labels": list(clabels or spec.constraint_labels),
            # keys this problem does NOT honour, carried out rather than
            # dropped in silence — the whole point of the split in
            # `sanitise_flags`. Empty on every well-formed config, which is
            # every config a shell builds now that both shells sanitise.
            "unhonoured_flags": list(_dropped)}


def dihedral_for_spiral(report: dict, **kw) -> dict:
    """How much WING DIHEDRAL turns this design's spiral — as a plain dict.

    The measurement itself is :func:`aerobo.flightmodel.dihedral_for_spiral`:
    about twenty rebuilds of the design at successive cants, scored by the
    criterion carrying the trim attitude, returning the SMALLEST cant whose
    margin is not negative. This is the shell-facing half, and it exists for
    two reasons.

    ONE, a design shell asks the api and not the rebuild. Stage 3's wing card
    has to answer "how much dihedral does this need" — that is a question
    about the wing's geometry, which is its own — while the rebuild that can
    measure it belongs to the flying stages. Routing it through here keeps
    the design shell free of the flight machinery (``tests/test_v4_stages.py``
    asserts that separation file by file) and gives both shells ONE answer.

    TWO, the payload is JSON-safe, so a shell can cache it, store it in a
    session and compare it with the one it showed last time.

    ``status`` is the answer and the number is only meaningful under it —
    ``found``, ``converges``, ``out_of_reach`` or ``refused`` (a design that
    does not weathercock is refused rather than scored, the same guard
    :func:`wing_score.spiral_refusal` applies). ``fin_stated`` says whether
    the yaw stiffness came from a fin the DESIGN carries or one the rebuild
    invented, because a report with no vertical surface gets one sized off
    its span and every lateral number then rides that assumption.
    """
    from .flightmodel import dihedral_for_spiral as _fix

    fix = _fix(report, **kw)
    return {"status": fix.status, "gamma_deg": fix.gamma_deg,
            "margin_now": fix.margin_now, "margin_at": fix.margin_at,
            "flown_deg": fix.flown_deg, "Cn_beta": fix.Cn_beta,
            "extra_deg": fix.extra_deg, "band": list(fix.band),
            "fin_stated": bool((report.get("geometry") or {}).get("fin"))}


#: how a breakdown spells the wing's own cant and the lateral deck it flew.
#: ``sweep_deg`` is the WING's quarter-chord sweep; ``dihedral_deg`` in the
#: same breakdown is the TAIL's cant (a V-tail's panel angle), which is why
#: the wing's is spelled out in full and this mapping exists at all.
LATERAL_KEYS = ("wing_dihedral_deg", "sweep_deg", "Cl_beta", "Cn_beta",
                "Cl_r", "Cn_r", "Cl_p", "Cn_p", "spiral_margin",
                "spiral_theta0_deg")


def anhedral_note(gamma_deg) -> str:
    """What a NEGATIVE wing cant means, in one sentence, or ``""``.

    ONE author, because the same fact has to be said in three places and a
    user meets it at whichever one they touch first: beside the field where
    a cant is STATED, beside the row where one is PINNED, and on the result
    where one was FLOWN. The searched row is floored at 0 while nothing
    prices its sign (``gui.v3.session.CANT_FLOOR_DEG``) — a default, not a
    ban — and the other two ways of asking for anhedral were silent: a
    stated -60 deg was reported back as "your own number, not one of the
    steps above", and a pinned negative flew verbatim with nothing said.

    Neither is refused. A stated number is the user's answer and this
    package does not turn a measurement into a gate; what it must not do is
    take that answer without saying what it is.
    """
    if isinstance(gamma_deg, bool) or not isinstance(gamma_deg, (int, float)):
        return ""
    g = float(gamma_deg)
    if not (g == g and g < 0.0):
        return ""
    return (f"THIS WING IS ANHEDRAL: its tips are {abs(g):.2f} deg BELOW "
            f"the root. That is the destabilising sign of the only "
            f"wing-side dihedral effect that holds AT ANY LIFT — the "
            f"aeroplane rolls INTO a sideslip rather than out of it. A "
            f"swept wing has one too, but it goes as CL and so is smallest "
            f"exactly where the aeroplane spends its time.")


def lateral_verdict(breakdown: dict) -> dict:
    """WHICH WAY THIS ANSWER IS CANTED, AND WHETHER ITS SPIRAL CONVERGES.

    The lateral half of a finished run, as a plain dict a shell can print.

    It exists because the Results stage named NEITHER. A run whose design
    box searched the cant came back with ``wing_dihedral_deg`` sitting in
    the design-vector table as a signed number, and a user who reads
    "-10.00" there has to know, unaided, that the tips point DOWN and that
    the aeroplane therefore rolls INTO a sideslip instead of out of it.
    That is not a table's job. Everything below is read off the breakdown
    the run recorded — never re-derived, never re-flown — so it is the
    aeroplane that was SCORED and not one rebuilt beside it.

    ``status`` is the answer and the rest is only meaningful under it:

    ``not_applicable``
        the breakdown states no wing cant at all, so this family never had
        the question. Draw nothing.
    ``not_measured``
        it states a cant and no lateral deck. Nothing in the run's
        objective priced the lateral half, so the deck was never built
        (``wingtail.WingTailProblem.lateral``) — the cant is reported, and
        whether it converges anything is NOT known from this run.
    ``refused``
        a deck, and ``Cn_beta <= 0``. The criterion is a difference of two
        products and driving the yaw stiffness through zero flips the sign
        of the second one, so a positive margin here is arithmetic — the
        sentence is :func:`wing_score.spiral_refusal`'s own, with one
        author, because it is the same guard the composite applies.
    ``diverges`` / ``converges``
        the measured criterion, at the trim attitude it was measured at
        (``spiral_theta0_deg``; the classical form drops that term and runs
        2.13 deg of dihedral optimistic — :func:`dynamics.spiral_margin`).

    ``level`` is the presentation weight — ``""``, ``"warn"`` or ``"bad"``
    — and ANHEDRAL RAISES IT WHATEVER THE SPIRAL SAYS: tips-down is the
    destabilising sign of the only wing-side ``Cl_beta`` that holds at any
    lift (sweep gives one that goes as CL, ``geometry.Wing``), and a
    design that reaches it because nothing priced the row is the failure
    the searched-cant floor exists to catch. A convergent spiral bought
    somewhere else does not make the cant unremarkable.
    """
    from .wing_score import spiral_refusal

    bd = breakdown or {}

    def _num(key):
        v = bd.get(key)
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return None
        return None if v != v else float(v)          # NaN is not a number

    out = {k: _num(k) for k in LATERAL_KEYS}
    gam, margin = out["wing_dihedral_deg"], out["spiral_margin"]
    # THE GATE IS THE CANT, NOT THE SWEEP. Every wing-like family records a
    # ``sweep_deg`` — ``objective.py`` writes one for a plain wing and a
    # winglet — and none of those has a fin, a spiral or anything that could
    # price a dihedral. Drawing a lateral card there would end in
    # "weight `spiral` on stage 3's objective card", which is a
    # recommendation that does not clear the gate it cites. Only a design
    # that STATES a wing cant, or that carries a lateral deck, is answered.
    if gam is None and margin is None:
        return {**out, "status": "not_applicable", "level": "", "says": ""}

    anhedral = bool(anhedral_note(gam))
    cant_says = ""
    if anhedral:
        cant_says = anhedral_note(gam).rstrip(".")
    elif gam is not None and gam > 0.0:
        cant_says = f"tips {gam:.2f} deg UP"
    elif gam is not None:
        cant_says = "planar — no wing-side dihedral effect at all"

    if margin is None:
        says = (cant_says + ". Whether its spiral converges is NOT known "
                "from this run: nothing in the objective priced the lateral "
                "half, so no lateral deck was built. Weight `spiral` on "
                "stage 3's objective card to have it measured.")
        return {**out, "status": "not_measured",
                "level": "warn" if anhedral else "", "says": says.lstrip(". ")}

    why = spiral_refusal(bd)
    if why:
        return {**out, "status": "refused", "level": "warn",
                "says": (cant_says + ". " if cant_says else "") + why}

    theta = out["spiral_theta0_deg"]
    at = ("" if theta is None else
          f", at the {theta:+.2f} deg nose-up attitude it trims to")
    if margin < 0.0:
        says = (f"THE SPIRAL DIVERGES: margin {margin:+.6f}{at}. Left "
                f"alone this aeroplane rolls further into a disturbance "
                f"instead of levelling.")
        return {**out, "status": "diverges", "level": "bad",
                "says": (cant_says + ". " if cant_says else "") + says}
    says = f"the spiral converges: margin {margin:+.6f}{at}."
    return {**out, "status": "converges",
            "level": "warn" if anhedral else "",
            "says": (cant_says + ". " if cant_says else "") + says}


def _section_weights(cfg: RunConfig, x):
    """(section AirfoilProblem, w_upper, w_lower) for any CST-carrying problem.

    The CST weights are located by LABEL (``w_upper_i`` / ``w_lower_i``), so
    every problem that embeds a section — the standalone ``airfoil
    (section)`` and the 13-D winglet+section pair, plus any future
    composition — is covered without a name list. Returns None when the
    problem carries no section.
    """
    spec = PROBLEM_SPECS.get(cfg.problem_name)
    if spec is None:
        return None
    # a READ-ONLY path (`section_coords`, `section_report`), so it sanitises
    # rather than refusing — see `sanitise_flags` for why the two classes of
    # entry point differ. Without this, opening the section view of an
    # archived run whose config carries an unhonoured key would raise.
    _flags, _dropped = sanitise_flags(cfg.problem_name, cfg.flags)
    built = spec.build(cfg.mission_kwargs or {}, _flags,
                       cfg.bounds_overrides)
    # labels come from the BUILT problem: ProblemSpec.param_labels is empty
    # for the standalone section problem, whose labels are derived at build
    # time from n_cst.
    labels = list(built.param_labels)
    up = [i for i, l in enumerate(labels) if str(l).startswith("w_upper_")]
    lo = [i for i, l in enumerate(labels) if str(l).startswith("w_lower_")]
    if not up:
        return None
    xa = np.asarray(x, dtype=float).ravel()
    if xa.size != len(labels):
        return None
    sec = getattr(built.problem, "section", built.problem)
    if not hasattr(sec, "n_cst"):
        return None
    # A SYMMETRIC SECTION CARRIES NO LOWER SURFACE IN ITS VECTOR — it is the
    # upper one mirrored (AirfoilProblem.split_weights), which is what makes
    # a fin's search four variables instead of eight. Requiring a matching
    # pair of label lists here read that as "this problem has no section",
    # so every consumer of the shape — the section card, the coordinates,
    # the CAD loft, the export — got None for the one surface whose section
    # is symmetric by definition.
    if not lo and getattr(sec, "symmetric", False):
        w_u = xa[up]
        return sec, w_u, -w_u
    if len(up) != len(lo):
        return None
    return sec, xa[up], xa[lo]


def section_coords(cfg: RunConfig, x) -> list | None:
    """Closed-loop airfoil coordinates for a CST-section design vector.

    Works for every problem whose design vector carries CST weights (the
    standalone section problem and the winglet+section pair); returns None
    for the rest. Pure geometry (the CST evaluation the problem itself
    uses) — no aerodynamics is run.
    """
    from . import airfoil
    got = _section_weights(cfg, x)
    if got is None:
        return None
    sec, w_u, w_l = got
    coords = airfoil.cst_coords(w_u, w_l, dz_te=float(getattr(sec, "dz_te",
                                                              0.0)))
    return _json_safe(np.asarray(coords, dtype=float))


def flown_section_coords(cfg: RunConfig, x, aft: bool = False) -> list | None:
    """Coordinates of the section this design actually FLEW, or None.

    Three ways a section reaches a solve, and this covers all of them in the
    order they bind:

    1. OPTIMISED — CST weights inside the design vector (:func:`section_coords`);
    2. CHOSEN — a library name or a CST weight dict in ``flags`` under
       :data:`SECTION_KEY` (:data:`SECTION_AFT_KEY` for the second surface);
    3. neither — None, and the caller is entitled to fall back to a stand-in
       section at the solved t/c, which is all the polar family ever claimed.

    Written for the geometry consumers (the 3-D view, ``cad``): exporting a
    NACA stand-in for a design that flew ``hg40`` would put a section in the
    file that no polar in the run ever came from.
    """
    from . import airfoil

    if not aft:
        got = section_coords(cfg, x)
        if got is not None:
            return got
    return chosen_section_coords(
        (cfg.flags or {}).get(SECTION_AFT_KEY if aft else SECTION_KEY))


def chosen_section_coords(value) -> list | None:
    """Coordinates of a CHOSEN section, from any shape the flag can carry.

    The value half of :func:`flown_section_coords`, split out because it is
    no longer a two-way aft/main question: the car endplate is a third
    surface with its own section slot (:data:`SECTION_PLATE_KEY`), and its
    STIFFNESS is integrated from these coordinates
    (``endplate.section_inertia_factor``). Restating the three shapes per
    caller is how the second surface's section came to be readable and the
    third's would not have been.

    Returns None for "no section chosen" and for a library member whose
    coordinate sidecar is missing — the caller decides what a missing shape
    means, exactly as :func:`symmetric_section_names` leaves "cannot
    restrict" to its caller.
    """
    from . import airfoil

    if isinstance(value, str):
        try:
            return _json_safe(_library_coords(value))
        except (FileNotFoundError, ValueError):
            return None
    if isinstance(value, dict) and value.get("w_upper") is not None:
        return _json_safe(np.asarray(airfoil.cst_coords(
            np.asarray(value["w_upper"], dtype=float),
            np.asarray(value["w_lower"], dtype=float),
            dz_te=float(value.get("dz_te", 0.0))), dtype=float))
    if isinstance(value, dict) and value.get("name"):
        return chosen_section_coords(str(value["name"]))
    return None


def _thickness_profile(sec, w_u, w_l, n: int = 2001):
    """(x/c, thickness/c) on the SAME cosine grid airfoil.cst_thickness uses.

    Sharing the grid matters: the reported max-thickness location must be the
    argmax of the very array whose max IS the t/c behind the g_tc constraint,
    or the view would print a thickness the solver never saw.
    """
    from .airfoil import _surface_y
    psi = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, n)))
    dz = float(getattr(sec, "dz_te", 0.0))
    t = (_surface_y(psi, w_u, +0.5 * dz) - _surface_y(psi, w_l, -0.5 * dz))
    return psi, t


def _polar_payload(pol, sec) -> dict:
    """XfoilPolarResult -> JSON-safe polar block with the design-lift readouts."""
    from .airfoil import _longest_increasing_run
    out = {
        "alpha_deg": _json_safe(np.asarray(pol.alpha_deg, dtype=float)),
        "cl": _json_safe(np.asarray(pol.cl, dtype=float)),
        "cd": _json_safe(np.asarray(pol.cd, dtype=float)),
        "cm": _json_safe(np.asarray(pol.cm, dtype=float)),
        "n_converged": int(pol.n_converged),
        "n_requested": int(len(list(sec.alphas))),
        "from_cache": bool(getattr(pol, "from_cache", False)),
        "re": float(sec.re), "mach": float(sec.mach),
    }
    cl = np.asarray(pol.cl, dtype=float)
    if cl.size >= 2:
        sl = _longest_increasing_run(cl)
        # The cd/cm readouts are interpolated on the PRE-STALL MONOTONE branch
        # only; a view that draws the whole sweep as one confident curve hides
        # which points the numbers actually came from.
        out["branch"] = [int(sl.start or 0),
                         int(sl.stop if sl.stop is not None else cl.size)]
        cl_b = cl[sl]
        if cl_b.size >= 2 and cl_b[0] <= sec.cl_design <= cl_b[-1]:
            out["cd_at_cl_design"] = float(
                np.interp(sec.cl_design, cl_b, np.asarray(pol.cd, float)[sl]))
            out["cm_at_cl_design"] = float(
                np.interp(sec.cl_design, cl_b, np.asarray(pol.cm, float)[sl]))
            out["alpha_at_cl_design"] = float(
                np.interp(sec.cl_design, cl_b,
                          np.asarray(pol.alpha_deg, float)[sl]))
    return out


def section_report(cfg: RunConfig, x, baseline: bool = True) -> dict | None:
    """Section shape + its viscous polar, for the airfoil result view.

    Returns None when the problem carries no CST section. The DESIGN's own
    polar comes from :func:`xfoil_run.run_xfoil_polar`, which hits the disk
    cache for any point the run itself evaluated (the normal case — the
    sweep is keyed on the coordinates). The BASELINE (the problem's anchor
    section, NACA 2412 by default) is served CACHE-ONLY: its shape is free
    geometry and always returned, but its polar appears only if that exact
    sweep is already on disk, so opening this view can never silently spend
    a minute of XFOIL time on a comparison the user did not ask for.
    """
    from . import airfoil
    from .xfoil_run import (DEFAULT_CACHE_DIR, _cache_load, cache_key,
                            run_xfoil_polar)

    got = _section_weights(cfg, x)
    if got is None:
        return None
    sec, w_u, w_l = got
    dz = float(getattr(sec, "dz_te", 0.0))

    def shape(wu, wl) -> dict:
        psi, t = _thickness_profile(sec, wu, wl)
        i = int(np.argmax(t))
        return {
            # the WEIGHTS the shape was drawn from travel with it: a consumer
            # that wants to fly this section again (the pipeline's stage 3)
            # must never re-fit them from the coordinates, and must never
            # have to rediscover them from the design vector's labels
            "w_upper": [float(v) for v in np.asarray(wu, dtype=float)],
            "w_lower": [float(v) for v in np.asarray(wl, dtype=float)],
            "dz_te": float(dz),
            "coords": _json_safe(np.asarray(
                airfoil.cst_coords(wu, wl, dz_te=dz), dtype=float)),
            "tc": float(airfoil.cst_thickness(wu, wl, dz_te=dz)),
            "tc_max_xc": float(psi[i]),
            "thickness_xc": _json_safe(psi),
            "thickness": _json_safe(t),
        }

    def cached_polar(wu, wl):
        coords = np.asarray(airfoil.cst_coords(wu, wl, dz_te=dz), dtype=float)
        key = cache_key(coords, sec.re, sec.mach, list(sec.alphas),
                        sec.n_panel)
        root = Path(sec.cache_dir if sec.cache_dir is not None
                    else DEFAULT_CACHE_DIR)
        # LOADABLE, not merely present. An entry written before the polar
        # validity gate carries no CDp column, so its rows cannot be checked
        # and _cache_load rejects it — the file exists but is a miss, and
        # probing with exists() would spend a full XFOIL solve here in a
        # view whose contract is that it never does.
        if _cache_load(root / f"{key}.json", len(list(sec.alphas))) is None:
            return None
        return run_xfoil_polar(coords, sec.re, sec.mach, sec.alphas,
                               timeout_s=sec.timeout_s,
                               cache_dir=sec.cache_dir, n_panel=sec.n_panel)

    out = {"design": shape(w_u, w_l),
           "cl_design": float(sec.cl_design),
           "tc_min": float(sec.tc_min),
           "cm_max": float(sec.cm_max)}
    coords = np.asarray(out["design"]["coords"], dtype=float)
    pol = run_xfoil_polar(coords, sec.re, sec.mach, sec.alphas,
                          timeout_s=sec.timeout_s, cache_dir=sec.cache_dir,
                          n_panel=sec.n_panel)
    out["design"]["polar"] = _polar_payload(pol, sec)

    if baseline:
        # the ANCHOR splits by the problem's own rule too: a symmetric w0 is
        # n_cst long, so slicing it in half gave an empty lower surface
        b_u, b_l = sec.split_weights(np.asarray(sec.w0, dtype=float))
        if not (np.allclose(b_u, w_u) and np.allclose(b_l, w_l)):
            base = shape(b_u, b_l)
            # w0 is the n_cst-weight CST PROJECTION of the named section
            # (t/c 0.1199, not 0.12, for "2412"), clipped into the box —
            # calling it "NACA 2412" would overstate what is drawn.
            # ...and a SYMMETRIC problem says so: its anchor is the named
            # section with its camber removed (AirfoilProblem symmetrises w0),
            # so "NACA 2412" would name a shape that is not what is drawn
            sym = " symmetrised" if getattr(sec, "symmetric", False) else ""
            base["name"] = (f"NACA {sec.anchor} CST refit{sym} (anchor w0)"
                            if isinstance(sec.anchor, str)
                            else f"base section CST fit{sym} (anchor w0)")
            bp = cached_polar(b_u, b_l)
            if bp is not None:
                base["polar"] = _polar_payload(bp, sec)
            out["baseline"] = base
    return out


# ---------------------------------------------------------------------
# Airfoil-only optimisation (the standalone Tier-B CST + live-XFOIL BO)
# ---------------------------------------------------------------------


def _score_weight_items(weights):
    """``(key, value)`` pairs of a weight set given as EITHER form.

    :func:`screen_weights` has always accepted a preset name, a partial dict
    or a :class:`~aerobo.airfoil_select.ScoreWeights`; the flag builder that
    puts the same weights on a RunConfig accepted only the first two, so the
    one type the package hands out was the one it could not be handed back.
    Stated as a helper so both directions read from one rule.
    """
    from dataclasses import asdict, is_dataclass

    if is_dataclass(weights) and not isinstance(weights, type):
        return asdict(weights).items()
    return dict(weights).items()


def airfoil_run_config(*, symmetric: bool = False,
                       re: float = 1e6, mach: float = 0.0,
                       cl_design: float = 0.5, tc_min: float = 0.10,
                       cm_max: float = 0.08, anchor=None, n_cst: int = 4,
                       wing: dict | None = None, twist_order: int = 1,
                       twist_max_deg: float = 6.0, alpha_max_deg: float = 10.0,
                       chord_order: int = 0, chord_max_frac: float = 0.5,
                       re_strip: str = "mac", re_bank: int = 1,
                       objective: str = "cd", score_weights=None,
                       score_reference=None, censored: str | None = None,
                       score_goals=None, goal_penalty: float | None = None,
                       goal_tol: float | None = None,
                       asf_rho: float | None = None,
                       asf_lambda: str | None = None,
                       asf_missing: str | None = None,
                       pareto_acqf: str | None = None,
                       pareto_q: int | None = None,
                       pareto_criteria=None,
                       optimiser: str = "bo", budget: int = 32,
                       seed: int = 0, refusal: str | None = None,
                       feasibility: str | None = None,
                       n_init: int | None = None,
                       sweep_alpha_max_deg: float | str | None = "auto",
                       ) -> RunConfig:
    """The RunConfig for a standalone airfoil-only shape optimisation.

    Pins the problem to ``airfoil (section)`` and carries the section operating
    point + design gates + optional base-seed anchor as airfoil-namespaced
    flags (:data:`AIRFOIL_FLAG_KEYS`). Pure — builds no problem and runs no
    XFOIL, so the GUI/tests can inspect exactly what will be launched. The 2-D
    section has NO aircraft mission, so ``mission_kwargs`` stays empty.

    ``wing`` (a :class:`section_wing.WingGuess` kwargs dict — mass, speed,
    altitude, the AREA GUESS, aspect ratio, taper) switches the problem into
    SECTION + TWIST mode (:data:`AIRFOIL_WING_FLAG_KEYS`): the design vector
    grows ``twist_order`` polynomial twist coefficients, the objective becomes
    wing L/D at the DERIVED CL = W/(qS), and ``re`` / ``cl_design`` are
    ignored because both now follow from the wing. ``twist_max_deg`` and
    ``alpha_max_deg`` are the user's two angle caps (twist envelope over the
    semi-span; largest local geometric incidence at trim).

    ``chord_order`` > 0 additionally makes the CHORD DISTRIBUTION a design
    variable: that many polynomial coefficients reshape the straight-taper
    baseline at constant area, capped by ``chord_max_frac`` (the largest
    fractional departure from the baseline chord anywhere on the span). The
    default 0 leaves the planform as the fixed trapezoid the taper ratio
    implies, which is the legacy behaviour bit-for-bit.

    ``objective`` picks WHICH SCALAR is maximised (:data:`AIRFOIL_OBJECTIVES`):
    ``cd`` (default, bit-for-bit the frozen study) minimises section drag at
    the design lift; ``composite`` maximises the GDP weighted composite J of
    the six criteria the library screen ranks on, under ``score_weights`` (a
    preset name or partial dict) and normalised against ``score_reference``
    (a frozen band payload; absent = the shipped library band). A composite run
    is a SECTION run: passing ``wing`` alongside it raises, because a twist law
    searched against a 2-D score is a freedom the objective cannot see.

    ``composite_goal`` is that same J with a REFERENCE POINT under it:

        J_goal = J - goal_penalty * sum_k w_k max(0, s_k(goal) - tol - s_k(x))

    a one-sided penalty on ending BELOW a per-criterion goal, and nothing at
    all above it. It exists because a weighted sum is free to sell: measured on
    the frozen sec8_composite case the plain composite gains 11 J while cruise
    L/D falls 12 % and cd at the design lift rises 14 %, because over the
    shipped band 0.01 of |Cm| is worth four counts of L/D and the stall angle
    is worth nothing (weight 0). ``score_goals`` states the goals in RAW
    criterion units (``{"ldcr": 80.0, "clmax": 1.4, ...}`` — t/c, cl_max,
    (L/D)max, (L/D) at the design lift, the stall angle, |Cm|); the default
    measures them off the SEED, i.e. "do not hand me back a section worse than
    the one I already have". ``goal_penalty`` (default
    ``airfoil_select.GOAL_PENALTY``) prices a shortfall as a multiple of the
    composite's own exchange rate — 0 recovers ``composite`` exactly, large
    values approach the hard-constrained problem — and ``goal_tol`` allows that
    many sub-score points of slack before a shortfall counts. All three raise
    on any other objective rather than being silently dropped.

    ``composite_asf`` is the AUGMENTED TCHEBYCHEFF / achievement scalarising
    function on the same band, weights and reference point:

        J_asf = min_k w_k (s_k - r_k) + asf_rho * sum_k w_k (s_k - r_k)

    It exists because ``composite_goal`` fixes the SALE and not the
    REACHABILITY — a weighted sum plus a hinge penalty is still a weighted sum,
    and a weighted sum only ever returns points on the CONVEX HULL of the
    achievable set, at every weight vector. The leading min term holds the
    search to its WORST criterion relative to the reference point, which no
    exchange rate can buy off; ``asf_rho`` (default
    ``airfoil_select.RHO_ASF`` = 0.05) is the AUGMENTATION that rules out
    weakly-Pareto points, and ``asf_rho=0`` is the pure Tchebycheff — reachable
    and stated, not the default. It shares ``score_goals`` and ``goal_tol``
    with the goal composite and refuses ``goal_penalty`` (there is no
    multiplier in an achievement function); ``asf_rho`` is refused everywhere
    else for the mirror-image reason. **J_asf is not a composite J** and is not
    comparable to one: it is ~0 at the reference point, so any cross-objective
    comparison must re-score both winners under ``composite``.

    ``censored`` picks what a RIGHT-CENSORED ``cl_max`` — a stall sweep that
    ended with cl still rising, so XFOIL bounded the maximum below rather than
    measuring it — scores as. ``None`` / ``"refuse"`` is the published path
    bit-for-bit (the candidate is a solver failure); ``"lower_bound"`` scores J
    at the bound, which is a true lower bound on the true J because both
    censorable criteria are higher-better and the rest come from the converged
    cruise sweep. It is a COMPOSITE policy and raises on a ``-cd`` run, where
    nothing reads ``cl_max``. See :mod:`aerobo.airfoil_select` and report
    §16.3a for why the default has not moved."""
    flags: dict = {
        "airfoil_re": float(re), "airfoil_mach": float(mach),
        "airfoil_cl_design": float(cl_design),
        "airfoil_tc_min": float(tc_min), "airfoil_cm_max": float(cm_max),
    }
    if symmetric:
        # only when TRUE: an unstated flag keeps every frozen run's config
        # hash exactly as it was
        flags["airfoil_symmetric"] = True
    if anchor is not None:
        flags["airfoil_anchor"] = anchor
    # THE CRUISE SWEEP'S TOP END, which is a ceiling on the design lift
    # (:data:`airfoil.ALPHA_SWEEP_DEG`). ``"auto"`` — the default — applies the
    # measured rule ``airfoil.sweep_ceiling_for``: the published sweep at and
    # below cl 1.0, the reliability ceiling above it. That leaves EVERY frozen
    # run bit-for-bit, because the rule only widens above cl 1.0 and no
    # published run asks for more (the car's own reference lift is exactly
    # 1.0), and it fixes the regime where the flag would otherwise have to be
    # discovered from a refusal to get any answer at all. ``None`` pins the
    # published sweep whatever the lift; a number states the ceiling outright.
    # Declared only when it is NOT the default, same rule as the objective and
    # the censoring, so a default call sends the legacy flag set unchanged.
    if sweep_alpha_max_deg is not None:
        from .airfoil import ALPHA_SWEEP_DEG, sweep_ceiling_for
        ceiling = (sweep_ceiling_for(cl_design)
                   if sweep_alpha_max_deg == "auto"
                   else float(sweep_alpha_max_deg))
        if ceiling != float(ALPHA_SWEEP_DEG[1]):
            flags["airfoil_sweep_alpha_max_deg"] = float(ceiling)
    if int(n_cst) != 4:
        # only declared off the published dimension, so a default call sends
        # the legacy flag set unchanged
        if int(n_cst) < 2:
            raise ValueError(f"n_cst must be >= 2, got {n_cst}")
        flags["airfoil_n_cst"] = int(n_cst)
    if wing:
        flags["airfoil_wing"] = {k: (int(v) if k == "n_stations"
                                     else float(v))
                                 for k, v in dict(wing).items()}
        flags["airfoil_twist_order"] = int(twist_order)
        flags["airfoil_twist_max_deg"] = float(twist_max_deg)
        flags["airfoil_alpha_max_deg"] = float(alpha_max_deg)
        if int(chord_order):
            # only declared when a chord law is actually asked for, so an
            # untouched planform sends the legacy flag set unchanged
            flags["airfoil_chord_order"] = int(chord_order)
            flags["airfoil_chord_max_frac"] = float(chord_max_frac)
            # the chord law strictly GENERALISES the fixed trapezoid, so it
            # must not be able to score worse; seeding the un-reshaped design
            # keeps that true at a real budget (see :func:`_bo_x_init`)
            flags[BO_WARM_START_FLAG] = True
        from .section_wing import RE_STRIP_MODES
        if str(re_strip) not in RE_STRIP_MODES:
            raise ValueError(f"unknown re_strip mode {re_strip!r}; "
                             f"choose from {list(RE_STRIP_MODES)}")
        if int(re_bank) < 1:
            raise ValueError(f"re_bank must be >= 1, got {re_bank}")
        # ...and as a PAIR, which is the only way the problem itself accepts
        # them (section_wing.SectionWingProblem.__post_init__). Validated
        # separately, re_strip="bank" with the DEFAULT re_bank=1 — the call
        # this function's own docstring invites — returned a RunConfig that
        # raises the moment anything builds it. It raises in 0.00 s so no
        # XFOIL is wasted, but a config is a promise that the run is
        # launchable, and this one was not. The two messages are the
        # problem's own, and a test gates the refusal SETS equal over the
        # whole (mode x bank) grid so the two cannot drift.
        if str(re_strip) == "bank" and int(re_bank) < 2:
            raise ValueError(
                "re_strip='bank' needs re_bank >= 2 XFOIL sweeps to "
                "interpolate between; re_bank=1 IS the 'mac' mode")
        if str(re_strip) != "bank" and int(re_bank) != 1:
            raise ValueError(
                f"re_bank={re_bank} only means anything under "
                f"re_strip='bank' (got {re_strip!r}) — a sweep count "
                f"that no mode reads is a setting that does nothing")
        # only declared when they are not the default: a run that leaves the
        # spanwise Reynolds treatment alone sends the legacy flag set
        if str(re_strip) != "mac":
            flags["airfoil_re_strip"] = str(re_strip)
        if int(re_bank) > 1:
            flags["airfoil_re_bank"] = int(re_bank)
    elif int(re_bank) != 1 or str(re_strip) != "mac":
        raise ValueError(
            "the spanwise Reynolds treatment needs a wing: a 2-D section has "
            "one Reynolds number by definition. Pass wing=... as well, or "
            "drop re_strip / re_bank")
    if str(objective) != "cd":
        # only declared when it is not the default, so a -cd run sends the
        # legacy flag set unchanged (and reuses its cached result cells)
        flags["airfoil_objective"] = str(objective)
        if score_weights is not None:
            # A ScoreWeights is the package's OWN weight type — what
            # ``screen_weights`` returns and what ``airfoil_select.PRESETS``
            # holds — so a caller who reached for a preset and edited one
            # field was handing this the most natural object there is. It
            # was the one form that did not survive: ``dict(dataclass)``
            # raises, and only on the composite branch, so the -cd run
            # beside it worked and the failure read as "the composite
            # objective produces nothing". Converted here rather than by
            # routing the whole line through ``screen_weights``: that
            # would merge a PARTIAL dict up to a full one, and the flag is
            # hashed into the run's provenance and its cached cells.
            flags["airfoil_score_weights"] = (
                score_weights if isinstance(score_weights, str)
                else {k: float(v)
                      for k, v in _score_weight_items(score_weights)})
        if score_reference is not None:
            flags["airfoil_score_reference"] = score_reference
    # The goal keys are declared wherever the caller stated them — including
    # on an objective that has no goals, where `_airfoil_goal_settings` raises
    # at the bottom of this function. A flag that is quietly dropped is worse
    # than one that is refused, and this is the only place that can tell.
    if score_goals is not None:
        flags["airfoil_score_goals"] = (
            score_goals if isinstance(score_goals, str)
            else {str(k): float(v) for k, v in dict(score_goals).items()})
    if goal_penalty is not None:
        flags["airfoil_goal_penalty"] = float(goal_penalty)
    if goal_tol is not None:
        flags["airfoil_goal_tol"] = float(goal_tol)
    if asf_rho is not None:
        # declared wherever stated (including on an objective with no
        # augmentation, where the check at the bottom raises), and only when
        # stated — so a default composite_asf call sends one flag, not two
        from .airfoil_select import check_asf_rho as _check_rho
        flags["airfoil_asf_rho"] = _check_rho(asf_rho)
    if asf_lambda is not None:
        from .airfoil_select import (DEFAULT_ASF_LAMBDA as _DL,
                                     check_asf_lambda as _check_lam)
        if _check_lam(asf_lambda) != _DL:
            # only when it is NOT the default, so every run measured under
            # lambda = the weights keeps sending the flag set it was measured
            # with — the same rule the objective and the censoring follow
            flags["airfoil_asf_lambda"] = _check_lam(asf_lambda)
    if asf_missing is not None:
        from .airfoil_select import (DEFAULT_ASF_MISSING as _DM,
                                     check_asf_missing as _check_miss)
        # same rule again: a run at the legacy "drop" sends no flag, so every
        # published ASF config is byte-for-byte what it was measured as
        if _check_miss(asf_missing) != _DM:
            flags["airfoil_asf_missing"] = _check_miss(asf_missing)
    # The front's three keys, declared wherever stated and validated HERE, so
    # an unknown acquisition or a bad criteria set is refused before any XFOIL
    # runs rather than 40 evaluations in. Only when stated and only when not
    # the default, so a plain front run sends one flag, not four.
    if pareto_acqf is not None:
        if str(pareto_acqf) not in PARETO_ACQFS:
            raise ValueError(f"unknown pareto acqf {pareto_acqf!r}; "
                             f"choose from {list(PARETO_ACQFS)}")
        if str(pareto_acqf) != PARETO_DEFAULT_ACQF:
            flags["airfoil_pareto_acqf"] = str(pareto_acqf)
    if pareto_q is not None and int(pareto_q) != PARETO_DEFAULT_Q:
        if int(pareto_q) < 1:
            raise ValueError(f"pareto_q must be >= 1, got {pareto_q!r}")
        flags["airfoil_pareto_q"] = int(pareto_q)
    if pareto_criteria is not None:
        from .airfoil_select import check_pareto_criteria as _check_crit
        flags["airfoil_pareto_criteria"] = list(_check_crit(pareto_criteria))
    if str(objective) in AIRFOIL_REFERENCE_OBJECTIVES \
            or str(objective) == PARETO_OBJECTIVE_NAME:
        # THE SEED GOES IN THE TRAINING SET. Both reference-point objectives
        # are defined RELATIVE to the seed — the goal composite floors at it,
        # the ASF measures achievement from it — and a search that never
        # evaluates that section can still return something worse than it, so
        # the claim would be about the scoring and not about the answer.
        # Costs nothing: the Sobol draw shrinks by the one row it gains, and
        # the seed's sweeps are the ones the reference point already paid for.
        #
        # The front is the same statement in its strongest form: the seed IS
        # the hypervolume reference point, so a front that never evaluated it
        # would be measuring volume from a section it had not flown. It also
        # keeps `run(objective="pareto")` and `pareto_airfoil` — whose own
        # default is warm_start=True — from disagreeing about it.
        flags[BO_WARM_START_FLAG] = True
    if censored is not None:
        from .airfoil_select import DEFAULT_CENSORED, check_censored
        if check_censored(censored) != DEFAULT_CENSORED:
            # only declared when it is not the default, so every published
            # composite run sends the legacy flag set unchanged (and reuses
            # its cached result cells) — same rule as the objective itself
            flags["airfoil_censored"] = check_censored(censored)
    if refusal is not None:
        from .optimize.refusal import DEFAULT_MODE, check_mode
        if check_mode(refusal) != DEFAULT_MODE:
            # a search decision, and stated only when it is not the default so
            # an untouched call sends the legacy flag set unchanged
            flags[BO_REFUSAL_FLAG] = check_mode(refusal)
    if feasibility is not None:
        from .optimize.feasible import (DEFAULT_MODE as _F_DEFAULT,
                                        check_mode as _f_check)
        if _f_check(feasibility) != _F_DEFAULT:
            # What a run that has found nothing FEASIBLE does about it
            # (optimize.feasible). It belongs on this surface and not only on
            # RunConfig because the section search is launched through here,
            # and it is the surface where the failure was measured: the
            # reported `airfoil (section)` run found 0 of 164 evaluations
            # feasible (58 of 64 box draws refused with "cl_design not
            # bracketed by the pre-stall monotone branch"), and the same
            # config with a screened initial design found 142 of 164. Stated
            # only when it is not the default, so an untouched call sends the
            # legacy flag set unchanged.
            flags[BO_FEASIBILITY_FLAG] = _f_check(feasibility)
    if n_init is not None:
        # The Sobol seed size the study recommends. There was no way to state
        # it here at all, so a caller with a measured split silently flew the
        # 2d-capped-at-16 default instead — invisible only while the two
        # happened to agree. They no longer do: the section's own split arms
        # rank a four-point seed first, which is 0.5 d rather than 2 d.
        #
        # Validated HERE, at the surface that takes it, and not only in
        # _bo_n_init at run time: n_init < 1 raised only once something ran,
        # and n_init >= budget was silently CLAMPED by _bo_split while the
        # stored config kept the requested number — so RunResult.config
        # stated a seed size the run did not use. That is the repo's own
        # "the box shown is the box searched" failure one field over.
        if int(n_init) < 1:
            raise ValueError(f"n_init must be >= 1, got {n_init}")
        if int(n_init) >= int(budget):
            raise ValueError(
                f"n_init={n_init} leaves no evaluations for the BO loop at "
                f"budget={budget}: a seed that IS the whole budget is a "
                f"Sobol run, not a BO run. Use n_init <= {int(budget) - 1}, "
                f"or raise the budget")
        flags[BO_N_INIT_FLAG] = int(n_init)
    # BUILD IT NOW, not at run time: an objective that cannot be assembled
    # (wing mode + composite, or no frozen band) must fail where the caller
    # asked for it — a config is a promise that this run is launchable.
    _airfoil_objective(flags)
    # …and the same for the goal settings: an unknown criterion, or a goal
    # stated on an objective that has none, is a caller error here rather than
    # a surprise on the first evaluation.
    _airfoil_goal_settings(flags, str(objective))
    return RunConfig(problem_name="airfoil (section)", flags=flags,
                     optimiser=optimiser, budget=int(budget), seed=int(seed))


def _seed_design_report(cfg: RunConfig) -> dict | None:
    """:func:`design_report` of the point the section search STARTED from.

    The seed vector has two spellings because the two problems the section
    config can build name it differently: ``x0`` on the section+twist wing
    (:class:`section_wing.SectionWingProblem`, the library section at zero
    twist and the un-reshaped chord law) and ``w0`` on the bare 2-D section
    (:class:`airfoil.AirfoilProblem`, the anchor CST weights clipped into the
    box). Reading only ``x0`` is why a 2-D run had no seed to compare against
    — it silently found nothing and returned nothing.

    None when the built problem declares neither. COSTS ONE EVALUATION of the
    seed: an XFOIL sweep, plus the wide stall sweep on a composite run.
    """
    spec = PROBLEM_SPECS[cfg.problem_name]
    built = spec.build(cfg.mission_kwargs or {}, cfg.flags or {},
                       cfg.bounds_overrides)
    x0 = getattr(built.problem, "x0", None)
    if x0 is None:
        x0 = getattr(built.problem, "w0", None)
    if x0 is None:
        return None
    return design_report(cfg, x0)


#: the multi-objective acquisitions ``pareto_airfoil`` offers.
PARETO_ACQFS = ("qnehvi", "qnparego")

#: The front's defaults, in one place because :func:`pareto_airfoil` and the
#: ``objective="pareto"`` flag path must not be able to disagree about them.
PARETO_DEFAULT_ACQF = "qnehvi"
PARETO_DEFAULT_Q = 1


def pareto_airfoil(*, re: float = 1e6, mach: float = 0.0,
                   cl_design: float = 0.5, tc_min: float = 0.10,
                   cm_max: float = 0.08, anchor=None,
                   score_weights=None, score_reference=None,
                   censored: str | None = None,
                   budget: int = 48, seed: int = 0, n_init: int | None = None,
                   q: int = PARETO_DEFAULT_Q,
                   acqf: str = PARETO_DEFAULT_ACQF, criteria=None,
                   refusal: str | None = None,
                   warm_start: bool = True,
                   progress_cb: Callable | None = None) -> dict:
    """Shape-optimise a section for a PARETO FRONT, not for one number.

    Three objectives — cruise L/D at the design lift, ``cl_max`` and |Cm| —
    each as its sub-score on the FROZEN band, all higher-better
    (:data:`airfoil_select.PARETO_CRITERIA`). ``t/c`` and |Cm| stay in the
    constraint channel exactly as the composite has them, so the front and
    every scalarised arm share ONE feasible region and their designs compare
    candidate for candidate.

    Constrained qNEHVI (or ``acqf="qnparego"``) from BoTorch, which is already
    a dependency. The hypervolume reference point is the SEED's own objective
    vector, so a design contributes volume only if it beats the section the
    user already has on all three at once, and the hypervolume of two runs is
    the same number (a reference point read off the run would not be).

    Returns ``{config, conditions, front, result}``:

    * ``front`` — one row per non-dominated feasible design, each carrying its
      design vector, its three objectives in sub-score points AND in raw
      units, its plain composite J, and every one of the six criteria so the
      user picking a point can see what the other three do;
    * rows are ordered by the user's OWN weights restricted to the three
      criteria and renormalised (``airfoil_select.pareto_weights``) — the
      composite doing the job it should always have had, RANKING a front the
      search produced rather than deciding what the search may reach;
    * ``result.hypervolume`` and its per-evaluation trace, ``result.n_feasible``
      and the refusal count.

    ``q > 1`` is wired but the evaluations still run one XFOIL at a time, so a
    batch buys nothing until the evaluator is parallel — measured, not
    assumed, by ``scripts/xfoil_batch_probe.py``.
    """
    # the SAME config the composite would build, so the band, the weights, the
    # censoring policy and the gates are resolved by one code path
    cfg = airfoil_run_config(re=re, mach=mach, cl_design=cl_design,
                             tc_min=tc_min, cm_max=cm_max, anchor=anchor,
                             objective="composite", score_weights=score_weights,
                             score_reference=score_reference,
                             censored=censored, optimiser="bo",
                             budget=budget, seed=seed, refusal=refusal,
                             n_init=n_init)
    return _pareto_from_config(cfg, q=q, acqf=acqf, criteria=criteria,
                               warm_start=warm_start,
                               progress_cb=progress_cb)[0]


def car_pareto_families() -> tuple:
    """Every registered family a car-wing FRONT can be searched on.

    DERIVED from the registry's own declaration — a family is a car family
    exactly when it accepts ``car_objective`` — and never a list here. A
    literal would be one more place for the registry to be restated and the
    first to go stale when a family is added.
    """
    return tuple(k for k, v in PROBLEM_SPECS.items()
                 if "car_objective" in v.flags)


def pareto_car_wing(problem_name: str = "car rear wing", *,
                    flags: dict | None = None,
                    bounds_overrides: dict | None = None,
                    budget: int = 48, seed: int = 0,
                    n_init: int | None = None,
                    q: int = PARETO_DEFAULT_Q,
                    acqf: str = PARETO_DEFAULT_ACQF,
                    refusal: str | None = None,
                    progress_cb: Callable | None = None) -> dict:
    """Design a car rear wing for a PARETO FRONT, not for one number.

    Two objectives, both FORCES and both what the car feels: the downforce
    ``F_z`` maximised and the drag ``D`` minimised
    (:data:`carwing.CAR_PARETO_OBJECTIVES`). No weight, no downforce floor
    and no circuit — the run returns the whole trade and the choosing happens
    afterwards, which is the one formulation here that does not decide the
    answer before it searches.

    WHY THIS EXISTS BESIDE SIX SCALAR OBJECTIVES. Each of those picks one
    point of this trade and cannot be aimed at another without changing the
    question, and the front is strongly CONCAVE, so the linear ones are blind
    to a region no normalisation recovers. Measured over a 14 803-design grid
    of the endplate family (618 front points), sweeping each form's own knob:
    a downforce/efficiency weight reaches 2.6 %, ``CZ - lambda*CD`` 6.3 %, an
    augmented Tchebycheff 54.5 %, ``efficiency`` under a downforce floor
    38.5 %, ``drag`` under the same floor 98.5 %. A front is the trade itself.

    Constrained qNEHVI (or ``acqf="qnparego"``) through
    :func:`optimize.mobo.run_mobo_constrained`, with the family's own margins
    in the constraint channel unchanged — so a front row is feasible in
    exactly the sense every scalar run's winner is.

    THE REFERENCE POINT IS THE INCUMBENT, fixed before the run: the box
    centre's own ``(F_z, -D)``. A design contributes hypervolume only if it
    beats the wing you already have on BOTH axes, and two runs of one problem
    give comparable numbers. Read off the run instead and they would not.

    Returns ``{config, objectives, ordered_by, reference_point, front,
    result}``:

    * ``front`` — one row per non-dominated feasible design, each with its
      design vector, its two objectives in newtons, and its full breakdown,
      so whoever picks a point can see everything else it does;
    * rows ordered by DOWNFORCE, descending — a stated order, not a
      preference dressed as one. With a circuit on the problem they are
      ordered by LAP TIME instead, which is the honest ranker: the circuit
      supplies the exchange rate rather than the user choosing it, and
      ranking a front cannot make an unreachable design reachable
      (``optimize.mobo.rank_front`` makes the same argument for the section);
    * ``result`` — hypervolume, its trace, and the feasible count.
    """
    from . import carwing as _cw
    from .optimize import mobo as _mobo

    if str(acqf) not in PARETO_ACQFS:
        raise ValueError(f"unknown pareto acqf {acqf!r}; "
                         f"choose from {list(PARETO_ACQFS)}")
    if problem_name not in PROBLEM_SPECS:
        raise ValueError(f"unknown problem {problem_name!r}")
    spec = PROBLEM_SPECS[problem_name]
    if "car_objective" not in spec.flags:
        raise ValueError(
            f"pareto_car_wing is for the car families and {problem_name!r} is "
            f"not one: its two objectives are a downforce and a drag, both in "
            f"newtons, and only a car family reports those. Choose from "
            f"{list(car_pareto_families())} — the section front is "
            f"api.pareto_airfoil")

    cfg = RunConfig(problem_name=problem_name, flags=dict(flags or {}),
                    optimiser="bo", budget=int(budget), seed=int(seed),
                    bounds_overrides=bounds_overrides)
    built = spec.build({}, cfg.flags, bounds_overrides)
    prob = built.problem

    def _ev(x, _p=None):
        return built.evaluate(x)

    ref_point = _cw.car_pareto_reference(prob, _ev)
    if ref_point is None:
        raise ValueError(
            "the centre of this design box does not fly, so there is no "
            "incumbent to measure a hypervolume against. A front needs a "
            "reference point, and inventing one out of a failed evaluation is "
            "the mistake api.pareto_airfoil already refuses to make — widen "
            "the box, or relax whatever refuses its centre (gui.diagnose "
            "reads the reason)")

    n_init_eff = _bo_split(int(budget), prob.dim, n_init or _bo_n_init(cfg))[0]
    fg = _cw.make_car_pareto_fg(prob, _ev)
    n_seen = {"i": 0}

    def counted(x):
        y, g = fg(x)
        n_seen["i"] += 1
        if progress_cb is not None:
            try:
                progress_cb(n_seen["i"], None)
            except Exception:
                pass
        return y, g

    t0 = time.time()
    hist = _mobo.run_mobo_constrained(
        counted, prob.bounds, ref_point=ref_point, n_init=n_init_eff,
        n_iter=int(budget) - n_init_eff, seed=int(seed), q=int(q),
        acqf=str(acqf), refusal=refusal or _bo_refusal(cfg),
        name=f"{problem_name} (front)")
    wall = time.time() - t0

    X_front, Y_front = hist.front()
    rows = []
    for x, y in zip(X_front, Y_front):
        out = built.evaluate(x)
        rows.append({"x": [float(v) for v in x],
                     "downforce_N": float(y[0]), "drag_N": float(-y[1]),
                     "lap_time_s": out.get("lap_time_s"),
                     "breakdown": out})
    if rows and rows[0].get("lap_time_s") is not None:
        rows.sort(key=lambda r: r["lap_time_s"])
        ordered_by = "lap_time_s"
    else:
        rows.sort(key=lambda r: -r["downforce_N"])
        ordered_by = "downforce_N"
    return {
        "config": cfg,
        "objectives": list(_cw.CAR_PARETO_OBJECTIVES),
        "ordered_by": ordered_by,
        "reference_point": [float(v) for v in ref_point],
        "front": rows,
        "result": {
            "n_evals": int(hist.X.shape[0]),
            "n_feasible": int(hist.feasible.sum()),
            "n_front": len(rows),
            "hypervolume": (float(hist.hv_trace[-1])
                            if len(hist.hv_trace) else 0.0),
            "hv_trace": [float(v) for v in hist.hv_trace],
            "wall_s": float(wall),
        },
    }


def _pareto_from_config(cfg: RunConfig, *, q: int = PARETO_DEFAULT_Q,
                        acqf: str = PARETO_DEFAULT_ACQF, criteria=None,
                        warm_start: bool = True,
                        progress_cb: Callable | None = None) -> tuple:
    """``(report, history)`` — the front search itself, from a built RunConfig.

    The ONE search path. :func:`pareto_airfoil` reaches it by building a config
    out of keyword arguments; ``run`` reaches it with the config it was handed,
    reading the front's settings off that config's flags. Two entry points and
    one search, because two searches would be two places to disagree about the
    band, the reference point or the criteria set — and the objective a run
    reports must be the objective it performed.

    The history is returned beside the report because ``run`` needs the
    per-evaluation log (``X``, ``Y``, the hypervolume trace) that the report,
    which is a front and not a search log, does not carry.
    """
    import numpy as _np

    from . import airfoil as _af
    from .airfoil_select import (check_pareto_criteria, make_pareto_fg,
                                 pareto_evaluation, pareto_weights,
                                 seed_pareto_point)
    from .optimize import mobo as _mobo

    PARETO_CRITERIA = check_pareto_criteria(criteria)
    if str(acqf) not in PARETO_ACQFS:
        raise ValueError(f"unknown pareto acqf {acqf!r}; "
                         f"choose from {list(PARETO_ACQFS)}")
    budget, seed = int(cfg.budget), int(cfg.seed)
    prob = _af.AirfoilProblem(**_airfoil_kwargs(cfg.flags))
    _name, ref, w, cens = _airfoil_objective(cfg.flags)

    ref_point = seed_pareto_point(prob, ref, w, censored=cens,
                                  criteria=PARETO_CRITERIA)
    if ref_point is None:
        raise ValueError(
            "the seed section could not be scored, so there is no reference "
            "point to measure a hypervolume against. A front needs one, and "
            "inventing it out of a failed evaluation is the mistake the goal "
            "composite already refuses to make")

    n_init_eff = _bo_split(int(budget), prob.dim, _bo_n_init(cfg))[0]
    n_iter = int(budget) - n_init_eff
    x_init = (_np.atleast_2d(_np.asarray(prob.w0, dtype=float))
              if warm_start else None)

    fg = make_pareto_fg(prob, ref, w, censored=cens,
                        criteria=PARETO_CRITERIA)
    n_seen = {"i": 0}

    def counted(x):
        y, g = fg(x)
        n_seen["i"] += 1
        if progress_cb is not None:
            try:
                progress_cb(n_seen["i"], None)
            except Exception:
                pass
        return y, g

    t0 = time.time()
    hist = _mobo.run_mobo_constrained(
        counted, prob.bounds, ref_point=ref_point, n_init=n_init_eff,
        n_iter=n_iter, seed=int(seed), q=int(q), acqf=str(acqf),
        x_init=x_init, refusal=_bo_refusal(cfg), name="airfoil (front)")
    wall = time.time() - t0

    X_front, Y_front = hist.front()
    order = _mobo.rank_front(Y_front, pareto_weights(w, PARETO_CRITERIA)) \
        if len(X_front) else _np.array([], dtype=int)
    rows = []
    for rank, i in enumerate(order):
        x = X_front[i]
        ev = pareto_evaluation(x, prob, ref, w, censored=cens,
                               criteria=PARETO_CRITERIA)
        rows.append({
            "rank": int(rank),
            "x": [float(v) for v in x],
            "objectives": {k: float(v) for k, v in
                           zip(PARETO_CRITERIA, Y_front[i])},
            "raw": {k: (abs(float(ev["cm"])) if k == "cm"
                        else ev.get(k))
                    for k in PARETO_CRITERIA
                    if k != "cm" or ev.get("cm") is not None},
            # every criterion on every row: a front the user picks off must
            # not hide the three it was not searched on
            "scores": {k: float(v) for k, v in (ev.get("scores") or {}).items()},
            "composite": ev.get("composite"),
            "tc": ev.get("tc"), "astall": ev.get("astall"),
            "cd_at_cl": (None if ev.get("f_cd") is None
                         else float(-ev["f_cd"])),
            "hypervolume_alone": _mobo.hv_needed_for(Y_front[i], ref_point),
        })
    seed_ev = pareto_evaluation(_np.asarray(prob.w0, dtype=float), prob, ref,
                                w, censored=cens, criteria=PARETO_CRITERIA)
    n_ref = int((_np.asarray(hist.Y) <= -100.0).all(axis=1).sum())
    # The conditions come off the BUILT PROBLEM, not off the arguments that
    # built it. `run` arrives here with a config nobody passed re/mach/anchor
    # to, and `_airfoil_kwargs` only carries the flags that were STATED — so
    # indexing it would raise on a default operating point, and echoing this
    # function's own defaults instead would report conditions the search did
    # not use. `prob` holds the values that were actually flown.
    report = {
        "config": cfg.to_dict(),
        "conditions": {
            "re": float(prob.re), "mach": float(prob.mach),
            "cl_design": float(prob.cl_design),
            "tc_min": float(prob.tc_min), "cm_max": float(prob.cm_max),
            "seeded": (cfg.flags or {}).get("airfoil_anchor") is not None,
            "objective": PARETO_OBJECTIVE_NAME,
            "criteria": list(PARETO_CRITERIA),
            "score": {"weights": w.normalised(), "reference_sha": ref.sha,
                      "reference_n": int(ref.n_records), "censored": str(cens),
                      "rank_weights": [float(v) for v in
                             pareto_weights(w, PARETO_CRITERIA)]},
            "reference_point": [float(v) for v in ref_point],
            "acqf": str(acqf), "q": int(q), "warm_start": bool(warm_start),
        },
        "seed": {
            "x": [float(v) for v in _np.asarray(prob.w0, dtype=float)],
            "objectives": {k: float(v) for k, v in
                           zip(PARETO_CRITERIA, ref_point)},
            "raw": {k: (abs(float(seed_ev["cm"])) if k == "cm"
                        else seed_ev.get(k))
                    for k in PARETO_CRITERIA
                    if k != "cm" or seed_ev.get("cm") is not None},
            "composite": seed_ev.get("composite"),
        },
        "front": rows,
        "result": {
            "n_evals": int(hist.X.shape[0]),
            "n_feasible": int(hist.feasible.sum()),
            "n_refused": n_ref,
            "n_front": int(len(rows)),
            "hypervolume": float(hist.hypervolume),
            "hypervolume_trace": [float(v) for v in hist.hv_trace],
            "spread": float(_mobo.spread(Y_front)) if len(rows) > 1 else 0.0,
            # WITHIN-RUN ONLY: measured from this run's own final feasible
            # nadir, so it is monotone and readable as progress and is NOT
            # comparable across runs. The seed hypervolume above is the
            # cross-run number. Never mix them.
            "hypervolume_nadir_within_run": float(hist.hypervolume_nadir),
            "hypervolume_nadir_trace_within_run": [
                float(v) for v in hist.hv_trace_nadir],
            "nadir_within_run": [float(v) for v in hist.nadir],
            "n_init": int(hist.n_init),
            "gp_failures": list(hist.gp_failures),
        },
        "wall_time_s": float(wall),
    }
    return report, hist


def optimize_airfoil(*, re: float = 1e6, mach: float = 0.0,
                     cl_design: float = 0.5, tc_min: float = 0.10,
                     cm_max: float = 0.08, anchor=None,
                     symmetric: bool = False,
                     wing: dict | None = None, twist_order: int = 1,
                     twist_max_deg: float = 6.0, alpha_max_deg: float = 10.0,
                     chord_order: int = 0, chord_max_frac: float = 0.5,
                     re_strip: str = "mac", re_bank: int = 1,
                     objective: str = "cd", score_weights=None,
                     score_reference=None, censored: str | None = None,
                     score_goals=None, goal_penalty: float | None = None,
                     goal_tol: float | None = None,
                     asf_rho: float | None = None,
                     asf_lambda: str | None = None,
                     asf_missing: str | None = None,
                     pareto_acqf: str | None = None,
                     pareto_q: int | None = None,
                     pareto_criteria=None,
                     optimiser: str = "bo", budget: int = 32, seed: int = 0,
                     progress_cb: Callable | None = None,
                     results_dir: str | Path | None = None,
                     with_section: bool = True,
                     with_baseline: bool = True,
                     refusal: str | None = None,
                     feasibility: str | None = None,
                     n_init: int | None = None,
                     sweep_alpha_max_deg: float | str | None = "auto",
                     stop_rule: Callable | None = None,
                     stop_reason: str | None = None,
                     resume: dict | None = None) -> dict:
    """Run the airfoil-only CST + live-XFOIL shape optimisation; JSON-safe report.

    The optimise twin of :func:`screen_airfoils`: where the screen PICKS the
    best known aerofoil for the operating point, this SHAPE-OPTIMISES an 8-D CST
    section (real viscous XFOIL polar per candidate — SLOW, seconds per new
    design). ``anchor`` seeds the design box: a NACA code, or an explicit
    ``[w_upper, w_lower]`` CST fit of a library winner (the base-seeded BO). The
    default call is bit-for-bit the published Tier-B AirfoilProblem() study.

    Returns ``{config, conditions, result, wall_time_s}`` plus, when
    ``with_section`` and a feasible incumbent was found, ``section`` — a
    :func:`section_report`-shaped block (optimised shape + live polar, with the
    anchor section overlaid) the GUI draws directly. ``progress_cb(i, best)``
    fires once per evaluation for a live convergence trace (rich payload if it
    declares ``**kwargs``).

    A FEASIBLE INCUMBENT ALSO BRINGS ITS SEED. ``design`` is the incumbent's
    own :func:`design_report` (a cache hit — the run just evaluated it) and,
    with ``with_baseline``, ``baseline`` is the SAME report for the point the
    search started from (:func:`_seed_design_report`). That is what makes the
    result a COMPARISON rather than a number: every row of the GUI's
    seed-vs-optimised table is a like-for-like difference at one operating
    point, and evaluating the seed here warms its polar so the section view's
    cache-only overlay has a curve to draw. It costs one extra XFOIL sweep of
    the seed (two on a composite run); ``with_baseline=False`` is the opt-out
    for a caller that is timing the search itself.

    Passing ``wing`` (a WingGuess kwargs dict — see :func:`airfoil_run_config`)
    switches to SECTION + TWIST mode: the objective becomes WING L/D at the
    design lift the area guess implies, ``twist_order`` polynomial twist
    coefficients join the design vector under the ``twist_max_deg`` /
    ``alpha_max_deg`` caps, and the report gains ``conditions.wing`` (every
    derived quantity: CL_design, Re at the MAC, span, MAC, q) plus ``design``
    — the :func:`design_report` block whose ``geometry`` carries the spanwise
    chord / Cl / effective-alpha arrays for the loading plots.

    ``chord_order`` > 0 additionally opens the CHORD DISTRIBUTION: that many
    polynomial coefficients reshape the straight-taper planform at constant
    area, under the ``chord_max_frac`` deviation cap, and the report gains
    ``conditions.chord``.

    ``refusal`` picks what a design XFOIL could not fly (the -100 sentinel)
    is shown to the BO surrogate as — see :mod:`aerobo.optimize.refusal`. It
    matters most here: a section run refuses of the order of a tenth of its
    candidates, and the objective it is standardised against is a Cd of order
    1e-3. ``None`` / ``"sentinel"`` is the legacy path.

    ``stop_rule(i, best) -> bool`` (optional) ends the search as soon as it
    has stopped improving (:class:`optimize.budget.ConvergenceStop`); the
    report's ``result`` block is then a PARTIAL one, carrying every
    evaluation that was paid for — which on this problem is XFOIL time, and
    is exactly why the early stop rebuilds the result instead of raising.

    ``objective="composite"`` maximises the screen's own weighted composite J
    instead of -cd (``score_weights`` / ``score_reference``, see
    :func:`airfoil_run_config`); the report's ``conditions.objective`` block
    names the scalar, the weights and the band's SHA, so a stored run can never
    be read as the other one. ``censored`` picks the right-censored ``cl_max``
    policy for that objective and is reported in the same block, because two
    runs under different policies are not comparable rows."""
    cfg = airfoil_run_config(symmetric=symmetric,
                             re=re, mach=mach, cl_design=cl_design,
                             tc_min=tc_min, cm_max=cm_max, anchor=anchor,
                             wing=wing, twist_order=twist_order,
                             twist_max_deg=twist_max_deg,
                             alpha_max_deg=alpha_max_deg,
                             chord_order=chord_order,
                             chord_max_frac=chord_max_frac,
                             re_strip=re_strip, re_bank=re_bank,
                             objective=objective, score_weights=score_weights,
                             score_reference=score_reference,
                             censored=censored, score_goals=score_goals,
                             goal_penalty=goal_penalty, goal_tol=goal_tol,
                             asf_rho=asf_rho, asf_lambda=asf_lambda,
                             asf_missing=asf_missing,
                             pareto_acqf=pareto_acqf, pareto_q=pareto_q,
                             pareto_criteria=pareto_criteria,
                             optimiser=optimiser, budget=budget, seed=seed,
                             refusal=refusal, feasibility=feasibility,
                             n_init=n_init,
                             sweep_alpha_max_deg=sweep_alpha_max_deg)
    # ``resume`` is the section's continuation: the previous search's
    # evaluations become this one's training set, so a section continued by 8
    # buys 8 XFOIL polars and not its whole prefix again (``run``).
    res = run(cfg, progress_cb=progress_cb, results_dir=results_dir,
              stop_rule=stop_rule, stop_reason=stop_reason, resume=resume)
    cond = {"re": float(re), "mach": float(mach),
            "cl_design": float(cl_design),
            "tc_min": float(tc_min), "cm_max": float(cm_max),
            "seeded": anchor is not None, "wing_mode": bool(wing),
            "objective": str(objective)}
    if str(objective) in AIRFOIL_COMPOSITE_OBJECTIVES:
        _name, ref, w, cens = _airfoil_objective(cfg.flags)
        cond["score"] = {"weights": w.normalised(), "reference_sha": ref.sha,
                         "reference_n": int(ref.n_records),
                         # the censoring policy is part of WHAT WAS SCORED, so
                         # it belongs beside the band and the weights: two runs
                         # under different policies are not comparable rows
                         "censored": str(cens)}
        if str(objective) in AIRFOIL_REFERENCE_OBJECTIVES:
            # the REFERENCE POINT is as much a part of the scalar as the band:
            # the same design scores differently against a different seed, so a
            # stored run states what it was measured from. Free here — the run
            # has already resolved (and cached) these sweeps.
            from . import airfoil as _af
            goals = _resolve_airfoil_goals(
                _af.AirfoilProblem(**_airfoil_kwargs(cfg.flags)), cfg.flags,
                ref, w, cens, str(objective))
            cond["score"]["goals"] = goals.to_dict()
            if str(objective) == "composite_asf":
                # …and the AUGMENTATION weight, for the same reason: rho
                # decides whether this run could return a weakly-Pareto point,
                # so a stored ASF run says which function it maximised.
                _st = _airfoil_goal_settings(cfg.flags, str(objective))
                cond["score"]["asf_rho"] = float(_st[3])
                # …and WHICH scaling coefficient: lambda decides which
                # criterion the max-min is held to, so a stored ASF run that
                # did not say would not be reproducible from its own report.
                cond["score"]["asf_lambda"] = str(_st[4])
                # …and what happened to a weighted criterion the reference
                # point does not carry: "drop" and "band_floor" maximise
                # different functions over the same box, so a stored run that
                # did not say could not be re-derived from its own report.
                cond["score"]["asf_missing"] = str(_st[5])
    if wing:
        from .section_wing import (CHORD_ORDER_NAMES, TWIST_ORDER_NAMES,
                                   WingGuess)
        wg = WingGuess(**{k: (int(v) if k == "n_stations" else float(v))
                          for k, v in dict(wing).items()})
        cond["wing"] = wg.to_dict()
        # in wing mode BOTH are derived — report what was actually flown, not
        # the ignored arguments
        cond["re"] = float(wg.re_mac)
        cond["cl_design"] = float(wg.cl_design)
        cond["twist"] = {"order": int(twist_order),
                         "name": TWIST_ORDER_NAMES[int(twist_order)],
                         "twist_max_deg": float(twist_max_deg),
                         "alpha_max_deg": float(alpha_max_deg)}
        cond["chord"] = {"order": int(chord_order),
                         "name": CHORD_ORDER_NAMES[int(chord_order)],
                         "chord_max_frac": float(chord_max_frac),
                         "taper_baseline": float(wg.taper)}
    out: dict = {
        "config": cfg.to_dict(),
        "conditions": cond,
        "result": res.to_dict(),
        "wall_time_s": float(res.wall_time_s),
    }
    if res.best_x is not None:
        try:                              # spanwise loading + full breakdown
            out["design"] = _json_safe(design_report(cfg, res.best_x))
            # …and the SAME breakdown for what the run started from: the
            # library seed (at zero twist, in wing mode). That is the honest
            # reference for "what did the optimisation actually buy?", and
            # evaluating it here also warms its polar, so the section view's
            # dotted overlay (which is deliberately cache-only) has curves to
            # draw.
            #
            # IN EVERY MODE, not only wing mode. A 2-D (or composite) run used
            # to return no baseline at all, so its result card compared the
            # optimised section against nothing and its polar plot drew ONE
            # curve — the seed's sweep is not the run's own, so the cache-only
            # overlay found nothing and said nothing. The price is one extra
            # XFOIL sweep of a section the user already chose, which is the
            # cheapest number on this page and the only one that makes the
            # rest of it readable.
            if with_baseline:
                seed_rep = _seed_design_report(cfg)
                if seed_rep is not None:
                    out["baseline"] = _json_safe(seed_rep)
        except (ValueError, OSError) as exc:      # a view, never fatal
            out["design_error"] = f"{type(exc).__name__}: {exc}"
    if with_section and res.best_x is not None:
        try:
            sec = section_report(cfg, res.best_x)
            if sec is not None:
                out["section"] = _json_safe(sec)
        except (ValueError, OSError) as exc:      # a view, never fatal
            out["section_error"] = f"{type(exc).__name__}: {exc}"
    return out


def screen_seed_candidates(report: dict, n: int = 6) -> list[dict]:
    """CST refits of the top ``n`` screened sections — the seed SHORTLIST.

    The screen only refits its single winner; this refits the leaders so the
    seed can be chosen by the thing that actually matters (wing L/D at the
    cruise CL) instead of by the composite proxy alone. Pure geometry —
    coordinates come from the sidecar, so no XFOIL and no evicted-file read.
    Sections whose loop cannot be refitted are skipped, not faked.
    """
    from . import airfoil

    out: list[dict] = []
    for rank, row in enumerate(report.get("ranked") or [], start=1):
        if len(out) >= int(n):
            break
        name = row.get("name")
        if not name:
            continue
        try:
            coords = _screen_coords({"name": name, "path": ""})
            w_u, w_l = airfoil.cst_anchor_from_coords(coords)
        except (ValueError, OSError, KeyError):
            continue
        out.append({"name": name, "rank": int(rank),
                    "composite": _finite_or_none(row.get("composite")),
                    "ldcr": _finite_or_none(row.get("ldcr")),
                    "tc": _finite_or_none(row.get("tc")),
                    # THE SECTION'S OWN COORDINATES, not just its refit.
                    # `cst_anchor_from_coords` drops the fitted TE gap (the
                    # Tier B problem fixes dz_te = 0), so drawing a candidate
                    # from its weights draws a sharp trailing edge onto an
                    # aerofoil that has a blunt one — MS3-15Retro is 0.0079c
                    # open. The weights are what will FLY; these are what the
                    # row IS, and a picture of a section should be the
                    # section.
                    "coords": _json_safe(np.asarray(coords, dtype=float)),
                    "te_gap": _te_gap(coords),
                    "w_upper": [float(v) for v in w_u],
                    "w_lower": [float(v) for v in w_l]})
    return out


def _te_gap(coords) -> float:
    """The ordinate gap at the aft-most station of a closed loop, in chords.

    The same rule :func:`airfoil.fit_cst` reads ``dz_te`` by, so a section's
    reported gap and the gap its own fit would recover cannot disagree. A
    sharp-TE file answers 0.0.
    """
    c = np.asarray(coords, dtype=float)
    if c.ndim != 2 or c.shape[0] < 3:
        return 0.0
    return float(abs(c[0, 1] - c[-1, 1]))


def rank_seeds_by_wing_ld(candidates: list[dict], *, wing: dict,
                          tc_min: float = 0.10, cm_max: float = 0.08,
                          twist_order: int = 1, twist_max_deg: float = 6.0,
                          alpha_max_deg: float = 10.0,
                          progress_cb: Callable | None = None) -> dict:
    """Score a seed shortlist by WING L/D at the cruise CL and rank it.

    Each candidate is flown UNTWISTED at the derived cruise point — one real
    XFOIL sweep apiece (seconds each, so keep the shortlist short) — and
    ranked by the objective the optimisation itself maximises, rather than by
    the 2-D composite that produced the shortlist. Candidates that fail
    (unconvergeable, cannot reach the cruise lift, cannot be trimmed) are
    reported with their reason and ranked last; DESIGN-infeasible ones (a gate
    margin below zero) keep their true L/D and are flagged, because the
    optimiser starts from a box around the seed and can move off a violated
    gate — refusing them outright would throw away good starting points.

    Returns ``{"ranked": [...], "best": {...} | None}``; every row carries
    ``ld``, ``feasible``, ``margins`` and the composite rank it came in with,
    so a shell can show WHY the seed changed.
    """
    from . import airfoil, section_wing as sw

    wg = sw.WingGuess(**{k: (int(v) if k == "n_stations" else float(v))
                         for k, v in dict(wing).items()})
    rows: list[dict] = []
    for i, cand in enumerate(candidates):
        row = {k: cand.get(k) for k in ("name", "rank", "composite", "ldcr",
                                        "tc")}
        row.update(ld=None, feasible=False, reason="", margins=None)
        try:
            base = airfoil.AirfoilProblem(
                tc_min=float(tc_min), cm_max=float(cm_max),
                anchor=(np.asarray(cand["w_upper"], dtype=float),
                        np.asarray(cand["w_lower"], dtype=float)))
            prob = sw.SectionWingProblem(
                airfoil=base, wing=wg, twist_order=int(twist_order),
                twist_max_deg=float(twist_max_deg),
                alpha_max_deg=float(alpha_max_deg))
            out = sw.evaluate_section_wing(prob.x0, prob)
        except (ValueError, KeyError, TypeError) as exc:
            row["reason"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
            if progress_cb is not None:
                progress_cb(i + 1, len(candidates), row)
            continue
        if out.get("feasible"):
            g = np.asarray(out["g"], dtype=float)
            row.update(ld=float(out["LD"]),
                       feasible=bool(np.all(g >= 0.0)),
                       margins=[float(v) for v in g],
                       cd=_finite_or_none(out.get("cd")),
                       cm=_finite_or_none(out.get("cm")),
                       e=_finite_or_none(out.get("e")),
                       alpha_root_deg=_finite_or_none(
                           out.get("alpha_root_deg")))
            if not row["feasible"]:
                worst = int(np.argmin(g))
                row["reason"] = (
                    f"{prob.constraint_labels[worst]} violated at the seed "
                    f"({g[worst]:+.3f}) — the optimiser can move off it")
        else:
            row["reason"] = out.get("reason", "evaluation failed")
        rows.append(row)
        if progress_cb is not None:
            progress_cb(i + 1, len(candidates), row)

    # rank: usable seeds by L/D (descending), gate-violating ones after the
    # clean ones at equal L/D, failures last
    rows.sort(key=lambda r: (r["ld"] is None, not r["feasible"],
                             -(r["ld"] if r["ld"] is not None else 0.0)))
    best = next((r for r in rows if r["ld"] is not None), None)
    return {"ranked": rows, "best": best}


def wing_guess_preview(**kwargs) -> dict:
    """Everything the wing AREA GUESS implies, without running any physics.

    Pure derivation (no XFOIL, no optimiser): design lift coefficient
    W/(qS), Reynolds number at the MAC, span, root chord, MAC, dynamic
    pressure and density. This is what lets a shell show the user the
    consequences of their guess as they type it, instead of only after a run.
    Raises ValueError on an unphysical guess (the WingGuess validation)."""
    from .section_wing import WingGuess
    return WingGuess(**{k: (int(v) if k == "n_stations" else float(v))
                        for k, v in kwargs.items()}).to_dict()


#: media a design point can be stated in. "track" is a car rear wing: it
#: flies in AIR at the track's air state, so it shares the air branch — it
#: is named separately only so a caller cannot silently ask for sea water
#: because it typed the wrong string.
DESIGN_POINT_MEDIA = ("air", "water", "track")


def design_point(*, medium: str = "air", W_N: float, V: float,
                 s_ref_m2: float, aspect_ratio: float, taper: float,
                 altitude_m: float = 0.0, depth_m: float | None = None,
                 water: str = "sea") -> dict:
    """Flow state + trim target + planform a MISSION implies, in any medium.

    The medium-general twin of :func:`wing_guess_preview`, which is air-only
    (its WingGuess reads the ISA at an altitude). Both are pure derivation —
    no XFOIL, no optimiser — and both compose the SAME physics rather than
    restating it: ``mission.MissionSpec`` owns rho/mu/q/CL_target on both
    branches (sea water via hydrofoil.RHO_WATER / MU_WATER), and
    ``geometry.Wing`` owns the trapezoid, so the mean aerodynamic chord the
    Reynolds number is quoted at is the one the solvers actually fly.

    Exists because a shell that asks the user for a mission BEFORE it asks
    for a wing needs the section design point (Re, Cl) of a hydrofoil or a
    car rear wing too, and deriving those in the shell would put two
    definitions of "the Reynolds number of this mission" in the repo.

    ``medium`` is one of :data:`DESIGN_POINT_MEDIA`; ``"track"`` shares the
    air branch (a rear wing flies in air) and only names itself so the
    caller's intent is explicit. ``water`` picks which water the water
    branch flies in (:func:`water_kinds`). Returns the same keys
    ``wing_guess_preview`` returns for its own fields, plus ``medium``,
    ``mu``, ``depth_m`` and ``mach`` — the Mach number the speed and the
    altitude already imply (``None`` in water, where it means nothing), so a
    shell never has to ask for a number it can derive. Raises ValueError on
    an unphysical mission.
    """
    from . import geometry
    from .mission import MissionSpec

    if medium not in DESIGN_POINT_MEDIA:
        raise ValueError(f"unknown medium {medium!r}; "
                         f"choose from {DESIGN_POINT_MEDIA}")
    W, v, S = float(W_N), float(V), float(s_ref_m2)
    ar, lam = float(aspect_ratio), float(taper)
    if not (W > 0.0 and v > 0.0 and S > 0.0 and ar > 0.0):
        raise ValueError(
            "W_N, V, s_ref_m2 and aspect_ratio must all be > 0 "
            f"(got {W}, {v}, {S}, {ar})")
    if not 0.0 < lam <= 1.0:
        raise ValueError(f"taper must be in (0, 1], got {lam}")
    spec = MissionSpec(
        W_N=W, V=v, altitude_m=float(altitude_m),
        depth_m=(None if depth_m is None else float(depth_m)),
        medium=("water" if medium == "water" else "air"),
        water=str(water))
    b = math.sqrt(ar * S)                      # AR = b^2/S, by definition
    wing = geometry.Wing(b=b, S=S, taper=lam)
    from .mission import speed_of_sound

    mach = (None if medium == "water"
            else float(v / speed_of_sound(float(altitude_m))))
    return {
        "medium": medium, "water": (water if medium == "water" else None),
        "mach": mach,
        "W_N": W, "v_ms": v, "altitude_m": float(altitude_m),
        "depth_m": (None if depth_m is None else float(depth_m)),
        "s_ref_m2": S, "aspect_ratio": ar, "taper": lam,
        "rho": float(spec.rho), "mu": float(spec.mu), "q": float(spec.q),
        "weight_n": W,
        "cl_design": float(spec.cl_target(S)),
        "b": float(b), "c_root": float(wing.c_root), "mac": float(wing.mac),
        "re_mac": float(spec.rho * v * wing.mac / spec.mu),
    }


#: fluids a FLOW POINT can be stated in. ``"air"`` reads the ISA at an
#: altitude, ``"water"`` is one of :func:`water_kinds`, and ``"custom"`` is
#: the escape hatch that takes rho and mu VERBATIM — a tunnel, a scaled
#: test, a fluid this repo carries no model of.
FLOW_FLUIDS = ("air", "water", "custom")


def flow_point(*, fluid: str = "air", V: float, chord_m: float,
               altitude_m: float = 0.0, water: str = "sea",
               rho: float | None = None, mu: float | None = None,
               mach: float | None = None, cl_design: float = 0.5) -> dict:
    """The flow a SECTION is designed in, stated DIRECTLY — no vehicle.

    :func:`design_point` answers "what point does this MISSION imply?", and
    to get there it needs a weight, a reference area and an aspect ratio:
    the chord and the lift coefficient are consequences of an aircraft. This
    answers the other question, the one an aerofoil study actually asks —
    the flow is already known. A section is designed at exactly three
    numbers (``re``, ``mach``, ``cl_design`` — the operating point every one
    of :func:`optimize_airfoil`, :func:`screen_airfoils`,
    :func:`screen_at_point` and :func:`score_sections` takes), and all three
    can be stated without any vehicle existing at all.

    Two faces of the same point, and the ``fluid`` picks which:

    * ``"air"`` / ``"water"`` — the FLUID is named and the state is derived:
      ISA density, Sutherland viscosity and the speed of sound at
      ``altitude_m`` (:mod:`aerobo.mission`), or the density and viscosity
      of the named water (:func:`water_kinds`). Nothing is typed twice, so
      the Mach number cannot disagree with the altitude it came from.
    * ``"custom"`` — ``rho``, ``mu`` and (optionally) ``mach`` are the
      user's, verbatim. That is what a wind tunnel, a scaled test or a fluid
      outside the ISA/water pair needs, and it is a separate face rather
      than an override BECAUSE the derived faces must never silently ignore
      a number somebody typed: passing ``rho``/``mu``/``mach`` to a named
      fluid raises rather than being dropped.

    The one input neither face can do without is a LENGTH: a Reynolds number
    is not a property of a flow, it is a property of a flow and a body. So
    ``chord_m`` is asked, and ``re = rho V c / mu`` is what the section is
    then screened and optimised at.

    ``cl_design`` is carried (and echoed back) because it is part of the
    section's operating point, not of the flow — it is the third number the
    scoring needs, and ``lift_per_span_n_m = cl_design q c`` says what
    choosing it means in newtons per metre of span.

    Returns a flat, JSON-safe dict::

        {fluid, water, v_ms, chord_m, altitude_m, rho, mu, nu, q, a_ms,
         mach, re, cl_design, lift_per_span_n_m}

    ``a_ms`` is the speed of sound where one is known (air, or a custom
    fluid whose Mach was stated) and ``None`` otherwise; ``mach`` is 0.0 in
    water, where compressibility is not modelled and every polar in this
    repo is run incompressible. Raises ValueError on an unphysical or
    over-stated point.
    """
    from .hydrofoil import water_properties
    from .mission import (isa_density, isa_temperature, speed_of_sound,
                          sutherland_mu)

    if fluid not in FLOW_FLUIDS:
        raise ValueError(f"unknown fluid {fluid!r}; choose from {FLOW_FLUIDS}")
    v, c = float(V), float(chord_m)
    if not (v > 0.0 and c > 0.0):
        raise ValueError(f"V and chord_m must both be > 0 (got {v}, {c})")
    cl = float(cl_design)
    if not math.isfinite(cl):
        raise ValueError(f"cl_design must be finite, got {cl}")

    # A NAMED FLUID DERIVES ITS OWN STATE. Accepting rho/mu/mach here and
    # ignoring them is the exact defect the two faces exist to prevent: a
    # form that lets a density be typed beside an altitude has two answers
    # to one question, and the one that flies is whichever the reader
    # guessed. Refuse instead, and name the face that does take them.
    if fluid in ("air", "water"):
        stated = [n for n, x in (("rho", rho), ("mu", mu), ("mach", mach))
                  if x is not None]
        if stated:
            raise ValueError(
                f"{fluid!r} derives its own state, so {', '.join(stated)} "
                f"would be ignored — state the point as fluid='custom' to "
                f"type them")

    if fluid == "air":
        rho_v = float(isa_density(float(altitude_m)))
        mu_v = float(sutherland_mu(isa_temperature(float(altitude_m))))
        a_ms: float | None = float(speed_of_sound(float(altitude_m)))
        mach_v = v / a_ms
        alt = float(altitude_m)
    elif fluid == "water":
        if float(altitude_m) != 0.0:
            raise ValueError("water has no altitude; leave altitude_m at 0 "
                             "(depth changes neither rho nor mu at Tier "
                             "level — see aerobo.mission)")
        props = water_properties(str(water))
        rho_v, mu_v = float(props["rho"]), float(props["mu"])
        # water is flown incompressible everywhere in this repo: every
        # cached polar and every live XFOIL sweep of a hydrofoil section is
        # run at M = 0, so reporting anything else would be a number no
        # solver honours
        a_ms, mach_v, alt = None, 0.0, 0.0
    else:
        if rho is None or mu is None:
            raise ValueError("a custom fluid is stated by its density and "
                             "its viscosity — pass rho and mu")
        if float(altitude_m) != 0.0:
            raise ValueError("a custom fluid carries no atmosphere; its "
                             "density and viscosity ARE the state, so "
                             "altitude_m must be 0")
        rho_v, mu_v = float(rho), float(mu)
        if not (rho_v > 0.0 and mu_v > 0.0):
            raise ValueError(f"rho and mu must both be > 0 "
                             f"(got {rho_v}, {mu_v})")
        mach_v = 0.0 if mach is None else float(mach)
        if not mach_v >= 0.0 or not math.isfinite(mach_v):
            raise ValueError(f"mach must be >= 0 and finite, got {mach_v}")
        # the speed of sound the pair (V, M) implies, where one was stated:
        # a read-out, never an input, so it cannot disagree with the Mach
        a_ms = (v / mach_v) if mach_v > 0.0 else None
        alt = 0.0

    q = 0.5 * rho_v * v * v
    return {"fluid": fluid,
            "water": (str(water) if fluid == "water" else None),
            "v_ms": v, "chord_m": c, "altitude_m": alt,
            "rho": rho_v, "mu": mu_v, "nu": mu_v / rho_v, "q": float(q),
            "a_ms": a_ms, "mach": float(mach_v),
            "re": float(rho_v * v * c / mu_v),
            "cl_design": cl,
            "lift_per_span_n_m": float(cl * q * c)}


def family_design_point(problem_name: str, flags: dict | None = None
                        ) -> dict:
    """The operating point a problem carries ITSELF — weight, speed, size.

    :func:`default_mission_values` answers "what may the caller override?"
    and is therefore empty for every family whose operating point is not a
    mission (the hydrofoil flies speed and depth as design variables; the
    car's speed is a track condition). A shell that asks for a MISSION
    before it asks for a wing needs the other question answered: what is
    this family's own design point, so its form can open on it instead of
    on some other family's numbers.

    Everything is read off the BUILT problem (the same way
    :func:`planform_size` and :func:`mission_sref` do), so it cannot drift
    from what is flown:

    * ``W_N`` — the lift/weight the surface is designed to carry, with
      ``W_source`` naming where it came from. ``None`` (and an empty source)
      when the family declares no lift target at all — the car rear wing
      MAXIMISES downforce under a drag budget, so it has none, and inventing
      one here would be a number nobody chose;
    * ``V`` / ``V_source`` — the fixed speed, or the middle of the speed
      box when speed is a design variable (``V_source == "design box"``);
    * ``depth_m`` — same treatment for the water families' depth;
    * ``s_ref_m2`` / ``b_m`` — the planform actually flown, or ``None``.

    A SECTION-DESIGNING family is read through its wrapper
    (:func:`_flown_family`). Those problems are the same aircraft with its
    aerofoil in the design vector, so they carry the same design load: asking
    the wrapper produced ``W_N = None`` for 457 registered problems and made
    the V3 mission form re-open at a fabricated reference CL on families that
    publish a design point of their own.
    """
    from .mission import weight_for

    spec = PROBLEM_SPECS[problem_name]
    built = spec.build({}, flags or {}, None)
    prob = _flown_family(built.problem)
    out: dict = {"problem": problem_name, "medium": built.medium,
                 "W_N": None, "W_source": "", "V": None, "V_source": "",
                 "depth_m": None, "altitude_m": 0.0,
                 "s_ref_m2": None, "b_m": None}

    size = planform_size(problem_name, flags)
    if size is not None:
        out["b_m"], out["s_ref_m2"] = float(size[0]), float(size[1])

    def _box_mid(*labels):
        box = spec.default_bounds
        for lbl in labels:
            if lbl in box:
                lo, hi = box[lbl]
                return 0.5 * (float(lo) + float(hi))
        return None

    # ---- speed
    v = getattr(prob, "V", None)
    if isinstance(v, (int, float)) and float(v) > 0.0:
        out["V"] = float(v)
        out["V_source"] = "the problem's own operating point"
    else:
        mid = _box_mid("V_ms", "V")
        if mid is not None:
            out["V"] = float(mid)
            out["V_source"] = "design box"

    # ---- submergence
    depth = getattr(prob, "depth", None)
    if isinstance(depth, (int, float)) and float(depth) > 0.0:
        out["depth_m"] = float(depth)
    else:
        mid = _box_mid("depth_m")
        if mid is not None:
            out["depth_m"] = float(mid)

    # ---- altitude, where the family carries a mission
    mission = getattr(prob, "mission", None)
    if mission is not None and getattr(mission, "altitude_m", None) is not None:
        out["altitude_m"] = float(mission.altitude_m)

    # ---- design lift, in order of how directly the problem states it
    if mission is not None and getattr(mission, "W_N", None):
        out["W_N"] = float(mission.W_N)
        out["W_source"] = "the problem's MissionSpec"
        out["V"] = float(mission.V)
        out["V_source"] = "the problem's MissionSpec"
        return out
    lift = getattr(prob, "L_design", None)
    if isinstance(lift, (int, float)) and float(lift) > 0.0:
        out["W_N"] = float(lift)
        out["W_source"] = "the problem's design lift L_design"
        return out
    cl = getattr(prob, "CL_target", None)
    if isinstance(cl, (int, float)) and out["V"] and out["s_ref_m2"]:
        # the weight this family's own trim target already implies, in the
        # same closed form mission.weight_for documents
        out["W_N"] = float(weight_for(float(cl), out["V"], out["s_ref_m2"]))
        out["W_source"] = f"CL_target {float(cl):g} at the problem's own point"
    return out


def airfoil_twist_orders() -> dict:
    """``{order: display name}`` for the offered twist-law polynomials."""
    from .section_wing import TWIST_ORDER_NAMES, TWIST_ORDERS
    return {int(k): TWIST_ORDER_NAMES[k] for k in TWIST_ORDERS}


def airfoil_chord_orders() -> dict:
    """``{order: display name}`` for the offered chord-law polynomials.

    Order 0 is the fixed straight taper (no chord design variables).
    """
    from .section_wing import CHORD_ORDER_NAMES, CHORD_ORDERS
    return {int(k): CHORD_ORDER_NAMES[k] for k in CHORD_ORDERS}


# =====================================================================
# Airfoil library screen (weighted multi-criterion selection)
# =====================================================================
#
# The GDP three-step design flow (airfoil_select.py): fix the criterion
# weights, screen a coordinate database at the mission Re/Mach, rank by the
# weighted composite, pick the best. This is the LIBRARY-SELECTION twin of
# the 8-D CST BO problem: instead of shape-optimising one section it chooses
# the best-scoring KNOWN aerofoil, which is near-instant when the screen
# checkpoint is warm (every candidate's XFOIL polar is SHA-cached).

#: the fetched UIUC coordinate database and the reusable screen checkpoint.
UIUC_DB_DIR = Path(__file__).resolve().parents[2] / "data" / "airfoils" / "uiuc"
SCREEN_CHECKPOINT = (Path(__file__).resolve().parents[2] / "results"
                     / "airfoil_screen_checkpoint.json")
#: name -> closed-loop coords sidecar. The screen ranks entirely from the
#: checkpoint metrics, but drawing a winner's shape needs its coordinates, and
#: the .dat database lives on an iCloud-evicted volume where a cold read blocks
#: for tens of seconds. This sidecar caches the parsed loop for every screened
#: section so the result view never touches the raw (possibly evicted) file.
SCREEN_COORDS_CACHE = (Path(__file__).resolve().parents[2] / "results"
                       / "airfoil_screen_coords.json")

_SCREEN_COORDS: dict | None = None      # process-level memo of the sidecar


def library_database_available() -> bool:
    """Is the UIUC ``.dat`` library on this machine?

    ``data/airfoils/uiuc/`` is gitignored (2174 files), so a fresh clone has
    no library to screen and :func:`screen_airfoils` raises out of a glob.
    Two tests in ``test_api.py`` screen it; they ask this first and skip
    with a reason, because an absent input is not a failing assertion.
    """
    return any(UIUC_DB_DIR.glob("*.dat"))


def library_section_available(name: str) -> bool:
    """Can this named library section be READ on this machine?

    Two places hold one: the coordinate sidecar (``results/``, gitignored)
    and the UIUC ``.dat`` tree (``data/airfoils/uiuc/``, gitignored too, and
    2174 files). A fresh clone has neither, so a test or a shell that names
    a section has to be able to ask BEFORE it builds — the alternative is
    ``FileNotFoundError`` out of a builder, which is what a fresh clone got.

    Public because the answer is not a test's business alone: a menu that
    offers `hg40` on a machine that cannot open it is offering nothing.
    """
    if _coords_sidecar().get(str(name)) is not None:
        return True
    return (UIUC_DB_DIR / f"{name}.dat").exists()


def _coords_sidecar() -> dict:
    """Load (and memoise) the name -> coords sidecar.

    Absent, it is BUILT from the coordinate database the first time anything
    asks. A downloaded copy has every ``.dat`` file beside it and has never
    run a screen, and the sidecar's reason to exist — a database on an
    evicted volume — does not apply to it; without this, the fin and endplate
    stages refused a fresh copy ("the section coordinate cache is not built")
    until the wing's screen had been run once. Empty only when there is no
    database to build from either.
    """
    global _SCREEN_COORDS
    if _SCREEN_COORDS is None:
        try:
            _SCREEN_COORDS = json.loads(SCREEN_COORDS_CACHE.read_text())
        except (OSError, json.JSONDecodeError):
            _SCREEN_COORDS = {}          # set FIRST: the builder reads it back
            if UIUC_DB_DIR.is_dir():
                try:
                    build_screen_coords_cache(
                        names=[p.stem for p in sorted(UIUC_DB_DIR.glob("*.dat"))],
                        verbose=False)
                except OSError:
                    pass
    return _SCREEN_COORDS


#: how much camber still counts as SYMMETRIC, as a fraction of chord.
#:
#: Not zero, because a digitised library section is not analytically
#: symmetric: measured over the 2174-section sidecar, a NACA 0012 comes back
#: at 0.00000 and the cambered sections start well above this — GOE741, the
#: one that prompted the filter, is at 0.04802. Anything under half a per
#: cent of chord is a coordinate-file artefact rather than camber somebody
#: designed in.
SYMMETRIC_CAMBER_TOL = 0.005


def section_max_camber(coords) -> float:
    """Max |mean line| of a closed section loop, as a fraction of chord.

    Measured rather than read off a name: "is this section symmetric" has to
    be a property of the COORDINATES, or a library member whose name says
    nothing (and most of them do) is judged by its spelling.
    """
    xy = np.asarray(coords, dtype=float)
    if xy.ndim != 2 or xy.shape[0] < 8:
        return float("inf")
    i = int(np.argmin(xy[:, 0]))
    up, lo = xy[:i + 1][::-1], xy[i:]
    if up.shape[0] < 3 or lo.shape[0] < 3:
        return float("inf")
    xs = np.linspace(0.02, 0.98, 60)
    zu = np.interp(xs, up[:, 0], up[:, 1])
    zl = np.interp(xs, lo[:, 0], lo[:, 1])
    return float(np.max(np.abs(0.5 * (zu + zl))))


def symmetric_section_names(tol: float = SYMMETRIC_CAMBER_TOL) -> tuple:
    """The library members a VERTICAL STABILISER may be given, measured.

    A fin at zero sideslip must make no side force, so a cambered section on
    one is a permanent side load the trim solve has nothing to balance —
    ``vlm.VerticalSurface`` pins its zero-lift angle to 0 for that reason.
    Screening the whole library for a fin therefore offers, and can pick,
    sections the aeroplane cannot fly: GOE741 won a fin screen at 4.8 % max
    camber, which is what this exists to stop.

    Returns ``()`` when the coordinate sidecar is absent — the caller must
    treat that as "cannot restrict" and say so, NOT as "nothing qualifies".
    """
    out = [name for name, xy in _coords_sidecar().items()
           if section_max_camber(xy) <= float(tol)]
    return tuple(sorted(out))


def _screen_coords(rec: dict):
    """Closed-loop coords for a screened section: sidecar first (fast), raw
    .dat fallback (may block on an evicted volume — the reason the sidecar
    exists)."""
    hit = _coords_sidecar().get(rec["name"])
    if hit is not None:
        return np.asarray(hit, dtype=float)
    from .airfoil_select import load_airfoil_dat
    return load_airfoil_dat(rec["path"])


def build_screen_coords_cache(names=None, db_dir=None, verbose=True) -> int:
    """One-time builder for SCREEN_COORDS_CACHE: parse each section's .dat once
    and store its closed loop, so the GUI never reads the evicted database.

    ``names`` restricts the build (default: every eligible section in the
    checkpoint). Returns the number of sections written. Reading the raw files
    IS the slow step (iCloud materialisation) — run it in the background once;
    thereafter every screen result opens instantly."""
    from .airfoil_select import load_airfoil_dat
    db = Path(db_dir) if db_dir is not None else UIUC_DB_DIR
    if names is None:
        try:
            cp = json.loads(SCREEN_CHECKPOINT.read_text())
            names = [k for k, v in cp.items() if v.get("eligible")]
        except (OSError, json.JSONDecodeError):
            names = [p.stem for p in sorted(db.glob("*.dat"))]
    out = dict(_coords_sidecar())          # extend, never lose prior work
    n0 = len(out)
    for i, name in enumerate(names):
        if name in out:
            continue
        try:
            coords = load_airfoil_dat(db / f"{name}.dat")
        except (ValueError, OSError):
            continue
        out[name] = [[float(x), float(y)] for x, y in coords]
        if verbose and (i + 1) % 50 == 0:
            print(f"coords cache: {i + 1}/{len(names)}", flush=True)
    SCREEN_COORDS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    tmp = SCREEN_COORDS_CACHE.with_suffix(f".tmp{uuid.uuid4().hex}")
    tmp.write_text(json.dumps(out))
    tmp.replace(SCREEN_COORDS_CACHE)
    global _SCREEN_COORDS
    _SCREEN_COORDS = out
    return len(out) - n0

#: name -> pre-stall polar BRANCH sidecar, keyed by ONE operating point.
#:
#: Why it exists: the screen checkpoint stores FINISHED metrics, several of
#: which (cd_at, cm_at, alpha_at, ldcr, and the no_cl_bracket status) depend on
#: the design lift coefficient. With a warm checkpoint and ``trust_cache`` the
#: screen never re-ran them, so changing the design Cl silently returned the
#: ranking computed at the checkpoint's Cl — a knob the model was not
#: honouring. Re-deriving them needs only the pre-stall (cl, cd, cm, alpha)
#: branch of each cruise polar, which is what this sidecar holds: the
#: recomputation is then pure interpolation in memory (milliseconds for the
#: whole database) instead of thousands of cache reads on an iCloud-evicted
#: volume, where a single cold read can block for over a minute.
#:
#: Re / Mach / panelling changes are NOT recoverable this way — they need
#: genuinely new XFOIL polars — so those are REPORTED as unhonoured rather
#: than silently approximated (see :func:`screen_point_status`).
SCREEN_BRANCH_CACHE = (Path(__file__).resolve().parents[2] / "results"
                       / "airfoil_screen_branch.json")

#: the CORRECTED twin of the sidecar above (scripts/regen_screen_sidecar.py).
#:
#: The v1 file's ``clmax_censored`` flags were baked under the OLD rule, which
#: compared the cl_max alpha against the TOP OF THE REQUESTED sweep. The rule
#: `polar_metrics` has used since is the physical one: a cl_max is censored iff
#: it sits at the LAST CONVERGED alpha of the combined cruise+stall sweep,
#: because XFOIL silently drops every alpha whose boundary-layer march
#: diverged, so a march that dies early leaves cl still climbing. v2 replays
#: that rule against the cached full polars (no XFOIL) and **284 of 2169 flags
#: change** — 266 False -> True and 16 True -> False, taking the censored share
#: from 5.0 % to 16.4 %. Reading v1 therefore under-reports censoring by a
#: factor of three, and censoring is the single largest refusal category in
#: the section problem (report §16.3a).
#:
#: v2 is preferred where present and v1 is the fallback, because `results/` is
#: gitignored: a fresh clone may have neither, one, or both. Two sections
#: (``bw050209``, ``hn354a``) are recorded ``"unknown"`` there — the evidence
#: needed to apply the rule is no longer in the cache — and are read as the
#: conservative True, the same default every other unknown takes.
SCREEN_BRANCH_CACHE_V2 = (Path(__file__).resolve().parents[2] / "results"
                          / "airfoil_screen_branch_v2.json")

#: ...and the TRACKED copy, because ``results/`` is gitignored and two tests
#: in ``test_api.py`` read this file. On a fresh clone they did not fail
#: loudly: ``_branch_sidecar`` returns ``{}`` for an absent file, so
#: ``rederived`` came back 2174 instead of 0 and the shortlist was empty —
#: a wrong answer, not an error. The repo's rule is that a record belongs in
#: the tracked tree (``records/``), and this is a record: the censoring
#: flags behind report 16.3a's "16.4 %".
#:
#: READ-ONLY FALLBACK. ``results/`` still wins whole when it has anything,
#: and ``build_screen_branch_cache`` still WRITES there, so a rebuild lands
#: where it always did and cannot half-shadow the tracked copy.
SCREEN_BRANCH_TRACKED = (Path(__file__).resolve().parents[2] / "records"
                         / "airfoil_screen_branch_v2.json")

_SCREEN_BRANCH: dict | None = None      # process-level memo of the sidecar
_SCREEN_BRANCH_SOURCE: str = ""         # which file the memo came from


def screen_point(prob) -> dict:
    """The cl-INDEPENDENT part of a screening operating point.

    Two screens sharing this dict share every XFOIL polar, so one can be
    re-derived from the other by interpolation alone; differing on it means
    different polars and therefore real XFOIL work.

    Defined in ``airfoil_select`` (the screen keys its own records by it) and
    re-exported here, so there is exactly one definition of "the same point".
    """
    from .airfoil_select import screen_point as _point
    return _point(prob)


def screen_checkpoint(prob) -> Path:
    """The checkpoint file that holds records screened at ``prob``'s point.

    ONE FILE PER POINT. A screened record's metrics belong to its operating
    point, so re-screening at a second Reynolds number under one filename
    would have the two points evict each other section by section. The
    default point keeps the shipped path (:data:`SCREEN_CHECKPOINT`) — a warm
    library checkpoint stays warm, bit-for-bit — and every other point gets
    its own file named after a short digest of the point itself.
    """
    from .airfoil_select import LEGACY_SCREEN_POINT

    pt = screen_point(prob)
    if pt == LEGACY_SCREEN_POINT:
        return SCREEN_CHECKPOINT
    tag = hashlib.sha256(
        json.dumps(pt, sort_keys=True).encode()).hexdigest()[:12]
    return SCREEN_CHECKPOINT.with_name(
        f"{SCREEN_CHECKPOINT.stem}_{tag}.json")


def _branch_sidecar() -> dict:
    """Load (and memoise) the branch sidecar; ``{}`` when absent.

    Prefers :data:`SCREEN_BRANCH_CACHE_V2` (the corrected censoring flags) and
    falls back to :data:`SCREEN_BRANCH_CACHE`. Both files carry the same
    fields for the same 2169 sections at the same operating point; only the
    ``clmax_censored`` flags differ, and v2's are the ones the current
    ``polar_metrics`` rule produces.

    An EMPTY memo is retried on the next call rather than cached forever: the
    sidecar is built by a slow background pass, and a long-running shell that
    started before it landed must be able to pick it up."""
    global _SCREEN_BRANCH, _SCREEN_BRANCH_SOURCE
    if not _SCREEN_BRANCH:
        for path in _branch_sidecar_order():
            try:
                _SCREEN_BRANCH = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if _SCREEN_BRANCH:
                _SCREEN_BRANCH_SOURCE = path.name
                break
        else:
            _SCREEN_BRANCH = {}
            _SCREEN_BRANCH_SOURCE = ""
    return _SCREEN_BRANCH


def _branch_sidecar_order() -> tuple[Path, ...]:
    """(preferred, fallback) sidecar paths.

    v2 is preferred ONLY while it is still a correction OF THE v1 FILE ON
    DISK. `build_screen_branch_cache` writes v1, under the current (correct)
    censoring rule — so a rebuilt v1 makes the derived v2 stale, and silently
    preferring a stale derivative would be the worst of both. v2 records the
    sha256 of the v1 file it was generated from, so the check is exact rather
    than a guess about mtimes: sha matches (or v1 is absent) -> v2 first;
    otherwise v1 ONLY and v2 is ignored.

    "Ignored" is literal, and it used to be a demotion. With a stale v2 the
    order was ``(v1, v2)``, and :func:`_branch_sidecar` falls THROUGH an
    unreadable file — so a CORRUPT v1 promoted the stale v2 and
    :func:`screen_branch_source` reported ``corrected: True``, about a
    correction of a file that no longer parses, against this docstring.
    Narrow (the writes are atomic), but the V2 library page footnotes that
    flag, and a footnote derived from a contradiction is worse than no
    branch data at all.
    """
    if not any(p.exists() for p in (SCREEN_BRANCH_CACHE_V2,
                                    SCREEN_BRANCH_CACHE)):
        # a fresh clone: ``results/`` is gitignored, so read the tracked
        # copy. Only when there is NOTHING live — a stale or corrupt
        # ``results/`` pair must keep its own resolution below, or this
        # would resurrect exactly the stale-v2 promotion that paragraph is
        # about.
        return (SCREEN_BRANCH_TRACKED,)
    try:
        prov = json.loads(SCREEN_BRANCH_CACHE_V2.read_text()).get(
            "provenance", {})
    except (OSError, json.JSONDecodeError):
        return (SCREEN_BRANCH_CACHE,)
    recorded = str(prov.get("source_sidecar_sha256") or "")
    try:
        live = hashlib.sha256(SCREEN_BRANCH_CACHE.read_bytes()).hexdigest()
    except OSError:
        return (SCREEN_BRANCH_CACHE_V2, SCREEN_BRANCH_CACHE)
    if recorded and recorded != live:
        return (SCREEN_BRANCH_CACHE,)
    return (SCREEN_BRANCH_CACHE_V2, SCREEN_BRANCH_CACHE)


def screen_branch_source() -> dict:
    """Which sidecar the screen's branch data is coming from, for display.

    ``{"file": name, "corrected": bool, "n_sections": int}`` — the GUIs
    footnote it, because "5 % of the library is censored" and "16 % is" are
    different statements about the same library and the page must say which
    one it is making.
    """
    blob = _branch_sidecar()
    return {
        "file": _SCREEN_BRANCH_SOURCE,
        "corrected": _SCREEN_BRANCH_SOURCE == SCREEN_BRANCH_CACHE_V2.name,
        "n_sections": len(blob.get("sections", {})),
    }


def build_screen_branch_cache(prob=None, names=None, verbose=True,
                              workers: int = 16) -> int:
    """One-time builder for :data:`SCREEN_BRANCH_CACHE` at ``prob``'s point.

    For every screened section: read its cruise (and stall) polar from the
    XFOIL disk cache — CACHE-ONLY, never a fresh run — and store the
    cl-independent metrics plus the pre-stall monotone branch the cl-dependent
    ones are interpolated from. Coordinates come from the coords sidecar, so
    the raw (possibly evicted) .dat database is never touched.

    Slow ONCE (thousands of small reads, each of which may have to be
    materialised from cloud storage — a single cold read can block for over a
    minute, which is why the reads are threaded) and instant thereafter — the
    same trade-off :func:`build_screen_coords_cache` makes. Returns the number
    of sections written. Rebuilding at a DIFFERENT point replaces the file:
    the sidecar describes exactly one operating point and says which.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from . import airfoil
    from .airfoil_select import STALL_ALPHAS, geometric_tc, polar_metrics

    prob = prob or airfoil.AirfoilProblem()
    coords_by_name = _coords_sidecar()
    if names is None:
        try:
            names = sorted(json.loads(SCREEN_CHECKPOINT.read_text()))
        except (OSError, json.JSONDecodeError):
            names = sorted(coords_by_name)

    def one(name):
        raw = coords_by_name.get(name)
        if raw is None:
            return name, None
        coords = np.asarray(raw, dtype=float)
        pol = _cached_dat_polar(prob, coords)
        if pol is None:
            return name, None
        stall = _cached_dat_polar(prob, coords, alphas=STALL_ALPHAS)
        # cl-INDEPENDENT metrics: taken from polar_metrics itself so the
        # sidecar can never drift from the screen's own definitions.
        m = polar_metrics(pol, coords, prob, pol_stall=stall)
        ent = {"tc": float(geometric_tc(coords)),
               "n_converged": int(pol.n_converged),
               "clmax": _finite_or_none(m.get("clmax")),
               "astall": _finite_or_none(m.get("astall")),
               "clmax_censored": bool(m.get("clmax_censored", True)),
               "ldmax": _finite_or_none(m.get("ldmax"))}
        if m["status"] == "unconverged":
            ent["status"] = "unconverged"
            ent["reason"] = m["reason"]
        else:
            sl = airfoil._longest_increasing_run(pol.cl)
            ent["cl"] = _json_safe(np.asarray(pol.cl, float)[sl])
            ent["cd"] = _json_safe(np.asarray(pol.cd, float)[sl])
            ent["cm"] = _json_safe(np.asarray(pol.cm, float)[sl])
            ent["alpha"] = _json_safe(np.asarray(pol.alpha_deg, float)[sl])
        return name, ent

    out: dict = {}
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
        futs = [ex.submit(one, n) for n in names]
        for i, fut in enumerate(as_completed(futs)):
            name, ent = fut.result()
            if ent is not None:
                out[name] = ent
            if verbose and (i + 1) % 50 == 0:
                print(f"branch cache: {i + 1}/{len(names)} "
                      f"({len(out)} stored)", flush=True)
    if not out:
        # nothing was cached at this point: writing an empty sidecar would
        # REPLACE a good one with a file claiming a point it cannot serve
        if verbose:
            print("branch cache: no polars cached at this point — "
                  "nothing written", flush=True)
        return 0
    blob = {"point": screen_point(prob), "sections": out}
    SCREEN_BRANCH_CACHE.parent.mkdir(parents=True, exist_ok=True)
    tmp = SCREEN_BRANCH_CACHE.with_suffix(f".tmp{uuid.uuid4().hex}")
    tmp.write_text(json.dumps(blob))
    tmp.replace(SCREEN_BRANCH_CACHE)
    global _SCREEN_BRANCH, _SCREEN_BRANCH_SOURCE
    # a rebuild makes any derived v2 file stale, and this blob was built under
    # the CURRENT censoring rule, so it is the authority from here on
    _SCREEN_BRANCH = blob
    _SCREEN_BRANCH_SOURCE = SCREEN_BRANCH_CACHE.name
    return len(out)


def screen_library_point() -> dict | None:
    """The operating point the branch sidecar covers — i.e. the Re/Mach the
    library can be screened at INSTANTLY and at any design Cl. ``None`` when
    the sidecar has not been built (:func:`build_screen_branch_cache`)."""
    pt = _branch_sidecar().get("point")
    return dict(pt) if pt else None


def screen_point_status(prob) -> dict:
    """What the branch sidecar can honour for ``prob``'s operating point.

    ``matched`` True means every cl-dependent metric can be recomputed EXACTLY
    at this design Cl (same polars, pure interpolation). False means the
    request differs in Re / Mach / panelling / alpha sweep, which no amount of
    interpolation can supply — the caller must either screen for real or say
    plainly that the ranking rests on the cached point.
    """
    br = _branch_sidecar()
    pt = br.get("point")
    want = screen_point(prob)
    return {"matched": bool(pt is not None and pt == want),
            "requested": want, "cached": pt,
            "n_sections": len(br.get("sections") or {})}


def _rederive_screen_record(rec: dict, ent: dict, cl_design: float) -> dict:
    """A screened record's metrics recomputed at ``cl_design`` from a branch.

    Mirrors :func:`airfoil_select.polar_metrics` field for field for the
    cl-dependent half (cd_at / cm_at / alpha_at / ldcr and the
    ``no_cl_bracket`` status); the cl-independent half is carried over from
    the sidecar entry. Gate-derived ``eligible`` is deliberately NOT set here:
    scoring re-derives it from the raw metrics at the caller's gates.
    """
    nan = float("nan")
    out = dict(rec)
    out.update({"tc": ent["tc"], "n_converged": ent["n_converged"],
                "clmax": ent["clmax"] if ent["clmax"] is not None else nan,
                "astall": ent["astall"] if ent["astall"] is not None else nan,
                # bool(), not the raw field: the corrected v2 sidecar records
                # "unknown" for the two sections whose evidence has been
                # evicted from the cache, and every consumer downstream treats
                # a censored flag as a boolean. "unknown" reads as True, which
                # is the conservative default the rest of the pipeline uses.
                "clmax_censored": bool(ent["clmax_censored"]),
                "ldmax": ent["ldmax"] if ent["ldmax"] is not None else nan,
                "cd_at": nan, "cm_at": nan, "alpha_at": nan, "ldcr": nan,
                "status": "ok", "reason": ""})
    if ent.get("status") == "unconverged":
        out.update(status="unconverged", reason=ent.get("reason", ""))
        return out
    cl_b = np.asarray(ent["cl"], dtype=float)
    cd_b = np.asarray(ent["cd"], dtype=float)
    cl_d = float(cl_design)
    if cl_b.size < 2 or cl_b[-1] < cl_d or cl_b[0] > cl_d:
        out.update(status="no_cl_bracket",
                   reason="cl_design not bracketed by the pre-stall "
                          "monotone branch")
        return out
    cd_at = float(np.interp(cl_d, cl_b, cd_b))
    out["cd_at"] = cd_at
    out["cm_at"] = float(np.interp(cl_d, cl_b,
                                   np.asarray(ent["cm"], dtype=float)))
    out["alpha_at"] = float(np.interp(cl_d, cl_b,
                                      np.asarray(ent["alpha"], dtype=float)))
    if cd_at <= 0.0 or not np.isfinite(cd_at):
        out.update(status="bad_cd", reason=f"cd_at = {cd_at}")
        return out
    # |cl|/cd — the magnitude, for the reason airfoil_select.polar_metrics
    # gives: a trimming surface's design lift is negative, and a signed
    # ratio would rank the database upside down. Unchanged for cl_d > 0.
    out["ldcr"] = abs(cl_d) / cd_at
    return out


def _record_branch_entry(rec: dict) -> dict | None:
    """A sidecar-shaped entry built from a record's OWN stored branch.

    The branch sidecar covers exactly one point (the shipped library one).
    A record screened at any other point carries its own branch instead
    (``airfoil_select.screen_one``), and this adapts it to the same shape so
    there is one re-derivation path rather than two.
    """
    branch = rec.get("branch")
    status = rec.get("status")
    if status == "unconverged":
        return {"tc": rec.get("tc"), "n_converged": rec.get("n_converged", 0),
                "clmax": None, "astall": None, "clmax_censored": True,
                "ldmax": None, "status": "unconverged",
                "reason": rec.get("reason", "")}
    if not branch:
        return None
    return {"tc": rec.get("tc"), "n_converged": rec.get("n_converged", 0),
            "clmax": rec.get("clmax"), "astall": rec.get("astall"),
            "clmax_censored": bool(rec.get("clmax_censored", True)),
            "ldmax": rec.get("ldmax"),
            "cl": branch["cl"], "cd": branch["cd"],
            "cm": branch["cm"], "alpha": branch["alpha"]}


#: metric labels for the screen result table / weight editor (display order).
SCREEN_METRICS = (
    ("ldcr", "L/D at design Cl"),
    ("ldmax", "(L/D) max"),
    ("clmax", "Cl max"),
    ("astall", "stall angle [deg]"),
    ("thick", "t/c"),
    ("cm", "|Cm| at design Cl"),
    # the drag at the design lift, lower-better. Beside the two efficiency
    # criteria on purpose: it is the one of the three that still means
    # something on a surface whose design lift is ZERO (a vertical
    # stabiliser), where both ratios above are measured at a lift it never
    # carries — see airfoil_select.LOWER_BETTER.
    ("cdcr", "cd at design Cl"),
)

#: floor keys the screen accepts as MINIMUMS (higher-is-better metrics only;
#: t/c and |Cm| are the AirfoilProblem's own two gates, edited via tc_min /
#: cm_max rather than here).
SCREEN_FLOOR_KEYS = ("clmax", "ldcr", "ldmax", "astall")

#: The one floor that is not a higher-is-better METRIC: how NOSE-DOWN the
#: section's moment has to be (it gates -cm, so 0.0 admits cm <= 0 and
#: rejects every reflexed section). Kept out of SCREEN_FLOOR_KEYS because
#: that tuple is what a shell builds its "minimum" number rows from, and this
#: is a yes/no property of the aircraft rather than a number to tune: a wing
#: that HAS a trimming surface should not be handed an aerofoil designed to
#: not need one. See airfoil_select.NOSE_DOWN_FLOOR for the measurement.
SCREEN_NOSE_DOWN_KEY = "nose_down"


def screen_weights(weights: dict | str | None):
    """Resolve a ScoreWeights from a preset name, a partial dict, or None.

    A dict may set any subset of ``airfoil_select.CRITERIA`` — the rest fall
    back to the GDP default preset; auto-normalised downstream. The accepted
    keys are READ OFF the criterion set rather than restated here: a criterion
    the engine scores but this function refuses is a weight the user can set
    in one shell and not in another (``cdcr``, the drag at the design lift,
    arrived exactly that way).
    """
    from .airfoil_select import CRITERIA, PRESETS, ScoreWeights
    if weights is None:
        return ScoreWeights()
    if isinstance(weights, str):
        if weights not in PRESETS:
            raise ValueError(f"unknown weight preset {weights!r}; "
                             f"choose from {sorted(PRESETS)}")
        return PRESETS[weights]
    if isinstance(weights, ScoreWeights):
        return weights
    base = {f: getattr(ScoreWeights(), f) for f in CRITERIA}
    for k, v in dict(weights).items():
        if k not in base:
            raise ValueError(f"unknown weight key {k!r}; keys={sorted(base)}")
        base[k] = float(v)
    return ScoreWeights(**base)


def _dat_thickness_profile(coords, n: int = 1001):
    """(x/c, t/c) of a raw coordinate loop on a cosine grid — the same recipe
    airfoil_select.geometric_tc measures t/c from, so the drawn max matches."""
    from .airfoil_select import split_surfaces
    xg, yu, yl = split_surfaces(np.asarray(coords, float), n_grid=n)
    return xg, (yu - yl)


def _screen_winner_design(prob, rec: dict) -> dict:
    """A section_report-style ``design`` block for a screened winner, so the
    GUI reuses the very same shape/polar figures the CST view draws.

    Uses the REAL library coordinates (not a CST refit) and the cached cruise
    polar the screen already computed (cache-only — opening the result never
    spends XFOIL time)."""
    coords = _screen_coords(rec)
    xg, t = _dat_thickness_profile(coords)
    i = int(np.argmax(t))
    design = {
        "name": rec["name"],
        "coords": _json_safe(np.asarray(coords, float)),
        "tc": float(rec.get("tc", float(t[i]))),
        "tc_max_xc": float(xg[i]),
        "thickness_xc": _json_safe(xg),
        "thickness": _json_safe(t),
    }
    pol = _cached_dat_polar(prob, coords)
    if pol is not None:
        design["polar"] = _polar_payload(pol, prob)
    return design


def _cached_dat_polar(prob, coords, alphas=None):
    """Polar for a raw section, CACHE-ONLY (None on a miss).

    ``alphas`` defaults to the problem's cruise sweep; pass
    ``airfoil_select.STALL_ALPHAS`` for the wide stall sweep.
    """
    from .xfoil_run import (DEFAULT_CACHE_DIR, _cache_load, cache_key,
                            run_xfoil_polar)
    coords = np.asarray(coords, float)
    alphas = prob.alphas if alphas is None else alphas
    key = cache_key(coords, prob.re, prob.mach, list(alphas), prob.n_panel)
    root = Path(prob.cache_dir if prob.cache_dir is not None
                else DEFAULT_CACHE_DIR)
    # LOADABLE, not merely present — see cached_polar: a pre-gate entry has
    # no CDp column and is a miss, so exists() would turn this cache-only
    # helper into a synchronous XFOIL solve.
    if _cache_load(root / f"{key}.json", len(list(alphas))) is None:
        return None
    return run_xfoil_polar(coords, prob.re, prob.mach, alphas,
                           timeout_s=prob.timeout_s, cache_dir=prob.cache_dir,
                           n_panel=prob.n_panel)


def _screen_row(rec: dict) -> dict:
    """A ranked record trimmed to the JSON-safe fields the table needs."""
    keys = ("name", "tc", "clmax", "astall", "clmax_censored", "ldmax",
            "ldcr", "cd_at", "cm_at", "composite")
    row = {k: rec.get(k) for k in keys}
    row["scores"] = {m: rec.get(f"score_{m}")
                     for m, _ in SCREEN_METRICS if f"score_{m}" in rec}
    return _json_safe(row)


def screen_airfoils(
    weights: dict | str | None = None,
    *,
    re: float = 1e6,
    mach: float = 0.0,
    cl_design: float = 0.5,
    tc_min: float = 0.15,
    cm_max: float = 0.08,
    floors: dict | None = None,
    n_candidates: int | None = None,
    names=None,
    workers: int = 6,
    top_n: int = 12,
    db_dir=None,
    checkpoint=None,
    reference=None,
    progress_cb=None,
    cancel=None,
    verbose: bool = False,
    trust_cache: bool = True,
    with_shape: bool = True,
    measure_reference: bool = False,
) -> dict:
    """Weighted multi-criterion selection of the best KNOWN aerofoil.

    Screens the UIUC coordinate database at (``re``, ``mach``, ``cl_design``),
    scores each eligible section by the GDP composite of six criteria
    (L/D at the design Cl, (L/D)max, Cl_max, stall angle, t/c, |Cm|) under
    the caller's ``weights``, and returns the ranking plus the winner's shape
    and polar. ``tc_min`` (default 0.15) and ``cm_max`` are the two hard
    gates; ``floors`` adds optional minimums on the higher-is-better metrics
    (SCREEN_FLOOR_KEYS). Near-instant on a warm checkpoint — every candidate's
    polar is SHA-cached, so no XFOIL runs unless a .dat changed.

    ``names`` restricts the screen to those sections (a SHORTLIST re-screen —
    see :func:`screen_at_point`); ``n_candidates`` truncates the file list
    instead (a smoke run). ``reference`` freezes the normalisation band so two
    screens over different populations stay comparable;
    ``measure_reference`` measures one over THIS screen's eligible population
    and returns it under ``reference`` for a later pass to be scored on.

    THE OPERATING POINT IS HONOURED OR REPORTED, never quietly substituted.
    Asking for a Reynolds number the checkpoint was not screened at now
    re-runs XFOIL for real (``airfoil_select.screen_database`` keys its cache
    by the point), which is minutes-to-hours over the whole database — the
    reason :func:`screen_at_point` exists. ``point`` in the report says which
    of the requested metrics are the requested point's.

    Returns a JSON-safe report: ``conditions``, ``weights`` (normalised),
    ``floors``, ``n_screened`` / ``n_eligible`` / ``status_counts``,
    ``ranked`` (top ``top_n`` rows), and ``winner`` — a section_report-style
    block (``design`` + ``cl_design`` / ``tc_min`` / ``cm_max``) the GUI draws
    with the same figures as the CST view, plus the winner's CST refit
    (``w_upper`` / ``w_lower``) so the pick can seed the 8-D BO problem.
    """
    from . import airfoil
    from .airfoil_select import screen_database

    db = Path(db_dir) if db_dir is not None else UIUC_DB_DIR
    files = sorted(db.glob("*.dat"))
    if not files:
        raise FileNotFoundError(
            f"no .dat files in {db} — run scripts/fetch_uiuc_db.py first")
    if names is not None:
        want = {str(n) for n in names}
        files = [p for p in files if p.stem in want]
        if not files:
            raise FileNotFoundError(
                f"none of the {len(want)} requested sections is in {db}")
    if n_candidates is not None:
        files = files[:int(n_candidates)]

    prob = airfoil.AirfoilProblem(re=float(re), mach=float(mach),
                                  cl_design=float(cl_design),
                                  tc_min=float(tc_min), cm_max=float(cm_max))
    w = screen_weights(weights)
    floors = {k: float(v) for k, v in (floors or {}).items()
              if k in SCREEN_FLOOR_KEYS + (SCREEN_NOSE_DOWN_KEY,)
              and v is not None}
    # ONE CHECKPOINT PER POINT: the default point keeps the shipped file, so a
    # warm library screen is untouched; any other point writes its own.
    checkpoint = (screen_checkpoint(prob) if checkpoint is None
                  else Path(checkpoint))

    t0 = time.time()
    res = screen_database(files, prob, w, workers=workers,
                          checkpoint=Path(checkpoint) if checkpoint else None,
                          verbose=verbose, floors=floors,
                          progress_cb=progress_cb, cancel=cancel,
                          trust_cache=trust_cache, reference=reference)
    wall = time.time() - t0

    records = res["records"]
    ranked = res["ranked"]

    # ---- honour the DESIGN Cl -----------------------------------------
    # A warm checkpoint holds metrics finished at whatever Cl it was screened
    # at, so every cl-dependent metric is recomputed here at the REQUESTED Cl
    # (exact, same polars) from a pre-stall branch, and the ranking is rebuilt
    # from those. Two branch sources, same shape and same arithmetic: the
    # sidecar, which covers the shipped library point for the whole database,
    # and the record's own stored branch, which covers every other point.
    # Anything with neither is BLANKED rather than silently approximated, and
    # ``point`` in the report names it.
    from .airfoil_select import score_candidates

    point = screen_point_status(prob)
    point.update(rederived=0, from_sidecar=0, from_record=0)
    sections = ((_branch_sidecar().get("sections") or {})
                if point["matched"] else {})
    fresh = {}
    stale = []
    for name, r in records.items():
        ent, src = sections.get(name), "sidecar"
        if ent is None:
            ent, src = _record_branch_entry(r), "record"
        if ent is None:
            # No branch to re-score from, and the checkpoint's own
            # cl-dependent metrics were finished at whatever design Cl IT
            # was screened at — which is not necessarily this one. Ranking
            # them anyway let a section screened at another lift
            # coefficient outrank the whole database (at Cl = 0, the five
            # that have no branch entry took the top four places on
            # L/D numbers belonging to Cl = 0.5). So the cl-dependent
            # half is blanked, which makes the record ineligible in
            # scoring, and the report names it.
            fresh[name] = dict(r, cd_at=float("nan"),
                               cm_at=float("nan"),
                               alpha_at=float("nan"),
                               ldcr=float("nan"),
                               status="no_branch",
                               reason="no cached polar branch — its "
                                      "cl-dependent metrics cannot be "
                                      "re-scored at this design Cl")
            stale.append(name)
            continue
        fresh[name] = _rederive_screen_record(r, ent, prob.cl_design)
        point["rederived"] += 1
        point[f"from_{src}"] += 1
    records = fresh
    point["no_branch"] = sorted(stale)
    ranked = score_candidates(list(records.values()), w, floors=floors,
                              tc_min=prob.tc_min, cm_max=prob.cm_max,
                              reference=reference)
    point["not_rederived"] = len(records) - point["rederived"]
    # WHAT THE SHELLS READ. Every RANKED record was measured at the requested
    # point (the screen's cache is keyed by it) and re-derived at the requested
    # design Cl — the ones that could be neither are blanked above, which makes
    # them ineligible, so they cannot reach the table. It used to be possible
    # for neither to hold while the report printed the requested Re and Cl as
    # its own conditions; ``honoured`` is that guarantee, stated.
    point["honoured"] = bool(ranked)
    point["excluded"] = len(stale)

    status_counts: dict[str, int] = {}
    for r in records.values():
        status_counts[r.get("status", "?")] = status_counts.get(
            r.get("status", "?"), 0) + 1

    out = {
        "conditions": {"re": float(re), "mach": float(mach),
                       "cl_design": float(cl_design),
                       "tc_min": float(tc_min), "cm_max": float(cm_max)},
        "weights": res["weights"],
        "floors": floors,
        "n_screened": len(records),
        "n_eligible": len(ranked),
        "status_counts": status_counts,
        "point": point,
        "ranked": [_screen_row(r) for r in ranked[:int(top_n)]],
        "wall_time_s": float(wall),
        "winner": None,
    }
    if measure_reference and ranked:
        from .airfoil_select import build_screen_reference
        try:
            out["reference"] = build_screen_reference(
                list(records.values()), w, preset="",
                tc_min=float(tc_min), cm_max=float(cm_max), floors=floors,
                source=f"screen_airfoils at Re {float(re):.4g}")
        except ValueError:
            pass        # degenerate population: no band, and no claim of one
    if ranked:
        best = ranked[0]
        winner = {
            "cl_design": float(cl_design),
            "tc_min": float(tc_min), "cm_max": float(cm_max),
            "metrics": _screen_row(best),
            "rank": 1,
        }
        # ``with_shape`` gates the only step that reads a raw coordinate file
        # (shape + CST refit): callers that just want the ranking skip the
        # possibly-slow read entirely.
        if with_shape:
            winner["design"] = _screen_winner_design(prob, best)
            try:
                w_u, w_l = airfoil.cst_anchor_from_coords(
                    _screen_coords(best), prob.n_cst)
                winner["w_upper"] = [float(v) for v in w_u]
                winner["w_lower"] = [float(v) for v in w_l]
            except (ValueError, OSError):  # CST refit is a best-effort extra
                pass
        out["winner"] = winner
    return out


#: how many of the library-point ranking's leaders are re-screened at the
#: mission's own point by default. Screening the whole database there is a
#: fresh XFOIL sweep per section (two, with the stall sweep) — hours. The
#: shortlist is the same screen restricted to the sections that could plausibly
#: win: the library point already ranks every section on the same six criteria,
#: and Reynolds number reorders the leaders rather than promoting the tail.
SHORTLIST_N = 24


def screen_at_point(
    weights: dict | str | None = None,
    *,
    re: float,
    mach: float = 0.0,
    cl_design: float = 0.5,
    tc_min: float = 0.15,
    cm_max: float = 0.08,
    floors: dict | None = None,
    shortlist: int = SHORTLIST_N,
    #: restrict the candidate set BEFORE anything is ranked or
    #: shortlisted — a vertical stabiliser may only be given a symmetric
    #: section, and filtering afterwards would shortlist cambered ones
    #: and then throw the shortlist away (see symmetric_section_names)
    names=None,
    top_n: int = 12,
    workers: int = 6,
    db_dir=None,
    progress_cb=None,
    cancel=None,
    verbose: bool = False,
    with_shape: bool = True,
) -> dict:
    """Rank the library at a REQUESTED operating point, in minutes not hours.

    Two passes, and the report says so:

    1. the whole database at the cached library point
       (:func:`screen_library_point`) — instant, and exact in the design Cl,
       because every polar's pre-stall branch is on disk. This pass exists to
       choose WHO is worth a real XFOIL sweep, not to answer the question;
    2. the top ``shortlist`` of that ranking, screened for real at
       (``re``, ``mach``) — a live viscous sweep per section, cached like
       every other polar, so the second visit to the same point is instant.

    Both passes are scored against ONE frozen normalisation band, measured
    over pass 1's eligible population (``build_screen_reference``). Without it
    the GDP composite is min-max normalised across whatever population it is
    handed, so a 24-section table and a 2000-section table would put the same
    six metrics on different scales and the two passes could not be compared
    at all — the shortlist's ranking would be an artefact of its own size.

    Returns pass 2's report with a ``shortlist`` block naming what pass 1 fed
    it, and the pass-1 report under ``library_pass`` so a shell can show the
    reordering. Falls back to a plain :func:`screen_airfoils` at the requested
    point when there is no library cache to shortlist from — correct, just
    slow, and ``shortlist["source"]`` says which happened.
    """
    lib = screen_library_point()
    common = dict(re=float(re), mach=float(mach), cl_design=float(cl_design),
                  tc_min=float(tc_min), cm_max=float(cm_max),
                  floors=floors, workers=workers, db_dir=db_dir,
                  cancel=cancel, verbose=verbose)
    # ASKED FOR THE CACHED POINT: there is nothing to sweep and nothing to
    # shortlist. Restricting the ranking to N sections here would truncate a
    # screen that answers over the whole database instantly.
    at_library = bool(lib) and (
        abs(float(re) - float(lib["re"])) <= 1e-9 * abs(float(re))
        and float(mach) == float(lib.get("mach", 0.0)))
    if not lib or at_library:
        out = screen_airfoils(weights, top_n=top_n, with_shape=with_shape,
                              names=names,
                              progress_cb=progress_cb, **common)
        out["shortlist"] = {
            "source": "cache" if at_library else "none",
            "n": out["n_screened"], "names": [],
            "note": ("this IS the cached point, so the whole library was "
                     "ranked here — no shortlist and no sweep"
                     if at_library else
                     "no cached library screen on this machine, so the whole "
                     "database was screened at the requested point")}
        return out

    # ---- pass 1: the whole library, at the point it is cached at ---------
    first = screen_airfoils(
        weights, re=float(lib["re"]), mach=float(lib.get("mach", 0.0)),
        cl_design=float(cl_design), tc_min=float(tc_min), cm_max=float(cm_max),
        floors=floors, top_n=max(int(shortlist), int(top_n)),
        workers=workers, db_dir=db_dir, verbose=verbose, with_shape=False,
        # the CALLER's restriction, on pass 1 as well: shortlisting from the
        # whole library and filtering afterwards would spend the shortlist on
        # sections the surface cannot use
        names=names,
        cancel=cancel, measure_reference=True)
    # the shortlist is a DIFFERENT list from the caller's restriction, and it
    # used to reuse the name — so pass 2 ran on whichever had been assigned
    # last and the restriction silently expired between the passes
    short = [r["name"] for r in (first.get("ranked") or [])][:int(shortlist)]
    if not short:
        raise ValueError(
            "the library screen ranked nothing at these gates, so there is "
            "no shortlist to re-screen — loosen t/c or |Cm| first")

    # ---- the band both passes are scored on ------------------------------
    # Measured on pass 1 (the big, stable population); pass 2's sections score
    # against the same fixed map even though their metrics move with Re, which
    # is exactly what makes "it dropped four places at the real Reynolds
    # number" a statement about the section rather than about the sample.
    payload = first.get("reference")
    # ONE loader, because this used to be a second one: read verbatim it
    # dropped the payload's ``unbanded`` block, so a fin's band — which is a
    # band, on the six criteria a zero-lift population separates — was refused
    # here and nowhere else, and pass 2 fell back to a live min-max for a
    # reason that no longer existed.
    ref, _ref_info = (_score_reference_arg(payload) if payload else (None, {}))

    # ---- ...and the shortlist chosen on the map pass 2 is SCORED on ------
    # Pass 1 above ranks on a LIVE min-max over its own population, because
    # the band does not exist until it has been measured. Pass 2 is then
    # scored on the frozen band — a different map — so the sections allowed
    # to compete were chosen by one ranking and judged by another. Only the
    # band's WIDTH enters the composite, so the two maps re-weight the
    # criteria against each other: on this library the declared
    # (.10/.20/.15/.35/.20) becomes an effective (.105/.228/.116/.325/.226),
    # and the orderings disagree (Kendall tau 0.804 measured at Re 3e5).
    #
    # Re-ranking the SAME records on the frozen band costs no XFOIL at all —
    # the cached point is on disk, so this call is a cache hit — and the
    # shortlist is then the UNION of the two orderings' leaders: a section
    # either map rates highly gets its real sweep, and neither map silently
    # decides the answer on its own.
    if ref is not None:
        by_band = screen_airfoils(
            weights, re=float(lib["re"]), mach=float(lib.get("mach", 0.0)),
            cl_design=float(cl_design), tc_min=float(tc_min),
            cm_max=float(cm_max), floors=floors,
            top_n=max(int(shortlist), int(top_n)), workers=workers,
            db_dir=db_dir, verbose=verbose, with_shape=False, cancel=cancel,
            # the CALLER's restriction on THIS map too. It is the same
            # library ranked a second way, so a pass that did not carry it
            # unioned in leaders the surface may not use — a fin's screen
            # shortlisted, swept and ranked CAMBERED sections, which is the
            # whole defect ``names`` exists to prevent.
            names=names,
            reference=ref)
        banded = [r["name"] for r in (by_band.get("ranked") or [])
                  ][:int(shortlist)]
        seen = set(short)
        short = short + [n for n in banded if not (n in seen or seen.add(n))]

    # ---- pass 2: the shortlist, at the point that was asked for ----------
    out = screen_airfoils(weights, names=short, top_n=top_n,
                          reference=ref, with_shape=with_shape,
                          progress_cb=progress_cb, **common)
    was = {r["name"]: i + 1 for i, r in enumerate(first.get("ranked") or [])}
    for row in out.get("ranked") or []:
        row["rank_library"] = was.get(row.get("name"))
    out["shortlist"] = {
        "source": "library",
        "n": len(short),
        "names": short,
        "library_point": dict(lib),
        "n_library_eligible": int(first.get("n_eligible", 0)),
        "note": f"the {len(short)} best of {first.get('n_eligible', 0)} "
                f"eligible sections at the cached library point "
                f"(Re {float(lib['re']):.3g}) were re-screened at "
                f"Re {float(re):.3g} — a live XFOIL sweep each. The "
                f"shortlist is the union of the leaders under BOTH maps the "
                f"library point can be ranked on (a live min-max over the "
                f"whole population, and the frozen band pass 2 is scored on), "
                f"so neither map decides on its own. Sections outside that "
                f"shortlist were not swept at this point and cannot appear "
                f"below.",
    }
    if ref is not None:
        out["shortlist"]["reference_sha"] = ref.sha
    out["library_pass"] = {k: first.get(k) for k in
                           ("conditions", "n_screened", "n_eligible",
                            "ranked", "point", "wall_time_s")}
    # The band pass 2 was actually scored on, so a caller can put ANOTHER
    # section (a shape optimiser's seed and its answer) on the same map as
    # this table rather than on a second one of its own.
    #
    # ONLY IF IT LOADED. A payload this function had already REFUSED (ref is
    # None above, and the table you are reading was ranked on a live min-max)
    # used to be published under the same key regardless, and the shell then
    # handed it to the shape optimiser — which refused the whole run with "the
    # composite objective needs a FROZEN normalisation band and none could be
    # loaded". A band nothing was scored on is not this table's band.
    if payload and ref is not None:
        out["reference"] = payload
    elif payload:
        out["reference_error"] = (
            "the band measured over the library pass could not be loaded, so "
            "this table was ranked on a live min-max over its own population "
            "and there is no frozen band to hand on")
    return out


# ---------------------------------------------------------------------
# Scoring NAMED shapes on the screen's own six criteria
# ---------------------------------------------------------------------


def _score_reference_arg(reference):
    """``(ScoreReference | None, provenance dict)`` from whatever was passed.

    Accepts a :class:`airfoil_select.ScoreReference`, a stored payload dict
    (``build_screen_reference`` / a screen report's ``reference`` block), or
    None — which loads the SHIPPED library band. Never raises: an unusable
    reference is reported as a reason, because the caller's alternative is
    worse than no score (see :func:`score_sections`).
    """
    from .airfoil_select import (SCREEN_REFERENCE_PATH, ScoreReference,
                                 load_screen_reference)

    if isinstance(reference, ScoreReference):
        return reference, {"source": "caller", "sha": reference.sha,
                           "n_records": int(reference.n_records)}
    if isinstance(reference, (str, Path)):
        # A PATH to a band file. Flags have to stay JSON-serialisable to be
        # checkpointed, so a study that scores on a non-default band (one per
        # Reynolds number, session 52) can only carry it as a path. Before
        # this, a path fell through to the shipped band SILENTLY: the run
        # scored a Re-3e5 section on the Re-1e6 band, which puts it at the
        # bottom of that band by construction, and nothing said so. Loading it
        # here, and REFUSING an unreadable one, is the only version that
        # cannot substitute a different band for the one asked for.
        try:
            ref = load_screen_reference(Path(reference))
        except (ValueError, OSError) as exc:
            return None, {"source": "path", "path": str(reference),
                          "reason": f"unusable band file: {exc}"}
        return ref, {"source": "path", "path": str(reference),
                     "sha": ref.sha, "n_records": int(ref.n_records)}
    if isinstance(reference, dict):
        try:
            ref = ScoreReference(
                bounds={k: (float(v[0]), float(v[1]))
                        for k, v in reference["bounds"].items()},
                sha=str(reference.get("sha", "")),
                n_records=int(reference.get("n_records", 0)),
                preset=str(reference.get("preset", "")),
                # the criteria the measured population could not separate,
                # carried so the band loads instead of being refused whole
                # (a fin's zero-lift L/D — airfoil_select.build_screen_reference)
                unbanded={str(k): str(v) for k, v in
                          (reference.get("unbanded") or {}).items()})
        except (KeyError, TypeError, ValueError) as exc:
            return None, {"source": "caller", "reason": f"unusable band: {exc}"}
        return ref, {"source": "caller", "sha": ref.sha,
                     "n_records": int(ref.n_records)}
    try:
        ref = load_screen_reference()
    except ValueError as exc:
        return None, {"source": "shipped", "reason": str(exc)}
    return ref, {"source": "shipped", "sha": ref.sha,
                 "n_records": int(ref.n_records),
                 "path": str(SCREEN_REFERENCE_PATH)}


def score_exchange_rates(weights: dict | str | None = None,
                         reference=None) -> dict:
    """What each screening criterion is WORTH to the composite, per unit.

    ``dJ/dv_k = 100 w_k / (hi_k - lo_k)`` — see
    :func:`airfoil_select.exchange_rates`. A weight alone does not say what a
    search will trade, because the WIDTH of the band is the other half of every
    exchange: under the shipped band the GDP preset's largest weight (0.35 on
    cruise L/D) buys four counts of L/D for one hundredth of |Cm|, which is
    scored at 0.20. Pure — no XFOIL, no run — so a shell can state the rate
    before the search rather than explain it afterwards.

    Returns ``{}`` when no frozen band can be loaded, for the same reason
    :func:`score_sections` returns no score there: an exchange rate against a
    live min-max would be a different number every evaluation.
    """
    from .airfoil_select import exchange_rates

    ref, _info = _score_reference_arg(reference)
    if ref is None:
        return {}
    return exchange_rates(screen_weights(weights), ref)


def score_sections(
    sections,
    weights: dict | str | None = None,
    *,
    re: float = 1e6,
    mach: float = 0.0,
    cl_design: float = 0.5,
    tc_min: float = 0.10,
    cm_max: float = 0.08,
    reference=None,
    cache_only: bool = False,
    stall_sweep: bool = True,
    timeout_s: float | None = None,
    n_panel: int | None = None,
) -> dict:
    """Score NAMED coordinate loops on the screen's six criteria.

    The screen (:func:`screen_airfoils`) answers "which library section best
    fits these weights"; the shape optimiser then answers a DIFFERENT question
    (minimum cd at the design lift, or wing L/D), so its winner arrives with no
    statement about the five criteria the weights also asked for. This is that
    statement: the same metric extraction (``airfoil_select.polar_metrics``,
    cruise sweep + the wide stall sweep ``clmax``/``astall`` need) and the same
    composite (``score_candidates``) applied to shapes the caller names —
    typically a run's SEED and the section it optimised into.

    ``sections``: an iterable of ``{"name": str, "coords": (n, 2) array}``
    (extra keys are carried through untouched).

    THE BAND IS FROZEN OR THERE IS NO SCORE. The GDP composite is min-max
    normalised across the population it is handed, so scoring two sections
    against each other would hand out 100 and 0 by construction no matter how
    close they are — a number that looks like a verdict and is arithmetic.
    ``reference`` therefore supplies a fixed (lo, hi) per criterion: a
    ScoreReference, a screen report's ``reference`` payload, or None for the
    shipped p2/p98 library band. When none can be loaded every section comes
    back with ``composite = None`` and the reference block says why — a
    missing score, never a manufactured one.

    Gates are REPORTED, not applied: each section carries ``gates`` (t/c and
    |Cm| against ``tc_min`` / ``cm_max``) while still being scored, because the
    caller is comparing two designs and needs to see a gate fail as a fact
    about one of them rather than as a blank row.

    ``cache_only`` restricts every polar to the XFOIL disk cache (a section
    whose sweep is not cached comes back ``status = "no_polar"``): the honest
    setting for a view that must not spend a minute of XFOIL time on a
    comparison nobody asked for. The default runs the sweeps for real — one
    cruise sweep (usually already cached, since the optimiser flew it) plus,
    with ``stall_sweep``, the wide 0…20 deg sweep that ``clmax`` and
    ``astall`` need and that nothing else caches.
    """
    from . import airfoil
    from .airfoil_select import (MIN_CONVERGED, STALL_ALPHAS, polar_metrics,
                                 score_candidates)
    from .xfoil_run import run_xfoil_polar

    prob = airfoil.AirfoilProblem(re=float(re), mach=float(mach),
                                  cl_design=float(cl_design),
                                  tc_min=float(tc_min), cm_max=float(cm_max))
    if timeout_s is not None:
        prob.timeout_s = float(timeout_s)
    if n_panel is not None:
        prob.n_panel = int(n_panel)
    w = screen_weights(weights)
    ref, ref_info = _score_reference_arg(reference)

    def polar(coords, alphas):
        if cache_only:
            return _cached_dat_polar(prob, coords, alphas=alphas)
        return run_xfoil_polar(coords, prob.re, prob.mach, alphas,
                               timeout_s=prob.timeout_s,
                               cache_dir=prob.cache_dir, n_panel=prob.n_panel)

    records: list[dict] = []
    for i, sec in enumerate(sections):
        name = str(sec.get("name") or f"section {i + 1}")
        coords = np.asarray(sec.get("coords"), dtype=float)
        rec: dict = {"name": name, "label": sec.get("label", name)}
        if coords.ndim != 2 or coords.shape[0] < 5 or coords.shape[1] != 2:
            rec.update(status="bad_coords",
                       reason=f"{coords.shape} is not an (n, 2) loop")
            records.append(rec)
            continue
        pol = polar(coords, prob.alphas)
        if pol is None:
            rec.update(status="no_polar",
                       reason="no cached XFOIL sweep for this shape at this "
                              "operating point")
            records.append(rec)
            continue
        stall = None
        if stall_sweep and pol.n_converged >= MIN_CONVERGED:
            stall = polar(coords, STALL_ALPHAS)
        rec.update(polar_metrics(pol, coords, prob, pol_stall=stall))
        rec["stall_sweep"] = stall is not None
        records.append(rec)

    # Gates are a FACT about each section, not a filter: score_candidates is
    # asked for no gates at all (t/c floor 0, no |Cm| cap) so a design that
    # fails one is still scored and still comparable, and the pass/fail is
    # reported next to the number.
    scored = (score_candidates([r for r in records if r.get("status") == "ok"],
                               w, tc_min=0.0, cm_max=float("inf"),
                               reference=ref)
              if ref is not None else [])
    by_name = {r["name"]: r for r in scored}

    out_sections = []
    for rec in records:
        row = {
            "name": rec["name"], "label": rec.get("label", rec["name"]),
            "status": rec.get("status", "ok"), "reason": rec.get("reason", ""),
            "composite": None, "scores": {},
            "metrics": {k: _finite_or_none(rec.get(k))
                        for k in ("tc", "clmax", "astall", "ldmax", "ldcr",
                                  "cd_at", "cm_at", "alpha_at")},
            "clmax_censored": bool(rec.get("clmax_censored", True)),
            "stall_sweep": bool(rec.get("stall_sweep", False)),
            "n_converged": int(rec.get("n_converged", 0) or 0),
        }
        tc, cm = row["metrics"]["tc"], row["metrics"]["cm_at"]
        row["gates"] = {
            "tc": None if tc is None else bool(tc >= prob.tc_min),
            "cm": None if cm is None else bool(abs(cm) <= prob.cm_max),
        }
        got = by_name.get(rec["name"])
        if got is not None:
            row["composite"] = _finite_or_none(got.get("composite"))
            row["scores"] = {k: _finite_or_none(got.get(f"score_{k}"))
                             for k, _ in SCREEN_METRICS
                             if got.get(f"score_{k}") is not None}
        elif ref is None:
            row["reason"] = row["reason"] or (
                "no frozen normalisation band, so the composite is not "
                "defined for this comparison")
        out_sections.append(row)

    ranked = [s for s in out_sections if s["composite"] is not None]
    return {
        "conditions": {"re": float(re), "mach": float(mach),
                       "cl_design": float(cl_design),
                       "tc_min": float(tc_min), "cm_max": float(cm_max)},
        "weights": w.normalised(),
        "reference": ref_info,
        "sections": out_sections,
        "best": (max(ranked, key=lambda s: s["composite"])["name"]
                 if ranked else None),
    }


def score_optimised_section(report: dict, weights: dict | str | None = None,
                            *, reference=None, cache_only: bool = False,
                            stall_sweep: bool = True) -> dict | None:
    """SEED vs OPTIMISED on the screening criteria — the run's other half.

    ``report`` is an :func:`optimize_airfoil` report. The two shapes come from
    its own section block (``baseline`` = the seed the box was anchored on,
    ``design`` = what the search returned) and the operating point comes from
    ``conditions``, which in wing mode already carries the DERIVED Re and
    design Cl the run actually flew — so both sections are scored where they
    were designed, not at the form's typed numbers.

    Returns None when the report carries no section shape at all. Otherwise a
    :func:`score_sections` report with ``seed`` / ``optimised`` rows and their
    ``delta`` (optimised - seed, per criterion and on the composite). A run
    whose design never moved off its anchor has no separate baseline shape;
    ``seed`` is then None and the delta is empty, which is the truth rather
    than a zero.

    The optimiser did NOT search this composite (it maximises -cd, or wing L/D
    in wing mode — ``airfoil.AirfoilProblem`` / ``section_wing``), so a
    negative delta is a real finding about the trade, not a bug: the weights
    chose the seed, and the search was free to spend the other five criteria
    on the one it was given.
    """
    sec = report.get("section") or {}
    cond = report.get("conditions") or {}
    design = sec.get("design") or {}
    base = sec.get("baseline") or {}
    if not design.get("coords"):
        return None

    wanted = [{"name": "optimised", "label": "optimised",
               "coords": design["coords"]}]
    if base.get("coords"):
        wanted.insert(0, {"name": "seed",
                          "label": base.get("name") or "seed",
                          "coords": base["coords"]})

    out = score_sections(
        wanted, weights,
        re=float(cond.get("re", 1e6)), mach=float(cond.get("mach", 0.0)),
        cl_design=float(cond.get("cl_design", 0.5)),
        tc_min=float(cond.get("tc_min", 0.10)),
        cm_max=float(cond.get("cm_max", 0.08)),
        reference=reference, cache_only=cache_only, stall_sweep=stall_sweep)

    rows = {s["name"]: s for s in out["sections"]}
    seed, opt = rows.get("seed"), rows.get("optimised")
    out["seed"], out["optimised"] = seed, opt
    delta: dict = {"composite": None, "scores": {}, "metrics": {}}
    if seed and opt:
        if seed["composite"] is not None and opt["composite"] is not None:
            delta["composite"] = float(opt["composite"] - seed["composite"])
        for key in seed["scores"]:
            if opt["scores"].get(key) is not None:
                delta["scores"][key] = float(opt["scores"][key]
                                             - seed["scores"][key])
        for key, val in seed["metrics"].items():
            if val is not None and opt["metrics"].get(key) is not None:
                delta["metrics"][key] = float(opt["metrics"][key] - val)
    out["delta"] = delta
    return out


# =====================================================================
# The wing's own composite: measuring its band, and reading a design
# against it (wing_score.py)
# =====================================================================


def _needs_lateral(flags: dict | None) -> bool:
    """Does this run's weighting ask for a lateral deck?

    ONE answer, read in both places that build the family: the run itself
    (:func:`_with_wing_objective`) and the band the run is normalised
    against (:func:`wing_score_reference`). They must agree — a band
    measured without the deck carries no ``spiral`` row, and the composite
    then refuses the very weight that asked for it, which is the
    chicken-and-egg this function exists to break.

    Reads the WEIGHTS, never a separate switch, so the cost is opted into
    by asking for the criterion and by nothing else.

    ...OR BY ASKING FOR THE GATE. ``handling_level`` needs the same deck for
    the same reason — an aeroplane with no vertical surface in the SCORED
    lattice has ``Cn_beta`` exactly zero, so its Dutch roll and its spiral
    are absences rather than measurements. Both switches answer here so the
    run and the band it is normalised against cannot arm the deck
    differently; ``WingTailProblem.__post_init__`` states the same
    implication for a caller that builds the problem directly.
    """
    if (flags or {}).get(HANDLING_LEVEL_KEY) is not None:
        return True
    try:
        weights = wing_score_weights((flags or {}).get("wing_score_weights"))
    except (ValueError, KeyError, TypeError):
        return False
    return float(getattr(weights, "spiral", 0.0)) > 0.0


def wants_spiral(flags: dict | None) -> bool:
    """Does this run's weighting PRICE the lateral half of stability?

    The public half of :func:`_needs_lateral`, for a shell that has to say
    whether a searched dihedral has anything to buy: with no weight on
    ``spiral`` the objective pays for projected span alone, so the row goes
    to whichever bound costs least and the answer is as planar as a fixed
    one. Asked of the WEIGHTS, so a caller cannot get a different answer
    from the one the run itself will act on.
    """
    return _needs_lateral(flags)


def _arm_lateral(built, flags: dict | None):
    """Switch the lateral deck on where the family has one, and say so.

    ...INCLUDING A FAMILY THAT CARRIES ONE INSIDE ANOTHER PROBLEM. The
    wing+tail CST-section family is a section problem WRAPPING a
    ``WingTailProblem``, so ``hasattr(built.problem, "lateral")`` was False
    on it and the deck was never armed. It reports the wing+tail evaluate's
    own dict, so the criterion travels the moment the inner problem is
    switched on — and without this the searched-cant twins of that family
    would be a design row NOTHING PRICES, which is the exact failure the
    ``spiral`` criterion exists to prevent.
    """
    if not _needs_lateral(flags):
        return built
    for obj in (built.problem, getattr(built.problem, "wing_tail", None)):
        if obj is not None and hasattr(obj, "lateral"):
            obj.lateral = True
    return built


def _strip_wing_objective(flags: dict | None) -> dict:
    """``flags`` without the composite keys — the family's OWN objective.

    The band is measured on the plain breakdown, which is also what breaks
    the chicken-and-egg: building a composite problem needs a band, and
    measuring a band needs a problem.
    """
    return {k: v for k, v in (flags or {}).items()
            if k not in WING_OBJECTIVE_FLAG_KEYS}


def wing_score_reference(problem_name: str, *, flags: dict | None = None,
                         mission_kwargs: dict | None = None,
                         bounds_overrides: dict | None = None,
                         n: int | None = None, seed: int = 0,
                         progress: Callable | None = None,
                         pinned: dict | None = None,
                         screen: bool = False) -> dict:
    """Measure the frozen normalisation band over ONE design box.

    Sweeps the box ``n`` times (its centre first, then a scrambled Sobol
    sequence), keeps the feasible samples and returns the p5/p95 band per
    criterion as a JSON-safe payload — the thing that travels on a
    ``RunConfig``'s ``wing_score_reference`` flag and gets hashed into the
    run's own provenance.

    COSTS ``n`` EVALUATIONS of this family: microseconds each for a lifting
    line, seconds each for anything holding an XFOIL sweep. That is the whole
    price of the composite objective, it is paid once per box, and a caller
    that wants to show it a progress bar passes ``progress(i, n)``.

    ``pinned`` is the run's own :attr:`RunConfig.pinned`, and it belongs here
    for the same reason it belongs on the run: a criterion is normalised
    against the population the design box would produce, and a variable the
    user has FIXED does not vary in that population. Sweeping it anyway would
    score every design against a band containing planforms this session can
    never return — the composite's equivalent of quoting the middle of a row
    nobody searches.

    Raises ValueError when the box is too infeasible to measure (fewer than
    ``wing_score.MIN_REFERENCE_SAMPLES`` samples flew) — a band nothing flew
    in would score every design against noise.

    ``screen`` (default False, the published path) keeps drawing until ``n``
    samples have FLOWN instead of taking whatever flies out of ``n`` draws —
    see :func:`wing_score.sample_reference`. It turns a box the SEARCH solves
    but the band could not measure into one both can.
    """
    from . import wing_score as wsc
    spec = PROBLEM_SPECS[problem_name]
    built = spec.build(mission_kwargs or {}, _strip_wing_objective(flags),
                       bounds_overrides)
    # ...and the band must be measured on the SAME aeroplane the run scores.
    # The strip above removes the composite keys (that is what breaks the
    # chicken-and-egg), but the lateral deck is not an objective, it is a
    # MEASUREMENT the family either takes or does not — and a band with no
    # ``spiral`` row makes the composite refuse the weight that asked for it.
    _arm_lateral(built, flags)
    pin = _pin_of(built, pinned)
    ref = wsc.sample_reference(
        built.evaluate, built.bounds if pin is None else pin.bounds,
        problem=problem_name,
        n=int(n or wsc.REFERENCE_SAMPLES), seed=int(seed),
        labels=tuple(built.param_labels),
        metrics=_wing_metrics_fn(built, flags), progress=progress,
        screen=bool(screen))
    return ref.payload()


def _strip_foil_objective(flags: dict | None) -> dict:
    """``flags`` without the foiling composite keys — the family's OWN
    objective. Breaks the same chicken-and-egg ``_strip_wing_objective`` does
    (a composite problem needs a band; a band needs a problem), and it is what
    the TURN twin is built with, so the twin cannot recurse into a composite.
    """
    return {k: v for k, v in (flags or {}).items()
            if k not in FOIL_OBJECTIVE_FLAG_KEYS}


def foil_score_reference(problem_name: str, *, flags: dict | None = None,
                         mission_kwargs: dict | None = None,
                         bounds_overrides: dict | None = None,
                         n: int | None = None, seed: int = 0,
                         progress: Callable | None = None,
                         pinned: dict | None = None) -> dict:
    """Measure the frozen band for the FOILING composite over one design box.

    Sweeps the box ``n`` times and flies each sample at all four points, so it
    costs about ``4n`` evaluations — milliseconds each for these families,
    which is what makes a per-box band affordable at all. All four criteria
    are measured whatever the weights say, so a user can re-weight afterwards
    without paying for the sweep again.

    ``pinned`` belongs here for :func:`wing_score_reference`'s reason: a
    criterion is normalised against the population the box would produce, and
    a row the user has fixed does not vary in it. For a placed craft that is
    most of the vector, which is exactly the point — the band is the
    population of shapes THIS craft can still be given.

    Raises ValueError when too little of the box flew anywhere, and reports
    per-criterion how many samples reached each point: a light-wind band
    measured on 3 of 32 samples is not a scale, and the caller is told so by
    name rather than handed a number.
    """
    from . import foil_score as fsc

    spec = PROBLEM_SPECS[problem_name]
    plain = _strip_foil_objective(flags)
    built = spec.build(mission_kwargs or {}, plain, bounds_overrides)
    pin = _pin_of(built, pinned)
    points = _foil_points(flags)
    metrics = _foil_metrics_fn(built, spec, mission_kwargs, plain,
                               bounds_overrides,
                               fsc.FoilScoreWeights(ld_cruise=1.0,
                                                    ld_light=1.0, cav_top=1.0,
                                                    margin_turn=1.0),
                               points, spec.build)
    ref = fsc.sample_reference(
        metrics, built.bounds if pin is None else pin.bounds,
        problem=problem_name, n=int(n or fsc.REFERENCE_SAMPLES),
        seed=int(seed), labels=tuple(built.param_labels), progress=progress,
        points=points)
    return ref.payload()


_REFUSAL_NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def _refusal_kind(reason: str) -> str:
    """A refusal reason with its NUMBERS blanked, so the same gate counts once.

    "aspect ratio 115 outside the band (3.0, 40.0)" and "aspect ratio 55.6
    outside the band (3.0, 40.0)" are one finding, not two, and a probe that
    ranked them separately would bury the gate that fired 59 times under 59
    rows of one.
    """
    if not reason:
        return "refused (no reason recorded)"
    return _REFUSAL_NUMBER.sub("#", str(reason))


#: below this many surviving draws the per-row spans are noise, not a finding
_ROW_MIN_KEPT = 4

#: report a row only when this much of its band produced nothing at all
_ROW_DEAD_FRAC = 0.30

#: ...and only when the survivors' span is narrower than chance would give.
#: Family-wise, over every row tested (Bonferroni): one row in twenty clearing
#: a 5 % bar is exactly what a 20-row design box produces by accident, and a
#: card that names a row is telling the user which control to move.
_ROW_ALPHA = 0.05


def _span_p_value(frac: float, k: int) -> float:
    """P(k uniform draws span this little of a row), i.e. Beta(k-1, 2) CDF.

    THE BIAS THIS EXISTS FOR. The observed span of k draws understates the
    interval they came from — E[span] = (k-1)/(k+1) of it — so with 5
    survivors a row that is entirely healthy shows a third of itself "dead"
    on average, and a fixed threshold would report every row of every box
    that is mostly refused. The closed form is exact for a uniform draw:
    ``F(r) = r**(k-1) * (k - (k-1) r)``.
    """
    if k < 2:
        return 1.0
    r = min(1.0, max(0.0, float(frac)))
    return float(r ** (k - 1) * (k - (k - 1) * r))


def _refusal_by_row(draws, box: np.ndarray, labels: list) -> list:
    """WHERE IN EACH ROW the draws that survived actually live, ranked by how
    much of the row produced nothing.

    Not a refusal rate per half: on a box that is 92 % refused the base rate
    swamps every marginal (measured — the widest half-to-half gap on the
    reported box is 0.16, and the row it points at is not the binding one).
    The informative direction is the other one: take the draws that PASSED and
    ask how much of each row they never came from. "Every draw that survived
    had b_m below 14.2 m, so the top 64 % of that row produced nothing" is a
    sentence about a control the user can move.

    It is a marginal and it says so: the gates are joint conditions in several
    rows at once (the aspect ratio is b^2/S), so a dead span means "nothing
    here survived WITH THE REST OF THIS BOX", not "this row is wrong". Where
    the joint statement is available in closed form it is the better one —
    :func:`size_box_conflicts`.
    """
    if not draws:
        return []
    X = np.asarray([d[0] for d in draws], dtype=float)
    kept = ~np.asarray([d[1] for d in draws], dtype=bool)
    n_kept = int(kept.sum())
    if n_kept < _ROW_MIN_KEPT or n_kept == len(draws):
        return []
    tested = [i for i, _l in enumerate(labels)
              if i < X.shape[1] and float(box[i][1]) > float(box[i][0])]
    alpha = _ROW_ALPHA / max(1, len(tested))
    out = []
    for i in tested:
        label = labels[i]
        lo, hi = float(box[i][0]), float(box[i][1])
        col = X[kept, i]
        k_lo, k_hi = float(col.min()), float(col.max())
        dead = 1.0 - (k_hi - k_lo) / (hi - lo)
        p = _span_p_value((k_hi - k_lo) / (hi - lo), n_kept)
        if dead < _ROW_DEAD_FRAC or p > alpha:
            continue
        out.append({
            "label": label, "bounds": [lo, hi], "kept": [k_lo, k_hi],
            "dead_frac": float(dead), "n_kept": n_kept,
            "n": int(len(draws)), "p": p,
            # which END of the row is dead, where it is one end rather than
            # both: that is the bound to move, and it is what a card says
            "end": ("top" if (hi - k_hi) > (k_lo - lo) else "bottom"),
        })
    out.sort(key=lambda d: d["p"])
    return out


def recommend_box(cfg: "RunConfig", *, n: int = 128, seed: int = 0,
                  pinned: dict | None = None, verify: bool = False,
                  escalate: float = 3.0, always: bool = True,
                  points: dict | None = None) -> dict:
    """Where this mission's best admissible designs live, for THIS config.

    The api-level twin of :func:`box_refusal_probe`, and it exists because the
    shell was doing this by hand and getting three things wrong that
    ``box_refusal_probe`` had already got right:

    * ``sanitise_flags`` — a flag the family does not declare is dropped
      rather than raising;
    * the WING OBJECTIVE — a composite run with no frozen normalisation band
      cannot be built at all (``_wing_objective``'s refusal), so the shell's
      bare ``spec.build`` raised, the caller swallowed it, and the button
      became a silent no-op. The objective is stripped ONLY when it cannot be
      built: where a band HAS been measured the composite builds, and
      ``wing_score`` puts J in the row's ``score``, so the recommendation is
      ranked on the same number the run maximises. ``stripped`` says which
      happened, so the card can quote the right one;
    * the PINS — ``_pinned_built`` rewrites ``param_labels`` to the free rows,
      so a fixed row can no longer be measured as free and proposed as a band
      the run would never search.

    ``_pinned_built`` is used rather than ``_search_problem`` deliberately:
    the latter refuses ``blocks`` (which the shell selects automatically on
    492 families) and would make the recommendation depend on the OPTIMISER,
    which decides how a box is searched and not which designs fly.

    ``verify`` defaults to False here, against the module default: the
    verification pass costs another ``n`` evaluations to re-measure a figure
    every caller in this repo recomputes for its own subset anyway.

    ``points`` (optional) are designs a RUN of this same search already
    found feasible (:func:`feasible_points_of`). They are used only when the
    sample came back with nothing — and then they are the best evidence there
    is: on a family whose feasible set is a per-cent of its box, 96 uniform
    draws find nothing while the search that has already been paid for found
    50 designs. ``basis`` is then ``"run"``.

    ``always`` (default True) adds the LAST rung of the ladder
    (:func:`recommend.recommend_box` owns the first three): where not one draw
    reached a solver, the two cheap gates are asked in closed form
    (:func:`size_box_conflicts`) and their own suggestions become the
    recommendation. ``basis`` then reads:

    ``"admissible"``  bands around designs that fly this mission
    ``"run"``         the sample found nothing; bands around the FEASIBLE
                      designs a previous run of this same search evaluated
    ``"solved"``      nothing flew; bands around designs whose solver RAN
    ``"gates"``       nothing ran; a band a refusing gate can accept
    ``"mission"``     no band can help — the box is ruled out by a number
                      stated OUTSIDE it (``conflicts`` says which, with the
                      value that would clear it)
    None              nothing measurable and no conflict found

    Never a silent empty answer: every one of those is something a card can
    say, and "no recommendation" was the least useful of the five.
    """
    from . import recommend as _recommend

    search, stripped = _recommendable_problem(cfg, pinned)
    got = _recommend.recommend_box(search, n=n, seed=seed, verify=verify,
                                   escalate=escalate)
    got["objective_stripped"] = bool(stripped)
    if got.get("basis") == "admissible" or not always:
        return got
    # A SEARCH BEATS A SAMPLE. Before falling to the weaker rungs, ask what
    # a run of this same configuration already found: those designs are
    # feasible by evaluation, not by sampling luck.
    rows = _rows_from_points(points, search, pin_of_cfg(cfg, pinned))
    if rows:
        got["rows"] = rows
        got["basis"] = "run"
        got["n_points"] = int((points or {}).get("n") or len(rows))
        return got
    if got.get("basis") is not None:
        return got
    # NOTHING REACHED A SOLVER. Sampling has nothing left to say — the answer
    # is the closed-form one, which is exact and costs no evaluations.
    try:
        conflicts = size_box_conflicts(cfg, pinned=pinned)
    except Exception:              # noqa: BLE001 — a recommendation, never fatal
        return got
    got["conflicts"] = conflicts
    labels = list(got.get("labels") or [])
    rows = {}
    for c in conflicts:
        row, band = c.get("row"), c.get("suggest")
        if row and band and row in labels:
            rows[str(row)] = (float(band[0]), float(band[1]))
    if rows:
        got["rows"] = rows
        got["basis"] = "gates"
    elif conflicts:
        # the box cannot be moved into an answer, because what rules it out is
        # not a row of it. The finding IS the recommendation.
        got["basis"] = "mission"
    return got


def pin_of_cfg(cfg: "RunConfig", pinned: dict | None = None):
    """The :class:`_Pin` a recommendation is measured under, or None."""
    spec = PROBLEM_SPECS[cfg.problem_name]
    flags, _dropped = sanitise_flags(cfg.problem_name, cfg.flags)
    try:
        built = spec.build(cfg.mission_kwargs or {},
                           _strip_wing_objective(flags), cfg.bounds_overrides)
    except ValueError:
        return None
    return _pin_of(built, cfg.pinned if pinned is None else pinned)


def _rows_from_points(points: dict | None, search: "_BuiltProblem",
                      pin: "_Pin | None") -> dict:
    """Bands around designs a RUN found — clipped into the box, or ``{}``.

    Points OUTSIDE the box are dropped rather than clipped: they belong to a
    search over a different box, and a band drawn through them would be about
    that one. Everything else is :func:`recommend.box_around`, so the band is
    padded, floored and clipped exactly as a measured one is.
    """
    from . import recommend as _recommend

    rows = list((points or {}).get("x") or [])
    if not rows:
        return {}
    X = np.atleast_2d(np.asarray(rows, dtype=float))
    if pin is not None and X.shape[1] == len(pin.labels):
        X = np.atleast_2d(pin.reduce(X))
    box = np.asarray(search.bounds, dtype=float)
    if X.shape[1] != box.shape[0]:
        return {}
    inside = X[np.all((X >= box[:, 0] - 1e-9) & (X <= box[:, 1] + 1e-9),
                      axis=1)]
    if not len(inside):
        return {}
    rec = _recommend.box_around(inside, box)
    return {lab: (float(rec[i][0]), float(rec[i][1]))
            for i, lab in enumerate(search.param_labels)}


def _recommendable_problem(cfg: "RunConfig", pinned: dict | None = None):
    """(the problem a recommendation is measured on, was the objective
    stripped).

    The wing objective's refusal is raised by the BUILDER, not by
    ``check_wing_objective`` — the band is checked where the score is
    assembled (``_with_wing_objective``) — so the only honest test is to try
    the build and fall back. The original error is re-raised if the fallback
    fails too, so a genuine build problem is never reported as an objective
    one.
    """
    spec = PROBLEM_SPECS[cfg.problem_name]
    flags, _dropped = sanitise_flags(cfg.problem_name, cfg.flags)
    stripped = False
    try:
        built = spec.build(cfg.mission_kwargs or {}, flags,
                           cfg.bounds_overrides)
    except ValueError:
        bare = _strip_wing_objective(flags)
        if bare == flags:
            raise
        built = spec.build(cfg.mission_kwargs or {}, bare,
                           cfg.bounds_overrides)
        stripped = True
    pin = _pin_of(built, cfg.pinned if pinned is None else pinned)
    return (built if pin is None else _pinned_built(built, pin)), stripped


def recommendation_probe(cfg: "RunConfig", rows: dict, *, n: int = 128,
                         seed: int = 0) -> tuple:
    """``(frac admissible, best score)`` over this box with ``rows`` narrowed.

    What TAKING a recommendation would be worth, measured on the same problem
    the recommendation was measured on — including the same objective
    fallback and the same pins, which is why it lives here beside
    :func:`recommend_box` rather than in the shell.
    """
    from . import recommend as _recommend

    search, _stripped = _recommendable_problem(cfg)
    labs = list(search.param_labels)
    cand = np.asarray(search.bounds, dtype=float).copy()
    for lab, band in (rows or {}).items():
        if lab in labs:
            cand[labs.index(lab)] = (float(band[0]), float(band[1]))
    return _recommend.probe_box(search, cand, int(n), int(seed))


def box_refusal_probe(cfg: "RunConfig", *, n: int = 64, seed: int = 0,
                      pinned: dict | None = None) -> dict:
    """How much of THIS design box is refused before its solver runs, and why.

    Draws ``n`` scrambled Sobol points over the box the run would search (the
    pinned rows collapsed, as the search sees them), evaluates each, and splits
    the outcome three ways: ADMISSIBLE, refused (the -100 failure sentinel — a
    declared gate, or a solver that could not run), and solved-but-inadmissible.
    Returns the counts, the fraction refused, the refusal reasons ranked by how
    often they fired, and the same fractions.

    ADMISSIBLE means what the SEARCH means by it: the design solved AND every
    constraint margin is >= 0. Counting the breakdown's own ``feasible`` flag
    instead over-reports, because a design can solve and still miss a limit —
    measured on the reported box, 5 of 64 draws solved and only 2 of those were
    admissible, so a probe quoting 5 would tell a user their box is 2.5x
    healthier than the search finds it.

    WHY IT IS WORTH A FUNCTION. The failure this exists for looks like a
    solver problem and is a BOX problem: on the run that motivated
    :mod:`optimize.feasible`, 92.6 % of a published design box was refused by
    two cheap gates — an aspect-ratio band the box's span x area rows overshoot
    by 5x, and the mission's own wing-loading ceiling — so a 10-point Sobol
    seed was blind more often than not. No number on any card said that, and
    the sentence a user needs is not "no solution": it is "your span and area
    rows disagree with the aspect-ratio band over 93 % of the box".

    CHEAP, and for the same reason the screen is: a refusal returns before its
    solver, measured 134x faster than a design that flies, so a 64-draw probe
    on that box costs about a tenth of a second. It is NOT free on a family
    whose refusals hold an XFOIL sweep — pass a smaller ``n`` there.
    """
    from collections import Counter

    spec = PROBLEM_SPECS[cfg.problem_name]
    flags, _dropped = sanitise_flags(cfg.problem_name, cfg.flags)
    built = spec.build(cfg.mission_kwargs or {}, _strip_wing_objective(flags),
                       cfg.bounds_overrides)
    pin = _pin_of(built, pinned if pinned is not None
                  else getattr(cfg, "pinned", None))
    box = np.asarray(built.bounds if pin is None else pin.bounds, dtype=float)
    from .optimize.feasible import is_refusal, sobol_pool

    n = max(1, int(n))
    reasons: Counter = Counter()
    n_refused = n_feasible = n_inadmissible = 0
    draws: list = []
    for x in sobol_pool(box, n, int(seed)):
        xa = np.asarray(x, dtype=float)
        raw = built.evaluate(xa)
        f = raw.get("score", raw.get("LoD", raw.get("f")))
        reason = str(raw.get("reason") or "")
        if bool(raw.get("feasible")):
            # ...and then the MARGINS, from the same call the optimiser makes:
            # the breakdown's flag is the solver's verdict, not the search's.
            #
            # ONLY WHERE THERE ARE MARGINS. An unconstrained family's callable
            # returns a bare float, so unpacking it before asking
            # ``is_constrained`` raised ``TypeError: cannot unpack
            # non-iterable float object`` — on the DEFAULT V3 configuration,
            # the moment one draw came back feasible. That took down "Measure
            # the design box" and every no-solution diagnostic that reads this
            # probe. :func:`recommend._admissible` is the same test written
            # the right way round; this now matches it.
            admissible = True
            if built.is_constrained:
                _fx, gx = built.callable(xa)
                admissible = bool(np.all(
                    np.atleast_1d(np.asarray(gx, dtype=float)) >= 0.0))
            if admissible:
                n_feasible += 1
            else:
                n_inadmissible += 1
            draws.append((xa, False))
            continue
        # a refusal is the sentinel, whatever the family calls its objective;
        # where no scalar is exposed on the breakdown the REASON is the signal,
        # which is the same contract read from the other end
        refused = (is_refusal(f) if isinstance(f, (int, float))
                   else bool(reason))
        if refused:
            n_refused += 1
            reasons[_refusal_kind(reason)] += 1
        draws.append((xa, bool(refused)))
    return {
        # WHICH ROW the refusals live at. A gate name tells a user what the
        # physics objected to; this tells them which control to move, which is
        # the question they actually have in front of a design box.
        "rows": _refusal_by_row(draws, box, list(built.param_labels)),
        "n": n,
        "n_feasible": n_feasible,
        "n_refused": n_refused,
        "n_infeasible": n - n_feasible - n_refused,
        # of the designs that SOLVED, how many then missed a limit. Split out
        # because it takes the opposite fix from a refusal: a wider box does
        # not help a design that flies and is inadmissible.
        "n_solved_inadmissible": n_inadmissible,
        "frac_feasible": n_feasible / n,
        "frac_refused": n_refused / n,
        "reasons": [{"reason": r, "n": c} for r, c in reasons.most_common()],
        "labels": list(built.param_labels),
        "bounds": [[float(lo), float(hi)] for lo, hi in box],
    }


def _simpson_area(lo: float, hi: float, f) -> float:
    """Exact integral of a piecewise-QUADRATIC ``f`` over one smooth piece.

    Every window this module integrates is a difference of terms that are
    either constant in the span or a multiple of ``b**2`` (the aspect-ratio
    band's two edges are ``b**2 / AR``), so Simpson's rule on a single panel
    is exact rather than approximate. Kept as a named helper so the caller
    can subdivide at the kinks and still claim an exact answer.
    """
    if not (hi > lo):
        return 0.0
    mid = 0.5 * (lo + hi)
    return (hi - lo) / 6.0 * (f(lo) + 4.0 * f(mid) + f(hi))


def size_box_conflicts(cfg: "RunConfig", *, pinned: dict | None = None
                       ) -> list[dict]:
    """Where THIS size box and THIS mission cannot both be satisfied — in
    closed form, before a single design is evaluated.

    WHY IT IS NOT :func:`box_refusal_probe`. The probe SAMPLES: it says "62 of
    64 draws were refused, mostly by the wing-loading gate", which is a
    measurement of a box and needs the box to contain something for the
    remaining draws to find. This function ANSWERS THE PRIOR QUESTION —
    whether anything in the box can pass the two cheap gates at all — and it
    answers it exactly, because both gates are inequalities in the same two
    numbers the box states:

    * ``sizing.check_wing_loading``: ``W_total / S <= wing_loading_max_Pa``,
      and ``W_total >= W_fixed`` (the weight loop only ADDS the wing), so
      ``S >= W_fixed / cap`` is NECESSARY for any design in the box;
    * ``sizing.check_ar``: ``AR_LIMITS[0] <= b*b/S <= AR_LIMITS[1]``.

    A box whose area row tops out below ``W_fixed / cap`` therefore cannot
    contain a design that flies, and no search, seed or screen can find one:
    every draw is refused before its solver. That is the reported failure
    this exists for — a mission at 7000 N against an area row of 8-22 m^2 and
    a 75.2 Pa ceiling off its own constraint diagram, i.e. a wing at least
    4.2x too small at the very best point of the box, run three times for
    12-21 s each and reported to the user as "no solution was found".

    Returns a list of findings, worst first. Each is a dict:

    ``kind``      which gate(s) — ``"wing loading"``, ``"aspect ratio"``, or
                  ``"wing loading x aspect ratio"`` for the joint window;
    ``empty``     True when the box PROVABLY contains nothing that can pass;
    ``text``      the sentence for a card, with its numbers;
    ``row``       the design-box row to move (``None`` where the conflict is
                  with a value the user typed rather than a row);
    ``suggest``   a band for that row that admits something, or ``None``;
    ``frac``      the exact share of the box's own (span, area) rectangle
                  that passes BOTH gates, where both are rows.

    An empty list means the two cheap gates do not rule this box out. It does
    NOT mean the box contains a design that flies: this is a necessary
    condition checked exactly, not a feasibility proof — the solvers, the
    chord law and every real constraint are downstream of it. Where the list
    is empty and the run still returns nothing, ``box_refusal_probe`` and
    ``gui.diagnose`` are the next questions, in that order.
    """
    from math import sqrt

    from . import sizing
    from .mission import weight_for

    spec = PROBLEM_SPECS[cfg.problem_name]
    flags, _dropped = sanitise_flags(cfg.problem_name, cfg.flags)
    built = spec.build(cfg.mission_kwargs or {}, _strip_wing_objective(flags),
                       cfg.bounds_overrides)
    pin = _pin_of(built, pinned if pinned is not None
                  else getattr(cfg, "pinned", None))
    box = np.asarray(built.bounds if pin is None else pin.bounds, dtype=float)
    labels = list(built.param_labels)
    prob = built.problem
    cap = getattr(prob, "wing_loading_max_Pa", None)

    def row(name):
        if name not in labels:
            return None
        lo, hi = box[labels.index(name)]
        return float(lo), float(hi)

    def _weight():
        """The payload weight the sized families price against [N], or None.

        The same expression ``sizing`` closes its weight loop around
        (``wingtail``/``aircraft``: the problem's own ``W_fixed_N`` where it
        has one, else the weight its published trim point implies), so this
        check and the gate cannot disagree about what the aircraft weighs.
        """
        w = getattr(prob, "W_fixed_N", None)
        if w is not None:
            return float(w)
        try:
            return float(weight_for(prob.CL_target, prob.V, prob.S))
        except (AttributeError, TypeError, ValueError):
            return None

    out: list[dict] = []
    ar_lo, ar_hi = (float(v) for v in sizing.AR_LIMITS)
    b_row, s_row, ws_row = row("b_m"), row("S_m2"), row(sizing.WS_LABEL)
    W = _weight()

    # ---- the loading the box can reach, against the mission's own ceiling
    if cap is not None and float(cap) > 0.0:
        cap = float(cap)
        if ws_row is not None and ws_row[0] > cap:
            out.append({
                "kind": "wing loading", "empty": True, "row": sizing.WS_LABEL,
                "suggest": [min(ws_row[0], cap), cap],
                "frac": 0.0,
                "text": (f"the searched wing loading starts at "
                         f"{ws_row[0]:.4g} Pa and this mission allows at most "
                         f"{cap:.4g} Pa, so every draw is refused before its "
                         f"solver. Lower the row's floor to {cap:.4g} Pa or "
                         f"raise the mission's stall / landing requirement."),
            })
        stated = getattr(prob, "wing_loading_Pa", None)
        if ws_row is None and stated is not None and float(stated) > cap:
            out.append({
                "kind": "wing loading", "empty": True, "row": None,
                "suggest": None, "frac": 0.0,
                "text": (f"the wing loading you stated, {float(stated):.4g} "
                         f"Pa, is above the {cap:.4g} Pa this mission's own "
                         f"constraint diagram allows: every design is refused "
                         f"before its solver."),
            })
        if W is not None and W > 0.0:
            s_need = W / cap                      # NECESSARY: W_total >= W
            if s_row is not None and s_row[1] < s_need:
                # the band OFFERED starts at the necessary minimum and ends
                # where the span row's own aspect-ratio band does, so it is
                # a band a design can sit INSIDE rather than one whose single
                # top endpoint is the only admissible point — and the wing's
                # own weight, which this bound cannot see, has room in it.
                top = 2.0 * s_need
                if b_row is not None:
                    top = max(top, b_row[1] * b_row[1] / ar_lo)
                out.append({
                    "kind": "wing loading", "empty": True, "row": "S_m2",
                    "suggest": [s_need, top], "frac": 0.0,
                    "text": (f"at {W:.4g} N this mission allows at most "
                             f"{cap:.4g} Pa, so the wing must be at least "
                             f"{s_need:.4g} m^2 — and the area row stops at "
                             f"{s_row[1]:.4g} m^2, {s_need / s_row[1]:.3g}x "
                             f"too small at its own best point. Every design "
                             f"in this box is refused before its solver runs."),
                })
            elif s_row is None and ws_row is None:
                S_fixed = getattr(prob, "S", None)
                if S_fixed and float(S_fixed) > 0.0 and W / float(S_fixed) > cap:
                    out.append({
                        "kind": "wing loading", "empty": True, "row": None,
                        "suggest": None, "frac": 0.0,
                        "text": (f"the fixed wing area {float(S_fixed):.4g} "
                                 f"m^2 flies at {W / float(S_fixed):.4g} Pa "
                                 f"and this mission allows {cap:.4g} Pa: "
                                 f"every design is refused before its solver."),
                    })

    # ---- and the two size rows against each other, and against that ceiling
    if b_row is not None and s_row is not None:
        b_lo, b_hi = b_row
        s_lo, s_hi = s_row
        s_need = (W / cap if (cap and W and float(cap) > 0.0) else 0.0)
        ar_box = (b_lo * b_lo / s_hi, b_hi * b_hi / s_lo)
        rect = (b_hi - b_lo) * (s_hi - s_lo)

        def _share(floor: float) -> float:
            """Share of the (b, S) rectangle with the aspect ratio in band and
            ``S >= floor``. For each span the admissible areas are ONE
            interval whose width is piecewise quadratic in b, with kinks only
            where an edge crosses a row's end — so Simpson between the kinks
            is exact, not a sample.
            """
            def width(b):
                return max(0.0, min(s_hi, b * b / ar_lo)
                           - max(s_lo, floor, b * b / ar_hi))

            kinks = {b_lo, b_hi}
            for edge, target in ((ar_lo, s_lo), (ar_lo, s_hi), (ar_hi, s_lo),
                                 (ar_hi, s_hi), (ar_hi, floor)):
                if target > 0.0:
                    b_k = sqrt(edge * target)
                    if b_lo < b_k < b_hi:
                        kinks.add(b_k)
            knots = sorted(kinks)
            return sum(_simpson_area(a, b, width)
                       for a, b in zip(knots, knots[1:])) / rect

        def _reachable(floor: float) -> bool:
            """Is any (b, S) in the box, with ``S >= floor``, in the band?

            ``b*b/S`` is continuous on a rectangle, so its range there is
            exactly [b_lo^2/S_hi, b_hi^2/S_lo] and the question is an interval
            overlap — which is also the ONLY form that survives a PINNED row,
            where the rectangle has no area to take a share of.
            """
            s_bot = max(s_lo, floor)
            if s_bot > s_hi:
                return False
            return (b_hi * b_hi / s_bot >= ar_lo
                    and b_lo * b_lo / s_hi <= ar_hi)

        # the aspect-ratio band is EXACT here — it is a function of (b, S) and
        # nothing else. The loading is a NECESSARY condition only (the weight
        # loop adds the wing's own weight, so the loading a design ends up
        # flying is >= W/S), which is why the two are reported apart: one is
        # the share of the box that survives, the other an upper bound on it.
        # A share needs an area to be a share OF; a pinned row has none, and
        # there the existence test above is the whole answer.
        def _ar_fix(floor: float):
            """``(row, band, why)`` — the ONE row whose band makes the aspect
            ratio reachable with ``S >= floor``, and a band that provably
            does.

            The suggestion has to clear the gate it cites, and the area row
            cannot always do it. An area band works iff some ``S >= floor``
            has ``b*b/S`` in band for some span the row already allows, i.e.
            iff ``b_hi^2 / ar_lo >= floor`` — and where it does, the widest
            such band is ``[max(floor, b_lo^2/ar_hi), b_hi^2/ar_lo]``, which
            satisfies both ends by construction.

            Where it does NOT, no area band exists at all: the span row tops
            out too low to fly a wing that big at any aspect ratio these
            solvers are valid over, and recommending an area is the loop the
            user reported — the loading gate pushes the row up, this gate
            pushes it back down, and neither press empties the box less. The
            row to move is the SPAN, to ``sqrt(ar_lo * floor)`` at the very
            least; the band offered carries the same 2x headroom the loading
            finding's ceiling does, because ``floor`` is blind to the wing's
            own weight and the sizing loop adds it.
            """
            s_top = b_hi * b_hi / ar_lo
            if s_top >= floor:
                return ("S_m2", [max(floor, b_lo * b_lo / ar_hi), s_top],
                        "area")
            if "b_m" not in labels:
                return (None, None, "span")
            return ("b_m", [b_lo, max(b_hi, sqrt(ar_lo * 2.0 * floor))],
                    "span")

        frac_ar = _share(0.0) if rect > 0.0 else float(_reachable(0.0))
        if not _reachable(0.0):
            # ...at the LOADING FLOOR, not at zero. A band that makes the
            # aspect ratio reachable and drops back under ``s_need`` empties
            # the box by the other gate, and the two suggestions then undo
            # each other press after press.
            fix_row, fix_band, which = _ar_fix(s_need)
            out.append({
                "kind": "aspect ratio", "empty": True, "row": fix_row,
                "suggest": fix_band,
                "frac": 0.0,
                "text": (f"no (span, area) pair in this box has an aspect "
                         f"ratio these solvers are valid over: the rows as "
                         f"stated span {ar_box[0]:.3g}-{ar_box[1]:.3g} "
                         f"against the band {ar_lo:g}-{ar_hi:g}, so every "
                         f"design is refused before its solver."
                         + (f" No area band can fix that on its own — a span "
                            f"of at most {b_hi:.4g} m cannot reach aspect "
                            f"ratio {ar_lo:g} over the "
                            f"{s_need:.4g} m^2 this mission needs — so it is "
                            f"the SPAN row that has to move."
                            if which == "span" and s_need > 0.0 else "")),
            })
        elif s_need > 0.0 and not _reachable(s_need):
            fix_row, fix_band, which = _ar_fix(s_need)
            out.append({
                "kind": "wing loading x aspect ratio", "empty": True,
                "row": fix_row, "suggest": fix_band,
                "frac": 0.0,
                "text": (f"every (span, area) pair whose aspect ratio is in "
                         f"band sits below the {s_need:.4g} m^2 of wing this "
                         f"mission's {cap:.4g} Pa ceiling needs at "
                         f"{W:.4g} N, so the two rows together admit nothing: "
                         f"each design is refused before its solver."
                         + (f" No area band can fix that on its own: at "
                            f"aspect ratio {ar_lo:g} a {s_need:.4g} m^2 wing "
                            f"needs at least "
                            f"{sqrt(ar_lo * s_need):.4g} m of span and this "
                            f"row stops at {b_hi:.4g} m. Move the SPAN row "
                            f"and the area one follows."
                            if which == "span" else "")),
            })
        elif frac_ar < 0.5:              # over half the rectangle is refused
            out.append({
                "kind": "aspect ratio", "empty": False, "row": "S_m2",
                "suggest": None, "frac": float(frac_ar),
                "text": (f"{100.0 * (1.0 - frac_ar):.1f} % of this box's "
                         f"span x area rectangle is outside the aspect-ratio "
                         f"band {ar_lo:g}-{ar_hi:g} these solvers are valid "
                         f"over — the rows as stated span "
                         f"{ar_box[0]:.3g}-{ar_box[1]:.3g}"
                         + (f", and the mission's own {cap:.4g} Pa ceiling "
                            f"cuts the rest further" if s_need > 0.0 else "")
                         + ". A search over this box spends most of its "
                         "budget on draws refused before their solver."),
            })
    out.sort(key=lambda d: (not d["empty"], d.get("frac") or 0.0))
    return out


def _wing_reference_arg(reference, cfg: "RunConfig | None" = None):
    """``(WingScoreReference | None, provenance)`` from an explicit argument,
    else from the run's own flags. Never raises: a display path that cannot
    resolve a band still reports the metrics, with the reason the composite
    is missing.
    """
    from . import wing_score as wsc
    payload = reference
    where = "argument"
    if payload is None and cfg is not None:
        payload = (cfg.flags or {}).get("wing_score_reference")
        where = "the run's own flags"
    if isinstance(payload, wsc.WingScoreReference):
        return payload, {"sha": payload.sha, "n_samples": payload.n_samples,
                         "n_feasible": payload.n_feasible, "source": where}
    if not payload:
        return None, {"reason": "no frozen band: the composite is not "
                                "defined without one"}
    try:
        ref = wsc.reference_from_payload(payload)
    except ValueError as exc:
        return None, {"reason": str(exc)}
    return ref, {"sha": ref.sha, "n_samples": ref.n_samples,
                 "n_feasible": ref.n_feasible, "source": where}


def score_design(cfg: "RunConfig", x, *, weights=None, reference=None,
                 built=None) -> dict:
    """The six wing criteria of ONE design, scored against the frozen band.

    Costs one physics evaluation (``design_report``'s cost — an XFOIL sweep
    for the section families). Returns ``metrics`` (the raw criteria, with
    their units), ``scores`` (0-100 sub-scores where the band covers them),
    ``composite`` (J, or None with a ``reason``) and ``feasible``.

    Works on ANY run, composite or not: the criteria are read off the
    breakdown the family already returns, so an ordinary L/D run can be read
    against the same six numbers it was not searched on — which is how a user
    finds out what the search spent.
    """
    from . import wing_score as wsc
    spec = PROBLEM_SPECS[cfg.problem_name]
    _flags, _dropped = sanitise_flags(cfg.problem_name, cfg.flags)
    if built is None:
        built = spec.build(cfg.mission_kwargs or {},
                           _strip_wing_objective(_flags),
                           cfg.bounds_overrides)
    raw = built.evaluate(np.asarray(x, dtype=float))
    out = {"x": [float(v) for v in np.asarray(x, dtype=float)],
           "feasible": bool(raw.get("feasible")),
           "reason": str(raw.get("reason") or ""),
           "metrics": {}, "scores": {}, "composite": None}
    if not raw.get("feasible"):
        out["reason"] = out["reason"] or "the design does not fly"
        return out
    out["metrics"] = {
        k: _finite_or_none(v)
        for k, v in _wing_metrics_fn(built, cfg.flags)(raw).items()}
    ref, info = _wing_reference_arg(reference, cfg)
    out["reference"] = info
    if ref is None:
        out["reason"] = info.get("reason", "")
        return out
    j, scores, reason = wsc.composite(out["metrics"], ref,
                                      wing_score_weights(weights))
    out["scores"] = {k: float(v) for k, v in scores.items()}
    out["composite"] = None if j is None else float(j)
    if j is None:
        out["reason"] = reason
    return out


def baseline_x(cfg: "RunConfig", built=None) -> list:
    """The design the search is measured AGAINST: the centre of the box.

    Not a random draw and not the first evaluation — the centre is the design
    a user gets by drawing a box and not searching it, it is reproducible, and
    it is sample 0 of the band sweep (``wing_score._sample_box``), so the
    baseline is inside the population its own score is normalised against.

    A PINNED variable is at its pin here too: the centre of a row the user
    took out of the search is not a design this run could ever return, so
    comparing the optimised design against it would price the pin as if the
    search had been free to move it.
    """
    if built is None:
        built = PROBLEM_SPECS[cfg.problem_name].build(
            cfg.mission_kwargs or {}, _strip_wing_objective(cfg.flags),
            cfg.bounds_overrides)
    pin = _pin_of(built, getattr(cfg, "pinned", None))
    box = np.asarray(built.bounds if pin is None else pin.bounds, dtype=float)
    return [float(v) for v in box.mean(axis=1)]


def score_optimised_design(cfg: "RunConfig", x, *, weights=None,
                           reference=None, baseline=None) -> dict:
    """BASELINE vs OPTIMISED on the six wing criteria — the run's other half.

    ``x`` is what the search returned; ``baseline`` defaults to the centre of
    the design box (:func:`baseline_x`). Returns both rows plus ``delta``
    (optimised - baseline, per criterion, per sub-score and on J), the weights
    they were read under, and the band's provenance.

    Two physics evaluations. On an ``lod`` run the optimiser did NOT search
    this composite, so a negative delta on a criterion is a real finding about
    the trade rather than a bug — it is the number that says what maximising
    L/D alone cost.
    """
    spec = PROBLEM_SPECS[cfg.problem_name]
    _flags, _dropped = sanitise_flags(cfg.problem_name, cfg.flags)
    built = spec.build(cfg.mission_kwargs or {},
                       _strip_wing_objective(_flags),
                       cfg.bounds_overrides)
    base_x = baseline_x(cfg, built) if baseline is None else list(baseline)
    w = wing_score_weights(weights)
    seed = score_design(cfg, base_x, weights=w, reference=reference,
                        built=built)
    opt = score_design(cfg, x, weights=w, reference=reference, built=built)

    delta: dict = {"composite": None, "scores": {}, "metrics": {}}
    if seed["composite"] is not None and opt["composite"] is not None:
        delta["composite"] = float(opt["composite"] - seed["composite"])
    for key, val in seed["scores"].items():
        if opt["scores"].get(key) is not None:
            delta["scores"][key] = float(opt["scores"][key] - val)
    for key, val in seed["metrics"].items():
        if val is not None and opt["metrics"].get(key) is not None:
            delta["metrics"][key] = float(opt["metrics"][key] - val)
    # …and where the answer sits in the POPULATION the band was measured
    # over. The centre is one design; "beats 29 of 31 box samples" is the
    # question a user actually asked, and the samples are already on disk.
    from . import wing_score as wsc
    ref, _ = _wing_reference_arg(reference, cfg)
    beats = (None if ref is None
             else wsc.population_rank(opt["composite"], ref, w))
    return {"problem": cfg.problem_name, "objective":
            str((cfg.flags or {}).get("wing_objective") or "lod"),
            "weights": w.normalised(),
            "reference": opt.get("reference") or seed.get("reference") or {},
            "baseline": seed, "optimised": opt, "delta": delta,
            "beats": beats}
