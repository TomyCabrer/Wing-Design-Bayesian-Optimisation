"""The SIZE modifier: span and area as design variables, paid for in weight.

The trap, restated once (aircraft.py derives it in full)
-------------------------------------------------------
Free the span in a pure-aero model and the optimiser runs away: at fixed
lift D_i ~ 1/b^2 while the profile drag is span-blind, so "maximise L/D with
free b" slams the box. What closes it is structure — span costs wing weight,
weight costs lift, lift costs induced drag — which is why this modifier is
NOT "two more rows in the box". Switching it on changes three things at once,
and all three are needed for the question to be honest:

1. the wing WEIGHS what its size implies (weights.total_weight, Raymer's
   statistical GA wing weight closed as a fixed point in the gross weight);
2. the trim target follows that weight: CL = W_total / (q S) per candidate,
   so a bigger wing is not simply a bigger wing;
3. the score becomes PAYLOAD L/D, f = W_fixed / D. With W varying across
   candidates "maximise L/D" would reward a heavier wing that happens to lift
   better; W_fixed is the same constant for every candidate, so maximising
   W_fixed/D IS minimising drag IS minimising thrust and power at fixed V
   (mission.py's equivalence chain).

...and it ADDS a constraint: the root bending stress margin
g = ln(sigma_allow / sigma_root) >= 0, in the LOG form aircraft.py measured
to be the GP-friendly one (sigma ~ b^4 spans two decades over a span box, and
the linear margin's tail flattens the feasible island into numerical noise).

What the reference weight means
-------------------------------
A family carrying this modifier has to say what the aircraft weighs BESIDE
its wing. There is no way to read that off a fixed-CL problem — CL_target
tells you the TOTAL the wing lifts, not the split — so the default is the
honest, stated one: ``W_fixed`` defaults to the weight the family's own trim
target implies (mission.weight_for), and the wing's structural weight is then
added ON TOP. A sized problem is therefore HEAVIER than the fixed-CL problem
it extends, and its payload L/D is correspondingly lower; that is the price
of asking the question rather than a discrepancy. Pass ``W_fixed_N``
explicitly for a real weight breakdown.

Three ways to size, not two
---------------------------
``size_free`` carries the MODE (:func:`size_mode`), and there are three:

* :data:`SIZE_MODE_FREE` — span AND area are design variables. The wing
  loading is then an OUTPUT: whatever ``W_total/S`` the optimiser lands on.
* :data:`SIZE_MODE_WS` — the mission states W/S and the area follows it
  through the weight loop, so the vector carries the span alone and the trim
  lift coefficient is fixed at ``(W/S)/q`` by construction.
* :data:`SIZE_MODE_WS_FREE` — the wing loading is itself a design variable,
  inside a band stated in the units the mission thinks in (N/m²). The area
  still follows it through the same fixed point, so every candidate is a
  closed aircraft; what changes is that the search may re-open the mission's
  question, on the record, between two loadings the user is willing to fly.

Why the third mode exists, when the free one already moves the loading: in
the free mode W/S is reachable only through the AREA row, whose band is a
fraction of the family's own reference area. That band is blind to the
mission — a 1500 N aircraft and a 300 N one get the same 8-22 m² box — so the
loadings a search can actually reach are an accident of the reference size
rather than a decision. Asking for the loading directly puts the question in
the units the constraint diagram answers it in, and lets the mission's own
ceiling (:data:`wing_loading_max_Pa`, from constraint_diagram) clip the box
instead of being a card nobody reads.

Honesty labels inherited from aircraft.py: Raymer's correlation is calibrated
for GA-class aircraft; the wing weight is span-soft (~b^1.2 at fixed S), so
pure aerodynamics wants sailplane proportions and it is the STRESS constraint
that pulls the design back — the good feasible designs ride g = 0. The
statistical weight has no chord-distribution term, so a chord law changes the
aerodynamics and the trim target, never W_wing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import materials, weights
from .weights import N_ULT_DEFAULT, SIGMA_ALLOW_PA

#: design-vector labels of the size block, in order.
SIZE_LABELS = ("b_m", "S_m2")

#: the SPAN row of surface ``i`` when a family carries more than one surface
#: of its own — the tandem pair. Surface 0 keeps the plain ``b_m`` name (it is
#: the wing every single-surface family means by "the span"), and the rest are
#: named for the surface they belong to, so a design box row still says which
#: wing it widens. See :func:`span_labels`.
EXTRA_SPAN_LABELS = ("b_rear_m",)

#: size box as a FRACTION of the family's own published (b, S). The numbers
#: are aircraft.py's absolute boxes divided by its 10 m / 10 m^2 reference:
#: span up to 4x (generous ON PURPOSE — the unconstrained aerodynamic optimum
#: is sailplane-like and must be INTERIOR so the box never picks the answer),
#: area 0.8-2.2x. Fractional rather than absolute so a 1.2 m foil and a 10 m
#: wing both get a box around their own size.
B_FRAC_BOUNDS = (0.6, 4.0)
S_FRAC_BOUNDS = (0.8, 2.2)

#: aspect-ratio band the wing solvers stay honest over (api.PLANFORM_AR_LIMITS
#: restated here to avoid importing the api from the physics layer). Outside
#: it a candidate is an in-contract FAILURE — the penalty contract — not a
#: scored design: below ~3 the lifting-line reduction and the one-chordwise-
#: panel Weissinger VLM stop describing a wing, above ~40 the fixed-Re section
#: polar and the rigid-planform structure are the binding lie.
AR_LIMITS = (3.0, 40.0)

#: sweep the statistical weight is evaluated at. Sweep is a design variable
#: only in the tier_a_plus family; everywhere else the wing is unswept, and
#: aircraft.py passes 0 explicitly for exactly this reason.
SWEEP_FIXED_DEG = 0.0


#: the three ways a family can be sized. ``size_free`` carries the MODE, not
#: a bool — ``False`` is off, ``True`` is the two-variable mode this module
#: shipped with, :data:`SIZE_MODE_WS` is the one-variable mode below, and
#: :data:`SIZE_MODE_WS_FREE` searches the loading itself. All are truthy, so
#: every ``if prob.size_free:`` in the package still means "this problem is
#: sized".
SIZE_MODE_FREE = "free"
SIZE_MODE_WS = "wing_loading"
SIZE_MODE_WS_FREE = "wing_loading_free"

#: the modes whose AREA comes out of the weight loop rather than the design
#: vector. Both need a fixed weight and a dynamic pressure to close that loop
#: (:func:`area_for_wing_loading`); they differ only in WHERE the loading
#: comes from — a stated number, or a row of the box.
WS_MODES = (SIZE_MODE_WS, SIZE_MODE_WS_FREE)

#: the W/S mode's design vector: the SPAN, and nothing else.
SPAN_LABELS = ("b_m",)

#: the searched-loading mode's extra row. A loading is a PRESSURE, and the
#: label says so — the same rule every other row follows (b_m, S_m2, l_t_m).
WS_LABEL = "ws_pa"

#: default band for that row, as a fraction of the loading the family
#: already flies (CL_target x q). MODEL CHOICE, and the same argument as
#: B_FRAC_BOUNDS: wide enough that the answer is INTERIOR — half to double
#: the current loading covers the whole span of GA practice around any
#: sensible starting point — and stated here so a user who wants a different
#: one edits a band rather than discovers a ban. The mission's own ceiling
#: narrows it further wherever one is given.
WS_FRAC_BOUNDS = (0.5, 2.0)


def size_mode(size_free) -> str:
    """``"off"`` / ``"free"`` / ``"wing_loading"`` / ``"wing_loading_free"``.

    One place decides what the state means, so no family has to know that
    ``True`` is the two-variable mode.
    """
    if not size_free:
        return "off"
    if size_free is True or str(size_free) == SIZE_MODE_FREE:
        return SIZE_MODE_FREE
    if str(size_free) in WS_MODES:
        return str(size_free)
    raise ValueError(
        f"unknown size mode {size_free!r}; choose from "
        f"{(False, True) + WS_MODES}")


def span_labels(n_spans: int = 1) -> tuple:
    """The size block's SPAN labels for a family with ``n_spans`` surfaces.

    One row per surface that has a span of its own. A tandem's two wings do:
    tying the rear wing's span to the front's was a modelling shortcut, not a
    property of the aeroplane, and the pair's induced drag is a function of
    both spans separately (the mutual term is what this family exists to
    study). Every other family has exactly one.
    """
    n = int(n_spans)
    if n < 1 or n > 1 + len(EXTRA_SPAN_LABELS):
        raise ValueError(
            f"n_spans must be 1..{1 + len(EXTRA_SPAN_LABELS)}, got {n_spans}")
    return SPAN_LABELS + EXTRA_SPAN_LABELS[:n - 1]


def size_bounds(b0: float, S0: float, n_spans: int = 1,
                span_bounds_m=None, area_bounds_m2=None) -> np.ndarray:
    """(1 + n_spans, 2) box for the size block around a family's own (b, S).

    Every span row opens on the SAME fractional band around ``b0``: the two
    wings of a pair are alternatives for the same job, so opening the rear
    one narrower would answer the question the search is being asked.

    ``span_bounds_m`` / ``area_bounds_m2`` are the user's own bands, in
    metres and square metres. They are the reason this mode is usable off the
    family's published mission at all: the fractional box is a fraction of a
    REFERENCE SIZE, so a 300 N aircraft and a 3000 N one are handed the same
    8-22 m² of area to choose from and only one of them can contain its own
    design point. Left None each falls back to that fractional band, so every
    published run is unchanged.
    """
    rows = list(span_bounds(b0, span_bounds_m, n_spans))
    rows.append(area_bounds(S0, area_bounds_m2)[0])
    return np.array(rows, dtype=float)


#: the smallest positive number a design-box row may start at. "As close to
#: zero as this problem allows" is a real answer — a vanishing tip chord, a
#: stabiliser shrunk until the trim solve gives up, a foil at the surface —
#: and refusing the literal 0 that expresses it made the shell say no to a
#: question the physics answers perfectly well (it comes back INFEASIBLE with
#: a reason, which is the honest outcome). It is the value ``tail.py`` already
#: uses for its own documented ``S_t -> 0`` probe.
MIN_POSITIVE = 1e-9


def positive_floor(lo: float, hi: float, what: str) -> float:
    """``lo``, lifted to :data:`MIN_POSITIVE` when it is zero or below.

    A NEGATIVE end is still refused where the quantity cannot be negative —
    that is a different statement from zero, and one the user cannot have
    meant. An inverted or collapsed pair is refused by the caller, unchanged.
    """
    lo = float(lo)
    if lo > 0.0:
        return lo
    if lo < 0.0:
        raise ValueError(
            f"{what} starts at {lo}, which is below zero — a negative one "
            f"is not a smaller one, it is a different sign. Zero is accepted "
            f"and read as {MIN_POSITIVE:g}, the closest this solve can get.")
    if not float(hi) > MIN_POSITIVE:
        raise ValueError(
            f"{what} is ({lo}, {hi}); with the low end read as "
            f"{MIN_POSITIVE:g} there is no interval left to search.")
    return MIN_POSITIVE


def area_bounds(S0: float, area_bounds_m2=None) -> np.ndarray:
    """(1, 2) box for the free mode's AREA row, defaulted or stated."""
    lo, hi = _ends(area_bounds_m2, S_FRAC_BOUNDS, S0)
    lo = positive_floor(lo, hi, f"the area band ({lo}, {hi}) m^2")
    if not hi > lo:
        raise ValueError(
            f"area bounds must satisfy 0 <= min < max, got ({lo}, {hi})")
    return np.array([[lo, hi]], dtype=float)


def ws_bounds(ws0: float, ws_bounds_pa=None,
              wing_loading_max_Pa: float | None = None) -> np.ndarray:
    """(1, 2) box for the searched-loading row [Pa].

    ``ws_bounds_pa`` is the user's own band; left None it is
    :data:`WS_FRAC_BOUNDS` around the loading the family already flies, so
    the mode opens on that operating point rather than on somebody's idea of
    a wing loading.

    ``wing_loading_max_Pa`` is the MISSION's ceiling — the binding ws_max
    line of its constraint diagram (stall, landing field, fly-up, cavitation)
    — and it CLIPS the band rather than replacing it: a search may not choose
    a loading the mission forbids, and a user who asked for a narrower band
    than the mission allows keeps their own. A band that sits entirely above
    the ceiling is refused here, loudly, because silently returning an empty
    interval would hand the sampler a collapsed dimension.
    """
    lo, hi = _ends(ws_bounds_pa, WS_FRAC_BOUNDS, ws0)
    lo = positive_floor(lo, hi, f"the wing-loading band ({lo}, {hi}) Pa")
    if not hi > lo:
        raise ValueError(
            f"wing-loading bounds must satisfy 0 <= min < max, "
            f"got ({lo}, {hi})")
    if wing_loading_max_Pa is not None:
        cap = float(wing_loading_max_Pa)
        if not (cap > 0.0):
            raise ValueError(
                f"the mission's wing-loading ceiling must be > 0, got {cap}")
        if cap <= lo:
            raise ValueError(
                f"the wing-loading band ({lo:g}, {hi:g}) Pa lies at or above "
                f"the mission's own ceiling of {cap:g} Pa — the mission "
                f"allows no loading this search may fly (constraint_diagram)")
        hi = min(hi, cap)
    return np.array([[lo, hi]], dtype=float)


def ws_of(CL_target: float, rho: float, V: float) -> float:
    """The wing loading a trimmed problem already flies [Pa]: ``CL q``.

    One line, one place: it is what the searched-loading band opens around
    in every family, and six families computing ``CL * 0.5 * rho * V**2``
    each is six chances for one of them to drift.
    """
    return float(CL_target) * 0.5 * float(rho) * float(V) ** 2


def _ends(band, frac_bounds, ref: float) -> tuple:
    """``(lo, hi)`` of a user band, each end defaulted where it is None.

    One place turns "the user stated part of a band" into two numbers, so
    the span, the area and the loading all read a half-stated band the same
    way — against their OWN reference size, which is the only one that means
    anything (a 1.2 m foil and a 10 m wing share no absolute default).
    """
    lo = hi = None
    if band is not None:
        lo, hi = band[0], band[1]
    return (frac_bounds[0] * float(ref) if lo is None else float(lo),
            frac_bounds[1] * float(ref) if hi is None else float(hi))


def span_bounds(b0: float, span_bounds_m=None,
                n_spans: int = 1) -> np.ndarray:
    """(n_spans, 2) box for the W/S mode's span rows — one per surface.

    ``span_bounds_m`` is the user's own (min, max) in metres, which is what
    the mode is for: with the area following the wing loading, the span is
    the one thing left to decide, and a span is decided by a hangar, a
    trailer, a class rule or a spar — never by a fraction of some default.
    Left None the box is the same fractional one the two-variable mode uses,
    so the mode is usable before anyone has typed a limit. ONE end may be
    None inside the pair, and only that end falls back: "no wider than 15 m"
    is a complete sentence, and the answer to it is not a floor nobody asked
    for.
    """
    lo, hi = _ends(span_bounds_m, B_FRAC_BOUNDS, b0)
    lo = positive_floor(lo, hi, f"the span band ({lo}, {hi}) m")
    if not hi > lo:
        raise ValueError(
            f"span bounds must satisfy 0 <= min < max, got ({lo}, {hi})")
    return np.array([[lo, hi]] * len(span_labels(n_spans)), dtype=float)


def size_labels(size_free, n_spans: int = 1) -> tuple:
    mode = size_mode(size_free)
    if mode == "off":
        return ()
    spans = span_labels(n_spans)
    if mode == SIZE_MODE_WS:
        return spans
    return spans + ((WS_LABEL,) if mode == SIZE_MODE_WS_FREE else ("S_m2",))


def with_size_bounds(box, size_free, b0: float, S0: float,
                     span_bounds_m=None, n_spans: int = 1, *,
                     area_bounds_m2=None, ws0: float | None = None,
                     ws_bounds_pa=None,
                     wing_loading_max_Pa: float | None = None) -> np.ndarray:
    """Append the size rows to a family's own box.

    Call this BEFORE the flight and chord blocks: the package's modifier
    stacking order is ``[family][size][flight][chord]`` and every modifier
    block is read back from the END of the vector (see :func:`size_from_x`).

    ``n_spans`` is how many surfaces of its own the family carries — 1
    everywhere except the tandem pair, whose two wings each get a span row
    (:func:`span_labels`).

    ``ws0`` is the loading the family already flies, and the searched-loading
    mode needs it: it is what the default band opens around. A family that
    asks for that mode without one is asking to search a quantity it cannot
    state, which is refused here rather than defaulted to a number nobody
    chose.
    """
    base = np.asarray(box, dtype=float)
    mode = size_mode(size_free)
    if mode == "off":
        return base
    if mode == SIZE_MODE_WS:
        rows = span_bounds(b0, span_bounds_m, n_spans)
    elif mode == SIZE_MODE_WS_FREE:
        if ws0 is None and (ws_bounds_pa is None
                            or any(v is None for v in ws_bounds_pa)):
            raise ValueError(
                "the searched wing-loading mode needs the loading the family "
                "already flies (ws0) or a fully stated band to open on")
        rows = np.vstack([
            span_bounds(b0, span_bounds_m, n_spans),
            ws_bounds(0.0 if ws0 is None else ws0, ws_bounds_pa,
                      wing_loading_max_Pa)])
    else:
        rows = size_bounds(b0, S0, n_spans, span_bounds_m, area_bounds_m2)
    return np.vstack([base, rows])


def n_size_rows(size_free, n_spans: int = 1) -> int:
    """How many rows the size block occupies: the span rows, plus ONE more
    unless the loading is stated — the AREA in the free mode, the LOADING in
    the searched-loading mode, and nothing at all in the W/S mode (there the
    area follows a number the mission already gave)."""
    mode = size_mode(size_free)
    if mode == "off":
        return 0
    return len(span_labels(n_spans)) + (0 if mode == SIZE_MODE_WS else 1)


def spans_from_x(x, size_free, flight_free: bool = False,
                 chord_order: int = 0, n_spans: int = 1):
    """The size block read back out of a design vector, or None.

    ``([b_0, ...], S)`` in the two-variable mode; ``([b_0, ...], None)`` in
    BOTH loading modes, where the area is not a design variable at all — it
    follows a wing loading through the weight loop
    (:func:`area_for_wing_loading`), whether that loading was stated or
    searched (:func:`loading_from_x`).

    The size block sits ahead of the flight pair, which sits ahead of the
    chord coefficients — the exact inverse of the stacking rule, needing no
    knowledge of the family block's length. Within the block the span rows
    come first, in surface order, and the one derived-from row (the area, or
    the loading) last.
    """
    mode = size_mode(size_free)
    if mode == "off":
        return None
    n_rows = n_size_rows(size_free, n_spans)
    n_b = len(span_labels(n_spans))
    xs = np.asarray(x, dtype=float)
    n_after = int(chord_order) + (2 if flight_free else 0)
    if xs.size < n_after + n_rows:
        raise ValueError(
            f"design vector of length {xs.size} carries no room for a size "
            f"block of {n_rows} ahead of {n_after} modifier entries")
    j = xs.size - n_after - n_rows
    spans = [float(v) for v in xs[j:j + n_b]]
    area = None if mode in WS_MODES else float(xs[j + n_b])
    return spans, area


def loading_from_x(x, size_free, flight_free: bool = False,
                   chord_order: int = 0, n_spans: int = 1):
    """The W/S row [Pa] of a searched-loading vector, or None.

    None for every other mode — including the stated-loading one, whose
    loading is not in the vector at all — so a caller can ask this question
    of any candidate and get "the vector does not decide it" as an answer.
    """
    mode = size_mode(size_free)
    if mode != SIZE_MODE_WS_FREE:
        return None
    n_rows = n_size_rows(size_free, n_spans)
    n_b = len(span_labels(n_spans))
    xs = np.asarray(x, dtype=float)
    n_after = int(chord_order) + (2 if flight_free else 0)
    if xs.size < n_after + n_rows:
        raise ValueError(
            f"design vector of length {xs.size} carries no room for a size "
            f"block of {n_rows} ahead of {n_after} modifier entries")
    return float(xs[xs.size - n_after - n_rows + n_b])


def size_from_x(x, size_free, flight_free: bool = False,
                chord_order: int = 0):
    """``(b, S)`` for a single-surface family — :func:`spans_from_x`'s
    one-span case, which is every family but the tandem pair."""
    got = spans_from_x(x, size_free, flight_free, chord_order, n_spans=1)
    if got is None:
        return None
    spans, area = got
    return spans[0], area


def flow_state_for(x, *, mission, flight_free: bool, n_trailing: int,
                   rho: float, V: float) -> tuple:
    """``(rho, V)`` a candidate flies at, BEFORE anything needs an area.

    Only the wing-loading mode needs this: the area it derives depends on the
    dynamic pressure, and with the flight modifier on that pressure is itself
    a design variable. Density and speed do not depend on the area, so they
    can be read early — the trim target, which does, is still built once,
    later, from the resolved size (mission.flight_state).

    ``n_trailing`` is how many rows sit AFTER the flight block (the chord
    coefficients — one block per designed surface, so families with two
    surfaces pass their own count).
    """
    from . import geometry

    flight = geometry.flight_from_x(x, flight_free, int(n_trailing))
    if flight is None:
        return float(rho), float(V)
    from .mission import flight_state

    st = flight_state(mission, flight[0], flight[1], 1.0)   # S sets CL only
    return float(st["rho"]), float(st["V"])


def area_for_wing_loading(*, W_fixed_N: float, b: float,
                          wing_loading_Pa: float, taper: float, tc: float,
                          q_Pa: float, area_fracs=None, wing_spans=None,
                          n_ult: float = N_ULT_DEFAULT,
                          sweep_deg: float = SWEEP_FIXED_DEG,
                          material=None,
                          tol: float = 1e-10, max_iter: int = 200) -> float:
    """The AREA a chosen wing loading implies, with the weight closed.

    W/S is the mission's answer (see constraint_diagram.py) and the span is
    the aerodynamicist's, so this mode asks the optimiser for the span alone
    and derives the area from

        S = W_total / (W/S),     W_total = W_fixed + W_wing(b, S, W_total)

    — a fixed point, because the wing's own statistical weight depends on the
    area it is being solved for. It converges from below for the same reason
    :func:`sized_state`'s inner loop does: Raymer's wing weight grows
    sub-linearly in S, so the map is a contraction over any sane loading.

    The consequence worth stating: with W/S fixed the TRIM LIFT COEFFICIENT
    is fixed too — ``CL = W_total/(q S) = (W/S)/q`` — whatever span comes
    out. That is exactly why choosing a wing loading is choosing an operating
    point, and why this mode does not re-open the mission's question.

    ``area_fracs`` (optional) is how a multi-surface family splits the
    reference area between wings that are weighed separately (a tandem
    pair); each is weighed at the shared gross weight, as in
    :func:`sized_state`.

    ``wing_spans`` (optional) is the SPAN of each of those wings, in the same
    order. A pair whose two wings need not be the same width is two
    structures of two different spans, and Raymer's weight is superlinear in
    b — weighing both on the front wing's span made the rear one free to be
    the shorter wing at no saving, which is half the trade the pair exists to
    study. Omitted, every surface is weighed on ``b`` (one wing, or the
    equal-span pair every published run flew).
    """
    if not (wing_loading_Pa > 0.0):
        raise ValueError(f"wing loading must be > 0, got {wing_loading_Pa}")
    if not (W_fixed_N > 0.0 and b > 0.0):
        raise ValueError("fixed weight and span must both be > 0")
    fracs = [1.0] if area_fracs is None else [float(f) for f in area_fracs]
    spans = ([float(b)] * len(fracs) if wing_spans is None
             else [float(v) for v in wing_spans])
    if len(spans) != len(fracs):
        raise ValueError(
            f"one span per weighed surface: got {len(spans)} spans for "
            f"{len(fracs)} area fractions")
    if not all(v > 0.0 for v in spans):
        raise ValueError(f"every wing span must be > 0, got {tuple(spans)}")
    k_w = materials.resolve(material).k_w
    S = float(W_fixed_N) / float(wing_loading_Pa)
    for _ in range(int(max_iter)):
        areas = [f * S for f in fracs]
        W_total = float(W_fixed_N)
        for _inner in range(100):
            W_wing = sum(
                weights.wing_weight_raymer(b_i, a, taper, sweep_deg, tc,
                                           n_ult, W_dg_N=W_total, q_Pa=q_Pa,
                                           k_w=k_w)
                for b_i, a in zip(spans, areas))
            new = float(W_fixed_N) + W_wing
            if abs(new - W_total) < 1e-8:
                W_total = new
                break
            W_total = new
        else:
            raise ValueError("wing-weight fixed point did not converge")
        S_new = W_total / float(wing_loading_Pa)
        if abs(S_new - S) <= tol * max(1.0, S):
            return float(S_new)
        S = S_new
    raise ValueError(
        f"area / wing-loading fixed point did not converge for b = {b:g} m "
        f"at W/S = {wing_loading_Pa:g} Pa")


def resolve_spans(x, size_free, flight_free: bool = False,
                  chord_order: int = 0, *, n_spans: int = 1,
                  wing_loading_Pa: float | None = None,
                  W_fixed_N: float | None = None, taper: float = 1.0,
                  tc: float = 0.12, q_Pa: float = 1.0, area_fracs=None,
                  n_ult: float = N_ULT_DEFAULT,
                  sweep_deg: float = SWEEP_FIXED_DEG,
                  wing_loading_max_Pa: float | None = None):
    """``([b_0, ...], S)`` for a candidate, whichever size mode is on.

    The one entry point a family needs, for any number of surfaces that carry
    a span of their own (:func:`span_labels`): it reads the block out of the
    vector and, in either loading mode, closes the area's own fixed point.
    Raises ValueError for anything a caller should map to the penalty contract
    (a loading that cannot be closed, a missing loading, a loading the mission
    forbids).

    In the SEARCHED-loading mode the loading comes out of the vector and any
    ``wing_loading_Pa`` passed in is ignored: one number, one owner, and the
    owner is the design box.

    The area the loading implies is the family's REFERENCE area — the pair's
    total, not one wing's — so ``area_fracs`` splits it exactly as the
    family's own solver does and each span is weighed on its own structure
    (:func:`area_for_wing_loading`).
    """
    mode = size_mode(size_free)
    if mode == "off":
        return None
    spans, S = spans_from_x(x, size_free, flight_free, chord_order,
                            n_spans=n_spans)
    if mode == SIZE_MODE_FREE:
        return spans, S
    if mode == SIZE_MODE_WS_FREE:
        wing_loading_Pa = loading_from_x(x, size_free, flight_free,
                                         chord_order, n_spans)
    if wing_loading_Pa is None or W_fixed_N is None:
        raise ValueError(
            "the wing-loading size mode needs a wing loading and a fixed "
            "weight to derive the area from")
    check_wing_loading(float(wing_loading_Pa), wing_loading_max_Pa)
    # every weighed surface on its own span, in the family's own order: with
    # one span row this is the single wing, with two it is the pair (and
    # ``area_fracs`` is then the split the same candidate flies). Materialised
    # once — it is read twice here, and a caller passing a generator would
    # otherwise hand the fixed point an empty one.
    fracs = None if area_fracs is None else tuple(area_fracs)
    spans_for_weight = (None if fracs is None
                        else [spans[min(i, len(spans) - 1)]
                              for i in range(len(fracs))])
    S = area_for_wing_loading(
        W_fixed_N=W_fixed_N, b=spans[0], wing_loading_Pa=wing_loading_Pa,
        taper=taper, tc=tc, q_Pa=q_Pa, area_fracs=fracs,
        wing_spans=spans_for_weight, n_ult=n_ult, sweep_deg=sweep_deg)
    return spans, S


def resolve_size(x, size_free, flight_free: bool = False,
                 chord_order: int = 0, *, wing_loading_Pa: float | None = None,
                 W_fixed_N: float | None = None, taper: float = 1.0,
                 tc: float = 0.12, q_Pa: float = 1.0, area_fracs=None,
                 n_ult: float = N_ULT_DEFAULT,
                 sweep_deg: float = SWEEP_FIXED_DEG,
                 wing_loading_max_Pa: float | None = None):
    """``(b, S)`` — :func:`resolve_spans`'s one-span case, which is every
    family but the tandem pair."""
    got = resolve_spans(x, size_free, flight_free, chord_order, n_spans=1,
                        wing_loading_Pa=wing_loading_Pa, W_fixed_N=W_fixed_N,
                        taper=taper, tc=tc, q_Pa=q_Pa, area_fracs=area_fracs,
                        n_ult=n_ult, sweep_deg=sweep_deg,
                        wing_loading_max_Pa=wing_loading_max_Pa)
    if got is None:
        return None
    spans, S = got
    return spans[0], S


#: the units string every sized family's ``score`` carries (see
#: :meth:`SizedState.payload_lod`). Restated in ``api`` as
#: ``api.PAYLOAD_LOD_UNITS`` so ``import aerobo.api`` stays free of
#: physics imports; a test keeps the two equal. NOT "L/D": this is
#: W_fixed/D, smaller than the aero L/D by exactly W_fixed/W_total, and
#: the gap measures 17.0-34.7 L/D points across the 376 registered sized
#: variants. A shell handed only the number cannot tell them apart,
#: which is why the name travels with it.
PAYLOAD_LOD_UNITS = "payload L/D"


@dataclass(frozen=True)
class SizedState:
    """What a candidate size implies: weight, trim target, stress margin."""

    b: float
    S: float
    W_fixed_N: float
    W_total_N: float
    W_wing_N: float
    CL_target: float
    sigma_root_Pa: float
    sigma_allow_Pa: float
    g_sigma: float            # ln(sigma_allow / sigma_root) — the CONSTRAINT
    g_sigma_lin: float        # linear margin, for engineering readability
    #: what the EMPENNAGE weighs [N] — 0.0 for a family that carries none,
    #: which is what every one of them did before the layout was priced.
    #: Kept apart from ``W_wing_N`` because "the wing weighs this" and "the
    #: aeroplane's surfaces weigh this" are different statements, and the
    #: wing's own composite scores the first one.
    W_tail_N: float = 0.0
    #: WHAT IT IS BUILT OF (materials.Material). Carried on the state rather
    #: than left in the caller because the weight and the allowable are both
    #: consequences of it, and a result that reports the two numbers without
    #: naming their source is a result nobody can reproduce.
    material: object = None

    def payload_lod(self, q_Pa: float, CD: float) -> float:
        """f = W_fixed / D — the objective a sized problem maximises."""
        D_N = q_Pa * self.S * float(CD)
        return float(self.W_fixed_N / D_N) if D_N > 0.0 else float("nan")

    @property
    def wing_loading_Pa(self) -> float:
        """W/S the candidate actually flies — an OUTPUT in the free mode, the
        stated or searched number in the loading modes, and the same quantity
        either way, so a result can always be read as a wing loading."""
        return float(self.W_total_N / self.S)

    def report(self) -> dict:
        """Breakdown keys every sized family adds to its result dict.

        ``score_units`` is one of them, and it is not decoration: every
        caller of this method goes on to overwrite ``score`` with
        :meth:`payload_lod`, and the two statements are the same edit. A
        family that adds the size block without renaming its axis is the
        defect this key exists to make unmakeable.
        """
        out = {
            "score_units": PAYLOAD_LOD_UNITS,
            "b_m": self.b, "S_m2": self.S,
            "W_fixed_N": self.W_fixed_N, "W_total_N": self.W_total_N,
            "W_wing_N": self.W_wing_N,
            # ...AND WHAT THE EMPENNAGE COST. Reported always, including the
            # 0.0 a family with no tail carries: "the tail weighs nothing
            # here" is a statement about the model, and a key that appeared
            # only when it was non-zero would make its absence ambiguous.
            "W_tail_N": self.W_tail_N,
            "wing_loading_Pa": self.wing_loading_Pa,
            "sigma_root_Pa": self.sigma_root_Pa,
            "sigma_allow_Pa": self.sigma_allow_Pa,
            "g_sigma": self.g_sigma, "g_sigma_lin": self.g_sigma_lin,
        }
        if self.material is not None:
            out.update(self.material.report())
        return out


def check_wing_loading(ws_Pa: float, wing_loading_max_Pa: float | None
                       ) -> None:
    """Refuse a wing loading the MISSION does not allow.

    ``wing_loading_max_Pa`` is the binding ``ws_max`` line of the mission's
    own constraint diagram — stall or landing field length in air, fly-up or
    cavitation in water. Until this existed that diagram was a card at stage
    1 and nothing else: a sized search was free to buy payload L/D with a
    loading the aircraft could not land at, and did (measured: the free
    planform's optimum sits at roughly twice the published mission's W/S,
    with the stall speed inside 20 % of cruise).

    It is an IN-CONTRACT refusal, like the aspect-ratio band and for the same
    reason: the design is not a failure of the solver, it is a design the
    mission forbids, so the optimiser must see a penalty rather than an
    exception. Callers already wrap this path and map ValueError to
    ``_fail("size: ...")``.

    None means "this mission states no ceiling", which is every published run.
    """
    if wing_loading_max_Pa is None:
        return
    cap = float(wing_loading_max_Pa)
    if not (cap > 0.0):
        raise ValueError(
            f"the mission's wing-loading ceiling must be > 0, got {cap}")
    if float(ws_Pa) > cap:
        raise ValueError(
            f"wing loading {float(ws_Pa):.4g} Pa above the mission's own "
            f"limit of {cap:.4g} Pa (constraint_diagram)")


def check_ar(b: float, S: float, limits=None) -> str | None:
    """Reason string if (b, S) leaves the honest aspect-ratio band, else None.

    ``limits`` is the band the USER asked for — ``(ar_min, ar_max)``, either
    end None for "no opinion" — and it is a SECOND band, not a replacement:
    :data:`AR_LIMITS` is where these solvers describe a wing at all, so a
    user band is intersected with it and can only ever narrow. The two are
    named apart in the refusal, because "the solvers do not model this" and
    "you asked me not to draw this" are different answers to "why was my
    design refused" and a card that quotes the wrong one sends the user to
    change the wrong number.

    With ``limits`` unset this is bit-for-bit the published gate, which is
    what every run before the limit existed flew.
    """
    if not (b > 0.0 and S > 0.0):
        return f"non-positive size (b = {b:g} m, S = {S:g} m^2)"
    ar = b * b / S
    lo, hi = AR_LIMITS
    if not (lo <= ar <= hi):
        return (f"aspect ratio {ar:.3g} outside the band {AR_LIMITS} these "
                f"solvers are valid over")
    return check_ar_limit(b, S, limits)


def check_ar_limit(b: float, S: float, limits=None) -> str | None:
    """The USER's aspect-ratio band alone — no solver band, no default.

    Split out of :func:`check_ar` so a family that has no validity gate of
    its own (the water families size their foil against a draught cap
    instead) can honour the limit a user set without acquiring a band nobody
    asked it for. Returns None where no limit was set, so an unset limit is
    not a gate that always passes: it is no gate at all.
    """
    if limits is None:
        return None
    lo_u, hi_u = limits
    if lo_u is None and hi_u is None:
        return None
    if not (b > 0.0 and S > 0.0):
        return f"non-positive size (b = {b:g} m, S = {S:g} m^2)"
    ar = b * b / S
    if lo_u is not None and ar < float(lo_u):
        return (f"aspect ratio {ar:.3g} below the minimum {float(lo_u):.3g} "
                f"you set")
    if hi_u is not None and ar > float(hi_u):
        return (f"aspect ratio {ar:.3g} above the maximum {float(hi_u):.3g} "
                f"you set")
    return None


def sized_state(*, W_fixed_N: float, b: float, S: float, taper: float,
                tc: float, q_Pa: float, wing_areas=None, wing_spans=None,
                n_ult: float = N_ULT_DEFAULT,
                sigma_allow: float | None = None,
                material=None,
                c_root: float | None = None,
                sweep_deg: float = SWEEP_FIXED_DEG,
                empennage=(),
                wing_loading_max_Pa: float | None = None) -> SizedState:
    """Close the weight loop for one candidate size and price its spar.

    ``wing_areas`` (optional) lists the areas of the individual wings when
    the family carries more than one surface of its own — the tandem pair —
    so each is weighed as its own wing at the SHARED gross weight, and their
    sum is the structural weight. ``S`` stays the reference area the trim
    target is defined on. Left None the wing is a single surface of area S.

    ``wing_spans`` (optional) is the matching list of SPANS, for a pair whose
    two wings are not the same width. Structure is what makes span cost
    something here (module docstring), so a rear wing 30 % shorter than the
    front one has to be weighed and stressed as the shorter wing it is, or
    the modifier would charge the pair for span it does not have. Left None
    every wing is ``b`` wide, which is what every single-surface family — and
    every equal-span pair — means, so those runs are unchanged.

    The stress margin is the WORST of the wings: the constraint is "this
    aeroplane's spar holds", and it is the wing nearest its allowable that
    decides. Each wing is priced against the reference area exactly as the
    single-surface call is, so an equal-span pair reproduces the one call it
    used to make, bit-for-bit.

    ``empennage`` (optional) lists the tail surfaces this layout carries
    (:class:`weights.TailSurface`), and they are weighed in the SAME fixed
    point: a tailplane and a fin both grow with the gross weight they
    stabilise, so ``W_total = W_fixed + W_wing(W_total) + W_tails(W_total)``
    and the map is still a contraction (every exponent is below 1). Left
    EMPTY — which every family without a tail passes — this call is
    bit-for-bit the one it always made, so nothing that never carried an
    empennage moves.

    What it changes where it IS passed: the empennage stops being a constant
    inside ``W_fixed``. A V-tail's two panels weigh what two panels weigh and
    it carries no fin at all; a T-tail's fin carries the tailplane and
    Raymer charges 20 % for it. Those are the layout differences the mission
    stage asks about, finally priced.

    Raises ValueError only for a genuinely invalid size (the caller maps that
    to the penalty contract); an overstressed-but-flyable design returns a
    NEGATIVE g_sigma, never an exception, because a constrained optimiser has
    to see the violation's magnitude.
    """
    reason = check_ar(b, S)
    if reason is not None:
        raise ValueError(reason)
    # ONE HOME FOR "what it is built of". The material carries both numbers a
    # material decides — the allowable the spar is sized to and the factor the
    # statistical weight is scaled by — so the two cannot be set to disagree.
    # ``sigma_allow`` survives as an explicit OVERRIDE for a caller that wants
    # the stress gate moved without touching the weight (the calibration
    # studies do); left None it is the material's own.
    mat = materials.resolve(material)
    k_w = mat.k_w
    sigma_allow = (mat.sigma_allow_Pa if sigma_allow is None
                   else float(sigma_allow))
    areas = [float(S)] if wing_areas is None else [float(a) for a in wing_areas]
    spans = ([float(b)] * len(areas) if wing_spans is None
             else [float(v) for v in wing_spans])
    if len(spans) != len(areas):
        raise ValueError(
            f"wing_spans has {len(spans)} entries against {len(areas)} wing "
            f"areas — one span per surface, in the same order")

    # fixed point: each wing's statistical weight takes the gross weight it
    # helps carry as an input, so W_total = W_fixed + sum_i W_wing_i(W_total)
    W_total = float(W_fixed_N)
    W_wing = W_tail = 0.0
    for _ in range(100):
        W_wing = sum(
            weights.wing_weight_raymer(b_i, a, taper, sweep_deg, tc, n_ult,
                                       W_dg_N=W_total, q_Pa=q_Pa, k_w=k_w)
            for b_i, a in zip(spans, areas))
        # ...AND THE EMPENNAGE, in the same iteration: it stabilises the
        # gross weight it is part of, exactly as the wing lifts it. Zero
        # where no tail was described, which is every family that has none.
        W_tail = weights.empennage_weight(empennage, n_ult, W_total,
                                          q_Pa=q_Pa, k_w=k_w)
        new = float(W_fixed_N) + W_wing + W_tail
        if abs(new - W_total) < 1e-8:
            W_total = new
            break
        W_total = new
    else:
        raise ValueError("wing-weight fixed point did not converge")
    if not (np.isfinite(W_total) and W_total > 0.0):
        raise ValueError("non-finite / non-positive total weight")

    # the loading this candidate ends up flying, against the mission's own
    # ceiling. Checked HERE because this is the one place every sized family
    # closes the weight loop, so the free mode — where W/S is an outcome of
    # the area row rather than an input — is covered by the same line as the
    # loading modes (which also clip their box, so this is their backstop).
    check_wing_loading(W_total / float(S), wing_loading_max_Pa)

    sigma = max(weights.root_bending_stress(b_i, S, taper, tc, W_total, n_ult,
                                            c_root=c_root)
                for b_i in ([float(b)] if wing_spans is None else spans))
    g_log = float(np.log(sigma_allow / sigma))
    return SizedState(
        b=float(b), S=float(S), W_fixed_N=float(W_fixed_N),
        W_total_N=float(W_total), W_wing_N=float(W_wing),
        W_tail_N=float(W_tail),
        CL_target=float(W_total / (q_Pa * S)),
        sigma_root_Pa=float(sigma), sigma_allow_Pa=float(sigma_allow),
        g_sigma=g_log,
        g_sigma_lin=float((sigma_allow - sigma) / sigma_allow),
        material=mat,
    )
