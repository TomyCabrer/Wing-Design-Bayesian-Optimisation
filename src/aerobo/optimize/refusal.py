"""What a REFUSED design should look like to the surrogate.

Every objective in this repo obeys one failure contract (``objective.py``):
a design the physics cannot score — solver divergence, an untrimmable
tail, an XFOIL polar with no converged point — returns exactly
``PENALTY = -100.0``. As a *label* that sentinel is well chosen: finite,
never NaN, and ordered below every real value, so an incumbent can never
be a refusal.

As a *training target* for a Gaussian process it is a different object,
and a bad one. The GP is fitted on standardised outputs, so the sentinel
sets the scale that every real observation is then measured against:

* the section problems maximise a composite of order 1, or minimise a
  drag coefficient of order 1e-3. Put one -100 beside them and the
  standardised feasible points collapse into a band roughly 1e-2 wide —
  the surrogate can no longer tell a good aerofoil from a bad one,
  because next to the cliff they are the same height;
* the length scales the marginal likelihood then fits are the length
  scales of the cliff, not of the design landscape, so the model becomes
  smooth and confident exactly where the interesting structure is;
* and the refusal region is usually a thin shell around a boundary, so
  the GP has to spend its flexibility interpolating a 100-unit step it
  will never be asked to predict.

This module is the alternative the study prices: keep the -100 as the
recorded value and the ordering, but show the MODEL a scale-aware stand-in
— the worst objective that was actually achieved. The refusal stays the
worst thing on the board (nothing can be preferred to it), and it stops
setting the units.

Modes
-----
``sentinel``
    Leave the -100 in place. The legacy path, and the default everywhere:
    :func:`impute` returns the values unchanged, so a run with this mode
    is bit-for-bit the run this repo has always done.
``worst``
    Replace every refusal by the worst FEASIBLE value observed so far.
    Refusals become ties with the worst real design.
``worst_margin``
    ``worst`` less :data:`MARGIN` of the observed feasible spread, so a
    refusal is still strictly worse than every design that flew — the
    model keeps a gradient pointing out of the refusal region instead of
    a plateau.

Conventions and edge cases
--------------------------
* MAXIMISATION only. The sentinel is only "the worst value" if bigger is
  better; :func:`impute` raises rather than guess for a minimised
  objective (callers flip the sign AFTER imputing).
* A refusal is recognised by EXACT equality with the sentinel, which is
  the contract: an objective returns ``PENALTY`` itself, never something
  near it. A real value of exactly -100.0 is not reachable by any
  objective in the registry (they are L/D, -Cd, or a 0-1 composite).
* If NOTHING feasible has been observed yet, there is no scale to be
  aware of, and every mode returns the values unchanged — an all-refusal
  training set is degenerate whatever number is in it, and the BO loop
  already handles the GP failure it causes by falling back to a random
  point.
"""

from __future__ import annotations

import numpy as np

#: the failure contract's value (objective.PENALTY, repeated here so the
#: optimisers do not import the physics layer)
PENALTY = -100.0

#: how far below the worst feasible design a refusal sits, as a fraction of
#: the observed feasible spread, in ``worst_margin``. Small enough that the
#: refusals do not set the scale (which is the whole point), large enough
#: that the model still sees a step.
MARGIN = 0.05

MODES: tuple[str, ...] = ("sentinel", "worst", "worst_margin")
DEFAULT_MODE = "sentinel"


def check_mode(mode: str | None) -> str:
    """Validate a mode name; ``None`` means the default."""
    m = DEFAULT_MODE if mode is None else str(mode)
    if m not in MODES:
        raise ValueError(f"unknown refusal mode {mode!r}; choices: {MODES}")
    return m


def is_refusal(y, penalty: float = PENALTY) -> np.ndarray:
    """Boolean mask of the entries that are the failure sentinel."""
    return np.asarray(y, dtype=float) == float(penalty)


def impute(y, mode: str | None = DEFAULT_MODE, *,
           penalty: float = PENALTY, margin: float = MARGIN,
           maximize: bool = True) -> np.ndarray:
    """Return ``y`` with refusal sentinels replaced per ``mode``.

    Never mutates the input, and never touches a value that is not the
    sentinel. In ``sentinel`` mode the returned array equals the input
    exactly, which is what makes the default path unchanged.
    """
    m = check_mode(mode)
    arr = np.array(y, dtype=float)          # copy
    if m == "sentinel":
        return arr
    if not maximize:
        raise ValueError(
            "refusal imputation is defined in the MAXIMISE convention "
            f"(the {penalty} sentinel is the worst value); impute before "
            "the sign flip, not after")
    bad = is_refusal(arr, penalty)
    if not bad.any():
        return arr
    good = arr[~bad]
    if good.size == 0:                      # nothing flew: no scale to use
        return arr
    worst = float(good.min())
    if m == "worst":
        arr[bad] = worst
        return arr
    spread = float(good.max()) - worst
    if spread <= 0.0:                       # one distinct feasible value
        spread = max(abs(worst), 1.0)
    arr[bad] = worst - float(margin) * spread
    return arr
