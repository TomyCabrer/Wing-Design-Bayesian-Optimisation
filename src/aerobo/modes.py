"""Which eigenvalue is which mode, and whether it grows.

WHY THIS IS IN THE ENGINE. It was written for the V4 Modes view and lived in
``gui/v4/modes.py`` while a view was its only consumer. :mod:`aerobo.handling`
scores designs on these modes — a gate the optimiser is run under cannot
import from a shell, and ``scripts/gate_ladder.py`` was already reaching into
one (``sys.path.insert(0, "gui")``) to measure the ladder. So the
implementation is here and ``gui/v4/modes.py`` re-exports it under every name
its callers already use, object for object: :func:`spiral_margin` is asserted
IDENTICAL through both paths (``test_the_objective_can_price_roll``), and the
same must stay true of everything else here. There is no adapter and no second
copy — the file that moved is this one.

The view's own account of why the modes are NAMED rather than sorted follows,
because it is the argument for the whole module and it did not change:

The Modes view already showed the eigenvalues of :func:`sixdof.linearise`
sorted by real part, which is honest and unreadable: eight numbers, and no
way to tell the phugoid the pilot is about to feel from the roll subsidence
they will never see. Worse, the one thing a pilot actually asks — "is it
stable?" — has two different answers here, and the interesting one is not
the one on the derivative deck:

    Cm_alpha  -2.119   pitch stiffness    STABLE
    Cl_beta   -0.020   dihedral effect    STABLE
    spiral    +0.0708  time to double 9.8 s   DIVERGENT

An aeroplane can pass every static sign test and still roll steadily into
the ground with the stick free, because the spiral is a RATIO — it goes
divergent when ``Cn_beta * Cl_r`` beats ``|Cl_beta| * |Cn_r|``, i.e. when
the fin is large for the dihedral it flies with. So the modes are named
here, and named from the EIGENVECTORS rather than by sorting.

THE SPLIT IS EXACT, NOT A HEURISTIC. A rigid aeroplane linearised about
straight and level flight has no coupling between the longitudinal states
(u, w, q, pitch) and the lateral ones (v, p, r, roll/yaw), so each
eigenvector lives wholly in one half. Measured on the tail design the
participations come out 1.00/0.00 to two decimals in all five modes, which
is what makes labelling by participation safe. Where a mode does NOT split
cleanly — a strongly asymmetric design, or a state far from level — the
participation is reported with the mode rather than hidden, so the label
can be disbelieved.

Everything here is arithmetic on a matrix: no nicegui, no session, no
solver. The tests are against this.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import dynamics as _dyn

__all__ = ["Mode", "classify", "STATE_SLICES", "participation",
           "NEUTRAL_1_PER_S", "spiral_margin", "spiral_margin_of",
           "SLOW_LATERAL", "LATERAL_OSC", "why_not_a_spiral",
           "ALIAS_WHEN_FINLESS", "find"]

#: where each physical quantity sits in ``sixdof.State.to_vector()``:
#: ``[pos(3), quat(4), vel(3), rates(3)]``.
STATE_SLICES = {"pos": slice(0, 3), "quat": slice(3, 7),
                "vel": slice(7, 10), "rates": slice(10, 13)}

#: an eigenvalue this small in magnitude is not a mode. Four of the thirteen
#: states are position (which no force depends on) and one quaternion
#: direction is the norm, so five eigenvalues are structurally zero.
_ZERO = 1e-8

#: a real part this small [1/s] is NEUTRAL, not a divergence. Set two orders
#: above the measured agreement between the finite-difference Jacobian and a
#: time-domain fit (6e-6 vs 1.7e-5 on the tail design's phugoid), and
#: physically it is a time to double of ``ln2 / 1e-4`` = 6931 s — one hour
#: and 55 minutes, which no flight in this stage reaches. It is a floor on
#: what can be RESOLVED, not a handling-qualities threshold: a mode that
#: doubles in 60 s is still reported as divergent, and should be.
NEUTRAL_1_PER_S = 1e-4


@dataclass(frozen=True)
class Mode:
    """One named mode, with everything a pilot or a plot needs about it."""

    name: str                 # "phugoid", "short period", ...
    real: float               # [1/s]
    imag: float               # [rad/s], 0 for a real mode
    lateral: float            # 0..1 — how much of it is v, p, r, roll, yaw

    @property
    def oscillatory(self) -> bool:
        return abs(self.imag) > 1e-6

    @property
    def period_s(self) -> float | None:
        return (2.0 * math.pi / abs(self.imag)) if self.oscillatory else None

    @property
    def damping(self) -> float | None:
        """Damping ratio, or None for a non-oscillatory mode."""
        mag = math.hypot(self.real, self.imag)
        return (-self.real / mag) if (self.oscillatory and mag > 0) else None

    @property
    def tau_s(self) -> float | None:
        """Time constant of a real mode — None for an oscillatory one."""
        if self.oscillatory or abs(self.real) < _ZERO:
            return None
        return 1.0 / abs(self.real)

    @property
    def neutral(self) -> bool:
        """Is this mode's real part indistinguishable from zero?

        A BARE ``real < 0`` HAS NO TOLERANCE, and the Jacobian it comes from
        is a finite difference. Measured on the shipped tail design the
        phugoid lands at ``+6e-6`` — the same value at every step size from
        1e-8 to 1e-3, and a 400 s time-domain fit of the speed envelope
        agrees at ``+1.7e-5`` — so the mode is neutral to well under
        :data:`NEUTRAL_1_PER_S`, and the panel was labelling it DIVERGENT
        with a time to double of 115 795 s. Thirty-two hours is not a
        divergence a pilot meets; it is zero.
        """
        return abs(self.real) < NEUTRAL_1_PER_S

    @property
    def stable(self) -> bool:
        """Convergent, or neutral. A mode that neither grows nor decays is
        not a thing to colour red."""
        return self.real < 0.0 or self.neutral

    @property
    def diverges(self) -> bool:
        """Growing, and by more than the numerics can account for. The
        positive statement, so a view can distinguish "converges" from
        "neutral" from "DIVERGES" instead of splitting a line in two."""
        return self.real > 0.0 and not self.neutral

    @property
    def double_or_half_s(self) -> float | None:
        """Time to double (unstable) or to half (stable) the amplitude.

        The number a handling-qualities requirement is written in, and the
        one that says whether a divergence matters: a spiral that doubles in
        60 s is a trim change, one that doubles in 8 s is a workload.

        None for a NEUTRAL mode, which is the honest answer: it never
        doubles and it never halves.
        """
        if self.neutral:
            return None
        return math.log(2.0) / abs(self.real)


def participation(vec: np.ndarray, V: float) -> float:
    """Fraction of an eigenvector that is LATERAL, in 0..1.

    Speeds are divided by the flight speed so they compare with the
    quaternion components, which are already half-angles; the body RATES are
    left out entirely, because for an eigenvector they are just the angles
    multiplied by the eigenvalue and would weight fast modes by their own
    frequency.
    """
    q = np.abs(np.asarray(vec)[STATE_SLICES["quat"]])
    u, v, w = np.abs(np.asarray(vec)[STATE_SLICES["vel"]]) / max(abs(V), 1e-9)
    lon = q[2] ** 2 + u ** 2 + w ** 2          # qy (pitch), u, w
    lat = q[1] ** 2 + q[3] ** 2 + v ** 2       # qx (roll), qz (yaw), v
    tot = lon + lat
    return 0.5 if tot <= 0 else float(lat / tot)


#: what the slow lateral real root is CALLED on an aeroplane that does not
#: weathercock. It is still a root, still measured, still shown — it is just
#: not the spiral, because with no yaw stiffness the yaw equation decouples
#: and what is left is a sideslip subsidence. See :func:`why_not_a_spiral`.
SLOW_LATERAL = "slow lateral root"
#: ...and the lateral oscillation on the same design, which is not a Dutch
#: roll for the same reason: the Dutch roll IS yaw stiffness against roll.
LATERAL_OSC = "lateral oscillation"


#: what a view looking for the named mode should fall back to when the
#: design does not weathercock. The ROOT is still there and still worth
#: showing; only its name changed, so a panel keyed on "spiral" reads this
#: rather than losing the number — which on `hydrofoil + elevator` is a
#: +0.44979 divergence that DOUBLES IN 1.54 s.
ALIAS_WHEN_FINLESS = {"spiral": SLOW_LATERAL, "dutch roll": LATERAL_OSC}


def find(named: dict, key: str):
    """The mode a view asked for, under whichever name it came back under.

    ``(mode, name)`` — the name is what to LABEL it, so a panel never calls
    a sideslip subsidence "spiral". ``(None, key)`` when neither is there.
    """
    m = named.get(key)
    if m is not None:
        return m, key
    alias = ALIAS_WHEN_FINLESS.get(key)
    if alias is not None and named.get(alias) is not None:
        return named[alias], alias
    return None, key


def why_not_a_spiral(Cn_beta) -> str:
    """Why this design's slow lateral root is not called the spiral, or ``""``.

    The sentence a panel shows where :data:`SLOW_LATERAL` appears instead of
    ``"spiral"``. It asks :func:`dynamics.weathercocks`, which is also what
    :func:`wing_score.spiral_refusal` asks, so the criterion and the mode
    namer cannot disagree about the same aeroplane.
    """
    if Cn_beta is None or _dyn.weathercocks(Cn_beta):
        return ""
    return (f"Cn_beta = {float(Cn_beta):+.6f}: this design does NOT "
            f"weathercock, so its slow lateral root is not a spiral — with "
            f"no yaw stiffness the yaw equation decouples and what is left "
            f"is a sideslip subsidence. Its eigenvalue is real and is shown "
            f"under {SLOW_LATERAL!r}; what must not be said about it is that "
            f"the spiral converges.")


def classify(J: np.ndarray, V: float, *,
             Cn_beta: float | None = None) -> dict[str, Mode]:
    """Name the modes of a linearised aeroplane.

    ``J`` is :func:`sixdof.linearise`'s 13x13 Jacobian and ``V`` the flight
    speed it was taken at. Returns the modes that were actually found, keyed
    by name — a design with no fin has no Dutch roll and no spiral to
    return, and inventing them would be worse than the gap.

    ``Cn_beta`` IS THAT RULE, MADE OPERATIONAL. Without it the namer had no
    way to know whether an aeroplane weathercocks, so it named the slow
    lateral real root "spiral" on every design — including one with the fin
    switched off, where the root is -0.02221 and the panel therefore said
    the spiral HALVES in 31.2 s on an aircraft with ``Cn_beta`` of exactly
    -0.0, while ``api.dihedral_for_spiral`` refused the very same design.
    Two instruments, one aeroplane, opposite answers. Pass the deck's
    ``Cn_beta`` and the roots that are not those modes are returned under
    :data:`SLOW_LATERAL` and :data:`LATERAL_OSC` instead — nothing is
    hidden, and the eigenvalue is unchanged. Omit it and the naming is
    exactly what it always was, so no existing caller moves.

    Within each half the ordering is the physics and not a preference:
    of the two longitudinal oscillations the SLOWER is the phugoid (it is
    an energy exchange between height and speed, and gravity sets its
    frequency); of the two lateral real roots the FASTER is roll
    subsidence (it is the roll damping working against the roll inertia).
    """
    cocks = Cn_beta is None or _dyn.weathercocks(Cn_beta)
    spiral_name = "spiral" if cocks else SLOW_LATERAL
    dutch_name = "dutch roll" if cocks else LATERAL_OSC
    vals, vecs = np.linalg.eig(np.asarray(J, dtype=float))
    lon_osc, lat_osc, lon_real, lat_real = [], [], [], []
    seen: set[int] = set()
    for i in range(vals.size):
        lam = vals[i]
        if abs(lam) < _ZERO:
            continue
        if i in seen:
            continue
        # a conjugate pair is ONE mode. Take the +imag member and mark its
        # partner, so a pair is never reported twice under two names.
        if abs(lam.imag) > 1e-6:
            for j in range(i + 1, vals.size):
                if j not in seen and abs(vals[j] - np.conj(lam)) < 1e-9:
                    seen.add(j)
                    break
            if lam.imag < 0:
                continue
        lat = participation(vecs[:, i], V)
        m = Mode(name="", real=float(lam.real), imag=float(abs(lam.imag)),
                 lateral=lat)
        if abs(lam.imag) > 1e-6:
            (lat_osc if lat > 0.5 else lon_osc).append(m)
        else:
            (lat_real if lat > 0.5 else lon_real).append(m)

    out: dict[str, Mode] = {}

    def _named(m: Mode, name: str):
        out[name] = Mode(name=name, real=m.real, imag=m.imag,
                         lateral=m.lateral)

    # longitudinal: slowest oscillation is the phugoid, the other the short
    # period. With only one, its frequency decides which it is called.
    lon_osc.sort(key=lambda m: math.hypot(m.real, m.imag))
    if len(lon_osc) >= 2:
        _named(lon_osc[0], "phugoid")
        _named(lon_osc[-1], "short period")
    elif lon_osc:
        _named(lon_osc[0],
               "phugoid" if math.hypot(lon_osc[0].real, lon_osc[0].imag) < 1.0
               else "short period")

    # lateral: the oscillation is the Dutch roll; of the real roots the fast
    # one is roll subsidence and the slow one the spiral.
    if lat_osc:
        lat_osc.sort(key=lambda m: math.hypot(m.real, m.imag))
        _named(lat_osc[-1], dutch_name)
    lat_real.sort(key=lambda m: abs(m.real))
    if len(lat_real) >= 2:
        _named(lat_real[0], spiral_name)
        _named(lat_real[-1], "roll subsidence")
    elif lat_real:
        _named(lat_real[0],
               "roll subsidence" if abs(lat_real[0].real) > 1.0
               else spiral_name)
    return out


#: THE ENGINE OWNS THIS ONE NOW. It was defined here while the only
#: consumer was a view; ``wing_score``'s roll criterion scores designs on it,
#: and a number an objective maximises cannot have its definition in a shell.
#: Re-exported under the name every caller already uses, so there is one
#: formula and no adapter.
spiral_margin = _dyn.spiral_margin
#: ...and the one a caller WITH A DECK should use: it reads the trim
#: attitude off the deck, which the classical form drops and which is
#: 2.13 deg of dihedral on the family this panel flies.
spiral_margin_of = _dyn.spiral_margin_of
