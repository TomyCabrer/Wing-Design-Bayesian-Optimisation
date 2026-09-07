"""A design box RECOMMENDED for a mission — measured, not asserted.

WHAT THIS IS FOR. A family's published design box is the box the solvers were
calibrated over, and the size rows of it already follow the aeroplane (a tail
area is a fraction of the wing's, an arm a fraction of its span). What none of
it follows is THIS MISSION: the same 0.2-1 taper, the same -4..4 root twist
and the same +-0.5 chord-law coefficients are searched whether the wing is
carrying 4.9 N at 12 m/s or 7000 N at 55, and most of that box does not fly
either of them. Measured on the run that motivated :mod:`optimize.feasible`,
92.6 % of a published box was refused before its solver ever ran.

So the recommendation is not a table of rules of thumb. It is a MEASUREMENT:

    draw the family's own box, evaluate, keep the designs that actually fly
    this mission, and report the smallest box around them.

Every number it returns therefore comes from this codebase's own physics
evaluated at this mission, not from a textbook constant restated here — and
every row gets a band, because every row is a coordinate of the draws that
flew. A box built this way is FEASIBLE by construction (it is drawn around
admissible points and nowhere else) and COHERENT by construction (those points
are jointly admissible, so the rows agree with each other rather than each
being defensible alone).

WHAT IT DOES NOT CLAIM. A box is a rectangle and a feasible region is not, so
the recommendation still contains corners that do not fly. It does not pretend
otherwise: :func:`recommend_box` re-probes its own answer and reports the
measured admissible fraction of it beside the published box's, so the card can
quote the improvement instead of asserting one.

THREE RULES IT KEEPS.

* DETERMINISTIC. Same mission, same family, same recommendation, bit-for-bit —
  the Sobol seed is an argument and defaults to 0. A recommendation that moved
  under a user would make the run that took it unreproducible, and
  reproducibility here is a property of the problem, not of the session.
* NEVER WIDER than the box it recommends for. A recommendation is not a
  licence: answering one more question must never widen a search, so every
  band returned is clipped into the family's own published row.
* NEVER A BAND OF WIDTH ZERO. The samplers raise on one and constrained BO
  quietly degrades to random draws while still calling itself BO, so a row
  whose admissible draws all landed on one value is widened to
  :data:`MIN_WIDTH_FRAC` of its published width rather than collapsed.
"""

from __future__ import annotations

import numpy as np

#: the smallest band a recommendation may return, as a fraction of the row's
#: own published width. A row can easily come back with every admissible draw
#: at one value — a coefficient the gates pin, a span the wing loading fixes —
#: and a zero-width band is refused by the api for good reason.
MIN_WIDTH_FRAC = 0.02

#: how far outside the admissible draws the band is drawn, as a fraction of
#: the row's published width. The draws are a SAMPLE of the feasible region,
#: so its true edge is a little outside the furthest point that landed; the
#: pad is what stops the recommendation from cutting off the optimum it was
#: meant to contain. Clipped into the published row afterwards.
PAD_FRAC = 0.05

#: the share of the admissible draws a recommendation is drawn around, ranked
#: by the family's own objective. Not a tuning constant with a measured
#: optimum: it is the answer to "how much of what flies do you want kept", and
#: a quarter leaves enough draws to bound a box with while cutting the half of
#: the box the mission's designs demonstrably do not use.
DEFAULT_QUANTILE = 0.25

#: draws taken to find the admissible region. A refusal returns before its
#: solver (measured 134x faster than a design that flies), so most of this is
#: cheap on the families where most of the box is refused — which are exactly
#: the families the recommendation is worth having on.
DEFAULT_N = 128


def _admissible(built, x):
    """``(flew, score)`` for one draw — solved, every margin >= 0, and what it
    scored. ``score`` is None where the draw did not fly.

    The SEARCH's definition of admissible, not the breakdown's: a design can
    solve and still miss a limit, and counting the ``feasible`` flag alone
    over-reports (on the box that motivated the probe, 5 of 64 draws solved and
    2 were admissible). Read from the same call the optimiser makes, so the
    designs this recommends over are the designs the run would accept.

    The score is the family's own objective, whatever it calls it, and it is
    MAXIMISED — the refusal sentinel is a large negative, which is the same
    contract read from the other end.
    """
    xa = np.asarray(x, dtype=float)
    raw = built.evaluate(xa)
    if not bool(raw.get("feasible")):
        return False, None
    if built.is_constrained:
        _fx, gx = built.callable(xa)
        if not bool(np.all(np.atleast_1d(np.asarray(gx, dtype=float)) >= 0.0)):
            return False, None
    f = raw.get("score", raw.get("LoD", raw.get("f")))
    return True, (float(f) if isinstance(f, (int, float)) else None)


def _classify(built, x):
    """``(verdict, score)`` for one draw — ``"admissible"``, ``"solved"`` or
    ``"refused"``.

    The three-way split :func:`_admissible` collapses into two, and the
    middle one is the whole reason this exists: a design that SOLVED and then
    missed a margin is a real evaluated design of this mission, while a
    REFUSED one never reached its solver at all. When nothing is admissible,
    the difference is between "here is where the physics at least runs" and
    "nothing in this box is even a wing" — two different answers, and a card
    that cannot tell them apart can only say "no recommendation".
    """
    xa = np.asarray(x, dtype=float)
    raw = built.evaluate(xa)
    if not bool(raw.get("feasible")):
        return "refused", None
    f = raw.get("score", raw.get("LoD", raw.get("f")))
    score = float(f) if isinstance(f, (int, float)) else None
    if built.is_constrained:
        _fx, gx = built.callable(xa)
        if not bool(np.all(np.atleast_1d(np.asarray(gx, dtype=float)) >= 0.0)):
            return "solved", score
    return "admissible", score


def classified_draws(built, box, n: int, seed: int):
    """``(n_drawn, admissible[], their scores[], solved-but-inadmissible[])``.

    One pass, three outcomes — see :func:`_classify`. The solved set is the
    fallback a recommendation falls back TO, so it is collected on the same
    draws rather than bought a second time.
    """
    from .optimize.feasible import sobol_pool

    box = np.asarray(box, dtype=float)
    kept, scores, solved = [], [], []
    pool = sobol_pool(box, max(1, int(n)), int(seed))
    for x in pool:
        verdict, f = _classify(built, x)
        if verdict == "admissible":
            kept.append(np.asarray(x, dtype=float))
            scores.append(f)
        elif verdict == "solved":
            solved.append(np.asarray(x, dtype=float))
    return len(pool), kept, scores, solved


def admissible_draws(built, box, n: int, seed: int):
    """``(n_drawn, [x that flew], [their scores])`` over ``box``."""
    drawn, kept, scores, _solved = classified_draws(built, box, n, seed)
    return drawn, kept, scores


def best_share(points, scores, quantile: float):
    """The ``quantile`` share of ``points`` that scored highest.

    Split out because it is the whole of the second criterion and it is pure:
    a box around everything that FLIES is the published box on any family whose
    box is already healthy (measured 90.6 % admissible on the reference tail at
    two mission scales, which recommends nothing at all), and the question a
    user is really asking a recommendation is not "what flies" but "where are
    the good ones". Ties are kept rather than broken, and a draw whose family
    exposes no scalar objective (``score`` None) falls back to the whole set —
    a recommendation must not silently rank on a number it does not have.
    """
    have = [f for f in scores if f is not None]
    if not points or len(have) != len(scores) or not (0.0 < quantile < 1.0):
        return list(points), None
    k = max(1, int(round(quantile * len(points))))
    cut = float(np.sort(np.asarray(have, dtype=float))[::-1][k - 1])
    keep = [p for p, f in zip(points, scores) if f >= cut]
    return keep, cut


def box_around(points, box, *, pad: float = PAD_FRAC,
               min_width: float = MIN_WIDTH_FRAC) -> np.ndarray:
    """The smallest box containing ``points``, padded, clipped into ``box``.

    Pure geometry and no physics, so the rule the card explains is the rule
    the test can state: min/max per coordinate, widened by ``pad`` of the
    published width on each side, floored at ``min_width`` of it so no row
    comes back with a band of width zero, and finally clipped — a
    recommendation may narrow a search and may never widen one.
    """
    box = np.asarray(box, dtype=float)
    P = np.asarray(points, dtype=float).reshape(len(points), -1)
    width = box[:, 1] - box[:, 0]
    lo = P.min(axis=0) - pad * width
    hi = P.max(axis=0) + pad * width
    # ...the floor, applied about the band's own centre so a row pinned by the
    # gates is recommended AT the value that flew rather than beside it
    short = (hi - lo) < min_width * width
    if np.any(short):
        mid = 0.5 * (lo + hi)
        half = 0.5 * min_width * width
        lo = np.where(short, mid - half, lo)
        hi = np.where(short, mid + half, hi)
    # ...and never outside the published row, in either direction. Clipped
    # after the floor, so a row already at the edge keeps its width by moving
    # inwards instead of being cut to nothing.
    lo = np.clip(lo, box[:, 0], box[:, 1])
    hi = np.clip(hi, box[:, 0], box[:, 1])
    return np.stack([np.minimum(lo, hi), np.maximum(lo, hi)], axis=1)


def probe_box(built, box, n: int, seed: int):
    """``(admissible fraction, best score)`` of a box — the two numbers a card
    quotes about a recommendation it is offering."""
    drawn, kept, scores = admissible_draws(built, box, n, seed)
    have = [f for f in scores if f is not None]
    return ((len(kept) / drawn if drawn else 0.0),
            (max(have) if have else None))


def recommend_box(built, *, n: int = DEFAULT_N, seed: int = 0,
                  pad: float = PAD_FRAC, quantile: float = DEFAULT_QUANTILE,
                  verify: bool = True, escalate: float = 3.0) -> dict:
    """The box this mission's BEST admissible designs live in.

    ``built`` is a built problem (``api.PROBLEM_SPECS[name].build(...)``), so
    the box drawn from is the one the run would search — this mission's flags
    already in it, the family's size rows already scaled to the aeroplane.

    Two criteria, in this order, because they answer two different failures:

    * ADMISSIBLE — the design flew. On a box most of which is refused this is
      the whole recommendation, and it is a large one (92.6 % of the box that
      motivated :mod:`optimize.feasible` was refused before its solver).
    * BEST ``quantile`` BY SCORE — where the good ones are. On a healthy box
      the first criterion recommends nothing at all (measured 90.6 % admissible
      on the reference tail, at both a 4.9 N and a 653 N mission), and this is
      the criterion that carries the answer.

    Returns:

    ``rows``        ``{label: (lo, hi)}``, the recommendation, or ``{}``
    ``empty``       True when NOTHING in the published box flew this mission.
                    A finding, not a failure to recommend: no narrowing helps a
                    box that contains no answer, and the caller belongs at the
                    conflict card instead (``api.size_box_conflicts``).
    ``n_admissible``/``n_best``/``frac_before``   what the measurement saw
    ``frac_after``/``best_after``   the recommendation RE-PROBED, on the same
                    seed and the same draw count
    ``best_before``/``score_cut``   the best score in the published box, and
                    the score a draw had to beat to be recommended over
    ``seed``        so the answer can be reproduced, and quoted when it is
    ``basis``       WHICH of the three measurements the bands came from:
                    ``"admissible"`` (designs that fly), ``"solved"`` (the
                    solver ran and a margin was missed — nothing flew), or
                    None (not one draw reached a solver). Always present, and
                    a caller that quotes a band without quoting this is
                    selling a box it cannot stand behind
    ``escalated``   True when the first sample found nothing and a bigger one
                    was taken

    WHY THERE IS A LADDER AT ALL. "No recommendation" is the least useful
    answer a card can give, and it was the common one on exactly the missions
    that need the box narrowed most: a thin admissible region is missed by a
    48-draw sample, and a box where NOTHING flies still has a place where the
    physics at least runs. So an empty first pass is followed by a bigger
    sample (``escalate`` x the draws, a different seed), and an empty second
    pass falls back to the solved-but-inadmissible draws — each labelled, none
    of them silent. What no measurement can invent is a box for a mission the
    gates rule out in closed form; that is :func:`api.size_box_conflicts`, and
    it is the caller's next question.

    WHAT A NARROWER BOX IS AND IS NOT. It is a smaller search: fewer draws are
    spent where this mission's designs do not live. It is NOT a better answer,
    and this project has measured that — over 42 seeds, narrowing the band did
    nothing for the objective and the tightest arm was the worst of them. So a
    caller may offer this as a starting point and must not sell it as a gain.
    """
    box = np.asarray(built.bounds, dtype=float)
    labels = list(built.param_labels)
    drawn, kept, scores, solved = classified_draws(built, box, n, seed)
    escalated = False
    if not kept and float(escalate) > 1.0:
        # A THIN REGION IS NOT AN EMPTY ONE. 48 draws over an 18-dimensional
        # box find nothing at all where 1 % of it flies, and "no
        # recommendation" is then a statement about the sample rather than
        # about the mission. Paid only when the first pass found nothing,
        # and on a different seed so it is a new sample and not the same one
        # extended.
        escalated = True
        more = int(max(1, round(float(escalate) * int(n))))
        d2, k2, s2, sv2 = classified_draws(built, box, more, int(seed) + 1)
        drawn, kept, scores = d2, k2, s2
        solved = solved + sv2
    have = [f for f in scores if f is not None]
    out = {"n": drawn, "n_admissible": len(kept), "seed": int(seed),
           "labels": labels, "quantile": float(quantile),
           "frac_before": (len(kept) / drawn if drawn else 0.0),
           "best_before": (max(have) if have else None),
           "n_best": 0, "score_cut": None,
           "frac_after": None, "best_after": None,
           "n_solved": len(solved), "escalated": escalated,
           "rows": {}, "empty": not kept, "basis": None}
    if not kept:
        if solved:
            # NOTHING FLIES, AND THAT IS ITSELF AN ANSWER ABOUT WHERE. These
            # designs reached their solver and missed a margin, so the box
            # around them is where this mission's physics at least runs —
            # offered as that, labelled as that, and never counted as an
            # admissible region.
            rec = box_around(solved, box, pad=pad)
            out["rows"] = {lab: (float(rec[i][0]), float(rec[i][1]))
                           for i, lab in enumerate(labels)}
            out["basis"] = "solved"
        return out
    out["basis"] = "admissible"
    best, cut = best_share(kept, scores, quantile)
    out["n_best"] = len(best)
    out["score_cut"] = cut
    # ...and the POINTS themselves, so a caller can measure a band on a
    # quantity that is not a row of the design vector. The second surface's
    # SPAN is the case that needs it: the solver searches that surface as an
    # area and an aspect ratio, and b = sqrt(AR*S) is exact — but it is a
    # FLAG PAIR, not a label, so nothing derived from `rows` alone could
    # propose it. A corner band built from the two rows is far wider than the
    # span the best designs actually use.
    out["best_x"] = [[float(v) for v in p] for p in best]
    rec = box_around(best, box, pad=pad)
    out["rows"] = {lab: (float(rec[i][0]), float(rec[i][1]))
                   for i, lab in enumerate(labels)}
    if verify:
        # the recommendation is RE-PROBED rather than asserted. A box is a
        # rectangle and a feasible region is not, so the honest numbers are a
        # measured fraction and a measured best score for THIS box — not a
        # claim that it contains only designs that fly, and not a claim that
        # its optimum is better than the one already found.
        out["frac_after"], out["best_after"] = probe_box(built, rec, drawn, seed)
    return out
