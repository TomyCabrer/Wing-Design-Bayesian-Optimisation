"""A field whose blank means a DEFAULT opens on that default.

THE BUG THIS EXISTS FOR. Stage 5's fin height, chord and leading-edge
station, and stage 6's maximum thrust, drag coefficient and span efficiency,
all follow the same rule: leave it blank and the model chooses. That rule is
right — a calibration is a default, not a ban, and a fin that follows the
span it is bolted to is better than one pinned the first time the panel was
opened. What was wrong is what it LOOKED like:

    height [m]   [        ]
    chord  [m]   [        ]

Three empty boxes on a design that has a fin 1.200 m tall with a 0.664 m
chord at x = 5.334 m. The number existed, the lattice was flying it, and the
one place a user looks for it said nothing. "There is a design but the box is
empty" is exactly right, and the same sentence came back about max thrust:
"default was 52 but the box is empty".

So a defaulted field OPENS ON THE VALUE THAT IS BEING FLOWN, and stores
nothing until it is typed into. Both halves matter:

* **opens on it**, so the box is never empty on a design that has an answer,
  and the number in the box is the number in the model — not a restatement
  of the default rule, which is how the two drift;
* **stores nothing**, so the field goes on following the design. Change the
  span and a blank height moves with it; it is only pinned once somebody
  types.

Clearing a pinned box puts it straight back to the model's own value, which
is why a box here can never be empty for long. The hint under the row says
which of the two states it is in and what clearing it would give back —
without that, "1.200" typed by the user and "1.200" chosen by the model are
the same pixels.
"""

from __future__ import annotations

__all__ = ["defaulted", "shown_value", "is_pinned"]


def is_pinned(stored) -> bool:
    """Has anybody actually answered this field?"""
    return stored not in (None, "")


def shown_value(stored, flown, dp: int | None = None):
    """What goes IN the box: the answer if there is one, else what is flown.

    Returns None only when there is no answer and nothing has been built
    yet — the one case where an empty box is the truth.

    ``dp`` ROUNDS WHAT IS SHOWN, and it is not cosmetic pedantry: the
    model's own fin chord is ``0.6635416666666665`` and the throttle ceiling
    ``53.99481778190805``, which overflow their boxes and read as noise.
    Only the DEFAULT is rounded — an answer somebody typed is shown exactly
    as they typed it — so the rounding can never quietly edit a number the
    user chose.

    What it CAN do is pin a rounded value if the field is focused and blurred
    without being touched, and that is a deliberate trade: 4 decimal places
    on a length is 0.1 mm, the exact value is printed to full precision in
    "Fin, as built" immediately below, and the alternative is a box showing
    eighteen digits of which four matter.
    """
    if is_pinned(stored):
        return float(stored)
    if flown is None:
        return None
    return float(flown) if dp is None else round(float(flown), dp)


def defaulted(num, label, *, stored, flown, default, on_change,
              fmt: str = "{:.4g}", unit: str = "", why: str = "",
              dp: int | None = None, **kw):
    """One defaulted row: the field, then a hint naming its state.

    ``num``      the caller's own field builder (each stage has one, with its
                 own debounce policy — a stage gets its own timing, not its
                 own widget).
    ``stored``   what the session holds: None means "the model chooses".
    ``flown``    what the model IS using right now. This is what the box
                 shows when nothing is stored, and it is read off the built
                 model rather than recomputed.
    ``default``  what clearing the box would give back. Only used in the
                 hint, and only when the field is pinned — so a caller may
                 pass None and pay nothing to find it out.
    ``why``      the rule in words, e.g. "12 % of the span".
    """
    from gui.v3 import theme, widgets

    el = num(label, shown_value(stored, flown, dp), on_change, unit=unit,
             **kw)
    pinned = is_pinned(stored)
    rule = f" ({why})" if why else ""
    if not pinned:
        widgets.hint(
            f"the model's own{rule} — it follows the design until you type "
            f"a number here.")
    else:
        back = ("" if default is None
                else f" Clear the box to go back to {fmt.format(float(default))}"
                     f"{(' ' + unit) if unit else ''}{rule}.")
        widgets.hint(f"YOURS, not the model's.{back}", "warn")
    del theme
    return el
