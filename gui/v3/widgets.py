"""The CAE-desktop primitives every V3 stage is built from.

Contract (stages use ONLY these plus raw nicegui):

* ``with pane("Simulation tree"): ...``      — framed pane, with a title bar
* ``with group_box("Operating point"): ...`` — group box in a work area
* ``Tree(on_select, on_toggle)``             — the object tree (left pane)
* ``PropertyGrid()``                         — the properties table below it
* ``TabStrip(on_change)``                    — the work-area tab row
* ``OutputLog()`` / ``StatusBar()``          — the bottom window and strip
* ``number_field / readout / kv / tag / hint`` — dense form + read-out bits

Nothing here knows what a wing is: the stages own the meaning, these own
the chrome. nicegui is imported lazily inside every call, so importing
``gui.v3.widgets`` starts nothing.
"""

from __future__ import annotations

import re

import time

from . import theme

#: tree-node state -> (material icon, colour) — the glyph IS the status, so
#: a node never needs a second badge to say "done"
NODE_STATE_ICONS = {
    "done": ("check_circle", theme.GOOD),
    "ready": ("radio_button_unchecked", theme.INK_MUTED),
    "active": ("play_circle", theme.ACCENT),
    "running": ("pending", theme.ACCENT),
    "locked": ("lock", theme.INK_FAINT),
    "warn": ("error_outline", theme.WARN),
    "error": ("cancel", theme.BAD),
}


#: A Quasar body cell painted by its row's own VERDICT (the row's ``dir``
#: field: ``better`` / ``worse`` / ``same`` / ``""``). Green for a metric the
#: search improved, red for one it gave up, grey for one that did not move or
#: has no preferred direction at all.
#:
#: Every seed-vs-optimised table in the shell asks the same question — "did
#: this get better?" — and every one of them used to answer it with a signed
#: number in a column of signed numbers, leaving the reader to do the sum per
#: row. One template, so the answer cannot be painted two different ways on
#: two stages.
#:
#: The arrow is NOT decoration: colour alone is not a readable difference for
#: a red-green colour-blind reader, so the verdict is carried twice — once in
#: hue, once in a glyph — the same rule this shell's status tags follow.
VERDICT_TPL = """
<q-td :props="props" class="text-right">
  <span :style="props.row.dir === 'better' ? 'color:__GOOD__;font-weight:600'
        : props.row.dir === 'worse' ? 'color:__BAD__;font-weight:600'
        : 'color:__FAINT__'">
    {{ props.row.dir === 'better' ? '▲ ' :
       props.row.dir === 'worse' ? '▼ ' :
       props.row.dir === 'same' ? '= ' : '' }}{{ props.value }}
  </span>
</q-td>
"""

#: the same verdict on the OPTIMISED value rather than on the change: the
#: number the eye lands on first is the one worth colouring, and the change
#: column beside it says by how much
VERDICT_VALUE_TPL = """
<q-td :props="props" class="text-right">
  <span :style="props.row.dir === 'better' ? 'color:__GOOD__;font-weight:600'
        : props.row.dir === 'worse' ? 'color:__BAD__;font-weight:600' : ''">
    {{ props.value }}
  </span>
</q-td>
"""


def paint(tpl: str) -> str:
    """A Quasar slot template with this shell's palette substituted in."""
    return (tpl.replace("__GOOD__", theme.GOOD)
               .replace("__BAD__", theme.BAD)
               .replace("__FAINT__", theme.INK_FAINT))


def verdict_slots(table, *, value_field: str = "new",
                  change_field: str = "change"):
    """Colour a seed-vs-optimised table by each row's ``dir``; returns it.

    Every row handed to such a table must carry a ``dir``
    (:func:`gui.metrics.verdict`, ``nice_app._cmp_row``) — a template reading
    ``props.row.dir`` on rows without one paints everything grey, which looks
    exactly like a working colour rule on a run where nothing changed.
    """
    table.add_slot(f"body-cell-{value_field}", paint(VERDICT_VALUE_TPL))
    table.add_slot(f"body-cell-{change_field}", paint(VERDICT_TPL))
    return table


# ------------------------------------------------------------------ frames
def pane(title: str, *, white: bool = False, pad: bool = False,
         header_extra=None, flex: str = ""):
    """Framed pane with a title bar; returns the BODY (a context manager).

    ``header_extra`` is a zero-arg callable rendered right-aligned in the
    title bar (tool buttons). ``white`` puts the body on the work surface
    instead of the panel grey; ``pad`` adds the standard inner padding.

    ``flex`` styles the FRAME, not the body — the caller needs to size the
    pane inside its column, and styling what this returns would instead
    stretch the scrolling body inside the frame (which is how the output
    log first came out three times its height).
    """
    from nicegui import ui

    root = ui.element("div").classes("pane w-full h-full")
    if flex:
        root.style(flex)
    with root:
        if title or header_extra is not None:
            with ui.element("div").classes("pane-title"):
                ui.label(title)
                ui.space()
                if header_extra is not None:
                    header_extra()
        body = ui.column().classes(
            "pane-body w-full gap-0" + (" white" if white else "")
            + (" pane-pad" if pad else ""))
    body.pane_root = root
    return body


def group_box(title: str, *, pad: bool = True, help: str = "",
              help_title: str = ""):
    """Group box inside a work area; returns the body column.

    ``help`` puts a ``?`` in the TITLE BAR — the one place a whole group of
    controls, or a figure, can be explained without a paragraph above it.
    """
    from nicegui import ui

    root = ui.element("div").classes("grouping w-full")
    with root:
        if title:
            with ui.element("div").classes("group-title w-full"):
                if help:
                    with ui.row().classes("items-center gap-1 no-wrap"):
                        ui.label(title)
                        help_dot(help, title=help_title or title)
                else:
                    ui.label(title)
        body = ui.column().classes(
            "w-full gap-2" + (" group-pad" if pad else ""))
    body.group_root = root
    return body


def hairline():
    from nicegui import ui

    return ui.element("div").style(
        f"height:1px;width:100%;background:{theme.RULE_SOFT}")


# ------------------------------------------------------------------- bits
#: how many words a quiet line may keep ON SCREEN. Past this it is not a
#: hint, it is a paragraph, and a paragraph beside a control is what buries
#: the control — so :func:`hint` keeps its opening sentence and puts the
#: rest behind a ``?``. THE RULE LIVES HERE, not at the ~200 call sites: the
#: shell writes its explanations as whole paragraphs (which is right — they
#: are read in full in the popup, and they are what the reader needs when
#: they need anything), and one place decides how much of each is on screen.
#: Nothing is deleted by it: :func:`split_hint` moves words, it never drops
#: them, and a line that cannot be split cleanly keeps its whole text in the
#: popup rather than being cut.
HINT_WORDS_ON_SCREEN = 24

#: sentence boundaries a hint may be split at. A decimal point does not
#: match ("0.488 m in front"), because the next character has to open a
#: sentence.
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z“‘\"(])")
#: ...and the fallback, for a first sentence that is itself a paragraph: the
#: em-dash and the semicolon are where this shell's long lines actually
#: turn. The separator STAYS ON THE LEAD (the lookbehind matches the space
#: AFTER it): a line ending "…for the job —" says out loud that there is
#: more, and a split that swallowed its own dash would be deleting a
#: character rather than moving one, which is the one thing this may not do.
_CLAUSE = re.compile(r"(?<=[—;])\s+")


def split_hint(text: str, limit: int = HINT_WORDS_ON_SCREEN) -> tuple:
    """``(on_screen, behind_the_mark)`` for one quiet line.

    The opening sentence stays where it is and the rest goes behind the
    ``?`` — which is the shape these lines were already written in, most of
    them leading with the measurement and following with why it is that
    number. A first sentence longer than ``limit`` is split at its last
    clause boundary instead; one with no boundary at all keeps its first
    ``limit`` words on screen and its WHOLE text in the popup, so the
    sentence can still be read as a sentence.
    """
    words = str(text).split()
    if len(words) <= limit:
        return (str(text), "")
    for pattern in (_SENTENCE, _CLAUSE):
        best = None
        for m in pattern.finditer(str(text)):
            lead = str(text)[:m.start()]
            if len(lead.split()) > limit:
                break
            best = m
        if best is not None:
            return (str(text)[:best.start()].strip(),
                    str(text)[best.end():].strip())
    return (" ".join(words[:limit]) + " …", str(text))


def hint(text: str, kind: str = "", *, title: str = "", split: bool = True):
    """Quiet explanatory line ('' | 'warn' | 'bad').

    Long ones keep their first sentence and hand the rest to a ``?``
    (:data:`HINT_WORDS_ON_SCREEN`). ``split=False`` is for the two callers
    that hold the returned label and rewrite its text later — a popup built
    from words that have since been replaced is worse than a plain line.
    """
    from nicegui import ui

    cls = "hint" + (f" hint-{kind}" if kind else "")
    if split:
        lead, rest = split_hint(text)
        if rest:
            return hint_help(lead, rest, title=title, kind=kind)
    return ui.label(text).classes(cls)


def help_dot(text: str, *, title: str = ""):
    """A VISIBLE ``?`` carrying the explanation short label text cannot hold.

    This shell already had two explanation channels and neither does this
    job. ``tip=`` is a Quasar tooltip on a field's LABEL: nothing on screen
    says it is there, so it is read only by a reader who already knew. And
    :func:`hint` is a line of quiet prose UNDER the control, which is right
    for eight words and wrong for four sentences — a paragraph per field is
    what turned these stages into a wall of text with the controls buried
    in it.

    So: a mark the eye finds, next to the thing it explains, holding the
    paragraph. Click, not hover — the text is two to four sentences and a
    hover popup dies the moment the pointer moves toward it.

    ``text`` is split on blank lines into paragraphs. ``title`` is optional
    and is the question being answered, not a restatement of the label.
    """
    from nicegui import ui

    dot = ui.element("div").classes("help-dot").props('role="button"')
    with dot:
        ui.label("?")
        with ui.menu().classes("help-pop"):
            with ui.column().classes("gap-1").style(
                    "padding:8px 10px;max-width:320px"):
                if title:
                    ui.label(title).classes("help-pop-title")
                for para in [p.strip() for p in text.split("\n\n") if p.strip()]:
                    ui.label(para).classes("hint")
    return dot


def tag(text: str, color: str):
    """State pill (FEASIBLE / RUNNING / 6-D …)."""
    from nicegui import ui

    return ui.label(text).classes("tag").style(
        f"color:{color};border-color:{color};background:{_tint(color)}")


def retag(el, text: str, color: str):
    """Re-letter an existing pill in place.

    The pill's own row may hold a ``ui.number`` the user is typing into, and
    rebuilding that row swallows the rest of the number — so a caller that
    only needs the WORD to change writes it here instead of redrawing.
    """
    el.set_text(text)
    return el.style(f"color:{color};border-color:{color};"
                    f"background:{_tint(color)}")


def _tint(color: str, alpha: float = 0.10) -> str:
    c = color.lstrip("#")
    if len(c) != 6:
        return "transparent"
    try:
        r, g, b = (int(c[i:i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return "transparent"
    return f"rgba({r},{g},{b},{alpha})"


def range_bar(frac: float, *, color: str = theme.ACCENT, height: int = 9):
    """Where a value sits inside a range: a track with a marker at ``frac``.

    ``frac`` is 0 at the low end and 1 at the high one. A value OUTSIDE the
    range is clamped for drawing (the marker sits on the edge it left, which
    is where a reader looks for it) — the caller says so in words; a marker
    drawn off the track would simply be invisible.

    A faint tick marks the middle, so a row can be read as low/centred/high
    without reading its two numbers.
    """
    from nicegui import ui

    pct = max(0.0, min(1.0, float(frac))) * 100.0
    track = ui.element("div").classes("w-full").style(
        f"position:relative;height:{height}px;background:{theme.WELL};"
        f"border:1px solid {theme.RULE_SOFT};overflow:hidden")
    with track:
        ui.element("div").style(
            f"position:absolute;left:calc(50% - 0.5px);top:0;width:1px;"
            f"height:100%;background:{theme.RULE_SOFT}")
        ui.element("div").style(
            f"position:absolute;left:calc({pct:.3f}% - 1.5px);top:0;"
            f"width:3px;height:100%;background:{color}")
    return track


def kv(key: str, value: str, *, color: str = "", tip: str = "",
       help: str = "", help_title: str = ""):
    """One key/value line in a work area (not the property grid).

    A long ``tip`` becomes the ``?`` (:data:`TIP_WORDS_ON_HOVER`), on the
    same rule a field row's does.
    """
    from nicegui import ui

    tip, help = _tip_that_shows(tip, help)
    with ui.row().classes("w-full items-center gap-2 no-wrap"):
        k = ui.label(key).classes("field-label").style(
            f"color:{theme.INK_MUTED};min-width:150px")
        v = ui.label(value).classes("readout")
        if color:
            v.style(f"color:{color}")
        if tip:
            k.tooltip(tip)
        if help:
            help_dot(help, title=help_title or key)
    return v


def readout(label: str, value: str = "—", unit: str = "", *, tip: str = "",
            color: str = "", help: str = "", help_title: str = ""):
    """Big monospace number with a small caption — the stage head-line.

    The caption carries the ``?`` when there is one, so the mark sits with
    the WORDS rather than beside the number. A long ``tip`` becomes one
    (:data:`TIP_WORDS_ON_HOVER`): a read-out's tooltip is usually the
    definition of the quantity, which is the thing a reader most often
    wants and least often finds.
    """
    from nicegui import ui

    tip, help = _tip_that_shows(tip, help)
    with ui.column().classes("gap-0").style(
            f"min-width:104px;border-left:2px solid {theme.RULE_SOFT};"
            "padding-left:8px") as col:
        if help:
            with ui.row().classes("items-center gap-1 no-wrap"):
                ui.label(label).classes("readout-label")
                help_dot(help, title=help_title or label)
        else:
            ui.label(label).classes("readout-label")
        with ui.row().classes("items-baseline gap-1 no-wrap"):
            val = ui.label(value).classes("readout-big")
            if color:
                val.style(f"color:{color}")
            if unit:
                ui.label(unit).classes("field-unit")
    if tip:
        col.tooltip(tip)
    val.column = col
    return val


#: decimals a numeric FIELD shows. The value behind it is not rounded: the
#: published operating points are exact IEEE expressions (the trim wing's
#: 652.8022140185119 N), and rounding the state would turn an untouched
#: mission into an edited one and send it to the solver. So the field shows
#: ``shown(v)`` and ``is_echo`` recognises that rounded number coming back
#: from the browser as "the user typed nothing".
FIELD_DIGITS = 4


def shown(value, nd: int = FIELD_DIGITS):
    """What a numeric field displays for ``value`` (never what it stores)."""
    try:
        return round(float(value), nd)
    except (TypeError, ValueError):
        return value


def is_echo(new, stored, nd: int = FIELD_DIGITS) -> bool:
    """True when ``new`` is just the rounded value we drew, coming back."""
    try:
        return float(new) == shown(stored, nd)
    except (TypeError, ValueError):
        return False


def _row_note(note: str, help: str, help_title: str):
    """The right-hand half of a field row: short caption, then the ``?``.

    Both are optional and both pin RIGHT, so a column of fields reads as a
    column of captions with a column of question marks beside it — the mark
    is findable without hunting along each row for it.
    """
    from nicegui import ui

    if note:
        ui.label(note).classes("field-note")
    if note or help:
        ui.space()
    if help:
        help_dot(help, title=help_title)


#: the longest a ``tip`` may be and still be worth leaving on hover. Past
#: it the tooltip is carrying an explanation, and an explanation with no
#: mark on screen is one only a reader who already knew goes looking for —
#: which is the whole reason ``help_dot`` exists. Six words is a unit or a
#: restatement of the label; twenty is an argument.
TIP_WORDS_ON_HOVER = 8


def _tip_that_shows(tip: str, help: str) -> tuple:
    """``(tip, help)`` with a long hover-only tip promoted to the ``?``.

    A field that states its own ``help`` as WELL keeps both — they are two
    different sentences by the author's choice, not by accident — but the
    promoted tip JOINS the popup rather than staying on hover beside it.
    Leaving it there would leave one of the two paragraphs in the channel
    this whole rule exists to empty.
    """
    if tip and len(str(tip).split()) > TIP_WORDS_ON_HOVER:
        return ("", "\n\n".join(t for t in (str(help), str(tip)) if t))
    return (tip, help)


def explain(el, text: str, *, title: str = ""):
    """Hover for a short note; a VISIBLE ``?`` for anything longer.

    The same rule :func:`_tip_that_shows` applies to a field row, for the
    controls that are not field rows — a switch, a button, a chart, a
    read-out. ``el`` is returned either way, so it drops into a chain.

    The ``?`` is drawn in whatever slot is open, which is the row ``el`` was
    just created in: call it where the control is built, not later.
    """
    if not text:
        return el
    if len(str(text).split()) <= TIP_WORDS_ON_HOVER:
        el.tooltip(str(text))
    else:
        help_dot(str(text), title=title)
    return el


def number_field(label: str, value, on_change, *, unit: str = "",
                 step: float = 1.0, width: str = "w-28", tip: str = "",
                 fmt: str | None = None, enabled: bool = True,
                 note: str = "", help: str = "", help_title: str = ""):
    """The standard dense numeric row: label · input · unit · note · ``?``.

    ``note`` is the short text that says what the field DOES (a handful of
    words, on the row itself); ``help`` is the paragraph behind a visible
    ``?`` for a field that needs one. ``tip`` predates both and is a
    hover-only tooltip on the label — prefer ``note``/``help``, which can
    be seen. A LONG ``tip`` becomes a ``?`` of its own
    (:data:`TIP_WORDS_ON_HOVER`), because a paragraph nothing on screen says
    is there is a paragraph nobody reads.
    """
    from nicegui import ui

    tip, help = _tip_that_shows(tip, help)
    with ui.row().classes("w-full items-center gap-2 no-wrap"):
        lab = ui.label(label).classes("field-label").style("min-width:150px")
        if tip:
            lab.tooltip(tip)
        # shrink-0: a long unit caption next to it would otherwise squeeze
        # the input to a few pixels and hide the value it is showing
        num = ui.number(value=value, step=step, on_change=on_change) \
            .props("outlined dense hide-bottom-space") \
            .classes(f"{width} shrink-0")
        if fmt:
            num.props(f'format="{fmt}"')
        if not enabled:
            num.disable()
        ui.label(unit).classes("field-unit")
        _row_note(note, help, help_title)
    return num


def select_field(label: str, options, value, on_change, *,
                 width: str = "w-56", tip: str = "", note: str = "",
                 help: str = "", help_title: str = ""):
    """Label + dense select on one line.

    Quasar's own floating label is NOT used: the shell pins field height at
    26px, and a stacked label inside that box overlaps the value.

    ``note`` and ``help`` are :func:`number_field`'s, and mean the same —
    including what a long ``tip`` turns into.
    """
    from nicegui import ui

    tip, help = _tip_that_shows(tip, help)
    with ui.row().classes("w-full items-center gap-2 no-wrap"):
        lab = ui.label(label).classes("field-label").style("min-width:150px")
        if tip:
            lab.tooltip(tip)
        sel = ui.select(options, value=value, on_change=on_change) \
            .props("outlined dense").classes(f"{width} shrink-0")
        _row_note(note, help, help_title)
    return sel


def text_field(label: str, value, on_change, *, width: str = "w-56",
               grow: bool = False, tip: str = ""):
    """Caption + dense TEXT input, side by side, on a row the CALLER owns.

    Quasar's own floating label is not used, for the reason spelled out in
    :func:`select_field`: this shell pins a field at 26 px, and a label
    floated into a box that short lands ON the value — the export card
    rendered "save to" over the first six characters of the path it was
    reporting, which is the one thing a field naming a save location may
    not do.

    Unlike :func:`number_field` and :func:`select_field` this does NOT open
    a row of its own: the pair it exists for (a folder and a file name) is
    one question asked on one line, and each half wants its own width.
    ``grow`` is for the half that should take the slack — a path is long and
    a stem is short.
    """
    from nicegui import ui

    lab = ui.label(label).classes("field-label shrink-0")
    if tip:
        lab.tooltip(tip)
    field = ui.input(value=value, on_change=on_change) \
        .props("outlined dense hide-bottom-space")
    field.classes("grow min-w-0" if grow else f"{width} shrink-0")
    return field


def hint_help(text: str, help: str, *, title: str = "", kind: str = ""):
    """A short :func:`hint` line with the long version behind a ``?``.

    The pattern this shell is being finalised INTO: eight words on screen
    saying what the thing does, and the paragraph that used to sit there in
    full one click away. Used for anything a field row cannot carry — a
    group of controls, a chart, a verdict.
    """
    from nicegui import ui

    with ui.row().classes("w-full items-center gap-1 no-wrap"):
        # split=False, and it is not an optimisation: hint() calls THIS
        # function for a long line, and a lead that is still long (the
        # no-boundary fallback appends an ellipsis, so it can be) would call
        # it straight back. The lead is already the short half by
        # construction.
        lab = hint(text, kind, split=False)
        help_dot(help, title=title)
    return lab


def toolbar_button(icon: str, tip: str, on_click, *, label: str = "",
                   primary: bool = False):
    """A tool-strip button.

    The colour is set through Quasar's OWN props rather than through the
    stylesheet: its ``.text-primary`` rule carries ``!important`` from
    inside an at-rule block, so a stylesheet override lost the cascade and
    painted the primary button blue text on a blue fill — an invisible Run
    button. The class here carries geometry only.
    """
    from nicegui import ui

    props = ("dense no-caps unelevated color=primary text-color=white"
             if primary else "dense no-caps flat color=grey-9")
    btn = ui.button(label, icon=icon, on_click=on_click).props(props) \
        .classes("tb-btn" + (" tb-primary" if primary else ""))
    btn.tooltip(tip)
    return btn


def toolbar_sep():
    from nicegui import ui

    return ui.element("div").classes("tb-sep")


# ------------------------------------------------------------------- tree
class Tree:
    """The simulation object tree.

    ``render(nodes)`` takes a FLAT, already-ordered list of node dicts:

        {"key", "label", "level", "icon", "state", "badge", "enabled",
         "children" (bool), "expanded" (bool), "tip"}

    The owner decides what is visible (i.e. filters out the children of a
    collapsed parent); this class only draws and reports clicks.
    """

    def __init__(self, on_select, on_toggle=None):
        from nicegui import ui

        self.on_select = on_select
        self.on_toggle = on_toggle
        self.selected: str | None = None
        self.container = ui.column().classes("w-full gap-0 py-1")

    def render(self, nodes: list[dict]):
        from nicegui import ui

        self.container.clear()
        with self.container:
            for nd in nodes:
                enabled = nd.get("enabled", True)
                cls = "tree-row w-full"
                if nd["key"] == self.selected:
                    cls += " tree-sel"
                if not enabled:
                    cls += " tree-locked"
                row = ui.element("div").classes(cls).style(
                    f"padding-left:{4 + 13 * int(nd.get('level', 0))}px")
                with row:
                    if nd.get("children"):
                        tw = ui.label("▾" if nd.get("expanded") else "▸") \
                            .classes("tree-twisty")
                        tw.on("click", lambda _, k=nd["key"]:
                              self._toggle(k))
                    else:
                        ui.label("").classes("tree-twisty")
                    icon, color = NODE_STATE_ICONS.get(
                        nd.get("state", "ready"),
                        NODE_STATE_ICONS["ready"])
                    ui.icon(nd.get("icon") or icon).classes("tree-icon") \
                        .style(f"color:{color}")
                    ui.label(nd["label"]).classes("tree-label")
                    if nd.get("badge"):
                        ui.label(nd["badge"]).classes("tree-badge")
                if nd.get("tip"):
                    row.tooltip(nd["tip"])
                row.on("click", lambda _, k=nd["key"]: self.on_select(k))

    def _toggle(self, key: str):
        if self.on_toggle is not None:
            self.on_toggle(key)


# ------------------------------------------------------------- properties
class PropertyGrid:
    """The properties table under the tree: what the selected node IS.

    ``set(rows)`` where a row is ``("group", title)`` for a section header
    or ``(key, value)`` / ``(key, value, colour)`` for a fact. Read-only by
    design: editing happens in the stage's own work area, and a value that
    could be typed in two places is a value that can disagree with itself.
    """

    def __init__(self):
        from nicegui import ui

        self.container = ui.column().classes("w-full gap-0")

    def set(self, rows: list[tuple]):
        from nicegui import ui

        self.container.clear()
        with self.container:
            if not rows:
                ui.label("nothing selected").classes("hint").style(
                    "padding:6px 8px")
                return
            for row in rows:
                if row[0] == "group":
                    ui.label(row[1]).classes("prop-group w-full")
                    continue
                key, value = row[0], row[1]
                color = row[2] if len(row) > 2 else ""
                with ui.element("div").classes("prop-row w-full"):
                    ui.label(str(key)).classes("prop-key").tooltip(str(key))
                    lab = ui.label(str(value)).classes("prop-val")
                    lab.tooltip(str(value))
                    if color:
                        lab.style(f"color:{color}")


# --------------------------------------------------------------- tab strip
class TabStrip:
    """The work-area tab row. The owner shows/hides the view containers."""

    def __init__(self, on_change):
        from nicegui import ui

        self.on_change = on_change
        self.active: str | None = None
        self.container = ui.element("div").classes("cae-tabstrip w-full")

    def render(self, tabs: list[tuple], active: str | None = None):
        """``tabs``: list of ``(key, label)`` or ``(key, label, badge)``."""
        from nicegui import ui

        if active is not None:
            self.active = active
        self.container.clear()
        with self.container:
            for tab in tabs:
                key, label = tab[0], tab[1]
                badge = tab[2] if len(tab) > 2 else ""
                cls = "cae-tab" + (" cae-tab-active"
                                   if key == self.active else "")
                el = ui.element("div").classes(cls)
                with el:
                    ui.label(label)
                    if badge:
                        ui.label(badge).classes("field-unit")
                el.on("click", lambda _, k=key: self.on_change(k))


# --------------------------------------------------------------- log/status
class OutputLog:
    """The bottom output window: one timestamped line per thing that happened.

    Capped at ``max_lines`` (oldest dropped) and auto-scrolled to the tail,
    so a long screening run cannot grow the DOM without bound.

    **Only the thread that built the log builds elements in it.** Creating a
    nicegui element writes ``binding.bindable_properties`` (a global
    WeakValueDictionary, 2499 entries in the assembled shell) while any
    concurrent deletion — the ``max_lines`` prune here, or ``_paint_shell``
    clearing and rebuilding the tree/tabs/properties on every dirty 0.5 s
    tick — iterates that same dict via ``list(...)``.  The two collide with
    ``RuntimeError: dictionary changed size during iteration``, losing the
    log line plus a stderr traceback, or aborting a repaint and leaving the
    chrome half-drawn until the next dirty tick.  The window was measured at
    79 µs per line without the prune and 428 µs with it, and the workers
    that log are real (``_auto_recommend`` in stage 3, the result writers in
    stage 4), so a line written off the loop thread is QUEUED with its own
    timestamp and turned into elements by :meth:`drain`, which the shell's
    heartbeat calls (``Ctx`` registers it as a poll).  Callers do not change:
    ``write`` decides for them.
    """

    LEVEL_CLASS = {"info": "log-info", "ok": "log-ok", "warn": "log-warn",
                   "error": "log-err"}

    def __init__(self, max_lines: int = 400):
        import threading

        from nicegui import ui

        self.max_lines = int(max_lines)
        self.scroll = ui.scroll_area().classes("w-full h-full cae-log") \
            .props("visible")
        with self.scroll:
            self.column = ui.column().classes("w-full gap-0")
        self.lines: list = []
        #: the thread that built these elements — the loop thread in the
        #: shell, the test's own thread under pytest.  Anything else is a
        #: worker and gets the queue.
        self._home = threading.get_ident()
        self._pending: list[tuple] = []
        self._lock = threading.Lock()

    def write(self, text: str, level: str = "info"):
        import threading

        stamp = time.strftime("%H:%M:%S")
        if threading.get_ident() != self._home:
            with self._lock:
                self._pending.append((stamp, str(text), level))
                # a burst cannot outgrow what the window would keep anyway
                while len(self._pending) > self.max_lines:
                    self._pending.pop(0)
            return
        self.drain()
        self._emit(stamp, str(text), level)

    def drain(self) -> int:
        """Turn every queued worker line into elements. Loop thread only.

        Returns the number of lines drawn, so a caller can tell "nothing was
        waiting" from "the log moved". Never raises: a drain that dies would
        take the heartbeat's other polls with it.
        """
        with self._lock:
            queued, self._pending = self._pending, []
        for stamp, text, level in queued:
            self._emit(stamp, text, level)
        return len(queued)

    def _emit(self, stamp: str, text: str, level: str):
        from nicegui import ui

        with self.column:
            with ui.row().classes("no-wrap gap-2 items-start log-line") as row:
                ui.label(stamp).classes("log-time")
                ui.label(str(text)).classes(
                    self.LEVEL_CLASS.get(level, "log-info"))
        self.lines.append(row)
        while len(self.lines) > self.max_lines:
            self.lines.pop(0).delete()
        self.scroll.scroll_to(percent=1.0)

    def clear(self):
        with self._lock:
            self._pending = []
        self.column.clear()
        self.lines = []


class StatusBar:
    """The bottom strip: current message, live cells, and a progress bar."""

    def __init__(self):
        from nicegui import ui

        with ui.element("div").classes("cae-status w-full"):
            self.dot = ui.element("div").style(
                f"width:7px;height:7px;border-radius:1px;"
                f"background:{theme.INK_FAINT}")
            self.msg = ui.label("Ready")
            ui.space()
            self.bar = ui.linear_progress(value=0.0, show_value=False,
                                          size="8px").classes("w-40")
            self.bar.set_visibility(False)
            self.cells: dict = {}
            self.cell_row = ui.row().classes("items-center gap-3 no-wrap")

    def message(self, text: str, kind: str = "idle"):
        color = {"idle": theme.INK_FAINT, "busy": theme.ACCENT,
                 "ok": theme.GOOD, "warn": theme.WARN,
                 "error": theme.BAD}.get(kind, theme.INK_FAINT)
        self.dot.style(f"width:7px;height:7px;border-radius:1px;"
                       f"background:{color}")
        self.msg.set_text(text)

    def progress(self, frac: float | None):
        if frac is None:
            self.bar.set_visibility(False)
            return
        self.bar.set_visibility(True)
        self.bar.set_value(max(0.0, min(1.0, float(frac))))

    def set_cells(self, cells: list[tuple]):
        from nicegui import ui

        self.cell_row.clear()
        with self.cell_row:
            for name, value in cells:
                ui.label(f"{name} {value}").classes("status-cell")
