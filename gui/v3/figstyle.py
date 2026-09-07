"""Plotly restyler for the V3 (light CAE) shell.

V1's figure builders stay the single source of figure CONTENT — V3 wraps
them: ``figstyle.plot(v1.fig_planform(...))``.  The restyler overrides V1's
``_base_layout`` explicitly (a template default would lose to it) and remaps
V1's hard-coded sky/emerald/red/amber traces onto the desk palette, because
a figure that keeps its own colours reads as a foreign object pasted into
the frame.

Everything here draws on WHITE with dark ink, so the on-screen figure and
the PNG the modebar camera saves are the same picture — a light shell has
no export/display split to bridge (the V2 dark shell needed one).

Import-safe: plotly at module scope, nicegui lazily.
"""

from __future__ import annotations

import plotly.graph_objects as go

from . import theme

_MONO = theme.MONO
_SANS = theme.SANS

#: V1 figure palette (gui.nice_app) -> V3 desk palette
_COLORS = {
    "#0ea5e9": theme.ACCENT,          # V1 ACCENT (sky)   -> desk blue
    "#10b981": theme.GOOD,            # V1 GOOD           -> green
    "#ef4444": theme.BAD,             # V1 BAD            -> red
    "#f59e0b": theme.WARN,            # V1 WARN           -> ochre
    "#94a3b8": theme.INK_MUTED,       # V1 MUTED (axes)   -> ink
    "rgba(148,163,184,0.18)": theme.GRID,
    "rgba(14,165,233,0.18)": theme.BAND,
    "rgba(240,160,60,0.28)": "rgba(148,103,10,0.22)",   # elevator hatch
}

_3D_TYPES = {"scatter3d", "surface", "mesh3d", "cone", "volume"}

#: V1 lofts its 3-D skins with Viridis/Blues; on white paper Viridis' high
#: end (fluorescent yellow) disappears. This ramp keeps the same monotone
#: lightness ordering in inks that survive on a white pane.
SURFACE_SCALE = ((0.00, "#16305c"), (0.30, "#1d5fa4"), (0.58, "#2c8fb5"),
                 (0.80, "#3aa37a"), (1.00, "#8fae3c"))
_SWAPPABLE_SCALES = {"viridis", "blues", "cividis"}


def _is_named_ramp(cs) -> bool:
    """True for a real colour RAMP, false for a solid single-colour scale.

    Plotly expands ``colorscale="Viridis"`` into its stop list as soon as
    the figure is built, so the string rarely survives to be matched; a ramp
    is instead recognised by having more than two DISTINCT stop colours.
    V1's one deliberate solid scale (``[[0, WARN], [1, WARN]]``, a highlight)
    stays untouched by that test.
    """
    if isinstance(cs, str):
        return cs.lower() in _SWAPPABLE_SCALES
    if not cs:
        return False
    try:
        colors = {str(stop[1]) for stop in cs}
    except (TypeError, IndexError):
        return False
    return len(colors) > 2


def _has_3d(fig: go.Figure) -> bool:
    """True when the figure draws into a 3-D scene (``layout.scene`` is
    materialised lazily by plotly, so the trace types are the honest test)."""
    return any(getattr(tr, "type", "") in _3D_TYPES for tr in fig.data)


def _swap(value, mapping: dict):
    if isinstance(value, str):
        return mapping.get(value.lower(), mapping.get(value, value))
    return value


def recolor(fig: go.Figure, mapping: dict = None) -> go.Figure:
    """Remap every hard-coded trace/shape/annotation colour in ``fig``."""
    mapping = _COLORS if mapping is None else mapping
    for tr in fig.data:
        for attr in ("line", "marker", "textfont"):
            obj = getattr(tr, attr, None)
            if obj is not None and getattr(obj, "color", None) is not None:
                obj.color = _swap(obj.color, mapping)
        mk = getattr(tr, "marker", None)
        mk_line = getattr(mk, "line", None) if mk is not None else None
        if mk_line is not None and getattr(mk_line, "color", None) is not None:
            mk_line.color = _swap(mk_line.color, mapping)
        if getattr(tr, "fillcolor", None) is not None:
            tr.fillcolor = _swap(tr.fillcolor, mapping)
    for shp in fig.layout.shapes or ():
        if shp.line is not None and shp.line.color is not None:
            shp.line.color = _swap(shp.line.color, mapping)
        if getattr(shp, "fillcolor", None) is not None:
            shp.fillcolor = _swap(shp.fillcolor, mapping)
    for ann in fig.layout.annotations or ():
        if ann.font is not None and ann.font.color is not None:
            ann.font.color = _swap(ann.font.color, mapping)
    return fig


def plot(fig: go.Figure, h: int | None = None, *,
         keep_title: bool = False) -> go.Figure:
    """Mutate ``fig`` into the V3 look and return it.

    Pane titles carry the caption in this shell, so chart titles are
    stripped by default (a title inside the frame would repeat the tab).
    """
    recolor(fig)
    fig.update_layout(
        paper_bgcolor=theme.WELL,
        plot_bgcolor=theme.WELL,
        colorway=list(theme.SERIES),
        font=dict(family=_SANS, size=11, color=theme.INK),
        hoverlabel=dict(font=dict(family=_MONO, size=11), bgcolor="#2b3138",
                        bordercolor="#2b3138"),
        margin=dict(l=54, r=14, t=(28 if keep_title else 10), b=40),
        legend=dict(bgcolor="rgba(255,255,255,0.85)",
                    bordercolor=theme.RULE_SOFT, borderwidth=1,
                    font=dict(family=_SANS, size=10.5,
                              color=theme.INK_MUTED)),
    )
    if not keep_title:
        fig.update_layout(title=None)
    if h is not None:
        fig.update_layout(height=h)
    axis = dict(gridcolor=theme.GRID, zerolinecolor="rgba(24,28,34,0.28)",
                linecolor=theme.INK_MUTED,
                tickfont=dict(family=_MONO, size=10, color=theme.INK_MUTED),
                title_font=dict(family=_SANS, size=11, color=theme.INK))
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    if _has_3d(fig):
        for tr in fig.data:
            if _is_named_ramp(getattr(tr, "colorscale", None)):
                tr.colorscale = [list(p) for p in SURFACE_SCALE]
            cb = getattr(tr, "colorbar", None)
            if cb is not None and getattr(tr, "showscale", None):
                cb.outlinewidth = 0
                cb.tickfont = dict(family=_MONO, size=9,
                                   color=theme.INK_MUTED)
                cb.title = dict(text=(cb.title.text if cb.title else None),
                                font=dict(family=_SANS, size=10,
                                          color=theme.INK_MUTED))
        pane = dict(backgroundcolor=theme.WELL, gridcolor="#d8dde3",
                    zerolinecolor="#b6bec8", showbackground=True,
                    showspikes=False, color=theme.INK_MUTED,
                    title_font=dict(family=_SANS, size=11, color=theme.INK),
                    tickfont=dict(family=_MONO, size=9,
                                  color=theme.INK_MUTED))
        fig.update_layout(scene=dict(xaxis=pane, yaxis=pane, zaxis=pane))
    # V1 annotates in colours picked for a dark canvas; anything still that
    # light after the remap would vanish on white
    for ann in fig.layout.annotations or ():
        if ann.font and ann.font.color in ("#94a3b8", "#cbd5e1", "#e2e8f0"):
            ann.font.color = theme.INK_MUTED
    return fig


def export_config(filename: str) -> dict:
    """Plotly config: the modebar camera saves a 3x PNG named ``filename``."""
    return {
        "displaylogo": False,
        "toImageButtonOptions": {"format": "png", "filename": filename,
                                 "scale": 3},
    }


def empty(msg: str, h: int = 280) -> go.Figure:
    """Placeholder figure for a view with nothing to draw yet."""
    fig = go.Figure()
    fig.add_annotation(text=msg or "—", showarrow=False,
                       font=dict(family=_SANS, size=12,
                                 color=theme.INK_FAINT))
    fig.update_layout(paper_bgcolor=theme.WELL, plot_bgcolor=theme.WELL,
                      height=h, margin=dict(l=8, r=8, t=8, b=8))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def payload(fig: go.Figure, name: str, h: int | None = None) -> dict:
    """The restyled figure as the declarative dict ``ui.plotly`` renders."""
    out = plot(fig, h).to_plotly_json()
    out["config"] = export_config(name.replace(" ", "_"))
    return out


def show(fig: go.Figure, name: str, h: int | None = None):
    """Render a restyled figure with the export-friendly modebar config."""
    from nicegui import ui

    return ui.plotly(payload(fig, name, h)).classes("w-full")


def update(element, fig: go.Figure, name: str, h: int | None = None):
    """Redraw a pane ALREADY ON SCREEN, in place. Returns the element.

    A figure redrawn by clearing its container is a different DOM node: the
    pane is empty until the client has finished drawing the new one, so the
    scrolling work area's content briefly shrinks and the browser clamps its
    scroll position — which on a view repainted twice a second threw the
    page back to the top twice a second, exactly while the user was trying
    to read the live trace. Updating the payload of the element that is
    already there swaps the numbers without the page ever changing height.

    ``None`` is accepted (the pane has not been built yet, or belongs to a
    previous build of the view) and does nothing.
    """
    if element is None:
        return None
    element.update_figure(payload(fig, name, h))
    return element
