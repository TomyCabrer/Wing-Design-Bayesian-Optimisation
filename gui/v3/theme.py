"""Design tokens + stylesheet for the V3 shell.

Import-safe: no nicegui / aerobo imports at module scope. ``head_html()``
returns the stylesheet; ``apply()`` (lazy nicegui import) installs light
mode, Quasar brand colours and that stylesheet.

Look (chosen by the user, session 25): a LIGHT CAE DESKTOP — the chrome of
a simulation package (Star-CCM+, ANSYS, Abaqus) rather than of a web app.
The rules that produce it, and that every widget here follows:

* structure comes from HAIRLINES and TITLE BARS, never from shadow, glow,
  gradient or a rounded card. Radius is 2px, once, on the outermost frame;
* the canvas is a light neutral grey and the WORK surfaces are white, so a
  plot, a table and a coordinate view all sit on the same paper;
* one accent (a desk blue) marks SELECTION and the primary action; every
  other colour is state (ok / caution / violated);
* the type is the operating system's own UI face at 12px, with a monospace
  face for every NUMBER — no downloaded font, so the shell also renders the
  same with no network;
* nothing is centred, nothing is oversized, and no control is decorative:
  the density is the point, because a stage of this pipeline has a lot of
  small facts to show at once.
"""

# ------------------------------------------------------------------ tokens
CANVAS = "#c8ccd3"        # desktop grey behind the frames
CHROME = "#e6e9ed"        # menu bar / tool bar / tab strip
PANEL = "#f1f3f5"         # pane background (tree, properties, forms)
WELL = "#ffffff"          # work surfaces: plots, tables, log
RULE = "#adb4bf"          # hairline (frame edges)
RULE_SOFT = "#d2d7de"     # hairline (inside a pane)
HEADER = "#dfe3e8"        # pane title bar / table header

INK = "#181c22"
INK_MUTED = "#4d5763"
INK_FAINT = "#78828f"

ACCENT = "#1d5fa4"        # desk blue: selection + primary action
ACCENT_FILL = "#cfe0f3"   # selected tree row / active tab tint
GOOD = "#1a7a46"          # feasible / converged / complete
WARN = "#94670a"          # caution / bound-riding / cancelled
BAD = "#a32a25"           # violated / error

#: categorical series colours for multi-trace figures, ordered so the first
#: series is the accent and no two adjacent entries share a hue family
SERIES = ("#1d5fa4", "#1a7a46", "#94670a", "#a32a25", "#5b4a9e", "#0d7a86")

GRID = "rgba(24,28,34,0.12)"        # plot gridlines
BAND = "rgba(29,95,164,0.16)"       # seed band / filled planform

SANS = ('-apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, '
        'Roboto, "Helvetica Neue", sans-serif')
MONO = ('ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, '
        '"Liberation Mono", monospace')

CSS = f"""
:root{{--canvas:{CANVAS};--chrome:{CHROME};--panel:{PANEL};--well:{WELL};
--rule:{RULE};--rule-soft:{RULE_SOFT};--header:{HEADER};
--ink:{INK};--ink-muted:{INK_MUTED};--ink-faint:{INK_FAINT};
--accent:{ACCENT};--accent-fill:{ACCENT_FILL};
--good:{GOOD};--warn:{WARN};--bad:{BAD};}}

html,body,body.body--light{{background:var(--canvas)!important;
color:var(--ink);font-family:{SANS};font-size:12px;line-height:1.45;
-webkit-font-smoothing:antialiased;margin:0;overflow:hidden;}}
.mono,code,pre,kbd,.num{{font-family:{MONO};
font-variant-numeric:tabular-nums;}}
.q-page,.nicegui-content{{padding:0!important;gap:0!important;}}

/* ------------------------------------------------------------ menu bar */
.cae-menubar{{display:flex;align-items:center;height:26px;
background:var(--chrome);border-bottom:1px solid var(--rule);
padding:0 4px;gap:1px;user-select:none;}}
.cae-menu{{padding:2px 9px;font-size:12px;color:var(--ink);
border-radius:2px;cursor:default;}}
.cae-menu:hover{{background:var(--accent-fill);}}
.cae-appname{{font-size:11px;font-weight:600;color:var(--ink-muted);
padding:0 10px 0 6px;letter-spacing:.02em;}}

/* ------------------------------------------------------------ tool bar */
.cae-toolbar{{display:flex;align-items:center;height:32px;
background:var(--chrome);border-bottom:1px solid var(--rule);
padding:0 6px;gap:3px;}}
/* Geometry only. The tool buttons' COLOUR comes from Quasar's own props
   (see widgets.toolbar_button): its .text-primary rule is !important from
   inside an at-rule block and beats a stylesheet override, which is how the
   primary button ended up blue-on-blue and invisible. */
.q-btn.tb-btn{{min-width:26px!important;height:24px!important;
padding:0 6px!important;border:1px solid transparent;border-radius:2px;
font-size:11.5px!important;font-weight:500;text-transform:none;}}
.q-btn.tb-btn:hover{{border-color:var(--rule);}}
.q-btn.tb-btn[disabled]{{opacity:.38;}}
.q-btn.tb-btn .q-icon{{font-size:17px;}}
.q-btn.tb-primary{{border-color:#164c85;}}
.tb-sep{{width:1px;height:18px;background:var(--rule);margin:0 4px;}}
.tb-crumb{{font-size:11px;color:var(--ink-muted);}}
.tb-crumb b{{color:var(--ink);font-weight:600;}}

/* ---------------------------------------------------------- pane frame */
.pane{{background:var(--panel);border:1px solid var(--rule);
border-radius:2px;display:flex;flex-direction:column;overflow:hidden;}}
.pane-title{{display:flex;align-items:center;gap:6px;height:22px;
padding:0 7px;background:var(--header);border-bottom:1px solid var(--rule);
font-size:11px;font-weight:600;color:var(--ink-muted);flex:0 0 auto;
letter-spacing:.01em;}}
.pane-body{{flex:1 1 auto;overflow:auto;background:var(--panel);}}
.pane-body.white{{background:var(--well);}}
.pane-pad{{padding:8px 9px;}}

/* ---------------------------------------------------------------- tree */
.tree-row{{display:flex;align-items:center;gap:5px;height:21px;
padding:0 6px 0 2px;font-size:12px;color:var(--ink);cursor:default;
white-space:nowrap;}}
.tree-row:hover{{background:#e3e8ee;}}
.tree-sel{{background:var(--accent-fill);}}
.tree-sel .tree-label{{font-weight:600;}}
.tree-locked{{color:var(--ink-faint);}}
.tree-twisty{{width:12px;text-align:center;font-size:9px;
color:var(--ink-muted);cursor:pointer;}}
.tree-icon{{font-size:14px!important;}}
.tree-badge{{font-family:{MONO};font-size:9.5px;color:var(--ink-faint);
margin-left:auto;padding-left:8px;}}

/* --------------------------------------------------------- properties */
.prop-row{{display:grid;grid-template-columns:44% 56%;align-items:center;
border-bottom:1px solid var(--rule-soft);min-height:20px;}}
.prop-key{{padding:2px 6px;font-size:11px;color:var(--ink-muted);
border-right:1px solid var(--rule-soft);overflow:hidden;
text-overflow:ellipsis;white-space:nowrap;}}
.prop-val{{padding:2px 6px;font-family:{MONO};font-size:11px;
color:var(--ink);overflow:hidden;text-overflow:ellipsis;
white-space:nowrap;}}
.prop-group{{padding:3px 6px;background:var(--header);font-size:10.5px;
font-weight:600;color:var(--ink-muted);border-bottom:1px solid var(--rule);}}

/* --------------------------------------------------------------- tabs */
.cae-tabstrip{{display:flex;align-items:flex-end;height:25px;gap:2px;
background:var(--chrome);border-bottom:1px solid var(--rule);
padding:3px 5px 0 5px;overflow-x:auto;}}
.cae-tab{{display:flex;align-items:center;gap:5px;height:22px;
padding:0 11px;font-size:11.5px;color:var(--ink-muted);cursor:default;
background:#dde1e6;border:1px solid var(--rule);border-bottom:none;
border-radius:2px 2px 0 0;white-space:nowrap;}}
.cae-tab:hover{{background:#e9edf1;}}
.cae-tab-active{{background:var(--well);color:var(--ink);font-weight:600;
margin-bottom:-1px;padding-bottom:1px;}}

/* ---------------------------------------------------------- work area */
.work{{background:var(--well);flex:1 1 auto;overflow:auto;}}
.work-pad{{padding:10px 12px;}}
.grouping{{border:1px solid var(--rule-soft);border-radius:2px;
background:var(--well);}}
.grouping>legend,.group-title{{font-size:11px;font-weight:600;
color:var(--ink-muted);padding:3px 8px;background:var(--panel);
border-bottom:1px solid var(--rule-soft);}}
.group-pad{{padding:8px 9px;}}
.hint{{font-size:11px;color:var(--ink-muted);line-height:1.5;}}
.hint-warn{{color:var(--warn);}}
.hint-bad{{color:var(--bad);}}
.hint-ok{{color:var(--good);}}
.field-label{{font-size:11.5px;color:var(--ink);}}
.field-unit{{font-family:{MONO};font-size:10.5px;color:var(--ink-faint);}}
.field-note{{font-size:10.5px;color:var(--ink-muted);white-space:nowrap;}}
.help-dot{{width:14px;height:14px;min-width:14px;border-radius:7px;
border:1px solid var(--rule);color:var(--ink-muted);background:var(--panel);
font-size:10px;line-height:12px;font-weight:600;text-align:center;
cursor:pointer;user-select:none;position:relative;
display:flex;align-items:center;justify-content:center;}}
.help-dot:hover{{border-color:var(--accent);color:var(--accent);
background:var(--accent-fill);}}
.help-pop .hint{{font-size:11.5px;}}
.help-pop-title{{font-size:11.5px;font-weight:600;color:var(--ink);}}
.readout{{font-family:{MONO};font-size:12px;color:var(--ink);}}
/* overflow-wrap: a read-out usually holds a short number, but the export
   card's holds an absolute PATH — and a path is one long token. Without
   this it breaks only where the folder names happen to contain hyphens
   (the repo's own "urop-bo-aero" hid the bug), and a folder without one
   ran the file's own path out past the right edge of the card. */
.readout-big{{font-family:{MONO};font-size:19px;color:var(--ink);
line-height:1.15;overflow-wrap:anywhere;}}
.readout-label{{font-size:10.5px;color:var(--ink-faint);}}
.tag{{font-size:10px;font-weight:600;padding:0 5px;border-radius:2px;
border:1px solid;white-space:nowrap;line-height:15px;display:inline-block;}}
.sect-head{{font-size:12px;font-weight:600;color:var(--ink);}}

/* ----------------------------------------------------------- log/status */
.cae-log{{background:var(--well);font-family:{MONO};font-size:11px;
line-height:1.55;padding:4px 8px;}}
.log-line{{white-space:pre-wrap;color:var(--ink);}}
.log-time{{color:var(--ink-faint);}}
.log-info{{color:var(--ink);}}
.log-ok{{color:var(--good);}}
.log-warn{{color:var(--warn);}}
.log-err{{color:var(--bad);}}
.cae-status{{display:flex;align-items:center;gap:8px;height:23px;
background:var(--chrome);border-top:1px solid var(--rule);
padding:0 8px;font-size:11px;color:var(--ink-muted);
overflow:hidden;white-space:nowrap;flex:0 0 auto;}}
.status-cell{{border-left:1px solid var(--rule);padding-left:8px;
font-family:{MONO};overflow:hidden;text-overflow:ellipsis;
max-width:34ch;}}

/* ------------------------------------------------------ quasar overrides */
/* Dense fields are 26px tall in this shell, so Quasar's own input metrics
   (line-height 28px + 6px of vertical padding = a 40px content box) have to
   be neutralised as well — otherwise the digits are pushed out of the box
   and the value reads as its own top half. Height, padding and line-height
   are set together for that reason; changing one alone re-opens the bug. */
.q-field--outlined .q-field__control{{border-radius:2px;background:#fff;
min-height:26px;height:26px;font-size:11.5px;padding:0 8px;}}
.q-field--outlined .q-field__control:before{{border-color:#9aa2ae;}}
.q-field--dense .q-field__control{{height:26px;min-height:26px;}}
.q-field__control-container{{padding-top:0!important;height:26px;}}
.q-field__native,.q-field__input{{font-family:{MONO};font-size:11.5px;
color:var(--ink);padding:0!important;line-height:24px;height:24px;
min-height:24px;}}
.q-field--dense .q-field__native,.q-field--dense .q-field__input{{
padding:0!important;}}
/* Chrome's number spinners sit ON the value in a box this size */
input[type=number]::-webkit-inner-spin-button,
input[type=number]::-webkit-outer-spin-button{{-webkit-appearance:none;
appearance:none;margin:0;}}
input[type=number]{{-moz-appearance:textfield;}}
.q-field__label{{font-size:11px;color:var(--ink-muted);}}
.q-field__marginal{{height:26px;min-height:26px;color:var(--ink-faint);}}
.q-field--dense .q-field__append,.q-field--dense .q-field__prepend{{
height:26px;}}
.q-btn{{text-transform:none;border-radius:2px;font-weight:500;
font-size:11.5px;}}
.q-btn--outline .q-btn__content{{color:var(--ink);}}
.q-checkbox__label,.q-radio__label,.q-toggle__label{{font-size:11.5px;}}
.q-table__container{{border:1px solid var(--rule);border-radius:2px;
box-shadow:none!important;background:var(--well);}}
.q-table thead th{{background:var(--header);color:var(--ink-muted);
font-size:11px;font-weight:600;height:24px;
border-bottom:1px solid var(--rule);position:sticky;top:0;z-index:1;}}
.q-table tbody td{{font-family:{MONO};font-size:11px;height:22px;
padding:0 8px;border-bottom:1px solid var(--rule-soft);}}
.q-table tbody tr:nth-child(even){{background:#f7f9fa;}}
.q-table tbody tr:hover{{background:var(--accent-fill);}}
.q-table tbody tr.selected{{background:var(--accent-fill);}}
.q-separator{{background:var(--rule-soft)!important;}}
/* Expansions are DISCLOSURES here, not cards: an advanced escape hatch
   should read as one quiet line until it is opened, so the default 48px
   Quasar item and its indented body are pulled back to this shell's
   density. */
.q-expansion-item .q-item{{min-height:22px;padding:0 2px;
font-size:11.5px;color:var(--ink-muted);}}
.q-expansion-item .q-item__section--side{{padding-right:4px;min-width:0;}}
.q-expansion-item__content{{padding:4px 0 2px 2px;
border-left:2px solid var(--rule-soft);margin-left:3px;}}
.q-expansion-item__content>.q-card{{box-shadow:none;background:transparent;}}
.q-menu{{background:#fff;border:1px solid var(--rule);border-radius:2px;
box-shadow:0 6px 18px rgba(20,26,34,.18)!important;}}
.q-item{{min-height:26px;font-size:12px;padding:2px 12px;}}
.q-tooltip{{background:#2b3138!important;color:#fff!important;
font-size:11px;border-radius:2px;}}
.q-linear-progress{{border-radius:0;}}
.q-slider__track-container{{border-radius:0;}}
.q-notification{{border-radius:2px;font-size:12px;}}
::-webkit-scrollbar{{width:12px;height:12px;}}
::-webkit-scrollbar-track{{background:#e9ecef;}}
::-webkit-scrollbar-thumb{{background:#bcc3cc;border:3px solid #e9ecef;
border-radius:6px;}}
::-webkit-scrollbar-thumb:hover{{background:#a4acb7;}}
a{{color:var(--accent);}}
"""


def head_html() -> str:
    return "<style>" + CSS + "</style>"


def apply():
    """Install light mode + brand colours + stylesheet (call inside a page)."""
    from nicegui import ui

    ui.dark_mode(False)
    ui.colors(primary=ACCENT, secondary=INK_MUTED, positive=GOOD,
              negative=BAD, warning=WARN, dark=PANEL, dark_page=CANVAS)
    ui.add_head_html(head_html())
