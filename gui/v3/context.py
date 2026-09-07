"""The shared context every V3 stage's ``build(ctx)`` receives.

One object carries the session, the run manager and the four shell widgets
(tree, properties, output log, status bar), plus the two things stages do
to each other:

    ctx.select(stage, view)   move the shell to a stage/view
    ctx.render(stage, view)   ask a built view to redraw from state
    ctx.refresh(*skip)        redraw the chrome AND every view that quotes
                              the design vector (``on_derived``), except the
                              one the control being used lives in

Cross-stage actions are registered by the stage that owns the behaviour and
called by name, so a stage that failed to build cannot take another one
down (``act`` on an unknown name is a no-op):

    "launch"        wing.py      () -> None     launch the configured run
    "cancel"        wing.py      () -> None     cancel the running queue
    "set_result"    results.py   (rd) -> None   show a finished run record
    "use_section"   airfoil.py   () -> None     re-apply the section link
"""

from __future__ import annotations

from typing import Any, Callable


class Ctx:
    def __init__(self, session: dict, manager):
        self.S = session
        self.manager = manager                 # gui.nice_app.RunManager
        self.tree = None                       # widgets.Tree
        self.props = None                      # widgets.PropertyGrid
        self.tabs = None                       # widgets.TabStrip
        self.output = None                     # widgets.OutputLog
        self.statusbar = None                  # widgets.StatusBar
        self.views: dict[tuple, Any] = {}      # (stage, view) -> container
        self.renderers: dict[tuple, Callable] = {}
        self.actions: dict[str, Callable] = {}
        #: zero-arg callables run on the shell's 0.5 s heartbeat; each
        #: returns True when it changed state, so the shell repaints once
        #: per tick instead of once per worker.
        #: The first one is the output log's drain: ``ctx.log`` is called
        #: straight from worker threads (stage 3's ``_auto_recommend``,
        #: stage 4's result writers), and building nicegui elements off the
        #: loop thread races the same global binding table that every
        #: repaint deletes from. Registered here rather than at the call
        #: sites so EVERY caller is safe without knowing it.
        self.polls: list[Callable] = [self._drain_log]
        #: (stage, view) pairs whose TEXT quotes the design vector — the
        #: dimension the search flies, the budget and wall clock that follow
        #: from it, the Reynolds interval the section is screened over, the
        #: family sentence. They are drawn by one stage and moved by ANOTHER
        #: stage's controls, and every setter that moved one used to hand-pick
        #: the two or three ``_render_*`` helpers it happened to remember —
        #: so pinning taper left stage 1's Search tab and stage 3's Solver
        #: card quoting "29 evaluations / 6 design variables" for a run that
        #: flies 26 in 5-D, and stage 2's screen quoting the un-pinned Re
        #: interval. Registered by the stage that DRAWS them and repainted by
        #: :meth:`refresh`, so a setter written tomorrow is right by default.
        self.derived: list[tuple] = []
        #: (stage, view) -> (signature callable, its value at the last
        #: repaint). A derived view whose whole derived content is one cheap
        #: expression says so, and is then skipped when that expression has
        #: not moved: repainting all eight of them costs ~206 ms, and
        #: ``_set_bound`` runs on every KEYSTROKE. A signature that misses a
        #: dependency is caught by tests/test_v3_act_render_fixpoint.py,
        #: which renders unconditionally and compares.
        self._derived_sig: dict = {}
        #: (stage, view) pairs whose paint is OWED — they were asked to
        #: redraw while off screen, so the redraw was deferred to the moment
        #: they are shown. See :meth:`render_when_shown` for the measurement
        #: that made this necessary.
        self._owed: set = set()
        self._repainting = False
        self._select: Callable = lambda stage, view=None: None
        self._refresh: Callable = lambda: None

    # ---- cross-stage actions
    def register(self, name: str, fn: Callable):
        self.actions[name] = fn

    def act(self, name: str, *a, **kw) -> Any:
        fn = self.actions.get(name)
        return None if fn is None else fn(*a, **kw)

    # ---- shell services (app.py installs the real implementations)
    def select(self, stage: str, view: str | None = None):
        self._select(stage, view)

    def refresh(self, *skip: tuple):
        """Redraw the chrome, then every view that quotes the design vector.

        ``skip`` names the view the control being used lives in. Rebuilding
        the container a ``ui.number`` sits in from that number's own
        ``on_change`` destroys the input mid-number and swallows the rest of
        it — the focus trap this shell keeps closing — so a handler that is
        driven by a typed field passes its own view here and repaints the
        derived read-outs INSIDE that view itself, the way it already did.

        The 0.5 s heartbeat does NOT come through here (``app._tick`` calls
        ``_paint_shell`` directly): a run in flight marks the shell dirty
        twice a second, and rebuilding five views at that rate is the
        "a repaint is not a rebuild" failure, not a fix for staleness.

        It is not free: measured on a fresh air session, the eight declared
        views cost ~145 ms per action on top of what the setter already paid
        (``set_row_fixed`` 56 -> 201 ms, ``set_bound`` 38 -> 155 ms), and the
        action-heavy V3 test files run 1.94x longer (114.7 -> 222.2 s over
        six of them). Two of the eight carry a signature for that reason.
        """
        self._refresh()
        self.repaint_derived(*skip)

    def on_derived(self, stage: str, view: str, sig: Callable | None = None):
        """Declare that this view's text follows the design vector.

        ``sig``, where a stage can name one, is a cheap expression that the
        whole of this view's derived content is a function of; the repaint is
        then skipped while it stands still. Give one only where the answer is
        obvious — a wrong signature is silent staleness again, and the only
        thing standing over it is the fixpoint test.
        """
        if (stage, view) not in self.derived:
            self.derived.append((stage, view))
        if sig is not None:
            self._derived_sig[(stage, view)] = (sig, object())

    def visible(self, stage: str, view: str) -> bool:
        """Is this the one view the work area is actually showing?

        Exactly one (stage, view) pair is on screen at a time — ``app._show``
        makes every other container invisible — so this is the whole of "can
        the user see what I am about to draw".
        """
        ui = self.S.get("ui") or {}
        if ui.get("selected") != stage:
            return False
        return ((ui.get("tab") or {}).get(stage)) == view

    def render_when_shown(self, stage: str, view: str | None = None):
        """Redraw a view IF it is on screen; otherwise owe it the paint.

        An invisible container costs exactly as much to rebuild as a visible
        one: nicegui ships every element it creates down the socket whether
        or not the page will show it. Measured on this shell, in a browser,
        with one design in it:

            action                          bytes pushed to the client
            toggle "add a tail" (stage 1)        968 KB
            accept the mission                   627 KB
            a run finishing (stage 3 -> 4)      1.93 MB

        ...of which ~277 KB per repaint is the ten ``on_derived`` views and
        ~250 KB is the Results geometry — NONE of them on screen at the
        time. Chrome parses that in a few hundred ms; the NATIVE window is a
        WKWebView, and there it is the "it freezes for a while after the run
        finishes" this method exists to remove.

        The paint is not skipped, it is DEFERRED: ``app.select`` renders the
        view it moves to, so the owed paint lands the moment the view is
        looked at, and a view that is never looked at is never paid for.
        """
        for st, vw in list(self.renderers):
            if st != stage or (view is not None and vw != view):
                continue
            if self.visible(st, vw):
                self.render(st, vw)
            else:
                self._owed.add((st, vw))

    def pay_owed(self):
        """Draw every deferred paint now, on screen or not.

        The shell never needs this — ``app.select`` pays each view as it is
        opened — but anything that INSPECTS a view without opening it does:
        tests/test_v3_act_render_fixpoint.py asserts that an action leaves no
        view stale, and with deferral the honest form of that claim is "no
        view is stale once what is owed has been paid".
        """
        # a renderer may owe another view on its way past (a card that calls
        # back into refresh), so this drains rather than sweeps once —
        # bounded, because a view that re-owes itself must not spin here
        for _ in range(3):
            owed = sorted(self._owed)
            if not owed:
                break
            self._owed.clear()
            for stage, view in owed:
                self.render(stage, view)
        self._owed.clear()

    def repaint_derived(self, *skip: tuple):
        """Redraw every declared derived view except the ones named.

        Off-screen views are owed rather than drawn (:meth:`render_when_shown`).
        """
        if self._repainting:
            # a derived view whose own render calls back into refresh would
            # otherwise recurse through the whole list once per view
            return
        self._repainting = True
        try:
            for stage, view in list(self.derived):
                if (stage, view) in skip or stage in skip:
                    continue
                pair = self._derived_sig.get((stage, view))
                if pair is not None:
                    try:
                        now = pair[0]()
                    except Exception:       # a signature may never decide
                        now = object()      # whether the view is drawn
                    if now == pair[1]:
                        continue
                    self._derived_sig[(stage, view)] = (pair[0], now)
                self.render_when_shown(stage, view)
        finally:
            self._repainting = False

    def log(self, text: str, level: str = "info"):
        if self.output is not None:
            self.output.write(text, level)

    def _drain_log(self) -> bool:
        """Draw the lines the worker threads queued. Runs on the heartbeat.

        Returns False on purpose: new log lines are their own elements and
        need no chrome repaint, and a drain that forced one would rebuild
        the tree, tab strip and property grid on every worker line.
        """
        drain = getattr(self.output, "drain", None)
        if drain is not None:
            try:
                drain()
            except Exception:                 # a poll must never kill the tick
                import sys
                import traceback
                print(f"[v3] output log drain failed:\n"
                      f"{traceback.format_exc()}", file=sys.stderr)
        return False

    def status(self, text: str, kind: str = "idle",
               progress: float | None = None):
        if self.statusbar is not None:
            self.statusbar.message(text, kind)
            self.statusbar.progress(progress)

    def add_poll(self, fn: Callable):
        self.polls.append(fn)

    # ---- view registry
    def on_render(self, stage: str, view: str, fn: Callable):
        self.renderers[(stage, view)] = fn

    def render(self, stage: str, view: str | None = None):
        """Redraw one view, or every view of a stage.

        A renderer that raises is contained HERE: it gets a loud panel in
        its own view and the traceback on stderr, and the caller carries on.
        Letting it propagate desynchronised the shell — the container had
        already been switched, so the tab strip and the tree kept painting
        the previous view while a different one was on screen.
        """
        import sys
        import traceback

        for (st, vw), fn in list(self.renderers.items()):
            if st != stage or (view is not None and vw != view):
                continue
            # whatever this view was owed, it is being paid now
            self._owed.discard((st, vw))
            try:
                fn()
            except Exception:
                tb = traceback.format_exc()
                print(f"[v3] view '{st}/{vw}' failed to render:\n{tb}",
                      file=sys.stderr)
                self._log_render_error(st, vw, tb)
                self._render_error(st, vw, tb)

    #: where a failed render is RECORDED, under the repo root. stderr goes
    #: to whichever terminal launched the shell (and to nowhere at all in the
    #: native window), and the panel on screen is the only other copy — so a
    #: crash reported as "it says Traceback (most recent call last): File
    #: context.py, line 199, in render ..." could not be read by anyone who
    #: was not sitting in front of it. This file is the copy that survives.
    ERROR_LOG = "results/v3_render_errors.log"

    def _log_render_error(self, stage: str, view: str, tb: str):
        """Append a failed render to :data:`ERROR_LOG`, with the state that
        would be needed to reproduce it. Never raises: a logger that took the
        shell down would be worse than the crash it is recording."""
        import time
        from pathlib import Path

        try:
            W = (self.S.get("wing") or {})
            head = (f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {stage}/{view}\n"
                    f"  problem : {W.get('problem')}\n"
                    f"  choices : {W.get('choices')}\n"
                    f"  flags   : {W.get('flags')}\n"
                    f"  fixed   : {W.get('fixed')}\n"
                    f"  bounds  : {sorted((W.get('bounds') or {}))}\n")
            path = Path(__file__).resolve().parents[2] / self.ERROR_LOG
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(head + tb + "\n")
        except Exception:                      # noqa: BLE001 — a logger
            pass

    @staticmethod
    def _headline(tb: str) -> str:
        """The one line worth reading out of a traceback: WHERE it broke and
        WHAT it raised.

        A Python traceback is printed oldest frame first, so the top of the
        panel is always this file's ``fn()`` — the containment, identical for
        all nineteen views — and the frame that names the defect is at the
        bottom, past the fold. Every crash reported by hand arrived as
        "context.py, line 199, in render ...", which identifies nothing. This
        puts the last frame and the exception at the TOP, where a glance (or
        a copied first line) carries the defect.
        """
        where, lines = "", [ln.rstrip() for ln in tb.splitlines() if ln.strip()]
        for ln in lines:
            text = ln.strip()
            if text.startswith("File \"") and ", line " in text:
                try:
                    path = text.split("\"")[1]
                    rest = text.split(", line ", 1)[1]
                    num = rest.split(",", 1)[0]
                    fn = rest.split(" in ", 1)[1] if " in " in rest else ""
                except (IndexError, ValueError):
                    continue
                where = f"{path.rsplit('/', 1)[-1]}:{num}" + (f" in {fn}"
                                                             if fn else "")
        # the exception is the last non-indented line of the traceback
        raised = next((ln for ln in reversed(lines)
                       if ln[:1] not in (" ", "\t") and ":" in ln), "")
        return " — ".join(p for p in (raised.strip(), where) if p)

    def _render_error(self, stage: str, view: str, tb: str):
        from nicegui import ui

        container = self.views.get((stage, view))
        if container is None:
            return
        with container:
            ui.label(f"This view failed to render ({stage}/{view}). The "
                     "rest of the session is unaffected.").classes("hint")
            head = self._headline(tb)
            if head:
                # ...the defect itself, above the traceback rather than 30
                # lines into it: this is the line to quote when reporting it
                ui.label(head).classes("w-full").style(
                    "font-family:ui-monospace,monospace;font-weight:600;"
                    "white-space:pre-wrap;word-break:break-word")
            ui.label(f"Recorded in {self.ERROR_LOG} with the state that "
                     f"reproduces it.").classes("hint")
            ui.code(tb).classes("w-full")
