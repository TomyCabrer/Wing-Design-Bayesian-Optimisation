"""V4 shell assembly — V3's shell, mounting two more stages.

There is no copy of V3's chrome here. ``gui/v3/app.py``'s SOURCE is executed
a second time as a distinct module object (:data:`SHELL_MODULE`) whose
``session`` global is rebound to :mod:`gui.v4.session`, so every function in
it — the tree, the tab strip, the breadcrumb, the status bar, the stale-source
tripwire, ``File > New session`` — reads V4's seven-stage list while V3's own
module object goes on reading its five.

That is what makes "V3 is unchanged" true in the strong sense. It is not
that V3's file has had its V4 lines deleted and could grow them back: V3
cannot see stage 5 or stage 6, because the only place they exist is a
namespace V3 never looks at. And V4 cannot fall behind V3, because it is
running V3's code.

The two things V4 does have to say for itself are WHERE its stage modules
live (V3's ``_mount`` hard-codes ``gui.v3.stages.``) and which module each
stage is, so both are re-bound after the source runs.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _p not in sys.path:                            # editable-install safety
        sys.path.insert(0, _p)

from gui.v3 import app as _v3app, theme, widgets      # noqa: E402
from gui.v3.context import Ctx                        # noqa: E402

from . import session                                 # noqa: E402

__all__ = ["assemble", "run", "STAGE_MODULES", "SHELL_MODULE"]

#: stage -> the module that builds it, as a DOTTED PATH. V3's five come from
#: ``gui.v3.stages`` untouched; V4's two are its own. The second surface's
#: stage is the SAME module built a second time, exactly as in V3.
STAGE_MODULES = {
    **{k: f"gui.v3.stages.{v}" for k, v in _v3app.STAGE_MODULES.items()},
    "controls": "gui.v4.stages.controls",
    "flight": "gui.v4.stages.flight",
}

#: the name the re-executed shell is registered under
SHELL_MODULE = "gui.v4._shell"


def _mount(ctx: Ctx, stage: str):
    """V3's ``_mount``, with the module path taken from :data:`STAGE_MODULES`
    instead of assembled from a hard-coded package.

    A stage that fails to import or build gets a loud panel in its own work
    area rather than taking the shell down — same contract as V3's.
    """
    from nicegui import ui

    try:
        mod = importlib.import_module(STAGE_MODULES[stage])
        if stage in session.STAGE_SURFACE:
            mod.build(ctx, stage)
        else:
            mod.build(ctx)
    except Exception:
        tb = traceback.format_exc()
        print(f"[v4] stage '{stage}' failed to build:\n{tb}", file=sys.stderr)
        container = ctx.views.get((stage, session.VIEWS[stage][0][0]))
        if container is None:
            return
        with container:
            ui.label(f"STAGE ERROR — {stage}").classes("sect-head") \
                .style(f"color:{theme.BAD}")
            ui.code(tb).classes("w-full")


def _load_shell():
    """Execute ``gui/v3/app.py`` as ``gui.v4._shell`` and rebind its session.

    ``from . import session, theme, widgets`` inside that source resolves
    against THIS package, so the three submodules are pre-seeded to V3's
    objects first — V4 has a ``session`` of its own and deliberately has no
    ``theme`` or ``widgets`` of its own, because a second stylesheet is
    exactly the kind of drift this module exists to prevent.
    """
    if SHELL_MODULE in sys.modules:
        return sys.modules[SHELL_MODULE]

    pkg = sys.modules[__package__]
    for name, mod in (("session", session), ("theme", theme),
                      ("widgets", widgets)):
        sys.modules[f"{__package__}.{name}"] = mod
        setattr(pkg, name, mod)
    sys.modules[f"{__package__}.context"] = sys.modules["gui.v3.context"]

    spec = importlib.util.spec_from_file_location(
        SHELL_MODULE, _v3app.__file__)
    shell = importlib.util.module_from_spec(spec)
    sys.modules[SHELL_MODULE] = shell
    try:
        spec.loader.exec_module(shell)
    except Exception:
        del sys.modules[SHELL_MODULE]
        raise
    # ...and the two bindings that make it V4's. ``session`` is already V4's
    # by the pre-seed above; setting it again is the statement, not the
    # mechanism, and it is what a reader of this file needs to see.
    shell.session = session
    shell.STAGE_MODULES = STAGE_MODULES
    shell._mount = _mount
    # ...and what RUN means on the two stages V3's table has never heard of.
    # Extended rather than replaced, so a V3 stage's Run action cannot drift
    # apart from V4's copy of it.
    shell.RUN_ACTIONS = {**shell.RUN_ACTIONS,
                         "controls": "controls_rebuild",
                         "flight": "flight_arm"}
    return shell


class _PinnedSession:
    """:mod:`gui.v4.session`, with ``make_session`` pinned to ONE dict.

    Every stage closes over its own sub-dict at BUILD time (``A``, ``W``,
    ``C``, ``F``), so a page that is to carry an existing design has to be
    handed that design before a single stage is mounted — swapping ``ctx.S``
    afterwards would leave seven stages writing into a dict nothing reads.

    So the shell's view of the session module is replaced for exactly the
    length of one :func:`assemble` call. Everything except ``make_session``
    forwards, which matters: the shell reads ``STAGES``, ``VIEWS``,
    ``stage_states`` and a dozen helpers off the same name, and
    ``_new_session`` calls ``make_session`` LATER — after the swap is
    undone — so File > New session still gets a genuinely fresh one.
    """

    def __init__(self, state: dict):
        self._state = state

    def __getattr__(self, name):            # everything else is V4's own
        return getattr(session, name)

    def make_session(self, medium: str = "air") -> dict:
        return self._state


def assemble(medium: str = "air", *, state: dict | None = None,
             lazy: bool = False) -> Ctx:
    """Build the whole V4 shell and return its context (no server started).

    ``state`` is an EXISTING session to build the shell around, which is
    what :func:`page` hands back on a reload. Left out, a fresh one is made,
    so a caller that just wants a shell (every test in the suite) is
    unaffected.
    """
    shell = _load_shell()
    if state is None:
        ctx = shell.assemble(medium, lazy=lazy)
    else:
        real, shell.session = shell.session, _PinnedSession(state)
        try:
            ctx = shell.assemble(medium, lazy=lazy)
        finally:
            shell.session = real
    # ...and what the session actually HAS. Announcing two stages the tree
    # is not going to show is worse than announcing nothing: a car rear wing
    # is bolted to a car, so it has no controls to cut and nothing to fly,
    # and the honest line names the pipeline this session is. Read off the
    # SESSION and not the ``medium`` argument, which is "air" on the carried
    # state path (``page`` rebuilds every window through it).
    if not session.free_flight(ctx.S):
        # ...and the REASON, from the one place that owns it, so the banner
        # cannot promise a stage the tree beside it does not show. Three
        # configurations end at stage 4 now, not one.
        why = session.stage_states(ctx.S)["flight"][1]
        ctx.log(f"V4 — the pipeline ends at stage 4 here: {why}.", "info")
    else:
        ctx.log("V4 — stage 5 cuts the control surfaces out of the design "
                "and adds the fin the lattice never had; stage 6 flies it. "
                "Arrow keys pitch and yaw, A/D roll, W/S throttle.", "info")
    return ctx


#: THE DESIGN, held across page builds. See :func:`page` for why this is a
#: module global and not a nicegui storage entry: it holds live objects (a
#: built ``FlightModel``, a numpy state vector, a run record), none of which
#: survive being serialised, and there is exactly one desktop user.
PAGE_STATE: dict = {"S": None}


def page() -> Ctx:
    """Build the shell for ONE page, around the design the last one had.

    NiceGUI 3 runs a *root page function* per request, so everything this
    module builds is built again for every page load — a reload, a native
    window that lost its socket and reloaded itself, a second tab, or any
    URL that 404s (nicegui's 404 handler builds the root page too). With the
    UI declared at module scope and no root function, nicegui falls back to
    SCRIPT MODE and re-executes the launcher with ``runpy`` to get one,
    which called ``assemble()`` again — and ``assemble`` calls
    ``make_session``. Measured: a typed aileron station of 0.42 read 0.6
    again after one reload, and the shell was back on stage 1 with the
    design gone. That is the whole of "it resets everything".

    The elements cannot be kept — they belong to the client that went away —
    but the ANSWERS can, so only the view is rebuilt. One flag is cleared
    first, because it describes the page and not the design: the flight
    loop's ``running``. Its ``ui.timer`` died with the page, so a session
    that still said it was flying would paint a Flight stage the tree calls
    "running" with nothing moving in it. Everything the stages hold in their
    own per-build ``view`` dicts — element handles, the aircraft mesh's
    static URL, the installed browser half — is rebuilt with them.
    """
    S = PAGE_STATE["S"]
    if S is not None:
        (S.get("flight") or {}).__setitem__("running", False)
        # ...and off a stage this configuration no longer HAS. The medium is
        # switchable mid-session, and the tab strip and the properties grid
        # are painted from ``S["ui"]["selected"]`` without consulting
        # :func:`gui.v4.session.stage_visible` — so a session left on Flight
        # whose medium then became the track would paint a Flight tab strip
        # over a tree with no Flight node in it. Done HERE and not in the
        # handler that changes the medium: the V3 stage modules resolve
        # ``session`` to V3's own, whose ``stage_visible`` does not carry
        # this rule.
        if not session.stage_visible(S, S["ui"]["selected"]):
            S["ui"]["selected"] = "mission"
    ctx = assemble(state=S, lazy=True)
    if S is not None:
        ctx.log("the window was rebuilt (reload, reconnect or a second "
                "tab) — the design, the section, the run and the controls "
                "are the ones this session already had", "info")
    PAGE_STATE["S"] = ctx.S
    return ctx


def _only_the_root_page_is_a_page():
    """Stop a WRONG URL from rebuilding the entire shell.

    NiceGUI 3 routes every 404 to the root page function
    (``nicegui/nicegui.py::_exception_handler_404``), which is how "/" is
    served at all — and it means any other request the browser invents
    (``/favicon.ico`` where none is registered, a devtools probe, a stale
    ``/_flight/<t>.stl`` from a previous build, a mistyped address) builds
    all seven stages again. Measured on this shell: ``GET /nope`` answered
    **200 in 1.0–2.8 s**, with the event loop blocked for the whole of it,
    so nothing else on the page moved — and :func:`page` clears
    ``S["flight"]["running"]`` on the way through, which stops a simulation
    that was flying.

    So the root path keeps nicegui's behaviour and everything else gets a
    plain, cheap 404 — which is what a 404 was supposed to be.
    """
    from nicegui import app as ngapp
    from starlette.responses import PlainTextResponse

    original = ngapp.exception_handlers.get(404)
    if original is None:                       # nicegui changed: leave it be
        return

    async def _handler(request, exc):
        if request.url.path in ("", "/"):
            return await original(request, exc)
        return PlainTextResponse(f"{request.url.path} is not a page here",
                                 status_code=404)

    ngapp.add_exception_handler(404, _handler)


def run(native: bool = True, port: int = 8769):
    from nicegui import ui

    _only_the_root_page_is_a_page()
    ui.run(
        page,
        title="AeroBO — design pipeline + flight",
        favicon="🛩️",
        dark=False,
        native=native,
        window_size=(1600, 1000) if native else None,
        reload=False,
        port=port,
        show=not native,
        # HOW LONG A SOCKET MAY BE AWAY BEFORE THE PAGE IS THROWN AWAY.
        # NiceGUI's default is 3 s: miss it and the server forgets the
        # client, the browser reloads itself, and the shell comes back on
        # stage 1 with the design carried over but the tab, the scroll and
        # the flight loop gone — "it gets bugged and takes a while before we
        # can use the program". Three seconds is not a lot of margin for a
        # WKWebView that has just been handed a stage to paint while a
        # search is running, so the window gets ten.
        reconnect_timeout=10.0,
    )
