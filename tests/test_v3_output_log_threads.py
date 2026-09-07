"""A worker thread must not build nicegui elements in the output log.

``ctx.log`` is called straight from worker threads — stage 3's
``_auto_recommend`` and stage 4's result writers — and every element it
builds writes nicegui's *global* ``binding.bindable_properties`` while the
0.5 s repaint (and the log's own ``max_lines`` prune) iterates that same
dict via ``list(...)``. The collision is a ``RuntimeError: dictionary
changed size during iteration`` that costs either the log line plus a
traceback, or an aborted repaint that leaves the chrome half-drawn.

So the contract asserted here is about the OUTCOME, not about who called
what: a line written off the log's home thread produces **no elements and
no binding-table entries** at write time, and appears — with the timestamp
it was written at, in the right order — once the shell's heartbeat runs.
"""

from __future__ import annotations

import threading


def _texts(row) -> list[str]:
    return [getattr(e, "text", "") or "" for e in row.descendants()]


def _log_from_thread(ctx, text: str, level: str = "info") -> list[str]:
    """Call ``ctx.log`` on a real worker thread; return anything it raised."""
    boom: list[str] = []

    def worker():
        try:
            ctx.log(text, level)
        except Exception as exc:                       # pragma: no cover
            boom.append(repr(exc))

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    return boom


def test_a_worker_log_line_builds_nothing_until_the_heartbeat(capsys):
    from nicegui import binding

    from gui.v3.app import assemble

    ctx = assemble()
    n_lines = len(ctx.output.lines)
    n_bind = len(binding.bindable_properties)

    boom = _log_from_thread(ctx, "a worker wrote this", "warn")
    assert boom == [], boom
    # the worker touched neither the DOM nor the global binding table: this
    # is the whole defect — those two writes are what race the repaint
    assert len(ctx.output.lines) == n_lines
    assert len(binding.bindable_properties) == n_bind

    # ...and the line is not lost, only waiting
    drawn = [fn() for fn in list(ctx.polls)]           # the 0.5 s heartbeat
    assert len(ctx.output.lines) == n_lines + 1
    assert "a worker wrote this" in _texts(ctx.output.lines[-1])
    levels = [" ".join(e.classes) for e in ctx.output.lines[-1].descendants()]
    assert any("log-warn" in c for c in levels), levels
    # the drain must not force a chrome repaint on every worker line
    assert drawn[0] is False, drawn

    err = capsys.readouterr().err
    assert "Traceback" not in err, err


def test_the_queued_line_keeps_the_time_it_was_written(capsys):
    """The stamp is taken when the worker wrote, not when the tick drew.

    A queued line drawn a heartbeat later would otherwise read as if it had
    happened at the drain, which is exactly the kind of lie this log exists
    to avoid.
    """
    from gui.v3 import widgets
    from gui.v3.app import assemble

    ctx = assemble()
    real = widgets.time.strftime
    widgets.time.strftime = lambda *_a, **_kw: "01:02:03"
    try:
        assert _log_from_thread(ctx, "written at 01:02:03") == []
    finally:
        widgets.time.strftime = real

    widgets.time.strftime = lambda *_a, **_kw: "09:09:09"
    try:
        for fn in list(ctx.polls):
            fn()
    finally:
        widgets.time.strftime = real

    row = _texts(ctx.output.lines[-1])
    assert "01:02:03" in row and "09:09:09" not in row, row

    err = capsys.readouterr().err
    assert "Traceback" not in err, err


def test_the_home_thread_still_draws_at_once_and_in_order(capsys):
    """Queuing is for workers only, and it does not reorder the log.

    The shell logs from its own thread all over the place (menu handlers,
    renderers, ``_stop``); those must still be on screen the moment they
    are written, and a worker line written BEFORE one of them must not jump
    ahead of it just because the drain runs later.
    """
    from gui.v3.app import assemble

    ctx = assemble()
    assert _log_from_thread(ctx, "worker first") == []
    n = len(ctx.output.lines)
    ctx.log("main thread second")
    # the home-thread line drew immediately AND took the queued one with it,
    # so the two are on screen in the order they were written
    assert len(ctx.output.lines) == n + 2
    assert "worker first" in _texts(ctx.output.lines[-2])
    assert "main thread second" in _texts(ctx.output.lines[-1])
    assert ctx.output._pending == []

    err = capsys.readouterr().err
    assert "Traceback" not in err, err
