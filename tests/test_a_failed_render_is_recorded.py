"""A view that falls over leaves a copy somebody else can read.

``Ctx.render`` contains a raising renderer — the view gets a traceback panel
and the rest of the shell carries on — and until now that traceback existed
in exactly two places nobody else could reach: the panel on screen, and
stderr, which goes to whichever terminal launched the shell and to nowhere at
all in the native window.

So a crash arrives as a screenshot of its own first two lines::

    Traceback (most recent call last):
      File "gui/v3/context.py", line 199, in render …

— which names the containment and not the defect: line 199 is ``fn()``, the
same line for all nineteen views. Every frame that says which card broke is
below the fold.

The failed render is now appended to ``results/v3_render_errors.log`` (the
gitignored working directory) with the state that reproduces it: the problem,
the choices, the flags, the pins and which rows were typed.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _shell_with_a_broken_view(tmp_path, monkeypatch):
    from gui.v3.app import assemble
    from gui.v3.context import Ctx

    log = tmp_path / "render_errors.log"
    monkeypatch.setattr(Ctx, "ERROR_LOG", str(log))
    ctx = assemble("air")
    ctx.act("accept_mission")

    def _boom():
        raise RuntimeError("the card this test broke on purpose")

    ctx.renderers[("wing", "box")] = _boom
    return ctx, log


def test_the_traceback_and_the_state_are_written(tmp_path, monkeypatch,
                                                 capsys):
    ctx, log = _shell_with_a_broken_view(tmp_path, monkeypatch)
    ctx.render("wing", "box")

    assert log.exists(), "a failed render wrote nothing"
    text = log.read_text(encoding="utf-8")
    assert "wing/box" in text                       # WHICH view
    assert "the card this test broke on purpose" in text
    assert "RuntimeError" in text
    for field in ("problem", "choices", "flags", "fixed", "bounds"):
        assert field in text, field                 # ...and how to repro it


def test_the_view_is_still_contained_and_the_shell_carries_on(tmp_path,
                                                             monkeypatch):
    """The behaviour the log is added to, unchanged: a broken card is a panel
    in its own view, not an exception out of ``render``."""
    from gui.v3.app import assemble
    from gui.v3.context import Ctx

    monkeypatch.setattr(Ctx, "ERROR_LOG", str(tmp_path / "log"))
    ctx = assemble("air")
    ctx.act("accept_mission")
    ctx.renderers[("wing", "box")] = lambda: 1 / 0
    ctx.render("wing")                               # must not raise
    texts = [getattr(e, "text", "") or ""
             for e in ctx.views[("wing", "box")].descendants()]
    assert any("failed to render" in t for t in texts)
    # ...and the view beside it drew normally
    others = [getattr(e, "text", "") or ""
              for e in ctx.views[("wing", "type")].descendants()]
    assert not any("failed to render" in t for t in others)


def test_the_logger_never_takes_the_shell_down(tmp_path, monkeypatch):
    """A logger that raised would be worse than the crash it records — so an
    unwritable path is swallowed and the panel is still drawn."""
    from gui.v3.app import assemble
    from gui.v3.context import Ctx

    # a path whose parent cannot be created (a file, not a directory)
    blocked = tmp_path / "file"
    blocked.write_text("not a directory")
    monkeypatch.setattr(Ctx, "ERROR_LOG", str(blocked / "deeper" / "log"))

    ctx = assemble("air")
    ctx.act("accept_mission")
    ctx.renderers[("wing", "box")] = lambda: 1 / 0
    ctx.render("wing", "box")                        # must not raise
    texts = [getattr(e, "text", "") or ""
             for e in ctx.views[("wing", "box")].descendants()]
    assert any("failed to render" in t for t in texts)


# ------------------------------------------------ the panel names the defect

def test_the_panel_leads_with_the_last_frame_not_the_containment():
    """What a user can actually copy. The first frame of every one of these
    tracebacks is ``context.py`` line 199 ``fn()`` — the guard — so a panel
    that starts there tells the reader which of the nineteen views is broken
    and nothing else. The headline is the LAST frame and the exception."""
    from gui.v3.context import Ctx

    tb = ('Traceback (most recent call last):\n'
          '  File "/x/gui/v3/context.py", line 199, in render\n'
          '    fn()\n'
          '  File "/x/gui/v3/stages/wing.py", line 3542, in _chord_law_panel\n'
          '    reach, _, _ = _chord_reach(key, labels, eff)\n'
          "ValueError: chord law 'elliptic' of order 3 exceeds the largest "
          'order it offers (1)\n')
    head = Ctx._headline(tb)
    assert head.startswith("ValueError: chord law 'elliptic'")
    assert "wing.py:3542 in _chord_law_panel" in head
    assert "context.py" not in head


def test_the_headline_survives_a_traceback_it_cannot_parse():
    """A logger that raised would be worse than the crash it records, and so
    would a headline: anything unparseable yields an empty string and the
    panel simply shows the traceback."""
    from gui.v3.context import Ctx

    assert Ctx._headline("") == ""
    assert isinstance(Ctx._headline("not a traceback at all"), str)


def test_the_panel_shows_the_headline_and_the_log_path(tmp_path,
                                                      monkeypatch):
    from gui.v3.app import assemble
    from gui.v3.context import Ctx

    monkeypatch.setattr(Ctx, "ERROR_LOG", str(tmp_path / "log"))
    ctx = assemble("air")
    ctx.act("accept_mission")
    ctx.renderers[("wing", "box")] = lambda: 1 / 0
    ctx.render("wing", "box")
    texts = [str(getattr(e, "text", "") or "")
             for e in ctx.views[("wing", "box")].descendants()]
    assert any("ZeroDivisionError" in t for t in texts)
    assert any(ctx.ERROR_LOG in t for t in texts)
