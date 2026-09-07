"""A shell serving code older than the tree has to say so.

``ui.run(reload=False)``: a V3 window keeps the modules it imported at
start-up for as long as it is open. That is deliberate — a live reload in
the middle of a run would be worse — but it is invisible, and it has now
cost real time twice. The reported case: a fix for "no solution was found"
landed at 13:34, and the three runs that came back with nothing at 17:12
were launched from a window started the previous evening, so the fix's flag
is simply absent from their stored configs. Nothing on screen could have
told anyone.

The check is a comparison and a throttle, and both are tested here against
independently computed mtimes rather than against themselves.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ROOTS = (REPO_ROOT / "gui", REPO_ROOT / "src" / "aerobo")


def _sources() -> list[Path]:
    return [p for root in ROOTS for p in root.rglob("*.py")
            if "__pycache__" not in p.parts]


def test_the_shell_notices_source_newer_than_itself(monkeypatch):
    """The direction of the comparison, against a cut that splits the tree:
    an "everything is stale" or "nothing is stale" bug passes neither half."""
    from gui.v3 import app

    paths = _sources()
    assert len(paths) > 10, "the tripwire is watching nothing"
    stamps = sorted(p.stat().st_mtime for p in paths)
    cut = stamps[len(stamps) // 2]

    monkeypatch.setattr(app, "_LOADED_AT", cut)
    app._stale_cache.update(at=0.0, files=())
    expected = {p.name for p in paths if p.stat().st_mtime > cut}
    assert set(app._stale_sources()) == expected
    assert expected, "the cut chose nothing — the test proves nothing"

    # ...and a window younger than every edit reports a clean tree
    monkeypatch.setattr(app, "_LOADED_AT", stamps[-1] + 1.0)
    app._stale_cache.update(at=0.0, files=())
    assert app._stale_sources() == ()


def test_the_throttle_serves_the_cache_and_not_a_stale_answer(monkeypatch):
    """Re-stating the tree on every 0.5 s repaint would be wasteful, so the
    answer is cached — but a cache that never expires is a tripwire that
    fires once and then lies. Both halves are asserted."""
    from gui.v3 import app

    monkeypatch.setattr(app, "_LOADED_AT", 0.0)
    app._stale_cache.update(at=0.0, files=())
    first = app._stale_sources()
    assert first, "with a zero load time every file is newer"

    # within the window the cache answers, whatever the tree says
    app._stale_cache.update(files=("sentinel.py",))
    assert app._stale_sources() == ("sentinel.py",)

    # ...and past it, the tree does
    app._stale_cache["at"] -= app._STALE_EVERY_S + 1.0
    assert set(app._stale_sources()) == set(first)
