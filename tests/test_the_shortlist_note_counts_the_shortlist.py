"""The shortlist's note counts the SHORTLIST, not the caller's restriction.

``screen_at_point`` grew a ``names`` PARAMETER — the caller's restriction on
who may be ranked at all, which a vertical stabiliser needs (a fin may only be
given a symmetric section). The local list of shortlisted names was called
``names`` too, so it was renamed ``short``. One reader was left behind, inside
the note's f-string, and it now read the parameter: ``None`` for every surface
that restricts nothing, i.e. the wing, the tail and the airfoil-only mode.

    TypeError: object of type 'NoneType' has no len()

...raised AFTER pass 2's live XFOIL sweeps had been paid for, from a line that
only builds a sentence. The shell reported it as a failed section screen.

The invariant asserted here is the one that cannot be got wrong twice: the
number the note quotes is the number of sections the report says were
shortlisted. It binds whether or not a restriction was passed, which is what
makes it a test of the sentence rather than of the None.
"""

from __future__ import annotations

import re as _re
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aerobo import api                                   # noqa: E402

RE_OFF_LIBRARY = 3e5      # not the cached point, so there IS a shortlist


def _library_point():
    lib = api.screen_library_point()
    if not lib:
        pytest.skip("no cached whole-library screen on this machine")
    return lib


def _quoted_count(note: str) -> int:
    m = _re.search(r"the (\d+) best of", note)
    assert m, f"the note no longer states a count: {note!r}"
    return int(m.group(1))


def _screen(**kw):
    lib = _library_point()
    assert abs(RE_OFF_LIBRARY - float(lib["re"])) > 1e-6 * float(lib["re"]), (
        "this machine's library cache sits at the point the test screens at, "
        "so there is no second pass and nothing to shortlist")
    return api.screen_at_point(re=RE_OFF_LIBRARY, tc_min=0.15, top_n=3,
                               with_shape=False, workers=6, **kw)


def test_an_unrestricted_screen_states_how_many_it_shortlisted():
    """No restriction is the wing's, the tail's and airfoil-only mode's call
    — the one that used to raise before it could return."""
    got = _screen()
    block = got["shortlist"]
    assert block["source"] == "library"
    assert _quoted_count(block["note"]) == len(block["names"]) == block["n"]


def test_a_restricted_screen_still_counts_its_shortlist():
    """With a restriction the two numbers are both available, and the note
    must quote the one it names: the sections that were re-screened."""
    first = _screen()
    only = list(first["shortlist"]["names"])        # all swept already: cached
    got = _screen(names=only)
    block = got["shortlist"]
    assert _quoted_count(block["note"]) == len(block["names"]) == block["n"]
