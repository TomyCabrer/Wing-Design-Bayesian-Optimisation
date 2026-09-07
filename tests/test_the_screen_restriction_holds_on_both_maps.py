"""A caller's restriction survives every pass of the two-pass screen.

``screen_at_point`` ranks the cached library point TWICE — once on a live
min-max over its own population, once on the frozen band pass 2 is scored on —
and shortlists the UNION of the two leaders, so neither map decides alone.

``names`` is the caller's restriction on who may be ranked at all. A vertical
stabiliser needs it: at zero sideslip a fin must make no side force, so it may
only be given a symmetric section (``symmetric_section_names``), and filtering
after the shortlist would spend the sweep on sections it cannot use.

The restriction reached pass 1 and pass 2 but NOT the band pass, so the union
put that pass's whole-library leaders back in. Measured off this machine's
cache, a restricted screen came back with `mrc-20` and `e502` — two sections
the caller had excluded — shortlisted, swept for real, and eligible to win.
For a fin that is a cambered section on a surface that cannot trim one.

Asserted here on the report, which is the only place a shell can see it: a
screen restricted to a set must not shortlist, sweep or rank anything outside
that set.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aerobo import api                                   # noqa: E402

RE_OFF_LIBRARY = 3e5      # not the cached point, so both maps are ranked


def _screen(**kw):
    lib = api.screen_library_point()
    if not lib:
        pytest.skip("no cached whole-library screen on this machine")
    if abs(RE_OFF_LIBRARY - float(lib["re"])) <= 1e-6 * float(lib["re"]):
        pytest.skip("this machine's cache sits at the screened point")
    return api.screen_at_point(re=RE_OFF_LIBRARY, tc_min=0.15, top_n=3,
                               with_shape=False, workers=6, **kw)


def test_nothing_outside_the_restriction_is_shortlisted_or_ranked():
    # the unrestricted screen's own shortlist: every one of these is already
    # swept and cached at this point, so the restricted run costs no XFOIL
    only = list(_screen()["shortlist"]["names"])
    assert len(only) > api.SHORTLIST_N, (
        "the union added nothing on this cache, so this test cannot see "
        "whether the second map carried the restriction")
    keep = set(only[:api.SHORTLIST_N])          # drop what the union added

    got = _screen(names=sorted(keep))
    assert set(got["shortlist"]["names"]) <= keep
    assert {r["name"] for r in got["ranked"]} <= keep


def test_a_fin_is_never_shortlisted_a_cambered_section():
    """The restriction as the shell states it (``screen_names_for('fin')``):
    the symmetric members, measured from coordinates."""
    names = api.symmetric_section_names()
    if not names:
        pytest.skip("the section coordinate cache is not built")
    got = _screen(names=list(names))
    allowed = set(names)
    assert set(got["shortlist"]["names"]) <= allowed
    assert {r["name"] for r in got["ranked"]} <= allowed
