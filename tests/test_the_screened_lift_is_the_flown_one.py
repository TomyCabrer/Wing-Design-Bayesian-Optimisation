"""The upright catalogue is read at MINUS the load, not at its magnitude.

`api.trim_surface_cl` returns two numbers about the same surface: `cl`, the
load it carries in the AIRCRAFT frame, and `cl_section`, the lift at which the
UPRIGHT catalogue is read to screen the section it will fly. `polar.py` fixes
the map between them — ``InvertedPolar.cl(a) = -base.cl(-a)`` — so a surface
mounted inverted carrying aircraft-frame C flies its base section at **-C**.

That identity used to be written ``abs(cl)``. It is right for a surface
carrying a DOWNLOAD, which is 1920 of the 2064 registry families with a second
surface and every case the identity was written against. It is wrong for a
surface STATED inverted that trims to an UP-load — the water elevator at its
default CG — and there ``abs()`` screens the section on the cambered side it
never works.

WHY THIS FILE EXISTS SEPARATELY. The only other test that can tell the two
forms apart is
``test_trim_surface_lift.py::test_the_second_stage_screens_that_surface_at_its_trim_lift[water]``,
and it can only do so while ``gui/v3/config.py`` pins ``TAIL_MOUNT =
"inverted"``. That pin is a shell decision, not a property of the engine: set
it back to ``"auto"`` and no registry family reaches ``inverted=True`` with a
positive load, so the distinction becomes unreachable and the contract is
unguarded. This file states the contract where it lives — at the `api` level,
off a flag the caller passes — so it holds whatever the shell decides.

The assertions are on the OUTCOME (the drag and the suction peak the surface
actually flies at), not on the expression, because a test that restates the
code is not a test: ``abs()`` and ``-cl`` differ here by 50.7 % of profile drag
and a factor 3.4 in cp_min, and that is what is asserted.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aerobo import api                                          # noqa: E402
from aerobo import polar as _polar                              # noqa: E402
from aerobo.tail import orient_section                          # noqa: E402


#: the one family in the registry that a STATED inverted mount puts on the
#: wrong side of its own polar: its elevator trims to an UP-load
_UP_LOADED = "hydrofoil + elevator"


def _alpha_for_cl(pol, cl: float) -> float:
    """WHERE on the stored curve a given lift sits. The polars here are
    tables; this is an inverse lookup, not a fit."""
    lo, hi = pol.alpha_valid
    a = np.linspace(float(lo), float(hi), 20001)
    return float(np.interp(float(cl), np.asarray(pol.cl(a), dtype=float), a))


def _stated(name: str, mount: str) -> dict:
    return api.trim_surface_cl(name, flags={api.TAIL_MOUNT_KEY: mount})


# --------------------------------------------------- 1. the case exists
def test_a_stated_inverted_surface_can_trim_to_an_UP_load():
    """The premise, out loud. If this ever stops being true the rest of the
    file proves nothing and needs a new home — so it is asserted, not assumed."""
    t = _stated(_UP_LOADED, "inverted")
    assert t is not None
    assert t["inverted"] is True, "the mount was STATED and must be honoured"
    assert t["cl"] > 0.02, (
        f"{_UP_LOADED} is here because a stated-inverted surface can carry an "
        f"UP-load; its default has moved (cl {t['cl']:+.4g})")


# ------------------------------------------- 2. the contract, on the outcome
def test_the_screened_point_is_the_point_the_surface_actually_flies():
    """Not the expression — the DRAG. The catalogue is stored upright; the
    surface mounts its section inverted; the screen must land on the same
    place of the same curve, and therefore on the same drag and the same
    suction peak."""
    t = _stated(_UP_LOADED, "inverted")
    pol = _polar.default_polar()
    assert float(pol.cl(0.0)) > 0.05, \
        "a symmetric section is its own mirror and would prove nothing here"

    mounted = orient_section(pol, None, True)
    a_flown = _alpha_for_cl(mounted, t["cl"])            # what it flies
    a_screen = _alpha_for_cl(pol, t["cl_section"])       # where it is screened

    assert float(mounted.cl(a_flown)) == pytest.approx(t["cl"], rel=1e-6)
    assert a_screen == pytest.approx(-a_flown, abs=1e-3)
    assert float(pol.cd(a_screen)) == pytest.approx(
        float(mounted.cd(a_flown)), rel=1e-6)
    assert float(pol.cp_min(a_screen)) == pytest.approx(
        float(mounted.cp_min(a_flown)), rel=1e-6)


def test_screening_at_the_magnitude_would_understate_the_drag_it_flies():
    """The price of the wrong identity, so the test cannot be satisfied by
    weakening it. On the cached NACA 2412 Re 1e6 table the two points are 50 %
    apart in profile drag and 3.4x apart in cp_min — and cp_min feeds a
    cavitation gate 1:1 on a hydrofoil, so the wrong one can clear a section
    the surface cavitates on."""
    t = _stated(_UP_LOADED, "inverted")
    pol = _polar.default_polar()

    a_right = _alpha_for_cl(pol, -t["cl"])          # the contract
    a_wrong = _alpha_for_cl(pol, abs(t["cl"]))      # the superseded identity
    cd_right, cd_wrong = float(pol.cd(a_right)), float(pol.cd(a_wrong))
    cp_right, cp_wrong = float(pol.cp_min(a_right)), float(pol.cp_min(a_wrong))

    assert cd_right > 1.4 * cd_wrong, (
        f"the two identities must be far apart for this case to be worth "
        f"gating: {cd_right*1e4:.1f} vs {cd_wrong*1e4:.1f} counts")
    assert cp_right < 2.5 * cp_wrong, (
        f"cp_min {cp_right:+.3f} vs {cp_wrong:+.3f}")
    # ...and the shipped value is the one the surface flies
    assert float(pol.cd(_alpha_for_cl(pol, t["cl_section"]))) == \
        pytest.approx(cd_right, rel=1e-9)


# ------------------------------------------------- 3. and it is not one case
def test_the_identity_holds_for_every_family_that_can_state_a_mount():
    """A registry sweep, because `cl_section` is a contract of the read-out
    and not a property of one family. Both signs of load are exercised: the
    sweep asserts it found some of each, so it cannot pass by covering only
    the half where the two identities agree."""
    up = down = 0
    for name, spec in api.PROBLEM_SPECS.items():
        if api.TAIL_MOUNT_KEY not in spec.flags:
            continue
        t = _stated(name, "inverted")
        if t is None or not np.isfinite(t["cl"]):
            continue
        assert t["inverted"] is True, name
        assert t["cl_section"] == pytest.approx(-t["cl"], abs=1e-15), name
        if t["cl"] > 0.0:
            up += 1
        elif t["cl"] < 0.0:
            down += 1
    assert down > 0, "no download family swept — the sweep is not covering"
    assert up > 0, (
        "no up-loaded family swept, so this sweep cannot tell -cl from abs(cl); "
        "the case the contract exists for has left the registry")


def test_an_upright_mount_reads_the_catalogue_at_the_load_itself():
    """The other half of the branch, so the contract cannot be satisfied by
    negating unconditionally."""
    seen = 0
    for name, spec in api.PROBLEM_SPECS.items():
        if api.TAIL_MOUNT_KEY not in spec.flags:
            continue
        t = _stated(name, "upright")
        if t is None or not np.isfinite(t["cl"]):
            continue
        assert t["inverted"] is False, name
        assert t["cl_section"] == pytest.approx(t["cl"], abs=1e-15), name
        seen += 1
    #: a loop that can legitimately be empty asserts nothing; say how many it
    #: covered, so a registry change that stops producing upright read-outs
    #: turns this red instead of passing vacuously
    assert seen > 0, "no family swept upright — this loop asserted nothing"
