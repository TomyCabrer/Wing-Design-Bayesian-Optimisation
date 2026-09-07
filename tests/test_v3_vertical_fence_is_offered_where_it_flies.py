"""A tip-device SHAPE carried as a FLAG is offered only where the flag lands.

"Vertical fence (90°)" is not a solver family: the pair ``("free",
"vertical")`` selects an ordinary free-span winglet problem and the 90° is a
VALUE on top of it, ``winglet_type``, which narrows the cant row onto
``objective.WINGLET_TYPES['vertical']``'s 84–90°. A family that does not
declare the key never sees the word — ``api.sanitise_flags`` drops it on the
way to the Run button — and the search flies the family's own cant band
instead (−90…90 on the hydrofoil), a SUPERSET of the one asked for.

The blend has been asked of the registry this way since it was added
(``_blend_flag_available``). The fence was not, so the entry appeared on every
configuration with a tip device and ``winglet_shape_key`` read the word back
out of the state — the select went on saying "vertical fence (90°)" over a
search that had never heard of it. The repo's own rule: a control on screen
whose value is thrown away is the failure ``api.check_flags`` exists for, and
report §17.12's closing paragraph retracts the argument that the shells'
discarding is exempt from it.

The cure is at the MENU: offer the shape only where the derived family honours
the flag, and say why where it does not.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aerobo import api                                        # noqa: E402

from gui import nice_app as v1                                # noqa: E402

from tests.test_every_combination_builds import ALL           # noqa: E402


def test_every_offered_fence_reaches_the_solver_as_a_fence():
    """The sweep the audit measured on: 14 derived families offered the entry
    and dropped the flag — every hydrofoil one and every nonplanar tandem
    one. The flag the shell would send has to survive ``sanitise_flags``."""
    offered = 0
    for label, ch in ALL:
        if "vertical" not in v1.winglet_shapes(ch):
            continue
        offered += 1
        c = dict(ch)
        v1.set_winglet_shape(c, "vertical")
        v1.normalise_choices(c)
        name, _ = v1.derive_problem(c)
        flags = v1.winglet_flags(c)
        assert flags.get("winglet_type") == "vertical", (label, flags)
        assert api.unhonoured_flags(name, flags) == [], (label, name)
    assert offered > 20, offered


def test_the_reason_the_fence_is_missing_is_true_where_it_is_shown():
    """``WINGLET_SHAPE_WHY``'s contract: the reason has to be true of the
    configuration on screen. The new one names the missing FLAG, which is only
    an answer where a tip device exists at all — the car says something else
    (OPTION_WHY_BY_MEDIUM['track']: the endplates are its tip device), and no
    other configuration in the sweep omits the fence without offering a canted
    device. A tripwire: if one ever does, this reason becomes a lie there."""
    for label, ch in ALL:
        if "vertical" in v1.winglet_shapes(ch) or ch["medium"] == "track":
            continue
        assert "canted" in v1.winglet_shapes(ch), label
        assert "winglet_type" in v1.option_why(ch, v1.WINGLET_SHAPE_WHY)[
            "vertical"]
    # ...and the car's own answer is untouched by the flag test
    track = v1.start_choices(medium="track")
    assert "endplates" in v1.option_why(
        track, v1.WINGLET_SHAPE_WHY)["vertical"]


def test_the_narrowed_cant_band_is_what_the_offered_run_searches():
    """The value test, not a name test: the fence's own band (84–90°) has to
    be the row the built problem hands the optimiser."""
    from aerobo import objective

    ch = v1.start_choices(medium="air")
    assert "vertical" in v1.winglet_shapes(ch)
    v1.set_winglet_shape(ch, "vertical")
    name, _ = v1.derive_problem(ch)
    spec = api.PROBLEM_SPECS[name]
    built = spec.build({} if spec.uses_mission else None,
                       v1.winglet_flags(ch), None)
    lo, hi = built.bounds[list(built.param_labels).index("winglet_cant_deg")]
    want = objective.WINGLET_TYPES["vertical"]["cant_bounds"]
    assert (float(lo), float(hi)) == (float(want[0]), float(want[1]))


def test_the_menu_offers_the_fence_exactly_where_the_flag_lands():
    """The RULE, over the whole sweep — the entry is on screen if and only if
    the derived family honours the flag.

    It used to have a standing exception with 16 members: the water families
    and the nonplanar pair declared no ``winglet_type`` because their tip
    device's cant band was a bare constant no builder keyword could reach, so
    the run searched −90…90 under a select reading "vertical fence (90°)" and
    the menu's answer was to drop the entry. The band is settable now
    (``winglet_cant_bounds`` on hydrofoil / hydrotail / tandemvlm) and the
    families declare the key, so the exception is empty — but the rule is
    what this asserts, not the count, because the rule is what keeps a
    control from being drawn over a search that never heard of it.
    """
    for label, ch in ALL:
        if "canted" not in v1.winglet_shapes(ch):
            continue                       # no tip device at all here
        trial = dict(ch, winglets="free", winglet_type="vertical")
        name, _ = v1.derive_problem(trial)
        honours = "winglet_type" in api.accepted_flags(name)
        assert ("vertical" in v1.winglet_shapes(ch)) is honours, (label, name)
        if not honours:
            # ...and where it does not, the reason is on screen
            assert "winglet_type" in v1.option_why(
                ch, v1.WINGLET_SHAPE_WHY)["vertical"], label


def test_a_fence_carried_into_another_medium_reads_as_what_it_flies():
    """The stale-menu half. Choose the fence in air, switch the medium to
    water: the shape the card shows has to be the shape the run flies.

    That used to mean "canted" — the word survived normalisation and the
    flag did not survive the trip to the solver. Now the water family
    honours it, so the card keeps saying "vertical fence" AND the run
    searches a fence: |cant| 84–90, on the DOWN side, which is the sign the
    water band carries (api.WATER_TIP_DIRECTION).
    """
    ch = v1.start_choices(medium="air")
    v1.set_winglet_shape(ch, "vertical")
    v1.normalise_choices(ch, keep="winglets")
    assert v1.winglet_shape_key(ch) == "vertical"

    ch["medium"] = "water"
    v1.normalise_choices(ch, keep="medium")
    assert ch.get("winglet_type") == "vertical", \
        "the premise: the word really does survive normalisation"
    assert v1.winglet_shape_key(ch) == "vertical"
    assert v1.winglet_shape_key(ch) in v1.winglet_shapes(ch)

    name, _ = v1.derive_problem(ch)
    flags = v1.winglet_flags(ch)
    assert api.unhonoured_flags(name, flags) == []
    built = api.PROBLEM_SPECS[name].build({}, flags, None)
    row = built.bounds[list(built.param_labels).index("winglet_cant_deg")]
    assert tuple(float(v) for v in row) == (-90.0, -84.0), name


@pytest.mark.parametrize("medium", ["air", "water", "track"])
def test_the_shape_a_session_holds_is_always_one_its_menu_offers(medium):
    """The invariant the select depends on, over every shape and medium."""
    for shape in list(v1.WINGLET_SHAPE_LABELS):
        ch = v1.start_choices(medium=medium)
        if shape not in v1.winglet_shapes(ch):
            continue
        v1.set_winglet_shape(ch, shape)
        v1.normalise_choices(ch, keep="winglets")
        assert v1.winglet_shape_key(ch) == shape, (medium, shape)
