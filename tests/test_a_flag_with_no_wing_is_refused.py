"""Seven flags that configure a wing are refused when there is no wing.

``AIRFOIL_WING_FLAG_KEYS`` names eight keys. Seven of them configure the 3-D
wing a section is judged on; the eighth, ``airfoil_wing``, is the wing. State
one of the seven without the eighth and ``_section_wing_kwargs`` returned
``None``, discarding them — while ``accepted_flags`` had already said yes.

Measured: on ``airfoil (section)``, whose trim attitude is
``alpha_deg = 3.4796``, stating ``airfoil_alpha_max_deg`` at 3.0, 2.0, 1.0 and
even 0.0 all returned a bit-identical ``score = -0.006622871125611746`` with
``reason = ''`` — a gate that could not fire in either direction, at any
setting, silently.

The function's own docstring states the opposite policy for keys INSIDE the
block ("Unknown wing keys are rejected rather than ignored: a typo'd ``area``
silently falling back to the default would move the derived CL_design without
saying so"). This file extends that to the keys that need it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aerobo import api                                           # noqa: E402

_ORPHANS = [k for k in api.AIRFOIL_WING_FLAG_KEYS if k != "airfoil_wing"]


@pytest.mark.parametrize("key", _ORPHANS)
def test_each_dependent_key_alone_is_refused(key):
    """All seven, by name — one that slipped through would be a gate that
    cannot fire."""
    value = 1 if key.endswith("_order") else (
        "power" if key in ("airfoil_re_strip", "airfoil_re_bank") else 1.0)
    with pytest.raises(ValueError, match="read by nothing"):
        api._section_wing_kwargs({key: value})


def test_the_same_key_is_accepted_with_a_wing_block():
    """The control: the refusal is about the MISSING wing, not the key."""
    got = api._section_wing_kwargs({"airfoil_wing": {"mass_kg": 5.0},
                                    "airfoil_alpha_max_deg": 1.0})
    assert isinstance(got, dict)
    assert got["alpha_max_deg"] == pytest.approx(1.0)


def test_no_flags_and_no_wing_stay_in_the_pure_section_mode():
    """Empty in, ``None`` out — the 2-D section mode is not a wing and must
    not start raising."""
    assert api._section_wing_kwargs({}) is None
    assert api._section_wing_kwargs(None) is None
    assert api._section_wing_kwargs({"wing_objective": "lod"}) is None


def test_the_message_names_the_key_and_the_cure():
    """A refusal a user cannot act on is a crash with better manners."""
    with pytest.raises(ValueError) as exc:
        api._section_wing_kwargs({"airfoil_re_strip": "power"})
    text = str(exc.value)
    assert "airfoil_re_strip" in text
    assert "airfoil_wing" in text
