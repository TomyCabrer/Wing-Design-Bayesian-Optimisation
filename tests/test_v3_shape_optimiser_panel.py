"""Stage 2's SHAPE OPTIMISATION card says what it found, and says it once.

Two things were wrong with it, and both were the card reporting rather than
the search:

* the ORIGINAL-vs-OPTIMISED table drew four columns and filled one. V3
  declared its fields ``base`` / ``opt`` / ``delta``; ``airfoil_compare_rows``
  returns ``metric`` / ``original`` / ``new`` / ``change``, so Quasar found
  nothing under three of the four names and the seed, the optimised value and
  the change were all blank.
* LIVE METRICS offered every numeric key the breakdown carried — 13 on a 2-D
  section and 46 the moment the section is scored on a wing — most of them
  the same number under a second name. A wall of checkboxes is not a menu.
"""

from __future__ import annotations

import pytest

from gui import metrics as mc
from gui import nice_app as v1


# ------------------------------------------------- the comparison table
def test_the_table_asks_for_the_fields_the_rows_carry():
    """The bug, stated as a rule: every column the card declares has to name
    a key the row dicts actually have, or it renders empty."""
    import re
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "gui/v3/stages/airfoil.py"
    text = src.read_text()
    block = text.split("v1.airfoil_compare_rows(rep", 1)[1][:900]
    fields = set(re.findall(r'"field":\s*"([a-z_]+)"', block))
    assert fields, "the compare table declares no fields at all"

    rows = v1.airfoil_compare_rows(_report())
    assert rows, "the fixture produced no comparison rows"
    assert fields <= set(rows[0]), (fields - set(rows[0]))


def _report() -> dict:
    """The shape of report ``api.optimize_airfoil`` hands the card — enough
    of it for ``airfoil_compare_rows`` to produce rows."""
    return {
        "baseline": {"breakdown": {"LD": 40.0, "CD": 0.0125, "CDi": 0.004,
                                   "CDp": 0.0085, "e": 0.94, "cm": -0.05,
                                   "tc": 0.12}},
        "design": {"breakdown": {"LD": 44.0, "CD": 0.0114, "CDi": 0.0038,
                                 "CDp": 0.0076, "e": 0.96, "cm": -0.04,
                                 "tc": 0.13}},
        "section": {"baseline": {"tc": 0.12, "tc_max_xc": 0.30,
                                 "polar": {"cd_at_cl_design": 0.0080,
                                           "alpha_at_cl_design": 2.5}},
                    "design": {"tc": 0.13, "tc_max_xc": 0.34,
                               "polar": {"cd_at_cl_design": 0.0071,
                                         "alpha_at_cl_design": 2.1}}},
    }


def test_every_row_carries_a_seed_an_optimised_value_and_a_change():
    for row in v1.airfoil_compare_rows(_report()):
        assert row["original"] and row["new"] and row["change"], row
        assert row["change"][0] in "+-", row


# ----------------------------------------------------- the live metrics
#: what the incumbent sampler really reports once the section is scored on a
#: wing (measured on ``api.design_report`` for the wing-mode section problem)
WING_MODE_KEYS = [
    "AR", "CD", "CD_counts", "CDi", "CDi_counts", "CDp", "CDp_counts", "CL",
    "LD", "LoD", "S", "S_llt", "S_ref", "a_per_rad", "alpha_L0_deg",
    "alpha_geo_max_deg", "alpha_root_deg", "b", "cd", "chord_dev",
    "chord_root_m", "chord_tip_m", "cl_design", "cl_y_max", "cl_y_min", "cm",
    "e", "f", "g_alpha", "g_chord", "g_cm", "g_tc", "g_twist", "mac",
    "mac_true", "n_branch", "n_clamped_low", "n_converged", "re", "re_true",
    "score", "taper", "taper_flown", "tc", "twist_env_deg", "twist_tip_deg",
]

#: ...and on a plain 2-D section
TWO_D_KEYS = ["alpha_deg", "cd", "cl_design", "cl_max_branch", "cm", "f",
              "g_cm", "g_tc", "n_branch", "n_converged", "r_le", "score",
              "tc"]


def test_the_shortlist_is_shorter_than_the_dump():
    short, rest = mc.live_metric_menu(WING_MODE_KEYS)
    assert len(short) < len(WING_MODE_KEYS) / 2
    # nothing is LOST, only ranked: every key the run reported is either
    # offered, a CONSTANT nobody wants a flat line of, or a second spelling
    # of an entry that IS offered
    alternates = {k for keys, _ in mc.LIVE_METRICS for k in keys}
    accounted = (set(k for k, _ in short) | set(rest)
                 | mc.LIVE_METRIC_NEVER | alternates)
    assert accounted >= set(WING_MODE_KEYS)


def test_one_entry_per_question():
    """The redundancies this exists to remove: the objective four times, every
    drag in coefficients AND counts, three spellings of the reference area."""
    short, rest = mc.live_metric_menu(WING_MODE_KEYS)
    keys = [k for k, _ in short]
    for family in (("LoD", "LD", "f", "score"),
                   ("CD", "CD_counts"), ("CDi", "CDi_counts", "CDi_total"),
                   ("CDp", "CDp_counts")):
        assert sum(k in family for k in keys) == 1, (family, keys)
    # ...and the same de-duplication holds in the 'rest' list: a second
    # spelling may be there, but never as a shortlist entry
    assert not set(keys) & set(rest)


def test_a_constant_is_never_offered_as_a_series():
    """A flat line and an evaluation counter are not results."""
    short, rest = mc.live_metric_menu(WING_MODE_KEYS)
    offered = set(k for k, _ in short) | set(rest)
    for key in ("cl_design", "re", "re_true", "S_ref", "n_converged",
                "n_branch"):
        assert key not in offered, key


def test_every_shortlist_entry_has_a_readable_name():
    short, _ = mc.live_metric_menu(WING_MODE_KEYS)
    for key, label in short:
        assert label and label != key, key


def test_the_defaults_are_three_and_open_on_the_objective():
    picked = mc.live_metric_defaults(WING_MODE_KEYS)
    assert 1 <= len(picked) <= 3
    assert picked[0] == "LoD"


def test_the_defaults_follow_the_family_that_spells_it_differently():
    """A 2-D section reports its objective as ``f``, not ``LoD``. A stored
    literal ``["LoD"]`` left that run's plot empty; the catalogue answers."""
    picked = mc.live_metric_defaults(TWO_D_KEYS)
    assert picked and picked[0] == "f"
    assert "cd" in picked                       # the drag it is made of


def test_the_two_d_menu_is_the_short_one_it_should_be():
    short, rest = mc.live_metric_menu(TWO_D_KEYS)
    assert len(short) <= 10
    assert set(k for k, _ in short) >= {"f", "cd", "cm", "tc", "r_le"}
    assert not rest                             # nothing left over to hide


def test_an_empty_sample_answers_rather_than_raising():
    """The panel asks this while drawing itself, before the first sample."""
    assert mc.live_metric_menu([]) == ((), ())
    assert mc.live_metric_defaults([]) == []


# ---------------------------------------------------- and on the real card
def test_the_card_opens_on_the_catalogues_defaults(monkeypatch):
    """The stage holds no literal list any more: what opens is what the
    catalogue says, filtered to what this run reported."""
    from gui.v3.app import assemble

    ctx = assemble("air")
    ctx.act("accept_mission")
    ctx.render("airfoil", "optimise")
    texts = [(getattr(e, "text", "") or "")
             for e in ctx.views[("airfoil", "optimise")].descendants()]
    # the panel is there and says what it samples
    assert any("sample the incumbent" in t for t in texts)
    # ...and the seed/optimised/change table's own note
    assert any("Live metrics" in t or "incumbent" in t for t in texts)


def test_the_stage_holds_no_hard_coded_metric_list():
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1]
           / "gui/v3/stages/airfoil.py").read_text()
    assert '"keys": ["LoD"]' not in src
    assert "metric_catalogue.live_metric_menu" in src
    assert "metric_catalogue.live_metric_defaults" in src


@pytest.mark.parametrize("key", ["LoD", "cd_counts", "cl_max"])
def test_the_named_defaults_exist_in_the_catalogue(key):
    """A default naming a key no LIVE_METRICS entry carries would silently
    never be picked."""
    assert any(key in keys for keys, _ in mc.LIVE_METRICS), key
