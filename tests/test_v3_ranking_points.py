"""The section ranking says WHERE each score came from, not only how big it is.

Stage 2's table ranked the library on a weighted composite of six criteria
and then showed two things: the score, and the raw metrics behind it. Between
them sat the number the user actually wanted — what each criterion CONTRIBUTED
to that score — and it was nowhere: a section could top the table on t/c while
its cruise L/D was mid-field, and the table read the same either way.

So every weighted criterion cell now carries ``weight x sub-score`` beside its
raw number, and the six terms add up to the score column. And a criterion at
zero weight is drawn DIM: it took no part in the order the rows are in, and a
column that ranked nothing must not look like one that did.

The tests below are the two contracts that can silently break: the terms have
to sum to the score the report published (not to a re-derived one), and the
cell fields the table's slots ask for have to be the fields the rows carry —
the failure that once left four columns of a card on this same stage blank.
"""

from __future__ import annotations

import pytest

from aerobo import api
from aerobo.airfoil_select import ScoreWeights, score_candidates
from gui.v3.stages import airfoil as stage


def _population(n: int = 6) -> list[dict]:
    """Screened records, spread over every criterion, no XFOIL involved.

    Only the RAW metric fields matter here: ``score_candidates`` re-derives
    eligibility and every sub-score from them.
    """
    return [dict(name=f"a{i}", tc=0.10 + 0.01 * i, cd_at=0.008 + 5e-4 * i,
                 cm_at=-0.02 - 0.01 * i, ldcr=60.0 + 5 * i,
                 clmax=1.2 + 0.1 * i, ldmax=80.0 + 4 * i, astall=10.0 + i,
                 eligible=True)
            for i in range(n)]


def _ranked(weights: ScoreWeights):
    """(rows as the report publishes them, normalised weights)."""
    ranked = score_candidates(_population(), weights, tc_min=0.0, cm_max=1e9)
    return [api._screen_row(r) for r in ranked], weights.normalised()


# ------------------------------------------------- the terms ARE the score
def test_the_six_terms_add_up_to_the_score_column():
    """The claim the cells make. Each is one criterion's share of the score,
    so their sum is the composite the report ranked on — which is the whole
    reason they can be read as "where this section's score came from"."""
    rows, w = _ranked(ScoreWeights())
    assert rows, "the fixture ranked nothing"
    for row in rows:
        pts = stage.weighted_points(row, w)
        total = sum(float(v) for v in pts.values() if v not in ("", "—"))
        # one decimal per cell, six cells: the display rounding is the only
        # gap allowed between the parts and the whole
        assert total == pytest.approx(row["composite"], abs=0.3), (
            f"{row['name']}: terms {pts} sum to {total}, score "
            f"{row['composite']}")


def test_a_term_moves_with_the_weight_it_is_priced_at():
    """Doubling a criterion's weight doubles what it puts in — the cell is a
    contribution, not the sub-score under a different name."""
    rows_a, w_a = _ranked(ScoreWeights(thick=0.10, clmax=0.20, ldmax=0.15,
                                       ldcr=0.35, cm=0.20, astall=0.0))
    rows_b, w_b = _ranked(ScoreWeights(thick=0.10, clmax=0.20, ldmax=0.15,
                                       ldcr=0.70, cm=0.20, astall=0.0))
    by_a = {r["name"]: r for r in rows_a}
    name = rows_a[0]["name"]
    a = float(stage.weighted_points(by_a[name], w_a)["pts_ldcr"])
    b = float(stage.weighted_points({r["name"]: r for r in rows_b}[name],
                                    w_b)["pts_ldcr"])
    ratio = (w_b["ldcr"] / w_a["ldcr"])
    if a == 0.0:
        pytest.skip("this section scored 0 on ldcr; nothing to scale")
    assert b == pytest.approx(a * ratio, rel=0.02)


def test_a_criterion_with_no_weight_shows_no_term():
    """Zero weight contributes exactly zero, and "+0.0" in every row is six
    columns of noise. The dimmed column is what says it counted for nothing;
    the cell says nothing at all."""
    rows, w = _ranked(ScoreWeights(thick=0.10, clmax=0.20, ldmax=0.15,
                                   ldcr=0.35, cm=0.20, astall=0.0))
    assert w["astall"] == 0.0
    for row in rows:
        assert stage.weighted_points(row, w)["pts_astall"] == ""
        # ...and a criterion that IS weighted still speaks
        assert stage.weighted_points(row, w)["pts_ldcr"] != ""


def test_an_unmeasured_sub_score_is_a_dash_not_a_zero():
    """"Not measured" and "contributed nothing" are different facts, and a
    row missing a sub-score must not be read as a section that scored zero
    on it."""
    rows, w = _ranked(ScoreWeights())
    row = dict(rows[0])
    row["scores"] = {k: v for k, v in row["scores"].items() if k != "clmax"}
    assert stage.weighted_points(row, w)["pts_clmax"] == "—"


def test_a_term_below_the_band_keeps_its_sign():
    """Sub-scores are not clipped to the frozen band, so a section outside it
    contributes NEGATIVE points — and the sign is the interesting part."""
    rows, w = _ranked(ScoreWeights())
    row = dict(rows[0], scores=dict(rows[0]["scores"], ldcr=-40.0))
    assert stage.weighted_points(row, w)["pts_ldcr"].startswith("-")


# -------------------------------------------------------- the dim columns
def test_the_unweighted_column_is_dimmed_and_the_weighted_one_is_not():
    w = ScoreWeights(thick=0.10, clmax=0.20, ldmax=0.15, ldcr=0.35,
                     cm=0.20, astall=0.0).normalised()
    cols = {c["name"]: c for c in stage.ranking_columns(stage.RANK_COLUMNS, w)}
    assert cols["astall"].get("style") == stage.UNWEIGHTED_STYLE
    assert cols["astall"].get("headerStyle") == stage.UNWEIGHTED_STYLE
    assert "style" not in cols["ldcr"]
    # the weight is ON the header, so "dim" is explained rather than mysterious
    assert "0.35" in cols["ldcr"]["label"]
    assert "0.00" in cols["astall"]["label"]


def test_a_column_that_is_not_a_criterion_carries_no_weight_at_all():
    """``cd @Cl`` is not weighted by anyone and never can be. Labelling it
    "w 0.00" would file it with a criterion the USER switched off, and
    dimming it would hide the column the trimming-surface note sends the
    reader to."""
    w = ScoreWeights().normalised()
    cols = {c["name"]: c for c in stage.ranking_columns(stage.RANK_COLUMNS, w)}
    for key in ("cd_at", "name", "rank", "composite"):
        assert "w " not in cols[key]["label"], key
        assert "style" not in cols[key], key


def test_every_priced_column_names_a_real_column_and_a_real_criterion():
    """The map is between two vocabularies that are spelled differently on
    purpose (``tc``/``thick``, ``cm_at``/``cm``), so both ends are checked."""
    columns = {k for k, _l, _n in stage.RANK_COLUMNS}
    criteria = {k for k, _l in api.SCREEN_METRICS}
    assert set(stage.CRITERION_OF_COLUMN) <= columns
    assert set(stage.CRITERION_OF_COLUMN.values()) == criteria


# --------------------------------------------------- the slots and the rows
class _FakeTable:
    def __init__(self):
        self.slots: dict[str, str] = {}

    def add_slot(self, name, template):
        self.slots[name] = template


def test_every_cell_slot_asks_for_a_field_the_rows_carry():
    """The blank-column bug, as a rule: a Quasar template that names a field
    no row has renders empty and looks exactly like a missing number."""
    rows, w = _ranked(ScoreWeights())
    fields = set(stage.weighted_points(rows[0], w))
    table = stage.points_slots(_FakeTable(), stage.RANK_COLUMNS)
    assert table.slots, "no criterion column got a cell template"
    for name, tpl in table.slots.items():
        key = name.replace("body-cell-", "")
        assert key in stage.CRITERION_OF_COLUMN
        assert f"props.row.pts_{key}" in tpl
        assert f"pts_{key}" in fields
        assert "__FIELD__" not in tpl and "__FAINT__" not in tpl


def test_only_the_criterion_columns_get_a_cell_template():
    """The score, the name and cd @Cl have nothing to add — a template on
    them would render a stray faint blank after every number."""
    table = stage.points_slots(_FakeTable(), stage.RANK_COLUMNS)
    assert set(table.slots) == {f"body-cell-{k}"
                                for k in stage.CRITERION_OF_COLUMN}


# ------------------------------------------------- the report with no weights
def test_a_report_that_names_no_weights_prices_nothing_and_dims_nothing():
    """An old saved run carries a ranking and no record of the weights that
    produced it. Reading that as "every weight is zero" would dim all six
    columns and say the order came from nothing — so the table falls back to
    the plain metrics it has always shown, and claims no prices."""
    rows, _w = _ranked(ScoreWeights())
    assert set(stage.weighted_points(rows[0], {}).values()) == {""}
    cols = stage.ranking_columns(stage.RANK_COLUMNS, {})
    assert not any("w " in c["label"] for c in cols)
    assert not any("style" in c for c in cols)
