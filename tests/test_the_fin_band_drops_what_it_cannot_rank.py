"""A criterion the screened population cannot separate has NO band.

THE DEFECT. A vertical stabiliser is screened at zero design lift, and the
screen's cruise-efficiency criterion is ``ldcr = |cl_design| / cd_at`` — so it
is identically 0.0 for every candidate a fin could ever be given. The measured
band came out ``(0.0, 0.0)``, ``ScoreReference`` refused the whole payload as
degenerate, and the shape optimiser turned that into

    ValueError: the composite objective needs a FROZEN normalisation band and
    none could be loaded (unusable band: degenerate reference band for ldcr:
    hi 0.0 <= lo 0.0)

on EVERY fin — including the shipped fin preset, which weights that criterion
at zero and never wanted a band for it. A second copy of the same defect
published the refused payload anyway: ``api.screen_at_point`` had already
fallen back to a live min-max for its own ranking, and still handed the band
it had rejected to the caller, which is where the shell picked it up.

THE RULE. A band covers what its population separates and DECLARES what it
does not (``unbanded``). Scoring then refuses a WEIGHT on an uncovered
criterion — loudly, and before the run pays for an XFOIL sweep — and ignores a
zero weight, because a criterion nobody weighted was never in J.

This is ``wing_score``'s rule, which drops a criterion that was constant
across its box sample, brought over to the screen.
"""

import json

import numpy as np
import pytest

from aerobo import airfoil_select as A
from aerobo import api


FIN_WEIGHTS = {"ldcr": 0.0, "ldmax": 0.0, "cm": 0.0,
               "clmax": 0.25, "cdcr": 0.35, "thick": 0.20, "astall": 0.20}


def _fin_records(n: int = 12) -> list[dict]:
    """A screened population at ZERO design lift: every section's ``ldcr`` is
    |0| / cd = 0, which is the whole point — the criterion is constant for
    physical reasons, not because the sample was small."""
    return [dict(name=f"sym{i}", path="", eligible=True, status="ok",
                 tc=0.09 + 0.004 * i, clmax=0.9 + 0.03 * i,
                 ldmax=55.0 + 1.5 * i, ldcr=0.0, astall=9.0 + 0.2 * i,
                 cm_at=-1e-4 * i, cd_at=0.0060 + 2e-4 * i,
                 alpha_at=0.0, clmax_censored=False)
            for i in range(n)]


def _fin_band(records=None) -> dict:
    return A.build_screen_reference(
        records if records is not None else _fin_records(),
        api.screen_weights(FIN_WEIGHTS), preset="",
        tc_min=0.05, cm_max=1.0, source="a fin screen at zero lift")


# ------------------------------------------------------------- the band
def test_the_band_omits_the_criterion_and_says_why():
    """Not (0.0, 0.0), and not silence: the payload names the criterion it
    could not band and states the measurement that decided it."""
    payload = _fin_band()

    assert "ldcr" not in payload["bounds"]
    assert set(payload["bounds"]) == set(A.CRITERIA) - {"ldcr"}
    why = payload["unbanded"]["ldcr"]
    assert "0" in why and "12" in why            # what was measured, over what
    # the raw measurement is still stated for it — that IS the evidence
    assert payload["raw_bounds"]["ldcr"] == [0.0, 0.0]


def test_the_fin_band_loads_where_the_degenerate_one_could_not():
    """The user-facing outcome: a fin's own measured band is a band."""
    ref, info = api._score_reference_arg(_fin_band())

    assert ref is not None, info.get("reason")
    assert ref.covers("cdcr") and not ref.covers("ldcr")
    assert ref.band("cdcr")[1] > ref.band("cdcr")[0]


def test_asking_the_band_for_the_criterion_it_dropped_is_an_error():
    """Never a fabricated band. The caller holding the weights decides what to
    do about a criterion that cannot be scored, so ``band`` refuses."""
    ref, _ = api._score_reference_arg(_fin_band())

    with pytest.raises(ValueError, match="no normalisation band for ldcr"):
        ref.band("ldcr")


# ------------------------------------------------- the run that was refused
def test_a_fin_shape_optimisation_is_no_longer_refused():
    """The reported failure, as the shell reaches it: fin weights, a fin's own
    measured band, the composite objective — and a config, not a ValueError."""
    flags = {"airfoil_objective": "composite",
             "airfoil_score_reference": _fin_band(),
             "airfoil_score_weights": dict(FIN_WEIGHTS)}

    name, ref, weights, _censored = api._airfoil_objective(flags)

    assert name == "composite"
    assert not ref.covers("ldcr")
    assert weights.normalised()["cdcr"] > 0.0


def test_the_run_config_carries_the_fin_band_to_the_solver():
    """A band that loads but does not ARRIVE is the flag-nobody-reads bug: read
    it back off the BUILT config, not off the payload that was passed in."""
    payload = _fin_band()
    cfg = api.airfoil_run_config(objective="composite",
                                 score_reference=payload,
                                 score_weights=dict(FIN_WEIGHTS))

    _name, ref, _w, _c = api._airfoil_objective(cfg.flags)

    assert ref.sha == payload["sha"]
    assert not ref.covers("ldcr")


def test_j_is_exactly_the_criteria_the_band_can_score():
    """A dropped criterion is not scored at zero — it is not in J at all, and
    the composite is the weighted sum of what is left."""
    payload = _fin_band()
    ref, _ = api._score_reference_arg(payload)
    w = api.screen_weights(FIN_WEIGHTS).normalised()
    recs = _fin_records()

    scored = A.score_candidates(recs, api.screen_weights(FIN_WEIGHTS),
                                tc_min=0.05, cm_max=1.0, reference=ref)

    top = scored[0]
    assert "score_ldcr" not in top              # no sub-score for it
    by_hand = sum(w[k] * top[f"score_{k}"] for k in payload["bounds"])
    assert top["composite"] == pytest.approx(by_hand)


# ------------------------------------------------------ the weight refusal
def test_a_weight_on_the_dropped_criterion_is_refused_before_the_run():
    """A user who leaves the wing preset's 0.35 on cruise L/D is searching
    against a number identical for every section. That is refused where a
    config is still a promise the run is launchable — not after the seed's
    XFOIL sweeps have been paid for."""
    flags = {"airfoil_objective": "composite",
             "airfoil_score_reference": _fin_band(),
             "airfoil_score_weights": dict(FIN_WEIGHTS, ldcr=0.35)}

    with pytest.raises(ValueError) as exc:
        api._airfoil_objective(flags)

    msg = str(exc.value)
    assert "ldcr" in msg
    assert "cdcr" in msg                        # …and what to weight instead


def test_scoring_refuses_the_same_pair_the_config_does():
    """The backstop. A band and a weight set can also meet inside a run that
    did not come through the config (a re-score, a saved study), and scoring a
    criterion nobody can rank as zero would read as "it did badly"."""
    ref, _ = api._score_reference_arg(_fin_band())

    with pytest.raises(ValueError, match="ldcr"):
        A.score_candidates(_fin_records(),
                           api.screen_weights(dict(FIN_WEIGHTS, ldcr=0.35)),
                           tc_min=0.05, cm_max=1.0, reference=ref)


def test_a_stale_sub_score_does_not_survive_a_re_score():
    """``score_candidates`` mutates the records it is given, so a record
    scored once on a band that DID cover the criterion must not keep that
    sub-score when it is re-scored on a band that does not."""
    recs = _fin_records()
    A.score_candidates(recs, api.screen_weights(FIN_WEIGHTS),
                       tc_min=0.05, cm_max=1.0)         # live min-max: all 7
    assert "score_ldcr" in recs[0]

    ref, _ = api._score_reference_arg(_fin_band())
    A.score_candidates(recs, api.screen_weights(FIN_WEIGHTS),
                       tc_min=0.05, cm_max=1.0, reference=ref)

    assert "score_ldcr" not in recs[0]


# --------------------------------------------------- what did NOT change
def test_the_shipped_band_still_covers_every_criterion():
    """The exemption is for a population that could not separate a criterion,
    never a licence for a band to be short. The shipped library band is
    unchanged: it covers all seven and declares nothing unbanded."""
    lib = A.load_screen_reference()

    assert all(lib.covers(k) for k in A.CRITERIA)
    assert not (lib.unbanded or {})
    for k in A.CRITERIA:
        lo, hi = lib.band(k)
        assert hi > lo


def test_a_truncated_payload_is_still_refused():
    """A missing band has to be DECLARED. Deleting a criterion from a payload
    without saying so is a corrupt file, and it still raises."""
    payload = _fin_band()
    payload["bounds"].pop("clmax")

    ref, info = api._score_reference_arg(payload)

    assert ref is None
    assert "clmax" in info["reason"]


def test_a_payload_cannot_both_band_and_disown_a_criterion():
    """Two answers to one question is a payload nobody can score against."""
    with pytest.raises(ValueError, match="bands and declares unbanded"):
        A.ScoreReference(bounds={k: (0.0, 1.0) for k in A.CRITERIA},
                         unbanded={"cm": "measured constant"})


def test_a_band_that_separates_nothing_is_still_refused():
    """The exemption drops criteria, never the whole map: a population that
    separates none of them has nothing to normalise against and says so."""
    flat = [dict(r, tc=0.12, clmax=1.2, ldmax=60.0, ldcr=0.0, astall=10.0,
                 cm_at=0.0, cd_at=0.007) for r in _fin_records()]

    with pytest.raises(ValueError, match="nothing to normalise against"):
        _fin_band(flat)


# ------------------------------------------- the band nothing was scored on
def test_a_table_ranked_on_a_live_min_max_publishes_no_frozen_band(monkeypatch):
    """The second half of the defect. ``screen_at_point`` falls back to a live
    min-max when the measured band will not load — and used to publish that
    same refused payload for the caller to score its optimisation on. A band
    the table was NOT scored on is not the table's band."""
    bad = {"version": A.REFERENCE_VERSION, "bounds": {"ldcr": [0.0, 0.0]},
           "sha": "deadbeef", "n_records": 3}
    ranked = [{"name": "a", "composite": 1.0}, {"name": "b", "composite": 0.5}]

    monkeypatch.setattr(api, "screen_library_point",
                        lambda: {"re": 1.0e6, "mach": 0.0})

    def fake_screen(weights=None, **kw):
        out = {"ranked": list(ranked), "n_eligible": len(ranked),
               "conditions": {}, "n_screened": len(ranked), "point": {},
               "wall_time_s": 0.0}
        if kw.get("measure_reference"):
            out["reference"] = dict(bad)
        return out

    monkeypatch.setattr(api, "screen_airfoils", fake_screen)

    out = api.screen_at_point(re=3.0e5, cl_design=0.0)

    assert "reference" not in out
    assert "live min-max" in out["reference_error"]


def test_a_band_that_loads_is_published(monkeypatch):
    """…and the honest case still hands the band on, or the fix would have
    cured the refusal by removing the feature."""
    good = _fin_band()
    ranked = [{"name": "a", "composite": 1.0}, {"name": "b", "composite": 0.5}]

    monkeypatch.setattr(api, "screen_library_point",
                        lambda: {"re": 1.0e6, "mach": 0.0})

    def fake_screen(weights=None, **kw):
        out = {"ranked": list(ranked), "n_eligible": len(ranked),
               "conditions": {}, "n_screened": len(ranked), "point": {},
               "wall_time_s": 0.0}
        if kw.get("measure_reference"):
            out["reference"] = json.loads(json.dumps(good))
        return out

    monkeypatch.setattr(api, "screen_airfoils", fake_screen)

    out = api.screen_at_point(re=3.0e5, cl_design=0.0)

    assert out["reference"]["sha"] == good["sha"]
    assert "reference_error" not in out


# ------------------------------------------------------- what a shell shows
def test_the_exchange_rate_table_names_the_criterion_it_cannot_price():
    """A rate table missing a row the weights panel shows is the silence the
    zero-weight rule exists to break, so the row stays and carries the reason
    instead of a number."""
    rates = api.score_exchange_rates(dict(FIN_WEIGHTS), _fin_band())

    assert set(rates) == set(A.CRITERIA)
    assert rates["ldcr"]["per_step"] is None
    assert rates["ldcr"]["unbanded"]
    assert rates["cdcr"]["per_step"] > 0.0
    assert not rates["cdcr"]["unbanded"]


def test_the_reference_point_carries_no_goal_the_band_cannot_score():
    """The goal composite floors the seed's own criteria. One the band cannot
    score has no floor to be held to — a goal in raw units is read in
    sub-score points, and there is no map for it."""
    ref, _ = api._score_reference_arg(_fin_band())
    scores = {k: 50.0 for k in ref.bounds}
    goals = A.ScoreGoals(goals={k: 1.0 for k in ref.bounds}, scores=scores)

    out = A.goal_shortfalls(scores, goals, ref,
                            api.screen_weights(FIN_WEIGHTS))

    assert "ldcr" not in out["rows"]
    assert np.isfinite(out["penalty"])


def test_a_front_axis_the_band_cannot_score_is_refused_too():
    """A front's axis is not a weight. The pareto objective searches named
    criteria as a VECTOR, so one with no band is an axis with no coordinate —
    every evaluation comes back unscoreable and the run reads as "the search
    found nothing" rather than as the refusal it is."""
    flags = {"airfoil_objective": "pareto",
             "airfoil_score_reference": _fin_band(),
             "airfoil_score_weights": dict(FIN_WEIGHTS),
             "airfoil_pareto_criteria": list(A.PARETO_CRITERIA)}

    with pytest.raises(ValueError) as exc:
        api._airfoil_objective(flags)

    assert "ldcr" in str(exc.value) and "front axis" in str(exc.value)


def test_a_front_on_axes_the_band_covers_still_builds():
    """…and the refusal is about the axis, not about the surface: a front over
    criteria this population separates is a run."""
    flags = {"airfoil_objective": "pareto",
             "airfoil_score_reference": _fin_band(),
             "airfoil_score_weights": dict(FIN_WEIGHTS),
             "airfoil_pareto_criteria": ["cdcr", "clmax", "thick"]}

    name, ref, _w, _c = api._airfoil_objective(flags)

    assert name == "pareto" and ref.covers("cdcr")
