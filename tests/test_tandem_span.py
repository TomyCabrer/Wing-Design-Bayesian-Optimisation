"""A tandem's two wings share a fuselage, not a SPAN.

Both tandem solvers were written with one ``b`` for the pair: the rear wing
was built at the front wing's span, always, and there was no way — typed or
searched — to fly the pair whose second wing is the narrower one. That is a
modelling shortcut, not a property of the aeroplane, and it hid the trade the
family exists to study: the rear wing sits in the front wing's downwash, and
HOW MUCH of it depends on how far out its tip reaches (a rear wing well
inboard of the front tip vortices sees a different field from one that
matches them).

Two ways to answer it, and the split is the package's usual one:

* PLANFORM FIXED — the rear span is a VALUE, in metres, beside the stagger
  (``b_rear_m``, api.TANDEM_SPAN_KEYS). Stated it travels verbatim; blank the
  rear wing is as wide as the front one, so every published pair is
  bit-for-bit what it was.
* PLANFORM FREE — both spans are DESIGN VARIABLES, one row per wing
  (``b_m``, ``b_rear_m`` — sizing.span_labels), and the typed value is
  stripped from the flags, because a wing is sized one way.

Plus the physics that has to follow the span wherever it comes from: the
lifting lines are sampled on their own stations, each wing weighs and is
stressed as the wing it is, the tip devices are sized on their own
semi-spans, and Munk's stagger theorem — the family's validation gate — still
holds exactly when the two wings are not the same width.
"""

import numpy as np
import pytest

from aerobo import api, sizing
from aerobo import tandem as td
from aerobo import tandemvlm as tvm

PAIR = "tandem (nonplanar) + winglets"
SIZED = "tandem + free planform"
SIZED_VLM = "tandem (nonplanar) + winglets + free planform"

X_PLANAR = np.array([0.6, 0.6, 0.0, -1.0, -2.0, 0.0, -1.0, -2.0, 0.5, 0.0])
X_VLM = np.array([0.6, 0.6, 0.0, -2.0, 0.0, -2.0, 0.5, 0.0,
                  0.12, 90.0, 0.12, 90.0])


def _built(name, **flags):
    return api.PROBLEM_SPECS[name].build({}, dict(flags), None)


def _mid(built):
    return 0.5 * (built.bounds[:, 0] + built.bounds[:, 1])


# ------------------------------------------ 1. the rear span is a real span

def test_an_untouched_pair_flies_one_span_on_both_wings():
    """The published pair, bit-for-bit: nothing is stated, so the rear wing
    is as wide as the front one and says so."""
    for prob, x, evaluate in ((td.TandemProblem(), X_PLANAR,
                               td.evaluate_tandem),
                              (tvm.TandemVLMProblem(winglets=True), X_VLM,
                               tvm.evaluate_tandem_vlm)):
        assert prob.b_rear is None
        assert prob.b_r == prob.b
        out = evaluate(x, prob)
        assert out["feasible"], out["reason"]
        assert out["b_front"] == out["b_rear"] == pytest.approx(prob.b)


@pytest.mark.parametrize("factory,x,evaluate", [
    (td.TandemProblem, X_PLANAR, td.evaluate_tandem),
    (lambda **kw: tvm.TandemVLMProblem(winglets=True, **kw), X_VLM,
     tvm.evaluate_tandem_vlm),
])
def test_a_stated_rear_span_is_the_span_the_rear_wing_flies(factory, x,
                                                            evaluate):
    """Not a label on the report: the rear surface is narrower, its area is
    the pair's own split, so it flies at a LOWER aspect ratio and pays the
    induced drag of one."""
    base = evaluate(x, factory())
    short = evaluate(x, factory(b_rear=7.0))
    assert base["feasible"] and short["feasible"], short.get("reason")
    assert short["b_rear"] == pytest.approx(7.0)
    assert short["b_front"] == pytest.approx(10.0)
    # the same area on less span: a stubbier wing, and a worse one here
    assert short["S_rear"] == pytest.approx(base["S_rear"], rel=1e-6)
    assert short["LoD"] < base["LoD"]


def test_the_planar_pair_samples_each_lifting_line_on_its_own_stations():
    """A lifting line's stations belong to that line. Sampling the rear wing
    on the front wing's y put its tip station outboard of its own tip."""
    out = td.evaluate_tandem(X_PLANAR, td.TandemProblem(b_rear=6.0))
    assert out["feasible"], out["reason"]
    res = out["tandem"]
    assert 2.0 * np.max(np.abs(res.rear.y)) < 6.0
    assert 2.0 * np.max(np.abs(res.rear.y)) > 0.9 * 6.0
    assert 2.0 * np.max(np.abs(res.front.y)) > 9.0
    # ...and the reported aspect ratios are the two wings' own
    assert res.rear.AR == pytest.approx(36.0 / res.rear.S, rel=1e-6)
    assert res.front.AR == pytest.approx(100.0 / res.front.S, rel=1e-6)


def test_the_nonplanar_pair_panelises_the_rear_wing_at_its_own_width():
    out = tvm.evaluate_tandem_vlm(
        X_VLM, tvm.TandemVLMProblem(winglets=True, b_rear=6.0))
    assert out["feasible"], out["reason"]
    res = out["vlm"]
    rear, wl = res.is_second, res.is_winglet
    panel = rear & ~wl
    assert 2.0 * np.max(np.abs(res.y[panel])) <= 6.0 + 1e-9
    assert 2.0 * np.max(np.abs(res.y[~rear & ~wl])) > 9.0
    # the rear wing's TIP DEVICE is a fraction of ITS OWN semi-span, so it is
    # shorter than the front one's at the same h_frac
    z_dev_rear = np.max(res.z[rear & wl]) - out["dz"]
    assert z_dev_rear == pytest.approx(0.12 * 3.0, rel=0.25)
    assert np.max(res.z[~rear & wl]) > z_dev_rear


def test_a_rear_span_that_is_not_a_length_is_refused_where_it_is_stated():
    for factory in (td.TandemProblem, tvm.TandemVLMProblem):
        with pytest.raises(ValueError, match="positive length"):
            factory(b_rear=0.0)
        with pytest.raises(ValueError, match="positive length"):
            factory(b_rear=-3.0)


def test_munk_still_holds_when_the_two_wings_are_not_the_same_width():
    """The family's validation gate, re-run on the asymmetric pair: total
    mutual induced drag is independent of streamwise stagger for FIXED
    circulations, exactly, whatever the two spans are."""
    from aerobo.llt import cosine_stations
    from aerobo.tandem import VortexSystem, mutual_cdi

    N, b_f, b_r = 40, 10.0, 6.5
    y_f = cosine_stations(N, b_f)[1]
    y_r = cosine_stations(N, b_r)[1]
    Gam_f = np.sqrt(np.clip(1.0 - (2 * y_f / b_f) ** 2, 0.0, None))
    Gam_r = 0.7 * np.sqrt(np.clip(1.0 - (2 * y_r / b_r) ** 2, 0.0, None))
    sums, rears = [], []
    for dx in (0.0, 1.0, 5.0, 50.0):
        a = VortexSystem(b=b_f, N=N, x=0.0, z=0.0)
        c = VortexSystem(b=b_r, N=N, x=dx, z=1.5)
        cdi_a, cdi_b = mutual_cdi(a, c, Gam_f, Gam_r, V=1.0, Sref=20.0)
        sums.append(cdi_a + cdi_b)
        rears.append(cdi_b)
    sums = np.asarray(sums)
    assert (sums.max() - sums.min()) / abs(sums.mean()) < 1e-12
    assert (max(rears) - min(rears)) > 0.1 * abs(sums.mean())   # teeth


# ------------------------------------------------ 2. sized: a row per wing

@pytest.mark.parametrize("name,evaluate,family_block", [
    (SIZED, td.evaluate_tandem, X_PLANAR),
    (SIZED_VLM, tvm.evaluate_tandem_vlm, X_VLM),
])
def test_the_sized_pair_searches_a_span_per_wing(name, evaluate,
                                                 family_block):
    built = _built(name)
    lab = list(built.param_labels)
    assert lab[-3:] == list(sizing.span_labels(2)) + ["S_m2"]
    assert lab.count("b_m") == 1 and lab.count("b_rear_m") == 1
    # the published pair AT its own size: two 10 m wings over 20 m² total
    x = np.concatenate([family_block, [10.0, 10.0, 20.0]])
    base = evaluate(x, built.problem)
    assert base["feasible"], base["reason"]
    # move ONLY the rear span: the answer, and the structure, follow it
    xx = np.asarray(x, dtype=float).copy()
    xx[lab.index("b_rear_m")] = 0.6 * float(x[lab.index("b_m")])
    short = evaluate(xx, built.problem)
    assert short["feasible"], short["reason"]
    assert short["b_rear"] < short["b_front"]
    assert short["b_front"] == pytest.approx(base["b_front"])
    assert short["W_wing_N"] < base["W_wing_N"]     # a shorter spar weighs less
    assert short["score"] != base["score"]


@pytest.mark.parametrize("name,evaluate", [
    (SIZED, td.evaluate_tandem),
    (SIZED_VLM, tvm.evaluate_tandem_vlm),
])
def test_the_rear_span_obeys_the_aspect_ratio_band_too(name, evaluate):
    """The band is what makes these solvers honest, and it is asked of the
    wing the row belongs to: a rear span at the top of its box against a
    small area is a 40+ AR wing and is REFUSED, not scored."""
    built = _built(name)
    lab = list(built.param_labels)
    x = _mid(built)
    x[lab.index("S_m2")] = float(built.bounds[lab.index("S_m2")][0])
    x[lab.index("b_rear_m")] = float(built.bounds[lab.index("b_rear_m")][1])
    out = evaluate(x, built.problem)
    assert not out["feasible"]
    assert "aspect ratio" in out["reason"]


def test_the_two_span_rows_open_on_the_same_band():
    """They are the same question asked twice — opening the rear wing's row
    narrower would answer it for the search."""
    built = _built(SIZED)
    lab = list(built.param_labels)
    front = built.bounds[lab.index("b_m")]
    rear = built.bounds[lab.index("b_rear_m")]
    assert np.allclose(front, rear)


# ------------------------------------------------------------ 3. the registry

def test_the_rear_span_is_a_flag_where_it_is_typed_and_a_row_where_it_is_not():
    for name in ("tandem", PAIR):
        assert set(api.TANDEM_SPAN_KEYS) <= set(api.PROBLEM_SPECS[name].flags)
        assert "b_rear_m" not in api.PROBLEM_SPECS[name].param_labels
    for name in (SIZED, SIZED_VLM):
        spec = api.PROBLEM_SPECS[name]
        # a wing is sized ONE way: the typed span leaves the flags exactly
        # where the searched one enters the vector
        assert not set(api.TANDEM_SPAN_KEYS) & set(spec.flags), name
        assert "b_rear_m" in spec.param_labels, name


def test_a_typed_rear_span_reaches_the_problem_verbatim():
    for name in ("tandem", PAIR):
        prob = _built(name, b_rear_m=7.25).problem
        assert prob.b_rear == pytest.approx(7.25)
        assert prob.b == 10.0            # the front wing is untouched
        assert prob.b_r == pytest.approx(7.25)
    # ...and it is a size, so it is checked as one, at build time
    with pytest.raises(ValueError, match="aspect ratio"):
        _built("tandem", b_rear_m=2.0)
    with pytest.raises(ValueError, match="aspect ratio"):
        _built("tandem", b_rear_m=40.0)


def test_stating_only_the_rear_span_leaves_the_rest_of_the_layout_published():
    """"the back wing is 7 m" is a complete sentence: it says nothing about
    the front wing, the area or the stagger, and none of them move."""
    prob = _built("tandem", b_rear_m=7.0).problem
    assert (prob.b, prob.S_total) == (10.0, 20.0)
    assert (prob.dx, prob.dz) == (5.0, 1.0)


# ------------------------------------------------------------- 4. the shells

def test_the_card_sends_the_rear_span_only_where_it_can_be_flown():
    from gui import nice_app as v1

    air = {"system": "tandem", "medium": "air"}
    assert v1.tandem_flags(air) == {}
    stated = {**air, "tandem_b_rear_m": 7.0}
    assert v1.tandem_flags(stated) == {"b_rear_m": 7.0}
    # a single wing, or a medium with no second surface, sends nothing
    assert v1.tandem_flags({**stated, "system": "single"}) == {}
    assert v1.tandem_flags({**stated, "medium": "water"}) == {}
    # ...and neither does a pair that SEARCHES its spans: the registry strips
    # the flag there, so sending it would be a number silently dropped
    assert v1.tandem_flags(stated, SIZED) == {}
    assert "b_rear_m" not in v1.tandem_flags({**stated, "planform": "free"})


def test_v3_asks_the_rear_span_on_the_size_card_and_sends_it():
    """WHERE it is asked moved (tests/test_tandem_two_spans.py): a span is a
    span whichever wing carries it, so V3 asks both of the pair's together on
    the size card instead of one there and one beside the stagger. V1 and V2
    have no such card and keep the field where it was."""
    from gui.v3 import config
    from gui.v3.app import assemble

    ctx = assemble()
    ctx.act("accept_mission")
    ctx.act("set_choice", "system", "tandem")
    ctx.render("wing", "type")
    labels = [str(getattr(e, "text", "")) for e in
              ctx.views[("wing", "type")].descendants()]
    assert "span b — rear wing" in labels
    assert "span b — front wing" in labels
    assert not any("Rear wing span" in t for t in labels)   # asked once
    assert any("Rear wing, aft by" in t for t in labels)    # the stagger stays

    ctx.act("set_choice", "tandem_b_rear_m", 7.5)
    flags = config.flags(ctx.S)
    assert flags.get("b_rear_m") == pytest.approx(7.5)
    prob = _built(ctx.S["wing"]["problem"],
                  **{k: v for k, v in flags.items()
                     if k in api.TANDEM_SPAN_KEYS}).problem
    assert prob.b_r == pytest.approx(7.5)


def test_v3_puts_both_span_bands_at_the_top_of_the_design_box():
    """Free the planform and the question moves from the card to the box —
    one bar per wing, and they are read together."""
    from gui.v3 import config
    from gui.v3.app import assemble

    ctx = assemble()
    ctx.act("accept_mission")
    ctx.act("set_choice", "system", "tandem")
    ctx.act("set_planform", "free")
    assert "free planform" in ctx.S["wing"]["problem"]
    eff = config.effective_bounds(ctx.S)
    assert "b_m" in eff and "b_rear_m" in eff
    from gui.v3.stages import wing as wing_stage
    assert wing_stage.SPAN_ROWS == ("b_m", "b_rear_m")
    # ...and the card stops asking for a number it is now searching
    ctx.render("wing", "type")
    labels = [str(getattr(e, "text", "")) for e in
              ctx.views[("wing", "type")].descendants()]
    assert not any("Rear wing span" in t for t in labels)


# --------------------------------------------------------- 5. what is drawn

def test_the_report_hands_the_rear_wing_its_own_span():
    """A rear surface lofted against the FRONT wing's span is stretched to
    the wrong width and given its tip twist somewhere past its tip
    (cad.second_wing_twist)."""
    from aerobo import cad

    cfg = api.RunConfig(problem_name=PAIR, flags={"b_rear_m": 6.5})
    rep = api.design_report(cfg, X_VLM)
    geom = rep["geometry"]
    blk = geom.get("second_surface")
    assert isinstance(blk, dict)
    assert blk["b"] == pytest.approx(6.5)
    assert geom["b"] == pytest.approx(10.0)         # still the front wing's

    labels = rep["param_labels"]
    y = np.array([0.0, 3.25])
    tw_own = cad.second_wing_twist(geom, y, X_VLM, labels, b=blk["b"])
    tw_front = cad.second_wing_twist(geom, y, X_VLM, labels)
    # at its own tip the rear wing gets its stated tip twist; read on the
    # front wing's span it is only part of the way there
    assert abs(tw_own[1]) > abs(tw_front[1])
    assert np.rad2deg(tw_own[1]) == pytest.approx(X_VLM[5] + X_VLM[7],
                                                  rel=1e-9)
