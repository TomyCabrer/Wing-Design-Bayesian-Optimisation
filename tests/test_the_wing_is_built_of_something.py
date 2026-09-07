"""A material has to reach the WEIGHT, or the menu is a placebo.

``weights.py`` carried exactly one material assumption — 2024-T3's allowable,
in the stress constraint — and the wing WEIGHT, a statistical GA regression,
had no material argument at all. Measured before ``materials.py`` existed, on
a 2-D (b, S) grid of "trim wing + free planform + free chord law":

    sigma_allow 230 -> 500 MPa    b* 13.00 -> 13.50 m, score +1.0 %
    sigma_allow 500 -> infinity   no change whatever
    k_w         1.00 -> 0.87      score +12.7 %

    d ln f / d ln k_w   ~ -1.0        d ln f / d ln sigma ~ +0.01

So a material control wired only to the allowable would be a control that
moves nothing, and this file's job is to keep the material attached to the
half that matters. Four things are gated:

1. the two ANCHORS hold exactly (aluminium 1.000, carbon prepreg 0.87);
2. eta cannot be dropped — no phi in [0, 1] reproduces the empirical
   composite factor, which is the whole reason the model has a second
   parameter rather than being naive rho/sigma;
3. a default problem is bit-for-bit the published one;
4. the flag is honoured where a wing is weighed and REFUSED where it is not.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aerobo import api, materials, sizing, weights     # noqa: E402

#: the family the numbers in the docstring were measured on
SIZED = "trim wing + free planform + free chord law"

#: ...and a family that weighs nothing: fixed size, so no W_wing and no
#: sigma_root anywhere in its breakdown
UNSIZED = "wing (free chord law)"


def _centre(name: str, flags: dict | None = None) -> dict:
    spec = api.PROBLEM_SPECS[name]
    prob = spec.build({}, flags or {}, None)
    lo, hi = np.array(prob.bounds).T
    return prob.evaluate((lo + hi) / 2.0)


# --------------------------------------------------------------- the anchors

def test_the_reference_material_is_exactly_one():
    """Not approximately: ``==``.

    Raymer's correlation was regressed over aluminium aeroplanes, so 2024-T3
    is the material the whole package is already calibrated at. If its factor
    were 0.9999999 every stored sized run would move in the last digits and
    every regression fixture in this suite would need a tolerance it does not
    have.
    """
    assert materials.resolve(None).key == materials.REFERENCE_KEY
    assert materials.resolve(None).k_w == 1.0
    assert materials.get(materials.REFERENCE_KEY).naive == 1.0


def test_the_carbon_anchor_is_where_eta_was_fitted():
    """The one empirical number the extrapolation hangs on.

    ``ETA`` is solved from this point rather than written down, so the test is
    a tripwire on the arithmetic — it does NOT verify the 0.87 itself, which
    is Raymer's composite weight-savings factor recalled and not checked
    against a primary source in this repo (materials.py says so out loud).
    """
    m = materials.get(materials.ETA_ANCHOR_KEY)
    assert m.k_w == pytest.approx(materials.ETA_ANCHOR_K_W, abs=1e-12)
    assert 0.85 <= m.k_w <= 0.90


def test_no_split_of_the_wing_reaches_the_empirical_factor():
    """WHY the model needs eta at all — the finding, asserted.

    Give carbon full credit for its strength (phi = 1) and it should weigh
    0.26 of an aluminium wing; give it none at all (phi = 0) and density
    alone still says 0.575. Real composite wings weigh ~0.87. So the missing
    physics is not the CAP/SKIN SPLIT — it is that a composite part is not
    thickness-equivalent to an aluminium one (minimum gauge, ply drops,
    buckling-driven skins, joints, damage tolerance). A naive rho/sigma
    material menu is optimistic by about a factor of two, and this is the
    line that stops anyone deleting eta as a fudge.
    """
    m = materials.get(materials.ETA_ANCHOR_KEY)
    best = max(materials.naive_ratio(m.rho_cap_kgm3, m.sigma_allow_Pa,
                                     m.rho_skin, phi=phi)
               for phi in np.linspace(0.0, 1.0, 101))
    assert best < materials.ETA_ANCHOR_K_W, (
        f"the most pessimistic phi still gives {best:.3f} against the "
        f"empirical {materials.ETA_ANCHOR_K_W}; if this ever passes, eta has "
        f"stopped being necessary and the model should be re-derived")
    assert 0.15 < materials.ETA < 0.35


# ---------------------------------------------------- nothing published moved

#: The solve's own floating-point noise, in ulps. The lattice solve is a
#: dense linear system and a threaded BLAS does not fix the reduction order,
#: so the same problem at the same point returns several values spanning a
#: few ulp — measured on `tail + winglet`: 3 distinct scores over 60
#: evaluations at 4 threads, exactly 1 at OMP_NUM_THREADS=1.
SOLVER_ULP_NOISE = 8


def _within_noise(got: float, want: float) -> bool:
    import math
    return abs(got - want) <= SOLVER_ULP_NOISE * math.ulp(abs(want))


def test_a_problem_with_no_material_is_the_published_one():
    """The default IS the calibration, to the solver's own noise floor.

    The two numbers are a REGRESSION ANCHOR, not a physical claim: re-derive
    them by building this spec with empty flags and evaluating the centre of
    its own box. They are pinned because the whole safety argument for adding
    a material is that ``k_w = 1.0`` multiplies exactly and changes nothing.

    They were pinned with ``==`` when this was written, and that was wrong —
    not because the material factor is inexact (it is: see
    :func:`test_the_weight_correlation_multiplies_exactly`, which still uses
    ``==`` because it is scalar arithmetic with no solve in it) but because
    the SOLVE underneath them is not reproducible to the last bit. An
    equality here was testing the BLAS thread schedule. The rule, agreed
    across both sessions working this tree: ARRAYS EXACT, FORCES TO A STATED
    FLOOR.

    RE-DERIVED 2026-08-31, AND THE OLD PAIR NEVER MATCHED THIS TREE. The two
    numbers below replace 47.50627326777426 / 14.883533548076356, which were
    introduced by ``86aa727`` — the same commit that introduced this file —
    and which fail AT that commit, at ``9c37b96`` and at ``5935e0d`` with the
    identical observed value. So this is not a regression dated to today: the
    anchor was born mismatched, which is exactly what a number captured by
    hand mid-session does.

    The gap was 2.92e-7 relative on BOTH numbers — coherent, not noise, and
    about 2e9 ulp against the 8-ulp floor above. It is not the solver wobble
    that floor was written for: measured bit-identical over five repeat
    builds, over ``OMP_NUM_THREADS`` 1/2/4/8, and again in a worktree with an
    empty XFOIL cache. Re-derived the way the paragraph above says to, by
    building the spec with empty flags and evaluating the centre of its own
    box — so the next reader can do the same rather than trust this line.
    """
    out = _centre(SIZED)
    assert _within_noise(out["LoD"], 47.506259382507814), out["LoD"]
    assert _within_noise(out["score"], 14.883529238364059), out["score"]
    assert out["sigma_allow_Pa"] == weights.SIGMA_ALLOW_PA
    assert out["material"] == materials.REFERENCE_KEY


def test_the_weight_correlation_multiplies_exactly():
    w = weights.wing_weight_raymer(11.0, 16.0, 0.45, 0.0, 0.12, 5.7, 9500.0)
    assert weights.wing_weight_raymer(
        11.0, 16.0, 0.45, 0.0, 0.12, 5.7, 9500.0, k_w=1.0) == w
    assert weights.wing_weight_raymer(
        11.0, 16.0, 0.45, 0.0, 0.12, 5.7, 9500.0, k_w=0.5) == 0.5 * w
    with pytest.raises(ValueError, match="weight factor"):
        weights.wing_weight_raymer(11.0, 16.0, 0.45, 0.0, 0.12, 5.7, 9500.0,
                                   k_w=0.0)


# ------------------------------------------- it reaches BOTH halves, not one

def test_a_material_moves_the_weight_and_the_allowable_together():
    """The defect this module exists for: one of the two is not enough."""
    al = sizing.sized_state(W_fixed_N=6000.0, b=12.0, S=14.0, taper=0.5,
                            tc=0.12, q_Pa=1500.0)
    cf = sizing.sized_state(W_fixed_N=6000.0, b=12.0, S=14.0, taper=0.5,
                            tc=0.12, q_Pa=1500.0, material="cfrp_prepreg")
    assert cf.W_wing_N < al.W_wing_N            # the half that decides span
    assert cf.sigma_allow_Pa > al.sigma_allow_Pa   # ...and the constraint
    # not exactly k_w, because the weight loop closes: a lighter wing carries
    # a lighter gross weight and so weighs less again
    assert cf.W_wing_N / al.W_wing_N < materials.get("cfrp_prepreg").k_w
    assert cf.report()["material"] == "cfrp_prepreg"


def test_the_state_says_what_it_was_built_of():
    """A weight nobody can trace back to a material is not reproducible."""
    st = sizing.sized_state(W_fixed_N=6000.0, b=12.0, S=14.0, taper=0.5,
                            tc=0.12, q_Pa=1500.0, material="spruce")
    rep = st.report()
    assert rep["material"] == "spruce"
    assert rep["material_k_w"] == materials.get("spruce").k_w
    assert rep["material_rho_kgm3"] == 450.0


def test_the_reported_allowable_is_the_one_the_margin_used():
    """A material's DECLARED allowable must not overwrite the USED one.

    ``sized_state`` still accepts an explicit ``sigma_allow`` — the
    calibration studies move the stress gate without touching the weight —
    and the material carries an allowable of its own. Reported under one key
    they collided, and the breakdown would have shown a number the margin was
    not computed from.
    """
    st = sizing.sized_state(W_fixed_N=6000.0, b=12.0, S=14.0, taper=0.5,
                            tc=0.12, q_Pa=1500.0, material="cfrp_prepreg",
                            sigma_allow=400e6)
    rep = st.report()
    assert rep["sigma_allow_Pa"] == 400e6           # what g was measured on
    assert rep["material_sigma_allow_Pa"] == 550e6  # what the material says
    assert st.g_sigma == pytest.approx(
        float(np.log(400e6 / st.sigma_root_Pa)), abs=1e-12)


def test_a_heavier_weaker_material_costs_weight():
    """The map is a ratio, not a discount — it may make a wing worse."""
    assert materials.get("steel_4130").k_w > 1.0
    assert materials.get("eps_foam").k_w > 1.0        # a core, not a structure
    assert materials.get("cfrp_prepreg").k_w < 1.0
    assert (materials.get("cfrp_wet").k_w
            > materials.get("cfrp_prepreg").k_w)      # lower allowable


def test_two_materials_in_one_wing():
    """Foam core + carbon spar is TWO densities, which is the point of phi."""
    solid = materials.custom(1600.0, 500e6)
    cored = materials.custom(1600.0, 500e6, 25.0)
    assert cored.k_w < solid.k_w
    assert cored.k_w == pytest.approx(materials.get("foam_cf_spar").k_w,
                                      abs=1e-12)


# ------------------------------------------------------------- the api gate

def test_the_flag_is_refused_where_nothing_is_weighed():
    """A fixed-size family computes no weight and no stress at all, so a
    material there would be accepted and silently ignored — the exact shape
    ``check_flags`` exists to refuse."""
    out = _centre(UNSIZED)
    assert "W_wing_N" not in out and "sigma_root_Pa" not in out
    with pytest.raises(KeyError, match="does not honour"):
        api.check_flags(UNSIZED, {api.MATERIAL_KEY: "cfrp_prepreg"})
    api.check_flags(SIZED, {api.MATERIAL_KEY: "cfrp_prepreg"})


def test_half_a_material_is_refused():
    """A density without an allowable would be completed with a guess."""
    spec = api.PROBLEM_SPECS[SIZED]
    with pytest.raises(ValueError, match="BOTH"):
        spec.build({}, {api.MATERIAL_RHO_KEY: 1600.0}, None)
    with pytest.raises(ValueError, match="BOTH"):
        spec.build({}, {api.MATERIAL_SIGMA_KEY: 500.0}, None)


def test_the_flag_changes_the_answer_through_the_api():
    al = _centre(SIZED)
    cf = _centre(SIZED, {api.MATERIAL_KEY: "cfrp_prepreg"})
    assert cf["W_wing_N"] < al["W_wing_N"]
    assert cf["score"] > al["score"]
    assert cf["sigma_allow_Pa"] == 550e6
    custom = _centre(SIZED, {api.MATERIAL_RHO_KEY: 1600.0,
                             api.MATERIAL_SIGMA_KEY: 500.0,
                             api.MATERIAL_SKIN_RHO_KEY: 25.0})
    assert custom["material"] == "custom"
    assert custom["W_wing_N"] < cf["W_wing_N"]       # the core is lighter


def test_an_unknown_material_names_what_exists():
    with pytest.raises(KeyError, match="unknown material"):
        materials.get("unobtainium")


# ------------------------------------------------- the composite sees it too

def test_the_mass_criterion_is_priced_in_the_chosen_material():
    """``wing_score``'s mass criterion is Raymer as well.

    Left material-blind it would price a carbon wing as an aluminium one —
    and "mass" is exactly the criterion a user who has just chosen carbon is
    most likely to weight.
    """
    from aerobo import wing_score

    spec = api.PROBLEM_SPECS[SIZED]
    raw = _centre(SIZED)
    al_prob = spec.build({}, {}, None).problem
    cf_prob = spec.build({}, {api.MATERIAL_KEY: "cfrp_prepreg"},
                         None).problem
    al = wing_score.design_metrics(raw, al_prob).get("mass")
    cf = wing_score.design_metrics(raw, cf_prob).get("mass")
    assert al is not None and cf is not None
    assert cf < al
    assert cf / al == pytest.approx(materials.get("cfrp_prepreg").k_w,
                                    rel=1e-9)


# ------------------------------------------------------------ the V3 control

@pytest.fixture(scope="module")
def shell():
    from gui.v3.app import assemble

    return assemble("air")


def _type_view_texts(ctx) -> list:
    """Texts of the WING TYPE view only.

    Scoped to one view on purpose: the shipped V3 suite shares a client, and
    scanning every element on the page makes an assertion that depends on
    what some other test left behind.
    """
    view = ctx.views[("wing", "type")]
    return [getattr(e, "text", "") or "" for e in view.descendants()]


def test_the_control_appears_only_where_a_wing_is_weighed(shell):
    """Fixed size, no weight, no material question.

    Rendered rather than grepped: the card is built inside ``_size_controls``
    behind a flag-declaration gate, and a gate that reads correctly in the
    source is still a gate nobody has watched fire.
    """
    from gui.v3 import session

    ctx = shell
    session.set_planform(ctx.S, "fixed")
    ctx.render("wing")
    assert "built of" not in _type_view_texts(ctx)

    session.set_planform(ctx.S, "free")
    ctx.render("wing")
    texts = _type_view_texts(ctx)
    assert "built of" in texts
    assert "weight factor" in texts and "allowable" in texts


def test_choosing_a_material_reaches_the_config(shell):
    """The menu writes a flag, and the flag is what the run is built with."""
    from gui.v3 import config, session

    ctx = shell
    session.set_planform(ctx.S, "free")
    ctx.S["wing"]["flags"][api.MATERIAL_KEY] = "cfrp_prepreg"
    flags = config.flags(ctx.S)
    assert flags[api.MATERIAL_KEY] == "cfrp_prepreg"
    api.check_flags(ctx.S["wing"]["problem"], flags)
    ctx.render("wing")
    assert "0.870" in _type_view_texts(ctx)
    ctx.S["wing"]["flags"].pop(api.MATERIAL_KEY, None)


def test_the_reference_material_writes_no_flag(shell):
    """A run with no material flag must stay the published calibration."""
    from gui.v3 import config, session

    ctx = shell
    session.set_planform(ctx.S, "free")
    for key in api.MATERIAL_FLAG_KEYS:
        ctx.S["wing"]["flags"].pop(key, None)
    flags = config.flags(ctx.S)
    assert not any(k in flags for k in api.MATERIAL_FLAG_KEYS)


def test_the_card_refuses_half_a_material_the_way_the_builder_does(shell):
    """Card and builder must answer the same question the same way.

    Clearing one of the two custom numbers used to leave the menu showing the
    reference material at k_w 1.000 while the OTHER number sat in the flags
    waiting to be refused at launch. Nothing on screen said so — the card
    disagreed with the run.
    """
    from gui.v3 import session

    ctx = shell
    session.set_planform(ctx.S, "free")
    for key in api.MATERIAL_FLAG_KEYS:
        ctx.S["wing"]["flags"].pop(key, None)
    ctx.S["wing"]["flags"][api.MATERIAL_SIGMA_KEY] = 500.0   # no density
    ctx.render("wing")
    texts = _type_view_texts(ctx)
    assert "density" in texts, "still the custom form, not the reference"
    assert any("half a material" in t for t in texts)
    # ...and the builder agrees, which is the whole point
    spec = api.PROBLEM_SPECS[ctx.S["wing"]["problem"]]
    with pytest.raises(ValueError, match="BOTH"):
        spec.build({}, {api.MATERIAL_SIGMA_KEY: 500.0}, None)
    for key in api.MATERIAL_FLAG_KEYS:
        ctx.S["wing"]["flags"].pop(key, None)
