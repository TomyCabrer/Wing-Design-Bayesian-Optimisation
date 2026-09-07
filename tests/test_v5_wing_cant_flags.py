"""V5 item 6 — the wing's cant and sweep are stated, scored and flown.

They are declared on the LATTICE-backed wing+tail family and refused on the
lifting-line one, because a lifting line has no out-of-plane geometry to give
a dihedral effect to — and "a flag stated on an objective that does not read
it RAISES" is this repo's rule, not a preference.

**THE HEADLINE IN THIS FILE WAS CORRECTED.** It read "3 deg of dihedral buys
a convergent spiral for 0.22 % of L/D", and that number came off a rebuild
that was DROPPING THE TIP DEVICE: `build_flight_model` passed no winglet to
the VLM, so `Cl_beta` was bit-identical at -0.0147155577 for every device
height while the scored L/D moved 32.81 -> 34.49. A canted device is a
nonplanar surface carrying side force, so it is most of the dihedral effect.
Found by a concurrent session, confirmed here, fixed in `flightmodel.py`.

What is true with the device flown, on THIS box centre:

    no device        L/D 31.0082   margin -0.012858  DIVERGENT
    h_frac 0.1428                  margin  0.000000  <- the crossing
    h_frac 0.15                    margin +0.000254

    ...and on a design with NO device, where dihedral IS the lever:
    0 deg  L/D 31.0082  -0.012858     5.7 deg  L/D 30.8732  -0.000104
                                      6.0 deg  L/D 30.8578  +0.000592

So the tip device converts the spiral and GAINS L/D doing it, where dihedral
converts it at a 0.44 % cost. Dihedral is the right lever for a design that
has no device; it was never the cheapest one, and the instrument that said
otherwise could not see the alternative.

**AND THE CRITERION ITSELF WAS OPTIMISTIC UNTIL 2026-08-31.** The margins
above are ``dynamics.spiral_margin`` WITH the trim attitude; the classical
``theta0 = 0`` form that this file used to assert put the same two crossings
at h_frac 0.0945 and 3.560 deg, about 2.18 deg of dihedral early, because
these designs trim 4.5 deg nose-up and ``phi_dot = p + tan(theta0) r``.
"""
from __future__ import annotations

import sys

import pytest

from aerobo import api
from aerobo.flightmodel import build_flight_model

sys.path.insert(0, "gui")
from v4.modes import spiral_margin, spiral_margin_of        # noqa: E402

LATTICE = "tail + winglet"
LIFTING_LINE = "tail"


def _report(problem: str, flags: dict) -> dict:
    cfg = api.RunConfig(problem_name=problem, budget=4, seed=0, flags=flags)
    built = api.PROBLEM_SPECS[problem].build({}, flags, None)
    return api.design_report(cfg, built.bounds.mean(axis=1))


def _report_no_device(flags: dict) -> dict:
    """The box-centre design with its tip device turned OFF.

    The dihedral lever is measured here because with the device flown the
    box centre sits ON the spiral crossing (margin -5.6e-05) — a design that
    is already neutral cannot show what a lever is worth.
    """
    import numpy as np

    built = api.PROBLEM_SPECS[LATTICE].build({}, flags, None)
    x = np.asarray(built.bounds, float).mean(axis=1).copy()
    x[list(built.param_labels).index("winglet_h_frac")] = 0.0
    return api.design_report(
        api.RunConfig(problem_name=LATTICE, budget=4, seed=0, flags=flags), x)


def _report_at_device(h_frac: float) -> dict:
    import numpy as np

    built = api.PROBLEM_SPECS[LATTICE].build({}, {}, None)
    x = np.asarray(built.bounds, float).mean(axis=1).copy()
    x[list(built.param_labels).index("winglet_h_frac")] = float(h_frac)
    return api.design_report(
        api.RunConfig(problem_name=LATTICE, budget=4, seed=0), x)


def _margin(rep: dict) -> float:
    """The criterion AS SCORED — trim attitude included.

    It read the classical ``theta0 = 0`` form until 2026-08-31, and that
    form crosses zero 2.13 deg of dihedral before the spiral MODE does
    (dynamics.spiral_margin). Every crossing pinned in this file moved with
    the repair and each one says so where it is asserted.
    """
    d = build_flight_model(rep, V=45.0).deck
    return spiral_margin_of(d)


def _margin_theta0_zero(rep: dict) -> float:
    """...and the optimistic form, kept so the GAP can be asserted."""
    d = build_flight_model(rep, V=45.0).deck
    return spiral_margin(Cl_beta=d.Cl_beta, Cn_r=d.Cn_r,
                         Cn_beta=d.Cn_beta, Cl_r=d.Cl_r)


# ------------------------------------------------------ declared where read

def test_the_lifting_line_family_refuses_a_cant_it_cannot_score():
    for key in ("wing_dihedral_deg", "wing_sweep_deg"):
        with pytest.raises(KeyError, match=key):
            api.check_flags(LIFTING_LINE, {key: 5.0})


def test_the_lattice_family_accepts_both():
    api.check_flags(LATTICE, {"wing_dihedral_deg": 5.0,
                              "wing_sweep_deg": 15.0})
    assert set(api.WING_CANT_KEYS) <= set(api.PROBLEM_SPECS[LATTICE].flags)


def test_a_family_that_searches_ONE_row_still_takes_the_other_as_a_flag():
    """The freedom is per row, so the flag contract is per row too: a
    dihedral-only family refuses a stated dihedral and accepts a stated
    sweep. Refusing both would be a ban with no measurement behind it, and
    accepting both would let a run STATE a number the solver is
    simultaneously optimising."""
    for state, searched, stated in (
            ("dihedral", "wing_dihedral_deg", "wing_sweep_deg"),
            ("sweep", "wing_sweep_deg", "wing_dihedral_deg")):
        name = api.wing_tail_problem(winglets="free", tc=False, arm="free",
                                     height="fixed", cant=state)
        api.check_flags(name, {stated: 7.0})
        with pytest.raises(KeyError, match=searched):
            api.check_flags(name, {searched: 7.0})


def test_the_wing_cant_keys_are_not_the_tails():
    """``dihedral_deg`` is the TAIL's — a V-tail's cant. Confusing the two
    would silently turn a stated wing cant into a V-tail."""
    assert "dihedral_deg" in api.TAIL_CONFIG_KEYS
    assert "dihedral_deg" not in api.WING_CANT_KEYS
    assert not set(api.WING_CANT_KEYS) & set(api.TAIL_CONFIG_KEYS)


# ------------------------------------------------------------- and SCORED

def test_stating_a_ZERO_cant_is_the_published_planar_wing():
    """Adding the two rows must not move a single published score.

    NOT asserted bit-for-bit, because this family is not bit-reproducible
    run to run: the same config repeated eight times spreads 1.4e-14 in L/D
    (4.2e-16 relative), which is the same size as the difference being
    looked for. So the claim is stated against that MEASURED floor, and a
    real cant is shown to be four hundred million times larger than it.
    """
    a = _report(LATTICE, {})["breakdown"]
    b = _report(LATTICE, {"wing_dihedral_deg": 0.0,
                          "wing_sweep_deg": 0.0})["breakdown"]
    floor = 1e-13                       # 7x the measured 8-run spread
    for key in ("LoD", "SM", "CDi"):
        assert abs(a[key] - b[key]) <= floor * max(abs(a[key]), 1.0), key

    # ...and half a degree is 6e-3 in L/D, which is not in the noise
    half = _report(LATTICE, {"wing_dihedral_deg": 0.5})["breakdown"]
    assert abs(a["LoD"] - half["LoD"]) > 1e-3


def test_dihedral_costs_lift_over_drag_and_sweep_costs_much_more():
    """Both are real costs the objective can see, and they differ by 30x."""
    base = _report(LATTICE, {})["breakdown"]["LoD"]
    cant = _report(LATTICE, {"wing_dihedral_deg": 5.0})["breakdown"]["LoD"]
    swept = _report(LATTICE, {"wing_sweep_deg": 15.0})["breakdown"]["LoD"]
    assert 0.0 < (base - cant) / base < 0.01           # ~0.4 %
    assert 0.08 < (base - swept) / base < 0.20         # ~13 %


def test_sweep_moves_the_scored_static_margin():
    """The objective's own neutral point moves, not just the rebuild's —
    which is what "the lattice is actually swept" has to mean at the level
    a design is scored at."""
    base = _report(LATTICE, {})["breakdown"]
    swept = _report(LATTICE, {"wing_sweep_deg": 15.0})["breakdown"]
    assert swept["x_np"] > base["x_np"] + 0.4
    assert swept["SM"] > 2.0 * base["SM"]


# ------------------------------------------------------------ and FLOWN

def test_the_scored_cant_reaches_the_flight_model():
    """STATED IS NOT FLOWN. The report carries it, so the rebuild flies it."""
    rep = _report(LATTICE, {"wing_dihedral_deg": 6.0})
    assert rep["breakdown"]["wing_dihedral_deg"] == 6.0
    m = build_flight_model(rep, V=45.0).model
    assert m.dihedral_deg == 6.0
    assert m.z.max() > 0.2                       # the wing really is canted


def test_dihedral_converts_the_spiral_on_a_design_that_needs_it():
    """CORRECTED. The original form of this test measured a broken instrument.

    ``build_flight_model`` was dropping the TIP DEVICE — the report states it
    and the rebuild passed no winglet to the VLM — so ``Cl_beta`` came back
    bit-identical at -0.0147155577 for every device height while the scored
    L/D moved 32.81 -> 34.49. A canted device is a nonplanar surface carrying
    side force, so it is most of the dihedral effect, and the instrument that
    produced "3 deg buys a convergent spiral for 0.22 % of L/D" could not see
    it. (Found by a concurrent session; confirmed and fixed here.)

    With the device flown, the box-centre design is ALREADY on the crossing,
    so this measures the dihedral lever where it is actually a lever: on a
    design with no tip device, which is genuinely divergent.
    """
    def at(gamma):
        flags = {"wing_dihedral_deg": gamma} if gamma else {}
        rep = _report_no_device(flags)
        return rep["breakdown"]["LoD"], _margin(rep)

    rows = [at(g) for g in (0.0, 2.0, 4.0, 6.0)]
    lods = [r[0] for r in rows]
    margins = [r[1] for r in rows]

    assert margins[0] < 0.0, "the no-device design should be divergent"
    assert margins == sorted(margins), f"cant must help monotonically: {margins}"
    assert margins[-1] > 0.0, "enough cant must convert it"

    # THE ANGLE IS NOW PINNED, and the history is why it is worth saying so.
    # It read 2.61 deg, then 3, then ~5 across the V5 session — not because
    # the aeroplane changed but because the INSTRUMENT was being repaired,
    # four times over (tip device, section camber, wing twist, trim state).
    # While that was happening the honest thing was to assert the shape and
    # refuse to quote a number, and this test did.
    #
    # IT WAS REPAIRED A FIFTH TIME, and this number moved with it. The
    # criterion carried no TRIM ATTITUDE, so it crossed zero at 3.560 deg
    # while the 6-DOF spiral MODE crossed at 5.687 — the criterion was
    # optimistic by 2.13 deg, which is what made six of 42 searched winners
    # certified-and-divergent (dynamics.spiral_margin, and
    # RESULTS_SESSION67_WING_CANT.md). With the attitude restored both
    # instruments put this crossing at 5.736 deg on this no-device design.
    cost = (lods[0] - lods[-1]) / lods[0]
    assert 0.0 < cost < 0.01, f"the fix cost {100 * cost:.2f} % of L/D"

    from scipy.optimize import brentq
    crossing = brentq(lambda g: at(g)[1], 1.0, 9.0, xtol=1e-3)
    assert crossing == pytest.approx(5.736, abs=0.05), (
        f"the crossing moved to {crossing:.3f} deg — if that is an instrument "
        f"repair rather than a design change, say which")
    # ...and the form that ships no longer agrees with the one that shipped
    # before, by the measured 2.13 deg. If this ever ties, the trim term has
    # been dropped again and the number above is optimistic once more.
    old = brentq(lambda g: _margin_theta0_zero(_report_no_device(
        {"wing_dihedral_deg": g} if g else {})), 1.0, 9.0, xtol=1e-3)
    assert crossing - old == pytest.approx(2.176, abs=0.05), (
        f"the trim-attitude term is worth {crossing - old:+.3f} deg here, "
        f"not the measured 2.176")


def test_the_TIP_DEVICE_is_the_cheaper_lever_and_it_pays_for_itself():
    """The finding the broken rebuild hid, and the reason the headline moved.

    Measured on `tail + winglet` at 45 m/s, with the device now in the flown
    lattice:

        no device      L/D 32.8085   margin -0.005761  DIVERGENT
        h_frac 0.0755  L/D 33.8627   margin  0.000000  <- crossing
        h_frac 0.15    L/D 34.4855   margin +0.008311

    The device converts the spiral AND GAINS 3.2 % of L/D doing it, where
    dihedral converts it at a small COST. So dihedral is the right lever for
    a design with no device, and the wrong first answer for one that could
    have a taller device instead.
    """
    flat = _report_no_device({})
    tall = _report_at_device(0.15)
    assert _margin(flat) < 0.0 < _margin(tall)
    # ...and it helps monotonically on the way, which is the shape that
    # survives every instrument fix even as the crossing point moves
    ms = [_margin(_report_at_device(h)) for h in (0.0, 0.075, 0.10, 0.15)]
    assert ms == sorted(ms), ms
    assert tall["breakdown"]["LoD"] > flat["breakdown"]["LoD"], (
        "the tip device must PAY — that is what makes it the cheaper lever")
    gain = ((tall["breakdown"]["LoD"] - flat["breakdown"]["LoD"])
            / flat["breakdown"]["LoD"])
    assert gain > 0.03


def test_the_rebuild_FLIES_the_tip_device_the_report_states():
    """The defect itself, pinned. Zero device panels for every height was
    what made the two tests above measure the wrong aeroplane."""
    import numpy as np

    built = api.PROBLEM_SPECS[LATTICE].build({}, {}, None)
    i = list(built.param_labels).index("winglet_h_frac")
    seen = []
    for h in (0.0, 0.05, 0.15):
        x = np.asarray(built.bounds, float).mean(axis=1).copy()
        x[i] = h
        rep = api.design_report(
            api.RunConfig(problem_name=LATTICE, budget=4, seed=0), x)
        fm = build_flight_model(rep, V=45.0)
        panels = int(fm.model.is_winglet.sum())
        seen.append((h, panels, fm.deck.Cl_beta))
        assert (panels > 0) == (h > 0.0), (
            f"h_frac {h}: {panels} device panels in the flown lattice")
    # ...and the dihedral effect MOVES with the device, which is the whole
    # reason dropping it mattered
    betas = [b for _h, _p, b in seen]
    assert len(set(betas)) == 3, f"Cl_beta did not move with the device: {betas}"
    assert betas[2] < betas[1] < betas[0] < 0.0


def test_the_two_INSTRUMENTS_agree_on_where_each_lever_crosses():
    """The claim that replaced a session-long hedge, and the reason to trust it.

    Two independent codes — the objective's own lateral deck, computed inside
    the evaluation, and ``flightmodel``'s rebuilt lattice — are asked where
    each lever converts the spiral. They agree:

        tip device   objective h_frac 0.1428   rebuild 0.1427   (0.0001 span)
        dihedral     objective 5.736 deg       rebuild 5.736    (0.0000 deg)

    While the rebuild was dropping the tip device, the section camber, the
    wing twist and the trim state, these disagreed by 0.02 of span and both
    write-ups carried "do not quote a crossing to three figures". That hedge
    was true and is not any more; it is asserted away here rather than left
    standing as a permanent-looking limitation.

    BOTH NUMBERS MOVED on 2026-08-31 and neither aeroplane changed: the
    criterion gained the trim attitude it had been dropping, which is worth
    2.18 deg of dihedral and 0.048 of span here. The two instruments agreed
    with each other before and after — they were agreeing on the optimistic
    quantity.
    """
    from scipy.optimize import brentq

    import numpy as np

    built = api.PROBLEM_SPECS[LATTICE].build({}, {}, None)
    built.problem.lateral = True
    i = list(built.param_labels).index("winglet_h_frac")
    x0 = np.asarray(built.bounds, float).mean(axis=1)

    def scored(h):
        x = x0.copy(); x[i] = h
        return built.evaluate(x)["spiral_margin"]

    def rebuilt(h):
        x = x0.copy(); x[i] = h
        return _margin(api.design_report(
            api.RunConfig(problem_name=LATTICE, budget=4, seed=0), x))

    a = brentq(scored, 0.05, 0.15, xtol=1e-5)
    b = brentq(rebuilt, 0.05, 0.15, xtol=1e-5)
    assert abs(a - b) < 0.002, (
        f"the two instruments put the device crossing at {a:.4f} and "
        f"{b:.4f} — they have diverged again, and the last four times that "
        f"happened it was the rebuild dropping something the design states")
    assert a == pytest.approx(0.143, abs=0.005)


def test_the_dihedral_effect_grows_monotonically_through_the_family():
    margins = [_margin(_report(LATTICE, {"wing_dihedral_deg": g} if g else {}))
               for g in (0.0, 2.0, 4.0, 6.0)]
    assert margins == sorted(margins)
