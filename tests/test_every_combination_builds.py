"""No combination the shell OFFERS may be one the registry refuses.

The menus are built by asking the registry what it has a solver for
(``option_available``), so an entry that appears is a promise: choosing it,
together with everything else already chosen, produces a problem that builds
and evaluates. This sweeps the product of every menu on stage 3 — medium x
lifting system x second surface x the WING's tip-device shape x the SECOND
SURFACE's tip-device shape x the planform/size mode x the chord law — and
holds that promise to it.

What it is NOT: a claim that every design in every box flies. A candidate can
fail IN CONTRACT (a pair whose two tip devices meet, an aspect ratio outside
the sizing band) and that is a physical answer with a named remedy, not a
refused combination. The distinction this file draws is exactly:

* BUILD failures are bugs — a menu offered something the registry has no
  solver for, or the flags the shell sends are ones the builder raises on;
* FEASIBILITY at one point is not a promise, so where the middle of the box
  is refused, the test asserts the refusal is a NAMED in-contract one and
  that the box still contains flyable designs.
"""

from __future__ import annotations

import numpy as np
import pytest

from aerobo import api

from gui import nice_app as v1


def _start(**kw) -> dict:
    ch = dict(v1.BUILDER_START)
    ch.update(kw)
    v1.normalise_choices(ch)
    return ch


def _flags(ch: dict, name: str) -> dict:
    """Exactly what the shells send for these choices, filtered to what the
    problem declares — the same rule ``gui/v3/config.flags`` applies."""
    spec = api.PROBLEM_SPECS[name]
    out: dict = {}
    out.update(v1.tail_flags(ch))
    out.update(v1.winglet_flags(ch))
    out.update(v1.car_flags(ch))
    out.update(v1.tandem_flags(ch, name))
    out.update(v1.planform_flags(ch, name))
    return {k: v for k, v in out.items() if k in spec.flags}


def _combinations():
    """Every configuration the stage-3 menus can be walked into."""
    for medium in ("air", "water", "track"):
        for system in ("single", "tandem"):
            for tail in (False, True):
                for chord in ("fixed", "free"):
                    ch = _start(medium=medium, system=system, tail=tail,
                                chord=chord)
                    # a pairing the shell would not offer (normalisation
                    # dropped it) is not a combination
                    if bool(ch.get("tail")) != tail \
                            or ch.get("system") != system:
                        continue
                    for wl in v1.winglet_shapes(ch):
                        for tt in v1.tail_tip_shapes(ch):
                            base = dict(ch)
                            v1.set_winglet_shape(base, wl)
                            if tt != "none":
                                v1.set_tail_tip_shape(base, tt)
                                base["tail_winglet_dir"] = "follow"
                            v1.normalise_choices(base)
                            for pf in v1.planform_options(base):
                                if pf == "aircraft":
                                    continue    # its own family; V3 drops it
                                # ...and, on the TRACK, whether the section is
                                # SLOTTED. It is a family of its own (four
                                # design rows), it is not reachable through
                                # the `system` switch — a car's rear wing has
                                # nothing behind it, so that switch normalises
                                # away — and a sweep that never set it would
                                # leave the two-element wing out of every
                                # build check below.
                                slots = ((False, True) if medium == "track"
                                         else (False,))
                                for slot in slots:
                                    c = dict(base, planform=pf,
                                             car_two_element=slot)
                                    v1.normalise_choices(c)
                                    yield (f"{medium}/{system}"
                                           f"{'+tail' if tail else ''}"
                                           f"/tip:{wl}/aft-tip:{tt}/{pf}"
                                           f"/chord:{chord}"
                                           f"{'/slot' if slot else ''}", c)


ALL = list(_combinations())
IDS = [name for name, _ in ALL]


def test_the_sweep_is_the_size_the_menus_imply():
    """A guard on the guard: a normalisation change that silently emptied
    the product would make every test below vacuous."""
    assert len(ALL) > 150, len(ALL)


@pytest.mark.parametrize("_name,ch", ALL, ids=IDS)
def test_every_offered_combination_builds_and_evaluates(_name, ch):
    name, _notes = v1.derive_problem(ch)
    spec = api.PROBLEM_SPECS[name]           # KeyError = an offered non-problem
    flags = _flags(ch, name)
    built = spec.build({} if spec.uses_mission else None, flags, None)
    assert built.dim == len(built.param_labels)
    x = np.array([0.5 * (lo + hi) for lo, hi in built.bounds], dtype=float)
    out = built.evaluate(x)                  # never raises, by contract
    if out["feasible"]:
        assert np.isfinite(out["score"])
        return
    # ...and where the MIDDLE of the box is refused, the refusal is a named
    # in-contract one (never a traceback rendered as a reason), and the box
    # still holds designs that fly
    reason = str(out.get("reason") or "")
    assert reason and "Traceback" not in reason
    lo, hi = built.bounds[:, 0], built.bounds[:, 1]
    rng = np.random.default_rng(0)
    flyable = sum(built.evaluate(lo + rng.random(lo.size) * (hi - lo)
                                 )["feasible"] for _ in range(60))
    assert flyable > 0, (name, reason)


# ------------------------------------------- the two the user asked about
def test_the_blended_tip_is_offered_wherever_a_tip_device_is():
    """"Blended" is a SHAPE of tip device, not a family of its own.

    It used to vanish the moment an empennage was added — the wing+tail, the
    tandem and the hydrofoil+elevator solvers had no blend at all — so the
    menu offered three shapes on a wing and two on the same wing with a tail.
    """
    seen = 0
    for name, ch in ALL:
        shapes = v1.winglet_shapes(ch)
        if list(shapes) == ["none"]:
            continue                 # the car: its tip device is an ENDPLATE
        seen += 1
        assert "blended" in shapes, name
    assert seen > 100, seen


def test_the_second_surfaces_tip_device_is_asked_for_a_shape_too():
    """Wherever there is a second surface with a designed planform, its tip
    device is the same four-answer question the wing's is."""
    seen = 0
    for name, ch in ALL:
        if not ch.get("tail") or ch.get("tail_design") == "fixed":
            continue
        shapes = v1.tail_tip_shapes(ch)
        seen += 1
        assert set(shapes) == {"none", "vertical", "canted", "blended"}, name
    assert seen > 20, seen


def test_every_configuration_can_be_given_a_size():
    """The planform menu always has an answer, and the answer is real.

    Under water it used to be a single disabled entry — the size card was not
    even drawn — which made "how big is the foil?" a question the shell could
    not be asked. Now every configuration either states its size, searches it,
    or already carries it in the design vector.
    """
    for name, ch in ALL:
        opts = v1.planform_options(ch)
        assert opts, name
        problem = v1.derive_problem(ch)[0]
        stateable = v1.planform_resizable(problem)
        searched = any(str(lbl) in ("b_m", "b_rear_m", "S_m2")
                       for lbl in api.PROBLEM_SPECS[problem].param_labels)
        assert stateable or searched or len(opts) > 1, (name, problem)


def test_a_stated_water_size_is_the_size_that_flies():
    """The hydrofoil families take b and S (api._RESIZABLE_PROBLEMS), and an
    untouched build is still the published 1.2 m / 0.144 m² foil."""
    for problem in ("hydrofoil", "hydrofoil + winglet",
                    "hydrofoil + elevator",
                    "hydrofoil + elevator [designed elevator + tip device]"):
        spec = api.PROBLEM_SPECS[problem]
        assert set(api.PLANFORM_KEYS) <= set(spec.flags), problem
        plain = spec.build(None, {}, None).problem
        assert (plain.b, plain.S) == (1.2, 0.144), problem
        sized = spec.build(None, {"b_m": 1.6, "S_m2": 0.20}, None)
        assert (sized.problem.b, sized.problem.S) == (1.6, 0.20)
        out = sized.evaluate(np.array([0.5 * (lo + hi)
                                       for lo, hi in sized.bounds]))
        assert out["feasible"], (problem, out.get("reason"))
        # a bigger foil at the same weight is a lower-loading foil: it makes
        # the lift at less incidence, so L/D rises. The point is that the
        # number MOVED — the size is flown, not merely stored.
        base = spec.build(None, {}, None)
        assert out["score"] != pytest.approx(
            base.evaluate(np.array([0.5 * (lo + hi)
                                    for lo, hi in base.bounds]))["score"])
