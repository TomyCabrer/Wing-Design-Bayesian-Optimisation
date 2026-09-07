"""The tip device's chord may CONTINUE the wing's, instead of holding the tip.

Every published winglet run flew a RECTANGLE: the device's panels all carry
the wing's tip chord (``vlm.py``), whatever the wing's own chord distribution
was doing when it got there. On a tapered wing — never mind a cubic chord law
that is still falling at the tip — that is a step in the planform at the one
junction the whole blend machinery exists to smooth.

So ``winglet_chord_follows`` (api.WINGLET_CHORD_KEY) removes one clip: the
spanwise line is already parameterised by developed arc, so the device's
panels are sampled from the wing's own chord law at their own arc (eta > 1)
and wing and device become ONE chord distribution. The rules:

* OFF is the default and is bit-for-bit the rectangle — no published result
  moves, and no family that has no tip device is ever asked;
* it is a VALUE: the design vector, its bounds and the optimiser are
  untouched;
* it reaches every DESIGNED surface's device (a tail's, a tandem's rear
  wing's), because it is a statement about the design, not about one surface;
* a law that continues the chord to a knife edge is REFUSED — the in-contract
  failure a collapsed chord law already is — never quietly squared off.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aerobo import api, geometry           # noqa: E402
from aerobo.vlm import VLM, DEVICE_CHORD_FLOOR   # noqa: E402

_KW = dict(N=24, winglet_h_frac=0.14, winglet_cant_deg=75.0, n_winglet=8)

#: a winglet run's design vector: taper, twists, device height, cant
X_WINGLET = np.array([0.45, 2.0, -2.0, 0.12, 75.0])


def _tapered(taper: float = 0.4, coeffs=()) -> geometry.Wing:
    return geometry.Wing(b=10.0, S=10.0, taper=taper, chord_coeffs=coeffs)


def _device_arc(b: float = 10.0) -> np.ndarray:
    """Arc length of the device's panel STATIONS from the junction — the
    cosine distribution vlm.py lays along the device."""
    h = _KW["winglet_h_frac"] * b / 2.0
    j = np.arange(1, _KW["n_winglet"] + 1)
    return (h / 2.0) * (1.0 - np.cos(np.pi * (j - 0.5) / _KW["n_winglet"]))


#: a law that is legal on the WING (it clears the collapse floor over
#: eta in [0, 1]) and runs the chord past zero on the device
_KNIFE_EDGE = (-0.13, -0.42, -0.31)


# ------------------------------------------------------------ the panelisation
def test_off_is_the_rectangle_it_always_was():
    wing = _tapered()
    model = VLM(wing, **_KW)
    c_dev = model.c[model.is_winglet]
    c_tip = float(wing.chord(np.array([wing.b / 2.0]))[0])
    assert c_dev.size >= 8
    assert np.allclose(c_dev, c_tip, rtol=0, atol=1e-12)


def test_on_the_device_is_sampled_from_the_wing_s_own_law():
    """Not a rescaled copy of it: the SAME function, at the device's own arc."""
    wing = _tapered()
    model = VLM(wing, **_KW, winglet_chord_follows=True)
    dev = model.is_winglet
    # the device's own panel stations, in developed arc from the centreline
    # (cosine along the device's arc, as vlm.py distributes them), on both
    # sides of a symmetric wing
    arc = wing.b / 2.0 + _device_arc()
    assert np.allclose(np.sort(model.c[dev]),
                       np.sort(np.concatenate([wing.chord(arc)] * 2)),
                       rtol=1e-12, atol=1e-12)


def test_on_a_tapered_wing_the_device_keeps_tapering():
    wing = _tapered(taper=0.4)
    c_tip = float(wing.chord(np.array([wing.b / 2.0]))[0])
    dev = VLM(wing, **_KW, winglet_chord_follows=True)
    c = dev.c[dev.is_winglet]
    assert c.max() < c_tip          # every station is OUTBOARD of the tip
    assert c.min() > 0.0
    # and the junction is a junction, not a step: the innermost device panel
    # sits within half a panel of the wing's tip chord
    assert c.max() == pytest.approx(c_tip, rel=0.05)


def test_a_rectangular_wing_is_unmoved_by_the_switch():
    """Nothing to continue: the tip chord IS the chord everywhere."""
    wing = geometry.Wing(b=10.0, S=10.0, taper=1.0)
    off, on = VLM(wing, **_KW), VLM(wing, **_KW, winglet_chord_follows=True)
    assert np.allclose(off.c, on.c, rtol=0, atol=1e-12)


def test_a_chord_law_reaches_the_device_too():
    wing = _tapered(taper=0.7, coeffs=(0.2, -0.4, 0.1))
    on = VLM(wing, **_KW, winglet_chord_follows=True)
    off = VLM(wing, **_KW)
    dev = on.is_winglet
    assert not np.allclose(on.c[dev], off.c[dev])
    # the law, not the tip tangent: a cubic sampled past 1 is not a straight
    # extension of itself, so the device's own chords are not collinear
    c = np.sort(on.c[dev])[::-1]
    assert not np.allclose(np.diff(c, 2), 0.0, atol=1e-9)


def test_the_scale_still_scales_what_the_law_drew():
    """``winglet_chord_scale`` (a car endplate's chord ratio) composes with the
    continuation rather than being replaced by it."""
    wing = _tapered()
    plain = VLM(wing, **_KW, winglet_chord_follows=True)
    scaled = VLM(wing, **_KW, winglet_chord_follows=True,
                 winglet_chord_scale=0.5)
    assert np.allclose(scaled.c[scaled.is_winglet],
                       0.5 * plain.c[plain.is_winglet])


def test_a_knife_edge_device_is_refused_not_squared_off():
    """A law still falling steeply at the tip runs the device to nothing.
    That is a planform nobody asked for, so it is refused — the same
    in-contract failure a collapsed chord law is."""
    wing = geometry.Wing(b=10.0, S=10.0, taper=0.2,
                         chord_coeffs=_KNIFE_EDGE)
    # legal as a WING: the law clears the collapse floor over eta in [0, 1]
    assert float(wing.chord(np.array([wing.b / 2.0]))[0]) > 0.0
    VLM(wing, **_KW)                       # ...and flies, rectangular device
    with pytest.raises(ValueError, match="knife edge"):
        VLM(wing, **_KW, winglet_chord_follows=True)


def test_the_floor_is_the_only_thing_that_refuses():
    """Just inside it builds; the message names the switch that caused it."""
    wing = _tapered(taper=0.3)
    model = VLM(wing, **_KW, winglet_chord_follows=True)
    c_tip = float(wing.chord(np.array([wing.b / 2.0]))[0])
    assert model.c[model.is_winglet].min() > DEVICE_CHORD_FLOOR * c_tip
    with pytest.raises(ValueError, match="winglet_chord_follows"):
        VLM(geometry.Wing(b=10.0, S=10.0, taper=0.2,
                          chord_coeffs=_KNIFE_EDGE),
            **_KW, winglet_chord_follows=True)


def test_a_second_surface_and_a_designed_tail_inherit_it():
    """One statement about the design, so the tail's device and the tandem's
    rear device follow their OWN wings' laws — not the front wing's."""
    from aerobo.vlm import SecondWing, TailSurface

    wing = _tapered(taper=0.5)
    rear = _tapered(taper=0.3)
    pair = VLM(wing, **_KW, winglet_chord_follows=True,
               second=SecondWing(wing=rear, x=2.0, z=0.0, N=16,
                                 winglet_h_frac=0.12, winglet_cant_deg=80.0))
    dev_rear = pair.is_winglet & pair.is_second
    assert dev_rear.any()
    c_tip_rear = float(rear.chord(np.array([rear.b / 2.0]))[0])
    assert pair.c[dev_rear].max() < c_tip_rear

    tailed = VLM(wing, **_KW, winglet_chord_follows=True,
                 tail=TailSurface(S=2.0, x=4.0, z=0.5, wing=rear, N=16,
                                  winglet_h_frac=0.12, winglet_cant_deg=80.0))
    dev_tail = tailed.is_winglet & tailed.is_tail
    assert dev_tail.any()
    assert tailed.c[dev_tail].max() < c_tip_rear


# --------------------------------------------------------------- the registry
def test_every_family_with_a_winglet_declares_it_and_nothing_else_does():
    declared = {n for n, sp in api.PROBLEM_SPECS.items()
                if api.WINGLET_CHORD_KEY in sp.flags}
    # read off the DECLARED design vector, never off a family list
    expect = {n for n, sp in api.PROBLEM_SPECS.items()
              if any(str(lbl).startswith("winglet_h")
                     for lbl in sp.param_labels)}
    assert declared == expect and declared
    # the deliberate exclusions: no device at all, and the car's ENDPLATE,
    # whose chord is already a design variable of its own
    assert not ({"trim wing", "car rear wing", "car rear wing + endplate",
                 "airfoil (section)"} & declared)


def test_the_flag_reaches_the_planform_of_every_family_that_declares_it():
    def carries(obj, depth=0):
        if depth > 3 or obj is None:
            return False
        if getattr(obj, "winglet_chord_follows", False):
            return True
        return any(carries(getattr(obj, a, None), depth + 1)
                   for a in ("foil", "wing_tail", "_wing_prob", "problem"))

    for name in ("winglet", "winglet_capped", "winglet + free chord law",
                 "hydrofoil + winglet", "hydrofoil + elevator + winglet",
                 "hydrofoil + elevator [designed elevator + tip device]",
                 "tail [designed tail + tip device]",
                 "tandem (nonplanar) + winglets"):
        spec = api.PROBLEM_SPECS[name]
        assert api.WINGLET_CHORD_KEY in spec.flags, name
        built = spec.build({}, {api.WINGLET_CHORD_KEY: True}, None)
        assert carries(built.problem), name


def test_off_is_bit_for_bit_and_on_moves_the_wing_it_flies():
    spec = api.PROBLEM_SPECS["winglet"]
    off = spec.build({}, {}, None)
    absent = spec.build({}, {api.WINGLET_CHORD_KEY: False}, None)
    on = spec.build({}, {api.WINGLET_CHORD_KEY: True}, None)

    # a VALUE: same vector, same box, same optimiser
    assert on.dim == off.dim and on.param_labels == off.param_labels
    assert np.array_equal(on.bounds, off.bounds)

    r_off = off.evaluate(X_WINGLET)
    r_absent = absent.evaluate(X_WINGLET)
    r_on = on.evaluate(X_WINGLET)
    assert r_absent["LoD"] == pytest.approx(r_off["LoD"], rel=0, abs=0)
    assert r_on["feasible"] and r_off["feasible"]
    assert r_on["LoD"] != r_off["LoD"]

    # and the run SAYS which planform it flew
    assert r_off["winglet"]["chord_follows"] is False
    assert r_on["winglet"]["chord_follows"] is True
    c_tip = 2.0 * 10.0 / (10.0 * (1.0 + X_WINGLET[0]))
    assert r_off["winglet"]["chord_min_m"] == pytest.approx(
        c_tip * X_WINGLET[0], rel=1e-9)
    assert r_on["winglet"]["chord_min_m"] < r_off["winglet"]["chord_min_m"]


def test_a_refused_device_is_a_penalty_not_an_exception():
    built = api.PROBLEM_SPECS["winglet + free chord law"].build(
        {}, {api.WINGLET_CHORD_KEY: True}, None)
    x = np.array([0.2, 0.0, 0.0, 0.14, 75.0, *_KNIFE_EDGE])
    out = built.evaluate(x)
    assert out["feasible"] is False
    assert "tip device" in out["reason"]
    assert built.callable(x) == api.PENALTY


# ------------------------------------------------------------------- the shell
def test_stage_three_offers_it_beside_the_device_it_describes(capsys):
    from gui.v3 import config
    from gui.v3.app import assemble

    ctx = assemble("air")
    S = ctx.S
    ctx.act("set_winglet", "canted")
    ctx.render("wing", "type")
    texts = [getattr(e, "text", "") or ""
             for e in ctx.views[("wing", "type")].descendants()]
    assert "tip device" in texts
    assert any("follows the wing's chord distribution" in t for t in texts)

    ctx.act("set_tip_chord", True)
    assert config.cfg_dict(S)["flags"][api.WINGLET_CHORD_KEY] is True
    ctx.act("set_tip_chord", False)
    assert api.WINGLET_CHORD_KEY not in config.cfg_dict(S)["flags"]

    err = capsys.readouterr().err
    assert "Traceback" not in err, err


def test_a_planform_change_does_not_deselect_it():
    """It is a VALUE, and the planform menu is not a question about it.

    Every planform entry selects a DIFFERENT registered problem ("tandem
    (nonplanar) + winglets + free chord law" -> "tandem (nonplanar) +
    winglets"), and the shell used to answer a problem change by wiping
    ``W["flags"]`` outright — so the switch the user had just turned on
    turned itself back off, with its own control still on screen offering it.
    The new family declares the same flag; the answer is still the answer.
    """
    from gui.v3 import config, session
    from gui.v3.app import assemble

    ctx = assemble("air")
    S = ctx.S
    S["wing"]["choices"]["system"] = "tandem"
    ctx.act("set_winglet", "canted")
    ctx.act("set_tip_chord", True)
    before = S["wing"]["problem"]
    assert api.WINGLET_CHORD_KEY in api.PROBLEM_SPECS[before].flags

    for law in ("fixed", "free", "fixed"):
        ctx.act("set_choice", "chord", law)
        name = S["wing"]["problem"]
        assert api.WINGLET_CHORD_KEY in api.PROBLEM_SPECS[name].flags, name
        assert S["wing"]["flags"].get(api.WINGLET_CHORD_KEY) is True, name
        assert config.cfg_dict(S)["flags"][api.WINGLET_CHORD_KEY] is True

    # ...and the control still SHOWS it on, which is what the user reported
    ctx.render("wing", "type")
    switches = [e for e in ctx.views[("wing", "type")].descendants()
                if "follows the wing's chord distribution"
                in (getattr(e, "text", "") or "")]
    assert switches, "the switch is not on screen at all"
    assert all(getattr(e, "value", None) is True for e in switches), \
        "the switch is drawn OFF after a planform change"

    # the rule is "the new family declares it", not "keep everything": a
    # MEDIUM change is a different craft and still clears the lot
    S["wing"]["choices"]["medium"] = "water"
    session.apply_choices(S)
    assert S["wing"]["flags"] == {}


def test_a_design_with_no_device_is_never_asked_and_never_sends_it():
    from gui.v3 import config
    from gui.v3.app import assemble

    ctx = assemble("air")
    ctx.act("set_tip_chord", True)          # asked while a device was offered
    ctx.act("set_winglet", "none")          # ...and then the device went away
    ctx.render("wing", "type")
    texts = [getattr(e, "text", "") or ""
             for e in ctx.views[("wing", "type")].descendants()]
    assert not any("follows the wing's chord" in t for t in texts)
    assert api.WINGLET_CHORD_KEY not in config.cfg_dict(ctx.S)["flags"]
