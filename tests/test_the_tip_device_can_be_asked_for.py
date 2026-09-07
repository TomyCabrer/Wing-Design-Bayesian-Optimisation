"""A freedom the registry has and the shell could not reach.

Reported as: "doesn't let tip opt for elevator when vertical fin used" — with
the goal "make sure that there is full liberty in the parameters for
optimisation".

Reproduced headlessly, it is one defect with two halves and a third of the
same shape beside it:

1. THE OPENING AEROPLANE'S TAIL WAS A FITTING. ``session.V3_START_CHOICES``
   opens a fresh AIR session on a whole aeroplane (``tail: True``), and it
   writes that choice straight into the state — so the surface never went
   through ``mission.set_tail``, which is the ONE place that applies
   ``nice_app.tail_design_start`` ("a stabiliser is a SURFACE, not a
   fitting"). Every fresh air session therefore held
   ``tail_design="fixed"``: the published AR-4 rectangle. The design box lost
   the surface's own rows (``taper_t``, ``AR_t``, ``washout_t_deg``, and its
   chord-law block) on a family that declares all of them.

2. AND ITS TIP-DEVICE MENU WAS DEAD. ``wing._surface_design_controls``
   disabled the shape select whenever that planform toggle was on the
   rectangle. The registry HAS the variant — ``tail [free height, designed
   tail + tip device]``, ``hydrofoil + elevator [designed elevator + tip
   device]`` — and ``_set_tail_tip`` writes the composite ``tail_design``
   that selects it, so the greying was an ORDERING the shell imposed, not a
   refusal any solver makes. Opening state + dead menu together: on the
   aeroplane (the configuration that carries the fin) the tip device on the
   second surface could not be asked for at all.

3. A T-TAIL FOLLOWED THE SESSION INTO THE WATER. ``tail_type`` survives a
   medium switch like every other choice, and the vertical-separation control
   read it raw — so an air session that picked a T-tail and moved to water
   had its ELEVATOR's depth toggle disabled and displayed as "fixed" ("a
   T-tail's height IS its fin span") while the family it was on was
   ``hydrofoil + elevator [free depth]``, searching ``z_t_m``. A control
   lying about the design vector, on a craft with no fin at all.

Every test asserts an OUTCOME: a row in the design vector, a widget the user
can actually click, a toggle whose value matches the family being flown.
"""

from __future__ import annotations

import pytest

from aerobo import api
from gui import nice_app as v1

#: the second surface's own planform rows — what "designed, not a rectangle"
#: means in the design vector
SURFACE_ROWS = ("taper_t", "AR_t", "washout_t_deg")

#: ...and its tip device's own two
TIP_ROWS = ("winglet_h_frac_t", "winglet_cant_t_deg")


def _labels(problem: str, **flags) -> tuple:
    return tuple(api.PROBLEM_SPECS[problem].param_labels)


def _widgets(ctx, label: str, view: str = "type"):
    """Every select/toggle in a rendered view sitting under ``label``.

    Scoped to one view and one row label on purpose: this shell builds every
    stage into one client, so a scan of "all elements" is order-dependent
    (see the note in tests/test_v3_act_render_fixpoint.py).
    """
    from nicegui import ui

    ctx.render("wing", view)
    out, last = [], None

    def walk(el):
        nonlocal last
        if isinstance(el, ui.label):
            last = el.text
        if isinstance(el, (ui.select, ui.toggle)) and last == label:
            out.append(el)
        for child in el.default_slot.children:
            walk(child)

    walk(ctx.views[("wing", view)])
    return out


@pytest.fixture()
def air():
    from gui.v3.app import assemble

    ctx = assemble("air")
    ctx.act("accept_mission")
    return ctx


# ------------------------------- 1. the opening aeroplane's tail is a surface

def test_a_fresh_air_session_opens_with_its_second_surface_designed(air):
    """The box carries the surface's own planform, not a rectangle."""
    from gui.v3 import config

    ch = air.S["wing"]["choices"]
    assert ch["tail"], "the opening aeroplane carries a second surface"
    assert ch["tail_design"] != "fixed", \
        "the opening aeroplane's tail opened as a published rectangle"
    rows = list(config.effective_bounds(air.S))
    for row in SURFACE_ROWS:
        assert row in rows, \
            f"{row!r} is not in the design box of {air.S['wing']['problem']!r}"


def test_the_opening_design_is_still_one_the_registry_honours(air):
    """Opened through the registry, not by writing a value into the state."""
    ch = air.S["wing"]["choices"]
    assert v1.option_available(ch, "tail_design", ch["tail_design"])
    # ...and it is the same answer the handler gives a surface switched on by
    # hand, which is the rule this opening state was skipping
    assert ch["tail_design"] == v1.tail_design_start(ch)


# ------------------------------------------ 2. the tip device can be asked for

def test_the_tip_device_menu_is_live_on_the_opening_aeroplane(air):
    """The control the report is about: clickable, with every shape in it."""
    tips = _widgets(air, "tip device")
    assert len(tips) == 2, \
        "expected the wing's tip-device menu and the second surface's"
    surface = tips[-1]                     # the second surface's, drawn last
    assert surface.enabled, \
        "the second surface's tip-device menu is greyed out on a family that " \
        "has the solver for it"
    assert set(v1.TAIL_TIP_SHAPE_LABELS) <= set(surface._props["options"] and
                                                v1.tail_tip_shapes(
                                                    air.S["wing"]["choices"]))


@pytest.mark.parametrize("shape", ["vertical", "canted", "blended"])
def test_asking_for_a_shape_reaches_the_solver_that_draws_it(air, shape):
    """One click, and the tip device's rows are in the design vector."""
    from gui.v3 import config

    air.act("set_tail_tip", shape)
    assert air.S["wing"]["choices"]["tail_design"] == "planform+tip"
    rows = list(config.effective_bounds(air.S))
    for row in TIP_ROWS:
        assert row in rows, \
            f"asking for a {shape} tip device left {row!r} out of the box"


def test_the_menu_is_live_even_on_the_published_rectangle(air):
    """The ordering the shell used to impose: pick the device, get both.

    A rectangle at AR 4 has no tip to hang a device on, which is TRUE of the
    solver and is not a reason to grey the control: the one handler behind it
    writes the ``tail_design`` that designs the planform WITH the device.
    """
    air.act("set_choice", "tail_design", "fixed")
    surface = _widgets(air, "tip device")[-1]
    assert surface.enabled, \
        "the shape menu is dead while the surface is a rectangle — the " \
        "user has to find an unrelated toggle before the device can be asked"
    air.act("set_tail_tip", "canted")
    assert air.S["wing"]["choices"]["tail_design"] == "planform+tip"
    assert "tip device" in air.S["wing"]["problem"]


def test_no_second_surface_menu_where_there_is_no_second_surface(air):
    """...and nothing is offered where the registry has nothing.

    The PAIR is the case: no tandem family designs a tip device on the rear
    wing (``tail_tip_shapes`` is ``("none",)`` for every tandem
    configuration), and a tandem has no second-surface card at all — so the
    one tip-device menu on screen is the WING's, which is real. What must
    not happen is a second menu offering shapes nothing can build.
    """
    air.act("set_choice", "system", "tandem")
    ch = air.S["wing"]["choices"]
    assert not ch["tail"], "a pair carries no tail card"
    assert list(v1.tail_tip_shapes(ch)) == ["none"], \
        "no tandem family designs a tip device on the rear wing"
    tips = _widgets(air, "tip device")
    assert len(tips) == 1 and tips[0].enabled, \
        "the pair should show the WING's tip-device menu and no other"


# ------------------------------------ 3. a layout is answered where it is asked

def test_every_registered_problem_is_reachable_from_the_menus():
    """4279 of 4279 — nothing is registered that cannot be clicked to.

    The whole-registry version of the per-family rule in
    tests/test_menus_offer_what_exists.py, which stops at the first missing
    name and so reported ONE problem where 384 were unreachable: the section
    branch of ``nice_app._derive_tail`` computed the wing's cant and then did
    not pass it to ``api.wing_tail_section_problem``, so every free-cant x
    CST-section combination — a family that builds, flies and panels the
    dihedral (15-D against the fixed-cant twin's 13-D) — could not be
    selected by any sequence of clicks, and the shell printed "the wing's
    dihedral and sweep are not design variables in this family" over it.

    Kept as a SWEEP rather than a list of names so a family added later is
    covered the day it is registered.
    """
    import itertools

    axes = {
        "medium": ("air", "water", "track"),
        "system": ("single", "tandem"),
        "tail": (False, True),
        "airfoil": tuple(v1.AIRFOIL_OPTION_LABELS),
        "winglets_key": tuple(v1.WINGLET_OPTIONS),
        "planform": tuple(v1.PLANFORM_CHOICE_LABELS),
        "chord": ("fixed", "free"),
        "flight": ("fixed", "free"),
        "tail_arm": ("free", "fixed"),
        "tail_height": ("fixed", "free"),
        "tail_design": tuple(api.TAIL_DESIGNS),
        "wing_cant": tuple(api.WING_CANTS),
        "fly_section": (False, True),
        "car_two_element": (False, True),
        "car_endplates": (False, True),
    }
    reached = set()
    for combo in itertools.product(*axes.values()):
        d = dict(zip(axes, combo))
        ch = dict(v1.BUILDER_DEFAULTS)
        ch["winglets"], ch["winglet_type"] = \
            v1.WINGLET_OPTIONS[d.pop("winglets_key")]
        ch.update(d)
        try:
            reached.add(v1.derive_problem(ch)[0])
        except Exception:                       # noqa: BLE001 — a sweep
            pass
    missing = sorted(set(api.PROBLEM_SPECS) - reached)
    assert not missing, (
        f"{len(missing)} registered problems no combination of choices "
        f"reaches, e.g. {missing[:3]}")


def test_a_t_tail_does_not_follow_the_session_into_the_water():
    """The elevator's depth is searchable, and the toggle says so."""
    from gui.v3.app import assemble

    ctx = assemble("air")
    ctx.act("accept_mission")
    ctx.act("set_empennage", "t_tail")
    ctx.act("set_medium", "water")
    assert ctx.S["wing"]["choices"]["tail_type"] == "t_tail", \
        "this test's premise: the choice survives the medium switch"
    free = ctx.S["wing"]["choices"].get("tail_height") == "free"
    assert free and "free depth" in ctx.S["wing"]["problem"], \
        "this test's premise: the water family being flown searches z_t_m"
    depth = _widgets(ctx, "depth")
    assert depth, "the elevator has no depth control at all"
    assert depth[-1].enabled, \
        "the elevator's depth is greyed out by an EMPENNAGE the water solver " \
        "does not have"
    assert depth[-1].value == "free", \
        "the toggle shows a stated depth while the run searches one"
